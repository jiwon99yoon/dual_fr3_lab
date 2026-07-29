# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Differential IK로 dual FR3 양팔의 end-effector를 목표 pose로 보내는 데모.

- 좌/우 팔 각각 DifferentialIKController를 두고, 각자 목표 pose 3개를 순회한다.
- ee body는 merge-joints 변환으로 fr3_hand가 link7에 병합됐으므로 left/right_fr3_link7을 사용.
- 목표 pose는 TCP(그리퍼 파지점) 기준이며, link7→TCP 오프셋(z+0.2104, yaw -45도)을
  역변환해서 IK에 link7 목표로 넘긴다. 마커도 TCP frame을 표시한다.
- 목표는 root(base, 팔 장착 상판) frame 기준 좌우 대칭.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_dual_fr3_diff_ik.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Differential IK demo for the dual FR3 robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_conjugate, subtract_frame_transforms

##
# Pre-defined configs
##
from dual_fr3_lab.robots import DUAL_FR3_HIGH_PD_CFG  # isort:skip

# link7 → TCP(그리퍼 파지점) 오프셋
# URDF 체인: link7 -(joint8: z+0.107)- link8 -(hand_joint: yaw -45도)- hand -(z+0.1034)- TCP
# merge-joints로 hand가 link7에 병합됐으므로 이 오프셋을 상수로 적용한다.
TCP_OFFSET_POS = (0.0, 0.0, 0.2104)
TCP_OFFSET_QUAT = (0.9238795, 0.0, 0.0, -0.3826834)  # Rz(-45도), (w, x, y, z)


@configclass
class DualFr3SceneCfg(InteractiveSceneCfg):
    """Scene with the dual FR3 robot on a ground plane."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation (init_state.pos z=0.405로 바퀴가 지면에 닿음)
    robot = DUAL_FR3_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class ArmIK:
    """한쪽 팔의 IK controller + 목표 + 마커 묶음."""

    def __init__(self, side: str, scene: InteractiveScene, robot, device):
        self.side = side
        self.robot = robot
        # IK controller
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=device)
        # entity: 팔 7개 joint + ee body(link7, hand가 병합된 링크)
        self.entity_cfg = SceneEntityCfg(
            "robot", joint_names=[f"{side}_fr3_joint[1-7]"], body_names=[f"{side}_fr3_link7"]
        )
        self.entity_cfg.resolve(scene)
        # fixed base이므로 jacobian의 body 인덱스는 1 감소
        if robot.is_fixed_base:
            self.ee_jacobi_idx = self.entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.entity_cfg.body_ids[0]
        # 목표 pose (root=base frame 기준, 좌우는 y 부호만 반전)
        y_sign = 1.0 if side == "left" else -1.0
        self.goal_names = ["down-front (아래보기)", "pitch-90 (앞보기)", "down-side (아래보기, 바깥쪽)"]
        self.ee_goals = torch.tensor(
            [
                [0.45, y_sign * 0.40, 0.35, 0.0, 1.0, 0.0, 0.0],  # 아래 보기
                [0.45, y_sign * 0.25, 0.45, 0.707, 0.0, 0.707, 0.0],  # 앞 보기 (y축 +90도 pitch)
                [0.35, y_sign * 0.50, 0.45, 0.0, 1.0, 0.0, 0.0],  # 아래 보기, 바깥쪽 위치
            ],
            device=device,
        )
        self.goal_idx = 0
        # commands는 TCP 기준 목표 pose
        self.commands = torch.zeros(scene.num_envs, self.controller.action_dim, device=device)
        self.commands[:] = self.ee_goals[self.goal_idx]
        # link7 → TCP 오프셋과 그 역변환(TCP → link7) 준비
        self.tcp_pos_l7 = torch.tensor(TCP_OFFSET_POS, device=device).repeat(scene.num_envs, 1)
        self.tcp_quat_l7 = torch.tensor(TCP_OFFSET_QUAT, device=device).repeat(scene.num_envs, 1)
        self.l7_quat_tcp = quat_conjugate(self.tcp_quat_l7)
        self.l7_pos_tcp = -quat_apply(self.l7_quat_tcp, self.tcp_pos_l7)
        # markers
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path=f"/Visuals/{side}_ee_current"))
        self.goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path=f"/Visuals/{side}_ee_goal"))

    def reset(self):
        """controller 리셋 + 다음 목표로 전환."""
        self.commands[:] = self.ee_goals[self.goal_idx]
        self.controller.reset()
        # TCP 목표를 link7 목표로 역변환해서 IK에 넘긴다 (IK는 link7 frame을 푼다)
        l7_goal_pos, l7_goal_quat = combine_frame_transforms(
            self.commands[:, 0:3], self.commands[:, 3:7], self.l7_pos_tcp, self.l7_quat_tcp
        )
        self.controller.set_command(torch.cat([l7_goal_pos, l7_goal_quat], dim=-1))
        goal = self.ee_goals[self.goal_idx].tolist()
        print(
            f"[INFO] {self.side:5s} arm goal #{self.goal_idx}: {self.goal_names[self.goal_idx]}"
            f" | pos={goal[0:3]} quat={goal[3:7]}"
        )
        self.goal_idx = (self.goal_idx + 1) % len(self.ee_goals)

    def compute(self):
        """현재 상태에서 IK를 풀어 목표 joint position을 반환."""
        robot = self.robot
        jacobian = robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, self.entity_cfg.body_ids[0]]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, self.entity_cfg.joint_ids]
        # ee pose를 root(base) frame으로 변환
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        return self.controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

    def visualize(self, scene: InteractiveScene):
        """현재 TCP frame과 목표 frame 마커 갱신."""
        robot = self.robot
        # 현재 TCP = link7 world pose에 오프셋 적용
        l7_pose_w = robot.data.body_state_w[:, self.entity_cfg.body_ids[0], 0:7]
        tcp_pos_w, tcp_quat_w = combine_frame_transforms(
            l7_pose_w[:, 0:3], l7_pose_w[:, 3:7], self.tcp_pos_l7, self.tcp_quat_l7
        )
        self.ee_marker.visualize(tcp_pos_w, tcp_quat_w)
        # 목표(TCP, root frame 기준)는 root 위치(z=0.405 포함)를 더해 world로
        root_pose_w = robot.data.root_pose_w
        self.goal_marker.visualize(root_pose_w[:, 0:3] + self.commands[:, 0:3], self.commands[:, 3:7])


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    robot = scene["robot"]

    # 좌/우 팔 IK 구성
    arms = [ArmIK("left", scene, robot, sim.device), ArmIK("right", scene, robot, sim.device)]

    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset (매 300 스텝 = 3초마다 목표 전환)
        if count % 300 == 0:
            count = 0
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            for arm in arms:
                arm.reset()
        else:
            # 각 팔의 IK 풀이 → 해당 팔 joint에만 target 인가
            for arm in arms:
                joint_pos_des = arm.compute()
                robot.set_joint_position_target(joint_pos_des, joint_ids=arm.entity_cfg.joint_ids)

        scene.write_data_to_sim()
        # perform step
        sim.step()
        count += 1
        scene.update(sim_dt)

        # update markers
        for arm in arms:
            arm.visualize(scene)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.8, 0.0, 1.8], [0.3, 0.0, 0.5])
    # Design scene
    scene_cfg = DualFr3SceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
