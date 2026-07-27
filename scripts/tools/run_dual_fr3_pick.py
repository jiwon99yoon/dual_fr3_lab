# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
오른팔 diff IK + rule-based 그리퍼로 cube를 집어 0.2m 들어올리는 데모.

시퀀스 (cube 위치/초기자세 고정):
  phase 0 (0~3초): diff IK로 TCP를 cube 위치로 이동, 그리퍼 open
  phase 1 (3~4초): 그 자리에서 그리퍼 close
  phase 2 (4~6초): 목표 z +0.2m로 들어올림
  phase 3 (6~8초): 유지, 7.8초에 성공 판정(큐브가 상판에서 15cm 이상) 출력 후 리셋

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_dual_fr3_pick.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Rule-based pick-and-lift demo with the dual FR3 right arm.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_conjugate, subtract_frame_transforms

##
# Pre-defined configs
##
from dual_fr3_lab.robots import DUAL_FR3_HIGH_PD_CFG  # isort:skip

# ---------- 배치 (preview_dual_fr3_table.py와 동일) ----------
ROOT_Z = 0.405  # base frame 높이 (바퀴가 지면에 닿는 스폰 높이)
TABLE_SIZE = (0.8, 1.0, 0.8)  # 상판 z=0.8
TABLE_POS = (0.6, 0.0, 0.4)
CUBE_POS_W = (0.45, -0.25, 0.83)  # 상판 위 3cm (world)

# ---------- link7 -> TCP 오프셋 (run_dual_fr3_diff_ik.py와 동일) ----------
TCP_OFFSET_POS = (0.0, 0.0, 0.2104)
TCP_OFFSET_QUAT = (0.9238795, 0.0, 0.0, -0.3826834)  # Rz(-45도)

# ---------- 시퀀스 타이밍 (dt=0.01 기준 스텝 수) ----------
T_REACH = 300  # 0~3초: cube로 접근
T_CLOSE = 400  # 3~4초: 그리퍼 close
T_LIFT = 600  # 4~6초: +0.2m 들어올림
T_END = 800  # 6~8초: 유지 후 판정/리셋
LIFT_HEIGHT = 0.2
GRIPPER_OPEN = 0.04
GRIPPER_CLOSE = 0.0
GRASP_QUAT = (0.0, 1.0, 0.0, 0.0)  # 아래보기 (base frame)


@configclass
class PickSceneCfg(InteractiveSceneCfg):
    """Scene: 바닥 + 테이블 + 큐브 + dual FR3."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POS),
        spawn=sim_utils.MeshCuboidCfg(
            size=TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.4, 0.25)),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_POS_W, rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
        ),
    )

    robot = DUAL_FR3_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    robot = scene["robot"]
    cube = scene["object"]
    device = sim.device

    # ---------- 오른팔 IK 준비 (run_dual_fr3_diff_ik.py와 동일 구성) ----------
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=device)
    entity_cfg = SceneEntityCfg("robot", joint_names=["right_fr3_joint[1-7]"], body_names=["right_fr3_link7"])
    entity_cfg.resolve(scene)
    ee_jacobi_idx = entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else entity_cfg.body_ids[0]
    finger_ids, _ = robot.find_joints(["right_fr3_finger_joint.*"])

    # TCP 오프셋(link7->TCP)과 역변환(TCP->link7)
    tcp_pos_l7 = torch.tensor(TCP_OFFSET_POS, device=device).repeat(scene.num_envs, 1)
    tcp_quat_l7 = torch.tensor(TCP_OFFSET_QUAT, device=device).repeat(scene.num_envs, 1)
    l7_quat_tcp = quat_conjugate(tcp_quat_l7)
    l7_pos_tcp = -quat_apply(l7_quat_tcp, tcp_pos_l7)

    # ---------- 목표 (base frame): cube 위치 / lift 위치 ----------
    grasp_pos_b = torch.tensor(
        [CUBE_POS_W[0], CUBE_POS_W[1], CUBE_POS_W[2] - ROOT_Z], device=device
    ).repeat(scene.num_envs, 1)
    lift_pos_b = grasp_pos_b.clone()
    lift_pos_b[:, 2] += LIFT_HEIGHT
    grasp_quat = torch.tensor(GRASP_QUAT, device=device).repeat(scene.num_envs, 1)

    def set_tcp_command(tcp_pos_b, tcp_quat_b):
        """TCP 목표를 link7 목표로 역변환해서 IK controller에 설정."""
        l7_pos, l7_quat = combine_frame_transforms(tcp_pos_b, tcp_quat_b, l7_pos_tcp, l7_quat_tcp)
        controller.set_command(torch.cat([l7_pos, l7_quat], dim=-1))

    # markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    tcp_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/tcp_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/tcp_goal"))

    table_top_z = TABLE_POS[2] + TABLE_SIZE[2] / 2
    sim_dt = sim.get_physics_dt()
    count = 0
    gripper_target = GRIPPER_OPEN
    goal_pos_b = grasp_pos_b

    while simulation_app.is_running():
        # ---------- phase 전환 ----------
        if count == 0:
            # 리셋: 로봇 ready pose, 큐브 원위치
            robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
            robot.reset()
            cube_state = cube.data.default_root_state.clone()
            cube_state[:, :3] += scene.env_origins
            cube.write_root_pose_to_sim(cube_state[:, :7])
            cube.write_root_velocity_to_sim(cube_state[:, 7:])
            cube.reset()
            controller.reset()
            goal_pos_b = grasp_pos_b
            gripper_target = GRIPPER_OPEN
            set_tcp_command(goal_pos_b, grasp_quat)
            print("[INFO] phase 0: reach cube (3s, gripper open)")
        elif count == T_REACH:
            gripper_target = GRIPPER_CLOSE
            print("[INFO] phase 1: close gripper (1s)")
        elif count == T_CLOSE:
            goal_pos_b = lift_pos_b
            set_tcp_command(goal_pos_b, grasp_quat)
            print(f"[INFO] phase 2: lift +{LIFT_HEIGHT}m (2s)")
        elif count == T_LIFT:
            print("[INFO] phase 3: hold (2s)")
        elif count == T_END - 20:
            cube_z = cube.data.root_pos_w[:, 2] - scene.env_origins[:, 2]
            lifted = (cube_z - table_top_z) > 0.15
            print(f"[RESULT] cube height above table: {(cube_z - table_top_z).tolist()} -> success={lifted.tolist()}")
        elif count >= T_END:
            count = 0
            continue

        # ---------- 제어 계산 ----------
        # 기본: 전체 joint를 ready pose로 (왼팔 유지)
        joint_pos_target = robot.data.default_joint_pos.clone()
        # 오른팔: IK
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, entity_cfg.joint_ids]
        l7_pose_w = robot.data.body_pose_w[:, entity_cfg.body_ids[0]]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, entity_cfg.joint_ids]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], l7_pose_w[:, 0:3], l7_pose_w[:, 3:7]
        )
        joint_pos_target[:, entity_cfg.joint_ids] = controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        # 그리퍼: rule-based open/close
        joint_pos_target[:, finger_ids] = gripper_target

        robot.set_joint_position_target(joint_pos_target)
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

        # ---------- 마커 ----------
        l7_pose_w = robot.data.body_state_w[:, entity_cfg.body_ids[0], 0:7]
        tcp_pos_w, tcp_quat_w = combine_frame_transforms(
            l7_pose_w[:, 0:3], l7_pose_w[:, 3:7], tcp_pos_l7, tcp_quat_l7
        )
        tcp_marker.visualize(tcp_pos_w, tcp_quat_w)
        goal_marker.visualize(root_pose_w[:, 0:3] + goal_pos_b, grasp_quat)


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.2, -1.8, 1.5], [0.5, -0.25, 0.8])
    scene_cfg = PickSceneCfg(num_envs=1, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
