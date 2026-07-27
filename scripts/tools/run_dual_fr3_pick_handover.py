# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
양팔 diff IK + rule-based 그리퍼로 cube를 오른손 -> 왼손 handover 후 왼쪽 테이블에 놓는 데모.

시퀀스 (양팔 병렬, 시간은 phases 리스트의 스텝 수 참고 / 100스텝 = 1초):
  P0 reach     : 오른팔 TCP -> cube (위에서, 손가락이 cube ±y면), 그리퍼 open
  P1 grasp     : 오른손 close
  P2a lift+wait: 오른팔 수직 +0.2m (순수 병진) / 왼팔은 동시에 대기 위치로 (open)
  P2b move&rotate+approach: 오른팔이 handover 지점으로 이동하며 손목 90도 회전 /
                 왼팔은 동시에 cube를 감싸는 최종 접근 (양손가락 직교, 충돌 없음, 보간으로 감속)
  P3 l_close   : 왼손 close (양손이 잠시 함께 잡음)
  P4 r_open    : 오른손 open
  P5 home+place: 오른팔 ready 복귀 / 왼팔은 동시에 cube를 왼쪽 테이블 위로 운반 (보간)
  P6 l_open    : 왼손 open (cube 안착)
  P7 done      : 왼팔 바로 ready pose 복귀 + 성공 판정 출력 후 리셋

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_dual_fr3_pick_handover.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Rule-based pick-handover-place demo with the dual FR3.")
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

# ---------- 배치 (이 task는 회전 공간 확보를 위해 낮은 테이블 0.6m 사용) ----------
ROOT_Z = 0.405
TABLE_SIZE = (0.8, 1.0, 0.6)  # 상판 z=0.6
TABLE_POS = (0.6, 0.0, 0.3)
CUBE_POS_W = (0.45, -0.25, 0.63)  # 오른팔 쪽 (상판 위 3cm)

# ---------- link7 -> TCP 오프셋 ----------
TCP_OFFSET_POS = (0.0, 0.0, 0.2104)
TCP_OFFSET_QUAT = (0.9238795, 0.0, 0.0, -0.3826834)  # Rz(-45도)

# ---------- 주요 pose (base frame) ----------
GRASP_QUAT_R = (0.0, 1.0, 0.0, 0.0)  # 오른손 pick: 위에서 아래보기, 손가락이 world ±y면을 잡음
# 오른손 handover: 손목을 90도 돌려 수평으로 왼손을 마주봄 (TCP z=+y, 손가락은 수직 ±z)
HANDOVER_QUAT_R = (0.0, 0.0, 0.7071, 0.7071)
# 왼손: +y쪽에서 -y방향으로 접근(TCP z=-y), 손가락은 world x축(±x면)을 잡음
# -> 오른손가락(수직)과 왼손가락(x축)이 직교해 충돌 없음
GRASP_QUAT_L = (0.5, 0.5, 0.5, -0.5)
HANDOVER_POS_B = (0.5, 0.0, 0.45)  # 양팔 중간, 공중 (world z=0.855)
L_WAIT_OFFSET_Y = 0.18  # 왼손 대기 위치: handover에서 +y로 이 거리만큼
# 왼쪽 테이블 위 놓는 지점. world z=0.705 -> open 시 cube가 상판(0.6)까지 약 4cm 낙하
# (상판에 살짝 닿게 놓으려면 z=0.23: 상판 0.6 - ROOT_Z 0.405 + cube 반높이 0.026 + 여유)
PLACE_POS_B = (0.45, 0.25, 0.3)
GRIPPER_OPEN = 0.04
GRIPPER_CLOSE = 0.0


@configclass
class HandoverSceneCfg(InteractiveSceneCfg):
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


class ArmIK:
    """한쪽 팔의 IK controller + TCP 오프셋 + 그리퍼 상태 묶음."""

    def __init__(self, side: str, scene: InteractiveScene, robot, device):
        self.side = side
        self.robot = robot
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=device)
        self.entity_cfg = SceneEntityCfg(
            "robot", joint_names=[f"{side}_fr3_joint[1-7]"], body_names=[f"{side}_fr3_link7"]
        )
        self.entity_cfg.resolve(scene)
        self.ee_jacobi_idx = self.entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else self.entity_cfg.body_ids[0]
        self.finger_ids, _ = robot.find_joints([f"{side}_fr3_finger_joint.*"])
        # TCP 오프셋과 역변환
        self.tcp_pos_l7 = torch.tensor(TCP_OFFSET_POS, device=device).repeat(scene.num_envs, 1)
        self.tcp_quat_l7 = torch.tensor(TCP_OFFSET_QUAT, device=device).repeat(scene.num_envs, 1)
        self.l7_quat_tcp = quat_conjugate(self.tcp_quat_l7)
        self.l7_pos_tcp = -quat_apply(self.l7_quat_tcp, self.tcp_pos_l7)
        # 상태
        self.active = False  # False면 ready pose 유지
        self.gripper = GRIPPER_OPEN
        self.num_envs = scene.num_envs
        self.device = device
        self._interp = None  # 목표 보간 상태 (set_goal의 duration 옵션)

    def set_goal(self, pos_b, quat_b, duration: int | None = None):
        """TCP 목표(base frame)를 link7 목표로 역변환해 설정하고 IK를 활성화.

        duration(스텝 수)을 주면 현재 위치에서 목표까지 위치를 선형 보간한 중간 목표를
        매 스텝 갱신한다 -> 팔의 이동 속도를 직접 제어 (느리게 접근시킬 때 사용).
        """
        pos = torch.tensor(pos_b, device=self.device).repeat(self.num_envs, 1)
        quat = torch.tensor(quat_b, device=self.device).repeat(self.num_envs, 1)
        self.goal_pos_b, self.goal_quat_b = pos, quat  # logging용 (최종 목표)
        if duration is None:
            self._interp = None
            self._send_command(pos, quat)
        else:
            # 현재 TCP 위치(base frame)를 시작점으로 저장
            tcp_pos_w, tcp_quat_w = self.tcp_pose_w()
            root_pose_w = self.robot.data.root_pose_w
            start_pos, _ = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], tcp_pos_w, tcp_quat_w
            )
            self._interp = {"start": start_pos.clone(), "t": 0, "T": duration}
            self._send_command(start_pos, quat)
        self.active = True

    def _send_command(self, pos, quat):
        l7_pos, l7_quat = combine_frame_transforms(pos, quat, self.l7_pos_tcp, self.l7_quat_tcp)
        self.controller.set_command(torch.cat([l7_pos, l7_quat], dim=-1))

    def step_command(self):
        """보간 중이면 중간 목표를 한 스텝 전진시킨다. 매 시뮬 스텝 호출."""
        if self._interp is None:
            return
        self._interp["t"] += 1
        alpha = min(self._interp["t"] / self._interp["T"], 1.0)
        pos = self._interp["start"] * (1.0 - alpha) + self.goal_pos_b * alpha
        self._send_command(pos, self.goal_quat_b)
        if alpha >= 1.0:
            self._interp = None

    def tcp_error(self):
        """현재 TCP의 위치 오차[m]와 자세 오차[rad]를 반환 (base frame 목표 기준)."""
        if not self.active:
            return 0.0, 0.0
        tcp_pos_w, tcp_quat_w = self.tcp_pose_w()
        root_pose_w = self.robot.data.root_pose_w
        tcp_pos_b, tcp_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], tcp_pos_w, tcp_quat_w
        )
        pos_err = torch.norm(self.goal_pos_b - tcp_pos_b, dim=-1)[0].item()
        dot = torch.abs(torch.sum(self.goal_quat_b * tcp_quat_b, dim=-1)).clamp(max=1.0)
        ang_err = (2.0 * torch.acos(dot))[0].item()
        return pos_err, ang_err

    def compute(self):
        """IK를 풀어 해당 팔 joint 목표를 반환."""
        robot = self.robot
        jacobian = robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.entity_cfg.joint_ids]
        l7_pose_w = robot.data.body_pose_w[:, self.entity_cfg.body_ids[0]]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, self.entity_cfg.joint_ids]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], l7_pose_w[:, 0:3], l7_pose_w[:, 3:7]
        )
        return self.controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

    def reset(self):
        self.controller.reset()
        self.active = False
        self.gripper = GRIPPER_OPEN
        self._interp = None

    def tcp_pose_w(self):
        l7_pose_w = self.robot.data.body_state_w[:, self.entity_cfg.body_ids[0], 0:7]
        return combine_frame_transforms(l7_pose_w[:, 0:3], l7_pose_w[:, 3:7], self.tcp_pos_l7, self.tcp_quat_l7)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    robot = scene["robot"]
    cube = scene["object"]
    device = sim.device

    right = ArmIK("right", scene, robot, device)
    left = ArmIK("left", scene, robot, device)

    # 파생 좌표
    cube_grasp_b = (CUBE_POS_W[0], CUBE_POS_W[1], CUBE_POS_W[2] - ROOT_Z)
    ho = HANDOVER_POS_B
    l_wait_b = (ho[0], ho[1] + L_WAIT_OFFSET_Y, ho[2])
    table_top_z = TABLE_POS[2] + TABLE_SIZE[2] / 2

    # ---------- phase 정의: (이름, 스텝 수, 진입 시 실행 함수) ----------
    def p_reach():
        right.set_goal(cube_grasp_b, GRASP_QUAT_R)
        right.gripper = GRIPPER_OPEN

    def p_grasp():
        right.gripper = GRIPPER_CLOSE

    def p_lift_l_wait():
        # 오른팔: 잡은 자세 그대로 수직 +0.2m (순수 병진, 회전은 P2b에서)
        # 왼팔: 동시에 handover 옆 대기 위치로 이동
        right.set_goal((cube_grasp_b[0], cube_grasp_b[1], cube_grasp_b[2] + 0.2), GRASP_QUAT_R)
        left.set_goal(l_wait_b, GRASP_QUAT_L)
        left.gripper = GRIPPER_OPEN

    def p_move_rotate_l_approach():
        # 오른팔: handover 지점으로 이동 + 손목 90도 회전 동시 (최대 속도, ~1.5초에 도착)
        # 왼팔: cube를 감싸는 최종 접근을 180스텝(1.8초) 보간으로 천천히 -> 오른팔이 먼저 도착
        right.set_goal(ho, HANDOVER_QUAT_R)
        left.set_goal(ho, GRASP_QUAT_L, duration=180)

    def p_l_close():
        left.gripper = GRIPPER_CLOSE

    def p_r_open():
        right.gripper = GRIPPER_OPEN

    def p_r_home_l_place():
        # 오른팔: ready pose 복귀 / 왼팔: 동시에 place 위치로 운반
        # (보간으로 등속 운반 -> 가감속에 의한 grip 내 미끄러짐 방지)
        right.active = False
        left.set_goal(PLACE_POS_B, GRASP_QUAT_L, duration=80)

    def p_l_open():
        left.gripper = GRIPPER_OPEN

    def p_done():
        left.active = False  # 왼팔도 ready pose로 복귀
        cube_pos = cube.data.root_pos_w[0] - scene.env_origins[0]
        # 성공: cube 중심이 상판 위 cube 반높이(0.026m) 근처(±6cm) & 왼쪽 영역(y>0.1)
        on_table = abs(cube_pos[2].item() - (table_top_z + 0.026)) < 0.06
        on_left = cube_pos[1].item() > 0.1
        print(
            f"[RESULT] cube pos (world): x={cube_pos[0]:.3f} y={cube_pos[1]:.3f} z={cube_pos[2]:.3f}"
            f" | on_table={on_table} on_left_side={on_left} -> success={on_table and on_left}"
        )

    phases = [
        ("P0 reach (right->cube)", 100, p_reach),
        ("P1 grasp (right close)", 50, p_grasp),
        ("P2a lift + left wait (동시)", 100, p_lift_l_wait),
        ("P2b move&rotate + left approach (동시)", 200, p_move_rotate_l_approach),
        ("P3 left close", 50, p_l_close),
        ("P4 right open", 50, p_r_open),
        ("P5 right home + left place (동시)", 150, p_r_home_l_place),
        ("P6 left open", 50, p_l_open),
        ("P7 done (left home + 판정)", 150, p_done),
    ]
    boundaries = []
    acc = 0
    for name, steps, fn in phases:
        boundaries.append((acc, name, fn))
        acc += steps
    total_steps = acc

    # markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
    r_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/right_tcp"))
    l_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/left_tcp"))

    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        # ---------- 리셋 ----------
        if count == 0:
            robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
            robot.reset()
            cube_state = cube.data.default_root_state.clone()
            cube_state[:, :3] += scene.env_origins
            cube.write_root_pose_to_sim(cube_state[:, :7])
            cube.write_root_velocity_to_sim(cube_state[:, 7:])
            cube.reset()
            right.reset()
            left.reset()

        # ---------- phase 진입 ----------
        for start, name, fn in boundaries:
            if count == start:
                fn()
                print(f"[INFO] {name}")

        # ---------- 제어 ----------
        joint_pos_target = robot.data.default_joint_pos.clone()
        for arm in (right, left):
            if arm.active:
                arm.step_command()  # 보간 중이면 중간 목표 갱신
                joint_pos_target[:, arm.entity_cfg.joint_ids] = arm.compute()
            joint_pos_target[:, arm.finger_ids] = arm.gripper

        robot.set_joint_position_target(joint_pos_target)
        scene.write_data_to_sim()
        sim.step()
        count += 1
        if count >= total_steps:
            count = 0
        scene.update(sim_dt)

        # ---------- 오차 로깅 (0.5초마다) ----------
        if count % 50 == 0:
            phase_name = [n for s, n, _ in boundaries if s <= count][-1]
            r_pos_err, r_ang_err = right.tcp_error()
            l_pos_err, l_ang_err = left.tcp_error()
            cube_pos = cube.data.root_pos_w[0] - scene.env_origins[0]
            r_tcp_pos_w, _ = right.tcp_pose_w()
            l_tcp_pos_w, _ = left.tcp_pose_w()
            cube_to_rtcp = torch.norm(cube.data.root_pos_w[0] - r_tcp_pos_w[0]).item()
            cube_to_ltcp = torch.norm(cube.data.root_pos_w[0] - l_tcp_pos_w[0]).item()
            print(
                f"[LOG] {phase_name:32s} | R err: {r_pos_err:.3f}m/{r_ang_err:.2f}rad"
                f" | L err: {l_pos_err:.3f}m/{l_ang_err:.2f}rad"
                f" | cube z={cube_pos[2]:.3f} y={cube_pos[1]:.3f}"
                f" | cube<->R {cube_to_rtcp:.3f}m L {cube_to_ltcp:.3f}m"
            )

        # ---------- 마커 ----------
        r_marker.visualize(*right.tcp_pose_w())
        l_marker.visualize(*left.tcp_pose_w())


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.4, -1.2, 1.6], [0.5, 0.0, 0.8])
    scene_cfg = HandoverSceneCfg(num_envs=1, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
