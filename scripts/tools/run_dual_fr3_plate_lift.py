# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
양팔 판(plate) EE로 바닥의 박스를 양쪽에서 조여(squeeze) 들어올리는 rule-based 데모.

dual_fr3_plate_ee.usd 검증용: 그리퍼 없이 두 판의 마찰력만으로 박스를 든다.

시퀀스 (100스텝 = 1초):
  P0 pre     : 양팔 판을 박스 양옆 바깥(±8cm)에 수직으로 정렬
  P1 approach: 판을 박스 면 안쪽 2cm까지 천천히 전진 (squeeze 시작)
  P2 hold    : 접촉 안정화 대기
  P3 lift    : 조인 채로 양팔 목표를 동시에 +0.35m 상승 (등속 보간)
  P4 hold    : 공중 유지 + 들기 성공 판정 출력
  P5 lower   : 바닥 근처로 하강 (등속, 조임 유지)
  P6 release : 판을 바깥으로 벌려 박스 해방
  P7 retreat : 판을 위·바깥으로 후퇴 (복귀 스윙이 박스를 치지 않게)
  P8 done    : ready pose 복귀 + 최종 판정 출력 후 리셋

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_dual_fr3_plate_lift.py
    # 1 사이클만 돌고 종료 (headless 검증용)
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_dual_fr3_plate_lift.py --headless --cycles 1

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Rule-based dual-arm plate squeeze-lift demo.")
parser.add_argument("--cycles", type=int, default=0, help="종료 전 반복 사이클 수 (0 = 무한).")
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
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_conjugate, subtract_frame_transforms

##
# Pre-defined configs
##
from dual_fr3_lab.robots import DUAL_FR3_PLATE_EE_HIGH_PD_CFG, PLATE_TCP_OFFSET_POS  # isort:skip

# ---------- 배치 (world 기준) ----------
ROOT_Z = 0.405  # base frame 높이 (스폰 포함)
BOX_SIZE = 0.40  # 정육면체 박스 (판 140mm보다 큰 "큰 물체")
BOX_MASS = 2.0
BOX_POS_W = (0.5, 0.0, BOX_SIZE / 2)  # 바닥 위
PLATE_HALF_T = 0.0125  # 판 반두께 (TCP는 작업면 중심이므로 침투 계산에는 불필요, 참고용)

# ---------- link7 -> TCP(판 작업면 중심) 오프셋 ----------
TCP_OFFSET_POS = PLATE_TCP_OFFSET_POS  # (0, 0, 0.132)
TCP_OFFSET_QUAT = (1.0, 0.0, 0.0, 0.0)

# ---------- 파생 좌표 (base frame) ----------
BOX_C_B = (BOX_POS_W[0], BOX_POS_W[1], BOX_POS_W[2] - ROOT_Z)  # 박스 중심
CONTACT_Z_B = BOX_POS_W[2] + 0.05 - ROOT_Z  # 접촉 높이: 박스 중심보다 5cm 위 (기울어짐 방지)
HALF = BOX_SIZE / 2
SQUEEZE = 0.03  # 판 목표를 박스 면 안쪽으로 넣는 깊이 -> PD 오차가 누르는 힘이 됨
PRE_GAP = 0.08  # 접근 전 판-박스면 간격
LIFT_H = 0.35  # 들어올리는 높이

# 판 작업면 normal(TCP z축)이 박스를 향하도록: 오른팔(-y쪽) z->+y, 왼팔(+y쪽) z->-y
QUAT_R = (0.70711, -0.70711, 0.0, 0.0)  # Rx(-90도)
QUAT_L = (0.70711, 0.70711, 0.0, 0.0)  # Rx(+90도)


@configclass
class PlateLiftSceneCfg(InteractiveSceneCfg):
    """Scene: 바닥 + 박스 + plate EE dual FR3."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=BOX_POS_W, rot=[1, 0, 0, 0]),
        spawn=sim_utils.MeshCuboidCfg(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=5.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # 판-박스 마찰이 유일한 파지력이므로 마찰을 넉넉히
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.5, 0.2)),
        ),
    )

    robot = DUAL_FR3_PLATE_EE_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class ArmIK:
    """한쪽 팔의 IK controller + TCP 오프셋 묶음 (그리퍼 없음)."""

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
        # TCP 오프셋과 역변환
        self.tcp_pos_l7 = torch.tensor(TCP_OFFSET_POS, device=device).repeat(scene.num_envs, 1)
        self.tcp_quat_l7 = torch.tensor(TCP_OFFSET_QUAT, device=device).repeat(scene.num_envs, 1)
        self.l7_quat_tcp = quat_conjugate(self.tcp_quat_l7)
        self.l7_pos_tcp = -quat_apply(self.l7_quat_tcp, self.tcp_pos_l7)
        # 상태
        self.active = False  # False면 ready pose 유지
        self.num_envs = scene.num_envs
        self.device = device
        self._interp = None

    def set_goal(self, pos_b, quat_b, duration: int | None = None, start_b=None):
        """TCP 목표(base frame)를 link7 목표로 역변환해 설정. duration을 주면 위치를 선형 보간.

        start_b를 주면 보간 시작점을 현재 TCP 위치 대신 그 값으로 쓴다.
        squeeze 유지 이동에 필수: 현재 TCP는 박스 면에 막혀 있어서(침투 명령 미달성)
        현재 위치에서 보간을 시작하면 조임 명령이 풀려버린다. start의 y를 침투 목표와
        같게 주면 이동 내내 조임 명령이 유지된다.
        """
        pos = torch.tensor(pos_b, device=self.device).repeat(self.num_envs, 1)
        quat = torch.tensor(quat_b, device=self.device).repeat(self.num_envs, 1)
        self.goal_pos_b, self.goal_quat_b = pos, quat
        if duration is None:
            self._interp = None
            self._send_command(pos, quat)
        else:
            if start_b is not None:
                start_pos = torch.tensor(start_b, device=self.device).repeat(self.num_envs, 1)
            else:
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
        if self._interp is None:
            return
        self._interp["t"] += 1
        alpha = min(self._interp["t"] / self._interp["T"], 1.0)
        pos = self._interp["start"] * (1.0 - alpha) + self.goal_pos_b * alpha
        self._send_command(pos, self.goal_quat_b)
        if alpha >= 1.0:
            self._interp = None

    def tcp_error(self):
        """현재 TCP의 위치 오차[m]와 자세 오차[rad] (base frame 최종 목표 기준)."""
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
        self._interp = None

    def tcp_pose_w(self):
        l7_pose_w = self.robot.data.body_state_w[:, self.entity_cfg.body_ids[0], 0:7]
        return combine_frame_transforms(l7_pose_w[:, 0:3], l7_pose_w[:, 3:7], self.tcp_pos_l7, self.tcp_quat_l7)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    robot = scene["robot"]
    box = scene["object"]
    device = sim.device

    right = ArmIK("right", scene, robot, device)
    left = ArmIK("left", scene, robot, device)

    x, z = BOX_C_B[0], CONTACT_Z_B
    lift_results = []

    # ---------- phase 정의 ----------
    def p_pre():
        # duration으로 접근 속도 제어 (없으면 목표로 최대 속도 스윙)
        right.set_goal((x, -(HALF + PRE_GAP), z), QUAT_R, duration=200)
        left.set_goal((x, +(HALF + PRE_GAP), z), QUAT_L, duration=200)

    def p_approach():
        right.set_goal((x, -(HALF - SQUEEZE), z), QUAT_R, duration=150)
        left.set_goal((x, +(HALF - SQUEEZE), z), QUAT_L, duration=150)

    def p_hold():
        pass

    def p_lift():
        # start_b의 y를 침투 목표와 같게 -> 상승 내내 조임 명령 유지 (z만 보간)
        right.set_goal((x, -(HALF - SQUEEZE), z + LIFT_H), QUAT_R, duration=250, start_b=(x, -(HALF - SQUEEZE), z))
        left.set_goal((x, +(HALF - SQUEEZE), z + LIFT_H), QUAT_L, duration=250, start_b=(x, +(HALF - SQUEEZE), z))

    def p_lift_check():
        box_z = (box.data.root_pos_w[0] - scene.env_origins[0])[2].item()
        lifted = box_z > BOX_POS_W[2] + 0.5 * LIFT_H
        lift_results.append(lifted)
        print(f"[RESULT] lift: box z={box_z:.3f} (spawn {BOX_POS_W[2]:.3f}) -> lifted={lifted}")

    def p_lower():
        # 조임 유지한 채 z만 하강
        right.set_goal((x, -(HALF - SQUEEZE), z), QUAT_R, duration=200, start_b=(x, -(HALF - SQUEEZE), z + LIFT_H))
        left.set_goal((x, +(HALF - SQUEEZE), z), QUAT_L, duration=200, start_b=(x, +(HALF - SQUEEZE), z + LIFT_H))

    def p_release():
        right.set_goal((x, -(HALF + PRE_GAP), z), QUAT_R, duration=100)
        left.set_goal((x, +(HALF + PRE_GAP), z), QUAT_L, duration=100)

    def p_retreat():
        # 박스에서 위·바깥으로 천천히 빠진 뒤에 ready 복귀 (스윙이 박스를 치는 것 방지)
        right.set_goal((x - 0.1, -(HALF + PRE_GAP + 0.1), z + 0.3), QUAT_R, duration=100)
        left.set_goal((x - 0.1, +(HALF + PRE_GAP + 0.1), z + 0.3), QUAT_L, duration=100)

    def p_done():
        right.active = False
        left.active = False
        box_pos = box.data.root_pos_w[0] - scene.env_origins[0]
        placed = abs(box_pos[2].item() - BOX_POS_W[2]) < 0.05
        print(
            f"[RESULT] final: box x={box_pos[0]:.3f} y={box_pos[1]:.3f} z={box_pos[2]:.3f}"
            f" | lifted={lift_results[-1] if lift_results else False} placed_back={placed}"
            f" -> success={bool(lift_results and lift_results[-1] and placed)}"
        )

    phases = [
        ("P0 pre-align", 250, p_pre),
        ("P1 approach (squeeze)", 200, p_approach),
        ("P2 hold", 50, p_hold),
        ("P3 lift", 300, p_lift),
        ("P4 hold + 판정", 100, p_lift_check),
        ("P5 lower", 250, p_lower),
        ("P6 release", 120, p_release),
        ("P7 retreat", 120, p_retreat),
        ("P8 done (ready 복귀)", 110, p_done),
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
    cycle = 0

    while simulation_app.is_running():
        # ---------- 리셋 ----------
        if count == 0:
            robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
            robot.reset()
            box_state = box.data.default_root_state.clone()
            box_state[:, :3] += scene.env_origins
            box.write_root_pose_to_sim(box_state[:, :7])
            box.write_root_velocity_to_sim(box_state[:, 7:])
            box.reset()
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
                arm.step_command()
                joint_pos_target[:, arm.entity_cfg.joint_ids] = arm.compute()

        robot.set_joint_position_target(joint_pos_target)
        scene.write_data_to_sim()
        sim.step()
        count += 1
        if count >= total_steps:
            count = 0
            cycle += 1
            if args_cli.cycles > 0 and cycle >= args_cli.cycles:
                break
        scene.update(sim_dt)

        # ---------- 오차 로깅 (0.5초마다) ----------
        if count % 50 == 0:
            phase_name = [n for s, n, _ in boundaries if s <= count][-1]
            r_pos_err, r_ang_err = right.tcp_error()
            l_pos_err, l_ang_err = left.tcp_error()
            box_pos = box.data.root_pos_w[0] - scene.env_origins[0]
            print(
                f"[LOG] {phase_name:24s} | R err: {r_pos_err:.3f}m/{r_ang_err:.2f}rad"
                f" | L err: {l_pos_err:.3f}m/{l_ang_err:.2f}rad"
                f" | box z={box_pos[2]:.3f} y={box_pos[1]:.3f}"
            )

        # ---------- 마커 ----------
        r_marker.visualize(*right.tcp_pose_w())
        l_marker.visualize(*left.tcp_pose_w())


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.2, -1.4, 1.4], [0.5, 0.0, 0.5])
    scene_cfg = PlateLiftSceneCfg(num_envs=1, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
