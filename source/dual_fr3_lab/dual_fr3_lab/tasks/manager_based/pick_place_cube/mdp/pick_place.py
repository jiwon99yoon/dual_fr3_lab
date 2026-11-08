# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""큐브 pick-and-place MDP (2026-11-07). 오른팔 + Franka hand(UMI 팁), 75 cm 책상.

책상 위에 세워 놓인 큐브(5x5x10 cm, yaw ±45°)를 위에서 집어 무작위 목표 위치에 세워 놓고 손을 뗀 뒤 올린다.
보상은 핸드오버와 같은 단계 판정 하나에 모은다 (PickPlaceState):

    progress = STAGE_BASE[s] + shaping_s,   shaping_s in [0, 1]

    0 REACH     오른 TCP -> 파지점 (위에서, 손가락 닫힘 방향을 큐브 면에 맞춤), 파지, 쥔 채 들어 올린 높이
    1 HELD      오른손 파지 & 한 번 들었음. 큐브 -> 목표 (높이 무관: 내려놓는 중에도 1)
    2 PLACED    들었던 큐브가 목표 xy XY_TOL 이내, 기울기 UPRIGHT_DEG 이내로 책상에 놓임 & 손가락 접촉. 열기 요청, 손가락 벌리기
    3 RELEASED  위 + 손가락 비접촉. 손 올리기. 한 번 PLACED였으면 놓은 뒤 목표 RELEASE_XY_TOL·RELEASE_TILT_DEG 안이면 유지
    4 SUCCESS   위(XY_TOL, UPRIGHT_DEG) + TCP가 큐브 윗면보다 HAND_CLEAR 위 (성공)

성공 조건(사용자 11-07): 목표 xy 1 cm, 기울기 5°, 놓음, 손 올림. 11-08 xy 2.5 cm로 완화 (ManiSkill PickCube 2.5 cm,
실물 쌓기 RGB-Stacking 3 cm와 같은 수준. 1 cm로 실물에서 놓은 레퍼런스는 못 찾음).
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.commands import UniformPoseCommand, UniformPoseCommandCfg
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_from_angle_axis, quat_mul, subtract_frame_transforms

from dual_fr3_lab.robots import UMI_TCP_OFFSET_POS, UMI_TCP_OFFSET_ROT
from dual_fr3_lab.tasks.manager_based.lift_plate.joint_pos_env_cfg import (  # isort: skip
    BOX_OBS_DELAY_STEPS,
    _box_obs_dropout_mask,
    _box_obs_with_dropout,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

# ---------- 배치 (base frame. base는 world와 회전이 같고 world z ROBOT_BASE_Z에 있다) ----------
ROBOT_BASE_Z = 0.405
TABLE_TOP_Z_B = 0.345  # 75 cm 책상 (캐비닛 Table75와 같음)
TABLE_X_B = (0.40, 1.00)  # 실물 600 x 1000 책상, 앞 모서리는 캐비닛 배치와 같은 x 0.40
TABLE_WIDTH = 1.0
CUBE_SIZE = (0.05, 0.05, 0.10)  # cube lift와 같은 실물 큐브
CUBE_REST_Z_B = TABLE_TOP_Z_B + 0.5 * CUBE_SIZE[2]  # 세워 놓였을 때 중심 높이 (base)
# 집는 곳·놓는 곳 범위 (큐브 중심, base): 책상 안 사각형 & 오른팔 link0 축에서 수평 REACH_RADIUS 안.
# rule-based(11-07): 위에서 집는 자세로 (0.63, -0.25)(link0에서 0.63 m)는 됨, (0.65, -0.45)·(0.65, 0.05)(0.69~0.70 m)는
# 팔꿈치가 거의 펴진 채 3 cm 못 미침
CUBE_X_B = (0.45, 0.65)
CUBE_Y_B = (-0.45, 0.05)
RIGHT_LINK0_XY_B = (0.0, -0.215)
REACH_RADIUS = 0.62
CUBE_YAW = math.radians(45.0)
GOAL_MIN_DIST = 0.10  # 놓는 곳은 집는 곳에서 이만큼 떨어뜨린다

# ---------- DR (11-08) ----------
# 책상 높이: 실물 책상과 로봇 base 높이 차의 불확실성. 판정·큐브 reset은 env별 실제 높이를 쓰고, 목표 관측 z는 실물 노드처럼
# 명목 높이(CUBE_REST_Z_B)로 둔다
TABLE_DZ_RANGE = (-0.01, 0.01)
# 큐브 pose 관측에 에피소드마다 고정되는 오프셋 (손-눈 보정 오차). step마다 바뀌는 잡음과 달리 평균으로 지워지지 않는다
CUBE_OBS_BIAS_POS = 0.005  # m, 축마다 균일
CUBE_OBS_BIAS_YAW = math.radians(3.0)

# ---------- 판정 ----------
# 파지점 = 큐브 중심 + 큐브 z축 x GRASP_OFFSET. 11-09에 2 cm(손가락 끝이 책상 위 7 cm)에서 0(큐브 중심, 책상 위 5 cm)으로 내렸다.
# rule-based(정책과 같은 큐브 관측, 256 env): 7 cm 251, 5 cm 255, 4 cm 254, 2.3 cm 232 성공. 닫을 때 기울기 p90 2.8° -> 1.0°,
# 손가락-책상 접촉 env 0 / 7 / 36 / 95. A model_700은 보상과 달리 손가락 끝 2.3 cm에서 쥐었다
GRASP_OFFSET = 0.0
GRASP_FORCE = 0.5  # N, 양 패드 모두 이 이상이면 파지 (cube lift와 같음)
LIFT_DELTA = 0.04  # 들림: 중심이 세워 놓인 높이 + 4 cm 위
REST_TOL = 0.01  # 중심이 세워 놓인 높이 + 1 cm 아래면 책상에 놓인 것 (누운 큐브도 포함)
XY_TOL = 0.025  # 성공: 목표 xy 2.5 cm 이내
UPRIGHT_DEG = 5.0  # 성공: 기울기 5° 이내
TOPPLE_DEG = 45.0  # 쥐지 않은 큐브가 이보다 기울면 종료
HAND_CLEAR = 0.03  # 성공: TCP(손가락 끝 중앙)가 큐브 윗면보다 3 cm 위
# 손가락 끝(TCP) 높이 하한: 책상 위 이보다 낮으면 벌점 (책상 높이 DR ±1 cm, 큐브 관측 오프셋 ±5 mm에 여유).
# 11-09 A model_700은 의도(책상 위 7 cm)보다 깊게, 손가락 끝이 책상 위 중앙값 2.3 cm(최소 0.7 cm)에서 쥐었다
TIP_FLOOR = 0.03
# 하한은 손이 책상 위에 있을 때만 (앞 모서리 x 0.40에서 손가락 폭만큼 앞까지). ready TCP는 x 0.30, 책상 높이 + 6.6 cm(중앙값)라
# 책상 높이 DR로 256개 중 5개가 3 cm 아래로 잡혔다
TIP_FLOOR_EDGE_MARGIN = 0.03
TILT_EXCESS_DEG = 15.0  # -Tilt 변형: 기울기 중 이 각을 넘는 만큼 벌점 (캐비닛 책 -Tilt와 같은 15°)
DROP_GOAL_RADIUS = 0.05  # 들었던 큐브를 목표 5 cm 밖에서 놓으면 낙하로 종료
# 놓은 뒤 유지 범위 (11-09): 한 번 PLACED였으면 손을 놓은 뒤 이 안에 서 있는 동안 RELEASED를 유지한다. 4090 A는 놓을 위치에서
# 쥔 채 버텼다 (PLACED 초당 20, 놓다가 2.5 cm·5°를 벗어나면 단계 0으로 초당 약 3). 성공은 그대로 XY_TOL, UPRIGHT_DEG
RELEASE_XY_TOL = DROP_GOAL_RADIUS
RELEASE_TILT_DEG = 15.0
FINGER_OPEN = 0.04
REACH_STD = 0.1
TRANSPORT_STD = 0.3
PLACE_STD = 0.03

STAGE_NAMES = ("reach", "held", "placed", "released", "success")
NUM_STAGES = len(STAGE_NAMES)
STAGE_SUCCESS = NUM_STAGES - 1
STAGE_BASE = (0.0, 2.0, 4.0, 6.0, 8.0)

PAD_SENSORS = ("right_leftfinger_contact", "right_rightfinger_contact")
GOAL_COMMAND = "goal_pose"
GRIPPER_ACTION = "gripper_action"


def _quat_z(angle: float, device) -> torch.Tensor:
    return torch.tensor([0.0, 0.0, math.sin(0.5 * angle), math.cos(0.5 * angle)], device=device)


def canonical_cube_quat(quat: torch.Tensor) -> torch.Tensor:
    """큐브 단면이 정사각형이라 z축 90° 회전 4개는 같은 큐브다. 그중 회전각이 가장 작은 것(|w| 최대, w>=0).

    선 큐브면 yaw를 ±45°로 접는다. 실물 노드도 ArUco 자세에 같은 처리를 한다. quat은 (x, y, z, w).
    """
    rots = torch.stack([_quat_z(k * 0.5 * math.pi, quat.device) for k in range(4)])  # [4, 4]
    cands = torch.stack([quat_mul(quat, r.expand_as(quat)) for r in rots], 0)  # [4, n, 4]
    best = cands[..., 3].abs().argmax(0)
    q = cands[best, torch.arange(quat.shape[0], device=quat.device)]
    return q * torch.where(q[:, 3:4] >= 0.0, 1.0, -1.0)


class PickPlaceState:
    """env당 단계와 기억(ever_lifted)을 policy step마다 한 번 계산해 모든 term이 공유한다 (HandoverState와 같은 방식)."""

    def __init__(self, env: ManagerBasedRLEnv):
        self.env = env
        n, dev = env.num_envs, env.device
        self.stage = torch.zeros(n, dtype=torch.long, device=dev)
        self.max_stage = torch.zeros_like(self.stage)
        self.ever_lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_placed = torch.zeros_like(self.ever_lifted)
        self.ema = {f"reach_{name}": 0.0 for name in STAGE_NAMES[1:]}
        robot = env.scene["robot"]
        self._link7 = robot.body_names.index("right_fr3_link7")
        self._finger = robot.find_joints("right_fr3_finger_joint1")[0][0]
        self._off_p = torch.tensor(UMI_TCP_OFFSET_POS, device=dev).repeat(n, 1)
        self._off_q = torch.tensor(UMI_TCP_OFFSET_ROT, device=dev).repeat(n, 1)
        self._ex = torch.tensor([1.0, 0.0, 0.0], device=dev).repeat(n, 1)
        self._ey = torch.tensor([0.0, 1.0, 0.0], device=dev).repeat(n, 1)
        self._ez = torch.tensor([0.0, 0.0, 1.0], device=dev).repeat(n, 1)
        self._base = torch.tensor(STAGE_BASE, device=dev)
        self._upright_cos = math.cos(math.radians(UPRIGHT_DEG))
        self._topple_cos = math.cos(math.radians(TOPPLE_DEG))
        self._step = -1
        self.update()

    def reset(self, env_ids):
        self.stage[env_ids] = 0
        self.max_stage[env_ids] = 0
        self.ever_lifted[env_ids] = False
        self.ever_placed[env_ids] = False
        self._step = -1

    def _pad_forces(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(min, max) 오른손 패드-큐브 접촉력. 패드 센서 filter[0]=Object."""
        forces = []
        for name in PAD_SENSORS:
            fm = self.env.scene.sensors[name].data.force_matrix_w.torch[:, :, 0, :]
            forces.append(torch.norm(fm, dim=-1).amax(dim=-1))
        return torch.minimum(*forces), torch.maximum(*forces)

    def update(self):
        env = self.env
        if self._step == env.common_step_counter:
            return
        self._step = env.common_step_counter
        robot, obj = env.scene["robot"], env.scene["object"]
        root_p, root_q = robot.data.root_pos_w.torch, robot.data.root_quat_w.torch

        l7_p = robot.data.body_pos_w.torch[:, self._link7]
        l7_q = robot.data.body_quat_w.torch[:, self._link7]
        tcp_w = l7_p + quat_apply(l7_q, self._off_p)
        tcp_q = quat_mul(l7_q, self._off_q)
        cube_w, cube_q = obj.data.root_pos_w.torch, obj.data.root_quat_w.torch
        self.tcp_b = subtract_frame_transforms(root_p, root_q, tcp_w)[0]
        self.cube_b = subtract_frame_transforms(root_p, root_q, cube_w)[0]
        cube_z = quat_apply(cube_q, self._ez)
        self.cube_z = cube_z

        min_f, max_f = self._pad_forces()
        self.grasp, self.touch = min_f > GRASP_FORCE, max_f > GRASP_FORCE

        goal_b = env.command_manager.get_command(GOAL_COMMAND)[:, :3]
        self.goal_b = goal_b
        d_goal = torch.norm(self.cube_b - goal_b, dim=-1)
        self.d_goal_xy = torch.norm(self.cube_b[:, :2] - goal_b[:, :2], dim=-1)

        # 접근: 파지점까지 거리, 위에서(TCP z축이 아래), 손가락 닫힘 방향(TCP y)이 큐브 면 법선과 나란히 (90° 주기)
        grasp_pt = self.cube_b + cube_z * GRASP_OFFSET
        d_grasp = torch.norm(self.tcp_b - grasp_pt, dim=-1)
        top_down = (-quat_apply(tcp_q, self._ez)[:, 2]).clamp(0.0, 1.0)
        close_dir = quat_apply(tcp_q, self._ey)
        cube_x = quat_apply(cube_q, self._ex)
        dyaw = torch.atan2(close_dir[:, 1], close_dir[:, 0]) - torch.atan2(cube_x[:, 1], cube_x[:, 0])
        align = 0.5 * (1.0 + torch.cos(4.0 * dyaw))
        self.orient = top_down * align

        rest_z = CUBE_REST_Z_B + table_dz(env)
        lifted = self.cube_b[:, 2] > rest_z + LIFT_DELTA
        self.lifted = lifted
        self.resting = self.cube_b[:, 2] < rest_z + REST_TOL
        self.tilt = torch.acos(cube_z[:, 2].clamp(-1.0, 1.0))
        upright = cube_z[:, 2] > self._upright_cos
        self.toppled = cube_z[:, 2] < self._topple_cos
        # reset 직후(물리 step 전)에는 접촉 센서가 이전 에피소드 값이라 기억을 갱신하지 않는다
        fresh = env.episode_length_buf > 0
        self.ever_lifted |= lifted & self.grasp & fresh
        held = self.grasp & self.ever_lifted
        at_goal = self.ever_lifted & self.resting & upright & (self.d_goal_xy < XY_TOL)
        self.clear = self.tcp_b[:, 2] - (self.cube_b[:, 2] + 0.5 * CUBE_SIZE[2])
        self.at_goal = at_goal
        self.ever_placed |= at_goal & self.touch & fresh
        near_goal = (
            self.ever_placed & self.resting & (self.d_goal_xy < RELEASE_XY_TOL) & (self.tilt < math.radians(RELEASE_TILT_DEG))
        )

        conditions = (
            held,
            at_goal & self.touch,
            (at_goal | near_goal) & ~self.touch,
            at_goal & ~self.touch & (self.clear > HAND_CLEAR),
        )
        stage = torch.zeros_like(self.stage)
        for s, cond in enumerate(conditions, start=1):
            stage = torch.where(cond, s, stage)

        finger = robot.data.joint_pos.torch[:, self._finger]
        half = 0.5 * CUBE_SIZE[0]
        # PLACED: 쥔 손가락은 23.4 mm라 25 mm를 넘으면 바로 비접촉(RELEASED)이 된다. 손가락 벌림만으로는 이 단계 shaping이 늘 0이라
        # 열기를 요청한 순간에 준다 (명령이 실행되기까지 0.45~0.9 s)
        open_request = (~env.action_manager.get_term(GRIPPER_ACTION)._request_close).float()
        # 쥔 채 들어 올린 높이 (0~LIFT_DELTA -> 0~1). 11-09 첫 run은 이 항 없이 쥐고 버티기에서 멈췄다 (들기 0/64)
        lift_frac = ((self.cube_b[:, 2] - rest_z) / LIFT_DELTA).clamp(0.0, 1.0) * self.grasp.float()
        shaping = torch.stack(
            (
                0.4 * (1.0 - torch.tanh(d_grasp / REACH_STD)) + 0.2 * self.orient + 0.2 * self.grasp.float()
                + 0.2 * lift_frac,
                0.5 * (1.0 - torch.tanh(d_goal / TRANSPORT_STD)) + 0.5 * (1.0 - torch.tanh(d_goal / PLACE_STD)),
                0.5 * open_request + 0.5 * ((finger - half) / (FINGER_OPEN - half)).clamp(0.0, 1.0),  # 열기 요청, 손가락 벌리기
                ((self.clear + HAND_CLEAR) / (2.0 * HAND_CLEAR)).clamp(0.0, 1.0),  # 손 올리기
                torch.ones_like(d_grasp),
            ),
            dim=-1,
        )
        self.stage = stage
        self.max_stage = torch.where(fresh, torch.maximum(self.max_stage, stage), self.max_stage)
        self.shaping = shaping.gather(1, stage[:, None]).squeeze(1)
        self.progress = self._base[stage] + self.shaping


def pick_place_state(env: ManagerBasedRLEnv) -> PickPlaceState:
    """env에 하나뿐인 PickPlaceState (처음 부르는 manager가 만든다)."""
    if not hasattr(env, "_pick_place_state"):
        env._pick_place_state = PickPlaceState(env)
    state = env._pick_place_state
    state.update()
    return state


# ---------- rewards ----------


class pick_place_progress(ManagerTermBase):
    """STAGE_BASE[s] + shaping_s. 이 term의 reset이 에피소드 기억(ever_lifted 등)을 지운다."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._state = pick_place_state(env)

    def reset(self, env_ids=None):
        self._state.reset(slice(None) if env_ids is None else env_ids)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        return pick_place_state(env).progress


def pick_place_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """성공 단계(목표에 세워 놓고 손 떼고 올림) 지표. 작은 weight로 TB 기록용."""
    return (pick_place_state(env).stage == STAGE_SUCCESS).float()


def cube_tilt_while_held(env: ManagerBasedRLEnv) -> torch.Tensor:
    """들린 큐브를 쥐고 있을 때 큐브 z축 기울기 (rad). 세워 놓아야 하므로 운반 중에도 세운다."""
    s = pick_place_state(env)
    return s.tilt * (s.lifted & s.grasp).float()


def cube_tilt_excess(env: ManagerBasedRLEnv, threshold_deg: float = TILT_EXCESS_DEG) -> torch.Tensor:
    """큐브 기울기 중 threshold_deg를 넘는 만큼 (rad), 언제나 (캐비닛 책 -Tilt의 book_tilt_excess와 같은 형태)."""
    return (pick_place_state(env).tilt - math.radians(threshold_deg)).clamp(min=0.0)


def finger_table_force(env: ManagerBasedRLEnv) -> torch.Tensor:
    """오른손 두 손가락 중 큰 쪽의 손가락-책상 접촉력 (N). 패드 센서 filter[1] = Table. (num_envs,)"""
    forces = [torch.norm(env.scene.sensors[name].data.force_matrix_w.torch[:, :, 1, :], dim=-1).amax(-1) for name in PAD_SENSORS]
    return torch.maximum(*forces)


def finger_table_contact(env: ManagerBasedRLEnv, threshold: float = 1.0) -> torch.Tensor:
    """손가락이 책상에 닿음 1 / 아님 0. 11-09 A model_700은 쥐는 순간 128개 중 103개가 최대 136 N으로 책상을 찍었다."""
    return (finger_table_force(env) > threshold).float()


def tip_below_floor(env: ManagerBasedRLEnv, floor: float = TIP_FLOOR) -> torch.Tensor:
    """손가락 끝(TCP)이 책상 위 floor보다 낮은 만큼 (0 -> 책상 높이에서 1), 손이 책상 위에 있을 때. 책상 높이는 env별 실제 값."""
    s = pick_place_state(env)
    height = s.tcp_b[:, 2] - (TABLE_TOP_Z_B + table_dz(env))
    over_table = s.tcp_b[:, 0] > TABLE_X_B[0] - TIP_FLOOR_EDGE_MARGIN
    return ((floor - height) / floor).clamp(0.0, 1.0) * over_table.float()


# ---------- terminations ----------


def cube_dropped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """한 번 들린 큐브가 쥐이지 않은 채 목표 DROP_GOAL_RADIUS 밖 책상에 놓이면 종료."""
    s = pick_place_state(env)
    return s.ever_lifted & s.resting & ~s.grasp & (s.d_goal_xy > DROP_GOAL_RADIUS)


def cube_toppled(env: ManagerBasedRLEnv) -> torch.Tensor:
    """쥐지 않은 큐브가 TOPPLE_DEG보다 기울면(쓰러뜨림) 종료."""
    s = pick_place_state(env)
    return s.toppled & ~s.grasp


# ---------- observations (actor: 실물에서 얻는 신호) ----------


def cube_position_vision(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """ArUco 큐브 중심 (base). box BoxDrop과 같은 지연·끊김(끊기면 마지막 값 유지). (num_envs, 3)"""
    robot, obj = env.scene[robot_cfg.name], env.scene[object_cfg.name]
    pos = subtract_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, obj.data.root_pos_w.torch)[0]
    return _box_obs_with_dropout(env, "cube_position", pos + _cube_obs_bias(env)["pos"])


def cube_orientation_vision(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """ArUco 큐브 자세 (base, xyzw, 90° 대칭을 접은 것), 위치와 같은 지연·끊김과 고정 yaw 오프셋. (num_envs, 4)"""
    quat = quat_mul(_cube_obs_bias(env)["quat"], _cube_quat_b(env, robot_cfg, object_cfg))
    return _box_obs_with_dropout(env, "cube_orientation", canonical_cube_quat(quat))


def _cube_obs_bias(env: ManagerBasedRLEnv) -> dict:
    """에피소드마다 고정되는 큐브 pose 관측 오프셋: pos (n, 3), base z축 yaw quat (n, 4). 에피소드 첫 관측에서 다시 뽑는다."""
    n, dev = env.num_envs, env.device
    if not hasattr(env, "_cube_obs_bias"):
        env._cube_obs_bias = {
            "pos": torch.zeros(n, 3, device=dev),
            "quat": torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).repeat(n, 1),
            "step": -1,
        }
    st = env._cube_obs_bias
    if st["step"] != env.common_step_counter:
        st["step"] = env.common_step_counter
        fresh = env.episode_length_buf == 0
        k = int(fresh.sum())
        if k:
            st["pos"][fresh] = torch.empty(k, 3, device=dev).uniform_(-CUBE_OBS_BIAS_POS, CUBE_OBS_BIAS_POS)
            yaw = torch.empty(k, device=dev).uniform_(-CUBE_OBS_BIAS_YAW, CUBE_OBS_BIAS_YAW)
            st["quat"][fresh] = quat_from_angle_axis(yaw, torch.tensor([0.0, 0.0, 1.0], device=dev).repeat(k, 1))
    return st


def cube_pos_rel_tcp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """관측된 큐브 중심(cube_position_vision과 같은 값) - 오른 TCP (base). 실물: ArUco - FK. (num_envs, 3)

    cube_position_vision이 같은 step에 먼저 계산돼야 한다 (obs group에서 앞에 둔다).
    """
    held = env._box_obs_dropout["held"].get("cube_position") if hasattr(env, "_box_obs_dropout") else None
    s = pick_place_state(env)
    return (s.cube_b if held is None else held) - s.tcp_b


def goal_position(env: ManagerBasedRLEnv, command_name: str = GOAL_COMMAND) -> torch.Tensor:
    """놓을 곳의 큐브 중심 (base). 놓는 자세는 세움(yaw 무관)이라 quat은 뺀다."""
    return env.command_manager.get_command(command_name)[:, :3]


# ---------- observations (critic 전용) ----------


def cube_orientation_true(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    return canonical_cube_quat(_cube_quat_b(env, robot_cfg, object_cfg))


def _cube_quat_b(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    robot, obj = env.scene[robot_cfg.name], env.scene[object_cfg.name]
    _, quat = subtract_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, obj.data.root_pos_w.torch, obj.data.root_quat_w.torch
    )
    return quat


def cube_position_true(env: ManagerBasedRLEnv) -> torch.Tensor:
    return pick_place_state(env).cube_b


def tcp_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    return pick_place_state(env).tcp_b


def pick_place_privileged(env: ManagerBasedRLEnv) -> torch.Tensor:
    """critic 전용: 단계 one-hot + 파지/접촉/들었음 + 비전 끊김 여부 + 책상 높이 오프셋 (num_envs, NUM_STAGES + 5)."""
    s = pick_place_state(env)
    one_hot = torch.nn.functional.one_hot(s.stage, NUM_STAGES).float()
    drop = _box_obs_dropout_mask(env)["drop"]
    flags = torch.stack((s.grasp, s.touch, s.ever_lifted, drop), dim=-1).float()
    return torch.cat((one_hot, flags, table_dz(env)[:, None] / TABLE_DZ_RANGE[1]), dim=-1)



def dr_privileged(env: ManagerBasedRLEnv, finger_cfg: SceneEntityCfg, nominal_stiffness: float, nominal_speed: float):
    """critic 전용 DR 값, 각각 범위로 나눠 대략 [-1, 1]: 큐브 pose 관측 오프셋 (위치 3, yaw 1), 손가락 강성·속도 배율 - 1,
    비전 지연 step. (num_envs, 7)"""
    bias = _cube_obs_bias(env)
    yaw = 2.0 * torch.atan2(bias["quat"][:, 2], bias["quat"][:, 3])
    robot, j = env.scene[finger_cfg.name], finger_cfg.joint_ids[0]
    k = robot.data.joint_stiffness.torch[:, j] / nominal_stiffness - 1.0
    v = robot.data.joint_vel_limits.torch[:, j] / nominal_speed - 1.0
    delay = _box_obs_dropout_mask(env)["delay"].float() / BOX_OBS_DELAY_STEPS
    scaled = (yaw / CUBE_OBS_BIAS_YAW, k / 0.2, v / 0.1, delay)
    return torch.cat((bias["pos"] / CUBE_OBS_BIAS_POS, torch.stack(scaled, dim=-1)), dim=-1)


# ---------- command ----------


class PickPlaceGoalCommand(UniformPoseCommand):
    """UniformPoseCommand + 놓는 곳을 오른팔 반경(reach_radius) 안, 큐브(집는 곳)에서 min_distance 이상으로.

    command manager는 reset event(큐브 위치) 다음에 reset되므로 여기서 보는 큐브는 새 에피소드 위치다.
    """

    cfg: PickPlaceGoalCommandCfg

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        obj = self._env.scene[self.cfg.object_name]
        cube_b = subtract_frame_transforms(
            self.robot.data.root_pos_w.torch[ids], self.robot.data.root_quat_w.torch[ids], obj.data.root_pos_w.torch[ids]
        )[0]
        center = torch.tensor(RIGHT_LINK0_XY_B, device=self.device)
        for _ in range(50):
            xy = self.pose_command_b[ids, :2]
            bad = torch.norm(xy - cube_b[:, :2], dim=-1) < self.cfg.min_distance
            if self.cfg.reach_radius is not None:
                bad |= torch.norm(xy - center, dim=-1) > self.cfg.reach_radius
            if not bool(bad.any()):
                break
            redo = ids[bad]
            r = torch.empty(len(redo), device=self.device)
            self.pose_command_b[redo, 0] = r.uniform_(*self.cfg.ranges.pos_x)
            self.pose_command_b[redo, 1] = r.uniform_(*self.cfg.ranges.pos_y)


@configclass
class PickPlaceGoalCommandCfg(UniformPoseCommandCfg):
    class_type: type = PickPlaceGoalCommand
    object_name: str = "object"
    min_distance: float = MISSING
    reach_radius: float | None = REACH_RADIUS


# ---------- events ----------


def sample_reachable_xy(k: int, x_range, y_range, device, reach_radius: float | None = REACH_RADIUS) -> torch.Tensor:
    """(x, y) 범위에서 균일하게 뽑되 오른팔 link0 축에서 reach_radius 밖이면 다시 뽑는다 (base). (k, 2)"""
    xy = torch.empty(k, 2, device=device)
    todo = torch.ones(k, dtype=torch.bool, device=device)
    center = torch.tensor(RIGHT_LINK0_XY_B, device=device)
    for _ in range(50):
        m = int(todo.sum())
        if m == 0:
            break
        xy[todo, 0] = torch.empty(m, device=device).uniform_(*x_range)
        xy[todo, 1] = torch.empty(m, device=device).uniform_(*y_range)
        if reach_radius is None:
            break
        todo = torch.norm(xy - center, dim=-1) > reach_radius
    return xy


def table_dz(env: ManagerBasedRLEnv) -> torch.Tensor:
    """env별 책상 윗면 높이 - 명목 높이 (m). reset_table_height가 정한다. (num_envs,)"""
    if not hasattr(env, "_table_dz"):
        env._table_dz = torch.zeros(env.num_envs, device=env.device)
    return env._table_dz


def reset_table_height(env: ManagerBasedRLEnv, env_ids: torch.Tensor, dz_range=TABLE_DZ_RANGE,
                       asset_cfg: SceneEntityCfg = SceneEntityCfg("table")):
    """책상 상판(kinematic)을 기본 위치에서 z로 dz만큼 옮긴다. reset_cube보다 먼저 와야 한다."""
    table = env.scene[asset_cfg.name]
    k = len(env_ids)
    dz = torch.empty(k, device=env.device).uniform_(*dz_range)
    table_dz(env)[env_ids] = dz
    pose = table.data.default_root_pose.torch[env_ids].clone()
    pose[:, :3] += env.scene.env_origins[env_ids]
    pose[:, 2] += dz
    table.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)


def reset_cube_upright(env: ManagerBasedRLEnv, env_ids: torch.Tensor, x_range, y_range, yaw_range,
                       reach_radius: float | None = REACH_RADIUS, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")):
    """큐브를 책상 위 (x, y) 범위 & 오른팔 반경 안에 세워 놓는다 (base frame, yaw 균일). 속도 0."""
    obj, robot = env.scene[asset_cfg.name], env.scene["robot"]
    k = len(env_ids)
    xy = sample_reachable_xy(k, x_range, y_range, env.device, reach_radius)
    pos_b = torch.cat((xy, (CUBE_REST_Z_B + 0.001 + table_dz(env)[env_ids])[:, None]), dim=-1)
    pos_w = combine_frame_transforms(robot.data.root_pos_w.torch[env_ids], robot.data.root_quat_w.torch[env_ids], pos_b)[0]
    yaw = torch.empty(k, device=env.device).uniform_(*yaw_range)
    quat = quat_from_angle_axis(yaw, torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(k, 1))
    obj.write_root_pose_to_sim_index(root_pose=torch.cat((pos_w, quat), dim=-1), env_ids=env_ids)
    obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(k, 6, device=env.device), env_ids=env_ids)


# ---------- curriculum ----------


def track_pick_place_progress(env: ManagerBasedRLEnv, env_ids) -> dict[str, float]:
    """끝난 에피소드의 단계 도달률을 EMA로 모아 TB에 남긴다 (Curriculum/<term>/reach_*)."""
    s = pick_place_state(env)
    if env_ids is not None and len(env_ids) > 0:
        max_stage = s.max_stage[env_ids]
        alpha = min(0.5, 3.0 * len(max_stage) / env.num_envs)
        for k, name in enumerate(STAGE_NAMES):
            if k > 0:
                key = f"reach_{name}"
                s.ema[key] += alpha * ((max_stage >= k).float().mean().item() - s.ema[key])
    return dict(s.ema)


def modify_reward_weight_on_placed(env: ManagerBasedRLEnv, env_ids, term_name: str, weight: float,
                                   threshold: float = 0.3, key: str = "reach_placed"):
    """EMA[key](track_pick_place_progress가 먼저 갱신)가 threshold를 넘으면 weight를 바꾼다 (latch)."""
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    if term_cfg.weight != weight and pick_place_state(env).ema[key] > threshold:
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)
    return term_cfg.weight
