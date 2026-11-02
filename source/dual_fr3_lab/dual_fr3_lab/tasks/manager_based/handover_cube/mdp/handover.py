# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""큐브 핸드오버 MDP (2026-10-25).

오른팔이 테이블 오른쪽 끝의 큐브(5x5x10 cm, cube lift v3 실물)를 집어 중앙 핸드오버 지점으로 옮기고,
왼팔이 받아 왼쪽 끝 목표에 세워 놓는다. 보상은 단계 판정(HandoverState) 하나에 모은다:

    progress = STAGE_BASE[s] + shaping_s,   shaping_s in [0, 1]

s는 지금 상태에서 만족하는 가장 높은 단계다. 다음 단계의 base가 이전 단계 최댓값보다 커서 단계를 넘으면
보상이 항상 오른다. 단계별 항을 따로 더하면 "오른손을 놓는 순간 운반 보상이 끊겨 안 놓는" 함정이 생긴다
(box Level에서 gate가 lifting 보상을 깎아 커리큘럼이 안 켜진 것과 같은 종류).

    0 REACH       오른 TCP -> 오른쪽 파지점 (+ 테이블 위 오른손 파지 0.5)
    1 LIFTED_R    오른손 파지 & 들림. 큐브 -> 핸드오버 지점
    2 AT_HANDOVER 위 + 핸드오버 지점 HANDOVER_TOL 이내. 왼 TCP -> 왼쪽 파지점
    3 BOTH        왼손 파지 & 오른손 접촉 & 들림
    4 LEFT_ONLY   왼손 파지 & 오른손 비접촉 & (들림 | handed). 들린 채 처음 되면 handed 기록. 큐브 -> 목표
    5 PLACED      넘겨받은 뒤 목표 GOAL_XY_TOL 이내에 세워 놓음 & 왼손 접촉
    6 RELEASED    위 + 양손 비접촉 (성공)

v1 (2026-11-02): env cfg의 handover_zone이 있으면 단계 3·4 진입(넘겨받기)을 핸드오버 지점 그 반경 안에서만
인정한다. v0(None)은 위치 조건이 없어서 gatedC 학습 중 넘겨받는 위치가 목표 쪽으로 30 cm 밀려났다
(오른팔이 중앙선을 넘어 운반하고 왼손은 내려놓기만 함).

시작 상태 커리큘럼: 자연 시작(A)이 핸드오버 지점까지 오는 것을 배우면 일부 env를 "왼손이 쥔 상태(C)"에서
시작시키고, A 성공률이 오르면 비율을 줄인다 (anneal_start_modes). "오른손이 쥔 상태(B)"도 있지만 기본 0이다.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

# ---------- 배치 (env 원점 기준 world, lift_cube와 같은 MJCF 테이블) ----------
TABLE_TOP_Z = 0.75
CUBE_SIZE = (0.05, 0.05, 0.10)  # 실물 cube v3 (50x50x100 실측)
CUBE_REST_Z = TABLE_TOP_Z + CUBE_SIZE[2] / 2  # 세워 놓였을 때 중심 높이
CUBE_SPAWN_POS = (0.45, -0.35, CUBE_REST_Z)  # 테이블 오른쪽 끝 (reset에서 x, y ±0.05)
HANDOVER_POS = (0.50, 0.0, TABLE_TOP_Z + 0.20)  # 중앙, 상판 위 20 cm (큐브 중심 기준)

LIFT_DELTA = 0.04  # 들림 판정: 중심이 세워 놓인 높이 + 4 cm 위 (cube lift와 같은 +4 cm)
HANDOVER_TOL = 0.05
GOAL_XY_TOL = 0.03
REST_TOL = 0.01  # 중심이 CUBE_REST_Z + 1 cm 아래면 테이블에 놓인 것 (누운 큐브도 포함)
UPRIGHT_COS = math.cos(math.radians(15.0))
GRASP_FORCE = 0.5  # N, 양 패드 모두 이 이상이면 파지 (cube lift와 같음)
DROP_GOAL_RADIUS = 0.10  # 넘겨받은 뒤 목표 10 cm 안에서 놓인 것은 낙하로 보지 않음

# 파지점: 큐브 z축 방향 오프셋. 오른손은 위쪽(위에서 집기), 왼손은 아래쪽(옆에서 받기)을 잡아야 손이 안 겹친다
RIGHT_GRASP_OFFSET = 0.02
LEFT_GRASP_OFFSET = -0.025
FINGER_OPEN = 0.04

STAGE_NAMES = ("reach", "lifted_r", "at_handover", "both", "left_only", "placed", "released")
NUM_STAGES = len(STAGE_NAMES)
STAGE_SUCCESS = NUM_STAGES - 1
STAGE_HANDED = STAGE_NAMES.index("left_only")
STAGE_BASE = (0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
REACH_STD = 0.1
TRANSPORT_STD = 0.3

# 센서/프레임 이름 (env cfg와 공유)
TCP_FRAME = "tcp_frame"
PAD_SENSORS = {
    "left": ("left_leftfinger_contact", "left_rightfinger_contact"),
    "right": ("right_leftfinger_contact", "right_rightfinger_contact"),
}
GOAL_COMMAND = "goal_pose"

# 시작 모드: 0 = 자연 시작(테이블 위), 1 = 오른손이 핸드오버 지점에서 쥠, 2 = 왼손이 쥠
START_NATURAL, START_RIGHT_HOLD, START_LEFT_HOLD = 0, 1, 2
# 쥔 자세 (joint1~7, 해당 팔만) + 그때의 큐브 pose (env 원점 기준, pos + quat xyzw). None이면 그 모드를 안 쓴다.
# sim 안 DLS IK로 구함 (2026-10-25): 오른손은 ready처럼 위에서(손가락 축 world y), 왼손은 +y쪽에서 옆으로
# (접근 -y, 손가락 축 world x). TCP 오차 ~1 mm, datasheet 한계 여유 ≥0.88 rad. 16 env에서 두 모드 모두
# 2 s 동안 양 패드 파지가 유지됨을 확인 (action 0이면 measured-relative 제어라 팔이 큐브 무게로 처진다)
RIGHT_HOLD_Q: tuple[float, ...] | None = (0.7276, 0.0026, 0.2023, -1.4534, -0.0005, 1.4562, 1.1916)
RIGHT_HOLD_CUBE_POSE: tuple[float, ...] | None = HANDOVER_POS + (0.0, 0.0, 0.0, 1.0)
LEFT_HOLD_Q: tuple[float, ...] | None = (-0.5973, -0.131, 0.0596, -2.1652, -1.5702, 1.5873, -0.3216)
LEFT_HOLD_CUBE_POSE: tuple[float, ...] | None = HANDOVER_POS + (0.0, 0.0, 0.0, 1.0)
HOLD_FINGER_POS = CUBE_SIZE[0] / 2 + 0.0005  # 큐브 폭보다 0.5 mm씩 벌린 채 놓고 target 0으로 조인다


class HandoverState:
    """env당 핸드오버 단계와 기억(handed, ever_lifted)을 스텝마다 한 번 계산해 모든 term이 공유한다.

    termination -> reward -> (reset) -> observation 순으로 같은 스텝에 여러 번 불리므로
    env.common_step_counter로 한 번만 갱신한다. reset 직후에는 강제로 다시 계산한다.
    """

    def __init__(self, env: ManagerBasedRLEnv):
        self.env = env
        n, dev = env.num_envs, env.device
        self.stage = torch.zeros(n, dtype=torch.long, device=dev)
        self.max_stage = torch.zeros_like(self.stage)
        self.start_mode = torch.zeros_like(self.stage)
        self.handed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_lifted = torch.zeros_like(self.handed)
        self.shaping = torch.zeros(n, device=dev)
        self.start_probs = torch.tensor([1.0, 0.0, 0.0], device=dev)
        self.assist_on = False  # 쥔 상태 시작 사용 여부 (anneal_start_modes가 latch)
        # 커리큘럼 판정용 EMA (자연 시작 에피소드 기준)
        self.ema = {"success": 0.0, "handover": 0.0}
        self.ema.update({f"reach_{name}": 0.0 for name in STAGE_NAMES[1:]})
        self._ez = torch.tensor([0.0, 0.0, 1.0], device=dev).repeat(n, 1)
        self._handover_pos = torch.tensor(HANDOVER_POS, device=dev)
        self._zone = getattr(env.cfg, "handover_zone", None)
        self._base = torch.tensor(STAGE_BASE, device=dev)
        names = env.scene[TCP_FRAME].data.target_frame_names
        self._li, self._ri = names.index("left_tcp"), names.index("right_tcp")
        self._step = -1
        self.update()

    def reset(self, env_ids):
        for buf in (self.stage, self.max_stage):
            buf[env_ids] = 0
        for buf in (self.handed, self.ever_lifted):
            buf[env_ids] = False
        self._step = -1

    def _pad_forces(self, side: str) -> tuple[torch.Tensor, torch.Tensor]:
        """(min, max) 패드-큐브 접촉력. 패드 센서 filter[0]=Object."""
        forces = []
        for name in PAD_SENSORS[side]:
            fm = self.env.scene.sensors[name].data.force_matrix_w.torch[:, :, 0, :]
            forces.append(torch.norm(fm, dim=-1).amax(dim=-1))
        return torch.minimum(*forces), torch.maximum(*forces)

    def update(self):
        env = self.env
        if self._step == env.common_step_counter:
            return
        self._step = env.common_step_counter
        origins = env.scene.env_origins
        obj = env.scene["object"]
        robot = env.scene["robot"]

        cube = obj.data.root_pos_w.torch - origins
        cube_z = quat_apply(obj.data.root_quat_w.torch, self._ez)
        tcp = env.scene[TCP_FRAME].data.target_pos_w.torch - origins[:, None, :]
        self.tcp_l, self.tcp_r = tcp[:, self._li], tcp[:, self._ri]
        self.cube_pos, self.cube_z = cube, cube_z

        min_l, max_l = self._pad_forces("left")
        min_r, max_r = self._pad_forces("right")
        self.grasp_l, self.touch_l = min_l > GRASP_FORCE, max_l > GRASP_FORCE
        self.grasp_r, self.touch_r = min_r > GRASP_FORCE, max_r > GRASP_FORCE

        goal_b = env.command_manager.get_command(GOAL_COMMAND)[:, :3]
        goal = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, goal_b)[0]
        goal = goal - origins
        self.d_goal = torch.norm(cube - goal, dim=-1)
        self.d_goal_xy = torch.norm(cube[:, :2] - goal[:, :2], dim=-1)
        self.d_handover = torch.norm(cube - self._handover_pos, dim=-1)
        d_r = torch.norm(self.tcp_r - (cube + cube_z * RIGHT_GRASP_OFFSET), dim=-1)
        d_l = torch.norm(self.tcp_l - (cube + cube_z * LEFT_GRASP_OFFSET), dim=-1)

        lifted = cube[:, 2] > CUBE_REST_Z + LIFT_DELTA
        self.resting = cube[:, 2] < CUBE_REST_Z + REST_TOL
        upright = cube_z[:, 2] > UPRIGHT_COS
        self.lifted = lifted
        held_r = self.grasp_r & lifted
        near_ho = self.d_handover < self._zone if self._zone is not None else torch.ones_like(lifted)
        left_only = self.grasp_l & ~self.touch_r & lifted & near_ho
        # reset 직후(물리 step 전)에는 접촉 센서가 이전 에피소드 값이라 기억과 최고 단계를 갱신하지 않는다
        # (안 막으면 B 시작 env가 이전 에피소드의 왼손 파지로 handed가 켜져 핸드오버 없이 성공할 수 있다)
        fresh = env.episode_length_buf > 0
        self.handed |= left_only & fresh
        self.ever_lifted |= lifted & (self.grasp_l | self.grasp_r) & fresh
        at_goal = self.handed & (self.d_goal_xy < GOAL_XY_TOL) & self.resting & upright & ~self.touch_r

        # 단계 사이에 빈 상태가 없게 접촉(touch)과 파지(grasp)를 나눠 쓴다. 빈 상태가 있으면 그 사이를 지날 때
        # 단계 0으로 떨어져서 정책이 그 전이를 피한다:
        #   3: 왼손 파지 & 오른손이 아직 닿음 (오른 패드 하나만 남아도 3)
        #   4: 넘겨받은 뒤에는 높이·위치와 무관 (내려놓는 중 +1~+4 cm 구간도 4)
        #   5: 놓는 중 왼 패드 하나만 남아도 5
        conditions = (
            held_r,
            held_r & (self.d_handover < HANDOVER_TOL),
            self.grasp_l & self.touch_r & lifted & near_ho,
            self.grasp_l & ~self.touch_r & (left_only | self.handed),
            at_goal & self.touch_l,
            at_goal & ~self.touch_l,
        )
        stage = torch.zeros_like(self.stage)
        for s, cond in enumerate(conditions, start=1):
            stage = torch.where(cond, s, stage)

        half = torch.full_like(d_r, 0.5)
        shaping = torch.stack(
            (
                0.5 * (1.0 - torch.tanh(d_r / REACH_STD)) + 0.5 * self.grasp_r.float(),
                1.0 - torch.tanh(self.d_handover / TRANSPORT_STD),
                1.0 - torch.tanh(d_l / REACH_STD),
                half,  # 양손 파지 -> 오른손 놓기: binary 그리퍼라 한 번에 넘어간다
                1.0 - torch.tanh(self.d_goal / TRANSPORT_STD),
                half,  # 놓기
                torch.ones_like(d_r),
            ),
            dim=-1,
        )
        self.stage = stage
        self.max_stage = torch.where(fresh, torch.maximum(self.max_stage, stage), self.max_stage)
        self.shaping = shaping.gather(1, stage[:, None]).squeeze(1)
        self.progress = self._base[stage] + self.shaping


def handover_state(env: ManagerBasedRLEnv) -> HandoverState:
    """env에 하나뿐인 HandoverState (처음 부르는 manager가 만든다: 보통 observation manager)."""
    if not hasattr(env, "_handover_state"):
        env._handover_state = HandoverState(env)
    state = env._handover_state
    state.update()
    return state


# ---------- rewards ----------


class handover_progress(ManagerTermBase):
    """STAGE_BASE[s] + shaping_s. 이 term의 reset이 에피소드 기억(handed 등)을 지운다."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._state = handover_state(env)

    def reset(self, env_ids=None):
        self._state.reset(slice(None) if env_ids is None else env_ids)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        return handover_state(env).progress


def handover_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """성공 단계(양손 놓고 목표에 세워 둠) 지표. 작은 weight로 TB 기록용."""
    return (handover_state(env).stage == STAGE_SUCCESS).float()


def tcp_midline_crossing(env: ManagerBasedRLEnv, margin: float) -> torch.Tensor:
    """각 팔 TCP가 중앙선(y 0)에서 margin을 넘어 상대 팔 쪽으로 들어간 거리 합 (m)."""
    s = handover_state(env)
    return torch.relu(s.tcp_r[:, 1] - margin) + torch.relu(-margin - s.tcp_l[:, 1])


def cube_tilt_while_held(env: ManagerBasedRLEnv) -> torch.Tensor:
    """들린 큐브를 어느 손이든 쥐고 있을 때 큐브 z축 기울기 (rad). 목표에 세워 놓아야 하므로 운반 중에도 세운다."""
    s = handover_state(env)
    tilt = torch.acos(s.cube_z[:, 2].clamp(-1.0, 1.0))
    return tilt * (s.lifted & (s.grasp_l | s.grasp_r)).float()


# ---------- terminations ----------


def cube_dropped_midway(env: ManagerBasedRLEnv) -> torch.Tensor:
    """한 번 들린 큐브가 어느 손에도 안 쥐인 채 테이블에 놓이면 종료 (테이블에 내려놓고 다시 집기 방지).

    넘겨받은 뒤 목표 DROP_GOAL_RADIUS 안에 놓인 것은 놓기 과정이라 제외한다.
    """
    s = handover_state(env)
    near_goal = s.handed & (s.d_goal_xy < DROP_GOAL_RADIUS)
    return s.ever_lifted & s.resting & ~s.grasp_l & ~s.grasp_r & ~near_goal


# ---------- observations ----------


def cube_pos_rel_tcps(env: ManagerBasedRLEnv) -> torch.Tensor:
    """큐브 중심 - (왼 TCP, 오른 TCP), world 축 (base frame은 world와 회전이 같다). (num_envs, 6)"""
    s = handover_state(env)
    return torch.cat((s.cube_pos - s.tcp_l, s.cube_pos - s.tcp_r), dim=-1)


def goal_position(env: ManagerBasedRLEnv, command_name: str = GOAL_COMMAND) -> torch.Tensor:
    """목표 위치 (base frame). 목표 자세는 항상 세움이라 quat은 뺀다."""
    return env.command_manager.get_command(command_name)[:, :3]


def handover_privileged(env: ManagerBasedRLEnv) -> torch.Tensor:
    """critic 전용: 단계 one-hot + handed + 양손 파지/접촉 (num_envs, NUM_STAGES + 5)."""
    s = handover_state(env)
    one_hot = torch.nn.functional.one_hot(s.stage, NUM_STAGES).float()
    flags = torch.stack((s.handed, s.grasp_l, s.grasp_r, s.touch_l, s.touch_r), dim=-1).float()
    return torch.cat((one_hot, flags), dim=-1)


# ---------- events ----------


def _hold_modes():
    modes = {}
    if RIGHT_HOLD_Q is not None:
        modes[START_RIGHT_HOLD] = ("right", RIGHT_HOLD_Q, RIGHT_HOLD_CUBE_POSE)
    if LEFT_HOLD_Q is not None:
        modes[START_LEFT_HOLD] = ("left", LEFT_HOLD_Q, LEFT_HOLD_CUBE_POSE)
    return modes


def reset_start_state(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """시작 모드를 뽑고, B/C면 그 팔을 쥔 자세로 두고 큐브를 손가락 사이에 놓는다 (reset_all 뒤에 실행).

    쥔 팔의 PD 목표도 그 자세로 맞춘다. 그리퍼 목표는 닫힘(0)이라 첫 정책 step 전에 조여지고,
    그 뒤로는 정책의 binary 그리퍼 action이 덮어쓴다.
    """
    s = handover_state(env)
    modes = torch.multinomial(s.start_probs.expand(len(env_ids), -1), 1).squeeze(1)
    s.start_mode[env_ids] = modes
    robot, obj = env.scene["robot"], env.scene["object"]
    for mode, (side, q_hold, cube_pose) in _hold_modes().items():
        ids = env_ids[modes == mode]
        if len(ids) == 0:
            continue
        arm_ids = robot.find_joints([f"{side}_fr3_joint{j}" for j in range(1, 8)], preserve_order=True)[0]
        finger_ids = robot.find_joints(f"{side}_fr3_finger_joint.*")[0]
        joint_ids = arm_ids + finger_ids
        q = torch.tensor(list(q_hold) + [HOLD_FINGER_POS] * len(finger_ids), device=env.device).repeat(len(ids), 1)
        target = q.clone()
        target[:, len(arm_ids):] = 0.0
        robot.write_joint_position_to_sim_index(position=q, joint_ids=joint_ids, env_ids=ids)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=joint_ids, env_ids=ids)
        robot.set_joint_position_target_index(target=target, joint_ids=joint_ids, env_ids=ids)
        pose = torch.tensor(cube_pose, device=env.device).repeat(len(ids), 1)
        pose[:, :3] += env.scene.env_origins[ids]
        obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=env.device), env_ids=ids)


# ---------- curriculum ----------


def anneal_start_modes(
    env: ManagerBasedRLEnv,
    env_ids,
    p_right_hold: float = 0.0,
    p_left_hold: float = 0.25,
    enable_key: str = "reach_at_handover",
    enable_threshold: float = 0.5,
    success_ref: float = 0.5,
    min_ratio: float = 0.2,
) -> dict[str, float]:
    """끝난 자연 시작 에피소드의 단계 도달률을 EMA로 모아 기록하고, 시작 모드 확률을 정한다.

    쥔 상태 시작(B/C)은 자연 시작이 EMA[enable_key] > enable_threshold가 된 뒤에만 켜고(latch), 그 뒤
    p = p_* x max(min_ratio, 1 - ema_success / success_ref)로 줄인다. 쥔 자세가 None인 모드는 0이다.
    파일럿(2026-10-25, 300 iter): 처음부터 B/C를 섞으면(각 25% 또는 10%) 자연 시작이 한 번도 못 집었다
    (B/C에서 배운 "그리퍼 닫고 버티기"가 자연 시작에도 나옴). 자연 시작만이면 같은 기간에 99%가 들었다.
    B(오른손이 핸드오버 지점에서 쥠)는 자연 시작이 이미 도달하는 상태라 기본 0이다.
    EMA 가중치는 modify_reward_weight_on_lifting_learned와 같은 배치 크기 가중이다.
    반환한 dict는 Curriculum/<term>/<key>로 TB에 남는다.
    """
    s = handover_state(env)
    if env_ids is not None and len(env_ids) > 0:
        natural = s.start_mode[env_ids] == START_NATURAL
        max_stage = s.max_stage[env_ids][natural]
        if len(max_stage) > 0:
            alpha = min(0.5, 3.0 * len(max_stage) / env.num_envs)
            samples = {"success": max_stage >= STAGE_SUCCESS, "handover": max_stage >= STAGE_HANDED}
            samples.update({f"reach_{name}": max_stage >= k for k, name in enumerate(STAGE_NAMES) if k > 0})
            for key, hit in samples.items():
                s.ema[key] += alpha * (hit.float().mean().item() - s.ema[key])
    s.assist_on = s.assist_on or s.ema[enable_key] > enable_threshold
    ratio = max(min_ratio, 1.0 - s.ema["success"] / success_ref) if s.assist_on else 0.0
    hold_modes = _hold_modes()
    p_b = p_right_hold * ratio if START_RIGHT_HOLD in hold_modes else 0.0
    p_c = p_left_hold * ratio if START_LEFT_HOLD in hold_modes else 0.0
    s.start_probs = torch.tensor([1.0 - p_b - p_c, p_b, p_c], device=env.device)
    return {**s.ema, "assist_on": float(s.assist_on), "p_right_hold": p_b, "p_left_hold": p_c}


def modify_reward_weight_on_handover_learned(
    env: ManagerBasedRLEnv, env_ids, term_name: str, weight: float, threshold: float = 0.3
):
    """자연 시작 에피소드의 핸드오버 도달률 EMA가 threshold를 넘으면 weight를 바꾼다 (latch).

    anneal_start_modes가 같은 reset에서 먼저 EMA를 갱신해야 한다 (curriculum cfg에서 앞에 둔다).
    """
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    if term_cfg.weight != weight and handover_state(env).ema["handover"] > threshold:
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)
    return term_cfg.weight


def undesired_contacts_any(env: ManagerBasedRLEnv, threshold: float, sensor_cfgs: list[SceneEntityCfg]) -> torch.Tensor:
    """여러 센서의 threshold 초과 접촉 body 수 합 (양팔 link1~7)."""
    total = torch.zeros(env.num_envs, device=env.device)
    for cfg in sensor_cfgs:
        forces = env.scene.sensors[cfg.name].data.net_forces_w_history.torch
        total += (torch.norm(forces, dim=-1).amax(dim=1) > threshold).sum(dim=-1)
    return total
