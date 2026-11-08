# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""실물 Franka Hand 타이밍을 넣은 binary 그리퍼 action과 손가락 폭 관측 (2026-11-08).

실측 (fr3_husky_etri data/gripper_latency/20261108_*, docs/dual_fr3_liftcube_sim2real_experi_step.md "측정 결과",
오른손 빈 손가락, 팔 정지, 경우마다 10~30회):

- 명령 -> 손가락이 움직이기 시작: 0.65 s (0.45~0.9). grasp/Move, 닫기/열기, 속도 0.05/0.1 모두 같다
- 그 뒤 명령 속도 그대로 (0.1 m/s = 두 손가락 폭 합). 속도 제한은 task가 hands actuator의 velocity_limit_sim으로 둔다
- 손가락이 멈춘 뒤 action result까지 0.3~0.4 s. 다음 명령은 앞 명령이 끝날 때까지 기다린다 (11-07 세션 220138)
- 손가락 폭 관측: /franka_gripper/joint_states는 10 Hz지만 값은 약 0.27 s(p95 0.38 s)마다만 바뀐다
- 조이는 힘: 실물 노드는 grasp force 30 N (fr3_husky_etri config/dual_fr3_lift_sim2real100hz.yaml grasp_force_n)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

GRIPPER_DELAY_RANGE_S = (0.45, 0.9)  # 명령 -> 움직임 시작
GRIPPER_SETTLE_RANGE_S = (0.3, 0.4)  # 멈춤 -> result (다음 명령을 받기까지)
FINGER_OBS_UPDATE_RANGE_S = (0.25, 0.38)  # 손가락 폭 관측 갱신 간격
GRIPPER_GRASP_FORCE_N = 30.0  # 실물 grasp force: 손가락 하나가 물체를 미는 힘
# DR (env마다 startup에 고정): 실물 힘 제어 정확도·손가락 속도를 모르는 만큼. 힘은 손가락 PD 강성 배율로 25~35 N
GRIPPER_FORCE_SCALE_RANGE = (25.0 / GRIPPER_GRASP_FORCE_N, 35.0 / GRIPPER_GRASP_FORCE_N)
GRIPPER_SPEED_SCALE_RANGE = (0.9, 1.1)

IDLE, DELAY, MOVE, SETTLE = range(4)


class DelayedBinaryJointPositionAction(BinaryJointPositionAction):
    """Binary 그리퍼 + 실물 명령 처리: 명령이 바뀌면 지연 뒤 움직이고, 멈춘 뒤 settle까지 다음 명령을 대기시킨다.

    env마다 IDLE -> DELAY(명령마다 지연을 새로 뽑음) -> MOVE(손가락이 멈출 때까지) -> SETTLE -> IDLE.
    바쁜 동안 들어온 명령은 마지막 것만 남고, IDLE이 되면 그것이 실행 중인 명령과 다를 때 다시 보낸다.
    """

    cfg: DelayedBinaryJointPositionActionCfg

    def __init__(self, cfg: DelayedBinaryJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        n = self.num_envs
        self._request_close = torch.zeros(n, dtype=torch.bool, device=self.device)  # 정책의 마지막 명령
        self._sent_close = torch.zeros_like(self._request_close)  # 보낸(처리 중인) 명령
        self._target_close = torch.zeros_like(self._request_close)  # 손가락 PD 목표
        self._phase = torch.full((n,), IDLE, dtype=torch.long, device=self.device)
        self._timer = torch.zeros(n, device=self.device)  # DELAY/SETTLE: 남은 시간, MOVE: 지난 시간
        self._prev_q = self._asset.data.joint_pos.torch[:, self._joint_ids].clone()  # 지난 policy step의 손가락 위치
        self._processed_actions = self._open_command.repeat(n, 1)

    @property
    def settled(self) -> torch.Tensor:
        """IDLE이고 마지막 명령까지 실행이 끝남 (실물의 action result를 받은 상태). (num_envs,)"""
        return (self._phase == IDLE) & (self._target_close == self._request_close)

    def _sample(self, k: int, lo_hi: tuple[float, float]) -> torch.Tensor:
        return torch.empty(k, device=self.device).uniform_(*lo_hi)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._request_close = (actions == 0) if actions.dtype == torch.bool else (actions < 0)
        self._request_close = self._request_close.reshape(-1)
        dt = self._env.step_dt
        ph = self._phase

        # DELAY -> MOVE: 보낸 명령을 손가락 목표로
        self._timer = torch.where(ph == DELAY, self._timer - dt, self._timer)
        start = (ph == DELAY) & (self._timer <= 0.0)
        self._target_close = torch.where(start, self._sent_close, self._target_close)
        ph = torch.where(start, MOVE, ph)
        self._timer = torch.where(start, 0.0, self._timer)

        # MOVE -> SETTLE: 손가락이 목표에 닿았거나 물체에 막혀 멈춤. 속도는 policy step 사이 위치 변화로 잰다
        # (물체를 쥔 손가락은 위치가 그대로여도 PhysX joint_vel이 0.02~0.05 m/s로 나온다)
        q = self._asset.data.joint_pos.torch[:, self._joint_ids]
        vel = (q - self._prev_q).abs().amax(-1) / dt
        self._prev_q = q.clone()
        moving = ph == MOVE
        self._timer = torch.where(moving, self._timer + dt, self._timer)
        stop = moving & (self._timer >= self.cfg.min_move_s) & (vel < self.cfg.stop_speed)
        ph = torch.where(stop, SETTLE, ph)
        self._timer = torch.where(stop, self._sample(self.num_envs, self.cfg.settle_range_s), self._timer)

        # SETTLE -> IDLE
        settling = ph == SETTLE
        self._timer = torch.where(settling & ~stop, self._timer - dt, self._timer)
        done = settling & (self._timer <= 0.0)
        ph = torch.where(done, IDLE, ph)

        # IDLE에서 마지막 명령이 실행 중인 것과 다르면 보낸다
        send = (ph == IDLE) & (self._request_close != self._target_close)
        self._sent_close = torch.where(send, self._request_close, self._sent_close)
        ph = torch.where(send, DELAY, ph)
        self._timer = torch.where(send, self._sample(self.num_envs, self.cfg.delay_range_s), self._timer)

        self._phase = ph
        self._processed_actions = torch.where(self._target_close[:, None], self._close_command, self._open_command)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        for buf in (self._request_close, self._sent_close, self._target_close):
            buf[ids] = False  # reset 자세는 손가락 열림
        self._phase[ids] = IDLE
        self._timer[ids] = 0.0
        self._processed_actions[ids] = self._open_command
        self._prev_q[ids] = self._open_command


@configclass
class DelayedBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    class_type: type = DelayedBinaryJointPositionAction
    delay_range_s: tuple[float, float] = GRIPPER_DELAY_RANGE_S
    settle_range_s: tuple[float, float] = GRIPPER_SETTLE_RANGE_S
    stop_speed: float = 0.005  # m/s, 손가락 위치가 policy step 사이 이보다 느리게 바뀌면 멈춘 것
    min_move_s: float = 0.1  # MOVE 시작 직후(아직 가속 전)의 정지를 멈춤으로 보지 않게


class finger_pos_held(ManagerTermBase):
    """손가락 관절 위치를 update_range_s마다만 갱신하고 사이에는 마지막 값을 준다 (실물 폭 갱신 ~0.27 s).

    갱신 시각의 위상은 에피소드마다 무작위이고, 에피소드 첫 관측은 현재 값이다.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._held: torch.Tensor | None = None
        self._next_s = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        update_range_s: tuple[float, float] = FINGER_OBS_UPDATE_RANGE_S,
    ) -> torch.Tensor:
        value = env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids]
        if self._held is None:
            self._held = value.clone()
        t = env.episode_length_buf * env.step_dt
        fresh = env.episode_length_buf == 0
        n = env.num_envs
        lo, hi = update_range_s
        first = torch.rand(n, device=env.device) * hi  # 첫 갱신까지: 무작위 위상
        self._next_s = torch.where(fresh, first, self._next_s)
        update = fresh | (t >= self._next_s)
        self._held = torch.where(update[:, None], value, self._held)
        step = torch.empty(n, device=env.device).uniform_(lo, hi)
        self._next_s = torch.where(update & ~fresh, self._next_s + step, self._next_s)
        return self._held.clone()


def randomize_finger_speed(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    speed: float,
    scale_range: tuple[float, float] = GRIPPER_SPEED_SCALE_RANGE,
):
    """손가락 관절 속도 한계(PhysX)를 env마다 speed x U(scale_range)로 바꾼다. 같은 env의 손가락끼리는 같은 값."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    scale = torch.empty(len(env_ids), 1, device=asset.device).uniform_(*scale_range)
    limits = (speed * scale).expand(-1, len(asset_cfg.joint_ids)).contiguous()
    asset.write_joint_velocity_limit_to_sim_index(limits=limits, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def gripper_busy(env: ManagerBasedRLEnv, action_name: str = "gripper_action") -> torch.Tensor:
    """그리퍼 명령이 아직 안 끝남 1 / 끝남 0 (settled의 반대). 실물: 노드가 보낸 goal의 result를 아직 못 받음. (num_envs, 1)"""
    term: DelayedBinaryJointPositionAction = env.action_manager.get_term(action_name)
    return (~term.settled).float().unsqueeze(-1)


def gripper_privileged(env: ManagerBasedRLEnv, action_name: str = "gripper_action") -> torch.Tensor:
    """critic 전용: 그리퍼 단계 one-hot (IDLE/DELAY/MOVE/SETTLE), 단계 타이머 (s), 요청·보냄·목표 닫힘 여부. (num_envs, 8)"""
    term: DelayedBinaryJointPositionAction = env.action_manager.get_term(action_name)
    one_hot = torch.nn.functional.one_hot(term._phase, 4).float()
    bits = torch.stack((term._request_close, term._sent_close, term._target_close), dim=-1).float()
    return torch.cat((one_hot, term._timer.unsqueeze(-1), bits), dim=-1)
