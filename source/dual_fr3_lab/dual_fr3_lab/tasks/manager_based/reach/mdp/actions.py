# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Reach-specific joint action terms."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.utils.configclass import configclass


class ReachRelativeJointPositionAction(RelativeJointPositionAction):
    r"""Relative joint action with a selectable position-reference update rate.

    ``physics_step`` preserves Isaac Lab's standard relative-action behavior::

        q_target(t) = q_measured(t) + delta_q[k]

    ``policy_step`` snapshots the measured position when the policy action is
    processed and holds the resulting absolute target throughout decimation::

        q_target[k] = q_measured(t_k) + delta_q[k]
    """

    cfg: ReachRelativeJointPositionActionCfg

    def __init__(self, cfg: ReachRelativeJointPositionActionCfg, env):
        super().__init__(cfg, env)
        if cfg.reference_mode not in ("physics_step", "policy_step"):
            raise ValueError(
                "Reach relative-action reference_mode must be 'physics_step' or "
                f"'policy_step', got {cfg.reference_mode!r}."
            )
        self._policy_step_targets = torch.zeros_like(self.processed_actions)
        self._target_limits = None
        if cfg.target_limits is not None:
            self._target_limits = torch.tensor(cfg.target_limits, device=self.device)

    def _clamp_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if self._target_limits is None:
            return targets
        return torch.clamp(targets, self._target_limits[:, 0], self._target_limits[:, 1])

    @property
    def position_targets(self) -> torch.Tensor:
        """Return the position targets represented by the current action."""
        if self.cfg.reference_mode == "policy_step":
            return self._policy_step_targets
        current_positions = self._asset.data.joint_pos.torch[:, self._joint_ids]
        return self._clamp_targets(current_positions + self.processed_actions)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process one policy action and optionally snapshot its absolute target."""
        super().process_actions(actions)
        if self.cfg.reference_mode == "policy_step":
            current_positions = self._asset.data.joint_pos.torch[:, self._joint_ids]
            self._policy_step_targets[:] = self._clamp_targets(current_positions + self.processed_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if self.cfg.reference_mode == "policy_step":
            resolved_env_ids = slice(None) if env_ids is None else env_ids
            current_positions = self._asset.data.joint_pos.torch[:, self._joint_ids]
            self._policy_step_targets[resolved_env_ids] = current_positions[resolved_env_ids]

    def apply_actions(self) -> None:
        if self.cfg.reference_mode == "physics_step" and self._target_limits is None:
            super().apply_actions()
            return
        self._asset.set_joint_position_target_index(
            target=self.position_targets,
            joint_ids=self._joint_ids,
        )


@configclass
class ReachRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for :class:`ReachRelativeJointPositionAction`."""

    class_type: type[ReachRelativeJointPositionAction] = ReachRelativeJointPositionAction

    reference_mode: str = "physics_step"
    """Position-reference update rate: ``physics_step`` or ``policy_step``."""

    target_limits: tuple[tuple[float, float], ...] | None = None
    """Per-joint (lower, upper) clamp on the absolute target, in ``joint_names`` order. None = no clamp.

    The deployment node has to apply the same clamp, since the real controller rejects targets beyond its limits.
    """


class DelayedReachRelativeJointPositionAction(ReachRelativeJointPositionAction):
    r"""``policy_step`` relative joint action with joint-state and command latency, sampled per env at reset.

    On the robot the node reads ``q`` that is already ``d_obs`` old, sends ``q_target = q(t_k - d_obs) + delta_q[k]``,
    and the controller starts tracking it ``d_cmd`` after ``t_k``. Here:

    - ``obs_delay``: per env, uniform in ``obs_delay_range_s``. The reference is the first-order delayed measurement
      ``q(t_k) - qd(t_k) * d_obs``. Observation terms that should see the same delayed ``q`` read :attr:`obs_delay`.
    - ``cmd_delay``: per env, integer physics steps in ``cmd_delay_steps_range`` (inclusive). The previous target is
      held for that many physics steps of each policy window, then the new one.
    """

    cfg: DelayedReachRelativeJointPositionActionCfg

    def __init__(self, cfg: DelayedReachRelativeJointPositionActionCfg, env):
        super().__init__(cfg, env)
        if cfg.reference_mode != "policy_step":
            raise ValueError("DelayedReachRelativeJointPositionAction needs reference_mode='policy_step'.")
        self.obs_delay = torch.zeros(self.num_envs, device=self.device)
        self.cmd_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_targets = torch.zeros_like(self._policy_step_targets)
        self._substep = 0
        self._sample_delays(slice(None))

    def _sample_delays(self, env_ids) -> None:
        n = self.num_envs if isinstance(env_ids, slice) else len(env_ids)
        lo, hi = self.cfg.obs_delay_range_s
        self.obs_delay[env_ids] = lo + (hi - lo) * torch.rand(n, device=self.device)
        lo, hi = self.cfg.cmd_delay_steps_range
        self.cmd_delay[env_ids] = torch.randint(lo, hi + 1, (n,), device=self.device)

    def delayed_joint_pos(self) -> torch.Tensor:
        """First-order delayed joint positions of this term's joints (num_envs, num_joints)."""
        q = self._asset.data.joint_pos.torch[:, self._joint_ids]
        qd = self._asset.data.joint_vel.torch[:, self._joint_ids]
        return q - qd * self.obs_delay.unsqueeze(-1)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._prev_targets[:] = self._policy_step_targets
        super().process_actions(actions)
        self._policy_step_targets[:] = self._clamp_targets(self.delayed_joint_pos() + self.processed_actions)
        self._substep = 0

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        resolved_env_ids = slice(None) if env_ids is None else env_ids
        self._prev_targets[resolved_env_ids] = self._policy_step_targets[resolved_env_ids]
        self._sample_delays(resolved_env_ids)

    def apply_actions(self) -> None:
        hold = (self.cmd_delay > self._substep).unsqueeze(-1)
        self._asset.set_joint_position_target_index(
            target=torch.where(hold, self._prev_targets, self._policy_step_targets),
            joint_ids=self._joint_ids,
        )
        self._substep += 1


@configclass
class DelayedReachRelativeJointPositionActionCfg(ReachRelativeJointPositionActionCfg):
    """Configuration for :class:`DelayedReachRelativeJointPositionAction`."""

    class_type: type[DelayedReachRelativeJointPositionAction] = DelayedReachRelativeJointPositionAction

    reference_mode: str = "policy_step"

    obs_delay_range_s: tuple[float, float] = (0.0, 0.0)
    """Joint-state age (s) per env, applied to the reference and to observation terms that read :attr:`obs_delay`."""

    cmd_delay_steps_range: tuple[int, int] = (0, 0)
    """Command latency in physics steps per env (inclusive): the previous target is held this long each window."""
