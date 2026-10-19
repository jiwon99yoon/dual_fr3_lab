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

    @property
    def position_targets(self) -> torch.Tensor:
        """Return the position targets represented by the current action."""
        if self.cfg.reference_mode == "policy_step":
            return self._policy_step_targets
        current_positions = self._asset.data.joint_pos.torch[:, self._joint_ids]
        return current_positions + self.processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process one policy action and optionally snapshot its absolute target."""
        super().process_actions(actions)
        if self.cfg.reference_mode == "policy_step":
            current_positions = self._asset.data.joint_pos.torch[:, self._joint_ids]
            self._policy_step_targets[:] = current_positions + self.processed_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if self.cfg.reference_mode == "policy_step":
            resolved_env_ids = slice(None) if env_ids is None else env_ids
            current_positions = self._asset.data.joint_pos.torch[:, self._joint_ids]
            self._policy_step_targets[resolved_env_ids] = current_positions[resolved_env_ids]

    def apply_actions(self) -> None:
        if self.cfg.reference_mode == "physics_step":
            super().apply_actions()
            return
        self._asset.set_joint_position_target_index(
            target=self._policy_step_targets,
            joint_ids=self._joint_ids,
        )


@configclass
class ReachRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for :class:`ReachRelativeJointPositionAction`."""

    class_type: type[ReachRelativeJointPositionAction] = ReachRelativeJointPositionAction

    reference_mode: str = "physics_step"
    """Position-reference update rate: ``physics_step`` or ``policy_step``."""
