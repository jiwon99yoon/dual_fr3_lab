# Copyright (c) 2026, DYROS.
# SPDX-License-Identifier: BSD-3-Clause

"""Snapshot-relative joint position action for the 1-vs-3 action ablation.

Isaac Lab's :class:`RelativeJointPositionAction` computes ``q + delta`` every
physics step.  This term instead samples ``q`` once when the policy action is
received, then holds that absolute position target through every physics
substep until the next policy action.
"""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions import RelativeJointPositionAction, RelativeJointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


class SnapshotRelativeJointPositionAction(RelativeJointPositionAction):
    """Apply ``q_meas(t_k) + delta[k]`` as a held absolute position target.

    ``process_actions()`` runs once per RL policy step, so it snapshots the
    measured joint position at ``t_k``.  ``apply_actions()`` runs at every
    physics substep but only writes that already-sampled target.  Therefore
    this is the diagram's controller-1 semantics, while retaining the same
    14-D delta action, scale, actuator PD, and physics configuration as the
    existing controller-3 Reach task.
    """

    cfg: SnapshotRelativeJointPositionActionCfg

    def __init__(self, cfg: SnapshotRelativeJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._snapshot_joint_pos_target = torch.zeros(
            self.num_envs,
            self.action_dim,
            device=self.device,
            dtype=self._asset.data.joint_pos.dtype,
        )

    @property
    def snapshot_joint_pos_target(self) -> torch.Tensor:
        """The absolute target captured at the most recent policy tick."""
        return self._snapshot_joint_pos_target

    def process_actions(self, actions: torch.Tensor):
        """Process delta and capture the current measured joint position once."""
        super().process_actions(actions)
        self._snapshot_joint_pos_target[:] = (
            self.processed_actions + self._asset.data.joint_pos[:, self._joint_ids]
        )

    def apply_actions(self):
        """Hold the policy-tick snapshot; do not recompute from current q."""
        self._asset.set_joint_position_target(
            self._snapshot_joint_pos_target,
            joint_ids=self._joint_ids,
        )


@configclass
class SnapshotRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for :class:`SnapshotRelativeJointPositionAction`."""

    class_type: type[ActionTerm] = SnapshotRelativeJointPositionAction
