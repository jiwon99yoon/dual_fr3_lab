# Copyright (c) 2026, DYROS.
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit PD actuator with libfranka-style torque magnitude/rate limiting."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.types import ArticulationActions


class TorqueRateLimitedPDActuator(IdealPDActuator):
    """PD actuator that clips torque magnitude and then its per-step change.

    The rate limiter mirrors the ordering used by the real deployment path:
    PolicyControl first clips the desired effort to the FR3 effort limit,
    and franka_hardware then applies franka::kMaxTorqueRate.
    """

    cfg: "TorqueRateLimitedPDActuatorCfg"

    def __init__(self, cfg: "TorqueRateLimitedPDActuatorCfg", *args, **kwargs):
        if cfg.torque_rate_limit <= 0.0:
            raise ValueError("torque_rate_limit must be positive")
        if cfg.control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        super().__init__(cfg, *args, **kwargs)
        self._max_effort_step = float(cfg.torque_rate_limit * cfg.control_dt)
        self._previous_applied_effort = torch.zeros_like(self.applied_effort)

    def reset(self, env_ids: Sequence[int] | slice | torch.Tensor):
        self._previous_applied_effort[env_ids] = 0.0

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        self.computed_effort = (
            self.stiffness * error_pos
            + self.damping * error_vel
            + control_action.joint_efforts
        )

        magnitude_limited_effort = self._clip_effort(self.computed_effort)
        effort_delta = torch.clamp(
            magnitude_limited_effort - self._previous_applied_effort,
            min=-self._max_effort_step,
            max=self._max_effort_step,
        )
        self.applied_effort = self._previous_applied_effort + effort_delta
        self._previous_applied_effort.copy_(self.applied_effort)

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class TorqueRateLimitedPDActuatorCfg(IdealPDActuatorCfg):
    """Configuration for TorqueRateLimitedPDActuator."""

    class_type: type[TorqueRateLimitedPDActuator] = TorqueRateLimitedPDActuator
    torque_rate_limit: float = 1000.0
    """Maximum absolute torque change rate in Nm/s (FR3/libfranka: 1000)."""
    control_dt: float = 0.01
    """Actuator update period in seconds; must match the simulation physics dt."""

