# Copyright (c) 2026, DYROS.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom actuator models used by dual_fr3_lab sim-to-real tasks."""

from .torque_rate_limited_pd import TorqueRateLimitedPDActuator, TorqueRateLimitedPDActuatorCfg

__all__ = ["TorqueRateLimitedPDActuator", "TorqueRateLimitedPDActuatorCfg"]

