# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""dual_fr3_lab 로봇 설정 모음 (husky 베이스 + 양팔 FR3 계열)."""

from .dual_fr3 import (  # noqa: F401
    DUAL_FR3_CFG,
    DUAL_FR3_HIGH_PD_CFG,
    FR3_ARM_MUJOCO_PASSIVE_DYNAMICS,
    with_arm_passive_dynamics,
    with_torque_rate_limited_arms,
)
from .dual_fr3_plate_ee import (  # noqa: F401
    DUAL_FR3_PLATE_EE_CFG,
    DUAL_FR3_PLATE_EE_HIGH_PD_CFG,
    DUAL_FR3_PLATE_EE_SIM2REAL_100HZ_FRICTION_CFG,
    DUAL_FR3_PLATE_EE_SIM2REAL_100HZ_NO_FRICTION_CFG,
    DUAL_FR3_PLATE_EE_SIM2REAL_1KHZ_FRICTION_CFG,
    DUAL_FR3_PLATE_EE_SIM2REAL_1KHZ_NO_FRICTION_CFG,
    DUAL_FR3_PLATE_EE_SIM2REAL_CFG,
    DUAL_FR3_PLATE_EE_SIM2REAL_V2_CFG,
    PLATE_TCP_OFFSET_POS,
)
from .dual_fr3_umi import (  # noqa: F401
    DUAL_FR3_UMI_CFG,
    DUAL_FR3_UMI_SIM2REAL_100HZ_CFG,
    UMI_TCP_OFFSET_POS,
    UMI_TCP_OFFSET_ROT,
)
