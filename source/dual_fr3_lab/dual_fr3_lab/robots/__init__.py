# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""dual_fr3_lab 로봇 설정 모음 (husky 베이스 + 양팔 FR3 계열)."""

from .dual_fr3 import DUAL_FR3_CFG, DUAL_FR3_HIGH_PD_CFG  # noqa: F401
from .dual_fr3_plate_ee import (  # noqa: F401
    DUAL_FR3_PLATE_EE_CFG,
    DUAL_FR3_PLATE_EE_HIGH_PD_CFG,
    DUAL_FR3_PLATE_EE_SIM2REAL_CFG,
    PLATE_TCP_OFFSET_POS,
)
