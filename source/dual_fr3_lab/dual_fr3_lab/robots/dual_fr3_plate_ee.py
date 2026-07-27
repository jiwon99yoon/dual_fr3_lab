# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the DYROS dual FR3 robot with flat-plate end-effectors.

The following configurations are available:

* :obj:`DUAL_FR3_PLATE_EE_CFG`: dual FR3 arms (gripper 없음) + 140x140x25mm 판 EE.
* :obj:`DUAL_FR3_PLATE_EE_HIGH_PD_CFG`: task-space(diff IK) 제어용 고게인 버전.

USD: dual_fr3_flat_no_handeye.usd(그리퍼 제거) 양쪽 link7 플랜지에
plate_ee_140x140x25.usd(볼트 인터페이스 유지, 박스 collider)를 부착한 것.
- 구동 joint 14개: left/right_fr3_joint1~7 (finger joint 없음)
- 판 collider는 link7 body 소속 -> 자기 링크와는 충돌하지 않고 외부 물체와만 접촉
- 판 작업면(TCP): link7 기준 z +0.132 (플랜지 0.107 + 판 두께 0.025)
"""

import os

from .dual_fr3 import DUAL_FR3_CFG

##
# Configuration
##

DUAL_FR3_PLATE_EE_USD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "dual_fr3_plate_ee.usd"
)

# link7 -> 판 작업면 중심 오프셋 (판은 대칭이라 회전 오프셋 불필요)
PLATE_TCP_OFFSET_POS = (0.0, 0.0, 0.132)

DUAL_FR3_PLATE_EE_CFG = DUAL_FR3_CFG.copy()
DUAL_FR3_PLATE_EE_CFG.spawn.usd_path = DUAL_FR3_PLATE_EE_USD_PATH
# finger joint이 없으므로 hands 액추에이터/초기값 제거
del DUAL_FR3_PLATE_EE_CFG.actuators["hands"]
del DUAL_FR3_PLATE_EE_CFG.init_state.joint_pos[".*_fr3_finger_joint.*"]
"""Dual FR3 arms with flat-plate end-effectors on a husky base (base fixed)."""


DUAL_FR3_PLATE_EE_HIGH_PD_CFG = DUAL_FR3_PLATE_EE_CFG.copy()
DUAL_FR3_PLATE_EE_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
DUAL_FR3_PLATE_EE_HIGH_PD_CFG.actuators["arms_shoulder"].stiffness = 400.0
DUAL_FR3_PLATE_EE_HIGH_PD_CFG.actuators["arms_shoulder"].damping = 80.0
DUAL_FR3_PLATE_EE_HIGH_PD_CFG.actuators["arms_forearm"].stiffness = 400.0
DUAL_FR3_PLATE_EE_HIGH_PD_CFG.actuators["arms_forearm"].damping = 80.0
"""Stiffer PD control. task-space (differential IK) 제어용."""
