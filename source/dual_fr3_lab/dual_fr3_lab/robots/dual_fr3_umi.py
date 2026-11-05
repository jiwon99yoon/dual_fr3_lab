# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the DYROS dual FR3 robot with UMI fingertips on the franka hands.

The following configurations are available:

* :obj:`DUAL_FR3_UMI_CFG`: dual FR3 arms + franka hands with UMI tips on a (static) husky base.
* :obj:`DUAL_FR3_UMI_SIM2REAL_100HZ_CFG`: 100Hz, plate Sim2Real-100Hz와 같은 팔 계약(explicit PD + 토크 변화율 제한
  + MuJoCo-matched passive dynamics). 손가락은 stiff PD로 task가 정한 폭에 고정한다.

USD: fr3_husky `devel` description(`dual_fr3_husky.urdf.xacro`, umi_tip:=true, use_d435i:=false, with_sc:=false)
+ franka_description `954350f`(UMI tip). 만든 절차는 content/make_usd.md의 "UMI 그리퍼 USD".
- 구동 joint 18개: left/right_fr3_joint1~7, left/right_fr3_finger_joint1~2 (기존 DUAL_FR3_CFG와 같은 이름)
- root body는 `base`(팔 장착 상판, 두 팔 base의 중점)로 다시 잡았다. 기존 USD·실로봇 base frame과 같다.
- 손가락 visual/collision은 UMI_tip.stl. 손가락 1개당 0~0.04 m(양손가락 폭 0~0.08 m)
- TCP(손가락 끝 중앙): hand 기준 z +0.1774 (기존 franka 손가락 0.1034), link7 기준은 UMI_TCP_OFFSET_POS/ROT
"""

import os

from .dual_fr3 import DUAL_FR3_CFG, with_torque_rate_limited_arms

##
# Configuration
##

DUAL_FR3_UMI_USD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "dual_fr3_umi.usd"
)

# link7 -> UMI 손가락 끝 중앙(hand_tcp). hand는 merge-joints로 link7에 병합되어 link7 기준으로 둔다.
# URDF: link7 -(z 0.107)-> link8 -(yaw −45°)-> hand -(z 0.1774, fr3_husky devel tcp_xyz)-> hand_tcp
UMI_TCP_OFFSET_POS = (0.0, 0.0, 0.2844)
UMI_TCP_OFFSET_ROT = (0.0, 0.0, -0.3826834, 0.9238795)  # (x, y, z, w) = Isaac Lab 3 순서, z축 −45°

DUAL_FR3_UMI_CFG = DUAL_FR3_CFG.copy()
DUAL_FR3_UMI_CFG.spawn.usd_path = DUAL_FR3_UMI_USD_PATH
# plate 계열과 같이 로봇 링크 중력 OFF = 실물 FR3 내장 중력보상. 손에 든 물체 무게는 그대로 전달된다.
DUAL_FR3_UMI_CFG.spawn.rigid_props.disable_gravity = True
"""Dual FR3 arms with UMI-tip franka hands on a husky base (base fixed)."""

DUAL_FR3_UMI_SIM2REAL_100HZ_CFG = with_torque_rate_limited_arms(
    DUAL_FR3_UMI_CFG, control_dt=0.01, match_mujoco_passive_dynamics=True
)
"""100Hz arm contract shared with lift_plate Sim2Real-100Hz. Fingers keep the stiff PD of ``DUAL_FR3_CFG``."""
