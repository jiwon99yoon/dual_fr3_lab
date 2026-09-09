# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the DYROS dual FR3 robot with flat-plate end-effectors.

The following configurations are available:

* :obj:`DUAL_FR3_PLATE_EE_CFG`: dual FR3 arms (gripper 없음) + 140x140x25mm 판 EE.
* :obj:`DUAL_FR3_PLATE_EE_SIM2REAL_CFG`: reach v1, 100Hz explicit PD.
* :obj:`DUAL_FR3_PLATE_EE_SIM2REAL_V2_CFG`: reach v2, 1kHz explicit PD.
* :obj:`DUAL_FR3_PLATE_EE_HIGH_PD_CFG`: task-space(diff IK) 제어용 고게인 버전.

USD: dual_fr3_flat_no_handeye.usd(그리퍼 제거) 양쪽 link7 플랜지에
plate_ee_140x140x25.usd(볼트 인터페이스 유지, 박스 collider)를 부착한 것.
- 구동 joint 14개: left/right_fr3_joint1~7 (finger joint 없음)
- 판 collider는 link7 body 소속 -> 자기 링크와는 충돌하지 않고 외부 물체와만 접촉
- 판 작업면(TCP): link7 기준 z +0.132 (플랜지 0.107 + 판 두께 0.025)
"""

import os

from dual_fr3_lab.actuators import TorqueRateLimitedPDActuatorCfg

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
# 로봇 링크 중력 OFF = 실물 FR3 내장 중력보상 g(q) 모델링 (payload/박스 무게는 그대로 전달됨).
# Run P6 A/B에서 중력보상 쪽이 전 지표 우세 (reward 140.8 vs 130.6, alignment 3.8배 등)로 기본값 채택.
# [잠정 결정 — sim 검증 단계 한정] 아직 sim에서만 검증된 선택이므로, sim2real 실로봇 실험 전까지만
# 기본값으로 사용한다. SR(성공률)과 전체 파이프라인이 어느 정도 안정된 뒤에는 a(중력 ON)/b(중력 OFF)
# 둘 다 학습해 실로봇 transfer 성능을 비교하고, 실물에서 더 나은 쪽으로 최종 고정할 것.
DUAL_FR3_PLATE_EE_CFG.spawn.rigid_props.disable_gravity = True
"""Dual FR3 arms with flat-plate end-effectors on a husky base (base fixed)."""


DUAL_FR3_PLATE_EE_SIM2REAL_CFG = DUAL_FR3_PLATE_EE_CFG.copy()
DUAL_FR3_PLATE_EE_SIM2REAL_CFG.actuators["arms_shoulder"] = TorqueRateLimitedPDActuatorCfg(
    joint_names_expr=[".*_fr3_joint[1-4]"],
    effort_limit=87.0,
    effort_limit_sim=87.0,
    stiffness=80.0,
    damping=4.0,
    armature=0.1,
    friction=0.2,          # static friction effort [Nm]
    dynamic_friction=0.2,  # dynamic friction effort [Nm]
    viscous_friction=0.003, # [Nm·s/rad]
    torque_rate_limit=1000.0,
    control_dt=0.01,
)
DUAL_FR3_PLATE_EE_SIM2REAL_CFG.actuators["arms_forearm"] = TorqueRateLimitedPDActuatorCfg(
    joint_names_expr=[".*_fr3_joint[5-7]"],
    effort_limit=12.0,
    effort_limit_sim=12.0,
    stiffness=80.0,
    damping=4.0,
    armature=0.1,
    friction=0.2,          # static friction effort [Nm]
    dynamic_friction=0.2,  # dynamic friction effort [Nm]
    viscous_friction=0.003, # [Nm·s/rad]
    torque_rate_limit=1000.0,
    control_dt=0.01,
)
"""Reach Sim2Real v1: explicit PD + FR3 effort and torque-rate limits."""


DUAL_FR3_PLATE_EE_SIM2REAL_V2_CFG = DUAL_FR3_PLATE_EE_SIM2REAL_CFG.copy()
DUAL_FR3_PLATE_EE_SIM2REAL_V2_CFG.actuators["arms_shoulder"].control_dt = 0.001
DUAL_FR3_PLATE_EE_SIM2REAL_V2_CFG.actuators["arms_forearm"].control_dt = 0.001
"""Reach Sim2Real v2: 1kHz explicit PD with a 1Nm/update torque-rate limit."""


DUAL_FR3_PLATE_EE_HIGH_PD_CFG = DUAL_FR3_PLATE_EE_CFG.copy()
DUAL_FR3_PLATE_EE_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
DUAL_FR3_PLATE_EE_HIGH_PD_CFG.actuators["arms_shoulder"].stiffness = 400.0
DUAL_FR3_PLATE_EE_HIGH_PD_CFG.actuators["arms_shoulder"].damping = 80.0
DUAL_FR3_PLATE_EE_HIGH_PD_CFG.actuators["arms_forearm"].stiffness = 400.0
DUAL_FR3_PLATE_EE_HIGH_PD_CFG.actuators["arms_forearm"].damping = 80.0
"""Stiffer PD control. task-space (differential IK) 제어용."""
