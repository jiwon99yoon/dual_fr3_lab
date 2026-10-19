# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the DYROS dual FR3 (on husky base) robot.

The following configurations are available:

* :obj:`DUAL_FR3_CFG`: dual FR3 arms + franka hands on a (static) husky base.

USD 생성 절차는 /home/dyros/fr3_ws/make_usd.md 참고.
- 구동 joint 18개: left/right_fr3_joint1~7, left/right_fr3_finger_joint1~2
- husky/휠/카메라 링크는 merge-joints로 base에 병합된 정적 mesh
- joint4/joint6은 가동범위에 0이 없으므로 init_state 필수
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

# 패키지 동봉 에셋 (생성 파이프라인: repo의 content/make_usd.md, 빌드 소스는 fr3_ws)
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
DUAL_FR3_USD_PATH = os.path.join(_ASSETS_DIR, "dual_fr3_no_handeye.usd")

# -----------------------------------------------------------------------------
# Arm controller and plant contract
# -----------------------------------------------------------------------------
#
# The first two constants are the joint-side PD gains used by the real
# PolicyControl path.  They are controller parameters -- in particular,
# ``FR3_ARM_KD_NM_S_PER_RAD`` is *not* the physical viscous-friction value.
FR3_ARM_KP_NM_PER_RAD = 80.0
FR3_ARM_KD_NM_S_PER_RAD = 4.0
FR3_SHOULDER_EFFORT_LIMIT_NM = 87.0
FR3_FOREARM_EFFORT_LIMIT_NM = 12.0
# This limiter belongs to the real control path, not to an implicit actuator.
# SysID runners apply it explicitly after the PD calculation.
FR3_TORQUE_RATE_LIMIT_NM_S = 1000.0

# Identified joint-side plant parameters used by the current dual-FR3
# sim-to-real baseline.  These are independent of whether the arm controller
# is an implicit PhysX drive or an explicit/custom torque controller:
#
#   [M(q) + A] qdd + C(q, qd) + B qd + tau_dry = tau_applied.
#
# Isaac Lab authors these fields to PhysX joint properties during articulation
# construction.  Keep the units explicit: the friction value is the Isaac Sim
# 5.0+ joint static-friction effort used by the existing SysID recorder.
FR3_ARMATURE_KG_M2 = 0.1
FR3_DRY_FRICTION_NM = 0.2
FR3_VISCOUS_FRICTION_NM_S_PER_RAD = 0.003

DUAL_FR3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=DUAL_FR3_USD_PATH,
        # finger pad ContactSensor(파지 판정)를 위해 contact reporting 활성화
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            # USD의 root(/fr3/base)는 floating이므로 스폰 위치에 고정한다
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # root frame(base)은 팔 장착 상판 기준이고 husky/휠이 z=-0.405까지 내려오므로
        # 바퀴가 지면에 닿도록 올려서 스폰한다
        pos=(0.0, 0.0, 0.405),
        joint_pos={
            # SRDF의 "ready" group_state 기반 (joint4, joint6은 0이 가동범위 밖)
            # joint1은 팔 장착 yaw(±30도)를 보상해서 양팔이 정면을 향하게 한다
            "left_fr3_joint1": -0.5236,  # -pi/6
            "right_fr3_joint1": 0.5236,  # +pi/6
            ".*_fr3_joint2": -0.7854,  # -pi/4
            ".*_fr3_joint3": 0.0,
            ".*_fr3_joint4": -2.3562,  # -3*pi/4
            ".*_fr3_joint5": 0.0,
            ".*_fr3_joint6": 1.5708,  # pi/2
            ".*_fr3_joint7": 0.7854,  # pi/4
            ".*_fr3_finger_joint.*": 0.04,
        },
    ),
    actuators={
        # 게인은 공식 FRANKA_PANDA_CFG와 동일, effort limit은 fr3 joint_limits.yaml 기준
        "arms_shoulder": ImplicitActuatorCfg(
            joint_names_expr=[".*_fr3_joint[1-4]"],
            effort_limit_sim=FR3_SHOULDER_EFFORT_LIMIT_NM,
            stiffness=FR3_ARM_KP_NM_PER_RAD,
            damping=FR3_ARM_KD_NM_S_PER_RAD,
            armature=FR3_ARMATURE_KG_M2,
            friction=FR3_DRY_FRICTION_NM,
            viscous_friction=FR3_VISCOUS_FRICTION_NM_S_PER_RAD,
        ),
        "arms_forearm": ImplicitActuatorCfg(
            joint_names_expr=[".*_fr3_joint[5-7]"],
            effort_limit_sim=FR3_FOREARM_EFFORT_LIMIT_NM,
            stiffness=FR3_ARM_KP_NM_PER_RAD,
            damping=FR3_ARM_KD_NM_S_PER_RAD,
            armature=FR3_ARMATURE_KG_M2,
            friction=FR3_DRY_FRICTION_NM,
            viscous_friction=FR3_VISCOUS_FRICTION_NM_S_PER_RAD,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[".*_fr3_finger_joint.*"],
            effort_limit_sim=200.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of dual FR3 arms on a husky base (base fixed)."""


DUAL_FR3_HIGH_PD_CFG = DUAL_FR3_CFG.copy()
DUAL_FR3_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
DUAL_FR3_HIGH_PD_CFG.actuators["arms_shoulder"].stiffness = 400.0
DUAL_FR3_HIGH_PD_CFG.actuators["arms_shoulder"].damping = 80.0
DUAL_FR3_HIGH_PD_CFG.actuators["arms_forearm"].stiffness = 400.0
DUAL_FR3_HIGH_PD_CFG.actuators["arms_forearm"].damping = 80.0
"""Stiffer PD control. task-space (differential IK) 제어용."""
