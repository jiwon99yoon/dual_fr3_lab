# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dual FR3 큐브 pick-and-place (오른팔, Franka hand + UMI 팁, 75 cm 책상) v0 (2026-11-07).

책상 위에 세워 놓인 큐브(5x5x10 cm, yaw ±45°, 70~100 g)를 오른손으로 위에서 집어 무작위 목표 위치에 세워 놓고,
손을 떼고 올린다. 단계 판정/보상은 mdp/pick_place.py. 왼팔은 ready 자세로 둔다(action 없음).

- 로봇: DUAL_FR3_UMI_SIM2REAL_100HZ_CFG (캐비닛 책 넣기와 같은 손, 같은 100 Hz 팔 계약). TCP = UMI 손가락 끝 중앙
- 책상: 실물 600(x) x 1000(y), 앞 모서리 base x 0.40, 윗면 base +0.345 (캐비닛 Table75와 같은 위치·높이)
- 제어: Reach Sim2Real v2 계약 (apply_sim2real_control: 100 Hz 물리 / 20 Hz 정책, Δq = clip(0.1·a, ±0.1 rad))
  + 오른손 binary 그리퍼. 실물 타이밍 (11-08 실측, mdp/delayed_gripper.py): 명령 지연 0.45~0.9 s, 손가락 한 개
  0.05 m/s, 멈춘 뒤 0.3~0.4 s 동안 다음 명령 대기, 손가락 폭 관측은 0.25~0.38 s마다 갱신,
  조이는 힘 30 N (손가락 PD 강성으로 맞춤. 전에는 5 cm 큐브를 손가락당 47 N으로 조였다)
- 큐브 관측: box BoxDrop과 같은 비전 지연·끊김(0~150 ms, 끊기면 마지막 값 유지), critic은 실제 pose
- DR (11-08 추가, 레퍼런스 비교 후): 큐브 pose 관측에 에피소드 고정 오프셋(위치 ±5 mm, yaw ±3°, 손-눈 보정 오차),
  조이는 힘 25~35 N·손가락 속도 ±10% (env마다 고정), 시작 오른팔 관절 ±0.05 rad, 책상 높이 ±1 cm (목표 관측 z는 명목값).
  원래 있던 것: 큐브 마찰 0.4~1.2·질량 70~100 g, 팔 PD 게인 ±15%, 관측 잡음, 그리퍼 명령 지연(명령마다)
- 성공: 목표 xy 2.5 cm (11-08, 전에는 1 cm), 기울기 5°, 손 뗌, 손 올림
- 집는 곳·놓는 곳: x 0.45~0.65, y -0.45~0.05 & 오른팔 link0 축에서 0.62 m 안 (mdp.REACH_RADIUS)

rule-based 확인 (scripts/tools/eval_dual_fr3_pick_place_rule_based.py, 256 env, 실제 큐브 pose):
- 11-07 (그리퍼 지연 없음, 0.6 s 대기): 249/256 성공, 성공 시각 중앙값 7.2 s. 실패 7개는 손을 펼 때 큐브가
  0.7~1.1 cm 밀려 1 cm를 넘은 경우 (들고 옮길 때 큐브가 손 안에서 최대 10° 기운다)
- 11-08 (지연 그리퍼, 그리퍼 명령이 끝날 때까지 대기): 251/256, 성공 시각 중앙값 9.1 s, p90 10.3 s, 최대 11.8 s.
  명령 -> 끝 닫기 1.25~1.75 s, 열기 1.35~1.80 s. 실패 5개 중 4개는 12 s 안에 못 끝냄, 1개는 놓을 때 1.3 cm 밀림
- 11-08 조임 30 N (손가락 강성 1282 N/m) + 14 s: 253/256, 들고 있을 때 패드 힘 30.4 N, 손가락 23.7 / 24.3 mm,
  성공 시각 중앙값 9.2 s, 최대 12.4 s. 실패 3개: 2개는 손을 펼 때 1.0~1.5 cm 밀림, 1개는 14 s 안에 못 놓음.
  같은 14 s에 강성 2000 (패드 힘 46.8 N)은 254/256
- 11-08 성공 xy 2.5 cm: 255/256 (최종 xy 오차 최대 1.5 cm). 실패 1개는 스크립트가 큐브 4 mm 안에 들 때까지 내리기를
  계속하다 시간이 다 된 경우 (그때 오차 0.4 cm)
- 11-08 DR 추가 후 (2.5 cm): 254/256. 책상 dz -0.98~+0.99 cm, 조이는 힘 25.1~35.0 N, 손가락 45~55 mm/s, 시작 관절 ±0.05 rad
  적용 확인. 실패 2개는 스크립트 한계: 4 mm 안까지 내리다 시간 초과 1개, yaw -44° 큐브의 파지 자세가 관절 한계 근처(0.27 rad)라
  스크립트 IK의 한계 회피 null-space가 새어 TCP가 3 cm 못 미친 채 닫은 1개
- 11-09 파지 높이 (정책과 같은 큐브 관측: 오프셋+잡음+지연을 다가가는 동안 평균, 오차 중앙값 4.2 mm): 손가락 끝 책상 위
  7 cm 251, 5 cm 255, 4 cm 254, 2.3 cm 232. 닫을 때 기울기 p90 2.8 / 1.0 / 0.9 / 0.3°, 손가락-책상 접촉 0 / 7 / 36 / 95 env
  -> 파지점을 큐브 중심(5 cm)으로. 천천히 다가가 멈춘 뒤 닫으면 어느 높이든 쥔 순간 기울기 2° 이하
4096 env 학습 3 iteration: 5070 GPU 최대 4.9 GB, 1.5~1.7 s/iteration (DR 추가 후 1.6~2.0 s, critic 23D).

11-09 수정 (첫 학습들의 결과, docs/dual_fr3_pick_place_cube.md): 쥔 채 들기 높이 shaping, 파지점 큐브 중심, 손가락-책상
접촉·손가락 끝 하한 벌점, PLACED에서 열기 요청 shaping, 한 번 놓았으면 RELEASED 유지 범위(5 cm, 15°)

11-09 수정 2 (5070 fix_5070 model_800, 4090 B 1423 it): PLACED·RELEASED shaping에 그리퍼 행동값 자체(연속, 0/1 요청 대신.
PLACED에서도 행동 -0.72라 열기가 step당 0.1%), PLACED에 큐브 정지(ManiSkill StackCube, 늘 0이던 손가락 벌림 대신. 열기 지연
동안 팔이 움직여 큐브가 끌려감), RELEASED 손 올리기를 큐브 바닥 높이부터 (쥔 깊이만큼 기울기 0이었음)
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise
from isaaclab_physx.physics import PhysxCfg

from . import mdp

##
# Pre-defined configs
##
from dual_fr3_lab.robots import DUAL_FR3_UMI_SIM2REAL_100HZ_CFG  # isort: skip
from dual_fr3_lab.tasks.manager_based.cabinet_book_insert.cabinet_book_insert_env_cfg import arm_link_path  # isort: skip
from dual_fr3_lab.tasks.manager_based.handover_cube.handover_env_cfg import GRIPPER_FINGER_SPEED  # isort: skip
from dual_fr3_lab.tasks.manager_based.lift_cube.joint_pos_env_cfg import apply_sim2real_control  # isort: skip
from dual_fr3_lab.tasks.manager_based.lift_plate.joint_pos_env_cfg import (  # isort: skip
    FR3_JOINT_POS_LIMITS,
    fr3_joint_limit_margin,
    set_fr3_datasheet_joint_limits,
)
from dual_fr3_lab.tasks.manager_based.reach.joint_pos_env_cfg import DUAL_ARM_JOINT_NAMES  # isort: skip

# 20 Hz x 14 s = 280 step. rule-based(0.25 m/s, 지연 그리퍼) 성공 시각 중앙값 9.1 s, p90 10.3 s, 최대 11.8 s라
# 12 s에서는 늦은 env가 잘렸다 (11-08)
EPISODE_LENGTH_S = 14.0
PROGRESS_WEIGHT = 5.0  # 단계 보상 상한 (8 + 1) x 5 = 45/s (handover와 같음)
# 손가락 PD: 닫기 목표 0에서 5 cm 큐브를 쥐면 손가락 q = 23.4 mm (rule-based 실측, 강성 2000에서 패드 힘 46.7 N =
# 2000 x 0.0234). 강성을 낮춰 이때 패드 힘이 실물 grasp force 30 N이 되게 한다. effort 상한으로는 힘이 상한을 따르지
# 않고 (상한 10/20/30 N -> 8/22/41 N) 두 손가락이 같이 포화되면 큐브가 옆으로 밀린다 (상한 20 N: 손가락 12 / 37 mm)
FINGER_GRASP_Q = 0.0234
FINGER_STIFFNESS = mdp.GRIPPER_GRASP_FORCE_N / FINGER_GRASP_Q
LEARNED_THRESHOLD = 0.3  # 끝난 에피소드의 placed 도달률 EMA가 넘으면 페널티 강화
RIGHT_ARM_JOINT_NAMES = [f"right_fr3_joint{i}" for i in range(1, 8)]
# 팔 링크 접촉 (손가락 제외, link7은 hand가 병합된 body). 왼팔은 ready에 서 있어서 오른팔과 닿을 수 있는 쪽만
ARM_CONTACT_LINKS = {"right": ("link3", "link4", "link5", "link6", "link7"), "left": ("link5", "link6", "link7")}
ARM_CONTACT_SENSORS = [f"{side}_{link}_contact" for side, links in ARM_CONTACT_LINKS.items() for link in links]

_TABLE_CX = 0.5 * (mdp.TABLE_X_B[0] + mdp.TABLE_X_B[1])
_TABLE_DX = mdp.TABLE_X_B[1] - mdp.TABLE_X_B[0]
_TABLE_H = mdp.TABLE_TOP_Z_B + mdp.ROBOT_BASE_Z  # 바닥에서 윗면까지


def _b2w(p):
    return (p[0], p[1], p[2] + mdp.ROBOT_BASE_Z)


def _box(name: str, size, pos_b, color, collide: bool = True):
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        init_state=AssetBaseCfg.InitialStateCfg(pos=_b2w(pos_b)),
        spawn=sim_utils.MeshCuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg() if collide else None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
    )


def _right_arm_joints():
    return SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES, preserve_order=True)


def _both_arm_joints():
    return SceneEntityCfg("robot", joint_names=DUAL_ARM_JOINT_NAMES, preserve_order=True)


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    robot = DUAL_FR3_UMI_SIM2REAL_100HZ_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        # Lab 3.0 quat은 (x, y, z, w). 위치·yaw는 reset_cube가 정한다
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_b2w((0.55, -0.20, mdp.CUBE_REST_Z_B + 0.001)), rot=[0, 0, 0, 1]
        ),
        spawn=sim_utils.MeshCuboidCfg(
            size=mdp.CUBE_SIZE,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.09),  # 실측 90 g (DR 70~100 g)
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.3, 0.85)),
        ),
    )

    # 책상: 상판 3 cm, 다리 2개 (캐비닛 Table75와 같은 높이). 상판은 높이 DR(reset_table)로 env마다 움직이게 kinematic
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_b2w((_TABLE_CX, 0.0, mdp.TABLE_TOP_Z_B - 0.015))),
        spawn=sim_utils.MeshCuboidCfg(
            size=(_TABLE_DX, mdp.TABLE_WIDTH, 0.03),
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.6, 0.4)),
        ),
    )
    table_leg_l = _box(
        "TableLegL", (0.04, 0.04, _TABLE_H - 0.03),
        (_TABLE_CX, 0.5 * mdp.TABLE_WIDTH - 0.05, mdp.TABLE_TOP_Z_B - 0.03 - 0.5 * (_TABLE_H - 0.03)), (0.8, 0.8, 0.8),
    )
    table_leg_r = _box(
        "TableLegR", (0.04, 0.04, _TABLE_H - 0.03),
        (_TABLE_CX, -0.5 * mdp.TABLE_WIDTH + 0.05, mdp.TABLE_TOP_Z_B - 0.03 - 0.5 * (_TABLE_H - 0.03)), (0.8, 0.8, 0.8),
    )
    # 시각 전용 (충돌체 없음): 큐브 중심이 놓이는 범위. DR_VIS=0이면 뺀다
    cube_zone = _box(
        "CubeZone",
        (mdp.CUBE_X_B[1] - mdp.CUBE_X_B[0], mdp.CUBE_Y_B[1] - mdp.CUBE_Y_B[0], 0.002),
        (0.5 * sum(mdp.CUBE_X_B), 0.5 * sum(mdp.CUBE_Y_B), mdp.TABLE_TOP_Z_B + 0.001),
        (0.75, 0.75, 0.9),
        collide=False,
    )

    # 오른손 손가락 접촉력. filter[0] = Object (파지 판정), filter[1] = Table (손가락-책상 접촉 벌점, 11-09)
    right_leftfinger_contact = ContactSensorCfg(
        prim_path=arm_link_path("right", "leftfinger"), filter_prim_paths_expr=["{ENV_REGEX_NS}/Object", "{ENV_REGEX_NS}/Table"]
    )
    right_rightfinger_contact = ContactSensorCfg(
        prim_path=arm_link_path("right", "rightfinger"), filter_prim_paths_expr=["{ENV_REGEX_NS}/Object", "{ENV_REGEX_NS}/Table"]
    )

    plane = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    # 놓을 곳: 책상 위에 세워 놓인 큐브 중심 (base frame). 에피소드당 1개, 집는 곳에서 GOAL_MIN_DIST 이상
    goal_pose = mdp.PickPlaceGoalCommandCfg(
        asset_name="robot",
        body_name="right_fr3_link7",
        # 에피소드보다 길게: 목표는 reset 때만 뽑는다. (14, 14)이면 자동 reset 뒤 같은 step에 command가 한 번 더 줄어서, 두 번째
        # 에피소드부터 시간 초과 한 step 전에 목표가 바뀌었다 -> 놓고 손 뗀 큐브가 마지막 step에 "목표 밖 낙하"(-100), 단계 0
        # (11-09 fix_5070 학습 로그 cube_dropped 0.87·time_out 0.8 동시, 새로 시작한 평가는 0.09. lift_plate regrasp와 같은 처리)
        resampling_time_range=(EPISODE_LENGTH_S + 1.0, EPISODE_LENGTH_S + 1.0),
        debug_vis=True,
        min_distance=mdp.GOAL_MIN_DIST,
        ranges=mdp.PickPlaceGoalCommandCfg.Ranges(
            pos_x=mdp.CUBE_X_B,
            pos_y=mdp.CUBE_Y_B,
            pos_z=(mdp.CUBE_REST_Z_B, mdp.CUBE_REST_Z_B),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    # 오른팔 7 + 오른손 그리퍼 1 = 8. arm_action은 __post_init__에서 apply_sim2real_control이
    # ReachRelativeJointPositionActionCfg(±0.1 rad, policy_step)로 바꾼다. 왼팔은 PD 목표가 ready로 남는다
    arm_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=RIGHT_ARM_JOINT_NAMES, preserve_order=True, scale=0.1
    )
    gripper_action = mdp.DelayedBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_fr3_finger_joint.*"],
        open_command_expr={"right_fr3_finger_joint.*": mdp.FINGER_OPEN},
        close_command_expr={"right_fr3_finger_joint.*": 0.0},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """actor 36D: 실물에서 얻는 신호만 (오른팔 엔코더, 손가락 폭, ArUco 큐브 pose, FK TCP, 목표)."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _right_arm_joints()}, noise=Gnoise(std=0.005))
        gripper_pos = ObsTerm(
            func=mdp.finger_pos_held,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["right_fr3_finger_joint1"])},
            noise=Gnoise(std=0.001),
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _right_arm_joints()}, noise=Gnoise(std=0.05))
        # 큐브 위치·자세: box BoxDrop과 같은 잡음(1 cm, 0.02)과 지연·끊김. cube_rel_tcp보다 먼저 계산해야 한다
        cube_position = ObsTerm(func=mdp.cube_position_vision, noise=Gnoise(std=0.01))
        cube_orientation = ObsTerm(func=mdp.cube_orientation_vision, noise=Gnoise(std=0.02))
        cube_rel_tcp = ObsTerm(func=mdp.cube_pos_rel_tcp, noise=Gnoise(std=0.01))
        goal_position = ObsTerm(func=mdp.goal_position)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """critic 전용 23D (policy 36D에 이어 붙임): 실제 큐브 pose·속도, TCP, 단계 one-hot, 파지/접촉/끊김, 책상 높이."""

        cube_position = ObsTerm(func=mdp.cube_position_true)
        cube_orientation = ObsTerm(func=mdp.cube_orientation_true)
        cube_lin_vel = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("object")})
        tcp_position = ObsTerm(func=mdp.tcp_position)
        privileged = ObsTerm(func=mdp.pick_place_privileged)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    # reset_joint_targets: 정책이 안 쓰는 왼팔의 PD 목표가 ready로 남게 (joint4/6은 0이 가동범위 밖)
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})
    # 시작 자세: 오른팔 관절마다 ready ±0.05 rad (왼팔은 ready 그대로)
    reset_right_arm = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"asset_cfg": _right_arm_joints(), "position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)},
    )
    # 책상 높이 ±1 cm. 큐브를 그 위에 놓으니 reset_cube보다 먼저
    reset_table = EventTerm(func=mdp.reset_table_height, mode="reset", params={"dz_range": mdp.TABLE_DZ_RANGE})
    reset_cube = EventTerm(
        func=mdp.reset_cube_upright,
        mode="reset",
        params={"x_range": mdp.CUBE_X_B, "y_range": mdp.CUBE_Y_B, "yaw_range": (-mdp.CUBE_YAW, mdp.CUBE_YAW)},
    )
    # 물성 DR: 마찰은 cube lift Run C와 같음, 질량은 실측 90 g 기준 70~100 g
    cube_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    cube_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.07, 0.10),
            "operation": "abs",
            "recompute_inertia": True,
        },
    )
    # 그리퍼: 조이는 힘 25~35 N (손가락 PD 강성 배율), 손가락 속도 ±10%. 명령 지연은 gripper_action이 명령마다 뽑는다
    finger_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["right_fr3_finger_joint.*"]),
            "stiffness_distribution_params": mdp.GRIPPER_FORCE_SCALE_RANGE,
            "operation": "scale",
        },
    )
    finger_speed = EventTerm(
        func=mdp.randomize_finger_speed,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["right_fr3_finger_joint.*"]), "speed": GRIPPER_FINGER_SPEED},
    )
    fr3_joint_limits = EventTerm(func=set_fr3_datasheet_joint_limits, mode="startup", params={"asset_cfg": _both_arm_joints()})


@configclass
class RewardsCfg:
    progress = RewTerm(func=mdp.pick_place_progress, weight=PROGRESS_WEIGHT)
    # 성공 상태에 머문 초 (TB 비교용)
    success = RewTerm(func=mdp.pick_place_success, weight=1.0)
    drop = RewTerm(
        func=mdp.is_terminated_term, weight=-100.0, params={"term_keys": ["cube_dropped", "cube_toppled", "cube_off_table"]}
    )
    # handover v1은 -1에서 운반 중 기울기 29~44°가 남았다 -> 더 세게 (45° 기울이면 스텝당 -1.6)
    cube_tilt = RewTerm(func=mdp.cube_tilt_while_held, weight=-2.0)
    joint_limit_margin = RewTerm(func=fr3_joint_limit_margin, weight=-1.0, params={"asset_cfg": _both_arm_joints()})
    # 11-09: A model_700이 의도보다 4~5 cm 깊게 쥐며 손가락으로 책상을 찍었다 (128개 중 103개, 최대 136 N). 실물은 충돌 반사로
    # 멈춘다. 처음부터 켠다: 파지점(손가락 끝 책상 위 GRASP_OFFSET + 5 cm)은 하한보다 높아 다가가기는 벌점이 없다
    finger_table = RewTerm(func=mdp.finger_table_contact, weight=-5.0)
    tip_floor = RewTerm(func=mdp.tip_below_floor, weight=-10.0)
    # 아래 셋은 놓기를 배운 뒤 커리큘럼으로 강화한다 (handover와 같은 최종값)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": _right_arm_joints()})
    arm_contact = RewTerm(
        func=mdp.undesired_contacts_any,
        weight=-1e-4,
        params={"threshold": 1.0, "sensor_cfgs": [SceneEntityCfg(name) for name in ARM_CONTACT_SENSORS]},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cube_off_table = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": mdp.TABLE_TOP_Z_B + mdp.ROBOT_BASE_Z - 0.1, "asset_cfg": SceneEntityCfg("object")},
    )
    cube_dropped = DoneTerm(func=mdp.cube_dropped)
    cube_toppled = DoneTerm(func=mdp.cube_toppled)


@configclass
class CurriculumCfg:
    # progress가 먼저 EMA를 갱신해야 아래 페널티 판정이 같은 reset의 값을 본다
    progress = CurrTerm(func=mdp.track_pick_place_progress)
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight_on_placed,
        params={"term_name": "action_rate", "weight": -1e-1, "threshold": LEARNED_THRESHOLD},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight_on_placed,
        params={"term_name": "joint_vel", "weight": -1e-1, "threshold": LEARNED_THRESHOLD},
    )
    arm_contact = CurrTerm(
        func=mdp.modify_reward_weight_on_placed,
        params={"term_name": "arm_contact", "weight": -1.0, "threshold": LEARNED_THRESHOLD},
    )


@configclass
class DualFR3PickPlaceCubeEnvCfg(ManagerBasedRLEnvCfg):
    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.episode_length_s = EPISODE_LENGTH_S
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=64 * 1024,
            friction_correlation_distance=0.00625,
        )
        self.viewer.eye = (1.8, -1.6, 1.6)
        self.viewer.lookat = (0.55, -0.2, 0.75)
        for side, links in ARM_CONTACT_LINKS.items():
            for link in links:
                setattr(self.scene, f"{side}_{link}_contact", ContactSensorCfg(prim_path=arm_link_path(side, link)))

        # 물리/PD 100 Hz, 정책 20 Hz, explicit PD + 토크 한계/변화율, ±0.1 rad, 팔 게인 DR (lift Sim2Real와 같음)
        apply_sim2real_control(self, physics_dt=0.01)
        self.actions.arm_action.target_limits = FR3_JOINT_POS_LIMITS
        # 손가락 속도·조이는 힘을 실물 수준으로 (sim 기본은 0.1 s 안에 닫히고 5 cm 큐브를 손가락당 47 N으로 조임).
        # 명령 지연·대기는 gripper_action이 넣는다
        actuators = dict(self.scene.robot.actuators)
        actuators["hands"] = actuators["hands"].replace(
            velocity_limit_sim=GRIPPER_FINGER_SPEED, stiffness=FINGER_STIFFNESS
        )
        self.scene.robot = self.scene.robot.replace(actuators=actuators)

        cube_marker = sim_utils.CuboidCfg(
            size=mdp.CUBE_SIZE, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.3), opacity=0.35)
        )
        self.commands.goal_pose.goal_pose_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Command/goal_pose", markers={"cuboid": cube_marker}
        )
        self.commands.goal_pose.current_pose_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Command/body_pose",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=1.0e-4, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0))
                )
            },
        )
        if os.environ.get("DR_VIS", "1") == "0":
            del self.scene.cube_zone


def _apply_play(cfg) -> None:
    """평가용: obs noise 끔 (비전 지연·끊김과 물성 DR은 유지)."""
    cfg.scene.num_envs = 50
    cfg.scene.env_spacing = 3.0
    cfg.observations.policy.enable_corruption = False


@configclass
class DualFR3PickPlaceCubeEnvCfg_PLAY(DualFR3PickPlaceCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play(self)


# ---------- 실험 변형 (11-09, 4090). 기준은 5070의 DualFR3PickPlaceCubeEnvCfg 학습 ----------


@configclass
class DualFR3PickPlaceCubeGripperBusyEnvCfg(DualFR3PickPlaceCubeEnvCfg):
    """실험 A: actor 관측 맨 뒤에 그리퍼 명령 진행 중 비트 (36 -> 37D).

    실물: 노드가 그리퍼 goal을 보낸 뒤 result를 받기 전까지 1 (fr3_husky_etri 노드의 send/result 이벤트).
    지금 actor는 0.27 s마다 바뀌는 손가락 폭만 보고 명령 지연을 추측해야 한다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.gripper_busy = ObsTerm(func=mdp.gripper_busy)


@configclass
class DualFR3PickPlaceCubeGripperBusyEnvCfg_PLAY(DualFR3PickPlaceCubeGripperBusyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play(self)


@configclass
class DualFR3PickPlaceCubeGripperBusyTiltEnvCfg(DualFR3PickPlaceCubeGripperBusyEnvCfg):
    """A + 기울기 벌점: 기울기 중 15°를 넘는 만큼 weight -5, 언제나 (캐비닛 책 -Tilt와 같음).

    11-09 A model_700은 들고 옮기는 동안 최대 기울기가 잡는 깊이와 관계없이 32~35°였다 (cube_tilt -2로는 부족).
    A(파지 수정만)와 동시에 돌려 기울기 벌점의 효과만 따로 본다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.cube_tilt_excess = RewTerm(
            func=mdp.cube_tilt_excess, params={"threshold_deg": mdp.TILT_EXCESS_DEG}, weight=-5.0
        )


@configclass
class DualFR3PickPlaceCubeGripperBusyTiltEnvCfg_PLAY(DualFR3PickPlaceCubeGripperBusyTiltEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play(self)


def _add_critic_plus(cfg) -> None:
    """critic에 그리퍼 내부 상태 8D (단계 one-hot, 타이머, 명령 비트)와 DR 값 7D (큐브 pose 관측 오프셋, 손가락 강성·속도 배율,
    비전 지연)를 더한다 (critic 23 -> 38D). actor는 그대로다."""
    cfg.observations.critic.gripper_state = ObsTerm(func=mdp.gripper_privileged)
    cfg.observations.critic.dr_values = ObsTerm(
        func=mdp.dr_privileged,
        params={
            "finger_cfg": SceneEntityCfg("robot", joint_names=["right_fr3_finger_joint1"]),
            "nominal_stiffness": FINGER_STIFFNESS,
            "nominal_speed": GRIPPER_FINGER_SPEED,
        },
    )


@configclass
class DualFR3PickPlaceCubeGripperBusyCriticPlusEnvCfg(DualFR3PickPlaceCubeGripperBusyEnvCfg):
    """A + B: actor에 그리퍼 진행 중 비트 (37D), critic에 그리퍼 내부 상태와 DR 값 (38D). actor는 A와 같아 배포도 A와 같다.

    11-09 4090 model_800: B는 A보다 쥐는 순간 기울기 3° (A 8~29°), 옮기는 동안 8~12° (A 28~33°), 손가락-책상 접촉 env
    14/128 (A 80/128)로 파지가 더 조심스러웠다. 다만 들기를 A보다 약 370 iteration 늦게(620~650) 배웠다.
    """

    def __post_init__(self):
        super().__post_init__()
        _add_critic_plus(self)


@configclass
class DualFR3PickPlaceCubeGripperBusyCriticPlusEnvCfg_PLAY(DualFR3PickPlaceCubeGripperBusyCriticPlusEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play(self)


@configclass
class DualFR3PickPlaceCubeCriticPlusEnvCfg(DualFR3PickPlaceCubeEnvCfg):
    """실험 B: critic에만 그리퍼 내부 상태 8D (단계 one-hot, 타이머, 명령 비트)와 DR 값 7D (큐브 pose 관측 오프셋,
    손가락 강성·속도 배율, 비전 지연) 추가 (critic 23 -> 38D). actor는 36D 그대로라 배포는 기준 run과 같다.
    """

    def __post_init__(self):
        super().__post_init__()
        _add_critic_plus(self)


@configclass
class DualFR3PickPlaceCubeCriticPlusEnvCfg_PLAY(DualFR3PickPlaceCubeCriticPlusEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play(self)


def _apply_near_table(cfg) -> None:
    """책상을 로봇 쪽으로 당긴 배치: 상판·다리·영역 표시, 큐브 시작과 목표 x 범위 (시작은 ready 손 앞 10 cm 밖),
    손가락 끝 하한의 책상 앞 모서리."""
    table_x, cube_x = mdp.NEAR_TABLE_X_B, mdp.NEAR_CUBE_X_B
    cx = 0.5 * (table_x[0] + table_x[1])
    scene = cfg.scene
    scene.table = scene.table.replace(
        spawn=scene.table.spawn.replace(size=(table_x[1] - table_x[0], *scene.table.spawn.size[1:])),
        init_state=scene.table.init_state.replace(pos=(cx, *scene.table.init_state.pos[1:])),
    )
    for name in ("table_leg_l", "table_leg_r"):
        leg = getattr(scene, name)
        setattr(scene, name, leg.replace(init_state=leg.init_state.replace(pos=(cx, *leg.init_state.pos[1:]))))
    if getattr(scene, "cube_zone", None) is not None:
        zone = scene.cube_zone
        scene.cube_zone = zone.replace(
            spawn=zone.spawn.replace(size=(cube_x[1] - cube_x[0], *zone.spawn.size[1:])),
            init_state=zone.init_state.replace(pos=(0.5 * sum(cube_x), *zone.init_state.pos[1:])),
        )
    cfg.events.reset_cube.params["x_range"] = cube_x
    cfg.events.reset_cube.params["keep_out_radius"] = mdp.NEAR_START_KEEP_OUT
    cfg.commands.goal_pose.ranges.pos_x = cube_x
    cfg.rewards.tip_floor.params["table_front_x"] = table_x[0]


@configclass
class DualFR3PickPlaceCubeGripperBusyNearEnvCfg(DualFR3PickPlaceCubeGripperBusyEnvCfg):
    """A + 책상을 10 cm 당긴 배치 (11-09): 앞 모서리 x 0.40 -> 0.30, 큐브 시작·목표 x 0.45~0.65 -> 0.35~0.65 (y와 오른팔
    반경 0.62 m는 같음). 목표 영역 약 1.65배, 시작은 ready 손 앞 10 cm를 빼서 약 1.55배.
    actor는 A와 같아 model_4000(release_goalfix_5070)에서 이어 학습한다. model_4000을 그대로 쓰면 71.5 % (기준 배치 92 %),
    시작·목표 모두 x 0.35~0.45면 48 %.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_near_table(self)


@configclass
class DualFR3PickPlaceCubeGripperBusyNearEnvCfg_PLAY(DualFR3PickPlaceCubeGripperBusyNearEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play(self)


# ---------- -Curriculum (2026-11-10): 11-09 실험 변형과 후보 항목 전부를 한 run에 ----------
# 기준은 GripperBusy(4000 계보: busy 비트, 실물 그리퍼 지연, 큐브 pose 오프셋). 항목은 CUBE_CURRICULUM_FLAGS로 켜고 끄고, 결과를 보고
# 나쁜 것을 빼는 ablation 절차는 docs/dual_fr3_curriculum_eval.md.
from dual_fr3_lab.tasks.manager_based.lift_plate.joint_pos_env_cfg import set_env_attrs  # noqa: E402  # isort: skip

CUBE_CURRICULUM_FLAGS = {
    "critic_plus": True,  # critic에 그리퍼 내부 상태 8D + DR 값 7D (4090 B: 파지·운반 기울기 작음)
    "tilt": True,  # 기울기 15° 초과 벌점 -5, 들기(reach_held EMA 0.5)를 배운 뒤
    "carry_strict": True,  # 놓기(reach_placed EMA 0.3)를 배운 뒤 들었던 큐브를 끌면 HELD가 아님 (목표 5 cm 안은 예외)
    "finger_force": True,  # 손가락-책상 접촉력 비례 벌점 (20 N에 -2/s, 100 N에 -10/s). 닿음 0/1 벌점 -5는 그대로
    "gripper_settle": True,  # 그리퍼 멈춤 -> result 대기 0.8~1.2 s (sim 기본 0.3~0.4)
    "near_ramp": True,  # 책상은 처음부터 -Near(앞 모서리 0.30), 큐브 시작·목표 x 0.45~0.65 -> 0.35~0.65 ramp (놓기를 배운 뒤 200 iter)
}
GRIPPER_SETTLE_REAL_S = (0.8, 1.2)
CUBE_TILT_WEIGHT = -5.0
CUBE_TILT_KEY, CUBE_TILT_THRESHOLD = "reach_held", 0.5
FINGER_FORCE_WEIGHT = -2.0
CUBE_NEAR_RAMP_STEPS = 4800  # 200 iter


def _apply_cube_curriculum(cfg, flags: dict) -> None:
    if flags["critic_plus"]:
        _add_critic_plus(cfg)
    if flags["tilt"]:
        cfg.rewards.cube_tilt_excess = RewTerm(
            func=mdp.cube_tilt_excess, params={"threshold_deg": mdp.TILT_EXCESS_DEG}, weight=-1e-4
        )
        cfg.curriculum.cube_tilt_excess = CurrTerm(
            func=mdp.modify_reward_weight_on_placed,
            params={"term_name": "cube_tilt_excess", "weight": CUBE_TILT_WEIGHT, "threshold": CUBE_TILT_THRESHOLD, "key": CUBE_TILT_KEY},
        )
    if flags["carry_strict"]:
        cfg.curriculum.carry_strict = CurrTerm(
            func=mdp.set_env_attr_on_placed, params={"attr": "_pp_carry_strict", "threshold": LEARNED_THRESHOLD}
        )
    if flags["finger_force"]:
        cfg.rewards.finger_table_force = RewTerm(func=mdp.finger_table_force_norm, weight=FINGER_FORCE_WEIGHT)
    if flags["gripper_settle"]:
        cfg.actions.gripper_action.settle_range_s = GRIPPER_SETTLE_REAL_S
    if flags["near_ramp"]:
        _apply_near_table(cfg)
        cfg.events.reset_cube.params["x_range"] = mdp.CUBE_X_B
        cfg.commands.goal_pose.ranges.pos_x = mdp.CUBE_X_B
        cfg.curriculum.near_x = CurrTerm(
            func=mdp.ramp_cube_x_range_on_placed,
            params={
                "start": mdp.CUBE_X_B,
                "end": mdp.NEAR_CUBE_X_B,
                "ramp_steps": CUBE_NEAR_RAMP_STEPS,
                "threshold": LEARNED_THRESHOLD,
            },
        )


def _apply_cube_curriculum_final(cfg, flags: dict) -> None:
    """평가 cfg: 커리큘럼 최종 상태 (기울기 벌점 최종, 끌기 금지, Near 전체 범위)."""
    if flags["tilt"]:
        cfg.rewards.cube_tilt_excess.weight = CUBE_TILT_WEIGHT
        cfg.curriculum.cube_tilt_excess = None
    if flags["carry_strict"]:
        cfg.events.curriculum_final = EventTerm(func=set_env_attrs, mode="startup", params={"attrs": {"_pp_carry_strict": 1.0}})
        cfg.curriculum.carry_strict = None
    if flags["near_ramp"]:
        cfg.events.reset_cube.params["x_range"] = mdp.NEAR_CUBE_X_B
        cfg.commands.goal_pose.ranges.pos_x = mdp.NEAR_CUBE_X_B
        cfg.curriculum.near_x = None


@configclass
class DualFR3PickPlaceCubeCurriculumEnvCfg(DualFR3PickPlaceCubeGripperBusyEnvCfg):
    """GripperBusy(4000 계보) + 후보 항목 전부 (CUBE_CURRICULUM_FLAGS), 처음부터 한 run.

    actor는 GripperBusy와 같은 37D라 배포 노드를 그대로 쓴다. critic은 CriticPlus(38D)라 4000 checkpoint는 못 읽는다.
    ablation은 _flags를 덮어쓴 하위 클래스로 만든다.
    """

    def _flags(self) -> dict:
        return dict(CUBE_CURRICULUM_FLAGS)

    def __post_init__(self):
        super().__post_init__()
        _apply_cube_curriculum(self, self._flags())


@configclass
class DualFR3PickPlaceCubeCurriculumEnvCfg_PLAY(DualFR3PickPlaceCubeCurriculumEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play(self)
        _apply_cube_curriculum_final(self, self._flags())
