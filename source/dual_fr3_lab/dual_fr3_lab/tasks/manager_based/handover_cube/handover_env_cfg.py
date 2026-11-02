# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dual FR3 큐브 핸드오버 (오른손 -> 왼손) task v0 (2026-10-25). 설계: docs/dual_fr3_handover_cube.md

테이블 오른쪽 끝(y -0.40~-0.30)의 큐브(5x5x10 cm)를 오른팔이 집어 중앙 핸드오버 지점(y 0, 상판 위 20 cm)
으로 옮기고, 왼팔이 받아 왼쪽 끝 목표(y +0.30~+0.40)에 세워 놓는다. 단계 판정/보상은 mdp/handover.py.

- 씬/큐브/테이블: lift_cube v3와 같은 MJCF 테이블과 실물 cube (크기 DR 없음: 실측 정확, 쥔 상태 시작에 필요)
- 제어: Reach Sim2Real v2 계약 (apply_sim2real_control, 100 Hz 물리 / 20 Hz 정책, ±0.1 rad) + 양팔 14관절
  (left1~7, right1~7) + binary 그리퍼 2개 (left, right). PD 목표는 FR3 datasheet 한계로 자른다
- 관절 한계: startup에서 sim 한계를 datasheet로 좁히고 한계 여유 페널티(-1)를 처음부터 건다 (box PlateOnly와 같음)
- 페널티 커리큘럼: 자연 시작 에피소드의 핸드오버 도달률 EMA > 0.3이면 action_rate/joint_vel/팔 접촉을 강화

v1 (2026-11-02, DualFR3HandoverCubeV1EnvCfg): v0 본 학습에서 본 세 문제를 고친다
- 넘겨받는 위치가 목표 쪽으로 밀림 -> 핸드오버 지점 HANDOVER_ZONE 안에서만 넘겨받기 인정 + TCP 중앙선 페널티
- 큐브를 27~56° 기울여 운반 -> 쥐고 있는 동안 기울기 페널티
- 양손 파지 시간이 거의 0 (sim 그리퍼가 즉시 닫힘) -> 손가락 속도를 실물 수준으로 제한
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
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise
from isaaclab_physx.physics import PhysxCfg

from . import mdp

##
# Pre-defined configs
##
from dual_fr3_lab.robots.dual_fr3 import DUAL_FR3_CFG  # isort: skip
from dual_fr3_lab.tasks.manager_based.lift_cube.joint_pos_env_cfg import (  # isort: skip
    ROOT_Z,
    TABLE_TOP_SIZE,
    TABLE_X,
    TCP_OFFSET_POS,
    TCP_OFFSET_QUAT,
    apply_sim2real_control,
)
from dual_fr3_lab.tasks.manager_based.lift_plate.joint_pos_env_cfg import (  # isort: skip
    FR3_JOINT_POS_LIMITS,
    fr3_joint_limit_margin,
    set_fr3_datasheet_joint_limits,
)
from dual_fr3_lab.tasks.manager_based.reach.joint_pos_env_cfg import DUAL_ARM_JOINT_NAMES  # isort: skip

EPISODE_LENGTH_S = 12.0  # 20 Hz x 12 s = 240 step (cube lift 8 s에 핸드오버/놓기 시간 추가)
PROGRESS_WEIGHT = 5.0  # 단계 보상 상한 (8 + 1) x 5 = 45/s
GOAL_POS_X = (0.40, 0.50)  # base frame (base는 world와 회전이 같고 z만 ROOT_Z 위)
GOAL_POS_Y = (0.30, 0.40)
HANDOVER_LEARNED_THRESHOLD = 0.3  # 자연 시작 에피소드의 핸드오버 도달률 EMA

# v1 (2026-11-02)
HANDOVER_ZONE = 0.08  # 넘겨받기 인정 반경 (큐브 중심 - 핸드오버 지점, m)
MIDLINE_MARGIN = 0.05  # TCP가 중앙선을 이만큼 넘으면 페널티 (핸드오버 때 두 TCP는 y ~0)
# 손가락당 최대 속도 (m/s). Franka Hand 폭 기준 0.1 m/s (franka_gripper 기본 속도)를 손가락 둘로 나눔.
# 열림 -> 큐브 파지(4 -> 2.5 cm)에 0.3 s. 실측 닫힘 시간이 있으면 그 값으로 바꾼다
GRIPPER_FINGER_SPEED = 0.05


def _table_part(name: str, size, pos, color=(0.9, 0.9, 0.9)):
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        spawn=sim_utils.MeshCuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
    )


def _pad_sensor(side: str, finger: str):
    # filter 순서 고정: [0]=Object(파지 판정), [1]=Table. PhysX filter는 sensor body당 1:1이라 패드마다 분리
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_fr3_{finger}",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object", "{ENV_REGEX_NS}/Table"],
    )


def _visual_plate(name: str, pos, size, color):
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        spawn=sim_utils.MeshCuboidCfg(size=size, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)),
    )


@configclass
class HandoverSceneCfg(InteractiveSceneCfg):
    robot = DUAL_FR3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        # Lab 3.0 quat은 (x, y, z, w)
        init_state=RigidObjectCfg.InitialStateCfg(pos=mdp.CUBE_SPAWN_POS, rot=[0, 0, 0, 1]),
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
            mass_props=sim_utils.MassPropertiesCfg(density=1000.0),  # 0.25 kg (mass DR x0.7~1.3)
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.3, 0.85)),
        ),
    )
    tcp_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left_fr3_link7",
                name="left_tcp",
                offset=OffsetCfg(pos=TCP_OFFSET_POS, rot=TCP_OFFSET_QUAT),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_fr3_link7",
                name="right_tcp",
                offset=OffsetCfg(pos=TCP_OFFSET_POS, rot=TCP_OFFSET_QUAT),
            ),
        ],
    )

    # MJCF 테이블 (lift_cube와 같음): 상판 윗면 z=TABLE_TOP_Z, 다리 y=±0.49, 바닥 러너
    table = _table_part("Table", TABLE_TOP_SIZE, (TABLE_X, 0.0, mdp.TABLE_TOP_Z - 0.01), (0.8, 0.6, 0.4))
    table_left_leg = _table_part("TableLeftLeg", (0.02, 0.02, 0.71), (TABLE_X, 0.49, 0.375))
    table_right_leg = _table_part("TableRightLeg", (0.02, 0.02, 0.71), (TABLE_X, -0.49, 0.375))
    table_left_runner = _table_part("TableLeftRunner", (0.6, 0.02, 0.02), (TABLE_X, 0.49, 0.01))
    table_right_runner = _table_part("TableRightRunner", (0.6, 0.02, 0.02), (TABLE_X, -0.49, 0.01))

    # 시각 전용 (충돌체 없음): 스폰 범위(파랑), 목표 범위(초록), 핸드오버 지점(주황). DR_VIS=0이면 뺀다
    spawn_zone = _visual_plate(
        "SpawnZone", (mdp.CUBE_SPAWN_POS[0], mdp.CUBE_SPAWN_POS[1], mdp.TABLE_TOP_Z + 0.001),
        (0.10 + 0.05, 0.10 + 0.05, 0.002), (0.1, 0.3, 0.9),
    )
    goal_zone = _visual_plate(
        "GoalZone", (sum(GOAL_POS_X) / 2, sum(GOAL_POS_Y) / 2, mdp.TABLE_TOP_Z + 0.001),
        (GOAL_POS_X[1] - GOAL_POS_X[0] + 0.05, GOAL_POS_Y[1] - GOAL_POS_Y[0] + 0.05, 0.002), (0.1, 0.8, 0.3),
    )
    handover_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/HandoverMarker",
        init_state=AssetBaseCfg.InitialStateCfg(pos=mdp.HANDOVER_POS),
        spawn=sim_utils.SphereCfg(radius=0.012, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0))),
    )

    # 파지 판정: 양손 각 패드-큐브 접촉력
    left_leftfinger_contact = _pad_sensor("left", "leftfinger")
    left_rightfinger_contact = _pad_sensor("left", "rightfinger")
    right_leftfinger_contact = _pad_sensor("right", "leftfinger")
    right_rightfinger_contact = _pad_sensor("right", "rightfinger")
    # 팔 링크 접촉(손가락 제외): 테이블, 반대 팔, 자기 몸. link7은 hand가 병합된 body
    left_arm_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/left_fr3_link[1-7]")
    right_arm_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/right_fr3_link[1-7]")

    plane = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    # 목표: 왼쪽 끝 테이블 위에 세워 놓인 큐브 중심 (base frame). 에피소드당 1개
    goal_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="left_fr3_link7",
        resampling_time_range=(EPISODE_LENGTH_S, EPISODE_LENGTH_S),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=GOAL_POS_X,
            pos_y=GOAL_POS_Y,
            pos_z=(mdp.CUBE_REST_Z - ROOT_Z, mdp.CUBE_REST_Z - ROOT_Z),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    # 순서: 양팔 14 (left1~7, right1~7) + 왼손 그리퍼 + 오른손 그리퍼 = 16. arm_action은 __post_init__에서
    # apply_sim2real_control이 ReachRelativeJointPositionActionCfg(±0.1 rad, policy_step)로 바꾼다
    arm_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=DUAL_ARM_JOINT_NAMES, preserve_order=True, scale=0.1
    )
    left_gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_fr3_finger_joint.*"],
        open_command_expr={"left_fr3_finger_joint.*": mdp.FINGER_OPEN},
        close_command_expr={"left_fr3_finger_joint.*": 0.0},
    )
    right_gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_fr3_finger_joint.*"],
        open_command_expr={"right_fr3_finger_joint.*": mdp.FINGER_OPEN},
        close_command_expr={"right_fr3_finger_joint.*": 0.0},
    )


def _arm_joints():
    return SceneEntityCfg("robot", joint_names=DUAL_ARM_JOINT_NAMES, preserve_order=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """actor 62D: 실물에서 얻을 수 있는 신호만 (관절 엔코더, 그리퍼 폭, 큐브 6D pose, 목표)."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _arm_joints()}, noise=Gnoise(std=0.005))
        gripper_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["left_fr3_finger_joint1", "right_fr3_finger_joint1"], preserve_order=True
                )
            },
            noise=Gnoise(std=0.001),
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _arm_joints()}, noise=Gnoise(std=0.05))
        cube_position = ObsTerm(func=mdp.object_position_in_robot_root_frame, noise=Gnoise(std=0.01))
        cube_orientation = ObsTerm(func=mdp.object_orientation_in_robot_root_frame, noise=Gnoise(std=0.02))
        cube_rel_tcps = ObsTerm(func=mdp.cube_pos_rel_tcps, noise=Gnoise(std=0.01))
        goal_position = ObsTerm(func=mdp.goal_position)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """critic 전용 15D (policy 62D에 이어 붙임): 단계 one-hot, handed, 양손 파지/접촉, 큐브 속도."""

        handover = ObsTerm(func=mdp.handover_privileged)
        cube_lin_vel = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("object")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    # reset 순서: 기본 자세 -> 큐브 위치 랜덤 -> 시작 모드(B/C면 쥔 자세로 덮어씀)
    # reset_joint_targets: 정책이 안 쓰는 joint의 PD 목표가 0으로 남지 않게 (joint4/6은 0이 가동범위 밖)
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})
    reset_cube_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    reset_start_state = EventTerm(func=mdp.reset_start_state, mode="reset")
    # 물성 DR: cube lift Run C와 같음 (obs noise와 세트)
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
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    fr3_joint_limits = EventTerm(func=set_fr3_datasheet_joint_limits, mode="startup", params={"asset_cfg": _arm_joints()})


@configclass
class RewardsCfg:
    progress = RewTerm(func=mdp.handover_progress, weight=PROGRESS_WEIGHT)
    # 성공 상태에 머문 초 (TB 비교용, 보상 기여는 progress의 1/45 수준)
    success = RewTerm(func=mdp.handover_success, weight=1.0)
    drop = RewTerm(
        func=mdp.is_terminated_term, weight=-100.0, params={"term_keys": ["cube_dropped_midway", "cube_off_table"]}
    )
    joint_limit_margin = RewTerm(func=fr3_joint_limit_margin, weight=-1.0, params={"asset_cfg": _arm_joints()})
    # 아래 셋은 핸드오버를 배운 뒤 커리큘럼으로 강화한다 (cube lift와 같은 최종값)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": _arm_joints()})
    arm_contact = RewTerm(
        func=mdp.undesired_contacts_any,
        weight=-1e-4,
        params={"threshold": 1.0, "sensor_cfgs": [SceneEntityCfg("left_arm_contact"), SceneEntityCfg("right_arm_contact")]},
    )


@configclass
class RewardsV1Cfg(RewardsCfg):
    # 넘겨받는 위치를 중앙에 묶는 것은 HANDOVER_ZONE이 하고, 이 항은 팔끼리 겹치는 것을 막는다
    midline = RewTerm(func=mdp.tcp_midline_crossing, weight=-2.0, params={"margin": MIDLINE_MARGIN})
    # 45° 기울이면 스텝당 -0.79 (progress 단계 하나가 +5)
    cube_tilt = RewTerm(func=mdp.cube_tilt_while_held, weight=-1.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cube_off_table = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": mdp.TABLE_TOP_Z - 0.1, "asset_cfg": SceneEntityCfg("object")},
    )
    cube_dropped_midway = DoneTerm(func=mdp.cube_dropped_midway)


@configclass
class CurriculumCfg:
    # start_modes가 먼저 EMA를 갱신해야 아래 페널티 판정이 같은 reset의 값을 본다
    start_modes = CurrTerm(
        func=mdp.anneal_start_modes,
        params={"p_right_hold": 0.0, "p_left_hold": 0.25, "enable_key": "reach_at_handover", "enable_threshold": 0.5},
    )
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight_on_handover_learned,
        params={"term_name": "action_rate", "weight": -1e-1, "threshold": HANDOVER_LEARNED_THRESHOLD},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight_on_handover_learned,
        params={"term_name": "joint_vel", "weight": -1e-1, "threshold": HANDOVER_LEARNED_THRESHOLD},
    )
    arm_contact = CurrTerm(
        func=mdp.modify_reward_weight_on_handover_learned,
        params={"term_name": "arm_contact", "weight": -1.0, "threshold": HANDOVER_LEARNED_THRESHOLD},
    )


@configclass
class DualFR3HandoverCubeEnvCfg(ManagerBasedRLEnvCfg):
    scene: HandoverSceneCfg = HandoverSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    # 넘겨받기 인정 반경 (mdp.HandoverState가 읽음). None이면 위치 조건 없음 (v0)
    handover_zone: float | None = None

    def __post_init__(self):
        self.episode_length_s = EPISODE_LENGTH_S
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            # 양손 파지/팔끼리 접촉으로 접촉쌍이 cube lift보다 많다 (cube lift도 16k -> 64k로 키움)
            gpu_total_aggregate_pairs_capacity=64 * 1024,
            friction_correlation_distance=0.00625,
        )
        # 중력보상 모델링: 로봇 링크 중력 제거 (실물 FR3 제어기가 g(q) 보상). 큐브 중력은 그대로
        self.scene.robot = self.scene.robot.copy()
        self.scene.robot.spawn.rigid_props.disable_gravity = True

        # 물리/PD 100 Hz, 정책 20 Hz, explicit PD + 토크 한계/변화율, ±0.1 rad, 팔 게인 DR (lift Sim2Real와 같음)
        apply_sim2real_control(self, physics_dt=0.01)
        self.actions.arm_action.target_limits = FR3_JOINT_POS_LIMITS * 2

        self.commands.goal_pose.goal_pose_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Command/goal_pose",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.02, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.2))
                )
            },
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
            del self.scene.spawn_zone
            del self.scene.goal_zone
            del self.scene.handover_marker


def _to_play(cfg: DualFR3HandoverCubeEnvCfg):
    """평가용: 자연 시작만, obs noise 끔 (물성 DR은 유지)."""
    cfg.scene.num_envs = 50
    cfg.scene.env_spacing = 3.0
    cfg.observations.policy.enable_corruption = False
    cfg.curriculum.start_modes.params["p_right_hold"] = 0.0
    cfg.curriculum.start_modes.params["p_left_hold"] = 0.0


@configclass
class DualFR3HandoverCubeEnvCfg_PLAY(DualFR3HandoverCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _to_play(self)


@configclass
class DualFR3HandoverCubeV1EnvCfg(DualFR3HandoverCubeEnvCfg):
    """v1: 핸드오버 위치 고정 + 중앙선/기울기 페널티 + 실물 그리퍼 속도. 나머지는 v0과 같다."""

    rewards: RewardsV1Cfg = RewardsV1Cfg()
    handover_zone: float | None = HANDOVER_ZONE

    def __post_init__(self):
        super().__post_init__()
        actuators = dict(self.scene.robot.actuators)
        actuators["hands"] = actuators["hands"].replace(velocity_limit_sim=GRIPPER_FINGER_SPEED)
        self.scene.robot = self.scene.robot.replace(actuators=actuators)


@configclass
class DualFR3HandoverCubeV1EnvCfg_PLAY(DualFR3HandoverCubeV1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _to_play(self)
