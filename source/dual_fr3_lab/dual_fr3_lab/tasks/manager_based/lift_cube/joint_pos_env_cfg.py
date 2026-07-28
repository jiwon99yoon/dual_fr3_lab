# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dual FR3 (husky 베이스) 오른팔로 cube lift task.

구도 (Run B, 2026-07-27): 바닥 z=0에 로봇, 앞에 MJCF 테이블(얇은 상판 z=0.75 + 양옆 다리,
`lift/config/dual_fr3_table.md` 참조) + DexCube. 로봇 링크는 disable_gravity로 실물 FR3의
중력보상을 모델링. grasp gating(양 finger pad 접촉) + 충돌 페널티 + 측정 joint 토크 obs 포함.

- base frame(팔 장착 상판)이 world z=0.405에 있으므로 world/base 변환에 주의
- 오른팔(yaw -30도 장착)만 action으로 제어, 왼팔은 ready pose를 PD가 유지
- ee body는 merge-joints로 hand가 병합된 right_fr3_link7 + TCP 오프셋(z+0.2104, yaw -45도)
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

# 베이스 프레임워크(LiftEnvCfg)는 isaaclab_tasks에서 import, 커스텀 mdp는 이 패키지 것 사용
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from . import mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from dual_fr3_lab.robots.dual_fr3 import DUAL_FR3_CFG  # isort: skip


# ---------- 배치 상수 (world 기준, MJCF 테이블: lift/config/dual_fr3_table.md) ----------
# 선배 MuJoCo 세팅의 얇은 상판(0.6x1.0x0.02, 윗면 z=0.75) + 양옆 다리 구조.
# 단, x 위치는 MJCF 원본(0.8)이 아니라 0.6 유지 — 0.8이면 cube 스폰존이 상판 앞모서리 밖 (2026-07-27 합의)
ROOT_Z = 0.405  # base frame 높이 (바퀴가 지면에 닿는 스폰 높이, DUAL_FR3_CFG에 포함)
TABLE_X = 0.6  # 테이블 중심 x (앞면 x=0.3, husky 앞머리 x~0.15와 15cm 마진)
TABLE_TOP_SIZE = (0.6, 1.0, 0.02)  # MJCF size(half-extent)의 2배
TABLE_TOP_Z = 0.75  # 상판 윗면 (base 기준 +0.345)
CUBE_POS_W = (0.45, -0.25, TABLE_TOP_Z + 0.03)  # 오른팔 workspace 중앙, 상판 위 3cm

# link7 -> TCP(그리퍼 파지점) 오프셋: joint8(z+0.107) + hand yaw -45도 + TCP(z+0.1034)
TCP_OFFSET_POS = (0.0, 0.0, 0.2104)
TCP_OFFSET_QUAT = (0.9238795, 0.0, 0.0, -0.3826834)  # Rz(-45도), (w, x, y, z)


def modify_reward_weight_on_lifting_learned(
    env, env_ids, term_name: str, weight: float, lifting_term: str = "lifting_object", threshold: float = 10.0
):
    """성능 기반 커리큘럼: '들기'를 배운 뒤에만 페널티를 강화한다.

    리셋되는 env들의 lifting 보상 에피소드 합(초당 평균, 만렙 ~14)을 **배치 크기 가중
    EMA**로 누적하고, EMA가 threshold를 넘으면 지정한 reward 항의 weight를 교체한다.
    한 번 발동하면 되돌리지 않으며(latch), 반환값은 Curriculum/ 태그로 기록된다.

    단순 배치 평균 판정의 사고 이력 (2026-07-27 Run C): drop으로 종료되는 env는 "들고
    다니다 놓친" 것이라 lifting 합이 구조적으로 높음 -> 소수 drop 배치가 전체 성능 ~4/s
    시점에 threshold를 넘겨 조기 latch -> 탐색 중(noise_std ~5.5) 정책에 action_rate
    페널티 -59/s 직격, 보상 -288 붕괴. EMA 가중치 설계:
    - 소수 배치(drop, 1~2 env): alpha ~ 0.001 -> 편향 표본이 연속으로 와도 못 뒤집음
    - 대형 배치(timeout, 전체의 대부분): alpha 상한 0.5 -> 스파이크 1회로는 못 넘고
      2회 연속 threshold 초과해야 발동 (발동이 에피소드 1주기 늦어지는 대가)
    - EMA는 term별 분리 저장 (3개 커리큘럼 항이 공유 시 실효 alpha 3배 문제)
    """
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    # 이미 발동됐으면 유지 (성능이 일시적으로 떨어져도 되돌리지 않음)
    if term_cfg.weight == weight:
        return weight
    if not hasattr(env, "_lift_curriculum_ema"):
        env._lift_curriculum_ema = {}
    episode_sums = env.reward_manager._episode_sums[lifting_term][env_ids]
    lifting_per_sec = episode_sums.mean().item() / env.max_episode_length_s
    alpha = min(0.5, 3.0 * len(env_ids) / env.num_envs)
    ema = env._lift_curriculum_ema.get(term_name, 0.0)
    ema = ema + alpha * (lifting_per_sec - ema)
    env._lift_curriculum_ema[term_name] = ema
    if ema > threshold:
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)
    return term_cfg.weight


@configclass
class DualFR3CubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ---------- 로봇 ----------
        self.scene.robot = DUAL_FR3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 중력보상 모델링: 실물 FR3는 제어기가 g(q)를 항상 보상 -> sim에서는 로봇 링크 중력 제거가 등가.
        # payload(cube) 중력은 살아 있어 파지 후 무게는 정상적으로 팔에 전달됨 (실물 중력보상도 payload는 모름)
        self.scene.robot.spawn.rigid_props.disable_gravity = True

        # ---------- 씬: 바닥 위 로봇 + MJCF 테이블 (얇은 상판 + 양옆 다리) ----------
        self.scene.plane.init_state.pos = (0.0, 0.0, 0.0)  # 바닥을 z=0으로 (franka는 -1.05)

        # PhysX GPU 충돌쌍 버퍼 증설 (베이스 16384 -> 65536): C 3차 it~4533에서 4096 env가
        # 동시에 파지 상태가 되자 16388~16395개를 요구하며 초과 에러 발생 (miss interactions).
        # 버퍼 크기만 키우는 것이라 물리/학습 거동에는 영향 없음
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024

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

        # 상판 (윗면 z=TABLE_TOP_Z), 수직 다리 y=±0.49 (z 0.02~0.73), 바닥 러너 (z 0~0.02)
        self.scene.table = _table_part("Table", TABLE_TOP_SIZE, (TABLE_X, 0.0, TABLE_TOP_Z - 0.01), (0.8, 0.6, 0.4))
        self.scene.table_left_leg = _table_part("TableLeftLeg", (0.02, 0.02, 0.71), (TABLE_X, 0.49, 0.375))
        self.scene.table_right_leg = _table_part("TableRightLeg", (0.02, 0.02, 0.71), (TABLE_X, -0.49, 0.375))
        self.scene.table_left_runner = _table_part("TableLeftRunner", (0.6, 0.02, 0.02), (TABLE_X, 0.49, 0.01))
        self.scene.table_right_runner = _table_part("TableRightRunner", (0.6, 0.02, 0.02), (TABLE_X, -0.49, 0.01))

        # ---------- DR 범위 시각화 (시각 전용, 충돌체 없음 -> 물리/학습 영향 없음) ----------
        # 주의: RTX Real-Time 렌더러는 opacity<0.5를 cutout으로 잘라 아예 안 보이므로 불투명 판으로 표시
        # 파란 판: 큐브 스폰 범위 (CUBE_POS_W "중심" 기준 x ±0.08, y ±0.12), 상판 바로 위
        # 판 크기는 큐브 몸체까지 포함하도록 최대 큐브(5.5cm)의 반변 2.75cm를 양쪽에 더함 (+0.055)
        self.scene.dr_spawn_zone = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/DrSpawnZone",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(CUBE_POS_W[0], CUBE_POS_W[1], TABLE_TOP_Z + 0.001)),
            spawn=sim_utils.MeshCuboidCfg(
                size=(0.16 + 0.055, 0.24 + 0.055, 0.002),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 0.9)),
            ),
        )
        # 초록 판 2장: 목표 샘플 범위의 아래/위 경계
        # (base 기준 x 0.35~0.55, y -0.45~-0.15, z 0.45~0.60 -> world z 0.855~1.005)
        # 목표도 큐브 "중심" 기준이므로 큐브 몸체 여유(+0.055)를 판 크기에 반영
        self.scene.dr_target_zone_bottom = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/DrTargetZoneBottom",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, -0.30, 0.45 + ROOT_Z)),
            spawn=sim_utils.MeshCuboidCfg(
                size=(0.20 + 0.055, 0.30 + 0.055, 0.003),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.3)),
            ),
        )
        self.scene.dr_target_zone_top = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/DrTargetZoneTop",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, -0.30, 0.60 + ROOT_Z)),
            spawn=sim_utils.MeshCuboidCfg(
                size=(0.20 + 0.055, 0.30 + 0.055, 0.003),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.3)),
            ),
        )

        # ---------- 왼팔 고정 ----------
        # action이 없는 joint는 position target이 0으로 초기화돼 PD가 0도 자세로 끌고 간다
        # (joint4/6은 0이 가동범위 밖이라 리밋으로 휩쓸림). 리셋 시 target을 default(ready)로
        # 초기화하면 왼팔은 ready pose를 유지하고, 오른팔 target은 매 스텝 action이 덮어쓴다.
        self.events.reset_all.params["reset_joint_targets"] = True

        # ---------- action: 오른팔 7 joint delta + 오른손 그리퍼 (왼팔은 PD가 ready pose 유지) ----------
        # Run C 3차부터 절대(scale 0.5) -> relative(delta, scale 0.1) 전환 (2026-07-28, plate와 통일).
        # 근거: "절대 action x obs noise" 조합에서 탐색 노이즈가 ±0.5rad 목표 점프로 증폭되어
        # noise_std 폭주 (C1: 5.5, C2: 12.8). delta는 스텝당 이동이 ±0.1rad로 구조적으로 캡되어
        # 탐색이 잔떨림으로 제한됨 — plate P2/P4가 같은 noise에서 안정 학습한 것이 실증.
        # 문헌: Demystifying Action Space Design(2602.23408, delta 일관 우위), gSDE(저크 탐색 문제)
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=["right_fr3_joint[1-7]"], scale=0.1
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["right_fr3_finger_joint.*"],
            open_command_expr={"right_fr3_finger_joint.*": 0.04},
            close_command_expr={"right_fr3_finger_joint.*": 0.0},
        )

        # ---------- observation: 오른팔 관련 joint만 (왼팔 9개 제외로 obs 차원 축소) ----------
        # 주의: SceneEntityCfg는 resolve 시 내부에 joint_ids가 채워지므로 term마다 별도 인스턴스 필요
        right_joint_names = ["right_fr3_joint[1-7]", "right_fr3_finger_joint.*"]
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(right_joint_names)
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(right_joint_names)
        )
        # 오른팔 측정 joint 토크 7 dims — FR3 내장 토크 센서(tau_ext) 대응이라 실배포 가능한 신호.
        # 충돌 페널티와 세트: 정책이 접촉을 "느끼고" 회피/반응하는 것을 배울 수 있게 한다
        self.observations.policy.joint_torques = ObsTerm(
            func=mdp.joint_measured_torques,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["right_fr3_joint[1-7]"])},
        )
        # cube orientation obs는 Run E(yaw DR)와 세트로 연기 — yaw 없는 지금은 cube 기울기가
        # 그리퍼 기울기와 같아 joint_pos로 추론 가능 (upright 제어에 별도 관측 불필요)

        # ---------- obs noise (Run C): 실물 센서/인식 수준 ----------
        # 강건성 진단(2026-07-27, Run B 정책): obs noise 단독 drop 0, 물성 DR 단독 drop 0,
        # "조합"만 슬립 드랍 468~750회/128ep -> 두 축을 반드시 함께 학습에 넣는다
        self.observations.policy.joint_pos.noise = Gnoise(std=0.005)  # rad, 엔코더+캘리브레이션
        self.observations.policy.joint_vel.noise = Gnoise(std=0.05)  # rad/s
        self.observations.policy.object_position.noise = Gnoise(std=0.01)  # m, perception 1cm
        self.observations.policy.joint_torques.noise = Gnoise(std=0.5)  # Nm, FR3 tau_ext 노이즈

        # ---------- command: 물체를 가져갈 목표 pose (base frame 기준) ----------
        self.commands.object_pose.body_name = "right_fr3_link7"
        # 오른팔 workspace: cube 시작점(base 기준 x0.45, y-0.25, z0.225) 주변 위쪽
        self.commands.object_pose.ranges.pos_x = (0.35, 0.55)
        # y 상한 -0.15: 몸 중앙선 근처 goal은 실익 없이 왼팔 충돌 위험만 키움 (Run C에서 확인,
        # 2026-07-27 원복 결정). 충돌 페널티는 테이블/자기충돌 대비용으로 유지
        self.commands.object_pose.ranges.pos_y = (-0.45, -0.15)
        # 상판 0.75에 맞춰 상향: world 0.855~1.005 (하한이 상판 +0.105)
        self.commands.object_pose.ranges.pos_z = (0.45, 0.60)
        # 시각화 정리: 목표는 orientation이 무의미하므로 축 대신 초록 구로,
        # current(link7) 마커는 보상에 안 쓰이므로 사실상 보이지 않게 축소
        self.commands.object_pose.goal_pose_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Command/goal_pose",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.025,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.2)),
                )
            },
        )
        self.commands.object_pose.current_pose_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Command/body_pose",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=1.0e-4,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
                )
            },
        )

        # ---------- object: DexCube (테이블 위) ----------
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_POS_W, rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        # 리셋 시 물체 위치 랜덤화: 오른팔 workspace를 벗어나지 않게 franka(y ±0.25)보다 좁힘
        # 스폰 yaw DR(±45°)은 Run E로 연기 (2026-07-28 결정): upright/finger_impact 거동 교정을
        # 먼저 단독 검증. yaw DR을 넣을 때 orientation obs(43->47)도 세트로 (파지 정합에 관측 필수)
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.08, 0.08),
            "y": (-0.12, 0.12),
            "z": (0.0, 0.0),
        }
        # 큐브 크기 랜덤화: 한 변 4.5~5.5cm (원본 DexCube 6.5cm 기준 절대 scale로 덮어씀)
        # mode="usd": 시뮬 시작 전 1회 적용 -> env마다 다른 크기가 학습 내내 유지 (4096개 분포로 커버)
        self.events.randomize_cube_size = EventTerm(
            func=mdp.randomize_rigid_body_scale,
            mode="usd",
            params={
                "scale_range": (4.5 / 6.5, 5.5 / 6.5),  # (0.692, 0.846)
                "asset_cfg": SceneEntityCfg("object"),
            },
        )
        # ---------- 물성 DR (Run C): 마찰/질량 — 위 obs noise와 세트 (진단 근거 동일) ----------
        self.events.randomize_cube_friction = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "static_friction_range": (0.4, 1.2),
                "dynamic_friction_range": (0.3, 1.0),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
                "make_consistent": True,  # dynamic <= static 보장
            },
        )
        self.events.randomize_cube_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "mass_distribution_params": (0.7, 1.3),
                "operation": "scale",
                "recompute_inertia": True,
            },
        )
        # env별로 다른 scale의 물리 속성을 개별 파싱하도록 필수 설정
        self.scene.replicate_physics = False

        # ---------- grasp 판정 센서: 오른손 양 finger pad와 Object 사이 접촉력만 필터링 ----------
        # 양쪽 pad가 동시에 물체를 눌러야 파지로 인정 -> 손등에 얹어 나르는 exploit은 gate 통과 불가
        # (2026-07-25 run에서 확인: reaching 보상 0.86 -> 0.07로 붕괴하며 서커스 거동 발현)
        # PhysX 제약: filter는 sensor body당 1:1 매칭이라 finger마다 sensor를 분리해야 함
        # filter 순서 고정: [0]=Object(파지 gate용), [1]=Table(충격 페널티용, Run D)
        self.scene.leftfinger_object_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_fr3_leftfinger",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object", "{ENV_REGEX_NS}/Table"],
        )
        self.scene.rightfinger_object_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_fr3_rightfinger",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object", "{ENV_REGEX_NS}/Table"],
        )
        # 오른팔 링크 충돌 감지 (finger 제외 — 물체를 만져야 하는 부위): 테이블/왼팔/자기몸 등
        # 모든 접촉의 net force를 본다 (filter 없음 -> body 수 제약 없음)
        self.scene.right_arm_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_fr3_link[1-7]",
        )

        # ---------- reward/termination: world z 기준값을 테이블 높이에 맞게 보정 ----------
        # franka는 상판=world z=0이라 minimal_height=0.04였음 (상판+4cm)
        lifted_height = TABLE_TOP_Z + 0.04  # franka와 동일한 +4cm (엄격하면 들기 발견이 늦어짐)
        grasp_gate_params = {
            "finger1_sensor_cfg": SceneEntityCfg("leftfinger_object_contact"),
            "finger2_sensor_cfg": SceneEntityCfg("rightfinger_object_contact"),
            "threshold": 0.5,
        }
        # 들기/운반 보상 전부를 "실제 파지 중"일 때만 지급 (높이 gate만으로는 얹어 나르기가 이득)
        self.rewards.lifting_object.func = mdp.object_is_lifted_and_grasped
        self.rewards.lifting_object.params = {"minimal_height": lifted_height, **grasp_gate_params}
        self.rewards.object_goal_tracking.func = mdp.object_goal_distance_grasped
        self.rewards.object_goal_tracking.params = {
            "std": 0.3,
            "minimal_height": lifted_height,
            "command_name": "object_pose",
            **grasp_gate_params,
        }
        self.rewards.object_goal_tracking_fine_grained.func = mdp.object_goal_distance_grasped
        self.rewards.object_goal_tracking_fine_grained.params = {
            "std": 0.05,
            "minimal_height": lifted_height,
            "command_name": "object_pose",
            **grasp_gate_params,
        }
        # fine_grained 5 유지: 10으로 올렸던 2026-07-25 run에서 exploit 발현에 기여 -> gating 검증 후 재상향 검토
        self.rewards.object_goal_tracking_fine_grained.weight = 5.0
        # 물체가 상판 아래로 떨어지면 종료 (franka는 -0.05)
        self.terminations.object_dropping.params["minimum_height"] = TABLE_TOP_Z - 0.1

        # ---------- curriculum: 스텝 수가 아니라 "들기를 배웠는가"로 페널티 강화 (성능 기반) ----------
        # 고정 스텝(franka 기본 10k)은 들기 발견 전에 발동하면 정책이 "안 움직이는" local optimum으로
        # 붕괴함 (2026-07-24 run에서 확인). lifting 보상(초당 평균, 만렙 ~14)이 10을 넘은 뒤에만 발동.
        self.curriculum.action_rate = CurrTerm(
            func=modify_reward_weight_on_lifting_learned,
            params={"term_name": "action_rate", "weight": -1e-1, "threshold": 10.0},
        )
        self.curriculum.joint_vel = CurrTerm(
            func=modify_reward_weight_on_lifting_learned,
            params={"term_name": "joint_vel", "weight": -1e-1, "threshold": 10.0},
        )

        # ---------- 충돌 페널티 (Run B): 오른팔 링크가 무엇이든 1N 이상으로 누르면 감점 ----------
        # goal y 범위 복원(-0.05)으로 왼팔 영역 침범이 다시 가능해진 것을 이 페널티가 막는다.
        # v1 교훈: 들기 발견 전에 페널티가 세지면 "안 움직이는" 정책으로 붕괴 -> 성능 기반 curriculum 필수
        self.rewards.undesired_contact_penalty = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1e-4,
            params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("right_arm_contact")},
        )
        # goal 10cm 이내에서만 cube 속도 페널티 -> "도착 후 가만히 들고 있기" 유도 (Run C).
        # 이동 중엔 gate가 닫혀 운반 속도를 안 죽임. goal 근처 도달 자체가 lifting 학습 후에만
        # 가능하므로 v1식 초기 붕괴 위험이 없어 curriculum 없이 즉시 활성
        self.rewards.object_hold_still = RewTerm(
            func=mdp.object_velocity_near_goal,
            weight=-0.5,
            params={"command_name": "object_pose", "dist_threshold": 0.1, "ang_vel_scale": 0.1},
        )
        # action_l2 (Run C 4차, delta 전환의 후속): delta 의미론에서 ‖a‖² = "움직임 명령 크기".
        # 평균 출력을 0으로 끌어당겨 null-space 유영(goal 도달 후 joint1/2 무의미 배회 — delta
        # 바이어스 적분 + 보상이 cube만 봐서 여유 자유도에 무관심)을 차단. dexsuite가 delta와
        # 세트로 쓰는 항. 절대 action 시절엔 "ready 편차 페널티"가 되어 태스크와 상충했으므로 부재.
        # 탐색기(std~3)엔 ‖a‖²가 커서 즉시 -0.005를 걸면 탐색 억압 -> 성능 curriculum에 태움
        self.rewards.action_l2 = RewTerm(func=mdp.action_l2, weight=-1e-4)
        # Run D: 운반 중 윗면 유지 (cube z축·world z 정렬도 × 파지∧들림 gate)
        self.rewards.object_upright = RewTerm(
            func=mdp.object_upright_while_carried,
            weight=3.0,
            params={"minimal_height": lifted_height, **grasp_gate_params},
        )
        # Run D: finger가 테이블을 10N 이상으로 때리면 페널티 ("콩 박기" 대응, 가벼운 스침은 무죄)
        self.rewards.finger_table_impact = RewTerm(
            func=mdp.finger_table_impact_penalty,
            weight=-1e-4,
            params={
                "finger1_sensor_cfg": SceneEntityCfg("leftfinger_object_contact"),
                "finger2_sensor_cfg": SceneEntityCfg("rightfinger_object_contact"),
                "threshold": 10.0,
            },
        )
        self.curriculum.contact_penalty = CurrTerm(
            func=modify_reward_weight_on_lifting_learned,
            params={"term_name": "undesired_contact_penalty", "weight": -1.0, "threshold": 10.0},
        )
        self.curriculum.action_l2 = CurrTerm(
            func=modify_reward_weight_on_lifting_learned,
            params={"term_name": "action_l2", "weight": -0.005, "threshold": 10.0},
        )
        self.curriculum.finger_table_impact = CurrTerm(
            func=modify_reward_weight_on_lifting_learned,
            params={"term_name": "finger_table_impact", "weight": -1.0, "threshold": 10.0},
        )

        # ---------- ee frame: right_fr3_link7 + TCP 오프셋 ----------
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_fr3_link7",
                    name="end_effector",
                    offset=OffsetCfg(pos=TCP_OFFSET_POS, rot=TCP_OFFSET_QUAT),
                ),
            ],
        )


@configclass
class DualFR3CubeLiftEnvCfg_PLAY(DualFR3CubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3CubeLiftEnvCfg_PLAY_NOISY(DualFR3CubeLiftEnvCfg_PLAY):
    """sim2real 강건성 평가용: 학습 분포 그대로 + obs noise 활성.

    Run C부터 obs noise/물성 DR이 학습 cfg에 포함되므로, 이 cfg는 PLAY가 끈 corruption을
    되켜는 것만 담당한다 (물성 DR은 PLAY에도 상속되어 이미 활성).
    강건성 진단 이력(2026-07-27, Run B 정책): 단독 축 drop 0 / 조합 468~750회 -> Run C 근거.
    평가 도구: scripts/tools/eval_dual_fr3_lift_robustness.py
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True
