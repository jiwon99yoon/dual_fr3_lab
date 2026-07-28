# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dual FR3 + 판(plate) EE 양팔로 바닥의 큰 박스를 조여서 드는 lift task.

구도: 바닥 z=0에 로봇, 앞 바닥에 30~50cm 박스. 그리퍼가 없으므로 양 판으로
박스를 조인(squeeze) 마찰력만으로 들어 목표 pose로 운반해야 한다.
rule-based 검증: scripts/tutorials/05_controllers/run_dual_fr3_plate_lift.py

- action: 양팔 14 joint delta position (relative, scale 0.1, 그리퍼 없음)
- 힘 제어는 침투 깊이로 암묵적으로 학습됨 (PD 오차 = 조임력).
  박스 크기(30~50cm)/질량(밀도) DR로 상황별 힘 조절을 학습시킨다
- obs에 양 판 접촉력 포함 (실물은 FR3 내장 τ_ext 외력 추정으로 대응 — FT 센서 없음)
- 들기/운반 보상은 "양 판 동시 접촉" gate 통과 시에만 지급 (dual_fr3 finger gate와 동일 구조)
"""

import os
import torch

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
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, quat_apply, subtract_frame_transforms
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

from dual_fr3_lab.tasks.manager_based.lift_cube import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from dual_fr3_lab.robots.dual_fr3_plate_ee import DUAL_FR3_PLATE_EE_CFG, PLATE_TCP_OFFSET_POS  # isort: skip

# 성능 기반 커리큘럼 (스텝 고정 커리큘럼은 들기 발견 전 발동 시 정책 붕괴 -> dual_fr3에서 검증됨)
from dual_fr3_lab.tasks.manager_based.lift_cube.joint_pos_env_cfg import (  # isort: skip
    modify_reward_weight_on_lifting_learned,
)


# ---------- 배치 상수 (world 기준) ----------
ROOT_Z = 0.405  # base frame 높이 (DUAL_FR3_PLATE_EE_CFG 스폰 포함)
BOX_BASE_SIZE = 0.40  # 기준 박스 크기. scale DR (0.75~1.25) -> 실제 0.30~0.50m
BOX_SCALE_RANGE = (0.30 / BOX_BASE_SIZE, 0.50 / BOX_BASE_SIZE)
BOX_DENSITY = 30.0  # kg/m^3 (0.4박스 1.9kg 기준). 질량 scale DR가 밀도 DR 역할
BOX_MASS_SCALE_RANGE = (0.6, 1.4)  # 밀도 18~42 kg/m^3 상당 (0.3박스 0.5kg ~ 0.5박스 5.3kg)
# 스폰 z는 최대 박스(half 0.25)도 바닥에 안 박히게 잡고, 작은 박스는 리셋 직후 낙하해 안착
BOX_SPAWN_POS_W = (0.5, 0.0, 0.26)
# 들기 판정: 박스 중심이 "안착 중심(env별 반너비) + 이 값"을 넘으면 lifted.
# 이력: P5에서 tipping(기울이기, 최대 중심상승 +10.4cm) 치팅을 막으려 0.20으로 올렸으나
# "무보상 +20cm 완주"가 탐색 범위 밖이라 lifting 발견 자체가 실패 (P5a/b 둘 다 lifting 0).
# P6에서 0.10 원복 — tipping은 발견의 디딤돌로 허용하고, "기울어진 채 유지"는
# box_upright(-2) + misalignment(-4) + min-alignment가 수평 들기로 정제하는 구조
LIFT_DELTA = 0.10


# ---------- 커스텀 MDP 항 ----------
def _box_half_sizes(env, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """env별 박스 반너비 (num_envs,). 크기 DR(mode='usd')이 쓴 xformOp:scale을 startup에 1회 캐싱."""
    cache = getattr(env, "_plate_box_half_cache", None)
    if cache is None:
        stage = env.scene.stage
        scales = []
        for i in range(env.num_envs):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Object")
            attr = prim.GetAttribute("xformOp:scale") if prim else None
            val = attr.Get() if attr and attr.HasValue() else None
            scales.append(float(val[1]) if val is not None else 1.0)
        cache = torch.tensor(scales, device=env.device) * (BOX_BASE_SIZE / 2)
        env._plate_box_half_cache = cache
    return cache


def _ee_lr_indices(ee_frame) -> tuple[int, int]:
    names = ee_frame.data.target_frame_names
    return names.index("left_ee"), names.index("right_ee")


def _face_dists(env, object_cfg: SceneEntityCfg, ee_frame_cfg: SceneEntityCfg):
    """양 판 TCP <-> 담당 측면 중심점 거리와 박스 +y면 법선, 면 중심점을 계산 (공용 헬퍼).

    Returns: (li, ri, y_axis_w, face_l, face_r, d_l, d_r)
    """
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    li, ri = _ee_lr_indices(ee_frame)
    half = _box_half_sizes(env, object_cfg).unsqueeze(-1)
    y_axis = torch.zeros_like(obj.data.root_pos_w)
    y_axis[:, 1] = 1.0
    y_axis_w = quat_apply(obj.data.root_quat_w, y_axis)  # 박스 +y면 법선 (world)
    face_l = obj.data.root_pos_w + y_axis_w * half
    face_r = obj.data.root_pos_w - y_axis_w * half
    d_l = torch.norm(ee_frame.data.target_pos_w[:, li] - face_l, dim=-1)
    d_r = torch.norm(ee_frame.data.target_pos_w[:, ri] - face_r, dim=-1)
    return li, ri, y_axis_w, face_l, face_r, d_l, d_r


def _proximity_factor(d_l, d_r, prox_std: float):
    """양 판이 "둘 다" 측면 중심점 근처일 때만 1에 가까운 factor (곱 형태).

    v5: 파지 품질 항(alignment/misalignment)을 이 factor로 게이팅한다. Run P3에서
    alignment는 제자리에서 공짜로 만점, misalignment -4는 탐색 세금이 되어
    "가만히 정렬만 유지"하는 국소해에 빠짐 (reaching 0.14, 접촉 0) -> 근처에서만 유효화.
    """
    return (1 - torch.tanh(d_l / prox_std)) * (1 - torch.tanh(d_r / prox_std))


def object_dual_ee_face_distance(
    env,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """양 판 TCP와 박스 "양 측면 중심점" 사이 거리의 tanh-kernel 평균.

    왼판은 +y면 중심, 오른판은 -y면 중심(박스 yaw 반영, env별 반너비 반영)을 목표로 해서
    "양옆에서 마주 보고 꼬집는" 접근을 유도한다. 박스 중심 기준이었던 v1은 위/모서리
    어디로 접근해도 같은 보상이라 모서리 걸침이 나왔음 (Run P1에서 관찰).
    """
    _, _, _, _, _, d_l, d_r = _face_dists(env, object_cfg, ee_frame_cfg)
    return 1 - 0.5 * (torch.tanh(d_l / std) + torch.tanh(d_r / std))


def plate_face_alignment(
    env,
    prox_std: float = 0.15,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """판 법선(TCP z축)이 박스 해당 측면 법선과 반평행일수록 1 (면-면 접촉 자세 유도).

    왼판(+y면 담당)은 법선이 박스 -y 방향, 오른판은 +y 방향이어야 함. dot을 [0,1]로
    clamp해서 반대 방향 자세엔 gradient만 없고 페널티는 없음.
    v5: 근접 factor 곱 — 박스에서 멀면 0 (제자리 공짜 보상 차단, P3 국소해 원인).
    v6: mean -> min — P4에서 mean은 왼판만 완벽해도 절반을 받아 오른판 정렬이 포기됨
    (alignment tail 0.078, play에서 오른판만 기울어짐). min은 못하는 팔이 보상을 결정
    -> "한쪽 포기" 차단 + 이 값 자체가 "더 못하는 팔" 지표 (arXiv:2310.03191 대조로 근거 확인).
    """
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    li, ri, y_axis_w, _, _, d_l, d_r = _face_dists(env, object_cfg, ee_frame_cfg)
    z_axis = torch.zeros_like(obj.data.root_pos_w)
    z_axis[:, 2] = 1.0
    n_l = quat_apply(ee_frame.data.target_quat_w[:, li], z_axis)  # 왼판 작업면 법선
    n_r = quat_apply(ee_frame.data.target_quat_w[:, ri], z_axis)
    a_l = (-(n_l * y_axis_w).sum(dim=-1)).clamp(0.0, 1.0)
    a_r = ((n_r * y_axis_w).sum(dim=-1)).clamp(0.0, 1.0)
    return torch.minimum(a_l, a_r) * _proximity_factor(d_l, d_r, prox_std)


def plates_pair_misalignment(
    env,
    prox_std: float = 0.15,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """두 판의 높이차 |z_L - z_R| + 전후차 |x_L - x_R| (m). 마주 보고 꼬집게 하는 페널티.

    v5: 근접 factor 곱 — 접근/탐색 단계의 비대칭에는 세금을 물리지 않고 (P3 국소해 원인),
    잡는 순간의 품질에만 작동.
    P6 (yaw-aware 버그픽스): 기존엔 world 축 기준 |x_L-x_R|+|z_L-z_R|이라, yaw 회전한 박스를
    "올바르게" 잡으면(면 중심점끼리 world x로 2*half*sin(yaw) 차이) 오히려 벌금을 물어
    reaching_fine과 모순 -> 회전 박스 파지 실패 (P5b play에서 관찰). 이상적인 쌍 벡터
    (face_L - face_R, 박스와 함께 회전)를 기준으로 한 편차로 교체 — yaw 0이면 기존과 동일.
    """
    ee_frame = env.scene[ee_frame_cfg.name]
    li, ri, _, face_l, face_r, d_l, d_r = _face_dists(env, object_cfg, ee_frame_cfg)
    p = ee_frame.data.target_pos_w
    pair_error = (p[:, li] - p[:, ri]) - (face_l - face_r)
    misalign = pair_error[:, 0].abs() + pair_error[:, 2].abs()
    return misalign * _proximity_factor(d_l, d_r, prox_std)


def object_orientation_in_robot_root_frame(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """박스 orientation (robot root frame 기준 quat, w>=0 부호 고정) (num_envs, 4).

    P5 추가: yaw DR(+-0.3)이 있는데 policy가 박스 회전을 관측할 수 없으면 min-alignment가
    관측 불가능한 목표가 됨. 실물은 ArUco 마커 6DoF 추정으로 대응 (orientation 오차가
    위치보다 커서 noise sigma를 크게 걸어둠). w>=0 고정은 q ≡ -q 불연속 제거용.
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    _, quat = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, obj.data.root_pos_w, obj.data.root_quat_w
    )
    sign = torch.where(quat[:, 0:1] >= 0.0, 1.0, -1.0)
    return quat * sign


def box_upright(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """박스 기울기 페널티: 1 - z_box·z_world (0=수평 ~ 2=뒤집힘). 페널티 weight로 사용.

    v6: 들고 나르는 중 박스가 회전한 채 운반되는 문제 대응. 문헌(arXiv:2310.03191)의
    |roll|+|pitch| 수평 유지 항에 해당. 바닥 안착 시 자연히 0이라 gate 불필요, yaw는 자유.
    """
    obj = env.scene[object_cfg.name]
    z_axis = torch.zeros_like(obj.data.root_pos_w)
    z_axis[:, 2] = 1.0
    box_z = quat_apply(obj.data.root_quat_w, z_axis)
    return 1.0 - box_z[:, 2]


def _lifted_perenv(env, lift_delta: float, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """env별 들기 판정: 박스 중심 z > 안착 중심(반너비) + lift_delta. 크기 DR 공평화 (§3-3)."""
    obj = env.scene[object_cfg.name]
    half = _box_half_sizes(env, object_cfg)
    return (obj.data.root_pos_w[:, 2] > half + lift_delta).float()


def object_is_lifted_perenv_and_squeezed(
    env,
    lift_delta: float,
    finger1_sensor_cfg: SceneEntityCfg,
    finger2_sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """env별 높이 gate x 양판 squeeze gate. 고정 높이(0.35)는 작은 박스에 +20cm를 요구했음."""
    gate = mdp.object_grasped(env, finger1_sensor_cfg, finger2_sensor_cfg, threshold)
    return _lifted_perenv(env, lift_delta, object_cfg) * gate


def object_goal_distance_perenv_squeezed(
    env,
    std: float,
    lift_delta: float,
    command_name: str,
    finger1_sensor_cfg: SceneEntityCfg,
    finger2_sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """goal 거리 tanh 보상 x env별 높이 gate x 양판 squeeze gate."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)
    gate = mdp.object_grasped(env, finger1_sensor_cfg, finger2_sensor_cfg, threshold)
    return _lifted_perenv(env, lift_delta, object_cfg) * gate * (1 - torch.tanh(distance / std))


def plate_contact_forces(
    env,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_plate_object_contact"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_plate_object_contact"),
    scale: float = 0.02,
) -> torch.Tensor:
    """양 판이 박스에 가하는 접촉력 크기 (num_envs, 2). 실물은 FR3 τ_ext 외력 추정(O_F_ext_hat_K)으로 대응.

    PhysX는 sensor body당 filter 1개를 요구하므로 판별로 센서가 분리되어 있다.
    scale로 대략 [0,1] 범위로 정규화 (50N -> 1.0).
    """
    forces = []
    for cfg in (left_sensor_cfg, right_sensor_cfg):
        sensor = env.scene.sensors[cfg.name]
        # force_matrix_w: (num_envs, 1, 1, 3)
        forces.append(torch.norm(sensor.data.force_matrix_w, dim=-1).amax(dim=(-1, -2)))
    return torch.stack(forces, dim=-1) * scale


@configclass
class DualFR3PlateBoxLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # PhysX GPU aggregate pair 버퍼 확대 (cube에서 4096 env 동시 접촉 시 용량 초과로
        # 접촉 누락이 터진 이슈의 예방 적용 — plate도 양판 squeeze 동시 접촉 구조라 동일 리스크)
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024

        # ---------- 로봇 ----------
        self.scene.robot = DUAL_FR3_PLATE_EE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ---------- 씬: 테이블 없음, 바닥 위 로봇 + 바닥 위 박스 ----------
        self.scene.table = None
        self.scene.plane.init_state.pos = (0.0, 0.0, 0.0)  # 바닥을 z=0으로 (franka는 -1.05)

        # ---------- DR 범위 시각화 (시각 전용, 충돌체 없음 -> 물리/학습 영향 없음) ----------
        # 토글: `DR_VIS=0 python ... play.py ...` 로 실행하면 시각화 판을 생성하지 않음 (기본 표시)
        # 주의: RTX Real-Time 렌더러는 opacity<0.5를 cutout으로 잘라 안 보이므로 불투명 판 사용
        dr_vis = os.environ.get("DR_VIS", "1") != "0"
        # 파란 판: 박스 스폰 범위 (중심 x 0.45~0.60, y ±0.10) + 최대 박스(0.5m) 몸체 여유
        self.scene.dr_spawn_zone = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/DrSpawnZone",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.525, 0.0, 0.001)),
            spawn=sim_utils.MeshCuboidCfg(
                size=(0.15 + 0.50, 0.20 + 0.50, 0.002),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 0.9)),
            ),
        )
        # 초록 판 2장: goal 샘플 범위의 아래/위 경계 (박스 "중심" 기준 + 몸체 여유)
        # base 기준 x 0.45~0.60, y ±0.10, z 0.15~0.35 -> world z 0.555~0.755
        self.scene.dr_target_zone_bottom = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/DrTargetZoneBottom",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.525, 0.0, 0.15 + ROOT_Z)),
            spawn=sim_utils.MeshCuboidCfg(
                size=(0.15 + 0.50, 0.20 + 0.50, 0.003),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.3)),
            ),
        )
        self.scene.dr_target_zone_top = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/DrTargetZoneTop",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.525, 0.0, 0.35 + ROOT_Z)),
            spawn=sim_utils.MeshCuboidCfg(
                size=(0.15 + 0.50, 0.20 + 0.50, 0.003),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.3)),
            ),
        )
        if not dr_vis:
            self.scene.dr_spawn_zone = None
            self.scene.dr_target_zone_bottom = None
            self.scene.dr_target_zone_top = None

        # ---------- 양팔 고정 자세 유지 ----------
        # 리셋 시 joint target을 default(ready)로 초기화 (초기 target 0으로 리밋에 휩쓸리는 것 방지)
        self.events.reset_all.params["reset_joint_targets"] = True

        # ---------- action: 양팔 14 joint delta (그리퍼 없음) ----------
        # relative(delta) 방식: contact-rich task에서 조임 유지가 부드럽고(스텝당 0.1rad 제한),
        # ready ±0.5rad 범위 제약이 없어 바닥 박스까지 도달 가능 (공식 dexsuite와 동일 설정)
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=[".*_fr3_joint[1-7]"], scale=0.1
        )
        self.actions.gripper_action = None

        # ---------- observation: 기본(joint 14개 전부) + 양 판 접촉력 ----------
        self.observations.policy.plate_forces = ObsTerm(func=plate_contact_forces, noise=Gnoise(std=0.04))
        # 박스 orientation (P5): yaw DR 관측 가능화 + min-alignment 목표의 관측성 확보.
        # 실물은 ArUco 6DoF 추정 대응 — 각도 오차가 위치보다 커서 noise σ0.02(~2°) 적용
        self.observations.policy.object_orientation = ObsTerm(
            func=object_orientation_in_robot_root_frame, noise=Gnoise(std=0.02)
        )

        # ---------- obs noise (v3): 실물 센서/인식 수준 — cube task Run C와 동일 근거 ----------
        # 강건성 진단(cube, 2026-07-27): noise 단독/물성 DR 단독은 drop 0인데 "조합"만 슬립 드랍
        # -> 두 축을 반드시 함께 학습에 넣는다. 마찰 파지인 이 task는 더 민감할 것
        self.observations.policy.joint_pos.noise = Gnoise(std=0.005)  # rad, 엔코더+캘리브레이션
        self.observations.policy.joint_vel.noise = Gnoise(std=0.05)  # rad/s
        self.observations.policy.object_position.noise = Gnoise(std=0.01)  # m, perception 1cm
        # plate_forces noise=0.04 (scale 0.02 기준 2N 상당) — FR3 tau_ext 추정 오차 대응

        # ---------- command: 박스를 가져갈 목표 pose (base frame 기준) ----------
        self.commands.object_pose.body_name = "base"
        self.commands.object_pose.ranges.pos_x = (0.45, 0.60)
        self.commands.object_pose.ranges.pos_y = (-0.10, 0.10)
        # 목표는 박스 "중심" 기준. 안착 중심(0.15~0.25)에서 0.3~0.5m 상승 수준으로 설정
        # (0.25~0.45는 최대 0.7m 상승이라 과도 -> rule-based 데모 검증 범위 0.55 부근을 포함하게 완화)
        self.commands.object_pose.ranges.pos_z = (0.15, 0.35)  # world 0.555~0.755
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

        # ---------- object: 바닥 위 박스 (크기/밀도 DR) ----------
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=BOX_SPAWN_POS_W, rot=[1, 0, 0, 0]),
            spawn=sim_utils.MeshCuboidCfg(
                size=(BOX_BASE_SIZE, BOX_BASE_SIZE, BOX_BASE_SIZE),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_depenetration_velocity=5.0,
                ),
                # density 지정 -> scale DR 후 부피에 비례해 질량이 정해짐
                mass_props=sim_utils.MassPropertiesCfg(density=BOX_DENSITY),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                # 판-박스 마찰이 유일한 파지력 (판 collider는 기본 재질 0.5와 combine)
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.5, 0.2)),
            ),
        )
        # 리셋 시 박스 위치/yaw 랜덤화 (양팔 공용 workspace라 y는 좌우 대칭)
        # yaw ±0.3 rad: 판이 살짝 기울어진 면을 조이는 상황을 경험시킴 (§4-3 -> 반영)
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.05, 0.10),
            "y": (-0.10, 0.10),
            "z": (0.0, 0.0),
            "yaw": (-0.3, 0.3),
        }
        # 박스 크기 DR: 30~50cm. mode="usd": 시뮬 시작 전 1회 -> env별 크기가 학습 내내 유지
        self.events.randomize_box_size = EventTerm(
            func=mdp.randomize_rigid_body_scale,
            mode="usd",
            params={"scale_range": BOX_SCALE_RANGE, "asset_cfg": SceneEntityCfg("object")},
        )
        # 밀도 DR: density 기반 질량을 startup에서 env별로 scale (관성도 재계산)
        self.events.randomize_box_density = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "mass_distribution_params": BOX_MASS_SCALE_RANGE,
                "operation": "scale",
                "recompute_inertia": True,
            },
        )
        # 마찰 DR: 마찰이 파지력의 전부인 task (§4-5 -> 반영). 판은 USD에 재질 명시(정지 0.6)
        # v3: cube task Run C와 같은 폭(0.4~1.2)으로 확대 -> 유효 static(average) 0.5~0.9
        self.events.randomize_box_friction = EventTerm(
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
        # joint 물성 DR (v3): 액추에이터 PD 게인 ±15% — 실물 게인/마찰 불확실성 대응.
        # delta action이라 게인 변화 = 같은 명령에 다른 응답 -> policy가 피드백으로 보정하게 학습
        self.events.randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_fr3_joint[1-7]"]),
                "stiffness_distribution_params": (0.85, 1.15),
                "damping_distribution_params": (0.85, 1.15),
                "operation": "scale",
            },
        )
        # env별로 다른 scale/질량의 물리 속성을 개별 파싱하도록 필수 설정
        self.scene.replicate_physics = False

        # ---------- squeeze 판정 센서: 판(link7 body)과 박스 사이 접촉력만 필터링 ----------
        # PhysX는 sensor body당 filter 1개를 요구하므로 판별로 센서를 분리한다.
        # 양 판이 동시에 박스를 눌러야 파지로 인정 (판 collider는 link7 body 소속)
        self.scene.left_plate_object_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_fr3_link7",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        self.scene.right_plate_object_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_fr3_link7",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        # 양팔 링크 충돌 감지 (link7=판 제외 — 박스를 만져야 하는 부위): 팔끼리/바닥/박스 팔뚝 접촉 등
        # 모든 접촉의 net force를 본다 (filter 없음 -> body 수 제약 없음). cube Run B와 동일 패턴
        self.scene.arms_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_fr3_link[1-6]",
        )

        # ---------- reward: 양팔 접근 + 양판 접촉 + squeeze-gate 들기/운반 ----------
        squeeze_gate_params = {
            "finger1_sensor_cfg": SceneEntityCfg("left_plate_object_contact"),
            "finger2_sensor_cfg": SceneEntityCfg("right_plate_object_contact"),
            "threshold": 1.0,
        }
        # 접근: 양 판 -> 박스 "양 측면 중심점" 거리 평균 (마주 보고 꼬집는 접근 유도, v2)
        self.rewards.reaching_object.func = object_dual_ee_face_distance
        self.rewards.reaching_object.params = {"std": 0.3}
        self.rewards.reaching_object.weight = 2.0
        # 중심 정밀 접근 (v4): std 0.08의 날카로운 kernel — P2에서 모서리 잡기 잔존
        # (reaching 0.61/2, misalignment 악화 -0.23->-0.38: 중심 파지의 기회비용 ~1.5/s로 너무 쌌음)
        self.rewards.reaching_fine = RewTerm(
            func=object_dual_ee_face_distance, params={"std": 0.08}, weight=3.0
        )
        # 판 법선-박스 면 법선 정렬 (면-면 접촉 자세 유도, v2 핵심 — 모서리 걸침 방지)
        # v5: 근접 게이팅 (prox_std 0.15) — P3에서 제자리 공짜 만점이 "안 움직이는" 국소해를 만듦
        # v6: min-form + weight 2 — P4에서 mean/w1은 오른판 정렬 포기를 허용 (tail 0.078)
        self.rewards.plate_alignment = RewTerm(func=plate_face_alignment, params={"prox_std": 0.15}, weight=2.0)
        # 두 판의 높이/전후 어긋남 페널티 (마주 보는 꼬집기 강제, v4에서 -4)
        # v5: 근접 게이팅 — 접근 단계 탐색(순간 비대칭)에 세금을 물리지 않음
        self.rewards.plates_misalignment = RewTerm(
            func=plates_pair_misalignment, params={"prox_std": 0.15}, weight=-4.0
        )
        # 박스 수평 유지 (v6): 기울어진 채 운반 방지. 문헌의 roll/pitch 페널티 상당, yaw 자유
        self.rewards.box_upright = RewTerm(func=box_upright, weight=-2.0)
        # 양판 동시 접촉 자체에 보상 (squeeze 발견 가속)
        self.rewards.squeezing_object = RewTerm(func=mdp.object_grasped, params=dict(squeeze_gate_params), weight=2.0)
        # 들기/운반 보상 전부 "양판 접촉 중"일 때만 지급 (밀어서 옮기는 exploit 차단)
        # 높이 gate는 env별(안착 중심 + LIFT_DELTA): 고정 0.35는 작은 박스에 2배 불리했음 (v3)
        self.rewards.lifting_object.func = object_is_lifted_perenv_and_squeezed
        self.rewards.lifting_object.params = {"lift_delta": LIFT_DELTA, **squeeze_gate_params}
        self.rewards.object_goal_tracking.func = object_goal_distance_perenv_squeezed
        self.rewards.object_goal_tracking.params = {
            "std": 0.3,
            "lift_delta": LIFT_DELTA,
            "command_name": "object_pose",
            **squeeze_gate_params,
        }
        self.rewards.object_goal_tracking_fine_grained.func = object_goal_distance_perenv_squeezed
        self.rewards.object_goal_tracking_fine_grained.params = {
            "std": 0.05,
            "lift_delta": LIFT_DELTA,
            "command_name": "object_pose",
            **squeeze_gate_params,
        }

        # ---------- termination: 박스는 바닥 위라 낙하 종료가 무의미 -> timeout만 사용 ----------
        self.terminations.object_dropping = None

        # 충돌 페널티 (v3): 팔끼리/바닥/팔뚝-박스 접촉. 초기엔 -1e-4로 약하게 (탐색 방해 방지),
        # 들기 습득 후 curriculum으로 강화 — cube Run B에서 검증된 구조 (v1 교훈: 조기 강화 = 붕괴)
        self.rewards.undesired_contact_penalty = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1e-4,
            params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("arms_contact")},
        )

        # ---------- curriculum: "들기를 배웠는가" 기반 페널티 강화 ----------
        self.curriculum.action_rate = CurrTerm(
            func=modify_reward_weight_on_lifting_learned,
            params={"term_name": "action_rate", "weight": -1e-1, "threshold": 10.0},
        )
        self.curriculum.joint_vel = CurrTerm(
            func=modify_reward_weight_on_lifting_learned,
            params={"term_name": "joint_vel", "weight": -1e-1, "threshold": 10.0},
        )
        self.curriculum.contact_penalty = CurrTerm(
            func=modify_reward_weight_on_lifting_learned,
            params={"term_name": "undesired_contact_penalty", "weight": -1.0, "threshold": 10.0},
        )

        # ---------- ee frame: 양쪽 link7 + 판 작업면 오프셋 ----------
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/left_fr3_link7",
                    name="left_ee",
                    offset=OffsetCfg(pos=PLATE_TCP_OFFSET_POS),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_fr3_link7",
                    name="right_ee",
                    offset=OffsetCfg(pos=PLATE_TCP_OFFSET_POS),
                ),
            ],
        )


@configclass
class DualFR3PlateBoxLiftEnvCfg_PLAY(DualFR3PlateBoxLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
