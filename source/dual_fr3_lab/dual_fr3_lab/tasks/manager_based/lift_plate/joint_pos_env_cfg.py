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
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
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
def _box_halves(env, object_cfg: SceneEntityCfg = SceneEntityCfg("object")):
    """env별 박스 (반폭_y, 반높이_z) 튜플. 크기 DR(mode='usd')의 xformOp:scale을 startup에 1회 캐싱.

    v1(직육면체 475x555x275)부터 y(잡는 방향 - 면 오프셋)와 z(높이 - 들기 판정)를 분리.
    기준 크기는 env cfg의 object.spawn.size에서 읽으므로 v0(정육면체)/v1 공용.
    """
    cache = getattr(env, "_plate_box_halves_cache", None)
    if cache is None:
        stage = env.scene.stage
        size = env.cfg.scene.object.spawn.size  # (x, y, z) 기준 크기
        hy, hz = [], []
        for i in range(env.num_envs):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Object")
            attr = prim.GetAttribute("xformOp:scale") if prim else None
            val = attr.Get() if attr and attr.HasValue() else None
            sy = float(val[1]) if val is not None else 1.0
            sz = float(val[2]) if val is not None else 1.0
            hy.append(sy * size[1] / 2.0)
            hz.append(sz * size[2] / 2.0)
        cache = (torch.tensor(hy, device=env.device), torch.tensor(hz, device=env.device))
        env._plate_box_halves_cache = cache
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
    half = _box_halves(env, object_cfg)[0].unsqueeze(-1)  # 잡는 방향(y) 반폭
    y_axis = torch.zeros_like(obj.data.root_pos_w)
    y_axis[:, 1] = 1.0
    y_axis_w = quat_apply(obj.data.root_quat_w, y_axis)  # 박스 +y면 법선 (world)
    face_l = obj.data.root_pos_w + y_axis_w * half
    face_r = obj.data.root_pos_w - y_axis_w * half
    d_l = torch.norm(ee_frame.data.target_pos_w[:, li] - face_l, dim=-1)
    d_r = torch.norm(ee_frame.data.target_pos_w[:, ri] - face_r, dim=-1)
    return li, ri, y_axis_w, face_l, face_r, d_l, d_r


def _proximity_factor(d_l, d_r, prox_std: float):
    """[P11부터 미사용 - 이력 보존용] 양 판이 "둘 다" 측면 중심점 근처일 때만 1인 factor.

    v5: 파지 품질 항(alignment/misalignment)을 이 factor로 게이팅했다 (P3 "제자리 공짜 보상"
    국소해 차단). P10에서 비용 발견: 오른판이 면 중심을 벗어나면 prox->0이라 "잡고 있는데
    기울어진" 상태를 고칠 gradient까지 소멸 (alignment 0.034 고착, 오른판 비틀림 방치).
    P11에서 접촉 게이트(G)로 교체 - G도 P3 함정을 동일하게 막으면서(접촉 없인 발동 불가)
    잡는 중에는 위치와 무관하게 방향 gradient가 살아있음. v5 당시 근접/접촉을 비교 검토 없이
    근접을 선택했던 것이 미문서화 결정이었음 (2026-07-29 사용자 검증에서 발견).
    """
    return (1 - torch.tanh(d_l / prox_std)) * (1 - torch.tanh(d_r / prox_std))


def object_dual_ee_face_distance(
    env,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """양 판 TCP와 박스 "양 측면 중심점" 사이 거리 kernel의 min (coarse reaching, 접근).

    왼판은 +y면 중심, 오른판은 -y면 중심(박스 yaw 반영, env별 반너비 반영)을 목표로 해서
    "양옆에서 마주 보고 꼬집는" 접근을 유도한다.
    P15: mean -> min — "양 팔 모두" 접근해야 함 (mean은 한 팔만 도착해도 절반 대납 -> 다른 팔
    포기 허용, P4 오른판 비틀림 계열). min은 못 온 팔이 보상을 결정.
    """
    _, _, _, _, _, d_l, d_r = _face_dists(env, object_cfg, ee_frame_cfg)
    k_l = 1 - torch.tanh(d_l / std)
    k_r = 1 - torch.tanh(d_r / std)
    return torch.minimum(k_l, k_r)


def reaching_aligned(
    env,
    std: float = 0.3,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """[07302019 수정] 접근×정렬 통합: min_i[(1−tanh(d_i/std)) · align_i]. std 0.3(넓음).

    P16까지 정렬(plate_alignment)을 별도 항으로 6번 시도 다 실패 (게이트有=무시/게이트無=P3).
    근본: reaching과 alignment가 덧셈이라 "각각 따로" 챙길 수 있어 동시 달성이 안 됨 (P16 스크린샷:
    판 완벽 평행한데 허공에 떠서 접근 0). -> 팔별 곱(거리×정렬)의 min으로 통합:
    ① 곱셈이라 "가깝고 AND 평행"만 보상 (멀면 거리≈0→P3 차단, 안평행이면 align≈0→평행 강제)
    ② std 0.3(넓음)이라 침투 파지 중에도 거리 factor 안 죽음 (A안 P15는 std 0.08이라 파지 중 사망)
       -> 거리 factor 자체가 자연 게이트라 별도 접촉 게이트 불필요
    ③ min_i: 팔별 곱을 먼저 계산 후 min = 가장 못하는 팔(멀거나 안평행)을 집어 몰아줌 (P4 포기 방지)
    리스크: 곱셈이라 초기 탐색 어려울 수 있음(접근+평행 동시). std 0.3이 관대해 완화되나, 발견 늦으면
    완충곱(0.5+0.5·align)이 첫 조정 카드. 원복: 이 항 대신 reaching(w2)+reaching_fine(w3)+plate_alignment(w6).
    """
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    li, ri, y_axis_w, _, _, d_l, d_r = _face_dists(env, object_cfg, ee_frame_cfg)
    z_axis = torch.zeros_like(obj.data.root_pos_w)
    z_axis[:, 2] = 1.0
    n_l = quat_apply(ee_frame.data.target_quat_w[:, li], z_axis)  # 왼판 작업면 법선
    n_r = quat_apply(ee_frame.data.target_quat_w[:, ri], z_axis)
    a_l = (-(n_l * y_axis_w).sum(dim=-1)).clamp(0.0, 1.0)  # 왼판: -y 방향 (박스 +y면 반평행)
    a_r = ((n_r * y_axis_w).sum(dim=-1)).clamp(0.0, 1.0)  # 오른판: +y 방향
    v_l = (1 - torch.tanh(d_l / std)) * a_l  # 팔별: 거리 kernel × 평행
    v_r = (1 - torch.tanh(d_r / std)) * a_r
    return torch.minimum(v_l, v_r)


def plate_alignment_min(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """판 평행도(면 법선 반평행)를 별도 항으로, min(양 팔). 게이트 없음 (P16, 2026-07-30).

    이력: 정렬 항은 게이트를 근접(P, P10)/접촉(G, P11)/거리(0.15, P15-v2) 셋 다 붙여봤으나
    전부 붕괴. 5번 반복 후 결론 = "게이트 문제가 아니라 경제학 문제" (정렬 안 해도 sim success
    75% 나와 w2를 무시). P16: weight 2->6 상향 + 거리 게이트 제거.
    - 게이트 제거 근거(사용자 결정): "허공에서부터 판이 평행하게 접근"이 목표 거동. 거리게이트는
      "면 근처에서만 정렬"이라 오히려 접근 중 정렬을 막음. 지금은 lifting/squeezing이 접촉 G 없인
      안 나와서, 정렬(6점)만 허공에서 따먹고 나머지 69점 포기할 유인 없음(P3는 발견 전이라 정렬이
      유일 보상이었던 것) -> 무게이트 안전.
    - min(양 팔): mean은 한쪽 포기 허용(P4). 게이트 없어도 min은 유지.
    P16 실패 신호: 초반(it500) "정렬 오르는데 reaching/lifting 안 오름" = P3 재발 -> G 게이트로 전환.
    """
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    li, ri, y_axis_w, _, _, _, _ = _face_dists(env, object_cfg, ee_frame_cfg)
    z_axis = torch.zeros_like(obj.data.root_pos_w)
    z_axis[:, 2] = 1.0
    n_l = quat_apply(ee_frame.data.target_quat_w[:, li], z_axis)  # 왼판 작업면 법선
    n_r = quat_apply(ee_frame.data.target_quat_w[:, ri], z_axis)
    a_l = (-(n_l * y_axis_w).sum(dim=-1)).clamp(0.0, 1.0)  # 왼판: -y 방향 (박스 +y면 반평행)
    a_r = ((n_r * y_axis_w).sum(dim=-1)).clamp(0.0, 1.0)  # 오른판: +y 방향
    return torch.minimum(a_l, a_r)


def plate_face_alignment(
    env,
    finger1_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_plate_object_contact"),
    finger2_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_plate_object_contact"),
    threshold: float = 1.0,
    upright_std: float = 0.034,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """판 법선(TCP z축)이 박스 해당 측면 법선과 반평행일수록 1 (면-면 접촉 자세 유도).

    왼판(+y면 담당)은 법선이 박스 -y 방향, 오른판은 +y 방향이어야 함. dot을 [0,1]로
    clamp해서 반대 방향 자세엔 gradient만 없고 페널티는 없음.
    v5: 근접 factor 게이팅 (P3 "제자리 공짜 보상" 국소해 차단).
    v6: mean -> min — P4에서 mean은 왼판만 완벽해도 절반을 받아 오른판 정렬이 포기됨.
    P11: 게이트를 근접(P) -> 접촉(G, 양판 동시 threshold N)으로 교체. 근접 게이트는 판이
    면 중심을 벗어나면 방향 gradient까지 꺼지는 함정이 있었음 (P10: 잡은 채 기울어진
    오른판 방치, alignment 0.034 고착). 접촉 게이트는 P3 함정을 동일하게 막으면서
    "잡고 있는 동안"은 위치와 무관하게 방향 압력이 유지됨.
    P12: upright_factor 곱 (upright_std로 폭 지정). alignment는 판↔박스 "면" 상대 자세만
    보므로, 박스째 기울여 판을 기울어진 면에 붙이면 alignment 만점을 받는 "기울인
    세트-그립" exploit이 나옴 (P11: alignment 1.05인데 box 기울기 ~29도, success 4% 붕괴).
    수평 factor(수평 1 ~ upright_std에서 0)를 곱해 기울인 밀착은 alignment도 0 -> upright와
    alignment의 시소를 없애고 "수평 + 면-면 파지"만 유일 최적점으로.
    """
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    li, ri, y_axis_w, _, _, _, _ = _face_dists(env, object_cfg, ee_frame_cfg)
    z_axis = torch.zeros_like(obj.data.root_pos_w)
    z_axis[:, 2] = 1.0
    n_l = quat_apply(ee_frame.data.target_quat_w[:, li], z_axis)  # 왼판 작업면 법선
    n_r = quat_apply(ee_frame.data.target_quat_w[:, ri], z_axis)
    a_l = (-(n_l * y_axis_w).sum(dim=-1)).clamp(0.0, 1.0)
    a_r = ((n_r * y_axis_w).sum(dim=-1)).clamp(0.0, 1.0)
    gate = mdp.object_grasped(env, finger1_sensor_cfg, finger2_sensor_cfg, threshold)
    # 수평 factor: box_upright = 1 - z_box·z_world (0=수평). success 조건과 같은 15도(0.034)에서 0
    upright_factor = (1.0 - box_upright(env, object_cfg) / upright_std).clamp(0.0, 1.0)
    return torch.minimum(a_l, a_r) * gate * upright_factor


def plates_pair_misalignment(
    env,
    finger1_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_plate_object_contact"),
    finger2_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_plate_object_contact"),
    threshold: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """두 판의 이상 쌍 배치 대비 편차 (m). 마주 보고 꼬집게 하는 페널티.

    v5: 근접 게이팅 — 접근/탐색 단계의 비대칭에는 세금을 물리지 않음 (P3 국소해 방지).
    P6 (yaw-aware): 이상 쌍 벡터(face_L - face_R, 박스와 함께 회전) 기준 편차 — world 축
    고정이던 버그 수정 (회전 박스를 올바르게 잡으면 벌금 물던 모순 제거).
    P11: 게이트 근접(P) -> 접촉(G) 교체 (plate_face_alignment와 동일 근거 - 잡는 중
    어긋난 배치에 페널티가 계속 작동해야 오른판 이탈이 무과세가 되지 않음).
    """
    ee_frame = env.scene[ee_frame_cfg.name]
    li, ri, _, face_l, face_r, _, _ = _face_dists(env, object_cfg, ee_frame_cfg)
    p = ee_frame.data.target_pos_w
    pair_error = (p[:, li] - p[:, ri]) - (face_l - face_r)
    misalign = pair_error[:, 0].abs() + pair_error[:, 2].abs()
    gate = mdp.object_grasped(env, finger1_sensor_cfg, finger2_sensor_cfg, threshold)
    return misalign * gate


def plate_face_z_offset(
    env,
    finger1_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_plate_object_contact"),
    finger2_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_plate_object_contact"),
    threshold: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """각 판 TCP와 담당 box face 중심점 사이의 절대 z-오프셋 합산 (G-gate).

    plates_pair_misalignment의 z항은 (L_z - R_z) - (face_L_z - face_R_z), 즉 상대 높이 차만 봄.
    양팔이 동시에 같은 높이로 높아지면 pair_error_z≈0 → 기존 함수가 포착 불가.
    이 함수는 각 판 TCP의 담당 면 중심점 절대 z-오프셋을 per-arm으로 합산:
      z_err_L = |EE_L_z - face_L_z|
      z_err_R = |EE_R_z - face_R_z|
    → 양팔이 모두 높거나 낮아도 동일하게 패널티 부여.
    G-gate: 파지 전 접근 phase에서는 비활성 (P3 국소해 방지).
    """
    ee_frame = env.scene[ee_frame_cfg.name]
    li, ri, _, face_l, face_r, _, _ = _face_dists(env, object_cfg, ee_frame_cfg)
    p = ee_frame.data.target_pos_w
    z_err_l = (p[:, li, 2] - face_l[:, 2]).abs()
    z_err_r = (p[:, ri, 2] - face_r[:, 2]).abs()
    gate = mdp.object_grasped(env, finger1_sensor_cfg, finger2_sensor_cfg, threshold)
    return (z_err_l + z_err_r) * gate


def plates_approach_symmetry(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    proximity: float = 0.05,
) -> torch.Tensor:
    """[P37] 양 판이 담당 면에 '동시에' 닿도록 유도: 근접 시 |d_left - d_right| 페널티 (접촉 타이밍 대칭).

    yaw origin 차단: 한 판이 먼저 닿으면 반대편 힘 없이 박스를 밀어 회전(yaw) -> align/grasp이 그 틀어진
    상태를 고정(35° 락인, play 관찰). 동시 터치다운이면 두 법선력이 상쇄 -> 순 토크 0 -> yaw 애초에 안 생김.
    plates_misalignment(위치 편차, G-gate=파지 후)와 상보: 이건 '접근 타이밍'(면까지 거리 차)을 봄.
    근접 게이트(가까운 판이 proximity 이내일 때만)로 먼 접근은 무과세 -> 마지막 동기화 구간만 -> P3(대칭
    hover) 완화. 페널티라 대칭이면 0(hover 유인 없음), 비대칭(한 판만 붙음)만 벌금. 파지 후엔 d_l~d_r~0이라 ~0.
    """
    _, _, _, _, _, d_l, d_r = _face_dists(env, object_cfg, ee_frame_cfg)
    gate = torch.minimum(d_l, d_r) < proximity  # 가까운 판이 접촉 임박 = 마지막 접근 구간
    return torch.where(gate, (d_l - d_r).abs(), torch.zeros_like(d_l))


def one_sided_plate_contact(
    env,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_plate_object_contact"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_plate_object_contact"),
    threshold: float = 1.0,
    filter_idx: int = 0,
) -> torch.Tensor:
    """[P39] 정확히 한 판만 박스와 접촉한 control step을 1로 반환한다.

    별도 timestamp/latch 없이 RewardManager가 매 step ``weight * dt``로 적분하므로 이 raw XOR가
    one-sided contact 지속시간을 직접 비용화한다. 최초 접촉 시차뿐 아니라 운반 중 한쪽 접촉을
    잃는 구간도 포함한다. 두 판 모두 비접촉/접촉이면 0이라 좌우 어느 쪽도 선호하지 않는다.

    ``force_matrix_w``에서 Object filter만 명시적으로 선택한다. 이후 센서에 Table 등 다른 filter가
    추가돼도 그 접촉을 박스 접촉으로 오인하지 않기 위함이다.
    """
    contact_flags = []
    for cfg in (left_sensor_cfg, right_sensor_cfg):
        sensor = env.scene.sensors[cfg.name]
        force = torch.norm(sensor.data.force_matrix_w[:, :, filter_idx, :], dim=-1).amax(dim=-1)
        contact_flags.append(force > threshold)
    return torch.logical_xor(contact_flags[0], contact_flags[1]).float()


def j6_mirror_loss(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    finger1_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_plate_object_contact"),
    finger2_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_plate_object_contact"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """[P40] j6(손목 pitch) 좌우 비대칭 |q_L_j6 - q_R_j6| (rad) 페널티 — G 게이트.

    eval 진단(2026-08-10): hold 중 j6 mirror error 중간값 13.87°, p95 31.21°.
    j6 비대칭이 plate face의 수직 기울기를 만들어 부분 면접촉(~2/3)을 유발.
    sign=+1(대칭 조건 q_R=q_L)이므로 단순 절댓값 차가 mirror error.
    G 게이트(양판 접촉 중)에서만 적용 — 접근 단계 간섭 없음.
    첫 접촉 직후엔 55° 오차가 hold에서 9°로 수렴하므로, G 게이트로 수렴을 더 빠르게 유도.
    [P41 분석] rule-based에서 j6 error=45.33°인데 grip-axis offset=1.03°. FR3가 redundant이라
    joint별 대칭 없이도 TCP geometry 대칭이 가능 → j6만 타겟하는 이 함수는 잘못된 방향.
    P41에서 grip_axis_alignment로 대체됨.
    """
    robot = env.scene[robot_cfg.name]
    if not hasattr(env, "_j6_lr_ids"):
        jnames = robot.data.joint_names
        env._j6_lr_ids = (
            next(i for i, n in enumerate(jnames) if n == "left_fr3_joint6"),
            next(i for i, n in enumerate(jnames) if n == "right_fr3_joint6"),
        )
    l_idx, r_idx = env._j6_lr_ids
    gate = mdp.object_grasped(env, finger1_sensor_cfg, finger2_sensor_cfg, threshold)
    diff = (robot.data.joint_pos[:, l_idx] - robot.data.joint_pos[:, r_idx]).abs()
    return gate * diff


def grip_axis_alignment(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    finger1_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_plate_object_contact"),
    finger2_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_plate_object_contact"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """[P41] 오른-왼 EE 연결벡터와 로봇 +y축의 yaw 오프셋 (rad) 페널티 — G 게이트.

    TCP geometry를 직접 최적화. joint별 대칭 조건 없이 j1~j7 어떤 조합이라도
    grip axis가 로봇 y축과 정렬되면 reward를 받음.

    측정값 (2026-08-12):
      rule-based: grip-axis offset = 1.03°  (j6 err = 45° 에도 불구하고)
      P40 RL:     grip-axis offset = 30.5°  (j6_mirror_loss가 잘못된 joint 조합으로 수렴시킴)

    계산: grip = right_ee_pos - left_ee_pos (world xy),
          ry = robot root +y 방향 (world),
          grip_off = min(|atan2(cross_z, dot_xy)|, π - |atan2(cross_z, dot_xy)|)
          → 0이 이상적, G 게이트에서만 적용.
    """
    robot = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    li, ri = _ee_lr_indices(ee_frame)

    right_pos = ee_frame.data.target_pos_w[:, ri]
    left_pos = ee_frame.data.target_pos_w[:, li]
    grip = right_pos - left_pos  # (N, 3)

    y_ax = torch.zeros(grip.shape[0], 3, device=grip.device)
    y_ax[:, 1] = 1.0
    ry = quat_apply(robot.data.root_quat_w, y_ax)  # robot +y in world

    cross_z = grip[:, 0] * ry[:, 1] - grip[:, 1] * ry[:, 0]
    dot_xy = grip[:, 0] * ry[:, 0] + grip[:, 1] * ry[:, 1]
    raw = torch.atan2(cross_z, dot_xy).abs()
    grip_off = torch.minimum(raw, torch.pi - raw)

    gate = mdp.object_grasped(env, finger1_sensor_cfg, finger2_sensor_cfg, threshold)
    return gate * grip_off


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


def _box_yaw_error(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """박스 x축과 base x축의 수평면 yaw 오차 |각도| (rad). "똑바로 들기(world ori ∥ base)" 측정.

    P31 A(1-cos 별도 페널티 -2~-3)는 무효였음(eval 37°): 진단 결과 grip을 35°로 놓아(placement)
    박스가 따라간 것 -> align(36)이 박스∥판을 강제해 약한 페널티로 못 이김. P32(A')는 이 오차를
    goal_tracking에 곱셈 앵커(_yawtrack_scale ramp) — q'가 additive 정렬을 이긴 것과 동일 논리로
    지배 스케일을 빌린다. yaw는 world/base 기준 측정 필수(교훈 ⑨: box-local은 회전 불변).
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    x_axis = torch.zeros_like(obj.data.root_pos_w)
    x_axis[:, 0] = 1.0
    bx = quat_apply(obj.data.root_quat_w, x_axis)  # 박스 x (world)
    rx = quat_apply(robot.data.root_quat_w, x_axis)  # base x (world)
    return torch.atan2(
        bx[:, 0] * rx[:, 1] - bx[:, 1] * rx[:, 0], bx[:, 0] * rx[:, 0] + bx[:, 1] * rx[:, 1]
    ).abs()


def excessive_squeeze_penalty(
    env,
    force_threshold: float = 60.0,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_plate_object_contact"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_plate_object_contact"),
) -> torch.Tensor:
    """과조임 페널티 (P7): 판별 접촉력이 threshold(N)를 넘는 초과분 합 [N].

    sim에서는 박스가 안 부서져 과조임이 무해하지만 실물에서는 tau_ext 포화/물체 손상으로
    이어짐 — Bi-Touch(arXiv:2307.06423)가 동일 구조에서 실물 "unwanted squeezing"을 겪고
    페널티로 해결한 사례가 직접 근거. 임계값은 "정당하게 필요한 힘"보다 반드시 위여야 함.

    임계값 이력 (P9에서 60 -> 100N):
      P8의 60N은 필요 힘 과소평가였음. 센서가 재는 것은 접촉력의 "합력 norm"이라
      법선(조임)력 + 박스 무게를 지탱하는 수직 마찰력이 합산됨. 최악 DR 케이스
      (5.3kg, 유효 정지마찰 0.5) 기준:
        정적: sqrt(법선 52^2 + 수직마찰 26^2) ~ 58N (60N 임계에 이미 닿음)
        들어올리는 가감속 포함: ~75-87N / 미끄럼 순간(동마찰 유효 ~0.43)은 그 이상
      -> 60N은 정당 파지에 상시 벌금 -> 정책이 벌금 회피용 "모서리 쐐기 파지"를 발명
         (P8: alignment 0.008 붕괴, 오른판이 윗모서리를 누르는 파지, 박스 기울어짐).
      100N = 최악 과도상태(~87N+오차여유)를 무벌 구간에 포함, 습관성 과조임(100N+)만 벌줌.
    """
    forces = []
    for cfg in (left_sensor_cfg, right_sensor_cfg):
        sensor = env.scene.sensors[cfg.name]
        forces.append(torch.norm(sensor.data.force_matrix_w, dim=-1).amax(dim=(-1, -2)))
    f = torch.nan_to_num(torch.stack(forces, dim=-1), nan=0.0, posinf=0.0)
    # NaN 가드 3층 (P11 2차 it2103 크래시): PhysX 충격 접촉력(판-박스)은 한 스텝에 수만 N+가
    # 보고될 수 있음 (침투 해소용 수치 스파이크 - 속도 상한과 무관) -> 무클램프 페널티가
    # reward를 -inf로 오염 -> value loss inf -> 정책 NaN(std<0). 초과분을 200N으로 클램프해
    # 스텝당 페널티를 유한하게 (-0.05 x 400 = 최악 -20/s: 과조임 교정 gradient는 충분, 폭발 차단)
    return (f - force_threshold).clamp(min=0.0, max=200.0).sum(dim=-1)


def object_out_of_bounds(
    env,
    x_range: tuple = (-0.85, 0.85),
    y_range: tuple = (-0.84, 0.84),
    z_range: tuple = (-0.05, 1.66),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """박스가 도달 가능 영역을 벗어나면 종료 (NaN 가드 2층 + 무의미 에피소드 조기 정리).

    경계 근거 (2026-07-29, scripts/tools/compute_dual_fr3_workspace.py 몬테카를로 FK 61k 샘플):
    양팔 pinch 가능한 박스 중심 영역(base frame) x ±0.75 / y ±0.73 / z -0.18~1.16 + 여유 0.1.
    z는 world 기준 (base+0.405): 하한 -0.05 = 바닥 뚫림(물리 붕괴), 상한 1.66 = 사출.
    배경: v3에서 object_dropping을 제거("바닥 task라 낙하 무의미")했는데 "튕겨 사출" 케이스를
    미고려 -> P11 크래시에서 cube의 dropping 종료 같은 오염 차단막이 없었음이 드러남.
    """
    obj = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    out = (
        (pos[:, 0] < x_range[0]) | (pos[:, 0] > x_range[1])
        | (pos[:, 1] < y_range[0]) | (pos[:, 1] > y_range[1])
        | (pos[:, 2] < z_range[0]) | (pos[:, 2] > z_range[1])
    )
    return out


class success_bonus(ManagerTermBase):
    """Success 판정 + 1회성/지속 보너스 (P7, P29) + 선택적 base-yaw 품질 (P38).

    success_step = [박스-goal 거리 < dist_threshold] x [기울기 < tilt_threshold] x [양판 접촉 G]
    이 조건이 hold_steps(기본 25 step = 0.5s) 연속 유지되면 에피소드당 1회 보너스 지급 (latch).
    설계 근거: humanoid box loco-mani(arXiv:2310.03191)의 "목표 위치 + flat orientation" 성공
    정의 + Bi-Touch의 "N-step 유지 후 성공 인정". 기울기 조건이 현 정책의 병목(P6b 강건성
    평가에서 upright hold 0.20)이라 success 보상 자체가 수평 운반을 당기는 두 번째 압력이 됨.
    TB 지표: Episode_Reward/success = (weight x dt x 1회) / 에피소드길이(s) = 6 x 성공률
    (weight 1500, dt 0.02, 5s 기준) -> 값/6 = 성공률.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._count = torch.zeros(env.num_envs, device=env.device)
        self._latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._count[env_ids] = 0.0
        self._latched[env_ids] = False
        return {}

    def __call__(
        self,
        env,
        dist_threshold: float,
        tilt_threshold: float,
        hold_steps: int,
        finger1_sensor_cfg: SceneEntityCfg,
        finger2_sensor_cfg: SceneEntityCfg,
        threshold: float = 1.0,
        command_name: str = "object_pose",
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        sustained: bool = False,
        yaw_alpha: float = 0.0,
        yaw_std: float = 0.35,
    ) -> torch.Tensor:
        # [P38] P37 접촉 대칭이 transient yaw를 31°->24.1°로 줄였지만 grip-axis도 24° off라
        # kinematic placement 잔차가 남음. cause(P37) + outcome(P38)을 결합하되, yaw가 나빠도
        # success의 최소 (1-yaw_alpha)를 보존하는 soft factor로 위치/upright 붕괴를 방지한다.
        robot = env.scene[robot_cfg.name]
        obj = env.scene[object_cfg.name]
        cmd = env.command_manager.get_command(command_name)[:, :3]
        goal_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, cmd)
        near = torch.norm(goal_w - obj.data.root_pos_w, dim=1) < dist_threshold
        upright = box_upright(env, object_cfg) < tilt_threshold
        gate = mdp.object_grasped(env, finger1_sensor_cfg, finger2_sensor_cfg, threshold).bool()
        ok = near & upright & gate
        self._count = torch.where(ok, self._count + 1.0, torch.zeros_like(self._count))
        if sustained:
            # [08041] per-step 지속 (v2): 0.5s 진입(count>=hold_steps) 후 zone 유지하는 "매 스텝" 지급.
            # 1회 latch의 게이밍(끝 0.5s만 정렬해 훔치기, P27/P28) 차단 + termination 없어 "일찍 정렬해
            # 오래 유지"가 최대 이득 = 정렬 유지가 최고 reward로 인센티브 역전. count>=hold_steps는 현재도
            # zone 유지 중임을 함의(zone 벗어나면 count 0 리셋). ⚠ 이 모드에선 Episode_Reward/success가
            # "지속 유지량"이라 기존 "값/6=성공률" 해석 무효 -> 진짜 성공률은 eval 스크립트로.
            reward = (self._count >= hold_steps).float()
            if yaw_alpha > 0.0:
                yaw_scale = getattr(env, "_yawtrack_scale", 0.0)
                yaw_factor = 1.0 - yaw_alpha * yaw_scale * torch.tanh(_box_yaw_error(env) / yaw_std)
                reward = reward * yaw_factor
            return reward
        fire = (self._count >= hold_steps) & ~self._latched
        self._latched |= fire
        return fire.float()


def _lifted_perenv(env, lift_delta: float, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """env별 들기 판정: 박스 중심 z > 안착 중심(반높이_z) + lift_delta. 크기 DR 공평화 (§3-3)."""
    obj = env.scene[object_cfg.name]
    half_z = _box_halves(env, object_cfg)[1]
    return (obj.data.root_pos_w[:, 2] > half_z + lift_delta).float()


def _contact_normal_ratio(
    env,
    finger1_sensor_cfg: SceneEntityCfg,
    finger2_sensor_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """[07302119] 접촉 품질 q = min_i[|F_i·n̂_face|/|F_i|] ∈[0,1]. 면-면 접촉=1, 모서리 걸침=낮음.

    P17 관찰: G(1N 이진)가 너무 얕아 판이 박스 모서리를 1N으로 걸치고도 파지 인정 -> downstream
    (lifting/tracking/success)이 압도해 면-면 정렬을 놓음 (2026-07-30 스샷). q는 접촉힘 벡터가 면
    법선과 나란한 정도 = "얼마나 면으로 눌렀나"의 물리적 척도. 옵션 Y: G는 이진 유지하되 이 q를
    high-value 보상에 연속 곱 -> 비스듬 파지는 보상 깎임(급락 없이 면-면 유도). 발견 차단 회피
    위해 lift_progress엔 미적용(안전밸브). 접촉 없으면 0. |F|=0 나눗셈은 nan_to_num+clamp로 방어.
    잘 안 되면 curriculum으로 G에 하드 q 조건(옵션 X) 추가가 다음 카드.
    """
    obj = env.scene[object_cfg.name]
    y = torch.zeros_like(obj.data.root_pos_w)
    y[:, 1] = 1.0
    n_face = quat_apply(obj.data.root_quat_w, y)  # 박스 +y면 법선 (world, unit)
    qs = []
    for cfg in (finger1_sensor_cfg, finger2_sensor_cfg):
        sensor = env.scene.sensors[cfg.name]
        f = torch.nan_to_num(sensor.data.force_matrix_w.reshape(env.num_envs, -1, 3).sum(dim=1))
        mag = torch.norm(f, dim=-1)
        ratio = (f * n_face).sum(dim=-1).abs() / mag.clamp(min=1e-6)  # |F·n̂|/|F|
        ratio = torch.where(mag > 1e-3, ratio, torch.zeros_like(ratio))  # 접촉 없으면 0
        qs.append(ratio.clamp(0.0, 1.0))
    q = torch.minimum(qs[0], qs[1])
    # [08032/q'] 판 "자세" blend: q는 접촉힘 "방향"만 봐서 판이 기울어 모서리로 닿아도(힘은 면수직)
    # 만점 -> P24 "세트째 기울인 파지" 사각지대. align(판 법선 vs 면 법선, min 양팔)을 곱해
    # "면에 딱 붙은(flush)" 파지만 만점이 되게 한다. 지배 보상(lifting/tracking 36)이 q를 곱하므로
    # flush가 그 스케일을 빌림 (덧셈 reaching은 침식됨이 P20~P24 실증 - 곱셈만 생존).
    # 도입은 soft ramp: q_eff = q·(1-s+s·align), s=env._qprime_scale(성공 배운 뒤 0->1, v2 커리큘럼).
    # s=0이면 q 그대로 -> v1/v0은 이 속성이 없어 완전 무영향 (P12 vanishing 곱셈 리스크도 ramp로 완화).
    # [P30 철회] centrality 곱셈(q''=q·align·centrality)은 박스 로컬 측정이라 world yaw에 불변 ->
    # yaw drift(eval 35°) 못 잡고 goal_tracking까지 깎아 precise-hold 붕괴(0.71->0.31)+커리큘럼 지연.
    # q'로 복귀(P29 상태). yaw는 P31 A(별도 페널티) 무효(grip 35° placement) -> P32 A'로 goal_tracking에 곱셈 앵커.
    s = getattr(env, "_qprime_scale", 0.0)
    if s > 0.0:
        align = _plate_align_min(env, object_cfg)
        q = q * (1.0 - s + s * align)
    return q


def _plate_align_min(env, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """[08032/q'] 판 자세 정렬 min: 양 판 작업면 법선이 박스 면 법선과 반평행한 정도 [0,1].

    reaching_aligned의 align_i와 동일 계산 (판 법선 = TCP frame local z). 곱셈 품질(q')용 헬퍼.
    """
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene["ee_frame"]
    li, ri = _ee_lr_indices(ee_frame)
    y_axis = torch.zeros_like(obj.data.root_pos_w)
    y_axis[:, 1] = 1.0
    y_axis_w = quat_apply(obj.data.root_quat_w, y_axis)
    z_axis = torch.zeros_like(obj.data.root_pos_w)
    z_axis[:, 2] = 1.0
    n_l = quat_apply(ee_frame.data.target_quat_w[:, li], z_axis)
    n_r = quat_apply(ee_frame.data.target_quat_w[:, ri], z_axis)
    a_l = (-(n_l * y_axis_w).sum(dim=-1)).clamp(0.0, 1.0)
    a_r = ((n_r * y_axis_w).sum(dim=-1)).clamp(0.0, 1.0)
    return torch.minimum(a_l, a_r)


def squeezing_quality(
    env,
    finger1_sensor_cfg: SceneEntityCfg,
    finger2_sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """[07302119] 양판 접촉 gate x 접촉 품질 q. 기존 squeezing(접촉 여부 0/1)을 "잘 붙을수록 더"로.

    발견용이라 G(1N)는 이진 유지(발견 안 막음), q를 곱해 면-면 접촉을 애초에 유도.
    """
    gate = mdp.object_grasped(env, finger1_sensor_cfg, finger2_sensor_cfg, threshold)
    q = _contact_normal_ratio(env, finger1_sensor_cfg, finger2_sensor_cfg, object_cfg)
    return gate * q


def object_is_lifted_perenv_and_squeezed(
    env,
    lift_delta: float,
    finger1_sensor_cfg: SceneEntityCfg,
    finger2_sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """env별 높이 gate x 양판 squeeze gate x 접촉 품질 q (07302119: 면-면 파지 유도)."""
    gate = mdp.object_grasped(env, finger1_sensor_cfg, finger2_sensor_cfg, threshold)
    q = _contact_normal_ratio(env, finger1_sensor_cfg, finger2_sensor_cfg, object_cfg)
    return _lifted_perenv(env, lift_delta, object_cfg) * gate * q


def lift_progress(
    env,
    lift_delta: float,
    finger1_sensor_cfg: SceneEntityCfg,
    finger2_sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """발견 사다리 (P14): 잡은 채 들어올린 높이의 연속 진행도 [0,1].

    lifting_object는 "z > 안착 + lift_delta"의 이진 gate라, 잡은 상태에서 그 문턱까지는
    무보상 절벽이었음 -> 정책이 "잡고 안 들기" 국소해에 갇힘 (P5/P13에서 실증, noise_std 폭주).
    이 항은 안착~lift_delta 사이를 선형으로 채워 squeezing -> lifting 사이 계단을 놓는다.
    양판 접촉(G) 게이팅이라 "안 잡고 밀어 올리기"는 무보상.
    문헌 대조: humanoid(2310.03191)는 dense waypoint 안무로 발견을 회피 - 우리는 안무 없이
    상태 기반(높이 진행도)으로 같은 목적 달성 (재시도 능력 보존).
    """
    obj = env.scene[object_cfg.name]
    half_z = _box_halves(env, object_cfg)[1]
    gate = mdp.object_grasped(env, finger1_sensor_cfg, finger2_sensor_cfg, threshold)
    progress = ((obj.data.root_pos_w[:, 2] - half_z) / lift_delta).clamp(0.0, 1.0)
    return gate * progress


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
    """goal 거리 tanh 보상 x 높이 gate x squeeze gate x 접촉 품질 q (07302119: 면-면 파지 유도)."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)
    gate = mdp.object_grasped(env, finger1_sensor_cfg, finger2_sensor_cfg, threshold)
    q = _contact_normal_ratio(env, finger1_sensor_cfg, finger2_sensor_cfg, object_cfg)
    # [P36] yaw_factor는 tracking에서 제거 -> success로 이동. P35 실증: tracking(w16)에 넣으면 지배 보상
    # success(w20 sustained, yaw-blind)가 이겨 무시됨(yaw 30.8°). yaw는 success에 곱셈(success_bonus 참조).
    return _lifted_perenv(env, lift_delta, object_cfg) * gate * q * (1 - torch.tanh(distance / std))


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
    # v0 전용 obs 방어 (P11 크래시 교훈의 예방 이식): NaN 제거 + 500N 클램프 (scale 후 최대 10)
    f = torch.nan_to_num(torch.stack(forces, dim=-1), nan=0.0, posinf=0.0)
    return f.clamp(max=500.0) * scale


# ---------- 커스텀 커맨드: 순수 수직 리프트 (P13) ----------
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand  # noqa: E402


class VerticalLiftPoseCommand(UniformPoseCommand):
    """goal의 x/y를 박스 스폰 위치(base frame)로 고정하고 z만 랜덤 샘플 = 제자리 수직 리프트.

    P13 (2026-07-29): 기존 UniformPoseCommand는 goal x/y를 박스와 독립적으로 샘플해
    "옆으로 옮기기"가 섞여 있었음. 실물 시나리오가 "제자리에서 위로만"이라 goal x/y를
    리셋 시점 박스 위치에 묶어 순수 수직 리프트로 만든다 (z만 pos_z 범위에서 DR).
    리셋 순서상 이 커맨드 resample 시점에 박스는 이미 스폰 위치로 리셋된 상태.
    """

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)  # z / orientation은 그대로 샘플
        obj = self._env.scene["object"]
        obj_pos_b, _ = subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids],
            self.robot.data.root_quat_w[env_ids],
            obj.data.root_pos_w[env_ids],
        )
        self.pose_command_b[env_ids, 0] = obj_pos_b[:, 0]  # goal x = 박스 x
        self.pose_command_b[env_ids, 1] = obj_pos_b[:, 1]  # goal y = 박스 y (z는 super가 샘플)


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
        # P13: 순수 수직 리프트 — goal x/y는 박스 스폰 위치 추종(커맨드가 덮어씀), z만 DR.
        # pos_x/pos_y 범위는 커맨드가 무시하지만 유효값 필요 (박스 스폰 범위로 둠)
        self.commands.object_pose.class_type = VerticalLiftPoseCommand
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
                    # NaN 가드 (P11 it937 크래시 대응): squeeze 폭발로 박스가 초고속 사출되면
                    # object_position obs 오염 -> value loss inf -> 정책 NaN(std<0) 크래시.
                    # 물리적으로 정상 범위(들기 ~1m/s)만 허용하는 상한 - 정상 학습엔 무영향
                    max_linear_velocity=10.0,
                    max_angular_velocity=30.0,
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
        # yaw ±0.1 rad [0804]: "똑바로 들어올리기"(박스 world ori == base frame) 목표 도입.
        #   ±0.3(±17°)은 평판 2개로 잡은 채 능동 재정렬해야 하는 도달 난이도 -> ±0.1로 축소해
        #   목표를 도달 가능하게(안정화기). 실물 배치 yaw robustness는 이후 재확대로 커리큘럼.
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.05, 0.10),
            "y": (-0.10, 0.10),
            "z": (0.0, 0.0),
            "yaw": (-0.1, 0.1),
        }
        # 박스 크기 DR: 30~50cm. env별 크기가 학습 내내 유지 (replicate_physics=False 필수)
        # [08033 버그픽스] mode="usd" -> "prestartup": usd 모드는 아무도 호출 안 해 v0~P24 내내
        # 크기 DR이 무효였음 (8env 동일 크기 실측). dexsuite가 쓰는 prestartup이 정답 (cube 세션 발견).
        self.events.randomize_box_size = EventTerm(
            func=mdp.randomize_rigid_body_scale,
            mode="prestartup",
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
        # [07302019 수정] 접근×정렬 통합: reaching(coarse)+reaching_fine+plate_alignment 3항 ->
        # reaching_aligned 1항 (w6, min_i[(1-tanh(d_i/0.3))·align_i]). P16 스크린샷=판 평행한데 허공에
        # 뜬 채 접근 0 (덧셈이라 정렬만 따먹음) -> 곱셈으로 "가깝고 AND 평행"만 보상. 함수 docstring 참조.
        self.rewards.reaching_object.func = reaching_aligned
        self.rewards.reaching_object.params = {"std": 0.3}
        self.rewards.reaching_object.weight = 6.0
        self.rewards.reaching_fine = None  # [07302019] 제거 (필요시 커리큘럼으로 재도입)
        self.rewards.plate_alignment = None  # [07302019] reaching_aligned로 통합
        # --- [07302019 원복용] 이전 3항 (덧셈 구조, P16까지). 위 통합을 되돌리려면 아래를 살리고 위를 지울 것 ---
        # self.rewards.reaching_object.func = object_dual_ee_face_distance
        # self.rewards.reaching_object.params = {"std": 0.3}
        # self.rewards.reaching_object.weight = 2.0
        # self.rewards.reaching_fine = RewTerm(func=object_dual_ee_face_distance, params={"std": 0.08}, weight=3.0)
        # self.rewards.plate_alignment = RewTerm(func=plate_alignment_min, weight=6.0)
        # 두 판의 이상 쌍 배치 대비 편차 페널티 (마주 보는 꼬집기 강제, v4 -4, P6 yaw-aware, P11 G 게이트)
        self.rewards.plates_misalignment = RewTerm(
            func=plates_pair_misalignment, params=dict(squeeze_gate_params), weight=-4.0
        )
        # 박스 수평 유지 (v6): 기울어진 채 운반 방지. 문헌의 roll/pitch 페널티 상당, yaw 자유
        # P8: 시작 -2 (P6b 값 - 발견 허용 검증됨) -> 커리큘럼으로 lifting>10 후 -3.
        # P7 교훈: -3 상시 + excessive_squeeze 상시는 발견 사다리(tipping/힘찬 조임)에 세금
        # -> "곱게 잡고 안 들기" 국소해 재발 (lifting 0). 페널티 강화는 반드시 성능 기반 커리큘럼으로
        self.rewards.box_upright = RewTerm(func=box_upright, weight=-2.0)
        # [P37] 접촉 타이밍 대칭(yaw origin 차단): 근접 시 |d_left-d_right| 페널티. 한 판 먼저 닿아 박스를
        # 밀어 yaw 생기던 것(play 관찰)을 동시 터치다운으로 방지. 시작 -1e-4 -> lifting 배운 뒤 커리큘럼 -20
        # (교훈 ③: 발견 전엔 방해 최소, 발견 후 접촉 대칭 압력). yaw를 reward로 되돌리는 대신 origin 차단.
        self.rewards.plate_approach_symmetry = RewTerm(func=plates_approach_symmetry, weight=-1e-4)
        # [P32→P37 파킹] yaw 별도 페널티(A)/tracking 곱(A')/success 곱(P36) 모두 접힘. 접촉 대칭으로 대체 시도.
        # 과조임 페널티 (P7 신설, P8 커리큘럼화, P9 임계 60->100N — 근거는 함수 docstring):
        # 시작 -1e-4 (명목값) -> lifting>10 후 -0.05 (20N 초과 시 -1/s)
        self.rewards.excessive_squeeze = RewTerm(
            func=excessive_squeeze_penalty, params={"force_threshold": 100.0}, weight=-1e-4
        )
        # Success 보너스 (P7): goal 10cm + 기울기 15도 + 양판 접촉을 0.5s 유지 -> 1회 지급.
        # weight 1500 = 실효 +30 (RewardManager가 dt 0.02를 곱하므로 1500 x 0.02 = 30)
        self.rewards.success = RewTerm(
            func=success_bonus,
            weight=1500.0,
            params={
                "dist_threshold": 0.10,
                "tilt_threshold": 0.034,  # 1 - cos(15도)
                "hold_steps": 25,  # 0.5s @ 50Hz
                **squeeze_gate_params,
            },
        )
        # 양판 접촉 x 접촉 품질 q ([07302119]: 잘 붙을수록 더. 원복: func=mdp.object_grasped)
        self.rewards.squeezing_object = RewTerm(func=squeezing_quality, params=dict(squeeze_gate_params), weight=2.0)
        # 발견 사다리 (P14): squeezing(잡기) -> lifting(들기) 사이 무보상 절벽을 연속 진행도로 메움.
        # w8 = 안착~lift_delta 구간을 0->8/s로 채우고 lifting(15)이 문턱에서 이어받음.
        # 양의 shaping이라 상시 켜둠(발견을 돕는 항 - P7의 "페널티는 커리큘럼" 규칙과 무관).
        self.rewards.lift_progress = RewTerm(
            func=lift_progress, params={"lift_delta": LIFT_DELTA, **squeeze_gate_params}, weight=8.0
        )
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

        # ---------- termination ----------
        # 낙하 종료(object_dropping)는 바닥 task라 무의미 -> 제거 (v3).
        # 대신 out-of-bounds 종료 (P11 크래시 대응): 도달 불가 영역으로 사출/밀려남 = 즉시 리셋
        # -> 오염 obs가 학습 배치에 유입되는 것을 차단 (cube의 dropping 종료와 같은 역할)
        self.terminations.object_dropping = None
        self.terminations.object_out_of_bounds = DoneTerm(func=object_out_of_bounds)

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
        # P8: P7의 상시 페널티 2종을 성능 기반 커리큘럼으로 이동 (발견 후에만 품질 압력)
        self.curriculum.box_upright = CurrTerm(
            func=modify_reward_weight_on_lifting_learned,
            params={"term_name": "box_upright", "weight": -3.0, "threshold": 10.0},
        )
        # [P37] 접촉 대칭도 lifting 배운 뒤 강화 (-1e-4 -> -20). 발견 전엔 무해, 발견 후 동시 터치다운 압력.
        self.curriculum.plate_approach_symmetry = CurrTerm(
            func=modify_reward_weight_on_lifting_learned,
            params={"term_name": "plate_approach_symmetry", "weight": -20.0, "threshold": 10.0},
        )
        self.curriculum.excessive_squeeze = CurrTerm(
            func=modify_reward_weight_on_lifting_learned,
            params={"term_name": "excessive_squeeze", "weight": -0.05, "threshold": 10.0},
        )
        # [07311] 들기 배운 뒤 "제대로 잡기(reaching 통합항)" 강화: 6 -> 9.
        # P18 관찰: 커리큘럼 발동 후 페널티만 강해지고 reaching은 그대로라, success에 필사적이 되어
        # 파지 정밀도 희생(판 윗가장자리 잡고 들 때 기울어짐). reaching도 같이 올려 "발견 후엔
        # 제대로 잡기"에 집중. positive shaping이라 발견 전엔 6(방해 X), 발견 후 9(정밀 압력).
        # [07311b] 6->12는 과잉 지배(P19: reaching이 goal_tracking 압도, success 0.44->0 붕괴)
        #          -> 12를 9로 완화. 중간값으로 파지 품질↑ 유지하되 goal 옮기기 여지 확보.
        self.curriculum.reaching = CurrTerm(
            func=modify_reward_weight_on_lifting_learned,
            params={"term_name": "reaching_object", "weight": 9.0, "threshold": 10.0},
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


@configclass
class DualFR3PlateBoxLiftEnvCfg_PLAY_NOISY(DualFR3PlateBoxLiftEnvCfg_PLAY):
    """sim2real 강건성 평가용: 학습 분포 그대로 + obs noise 활성 (cube Play-Noisy 패턴 이식).

    obs noise/물성 DR이 학습 cfg에 포함되어 있으므로, 이 cfg는 PLAY가 끈 corruption을
    되켜는 것만 담당한다 (물성/크기 DR은 PLAY에도 상속되어 이미 활성).
    평가 도구: scripts/tools/eval_dual_fr3_plate_robustness.py
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True


# ---------- v1: 실물 1차 검증용 특정 박스 (2026-07-29 합의) ----------
# 사용자 실물 박스: 가로 475 x 세로 555 x 높이 275 mm, 4kg. 잡는 방향(y) = 475mm.
# "안다"의 구현 = obs 추가가 아니라 좁은 DR (단일 종이라 obs 값이 상수 -> 정보량 0,
# humanoid 논문(arXiv:2310.03191)의 "명목값 conditioning + 오차 DR" 구조의 단일 박스 특수화).
# 로드맵: v1 학습 -> P9 zero-shot 대조 -> (이후) v1-B(범위 conditioning)/재질·무게 다양화.
V1_BOX_SIZE = (0.555, 0.475, 0.275)  # (x 전후, y 잡는 방향, z 높이) [m]
V1_BOX_MASS = 4.0  # kg
V1_SIZE_TOL = 0.015  # 축별 측정 오차 ±15mm


def _apply_v1_box(cfg):
    """v0 cfg에 v1 박스 물리(크기/질량/DR)만 덮어쓴다 (obs는 건드리지 않음 - zero-shot cfg 공용)."""
    cfg.scene.object.spawn.size = V1_BOX_SIZE
    cfg.scene.object.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=V1_BOX_MASS)
    # 스폰 z: 최대 반높이(~0.145) + 낙하 여유
    cfg.scene.object.init_state.pos = (0.5, 0.0, 0.16)
    # 크기 DR: 축별 ±15mm (측정 오차)
    cfg.events.randomize_box_size.params["scale_range"] = {
        "x": (1 - V1_SIZE_TOL / V1_BOX_SIZE[0], 1 + V1_SIZE_TOL / V1_BOX_SIZE[0]),
        "y": (1 - V1_SIZE_TOL / V1_BOX_SIZE[1], 1 + V1_SIZE_TOL / V1_BOX_SIZE[1]),
        "z": (1 - V1_SIZE_TOL / V1_BOX_SIZE[2], 1 + V1_SIZE_TOL / V1_BOX_SIZE[2]),
    }
    # 질량 DR: 4.0 ± 0.5 kg 절대값 (v0의 density-scale 방식 대체)
    cfg.events.randomize_box_density.params.update(
        {"mass_distribution_params": (V1_BOX_MASS - 0.5, V1_BOX_MASS + 0.5), "operation": "abs"}
    )
    # 마찰 DR(0.4~1.2)은 넓게 유지 (사용자 결정 - 재질 미확정), yaw/스폰/goal도 v0 그대로


@configclass
class DualFR3PlateBoxLiftEnvCfg_V1(DualFR3PlateBoxLiftEnvCfg):
    """v1: 특정 박스(475x555x275, 4kg) + plate_forces obs 제거 (58 -> 56).

    obs 제거 근거: 질량을 아는 세팅에서는 필요 조임력이 예측 가능 -> excessive_squeeze
    reward로 힘 습관을 sim에서 만들고 실행은 open-loop (humanoid 논문 방식).
    실물에서 tau_ext 파이프라인(payload 등록/노이즈 모델) 의존이 통째로 사라짐.
    tau_ext는 안전 reflex 전용으로 강등. phase indicator는 미도입 (재시도 능력 보존).
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_v1_box(self)
        self.observations.policy.plate_forces = None


@configclass
class DualFR3PlateBoxLiftEnvCfg_V1_PLAY(DualFR3PlateBoxLiftEnvCfg_V1):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_V1_PLAY_NOISY(DualFR3PlateBoxLiftEnvCfg_V1_PLAY):
    """v1 정책 강건성 평가용 (obs noise on)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True


@configclass
class DualFR3PlateBoxLiftEnvCfg_V1_ZEROSHOT(DualFR3PlateBoxLiftEnvCfg_PLAY_NOISY):
    """P9(v0 정책, obs 58)를 v1 박스에서 zero-shot 평가하기 위한 cfg.

    v0의 PLAY_NOISY를 상속해 obs 파이프라인(순서 포함)을 그대로 두고 물리만 v1으로 교체.
    주의: V1 cfg를 상속하면 plate_forces를 다시 붙일 때 obs "순서"가 바뀌어
    P9 정책 입력이 뒤섞이므로 반드시 이 방향(v0 상속 + 물리 오버라이드)으로 구성할 것.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_v1_box(self)


##
# v2: v1(실물 박스) + 외란(disturbance) 주입 = 파지 안정성을 물리로 강제 (2026-08-03)
##
# 철학: "좋은 파지"를 reward로 명세하는 대신, 외란 wrench로 나쁜 파지(모서리/비스듬/비대칭)를
#   물리적으로 실패시킨다 -> success 자체가 안정 파지를 요구 -> 품질↔성공 트레이드오프 소멸.
#   외란은 force closure의 몬테카를로 테스트에 해당. reward/DR 구조는 v1 그대로 (곱셈 q 등 유지).
#   커리큘럼: 들기(lifting EMA>10)를 배운 뒤에만 외란 ON (발견 안전 - P7 교훈의 외란판).
DISTURB_FORCE = 5.0  # N, 각 축 ± ([08031b] P23 10N은 과함(success 0 - 박스가 goal서 계속 흔들림) -> 5N)
DISTURB_TORQUE = 1.0  # Nm, 각 축 ± ([08031b] 2->1)
DISTURB_INTERVAL = (0.4, 0.8)  # s, wrench 재샘플 주기 (지속+방향전환 외란)


def apply_disturbance_wrench(env, env_ids, force: float, torque: float, asset_cfg=SceneEntityCfg("object")):
    """[v2] 박스에 랜덤 외란 wrench 주입. 세기는 env._disturb_scale(커리큘럼)로 0->1 스케일.

    나쁜 파지는 이 외란에 미끄러져/회전해 낙하(=success 실패)하므로, 물리가 좋은 파지를 강제한다.
    scale=0(발견 단계)이면 아무것도 안 함 -> v1과 동일 거동으로 들기 먼저 학습.
    """
    scale = getattr(env, "_disturb_scale", 0.0)
    if scale <= 0.0:
        return
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    size = (len(env_ids), 1, 3)  # object는 단일 body
    forces = torch.empty(size, device=asset.device).uniform_(-force * scale, force * scale)
    torques = torch.empty(size, device=asset.device).uniform_(-torque * scale, torque * scale)
    asset.set_external_force_and_torque(forces, torques, env_ids=env_ids)


def ramp_disturbance_on_success(env, env_ids, success_term: str = "success", lo: float = 1.0, hi: float = 3.0):
    """[v2/A, 08031b] success를 배운 만큼 외란을 점진적으로 켠다 (env._disturb_scale 0->1, 비감소).

    P23 교훈: lifting 기준 + 즉시 full latch로 외란이 it444(success 나오기도 전)에 최대로 켜져,
    정책이 '쐐기로 버티는' 안정 국소해에 갇히고 success가 끝까지 0이었음. -> success EMA 기준으로
    lo(성공 시작)~hi(견고한 성공)를 지나며 scale 0->1 점진 상승. 비감소 latch(외란으로 success가
    잠시 떨어져도 완화 안 함)라 '깨끗한 성공 학습 -> 점진 단련 -> 재안정' 커리큘럼이 됨.
    success가 lo에 못 미치면 외란 OFF -> v1과 동일 (안전한 저하). 반환값은 Curriculum/disturbance 기록.
    """
    prev = getattr(env, "_disturb_scale", 0.0)
    if prev >= 1.0:
        return 1.0
    if not hasattr(env, "_lift_curriculum_ema"):
        env._lift_curriculum_ema = {}
    episode_sums = env.reward_manager._episode_sums[success_term][env_ids]
    success_level = episode_sums.mean().item()
    alpha = min(0.5, 3.0 * len(env_ids) / env.num_envs)
    ema = env._lift_curriculum_ema.get("_disturb_succ", 0.0)
    ema = ema + alpha * (success_level - ema)
    env._lift_curriculum_ema["_disturb_succ"] = ema
    target = max(0.0, min(1.0, (ema - lo) / (hi - lo)))
    env._disturb_scale = max(prev, target)  # 비감소 (한 번 오른 외란은 안 내림)
    return env._disturb_scale


def ramp_qprime_on_success(env, env_ids, success_term: str = "success", lo: float = 1.0, hi: float = 3.0):
    """[08032/q'] success를 배운 만큼 q에 판 자세(align)를 점진 곱한다 (env._qprime_scale 0->1, 비감소).

    ramp_disturbance_on_success와 동일 패턴 (성공 먼저 -> 점진 강화). s=1이면 q_eff = q·align:
    지배 보상(lifting/tracking)이 "면에 딱 붙은 파지"만 만점 지급. 반환값 Curriculum/qprime 기록.
    """
    prev = getattr(env, "_qprime_scale", 0.0)
    if prev >= 1.0:
        return 1.0
    if not hasattr(env, "_lift_curriculum_ema"):
        env._lift_curriculum_ema = {}
    episode_sums = env.reward_manager._episode_sums[success_term][env_ids]
    success_level = episode_sums.mean().item()
    alpha = min(0.5, 3.0 * len(env_ids) / env.num_envs)
    ema = env._lift_curriculum_ema.get("_qprime_succ", 0.0)
    ema = ema + alpha * (success_level - ema)
    env._lift_curriculum_ema["_qprime_succ"] = ema
    target = max(0.0, min(1.0, (ema - lo) / (hi - lo)))
    env._qprime_scale = max(prev, target)
    return env._qprime_scale


def ramp_yawtrack_on_success(
    env, env_ids, success_term: str = "success", trigger: float = 3.0, ramp_steps: int = 72000
):
    """[P34/P38] iteration 기반 완만 ramp: success EMA가 처음 trigger 도달한 뒤, 그 시점부터 ramp_steps
    env-step에 걸쳐 _yawtrack_scale 0->1 선형.

    P33 문제: success EMA가 급등(it9490에 3->6을 13iter만에)해 EMA기반 ramp(lo/hi)가 사실상 step이 됨
    -> yaw가 순간 full로 걸려 position 붕괴(track 10.9->3.85, precise 0.004). 그래도 yaw는 3.1°까지 갔음
    (H1 확증: 도달 가능). 해법: 발동 "조건"은 success로 유지(발견 안전) but ramp "속도"는 iteration으로
    고정 -> EMA 급등과 무관하게 완만 + position 통합 시간 보장. 늦게 발동해도 급발진 없이 부분 적용(graceful).
    ramp_steps=72000 = num_steps_per_env(24) x 3000 iter. 반환값 Curriculum/yawtrack 기록.
    """
    scale = getattr(env, "_yawtrack_scale", 0.0)
    if scale >= 1.0:
        return 1.0
    cur = env.common_step_counter
    start = getattr(env, "_yawtrack_start", None)
    if start is None:
        # 최초 발동 판정: success EMA >= trigger (disturbance full 지점 = 성공 단단해짐). 발동 전엔 s=0.
        if not hasattr(env, "_lift_curriculum_ema"):
            env._lift_curriculum_ema = {}
        episode_sums = env.reward_manager._episode_sums[success_term][env_ids]
        success_level = episode_sums.mean().item()
        alpha = min(0.5, 3.0 * len(env_ids) / env.num_envs)
        ema = env._lift_curriculum_ema.get("_yawtrack_succ", 0.0)
        ema = ema + alpha * (success_level - ema)
        env._lift_curriculum_ema["_yawtrack_succ"] = ema
        if ema >= trigger:
            env._yawtrack_start = cur
        return 0.0
    env._yawtrack_scale = max(scale, min(1.0, (cur - start) / ramp_steps))
    return env._yawtrack_scale


def plate_height_symmetry(
    env,
    finger1_sensor_cfg: SceneEntityCfg,
    finger2_sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """[v2/B, 08031b] 양 판 접촉 높이차 |z_L - z_R| (G 게이트). 음수 weight로 페널티.

    P23 play: 한 판이 아래서 받치고 한 판이 위에서 눌러 박스가 기우는 '쐐기 브레이싱' 관찰.
    외란(안정)만으론 이 비대칭 안정해를 못 막음 -> 높이 대칭을 구조로 강제. 양 판이 같은 높이로
    맞물려야 모멘트가 사라져 박스가 수평 유지. G(양판 접촉) 중에만 적용(발견 방해 X).
    """
    gate = mdp.object_grasped(env, finger1_sensor_cfg, finger2_sensor_cfg, threshold)
    ee_frame = env.scene[ee_frame_cfg.name]
    li, ri = _ee_lr_indices(ee_frame)
    dz = (ee_frame.data.target_pos_w[:, li, 2] - ee_frame.data.target_pos_w[:, ri, 2]).abs()
    return gate * dz


@configclass
class DualFR3PlateBoxLiftEnvCfg_V2(DualFR3PlateBoxLiftEnvCfg_V1):
    """Current P38/V2 baseline: fixed real box + force-free actor + robust grasp/yaw curricula.

    V1에서 시작했지만 현재는 disturbance, plate-height symmetry, qprime, sustained success와
    P38 soft base-yaw factor까지 포함한다. P39는 이 class를 동결 기준점으로 상속한다.
    """

    def __post_init__(self):
        super().__post_init__()
        # [08033b] q' 격리: 크기 DR OFF (scale_range=(1,1)). P25에서 q'+크기DR 동시 변경으로 confound
        # (크기 DR이 궤적을 나쁜 basin으로 밀어 success 0 -> q' 미작동). ±15mm는 품질에 무의미(크기-불변
        # 기울임)하므로 끄는 비용 0. 크기 DR(prestartup 픽스 자체는 유효)은 q' 검증 후 별도 run으로.
        self.events.randomize_box_size.params["scale_range"] = {"x": (1.0, 1.0), "y": (1.0, 1.0), "z": (1.0, 1.0)}
        # 외란 이벤트: interval마다 박스에 랜덤 wrench (env._disturb_scale로 커리큘럼 스케일)
        self.events.disturb_box = EventTerm(
            func=apply_disturbance_wrench,
            mode="interval",
            interval_range_s=DISTURB_INTERVAL,
            params={"force": DISTURB_FORCE, "torque": DISTURB_TORQUE, "asset_cfg": SceneEntityCfg("object")},
        )
        # [08031b/A] success를 배운 만큼 외란 점진 ON (P23: lifting 기준 즉시 full은 너무 이르고 셈)
        self.curriculum.disturbance = CurrTerm(
            func=ramp_disturbance_on_success,
            params={"success_term": "success", "lo": 1.0, "hi": 3.0},
        )
        # [08031b/B] 판 높이차 대칭 페널티: 쐐기 브레이싱(비대칭) 차단 -> 박스 수평 유지.
        # 외란(안정)이 못 막는 비대칭 안정해를 구조로 차단. P7 규칙: -1e-4 시작 + 들기 후 커리큘럼 강화.
        self.rewards.plate_height_symmetry = RewTerm(
            func=plate_height_symmetry,
            params={
                "finger1_sensor_cfg": SceneEntityCfg("left_plate_object_contact"),
                "finger2_sensor_cfg": SceneEntityCfg("right_plate_object_contact"),
                "threshold": 1.0,
            },
            weight=-1e-4,
        )
        self.curriculum.plate_height_symmetry = CurrTerm(
            func=modify_reward_weight_on_lifting_learned,
            params={"term_name": "plate_height_symmetry", "weight": -6.0, "threshold": 10.0},
        )
        # [08032/q'] P24 잔존 문제("세트째 기울인 파지" - 판끼리 평행하나 박스 면과 안 평행) 대응:
        # q(힘 방향)에 판 자세(align)를 success 배운 뒤 점진 곱셈 -> flush 신호가 지배 보상(36)의
        # 스케일을 빌림. weight 재조정 축은 P15-v2(2 무시)/P20(9 부분)/P19(12 붕괴)로 기각 완료.
        self.curriculum.qprime = CurrTerm(
            func=ramp_qprime_on_success,
            params={"success_term": "success", "lo": 1.0, "hi": 3.0},
        )
        # [P38] P37 접촉 대칭(cause)은 유지하고 base-yaw outcome을 success에 soft 앵커.
        # P36은 discovery가 it9490으로 늦어 yaw scale 0.06에서 끝나 미검증이었지만, P37은 success를
        # 일찍 배워 3000 iter ramp를 완주할 조건이 생김. yaw_alpha=0.5라 full ramp에서도 success
        # 보상의 50% floor를 보존해 P33식 position/upright 붕괴를 방지한다.
        self.curriculum.yawtrack = CurrTerm(
            func=ramp_yawtrack_on_success,
            params={"success_term": "success", "trigger": 3.0, "ramp_steps": 72000},
        )
        # [08041] success 1회 latch -> per-step 지속 (v2 전용). P27/P28 게이밍 대응(끝 0.5s만 정렬).
        # weight 1500(1회 delivered 30) -> 20 (지속 중 delivered ~20/s, dense lifting 15/s와 경쟁/우위 ->
        # success-zone 유지가 최대 이득). 0.5s 진입 조건(hold_steps)은 유지(순간 스침 방지). v0/v1은 sustained
        # 미설정=False로 기존 1회성 그대로 (동시 실행 불변).
        self.rewards.success.weight = 20.0
        self.rewards.success.params["sustained"] = True
        self.rewards.success.params.update({"yaw_alpha": 0.5, "yaw_std": 0.35})


@configclass
class DualFR3PlateBoxLiftEnvCfg_V2_PLAY(DualFR3PlateBoxLiftEnvCfg_V2):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_V2_PLAY_NOISY(DualFR3PlateBoxLiftEnvCfg_V2_PLAY):
    """v2 정책 강건성 평가용 (obs noise on)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True


@configclass
class DualFR3PlateBoxLiftEnvCfg_P39(DualFR3PlateBoxLiftEnvCfg_V2):
    """P39 = frozen P38/V2 baseline + actual one-sided-contact XOR duration penalty only.

    The term starts at exactly zero, so reward is identical to P38 until the shared lifting EMA exceeds 10.
    It then changes once to -5.0. Existing geometric ``plate_approach_symmetry`` remains unchanged so the
    P38->P39 comparison has one new reward signal rather than a replacement.
    """

    def __post_init__(self):
        super().__post_init__()
        contact_params = {
            "left_sensor_cfg": SceneEntityCfg("left_plate_object_contact"),
            "right_sensor_cfg": SceneEntityCfg("right_plate_object_contact"),
            "threshold": 1.0,
            "filter_idx": 0,
        }
        self.rewards.one_sided_contact = RewTerm(
            func=one_sided_plate_contact,
            params=contact_params,
            weight=0.0,
        )
        self.curriculum.one_sided_contact = CurrTerm(
            func=modify_reward_weight_on_lifting_learned,
            params={"term_name": "one_sided_contact", "weight": -5.0, "threshold": 10.0},
        )


@configclass
class DualFR3PlateBoxLiftEnvCfg_P39_PLAY(DualFR3PlateBoxLiftEnvCfg_P39):
    """P39 visualization/evaluation cfg with observation corruption disabled."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_P39_PLAY_NOISY(DualFR3PlateBoxLiftEnvCfg_P39_PLAY):
    """P39 robustness-evaluation cfg with training observation corruption enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True


@configclass
class DualFR3PlateBoxLiftEnvCfg_P40(DualFR3PlateBoxLiftEnvCfg_P39):
    """P40 = frozen P39 baseline + j6 wrist-pitch mirror loss.

    단일 변경: 양판 접촉(G 게이트) 중 |q_L_j6 - q_R_j6|(rad) 페널티.
    진단 근거: hold 중 j6 L=127°/R=118° 잔존 비대칭(중간값 13.87°, p95 31.21°)이
    plate face 기울기를 만들어 부분 면접촉(~2/3)을 유발함. sign=+1 대칭 조건.
    weight 0.0 시작 → lifting EMA>10 후 -0.1 latch.
    초기 -0.5 run(2026-08-10) dense gradient → entropy 붕괴 확인 → -0.1로 완화(2026-08-11).
    P39 전 항목(one_sided_contact, plate_approach_symmetry, yawtrack, qprime 등) 불변.
    """

    def __post_init__(self):
        super().__post_init__()
        grasp_params = {
            "finger1_sensor_cfg": SceneEntityCfg("left_plate_object_contact"),
            "finger2_sensor_cfg": SceneEntityCfg("right_plate_object_contact"),
            "threshold": 1.0,
        }
        self.rewards.j6_mirror_loss = RewTerm(
            func=j6_mirror_loss,
            params=grasp_params,
            weight=0.0,
        )
        self.curriculum.j6_mirror_loss = CurrTerm(
            func=modify_reward_weight_on_lifting_learned,
            params={"term_name": "j6_mirror_loss", "weight": -0.1, "threshold": 10.0},
        )


@configclass
class DualFR3PlateBoxLiftEnvCfg_P40_PLAY(DualFR3PlateBoxLiftEnvCfg_P40):
    """P40 visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_P40_PLAY_NOISY(DualFR3PlateBoxLiftEnvCfg_P40_PLAY):
    """P40 robustness-evaluation cfg with training observation corruption enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True


@configclass
class DualFR3PlateBoxLiftEnvCfg_P41(DualFR3PlateBoxLiftEnvCfg_P39):
    """P41 = frozen P39 baseline + grip_axis_alignment (j6_mirror_loss 대체).

    변경 사항 (2026-08-12):
      P40의 j6_mirror_loss는 rule-based 분석 결과 잘못된 방향임을 확인.
      rule-based에서 j6 error=45.33° 임에도 grip-axis offset=1.03° — FR3가 redundant이라
      joint 개별 대칭 없이도 TCP geometry 대칭 달성 가능. P40 RL은 j6만 타겟해 이상한
      joint 조합으로 수렴(grip-axis 30.5°).
      → joint 개별 대칭 대신 TCP geometry(오른-왼 연결벡터 vs 로봇 +y축 yaw 오프셋)를
        직접 최적화. j1~j7 어떤 조합이라도 grip axis만 맞추면 reward 지급.
    weight 0.0 시작 → lifting EMA>10 후 -0.1 latch (P40과 동일 schedule).
    P39 전 항목 불변.
    """

    def __post_init__(self):
        super().__post_init__()
        grasp_params = {
            "finger1_sensor_cfg": SceneEntityCfg("left_plate_object_contact"),
            "finger2_sensor_cfg": SceneEntityCfg("right_plate_object_contact"),
            "threshold": 1.0,
        }
        self.rewards.grip_axis_alignment = RewTerm(
            func=grip_axis_alignment,
            params=grasp_params,
            weight=0.0,
        )
        self.curriculum.grip_axis_alignment = CurrTerm(
            func=modify_reward_weight_on_lifting_learned,
            params={"term_name": "grip_axis_alignment", "weight": -0.1, "threshold": 10.0},
        )


@configclass
class DualFR3PlateBoxLiftEnvCfg_P41_PLAY(DualFR3PlateBoxLiftEnvCfg_P41):
    """P41 visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_P41_PLAY_NOISY(DualFR3PlateBoxLiftEnvCfg_P41_PLAY):
    """P41 robustness-evaluation cfg with training observation corruption enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True


@configclass
class DualFR3PlateBoxLiftEnvCfg_P42(DualFR3PlateBoxLiftEnvCfg_P39):
    """P42 = frozen P39 baseline + plates_pair_misalignment weight -4.0 → -6.0.

    변경 사항 (2026-08-13):
      P41 eval에서 box y축 쏠림(p95 +0.194m) + P39 play에서 2/3 면접촉 관찰.
      plates_pair_misalignment(파지 중 위치 편차 페널티)를 강화해 파지 후 위치 비대칭 압력 증가.
      -4→-6 (중간 단계). 역사: -4 미게이팅(P3)은 정지 국소해 유발했으나 현재는 G-게이트라 안전.
      -6→-8은 이력 없어 jump 크므로 우선 -6에서 검증.
    P39 전 항목 불변.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.plates_misalignment.weight = -6.0


@configclass
class DualFR3PlateBoxLiftEnvCfg_P42_PLAY(DualFR3PlateBoxLiftEnvCfg_P42):
    """P42 visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_P42_PLAY_NOISY(DualFR3PlateBoxLiftEnvCfg_P42_PLAY):
    """P42 robustness-evaluation cfg with training observation corruption enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True


@configclass
class DualFR3PlateBoxLiftEnvCfg_P43(DualFR3PlateBoxLiftEnvCfg_P39):
    """P43 = frozen P39 baseline + plates_approach_symmetry proximity 0.05 → 0.08.

    변경 사항 (2026-08-18):
      P42 eval에서 first-contact tie 20.3% (P39: 47.7%)로 감소 → yaw 4.5° 악화 원인.
      plates_approach_symmetry의 proximity를 0.05→0.08로 확대해 더 일찍(8cm부터) 좌우
      대칭 접근 압력을 줌으로써 tie 회복을 시도.
      proximity는 P37 신설 이후 한 번도 건드리지 않은 노브.
      P39 전 항목 불변 (plates_misalignment -4.0, P42 변경 미포함).
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.plate_approach_symmetry = RewTerm(
            func=plates_approach_symmetry,
            params={"proximity": 0.08},
            weight=-1e-4,
        )
        self.curriculum.plate_approach_symmetry = CurrTerm(
            func=modify_reward_weight_on_lifting_learned,
            params={"term_name": "plate_approach_symmetry", "weight": -20.0, "threshold": 10.0},
        )


@configclass
class DualFR3PlateBoxLiftEnvCfg_P43_PLAY(DualFR3PlateBoxLiftEnvCfg_P43):
    """P43 visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_P43_PLAY_NOISY(DualFR3PlateBoxLiftEnvCfg_P43_PLAY):
    """P43 robustness-evaluation cfg with training observation corruption enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True


@configclass
class DualFR3PlateBoxLiftEnvCfg_P44(DualFR3PlateBoxLiftEnvCfg_P39):
    """P44 = frozen P39 baseline + yawtrack trigger 3.0 → 1.5 (조기 발동).

    변경 사항 (2026-08-18):
      P39/P42/P43 모두 play에서 초기 접촉 시 yaw 발생 후 나중에 정렬하는 패턴 잔존.
      proximity 확대(P43)로 tie 미회복 → yaw origin 차단이 아닌 outcome 조기 강화 시도.
      yawtrack trigger를 3.0→1.5로 낮춰 success를 더 일찍 배운 시점부터 yaw 압력 시작.
      ramp_steps=72000(3000iter) 유지 — P33 전례(급발진 collapse) 고려, 속도는 불변.
    P39 전 항목 불변.
    """

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.yawtrack = CurrTerm(
            func=ramp_yawtrack_on_success,
            params={"success_term": "success", "trigger": 1.5, "ramp_steps": 72000},
        )


@configclass
class DualFR3PlateBoxLiftEnvCfg_P44_PLAY(DualFR3PlateBoxLiftEnvCfg_P44):
    """P44 visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_P44_PLAY_NOISY(DualFR3PlateBoxLiftEnvCfg_P44_PLAY):
    """P44 robustness-evaluation cfg with training observation corruption enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True


@configclass
class DualFR3PlateBoxLiftEnvCfg_P45(DualFR3PlateBoxLiftEnvCfg_P44):
    """P45 = P44 + one_sided_contact curriculum weight -5.0 → -7.0.

    변경 사항 (2026-08-21):
      P44 eval: one-sided duration 1.31 steps (P39: 0.67), upright 0.752 (P39: 0.777).
      hold 중 한쪽 판 떨어지는 패턴이 잔존 → one_sided_contact 페널티 강화로 직접 억제.
      -5.0 → -7.0 (40% 강화). 이력: -5.0은 P39에서 검증됨, -7.0은 첫 시도.
    P44 전 항목 불변 (yawtrack trigger=1.5 포함).
    """

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.one_sided_contact = CurrTerm(
            func=modify_reward_weight_on_lifting_learned,
            params={"term_name": "one_sided_contact", "weight": -7.0, "threshold": 10.0},
        )


@configclass
class DualFR3PlateBoxLiftEnvCfg_P45_PLAY(DualFR3PlateBoxLiftEnvCfg_P45):
    """P45 visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_P45_PLAY_NOISY(DualFR3PlateBoxLiftEnvCfg_P45_PLAY):
    """P45 robustness-evaluation cfg with training observation corruption enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True


@configclass
class DualFR3PlateBoxLiftEnvCfg_P46(DualFR3PlateBoxLiftEnvCfg_P44):
    """P46 = P44 + box_upright curriculum weight -3.0 → -4.0.

    변경 사항 (2026-08-21):
      P44 eval: upright hold 0.752 (P39: 0.777). hold 중 박스 기울어짐 잔존.
      box_upright 페널티 상한을 강화해 수평 자세 유지 압력 증가.
      -3.0 → -4.0 (33% 강화). one_sided는 P44 그대로 -5.0.
    P44 전 항목 불변 (yawtrack trigger=1.5 포함).
    """

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.box_upright = CurrTerm(
            func=modify_reward_weight_on_lifting_learned,
            params={"term_name": "box_upright", "weight": -4.0, "threshold": 10.0},
        )


@configclass
class DualFR3PlateBoxLiftEnvCfg_P46_PLAY(DualFR3PlateBoxLiftEnvCfg_P46):
    """P46 visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_P46_PLAY_NOISY(DualFR3PlateBoxLiftEnvCfg_P46_PLAY):
    """P46 robustness-evaluation cfg with training observation corruption enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True


@configclass
class DualFR3PlateBoxLiftEnvCfg_P47(DualFR3PlateBoxLiftEnvCfg_P44):
    """P47 = P44 + plate_height_symmetry curriculum weight -6.0 → -8.0.

    변경 사항 (2026-08-21):
      P44 eval: plate height diff p95 7.6cm. hold 중 좌우 판 높이 비대칭 잔존.
      plate_height_symmetry 페널티 강화로 양판 높이 균등 압력 증가.
      -6.0 → -8.0 (33% 강화). one_sided는 P44 그대로 -5.0.
    P44 전 항목 불변 (yawtrack trigger=1.5 포함).
    """

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.plate_height_symmetry = CurrTerm(
            func=modify_reward_weight_on_lifting_learned,
            params={"term_name": "plate_height_symmetry", "weight": -8.0, "threshold": 10.0},
        )


@configclass
class DualFR3PlateBoxLiftEnvCfg_P47_PLAY(DualFR3PlateBoxLiftEnvCfg_P47):
    """P47 visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_P47_PLAY_NOISY(DualFR3PlateBoxLiftEnvCfg_P47_PLAY):
    """P47 robustness-evaluation cfg with training observation corruption enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True


@configclass
class DualFR3PlateBoxLiftEnvCfg_P48(DualFR3PlateBoxLiftEnvCfg_P44):
    """P48 = P44 + excessive_squeeze force_threshold 100N → 120N.

    변경 사항 (2026-08-21):
      P44 eval: hold 중 가끔 squeeze 과다 패널티로 grip 느슨해지는 패턴 의심.
      force_threshold를 높여 더 강한 조임을 허용 → 안정적 hold 유도.
      100N → 120N. one_sided는 P44 그대로 -5.0.
    P44 전 항목 불변 (yawtrack trigger=1.5 포함).
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.excessive_squeeze = RewTerm(
            func=excessive_squeeze_penalty, params={"force_threshold": 120.0}, weight=-1e-4
        )


@configclass
class DualFR3PlateBoxLiftEnvCfg_P48_PLAY(DualFR3PlateBoxLiftEnvCfg_P48):
    """P48 visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_P48_PLAY_NOISY(DualFR3PlateBoxLiftEnvCfg_P48_PLAY):
    """P48 robustness-evaluation cfg with training observation corruption enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True


@configclass
class DualFR3PlateBoxLiftEnvCfg_P49(DualFR3PlateBoxLiftEnvCfg_P47):
    """P49 = P47 + plate_face_alignment reward 활성화 (curriculum).

    변경 사항 (2026-08-24):
      play 관찰: P44는 오른팔 평행·왼팔 모서리, P47은 왼팔 평행·오른팔 모서리.
      '한쪽만 좋으면 되는 local optimum' 원인:
        plates_misalignment(판↔판 상대 위치)와 plate_approach_symmetry(접근 타이밍)는
        각 판과 box face 사이의 절대 각도를 직접 보지 않음 → 한 팔 완벽+한 팔 모서리
        = reward 거의 동일한 해가 존재.

      `plate_face_alignment` 함수는 P12부터 정의되어 있으나 P39 이후 reward에 등록되지 않음.
      구조:
        ① per-arm box face normal 기준 각도: n_L·(-y_box), n_R·(+y_box) 각각 측정
        ② min(a_L, a_R): 양팔 모두 좋아야 reward 오름 (한쪽만으론 절반도 못 받음)
        ③ G-gate (양판 동시 접촉 threshold=1N): 접근 phase에서 P3 회피
        ④ upright_factor: 박스 기울여 판 붙이는 exploit 차단 (P11 역사 교훈)

      P41 yaw 붕괴와의 차이:
        P41은 squeeze axis(판↔판 축) 기준 → yawtrack과 충돌.
        P49는 box face normal(y_box) 기준 → yawtrack과 방향 일치
        (box yaw 정렬될수록 plate_face_alignment와 yawtrack 둘 다 개선됨).
      P45 P3 패턴과의 차이:
        G-gate로 접근 phase 제외 → 접촉 회피 유인 없음.
        positive reward(0~1)라 페널티 과다 붕괴 위험 낮음.

      curriculum: lifting EMA > 10 이후 1e-4 → 2.0 활성
        (다른 curriculum 항목과 동일 threshold, 발견 전엔 무해)
    P47 전 항목 불변 (plate_height_symmetry -8.0, yawtrack trigger=1.5 포함).
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.plate_face_alignment = RewTerm(
            func=plate_face_alignment, weight=1e-4
        )
        self.curriculum.plate_face_alignment = CurrTerm(
            func=modify_reward_weight_on_lifting_learned,
            params={"term_name": "plate_face_alignment", "weight": 2.0, "threshold": 10.0},
        )


@configclass
class DualFR3PlateBoxLiftEnvCfg_P49_PLAY(DualFR3PlateBoxLiftEnvCfg_P49):
    """P49 visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_P49_PLAY_NOISY(DualFR3PlateBoxLiftEnvCfg_P49_PLAY):
    """P49 robustness-evaluation cfg with training observation corruption enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True


@configclass
class DualFR3PlateBoxLiftEnvCfg_P50(DualFR3PlateBoxLiftEnvCfg_P49):
    """P50 = P49 + plate_face_alignment curriculum weight 2.0 → 1.0.

    변경 사항 (2026-08-24):
      P49 eval: grip-axis median 17.3° (P47: 3.3°). weight=2.0이 다른 reward와 균형을
      깨뜨려 판이 y_box를 과도하게 추적, 세계좌표 기준 grip axis에서 이탈 가능성.
      weight 절반으로 완화해 plate_face_alignment와 기존 항목 간 균형 조정.
    P49 전 항목 불변.
    """

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.plate_face_alignment = CurrTerm(
            func=modify_reward_weight_on_lifting_learned,
            params={"term_name": "plate_face_alignment", "weight": 1.0, "threshold": 10.0},
        )


@configclass
class DualFR3PlateBoxLiftEnvCfg_P50_PLAY(DualFR3PlateBoxLiftEnvCfg_P50):
    """P50 visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_P50_PLAY_NOISY(DualFR3PlateBoxLiftEnvCfg_P50_PLAY):
    """P50 robustness-evaluation cfg with training observation corruption enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True


@configclass
class DualFR3PlateBoxLiftEnvCfg_P51(DualFR3PlateBoxLiftEnvCfg_P49):
    """P51 = P49 + plate_face_z_offset 페널티 추가 (curriculum).

    변경 사항 (2026-08-24):
      P49 play: 오른팔 plate가 box 위쪽 부분에만 접촉.
      plates_pair_misalignment(z항)는 (L_z-R_z) 상대 높이 차만 → 양팔이 같은 높이로
      높아지면 포착 불가. 새 함수 plate_face_z_offset으로 각 판의 담당 face 중심 대비
      절대 z-오프셋을 per-arm 합산 → 양팔 동시 높이 오류까지 커버.
      G-gate 조건: 파지 후에만 활성 (P3 방지). curriculum: 1e-4 → -2.0 after lifting EMA>10.
    P49 전 항목 불변 (plate_face_alignment weight=2.0 포함).
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.plate_face_z_offset = RewTerm(
            func=plate_face_z_offset, weight=-1e-4
        )
        self.curriculum.plate_face_z_offset = CurrTerm(
            func=modify_reward_weight_on_lifting_learned,
            params={"term_name": "plate_face_z_offset", "weight": -2.0, "threshold": 10.0},
        )


@configclass
class DualFR3PlateBoxLiftEnvCfg_P51_PLAY(DualFR3PlateBoxLiftEnvCfg_P51):
    """P51 visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_P51_PLAY_NOISY(DualFR3PlateBoxLiftEnvCfg_P51_PLAY):
    """P51 robustness-evaluation cfg with training observation corruption enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True
