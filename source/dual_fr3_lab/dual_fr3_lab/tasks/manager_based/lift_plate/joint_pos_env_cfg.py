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

import copy
import os
import numpy as np
import torch
import warp as wp

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
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_apply_inverse,
    quat_from_angle_axis,
    quat_mul,
    subtract_frame_transforms,
)
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise

from dual_fr3_lab.tasks.manager_based.lift_cube import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from dual_fr3_lab.robots.dual_fr3_plate_ee import DUAL_FR3_PLATE_EE_CFG, PLATE_TCP_OFFSET_POS  # isort: skip
from dual_fr3_lab.robots.dual_fr3 import with_arm_passive_dynamics  # isort: skip

# 성능 기반 커리큘럼 (스텝 고정 커리큘럼은 들기 발견 전 발동 시 정책 붕괴 -> dual_fr3에서 검증됨)
from dual_fr3_lab.tasks.manager_based.lift_cube.joint_pos_env_cfg import (  # isort: skip
    apply_sim2real_control,
    modify_reward_weight_on_lifting_learned,
)
# Reach Sim2Real 관절 순서 계약 (left joint1~7, right joint1~7)
from dual_fr3_lab.tasks.manager_based.reach.joint_pos_env_cfg import DUAL_ARM_JOINT_NAMES  # isort: skip
from dual_fr3_lab.tasks.manager_based.reach.mdp.actions import DelayedReachRelativeJointPositionActionCfg  # isort: skip


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
    y_axis = torch.zeros_like(obj.data.root_pos_w.torch)
    y_axis[:, 1] = 1.0
    y_axis_w = quat_apply(obj.data.root_quat_w.torch, y_axis)  # 박스 +y면 법선 (world)
    face_l = obj.data.root_pos_w.torch + y_axis_w * half
    face_r = obj.data.root_pos_w.torch - y_axis_w * half
    d_l = torch.norm(ee_frame.data.target_pos_w.torch[:, li] - face_l, dim=-1)
    d_r = torch.norm(ee_frame.data.target_pos_w.torch[:, ri] - face_r, dim=-1)
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
    z_axis = torch.zeros_like(obj.data.root_pos_w.torch)
    z_axis[:, 2] = 1.0
    n_l = quat_apply(ee_frame.data.target_quat_w.torch[:, li], z_axis)  # 왼판 작업면 법선
    n_r = quat_apply(ee_frame.data.target_quat_w.torch[:, ri], z_axis)
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
    z_axis = torch.zeros_like(obj.data.root_pos_w.torch)
    z_axis[:, 2] = 1.0
    n_l = quat_apply(ee_frame.data.target_quat_w.torch[:, li], z_axis)  # 왼판 작업면 법선
    n_r = quat_apply(ee_frame.data.target_quat_w.torch[:, ri], z_axis)
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
    z_axis = torch.zeros_like(obj.data.root_pos_w.torch)
    z_axis[:, 2] = 1.0
    n_l = quat_apply(ee_frame.data.target_quat_w.torch[:, li], z_axis)  # 왼판 작업면 법선
    n_r = quat_apply(ee_frame.data.target_quat_w.torch[:, ri], z_axis)
    a_l = (-(n_l * y_axis_w).sum(dim=-1)).clamp(0.0, 1.0)
    a_r = ((n_r * y_axis_w).sum(dim=-1)).clamp(0.0, 1.0)
    gate = mdp.object_grasped(env, finger1_sensor_cfg, finger2_sensor_cfg, threshold)
    # 수평 factor: box_upright = 1 - z_box·z_world (0=수평). success 조건과 같은 15도(0.034)에서 0
    upright_factor = (1.0 - box_upright(env, object_cfg) / upright_std).clamp(0.0, 1.0)
    return torch.minimum(a_l, a_r) * gate * upright_factor


def plate_face_tilt(
    env,
    finger1_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_plate_object_contact"),
    finger2_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_plate_object_contact"),
    threshold: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """판 법선(TCP z축)과 담당 박스 면 법선 사이 각도(rad)의 양 판 평균. 양 판 접촉(G) 중에만 (면-면 파지 페널티).

    Wide-RollAlign 2999는 들 때 판을 V자로 젖혀(중앙값 최대 ~25도, 법선이 위로) 아래 모서리로 박스를 받친다.
    plate_face_alignment(cos, 양의 보상)는 25도에서도 0.91이라 차이가 작아 각도 자체를 쓴다.
    V자는 양 판 법선이 함께 위로 향해서 박스를 굴려 면을 맞추는 것으로는 줄지 않는다(박스 기울기는 box_upright가 따로 본다).
    """
    ang_l, ang_r = _plate_face_angles(env, object_cfg, ee_frame_cfg)
    gate = mdp.object_grasped(env, finger1_sensor_cfg, finger2_sensor_cfg, threshold)
    return 0.5 * (ang_l + ang_r) * gate


def _plate_face_angles(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """판별 판 법선(TCP z축)과 담당 박스 면 법선 사이 각도(rad), (왼판, 오른판). plate_face_tilt 계열 공용."""
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    li, ri, y_axis_w, _, _, _, _ = _face_dists(env, object_cfg, ee_frame_cfg)
    z_axis = torch.zeros_like(obj.data.root_pos_w.torch)
    z_axis[:, 2] = 1.0
    n_l = quat_apply(ee_frame.data.target_quat_w.torch[:, li], z_axis)  # 왼판은 박스 -y 방향을 봐야 한다
    n_r = quat_apply(ee_frame.data.target_quat_w.torch[:, ri], z_axis)
    ang_l = torch.acos((-(n_l * y_axis_w).sum(dim=-1)).clamp(-1.0, 1.0))
    ang_r = torch.acos(((n_r * y_axis_w).sum(dim=-1)).clamp(-1.0, 1.0))
    return ang_l, ang_r


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
    p = ee_frame.data.target_pos_w.torch
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
    p = ee_frame.data.target_pos_w.torch
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
        force = torch.norm(sensor.data.force_matrix_w.torch[:, :, filter_idx, :], dim=-1).amax(dim=-1)
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
    diff = (robot.data.joint_pos.torch[:, l_idx] - robot.data.joint_pos.torch[:, r_idx]).abs()
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

    right_pos = ee_frame.data.target_pos_w.torch[:, ri]
    left_pos = ee_frame.data.target_pos_w.torch[:, li]
    grip = right_pos - left_pos  # (N, 3)

    y_ax = torch.zeros(grip.shape[0], 3, device=grip.device)
    y_ax[:, 1] = 1.0
    ry = quat_apply(robot.data.root_quat_w.torch, y_ax)  # robot +y in world

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
    Lab 3.0부터 quat 순서가 (x, y, z, w)라 출력도 xyzw이고 w는 마지막 성분이다.
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    _, quat = subtract_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, obj.data.root_pos_w.torch, obj.data.root_quat_w.torch
    )
    sign = torch.where(quat[:, 3:4] >= 0.0, 1.0, -1.0)
    return quat * sign


def _held_at_reset(env, key: str, value: torch.Tensor) -> torch.Tensor:
    """에피소드 첫 관측(episode_length_buf == 0)의 값을 env에 저장해 두고, 그 뒤로는 저장한 값을 돌려준다."""
    held = env.__dict__.setdefault("_obs_held_at_reset", {})
    if key not in held or held[key].shape != value.shape:
        held[key] = value.clone()
    fresh = env.episode_length_buf == 0
    held[key][fresh] = value[fresh]
    return held[key].clone()


def object_position_at_reset(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """에피소드 시작 때의 박스 위치(robot root frame)를 에피소드 내내 그대로 준다 (num_envs, 3).

    비전 없이 박스 pose를 놓은 위치로 고정 publish하는 실물 조건. 들기 시작한 뒤의 박스 위치는 관측하지 않는다.
    """
    return _held_at_reset(env, "object_position", mdp.object_position_in_robot_root_frame(env, robot_cfg, object_cfg))


def object_orientation_at_reset(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """에피소드 시작 때의 박스 orientation(xyzw, w>=0)을 에피소드 내내 그대로 준다 (num_envs, 4)."""
    return _held_at_reset(env, "object_orientation", object_orientation_in_robot_root_frame(env, robot_cfg, object_cfg))


# 비전 박스 pose의 지연·끊김 (-BoxDrop). 20 Hz policy step 기준.
BOX_OBS_DELAY_STEPS = 3  # 에피소드마다 0~3 step(0~150 ms) 지연을 뽑아 고정
BOX_OBS_DROP_START_PROB = 0.02  # 끊기지 않은 step마다 끊김이 시작될 확률 (평균 2.5 s에 한 번)
BOX_OBS_DROP_STEPS = (2, 20)  # 끊김 길이 0.1~1.0 s, 그동안 마지막 값을 유지
BOX_OBS_FULL_DROP_PROB = 0.1  # 에피소드의 10%는 임의 시점(0.05~2 s)부터 끝까지 끊김


def _box_obs_dropout_mask(env) -> dict:
    """박스 pose 관측의 env별 지연·끊김 상태. 한 policy step에 한 번만 갱신해 위치·자세가 같은 mask를 쓴다."""
    st = env.__dict__.get("_box_obs_dropout")
    n, dev = env.num_envs, env.device
    if st is None:
        st = {
            "left": torch.zeros(n, dtype=torch.long, device=dev),  # 남은 끊김 step
            "delay": torch.zeros(n, dtype=torch.long, device=dev),
            "full_at": torch.full((n,), -1, dtype=torch.long, device=dev),
            "drop": torch.zeros(n, dtype=torch.bool, device=dev),
            "hist": {},
            "held": {},
            "step": None,
        }
        env._box_obs_dropout = st
    if st["step"] == env.common_step_counter:
        return st
    st["step"] = env.common_step_counter
    t = env.episode_length_buf
    fresh = t == 0
    k = int(fresh.sum())
    if k:
        st["delay"][fresh] = torch.randint(0, BOX_OBS_DELAY_STEPS + 1, (k,), device=dev)
        full = torch.rand(k, device=dev) < BOX_OBS_FULL_DROP_PROB
        st["full_at"][fresh] = torch.where(full, torch.randint(1, 41, (k,), device=dev), torch.full_like(st["left"][fresh], -1))
        st["left"][fresh] = 0
    going = st["left"] > 0
    st["left"] = (st["left"] - 1).clamp(min=0)
    start = ~going & ~fresh & (torch.rand(n, device=dev) < BOX_OBS_DROP_START_PROB)
    lo, hi = BOX_OBS_DROP_STEPS
    st["left"] = torch.where(start, torch.randint(lo - 1, hi, (n,), device=dev), st["left"])
    full_now = (st["full_at"] >= 0) & (t >= st["full_at"])
    st["drop"] = (going | start | full_now) & ~fresh
    return st


def _box_obs_with_dropout(env, key: str, value: torch.Tensor) -> torch.Tensor:
    """value를 env별 지연만큼 늦추고, 끊긴 step에는 마지막으로 들어온 값을 준다. 에피소드 첫 관측은 항상 현재 값이다."""
    st = _box_obs_dropout_mask(env)
    fresh = env.episode_length_buf == 0
    hist = st["hist"].get(key)
    if hist is None or hist.shape[1:] != value.shape:
        hist = value.unsqueeze(0).repeat(BOX_OBS_DELAY_STEPS + 1, 1, 1)
    hist = torch.cat([value.unsqueeze(0), hist[:-1]], 0)  # hist[0]이 최신
    hist[:, fresh] = value[fresh]
    st["hist"][key] = hist
    delayed = hist[st["delay"], torch.arange(env.num_envs, device=env.device)]
    held = st["held"].get(key)
    if held is None or held.shape != value.shape:
        held = delayed.clone()
    held = torch.where(st["drop"].unsqueeze(-1), held, delayed)
    held[fresh] = value[fresh]
    st["held"][key] = held
    return held.clone()


def object_position_dropout(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """박스 위치(robot root frame)에 비전 지연·끊김을 넣은 관측 (num_envs, 3). 설정은 BOX_OBS_* 상수."""
    return _box_obs_with_dropout(env, "object_position", mdp.object_position_in_robot_root_frame(env, robot_cfg, object_cfg))


def object_orientation_dropout(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """박스 orientation(xyzw, w>=0)에 위치와 같은 지연·끊김을 넣은 관측 (num_envs, 4)."""
    return _box_obs_with_dropout(env, "object_orientation", object_orientation_in_robot_root_frame(env, robot_cfg, object_cfg))


def box_upright(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """박스 기울기 페널티: 1 - z_box·z_world (0=수평 ~ 2=뒤집힘). 페널티 weight로 사용.

    v6: 들고 나르는 중 박스가 회전한 채 운반되는 문제 대응. 문헌(arXiv:2310.03191)의
    |roll|+|pitch| 수평 유지 항에 해당. 바닥 안착 시 자연히 0이라 gate 불필요, yaw는 자유.
    """
    obj = env.scene[object_cfg.name]
    z_axis = torch.zeros_like(obj.data.root_pos_w.torch)
    z_axis[:, 2] = 1.0
    box_z = quat_apply(obj.data.root_quat_w.torch, z_axis)
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
    x_axis = torch.zeros_like(obj.data.root_pos_w.torch)
    x_axis[:, 0] = 1.0
    bx = quat_apply(obj.data.root_quat_w.torch, x_axis)  # 박스 x (world)
    rx = quat_apply(robot.data.root_quat_w.torch, x_axis)  # base x (world)
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
        forces.append(torch.norm(sensor.data.force_matrix_w.torch, dim=-1).amax(dim=(-1, -2)))
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
    pos = obj.data.root_pos_w.torch - env.scene.env_origins
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
        goal_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, cmd)
        near = torch.norm(goal_w - obj.data.root_pos_w.torch, dim=1) < dist_threshold
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
    return (obj.data.root_pos_w.torch[:, 2] > half_z + lift_delta).float()


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
    y = torch.zeros_like(obj.data.root_pos_w.torch)
    y[:, 1] = 1.0
    n_face = quat_apply(obj.data.root_quat_w.torch, y)  # 박스 +y면 법선 (world, unit)
    qs = []
    for cfg in (finger1_sensor_cfg, finger2_sensor_cfg):
        sensor = env.scene.sensors[cfg.name]
        f = torch.nan_to_num(sensor.data.force_matrix_w.torch.reshape(env.num_envs, -1, 3).sum(dim=1))
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
    y_axis = torch.zeros_like(obj.data.root_pos_w.torch)
    y_axis[:, 1] = 1.0
    y_axis_w = quat_apply(obj.data.root_quat_w.torch, y_axis)
    z_axis = torch.zeros_like(obj.data.root_pos_w.torch)
    z_axis[:, 2] = 1.0
    n_l = quat_apply(ee_frame.data.target_quat_w.torch[:, li], z_axis)
    n_r = quat_apply(ee_frame.data.target_quat_w.torch[:, ri], z_axis)
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
    progress = ((obj.data.root_pos_w.torch[:, 2] - half_z) / lift_delta).clamp(0.0, 1.0)
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
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3])
    distance = torch.norm(des_pos_w - obj.data.root_pos_w.torch, dim=1)
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
        forces.append(torch.norm(sensor.data.force_matrix_w.torch, dim=-1).amax(dim=(-1, -2)))
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
            self.robot.data.root_pos_w.torch[env_ids],
            self.robot.data.root_quat_w.torch[env_ids],
            obj.data.root_pos_w.torch[env_ids],
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
        self.sim.physics.gpu_total_aggregate_pairs_capacity = 64 * 1024  # Lab 3.0: sim.physx -> sim.physics

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
            # Lab 3.0 quat은 (x, y, z, w): identity = [0, 0, 0, 1] ([1, 0, 0, 0]이면 x축 180도 뒤집힘)
            init_state=RigidObjectCfg.InitialStateCfg(pos=BOX_SPAWN_POS_W, rot=[0, 0, 0, 1]),
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
                    # 단위는 deg/s (m/s 아님). 2026-11-04까지 30(=0.52 rad/s)이라 박스가 판 사이에서 거의 못 돌았고,
                    # 외란 토크도 사실상 무효였다. 1000 deg/s(17 rad/s, lift_cube와 같음)는 상한 역할만 한다.
                    max_angular_velocity=1000.0,
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


def ramp_event_params_by_steps(env, env_ids, term_name: str, start: dict, end: dict, ramp_steps: int):
    """첫 호출부터 ramp_steps env-step에 걸쳐 event term_name의 params를 start에서 end로 선형으로 바꾼다.

    반환값(Curriculum/<curriculum 이름>)은 end의 첫 항목의 현재 값이다.
    """
    t0 = env.__dict__.setdefault("_event_ramp_start", {}).setdefault(term_name, env.common_step_counter)
    frac = min(1.0, (env.common_step_counter - t0) / ramp_steps)
    cfg = env.event_manager.get_term_cfg(term_name)
    for k in end:
        cfg.params[k] = start[k] + (end[k] - start[k]) * frac
    env.event_manager.set_term_cfg(term_name, cfg)
    return cfg.params[next(iter(end))]


def ramp_reward_weight_on_success(
    env, env_ids, term_name: str, weight: float, success_term: str = "success", trigger: float = 3.0, ramp_steps: int = 4800
):
    """success EMA가 처음 trigger에 닿은 뒤 ramp_steps env-step에 걸쳐 term_name의 weight를 cfg 값에서 weight로 선형으로 올린다.

    ramp_yawtrack_on_success와 같은 방식(발동은 success, 속도는 step 수). 반환값은 Curriculum/<curriculum 이름>에 기록된다.
    """
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    st = env.__dict__.setdefault("_reward_ramp", {}).setdefault(term_name, {"w0": term_cfg.weight, "start": None})
    if term_cfg.weight == weight:
        return weight
    cur = env.common_step_counter
    if st["start"] is None:
        if not hasattr(env, "_lift_curriculum_ema"):
            env._lift_curriculum_ema = {}
        success_level = env.reward_manager._episode_sums[success_term][env_ids].mean().item()
        alpha = min(0.5, 3.0 * len(env_ids) / env.num_envs)
        key = f"_ramp_{term_name}_succ"
        ema = env._lift_curriculum_ema.get(key, 0.0)
        ema = ema + alpha * (success_level - ema)
        env._lift_curriculum_ema[key] = ema
        if ema < trigger:
            return term_cfg.weight
        st["start"] = cur
    scale = min(1.0, (cur - st["start"]) / ramp_steps)
    term_cfg.weight = st["w0"] + (weight - st["w0"]) * scale
    env.reward_manager.set_term_cfg(term_name, term_cfg)
    return term_cfg.weight


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
    dz = (ee_frame.data.target_pos_w.torch[:, li, 2] - ee_frame.data.target_pos_w.torch[:, ri, 2]).abs()
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
class DualFR3PlateBoxLiftEnvCfg_P47_Friction(DualFR3PlateBoxLiftEnvCfg_P47):
    """P47-Friction = P47 + 팔 관절 armature/friction/viscous 명시. from-scratch 재학습.

    변경 사항 (2026-10-21):
      arms_shoulder/arms_forearm에 armature 0.1, friction 0.2, dynamic_friction 0.2,
      viscous_friction 0.003 (Reach Sim2Real *-Friction과 동일 값, FR3_ARM_MUJOCO_PASSIVE_DYNAMICS).
      P47은 armature/viscous를 지정하지 않아 0이고 마찰은 USD의 physxJoint:jointFriction(0.2)에 의존.
    액추에이터는 P47 그대로 implicit PD @100Hz(정책 50Hz), PD 게인 DR ±15%도 유지.
    reward/curriculum 등 P47 전 항목 불변. P47 checkpoint와 관절 dynamics가 달라 이어받기 불가.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = with_arm_passive_dynamics(self.scene.robot)


@configclass
class DualFR3PlateBoxLiftEnvCfg_P47_Friction_PLAY(DualFR3PlateBoxLiftEnvCfg_P47_Friction):
    """P47-Friction visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_P47_Friction_PLAY_NOISY(DualFR3PlateBoxLiftEnvCfg_P47_Friction_PLAY):
    """P47-Friction robustness-evaluation cfg with training observation corruption enabled."""

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


def box_anti_yaw(
    env,
    finger1_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_plate_object_contact"),
    finger2_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_plate_object_contact"),
    threshold: float = 1.0,
    lift_delta: float = LIFT_DELTA,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """[P52] 들기 중 박스 yaw 회전 직접 페널티: |box_x vs base_x| (rad). G-gate + lifted gate.

    P51 play: face_alignment가 박스에 비대칭 토크를 가해 들기 중 box yaw 회전 유발.
    EE 자세를 건드리지 않고 박스 결과(yaw)를 직접 압박.
    G-gate(양판 접촉) + lifted gate(들린 이후)로 파지+들기 중에만 활성.
    face_alignment 계열과 달리 TCP geometry에 영향 없어 grip-axis 왜곡 리스크 없음.
    curriculum: -1e-4 → lifting EMA>10 이후 -2.0.
    """
    gate = mdp.object_grasped(env, finger1_sensor_cfg, finger2_sensor_cfg, threshold)
    lifted = _lifted_perenv(env, lift_delta, object_cfg)
    yaw_err = _box_yaw_error(env, robot_cfg, object_cfg)
    return gate * lifted * yaw_err


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


@configclass
class DualFR3PlateBoxLiftEnvCfg_P52(DualFR3PlateBoxLiftEnvCfg_P47):
    """P52 = P47 + box_anti_yaw 페널티 (G-gate + lifted gate, curriculum).

    변경 사항 (2026-08-25):
      P51 play: 들기 중 box yaw 회전 관찰. P49~P51 face_alignment 계열은 EE 자세를
      건드리며 grip-axis를 왜곡(P47 3.3° → P51 46.1°)하는 구조적 문제를 확인.

      새 접근: 박스 결과(yaw deviation)를 직접 페널티 — EE geometry 비개입.
        `box_anti_yaw` = _box_yaw_error (rad) × G-gate × lifted gate
        G-gate: 양판 접촉 중에만 → 접근 phase 무해
        lifted gate: 들린 이후에만 → 바닥 안착 중 yaw DR 간섭 없음

      face_alignment와의 차이:
        face_alignment: EE z축이 box face normal을 향하게 → TCP 자세 변화 → grip-axis 왜곡
        box_anti_yaw:   box 결과만 측정 → EE 자세 미관여 → grip-axis 영향 최소화 예상

    curriculum: -1e-4 → lifting EMA>10 이후 -2.0.
    P47 전 항목 불변 (plate_height_symmetry -8.0, yawtrack trigger=1.5 포함).
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.box_anti_yaw = RewTerm(
            func=box_anti_yaw, weight=-1e-4
        )
        self.curriculum.box_anti_yaw = CurrTerm(
            func=modify_reward_weight_on_lifting_learned,
            params={"term_name": "box_anti_yaw", "weight": -2.0, "threshold": 10.0},
        )


@configclass
class DualFR3PlateBoxLiftEnvCfg_P52_PLAY(DualFR3PlateBoxLiftEnvCfg_P52):
    """P52 visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_P52_PLAY_NOISY(DualFR3PlateBoxLiftEnvCfg_P52_PLAY):
    """P52 robustness-evaluation cfg with training observation corruption enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True


##
# Sim2Real: P47 + Reach Sim2Real v2 제어 계약 (2026-10-22) — 로그 dual_fr3_plate_lift_sim2real_{1khz,100hz}
##


def _apply_plate_sim2real(env_cfg, *, physics_dt: float) -> None:
    """P47에 Reach Sim2Real v2 제어 계약을 적용하고, 정책 주기 50 Hz -> 20 Hz에 따른 box 전용 보정을 한다."""
    apply_sim2real_control(env_cfg, physics_dt=physics_dt)
    # 관절 순서를 Reach Sim2Real 계약(left1~7, right1~7)으로 고정. P47의 regex 순서는 articulation
    # 순서(좌우 교차)라 배포 노드 순서와 다르다. action과 joint obs를 같은 순서로 맞춘다.
    env_cfg.actions.arm_action.joint_names = DUAL_ARM_JOINT_NAMES
    env_cfg.actions.arm_action.preserve_order = True
    for term in (env_cfg.observations.policy.joint_pos, env_cfg.observations.policy.joint_vel):
        term.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=DUAL_ARM_JOINT_NAMES, preserve_order=True)
    # success 진입 조건은 0.5 s 유지 그대로: 50 Hz 25 step -> 20 Hz 10 step.
    # 그 밖의 reward는 RewardManager가 dt를 곱하고, 커리큘럼은 EMA/iteration 기준이라 주기와 무관.
    policy_dt = env_cfg.sim.dt * env_cfg.decimation
    env_cfg.rewards.success.params["hold_steps"] = round(0.5 / policy_dt)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real1kHz(DualFR3PlateBoxLiftEnvCfg_P47):
    """Sim2Real-1kHz = P47 + Reach Sim2Real v2 제어 계약. from-scratch 재학습.

    P47-Friction은 passive dynamics만 맞춘 것이고, 이 cfg는 제어 경로 전체를 맞춘다:
      물리/PD 100 Hz -> 1 kHz, 정책 50 Hz -> 20 Hz (decimation 2 -> 50), implicit PD -> explicit PD +
      토크 87/12 Nm + 1000 Nm/s 변화율 제한 + MuJoCo-matched passive dynamics, 팔 action ±0.1 rad clip +
      policy-step target, actor tanh-squashed Gaussian(runner cfg), 관절 순서 left1~7/right1~7.
      success hold 25 -> 10 step (0.5 s 유지). PD 게인 DR ±15%는 P47에도 있던 것과 같은 값.
    episode 5 s(= 100 policy step), reward/curriculum/물체 DR은 P47 그대로.
    ⚠ 정책 주기가 바뀌어 P47 보상 튜닝이 20 Hz에서도 그대로 통하는지는 학습으로 확인해야 함
      (cube는 v1->v2에서 50 -> 20 Hz 전환이 문제없었음).
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_plate_sim2real(self, physics_dt=0.001)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real1kHz_PLAY(DualFR3PlateBoxLiftEnvCfg_Sim2Real1kHz):
    """Sim2Real-1kHz visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real1kHz_PLAY_NOISY(DualFR3PlateBoxLiftEnvCfg_Sim2Real1kHz_PLAY):
    """Sim2Real-1kHz robustness-evaluation cfg with training observation corruption enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz(DualFR3PlateBoxLiftEnvCfg_P47):
    """Sim2Real-100Hz = Sim2Real-1kHz와 같되 물리/PD 100 Hz(decimation 5). 1 kHz가 너무 느릴 때의 대안."""

    def __post_init__(self):
        super().__post_init__()
        _apply_plate_sim2real(self, physics_dt=0.01)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_PLAY(DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz):
    """Sim2Real-100Hz visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_PLAY_NOISY(DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_PLAY):
    """Sim2Real-100Hz robustness-evaluation cfg with training observation corruption enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Offset03(DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz):
    """Sim2Real-100Hz와 같되 팔 action offset 상한만 ±0.1 -> ±0.3 rad. box Sim2Real 본 학습용.

    measured-relative PD라 접촉으로 막힌 관절의 최대 PD 토크 = Kp * offset (80 * 0.1 = 8 Nm -> 24 Nm).
    ±0.1 rad(8 Nm)로는 4 kg box를 못 듦 (A/B 확인 2026-10-23). Reach/cube 계약(±0.1 rad)과 다르므로
    box 배포 노드에 scale/clip/fault 변경이 필요 — 체크리스트: docs/dual_fr3_lift_sim2real.md.
    tanh actor 출력이 (-1, 1)이므로 clip만 넓히면 효과가 없어 scale도 같이 0.3으로 올린다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action.scale = 0.3
        self.actions.arm_action.clip = {".*": (-0.3, 0.3)}


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Offset03_PLAY(DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Offset03):
    """Sim2Real-100Hz-Offset03 visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


# ---------- Offset03-Level: 수평 들기 gate + FR3 관절 한계 여유 (2026-10-24) ----------
# Offset03 model_2000은 박스를 굴려서 든다(들고 있는 동안 기울기 평균 46~57도). 들기/goal 보상에
# 기울기 gate가 없어서 box_upright(-3)를 감수해도 기울여 드는 쪽이 이득이었다. 조임 힘 테스트로는
# 판당 ~44 N이면 박스 μ 0.45 이상(4.5 kg은 0.6 이상)에서 수평 들기가 가능하므로 물리 한계가 아니다.
LEVEL_TILT_GATE = 0.293  # 1 - cos(45도): 이 기울기에서 들기/goal 보상 0, 수평이면 그대로
# FR3 관절 위치 한계 [rad]: Franka FCI 문서 "Robot and interface specifications"의 q_min/q_max.
# (2026-11-03 정정: 처음에 "datasheet"라고 불렀지만 제품 datasheet R02212 2.6은 A1 ±166°, A2 ±105°,
# A4 -176/-7°, A6 25/265° 등으로 더 넓고, franka_description URDF·sim USD·배포 노드 검사 값이 그쪽이다.)
# 실로봇이 어느 값에서 멈추는지는 실물에서 확인하지 않았다. 둘 중 좁은 쪽이라 이 값으로 자른다.
# Offset03 model_2000은 왼팔 j1·j2가 env-step의 약 80%에서 이 한계를 넘어 USD 한계에 붙어 있었다.
FR3_JOINT_POS_LIMITS = (
    (-2.7437, 2.7437), (-1.7837, 1.7837), (-2.9007, 2.9007), (-3.0421, -0.1518),
    (-2.8065, 2.8065), (0.5445, 4.5169), (-3.0159, 3.0159),
)


def _level_factor(env, object_cfg: SceneEntityCfg, tilt_gate: float) -> torch.Tensor:
    """수평 1 -> box_upright가 tilt_gate일 때 0 (plate_face_alignment의 upright_factor와 같은 형태)."""
    return (1.0 - box_upright(env, object_cfg) / tilt_gate).clamp(0.0, 1.0)


def object_is_lifted_perenv_and_squeezed_level(
    env,
    lift_delta: float,
    finger1_sensor_cfg: SceneEntityCfg,
    finger2_sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    tilt_gate: float = LEVEL_TILT_GATE,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """object_is_lifted_perenv_and_squeezed x 수평 factor."""
    lifted = object_is_lifted_perenv_and_squeezed(
        env, lift_delta, finger1_sensor_cfg, finger2_sensor_cfg, threshold, object_cfg
    )
    return lifted * _level_factor(env, object_cfg, tilt_gate)


def lift_progress_level(
    env,
    lift_delta: float,
    finger1_sensor_cfg: SceneEntityCfg,
    finger2_sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    tilt_gate: float = LEVEL_TILT_GATE,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """lift_progress x 수평 factor. 굴리기만 해도 중심이 올라가 진행도를 받던 경로를 막는다."""
    progress = lift_progress(env, lift_delta, finger1_sensor_cfg, finger2_sensor_cfg, threshold, object_cfg)
    return progress * _level_factor(env, object_cfg, tilt_gate)


def object_goal_distance_perenv_squeezed_level(
    env,
    std: float,
    lift_delta: float,
    command_name: str,
    finger1_sensor_cfg: SceneEntityCfg,
    finger2_sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    tilt_gate: float = LEVEL_TILT_GATE,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """object_goal_distance_perenv_squeezed x 수평 factor."""
    tracking = object_goal_distance_perenv_squeezed(
        env, std, lift_delta, command_name, finger1_sensor_cfg, finger2_sensor_cfg, threshold, robot_cfg, object_cfg
    )
    return tracking * _level_factor(env, object_cfg, tilt_gate)


def fr3_joint_limits_inset(inset: float) -> tuple:
    """FR3_JOINT_POS_LIMITS를 양쪽에서 inset[rad]만큼 좁힌 한 팔(7관절) 한계."""
    return tuple((lo + inset, hi - inset) for lo, hi in FR3_JOINT_POS_LIMITS)


def fr3_joint_limit_margin(
    env,
    margin: float = 0.1,
    inset: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=DUAL_ARM_JOINT_NAMES, preserve_order=True),
) -> torch.Tensor:
    """FR3 한계(inset만큼 좁힌 값)까지 거리가 margin[rad] 미만인 관절마다 ((margin - 거리) / margin)^2, 양팔 합.

    한계에서 1, 한계를 넘으면 더 커진다(0.1 rad 넘으면 4). asset_cfg 순서는 left1~7, right1~7이고,
    RewardManager는 params로 넘긴 SceneEntityCfg만 resolve하므로 RewTerm params에 명시해야 한다.
    """
    q = env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids]
    limits = torch.tensor(fr3_joint_limits_inset(inset) * 2, device=q.device)
    dist = torch.minimum(q - limits[:, 0], limits[:, 1] - q)
    return (((margin - dist).clamp(min=0.0) / margin) ** 2).sum(dim=-1)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Offset03_Level(DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Offset03):
    """Offset03 + 수평 들기 gate + FR3 관절 한계 여유 페널티. from-scratch.

    - lifting_object, lift_progress, object_goal_tracking(+fine_grained)에 수평 factor를 곱한다
      (수평 1 -> 45도에서 0). 15도 gate는 success에 이미 있고, 발견 단계의 기울이기는 부분 보상으로 남긴다.
    - fr3_joint_limit_margin(-1.0, 처음부터): FR3 datasheet 한계 0.1 rad 안쪽부터 페널티.
      관절 공간 대부분에서 0이라 탐색을 막지 않는다. 오른팔은 Offset03에서도 한계 안에서 과제를 했다.
    한 run에 두 축을 바꿨다: 실패하면 gate만 / 한계만으로 분리한다.
    """

    def __post_init__(self):
        super().__post_init__()
        level_funcs = {
            object_is_lifted_perenv_and_squeezed: object_is_lifted_perenv_and_squeezed_level,
            lift_progress: lift_progress_level,
            object_goal_distance_perenv_squeezed: object_goal_distance_perenv_squeezed_level,
        }
        for name in ("lifting_object", "lift_progress", "object_goal_tracking", "object_goal_tracking_fine_grained"):
            term = getattr(self.rewards, name)
            term.func = level_funcs[term.func]
        self.rewards.fr3_joint_limit_margin = RewTerm(
            func=fr3_joint_limit_margin,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=DUAL_ARM_JOINT_NAMES, preserve_order=True)},
            weight=-1.0,
        )


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Offset03_Level_PLAY(DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Offset03_Level):
    """Sim2Real-100Hz-Offset03-Level visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


# ---------- Offset03-PlateOnly: 판으로만 들기 + FR3 관절 한계를 sim에 강제 (2026-10-24) ----------
# Level/Offset03 정책 접촉 측정(64 env x 300 step, 1 N 초과): 박스가 들린 스텝 중 왼팔 link4 14%(Level 4500),
# 왼팔 link5 11%(Level 9999), 베이스 13%·왼팔 link2 5%(Offset03 resume 9999)에서 판 외 부위가 박스에 닿았다.
# link7 원통(판과 같은 body라 기존 squeeze 센서는 판 접촉으로 셈)은 ~0%. 기존 undesired_contacts는 -1e-4
# 시작이라 들기 보상을 못 이겼고 베이스는 아예 안 봤다. 실로봇에선 충돌 반사/파손 원인이므로 들기/goal/
# success 보상을 "판 외 접촉이 없을 때"만 준다. Level의 j7은 USD 한계에 붙어 있었다(32%).
PLATE_ONLY_FORCE_REF = 10.0  # [N] 판 외 접촉력 합이 이 값 이상이면 들기/goal/success 보상 0
# [m] link7 frame z: 판 collider는 0.107~0.132(link7 z축 기준 45도 돌아간 140 mm 정사각형, 모서리가 x/y축 위
# 0.099), link7 원통 collider는 z 0.050~0.105. 그래서 접촉점 z만으로 판/원통을 나눈다.
PLATE_CONTACT_Z_MIN = 0.106
LIFT_MONITOR_WEIGHT = 1e-3  # 커리큘럼 발동 판정용 무gate lifting 항 (보상 기여는 무시할 수준)
# 커리큘럼 "들기 습득" 기준 (lifting_object weight 15 단위, 초당). 기존 10은 무gate lift_monitor로도 최대 9.87이라
# 도달 불가였다(run 2026-10-24_21-11-54). 8은 Run C 조기 발동 사고(~4/s)보다 충분히 높다.
LIFT_LEARNED_THRESHOLD = 8.0


def _link7_housing_force(env, sensor_name: str, link7_id: int) -> torch.Tensor:
    """link7-박스 접촉점 중 판 밖(link7 원통)에 있는 점들의 법선력 합 [N], (num_envs,).

    PhysX 센서는 body 단위라 force_matrix_w로는 판과 원통을 못 나눈다 -> 접촉점별 raw 데이터를 link7
    frame으로 옮겨 판 collider 높이보다 아래인 점만 더한다. 센서 body가 env당 1개, filter가 박스 1개라
    counts/starts는 env 순서다.
    """
    sensor = env.scene.sensors[sensor_name]
    forces, points, _, _, counts, starts = sensor.contact_view.get_contact_data(dt=env.physics_dt)
    counts = wp.to_torch(counts).reshape(-1).to(torch.int64)
    out = torch.zeros(env.num_envs, device=env.device)
    total = int(counts.sum())
    if total == 0:
        return out
    starts = wp.to_torch(starts).reshape(-1).to(torch.int64)
    env_ids = torch.repeat_interleave(torch.arange(env.num_envs, device=env.device), counts)
    first = torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
    idx = starts[env_ids] + torch.arange(total, device=env.device) - first
    f = wp.to_torch(forces).reshape(-1)[idx]
    p = wp.to_torch(points).reshape(-1, 3)[idx]
    robot = env.scene["robot"]
    pos = robot.data.body_pos_w.torch[env_ids, link7_id]
    quat = robot.data.body_quat_w.torch[env_ids, link7_id]
    local = quat_apply_inverse(quat, p - pos)
    return out.index_add_(0, env_ids, f * (local[:, 2] < PLATE_CONTACT_Z_MIN).float())


def nonplate_contact_forces(env) -> torch.Tensor:
    """판 외 접촉력 [N], (num_envs, 15): 양팔 link1~6 net(상대 무관) 12 + 베이스-박스 1 + link7 원통-박스 2.

    reward 항 여러 개가 같은 스텝에 부르므로 common_step_counter로 캐시한다.
    """
    cache = getattr(env, "_nonplate_cache", None)
    if cache is not None and cache[0] == env.common_step_counter:
        return cache[1]
    if not hasattr(env, "_link7_ids"):
        robot = env.scene["robot"]
        env._link7_ids = {s: robot.find_bodies(f"{s}_fr3_link7")[0][0] for s in ("left", "right")}
    arms = torch.linalg.norm(env.scene.sensors["arms_contact"].data.net_forces_w.torch, dim=-1)
    base = torch.linalg.norm(env.scene.sensors["base_object_contact"].data.force_matrix_w.torch, dim=-1)
    housing = [_link7_housing_force(env, f"{s}_plate_object_contact", env._link7_ids[s]) for s in ("left", "right")]
    forces = torch.cat([arms, base.sum(dim=(-1, -2)).unsqueeze(-1), torch.stack(housing, dim=-1)], dim=-1)
    env._nonplate_cache = (env.common_step_counter, forces)
    return forces


def _plate_only_factor(env, force_ref: float) -> torch.Tensor:
    """판 외 접촉 없음 1 -> 접촉력 합이 force_ref일 때 0."""
    return (1.0 - nonplate_contact_forces(env).sum(dim=-1) / force_ref).clamp(0.0, 1.0)


def nonplate_contact_count(env, threshold: float = 1.0) -> torch.Tensor:
    """threshold[N]를 넘는 판 외 접촉 부위 수. undesired_contacts(link1~6)에 베이스·link7 원통을 더한 것."""
    return (nonplate_contact_forces(env) > threshold).sum(dim=-1).float()


def object_is_lifted_perenv_and_squeezed_plateonly(
    env,
    lift_delta: float,
    finger1_sensor_cfg: SceneEntityCfg,
    finger2_sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    tilt_gate: float = LEVEL_TILT_GATE,
    force_ref: float = PLATE_ONLY_FORCE_REF,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """수평 gate 들기 x 판 전용 factor."""
    lifted = object_is_lifted_perenv_and_squeezed_level(
        env, lift_delta, finger1_sensor_cfg, finger2_sensor_cfg, threshold, tilt_gate, object_cfg
    )
    return lifted * _plate_only_factor(env, force_ref)


def lift_progress_plateonly(
    env,
    lift_delta: float,
    finger1_sensor_cfg: SceneEntityCfg,
    finger2_sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    tilt_gate: float = LEVEL_TILT_GATE,
    force_ref: float = PLATE_ONLY_FORCE_REF,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """수평 gate 들기 진행도 x 판 전용 factor."""
    progress = lift_progress_level(env, lift_delta, finger1_sensor_cfg, finger2_sensor_cfg, threshold, tilt_gate, object_cfg)
    return progress * _plate_only_factor(env, force_ref)


def object_goal_distance_perenv_squeezed_plateonly(
    env,
    std: float,
    lift_delta: float,
    command_name: str,
    finger1_sensor_cfg: SceneEntityCfg,
    finger2_sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    tilt_gate: float = LEVEL_TILT_GATE,
    force_ref: float = PLATE_ONLY_FORCE_REF,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """수평 gate goal 추종 x 판 전용 factor."""
    tracking = object_goal_distance_perenv_squeezed_level(
        env, std, lift_delta, command_name, finger1_sensor_cfg, finger2_sensor_cfg, threshold, tilt_gate,
        robot_cfg, object_cfg,
    )
    return tracking * _plate_only_factor(env, force_ref)


class success_bonus_plateonly(success_bonus):
    """success_bonus x 판 전용 factor (지급만 깎고 hold 카운터는 그대로)."""

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
        force_ref: float = PLATE_ONLY_FORCE_REF,
    ) -> torch.Tensor:
        reward = super().__call__(
            env, dist_threshold, tilt_threshold, hold_steps, finger1_sensor_cfg, finger2_sensor_cfg, threshold,
            command_name, robot_cfg, object_cfg, sustained, yaw_alpha, yaw_std,
        )
        return reward * _plate_only_factor(env, force_ref)


def set_fr3_datasheet_joint_limits(
    env,
    env_ids,
    inset: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=DUAL_ARM_JOINT_NAMES, preserve_order=True),
):
    """startup: sim 관절 위치 한계(USD)를 FR3 한계(FR3_JOINT_POS_LIMITS, inset만큼 더 좁힘)로 바꾼다.

    asset_cfg는 params로 넘겨야 resolve된다.
    """
    robot = env.scene[asset_cfg.name]
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)
    limits = torch.tensor(fr3_joint_limits_inset(inset) * 2, device=env.device).repeat(len(env_ids), 1, 1)
    robot.write_joint_position_limit_to_sim_index(limits=limits, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Offset03_PlateOnly(DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Offset03_Level):
    """Offset03-Level + 판으로만 들기 + FR3 관절 한계 강제 + 커리큘럼 발동 수정. from-scratch.

    - 들기/goal/success 보상에 판 전용 factor를 곱한다: 판 외 접촉력 합(양팔 link1~6, 베이스-박스,
      link7 원통-박스) 0 N이면 1, PLATE_ONLY_FORCE_REF(10 N)면 0. 수평 gate는 Level 그대로.
    - undesired_contact_penalty를 nonplate_contact_count로 교체 (link1~6에 베이스·link7 원통 추가,
      -1e-4 시작 -> 들기 습득 후 -1 커리큘럼은 그대로).
    - sim 관절 한계를 FR3 datasheet로 좁히고(startup event) PD 목표도 같은 한계로 자른다. 배포 노드도
      q_target을 같은 한계로 잘라야 한다. fr3_joint_limit_margin(-1)은 유지.
    - Level에서 lifting_object에 gate를 곱하자 "들기 습득" 판정(lifting_object 초당 EMA > 10)이 끝까지
      안 넘어서 페널티 커리큘럼이 하나도 안 켜졌다 -> gate 없는 lift_monitor 항으로 판정한다. 기준도
      10 -> LIFT_LEARNED_THRESHOLD(8)로 낮춘다 (첫 from-scratch run은 10이라 끝까지 안 켜졌음).
    """

    def __post_init__(self):
        super().__post_init__()
        plate_only_funcs = {
            object_is_lifted_perenv_and_squeezed_level: object_is_lifted_perenv_and_squeezed_plateonly,
            lift_progress_level: lift_progress_plateonly,
            object_goal_distance_perenv_squeezed_level: object_goal_distance_perenv_squeezed_plateonly,
        }
        for name in ("lifting_object", "lift_progress", "object_goal_tracking", "object_goal_tracking_fine_grained"):
            term = getattr(self.rewards, name)
            term.func = plate_only_funcs[term.func]
        self.rewards.success.func = success_bonus_plateonly

        self.scene.base_object_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        # 판 + link7 원통이 같이 닿으면 접촉점이 기본 4개를 넘을 수 있어 여유를 둔다
        self.scene.left_plate_object_contact.max_contact_data_count_per_prim = 16
        self.scene.right_plate_object_contact.max_contact_data_count_per_prim = 16
        self.rewards.undesired_contact_penalty.func = nonplate_contact_count
        self.rewards.undesired_contact_penalty.params = {"threshold": 1.0}

        self.events.fr3_joint_limits = EventTerm(
            func=set_fr3_datasheet_joint_limits,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=DUAL_ARM_JOINT_NAMES, preserve_order=True)},
        )
        self.actions.arm_action.target_limits = FR3_JOINT_POS_LIMITS * 2

        lifting = self.rewards.lifting_object
        self.rewards.lift_monitor = RewTerm(
            func=object_is_lifted_perenv_and_squeezed,
            params={k: v for k, v in lifting.params.items() if k != "tilt_gate"},
            weight=LIFT_MONITOR_WEIGHT,
        )
        for term in self.curriculum.__dict__.values():
            if term is not None and term.func is modify_reward_weight_on_lifting_learned:
                term.params["lifting_term"] = "lift_monitor"
                term.params["threshold"] = LIFT_LEARNED_THRESHOLD * LIFT_MONITOR_WEIGHT / lifting.weight


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Offset03_PlateOnly_PLAY(DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Offset03_PlateOnly):
    """Sim2Real-100Hz-Offset03-PlateOnly visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


# ---------- Light: PlateOnly + reach/cube와 같은 ±0.1 rad + 실물 가벼운 박스 (2026-11-02) ----------
# 사용자: 실물 박스는 1 kg 미만이다 (7/29에 정한 4 kg과 다름). ±0.1 rad(Kp*0.1 = 8 Nm)로 4 kg은 못 들지만
# (A/B 2026-10-23), 스크립트 조임-들기 시험(scratch squeeze_test3, 판을 박스 중간 높이에 대고 DLS로 조인 뒤
# 20 cm 들기)에서 판당 조임이 16~22 N이었다. 0.5 kg은 박스 마찰 0.4~1.0 전부, 1.0 kg은 0.6 이상에서
# 12~16 cm 들렸고, 1.5~2 kg은 3~7 cm에서 미끄러졌고, 4 kg은 전혀 안 들렸다.
# 이 박스면 배포 노드 기준(max_policy_target_delta_rad 0.11)을 reach와 같이 그대로 쓴다.
LIGHT_BOX_MASS = 0.75  # kg, 명목값 (실측 전)
LIGHT_BOX_MASS_RANGE = (0.5, 1.0)  # kg, 실측하면 좁힌다


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light(DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Offset03_PlateOnly):
    """PlateOnly(판으로만 들기, datasheet 관절 한계, 커리큘럼 threshold 8)에서 action bound를 ±0.1 rad로
    되돌리고 박스를 0.5~1.0 kg으로 바꾼다. 외란은 질량에 비례해 줄인다 (4 kg 기준 ±5 N/±1 Nm와 같은 가속도).
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action.scale = 0.1
        self.actions.arm_action.clip = {".*": (-0.1, 0.1)}
        self.scene.object.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=LIGHT_BOX_MASS)
        self.events.randomize_box_density.params["mass_distribution_params"] = LIGHT_BOX_MASS_RANGE
        ratio = LIGHT_BOX_MASS / V1_BOX_MASS
        self.events.disturb_box.params.update(force=DISTURB_FORCE * ratio, torque=DISTURB_TORQUE * ratio)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_PLAY(DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light):
    """Sim2Real-100Hz-Light visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


# ---------- Light-Margin: 관절 한계를 FCI 한계보다 안쪽으로 (2026-11-03) ----------
# Light model_5999는 j2를 상한(1.7837)에 붙여 쓴다(env 62~75%, 시간 3~9%). sim 한계·PD 목표 clamp·한계
# 페널티를 모두 FR3_JOINT_LIMIT_INSET만큼 안쪽으로 옮긴다. j2는 1.7337이 되어 URDF/datasheet 한계(1.8361)와
# 0.10, 배포 노드 검사(URDF - 0.02)와 0.08 rad 떨어진다. model_5999에서 이어 학습한다.
FR3_JOINT_LIMIT_INSET = 0.05  # rad


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Margin(DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light):
    """Light + 관절 한계(sim 한계, q_target clamp, 한계 페널티)를 FR3_JOINT_LIMIT_INSET만큼 안쪽으로."""

    def __post_init__(self):
        super().__post_init__()
        self.events.fr3_joint_limits.params["inset"] = FR3_JOINT_LIMIT_INSET
        self.actions.arm_action.target_limits = fr3_joint_limits_inset(FR3_JOINT_LIMIT_INSET) * 2
        self.rewards.fr3_joint_limit_margin.params["inset"] = FR3_JOINT_LIMIT_INSET


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Margin_PLAY(DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Margin):
    """Sim2Real-100Hz-Light-Margin visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


# ---------- Box410: 실물 2차 박스 (2026-11-03 사용자) ----------
# 가로 410 x 세로 310 x 높이 280 mm, 빈 박스 501~561 g, 잡는 방향(y) = 310 mm.
# Light/Light-Margin(555 x 475 x 275, 0.5~1.0 kg)과 다르다. 그 task들과 체크포인트는 555 박스로 남겨 둔다.
BOX410_SIZE = (0.41, 0.31, 0.28)  # (x 전후, y 잡는 방향, z 높이) [m]
BOX410_MASS = 0.5  # kg, 명목값
BOX410_MASS_RANGE = (0.4, 0.6)  # kg, 실측 501~561 g에 여유
BOX410_X_RANGE = (-0.05, 0.02)  # 스폰 x 오프셋 [m] (박스 중심 x 0.45~0.52, Pregrasp 표와 같은 범위)


def _apply_box410(cfg):
    """박스 크기·질량 DR을 410 x 310 x 280 mm, 0.4~0.6 kg으로 바꾸고 외란을 Light처럼 질량에 비례해 줄인다."""
    cfg.scene.object.spawn.size = BOX410_SIZE
    cfg.scene.object.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=BOX410_MASS)
    cfg.scene.object.init_state.pos = (0.5, 0.0, BOX410_SIZE[2] / 2 + 0.0225)
    cfg.events.randomize_box_density.params["mass_distribution_params"] = BOX410_MASS_RANGE
    ratio = BOX410_MASS / V1_BOX_MASS
    cfg.events.disturb_box.params.update(force=DISTURB_FORCE * ratio, torque=DISTURB_TORQUE * ratio)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410(DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Margin):
    """Light-Margin(관절 한계 FCI - 0.05 rad)을 410 박스로 바꾼 것. 바닥의 박스에서 시작해 접근부터 들기까지 전부 RL.

    Pregrasp(접근 분리)와 비교하는 기준선이다. 박스 x는 Pregrasp 표와 같은 0.45~0.52(y ±0.1, yaw ±0.1 그대로).
    에피소드·보상·커리큘럼은 Light 그대로(5 s, 들기 습득 뒤 페널티 threshold 8).
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_box410(self)
        self.events.reset_object_position.params["pose_range"]["x"] = BOX410_X_RANGE


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_PLAY(DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410):
    """Sim2Real-100Hz-Light-Box410 visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


# ---------- Box410-T6: 페널티 커리큘럼 발동 기준 8 -> 6 (2026-11-03) ----------
# Box410 model_4999까지 lift_monitor가 초당 약 7.6에 머물러 threshold 8을 못 넘었다 -> P47의 자세·대칭·접촉·
# 매끄러움 페널티(box_upright -3, plate_height_symmetry -8, plate_approach_symmetry -20, one_sided -5,
# contact -1, action_rate/joint_vel -0.1)가 5000 iter 내내 꺼져 있었다 (Light도 같았음).
# 로그 폴더는 Box410과 같다 (model_4999에서 --resume으로 이어 학습하기 위해).
BOX410_LIFT_LEARNED_THRESHOLD = 6.0


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_T6(DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410):
    """Box410 + 페널티 커리큘럼 발동 기준 BOX410_LIFT_LEARNED_THRESHOLD(6). 그 밖에는 Box410 그대로."""

    def __post_init__(self):
        super().__post_init__()
        for term in self.curriculum.__dict__.values():
            if term is not None and term.func is modify_reward_weight_on_lifting_learned:
                term.params["threshold"] = (
                    BOX410_LIFT_LEARNED_THRESHOLD * LIFT_MONITOR_WEIGHT / self.rewards.lifting_object.weight
                )


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_T6_PLAY(DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_T6):
    """Sim2Real-100Hz-Light-Box410-T6 visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


# ---------- Box410-J1Mirror: j1 좌우 거울 페널티 (2026-11-03) ----------
# Box410 model_4999 진단(256 env, scratchpad lr_symmetry.py): 양팔이 base 기준 같은 방향(로봇 오른쪽)으로 ~90도
# 돌아간다. 끝 시점 j1 L/R = -1.99/-1.17 (ready -0.52/+0.52), j3가 +1.6~1.9로 상쇄해 판 위치는 대칭(|yL+yR| 0.4 cm)
# 이지만 팔 자세는 비대칭이다. 거울 대칭이면 q1_L = -q1_R인데 |q1_L + q1_R|가 끝 시점 3.2 rad. 판/박스 기준 보상만
# 있어서 7자유도 여유로 이 해에 갔다. P40은 j6만 잡아 실패(P41 docstring) -> 어깨 yaw인 j1을 잡는다.
# 박스 yaw(DR ±0.1 rad)를 돌려 잡는 데 필요한 j1 비대칭은 작으므로 J1_MIRROR_DEADBAND까지는 페널티 0.
J1_MIRROR_DEADBAND = 0.15  # rad
J1_MIRROR_WEIGHT = -1.0


def joint_mirror_error(
    env,
    asset_cfg: SceneEntityCfg,
    deadband: float = 0.0,
) -> torch.Tensor:
    """[J1Mirror] 거울 대칭 오차 |q_L + q_R| - deadband (rad, 0 아래는 0). asset_cfg = (left_jN, right_jN) 순서.

    양팔이 base y축 거울 배치(y ±0.215, yaw ±30도)라 j1/j3/j5/j7은 q_L = -q_R이 대칭이다 (ready j1 -0.52/+0.52).
    게이트 없음: 비대칭은 접근 단계에서 생기고, ready가 대칭(오차 0)이라 처음부터 켜도 탐색을 막지 않는다.
    """
    q = env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids]
    return ((q[:, 0] + q[:, 1]).abs() - deadband).clamp(min=0.0)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_J1Mirror(DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_T6):
    """Box410-T6 + j1 거울 페널티(J1_MIRROR_WEIGHT, 처음부터, deadband J1_MIRROR_DEADBAND). from-scratch."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.j1_mirror = RewTerm(
            func=joint_mirror_error,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["left_fr3_joint1", "right_fr3_joint1"], preserve_order=True
                ),
                "deadband": J1_MIRROR_DEADBAND,
            },
            weight=J1_MIRROR_WEIGHT,
        )


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_J1Mirror_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_J1Mirror
):
    """Sim2Real-100Hz-Light-Box410-J1Mirror visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


# ---------- Pregrasp: 접근은 rule-based/MoveIt, RL은 마지막 접근~조이기~들기 (2026-11-03) ----------
# Light RL의 박스 yaw(38도)·판 높이차·j2 한계는 판을 어디에 대느냐(접근)에서 생긴다(P37 "kinematic placement").
# 실물에서는 비전으로 박스 pose를 잡고 MoveIt/MoveToJoint로 판을 박스 면 4 cm 바깥, 잡는 높이에 놓은 뒤 정책을 켠다.
# sim은 그 넘겨받는 상태를 표로 재현한다: scripts/tools/eval_dual_fr3_plate_rule_based.py --pregrasp_gap 0.04로
# 박스 pose별 양팔 관절각을 미리 구해 두고(Light-Margin 관절 한계 안), reset마다 한 줄을 뽑아 박스와 팔을 같이 놓는다.
# 인식·플래너 오차는 박스 pose 노이즈로 넣는다. 박스는 Box410이고, 박스를 바꾸면 표도 그 크기로 다시 만든다(--box_size).
PREGRASP_TABLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "assets", "pregrasp", "plate_box410_pregrasp_gap004_margin005.npz",
)
# 같은 조건 + 판 모서리를 박스 모서리에 맞춘 표(--roll_align, 2026-11-03). 위 표는 판이 면 안에서 약 18도 돌아간 채
# 시작한다(IK가 법선축 회전을 정하지 않았음). Pregrasp model_2999는 위 표로 학습했으므로 그 표는 그대로 둔다.
PREGRASP_TABLE_ROLLALIGN_PATH = PREGRASP_TABLE_PATH.replace(".npz", "_rollalign.npz")
# 실물 배치는 긴 변(410 mm)이 로봇 좌우(y, 판이 잡는 방향)이고 310 mm가 앞뒤다(2026-11-03 정정). 위 Box410 task들은
# 410을 앞뒤로 둔 채 학습했다. 같은 박스를 90도 돌린 것이라 질량·높이·DR은 그대로 쓰고, 표는 이 박스로 다시 만든다.
# 표: eval_dual_fr3_plate_rule_based.py --box_size 0.31 0.41 0.28 --box_dx -0.10 0.0, 박스 중심 x 0.40~0.50
# (0.50 이상은 j2 한계로 유효율이 떨어지고 실물에서도 안 쓴다). 같은 seed로 두 표를 만들어 j7만 다르다.
# - WIDE: 판 회전을 정하지 않은 기존 Pregrasp 방식(판이 면 안에서 약 16도 돌아감).
# - WIDE_ROLLALIGN: --snap_j7, 기록 후 j7만 최대 45도 돌려 판 모서리를 박스에 맞춘다. --roll_align 표는 j7이 72도
#   돌아간 자세였고 그 RollAlign 정책은 박스를 40~67도 비틀었다.
BOX410_WIDE_SIZE = (BOX410_SIZE[1], BOX410_SIZE[0], BOX410_SIZE[2])  # (x 0.31 전후, y 0.41 잡는 방향, z 0.28)
PREGRASP_TABLE_WIDE_PATH = os.path.join(
    os.path.dirname(PREGRASP_TABLE_PATH), "plate_box410wide_pregrasp_gap004_margin005.npz"
)
PREGRASP_TABLE_WIDE_ROLLALIGN_PATH = PREGRASP_TABLE_WIDE_PATH.replace(".npz", "_snapj7.npz")
PREGRASP_BOX_POS_NOISE = 0.01  # m, 박스 xy (표의 pose 기준, 판-면 간격 4 cm 안)
PREGRASP_BOX_YAW_NOISE = 0.05  # rad (약 3도)
PREGRASP_JOINT_NOISE = 0.01  # rad
PREGRASP_EPISODE_S = 4.0


def reset_from_pregrasp_table(
    env,
    env_ids,
    table_path: str,
    box_pos_noise: float,
    box_yaw_noise: float,
    joint_noise: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """reset: 접근 끝 자세 표에서 env마다 한 줄을 뽑아 양팔 관절(속도 0, PD 목표도 같은 값)과 박스 pose를 놓는다.

    reset_all·reset_object_position 뒤에 와야 한다(이벤트는 cfg 정의 순서로 실행된다). 표는 env frame 기준이다.
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    tables = env.__dict__.setdefault("_start_tables", {})
    table = tables.get(table_path)
    if table is None:
        data = np.load(table_path)
        names = [str(x) for x in data["joint_names"]]
        order = [names.index(nm) for nm in robot.joint_names]
        table = {
            "q": torch.tensor(data["joint_pos"][:, order], dtype=torch.float32, device=env.device),
            "box": torch.tensor(data["box_pose"], dtype=torch.float32, device=env.device),
        }
        tables[table_path] = table
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    k = torch.randint(0, table["q"].shape[0], (n,), device=env.device)

    limits = robot.data.joint_pos_limits.torch[env_ids]
    q = table["q"][k] + (2 * torch.rand_like(table["q"][k]) - 1) * joint_noise
    q = torch.minimum(torch.maximum(q, limits[..., 0]), limits[..., 1])
    robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)
    robot.set_joint_position_target_index(target=q, env_ids=env_ids)
    robot.set_joint_velocity_target_index(target=torch.zeros_like(q), env_ids=env_ids)

    box = table["box"][k]
    pos = box[:, :3] + env.scene.env_origins[env_ids]
    # 표는 기준 크기 박스의 바닥 안착 높이다. 크기 DR(-DR)이면 env마다 반높이 차이만큼 올리거나 내린다.
    pos[:, 2] += _box_halves(env, object_cfg)[1][env_ids] - env.cfg.scene.object.spawn.size[2] / 2.0
    pos[:, :2] += (2 * torch.rand(n, 2, device=env.device) - 1) * box_pos_noise
    yaw = (2 * torch.rand(n, device=env.device) - 1) * box_yaw_noise
    z = torch.zeros(n, 3, device=env.device)
    z[:, 2] = 1.0
    quat = quat_mul(quat_from_angle_axis(yaw, z), box[:, 3:])
    obj.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], -1), env_ids=env_ids)
    obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=env.device), env_ids=env_ids)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp(DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Margin):
    """Light-Margin에서 박스를 410 x 310 x 280 mm, 0.4~0.6 kg으로 바꾸고, 에피소드를 접근 끝 자세(판이 박스 면
    4 cm 바깥)에서 시작한다. 보상·관측·행동은 그대로.

    에피소드는 4 s(80 step). 박스 pose 분포는 표가 정한다(x 0.45~0.52, y ±0.1, yaw ±0.1 + 노이즈).
    외란은 Light처럼 질량에 비례해 줄인다.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_box410(self)
        self.episode_length_s = PREGRASP_EPISODE_S
        self.events.reset_from_pregrasp = EventTerm(
            func=reset_from_pregrasp_table,
            mode="reset",
            params={
                "table_path": PREGRASP_TABLE_PATH,
                "box_pos_noise": PREGRASP_BOX_POS_NOISE,
                "box_yaw_noise": PREGRASP_BOX_YAW_NOISE,
                "joint_noise": PREGRASP_JOINT_NOISE,
            },
        )


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_PLAY(DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp):
    """Sim2Real-100Hz-Light-Pregrasp visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Upright(DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp):
    """Pregrasp + 똑바로 들기 항을 처음부터 켠다.

    P47에서 "들기 습득" 뒤에 켜지던 자세 품질 항(box_upright -3, plate_height_symmetry -8,
    plate_approach_symmetry -20, one_sided_contact -5)을 처음부터 최종 weight로 두고, success의 yaw 계수 ramp도
    바로 시작한다(1000 iter). 접근이 이미 끝난 상태라 들기 발견이 쉬워서, P7의 "작게 시작" 규칙이 막던 문제
    (페널티가 발견을 막음)가 덜할 것으로 본다(미확인). action_rate·joint_vel·판 외 접촉·reaching은 커리큘럼 그대로.
    """

    def __post_init__(self):
        super().__post_init__()
        for name, weight in (
            ("box_upright", -3.0),
            ("plate_height_symmetry", -8.0),
            ("plate_approach_symmetry", -20.0),
            ("one_sided_contact", -5.0),
        ):
            getattr(self.rewards, name).weight = weight
            setattr(self.curriculum, name, None)
        self.curriculum.yawtrack.params.update(trigger=0.0, ramp_steps=24000)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Upright_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Upright
):
    """Sim2Real-100Hz-Light-Pregrasp-Upright visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_RollAlign(DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp):
    """Pregrasp와 같되 시작 자세 표만 판 모서리 정렬 표(PREGRASP_TABLE_ROLLALIGN_PATH)로 바꾼다. from-scratch.

    리셋 직후 판 모서리 회전 -18도 -> -0.1도, 위치·간격·면 평행도·관절 여유는 거의 같다(docs/dual_fr3_lift_sim2real.md).
    보상·커리큘럼·노이즈는 Pregrasp 그대로라 model_2999와 시작 자세만 다르다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_pregrasp.params["table_path"] = PREGRASP_TABLE_ROLLALIGN_PATH


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_RollAlign_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_RollAlign
):
    """Sim2Real-100Hz-Light-Pregrasp-RollAlign visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide(DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp):
    """Pregrasp와 같되 박스를 90도 돌려 실물 배치(긴 변 410 mm가 좌우 = 판이 잡는 방향)로 둔다. from-scratch.

    크기 BOX410_WIDE_SIZE, 시작 자세 표는 이 박스로 같은 방식으로 만든 표(PREGRASP_TABLE_WIDE_PATH, 박스 중심 x 0.40~0.50).
    노이즈·보상·에피소드는 Pregrasp 그대로다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.object.spawn.size = BOX410_WIDE_SIZE
        self.events.reset_from_pregrasp.params["table_path"] = PREGRASP_TABLE_WIDE_PATH


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_PLAY(DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide):
    """Sim2Real-100Hz-Light-Pregrasp-Wide visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign(DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide):
    """Wide와 같되 시작 자세 표만 판 모서리 정렬 표(PREGRASP_TABLE_WIDE_ROLLALIGN_PATH, j7만 다름)로 바꾼다."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_pregrasp.params["table_path"] = PREGRASP_TABLE_WIDE_ROLLALIGN_PATH


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign
):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


def _add_privileged_critic(cfg) -> None:
    """policy 관측 그룹을 복사해 critic 그룹을 만든다(같은 항·순서, 노이즈 없음, 박스 pose는 실제 값).

    policy의 박스 관측 함수를 바꾸기 전에 불러야 한다. 차원이 policy와 같아 기존 checkpoint에서 이어 학습할 수 있다.
    """
    critic = copy.deepcopy(cfg.observations.policy)
    critic.enable_corruption = False
    cfg.observations.critic = critic


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FrozenBox(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign
):
    """Wide-RollAlign과 같되 policy의 박스 위치·자세 관측을 에피소드 시작 값으로 고정한다(비전 없는 실물 조건).

    critic 그룹은 policy와 같은 항·순서에 박스 pose를 매 step 실제 값으로(노이즈 없이) 받는다(asymmetric
    actor-critic). 차원이 같아 Wide-RollAlign checkpoint에서 이어 학습할 수 있다.
    Wide-RollAlign model_2999는 박스 관측을 고정하면 sim 성공 0%였다(goal을 지나쳐 들고 박스가 25° 기운다).
    """

    def __post_init__(self):
        super().__post_init__()
        _add_privileged_critic(self)
        self.observations.policy.object_position.func = object_position_at_reset
        self.observations.policy.object_orientation.func = object_orientation_at_reset


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FrozenBox_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FrozenBox
):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FrozenBox visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_BoxDrop(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign
):
    """Wide-RollAlign과 같되 policy의 박스 위치·자세 관측에 비전 지연·끊김을 넣는다(BOX_OBS_* 상수).

    - 에피소드마다 0~150 ms 지연, step마다 2% 확률로 0.1~1.0 s 끊김(마지막 값 유지), 에피소드 10%는 도중부터 끝까지 끊김.
    - 실물 노드는 비전이 끊기면 마지막 pose를 계속 넣는다.
    - critic 그룹은 실제 박스 pose를 노이즈 없이 받는다(asymmetric actor-critic).
    """

    def __post_init__(self):
        super().__post_init__()
        _add_privileged_critic(self)
        self.observations.policy.object_position.func = object_position_dropout
        self.observations.policy.object_orientation.func = object_orientation_dropout


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_BoxDrop_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_BoxDrop
):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-BoxDrop visualization/evaluation cfg (지연·끊김은 켜 둔다)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


# -FaceGrip: 판 면과 박스 면 사이 각도 페널티. Wide-RollAlign 2999에서 이어 학습한다.
FACEGRIP_TILT_WEIGHT = -10.0  # rad당 초당 페널티. 25도(0.44 rad)면 -4.4/s, 이 run 총 보상은 ~52/s
FACEGRIP_RAMP_STEPS = 4800  # 200 iter(24 step/iter)에 걸쳐 0 -> FACEGRIP_TILT_WEIGHT


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign
):
    """Wide-RollAlign + plate_face_tilt 페널티(판-박스 면 각도, 양 판 접촉 중). 판 V자 기울기를 줄여 면으로 잡게 한다.

    - 페널티는 success EMA가 3을 넘은 뒤 FACEGRIP_RAMP_STEPS에 걸쳐 FACEGRIP_TILT_WEIGHT까지 올린다.
    - Wide-RollAlign 2999에서 이어 학습하는 용도라 yawtrack은 바로 1로 둔다. 이어 학습하면 env가 새로 만들어져 curriculum이
      처음부터 다시 돌고, yawtrack은 기본 3000 iter ramp라 그대로 두면 2999(0.955)와 success 보상이 달라진다.
    - 관측·행동·나머지 보상은 Wide-RollAlign 그대로라 checkpoint를 그대로 읽는다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.plate_face_tilt = RewTerm(func=plate_face_tilt, weight=-1e-4)
        self.curriculum.plate_face_tilt = CurrTerm(
            func=ramp_reward_weight_on_success,
            params={"term_name": "plate_face_tilt", "weight": FACEGRIP_TILT_WEIGHT, "ramp_steps": FACEGRIP_RAMP_STEPS},
        )
        self.curriculum.yawtrack.params.update(trigger=0.0, ramp_steps=1)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip
):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


# -Disturb: 박스 외란을 질량 비례 기본값(±0.625 N, ±0.125 Nm)에서 키운다. Wide-RollAlign 2999에서 이어 학습한다.
DISTURB_BIG_FORCE = 2.5  # N, 각 축 ±. 박스 무게(0.4~0.6 kg, 4~6 N)의 절반쯤
DISTURB_BIG_TORQUE = 0.3  # Nm, 각 축 ±
DISTURB_BIG_RAMP_STEPS = 7200  # 300 iter(24 step/iter)에 걸쳐 기본값 -> 위 값


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_Disturb(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign
):
    """Wide-RollAlign + 박스 외란 확대. 더 단단히 조이고 미끄러짐을 바로잡게 해서 덜 놓치게 한다.

    - 세기만 DISTURB_BIG_RAMP_STEPS에 걸쳐 DISTURB_BIG_FORCE/TORQUE까지 올린다. 방향·주기(0.4~0.8 s)와
      success 기반 on/off(ramp_disturbance_on_success)는 그대로다.
    - 너무 크면 박스가 goal에서 계속 흔들려 success가 무너진다(P23: 4 kg 박스에 ±10 N).
    - Wide-RollAlign 2999에서 이어 학습하는 용도라 yawtrack은 바로 1로 둔다(FaceGrip과 같은 이유).
    """

    def __post_init__(self):
        super().__post_init__()
        p = self.events.disturb_box.params
        self.curriculum.disturb_strength = CurrTerm(
            func=ramp_event_params_by_steps,
            params={
                "term_name": "disturb_box",
                "start": {"force": p["force"], "torque": p["torque"]},
                "end": {"force": DISTURB_BIG_FORCE, "torque": DISTURB_BIG_TORQUE},
                "ramp_steps": DISTURB_BIG_RAMP_STEPS,
            },
        )
        self.curriculum.yawtrack.params.update(trigger=0.0, ramp_steps=1)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_Disturb_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_Disturb
):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-Disturb visualization/evaluation cfg (외란은 처음부터 최대)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        self.events.disturb_box.params.update(force=DISTURB_BIG_FORCE, torque=DISTURB_BIG_TORQUE)
        self.curriculum.disturb_strength = None


# ---------- Regrasp: 놓친 박스를 다시 잡기 (2026-11-04) ----------
# Wide-RollAlign 2999는 박스를 놓치면 다시 잡지 않는다(scripts/tools/box_slip_probe.py: 1.5 s에 마찰 0으로 0.3 s
# 미끄러뜨리면 100% 바닥까지 떨어지고, 박스는 서서 7 cm 밀려 착지, 남은 6.5 s 동안 다시 든 env 0%). 보상은 이미 다시
# 잡기를 막지 않는다(낙하 종료 없음, success는 매 step, reaching은 항상 켜짐). 놓친 경험과 다시 잡을 시간이 없었다.
# - 미끄러짐: 들고 있는 동안 박스와 양 판의 마찰을 잠깐 낮춘다. MuJoCo에서 본, 판 사이로 미끄러져 내려가는 낙하에 가깝다.
# - 놓친 직후 상태에서 시작: 2999로 미끄러뜨려 박스가 바닥에 멈춘 뒤의 (로봇 관절, 박스 pose) 표.
# - 에피소드 8 s.
REGRASP_EPISODE_S = 8.0
BOX_SLIP_PROB = 0.5  # 에피소드당 미끄러짐을 한 번 예약할 확률
BOX_SLIP_T = (1.0, 4.0)  # s, 에피소드 시작부터. 이 시각 뒤 처음 양 판으로 들고 있을 때 일으킨다
BOX_SLIP_DUR = (0.05, 0.4)  # s. 0.1 s 이하면 판이 다시 잡을 수 있고, 0.3 s면 바닥까지 떨어진다(probe 0.35 s)
BOX_SLIP_MU = 0.02  # 미끄러지는 동안 박스·판 정/동 마찰 계수 (PhysX 기본 average 결합)
REGRASP_DROP_STATE_PROB = 0.3  # reset 중 놓친 직후 상태 표에서 시작할 비율
REGRASP_DROP_TABLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(PREGRASP_TABLE_PATH)), "regrasp", "plate_box410wide_dropstates_wra2999.npz"
)


class box_slip(ManagerTermBase):
    """interval(매 policy step, global): 에피소드마다 prob로 미끄러짐 한 번을 예약하고, t_range 시각 뒤 처음 양 판으로
    박스를 들고 있을 때 박스와 양 판(fr3_link7 shape)의 마찰을 mu로 낮췄다가 dur_range 뒤 되돌린다.

    마찰은 PhysX tensor API(root_view.set_material_properties, CPU)로 바꾼다. 원래 값은 첫 호출 때(startup 마찰 DR 뒤) 저장한다.
    reset되는 env가 미끄러지는 중이면 마찰을 되돌리고 다시 예약한다.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, dev = env.num_envs, env.device
        self._robot = env.scene[cfg.params.get("robot_cfg", SceneEntityCfg("robot")).name]
        self._box = env.scene[cfg.params.get("object_cfg", SceneEntityCfg("object")).name]
        self._t_slip = torch.full((n,), float("inf"), device=dev)
        self._dur = torch.zeros(n, device=dev)
        self._end_t = torch.full((n,), float("inf"), device=dev)
        self._active = torch.zeros(n, dtype=torch.bool, device=dev)
        self._done = torch.zeros(n, dtype=torch.bool, device=dev)
        self._box_mat0 = None
        self.count = 0  # 일으킨 미끄러짐 수 (확인용)
        self._schedule(torch.arange(n, device=dev))

    def _schedule(self, ids):
        p = self.cfg.params
        k = len(ids)
        on = torch.rand(k, device=ids.device) < p["prob"]
        lo, hi = p["t_range"]
        t = lo + (hi - lo) * torch.rand(k, device=ids.device)
        self._t_slip[ids] = torch.where(on, t, torch.full_like(t, float("inf")))
        lo, hi = p["dur_range"]
        self._dur[ids] = lo + (hi - lo) * torch.rand(k, device=ids.device)
        self._done[ids] = False

    def _set(self, ids, low: bool):
        if self._box_mat0 is None:
            self._box_mat0 = wp.to_torch(self._box.root_view.get_material_properties()).clone()
            self._rob_mat0 = wp.to_torch(self._robot.root_view.get_material_properties()).clone()
            view = self._robot._physics_sim_view
            cols, start = [], 0
            for path in self._robot.root_view.link_paths[0]:
                k = view.create_rigid_body_view(path).max_shapes
                if path.endswith("fr3_link7"):
                    cols += list(range(start, start + k))
                start += k
            self._plate_cols = cols
            self._box_mat = self._box_mat0.clone()
            self._rob_mat = self._rob_mat0.clone()
        idx = ids.cpu()
        mu = self.cfg.params["mu"]
        if low:
            self._box_mat[idx, :, 0:2] = mu
            self._rob_mat[idx[:, None], torch.tensor(self._plate_cols)[None, :], 0:2] = mu
        else:
            self._box_mat[idx] = self._box_mat0[idx]
            self._rob_mat[idx] = self._rob_mat0[idx]
        wi = wp.from_torch(idx.to(torch.int32), dtype=wp.int32)
        self._box.root_view.set_material_properties(wp.from_torch(self._box_mat.contiguous(), dtype=wp.float32), wi)
        self._robot.root_view.set_material_properties(wp.from_torch(self._rob_mat.contiguous(), dtype=wp.float32), wi)

    def reset(self, env_ids=None):
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        back = env_ids[self._active[env_ids]]
        if len(back):
            self._set(back, low=False)
            self._active[back] = False
        self._end_t[env_ids] = float("inf")
        self._schedule(env_ids)
        return {}

    def __call__(
        self,
        env,
        env_ids,
        prob: float,
        t_range: tuple,
        dur_range: tuple,
        mu: float,
        lift_delta: float = 0.05,
        finger1_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_plate_object_contact"),
        finger2_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_plate_object_contact"),
        threshold: float = 1.0,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ):
        t = env.episode_length_buf.float() * env.step_dt
        stop = self._active & (t >= self._end_t)
        if stop.any():
            ids = stop.nonzero().squeeze(-1)
            self._set(ids, low=False)
            self._active[ids] = False
        due = ~self._done & (t >= self._t_slip)
        if due.any():
            lifted = _lifted_perenv(env, lift_delta, object_cfg).bool()
            grasped = mdp.object_grasped(env, finger1_sensor_cfg, finger2_sensor_cfg, threshold).bool()
            go = due & lifted & grasped
            if go.any():
                ids = go.nonzero().squeeze(-1)
                self._set(ids, low=True)
                self._active[ids] = True
                self._done[ids] = True
                self._end_t[ids] = t[ids] + self._dur[ids]
                self.count += len(ids)


def reset_from_drop_states(
    env,
    env_ids,
    prob: float,
    table_path: str,
    box_pos_noise: float,
    box_yaw_noise: float,
    joint_noise: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """reset: env_ids 중 prob 비율을 놓친 직후 상태 표(로봇 관절 + 바닥의 박스 pose, env frame)에서 시작시킨다.

    표 형식은 Pregrasp 표와 같아 reset_from_pregrasp_table로 쓴다. reset_from_pregrasp 뒤에 와야 한다(덮어쓴다).
    goal(x, y)은 command reset이 이벤트 뒤에 박스 위치로 다시 잡는다.
    """
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)
    sel = env_ids[torch.rand(len(env_ids), device=env.device) < prob]
    if len(sel):
        reset_from_pregrasp_table(env, sel, table_path, box_pos_noise, box_yaw_noise, joint_noise, robot_cfg, object_cfg)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_Regrasp(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign
):
    """Wide-RollAlign + 놓친 박스 다시 잡기: 미끄러짐 이벤트(box_slip), 놓친 직후 상태에서 시작(30%), 에피소드 8 s.

    - 보상·관측·행동·외란은 Wide-RollAlign 그대로다(외란 확대는 -Disturb에서 따로 본다).
    - goal 재샘플을 에피소드보다 길게 둔다(기본 5 s라 8 s 에피소드면 중간에 goal이 바뀐다).
    - Wide-RollAlign 2999에서 이어 학습하는 용도라 yawtrack은 바로 1로 둔다(FaceGrip과 같은 이유).
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = REGRASP_EPISODE_S
        self.commands.object_pose.resampling_time_range = (REGRASP_EPISODE_S + 1.0, REGRASP_EPISODE_S + 1.0)
        self.events.reset_from_drop_states = EventTerm(
            func=reset_from_drop_states,
            mode="reset",
            params={
                "prob": REGRASP_DROP_STATE_PROB,
                "table_path": REGRASP_DROP_TABLE_PATH,
                "box_pos_noise": PREGRASP_BOX_POS_NOISE,
                "box_yaw_noise": PREGRASP_BOX_YAW_NOISE,
                "joint_noise": PREGRASP_JOINT_NOISE,
            },
        )
        self.events.box_slip = EventTerm(
            func=box_slip,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            is_global_time=True,
            params={"prob": BOX_SLIP_PROB, "t_range": BOX_SLIP_T, "dur_range": BOX_SLIP_DUR, "mu": BOX_SLIP_MU},
        )
        self.curriculum.yawtrack.params.update(trigger=0.0, ramp_steps=1)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_Regrasp_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_Regrasp
):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-Regrasp visualization/evaluation cfg (미끄러짐·놓친 상태 시작은 켜 둔다)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


# ---------- DR 확장: 박스 크기, 판 마찰, 관절 상태·명령 지연 (2026-11-04) ----------
BOX_SIZE_DR_M = 0.01  # m, 축마다 ± (실물 치수 오차)
PLATE_FRICTION_STATIC = (0.6, 1.2)  # 고무 패드. PhysX average 결합이라 박스(0.4~1.2)와 평균한 값이 접촉 마찰이다
PLATE_FRICTION_DYNAMIC = (0.5, 1.0)
JOINT_STATE_DELAY_S = (0.0, 0.02)  # joint_states 나이 (추정치, 실측 전)
COMMAND_DELAY_STEPS = (0, 3)  # 물리 step(10 ms) 단위 명령 전달 지연, 0~30 ms (추정치, 실측 전)


def joint_pos_rel_delayed(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    action_name: str = "arm_action",
) -> torch.Tensor:
    """joint_pos_rel에 action term(DelayedReachRelativeJointPositionAction)의 env별 관절 상태 지연을 1차 근사로 넣은 값.

    q(t - d) = q(t) - qd(t) * d. 정책이 보는 q와 목표 기준 q가 같은 지연을 갖는다(실물 노드와 같음).
    """
    asset = env.scene[asset_cfg.name]
    term = env.action_manager.get_term(action_name)
    q = asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    qd = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return q - qd * term.obs_delay.unsqueeze(-1)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_DR(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign
):
    """Wide-RollAlign + 실물 차이 DR: 박스 크기 ±1 cm, 판(고무 패드) 마찰, 관절 상태 지연 0~20 ms, 명령 지연 0~30 ms.

    - 박스 크기: 축마다 ±BOX_SIZE_DR_M(배율). 시작 자세 표는 기준 크기 그대로라 판-면 간격이 4 cm ± 크기 차/2가 된다
      (MoveIt으로 표 자세에 보낸 뒤 실물 박스가 조금 다른 상황). 리셋 높이는 env별 반높이로 맞춘다.
    - 판 마찰: fr3_link7 shape(판)에 startup 마찰 DR.
    - 지연: action term이 env마다 관절 상태 나이와 명령 지연을 뽑는다(DelayedReachRelativeJointPositionAction).
      joint_pos 관측과 목표 기준 q가 같은 지연을 갖는다. 비전 지연은 -BoxDrop에서 따로 본다.
    - 그 밖의 보상·관측·외란은 Wide-RollAlign 그대로. 2999에서 이어 학습하는 용도라 yawtrack은 바로 1.
    """

    def __post_init__(self):
        super().__post_init__()
        sx, sy, sz = self.scene.object.spawn.size
        self.events.randomize_box_size.params["scale_range"] = {
            "x": (1.0 - BOX_SIZE_DR_M / sx, 1.0 + BOX_SIZE_DR_M / sx),
            "y": (1.0 - BOX_SIZE_DR_M / sy, 1.0 + BOX_SIZE_DR_M / sy),
            "z": (1.0 - BOX_SIZE_DR_M / sz, 1.0 + BOX_SIZE_DR_M / sz),
        }
        self.events.randomize_plate_friction = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*_fr3_link7"]),
                "static_friction_range": PLATE_FRICTION_STATIC,
                "dynamic_friction_range": PLATE_FRICTION_DYNAMIC,
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
                "make_consistent": True,
            },
        )
        arm = self.actions.arm_action
        self.actions.arm_action = DelayedReachRelativeJointPositionActionCfg(
            asset_name=arm.asset_name,
            joint_names=arm.joint_names,
            preserve_order=arm.preserve_order,
            scale=arm.scale,
            clip=arm.clip,
            reference_mode="policy_step",
            target_limits=arm.target_limits,
            obs_delay_range_s=JOINT_STATE_DELAY_S,
            cmd_delay_steps_range=COMMAND_DELAY_STEPS,
        )
        self.observations.policy.joint_pos.func = joint_pos_rel_delayed
        self.curriculum.yawtrack.params.update(trigger=0.0, ramp_steps=1)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_DR_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_DR
):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-DR visualization/evaluation cfg (DR은 켜 둔다)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


# -MinSqueeze: 최소 조임 페널티. FaceGrip 4699(실로봇에서 조임이 빠듯했던 정책)에서 이어 학습한다.
# 4699는 들고 있을 때 판 접촉력 ~11 N이고, 이때 관절 1·3의 PD 오차가 0.064~0.072 rad(행동 한계 0.1의 64~72%,
# 5~5.6 Nm / PD 상한 8 Nm)라 지금 자세로는 ~15 N이 sim 상한이다. 운반 중 보정 여유를 남겨 14 N으로 둔다.
MINSQUEEZE_FORCE_N = 14.0
MINSQUEEZE_WEIGHT = -0.5  # 모자란 N(양 판 합)당 초당 페널티. 4699 기준 ~6 N 부족 -> -3/s, 총 보상은 ~52/s
MINSQUEEZE_RAMP_STEPS = 2400  # 100 iter(24 step/iter)에 걸쳐 0 -> MINSQUEEZE_WEIGHT
MINSQUEEZE_LIFT_HEIGHT = 0.02  # 박스 중심이 안착 높이보다 이만큼 뜨면 들린 것으로 본다


def _plate_force_norms(
    env,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_plate_object_contact"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_plate_object_contact"),
) -> torch.Tensor:
    """판별 박스 접촉 합력 크기 [N], (num_envs, 2) = (왼판, 오른판). excessive_squeeze_penalty와 같은 값."""
    forces = []
    for cfg in (left_sensor_cfg, right_sensor_cfg):
        sensor = env.scene.sensors[cfg.name]
        forces.append(torch.norm(sensor.data.force_matrix_w.torch, dim=-1).amax(dim=(-1, -2)))
    return torch.nan_to_num(torch.stack(forces, dim=-1), nan=0.0, posinf=0.0)


def min_squeeze_penalty(
    env,
    force_min: float,
    lift_height: float = MINSQUEEZE_LIFT_HEIGHT,
    contact_threshold: float = 1.0,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_plate_object_contact"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_plate_object_contact"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """최소 조임 페널티: 판별 접촉력이 force_min(N)에 모자란 만큼의 합 [N]. excessive_squeeze_penalty의 반대쪽.

    양 판이 닿아 있거나(들기 전 미리 조이기) 박스가 들려 있을 때만 준다. 들린 채 한 판이 떨어지면 그 판은
    force_min 전체가 부족분이 된다. 힘은 excessive_squeeze_penalty와 같은 합력 norm(들고 있을 때 법선과 거의 같음).
    """
    f = _plate_force_norms(env, left_sensor_cfg, right_sensor_cfg)
    gate = (f > contact_threshold).all(dim=-1).float().maximum(_lifted_perenv(env, lift_height, object_cfg))
    return (force_min - f).clamp(min=0.0, max=force_min).sum(dim=-1) * gate


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_MinSqueeze(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip
):
    """FaceGrip + min_squeeze 페널티(판별 접촉력 MINSQUEEZE_FORCE_N 미만이면 부족분). 실물 조임 여유를 늘린다.

    - 페널티는 success EMA가 3을 넘은 뒤 MINSQUEEZE_RAMP_STEPS에 걸쳐 MINSQUEEZE_WEIGHT까지 올린다.
    - FaceGrip 4699에서 이어 학습하는 용도라 plate_face_tilt는 바로 FACEGRIP_TILT_WEIGHT로 둔다.
    - excessive_squeeze(100 N 초과분)는 그대로. 관측·행동은 FaceGrip 그대로라 checkpoint를 그대로 읽는다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.min_squeeze = RewTerm(
            func=min_squeeze_penalty, params={"force_min": MINSQUEEZE_FORCE_N}, weight=-1e-4
        )
        self.curriculum.min_squeeze = CurrTerm(
            func=ramp_reward_weight_on_success,
            params={"term_name": "min_squeeze", "weight": MINSQUEEZE_WEIGHT, "ramp_steps": MINSQUEEZE_RAMP_STEPS},
        )
        self.curriculum.plate_face_tilt.params.update(ramp_steps=1)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_MinSqueeze_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_MinSqueeze
):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-MinSqueeze visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


# -FaceGrip-MinSqueeze-Scratch: FaceGrip-MinSqueeze와 같은 보상을 처음부터 학습한다(이어 학습 없음).
# 4699 계보(Wide-RollAlign 2999 -> FaceGrip 3700 -> 4699)는 2999와 3700이 박스 회전 제한(30 deg/s) 물리에서 배웠다.
# 고친 물리에서 판 면 파지·최소 조임을 처음부터 넣어, 그 물리에서 생긴 V자 받침 습관 없이 잡는 법을 찾게 한다.
MINSQUEEZE_SCRATCH_RAMP_STEPS = 7200  # 300 iter. 처음부터면 성공 직후라 FaceGrip ramp(200 iter)보다 천천히 올린다


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_MinSqueeze_Scratch(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign
):
    """Wide-RollAlign + plate_face_tilt + min_squeeze, 처음부터 학습. 보상은 FaceGrip-MinSqueeze와 같다.

    - FaceGrip·MinSqueeze의 이어 학습용 설정(yawtrack·plate_face_tilt 즉시 최대)은 쓰지 않는다. yawtrack은
      Wide-RollAlign 기본 ramp(success EMA 1.5 이후 3000 iter), 두 페널티는 success EMA 3 이후에 ramp로 올린다.
    - plate_face_tilt: FACEGRIP_RAMP_STEPS에 걸쳐 FACEGRIP_TILT_WEIGHT까지.
    - min_squeeze: MINSQUEEZE_SCRATCH_RAMP_STEPS에 걸쳐 MINSQUEEZE_WEIGHT까지(판별 MINSQUEEZE_FORCE_N).
    - 관측·행동·나머지 보상과 시작 자세 표는 Wide-RollAlign 그대로라 실물 노드·시작 자세 후보를 그대로 쓴다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.plate_face_tilt = RewTerm(func=plate_face_tilt, weight=-1e-4)
        self.curriculum.plate_face_tilt = CurrTerm(
            func=ramp_reward_weight_on_success,
            params={"term_name": "plate_face_tilt", "weight": FACEGRIP_TILT_WEIGHT, "ramp_steps": FACEGRIP_RAMP_STEPS},
        )
        self.rewards.min_squeeze = RewTerm(
            func=min_squeeze_penalty, params={"force_min": MINSQUEEZE_FORCE_N}, weight=-1e-4
        )
        self.curriculum.min_squeeze = CurrTerm(
            func=ramp_reward_weight_on_success,
            params={"term_name": "min_squeeze", "weight": MINSQUEEZE_WEIGHT, "ramp_steps": MINSQUEEZE_SCRATCH_RAMP_STEPS},
        )


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_MinSqueeze_Scratch_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_MinSqueeze_Scratch
):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-MinSqueeze-Scratch visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


# ---------- Box410-Wide-J1Mirror: J1Mirror를 실물 박스 배치로 (2026-11-04) ----------
# Box410-J1Mirror model_4999는 410을 앞뒤로 둔 박스로 학습했다(3/3 들기, j1 L/R +1.06/-1.10). 박스만 Pregrasp-Wide와 같이
# 90도 돌린다(BOX410_WIDE_SIZE, 긴 변 410 mm가 좌우 = 판이 잡는 방향). 바닥에서 시작해 접근부터 전부 RL.
# 박스 중심 x는 Wide 표와 같은 0.40~0.50 (0.50 이상은 j2 한계로 잡기 어렵고 실물에서도 안 쓴다).
BOX410_WIDE_X_RANGE = (-0.10, 0.0)  # 스폰 x 오프셋 [m]


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_J1Mirror
):
    """Box410-J1Mirror와 같되 박스를 BOX410_WIDE_SIZE로, 스폰 x를 BOX410_WIDE_X_RANGE로. from-scratch."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.object.spawn.size = BOX410_WIDE_SIZE
        self.events.reset_object_position.params["pose_range"]["x"] = BOX410_WIDE_X_RANGE


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror
):
    """Sim2Real-100Hz-Light-Box410-Wide-J1Mirror visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False



# ---------- FaceGrip-BoxDrop-Scratch: 판 면 파지 + 비전 지연·끊김, 처음부터 (2026-11-04) ----------
# FaceGrip 4699(고친 물리)는 실로봇에서 들기가 확인됐다. 4699 계보는 앞 단계를 박스 회전 제한 물리에서 배웠으므로
# 이어 학습 대신 고친 물리에서 처음부터 학습한다(FaceGrip-MinSqueeze-Scratch와 같은 이유).
# Wide-RollAlign-BoxDrop(박스 관측 지연·끊김 BOX_OBS_*, 실제 pose critic)에 FaceGrip의 plate_face_tilt를 더한다.
@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_BoxDrop_Scratch(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_BoxDrop
):
    """Wide-RollAlign-BoxDrop + plate_face_tilt, 처음부터 학습. 보상은 FaceGrip과 같다.

    - yawtrack은 Wide-RollAlign 기본 ramp(success EMA 1.5 이후 3000 iter), plate_face_tilt는 success EMA 3 이후
      FACEGRIP_RAMP_STEPS에 걸쳐 FACEGRIP_TILT_WEIGHT까지(FaceGrip-MinSqueeze-Scratch와 같은 방식).
    - 관측 지연·끊김과 critic 그룹은 BoxDrop 그대로, 시작 자세 표는 Wide-RollAlign 그대로다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.plate_face_tilt = RewTerm(func=plate_face_tilt, weight=-1e-4)
        self.curriculum.plate_face_tilt = CurrTerm(
            func=ramp_reward_weight_on_success,
            params={"term_name": "plate_face_tilt", "weight": FACEGRIP_TILT_WEIGHT, "ramp_steps": FACEGRIP_RAMP_STEPS},
        )


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_BoxDrop_Scratch_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_BoxDrop_Scratch
):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-BoxDrop-Scratch visualization/evaluation cfg (지연·끊김은 켜 둔다)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


# ---------- -FaceGrip-SqueezeLift: 조인 다음 들기 + 판-박스 미끄럼 벌점 (2026-11-04) ----------
# MinSqueeze 5698(고친 물리)도 판이 닿자마자(0.2 s) 다 조이기 전에 들기 시작한다. 판 힘이 4699는 1 s 6.7 N -> 2 s 11 N,
# 5698은 10 N -> 16 N이고, 그동안 판이 박스 무게에 V자로 돌아간다(4699 판 pitch 0.3 s 3.7도 -> 0.5 s 8도 -> 1.5 s 0도).
# 접촉 전에도 판이 4~6도 기울어(plate_face_tilt는 양 판 접촉 뒤에만 걸림) 모서리부터 닿는다.
# - 들기 보상(lifting_object, lift_progress, object_goal_tracking[_fine_grained], success)에 조임 factor를 곱한다:
#   min(판 힘 L, R) / squeeze_force (0~1). 덜 조인 채 들면 들기 보상을 덜 받는다. lift_monitor(커리큘럼 판정)는 그대로.
# - plate_face_tilt_approach: 양 판이 다 닿기 전, 담당 면에서 APPROACH_TILT_GAP 안에 있는 판의 기울기 [rad].
# - plate_box_slip: 닿아 있는 판마다 판 중심점의 박스 대비 접선 속도 + 상대 각속도 x SLIP_ANG_LEVER [m/s].
SQUEEZELIFT_FORCE_N = MINSQUEEZE_FORCE_N  # 판별 이 힘 이상이면 들기 보상을 다 받는다
SQUEEZELIFT_RAMP_STEPS = 2400  # 100 iter: 조임 factor 0 -> 1, 두 벌점 0 -> weight
APPROACH_TILT_GAP = 0.06  # m, 판 중심이 담당 면 바깥 이 거리 안이면 본다 (시작 자세 판-면 4 cm)
APPROACH_TILT_WEIGHT = FACEGRIP_TILT_WEIGHT  # 접촉 뒤 plate_face_tilt와 같은 weight라 접촉 전후로 벌점이 이어진다
SLIP_ANG_LEVER = 0.1  # m, 상대 각속도를 판 가장자리 속도로 바꾸는 길이 (판 140 mm 정사각형 반대각선 ~0.1 m)
SLIP_WEIGHT = -10.0  # (m/s)당 초당. 0.1 m/s 미끄러지면 -1/s (MinSqueeze 5698 success 보상 ~11/s)


def ramp_env_scale_on_success(
    env, env_ids, attr: str, success_term: str = "success", trigger: float = 3.0, ramp_steps: int = SQUEEZELIFT_RAMP_STEPS
):
    """success EMA가 처음 trigger에 닿은 뒤 ramp_steps env-step에 걸쳐 env.<attr>를 0에서 1로 올린다(비감소).

    ramp_yawtrack_on_success와 같은 방식(발동은 success, 속도는 step 수)을 속성 이름만 받아 쓴다.
    반환값은 Curriculum/<curriculum 이름>에 기록된다.
    """
    scale = getattr(env, attr, 0.0)
    if scale >= 1.0:
        return 1.0
    starts = env.__dict__.setdefault("_scale_ramp_start", {})
    cur = env.common_step_counter
    if attr not in starts:
        if not hasattr(env, "_lift_curriculum_ema"):
            env._lift_curriculum_ema = {}
        success_level = env.reward_manager._episode_sums[success_term][env_ids].mean().item()
        alpha = min(0.5, 3.0 * len(env_ids) / env.num_envs)
        key = f"_scale_{attr}_succ"
        ema = env._lift_curriculum_ema.get(key, 0.0)
        ema = ema + alpha * (success_level - ema)
        env._lift_curriculum_ema[key] = ema
        if ema < trigger:
            return 0.0
        starts[attr] = cur
    value = max(scale, min(1.0, (cur - starts[attr]) / max(ramp_steps, 1)))
    setattr(env, attr, value)
    return value


def _squeeze_lift_factor(env, force_ref: float, left_sensor_cfg: SceneEntityCfg, right_sensor_cfg: SceneEntityCfg):
    """1 - s + s * clamp(min(판 힘 L, R) / force_ref, 0, 1). s = env._squeezelift_scale (커리큘럼, 0 -> 1)."""
    s = getattr(env, "_squeezelift_scale", 0.0)
    f = _plate_force_norms(env, left_sensor_cfg, right_sensor_cfg).amin(dim=-1)
    return 1.0 - s + s * (f / force_ref).clamp(0.0, 1.0)


def object_is_lifted_perenv_and_squeezed_firm(
    env,
    lift_delta: float,
    finger1_sensor_cfg: SceneEntityCfg,
    finger2_sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    tilt_gate: float = LEVEL_TILT_GATE,
    force_ref: float = PLATE_ONLY_FORCE_REF,
    squeeze_force: float = SQUEEZELIFT_FORCE_N,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """판 전용 들기 x 조임 factor."""
    lifted = object_is_lifted_perenv_and_squeezed_plateonly(
        env, lift_delta, finger1_sensor_cfg, finger2_sensor_cfg, threshold, tilt_gate, force_ref, object_cfg
    )
    return lifted * _squeeze_lift_factor(env, squeeze_force, finger1_sensor_cfg, finger2_sensor_cfg)


def lift_progress_firm(
    env,
    lift_delta: float,
    finger1_sensor_cfg: SceneEntityCfg,
    finger2_sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    tilt_gate: float = LEVEL_TILT_GATE,
    force_ref: float = PLATE_ONLY_FORCE_REF,
    squeeze_force: float = SQUEEZELIFT_FORCE_N,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """판 전용 들기 진행도 x 조임 factor."""
    progress = lift_progress_plateonly(
        env, lift_delta, finger1_sensor_cfg, finger2_sensor_cfg, threshold, tilt_gate, force_ref, object_cfg
    )
    return progress * _squeeze_lift_factor(env, squeeze_force, finger1_sensor_cfg, finger2_sensor_cfg)


def object_goal_distance_perenv_squeezed_firm(
    env,
    std: float,
    lift_delta: float,
    command_name: str,
    finger1_sensor_cfg: SceneEntityCfg,
    finger2_sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    tilt_gate: float = LEVEL_TILT_GATE,
    force_ref: float = PLATE_ONLY_FORCE_REF,
    squeeze_force: float = SQUEEZELIFT_FORCE_N,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """판 전용 goal 추종 x 조임 factor."""
    tracking = object_goal_distance_perenv_squeezed_plateonly(
        env, std, lift_delta, command_name, finger1_sensor_cfg, finger2_sensor_cfg, threshold, tilt_gate, force_ref,
        robot_cfg, object_cfg,
    )
    return tracking * _squeeze_lift_factor(env, squeeze_force, finger1_sensor_cfg, finger2_sensor_cfg)


class success_bonus_firm(success_bonus_plateonly):
    """success_bonus_plateonly x 조임 factor (지급만 깎고 hold 카운터는 그대로)."""

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
        force_ref: float = PLATE_ONLY_FORCE_REF,
        squeeze_force: float = SQUEEZELIFT_FORCE_N,
    ) -> torch.Tensor:
        reward = super().__call__(
            env, dist_threshold, tilt_threshold, hold_steps, finger1_sensor_cfg, finger2_sensor_cfg, threshold,
            command_name, robot_cfg, object_cfg, sustained, yaw_alpha, yaw_std, force_ref,
        )
        return reward * _squeeze_lift_factor(env, squeeze_force, finger1_sensor_cfg, finger2_sensor_cfg)


def plate_face_tilt_approach(
    env,
    gap_max: float = APPROACH_TILT_GAP,
    finger1_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_plate_object_contact"),
    finger2_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_plate_object_contact"),
    threshold: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """접근 중 판 기울기 [rad]: 양 판이 다 닿기 전(G = 0), 판 중심이 담당 면 바깥 gap_max 안(안쪽 포함)인 판의 각도, 양 판 평균.

    각도는 plate_face_tilt(양 판 접촉 중)와 같아서 접촉 전후로 벌점이 이어진다. 시작 자세는 판-면 4 cm라 처음부터 걸린다.
    """
    ee_frame = env.scene[ee_frame_cfg.name]
    li, ri, y_axis_w, face_l, face_r, _, _ = _face_dists(env, object_cfg, ee_frame_cfg)
    p = ee_frame.data.target_pos_w.torch
    near_l = (((p[:, li] - face_l) * y_axis_w).sum(dim=-1) < gap_max).float()  # 왼판은 +y면 바깥
    near_r = (((face_r - p[:, ri]) * y_axis_w).sum(dim=-1) < gap_max).float()
    z_axis = torch.zeros_like(face_l)
    z_axis[:, 2] = 1.0
    n_l = quat_apply(ee_frame.data.target_quat_w.torch[:, li], z_axis)
    n_r = quat_apply(ee_frame.data.target_quat_w.torch[:, ri], z_axis)
    ang_l = torch.acos((-(n_l * y_axis_w).sum(dim=-1)).clamp(-1.0, 1.0))
    ang_r = torch.acos(((n_r * y_axis_w).sum(dim=-1)).clamp(-1.0, 1.0))
    gate = 1.0 - mdp.object_grasped(env, finger1_sensor_cfg, finger2_sensor_cfg, threshold)
    return 0.5 * (ang_l * near_l + ang_r * near_r) * gate


def plate_box_slip(
    env,
    ang_lever: float = SLIP_ANG_LEVER,
    threshold: float = 1.0,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_plate_object_contact"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_plate_object_contact"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """판-박스 미끄럼 [m/s]: threshold N 넘게 닿은 판마다 |판 중심점의 박스 대비 접선 속도| + ang_lever x |상대 각속도|, 양 판 합.

    판 중심점(TCP) p에서 판(link7) 위 점의 속도 v7 + w7 x (p - x7)와 박스 위 같은 점의 속도 vb + wb x (p - xb)를 뺀 뒤
    박스 면 법선 성분(조이고 푸는 움직임)을 뺀다. 상대 각속도는 판이 V자로 돌아가는 것과 박스가 판 사이에서 도는 것을 같이 본다.
    """
    robot = env.scene["robot"]
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    if not hasattr(env, "_link7_ids"):
        env._link7_ids = {s: robot.find_bodies(f"{s}_fr3_link7")[0][0] for s in ("left", "right")}
    li, ri = _ee_lr_indices(ee_frame)
    xb = obj.data.root_link_pos_w.torch
    vb = obj.data.root_link_lin_vel_w.torch
    wb = obj.data.root_link_ang_vel_w.torch
    y_axis = torch.zeros_like(xb)
    y_axis[:, 1] = 1.0
    n = quat_apply(obj.data.root_link_quat_w.torch, y_axis)
    f = _plate_force_norms(env, left_sensor_cfg, right_sensor_cfg)
    out = torch.zeros(env.num_envs, device=env.device)
    for k, (side, idx) in enumerate((("left", li), ("right", ri))):
        bid = env._link7_ids[side]
        p = ee_frame.data.target_pos_w.torch[:, idx]
        x7 = robot.data.body_link_pos_w.torch[:, bid]
        v7 = robot.data.body_link_lin_vel_w.torch[:, bid]
        w7 = robot.data.body_link_ang_vel_w.torch[:, bid]
        v_rel = v7 + torch.cross(w7, p - x7, dim=-1) - vb - torch.cross(wb, p - xb, dim=-1)
        v_t = v_rel - (v_rel * n).sum(dim=-1, keepdim=True) * n
        slip = v_t.norm(dim=-1) + ang_lever * (w7 - wb).norm(dim=-1)
        out += slip * (f[:, k] > threshold).float()
    return torch.nan_to_num(out, nan=0.0, posinf=0.0)


def _apply_squeeze_lift(cfg, squeeze_force: float, ramp_steps: int) -> None:
    """들기 보상에 조임 factor(판별 squeeze_force), 접근 기울기·미끄럼 벌점을 붙인다. 셋 다 success EMA 3 뒤 ramp_steps에 걸쳐 켠다."""
    firm = {
        object_is_lifted_perenv_and_squeezed_plateonly: object_is_lifted_perenv_and_squeezed_firm,
        lift_progress_plateonly: lift_progress_firm,
        object_goal_distance_perenv_squeezed_plateonly: object_goal_distance_perenv_squeezed_firm,
    }
    for name in ("lifting_object", "lift_progress", "object_goal_tracking", "object_goal_tracking_fine_grained"):
        term = getattr(cfg.rewards, name)
        term.func = firm[term.func]
        term.params["squeeze_force"] = squeeze_force
    cfg.rewards.success.func = success_bonus_firm
    cfg.rewards.success.params["squeeze_force"] = squeeze_force
    cfg.curriculum.squeeze_lift = CurrTerm(
        func=ramp_env_scale_on_success, params={"attr": "_squeezelift_scale", "ramp_steps": ramp_steps}
    )
    cfg.rewards.plate_face_tilt_approach = RewTerm(func=plate_face_tilt_approach, weight=-1e-4)
    cfg.curriculum.plate_face_tilt_approach = CurrTerm(
        func=ramp_reward_weight_on_success,
        params={"term_name": "plate_face_tilt_approach", "weight": APPROACH_TILT_WEIGHT, "ramp_steps": ramp_steps},
    )
    cfg.rewards.plate_box_slip = RewTerm(func=plate_box_slip, weight=-1e-4)
    cfg.curriculum.plate_box_slip = CurrTerm(
        func=ramp_reward_weight_on_success,
        params={"term_name": "plate_box_slip", "weight": SLIP_WEIGHT, "ramp_steps": ramp_steps},
    )


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_SqueezeLift(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_MinSqueeze
):
    """FaceGrip-MinSqueeze + 조인 다음 들기(_apply_squeeze_lift, 판별 SQUEEZELIFT_FORCE_N). MinSqueeze 5698에서 이어 학습한다.

    - 조임 factor·접근 기울기·미끄럼 벌점은 success EMA 3 뒤 SQUEEZELIFT_RAMP_STEPS(100 iter)에 걸쳐 켠다.
    - 관측·행동(action scale 0.1)·시작 자세 표는 그대로라 checkpoint를 그대로 읽고 실물 노드도 그대로 쓴다.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_squeeze_lift(self, SQUEEZELIFT_FORCE_N, SQUEEZELIFT_RAMP_STEPS)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_SqueezeLift_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_SqueezeLift
):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-SqueezeLift visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


# ---------- -FaceGrip-SqueezeLift-LowMuHeavy-Act015-Scratch: 미끄럽고 무거운 박스 + 행동 범위 0.15 (2026-11-04) ----------
# 판-박스 마찰을 0.25~0.5로 낮추고 박스를 0.4~0.8 kg으로 무겁게 하면 버티는 데 판당 m g / (2 mu)가 최대 15.7 N
# (0.8 kg, 0.25) 든다. 행동 범위 0.1 rad(PD 상한 80 x 0.1 = 8 Nm)로는 MinSqueeze 5698이 16 N에서 관절 1·3을 77~90% 쓰므로
# 모자라다 -> action scale·clip을 0.15로 올린다(PD 상한 12 Nm). 실물 노드는 manifest의 scale/clip을 쓰고, 한 step 목표
# 변화 0.15 rad는 노드 상한(max_policy_target_delta_rad 0.20) 안이라 노드 수정은 없다. 행동 크기가 바뀌므로 처음부터 학습한다.
HARDGRIP_ACTION_SCALE = 0.15
HARDGRIP_FORCE_N = 20.0  # min_squeeze 목표와 조임 factor 기준 (최악 15.7 N + 여유)
HARDGRIP_MASS_RANGE = (0.4, 0.8)  # kg (실측 501~561 g)
HARDGRIP_FRICTION_STATIC = (0.25, 0.5)  # 박스·판 둘 다. PhysX average 결합이라 접촉 마찰도 0.25~0.5
HARDGRIP_FRICTION_DYNAMIC = (0.2, 0.45)
HARDGRIP_RAMP_STEPS = 7200  # 300 iter. 처음부터라 min_squeeze(MINSQUEEZE_SCRATCH_RAMP_STEPS)와 같이 천천히


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_SqueezeLift_LowMuHeavy_Act015_Scratch(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_MinSqueeze_Scratch
):
    """FaceGrip-MinSqueeze-Scratch + 조인 다음 들기 + 낮은 마찰·무거운 박스 + action scale 0.15. 처음부터 학습.

    - 행동: scale·clip HARDGRIP_ACTION_SCALE (q_target = q + clip(0.15 a, ±0.15)). export manifest에 그대로 실린다.
    - 박스 질량 HARDGRIP_MASS_RANGE, 박스·판(fr3_link7) 마찰 HARDGRIP_FRICTION_* (startup DR).
    - min_squeeze 목표와 조임 factor 기준 HARDGRIP_FORCE_N. 조임 factor·두 벌점은 success EMA 3 뒤 HARDGRIP_RAMP_STEPS에 걸쳐.
    - 관측·시작 자세 표는 Wide-RollAlign 그대로.
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action.scale = HARDGRIP_ACTION_SCALE
        self.actions.arm_action.clip = {".*": (-HARDGRIP_ACTION_SCALE, HARDGRIP_ACTION_SCALE)}
        self.events.randomize_box_density.params["mass_distribution_params"] = HARDGRIP_MASS_RANGE
        self.events.randomize_box_friction.params.update(
            static_friction_range=HARDGRIP_FRICTION_STATIC, dynamic_friction_range=HARDGRIP_FRICTION_DYNAMIC
        )
        self.events.randomize_plate_friction = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*_fr3_link7"]),
                "static_friction_range": HARDGRIP_FRICTION_STATIC,
                "dynamic_friction_range": HARDGRIP_FRICTION_DYNAMIC,
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
                "make_consistent": True,
            },
        )
        self.rewards.min_squeeze.params["force_min"] = HARDGRIP_FORCE_N
        _apply_squeeze_lift(self, HARDGRIP_FORCE_N, HARDGRIP_RAMP_STEPS)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_SqueezeLift_LowMuHeavy_Act015_Scratch_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_SqueezeLift_LowMuHeavy_Act015_Scratch
):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-SqueezeLift-LowMuHeavy-Act015-Scratch visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


# ---------- -FaceGrip-BoxDrop-FlatOnly / -FlatSqueeze: 더 기운 판 기준 평행 + 면 앞뒤 중앙 (+ 판 법선 방향 조임) (2026-11-05) ----------
# FaceGrip-BoxDrop-Scratch 2999는 판 V자가 없지만(0.5 s pitch +1도) 들어 올릴 때(0.25~1 s) 왼판만 수직축 둘레로 ~11도 돌아간다
# (오른판 0도). 박스는 1도 안쪽으로만 돌고, 왼팔 j4·j6(들기)와 j5가 판을 돌리는데 오른팔은 같은 때 j5로 되돌린다.
# plate_face_tilt는 두 판 평균이라 이것을 5.5도로 세고 ~1 s뿐이라 에피소드당 ~-1(총 보상 ~200)밖에 안 된다.
# 판 힘은 13.4 N(4699 11 N, 5698 16 N)이고 j1·j3을 행동 한계의 ~70%만 쓴다.
FLATSQUEEZE_TILT_WEIGHT = -20.0  # 더 기운 판의 rad당 초당. 11도(0.19 rad) 1 s면 에피소드당 ~-3.8
FLATSQUEEZE_OFFCENTER_WEIGHT = -20.0  # 판별 면 앞뒤 오차 합(m)당 초당. 두 판 2 cm씩이면 -0.8/s
FLATSQUEEZE_FORCE_N = 16.0  # 판 법선 방향 최소 조임(판별). 5698 16 N에서 j1·j3 77~86%
FLATSQUEEZE_RAMP_STEPS = 2400  # 100 iter(24 step/iter)


def plate_face_tilt_worst(
    env,
    finger1_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_plate_object_contact"),
    finger2_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_plate_object_contact"),
    threshold: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """plate_face_tilt와 같되 두 판 중 더 기운 판의 각도(rad). 한 판만 돌아가도 평균으로 반이 되지 않는다."""
    ang_l, ang_r = _plate_face_angles(env, object_cfg, ee_frame_cfg)
    gate = mdp.object_grasped(env, finger1_sensor_cfg, finger2_sensor_cfg, threshold)
    return torch.maximum(ang_l, ang_r) * gate


def plate_face_offcenter(
    env,
    finger1_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_plate_object_contact"),
    finger2_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_plate_object_contact"),
    threshold: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """판 중심(TCP)이 담당 박스 면 중심에서 면 가로(박스 x) 방향으로 벗어난 거리(m)의 양 판 합. 양 판 접촉 중에만.

    - plates_pair_misalignment는 두 판의 상대 배치만 봐서 두 판이 함께 앞이나 뒤로 밀려도 0이다.
    - 높이(박스 z)는 보지 않는다. 판이 박스 중심보다 위를 잡으면 박스가 판-판 축에 매달린 진자처럼 바로 서려 한다.
    - 들기 보상에 곱하지 않고 따로 둔다. 중앙성을 q에 곱한 P30은 goal_tracking까지 깎아 정밀 유지가 무너졌다.
    """
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    li, ri, _, face_l, face_r, _, _ = _face_dists(env, object_cfg, ee_frame_cfg)
    x_axis = torch.zeros_like(obj.data.root_pos_w.torch)
    x_axis[:, 0] = 1.0
    x_axis_w = quat_apply(obj.data.root_quat_w.torch, x_axis)  # 박스 x축 = 판이 닿는 면의 가로 방향 (world)
    p = ee_frame.data.target_pos_w.torch
    off_l = ((p[:, li] - face_l) * x_axis_w).sum(dim=-1).abs()
    off_r = ((p[:, ri] - face_r) * x_axis_w).sum(dim=-1).abs()
    gate = mdp.object_grasped(env, finger1_sensor_cfg, finger2_sensor_cfg, threshold)
    return (off_l + off_r) * gate


def _plate_normal_forces(
    env,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_plate_object_contact"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_plate_object_contact"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """판별 박스 접촉 합력의 판 법선(TCP z축) 성분 크기 [N], (num_envs, 2) = (왼판, 오른판)."""
    ee_frame = env.scene[ee_frame_cfg.name]
    li, ri = _ee_lr_indices(ee_frame)
    z_axis = torch.zeros(env.num_envs, 3, device=env.device)
    z_axis[:, 2] = 1.0
    forces = []
    for cfg, idx in ((left_sensor_cfg, li), (right_sensor_cfg, ri)):
        f = env.scene.sensors[cfg.name].data.force_matrix_w.torch.reshape(env.num_envs, -1, 3).sum(dim=1)
        n = quat_apply(ee_frame.data.target_quat_w.torch[:, idx], z_axis)
        forces.append((f * n).sum(dim=-1).abs())
    return torch.nan_to_num(torch.stack(forces, dim=-1), nan=0.0, posinf=0.0)


def min_squeeze_normal_penalty(
    env,
    force_min: float,
    lift_height: float = MINSQUEEZE_LIFT_HEIGHT,
    contact_threshold: float = 1.0,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_plate_object_contact"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_plate_object_contact"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """min_squeeze_penalty와 같되 판별 힘을 판 법선(TCP z축) 성분으로 잰다. 판에 수직으로 누르는 힘만 조임으로 친다.

    합력 norm은 박스 무게를 받치는 판 면 방향 마찰까지 더해 조임을 크게 센다. 판이 기울면 법선 성분은 cos만큼 줄어든다.
    """
    f = _plate_normal_forces(env, left_sensor_cfg, right_sensor_cfg, ee_frame_cfg)
    gate = (f > contact_threshold).all(dim=-1).float().maximum(_lifted_perenv(env, lift_height, object_cfg))
    return (force_min - f).clamp(min=0.0, max=force_min).sum(dim=-1) * gate


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_BoxDrop_FlatOnly(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_BoxDrop_Scratch
):
    """FaceGrip-BoxDrop + 더 기운 판 기준 평행, 면 앞뒤 중앙. 조임은 2999 그대로(min_squeeze 없음). FaceGrip-BoxDrop-Scratch 2999에서 이어 학습한다.

    - plate_face_tilt: func를 plate_face_tilt_worst로 바꾸고, weight는 학습된 FACEGRIP_TILT_WEIGHT에서 시작해 success EMA 3 뒤
      FLATSQUEEZE_RAMP_STEPS에 걸쳐 FLATSQUEEZE_TILT_WEIGHT까지 올린다.
    - plate_face_offcenter: -1e-4에서 같은 방식으로 FLATSQUEEZE_OFFCENTER_WEIGHT까지 올린다.
    - 이어 학습하면 env가 새로 만들어져 curriculum이 처음부터 다시 돈다. yawtrack은 FaceGrip과 같은 이유로 바로 1로 둔다.
    - 관측·행동·critic 그룹·박스 관측 지연·끊김은 그대로라 checkpoint와 실물 노드를 그대로 쓴다.
    - FlatSqueeze(+ 판 법선 조임)와 나란히 학습해 조임 페널티가 평행·대칭에 주는 영향을 가른다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.plate_face_tilt.func = plate_face_tilt_worst
        self.rewards.plate_face_tilt.weight = FACEGRIP_TILT_WEIGHT
        self.curriculum.plate_face_tilt.params.update(weight=FLATSQUEEZE_TILT_WEIGHT, ramp_steps=FLATSQUEEZE_RAMP_STEPS)
        self.rewards.plate_face_offcenter = RewTerm(func=plate_face_offcenter, weight=-1e-4)
        self.curriculum.plate_face_offcenter = CurrTerm(
            func=ramp_reward_weight_on_success,
            params={
                "term_name": "plate_face_offcenter",
                "weight": FLATSQUEEZE_OFFCENTER_WEIGHT,
                "ramp_steps": FLATSQUEEZE_RAMP_STEPS,
            },
        )
        self.curriculum.yawtrack.params.update(trigger=0.0, ramp_steps=1)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_BoxDrop_FlatOnly_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_BoxDrop_FlatOnly
):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-BoxDrop-FlatOnly visualization/evaluation cfg (지연·끊김은 켜 둔다)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_BoxDrop_FlatSqueeze(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_BoxDrop_FlatOnly
):
    """FaceGrip-BoxDrop-FlatOnly + 판 법선 방향 조임. FaceGrip-BoxDrop-Scratch 2999에서 이어 학습한다.

    - min_squeeze(min_squeeze_normal_penalty, 판별 FLATSQUEEZE_FORCE_N): -1e-4에서 success EMA 3 뒤 FLATSQUEEZE_RAMP_STEPS에 걸쳐
      MINSQUEEZE_WEIGHT까지 올린다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.min_squeeze = RewTerm(
            func=min_squeeze_normal_penalty, params={"force_min": FLATSQUEEZE_FORCE_N}, weight=-1e-4
        )
        self.curriculum.min_squeeze = CurrTerm(
            func=ramp_reward_weight_on_success,
            params={"term_name": "min_squeeze", "weight": MINSQUEEZE_WEIGHT, "ramp_steps": FLATSQUEEZE_RAMP_STEPS},
        )


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_BoxDrop_FlatSqueeze_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_BoxDrop_FlatSqueeze
):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-BoxDrop-FlatSqueeze visualization/evaluation cfg (지연·끊김은 켜 둔다)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


# ---------- -FaceGrip-SqueezeLift-LowMuFlat: SqueezeLift 6697 + 낮은 마찰 + 더 기운 판 기준 평행 (2026-11-05) ----------
# 실로봇(11-05): SqueezeLift 6697이 FaceGrip-BoxDrop-Scratch 2999보다 강건하고, 2999는 판-박스가 미끄러진다. sim에서는 2999가
# 덜 미끄러진다(들린 동안 slip median 0.028 vs 0.070 m/s). 판 0.5와 박스 0.4~1.2의 average 결합이라 sim 접촉 마찰이
# 0.45~0.85로 실물보다 관대할 수 있다(실물 미측정). 6697은 들기 전 19 N, 들고 있을 때 17 N으로 조이고 천천히 든다.
# 6697은 오른판이 들 때 -12도(수직축)로 돌아가 있고, 판 높이 차가 3.4 cm다.
# 박스 마찰 범위를 아래로, 판 마찰도 조금 섞어 접촉 마찰(average)을 0.3~0.85로 넓힌다. 질량은 그대로(실측 501~561 g).
LOWMU_BOX_FRICTION_STATIC = (0.2, 1.2)
LOWMU_BOX_FRICTION_DYNAMIC = (0.15, 1.0)
LOWMU_PLATE_FRICTION_STATIC = (0.4, 0.5)  # 판(fr3_link7) 기본은 0.5
LOWMU_PLATE_FRICTION_DYNAMIC = (0.35, 0.45)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_SqueezeLift_LowMuFlat(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_SqueezeLift
):
    """FaceGrip-SqueezeLift + 낮은 마찰(박스 LOWMU_BOX_*, 판 LOWMU_PLATE_*, startup DR) + 더 기운 판 기준 평행. SqueezeLift 6697에서 이어 학습한다.

    - 6697이 배운 항(min_squeeze, 조임 factor, 접근 기울기·미끄럼 벌점)은 처음부터 최종 값으로 둔다. 이어 학습하면 curriculum이
      처음부터 다시 돌아서, 그대로 두면 100 iter 동안 6697과 보상이 달라진다.
    - plate_face_tilt: func를 plate_face_tilt_worst로 바꾸고, 학습된 FACEGRIP_TILT_WEIGHT에서 success EMA 3 뒤
      FLATSQUEEZE_RAMP_STEPS에 걸쳐 FLATSQUEEZE_TILT_WEIGHT까지 올린다(FlatOnly와 같다).
    - 관측·행동(action scale 0.1)·시작 자세 표는 그대로라 checkpoint와 실물 노드를 그대로 쓴다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_box_friction.params.update(
            static_friction_range=LOWMU_BOX_FRICTION_STATIC, dynamic_friction_range=LOWMU_BOX_FRICTION_DYNAMIC
        )
        self.events.randomize_plate_friction = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*_fr3_link7"]),
                "static_friction_range": LOWMU_PLATE_FRICTION_STATIC,
                "dynamic_friction_range": LOWMU_PLATE_FRICTION_DYNAMIC,
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
                "make_consistent": True,
            },
        )
        self.rewards.min_squeeze.weight = MINSQUEEZE_WEIGHT
        self.rewards.plate_face_tilt_approach.weight = APPROACH_TILT_WEIGHT
        self.rewards.plate_box_slip.weight = SLIP_WEIGHT
        self.curriculum.squeeze_lift.params.update(trigger=0.0, ramp_steps=1)
        self.rewards.plate_face_tilt.func = plate_face_tilt_worst
        self.rewards.plate_face_tilt.weight = FACEGRIP_TILT_WEIGHT
        self.curriculum.plate_face_tilt.params.update(weight=FLATSQUEEZE_TILT_WEIGHT, ramp_steps=FLATSQUEEZE_RAMP_STEPS)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_SqueezeLift_LowMuFlat_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_SqueezeLift_LowMuFlat
):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-SqueezeLift-LowMuFlat visualization/evaluation cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


# ---------- -FaceGrip-BoxDrop-FlatSqueeze-LowMuSoftLift: 낮은 접촉 마찰 + 천천히 들기 + 박스 yaw 고정 + 박스 크기 DR (2026-11-05) ----------
# 실로봇 bag(11-05): FlatSqueeze 3998은 조임 토크가 sim과 같은데, 들기 시작(0.25~0.6 s)에 판이 박스 면을 따라 3~8 cm 미끄러져
# 올라가고(sim 0.1 cm) 박스가 잡는 축 둘레로 40~70도 돈다(12번 중 11번). sim은 접촉 마찰 0.22~0.25에서 이 미끄럼을 재현한다.
# 3998은 들 때 판 힘이 18.5 N에서 14 N으로 줄고, 박스 위 가속이 0.2 s에 ~5 m/s^2(env median, p90 8.7)다. 마찰 0.22, 판 14 N이면
# 0.55 kg 박스를 붙잡고 올릴 수 있는 가속은 ~1.4 m/s^2다. 판은 실물 그대로(고무 패드 없음)라 판 마찰은 낮은 쪽으로만 흔든다.
# LowMuFlat(6697 + 낮은 마찰 + plate_face_tilt_worst)은 7200 뒤 박스를 수직축 둘레로 33도 돌려 판과 면을 맞추는 쪽으로 무너졌다.
SOFTLIFT_BOX_FRICTION_STATIC = (0.1, 0.8)
SOFTLIFT_BOX_FRICTION_DYNAMIC = (0.08, 0.7)
SOFTLIFT_PLATE_FRICTION_STATIC = (0.3, 0.5)  # 판(fr3_link7) 기본 0.5. 접촉 마찰(average) 0.2~0.65, 0.3 미만 ~14%, 0.4 미만 ~43%
SOFTLIFT_PLATE_FRICTION_DYNAMIC = (0.25, 0.45)
SOFTLIFT_ACCEL_CAP = 1.5  # m/s^2, 양 판으로 잡고 있을 때 박스 위 가속 상한
SOFTLIFT_ACCEL_WEIGHT = -20.0  # 상한 초과(m/s^2)당 초당. 3998 들기(초과분 시간 적분 ~0.2 m/s)면 에피소드당 ~-4
BOX_YAW_DRIFT_WEIGHT = -10.0  # 시작 대비 박스 yaw 변화(rad)당 초당. LowMuFlat 7696의 33도면 -5.8/s (success ~12/s)
SOFTLIFT_RAMP_STEPS = FLATSQUEEZE_RAMP_STEPS


def box_lift_accel(
    env,
    accel_cap: float = SOFTLIFT_ACCEL_CAP,
    finger1_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_plate_object_contact"),
    finger2_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_plate_object_contact"),
    threshold: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """양 판 접촉 중 박스 위(world z) 가속의 accel_cap 초과분 [m/s^2]. 가속은 policy step 사이 속도 차분이다.

    판이 박스를 붙잡고 올리려면 판당 m (g + a) / (2 mu)가 들어서 위 가속만 미끄럼에 걸린다. 아래 가속(내려놓기·떨어짐)은 보지 않는다.
    에피소드 첫 step(episode_length_buf <= 1)은 이전 속도가 지난 에피소드 것이라 0으로 둔다.
    """
    vz = env.scene[object_cfg.name].data.root_link_lin_vel_w.torch[:, 2]
    prev = getattr(env, "_box_vz_prev", None)
    if prev is None:
        prev = vz.clone()
    az = (vz - prev) / env.step_dt
    az[env.episode_length_buf <= 1] = 0.0
    env._box_vz_prev = vz.clone()
    gate = mdp.object_grasped(env, finger1_sensor_cfg, finger2_sensor_cfg, threshold)
    return torch.nan_to_num((az - accel_cap).clamp(min=0.0) * gate, nan=0.0, posinf=0.0)


def box_yaw_drift(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """에피소드 시작 대비 박스 yaw(world z 둘레) 변화 |rad|. 시작 yaw는 episode_length_buf <= 1인 step에 env별로 다시 적는다.

    _box_yaw_error(base x 기준)는 리셋 yaw DR(±0.1 rad)까지 세서 박스를 돌려 바로 세우는 것도 보상한다. 이 항은 박스를 돌린 만큼만 본다.
    """
    obj = env.scene[object_cfg.name]
    x_axis = torch.zeros_like(obj.data.root_pos_w.torch)
    x_axis[:, 0] = 1.0
    bx = quat_apply(obj.data.root_quat_w.torch, x_axis)  # 박스 x (world)
    yaw = torch.atan2(bx[:, 1], bx[:, 0])
    yaw0 = getattr(env, "_box_yaw0", None)
    if yaw0 is None:
        yaw0 = yaw.clone()
    yaw0 = torch.where(env.episode_length_buf <= 1, yaw, yaw0)
    env._box_yaw0 = yaw0
    d = yaw - yaw0
    return torch.nan_to_num(torch.atan2(torch.sin(d), torch.cos(d)).abs(), nan=0.0)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_BoxDrop_FlatSqueeze_LowMuSoftLift(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_BoxDrop_FlatSqueeze
):
    """FaceGrip-BoxDrop-FlatSqueeze + 낮은 접촉 마찰·박스 크기 DR + 들기 가속 상한·판-박스 미끄럼·박스 yaw 변화 벌점. FlatSqueeze 3998에서 이어 학습한다.

    - 마찰: 박스 SOFTLIFT_BOX_*, 판(fr3_link7) SOFTLIFT_PLATE_* (startup DR). 접촉 마찰(average) 0.2~0.65.
    - 박스 크기: 축마다 ±BOX_SIZE_DR_M (prestartup 배율). 시작 자세 표는 기준 크기 그대로라 판-면 간격이 4 cm ± 0.5 cm가 된다.
      행동은 현재 q 기준 ±0.1 rad라 조임 한계(PD 오차 상한)는 박스 폭과 무관하고, 폭에 따라 닫는 양만 달라진다.
    - 3998이 배운 항(plate_face_tilt_worst -20, plate_face_offcenter -20, min_squeeze -0.5)은 처음부터 최종 값으로 둔다.
    - box_lift_accel, plate_box_slip, box_yaw_drift: -1e-4에서 success EMA 3 뒤 SOFTLIFT_RAMP_STEPS에 걸쳐 각 weight까지.
    - 관측·행동(action scale 0.1)·시작 자세 표는 그대로라 checkpoint와 실물 노드를 그대로 쓴다.
    """

    def __post_init__(self):
        super().__post_init__()
        sx, sy, sz = self.scene.object.spawn.size
        self.events.randomize_box_size.params["scale_range"] = {
            "x": (1.0 - BOX_SIZE_DR_M / sx, 1.0 + BOX_SIZE_DR_M / sx),
            "y": (1.0 - BOX_SIZE_DR_M / sy, 1.0 + BOX_SIZE_DR_M / sy),
            "z": (1.0 - BOX_SIZE_DR_M / sz, 1.0 + BOX_SIZE_DR_M / sz),
        }
        self.events.randomize_box_friction.params.update(
            static_friction_range=SOFTLIFT_BOX_FRICTION_STATIC, dynamic_friction_range=SOFTLIFT_BOX_FRICTION_DYNAMIC
        )
        self.events.randomize_plate_friction = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*_fr3_link7"]),
                "static_friction_range": SOFTLIFT_PLATE_FRICTION_STATIC,
                "dynamic_friction_range": SOFTLIFT_PLATE_FRICTION_DYNAMIC,
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
                "make_consistent": True,
            },
        )
        self.rewards.plate_face_tilt.weight = FLATSQUEEZE_TILT_WEIGHT
        self.rewards.plate_face_offcenter.weight = FLATSQUEEZE_OFFCENTER_WEIGHT
        self.rewards.min_squeeze.weight = MINSQUEEZE_WEIGHT
        for name, func, weight in (
            ("box_lift_accel", box_lift_accel, SOFTLIFT_ACCEL_WEIGHT),
            ("plate_box_slip", plate_box_slip, SLIP_WEIGHT),
            ("box_yaw_drift", box_yaw_drift, BOX_YAW_DRIFT_WEIGHT),
        ):
            setattr(self.rewards, name, RewTerm(func=func, weight=-1e-4))
            setattr(
                self.curriculum,
                name,
                CurrTerm(
                    func=ramp_reward_weight_on_success,
                    params={"term_name": name, "weight": weight, "ramp_steps": SOFTLIFT_RAMP_STEPS},
                ),
            )


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_BoxDrop_FlatSqueeze_LowMuSoftLift_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_BoxDrop_FlatSqueeze_LowMuSoftLift
):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-BoxDrop-FlatSqueeze-LowMuSoftLift visualization/evaluation cfg (DR·지연·끊김은 켜 둔다)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


# ---------- Box410-Wide-J1Mirror-FaceGrip-BoxDrop: ready pose에서 접근부터 FaceGrip-BoxDrop 보상으로 (2026-11-05) ----------
# Pregrasp 계열은 판이 박스 면 4 cm 바깥에 놓인 상태만 배워서 실물에서 MoveIt 배치가 조금만 어긋나도 약하다.
# Wide-J1Mirror(바닥 시작, 5 s)에 FaceGrip-BoxDrop-Scratch의 plate_face_tilt ramp와 박스 관측 지연·끊김(privileged critic)을 더한다.
# Wide-J1Mirror 4999(박스 회전 제한 물리)는 들 때 판 V자 26도, 들 때 박스 기울기 최대 median 12도였다(FG-BD 2999는 5도, 5도).
# 두 계열의 보상·패널티 커리큘럼은 이 두 항 말고 같다(params/env.yaml 비교).
# 박스 yaw를 ±0.1에서 ±10도로 넓히고(yawtrack으로 똑바로 세운다), goal xy를 박스 위치와 따로 고정 범위에서 뽑는다(P13 이전처럼).
WIDE_READY_YAW_RANGE = (-0.175, 0.175)  # rad, 박스 스폰 yaw (±10도)
WIDE_READY_GOAL_X_RANGE = (0.40, 0.50)  # m, base frame goal x (Wide 박스 중심 x 0.40~0.50과 같은 범위)
WIDE_READY_GOAL_Y_RANGE = (-0.10, 0.10)  # m, base frame goal y


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror_FaceGrip_BoxDrop(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror
):
    """Wide-J1Mirror(ready pose 시작) + plate_face_tilt + 박스 관측 지연·끊김, 처음부터 학습.

    - plate_face_tilt는 success EMA 3 뒤 FACEGRIP_RAMP_STEPS에 걸쳐 FACEGRIP_TILT_WEIGHT까지(FaceGrip-BoxDrop-Scratch와 같다).
    - 박스 관측은 BoxDrop과 같은 지연·끊김(BOX_OBS_*), critic은 실제 박스 pose.
    - 박스 스폰 yaw WIDE_READY_YAW_RANGE. yawtrack은 Wide-J1Mirror 기본 ramp 그대로.
    - goal xy는 UniformPoseCommand로 WIDE_READY_GOAL_X/Y_RANGE에서 박스와 따로 뽑는다(옮기며 들기). z는 그대로.
    """

    def __post_init__(self):
        super().__post_init__()
        _add_privileged_critic(self)
        self.observations.policy.object_position.func = object_position_dropout
        self.observations.policy.object_orientation.func = object_orientation_dropout
        self.rewards.plate_face_tilt = RewTerm(func=plate_face_tilt, weight=-1e-4)
        self.curriculum.plate_face_tilt = CurrTerm(
            func=ramp_reward_weight_on_success,
            params={"term_name": "plate_face_tilt", "weight": FACEGRIP_TILT_WEIGHT, "ramp_steps": FACEGRIP_RAMP_STEPS},
        )
        self.events.reset_object_position.params["pose_range"]["yaw"] = WIDE_READY_YAW_RANGE
        self.commands.object_pose.class_type = UniformPoseCommand
        self.commands.object_pose.ranges.pos_x = WIDE_READY_GOAL_X_RANGE
        self.commands.object_pose.ranges.pos_y = WIDE_READY_GOAL_Y_RANGE


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror_FaceGrip_BoxDrop_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror_FaceGrip_BoxDrop
):
    """Sim2Real-100Hz-Light-Box410-Wide-J1Mirror-FaceGrip-BoxDrop visualization/evaluation cfg (지연·끊김은 켜 둔다)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror_FaceGrip_BoxDrop_FlatOnly(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror_FaceGrip_BoxDrop
):
    """Wide-J1Mirror-FaceGrip-BoxDrop + 더 기운 판 기준 평행, 면 앞뒤 중앙. 그 task의 checkpoint에서 이어 학습한다.

    보상·커리큘럼 변경은 Pregrasp-Wide-RollAlign-FaceGrip-BoxDrop-FlatOnly와 같다(yawtrack 바로 1 포함).
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.plate_face_tilt.func = plate_face_tilt_worst
        self.rewards.plate_face_tilt.weight = FACEGRIP_TILT_WEIGHT
        self.curriculum.plate_face_tilt.params.update(weight=FLATSQUEEZE_TILT_WEIGHT, ramp_steps=FLATSQUEEZE_RAMP_STEPS)
        self.rewards.plate_face_offcenter = RewTerm(func=plate_face_offcenter, weight=-1e-4)
        self.curriculum.plate_face_offcenter = CurrTerm(
            func=ramp_reward_weight_on_success,
            params={
                "term_name": "plate_face_offcenter",
                "weight": FLATSQUEEZE_OFFCENTER_WEIGHT,
                "ramp_steps": FLATSQUEEZE_RAMP_STEPS,
            },
        )
        self.curriculum.yawtrack.params.update(trigger=0.0, ramp_steps=1)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror_FaceGrip_BoxDrop_FlatOnly_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror_FaceGrip_BoxDrop_FlatOnly
):
    """Sim2Real-100Hz-Light-Box410-Wide-J1Mirror-FaceGrip-BoxDrop-FlatOnly visualization/evaluation cfg (지연·끊김은 켜 둔다)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror_FaceGrip_BoxDrop_FlatSqueeze(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror_FaceGrip_BoxDrop_FlatOnly
):
    """Wide-J1Mirror-FaceGrip-BoxDrop-FlatOnly + 판 법선 방향 조임(Pregrasp FlatSqueeze와 같은 min_squeeze) + 접근 중 판 기울기.

    - plate_face_tilt_approach: A 1550은 첫 양판 접촉 때 더 기운 판 32도로 모서리부터 닿는다(plate_face_tilt는 접촉 뒤에만 걸림).
      접촉 뒤 plate_face_tilt(FLATSQUEEZE_TILT_WEIGHT)와 같은 weight까지 -1e-4에서 올려 접촉 전후로 벌점이 이어지게 한다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.min_squeeze = RewTerm(
            func=min_squeeze_normal_penalty, params={"force_min": FLATSQUEEZE_FORCE_N}, weight=-1e-4
        )
        self.curriculum.min_squeeze = CurrTerm(
            func=ramp_reward_weight_on_success,
            params={"term_name": "min_squeeze", "weight": MINSQUEEZE_WEIGHT, "ramp_steps": FLATSQUEEZE_RAMP_STEPS},
        )
        self.rewards.plate_face_tilt_approach = RewTerm(func=plate_face_tilt_approach, weight=-1e-4)
        self.curriculum.plate_face_tilt_approach = CurrTerm(
            func=ramp_reward_weight_on_success,
            params={
                "term_name": "plate_face_tilt_approach",
                "weight": FLATSQUEEZE_TILT_WEIGHT,
                "ramp_steps": FLATSQUEEZE_RAMP_STEPS,
            },
        )


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror_FaceGrip_BoxDrop_FlatSqueeze_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror_FaceGrip_BoxDrop_FlatSqueeze
):
    """Sim2Real-100Hz-Light-Box410-Wide-J1Mirror-FaceGrip-BoxDrop-FlatSqueeze visualization/evaluation cfg (지연·끊김은 켜 둔다)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


# ---------- Box410-Wide-J1Mirror-FaceGrip-BoxDrop-MinSqueeze: 처음부터 판 법선 조임 + 접근 판 기울기 (2026-11-05) ----------
# Wide-J1Mirror-FaceGrip-BoxDrop 1550(256 env): 첫 양판 접촉 때 더 기운 판 32도(p90 39도)로 모서리부터 닿고 0.15 s 뒤 들기 시작한다.
# 판 법선 힘은 들기 시작 10.6 N, 들기 +0.5 s 9 N(약한 판 p10 5.4 N), 들고 나서 1 s 동안 박스가 판 대비 0.9 cm(p90 2.1 cm) 내려간다.
# plate_face_tilt는 양 판 접촉 뒤에만 걸려서 접근 중 기울기는 벌점이 없다. 실물은 판당 ~15 N은 조여야 든다.
# A(Wide-J1Mirror-FaceGrip-BoxDrop)와 나란히 처음부터 학습해, 조임을 나중에 붙이는 B(-FlatSqueeze)와 비교한다.
@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror_FaceGrip_BoxDrop_MinSqueeze(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror_FaceGrip_BoxDrop
):
    """Wide-J1Mirror-FaceGrip-BoxDrop + 판 법선 최소 조임(판별 FLATSQUEEZE_FORCE_N) + 접근 중 판 기울기, 처음부터 학습.

    - min_squeeze(min_squeeze_normal_penalty): success EMA 3 뒤 MINSQUEEZE_SCRATCH_RAMP_STEPS에 걸쳐 MINSQUEEZE_WEIGHT까지.
    - plate_face_tilt_approach(양 판이 다 닿기 전, 면 APPROACH_TILT_GAP 안의 판 각도): 같은 방식으로 APPROACH_TILT_WEIGHT까지.
    - plate_face_tilt(양 판 접촉 중)는 A 그대로라 접촉 전후로 판 각도 벌점이 이어진다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.min_squeeze = RewTerm(
            func=min_squeeze_normal_penalty, params={"force_min": FLATSQUEEZE_FORCE_N}, weight=-1e-4
        )
        self.curriculum.min_squeeze = CurrTerm(
            func=ramp_reward_weight_on_success,
            params={"term_name": "min_squeeze", "weight": MINSQUEEZE_WEIGHT, "ramp_steps": MINSQUEEZE_SCRATCH_RAMP_STEPS},
        )
        self.rewards.plate_face_tilt_approach = RewTerm(func=plate_face_tilt_approach, weight=-1e-4)
        self.curriculum.plate_face_tilt_approach = CurrTerm(
            func=ramp_reward_weight_on_success,
            params={
                "term_name": "plate_face_tilt_approach",
                "weight": APPROACH_TILT_WEIGHT,
                "ramp_steps": MINSQUEEZE_SCRATCH_RAMP_STEPS,
            },
        )


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror_FaceGrip_BoxDrop_MinSqueeze_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror_FaceGrip_BoxDrop_MinSqueeze
):
    """Sim2Real-100Hz-Light-Box410-Wide-J1Mirror-FaceGrip-BoxDrop-MinSqueeze visualization/evaluation cfg (지연·끊김은 켜 둔다)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


# ---------- Box410-Wide-J1Mirror-FaceGrip-BoxDrop-FlatSqueeze-LowMuSoftLift: ready 시작 B2 + Pregrasp LowMuSoftLift DR·벌점 (2026-11-05) ----------
# B2(-FlatSqueeze) 5400(256 env): 들고 있을 때 판 법선 16.8 N이지만, 첫 양판 접촉 0.15 s 뒤 들기 시작할 때 8.8/8.9 N, 더 기운 판 22도다.
# sim 접촉 마찰(0.45~0.85)에서는 이렇게 들어도 안 미끄러진다. 실물 bag(FlatSqueeze 3998)은 바로 이 구간에서 판이 면을 따라 미끄러졌다.
# Pregrasp LowMuSoftLift와 같은 마찰·박스 크기 DR과 box_lift_accel·plate_box_slip을 넣는다.
# box_yaw_drift(시작 대비 yaw 변화)는 ready 시작의 박스 yaw ±10도를 yawtrack으로 돌려 세우는 것까지 벌한다 -> 시작보다 base x에서
# 더 멀어진 만큼만 보는 box_yaw_away로 바꾼다.


def box_yaw_away(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """박스 yaw 오차(_box_yaw_error, base x 기준)가 에피소드 시작보다 커진 만큼 [rad] (0 아래는 0).

    시작 오차는 episode_length_buf <= 1인 step에 env별로 다시 적는다. 바로 세우는 쪽으로 돌리는 것은 0이다.
    """
    err = _box_yaw_error(env, robot_cfg, object_cfg)
    err0 = getattr(env, "_box_yaw_err0", None)
    if err0 is None:
        err0 = err.clone()
    err0 = torch.where(env.episode_length_buf <= 1, err, err0)
    env._box_yaw_err0 = err0
    return torch.nan_to_num((err - err0).clamp(min=0.0), nan=0.0)


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror_FaceGrip_BoxDrop_FlatSqueeze_LowMuSoftLift(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror_FaceGrip_BoxDrop_FlatSqueeze
):
    """Wide-J1Mirror-FaceGrip-BoxDrop-FlatSqueeze + 낮은 접촉 마찰·박스 크기 DR + 들기 가속 상한·판-박스 미끄럼·박스 yaw 벌점. B2 5400에서 이어 학습한다.

    - 마찰·크기 DR은 Pregrasp LowMuSoftLift와 같다(박스 SOFTLIFT_BOX_*, 판 SOFTLIFT_PLATE_*, 축마다 ±BOX_SIZE_DR_M).
    - B2가 배운 항(plate_face_tilt, plate_face_offcenter, min_squeeze, plate_face_tilt_approach)은 처음부터 최종 값으로 둔다.
    - box_lift_accel, plate_box_slip, box_yaw_away: -1e-4에서 SOFTLIFT_RAMP_STEPS에 걸쳐 각 weight까지. 낮은 마찰에서 success EMA가
      3 아래로 떨어지면 끝까지 안 켜질 수 있어 trigger 0으로 바로 올린다.
    """

    def __post_init__(self):
        super().__post_init__()
        sx, sy, sz = self.scene.object.spawn.size
        self.events.randomize_box_size.params["scale_range"] = {
            "x": (1.0 - BOX_SIZE_DR_M / sx, 1.0 + BOX_SIZE_DR_M / sx),
            "y": (1.0 - BOX_SIZE_DR_M / sy, 1.0 + BOX_SIZE_DR_M / sy),
            "z": (1.0 - BOX_SIZE_DR_M / sz, 1.0 + BOX_SIZE_DR_M / sz),
        }
        self.events.randomize_box_friction.params.update(
            static_friction_range=SOFTLIFT_BOX_FRICTION_STATIC, dynamic_friction_range=SOFTLIFT_BOX_FRICTION_DYNAMIC
        )
        self.events.randomize_plate_friction = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*_fr3_link7"]),
                "static_friction_range": SOFTLIFT_PLATE_FRICTION_STATIC,
                "dynamic_friction_range": SOFTLIFT_PLATE_FRICTION_DYNAMIC,
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
                "make_consistent": True,
            },
        )
        self.rewards.plate_face_tilt.weight = FLATSQUEEZE_TILT_WEIGHT
        self.rewards.plate_face_offcenter.weight = FLATSQUEEZE_OFFCENTER_WEIGHT
        self.rewards.min_squeeze.weight = MINSQUEEZE_WEIGHT
        self.rewards.plate_face_tilt_approach.weight = FLATSQUEEZE_TILT_WEIGHT
        for name, func, weight in (
            ("box_lift_accel", box_lift_accel, SOFTLIFT_ACCEL_WEIGHT),
            ("plate_box_slip", plate_box_slip, SLIP_WEIGHT),
            ("box_yaw_away", box_yaw_away, BOX_YAW_DRIFT_WEIGHT),
        ):
            setattr(self.rewards, name, RewTerm(func=func, weight=-1e-4))
            setattr(
                self.curriculum,
                name,
                CurrTerm(
                    func=ramp_reward_weight_on_success,
                    params={"term_name": name, "weight": weight, "trigger": 0.0, "ramp_steps": SOFTLIFT_RAMP_STEPS},
                ),
            )


@configclass
class DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror_FaceGrip_BoxDrop_FlatSqueeze_LowMuSoftLift_PLAY(
    DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror_FaceGrip_BoxDrop_FlatSqueeze_LowMuSoftLift
):
    """Sim2Real-100Hz-Light-Box410-Wide-J1Mirror-FaceGrip-BoxDrop-FlatSqueeze-LowMuSoftLift visualization/evaluation cfg (DR·지연·끊김은 켜 둔다)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
