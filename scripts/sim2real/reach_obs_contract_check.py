#!/usr/bin/env python
"""Test A (offline) — dual_fr3 reach sim↔real observation/actor 계약 bit-match.

로봇·Isaac 런타임 없이 지금 실행 가능한 파트만 검증한다.

검증 항목:
  1. 배포된 npz actor(fr3_husky_etri)의 구조·메타데이터·tanh·유한성.
  2. real `build_reach_observation`(58D)이 sim ObservationsCfg가 규정한
     레이아웃·기본자세 빼기·quaternion(wxyz) 규약과 element-wise 일치하는지.
  3. real `DEFAULT_DUAL_ARM_JOINT_POSITION`이 sim robot cfg init_state와 일치하는지.

한계(별도 확인 필요, Isaac 런타임 요구):
  - Isaac articulation의 실제 joint 순서가 left_j1..7 → right_j1..7 인지.
    → scripts/sim2real/reach_joint_order_check.py 를 env_isaaclab에서 실행.
  - 배포 정확본(dual_fr3_reach_sim2real_v2/2026-09-09_11-05-39/model_4999.pt)과
    npz의 수치 일치. 해당 .pt는 이 PC에 없어(로컬은 2026-08-30 run) 여기서 미수행.

사용:
  ~/anaconda3/envs/env_isaaclab/bin/python scripts/sim2real/reach_obs_contract_check.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

# ── real 배포 코드 경로 (fr3_husky_etri) ────────────────────────────────
ETRI_PKG = Path.home() / "fr3_husky_etri" / "fr3_husky_nn_policy" / "fr3_husky_nn_policy"
NPZ_PATH = Path.home() / "fr3_husky_etri" / "fr3_husky_nn_policy" / "models" / "dual_fr3_reach_actor.npz"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


obs_mod = _load_module("etri_observation", ETRI_PKG / "observation.py")
actor_mod = _load_module("etri_numpy_actor", ETRI_PKG / "numpy_actor.py")

build_reach_observation = obs_mod.build_reach_observation
DEFAULT_DUAL_ARM = np.asarray(obs_mod.DEFAULT_DUAL_ARM_JOINT_POSITION, dtype=np.float64)
NumpyMLPActor = actor_mod.NumpyMLPActor

# ── sim 기준값 (source of truth: dual_fr3_lab cfg) ──────────────────────
# robots/dual_fr3.py init_state.joint_pos, joint 순서 = [left j1..7, right j1..7]
SIM_DEFAULT_DUAL_ARM = np.array(
    [-0.5236, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854,   # left  (j1=-pi/6)
     +0.5236, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854],  # right (j1=+pi/6)
    dtype=np.float64,
)


def sim_reference_reach_obs(q, qd, target, prev_action):
    """dual_fr3_lab reach ObservationsCfg.PolicyCfg 항 순서를 그대로 재현한 58D 기준.

    concatenate_terms=True 순서:
      joint_pos_rel(14) | joint_vel_rel(14) | reach_center_pos(3)
      | generated_commands = pos(3)+quat_wxyz(4) | last_action(14)
      | zeros_plate_forces(2) | zeros_obj_orientation(4)
    reach는 UniformPoseCommand zero-RPY → quat = (w,x,y,z)=(1,0,0,0).
    """
    q = np.asarray(q, np.float64)
    qd = np.asarray(qd, np.float64)
    target = np.asarray(target, np.float64)
    prev_action = np.asarray(prev_action, np.float64)
    quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate([
        q - SIM_DEFAULT_DUAL_ARM,   # joint_pos_rel
        qd,                          # joint_vel_rel (default vel = 0)
        target,                      # object_position 슬롯 = goal center
        target, quat_wxyz,           # target_object_position = full command
        prev_action,                 # last_action
        np.zeros(2),                 # plate_forces 슬롯
        np.zeros(4),                 # object_orientation 슬롯
    ])


def section(title):
    print(f"\n{'='*68}\n{title}\n{'='*68}")


def main() -> int:
    ok = True
    rng = np.random.default_rng(0)

    # ── 1. npz actor ────────────────────────────────────────────────────
    section("1. 배포 npz actor 구조/메타데이터")
    actor = NumpyMLPActor(NPZ_PATH)
    print(f"  input_dim={actor.input_dim}  output_dim={actor.output_dim}")
    print(f"  num_layers={actor.num_layers}  hidden_act={actor.activation}  out_act={actor.output_activation}")
    print(f"  layer shapes: {[w.shape for w in actor.weights]}")
    print(f"  source_sha256={actor.source_sha256[:16]}...")
    checks = {
        "input_dim==58": actor.input_dim == 58,
        "output_dim==14": actor.output_dim == 14,
        "hidden==elu": actor.activation == "elu",
        "output==tanh (unsquashed 배포 방지)": actor.output_activation == "tanh",
    }
    for name, cond in checks.items():
        print(f"    [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= cond
    # 샘플 추론 → tanh 경계·유한성
    amax = 0.0
    for _ in range(2000):
        a = actor(rng.standard_normal(58).astype(np.float32))
        amax = max(amax, float(np.max(np.abs(a))))
        if not np.all(np.isfinite(a)):
            print("    [FAIL] actor output non-finite"); ok = False; break
    bounded = amax <= 1.0 + 1e-6
    print(f"    [{'PASS' if bounded else 'FAIL'}] 2000 random obs → |action|max={amax:.6f} ≤ 1.0 (tanh)")
    ok &= bounded

    # ── 2. 기본자세 상수 일치 ────────────────────────────────────────────
    section("2. real DEFAULT_DUAL_ARM_JOINT_POSITION == sim init_state")
    same = np.allclose(DEFAULT_DUAL_ARM, SIM_DEFAULT_DUAL_ARM, atol=0.0)
    print(f"  real : {DEFAULT_DUAL_ARM.tolist()}")
    print(f"  sim  : {SIM_DEFAULT_DUAL_ARM.tolist()}")
    print(f"    [{'PASS' if same else 'FAIL'}] 기본자세(ready pose) 14D 정확 일치")
    ok &= same

    # ── 3. obs 레이아웃 element-wise ─────────────────────────────────────
    section("3. real build_reach_observation ≡ sim 레이아웃 (fuzz 5000)")
    max_diff = 0.0
    for _ in range(5000):
        q = rng.uniform(-2.5, 2.5, 14)
        qd = rng.uniform(-1.0, 1.0, 14)
        target = rng.uniform([0.40, -0.10, 0.10], [0.60, 0.10, 0.35])
        prev = rng.uniform(-1.0, 1.0, 14)
        real = build_reach_observation(q, qd, target, prev).astype(np.float64)
        ref = sim_reference_reach_obs(q, qd, target, prev)
        max_diff = max(max_diff, float(np.max(np.abs(real - ref))))
    layout_ok = max_diff < 1e-6
    print(f"    [{'PASS' if layout_ok else 'FAIL'}] 58D element-wise max|Δ| = {max_diff:.3e}")
    print("    (검증: 58D 순서, joint_pos_rel 기본자세 빼기, quat wxyz=(1,0,0,0),")
    print("           plate_forces/object_orientation zero 슬롯)")
    ok &= layout_ok

    # ── 결론 ─────────────────────────────────────────────────────────────
    section("결과")
    print(f"  {'✅ ALL PASS' if ok else '❌ FAIL 있음'}")
    print("\n  [Isaac 런타임 필요 — 별도 확인]")
    print("   - articulation joint 순서(left j1..7 → right j1..7) 실제 확인:")
    print("       scripts/sim2real/reach_joint_order_check.py 를 env_isaaclab에서 실행")
    print("   - npz ↔ 배포 정확 .pt(2026-09-09_11-05-39/model_4999.pt) 수치 일치:")
    print("       해당 체크포인트 있는 학습 PC에서 export_ppo_actor 재출력 후 비교")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
