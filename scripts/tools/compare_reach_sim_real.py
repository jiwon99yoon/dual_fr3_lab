# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""dual_fr3 reach — sim vs real trajectory 비교 (sim2real gap 시각화).

``eval_dual_fr3_reach_traj.py``가 저장한 sim ``.npz``와, 실물 ROS2 노드가 같은 규약으로
수집한 real ``.csv``를 겹쳐 그린다. remove_hook의 compare_sim_real_trajectory.py 구조를
양팔(14 joint) + EE tracking으로 확장했다.

real CSV 규약 (한 파일 = 한 에피소드, control-step당 한 행):
  t_sec,
  qreal_L1..L7, qreal_R1..R7        (실측 joint pos [rad], sim joint_order와 동일 순서)
  ee_left_x/y/z, ee_right_x/y/z     (실측 EE pos [m], robot base(local) frame)
  tgt_left_x/y/z, tgt_right_x/y/z   (목표점 [m], 옵션 — 없으면 sim target으로 대체)
컬럼 일부가 없으면 있는 것만 그린다(joint만, 또는 EE만).

생성 plot:
  A. joint_dq.png       : 14관절 Δq(초기값 기준) — 제어 모션 shape 비교 (offset 무관)
  B. ee_error.png       : left/right EE tracking error 시계열 — 제어 지연·정상상태 오차
  C. ee_xyz.png         : left/right EE XYZ 궤적 (sim vs real 겹침)

.. code-block:: bash

    python scripts/tools/compare_reach_sim_real.py \
        --sim_npz logs/rsl_rl/dual_fr3_reach/<run>/trajectories/reach_traj.npz \
        --real_csvs real_logs/reach_ep*.csv \
        --out_dir logs/reach_sim_real_comparison
"""

import argparse
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── 데이터 로더 ────────────────────────────────────────────────────────────────

def load_sim_npz(path):
    """sim npz → per-env(=에피소드) dict 리스트. (T,N,dim) → env별 (T,dim)."""
    data = np.load(path, allow_pickle=True)
    n = int(data["num_envs"])
    q = data["q"]            # (T, N, 14)
    ee_l = data["ee_left"]   # (T, N, 3)
    ee_r = data["ee_right"]
    tgt_l = data["target_left"]
    tgt_r = data["target_right"]
    err_l = data["err_left"]  # (T, N)
    err_r = data["err_right"]
    eps = []
    for i in range(n):
        eps.append({
            "q": q[:, i, :],
            "ee_left": ee_l[:, i, :], "ee_right": ee_r[:, i, :],
            "tgt_left": tgt_l[:, i, :], "tgt_right": tgt_r[:, i, :],
            "err_left": err_l[:, i], "err_right": err_r[:, i],
        })
    return eps, {
        "dt": float(data["dt"]),
        "hz": float(data["hz"]),
        "joint_order": [str(x) for x in data["joint_order"]],
    }


def load_real_csv(path):
    """real CSV → dict. 있는 컬럼만 채운다(joint/EE/target 각각 옵션)."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 3:
        return None
    cols = rows[0].keys()

    def col(name):
        return np.array([float(r[name]) for r in rows])

    ep = {"n_steps": len(rows), "fname": os.path.basename(path)}

    # joint (14): qreal_L1..L7, qreal_R1..R7
    jnames = [f"qreal_L{i}" for i in range(1, 8)] + [f"qreal_R{i}" for i in range(1, 8)]
    if all(n in cols for n in jnames):
        ep["q"] = np.stack([col(n) for n in jnames], axis=1)  # (T, 14)

    # EE pos
    for side in ("left", "right"):
        keys = [f"ee_{side}_{a}" for a in ("x", "y", "z")]
        if all(k in cols for k in keys):
            ep[f"ee_{side}"] = np.stack([col(k) for k in keys], axis=1)
        tkeys = [f"tgt_{side}_{a}" for a in ("x", "y", "z")]
        if all(k in cols for k in tkeys):
            ep[f"tgt_{side}"] = np.stack([col(k) for k in tkeys], axis=1)

    ep["t"] = col("t_sec") if "t_sec" in cols else None
    return ep


# ── 색 ──────────────────────────────────────────────────────────────────────
SIM_COLORS = ["#2166ac", "#4393c3", "#74add1", "#abd9e9"]   # 파랑 계열 (sim eps)
REAL_COLORS = ["#d73027", "#f46d43", "#fdae61", "#fee090"]  # 빨강/주황 (real eps)


def _time(n, hz):
    return np.arange(n) / hz


# ── Figure A: 14관절 Δq (모션 shape) ──────────────────────────────────────────

def plot_joint_dq(sim_eps, real_eps, hz, joint_order, out_dir):
    """Δq = q(t) - q(0). 절대 offset 제거 → 순수 제어 모션 비교. 7x2 grid (L 왼열, R 오른열)."""
    have_sim = [e for e in sim_eps if "q" in e]
    have_real = [e for e in real_eps if "q" in e]
    if not have_sim and not have_real:
        print("  [joint_dq] joint 데이터 없음 — skip")
        return

    fig, axes = plt.subplots(7, 2, figsize=(14, 20))
    for j in range(14):
        r, c = j % 7, j // 7  # 0~6=left(왼열), 7~13=right(오른열)
        ax = axes[r, c]
        for k, ep in enumerate(have_sim):
            y = ep["q"][:, j] - ep["q"][0, j]
            ax.plot(_time(len(y), hz), y, color=SIM_COLORS[k % len(SIM_COLORS)],
                    lw=1.3, alpha=0.7, label=f"sim ep{k+1}" if k < 4 else "_nolegend_")
        for k, ep in enumerate(have_real):
            y = ep["q"][:, j] - ep["q"][0, j]
            t = ep["t"] if ep.get("t") is not None else _time(len(y), hz)
            ax.plot(t, y, color=REAL_COLORS[k % len(REAL_COLORS)],
                    lw=1.4, alpha=0.8, label=f"real ep{k+1}" if k < 4 else "_nolegend_")
        name = joint_order[j] if j < len(joint_order) else f"j{j}"
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("time [s]"); ax.set_ylabel("Δq [rad]")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Joint Δq from init — motion shape (sim vs real)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(out_dir, "joint_dq.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  saved: {out}")


# ── Figure B: EE tracking error 시계열 ────────────────────────────────────────

def plot_ee_error(sim_eps, real_eps, hz, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for si, side in enumerate(("left", "right")):
        ax = axes[si]
        for k, ep in enumerate(sim_eps):
            key = f"err_{side}"
            if key not in ep:
                continue
            y = ep[key]
            ax.plot(_time(len(y), hz), y * 100, color=SIM_COLORS[k % len(SIM_COLORS)],
                    lw=1.3, alpha=0.7, label=f"sim ep{k+1}" if k < 4 else "_nolegend_")
        for k, ep in enumerate(real_eps):
            # real은 error 컬럼 대신 ee/target으로 계산 (target 없으면 skip)
            if f"ee_{side}" not in ep or f"tgt_{side}" not in ep:
                continue
            y = np.linalg.norm(ep[f"ee_{side}"] - ep[f"tgt_{side}"], axis=1)
            t = ep["t"] if ep.get("t") is not None else _time(len(y), hz)
            ax.plot(t, y * 100, color=REAL_COLORS[k % len(REAL_COLORS)],
                    lw=1.4, alpha=0.8, label=f"real ep{k+1}" if k < 4 else "_nolegend_")
        ax.axhline(3.0, color="k", ls="--", lw=0.7, alpha=0.5, label="3cm")
        ax.set_title(f"{side} EE tracking error", fontsize=11)
        ax.set_xlabel("time [s]"); ax.set_ylabel("error [cm]")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("EE tracking error (sim vs real)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(out_dir, "ee_error.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  saved: {out}")


# ── Figure C: EE XYZ 궤적 ─────────────────────────────────────────────────────

def plot_ee_xyz(sim_eps, real_eps, hz, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for si, side in enumerate(("left", "right")):
        for a, axis in enumerate(("x", "y", "z")):
            ax = axes[si, a]
            for k, ep in enumerate(sim_eps):
                if f"ee_{side}" not in ep:
                    continue
                y = ep[f"ee_{side}"][:, a]
                ax.plot(_time(len(y), hz), y, color=SIM_COLORS[k % len(SIM_COLORS)],
                        lw=1.3, alpha=0.7, label=f"sim ep{k+1}" if k < 4 else "_nolegend_")
                if f"tgt_{side}" in ep:  # 목표선 (첫 sim ep만)
                    if k == 0:
                        ax.axhline(ep[f"tgt_{side}"][0, a], color="green", ls=":", lw=1.0,
                                   alpha=0.7, label="target")
            for k, ep in enumerate(real_eps):
                if f"ee_{side}" not in ep:
                    continue
                y = ep[f"ee_{side}"][:, a]
                t = ep["t"] if ep.get("t") is not None else _time(len(y), hz)
                ax.plot(t, y, color=REAL_COLORS[k % len(REAL_COLORS)],
                        lw=1.4, alpha=0.8, label=f"real ep{k+1}" if k < 4 else "_nolegend_")
            ax.set_title(f"{side} EE {axis}", fontsize=10)
            ax.set_xlabel("time [s]"); ax.set_ylabel(f"{axis} [m]")
            ax.grid(True, alpha=0.3)
            if si == 0 and a == 0:
                ax.legend(fontsize=8)
    fig.suptitle("EE position trajectory (sim vs real, base frame)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(out_dir, "ee_xyz.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_npz", required=True)
    ap.add_argument("--real_csvs", nargs="*", default=[],
                    help="실물 에피소드 CSV (glob 패턴 가능). 없으면 sim만 그린다.")
    ap.add_argument("--out_dir", default="logs/reach_sim_real_comparison")
    ap.add_argument("--max_sim_eps", type=int, default=4, help="그릴 sim env(에피소드) 최대 수")
    ap.add_argument("--min_steps", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading sim: {args.sim_npz}")
    sim_eps, meta = load_sim_npz(args.sim_npz)
    sim_eps = sim_eps[:args.max_sim_eps]
    hz = meta["hz"]
    print(f"  {len(sim_eps)} sim episodes (env), {hz:.0f}Hz, joints={len(meta['joint_order'])}")

    real_eps = []
    paths = sorted(set(p for pat in args.real_csvs for p in glob.glob(pat)))
    for p in paths:
        ep = load_real_csv(p)
        if ep is None or ep["n_steps"] < args.min_steps:
            print(f"  skip (짧음/빈파일): {os.path.basename(p)}")
            continue
        real_eps.append(ep)
        have = [k for k in ("q", "ee_left", "ee_right") if k in ep]
        print(f"  [real] {ep['fname']}: {ep['n_steps']} steps, cols={have}")

    if not real_eps:
        print("  (real CSV 없음 — sim 단독 plot)")

    print("\nGenerating plots...")
    plot_joint_dq(sim_eps, real_eps, hz, meta["joint_order"], args.out_dir)
    plot_ee_error(sim_eps, real_eps, hz, args.out_dir)
    plot_ee_xyz(sim_eps, real_eps, hz, args.out_dir)

    print(f"\nOutput: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
