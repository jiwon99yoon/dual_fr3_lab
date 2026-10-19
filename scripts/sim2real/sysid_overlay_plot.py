#!/usr/bin/env python
"""SysID 3자(real/mujoco/isaac) overlay 플롯.

같은 관절·profile·amp의 CSV를 겹쳐 position 추종(q-q0)과 torque(tau_meas)를
시간축으로 비교한다. sim2sim(isaac↔mujoco), sim2real(isaac/mujoco↔real) gap을
눈으로 확인하기 위한 용도. Isaac 불필요, CSV만 읽는다.

실행:
  python scripts/sim2real/sysid_overlay_plot.py \
      --base ~/Downloads/sysid_logs_260911_1422/sysid_logs \
      --joint left_fr3_joint1 --profile ramp --amp 0.250
  # --joint all 로 14관절 한 번에
"""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DUAL_ARM = [f"{s}_fr3_joint{i}" for s in ("left", "right") for i in range(1, 8)]
SRCS = [("real", "red"), ("mujoco", "blue"), ("isaac", "green")]


def newest(srcdir, joint, profile, amp):
    pat = os.path.join(srcdir, f"{joint}_{profile}_amp{amp:.3f}_*.csv")
    m = sorted(glob.glob(pat), key=os.path.getmtime)
    return m[-1] if m else None


def read(path, joint):
    rows = list(csv.DictReader(open(path)))
    qk = f"q_{joint}"
    q0k = f"q0_{joint}"
    q0 = float(rows[0][q0k]) if q0k in rows[0] else float(rows[0][qk])
    t = [float(r["t_sec"]) for r in rows]
    dq = [float(r[qk]) - q0 for r in rows]
    tau = [float(r[f"tau_meas_{joint}"]) for r in rows]
    ref = [float(r.get("q_ref_delta_from_q0_rad", "nan") or "nan") for r in rows]
    return t, dq, tau, ref


def plot_joint(srcdirs, joint, profile, amp, out_dir, tag):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ref_drawn = False
    found = []
    for src, color in SRCS:
        p = newest(srcdirs[src], joint, profile, amp)
        if not p:
            continue
        found.append(src)
        t, dq, tau, ref = read(p, joint)
        if not ref_drawn and any(r == r for r in ref):  # ref_delta 존재
            ax1.plot(t, ref, "k--", lw=1, alpha=0.6, label="commanded (ref_delta)")
            ref_drawn = True
        ax1.plot(t, dq, color=color, label=src)
        ax2.plot(t, tau, color=color, label=src)
    ax1.set_ylabel("q − q0 [rad]")
    ax1.set_title(f"{joint}  ({profile}, amp={amp:.3f})   sources: {', '.join(found)}")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_ylabel("tau_meas [Nm]"); ax2.set_xlabel("t [s]")
    ax2.legend(); ax2.grid(alpha=0.3)
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    out = os.path.join(out_dir, f"overlay_{joint}_{profile}_amp{amp:.3f}{suffix}.png")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print(f"  saved {out}  (sources: {found})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="sysid_logs 루트(real/mujoco/isaac 하위폴더)")
    ap.add_argument("--joint", default="all")
    ap.add_argument("--profile", default="ramp")
    ap.add_argument("--amp", type=float, default=0.250)
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="", help="PNG 파일명 접미사(ablation 구분용, 예: fric0.0). 덮어쓰기 방지.")
    ap.add_argument("--real_dir", default="", help="real CSV 폴더 override(기본 <base>/real)")
    ap.add_argument("--mujoco_dir", default="", help="mujoco CSV 폴더 override(기본 <base>/mujoco)")
    ap.add_argument("--isaac_dir", default="", help="isaac CSV 폴더 override(ablation 변형 폴더 지정용)")
    a = ap.parse_args()
    base = os.path.expanduser(a.base)
    srcdirs = {
        "real": os.path.expanduser(a.real_dir) if a.real_dir else os.path.join(base, "real"),
        "mujoco": os.path.expanduser(a.mujoco_dir) if a.mujoco_dir else os.path.join(base, "mujoco"),
        "isaac": os.path.expanduser(a.isaac_dir) if a.isaac_dir else os.path.join(base, "isaac"),
    }
    out_dir = os.path.expanduser(a.out) if a.out else os.path.join(base, "overlay")
    joints = DUAL_ARM if a.joint == "all" else [a.joint]
    for j in joints:
        plot_joint(srcdirs, j, a.profile, a.amp, out_dir, a.tag)


if __name__ == "__main__":
    main()
