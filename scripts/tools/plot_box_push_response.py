# Copyright (c) 2026, DYROS.
# SPDX-License-Identifier: BSD-3-Clause
"""Plot how the rule-based controller and an RL policy respond to a push on the held box (2026-11-13).

Reads the ``--log`` npz files of ``eval_dual_fr3_plate_rule_based.py`` and ``eval_dual_fr3_box_policy.py`` run with the same
``--push_*`` arguments (and ``--disturb 0``). Only envs that hold the box in the success zone (goal < 10 cm, tilt < 15 deg,
both plates > 1 N) just before the push are kept. Per row (one push type) and controller it draws the median and the
25-75 % band of: box distance to the goal, box tilt, box height relative to the plate midpoint (slip in the plates, zero just
before the push) and the arm command size (mean |dq| over the 14 joints).

    python scripts/tools/plot_box_push_response.py --out docs/figures/box_push_response.png \\
        --row "down 15 N" rule_down.npz rl_down.npz --row "sideways 10 N" rule_x.npz rl_x.npz
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--row", nargs=3, action="append", metavar=("LABEL", "RULE_NPZ", "RL_NPZ"), required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--after", type=float, default=3.0, help="[s] window after the push start")
parser.add_argument("--before", type=float, default=1.0, help="[s] window before the push start")
args = parser.parse_args()

SERIES = {"rule-based": "#2a78d6", "RL 5499": "#eb6834"}  # dataviz categorical slots 1, 2
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(path, kind):
    d = np.load(path)
    dt = float(d["dt"])
    t0, dur = float(d["push"][0]), float(d["push"][1])
    k0 = int(round(t0 / dt))
    box, goal, tilt, force = d["box"], d["goal"], d["tilt"], d["force"]
    plates = d["tcp"] if kind == "rule" else d["plates"]  # (env, step, side, 3)
    dq = d["dq"][..., :14]
    dist = np.linalg.norm(box - goal[:, None, :], axis=-1)
    held = (dist[:, k0 - 1] < 0.10) & (tilt[:, k0 - 1] < 15.0) & (force[:, k0 - 1] > 1.0).all(-1)
    rel_z = box[..., 2] - plates[..., 2].mean(-1)
    rel_z = rel_z - rel_z[:, k0 - 1:k0]
    sl = slice(max(0, k0 - int(round(args.before / dt))), min(box.shape[1], k0 + int(round(args.after / dt))))
    t = (np.arange(box.shape[1]) * dt - t0)[sl]
    after = slice(k0 + int(round(args.after / dt)) - 1, k0 + int(round(args.after / dt)))
    end_ok = (dist[:, after][:, 0] < 0.10) & (tilt[:, after][:, 0] < 15.0) & (force[:, after][:, 0] > 1.0).all(-1)
    dropped = box[:, after, 2][:, 0] - box[:, 0, 2] < 0.02
    return dict(t=t, dur=dur, n=int(held.sum()), n_all=box.shape[0],
                dist=100 * dist[held][:, sl], tilt=tilt[held][:, sl], slip=100 * rel_z[held][:, sl],
                cmd=np.abs(dq[held][:, sl]).mean(-1),
                back=float(end_ok[held].mean()) if held.any() else float("nan"),
                drop=float(dropped[held].mean()) if held.any() else float("nan"))


panels = [("dist", "box to goal [cm]"), ("tilt", "box tilt [deg]"), ("slip", "box z rel. plates [cm]"),
          ("cmd", "arm command mean |Δq| [rad]")]
rows = [(lab, load(r, "rule"), load(l, "rl")) for lab, r, l in args.row]
fig, axes = plt.subplots(len(rows), len(panels), figsize=(3.3 * len(panels), 2.6 * len(rows)), squeeze=False,
                         facecolor=SURF)
for i, (lab, ru, rl) in enumerate(rows):
    for j, (key, ylab) in enumerate(panels):
        ax = axes[i, j]
        ax.set_facecolor(SURF)
        ax.axvspan(0.0, ru["dur"], color=GRID, lw=0)
        for name, d in (("rule-based", ru), ("RL 5499", rl)):
            if d["n"] == 0:
                continue
            y = d[key]
            med, lo, hi = np.median(y, 0), np.percentile(y, 25, 0), np.percentile(y, 75, 0)
            ax.fill_between(d["t"], lo, hi, color=SERIES[name], alpha=0.18, lw=0)
            ax.plot(d["t"], med, color=SERIES[name], lw=2, label=f"{name} (n {d['n']})")
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=INK2, labelsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if i == len(rows) - 1:
            ax.set_xlabel("time from push start [s]", color=INK2, fontsize=8)
        ax.set_ylabel(ylab, color=INK2, fontsize=8)
        if j == 0:
            ax.set_title(f"{lab}: back in zone after {args.after:.0f} s  rule {100 * ru['back']:.0f} % / RL {100 * rl['back']:.0f} %",
                         loc="left", color=INK, fontsize=9)
        if key == "dist":
            ax.axhline(10.0, color=INK2, lw=0.8, ls=(0, (3, 3)))
    axes[i, 0].legend(frameon=False, fontsize=8, labelcolor=INK)
fig.tight_layout()
fig.savefig(args.out, dpi=150, facecolor=SURF)
for lab, ru, rl in rows:
    print(f"{lab}: held before push rule {ru['n']}/{ru['n_all']}, RL {rl['n']}/{rl['n_all']} | back in zone after {args.after} s"
          f" rule {100 * ru['back']:.0f} %, RL {100 * rl['back']:.0f} % | dropped rule {100 * ru['drop']:.0f} %, RL {100 * rl['drop']:.0f} %"
          f" | peak median dist rule {np.median(ru['dist'].max(1)):.1f}, RL {np.median(rl['dist'].max(1)):.1f} cm"
          f" | median slip at end rule {np.median(ru['slip'][:, -1]):.1f}, RL {np.median(rl['slip'][:, -1]):.1f} cm")
print(f"-> {args.out}")
