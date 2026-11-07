"""Book start range of the 75 cm cabinet book task, measured from the cabinet (top and front views, matplotlib).

Numbers come from the Table75 layout in cabinet_book_insert_env_cfg.py (copied below so the script runs without Isaac Sim)
and the book start rows in assets/cabinet_book/start_table_75cm.npz. Cabinet-relative axes:
  d = depth past the cabinet's outer front face (+ away from the robot, base +x), cm
  r = distance right of the cabinet's right outer side wall (robot's right, base -y), cm

    python scripts/tools/draw_cabinet_book_start_range.py \
        --out ~/etri_ws/src/fr3_husky_etri/docs/figures/cabinet_book_75cm_book_start_range.png
"""

import argparse
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Arc, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
START_TABLE = ROOT / "source/dual_fr3_lab/dual_fr3_lab/assets/cabinet_book/start_table_75cm.npz"

# Table75 layout (base frame, m), from cabinet_book_insert_env_cfg.py
TABLE_TOP_Z = 0.345
TABLE_X = (0.40, 1.25)
TABLE_HALF_Y = 0.6
CAB_X, CAB_Y = 0.85, 0.10
CAB_HALF, WALL = 0.195, 0.020
HINGE = (0.665, 0.2555)  # door hinge axis (door mid-thickness)
DOOR_FREE_Y = CAB_Y - 0.1725  # closed door's free edge
HANDLE_ALONG, HANDLE_OUT = HINGE[1] - (CAB_Y - 0.135), 0.5 * WALL + 0.045  # bar centre from the hinge
BOOK = (0.15, 0.03, 0.22)  # depth (x), thickness (y), height
GOAL = (0.795, 0.06, 0.54)
PRE = (0.53, 0.06, 0.54)
WAIT_Y = -0.14  # DoorFirst: book centre stays at y < this until the door is open
LINK0_Y = 0.215

FRONT_X = CAB_X - CAB_HALF  # 0.655
RIGHT_Y = CAB_Y - CAB_HALF  # -0.095

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, TABLE_FILL, CAB_FILL = "#fcfcfb", "#f0efec", "#d9d8d3"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def d_of(x):
    return 100.0 * (x - FRONT_X)


def r_of(y):
    return 100.0 * (RIGHT_Y - y)


def door_point(deg, along, out=0.0):
    """Point on the door `along` m from the hinge toward the free edge and `out` m off its outer face, base xy."""
    a = math.radians(deg)
    # closed door runs from the hinge toward -y; opening swings the free edge toward the robot (-x)
    ux, uy = -math.sin(a), -math.cos(a)
    nx, ny = -math.cos(a), math.sin(a)  # outward normal (toward the robot when closed)
    return HINGE[0] + along * ux + out * nx, HINGE[1] + along * uy + out * ny


def rect_dr(ax, x, y, dx, dy, **kw):
    """Base-frame centre (x, y) and extents (dx, dy) as a rectangle in (r, d) axes."""
    ax.add_patch(Rectangle((r_of(y + 0.5 * dy), d_of(x - 0.5 * dx)), 100 * dy, 100 * dx, **kw))


def dim(ax, p0, p1, text, off=(0, 0), color=INK2, fs=8.5, rot=0):
    ax.annotate("", p1, p0, arrowprops=dict(arrowstyle="<->", color=color, lw=0.9, shrinkA=0, shrinkB=0))
    mx, my = 0.5 * (p0[0] + p1[0]) + off[0], 0.5 * (p0[1] + p1[1]) + off[1]
    ax.text(mx, my, text, color=color, fontsize=fs, ha="center", va="center", rotation=rot,
            bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.9))


def top_view(ax, starts):
    tl = 100 * (TABLE_X[1] - TABLE_X[0])
    ax.add_patch(Rectangle((r_of(TABLE_HALF_Y), d_of(TABLE_X[0])), 200 * TABLE_HALF_Y, tl, fc=TABLE_FILL, ec=MUTED, lw=0.8))
    ax.text(-57, 45, "sim table 0.85 x 1.20 m\n(top base +0.345)", fontsize=8, color=INK2, va="top", ha="left")

    # cabinet: outer square, inner square (20 mm walls), closed door
    rect_dr(ax, CAB_X, CAB_Y, 2 * CAB_HALF, 2 * CAB_HALF, fc=CAB_FILL, ec=INK2, lw=1.0)
    rect_dr(ax, CAB_X, CAB_Y, 2 * (CAB_HALF - WALL), 2 * (CAB_HALF - WALL), fc=SURFACE, ec=INK2, lw=0.8)
    ax.text(r_of(CAB_Y), d_of(CAB_X) + 6, "cabinet\n0.39 m cube, 20 mm walls", ha="center", va="center", fontsize=8.5,
            color=INK)

    # door: closed (solid), 90 and 120 deg (dashed), free-edge sweep, handle bar
    door_len = HINGE[1] - DOOR_FREE_Y
    for deg, ls, lw in ((0, "-", 2.6), (90, "--", 1.6), (120, ":", 1.6)):
        x1, y1 = door_point(deg, door_len)
        ax.plot([r_of(HINGE[1]), r_of(y1)], [d_of(HINGE[0]), d_of(x1)], color=ORANGE, ls=ls, lw=lw,
                solid_capstyle="butt")
        hx, hy = door_point(deg, HANDLE_ALONG, HANDLE_OUT)
        ax.plot(r_of(hy), d_of(hx), "o", ms=4.5, mfc=SURFACE, mec=ORANGE, mew=1.3)
    ax.add_patch(Arc((r_of(HINGE[1]), d_of(HINGE[0])), 200 * door_len, 200 * door_len, theta1=-120, theta2=0,
                     color=ORANGE, lw=0.8, ls=(0, (2, 2))))
    x90, y90 = door_point(90, door_len)
    ax.text(r_of(y90) + 1.0, d_of(x90) - 1.0, "door 90°", color=ORANGE, fontsize=8.5, ha="left", va="top")
    x120, y120 = door_point(120, door_len)
    ax.text(r_of(y120) - 1.0, d_of(x120), "120°\n(sim limit)", color=ORANGE, fontsize=8, ha="right", va="center")
    ax.plot(r_of(HINGE[1]), d_of(HINGE[0]), "o", ms=5, color=ORANGE)
    ax.text(r_of(HINGE[1]) - 1.2, d_of(HINGE[0]) + 1.5, "hinge", color=ORANGE, fontsize=8, ha="right")

    # DoorFirst wait line
    ax.axvline(r_of(WAIT_Y), ymin=0, ymax=1, color=ORANGE, lw=0.9, ls=(0, (6, 3)), alpha=0.8)
    ax.text(r_of(WAIT_Y) + 0.8, 46, "wait line (r 4.5 cm): book centre stays\nright of it until the door is open",
            color=ORANGE, fontsize=7.8, ha="left", va="top")

    # pre-insert and goal
    for p, label, fill in ((PRE, "pre-insert", SURFACE), (GOAL, "goal (12 cm in)", AQUA)):
        rect_dr(ax, p[0], p[1], BOOK[0], BOOK[1], fc=fill, ec=AQUA, lw=1.4, alpha=0.9)
        ax.text(r_of(p[1]) - 2.6, d_of(p[0]), label, color=INK, fontsize=8.5, ha="right", va="center")

    # book start region: centres (121 rows) and the book footprint at the four corner rows
    xs, ys = starts[:, 0], starts[:, 1]
    rect_dr(ax, 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max()), xs.max() - xs.min(), ys.max() - ys.min(),
            fc=BLUE, ec=BLUE, lw=1.2, alpha=0.10)
    ax.plot(r_of(ys), d_of(xs), "o", ms=2.2, color=BLUE)
    for x in (xs.min(), xs.max()):
        for y in (ys.min(), ys.max()):
            rect_dr(ax, x, y, BOOK[0], BOOK[1], fc="none", ec=BLUE, lw=1.0, ls="--")
    ax.text(r_of(0.5 * (ys.min() + ys.max())), 24, f"book start\n{len(xs)} rows", color=BLUE, fontsize=8.5,
            ha="center", va="top", fontweight="bold")

    # dimensions, cabinet-relative
    y_near, y_far, x_near, x_far = ys.max(), ys.min(), xs.min(), xs.max()
    dim(ax, (0, 31), (r_of(y_near), 31), f"{r_of(y_near):.1f}", off=(0, 1.8))
    dim(ax, (0, 37), (r_of(y_far), 37), f"{r_of(y_far):.1f} cm", off=(0, 1.8))
    face = r_of(y_near + 0.5 * BOOK[1])
    dim(ax, (0, d_of(x_near) - 11), (face, d_of(x_near) - 11), f"face {face:.0f}", off=(-0.5, -2.2))
    rr = r_of(y_far) + 4
    dim(ax, (rr, 0), (rr, d_of(x_near)), f"{d_of(x_near):+.1f}", off=(3.4, 0), rot=0)
    dim(ax, (rr, 0), (rr, d_of(x_far)), f"{d_of(x_far):+.1f}", off=(3.4, 0), rot=0)
    for v in (0.0, r_of(y_near), r_of(y_far)):
        ax.plot([v, v], [d_of(x_far) + 8, 38], color=MUTED, lw=0.6)
    ax.plot([0, rr + 1], [0, 0], color=MUTED, lw=0.6)

    # robot
    for y, name in ((LINK0_Y, "left link0"), (-LINK0_Y, "right link0")):
        ax.plot(r_of(y), d_of(0.0), "s", ms=6, color=INK2)
        ax.text(r_of(y), d_of(0.0) + 2.5, name, fontsize=8, color=INK2, ha="center")
    ax.plot(r_of(0.0), d_of(0.0), "+", ms=10, mew=1.5, color=INK)
    ax.text(r_of(0.0), d_of(0.0) - 3.0, "base origin", fontsize=8, color=INK, ha="center", va="top")
    ax.annotate("", (r_of(0.0), d_of(0.0) + 10), (r_of(0.0), d_of(0.0) + 2),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    ax.text(r_of(0.0) + 1, d_of(0.0) + 8, "+x", fontsize=8, color=INK)

    ax.plot(0, 0, "o", ms=4, color=INK)
    ax.text(0.8, -1.0, "origin: cabinet\nfront-right\nouter corner", fontsize=7.8, color=INK, ha="left", va="top")
    ax.set_xlim(-58, 52)
    ax.set_ylim(-73, 47)
    ax.set_aspect("equal")
    ax.set_xlabel("r: right of the cabinet's right outer wall [cm]   (robot's right, base −y)", fontsize=9)
    ax.set_ylabel("d: past the cabinet's outer front face [cm]   (base +x)", fontsize=9)
    ax.set_title("Top view", fontsize=11, loc="left", color=INK)


def front_view(ax, starts):
    """Seen from the robot: r to the right, height above the table up. The goal book sits behind the closed door."""
    ys = starts[:, 1]
    ax.add_patch(Rectangle((-58, -3), 110, 3, fc=TABLE_FILL, ec=MUTED, lw=0.8))
    ax.text(-49, -1.5, "table top", fontsize=8, color=INK2, va="center")
    s, w = 100 * 2 * CAB_HALF, 100 * WALL
    ax.add_patch(Rectangle((-s, 0), s, s, fc=CAB_FILL, ec=INK2, lw=1.0))
    gap = 100 * (CAB_HALF - 0.1725)  # door is 2.25 cm smaller than the front on every side
    ax.add_patch(Rectangle((-s + gap, gap), s - 2 * gap, s - 2 * gap, fc="#fbe3d9", ec=ORANGE, lw=1.4))
    ax.add_patch(Rectangle((-s + w, w), s - 2 * w, s - 2 * w, fc="none", ec=INK2, lw=0.7, ls=(0, (3, 2))))
    rh = r_of(CAB_Y - 0.135)
    ax.plot([rh, rh], [s / 2 - 13.25, s / 2 + 13.25], color=ORANGE, lw=3.0, solid_capstyle="round")
    ax.text(rh, s / 2 - 14.5, "handle", color=ORANGE, fontsize=8, ha="center", va="top")
    ax.text(-s + gap + 1.0, gap + 1.0, "door (closed)\nhinge on this side", color=ORANGE, fontsize=8, va="bottom")
    hb, hc = 100 * BOOK[2], 100 * (GOAL[2] - TABLE_TOP_Z)
    rg = r_of(GOAL[1])
    ax.add_patch(Rectangle((rg - 50 * BOOK[1], hc - hb / 2), 100 * BOOK[1], hb, fc="none", ec=AQUA, lw=1.4, ls="--"))
    ax.text(rg - 2.2, hc, "goal\n(behind\nthe door)", fontsize=8, ha="right", va="center", color=INK)
    for y in (ys.max(), ys.min()):
        ax.add_patch(Rectangle((r_of(y) - 50 * BOOK[1], 0), 100 * BOOK[1], hb, fc=BLUE, ec=BLUE, lw=1.0, alpha=0.35))
    ax.add_patch(Rectangle((r_of(ys.max()), 0), r_of(ys.min()) - r_of(ys.max()), hb, fc=BLUE, ec="none", alpha=0.08))
    ax.text(0.5 * (r_of(ys.max()) + r_of(ys.min())), hb + 1.2, "book start\n(standing)", color=BLUE, fontsize=8.5,
            ha="center", va="bottom", fontweight="bold")
    ax.axvline(r_of(WAIT_Y), color=ORANGE, lw=0.9, ls=(0, (6, 3)), alpha=0.8)
    ax.text(r_of(WAIT_Y) + 0.6, 44, "wait line", color=ORANGE, fontsize=8, va="top")
    dim(ax, (0, 30), (r_of(ys.max()), 30), f"{r_of(ys.max()):.1f}", off=(0, 1.8), fs=8)
    dim(ax, (0, 36), (r_of(ys.min()), 36), f"{r_of(ys.min()):.1f}", off=(0, 1.8), fs=8)
    dim(ax, (r_of(ys.min()) + 5, 0), (r_of(ys.min()) + 5, hb), f"{hb:.0f}", off=(2.6, 0), fs=8)
    dim(ax, (-s - 3, 0), (-s - 3, s), f"{s:.0f}", off=(-2.6, 0), fs=8)
    dim(ax, (rg + 4, hc + hb / 2), (rg + 4, s - w), f"{(s - w) - (hc + hb / 2):.1f}", off=(2.8, 0), fs=8)
    ax.set_xlim(-50, 52)
    ax.set_ylim(-5, 47)
    ax.set_aspect("equal")
    ax.set_xlabel("r [cm]   (0 = cabinet's right outer wall)", fontsize=9)
    ax.set_ylabel("height above the table [cm]", fontsize=9)
    ax.set_title("Front view (from the robot)", fontsize=11, loc="left", color=INK)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="cabinet_book_75cm_book_start_range.png")
    args = parser.parse_args()

    starts = np.load(START_TABLE)["book_start_b"]
    fig = plt.figure(figsize=(13.5, 8.0), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 0.62], wspace=0.18, hspace=0.25,
                          top=0.93)
    top = fig.add_subplot(gs[:, 0])
    side = fig.add_subplot(gs[0, 1])  # front view
    note = fig.add_subplot(gs[1, 1])
    for ax in (top, side):
        ax.set_facecolor(SURFACE)
        ax.grid(color="#e6e5e0", lw=0.5)
        ax.tick_params(labelsize=8, colors=INK2)
        for sp in ax.spines.values():
            sp.set_color(MUTED)
    top_view(top, starts)
    front_view(side, starts)

    xs, ys = starts[:, 0], starts[:, 1]
    lines = [
        ("Book start (one row per episode, uniform)", ""),
        ("  right of the cabinet's right wall (centre)", f"{r_of(ys.max()):.1f} ~ {r_of(ys.min()):.1f} cm"),
        ("  past the cabinet front face (centre)", f"{d_of(xs.min()):+.1f} ~ {d_of(xs.max()):+.1f} cm"),
        ("  from the cabinet centre (dx, dy)", f"{100 * (xs.min() - CAB_X):.0f}~{100 * (xs.max() - CAB_X):.0f}, "
                                                f"{100 * (ys.min() - CAB_Y):.0f}~{100 * (ys.max() - CAB_Y):.0f} cm"),
        ("  base frame (x, y)", f"{xs.min():.2f}~{xs.max():.2f}, {ys.min():.2f}~{ys.max():.2f} m"),
        ("  pose", "standing, axes parallel to the base; no yaw DR"),
        ("  mass", "0.15 ~ 0.30 kg (real 0.25 kg)"),
        ("  top view dashed outlines", "book footprint at the 4 corner rows"),
        ("Book 15 x 3 x 22 cm (depth x thickness x height)", ""),
        ("Goal: 14 cm past the front face (12 cm past the opening),", ""),
        ("  15.5 cm left of the right wall, 19.5 cm above the table", ""),
    ]
    note.axis("off")
    for i, (a, b) in enumerate(lines):
        bold = not a.startswith("  ")
        note.text(0.0, 1.0 - i * 0.092, a, fontsize=9, color=INK, fontweight="bold" if bold else "normal",
                  transform=note.transAxes, va="top")
        note.text(1.0, 1.0 - i * 0.092, b, fontsize=9, color=INK, transform=note.transAxes, va="top", ha="right")

    fig.suptitle("Cabinet book 75 cm (Table75-DoorFirst): book start range from the cabinet", fontsize=13, x=0.06,
                 ha="left", color=INK)
    out = os.path.expanduser(args.out)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {out}")
    print(f"book start r {r_of(ys.max()):.1f}..{r_of(ys.min()):.1f} cm, d {d_of(xs.min()):+.1f}..{d_of(xs.max()):+.1f} cm")


if __name__ == "__main__":
    main()
