"""Cube pick-and-place layout for the real robot: table, cube start / goal range, test pairs (matplotlib).

Numbers come from pick_place_cube/mdp/pick_place.py (copied below so the script runs without Isaac Sim). Base frame,
robot facing +x, left arm on +y; table top base +0.345 (sim world 0.75). Test-pair success rates are from
model_4000 of release_goalfix_5070 (32 envs per pair, deterministic, DR on, 2026-11-09).

    python scripts/tools/draw_pick_place_layout.py \
        --out ~/etri_ws/src/fr3_husky_etri/docs/figures/pick_place_cube_layout.png
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

# pick_place_cube, base frame [m]
TABLE_TOP_Z = 0.345
TABLE_X, TABLE_HALF_Y = (0.40, 1.00), 0.50  # real 600 x 1000 mm
CUBE = (0.05, 0.05, 0.10)
AREA_X, AREA_Y = (0.45, 0.65), (-0.45, 0.05)  # cube start and goal (same range)
LINK0_Y = 0.215
REACH_RADIUS = 0.62  # from the right link0 axis
GOAL_MIN_DIST = 0.10
XY_TOL = 0.025
READY_TCP = (0.307, -0.215, 0.413)
GRASP_TIP = 0.05  # fingertip above the table at grasp (rule-based); model_4000 median 3.6 cm
HAND_CLEAR = 0.03

# name, cube (x, y, yaw deg), goal (x, y), sim success (model_4000, 32 envs)
PAIRS = [
    ("A", (0.50, -0.30, 0.0), (0.55, -0.10), "30/32"),
    ("B", (0.55, -0.10, 0.0), (0.50, -0.30), "31/32"),
    ("C", (0.50, -0.15, 0.0), (0.58, -0.35), "27/32"),
    ("D", (0.58, -0.35, 0.0), (0.50, -0.15), "30/32"),
    ("E", (0.50, -0.40, 0.0), (0.50, -0.25), "31/32"),
    ("F", (0.55, -0.25, 30.0), (0.50, 0.00), "32/32"),
    ("G", (0.55, -0.25, -30.0), (0.58, -0.40), "29/32"),
]

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, TABLE_FILL = "#fcfcfb", "#f0efec"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def h(y):
    """Horizontal plot coordinate [cm]: robot's left (+y) on the left."""
    return -100.0 * y


def v(x):
    return 100.0 * x


def rect(ax, x0, x1, y0, y1, **kw):
    return ax.add_patch(Rectangle((h(y1), v(x0)), 100 * (y1 - y0), 100 * (x1 - x0), **kw))


def cube_square(ax, x, y, yaw_deg, **kw):
    a, s = math.radians(yaw_deg), 0.5 * CUBE[0]
    corners = []
    for cx, cy in ((s, s), (-s, s), (-s, -s), (s, -s)):
        px = x + cx * math.cos(a) - cy * math.sin(a)
        py = y + cx * math.sin(a) + cy * math.cos(a)
        corners.append((h(py), v(px)))
    ax.add_patch(Polygon(corners, closed=True, **kw))


def top_view(ax):
    rect(ax, *TABLE_X, -TABLE_HALF_Y, TABLE_HALF_Y, fc=TABLE_FILL, ec=INK2, lw=1.2)
    ax.text(h(0.48), v(0.985), "table 600 x 1000 mm: x 0.40~1.00, y -0.50~0.50\ntop = base z +0.345", fontsize=8.5,
            color=INK, va="top")
    ax.text(h(0.48), v(TABLE_X[0]) + 0.8, "table front edge x 0.40", fontsize=8, color=INK2, va="bottom")

    # cube start / goal area: rectangle clipped by the right-arm reach circle
    reach = Circle((h(-LINK0_Y), v(0.0)), 100 * REACH_RADIUS, fc="none", ec=MUTED, lw=0.9, ls=(0, (5, 3)))
    ax.add_patch(reach)
    area = rect(ax, *AREA_X, *AREA_Y, fc=BLUE, ec="none", alpha=0.13)
    area.set_clip_path(Circle((h(-LINK0_Y), v(0.0)), 100 * REACH_RADIUS, transform=ax.transData))
    rect(ax, *AREA_X, *AREA_Y, fc="none", ec=BLUE, lw=1.0, ls=(0, (4, 2)))
    ax.text(h(AREA_Y[1]) - 0.5, v(AREA_X[1]) + 0.8,
            "cube start & goal range (centre)\nx 0.45~0.65, y -0.45~0.05,\nwithin 0.62 m of the right link0", fontsize=8.5,
            color=BLUE, va="bottom", fontweight="bold")
    ax.text(h(-0.60), v(0.30), "right-arm reach\n0.62 m", fontsize=7.5, color=MUTED, ha="center")

    ax.plot(h(READY_TCP[1]), v(READY_TCP[0]), "o", ms=6, color=AQUA, mec=SURFACE, mew=0.8)
    ax.text(h(READY_TCP[1]) + 1.5, v(READY_TCP[0]), "ready TCP (0.31, -0.22)\n6.8 cm above the table height", fontsize=7.8,
            color=AQUA, va="center")
    for y, lab in ((LINK0_Y, "left link0"), (0.0, "base (0, 0)"), (-LINK0_Y, "right link0")):
        ax.plot(h(y), v(0.0), "s", ms=6, color=INK2)
        ax.text(h(y), v(0.0) - 2.5, f"{lab}\ny {y:+.3f}" if y else lab, fontsize=7.8, color=INK2, ha="center", va="top")

    ax.set_xlim(h(0.52), h(-0.66))
    ax.set_ylim(v(-0.08), v(1.03))
    ax.set_aspect("equal")
    ticks_y = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6]
    ax.set_xticks([h(t) for t in ticks_y])
    ax.set_xticklabels([f"{t:.1f}" for t in ticks_y])
    ticks_x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_yticks([v(t) for t in ticks_x])
    ax.set_yticklabels([f"{t:.1f}" for t in ticks_x])
    ax.set_xlabel("base y [m]   (robot's left = image left)", fontsize=9)
    ax.set_ylabel("base x [m]   (robot's front = up)", fontsize=9)
    ax.set_title("Top view (base frame)", fontsize=11, loc="left", color=INK)


def pair_view(ax, name, cube, goal, succ):
    """One test pair, zoomed on the start/goal range."""
    cx, cy, yaw = cube
    gx, gy = goal
    rect(ax, *AREA_X, *AREA_Y, fc=BLUE, ec=BLUE, lw=0.8, ls=(0, (4, 2)), alpha=0.07)
    ax.add_patch(Circle((h(gy), v(gx)), 100 * XY_TOL, fc=ORANGE, ec=ORANGE, lw=0.8, alpha=0.25))
    ax.plot(h(gy), v(gx), "+", ms=8, color=ORANGE, mew=1.5)
    ax.annotate("", (h(gy), v(gx)), (h(cy), v(cx)), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0, shrinkA=5,
                                                                      shrinkB=5))
    cube_square(ax, cx, cy, yaw, fc=BLUE, ec=INK, lw=0.7)
    ax.set_xlim(h(0.09), h(-0.49))
    ax.set_ylim(v(0.42), v(0.68))
    ax.set_aspect("equal")
    ax.set_xticks([h(t) for t in (0.0, -0.2, -0.4)])
    ax.set_xticklabels(["0.0", "-0.2", "-0.4"])
    ax.set_yticks([v(t) for t in (0.45, 0.55, 0.65)])
    ax.set_yticklabels(["0.45", "0.55", "0.65"])
    move = 100 * math.dist((cx, cy), goal)
    ax.set_title(f"{name}: sim {succ}", fontsize=10, loc="left", color=INK, fontweight="bold")
    ax.set_xlabel(f"cube ({cx:.2f}, {cy:+.2f}), yaw {yaw:+.0f}\ngoal ({gx:.2f}, {gy:+.2f}), move {move:.0f} cm",
                  fontsize=8, color=INK)


def side_view(ax):
    """From the robot's right: base x to the right, height above the table up."""
    ax.add_patch(Rectangle((v(TABLE_X[0]), -3), 100 * (TABLE_X[1] - TABLE_X[0]), 3, fc=TABLE_FILL, ec=INK2, lw=1.0))
    ax.text(v(TABLE_X[1]) - 1, -1.5, "table top (base z 0.345)", fontsize=8, color=INK2, ha="right", va="center")
    x0 = 0.50
    ax.add_patch(Rectangle((v(x0) - 2.5, 0), 5, 10, fc=BLUE, ec=INK, lw=0.8))
    ax.text(v(x0) + 3.5, 9.5, "cube 50 x 50 x 100, standing", fontsize=8, color=BLUE, va="top")
    ax.plot([v(x0) - 6, v(x0) + 6], [100 * GRASP_TIP] * 2, color=AQUA, lw=1.4)
    ax.text(v(x0) + 6.5, 100 * GRASP_TIP, "grasp: fingertips ~5 cm above the table\n(cube centre; policy median 3.6 cm)",
            fontsize=7.8, color=AQUA, va="center")
    top_clear = 100 * (CUBE[2] + HAND_CLEAR)
    ax.plot([v(x0) - 6, v(x0) + 6], [top_clear] * 2, color=ORANGE, lw=1.2, ls=(0, (4, 2)))
    ax.text(v(x0) + 6.5, top_clear, "success: fingertips >= 3 cm above the cube top", fontsize=7.8, color=ORANGE,
            va="center")
    ax.plot(v(READY_TCP[0]), 100 * (READY_TCP[2] - TABLE_TOP_Z), "o", ms=6, color=AQUA, mec=SURFACE, mew=0.8)
    ax.text(v(READY_TCP[0]), 100 * (READY_TCP[2] - TABLE_TOP_Z) + 1.5, "ready TCP", fontsize=7.8, color=AQUA,
            ha="center", va="bottom")
    ax.set_xlim(25, 100)
    ax.set_ylim(-5, 20)
    ax.set_aspect("equal")
    ax.set_xticks(range(30, 101, 10))
    ax.set_xticklabels([f"{t / 100:.1f}" for t in range(30, 101, 10)])
    ax.set_xlabel("base x [m]", fontsize=9)
    ax.set_ylabel("height above the table [cm]", fontsize=9)
    ax.set_title("Side view (from the robot's right)", fontsize=11, loc="left", color=INK)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="pick_place_cube_layout.png")
    args = parser.parse_args()

    fig = plt.figure(figsize=(16.5, 13.5), facecolor=SURFACE)
    outer = fig.add_gridspec(2, 1, height_ratios=[2.3, 0.75], hspace=0.12, top=0.94)
    gs = outer[0].subgridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[0.5, 1.0], wspace=0.12, hspace=0.25)
    top = fig.add_subplot(gs[:, 0])
    side = fig.add_subplot(gs[0, 1])
    note = fig.add_subplot(gs[1, 1])
    pgs = outer[1].subgridspec(1, len(PAIRS), wspace=0.18)
    pair_axes = [fig.add_subplot(pgs[0, i]) for i in range(len(PAIRS))]
    for ax in (top, side, *pair_axes):
        ax.set_facecolor(SURFACE)
        ax.grid(color="#e6e5e0", lw=0.5)
        ax.tick_params(labelsize=8, colors=INK2)
        for sp in ax.spines.values():
            sp.set_color(MUTED)
    top_view(top)
    side_view(side)
    for ax, (name, cube, goal, succ) in zip(pair_axes, PAIRS):
        pair_view(ax, name, cube, goal, succ)
        ax.tick_params(labelsize=7)
    y_head = pair_axes[0].get_position().y1 + 0.035
    fig.text(0.125, y_head, "Test pairs (zoom on the start/goal range; blue = cube start with yaw, orange = goal with the "
             "2.5 cm success circle; sim = model_4000 successes out of 32 with DR)", fontsize=10.5, color=INK)

    lines = [
        ("Policy", "PickPlace-Cube GripperBusy, model_4000"),
        ("  run", "release_goalfix_5070 (sim 91% deterministic)"),
        ("  start", "right arm ready pose, gripper open; left arm fixed"),
        ("Cube", "50 x 50 x 100 mm standing, 70~100 g (real 90 g)"),
        ("  yaw", "-45~+45 deg (any face toward the robot)"),
        ("Start & goal", "x 0.45~0.65, y -0.45~0.05, <= 0.62 m from right link0"),
        ("  goal", ">= 10 cm from the start, cube standing on the table"),
        ("Success", "goal xy <= 2.5 cm, tilt <= 5 deg, released,"),
        ("", "fingertips >= 3 cm above the cube top"),
        ("Episode", "14 s at 20 Hz (sim: success after ~3 s)"),
        ("Gripper (node)", "grasp 30 N / move 0.1 m/s, hysteresis 0,"),
        ("", "busy bit = goal sent, result not received"),
        ("Vision", "cube pose in base frame; hold the last value when hidden"),
        ("Robot", "base frame = Husky base; x front, y left, z up"),
    ]
    note.axis("off")
    step = 1.0 / (len(lines) + 0.5)
    for i, (a, b) in enumerate(lines):
        bold = a and not a.startswith("  ")
        note.text(0.0, 1.0 - i * step, a, fontsize=9, color=INK, fontweight="bold" if bold else "normal",
                  transform=note.transAxes, va="top")
        note.text(1.0, 1.0 - i * step, b, fontsize=9, color=INK, transform=note.transAxes, va="top", ha="right")

    fig.suptitle("Cube pick-and-place (model_4000): table, cube start / goal and test pairs (base frame)", fontsize=13,
                 x=0.05, ha="left", color=INK)
    out = os.path.expanduser(args.out)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
