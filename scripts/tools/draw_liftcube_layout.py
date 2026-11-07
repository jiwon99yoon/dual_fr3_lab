"""Cube lift Sim2Real-100Hz layout for the real robot: training cube/goal ranges, sim vs real table (matplotlib).

Numbers come from lift_cube/joint_pos_env_cfg.py (copied below so the script runs without Isaac Sim). Base frame,
robot facing +x, left arm on +y; table top base +0.345 (sim world 0.75).

    python scripts/tools/draw_liftcube_layout.py \
        --out ~/etri_ws/src/fr3_husky_etri/docs/figures/liftcube_sim2real_layout.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# lift_cube (v3 / Sim2Real-100Hz), base frame [m]
TABLE_TOP_Z = 0.345
SIM_TABLE_X, TABLE_HALF_Y = (0.30, 0.90), 0.50
REAL_TABLE_X = (0.40, 1.00)  # 600 x 1000 mm, front edge where the cabinet layout has it
CUBE = (0.05, 0.05, 0.10)
CUBE_NOMINAL = (0.45, -0.25)
CUBE_X = (0.37, 0.53)  # nominal +-0.08
CUBE_Y = (-0.37, -0.13)  # nominal +-0.12
GOAL_X, GOAL_Y, GOAL_Z = (0.35, 0.55), (-0.45, -0.15), (0.45, 0.60)
GOAL_DEFAULT = (0.45, -0.30, 0.50)
LINK0_Y = 0.215
EDGE_MARGIN = 0.025  # cube half width: the whole cube on the real table
PLACE_X = (REAL_TABLE_X[0] + EDGE_MARGIN + 0.025, CUBE_X[1])  # 2.5 cm off the edge: 0.45-0.53

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, TABLE_FILL = "#fcfcfb", "#f0efec"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def h(y):
    """Horizontal plot coordinate [cm]: robot's left (+y) on the left."""
    return -100.0 * y


def v(x):
    return 100.0 * x


def rect(ax, x0, x1, y0, y1, **kw):
    ax.add_patch(Rectangle((h(y1), v(x0)), 100 * (y1 - y0), 100 * (x1 - x0), **kw))


def top_view(ax):
    rect(ax, *REAL_TABLE_X, -TABLE_HALF_Y, TABLE_HALF_Y, fc=TABLE_FILL, ec=INK2, lw=1.2)
    ax.text(h(0.11), v(0.735), "real table 600 x 1000 mm (y -0.5~0.5)", fontsize=8.5, color=INK, va="top")
    ax.text(h(0.11), v(REAL_TABLE_X[0]) + 0.8, "real table front edge x 0.40", fontsize=8, color=INK2, va="bottom")
    ax.axhline(v(SIM_TABLE_X[0]), color=MUTED, lw=1.0, ls=(0, (5, 3)))
    ax.text(h(0.11), v(SIM_TABLE_X[0]) + 0.8, "sim table front edge x 0.30 (training)", fontsize=8, color=MUTED,
            va="bottom")

    rect(ax, *GOAL_X, *GOAL_Y, fc="none", ec=ORANGE, lw=1.3, ls=(0, (4, 2)))
    ax.text(h(GOAL_Y[0]) + 0.8, v(GOAL_X[1]), "goal x-y\nrange", fontsize=8, color=ORANGE, va="top")
    ax.plot(h(GOAL_DEFAULT[1]), v(GOAL_DEFAULT[0]), "*", ms=12, color=ORANGE, mec=SURFACE, mew=0.8)
    ax.annotate("default goal\n(0.45, -0.30, 0.50)", (h(GOAL_DEFAULT[1]) + 1.0, v(GOAL_DEFAULT[0])),
                (h(GOAL_Y[0]) + 1.0, v(0.43)), fontsize=8, color=ORANGE, va="center",
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.7))

    rect(ax, *CUBE_X, *CUBE_Y, fc=BLUE, ec=BLUE, lw=1.2, alpha=0.12)
    rect(ax, *PLACE_X, *CUBE_Y, fc=AQUA, ec=AQUA, lw=1.5, alpha=0.30)
    rect(ax, CUBE_NOMINAL[0] - 0.025, CUBE_NOMINAL[0] + 0.025, CUBE_NOMINAL[1] - 0.025, CUBE_NOMINAL[1] + 0.025,
         fc=BLUE, ec=INK, lw=0.8)
    ax.text(h(CUBE_Y[1]) - 1.0, v(0.515), "cube centre training range\nx 0.37~0.53, y -0.37~-0.13", fontsize=8.5,
            color=BLUE, ha="right", va="center", fontweight="bold")
    ax.text(h(CUBE_Y[1]) - 1.0, v(0.465), "put the cube centre here\n(real table): x 0.45~0.53", fontsize=8.5,
            color=AQUA, ha="right", va="center", fontweight="bold")
    ax.text(h(CUBE_NOMINAL[1]), v(0.385), "nominal cube (0.45, -0.25)", fontsize=7.8, color=INK, ha="center",
            va="center")
    ax.annotate("robot base (0, 0) below", (h(0.0), v(0.222)), (h(0.0), v(0.262)), fontsize=8,
                color=INK2, ha="center", va="bottom", arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9))
    ax.annotate("right link0 (0, -0.215) below", (h(-LINK0_Y), v(0.222)), (h(-LINK0_Y), v(0.262)), fontsize=8,
                color=INK2, ha="center", va="bottom", arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9))

    ax.set_xlim(h(0.12), h(-0.60))
    ax.set_ylim(v(0.22), v(0.75))
    ax.set_aspect("equal")
    ticks_y = [0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6]
    ax.set_xticks([h(t) for t in ticks_y])
    ax.set_xticklabels([f"{t:.1f}" for t in ticks_y])
    ticks_x = [0.3, 0.4, 0.5, 0.6, 0.7]
    ax.set_yticks([v(t) for t in ticks_x])
    ax.set_yticklabels([f"{t:.1f}" for t in ticks_x])
    ax.set_xlabel("base y [m]   (robot's left = image left)", fontsize=9)
    ax.set_ylabel("base x [m]", fontsize=9)
    ax.set_title("Top view (base frame)", fontsize=11, loc="left", color=INK)


def side_view(ax):
    """From the robot's right: base x to the right, height above the table up."""
    ax.add_patch(Rectangle((v(REAL_TABLE_X[0]), -3), 100 * (REAL_TABLE_X[1] - REAL_TABLE_X[0]), 3, fc=TABLE_FILL,
                           ec=INK2, lw=1.0))
    ax.plot([v(SIM_TABLE_X[0]), v(REAL_TABLE_X[0])], [0, 0], color=MUTED, lw=1.0, ls=(0, (5, 3)))
    ax.text(v(SIM_TABLE_X[0]), -1.2, "sim table edge", fontsize=7.8, color=MUTED, va="top")
    ax.text(v(REAL_TABLE_X[1]) - 1, -1.5, "real table top (base z 0.345)", fontsize=8, color=INK2, ha="right",
            va="center")
    gz0, gz1 = 100 * (GOAL_Z[0] - TABLE_TOP_Z), 100 * (GOAL_Z[1] - TABLE_TOP_Z)
    ax.add_patch(Rectangle((v(GOAL_X[0]), gz0), 100 * (GOAL_X[1] - GOAL_X[0]), gz1 - gz0, fc="none", ec=ORANGE,
                           lw=1.3, ls=(0, (4, 2))))
    ax.plot(v(GOAL_DEFAULT[0]), 100 * (GOAL_DEFAULT[2] - TABLE_TOP_Z), "*", ms=11, color=ORANGE, mec=SURFACE, mew=0.8)
    ax.text(v(GOAL_X[1]) + 1, gz1, "goal (cube centre)\n10.5~25.5 cm above\nthe table", fontsize=8, color=ORANGE,
            va="top")
    for x in CUBE_X:
        ax.add_patch(Rectangle((v(x) - 2.5, 0), 5, 10, fc=BLUE, ec=BLUE, lw=0.9, alpha=0.18, ls="--"))
    ax.add_patch(Rectangle((v(CUBE_NOMINAL[0]) - 2.5, 0), 5, 10, fc=BLUE, ec=INK, lw=0.8))
    ax.add_patch(Rectangle((v(PLACE_X[0]) - 2.5, -0.6), 100 * (PLACE_X[1] - PLACE_X[0]) + 5, 0.6, fc=AQUA, ec="none"))
    ax.text(v(0.60), 5, "cube 50 x 50 x 100, standing\n(centre 5 cm above the table);\nfaded = training x extremes,\ngreen = place on the real table",
            fontsize=8, color=BLUE, ha="left", va="center")
    ax.set_xlim(25, 105)
    ax.set_ylim(-5, 30)
    ax.set_aspect("equal")
    ax.set_xticks(range(30, 101, 10))
    ax.set_xticklabels([f"{t / 100:.1f}" for t in range(30, 101, 10)])
    ax.set_xlabel("base x [m]", fontsize=9)
    ax.set_ylabel("height above the table [cm]", fontsize=9)
    ax.set_title("Side view (from the robot's right)", fontsize=11, loc="left", color=INK)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="liftcube_sim2real_layout.png")
    args = parser.parse_args()

    fig = plt.figure(figsize=(13.5, 7.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[0.8, 1.0], wspace=0.15, hspace=0.3, top=0.92)
    top = fig.add_subplot(gs[:, 0])
    side = fig.add_subplot(gs[0, 1])
    note = fig.add_subplot(gs[1, 1])
    for ax in (top, side):
        ax.set_facecolor(SURFACE)
        ax.grid(color="#e6e5e0", lw=0.5)
        ax.tick_params(labelsize=8, colors=INK2)
        for sp in ax.spines.values():
            sp.set_color(MUTED)
    top_view(top)
    side_view(side)

    lines = [
        ("Policy", "Lift-Cube Sim2Real-100Hz, model_4999"),
        ("  fingertips", "standard Franka (not UMI)"),
        ("  start", "right arm ready pose, gripper open"),
        ("Cube (training)", "50 x 50 x 100 mm (+-5 mm DR), standing"),
        ("  centre x / y", "0.37~0.53 / -0.37~-0.13 m"),
        ("  yaw", "0 (faces parallel to the base axes)"),
        ("  mass", "0.175~0.325 kg"),
        ("Real table", "front edge x 0.40 -> keep x >= 0.45"),
        ("Goal (node parameter)", "x 0.35~0.55, y -0.45~-0.15, z 0.45~0.60"),
        ("Episode", "8 s at 20 Hz"),
    ]
    note.axis("off")
    for i, (a, b) in enumerate(lines):
        bold = not a.startswith("  ")
        note.text(0.0, 1.0 - i * 0.095, a, fontsize=9, color=INK, fontweight="bold" if bold else "normal",
                  transform=note.transAxes, va="top")
        note.text(1.0, 1.0 - i * 0.095, b, fontsize=9, color=INK, transform=note.transAxes, va="top", ha="right")

    fig.suptitle("Cube lift Sim2Real-100Hz (model_4999): where to put the cube", fontsize=13, x=0.06, ha="left",
                 color=INK)
    out = os.path.expanduser(args.out)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
