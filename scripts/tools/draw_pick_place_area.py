"""Cube pick-and-place training area for marking the real table: sampled start/goal and mm from the table edges.

Same sampling as pick_place_cube/mdp/pick_place.py (``sample_reachable_xy`` and ``PickPlaceGoalCommand``): cube start
uniform in the x/y rectangle, redrawn outside the right-arm reach circle; goal the same, also redrawn closer than
GOAL_MIN_DIST to the start. Base frame, robot facing +x, left arm on +y.

    python scripts/tools/draw_pick_place_area.py \
        --out ~/etri_ws/src/fr3_husky_etri/docs/figures/pick_place_cube_area.png
"""

import argparse
import math
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

TABLE_X, TABLE_HALF_Y = (0.40, 1.00), 0.50  # real 600 x 1000 mm, centred on the robot (y = 0)
AREA_X, AREA_Y = (0.45, 0.65), (-0.45, 0.05)
LINK0 = (0.0, -0.215)  # right arm
REACH_RADIUS = 0.62
GOAL_MIN_DIST = 0.10
CUBE = 0.05

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, TABLE_FILL = "#fcfcfb", "#f0efec"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def h(y):
    """Horizontal plot coordinate [cm]: robot's left (+y) on the left."""
    return -100.0 * y


def v(x):
    return 100.0 * x


def reachable(x, y):
    return np.hypot(x - LINK0[0], y - LINK0[1]) <= REACH_RADIUS


def sample(rng, k, start=None):
    out = np.empty((k, 2))
    todo = np.ones(k, bool)
    while todo.any():
        m = int(todo.sum())
        x = rng.uniform(*AREA_X, m)
        y = rng.uniform(*AREA_Y, m)
        ok = reachable(x, y)
        if start is not None:
            ok &= np.hypot(x - start[todo, 0], y - start[todo, 1]) >= GOAL_MIN_DIST
        idx = np.where(todo)[0][ok]
        out[idx, 0], out[idx, 1] = x[ok], y[ok]
        todo[idx] = False
    return out


def area_outline(n=60):
    """Rectangle clipped by the reach circle, as (x, y) points counter-clockwise from the front-right corner."""
    x0, x1 = AREA_X
    y0, y1 = AREA_Y
    pts = [(x0, y0), (x0, y1)]
    x_left = math.sqrt(REACH_RADIUS**2 - (y1 - LINK0[1]) ** 2) + LINK0[0]  # circle meets y = y1
    x_right = math.sqrt(REACH_RADIUS**2 - (y0 - LINK0[1]) ** 2) + LINK0[0]  # circle meets y = y0
    pts.append((min(x_left, x1), y1))
    a_left = math.atan2(y1 - LINK0[1], x_left - LINK0[0])
    a_right = math.atan2(y0 - LINK0[1], x_right - LINK0[0])
    for a in np.linspace(a_left, a_right, n):
        pts.append((LINK0[0] + REACH_RADIUS * math.cos(a), LINK0[1] + REACH_RADIUS * math.sin(a)))
    pts.append((min(x_right, x1), y0))
    return pts, x_left, x_right


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color="#e6e5e0", lw=0.5)
    ax.tick_params(labelsize=8, colors=INK2)
    for sp in ax.spines.values():
        sp.set_color(MUTED)


def draw_table(ax):
    ax.add_patch(Rectangle((h(TABLE_HALF_Y), v(TABLE_X[0])), 200 * TABLE_HALF_Y, 100 * (TABLE_X[1] - TABLE_X[0]),
                           fc=TABLE_FILL, ec=INK2, lw=1.2))


def sample_view(ax, starts, goals):
    draw_table(ax)
    pts, _, _ = area_outline()
    ax.add_patch(Polygon([(h(y), v(x)) for x, y in pts], closed=True, fc="none", ec=BLUE, lw=1.4))
    ax.scatter(h(goals[:, 1]), v(goals[:, 0]), s=3, color=ORANGE, alpha=0.35, lw=0, label="goal (cube centre)")
    ax.scatter(h(starts[:, 1]), v(starts[:, 0]), s=3, color=BLUE, alpha=0.35, lw=0, label="cube start (centre)")
    ax.add_patch(Circle((h(LINK0[1]), v(LINK0[0])), 100 * REACH_RADIUS, fc="none", ec=MUTED, lw=0.9, ls=(0, (5, 3))))
    ax.text(h(-0.47), v(0.66), "right-arm reach\n0.62 m from link0", fontsize=8, color=MUTED, ha="right", va="bottom")
    for y, lab in ((-LINK0[1], "left link0"), (0.0, "base (0, 0)"), (LINK0[1], "right link0")):
        ax.plot(h(y), v(0.0), "s", ms=6, color=INK2)
        ax.text(h(y), v(0.0) - 2.5, lab, fontsize=7.8, color=INK2, ha="center", va="top")
    ax.axvline(h(0.0), color=MUTED, lw=0.8, ls=":")
    ax.text(h(0.0) + 0.5, v(0.96), "robot centre line y = 0", fontsize=7.8, color=MUTED, rotation=90, va="top")
    ax.legend(loc="upper left", fontsize=8, markerscale=4, frameon=False)
    ax.set_xlim(h(0.55), h(-0.62))
    ax.set_ylim(v(-0.08), v(1.03))
    ax.set_aspect("equal")
    ticks_y = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6]
    ax.set_xticks([h(t) for t in ticks_y])
    ax.set_xticklabels([f"{t:.1f}" for t in ticks_y])
    ticks_x = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks([v(t) for t in ticks_x])
    ax.set_yticklabels([f"{t:.1f}" for t in ticks_x])
    ax.set_xlabel("base y [m]   (robot's left = image left)", fontsize=9)
    ax.set_ylabel("base x [m]", fontsize=9)
    ax.set_title(f"Sim sampling, {len(starts)} episodes (start and goal drawn independently, >= 10 cm apart)",
                 fontsize=10, loc="left", color=INK)


def mm_view(ax):
    """Zoom on the area in table coordinates: mm from the table front edge (up) and from the robot centre line."""
    pts, x_left, x_right = area_outline()
    fx = TABLE_X[0]

    def P(x, y):
        return (-1000.0 * y, 1000.0 * (x - fx))  # (mm right of centre line -> image left = +y, mm from front edge)

    ax.add_patch(Rectangle((-1000 * TABLE_HALF_Y, 0), 1000 * 2 * TABLE_HALF_Y, 1000 * (TABLE_X[1] - fx), fc=TABLE_FILL,
                           ec=INK2, lw=1.2))
    poly = [P(x, y) for x, y in pts]
    ax.add_patch(Polygon(poly, closed=True, fc=BLUE, ec=BLUE, lw=1.6, alpha=0.15))
    ax.add_patch(Polygon(poly, closed=True, fc="none", ec=BLUE, lw=1.6))
    # cube footprint at the area corners: the whole cube stays on the table
    ax.add_patch(Rectangle((P(AREA_X[0], AREA_Y[0])[0] - 25, P(AREA_X[0], AREA_Y[0])[1] - 25), 50, 50, fc="none",
                           ec=INK2, lw=0.8, ls="--"))
    ax.text(P(AREA_X[0], AREA_Y[0])[0] - 30, P(AREA_X[0], AREA_Y[0])[1] + 2, "cube footprint\nat a corner", fontsize=7.5,
            color=INK2, va="center", ha="right")
    ax.axvline(0, color=MUTED, lw=0.8, ls=":")

    marks = [
        ("P1", AREA_X[0], AREA_Y[1]),
        ("P2", AREA_X[0], AREA_Y[0]),
        ("P3", x_right, AREA_Y[0]),
        ("P4", x_left, AREA_Y[1]),
        ("P5", LINK0[0] + REACH_RADIUS, LINK0[1]),
    ]
    for x_arc in (0.58, 0.60):
        dy = math.sqrt(REACH_RADIUS**2 - (x_arc - LINK0[0]) ** 2)
        marks.append((f"arc {x_arc:.2f}", x_arc, LINK0[1] - dy))
        marks.append((f"arc {x_arc:.2f}", x_arc, LINK0[1] + dy))
    for name, x, y in marks:
        px, py = P(x, y)
        ax.plot(px, py, "o", ms=4.5, color=INK)
        if name.startswith("P"):
            ax.text(px + (12 if px >= -300 else -12), py + 10, name, fontsize=9, color=INK, fontweight="bold",
                    ha="left" if px >= -300 else "right", va="bottom")

    # dimensions
    def hdim(y_mm, a, b, label):
        ax.annotate("", (a, y_mm), (b, y_mm), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.8))
        ax.text(0.5 * (a + b), y_mm + 6, label, fontsize=8, color=INK2, ha="center", va="bottom")

    def vdim(x_mm, a, b, label):
        ax.annotate("", (x_mm, a), (x_mm, b), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.8))
        ax.text(x_mm - 8, 0.5 * (a + b), label, fontsize=8, color=INK2, ha="right", va="center")

    hdim(-30, -50, 450, "500 mm  (y +0.05 ... -0.45)")
    hdim(-75, 450, 500, "50")
    hdim(-75, -500, -50, "450 mm to the left table edge")
    vdim(-150, 0, 50, "50")
    vdim(-150, 50, 220, "to 220 mm at P5\n(top of the reach arc;\nsides: 161 mm at P4,\n174 mm at P3)")
    ax.text(-480, 585, "table front edge = base x 0.40 (bottom line), table centred on the robot (y = 0)", fontsize=8,
            color=INK2, va="top")

    ax.set_xlim(-520, 640)
    ax.set_ylim(-110, 610)
    ax.set_aspect("equal")
    ax.set_xticks(range(-500, 501, 100))
    ax.set_xticklabels([f"{-t:+d}" if t else "0" for t in range(-500, 501, 100)])
    ax.set_xlabel("mm from the robot centre line (+ = robot's left)", fontsize=9)
    ax.set_ylabel("mm from the table front edge", fontsize=9)
    ax.set_title("Area on the real table (mm)", fontsize=11, loc="left", color=INK)
    return marks


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="pick_place_cube_area.png")
    parser.add_argument("--n", type=int, default=3000)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    starts = sample(rng, args.n)
    goals = sample(rng, args.n, start=starts)
    dist = np.hypot(*(goals - starts).T)

    fig = plt.figure(figsize=(17.0, 9.0), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], height_ratios=[1.0, 0.42], wspace=0.12, hspace=0.22,
                          top=0.92)
    ax_s = fig.add_subplot(gs[:, 0])
    ax_m = fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1, 1])
    style(ax_s)
    style(ax_m)
    sample_view(ax_s, starts, goals)
    marks = mm_view(ax_m)

    ax_t.axis("off")
    rows = [("point", "base x, y [m]", "front edge + [mm]", "centre line [mm]", "right edge + [mm]")]
    for name, x, y in marks:
        rows.append((name, f"{x:.3f}, {y:+.3f}", f"{1000 * (x - TABLE_X[0]):.0f}",
                     f"{1000 * y:+.0f} ({'left' if y > 0 else 'right'})" if abs(y) > 1e-9 else "0",
                     f"{1000 * (y + TABLE_HALF_Y):.0f}"))
    col_x = [0.0, 0.13, 0.34, 0.56, 0.80]
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            ax_t.text(col_x[j], 1.0 - i * 0.105, cell, fontsize=8.5, color=INK, transform=ax_t.transAxes, va="top",
                      fontweight="bold" if i == 0 else "normal")
    ax_t.text(0.0, 1.0 - len(rows) * 0.105 - 0.02,
              f"start-goal distance in sim: median {100 * np.median(dist):.0f} cm, 10-90 % {100 * np.percentile(dist, 10):.0f}"
              f"-{100 * np.percentile(dist, 90):.0f} cm, max {100 * dist.max():.0f} cm.  Cube yaw -45~+45 deg, standing.",
              fontsize=8.5, color=INK2, transform=ax_t.transAxes, va="top")

    fig.suptitle("Cube pick-and-place: where the cube starts and goes in training (cube centre, base frame)", fontsize=13,
                 x=0.05, ha="left", color=INK)
    out = os.path.expanduser(args.out)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {out}")
    for name, x, y in marks:
        print(f"{name}: base ({x:.3f}, {y:+.3f}) front edge +{1000 * (x - TABLE_X[0]):.0f} mm, centre line {1000 * y:+.0f} mm")


if __name__ == "__main__":
    main()
