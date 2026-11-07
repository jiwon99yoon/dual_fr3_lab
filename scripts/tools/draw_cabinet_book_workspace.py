"""Cabinet book workspace figure for real-robot placement: robot root frame, table, cabinet/door sweep, book ranges.

Like ``draw_box_workspace.py``: the overlays are projected with each camera's intrinsics and pose and drawn on top of
the Isaac Sim render (robot at the start pose of ``--row``, door closed), with a legend panel on the right. Top and
side are taken from far away with a long lens (nearly orthographic, so the rulers hold at any depth); back is a
perspective overview from behind the robot.
  - O: robot root frame (``base``, the arm mounting plate) and its x / y / z axes in R / G / B; L0 / R0: link0 origins
  - white: table top; brown: cabinet (outer box) and the opening (inner front, dotted)
  - door: closed (brown), 90 deg (orange), joint limit 120 deg (orange dashed) and the free edge's arc (dotted)
  - magenta: handle bar axis (closed / 90 / 120 deg)
  - cyan dashed: book centre start range (start table rows), cyan: book footprint over it, blue: book at row ``--row``
  - green: book at the insert goal, green dashed: pre-insert
  - red dashed: DoorFirst wait line (book centre y), drawn when the task has ``book_wait``
Writes <out_dir>/<prefix>_{top,side,back}.png and prints the numbers that went into them.

    python scripts/tools/draw_cabinet_book_workspace.py --viz none \\
        --task Isaac-Cabinet-BookInsert-DualFR3-UMI-Table75-DoorFirst-Play-v0 --out_dir docs/figures --prefix cabinet_book_75cm_workspace
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", default="Isaac-Cabinet-BookInsert-DualFR3-UMI-Table75-DoorFirst-Play-v0")
parser.add_argument("--row", type=int, default=60, help="start-table row the robot and book are rendered at")
parser.add_argument("--out_dir", default="docs/figures")
parser.add_argument("--prefix", default="cabinet_book_75cm_workspace")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=960)
parser.add_argument("--panel", type=int, default=620, help="legend panel width (px)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors.camera import CameraCfg  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import dual_fr3_lab.tasks  # noqa: E402,F401
from dual_fr3_lab.tasks.manager_based.cabinet_book_insert import cabinet_book_insert_env_cfg as C  # noqa: E402

RED, GREEN, BLUE = (235, 45, 45), (40, 200, 80), (60, 120, 255)
ORANGE, CYAN, WHITE, GREY = (255, 150, 30), (0, 215, 230), (245, 245, 245), (170, 170, 170)
BROWN, MAGENTA, YELLOW = (215, 140, 70), (235, 70, 235), (255, 220, 0)
DOOR_LIMIT_DEG = 120.0  # MJCF drawer_joint range -2.0944..0
TCP_OFFSET = 0.2844  # UMI TCP along link7 z


def T(x):
    return x.torch if hasattr(x, "torch") else x


cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
cfg.scene.num_envs = 1
cfg.observations.policy.enable_corruption = False
cfg.events.reset_start.params["fixed_row"] = args.row
# base-frame layout (the scene cfg is in env coordinates, base frame = env + (0, 0, ROBOT_BASE_Z))
tz = cfg.scene.table.init_state.pos[2] + 0.5 * cfg.scene.table.spawn.size[2] - C.ROBOT_BASE_Z
tcx = cfg.scene.table.init_state.pos[0]
tdx, tdy = cfg.scene.table.spawn.size[:2]
cx, cy, cz = cfg.scene.cabinet.init_state.pos
cz -= C.ROBOT_BASE_Z
look_z = tz + C.ROBOT_BASE_Z  # table top in env coordinates
# (eye, lookat) relative to the env origin [m], focal length [mm]. top / side: far away with a long lens, close to an
# orthographic view so the rulers line up with objects at any depth; back: perspective overview behind the robot
VIEWS = {
    "top": ((0.48, 0.0, look_z + 8.0), (0.50, 0.0, look_z), 66.0),
    "side": ((0.55, -8.0, 0.60), (0.55, 0.0, 0.60), 62.0),
    "back": ((-1.3, -0.5, look_z + 1.75), (0.62, -0.05, look_z), 24.0),
}
LABELS = {"top": "top (looking down, +x up, robot left = image left)",
          "side": "side (from robot right, looking +y)",
          "back": "behind the robot (looking +x, robot left = image left)"}
for name, (eye, look, focal) in VIEWS.items():
    setattr(cfg.scene, f"fig_{name}", CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/FigCam_{name}", update_period=0.0,
        width=args.width, height=args.height, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=focal, clipping_range=(0.01, 30.0)),
    ))

rew = cfg.rewards
pre = tuple(rew.book_approach.params["pre_pos_b"])
goal = tuple(rew.insert_success.params["goal_pos_b"])
opening_x = rew.insert_success.params["opening_x_b"]
wait = getattr(rew, "book_wait", None)
y_wait = wait.params["y_wait"] if wait is not None else None
starts = np.load(cfg.events.reset_start.params["table_path"])["book_start_b"]  # [rows, 3] base frame
xr = (float(starts[:, 0].min()), float(starts[:, 0].max()))
yr = (float(starts[:, 1].min()), float(starts[:, 1].max()))
zb = float(starts[:, 2].mean())
bs = C.BOOK_SIZE

env = gym.make(args.task, cfg=cfg)
raw = env.unwrapped
dev = raw.device
origin = raw.scene.env_origins[0]
cams = {k: raw.scene[f"fig_{k}"] for k in VIEWS}
for k, (eye, look, _) in VIEWS.items():
    cams[k].set_world_poses_from_view(torch.tensor([eye], device=dev) + origin, torch.tensor([look], device=dev) + origin)
env.reset()
act = torch.zeros(1, raw.action_manager.total_action_dim, device=dev)
for _ in range(4):
    env.step(act)
robot = raw.scene["robot"]
rp, rq = T(robot.data.root_pos_w)[0].clone(), T(robot.data.root_quat_w)[0].clone()
floor_z = float(origin[2] - rp[2])


def to_world(p):
    p = torch.as_tensor(np.asarray(p, dtype=np.float32), device=dev).reshape(-1, 3)
    return rp + quat_apply(rq.expand(len(p), 4), p)


def to_base(pw):
    return quat_apply_inverse(rq[None], pw[None] - rp)[0].cpu().numpy()


# link0 is merged into base in the USD: the arm mounting flange is 0.333 m below joint1 along the link1 z axis
link0 = {}
for s in ("left", "right"):
    b = robot.find_bodies(f"{s}_fr3_link1")[0][0]
    zj = quat_apply(T(robot.data.body_quat_w)[:1, b], torch.tensor([[0.0, 0.0, 1.0]], device=dev))
    link0[s] = to_base(T(robot.data.body_pos_w)[0, b] - 0.333 * zj[0])
b7 = robot.find_bodies("left_fr3_link7")[0][0]
z7 = quat_apply(T(robot.data.body_quat_w)[:1, b7], torch.tensor([[0.0, 0.0, 1.0]], device=dev))
left_tcp = to_base(T(robot.data.body_pos_w)[0, b7] + TCP_OFFSET * z7[0])
book_now = to_base(T(raw.scene["book"].data.root_pos_w)[0])


# ---- cabinet geometry (MJCF, cabinet frame: root on the table top, yawed 180 deg so local (x, y) -> base (-x, -y))
def cab(p):
    return (cx - p[0], cy - p[1], cz + p[2])


DOOR_X, HINGE_Y, DOOR_HALF, DOOR_ZC = C.DOOR_IN_CABINET[0], C.DOOR_HINGE_IN_DOOR[1], 0.1725, C.DOOR_IN_CABINET[2]
BAR_X = DOOR_X + C.HANDLE_IN_DOOR[0] + C.HANDLE_BAR_CENTER_IN_HANDLE[0]
BAR_Y, BAR_HALF = C.HANDLE_IN_DOOR[1], 0.1325
hinge = np.array(cab((DOOR_X, HINGE_Y, 0.0)))


def door_rot(p, deg):
    """Rotate a base-frame point about the hinge axis: the door opens clockwise seen from above (joint goes negative)."""
    a = -math.radians(deg)
    dx, dy = p[0] - hinge[0], p[1] - hinge[1]
    return (hinge[0] + math.cos(a) * dx - math.sin(a) * dy, hinge[1] + math.sin(a) * dx + math.cos(a) * dy, p[2])


def door_rect(deg):
    pts = [cab((DOOR_X, y, DOOR_ZC + z)) for y, z in ((-DOOR_HALF, -DOOR_HALF), (DOOR_HALF, -DOOR_HALF),
                                                      (DOOR_HALF, DOOR_HALF), (-DOOR_HALF, DOOR_HALF))]
    return [door_rot(p, deg) for p in pts]


def bar(deg, dz=0.0):
    return door_rot(cab((BAR_X, BAR_Y, DOOR_ZC + dz)), deg)


free_edge0 = cab((DOOR_X, DOOR_HALF, DOOR_ZC - DOOR_HALF))
arc = [door_rot(free_edge0, d) for d in np.linspace(0.0, DOOR_LIMIT_DEG, 49)]
bar_arc = [bar(d) for d in np.linspace(0.0, DOOR_LIMIT_DEG, 49)]
sweep_x = min(p[0] for p in arc + bar_arc)
H = C.CABINET_OUTER_HALF
cab_lo = (cx - H, cy - H, cz)
cab_hi = (cx + H, cy + H, cz + 2 * H)
inner = C.CABINET_INNER_HALF


def box_edges(c, size):
    hx, hy, hz = (0.5 * s for s in size)
    v = [(c[0] + sx * hx, c[1] + sy * hy, c[2] + sz * hz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    return [(v[i], v[j]) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]


def rect(z, xa, xb, ya, yb):
    return [(xa, ya, z), (xb, ya, z), (xb, yb, z), (xa, yb, z), (xa, ya, z)]


info = [
    f"O = base (arm mounting plate, between the link0s); sim floor z {floor_z:+.3f}",
    f"table top z {tz:+.3f} = {tz - floor_z:.3f} m above the sim floor (real mount ~0.38 m up -> {tz + 0.38:.3f} m)",
    f"table top x {tcx - tdx / 2:.2f}~{tcx + tdx / 2:.2f}, y {-tdy / 2:+.2f}~{tdy / 2:+.2f}",
    f"cabinet {2 * H:.2f} m cube, bottom centre ({cx:.3f}, {cy:+.3f}, {cz:+.3f}), yaw 180 (door faces robot)",
    f"cabinet front face x {cx - H:.3f}, opening (inner front) x {opening_x:.3f}, y {cy - inner:+.3f}~{cy + inner:+.3f}",
    f"hinge axis ({hinge[0]:.4f}, {hinge[1]:+.4f}), z {cz:+.3f}~{cz + 2 * H:+.3f}",
    "handle bar axis (mid height z {:+.3f}): closed ({:.3f}, {:+.3f}), 90 ({:.3f}, {:+.3f}), 120 ({:.3f}, {:+.3f})".format(
        cz + DOOR_ZC, *bar(0)[:2], *bar(90)[:2], *bar(DOOR_LIMIT_DEG)[:2]),
    f"door 0~{DOOR_LIMIT_DEG:.0f} deg (joint limit): door/handle reach x {sweep_x:.3f}",
    f"book {bs[0]:.2f} (x) x {bs[1]:.2f} (y, grip) x {bs[2]:.2f} (z) m, standing, faces along the base axes",
    f"book centre start: x {xr[0]:.2f}~{xr[1]:.2f}, y {yr[0]:+.2f}~{yr[1]:+.2f}, z {zb:+.3f} ({len(starts)} rows)",
    f"book footprint: x {xr[0] - bs[0] / 2:.3f}~{xr[1] + bs[0] / 2:.3f}, y {yr[0] - bs[1] / 2:+.3f}~{yr[1] + bs[1] / 2:+.3f}",
    f"pre-insert ({pre[0]:.3f}, {pre[1]:+.3f}, {pre[2]:+.3f}), goal ({goal[0]:.3f}, {goal[1]:+.3f}, {goal[2]:+.3f})"
    f" = {100 * (goal[0] - opening_x):.0f} cm past the opening",
]
if y_wait is not None:
    info.append(f"DoorFirst: book centre stays at y < {y_wait:+.3f} until the door is open (80~90 deg)")
legend = [
    ("x / y / z axes R / G / B (0.25 m); L0 / R0 = link0", WHITE), ("white: table top", WHITE),
    ("brown: cabinet, dotted = opening", BROWN), ("door: closed / 90 deg / 120 deg limit (dashed)", ORANGE),
    ("dotted arc: door free edge 0 -> 120 deg", ORANGE), ("magenta: handle bar axis", MAGENTA),
    ("cyan dashed: book centre start range", CYAN), ("cyan: book footprint over that range", CYAN),
    (f"blue: book at start row {args.row}", BLUE), ("green: book at goal, dashed: pre-insert", GREEN),
]
if y_wait is not None:
    legend.append(("red dashed: DoorFirst wait line (book centre y)", RED))
print("[ws] " + "\n[ws] ".join(info))
print(f"[ws] rendered row {args.row}: book centre {book_now.round(4)}, left TCP {left_tcp.round(4)} (FK node "
      f"left_tcp_closed_position_base), hinge {hinge.round(4)} (hinge_position_base, z = cabinet mid height "
      f"{cz + DOOR_ZC:+.3f})")
print(f"[ws] link0 L {link0['left'].round(4)} R {link0['right'].round(4)}")

font = ImageFont.load_default(size=22)
small = ImageFont.load_default(size=17)


def project(cam, pts):
    pw = to_world(pts)
    K = T(cam.data.intrinsic_matrices)[0]
    pc = quat_apply_inverse(T(cam.data.quat_w_ros)[0].expand(len(pw), 4), pw - T(cam.data.pos_w)[0])
    uv = (K @ pc.T).T
    return (uv[:, :2] / uv[:, 2:3]).cpu().numpy()


def text(draw, xy, s, color, f=small):
    x, y = xy
    l, t, r, b = draw.textbbox((x, y), s, font=f)
    draw.rectangle((l - 3, t - 2, r + 3, b + 2), fill=(15, 15, 15))
    draw.text((x, y), s, fill=color, font=f)


def line(draw, pts, color, width=3, dash=0):
    for a, b in zip(pts[:-1], pts[1:]):
        if not dash:
            draw.line([tuple(a), tuple(b)], fill=color, width=width)
            continue
        n = max(int(np.linalg.norm(b - a) / dash), 1)
        for i in range(0, n, 2):
            draw.line([tuple(a + (b - a) * i / n), tuple(a + (b - a) * min(i + 1, n) / n)], fill=color, width=width)


def poly(draw, cam, pts, color, width=3, dash=0, close=False):
    pts = list(pts) + ([pts[0]] if close else [])
    line(draw, project(cam, pts), color, width, dash)


def edges(draw, cam, segs, color, width=2, dash=0):
    for a, b in segs:
        line(draw, project(cam, [a, b]), color, width, dash)


def dot(draw, cam, p, color, r=6):
    u = project(cam, [p])[0]
    draw.ellipse((u[0] - r, u[1] - r, u[0] + r, u[1] + r), fill=color, outline=(0, 0, 0), width=2)
    return u


def label(draw, cam, p, s, color, off=(10, -10)):
    u = project(cam, [p])[0]
    text(draw, (u[0] + off[0], u[1] + off[1]), s, color)


def arrow(draw, a, b, color, width=5):
    draw.line([tuple(a), tuple(b)], fill=color, width=width)
    d = b - a
    n = np.linalg.norm(d)
    if n < 1:
        return
    d /= n
    side = np.array([-d[1], d[0]])
    head = 18
    draw.polygon([tuple(b), tuple(b - head * d + 0.45 * head * side), tuple(b - head * d - 0.45 * head * side)], fill=color)


def wrap(draw, s, width):
    rows, cur = [], ""
    for w in s.split(" "):
        nxt = f"{cur} {w}" if cur else w
        if cur and draw.textbbox((0, 0), nxt, font=small)[2] > width:
            rows.append(cur)
            nxt = "    " + w
        cur = nxt
    return rows + [cur]


def ruler(draw, cam, ticks, fmt, off, label_every=1, color=GREY):
    pts = [p for _, p in ticks]
    line(draw, project(cam, [pts[0], pts[-1]]), color, width=2)
    for i, (t, p) in enumerate(ticks):
        u = project(cam, [p])[0]
        draw.ellipse((u[0] - 4, u[1] - 4, u[0] + 4, u[1] + 4), fill=color)
        if i % label_every == 0:
            text(draw, (u[0] + off[0], u[1] + off[1]), fmt.format(t), color)


os.makedirs(args.out_dir, exist_ok=True)
L = 0.25
tx0, tx1 = tcx - tdx / 2, tcx + tdx / 2
for k, cam in cams.items():
    rgb = T(cam.data.output["rgb"])[0, ..., :3].cpu().numpy().astype(np.uint8)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    # table top, cabinet, opening
    poly(draw, cam, rect(tz, tx0, tx1, -tdy / 2, tdy / 2), WHITE, width=2)
    edges(draw, cam, box_edges(((cab_lo[0] + cab_hi[0]) / 2, cy, cz + H), (2 * H, 2 * H, 2 * H)), BROWN, width=3)
    poly(draw, cam, [(opening_x, cy - inner, cz + H - inner), (opening_x, cy + inner, cz + H - inner),
                     (opening_x, cy + inner, cz + H + inner), (opening_x, cy - inner, cz + H + inner)],
         BROWN, width=2, dash=8, close=True)
    # door: closed, 90, limit; free edge arc at the bottom; handle bar
    poly(draw, cam, door_rect(0.0), BROWN, width=5, close=True)
    poly(draw, cam, door_rect(90.0), ORANGE, width=4, close=True)
    poly(draw, cam, door_rect(DOOR_LIMIT_DEG), ORANGE, width=3, dash=12, close=True)
    poly(draw, cam, arc, ORANGE, width=2, dash=6)
    poly(draw, cam, bar_arc, MAGENTA, width=2, dash=6)
    for d in (0.0, 90.0, DOOR_LIMIT_DEG):
        poly(draw, cam, [bar(d, -BAR_HALF), bar(d, BAR_HALF)], MAGENTA, width=5)
    poly(draw, cam, [(hinge[0], hinge[1], cz), (hinge[0], hinge[1], cz + 2 * H)], YELLOW, width=3)
    # book: start range (centre, at the start height), footprint, rendered row, pre-insert, goal
    poly(draw, cam, rect(zb, xr[0], xr[1], yr[0], yr[1]), CYAN, width=3, dash=12)
    poly(draw, cam, rect(tz + 0.002, xr[0] - bs[0] / 2, xr[1] + bs[0] / 2, yr[0] - bs[1] / 2, yr[1] + bs[1] / 2), CYAN,
         width=2)
    edges(draw, cam, box_edges(book_now, bs), BLUE, width=3)
    edges(draw, cam, box_edges(pre, bs), GREEN, width=2, dash=8)
    edges(draw, cam, box_edges(goal, bs), GREEN, width=3)
    if y_wait is not None:
        poly(draw, cam, [(tx0, y_wait, pre[2]), (cx - H, y_wait, pre[2])], RED, width=3, dash=14)
    # rulers (base-frame coordinates, top / side): tick every 0.1 m
    if k == "top":
        ruler(draw, cam, [(x / 10, (x / 10, -tdy / 2 - 0.08, tz)) for x in range(-3, 14)], "x {:.1f}",
              off=(10, -10), label_every=2)
        ruler(draw, cam, [(y / 10, (tx1 + 0.08, y / 10, tz)) for y in range(-6, 7)], "y {:+.1f}",
              off=(-24, -34), label_every=2)
    if k == "side":
        ruler(draw, cam, [(x / 10, (x / 10, 0.0, floor_z)) for x in range(-3, 14)], "x {:.1f}",
              off=(-20, 10), label_every=2)
        zx = tx1 + 0.12
        zs = [(floor_z, "floor"), (0.0, "base O"), (tz, "table top"), (zb, "book centre (start)"),
              (pre[2], "lane, handle mid"), (cz + 2 * H, "cabinet top")]
        ruler(draw, cam, [(z, (zx, 0.0, z)) for z, _ in zs], "", off=(0, 0))
        for z, s in zs:
            u = project(cam, [(zx, 0.0, z)])[0]
            text(draw, (u[0] + 10, u[1] - 10), f"z {z:+.3f} {s}", GREY)
        for x, s, dy in ((tx0, "table front", 40), (cx - H, "cabinet front", 64), (sweep_x, "door reach", 88)):
            u = project(cam, [(x, 0.0, tz)])[0]
            line(draw, [u, u + np.array([0.0, dy - 8])], GREY, width=1)
            text(draw, (u[0] - 40, u[1] + dy - 6), f"{s} x {x:.3f}", GREY)
    # labels (top / back: x / y; side: the heights are on the z ruler)
    if k in ("top", "back"):
        label(draw, cam, (hinge[0], hinge[1], cz + 2 * H), f"hinge ({hinge[0]:.3f}, {hinge[1]:+.3f})", YELLOW, (10, -28))
        for d, off in ((0.0, (12, -4)), (90.0, (-200, -8)), (DOOR_LIMIT_DEG, (-210, 10)))[k == "back":]:
            p = bar(d, BAR_HALF)
            label(draw, cam, p, f"handle {d:.0f}: ({p[0]:.3f}, {p[1]:+.3f})", MAGENTA, off)
        label(draw, cam, (xr[1], yr[0], zb), f"({xr[1]:.2f}, {yr[0]:+.2f})", CYAN, (8, -24))
        label(draw, cam, (xr[0], yr[0], zb), f"({xr[0]:.2f}, {yr[0]:+.2f})", CYAN, (8, 4))
        label(draw, cam, (xr[1], yr[1], zb), f"({xr[1]:.2f}, {yr[1]:+.2f})", CYAN, (-130, -24))
        if k == "top":  # crowded in the perspective view
            label(draw, cam, pre, f"pre ({pre[0]:.2f}, {pre[1]:+.2f})", GREEN, (14, 10))
            label(draw, cam, goal, f"goal ({goal[0]:.3f}, {goal[1]:+.2f})", GREEN, (14, -10))
            label(draw, cam, (cx - H, cy - H, cz + 2 * H), f"front corner ({cx - H:.3f}, {cy - H:+.3f})", BROWN, (10, -4))
            label(draw, cam, (cx, cy, cz + 2 * H), f"cabinet ({cx:.2f}, {cy:+.2f})", BROWN, (-60, -10))
        if y_wait is not None:
            label(draw, cam, (tx0, y_wait, pre[2]), f"wait y {y_wait:+.2f}", RED, (6, 8))
        label(draw, cam, (tx0, tdy / 2, tz), f"table ({tx0:.2f}, {tdy / 2:+.2f})", WHITE, (6, -30))
        label(draw, cam, (tx0, -tdy / 2, tz), f"({tx0:.2f}, {-tdy / 2:+.2f})", WHITE, (6, -30))
    if k == "side":
        label(draw, cam, goal, "goal", GREEN, (10, -36))
        label(draw, cam, book_now, f"book row {args.row}", BLUE, (-60, 60))
        label(draw, cam, bar(0.0, BAR_HALF), "handle", MAGENTA, (-30, -30))
    # root frame and link0s
    o, ox, oy, oz = project(cam, [(0, 0, 0), (L, 0, 0), (0, L, 0), (0, 0, L)])
    of = project(cam, [(0, 0, floor_z)])[0]
    line(draw, [o, of], WHITE, width=2, dash=10)
    for tip, c, s in ((ox, RED, "x"), (oy, GREEN, "y"), (oz, BLUE, "z")):
        if np.linalg.norm(tip - o) > 12:
            arrow(draw, o, tip, c)
            text(draw, (tip[0] + 6, tip[1] - 14), s, c, font)
    draw.ellipse((o[0] - 7, o[1] - 7, o[0] + 7, o[1] + 7), fill=WHITE, outline=(0, 0, 0), width=2)
    text(draw, (o[0] - 70, o[1] - 50), "O (base)", WHITE, font)
    for s, tag in (("left", "L0"), ("right", "R0")):
        p = link0[s]
        yaw = math.radians(30.0 if s == "left" else -30.0)
        a, b = project(cam, [p, (p[0] + 0.12 * math.cos(yaw), p[1] + 0.12 * math.sin(yaw), p[2])])
        if np.linalg.norm(b - a) > 12:
            arrow(draw, a, b, RED, width=3)
        draw.ellipse((a[0] - 5, a[1] - 5, a[0] + 5, a[1] + 5), fill=WHITE)
        text(draw, (a[0] + (-40 if s == "left" else 10), a[1] + 8), tag, WHITE)
    # legend panel
    out = Image.new("RGB", (args.width + args.panel, args.height), (20, 20, 20))
    out.paste(img, (0, 0))
    d2 = ImageDraw.Draw(out)
    x, y = args.width + 14, 14
    d2.text((x, y), LABELS[k], fill=WHITE, font=font)
    y += 38
    for s in info:
        for row in wrap(d2, s, args.panel - 28):
            d2.text((x, y), row, fill=WHITE, font=small)
            y += 22
        y += 4
    y += 10
    for s, c in legend:
        d2.text((x, y), s, fill=c, font=small)
        y += 23
    path = os.path.join(args.out_dir, f"{args.prefix}_{k}.png")
    out.save(path)
    print(f"[ws] wrote {path}")

env.close()
app.close()
