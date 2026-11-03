"""Box lift workspace figure for real-robot placement: robot root frame + box spawn DR range, front / side / top.

The overlays are projected with each camera's intrinsics and pose and drawn on top of the Isaac Sim render, so the
root frame stays visible through the robot mesh:
  - O: robot root frame (``base``, the arm mounting plate) and its x / y / z axes in R / G / B
  - O': O dropped to the floor
  - orange dashed: box centre spawn range (x, y) on the floor
  - yellow: box bottom footprint envelope over the whole x / y / yaw range
  - cyan: box bottom at the middle of the range, yaw 0 (the rendered box sits there)
  - rulers on the floor (x, y) and a vertical z ruler, all in root-frame coordinates
Writes <out_dir>/<prefix>_{front,side,top}.png and prints the numbers that went into them.

    python scripts/tools/draw_box_workspace.py --viz none \
        --task Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Box410-Play-v0 --out_dir docs/figures --prefix box410_workspace
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", default="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Box410-Play-v0")
parser.add_argument("--out_dir", default="docs/figures")
parser.add_argument("--prefix", default="box410_workspace")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=960)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
os.environ["DR_VIS"] = "0"  # the blue/green DR plates of the base cfg still show the old 555-box range
app = AppLauncher(args).app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors.camera import CameraCfg  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_angle_axis  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import dual_fr3_lab.tasks  # noqa: E402,F401

# (eye, lookat) relative to the env origin [m]; robot root above the origin facing +x, left arm on +y
VIEWS = {
    "front": ((2.6, 0.0, 1.15), (0.30, 0.0, 0.25)),
    "side": ((0.30, -2.7, 0.75), (0.30, 0.0, 0.30)),
    "top": ((0.28, 0.0, 3.0), (0.30, 0.0, 0.0)),
}
LABELS = {"front": "front (looking -x, robot left = image right)", "side": "side (from robot right, looking +y)",
          "top": "top (looking down, +x up, robot left = image left)"}
RED, GREEN, BLUE = (235, 45, 45), (40, 190, 70), (50, 110, 255)
ORANGE, YELLOW, CYAN, WHITE = (255, 140, 0), (255, 220, 0), (0, 215, 230), (245, 245, 245)


def T(x):
    return x.torch if hasattr(x, "torch") else x


cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
cfg.scene.num_envs = 1
cfg.observations.policy.enable_corruption = False
for name in VIEWS:
    setattr(cfg.scene, f"fig_{name}", CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/FigCam_{name}", update_period=0.0,
        width=args.width, height=args.height, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 20.0)),
    ))
size = tuple(cfg.scene.object.spawn.size)
x0, y0 = cfg.scene.object.init_state.pos[:2]
pr = cfg.events.reset_object_position.params["pose_range"]
xr = (x0 + pr["x"][0], x0 + pr["x"][1])
yr = (y0 + pr["y"][0], y0 + pr["y"][1])
yawr = tuple(pr.get("yaw", (0.0, 0.0)))
goal = cfg.commands.object_pose.ranges

env = gym.make(args.task, cfg=cfg)
raw = env.unwrapped
dev = raw.device
origin = raw.scene.env_origins[0]
cams = {k: raw.scene[f"fig_{k}"] for k in VIEWS}
for k, (eye, look) in VIEWS.items():
    cams[k].set_world_poses_from_view(torch.tensor([eye], device=dev) + origin, torch.tensor([look], device=dev) + origin)
env.reset()
raw._disturb_scale = 0.0
robot, box = raw.scene["robot"], raw.scene["object"]
rp, rq = T(robot.data.root_pos_w)[0].clone(), T(robot.data.root_quat_w)[0].clone()
floor_z = float(origin[2] - rp[2])  # floor in the root frame (plane at env z = 0)


def to_world(p):
    p = torch.as_tensor(np.asarray(p, dtype=np.float32), device=dev).reshape(-1, 3)
    return rp + quat_apply(rq.expand(len(p), 4), p)


# the rendered box: middle of the range, yaw 0, resting on the floor
xm, ym = 0.5 * (xr[0] + xr[1]), 0.5 * (yr[0] + yr[1])
pos = to_world([xm, ym, floor_z + size[2] / 2])
box.write_root_pose_to_sim_index(root_pose=torch.cat([pos, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=dev)], -1),
                                 env_ids=torch.tensor([0], device=dev))
box.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev), env_ids=torch.tensor([0], device=dev))
act = torch.zeros(1, raw.action_manager.total_action_dim, device=dev)
for _ in range(12):
    env.step(act)

bc = quat_apply_inverse(rq[None], T(box.data.root_pos_w)[:1] - rp)[0]
# link0 is merged into base in the USD: the arm mounting flange is 0.333 m below joint1 along the link1 z axis
link0 = {}
for s in ("left", "right"):
    b = robot.find_bodies(f"{s}_fr3_link1")[0][0]
    zj = quat_apply(T(robot.data.body_quat_w)[:1, b], torch.tensor([[0.0, 0.0, 1.0]], device=dev))
    link0[s] = quat_apply_inverse(rq[None], T(robot.data.body_pos_w)[:1, b] - 0.333 * zj - rp)[0]
stage = raw.sim.stage
bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
    stage.GetPrimAtPath("/World/envs/env_0/Robot/base")).ComputeAlignedRange()
bmin, bmax = np.array(bb.GetMin()) - rp.cpu().numpy(), np.array(bb.GetMax()) - rp.cpu().numpy()


def footprint(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy = size[0] / 2, size[1] / 2
    return [(x + c * px - s * py, y + s * px + c * py) for px, py in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))]


def hull(pts):
    pts = sorted(set(pts))
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and (lower[-1][0] - lower[-2][0]) * (p[1] - lower[-2][1]) - (lower[-1][1] - lower[-2][1]) * (p[0] - lower[-2][0]) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and (upper[-1][0] - upper[-2][0]) * (p[1] - upper[-2][1]) - (upper[-1][1] - upper[-2][1]) * (p[0] - upper[-2][0]) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


corners = [c for x in xr for y in yr for yaw in np.linspace(*yawr, 11) for c in footprint(x, y, yaw)]
env_hull = hull([(round(a, 5), round(b, 5)) for a, b in corners])
ex = (min(p[0] for p in corners), max(p[0] for p in corners))
ey = (min(p[1] for p in corners), max(p[1] for p in corners))

info = [
    f"root frame O = arm mounting plate ('base'), floor at z = {floor_z:+.3f} m",
    f"box {size[0]:.3f} (x) x {size[1]:.3f} (y, grip) x {size[2]:.3f} (z) m",
    f"box centre: x {xr[0]:.3f}~{xr[1]:.3f}, y {yr[0]:+.3f}~{yr[1]:+.3f}, yaw {math.degrees(yawr[0]):+.1f}~{math.degrees(yawr[1]):+.1f} deg",
    f"box bottom z {floor_z:+.3f} (on the floor), centre z {floor_z + size[2] / 2:+.4f}",
    f"bottom footprint envelope: x {ex[0]:.3f}~{ex[1]:.3f}, y {ey[0]:+.3f}~{ey[1]:+.3f}",
    f"goal (box centre): same x/y, z {goal.pos_z[0]:+.2f}~{goal.pos_z[1]:+.2f}",
    f"base front x {bmax[0]:+.3f} -> nearest box face {ex[0] - bmax[0]:.3f} m ahead (worst case)",
]
print("[ws] " + "\n[ws] ".join(info))
print(f"[ws] rendered box centre (root frame, after settling): {bc[0]:.4f} {bc[1]:+.4f} {bc[2]:+.4f}")
print(f"[ws] arm mounting flange (link0 origin) L {link0['left'].cpu().numpy().round(4)}  R {link0['right'].cpu().numpy().round(4)} (root frame)")
print(f"[ws] base body bbox (root frame) x {bmin[0]:+.3f}~{bmax[0]:+.3f}  y {bmin[1]:+.3f}~{bmax[1]:+.3f}  z {bmin[2]:+.3f}~{bmax[2]:+.3f}")
print(f"[ws] gap base front (x {bmax[0]:.3f}) -> nearest box face: {ex[0] - bmax[0]:.3f} m (worst case), "
      f"{xr[0] - size[0] / 2 - bmax[0]:.3f} m at yaw 0, x min")

font = ImageFont.load_default(size=22)
small = ImageFont.load_default(size=18)


def project(cam, pts_root):
    pw = to_world(pts_root)
    K = T(cam.data.intrinsic_matrices)[0]
    pc = quat_apply_inverse(T(cam.data.quat_w_ros)[0].expand(len(pw), 4), pw - T(cam.data.pos_w)[0])
    uv = (K @ pc.T).T
    return (uv[:, :2] / uv[:, 2:3]).cpu().numpy()


def text(draw, xy, s, color, f=font):
    x, y = xy
    l, t, r, b = draw.textbbox((x, y), s, font=f)
    draw.rectangle((l - 4, t - 3, r + 4, b + 3), fill=(15, 15, 15))
    draw.text((x, y), s, fill=color, font=f)


def line(draw, pts, color, width=3, dash=0):
    for a, b in zip(pts[:-1], pts[1:]):
        if not dash:
            draw.line([tuple(a), tuple(b)], fill=color, width=width)
            continue
        d = np.linalg.norm(b - a)
        n = max(int(d / dash), 1)
        for i in range(0, n, 2):
            p, q = a + (b - a) * i / n, a + (b - a) * min(i + 1, n) / n
            draw.line([tuple(p), tuple(q)], fill=color, width=width)


def poly3(z, xy):
    return [(x, y, z) for x, y in list(xy) + [xy[0]]]


def arrow(draw, a, b, color, width=5):
    draw.line([tuple(a), tuple(b)], fill=color, width=width)
    d = b - a
    n = np.linalg.norm(d)
    if n < 1:
        return
    d /= n
    side = np.array([-d[1], d[0]])
    head = 20
    draw.polygon([tuple(b), tuple(b - head * d + 0.45 * head * side), tuple(b - head * d - 0.45 * head * side)], fill=color)


def ruler(draw, cam, p0, p1, ticks, fmt, color=WHITE, off=(8, -10), label_every=1):
    a, b = project(cam, [p0, p1])
    line(draw, [a, b], color, width=2)
    for i, (t, p) in enumerate(ticks):
        u = project(cam, [p])[0]
        draw.ellipse((u[0] - 4, u[1] - 4, u[0] + 4, u[1] + 4), fill=color)
        if i % label_every == 0:
            text(draw, (u[0] + off[0], u[1] + off[1]), fmt.format(t), color, small)


os.makedirs(args.out_dir, exist_ok=True)
L = 0.25
for k, cam in cams.items():
    rgb = T(cam.data.output["rgb"])[0, ..., :3].cpu().numpy().astype(np.uint8)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    # floor geometry
    line(draw, project(cam, poly3(floor_z, env_hull)), YELLOW, width=3)
    rect = [(xr[0], yr[0]), (xr[1], yr[0]), (xr[1], yr[1]), (xr[0], yr[1])]
    line(draw, project(cam, poly3(floor_z, rect)), ORANGE, width=3, dash=14)
    line(draw, project(cam, poly3(floor_z, footprint(xm, ym, 0.0))), CYAN, width=2)
    for x, y in rect:
        u = project(cam, [(x, y, floor_z)])[0]
        draw.ellipse((u[0] - 5, u[1] - 5, u[0] + 5, u[1] + 5), fill=ORANGE)
    # rulers (root-frame coordinates): tick every 0.1 m, label every 0.2 m
    if k in ("side", "top"):
        yy = ey[0] - 0.12
        ruler(draw, cam, (0.0, yy, floor_z), (0.8, yy, floor_z),
              [(x / 10, (x / 10, yy, floor_z)) for x in range(0, 9)], "x {:.1f}",
              off=(10, -10) if k == "top" else (-22, 10), label_every=2)
    if k in ("front", "top"):
        xx = ex[1] + 0.12
        ruler(draw, cam, (xx, -0.4, floor_z), (xx, 0.4, floor_z),
              [(y / 10, (xx, y / 10, floor_z)) for y in range(-4, 5)], "y {:+.1f}",
              off=(-26, 10) if k == "front" else (-26, -36), label_every=2)
    if k in ("front", "side"):
        zx, zy = (ex[1] + 0.12, ey[1] + 0.12) if k == "front" else (ex[1] + 0.12, ey[0] - 0.12)
        ruler(draw, cam, (zx, zy, floor_z), (zx, zy, 0.3),
              [(z, (zx, zy, z)) for z in (floor_z, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)], "z {:+.3f}")
    # range corners with their coordinates (box centre)
    for (x, y), dy in zip(rect, (1, 1, -1, -1)):
        if k == "top":
            u = project(cam, [(x, y, floor_z)])[0]
            text(draw, (u[0] + (10 if y < 0 else -150), u[1] - 12 + (14 if x == xr[0] else -14)), f"({x:.2f}, {y:+.2f})", ORANGE, small)
    if k == "side":
        u = project(cam, [(xr[1], 0.0, floor_z)])[0]
        text(draw, (u[0] + 14, u[1] - 40), f"centre x {xr[0]:.2f}~{xr[1]:.2f}", ORANGE, small)
    if k == "front":
        u = project(cam, [(xr[0], yr[1], floor_z)])[0]
        text(draw, (u[0] + 14, u[1] - 4), f"centre y {yr[0]:+.2f}~{yr[1]:+.2f}", ORANGE, small)
    # root frame
    o, ox, oy, oz = project(cam, [(0, 0, 0), (L, 0, 0), (0, L, 0), (0, 0, L)])
    of = project(cam, [(0, 0, floor_z)])[0]
    line(draw, [o, of], WHITE, width=2, dash=10)
    draw.ellipse((of[0] - 6, of[1] - 6, of[0] + 6, of[1] + 6), outline=WHITE, width=3)
    if k != "top":  # O and O' coincide from above
        text(draw, (of[0] + 10, of[1] + 4), "O' (O on the floor)", WHITE, small)
    for tip, c, s in ((ox, RED, "x"), (oy, GREEN, "y"), (oz, BLUE, "z")):
        if np.linalg.norm(tip - o) > 12:
            arrow(draw, o, tip, c)
            text(draw, (tip[0] + 6, tip[1] - 14), s, c)
    draw.ellipse((o[0] - 7, o[1] - 7, o[0] + 7, o[1] + 7), fill=WHITE, outline=(0, 0, 0), width=2)
    text(draw, (o[0] - 60, o[1] - 52), "O (root)", WHITE)
    # legend + numbers
    lines = [(LABELS[k], WHITE)] + [(s, WHITE) for s in info] + [
        ("orange dashed: box centre range", ORANGE), ("yellow: box bottom envelope (x, y, yaw)", YELLOW),
        ("cyan: box at range middle, yaw 0", CYAN), ("x / y / z axes: R / G / B, 0.25 m", WHITE)]
    y, x = 10, 12
    if k == "top":  # top: the floor at the bottom right is empty
        w = max(draw.textbbox((0, 0), s, font=small)[2] for s, _ in lines)
        y, x = args.height - 26 * len(lines) - 8, args.width - w - 14
    for s, c in lines:
        text(draw, (x, y), s, c, small)
        y += 26
    path = os.path.join(args.out_dir, f"{args.prefix}_{k}.png")
    img.save(path)
    print(f"[ws] wrote {path}")

env.close()
app.close()
