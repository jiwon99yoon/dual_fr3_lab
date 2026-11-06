"""Solve the cabinet book-insert start poses (and sweep cabinet placements) with path-aware batched IK.

Kinematic check on the robot alone (no cabinet/book in the scene; their frames come from the env cfg constants).
Paths follow the rule-based order: the left arm opens the door 0 -> ``--door_deg`` (handle grasp frame rotated about
the hinge) while the right arm lifts the book and moves it to the pre-insert x (``book_path_points`` segments 1-2);
with the door open, the right arm slides the book to the lane and inserts it (segments 3-4), the left arm holding.
  stage 1  left arm: every env a different seed (structured + random) x finger orientation, right arm parked at
           its ready pose. Best left = converges at every waypoint, collision free, largest joint margin on the path.
  stage 2  right arm: every env a different seed x orientation, left arm moving along the stage-1 winner, book
           starting at the centre of the start region (``BOOK_START_POS_B``). Best right likewise (both arms OK).
  stage 3  table: one env per book start on a ``--grid`` x ``--grid`` grid over ``BOOK_START_REGION_X/Y`` and per
           right candidate (the ``--right_candidates`` best of stage 2). The candidate's arm walks the book along the
           table from the centre to that start (2 cm steps), then both arms follow the paths, the right null space
           pulled toward the candidate's stage-2 joint path (nearest waypoint by book position) throughout. Without
           the walk and the pull, rows off the centre land on another self-motion branch and end on a joint limit.
           The candidate covering the most rows wins (the best at the centre alone can cover much less: 86 vs 121 of
           121 on 2026-11-05); its failing rows are dropped.

Writes the start table the env resets from (``START_TABLE_PATH``; one row per book start, plus the solver joint paths
the rule-based uses as null-space references):

    python scripts/tools/solve_cabinet_book_start.py --headless

Placement sweep (stages 1-2, book path moves with the cabinet; prints how far each arm gets, writes nothing):

    python scripts/tools/solve_cabinet_book_start.py --headless --cabinet_x 0.9 0.95 1.0 --cabinet_y 0.0 0.1 0.2

The table uses the ``LEFT/RIGHT_GRASP_FLIP`` orientation from the env cfg (the attachment joints use the same
constants); if the other orientation would win, the script says so.

Collisions: the scene keeps the table and the robot's self-collisions (arm-arm included). Every IK iteration
teleports the arms and steps once with the PD target at the commanded joints, so any contact shows up as the measured
joints moving off the command; more than ``COLLISION_RAD`` counts as a collision at that waypoint (cabinet and book
are not in the scene).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--iters", type=int, default=300, help="IK iterations for the start grasp")
parser.add_argument("--path_iters", type=int, default=30, help="IK iterations per path waypoint")
parser.add_argument("--num_random", type=int, default=8, help="random seeds on top of the structured ones")
parser.add_argument("--door_deg", type=float, default=90.0)
parser.add_argument("--door_step_deg", type=float, default=5.0)
parser.add_argument("--cabinet_x", type=float, nargs="+", default=None, help="cabinet centre x (base), sweep")
parser.add_argument("--cabinet_y", type=float, nargs="+", default=None, help="cabinet centre y (base), sweep")
parser.add_argument("--pitch", type=float, nargs="+", default=None, help="grasp approach pitch (deg, both arms), sweep")
parser.add_argument("--table_dz", type=float, default=0.0, help="raise the table (its collision box, cabinet, book path) by this much; writes nothing")
parser.add_argument("--k_sigma", type=float, default=0.5, help="null-space ascent of the TCP Jacobian's smallest singular value")
parser.add_argument("--sigma_cap", type=float, default=0.15, help="no ascent above this sigma_min")
parser.add_argument("--grid", type=int, default=9, help="book start rows per axis over the start region (table)")
parser.add_argument("--right_candidates", type=int, default=4, help="stage-2 right arms tried on the whole table")
parser.add_argument("--region_x", type=float, nargs=2, default=None, help="explore another book start region (writes nothing)")
parser.add_argument("--region_y", type=float, nargs=2, default=None, help="explore another book start region (writes nothing)")
parser.add_argument("--out", default=None, help="write the start table here instead (also for --table_dz/--region/--cabinet runs; book starts shifted with the cabinet and table)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import itertools  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.utils.math import compute_pose_error, quat_apply, quat_from_angle_axis, quat_mul  # noqa: E402

import dual_fr3_lab.tasks  # noqa: E402, F401
from dual_fr3_lab.robots import UMI_TCP_OFFSET_POS, UMI_TCP_OFFSET_ROT  # noqa: E402
from dual_fr3_lab.tasks.manager_based.cabinet_book_insert import cabinet_book_insert_env_cfg as C  # noqa: E402
from dual_fr3_lab.tasks.manager_based.lift_plate.joint_pos_env_cfg import (  # noqa: E402
    FR3_JOINT_LIMIT_INSET,
    fr3_joint_limits_inset,
)

# left-arm seeds (the right arm uses the mirror): ready pose, forward-reaching variants
SEEDS = [
    (-0.5236, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854),
    (0.0, 0.3, 0.0, -2.0, 0.0, 0.73, 0.7854),
    (0.0, 0.3, 0.0, -2.0, 0.0, 0.73, -0.7854),
    (-0.3, 0.2, 0.0, -2.0, 0.0, 2.2, 0.7854),
    (-0.3, 0.2, 0.0, -2.0, 0.0, 2.2, -0.7854),
    (0.0, 0.4, 0.0, -1.6, 1.57, 1.6, 0.0),
    (0.0, 0.4, 0.0, -1.6, -1.57, 1.6, 0.0),
    (0.5, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7854),
]


COLLISION_RAD = 0.003  # measured joint - command after the step: contact pushed the arm
# FR3 kinematics (= Panda), modified DH (Craig): (a_{i-1}, d_i, alpha_{i-1}) for joints 1-7; link7 frame = DH frame 7
FR3_DH = ((0.0, 0.333, 0.0), (0.0, 0.0, -math.pi / 2), (0.0, 0.316, math.pi / 2), (0.0825, 0.0, math.pi / 2),
          (-0.0825, 0.384, -math.pi / 2), (0.0, 0.0, math.pi / 2), (0.088, 0.0, math.pi / 2))


def T(x):
    return x.torch if hasattr(x, "torch") else x


def skew(r):
    z = torch.zeros_like(r[:, 0])
    return torch.stack([torch.stack([z, -r[:, 2], r[:, 1]], -1),
                        torch.stack([r[:, 2], z, -r[:, 0]], -1),
                        torch.stack([-r[:, 1], r[:, 0], z], -1)], 1)


def rot_z(angle):
    axis = torch.zeros(angle.shape[0], 3, device=angle.device)
    axis[:, 2] = 1.0
    return quat_from_angle_axis(angle, axis)


def mirror(seed):
    # left/right arms are mirror images about the x-z plane: joints 1, 3, 5, 7 flip sign
    return tuple(-v if k in (0, 2, 4, 6) else v for k, v in enumerate(seed))


xs = args.cabinet_x or [C.CABINET_XY_B[0]]
ys = args.cabinet_y or [C.CABINET_XY_B[1]]
if args.pitch is None:
    pitches = [(C.LEFT_GRASP_PITCH, C.RIGHT_GRASP_PITCH)]
else:
    pitches = [(math.radians(d), math.radians(d)) for d in args.pitch]
places = list(itertools.product(xs, ys, pitches))
P = len(places)
table = P == 1  # stage 3
write = table and (args.out is not None or (args.table_dz == 0.0 and args.region_x is None and args.region_y is None))
G = args.grid if table else 1
region_x, region_y = args.region_x or C.BOOK_START_REGION_X, args.region_y or C.BOOK_START_REGION_Y
gx = np.linspace(*region_x, G) if G > 1 else [C.BOOK_START_POS_B[0]]
gy = np.linspace(*region_y, G) if G > 1 else [C.BOOK_START_POS_B[1]]
rows_b = [(float(x), float(y), C.BOOK_ON_TABLE_Z_B) for x in gx for y in gy]
R = len(rows_b)
lim = torch.tensor(fr3_joint_limits_inset(FR3_JOINT_LIMIT_INSET))
gen = torch.Generator().manual_seed(0)
num_random = max(args.num_random, math.ceil(R / 2) - len(SEEDS))  # stage 2 needs one env per row
rand = lim[:, 0] + 0.1 + (lim[:, 1] - lim[:, 0] - 0.2) * torch.rand(num_random, 7, generator=gen)
seeds = torch.cat([torch.tensor(SEEDS), rand])
S = seeds.shape[0]
K = args.right_candidates if table else 1
N = max(P * S * 2, K * R)  # stages 1-2 env e: placement e // (2S), seed (e // 2) % S, flip e % 2; stage 3: candidate
# e // R, row e % R

scene_cfg = C.CabinetBookInsertSceneCfg(num_envs=N, env_spacing=2.5, replicate_physics=False)
scene_cfg.cabinet = None  # kinematic check: frames from the cfg constants
scene_cfg.book = None
scene_cfg.contact_book = None  # sensors on the removed assets
scene_cfg.cabinet_wrench = None
if args.table_dz:  # the arms must clear the raised table top too
    _p = scene_cfg.table.init_state.pos
    scene_cfg.table.init_state.pos = (_p[0], _p[1], _p[2] + args.table_dz)
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, physics=C.PhysxCfg()))
scene = InteractiveScene(scene_cfg)
sim.reset()
dev = sim.device
lim = lim.to(dev)
robot = scene["robot"]
base = scene.env_origins + torch.tensor([0.0, 0.0, C.ROBOT_BASE_Z], device=dev)
all_ids = torch.arange(N, device=dev)
place_id = torch.clamp(all_ids // (2 * S), max=P - 1)
flip = {s: (all_ids % 2).bool() for s in ("left", "right")}
cfg_flip = {"left": C.LEFT_GRASP_FLIP, "right": C.RIGHT_GRASP_FLIP}
shift = torch.tensor([[x - C.CABINET_XY_B[0], y - C.CABINET_XY_B[1], args.table_dz] for x, y, _ in places], device=dev)[place_id]
pitch = {s: torch.tensor([pt[k] for _, _, pt in places], device=dev)[place_id] for k, s in enumerate(("left", "right"))}
off_p = torch.tensor(UMI_TCP_OFFSET_POS, device=dev).repeat(N, 1)
off_q = torch.tensor(UMI_TCP_OFFSET_ROT, device=dev).repeat(N, 1)
eye6, eye7 = torch.eye(6, device=dev), torch.eye(7, device=dev)


def vec(v):
    return torch.tensor(v, device=dev, dtype=torch.float32).repeat(N, 1)


def compose(p, q, lp, lq):
    return p + quat_apply(q, lp), quat_mul(q, lq)


def grasp_local(side):
    sign, pos_fn = (-1.0, C.left_grasp_pos) if side == "left" else (1.0, C.right_grasp_pos)
    pos = torch.tensor([pos_fn(float(b)) for b in pitch[side]], device=dev)
    q0 = torch.tensor([C.grasp_rot(sign, False, float(b)) for b in pitch[side]], device=dev)
    q1 = torch.tensor([C.grasp_rot(sign, True, float(b)) for b in pitch[side]], device=dev)
    return pos, torch.where(flip[side][:, None], q1, q0)


# cabinet frames (world) at door 0, from the cfg constants
cab_p = base + vec((C.CABINET_XY_B[0], C.CABINET_XY_B[1], C.TABLE_TOP_Z_B)) + shift
cab_q = vec(C.CABINET_ROT)
door_p, door_q = compose(cab_p, cab_q, vec(C.DOOR_IN_CABINET), vec((0.0, 0.0, 0.0, 1.0)))
hinge = door_p + quat_apply(door_q, vec(C.DOOR_HINGE_IN_DOOR))
hp, hq = compose(door_p, door_q, vec(C.HANDLE_IN_DOOR), vec((0.0, 0.0, 0.0, 1.0)))
book_q = vec((0.0, 0.0, 0.0, 1.0))
SEG_NAMES = ("lift", "to pre-insert x", "slide to the lane", "insert")
seg_end = np.cumsum((0,) + C.BOOK_PATH_SEGMENT_STEPS)  # waypoint index at the end of each segment


def book_path_of(start):
    """Book centre waypoints (base frame, [W, 3]) along book_path_points(start), BOOK_PATH_SEGMENT_STEPS per segment."""
    pts = torch.tensor(C.book_path_points(start), device=dev)
    out = [pts[0]]
    for i, m in enumerate(C.BOOK_PATH_SEGMENT_STEPS):
        out += [pts[i] + (pts[i + 1] - pts[i]) * (k / m) for k in range(1, m + 1)]
    return torch.stack(out)


row_paths = torch.stack([book_path_of(r) for r in rows_b])  # [R, W, 3]
book_path = book_path_of(C.BOOK_START_POS_B)[None].repeat(N, 1, 1)  # stages 1-2: centre of the start region
door_path = [math.radians(d) for d in np.arange(0.0, args.door_deg + 1e-6, args.door_step_deg)]

arms = {}
jn = robot.joint_names
for side in ("left", "right"):
    arms[side] = (robot.body_names.index(f"{side}_fr3_link7"), [jn.index(f"{side}_fr3_joint{k}") for k in range(1, 8)])
q = T(robot.data.default_joint_pos).clone()
seed_of = (all_ids // 2) % S


def seeded(side):
    sd = seeds if side == "left" else torch.tensor([mirror(s) for s in seeds.tolist()])
    return sd.to(dev)[seed_of]


q[:, arms["left"][1]] = seeded("left")  # stage 1; the right arm per stage
fingers = {"left": C.LEFT_FINGER_POS, "right": C.RIGHT_FINGER_POS}
for side in ("left", "right"):
    for k in (1, 2):
        q[:, jn.index(f"{side}_fr3_finger_joint{k}")] = fingers[side]


ACTIVE = ["left", "right"]  # arms the IK moves; the others hold their joints
MODEL_CHECKED = {"left": False, "right": False}
R_LIFTED = int(seg_end[2])  # right waypoint at the pre-insert x (lifted): held while the door opens


def schedule():
    """(left door waypoint, right book waypoint) per step, in the rule-based order."""
    d, w = len(door_path), book_path.shape[1]
    s = [(min(k, d - 1), min(k, R_LIFTED)) for k in range(max(d, R_LIFTED + 1))]
    return s + [(d - 1, k) for k in range(s[-1][1] + 1, w)]


def targets(wl, wr):
    """TCP targets (world): left at door waypoint wl, right at book waypoint wr."""
    rz = rot_z(torch.full((N,), -door_path[wl], device=dev))
    left0 = compose(hp, hq, *grasp_local("left"))  # per call: the flips change between stages
    lt = (hinge + quat_apply(rz, left0[0] - hinge), quat_mul(rz, left0[1]))
    bp = book_path[:, wr] + base + shift
    return {"left": lt, "right": compose(bp, book_q, *grasp_local("right"))}


def ik(tg, iters, pull=None, k_pull=0.0):
    """DLS IK on the ACTIVE arms; ``pull`` = {side: joint q ([7] or [N, 7])} pulled toward in the null space."""
    for _ in range(iters):
        robot.write_joint_position_to_sim_index(position=q)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
        robot.set_joint_position_target(q)  # PD holds the teleported pose during the step
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(0.01)
        err = {}
        for side in ACTIVE:
            b, ids = arms[side]
            push_dev = (T(robot.data.joint_pos)[:, ids] - q[:, ids]).abs().amax(-1)
            p7, q7 = T(robot.data.body_pos_w)[:, b], T(robot.data.body_quat_w)[:, b]
            r = quat_apply(q7, off_p)
            e_p, e_r = compute_pose_error(p7 + r, quat_mul(q7, off_q), tg[side][0], tg[side][1], rot_error_type="axis_angle")
            jac = T(robot.data.body_link_jacobian_w)[:, b - 1][:, :, ids]
            jac = torch.cat([jac[:, :3] - skew(r) @ jac[:, 3:], jac[:, 3:]], 1)
            jt = jac.transpose(1, 2)
            jpinv = jt @ torch.linalg.inv(jac @ jt + (0.05**2) * eye6)
            qj = q[:, ids]
            push = torch.relu(0.3 - (qj - lim[:, 0])) - torch.relu(0.3 - (lim[:, 1] - qj))
            if pull is not None and side in pull:
                push = push + k_pull / 0.3 * (pull[side] - qj)
            if args.k_sigma > 0.0:  # away from singular configurations (handle lock forces, see module doc)
                push = push + args.k_sigma / 0.3 * sigma_grad(qj)
            dq = (jpinv @ torch.cat([e_p, e_r], -1)[..., None])[..., 0]
            dq += ((eye7 - jpinv @ jac) @ (0.3 * push)[..., None])[..., 0]
            q[:, ids] = torch.clamp(qj + dq.clamp(-0.1, 0.1), lim[:, 0], lim[:, 1])
            # smallest singular value of the TCP Jacobian (position rows in m/rad, rotation rows in rad/rad)
            sig = torch.linalg.svdvals(jac)[:, -1]
            if not MODEL_CHECKED[side]:
                MODEL_CHECKED[side] = True
                print(f"[start] {side} sigma_min: DH model vs sim Jacobian, max diff {(fr3_sigma_min(qj) - sig).abs().max():.2e}")
            err[side] = (torch.norm(e_p, dim=-1), torch.norm(e_r, dim=-1), push_dev, sig)
    return err


def fr3_sigma_min(qa):
    """Smallest singular value of the 6x7 TCP Jacobian (UMI TCP 0.2844 m along link7 z), from the DH model.

    Rotation invariant, so the arm mounting does not matter. Differentiable (null-space ascent).
    """
    n_ = qa.shape[0]
    tf = torch.eye(4, device=qa.device).repeat(n_, 1, 1)
    ps, zs = [], []
    for i, (a, d, al) in enumerate(FR3_DH):
        ca, sa = math.cos(al), math.sin(al)
        c, s = torch.cos(qa[:, i]), torch.sin(qa[:, i])
        z0, o = torch.zeros_like(c), torch.ones_like(c)
        step = torch.stack([
            torch.stack([c, -s, z0, a * o], -1),
            torch.stack([s * ca, c * ca, -sa * o, -d * sa * o], -1),
            torch.stack([s * sa, c * sa, ca * o, d * ca * o], -1),
            torch.stack([z0, z0, z0, o], -1)], 1)
        tf = tf @ step
        ps.append(tf[:, :3, 3])
        zs.append(tf[:, :3, 2])
    tcp_p = tf[:, :3, 3] + tf[:, :3, 2] * UMI_TCP_OFFSET_POS[2]
    jv = torch.stack([torch.cross(z, tcp_p - pp, dim=-1) for z, pp in zip(zs, ps)], -1)
    return torch.linalg.svdvals(torch.cat([jv, torch.stack(zs, -1)], 1))[:, -1]


def sigma_grad(qa):
    """d min(sigma_min, cap) / dq."""
    qa = qa.detach().requires_grad_(True)
    with torch.enable_grad():
        sg = torch.clamp(fr3_sigma_min(qa), max=args.sigma_cap)
        return torch.autograd.grad(sg.sum(), qa)[0]


def margin(side):
    qj = q[:, arms[side][1]]
    return torch.minimum(qj - lim[:, 0], lim[:, 1] - qj).amin(-1)


def describe(side, w):
    if w < 0:
        return "nothing"
    if side == "left":
        return f"door {math.degrees(door_path[w]):.0f} deg"
    if w == 0:
        return "start"
    i = int(np.searchsorted(seg_end, w)) - 1
    return f"{SEG_NAMES[i]} {w - seg_end[i]}/{C.BOOK_PATH_SEGMENT_STEPS[i]}"


def run_paths(iters, ref=None, k_pull=0.0):
    """Start grasp, then the schedule; per ACTIVE arm: ok (whole path, collision free), margins, furthest waypoint,
    whether the first failure was a collision, joint path (one entry per own waypoint).

    ``ref`` = (book waypoints [W', 3], right joint path per env [N, W', 7]): null-space pull of the right arm toward
    the joint path at the waypoint nearest to each env's book target.
    """

    def pull(wr):
        if ref is None:
            return None
        return {"right": ref[1][all_ids, torch.argmin(torch.cdist(book_path[:, wr], ref[0]), dim=-1)]}

    sched = schedule()
    err = ik(targets(*sched[0]), iters, pull(0), k_pull)
    res = {"q_start": q.clone()}
    for s in ACTIVE:
        free = err[s][2] < COLLISION_RAD
        ok = (err[s][0] < 0.002) & (err[s][1] < math.radians(1.0)) & free
        res[s] = {"start_margin": margin(s), "path_ok": ok, "reach": torch.where(ok, 0, -1), "hit": ~free,
                  "path_q": [q[:, arms[s][1]].clone()], "n_way": len(door_path) if s == "left" else book_path.shape[1],
                  "sig_min": err[s][3].clone(), "path_sig": [err[s][3].clone()]}
        res[s]["path_margin"] = res[s]["start_margin"].clone()
    last = {"left": 0, "right": 0}
    for wl, wr in sched[1:]:
        err = ik(targets(wl, wr), args.path_iters, pull(wr), k_pull)
        for s in ACTIVE:
            w, r = (wl if s == "left" else wr), res[s]
            free = err[s][2] < COLLISION_RAD
            r["hit"] |= r["path_ok"] & ~free  # the first failure is a collision
            r["path_ok"] &= (err[s][0] < 0.005) & (err[s][1] < math.radians(2.0)) & free
            r["path_margin"] = torch.minimum(r["path_margin"], margin(s))
            r["reach"] = torch.where(r["path_ok"], w, r["reach"])
            r["sig_min"] = torch.minimum(r["sig_min"], err[s][3])
            if w != last[s]:
                r["path_q"].append(q[:, arms[s][1]].clone())
                r["path_sig"].append(err[s][3].clone())
                last[s] = w
    for s in ACTIVE:
        res[s]["path_q"] = torch.stack(res[s]["path_q"], 1)  # [N, own waypoints, 7]
        res[s]["path_sig"] = torch.stack(res[s]["path_sig"], 1)  # [N, own waypoints]
    return res


def score(r, ok):
    """whole path first, then joint margin (up to 0.3 rad) + 2 sigma_min (up to 0.15); else the furthest waypoint"""
    good = r["path_margin"].clamp(max=0.3) + 2.0 * r["sig_min"].clamp(max=0.15)
    return torch.where(ok, 10.0 + good, r["reach"] / r["n_way"] + 0.01 * r["start_margin"])


def pick(sc, s):
    """best env per placement (table: the cfg orientation only)"""
    out = [int(torch.argmax(torch.where(place_id == p, sc, -1e9))) for p in range(P)]
    if table:
        alt = out[0]
        out = [int(torch.argmax(torch.where(flip[s] == cfg_flip[s], sc, -1e9)))]
        if bool(flip[s][alt]) != cfg_flip[s] and sc[alt] > sc[out[0]] + 1e-6:
            print(f"[start] note: {s} flip={bool(flip[s][alt])} would score {sc[alt]:.2f} vs cfg {sc[out[0]]:.2f}")
    return out


def line(tag, s, r, e, ok):
    return (f" | {tag}: {'OK' if ok[e] else 'NO'}{' (collision)' if r['hit'][e] else ''} "
            f"{describe(s, int(r['reach'][e]))}, margin start {r['start_margin'][e]:.2f} path {r['path_margin'][e]:.2f} "
            f"sigma_min {r['sig_min'][e]:.3f} "
            f"(seed {(e // 2) % S}, flip {bool(flip[s][e])})")


def place_q(side, e_list):
    """joints [N, 7]: each env gets the start joints of its placement's winner"""
    qs = torch.stack([starts[side][e, arms[side][1]] for e in e_list])
    return qs[place_id]


starts = {}
# ---------- stage 1: left arm (right parked at its ready pose) ----------
ACTIVE[:] = ["left"]
q[:, arms["right"][1]] = torch.tensor(mirror(SEEDS[0]), device=dev)
st1 = run_paths(args.iters)
starts["left"] = st1["q_start"]
best_l = pick(score(st1["left"], st1["left"]["path_ok"]), "left")
el0 = best_l[0]
print("[start] left winner sigma_min along the door path: " + ", ".join(
    f"{math.degrees(a):.0f}:{v:.3f}" for a, v in zip(door_path, st1["left"]["path_sig"][el0].tolist())))
# ---------- stage 2: right arm with the left arm on its stage-1 path ----------
ACTIVE[:] = ["left", "right"]
flip["left"][:] = torch.tensor([bool(flip["left"][e]) for e in best_l], device=dev)[place_id]
q[:, arms["left"][1]] = place_q("left", best_l)
q[:, arms["right"][1]] = seeded("right")
st2 = run_paths(args.iters)
starts["right"] = st2["q_start"]
pair_ok = st2["right"]["path_ok"] & st2["left"]["path_ok"]
best_r = pick(score(st2["right"], pair_ok), "right")

print(f"[start] {S} seeds x 2 flips per placement; door 0-{args.door_deg:.0f} deg; table dz {args.table_dz:+.2f}; "
      f"book path (cfg placement) {np.round(np.array(C.book_path_points(C.BOOK_START_POS_B)), 3).tolist()}")
for p, (x, y, pt) in enumerate(places):
    el, er = best_l[p], best_r[p]
    print(f"[start] cabinet ({x:.2f}, {y:.2f}) pitch {math.degrees(pt[0]):.0f}"
          + line("left", "left", st1["left"], el, st1["left"]["path_ok"])
          + line("right (with that left)", "right", st2["right"], er, pair_ok))
    if not bool(st2["left"]["path_ok"][er]):
        print(f"[start]   left fails next to that right: {describe('left', int(st2['left']['reach'][er]))}"
              f"{' (collision)' if st2['left']['hit'][er] else ''}")

if table:
    for s, e, ok in (("left", best_l[0], st1["left"]["path_ok"]), ("right", best_r[0], pair_ok)):
        if not bool(ok[e]):
            print(f"[start] WARNING: no {s} candidate follows the whole path (move the cabinet/book or shorten the path)")
    # ---------- stage 3: one env per (right candidate, book start row) ----------
    sc2 = torch.where(flip["right"] == cfg_flip["right"], score(st2["right"], pair_ok), -1e9)
    cands = []  # distinct start configurations (envs past 2S repeat the seeds)
    for e in torch.argsort(sc2, descending=True).tolist():
        if len(cands) == K or not pair_ok[e] or sc2[e] < -1e8:
            break
        qe = starts["right"][e, arms["right"][1]]
        if all((qe - starts["right"][c, arms["right"][1]]).abs().max() > 0.2 for c in cands):
            cands.append(e)
    cands = cands or [best_r[0]]
    cand_of = torch.clamp(all_ids // R, max=len(cands) - 1)
    row_of = all_ids % R
    for s in ("left", "right"):
        flip[s][:] = cfg_flip[s]
    el = best_l[0]
    cand_env = torch.tensor(cands, device=dev)[cand_of]
    q[:, arms["left"][1]] = starts["left"][el, arms["left"][1]]
    q[:, arms["right"][1]] = starts["right"][cand_env][:, arms["right"][1]]
    ref = (book_path_of(C.BOOK_START_POS_B), st2["right"]["path_q"][cand_env])
    # walk the book start from the centre to each row (same self-motion branch as the candidate's arm)
    c0 = torch.tensor(C.BOOK_START_POS_B, device=dev)
    walk_n = max(1, math.ceil(torch.norm(row_paths[:, 0] - c0, dim=-1).max().item() / 0.02))
    book_path = torch.stack([c0 + (row_paths[row_of, 0] - c0) * (k / walk_n) for k in range(walk_n + 1)], 1)
    for k in range(walk_n + 1):
        ik(targets(0, k), args.path_iters, {"right": ref[1][:, 0]}, 0.05)
    book_path = row_paths[row_of]
    st3 = run_paths(args.iters, ref=ref, k_pull=0.05)
    rr, rl = st3["right"], st3["left"]
    env_ok = rr["path_ok"] & rl["path_ok"]
    cover = []
    for k, e in enumerate(cands):
        ids = all_ids[(all_ids // R == k)]
        m = rr["path_margin"][ids][env_ok[ids]]
        cover.append((int(env_ok[ids].sum()), float(m.mean()) if len(m) else 0.0))
        print(f"[start] right candidate seed {(e // 2) % S}: centre path margin {st2['right']['path_margin'][e]:.2f}, "
              f"table {cover[-1][0]}/{R} rows, mean margin {cover[-1][1]:.2f}")
    kb = max(range(len(cands)), key=lambda k: cover[k])
    er = cands[kb]
    rows_env = kb * R + torch.arange(R, device=dev)  # this candidate's envs, in row order
    row_ok = env_ok[rows_env]
    ok_rows = [r for r in range(R) if bool(row_ok[r])]
    q1 = {"left": starts["left"][el, arms["left"][1]], "right": starts["right"][er, arms["right"][1]]}
    print(f"[start] table: seed {(er // 2) % S}, {len(ok_rows)}/{R} book starts follow the whole path "
          f"(grid {G} x {G}, x {region_x}, y {region_y}, z {C.BOOK_ON_TABLE_Z_B:.3f})")
    # map: one line per x, one column per y; digit = whole path, right path margin * 10 (rad); . = start not solved,
    # l/x/s/i = right failed in lift / to pre-insert x / slide / insert, L = left failed; upper case (C at the
    # start) = by a collision
    print("[start] map (rows x, columns y):  y " + " ".join(f"{y:+.2f}"[1:] if y < 0 else f"{y:.2f}" for y in gy))
    for ix, x in enumerate(gx):
        cells = []
        for iy in range(len(gy)):
            r = int(rows_env[ix * len(gy) + iy])
            if bool(env_ok[r]):
                cells.append(str(min(9, int(10 * rr["path_margin"][r]))))
            elif not bool(rl["path_ok"][r]) and bool(rr["path_ok"][r]):
                cells.append("L")
            elif int(rr["reach"][r]) < 0:
                cells.append("C" if bool(rr["hit"][r]) else ".")
            else:
                c = "lxsi"[int(np.searchsorted(seg_end, int(rr["reach"][r]) + 1)) - 1]
                cells.append(c.upper() if bool(rr["hit"][r]) else c)
        print(f"[start]   x {x:.3f}  " + "    ".join(cells))
    ok_env = rows_env[ok_rows] if ok_rows else rows_env[:0]
    if ok_rows:
        m = rr["path_margin"][ok_env]
        d = (rr["path_q"][ok_env, 0] - q1["right"]).abs().amax(-1)
        print(f"[start] right path margin over the rows: min {m.min():.2f} mean {m.mean():.2f} rad | start q - centre "
              f"start q: max {d.max():.2f} rad | left path margin {st1['left']['path_margin'][el]:.2f} rad")
    if ok_rows and write:
        q_out = T(robot.data.default_joint_pos)[0].clone().repeat(len(ok_rows), 1)
        q_out[:, arms["left"][1]] = q1["left"]
        q_out[:, arms["right"][1]] = rr["path_q"][ok_env, 0]
        for s in ("left", "right"):
            for k in (1, 2):
                q_out[:, jn.index(f"{s}_fr3_finger_joint{k}")] = fingers[s]
        starts_b = np.array([rows_b[r] for r in ok_rows]) + shift[0].cpu().numpy()  # as solved (moved with the cabinet)
        book_pose_env = np.concatenate([starts_b + [0.0, 0.0, C.ROBOT_BASE_Z], np.tile([0.0, 0.0, 0.0, 1.0], (len(ok_rows), 1))], 1)
        out = args.out or C.START_TABLE_PATH
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez(
            out,
            joint_names=np.array(jn),
            joint_pos=q_out.cpu().numpy(),  # [rows, joints]
            book_start_b=starts_b,
            book_pose_env=book_pose_env,  # book root pose relative to the env origin, quat xyzw
            left_flip=cfg_flip["left"],
            right_flip=cfg_flip["right"],
            path_margin=np.stack([np.full(len(ok_rows), st1["left"]["path_margin"][el].item()), m.cpu().numpy()], 1),
            # the IK paths from these starts (null-space references for the rule-based)
            left_path_door=np.array(door_path),
            left_path_q=st1["left"]["path_q"][el].cpu().numpy(),  # [door waypoints, 7]
            right_path_book=(row_paths[ok_rows] + shift[0]).cpu().numpy(),  # [rows, W, 3] base frame
            right_path_q=rr["path_q"][ok_env].cpu().numpy(),  # [rows, W, 7]
        )
        print("[start] wrote", out, f"({len(ok_rows)} rows)")
app.close()
