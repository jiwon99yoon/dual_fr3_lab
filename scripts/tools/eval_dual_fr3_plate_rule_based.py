"""Rule-based squeeze-lift on the box Sim2Real-Light env (2026-11-02).

Every arm command goes through the env's own arm action, so the script obeys the same deploy contract as the RL
policy: q_target = q_measured + clip(0.1 a, +-0.1 rad) at 20 Hz, explicit PD Kp80/Kd4 (+-15 % DR), FR3 datasheet
limits. Box mass/friction/pose DR of the env stay on.

Per env (reference pose moves at a capped speed, the arm tracks it with DLS IK, gain * J+ e, scaled as a whole
so no joint step exceeds the bound):
  0 HIGH     plates 8 cm outside the box y faces, 10 cm above the box top, normals facing the box
             (rotation about the plate normal left to the redundancy until APPROACH)
  1 DOWN     same, at grip height (box centre + grip_dz)
  2 APPROACH push each reference `depth` past its face (blocked plate -> PD error -> squeeze force). Targets follow
             the observed box pose until the first plate touches; plate roll is servoed from here on, since it
             is the box pitch axis once the box is squeezed
  3 SETTLE   hold
  4 LIFT     raise both references so the box centre reaches the commanded goal height (squeeze reference kept)
  5 HOLD
Box pose and goal are what the policy observes (ground truth here, corruption off).
Success = the env's own success term count (goal 10 cm, tilt 15 deg, both plates >= 1 N, held 0.5 s).
Results and limits: docs/dual_fr3_lift_sim2real.md, section "rule-based".

    python scripts/tools/eval_dual_fr3_plate_rule_based.py --viz none --device cpu --num_envs 128 \
        --v_pre 0.3 --v_lift 0.15 --grip_dz 0.05 --box_dx -0.05 0.02 --disturb 1.0

With ``--pregrasp_gap g`` APPROACH stops g outside each face instead of pushing in, and once both plates hold
that pose the arm joints and box pose are recorded (one row per env). ``--save_table`` writes the rows that
converged without touching the box; the `-Pregrasp` box task starts its episodes from this table.

``--roll_align`` servos the whole plate orientation from HIGH on, with the square plate's edges parallel to the box
edges (without it the rotation about the plate normal is whatever the arm had when APPROACH started, ~18 deg).

``--snap_j7`` keeps the default approach and, when a row is recorded, turns joint 7 (the plate normal axis, so the
plate centre and normal stay put) by at most 45 deg until the plate edges are parallel to the box edges. Joint 7
stays near the un-aligned table; ``--roll_align`` instead lands it 72 deg away (2026-11-03: the RollAlign policy
trained from that start twisted the box 40-67 deg about z).

``--box_size X Y Z`` swaps the box (x front-back, y grip direction, z height) without touching the env cfg.
``--video out.mp4`` renders every env from three cameras (overview, front, side) and writes the envs one after
another; use a few envs (``--num_envs 3``). ``--log out.npz`` saves per-step phase, plate, joint, command and force
arrays for every env.

``--gain_free 1.0`` lowers the IK gain until the plates touch. With the default gain 2 everywhere, the light wrist
joints over-correct every 50 ms step and the plates shake on the way down (the squeeze still needs gain 2).
"""

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--mass", type=float, nargs=2, default=None, help="box mass range [kg]; default = env DR")
parser.add_argument("--box_dx", type=float, nargs=2, default=None, help="box spawn x offset range [m] (env: -0.05 0.10)")
parser.add_argument("--disturb", type=float, default=0.0, help="disturbance scale 0..1 (training ramps to 1)")
parser.add_argument("--depth", type=float, default=0.05, help="[m] squeeze reference past each face")
parser.add_argument("--pre_gap", type=float, default=0.08)
parser.add_argument("--grip_dz", type=float, default=0.0125, help="[m] plate centre above box centre")
parser.add_argument("--gain", type=float, default=2.0)
parser.add_argument("--gain_free", type=float, default=None,
                    help="IK gain before the plates touch (HIGH, DOWN, APPROACH until contact); default = --gain")
parser.add_argument("--null", type=float, default=0.1, help="null-space pull to the ready pose")
parser.add_argument("--damp", type=float, default=0.0,
                    help="[s] subtract damp * joint velocity from each joint step (adds Kp * damp to the PD's Kd 4)")
parser.add_argument("--lim_k", type=float, default=1.0, help="null-space push away from FR3 limits (within 0.3 rad)")
parser.add_argument("--limit_aware", action="store_true",
                    help="re-solve the IK without the joints the env's target clamp holds at a limit")
parser.add_argument("--fixed_roll", action="store_true", help="also servo the plate rotation about its normal")
parser.add_argument("--roll_align", action="store_true",
                    help="servo the full plate orientation from the start, with the square plate's edges parallel to "
                         "the box edges (implies --fixed_roll)")
parser.add_argument("--snap_j7", action="store_true",
                    help="pre-grasp rows: turn joint 7 by <= 45 deg so the plate edges are parallel to the box edges")
parser.add_argument("--v_pre", type=float, default=0.4)
parser.add_argument("--w_pre", type=float, default=1.5)
parser.add_argument("--v_app", type=float, default=0.15)
parser.add_argument("--v_lift", type=float, default=0.25)
parser.add_argument("--settle", type=int, default=6)
parser.add_argument("--a_ref", type=float, default=None,
                    help="[m/s^2] accelerate/decelerate the reference position instead of starting and stopping at full speed")
parser.add_argument("--v_contact", type=float, default=None,
                    help="[m/s] APPROACH speed once the reference is within --contact_zone of the box face (guarded contact)")
parser.add_argument("--contact_zone", type=float, default=0.02, help="[m] outside the face, see --v_contact")
parser.add_argument("--down_timeout", type=int, default=30, help="policy steps to wait for the plates at grip height")
parser.add_argument("--steps", type=int, default=220, help="policy steps (20 Hz)")
parser.add_argument("--trace", type=int, default=0, help="env to trace every 10 steps (-1 off)")
parser.add_argument("--pregrasp_gap", type=float, default=None,
                    help="[m] stop the approach this far outside each face and record the pre-grasp state")
parser.add_argument("--save_table", default=None, help="with --pregrasp_gap: write the pre-grasp table (.npz) here")
parser.add_argument("--seed", type=int, default=None, help="env seed (DR samples)")
parser.add_argument("--box_size", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="[m] box size: x front-back, y grip direction, z height (env: 0.555 0.475 0.275)")
parser.add_argument("--video", default=None, help="write an mp4 of every env from three cameras (use few envs)")
parser.add_argument("--video_size", type=int, nargs=2, default=(640, 480), metavar=("W", "H"))
parser.add_argument("--log", default=None, help="write per-step arrays (phase, plates, joints, commands, forces) to .npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.fixed_roll |= args.roll_align
if args.video:
    args.enable_cameras = True
    os.environ.setdefault("DR_VIS", "0")  # the DR range plates hide the lifted box
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors.camera import CameraCfg  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    axis_angle_from_quat,
    compute_pose_error,
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_from_matrix,
    quat_mul,
)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import dual_fr3_lab.tasks  # noqa: E402,F401
from dual_fr3_lab.robots.dual_fr3_plate_ee import PLATE_TCP_OFFSET_POS  # noqa: E402
from dual_fr3_lab.tasks.manager_based.lift_plate.joint_pos_env_cfg import (  # noqa: E402
    FR3_JOINT_POS_LIMITS,
    _box_halves,
    nonplate_contact_forces,
)

SIGN = {"left": 1.0, "right": -1.0}  # left plate on the box +y face, pushes toward -y
HIGH, DOWN, APPROACH, SETTLE, LIFT, HOLD = range(6)
NAMES = ["high", "down", "approach", "settle", "lift", "hold"]
# --video cameras (eye, lookat) relative to the env origin, same as scripts/rsl_rl/record_views.py "box"
VIEWS = {
    "overview": ((2.3, -1.9, 1.9), (0.40, 0.0, 0.45)),
    "front (robot L = image R)": ((2.7, 0.0, 1.0), (0.45, 0.0, 0.50)),
    "side (robot right)": ((0.45, -2.8, 0.75), (0.45, 0.0, 0.50)),
}


def T(x):
    return x.torch if hasattr(x, "torch") else x


def skew(r):
    z = torch.zeros_like(r[:, 0])
    return torch.stack([torch.stack([z, -r[:, 2], r[:, 1]], -1),
                        torch.stack([r[:, 2], z, -r[:, 0]], -1),
                        torch.stack([-r[:, 1], r[:, 0], z], -1)], 1)


def step_toward(ref, fin, max_step):
    d = fin - ref
    n = torch.norm(d, dim=-1, keepdim=True)
    return ref + d * torch.clamp(max_step / n.clamp(min=1e-9), max=1.0)


def rotate_toward(q, q_fin, max_angle):
    aa = axis_angle_from_quat(quat_mul(quat_conjugate(q), q_fin))
    ang = torch.norm(aa, dim=-1)
    axis = aa / ang.clamp(min=1e-9)[:, None]
    return quat_mul(q, quat_from_angle_axis(torch.minimum(ang, max_angle), axis))


def shortest_arc(u, v):
    # Isaac Lab 3 quaternions are (x, y, z, w)
    q = torch.cat([torch.cross(u, v, dim=-1), 1.0 + (u * v).sum(-1, keepdim=True)], -1)
    return q / torch.norm(q, dim=-1, keepdim=True)


def edge_aligned_quat(q_cur, n_des, y_b):
    """Plate orientation with its normal (link7 z) along n_des and its edges along the box x axis and world z.

    The 140 mm square sits 45 deg from link7 x/y (its corners are on those axes), so an edge direction is
    (x + y) / sqrt(2) in link7. Of the four such orientations the one closest to q_cur is returned.
    """
    up = torch.zeros_like(n_des)
    up[:, 2] = 1.0
    x_b = torch.cross(y_b, up, dim=-1)  # horizontal box x (y_b is horizontal)
    best, best_dot = None, None
    for a in (up, -up, x_b, -x_b):
        e2 = torch.cross(n_des, a, dim=-1)  # image of link7 (-x + y) / sqrt(2)
        q = quat_from_matrix(torch.stack([(a - e2) / 2**0.5, (a + e2) / 2**0.5, n_des], -1))
        d = (q * q_cur).sum(-1).abs()
        if best is None:
            best, best_dot = q, d
        else:
            better = d > best_dot
            best = torch.where(better[:, None], q, best)
            best_dot = torch.where(better, d, best_dot)
    return best


def video_frame(panels, labels, text):
    """Panels side by side, each labelled at the bottom, with a text strip on top."""
    font, big = ImageFont.load_default(size=18), ImageFont.load_default(size=20)
    out = []
    for rgb, lab in zip(panels, labels):
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        left, top, right, bottom = draw.textbbox((0, 0), lab, font=font)
        xy = ((img.width - (right - left)) // 2 - left, img.height - (bottom - top) - 10 - top)
        draw.rectangle(draw.textbbox(xy, lab, font=font), fill=(255, 255, 255))
        draw.text(xy, lab, fill=(0, 0, 0), font=font)
        out.append(np.asarray(img))
    row = np.concatenate(out, axis=1)
    strip = Image.new("RGB", (row.shape[1], 34), (24, 24, 24))
    ImageDraw.Draw(strip).text((12, 6), text, fill=(240, 240, 240), font=big)
    return np.concatenate([np.asarray(strip), row], axis=0)


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.episode_length_s = (args.steps + 20) * cfg.sim.dt * cfg.decimation
    cfg.commands.object_pose.resampling_time_range = (1e3, 1e3)
    for k in list(cfg.curriculum.__dict__):
        setattr(cfg.curriculum, k, None)
    for k in list(cfg.terminations.__dict__):
        if k != "time_out":
            setattr(cfg.terminations, k, None)
    if args.box_dx is not None:
        cfg.events.reset_object_position.params["pose_range"]["x"] = tuple(args.box_dx)
    if args.mass is not None:
        cfg.events.randomize_box_density.params["mass_distribution_params"] = tuple(args.mass)
    if args.box_size is not None:
        # _box_halves() and every size-dependent term read spawn.size; spawn 2 cm above the floor like the env
        cfg.scene.object.spawn.size = tuple(args.box_size)
        cfg.scene.object.init_state.pos = (*cfg.scene.object.init_state.pos[:2], args.box_size[2] / 2 + 0.0225)
    if args.video:
        cfg.scene.env_spacing = 8.0  # keep the neighbours out of the frame
        for k in range(len(VIEWS)):
            setattr(cfg.scene, f"rec_{k}", CameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/RecCam_{k}", update_period=0.0,
                width=args.video_size[0], height=args.video_size[1], data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 20.0)),
            ))

    env = gym.make(args.task, cfg=cfg)
    raw = env.unwrapped
    env.reset()
    frames = []
    if args.video:
        cams = [raw.scene[f"rec_{k}"] for k in range(len(VIEWS))]
        for cam, (eye, lookat) in zip(cams, VIEWS.values()):
            cam.set_world_poses_from_view(torch.tensor([eye], device=raw.device) + raw.scene.env_origins,
                                          torch.tensor([lookat], device=raw.device) + raw.scene.env_origins)
        for _ in range(5):  # renderer warm-up, no physics step
            raw.sim.render()
    raw._disturb_scale = args.disturb
    dev, n = raw.device, raw.num_envs
    dt = cfg.sim.dt * cfg.decimation
    robot, box = raw.scene["robot"], raw.scene["object"]
    origin = raw.scene.env_origins
    succ_term = raw.reward_manager.get_term_cfg("success").func
    hold_steps = raw.reward_manager.get_term_cfg("success").params["hold_steps"]

    term = raw.action_manager.get_term("arm_action")
    names = list(term._joint_names)
    jids = term._joint_ids if not isinstance(term._joint_ids, slice) else list(range(len(names)))
    scale = float(term.cfg.scale)
    clip = float(list(term.cfg.clip.values())[0][1])
    arms = {}
    for side in SIGN:
        body = robot.find_bodies(f"{side}_fr3_link7")[0][0]
        cols = [k for k, nm in enumerate(names) if nm.startswith(side)]
        arms[side] = (body, cols, [jids[k] for k in cols])
    offset = torch.tensor(PLATE_TCP_OFFSET_POS, device=dev).repeat(n, 1)
    q_ready = T(robot.data.default_joint_pos)
    lim = torch.tensor(FR3_JOINT_POS_LIMITS, device=dev)
    # the limits the arm action clamps q_target to (Light-Margin: FCI limits minus 0.05 rad)
    tlim = torch.tensor(term.cfg.target_limits, device=dev) if term.cfg.target_limits is not None else lim.repeat(2, 1)
    eye3, eye6, eye7 = torch.eye(3, device=dev), torch.eye(6, device=dev), torch.eye(7, device=dev)

    def tcp(side):
        body = arms[side][0]
        p, q = T(robot.data.body_pos_w)[:, body], T(robot.data.body_quat_w)[:, body]
        r = quat_apply(q, offset)
        return p + r - origin, q, r

    def plate_forces():
        return torch.stack([torch.norm(T(raw.scene[s].data.force_matrix_w), dim=-1).amax(dim=(-1, -2))
                            for s in ("left_plate_object_contact", "right_plate_object_contact")], -1)

    hy, hz = _box_halves(raw)
    up = torch.tensor([[0.0, 0.0, 1.0]], device=dev).repeat(n, 1)
    ey = torch.tensor([[0.0, 1.0, 0.0]], device=dev).repeat(n, 1)
    ex = torch.tensor([[1.0, 0.0, 0.0]], device=dev).repeat(n, 1)
    edge_l7 = torch.tensor([[0.5**0.5, 0.5**0.5, 0.0]], device=dev).repeat(n, 1)  # a plate edge in link7

    def box_pose():
        c = T(box.data.root_pos_w) - origin
        q = T(box.data.root_quat_w)
        tilt = torch.rad2deg(torch.acos(quat_apply(q, up)[:, 2].clamp(-1, 1)))
        y_b = quat_apply(q, ey)
        y_b[:, 2] = 0.0
        return c, q, tilt, y_b / torch.norm(y_b, dim=-1, keepdim=True)

    def goal_env():
        cmd = raw.command_manager.get_command("object_pose")[:, :3]
        return T(robot.data.root_pos_w) + quat_apply(T(robot.data.root_quat_w), cmd) - origin

    mass = T(box.data.body_mass)[:, 0].clone()
    mu = wp.to_torch(box.root_view.get_material_properties())[:, 0, 0].clone().to(dev)

    # ---- per-env state ----
    c0, _, _, yb0 = box_pose()
    phase = torch.zeros(n, dtype=torch.long, device=dev)
    tph = torch.zeros(n, dtype=torch.long, device=dev)
    ref_p, ref_q, fin_p, fin_q = {}, {}, {}, {}
    for side, s in SIGN.items():
        p, q, _ = tcp(side)
        n_des = -s * yb0
        z_cur = quat_apply(q, up)
        fin_q[side] = edge_aligned_quat(q, n_des, yb0) if args.roll_align else quat_mul(shortest_arc(z_cur, n_des), q)
        ref_p[side], ref_q[side] = p.clone(), q.clone()
        f = c0 + s * (hy + args.pre_gap)[:, None] * yb0
        f[:, 2] = c0[:, 2] + hz + 0.10
        fin_p[side] = f
        if args.trace >= 0:
            i = args.trace
            print(f"[rule] {side} ready TCP {[round(v, 3) for v in p[i].tolist()]} z-axis "
                  f"{[round(v, 2) for v in z_cur[i].tolist()]} -> HIGH {[round(v, 3) for v in f[i].tolist()]}", flush=True)
    lift_dz = torch.zeros(n, device=dev)
    t_phase = torch.full((n, 6), -1, dtype=torch.long, device=dev)
    t_phase[:, HIGH] = 0
    t_succ = torch.full((n,), -1, dtype=torch.long, device=dev)
    f_settle = torch.zeros(n, 2, device=dev)
    margin_min = torch.full((n,), 9.0, device=dev)
    sat_steps = torch.zeros(n, device=dev)
    nonplate_max = torch.zeros(n, device=dev)
    tilt_max_lift = torch.zeros(n, device=dev)
    lift_max = torch.zeros(n, device=dev)
    box_xy0 = c0[:, :2].clone()
    box_x0 = (c0[:, 0] - T(robot.data.root_pos_w)[:, 0] + origin[:, 0]).clone()
    grip_dx, grip_dz, grip_dzlr = (torch.full((n,), float("nan"), device=dev) for _ in range(3))
    down_conv = torch.zeros(n, dtype=torch.bool, device=dev)
    down_zerr, down_mlim = torch.zeros(n, 2, device=dev), torch.zeros(n, 2, device=dev)
    down_jlim = torch.zeros(n, 2, dtype=torch.long, device=dev)
    pregrasp = args.pregrasp_gap is not None
    pg_done = torch.zeros(n, dtype=torch.bool, device=dev)
    pg_q = torch.zeros(n, T(robot.data.joint_pos).shape[1], device=dev)
    pg_box = torch.zeros(n, 7, device=dev)
    pg_err = torch.zeros(n, 2, 3, device=dev)
    pg_twist = torch.zeros(n, 2, device=dev)  # joint-7 turn that squares the plate edges (--snap_j7)
    pg_still = torch.zeros(n, dtype=torch.long, device=dev)
    pg_touch = torch.zeros(n, device=dev)
    face =(hy + args.pregrasp_gap) if pregrasp else (hy - args.depth)
    log = {k: [] for k in ("phase", "dq", "q", "qd", "tcp", "ref", "force", "box")}
    ref_v = {side: torch.zeros(n, device=dev) for side in SIGN}  # reference speed, with --a_ref

    for step in range(args.steps):
        c, _, tilt, y_b = box_pose()
        f = plate_forces()
        errs = {}
        for side in SIGN:
            p, q, _ = tcp(side)
            e_p, e_r = compute_pose_error(p, q, fin_p[side], fin_q[side], rot_error_type="axis_angle")
            if not args.fixed_roll:  # conv only gates HIGH/DOWN, where only the plate normal counts
                e_r = torch.cross(quat_apply(q, up), quat_apply(fin_q[side], up), dim=-1)
            errs[side] = (torch.norm(e_p, dim=-1), torch.norm(e_r, dim=-1),
                          torch.norm(fin_p[side] - ref_p[side], dim=-1))
        ref_done = torch.stack([errs[s][2] < 1e-4 for s in SIGN], -1).all(-1)
        conv = torch.stack([(errs[s][0] < 0.015) & (errs[s][1] < 0.08) for s in SIGN], -1).all(-1)

        # ---- transitions ----
        nxt = torch.full_like(phase, -1)
        nxt = torch.where((phase == HIGH) & ref_done & (conv | (tph >= 40)), DOWN, nxt)
        nxt = torch.where((phase == DOWN) & ref_done & (conv | (tph >= args.down_timeout)), APPROACH, nxt)
        if not pregrasp:
            nxt = torch.where((phase == APPROACH) & ref_done, SETTLE, nxt)
        nxt = torch.where((phase == SETTLE) & (tph >= args.settle), LIFT, nxt)
        nxt = torch.where((phase == LIFT) & ref_done, HOLD, nxt)
        go = nxt >= 0
        if go.any():
            for side, s in SIGN.items():
                to_down = go & (nxt == DOWN)
                fin_p[side][to_down, 2] = (c[:, 2] + args.grip_dz)[to_down]
                to_app = go & (nxt == APPROACH)
                inside = c + s * face[:, None] * y_b
                inside[:, 2] = c[:, 2] + args.grip_dz
                fin_p[side][to_app] = inside[to_app]
                # from here the plate roll is servoed too: it is the box pitch axis once the box is squeezed
                _, q_now, _ = tcp(side)
                if args.roll_align:
                    q_lock = edge_aligned_quat(q_now, -s * y_b, y_b)
                else:
                    q_lock = quat_mul(shortest_arc(quat_apply(q_now, up), -s * y_b), q_now)
                fin_q[side][to_app] = q_lock[to_app]
                ref_q[side][to_app] = q_lock[to_app]
            to_lift = go & (nxt == LIFT)
            lift_dz = torch.where(to_lift, goal_env()[:, 2] - c[:, 2], lift_dz)
            for side in SIGN:
                fin_p[side][to_lift, 2] += lift_dz[to_lift]
            phase = torch.where(go, nxt, phase)
            tph = torch.where(go, torch.zeros_like(tph), tph)
            t_phase[go, phase[go]] = step

        # until the first plate touches, keep the approach targets on the observed box pose
        track = (phase == APPROACH) & (f.amax(-1) < 1.0)
        if track.any():
            for side, s in SIGN.items():
                inside = c + s * face[:, None] * y_b
                inside[:, 2] = c[:, 2] + args.grip_dz
                fin_p[side][track] = inside[track]
        if pregrasp:
            # record once the reference has arrived and the arms have stopped (reached or blocked by a limit);
            # the residual is kept per box axis so the table can be filtered afterwards
            still = T(robot.data.joint_vel)[:, jids].abs().amax(-1) < 0.05
            pg_still = torch.where(still & (phase == APPROACH) & ref_done, pg_still + 1, torch.zeros_like(pg_still))
            rec = (pg_still >= 10) & ~pg_done
            if rec.any():
                err = []
                for sd, s in SIGN.items():
                    p, q, _ = tcp(sd)
                    d = fin_p[sd] - p
                    _, e_r = compute_pose_error(p, q, fin_p[sd], fin_q[sd], rot_error_type="axis_angle")
                    # normal: - = plate stopped further out than its target; z: - = plate above its target
                    err.append(torch.stack([s * (d * y_b).sum(-1), d[:, 2], torch.norm(e_r, dim=-1)], -1))
                err = torch.stack(err, 1)  # (n, side, [normal, z, rot])
                pg_q[rec] = T(robot.data.joint_pos)[rec]
                pg_box[rec] = torch.cat([c, T(box.data.root_quat_w)], -1)[rec]
                pg_err[rec] = err[rec]
                if args.snap_j7:
                    for k, sd in enumerate(SIGN):
                        _, q, _ = tcp(sd)
                        nrm, e = quat_apply(q, up), quat_apply(q, edge_l7)
                        u = up - (up * nrm).sum(-1, keepdim=True) * nrm  # world up in the plate plane
                        phi = torch.atan2((torch.cross(u, e, dim=-1) * nrm).sum(-1), (u * e).sum(-1))
                        pg_twist[rec, k] = -(torch.remainder(phi + math.pi / 4, math.pi / 2) - math.pi / 4)[rec]
                    for k, sd in enumerate(SIGN):
                        pg_q[rec, jids[names.index(f"{sd}_fr3_joint7")]] += pg_twist[rec, k]
                pg_touch[rec] = torch.maximum(f.amax(-1), nonplate_contact_forces(raw).sum(-1))[rec]
                pg_done |= rec
        if go.any() and (go & (nxt == LIFT)).any():
            k = go & (nxt == LIFT)
            mid = 0.5 * (tcp("left")[0] + tcp("right")[0])
            x_b = quat_apply(box_pose()[1], ex)
            grip_dx = torch.where(k, ((mid - c) * x_b).sum(-1), grip_dx)
            grip_dz = torch.where(k, mid[:, 2] - c[:, 2], grip_dz)
            grip_dzlr = torch.where(k, tcp("left")[0][:, 2] - tcp("right")[0][:, 2], grip_dzlr)
        k = go & (nxt == APPROACH)
        if k.any():
            down_conv = torch.where(k, conv, down_conv)
            for j, side in enumerate(SIGN):
                down_zerr[:, j] = torch.where(k, tcp(side)[0][:, 2] - fin_p[side][:, 2], down_zerr[:, j])
                qj = T(robot.data.joint_pos)[:, arms[side][2]]
                mj = torch.minimum(qj - lim[:, 0], lim[:, 1] - qj)
                down_jlim[:, j] = torch.where(k, mj.argmin(-1) + 1, down_jlim[:, j])
                down_mlim[:, j] = torch.where(k, mj.amin(-1), down_mlim[:, j])

        # ---- reference motion ----
        v = torch.zeros(n, device=dev)
        v = torch.where(phase <= DOWN, args.v_pre, v)
        v = torch.where(phase == APPROACH, args.v_app, v)
        v = torch.where(phase == LIFT, args.v_lift, v)
        # the squeeze force comes from the gain (blocked plate -> saturated step -> PD offset); in free motion a
        # gain of 2 over-corrects the light wrist joints every 50 ms step and the plates shake
        gain = torch.full((n,), args.gain, device=dev)
        if args.gain_free is not None:
            gain = torch.where((phase <= DOWN) | track, args.gain_free, gain)
        actions = torch.zeros(n, raw.action_manager.total_action_dim, device=dev)
        for side, (body, cols, dofs) in arms.items():
            vs = v
            if args.v_contact is not None:
                out = SIGN[side] * ((ref_p[side] - c) * y_b).sum(-1) - hy  # reference outside the face [m]
                vs = torch.where((phase == APPROACH) & (out < args.contact_zone), vs.clamp(max=args.v_contact), vs)
            if args.a_ref is not None:
                # trapezoidal profile: ramp up by a_ref, and slow down so the reference stops at its target
                # (within 2 a_ref dt^2 of it the last step lands exactly on it)
                dist = torch.norm(fin_p[side] - ref_p[side], dim=-1)
                vs = torch.minimum(torch.minimum(vs, ref_v[side] + args.a_ref * dt), torch.sqrt(2 * args.a_ref * dist))
                ref_v[side] = vs
            ref_p[side] = step_toward(ref_p[side], fin_p[side], (vs * dt)[:, None])
            ref_q[side] = rotate_toward(ref_q[side], fin_q[side], torch.full((n,), args.w_pre * dt, device=dev))
            p, q, r = tcp(side)
            e_p, e_r = compute_pose_error(p, q, ref_p[side], ref_q[side], rot_error_type="axis_angle")
            jac = T(robot.data.body_link_jacobian_w)[:, body - 1 if robot.is_fixed_base else body][:, :, dofs]
            jac = torch.cat([jac[:, :3] - skew(r) @ jac[:, 3:], jac[:, 3:]], 1)
            if not args.fixed_roll:
                # the plate is square: leave the rotation about its normal to the redundancy until the approach
                zc = quat_apply(q, up)
                free = (phase <= DOWN)[:, None, None]
                proj = torch.where(free, eye3 - zc[:, :, None] * zc[:, None, :], eye3.expand(n, 3, 3))
                e_r = (proj @ e_r[..., None])[..., 0]
                jac = torch.cat([jac[:, :3], proj @ jac[:, 3:]], 1)
            qj = T(robot.data.joint_pos)[:, dofs]
            push = torch.relu(0.3 - (qj - lim[:, 0])) - torch.relu(0.3 - (lim[:, 1] - qj))

            def solve(jm):
                jt = jm.transpose(1, 2)
                jpinv = jt @ torch.linalg.inv(jm @ jt + (0.05 ** 2) * eye6)
                d = gain[:, None] * (jpinv @ torch.cat([e_p, e_r], -1)[..., None])[..., 0]
                return d + ((eye7 - jpinv @ jm) @ (args.null * (q_ready[:, dofs] - qj) + args.lim_k * push)[..., None])[..., 0]

            dq = solve(jac)
            if args.limit_aware:
                # a joint the env clamps at its limit does not move; solving as if it did sends the plate elsewhere
                lo, hi = tlim[cols, 0], tlim[cols, 1]
                held = lambda d: ((qj <= lo + 0.01) & (d < 0)) | ((qj >= hi - 0.01) & (d > 0))  # noqa: E731
                blocked = held(dq)
                if blocked.any():
                    dq = torch.where(blocked.any(-1, keepdim=True), solve(jac * (~blocked)[:, None, :].float()), dq)
                    dq = dq.masked_fill(held(dq), 0.0)
            # tau = Kp (q_target - q) - Kd qd with q_target = q + dq, so this term is extra joint damping
            dq -= args.damp * T(robot.data.joint_vel)[:, dofs]
            sat_steps += (dq.abs() >= clip).any(-1).float() / 2
            # keep the IK direction: scale the whole joint step instead of letting the env clip joints one by one
            dq = dq * torch.clamp(clip / dq.abs().amax(-1, keepdim=True).clamp(min=1e-9), max=1.0)
            actions[:, cols] = dq / scale
            margin_min = torch.minimum(margin_min, torch.minimum(qj - lim[:, 0], lim[:, 1] - qj).amin(-1))

        if args.log:
            log["phase"].append(phase.clone())
            log["dq"].append(actions[:, :] * scale)
            log["q"].append(T(robot.data.joint_pos)[:, jids].clone())
            log["qd"].append(T(robot.data.joint_vel)[:, jids].clone())
            log["tcp"].append(torch.stack([tcp(sd)[0] for sd in SIGN], 1))
            log["ref"].append(torch.stack([ref_p[sd] for sd in SIGN], 1))
        env.step(actions)
        tph += 1
        if args.log:
            log["force"].append(plate_forces())
            log["box"].append(box_pose()[0])
        in_settle = phase == SETTLE
        f_settle = torch.where(in_settle[:, None], plate_forces(), f_settle)
        c, _, tilt, _ = box_pose()
        lifted = (c[:, 2] - hz) > 0.02
        tilt_max_lift = torch.where(lifted, torch.maximum(tilt_max_lift, tilt), tilt_max_lift)
        lift_max = torch.maximum(lift_max, c[:, 2] - hz)
        nonplate_max = torch.where(phase >= APPROACH, torch.maximum(nonplate_max, nonplate_contact_forces(raw).sum(-1)),
                                   nonplate_max)
        newly = (succ_term._count >= hold_steps) & (t_succ < 0)
        t_succ = torch.where(newly, torch.full_like(t_succ, step + 1), t_succ)
        if args.video:
            rgb = [T(cam.data.output["rgb"])[..., :3].cpu().numpy().astype(np.uint8) for cam in cams]
            fp, dzlr = plate_forces(), tcp("left")[0][:, 2] - tcp("right")[0][:, 2]
            frames.append([video_frame(
                [r[i] for r in rgb], list(VIEWS),
                f"env {i} (box x {float(box_x0[i]):.2f} m, {float(mass[i]):.2f} kg)  t {(step + 1) * dt:4.1f}s"
                f"  {NAMES[int(phase[i])]:8s}  lift {100 * float(c[i, 2] - hz[i]):5.1f} cm"
                f"  tilt {float(tilt[i]):4.1f} deg  F L/R {float(fp[i, 0]):4.1f}/{float(fp[i, 1]):4.1f} N"
                f"  plate zL-zR {100 * float(dzlr[i]):+4.1f} cm  {'SUCCESS' if t_succ[i] > 0 else ''}")
                for i in range(n)])

        i = args.trace
        if i >= 0 and step % 10 == 0:
            pl, pr = tcp("left")[0][i], tcp("right")[0][i]
            print(f"[trace] t={step:3d} {NAMES[int(phase[i])]:8s} L {[round(x, 3) for x in pl.tolist()]}"
                  f" R {[round(x, 3) for x in pr.tolist()]} F {[round(x, 1) for x in plate_forces()[i].tolist()]}"
                  f" box z {float(c[i, 2]):.3f} tilt {float(tilt[i]):.1f} goal z {float(goal_env()[i, 2]):.3f}"
                  f" cnt {int(succ_term._count[i])}", flush=True)
            for side, (_, _, dofs) in arms.items():
                qj = T(robot.data.joint_pos)[i, dofs]
                print(f"          {side} q {[round(x, 2) for x in qj.tolist()]} margin "
                      f"{[round(x, 2) for x in torch.minimum(qj - lim[:, 0], lim[:, 1] - qj).tolist()]}", flush=True)

    if pregrasp:
        tilt0 = torch.rad2deg(torch.acos(quat_apply(pg_box[:, 3:], up)[:, 2].clamp(-1, 1)))
        slide0 = torch.norm(pg_box[:, :2] - box_xy0, dim=-1)
        # plates may stop short (outward) or high where a joint limit blocks them; the policy closes the rest
        near = (pg_err[..., 0].abs() < 0.03).all(-1) & (pg_err[..., 1] > -0.04).all(-1) & (pg_err[..., 1] < 0.01).all(-1) \
            & (pg_err[..., 2] < 0.15).all(-1)
        ok = pg_done & near & (pg_touch < 0.5) & (slide0 < 0.01) & (tilt0 < 2.0)
        q_arm = pg_q[:, jids]
        mj = torch.minimum(q_arm - lim.repeat(2, 1)[:, 0], lim.repeat(2, 1)[:, 1] - q_arm).amin(-1)
        print(f"\n[pregrasp] gap {args.pregrasp_gap} m, grip_dz {args.grip_dz} m: reached {pg_done.float().mean():.3f},"
              f" valid (no touch, box still) {ok.float().mean():.3f} ({int(ok.sum())}/{n})"
              f" | residual (target - plate) normal median {100 * pg_err[pg_done][..., 0].median():.1f}"
              f" p90 {100 * pg_err[pg_done][..., 0].quantile(0.9):.1f} cm, z median {100 * pg_err[pg_done][..., 1].median():.1f}"
              f" p90 {100 * pg_err[pg_done][..., 1].quantile(0.9):.1f} cm, rot median {pg_err[pg_done][..., 2].median():.3f} rad"
              f" | min joint margin to FCI limits median {mj[ok].median():.3f}, min {mj[ok].min():.3f} rad")
        if args.snap_j7:
            tw = torch.rad2deg(pg_twist[ok])
            print(f"[pregrasp]   snap_j7: joint-7 turn L median {tw[:, 0].median():+.1f} deg [{tw[:, 0].min():+.1f}, {tw[:, 0].max():+.1f}],"
                  f" R median {tw[:, 1].median():+.1f} deg [{tw[:, 1].min():+.1f}, {tw[:, 1].max():+.1f}]")
        print(f"[pregrasp]   rejected: not stopped {(~pg_done).sum()}, residual {(pg_done & ~near).sum()}"
              f" (normal {(pg_done & ~(pg_err[..., 0].abs() < 0.03).all(-1)).sum()}, z {(pg_done & ~((pg_err[..., 1] > -0.04) & (pg_err[..., 1] < 0.01)).all(-1)).sum()},"
              f" rot {(pg_done & ~(pg_err[..., 2] < 0.15).all(-1)).sum()}), touch {(pg_done & (pg_touch >= 0.5)).sum()},"
              f" box moved/tilted {(pg_done & ((slide0 >= 0.01) | (tilt0 >= 2.0))).sum()}")
        for lo, hi in ((0.40, 0.50), (0.50, 0.55), (0.55, 0.65)):
            m = (box_x0 >= lo) & (box_x0 < hi)
            print(f"[pregrasp]   box x0 [{lo:.2f},{hi:.2f}): valid {ok[m].float().mean() if m.any() else float('nan'):.2f} ({int(m.sum())})")
        y0 = box_xy0[:, 1] - T(robot.data.root_pos_w)[:, 1] + origin[:, 1]
        for lo, hi in ((-0.11, -0.05), (-0.05, 0.05), (0.05, 0.11)):
            m = (y0 >= lo) & (y0 < hi)
            print(f"[pregrasp]   box y0 [{lo:+.2f},{hi:+.2f}): valid {ok[m].float().mean() if m.any() else float('nan'):.2f} ({int(m.sum())})")
        if args.save_table:
            # .npz (not .pt): small, framework-free, and not caught by the repo's *.pt ignore rule
            np.savez(
                args.save_table,
                joint_names=np.array(robot.joint_names),
                joint_pos=pg_q[ok].cpu().numpy(),
                box_pose=pg_box[ok].cpu().numpy(),  # env frame: pos (3) + quat (x, y, z, w)
                residual=pg_err[ok].cpu().numpy(),  # (side L/R, [normal, z, rot]) target - plate
                meta=np.array(str({"task": args.task, "gap": args.pregrasp_gap, "grip_dz": args.grip_dz,
                                   "box_dx": args.box_dx, "null": args.null, "lim_k": args.lim_k, "seed": args.seed,
                                   "roll_align": args.roll_align, "snap_j7": args.snap_j7,
                                   "box_size": list(cfg.scene.object.spawn.size)})),
            )
            print(f"[pregrasp] saved {int(ok.sum())} rows -> {args.save_table}")
        env.close()
        return

    # ---- report ----
    c, _, tilt, _ = box_pose()
    g = goal_env()
    dist = torch.norm(g - c, dim=-1)
    fin_ok = (dist < 0.10) & (tilt < 15.0) & (plate_forces() > 1.0).all(-1)
    succ = t_succ > 0
    succ5 = succ & (t_succ <= round(5.0 / dt))
    slide = torch.norm(c[:, :2] - box_xy0, dim=-1)
    print(f"\n[rule] {n} envs, mass {mass.min():.2f}~{mass.max():.2f} kg, box mu {mu.min():.2f}~{mu.max():.2f},"
          f" disturb {args.disturb}, depth {args.depth}, gain {args.gain}, clip ±{clip} rad, dt {dt}s")
    print(f"[rule] success (env term, held 0.5 s): {succ.float().mean():.3f} | within 5 s: {succ5.float().mean():.3f}"
          f" | still in success zone at end: {fin_ok.float().mean():.3f}")
    ts = t_succ[succ].float() * dt
    if len(ts):
        print(f"[rule] time to success [s]: median {ts.median():.2f}, p90 {ts.quantile(0.9):.2f}, max {ts.max():.2f}")
    fail = ~succ
    up_end = (c[:, 2] - hz) > 0.02
    print(f"[rule] failures {int(fail.sum())}: never lifted 2 cm {int((fail & (lift_max <= 0.02)).sum())},"
          f" dropped {int((fail & (lift_max > 0.02) & ~up_end).sum())},"
          f" held but tilt >= 15 deg {int((fail & up_end & (tilt >= 15)).sum())},"
          f" held upright but >= 10 cm from goal {int((fail & up_end & (tilt < 15) & (dist >= 0.10)).sum())},"
          f" other {int((fail & up_end & (tilt < 15) & (dist < 0.10)).sum())}")
    for k, nm in enumerate(NAMES):
        reached = t_phase[:, k] >= 0
        tp = t_phase[reached, k].float() * dt
        print(f"[rule]   phase {nm:8s} reached {reached.float().mean():.3f}"
              + (f", start median {tp.median():.2f}s" if len(tp) else ""))
    print(f"[rule] settle plate force L/R median {f_settle[:, 0].median():.1f}/{f_settle[:, 1].median():.1f} N"
          f" (min {f_settle.amin():.1f}) | max tilt while lifted median {tilt_max_lift.median():.1f}"
          f" p90 {tilt_max_lift.quantile(0.9):.1f} deg | box xy slide median {100 * slide.median():.1f} cm")
    print(f"[rule] min joint margin to FR3 limits median {margin_min.median():.3f} rad, min {margin_min.min():.3f}"
          f" | steps with a joint at the ±{clip} clip: median {sat_steps.median():.0f}/{args.steps}"
          f" | non-plate contact force max (from approach) median {nonplate_max.median():.1f} N,"
          f" share > 1 N {(nonplate_max > 1).float().mean():.2f}")
    print(f"[rule] grip at lift start (plate midpoint - box centre): |dx along box x| median {100 * grip_dx.abs().nanmedian():.1f}"
          f" p90 {100 * grip_dx.abs().nanquantile(0.9):.1f} cm, dz median {100 * grip_dz.nanmedian():.1f} cm,"
          f" |zL - zR| median {100 * grip_dzlr.abs().nanmedian():.1f} cm")
    print(f"[rule] end of DOWN: converged {down_conv.float().mean():.2f} | plate z - target L/R median"
          f" {100 * down_zerr[:, 0].median():.1f}/{100 * down_zerr[:, 1].median():.1f} cm, p90"
          f" {100 * down_zerr.quantile(0.9, dim=0)[0]:.1f}/{100 * down_zerr.quantile(0.9, dim=0)[1]:.1f} cm")
    for j, side in enumerate(SIGN):
        at = down_mlim[:, j] < 0.02
        cnt = torch.bincount(down_jlim[at, j], minlength=8)[1:].tolist()
        print(f"[rule]   {side}: at a FR3 limit (<0.02 rad) {at.float().mean():.2f}, which joint j1..j7 {cnt}")
    for lo, hi in ((0.40, 0.50), (0.50, 0.55), (0.55, 0.65)):
        m = (box_x0 >= lo) & (box_x0 < hi)
        print(f"[rule]   box x0 [{lo:.2f},{hi:.2f}): success {succ[m].float().mean() if m.any() else float('nan'):.2f} ({int(m.sum())}),"
              f" DOWN converged {down_conv[m].float().mean() if m.any() else float('nan'):.2f}")
    print("[rule] by mass / mu (success rate, n):")
    for lo, hi in ((0.0, 0.625), (0.625, 0.75), (0.75, 0.875), (0.875, 9.0)):
        m = (mass >= lo) & (mass < hi)
        row = []
        for mlo, mhi in ((0.0, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 9.0)):
            k = m & (mu >= mlo) & (mu < mhi)
            row.append(f"mu[{mlo:.1f},{mhi:.1f}) {succ[k].float().mean() if k.any() else float('nan'):.2f} ({int(k.sum())})")
        print(f"   mass [{lo:.3f},{hi:.3f}): " + " | ".join(row))
    if args.log:
        np.savez(args.log, joint_names=np.array(names), dt=dt, box_x0=box_x0.cpu().numpy(),
                 **{k: torch.stack(v, 1).cpu().numpy() for k, v in log.items()})  # (env, step, ...)
    if args.video:
        os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
        # one env after another, each run in full
        imageio.mimwrite(args.video, [f[i] for i in range(n) for f in frames], fps=round(1.0 / dt),
                         codec="libx264", quality=8, macro_block_size=1)
        print(f"[rule] video: {n} envs x {len(frames)} frames ({len(frames) * dt:.1f} s each) -> {args.video}")
    fail = torch.nonzero(~succ).flatten()[:12].tolist()
    for i in fail:
        print(f"   fail env {i}: phase {NAMES[int(phase[i])]}, mass {float(mass[i]):.2f}, mu {float(mu[i]):.2f},"
              f" box lift {100 * float(c[i, 2] - hz[i]):.1f} cm (goal {100 * float(g[i, 2] - hz[i]):.1f}),"
              f" tilt {float(tilt[i]):.1f}, F_settle {[round(x, 1) for x in f_settle[i].tolist()]},"
              f" nonplate {float(nonplate_max[i]):.1f} N, margin {float(margin_min[i]):.3f},"
              f" grip dx {100 * float(grip_dx[i]):.1f} dz {100 * float(grip_dz[i]):.1f} dzLR {100 * float(grip_dzlr[i]):.1f} cm,"
              f" box x0 {float(box_x0[i]):.2f} y0 {float(box_xy0[i, 1]):.2f}, DOWN zerr {[round(100 * x, 1) for x in down_zerr[i].tolist()]}"
              f" limit j{down_jlim[i].tolist()}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
