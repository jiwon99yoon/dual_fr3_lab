"""Rule-based cabinet book insert on ``Isaac-Cabinet-BookInsert-DualFR3-UMI-v0`` (2026-11-05).

Every arm command goes through the env's own arm action (same contract as the RL policy and the plate rule-based:
q_target = q_measured + clip(0.1 a, +-0.1 rad) at 20 Hz, explicit PD Kp80/Kd4). Each arm tracks a reference TCP pose
with DLS IK (gain * J+ e, whole step scaled so no joint exceeds the clip). The null space is pulled toward the joint
path ``solve_cabinet_book_start.py`` followed from the same start (stored per row in the start table): without it the
left arm leaves that path near door 20-25 deg and ends on joint 6's limit at door ~86 deg.

Left arm (TCP welded to the handle): the door reference angle grows at ``--w_door`` deg/s up to ``--door_deg`` (never
more than ``--lead_deg`` ahead of the measured door); the TCP reference is the start grasp frame rotated about the
hinge by that angle, so it follows the handle's arc.
Right arm (TCP welded to the book), reference = book pose (axes kept parallel to the base axes), along
``book_path_points(start)`` (the book starts standing on the table, one start-table row per env):
  0 LIFT    straight up to the lane height
  1 TO_X    to the pre-insert x (right of the door sweep)
  2 WAIT    until the door is open past ``--gate_deg``
  3 SLIDE   sideways to the pre-insert point in front of the cabinet (``BOOK_PRE_POS_B``)
  4 INSERT  straight along +x to ``BOOK_INSERT_POS_B`` (``--insert_depth`` past the opening)
  5 HOLD
Success (RL criterion, plan 7-3): book centre >= ``--success_depth`` past the opening, within 3 cm of the lane in y and
z, tilt < 10 deg.

    python scripts/tools/eval_dual_fr3_cabinet_book_rule_based.py --headless
    python scripts/tools/eval_dual_fr3_cabinet_book_rule_based.py --headless --all_rows --contacts
    python scripts/tools/eval_dual_fr3_cabinet_book_rule_based.py --headless --video logs/rule_based/cabinet_book/rule.mp4

``--all_rows``: one env per start-table row (every book start). The run also prints the RL env's view of it:
observation sizes, every reward term summed over the run, the success term, the largest handle force and the
largest net contact force per env contact sensor (plan 7, check before training).

``--video`` also writes ``<video>_sheet.png`` (frames every ``--sheet_s`` s).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Isaac-Cabinet-BookInsert-DualFR3-UMI-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--duration", type=float, default=12.0, help="seconds")
parser.add_argument("--all_rows", action="store_true", help="one env per start-table row")
parser.add_argument("--row", type=int, default=None, help="every env starts from this start-table row")
parser.add_argument("--success_depth", type=float, default=0.09, help="book centre past the opening")
parser.add_argument("--door_deg", type=float, default=90.0, help="door opening the left arm aims for")
parser.add_argument("--w_door", type=float, default=25.0, help="door reference speed, deg/s")
parser.add_argument("--lead_deg", type=float, default=5.0, help="max door reference lead over the measured door")
parser.add_argument("--gate_deg", type=float, default=75.0, help="book starts moving once the door is open this far")
parser.add_argument("--insert_depth", type=float, default=None, help="book centre past the opening at the end (cfg BOOK_INSERT_DEPTH)")
parser.add_argument("--v_move", type=float, default=0.2, help="m/s")
parser.add_argument("--near", type=float, default=0.02, help="book within this of the waypoint before the next phase (it sags 1-1.6 cm under its weight, DR mass up to 0.5 kg)")
parser.add_argument("--v_insert", type=float, default=0.12, help="m/s")
parser.add_argument("--gain", type=float, default=2.0)
parser.add_argument("--ki", type=float, default=0.05, help="right arm only: integral gain on the TCP error (book weight sag), ki * sum(e) <= 2 cm; all rows: 0 -> 120/121 (z sag 3.3 cm), 0.05 -> 121/121, 0.2 -> 59/121 (y overshoot)")
parser.add_argument("--null", type=float, default=0.5, help="null-space pull toward the solver path (start pose NPZ), per step")
parser.add_argument("--lim_k", type=float, default=0.5, help="null-space push away from limits (within 0.3 rad)")
parser.add_argument("--video", default="", help="mp4 path")
parser.add_argument("--contacts", action="store_true", help="contact sensors on the forearms/hands/book, print the largest")
parser.add_argument("--sheet_s", type=float, default=1.0)
parser.add_argument("--print_every", type=int, default=0, help="also print every this many steps (0: every 10)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.sensors.camera import CameraCfg  # noqa: E402
from isaaclab.utils.math import compute_pose_error, quat_apply, quat_from_angle_axis, quat_mul  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import dual_fr3_lab.tasks  # noqa: E402, F401
from dual_fr3_lab.robots import UMI_TCP_OFFSET_POS, UMI_TCP_OFFSET_ROT  # noqa: E402
from dual_fr3_lab.tasks.manager_based.cabinet_book_insert import cabinet_book_insert_env_cfg as C  # noqa: E402
from dual_fr3_lab.tasks.manager_based.cabinet_book_insert import mdp  # noqa: E402

LIFT, TO_X, WAIT, SLIDE, INSERT, HOLD = range(6)
PHASES = ("LIFT", "TO_X", "WAIT", "SLIDE", "INSERT", "HOLD")
PHASE_WP = (1, 2, 2, 3, 4, 4)  # book_path_points index each phase moves toward
VIEWS = {  # eye, target in base frame
    "overview": ((1.5, -1.3, 1.1), (0.6, 0.0, 0.2)),
    "top": ((0.6, 0.0, 2.0), (0.6, 0.0, 0.2)),
}


arm_path = C.arm_link_path


CAB = {"carcass": "{ENV_REGEX_NS}/Cabinet/Geometry/drawer", "door": "{ENV_REGEX_NS}/Cabinet/Geometry/drawer/door",
       "handle": "{ENV_REGEX_NS}/Cabinet/Geometry/drawer/door/handle", "table": "{ENV_REGEX_NS}/Table"}
SENSED = {  # sensor -> (body path, {filter name: path})
    **{f"R_{lk}": (arm_path("right", lk), {**CAB, "book": "{ENV_REGEX_NS}/Book",
                                           **{f"L_link{k}": arm_path("left", f"link{k}") for k in (4, 5, 6, 7)}})
       for lk in ("link4", "link5", "link6", "link7", "leftfinger", "rightfinger")},
    **{f"L_{lk}": (arm_path("left", lk), {**CAB, "book": "{ENV_REGEX_NS}/Book"})
       for lk in ("link4", "link5", "link6", "link7", "leftfinger", "rightfinger")},
    "book": ("{ENV_REGEX_NS}/Book", CAB),
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


def rot_z(angle):
    """Quaternion (x, y, z, w) for a rotation about +z, batched."""
    axis = torch.zeros(angle.shape[0], 3, device=angle.device)
    axis[:, 2] = 1.0
    return quat_from_angle_axis(angle, axis)


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    table = np.load(C.START_TABLE_PATH)
    cfg.scene.num_envs = len(table["joint_pos"]) if args.all_rows else args.num_envs
    cfg.events.reset_start.params["sequential"] = args.all_rows
    cfg.events.reset_start.params["fixed_row"] = args.row
    cfg.episode_length_s = max(cfg.episode_length_s, args.duration + 1.0)  # no time-out reset mid-run
    if args.video:
        for name in VIEWS:
            setattr(cfg.scene, f"cam_{name}", CameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Cam_{name}", update_period=0.0, width=800, height=600, data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.01, 20.0)),
            ))
    if args.contacts:
        cfg.scene.book.spawn.activate_contact_sensors = True
        for name, (path, filt) in SENSED.items():
            setattr(cfg.scene, f"contact_{name}", ContactSensorCfg(prim_path=path, filter_prim_paths_expr=list(filt.values())))
    env = gym.make(args.task, cfg=cfg).unwrapped
    env.reset()
    dev, n = env.device, env.num_envs
    dt = env.step_dt
    robot, cabinet, book = env.scene["robot"], env.scene["cabinet"], env.scene["book"]
    base = env.scene.env_origins + torch.tensor([0.0, 0.0, C.ROBOT_BASE_Z], device=dev)

    term = env.action_manager.get_term("arm_action")
    names = list(term._joint_names)
    jids = term._joint_ids if not isinstance(term._joint_ids, slice) else list(range(len(names)))
    scale = float(term.cfg.scale)
    clip = float(list(term.cfg.clip.values())[0][1])
    tlim = torch.tensor(term.cfg.target_limits, device=dev)
    arms = {}
    for side in ("left", "right"):
        cols = [k for k, nm in enumerate(names) if nm.startswith(side)]
        arms[side] = (robot.body_names.index(f"{side}_fr3_link7"), cols, [jids[k] for k in cols])
    off_p = torch.tensor(UMI_TCP_OFFSET_POS, device=dev).repeat(n, 1)
    off_q = torch.tensor(UMI_TCP_OFFSET_ROT, device=dev).repeat(n, 1)
    eye6, eye7 = torch.eye(6, device=dev), torch.eye(7, device=dev)
    door_id = cabinet.body_names.index("door")
    handle_id = cabinet.body_names.index("handle")

    def tcp(side):
        b = arms[side][0]
        p, q = T(robot.data.body_pos_w)[:, b], T(robot.data.body_quat_w)[:, b]
        r = quat_apply(q, off_p)
        return p + r, quat_mul(q, off_q), r

    def door_angle():
        return T(cabinet.data.joint_pos)[:, 0]  # opening is negative

    def compose(p, q, lp, lq):
        lp = torch.tensor(lp, device=dev).repeat(n, 1)
        lq = torch.tensor(lq, device=dev).repeat(n, 1)
        return p + quat_apply(q, lp), quat_mul(q, lq)

    # hinge (world) and the left grasp frame at door angle 0
    dp, dq_ = T(cabinet.data.body_pos_w)[:, door_id], T(cabinet.data.body_quat_w)[:, door_id]
    hinge = dp + quat_apply(dq_, torch.tensor(C.DOOR_HINGE_IN_DOOR, device=dev).repeat(n, 1))
    th0 = door_angle().clone()
    hp, hq = T(cabinet.data.body_pos_w)[:, handle_id], T(cabinet.data.body_quat_w)[:, handle_id]
    unrot = rot_z(-th0)
    hp0, hq0 = hinge + quat_apply(unrot, hp - hinge), quat_mul(unrot, hq)
    grasp0_p, grasp0_q = compose(hp0, hq0, C.LEFT_GRASP_POS, C.LEFT_GRASP_ROT)

    center = base + torch.tensor(C.CABINET_INNER_CENTER_B, device=dev)
    book_ref_p = T(book.data.root_pos_w).clone()
    book_ref_q = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=dev).repeat(n, 1)
    row = env.book_start_row.clone()
    start_b = torch.tensor(table["book_start_b"], device=dev, dtype=torch.float32)[row]
    wp = torch.stack([torch.tensor(C.book_path_points(s), device=dev) for s in start_b.tolist()]) + base[:, None]
    if args.insert_depth is not None:
        wp[:, 4, 0] = base[:, 0] + C.CABINET_OPENING_X_B + args.insert_depth
    lane = wp[:, 4] - base  # inserted book centre (base frame): x goal, y/z = lane
    # joint paths the start solver followed from these starts (null-space references)
    l_door = torch.tensor(table["left_path_door"], device=dev, dtype=torch.float32)
    l_q = torch.tensor(table["left_path_q"], device=dev, dtype=torch.float32)
    r_book = torch.tensor(table["right_path_book"], device=dev, dtype=torch.float32)[row]  # [n, W, 3]
    r_q = torch.tensor(table["right_path_q"], device=dev, dtype=torch.float32)[row]  # [n, W, 7]

    def null_ref(side, phi_r, book_b):
        if side == "left":  # linear in the door angle
            x = torch.clamp(phi_r, l_door[0], l_door[-1]) / (l_door[1] - l_door[0])
            i0 = torch.clamp(x.floor().long(), max=len(l_door) - 2)
            a = (x - i0)[:, None]
            return l_q[i0] * (1 - a) + l_q[i0 + 1] * a
        near_w = torch.argmin(torch.cdist(book_b[:, None], r_book)[:, 0], dim=-1)  # nearest waypoint
        return r_q[torch.arange(n, device=dev), near_w]
    phase = torch.full((n,), LIFT, dtype=torch.long, device=dev)
    phi_ref = (-th0).clone()
    phi_goal, w_door, lead = (math.radians(v) for v in (args.door_deg, args.w_door, args.lead_deg))

    steps = int(args.duration / dt)
    t_phase = torch.full((n, len(PHASES)), float("nan"), device=dev)
    t_phase[:, LIFT] = 0.0
    err_max = {s: torch.zeros(n, device=dev) for s in ("left", "right")}
    err_int = {s: torch.zeros(n, 6, device=dev) for s in ("left", "right")}
    margin_min = torch.full((n,), 9.0, device=dev)
    sat_steps = torch.zeros(n, device=dev)
    frames = []
    contact_max = {}  # (sensor, filter) -> per-env max force [n], env-0 (first time > 5 N, phase then)

    def render():
        for name, (eye, tgt) in VIEWS.items():
            cam = env.scene[f"cam_{name}"]
            cam.set_world_poses_from_view(torch.tensor([eye], device=dev) + base[:1], torch.tensor([tgt], device=dev) + base[:1])
        for _ in range(2):
            env.sim.render()
        env.scene.update(env.physics_dt)

    if args.video:
        render()
    obs_dims = {g: tuple(v.shape[1:]) for g, v in env.observation_manager.compute().items()}
    handle_max = torch.zeros(n, device=dev)
    handle_t = torch.zeros(n, device=dev)
    rl_contact_max = torch.zeros(n, len(C.CONTACT_SENSORS), device=dev)
    t_success = torch.full((n,), float("nan"), device=dev)

    for k in range(steps):
        phi = -door_angle()
        # ---- references ----
        phi_ref = torch.minimum(torch.minimum(phi_ref + w_door * dt, torch.full_like(phi_ref, phi_goal)), phi + lead)
        rz = rot_z(-phi_ref)
        ref = {"left": (hinge + quat_apply(rz, grasp0_p - hinge), quat_mul(rz, grasp0_q))}
        tgt = wp[torch.arange(n, device=dev), torch.tensor(PHASE_WP, device=dev)[phase]]
        v = torch.where(phase == INSERT, args.v_insert, args.v_move)
        moving = (phase != WAIT) & (phase != HOLD)
        book_ref_p = torch.where(moving[:, None], step_toward(book_ref_p, tgt, (v * dt)[:, None]), book_ref_p)
        at_ref = torch.norm(book_ref_p - tgt, dim=-1) < 1e-4
        near = torch.norm(T(book.data.root_pos_w) - tgt, dim=-1) < args.near
        # every check uses this step's phase (one phase per step, no fall-through)
        adv = (((phase == LIFT) | (phase == TO_X) | (phase == SLIDE)) & at_ref & near) | ((phase == INSERT) & at_ref)
        adv |= (phase == WAIT) & (phi >= math.radians(args.gate_deg))
        phase = torch.where(adv, phase + 1, phase)
        for ph in range(1, len(PHASES)):
            t_phase[:, ph] = torch.where((phase == ph) & torch.isnan(t_phase[:, ph]), k * dt, t_phase[:, ph])
        ref["right"] = compose(book_ref_p, book_ref_q, C.RIGHT_GRASP_POS, C.RIGHT_GRASP_ROT)

        # ---- DLS IK -> arm action ----
        actions = torch.zeros(n, env.action_manager.total_action_dim, device=dev)
        for side in ("left", "right"):
            body, cols, dofs = arms[side]
            p, q, r = tcp(side)
            e_p, e_r = compute_pose_error(p, q, ref[side][0], ref[side][1], rot_error_type="axis_angle")
            err_max[side] = torch.maximum(err_max[side], torch.norm(e_p, dim=-1))
            jac = T(robot.data.body_link_jacobian_w)[:, body - 1][:, :, dofs]
            jac = torch.cat([jac[:, :3] - skew(r) @ jac[:, 3:], jac[:, 3:]], 1)
            qj = T(robot.data.joint_pos)[:, dofs]
            lo, hi = tlim[cols, 0], tlim[cols, 1]
            push = torch.relu(0.3 - (qj - lo)) - torch.relu(0.3 - (hi - qj))
            jt = jac.transpose(1, 2)
            jpinv = jt @ torch.linalg.inv(jac @ jt + (0.05**2) * eye6)
            e = torch.cat([e_p, e_r], -1)
            # the left reference leads the door on purpose, so only the right arm (book weight) gets the integral
            ki = args.ki if side == "right" else 0.0
            err_int[side] = (err_int[side] + e).clamp(-0.02 / max(ki, 1e-9), 0.02 / max(ki, 1e-9))
            dq = (jpinv @ (args.gain * e + ki * err_int[side])[..., None])[..., 0]
            q_ref = null_ref(side, phi_ref, book_ref_p - base)
            dq += ((eye7 - jpinv @ jac) @ (args.null * (q_ref - qj) + args.lim_k * push)[..., None])[..., 0]
            sat_steps += (dq.abs() >= clip).any(-1).float() / 2
            dq = dq * torch.clamp(clip / dq.abs().amax(-1, keepdim=True).clamp(min=1e-9), max=1.0)
            actions[:, cols] = dq / scale
            margin_min = torch.minimum(margin_min, torch.minimum(qj - lo, hi - qj).amin(-1))
        env.step(actions)
        hf = mdp.handle_force_norm(env)
        handle_t = torch.where(hf > handle_max, (k + 1) * dt, handle_t)
        handle_max = torch.maximum(handle_max, hf)
        rl_contact_max = torch.maximum(rl_contact_max, mdp.contact_force_norms(env, C.CONTACT_SENSORS))
        ok_now = mdp.insert_success(env, **C.SUCCESS_PARAMS).bool()
        t_success = torch.where(ok_now & torch.isnan(t_success), (k + 1) * dt, t_success)
        if args.contacts:
            for name, (_, filt) in SENSED.items():
                f = torch.norm(T(env.scene[f"contact_{name}"].data.force_matrix_w), dim=-1).amax(1)  # [n, filters]
                for j, fn in enumerate(filt):
                    old = contact_max.get((name, fn), (torch.zeros(n, device=dev), None))
                    first = old[1] if old[1] is not None or f[0, j] <= 5.0 else ((k + 1) * dt, PHASES[int(phase[0])])
                    contact_max[(name, fn)] = (torch.maximum(old[0], f[:, j]), first)
        if args.video:
            render()
            frames.append(np.concatenate(
                [T(env.scene[f"cam_{nm}"].data.output["rgb"])[0].cpu().numpy()[..., :3].astype(np.uint8) for nm in VIEWS], 1))
        if k % 10 == 0 or k == steps - 1 or (args.print_every and k % args.print_every == 0):
            bp = (T(book.data.root_pos_w)[0] - base[0]).cpu().numpy()
            print(f"[rule] t {k * dt:4.1f} s | phase {PHASES[int(phase[0])]:6s} | door {math.degrees(phi[0]):5.1f} deg "
                  f"(ref {math.degrees(phi_ref[0]):5.1f}) | book (base) {np.round(bp, 3).tolist()} | "
                  f"handle F {hf[0]:.0f} N | TCP err L {100 * torch.norm(tcp('left')[0][0] - ref['left'][0][0]):.1f} cm "
                  f"R {100 * torch.norm(tcp('right')[0][0] - ref['right'][0][0]):.1f} cm | q - path q: L "
                  f"{np.round((T(robot.data.joint_pos)[0, arms['left'][2]] - null_ref('left', phi_ref, book_ref_p - base)[0]).cpu().numpy(), 2).tolist()} "
                  f"R max {(T(robot.data.joint_pos)[0, arms['right'][2]] - null_ref('right', phi_ref, book_ref_p - base)[0]).abs().max():.2f} rad")

    # ---- result ----
    bp = T(book.data.root_pos_w) - center
    bq = T(book.data.root_quat_w)
    tilt = torch.rad2deg(2 * torch.acos(bq[:, 3].abs().clamp(max=1.0)))
    half = torch.tensor(C.BOOK_SIZE, device=dev) * 0.5
    front = -C.CABINET_INNER_HALF  # opening plane, relative to the centre
    depth = bp[:, 0] - front  # book centre past the opening
    inside = ((bp[:, 0] - half[0] >= front) & (bp[:, 1].abs() <= C.CABINET_INNER_HALF - half[1])
              & (bp[:, 2].abs() <= C.CABINET_INNER_HALF - half[2]))
    lane_err = (T(book.data.root_pos_w) - base)[:, 1:] - lane[:, 1:]
    success = (depth >= args.success_depth) & (lane_err.abs() < 0.03).all(-1) & (tilt < 10.0)
    phi = torch.rad2deg(-door_angle())
    print("[rule] ---- result ----")
    for i in range(n):
        if n > 4 and bool(success[i]):
            continue  # many envs: only the failures in detail
        tp = ", ".join(f"{PHASES[j]} {t_phase[i, j]:.2f}" for j in range(len(PHASES)) if not torch.isnan(t_phase[i, j]))
        print(f"[rule] env {i} (book start {np.round(start_b[i].cpu().numpy(), 3).tolist()}): door {phi[i]:.1f} deg | "
              f"book centre - cabinet centre {np.round(bp[i].cpu().numpy(), 3).tolist()} | depth past opening "
              f"{100 * depth[i]:.1f} cm | lane error y/z {np.round(100 * lane_err[i].cpu().numpy(), 1).tolist()} cm | "
              f"tilt {tilt[i]:.1f} deg | success {bool(success[i])} | phase start (s): {tp}")
        print(f"[rule] env {i}: max TCP error L {100 * err_max['left'][i]:.1f} cm, R {100 * err_max['right'][i]:.1f} cm | "
              f"min joint margin {margin_min[i]:.3f} rad | saturated steps {int(sat_steps[i])}")
        if n <= 4:
            for side in ("left", "right"):
                _, cols, dofs = arms[side]
                qj = T(robot.data.joint_pos)[i, dofs]
                lo, hi = tlim[cols, 0], tlim[cols, 1]
                print(f"[rule] env {i}: {side} q {np.round(qj.cpu().numpy(), 3).tolist()} | margin "
                      f"{np.round(torch.minimum(qj - lo, hi - qj).cpu().numpy(), 3).tolist()}")
    t_hold = t_phase[:, HOLD]
    print(f"[rule] success {int(success.sum())}/{n} (depth >= {100 * args.success_depth:.0f} cm, lane < 3 cm, tilt < 10 deg) | "
          f"fully inside {int(inside.sum())}/{n} | door {phi.min():.1f}-{phi.max():.1f} deg | depth "
          f"{100 * depth.min():.1f}-{100 * depth.max():.1f} cm | tilt max {tilt.max():.1f} deg | HOLD reached "
          f"{int((~torch.isnan(t_hold)).sum())}/{n}, at {np.nanmin(t_hold.cpu().numpy()):.2f}-{np.nanmax(t_hold.cpu().numpy()):.2f} s | "
          f"min joint margin {margin_min.min():.3f} rad | max TCP error L {100 * err_max['left'].max():.1f} R "
          f"{100 * err_max['right'].max():.1f} cm")
    # ---- RL env view (plan 7) ----
    rm = env.reward_manager
    print(f"[rule] RL obs dims {obs_dims} | success term at the end {int(mdp.insert_success(env, **C.SUCCESS_PARAMS).sum())}/{n}, "
          f"first success at {np.nanmin(t_success.cpu().numpy()):.2f}-{np.nanmax(t_success.cpu().numpy()):.2f} s")
    print("[rule] RL reward terms summed over the run (weighted, x dt; mean / min / max over envs):")
    for name in rm.active_terms:
        s = rm._episode_sums[name]
        print(f"[rule]   {name:15s} {s.mean():9.3f} {s.min():9.3f} {s.max():9.3f}")
    ih = int(handle_max.argmax())
    print(f"[rule] handle force max {handle_max.max():.1f} N (env {ih}, t {handle_t[ih]:.2f} s), mean of env maxima "
          f"{handle_max.mean():.1f} N, envs > {C.HANDLE_FORCE_LIMIT:.0f} N: {int((handle_max > C.HANDLE_FORCE_LIMIT).sum())}/{n}")
    print("[rule] RL contact sensors, max net force over the run (max / mean over envs, N): " + ", ".join(
        f"{nm[8:]} {rl_contact_max[:, j].max():.1f}/{rl_contact_max[:, j].mean():.1f}" for j, nm in enumerate(C.CONTACT_SENSORS)))
    if args.contacts:
        hits = sorted(((float(v[0].max()), k, v) for k, v in contact_max.items() if float(v[0].max()) > 1.0),
                      key=lambda h: h[0], reverse=True)
        print(f"[rule] contacts > 1 N (max over the run and envs): {len(hits)}")
        for fmax, (name, fn), (per_env, first) in hits[:15]:
            when = f", env 0 first > 5 N at {first[0]:.2f} s ({first[1]})" if first else ""
            print(f"[rule]   {name:14s} vs {fn:8s}: max {fmax:7.1f} N (env {int(per_env.argmax())}), "
                  f"> 5 N in {int((per_env > 5.0).sum())}/{n} envs{when}")

    if args.video:
        os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
        imageio.mimwrite(args.video, frames, fps=round(1.0 / dt))
        every = max(1, int(round(args.sheet_s / dt)))
        pick = frames[::every] + ([frames[-1]] if (len(frames) - 1) % every else [])
        cols_n = 3
        rows = [np.concatenate(pick[i:i + cols_n] + [np.zeros_like(pick[0])] * (cols_n - len(pick[i:i + cols_n])), 1)
                for i in range(0, len(pick), cols_n)]
        sheet = np.concatenate(rows, 0)[::2, ::2]
        sheet_path = os.path.splitext(args.video)[0] + "_sheet.png"
        imageio.imwrite(sheet_path, sheet)
        print("[rule] wrote", args.video, "and", sheet_path)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
