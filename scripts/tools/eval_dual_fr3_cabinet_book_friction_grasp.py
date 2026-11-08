# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cabinet book checkpoint with the sim hand welds replaced by a finger-friction grasp, as on the real robot.

Training welds the left UMI TCP to the handle and the right TCP to the book (fixed joints), so a left-arm command that
the door does not allow just builds force in the weld and the hand stays put. Real: the fingers hold the bar by friction.
Here a hand in ``friction`` mode has no weld and its fingers are commanded closed (target 0, the stiff finger PD
squeezes about 30 N per finger, like the 30 N franka grasp). With ``--obs fk`` the policy sees what the real FK node
publishes: door angle from the left TCP orbiting the calibrated hinge, book pose from the right TCP and the fixed grasp
transform (so a slipping hand shows up as a wrong observation, as on the robot).

``free``: no weld and the fingers stay where they are. With ``--no_cabinet`` / ``--no_book`` the cabinet / book is
moved out of reach, as in the real "motion only" test (no handle, no book): the door angle the policy sees then comes
only from where its own left hand goes.

The grippers close while the arms hold (``--settle_s``), then the policy runs one episode. Printed per run: success,
door angle (true and FK), hand slip from the grasp frame, and left-arm activity after the door is open (joint speed,
action changes, TCP motion), the "sloshing" seen on the robot.

    # as trained (welds, true observations), then both hands on friction with FK observations
    python scripts/tools/eval_dual_fr3_cabinet_book_friction_grasp.py --checkpoint <run>/model_2998.pt \\
        --left weld --right weld --obs true --headless
    python scripts/tools/eval_dual_fr3_cabinet_book_friction_grasp.py --checkpoint <run>/model_2998.pt --headless
    # real "motion only" test: no cabinet, no book, FK observations
    python scripts/tools/eval_dual_fr3_cabinet_book_friction_grasp.py --checkpoint <run>/model_2998.pt \\
        --left free --right free --no_cabinet --no_book --headless
    # video of one start row (env 0)
    python scripts/tools/eval_dual_fr3_cabinet_book_friction_grasp.py --checkpoint <run>/model_2998.pt --num_envs 1 \\
        --row 60 --video logs/cabinet_friction/2998_friction.mp4 --headless --enable_cameras
"""

import argparse
import importlib.metadata as metadata

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", default="Isaac-Cabinet-BookInsert-DualFR3-UMI-Table75-DoorFirst-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=121, help="env i starts on start-table row i (121 rows)")
parser.add_argument("--row", type=int, default=None, help="every env starts on this row")
parser.add_argument("--left", choices=("weld", "friction", "free"), default="friction")
parser.add_argument("--right", choices=("weld", "friction", "free"), default="friction")
parser.add_argument("--no_cabinet", action="store_true", help="cabinet out of reach (needs --left free)")
parser.add_argument("--no_book", action="store_true", help="book out of reach (needs --right free)")
parser.add_argument("--save", default="", help="npz with the per-step records (T, N, ...)")
parser.add_argument("--grip_mu", type=float, default=None,
                    help="friction coefficient (static = dynamic) of the fingers, the handle and the book; default: the USD's "
                         "finger material, handle 0.8 (MJCF), book 0.6/0.5")
parser.add_argument("--door_stiffness", type=float, default=None,
                    help="door hinge spring toward closed (N m/rad); training: 0.2 (MJCF / door_hinge actuator)")
parser.add_argument("--left_hold_deg", type=float, default=None,
                    help="deploy guard: once the FK door angle reaches this, the left-arm action is 0 (hold the measured "
                         "joints), as the welded left arm stayed still in training once the door was open")
parser.add_argument("--obs", choices=("fk", "true"), default="fk",
                    help="door angle / book pose observation: fk = real FK node, true = sim state (training)")
parser.add_argument("--hinge", type=float, nargs=2, default=(0.665, 0.2555),
                    help="FK node hinge_position_base x y (75 cm layout)")
parser.add_argument("--closed_tcp", type=float, nargs=2, default=(0.6241, -0.035),
                    help="FK node left_tcp_closed_position_base x y (75 cm layout)")
parser.add_argument("--door_sign", type=float, default=-1.0, help="FK node door_angle_sign")
parser.add_argument("--finger_target", type=float, default=0.0, help="finger joint target of a friction hand (m)")
parser.add_argument("--settle_s", type=float, default=0.5, help="grippers close with the arms held before the policy")
parser.add_argument("--after_s", type=float, default=2.0, help="activity window starts this long into the policy run")
parser.add_argument("--video", default="", help="mp4 path (env 0)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.no_cabinet and args.left != "free":
    parser.error("--no_cabinet needs --left free")
if args.no_book and args.right != "free":
    parser.error("--no_book needs --right free")
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab.envs.mdp as base_mdp  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.managers import EventTermCfg, SceneEntityCfg  # noqa: E402
from isaaclab.sensors.camera import CameraCfg  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, subtract_frame_transforms  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import dual_fr3_lab.tasks  # noqa: E402, F401
from dual_fr3_lab.robots import UMI_TCP_OFFSET_POS, UMI_TCP_OFFSET_ROT  # noqa: E402
from dual_fr3_lab.tasks.manager_based.cabinet_book_insert import cabinet_book_insert_env_cfg as C  # noqa: E402
from dual_fr3_lab.tasks.manager_based.cabinet_book_insert import mdp  # noqa: E402

VIEWS = {  # world offsets from the env origin for the 45 cm layout (record_views), shifted with the task goal
    "robot_side": ((0.40, -1.9, 1.35), (0.62, 0.0, 0.88)),
    "base_front": ((0.0, 0.0, 1.15), (0.95, 0.10, 0.64)),
}


def T(x):
    return x.torch if hasattr(x, "torch") else x


def tcp_b(env, side: str):
    """UMI TCP pose in the robot base frame (link7 FK + fixed offset), as the real FK node computes it."""
    robot = env.scene["robot"]
    i = robot.body_names.index(f"{side}_fr3_link7")
    p, q = T(robot.data.body_pos_w)[:, i], T(robot.data.body_quat_w)[:, i]
    off_p = torch.tensor(UMI_TCP_OFFSET_POS, device=env.device).expand_as(p)
    off_q = torch.tensor(UMI_TCP_OFFSET_ROT, device=env.device).expand_as(q)
    return subtract_frame_transforms(T(robot.data.root_pos_w), T(robot.data.root_quat_w), p + quat_apply(q, off_p),
                                     quat_mul(q, off_q))


def door_angle_fk_rad(env) -> torch.Tensor:
    """cabinet_book_insert_fk_node: left TCP orbiting the calibrated hinge (xy), relative to the closed-door TCP."""
    p, _ = tcp_b(env, "left")
    h = torch.tensor(args.hinge, device=env.device)
    v0 = torch.tensor(args.closed_tcp, device=env.device) - h
    v = p[:, :2] - h
    return args.door_sign * torch.atan2(v0[0] * v[:, 1] - v0[1] * v[:, 0], v @ v0)


def book_pose_fk(env):
    """cabinet_book_insert_fk_node: book = right TCP composed with the inverse of the fixed book grasp."""
    p, q = tcp_b(env, "right")
    g_p = torch.tensor(C.RIGHT_GRASP_POS, device=env.device).expand_as(p)
    g_q = torch.tensor(C.RIGHT_GRASP_ROT, device=env.device).expand_as(q)
    book_q = quat_mul(q, quat_inv(g_q))
    book_p = p - quat_apply(book_q, g_p)
    return book_p, torch.where(book_q[:, 3:4] < 0, -book_q, book_q)


def door_angle_fk(env):
    return door_angle_fk_rad(env)[:, None]


def book_pos_fk(env, target_pos_b):
    return book_pose_fk(env)[0] - torch.tensor(target_pos_b, device=env.device)


def book_quat_fk(env):
    return book_pose_fk(env)[1]


def grasp_slip(env, side: str) -> torch.Tensor:
    """Distance (m) from the TCP to where the weld would hold it: handle (left) / book (right) pose x grasp frame."""
    if side == "left":
        cab = env.scene["cabinet"]
        i = cab.body_names.index("handle")
        o_p, o_q, g = T(cab.data.body_pos_w)[:, i], T(cab.data.body_quat_w)[:, i], C.LEFT_GRASP_POS
    else:
        book = env.scene["book"]
        o_p, o_q, g = T(book.data.root_pos_w), T(book.data.root_quat_w), C.RIGHT_GRASP_POS
    robot = env.scene["robot"]
    want_w = o_p + quat_apply(o_q, torch.tensor(g, device=env.device).expand_as(o_p))
    want_b = subtract_frame_transforms(T(robot.data.root_pos_w), T(robot.data.root_quat_w), want_w)[0]
    return torch.norm(tcp_b(env, side)[0] - want_b, dim=-1)


def stat(x, fmt="{:.1f}"):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "-"
    return f"median {fmt.format(np.median(x))}, max {fmt.format(x.max())}"


def main():
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point"),
                                             metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.events.reset_start.params["sequential"] = True
    if args.row is not None:
        env_cfg.events.reset_start.params["fixed_row"] = args.row
    episode_s = env_cfg.episode_length_s
    env_cfg.episode_length_s = episode_s + args.settle_s + 1.0  # no time-out reset during the run
    if args.left != "weld":
        env_cfg.events.attach_left_handle = None
    if args.right != "weld":
        env_cfg.events.attach_right_book = None
    if args.grip_mu is not None:
        for name, asset in (("fingers", SceneEntityCfg("robot", body_names=".*finger")),
                            ("handle", SceneEntityCfg("cabinet", body_names="handle")), ("book", SceneEntityCfg("book"))):
            setattr(env_cfg.events, f"grip_mu_{name}", EventTermCfg(
                func=base_mdp.randomize_rigid_body_material, mode="startup",
                params={"asset_cfg": asset, "static_friction_range": (args.grip_mu, args.grip_mu),
                        "dynamic_friction_range": (args.grip_mu, args.grip_mu), "restitution_range": (0.0, 0.0),
                        "num_buckets": 1}))
    if args.door_stiffness is not None:
        acts = dict(env_cfg.scene.cabinet.actuators)
        acts["door_hinge"] = acts["door_hinge"].replace(stiffness=args.door_stiffness)
        env_cfg.scene.cabinet.actuators = acts
    if args.no_cabinet:  # fixed base: it stays 3 m to the robot's left; the door joint is still reset to 0
        x, y, z = env_cfg.scene.cabinet.init_state.pos
        env_cfg.scene.cabinet.init_state.pos = (x, y + 3.0, z)
    if args.obs == "fk":
        pol = env_cfg.observations.policy
        pol.door_angle.func, pol.door_angle.params = door_angle_fk, {}
        pol.book_pos.func = book_pos_fk
        pol.book_quat.func, pol.book_quat.params = book_quat_fk, {}
    if args.video:
        for name in VIEWS:
            setattr(env_cfg.scene, f"cam_{name}", CameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Cam_{name}", update_period=0.0, width=640, height=480, data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.01, 20.0)),
            ))
    env = gym.make(args.task, cfg=env_cfg)
    raw = env.unwrapped
    dev, n, dt = raw.device, raw.num_envs, raw.step_dt
    if args.video:
        goal = raw.cfg.rewards.insert_success.params["goal_pos_b"]
        shift = torch.tensor([g - g45 for g, g45 in zip(goal, C.BOOK_INSERT_POS_B)], device=dev)
        origin = raw.scene.env_origins[0]
        for name, (eye, lookat) in VIEWS.items():
            raw.scene[f"cam_{name}"].set_world_poses_from_view(torch.tensor([eye], device=dev) + origin + shift,
                                                               torch.tensor([lookat], device=dev) + origin + shift)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)  # resets the env
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=dev)

    robot = raw.scene["robot"]
    all_ids = torch.arange(n, device=dev)
    if args.no_book:  # on the floor 3 m to the robot's right
        book = raw.scene["book"]
        pose = torch.cat((T(book.data.root_pos_w), T(book.data.root_quat_w)), -1).clone()
        pose[:, :3] = raw.scene.env_origins + torch.tensor([0.5, -3.0, 0.2], device=dev)
        book.write_root_pose_to_sim_index(root_pose=pose, env_ids=all_ids)
        book.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=all_ids)
    for side, mode in (("left", args.left), ("right", args.right)):
        if mode == "friction":
            ids = robot.find_joints(f"{side}_fr3_finger_joint.*")[0]
            robot.set_joint_position_target_index(
                target=torch.full((n, len(ids)), args.finger_target, device=dev), joint_ids=ids, env_ids=all_ids)
    left_q_ids = [robot.joint_names.index(f"left_fr3_joint{j}") for j in range(1, 8)]
    right_q_ids = [robot.joint_names.index(f"right_fr3_joint{j}") for j in range(1, 8)]
    left_finger = robot.joint_names.index("left_fr3_finger_joint1")
    right_finger = robot.joint_names.index("right_fr3_finger_joint1")
    sp = dict(raw.cfg.rewards.insert_success.params)

    def frame():
        for _ in range(2):
            raw.sim.render()
        raw.scene.update(raw.physics_dt)
        return np.concatenate([T(raw.scene[f"cam_{nm}"].data.output["rgb"])[0].cpu().numpy()[..., :3].astype(np.uint8)
                               for nm in VIEWS], 1)

    frames = []
    obs = env.get_observations()
    zero = torch.zeros(n, env.num_actions, device=dev)
    with torch.inference_mode():
        for _ in range(int(round(args.settle_s / dt))):  # relative action 0 = hold the measured joints
            obs, *_ = env.step(zero)
            if args.video:
                frames.append(frame())
        fingers0 = (T(robot.data.joint_pos)[:, left_finger].clone(), T(robot.data.joint_pos)[:, right_finger].clone())
        steps = int(round(episode_s / dt)) - 1
        rec = {k: [] for k in ("door", "door_fk", "slip_l", "slip_r", "act_l", "qd_l", "act_r", "qd_r", "tcp_l", "tcp_r",
                               "book_fk", "handle", "succ")}
        hold = torch.zeros(n, dtype=torch.bool, device=dev)
        for _ in range(steps):
            act = policy(obs)
            if args.left_hold_deg is not None:
                hold |= door_angle_fk_rad(raw) >= math.radians(args.left_hold_deg)
                act[:, :7] = torch.where(hold[:, None], 0.0, act[:, :7])
            obs, *_ = env.step(act)
            rec["door"].append(torch.rad2deg(mdp.door_angle_rad(raw)))
            rec["door_fk"].append(torch.rad2deg(door_angle_fk_rad(raw)))
            rec["slip_l"].append(100 * grasp_slip(raw, "left"))
            rec["slip_r"].append(100 * grasp_slip(raw, "right"))
            rec["act_l"].append(act[:, :7].clamp(-1.0, 1.0))
            rec["qd_l"].append(T(robot.data.joint_vel)[:, left_q_ids])
            rec["act_r"].append(act[:, 7:].clamp(-1.0, 1.0))
            rec["qd_r"].append(T(robot.data.joint_vel)[:, right_q_ids])
            rec["tcp_l"].append(100 * tcp_b(raw, "left")[0])
            rec["tcp_r"].append(100 * tcp_b(raw, "right")[0])
            rec["book_fk"].append(100 * book_pose_fk(raw)[0])
            rec["handle"].append(mdp.handle_force_norm(raw))
            rec["succ"].append(mdp.insert_success(raw, **sp))
            if args.video:
                frames.append(frame())
    r = {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}  # (T, N, ...)
    w = slice(int(round(args.after_s / dt)), None)

    print(f"[grasp] {os.path.basename(args.checkpoint)} | left {args.left}, right {args.right}, obs {args.obs} | envs {n}"
          f"{'' if args.row is None else f' (row {args.row})'}"
          f"{'' if args.left_hold_deg is None else f' | left hold from FK door {args.left_hold_deg:.0f} deg'} | {steps} steps ({steps * dt:.1f} s) after {args.settle_s:.1f} s "
          f"settle")
    if "friction" in (args.left, args.right):
        print(f"[grasp] finger opening after settle (m, each finger): left {stat(fingers0[0].cpu().numpy(), '{:.4f}')}, "
              f"right {stat(fingers0[1].cpu().numpy(), '{:.4f}')}")
    succ = r["succ"] > 0.5
    print(f"[grasp] success ever {int(succ.any(0).sum())}, at end {int(succ[-1].sum())} / {n}")
    print(f"[grasp] door true (deg): max {stat(r['door'].max(0), '{:.0f}')} | final {stat(r['door'][-1], '{:.0f}')}")
    print(f"[grasp] door FK  (deg): max {stat(r['door_fk'].max(0), '{:.0f}')} | final {stat(r['door_fk'][-1], '{:.0f}')} | "
          f"|FK - true| max {stat(np.abs(r['door_fk'] - r['door']).max(0))}")
    if not (args.no_cabinet or args.no_book):
        print(f"[grasp] slip from the grasp frame (cm): left max {stat(r['slip_l'].max(0))}, final {stat(r['slip_l'][-1])} | "
              f"right max {stat(r['slip_r'].max(0))}, final {stat(r['slip_r'][-1])}")
    rad = np.linalg.norm(r["tcp_l"][..., :2] - 100 * np.asarray(args.hinge), axis=-1)  # left TCP distance from the hinge
    rad0 = 100 * math.dist(args.hinge, args.closed_tcp)
    print(f"[grasp] left TCP distance from the hinge axis (door radius {rad0:.1f} cm): min {stat(rad.min(0))}, max "
          f"{stat(rad.max(0))} | z range {stat(r['tcp_l'][..., 2].max(0) - r['tcp_l'][..., 2].min(0))} cm")
    depth = r["book_fk"][..., 0] - 100 * sp["opening_x_b"]  # FK book centre past the opening (cm)
    t_in = np.where((depth > 0).any(0), (depth > 0).argmax(0), -1)
    door_in = np.where(t_in >= 0, r["door_fk"][np.clip(t_in, 0, None), np.arange(n)], np.nan)
    print(f"[grasp] FK book centre past the opening: {int((t_in >= 0).sum())}/{n}, at {stat(np.where(t_in >= 0, t_in * dt, np.nan), '{:.2f}')} s, "
          f"FK door there {stat(door_in, '{:.0f}')} deg | final depth {stat(depth[-1])} cm")
    print(f"[grasp] handle force (N): whole run max {stat(r['handle'].max(0), '{:.0f}')} | from {args.after_s:.1f} s "
          f"mean {stat(r['handle'][w].mean(0), '{:.0f}')}")
    qd = r["qd_l"][w]
    da = np.abs(np.diff(r["act_l"][w], axis=0))
    ptp = (r["tcp_l"][w].max(0) - r["tcp_l"][w].min(0)).max(-1)
    print(f"[grasp] left arm from {args.after_s:.1f} s (door open): joint speed RMS {stat(np.sqrt((qd ** 2).mean((0, 2))), '{:.3f}')} "
          f"rad/s | |action change| per step {stat(da.mean((0, 2)), '{:.3f}')} | |action| > 0.95 "
          f"{stat(100 * (np.abs(r['act_l'][w]) > 0.95).mean((0, 2)), '{:.0f}')} % | TCP range (largest axis) "
          f"{stat(ptp)} cm")
    tlim = np.asarray(raw.action_manager.get_term("arm_action").cfg.target_limits[:7])  # left arm (policy targets)
    q_end = T(robot.data.joint_pos)[:, left_q_ids].cpu().numpy()
    margin = np.minimum(q_end - tlim[:, 0], tlim[:, 1] - q_end)
    print(f"[grasp] left q at the end (median) {np.round(np.median(q_end, 0), 3).tolist()} | margin to the policy target "
          f"limits (median per joint) {np.round(np.median(margin, 0), 3).tolist()} | envs with a joint within 0.01 rad: "
          f"{int((margin.min(-1) < 0.01).sum())}/{n}")
    for side in ("l", "r"):
        a = r[f"act_{side}"][w]
        print(f"[grasp] {'left' if side == 'l' else 'right'} arm from {args.after_s:.1f} s: joint speed RMS "
              f"{stat(np.sqrt((r[f'qd_{side}'][w] ** 2).mean((0, 2))), '{:.3f}')} rad/s | mean action per joint (median over "
              f"envs) {np.round(np.median(a.mean(0), 0), 3).tolist()} | sum |mean action| {stat(np.abs(a.mean(0)).sum(-1), '{:.2f}')}")
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        np.savez(args.save, dt=dt, hinge=np.asarray(args.hinge), opening_x=sp["opening_x_b"], **r)
        print("[grasp] wrote", args.save)
    if args.video:
        import imageio.v2 as imageio

        os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
        imageio.mimwrite(args.video, frames, fps=round(1.0 / dt))
        print("[grasp] wrote", args.video)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
