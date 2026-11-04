# Copyright (c) 2026, DYROS.
# SPDX-License-Identifier: BSD-3-Clause
"""Make the box slip out of the plates once and measure the drop, the landing and whether the policy re-grasps.

At ``--slip_t`` every env that holds the box with both plates gets box and plate (``fr3_link7`` shapes) friction set to
``--mu`` for ``--slip_dur``, then restored (PhysX tensor API). The episode runs ``--steps`` policy steps without
terminations or curricula. Reported: drop rate, time to floor, landing tilt / shift / yaw, re-lift rate and time,
success zone at the end. ``--no_slip`` runs the same long episode without slipping.

``--save_states`` writes the settled post-drop states (``--settle_t`` after slip start, box on the floor and tilt < 30 deg)
in the pre-grasp table format (``joint_pos``, ``joint_names``, ``box_pose`` = env-frame position + quaternion xyzw), used
by ``-Regrasp`` tasks to start episodes right after a drop.

    python scripts/tools/box_slip_probe.py --checkpoint logs/rsl_rl/<exp>/<run>/model_XXXX.pt --viz none
    python scripts/tools/box_slip_probe.py --checkpoint ... --seed 1 --save_states /tmp/drop_states_a.npz --viz none
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=160, help="policy steps (20 Hz)")
parser.add_argument("--slip_t", type=float, default=1.5)
parser.add_argument("--slip_dur", type=float, default=0.3)
parser.add_argument("--mu", type=float, default=0.0)
parser.add_argument("--no_slip", action="store_true")
parser.add_argument("--box_only", action="store_true", help="lower only the box friction, not the plates")
parser.add_argument("--save_states", default=None, help="npz path for settled post-drop states")
parser.add_argument("--settle_t", type=float, default=1.0, help="s after slip start at which states are saved")
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import dual_fr3_lab.tasks  # noqa: E402,F401
from dual_fr3_lab.tasks.manager_based.lift_plate.joint_pos_env_cfg import _box_halves  # noqa: E402


def T(x):
    return x.torch if hasattr(x, "torch") else x


cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
agent = handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point"), metadata.version("rsl-rl-lib"))
cfg.scene.num_envs = args.num_envs
if args.seed is not None:
    cfg.seed = args.seed
cfg.observations.policy.enable_corruption = False
cfg.episode_length_s = (args.steps + 20) * cfg.sim.dt * cfg.decimation
cfg.commands.object_pose.resampling_time_range = (1e3, 1e3)
for k in list(cfg.curriculum.__dict__):
    setattr(cfg.curriculum, k, None)
for k in list(cfg.terminations.__dict__):
    if k != "time_out":
        setattr(cfg.terminations, k, None)
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=agent.clip_actions)
raw = env.unwrapped
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=raw.device)
env.reset()
raw._disturb_scale = 1.0
obs = env.get_observations()

dev, n = raw.device, raw.num_envs
dt = cfg.sim.dt * cfg.decimation
robot, box = raw.scene["robot"], raw.scene["object"]
origin = raw.scene.env_origins
up = torch.tensor([[0.0, 0.0, 1.0]], device=dev).repeat(n, 1)
hy, hz = _box_halves(raw)

# --- materials: box shapes, robot shapes per link
box_mat0 = wp.to_torch(box.root_view.get_material_properties()).clone()  # (n, shapes, 3) cpu
links = [p.split("/")[-1] for p in robot.root_view.link_paths[0]]
nshape = [robot._physics_sim_view.create_rigid_body_view(p).max_shapes for p in robot.root_view.link_paths[0]]
starts = np.cumsum([0] + nshape)
rob_mat0 = wp.to_torch(robot.root_view.get_material_properties()).clone()
plate_links = [i for i, nm in enumerate(links) if "plate" in nm.lower()]
if not plate_links:
    plate_links = [i for i, nm in enumerate(links) if nm.endswith("link7") or nm.endswith("hand") or "ee" in nm.lower()]
plate_cols = [c for i in plate_links for c in range(starts[i], starts[i + 1])]
print(f"[probe] box shapes {box_mat0.shape[1]}, env0 box mat {box_mat0[0, 0].tolist()}")
print(f"[probe] robot links {len(links)}, shapes {rob_mat0.shape[1]}; plate links {[links[i] for i in plate_links]} -> shape cols {plate_cols}")
print(f"[probe] env0 plate mats {rob_mat0[0, plate_cols].tolist()}")
print(f"[probe] fix_root_link {getattr(cfg.scene.robot.spawn.articulation_props, 'fix_root_link', None)}")


def plate_forces():
    return torch.stack([torch.norm(T(raw.scene[s].data.force_matrix_w), dim=-1).amax(dim=(-1, -2))
                        for s in ("left_plate_object_contact", "right_plate_object_contact")], -1)


def box_state():
    c = T(box.data.root_pos_w) - origin
    q = T(box.data.root_quat_w)
    tilt = torch.rad2deg(torch.acos(quat_apply(q, up)[:, 2].clamp(-1, 1)))
    xb = quat_apply(q, torch.tensor([[1.0, 0.0, 0.0]], device=dev).repeat(n, 1))
    yaw = torch.rad2deg(torch.atan2(xb[:, 1], xb[:, 0]))
    return c, tilt, yaw


def goal_env():
    cmd = raw.command_manager.get_command("object_pose")[:, :3]
    return T(robot.data.root_pos_w) + quat_apply(T(robot.data.root_quat_w), cmd) - origin


def set_mu(ids, mu):
    ids_cpu = ids.cpu().to(torch.int32)
    bm = box_mat0.clone()
    if mu is not None:
        bm[ids_cpu.long(), :, 0:2] = mu
    box.root_view.set_material_properties(wp.from_torch(bm.contiguous(), dtype=wp.float32), wp.from_torch(ids_cpu, dtype=wp.int32))
    if not args.box_only:
        rm = rob_mat0.clone()
        if mu is not None:
            for c in plate_cols:
                rm[ids_cpu.long(), c, 0:2] = mu
        robot.root_view.set_material_properties(wp.from_torch(rm.contiguous(), dtype=wp.float32), wp.from_torch(ids_cpu, dtype=wp.int32))


k_slip = round(args.slip_t / dt)
k_end = k_slip + max(1, round(args.slip_dur / dt))
k_settle = k_slip + round(args.settle_t / dt)
slipped = torch.zeros(n, dtype=torch.bool, device=dev)
c_pre = torch.zeros(n, 3, device=dev)
yaw_pre = torch.zeros(n, device=dev)
landed_t = torch.full((n,), -1, dtype=torch.long, device=dev)
relift_t = torch.full((n,), -1, dtype=torch.long, device=dev)
tilt_settle = torch.zeros(n, device=dev)
dxy_settle = torch.zeros(n, device=dev)
dyaw_settle = torch.zeros(n, device=dev)
tilt_max_after = torch.zeros(n, device=dev)
lifted_pre = torch.zeros(n, dtype=torch.bool, device=dev)
saved = None
z_track = []
with torch.inference_mode():
    for step in range(args.steps):
        a = policy(obs)
        obs, _, _, _ = env.step(a)
        c, tilt, yaw = box_state()
        f = plate_forces()
        zrel = c[:, 2] - hz
        z_track.append(zrel.median().item())
        if step == k_slip - 1:
            lifted_pre = (zrel > 0.05) & (f > 1.0).all(-1)
            c_pre, yaw_pre = c.clone(), yaw.clone()
            if not args.no_slip:
                slipped = lifted_pre.clone()
                ids = slipped.nonzero().squeeze(-1)
                if len(ids):
                    set_mu(ids, args.mu)
        if step == k_end - 1 and not args.no_slip:
            ids = slipped.nonzero().squeeze(-1)
            if len(ids):
                set_mu(ids, None)
        if step >= k_slip:
            on_floor = zrel < 0.02
            landed_t = torch.where(slipped & on_floor & (landed_t < 0), step, landed_t)
            tilt_max_after = torch.maximum(tilt_max_after, tilt)
            relift = (landed_t >= 0) & (zrel > 0.10) & (f > 1.0).all(-1) & (relift_t < 0)
            relift_t = torch.where(relift, step, relift_t)
        if step == k_settle:
            tilt_settle = tilt.clone()
            dxy_settle = torch.norm(c[:, :2] - c_pre[:, :2], dim=-1)
            dyaw_settle = ((yaw - yaw_pre + 180) % 360 - 180).abs()
            if args.save_states:
                keep = slipped & (landed_t >= 0) & (tilt < 30.0)
                ids = keep.nonzero().squeeze(-1)
                saved = {
                    "joint_pos": T(robot.data.joint_pos)[ids].cpu().numpy(),
                    "joint_vel": T(robot.data.joint_vel)[ids].cpu().numpy(),
                    "joint_names": np.array(robot.joint_names),
                    "box_pose": torch.cat([c[ids], T(box.data.root_quat_w)[ids]], -1).cpu().numpy(),
                    "box_lin_vel": T(box.data.root_lin_vel_w)[ids].cpu().numpy(),
                    "robot_root_pos_env": (T(robot.data.root_pos_w) - origin)[ids].cpu().numpy(),
                    "slip_t": np.array(args.slip_t),
                    "slip_dur": np.array(args.slip_dur),
                    "settle_t": np.array(args.settle_t),
                    "checkpoint": np.array(str(args.checkpoint)),
                }
    c, tilt, yaw = box_state()
    g = torch.norm(goal_env() - c, dim=-1)
    zone = (g < 0.10) & (tilt < 15.0) & (plate_forces() > 1.0).all(-1)


def q(x):
    if x.numel() == 0:
        return "n/a"
    x = x.float()
    return f"median {x.median():.2f} p10 {x.quantile(0.1):.2f} p90 {x.quantile(0.9):.2f}"


print(f"[probe] {args.checkpoint}")
print(f"[probe] {n} envs, {args.steps} steps ({args.steps*dt:.1f} s), slip {'off' if args.no_slip else f'at {args.slip_t}s for {args.slip_dur}s mu {args.mu} ({'box only' if args.box_only else 'box+plates'})'}")
print(f"[probe] lifted (both plates) just before slip time: {lifted_pre.float().mean():.3f}")
print(f"[probe] box height median over time (m above rest): " + " ".join(f"{k*dt:.1f}s {z_track[k]:+.3f}" for k in range(0, args.steps, 10)))
if not args.no_slip:
    m = slipped
    dropped = m & (landed_t >= 0)
    print(f"[probe] slipped {int(m.sum())}, dropped to floor {dropped.float().sum()/max(int(m.sum()),1):.3f}, time to floor [s] {q((landed_t[dropped]-k_slip+1).float()*dt)}")
    print(f"[probe] at slip+{args.settle_t}s: tilt {q(tilt_settle[dropped])} deg, toppled(>45) {(tilt_settle[dropped]>45).float().mean():.3f}, xy shift {q(dxy_settle[dropped]*100)} cm, |dyaw| {q(dyaw_settle[dropped])} deg")
    print(f"[probe] max tilt after slip {q(tilt_max_after[dropped])} deg")
    rel = dropped & (relift_t >= 0)
    print(f"[probe] re-lifted (>10 cm, both plates) after drop: {rel.float().sum()/max(int(dropped.sum()),1):.3f}, time from floor [s] {q((relift_t[rel]-landed_t[rel]).float()*dt)}")
    print(f"[probe] in success zone at end: dropped {zone[dropped].float().mean():.3f}, not slipped {zone[~m].float().mean() if (~m).any() else float('nan'):.3f}")
else:
    print(f"[probe] in success zone at end (no slip): {zone.float().mean():.3f}")
if saved is not None:
    np.savez(args.save_states, **saved)
    print(f"[probe] saved {len(saved['joint_pos'])} post-drop states -> {args.save_states}")
env.close()
app.close()
