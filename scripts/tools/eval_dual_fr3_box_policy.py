# Copyright (c) 2026, DYROS.
# SPDX-License-Identifier: BSD-3-Clause
"""Box-lift RL policy eval with the same metrics and condition overrides as eval_dual_fr3_plate_rule_based.py.

Used for the rule-based vs RL comparison under stress conditions (2026-11-13): --corrupt (policy obs noise/delay/dropout,
off in Play cfgs), --box_mu / --plate_mu (friction DR ranges), --kp_scale, --mass, --disturb. Success = the env's own
success term (goal 10 cm, tilt 15 deg, both plates >= 1 N, held 0.5 s), counted once per env within --steps.

    python scripts/tools/eval_dual_fr3_box_policy.py --headless --num_envs 512 --seed 0 --steps 200 \
        --task Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-BoxDrop-FlatSqueeze-LowMuSoftLift-GoalXY-Play-v0 \
        --checkpoint logs/rsl_rl/<exp>/<run>/model_5499.pt [--corrupt] [--box_mu 0.1 0.3 --plate_mu 0.3 0.3] [--kp_scale 0.7]
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--max_ang_vel", type=float, default=None, help="box rigid_props.max_angular_velocity override (deg/s)")
parser.add_argument("--task", default="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--box_dx", type=float, nargs=2, default=None)
parser.add_argument("--mass", type=float, nargs=2, default=None)
parser.add_argument("--obs_delay", type=float, nargs=2, default=None)
parser.add_argument("--cmd_delay", type=int, nargs=2, default=None)
parser.add_argument("--size_y", type=float, nargs=2, default=None, help="box y scale range")
parser.add_argument("--disturb", type=float, default=1.0)
parser.add_argument("--plate_mu", type=float, nargs=2, default=None, help="plate (link7) static friction range, dynamic = 0.85x; needs randomize_plate_friction")
parser.add_argument("--steps", type=int, default=100, help="policy steps (20 Hz); training episode = 100")
parser.add_argument("--kp_scale", type=float, default=None, help="arm PD stiffness scale (weaker real squeeze proxy)")
parser.add_argument("--box_obs_offset", type=float, nargs=3, default=None, help="add to policy box position obs (base frame, m)")
parser.add_argument("--arm_friction", type=float, default=None, help="arm joint static/dynamic friction override (Nm)")
parser.add_argument("--box_mu", type=float, nargs=2, default=None, help="box static friction range (dynamic = 0.85x)")
parser.add_argument("--corrupt", action="store_true", help="enable the policy observation corruption (Play cfgs turn it off)")
parser.add_argument("--seed", type=int, default=None, help="env seed (DR samples)")
parser.add_argument("--push_t", type=float, default=None, help="[s] apply a box wrench from this time (use with --disturb 0)")
parser.add_argument("--push_dur", type=float, default=0.3, help="[s] push duration")
parser.add_argument("--push_force", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="world-frame force on the box [N]")
parser.add_argument("--push_torque", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="world-frame torque on the box [Nm]")
parser.add_argument("--log", default=None, help="npz with per-step box/goal/plate/action arrays (env, step, ...)")
parser.add_argument("--video", default=None, help="mp4 of every env from three cameras, one env after another (use few envs)")
parser.add_argument("--video_size", type=int, nargs=2, default=(640, 480), metavar=("W", "H"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    import os

    args.enable_cameras = True
    os.environ.setdefault("DR_VIS", "0")  # the DR range plates hide the lifted box
app = AppLauncher(args).app

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
import numpy as np  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import dual_fr3_lab.tasks  # noqa: E402,F401
from dual_fr3_lab.robots.dual_fr3_plate_ee import PLATE_TCP_OFFSET_POS  # noqa: E402
from dual_fr3_lab.tasks.manager_based.lift_plate.joint_pos_env_cfg import (  # noqa: E402
    FR3_JOINT_POS_LIMITS,
    _box_halves,
    nonplate_contact_forces,
)


def T(x):
    return x.torch if hasattr(x, "torch") else x


cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
agent = handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point"),
                                     metadata.version("rsl-rl-lib"))
cfg.scene.num_envs = args.num_envs
cfg.observations.policy.enable_corruption = bool(args.corrupt)
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
if args.plate_mu is not None:
    cfg.events.randomize_plate_friction.params.update(static_friction_range=tuple(args.plate_mu), dynamic_friction_range=tuple(0.85 * v for v in args.plate_mu))
    print(f"[cfg] plate friction {cfg.events.randomize_plate_friction.params['static_friction_range']}")
if args.box_mu is not None:
    cfg.events.randomize_box_friction.params.update(static_friction_range=tuple(args.box_mu), dynamic_friction_range=tuple(0.85 * v for v in args.box_mu))
    print(f"[cfg] box friction {cfg.events.randomize_box_friction.params['static_friction_range']}")
if args.mass is not None:
    cfg.events.randomize_box_density.params["mass_distribution_params"] = tuple(args.mass)
print(f"[cfg] obs corruption {cfg.observations.policy.enable_corruption}, seed {cfg.seed}, mass {cfg.events.randomize_box_density.params['mass_distribution_params']}")

if args.max_ang_vel is not None:
    cfg.scene.object.spawn.rigid_props.max_angular_velocity = args.max_ang_vel
print(f"[cfg] box max_angular_velocity {cfg.scene.object.spawn.rigid_props.max_angular_velocity} deg/s")
if args.obs_delay is not None:
    cfg.actions.arm_action.obs_delay_range_s = tuple(args.obs_delay)
if args.cmd_delay is not None:
    cfg.actions.arm_action.cmd_delay_steps_range = tuple(args.cmd_delay)
if args.size_y is not None:
    cfg.events.randomize_box_size.params["scale_range"]["y"] = tuple(args.size_y)
if args.box_obs_offset is not None:
    _f = cfg.observations.policy.object_position.func
    _off = list(args.box_obs_offset)

    import functools

    @functools.wraps(_f)  # the obs manager reads the wrapped signature
    def _offset_obs(env, *a, **kw):
        return _f(env, *a, **kw) + torch.tensor(_off, device=env.device)

    cfg.observations.policy.object_position.func = _offset_obs
    print(f"[cfg] policy box position obs offset {_off} m")
for name in ("arms_shoulder", "arms_forearm"):
    act = cfg.scene.robot.actuators[name]
    if args.kp_scale is not None:
        act.stiffness = act.stiffness * args.kp_scale
    if args.arm_friction is not None:
        act.friction = act.dynamic_friction = args.arm_friction
    print(f"[cfg] {name} Kp {act.stiffness} Kd {act.damping} friction {act.friction}/{act.dynamic_friction} Nm")
print(f"[rl] obs delay {getattr(cfg.actions.arm_action, 'obs_delay_range_s', None)} s, cmd delay {getattr(cfg.actions.arm_action, 'cmd_delay_steps_range', None)} steps, size scale {cfg.events.randomize_box_size.params['scale_range']}")
# --video cameras (eye, lookat) relative to the env origin, same as eval_dual_fr3_plate_rule_based.py
VIEWS = {
    "overview": ((2.3, -1.9, 1.9), (0.40, 0.0, 0.45)),
    "front (robot L = image R)": ((2.7, 0.0, 1.0), (0.45, 0.0, 0.50)),
    "side (robot right)": ((0.45, -2.8, 0.75), (0.45, 0.0, 0.50)),
}
if args.video:
    import isaaclab.sim as sim_utils
    from isaaclab.sensors.camera import CameraCfg

    cfg.scene.env_spacing = 8.0  # keep the neighbours out of the frame
    for _k in range(len(VIEWS)):
        setattr(cfg.scene, f"rec_{_k}", CameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/RecCam_{_k}", update_period=0.0,
            width=args.video_size[0], height=args.video_size[1], data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 20.0)),
        ))


def video_frame(panels, labels, text):
    """Panels side by side, each labelled at the bottom, with a text strip on top (as in the rule-based script)."""
    from PIL import Image, ImageDraw, ImageFont

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


env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=agent.clip_actions)
raw = env.unwrapped
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=raw.device)
env.reset()
raw._disturb_scale = args.disturb
obs = env.get_observations()
FRAMES = []
if args.video:
    CAMS = [raw.scene[f"rec_{_k}"] for _k in range(len(VIEWS))]
    for _cam, (_eye, _look) in zip(CAMS, VIEWS.values()):
        _cam.set_world_poses_from_view(torch.tensor([_eye], device=raw.device) + raw.scene.env_origins,
                                       torch.tensor([_look], device=raw.device) + raw.scene.env_origins)
    for _ in range(5):  # renderer warm-up, no physics step
        raw.sim.render()

dev, n = raw.device, raw.num_envs
dt = cfg.sim.dt * cfg.decimation
robot, box = raw.scene["robot"], raw.scene["object"]
origin = raw.scene.env_origins
succ_term = raw.reward_manager.get_term_cfg("success").func
hold_steps = raw.reward_manager.get_term_cfg("success").params["hold_steps"]
term = raw.action_manager.get_term("arm_action")
names = list(term._joint_names)
jids = term._joint_ids if not isinstance(term._joint_ids, slice) else list(range(len(names)))
lim = torch.tensor(FR3_JOINT_POS_LIMITS, device=dev).repeat(2, 1)
offset = torch.tensor(PLATE_TCP_OFFSET_POS, device=dev).repeat(n, 1)
up = torch.tensor([[0.0, 0.0, 1.0]], device=dev).repeat(n, 1)
bodies = {s: robot.find_bodies(f"{s}_fr3_link7")[0][0] for s in ("left", "right")}
hy, hz = _box_halves(raw)


def tcp(side):
    b = bodies[side]
    return T(robot.data.body_pos_w)[:, b] + quat_apply(T(robot.data.body_quat_w)[:, b], offset) - origin


def plate_forces():
    return torch.stack([torch.norm(T(raw.scene[s].data.force_matrix_w), dim=-1).amax(dim=(-1, -2))
                        for s in ("left_plate_object_contact", "right_plate_object_contact")], -1)


def box_state():
    c = T(box.data.root_pos_w) - origin
    tilt = torch.rad2deg(torch.acos(quat_apply(T(box.data.root_quat_w), up)[:, 2].clamp(-1, 1)))
    return c, tilt


def goal_env():
    cmd = raw.command_manager.get_command("object_pose")[:, :3]
    return T(robot.data.root_pos_w) + quat_apply(T(robot.data.root_quat_w), cmd) - origin


mass = T(box.data.body_mass)[:, 0].clone()
mu = wp.to_torch(box.root_view.get_material_properties())[:, 0, 0].clone().to(dev)
c0, _ = box_state()
_xb0 = quat_apply(T(box.data.root_quat_w), torch.tensor([[1.0, 0.0, 0.0]], device=dev).repeat(n, 1))
yaw0 = torch.rad2deg(torch.atan2(_xb0[:, 1], _xb0[:, 0]))
box_x0 = (c0[:, 0] - T(robot.data.root_pos_w)[:, 0] + origin[:, 0]).clone()
t_succ = torch.full((n,), -1, dtype=torch.long, device=dev)
t_lift = torch.full((n,), -1, dtype=torch.long, device=dev)
tilt_max_lift = torch.zeros(n, device=dev)
margin_min = torch.full((n,), 9.0, device=dev)
nonplate_max = torch.zeros(n, device=dev)
f_max = torch.zeros(n, 2, device=dev)
grip_dz = torch.full((n,), float("nan"), device=dev)
grip_dzlr = torch.full((n,), float("nan"), device=dev)
amax = torch.zeros(n, device=dev)
clip_steps = torch.zeros(n, device=dev)
at_lim_steps = torch.zeros(n, len(names), device=dev)  # steps within 0.02 rad of a FR3 limit, per joint
clip = float(list(term.cfg.clip.values())[0][1])
scale = float(term.cfg.scale)

LOG = {k: [] for k in ("box", "tilt", "plates", "force", "dq")}
with torch.inference_mode():
    for step in range(args.steps):
        a = policy(obs)
        if args.push_t is not None:
            k0, k1 = round(args.push_t / dt), round((args.push_t + args.push_dur) / dt)
            if k0 <= step < k1 or step == k1:
                on = float(k0 <= step < k1)
                fz = torch.tensor(args.push_force, device=dev).repeat(n, 1, 1) * on
                tz = torch.tensor(args.push_torque, device=dev).repeat(n, 1, 1) * on
                box.set_external_force_and_torque(fz, tz, is_global=True)
        if args.log:
            c_, tilt_ = box_state()
            LOG["box"].append(c_.clone()); LOG["tilt"].append(tilt_.clone())
            LOG["plates"].append(torch.stack([tcp("left"), tcp("right")], 1))
            LOG["force"].append(plate_forces())
            LOG["dq"].append((a[:, : len(names)] * scale).clone())
        da = (a[:, : len(names)] * scale).abs()
        amax = torch.maximum(amax, da.amax(-1))
        clip_steps += (da >= clip).any(-1).float()
        obs, _, _, _ = env.step(a)
        c, tilt = box_state()
        f = plate_forces()
        f_max = torch.maximum(f_max, f)
        lifted = (c[:, 2] - hz) > 0.02
        new_lift = lifted & (t_lift < 0) & (f > 1.0).all(-1)
        if new_lift.any():
            pl, pr = tcp("left"), tcp("right")
            grip_dz = torch.where(new_lift, 0.5 * (pl[:, 2] + pr[:, 2]) - c[:, 2], grip_dz)
            grip_dzlr = torch.where(new_lift, pl[:, 2] - pr[:, 2], grip_dzlr)
            t_lift = torch.where(new_lift, torch.full_like(t_lift, step + 1), t_lift)
        tilt_max_lift = torch.where(lifted, torch.maximum(tilt_max_lift, tilt), tilt_max_lift)
        qj = T(robot.data.joint_pos)[:, jids]
        mj = torch.minimum(qj - lim[:, 0], lim[:, 1] - qj)
        margin_min = torch.minimum(margin_min, mj.amin(-1))
        at_lim_steps += (mj < 0.02).float()
        nonplate_max = torch.maximum(nonplate_max, nonplate_contact_forces(raw).sum(-1))
        newly = (succ_term._count >= hold_steps) & (t_succ < 0)
        t_succ = torch.where(newly, torch.full_like(t_succ, step + 1), t_succ)
        if args.video:
            rgb = [T(cam.data.output["rgb"])[..., :3].cpu().numpy().astype(np.uint8) for cam in CAMS]
            pushing = args.push_t is not None and args.push_t <= step * dt < args.push_t + args.push_dur
            dg = torch.norm(goal_env() - c, dim=-1)
            FRAMES.append([video_frame(
                [r[i] for r in rgb], list(VIEWS),
                f"RL  env {i} ({float(mass[i]):.2f} kg, mu {float(mu[i]):.2f})  t {(step + 1) * dt:4.1f}s"
                f"  to goal {100 * float(dg[i]):5.1f} cm  tilt {float(tilt[i]):4.1f} deg  F L/R {float(f[i, 0]):4.1f}/{float(f[i, 1]):4.1f} N"
                f"  |dq| {float(da[i].mean()):.3f}  {'PUSH' if pushing else ''}") for i in range(n)])

c, tilt = box_state()
xb = quat_apply(T(box.data.root_quat_w), torch.tensor([[1.0, 0.0, 0.0]], device=dev).repeat(n, 1))
yaw_end = torch.rad2deg(torch.atan2(xb[:, 1], xb[:, 0]))
print(f"[rl] box yaw at end (deg, world): |yaw| median {yaw_end.abs().median():.1f} p90 {yaw_end.abs().quantile(0.9):.1f}"
      f" | yaw0 |median| {yaw0.abs().median():.1f}")
f_end = plate_forces()
print(f"[rl] plate force at end L/R median {f_end[:, 0].median():.1f}/{f_end[:, 1].median():.1f} N")
g = goal_env()
dist = torch.norm(g - c, dim=-1)
fin_ok = (dist < 0.10) & (tilt < 15.0) & (plate_forces() > 1.0).all(-1)
succ = t_succ > 0
print(f"\n[rl] {args.checkpoint}")
print(f"[rl] {n} envs, mass {mass.min():.2f}~{mass.max():.2f} kg, box mu {mu.min():.2f}~{mu.max():.2f}, "
      f"disturb {args.disturb}, steps {args.steps} ({args.steps * dt:.1f} s), box x0 {box_x0.min():.2f}~{box_x0.max():.2f}")
print(f"[rl] success (env term, held 0.5 s): {succ.float().mean():.3f} | in success zone at end: {fin_ok.float().mean():.3f}"
      f" | lifted with both plates: {(t_lift > 0).float().mean():.3f}")
ts = t_succ[succ].float() * dt
if len(ts):
    print(f"[rl] time to success [s]: median {ts.median():.2f}, p90 {ts.quantile(0.9):.2f}")
print(f"[rl] end: goal dist median {100 * dist.median():.1f} cm, tilt median {tilt.median():.1f} deg | max tilt while lifted "
      f"median {tilt_max_lift[t_lift > 0].median():.1f} p90 {tilt_max_lift[t_lift > 0].quantile(0.9):.1f} deg")
print(f"[rl] max plate force L/R median {f_max[:, 0].median():.1f}/{f_max[:, 1].median():.1f} N"
      f" | grip at lift: plate mid - box centre dz median {100 * grip_dz.nanmedian():.1f} cm, |zL-zR| median"
      f" {100 * grip_dzlr.abs().nanmedian():.1f} p90 {100 * grip_dzlr.abs().nanquantile(0.9):.1f} cm")
print(f"[rl] min joint margin to FR3 limits median {margin_min.median():.3f}, min {margin_min.min():.3f} rad,"
      f" share < 0.02 {(margin_min < 0.02).float().mean():.2f} | |0.1a| max median {amax.median():.2f} rad,"
      f" steps at ±{clip} clip median {clip_steps.median():.0f}/{args.steps}")
print(f"[rl] non-plate contact max median {nonplate_max.median():.1f} N, share > 1 N {(nonplate_max > 1).float().mean():.2f},"
      f" > 10 N {(nonplate_max > 10).float().mean():.2f}")
share = at_lim_steps / args.steps
print("[rl] time share within 0.02 rad of a FR3 limit (mean over envs) / share of envs ever there:")
for k, nm in enumerate(names):
    if share[:, k].mean() > 0.005:
        side = "hi" if (T(robot.data.joint_pos)[:, jids[k]] - lim[k, 0]).mean() > (lim[k, 1] - T(robot.data.joint_pos)[:, jids[k]]).mean() else "lo"
        print(f"[rl]   {nm:22s} {share[:, k].mean():.2f} / {(share[:, k] > 0).float().mean():.2f} (end side {side})")
for lo, hi in ((0.40, 0.50), (0.50, 0.55), (0.55, 0.65)):
    m = (box_x0 >= lo) & (box_x0 < hi)
    print(f"[rl]   box x0 [{lo:.2f},{hi:.2f}): any-joint-at-limit time share {share[m].amax(-1).mean():.2f}")
for lo, hi in ((0.40, 0.50), (0.50, 0.55), (0.55, 0.65)):
    m = (box_x0 >= lo) & (box_x0 < hi)
    print(f"[rl]   box x0 [{lo:.2f},{hi:.2f}): success {succ[m].float().mean() if m.any() else float('nan'):.2f} ({int(m.sum())})")
print("[rl] by mass / mu (success rate, n):")
for lo, hi in ((0.0, 0.625), (0.625, 0.75), (0.75, 0.875), (0.875, 9.0)):
    m = (mass >= lo) & (mass < hi)
    row = []
    for mlo, mhi in ((0.0, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 9.0)):
        k = m & (mu >= mlo) & (mu < mhi)
        row.append(f"mu[{mlo:.1f},{mhi:.1f}) {succ[k].float().mean() if k.any() else float('nan'):.2f} ({int(k.sum())})")
    print(f"   mass [{lo:.3f},{hi:.3f}): " + " | ".join(row))
if args.video:
    import imageio.v2 as imageio

    os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
    imageio.mimwrite(args.video, [fr[i] for i in range(n) for fr in FRAMES], fps=round(1.0 / dt),
                     codec="libx264", quality=8, macro_block_size=1)
    print(f"[rl] video: {n} envs x {len(FRAMES)} frames -> {args.video}")
if args.log:
    np.savez(args.log, dt=dt, goal=goal_env().cpu().numpy(), mass=mass.cpu().numpy(), mu=mu.cpu().numpy(),
             push=np.array([args.push_t if args.push_t is not None else -1.0, args.push_dur, *args.push_force, *args.push_torque]),
             **{k: torch.stack(v, 1).cpu().numpy() for k, v in LOG.items()})
    print(f"[rl] log -> {args.log}")
env.close()
app.close()
