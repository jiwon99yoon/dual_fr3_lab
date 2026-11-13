# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rule-based check of the cube pick-and-place task: is every sampled pick/place pair doable by the right arm?

A scripted DLS-IK controller drives the RL env through its own action interface (arm Δq = 0.1·a clipped to ±0.1 rad,
binary gripper), using the true cube pose (or, with --obs_pose, the pose the policy sees):

    PRE    TCP above the grasp point, top-down, closing direction along a cube face (yaw nearest the ready TCP)
    DESC   down to the grasp point (cube centre + 2 cm along the cube z axis)
    CLOSE  close the gripper, wait (a delayed gripper: also until its command is done, as for the real action result)
    LIFT   straight up
    MOVE   above the goal (keeping the grasp offset)
    LOWER  down until the cube centre is just above its resting height at the goal
    OPEN   open the gripper, wait (likewise)
    UP     straight up (hand clear)

At the end it prints the success count by the task's own stage (mdp.pick_place_state), the failures with their pick and
place positions, the reward sums and the joint margins.

    python scripts/tools/eval_dual_fr3_pick_place_rule_based.py --num_envs 256 --headless
    # grasp height with the policy's cube observation (fingertip 5 cm above the table = grasp point at the cube centre)
    python scripts/tools/eval_dual_fr3_pick_place_rule_based.py --num_envs 256 --headless --obs_pose --grasp_dz 0.0
    python scripts/tools/eval_dual_fr3_pick_place_rule_based.py --num_envs 4 --video logs/rule_based/pick_place/rule.mp4 \
        --headless --enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", default="Isaac-PickPlace-Cube-DualFR3-UMI-Play-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--duration", type=float, default=14.0, help="seconds (the task episode is 14 s)")
parser.add_argument("--finger_stiffness", type=float, default=None,
                    help="finger PD stiffness, N/m (default: the task's; 2000 = the robot cfg)")
parser.add_argument("--v_move", type=float, default=0.25, help="TCP reference speed, m/s")
parser.add_argument("--pre_dz", type=float, default=0.10, help="PRE height above the grasp point")
parser.add_argument("--lift_dz", type=float, default=0.12, help="LIFT/MOVE height of the grasp point above the table grasp height")
parser.add_argument("--place_dz", type=float, default=0.0,
                    help="cube centre above its resting height when opening (256 envs: 0.005 -> 247, 0 -> 249, -0.003 -> 243)")
parser.add_argument("--grasp_dz", type=float, default=None,
                    help="grasp point above the cube centre (default: task GRASP_OFFSET); fingertip above the table = 0.05 + this")
parser.add_argument("--obs_pose", action="store_true",
                    help="pick with the policy's cube observation (episode offset +-5 mm / +-3 deg, noise, delay, dropout), "
                         "averaged while reaching (PRE, DESC); placing still uses the true cube xy")
parser.add_argument("--wait_s", type=float, default=0.6,
                    help="CLOSE/OPEN wait; with the delayed gripper also until the gripper settles (command done)")
parser.add_argument("--place_tol", type=float, default=0.004, help="LOWER ends only with the cube xy this close to the goal")
parser.add_argument("--gain", type=float, default=1.0)
parser.add_argument("--null", type=float, default=0.0,
                    help="null-space pull toward the ready joints, per step (0.2 leaks ~1 cm into the TCP through the damped inverse)")
parser.add_argument("--lim_k", type=float, default=0.5, help="null-space push away from limits (within 0.3 rad)")
parser.add_argument("--seed", type=int, default=0)
# stress conditions shared with eval_dual_fr3_pick_place_grasp.py (rule-based vs RL, 2026-11-13)
parser.add_argument("--settle", type=float, nargs=2, default=None, help="gripper settle (command -> result) range [s]")
parser.add_argument("--cube_mu", type=float, nargs=2, default=None, help="cube static friction range (dynamic = 0.85x)")
parser.add_argument("--mass", type=float, nargs=2, default=None, help="cube mass range [kg]")
parser.add_argument("--finger_scale", type=float, default=None, help="scale the finger PD stiffness DR range (squeeze force)")
parser.add_argument("--kp_scale", type=float, default=None, help="arm PD stiffness scale")
parser.add_argument("--pick", type=float, nargs=2, default=None, help="every env: cube centre x y (base) instead of the random start")
parser.add_argument("--place", type=float, nargs=2, default=None, help="every env: goal x y (base) instead of the random goal")
parser.add_argument("--video", default="", help="mp4 path (env 0, needs --enable_cameras)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors.camera import CameraCfg  # noqa: E402
from isaaclab.utils.math import compute_pose_error, quat_apply, quat_from_angle_axis, quat_mul  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import dual_fr3_lab.tasks  # noqa: E402, F401
from dual_fr3_lab.robots import UMI_TCP_OFFSET_POS, UMI_TCP_OFFSET_ROT  # noqa: E402
from dual_fr3_lab.tasks.manager_based.pick_place_cube import mdp  # noqa: E402

PHASES = ("PRE", "DESC", "CLOSE", "LIFT", "MOVE", "LOWER", "OPEN", "UP", "DONE")
PRE, DESC, CLOSE, LIFT, MOVE, LOWER, OPEN, UP, DONE = range(len(PHASES))
VIEWS = {  # eye, target in base frame
    "front": ((1.45, -0.20, 0.75), (0.55, -0.20, 0.40)),
    "side": ((0.55, -1.25, 0.80), (0.55, -0.15, 0.40)),
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
    axis = torch.zeros(angle.shape[0], 3, device=angle.device)
    axis[:, 2] = 1.0
    return quat_from_angle_axis(angle, axis)


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    episode_s = cfg.episode_length_s  # the task's, for the summary
    cfg.episode_length_s = max(cfg.episode_length_s, args.duration + 1.0)  # no time-out reset mid-run
    # a scripted run keeps going after a drop: judge it at the end instead of resetting the env
    for name in ("cube_dropped", "cube_toppled", "cube_off_table"):
        setattr(cfg.terminations, name, None)
    cfg.rewards.drop = None
    if args.obs_pose:
        cfg.observations.policy.enable_corruption = True  # the Play cfg turns the noise off
    if args.settle is not None:
        cfg.actions.gripper_action.settle_range_s = tuple(args.settle)
    if args.cube_mu is not None:
        cfg.events.cube_friction.params.update(static_friction_range=tuple(args.cube_mu),
                                               dynamic_friction_range=tuple(0.85 * v for v in args.cube_mu))
    if args.mass is not None:
        cfg.events.cube_mass.params["mass_distribution_params"] = tuple(args.mass)
    if args.finger_scale is not None:
        lo, hi = cfg.events.finger_gains.params["stiffness_distribution_params"]
        cfg.events.finger_gains.params["stiffness_distribution_params"] = (lo * args.finger_scale, hi * args.finger_scale)
    if args.kp_scale is not None:
        acts = dict(cfg.scene.robot.actuators)
        for k, a in acts.items():
            if k != "hands":
                st = a.stiffness
                acts[k] = a.replace(stiffness={j: v * args.kp_scale for j, v in st.items()} if isinstance(st, dict) else st * args.kp_scale)
        cfg.scene.robot = cfg.scene.robot.replace(actuators=acts)
    print(f"[cfg] corruption {cfg.observations.policy.enable_corruption}, settle {getattr(cfg.actions.gripper_action, 'settle_range_s', None)},"
          f" cube mu {cfg.events.cube_friction.params['static_friction_range']}, mass {cfg.events.cube_mass.params['mass_distribution_params']},"
          f" finger stiffness scale {cfg.events.finger_gains.params['stiffness_distribution_params']}, arm kp scale {args.kp_scale}")
    cfg.commands.goal_pose.resampling_time_range = (args.duration + 1.0, args.duration + 1.0)
    if args.finger_stiffness is not None:
        actuators = dict(cfg.scene.robot.actuators)
        actuators["hands"] = actuators["hands"].replace(stiffness=args.finger_stiffness)
        cfg.scene.robot = cfg.scene.robot.replace(actuators=actuators)
    if args.pick is not None:
        cfg.events.reset_cube.params["x_range"] = (args.pick[0], args.pick[0])
        cfg.events.reset_cube.params["y_range"] = (args.pick[1], args.pick[1])
    if args.place is not None:
        cfg.commands.goal_pose.ranges.pos_x = (args.place[0], args.place[0])
        cfg.commands.goal_pose.ranges.pos_y = (args.place[1], args.place[1])
        cfg.commands.goal_pose.min_distance = 0.0
    if args.pick is not None or args.place is not None:  # probing: no reach limit
        cfg.events.reset_cube.params["reach_radius"] = None
        cfg.commands.goal_pose.reach_radius = None
    if args.video:
        for name in VIEWS:
            setattr(cfg.scene, f"cam_{name}", CameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Cam_{name}", update_period=0.0, width=800, height=600, data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.01, 20.0)),
            ))
    env = gym.make(args.task, cfg=cfg).unwrapped
    obs, _ = env.reset()
    dev, n, dt = env.device, env.num_envs, env.step_dt
    robot, cube = env.scene["robot"], env.scene["object"]
    base = T(robot.data.root_pos_w).clone()
    state = mdp.pick_place_state(env)

    term = env.action_manager.get_term("arm_action")
    names = list(term._joint_names)
    dofs = term._joint_ids if not isinstance(term._joint_ids, slice) else list(range(len(names)))
    finger_ids = [robot.joint_names.index(f"right_fr3_finger_joint{i}") for i in (1, 2)]
    scale = float(term.cfg.scale)
    clip = float(list(term.cfg.clip.values())[0][1])
    tlim = torch.tensor(term.cfg.target_limits, device=dev)
    body = robot.body_names.index("right_fr3_link7")
    off_p = torch.tensor(UMI_TCP_OFFSET_POS, device=dev).repeat(n, 1)
    off_q = torch.tensor(UMI_TCP_OFFSET_ROT, device=dev).repeat(n, 1)
    eye6, eye7 = torch.eye(6, device=dev), torch.eye(7, device=dev)
    q_ready = T(robot.data.default_joint_pos)[:, dofs].clone()

    def tcp():
        p, q = T(robot.data.body_pos_w)[:, body], T(robot.data.body_quat_w)[:, body]
        r = quat_apply(q, off_p)
        return p + r, quat_mul(q, off_q), r

    # grasp yaw: TCP x heading psi with the closing direction (TCP y) along a cube face, nearest the ready heading
    tcp_p0, tcp_q0, _ = tcp()
    x0 = quat_apply(tcp_q0, torch.tensor([1.0, 0.0, 0.0], device=dev).repeat(n, 1))
    psi0 = torch.atan2(x0[:, 1], x0[:, 0])
    cq = T(cube.data.root_quat_w)
    phi = 2.0 * torch.atan2(cq[:, 2], cq[:, 3])  # cube starts upright: pure yaw
    k = torch.round(wrap(psi0 - phi) / (0.5 * math.pi))
    psi = phi + k * 0.5 * math.pi
    q_down = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).repeat(n, 1)  # pi about x: TCP z points down
    ref_q = quat_mul(rot_z(psi), q_down)

    # --obs_pose: running mean of the observed cube centre (base) and of the folded yaw (circular, period 90 deg)
    om = env.observation_manager
    obs_slice, i0 = {}, 0
    for name, dim in zip(om.active_terms["policy"], om.group_obs_term_dim["policy"]):
        obs_slice[name] = slice(i0, i0 + int(np.prod(dim)))
        i0 += int(np.prod(dim))
    est_sum = torch.zeros(n, 3, device=dev)
    est_cs = torch.zeros(n, 2, device=dev)
    est_k = torch.zeros(n, 1, device=dev)

    def observe(obs_dict, active):
        po = obs_dict["policy"]
        pos, quat = po[:, obs_slice["cube_position"]], po[:, obs_slice["cube_orientation"]]
        yaw4 = 8.0 * torch.atan2(quat[:, 2], quat[:, 3])  # 4 x yaw of an upright cube
        a = active[:, None].float()
        est_sum.add_(pos * a)
        est_cs.add_(torch.stack([torch.cos(yaw4), torch.sin(yaw4)], -1) * a)
        est_k.add_(a)

    def estimate():
        k_ = est_k.clamp(min=1.0)
        return est_sum / k_ + base, 0.25 * torch.atan2(est_cs[:, 1], est_cs[:, 0])

    cube0 = T(cube.data.root_pos_w).clone()
    goal_b = state.goal_b.clone()
    goal_w = goal_b + base
    start_b = cube0 - base
    grasp_dz = torch.tensor([0.0, 0.0, mdp.GRASP_OFFSET if args.grasp_dz is None else args.grasp_dz], device=dev)
    grasp_w = cube0 + grasp_dz
    if args.obs_pose:
        observe(obs, torch.ones(n, dtype=torch.bool, device=dev))
    ref_p = tcp_p0.clone()
    phase = torch.full((n,), PRE, dtype=torch.long, device=dev)
    wait = torch.zeros(n, device=dev)
    grasp_off = torch.zeros(n, 3, device=dev)  # TCP - cube centre at the end of CLOSE
    t_phase = torch.full((n, len(PHASES)), float("nan"), device=dev)
    t_phase[:, PRE] = 0.0
    err_max = torch.zeros(n, device=dev)
    margin_min = torch.full((n,), 9.0, device=dev)
    sat_steps = torch.zeros(n, device=dev)
    t_success = torch.full((n,), float("nan"), device=dev)
    xy_at = torch.full((n, 2), float("nan"), device=dev)  # cube xy error when LOWER ends (gripper opens), when OPEN ends
    tilt_held_max = torch.zeros(n, device=dev)
    # grasp: est error when CLOSE starts; tilt while closing; tilt, push, fingertip height when CLOSE ends; finger-table force
    est_err = torch.full((n, 2), float("nan"), device=dev)  # xy (m), yaw (rad)
    tilt_close_max = torch.zeros(n, device=dev)
    at_grasp = torch.full((n, 4), float("nan"), device=dev)  # tilt (deg), push xy (m), fingertip above table (m), depth (m)
    table_f_max = torch.zeros(n, device=dev)
    tip_min = torch.full((n, len(PHASES)), float("inf"), device=dev)  # lowest fingertip above the table, per phase
    floor_steps = torch.zeros(n, len(PHASES), device=dev)  # steps with the task's tip_floor penalty on, per phase
    pad_f = []  # (min, max) pad-cube force on held steps
    link_names = [nm for nm in env.scene.sensors if nm.endswith("_contact") and "finger" not in nm]
    link_f = torch.zeros(n, len(link_names), device=dev)
    frames = []

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

    gripper = env.action_manager.get_term("gripper_action")

    def gripper_settled():
        return getattr(gripper, "settled", torch.ones(n, dtype=torch.bool, device=dev))

    gripper_s = torch.zeros(n, 2, device=dev)  # CLOSE / OPEN wait
    # the script knows the true table height (the goal command keeps the nominal one, like the real node)
    table_dz = mdp.table_dz(env)
    k_f = T(robot.data.joint_stiffness)[:, finger_ids[0]]
    v_f = T(robot.data.joint_vel_limits)[:, finger_ids[0]]
    print(f"[rule] DR: table dz {100 * table_dz.min():+.2f}..{100 * table_dz.max():+.2f} cm | finger stiffness "
          f"{k_f.min():.0f}..{k_f.max():.0f} N/m (x 0.0234 m = {0.0234 * k_f.min():.1f}..{0.0234 * k_f.max():.1f} N) | "
          f"finger speed {1000 * v_f.min():.1f}..{1000 * v_f.max():.1f} mm/s | start q offset "
          f"{(T(robot.data.joint_pos)[:, dofs] - q_ready).abs().max():.3f} rad max")

    steps = int(args.duration / dt)
    for step in range(steps):
        if args.obs_pose:
            est_on = phase <= DESC
            cube_est, phi_est = estimate()
            grasp_w = torch.where(est_on[:, None], cube_est + grasp_dz, grasp_w)
            k_est = torch.round(wrap(psi0 - phi_est) / (0.5 * math.pi))
            psi = torch.where(est_on, phi_est + k_est * 0.5 * math.pi, psi)
            ref_q = quat_mul(rot_z(psi), q_down)
        p, q, r = tcp()
        cube_w = T(cube.data.root_pos_w)
        up = torch.tensor([0.0, 0.0, 1.0], device=dev)
        carry_z = grasp_w[:, 2] + args.lift_dz
        # closed loop on the cube while carrying: put the TCP where the cube lands on the goal (in-hand slip, tilt)
        carrying = (phase == MOVE) | (phase == LOWER)
        place_xy = torch.where(carrying[:, None], p[:, :2] - (cube_w[:, :2] - goal_w[:, :2]), goal_w[:, :2] + grasp_off[:, :2])
        target = torch.stack([
            grasp_w + up * args.pre_dz,                                                        # PRE
            grasp_w,                                                                           # DESC
            ref_p,                                                                             # CLOSE
            torch.cat([ref_p[:, :2], carry_z[:, None]], -1),                                   # LIFT
            torch.cat([place_xy, carry_z[:, None]], -1),                                       # MOVE
            torch.cat([place_xy, (goal_w[:, 2] + table_dz + grasp_off[:, 2] + args.place_dz)[:, None]], -1),  # LOWER
            ref_p,                                                                             # OPEN
            torch.cat([ref_p[:, :2], (goal_w[:, 2] + grasp_off[:, 2] + 0.12)[:, None]], -1),  # UP
            ref_p,                                                                             # DONE
        ], 1)[torch.arange(n, device=dev), phase]
        ref_p = step_toward(ref_p, target, args.v_move * dt)
        at_ref = torch.norm(ref_p - target, dim=-1) < 1e-4
        near = torch.norm(p - target, dim=-1) < torch.where(phase == DESC, 0.005, 0.01)
        holding = (phase == CLOSE) | (phase == OPEN)
        wait = torch.where(holding, wait + dt, torch.zeros_like(wait))
        on_goal = torch.norm(cube_w[:, :2] - goal_w[:, :2], dim=-1) < args.place_tol
        adv = (~holding & (phase != DONE) & at_ref & near & ((phase != LOWER) | on_goal)) | (
            holding & (wait >= args.wait_s) & gripper_settled())
        for k, ph in enumerate((CLOSE, OPEN)):
            gripper_s[:, k] = torch.where(adv & (phase == ph), wait, gripper_s[:, k])
        ending_close = adv & (phase == CLOSE)
        starting_close = adv & (phase == DESC)
        if starting_close.any():
            cube_q_now = T(cube.data.root_quat_w)
            yaw_true = 2.0 * torch.atan2(cube_q_now[:, 2], cube_q_now[:, 3])
            e_yaw = wrap(4.0 * (psi - yaw_true)) / 4.0  # closing direction vs the true cube face (period 90 deg)
            est_err = torch.where(starting_close[:, None], torch.stack(
                [torch.norm(grasp_w[:, :2] - cube_w[:, :2], dim=-1), e_yaw.abs()], -1), est_err)
        grasp_off = torch.where(ending_close[:, None], p - cube_w, grasp_off)
        # OPEN: hold the hand where it is (the reference may still lead the TCP by up to 1 cm and drag the cube)
        ref_p = torch.where((adv & (phase == LOWER))[:, None], p, ref_p)
        for ph, k in ((LOWER, 0), (OPEN, 1)):
            snap = adv & (phase == ph)
            xy_at[:, k] = torch.where(snap, torch.norm(cube_w[:, :2] - goal_w[:, :2], dim=-1), xy_at[:, k])
        phase = torch.where(adv, phase + 1, phase)
        for ph in range(1, len(PHASES)):
            t_phase[:, ph] = torch.where((phase == ph) & torch.isnan(t_phase[:, ph]), step * dt, t_phase[:, ph])

        # ---- DLS IK -> arm action ----
        e_p, e_r = compute_pose_error(p, q, ref_p, ref_q, rot_error_type="axis_angle")
        err_max = torch.maximum(err_max, torch.norm(e_p, dim=-1))
        jac = T(robot.data.body_link_jacobian_w)[:, body - 1][:, :, dofs]
        jac = torch.cat([jac[:, :3] - skew(r) @ jac[:, 3:], jac[:, 3:]], 1)
        qj = T(robot.data.joint_pos)[:, dofs]
        lo, hi = tlim[:, 0], tlim[:, 1]
        push = torch.relu(0.3 - (qj - lo)) - torch.relu(0.3 - (hi - qj))
        jt = jac.transpose(1, 2)
        jpinv = jt @ torch.linalg.inv(jac @ jt + (0.05**2) * eye6)
        dq = (jpinv @ (args.gain * torch.cat([e_p, e_r], -1))[..., None])[..., 0]
        dq += ((eye7 - jpinv @ jac) @ (args.null * (q_ready - qj) + args.lim_k * push)[..., None])[..., 0]
        sat_steps += (dq.abs() >= clip).any(-1).float()
        dq = dq * torch.clamp(clip / dq.abs().amax(-1, keepdim=True).clamp(min=1e-9), max=1.0)
        margin_min = torch.minimum(margin_min, torch.minimum(qj - lo, hi - qj).amin(-1))
        closed = (phase >= CLOSE) & (phase < OPEN)
        actions = torch.cat([dq / scale, torch.where(closed, -1.0, 1.0)[:, None]], -1)
        obs = env.step(actions)[0]
        if args.obs_pose:
            observe(obs, phase <= DESC)

        s = mdp.pick_place_state(env)
        tilt_deg = torch.rad2deg(s.tilt)
        tilt_close_max = torch.where(phase == CLOSE, torch.maximum(tilt_close_max, tilt_deg), tilt_close_max)
        tip_h = s.tcp_b[:, 2] - (mdp.TABLE_TOP_Z_B + table_dz)
        depth = s.cube_b[:, 2] + 0.5 * mdp.CUBE_SIZE[2] - s.tcp_b[:, 2]
        at_grasp = torch.where(ending_close[:, None], torch.stack(
            [tilt_deg, torch.norm(s.cube_b[:, :2] - start_b[:, :2], dim=-1), tip_h, depth], -1), at_grasp)
        table_f_max = torch.maximum(table_f_max, mdp.finger_table_force(env))
        tip_min[torch.arange(n, device=dev), phase] = torch.minimum(tip_min[torch.arange(n, device=dev), phase], tip_h)
        floor_steps[torch.arange(n, device=dev), phase] += (mdp.tip_below_floor(env) > 0).float()
        t_success = torch.where((s.stage == mdp.STAGE_SUCCESS) & torch.isnan(t_success), (step + 1) * dt, t_success)
        tilt_held_max = torch.maximum(tilt_held_max, torch.rad2deg(s.tilt) * (s.lifted & s.grasp).float())
        held = s.lifted & s.grasp
        if held.any():
            # |force|, |force . closing axis (TCP y)| per pad, finger joint positions
            y_tcp = quat_apply(q, torch.tensor([0.0, 1.0, 0.0], device=dev).repeat(n, 1))
            fs = [T(env.scene.sensors[nm].data.force_matrix_w)[:, :, 0, :].sum(1) for nm in mdp.PAD_SENSORS]
            pad_f.append(torch.cat([torch.stack([torch.norm(f, dim=-1) for f in fs], -1),
                                    torch.stack([(f * y_tcp).sum(-1).abs() for f in fs], -1),
                                    T(robot.data.joint_pos)[:, finger_ids]], -1)[held])
        link_f = torch.maximum(link_f, torch.stack([torch.norm(T(env.scene.sensors[nm].data.net_forces_w), dim=-1).amax(-1)
                                                    for nm in link_names], -1))
        if args.video:
            render()
            frames.append(np.concatenate(
                [T(env.scene[f"cam_{nm}"].data.output["rgb"])[0].cpu().numpy()[..., :3].astype(np.uint8) for nm in VIEWS], 1))
        if step % 20 == 0 or step == steps - 1:
            print(f"[rule] t {step * dt:4.1f} s | env 0 phase {PHASES[int(phase[0])]:5s} | stage "
                  f"{mdp.STAGE_NAMES[int(s.stage[0])]:8s} | cube (base) {np.round(s.cube_b[0].cpu().numpy(), 3).tolist()} | "
                  f"TCP err {100 * torch.norm(p[0] - ref_p[0]):.1f} cm | phases done {int((phase == DONE).sum())}/{n}")

    # ---- result ----
    s = mdp.pick_place_state(env)
    success = s.stage == mdp.STAGE_SUCCESS
    xy_err = 100 * s.d_goal_xy
    tilt = torch.rad2deg(s.tilt)
    print("[rule] ---- result ----")
    for i in range(n):
        if bool(success[i]):
            continue
        tp = ", ".join(f"{PHASES[j]} {t_phase[i, j]:.2f}" for j in range(len(PHASES)) if not torch.isnan(t_phase[i, j]))
        print(f"[rule] FAIL env {i}: pick {np.round(start_b[i].cpu().numpy()[:2], 3).tolist()} yaw "
              f"{math.degrees(float(phi[i])):.0f} deg -> place {np.round(goal_b[i].cpu().numpy()[:2], 3).tolist()} | stage "
              f"{mdp.STAGE_NAMES[int(s.stage[i])]} | phase {PHASES[int(phase[i])]} | xy err {xy_err[i]:.1f} cm, tilt "
              f"{tilt[i]:.1f} deg, max tilt held {tilt_held_max[i]:.0f} deg | xy err at open {100 * xy_at[i, 0]:.1f} cm, "
              f"after open {100 * xy_at[i, 1]:.1f} cm | max TCP err {100 * err_max[i]:.1f} cm | "
              f"min joint margin {margin_min[i]:.3f} rad | phase start (s): {tp}")
        if n <= 8:
            e_p, e_r = compute_pose_error(*tcp()[:2], ref_p, ref_q, rot_error_type="axis_angle")
            hit = ", ".join(f"{nm[:-8]} {link_f[i, j]:.0f} N" for j, nm in enumerate(link_names) if link_f[i, j] > 1.0)
            print(f"[rule]   env {i}: TCP pos err {np.round(100 * e_p[i].cpu().numpy(), 1).tolist()} cm, rot err "
                  f"{math.degrees(float(torch.norm(e_r[i]))):.1f} deg | q {np.round(T(robot.data.joint_pos)[i, dofs].cpu().numpy(), 2).tolist()} "
                  f"| arm contacts > 1 N: {hit or 'none'}")
    ok = success.cpu().numpy()
    print(f"[rule] success {int(success.sum())}/{n} (task stage: xy < {100 * mdp.XY_TOL:.1f} cm, tilt < {mdp.UPRIGHT_DEG:.0f} deg, "
          f"released, TCP {100 * mdp.HAND_CLEAR:.0f} cm above the cube) | first success "
          f"{np.nanmin(t_success.cpu().numpy()) if ok.any() else float('nan'):.2f}-"
          f"{np.nanmax(t_success.cpu().numpy()) if ok.any() else float('nan'):.2f} s")
    reached = ~torch.isnan(t_success)
    print(f"[rule] success reached at least once {int(reached.sum())}/{n} (same count as the RL eval's 'stages reached ... success')")
    ts = t_success[success].cpu().numpy()
    if len(ts):
        print(f"[rule] first success time: median {np.median(ts):.2f} s, p90 {np.percentile(ts, 90):.2f} s, max {ts.max():.2f} s, "
              f"within the {episode_s:.0f} s episode {int((ts <= episode_s).sum())}/{n}")
    for k, ph in enumerate(("CLOSE", "OPEN")):
        w = gripper_s[:, k][gripper_s[:, k] > 0].cpu().numpy()
        if len(w):
            print(f"[rule] {ph} wait (command -> gripper done): median {np.median(w):.2f} s, min {w.min():.2f}, max {w.max():.2f} s")
    if pad_f:
        f = torch.cat(pad_f).cpu().numpy()
        print(f"[rule] while lifted and grasped (median): pad-cube force {np.median(f[:, 0]):.1f} / {np.median(f[:, 1]):.1f} N, "
              f"along the closing axis {np.median(f[:, 2]):.1f} / {np.median(f[:, 3]):.1f} N (left / right pad) | finger q "
              f"{1000 * np.median(f[:, 4]):.1f} / {1000 * np.median(f[:, 5]):.1f} mm")
    def pct(x, scale=1.0):
        x = x[~torch.isnan(x)].cpu().numpy() * scale
        return f"median {np.median(x):.1f}, p90 {np.percentile(x, 90):.1f}, max {x.max():.1f}" if len(x) else "-"

    gdz = mdp.GRASP_OFFSET if args.grasp_dz is None else args.grasp_dz
    print(f"[rule] grasp ({'policy cube obs' if args.obs_pose else 'true cube pose'}, grasp point {100 * gdz:+.1f} cm above the "
          f"cube centre = fingertip {100 * (0.5 * mdp.CUBE_SIZE[2] + gdz):.1f} cm above the table):")
    print(f"[rule]   pose error when closing starts: xy {pct(est_err[:, 0], 1000)} mm | closing direction vs cube face "
          f"{pct(est_err[:, 1], 180 / math.pi)} deg")
    print(f"[rule]   fingertip above the table at grasp {pct(at_grasp[:, 2], 100)} cm (min "
          f"{100 * np.nanmin(at_grasp[:, 2].cpu().numpy()):.1f}) | grasp depth from the cube top {pct(at_grasp[:, 3], 100)} cm")
    print(f"[rule]   cube tilt while closing (max) {pct(tilt_close_max)} deg | at grasp {pct(at_grasp[:, 0])} deg | "
          f"pushed while reaching+closing {pct(at_grasp[:, 1], 1000)} mm | tilt while held (max) {pct(tilt_held_max)} deg")
    print(f"[rule]   finger-table contact > 1 N: {int((table_f_max > 1.0).sum())}/{n} envs, max {table_f_max.max():.0f} N | lowest "
          f"fingertip above the table per phase (cm, min over envs): " + ", ".join(
              f"{PHASES[j]} {100 * tip_min[:, j].min():.1f}" for j in range(len(PHASES)) if torch.isfinite(tip_min[:, j]).any()))
    print(f"[rule]   steps under the fingertip floor ({100 * mdp.TIP_FLOOR:.0f} cm, over the table) per phase, mean over envs "
          f"(envs with any): " + ", ".join(f"{PHASES[j]} {floor_steps[:, j].mean():.1f} ({int((floor_steps[:, j] > 0).sum())})"
                                            for j in range(len(PHASES)) if floor_steps[:, j].any()))
    print(f"[rule] all envs: xy err median {xy_err.median():.2f} cm, max {xy_err.max():.2f} cm | final tilt max {tilt.max():.1f} deg | "
          f"tilt while held max {tilt_held_max.max():.0f} deg | max TCP err {100 * err_max.max():.1f} cm | min joint margin "
          f"{margin_min.min():.3f} rad | saturated steps mean {sat_steps.mean():.0f}")
    # reach map: pick = DESC finished (got to CLOSE), place = LOWER finished among those that got to MOVE
    picked = (phase >= CLOSE).cpu().numpy()
    moved = (phase >= MOVE).cpu().numpy()
    placed = (phase >= OPEN).cpu().numpy()
    xs = np.linspace(mdp.CUBE_X_B[0], mdp.CUBE_X_B[1], 5)
    ys = np.linspace(mdp.CUBE_Y_B[0], mdp.CUBE_Y_B[1], 6)
    for label, pos, ok_k, pool in (("pick reached", start_b, picked, np.ones(n, bool)), ("place reached", goal_b, placed, moved)):
        px, py = pos[:, 0].cpu().numpy(), pos[:, 1].cpu().numpy()
        print(f"[rule] {label} (rows x, cols y; hits/tried):")
        print("[rule]   x \\ y      " + " ".join(f"{ys[j]:+.2f}~{ys[j + 1]:+.2f}" for j in range(len(ys) - 1)))
        for i in range(len(xs) - 1):
            cells = []
            for j in range(len(ys) - 1):
                m = pool & (px >= xs[i]) & (px < xs[i + 1] + 1e-9) & (py >= ys[j]) & (py < ys[j + 1] + 1e-9)
                cells.append(f"{int(ok_k[m].sum()):>5d}/{int(m.sum()):<5d}")
            print(f"[rule]   {xs[i]:.2f}~{xs[i + 1]:.2f} " + " ".join(cells))
    print(f"[rule] RL obs dims {obs_dims}")
    rm = env.reward_manager
    print("[rule] RL reward terms summed over the run (weighted, x dt; mean / min / max over envs):")
    for name in rm.active_terms:
        v = rm._episode_sums[name]
        print(f"[rule]   {name:18s} {v.mean():9.3f} {v.min():9.3f} {v.max():9.3f}")

    if args.video:
        import imageio.v2 as imageio

        os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
        imageio.mimwrite(args.video, frames, fps=round(1.0 / dt))
        print("[rule] wrote", args.video)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
