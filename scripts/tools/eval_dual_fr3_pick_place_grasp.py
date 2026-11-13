"""Grasp height, finger-table contact, tilt and stage reach for a cube pick-and-place checkpoint (11-09).

Deterministic actions, one episode per env. Drop/topple/off-table terminations are off by default so every env runs the
whole 14 s (stage counts are then "reached at least once", not the training success rate; ``--keep_terminations`` keeps
them). Prints:
  - at the first two-pad grasp: fingertip (TCP) height above the env's table, finger overlap from the cube top, cube tilt
  - the lowest fingertip height while grasping a cube that is still near the table
  - finger-table contact (``mdp.finger_table_force``, pad sensor filter Table): envs > 1 N, peak force, time
  - grasp overlap bins vs tilt at grasp and max tilt while held (does a deeper grasp hold the cube straighter?)
  - stages reached, time in PLACED, how often the policy requests OPEN in PLACED / HELD, cube xy error to the goal

    python scripts/tools/eval_dual_fr3_pick_place_grasp.py --headless \
        --task Isaac-PickPlace-Cube-DualFR3-UMI-GripperBusy-Play-v0 --checkpoint logs/rsl_rl/<experiment>/<run>/model_N.pt

Use the -Play task of the run's variant (the actor/critic observation sizes must match the checkpoint). The env is built
from the current source, so for an old checkpoint the stage rules and rewards are today's, not the ones it trained with.
"""
import argparse
import importlib.metadata as metadata

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-PickPlace-Cube-DualFR3-UMI-GripperBusy-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--keep_terminations", action="store_true", help="keep drop/topple/off-table terminations")
# stress conditions shared with eval_dual_fr3_pick_place_rule_based.py (rule-based vs RL, 2026-11-13)
parser.add_argument("--seed", type=int, default=None, help="env seed (DR samples)")
parser.add_argument("--corrupt", action="store_true", help="enable the policy observation corruption (Play cfgs turn it off)")
parser.add_argument("--settle", type=float, nargs=2, default=None, help="gripper settle (command -> result) range [s]")
parser.add_argument("--cube_mu", type=float, nargs=2, default=None, help="cube static friction range (dynamic = 0.85x)")
parser.add_argument("--mass", type=float, nargs=2, default=None, help="cube mass range [kg]")
parser.add_argument("--finger_scale", type=float, default=None, help="scale the finger PD stiffness DR range (squeeze force)")
parser.add_argument("--kp_scale", type=float, default=None, help="arm PD stiffness scale")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import dual_fr3_lab.tasks  # noqa: E402,F401
from dual_fr3_lab.tasks.manager_based.pick_place_cube import mdp as M  # noqa: E402

env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
agent_cfg = handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point"),
                                         metadata.version("rsl-rl-lib"))
env_cfg.scene.num_envs = args.num_envs
if args.seed is not None:
    env_cfg.seed = args.seed


def apply_conditions(cfg):
    """--corrupt/--settle/--cube_mu/--mass/--finger_scale/--kp_scale on the env cfg (same code in the rule-based script)."""
    if args.corrupt:
        cfg.observations.policy.enable_corruption = True
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
          f" finger stiffness scale {cfg.events.finger_gains.params['stiffness_distribution_params']}, arm kp scale {args.kp_scale},"
          f" seed {cfg.seed}")


apply_conditions(env_cfg)
if not args.keep_terminations:
    for name in ("cube_dropped", "cube_toppled", "cube_off_table"):
        setattr(env_cfg.terminations, name, None)
    env_cfg.rewards.drop = None  # its term_keys name the terminations above
env = gym.make(args.task, cfg=env_cfg)
raw = env.unwrapped
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=raw.device)
grip = raw.action_manager.get_term(M.GRIPPER_ACTION)

n, T = raw.num_envs, int(raw.max_episode_length) - 1
keys = ("stage", "grasp", "tip", "overlap", "cube_h", "table_f", "tilt", "open_req", "xy_goal", "done")
rec = {k: [] for k in keys}
obs = env.get_observations()
with torch.inference_mode():
    for _ in range(T):
        obs, _, dones, _ = env.step(policy(obs))
        s = M.pick_place_state(raw)
        table = M.TABLE_TOP_Z_B + M.table_dz(raw)
        rec["stage"].append(s.stage.clone())
        rec["grasp"].append(s.grasp.clone())
        rec["tip"].append(s.tcp_b[:, 2] - table)
        rec["overlap"].append(s.cube_b[:, 2] + 0.5 * M.CUBE_SIZE[2] - s.tcp_b[:, 2])
        rec["cube_h"].append(s.cube_b[:, 2] - table)
        rec["table_f"].append(M.finger_table_force(raw))
        rec["tilt"].append(torch.rad2deg(s.tilt))
        rec["open_req"].append(~grip._request_close)
        rec["xy_goal"].append(s.d_goal_xy.clone())
        rec["done"].append(dones.bool())
R = {k: torch.stack(v) for k, v in rec.items()}  # (T, N)
if args.keep_terminations:  # count each env's first episode only
    alive = (R["done"].cumsum(0) - R["done"].long()) == 0
    for k in ("stage", "grasp", "open_req"):
        R[k] = R[k] & alive if R[k].dtype == torch.bool else torch.where(alive, R[k], torch.zeros_like(R[k]))
    R["table_f"] = R["table_f"] * alive
dt = raw.step_dt


def q(x, scale=1.0, unit=""):
    x = x.float() * scale
    if x.numel() == 0:
        return "-"
    return f"median {x.median():.1f}, min {x.min():.1f}, max {x.max():.1f}{unit} (n {x.numel()})"


g = R["grasp"]
ok = g.any(0)
first = g.float().argmax(0)
at_first = lambda k: R[k].gather(0, first[None])[0][ok]  # noqa: E731
print(f"\n=====  {args.checkpoint}")
print(f"envs {n}, steps {T} ({T * dt:.1f} s), terminations {'on' if args.keep_terminations else 'off'}, "
      f"grasp target: fingertip {100 * (0.5 * M.CUBE_SIZE[2] + M.GRASP_OFFSET):.1f} cm above the table, tip floor "
      f"{100 * M.TIP_FLOOR:.0f} cm")
print(f"grasped (both pads) : {int(ok.sum())}/{n}")
print(f"at first grasp      : fingertip above table {q(at_first('tip'), 100, ' cm')}")
print(f"                      overlap from cube top {q(at_first('overlap'), 100, ' cm')}")
print(f"                      cube tilt {q(at_first('tilt'), 1, ' deg')}")
low = torch.where(g & (R["cube_h"] < 0.5 * M.CUBE_SIZE[2] + M.REST_TOL), R["tip"], torch.full_like(R["tip"], 9.0)).amin(0)
print(f"lowest fingertip while grasping a cube on the table: {q(low[low < 9.0], 100, ' cm')}")
fmax = R["table_f"].amax(0)
ftime = (R["table_f"] > 1.0).float().sum(0) * dt
print(f"finger-table contact: envs > 1 N {int((fmax > 1.0).sum())}/{n}, peak {q(fmax[fmax > 1.0], 1, ' N')}, "
      f"time {q(ftime[fmax > 1.0], 1, ' s')}")
for st in range(M.NUM_STAGES):
    m = R["stage"] == st
    if m.any():
        print(f"  stage {M.STAGE_NAMES[st]:8s}: fingertip median {100 * R['tip'][m].median():.1f} cm, "
              f"finger-table > 1 N in {100 * (R['table_f'][m] > 1.0).float().mean():.0f}% of steps")

held = R["stage"] >= 1
tilt_held = torch.where(R["stage"] == 1, R["tilt"], torch.zeros_like(R["tilt"])).amax(0)
overlap, tilt_g, mx = at_first("overlap"), at_first("tilt"), R["stage"].amax(0)
print("overlap vs tilt (first grasp; max tilt in HELD):")
for lo, hi in ((0.0, 0.03), (0.03, 0.05), (0.05, 0.07), (0.07, 0.085), (0.085, 1.0)):
    m = (overlap >= lo) & (overlap < hi)
    if m.any():
        mh = m & held.any(0)[ok]
        th = f"{tilt_held[ok][mh].median():.1f}" if mh.any() else "-"
        print(f"  {100 * lo:4.1f}-{min(100 * hi, 99):4.1f} cm: n {int(m.sum()):3d}, tilt at grasp {tilt_g[m].median():5.1f} deg, "
              f"max tilt held {th:>5s} deg, held {int(mh.sum())}, placed {int((m & (mx >= 2)[ok]).sum())}")

print("stages reached      : " + ", ".join(f"{M.STAGE_NAMES[k]} {int((mx >= k).sum())}/{n}" for k in range(1, M.NUM_STAGES)))
for st, name in ((1, "HELD"), (2, "PLACED")):
    m = R["stage"] == st
    if m.any():
        t_in = m.float().sum(0) * dt
        print(f"{name:6s}: time per env {q(t_in[mx >= st], 1, ' s')}, policy requests OPEN in "
              f"{100 * R['open_req'][m].float().mean():.1f}% of steps, cube xy to goal median "
              f"{100 * R['xy_goal'][m].median():.2f} cm")
env.close()
app.close()
