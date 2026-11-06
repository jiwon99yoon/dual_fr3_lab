"""Door angle when the book reaches the cabinet opening, for a cabinet book-insert checkpoint (plan 7-13).

One env per start-table row, deterministic actions, one episode. Prints when the door passes 60 / 85 deg, the door angle
when the book reaches the lane and the opening, how many rows bring the book past the DoorFirst wait line
(BOOK_WAIT_Y_B) before the door is at 80 deg, tilt, depth, success and the reward terms summed over the episode.

    python scripts/tools/eval_dual_fr3_cabinet_book_door_entry.py --headless \
        --task Isaac-Cabinet-BookInsert-DualFR3-UMI-DoorFirst-Play-v0 --checkpoint logs/rsl_rl/<experiment>/<run>/model_N.pt

The task only sets the rewards and layout; observations are the same for v1, v2 and DoorFirst.
"""
import argparse
import importlib.metadata as metadata

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Cabinet-BookInsert-DualFR3-UMI-DoorFirst-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=121, help="start-table rows to run (env i on row i)")
parser.add_argument("--tag", default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import dual_fr3_lab.tasks  # noqa: E402,F401
from dual_fr3_lab.tasks.manager_based.cabinet_book_insert import cabinet_book_insert_env_cfg as C  # noqa: E402
from dual_fr3_lab.tasks.manager_based.cabinet_book_insert.mdp.observations import book_pose_b, door_angle_rad  # noqa: E402
from dual_fr3_lab.tasks.manager_based.cabinet_book_insert.mdp.rewards import insert_success  # noqa: E402

env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
agent_cfg = handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point"),
                                         metadata.version("rsl-rl-lib"))
env_cfg.scene.num_envs = args.num_envs
env_cfg.events.reset_start.params["sequential"] = True  # env i starts on row i
env = gym.make(args.task, cfg=env_cfg)
raw = env.unwrapped
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=raw.device)

sp = dict(raw.cfg.rewards.insert_success.params)
opening = sp["opening_x_b"]
obs = env.get_observations()
T = int(raw.max_episode_length) - 1
door, depth, tilt, succ, ys = [], [], [], [], []
with torch.inference_mode():
    for _ in range(T):
        obs, *_ = env.step(policy(obs))
        pos, quat = book_pose_b(raw)
        door.append(torch.rad2deg(door_angle_rad(raw)))
        depth.append(pos[:, 0] - opening)
        tilt.append(torch.rad2deg(2.0 * torch.acos(quat[:, 3].clamp(max=1.0))))
        succ.append(insert_success(raw, **sp))
        ys.append(pos[:, 1])
door, depth, tilt, succ, ys = (torch.stack(x).cpu().numpy() for x in (door, depth, tilt, succ, ys))  # (T, N)
dt = raw.step_dt


def first(mask):
    hit = mask.any(0)
    return np.where(hit, mask.argmax(0), -1)


def at(series, idx):
    n = np.arange(series.shape[1])
    return np.where(idx >= 0, series[np.clip(idx, 0, None), n], np.nan)


def stat(x, fmt="{:.0f}"):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return "-"
    return f"median {fmt.format(np.median(x))}, min {fmt.format(x.min())}, max {fmt.format(x.max())} (n {len(x)})"


half = 0.5 * C.BOOK_SIZE[0]
cab_y = raw.cfg.scene.cabinet.init_state.pos[1]
in_y = np.abs(ys - cab_y) < C.CABINET_INNER_HALF - 0.5 * C.BOOK_SIZE[1]  # book centre within the opening's y span
t_lane = first(np.abs(ys - sp["goal_pos_b"][1]) < 0.05)
t_front = first((depth >= -half) & in_y)  # book front edge at the opening plane (upright book), in front of it
t_ctr = first((depth >= 0.0) & in_y)
t_succ = first(succ > 0.5)
t_d85 = first(door >= 85.0)
t_d60 = first(door >= 60.0)
early = (ys > C.BOOK_WAIT_Y_B) & (door < 80.0)  # book past the DoorFirst wait line before the door is open
d_front, d_ctr = at(door, t_front), at(door, t_ctr)
print(f"\n===== {args.tag} {args.checkpoint}")
print(f"envs {door.shape[1]}, steps {T} ({T * dt:.1f} s)")
print(f"door >= 60 deg at   : {stat(np.where(t_d60 >= 0, t_d60 * dt, np.nan), '{:.2f}')} s")
print(f"door >= 85 deg at   : {stat(np.where(t_d85 >= 0, t_d85 * dt, np.nan), '{:.2f}')} s; never {int((t_d85 < 0).sum())}")
print(f"book y within 5 cm of lane: {stat(np.where(t_lane >= 0, t_lane * dt, np.nan), '{:.2f}')} s, door there {stat(at(door, t_lane))} deg")
print(f"book front at opening: {stat(np.where(t_front >= 0, t_front * dt, np.nan), '{:.2f}')} s")
print(f"  door there        : {stat(d_front)} deg; < 85 deg in {int(np.nansum(d_front < 85))}, "
      f"< 70 in {int(np.nansum(d_front < 70))}")
print(f"book centre at opening: door {stat(d_ctr)} deg; < 85 deg in {int(np.nansum(d_ctr < 85))}")
print(f"book past wait line (y > {C.BOOK_WAIT_Y_B:.2f}) with door < 80 deg: {int(early.any(0).sum())} rows, "
      f"{stat(np.where(early.any(0), early.sum(0) * dt, np.nan), '{:.2f}')} s")
print(f"door max            : {stat(door.max(0))} deg, final {stat(door[-1])} deg")
print(f"tilt at front entry : {stat(at(tilt, t_front))} deg, final {stat(tilt[-1])} deg")
print(f"final depth         : {stat(100 * depth[-1], '{:.1f}')} cm")
print(f"success ever {int((t_succ >= 0).sum())}, at end {int(succ[-1].sum())} / {door.shape[1]}")
rm = raw.reward_manager
tot = sum(v for v in rm._episode_sums.values())
print("reward sums (weighted, x dt; mean / min / max over envs):")
for k, v in rm._episode_sums.items():
    print(f"  {k:18s} {v.mean().item():8.3f} {v.min().item():8.3f} {v.max().item():8.3f}")
print(f"  {'TOTAL':18s} {tot.mean().item():8.3f} {tot.min().item():8.3f} {tot.max().item():8.3f}")
env.close()
app.close()
