# Copyright (c) 2026, DYROS.
# SPDX-License-Identifier: BSD-3-Clause
"""Dump a deployment-parity trace of the plate box-lift policy (obs, action, raw states).

For every policy step this records what the Isaac policy actually received and produced,
together with the raw quantities a real-robot node would have to rebuild the observation
from (joint states by name, robot root pose, box pose, goal command, previous action) and
the absolute joint target the action term held. The trace is consumed by
``fr3_husky_etri/scripts/tools/verify_boxlift_obs_parity.py`` (numpy only).

Use the PLAY task so observation corruption is off and rebuilt observations can match exactly.

.. code-block:: bash

    python scripts/tools/dump_boxlift_deploy_trace.py \
        --task Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Margin-Play-v0 \
        --checkpoint logs/rsl_rl/<experiment>/<run>/model_XXXX.pt \
        --num_envs 16 --steps 200 --output /path/to/trace.npz --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump a box-lift deployment parity trace.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=200, help="policy steps to record")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--output", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import hashlib  # noqa: E402
import importlib.metadata as metadata  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
import dual_fr3_lab.tasks  # noqa: E402, F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402


def _t(value) -> torch.Tensor:
    """Isaac Lab 3.0 data buffers may be warp-backed; return a torch tensor."""
    if hasattr(value, "torch"):
        return value.torch
    if isinstance(value, wp.array):
        return wp.to_torch(value)
    return value


def _np(value) -> np.ndarray:
    return _t(value).detach().cpu().numpy().copy()


def main():
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=uenv.device)

    robot = uenv.scene["robot"]
    box = uenv.scene["object"]
    arm_term = uenv.action_manager.get_term("arm_action")
    obs_mgr = uenv.observation_manager
    term_names = list(obs_mgr.active_terms["policy"])
    term_dims = [int(d[0] if isinstance(d, (tuple, list)) else d) for d in obs_mgr.group_obs_term_dim["policy"]]
    arm_joint_names = [robot.data.joint_names[j] for j in arm_term._joint_ids] if not isinstance(
        arm_term._joint_ids, slice) else list(robot.data.joint_names)
    target_limits = (
        arm_term._target_limits.detach().cpu().numpy() if arm_term._target_limits is not None else np.empty((0, 2))
    )

    rec = {k: [] for k in (
        "obs", "action", "joint_pos", "joint_vel", "root_pos_w", "root_quat_w", "box_pos_w", "box_quat_w",
        "command", "prev_action", "held_target", "done", "episode_step")}

    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            rec["obs"].append(_np(obs["policy"]))
            rec["joint_pos"].append(_np(robot.data.joint_pos))
            rec["joint_vel"].append(_np(robot.data.joint_vel))
            rec["root_pos_w"].append(_np(robot.data.root_pos_w))
            rec["root_quat_w"].append(_np(robot.data.root_quat_w))
            rec["box_pos_w"].append(_np(box.data.root_pos_w))
            rec["box_quat_w"].append(_np(box.data.root_quat_w))
            rec["command"].append(_np(uenv.command_manager.get_command("object_pose")))
            rec["prev_action"].append(_np(uenv.action_manager.action))
            rec["episode_step"].append(_np(uenv.episode_length_buf))
            actions = policy(obs)
            rec["action"].append(_np(actions))
            obs, _, dones, _ = env.step(actions)
            # Valid only where the env did not reset inside this step (reset overwrites the target).
            rec["held_target"].append(_np(arm_term.position_targets))
            rec["done"].append(_np(dones).astype(bool))

    ckpt = Path(args_cli.checkpoint).resolve()
    out = Path(args_cli.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        **{k: np.stack(v) for k, v in rec.items()},
        joint_names=np.asarray(robot.data.joint_names),
        arm_joint_names=np.asarray(arm_joint_names),
        default_joint_pos=_np(robot.data.default_joint_pos)[0],
        target_limits=target_limits,
        action_scale=np.asarray(float(arm_term.cfg.scale)),
        action_clip=np.asarray(str(arm_term.cfg.clip)),
        reference_mode=np.asarray(arm_term.cfg.reference_mode),
        obs_term_names=np.asarray(term_names),
        obs_term_dims=np.asarray(term_dims),
        physics_dt=np.asarray(float(uenv.cfg.sim.dt)),
        decimation=np.asarray(int(uenv.cfg.decimation)),
        task=np.asarray(args_cli.task),
        checkpoint=np.asarray(str(ckpt)),
        checkpoint_sha256=np.asarray(hashlib.sha256(ckpt.read_bytes()).hexdigest()),
    )
    print(f"[TRACE] wrote {out}: {args_cli.steps} steps x {args_cli.num_envs} envs; obs terms {list(zip(term_names, term_dims))}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
