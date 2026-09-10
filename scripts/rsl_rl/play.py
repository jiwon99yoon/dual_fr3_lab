# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

def _load_reach_goal_sequence(path: str) -> list[dict]:
    """Load and validate an explicit Reach goal sequence from JSON."""

    sequence_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(sequence_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to load goal sequence '{sequence_path}': {error}") from error

    goals = payload.get("goals") if isinstance(payload, dict) else payload
    if not isinstance(goals, list) or not goals:
        raise ValueError("goal sequence must be a non-empty JSON list or an object with a 'goals' list")

    validated = []
    for index, goal in enumerate(goals):
        if not isinstance(goal, dict):
            raise ValueError(f"goal {index} must be a JSON object")
        position = goal.get("position")
        duration_s = goal.get("duration_s")
        if (
            not isinstance(position, list)
            or len(position) != 3
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in position)
        ):
            raise ValueError(f"goal {index} position must contain three finite numbers")
        if (
            not isinstance(duration_s, (int, float))
            or not math.isfinite(duration_s)
            or duration_s <= 0.0
        ):
            raise ValueError(f"goal {index} duration_s must be a positive finite number")
        label = goal.get("label", f"goal_{index}")
        if not isinstance(label, str):
            raise ValueError(f"goal {index} label must be a string")
        validated.append(
            {
                "position": [float(value) for value in position],
                "duration_s": float(duration_s),
                "label": label,
            }
        )
    return validated


# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint with an RSL-RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--reach_goal_sequence",
    type=str,
    default=None,
    metavar="JSON",
    help="Reach-only JSON file containing explicit positions and durations; disables random goal sampling.",
)
parser.add_argument(
    "--reach_goal_sequence_loop",
    action="store_true",
    default=False,
    help="Repeat the explicit Reach goal sequence instead of exiting after its final duration.",
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
try:
    reach_goal_sequence = (
        _load_reach_goal_sequence(args_cli.reach_goal_sequence)
        if args_cli.reach_goal_sequence
        else None
    )
except ValueError as error:
    parser.error(str(error))
if reach_goal_sequence and (
    args_cli.task is None or "Isaac-Reach-DualFR3" not in args_cli.task
):
    parser.error("--reach_goal_sequence is supported only for Isaac-Reach-DualFR3 tasks")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import importlib.metadata as metadata
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
# IsaacLab 2.3+에서 pretrained_checkpoint 모듈 제거 -> 버전 호환 fallback
# (--use_pretrained_checkpoint 기능만 비활성화되고 일반 checkpoint 재생은 무관)
try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ImportError:

    def get_published_pretrained_checkpoint(*args, **kwargs):
        raise RuntimeError("IsaacLab 2.3+에서는 --use_pretrained_checkpoint 미지원 (모듈 제거됨)")

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import dual_fr3_lab.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    if reach_goal_sequence:
        command_cfg = env_cfg.commands.target_pose
        ranges = command_cfg.ranges
        limits = (ranges.pos_x, ranges.pos_y, ranges.pos_z)
        for index, goal in enumerate(reach_goal_sequence):
            for axis, value, (lower, upper) in zip("xyz", goal["position"], limits):
                if value < lower or value > upper:
                    raise ValueError(
                        f"goal {index} {axis}={value} is outside the training range [{lower}, {upper}]"
                    )
        # The script owns command transitions in this mode. Prevent both timed random
        # resampling and the normal eight-second episode timeout from replacing/resetting it.
        command_cfg.resampling_time_range = (1.0e9, 1.0e9)
        env_cfg.episode_length_s = 1.0e9

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_model_dir, exist_ok=True)

    # rsl_rl 3.x stores separate actor/critic MLPModel instances. The actor returned by
    # get_inference_policy() owns its observation normalizer and distribution export logic.
    if hasattr(policy, "as_jit") and hasattr(policy, "as_onnx"):
        jit_policy = policy.as_jit().cpu().eval()
        torch.jit.script(jit_policy).save(os.path.join(export_model_dir, "policy.pt"))

        onnx_policy = policy.as_onnx(verbose=False).cpu().eval()
        dummy_obs = torch.zeros(1, policy.obs_dim)
        torch.onnx.export(
            onnx_policy,
            dummy_obs,
            os.path.join(export_model_dir, "policy.onnx"),
            export_params=True,
            opset_version=18,
            input_names=["obs"],
            output_names=["actions"],
        )
    else:
        # Legacy rsl_rl actor-critic/student containers (Isaac Lab 2.x).
        if hasattr(runner.alg, "policy"):
            policy_nn = runner.alg.policy
        elif hasattr(runner.alg, "actor_critic"):
            policy_nn = runner.alg.actor_critic
        else:
            raise AttributeError("Unable to locate the policy module in the loaded rsl_rl runner.")

        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    sequence_term = None
    sequence_index = 0
    next_sequence_switch_s = math.inf

    def apply_sequence_goal(index: int, current_time_s: float):
        nonlocal next_sequence_switch_s
        goal = reach_goal_sequence[index]
        command = env.unwrapped.command_manager.get_command("target_pose")
        command[:, :3] = torch.tensor(
            goal["position"], device=command.device, dtype=command.dtype
        )
        command[:, 3:].zero_()
        command[:, 6] = 1.0  # quaternion xyzw = [0, 0, 0, 1]
        sequence_term.time_left.fill_(1.0e9)
        sequence_term._update_metrics()
        next_sequence_switch_s = current_time_s + goal["duration_s"]
        print(
            f"[REACH SEQUENCE] {index + 1}/{len(reach_goal_sequence)} "
            f"{goal['label']}: position={goal['position']} "
            f"duration={goal['duration_s']:.3f}s"
        )

    if reach_goal_sequence:
        sequence_term = env.unwrapped.command_manager.get_term("target_pose")
        apply_sequence_goal(sequence_index, 0.0)

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        elapsed_s = timestep * dt
        if reach_goal_sequence and elapsed_s + 1.0e-9 >= next_sequence_switch_s:
            sequence_index += 1
            if sequence_index >= len(reach_goal_sequence):
                if args_cli.reach_goal_sequence_loop:
                    sequence_index = 0
                else:
                    print("[REACH SEQUENCE] completed")
                    break
            apply_sequence_goal(sequence_index, elapsed_s)
            obs = env.get_observations()

        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
