#!/usr/bin/env python
"""Replay a Reach policy and measure the simulated q_cmd -> q_meas response.

For the Snapshot action ablation, ``q_cmd`` is the absolute position target
captured at a 20-Hz policy tick.  This script traces every 100-Hz physics
substep and reports how much the position error shrinks during each 50-ms
policy window.  It is intentionally simulation-only: no robot connection is
opened.

Example:
  conda run -n env_isaaclab python scripts/sim2real/analyze_reach_action_tracking.py \\
    --task Isaac-Reach-DualFR3-Snapshot-Play-v0 --headless \\
    --checkpoint logs/rsl_rl/dual_fr3_reach_snapshot/2026-09-17_18-07-36/model_2999.pt
"""

import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Trace simulated q_cmd and q_meas for an RSL-RL Reach policy.")
parser.add_argument("--task", type=str, default="Isaac-Reach-DualFR3-Snapshot-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the RSL-RL model_*.pt checkpoint.")
parser.add_argument("--policy-steps", type=int, default=120, help="Number of 20-Hz policy steps to trace (< 160).")
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("logs/sim2real/action_tracking"),
    help="Directory in which trace.csv and summary.json are written.",
)
parser.add_argument("--seed", type=int, default=42, help="Simulation seed for the evaluation rollout.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.policy_steps <= 0:
    parser.error("--policy-steps must be positive")

# hydra_task_config must only see Hydra overrides, not this script's CLI flags.
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import dual_fr3_lab.tasks  # noqa: F401  Register project tasks.

from dual_fr3_lab.tasks.manager_based.reach.snapshot_joint_action import (
    SnapshotRelativeJointPositionAction,
)


def _make_runner(env: RslRlVecEnvWrapper, agent_cfg: RslRlBaseRunnerCfg) -> OnPolicyRunner:
    """Construct RSL-RL runner, accommodating the Lab/RSL-RL version pair."""
    agent_dict = agent_cfg.to_dict()
    try:
        import inspect

        from rsl_rl.algorithms import PPO

        allowed = set(inspect.signature(PPO.__init__).parameters) | {"class_name"}
        algorithm_cfg = agent_dict.get("algorithm", {})
        for key in list(algorithm_cfg):
            if key not in allowed:
                algorithm_cfg.pop(key)
    except Exception:
        pass
    return OnPolicyRunner(env, agent_dict, log_dir=None, device=agent_cfg.device)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    if "Snapshot" not in args_cli.task:
        raise ValueError(
            "This tracer is for the controller-1 Snapshot task. "
            "Use Isaac-Reach-DualFR3-Snapshot-Play-v0 so q_cmd is an explicit held target."
        )

    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    term = unwrapped.action_manager.get_term("arm_action")
    if not isinstance(term, SnapshotRelativeJointPositionAction):
        raise RuntimeError(f"Expected SnapshotRelativeJointPositionAction, got {type(term).__name__}")

    resume_path = retrieve_file_path(args_cli.checkpoint)
    runner = _make_runner(env, agent_cfg)
    print(f"[INFO] Loading policy: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=unwrapped.device)

    joint_ids = term._joint_ids
    # Isaac Lab represents a contiguous joint selection as ``slice`` and a
    # non-contiguous selection as an index list; both are valid here.
    joint_names = list(robot.joint_names[joint_ids])
    traces: list[dict[str, float | int]] = []
    windows: list[dict[str, object]] = []
    current_policy_step = 0
    current_raw_action = torch.zeros(14, device=unwrapped.device)
    substep_in_window = 0

    # ActionManager calls this once for every physics substep.  Trace just
    # before simulation advances: this is exactly the q_cmd written to PhysX
    # and the q_meas on which the next PD torque is based.
    original_apply_actions = term.apply_actions

    def traced_apply_actions() -> None:
        nonlocal substep_in_window
        original_apply_actions()
        q_cmd = term.snapshot_joint_pos_target[0].detach().cpu().tolist()
        q_meas = robot.data.joint_pos[0, joint_ids].detach().cpu().tolist()
        q_vel = robot.data.joint_vel[0, joint_ids].detach().cpu().tolist()
        delta = term.processed_actions[0].detach().cpu().tolist()
        raw = current_raw_action.detach().cpu().tolist()
        row: dict[str, float | int] = {
            "policy_step": current_policy_step,
            "physics_substep": substep_in_window,
            "time_s": (current_policy_step * unwrapped.step_dt) + (substep_in_window * unwrapped.physics_dt),
        }
        for name, value in zip(joint_names, raw):
            row[f"action_{name}"] = value
        for name, value in zip(joint_names, delta):
            row[f"delta_q_{name}"] = value
        for name, value in zip(joint_names, q_cmd):
            row[f"q_cmd_{name}"] = value
        for name, value in zip(joint_names, q_meas):
            row[f"q_meas_{name}"] = value
        for name, value in zip(joint_names, q_vel):
            row[f"qd_meas_{name}"] = value
        traces.append(row)
        substep_in_window += 1

    term.apply_actions = traced_apply_actions
    env.reset()
    obs = env.get_observations()
    try:
        with torch.inference_mode():
            for policy_step in range(args_cli.policy_steps):
                current_policy_step = policy_step
                substep_in_window = 0
                action = policy(obs)
                current_raw_action = action[0].clone()
                first_trace = len(traces)
                obs, _, dones, _ = env.step(action)
                if bool(dones[0].item()):
                    raise RuntimeError("Episode reset during trace; choose --policy-steps below the 160-step horizon.")

                step_traces = traces[first_trace:]
                if len(step_traces) != unwrapped.cfg.decimation:
                    raise RuntimeError(
                        f"Expected {unwrapped.cfg.decimation} physics traces per policy tick, got {len(step_traces)}"
                    )
                q_cmd = term.snapshot_joint_pos_target[0].detach().cpu().tolist()
                q_start = [step_traces[0][f"q_meas_{name}"] for name in joint_names]
                q_end = robot.data.joint_pos[0, joint_ids].detach().cpu().tolist()
                q_cmd_drift = max(
                    abs(float(row[f"q_cmd_{name}"]) - q_cmd[j]) for row in step_traces for j, name in enumerate(joint_names)
                )
                windows.append(
                    {
                        "q_cmd": q_cmd,
                        "q_start": q_start,
                        "q_end": q_end,
                        "q_cmd_drift": q_cmd_drift,
                    }
                )
    finally:
        term.apply_actions = original_apply_actions

    # q_cmd - q_meas is the position-PD error.  Only ratio-test windows whose
    # starting error is meaningful; otherwise 0/0 around a settled joint is
    # not an informative "reduction percentage".
    start_abs = [[] for _ in joint_names]
    end_abs = [[] for _ in joint_names]
    reductions = [[] for _ in joint_names]
    moved_toward_target = [[] for _ in joint_names]
    for window in windows:
        for j in range(len(joint_names)):
            start_error = abs(float(window["q_cmd"][j]) - float(window["q_start"][j]))
            end_error = abs(float(window["q_cmd"][j]) - float(window["q_end"][j]))
            start_abs[j].append(start_error)
            end_abs[j].append(end_error)
            if start_error >= 0.002:
                reductions[j].append(1.0 - end_error / start_error)
                moved_toward_target[j].append(end_error < start_error)

    per_joint = []
    for j, name in enumerate(joint_names):
        per_joint.append(
            {
                "joint": name,
                "mean_abs_error_at_policy_tick_rad": _mean(start_abs[j]),
                "mean_abs_error_after_50ms_rad": _mean(end_abs[j]),
                "reduction_of_mean_abs_error_fraction": 1.0 - _mean(end_abs[j]) / _mean(start_abs[j]),
                "mean_50ms_error_reduction_fraction": _mean(reductions[j]),
                "windows_moved_toward_target_fraction": _mean(moved_toward_target[j]),
                "valid_reduction_windows": len(reductions[j]),
            }
        )
    all_start = [value for values in start_abs for value in values]
    all_end = [value for values in end_abs for value in values]
    all_reductions = [value for values in reductions for value in values]
    all_moved_toward_target = [value for values in moved_toward_target for value in values]
    summary = {
        "task": args_cli.task,
        "checkpoint": str(resume_path),
        "seed": args_cli.seed,
        "policy_steps": args_cli.policy_steps,
        "physics_dt_s": unwrapped.physics_dt,
        "policy_dt_s": unwrapped.step_dt,
        "physics_substeps_per_policy_step": unwrapped.cfg.decimation,
        "q_cmd_max_drift_inside_policy_window_rad": max(float(window["q_cmd_drift"]) for window in windows),
        "aggregate": {
            "mean_abs_error_at_policy_tick_rad": _mean(all_start),
            "mean_abs_error_after_50ms_rad": _mean(all_end),
            "reduction_of_mean_abs_error_fraction": 1.0 - _mean(all_end) / _mean(all_start),
            "mean_50ms_error_reduction_fraction": _mean(all_reductions),
            "windows_moved_toward_target_fraction": _mean(all_moved_toward_target),
            "valid_reduction_windows": len(all_reductions),
        },
        "per_joint": per_joint,
    }

    output_dir = args_cli.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.csv"
    summary_path = output_dir / "summary.json"
    with trace_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(traces[0]))
        writer.writeheader()
        writer.writerows(traces)
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)

    aggregate = summary["aggregate"]
    print("\n=== q_cmd -> q_meas, simulation-only replay ===")
    print(f"policy / physics: {unwrapped.step_dt * 1000:.0f} ms / {unwrapped.physics_dt * 1000:.0f} ms")
    print(f"q_cmd hold drift inside a policy window: {summary['q_cmd_max_drift_inside_policy_window_rad']:.3e} rad")
    print(
        "mean |q_cmd-q_meas|: "
        f"{aggregate['mean_abs_error_at_policy_tick_rad']:.5f} -> "
        f"{aggregate['mean_abs_error_after_50ms_rad']:.5f} rad "
        f"({aggregate['reduction_of_mean_abs_error_fraction'] * 100:.1f}% reduction)"
    )
    print(
        "mean 50-ms error reduction "
        f"(only |initial error| >= 0.002 rad): {aggregate['mean_50ms_error_reduction_fraction'] * 100:.1f}% "
        f"over {aggregate['valid_reduction_windows']} joint-windows"
    )
    print(
        "windows that moved toward q_cmd: "
        f"{aggregate['windows_moved_toward_target_fraction'] * 100:.1f}% "
        "(same non-trivial-error windows)"
    )
    for item in per_joint:
        print(
            f"  {item['joint']}: {item['mean_abs_error_at_policy_tick_rad']:.5f} -> "
            f"{item['mean_abs_error_after_50ms_rad']:.5f} rad, "
            f"mean reduction {item['reduction_of_mean_abs_error_fraction'] * 100:.1f}%, "
            f"moved {item['windows_moved_toward_target_fraction'] * 100:.1f}% "
            f"(n={item['valid_reduction_windows']})"
        )
    print(f"\nSaved: {trace_path}\nSaved: {summary_path}")
    sys.stdout.flush()
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    # Isaac Sim 5.0 may segfault on SimulationApp.close() after a successful
    # headless replay on this host.  Outputs were flushed, so bypass Kit close.
    os._exit(0)
