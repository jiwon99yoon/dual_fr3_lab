#!/usr/bin/env python
"""Verify controller-1 target hold semantics in the Snapshot Reach task.

The check injects a +0.05 rad left-joint1 delta once.  It verifies that the
absolute target captured at that policy tick is unchanged through the five
10-ms physics substeps of one 20-Hz RL step.

Run:
  conda run -n env_isaaclab python scripts/sim2real/reach_snapshot_action_check.py --headless
"""

import argparse
import os
import sys
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Verify controller-1 snapshot target hold semantics.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import dual_fr3_lab.tasks  # noqa: F401  Register project tasks.
from isaaclab_tasks.utils import parse_env_cfg

from dual_fr3_lab.tasks.manager_based.reach.snapshot_joint_action import (
    SnapshotRelativeJointPositionAction,
)


TASK = "Isaac-Reach-DualFR3-Snapshot-Play-v0"
DELTA_RAD = 0.05


def main() -> None:
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    env = gym.make(TASK, cfg=env_cfg)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    term = unwrapped.action_manager.get_term("arm_action")
    if not isinstance(term, SnapshotRelativeJointPositionAction):
        raise RuntimeError(f"Expected SnapshotRelativeJointPositionAction, got {type(term).__name__}")

    env.reset()
    joint_ids = term._joint_ids
    q_at_policy_tick = robot.data.joint_pos[:, joint_ids].clone()

    # scale=0.1, so raw action 0.5 becomes the requested +0.05 rad delta.
    action = torch.zeros(1, unwrapped.action_manager.total_action_dim, device=unwrapped.device)
    action[0, 0] = DELTA_RAD / 0.1
    unwrapped.action_manager.process_action(action)

    expected = q_at_policy_tick.clone()
    expected[0, 0] += DELTA_RAD
    snapshot_error = torch.max(torch.abs(term.snapshot_joint_pos_target - expected)).item()
    if snapshot_error > 1.0e-6:
        raise RuntimeError(f"Snapshot target mismatch: max error={snapshot_error:.3e} rad")

    held_target_errors = []
    for _ in range(unwrapped.cfg.decimation):
        unwrapped.action_manager.apply_action()
        target = robot.data.joint_pos_target[:, joint_ids].clone()
        held_target_errors.append(torch.max(torch.abs(target - expected)).item())
        unwrapped.scene.write_data_to_sim()
        unwrapped.sim.step(render=False)
        unwrapped.scene.update(unwrapped.physics_dt)

    max_hold_error = max(held_target_errors)
    if max_hold_error > 1.0e-6:
        raise RuntimeError(f"Target changed during hold window: max error={max_hold_error:.3e} rad")

    q_after = robot.data.joint_pos[:, joint_ids]
    q_motion = (q_after[0, 0] - q_at_policy_tick[0, 0]).item()
    print("PASS: controller-1 snapshot target is held for " f"{unwrapped.cfg.decimation} physics substeps.")
    print(f"  policy delta         = {DELTA_RAD:.6f} rad")
    print(f"  snapshot max error   = {snapshot_error:.3e} rad")
    print(f"  target hold max error= {max_hold_error:.3e} rad")
    print(f"  left joint1 movement = {q_motion:.6f} rad during 50 ms")
    # Isaac Sim shutdown may terminate quickly enough to bypass buffered output.
    sys.stdout.flush()
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        # See success path below: avoid the unstable Kit shutdown path on this host.
        os._exit(1)
    # Isaac Sim 5.0 on this host can segfault in SimulationApp.close() after a
    # completed headless check.  The result is already flushed above; exiting
    # here releases the process without turning a PASS into exit code 139.
    os._exit(0)
