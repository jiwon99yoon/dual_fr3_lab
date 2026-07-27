# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""dual_fr3 lift 정책의 강건성 평가 (sim2real 진단용).

동일 checkpoint를 Play-v0(학습 분포)과 Play-Noisy-v0(obs noise + 물성 DR)에서 각각 굴려
성능 하락 폭을 잰다. 지표는 스텝 단위 파지/들기 유지율과 goal 거리, 에피소드 단위 drop 수.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/eval_dual_fr3_lift_robustness.py \
        --task Isaac-Lift-Cube-DualFR3-Play-Noisy-v0 \
        --checkpoint logs/rsl_rl/dual_fr3_lift/2026-07-27_15-59-59/model_4999.pt --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate dual_fr3 lift policy robustness.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-DualFR3-Play-Noisy-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=4, help="에피소드 사이클 수 (사이클당 5s x num_envs)")
parser.add_argument("--no_obs_noise", action="store_true", help="obs noise만 끄기 (물성 DR 유지)")
parser.add_argument("--no_physics_dr", action="store_true", help="물성 DR만 끄기 (obs noise 유지)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import dual_fr3_lab.tasks  # noqa: F401  # 이 패키지의 gym id 등록
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from dual_fr3_lab.tasks.manager_based.lift_cube import mdp
from dual_fr3_lab.tasks.manager_based.lift_cube.agents.rsl_rl_ppo_cfg import LiftCubePPORunnerCfg
from isaaclab_tasks.utils import parse_env_cfg

LIFTED_H = 0.79  # 상판(0.75) + 4cm — env cfg의 lifted_height와 동일해야 함
SUCCESS_D = 0.05  # goal 5cm 이내면 "정밀 유지"로 카운트


def main():
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    # 축별 진단: noisy cfg에서 한 축만 끄고 평가 (원인 분리용)
    if args_cli.no_obs_noise:
        env_cfg.observations.policy.enable_corruption = False
    if args_cli.no_physics_dr:
        for name in ("randomize_cube_friction", "randomize_cube_mass"):
            if hasattr(env_cfg.events, name):
                delattr(env_cfg.events, name)
    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(wrapped, LiftCubePPORunnerCfg().to_dict(), log_dir=None, device="cuda:0")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")

    uenv = env.unwrapped
    f1 = SceneEntityCfg("leftfinger_object_contact")
    f2 = SceneEntityCfg("rightfinger_object_contact")

    steps_per_ep = int(uenv.max_episode_length)
    total_steps = steps_per_ep * args_cli.episodes
    obs = wrapped.get_observations()

    lifted_sum = grasp_sum = hold_sum = success_sum = 0.0
    err_when_lifted = []
    drops = 0
    for step in range(total_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = wrapped.step(actions)
        robot = uenv.scene["robot"]
        obj = uenv.scene["object"]
        # goal (base frame) -> world
        cmd = uenv.command_manager.get_command("object_pose")[:, :3]
        goal_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, cmd)
        err = torch.norm(goal_w - obj.data.root_pos_w, dim=1)
        lifted = (obj.data.root_pos_w[:, 2] > LIFTED_H).float()
        grasped = mdp.object_grasped(uenv, f1, f2, threshold=0.5)
        hold = lifted * grasped
        lifted_sum += lifted.mean().item()
        grasp_sum += grasped.mean().item()
        hold_sum += hold.mean().item()
        success_sum += (hold * (err < SUCCESS_D).float()).mean().item()
        if hold.any():
            err_when_lifted.append(err[hold.bool()].mean().item())
        drops += int(uenv.termination_manager.get_term("object_dropping").sum().item())
        if step % steps_per_ep == 0:
            print(f"[EVAL] episode cycle {step // steps_per_ep + 1}/{args_cli.episodes}", flush=True)

    n = float(total_steps)
    print("\n===== ROBUSTNESS EVAL =====", flush=True)
    print(f"task            : {args_cli.task}")
    print(f"checkpoint      : {args_cli.checkpoint}")
    print(f"envs x episodes : {args_cli.num_envs} x {args_cli.episodes} ({total_steps} steps)")
    print(f"lifted fraction : {lifted_sum / n:.3f}")
    print(f"grasped fraction: {grasp_sum / n:.3f}")
    print(f"hold (lift&grasp): {hold_sum / n:.3f}")
    print(f"precise hold (<{SUCCESS_D * 100:.0f}cm): {success_sum / n:.3f}")
    print(f"goal err while holding: {sum(err_when_lifted) / max(len(err_when_lifted), 1):.3f} m")
    print(f"drop terminations: {drops} (over {args_cli.num_envs * args_cli.episodes} episodes)", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
