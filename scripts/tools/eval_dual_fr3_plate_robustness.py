# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""dual_fr3 plate box-lift 정책의 강건성 평가 (sim2real 배포 전 관문 — cube 버전 이식).

동일 checkpoint를 Play-v0(corruption off)과 Play-Noisy-v0(obs noise on)에서 각각 굴려
성능 하락 폭을 잰다. 물성/크기 DR은 두 cfg 모두 활성(학습 분포). plate 고유 지표:
env별 판정 높이(안착중심+LIFT_DELTA), 양판 squeeze 유지율, 운반 중 박스 기울기,
drop 이벤트(잡고 있다가 놓친 전이 횟수 — plate는 낙하 종료가 없어 전이로 카운트).

.. code-block:: bash

    python scripts/tools/eval_dual_fr3_plate_robustness.py \
        --task Isaac-Lift-Box-DualFR3-Plate-Play-Noisy-v0 \
        --checkpoint logs/rsl_rl/dual_fr3_plate_lift/2026-07-28_15-47-21/model_4999.pt --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate dual_fr3 plate lift policy robustness.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Box-DualFR3-Plate-Play-Noisy-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=4, help="에피소드 사이클 수 (사이클당 5s x num_envs)")
parser.add_argument("--no_obs_noise", action="store_true", help="obs noise만 끄기 (물성 DR 유지)")
parser.add_argument(
    "--no_physics_dr", action="store_true", help="물성 DR만 끄기 (마찰/밀도/게인. 크기 DR은 유지, obs noise 유지)"
)
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
from isaaclab.utils.math import combine_frame_transforms, quat_apply
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from dual_fr3_lab.tasks.manager_based.lift_cube import mdp
from dual_fr3_lab.tasks.manager_based.lift_plate.joint_pos_env_cfg import LIFT_DELTA, _box_half_sizes
from dual_fr3_lab.tasks.manager_based.lift_plate.agents.rsl_rl_ppo_cfg import LiftCubePPORunnerCfg
from isaaclab_tasks.utils import parse_env_cfg

SUCCESS_D = 0.10  # goal 10cm 이내면 "정밀 유지"로 카운트 (P7 success 설계와 동일 기준)
UPRIGHT_TILT = 0.034  # 1 - cos(15도): 이보다 작으면 "수평"


def main():
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    # 축별 진단: noisy cfg에서 한 축만 끄고 평가 (원인 분리용)
    if args_cli.no_obs_noise:
        env_cfg.observations.policy.enable_corruption = False
    if args_cli.no_physics_dr:
        for name in ("randomize_box_friction", "randomize_box_density", "randomize_actuator_gains"):
            if hasattr(env_cfg.events, name):
                delattr(env_cfg.events, name)
    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env)

    agent_dict = LiftCubePPORunnerCfg().to_dict()
    # rsl_rl 버전 호환: isaaclab_rl(2.3+) cfg의 미지 키(optimizer 등)를 PPO 시그니처 기준 제거
    try:
        import inspect as _inspect

        from rsl_rl.algorithms import PPO as _PPO

        _allowed = set(_inspect.signature(_PPO.__init__).parameters) | {"class_name"}
        _alg = agent_dict.get("algorithm", {})
        _dropped = [k for k in list(_alg) if k not in _allowed]
        for _k in _dropped:
            _alg.pop(_k)
        if _dropped:
            print(f"[compat] rsl_rl PPO가 모르는 cfg 키 제거: {_dropped}")
    except Exception:
        pass
    runner = OnPolicyRunner(wrapped, agent_dict, log_dir=None, device="cuda:0")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")

    uenv = env.unwrapped
    f1 = SceneEntityCfg("left_plate_object_contact")
    f2 = SceneEntityCfg("right_plate_object_contact")

    steps_per_ep = int(uenv.max_episode_length)
    total_steps = steps_per_ep * args_cli.episodes
    obs = wrapped.get_observations()

    lifted_sum = grasp_sum = hold_sum = success_sum = upright_hold_sum = 0.0
    err_when_hold = []
    tilt_when_hold = []
    drops = 0
    prev_hold = torch.zeros(args_cli.num_envs, dtype=torch.bool, device="cuda:0")
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
        # env별 판정 높이 (안착 중심 + LIFT_DELTA)
        half = _box_half_sizes(uenv)
        lifted = (obj.data.root_pos_w[:, 2] > half + LIFT_DELTA).float()
        grasped = mdp.object_grasped(uenv, f1, f2, threshold=1.0)
        hold = lifted * grasped
        # 박스 기울기 (1 - z_box·z_world)
        z_axis = torch.zeros_like(obj.data.root_pos_w)
        z_axis[:, 2] = 1.0
        tilt = 1.0 - quat_apply(obj.data.root_quat_w, z_axis)[:, 2]

        lifted_sum += lifted.mean().item()
        grasp_sum += grasped.mean().item()
        hold_sum += hold.mean().item()
        success_sum += (hold * (err < SUCCESS_D).float()).mean().item()
        upright_hold_sum += (hold * (tilt < UPRIGHT_TILT).float()).mean().item()
        hold_b = hold.bool()
        if hold_b.any():
            err_when_hold.append(err[hold_b].mean().item())
            tilt_when_hold.append(tilt[hold_b].mean().item())
        # drop 이벤트: 직전 스텝 hold였다가 lifted가 꺼짐 (에피소드 리셋 경계 제외)
        at_reset = (uenv.episode_length_buf <= 1)
        drops += int((prev_hold & ~lifted.bool() & ~at_reset).sum().item())
        prev_hold = hold_b.clone()
        if step % steps_per_ep == 0:
            print(f"[EVAL] episode cycle {step // steps_per_ep + 1}/{args_cli.episodes}", flush=True)

    n = float(total_steps)
    print("\n===== PLATE ROBUSTNESS EVAL =====", flush=True)
    print(f"task            : {args_cli.task}")
    print(f"checkpoint      : {args_cli.checkpoint}")
    print(f"envs x episodes : {args_cli.num_envs} x {args_cli.episodes} ({total_steps} steps)")
    print(f"lifted fraction : {lifted_sum / n:.3f}")
    print(f"squeezed fraction: {grasp_sum / n:.3f}")
    print(f"hold (lift&squeeze): {hold_sum / n:.3f}")
    print(f"precise hold (<{SUCCESS_D * 100:.0f}cm): {success_sum / n:.3f}")
    print(f"upright hold (tilt<15deg): {upright_hold_sum / n:.3f}")
    print(f"goal err while holding: {sum(err_when_hold) / max(len(err_when_hold), 1):.3f} m")
    print(f"box tilt while holding: {sum(tilt_when_hold) / max(len(tilt_when_hold), 1):.4f} (1-cos)")
    print(f"drop events     : {drops} (over {args_cli.num_envs * args_cli.episodes} episodes)", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
