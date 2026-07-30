# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""그리퍼 이진 명령의 toggle(open<->close 전환) 빈도 측정 — sim2real 배포 진단.

정책이 매 스텝 뱉는 gripper action(action[-1])의 부호가 에피소드 동안 몇 번 뒤집히는지 센다.
실물 Franka 그리퍼는 grasp() 명령이 수백 ms 블로킹이라, 명령이 파닥거리면(채터링) 하드웨어가
못 따라간다. 이 측정으로 ① 채터링 유무 ② 배포 히스테리시스 N(연속 유지 step) 결정.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/eval_gripper_toggle.py \
        --task Isaac-Lift-Cube-DualFR3-Play-Noisy-v1 \
        --checkpoint logs/rsl_rl/dual_fr3_lift_v1/2026-07-29_12-38-27/model_4999.pt --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure gripper toggle frequency of a dual_fr3 lift policy.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-DualFR3-Play-Noisy-v1")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=4, help="에피소드 사이클 수 (사이클당 250 step)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import dual_fr3_lab.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from dual_fr3_lab.tasks.manager_based.lift_cube.agents.rsl_rl_ppo_cfg import LiftCubePPORunnerCfg
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env)

    agent_dict = LiftCubePPORunnerCfg().to_dict()
    try:
        import inspect as _inspect

        from rsl_rl.algorithms import PPO as _PPO

        _allowed = set(_inspect.signature(_PPO.__init__).parameters) | {"class_name"}
        _alg = agent_dict.get("algorithm", {})
        for _k in [k for k in list(_alg) if k not in _allowed]:
            _alg.pop(_k)
    except Exception:
        pass
    runner = OnPolicyRunner(wrapped, agent_dict, log_dir=None, device="cuda:0")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")

    uenv = env.unwrapped
    N = args_cli.num_envs
    dev = uenv.device
    steps_per_ep = int(uenv.max_episode_length)
    total_steps = steps_per_ep * args_cli.episodes

    obs = wrapped.get_observations()
    # 시퀀스 덤프: 전체 env의 그리퍼 명령(1=close, 0=open)을 첫 에피소드(steps_per_ep) 동안 기록
    # -> 첫 에피소드 종료 후 전환 2회 이상인 env를 골라 시퀀스 출력 (채터링 실체 확인)
    trace_envs = N
    seq_trace = [[] for _ in range(trace_envs)]
    prev_sign = None  # 그리퍼 명령 부호 (True=close(a<0), False=open)
    ep_toggles = torch.zeros(N, device=dev)  # 현재 에피소드 누적 전환 수
    ep_step = torch.zeros(N, device=dev)
    last_toggle_step = torch.zeros(N, device=dev)  # 마지막 전환이 있었던 ep 내 step
    # 집계 버킷
    completed_toggles = []  # 에피소드별 총 전환 수
    min_gap = 10**9  # 전체에서 관측된 최소 전환 간격 (step)
    gap_hist = {"<=5": 0, "6-10": 0, "11-25": 0, ">25": 0}

    def bucket(g):
        if g <= 5:
            gap_hist["<=5"] += 1
        elif g <= 10:
            gap_hist["6-10"] += 1
        elif g <= 25:
            gap_hist["11-25"] += 1
        else:
            gap_hist[">25"] += 1

    for step in range(total_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = wrapped.step(actions)
        sign = (actions[:, -1] < 0)  # True = close 명령
        if step < steps_per_ep:
            for e in range(trace_envs):
                seq_trace[e].append(1 if bool(sign[e]) else 0)
        ep_step += 1
        if prev_sign is not None:
            flipped = sign != prev_sign
            if flipped.any():
                idx = flipped.nonzero(as_tuple=True)[0]
                for i in idx.tolist():
                    gap = int((ep_step[i] - last_toggle_step[i]).item())
                    if last_toggle_step[i] > 0:  # 첫 전환은 간격 무의미
                        bucket(gap)
                        min_gap = min(min_gap, gap)
                    last_toggle_step[i] = ep_step[i].item()
                ep_toggles[idx] += 1
        prev_sign = sign
        # 에피소드 종료 처리
        done_mask = dones.to(dev).bool().squeeze(-1) if dones.dim() > 1 else dones.to(dev).bool()
        if done_mask.any():
            idx = done_mask.nonzero(as_tuple=True)[0]
            completed_toggles += ep_toggles[idx].tolist()
            ep_toggles[idx] = 0
            ep_step[idx] = 0
            last_toggle_step[idx] = 0

    n_ep = max(len(completed_toggles), 1)
    mean_tog = sum(completed_toggles) / n_ep
    print("\n===== GRIPPER TOGGLE 측정 =====", flush=True)
    print(f"task        : {args_cli.task}")
    print(f"checkpoint  : {args_cli.checkpoint}")
    print(f"episodes    : {len(completed_toggles)} (완료분), {N} envs x {args_cli.episodes} cycle")
    print(f"평균 전환/ep : {mean_tog:.2f}  (정상 파지 기대 1~2회, 채터링이면 수십회)")
    if completed_toggles:
        srt = sorted(completed_toggles)
        print(f"전환 분포   : min {srt[0]:.0f} / median {srt[len(srt)//2]:.0f} / max {srt[-1]:.0f}")
    print(f"최소 전환간격: {min_gap if min_gap < 10**9 else 'n/a'} step "
          f"(50Hz 기준 {min_gap*20 if min_gap<10**9 else 0}ms — 실물 그리퍼 ~수백ms면 이보다 커야 안전)")
    print(f"간격 히스토그램(step): {gap_hist}")
    tot = sum(gap_hist.values())
    if tot:
        fast = gap_hist["<=5"] + gap_hist["6-10"]
        print(f"채터링 위험(<=10step=200ms 내 재전환): {fast}/{tot} ({100*fast/tot:.0f}%)")
    print("판정 가이드: 평균~1-2회 & 위험% ~0 -> 히스테리시스 불필요 / 위험% 높음 -> N을 위험 간격 위로", flush=True)

    # --- 실제 시퀀스 덤프: 첫 에피소드에서 전환 2회 이상인 env만 (채터링 실체 확인) ---
    print("\n===== 첫 에피소드 그리퍼 전환 분포 (전 env) =====", flush=True)
    flip_counts = []
    for e in range(trace_envs):
        seq = seq_trace[e]
        flips = [i for i in range(1, len(seq)) if seq[i] != seq[i - 1]]
        flip_counts.append(len(flips))
    from collections import Counter

    print(f"전환 횟수 분포: {dict(sorted(Counter(flip_counts).items()))}  (env 수 기준)", flush=True)
    print("\n===== 전환 2회 이상 env 시퀀스 (C=close/.=open) =====", flush=True)
    shown = 0
    for e in range(trace_envs):
        seq = seq_trace[e]
        flips = [i for i in range(1, len(seq)) if seq[i] != seq[i - 1]]
        if len(flips) >= 2 and shown < 6:
            gaps = [flips[i] - flips[i - 1] for i in range(1, len(flips))]
            print(f"env{e}: {''.join('C' if v else '.' for v in seq)}")
            print(f"      전환 step: {flips}  간격: {gaps}", flush=True)
            shown += 1
    if shown == 0:
        print("(없음 — 모든 env가 전환 0~1회, 계단 1개 패턴 = 채터링 없음)", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
