# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""dual_fr3 reach 정책의 sim-side trajectory 수집 (sim2real gap 분석 전초).

reach task는 파지 성공률이 아니라 "sim 정책을 실물에 올렸을 때 tracking 오차·제어
지연이 얼마나 나는가"를 보는 task라, play.py(렌더링 전용)로는 비교 데이터를 못 뽑는다.
이 스크립트는 고정 seed로 N 에피소드를 굴리며 control-step마다
  joint_pos(14) / joint_vel(14) / action(14) / left·right EE pos(3+3) /
  left·right target pos(3+3) / left·right EE error(스칼라)
를 수집해 ``.npz``로 저장한다. 실물 ROS2 노드에서 같은 포맷 CSV를 수집하면
``compare_reach_sim_real.py``로 겹쳐 그린다.

obs noise on/off를 --no_obs_noise로 분리해 sim 자체 분산(정책 결정론성)도 잴 수 있다.
요약 통계(EE error mean/p50/p95, rise time, steady-state error)는 stdout에 출력한다.

.. code-block:: bash

    python scripts/tools/eval_dual_fr3_reach_traj.py \
        --checkpoint logs/rsl_rl/dual_fr3_reach/<run>/model_2999.pt \
        --num_envs 16 --episodes 1 --seed 42 --headless

    # obs noise 끄고 (sim 결정론적 궤적만) 수집
    python scripts/tools/eval_dual_fr3_reach_traj.py \
        --checkpoint <ckpt> --no_obs_noise --headless
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect dual_fr3 reach sim trajectories for sim2real comparison.")
parser.add_argument("--task", type=str, default="Isaac-Reach-DualFR3-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16, help="재현성 위해 소수 권장(startup DR 독립 표본)")
parser.add_argument("--episodes", type=int, default=1, help="env당 수집할 에피소드 사이클 수")
parser.add_argument("--seed", type=int, default=42, help="평가 env/DR seed. 음수면 cfg 기본값")
parser.add_argument("--no_obs_noise", action="store_true", help="obs noise만 끄기 (actuator DR은 유지)")
parser.add_argument(
    "--out",
    type=str,
    default=None,
    help="저장 .npz 경로. 미지정 시 <checkpoint_dir>/trajectories/reach_traj.npz",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import dual_fr3_lab.tasks  # noqa: F401  # gym id 등록
from isaaclab.utils.math import combine_frame_transforms
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

from dual_fr3_lab.tasks.manager_based.reach.joint_pos_env_cfg import REACH_OFFSET_Y
from dual_fr3_lab.tasks.manager_based.reach.agents.rsl_rl_ppo_cfg import DualFR3ReachPPORunnerCfg

RISE_EPS = 0.03  # EE error < 3cm 도달 시각 = rise time
STEADY_TAIL_S = 2.0  # 에피소드 마지막 2초 평균 = steady-state error


def _percentile(t: torch.Tensor, q: float) -> float:
    return float(torch.quantile(t.flatten(), q / 100.0)) if t.numel() else float("nan")


def main():
    eval_device = args_cli.device
    env_cfg = parse_env_cfg(args_cli.task, device=eval_device, num_envs=args_cli.num_envs)
    if args_cli.seed >= 0:
        env_cfg.seed = args_cli.seed
    if args_cli.no_obs_noise:
        env_cfg.observations.policy.enable_corruption = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env)

    agent_dict = DualFR3ReachPPORunnerCfg().to_dict()
    # rsl_rl 버전 호환: isaaclab_rl(2.3+) cfg의 미지 키를 PPO 시그니처 기준 제거
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
    runner = OnPolicyRunner(wrapped, agent_dict, log_dir=None, device=eval_device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=eval_device)

    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    ee_frame = uenv.scene["ee_frame"]
    names = ee_frame.data.target_frame_names
    li = names.index("left_ee")
    ri = names.index("right_ee")

    # 정책 action 컬럼 순서 = action term의 resolved joint 순서. L1..L7,R1..R7 가정 금지.
    joint_names = robot.data.joint_names
    left_joint_ids = sorted(
        [i for i, n in enumerate(joint_names) if n.startswith("left_fr3_joint")],
        key=lambda i: int(joint_names[i].rsplit("joint", 1)[1]),
    )
    right_joint_ids = sorted(
        [i for i, n in enumerate(joint_names) if n.startswith("right_fr3_joint")],
        key=lambda i: int(joint_names[i].rsplit("joint", 1)[1]),
    )
    if len(left_joint_ids) != 7 or len(right_joint_ids) != 7:
        raise RuntimeError(f"Expected 7 joints per arm, got L={left_joint_ids}, R={right_joint_ids}")
    arm_joint_ids = left_joint_ids + right_joint_ids

    num_envs = args_cli.num_envs
    steps_per_ep = int(uenv.max_episode_length)
    total_steps = steps_per_ep * args_cli.episodes
    dt = float(uenv.step_dt)

    # per-step 버퍼 (T, N, dim)
    buf_q, buf_qd, buf_act = [], [], []
    buf_ee_l, buf_ee_r = [], []
    buf_tgt_l, buf_tgt_r = [], []
    buf_err_l, buf_err_r = [], []
    buf_step = []  # 에피소드 내 step index (경계 분리용)

    obs = wrapped.get_observations()
    with torch.inference_mode():
        for _ in range(total_steps):
            actions = policy(obs)

            # target: goal center in world ± offset_y (reward 정의와 동일)
            cmd = uenv.command_manager.get_command("target_pose")[:, :3]
            center_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, cmd)
            off = torch.zeros_like(center_w)
            off[:, 1] = REACH_OFFSET_Y
            tgt_l = center_w + off
            tgt_r = center_w - off

            ee_l = ee_frame.data.target_pos_w[:, li]
            ee_r = ee_frame.data.target_pos_w[:, ri]
            err_l = torch.norm(ee_l - tgt_l, dim=-1)
            err_r = torch.norm(ee_r - tgt_r, dim=-1)

            # env_origins를 빼서 per-env local frame으로 정렬 (env 간 비교 가능)
            origins = uenv.scene.env_origins
            buf_q.append(robot.data.joint_pos[:, arm_joint_ids].cpu().numpy())
            buf_qd.append(robot.data.joint_vel[:, arm_joint_ids].cpu().numpy())
            buf_act.append(actions.cpu().numpy())
            buf_ee_l.append((ee_l - origins).cpu().numpy())
            buf_ee_r.append((ee_r - origins).cpu().numpy())
            buf_tgt_l.append((tgt_l - origins).cpu().numpy())
            buf_tgt_r.append((tgt_r - origins).cpu().numpy())
            buf_err_l.append(err_l.cpu().numpy())
            buf_err_r.append(err_r.cpu().numpy())
            buf_step.append(int(uenv.episode_length_buf[0].item()))

            obs, _, _, _ = wrapped.step(actions)

    # (T, N, ...) 스택
    q = np.stack(buf_q)          # (T, N, 14)
    qd = np.stack(buf_qd)
    act = np.stack(buf_act)      # (T, N, 14)
    ee_l = np.stack(buf_ee_l)    # (T, N, 3)
    ee_r = np.stack(buf_ee_r)
    tgt_l = np.stack(buf_tgt_l)
    tgt_r = np.stack(buf_tgt_r)
    err_l = np.stack(buf_err_l)  # (T, N)
    err_r = np.stack(buf_err_r)
    step_idx = np.array(buf_step)

    # 에피소드 경계: episode_length_buf가 감소(리셋)하는 지점으로 자른다. 단일 에피소드 수집
    # (episodes=1)이면 전체가 곧 한 에피소드. 여기선 env0 기준으로만 경계 판단해 npz에 저장하고,
    # 비교 스크립트는 각 env를 개별 에피소드로 다룬다.

    # ── 요약 통계 (obs noise 유무와 무관하게 tracking 품질을 본다) ──
    err_l_t = torch.from_numpy(err_l)
    err_r_t = torch.from_numpy(err_r)
    err_all = torch.cat([err_l_t, err_r_t], dim=1)

    tail_steps = max(1, int(STEADY_TAIL_S / dt))
    steady_l = err_l[-tail_steps:].mean()
    steady_r = err_r[-tail_steps:].mean()

    # rise time: env별 err<RISE_EPS 최초 도달 step → 시간. 못 도달 시 nan.
    def rise_time(err_tn):
        T = err_tn.shape[0]
        reached = err_tn < RISE_EPS
        times = []
        for n in range(err_tn.shape[1]):
            idx = np.argmax(reached[:, n]) if reached[:, n].any() else -1
            times.append(idx * dt if idx >= 0 else np.nan)
        return np.array(times)

    rt_l = rise_time(err_l)
    rt_r = rise_time(err_r)

    print("\n──────────── reach sim trajectory 요약 ────────────")
    print(f"task={args_cli.task}  num_envs={num_envs}  episodes={args_cli.episodes}  "
          f"obs_noise={'OFF' if args_cli.no_obs_noise else 'ON'}  dt={dt:.4f}s ({1/dt:.0f}Hz)")
    print(f"steps collected: {q.shape[0]} (per-ep {steps_per_ep})")
    print(f"\nEE error [m]      left            right")
    print(f"  mean          {err_l.mean():.4f}          {err_r.mean():.4f}")
    print(f"  p50           {_percentile(err_l_t, 50):.4f}          {_percentile(err_r_t, 50):.4f}")
    print(f"  p95           {_percentile(err_l_t, 95):.4f}          {_percentile(err_r_t, 95):.4f}")
    print(f"  steady(<{STEADY_TAIL_S:.0f}s tail) {steady_l:.4f}          {steady_r:.4f}")
    print(f"  rise(<{RISE_EPS*100:.0f}cm) mean  {np.nanmean(rt_l):.3f}s          {np.nanmean(rt_r):.3f}s")
    print(f"  reached ratio  {np.mean(~np.isnan(rt_l)):.2f}            {np.mean(~np.isnan(rt_r)):.2f}")

    # ── 저장 ──
    if args_cli.out is not None:
        out_path = Path(args_cli.out)
    else:
        out_path = Path(args_cli.checkpoint).parent / "trajectories" / "reach_traj.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        # 메타
        dt=dt,
        hz=1.0 / dt,
        num_envs=num_envs,
        steps_per_ep=steps_per_ep,
        obs_noise=not args_cli.no_obs_noise,
        joint_order=np.array([joint_names[i] for i in arm_joint_ids]),
        offset_y=REACH_OFFSET_Y,
        # per-step (T, N, dim)
        q=q, qd=qd, action=act,
        ee_left=ee_l, ee_right=ee_r,
        target_left=tgt_l, target_right=tgt_r,
        err_left=err_l, err_right=err_r,
        episode_step=step_idx,
    )
    print(f"\nsaved: {out_path.resolve()}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
