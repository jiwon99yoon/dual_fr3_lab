#!/usr/bin/env python
"""Test A (Isaac 파트) — sim articulation joint 순서 & obs 텐서 실측.

reach_obs_contract_check.py 가 검증하지 못하는 단 하나:
Isaac articulation이 실제로 뱉는 joint 순서가 real 배포가 가정하는
[left_j1..7, right_j1..7] 와 같은가? 여기서 어긋나면 정책 입력이 permute
되어 58D 계약이 통과해도 실물에서 전혀 다르게 움직인다.

이 스크립트는 Isaac Sim을 띄워 아래를 출력한다.
  - robot.joint_names 전체 순서
  - obs term(joint_pos_rel)이 쓰는 SceneEntityCfg(".*_fr3_joint[1-7]") 해석 순서
  - reset 직후 policy obs의 앞 14D(joint_pos_rel)와 기대값 비교

실행 (env_isaaclab):
  cd ~/dual_fr3_lab
  ~/anaconda3/envs/env_isaaclab/bin/python scripts/sim2real/reach_joint_order_check.py --headless
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Reach joint-order / obs layout check.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 이하 Isaac 런타임 필요 ──────────────────────────────────────────────
import gymnasium as gym
import numpy as np
import torch

import dual_fr3_lab.tasks  # noqa: F401  task 등록
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import parse_env_cfg

TASK = "Isaac-Reach-DualFR3-Play-v0"

# real 배포가 가정하는 순서/기본자세 (fr3_husky_etri observation.py)
REAL_JOINT_ORDER = [f"{side}_fr3_joint{i}" for side in ("left", "right") for i in range(1, 8)]
REAL_DEFAULT = np.array(
    [-0.5236, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854,
     +0.5236, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854]
)


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    env = gym.make(TASK, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]

    print("\n" + "=" * 68)
    print("1. robot.joint_names (articulation 전체 순서)")
    print("=" * 68)
    for i, n in enumerate(robot.joint_names):
        print(f"  [{i:2d}] {n}")

    print("\n" + "=" * 68)
    print("2. obs term SceneEntityCfg('.*_fr3_joint[1-7]') 해석 순서")
    print("=" * 68)
    cfg = SceneEntityCfg("robot", joint_names=[".*_fr3_joint[1-7]"])
    cfg.resolve(env.unwrapped.scene)
    sim_order = [robot.joint_names[i] for i in cfg.joint_ids]
    for k, (jid, name) in enumerate(zip(cfg.joint_ids, sim_order)):
        expect = REAL_JOINT_ORDER[k]
        mark = "OK " if name == expect else "!! "
        print(f"  [{k:2d}] sim_id={jid:2d} {name:22s} expect={expect:22s} {mark}")
    order_ok = sim_order == REAL_JOINT_ORDER
    print(f"\n  [{'PASS' if order_ok else 'FAIL'}] sim joint 순서 == real 가정(left j1..7 → right j1..7)")
    if not order_ok:
        print("  ⚠️  순서 불일치 → real observation.py 의 joint 인덱싱을 sim 순서로 재정렬 필요")

    print("\n" + "=" * 68)
    print("3. reset 직후 policy obs 앞 14D(joint_pos_rel) vs 기대(q_reset - default)")
    print("=" * 68)
    obs, _ = env.reset()
    policy_obs = obs["policy"][0].detach().cpu().numpy()
    q = robot.data.joint_pos[0, cfg.joint_ids].detach().cpu().numpy()
    default = robot.data.default_joint_pos[0, cfg.joint_ids].detach().cpu().numpy()
    print(f"  default_joint_pos (sim, obs 순서): {np.round(default, 4).tolist()}")
    print(f"  real DEFAULT              : {REAL_DEFAULT.tolist()}")
    default_ok = np.allclose(default, REAL_DEFAULT, atol=1e-3)
    print(f"    [{'PASS' if default_ok else 'FAIL'}] sim default == real DEFAULT")
    diff = np.max(np.abs(policy_obs[:14] - (q - default)))
    print(f"  obs[:14] == (q - default)? max|Δ|={diff:.3e}  "
          f"[{'PASS' if diff < 1e-4 else 'FAIL'}]")
    print(f"  obs shape = {policy_obs.shape} (기대 58)")

    env.close()
    print("\n완료. FAIL 있으면 real 배포 코드/문서 갱신 필요.")


if __name__ == "__main__":
    main()
    simulation_app.close()
