# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""obs 벡터의 실제 구성(관절 순서·항목별 인덱스)을 실측 출력 — 실물 배포 배선 검증용.

v1.md §3 obs 표의 "가정된 순서"를 IsaacLab이 실제로 쓰는 순서와 대조한다.
joint 순서가 가정(joint1..7 → finger)과 다르면 실물에서 obs를 잘못 조립해 정책이 조용히 오작동.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/dump_obs_layout.py --task Isaac-Lift-Cube-DualFR3-v1 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump observation layout of a dual_fr3 lift env.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-DualFR3-v1")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import dual_fr3_lab.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    _log = open("/tmp/obslayout_result.txt", "w")
    def _w(*a):
        _log.write(" ".join(str(x) for x in a) + "\n"); _log.flush()
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    env.reset()  # joint_names/데이터 버퍼 채우기
    robot = uenv.scene["robot"]

    _w("\n===== OBS LAYOUT 실측 =====")
    _w(f"task: {args_cli.task}")

    # 1) 로봇의 실제 관절 순서 (IsaacLab 내부 정렬)
    _w(f"\n[robot.data.joint_names] (전체 {len(robot.data.joint_names)}개, IsaacLab 내부 순서):")
    for i, nm in enumerate(robot.data.joint_names):
        _w(f"  {i:2d}: {nm}")

    # 2) joint_pos/joint_vel obs term이 실제로 쓰는 joint 부분집합 + 순서
    obs_mgr = uenv.observation_manager
    _w("\n[policy obs terms — 순서와 dim]:")
    group = "policy"
    dims = obs_mgr.group_obs_term_dim[group]
    names = obs_mgr.active_terms[group]
    idx = 0
    for nm, d in zip(names, dims):
        dim = d[0] if isinstance(d, (tuple, list)) else d
        _w(f"  obs[{idx:2d}:{idx+dim:2d}]  {nm}  (dim {dim})")
        idx += dim
    _w(f"  총 obs dim = {idx}")

    # 3) joint_pos term의 asset_cfg에 resolve된 joint_ids -> 실제 이름 순서
    for term in ("joint_pos", "joint_vel"):
        cfg = getattr(uenv.cfg.observations.policy, term, None)
        if cfg is not None and "asset_cfg" in cfg.params:
            ac = cfg.params["asset_cfg"]
            ids = ac.joint_ids
            if ids == slice(None):
                resolved = list(robot.data.joint_names)
            else:
                resolved = [robot.data.joint_names[j] for j in ids]
            _w(f"\n[{term}] obs에 들어가는 joint 순서 ({len(resolved)}개):")
            for k, nm in enumerate(resolved):
                _w(f"  obs 내부 #{k}: {nm}")

    simulation_app.close()


if __name__ == "__main__":
    main()
