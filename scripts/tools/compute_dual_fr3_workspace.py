# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""양팔 FR3의 도달 가능 영역(workspace)을 몬테카를로 FK로 추정하는 범용 도구.

관절 리밋 안에서 무작위 자세를 대량 샘플해(병렬 env x 스텝) 양팔 TCP 위치 구름을 얻고,
base frame 기준 도달 범위(percentile envelope)와 "양팔 pinch 가능한 박스 중심 영역"을 출력한다.

용도:
  ① 이상 상태(out-of-bounds) 종료 경계의 데이터 근거 (P11 크래시 대응)
  ② 새 task의 goal/스폰 범위 설계 (도달 불가 목표 샘플 방지)

.. code-block:: bash

    python scripts/tools/compute_dual_fr3_workspace.py --num_envs 2048 --steps 30 --half_y 0.2375
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Monte-Carlo workspace estimation for dual FR3.")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--steps", type=int, default=30, help="샘플 라운드 수 (총 표본 = num_envs x steps)")
parser.add_argument("--half_y", type=float, default=0.2375, help="박스 잡는 방향 반폭 [m] (pinch 중심 영역 계산용)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from dual_fr3_lab.robots.dual_fr3_plate_ee import DUAL_FR3_PLATE_EE_CFG, PLATE_TCP_OFFSET_POS


@configclass
class WsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = DUAL_FR3_PLATE_EE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def pct(t, q):
    return torch.quantile(t, q).item()


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda:0"))
    scene = InteractiveScene(WsSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0))
    sim.reset()
    robot = scene["robot"]

    lids, _ = robot.find_bodies(["left_fr3_link7"])
    rids, _ = robot.find_bodies(["right_fr3_link7"])
    limits = robot.data.soft_joint_pos_limits  # (N, J, 2)
    lo, hi = limits[..., 0], limits[..., 1]
    offset = torch.tensor(PLATE_TCP_OFFSET_POS, device=sim.device).repeat(args_cli.num_envs, 1)

    tcps_l, tcps_r = [], []
    for _ in range(args_cli.steps):
        q = lo + (hi - lo) * torch.rand_like(lo)
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        root = robot.data.root_pos_w
        for ids, acc in ((lids, tcps_l), (rids, tcps_r)):
            pose = robot.data.body_state_w[:, ids[0], 0:7]
            tcp = pose[:, 0:3] + quat_apply(pose[:, 3:7], offset)
            acc.append(tcp - root)  # base frame (root 무회전 가정)
    L = torch.cat(tcps_l)
    R = torch.cat(tcps_r)
    n = L.shape[0]

    print(f"\n===== DUAL FR3 WORKSPACE (base frame, {n} samples/arm) =====")
    for name, T in (("LEFT  TCP", L), ("RIGHT TCP", R)):
        print(f"\n[{name}] 1%~99% percentile envelope:")
        for i, ax in enumerate("xyz"):
            print(f"  {ax}: [{pct(T[:, i], 0.01):7.3f}, {pct(T[:, i], 0.99):7.3f}]"
                  f"   (min/max [{T[:, i].min():.3f}, {T[:, i].max():.3f}])")

    # 양팔 pinch 가능한 "박스 중심" 영역: 왼판이 +y면(중심 + half_y), 오른판이 -y면(중심 - half_y)
    h = args_cli.half_y
    cL = L - torch.tensor([0.0, h, 0.0], device=L.device)  # 왼팔이 커버 가능한 중심들
    cR = R + torch.tensor([0.0, h, 0.0], device=R.device)
    print(f"\n[PINCH 박스 중심 영역] (half_y={h}m, 각 팔 1%~99%):")
    inter = []
    for i, ax in enumerate("xyz"):
        loi = max(pct(cL[:, i], 0.01), pct(cR[:, i], 0.01))
        hii = min(pct(cL[:, i], 0.99), pct(cR[:, i], 0.99))
        inter.append((loi, hii))
        print(f"  {ax}: [{loi:7.3f}, {hii:7.3f}]  (양팔 교집합)")

    m = 0.10  # 종료 경계 여유
    print(f"\n[제안: out-of-bounds 종료 경계] (교집합 + 여유 {m}m, world z는 base+0.405 참고):")
    for (loi, hii), ax in zip(inter, "xyz"):
        print(f"  {ax}: [{loi - m:7.3f}, {hii + m:7.3f}]")
    print("  (z 하한은 바닥 위 물체이므로 -0.405(=world 0) 사용 권장)", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
