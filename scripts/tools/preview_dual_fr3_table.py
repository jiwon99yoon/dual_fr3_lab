# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
dual_fr3 lift task의 테이블/물체/DR존 배치 미리보기 + 수치 검증.

두 가지 배치를 전환해서 볼 수 있다:
- 기본: 현재 RL 환경(joint_pos_env_cfg.py)과 동일한 통짜 테이블 (상판 z=0.6)
- --mjcf: 선배 MuJoCo 세팅(lift/config/dual_fr3_table.md)의 얇은 상판 + 다리 테이블 (상판 z=0.75)
  단, x 위치는 MJCF의 0.8이 아니라 우리 로봇 기준 0.6 유지 (0.8이면 cube 스폰존이 테이블 밖)

시각화: 파란 판 = cube 스폰 DR존, 초록 판 2장 = goal 샘플 밴드 상/하한.
시작 시 배치 마진(husky-테이블, 스폰존-테이블 모서리, goal 밴드-상판)을 수치로 출력한다.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/05_controllers/preview_dual_fr3_table.py          # 현재 배치
    ./isaaclab.sh -p scripts/tutorials/05_controllers/preview_dual_fr3_table.py --mjcf   # MJCF 테이블

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Preview table placement for the dual FR3 lift task.")
parser.add_argument("--mjcf", action="store_true", help="MJCF(MuJoCo) 테이블 배치로 미리보기")
parser.add_argument("--table_x", type=float, default=0.6, help="테이블 중심 x (world)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from dual_fr3_lab.robots import DUAL_FR3_CFG  # isort:skip

# ---------- 공통 상수 (RL env와 동일해야 함: joint_pos_env_cfg.py 참조) ----------
ROOT_Z = 0.405  # 로봇 base frame world 높이
HUSKY_FRONT_X = 0.15  # husky 앞머리 대략치
CUBE_HALF_MAX = 0.055 / 2  # DR 최대 큐브(5.5cm) 반변
SPAWN_XY = (0.45, -0.25)  # cube 스폰 중심 (world x, y)
SPAWN_RANGE = (0.08, 0.12)  # 스폰 DR ±x, ±y
GOAL_RANGE_B_XY = ((0.35, 0.55), (-0.45, -0.15))  # goal 샘플 범위 (base 기준 x, y)

if args_cli.mjcf:
    # MJCF (dual_fr3_table.md): size는 half-extent -> 실치수 2배. 상판 z=0.75
    TABLE_TOP_Z = 0.75
    TABLE_TOP_SIZE = (0.6, 1.0, 0.02)
    GOAL_RANGE_B_Z = (0.45, 0.60)  # 상판이 높아진 만큼 goal 밴드도 상향 (제안값)
else:
    # 현재 RL 환경: 통짜 박스
    TABLE_TOP_Z = 0.6
    TABLE_TOP_SIZE = (0.8, 1.0, 0.6)
    GOAL_RANGE_B_Z = (0.35, 0.55)

TABLE_X = args_cli.table_x
CUBE_POS = (SPAWN_XY[0], SPAWN_XY[1], TABLE_TOP_Z + 0.03)


def _static_box(path: str, size, pos, color, opacity: float = 1.0, collide: bool = True):
    cfg = sim_utils.MeshCuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg() if collide else None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, opacity=opacity),
    )
    cfg.func(path, cfg, translation=pos)


def design_scene() -> dict:
    """Designs the scene."""
    # Ground-plane (z=0)
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # ---------- Table ----------
    if args_cli.mjcf:
        # 얇은 상판 (윗면 z=0.75)
        _static_box("/World/Table/top", TABLE_TOP_SIZE, (TABLE_X, 0.0, TABLE_TOP_Z - 0.01), (0.8, 0.6, 0.4))
        # 수직 다리 (z 0.02~0.73) + 바닥 러너 (z 0~0.02), y=±0.49
        for side, y in (("left", 0.49), ("right", -0.49)):
            _static_box(f"/World/Table/{side}_leg", (0.02, 0.02, 0.71), (TABLE_X, y, 0.375), (0.9, 0.9, 0.9))
            _static_box(f"/World/Table/{side}_runner", (0.6, 0.02, 0.02), (TABLE_X, y, 0.01), (0.9, 0.9, 0.9))
    else:
        _static_box("/World/Table", TABLE_TOP_SIZE, (TABLE_X, 0.0, TABLE_TOP_Z / 2), (0.55, 0.4, 0.25))

    # ---------- DR 존 시각화 (RL env와 동일 규칙, 충돌체 없음) ----------
    # 파란 판: cube 스폰존 (중심 기준 범위 + 최대 큐브 반변)
    _static_box(
        "/World/Zones/spawn",
        (2 * SPAWN_RANGE[0] + 2 * CUBE_HALF_MAX, 2 * SPAWN_RANGE[1] + 2 * CUBE_HALF_MAX, 0.002),
        (SPAWN_XY[0], SPAWN_XY[1], TABLE_TOP_Z + 0.001),
        (0.1, 0.3, 0.9),
        collide=False,
    )
    # 초록 판 2장: goal 밴드 하한/상한 (base -> world: +ROOT_Z)
    (gx0, gx1), (gy0, gy1) = GOAL_RANGE_B_XY
    gcx, gcy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
    gsz = (gx1 - gx0 + 2 * CUBE_HALF_MAX, gy1 - gy0 + 2 * CUBE_HALF_MAX, 0.003)
    for tag, gz in (("bottom", GOAL_RANGE_B_Z[0]), ("top", GOAL_RANGE_B_Z[1])):
        _static_box(f"/World/Zones/goal_{tag}", gsz, (gcx, gcy, gz + ROOT_Z), (0.1, 0.8, 0.3), collide=False)

    # ---------- Object (DexCube, RL env와 동일 asset) ----------
    object_cfg = RigidObjectCfg(
        prim_path="/World/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_POS, rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
        ),
    )
    obj = RigidObject(cfg=object_cfg)

    # ---------- Robot ----------
    robot_cfg = DUAL_FR3_CFG.copy()
    robot_cfg.prim_path = "/World/Robot"
    # 실험: 처짐이 stiffness 문제인지 확인 — 80 -> 300 (preview 전용, RL env cfg에는 영향 없음)
    robot_cfg.actuators["arms_shoulder"].stiffness = 300.0
    robot_cfg.actuators["arms_shoulder"].damping = 15.0
    robot_cfg.actuators["arms_forearm"].stiffness = 300.0
    robot_cfg.actuators["arms_forearm"].damping = 15.0
    robot = Articulation(cfg=robot_cfg)

    return {"robot": robot, "object": obj}


def print_layout_checks():
    """배치 마진을 수치로 출력한다. 음수 마진 = 문제."""
    front_edge = TABLE_X - TABLE_TOP_SIZE[0] / 2
    back_edge = TABLE_X + TABLE_TOP_SIZE[0] / 2
    spawn_x = (SPAWN_XY[0] - SPAWN_RANGE[0] - CUBE_HALF_MAX, SPAWN_XY[0] + SPAWN_RANGE[0] + CUBE_HALF_MAX)
    spawn_y = (SPAWN_XY[1] - SPAWN_RANGE[1] - CUBE_HALF_MAX, SPAWN_XY[1] + SPAWN_RANGE[1] + CUBE_HALF_MAX)
    goal_z_w = (GOAL_RANGE_B_Z[0] + ROOT_Z, GOAL_RANGE_B_Z[1] + ROOT_Z)

    mode = "MJCF (MuJoCo)" if args_cli.mjcf else "현재 RL env"
    print("\n========== LAYOUT CHECK ==========")
    print(f"mode                : {mode}")
    print(f"table top           : z={TABLE_TOP_Z:.3f}, x [{front_edge:.3f}, {back_edge:.3f}], y [±{TABLE_TOP_SIZE[1]/2:.2f}]")
    print(f"husky-table margin  : {front_edge - HUSKY_FRONT_X:+.3f} m  (앞면 {front_edge:.2f} - husky {HUSKY_FRONT_X:.2f})")
    print(f"cube spawn zone     : x [{spawn_x[0]:.3f}, {spawn_x[1]:.3f}], y [{spawn_y[0]:.3f}, {spawn_y[1]:.3f}], z={CUBE_POS[2]:.3f}")
    print(f"  ㄴ table 안쪽 마진 : front {spawn_x[0] - front_edge:+.3f} / back {back_edge - spawn_x[1]:+.3f}"
          f" / side {spawn_y[0] - (-TABLE_TOP_SIZE[1]/2):+.3f} m  (음수면 낙하!)")
    print(f"goal band (world z) : [{goal_z_w[0]:.3f}, {goal_z_w[1]:.3f}]  (base 기준 [{GOAL_RANGE_B_Z[0]:.2f}, {GOAL_RANGE_B_Z[1]:.2f}])")
    print(f"  ㄴ 상판과의 간격    : {goal_z_w[0] - TABLE_TOP_Z:+.3f} m  (하한 - 상판, 0.05 이상 권장)")
    if args_cli.mjcf:
        print(f"legs (y=±0.49)      : cube 존 y와 간격 {abs(-0.49 - spawn_y[0]):.3f} m")
    print("==================================\n")


def run_simulator(sim: sim_utils.SimulationContext, entities: dict):
    """ready pose를 유지하며 배치만 보여준다."""
    robot = entities["robot"]
    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        # ready pose 유지
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.6, -1.8, 1.6], [0.5, -0.1, 0.6])
    scene_entities = design_scene()
    sim.reset()
    print("[INFO]: Setup complete...")
    print_layout_checks()
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
