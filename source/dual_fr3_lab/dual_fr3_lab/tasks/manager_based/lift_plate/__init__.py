# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

##
# Joint Position Control (양팔 14 joint, plate EE)
##

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# sim2real 강건성 진단용: obs noise + 물성 DR을 씌운 play (학습 없이 정책 취약점 측정)
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Play-Noisy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_PLAY_NOISY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# ---------- v1: 실물 1차 검증용 특정 박스 (475x555x275mm, 4kg) - obs 56 (plate_forces 없음) ----------
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_V1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateV1PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_V1_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateV1PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Play-Noisy-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_V1_PLAY_NOISY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateV1PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# ---------- v2/P38 baseline: v1 + 외란·qprime·대칭·sustained yaw-aware success ----------
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_V2",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateV2PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Play-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_V2_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateV2PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Play-Noisy-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_V2_PLAY_NOISY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateV2PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# ---------- P45: P44 + one_sided_contact curriculum weight -5.0 → -7.0 ----------
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P45-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P45",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP45PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P45-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P45_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP45PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P45-Play-Noisy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P45_PLAY_NOISY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP45PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# ---------- P51: P49 + plate_face_z_offset 절대 z 오프셋 페널티 (curriculum) ----------
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P51-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P51",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP51PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P51-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P51_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP51PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P51-Play-Noisy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P51_PLAY_NOISY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP51PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# ---------- P52: P47 + box_anti_yaw 페널티 (G-gate+lifted gate, curriculum) ----------
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P52-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P52",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP52PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P52-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P52_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP52PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P52-Play-Noisy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P52_PLAY_NOISY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP52PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# ---------- P50: P49 + plate_face_alignment weight 2.0→1.0 ----------
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P50-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P50",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP50PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P50-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P50_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP50PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P50-Play-Noisy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P50_PLAY_NOISY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP50PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# ---------- P49: P47 + plate_face_alignment reward 활성화 (curriculum, G-gate+min+upright) ----------
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P49-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P49",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP49PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P49-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P49_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP49PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P49-Play-Noisy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P49_PLAY_NOISY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP49PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# ---------- P48: P44 + excessive_squeeze force_threshold 100N → 120N ----------
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P48-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P48",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP48PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P48-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P48_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP48PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P48-Play-Noisy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P48_PLAY_NOISY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP48PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# ---------- P47: P44 + plate_height_symmetry curriculum weight -6.0 → -8.0 ----------
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P47-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P47",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP47PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P47-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P47_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP47PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P47-Play-Noisy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P47_PLAY_NOISY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP47PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# ---------- P47-Friction: P47 + MuJoCo-matched 팔 관절 armature/friction/viscous ----------
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P47-Friction-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P47_Friction",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP47FrictionPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P47-Friction-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P47_Friction_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP47FrictionPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P47-Friction-Play-Noisy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P47_Friction_PLAY_NOISY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP47FrictionPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# ---------- P46: P44 + box_upright curriculum weight -3.0 → -4.0 ----------
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P46-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P46",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP46PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P46-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P46_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP46PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P46-Play-Noisy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P46_PLAY_NOISY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP46PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# ---------- P44: frozen P39 + yawtrack trigger 3.0 → 1.5 ----------
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P44-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P44",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP44PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P44-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P44_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP44PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P44-Play-Noisy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P44_PLAY_NOISY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP44PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# ---------- P43: frozen P39 + plates_approach_symmetry proximity 0.05 → 0.08 ----------
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P43-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P43",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP43PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P43-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P43_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP43PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P43-Play-Noisy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P43_PLAY_NOISY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP43PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# ---------- P42: frozen P39 + plates_misalignment -4 → -6 ----------
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P42-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P42",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP42PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P42-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P42_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP42PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P42-Play-Noisy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P42_PLAY_NOISY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP42PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# ---------- P41: frozen P39 + grip_axis_alignment (TCP geometry, j6_mirror_loss 대체) ----------
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P41-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P41",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP41PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P41-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P41_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP41PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P41-Play-Noisy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P41_PLAY_NOISY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP41PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# ---------- P40: frozen P39 + j6 wrist-pitch mirror loss ----------
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P40-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P40",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP40PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P40-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P40_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP40PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P40-Play-Noisy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P40_PLAY_NOISY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP40PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# ---------- P39: frozen P38/v2 + actual one-sided-contact XOR duration penalty ----------
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P39-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P39",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP39PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P39-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P39_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP39PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-P39-Play-Noisy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_P39_PLAY_NOISY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateP39PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# P9(v0 정책) zero-shot 평가용: 물리 v1 + obs v0(58, plate_forces 포함, noise on)
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-ZeroShot-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_V1_ZEROSHOT",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)


# ---------- Sim2Real: P47 + Reach Sim2Real v2 제어 계약 (1 kHz explicit PD/토크 변화율 제한, 20 Hz 정책,
# ±0.1 rad bounded action, policy-step target, 관절 순서 left1~7/right1~7, tanh actor) ----------
# actor가 tanh-squashed Gaussian이라 rsl_rl 진입점만 등록 (skrl/rl_games/sb3 cfg는 무제한 Gaussian용)
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-1kHz-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real1kHz",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real1kHzPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-1kHz-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real1kHz_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real1kHzPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-1kHz-Play-Noisy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real1kHz_PLAY_NOISY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real1kHzPPORunnerCfg",
    },
    disable_env_checker=True,
)


# 100Hz: 물리/PD 100 Hz 대안 (1 kHz가 너무 느릴 때) — 로그 dual_fr3_plate_lift_sim2real_100hz
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Play-Noisy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_PLAY_NOISY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzPPORunnerCfg",
    },
    disable_env_checker=True,
)


# 진단용 A/B: Sim2Real-100Hz에서 팔 action offset 상한만 ±0.1 -> ±0.3 rad — 로그 dual_fr3_plate_lift_sim2real_100hz_offset03
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Offset03-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Offset03",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzOffset03PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Offset03-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Offset03_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzOffset03PPORunnerCfg",
    },
    disable_env_checker=True,
)


# Offset03 + 수평 들기 gate + FR3 관절 한계 여유 — 로그 dual_fr3_plate_lift_sim2real_100hz_offset03_level
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Offset03-Level-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Offset03_Level",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzOffset03LevelPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Offset03-Level-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Offset03_Level_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzOffset03LevelPPORunnerCfg",
    },
    disable_env_checker=True,
)

# Offset03-Level + 판으로만 들기 + FR3 관절 한계 강제 — 로그 dual_fr3_plate_lift_sim2real_100hz_offset03_plateonly
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Offset03-PlateOnly-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Offset03_PlateOnly",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzOffset03PlateOnlyPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Offset03-PlateOnly-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Offset03_PlateOnly_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzOffset03PlateOnlyPPORunnerCfg",
    },
    disable_env_checker=True,
)

# PlateOnly + ±0.1 rad(reach/cube 배포 기준 그대로) + 1 kg 미만 실물 박스 — 로그 dual_fr3_plate_lift_sim2real_100hz_light
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPPORunnerCfg",
    },
    disable_env_checker=True,
)

# Light + 관절 한계 0.05 rad 안쪽 — Light model_5999에서 이어 학습하므로 로그도 같은 dual_fr3_plate_lift_sim2real_100hz_light
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Margin-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Margin",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Margin-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Margin_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPPORunnerCfg",
    },
    disable_env_checker=True,
)

# Light-Margin + 실물 2차 박스(410 x 310 x 280 mm, 0.4~0.6 kg), 바닥부터 전부 RL — Pregrasp 비교 기준선
# 로그 dual_fr3_plate_lift_sim2real_100hz_light_box410
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Box410-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightBox410PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Box410-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightBox410PPORunnerCfg",
    },
    disable_env_checker=True,
)

# Box410 + 페널티 커리큘럼 발동 기준 8 -> 6. 로그는 Box410과 같은 폴더(model_4999 이어 학습용)
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Box410-T6-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_T6",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightBox410PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Box410-T6-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_T6_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightBox410PPORunnerCfg",
    },
    disable_env_checker=True,
)

# Box410-T6 + j1 좌우 거울 페널티 (양팔이 같은 방향으로 ~90도 도는 해 방지), from-scratch
# 로그 dual_fr3_plate_lift_sim2real_100hz_light_box410_j1mirror
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Box410-J1Mirror-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_J1Mirror",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightBox410J1MirrorPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Box410-J1Mirror-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_J1Mirror_PLAY"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightBox410J1MirrorPPORunnerCfg",
    },
    disable_env_checker=True,
)

# Light-Margin을 접근 끝 자세(판이 박스 면 4 cm 바깥)에서 시작 — 접근은 rule-based/MoveIt, RL은 조이기~들기
# 로그 dual_fr3_plate_lift_sim2real_100hz_light_pregrasp[_upright]
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Upright-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Upright",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspUprightPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Upright-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Upright_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspUprightPPORunnerCfg",
    },
    disable_env_checker=True,
)

# Pregrasp + 판 모서리 정렬 시작 자세 표(--roll_align) — 로그 dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_rollalign
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-RollAlign-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_RollAlign",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspRollAlignPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-RollAlign-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_RollAlign_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspRollAlignPPORunnerCfg",
    },
    disable_env_checker=True,
)

# Pregrasp + 박스 90도 회전(긴 변 410 mm가 좌우, 실물 배치) — 로그 dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWidePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWidePPORunnerCfg",
    },
    disable_env_checker=True,
)

# Pregrasp-Wide + 판 모서리 정렬 표(--snap_j7) — 로그 dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWideRollAlignPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWideRollAlignPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FrozenBox-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FrozenBox",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWideRollAlignFrozenBoxPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FrozenBox-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FrozenBox_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWideRollAlignFrozenBoxPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-BoxDrop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_BoxDrop",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWideRollAlignBoxDropPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-BoxDrop-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_BoxDrop_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWideRollAlignBoxDropPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-Disturb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_Disturb",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWideRollAlignDisturbPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-Disturb-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_Disturb_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWideRollAlignDisturbPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-Regrasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_Regrasp",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWideRollAlignRegraspPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-Regrasp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_Regrasp_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWideRollAlignRegraspPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-DR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_DR",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWideRollAlignDRPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-DR-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_DR_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWideRollAlignDRPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-MinSqueeze-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_MinSqueeze",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripMinSqueezePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-MinSqueeze-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_MinSqueeze_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripMinSqueezePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-MinSqueeze-Scratch-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_MinSqueeze_Scratch",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripMinSqueezeScratchPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-MinSqueeze-Scratch-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_MinSqueeze_Scratch_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripMinSqueezeScratchPPORunnerCfg",
    },
    disable_env_checker=True,
)

# Box410-J1Mirror + 박스 90도 회전(긴 변 410 mm 좌우, 실물 배치), from-scratch
# 로그 dual_fr3_plate_lift_sim2real_100hz_light_box410_wide_j1mirror
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Box410-Wide-J1Mirror-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightBox410WideJ1MirrorPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Box410-Wide-J1Mirror-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Box410_Wide_J1Mirror_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightBox410WideJ1MirrorPPORunnerCfg",
    },
    disable_env_checker=True,
)

# FaceGrip + 박스 관측 지연·끊김(BoxDrop), 고친 물리에서 처음부터
# 로그 dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_facegrip_boxdrop_scratch
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-BoxDrop-Scratch-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_BoxDrop_Scratch",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripBoxDropScratchPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-BoxDrop-Scratch-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_BoxDrop_Scratch_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripBoxDropScratchPPORunnerCfg",
    },
    disable_env_checker=True,
)

# FaceGrip-MinSqueeze + 조인 다음 들기(들기 보상 x 조임 factor) + 접근 판 기울기·판-박스 미끄럼 벌점, MinSqueeze 5698에서 이어 학습
# 로그 dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_facegrip_squeezelift
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-SqueezeLift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_SqueezeLift",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripSqueezeLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-SqueezeLift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_SqueezeLift_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripSqueezeLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

# SqueezeLift + 박스·판 마찰 0.25~0.5, 박스 0.4~0.8 kg, 조임 20 N, action scale/clip 0.15, from-scratch
# 로그 dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_facegrip_squeezelift_lowmuheavy_act015_scratch
gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-SqueezeLift-LowMuHeavy-Act015-Scratch-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_SqueezeLift_LowMuHeavy_Act015_Scratch",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripSqueezeLiftLowMuHeavyAct015ScratchPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-SqueezeLift-LowMuHeavy-Act015-Scratch-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3PlateBoxLiftEnvCfg_Sim2Real100Hz_Light_Pregrasp_Wide_RollAlign_FaceGrip_SqueezeLift_LowMuHeavy_Act015_Scratch_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripSqueezeLiftLowMuHeavyAct015ScratchPPORunnerCfg",
    },
    disable_env_checker=True,
)
