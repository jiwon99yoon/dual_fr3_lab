# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""오른팔 큐브 pick-and-place (Franka hand + UMI 팁, 75 cm 책상, ArUco 큐브 pose). 설계: pick_place_env_cfg.py"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-PickPlace-Cube-DualFR3-UMI-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_env_cfg:DualFR3PickPlaceCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-Cube-DualFR3-UMI-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_env_cfg:DualFR3PickPlaceCubeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

# 실험 A, B (11-09, 4090): 기준 run(5070, Isaac-PickPlace-Cube-DualFR3-UMI-v0)과 한 가지씩 다르다. pick_place_env_cfg.py 참고
gym.register(
    id="Isaac-PickPlace-Cube-DualFR3-UMI-GripperBusy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_env_cfg:DualFR3PickPlaceCubeGripperBusyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceCubeGripperBusyPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-Cube-DualFR3-UMI-GripperBusy-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_env_cfg:DualFR3PickPlaceCubeGripperBusyEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceCubeGripperBusyPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-Cube-DualFR3-UMI-CriticPlus-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_env_cfg:DualFR3PickPlaceCubeCriticPlusEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceCubeCriticPlusPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-Cube-DualFR3-UMI-CriticPlus-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_env_cfg:DualFR3PickPlaceCubeCriticPlusEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceCubeCriticPlusPPORunnerCfg",
    },
    disable_env_checker=True,
)
