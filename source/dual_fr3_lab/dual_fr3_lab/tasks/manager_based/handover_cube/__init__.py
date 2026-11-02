# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

from . import agents

##
# 큐브 핸드오버 (오른손 -> 왼손), Reach Sim2Real v2 제어 계약. rsl_rl만 등록
##

gym.register(
    id="Isaac-Handover-Cube-DualFR3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.handover_env_cfg:DualFR3HandoverCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HandoverCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Handover-Cube-DualFR3-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.handover_env_cfg:DualFR3HandoverCubeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HandoverCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Handover-Cube-DualFR3-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.handover_env_cfg:DualFR3HandoverCubeV1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HandoverCubeV1PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Handover-Cube-DualFR3-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.handover_env_cfg:DualFR3HandoverCubeV1EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HandoverCubeV1PPORunnerCfg",
    },
    disable_env_checker=True,
)
