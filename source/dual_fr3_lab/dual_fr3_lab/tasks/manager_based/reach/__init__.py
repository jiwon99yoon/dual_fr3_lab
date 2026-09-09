# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

# ---------- Dual FR3 Reach (sim2real gap 분석용) ----------
gym.register(
    id="Isaac-Reach-DualFR3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3ReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DualFR3ReachPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Reach-DualFR3-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3ReachEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DualFR3ReachPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Reach-DualFR3-Sim2Real-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3ReachSim2RealEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DualFR3ReachSim2RealPPORunnerCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Reach-DualFR3-Sim2Real-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3ReachSim2RealEnvCfgPlay",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DualFR3ReachSim2RealPPORunnerCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Reach-DualFR3-Sim2Real-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3ReachSim2RealV2EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DualFR3ReachSim2RealV2PPORunnerCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Reach-DualFR3-Sim2Real-Play-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualFR3ReachSim2RealV2EnvCfgPlay",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DualFR3ReachSim2RealV2PPORunnerCfg"
        ),
    },
    disable_env_checker=True,
)
