"""Dual FR3 (UMI): open the cabinet with the left hand already on the handle, insert the book held in the right hand."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Cabinet-BookInsert-DualFR3-UMI-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cabinet_book_insert_env_cfg:DualFR3CabinetBookInsertEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetBookInsertPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Cabinet-BookInsert-DualFR3-UMI-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cabinet_book_insert_env_cfg:DualFR3CabinetBookInsertEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetBookInsertPPORunnerCfg",
    },
    disable_env_checker=True,
)
