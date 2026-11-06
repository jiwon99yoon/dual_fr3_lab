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

gym.register(
    id="Isaac-Cabinet-BookInsert-DualFR3-UMI-OpenDoor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cabinet_book_insert_env_cfg:DualFR3CabinetBookInsertOpenDoorEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetBookInsertOpenDoorPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Cabinet-BookInsert-DualFR3-UMI-OpenDoor-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cabinet_book_insert_env_cfg:DualFR3CabinetBookInsertOpenDoorEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetBookInsertOpenDoorPPORunnerCfg",
    },
    disable_env_checker=True,
)

# v3 (plan 7-10): 0.5 rad/s joint steps, strong handle-force penalty; open-door stage one, then this closed-door task
for _suffix, _cfg in (("", "SlowEnvCfg"), ("-OpenDoor", "SlowOpenDoorEnvCfg")):
    for _play in ("", "-Play"):
        gym.register(
            id=f"Isaac-Cabinet-BookInsert-DualFR3-UMI-Slow{_suffix}{_play}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={
                "env_cfg_entry_point": f"{__name__}.cabinet_book_insert_env_cfg:DualFR3CabinetBookInsert{_cfg}"
                + ("_PLAY" if _play else ""),
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetBookInsertSlow"
                + ("OpenDoor" if _suffix else "") + "PPORunnerCfg",
            },
            disable_env_checker=True,
        )
