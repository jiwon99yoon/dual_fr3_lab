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

gym.register(
    id="Isaac-Cabinet-BookInsert-DualFR3-UMI-Upright-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cabinet_book_insert_env_cfg:DualFR3CabinetBookInsertUprightEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetBookInsertPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Cabinet-BookInsert-DualFR3-UMI-Upright-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cabinet_book_insert_env_cfg:DualFR3CabinetBookInsertUprightEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetBookInsertPPORunnerCfg",
    },
    disable_env_checker=True,
)

# Table75 (plan 7-12): 75 cm table (base +0.345), cabinet 10 cm closer; v2 recipe: OpenDoor stage one, then closed door
for _suffix, _cfg in (("", "Table75EnvCfg"), ("-OpenDoor", "Table75OpenDoorEnvCfg")):
    for _play in ("", "-Play"):
        gym.register(
            id=f"Isaac-Cabinet-BookInsert-DualFR3-UMI-Table75{_suffix}{_play}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={
                "env_cfg_entry_point": f"{__name__}.cabinet_book_insert_env_cfg:DualFR3CabinetBookInsert{_cfg}"
                + ("_PLAY" if _play else ""),
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetBookInsertTable75"
                + ("OpenDoor" if _suffix else "") + "PPORunnerCfg",
            },
            disable_env_checker=True,
        )

# DoorFirst (2026-11-07): v2 + the book waits beside the door until it is open; same experiment as v2 so stage two
# resumes from the open-door stage one (open_door_transfer_2026-11-06/model_2499.pt)
for _play in ("", "-Play"):
    gym.register(
        id=f"Isaac-Cabinet-BookInsert-DualFR3-UMI-DoorFirst{_play}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.cabinet_book_insert_env_cfg:DualFR3CabinetBookInsertDoorFirstEnvCfg"
            + ("_PLAY" if _play else ""),
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetBookInsertPPORunnerCfg",
        },
        disable_env_checker=True,
    )

# Table75 + DoorFirst (2026-11-07): the 75 cm experiment, so stage two resumes from a Table75-OpenDoor run
for _play in ("", "-Play"):
    gym.register(
        id=f"Isaac-Cabinet-BookInsert-DualFR3-UMI-Table75-DoorFirst{_play}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.cabinet_book_insert_env_cfg:DualFR3CabinetBookInsertTable75DoorFirstEnvCfg"
            + ("_PLAY" if _play else ""),
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetBookInsertTable75PPORunnerCfg",
        },
        disable_env_checker=True,
    )

# Table75 DoorFirst curriculum (2026-11-07): Upright rewards, DoorFirst ramped in once the insertion succeeds (scratch)
for _play in ("", "-Play"):
    gym.register(
        id=f"Isaac-Cabinet-BookInsert-DualFR3-UMI-Table75-DoorFirstCurriculum{_play}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.cabinet_book_insert_env_cfg:"
            "DualFR3CabinetBookInsertTable75DoorFirstCurriculumEnvCfg" + ("_PLAY" if _play else ""),
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetBookInsertTable75PPORunnerCfg",
        },
        disable_env_checker=True,
    )

# + tilt penalty (2026-11-07): book tilt above 15 deg penalized at all times (the curriculum run carried it at 45 deg)
for _play in ("", "-Play"):
    gym.register(
        id=f"Isaac-Cabinet-BookInsert-DualFR3-UMI-Table75-DoorFirstCurriculum-Tilt{_play}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.cabinet_book_insert_env_cfg:"
            "DualFR3CabinetBookInsertTable75DoorFirstCurriculumTiltEnvCfg" + ("_PLAY" if _play else ""),
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetBookInsertTable75PPORunnerCfg",
        },
        disable_env_checker=True,
    )

# + Hold (2026-11-07): left action penalized once the door is open, episode ends above 60 N on the handle
# (real 2998 kept moving the left arm after opening: the training weld hid a steady action offset)
for _play in ("", "-Play"):
    gym.register(
        id=f"Isaac-Cabinet-BookInsert-DualFR3-UMI-Table75-DoorFirst-Hold{_play}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.cabinet_book_insert_env_cfg:"
            "DualFR3CabinetBookInsertTable75DoorFirstHoldEnvCfg" + ("_PLAY" if _play else ""),
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetBookInsertTable75PPORunnerCfg",
        },
        disable_env_checker=True,
    )
