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

# A + 기울기 벌점 (11-09): A(파지 수정 포함)와 동시에 돌려 기울기 벌점 효과만 본다
gym.register(
    id="Isaac-PickPlace-Cube-DualFR3-UMI-GripperBusy-Tilt-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_env_cfg:DualFR3PickPlaceCubeGripperBusyTiltEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceCubeGripperBusyTiltPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-Cube-DualFR3-UMI-GripperBusy-Tilt-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_env_cfg:DualFR3PickPlaceCubeGripperBusyTiltEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceCubeGripperBusyTiltPPORunnerCfg",
    },
    disable_env_checker=True,
)

# A + 책상 10 cm 당김 (11-09): 앞 모서리 x 0.30, 큐브 시작·목표 x 0.35~0.65. model_4000에서 이어 학습
gym.register(
    id="Isaac-PickPlace-Cube-DualFR3-UMI-GripperBusy-Near-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_env_cfg:DualFR3PickPlaceCubeGripperBusyNearEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceCubeGripperBusyNearPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-Cube-DualFR3-UMI-GripperBusy-Near-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_env_cfg:DualFR3PickPlaceCubeGripperBusyNearEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceCubeGripperBusyNearPPORunnerCfg",
    },
    disable_env_checker=True,
)

# A + B (11-09): actor는 A, critic은 B. 4090 B가 파지·운반 기울기가 A보다 작았다
gym.register(
    id="Isaac-PickPlace-Cube-DualFR3-UMI-GripperBusy-CriticPlus-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_env_cfg:DualFR3PickPlaceCubeGripperBusyCriticPlusEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceCubeGripperBusyCriticPlusPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-Cube-DualFR3-UMI-GripperBusy-CriticPlus-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_env_cfg:DualFR3PickPlaceCubeGripperBusyCriticPlusEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceCubeGripperBusyCriticPlusPPORunnerCfg",
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

# -Curriculum (11-10): GripperBusy + CriticPlus + 기울기·끌기·손가락 힘 벌점 커리큘럼 + 그리퍼 settle 0.8~1.2 s + Near 범위 ramp
gym.register(
    id="Isaac-PickPlace-Cube-DualFR3-UMI-Curriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_env_cfg:DualFR3PickPlaceCubeCurriculumEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceCubeCurriculumPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-Cube-DualFR3-UMI-Curriculum-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_env_cfg:DualFR3PickPlaceCubeCurriculumEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceCubeCurriculumPPORunnerCfg",
    },
    disable_env_checker=True,
)

# -Curriculum ablation (11-12): Near 배치·x ramp만 끔 (기준 배치). Play-NearEnd: Curriculum 정책을 x 0.35~0.45만으로 평가
for _id, _cls, _runner in (
    ("Isaac-PickPlace-Cube-DualFR3-UMI-Curriculum-NoNear-v0", "DualFR3PickPlaceCubeCurriculumNoNearEnvCfg", "PickPlaceCubeCurriculumNoNearPPORunnerCfg"),
    ("Isaac-PickPlace-Cube-DualFR3-UMI-Curriculum-NoNear-Play-v0", "DualFR3PickPlaceCubeCurriculumNoNearEnvCfg_PLAY", "PickPlaceCubeCurriculumNoNearPPORunnerCfg"),
    ("Isaac-PickPlace-Cube-DualFR3-UMI-Curriculum-Play-NearEnd-v0", "DualFR3PickPlaceCubeCurriculumEnvCfg_PLAY_NEAREND", "PickPlaceCubeCurriculumPPORunnerCfg"),
    # NoNear + tilt off (11-13 ablation 1)
    ("Isaac-PickPlace-Cube-DualFR3-UMI-Curriculum-NoNear-NoTilt-v0", "DualFR3PickPlaceCubeCurriculumNoNearNoTiltEnvCfg", "PickPlaceCubeCurriculumNoNearNoTiltPPORunnerCfg"),
    ("Isaac-PickPlace-Cube-DualFR3-UMI-Curriculum-NoNear-NoTilt-Play-v0", "DualFR3PickPlaceCubeCurriculumNoNearNoTiltEnvCfg_PLAY", "PickPlaceCubeCurriculumNoNearNoTiltPPORunnerCfg"),
    # NoNear + NoTilt + finger_force off (11-13 ablation 2)
    ("Isaac-PickPlace-Cube-DualFR3-UMI-Curriculum-NoNear-NoTilt-NoFinger-v0", "DualFR3PickPlaceCubeCurriculumNoNearNoTiltNoFingerEnvCfg", "PickPlaceCubeCurriculumNoNearNoTiltNoFingerPPORunnerCfg"),
    ("Isaac-PickPlace-Cube-DualFR3-UMI-Curriculum-NoNear-NoTilt-NoFinger-Play-v0", "DualFR3PickPlaceCubeCurriculumNoNearNoTiltNoFingerEnvCfg_PLAY", "PickPlaceCubeCurriculumNoNearNoTiltNoFingerPPORunnerCfg"),
    # NoNear + NoTilt + gripper_settle off (11-13 ablation 3; finger_force는 유지)
    ("Isaac-PickPlace-Cube-DualFR3-UMI-Curriculum-NoNear-NoTilt-NoSettle-v0", "DualFR3PickPlaceCubeCurriculumNoNearNoTiltNoSettleEnvCfg", "PickPlaceCubeCurriculumNoNearNoTiltNoSettlePPORunnerCfg"),
    ("Isaac-PickPlace-Cube-DualFR3-UMI-Curriculum-NoNear-NoTilt-NoSettle-Play-v0", "DualFR3PickPlaceCubeCurriculumNoNearNoTiltNoSettleEnvCfg_PLAY", "PickPlaceCubeCurriculumNoNearNoTiltNoSettlePPORunnerCfg"),
    # Near on + tilt off (11-13 ablation 4: run 2의 80 %가 Near 탓인지)
    ("Isaac-PickPlace-Cube-DualFR3-UMI-Curriculum-NoTilt-v0", "DualFR3PickPlaceCubeCurriculumNoTiltEnvCfg", "PickPlaceCubeCurriculumNoTiltPPORunnerCfg"),
    ("Isaac-PickPlace-Cube-DualFR3-UMI-Curriculum-NoTilt-Play-v0", "DualFR3PickPlaceCubeCurriculumNoTiltEnvCfg_PLAY", "PickPlaceCubeCurriculumNoTiltPPORunnerCfg"),
):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.pick_place_env_cfg:{_cls}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{_runner}",
        },
        disable_env_checker=True,
    )
