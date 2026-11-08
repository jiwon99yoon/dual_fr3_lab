# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from dual_fr3_lab.tasks.manager_based.handover_cube.agents.rsl_rl_ppo_cfg import HandoverCubePPORunnerCfg


@configclass
class PickPlaceCubePPORunnerCfg(HandoverCubePPORunnerCfg):
    """핸드오버와 같은 tanh actor/PPO ([512, 256, 128], gamma 0.99, critic에 특권 관측). 로그 폴더만 나눈다.

    env 4096개 기준. 5070(12 GB)에서 메모리가 모자라면 --num_envs 2048로 돌린다.
    """

    experiment_name = "dual_fr3_pick_place_cube"


@configclass
class PickPlaceCubeGripperBusyPPORunnerCfg(PickPlaceCubePPORunnerCfg):
    experiment_name = "dual_fr3_pick_place_cube_gripper_busy"


@configclass
class PickPlaceCubeCriticPlusPPORunnerCfg(PickPlaceCubePPORunnerCfg):
    experiment_name = "dual_fr3_pick_place_cube_critic_plus"
