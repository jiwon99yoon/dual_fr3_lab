# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from dual_fr3_lab.tasks.manager_based.reach.agents.rsl_rl_ppo_cfg import TanhSquashedGaussianDistributionCfg


@configclass
class HandoverCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """핸드오버 v0. cube lift Sim2Real와 같은 tanh actor/PPO에서 세 가지만 바꿈:

    - critic에 단계 one-hot 등 특권 관측(critic 그룹)을 더 준다 (actor는 실물 신호만)
    - 네트워크 [256, 128, 64] -> [512, 256, 128]: obs 62D/action 16D, 단계가 7개인 긴 과제
    - gamma 0.98 -> 0.99: 12 s 에피소드에서 다음 단계 보상을 더 멀리 본다 (실효 지평 ~5 s)
    """

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "dual_fr3_handover_cube"
    obs_groups = {"actor": ["policy"], "critic": ["policy", "critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=TanhSquashedGaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    # actor 출력이 이미 (-1, 1): wrapper clip은 수치 안전장치로만 유지 (Reach Sim2Real와 동일)
    clip_actions = 1.0
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HandoverCubeV1PPORunnerCfg(HandoverCubePPORunnerCfg):
    """핸드오버 v1 (env만 바뀜, PPO는 v0과 같음). 로그 폴더만 나눈다."""

    experiment_name = "dual_fr3_handover_cube_v1"
