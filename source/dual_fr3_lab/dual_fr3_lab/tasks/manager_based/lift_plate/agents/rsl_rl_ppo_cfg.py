# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class LiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "dual_fr3_plate_lift"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class LiftPlateV1PPORunnerCfg(LiftCubePPORunnerCfg):
    """v1(특정 박스) 전용 - 로그를 dual_fr3_plate_lift_v1/로 분리 (v0과 run 이력 혼합 방지)."""

    experiment_name = "dual_fr3_plate_lift_v1"


@configclass
class LiftPlateV2PPORunnerCfg(LiftCubePPORunnerCfg):
    """v2(v1 + 외란) 전용 - 로그를 dual_fr3_plate_lift_v2/로 분리 (v1/v2 동시 실행·비교)."""

    experiment_name = "dual_fr3_plate_lift_v2"


@configclass
class LiftPlateP39PPORunnerCfg(LiftPlateV2PPORunnerCfg):
    """P39 scratch training: P38 PPO/seed preserved, with a dedicated log namespace."""

    seed = 42
    max_iterations = 10000
    experiment_name = "dual_fr3_plate_lift_p39"
