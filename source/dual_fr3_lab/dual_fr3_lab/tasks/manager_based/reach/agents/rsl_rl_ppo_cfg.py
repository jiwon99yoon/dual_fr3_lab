# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DualFR3ReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Dual FR3 reach — plate task와 동일 네트워크/하이퍼파라미터.

    obs 58D (plate 구조 호환), action 14D (delta joint pos × 0.1).
    """

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "dual_fr3_reach"
    seed = 42

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
class DualFR3ReachSnapshotPPORunnerCfg(DualFR3ReachPPORunnerCfg):
    """Controller-1 target-hold ablation; only log namespace differs from v0."""

    experiment_name = "dual_fr3_reach_snapshot"
