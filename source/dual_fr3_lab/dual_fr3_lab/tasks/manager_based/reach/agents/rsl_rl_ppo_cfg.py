# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class TanhSquashedGaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """Configuration for the task-local bounded Gaussian policy."""

    class_name = (
        "dual_fr3_lab.tasks.manager_based.reach.agents.tanh_squashed_gaussian:"
        "TanhSquashedGaussianDistribution"
    )
    std_type = "log"
    epsilon: float = 1.0e-6


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
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
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
class DualFR3ReachSim2RealPPORunnerCfg(DualFR3ReachPPORunnerCfg):
    """Reach v1 runner using a tanh-squashed bounded Gaussian actor."""

    experiment_name = "dual_fr3_reach_sim2real_v1"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=TanhSquashedGaussianDistributionCfg(init_std=1.0),
    )
    # The actor is already in (-1, 1); retain the wrapper clip as a numerical safety guard.
    clip_actions = 1.0
    max_iterations = 5000
