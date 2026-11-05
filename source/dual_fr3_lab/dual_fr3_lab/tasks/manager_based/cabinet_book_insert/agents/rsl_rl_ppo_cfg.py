"""rsl_rl PPO for the cabinet book insert (plan 7-6): plate Sim2Real-100Hz actor, larger nets, longer horizon."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from dual_fr3_lab.tasks.manager_based.reach.agents.rsl_rl_ppo_cfg import TanhSquashedGaussianDistributionCfg


@configclass
class CabinetBookInsertPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "dual_fr3_cabinet_book_insert"
    # asymmetric actor-critic: the critic gets contact/handle forces, door velocity and the DR values
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    # tanh-squashed actor output is already in (-1, 1); the wrapper clip only guards numerics (plate Sim2Real)
    clip_actions = 1.0
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
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,  # 12 s = 240 policy steps (plate: 0.98)
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
