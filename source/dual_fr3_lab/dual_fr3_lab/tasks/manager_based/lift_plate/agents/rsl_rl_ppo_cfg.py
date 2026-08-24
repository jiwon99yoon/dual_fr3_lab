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


@configclass
class LiftPlateP40PPORunnerCfg(LiftPlateV2PPORunnerCfg):
    """P40 scratch training: j6 mirror loss, dedicated log namespace."""

    seed = 42
    max_iterations = 10000
    experiment_name = "dual_fr3_plate_lift_p40"


@configclass
class LiftPlateP41PPORunnerCfg(LiftPlateV2PPORunnerCfg):
    """P41 scratch training: grip_axis_alignment (TCP geometry), dedicated log namespace."""

    seed = 42
    max_iterations = 10000
    experiment_name = "dual_fr3_plate_lift_p41"


@configclass
class LiftPlateP42PPORunnerCfg(LiftPlateV2PPORunnerCfg):
    """P42 scratch training: plates_misalignment -4 → -6, dedicated log namespace."""

    seed = 42
    max_iterations = 10000
    experiment_name = "dual_fr3_plate_lift_p42"


@configclass
class LiftPlateP43PPORunnerCfg(LiftPlateV2PPORunnerCfg):
    """P43 scratch training: plate_approach_symmetry proximity 0.05→0.08, dedicated log namespace."""

    seed = 42
    max_iterations = 10000
    experiment_name = "dual_fr3_plate_lift_p43"


@configclass
class LiftPlateP44PPORunnerCfg(LiftPlateV2PPORunnerCfg):
    """P44 scratch training: yawtrack trigger 3.0→1.5, dedicated log namespace."""

    seed = 42
    max_iterations = 10000
    experiment_name = "dual_fr3_plate_lift_p44"


@configclass
class LiftPlateP45PPORunnerCfg(LiftPlateV2PPORunnerCfg):
    """P45 scratch training: P44 + one_sided_contact -5→-7, dedicated log namespace."""

    seed = 42
    max_iterations = 10000
    experiment_name = "dual_fr3_plate_lift_p45"


@configclass
class LiftPlateP46PPORunnerCfg(LiftPlateV2PPORunnerCfg):
    """P46 scratch training: P44 + box_upright curriculum -3→-4, dedicated log namespace."""

    seed = 42
    max_iterations = 10000
    experiment_name = "dual_fr3_plate_lift_p46"


@configclass
class LiftPlateP47PPORunnerCfg(LiftPlateV2PPORunnerCfg):
    """P47 scratch training: P44 + plate_height_symmetry curriculum -6→-8, dedicated log namespace."""

    seed = 42
    max_iterations = 10000
    experiment_name = "dual_fr3_plate_lift_p47"


@configclass
class LiftPlateP48PPORunnerCfg(LiftPlateV2PPORunnerCfg):
    """P48 scratch training: P44 + excessive_squeeze threshold 100→120N, dedicated log namespace."""

    seed = 42
    max_iterations = 10000
    experiment_name = "dual_fr3_plate_lift_p48"


@configclass
class LiftPlateP49PPORunnerCfg(LiftPlateV2PPORunnerCfg):
    """P49 scratch training: P47 + plate_face_alignment reward (curriculum), dedicated log namespace."""

    seed = 42
    max_iterations = 10000
    experiment_name = "dual_fr3_plate_lift_p49"


@configclass
class LiftPlateP50PPORunnerCfg(LiftPlateV2PPORunnerCfg):
    """P50 scratch training: P49 + plate_face_alignment weight 2.0→1.0, dedicated log namespace."""

    seed = 42
    max_iterations = 10000
    experiment_name = "dual_fr3_plate_lift_p50"


@configclass
class LiftPlateP51PPORunnerCfg(LiftPlateV2PPORunnerCfg):
    """P51 scratch training: P49 + plate_face_z_offset penalty (curriculum), dedicated log namespace."""

    seed = 42
    max_iterations = 10000
    experiment_name = "dual_fr3_plate_lift_p51"

