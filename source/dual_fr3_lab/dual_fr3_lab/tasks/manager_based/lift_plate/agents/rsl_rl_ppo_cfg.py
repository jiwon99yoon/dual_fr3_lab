# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from dual_fr3_lab.tasks.manager_based.reach.agents.rsl_rl_ppo_cfg import TanhSquashedGaussianDistributionCfg


@configclass
class LiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "dual_fr3_plate_lift"
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
class LiftPlateP47FrictionPPORunnerCfg(LiftPlateP47PPORunnerCfg):
    """P47-Friction scratch training: P47 + MuJoCo-matched arm passive dynamics, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_p47_friction"


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


@configclass
class LiftPlateP52PPORunnerCfg(LiftPlateV2PPORunnerCfg):
    """P52 scratch training: P47 + box_anti_yaw penalty (G+lifted gate, curriculum), dedicated log namespace."""

    seed = 42
    max_iterations = 10000
    experiment_name = "dual_fr3_plate_lift_p52"



@configclass
class LiftPlateSim2Real1kHzPPORunnerCfg(LiftPlateP47PPORunnerCfg):
    """Sim2Real-1kHz scratch training: Reach Sim2Real와 같은 tanh-squashed bounded actor, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_1khz"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=TanhSquashedGaussianDistributionCfg(init_std=1.0),
    )
    # actor 출력이 이미 (-1, 1): wrapper clip은 수치 안전장치로만 유지 (Reach Sim2Real와 동일)
    clip_actions = 1.0


@configclass
class LiftPlateSim2Real100HzPPORunnerCfg(LiftPlateSim2Real1kHzPPORunnerCfg):
    """Sim2Real-100Hz scratch training: 1kHz와 같은 actor/PPO, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz"


@configclass
class LiftPlateSim2Real100HzOffset03PPORunnerCfg(LiftPlateSim2Real100HzPPORunnerCfg):
    """Sim2Real-100Hz-Offset03 진단 run: 100Hz와 같은 actor/PPO, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_offset03"


@configclass
class LiftPlateSim2Real100HzOffset03LevelPPORunnerCfg(LiftPlateSim2Real100HzPPORunnerCfg):
    """Sim2Real-100Hz-Offset03-Level: Offset03와 같은 actor/PPO, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_offset03_level"


@configclass
class LiftPlateSim2Real100HzOffset03PlateOnlyPPORunnerCfg(LiftPlateSim2Real100HzPPORunnerCfg):
    """Sim2Real-100Hz-Offset03-PlateOnly: Offset03와 같은 actor/PPO, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_offset03_plateonly"


@configclass
class LiftPlateSim2Real100HzLightPPORunnerCfg(LiftPlateSim2Real100HzPPORunnerCfg):
    """Sim2Real-100Hz-Light: PlateOnly와 같은 actor/PPO, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light"


@configclass
class LiftPlateSim2Real100HzLightBox410PPORunnerCfg(LiftPlateSim2Real100HzPPORunnerCfg):
    """Sim2Real-100Hz-Light-Box410: Light와 같은 actor/PPO, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_box410"


@configclass
class LiftPlateSim2Real100HzLightBox410J1MirrorPPORunnerCfg(LiftPlateSim2Real100HzPPORunnerCfg):
    """Sim2Real-100Hz-Light-Box410-J1Mirror: Light와 같은 actor/PPO, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_box410_j1mirror"


@configclass
class LiftPlateSim2Real100HzLightPregraspPPORunnerCfg(LiftPlateSim2Real100HzPPORunnerCfg):
    """Sim2Real-100Hz-Light-Pregrasp: Light와 같은 actor/PPO, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_pregrasp"


@configclass
class LiftPlateSim2Real100HzLightPregraspUprightPPORunnerCfg(LiftPlateSim2Real100HzPPORunnerCfg):
    """Sim2Real-100Hz-Light-Pregrasp-Upright: Light와 같은 actor/PPO, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_upright"


@configclass
class LiftPlateSim2Real100HzLightPregraspRollAlignPPORunnerCfg(LiftPlateSim2Real100HzPPORunnerCfg):
    """Sim2Real-100Hz-Light-Pregrasp-RollAlign: Light와 같은 actor/PPO, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_rollalign"


@configclass
class LiftPlateSim2Real100HzLightPregraspWidePPORunnerCfg(LiftPlateSim2Real100HzPPORunnerCfg):
    """Sim2Real-100Hz-Light-Pregrasp-Wide: Light와 같은 actor/PPO, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide"


@configclass
class LiftPlateSim2Real100HzLightPregraspWideRollAlignPPORunnerCfg(LiftPlateSim2Real100HzPPORunnerCfg):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign: Light와 같은 actor/PPO, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign"


@configclass
class LiftPlateSim2Real100HzLightPregraspWideRollAlignFrozenBoxPPORunnerCfg(LiftPlateSim2Real100HzPPORunnerCfg):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FrozenBox: actor는 policy 그룹(박스 pose 시작 값 고정),
    critic은 critic 그룹(실제 박스 pose)."""

    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_frozenbox"


@configclass
class LiftPlateSim2Real100HzLightPregraspWideRollAlignBoxDropPPORunnerCfg(LiftPlateSim2Real100HzPPORunnerCfg):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-BoxDrop: actor는 policy 그룹(박스 pose 지연·끊김),
    critic은 critic 그룹(실제 박스 pose)."""

    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_boxdrop"


@configclass
class LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripPPORunnerCfg(LiftPlateSim2Real100HzPPORunnerCfg):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip: Wide-RollAlign과 같은 actor/PPO, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_facegrip"


@configclass
class LiftPlateSim2Real100HzLightPregraspWideRollAlignDisturbPPORunnerCfg(LiftPlateSim2Real100HzPPORunnerCfg):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-Disturb: Wide-RollAlign과 같은 actor/PPO, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_disturb"


@configclass
class LiftPlateSim2Real100HzLightPregraspWideRollAlignRegraspPPORunnerCfg(LiftPlateSim2Real100HzPPORunnerCfg):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-Regrasp: Wide-RollAlign과 같은 actor/PPO, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_regrasp"


@configclass
class LiftPlateSim2Real100HzLightPregraspWideRollAlignDRPPORunnerCfg(LiftPlateSim2Real100HzPPORunnerCfg):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-DR: Wide-RollAlign과 같은 actor/PPO, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_dr"


@configclass
class LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripMinSqueezePPORunnerCfg(LiftPlateSim2Real100HzPPORunnerCfg):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-MinSqueeze: FaceGrip과 같은 actor/PPO, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_facegrip_minsqueeze"


@configclass
class LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripMinSqueezeScratchPPORunnerCfg(LiftPlateSim2Real100HzPPORunnerCfg):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-MinSqueeze-Scratch: Wide-RollAlign과 같은 actor/PPO, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_facegrip_minsqueeze_scratch"


@configclass
class LiftPlateSim2Real100HzLightBox410WideJ1MirrorPPORunnerCfg(LiftPlateSim2Real100HzPPORunnerCfg):
    """Sim2Real-100Hz-Light-Box410-Wide-J1Mirror: Light와 같은 actor/PPO, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_box410_wide_j1mirror"


@configclass
class LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripBoxDropScratchPPORunnerCfg(LiftPlateSim2Real100HzPPORunnerCfg):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-BoxDrop-Scratch: actor는 policy 그룹(박스 pose 지연·끊김),
    critic은 critic 그룹(실제 박스 pose)."""

    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_facegrip_boxdrop_scratch"


@configclass
class LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripSqueezeLiftPPORunnerCfg(LiftPlateSim2Real100HzPPORunnerCfg):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-SqueezeLift: MinSqueeze와 같은 actor/PPO, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_facegrip_squeezelift"


@configclass
class LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripSqueezeLiftLowMuHeavyAct015ScratchPPORunnerCfg(
    LiftPlateSim2Real100HzPPORunnerCfg
):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-SqueezeLift-LowMuHeavy-Act015-Scratch: 같은 actor/PPO, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_facegrip_squeezelift_lowmuheavy_act015_scratch"


@configclass
class LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripBoxDropFlatSqueezePPORunnerCfg(
    LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripBoxDropScratchPPORunnerCfg
):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-BoxDrop-FlatSqueeze: FaceGrip-BoxDrop-Scratch와 같은 actor/critic 그룹, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_facegrip_boxdrop_flatsqueeze"


@configclass
class LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripBoxDropFlatSqueezeLowMuSoftLiftPPORunnerCfg(
    LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripBoxDropScratchPPORunnerCfg
):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-BoxDrop-FlatSqueeze-LowMuSoftLift: FlatSqueeze와 같은 actor/critic 그룹, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_facegrip_boxdrop_flatsqueeze_lowmusoftlift"


@configclass
class LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripBoxDropFlatSqueezeLowMuSoftLiftGoalXYPPORunnerCfg(
    LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripBoxDropScratchPPORunnerCfg
):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-BoxDrop-FlatSqueeze-LowMuSoftLift-GoalXY: FlatSqueeze와 같은 actor/critic 그룹, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_facegrip_boxdrop_flatsqueeze_lowmusoftlift_goalxy"


@configclass
class LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripBoxDropFlatOnlyPPORunnerCfg(
    LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripBoxDropScratchPPORunnerCfg
):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-BoxDrop-FlatOnly: FaceGrip-BoxDrop-Scratch와 같은 actor/critic 그룹, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_facegrip_boxdrop_flatonly"


@configclass
class LiftPlateSim2Real100HzLightPregraspWideRollAlignFaceGripSqueezeLiftLowMuFlatPPORunnerCfg(LiftPlateSim2Real100HzPPORunnerCfg):
    """Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-SqueezeLift-LowMuFlat: SqueezeLift와 같은 actor/PPO, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_facegrip_squeezelift_lowmuflat"


@configclass
class LiftPlateSim2Real100HzLightBox410WideJ1MirrorFaceGripBoxDropPPORunnerCfg(LiftPlateSim2Real100HzPPORunnerCfg):
    """Sim2Real-100Hz-Light-Box410-Wide-J1Mirror-FaceGrip-BoxDrop: actor는 policy 그룹(박스 pose 지연·끊김),
    critic은 critic 그룹(실제 박스 pose)."""

    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_box410_wide_j1mirror_facegrip_boxdrop"


@configclass
class LiftPlateSim2Real100HzLightBox410WideJ1MirrorFaceGripBoxDropFlatOnlyPPORunnerCfg(
    LiftPlateSim2Real100HzLightBox410WideJ1MirrorFaceGripBoxDropPPORunnerCfg
):
    """Sim2Real-100Hz-Light-Box410-Wide-J1Mirror-FaceGrip-BoxDrop-FlatOnly: 같은 actor/critic 그룹, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_box410_wide_j1mirror_facegrip_boxdrop_flatonly"


@configclass
class LiftPlateSim2Real100HzLightBox410WideJ1MirrorFaceGripBoxDropFlatSqueezePPORunnerCfg(
    LiftPlateSim2Real100HzLightBox410WideJ1MirrorFaceGripBoxDropPPORunnerCfg
):
    """Sim2Real-100Hz-Light-Box410-Wide-J1Mirror-FaceGrip-BoxDrop-FlatSqueeze: 같은 actor/critic 그룹, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_box410_wide_j1mirror_facegrip_boxdrop_flatsqueeze"


@configclass
class LiftPlateSim2Real100HzLightBox410WideJ1MirrorFaceGripBoxDropMinSqueezePPORunnerCfg(
    LiftPlateSim2Real100HzLightBox410WideJ1MirrorFaceGripBoxDropPPORunnerCfg
):
    """Sim2Real-100Hz-Light-Box410-Wide-J1Mirror-FaceGrip-BoxDrop-MinSqueeze: 같은 actor/critic 그룹, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_box410_wide_j1mirror_facegrip_boxdrop_minsqueeze"


@configclass
class LiftPlateSim2Real100HzLightBox410WideJ1MirrorFaceGripBoxDropFlatSqueezeLowMuSoftLiftPPORunnerCfg(
    LiftPlateSim2Real100HzLightBox410WideJ1MirrorFaceGripBoxDropPPORunnerCfg
):
    """Sim2Real-100Hz-Light-Box410-Wide-J1Mirror-FaceGrip-BoxDrop-FlatSqueeze-LowMuSoftLift: 같은 actor/critic 그룹, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_box410_wide_j1mirror_facegrip_boxdrop_flatsqueeze_lowmusoftlift"


@configclass
class LiftPlateSim2Real100HzLightBox410WideJ1MirrorFaceGripBoxDropFlatSqueezeLowMuSoftLiftGoalFinePPORunnerCfg(
    LiftPlateSim2Real100HzLightBox410WideJ1MirrorFaceGripBoxDropPPORunnerCfg
):
    """Sim2Real-100Hz-Light-Box410-Wide-J1Mirror-FaceGrip-BoxDrop-FlatSqueeze-LowMuSoftLift-GoalFine: 같은 actor/critic 그룹, dedicated log namespace."""

    experiment_name = "dual_fr3_plate_lift_sim2real_100hz_light_box410_wide_j1mirror_facegrip_boxdrop_flatsqueeze_lowmusoftlift_goalfine"
