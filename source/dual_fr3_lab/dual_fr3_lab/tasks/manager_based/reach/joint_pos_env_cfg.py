# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Dual FR3 양팔 EE reach task — sim2real gap 분석용.

목표: 양팔 EE를 goal center ± REACH_OFFSET_Y (y축) 목표점으로 동시에 이동.

obs 58D = plate task와 완전 동일 구조:
  joint_pos(14) + joint_vel(14) + center_pos(3) + goal_cmd(7)
  + prev_action(14) + zeros_forces(2) + zeros_ori(4)
plate_forces·object_orientation 슬롯은 0으로 채워 구조 호환 유지.
action semantics (delta joint pos × 0.1, 20Hz) 도 plate와 동일.
"""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.commands import UniformPoseCommand
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from dual_fr3_lab.robots.dual_fr3_plate_ee import DUAL_FR3_PLATE_EE_CFG, PLATE_TCP_OFFSET_POS  # isort: skip
from dual_fr3_lab.tasks.manager_based.lift_cube import mdp  # isort: skip

ROOT_Z = 0.405       # base frame 높이 (plate task와 동일)
REACH_OFFSET_Y = 0.20  # 각 팔 EE target = center ± 0.20m (y축)


# ──────────────────────────── custom obs ─────────────────────────────

def reach_center_pos(env, command_name: str = "target_pose") -> torch.Tensor:
    """goal center position in robot root frame (3D). object_position 슬롯 대응."""
    return env.command_manager.get_command(command_name)[:, :3]


def zeros_plate_forces(env) -> torch.Tensor:
    """(N, 2) 영벡터 — plate_forces 슬롯 구조 호환용."""
    return torch.zeros(env.num_envs, 2, device=env.device)


def zeros_obj_orientation(env) -> torch.Tensor:
    """(N, 4) 영벡터 — object_orientation 슬롯 구조 호환용."""
    return torch.zeros(env.num_envs, 4, device=env.device)


# ──────────────────────────── custom reward ──────────────────────────

def dual_ee_reach(
    env,
    std: float,
    command_name: str = "target_pose",
    offset_y: float = REACH_OFFSET_Y,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """양팔 EE → goal center ± offset_y 목표점 거리 kernel의 min.

    min(left, right): 뒤처진 팔이 보상을 결정 — 양팔 동시 도달 유도.
    """
    robot = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    names = ee_frame.data.target_frame_names
    li = names.index("left_ee")
    ri = names.index("right_ee")

    cmd = env.command_manager.get_command(command_name)[:, :3]
    center_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, cmd)

    off = torch.zeros_like(center_w)
    off[:, 1] = offset_y
    left_target = center_w + off   # 왼팔: +y쪽
    right_target = center_w - off  # 오른팔: -y쪽

    d_l = torch.norm(ee_frame.data.target_pos_w[:, li] - left_target, dim=-1)
    d_r = torch.norm(ee_frame.data.target_pos_w[:, ri] - right_target, dim=-1)
    return torch.minimum(1.0 - torch.tanh(d_l / std), 1.0 - torch.tanh(d_r / std))


# ──────────────────────────── scene ──────────────────────────────────

@configclass
class DualFR3ReachSceneCfg(InteractiveSceneCfg):
    robot = DUAL_FR3_PLATE_EE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    # ee_frame / arms_contact: __post_init__에서 self.scene.* 패턴으로 설정 (plate env와 동일)


# ──────────────────────────── dual-arm reach command (per-arm visualization) ──

class DualArmReachCommand(UniformPoseCommand):
    """UniformPoseCommand + 각 팔 target 구 마커 시각화.

    왼팔(파랑) = center + offset_y, 오른팔(주황) = center - offset_y.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_arm_visualizer"):
                self._arm_visualizer = VisualizationMarkers(
                    VisualizationMarkersCfg(
                        prim_path="/Visuals/DualReachTargets",
                        markers={
                            "left": sim_utils.SphereCfg(
                                radius=0.04,
                                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 1.0)),
                            ),
                            "right": sim_utils.SphereCfg(
                                radius=0.04,
                                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.1)),
                            ),
                        },
                    )
                )
            self._arm_visualizer.set_visibility(True)
        else:
            if hasattr(self, "_arm_visualizer"):
                self._arm_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        # center in world frame
        center_w = self.pose_command_w[:, :3]  # (N, 3)
        off = torch.zeros_like(center_w)
        off[:, 1] = REACH_OFFSET_Y
        left_w = center_w + off   # (N, 3)  왼팔 +y
        right_w = center_w - off  # (N, 3)  오른팔 -y
        # interleave: [left_0, right_0, left_1, right_1, ...]
        positions = torch.stack([left_w, right_w], dim=1).reshape(-1, 3)  # (2N, 3)
        marker_idx = torch.zeros(positions.shape[0], dtype=torch.long, device=self.device)
        marker_idx[1::2] = 1  # 짝수=left(0), 홀수=right(1)
        self._arm_visualizer.visualize(positions, marker_indices=marker_idx)


@configclass
class DualArmReachCommandCfg(mdp.UniformPoseCommandCfg):
    class_type: type = DualArmReachCommand


# ──────────────────────────── manager configs ─────────────────────────

@configclass
class CommandsCfg:
    target_pose = DualArmReachCommandCfg(
        asset_name="robot",
        body_name="base",
        resampling_time_range=(4.0, 8.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.40, 0.60),
            pos_y=(-0.10, 0.10),
            pos_z=(0.10, 0.35),   # base frame → world 0.555~0.755m
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    arm_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_fr3_joint[1-7]"],
        scale=0.1,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # joint_pos (14): plate task와 동일 asset_cfg
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_fr3_joint[1-7]"])},
            noise=Gnoise(std=0.005),
        )
        # joint_vel (14)
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_fr3_joint[1-7]"])},
            noise=Gnoise(std=0.05),
        )
        # object_position 슬롯 → goal center pos (3): plate의 박스 위치와 동일 차원
        object_position = ObsTerm(
            func=reach_center_pos,
            params={"command_name": "target_pose"},
            noise=Gnoise(std=0.01),
        )
        # target_object_position 슬롯 → full goal command (7: pos+quat)
        target_object_position = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "target_pose"},
        )
        # prev action (14)
        actions = ObsTerm(func=mdp.last_action)
        # plate_forces 슬롯 → zeros (2): 구조 호환
        plate_forces = ObsTerm(func=zeros_plate_forces)
        # object_orientation 슬롯 → zeros (4): 구조 호환
        object_orientation = ObsTerm(func=zeros_obj_orientation)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    # coarse (std=0.3): 탐색 초기 양팔 유도
    reach_coarse = RewTerm(
        func=dual_ee_reach,
        params={"std": 0.3, "command_name": "target_pose"},
        weight=2.0,
    )
    # fine (std=0.05): 목표점 근방 정밀 도달
    reach_fine = RewTerm(
        func=dual_ee_reach,
        params={"std": 0.05, "command_name": "target_pose"},
        weight=5.0,
    )
    # 동작 규칙성 (plate와 동일 weight)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_fr3_joint[1-7]"])},
        weight=-0.001,
    )
    # 팔 충돌 페널티
    undesired_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("arms_contact")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_fr3_joint[1-7]"]),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )


# ──────────────────────────── main env cfg ───────────────────────────

@configclass
class DualFR3ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Dual FR3 EE reach — plate obs 구조(58D) 완전 호환."""

    scene: DualFR3ReachSceneCfg = DualFR3ReachSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # ── decimation: cube task와 동일 (20 Hz policy, 100 Hz physics)
    #    plate task는 LiftEnvCfg 기본 decimation=2 (50 Hz) 를 그대로 사용한다.
    #    reach는 카메라 30 Hz 기준(20 < 30)으로 20 Hz를 채택 — cube와 동일.
    decimation = 5
    episode_length_s = 8.0

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.01  # 100 Hz
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024

        # ── EE frame transformer (plate env와 동일 패턴: __post_init__에서 scene에 추가) ──
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/left_fr3_link7",
                    name="left_ee",
                    offset=OffsetCfg(pos=PLATE_TCP_OFFSET_POS),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_fr3_link7",
                    name="right_ee",
                    offset=OffsetCfg(pos=PLATE_TCP_OFFSET_POS),
                ),
            ],
        )

        # ── 팔 충돌 감지 (팔-팔, 팔-지면 접촉 페널티용) ──
        self.scene.arms_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_fr3_link[1-7]",
            history_length=3,
            track_air_time=False,
        )


@configclass
class DualFR3ReachEnvCfg_PLAY(DualFR3ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
