# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""grasp gating / hold-still 보상 (2026-07-27, dual_fr3 cube lift Run A~C에서 검증).

- object_grasped: 양 finger pad가 동시에 Object를 threshold(N) 이상으로 누를 때만 1
  -> 높이 gate만으로는 "손등에 얹어 나르는" exploit이 이득 (2026-07-25 run에서 실증)
- object_velocity_near_goal: goal 근처에서만 물체 속도 페널티 (in-hand 환경들의 속도 감쇠 철학)
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import combine_frame_transforms

from isaaclab_tasks.manager_based.manipulation.lift.mdp.rewards import object_goal_distance, object_is_lifted

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_grasped(
    env: ManagerBasedRLEnv,
    finger1_sensor_cfg: SceneEntityCfg,
    finger2_sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    filter_idx: int = 0,
) -> torch.Tensor:
    """Binary gate: 1.0 if both finger pads are in contact with the filtered object.

    Each contact sensor must match a single finger body and set ``filter_prim_paths_expr`` with the
    object at index ``filter_idx`` (기본 0), so ``force_matrix_w`` reports finger-object forces only.
    PhysX requires one filter target per sensor body, hence one sensor per finger. Requiring *both*
    pads above the force threshold accepts a true grasp and rejects balancing the object on the
    back of the hand.

    주의 (Run D): finger 센서에 Table filter(index 1)가 추가되었으므로 반드시 filter를 인덱스로
    선택해야 한다 — 전체 filter에 amax를 취하면 테이블을 누르는 것도 파지로 오인함.
    """
    finger_forces = []
    for cfg in (finger1_sensor_cfg, finger2_sensor_cfg):
        sensor: ContactSensor = env.scene.sensors[cfg.name]
        # (num_envs, 1, num_filters, 3) -> Object filter만 선택 -> per-env force magnitude
        finger_forces.append(torch.norm(sensor.data.force_matrix_w[:, :, filter_idx, :], dim=-1).amax(dim=-1))
    return (torch.minimum(*finger_forces) > threshold).float()


def object_upright_while_carried(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    finger1_sensor_cfg: SceneEntityCfg,
    finger2_sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """운반 중 cube 윗면 유지 보상: cube z축과 world z축의 정렬도 [0,1] × (파지 ∧ 들림) gate.

    Run D: 수평 유지 운반은 파지면 비틀림 토크가 작아 슬립 여유가 크고 궤적이 예측 가능
    -> sim2real 안정성. 스폰이 yaw 회전만이라 시작 시 cube z축은 world z와 정렬 상태이며,
    이 항은 운반 중 기울임(tilt)만 벌한다 (yaw 회전은 자유 — 대칭 cube라 무의미).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ez = torch.zeros_like(obj.data.root_pos_w.torch)
    ez[:, 2] = 1.0
    from isaaclab.utils.math import quat_apply

    cube_z_w = quat_apply(obj.data.root_quat_w.torch, ez)
    upright = cube_z_w[:, 2].clamp(min=0.0)
    gate = object_is_lifted(env, minimal_height, object_cfg) * object_grasped(
        env, finger1_sensor_cfg, finger2_sensor_cfg, threshold
    )
    return upright * gate


def finger_table_impact_penalty(
    env: ManagerBasedRLEnv,
    finger1_sensor_cfg: SceneEntityCfg,
    finger2_sensor_cfg: SceneEntityCfg,
    threshold: float = 10.0,
    filter_idx: int = 1,
) -> torch.Tensor:
    """finger가 테이블을 threshold(N) 이상으로 때리면 finger당 1씩 카운트 (Run D).

    파지를 위해 finger가 상판 근처로 내려가 가볍게 스치는 것은 정당하므로 threshold를 높게
    (기본 10N) 잡아 "콩 하고 박는" 충격만 벌한다. Table filter는 finger 센서의 index 1.
    """
    count = torch.zeros(env.num_envs, device=env.device)
    for cfg in (finger1_sensor_cfg, finger2_sensor_cfg):
        sensor: ContactSensor = env.scene.sensors[cfg.name]
        f = torch.norm(sensor.data.force_matrix_w[:, :, filter_idx, :], dim=-1).amax(dim=-1)
        count += (f > threshold).float()
    return count


def object_is_lifted_and_grasped(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    finger1_sensor_cfg: SceneEntityCfg,
    finger2_sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward lifting only while the object is actually grasped (both finger pads in contact)."""
    gate = object_grasped(env, finger1_sensor_cfg, finger2_sensor_cfg, threshold)
    return object_is_lifted(env, minimal_height, object_cfg) * gate


class object_goal_distance_grasped(object_goal_distance):
    """Official goal-tracking reward gated on a maintained grasp."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        minimal_height: float,
        command_name: str,
        finger1_sensor_cfg: SceneEntityCfg,
        finger2_sensor_cfg: SceneEntityCfg,
        threshold: float = 0.5,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        success_threshold: float | None = None,
    ) -> torch.Tensor:
        goal_reward = super().__call__(
            env=env,
            std=std,
            minimal_height=minimal_height,
            command_name=command_name,
            robot_cfg=robot_cfg,
            object_cfg=object_cfg,
            success_threshold=success_threshold,
        )
        return goal_reward * object_grasped(env, finger1_sensor_cfg, finger2_sensor_cfg, threshold)


def object_velocity_near_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    dist_threshold: float = 0.1,
    ang_vel_scale: float = 0.1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalize object velocity only when near the goal — encourages holding still at the target.

    Gated on goal proximity so transport speed is not penalized; only the final "hold" phase is.
    Same philosophy as the angular-velocity damping in the in-hand manipulation envs.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3]
    )
    near_goal = torch.norm(des_pos_w - object.data.root_pos_w.torch, dim=1) < dist_threshold
    vel = torch.sum(torch.square(object.data.root_lin_vel_w.torch), dim=1)
    vel += ang_vel_scale * torch.sum(torch.square(object.data.root_ang_vel_w.torch), dim=1)
    return near_goal * vel
