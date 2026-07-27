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
) -> torch.Tensor:
    """Binary gate: 1.0 if both finger pads are in contact with the filtered object.

    Each contact sensor must match a single finger body and set ``filter_prim_paths_expr`` to the
    object, so ``force_matrix_w`` reports finger-object forces only (table/self contacts excluded).
    PhysX requires one filter target per sensor body, hence one sensor per finger. Requiring *both*
    pads above the force threshold accepts a true grasp and rejects balancing the object on the
    back of the hand.
    """
    finger_forces = []
    for cfg in (finger1_sensor_cfg, finger2_sensor_cfg):
        sensor: ContactSensor = env.scene.sensors[cfg.name]
        # (num_envs, 1, 1, 3) -> per-env force magnitude against the object
        finger_forces.append(torch.norm(sensor.data.force_matrix_w, dim=-1).amax(dim=(-1, -2)))
    return (torch.minimum(*finger_forces) > threshold).float()


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


def object_goal_distance_grasped(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    finger1_sensor_cfg: SceneEntityCfg,
    finger2_sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Goal-tracking reward gated on a maintained grasp, so carrying without grasping earns nothing."""
    goal_reward = object_goal_distance(env, std, minimal_height, command_name, robot_cfg, object_cfg)
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
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    near_goal = torch.norm(des_pos_w - object.data.root_pos_w, dim=1) < dist_threshold
    vel = torch.sum(torch.square(object.data.root_lin_vel_w), dim=1)
    vel += ang_vel_scale * torch.sum(torch.square(object.data.root_ang_vel_w), dim=1)
    return near_goal * vel
