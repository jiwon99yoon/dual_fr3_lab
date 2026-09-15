# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_orientation_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """cube orientation (robot root frame 기준 quat, w>=0 부호 고정) (num_envs, 4).

    Run D: 스폰 yaw DR을 넣으면 정책이 cube 회전을 관측할 수 있어야 파지 정합과
    운반 자세(upright) 제어가 가능해짐 (plate P5 교훈 — DR을 넣으면 관측도 줘야 함).
    실물은 6D pose 인식(ArUco/FoundationPose 등)으로 획득 가능한 신호.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    _, quat = subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        obj.data.root_pos_w.torch,
        obj.data.root_quat_w.torch,
    )
    sign = torch.where(quat[:, 0:1] >= 0.0, 1.0, -1.0)
    return quat * sign


def joint_measured_torques(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """관절축에 투영된 측정 joint 토크 — FR3 내장 관절 토크 센서의 sim 대응물.

    실물 FR3는 중력보상 위에서 외란 토크 추정치(tau_ext)를 제공한다. 로봇 링크의 중력을 끈
    (disable_gravity=True) 환경에서는 이 측정값에 링크 자중이 빠져 있어 실물 tau_ext와
    성격이 맞는다. 접촉/payload/동역학 성분이 담기므로 contact-aware 정책 학습용.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    measured = wp.to_torch(asset.root_physx_view.get_dof_projected_joint_forces())
    return measured[:, asset_cfg.joint_ids]
