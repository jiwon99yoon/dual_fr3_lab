# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
    measured = asset.root_physx_view.get_dof_projected_joint_forces()
    return measured[:, asset_cfg.joint_ids]
