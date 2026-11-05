"""Observations for the cabinet book-insert task (plan 7-2).

Policy terms are what the real robot has without a camera: the door angle from the left-hand FK and the cabinet
calibration, the book pose from the right-hand FK and the grasp offset (both hands are attached). The critic also gets
privileged terms (contact forces, handle force, door velocity, DR values).
"""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, quat_mul, quat_inv


def _t(x):
    return x.torch if hasattr(x, "torch") else x


def door_angle_rad(env, cabinet_cfg: SceneEntityCfg = SceneEntityCfg("cabinet")) -> torch.Tensor:
    """Door opening angle [N] (rad, positive = open; the hinge joint opens negative)."""
    return -_t(env.scene[cabinet_cfg.name].data.joint_pos)[:, 0]


def book_pose_b(
    env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"), book_cfg: SceneEntityCfg = SceneEntityCfg("book")
) -> tuple[torch.Tensor, torch.Tensor]:
    """Book centre position [N, 3] and orientation [N, 4] (x, y, z, w; w >= 0) in the robot base frame."""
    robot, book = env.scene[robot_cfg.name], env.scene[book_cfg.name]
    root_p, root_q = _t(robot.data.root_pos_w), _t(robot.data.root_quat_w)
    pos = quat_apply_inverse(root_q, _t(book.data.root_pos_w) - root_p)
    quat = quat_mul(quat_inv(root_q), _t(book.data.root_quat_w))
    return pos, torch.where(quat[:, 3:4] < 0, -quat, quat)


def door_angle(env, cabinet_cfg: SceneEntityCfg = SceneEntityCfg("cabinet")) -> torch.Tensor:
    return door_angle_rad(env, cabinet_cfg)[:, None]


def door_velocity(env, cabinet_cfg: SceneEntityCfg = SceneEntityCfg("cabinet")) -> torch.Tensor:
    return -_t(env.scene[cabinet_cfg.name].data.joint_vel)[:, :1]


def book_pos_rel(
    env,
    target_pos_b: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    book_cfg: SceneEntityCfg = SceneEntityCfg("book"),
) -> torch.Tensor:
    """Book centre minus the (fixed) insert goal, base frame: x = depth to go, y/z = lane error."""
    pos, _ = book_pose_b(env, robot_cfg, book_cfg)
    return pos - torch.tensor(target_pos_b, device=env.device)


def book_quat(
    env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"), book_cfg: SceneEntityCfg = SceneEntityCfg("book")
) -> torch.Tensor:
    return book_pose_b(env, robot_cfg, book_cfg)[1]


def contact_force_norms(env, sensor_names: list[str]) -> torch.Tensor:
    """Net contact force [N, sensors] (N, largest body per sensor)."""
    return torch.stack(
        [torch.norm(_t(env.scene[name].data.net_forces_w), dim=-1).amax(-1) for name in sensor_names], dim=-1
    )


def contact_forces(env, sensor_names: list[str], scale: float = 0.01) -> torch.Tensor:
    return scale * contact_force_norms(env, sensor_names)


def handle_force_norm(
    env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("cabinet_wrench"), body_name: str = "handle"
) -> torch.Tensor:
    """Force [N] through the handle's joint to the door = what the left hand pulls/pushes on the handle with."""
    sensor = env.scene[sensor_cfg.name]
    cache = env.__dict__.setdefault("_cabinet_book_body_ids", {})
    if (sensor_cfg.name, body_name) not in cache:
        cache[(sensor_cfg.name, body_name)] = sensor.find_bodies(body_name)[0][0]
    return torch.norm(_t(sensor.data.force)[:, cache[(sensor_cfg.name, body_name)]], dim=-1)


def handle_force(
    env, scale: float = 0.01, sensor_cfg: SceneEntityCfg = SceneEntityCfg("cabinet_wrench"), body_name: str = "handle"
) -> torch.Tensor:
    return scale * handle_force_norm(env, sensor_cfg, body_name)[:, None]


def book_mass(env, book_cfg: SceneEntityCfg = SceneEntityCfg("book")) -> torch.Tensor:
    return _t(env.scene[book_cfg.name].data.body_mass)[:, :1].to(env.device)


def door_damping(env, cabinet_cfg: SceneEntityCfg = SceneEntityCfg("cabinet")) -> torch.Tensor:
    return _t(env.scene[cabinet_cfg.name].data.joint_damping)[:, :1].to(env.device)
