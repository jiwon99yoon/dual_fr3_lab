"""Rewards for the cabinet book-insert task (plan 7-3), staged like the rule-based run.

The book terms are gated by the door opening (0 below ``gate_deg[0]``, 1 above ``gate_deg[1]``), so the book is only
paid for heading to the cabinet once the door is out of its way. Positions are in the robot base frame.
"""

from __future__ import annotations

import math

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import book_pose_b, contact_force_norms, door_angle_rad, handle_force_norm


def _door_gate(env, gate_deg: tuple[float, float]) -> torch.Tensor:
    lo, hi = (math.radians(d) for d in gate_deg)
    return ((door_angle_rad(env) - lo) / (hi - lo)).clamp(0.0, 1.0)


def _lane_error(pos: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    return torch.norm(pos[:, 1:] - goal[1:], dim=-1)


def door_open(env, goal_deg: float = 90.0) -> torch.Tensor:
    """Door opening progress, 1 at ``goal_deg`` and beyond."""
    return (door_angle_rad(env) / math.radians(goal_deg)).clamp(0.0, 1.0)


def book_approach(
    env, pre_pos_b: tuple[float, float, float], std: float = 0.3, gate_deg: tuple[float, float] = (60.0, 80.0)
) -> torch.Tensor:
    """Coarse: book centre toward the pre-insert point in front of the opening (wide tanh, the book starts 35-55 cm off)."""
    pos, _ = book_pose_b(env)
    d = torch.norm(pos - torch.tensor(pre_pos_b, device=env.device), dim=-1)
    return _door_gate(env, gate_deg) * (1.0 - torch.tanh(d / std))


def book_lane(
    env, goal_pos_b: tuple[float, float, float], std: float = 0.03, gate_deg: tuple[float, float] = (60.0, 80.0)
) -> torch.Tensor:
    """Fine: book centre on the insertion lane (y, z of the goal)."""
    pos, _ = book_pose_b(env)
    err = _lane_error(pos, torch.tensor(goal_pos_b, device=env.device))
    return _door_gate(env, gate_deg) * (1.0 - torch.tanh(err / std))


def book_depth(
    env,
    goal_pos_b: tuple[float, float, float],
    opening_x_b: float,
    start_depth: float = -0.15,
    lane_std: float = 0.03,
    gate_deg: tuple[float, float] = (60.0, 80.0),
) -> torch.Tensor:
    """Insertion progress from ``start_depth`` (in front of the opening) to the goal depth, scaled by lane alignment."""
    pos, _ = book_pose_b(env)
    goal = torch.tensor(goal_pos_b, device=env.device)
    depth = pos[:, 0] - opening_x_b
    progress = ((depth - start_depth) / (goal[0] - opening_x_b - start_depth)).clamp(0.0, 1.0)
    lane = 1.0 - torch.tanh(_lane_error(pos, goal) / lane_std)
    return _door_gate(env, gate_deg) * lane * progress


def insert_success(
    env,
    goal_pos_b: tuple[float, float, float],
    opening_x_b: float,
    min_depth: float = 0.09,
    lane_tol: float = 0.03,
    max_tilt_deg: float = 10.0,
) -> torch.Tensor:
    """Success (plan 7-3): book centre >= ``min_depth`` past the opening, y and z within ``lane_tol`` of the lane,
    tilt below ``max_tilt_deg``."""
    pos, quat = book_pose_b(env)
    goal = torch.tensor(goal_pos_b, device=env.device)
    tilt = 2.0 * torch.acos(quat[:, 3].clamp(max=1.0))
    ok = (pos[:, 0] - opening_x_b >= min_depth) & ((pos[:, 1:] - goal[1:]).abs() < lane_tol).all(-1)
    return (ok & (tilt < math.radians(max_tilt_deg))).float()


def contact_excess(env, sensor_names: list[str], threshold: float = 5.0) -> torch.Tensor:
    """Sum over the sensors of the contact force above ``threshold`` (N)."""
    return (contact_force_norms(env, sensor_names) - threshold).clamp(min=0.0).sum(-1)


def handle_force_excess(
    env,
    threshold: float = 30.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("cabinet_wrench"),
    body_name: str = "handle",
) -> torch.Tensor:
    """Handle force above ``threshold`` (N): pulling or twisting the handle harder than the real UMI grip holds."""
    return (handle_force_norm(env, sensor_cfg, body_name) - threshold).clamp(min=0.0)
