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


def _upright_gate(env, gate_deg: tuple[float, float] | None) -> torch.Tensor | float:
    """1 below ``gate_deg[0]`` of book tilt, 0 above ``gate_deg[1]``; None = no gate (v1)."""
    if gate_deg is None:
        return 1.0
    lo, hi = (math.radians(d) for d in gate_deg)
    return ((hi - _book_tilt(env)) / (hi - lo)).clamp(0.0, 1.0)


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
    env,
    goal_pos_b: tuple[float, float, float],
    std: float = 0.03,
    gate_deg: tuple[float, float] = (60.0, 80.0),
    tilt_gate_deg: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Fine: book centre on the insertion lane (y, z of the goal), optionally only while the book stays upright."""
    pos, _ = book_pose_b(env)
    err = _lane_error(pos, torch.tensor(goal_pos_b, device=env.device))
    return _door_gate(env, gate_deg) * _upright_gate(env, tilt_gate_deg) * (1.0 - torch.tanh(err / std))


def book_depth(
    env,
    goal_pos_b: tuple[float, float, float],
    opening_x_b: float,
    start_depth: float = -0.15,
    lane_std: float = 0.03,
    gate_deg: tuple[float, float] = (60.0, 80.0),
    overshoot: float | None = None,
    tilt_gate_deg: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Insertion progress from ``start_depth`` (in front of the opening) to the goal depth, scaled by lane alignment.

    ``overshoot``: past the goal depth the reward falls linearly to 0 over this distance (None: stays 1).
    ``tilt_gate_deg``: also scaled by the book staying upright (``_upright_gate``)."""
    pos, _ = book_pose_b(env)
    goal = torch.tensor(goal_pos_b, device=env.device)
    depth = pos[:, 0] - opening_x_b
    goal_depth = goal[0] - opening_x_b
    progress = ((depth - start_depth) / (goal_depth - start_depth)).clamp(0.0, 1.0)
    if overshoot is not None:
        progress = progress * (1.0 - (depth - goal_depth) / overshoot).clamp(0.0, 1.0)
    lane = 1.0 - torch.tanh(_lane_error(pos, goal) / lane_std)
    return _door_gate(env, gate_deg) * _upright_gate(env, tilt_gate_deg) * lane * progress


def book_orientation(env, power: int = 8) -> torch.Tensor:
    """Dense alignment with the upright insertion pose (identity in the robot base frame)."""
    _, quat = book_pose_b(env)
    # book_pose_b normalizes the sign so w >= 0. Raising cos(angle / 2) to a power
    # keeps a smooth signal while making a 20-30 degree wrist-induced tilt costly.
    return quat[:, 3].clamp(0.0, 1.0).pow(power)


def _book_tilt(env) -> torch.Tensor:
    _, quat = book_pose_b(env)
    return 2.0 * torch.acos(quat[:, 3].clamp(max=1.0))


def book_upright(env, std_deg: float = 10.0) -> torch.Tensor:
    """exp(-(tilt / std)^2): unlike cos(tilt / 2)^8 (0.97 at 10 deg) it still pays for the last few degrees."""
    return torch.exp(-(_book_tilt(env) / math.radians(std_deg)).square())


def book_upright_tanh(env, std: float = 0.3) -> torch.Tensor:
    """1 - tanh(tilt / std) (Upright task, std in rad)."""
    return 1.0 - torch.tanh(_book_tilt(env) / std)


def book_tilt_excess(env, threshold_deg: float = 15.0) -> torch.Tensor:
    """Tilt above ``threshold_deg`` (rad), at any time (v2 tilted the book ~30 deg while carrying it)."""
    return (_book_tilt(env) - math.radians(threshold_deg)).clamp(min=0.0)


def insert_success(
    env,
    goal_pos_b: tuple[float, float, float],
    opening_x_b: float,
    min_depth: float = 0.09,
    lane_tol: float = 0.03,
    max_tilt_deg: float = 10.0,
    max_depth: float | None = None,
    min_door_deg: float | None = None,
) -> torch.Tensor:
    """Success (plan 7-3): book centre >= ``min_depth`` past the opening, y and z within ``lane_tol`` of the lane,
    tilt below ``max_tilt_deg``; optionally book centre <= ``max_depth`` past the opening and door >= ``min_door_deg``."""
    pos, _ = book_pose_b(env)
    goal = torch.tensor(goal_pos_b, device=env.device)
    depth = pos[:, 0] - opening_x_b
    ok = (depth >= min_depth) & ((pos[:, 1:] - goal[1:]).abs() < lane_tol).all(-1)
    ok &= _book_tilt(env) < math.radians(max_tilt_deg)
    if max_depth is not None:
        ok &= depth <= max_depth
    if min_door_deg is not None:
        ok &= door_angle_rad(env) >= math.radians(min_door_deg)
    return ok.float()


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
