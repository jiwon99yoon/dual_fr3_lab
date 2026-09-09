#!/usr/bin/env python3
# Copyright (c) 2026, DYROS.
# SPDX-License-Identifier: BSD-3-Clause

"""Record open-loop PhysX torque-step responses for the Dual-FR3 Reach v2 asset.

This script deliberately bypasses the policy and the Reach position-PD controller.
It retains the Reach v2 asset, physics, gravity, armature, dry-friction, and viscous-
friction configuration, but replaces Kp/Kd with zero so the requested effort is an
open-loop plant input.  It resets the robot to the same initial state for every
trial, excites one joint at a time, and records the 1 kHz joint position, velocity,
acceleration estimate, and torque signals.  The resulting CSV is intended to be
replayed with the same initial state and torque sequence in MuJoCo.

By default, joints 1-4 receive 5 Nm and joints 5-7 receive 1 Nm.  The values are
repeated for the right arm, and both positive and negative directions are tested.
They are conservative identification inputs rather than identified FR3 limits.

Examples:

.. code-block:: bash

    # All 14 joints, positive and negative direct-torque steps (limiter disabled).
    python scripts/tools/record_reach_v2_torque_response.py --headless

    # Only left joint7 (action-order index 6), with the Reach v2 1000 Nm/s limiter.
    python scripts/tools/record_reach_v2_torque_response.py \
        --joint_ids 6 --torques 1.0 --torque_rate_limit 1000 --headless

    # Per-FR3-joint amplitudes, repeated for both arms.
    python scripts/tools/record_reach_v2_torque_response.py \
        --torques 4 4 3 3 1 1 0.5 --directions positive --headless
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Record 1 kHz PhysX torque-step responses for the Dual-FR3 Reach v2 asset."
)
parser.add_argument(
    "--joint_ids",
    type=int,
    nargs="+",
    default=None,
    help=(
        "0-based action-order joints to excite: left joint1..7 are 0..6 and "
        "right joint1..7 are 7..13. Default: all 14 joints."
    ),
)
parser.add_argument(
    "--torques",
    type=float,
    nargs="+",
    default=[5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0],
    metavar="NM",
    help=(
        "Positive torque amplitudes in Nm. Supply 1 value for every joint, 7 values "
        "to repeat per arm, or 14 action-order values."
    ),
)
parser.add_argument(
    "--directions",
    choices=("positive", "negative", "both"),
    default="both",
    help="Torque directions to test. Default: both.",
)
parser.add_argument("--settle_s", type=float, default=0.10, help="Zero-torque time before each pulse.")
parser.add_argument("--pulse_s", type=float, default=0.10, help="Torque-pulse duration.")
parser.add_argument("--recovery_s", type=float, default=0.20, help="Zero-torque time after each pulse.")
parser.add_argument(
    "--torque_rate_limit",
    type=float,
    default=0.0,
    help=(
        "Optional torque slew-rate limit in Nm/s. Zero applies a direct step, which is "
        "recommended for dynamics identification. Use 1000 to reproduce Reach v2."
    ),
)
parser.add_argument(
    "--max_displacement_rad",
    type=float,
    default=0.50,
    help="Stop a trial if any arm joint moves farther than this from its reset position.",
)
parser.add_argument(
    "--output",
    type=str,
    default="logs/torque_step/physx_reach_v2_torque_step.csv",
    help="Output CSV path. A JSON metadata file is written beside it.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _validate_cli() -> tuple[list[int], list[float]]:
    joint_ids = list(range(14)) if args_cli.joint_ids is None else args_cli.joint_ids
    if not joint_ids:
        parser.error("--joint_ids must not be empty")
    if len(set(joint_ids)) != len(joint_ids) or any(index < 0 or index >= 14 for index in joint_ids):
        parser.error("--joint_ids must contain unique values in [0, 13]")

    torque_values = args_cli.torques
    if len(torque_values) == 1:
        torque_by_joint = torque_values * 14
    elif len(torque_values) == 7:
        torque_by_joint = torque_values * 2
    elif len(torque_values) == 14:
        torque_by_joint = torque_values
    else:
        parser.error("--torques requires exactly 1, 7, or 14 values")
    if any(not math.isfinite(value) or value <= 0.0 for value in torque_by_joint):
        parser.error("--torques values must be finite and positive")

    for name in ("settle_s", "pulse_s", "recovery_s"):
        value = getattr(args_cli, name)
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name} must be finite and non-negative")
    if args_cli.pulse_s <= 0.0:
        parser.error("--pulse_s must be positive")
    if not math.isfinite(args_cli.torque_rate_limit) or args_cli.torque_rate_limit < 0.0:
        parser.error("--torque_rate_limit must be finite and non-negative")
    if not math.isfinite(args_cli.max_displacement_rad) or args_cli.max_displacement_rad <= 0.0:
        parser.error("--max_displacement_rad must be finite and positive")
    return joint_ids, torque_by_joint


SELECTED_JOINT_IDS, TORQUE_BY_ACTION_INDEX = _validate_cli()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""The remaining imports require the running Isaac Sim application."""

import csv
import json

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import Articulation

from dual_fr3_lab.robots.dual_fr3_plate_ee import DUAL_FR3_PLATE_EE_SIM2REAL_V2_CFG
from dual_fr3_lab.tasks.manager_based.reach.joint_pos_env_cfg import (
    DualFR3ReachSim2RealV2EnvCfg,
)


REACH_V2_ENV_CFG = DualFR3ReachSim2RealV2EnvCfg()
PHYSICS_DT = float(REACH_V2_ENV_CFG.sim.dt)


def _make_open_loop_actuator(source_cfg) -> IdealPDActuatorCfg:
    """Copy v2 joint/drive physics while removing only controller feedback gains."""

    return IdealPDActuatorCfg(
        joint_names_expr=source_cfg.joint_names_expr,
        effort_limit=source_cfg.effort_limit,
        velocity_limit=source_cfg.velocity_limit,
        effort_limit_sim=source_cfg.effort_limit_sim,
        velocity_limit_sim=source_cfg.velocity_limit_sim,
        stiffness=0.0,
        damping=0.0,
        armature=source_cfg.armature,
        friction=source_cfg.friction,
        dynamic_friction=source_cfg.dynamic_friction,
        viscous_friction=source_cfg.viscous_friction,
    )


def _ordered_arm_joints(robot: Articulation) -> tuple[list[int], list[str]]:
    """Return joint IDs/names in policy order: left 1-7, then right 1-7."""

    all_names = robot.data.joint_names
    left_ids = sorted(
        [index for index, name in enumerate(all_names) if name.startswith("left_fr3_joint")],
        key=lambda index: int(all_names[index].rsplit("joint", 1)[1]),
    )
    right_ids = sorted(
        [index for index, name in enumerate(all_names) if name.startswith("right_fr3_joint")],
        key=lambda index: int(all_names[index].rsplit("joint", 1)[1]),
    )
    if len(left_ids) != 7 or len(right_ids) != 7:
        raise RuntimeError(f"Expected seven joints per arm, got left={left_ids}, right={right_ids}")
    ids = left_ids + right_ids
    return ids, [all_names[index] for index in ids]


def _expected_action_order_values(training_actuator_cfgs: dict, field: str) -> list[float]:
    """Expand shoulder/forearm scalar cfg values to left-7/right-7 order."""

    shoulder_value = getattr(training_actuator_cfgs["arms_shoulder"], field)
    forearm_value = getattr(training_actuator_cfgs["arms_forearm"], field)
    if not isinstance(shoulder_value, (int, float)) or not isinstance(
        forearm_value, (int, float)
    ):
        raise TypeError(
            f"Torque-response validation requires scalar {field} values, got "
            f"shoulder={shoulder_value!r}, forearm={forearm_value!r}"
        )
    per_arm = [float(shoulder_value)] * 4 + [float(forearm_value)] * 3
    return per_arm * 2


def _validate_passive_joint_properties(
    robot: Articulation,
    arm_joint_ids: list[int],
    training_actuator_cfgs: dict,
) -> dict[str, list[float]]:
    """Fail fast if the direct-effort actuator lost any Reach v2 plant property."""

    actual_by_field = {
        "armature": robot.data.joint_armature.torch[0, arm_joint_ids],
        "friction": robot.data.joint_friction_coeff.torch[0, arm_joint_ids],
        "dynamic_friction": robot.data.joint_dynamic_friction_coeff.torch[0, arm_joint_ids],
        "viscous_friction": robot.data.joint_viscous_friction_coeff.torch[0, arm_joint_ids],
    }
    validated = {}
    for field, actual in actual_by_field.items():
        expected_values = _expected_action_order_values(training_actuator_cfgs, field)
        expected = torch.tensor(expected_values, device=actual.device, dtype=actual.dtype)
        if not torch.allclose(actual, expected, rtol=0.0, atol=1.0e-7):
            raise RuntimeError(
                f"Reach v2 {field} was not preserved by the measurement actuator: "
                f"expected={expected_values}, actual={actual.cpu().tolist()}"
            )
        validated[field] = actual.cpu().tolist()
    return validated


def _reset_robot(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    default_root_state: torch.Tensor,
    default_joint_pos: torch.Tensor,
    default_joint_vel: torch.Tensor,
    arm_joint_ids: list[int],
) -> torch.Tensor:
    """Restore an identical root/joint state and zero all actuator targets."""

    robot.reset()
    robot.write_root_link_pose_to_sim_index(root_pose=default_root_state[:, :7])
    robot.write_root_com_velocity_to_sim_index(root_velocity=default_root_state[:, 7:])
    robot.write_joint_position_to_sim_index(position=default_joint_pos)
    robot.write_joint_velocity_to_sim_index(velocity=default_joint_vel)

    zero_arm = torch.zeros((1, len(arm_joint_ids)), device=sim.device)
    robot.set_joint_position_target_index(
        target=default_joint_pos[:, arm_joint_ids], joint_ids=arm_joint_ids
    )
    robot.set_joint_velocity_target_index(target=zero_arm, joint_ids=arm_joint_ids)
    robot.set_joint_effort_target_index(target=zero_arm, joint_ids=arm_joint_ids)
    robot.write_data_to_sim()
    sim.forward()
    robot.update(PHYSICS_DT)
    return robot.data.joint_pos.torch[:, arm_joint_ids].clone()


def _phase(step: int, settle_steps: int, pulse_steps: int) -> str:
    if step < settle_steps:
        return "settle"
    if step < settle_steps + pulse_steps:
        return "pulse"
    return "recovery"


def main() -> None:
    output_path = Path(args_cli.output).expanduser().resolve()
    if output_path.suffix.lower() != ".csv":
        raise ValueError(f"--output must end in .csv: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(".json")

    # Start from the actual Reach v2 environment configuration instead of
    # reconstructing a similar-looking SimulationCfg in this script.
    sim_cfg = REACH_V2_ENV_CFG.sim.copy()
    sim_cfg.device = args_cli.device
    sim_cfg.render_interval = 1
    physics_cfg = sim_cfg.physics
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(2.8, 2.8, 2.3), target=(0.0, 0.0, 0.7))

    # Ground and lighting do not enter this response: the Husky root is fixed and
    # the arms move in free space.  Omitting them also avoids a remote ground-USD
    # dependency while preserving every robot/plate dynamics parameter.

    # Preserve every passive/drive property from the v2 actuator configuration.
    # Only Kp and Kd are set to zero so the experiment measures the open-loop
    # plant response to direct joint effort, without policy/PD feedback.
    robot_cfg = DUAL_FR3_PLATE_EE_SIM2REAL_V2_CFG.copy()
    robot_cfg.prim_path = "/World/Robot"
    training_actuator_cfgs = robot_cfg.actuators
    robot_cfg.actuators = {
        name: _make_open_loop_actuator(actuator_cfg)
        for name, actuator_cfg in training_actuator_cfgs.items()
    }
    robot = Articulation(robot_cfg)
    sim.reset()
    robot.reset()
    robot.update(PHYSICS_DT)

    arm_joint_ids, arm_joint_names = _ordered_arm_joints(robot)
    passive_joint_properties = _validate_passive_joint_properties(
        robot, arm_joint_ids, training_actuator_cfgs
    )
    default_root_state = robot.data.default_root_state.torch.clone()
    default_joint_pos = robot.data.default_joint_pos.torch.clone()
    default_joint_vel = robot.data.default_joint_vel.torch.clone()

    print("[config] Reach v2 plant properties verified")
    print(f"[config] physics dt:       {PHYSICS_DT:.6f} s")
    print(f"[config] armature:        {passive_joint_properties['armature']}")
    print(f"[config] static friction: {passive_joint_properties['friction']}")
    print(f"[config] dynamic friction:{passive_joint_properties['dynamic_friction']}")
    print(f"[config] viscous friction:{passive_joint_properties['viscous_friction']}")
    print("[config] measurement Kp/Kd: 0/0 (open-loop direct effort)")

    settle_steps = round(args_cli.settle_s / PHYSICS_DT)
    pulse_steps = round(args_cli.pulse_s / PHYSICS_DT)
    recovery_steps = round(args_cli.recovery_s / PHYSICS_DT)
    total_steps = settle_steps + pulse_steps + recovery_steps
    if pulse_steps < 1:
        raise ValueError("--pulse_s is shorter than one 1 ms physics step")
    if total_steps < 1:
        raise ValueError("The requested trial contains no physics steps")

    direction_signs = {
        "positive": (1.0,),
        "negative": (-1.0,),
        "both": (1.0, -1.0),
    }[args_cli.directions]

    scalar_columns = [
        "trial",
        "sample",
        "time_s",
        "phase",
        "excited_action_index",
        "excited_joint_name",
        "direction",
        "pulse_amplitude_nm",
        "max_displacement_rad",
        "safety_stop",
    ]
    vector_columns = []
    for prefix, unit in (
        ("q", "rad"),
        ("qd", "rad_s"),
        ("qdd_fd", "rad_s2"),
        ("tau_requested", "nm"),
        ("tau_commanded", "nm"),
        ("tau_applied", "nm"),
    ):
        vector_columns.extend(f"{prefix}_{unit}__{name}" for name in arm_joint_names)

    trial_count = 0
    safety_stop_count = 0
    interrupted = False
    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(scalar_columns + vector_columns)

        for action_index in SELECTED_JOINT_IDS:
            for direction in direction_signs:
                if not simulation_app.is_running():
                    interrupted = True
                    break

                initial_arm_q = _reset_robot(
                    sim,
                    robot,
                    default_root_state,
                    default_joint_pos,
                    default_joint_vel,
                    arm_joint_ids,
                )
                previous_qd = robot.data.joint_vel.torch[:, arm_joint_ids].clone()
                previous_command = torch.zeros((1, 14), device=sim.device)
                pulse_amplitude = direction * TORQUE_BY_ACTION_INDEX[action_index]
                trial_safety_stop = False

                print(
                    f"[trial {trial_count:02d}] {arm_joint_names[action_index]} "
                    f"torque={pulse_amplitude:+.3f} Nm"
                )

                for sample in range(total_steps):
                    phase = _phase(sample, settle_steps, pulse_steps)
                    requested = torch.zeros((1, 14), device=sim.device)
                    if phase == "pulse":
                        requested[0, action_index] = pulse_amplitude

                    if args_cli.torque_rate_limit > 0.0:
                        max_torque_step = args_cli.torque_rate_limit * PHYSICS_DT
                        commanded = previous_command + torch.clamp(
                            requested - previous_command,
                            min=-max_torque_step,
                            max=max_torque_step,
                        )
                    else:
                        commanded = requested
                    previous_command = commanded.clone()

                    robot.set_joint_effort_target_index(
                        target=commanded, joint_ids=arm_joint_ids
                    )
                    robot.write_data_to_sim()
                    sim.step(render=not args_cli.headless)
                    robot.update(PHYSICS_DT)

                    q = robot.data.joint_pos.torch[:, arm_joint_ids].clone()
                    qd = robot.data.joint_vel.torch[:, arm_joint_ids].clone()
                    qdd_fd = (qd - previous_qd) / PHYSICS_DT
                    previous_qd = qd
                    applied = robot.data.applied_torque.torch[:, arm_joint_ids].clone()
                    max_displacement = float(torch.max(torch.abs(q - initial_arm_q)).item())
                    finite = bool(
                        torch.isfinite(q).all().item()
                        and torch.isfinite(qd).all().item()
                        and torch.isfinite(applied).all().item()
                    )
                    trial_safety_stop = (
                        not finite or max_displacement > args_cli.max_displacement_rad
                    )

                    writer.writerow(
                        [
                            trial_count,
                            sample,
                            (sample + 1) * PHYSICS_DT,
                            phase,
                            action_index,
                            arm_joint_names[action_index],
                            "positive" if direction > 0.0 else "negative",
                            pulse_amplitude,
                            max_displacement,
                            int(trial_safety_stop),
                            *q[0].cpu().tolist(),
                            *qd[0].cpu().tolist(),
                            *qdd_fd[0].cpu().tolist(),
                            *requested[0].cpu().tolist(),
                            *commanded[0].cpu().tolist(),
                            *applied[0].cpu().tolist(),
                        ]
                    )

                    if trial_safety_stop:
                        safety_stop_count += 1
                        print(
                            f"  [safety stop] max displacement={max_displacement:.4f} rad, "
                            f"finite={finite}"
                        )
                        break
                    if not simulation_app.is_running():
                        interrupted = True
                        break

                zero_arm = torch.zeros((1, 14), device=sim.device)
                robot.set_joint_effort_target_index(target=zero_arm, joint_ids=arm_joint_ids)
                robot.write_data_to_sim()
                trial_count += 1
                if interrupted:
                    break
            if interrupted:
                break

    metadata = {
        "schema_version": 2,
        "simulator": "Isaac Sim / PhysX",
        "purpose": "open-loop joint torque response for PhysX-MuJoCo comparison",
        "source_environment_cfg": (
            "dual_fr3_lab.tasks.manager_based.reach.joint_pos_env_cfg:"
            "DualFR3ReachSim2RealV2EnvCfg"
        ),
        "source_robot_cfg": (
            "dual_fr3_lab.robots.dual_fr3_plate_ee:"
            "DUAL_FR3_PLATE_EE_SIM2REAL_V2_CFG"
        ),
        "asset": str(robot_cfg.spawn.usd_path),
        "physics_dt_s": PHYSICS_DT,
        "physics_rate_hz": 1.0 / PHYSICS_DT,
        "world_gravity_m_s2": list(sim_cfg.gravity),
        "robot_disable_gravity": bool(robot_cfg.spawn.rigid_props.disable_gravity),
        "physx_solver": "TGS" if physics_cfg.solver_type == 1 else "PGS",
        "enable_external_forces_every_iteration": bool(
            physics_cfg.enable_external_forces_every_iteration
        ),
        "solver_position_iterations": int(
            robot_cfg.spawn.articulation_props.solver_position_iteration_count
        ),
        "solver_velocity_iterations": int(
            robot_cfg.spawn.articulation_props.solver_velocity_iteration_count
        ),
        "measurement_controller": "zero-gain IdealPDActuator; direct joint effort input",
        "measurement_kp": 0.0,
        "measurement_kd": 0.0,
        "reach_v2_training_kp": {
            name: float(cfg.stiffness) for name, cfg in training_actuator_cfgs.items()
        },
        "reach_v2_training_kd": {
            name: float(cfg.damping) for name, cfg in training_actuator_cfgs.items()
        },
        "reach_v2_training_torque_rate_limit_nm_s": {
            name: float(cfg.torque_rate_limit)
            for name, cfg in training_actuator_cfgs.items()
        },
        "ground_plane": False,
        "ground_plane_omission_reason": (
            "fixed-base free-space response; no robot-ground contact"
        ),
        "torque_rate_limit_nm_s": float(args_cli.torque_rate_limit),
        "joint_order": arm_joint_names,
        "articulation_joint_ids": arm_joint_ids,
        "selected_action_indices": SELECTED_JOINT_IDS,
        "torque_amplitude_nm_by_action_index": TORQUE_BY_ACTION_INDEX,
        "directions": args_cli.directions,
        "settle_s": settle_steps * PHYSICS_DT,
        "pulse_s": pulse_steps * PHYSICS_DT,
        "recovery_s": recovery_steps * PHYSICS_DT,
        "max_displacement_rad": float(args_cli.max_displacement_rad),
        "default_arm_joint_position_rad": default_joint_pos[0, arm_joint_ids].cpu().tolist(),
        "effort_limit_nm_by_joint": _expected_action_order_values(
            training_actuator_cfgs, "effort_limit"
        ),
        "effort_limit_sim_nm_by_joint": _expected_action_order_values(
            training_actuator_cfgs, "effort_limit_sim"
        ),
        "armature_kg_m2_by_joint": passive_joint_properties["armature"],
        "static_friction_effort_nm_by_joint": passive_joint_properties["friction"],
        "dynamic_friction_effort_nm_by_joint": passive_joint_properties[
            "dynamic_friction"
        ],
        "viscous_friction_nm_s_per_rad_by_joint": passive_joint_properties[
            "viscous_friction"
        ],
        "trials_completed": trial_count,
        "safety_stops": safety_stop_count,
        "interrupted": interrupted,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"[done] CSV:      {output_path}")
    print(f"[done] metadata: {metadata_path}")
    print(f"[done] trials={trial_count}, safety_stops={safety_stop_count}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
