#!/usr/bin/env python
"""Deterministic dual-FR3 Isaac Lab system-identification runner.

This is deliberately *not* an RL environment rollout.  It reuses the same
Dual-FR3 plate asset and PhysX plant configuration used by the Reach task,
while excluding policy inference, reward/goal logic, observation noise, and
actuator domain randomization.

The learning environment retains its implicit PhysX PD drive.  This runner
intentionally disables that drive and applies the real controller's
measured-relative, torque-limited and torque-rate-limited PD command in
Python.  That keeps controller-path identification separate from training
while retaining the same USD, gravity treatment, armature, dry friction, and
viscous friction plant.

The generated CSV preserves the real/MuJoCo SysID probe fields and adds the
post-slew ``tau_applied_cmd_nm`` channel:

    q0 hold -> one-joint profile -> q0 return + settle -> next joint

All CSV joint columns use the physical/ROS logical order
``left joint1..7, right joint1..7``.  Isaac's articulation order is resolved
by joint name at runtime and is never assumed.

Examples (after this script is added)::

    # Visual smoke test: only left joint 1
    ~/anaconda3/envs/env_isaaclab/bin/python \
      scripts/sim2real/run_dual_fr3_sysid.py \
      --profile ramp --target-joint 0 --amplitude-rad 0.05

    # Calibration set equivalent to the 0.25-rad real/MuJoCo full sweep
    ~/anaconda3/envs/env_isaaclab/bin/python \
      scripts/sim2real/run_dual_fr3_sysid.py --headless \
      --sweep --profile ramp --amplitude-rad 0.25 \
      --amplitude-wrist-rad 0.25 --video --video-fps 30

The default physics timestep is 1 ms so the custom controller matches the
real 1-kHz effort path while holding its reference at 100 Hz.  Passing
``--physics-dt 0.01`` is a deliberate training-timestep sensitivity test; it
does not change the Reach training environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from isaaclab.app import AppLauncher


LOGICAL_JOINT_NAMES = [
    f"{side}_fr3_joint{joint}"
    for side in ("left", "right")
    for joint in range(1, 8)
]
PROFILE_TYPES = ("ramp", "step", "pulse", "hold_zero")


parser = argparse.ArgumentParser(
    description="Run a deterministic q0-return SysID sweep in Isaac Lab."
)
parser.add_argument("--profile", choices=PROFILE_TYPES, default="ramp")
parser.add_argument(
    "--sweep",
    action="store_true",
    help="Run selected joints sequentially. Otherwise only --target-joint runs.",
)
parser.add_argument(
    "--sweep-joints",
    type=int,
    nargs="+",
    default=None,
    metavar="INDEX",
    help="Logical zero-based joints for --sweep (default: all 0..13).",
)
parser.add_argument(
    "--target-joint",
    type=int,
    default=0,
    help="Logical zero-based joint for a single trial (default: 0 = left J1).",
)
parser.add_argument("--amplitude-rad", type=float, default=0.25, help="J1--J4 amplitude [rad].")
parser.add_argument(
    "--amplitude-wrist-rad",
    type=float,
    default=0.25,
    help="J5--J7 amplitude [rad].",
)
parser.add_argument("--t-start", type=float, default=1.0, help="q0 pre-hold duration [s].")
parser.add_argument("--ramp-time", type=float, default=1.0, help="Ramp duration [s].")
parser.add_argument("--hold-time", type=float, default=2.0, help="Profile plateau/step hold [s].")
parser.add_argument("--pulse-width", type=float, default=0.5, help="Pulse hold duration [s].")
parser.add_argument(
    "--initial-settle",
    type=float,
    default=0.75,
    help="Required stationary duration before live q0 capture [s].",
)
parser.add_argument(
    "--home-settle",
    type=float,
    default=0.75,
    help="Required q0 position/velocity settling before the next trial [s].",
)
parser.add_argument("--home-position-tolerance", type=float, default=0.015, help="q0 error threshold [rad].")
parser.add_argument(
    "--home-velocity-tolerance",
    type=float,
    default=0.02,
    help="q0 velocity threshold [rad/s].",
)
parser.add_argument(
    "--home-timeout",
    type=float,
    default=45.0,
    help="Maximum q0-return wait per trial after profile completion [s].",
)
parser.add_argument(
    "--command-rate-hz",
    type=float,
    default=100.0,
    help="Reference update rate [Hz]; q_ref is held between updates.",
)
parser.add_argument(
    "--physics-dt",
    type=float,
    default=0.001,
    help="Physics timestep [s]. Default 0.001 matches the real 1-kHz effort controller.",
)
parser.add_argument(
    "--max-target-delta",
    type=float,
    default=0.30,
    help="Mirror the real PolicyControl target-delta safety band [rad].",
)
parser.add_argument("--joint-limit-margin", type=float, default=0.02, help="Joint-limit safety margin [rad].")
parser.add_argument(
    "--robot-gravity",
    choices=("reach", "enabled", "disabled"),
    default="reach",
    help=(
        "'reach' preserves DUAL_FR3_PLATE_EE_CFG (currently disabled, modelling "
        "FR3 link gravity compensation). Other modes are explicit A/B tests."
    ),
)
parser.add_argument(
    "--ref-dir",
    type=Path,
    default=None,
    help=(
        "Optional real/MuJoCo SysID CSV directory. For every selected joint, "
        "reuse q0_<joint> values from the newest matching profile/amplitude CSV."
    ),
)
parser.add_argument(
    "--output-root",
    type=Path,
    default=Path("logs/sysid/isaaclab"),
    help="Directory below which one timestamped run directory is created.",
)
parser.add_argument("--run-name", default="", help="Optional deterministic output directory name.")
parser.add_argument("--video", action="store_true", help="Record the full SysID sweep to <run>/sysid.mp4.")
parser.add_argument("--video-fps", type=float, default=30.0, help="Output MP4 frame rate (default: 30).")
parser.add_argument("--video-width", type=int, default=640, help="Video frame width [px].")
parser.add_argument("--video-height", type=int, default=480, help="Video frame height [px].")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# A headless USD camera still needs Isaac Sim's camera/render extensions.
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Isaac imports must remain below AppLauncher construction.
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from dual_fr3_lab.robots.dual_fr3 import (
    FR3_ARM_KD_NM_S_PER_RAD,
    FR3_ARM_KP_NM_PER_RAD,
    FR3_ARMATURE_KG_M2,
    FR3_DRY_FRICTION_NM,
    FR3_FOREARM_EFFORT_LIMIT_NM,
    FR3_SHOULDER_EFFORT_LIMIT_NM,
    FR3_TORQUE_RATE_LIMIT_NM_S,
    FR3_VISCOUS_FRICTION_NM_S_PER_RAD,
)
from dual_fr3_lab.robots.dual_fr3_plate_ee import DUAL_FR3_PLATE_EE_CFG


@configclass
class DualFR3SysIdSceneCfg(InteractiveSceneCfg):
    """One deterministic copy of the same plate-EE robot used by Reach."""

    robot = DUAL_FR3_PLATE_EE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


@dataclass(frozen=True)
class TrialResult:
    logical_joint: int
    name: str
    csv_path: Path
    samples: int
    duration_s: float
    max_home_error_rad: float
    max_home_velocity_rad_s: float


@dataclass(frozen=True)
class ControllerSample:
    """Controller values calculated before one PhysX step, in ROS joint order."""

    joint_pos_rad: np.ndarray
    joint_vel_rad_s: np.ndarray
    computed_effort_nm: np.ndarray
    applied_effort_nm: np.ndarray


class RealPathTorqueController:
    """Replica of the real measured-relative FR3 effort-control path.

    The runner holds an absolute q-reference at ``command_rate_hz``.  At every
    physics step it recomputes the measured-relative offset, applies the
    real-side PD law, clips joint torque, then limits its change from the
    previously applied torque.  The implicit PhysX drives are disabled before
    this controller is created, so there is exactly one controller in the
    rollout.
    """

    def __init__(self, robot, sim_joint_ids: list[int], physics_dt: float):
        device = robot.data.joint_pos.device
        self._joint_ids = torch.tensor(sim_joint_ids, device=device, dtype=torch.long)
        self._kp = torch.full((1, len(sim_joint_ids)), FR3_ARM_KP_NM_PER_RAD, device=device)
        self._kd = torch.full((1, len(sim_joint_ids)), FR3_ARM_KD_NM_S_PER_RAD, device=device)
        per_arm_limits = [
            FR3_SHOULDER_EFFORT_LIMIT_NM,
            FR3_SHOULDER_EFFORT_LIMIT_NM,
            FR3_SHOULDER_EFFORT_LIMIT_NM,
            FR3_SHOULDER_EFFORT_LIMIT_NM,
            FR3_FOREARM_EFFORT_LIMIT_NM,
            FR3_FOREARM_EFFORT_LIMIT_NM,
            FR3_FOREARM_EFFORT_LIMIT_NM,
        ]
        self._effort_limit = torch.tensor(per_arm_limits * 2, device=device).unsqueeze(0)
        self._max_delta = FR3_TORQUE_RATE_LIMIT_NM_S * physics_dt
        self._previous_effort = torch.zeros_like(self._effort_limit)

    def reset(self) -> None:
        self._previous_effort.zero_()

    def apply(self, robot, reference: torch.Tensor) -> ControllerSample:
        q = robot.data.joint_pos[:, self._joint_ids]
        qd = robot.data.joint_vel[:, self._joint_ids]
        q_ref = reference[:, self._joint_ids]
        computed = self._kp * (q_ref - q) - self._kd * qd
        computed = torch.clamp(computed, min=-self._effort_limit, max=self._effort_limit)
        applied = self._previous_effort + torch.clamp(
            computed - self._previous_effort,
            min=-self._max_delta,
            max=self._max_delta,
        )
        self._previous_effort.copy_(applied)

        full_effort = torch.zeros_like(robot.data.joint_pos)
        full_effort[:, self._joint_ids] = applied
        robot.set_joint_effort_target(full_effort)
        return ControllerSample(
            joint_pos_rad=q[0].detach().cpu().numpy().astype(np.float64, copy=True),
            joint_vel_rad_s=qd[0].detach().cpu().numpy().astype(np.float64, copy=True),
            computed_effort_nm=computed[0].detach().cpu().numpy().astype(np.float64, copy=True),
            applied_effort_nm=applied[0].detach().cpu().numpy().astype(np.float64, copy=True),
        )


class VideoRecorder:
    """Stream the fixed world-camera RGB frames directly to an MP4 file."""

    def __init__(self, camera, output_path: Path, physics_dt: float):
        import cv2

        self._cv2 = cv2
        self._camera = camera
        self.output_path = output_path
        self._physics_dt = physics_dt
        self._frame_period = 1.0 / args_cli.video_fps
        self._next_frame_s = 0.0
        self._elapsed_s = 0.0
        self.frame_count = 0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(output_path), fourcc, args_cli.video_fps, (args_cli.video_width, args_cli.video_height)
        )
        if not self._writer.isOpened():
            raise RuntimeError(
                f"Could not open MP4 encoder for {output_path}. "
                "OpenCV must provide the mp4v VideoWriter codec."
            )

    def record_after_step(self) -> None:
        """Capture at the requested output FPS while simulation may run faster."""
        self._elapsed_s += self._physics_dt
        if self._elapsed_s + 1.0e-12 < self._next_frame_s:
            return
        rgb = self._camera.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.shape != (args_cli.video_height, args_cli.video_width, 3):
            raise RuntimeError(
                "Unexpected RGB camera frame shape: "
                f"{rgb.shape}; expected {(args_cli.video_height, args_cli.video_width, 3)}"
            )
        # Isaac Lab camera output is RGB; OpenCV VideoWriter expects BGR.
        frame_bgr = self._cv2.cvtColor(np.ascontiguousarray(rgb), self._cv2.COLOR_RGB2BGR)
        self._writer.write(frame_bgr)
        self.frame_count += 1
        while self._next_frame_s <= self._elapsed_s + 1.0e-12:
            self._next_frame_s += self._frame_period

    def close(self) -> None:
        self._writer.release()


def _finite_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative; got {value}")


def _validate_args() -> None:
    for name in (
        "amplitude_rad",
        "amplitude_wrist_rad",
        "t_start",
        "ramp_time",
        "hold_time",
        "pulse_width",
        "initial_settle",
        "home_settle",
        "home_position_tolerance",
        "home_velocity_tolerance",
        "home_timeout",
        "joint_limit_margin",
    ):
        _finite_nonnegative(name, float(getattr(args_cli, name)))
    if not math.isfinite(args_cli.physics_dt) or args_cli.physics_dt <= 0.0:
        raise ValueError(f"physics_dt must be positive; got {args_cli.physics_dt}")
    if not math.isfinite(args_cli.command_rate_hz) or args_cli.command_rate_hz <= 0.0:
        raise ValueError(f"command_rate_hz must be positive; got {args_cli.command_rate_hz}")
    if not math.isfinite(args_cli.max_target_delta) or args_cli.max_target_delta <= 0.0:
        raise ValueError(f"max_target_delta must be positive; got {args_cli.max_target_delta}")
    for name, amplitude in (
        ("amplitude_rad", args_cli.amplitude_rad),
        ("amplitude_wrist_rad", args_cli.amplitude_wrist_rad),
    ):
        if abs(amplitude) > args_cli.max_target_delta:
            raise ValueError(
            f"|{name}| must be <= max_target_delta ({args_cli.max_target_delta}); got {amplitude}"
            )

    if args_cli.video:
        if not math.isfinite(args_cli.video_fps) or args_cli.video_fps <= 0.0:
            raise ValueError(f"video_fps must be positive; got {args_cli.video_fps}")
        if args_cli.video_width <= 0 or args_cli.video_height <= 0:
            raise ValueError(
                f"video dimensions must be positive; got {args_cli.video_width}x{args_cli.video_height}"
            )

    physics_hz = 1.0 / args_cli.physics_dt
    command_steps = physics_hz / args_cli.command_rate_hz
    if command_steps < 1.0 - 1.0e-9 or not math.isclose(command_steps, round(command_steps), abs_tol=1.0e-8):
        raise ValueError(
            "command_rate_hz must divide the physics rate exactly and be no higher than it: "
            f"physics={physics_hz:.6g} Hz, command={args_cli.command_rate_hz:.6g} Hz"
        )


def _logical_joint_indices() -> list[int]:
    if args_cli.sweep:
        indices = list(range(14)) if args_cli.sweep_joints is None else list(args_cli.sweep_joints)
    else:
        indices = [args_cli.target_joint]
    if not indices:
        raise ValueError("At least one joint must be selected")
    if len(set(indices)) != len(indices):
        raise ValueError(f"Duplicate logical joint index in selection: {indices}")
    invalid = [index for index in indices if index < 0 or index >= len(LOGICAL_JOINT_NAMES)]
    if invalid:
        raise ValueError(f"Joint index must be in [0, 13]; got {invalid}")
    return indices


def _profile_offset(profile_t: float, amplitude: float) -> tuple[float, str]:
    """Return active-joint q0-relative offset and phase at a scheduled time."""
    if profile_t < args_cli.t_start:
        return 0.0, "q0_prehold"
    t = profile_t - args_cli.t_start

    if args_cli.profile == "hold_zero":
        return 0.0, "q0_hold"
    if args_cli.profile == "step":
        if t <= args_cli.hold_time:
            return amplitude, "step_up"
        return 0.0, "q0_stabilize"
    if args_cli.profile == "pulse":
        if t <= args_cli.pulse_width:
            return amplitude, "pulse_up"
        return 0.0, "q0_stabilize"

    # ramp: q0 -> amplitude -> q0.  A zero duration is intentionally an
    # instantaneous transition, just as the ROS probe treats a tiny ramp as a step.
    ramp = max(args_cli.ramp_time, 1.0e-9)
    if t <= ramp:
        return amplitude * (t / ramp), "ramp_up"
    if t <= ramp + args_cli.hold_time:
        return amplitude, "hold"
    if t <= 2.0 * ramp + args_cli.hold_time:
        return amplitude * (1.0 - (t - ramp - args_cli.hold_time) / ramp), "ramp_back"
    return 0.0, "q0_stabilize"


def _profile_end_s() -> float:
    if args_cli.profile == "ramp":
        return args_cli.t_start + 2.0 * max(args_cli.ramp_time, 1.0e-9) + args_cli.hold_time
    if args_cli.profile == "step":
        return args_cli.t_start + args_cli.hold_time
    if args_cli.profile == "pulse":
        return args_cli.t_start + args_cli.pulse_width
    return args_cli.t_start + args_cli.hold_time


def _amplitude_for(logical_joint: int) -> float:
    return args_cli.amplitude_wrist_rad if logical_joint % 7 >= 4 else args_cli.amplitude_rad


def _load_reference_home(logical_joint: int) -> tuple[np.ndarray, Path] | None:
    """Load the exact 14-D q0 target from the matching real/MuJoCo CSV.

    A trial must not silently borrow a q0 from a different profile or
    amplitude: that would make its controller command and initial condition
    incomparable with the reference trial.  The selected source path is later
    persisted in run_metadata.json.
    """
    if args_cli.ref_dir is None:
        return None

    ref_dir = args_cli.ref_dir.expanduser()
    if not ref_dir.is_dir():
        raise FileNotFoundError(f"--ref-dir is not a directory: {ref_dir}")
    name = LOGICAL_JOINT_NAMES[logical_joint]
    amplitude = _amplitude_for(logical_joint)
    matches = sorted(
        ref_dir.glob(f"{name}_{args_cli.profile}_amp{amplitude:.3f}_*.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(
            "No exact reference CSV for "
            f"joint={name}, profile={args_cli.profile}, amplitude={amplitude:.3f} in {ref_dir}."
        )

    source = matches[-1]
    with source.open(newline="") as file:
        row = next(csv.DictReader(file), None)
    if row is None:
        raise ValueError(f"Reference CSV has no data rows: {source}")
    missing = [f"q0_{joint}" for joint in LOGICAL_JOINT_NAMES if f"q0_{joint}" not in row]
    if missing:
        raise ValueError(f"Reference CSV lacks q0 columns: {missing}; source={source}")
    try:
        home = np.array([float(row[f"q0_{joint}"]) for joint in LOGICAL_JOINT_NAMES], dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"Reference CSV has non-numeric q0 values: {source}") from error
    if not np.all(np.isfinite(home)):
        raise ValueError(f"Reference CSV has non-finite q0 values: {source}")
    return home, source


def _home_target_from_canonical(robot, sim_joint_ids: list[int], canonical_home: np.ndarray) -> torch.Tensor:
    """Create a full-articulation target while replacing only the 14 arm DOFs."""
    home = robot.data.default_joint_pos.clone()
    home[0, sim_joint_ids] = torch.as_tensor(canonical_home, device=home.device, dtype=home.dtype)
    return home


def _canonical_values(value: torch.Tensor, sim_joint_ids: list[int]) -> np.ndarray:
    return value[0, sim_joint_ids].detach().cpu().numpy().astype(np.float64, copy=True)


def _torque_tensor(robot) -> tuple[torch.Tensor, str]:
    """Get the simulator actuator torque, with an explicit semantic label."""
    torque = getattr(robot.data, "applied_torque", None)
    if torque is not None:
        return torque, "ArticulationData.applied_torque"
    # Kept only for Isaac Lab version compatibility.  The metadata makes the
    # source explicit so it is never silently compared as a hardware sensor torque.
    torque = getattr(robot.data, "computed_torque", None)
    if torque is not None:
        return torque, "ArticulationData.computed_torque"
    raise RuntimeError(
        "This Isaac Lab version exposes neither robot.data.applied_torque nor "
        "robot.data.computed_torque; add the version-specific actuator torque accessor."
    )


def _read_plant_properties(robot, sim_joint_ids: list[int]) -> dict[str, np.ndarray]:
    """Read the values PhysX actually received, in canonical ROS joint order."""
    armature = robot.root_physx_view.get_dof_armatures().detach().cpu().numpy()[0, sim_joint_ids]
    friction = robot.root_physx_view.get_dof_friction_properties().detach().cpu().numpy()[0, sim_joint_ids]
    if friction.ndim != 2 or friction.shape[1] < 3:
        raise RuntimeError(
            "Expected Isaac Sim 5.0+ friction properties with static, dynamic, and viscous columns; "
            f"got shape {friction.shape}."
        )
    return {
        "armature_kg_m2": armature.astype(np.float64, copy=True),
        "static_friction_nm": friction[:, 0].astype(np.float64, copy=True),
        "dynamic_friction_nm": friction[:, 1].astype(np.float64, copy=True),
        "viscous_friction_nm_s_per_rad": friction[:, 2].astype(np.float64, copy=True),
    }


def _verify_identified_plant(plant: dict[str, np.ndarray]) -> None:
    """Fail fast instead of silently running SysID with USD default dynamics."""
    expected = {
        "armature_kg_m2": FR3_ARMATURE_KG_M2,
        "static_friction_nm": FR3_DRY_FRICTION_NM,
        "viscous_friction_nm_s_per_rad": FR3_VISCOUS_FRICTION_NM_S_PER_RAD,
    }
    for name, value in expected.items():
        if not np.allclose(plant[name], value, rtol=0.0, atol=1.0e-6):
            raise RuntimeError(
                f"PhysX {name} readback does not match the identified Config value {value}: "
                f"{plant[name].tolist()}"
            )


def _step(
    scene: InteractiveScene,
    robot,
    sim: SimulationContext,
    controller: RealPathTorqueController,
    reference: torch.Tensor,
    video_recorder: VideoRecorder | None = None,
) -> ControllerSample:
    controller_sample = controller.apply(robot, reference)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
    if video_recorder is not None:
        video_recorder.record_after_step()
    return controller_sample


def _wait_for_stationary(
    scene: InteractiveScene,
    robot,
    sim: SimulationContext,
    controller: RealPathTorqueController,
    target: torch.Tensor,
    sim_joint_ids: list[int],
    video_recorder: VideoRecorder | None = None,
) -> None:
    """Hold the reset target until all arm joint velocities are stationary."""
    stable_s = 0.0
    elapsed_s = 0.0
    while stable_s < args_cli.initial_settle:
        _step(scene, robot, sim, controller, target, video_recorder)
        elapsed_s += args_cli.physics_dt
        qd = _canonical_values(robot.data.joint_vel, sim_joint_ids)
        if np.all(np.abs(qd) <= args_cli.home_velocity_tolerance):
            stable_s += args_cli.physics_dt
        else:
            stable_s = 0.0
        if elapsed_s > args_cli.home_timeout:
            raise RuntimeError(
                "Robot did not become stationary before live q0 capture; "
                f"max |qd|={np.max(np.abs(qd)):.6f} rad/s"
            )


def _reset_and_settle_at_home(
    scene: InteractiveScene,
    robot,
    sim: SimulationContext,
    controller: RealPathTorqueController,
    home: torch.Tensor,
    sim_joint_ids: list[int],
    video_recorder: VideoRecorder | None = None,
) -> None:
    """Place the plant at a trial's q0 target before its probe starts."""
    robot.write_joint_state_to_sim(home, torch.zeros_like(robot.data.default_joint_vel))
    controller.reset()
    _wait_for_stationary(scene, robot, sim, controller, home, sim_joint_ids, video_recorder)


def _check_joint_limit(
    robot,
    home: torch.Tensor,
    logical_joint: int,
    sim_joint_ids: list[int],
) -> None:
    sim_joint = sim_joint_ids[logical_joint]
    amplitude = _amplitude_for(logical_joint)
    target = float(home[0, sim_joint].item() + amplitude)
    limits = robot.data.soft_joint_pos_limits[0, sim_joint]
    lower, upper = float(limits[0].item()), float(limits[1].item())
    margin = args_cli.joint_limit_margin
    if target < lower + margin or target > upper - margin:
        raise ValueError(
            f"{LOGICAL_JOINT_NAMES[logical_joint]} target {target:.4f} rad violates "
            f"[{lower + margin:.4f}, {upper - margin:.4f}] safety band"
        )


def _csv_header() -> list[str]:
    return (
        [
            "t_sec",
            "phase",
            "active_joint",
            "q_ref_active_rad",
            "q_ref_delta_from_q0_rad",
            "cmd_offset_rad",
            "tau_cmd_recon_nm",
            "tau_applied_cmd_nm",
        ]
        + [f"q_{name}" for name in LOGICAL_JOINT_NAMES]
        + [f"qd_{name}" for name in LOGICAL_JOINT_NAMES]
        # Retain this name for existing analysis-tool compatibility.  Metadata
        # declares this channel as simulator actuator torque, not a real sensor.
        + [f"tau_meas_{name}" for name in LOGICAL_JOINT_NAMES]
        + [f"q0_{name}" for name in LOGICAL_JOINT_NAMES]
    )


def _format_numbers(values: Iterable[float]) -> list[str]:
    return [f"{float(value):.6f}" for value in values]


def _run_trial(
    scene: InteractiveScene,
    robot,
    sim: SimulationContext,
    controller: RealPathTorqueController,
    home: torch.Tensor,
    sim_joint_ids: list[int],
    logical_joint: int,
    csv_path: Path,
    video_recorder: VideoRecorder | None = None,
) -> TrialResult:
    """Execute one absolute-q0 profile and wait for all 14 joints to settle."""
    _check_joint_limit(robot, home, logical_joint, sim_joint_ids)
    sim_joint = sim_joint_ids[logical_joint]
    amplitude = _amplitude_for(logical_joint)
    command_period = 1.0 / args_cli.command_rate_hz
    profile_end = _profile_end_s()
    home_canonical = _canonical_values(home, sim_joint_ids)

    elapsed_s = 0.0
    next_command_s = 0.0
    settled_since_s: float | None = None
    max_home_error = 0.0
    max_home_velocity = 0.0
    samples = 0
    reference = home.clone()
    phase = "q0_prehold"

    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(_csv_header())

        while True:
            # In the 1-kHz mode this deliberately holds each 100-Hz reference
            # for ten physics ticks, just like the real probe's command topic.
            if elapsed_s + 1.0e-12 >= next_command_s:
                offset, phase = _profile_offset(next_command_s, amplitude)
                reference = home.clone()
                reference[0, sim_joint] += offset
                next_command_s += command_period

            controller_sample = _step(scene, robot, sim, controller, reference, video_recorder)
            elapsed_s += args_cli.physics_dt

            # Log the same q/qd sample that produced this command.  The
            # controller runs before ``sim.step()``, matching the real control
            # loop's command/measurement semantics.
            q = controller_sample.joint_pos_rad
            qd = controller_sample.joint_vel_rad_s
            tau_tensor, _ = _torque_tensor(robot)
            tau = _canonical_values(tau_tensor, sim_joint_ids)
            reference_canonical = _canonical_values(reference, sim_joint_ids)

            if elapsed_s >= profile_end:
                # Force the exact captured home vector while checking the full
                # 14-D q0 position and velocity condition.
                reference = home.clone()
                reference_canonical = home_canonical
                phase = "q0_stabilize"
                q_after = _canonical_values(robot.data.joint_pos, sim_joint_ids)
                qd_after = _canonical_values(robot.data.joint_vel, sim_joint_ids)
                q_error = np.abs(q_after - home_canonical)
                qd_abs = np.abs(qd_after)
                max_home_error = max(max_home_error, float(np.max(q_error)))
                max_home_velocity = max(max_home_velocity, float(np.max(qd_abs)))
                at_home = bool(
                    np.all(q_error <= args_cli.home_position_tolerance)
                    and np.all(qd_abs <= args_cli.home_velocity_tolerance)
                )
                if at_home:
                    settled_since_s = elapsed_s if settled_since_s is None else settled_since_s
                else:
                    settled_since_s = None

            cmd_offset = float(reference_canonical[logical_joint] - q[logical_joint])
            # This is the real-path PD output after magnitude clipping but
            # before the torque-rate limiter.  ``tau_meas_*`` below records
            # the corresponding PhysX actuator torque after the simulation step.
            tau_recon = controller_sample.computed_effort_nm[logical_joint]
            row = (
                [
                    f"{elapsed_s:.6f}",
                    phase,
                    logical_joint,
                    f"{reference_canonical[logical_joint]:.6f}",
                    f"{reference_canonical[logical_joint] - home_canonical[logical_joint]:.6f}",
                    f"{cmd_offset:.6f}",
                    f"{tau_recon:.6f}",
                    f"{controller_sample.applied_effort_nm[logical_joint]:.6f}",
                ]
                + _format_numbers(q)
                + _format_numbers(qd)
                + _format_numbers(tau)
                + _format_numbers(home_canonical)
            )
            writer.writerow(row)
            samples += 1

            if elapsed_s >= profile_end:
                if settled_since_s is not None and elapsed_s - settled_since_s >= args_cli.home_settle:
                    break
                if elapsed_s - profile_end > args_cli.home_timeout:
                    raise RuntimeError(
                        f"{LOGICAL_JOINT_NAMES[logical_joint]} failed q0 return after "
                        f"{args_cli.home_timeout:.1f}s: max |q-q0|={np.max(np.abs(q_after - home_canonical)):.6f} rad, "
                        f"max |qd|={np.max(np.abs(qd_after)):.6f} rad/s"
                    )

    return TrialResult(
        logical_joint=logical_joint,
        name=LOGICAL_JOINT_NAMES[logical_joint],
        csv_path=csv_path,
        samples=samples,
        duration_s=elapsed_s,
        max_home_error_rad=max_home_error,
        max_home_velocity_rad_s=max_home_velocity,
    )


def _create_scene() -> tuple[SimulationContext, InteractiveScene, object, object | None, bool]:
    robot_cfg = DUAL_FR3_PLATE_EE_CFG.copy()
    if args_cli.robot_gravity == "enabled":
        robot_cfg.spawn.rigid_props.disable_gravity = False
    elif args_cli.robot_gravity == "disabled":
        robot_cfg.spawn.rigid_props.disable_gravity = True
    # Preserve the Reach setting for the default "reach" mode.
    gravity_disabled = bool(robot_cfg.spawn.rigid_props.disable_gravity)

    # The shared robot Config still authors armature/dry-friction/viscous-
    # friction into PhysX.  Disable only its implicit PD drive so this runner
    # has one, explicit controller path: RealPathTorqueController.
    for actuator in robot_cfg.actuators.values():
        actuator.stiffness = 0.0
        actuator.damping = 0.0

    # Config classes keep their fields on the class; make a one-off scene cfg
    # instance and replace only the robot definition for this process.
    scene_cfg = DualFR3SysIdSceneCfg(num_envs=1, env_spacing=3.0)
    scene_cfg.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    if args_cli.video:
        scene_cfg.video_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/SysIdVideoCamera",
            update_period=0.0,
            width=args_cli.video_width,
            height=args_cli.video_height,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 20.0),
            ),
        )
    sim = SimulationContext(sim_utils.SimulationCfg(dt=args_cli.physics_dt, device=args_cli.device))
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    video_camera = scene["video_camera"] if args_cli.video else None
    if video_camera is not None:
        # Same wide three-quarter view used by the existing table preview tool.
        eyes = torch.tensor([[2.6, -1.8, 1.6]], device=sim.device)
        targets = torch.tensor([[0.5, -0.1, 0.6]], device=sim.device)
        video_camera.set_world_poses_from_view(eyes=eyes, targets=targets)
    return sim, scene, scene["robot"], video_camera, gravity_disabled


def main() -> None:
    _validate_args()
    selected_joints = _logical_joint_indices()
    run_name = args_cli.run_name.strip() or time.strftime("%y%m%d_%H%M%S")
    out_dir = args_cli.output_root.expanduser() / run_name
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}. Use --run-name with a new value.")
    out_dir.mkdir(parents=True)

    video_recorder: VideoRecorder | None = None
    sim, scene, robot, video_camera, gravity_disabled = _create_scene()
    try:
        name_to_id = {name: index for index, name in enumerate(robot.joint_names)}
        missing = [name for name in LOGICAL_JOINT_NAMES if name not in name_to_id]
        if missing:
            raise RuntimeError(f"Robot articulation is missing expected joints: {missing}")
        sim_joint_ids = [name_to_id[name] for name in LOGICAL_JOINT_NAMES]

        # The real experiment reaches q0 before starting the probe.  Directly
        # initialise to the Reach default here, then capture the settled live q0.
        if video_camera is not None:
            video_recorder = VideoRecorder(video_camera, out_dir / "sysid.mp4", args_cli.physics_dt)

        controller = RealPathTorqueController(robot, sim_joint_ids, args_cli.physics_dt)
        default = robot.data.default_joint_pos.clone()
        _reset_and_settle_at_home(scene, robot, sim, controller, default, sim_joint_ids, video_recorder)
        default_home = robot.data.joint_pos.clone()
        plant = _read_plant_properties(robot, sim_joint_ids)
        _verify_identified_plant(plant)
        _, torque_source = _torque_tensor(robot)

        metadata = {
            "runner": "run_dual_fr3_sysid.py",
            "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "profile": args_cli.profile,
            "logical_joint_order": LOGICAL_JOINT_NAMES,
            "isaac_articulation_joint_order": list(robot.joint_names),
            "logical_to_isaac_joint_ids": sim_joint_ids,
            "selected_logical_joints": selected_joints,
            "amplitude_rad": args_cli.amplitude_rad,
            "amplitude_wrist_rad": args_cli.amplitude_wrist_rad,
            "t_start_s": args_cli.t_start,
            "ramp_time_s": args_cli.ramp_time,
            "hold_time_s": args_cli.hold_time,
            "pulse_width_s": args_cli.pulse_width,
            "command_rate_hz": args_cli.command_rate_hz,
            "physics_dt_s": args_cli.physics_dt,
            "physics_hz": 1.0 / args_cli.physics_dt,
            "robot_link_gravity_disabled": gravity_disabled,
            "home_position_tolerance_rad": args_cli.home_position_tolerance,
            "home_velocity_tolerance_rad_s": args_cli.home_velocity_tolerance,
            "home_settle_s": args_cli.home_settle,
            "q0_live_canonical_rad": _canonical_values(default_home, sim_joint_ids).tolist(),
            "reference_q0": {
                "enabled": args_cli.ref_dir is not None,
                "directory": str(args_cli.ref_dir.expanduser()) if args_cli.ref_dir is not None else None,
                "selection": "newest exact joint/profile/amplitude CSV",
            },
            "controller": {
                "mode": "custom_measured_relative_pd",
                "kp_nm_per_rad": FR3_ARM_KP_NM_PER_RAD,
                "kd_nm_s_per_rad": FR3_ARM_KD_NM_S_PER_RAD,
                "effort_limit_nm": [
                    FR3_SHOULDER_EFFORT_LIMIT_NM,
                    FR3_SHOULDER_EFFORT_LIMIT_NM,
                    FR3_SHOULDER_EFFORT_LIMIT_NM,
                    FR3_SHOULDER_EFFORT_LIMIT_NM,
                    FR3_FOREARM_EFFORT_LIMIT_NM,
                    FR3_FOREARM_EFFORT_LIMIT_NM,
                    FR3_FOREARM_EFFORT_LIMIT_NM,
                ]
                * 2,
                "torque_rate_limit_nm_s": FR3_TORQUE_RATE_LIMIT_NM_S,
                "implicit_drive_stiffness_nm_per_rad": 0.0,
                "implicit_drive_damping_nm_s_per_rad": 0.0,
            },
            "plant_physx_readback": {name: value.tolist() for name, value in plant.items()},
            "tau_meas_semantics": torque_source,
            "tau_meas_note": "Simulator actuator torque; it is not a real FR3 joint-torque sensor measurement.",
            "video": {
                "enabled": args_cli.video,
                "path": "sysid.mp4" if args_cli.video else None,
                "fps": args_cli.video_fps if args_cli.video else None,
                "width": args_cli.video_width if args_cli.video else None,
                "height": args_cli.video_height if args_cli.video else None,
                "camera_eye_world_m": [2.6, -1.8, 1.6] if args_cli.video else None,
                "camera_target_world_m": [0.5, -0.1, 0.6] if args_cli.video else None,
            },
            "status": "running",
        }
        metadata_path = out_dir / "run_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

        print("\n===== Isaac Lab dual-FR3 SysID =====")
        print(f"output: {out_dir.resolve()}")
        print(f"profile={args_cli.profile}, joints={selected_joints}, physics={1.0 / args_cli.physics_dt:.0f} Hz, "
              f"q_ref={args_cli.command_rate_hz:.0f} Hz")
        print(f"robot link gravity: {'disabled (Reach setting)' if gravity_disabled else 'enabled'}")
        print("logical -> Isaac IDs:")
        for logical, sim_id in enumerate(sim_joint_ids):
            print(f"  [{logical:2d}] {LOGICAL_JOINT_NAMES[logical]:22s} -> {sim_id:2d} ({robot.joint_names[sim_id]})")
        print("captured default q0:", np.array2string(_canonical_values(default_home, sim_joint_ids), precision=4))
        if args_cli.ref_dir is not None:
            print(f"reference q0 directory: {args_cli.ref_dir.expanduser()}")

        results: list[TrialResult] = []
        trial_homes: dict[int, tuple[np.ndarray, Path | None]] = {}
        stamp = time.strftime("%y%m%d_%H%M%S")
        for count, logical_joint in enumerate(selected_joints, start=1):
            name = LOGICAL_JOINT_NAMES[logical_joint]
            amplitude = _amplitude_for(logical_joint)
            reference_home = _load_reference_home(logical_joint)
            if reference_home is None:
                home = default_home
                reference_source = None
            else:
                canonical_home, reference_source = reference_home
                home = _home_target_from_canonical(robot, sim_joint_ids, canonical_home)
                _reset_and_settle_at_home(
                    scene, robot, sim, controller, home, sim_joint_ids, video_recorder
                )
            trial_homes[logical_joint] = (_canonical_values(home, sim_joint_ids), reference_source)
            csv_path = out_dir / f"{name}_{args_cli.profile}_amp{amplitude:.3f}_{stamp}.csv"
            source_label = reference_source.name if reference_source is not None else "live Isaac default q0"
            print(
                f"[{count}/{len(selected_joints)}] {name}: amplitude={amplitude:.4f} rad, q0={source_label}",
                flush=True,
            )
            result = _run_trial(
                scene,
                robot,
                sim,
                controller,
                home,
                sim_joint_ids,
                logical_joint,
                csv_path,
                video_recorder,
            )
            results.append(result)
            print(
                f"  done: {result.samples} samples, {result.duration_s:.3f}s, "
                f"return transient max |q-q0|={result.max_home_error_rad:.5f} rad",
                flush=True,
            )

        metadata["status"] = "complete"
        if video_recorder is not None:
            video_recorder.close()
            metadata["video"]["frames"] = video_recorder.frame_count
        metadata["trials"] = [
            {
                "logical_joint": result.logical_joint,
                "joint_name": result.name,
                "csv": result.csv_path.name,
                "samples": result.samples,
                "duration_s": result.duration_s,
                "max_home_error_rad": result.max_home_error_rad,
                "max_home_velocity_rad_s": result.max_home_velocity_rad_s,
                "q0_target_canonical_rad": trial_homes[result.logical_joint][0].tolist(),
                "reference_csv": (
                    str(trial_homes[result.logical_joint][1])
                    if trial_homes[result.logical_joint][1] is not None
                    else None
                ),
            }
            for result in results
        ]
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"\ncompleted {len(results)} trial(s): {out_dir.resolve()}")
    except Exception:
        # Keep an inspectable run manifest when a trial fails q0 settling.
        metadata_path = out_dir / "run_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            metadata["status"] = "failed"
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        raise
    finally:
        if video_recorder is not None:
            video_recorder.close()
        sim.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
