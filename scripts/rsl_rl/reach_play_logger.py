from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


class IsaacReachRunLogger:
    """Record one Isaac Reach play run and render comparison plots on shutdown."""

    EEF_FIELDNAMES = (
        "elapsed_s",
        "sim_time_ns",
        "goal_index",
        "target_center_x",
        "target_center_y",
        "target_center_z",
        "left_target_x",
        "left_target_y",
        "left_target_z",
        "right_target_x",
        "right_target_y",
        "right_target_z",
        "left_eef_x",
        "left_eef_y",
        "left_eef_z",
        "right_eef_x",
        "right_eef_y",
        "right_eef_z",
    )
    POLICY_FIELDNAMES = (
        "elapsed_s",
        "sim_time_ns",
        "goal_index",
        *(f"action_{index}" for index in range(14)),
        *(f"joint_position_{index}" for index in range(14)),
        *(f"joint_velocity_{index}" for index in range(14)),
        *(f"q_target_{index}" for index in range(14)),
    )

    def __init__(
        self,
        result_root: Path,
        task_name: str,
        checkpoint_path: str,
        sample_rate_hz: float,
        physics_dt_s: float,
        action_scale: float,
        action_reference_mode: str,
        reach_offset_y: float,
        logged_env_index: int,
        goal_sequence: list[dict[str, Any]] | None,
    ):
        started_at = datetime.now().astimezone()
        timestamp = started_at.strftime("%Y%m%d_%H%M%S_%f_%Z")
        run_dir = Path(result_root).expanduser().resolve() / timestamp
        suffix = 1
        while run_dir.exists():
            run_dir = run_dir.with_name(f"{timestamp}_{suffix:02d}")
            suffix += 1
        run_dir.mkdir(parents=True)

        self.run_dir = run_dir
        self.eef_csv_path = run_dir / "eef_trajectory.csv"
        self.policy_csv_path = run_dir / "policy_trace.csv"
        self.started_at = started_at
        self.task_name = task_name
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.checkpoint_sha256 = hashlib.sha256(
            Path(self.checkpoint_path).read_bytes()
        ).hexdigest()
        self.sample_rate_hz = float(sample_rate_hz)
        self.physics_dt_s = float(physics_dt_s)
        self.action_scale = float(action_scale)
        self.action_reference_mode = str(action_reference_mode)
        self.reach_offset_y = float(reach_offset_y)
        self.logged_env_index = int(logged_env_index)
        self.goal_sequence = goal_sequence
        self.sample_count = 0
        self._finalized = False

        self._eef_file = self.eef_csv_path.open("w", newline="", encoding="utf-8")
        self._eef_writer = csv.DictWriter(
            self._eef_file, fieldnames=self.EEF_FIELDNAMES
        )
        self._eef_writer.writeheader()
        self._policy_file = self.policy_csv_path.open(
            "w", newline="", encoding="utf-8"
        )
        self._policy_writer = csv.DictWriter(
            self._policy_file, fieldnames=self.POLICY_FIELDNAMES
        )
        self._policy_writer.writeheader()
        self._flush()

    def append(
        self,
        *,
        elapsed_s: float,
        goal_index: int,
        target_center: np.ndarray,
        left_eef: np.ndarray,
        right_eef: np.ndarray,
        action: np.ndarray,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
        q_target: np.ndarray,
    ) -> None:
        if self._finalized:
            return

        target_center = self._finite_vector(target_center, 3, "target_center")
        left_eef = self._finite_vector(left_eef, 3, "left_eef")
        right_eef = self._finite_vector(right_eef, 3, "right_eef")
        action = self._finite_vector(action, 14, "action")
        joint_position = self._finite_vector(
            joint_position, 14, "joint_position"
        )
        joint_velocity = self._finite_vector(
            joint_velocity, 14, "joint_velocity"
        )
        q_target = self._finite_vector(q_target, 14, "q_target")

        left_target = target_center + np.asarray(
            [0.0, self.reach_offset_y, 0.0], dtype=np.float64
        )
        right_target = target_center - np.asarray(
            [0.0, self.reach_offset_y, 0.0], dtype=np.float64
        )
        sim_time_ns = int(round(float(elapsed_s) * 1.0e9))
        common = {
            "elapsed_s": f"{float(elapsed_s):.9f}",
            "sim_time_ns": str(sim_time_ns),
            "goal_index": str(int(goal_index)),
        }

        eef_values = np.concatenate(
            (target_center, left_target, right_target, left_eef, right_eef)
        )
        eef_row = dict(common)
        eef_row.update(
            {
                name: f"{value:.9f}"
                for name, value in zip(self.EEF_FIELDNAMES[3:], eef_values)
            }
        )
        self._eef_writer.writerow(eef_row)

        policy_row = dict(common)
        for prefix, values in (
            ("action", action),
            ("joint_position", joint_position),
            ("joint_velocity", joint_velocity),
            ("q_target", q_target),
        ):
            policy_row.update(
                {
                    f"{prefix}_{index}": f"{value:.9f}"
                    for index, value in enumerate(values)
                }
            )
        self._policy_writer.writerow(policy_row)

        self.sample_count += 1
        if self.sample_count % max(1, round(self.sample_rate_hz)) == 0:
            self._flush()

    def finalize(self) -> tuple[Path, tuple[Path, ...]]:
        if self._finalized:
            return self.run_dir, self.plot_paths

        self._finalized = True
        self._flush()
        self._eef_file.close()
        self._policy_file.close()
        metadata = {
            "task_name": self.task_name,
            "checkpoint": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "started_at": self.started_at.isoformat(),
            "finished_at": datetime.now().astimezone().isoformat(),
            "sample_count": self.sample_count,
            "nominal_sample_rate_hz": self.sample_rate_hz,
            "policy_dt_s": 1.0 / self.sample_rate_hz,
            "physics_dt_s": self.physics_dt_s,
            "logged_env_index": self.logged_env_index,
            "coordinate_frame": "robot_root_local",
            "quaternion_order": "xyzw",
            "per_arm_target_offset_y_m": self.reach_offset_y,
            "action_reference_mode": self.action_reference_mode,
            "action_semantics": self._action_semantics_description(),
            "goal_sequence": self.goal_sequence,
            "eef_csv": self.eef_csv_path.name,
            "policy_trace_csv": self.policy_csv_path.name,
            "plots": [path.name for path in self.plot_paths],
        }
        (self.run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._render_plots()
        return self.run_dir, self.plot_paths

    @property
    def plot_paths(self) -> tuple[Path, ...]:
        return (
            self.run_dir / "eef_x_vs_target.png",
            self.run_dir / "eef_y_vs_target.png",
            self.run_dir / "eef_z_vs_target.png",
            self.run_dir / "eef_trajectory_3d.png",
            self.run_dir / "policy_actions.png",
            self.run_dir / "joint_positions_vs_targets.png",
            self.run_dir / "joint_velocities.png",
        )

    def _action_semantics_description(self) -> str:
        if self.action_reference_mode == "policy_step":
            return (
                "bounded_action[-1,1]; "
                f"q_target[k]=q_measured(t_k)+{self.action_scale:g}*action[k]; "
                "absolute q_target held until the next policy step"
            )
        return (
            "bounded_action[-1,1]; "
            f"q_target(t)=q_measured(t)+{self.action_scale:g}*action[k] at every physics step"
        )

    @staticmethod
    def _finite_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (size,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be a finite {size}-vector")
        return vector

    def _flush(self) -> None:
        self._eef_file.flush()
        self._policy_file.flush()

    @staticmethod
    def _load_csv(path: Path, sample_count: int) -> np.ndarray:
        if sample_count == 0:
            return np.empty(0)
        return np.atleast_1d(
            np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
        )

    @staticmethod
    def _configure_matplotlib():
        import matplotlib

        matplotlib.use("Agg", force=True)
        import mpl_toolkits

        matching_toolkits = (
            Path(matplotlib.__file__).resolve().parent.parent / "mpl_toolkits"
        )
        if matching_toolkits.is_dir():
            matching_path = str(matching_toolkits)
            mpl_toolkits.__path__ = [
                matching_path,
                *(path for path in mpl_toolkits.__path__ if path != matching_path),
            ]
        from mpl_toolkits.mplot3d import Axes3D as _Axes3D  # noqa: F401
        import matplotlib.pyplot as plt

        return plt

    def _render_plots(self) -> None:
        plt = self._configure_matplotlib()
        eef = self._load_csv(self.eef_csv_path, self.sample_count)
        policy = self._load_csv(self.policy_csv_path, self.sample_count)

        for axis_name, output_path in zip(("x", "y", "z"), self.plot_paths[:3]):
            fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
            if eef.size:
                time_s = eef["elapsed_s"]
                for side, label in (("left", "Left"), ("right", "Right")):
                    line = axis.plot(
                        time_s, eef[f"{side}_eef_{axis_name}"], label=f"{label} EEF"
                    )[0]
                    axis.plot(
                        time_s,
                        eef[f"{side}_target_{axis_name}"],
                        "--",
                        color=line.get_color(),
                        label=f"{label} target",
                    )
                axis.legend()
            axis.set_title(f"EEF {axis_name.upper()} position vs. reach target")
            axis.set_xlabel("Simulation time [s]")
            axis.set_ylabel(f"{axis_name.upper()} in robot-root frame [m]")
            axis.grid(True, alpha=0.3)
            fig.savefig(output_path, dpi=160)
            plt.close(fig)

        fig = plt.figure(figsize=(9, 8), constrained_layout=True)
        axis = fig.add_subplot(111, projection="3d")
        if eef.size:
            all_xyz = []
            for side, label in (("left", "Left"), ("right", "Right")):
                line = axis.plot(
                    eef[f"{side}_eef_x"],
                    eef[f"{side}_eef_y"],
                    eef[f"{side}_eef_z"],
                    label=f"{label} EEF",
                )[0]
                axis.plot(
                    eef[f"{side}_target_x"],
                    eef[f"{side}_target_y"],
                    eef[f"{side}_target_z"],
                    "--",
                    color=line.get_color(),
                    label=f"{label} target",
                )
                all_xyz.extend(
                    (
                        np.column_stack(
                            (
                                eef[f"{side}_eef_x"],
                                eef[f"{side}_eef_y"],
                                eef[f"{side}_eef_z"],
                            )
                        ),
                        np.column_stack(
                            (
                                eef[f"{side}_target_x"],
                                eef[f"{side}_target_y"],
                                eef[f"{side}_target_z"],
                            )
                        ),
                    )
                )
            points = np.concatenate(all_xyz, axis=0)
            center = 0.5 * (points.min(axis=0) + points.max(axis=0))
            radius = max(0.05, 0.5 * float(np.ptp(points, axis=0).max()))
            axis.set_xlim(center[0] - radius, center[0] + radius)
            axis.set_ylim(center[1] - radius, center[1] + radius)
            axis.set_zlim(center[2] - radius, center[2] + radius)
            axis.legend()
        axis.set_title("Dual-arm EEF trajectories in robot-root frame")
        axis.set_xlabel("X [m]")
        axis.set_ylabel("Y [m]")
        axis.set_zlabel("Z [m]")
        axis.set_box_aspect((1.0, 1.0, 1.0))
        fig.savefig(self.plot_paths[3], dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
        if policy.size:
            for side_index, (axis, side) in enumerate(zip(axes, ("Left", "Right"))):
                for joint in range(7):
                    action_index = side_index * 7 + joint
                    axis.plot(
                        policy["elapsed_s"],
                        policy[f"action_{action_index}"],
                        label=f"J{joint + 1}",
                    )
                axis.set_ylabel(f"{side} action")
                axis.grid(True, alpha=0.3)
                axis.legend(ncol=7, fontsize=8)
        axes[-1].set_xlabel("Simulation time [s]")
        fig.suptitle("Bounded policy actions")
        fig.savefig(self.plot_paths[4], dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(7, 2, figsize=(14, 18), sharex=True, constrained_layout=True)
        if policy.size:
            for side_index, side in enumerate(("Left", "Right")):
                for joint in range(7):
                    index = side_index * 7 + joint
                    axis = axes[joint, side_index]
                    axis.plot(
                        policy["elapsed_s"], policy[f"joint_position_{index}"], label="q"
                    )
                    axis.plot(
                        policy["elapsed_s"], policy[f"q_target_{index}"], "--", label="q target"
                    )
                    axis.set_title(f"{side} J{joint + 1}")
                    axis.grid(True, alpha=0.3)
                    if joint == 0:
                        axis.legend()
        for axis in axes[-1, :]:
            axis.set_xlabel("Simulation time [s]")
        fig.suptitle("Joint positions and policy targets")
        fig.savefig(self.plot_paths[5], dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
        if policy.size:
            for side_index, (axis, side) in enumerate(zip(axes, ("Left", "Right"))):
                for joint in range(7):
                    index = side_index * 7 + joint
                    axis.plot(
                        policy["elapsed_s"],
                        policy[f"joint_velocity_{index}"],
                        label=f"J{joint + 1}",
                    )
                axis.set_ylabel(f"{side} qdot [rad/s]")
                axis.grid(True, alpha=0.3)
                axis.legend(ncol=7, fontsize=8)
        axes[-1].set_xlabel("Simulation time [s]")
        fig.suptitle("Joint velocities")
        fig.savefig(self.plot_paths[6], dpi=160)
        plt.close(fig)
