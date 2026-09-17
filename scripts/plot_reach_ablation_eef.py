#!/usr/bin/env python3
"""Compare Reach EEF trajectories for the four dynamics ablation cases.

The script discovers the latest run for each of these cases below ``result/``:

* 100 Hz, unmatched passive dynamics (NoFriction task)
* 100 Hz, MuJoCo-matched passive dynamics (Friction task)
* 1 kHz, unmatched passive dynamics (NoFriction task)
* 1 kHz, MuJoCo-matched passive dynamics (Friction task)

It then overlays the per-arm goal and all four EEF trajectories in a 3-by-2
figure: rows are x/y/z and columns are left/right EEFs.

Example:
    python3 scripts/plot_reach_ablation_eef.py

    python3 scripts/plot_reach_ablation_eef.py \
        --result-root result \
        --output result/reach_ablation_eef_xyz_comparison.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


AXES = ("x", "y", "z")
ARMS = ("left", "right")


@dataclass(frozen=True)
class CaseSpec:
    key: str
    label: str
    physics_rate_hz: int
    matched: bool
    color: str
    linestyle: str


CASE_SPECS = (
    CaseSpec(
        key="100hz_unmatched",
        label="100 Hz / unmatched dynamics",
        physics_rate_hz=100,
        matched=False,
        color="#D55E00",
        linestyle="--",
    ),
    CaseSpec(
        key="100hz_matched",
        label="100 Hz / matched dynamics",
        physics_rate_hz=100,
        matched=True,
        color="#0072B2",
        linestyle="-",
    ),
    CaseSpec(
        key="1khz_unmatched",
        label="1 kHz / unmatched dynamics",
        physics_rate_hz=1000,
        matched=False,
        color="#CC79A7",
        linestyle="--",
    ),
    CaseSpec(
        key="1khz_matched",
        label="1 kHz / matched dynamics",
        physics_rate_hz=1000,
        matched=True,
        color="#009E73",
        linestyle="-",
    ),
)

REQUIRED_COLUMNS = {
    "elapsed_s",
    "goal_index",
    *(f"{arm}_{kind}_{axis}" for arm in ARMS for kind in ("target", "eef") for axis in AXES),
}


@dataclass
class RunData:
    run_dir: Path
    metadata: dict
    columns: dict[str, np.ndarray]


def _case_key(metadata: dict, run_dir: Path) -> str:
    """Classify a run from metadata, with its directory name as a fallback."""
    task_name = str(metadata.get("task_name", "")).lower()
    directory_name = run_dir.name.lower()

    physics_dt = metadata.get("physics_dt_s")
    if physics_dt is not None:
        physics_rate_hz = int(round(1.0 / float(physics_dt)))
    elif "1000hz" in directory_name or "1khz" in directory_name:
        physics_rate_hz = 1000
    elif "100hz" in directory_name:
        physics_rate_hz = 100
    else:
        raise ValueError(f"Cannot determine physics rate for {run_dir}")

    combined_name = f"{task_name} {directory_name}"
    if "nofriction" in task_name or "unmatched" in directory_name:
        matched = False
    elif "friction" in task_name or "matched dynamics" in directory_name:
        matched = True
    else:
        raise ValueError(f"Cannot determine passive-dynamics case for {run_dir}")

    if physics_rate_hz == 100:
        return "100hz_matched" if matched else "100hz_unmatched"
    if physics_rate_hz == 1000:
        return "1khz_matched" if matched else "1khz_unmatched"
    raise ValueError(
        f"Unsupported physics rate {physics_rate_hz} Hz in {run_dir}; "
        "expected 100 or 1000 Hz"
    )


def _read_run(run_dir: Path) -> RunData:
    metadata_path = run_dir / "metadata.json"
    csv_path = run_dir / "eef_trajectory.csv"
    if not metadata_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError(
            f"Expected both metadata.json and eef_trajectory.csv in {run_dir}"
        )

    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)

    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - fieldnames)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {', '.join(missing)}")
        rows.extend(reader)

    if len(rows) < 2:
        raise ValueError(f"{csv_path} contains fewer than two samples")

    columns = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in REQUIRED_COLUMNS
    }
    elapsed_s = columns["elapsed_s"]
    if not np.all(np.isfinite(elapsed_s)) or np.any(np.diff(elapsed_s) <= 0.0):
        raise ValueError(f"{csv_path}: elapsed_s must be finite and strictly increasing")

    for name, values in columns.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{csv_path}: column {name} contains a non-finite value")

    declared_count = metadata.get("sample_count")
    if declared_count is not None and int(declared_count) != len(rows):
        raise ValueError(
            f"{csv_path}: metadata sample_count={declared_count}, CSV rows={len(rows)}"
        )

    return RunData(run_dir=run_dir, metadata=metadata, columns=columns)


def discover_cases(result_root: Path) -> dict[str, RunData]:
    """Return the latest complete run for every expected ablation case."""
    candidates: dict[str, list[RunData]] = {spec.key: [] for spec in CASE_SPECS}
    for csv_path in sorted(result_root.glob("*/eef_trajectory.csv")):
        run_dir = csv_path.parent
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.is_file():
            print(f"Warning: skipping {run_dir}: metadata.json is missing", file=sys.stderr)
            continue
        run = _read_run(run_dir)
        try:
            key = _case_key(run.metadata, run_dir)
        except ValueError as exc:
            print(f"Warning: {exc}; skipping run", file=sys.stderr)
            continue
        candidates[key].append(run)

    selected: dict[str, RunData] = {}
    missing: list[str] = []
    for spec in CASE_SPECS:
        runs = candidates[spec.key]
        if not runs:
            missing.append(spec.label)
            continue
        runs.sort(
            key=lambda run: (
                str(run.metadata.get("started_at", "")),
                run.run_dir.stat().st_mtime_ns,
            )
        )
        selected[spec.key] = runs[-1]
        if len(runs) > 1:
            print(
                f"Warning: found {len(runs)} runs for {spec.label}; "
                f"using latest: {runs[-1].run_dir}",
                file=sys.stderr,
            )

    if missing:
        raise RuntimeError(
            "Could not find all four ablation cases below "
            f"{result_root}. Missing: {', '.join(missing)}"
        )
    return selected


def _validate_common_goal(cases: dict[str, RunData]) -> RunData:
    """Check that every case used the same timestamps and per-arm goal sequence."""
    reference = cases[CASE_SPECS[0].key]
    fields = ["elapsed_s", "goal_index"] + [
        f"{arm}_target_{axis}" for arm in ARMS for axis in AXES
    ]
    for spec in CASE_SPECS[1:]:
        candidate = cases[spec.key]
        for field in fields:
            lhs = reference.columns[field]
            rhs = candidate.columns[field]
            if lhs.shape != rhs.shape or not np.allclose(lhs, rhs, rtol=0.0, atol=1.0e-8):
                raise ValueError(
                    "The four runs do not share an identical time/goal sequence: "
                    f"{reference.run_dir.name} vs {candidate.run_dir.name}, field={field}"
                )
    return reference


def plot_comparison(cases: dict[str, RunData], output_path: Path, dpi: int) -> None:
    reference = _validate_common_goal(cases)
    time_s = reference.columns["elapsed_s"]
    goal_index = reference.columns["goal_index"]
    goal_change_rows = np.flatnonzero(np.diff(goal_index) != 0.0) + 1

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(16, 11), sharex=True, constrained_layout=False)

    for row, axis_name in enumerate(AXES):
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            ax.step(
                time_s,
                reference.columns[f"{arm}_target_{axis_name}"],
                where="post",
                color="#111111",
                linewidth=2.3,
                label="Goal",
                zorder=5,
            )
            for spec in CASE_SPECS:
                run = cases[spec.key]
                ax.plot(
                    run.columns["elapsed_s"],
                    run.columns[f"{arm}_eef_{axis_name}"],
                    color=spec.color,
                    linestyle=spec.linestyle,
                    linewidth=1.7,
                    alpha=0.95,
                    label=spec.label,
                )

            for change_row in goal_change_rows:
                ax.axvline(
                    time_s[change_row], color="#B8B8B8", linewidth=0.8, linestyle=":", zorder=0
                )
            ax.set_title(f"{arm.capitalize()} EEF: {axis_name}")
            ax.set_ylabel("Position [m]")
            ax.grid(True, color="#D7D7D7", linewidth=0.7, alpha=0.75)
            ax.margins(x=0.0)

    axes[-1, 0].set_xlabel("Elapsed time [s]")
    axes[-1, 1].set_xlabel("Elapsed time [s]")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "Isaac Sim Reach EEF trajectory comparison",
        fontsize=15,
        fontweight="semibold",
        y=0.992,
    )
    fig.text(
        0.5,
        0.012,
        "Coordinates: robot-root local frame. Vertical dotted lines mark goal changes.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0.035, 0.035, 0.985, 0.925), h_pad=1.4, w_pad=1.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_result_root = repo_root / "result"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=default_result_root,
        help=f"Directory containing logged run folders (default: {default_result_root})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <result-root>/reach_ablation_eef_xyz_comparison.png)",
    )
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution (default: 180)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else result_root / "reach_ablation_eef_xyz_comparison.png"
    )
    if not result_root.is_dir():
        raise FileNotFoundError(f"Result root does not exist: {result_root}")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    cases = discover_cases(result_root)
    plot_comparison(cases, output_path, args.dpi)

    print("Selected runs:")
    for spec in CASE_SPECS:
        run = cases[spec.key]
        print(f"  {spec.label}: {run.run_dir}")
    print(f"Saved plot: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
