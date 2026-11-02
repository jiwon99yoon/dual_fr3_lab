# Copyright (c) 2026, DYROS.
# SPDX-License-Identifier: BSD-3-Clause
"""Export a plate box-lift actor to a deployable NPZ with an embedded deployment manifest.

A ``model_*.pt`` checkpoint holds only network weights. Everything the real-robot node must
match (box size, joint order, default pose, action scale/clip, reference mode, joint-target
limits, goal and start ranges, timing) lives in the run's ``params/env.yaml``, which Isaac Lab
writes when training starts. This tool reads both and writes one NPZ:

* actor weights in the ``fr3_husky_nn_policy.numpy_actor.NumpyMLPActor`` format
  (``format_version`` 2, ``weight_i``/``bias_i``, ``output_activation`` ``tanh``);
* ``manifest_json``: the deployment manifest, validated on the robot by
  ``fr3_husky_nn_policy.boxlift_core``.

The export refuses when ``params/env.yaml`` is missing; it never reconstructs the training
configuration from the current source tree, which may have changed since the run.

Run on the training PC (needs torch, numpy, PyYAML; no Isaac Sim app):

.. code-block:: bash

    python scripts/tools/export_boxlift_actor.py \
        logs/rsl_rl/<experiment>/<run>/model_XXXX.pt  /path/to/boxlift_<run>.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

MANIFEST_SCHEMA = "dual_fr3_boxlift_manifest/1"
_TERM_FUNCS = {
    # term name -> function name expected at the end of the "module:function" path
    "joint_pos": "joint_pos_rel",
    "joint_vel": "joint_vel_rel",
    "object_position": "object_position_in_robot_root_frame",
    "target_object_position": "generated_commands",
    "actions": "last_action",
    "object_orientation": "object_orientation_in_robot_root_frame",
}


class _ParamsLoader(yaml.SafeLoader):
    """Read Isaac Lab ``params/env.yaml`` without executing any ``!!python/...`` tag."""


def _construct_python_tag(loader, _suffix, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


_ParamsLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_python_tag)


def _load_params(path: Path) -> dict:
    with path.open() as handle:
        return yaml.load(handle, Loader=_ParamsLoader)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _pair(value, name: str) -> list[float]:
    _require(isinstance(value, (list, tuple)) and len(value) == 2, f"{name} must be a (low, high) pair, got {value!r}")
    return [float(value[0]), float(value[1])]


def _resolve_joint_values(pattern_values: dict, joint_names: list[str], name: str) -> list[float]:
    """Resolve an Isaac Lab ``{regex: value}`` mapping onto an explicit joint list (exactly one match each)."""
    resolved = []
    for joint in joint_names:
        matches = [value for pattern, value in pattern_values.items() if re.fullmatch(pattern, joint)]
        _require(len(matches) == 1, f"{name}: joint {joint} matches {len(matches)} patterns in {pattern_values}")
        resolved.append(float(matches[0]))
    return resolved


def _find_event(events: dict, *, func_suffix: str, asset: str = "object") -> dict | None:
    for event in events.values():
        if not isinstance(event, dict):
            continue
        params = event.get("params") or {}
        if str(event.get("func", "")).endswith(func_suffix) and (params.get("asset_cfg") or {}).get("name") == asset:
            return event
    return None


def build_manifest(env: dict, agent: dict, checkpoint: Path, env_yaml: Path, checkpoint_sha256: str,
                   iteration: int) -> dict:
    # ---- policy / timing
    actor_cfg = agent.get("actor") or {}
    distribution = str((actor_cfg.get("distribution_cfg") or {}).get("class_name", ""))
    _require("TanhSquashedGaussian" in distribution, f"actor distribution must be tanh-squashed, got {distribution!r}")
    _require(not actor_cfg.get("obs_normalization", False), "actor obs_normalization must be false (no normalizer is exported)")
    physics_dt = float(env["sim"]["dt"])
    decimation = int(env["decimation"])

    # ---- observation layout
    policy_obs = env["observations"]["policy"]
    terms = [(name, str(term["func"]).split(":")[-1]) for name, term in policy_obs.items()
             if isinstance(term, dict) and "func" in term]
    for name, func in terms:
        _require(_TERM_FUNCS.get(name) == func, f"unexpected policy observation term {name} -> {func}")
    _require([name for name, _ in terms] == list(_TERM_FUNCS), f"policy observation order {terms} != {list(_TERM_FUNCS)}")

    # ---- joints / action
    action = env["actions"]["arm_action"]
    joint_names = [str(n) for n in action["joint_names"]]
    _require(bool(action.get("preserve_order")), "arm_action.preserve_order must be true")
    for term_name in ("joint_pos", "joint_vel"):
        obs_joints = ((policy_obs[term_name].get("params") or {}).get("asset_cfg") or {}).get("joint_names")
        _require([str(n) for n in obs_joints] == joint_names, f"{term_name} joint order differs from arm_action")
    clip = action.get("clip") or {}
    _require(list(clip) == [".*"], f"arm_action.clip must be a single '.*' range, got {clip!r}")
    limits = action.get("target_limits")
    _require(limits is not None and len(limits) == len(joint_names), "arm_action.target_limits must list every joint")
    robot = env["scene"]["robot"]
    default_position = _resolve_joint_values(robot["init_state"]["joint_pos"], joint_names, "robot.init_state.joint_pos")
    root_pos = [float(v) for v in robot["init_state"]["pos"]]
    root_rot = [float(v) for v in robot["init_state"]["rot"]]
    _require(np.allclose(root_rot, [0.0, 0.0, 0.0, 1.0]), f"robot root rotation must be identity (xyzw), got {root_rot}")

    # ---- box
    box = env["scene"]["object"]
    size = [float(v) for v in box["spawn"]["size"]]
    box_init = [float(v) for v in box["init_state"]["pos"]]
    reset = _find_event(env["events"], func_suffix=":reset_root_state_uniform")
    _require(reset is not None, "no reset_root_state_uniform event for the object")
    pose_range = reset["params"]["pose_range"]
    scale_event = _find_event(env["events"], func_suffix=":randomize_rigid_body_scale")
    mass_event = _find_event(env["events"], func_suffix=":randomize_rigid_body_mass")
    friction_event = _find_event(env["events"], func_suffix=":randomize_rigid_body_material")
    scale_range = ({axis: _pair(v, f"scale_range.{axis}") for axis, v in scale_event["params"]["scale_range"].items()}
                   if scale_event else {"x": [1.0, 1.0], "y": [1.0, 1.0], "z": [1.0, 1.0]})
    nominal_mass = (box["spawn"].get("mass_props") or {}).get("mass")

    # ---- goal command
    command = env["commands"]["object_pose"]
    ranges = command["ranges"]
    for axis in ("roll", "pitch", "yaw"):
        _require(_pair(ranges[axis], axis) == [0.0, 0.0], f"goal {axis} range must be zero (identity goal quaternion)")

    def start_range(axis: int, key: str) -> list[float]:
        low, high = _pair(pose_range.get(key, [0.0, 0.0]), f"pose_range.{key}")
        return [box_init[axis] + low - root_pos[axis], box_init[axis] + high - root_pos[axis]]

    run_dir = checkpoint.parent
    git_diff = run_dir / "git" / "dual_fr3_lab.diff"
    git_commit = None
    if git_diff.is_file():
        lines = git_diff.read_text(errors="replace").splitlines()
        for index, line in enumerate(lines):
            if line.strip() == "--- git commit ---" and index + 1 < len(lines):
                git_commit = lines[index + 1].strip()
                break

    return {
        "schema": MANIFEST_SCHEMA,
        "provenance": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "iteration": iteration,
            "experiment_name": agent.get("experiment_name"),
            "run_name": agent.get("run_name"),
            "env_yaml": str(env_yaml),
            "env_yaml_sha256": _sha256(env_yaml),
            "git_commit": git_commit,
            "exported_utc": datetime.now(timezone.utc).isoformat(),
            "exporter": "dual_fr3_lab/scripts/tools/export_boxlift_actor.py",
        },
        "policy": {
            "physics_dt": physics_dt,
            "decimation": decimation,
            "policy_dt": physics_dt * decimation,
            "episode_length_s": float(env["episode_length_s"]),
            "output_activation": "tanh",
        },
        "observation": {"terms": [[name, func] for name, func in terms]},
        "joints": {"names": joint_names, "default_position": default_position},
        "action": {
            "scale": float(action["scale"]),
            "clip": _pair(clip[".*"], "clip"),
            "reference_mode": str(action["reference_mode"]),
            "target_limits": [_pair(pair, "target_limits") for pair in limits],
        },
        "box": {
            "size": size,
            "size_scale_range": scale_range,
            "nominal_mass": None if nominal_mass is None else float(nominal_mass),
            "mass_range": _pair(mass_event["params"]["mass_distribution_params"], "mass") if mass_event else None,
            "static_friction_range": (_pair(friction_event["params"]["static_friction_range"], "friction")
                                      if friction_event else None),
            "start_x_range_base": start_range(0, "x"),
            "start_y_range_base": start_range(1, "y"),
            "start_yaw_range": _pair(pose_range.get("yaw", [0.0, 0.0]), "pose_range.yaw"),
            "rest_center_z_base": 0.5 * size[2] - root_pos[2],
        },
        "goal": {
            "command_class": str(command["class_type"]),
            "z_range_base": _pair(ranges["pos_z"], "pos_z"),
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "robot": {"root_height_m": root_pos[2]},
    }


def export(checkpoint: Path, output: Path, env_yaml: Path | None = None) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    run_dir = checkpoint.parent
    env_yaml = (env_yaml or run_dir / "params" / "env.yaml").expanduser().resolve()
    agent_yaml = env_yaml.parent / "agent.yaml"
    if not env_yaml.is_file() or not agent_yaml.is_file():
        raise FileNotFoundError(
            f"{env_yaml} / {agent_yaml} not found. Export from the training run directory "
            "(model_*.pt next to params/), or pass --env-yaml for that exact run."
        )

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    actor = state.get("actor_state_dict")
    if actor is None:
        raise KeyError("checkpoint has no actor_state_dict")
    layers = sorted(int(k.split(".")[1]) for k in actor if k.startswith("mlp.") and k.endswith(".weight"))
    if not layers or any(f"mlp.{i}.bias" not in actor for i in layers):
        raise ValueError("actor MLP weights/biases not found")

    checkpoint_sha256 = _sha256(checkpoint)
    iteration = int(state.get("iter", -1))
    manifest = build_manifest(_load_params(env_yaml), _load_params(agent_yaml), checkpoint, env_yaml,
                              checkpoint_sha256, iteration)

    payload = {
        "format_version": np.asarray(2, dtype=np.int64),
        "activation": np.asarray("elu"),
        "output_activation": np.asarray("tanh"),
        "num_layers": np.asarray(len(layers), dtype=np.int64),
        "source_sha256": np.asarray(checkpoint_sha256),
        "source_iteration": np.asarray(iteration, dtype=np.int64),
        "manifest_json": np.asarray(json.dumps(manifest, sort_keys=False)),
    }
    for out_index, state_index in enumerate(layers):
        payload[f"weight_{out_index}"] = actor[f"mlp.{state_index}.weight"].detach().cpu().numpy().astype(np.float32)
        payload[f"bias_{out_index}"] = actor[f"mlp.{state_index}.bias"].detach().cpu().numpy().astype(np.float32)
    in_dim = payload["weight_0"].shape[1]
    out_dim = payload[f"weight_{len(layers) - 1}"].shape[0]
    num_joints = len(manifest["joints"]["names"])
    expected_in = 3 * num_joints + 3 + 7 + 4  # q, qd, previous action + box pos, goal, box quat
    if (in_dim, out_dim) != (expected_in, num_joints):
        raise ValueError(f"actor is {in_dim}->{out_dim}, manifest implies {expected_in}->{num_joints}")

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **payload)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="model_*.pt inside a training run directory")
    parser.add_argument("output", type=Path, help="output .npz")
    parser.add_argument("--env-yaml", type=Path, default=None,
                        help="params/env.yaml of the same run (default: <checkpoint dir>/params/env.yaml)")
    args = parser.parse_args()
    path = export(args.checkpoint, args.output, args.env_yaml)
    manifest = json.loads(str(np.load(path)["manifest_json"]))
    box = manifest["box"]
    print(f"wrote {path}")
    print(f"  run {manifest['provenance']['experiment_name']}/{manifest['provenance']['run_name']} "
          f"iter {manifest['provenance']['iteration']}, sha256 {manifest['provenance']['checkpoint_sha256'][:12]}")
    print(f"  box size {box['size']} (scale {box['size_scale_range']}), mass {box['mass_range']}, "
          f"goal z {manifest['goal']['z_range_base']}, reference {manifest['action']['reference_mode']}")


if __name__ == "__main__":
    main()
