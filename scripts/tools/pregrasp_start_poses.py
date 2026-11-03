"""Pick pre-grasp start poses for real-robot box-lift trials from an exported policy NPZ (numpy only).

A ``-Pregrasp`` policy starts with the plates 4 cm outside the box faces, at one row of the pre-grasp table it was
trained with (embedded in the NPZ by ``export_boxlift_actor.py``). For each requested box pose (base frame) this
picks the table row whose box pose is closest in the training-noise metric max(|dxy| / 1 cm, |dyaw| / 0.05 rad),
and writes that row's box pose (where to put the box) and joint targets (where to send the arms, e.g. with
``fr3_husky_task_manager move_to_joint``) to YAML.

    python scripts/tools/pregrasp_start_poses.py /path/to/boxlift_pregrasp.npz /path/to/start_poses.yaml \
        --box 0.46 0.0 0 --box 0.48 0.0 0 --box 0.50 0.0 0
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


def yaw_of(q_xyzw: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = np.asarray(q_xyzw, dtype=np.float64).T
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("policy", type=Path, help="NPZ from export_boxlift_actor.py of a -Pregrasp run")
    parser.add_argument("output", type=Path, help="output .yaml")
    parser.add_argument("--box", type=float, nargs=3, action="append", metavar=("X", "Y", "YAW_DEG"),
                        help="requested box centre x, y [m] (base frame) and yaw [deg]; repeat for more poses")
    args = parser.parse_args()

    data = np.load(args.policy, allow_pickle=False)
    manifest = json.loads(str(data["manifest_json"]))
    start = manifest.get("start", {})
    if start.get("mode") != "pregrasp_table":
        raise SystemExit(f"{args.policy} is not a pre-grasp policy (start mode {start.get('mode')!r})")
    q = data[start["arrays"]["joint_position"]].astype(np.float64)
    pose = data[start["arrays"]["box_pose_base"]].astype(np.float64)
    yaw = yaw_of(pose[:, 3:7])
    names = manifest["joints"]["names"]
    limits = np.asarray(manifest["action"]["target_limits"])
    pos_noise, yaw_noise = start["box_pos_noise_m"], start["box_yaw_noise_rad"]
    half = [0.5 * float(v) for v in manifest["box"]["size"]]

    poses = []
    for x, y, yaw_deg in args.box or [(0.48, 0.0, 0.0)]:
        cost = np.maximum(np.hypot(pose[:, 0] - x, pose[:, 1] - y) / pos_noise,
                          np.abs(yaw - math.radians(yaw_deg)) / yaw_noise)
        i = int(cost.argmin())
        margin = np.minimum(q[i] - limits[:, 0], limits[:, 1] - q[i])
        left, right = [round(float(v), 4) for v in q[i, :7]], [round(float(v), 4) for v in q[i, 7:]]
        bx, by, bz, byaw = float(pose[i, 0]), float(pose[i, 1]), float(pose[i, 2]), float(yaw[i])
        poses.append({
            "requested_box": {"x": x, "y": y, "yaw_deg": yaw_deg},
            "table_row": i,
            "box_center_base": {"x": round(bx, 4), "y": round(by, 4), "z": round(bz, 4),
                                "yaw_deg": round(math.degrees(byaw), 2)},
            "box_placement": {
                "robot_side_face_x_m": round(bx - half[0], 4),
                "far_face_x_m": round(bx + half[0], 4),
                "left_face_y_m": round(by + half[1], 4),
                "right_face_y_m": round(by - half[1], 4),
            },
            "distance_to_request": {"xy_m": round(float(np.hypot(bx - x, by - y)), 4),
                                    "yaw_deg": round(math.degrees(byaw) - yaw_deg, 2)},
            "min_joint_margin_rad": round(float(margin.min()), 3),
            "left_target_positions": left,
            "right_target_positions": right,
            "move_to_joint": ("ros2 run fr3_husky_task_manager move_to_joint --left-target-positions "
                              + " ".join(f"{v}" for v in left) + " --right-target-positions "
                              + " ".join(f"{v}" for v in right)),
        })

    out = {
        "policy": str(args.policy),
        "provenance": {k: manifest["provenance"][k] for k in ("experiment_name", "run_name", "iteration")},
        "frame": "robot base (training root): arm mounting plate, midpoint of the two arm bases; x forward, y left, z up",
        "floor_z_base_m": -manifest["robot"]["root_height_m"],
        "box_size_m": manifest["box"]["size"],
        "training_noise": {"box_pos_m": pos_noise, "box_yaw_rad": yaw_noise, "joint_rad": start["joint_noise_rad"]},
        "joint_order": names,
        "poses": poses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=200))
    for p in poses:
        b = p["box_center_base"]
        print(f"row {p['table_row']:4d}: box ({b['x']:.3f}, {b['y']:+.3f}, yaw {b['yaw_deg']:+.1f} deg), "
              f"off request {100 * p['distance_to_request']['xy_m']:.1f} cm / {p['distance_to_request']['yaw_deg']:+.1f} deg, "
              f"joint margin {p['min_joint_margin_rad']:.3f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
