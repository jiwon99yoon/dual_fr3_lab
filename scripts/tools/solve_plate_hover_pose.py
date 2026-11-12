"""Hover start pose for the box lift -Curriculum task (2026-11-10).

Takes the pre-grasp table row whose box is closest to the nominal (0.45, 0), moves both plate TCPs up by ``--dz`` and
outward by ``--gap`` each (plate normals and edge alignment unchanged), solves the left arm with damped least squares on a
numeric Jacobian and mirrors it to the right arm (j1/j3/j5 negated, j7 -> pi/2 - j7_L). Prints HOVER_Q_LEFT/RIGHT for
lift_plate/joint_pos_env_cfg.py.

FK = FR3 modified DH (fr3_husky_etri fr3_kinematics.py) + mounts (0, +-0.215, 0) yaw +-30 deg on the base (z 0.405) +
plate TCP offset (0, 0, 0.132) in link7. The table is checked first: plate TCPs 4 cm outside the 0.41 m box faces.

    python scripts/tools/solve_plate_hover_pose.py --dz 0.25 --gap 0.075
"""

import argparse
import math
import os

import numpy as np

DH = ((0.0, 0.333, 0.0), (0.0, 0.0, -math.pi / 2), (0.0, 0.316, math.pi / 2), (0.0825, 0.0, math.pi / 2),
      (-0.0825, 0.384, -math.pi / 2), (0.0, 0.0, math.pi / 2), (0.088, 0.0, math.pi / 2))
TCP = np.array([0.0, 0.0, 0.132])
ROOT_Z = 0.405
LIMITS = np.array([(-2.7437, 2.7437), (-1.7837, 1.7837), (-2.9007, 2.9007), (-3.0421, -0.1518),
                   (-2.8065, 2.8065), (0.5445, 4.5169), (-3.0159, 3.0159)])
MIRROR = np.array([-1, 1, -1, 1, -1, 1, -1.0])
TABLE = os.path.join(os.path.dirname(__file__), "..", "..", "source", "dual_fr3_lab", "dual_fr3_lab", "assets", "pregrasp",
                     "plate_box410wide_pregrasp_gap004_margin005_snapj7.npz")


def rz(a, y):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, y], [0, 0, 1, ROOT_Z], [0, 0, 0, 1.0]])


MOUNT = {"left": rz(math.pi / 6, 0.215), "right": rz(-math.pi / 6, -0.215)}


def fk7(q):
    T = np.eye(4)
    for (a, d, al), th in zip(DH, q):
        ca, sa, ct, st = math.cos(al), math.sin(al), math.cos(th), math.sin(th)
        T = T @ np.array([[ct, -st, 0.0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1.0]])
    return T


def tcp(side, q):
    T = MOUNT[side] @ fk7(q)
    return T[:3, 3] + T[:3, :3] @ TCP, T[:3, :3]


def rot_err(R, Rd):
    E = Rd @ R.T
    v = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    s, c = np.linalg.norm(v) / 2.0, (np.trace(E) - 1) / 2.0
    return v / (2 * s) * math.atan2(s, c) if s > 1e-9 else np.zeros(3)


def ik(side, q0, pd, Rd, margin=0.10, iters=500):
    q = q0.copy()
    lo, hi = LIMITS[:, 0] + margin, LIMITS[:, 1] - margin
    for _ in range(iters):
        p, R = tcp(side, q)
        e = np.concatenate([pd - p, rot_err(R, Rd)])
        if np.linalg.norm(e[:3]) < 1e-5 and np.linalg.norm(e[3:]) < 1e-4:
            break
        J = np.zeros((6, 7))
        for j in range(7):
            dq = np.zeros(7)
            dq[j] = 1e-6
            p2, R2 = tcp(side, q + dq)
            J[:3, j], J[3:, j] = (p2 - p) / 1e-6, rot_err(R, R2) / 1e-6
        Jp = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6))
        dq = Jp @ e + 0.1 * (np.eye(7) - Jp @ J) @ (q0 - q)  # null space: stay near the table row
        q = np.clip(q + np.clip(dq, -0.2, 0.2), lo, hi)
    p, R = tcp(side, q)
    return q, np.linalg.norm(pd - p), np.linalg.norm(rot_err(R, Rd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=TABLE)
    ap.add_argument("--dz", type=float, default=0.25, help="[m] plate TCP rise above the pre-grasp height")
    ap.add_argument("--gap", type=float, default=0.075, help="[m] each plate further out (plate gap +2 gap)")
    ap.add_argument("--box", type=float, nargs=2, default=(0.45, 0.0), help="nominal box centre x y (env frame)")
    args = ap.parse_args()
    d = np.load(args.table)
    names = [str(x) for x in d["joint_names"]]
    Q, B = d["joint_pos"], d["box_pose"]
    qs = {s: Q[:, [names.index(f"{s}_fr3_joint{i}") for i in range(1, 8)]] for s in ("left", "right")}
    err = []
    for k in range(len(B)):
        pl, _ = tcp("left", qs["left"][k])
        pr, _ = tcp("right", qs["right"][k])
        err.append([pl[1] - B[k, 1] - 0.205, -(pr[1] - B[k, 1]) - 0.205, pl[2] - B[k, 2], pr[2] - B[k, 2]])
    err = np.array(err)
    print("table FK check (plate gap L, R, height above box centre L, R) mean", err.mean(0).round(4), "std", err.std(0).round(4))
    k = int(np.argmin(np.hypot(B[:, 0] - args.box[0], B[:, 1] - args.box[1])))
    pl, Rl = tcp("left", qs["left"][k])
    print(f"seed row {k}: box {B[k, :3].round(4).tolist()}, left TCP {pl.round(4).tolist()}")
    half_y = 0.205 + err[k, 0]  # keep the row's own gap (4 cm) and height
    target = np.array([args.box[0], args.box[1] + half_y + args.gap, pl[2] + args.dz])
    ql, ep, er = ik("left", qs["left"][k], target, Rl)
    qr = MIRROR * ql
    qr[6] = math.pi / 2 - ql[6]
    pl, Rl = tcp("left", ql)
    pr, Rr = tcp("right", qr)
    margin = np.minimum(np.c_[ql, qr] - LIMITS[:, [0]], LIMITS[:, [1]] - np.c_[ql, qr])
    print(f"IK err pos {ep:.1e} m rot {er:.1e} rad")
    print(f"TCP left {pl.round(4).tolist()} normal {(Rl @ [0, 0, 1]).round(3).tolist()}")
    print(f"TCP right {pr.round(4).tolist()} normal {(Rr @ [0, 0, 1]).round(3).tolist()}")
    print(f"joint margin min left {margin[:, 0].min():.2f} (j{margin[:, 0].argmin() + 1}) right {margin[:, 1].min():.2f} (j{margin[:, 1].argmin() + 1})")
    print("HOVER_Q_LEFT =", tuple(round(float(v), 4) for v in ql))
    print("HOVER_Q_RIGHT =", tuple(round(float(v), 4) for v in qr))


if __name__ == "__main__":
    main()
