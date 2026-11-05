"""Events for the cabinet book-insert task: hand attachments (prestartup) and reset to the solved start table."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from pxr import Gf, Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg


def _find_prim(root: Usd.Prim, name: str) -> Usd.Prim:
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim
    raise RuntimeError(f"no rigid body named {name!r} under {root.GetPath()}")


def attach_fixed_joint(
    env,
    env_ids,
    joint_name: str,
    body0: tuple[str, str],
    body1: tuple[str, str],
    local_pos0: tuple[float, float, float],
    local_rot0: tuple[float, float, float, float],
    local_pos1: tuple[float, float, float] = (0.0, 0.0, 0.0),
    local_rot1: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
):
    """prestartup: author a fixed joint between two rigid bodies in every env (outside both articulations).

    ``body0``/``body1`` are (scene prim name under the env, rigid body name), e.g. ("Robot", "left_fr3_link7").
    Local frames are in each body's frame, quaternions (x, y, z, w). The two frames must coincide at the start pose,
    so the solved start table (``reset_to_start_table``) and these frames come from the same grasp definition.
    """
    stage = sim_utils.get_current_stage()
    for i in range(env.num_envs):
        env_path = f"/World/envs/env_{i}"
        prims = [_find_prim(stage.GetPrimAtPath(f"{env_path}/{scope}"), name) for scope, name in (body0, body1)]
        joint = UsdPhysics.FixedJoint.Define(stage, f"{env_path}/{joint_name}")
        joint.CreateBody0Rel().SetTargets([prims[0].GetPath()])
        joint.CreateBody1Rel().SetTargets([prims[1].GetPath()])
        for set_pos, set_rot, pos, rot in (
            (joint.CreateLocalPos0Attr, joint.CreateLocalRot0Attr, local_pos0, local_rot0),
            (joint.CreateLocalPos1Attr, joint.CreateLocalRot1Attr, local_pos1, local_rot1),
        ):
            set_pos(Gf.Vec3f(*pos))
            set_rot(Gf.Quatf(rot[3], Gf.Vec3f(*rot[:3])))
        joint.CreateExcludeFromArticulationAttr(True)


@lru_cache(maxsize=4)
def load_start_pose(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def reset_to_start_table(
    env,
    env_ids: torch.Tensor,
    table_path: str,
    sequential: bool = False,
    fixed_row: int | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cabinet_cfg: SceneEntityCfg = SceneEntityCfg("cabinet"),
    book_cfg: SceneEntityCfg = SceneEntityCfg("book"),
):
    """reset: closed door, robot joints (arms + fingers) and the attached book at one row of the start table.

    ``table_path`` is the NPZ from ``scripts/tools/solve_cabinet_book_start.py``: ``joint_names``, ``joint_pos``
    [rows, joints] and ``book_pose_env`` [rows, 7] (book root pose relative to the env origin, quat xyzw), one row per
    book start on the table. Rows are drawn uniformly (``sequential``: env i gets row i % rows; ``fixed_row``: every env
    gets that row; both for evaluation).
    The joints are not perturbed: the welded hands must match the book and the handle exactly.
    The drawn row is kept in ``env.book_start_row`` (rule-based null-space reference, logging).
    """
    robot, cabinet, book = env.scene[robot_cfg.name], env.scene[cabinet_cfg.name], env.scene[book_cfg.name]
    table = load_start_pose(table_path)
    if not hasattr(env, "book_start_row"):
        env.book_start_row = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._start_table = {
            k: torch.tensor(table[k], dtype=torch.float32, device=env.device) for k in ("joint_pos", "book_pose_env")
        }
        env._start_joint_ids = [robot.joint_names.index(str(nm)) for nm in table["joint_names"]]
    rows = len(table["joint_pos"])
    n = len(env_ids)
    if fixed_row is not None:
        row = torch.full((n,), fixed_row, dtype=torch.long, device=env.device)
    elif sequential:
        row = env_ids.to(env.device).long() % rows
    else:
        row = torch.randint(0, rows, (n,), device=env.device)
    env.book_start_row[env_ids] = row
    q = env._start_table["joint_pos"][row]
    ids = env._start_joint_ids
    robot.write_joint_position_to_sim_index(position=q, joint_ids=ids, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=ids, env_ids=env_ids)
    robot.set_joint_position_target_index(target=q, joint_ids=ids, env_ids=env_ids)

    door = torch.zeros(n, cabinet.num_joints, device=env.device)
    cabinet.write_joint_position_to_sim_index(position=door, env_ids=env_ids)
    cabinet.write_joint_velocity_to_sim_index(velocity=door, env_ids=env_ids)

    pose = env._start_table["book_pose_env"][row].clone()
    pose[:, :3] += env.scene.env_origins[env_ids]
    book.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    book.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=env.device), env_ids=env_ids)
