"""Preview the cabinet book-insert start state: pictures plus a hold test (zero actions) for the attachments.

    python scripts/tools/preview_cabinet_book_insert.py --headless --enable_cameras

Writes ``<out_dir>/cabinet_book_start_{front,side,top,left_hand,right_hand}.png`` and prints how far the book,
handle grasp and joints drift while the policy outputs zero for ``--hold_s`` seconds.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Cabinet-BookInsert-DualFR3-UMI-Play-v0")
parser.add_argument("--out_dir", default="logs/rule_based/cabinet_book")
parser.add_argument("--hold_s", type=float, default=2.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors.camera import CameraCfg  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_mul  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import dual_fr3_lab.tasks  # noqa: E402, F401
from dual_fr3_lab.robots import UMI_TCP_OFFSET_POS, UMI_TCP_OFFSET_ROT  # noqa: E402
from dual_fr3_lab.tasks.manager_based.cabinet_book_insert import cabinet_book_insert_env_cfg as C  # noqa: E402

_HX = C.CABINET_XY_B[0] - C.DOOR_IN_CABINET[0] - C.HANDLE_IN_DOOR[0]  # handle body (door closed), base frame
_HY = C.CABINET_XY_B[1] - C.HANDLE_IN_DOOR[1]
_BX, _BY, _BZ = C.BOOK_START_POS_B
VIEWS = {  # eye, target in base frame
    "front": ((1.9, -0.9, 0.9), (0.65, 0.0, 0.2)),
    "side": ((0.6, -1.8, 0.6), (0.6, 0.0, 0.2)),
    "top": ((0.55, 0.0, 2.2), (0.6, 0.0, 0.2)),
    "left_hand": ((_HX - 0.35, _HY + 0.4, _BZ + 0.3), (_HX, _HY, _BZ)),
    "right_hand": ((_BX - 0.35, _BY - 0.4, _BZ + 0.3), (_BX, _BY, _BZ)),
}


def T(x):
    return x.torch if hasattr(x, "torch") else x


cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
cfg.scene.num_envs = 1
for name in VIEWS:
    setattr(cfg.scene, f"cam_{name}", CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Cam_{name}", update_period=0.0, width=1280, height=960, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.01, 20.0)),
    ))
env = gym.make(args.task, cfg=cfg).unwrapped
env.reset()
dev = env.device
robot, cabinet, book = env.scene["robot"], env.scene["cabinet"], env.scene["book"]
origin = env.scene.env_origins[0]
base = origin + torch.tensor([0.0, 0.0, C.ROBOT_BASE_Z], device=dev)
off_p, off_q = torch.tensor(UMI_TCP_OFFSET_POS, device=dev), torch.tensor(UMI_TCP_OFFSET_ROT, device=dev)


def body(asset, name):
    i = asset.body_names.index(name)
    return T(asset.data.body_pos_w)[0, i], T(asset.data.body_quat_w)[0, i]


def frames():
    """World (pos, quat) of each attachment frame, from both sides of the joint."""
    out = {}
    for side, other, lp, lq in (
        ("left", body(cabinet, "handle"), C.LEFT_GRASP_POS, C.LEFT_GRASP_ROT),
        ("right", (T(book.data.root_pos_w)[0], T(book.data.root_quat_w)[0]), C.RIGHT_GRASP_POS, C.RIGHT_GRASP_ROT),
    ):
        p7, q7 = body(robot, f"{side}_fr3_link7")
        tcp = p7 + quat_apply(q7, off_p)
        obj = other[0] + quat_apply(other[1], torch.tensor(lp, device=dev))
        out[side] = (tcp, obj)
    return out


def report(tag):
    f = frames()
    q = T(robot.data.joint_pos)[0]
    door = T(cabinet.data.joint_pos)[0, 0].item()
    bp = (T(book.data.root_pos_w)[0] - base).cpu().numpy()
    print(f"[preview] {tag}: door {np.degrees(door):+.2f} deg | book (base) {np.round(bp, 4).tolist()} | "
          f"TCP-grasp gap left {100 * (f['left'][0] - f['left'][1]).norm():.2f} cm, right {100 * (f['right'][0] - f['right'][1]).norm():.2f} cm")
    return q.clone(), bp


q_start = T(robot.data.joint_pos)[0].clone()


def hold(seconds):
    """Physics steps with the start joint angles as absolute PD targets (no action term)."""
    for _ in range(int(seconds / env.physics_dt)):
        robot.set_joint_position_target(q_start.unsqueeze(0))
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)


report("after reset (buffers)")
hold(0.05)
os.makedirs(args.out_dir, exist_ok=True)
for name, (eye, tgt) in VIEWS.items():
    cam = env.scene[f"cam_{name}"]
    cam.set_world_poses_from_view(torch.tensor([eye], device=dev) + base, torch.tensor([tgt], device=dev) + base)
for _ in range(3):
    env.sim.render()
    env.scene.update(env.physics_dt)
for name in VIEWS:
    img = T(env.scene[f"cam_{name}"].data.output["rgb"])[0].cpu().numpy()[..., :3].astype(np.uint8)
    path = os.path.join(args.out_dir, f"cabinet_book_start_{name}.png")
    Image.fromarray(img).save(path)
    print("[preview] wrote", path)

q0, b0 = report("start (0.05 s held)")
hold(args.hold_s)
q1, b1 = report(f"after {args.hold_s:.1f} s absolute hold")
print(f"[preview] absolute hold: joint drift max {np.degrees((q1 - q0).abs().max().item()):.2f} deg "
      f"(joint {robot.joint_names[int((q1 - q0).abs().argmax())]}), book drift {100 * np.linalg.norm(b1 - b0):.2f} cm")
env.reset()
hold(0.05)
q0, b0 = report("start again")
zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=dev)
for _ in range(int(args.hold_s / env.step_dt)):
    env.step(zero)
q1, b1 = report(f"after {args.hold_s:.1f} s zero action (relative contract)")
print(f"[preview] zero action: joint drift max {np.degrees((q1 - q0).abs().max().item()):.2f} deg "
      f"(joint {robot.joint_names[int((q1 - q0).abs().argmax())]}), book drift {100 * np.linalg.norm(b1 - b0):.2f} cm")
env.close()
app.close()
