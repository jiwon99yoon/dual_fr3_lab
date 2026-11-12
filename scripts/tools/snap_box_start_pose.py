"""Snapshot the episode start of a box lift task from three cameras (overview, front, side) and write a PNG grid.

Made for the -Curriculum task to look at the hover start (``-Play-Hover-v0``) next to the pre-grasp table start
(``-Play-Table-v0``). Each row of the grid is one env right after reset (``--settle`` zero-action steps later).

    OMNI_KIT_ACCEPT_EULA=YES python scripts/tools/snap_box_start_pose.py --headless --enable_cameras \
        --task Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-Curriculum-Play-Hover-v0 \
        --num_envs 3 --out /tmp/hover_start.png
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=3)
parser.add_argument("--settle", type=int, default=5, help="zero-action policy steps before the snapshot")
parser.add_argument("--size", type=int, nargs=2, default=(640, 480))
parser.add_argument("--out", required=True)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
os.environ.setdefault("DR_VIS", "0")
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors.camera import CameraCfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import dual_fr3_lab.tasks  # noqa: E402,F401

VIEWS = {
    "overview": ((2.3, -1.9, 1.9), (0.40, 0.0, 0.45)),
    "front (robot L = image R)": ((2.7, 0.0, 1.0), (0.45, 0.0, 0.50)),
    "side (robot right)": ((0.45, -2.8, 0.75), (0.45, 0.0, 0.50)),
}

cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
cfg.scene.num_envs = args.num_envs
cfg.scene.env_spacing = 8.0
cfg.sim.device = args.device
cfg.seed = args.seed
for k in range(len(VIEWS)):
    setattr(cfg.scene, f"rec_{k}", CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/RecCam_{k}", update_period=0.0, width=args.size[0], height=args.size[1],
        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 20.0)),
    ))
env = gym.make(args.task, cfg=cfg).unwrapped
env.reset()
cams = [env.scene[f"rec_{k}"] for k in range(len(VIEWS))]
for cam, (eye, lookat) in zip(cams, VIEWS.values()):
    cam.set_world_poses_from_view(torch.tensor([eye], device=env.device) + env.scene.env_origins,
                                  torch.tensor([lookat], device=env.device) + env.scene.env_origins)
for _ in range(args.settle):
    env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))
for _ in range(5):
    env.sim.render()
for cam in cams:
    cam.update(0.0)

ee = env.scene["ee_frame"].data.target_pos_w.torch - env.scene.env_origins[:, None]
box = env.scene["object"].data.root_pos_w.torch - env.scene.env_origins
goal = env.command_manager.get_command("object_pose")[:, :3]
rgb = [cam.data.output["rgb"].torch[..., :3].cpu().numpy().astype(np.uint8) for cam in cams]
W, H = args.size
grid = Image.new("RGB", (W * len(VIEWS), (H + 44) * env.num_envs), "white")
for i in range(env.num_envs):
    y0 = i * (H + 44)
    for k, (name, r) in enumerate(zip(VIEWS, rgb)):
        grid.paste(Image.fromarray(r[i]), (k * W, y0 + 44))
        ImageDraw.Draw(grid).text((k * W + 6, y0 + 26), name, fill="black")
    txt = (f"env {i}: box ({box[i, 0]:.2f}, {box[i, 1]:.2f}, {box[i, 2]:.2f})  plate L ({ee[i, 0, 0]:.2f}, {ee[i, 0, 1]:.2f}, {ee[i, 0, 2]:.2f})"
           f"  R ({ee[i, 1, 0]:.2f}, {ee[i, 1, 1]:.2f}, {ee[i, 1, 2]:.2f})  goal base ({goal[i, 0]:.2f}, {goal[i, 1]:.2f}, {goal[i, 2]:.2f})")
    ImageDraw.Draw(grid).text((6, y0 + 6), txt, fill="blue")
    print(txt)
grid.save(args.out)
print("[snap] wrote", args.out)
env.close()
app.close()
