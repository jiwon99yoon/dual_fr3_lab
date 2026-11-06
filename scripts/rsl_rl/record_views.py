"""Record lift, handover, or cabinet-book checkpoint behavior from three cameras side by side.

xhand_lab `scripts/nut_bolt/record_views.py`를 이 저장소 task에 맞춘 것이다. 창 없이 1 env를 돌리며
overview, front, side 세 시점을 가로로 붙여 mp4로 저장한다. 상단 HUD에 task별 성능 지표를 표시하고,
접촉력(box는 판 높이 차, handover는 단계와 양손 파지력)을 표시하고, 에피소드마다 종료 이유와 요약을 출력한다.
정책 주기 그대로 저장하므로 실제 속도로 재생된다(Sim2Real 20 fps, P47 50 fps).

    python scripts/rsl_rl/record_views.py --task Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Offset03-Play-v0 \\
        --checkpoint logs/rsl_rl/dual_fr3_plate_lift_sim2real_100hz_offset03/<run>/model_2000.pt --episodes 3

Writes <checkpoint dir>/videos/<checkpoint>_views_ep<N>.mp4 (and .json summary) unless ``--out`` is given.
handover는 성공 단계에 처음 도달하고 ``--tail_s``초 뒤에 에피소드를 끊는다 (성공 종료가 없어 12 s를 다 채우면
성공(~2 s) 뒤 10 s가 가만히 있는 장면이다). 끝까지 보려면 ``--tail_s -1``.
DR 범위 판(파랑/초록)은 들어 올린 박스를 가리므로 기본으로 끈다(`DR_VIS=1`로 켬).
박스 관측 지연·끊김 task(-BoxDrop)는 HUD에 비전 상태(OK / DROP, 지연 ms, 정책이 보는 박스 위치 오차)를 붙이고
끊긴 step은 HUD 줄을 빨갛게 칠한다. ``--box_drop_at S``는 에피소드마다 S초부터 끝까지 끊기게 한다.
``--only_drops``는 들었다가 놓친 에피소드(최대 들림 DROP_LIFT 이상 -> 그보다 DROP_FALL 아래로 내려감)만 저장한다.
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record a dual FR3 lift checkpoint from three cameras.")
parser.add_argument("--task", required=True, help="cube/box lift or cube handover task id (-Play-v0 recommended)")
parser.add_argument("--checkpoint", required=True, help="path to model_<N>.pt")
parser.add_argument("--episodes", type=int, default=3, help="stop after this many finished episodes")
parser.add_argument("--max_steps", type=int, default=2000, help="stop here even if fewer episodes finished")
parser.add_argument("--warmup", type=int, default=5, help="unrecorded steps before the first episode (renderer)")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default=None, help="output .mp4 path")
parser.add_argument("--tail_s", type=float, default=2.0,
                    help="handover: end each episode this many seconds after it first succeeds (<0: full episode)")
parser.add_argument("--box_drop_at", type=float, default=None,
                    help="-BoxDrop: force the box pose observation to drop from this time [s] to the episode end")
parser.add_argument("--only_drops", action="store_true",
                    help="keep only episodes where the lifted object was dropped (--episodes counts kept ones)")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True)
args_cli = parser.parse_args()
os.environ.setdefault("DR_VIS", "0")
simulation_app = AppLauncher(args_cli).app

import importlib.metadata as metadata  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors.camera import CameraCfg  # noqa: E402
from isaaclab.utils.math import combine_frame_transforms, quat_apply  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import dual_fr3_lab.tasks  # noqa: E402,F401

# (eye, lookat) relative to the env origin [m]. Robot base at the origin facing +x, left arm on +y.
# front는 로봇을 마주 보므로 로봇 왼팔이 화면 오른쪽에 온다.
VIEWS = {
    "box": {
        "overview": ((2.3, -1.9, 1.9), (0.40, 0.0, 0.45)),
        "front": ((2.7, 0.0, 1.0), (0.45, 0.0, 0.50)),
        "side": ((0.45, -2.8, 0.75), (0.45, 0.0, 0.50)),
    },
    "cube": {
        "overview": ((1.9, -1.6, 1.8), (0.40, -0.15, 0.85)),
        "front": ((1.8, -0.25, 1.1), (0.45, -0.25, 0.90)),
        "side": ((0.45, -1.6, 1.0), (0.45, -0.25, 0.90)),
    },
    # 테이블 양끝(y ±0.35)과 중앙 핸드오버 지점이 다 보이게
    "handover": {
        "overview": ((2.0, -1.5, 1.9), (0.45, 0.0, 0.85)),
        "front": ((2.1, 0.0, 1.25), (0.45, 0.0, 0.88)),
        "top": ((0.05, 0.0, 2.1), (0.50, 0.0, 0.80)),
    },
    # Cabinet is in front of the robot (+x). The first view keeps the usual
    # three-quarter overview, the second looks from the robot toward the door,
    # and the third shows the book/cabinet lane from above.
    "cabinet_book": {
        # Near the base-frame midpoint between the arms; look straight into the cabinet to expose insertion depth.
        "base_front": ((0.0, 0.0, 1.15), (0.95, 0.10, 0.64)),
        "robot_side": ((0.40, -1.9, 1.35), (0.62, 0.0, 0.88)),
        "top": ((0.40, 0.0, 3.0), (0.40, 0.0, 0.65)),
    },
}
LABELS = {"overview": "overview", "front": "front (robot L = image R)", "side": "side (robot right)",
          "top": "top (robot L = image L)", "robot_side": "robot side -> cabinet",
          "base_front": "base-frame front -> cabinet"}
# contact sensors whose filter 0 is the object: (left, right). handover: per hand, both pads
CONTACTS = {
    "box": ("left_plate_object_contact", "right_plate_object_contact"),
    "cube": ("leftfinger_object_contact", "rightfinger_object_contact"),
    "handover": (("left_leftfinger_contact", "left_rightfinger_contact"),
                 ("right_leftfinger_contact", "right_rightfinger_contact")),
}
GOAL_COMMAND = {"box": "object_pose", "cube": "object_pose", "handover": "goal_pose"}
REST_STEPS = 10  # per episode, the object's rest height is its lowest z over these first steps (spawn drop)
LIFTED = 0.05  # [m] "while lifted" threshold for the episode summary
DROP_LIFT = 0.10  # [m] --only_drops: the object must have been lifted at least this high
DROP_FALL = 0.08  # [m] ... and then fallen this far below its max lift


def _t(x):
    return x.torch if hasattr(x, "torch") else x


def label(frame: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw, font = ImageDraw.Draw(img), ImageFont.load_default(size=20)
    # bottom centre of the panel
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    xy = ((img.width - (right - left)) // 2 - left, img.height - (bottom - top) - 12 - top)
    draw.rectangle(draw.textbbox(xy, text, font=font), fill=(255, 255, 255))
    draw.text(xy, text, fill=(0, 0, 0), font=font)
    return np.asarray(img)


def hud(frame: np.ndarray, text: str, alert: bool = False) -> np.ndarray:
    """Full-width text strip above the panels (red when ``alert``)."""
    font = ImageFont.load_default(size=22)
    strip = Image.new("RGB", (frame.shape[1], 36), (150, 20, 20) if alert else (24, 24, 24))
    ImageDraw.Draw(strip).text((12, 6), text, fill=(240, 240, 240), font=font)
    return np.concatenate([np.asarray(strip), frame], axis=0)


def measure(raw, family: str) -> dict:
    """Env-0 quantities shown in the HUD (world frame, relative to the env origin)."""
    if family == "cabinet_book":
        from dual_fr3_lab.tasks.manager_based.cabinet_book_insert import cabinet_book_insert_env_cfg as C
        from dual_fr3_lab.tasks.manager_based.cabinet_book_insert.mdp.observations import (
            book_pose_b,
            contact_force_norms,
            door_angle_rad,
        )
        from dual_fr3_lab.tasks.manager_based.cabinet_book_insert.mdp.rewards import insert_success

        pos, quat = book_pose_b(raw)
        up = quat_apply(quat[:1], torch.tensor([[0.0, 0.0, 1.0]], device=raw.device))[0]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))
        success_params = raw.cfg.rewards.insert_success.params  # the task layout (Table75 moves the goal)
        goal = success_params["goal_pos_b"]
        m = {
            "z": float(pos[0, 2]),
            "tilt_deg": tilt,
            "door_deg": math.degrees(float(door_angle_rad(raw)[0])),
            "depth_cm": 100.0 * float(pos[0, 0] - success_params["opening_x_b"]),
            "lane_y_cm": 100.0 * float(pos[0, 1] - goal[1]),
            "lane_z_cm": 100.0 * float(pos[0, 2] - goal[2]),
            "success": bool(insert_success(raw, **success_params)[0]),
        }
        forces = contact_force_norms(raw, C.CONTACT_SENSORS)[0]
        for name, force in zip(C.CONTACT_SENSORS, forces):
            m[f"{name}_N"] = float(force)
        return m

    obj, robot = raw.scene["object"], raw.scene["robot"]
    pos = _t(obj.data.root_pos_w)[0] - raw.scene.env_origins[0]
    up = quat_apply(_t(obj.data.root_quat_w)[:1], torch.tensor([[0.0, 0.0, 1.0]], device=raw.device))[0]
    goal_b = raw.command_manager.get_command(GOAL_COMMAND[family])[:1, :3]
    goal_w, _ = combine_frame_transforms(_t(robot.data.root_pos_w)[:1], _t(robot.data.root_quat_w)[:1], goal_b)
    def pad_force(name):
        f = _t(raw.scene[name].data.force_matrix_w)[0, :, 0]  # (bodies, 3), filter 0 = object
        return float(torch.nan_to_num(f.norm(dim=-1)).max())

    # handover: 손마다 두 패드 중 약한 쪽 (파지 판정과 같은 기준)
    forces = [min(pad_force(n) for n in name) if isinstance(name, tuple) else pad_force(name)
              for name in CONTACTS[family]]
    m = {
        "z": float(pos[2]),
        "tilt_deg": math.degrees(math.acos(max(-1.0, min(1.0, float(up[2]))))),
        "goal_cm": 100.0 * float((_t(obj.data.root_pos_w)[0] - goal_w[0]).norm()),
        "force_l": forces[0],
        "force_r": forces[1],
    }
    if family == "box":
        ee = raw.scene["ee_frame"]
        names = ee.data.target_frame_names
        z = _t(ee.data.target_pos_w)[0, :, 2]
        m["plate_dz_cm"] = 100.0 * float(z[names.index("left_ee")] - z[names.index("right_ee")])
        st = raw.__dict__.get("_box_obs_dropout")
        if st is not None:
            from dual_fr3_lab.tasks.manager_based.lift_plate.joint_pos_env_cfg import mdp as lift_mdp

            true_pos = lift_mdp.object_position_in_robot_root_frame(raw)[0]
            m["vis_drop"] = bool(st["drop"][0])
            m["vis_delay_ms"] = 1000.0 * int(st["delay"][0]) * raw.step_dt
            m["vis_err_cm"] = 100.0 * float((st["held"]["object_position"][0] - true_pos).norm())
    if family == "handover":
        from dual_fr3_lab.tasks.manager_based.handover_cube.mdp import handover as H

        m["stage"] = int(H.handover_state(raw).stage[0])
    return m


def hud_text(ep: int, t: float, lift: float, m: dict, family: str) -> str:
    if family == "cabinet_book":
        return (f"ep {ep}  t {t:4.2f}s | door {m['door_deg']:5.1f} deg | depth {m['depth_cm']:+5.1f} cm"
                f" | lane dy {m['lane_y_cm']:+4.1f} dz {m['lane_z_cm']:+4.1f} cm"
                f" | tilt {m['tilt_deg']:4.1f} deg | success {'YES' if m['success'] else 'no'}")
    text = f"ep {ep}  t {t:4.2f}s | lift {100 * lift:+5.1f} cm | tilt {m['tilt_deg']:4.1f} deg | goal {m['goal_cm']:5.1f} cm"
    if family == "box":
        text += f" | plate dz(L-R) {m['plate_dz_cm']:+5.1f} cm | F plate L {m['force_l']:4.0f} R {m['force_r']:4.0f} N"
        if "vis_drop" in m:
            text += (f" | vision {'DROP' if m['vis_drop'] else 'ok  '} delay {m['vis_delay_ms']:3.0f} ms"
                     f" err {m['vis_err_cm']:4.1f} cm")
    elif family == "handover":
        from dual_fr3_lab.tasks.manager_based.handover_cube.mdp.handover import STAGE_NAMES

        text = (f"ep {ep}  t {t:4.2f}s | stage {m['stage']} {STAGE_NAMES[m['stage']]:<11} | lift {100 * lift:+5.1f} cm"
                f" | tilt {m['tilt_deg']:4.1f} deg | goal {m['goal_cm']:5.1f} cm"
                f" | grip L {m['force_l']:4.1f} R {m['force_r']:4.1f} N")
    else:
        text += f" | F finger L {m['force_l']:4.1f} R {m['force_r']:4.1f} N"
    return text


def dropped(steps: list[dict]) -> bool:
    peak = -math.inf
    for s in steps:
        peak = max(peak, s["lift"])
        if peak >= DROP_LIFT and s["lift"] < peak - DROP_FALL:
            return True
    return False


def summarize(steps: list[dict], dt: float, reasons: list[str], family: str) -> dict:
    if family == "cabinet_book":
        depth = [s["depth_cm"] for s in steps]
        contact_names = [k for k in steps[0] if k.startswith("contact_") and k.endswith("_N")]
        contact_peaks = {
            name: {
                "max_N": round(max(s[name] for s in steps), 2),
                "peak_time_s": round(max(range(len(steps)), key=lambda i: steps[i][name]) * dt, 2),
            }
            for name in contact_names
        }
        # Keep a compact trace around the action instead of emitting hundreds of JSON samples.
        trace_ids = sorted(set(range(0, len(steps), max(1, round(0.25 / dt)))) | {len(steps) - 1})
        trace = []
        for i in trace_ids:
            s = steps[i]
            sample = {"time_s": round((i + 1) * dt, 2), "depth_cm": round(s["depth_cm"], 1),
                      "tilt_deg": round(s["tilt_deg"], 1)}
            sample.update({name: round(s[name], 2) for name in contact_names})
            trace.append(sample)
        return {
            "termination": "+".join(reasons) or "?",
            "steps": len(steps),
            "duration_s": round(len(steps) * dt, 2),
            "final_door_deg": round(steps[-1]["door_deg"], 1),
            "max_depth_cm": round(max(depth), 1),
            "final_depth_cm": round(depth[-1], 1),
            "final_lane_error_cm": {"y": round(steps[-1]["lane_y_cm"], 1),
                                    "z": round(steps[-1]["lane_z_cm"], 1)},
            "max_tilt_deg": round(max(s["tilt_deg"] for s in steps), 1),
            "final_tilt_deg": round(steps[-1]["tilt_deg"], 1),
            "success": any(s["success"] for s in steps),
            "contact_peaks": contact_peaks,
            "contact_tilt_trace": trace,
        }

    lift = np.array([s["lift"] for s in steps])
    lifted = lift > LIFTED
    out = {
        "termination": "+".join(reasons) or "?",
        "steps": len(steps),
        "duration_s": round(len(steps) * dt, 2),
        "max_lift_cm": round(100 * float(lift.max()), 1),
        "lifted_s": round(float(lifted.sum()) * dt, 2),
        "min_goal_cm": round(min(s["goal_cm"] for s in steps), 1),
        "dropped": dropped(steps),
    }
    if lifted.any():
        held = [s for s, ok in zip(steps, lifted) if ok]
        out["tilt_deg_while_lifted"] = {"mean": round(float(np.mean([s["tilt_deg"] for s in held])), 1),
                                        "max": round(float(np.max([s["tilt_deg"] for s in held])), 1)}
        if family == "box":
            dz = np.array([s["plate_dz_cm"] for s in held])
            out["plate_dz_cm_while_lifted"] = {"mean": round(float(dz.mean()), 1),
                                               "mean_abs": round(float(np.abs(dz).mean()), 1)}
    if family == "box" and "vis_drop" in steps[0]:
        drop = np.array([s["vis_drop"] for s in steps])
        out["box_obs"] = {"delay_ms": round(steps[-1]["vis_delay_ms"]),
                          "drop_s": round(float(drop.sum()) * dt, 2),
                          "first_drop_s": round((int(drop.argmax()) + 1) * dt, 2) if drop.any() else None,
                          "max_err_cm": round(max(s["vis_err_cm"] for s in steps), 1)}
    if family == "handover":
        from dual_fr3_lab.tasks.manager_based.handover_cube.mdp.handover import STAGE_NAMES

        stages = [s["stage"] for s in steps]
        out["max_stage"] = STAGE_NAMES[max(stages)]
        # 각 단계에 처음 도달한 시각 [s]
        out["first_reached_s"] = {name: round((stages.index(k) + 1) * dt, 2)
                                  for k, name in enumerate(STAGE_NAMES) if k > 0 and k in stages}
        out["success_s"] = round(stages.count(len(STAGE_NAMES) - 1) * dt, 2)
    return out


def main():
    task = args_cli.task
    family = ("box" if "Lift-Box" in task else "cube" if "Lift-Cube" in task
              else "handover" if "Handover-Cube" in task
              else "cabinet_book" if "Cabinet-BookInsert" in task else None)
    if family is None:
        raise ValueError(f"unsupported task family for three-view recording: {task}")
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    for name in VIEWS[family]:
        setattr(env_cfg.scene, f"rec_{name}", CameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/RecCam_{name}", update_period=0.0,
            width=args_cli.width, height=args_cli.height, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 20.0)),
        ))
    env = gym.make(task, cfg=env_cfg)
    raw = env.unwrapped
    origin = raw.scene.env_origins[0]
    cams = {name: raw.scene[f"rec_{name}"] for name in VIEWS[family]}
    for name, (eye, lookat) in VIEWS[family].items():
        cams[name].set_world_poses_from_view(torch.tensor([eye], device=raw.device) + origin,
                                             torch.tensor([lookat], device=raw.device) + origin)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=raw.device)
    dt = raw.step_dt

    # a few unrecorded steps so the renderer has warmed up, then start from a fresh episode
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(args_cli.warmup):
            obs, _, _, _ = env.step(policy(obs))
        obs, _ = env.reset()

    frames, episodes, steps = [], [], []
    ep_frames, n_seen = [], 0
    step, z_rest = 0, math.inf
    success_stage = None
    if family == "handover":
        from dual_fr3_lab.tasks.manager_based.handover_cube.mdp.handover import STAGE_SUCCESS as success_stage
    t_success, cut = None, False
    while simulation_app.is_running() and len(episodes) < args_cli.episodes and step < args_cli.max_steps:
        st = raw.__dict__.get("_box_obs_dropout")
        if args_cli.box_drop_at is not None and st is not None:
            # 에피소드 시작 step에 뽑힌 full_at을 덮어쓴다 (다음 step의 mask 갱신부터 적용)
            st["full_at"][0] = round(args_cli.box_drop_at / dt)
        with torch.inference_mode():
            # without a visualizer (--viz none) Lab 3 never moves the goal marker; update it before this step renders
            raw.sim.vis_marker_registry.dispatch_callbacks()
            obs, _, dones, _ = env.step(policy(obs))
        step += 1
        m = measure(raw, family)
        if family == "cabinet_book":
            m["lift"] = 0.0  # the cabinet task HUD reports insertion metrics instead
        else:
            if len(steps) < REST_STEPS:
                z_rest = min(z_rest, m["z"])
            m["lift"] = m["z"] - z_rest
        steps.append(m)
        if success_stage is not None and args_cli.tail_s >= 0 and not cut:
            if t_success is None and m["stage"] == success_stage:
                t_success = len(steps)
            if t_success is not None and len(steps) - t_success >= round(args_cli.tail_s / dt):
                # 다음 step에서 time_out으로 끝나게 한다
                raw.episode_length_buf[0] = raw.max_episode_length - 1
                cut = True
        panels = []
        for name in VIEWS[family]:
            rgb = _t(cams[name].data.output["rgb"])[0, ..., :3].cpu().numpy().astype(np.uint8)
            panels.append(label(rgb, LABELS[name]))
        ep_frames.append(hud(np.concatenate(panels, axis=1), hud_text(n_seen + 1, len(steps) * dt, m["lift"],
                                                                   m, family), alert=m.get("vis_drop", False)))
        if bool(dones[0]):
            # measure() above read the post-reset state; the episode's last valid sample is the one before
            steps.pop()
            if ep_frames:
                ep_frames.pop()
            n_seen += 1
            tm = raw.termination_manager
            reasons = [name for name in tm.active_terms if bool(tm.get_term(name)[0])]
            if cut:
                reasons = [f"cut_{args_cli.tail_s:g}s_after_success"]
            summary = summarize(steps, dt, reasons, family) if steps else {"termination": "+".join(reasons)}
            print(f"[record] episode {n_seen}: {json.dumps(summary)}", flush=True)
            if not args_cli.only_drops or summary.get("dropped"):
                summary["episode"] = n_seen
                episodes.append(summary)
                frames += ep_frames
            ep_frames = []
            steps, z_rest = [], math.inf
            t_success, cut = None, False

    if not args_cli.only_drops:
        frames += ep_frames  # unfinished last episode (max_steps)
    if not frames:
        print(f"[record] no episode kept out of {n_seen} (--only_drops: none dropped)", flush=True)
        env.close()
        return
    ckpt = os.path.abspath(args_cli.checkpoint)
    out = args_cli.out or os.path.join(
        os.path.dirname(ckpt), "videos", f"{os.path.splitext(os.path.basename(ckpt))[0]}_views_ep{len(episodes)}.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    imageio.mimwrite(out, frames, fps=round(1.0 / dt), codec="libx264", quality=8, macro_block_size=1)
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump({"task": task, "checkpoint": ckpt, "seed": args_cli.seed, "episodes_run": n_seen,
                   "episodes": episodes}, f, indent=2)
    print(f"[record] {len(frames)} frames ({len(frames) * dt:.1f} s), {len(episodes)} episodes -> {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
