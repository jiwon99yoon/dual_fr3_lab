# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""dual_fr3 plate box-lift 정책의 강건성 평가 (sim2real 배포 전 관문).

동일 checkpoint를 Play-v2(corruption off)과 Play-Noisy-v2(obs noise on)에서 각각 굴려
성능 하락 폭을 잰다. 물성/크기 DR은 두 cfg 모두 활성(학습 분포). plate 고유 지표:
env별 판정 높이(안착중심+LIFT_DELTA), 양판 squeeze 유지율, 운반 중 박스 기울기,
drop 이벤트, P38의 0.5초 sustained success, yaw 10/15도 결합 성공률, 최초 좌우 접촉 시차,
P39의 actual-contact XOR 누적 지속시간, plate force/높이 비대칭을 함께 측정한다. Startup DR의
독립 표본 수를 늘리려면 episodes보다 num_envs를 늘리고, P38/P39 최종 난이도는
``--disturbance_scale 1.0``으로 명시한다.

.. code-block:: bash

    python scripts/tools/eval_dual_fr3_plate_robustness.py \
        --task Isaac-Lift-Box-DualFR3-Plate-Play-Noisy-v2 \
        --checkpoint logs/rsl_rl/dual_fr3_plate_lift_v2/<run>/model_9850.pt \
        --num_envs 128 --episodes 1 --seed 42 --disturbance_scale 1.0 --headless

    # 선택 env의 q/dq/raw action/contact/yaw/mirror error를 control-step CSV로 저장
    python scripts/tools/eval_dual_fr3_plate_robustness.py \
        --task Isaac-Lift-Box-DualFR3-Plate-P39-Play-v0 \
        --checkpoint logs/rsl_rl/dual_fr3_plate_lift_p39/<run>/model_9850.pt \
        --num_envs 32 --episodes 1 --seed 42 --disturbance_scale 1.0 \
        --joint_trace_csv /tmp/p39_joint_trace.csv --joint_trace_env_ids 0 --headless
"""

import argparse
import csv
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate dual_fr3 plate lift policy robustness.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Box-DualFR3-Plate-Play-Noisy-v2")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=4, help="에피소드 사이클 수 (사이클당 5s x num_envs)")
parser.add_argument("--seed", type=int, default=42, help="평가 env/DR seed. 음수면 cfg 기본값 사용")
parser.add_argument(
    "--disturbance_scale",
    type=float,
    default=None,
    help="v2 외란 scale을 이 값으로 고정(0~1 권장). 미지정 시 기존 curriculum 동작 유지",
)
parser.add_argument("--no_obs_noise", action="store_true", help="obs noise만 끄기 (물성 DR 유지)")
parser.add_argument(
    "--no_physics_dr",
    action="store_true",
    help="물성 DR만 끄기 (마찰/밀도/게인. 크기 DR은 유지, obs noise 유지)",
)
parser.add_argument(
    "--joint_trace_csv",
    type=str,
    default=None,
    help="지정하면 선택 env의 매 control-step joint/action/contact/yaw와 mirror error를 CSV로 저장",
)
parser.add_argument(
    "--joint_trace_env_ids",
    type=int,
    nargs="+",
    default=[0],
    help="--joint_trace_csv에 기록할 env index 목록 (기본: 0)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import dual_fr3_lab.tasks  # noqa: F401  # 이 패키지의 gym id 등록
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from dual_fr3_lab.tasks.manager_based.lift_cube import mdp
from dual_fr3_lab.tasks.manager_based.lift_plate.joint_pos_env_cfg import LIFT_DELTA, _box_halves
from dual_fr3_lab.tasks.manager_based.lift_plate.agents.rsl_rl_ppo_cfg import LiftPlateV2PPORunnerCfg
from isaaclab_tasks.utils import parse_env_cfg

SUCCESS_D = 0.10  # goal 10cm 이내면 "정밀 유지"로 카운트 (P7 success 설계와 동일 기준)
UPRIGHT_TILT = 0.034  # 1 - cos(15도): 이보다 작으면 "수평"
YAW_ALIGN = 0.2618  # 15도(rad): 박스 x축이 base x축과 이보다 가까우면 "정렬" [0804 똑바로들기]
YAW_STRICT = 0.174533  # 10도(rad): P38 목표
SUCCESS_HOLD_STEPS = 25  # 0.5s @ 50Hz
CONTACT_THRESHOLD = 1.0  # N, 학습의 bilateral grasp gate와 동일
OBJECT_FILTER_IDX = 0


def _object_contact_force(sensor, filter_idx: int = OBJECT_FILTER_IDX) -> torch.Tensor:
    """Object-filtered per-env contact-force magnitude, matching the P39 reward semantics."""
    matrix = sensor.data.force_matrix_w
    if matrix is None or matrix.ndim != 4:
        shape = None if matrix is None else tuple(matrix.shape)
        raise RuntimeError(f"{sensor.cfg.prim_path}: expected force_matrix_w (N,B,M,3), got {shape}")
    if filter_idx < 0 or filter_idx >= matrix.shape[2]:
        raise IndexError(f"{sensor.cfg.prim_path}: filter_idx={filter_idx}, num_filters={matrix.shape[2]}")
    return torch.norm(matrix[:, :, filter_idx, :], dim=-1).amax(dim=-1)


def main():
    eval_device = args_cli.device
    env_cfg = parse_env_cfg(args_cli.task, device=eval_device, num_envs=args_cli.num_envs)
    if args_cli.seed >= 0:
        env_cfg.seed = args_cli.seed
    # 축별 진단: noisy cfg에서 한 축만 끄고 평가 (원인 분리용)
    if args_cli.no_obs_noise:
        env_cfg.observations.policy.enable_corruption = False
    if args_cli.no_physics_dr:
        for name in ("randomize_box_friction", "randomize_box_density", "randomize_actuator_gains"):
            if hasattr(env_cfg.events, name):
                delattr(env_cfg.events, name)
    # checkpoint는 custom env curriculum state를 저장하지 않는다. 명시값이 있으면 episode 사이에
    # disturbance curriculum이 바뀌지 않도록 term을 제거하고 아래에서 physics scale을 고정한다.
    if args_cli.disturbance_scale is not None and hasattr(env_cfg.curriculum, "disturbance"):
        delattr(env_cfg.curriculum, "disturbance")
    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env)

    agent_dict = LiftPlateV2PPORunnerCfg().to_dict()
    # rsl_rl 버전 호환: isaaclab_rl(2.3+) cfg의 미지 키(optimizer 등)를 PPO 시그니처 기준 제거
    try:
        import inspect as _inspect

        from rsl_rl.algorithms import PPO as _PPO

        _allowed = set(_inspect.signature(_PPO.__init__).parameters) | {"class_name"}
        _alg = agent_dict.get("algorithm", {})
        _dropped = [k for k in list(_alg) if k not in _allowed]
        for _k in _dropped:
            _alg.pop(_k)
        if _dropped:
            print(f"[compat] rsl_rl PPO가 모르는 cfg 키 제거: {_dropped}")
    except Exception:
        pass
    runner = OnPolicyRunner(wrapped, agent_dict, log_dir=None, device=eval_device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=eval_device)

    uenv = env.unwrapped
    if args_cli.disturbance_scale is not None:
        uenv._disturb_scale = float(args_cli.disturbance_scale)
    f1 = SceneEntityCfg("left_plate_object_contact")
    f2 = SceneEntityCfg("right_plate_object_contact")
    left_sensor = uenv.scene.sensors[f1.name]
    right_sensor = uenv.scene.sensors[f2.name]
    for sensor in (left_sensor, right_sensor):
        filters = sensor.cfg.filter_prim_paths_expr
        if len(filters) <= OBJECT_FILTER_IDX or "Object" not in filters[OBJECT_FILTER_IDX]:
            raise RuntimeError(f"Object must be contact filter index {OBJECT_FILTER_IDX}: {filters}")

    robot_asset = uenv.scene["robot"]
    joint_names = robot_asset.data.joint_names
    left_joint_ids = [i for i, name in enumerate(joint_names) if name.startswith("left_fr3_joint")]
    right_joint_ids = [i for i, name in enumerate(joint_names) if name.startswith("right_fr3_joint")]
    left_joint_ids.sort(key=lambda i: int(joint_names[i].rsplit("joint", 1)[1]))
    right_joint_ids.sort(key=lambda i: int(joint_names[i].rsplit("joint", 1)[1]))
    if len(left_joint_ids) != 7 or len(right_joint_ids) != 7:
        raise RuntimeError(f"Expected 7 joints per arm, got L={left_joint_ids}, R={right_joint_ids}")

    # Base-y reflection for the mirrored dual-FR3 mounts. A symmetric pose satisfies
    # q_R = [-q_L1, +q_L2, -q_L3, +q_L4, -q_L5, +q_L6, -q_L7].
    mirror_signs = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], device=eval_device)

    # Policy action columns follow the action term's resolved joint order. Do not assume L1..L7,R1..R7.
    arm_action_term = uenv.action_manager.get_term("arm_action")
    action_joint_names = list(arm_action_term._joint_names)
    action_name_to_idx = {name: idx for idx, name in enumerate(action_joint_names)}
    left_action_ids = [action_name_to_idx[joint_names[idx]] for idx in left_joint_ids]
    right_action_ids = [action_name_to_idx[joint_names[idx]] for idx in right_joint_ids]

    trace_env_ids = sorted(set(args_cli.joint_trace_env_ids)) if args_cli.joint_trace_csv else []
    invalid_trace_ids = [idx for idx in trace_env_ids if idx < 0 or idx >= args_cli.num_envs]
    if invalid_trace_ids:
        raise ValueError(f"joint trace env ids out of range [0,{args_cli.num_envs}): {invalid_trace_ids}")
    trace_rows = []

    steps_per_ep = int(uenv.max_episode_length)
    total_steps = steps_per_ep * args_cli.episodes
    obs = wrapped.get_observations()

    num_envs = args_cli.num_envs
    lifted_sum = grasp_sum = hold_sum = success_sum = upright_hold_sum = 0.0
    yaw_hold_sum = sustained_base_sum = sustained_yaw10_sum = sustained_yaw15_sum = 0.0
    hold_sample_count = 0
    err_hold_sum = tilt_hold_sum = yaw_hold_err_sum = grip_off_hold_sum = 0.0
    force_samples = []
    force_imbalance_samples = []
    plate_height_diff_samples = []
    contact_delta_samples = []
    one_sided_duration_samples = []
    grasp_loss_count_samples = []
    bilateral_joint_path_left_samples = []
    bilateral_joint_path_right_samples = []
    mirror_error_hold_samples = []
    left_joint_hold_samples = []
    right_joint_hold_samples = []
    tts_base_samples = []
    tts_yaw10_samples = []
    tts_yaw15_samples = []

    # Success는 post-step scene을, contact timing은 auto-reset 편향을 피하려고 pre-step scene을 계측한다.
    episode_step = torch.zeros(num_envs, dtype=torch.long, device=eval_device)
    episode_index = torch.zeros(num_envs, dtype=torch.long, device=eval_device)
    contact_step = torch.zeros(num_envs, dtype=torch.long, device=eval_device)
    first_left_contact = torch.full((num_envs,), -1, dtype=torch.long, device=eval_device)
    first_right_contact = torch.full((num_envs,), -1, dtype=torch.long, device=eval_device)
    one_sided_steps = torch.zeros(num_envs, dtype=torch.long, device=eval_device)
    union_contact_steps = torch.zeros(num_envs, dtype=torch.long, device=eval_device)
    bilateral_contact_steps = torch.zeros(num_envs, dtype=torch.long, device=eval_device)
    ep_had_bilateral = torch.zeros(num_envs, dtype=torch.bool, device=eval_device)
    ep_grasp_loss_count = torch.zeros(num_envs, dtype=torch.long, device=eval_device)
    bilateral_joint_path_left = torch.zeros((num_envs, 7), device=eval_device)
    bilateral_joint_path_right = torch.zeros((num_envs, 7), device=eval_device)
    prev_joint_pos = robot_asset.data.joint_pos.clone()
    base_streak = torch.zeros(num_envs, dtype=torch.long, device=eval_device)
    yaw10_streak = torch.zeros(num_envs, dtype=torch.long, device=eval_device)
    yaw15_streak = torch.zeros(num_envs, dtype=torch.long, device=eval_device)
    ep_grasp = torch.zeros(num_envs, dtype=torch.bool, device=eval_device)
    ep_lift = torch.zeros_like(ep_grasp)
    ep_near = torch.zeros_like(ep_grasp)
    ep_base_success = torch.zeros_like(ep_grasp)
    ep_yaw10_success = torch.zeros_like(ep_grasp)
    ep_yaw15_success = torch.zeros_like(ep_grasp)

    completed_episodes = timeouts = out_of_bounds = 0
    episodes_grasp = episodes_lift = episodes_near = 0
    episodes_base_success = episodes_yaw10_success = episodes_yaw15_success = 0
    contact_pairs = contact_within_1 = contact_within_2 = contact_within_5 = 0
    left_first = right_first = tied_first = 0
    left_only = right_only = no_contact = 0
    bilateral_episodes = episodes_with_grasp_loss = 0
    xor_total_steps = union_total_steps = bilateral_total_steps = 0
    valid_post_samples = 0
    drops = grasp_loss_events = 0
    prev_hold = torch.zeros(num_envs, dtype=torch.bool, device=eval_device)
    prev_sample_hold = torch.zeros(num_envs, dtype=torch.bool, device=eval_device)
    for step in range(total_steps):
        # Contact metrics are sampled before stepping because post-step done envs have already auto-reset.
        left_force_pre = _object_contact_force(left_sensor)
        right_force_pre = _object_contact_force(right_sensor)
        left_pre = left_force_pre > CONTACT_THRESHOLD
        right_pre = right_force_pre > CONTACT_THRESHOLD
        xor_pre = torch.logical_xor(left_pre, right_pre)
        union_pre = left_pre | right_pre
        bilateral_pre = left_pre & right_pre
        bilateral_phase_before = ep_had_bilateral.clone()
        contact_step += 1
        new_left = left_pre & (first_left_contact < 0)
        new_right = right_pre & (first_right_contact < 0)
        first_left_contact[new_left] = contact_step[new_left]
        first_right_contact[new_right] = contact_step[new_right]
        one_sided_steps += xor_pre.long()
        union_contact_steps += union_pre.long()
        bilateral_contact_steps += bilateral_pre.long()
        xor_total_steps += int(xor_pre.sum().item())
        union_total_steps += int(union_pre.sum().item())
        bilateral_total_steps += int(bilateral_pre.sum().item())

        # Once an actual same-step bilateral contact occurred, track joint travel (not net displacement).
        joint_pos_pre = robot_asset.data.joint_pos
        joint_travel = (joint_pos_pre - prev_joint_pos).abs()
        bilateral_joint_path_left += joint_travel[:, left_joint_ids] * bilateral_phase_before.unsqueeze(-1).float()
        bilateral_joint_path_right += joint_travel[:, right_joint_ids] * bilateral_phase_before.unsqueeze(-1).float()
        prev_joint_pos = joint_pos_pre.clone()
        ep_had_bilateral |= bilateral_pre

        obj_pre = uenv.scene["object"]
        lifted_pre = obj_pre.data.root_pos_w[:, 2] > _box_halves(uenv)[1] + LIFT_DELTA
        hold_pre = lifted_pre & bilateral_pre
        loss_now = prev_sample_hold & lifted_pre & ~bilateral_pre
        grasp_loss_events += int(loss_now.sum().item())
        ep_grasp_loss_count += loss_now.long()
        prev_sample_hold = hold_pre

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = wrapped.step(actions)
        dones_b = dones.bool()
        active = ~dones_b
        robot = uenv.scene["robot"]
        obj = uenv.scene["object"]
        # goal (base frame) -> world
        cmd = uenv.command_manager.get_command("object_pose")[:, :3]
        goal_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, cmd)
        err = torch.norm(goal_w - obj.data.root_pos_w, dim=1)
        # env별 판정 높이 (안착 중심 + LIFT_DELTA)
        half_z = _box_halves(uenv)[1]
        lifted = (obj.data.root_pos_w[:, 2] > half_z + LIFT_DELTA).float()
        grasped = mdp.object_grasped(
            uenv, f1, f2, threshold=CONTACT_THRESHOLD, filter_idx=OBJECT_FILTER_IDX
        )
        hold = lifted * grasped
        left_force = _object_contact_force(left_sensor)
        right_force = _object_contact_force(right_sensor)
        # 박스 기울기 (1 - z_box·z_world)
        z_axis = torch.zeros_like(obj.data.root_pos_w)
        z_axis[:, 2] = 1.0
        tilt = 1.0 - quat_apply(obj.data.root_quat_w, z_axis)[:, 2]
        # 박스 yaw 정렬 오차 [0804]: 박스 로컬 x축 vs base 로컬 x축의 수평면 각도 (0=완전 정렬)
        x_axis = torch.zeros_like(obj.data.root_pos_w)
        x_axis[:, 0] = 1.0
        bx = quat_apply(obj.data.root_quat_w, x_axis)  # 박스 x (world)
        rx = quat_apply(robot.data.root_quat_w, x_axis)  # base x (world)
        yaw_signed = torch.atan2(
            bx[:, 0] * rx[:, 1] - bx[:, 1] * rx[:, 0], bx[:, 0] * rx[:, 0] + bx[:, 1] * rx[:, 1]
        )
        yaw_err = yaw_signed.abs()
        # [진단] grip 축은 무방향이므로 raw 0/180도를 동일 정렬로 보고 modulo-pi offset을 사용한다.
        ee = uenv.scene["ee_frame"]
        _names = ee.data.target_frame_names
        _li, _ri = _names.index("left_ee"), _names.index("right_ee")
        grip = ee.data.target_pos_w[:, _ri] - ee.data.target_pos_w[:, _li]  # 오른-왼 (world)
        y_axis = torch.zeros_like(obj.data.root_pos_w)
        y_axis[:, 1] = 1.0
        ry = quat_apply(robot.data.root_quat_w, y_axis)  # base +y (world)
        grip_yaw_raw = torch.atan2(
            grip[:, 0] * ry[:, 1] - grip[:, 1] * ry[:, 0], grip[:, 0] * ry[:, 0] + grip[:, 1] * ry[:, 1]
        ).abs()
        grip_off = torch.minimum(grip_yaw_raw, torch.pi - grip_yaw_raw)
        plate_height_diff = (ee.data.target_pos_w[:, _li, 2] - ee.data.target_pos_w[:, _ri, 2]).abs()
        q_left_post = robot.data.joint_pos[:, left_joint_ids]
        q_right_post = robot.data.joint_pos[:, right_joint_ids]
        q_right_mirror_expected = q_left_post * mirror_signs
        mirror_error_post = torch.atan2(
            torch.sin(q_right_post - q_right_mirror_expected),
            torch.cos(q_right_post - q_right_mirror_expected),
        )

        valid_post_samples += int(active.sum().item())
        lifted_sum += lifted[active].sum().item()
        grasp_sum += grasped[active].sum().item()
        hold_sum += hold[active].sum().item()
        success_sum += (hold * (err < SUCCESS_D).float())[active].sum().item()
        upright_hold_sum += (hold * (tilt < UPRIGHT_TILT).float())[active].sum().item()
        yaw_hold_sum += (hold * (yaw_err < YAW_ALIGN).float())[active].sum().item()
        hold_b = hold.bool() & active
        if hold_b.any():
            count = int(hold_b.sum().item())
            hold_sample_count += count
            err_hold_sum += err[hold_b].sum().item()
            tilt_hold_sum += tilt[hold_b].sum().item()
            yaw_hold_err_sum += yaw_err[hold_b].sum().item()
            grip_off_hold_sum += grip_off[hold_b].sum().item()
            force_samples.append(torch.maximum(left_force, right_force)[hold_b].detach().cpu())
            force_imbalance_samples.append((left_force - right_force).abs()[hold_b].detach().cpu())
            plate_height_diff_samples.append(plate_height_diff[hold_b].detach().cpu())
            mirror_error_hold_samples.append(mirror_error_post[hold_b].abs().detach().cpu())
            left_joint_hold_samples.append(q_left_post[hold_b].detach().cpu())
            right_joint_hold_samples.append(q_right_post[hold_b].detach().cpu())

        episode_step[active] += 1

        # Optional per-control-step trace. Terminal rows are excluded because Isaac Lab has already auto-reset
        # those envs when step() returns; every saved q/contact state therefore belongs to the labelled episode.
        if trace_env_ids:
            action_left = actions[:, left_action_ids]
            action_right = actions[:, right_action_ids]
            for env_id in trace_env_ids:
                if not active[env_id]:
                    continue
                row = {
                    "global_step": step,
                    "env_id": env_id,
                    "episode_index": int(episode_index[env_id].item()),
                    "episode_step": int(episode_step[env_id].item()),
                    "time_s": float(episode_step[env_id].item() * uenv.step_dt),
                    "left_contact": int((left_force[env_id] > CONTACT_THRESHOLD).item()),
                    "right_contact": int((right_force[env_id] > CONTACT_THRESHOLD).item()),
                    "one_sided_contact": int(
                        ((left_force[env_id] > CONTACT_THRESHOLD) ^ (right_force[env_id] > CONTACT_THRESHOLD)).item()
                    ),
                    "lifted": int(lifted[env_id].item()),
                    "hold": int(hold[env_id].item()),
                    "left_force_N": float(left_force[env_id].item()),
                    "right_force_N": float(right_force[env_id].item()),
                    "box_yaw_signed_deg": float(yaw_signed[env_id].item() * 57.2957795),
                }
                for joint_idx in range(7):
                    joint_num = joint_idx + 1
                    q_left_value = float(q_left_post[env_id, joint_idx].item())
                    q_right_value = float(q_right_post[env_id, joint_idx].item())
                    row.update(
                        {
                            f"left_j{joint_num}_q_rad": q_left_value,
                            f"right_j{joint_num}_q_rad": q_right_value,
                            f"left_j{joint_num}_q_deg": q_left_value * 57.2957795,
                            f"right_j{joint_num}_q_deg": q_right_value * 57.2957795,
                            f"right_j{joint_num}_mirror_expected_rad": float(
                                q_right_mirror_expected[env_id, joint_idx].item()
                            ),
                            f"mirror_error_j{joint_num}_rad": float(
                                mirror_error_post[env_id, joint_idx].item()
                            ),
                            f"mirror_error_j{joint_num}_deg": float(
                                mirror_error_post[env_id, joint_idx].item() * 57.2957795
                            ),
                            f"left_j{joint_num}_dq_rad_s": float(
                                robot.data.joint_vel[env_id, left_joint_ids[joint_idx]].item()
                            ),
                            f"right_j{joint_num}_dq_rad_s": float(
                                robot.data.joint_vel[env_id, right_joint_ids[joint_idx]].item()
                            ),
                            f"left_j{joint_num}_raw_action": float(action_left[env_id, joint_idx].item()),
                            f"right_j{joint_num}_raw_action": float(action_right[env_id, joint_idx].item()),
                        }
                    )
                trace_rows.append(row)

        near = err < SUCCESS_D
        upright = tilt < UPRIGHT_TILT
        base_ok = active & near & upright & grasped.bool()
        yaw10_ok = base_ok & (yaw_err < YAW_STRICT)
        yaw15_ok = base_ok & (yaw_err < YAW_ALIGN)
        base_streak = torch.where(base_ok, base_streak + 1, torch.zeros_like(base_streak))
        yaw10_streak = torch.where(yaw10_ok, yaw10_streak + 1, torch.zeros_like(yaw10_streak))
        yaw15_streak = torch.where(yaw15_ok, yaw15_streak + 1, torch.zeros_like(yaw15_streak))

        new_base_success = active & (base_streak >= SUCCESS_HOLD_STEPS) & ~ep_base_success
        new_yaw10_success = active & (yaw10_streak >= SUCCESS_HOLD_STEPS) & ~ep_yaw10_success
        new_yaw15_success = active & (yaw15_streak >= SUCCESS_HOLD_STEPS) & ~ep_yaw15_success
        if new_base_success.any():
            tts_base_samples.append((episode_step[new_base_success].float() * uenv.step_dt).detach().cpu())
        if new_yaw10_success.any():
            tts_yaw10_samples.append((episode_step[new_yaw10_success].float() * uenv.step_dt).detach().cpu())
        if new_yaw15_success.any():
            tts_yaw15_samples.append((episode_step[new_yaw15_success].float() * uenv.step_dt).detach().cpu())
        ep_base_success |= new_base_success
        ep_yaw10_success |= new_yaw10_success
        ep_yaw15_success |= new_yaw15_success
        ep_grasp |= active & grasped.bool()
        ep_lift |= active & lifted.bool()
        ep_near |= active & near

        sustained_base_sum += ((base_streak >= SUCCESS_HOLD_STEPS) & active).sum().item()
        sustained_yaw10_sum += ((yaw10_streak >= SUCCESS_HOLD_STEPS) & active).sum().item()
        sustained_yaw15_sum += ((yaw15_streak >= SUCCESS_HOLD_STEPS) & active).sum().item()

        # 기존 post-step drop 정의. Grasp-loss는 위 pre-step contact state machine에서 계측한다.
        drops += int((prev_hold & ~lifted.bool() & active).sum().item())
        prev_hold = torch.where(active, hold_b, torch.zeros_like(hold_b))

        if dones_b.any():
            completed_episodes += int(dones_b.sum().item())
            timeout_b = extras.get("time_outs", torch.zeros_like(dones_b)).bool()
            timeouts += int((dones_b & timeout_b).sum().item())
            out_of_bounds += int((dones_b & ~timeout_b).sum().item())
            episodes_grasp += int(ep_grasp[dones_b].sum().item())
            episodes_lift += int(ep_lift[dones_b].sum().item())
            episodes_near += int(ep_near[dones_b].sum().item())
            episodes_base_success += int(ep_base_success[dones_b].sum().item())
            episodes_yaw10_success += int(ep_yaw10_success[dones_b].sum().item())
            episodes_yaw15_success += int(ep_yaw15_success[dones_b].sum().item())
            one_sided_duration_samples.append((one_sided_steps[dones_b].float() * uenv.step_dt).detach().cpu())
            grasp_loss_count_samples.append(ep_grasp_loss_count[dones_b].float().detach().cpu())
            bilateral_joint_path_left_samples.append(bilateral_joint_path_left[dones_b].detach().cpu())
            bilateral_joint_path_right_samples.append(bilateral_joint_path_right[dones_b].detach().cpu())
            bilateral_episodes += int(ep_had_bilateral[dones_b].sum().item())
            episodes_with_grasp_loss += int((ep_grasp_loss_count[dones_b] > 0).sum().item())

            paired = dones_b & (first_left_contact >= 0) & (first_right_contact >= 0)
            if paired.any():
                signed_delta_steps = first_left_contact[paired] - first_right_contact[paired]
                delta_steps = signed_delta_steps.abs()
                contact_delta_samples.append((delta_steps.float() * uenv.step_dt).detach().cpu())
                contact_pairs += int(paired.sum().item())
                contact_within_1 += int((delta_steps <= 1).sum().item())
                contact_within_2 += int((delta_steps <= 2).sum().item())
                contact_within_5 += int((delta_steps <= 5).sum().item())
                left_first += int((signed_delta_steps < 0).sum().item())
                right_first += int((signed_delta_steps > 0).sum().item())
                tied_first += int((signed_delta_steps == 0).sum().item())

            left_only += int((dones_b & (first_left_contact >= 0) & (first_right_contact < 0)).sum().item())
            right_only += int((dones_b & (first_left_contact < 0) & (first_right_contact >= 0)).sum().item())
            no_contact += int((dones_b & (first_left_contact < 0) & (first_right_contact < 0)).sum().item())

            episode_step[dones_b] = 0
            contact_step[dones_b] = 0
            first_left_contact[dones_b] = -1
            first_right_contact[dones_b] = -1
            one_sided_steps[dones_b] = 0
            union_contact_steps[dones_b] = 0
            bilateral_contact_steps[dones_b] = 0
            ep_had_bilateral[dones_b] = False
            ep_grasp_loss_count[dones_b] = 0
            bilateral_joint_path_left[dones_b] = 0.0
            bilateral_joint_path_right[dones_b] = 0.0
            prev_sample_hold[dones_b] = False
            prev_joint_pos[dones_b] = robot.data.joint_pos[dones_b]
            base_streak[dones_b] = 0
            yaw10_streak[dones_b] = 0
            yaw15_streak[dones_b] = 0
            ep_grasp[dones_b] = False
            ep_lift[dones_b] = False
            ep_near[dones_b] = False
            ep_base_success[dones_b] = False
            ep_yaw10_success[dones_b] = False
            ep_yaw15_success[dones_b] = False
            episode_index[dones_b] += 1

        if step % steps_per_ep == 0:
            print(f"[EVAL] episode cycle {step // steps_per_ep + 1}/{args_cli.episodes}", flush=True)

    n = float(max(valid_post_samples, 1))

    def _cat(samples):
        return torch.cat(samples) if samples else torch.empty(0)

    def _quantile_text(samples, unit=""):
        values = _cat(samples)
        if values.numel() == 0:
            return "n/a"
        return (
            f"median {values.median().item():.4f}{unit}, "
            f"p95 {torch.quantile(values, 0.95).item():.4f}{unit}, max {values.max().item():.4f}{unit}"
        )

    def _episode_rate(count):
        return count / max(completed_episodes, 1)

    def _tts_text(samples):
        values = _cat(samples)
        if values.numel() == 0:
            return "n/a"
        return f"median {values.median().item():.2f}s, p95 {torch.quantile(values, 0.95).item():.2f}s"

    print("\n===== PLATE ROBUSTNESS EVAL =====", flush=True)
    print(f"task            : {args_cli.task}")
    print(f"checkpoint      : {args_cli.checkpoint}")
    print(f"envs x episodes : {args_cli.num_envs} x {args_cli.episodes} ({total_steps} steps)")
    print(f"seed / disturbance: {args_cli.seed} / {args_cli.disturbance_scale}")
    print(f"lifted fraction : {lifted_sum / n:.3f}")
    print(f"squeezed fraction: {grasp_sum / n:.3f}")
    print(f"hold (lift&squeeze): {hold_sum / n:.3f}")
    print(f"precise hold (<{SUCCESS_D * 100:.0f}cm): {success_sum / n:.3f}")
    print(f"upright hold (tilt<15deg): {upright_hold_sum / n:.3f}")
    print(f"yaw-aligned hold (<15deg): {yaw_hold_sum / n:.3f}")
    print(
        "sustained occupancy (base/yaw10/yaw15): "
        f"{sustained_base_sum / n:.3f} / {sustained_yaw10_sum / n:.3f} / {sustained_yaw15_sum / n:.3f}"
    )
    print(f"goal err while holding: {err_hold_sum / max(hold_sample_count, 1):.3f} m")
    print(f"box tilt while holding: {tilt_hold_sum / max(hold_sample_count, 1):.4f} (1-cos)")
    print(f"box yaw err while holding: {yaw_hold_err_sum / max(hold_sample_count, 1) * 57.29578:.1f} deg")
    print(
        "grip-axis offset while holding: "
        f"{grip_off_hold_sum / max(hold_sample_count, 1) * 57.29578:.1f} deg (mod pi)"
    )
    print(f"per-plate max force while holding: {_quantile_text(force_samples, 'N')}")
    print(f"left-right force imbalance: {_quantile_text(force_imbalance_samples, 'N')}")
    print(f"plate height difference: {_quantile_text(plate_height_diff_samples, 'm')}")
    print(f"completed episodes: {completed_episodes} (timeouts {timeouts}, OOB {out_of_bounds})")
    print(
        "episode stage rates grasp/lift/near: "
        f"{_episode_rate(episodes_grasp):.3f} / {_episode_rate(episodes_lift):.3f} / "
        f"{_episode_rate(episodes_near):.3f}"
    )
    print(
        "episode sustained SR base/yaw10/yaw15: "
        f"{_episode_rate(episodes_base_success):.3f} / {_episode_rate(episodes_yaw10_success):.3f} / "
        f"{_episode_rate(episodes_yaw15_success):.3f}"
    )
    print(
        "TTS base/yaw10/yaw15: "
        f"{_tts_text(tts_base_samples)} / {_tts_text(tts_yaw10_samples)} / {_tts_text(tts_yaw15_samples)}"
    )
    print(f"first-contact |dt|: {_quantile_text(contact_delta_samples, 's')} ({contact_pairs} paired episodes)")
    print(
        "first-contact synchrony <=20/40/100ms: "
        f"{contact_within_1 / max(contact_pairs, 1):.3f} / "
        f"{contact_within_2 / max(contact_pairs, 1):.3f} / "
        f"{contact_within_5 / max(contact_pairs, 1):.3f}"
    )
    xor_duration = _cat(one_sided_duration_samples)
    if xor_duration.numel() > 0:
        print(
            "one-sided contact duration/episode: "
            f"mean {xor_duration.mean().item():.4f}s "
            f"({xor_duration.mean().item() / uenv.step_dt:.2f} steps), "
            f"median {xor_duration.median().item():.4f}s, "
            f"p95 {torch.quantile(xor_duration, 0.95).item():.4f}s, "
            f"max {xor_duration.max().item():.4f}s"
        )
    else:
        print("one-sided contact duration/episode: n/a")
    print(
        "one-sided contact total: "
        f"{xor_total_steps * uenv.step_dt:.3f} env-s "
        f"(sampled env-time fraction "
        f"{xor_total_steps / max(args_cli.num_envs * total_steps, 1):.4f})"
    )
    print(f"paired-contact episode coverage: {contact_pairs / max(completed_episodes, 1):.3f}")
    print(
        "first-contact side L/R/tie: "
        f"{left_first}/{right_first}/{tied_first} "
        f"({left_first / max(contact_pairs, 1):.3f}/"
        f"{right_first / max(contact_pairs, 1):.3f}/"
        f"{tied_first / max(contact_pairs, 1):.3f})"
    )
    print(f"unpaired contact episodes L-only/R-only/none: {left_only}/{right_only}/{no_contact}")
    print(
        "actual bilateral coverage episode/contact-time: "
        f"{bilateral_episodes / max(completed_episodes, 1):.3f} / "
        f"{bilateral_total_steps / max(union_total_steps, 1):.3f}"
    )
    print(
        "grasp-loss episode coverage/count: "
        f"{episodes_with_grasp_loss / max(completed_episodes, 1):.3f} / "
        f"{_quantile_text(grasp_loss_count_samples)}"
    )
    path_left = _cat(bilateral_joint_path_left_samples)
    path_right = _cat(bilateral_joint_path_right_samples)
    if path_left.numel() > 0:
        total_left = path_left.sum().item()
        total_right = path_right.sum().item()
        print(
            "post-bilateral joint path L/R/ratio: "
            f"{total_left / max(bilateral_episodes, 1):.3f} / "
            f"{total_right / max(bilateral_episodes, 1):.3f} rad/bilateral-episode / "
            f"{total_right / max(total_left, 1e-12):.3f}"
        )
        per_joint_path_left = path_left.sum(dim=0) / max(bilateral_episodes, 1)
        per_joint_path_right = path_right.sum(dim=0) / max(bilateral_episodes, 1)
        print("post-bilateral per-joint path (rad/bilateral-episode):")
        print("  joint       left       right      R/L")
        for joint_idx in range(7):
            left_value = per_joint_path_left[joint_idx].item()
            right_value = per_joint_path_right[joint_idx].item()
            print(
                f"  j{joint_idx + 1:<2d}     {left_value:8.4f}   {right_value:8.4f}   "
                f"{right_value / max(left_value, 1e-12):7.3f}"
            )
    else:
        print("post-bilateral joint path L/R/ratio: n/a")

    mirror_hold = _cat(mirror_error_hold_samples)
    left_hold_q = _cat(left_joint_hold_samples)
    right_hold_q = _cat(right_joint_hold_samples)
    if mirror_hold.numel() > 0:
        mirror_median = mirror_hold.median(dim=0).values * 57.2957795
        mirror_p95 = torch.quantile(mirror_hold, 0.95, dim=0) * 57.2957795
        mirror_max = mirror_hold.max(dim=0).values * 57.2957795
        left_q_median = left_hold_q.median(dim=0).values * 57.2957795
        right_q_median = right_hold_q.median(dim=0).values * 57.2957795
        print("hold-pose mirror error |qR - sign*qL| wrapped (deg):")
        print("  joint   sign   qL_med   qR_med   err_med   err_p95   err_max")
        for joint_idx in range(7):
            print(
                f"  j{joint_idx + 1:<2d}     {int(mirror_signs[joint_idx].item()):+d}   "
                f"{left_q_median[joint_idx].item():7.2f}  {right_q_median[joint_idx].item():7.2f}  "
                f"{mirror_median[joint_idx].item():8.2f}  {mirror_p95[joint_idx].item():8.2f}  "
                f"{mirror_max[joint_idx].item():8.2f}"
            )
    else:
        print("hold-pose per-joint mirror error: n/a")
    assert left_first + right_first + tied_first == contact_pairs
    assert contact_pairs + left_only + right_only + no_contact == completed_episodes
    assert bilateral_episodes <= contact_pairs
    assert union_total_steps == xor_total_steps + bilateral_total_steps
    print(f"drop / grasp-loss events: {drops} / {grasp_loss_events}", flush=True)

    if args_cli.joint_trace_csv:
        trace_path = Path(args_cli.joint_trace_csv).resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        if trace_rows:
            with trace_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(trace_rows[0].keys()))
                writer.writeheader()
                writer.writerows(trace_rows)
            print(
                f"joint trace CSV: {trace_path} ({len(trace_rows)} rows, envs={trace_env_ids})",
                flush=True,
            )
        else:
            print(f"joint trace CSV: no active rows collected; file not written ({trace_path})", flush=True)


if __name__ == "__main__":
    main()
    # [좀비 방지] Isaac Sim 종료(simulation_app.close())가 자주 리턴을 안 해 프로세스가 결과 출력 뒤에도
    # CPU 코어를 잡은 채 안 죽음(19h+ 102% 관찰). 결과는 위에서 flush=True로 이미 기록됨 -> close를 건너뛰고
    # 즉시 강제 종료해 좀비 차단. os._exit는 finalizer/atexit를 건너뛰지만 여기선 출력만 있으면 되므로 안전.
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
