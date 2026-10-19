#!/usr/bin/env python
"""SysID Isaac recorder — fr3_sysid_probe_node(q0-reference 모드)를 IsaacSim에서 재현.

fr3_husky_etri/fr3_sysid_probe_node.py 의 **q0-reference 모드**와 동일한 제어
법칙·프로파일·CSV 스키마로 IsaacSim(dual_fr3_lab) rollout을 기록한다. 결과 CSV는
real/ · mujoco/ 로그와 그대로 overlay 가능하며 IsaacSim↔MuJoCo(sim2sim),
IsaacSim↔real(sim2real) gap 분석에 쓴다.

q0-reference 제어 법칙 (policy_control_action_server.cpp, isaac_relative_control):
    reference = home + excitation      # 모든 관절 home 고정, 활성 관절만 excitation
    offset    = reference - q_meas     # probe 가 보내는 값
    q_target  = clamp(q_meas + offset) = clamp(reference)   # = home + excitation
    effort    = Kp*(q_target - q_meas) - Kd*qd
    effort    = clamp(effort, ±effort_limit)
    applied   = prev + clamp(effort - prev, ±torque_rate*dt)   # 1000 Nm/s
→ 비활성 관절은 home 에 stiff 하게 고정(floppy 아님), 활성 관절만 프로파일 추종.

ramp 프로파일(return_to_start) phase:
    q0_prehold[0,t_start] → ramp_up[,+ramp] → hold[,+hold] → ramp_back[,+ramp] → q0_stabilize

중력: DUAL_FR3_PLATE_EE_CFG 는 disable_gravity=True (실물 내장 중력보상 모델링).
active_joint 는 probe 와 동일한 블록 인덱스(DUAL_ARM_JOINT_NAMES); sim articulation
(interleave)과는 이름으로 매핑. 정책·.pt·joint-order 무관한 open-loop plant 비교다.

실행:
  ~/anaconda3/envs/env_isaaclab/bin/python scripts/sim2real/sysid_isaac_recorder.py \
      --headless --sweep --profile ramp --amplitude 0.25 --amplitude_wrist 0.25 \
      --ref_dir ~/Downloads/sysid_logs_260911_1422/sysid_logs/real \
      --out_dir ~/Downloads/sysid_logs_260911_1422/sysid_logs/isaac
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="IsaacSim SysID q0-reference recorder.")
parser.add_argument("--profile", choices=("step", "ramp", "pulse"), default="ramp")
parser.add_argument("--amplitude", type=float, default=0.25, help="주 관절 excitation 진폭(rad)")
parser.add_argument("--amplitude_wrist", type=float, default=0.25, help="손목(j5-7) 진폭(rad)")
parser.add_argument("--t_start", type=float, default=1.0, help="excitation 전 prehold(s)")
parser.add_argument("--ramp_time", type=float, default=1.0)
parser.add_argument("--pulse_width", type=float, default=0.5)
parser.add_argument("--hold_time", type=float, default=2.0)
parser.add_argument("--stabilize_tail", type=float, default=1.0, help="복귀 후 안정화 기록(s)")
parser.add_argument("--sweep", action="store_true", help="14관절 전체 순차")
parser.add_argument("--target_joint", type=int, default=0, help="단일 모드 블록 인덱스[0-13]")
parser.add_argument("--ref_dir", type=str, default="",
                    help="관절별 home(q0)를 읽어올 real/mujoco 폴더(매칭 profile/amp 최신 파일)")
parser.add_argument("--joint_friction", type=float, default=-1.0,
                    help=">=0 이면 전 관절 static(coulomb) friction τ_dry 를 이 값으로 덮어씀(기본 -1=원본 유지).")
parser.add_argument("--armature", type=float, default=-1.0,
                    help=">=0 이면 전 관절 armature A(kg·m², [M+A]q̈)를 이 값으로 덮어씀(기본 -1=원본 유지).")
parser.add_argument("--joint_damping", type=float, default=-1.0,
                    help=">=0 이면 전 관절 viscous friction B(Nms, Bq̇)를 이 값으로 덮어씀. "
                         "Isaac Sim 5.0+ 필요(기본 -1=원본 유지).")
parser.add_argument("--solver_pos_iter", type=int, default=-1,
                    help=">=0 이면 articulation solver_position_iteration_count 덮어씀(기본 8).")
parser.add_argument("--solver_vel_iter", type=int, default=-1,
                    help=">=0 이면 articulation solver_velocity_iteration_count 덮어씀(기본 0).")
parser.add_argument("--actuator", choices=("implicit_manual", "explicit_pd"),
                    default="implicit_manual",
                    help="implicit_manual: ImplicitActuator(gain0)+수동 effort 주입(PhysX drive 경로). "
                         "explicit_pd: IdealPDActuator(Kp80/Kd4)가 파이썬에서 PD→외력으로 적용(implicit drive 미경유).")
parser.add_argument("--out_dir", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac 런타임 ────────────────────────────────────────────────────────────
import csv
import time
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from dual_fr3_lab.robots.dual_fr3 import (
    FR3_ARM_KD_NM_S_PER_RAD,
    FR3_ARM_KP_NM_PER_RAD,
    FR3_FOREARM_EFFORT_LIMIT_NM,
    FR3_SHOULDER_EFFORT_LIMIT_NM,
    FR3_TORQUE_RATE_LIMIT_NM_S,
)
from dual_fr3_lab.robots.dual_fr3_plate_ee import DUAL_FR3_PLATE_EE_CFG

DUAL_ARM_JOINT_NAMES = [
    f"{side}_fr3_joint{i}" for side in ("left", "right") for i in range(1, 8)
]

# policy_control_action_server.hpp 상수 (within-arm 7개)
KP = FR3_ARM_KP_NM_PER_RAD
KD = FR3_ARM_KD_NM_S_PER_RAD
EFFORT_LIMITS = np.array(
    [
        FR3_SHOULDER_EFFORT_LIMIT_NM,
        FR3_SHOULDER_EFFORT_LIMIT_NM,
        FR3_SHOULDER_EFFORT_LIMIT_NM,
        FR3_SHOULDER_EFFORT_LIMIT_NM,
        FR3_FOREARM_EFFORT_LIMIT_NM,
        FR3_FOREARM_EFFORT_LIMIT_NM,
        FR3_FOREARM_EFFORT_LIMIT_NM,
    ]
)
TORQUE_RATE = FR3_TORQUE_RATE_LIMIT_NM_S
LIMIT_MARGIN = 0.02
LOWER = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
UPPER = np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])
DT = 0.001


def _amplitude_for(block_index: int) -> float:
    return args_cli.amplitude_wrist if (block_index % 7) >= 4 else args_cli.amplitude


def _excitation(elapsed_s: float, block_index: int):
    """probe._offset_vector(return_to_start) 와 동일한 활성 관절 excitation + phase."""
    a = _amplitude_for(block_index)
    t = elapsed_s - args_cli.t_start
    if t < 0.0:
        return 0.0, "q0_prehold"
    if args_cli.profile == "step":
        return (a, "step_up") if t <= args_cli.hold_time else (0.0, "q0_stabilize")
    if args_cli.profile == "ramp":
        rt, ht = args_cli.ramp_time, args_cli.hold_time
        if t <= rt:
            return a * (t / rt), "ramp_up"
        if t <= rt + ht:
            return a, "hold"
        if t <= 2.0 * rt + ht:
            return a * (1.0 - (t - rt - ht) / rt), "ramp_back"
        return 0.0, "q0_stabilize"
    # pulse
    return (a, "pulse_up") if t <= args_cli.pulse_width else (0.0, "q0_stabilize")


def _total_time() -> float:
    ts = args_cli.t_start
    if args_cli.profile == "step":
        return ts + args_cli.hold_time + args_cli.stabilize_tail
    if args_cli.profile == "ramp":
        return ts + 2.0 * args_cli.ramp_time + args_cli.hold_time + args_cli.stabilize_tail
    return ts + args_cli.pulse_width + args_cli.stabilize_tail


def _load_home(joint_name: str, default_block: np.ndarray) -> np.ndarray:
    """ref_dir 에서 매칭 profile/amp 최신 파일의 home(q0, 14D 블록)을 읽는다."""
    if not args_cli.ref_dir:
        return default_block
    ref = Path(args_cli.ref_dir).expanduser()
    amp = _amplitude_for(DUAL_ARM_JOINT_NAMES.index(joint_name))
    pats = [f"{joint_name}_{args_cli.profile}_amp{amp:.3f}_*.csv",
            f"{joint_name}_*.csv"]
    for pat in pats:
        matches = sorted(ref.glob(pat), key=lambda p: p.stat().st_mtime)
        if matches:
            with open(matches[-1], newline="") as fh:
                row = next(csv.DictReader(fh))
            key = "q0_" if f"q0_{DUAL_ARM_JOINT_NAMES[0]}" in row else "q_"
            print(f"  [home] {joint_name} ← {matches[-1].name} ({key}*)")
            return np.array([float(row[f"{key}{n}"]) for n in DUAL_ARM_JOINT_NAMES])
    print(f"  [home] {joint_name}: ref 없음 → articulation default")
    return default_block


def main():
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=DT, device=args_cli.device))
    sim.set_camera_view([2.0, 2.0, 2.0], [0.0, 0.0, 0.5])
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func(
        "/World/light", sim_utils.DomeLightCfg(intensity=2000.0))

    robot_cfg = DUAL_FR3_PLATE_EE_CFG.replace(prim_path="/World/Robot")
    fric_set = args_cli.joint_friction >= 0.0
    explicit = args_cli.actuator == "explicit_pd"
    if explicit:
        # implicit drive 미경유: IdealPDActuator가 tau=Kp*(qd_pos-q)+Kd*(qd_vel-qd)를
        # 파이썬에서 계산해 외력으로 적용. 게인/effort limit은 real 컨트롤러(80/4, 87/12)와 동일.
        from isaaclab.actuators import IdealPDActuatorCfg
        new_acts = {}
        for name, act in robot_cfg.actuators.items():
            new_acts[name] = IdealPDActuatorCfg(
                joint_names_expr=act.joint_names_expr,
                effort_limit=act.effort_limit_sim,
                stiffness=KP,
                damping=KD,
                armature=act.armature,
                friction=act.friction,
                dynamic_friction=act.dynamic_friction,
                viscous_friction=act.viscous_friction,
            )
        robot_cfg.actuators = new_acts
    else:
        for act in robot_cfg.actuators.values():   # PD 끄고 토크 직접 주입
            act.stiffness = 0.0
            act.damping = 0.0
    print(f"[actuator] mode={args_cli.actuator}")
    ap = robot_cfg.spawn.articulation_props
    if args_cli.solver_pos_iter >= 0:
        ap.solver_position_iteration_count = args_cli.solver_pos_iter
    if args_cli.solver_vel_iter >= 0:
        ap.solver_velocity_iteration_count = args_cli.solver_vel_iter
    print(f"[solver] pos_iter={ap.solver_position_iteration_count} "
          f"vel_iter={ap.solver_velocity_iteration_count}")
    robot = Articulation(robot_cfg)
    sim.reset()

    # 이미지 dynamics [M+A]q̈ + Cq̇ + Bq̇ + τ_dry = τ_app 의 A·B·τ_dry 를 실제 관절 물성으로 주입.
    # 셋 다 implicit drive(stiffness=0/damping=0)와 무관한 PhysX plant 물성이라 manual-effort 경로에서도 적용됨.
    fric_props = robot.root_physx_view.get_dof_friction_properties().detach().cpu().numpy()
    if fric_set:  # τ_dry (static/coulomb friction)
        robot.write_joint_friction_coefficient_to_sim(args_cli.joint_friction)
    if args_cli.joint_damping >= 0.0:  # B (viscous friction) — Isaac Sim 5.0+
        robot.write_joint_viscous_friction_coefficient_to_sim(args_cli.joint_damping)
    if args_cli.armature >= 0.0:  # A (armature, joint-space inertia에 가산)
        robot.write_joint_armature_to_sim(args_cli.armature)
    if fric_set or args_cli.joint_damping >= 0.0 or args_cli.armature >= 0.0:
        fp = robot.root_physx_view.get_dof_friction_properties().detach().cpu().numpy()
        arm = robot.root_physx_view.get_dof_armatures().detach().cpu().numpy()
        print(f"[dyn] τ_dry(static)={args_cli.joint_friction} B(viscous)={args_cli.joint_damping} "
              f"A(armature)={args_cli.armature}")
        print(f"      readback j1  static={fp[0,0,0]:.4f} viscous={fp[0,0,2]:.4f} armature={arm[0,0]:.4f}")

    name_to_sim = {n: i for i, n in enumerate(robot.joint_names)}
    sim_ids = torch.tensor([name_to_sim[n] for n in DUAL_ARM_JOINT_NAMES],
                           device=sim.device, dtype=torch.long)
    default_block = robot.data.default_joint_pos[0, sim_ids].detach().cpu().numpy()

    limit_idx = np.array([j % 7 for j in range(14)])
    eff_lim = EFFORT_LIMITS[limit_idx]
    lower = LOWER[limit_idx] + LIMIT_MARGIN
    upper = UPPER[limit_idx] - LIMIT_MARGIN

    out_dir = (Path(args_cli.out_dir).expanduser() if args_cli.out_dir
               else Path(__file__).resolve().parents[2] / "logs" / "sysid" / "isaac")
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep = list(range(14)) if args_cli.sweep else [args_cli.target_joint]
    n_steps = int(round(_total_time() / DT))

    header = (["t_sec", "phase", "active_joint",
               "q_ref_active_rad", "q_ref_delta_from_q0_rad",
               "cmd_offset_rad", "tau_cmd_recon_nm"]
              + [f"q_{n}" for n in DUAL_ARM_JOINT_NAMES]
              + [f"qd_{n}" for n in DUAL_ARM_JOINT_NAMES]
              + [f"tau_meas_{n}" for n in DUAL_ARM_JOINT_NAMES]
              + [f"q0_{n}" for n in DUAL_ARM_JOINT_NAMES])

    for active in sweep:
        jname = DUAL_ARM_JOINT_NAMES[active]
        amp = _amplitude_for(active)
        home = _load_home(jname, default_block)
        home_clip = np.clip(home, lower, upper)

        full_q = robot.data.default_joint_pos.clone()
        full_q[0, sim_ids] = torch.tensor(home_clip, device=sim.device, dtype=full_q.dtype)
        robot.write_joint_state_to_sim(full_q, torch.zeros_like(robot.data.default_joint_vel))
        robot.reset()
        prev_applied = np.zeros(14)

        stamp = time.strftime("%y%m%d_%H%M%S")
        dyn_tag = ""
        if args_cli.armature >= 0.0:
            dyn_tag += f"_A{args_cli.armature:.3f}"
        if args_cli.joint_damping >= 0.0:
            dyn_tag += f"_B{args_cli.joint_damping:.3f}"
        if args_cli.joint_friction >= 0.0:
            dyn_tag += f"_fric{args_cli.joint_friction:.2f}"
        path = out_dir / f"{jname}_{args_cli.profile}_amp{amp:.3f}{dyn_tag}_{stamp}.csv"
        print(f"[{active:2d}] {jname} amp={amp:.3f} → {path.name}")

        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for step in range(n_steps):
                t = step * DT
                exc, phase = _excitation(t, active)
                excitation = np.zeros(14)
                excitation[active] = exc

                q = robot.data.joint_pos[0, sim_ids].detach().cpu().numpy()
                qd = robot.data.joint_vel[0, sim_ids].detach().cpu().numpy()

                q_target = np.clip(home_clip + excitation, lower, upper)
                cmd_offset = q_target - q                      # probe 가 보내는 offset

                if explicit:
                    # IdealPDActuator가 PD→외력을 처리. 우리는 목표만 준다.
                    qpos_t = robot.data.default_joint_pos.clone()
                    qpos_t[0, sim_ids] = torch.tensor(q_target, device=sim.device, dtype=qpos_t.dtype)
                    qvel_t = torch.zeros_like(robot.data.default_joint_vel)
                    robot.set_joint_position_target(qpos_t)
                    robot.set_joint_velocity_target(qvel_t)
                else:
                    # implicit drive에 수동 계산 effort 주입
                    effort = KP * (q_target - q) - KD * qd
                    effort = np.clip(effort, -eff_lim, eff_lim)
                    applied = prev_applied + np.clip(
                        effort - prev_applied, -TORQUE_RATE * DT, TORQUE_RATE * DT)
                    prev_applied = applied
                    tau_t = torch.zeros((1, robot.num_joints), device=sim.device)
                    tau_t[0, sim_ids] = torch.tensor(applied, device=sim.device, dtype=tau_t.dtype)
                    robot.set_joint_effort_target(tau_t)
                robot.write_data_to_sim()
                sim.step()
                robot.update(DT)

                # tau_meas = PhysX 가 실제로 관절에 가한 토크(real /joint_states effort 대응)
                tau_real = robot.data.applied_torque[0, sim_ids].detach().cpu().numpy()
                tau_recon = KP * cmd_offset[active] - KD * qd[active]
                writer.writerow(
                    [f"{t:.6f}", phase, active,
                     f"{q_target[active]:.6f}", f"{excitation[active]:.6f}",
                     f"{cmd_offset[active]:.6f}", f"{tau_recon:.6f}"]
                    + [f"{v:.6f}" for v in q]
                    + [f"{v:.6f}" for v in qd]
                    + [f"{v:.6f}" for v in tau_real]
                    + [f"{v:.6f}" for v in home_clip])

    print(f"\n완료. CSV → {out_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
