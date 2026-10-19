#!/usr/bin/env python
"""상수 토크로 관절 유효 관성 I_eff = tau / q̈ 직접 측정.

진짜 실험 home 자세에서 active joint 만 free(상수 토크)로 두고 나머지는 home 에
PD 고정한 뒤, 초기 각가속도로 유효 관성을 잰다. ramp lag(≈0.08)가 관성 때문인지
(I_eff 가 spec 대비 큰지) 최종 판정용.

실행:
  python scripts/sim2real/sysid_inertia_probe.py --headless --target_joint 0 --torque 4.0 \
      --ref_dir ~/Downloads/sysid_logs_260911_1422/sysid_logs/real
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--target_joint", type=int, default=0)
parser.add_argument("--torque", type=float, default=4.0, help="active joint 상수 토크(Nm)")
parser.add_argument("--settle", type=float, default=0.3, help="토크 전 정착(s)")
parser.add_argument("--apply", type=float, default=0.3, help="토크 인가 구간(s)")
parser.add_argument("--fit_window", type=float, default=0.05, help="q̈ 추정 fit 구간(s, 토크 인가 직후)")
parser.add_argument("--ref_dir", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv, glob, os
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from dual_fr3_lab.robots.dual_fr3_plate_ee import DUAL_FR3_PLATE_EE_CFG

DUAL_ARM = [f"{s}_fr3_joint{i}" for s in ("left", "right") for i in range(1, 8)]
KP, KD, DT = 80.0, 4.0, 0.001


def load_home(active_name):
    if not args_cli.ref_dir:
        return None
    m = sorted(glob.glob(os.path.join(os.path.expanduser(args_cli.ref_dir),
               f"{active_name}_ramp_amp0.250_*.csv")), key=os.path.getmtime)
    if not m:
        return None
    row = next(csv.DictReader(open(m[-1])))
    key = "q0_" if f"q0_{DUAL_ARM[0]}" in row else "q_"
    return np.array([float(row[f"{key}{n}"]) for n in DUAL_ARM])


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args_cli.device))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    cfg = DUAL_FR3_PLATE_EE_CFG.replace(prim_path="/World/Robot")
    for act in cfg.actuators.values():
        act.stiffness = 0.0
        act.damping = 0.0
    robot = Articulation(cfg)
    sim.reset()

    idx = {n: i for i, n in enumerate(robot.joint_names)}
    sim_ids = torch.tensor([idx[n] for n in DUAL_ARM], device=sim.device, dtype=torch.long)

    a = args_cli.target_joint
    home = load_home(DUAL_ARM[a])
    if home is None:
        home = robot.data.default_joint_pos[0, sim_ids].detach().cpu().numpy()
    full_q = robot.data.default_joint_pos.clone()
    full_q[0, sim_ids] = torch.tensor(home, device=sim.device, dtype=full_q.dtype)
    robot.write_joint_state_to_sim(full_q, torch.zeros_like(robot.data.default_joint_vel))
    robot.reset()

    base_idx = robot.body_names.index("base")
    base_pos0 = robot.data.body_pos_w[0, base_idx].detach().cpu().numpy().copy()

    n_settle = int(args_cli.settle / DT)
    n_apply = int(args_cli.apply / DT)
    ts, qs, qds = [], [], []
    base_lin_max = 0.0
    base_ang_max = 0.0
    base_drift_max = 0.0
    for step in range(n_settle + n_apply):
        q = robot.data.joint_pos[0, sim_ids].detach().cpu().numpy()
        qd = robot.data.joint_vel[0, sim_ids].detach().cpu().numpy()
        effort = KP * (home - q) - KD * qd     # 전 관절 home 고정
        if step >= n_settle:
            effort[a] = args_cli.torque         # active 만 상수 토크(free)
        tau_t = torch.zeros((1, robot.num_joints), device=sim.device)
        tau_t[0, sim_ids] = torch.tensor(effort, device=sim.device, dtype=tau_t.dtype)
        robot.set_joint_effort_target(tau_t)
        robot.write_data_to_sim()
        sim.step()
        robot.update(DT)
        if step >= n_settle:
            t = (step - n_settle) * DT
            ts.append(t); qs.append(robot.data.joint_pos[0, sim_ids[a]].item())
            qds.append(robot.data.joint_vel[0, sim_ids[a]].item())
            bl = float(np.linalg.norm(robot.data.body_lin_vel_w[0, base_idx].detach().cpu().numpy()))
            ba = float(np.linalg.norm(robot.data.body_ang_vel_w[0, base_idx].detach().cpu().numpy()))
            bd = float(np.linalg.norm(robot.data.body_pos_w[0, base_idx].detach().cpu().numpy() - base_pos0))
            base_lin_max = max(base_lin_max, bl)
            base_ang_max = max(base_ang_max, ba)
            base_drift_max = max(base_drift_max, bd)

    ts = np.array(ts); qs = np.array(qs); qds = np.array(qds)
    # q̈: (1) 초기 q(t)=q0+0.5*acc*t² 2차 fit, (2) qd 기울기 — 둘 다
    w = ts <= args_cli.fit_window
    acc_pos = 2.0 * np.polyfit(ts[w], qs[w], 2)[0]
    acc_vel = np.polyfit(ts[w], qds[w], 1)[0]
    jn = DUAL_ARM[a]
    print("\n" + "=" * 60)
    print(f"관절 {a} ({jn}) — 상수토크 {args_cli.torque} Nm")
    print(f"  fit window = {args_cli.fit_window*1000:.0f} ms, 샘플 {w.sum()}개")
    print(f"  q̈ (pos-fit)  = {acc_pos:8.3f} rad/s²  → I_eff = {args_cli.torque/acc_pos:.4f} kg·m²")
    print(f"  q̈ (vel-slope)= {acc_vel:8.3f} rad/s²  → I_eff = {args_cli.torque/acc_vel:.4f} kg·m²")
    print(f"  최종 qd={qds[-1]:.3f} rad/s, 변위={qs[-1]-qs[0]:+.4f} rad")
    print("  [참고] lag 0.08 설명하려면 I_eff~2.5, spec대로면 ~0.4")
    print("-" * 60)
    print(f"  base 링크 움직임 (고정이면 ~0 이어야 함):")
    print(f"    max |lin_vel|={base_lin_max:.5f} m/s, max |ang_vel|={base_ang_max:.5f} rad/s, "
          f"max drift={base_drift_max:.6f} m")
    print(f"    → base 움직이면 root-fix 실패(내 recorder 아티팩트)")
    print("=" * 60)


if __name__ == "__main__":
    main()
    simulation_app.close()
