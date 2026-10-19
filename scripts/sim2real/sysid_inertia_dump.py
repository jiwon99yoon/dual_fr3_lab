#!/usr/bin/env python
"""dual_fr3 reach articulation 의 관절 inertia/armature/drive 파라미터 덤프.

SysID 에서 Isaac 과도응답이 real/MuJoCo 대비 과도하게 sluggish 한 원인
(= 유효 관절 관성 과대)을 확인한다. home(q0) 자세에서:
  - joint armature (로터 관성)
  - joint friction / stiffness / damping (override 0 확인용)
  - generalized mass matrix 대각 = 관절별 유효 관성 M_jj
  - link mass
를 출력한다. real franka_description 값과 대조하면 용의자 확정.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from dual_fr3_lab.robots.dual_fr3_plate_ee import DUAL_FR3_PLATE_EE_CFG

DUAL_ARM = [f"{s}_fr3_joint{i}" for s in ("left", "right") for i in range(1, 8)]
HOME = np.array([0.0, 0.004, -0.523, -0.780, 0.0, -2.362, 0.006,
                 1.572, 0.799, 0.525, -0.782, 0.0, -2.363, 0.004])


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.001, device=args_cli.device))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    cfg = DUAL_FR3_PLATE_EE_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(cfg)
    sim.reset()

    names = robot.joint_names
    idx = {n: i for i, n in enumerate(names)}
    sim_ids = torch.tensor([idx[n] for n in DUAL_ARM], device=sim.device, dtype=torch.long)

    # home 자세로 세팅 후 한 스텝
    q = robot.data.default_joint_pos.clone()
    q[0, sim_ids] = torch.tensor(HOME, device=sim.device, dtype=q.dtype)
    robot.write_joint_state_to_sim(q, torch.zeros_like(robot.data.default_joint_vel))
    robot.reset()
    robot.write_data_to_sim()
    sim.step()
    robot.update(0.001)

    view = robot.root_physx_view

    def try_get(fn_name):
        try:
            v = getattr(view, fn_name)()
            return v.detach().cpu().numpy()[0] if hasattr(v, "detach") else np.asarray(v)[0]
        except Exception as e:
            return f"(n/a: {type(e).__name__})"

    arm = try_get("get_dof_armatures")
    fric = try_get("get_dof_friction_coefficients")
    stiff = try_get("get_dof_stiffnesses")
    damp = try_get("get_dof_dampings")
    maxf = try_get("get_dof_max_forces")

    # ── pose-무관 링크 질량/관성 (franka_description 대조용) ──
    masses = try_get("get_masses")           # (num_links,)
    inertias = try_get("get_inertias")       # (num_links, 9) diag 포함
    body_names = robot.body_names
    print("\n" + "=" * 78)
    print("링크 질량 (pose 무관) — franka_description FR3 링크와 대조")
    print("=" * 78)
    if isinstance(masses, np.ndarray):
        for bi, bn in enumerate(body_names):
            m = masses[bi]
            diag = ""
            if isinstance(inertias, np.ndarray):
                I = inertias[bi]
                diag = f"  Ixx,Iyy,Izz={I[0]:.4f},{I[4]:.4f},{I[8]:.4f}" if I.size == 9 else ""
            print(f"  [{bi:2d}] {bn:28s} m={m:8.4f} kg{diag}")
        print(f"  --- 총 질량 = {masses.sum():.3f} kg (FR3 단일팔 ~18kg, 양팔+plate 참고) ---")
    else:
        print(f"  get_masses n/a: {masses}")

    # generalized mass matrix (관절-공간 관성) 대각 = 유효 관성
    try:
        M = view.get_generalized_mass_matrices().detach().cpu().numpy()[0]
        Mdiag = np.diag(M)
    except Exception as e:
        Mdiag = None
        print(f"[mass matrix] n/a: {e}")

    print("\n" + "=" * 78)
    print(f"{'joint':22s} {'armature':>9s} {'friction':>9s} {'stiff':>7s} {'damp':>6s} {'M_jj':>9s}")
    print("=" * 78)
    def col(a, i):
        return a[i] if isinstance(a, np.ndarray) else None
    for n in DUAL_ARM:
        i = idx[n]
        a = col(arm, i); f = col(fric, i); s = col(stiff, i); d = col(damp, i)
        m = Mdiag[i] if Mdiag is not None else None
        def fmt(x, w, p=4):
            return f"{x:>{w}.{p}f}" if isinstance(x, (int, float, np.floating)) else f"{str(x):>{w}}"
        print(f"{n:22s} {fmt(a,9)} {fmt(f,9)} {fmt(s,7,1)} {fmt(d,6,1)} {fmt(m,9)}")

    print("\n[참고] real FR3 rotor inertia(반사관성)는 관절당 대략 0.0x~0.6 kg·m² 수준.")
    print("       M_jj(유효 관성)이 수 kg·m² 이상이면 과도응답이 sluggish 해진다.")


if __name__ == "__main__":
    main()
    simulation_app.close()
