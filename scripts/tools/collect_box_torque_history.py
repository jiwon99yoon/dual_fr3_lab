"""Collect sim box-lift rollouts (policy checkpoint) with per-step joint torque / plate force histories and per-episode
labels (box mass, box friction, plate friction) for an offline "can mass and friction be estimated from torque
history?" study (step 2 of the torque-adaptation idea, 11-13).

Per round: labels are re-randomized for every env (mass abs range, friction ranges), then E episodes are rolled out with
the deterministic policy. Saved arrays (npz):
  X      (N, T, D)  per step: q(14) dq(14) tau_applied(14) f_plate_L f_plate_R (contact normal magnitude, N) box_dz
                    box_vz action(14)
  lifted (N, T)     box z above its episode start by > LIFT_M
  y      (N, 4)     mass [kg], mu_box, mu_plate_L, mu_plate_R
  done_ok(N,)       episode ran full length (no early termination)

    OMNI_KIT_ACCEPT_EULA=YES python scripts/tools/collect_box_torque_history.py --headless --num_envs 512 --rounds 3 --episodes 1 \
        --checkpoint logs/rsl_rl/.../model_5499.pt --out <npz>
"""
import argparse
import importlib.metadata as metadata

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-BoxDrop-FlatSqueeze-LowMuSoftLift-GoalXY-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--rounds", type=int, default=3)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--mass_range", type=float, nargs=2, default=(0.3, 1.5))
parser.add_argument("--box_mu", type=float, nargs=2, default=(0.1, 0.8))
parser.add_argument("--plate_mu", type=float, nargs=2, default=(0.3, 0.5))
parser.add_argument("--out", required=True)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import dual_fr3_lab.tasks  # noqa: E402,F401
from dual_fr3_lab.tasks.manager_based.lift_plate import joint_pos_env_cfg as C  # noqa: E402

LIFT_M = 0.03

env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
agent_cfg = handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point"), metadata.version("rsl-rl-lib"))
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
# startup DR overrides (labels): absolute mass range, friction ranges
env_cfg.events.randomize_box_density.params.update({"mass_distribution_params": tuple(args.mass_range), "operation": "abs"})
env_cfg.events.randomize_box_friction.params.update(static_friction_range=tuple(args.box_mu),
                                                    dynamic_friction_range=(args.box_mu[0] * 0.85, args.box_mu[1] * 0.85))
env_cfg.events.randomize_plate_friction.params.update(static_friction_range=tuple(args.plate_mu),
                                                      dynamic_friction_range=(args.plate_mu[0] * 0.85, args.plate_mu[1] * 0.85))
print("[collect] mass", env_cfg.events.randomize_box_density.params, "\n[collect] box mu", env_cfg.events.randomize_box_friction.params,
      "\n[collect] plate mu", env_cfg.events.randomize_plate_friction.params)
env = gym.make(args.task, cfg=env_cfg)
raw = env.unwrapped
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=raw.device)

robot, obj = raw.scene["robot"], raw.scene["object"]
jnames = [f"left_fr3_joint{i}" for i in range(1, 8)] + [f"right_fr3_joint{i}" for i in range(1, 8)]
jids, _ = robot.find_joints(jnames, preserve_order=True)
jids = torch.as_tensor(jids, device=raw.device)
cL, cR = raw.scene["left_plate_object_contact"], raw.scene["right_plate_object_contact"]
n, T = raw.num_envs, int(raw.max_episode_length)
ids64 = torch.arange(n, device=raw.device)


def rerandomize():
    """Re-apply the startup DR terms (mass, box/plate friction) at runtime so every round has new labels."""
    em = raw.event_manager
    terms = dict(zip(em._mode_term_names["startup"], em._mode_term_cfgs["startup"]))
    for name in ("randomize_box_density", "randomize_box_friction", "randomize_plate_friction"):
        t = terms[name]
        if hasattr(t.func, "material_buckets") or hasattr(getattr(t.func, "impl", None), "material_buckets"):
            impl = getattr(t.func, "impl", t.func)
            # resample the material buckets inside the new range (term samples buckets once at __init__)
            lo, hi = t.params["static_friction_range"]
            dlo, dhi = t.params["dynamic_friction_range"]
            b = impl.material_buckets
            b[:, 0].uniform_(lo, hi)
            b[:, 1].uniform_(dlo, dhi)
        ok = False
        for ids in (ids64.to(torch.int32), ids64):
            try:
                t.func(raw, ids, **t.params)
                ok = True
                break
            except Exception as e:  # noqa: BLE001
                err = e
        if not ok:
            raise RuntimeError(f"{name}: {err}")
    raw.__dict__.pop("_box_dr_cache", None)


def labels():
    mass = obj.data.body_mass.torch[:, 0].to(raw.device)
    mu_box = C._static_friction_of(raw, obj)
    mu_L = C._static_friction_of(raw, robot, "left_fr3_link7")
    mu_R = C._static_friction_of(raw, robot, "right_fr3_link7")
    return torch.stack([mass, mu_box, mu_L, mu_R], -1)


def contact_n(sensor):
    f = sensor.data.force_matrix_w
    return f.reshape(n, -1, 3).sum(1).norm(dim=-1)


X_all, L_all, Y_all, OK_all = [], [], [], []
for r in range(args.rounds):
    if r > 0:
        rerandomize()
    y = labels()
    print(f"[collect] round {r}: mass {y[:, 0].min():.2f}~{y[:, 0].max():.2f}  mu_box {y[:, 1].min():.2f}~{y[:, 1].max():.2f}"
          f"  mu_L {y[:, 2].min():.2f}~{y[:, 2].max():.2f}  mu_R {y[:, 3].min():.2f}~{y[:, 3].max():.2f}")
    for e in range(args.episodes):
        if e == 0 and r == 0:
            obs = env.get_observations()
        else:
            env.reset()
            obs = env.get_observations()
        X = torch.zeros(n, T, 14 * 4 + 4, device=raw.device)
        lifted = torch.zeros(n, T, dtype=torch.bool, device=raw.device)
        ok = torch.ones(n, dtype=torch.bool, device=raw.device)
        z0 = (obj.data.root_pos_w.torch[:, 2] - raw.scene.env_origins[:, 2]).clone()
        for t in range(T - 1):
            with torch.no_grad():
                act = policy(obs)
            q = robot.data.joint_pos.torch[:, jids]
            dq = robot.data.joint_vel.torch[:, jids]
            tau = robot.data.applied_torque.torch[:, jids]
            bz = obj.data.root_pos_w.torch[:, 2] - raw.scene.env_origins[:, 2]
            bvz = obj.data.root_lin_vel_w.torch[:, 2]
            X[:, t] = torch.cat([q, dq, tau, contact_n(cL)[:, None], contact_n(cR)[:, None], (bz - z0)[:, None], bvz[:, None], act], -1)
            lifted[:, t] = bz - z0 > LIFT_M
            obs, _, dones, _ = env.step(act)
            if t < T - 2:
                ok &= ~dones.bool()
        X_all.append(X.cpu().numpy())
        L_all.append(lifted.cpu().numpy())
        Y_all.append(y.cpu().numpy())
        OK_all.append(ok.cpu().numpy())
        print(f"[collect]   episode {e}: lifted at end {lifted[:, T - 2].float().mean():.2f}  ok {ok.float().mean():.2f}")

np.savez_compressed(args.out, X=np.concatenate(X_all), lifted=np.concatenate(L_all), y=np.concatenate(Y_all), ok=np.concatenate(OK_all),
                    dt=raw.step_dt, cols="q14 dq14 tau14 fL fR box_dz box_vz act14")
print("[collect] wrote", args.out, "N =", sum(len(x) for x in X_all), "T =", T)
env.close()
app.close()
