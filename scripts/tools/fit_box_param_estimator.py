"""Offline estimation study on collect_box_torque_history.py data: from a short window of robot signals around lift onset,
how well can box mass and contact friction be regressed? (step 2 of the torque-adaptation idea, 11-13)

Feature sets (what a deployed estimator could see):
  qdqa     q, dq, action                       (= what the -History policy sees)
  tau      + applied joint torque (14)         (real: tau_J minus gravity model ~ tau_ext)
  taubox   + box dz, box vz                    (real: vision box pose)
  tauonly  joint torque only
Windows end at t_lift + k steps (k = 0, 5, 10, 20 -> 0, 0.25, 0.5, 1.0 s) and span W steps. Episodes without a lift use
the window ending at the last step (the "it never came up" case is itself informative for mass).
Targets: mass [kg], mu_box, mu_contact = mean over sides of (mu_box + mu_plate)/2 (PhysX average combine).
Reports MAE / R^2 on a held-out 20 % split (mean of 3 seeds), plus a variant with real-like torque corruption
(per-episode constant bias N(0, bias_nm) on every joint + white noise) applied to train and test.

    python scripts/tools/fit_box_param_estimator.py --data <npz> [--window 10] [--bias_nm 1.0] [--noise_nm 0.3] --out <csv>
"""
import argparse
import csv

import numpy as np
import torch
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--data", nargs="+", required=True)
ap.add_argument("--window", type=int, default=10)
ap.add_argument("--ks", type=int, nargs="+", default=(0, 5, 10, 20))
ap.add_argument("--bias_nm", type=float, default=1.0)
ap.add_argument("--noise_nm", type=float, default=0.3)
ap.add_argument("--epochs", type=int, default=400)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--out", required=True)
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

Xs, Ls, Ys = [], [], []
for p in args.data:
    d = np.load(p)
    Xs.append(d["X"]); Ls.append(d["lifted"]); Ys.append(d["y"])
X, L, Y = np.concatenate(Xs), np.concatenate(Ls), np.concatenate(Ys)
N, T, D = X.shape
print(f"episodes {N}, steps {T}, feat {D}; lifted-ever {L.any(1).mean():.2f}")
SL = {"q": slice(0, 14), "dq": slice(14, 28), "tau": slice(28, 42), "f": slice(42, 44), "box": slice(44, 46), "act": slice(46, 60)}
FEATS = {
    "qdqa": ["q", "dq", "act"],
    "tau": ["q", "dq", "act", "tau"],
    "taubox": ["q", "dq", "act", "tau", "box"],
    "tauonly": ["tau"],
}
mass, mu_box = Y[:, 0], Y[:, 1]
mu_c = 0.5 * ((mu_box + Y[:, 2]) / 2 + (mu_box + Y[:, 3]) / 2)
TARGETS = {"mass": mass, "mu_box": mu_box, "mu_contact": mu_c}
t_lift = np.array([np.argmax(l) if l.any() else T - 2 for l in L])


def windows(k, corrupt, seed):
    rng = np.random.default_rng(seed)
    Xc = X.copy()
    if corrupt:
        Xc[:, :, SL["tau"]] += rng.normal(0, args.bias_nm, (N, 1, 14)) + rng.normal(0, args.noise_nm, (N, T, 14))
    end = np.clip(t_lift + k, args.window, T - 2)
    idx = end[:, None] - args.window + 1 + np.arange(args.window)[None]
    return Xc[np.arange(N)[:, None], idx]  # (N, W, D)


def fit(F, y, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    ntr = int(0.8 * N)
    tr, te = perm[:ntr], perm[ntr:]
    F = torch.tensor(F, dtype=torch.float32, device=dev).reshape(N, -1)
    y = torch.tensor(y, dtype=torch.float32, device=dev)
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    ym, ys = y[tr].mean(), y[tr].std() + 1e-6
    Fn, yn = (F - mu) / sd, (y - ym) / ys
    net = nn.Sequential(nn.Linear(Fn.shape[1], 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    for ep in range(args.epochs):
        b = torch.tensor(rng.choice(tr, min(256, ntr), replace=False), device=dev)
        loss = ((net(Fn[b]).squeeze(-1) - yn[b]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = net(Fn[te]).squeeze(-1) * ys + ym
    err = (pred - y[te]).abs()
    r2 = 1 - ((pred - y[te]) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
    return err.mean().item(), r2.item(), (y[te] - y[te].mean()).abs().mean().item()


rows = []
for k in args.ks:
    for corrupt in (False, True):
        for fname, parts in FEATS.items():
            for tname, y in TARGETS.items():
                maes, r2s, base = [], [], []
                for s in range(args.seeds):
                    Fw = windows(k, corrupt, s)
                    F = np.concatenate([Fw[:, :, SL[p]] for p in parts], -1)
                    m, r, b = fit(F, y, s)
                    maes.append(m); r2s.append(r); base.append(b)
                rows.append({"k_steps": k, "corrupt": int(corrupt), "feats": fname, "target": tname,
                             "mae": np.mean(maes), "r2": np.mean(r2s), "mae_mean_pred": np.mean(base)})
                print(f"k={k:2d} corrupt={int(corrupt)} {fname:8s} {tname:10s} MAE {np.mean(maes):.3f} (mean-pred {np.mean(base):.3f})  R2 {np.mean(r2s):.2f}")
with open(args.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print("wrote", args.out)
