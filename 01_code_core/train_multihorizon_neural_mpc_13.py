"""
train_neural_mpc.py (v2) — Phase 3: Neural MPC ≈ Original MPC (supervised imitation، بدون SAC/fault).

    python train_neural_mpc.py --train mpc_train.npz --val mpc_val.npz --out neural_mpc.pt

- normalization ورودی فقط از train حساب و داخل checkpoint ذخیره می‌شود؛ روی val عیناً همان اعمال می‌شود.
- مدل: 10→256→256→128→2 ،  u = TAU_MAX·tanh(z)  (یعنی ũ=u/50 ∈ [-1,1] یاد گرفته می‌شود).
- loss: MSE روی (u_nn − u_mpc)/σ_u با وزن بیشتر برای |phi|>15°.
  (σ_u≈0.72 Nm؛ اگر loss روی ũ=u/50 باشد مقدارش ~1e-4 می‌شود و نسبت به وزن‌دهی/ε بد‌شرط است.)
- early stopping روی val RMSE، best checkpoint، گزارش کامل: RMSE/MAE/p95/p99 هر actuator،
  و تفکیک |phi|≤15 / >15 / >25، و خطای تقارن τ_f+τ_r.
- کلاس NeuralMPC و load_neural_mpc برای Phase 4 قابل import است.
"""
import argparse, json, time
import numpy as np
import torch, torch.nn as nn

TAU_MAX = 50.0


class NeuralMPC(nn.Module):
    def __init__(self, mu, sd, act="relu"):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.as_tensor(sd, dtype=torch.float32))
        A = nn.ReLU if act == "relu" else nn.SiLU
        self.net = nn.Sequential(nn.Linear(len(mu), 256), A(), nn.Linear(256, 256), A(),
                                 nn.Linear(256, 128), A(), nn.Linear(128, 2))

    def forward(self, x):                     # x: (B,10) خام (قبل از normalization)
        return TAU_MAX * torch.tanh(self.net((x - self.mu) / self.sd))


def load_neural_mpc(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = NeuralMPC(ck["mu"], ck["sd"], ck["args"].get("act", "relu")).to(device)
    m.load_state_dict(ck["state"]); m.eval()
    return m


def load_npz(path, drop_inaccurate=False):
    d = np.load(path, allow_pickle=True)
    X, Y, EP = d["X"], d["Y"], d["EP"]
    assert X.ndim == 2 and X.shape[1] == 13, "Expected 13 input features including u_prev and horizon"
    if drop_inaccurate:
        m = d["STATUS"] == "optimal"
        X, Y, EP = X[m], Y[m], EP[m]
    return X.astype(np.float32), Y.astype(np.float32), EP


def error_report(e, phi_deg, sig_u):
    """e: (N,2) = u_nn − u_mpc [Nm]."""
    def stats(err, name):
        a = np.abs(err)
        return {f"{name}_rmse": float(np.sqrt((err ** 2).mean())), f"{name}_mae": float(a.mean()),
                f"{name}_p95": float(np.percentile(a, 95)), f"{name}_p99": float(np.percentile(a, 99)),
                f"{name}_max": float(a.max())}
    rep = {}
    for tag, m in (("all", np.ones(len(e), bool)), ("phi<=15", phi_deg <= 15),
                   ("phi>15", phi_deg > 15), ("phi>25", phi_deg > 25)):
        if not m.any():
            continue
        r = {"n": int(m.sum())}
        r.update(stats(e[m], "both"))
        r.update(stats(e[m][:, 0], "tau_f")); r.update(stats(e[m][:, 1], "tau_r"))
        r["rel_rmse"] = r["both_rmse"] / sig_u
        rep[tag] = r
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="mpc_train.npz")
    ap.add_argument("--val", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--act", choices=["relu", "silu"], default="relu")
    ap.add_argument("--w_hard", type=float, default=3.0)
    ap.add_argument("--drop_inaccurate", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="neural_mpc.pt")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr, Ytr, EPtr = load_npz(a.train, a.drop_inaccurate)
    if a.val:
        Xva, Yva, _ = load_npz(a.val, a.drop_inaccurate)
    else:                                       # hold-out بر اساس اپیزود (نه نمونه)
        ue = np.unique(EPtr)
        ve = np.random.default_rng(a.seed).choice(ue, size=max(1, len(ue) // 10), replace=False)
        mv = np.isin(EPtr, ve)
        Xva, Yva, Xtr, Ytr = Xtr[mv], Ytr[mv], Xtr[~mv], Ytr[~mv]
    assert Xtr.shape[1] == Xva.shape[1] == 13
    assert not set(np.unique(Xtr[:,-1])) ^ set(np.unique(Xva[:,-1])), "Train/val horizon mismatch"
    print(f"train={len(Xtr)} val={len(Xva)} device={dev}", flush=True)

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6      # فقط از train
    sig_u = max(float(Ytr.std()), 1e-6)
    assert len(Xtr) and len(Xva), "Empty train or validation split"
    model = NeuralMPC(mu, sd, a.act).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=7, min_lr=1e-6)

    tt = lambda z: torch.as_tensor(z, device=dev)
    xtr, ytr, xva, yva = tt(Xtr), tt(Ytr), tt(Xva), tt(Yva)
    wtr = 1.0 + (a.w_hard - 1.0) * (xtr[:, 0].abs() > np.deg2rad(15)).float()
    phi_va = np.rad2deg(np.abs(Xva[:, 0]))
    nb = int(np.ceil(len(xtr) / a.bs))

    def predict_val():
        model.eval()
        with torch.no_grad():
            e = (model(xva) - yva).cpu().numpy()
        model.train()
        return e

    best, bad, t0 = 1e9, 0, time.time()
    for ep in range(a.epochs):
        perm = torch.randperm(len(xtr), device=dev); tot = 0.0
        for i in range(nb):
            idx = perm[i * a.bs:(i + 1) * a.bs]
            loss = (wtr[idx] * (((model(xtr[idx]) - ytr[idx]) / sig_u) ** 2).mean(1)).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += float(loss)
        e = predict_val(); rmse = float(np.sqrt((e ** 2).mean()))
        sched.step(rmse)
        m25 = phi_va > 25
        if rmse < best - 1e-5:
            best, bad = rmse, 0
            torch.save(dict(state=model.state_dict(), mu=mu, sd=sd, sig_u=sig_u,
                            tau_max=TAU_MAX, args=vars(a), epoch=ep), a.out)
        else:
            bad += 1
        if ep % 5 == 0 or bad == 0:
            print(f"ep {ep:3d} loss={tot / nb:.4f} val_rmse={rmse:.4f}Nm rel={rmse / sig_u:.3f} "
                  f">25°={np.sqrt((e[m25] ** 2).mean()) if m25.any() else float('nan'):.3f} "
                  f"best={best:.4f} lr={opt.param_groups[0]['lr']:.2e} {time.time() - t0:.0f}s", flush=True)
        if bad >= a.patience:
            print(f"early stop at ep {ep}"); break

    # ── گزارش نهایی با best checkpoint ─────────────────────────────────
    model = load_neural_mpc(a.out, dev)
    with torch.no_grad():
        e = (model(xva) - yva).cpu().numpy()
        up = model(xva).cpu().numpy()
    rep = error_report(e, phi_va, sig_u)
    rep["sym_err_mean_abs(tau_f+tau_r)"] = float(np.abs(up[:, 0] + up[:, 1]).mean())
    rep["sigma_u"] = sig_u
    for n in sorted(np.unique(Xva[:,-1])):
        m = Xva[:,-1] == n
        rep[f"N{int(n)}"] = error_report(e[m], phi_va[m], sig_u)["all"]
    print("\n=== VALIDATION REPORT (best checkpoint) ===")
    for tag in ("all", "phi<=15", "phi>15", "phi>25"):
        if tag in rep:
            r = rep[tag]
            print(f"{tag:8s} n={r['n']:6d} rmse={r['both_rmse']:.4f} rel={r['rel_rmse']:.3f} "
                  f"mae={r['both_mae']:.4f} p95={r['both_p95']:.3f} p99={r['both_p99']:.3f} "
                  f"max={r['both_max']:.2f} | f_rmse={r['tau_f_rmse']:.4f} r_rmse={r['tau_r_rmse']:.4f}")
    for n in sorted(np.unique(Xva[:,-1])):
        r=rep[f"N{int(n)}"]
        print(f"N={int(n)} val_n={r['n']} rmse={r['both_rmse']:.4f} Nm p99={r['both_p99']:.3f}")
    print("sym err |tau_f+tau_r| (NN):", round(rep["sym_err_mean_abs(tau_f+tau_r)"], 5))
    json.dump(rep, open(a.out.replace(".pt", "_report.json"), "w"), indent=1)
    print("saved:", a.out, "and", a.out.replace(".pt", "_report.json"))


if __name__ == "__main__":
    main()