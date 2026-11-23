"""
fault_sweep_nmpc.py — sweep ریز Iw_scale فقط برای Neural MPC (residual=0)، بدون solver ⇒ خیلی سریع.
    python fault_sweep_nmpc.py --ckpt neural_mpc.pt --episodes 100 --out fault_sweep_nmpc
- همان seedها و bucket=nominal مثل cl_fault (seed0=73000) تا با جدول Original MPC قابل مقایسه باشد.
- «شروع غیرقابل‌بازیابی» = سقوط در ≤10 گام در iw=1.0 (همان seedها از بقیه‌ی iw کنار گذاشته می‌شوند در surv_excl_early).
"""
import argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="neural_mpc.pt")
ap.add_argument("--episodes", type=int, default=100)
ap.add_argument("--seed0", type=int, default=73000)
ap.add_argument("--bucket", type=int, default=0)
ap.add_argument("--iw", nargs="*", type=float, default=[0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.42, 0.4, 0.35, 0.3])
ap.add_argument("--out", default="fault_sweep_nmpc")
a = ap.parse_args()

from motorcycle_env_residual import MotorcycleEnvResidual
from motorcycle_dynamics import Ts, TAU_MAX

env = MotorcycleEnvResidual(ckpt=a.ckpt, iw_list=(1.0,))
Z = np.zeros(2, np.float32)
EARLY = 10


def run(seed, iw):
    env.reset(seed=seed, options=dict(iw=iw, bucket=a.bucket))
    phi, U, e2, n = [], [], 0.0, 0
    while True:
        _, _, te, tr, info = env.step(Z)
        phi.append(info["phi"]); U.append(info["u_final"]); e2 += info["err_sq_sum"]; n += info["n_plant_steps"]
        if te or tr:
            break
    phi, U = np.array(phi), np.array(U)
    return dict(crashed=bool(te), steps=len(phi), rmse=float(np.sqrt(e2 / n)),
                peak_phi_deg=float(np.rad2deg(np.abs(phi).max())), mean_phi_deg=float(np.rad2deg(np.abs(phi).mean())),
                u_norm=float(np.linalg.norm(U, axis=1).mean()), u_absmax=float(np.abs(U).max()),
                du=float(np.linalg.norm(np.diff(U, axis=0), axis=1).mean()) if len(U) > 1 else 0.0)


iws = [1.0] + [x for x in a.iw if x != 1.0]
rows = []
for iw in iws:
    for i in range(a.episodes):
        rows.append(dict(iw=iw, seed=a.seed0 + i, **run(a.seed0 + i, iw)))
    d = pd.DataFrame([r for r in rows if r["iw"] == iw])
    print(f"iw={iw:.2f} surv={1 - d.crashed.mean():.3f} rmse_all={d.rmse.mean():.3f} steps={d.steps.mean():.0f}", flush=True)
D = pd.DataFrame(rows); D.to_csv(a.out + "_episodes.csv", index=False)
early = set(D[(D.iw == 1.0) & D.crashed & (D.steps <= EARLY)].seed)
D["early"] = D.seed.isin(early)


def agg(x):
    ok = x[~x.early]; alive = x[~x.crashed]; cr = ok[ok.crashed]
    return pd.Series(dict(n=len(x), surv=1 - x.crashed.mean(), surv_excl_early=1 - ok.crashed.mean(),
                          rmse_all=ok.rmse.mean(), rmse_alive=alive.rmse.mean(), mean_steps=ok.steps.mean(),
                          ttf_s=cr.steps.mean() * Ts if len(cr) else np.nan, peak_phi=ok.peak_phi_deg.mean(),
                          u_norm=ok.u_norm.mean(), u_absmax=ok.u_absmax.max(), du=ok.du.mean()))


S = D.groupby("iw").apply(agg, include_groups=False)
S.to_csv(a.out + "_summary.csv")
pd.set_option("display.width", 220)
print("\n=== Neural MPC fault sweep (residual=0) ===\n", S.round(3).to_string())
print("\nunrecoverable-start seeds (iw=1.0، ≤%d گام):" % EARLY, sorted(early))
