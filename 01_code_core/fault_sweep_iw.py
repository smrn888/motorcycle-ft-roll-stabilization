"""
fault_sweep_iw.py — Phase 0.5: sweep ریز روی Iw_scale با MPC ثابت (preset=mid).
    python fault_sweep_iw.py --episodes 30
خروجی:
    fault_sweep_iw_episodes.csv  (هر اپیزود)
    fault_sweep_iw_summary.csv   (خلاصه‌ی هر Iw_scale)
"""
import argparse, time
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=30)
ap.add_argument("--level", type=int, default=8)
ap.add_argument("--seed0", type=int, default=1000)
ap.add_argument("--iw", nargs="*", type=float,
                default=[1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
ap.add_argument("--out", default="fault_sweep_iw")
a = ap.parse_args()

from motorcycle_env_mpc19 import MotorcycleEnvMPC19
from motorcycle_dynamics import Ts

ACT = np.zeros(4, np.float32)          # preset mid


def run(env, seed):
    env.set_curriculum_level(a.level)
    env.reset(seed=seed)
    err2 = steps = n = sat = 0
    un = 0.0
    peak_phi = 0.0
    crashed = False
    ms = []
    while True:
        _, _, term, trunc, info = env.step(ACT)
        err2 += info["err_sq_sum"]
        steps += info["n_plant_steps"]
        un += info["u_norm"]
        sat += int(info["u_sat"])
        n += 1
        peak_phi = max(peak_phi, abs(info["phi"]))
        ms.append(info["solve_time_ms"])
        if term or trunc:
            crashed = bool(term)
            break
    return dict(
        crashed=crashed,
        survival_steps=steps,                                  # گام plant تا پایان/سقوط
        ttf_s=steps * Ts if crashed else np.nan,               # time-to-failure
        rmse=float(np.sqrt(err2 / max(steps, 1))),
        peak_phi_deg=float(np.rad2deg(peak_phi)),              # در انتهای هر تصمیم نمونه‌برداری شده
        u_norm=un / n, sat_rate=sat / n, solve_ms=float(np.mean(ms)),
    )


env = MotorcycleEnvMPC19()
rows = []
for iw in a.iw:
    env.set_fault(iw_scale=iw)
    t0 = time.time()
    for i in range(a.episodes):
        rows.append(dict(iw_scale=iw, ep=i, seed=a.seed0 + i, **run(env, a.seed0 + i)))
    d = pd.DataFrame([r for r in rows if r["iw_scale"] == iw])
    alive = d[~d.crashed]
    print(f"iw={iw:.2f} surv={1 - d.crashed.mean():.3f} "
          f"rmse_alive={alive.rmse.mean():.4f} steps={d.survival_steps.mean():.0f} "
          f"ttf={d.ttf_s.mean():.2f}s u={d.u_norm.mean():.2f} "
          f"sat={d.sat_rate.mean():.3f} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(a.out + "_episodes.csv", index=False)

D = pd.DataFrame(rows)
S = D.groupby("iw_scale").apply(lambda x: pd.Series(dict(
    surv=1 - x.crashed.mean(),
    rmse_alive=x[~x.crashed].rmse.mean(),
    rmse_all=x.rmse.mean(),
    mean_steps=x.survival_steps.mean(),
    mean_ttf_s=x.ttf_s.mean(),
    min_ttf_s=x.ttf_s.min(),
    peak_phi_deg=x.peak_phi_deg.mean(),
    u_norm=x.u_norm.mean(),
    sat=x.sat_rate.mean())), include_groups=False)
S.to_csv(a.out + "_summary.csv")
print("\n", S.round(3).to_string())
