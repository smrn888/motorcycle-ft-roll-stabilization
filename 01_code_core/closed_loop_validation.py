"""
closed_loop_validation.py — Phase 4: Original MPC vs Neural MPC در حلقه‌ی بسته (healthy plant).

برای هر (condition, seed) هر کنترلر روی «دقیقاً همان» اپیزود اجرا می‌شود:
  همان seed ⇒ همان reset/سناریو/مرجع/نویز v,alpha؛ همان IC (rng اپیزود از seed ساخته می‌شود)،
  همان plant، همان طول اپیزود (MAX_STEPS) و همان معیار سقوط (|phi|>π/4).

کنترلرها:  mpc      = Original MPC (preset mid؛ failed ⇒ τ=0، مثل Teacher/محیط)
           nmpc     = Neural MPC  (--ckpt)
           mpc_hold = اختیاری (--hold): MPC با «آخرین u معتبر» هنگام failed
condition: nominal (reset معمولی) | mid (perturb متوسط) | hard (نزدیک مرز سقوط)

    python closed_loop_validation.py --ckpt neural_mpc.pt --episodes 100 --out cl_val
    python closed_loop_validation.py --ckpt neural_mpc.pt --episodes 5 --conds nominal --out cl_test   (تست سریع)

خروجی: <out>_episodes.csv ، <out>_summary.csv ، <out>_pairs.csv ، <out>_traj.npz
"""
import argparse, time
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="neural_mpc.pt")
ap.add_argument("--episodes", type=int, default=100)
ap.add_argument("--seed0", type=int, default=70000)          # جدا از train(10000..) و val(90000..)
ap.add_argument("--level", type=int, default=8)
ap.add_argument("--conds", nargs="*", default=["nominal", "hard"], choices=["nominal", "mid", "hard", "extreme"])
ap.add_argument("--iw", nargs="*", type=float, default=[1.0], help="Iw_scale fault levels (1.0=healthy)")
ap.add_argument("--hold", action="store_true", help="baseline سوم: MPC با hold هنگام failed")
ap.add_argument("--gate_surv_drop", type=float, default=0.03)
ap.add_argument("--gate_ratio", type=float, default=1.25)
ap.add_argument("--out", default="cl_val")
a = ap.parse_args()

import torch
from motorcycle_env_mpc19 import MotorcycleEnvMPC19
from motorcycle_dynamics import TAU_MAX
from train_neural_mpc import load_neural_mpc

torch.set_num_threads(1)
OK_STATUS = {"optimal", "optimal_inaccurate"}
BUCKET = {"nominal": 0, "mid": 1, "hard": 2, "extreme": 3}
PHI_MAX_IC = 0.60
IC_RANGE = {1: (0.20, 0.50), 2: (0.50, 1.20), 3: (0.70, 2.00)}   # 3=extreme: بیرون از پوشش train (OOD)
EARLY = 10                                                   # سقوط در ≤10 گام = شروع غیرقابل‌بازیابی


class Env(MotorcycleEnvMPC19):
    def start(self, seed, level, bucket):
        rng = np.random.default_rng([int(seed), 17])         # IC فقط تابع seed، نه کنترلر
        self.set_fault(iw_scale=getattr(self, "_iw", 1.0))
        self.set_curriculum_level(level)
        self.reset(seed=seed)
        Qphi, Qphid, Rv, N = self.decode_action(np.zeros(4, np.float32))
        self._target = np.array([Qphi, Qphid, self._mid4[2], Rv]); self._N_fixed = int(N)
        self._last_ok_u = np.zeros(2)
        if bucket in IC_RANGE:
            pr, dr = IC_RANGE[bucket]
            lim = 0.70 if bucket == 3 else PHI_MAX_IC
            self.state[0] = float(np.clip(self.state[0] + rng.uniform(-pr, pr), -lim, lim))
            self.state[1] = float(self.state[1] + rng.uniform(-dr, dr))

    def obs10(self):
        phi_d = self._desired_roll()
        return np.array([*self.state[:6], self.v, self.alpha, phi_d, self.state[0] - phi_d], np.float32)

    def mpc_control(self):
        t0 = time.perf_counter()
        phi_d0 = self._desired_roll()
        self._smoothed_weights = ((1 - self.LP_ALPHA) * self._smoothed_weights + self.LP_ALPHA * self._target)
        w = self._smoothed_weights
        Q_diag = np.array([w[0], w[1], w[2], w[2], w[2] * 0.1, w[2] * 0.1]); R_diag = np.array([w[3], w[3]])
        N = self._N_fixed
        if N != self.mpc.N:
            up = self.mpc.u_prev.value.copy(); self.mpc = self._solvers[N]
            self.mpc.reset_warm_start(); self.mpc.u_prev.value = up
        self._N_idx = self.HORIZONS.index(N)
        x_d = np.zeros(6); x_d[0] = phi_d0
        A_d, B_d, bias_d = self.dyn.linearise(self.state[0])
        self.mpc.update_model(A_d, B_d, bias_d)
        P_chol = self._dare_chol(A_d, B_d, Q_diag=Q_diag, R_diag=R_diag)
        u = np.clip(self.mpc.solve(x0=self.state, xd=x_d, Q_diag=Q_diag, R_diag=R_diag, P_chol=P_chol),
                    -TAU_MAX, TAU_MAX)
        return u, str(self.mpc.last_solve_status), (time.perf_counter() - t0) * 1e3

    def plant_step(self, u):
        self._steps += 1
        self.state = self._plant_dyn.step(self.state, u)      # plant (ممکن است faulty)؛ MPC مدل نامی self.dyn را می‌بیند
        self.v = float(np.clip(self.v + self.np_random.normal(0.0, self.V_NOISE), 1.0, 8.0))
        self.alpha = float(np.clip(self.alpha + self.np_random.normal(0.0, self.ALPHA_NOISE), -0.20, 0.20))
        crashed = abs(self.state[0]) > np.pi / 4
        return crashed, crashed or self._steps >= self.MAX_STEPS


def run_episode(env, model, ctrl, seed, bucket):
    env.start(seed, a.level, bucket)
    phi, U, err2, cms, fails = [], [], [], [], 0
    crashed = False
    while True:
        if ctrl == "nmpc":
            x = env.obs10()
            t0 = time.perf_counter()
            with torch.no_grad():
                u = np.clip(model(torch.from_numpy(x)[None])[0].numpy().astype(np.float64), -TAU_MAX, TAU_MAX)
            cms.append((time.perf_counter() - t0) * 1e3); u_apply = u
        else:
            u, st, ms = env.mpc_control(); cms.append(ms)
            if st in OK_STATUS:
                env._last_ok_u = u.copy(); u_apply = u
            else:
                fails += 1; u_apply = env._last_ok_u if ctrl == "mpc_hold" else u
        crashed, done = env.plant_step(u_apply)
        p = float(env.state[0]); err = p - env._desired_roll()
        phi.append(p); U.append(np.asarray(u_apply, float).copy()); err2.append(err ** 2)
        if done:
            break
    phi, U, cms = np.array(phi), np.array(U), np.array(cms)
    return dict(
        crashed=bool(crashed), steps=len(phi), rmse=float(np.sqrt(np.mean(err2))),
        peak_phi_deg=float(np.rad2deg(np.abs(phi).max())), mean_phi_deg=float(np.rad2deg(np.abs(phi).mean())),
        u_norm=float(np.linalg.norm(U, axis=1).mean()), u_absmax=float(np.abs(U).max()),
        sat_rate=float(np.mean(np.any(np.abs(U) >= 0.99 * TAU_MAX, axis=1))),
        du_mean=float(np.linalg.norm(np.diff(U, axis=0), axis=1).mean()) if len(U) > 1 else 0.0,
        comp_ms_mean=float(cms.mean()), comp_ms_p95=float(np.percentile(cms, 95)), comp_ms_max=float(cms.max()),
        fail_frac=fails / len(phi)), phi, U


env = Env()
model = load_neural_mpc(a.ckpt, "cpu")
ctrls = ["mpc", "nmpc"] + (["mpc_hold"] if a.hold else [])
rows, traj = [], {}
t_all = time.time()
LABELS = [(c if iw == 1.0 else f"{c}@iw{iw}", c, iw) for iw in a.iw for c in a.conds]
for cond, base, iw in LABELS:
    env._iw = iw
    for i in range(a.episodes):
        seed = a.seed0 + i
        for c in ctrls:
            r, phi, U = run_episode(env, model, c, seed, BUCKET[base])
            rows.append(dict(cond=cond, ctrl=c, seed=seed, **r))
            traj[f"{cond}|{c}|{seed}|phi"] = phi.astype(np.float32); traj[f"{cond}|{c}|{seed}|u"] = U.astype(np.float32)
        if (i + 1) % 10 == 0 or i == a.episodes - 1:
            print(f"[{cond}] {i + 1}/{a.episodes} {time.time() - t_all:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(a.out + "_episodes.csv", index=False)
D = pd.DataFrame(rows); D.to_csv(a.out + "_episodes.csv", index=False)
np.savez_compressed(a.out + "_traj.npz", **traj)

# بازماندن در «شروع غیرقابل‌بازیابی»: MPC اصلی در ≤EARLY گام سقوط کرده ⇒ seed از زیرمجموعه‌ی بدون-early حذف
early = {(r.cond, r.seed) for r in D[(D.ctrl == "mpc") & D.crashed & (D.steps <= EARLY)].itertuples()}
D["early"] = [(c, s) in early for c, s in zip(D.cond, D.seed)]


def agg(x):
    alive = x[~x.crashed]; ok = x[~x.early]
    return pd.Series(dict(
        n=len(x), surv=1 - x.crashed.mean(), surv_excl_early=1 - ok.crashed.mean() if len(ok) else np.nan,
        rmse_all=x.rmse.mean(), rmse_alive=alive.rmse.mean(), peak_phi=x.peak_phi_deg.mean(),
        mean_phi=x.mean_phi_deg.mean(), u_norm=x.u_norm.mean(), u_absmax=x.u_absmax.max(),
        sat=x.sat_rate.mean(), du=x.du_mean.mean(), ep_len=x.steps.mean(), fail_frac=x.fail_frac.mean(),
        ms_mean=x.comp_ms_mean.mean(), ms_p95=x.comp_ms_p95.mean(), ms_max=x.comp_ms_max.max()))


S = D.groupby(["cond", "ctrl"]).apply(agg, include_groups=False)
S.to_csv(a.out + "_summary.csv")
pd.set_option("display.width", 250)
print("\n=== SUMMARY ===\n", S.round(4).to_string())

# ── paired: همان seed، mpc در برابر nmpc ───────────────────────────────
prow, gate_ok = [], True
for cond, _, _ in LABELS:
    P = D[(D.cond == cond)].pivot(index="seed", columns="ctrl", values="crashed")
    both = int((~P.mpc & ~P.nmpc).sum()); only_m = int((~P.mpc & P.nmpc).sum())
    only_n = int((P.mpc & ~P.nmpc).sum()); none = int((P.mpc & P.nmpc).sum())
    print(f"\n[{cond}] paired survival: both={both} only_MPC={only_m} only_NMPC={only_n} both_crash={none}")
    for seed in P.index:
        p1, p2 = traj[f"{cond}|mpc|{seed}|phi"], traj[f"{cond}|nmpc|{seed}|phi"]
        u1, u2 = traj[f"{cond}|mpc|{seed}|u"], traj[f"{cond}|nmpc|{seed}|u"]
        n = min(len(p1), len(p2))
        prow.append(dict(cond=cond, seed=seed, n_common=n,
                         dphi_rmse_deg=float(np.rad2deg(np.sqrt(np.mean((p1[:n] - p2[:n]) ** 2)))),
                         dphi_max_deg=float(np.rad2deg(np.abs(p1[:n] - p2[:n]).max())),
                         du_rmse=float(np.sqrt(np.mean((u1[:n] - u2[:n]) ** 2)))))
    s_m, s_n = S.loc[(cond, "mpc")], S.loc[(cond, "nmpc")]
    chk = dict(surv=s_n.surv >= s_m.surv - a.gate_surv_drop,
               rmse=s_n.rmse_alive <= a.gate_ratio * s_m.rmse_alive,
               peak=s_n.peak_phi <= a.gate_ratio * s_m.peak_phi)
    gate_ok &= all(chk.values())
    print(f"[{cond}] gate  survival:{'PASS' if chk['surv'] else 'FAIL'}  rmse_alive:{'PASS' if chk['rmse'] else 'FAIL'}  "
          f"peak_phi:{'PASS' if chk['peak'] else 'FAIL'}  | speedup (mean compute) = "
          f"{s_m.ms_mean / max(s_n.ms_mean, 1e-9):.1f}x")
Pf = pd.DataFrame(prow); Pf.to_csv(a.out + "_pairs.csv", index=False)
print("\n=== TRAJECTORY DIFF (mpc vs nmpc، گام‌های مشترک) ===")
print(Pf.groupby("cond")[["dphi_rmse_deg", "dphi_max_deg", "du_rmse"]].agg(["mean", "median", "max"]).round(3).to_string())
print("\nGATE (آستانه‌ها: surv_drop<=%.2f، ratio<=%.2f):" % (a.gate_surv_drop, a.gate_ratio), "PASS" if gate_ok else "FAIL")
print("early-crash seeds (MPC اصلی ≤%d گام):" % EARLY, sorted(early))
