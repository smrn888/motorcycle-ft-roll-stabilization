"""
generate_mpc_dataset.py — Phase 2: دیتاست Teacher برای Neural MPC  (نسخه‌ی v2: فیلتر failed)

Teacher = MPC فعلی (preset=mid)، plant سالم، بدون RL.
برای هر «گام plant» قبل از solve:
    X  (N,10) = [phi, phidot, df, dr, ddf, ddr, v, alpha, phi_d, err]
    Y  (N,2)  = [tau_f, tau_r] خروجی MPC بعد از clip به ±TAU_MAX
فقط نمونه‌هایی که status آن‌ها در OK_STATUS است وارد X/Y می‌شوند.
نمونه‌های بد (failed/infeasible/...) جدا ذخیره می‌شوند (F_*) و فقط برای تشخیص‌اند، نه training.

on_fail: وقتی solver شکست می‌خورد MPC خروجی صفر می‌دهد و plant با همین صفر جلو می‌رود.
    zero (پیش‌فرض) = وفادار به رفتار Teacher/Original MPC در Phase 4 (rollout همان است)
    hold           = آخرین u معتبر به plant داده شود (rollout تمیزتر؛ Teacher کمی فرق می‌کند)
در هر دو حالت label نمونه‌ی failed حذف می‌شود.

    python generate_mpc_dataset.py --episodes 30   --out mpc_sanity30b.npz
    python generate_mpc_dataset.py --episodes 1500 --seed0 10000 --out mpc_train.npz
    python generate_mpc_dataset.py --episodes 200  --seed0 90000 --out mpc_val.npz
"""
import argparse, json, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=5)
ap.add_argument("--seed0", type=int, default=10000)
ap.add_argument("--levels", nargs="*", type=int, default=[8])
ap.add_argument("--p_buckets", nargs=3, type=float, default=[0.4, 0.3, 0.3])
ap.add_argument("--on_fail", choices=["zero", "hold"], default="zero")
ap.add_argument("--max_fail_frac", type=float, default=0.10)
ap.add_argument("--out", default="mpc_dataset.npz")
a = ap.parse_args()

from motorcycle_env_mpc19 import MotorcycleEnvMPC19
from motorcycle_dynamics import TAU_MAX

OK_STATUS = {"optimal", "optimal_inaccurate"}
PHI_MAX_IC = 0.60
IC_RANGE = {1: (0.20, 0.50), 2: (0.50, 1.20)}


class TeacherEnv(MotorcycleEnvMPC19):
    """همان inner-loop در MotorcycleEnvMPC19.step، یک گام plant، با ثبت u و status."""

    def start(self, seed, level, bucket, rng):
        self.set_fault()
        self.set_curriculum_level(level)
        self.reset(seed=seed)
        Qphi, Qphid, Rv, N = self.decode_action(np.zeros(4, np.float32))
        self._target = np.array([Qphi, Qphid, self._mid4[2], Rv])
        self._N_fixed = int(N)
        self._last_ok_u = np.zeros(2)
        if bucket in IC_RANGE:
            pr, dr = IC_RANGE[bucket]
            self.state[0] = float(np.clip(self.state[0] + rng.uniform(-pr, pr),
                                          -PHI_MAX_IC, PHI_MAX_IC))
            self.state[1] = float(self.state[1] + rng.uniform(-dr, dr))

    def teacher_step(self):
        self._steps += 1
        phi0 = self.state[0]
        phi_d0 = self._desired_roll()
        x = np.array([*self.state[:6], self.v, self.alpha, phi_d0, phi0 - phi_d0],
                     dtype=np.float32)

        self._smoothed_weights = ((1.0 - self.LP_ALPHA) * self._smoothed_weights
                                  + self.LP_ALPHA * self._target)
        w = self._smoothed_weights
        Q_diag = np.array([w[0], w[1], w[2], w[2], w[2] * 0.1, w[2] * 0.1])
        R_diag = np.array([w[3], w[3]])

        N_new = self._N_fixed
        if N_new != self.mpc.N:
            u_prev_old = self.mpc.u_prev.value.copy()
            self.mpc = self._solvers[N_new]
            self.mpc.reset_warm_start()
            self.mpc.u_prev.value = u_prev_old
        self._N_idx = self.HORIZONS.index(N_new)

        x_d = np.zeros(6); x_d[0] = phi_d0
        A_d, B_d, bias_d = self.dyn.linearise(self.state[0])
        self.mpc.update_model(A_d, B_d, bias_d)
        P_chol = self._dare_chol(A_d, B_d, Q_diag=Q_diag, R_diag=R_diag)
        u_prev = np.array(self.mpc.u_prev.value, dtype=np.float32).reshape(2).copy()

        u = np.clip(self.mpc.solve(x0=self.state, xd=x_d, Q_diag=Q_diag,
                                   R_diag=R_diag, P_chol=P_chol), -TAU_MAX, TAU_MAX)
        status = str(self.mpc.last_solve_status)
        ok = status in OK_STATUS

        if ok:
            self._last_ok_u = u.copy()
            u_apply = u
        else:
            u_apply = self._last_ok_u if a.on_fail == "hold" else u
        self.state = self.dyn.step(self.state, u_apply)
        self.v = float(np.clip(self.v + self.np_random.normal(0.0, self.V_NOISE), 1.0, 8.0))
        self.alpha = float(np.clip(
            self.alpha + self.np_random.normal(0.0, self.ALPHA_NOISE), -0.20, 0.20))

        crashed = abs(self.state[0]) > np.pi / 4
        done = crashed or self._steps >= self.MAX_STEPS
        return x, u.astype(np.float32), u_prev, status, ok, crashed, done


env = TeacherEnv()
rng = np.random.default_rng(a.seed0)
p = np.asarray(a.p_buckets, float); p /= p.sum()

X, Y, UP, EP, T, SEED, BK, ST = [], [], [], [], [], [], [], []      # فقط ok
FX, FEP, FT, FBK, FST = [], [], [], [], []                          # failed (تشخیصی)
ep_info = []
t0 = time.time()
for i in range(a.episodes):
    seed = a.seed0 + i
    bucket = int(rng.choice(3, p=p))
    level = int(rng.choice(a.levels))
    env.start(seed, level, bucket, rng)
    t = n_fail = run = max_run = 0
    while True:
        x, u, up, st, ok, crashed, done = env.teacher_step()
        if ok:
            X.append(x); Y.append(u); UP.append(up); EP.append(i); T.append(t)
            SEED.append(seed); BK.append(bucket); ST.append(st)
            run = 0
        else:
            FX.append(x); FEP.append(i); FT.append(t); FBK.append(bucket); FST.append(st)
            n_fail += 1; run += 1; max_run = max(max_run, run)
        t += 1
        if done:
            break
    ep_info.append(dict(ep=i, seed=seed, bucket=bucket, level=level, steps=t,
                        crashed=bool(crashed), n_fail=n_fail, max_fail_run=max_run))
    if (i + 1) % 10 == 0 or i == a.episodes - 1:
        print(f"ep {i + 1}/{a.episodes} ok={len(X)} failed={len(FX)} {time.time() - t0:.0f}s",
              flush=True)

X = np.stack(X); Y = np.stack(Y); UP = np.stack(UP)
FX = np.stack(FX) if FX else np.zeros((0, 10), np.float32)
np.savez_compressed(a.out, X=X, Y=Y, U_PREV=UP, EP=np.array(EP), T=np.array(T),
                    SEED=np.array(SEED), BUCKET=np.array(BK), STATUS=np.array(ST),
                    F_X=FX, F_EP=np.array(FEP), F_T=np.array(FT), F_BUCKET=np.array(FBK),
                    F_STATUS=np.array(FST))
json.dump(dict(args=vars(a), ok_status=sorted(OK_STATUS), episodes=ep_info),
          open(a.out.replace(".npz", "_meta.json"), "w"), indent=1)

# ── sanity report ────────────────────────────────────────────────────────
n_ok, n_bad = len(X), len(FX)
print("\n=== SANITY ===")
print(f"total={n_ok + n_bad}  ok(kept)={n_ok}  failed(dropped)={n_bad}  "
      f"fail_frac={n_bad / max(n_ok + n_bad, 1):.4f}   on_fail={a.on_fail}")
allst = np.array(ST + FST)
vals, cnt = np.unique(allst, return_counts=True)
print("solver status (all):", dict(zip(vals.tolist(), cnt.tolist())))
print("X", X.shape, "Y", Y.shape, "| NaN/Inf X:", (~np.isfinite(X)).sum(), "Y:", (~np.isfinite(Y)).sum())
print(f"|Y|max={np.abs(Y).max():.3f} (حد {TAU_MAX})  "
      f"exact-zero rows in Y (باید ≈0): {int(np.sum(np.all(Y == 0, axis=1)))}  "
      f"sat={np.mean(np.any(np.abs(Y) >= 0.99 * TAU_MAX, axis=1)):.3f}")
phi_deg = np.rad2deg(np.abs(X[:, 0]))
print("kept |phi| deg p50/p90/p99/max:", np.percentile(phi_deg, [50, 90, 99]).round(2), phi_deg.max().round(2))
print("kept |phidot| p50/p90/p99/max:", np.percentile(np.abs(X[:, 1]), [50, 90, 99]).round(3),
      np.abs(X[:, 1]).max().round(3))
print(f"kept samples |phi|>25deg: {np.mean(phi_deg > 25):.3f}   >15deg: {np.mean(phi_deg > 15):.3f}")
if n_bad:
    fphi = np.rad2deg(np.abs(FX[:, 0]))
    print(f"failed |phi| deg median={np.median(fphi):.2f} max={fphi.max():.2f}  "
          f"failed t<10: {np.mean(np.array(FT) < 10):.3f}")
    print("failed by bucket:", np.bincount(FBK, minlength=3),
          " episodes with ≥1 fail:", sum(e['n_fail'] > 0 for e in ep_info), "/", len(ep_info),
          " max consecutive fail run:", max(e['max_fail_run'] for e in ep_info))
print("episodes:", len(ep_info), " crashed:", sum(e['crashed'] for e in ep_info),
      " bucket counts:", np.bincount([e['bucket'] for e in ep_info], minlength=3))
assert np.isfinite(X).all() and np.isfinite(Y).all()
assert np.abs(Y).max() <= TAU_MAX + 1e-3
if n_bad / max(n_ok + n_bad, 1) > a.max_fail_frac:
    print(f"WARNING: fail_frac > {a.max_fail_frac}؛ قبل از train علت failed را بررسی کن")
