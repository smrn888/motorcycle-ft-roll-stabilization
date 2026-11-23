"""
motorcycle_env_mpc18.py
=======================
RL = «استراتژیست adaptation» برای MPC (نه residual، نه کنترل‌کننده‌ی torque).

زیرکلاس MotorcycleEnvMPC (v17.1)؛ فایل v17 دست‌نخورده می‌ماند تا baselineها
و SIL قبلی همان‌طور کار کنند. MPC، plant، خطی‌سازی آنلاین، DARE و solver pool
همان v17 هستند.

تفاوت‌ها:
  1) Action سه‌بعدی در [-1,1]:   a = [a_Qphi, a_Qphidot, a_N]
        Qphi    = log-map  [10, 500]
        Qphidot = log-map  [1, 50]
        N       = یکی از HORIZONS (گسسته، از روی a_N)
     Qjoint و R ثابت = میانگین هندسی بازه (همان وزن‌های «mid»).
  2) RL با ۲۰ هرتز تصمیم می‌گیرد: هر step محیط = DECISION_K گام plant/MPC
     (K=5 ⇒ ۵۰ ms با Ts=10 ms). پارامترها بین دو تصمیم ثابت می‌مانند
     (وزن‌ها از فیلتر پایین‌گذر می‌گذرند). MAX_STEPS همان ۵۰۰ گام plant است
     ⇒ هر اپیزود ≤ ۱۰۰ تصمیم.
  3) افق تطبیقی rule-based v17 (|phi|→15/20/40) حذف شده؛ N را RL انتخاب می‌کند.
  4) reward همیشه v2 است (هر گام plant ∈ [0,2]، سقوط = -CRASH_PEN). reward هر
     step محیط = مجموع reward گام‌های plant آن تصمیم.
  5) observation ۱۳‌بعدی: ۱۰ عدد v17 + [Qphi, Qphidot, N] فعلی (نرمال [-1,1])،
     چون وزن‌های فیلترشده و افق فعلی بخشی از حالت مؤثر سیستم‌اند.
  6) جمله‌ی اختیاری horizon_cost (پیش‌فرض ۰) برای پاداش دادن به افق کوتاه‌تر.

برای SIL/ارزیابی: همین کلاس را مستقیم اجرا کن (action سه‌بعدی بده) یا از
decode_action() و DECISION_K استفاده کن.
"""

import numpy as np
import gymnasium as gym

from motorcycle_env_mpc17 import MotorcycleEnvMPC
from motorcycle_dynamics import TAU_MAX


class MotorcycleEnvMPC18(MotorcycleEnvMPC):

    HORIZONS     = (10, 15, 20, 30, 40)   # solver pool (هر کدام یک‌بار ساخته می‌شود)
    MPC_N        = 20                     # افق شروع هر اپیزود
    DECISION_K   = 5                      # گام plant به‌ازای هر تصمیم RL (20 Hz)
    LP_ALPHA     = 0.2                    # فیلتر پایین‌گذر وزن‌ها (به‌ازای هر گام plant)
    CRASH_PEN    = 1000.0
    HORIZON_COST = 0.0                    # ضریب ∈ [0,∞): reward -= c*(N-Nmin)/(Nmax-Nmin) هر گام

    _RL_LOW  = np.array([10.,  1.], dtype=np.float64)    # Qphi, Qphidot
    _RL_HIGH = np.array([500., 50.], dtype=np.float64)

    def __init__(self, crash_pen=None, decision_k=None, horizon_cost=None,
                 lp_alpha=None, horizons=None):
        if horizons is not None:
            self.HORIZONS = tuple(horizons)
            if self.MPC_N not in self.HORIZONS:
                self.MPC_N = self.HORIZONS[len(self.HORIZONS) // 2]
        super().__init__()
        if crash_pen    is not None: self.CRASH_PEN    = float(crash_pen)
        if decision_k   is not None: self.DECISION_K   = int(decision_k)
        if horizon_cost is not None: self.HORIZON_COST = float(horizon_cost)
        if lp_alpha     is not None: self.LP_ALPHA     = float(lp_alpha)

        self._mid4 = self._denorm_action(np.zeros(4))   # [Qphi, Qphidot, Qjoint, R] میانه
        self._N_idx = self.HORIZONS.index(self.MPC_N)

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        hi = self.observation_space.high
        self.observation_space = gym.spaces.Box(
            np.concatenate([-hi, -np.ones(3, np.float32)]),
            np.concatenate([hi,   np.ones(3, np.float32)]),
            dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────
    # action <-> پارامترهای MPC
    # ─────────────────────────────────────────────────────────────────────
    @classmethod
    def decode_action(cls, a):
        """a ∈ [-1,1]^3 → (Qphi, Qphidot, N)."""
        a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
        t = (a[:2] + 1.0) / 2.0
        w = np.exp(np.log(cls._RL_LOW) + t * (np.log(cls._RL_HIGH) - np.log(cls._RL_LOW)))
        K = len(cls.HORIZONS)
        idx = min(int((a[2] + 1.0) / 2.0 * K), K - 1)
        return float(w[0]), float(w[1]), int(cls.HORIZONS[idx])

    def _extra_obs(self):
        lo, hi = np.log(self._RL_LOW), np.log(self._RL_HIGH)
        w = np.log(np.maximum(self._smoothed_weights[:2], 1e-9))
        wn = 2.0 * (w - lo) / (hi - lo) - 1.0
        nn = 2.0 * self._N_idx / max(len(self.HORIZONS) - 1, 1) - 1.0
        return np.clip(np.array([wn[0], wn[1], nn], dtype=np.float32), -1.0, 1.0)

    # ─────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # super وزن‌ها را به میانه برگردانده؛ افق را هم به شروع برگردان
        self.mpc = self._solvers[self.MPC_N]
        self.mpc.reset_warm_start()
        self._N_idx = self.HORIZONS.index(self.MPC_N)
        self._prev_rl = self._mid4[:2].copy()
        return np.concatenate([obs, self._extra_obs()]).astype(np.float32), info

    def step(self, rl_action):
        Qphi, Qphid, N_new = self.decode_action(rl_action)
        target = np.array([Qphi, Qphid, self._mid4[2], self._mid4[3]])

        # جریمه‌ی تغییر پارامتر (نرمال به بازه) — یک‌بار به‌ازای هر تصمیم
        rng = self._RL_HIGH - self._RL_LOW
        smooth_pen = -0.5 * float(np.sum(((target[:2] - self._prev_rl) / rng) ** 2))
        self._prev_rl = target[:2].copy()

        Nmin, Nmax = min(self.HORIZONS), max(self.HORIZONS)
        hz_pen = self.HORIZON_COST * (N_new - Nmin) / max(Nmax - Nmin, 1)

        total_r = smooth_pen
        terminated = truncated = False
        n_sub = 0
        u_norms, st_norms, solve_ms, track_t, dphi_t, ctrl_t = [], [], [], [], [], []
        err_sq = []
        statuses, sat_any, last_info_phi = {}, False, 0.0
        x_d = np.zeros(6)

        for _ in range(self.DECISION_K):
            self._steps += 1
            n_sub += 1

            # فیلتر وزن‌ها
            self._smoothed_weights = ((1.0 - self.LP_ALPHA) * self._smoothed_weights
                                      + self.LP_ALPHA * target)
            w = self._smoothed_weights
            Q_diag = np.array([w[0], w[1], w[2], w[2], w[2] * 0.1, w[2] * 0.1])
            R_diag = np.array([w[3], w[3]])

            # سوییچ افق (solver pool؛ u_prev منتقل، warm-start صفر)
            if N_new != self.mpc.N:
                u_prev_old = self.mpc.u_prev.value.copy()
                self.mpc = self._solvers[N_new]
                self.mpc.reset_warm_start()
                self.mpc.u_prev.value = u_prev_old
            self._N_idx = self.HORIZONS.index(N_new)

            phi_d = self._desired_roll()
            x_d[0] = phi_d
            A_d, B_d, bias_d = self.dyn.linearise(self.state[0])
            self.mpc.update_model(A_d, B_d, bias_d)
            P_chol = self._dare_chol(A_d, B_d, Q_diag=Q_diag, R_diag=R_diag)

            u_opt = self.mpc.solve(x0=self.state, xd=x_d, Q_diag=Q_diag,
                                   R_diag=R_diag, P_chol=P_chol)
            u_opt = np.clip(u_opt, -TAU_MAX, TAU_MAX)
            self.state = self.dyn.step(self.state, u_opt)

            self.v = float(np.clip(self.v + self.np_random.normal(0.0, self.V_NOISE), 1.0, 8.0))
            self.alpha = float(np.clip(
                self.alpha + self.np_random.normal(0.0, self.ALPHA_NOISE), -0.20, 0.20))

            phi   = self.state[0]
            phi_d = self._desired_roll()
            err   = phi - phi_d
            track = float(np.exp(-20.0 * err ** 2))
            dphi  = 0.05 * abs(self.state[1])
            ctrl  = 0.002 * float(np.sum(np.abs(u_opt)))
            r = float(np.clip(1.0 + track - dphi - ctrl - hz_pen, 0.0, 2.0))

            u_norms.append(float(np.linalg.norm(u_opt)))
            st_norms.append(float(np.linalg.norm(self.state)))
            solve_ms.append(self.mpc.last_solve_time * 1000.0)
            statuses[self.mpc.last_solve_status] = statuses.get(self.mpc.last_solve_status, 0) + 1
            sat_any |= bool(np.any(np.abs(u_opt) >= TAU_MAX * 0.99))
            track_t.append(track); dphi_t.append(dphi); ctrl_t.append(ctrl)
            err_sq.append(float(err) ** 2)
            last_info_phi = float(phi)

            if abs(phi) > np.pi / 4:
                r = -self.CRASH_PEN
                terminated = True
            total_r += r
            if terminated:
                break
            if self._steps >= self.MAX_STEPS:
                truncated = True
                break

        obs10 = np.array([
            self.state[0], self.state[1], self.state[2], self.state[3],
            self.state[4], self.state[5], self.v, self.alpha, phi_d, err,
        ], dtype=np.float32)
        obs = np.concatenate([obs10, self._extra_obs()]).astype(np.float32)

        total_r = float(total_r)
        info = {
            "phi": last_info_phi, "phi_d": float(phi_d), "err": float(err),
            "v": float(self.v), "alpha": float(self.alpha),
            "curriculum_level": int(self.curriculum_level),
            # ── diagnostics (میانگین/ماکس روی گام‌های plant این تصمیم) ───────
            "u_norm": float(np.mean(u_norms)), "state_norm": float(np.max(st_norms)),
            "u_sat": bool(sat_any),
            "solve_time_ms": float(np.mean(solve_ms)),
            "solve_time_ms_max": float(np.max(solve_ms)),
            "solve_status": max(statuses, key=statuses.get),
            "n_plant_steps": int(n_sub), "err_sq_sum": float(np.sum(err_sq)),
            "mpc_N": int(N_new), "Qphi": float(Qphi), "Qphidot": float(Qphid),
            "reward_raw": total_r, "reward_clipped": total_r,
            "mpc_weights_smoothed": self._smoothed_weights.tolist(),
            "reward_components": {
                "tracking":      float(np.mean(track_t)),
                "phi_dot":       -float(np.mean(dphi_t)),
                "control":       -float(np.mean(ctrl_t)),
                "action_smooth": float(smooth_pen),
                "horizon_cost":  -float(hz_pen),
                "raw_total":     total_r,
                "clipped":       total_r,
            },
        }
        return obs, total_r, terminated, truncated, info