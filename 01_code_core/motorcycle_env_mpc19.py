"""
motorcycle_env_mpc19.py
=======================
v19 = v18 + (۱) R دوباره در action، (۲) reward با جمله‌های tracking–effort–smoothness–compute.
زیرکلاس MotorcycleEnvMPC18؛ v17/v18 دست‌نخورده می‌مانند (مدل v18 و SIL آن معتبر می‌ماند).

Action چهاربعدی در [-1,1]:  a = [a_Qphi, a_Qphidot, a_R, a_N]
    Qphi    log-map [10, 500]      Qphidot log-map [1, 50]
    R       log-map [A_LOW[3], A_HIGH[3]] (همان بازه‌ی env v17)
    N       یکی از HORIZONS (گسسته)         Qjoint ثابت = mid
Observation ۱۴‌بعدی: ۱۰ عدد معمول + [Qphi, Qphidot, R, N] فعلی (نرمال [-1,1]).

Reward هر گام plant:
    r = clip( 1 + track - dphi - W_U·‖u‖² - W_DU·‖Δu‖² - W_N·(N/Nmax), 0, 2 )
    track = exp(-20·err²),   dphi = 0.05·|φ̇|
    سقوط: r = -CRASH_PEN (خارج از clip)
مجموع هر جمله روی گام‌های «زنده»ی هر تصمیم در info ثبت می‌شود
(sum_track, sum_dphi, sum_u2, sum_du2, sum_nfrac, n_alive, n_clip)؛ چون reward
نسبت به ضرایب خطی است (به‌جز clip)، بعداً می‌شود ضرایب را «آفلاین» روی CSV
اجراهای ثابت کالیبره کرد بدون شبیه‌سازی دوباره (reward_unit_v19.py).
پیش‌فرض ضرایب صفر است (⇒ شبیه reward v18 بدون جمله‌ی effort L1 ناچیز قبلی).
"""
import numpy as np
import gymnasium as gym

from motorcycle_env_mpc18 import MotorcycleEnvMPC18
from motorcycle_dynamics import TAU_MAX, MotorcycleDynamics


class MotorcycleEnvMPC19(MotorcycleEnvMPC18):

    W_U  = 0.0     # ضریب ‖u‖²   (u/TAU_MAX، بی‌بعد)
    W_DU = 0.0     # ضریب ‖Δu‖²  (Δu/(2·TAU_MAX)، بی‌بعد)
    W_N  = 0.0     # ضریب N/Nmax (هزینه‌ی محاسباتی، مستقل از سخت‌افزار)
    CLIP = True    # False ⇒ reward بدون clip (برای آنکه کالیبره‌ی آفلاین دقیق بماند)
    _R_LOW  = float(MotorcycleEnvMPC18._A_LOW[3])
    _R_HIGH = float(MotorcycleEnvMPC18._A_HIGH[3])

    def __init__(self, w_u=None, w_du=None, w_n=None, clip=None, fault=None, **kw):
        super().__init__(**kw)
        if w_u  is not None: self.W_U  = float(w_u)
        if w_du is not None: self.W_DU = float(w_du)
        if w_n  is not None: self.W_N  = float(w_n)
        if clip is not None: self.CLIP = bool(clip)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        hi = np.asarray(self.observation_space.high)[:10]
        self.observation_space = gym.spaces.Box(
            np.concatenate([-hi, -np.ones(4, np.float32)]),
            np.concatenate([hi,   np.ones(4, np.float32)]), dtype=np.float32)
        self._u_prev_act = np.zeros(2)
        self.set_fault(**(fault or {}))

    # ── fault روی plant (MPC و مدل داخلی آن دست‌نخورده‌اند ⇒ mismatch) ─────
    def set_fault(self, eff=(1.0, 1.0), iw_scale=1.0, rate_max=None):
        """
        eff      : (αf, αr) ضریب اثربخشی گشتاور اعمالی  τ_applied = α·τ_cmd
        iw_scale : ضریب IwΩ در plant (افت سرعت فلایویل)؛ مدل MPC نامی می‌ماند
        rate_max : سقف سرعت گیمبال [rad/s] در plant (nominal = MAX_DELTA_DOT=20)
        پیش‌فرض ⇒ سالم؛ رفتار قبلی عیناً حفظ می‌شود.
        """
        self._fault_eff = np.asarray(eff, dtype=float).reshape(2)
        self._fault_rate = None if rate_max is None else float(rate_max)
        self._plant_dyn = (self.dyn if float(iw_scale) == 1.0
                           else MotorcycleDynamics(Iw_scale=float(iw_scale)))
        self._fault_cfg = dict(eff=tuple(self._fault_eff), iw_scale=float(iw_scale),
                               rate_max=self._fault_rate)

    # ── action <-> پارامترهای MPC ───────────────────────────────────────
    @classmethod
    def decode_action(cls, a):
        """a ∈ [-1,1]^4 → (Qphi, Qphidot, R, N)."""
        a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
        lo = np.log(np.array([cls._RL_LOW[0], cls._RL_LOW[1], cls._R_LOW]))
        hi = np.log(np.array([cls._RL_HIGH[0], cls._RL_HIGH[1], cls._R_HIGH]))
        w = np.exp(lo + (a[:3] + 1.0) / 2.0 * (hi - lo))
        K = len(cls.HORIZONS)
        idx = min(int((a[3] + 1.0) / 2.0 * K), K - 1)
        return float(w[0]), float(w[1]), float(w[2]), int(cls.HORIZONS[idx])

    def _extra_obs(self):
        lo = np.log(np.array([self._RL_LOW[0], self._RL_LOW[1], self._R_LOW]))
        hi = np.log(np.array([self._RL_HIGH[0], self._RL_HIGH[1], self._R_HIGH]))
        w = np.log(np.maximum(self._smoothed_weights[[0, 1, 3]], 1e-9))
        wn = 2.0 * (w - lo) / (hi - lo) - 1.0
        nn = 2.0 * self._N_idx / max(len(self.HORIZONS) - 1, 1) - 1.0
        return np.clip(np.concatenate([wn, [nn]]).astype(np.float32), -1.0, 1.0)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)   # v18: obs10 + _extra_obs() (۴ تایی)
        self._u_prev_act = np.zeros(2)
        return obs, info

    # ── step ─────────────────────────────────────────────────────────────
    def step(self, rl_action):
        Qphi, Qphid, Rv, N_new = self.decode_action(rl_action)
        target = np.array([Qphi, Qphid, self._mid4[2], Rv])
        Nmax = max(self.HORIZONS)
        nfrac = N_new / Nmax

        total_r = 0.0
        terminated = truncated = False
        n_sub = n_alive = n_clip = 0
        s_track = s_dphi = s_u2 = s_du2 = s_nfrac = 0.0
        u_norms, st_norms, solve_ms, err_sq = [], [], [], []
        statuses, sat_any, last_phi = {}, False, 0.0
        x_d = np.zeros(6)

        for _ in range(self.DECISION_K):
            self._steps += 1
            n_sub += 1

            self._smoothed_weights = ((1.0 - self.LP_ALPHA) * self._smoothed_weights
                                      + self.LP_ALPHA * target)
            w = self._smoothed_weights
            Q_diag = np.array([w[0], w[1], w[2], w[2], w[2] * 0.1, w[2] * 0.1])
            R_diag = np.array([w[3], w[3]])

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
            self.state = self._plant_dyn.step(self.state, self._fault_eff * u_opt)
            if self._fault_rate is not None:
                self.state[4:6] = np.clip(self.state[4:6], -self._fault_rate, self._fault_rate)

            self.v = float(np.clip(self.v + self.np_random.normal(0.0, self.V_NOISE), 1.0, 8.0))
            self.alpha = float(np.clip(
                self.alpha + self.np_random.normal(0.0, self.ALPHA_NOISE), -0.20, 0.20))

            phi   = self.state[0]
            phi_d = self._desired_roll()
            err   = phi - phi_d

            du = u_opt - self._u_prev_act
            self._u_prev_act = u_opt.copy()
            track = float(np.exp(-20.0 * err ** 2))
            dphi  = 0.05 * abs(self.state[1])
            u2    = float(np.sum((u_opt / TAU_MAX) ** 2))            # بی‌بعد
            du2   = float(np.sum((du / (2.0 * TAU_MAX)) ** 2))   # بی‌بعد
            raw   = 1.0 + track - dphi - self.W_U * u2 - self.W_DU * du2 - self.W_N * nfrac
            r     = float(np.clip(raw, 0.0, 2.0)) if self.CLIP else float(raw)
            n_clip += int(raw < 0.0 or raw > 2.0)

            u_norms.append(float(np.linalg.norm(u_opt)))
            st_norms.append(float(np.linalg.norm(self.state)))
            solve_ms.append(self.mpc.last_solve_time * 1000.0)
            statuses[self.mpc.last_solve_status] = statuses.get(self.mpc.last_solve_status, 0) + 1
            sat_any |= bool(np.any(np.abs(u_opt) >= TAU_MAX * 0.99))
            err_sq.append(float(err) ** 2)
            last_phi = float(phi)

            if abs(phi) > np.pi / 4:
                r = -self.CRASH_PEN
                terminated = True
            else:                                   # فقط گام‌های زنده در جمع‌ها
                n_alive += 1
                s_track += track; s_dphi += dphi; s_u2 += u2; s_du2 += du2; s_nfrac += nfrac
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
            "phi": last_phi, "phi_d": float(phi_d), "err": float(err),
            "v": float(self.v), "alpha": float(self.alpha),
            "curriculum_level": int(self.curriculum_level), "fault": str(self._fault_cfg),
            "u_norm": float(np.mean(u_norms)), "state_norm": float(np.max(st_norms)),
            "u_sat": bool(sat_any),
            "solve_time_ms": float(np.mean(solve_ms)),
            "solve_time_ms_max": float(np.max(solve_ms)),
            "solve_status": max(statuses, key=statuses.get),
            "n_plant_steps": int(n_sub), "err_sq_sum": float(np.sum(err_sq)),
            "mpc_N": int(N_new), "Qphi": float(Qphi), "Qphidot": float(Qphid), "R": float(Rv),
            "reward_raw": total_r, "reward_clipped": total_r,
            # ── جمع جمله‌ها (برای کالیبره‌ی آفلاین ضرایب) ──────────────────
            "n_alive": int(n_alive), "n_clip": int(n_clip),
            "sum_track": s_track, "sum_dphi": s_dphi, "sum_u2": s_u2,
            "sum_du2": s_du2, "sum_nfrac": s_nfrac,
            "reward_components": {
                "tracking": s_track / max(n_alive, 1),
                "phi_dot": -s_dphi / max(n_alive, 1),
                "effort": -self.W_U * s_u2 / max(n_alive, 1),
                "smooth": -self.W_DU * s_du2 / max(n_alive, 1),
                "compute": -self.W_N * s_nfrac / max(n_alive, 1),
                "raw_total": total_r, "clipped": total_r,
            },
        }
        return obs, total_r, terminated, truncated, info
