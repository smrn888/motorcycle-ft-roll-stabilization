"""
baseline4_sac_cbf_train.py
==========================
آموزش و ارزیابی دو مدل برای مقایسه منصفانه با RL-TVL-MPC v17:

  مدل A : SAC (بدون CBF)     ← "SAC Plain"
  مدل B : SAC + CBF Filter    ← "SAC + CBF"

──────────────────────────────────────────────────────────────────
سه اصلاح اصلی نسبت به نسخه قبل:

[FIX-1]  HOCBF  (High-Order CBF)  به‌جای CBF درجه-۱ اشتباه
    ─────────────────────────────────────────────────────────────
    دینامیک موتورسیکلت:
        φ̈ = f_phi(x) + g_phi(x) · τ   ← کنترل مستقیم روی φ̈ وارد می‌شود

    h(x) = (π/4)² − φ²   →   relative degree = 2
    (چون h فقط تابع φ است و τ در φ̈ ظاهر می‌شود، نه در φ̇)

    برای relative degree 2، باید زنجیره CBF بسازیم:
        ψ₀ = h  = (π/4)² − φ²
        ψ₁ = ψ̇₀ + α₁·ψ₀  =  −2φ·φ̇ + α₁·((π/4)² − φ²)

    شرط HOCBF روی ψ₁:
        ψ̇₁ + α₂·ψ₁ ≥ 0

    و ψ̇₁ شامل τ (کنترل) می‌شود — پس QP می‌تواند حل کند.

    محاسبه ψ̇₁:
        ψ̇₁ = d/dt[−2φφ̇ + α₁ψ₀]
             = −2φ̇² − 2φ·φ̈   +  α₁·(−2φφ̇)
             = −2φ̇² − 2φ·[f_phi(x) + g_phi(x)·τ]  −  2α₁φφ̇

    بنابراین:
        Lf_ψ₁ = −2φ̇² − 2φ·f_phi(x) − 2α₁·φ·φ̇       ← بدون τ
        Lg_ψ₁ = −2φ · g_phi(x)                         ← ضریب τ

    QP:
        min  ‖τ − τ_rl‖²
        s.t. Lf_ψ₁ + Lg_ψ₁·τ + α₂·ψ₁ ≥ 0
             ‖τ‖_∞ ≤ TAU_MAX

[FIX-2]  norm_obs=True, norm_reward=False
    ─────────────────────────────────────────────────────────────
    observation شامل مقیاس‌های بسیار متفاوت است:
        phi       ~ 0.5 rad
        phi_dot   ~ 10 rad/s
        v         ~ 8 m/s
        err       ~ 0.5 rad
    VecNormalize با norm_obs=True این‌ها را نرمال می‌کند → یادگیری SAC بهتر.
    norm_reward=False: reward خام حفظ می‌شود تا مقایسه با RL-MPC منصفانه باشد.

[FIX-3]  متریک‌های انرژی کنترل
    ─────────────────────────────────────────────────────────────
    در evaluate اضافه شد:
        mean_control_energy  = mean(Σ‖τ‖²  per episode)
        mean_tau_abs_mean    = mean(mean(‖τ‖₁) per episode)
        mean_tau_rms         = mean(√mean(τ²) per episode)
    این‌ها نشان می‌دهند SAC چقدر torque مصرف می‌کند.

[FIX-4]  Soft CBF — Slack Variable (always feasible)
    ─────────────────────────────────────────────────────────────
    QP اصلی ممکن است infeasible شود وقتی φ ≈ 45°.
    با اضافه کردن slack variable s ≥ 0:

        min  ‖τ − τ_rl‖² + λ·s²
        s.t. Lf_ψ₁ + Lg_ψ₁·τ + α₂·ψ₁ + s ≥ 0
             s ≥ 0
             ‖τ‖_∞ ≤ TAU_MAX

    QP همیشه feasible است. s* ≈ 0 در حالت عادی،
    s* > 0 فقط وقتی قید اصلی غیرقابل‌ارضاست.
    λ (--lambda_slack): بزرگ‌تر → قید سخت‌تر. پیشنهاد: 1e3 ~ 1e5.

──────────────────────────────────────────────────────────────────
Observation (10-بُعدی — همان RL-TVL-MPC v17):
    [φ, φ̇, δ_f, δ_r, δ̇_f, δ̇_r, v, α, φ_d, err]

Action (2-بُعدی — نرمالایز):
    a ∈ [-1,1]²  →  τ = a × TAU_MAX  [N·m]

Reward (همان mpc17):
    −10·err² + 5·exp(−50·err²) − 0.5·|φ̇| − 0.01·Σ|τ|
    − 40·(|φ|/(π/4))⁶ + 1.5

──────────────────────────────────────────────────────────────────
نحوه اجرا:

  # آموزش هر دو مدل (3M step):
  python baseline4_sac_cbf_train.py --mode train --steps 3000000

  # روی GPU:
  python baseline4_sac_cbf_train.py --mode train --steps 3000000 --device cuda

  # فقط SAC+CBF:
  python baseline4_sac_cbf_train.py --mode train --model sac_cbf --steps 3000000

  # ارزیابی بعد از آموزش:
  python baseline4_sac_cbf_train.py --mode eval --curriculum 8 --episodes 200

  # با slack نرم‌تر (feasibility بیشتر):
  python baseline4_sac_cbf_train.py --mode train --model sac_cbf --lambda_slack 1e3

  # با slack سخت‌تر (قید CBF قوی‌تر):
  python baseline4_sac_cbf_train.py --mode train --model sac_cbf --lambda_slack 1e6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from motorcycle_dynamics import (
    MotorcycleDynamics, TAU_MAX, DELTA_MAX, MAX_DELTA_DOT, Ts
)

# ═══════════════════════════════════════════════════════════════════════════
# ثابت‌های مشترک
# ═══════════════════════════════════════════════════════════════════════════

MODEL_PLAIN_PATH = "sac_plain_motorcycle"
MODEL_CBF_PATH   = "sac_cbf_motorcycle"

CURRICULUM_LEVELS = [
    (0.02, 0.05),
    (0.04, 0.10),
    (0.06, 0.15),
    (0.08, 0.20),
    (0.10, 0.25),
    (0.12, 0.30),
    (0.35, 0.80),
    (0.61, 1.20),
    (0.76, 1.50),
]

CRASH_ANGLE      = np.pi / 4
MAX_STEPS        = 500
V_NOISE          = 0.05
ALPHA_NOISE      = 0.005
REWARD_CLIP_LOW  = -50.0
REWARD_CLIP_HIGH =  20.0


# ═══════════════════════════════════════════════════════════════════════════
# [FIX-1]  High-Order CBF  (HOCBF)  — relative degree 2
# ═══════════════════════════════════════════════════════════════════════════

class HOCBFSafetyFilter:
    """
    High-Order CBF برای سیستم relative-degree 2.

    دینامیک roll:
        φ̈ = f_phi(φ, φ̇, δ̇_f, δ̇_r)  +  g_phi(φ) · τ

    که در آن:
        f_phi = (mgh·sin(φ) − IwΩ·(δ̇_f − δ̇_r)·cos(φ)) / I_phi
        g_phi = [∂φ̈/∂τ_f, ∂φ̈/∂τ_r]
              ≈ [−IwΩ·cos(φ)·Ts/(I_phi·I_d),
                 +IwΩ·cos(φ)·Ts/(I_phi·I_d)]
        (Ts/I_d: چون τ → δ̈ → δ̇_next، یه delay یه step داریم)

    Barrier chain:
        ψ₀ = (π/4)² − φ²
        ψ₁ = ψ̇₀ + α₁·ψ₀  =  −2φ·φ̇ + α₁·((π/4)² − φ²)

    شرط HOCBF (روی ψ₁):
        ψ̇₁ + α₂·ψ₁ ≥ 0

    ψ̇₁ = −2φ̇² − 2φ·φ̈ − 2α₁·φ·φ̇
         = [−2φ̇² − 2φ·f_phi − 2α₁·φ·φ̇]  +  [−2φ · g_phi] · τ
           ↑ Lf_ψ₁                           ↑ Lg_ψ₁

    QP:
        min  ‖τ − τ_rl‖²
        s.t. Lf_ψ₁ + Lg_ψ₁·τ + α₂·ψ₁ ≥ 0
             ‖τ‖_∞ ≤ TAU_MAX
    """

    def __init__(self, dyn: MotorcycleDynamics,
                 alpha1: float = 2.0,
                 alpha2: float = 2.0,
                 lambda_slack: float = 1e4):
        """
        alpha1       : class-K gain برای ψ₁ = ψ̇₀ + α₁·ψ₀
        alpha2       : class-K gain برای شرط  ψ̇₁ + α₂·ψ₁ ≥ 0
        lambda_slack : وزن جریمه slack variable در تابع هزینه
                       (بزرگ‌تر → slack کمتر → قید سخت‌تر، ولی احتمال infeasible بیشتر)
                       مقدار پیشنهادی: 1e3 ~ 1e5

        مقادیر بزرگ‌تر alpha → قید سخت‌تر، محافظ‌تر، ولی ممکن است torque بیشتر
        """
        import cvxpy as _cp
        self.cp          = _cp
        self.dyn         = dyn
        self.alpha1      = alpha1
        self.alpha2      = alpha2
        self.lambda_slack = lambda_slack

        self.mgh  = dyn.mgh
        self.IwO  = dyn.IwO
        self.Iphi = dyn.Iphi
        self.Id   = dyn.Id

        # ── QP با Slack Variable (always feasible) ──────────────────────
        # min  ‖τ − τ_rl‖² + λ·s²
        # s.t. Lf_ψ₁ + Lg_ψ₁·τ + α₂·ψ₁ + s ≥ 0
        #      s ≥ 0
        #      ‖τ‖_∞ ≤ TAU_MAX
        #
        # وقتی QP feasible باشد → s ≈ 0  (قید CBF به‌طور عملی برقرار)
        # وقتی infeasible باشد  → s > 0  (slack جبران می‌کند، سقوط نمی‌دهد)
        self._tau    = _cp.Variable(2)
        self._slack  = _cp.Variable(1, nonneg=True)   # s ≥ 0
        self._tau_rl = _cp.Parameter(2)
        self._Lf_p   = _cp.Parameter(1)   # Lf_ψ₁
        self._Lg_p   = _cp.Parameter(2)   # Lg_ψ₁
        self._psi1_p = _cp.Parameter(1)   # ψ₁ فعلی
        self._lam_p  = _cp.Parameter(1, nonneg=True)  # λ (قابل تغییر در runtime)

        cost = (
            _cp.sum_squares(self._tau - self._tau_rl)
            + self._lam_p[0] * _cp.sum_squares(self._slack)
        )
        hocbf_c = (
            self._Lf_p[0]
            + self._Lg_p @ self._tau
            + self.alpha2 * self._psi1_p[0]
            + self._slack[0]
            >= 0
        )
        self._qp = _cp.Problem(
            _cp.Minimize(cost),
            [hocbf_c, _cp.norm_inf(self._tau) <= TAU_MAX]
            # قید s ≥ 0 از طریق nonneg=True در Variable تعریف شده
        )

        self.n_filtered   = 0
        self.n_total      = 0
        self.n_slack_used = 0   # تعداد دفعاتی که s > threshold (قید relaxed شد)

    # ─────────────────────────────────────────────────────────────────────
    def _compute_lie(self, x: np.ndarray):
        """
        محاسبه Lf_ψ₁، Lg_ψ₁ و ψ₁.

        x = [φ, φ̇, δ_f, δ_r, δ̇_f, δ̇_r]
        """
        phi, phi_dot = x[0], x[1]
        ddot_f, ddot_r = x[4], x[5]

        # ψ₀ و ψ₁
        psi0 = CRASH_ANGLE ** 2 - phi ** 2
        psi1 = -2.0 * phi * phi_dot + self.alpha1 * psi0

        # f_phi(x): drift term roll acceleration (بدون کنترل)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        f_phi = (
            self.mgh * sin_phi
            - self.IwO * (ddot_f - ddot_r) * cos_phi
        ) / self.Iphi

        # g_phi: ∂φ̈/∂τ
        # τ → δ̈ → δ̇_next (یه step تأخیر)
        # ∂δ̇_f_next/∂τ_f ≈ Ts / I_d   (از دینامیک گیمبال)
        # φ̈ = (... − IwΩ·(δ̇_f−δ̇_r)·cos(φ)) / I_phi
        # ∂φ̈/∂τ_f ≈ −IwΩ·cos(φ)·(Ts/I_d) / I_phi
        # ∂φ̈/∂τ_r ≈ +IwΩ·cos(φ)·(Ts/I_d) / I_phi
        k = self.IwO * cos_phi * Ts / (self.Iphi * self.Id)
        g_phi = np.array([-k, +k])   # [∂φ̈/∂τ_f, ∂φ̈/∂τ_r]

        # ψ̇₁ = Lf_ψ₁ + Lg_ψ₁ · τ
        # Lf_ψ₁ = −2φ̇² − 2φ·f_phi − 2α₁·φ·φ̇
        Lf_psi1 = (
            -2.0 * phi_dot ** 2
            - 2.0 * phi * f_phi
            - 2.0 * self.alpha1 * phi * phi_dot
        )
        # Lg_ψ₁ = −2φ · g_phi
        Lg_psi1 = -2.0 * phi * g_phi

        return Lf_psi1, Lg_psi1, psi1, psi0

    # ─────────────────────────────────────────────────────────────────────
    def filter(self, x: np.ndarray, tau_rl: np.ndarray) -> np.ndarray:
        """
        اگر τ_rl شرط HOCBF را برآورده کند → بدون تغییر.
        وگرنه → QP با slack variable (همیشه feasible).

        Slack variable تضمین می‌کند که QP هرگز infeasible نمی‌شود.
        اگر s > 0 در جواب → قید CBF با جریمه λ·s² relaxed شده.
        """
        self.n_total += 1

        Lf_psi1, Lg_psi1, psi1, psi0 = self._compute_lie(x)

        # بررسی ایمنی
        cbf_val = Lf_psi1 + Lg_psi1 @ tau_rl + self.alpha2 * psi1
        if cbf_val >= 0 and psi1 >= 0:
            return np.clip(tau_rl, -TAU_MAX, TAU_MAX)

        # QP با slack
        self._tau_rl.value  = tau_rl
        self._Lf_p.value    = np.array([Lf_psi1])
        self._Lg_p.value    = Lg_psi1
        self._psi1_p.value  = np.array([psi1])
        self._lam_p.value   = np.array([self.lambda_slack])

        try:
            self._qp.solve(
                solver=self.cp.OSQP,
                warm_start=True,
                eps_abs=1e-5, eps_rel=1e-5,
                verbose=False,
            )
            if (self._qp.status in (self.cp.OPTIMAL, self.cp.OPTIMAL_INACCURATE)
                    and self._tau.value is not None):
                self.n_filtered += 1
                # ردیابی: آیا slack فعال شد؟
                s_val = float(self._slack.value[0]) if self._slack.value is not None else 0.0
                if s_val > 1e-3:
                    self.n_slack_used += 1
                return np.clip(self._tau.value, -TAU_MAX, TAU_MAX)
        except Exception:
            pass

        # fallback اضطراری: با slack QP نباید به اینجا برسد،
        # ولی برای هر احتمالی tau_rl clip‌شده برمی‌گردانیم.
        self.n_filtered += 1
        return np.clip(tau_rl, -TAU_MAX, TAU_MAX)

    def reset_stats(self):
        self.n_filtered   = 0
        self.n_total      = 0
        self.n_slack_used = 0

    @property
    def filter_rate(self) -> float:
        return self.n_filtered / self.n_total if self.n_total > 0 else 0.0

    @property
    def slack_rate(self) -> float:
        """نسبت دفعاتی که slack فعال شد (قید CBF relaxed شد)."""
        return self.n_slack_used / self.n_total if self.n_total > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# محیط پایه
# ═══════════════════════════════════════════════════════════════════════════

class MotorcycleSACEnv(gym.Env):
    """
    محیط SAC (با یا بدون HOCBF).

    Obs  (10): [φ, φ̇, δ_f, δ_r, δ̇_f, δ̇_r, v, α, φ_d, err]
    Act   (2): τ_norm ∈ [-1,1]²  →  τ = a·TAU_MAX  [N·m]
    Reward   : همان mpc17 (raw، بدون normalization)
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 use_cbf:      bool  = False,
                 cbf_alpha1:   float = 2.0,
                 cbf_alpha2:   float = 2.0,
                 lambda_slack: float = 1e4):
        super().__init__()

        self.use_cbf = use_cbf
        self.dyn     = MotorcycleDynamics()

        if use_cbf:
            self.hocbf = HOCBFSafetyFilter(
                self.dyn,
                alpha1=cbf_alpha1,
                alpha2=cbf_alpha2,
                lambda_slack=lambda_slack,
            )
        else:
            self.hocbf = None

        # ── Observation Space (همان mpc17) ──────────────────────────────
        obs_high = np.array([
            np.pi / 2, 10.0,
            np.pi / 2, np.pi / 2,
            20.0, 20.0,
            20.0, np.pi / 3,
            np.pi / 2, np.pi,
        ], dtype=np.float32)
        self.observation_space = spaces.Box(
            -obs_high, obs_high, dtype=np.float32)

        # ── Action Space: normalized torque ─────────────────────────────
        self.action_space = spaces.Box(
            np.full(2, -1.0, np.float32),
            np.full(2,  1.0, np.float32),
            dtype=np.float32,
        )

        self.state            = np.zeros(6)
        self.v                = 0.0
        self.alpha            = 0.0
        self._steps           = 0
        self.curriculum_level = 0

    # ─────────────────────────────────────────────────────────────────────
    def _desired_roll(self) -> float:
        p = self.dyn.p
        if abs(self.alpha) < 1e-4 or abs(self.v) < 1e-3:
            return 0.0
        return float(np.arctan(self.v ** 2 * self.alpha / (p['L'] * p['g'])))

    def _danger_penalty(self, phi: float) -> float:
        return -40.0 * (abs(phi) / CRASH_ANGLE) ** 6

    def set_curriculum_level(self, level: int):
        self.curriculum_level = int(np.clip(level, 0, len(CURRICULUM_LEVELS) - 1))

    # ─────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        self.state  = np.zeros(6)

        if self.hocbf is not None:
            self.hocbf.reset_stats()

        phi_r, phidot_r = CURRICULUM_LEVELS[self.curriculum_level]
        self.state[0] = self.np_random.uniform(-phi_r, phi_r)
        self.state[1] = self.np_random.uniform(-phidot_r, phidot_r)
        self.v     = float(self.np_random.uniform(1.0, 8.0))
        self.alpha = float(self.np_random.uniform(-0.20, 0.20))

        phi_d = self._desired_roll()
        err   = self.state[0] - phi_d
        return self._obs(phi_d, err), {}

    def _obs(self, phi_d: float, err: float) -> np.ndarray:
        return np.array([
            self.state[0], self.state[1],
            self.state[2], self.state[3],
            self.state[4], self.state[5],
            self.v, self.alpha, phi_d, err,
        ], dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────
    def step(self, rl_action: np.ndarray):
        self._steps += 1

        # denormalize
        tau_rl = np.clip(rl_action, -1.0, 1.0).astype(np.float64) * TAU_MAX

        # HOCBF filter
        if self.hocbf is not None:
            tau_applied    = self.hocbf.filter(self.state, tau_rl)
            cbf_intervened = bool(np.linalg.norm(tau_applied - tau_rl) > 1e-3)
        else:
            tau_applied    = tau_rl
            cbf_intervened = False

        # Plant RK4
        self.state = self.dyn.step(self.state, tau_applied)

        # محیط noise
        self.v = float(np.clip(
            self.v + self.np_random.normal(0.0, V_NOISE), 1.0, 8.0))
        self.alpha = float(np.clip(
            self.alpha + self.np_random.normal(0.0, ALPHA_NOISE),
            -0.20, 0.20))

        # Reward — همان mpc17، raw (بدون normalization)
        phi   = self.state[0]
        phi_d = self._desired_roll()
        err   = phi - phi_d

        tracking_reward = -10.0 * err ** 2
        stability_bonus =   5.0 * float(np.exp(-50.0 * err ** 2))
        phi_dot_penalty =  -0.5 * abs(self.state[1])
        control_cost    = -0.01 * np.sum(np.abs(tau_applied))
        danger_penalty  = self._danger_penalty(phi)
        survival_bonus  =   1.5

        # CBF penalty (فقط در مدل CBF) — تشویق RL برای کاهش نیاز به filter
        cbf_penalty = 0.0
        if self.hocbf is not None:
            cbf_penalty = -0.1 * float(np.linalg.norm(tau_applied - tau_rl))

        raw_reward = (
            tracking_reward + stability_bonus + phi_dot_penalty
            + control_cost + danger_penalty + survival_bonus
            + cbf_penalty
        )
        reward = float(np.clip(raw_reward, REWARD_CLIP_LOW, REWARD_CLIP_HIGH))

        terminated = bool(abs(phi) > CRASH_ANGLE)
        truncated  = self._steps >= MAX_STEPS

        info = {
            "phi":            float(phi),
            "phi_d":          float(phi_d),
            "err":            float(err),
            "tau_applied":    tau_applied.tolist(),
            "tau_rl":         tau_rl.tolist(),
            "cbf_intervened": cbf_intervened,
            "cbf_filter_rate": self.hocbf.filter_rate if self.hocbf else 0.0,
            "cbf_slack_rate":  self.hocbf.slack_rate  if self.hocbf else 0.0,
            # [FIX-3] متریک‌های انرژی — برای هر step ذخیره می‌شود
            "tau_sq_sum":     float(np.sum(tau_applied ** 2)),
            "tau_abs_sum":    float(np.sum(np.abs(tau_applied))),
            "tau_rms":        float(np.sqrt(np.mean(tau_applied ** 2))),
            "reward_raw":     float(raw_reward),
            "curriculum_level": int(self.curriculum_level),
        }
        return self._obs(phi_d, err), reward, terminated, truncated, info


# ═══════════════════════════════════════════════════════════════════════════
# Curriculum Wrapper
# ═══════════════════════════════════════════════════════════════════════════

class CurriculumWrapper(gym.Wrapper):
    """
    Auto-advance curriculum:
    اگر در WINDOW اپیزود اخیر، survival ≥ THRESHOLD → level + 1.
    """
    WINDOW    = 100
    THRESHOLD = 0.92
    MAX_LEVEL = len(CURRICULUM_LEVELS) - 1

    def __init__(self, env: MotorcycleSACEnv):
        super().__init__(env)
        self._history: list[int] = []
        self._level = 0
        env.set_curriculum_level(0)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            self._history.append(int(truncated))
            if len(self._history) > self.WINDOW:
                self._history.pop(0)

            if (len(self._history) == self.WINDOW
                    and np.mean(self._history) >= self.THRESHOLD
                    and self._level < self.MAX_LEVEL):
                self._level += 1
                self.env.set_curriculum_level(self._level)
                phi_deg = np.rad2deg(CURRICULUM_LEVELS[self._level][0])
                print(f"\n  ✦ Curriculum → level {self._level} "
                      f"(φ_max={phi_deg:.1f}°)")

        info["curriculum_level"] = self._level
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# آموزش
# ═══════════════════════════════════════════════════════════════════════════

def train(
    use_cbf:      bool  = False,
    total_steps:  int   = 3_000_000,
    n_envs:       int   = 8,
    seed:         int   = 0,
    device:       str   = "cpu",
    cbf_alpha1:   float = 2.0,
    cbf_alpha2:   float = 2.0,
    lambda_slack: float = 1e4,
    load_model:   str   = None,
) -> str:
    """آموزش SAC Plain یا SAC+HOCBF."""
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import (
            SubprocVecEnv, VecNormalize, DummyVecEnv)
        from stable_baselines3.common.callbacks import (
            EvalCallback, CheckpointCallback)
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        raise ImportError(
            "stable_baselines3 نصب نشده.\n"
            "  pip install stable-baselines3[extra]"
        )

    import torch
    if device == "cuda" and not torch.cuda.is_available():
        print("  ⚠  CUDA not available → CPU")
        device = "cpu"

    model_name = MODEL_CBF_PATH if use_cbf else MODEL_PLAIN_PATH
    label      = "SAC + HOCBF" if use_cbf else "SAC Plain"

    print("\n" + "═" * 65)
    print(f"  آموزش {label}")
    print(f"  steps={total_steps:,}  envs={n_envs}  "
          f"device={device}  seed={seed}")
    if use_cbf:
        print(f"  cbf_alpha1={cbf_alpha1}  cbf_alpha2={cbf_alpha2}  lambda_slack={lambda_slack:.0e}")
    print("═" * 65 + "\n")

    # ── ساخت محیط‌های موازی ───────────────────────────────────────────────
    def _make_env(rank: int):
        def _init():
            env = MotorcycleSACEnv(
                use_cbf=use_cbf,
                cbf_alpha1=cbf_alpha1,
                cbf_alpha2=cbf_alpha2,
                lambda_slack=lambda_slack,
            )
            env = CurriculumWrapper(env)
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env
        return _init

    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    raw_env = make_vec_env(
        _make_env(0),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=vec_cls,
    )

    # [FIX-4] norm_obs=True — مقیاس‌های observation بسیار متفاوت‌اند:
    # phi ~ 0.5 rad، phi_dot ~ 10 rad/s، v ~ 8 m/s، err ~ 0.5 rad
    # VecNormalize با norm_obs=True این‌ها را همگن می‌کند و یادگیری SAC را
    # بهبود می‌دهد. norm_reward=False حفظ می‌شود تا مقایسه با RL-MPC منصفانه باشد.
    env = VecNormalize(
        raw_env,
        norm_obs=True,
        norm_reward=False,   # ← raw reward، نه normalized
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    # eval env: level=4 (میانی)، بدون curriculum
    eval_raw = DummyVecEnv([lambda: Monitor(
        MotorcycleSACEnv(use_cbf=use_cbf,
                         cbf_alpha1=cbf_alpha1,
                         cbf_alpha2=cbf_alpha2,
                         lambda_slack=lambda_slack)
    )])
    eval_raw.envs[0].set_curriculum_level(4)
    eval_env = VecNormalize(
        eval_raw,
        norm_obs=True, norm_reward=False,
        clip_obs=10.0, clip_reward=10.0, gamma=0.99, training=False,
    )

    # ── SAC hyperparameters ───────────────────────────────────────────────
    if load_model and os.path.exists(load_model + ".zip"):
        print(f"  بارگذاری مدل: {load_model}.zip")
        model = SAC.load(load_model, env=env, device=device)
    else:
        model = SAC(
            policy          = "MlpPolicy",
            env             = env,
            learning_rate   = 3e-4,
            buffer_size     = 500_000,
            batch_size      = 256,
            gamma           = 0.99,
            tau             = 0.005,
            ent_coef        = "auto",
            target_entropy  = -1.0,
            train_freq      = (1, "step"),
            gradient_steps  = 1,
            learning_starts = 10_000,
            seed            = seed,
            policy_kwargs   = dict(
                net_arch        = [256, 256, 128],
                log_std_init    = -1.0,
                optimizer_kwargs= dict(eps=1e-5),
            ),
            device = device,
            verbose= 1,
        )

    # ── Callbacks ─────────────────────────────────────────────────────────
    best_dir = f"./best_{model_name}/"
    ckpt_dir = f"./ckpts_{model_name}/"
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=best_dir,
            log_path=best_dir,
            eval_freq=max(10_000 // n_envs, 1000),
            n_eval_episodes=20,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(100_000 // n_envs, 1),
            save_path=ckpt_dir,
            name_prefix=model_name,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────
    t0 = time.time()
    model.learn(
        total_timesteps    = total_steps,
        callback           = callbacks,
        progress_bar       = True,
        reset_num_timesteps= True,
    )
    elapsed = time.time() - t0
    print(f"\n  آموزش تمام: {elapsed / 3600:.2f}h  ({elapsed:.0f}s)")

    model.save(model_name)
    env.save(f"vecnorm_{model_name}.pkl")
    print(f"  مدل → {model_name}.zip")

    env.close()
    eval_env.close()
    return model_name


# ═══════════════════════════════════════════════════════════════════════════
# ارزیابی
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(
    use_cbf:          bool  = False,
    n_episodes:       int   = 200,
    curriculum_level: int   = 8,
    seed:             int   = 42,
    model_path:       str   = None,
    cbf_alpha1:       float = 2.0,
    cbf_alpha2:       float = 2.0,
    lambda_slack:     float = 1e4,
) -> dict:
    """ارزیابی SAC Plain یا SAC+HOCBF."""
    try:
        from stable_baselines3 import SAC as _SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        _SAC = None

    label = "SAC + HOCBF" if use_cbf else "SAC Plain"

    if model_path is None:
        model_path = MODEL_CBF_PATH if use_cbf else MODEL_PLAIN_PATH

    clean = model_path.replace(".zip", "")

    # ── ساخت محیط ارزیابی ──────────────────────────────────────────────────
    # باید همان pipeline ای که در train بود داشته باشیم:
    # MotorcycleSACEnv → Monitor → DummyVecEnv → VecNormalize
    # تا observation normalization دقیقاً مثل train باشد.
    raw_env = MotorcycleSACEnv(
        use_cbf=use_cbf, cbf_alpha1=cbf_alpha1, cbf_alpha2=cbf_alpha2,
        lambda_slack=lambda_slack)
    raw_env.set_curriculum_level(curriculum_level)

    vec_raw = DummyVecEnv([lambda: Monitor(raw_env)])

    vecnorm_path = f"vecnorm_{clean}.pkl"
    vecnorm_loaded = False
    if _SAC is not None and os.path.exists(vecnorm_path):
        # آمار میانگین/واریانس که در train یاد گرفته را load می‌کنیم.
        # training=False: آمار در eval به‌روزرسانی نمی‌شود.
        env = VecNormalize.load(vecnorm_path, vec_raw)
        env.training = False
        env.norm_reward = False
        vecnorm_loaded = True
        print(f"  VecNormalize stats → {vecnorm_path}")
    else:
        # اگر فایل آمار پیدا نشد، بدون normalization ادامه می‌دهیم
        # و هشدار می‌دهیم.
        env = VecNormalize(
            vec_raw,
            norm_obs=False, norm_reward=False,
            training=False,
        )
        if _SAC is not None:
            print(f"  ⚠  vecnorm_{clean}.pkl یافت نشد → obs خام (نتایج ممکن است نادرست باشد)")

    model = None
    if _SAC is not None:
        if os.path.exists(clean + ".zip"):
            print(f"  بارگذاری: {clean}.zip")
            model = _SAC.load(clean, env=env)
        else:
            print(f"  ⚠  مدل {model_path} یافت نشد → random policy")

    # دسترسی مستقیم به محیط داخلی برای خواندن info
    # (DummyVecEnv.envs[0] همان Monitor است؛ .env آن MotorcycleSACEnv)
    inner_env: MotorcycleSACEnv = vec_raw.envs[0].env

    # ── بافرهای آمار ──────────────────────────────────────────────────────
    ep_rewards    = []
    ep_lengths    = []
    survived      = []
    max_phi_abs   = []
    rmse_list     = []
    cbf_rates     = []
    cbf_int_count = []

    # [FIX-3] بافرهای انرژی کنترل
    ep_control_energy = []   # Σ‖τ‖² per episode
    ep_tau_abs_mean   = []   # mean(‖τ‖₁) per episode
    ep_tau_rms        = []   # √mean(τ²) per episode

    print(f"\n[{label}]  curriculum={curriculum_level}  "
          f"episodes={n_episodes}  seed={seed}")
    print(f"  vecnorm={'loaded ✓' if vecnorm_loaded else '⚠ نه load شد'}"
          f"  model={'loaded ✓' if model else '⚠ random'}")
    print()

    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        # VecEnv.reset → obs shape: (n_envs, obs_dim)
        obs = env.reset()

        # reset seed از طریق inner env (VecEnv seed را مستقیم پشتیبانی نمی‌کند)
        inner_env.np_random = np.random.default_rng(int(seed + ep))
        obs = env.reset()   # دوباره با seed جدید

        if inner_env.hocbf:
            inner_env.hocbf.reset_stats()

        ep_r        = 0.0
        ep_err_sq   = []
        ep_len      = 0
        max_ph      = 0.0
        n_intervene = 0
        step_tau_sq  = []
        step_tau_abs = []
        step_tau_rms = []

        while True:
            if model is not None:
                # obs shape: (1, obs_dim) → predict مستقیم می‌گیرد
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = rng.uniform(-1.0, 1.0, size=(1, 2)).astype(np.float32)

            # VecEnv.step → obs, reward, done, info (همه shape (n_envs,))
            obs, r_arr, done_arr, info_arr = env.step(action)
            r    = float(r_arr[0])
            done = bool(done_arr[0])
            info = info_arr[0]   # dict از inner env

            ep_r   += r
            ep_len += 1
            ep_err_sq.append(info["err"] ** 2)
            max_ph = max(max_ph, abs(info["phi"]))
            if info["cbf_intervened"]:
                n_intervene += 1

            step_tau_sq.append(info["tau_sq_sum"])
            step_tau_abs.append(info["tau_abs_sum"])
            step_tau_rms.append(info["tau_rms"])

            # VecEnv به‌طور خودکار reset می‌کند وقتی done=True؛
            # ما فقط آمار را ثبت و break می‌کنیم.
            if done:
                # trunc = truncated (survived): از info استخراج می‌کنیم
                # SB3 VecEnv: "TimeLimit.truncated" یا بررسی ep_len
                trunc = info.get("TimeLimit.truncated", ep_len >= MAX_STEPS)
                ep_rewards.append(ep_r)
                ep_lengths.append(ep_len)
                survived.append(int(trunc))
                max_phi_abs.append(max_ph)
                rmse_list.append(float(np.sqrt(np.mean(ep_err_sq))))
                cbf_rates.append(
                    inner_env.hocbf.filter_rate if inner_env.hocbf else 0.0)
                cbf_int_count.append(n_intervene)
                ep_control_energy.append(float(np.sum(step_tau_sq)))
                ep_tau_abs_mean.append(float(np.mean(step_tau_abs)))
                ep_tau_rms.append(float(np.mean(step_tau_rms)))
                break

        if (ep + 1) % 50 == 0:
            sr = np.mean(survived[-50:])
            rw = np.mean(ep_rewards[-50:])
            rm = np.mean(rmse_list[-50:])
            en = np.mean(ep_control_energy[-50:])
            extra = (f"  cbf={np.mean(cbf_rates[-50:]):.1%}"
                     if use_cbf else "")
            print(f"  ep {ep+1:4d}/{n_episodes}  "
                  f"survival={sr:.1%}  reward={rw:.1f}  "
                  f"rmse={rm:.4f}  energy={en:.1f}{extra}")

    method_name = ("SAC + HOCBF (Baseline 4)"
                   if use_cbf else "SAC Plain (Baseline 4b)")

    results: dict = {
        "method":            method_name,
        "use_cbf":           use_cbf,
        "curriculum":        curriculum_level,
        "n_episodes":        n_episodes,
        "seed":              seed,
        "model_path":        model_path or "random",
        # ── کیفیت کنترل ──────────────────────────────────────────────────
        "survival_rate":     float(np.mean(survived)),
        "mean_reward":       float(np.mean(ep_rewards)),
        "std_reward":        float(np.std(ep_rewards)),
        "mean_ep_length":    float(np.mean(ep_lengths)),
        "mean_rmse_rad":     float(np.mean(rmse_list)),
        "std_rmse_rad":      float(np.std(rmse_list)),
        "max_phi_mean_deg":  float(np.rad2deg(np.mean(max_phi_abs))),
        # [FIX-3] انرژی کنترل
        "mean_control_energy": float(np.mean(ep_control_energy)),
        "std_control_energy":  float(np.std(ep_control_energy)),
        "mean_tau_abs_mean":   float(np.mean(ep_tau_abs_mean)),
        "mean_tau_rms":        float(np.mean(ep_tau_rms)),
    }

    if use_cbf:
        results.update({
            "cbf_filter_rate":        float(np.mean(cbf_rates)),
            "cbf_interventions_mean": float(np.mean(cbf_int_count)),
            "cbf_slack_rate":         float(inner_env.hocbf.slack_rate) if inner_env.hocbf else 0.0,
            "cbf_alpha1":             cbf_alpha1,
            "cbf_alpha2":             cbf_alpha2,
            "cbf_lambda_slack":       lambda_slack,
            "cbf_type":               "HOCBF (relative_degree=2, soft)",
        })

    print("\n" + "=" * 62)
    print(f"  نتایج {label}")
    print("=" * 62)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:38s}: {v:.6f}")
        else:
            print(f"  {k:38s}: {v}")
    print("=" * 62)
    env.close()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# wrappers برای run_all_baselines.py
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_plain(n_episodes=200, curriculum_level=8, seed=42,
                   model_path=None) -> dict:
    return evaluate(use_cbf=False, n_episodes=n_episodes,
                    curriculum_level=curriculum_level, seed=seed,
                    model_path=model_path or MODEL_PLAIN_PATH)


def evaluate_cbf(n_episodes=200, curriculum_level=8, seed=42,
                 model_path=None, cbf_alpha=2.0, lambda_slack=1e4) -> dict:
    return evaluate(use_cbf=True, n_episodes=n_episodes,
                    curriculum_level=curriculum_level, seed=seed,
                    model_path=model_path or MODEL_CBF_PATH,
                    cbf_alpha1=cbf_alpha, cbf_alpha2=cbf_alpha,
                    lambda_slack=lambda_slack)


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=["train", "eval", "both"],
                        default="train")
    parser.add_argument("--model",
                        choices=["both", "sac_plain", "sac_cbf"],
                        default="both")
    parser.add_argument("--steps",        type=int,   default=3_000_000)
    parser.add_argument("--envs",         type=int,   default=8)
    parser.add_argument("--episodes",     type=int,   default=200)
    parser.add_argument("--curriculum",   type=int,   default=8)
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--device",       default="cpu")
    parser.add_argument("--cbf_alpha1",   type=float, default=2.0,
                        help="α₁: نرخ decay برای ψ₁")
    parser.add_argument("--cbf_alpha2",   type=float, default=2.0,
                        help="α₂: نرخ decay برای شرط HOCBF")
    parser.add_argument("--lambda_slack", type=float, default=1e4,
                        help="λ: وزن جریمه slack در QP (بزرگتر→سخت‌تر، کوچکتر→نرم‌تر)")
    parser.add_argument("--plain_model",  default=MODEL_PLAIN_PATH)
    parser.add_argument("--cbf_model",    default=MODEL_CBF_PATH)
    args = parser.parse_args()

    if args.device == "auto":
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    all_results = []

    # ── آموزش ────────────────────────────────────────────────────────────
    if args.mode in ("train", "both"):
        if args.model in ("both", "sac_plain"):
            train(use_cbf=False, total_steps=args.steps,
                  n_envs=args.envs, seed=args.seed,
                  device=args.device, load_model=args.plain_model)

        if args.model in ("both", "sac_cbf"):
            train(use_cbf=True, total_steps=args.steps,
                  n_envs=args.envs, seed=args.seed,
                  device=args.device,
                  cbf_alpha1=args.cbf_alpha1,
                  cbf_alpha2=args.cbf_alpha2,
                  lambda_slack=args.lambda_slack,
                  load_model=args.cbf_model)

    # ── ارزیابی ───────────────────────────────────────────────────────────
    if args.mode in ("eval", "both"):
        if args.model in ("both", "sac_plain"):
            r = evaluate(use_cbf=False,
                         n_episodes=args.episodes,
                         curriculum_level=args.curriculum,
                         seed=args.seed,
                         model_path=args.plain_model)
            all_results.append(r)
            fname = f"results_sac_plain_cl{args.curriculum}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=2, ensure_ascii=False)
            print(f"  → {fname}")

        if args.model in ("both", "sac_cbf"):
            r = evaluate(use_cbf=True,
                         n_episodes=args.episodes,
                         curriculum_level=args.curriculum,
                         seed=args.seed,
                         model_path=args.cbf_model,
                         cbf_alpha1=args.cbf_alpha1,
                         cbf_alpha2=args.cbf_alpha2,
                         lambda_slack=args.lambda_slack)
            all_results.append(r)
            fname = f"results_sac_cbf_cl{args.curriculum}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=2, ensure_ascii=False)
            print(f"  → {fname}")

        # جدول مقایسه‌ای
        if len(all_results) == 2:
            print("\n" + "═" * 88)
            print("  مقایسه SAC Plain vs SAC+HOCBF")
            print("═" * 88)
            print(f"  {'Metric':<32} {'SAC Plain':>22} {'SAC+HOCBF':>22}")
            print("─" * 88)
            keys = [
                ("survival_rate",       "Survival Rate"),
                ("mean_reward",         "Mean Reward"),
                ("mean_rmse_rad",       "RMSE [rad]"),
                ("max_phi_mean_deg",    "Max φ [°]"),
                ("mean_control_energy", "Control Energy Σ‖τ‖²"),
                ("mean_tau_rms",        "Mean τ_rms [N·m]"),
                ("mean_ep_length",      "Mean Ep Length"),
            ]
            for key, label in keys:
                v0 = all_results[0].get(key, float("nan"))
                v1 = all_results[1].get(key, float("nan"))
                print(f"  {label:<32} {v0:>22.4f} {v1:>22.4f}")
            if "cbf_filter_rate" in all_results[1]:
                print(f"  {'CBF Filter Rate':<32} {'—':>22} "
                      f"{all_results[1]['cbf_filter_rate']:>21.1%}")
                print(f"  {'CBF Slack Rate (relaxed)':<32} {'—':>22} "
                      f"{all_results[1].get('cbf_slack_rate', 0.0):>21.1%}")
            print("═" * 88)

    print("\n  تمام شد.\n")


if __name__ == "__main__":
    main()
