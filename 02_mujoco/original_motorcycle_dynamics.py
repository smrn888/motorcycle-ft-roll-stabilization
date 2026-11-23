"""
motorcycle_dynamics.py
======================
دینامیک غیرخطی موتورسیکلت — جدا از Gym/env برای توسعه کنترلر.

State vector:
    x = [φ, φ̇, δ_f, δ_r, δ̇_f, δ̇_r]

        φ        : roll angle          [rad]
        φ̇        : roll rate           [rad/s]
        δ_f      : front gimbal angle  [rad]
        δ_r      : rear gimbal angle   [rad]
        δ̇_f      : front gimbal rate   [rad/s]
        δ̇_r      : rear gimbal rate    [rad/s]

Control input:
    u = [τ_f, τ_r]   — CMG gimbal torques  [N·m]
    (مقیاس واقعی — نه نرمال‌شده)

پارامترهای فیزیکی:
    m      = 85.0   kg          — جرم موتور+سوار
    h      = 0.50   m           — ارتفاع مرکز جرم
    g      = 9.81   m/s²
    Iw     = 0.50   kg·m²       — ممان اینرسی چرخ CMG
    Omega  = 200.0  rad/s       — سرعت زاویه‌ای اسپین CMG
    I_delta= 0.10   kg·m²       — ممان اینرسی گیمبال
    L      = 1.20   m           — فاصله محور چرخ‌ها

معادلات اصلی (پیوسته):
    φ̈  = (m·g·h·sin(φ) − Iw·Ω·(δ̇_f − δ̇_r)·cos(φ)) / (m·h²)
    δ̈_f = τ_f / I_delta − c·δ̇_f
    δ̈_r = τ_r / I_delta − c·δ̇_r
    c  = GIMBAL_DAMP = 2.0   [s⁻¹]

رابط اصلی:
    dyn = MotorcycleDynamics()
    x_next = dyn.step(x, u)                   # یه قدم RK4  (Ts=0.01 s)
    xdot   = dyn.f(x, u)                      # مشتق پیوسته
    A_d, B_d, bias_d = dyn.linearise(phi)     # گسسته خطی‌شده حول phi
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, solve_discrete_are
from typing import Tuple


# ═══════════════════════════════════════════════════════════════════════════
# پارامترهای ثابت
# ═══════════════════════════════════════════════════════════════════════════

PARAMS = dict(
    m       = 85.0,     # kg
    h       = 0.50,     # m
    g       = 9.81,     # m/s²
    Iw      = 0.50,     # kg·m²
    Omega   = 200.0,    # rad/s
    I_delta = 0.10,     # kg·m²
    L       = 1.20,     # m
)

# محدودیت‌های ساختاری (برای clamp در step)
DELTA_MAX     = np.pi / 2       # [rad]
MAX_DELTA_DOT = 20.0            # [rad/s]
TAU_MAX       = 50.0            # [N·m]   — تورک حداکثر گیمبال
GIMBAL_DAMP   = 2.0             # [s⁻¹]   — میرایی گیمبال
Ts            = 0.01            # [s]      — زمان گام


# ═══════════════════════════════════════════════════════════════════════════
class MotorcycleDynamics:
    """
    دینامیک غیرخطی موتورسیکلت CMG.

    این کلاس فقط فیزیک است — هیچ وابستگی به Gym، reward، یا curriculum ندارد.
    برای توسعه هر نوع کنترلر (LQR، MPC، RL، sliding mode، ...) آماده است.

    پارامترهای سازنده:
        params      : dict پارامترهای فیزیکی (پیش‌فرض = PARAMS بالا)
        ts          : زمان گام گسسته [s]  (پیش‌فرض = 0.01)
        use_rk4     : True → RK4، False → اویلر صریح
        m_scale     : ضریب مقیاس جرم (برای uncertainty / robustness test)
        Iw_scale    : ضریب مقیاس Iw
        enforce_limits : True → clamp حالت‌های گیمبال بعد از هر step
    """

    def __init__(
        self,
        params: dict   = None,
        ts: float      = Ts,
        use_rk4: bool  = True,
        m_scale: float = 1.0,
        Iw_scale: float= 1.0,
        enforce_limits: bool = True,
    ):
        p = dict(PARAMS)
        if params is not None:
            p.update(params)
        self.p   = p
        self.Ts  = ts
        self.use_rk4       = use_rk4
        self.enforce_limits= enforce_limits

        # ── ثوابت مرکّب ────────────────────────────────────────────────────
        self.mgh   = p['m'] * p['g'] * p['h'] * m_scale          # N·m
        self.IwO   = p['Iw'] * p['Omega'] * Iw_scale             # kg·m²/s
        self.Iphi  = p['m'] * p['h'] ** 2 * m_scale              # kg·m²
        self.Id    = p['I_delta']                                 # kg·m²
        self.c     = GIMBAL_DAMP

    # ─────────────────────────────────────────────────────────────────────
    # ۱. معادله دیفرانسیل پیوسته  ẋ = f(x, u)
    # ─────────────────────────────────────────────────────────────────────
    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        مشتق پیوسته حالت.

        ورودی:
            x : (6,)  — [φ, φ̇, δ_f, δ_r, δ̇_f, δ̇_r]
            u : (2,)  — [τ_f, τ_r]  در مقیاس واقعی [N·m]

        خروجی:
            ẋ : (6,)  — [φ̇, φ̈, δ̇_f, δ̇_r, δ̈_f, δ̈_r]

        معادله اصلی roll:
            φ̈ = (mgh·sin(φ) − IwΩ·(δ̇_f − δ̇_r)·cos(φ)) / I_phi

        دینامیک گیمبال (angular velocity model با damping):
            δ̈_f = τ_f/I_delta − c·δ̇_f
            δ̈_r = τ_r/I_delta − c·δ̇_r
        """
        phi, phi_dot, delta_f, delta_r, ddot_f, ddot_r = x
        tau_f, tau_r = u

        cp = np.cos(phi)
        sp = np.sin(phi)

        phi_ddot = (
            self.mgh * sp
            - self.IwO * (ddot_f - ddot_r) * cp
        ) / self.Iphi

        return np.array([
            phi_dot,                            # dφ/dt
            phi_ddot,                           # dφ̇/dt
            ddot_f,                             # dδ_f/dt
            ddot_r,                             # dδ_r/dt
            tau_f / self.Id - self.c * ddot_f,  # dδ̇_f/dt
            tau_r / self.Id - self.c * ddot_r,  # dδ̇_r/dt
        ])

    # ─────────────────────────────────────────────────────────────────────
    # ۲. انتگرال‌گیری: RK4 یا اویلر
    # ─────────────────────────────────────────────────────────────────────
    def rk4_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """یه قدم RK4 به اندازه Ts."""
        dt = self.Ts
        k1 = self.f(x,               u)
        k2 = self.f(x + 0.5*dt*k1,  u)
        k3 = self.f(x + 0.5*dt*k2,  u)
        k4 = self.f(x +     dt*k3,  u)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def euler_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """یه قدم اویلر صریح (برای مقایسه سریع)."""
        return x + self.Ts * self.f(x, u)

    def step(self, x: np.ndarray, u: np.ndarray,
             clamp_u: bool = False) -> np.ndarray:
        """
        قدم اصلی — RK4 یا اویلر بسته به use_rk4.

        پارامترها:
            x       : حالت فعلی (6,)
            u       : کنترل (2,)  — در مقیاس واقعی [N·m]
            clamp_u : اگر True، ورودی را قبل از propagation کلمپ می‌کند

        خروجی:
            x_next : حالت بعدی (6,)   (limits اعمال‌شده اگر enforce_limits=True)
        """
        u = np.asarray(u, dtype=np.float64)
        if clamp_u:
            u = np.clip(u, -TAU_MAX, TAU_MAX)

        if self.use_rk4:
            x_next = self.rk4_step(x, u)
        else:
            x_next = self.euler_step(x, u)

        if self.enforce_limits:
            x_next[2] = np.clip(x_next[2], -DELTA_MAX,     DELTA_MAX)
            x_next[3] = np.clip(x_next[3], -DELTA_MAX,     DELTA_MAX)
            x_next[4] = np.clip(x_next[4], -MAX_DELTA_DOT, MAX_DELTA_DOT)
            x_next[5] = np.clip(x_next[5], -MAX_DELTA_DOT, MAX_DELTA_DOT)

        return x_next

    # ─────────────────────────────────────────────────────────────────────
    # ۳. خطی‌سازی حول یه نقطه کار
    # ─────────────────────────────────────────────────────────────────────
    def linearise(self, phi: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        خطی‌سازی دقیق حول φ = phi  (state بقیه صفر، u=0).

        معادله پیوسته خطی‌شده:
            ẋ ≈ Ac·x + Bc·u + bias_c

        گسسته‌سازی ZOH دقیق با Augmented Matrix:
            [v16-FIX1] bias_d از ماتریس augmented استخراج می‌شه:
            M = [[Ac, Bc, bias_c],
                 [0,  0,  0    ],
                 [0,  0,  0    ]]
            eM = expm(M * Ts)
            A_d    = eM[:n, :n]
            B_d    = eM[:n, n:n+m]
            bias_d = eM[:n, -1]       ← دقیق، نه تقریب اویلر

        تفاوت با روش قبلی: bias_d = bias_c * Ts فقط تقریب مرتبه اول
        بود. در phi > 30° این خطا به اندازه‌ای بزرگه که MPC
        مسیر اشتباه پیش‌بینی می‌کنه و EXPLODE رخ می‌ده.

        خروجی:
            A_d   : (6,6)
            B_d   : (6,2)
            bias_d: (6,)
        """
        cp = np.cos(phi)
        sp = np.sin(phi)

        # ── ماتریس پیوسته Jacobian  ∂f/∂x ───────────────────────────────
        Ac = np.zeros((6, 6))
        Ac[0, 1] =  1.0
        Ac[1, 0] =  self.mgh * cp / self.Iphi      # ∂φ̈/∂φ
        Ac[1, 4] = -self.IwO * cp / self.Iphi      # ∂φ̈/∂δ̇_f
        Ac[1, 5] =  self.IwO * cp / self.Iphi      # ∂φ̈/∂δ̇_r  (علامت مخالف)
        Ac[2, 4] =  1.0                             # dδ_f/dt = δ̇_f
        Ac[3, 5] =  1.0                             # dδ_r/dt = δ̇_r
        Ac[4, 4] = -self.c                          # damping گیمبال جلو
        Ac[5, 5] = -self.c                          # damping گیمبال عقب

        # ── ماتریس کنترل  ∂f/∂u ─────────────────────────────────────────
        Bc = np.zeros((6, 2))
        Bc[4, 0] = 1.0 / self.Id   # τ_f → δ̈_f
        Bc[5, 1] = 1.0 / self.Id   # τ_r → δ̈_r

        # ── bias پیوسته ناشی از خطی‌سازی حول phi != 0 ───────────────────
        # f(x0, 0) - Ac @ x0   که x0 = [phi, 0, 0, 0, 0, 0]
        bias_c    = np.zeros(6)
        bias_c[1] = self.mgh * (sp - cp * phi) / self.Iphi

        # ── [v16-FIX1] گسسته‌سازی ZOH دقیق با Augmented Matrix ──────────
        # ساختار ماتریس بزرگ:
        #   [ Ac    | Bc    | bias_c ]   ← n سطر
        #   [ 0     | 0     | 0      ]   ← m سطر (control)
        #   [ 0     | 0     | 0      ]   ← 1 سطر (bias scalar)
        # بعد از expm(M * Ts):
        #   A_d    = eM[:n, :n]
        #   B_d    = eM[:n, n:n+m]
        #   bias_d = eM[:n, -1]   ← این مقدار = ∫₀ᵀˢ exp(Ac*t)*bias_c dt
        n, m = 6, 2
        M = np.zeros((n + m + 1, n + m + 1))
        M[:n, :n]      = Ac
        M[:n, n:n + m] = Bc
        M[:n, -1]      = bias_c
        # سطرهای n تا آخر صفر هستند — standard ZOH augmentation

        eM     = expm(M * self.Ts)
        A_d    = eM[:n, :n]
        B_d    = eM[:n, n:n + m]
        bias_d = eM[:n, -1]          # ← دقیق، نه bias_c * Ts

        return A_d, B_d, bias_d

    # ─────────────────────────────────────────────────────────────────────
    # ۴. ابزارهای کمکی
    # ─────────────────────────────────────────────────────────────────────
    def jacobian_continuous(self, phi: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        ماتریس‌های Ac, Bc پیوسته برای phi داده‌شده.
        مستقیماً برای طراحی کنترلر پیوسته (pole placement، H∞، ...) استفاده می‌شود.
        """
        cp = np.cos(phi)
        Ac = np.zeros((6, 6))
        Ac[0, 1] =  1.0
        Ac[1, 0] =  self.mgh * cp / self.Iphi
        Ac[1, 4] = -self.IwO * cp / self.Iphi
        Ac[1, 5] =  self.IwO * cp / self.Iphi
        Ac[2, 4] =  1.0
        Ac[3, 5] =  1.0
        Ac[4, 4] = -self.c
        Ac[5, 5] = -self.c

        Bc = np.zeros((6, 2))
        Bc[4, 0] = 1.0 / self.Id
        Bc[5, 1] = 1.0 / self.Id
        return Ac, Bc

    def simulate(
        self,
        x0: np.ndarray,
        policy,
        n_steps: int,
        clamp_u: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        شبیه‌سازی open/closed-loop از x0.

        پارامترها:
            x0      : حالت اولیه (6,)
            policy  : callable  u = policy(x)  → (2,)
            n_steps : تعداد قدم‌ها
            clamp_u : اعمال محدودیت TAU_MAX روی u

        خروجی:
            X : (n_steps+1, 6)  — تاریخچه حالت
            U : (n_steps,   2)  — تاریخچه کنترل
        """
        X = np.zeros((n_steps + 1, 6))
        U = np.zeros((n_steps,     2))
        X[0] = np.asarray(x0, dtype=np.float64)

        for k in range(n_steps):
            u = np.asarray(policy(X[k]), dtype=np.float64)
            if clamp_u:
                u = np.clip(u, -TAU_MAX, TAU_MAX)
            U[k]     = u
            X[k + 1] = self.step(X[k], u)

        return X, U

    def is_crashed(self, x: np.ndarray, crash_angle: float = np.pi / 4) -> bool:
        """آیا زاویه roll از crash_angle (پیش‌فرض 45°) تجاوز کرده؟"""
        return bool(abs(x[0]) > crash_angle)

    @property
    def state_labels(self):
        return ["φ [rad]", "φ̇ [rad/s]", "δ_f [rad]", "δ_r [rad]",
                "δ̇_f [rad/s]", "δ̇_r [rad/s]"]

    @property
    def input_labels(self):
        return ["τ_f [N·m]", "τ_r [N·m]"]


# ═══════════════════════════════════════════════════════════════════════════
# تست سریع
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    dyn = MotorcycleDynamics()

    print("── پارامترها ──────────────────────────────────")
    for k, v in dyn.p.items():
        print(f"  {k:10s} = {v}")
    print(f"  Iphi = {dyn.Iphi:.4f}  kg·m²")
    print(f"  IwΩ  = {dyn.IwO:.4f}  kg·m²/s")

    print("\n── خطی‌سازی در phi=0 ──────────────────────────")
    A_d, B_d, bias_d = dyn.linearise(0.0)
    print(f"  A_d[1,0] = {A_d[1,0]:.6f}   (mgh*cos0/Iphi*Ts ≈ {dyn.mgh/dyn.Iphi*dyn.Ts:.6f})")
    print(f"  B_d[4,0] = {B_d[4,0]:.6f}")

    print("\n── شبیه‌سازی open-loop از phi=20° با u=0 ──────")
    x0 = np.array([np.deg2rad(20.), 0., 0., 0., 0., 0.])
    X, U = dyn.simulate(x0, policy=lambda x: np.zeros(2), n_steps=10)
    for k, xk in enumerate(X):
        print(f"  step {k:2d}  phi={np.rad2deg(xk[0]):+7.3f}°  phi_dot={xk[1]:+.4f}")

    print("\n── مقایسه linear vs nonlinear (یه قدم) ────────")
    print(f"  {'phi0':>6}  {'NL φ_next':>12}  {'Lin φ_next':>12}  {'err [°]':>10}")
    A0, B0, bias0 = dyn.linearise(0.0)
    u0 = np.zeros(2)
    for deg in [5, 10, 20, 30, 40]:
        phi0  = np.deg2rad(deg)
        xtest = np.array([phi0, 0., 0., 0., 0., 0.])
        nl    = dyn.step(xtest, u0)
        lin   = A0 @ xtest + B0 @ u0 + bias0
        err   = abs(np.rad2deg(nl[0] - lin[0]))
        print(f"  {deg:>5}°  {np.rad2deg(nl[0]):>12.6f}  {np.rad2deg(lin[0]):>12.6f}  {err:>10.6f}°")

    print("\n── [v16-FIX1] مقایسه bias_d قدیم (Euler) vs جدید (ZOH دقیق) ──")
    print(f"  {'phi':>6}  {'bias_old[1]':>14}  {'bias_new[1]':>14}  {'diff':>12}")
    for deg in [10, 20, 30, 40]:
        phi0 = np.deg2rad(deg)
        cp0, sp0 = np.cos(phi0), np.sin(phi0)
        bias_c_1 = dyn.mgh * (sp0 - cp0 * phi0) / dyn.Iphi
        bias_old = bias_c_1 * dyn.Ts                # تقریب اویلر قبلی
        _, _, bias_new = dyn.linearise(phi0)
        print(f"  {deg:>5}°  {bias_old:>14.8f}  {bias_new[1]:>14.8f}  {abs(bias_new[1]-bias_old):>12.2e}")
