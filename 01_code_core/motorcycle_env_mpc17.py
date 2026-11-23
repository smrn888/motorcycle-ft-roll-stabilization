"""
motorcycle_env_mpc17.py
=======================
RL-TVL-MPC v17 (patched v17.1: solver pool به‌جای rebuild) — چهار اصلاح ساختاری نسبت به mpc16:

  [FIX-1] جریمه نرخ تغییر کنترل (Δu):
        cost += 0.5 * ||u_k - u_{k-1}||²
        → نوسان کنترل کم می‌شه، QP پایدارتر می‌شه، survival بالا می‌ره

  [FIX-2] قید MAX_DELTA_DOT در MPC:
        constraints: ||X[4:6, k+1]||_inf ≤ MAX_DELTA_DOT = 20 rad/s
        → mismatch بین plant (که clip می‌کنه) و MPC (که نمی‌دونست) برطرف شد

  [FIX-3] Adaptive Horizon (N بر اساس |phi|):
        |phi| < 10°  → N = 15   (سریع‌تر)
        |phi| > 25°  → N = 40   (آینده‌نگری بیشتر در زوایای بحرانی)
        → در near-limit regime پیش‌بینی عمیق‌تر، در حالت عادی سریع‌تر

  [FIX-4] Terminal cost با P_mat کامل (PSD):
        cp.quad_form(X[:,N], P_mat)  به‌جای  cp.sum(multiply(P_diag, square(X[:,N])))
        → تمام coupling‌های DARE حفظ می‌شه — افت کیفیت terminal cost حذف می‌شه

(اصلاحات v16 محفوظند: bias_d دقیق، P آنلاین، Low-pass filter وزن‌ها)
"""

import numpy as np
import cvxpy as cp
import gymnasium as gym
from scipy.linalg import solve_discrete_are

from motorcycle_dynamics import MotorcycleDynamics, TAU_MAX, DELTA_MAX, MAX_DELTA_DOT

import os as _os
# [diag] سوییچ‌های آزمایشی (پیش‌فرض = رفتار قبلی):
#   MPC_MAX_ITER=20000          سقف تکرار OSQP (پیش‌فرض 4000)
#   MPC_FALLBACK=iterate        اگر OSQP به max_iter رسید، iterate آخر را استفاده کن (پیش‌فرض: zero)
_MPC_MAX_ITER = int(_os.environ.get("MPC_MAX_ITER", "4000"))
_MPC_FALLBACK = _os.environ.get("MPC_FALLBACK", "zero")
#   MPC_RATE=hard|soft|off      قید نرخ گیمبال |dδ̇|≤20: سخت (پیش‌فرض، مثل آموزش) / نرم با slack / حذف
_MPC_RATE = _os.environ.get("MPC_RATE", "hard")
#   MPC_RHO=1.0                 rho ثابت OSQP (adaptive_rho خاموش)؛ پیش‌فرض: تنظیم خودکار OSQP
#   MPC_EPS=1e-3                تلرانس eps_abs/eps_rel (پیش‌فرض 1e-4)
_MPC_RHO = _os.environ.get("MPC_RHO")
_MPC_EPS = float(_os.environ.get("MPC_EPS", "1e-4"))
_OSQP_EXTRA = ({"rho": float(_MPC_RHO), "adaptive_rho": False} if _MPC_RHO else {})
#   MPC_REWARD=v2               reward جدید: هر گام ≥ 0 و جریمه‌ی سقوط پایانی بزرگ (پیش‌فرض v1 = رفتار قبلی)
#   MPC_CRASH_PEN=300           اندازه‌ی جریمه‌ی سقوط در v2 (reward گام آخر = -MPC_CRASH_PEN)
_MPC_REWARD = _os.environ.get("MPC_REWARD", "v1")
_MPC_CRASH_PEN = float(_os.environ.get("MPC_CRASH_PEN", "300"))
#   MPC_TRACK_K=20              ضریب exp(-K·err²) در reward v2 (پیش‌فرض ۲۰ = رفتار قبلی)
#   MPC_TRACK_LIN=0             ضریب جمله‌ی خطی -LIN·|err| در v2 (غیراشباع؛ پیش‌فرض ۰ = رفتار قبلی)
_MPC_TRACK_K = float(_os.environ.get("MPC_TRACK_K", "20"))
_MPC_TRACK_LIN = float(_os.environ.get("MPC_TRACK_LIN", "0"))
assert _MPC_REWARD in ("v1", "v2")
assert _MPC_RATE in ("hard", "soft", "off")


# ═══════════════════════════════════════════════════════════════════════════
# MPC Solver  (بدون تغییر نسبت به mpc13)
# ═══════════════════════════════════════════════════════════════════════════

class MPCSolver:
    """
    QP محدب — حل‌کننده OSQP با warm-start.

    ورودی‌های online (Parameter):
        x0_param : حالت اولیه (خطای tracking)
        b_param  : bias خطی‌سازی حول نقطه کار
        Q_diag / R_diag / P_diag : وزن‌های هزینه (از RL میان)

    [v15] A_param / B_param : ماتریس‌های مدل online آپدیت می‌شن
          update_model(A_d, B_d, bias_d) رو هر step صدا بزن.
    """

    def __init__(self, A, B, N=20, u_max=TAU_MAX, delta_max=DELTA_MAX):
        self.N   = N
        self.n_x = A.shape[0]
        self.n_u = B.shape[1]
        self.u_max     = u_max
        self.delta_max = delta_max

        # [v15] A و B به‌صورت Parameter — قابل آپدیت هر step
        self.A_param = cp.Parameter((self.n_x, self.n_x))
        self.B_param = cp.Parameter((self.n_x, self.n_u))
        self.A_param.value = A.copy()
        self.B_param.value = B.copy()

        self.X = cp.Variable((self.n_x, N + 1))
        self.U = cp.Variable((self.n_u, N))

        self.x0_param = cp.Parameter(self.n_x)
        self.b_param  = cp.Parameter(self.n_x)
        self.Q_diag   = cp.Parameter(self.n_x, nonneg=True)
        self.R_diag   = cp.Parameter(self.n_u, nonneg=True)
        # [v17-FIX4] P_chol = cholesky(P_mat) — DPP-compliant terminal cost
        # terminal cost = ||P_chol.T @ x_N||² ≡ x_N.T P x_N
        self.P_chol   = cp.Parameter((self.n_x, self.n_x))

        # مقادیر پیش‌فرض
        self.Q_diag.value   = np.array([100., 10., 20., 20., 1., 1.])
        self.R_diag.value   = np.array([0.1, 0.1])
        _P0 = np.diag([500., 50., 100., 100., 5., 5.])
        self.P_chol.value   = np.linalg.cholesky(_P0)
        self.x0_param.value = np.zeros(self.n_x)
        self.b_param.value  = np.zeros(self.n_x)
        self._bias_nominal  = np.zeros(self.n_x)   # init — قبل از اولین update_model

        # [v17-FIX1] پارامتر u_prev برای جریمه Δu
        self.u_prev = cp.Parameter(self.n_u)
        self.u_prev.value = np.zeros(self.n_u)

        self._prob = self._build_problem()
        self.last_solve_time   = 0.0
        self.last_solve_status = "init"
        self.last_raw_status = "init"   # [diag] وضعیت خام cvxpy / exception
        print(f"[MPCSolver] DPP-compliant: {self._prob.is_dcp(dpp=True)}")

    def _build_problem(self):
        N = self.N
        A, B = self.A_param, self.B_param
        cost = 0
        constraints = [self.X[:, 0] == self.x0_param]

        for k in range(N):
            xt  = self.X[:, k]
            u_k = self.U[:, k]
            cost += cp.sum(cp.multiply(self.Q_diag, cp.square(xt)))
            cost += cp.sum(cp.multiply(self.R_diag, cp.square(u_k)))

            # [v17-FIX1] جریمه Δu — کاهش نوسان و افزایش stability
            if k == 0:
                du = u_k - self.u_prev          # تفاضل با آخرین u واقعی
            else:
                du = u_k - self.U[:, k - 1]    # تفاضل با u قبلی در افق
            cost += 0.5 * cp.sum_squares(du)

            constraints.append(
                self.X[:, k + 1] == A @ xt + B @ u_k + self.b_param)
            constraints.append(cp.norm_inf(u_k) <= self.u_max)
            constraints.append(
                cp.norm_inf(self.X[2:4, k + 1]) <= self.delta_max)
            # [v17-FIX2] قید نرخ گیمبال — سازگار با plant که clip می‌کنه
            if _MPC_RATE == "hard":
                constraints.append(
                    cp.norm_inf(self.X[4:6, k + 1]) <= MAX_DELTA_DOT)
            elif _MPC_RATE == "soft":
                # [diag] slack: QP همیشه شدنی می‌ماند؛ تخطی با جریمه‌ی سنگین
                s_k = cp.Variable(2, nonneg=True)
                constraints.append(self.X[4:6, k + 1] <= MAX_DELTA_DOT + s_k)
                constraints.append(self.X[4:6, k + 1] >= -MAX_DELTA_DOT - s_k)
                cost += 1e3 * cp.sum_squares(s_k)

        # [v17-FIX4] terminal cost با P کامل — DPP-compliant از طریق Cholesky
        # x_N.T P x_N = ||P_chol.T @ x_N||²  (P_chol = lower-triangular sqrt)
        cost += cp.sum_squares(self.P_chol.T @ self.X[:, N])
        return cp.Problem(cp.Minimize(cost), constraints)

    def update_model(self, A_d: np.ndarray, B_d: np.ndarray, bias_d: np.ndarray):
        """
        [v15] آپدیت online مدل خطی.
        هر step با phi جدید صدا زده می‌شه.
        bias_d به‌صورت جداگانه ذخیره می‌شه تا در solve تجمع نشه.
        """
        self.A_param.value = A_d
        self.B_param.value = B_d
        self._bias_nominal = bias_d.copy()   # ذخیره جداگانه — بدون تجمع

    def solve(self, x0, xd, Q_diag, R_diag, P_chol):
        import time as _time
        self.x0_param.value = x0 - xd
        AmI = self.A_param.value - np.eye(self.n_x)
        self.b_param.value  = self._bias_nominal + AmI @ xd
        self.Q_diag.value   = np.maximum(Q_diag, 1e-6)
        self.R_diag.value   = np.maximum(R_diag, 1e-6)
        # [v17-FIX4] P_chol = cholesky(DARE) — coupling کامل حفظ می‌شه
        self.P_chol.value   = P_chol

        _t0 = _time.perf_counter()
        try:
            self._prob.solve(
                solver=cp.OSQP,
                warm_start=True,
                eps_abs=_MPC_EPS,
                eps_rel=_MPC_EPS,
                max_iter=_MPC_MAX_ITER,
                **_OSQP_EXTRA,
                verbose=False,
            )
            self.last_raw_status = str(self._prob.status)
            if (self._prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                    and self.U.value is not None):
                self.last_solve_time = _time.perf_counter() - _t0
                self.last_solve_status = self._prob.status
                u_opt = self.U.value[:, 0].copy()
                # [v17-FIX1] آپدیت u_prev برای Δu گام بعدی
                self.u_prev.value = u_opt
                return u_opt
        except Exception as _e:
            self.last_raw_status = "exception:" + type(_e).__name__
        # [diag] fallback آزمایشی: iterate آخر OSQP وقتی به max_iter رسیده (MPC_FALLBACK=iterate)
        if (_MPC_FALLBACK == "iterate"
                and getattr(self._prob, "status", None) == cp.USER_LIMIT
                and self.U.value is not None
                and np.all(np.isfinite(self.U.value))):
            self.last_solve_time   = _time.perf_counter() - _t0
            self.last_solve_status = "user_limit_used"
            u_opt = self.U.value[:, 0].copy()
            self.u_prev.value = u_opt
            return u_opt
        self.last_solve_time   = _time.perf_counter() - _t0
        self.last_solve_status = "failed"
        return np.zeros(self.n_u)

    def reset_warm_start(self):
        self.X.value = np.zeros((self.n_x, self.N + 1))
        self.U.value = np.zeros((self.n_u, self.N))
        # [v17-FIX1] reset u_prev به صفر در شروع اپیزود جدید
        self.u_prev.value = np.zeros(self.n_u)


# ═══════════════════════════════════════════════════════════════════════════
# Environment
# ═══════════════════════════════════════════════════════════════════════════

class MotorcycleEnvMPC(gym.Env):
    """
    RL-TVL-MPC: RL وزن‌های Q,R رو تنظیم می‌کنه،
    MPC با مدل خطی local بهینه‌سازی می‌کنه،
    plant با RK4 nonlinear propagate می‌شه.

    Action (نرمال‌شده [-1,1]^4):
        [Q_phi, Q_phidot, Q_joint, R_tau]  → log-scale

    Observation (10-بُعدی):
        [φ, φ̇, δ_f, δ_r, δ̇_f, δ̇_r, v, α, φ_d, err]
    """

    MPC_N        = 20
    MAX_STEPS    = 500
    HORIZONS     = (15, 20, 40)   # [v17.1] افق‌های تطبیقی — solverها یک‌بار ساخته می‌شوند
    V_NOISE      = 0.05
    ALPHA_NOISE  = 0.005

    REWARD_CLIP_LOW  = -50.0
    REWARD_CLIP_HIGH =  20.0

    _A_LOW  = np.array([10.,   1.,   5.,  0.05], dtype=np.float64)
    _A_HIGH = np.array([500., 50., 100.,  2.0 ], dtype=np.float64)

    # [v15] curriculum کامل — شامل large-angle nonlinear regime
    CURRICULUM = [
        (0.02, 0.05),  # level 0  ~1.1°   linear regime
        (0.04, 0.10),  # level 1  ~2.3°
        (0.06, 0.15),  # level 2  ~3.4°
        (0.08, 0.20),  # level 3  ~4.6°
        (0.10, 0.25),  # level 4  ~5.7°
        (0.12, 0.30),  # level 5  ~6.9°   max مقاله قبلی
        (0.35, 0.80),  # level 6  ~20°    nonlinear regime شروع می‌شه
        (0.61, 1.20),  # level 7  ~35°    large-angle
        (0.76, 1.50),  # level 8  ~43°    near-limit — novelty مقاله
    ]

    def __init__(self):
        super().__init__()

        # ── دینامیک از ماژول جداگانه ─────────────────────────────────────
        self.dyn = MotorcycleDynamics()

        # خطی‌سازی اولیه در phi=0 فقط برای init MPC solver
        # [v16] _P_fixed_diag حذف شد — P هر step آنلاین حساب می‌شه (FIX-2)
        A0, B0, _ = self.dyn.linearise(phi=0.0)
        # [v17.1] solver pool: به‌جای rebuild در هر تغییر افق (که u_prev و
        # warm-start را از بین می‌برد و کند است)، هر افق یک‌بار ساخته می‌شود.
        assert self.MPC_N in self.HORIZONS
        self._solvers = {n: MPCSolver(A0, B0, N=n) for n in self.HORIZONS}
        self.mpc = self._solvers[self.MPC_N]

        # [v16] obs_high آپدیت شد برای large-angle curriculum
        obs_high = np.array([
            np.pi/2, 10.0, np.pi/2, np.pi/2,   # phi تا 90° (crash=45°)
            20.0, 20.0, 20.0, np.pi/3,
            np.pi/2, np.pi,                      # phi_d و err بزرگ‌تر
        ], dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            -obs_high, obs_high, dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low  = np.full(4, -1.0, dtype=np.float32),
            high = np.full(4,  1.0, dtype=np.float32),
            dtype=np.float32,
        )

        self.state            = np.zeros(6)
        self.v                = 0.0
        self.alpha            = 0.0
        self._steps           = 0
        self.curriculum_level = 0
        self.prev_action      = np.zeros(4, dtype=np.float32)

        # [v16-FIX3] Low-pass filter — وزن‌های smoothed MPC
        # مقدار اولیه = میانه log-scale برای هر وزن
        _t_mid = np.array([0.5, 0.5, 0.5, 0.5])
        self._smoothed_weights = self._denorm_action(_t_mid * 2.0 - 1.0)

    # ─────────────────────────────────────────────────────────────────────
    # helpers
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _dare_chol(A, B, Q_diag, R_diag):
        """[v17-FIX4] Cholesky factor از DARE — برای terminal cost DPP-compliant."""
        Q = np.diag(Q_diag)
        R = np.diag(R_diag)
        try:
            P = solve_discrete_are(A, B, Q, R)
            # symmetrize + epsilon برای اطمینان عددی
            P = (P + P.T) / 2.0 + np.eye(len(Q_diag)) * 1e-8
            return np.linalg.cholesky(P)
        except Exception:
            return np.diag(np.sqrt(Q_diag * 5.0))

    def _desired_roll(self) -> float:
        p = self.dyn.p
        if abs(self.alpha) < 1e-4 or abs(self.v) < 1e-3:
            return 0.0
        return float(np.arctan(
            self.v ** 2 * self.alpha / (p['L'] * p['g'])))

    def _denorm_action(self, norm_action: np.ndarray) -> np.ndarray:
        """
        Log-scale rescale: [-1,1] → [_A_LOW, _A_HIGH]
        مثل mpc13 (FIX-LAS v12).
        """
        t = (np.asarray(norm_action, dtype=np.float64).clip(-1.0, 1.0) + 1.0) / 2.0
        log_low  = np.log(self._A_LOW)
        log_high = np.log(self._A_HIGH)
        return np.exp(log_low + t * (log_high - log_low))

    def _danger_penalty(self, phi: float) -> float:
        # crash angle = 45° = pi/4
        ratio = abs(phi) / (np.pi / 4)
        return -40.0 * (ratio ** 6)

    def set_curriculum_level(self, level: int):
        self.curriculum_level = int(np.clip(level, 0, len(self.CURRICULUM) - 1))

    # ─────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._steps      = 0
        self.state       = np.zeros(6)
        self.prev_action = np.zeros(4, dtype=np.float32)

        # [v16-FIX3] reset فیلتر وزن‌ها به مقدار میانه
        _t_mid = np.array([0.5, 0.5, 0.5, 0.5])
        self._smoothed_weights = self._denorm_action(_t_mid * 2.0 - 1.0)

        phi_range, phidot_range = self.CURRICULUM[self.curriculum_level]
        self.state[0] = self.np_random.uniform(-phi_range, phi_range)
        self.state[1] = self.np_random.uniform(-phidot_range, phidot_range)

        self.v     = float(self.np_random.uniform(1.0, 8.0))
        self.alpha = float(self.np_random.uniform(-0.20, 0.20))

        self.mpc.reset_warm_start()

        phi_d = self._desired_roll()
        err   = self.state[0] - phi_d

        obs = np.array([
            self.state[0], self.state[1],
            self.state[2], self.state[3],
            self.state[4], self.state[5],
            self.v, self.alpha,
            phi_d, err,
        ], dtype=np.float32)

        return obs, {}

    def step(self, rl_action: np.ndarray):
        a = self._denorm_action(rl_action)
        self._steps += 1

        # جریمه smoothness (نرمال‌شده به range)
        _A_RANGE = self._A_HIGH - self._A_LOW
        delta_action_norm     = (a - self.prev_action) / _A_RANGE
        self.prev_action      = a.copy()
        reward_action_smooth  = -0.5 * np.sum(delta_action_norm ** 2)

        # ── [v16-FIX3] Low-pass filter روی وزن‌های MPC ───────────────────
        # از شوک ناگهانی QP جلوگیری می‌کنه:
        #   w_smooth = 0.8 * w_prev + 0.2 * w_new
        # RL یاد می‌گیره که action واقعی مشتق smoothed هست، نه raw
        _LP_ALPHA = 0.2   # سرعت tracking فیلتر
        self._smoothed_weights = (
            (1.0 - _LP_ALPHA) * self._smoothed_weights
            + _LP_ALPHA * a
        )
        a_smooth = self._smoothed_weights   # وزن‌های واقعی که به MPC می‌ره

        # ── وزن‌های MPC از action RL (smoothed) ──────────────────────────
        Q_diag = np.array([
            a_smooth[0], a_smooth[1],
            a_smooth[2], a_smooth[2],
            a_smooth[2] * 0.1, a_smooth[2] * 0.1,
        ])
        R_diag = np.array([a_smooth[3], a_smooth[3]])

        x_d      = np.zeros(6)
        phi_d    = self._desired_roll()
        x_d[0]  = phi_d

        # ── [v17-FIX3] Adaptive Horizon بر اساس |phi| ─────────────────────
        phi_deg = abs(np.rad2deg(self.state[0]))
        if phi_deg < 10.0:
            N_adaptive = 15
        elif phi_deg > 25.0:
            N_adaptive = 40
        else:
            N_adaptive = self.MPC_N   # = 20 حالت عادی
        if N_adaptive != self.mpc.N:
            # [v17.1] سوییچ بین solverهای از پیش‌ساخته؛ u_prev منتقل می‌شود
            # تا جریمه Δu در لحظه سوییچ جهش نکند. warm-start صفر می‌شود
            # (مقادیر قدیمی آن solver مربوط به اپیزود/زمان دیگری است).
            u_prev_old = self.mpc.u_prev.value.copy()
            self.mpc = self._solvers[N_adaptive]
            self.mpc.reset_warm_start()
            self.mpc.u_prev.value = u_prev_old

        # ── Online linearization حول phi فعلی ───────────────────────────
        # [FIX-1] bias_d حالا از motorcycle_dynamics دقیق استخراج می‌شه
        A_d, B_d, bias_d = self.dyn.linearise(self.state[0])
        self.mpc.update_model(A_d, B_d, bias_d)

        # ── [v17-FIX4] آپدیت آنلاین ماتریس هزینه نهایی P (کامل، نه diagonal) ──
        # P_chol = cholesky(DARE(A(phi), B(phi), Q, R))  هر step
        P_chol = self._dare_chol(A_d, B_d, Q_diag=Q_diag, R_diag=R_diag)

        # ── MPC → u_opt ──────────────────────────────────────────────────
        u_opt = self.mpc.solve(
            x0=self.state, xd=x_d,
            Q_diag=Q_diag, R_diag=R_diag, P_chol=P_chol,
        )
        u_opt = np.clip(u_opt, -TAU_MAX, TAU_MAX)

        # ── Plant: nonlinear RK4 ─────────────────────────────────────────
        self.state = self.dyn.step(self.state, u_opt)

        # ── noise محیطی (v, alpha) ────────────────────────────────────────
        self.v = float(np.clip(
            self.v + self.np_random.normal(0.0, self.V_NOISE), 1.0, 8.0))
        self.alpha = float(np.clip(
            self.alpha + self.np_random.normal(0.0, self.ALPHA_NOISE),
            -0.20, 0.20))

        # ── reward ───────────────────────────────────────────────────────
        phi  = self.state[0]
        phi_d = self._desired_roll()
        err  = phi - phi_d

        tracking_reward = -10.0 * err ** 2
        stability_bonus =   5.0 * float(np.exp(-50.0 * err ** 2))
        phi_dot_penalty =  -0.5 * abs(self.state[1])
        control_cost    = -0.01 * np.sum(np.abs(u_opt))
        danger_penalty  = self._danger_penalty(phi)
        survival_bonus  =   1.5

        raw_reward = (
            tracking_reward
            + stability_bonus
            + phi_dot_penalty
            + control_cost
            + danger_penalty
            + survival_bonus
            + reward_action_smooth
        )
        reward = float(np.clip(raw_reward, self.REWARD_CLIP_LOW, self.REWARD_CLIP_HIGH))
        if _MPC_REWARD == "v2":
            # [v2] هر گام زنده ماندن ≥ 0 (تا خودکشی زودهنگام سود نداشته باشد)،
            # ردیابی پاداش کران‌دار، و سقوط = جریمه‌ی بزرگ پایانی (خارج از clip)
            raw_reward = (1.0 + float(np.exp(-_MPC_TRACK_K * err ** 2))
                          - _MPC_TRACK_LIN * abs(err)
                          - 0.05 * abs(self.state[1])
                          - 0.002 * float(np.sum(np.abs(u_opt)))
                          + reward_action_smooth)
            reward = float(np.clip(raw_reward, 0.0, 2.0))
            if abs(phi) > np.pi / 4:
                reward = -_MPC_CRASH_PEN

        terminated = bool(abs(phi) > np.pi / 4)
        truncated  = self._steps >= self.MAX_STEPS

        obs = np.array([
            self.state[0], self.state[1],
            self.state[2], self.state[3],
            self.state[4], self.state[5],
            self.v, self.alpha,
            phi_d, err,
        ], dtype=np.float32)

        # ── diagnostics برای explode detection ──────────────────────────
        u_norm     = float(np.linalg.norm(u_opt))
        state_norm = float(np.linalg.norm(self.state))
        u_sat      = bool(np.any(np.abs(u_opt) >= TAU_MAX * 0.99))

        _EXPLODE_STATE = 50.0
        _EXPLODE_U     = TAU_MAX * 1.5
        if state_norm > _EXPLODE_STATE or u_norm > _EXPLODE_U:
            print(
                f"[EXPLODE] step={self._steps}  phi={np.rad2deg(phi):+.1f}°  "
                f"state_norm={state_norm:.2f}  u_norm={u_norm:.2f}  "
                f"curriculum={self.curriculum_level}"
            )

        info = {
            "phi":   float(phi),
            "phi_d": float(phi_d),
            "err":   float(err),
            "v":     float(self.v),
            "alpha": float(self.alpha),
            "curriculum_level": int(self.curriculum_level),
            # ── diagnostics ──────────────────────────────────────────────
            "u_norm":        u_norm,
            "state_norm":    state_norm,
            "u_sat":         u_sat,
            "solve_time_ms": self.mpc.last_solve_time * 1000.0,
            "solve_status":  self.mpc.last_solve_status,
            "reward_raw":    float(raw_reward),
            "reward_clipped": float(reward),
            # [v16] وزن‌های smoothed برای debug
            "mpc_weights_smoothed": self._smoothed_weights.tolist(),
            "reward_components": {
                "tracking":      float(tracking_reward),
                "stability":     float(stability_bonus),
                "phi_dot":       float(phi_dot_penalty),
                "control":       float(control_cost),
                "danger":        float(danger_penalty),
                "survival":      float(survival_bonus),
                "action_smooth": float(reward_action_smooth),
                "raw_total":     float(raw_reward),
                "clipped":       float(reward),
            },
        }

        return obs, reward, terminated, truncated, info
