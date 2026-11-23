"""
unified_fault_benchmark.py (v2) — benchmark مشترک: Neural MPC / Residual SAC / SAC Plain / SAC+HOCBF (/ Original MPC)

همه‌ی کنترلرها روی «یک» FaultEnv اجرا می‌شوند ⇒ دینامیک، IC، نویز v/alpha، مرجع roll، خرابی‌ها و متریک‌ها یکسان.
  nmpc          : residual = 0
  res:<stem>    : Residual SAC، obs ۱۲بعدی، u = clip(u_NN + res_scale·a)
  sac_plain_sK  : obs ۱۰بعدی اندازه‌گیری‌شده، u = a·TAU_MAX
  sac_hocbf_sK  : مثل بالا + HOCBF (alpha/lambda مثل baseline4)؛ پیش‌فرض روی state «اندازه‌گیری‌شده»
  sac_hocbf_oracle_sK : (--hocbf_state true) همان فیلتر ولی روی state «واقعی» ⇒ اثر خرابی حسگر روی فیلتر جدا می‌شود
  mpc           : Original MPC (اختیاری، کند)

تغییرات v2 نسبت به v1:
  1) گزارش اصلی روی «همه‌ی seedها»؛ تحلیل بدون early-failure فقط تکمیلی (فایل‌های *_excl_early_*).
  2) نام‌گذاری: early_failure_seeds (نه unrecoverable)؛ جدول می‌گوید همان seedها را کنترلرهای دیگر زنده می‌مانند یا نه.
  3) کنترل فرکانس: assert decision_k==1 ، ستون freq_ok (گام plant = گام فرمان)، مقایسه MAX_STEPS با baseline4.
  4) preflight مدل‌ها (وجود/جفت‌بودن .zip و .pkl، شکل obs، num_timesteps، training_summary) قبل از هر اجرا.
  5) HOCBF: تشخیص‌های جدا (slack، شکست QP، نقض شرط با state اندازه‌گیری‌شده/واقعی، خطای تقریب φ̈) + حالت oracle.
  6) آمار: bootstrap دوسطحی (مدل‌های آموزشی × اپیزودها)، مقایسه‌ی زوجی با همان seed اپیزود (bench_stats.py).

اجرا:
  python unified_fault_benchmark.py --ckpt neural_mpc.pt --res res_main1m_best --sac_root runs --preflight_only
  python unified_fault_benchmark.py ... --smoke --out ub_smoke
  python unified_fault_benchmark.py ... --episodes 100 --iw 1.0 --out ub_scn                      (۳۴ سناریو)
  python unified_fault_benchmark.py ... --episodes 100 --iw 1.0 0.6 0.5 0.45 0.42 0.4 0.35 0.3 --only healthy --out ub_iw
  موازی:  --shard i/n  (هر کدام فایل جدا) ؛ بعد  --aggregate --out ub_scn   (بدون نیاز به env)
"""
import argparse, glob, hashlib, json, os, pickle, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_stats as bs                                                    # noqa: E402

OK_STATUS = {"optimal", "optimal_inaccurate"}
EARLY = 10
SMOKE_SCN = ["healthy", "bias_roll_5deg", "noise_imu_3deg", "scale_gimbal_0.5", "act_stuck_zero", "act_sat_30pct",
             "param_iw_p50"]

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="neural_mpc.pt")
ap.add_argument("--res", nargs="*", default=[], help="stem[@res_scale] ... (stem بدون .zip)")
ap.add_argument("--sac_root", default="runs", help="ریشه‌ی جست‌وجوی (بازگشتی) مدل‌های SAC Plain/HOCBF")
ap.add_argument("--seeds_sac", nargs="*", type=int, default=[0, 1, 2, 3, 4])
ap.add_argument("--methods", nargs="*", default=["plain", "hocbf"], choices=["plain", "hocbf"])
ap.add_argument("--sac_steps", type=int, default=3_000_000)
ap.add_argument("--prefer", choices=["best", "final", "ckpt"], default="best")
ap.add_argument("--cbf_alpha1", type=float, default=2.0)
ap.add_argument("--cbf_alpha2", type=float, default=2.0)
ap.add_argument("--lambda_slack", type=float, default=1e4)
ap.add_argument("--hocbf_state", choices=["measured", "true"], default="measured")
ap.add_argument("--mpc", action="store_true")
ap.add_argument("--iw", nargs="*", type=float, default=[1.0])
ap.add_argument("--only", nargs="*", default=None, help="فقط این scenario idها")
ap.add_argument("--episodes", type=int, default=100)
ap.add_argument("--seed0", type=int, default=77000)
ap.add_argument("--bucket", type=int, default=0)
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--smoke", action="store_true", help="۵ اپیزود + سناریوهای نماینده")
ap.add_argument("--preflight_only", action="store_true")
ap.add_argument("--shard", default=None, help="i/n: فقط jobهای (iw,scenario) با index%%n==i")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--aggregate", action="store_true", help="فقط ادغام <out>*_episodes.csv و ساخت خلاصه‌ها")
ap.add_argument("--out", default="ub")
a = ap.parse_args()
if a.smoke:
    a.episodes = min(a.episodes, 5)
    a.only = a.only or SMOKE_SCN


# ═══════════ خلاصه‌ها (بدون env) ═══════════
def summarize(D, out, B):
    ref = D[(D.scenario == "healthy") & (D.ctrl == "nmpc") & (D.iw == 1.0)]
    early = sorted(int(s) for s in ref[ref.crashed & (ref.steps <= EARLY)].seed)
    D = D.copy(); D["early_failure"] = D.seed.isin(early)
    D.to_csv(out + "_all_episodes.csv", index=False)
    bs.analyze(D, out, "", B=B)                                         # گزارش اصلی: همه‌ی seedها
    if early:
        bs.analyze(D[~D.early_failure], out, "_excl_early", B=B, verbose=False)   # تکمیلی
    H = bs.hocbf_diag(D, out)
    if H is not None:
        pd.set_option("display.width", 250)
        print("\n===== HOCBF diagnostics (میانگین هر سناریو) =====\n", H.round(4).to_string())
    bad = D[(~D.inj_ok) | (~D.freq_ok)]
    print(f"\ninjection/frequency check ناموفق: {len(bad)} اپیزود" +
          (f" → {sorted(bad.groupby(['scenario', 'ctrl']).size().items())[:10]}" if len(bad) else " ✓"))
    print(f"\nearly_failure_seeds (nmpc، healthy، iw=1.0، سقوط ≤{EARLY} گام): {early}")
    if early:
        t = D[(D.scenario == "healthy") & (D.iw == 1.0) & D.seed.isin(early)].pivot_table(
            index="seed", columns="ctrl", values="crashed", aggfunc="first")
        print("سقوط (True) سایر کنترلرها روی همین seedها (این جدول اثبات نمی‌کند شروع «غیرقابل‌بازیابی» است):\n", t.to_string())
    print("گزارش اصلی = همه‌ی seedها؛ تحلیل بدون early-failure در فایل‌های *_excl_early_* (تکمیلی).")
    return early


def load_all(out):
    fs = [f for f in sorted(glob.glob(out + "*_episodes.csv")) if not f.endswith("_all_episodes.csv")]
    D = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(["iw", "scenario", "ctrl", "seed"])
    D["fam"] = D.ctrl.map(bs.fam_of)
    return D, fs


if a.aggregate:
    D, fs = load_all(a.out); print("ادغام:", fs, "→", len(D), "اپیزود")
    summarize(D, a.out, a.boot); raise SystemExit


# ═══════════ مدل‌ها + preflight ═══════════
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()[:16]


def find_sac(method, seed):
    tag = {"plain": "sacplain", "hocbf": "sachocbf"}[method]
    rt = f"{tag}_seed{seed}"
    g = lambda pat: sorted(glob.glob(os.path.join(a.sac_root, "**", pat), recursive=True))
    cand = {
    "best": (
                g(f"best_{rt}.zip"),
                g(f"vecnorm_best_{rt}.pkl")
            ),
            "final": (
                    g(f"final_{rt}.zip"),
                    g(f"vecnorm_final_{rt}.pkl")
                ),
                "ckpt": (
                    g(f"ckpt_{rt}_{a.sac_steps}_steps.zip"),
                    g(f"vecnorm_{rt}_{a.sac_steps}_steps.pkl")
                )
            }
    problems = []
    for k in [a.prefer]:
        z, p = cand[k]
        if len(z) == 1 and len(p) == 1:
            return z[0], p[0], k
        problems.append(f"{k}: zip={len(z)} pkl={len(p)}")
        if len(z) > 1 or len(p) > 1:
            raise RuntimeError(f"{rt}: چند فایل متناظر پیدا شد: {z + p}")
    raise FileNotFoundError(f"{rt}: جفت zip+pkl پیدا نشد ({'; '.join(problems)}) زیر {a.sac_root}. "
                            "best_model/best_model.zip عمداً استفاده نمی‌شود (vecnorm همان لحظه ندارد).")


class SACPolicy:
    """obs خام → VecNormalize ثابت (بدون آپدیت) → SAC قطعی."""
    def __init__(self, zip_path, vn_path, obs_dim):
        from stable_baselines3 import SAC
        self.model = SAC.load(zip_path, device="cpu")
        with open(vn_path, "rb") as f:
            vn = pickle.load(f)
        assert self.model.observation_space.shape == (obs_dim,), \
            f"{zip_path}: obs مدل {self.model.observation_space.shape} ≠ ({obs_dim},)"
        assert np.shape(vn.obs_rms.mean) == (obs_dim,), f"{vn_path}: شکل آمار {np.shape(vn.obs_rms.mean)}"
        assert getattr(vn, "norm_obs", True), f"{vn_path}: norm_obs=False؟"
        self.m, self.v = np.array(vn.obs_rms.mean, np.float64), np.array(vn.obs_rms.var, np.float64)
        self.eps, self.clip = float(vn.epsilon), float(vn.clip_obs)
        self.count = float(vn.obs_rms.count); self.ts = int(self.model.num_timesteps)
        self.zip, self.pkl = zip_path, vn_path

    def __call__(self, o):
        o = np.clip((np.asarray(o, np.float64) - self.m) / np.sqrt(self.v + self.eps), -self.clip, self.clip)
        return np.asarray(self.model.predict(o.astype(np.float32), deterministic=True)[0], np.float32)


def training_summary(zip_path):
    for d in (os.path.dirname(zip_path), os.path.dirname(os.path.dirname(zip_path))):
        p = os.path.join(d, "training_summary.json")
        if os.path.exists(p):
            return json.load(open(p))
    return None


def load_models():
    ctrls, files, warns = {}, {a.ckpt: sha(a.ckpt)}, []
    Z = np.zeros(2, np.float32)
    ctrls["nmpc"] = dict(kind="rl", pol=lambda o: Z, scale=5.0, fam="nmpc")
    for spec in a.res:
        stem, _, sc = spec.partition("@")
        pol = SACPolicy(stem + ".zip", stem + "_vecnorm.pkl", 12)
        ctrls[f"res:{os.path.basename(stem)}"] = dict(kind="rl", pol=pol, scale=float(sc) if sc else 5.0, fam="residual")
        files[pol.zip] = sha(pol.zip); files[pol.pkl] = sha(pol.pkl)
        print(f"[load] res:{os.path.basename(stem):20s} ts={pol.ts:,} vecnorm_count={pol.count:.3g} res_scale={ctrls[f'res:{os.path.basename(stem)}']['scale']}")
    for m in a.methods:
        for s in a.seeds_sac:
            z, p, kind = find_sac(m, s)
            pol = SACPolicy(z, p, 10)
            suffix = "_oracle" if (m == "hocbf" and a.hocbf_state == "true") else ""
            name = f"sac_{m}{suffix}_s{s}"
            ctrls[name] = dict(kind="sac", pol=pol, hocbf=(m == "hocbf"), fam=f"sac_{m}{suffix}", src=kind)
            files[z] = sha(z); files[p] = sha(p)
            ts = training_summary(z)
            msg = f"[load] {name:22s} {kind:5s} ts={pol.ts:,} vecnorm_count={pol.count:.3g} {os.path.basename(z)}"
            if kind != "best" and pol.ts != a.sac_steps:
                warns.append(f"{name}: num_timesteps={pol.ts:,} ≠ {a.sac_steps:,}")
            if pol.count < 0.5 * a.sac_steps:
                warns.append(f"{name}: vecnorm count={pol.count:.3g} خیلی کمتر از {a.sac_steps:,} ⇒ احتمالاً vecnorm نامتناظر")
            if ts is not None:
                msg += f" | summary: steps={ts.get('total_timesteps'):,} max_level={ts.get('reached_max_level')}"
                if not ts.get("reached_max_level", True):
                    warns.append(f"{name}: curriculum به سطح آخر نرسیده (final_level={ts.get('final_curriculum_level')})")
            print(msg, flush=True)
    if a.mpc:
        ctrls["mpc"] = dict(kind="mpc", fam="mpc")
    for w in warns:
        print("⚠ ", w)
    return ctrls, files, warns


ctrls, files, warns = load_models()
n_sac = sum(c["kind"] == "sac" for c in ctrls.values())
print(f"preflight: {len(ctrls)} کنترلر ({n_sac} مدل SAC baseline)، {len(warns)} هشدار")
if a.preflight_only:
    raise SystemExit(1 if warns else 0)

# Diagnostic wrapper: keeps the original filter policy and QP unchanged.
# Records branch, fresh solve status, fallback and slack for EACH call.
class InstrumentedHOCBF:
    def __init__(self, original):
        self.base = original
        self.last = {}

    def __getattr__(self, name):
        return getattr(self.base, name)

    def reset_stats(self):
        self.base.reset_stats()
        self.last = {}

    def filter(self, x, tau_rl):
        # Re-evaluate only the documented fast-path condition; never alter the action.
        lf, lg, psi1, _ = self.base._compute_lie(x)
        pre = float(lf + lg @ tau_rl + self.base.alpha2 * psi1)
        bypass = bool(pre >= 0 and psi1 >= 0)
        self.last = dict(qp_attempted=not bypass, qp_solved=False,
                         qp_failed=False, fallback=False, qp_status='not_attempted',
                         slack_value=np.nan, slack_used=False, bypass=bypass)
        # Prevent stale OSQP values/status being mistaken for this call.
        # Existing filter catches exceptions and falls back; no solver monkeypatching.
        # Replace only solve on this instance during one synchronous filter call.
        # This records whether OSQP was called and whether it returned or raised.
        qp = self.base._qp
        orig_solve = qp.solve
        outcome = {'called': False, 'raised': False, 'status': None, 'has_tau': False}
        def observed_solve(*args, **kwargs):
            outcome['called'] = True
            try:
                result = orig_solve(*args, **kwargs)
                outcome['status'] = qp.status
                outcome['has_tau'] = self.base._tau.value is not None
                return result
            except Exception:
                outcome['raised'] = True
                outcome['status'] = qp.status
                raise
        qp.solve = observed_solve
        try:
            tau = self.base.filter(x, tau_rl)
        finally:
            qp.solve = orig_solve
        if not bypass:
            valid = outcome['status'] in (self.base.cp.OPTIMAL, self.base.cp.OPTIMAL_INACCURATE)
            solved = bool(outcome['called'] and not outcome['raised'] and valid and outcome['has_tau'])
            self.last.update(qp_attempted=outcome['called'], qp_status=str(outcome['status']),
                             qp_solved=solved, qp_failed=not solved, fallback=not solved)
            if solved and self.base._slack.value is not None:
                sv = float(self.base._slack.value[0])
                self.last.update(slack_value=sv, slack_used=bool(sv > 1e-3))
        return tau


# ═══════════ env ═══════════
from final_fault_benchmark import SCENARIOS, FaultedPlant, measure                  # noqa: E402
from motorcycle_env_residual import MotorcycleEnvResidual                           # noqa: E402
from motorcycle_dynamics import TAU_MAX, MotorcycleDynamics, Ts                     # noqa: E402


class FaultEnv(MotorcycleEnvResidual):
    scn = ("none", {})

    def reset(self, seed=None, options=None):
        ft, fp = self.scn
        iw = float((options or {}).get("iw", 1.0))
        self._mc, self._mrng, self._last_meas_phi = {}, np.random.default_rng([int(seed), 41]), 0.0
        out = super().reset(seed=seed, options=options)           # _apply_iw(iw) داخل reset اجرا می‌شود
        if ft == "param_fault":
            p = dict(MotorcycleDynamics().p)
            p["m"] *= fp.get("m_scale", 1.0); p["h"] *= fp.get("h_scale", 1.0)
            dyn = MotorcycleDynamics(params=p, m_scale=1.0, Iw_scale=iw * fp.get("Iw_scale", 1.0))
        else:
            dyn = self._plant_dyn                                  # iw اعمال‌شده (iw=1 ⇒ self.dyn)
        self._plant_dyn = FaultedPlant(dyn, ft, fp, TAU_MAX)
        Qphi, Qphid, Rv, N = self.decode_action(np.zeros(4, np.float32))
        self._target = np.array([Qphi, Qphid, self._mid4[2], Rv]); self._N_fixed = int(N)
        self._last_ok_u = np.zeros(2)
        return out

    def _obs10(self):                                              # آنچه «کنترلر» می‌بیند (اندازه‌گیری خراب)
        k = self._steps
        if k not in self._mc:
            ft, fp = self.scn
            s = measure(self.state[:6], k, ft, fp, self._mrng, self._last_meas_phi)
            self._last_meas_phi = float(s[0]); phi_d = self._desired_roll()
            self._mc[k] = np.array([*s, self.v, self.alpha, phi_d, s[0] - phi_d], np.float32)
        return self._mc[k].copy()

    def _nn_u(self, x):
        t0 = time.perf_counter(); u = super()._nn_u(x); self.last_nn_ms = (time.perf_counter() - t0) * 1e3
        return u

    def mpc_control(self):
        t0 = time.perf_counter()
        x0 = self._obs10()[:6].astype(float); phi_d0 = self._desired_roll()
        self._smoothed_weights = (1 - self.LP_ALPHA) * self._smoothed_weights + self.LP_ALPHA * self._target
        w = self._smoothed_weights
        Qd = np.array([w[0], w[1], w[2], w[2], w[2] * 0.1, w[2] * 0.1]); Rd = np.array([w[3], w[3]])
        N = self._N_fixed
        if N != self.mpc.N:
            up = self.mpc.u_prev.value.copy(); self.mpc = self._solvers[N]
            self.mpc.reset_warm_start(); self.mpc.u_prev.value = up
        xd = np.zeros(6); xd[0] = phi_d0
        A, B, bias = self.dyn.linearise(x0[0]); self.mpc.update_model(A, B, bias)
        Pc = self._dare_chol(A, B, Q_diag=Qd, R_diag=Rd)
        u = np.clip(self.mpc.solve(x0=x0, xd=xd, Q_diag=Qd, R_diag=Rd, P_chol=Pc), -TAU_MAX, TAU_MAX)
        return u, str(self.mpc.last_solve_status), (time.perf_counter() - t0) * 1e3

    def raw_step(self, u):          # دقیقاً یک گام plant با یک فرمان؛ همان نویز v/alpha مثل step() با decision_k=1
        self._steps += 1
        self.state = self._plant_dyn.step(self.state, self._fault_eff * u)
        self.v = float(np.clip(self.v + self.np_random.normal(0.0, self.V_NOISE), 1.0, 8.0))
        self.alpha = float(np.clip(self.alpha + self.np_random.normal(0.0, self.ALPHA_NOISE), -0.20, 0.20))
        c = abs(self.state[0]) > np.pi / 4
        return c, c or self._steps >= self.MAX_STEPS


env = FaultEnv(ckpt=a.ckpt, iw_list=(1.0,))

# ── (3) کنترل فرکانس/افق: همه‌ی کنترلرها = یک فرمان به‌ازای هر گام plant (Ts) ──
assert env.decision_k == 1, f"decision_k={env.decision_k}: Residual با k>1 با SAC baseline (k=1) قابل‌مقایسه نیست"
print(f"[freq] decision_k={env.decision_k}  Ts={Ts}  MAX_STEPS(env)={env.MAX_STEPS}")
if n_sac:
    import baseline4_sac_cbf_train as b4
    if int(b4.MAX_STEPS) != int(env.MAX_STEPS):
        print(f"⚠  MAX_STEPS: env={env.MAX_STEPS} ولی baseline4={b4.MAX_STEPS}؛ SAC ها روی افق متفاوتی آموزش دیده‌اند")
    hocbf = None
    if any(c.get("hocbf") for c in ctrls.values()):
        hocbf = InstrumentedHOCBF(b4.HOCBFSafetyFilter(env.dyn, alpha1=a.cbf_alpha1, alpha2=a.cbf_alpha2, lambda_slack=a.lambda_slack))


def check_injection(ft, fp, Uc, Ua):
    """گشتاور اعمال‌شده = تبدیل خرابی روی فرمان (برای هر نوع کنترلر)."""
    if len(Uc) < 3:
        return True
    k0 = fp.get("fault_start", 0)
    try:
        if ft == "none":
            return bool(np.abs(Ua - Uc).max() < 1e-9)
        if ft == "scale_actuator":
            return bool(np.abs(Ua - np.clip(fp["gain"] * Uc, -TAU_MAX, TAU_MAX)).max() < 1e-9)
        if ft == "actuator_stuck":
            return bool(len(Ua) <= k0 or np.allclose(Ua[k0:], np.clip(fp["stuck_val"], -TAU_MAX, TAU_MAX)))
        if ft == "actuator_saturation":
            return bool(np.abs(Ua).max() <= TAU_MAX * fp["max_frac"] + 1e-9)
    except Exception:
        return False
    return True


def cbf_terms(x):
    """Lf_ψ₁، Lg_ψ₁، ψ₁ و پیش‌بینی φ̈ مدل فیلتر (f_phi + g_phi·τ) در state x."""
    Lf, Lg, psi1, psi0 = hocbf._compute_lie(x)
    phi = x[0]
    f_phi = (hocbf.mgh * np.sin(phi) - hocbf.IwO * (x[4] - x[5]) * np.cos(phi)) / hocbf.Iphi
    k = hocbf.IwO * np.cos(phi) * Ts / (hocbf.Iphi * hocbf.Id)
    return Lf, Lg, psi1, f_phi, np.array([-k, k])


def run(name, seed, iw):
    c = ctrls[name]
    ft, fp = env.scn
    env.reset(seed=seed, options=dict(iw=iw, bucket=a.bucket))
    use_cbf = c["kind"] == "sac" and c["hocbf"]
    if use_cbf:
        hocbf.reset_stats()
    obs12 = np.concatenate([env._obs10(), (env._u_nn / env.U_REF).astype(np.float32)])
    Ua, Uc, SAT, phi, cms, e2, n = [], [], [], [], [], 0.0, 0
    D = dict(cond_meas=[], cond_true=[], psi1_true=[], dd_err=[], qp_fail=[], qp_attempt=[], qp_success=[], qp_fallback=[], qp_bypass=[], slack=[], slack_used=[], cond_meas_margin=[], cond_true_margin=[])
    while True:
        if c["kind"] == "rl":                                   # nmpc / residual (env.step، decision_k=1)
            env.res_scale = c["scale"]
            t0 = time.perf_counter(); act = c["pol"](obs12); pm = (time.perf_counter() - t0) * 1e3
            obs12, r, crashed, tr, info = env.step(act); done = crashed or tr
            cms.append(pm + getattr(env, "last_nn_ms", 0.0))
            e2 += info["err_sq_sum"]; n += info["n_plant_steps"]; phi.append(info["phi"])
        else:
            x_true = env.state[:6].astype(np.float64).copy()
            if c["kind"] == "sac":
                x10 = env._obs10()
                t0 = time.perf_counter()
                tau = np.asarray(c["pol"](x10), np.float64).clip(-1.0, 1.0) * TAU_MAX
                if use_cbf:
                    x_f = x_true if a.hocbf_state == "true" else x10[:6].astype(np.float64)
                    tau = np.asarray(hocbf.filter(x_f, tau), np.float64)
                    z = hocbf.last
                    for col, key in (("qp_fail", "qp_failed"), ("qp_attempt", "qp_attempted"),
                                     ("qp_success", "qp_solved"), ("qp_fallback", "fallback"),
                                     ("qp_bypass", "bypass"), ("slack_used", "slack_used")):
                        D[col].append(float(z[key]))
                    D["slack"].append(z["slack_value"])
                    Lf, Lg, psi1, _, _ = cbf_terms(x_f)
                    margin = float(Lf + Lg @ tau + hocbf.alpha2 * psi1)
                    D["cond_meas_margin"].append(margin)
                    D["cond_meas"].append(float(margin < -1e-6))
                cms.append((time.perf_counter() - t0) * 1e3)
                u = tau
            else:                                               # mpc
                u, st, ms = env.mpc_control(); cms.append(ms)
            phidot0 = float(env.state[1])
            crashed, done = env.raw_step(u)
            p = float(env.state[0]); e2 += (p - env._desired_roll()) ** 2; n += 1; phi.append(p)
            if use_cbf:                                         # تشخیص با state واقعی و گشتاور «اعمال‌شده»
                ua = env._plant_dyn.last_app
                Lf_t, Lg_t, psi1_t, f_t, g_t = cbf_terms(x_true)
                margin_true = float(Lf_t + Lg_t @ ua + hocbf.alpha2 * psi1_t)
                D["cond_true_margin"].append(margin_true)
                D["cond_true"].append(float(margin_true < -1e-6))
                D["psi1_true"].append(float(psi1_t < 0))
                D["dd_err"].append((f_t + g_t @ ua) - (float(env.state[1]) - phidot0) / Ts)
        pl = env._plant_dyn
        Ua.append(pl.last_app.copy()); Uc.append(pl.last_cmd.copy())
        SAT.append(bool(np.any(np.abs(pl.last_cmd) >= 0.99 * pl.last_limit)))
        if done:
            break
    Ua, Uc, phi, cms = np.array(Ua), np.array(Uc), np.array(phi), np.array(cms)
    dd = lambda U: float(np.linalg.norm(np.diff(U, axis=0), axis=1).mean()) if len(U) > 1 else 0.0
    row = dict(crashed=bool(crashed), steps=len(phi), rmse=float(np.sqrt(e2 / max(n, 1))),
               peak_phi=float(np.rad2deg(np.abs(phi).max())),
               u_rms=float(np.sqrt(np.mean(Ua ** 2))), u2_sum=float(np.sum(Ua ** 2)), du=dd(Ua),
               u_rms_cmd=float(np.sqrt(np.mean(Uc ** 2))), du_cmd=dd(Uc), sat=float(np.mean(SAT)),
               ms=float(cms.mean()), inj_ok=check_injection(ft, fp, Uc, Ua),
               freq_ok=bool(pl.k == len(phi) == env._steps))
    if use_cbf:
        dde = np.array(D["dd_err"])
        row.update(cbf_filter_rate=hocbf.filter_rate, cbf_slack_rate=hocbf.slack_rate,
                   cbf_qp_fail_rate=float(np.mean(D["qp_fail"])), cbf_cond_viol_meas=float(np.mean(D["cond_meas"])),
                   cbf_cond_viol_true=float(np.mean(D["cond_true"])), cbf_psi1_viol_true=float(np.mean(D["psi1_true"])),
                   cbf_phiddot_rmse=float(np.sqrt(np.mean(dde ** 2))), cbf_phiddot_bias=float(dde.mean()),
                   cbf_qp_attempt_rate=float(np.mean(D["qp_attempt"])),
                   cbf_qp_success_rate=float(np.mean(D["qp_success"])),
                   cbf_fallback_rate=float(np.mean(D["qp_fallback"])),
                   cbf_bypass_rate=float(np.mean(D["qp_bypass"])),
                   cbf_slack_mean_on_solve=float(np.nanmean(D["slack"])) if np.isfinite(D["slack"]).any() else np.nan,
                   cbf_slack_max=float(np.nanmax(D["slack"])) if np.isfinite(D["slack"]).any() else np.nan,
                   cbf_cond_margin_meas_mean=float(np.mean(D["cond_meas_margin"])),
                   cbf_cond_margin_true_mean=float(np.mean(D["cond_true_margin"])),
                   cbf_cond_margin_true_min=float(np.min(D["cond_true_margin"])))
        assert abs(row["cbf_qp_attempt_rate"] + row["cbf_bypass_rate"] - 1) < 1e-10
        assert abs(row["cbf_qp_success_rate"] + row["cbf_qp_fail_rate"] - row["cbf_qp_attempt_rate"]) < 1e-10
    return row


# ═══════════ اجرا ═══════════
scn = [s for s in SCENARIOS if a.only is None or s[0] in a.only]
jobs = [(iw, s) for iw in a.iw for s in scn]
if a.shard:
    i_, n_ = map(int, a.shard.split("/")); jobs = [j for k, j in enumerate(jobs) if k % n_ == i_]
out_csv = a.out + (f"_shard{a.shard.split('/')[0]}" if a.shard else "") + "_episodes.csv"
rows, done_keys = [], set()
if a.resume and os.path.exists(out_csv):
    rows = pd.read_csv(out_csv).to_dict("records")
    done_keys = {(r["iw"], r["scenario"], r["ctrl"], r["seed"]) for r in rows}
    print(f"resume: {len(rows)} اپیزود موجود")

seeds = [a.seed0 + i for i in range(a.episodes)]
# مرجع early-failure: nmpc ، healthy ، iw=1.0 (در هر shard محاسبه می‌شود تا مجموعه یکسان باشد)
env.scn = ("none", {})
for s in seeds:
    if (1.0, "healthy", "nmpc", s) not in done_keys:
        rows.append(dict(iw=1.0, scenario="healthy", type="none", ctrl="nmpc", fam="nmpc", seed=s, **run("nmpc", s, 1.0)))
        done_keys.add((1.0, "healthy", "nmpc", s))

t0 = time.time()
for iw, (sid, ft, fp) in jobs:
    env.scn = (ft, fp)
    for name in ctrls:
        for s in [s for s in seeds if (iw, sid, name, s) not in done_keys]:
            rows.append(dict(iw=iw, scenario=sid, type=ft, ctrl=name, fam=ctrls[name]["fam"], seed=s, **run(name, s, iw)))
            done_keys.add((iw, sid, name, s))
        d = pd.DataFrame([r for r in rows if r["iw"] == iw and r["scenario"] == sid and r["ctrl"] == name])
        print(f"iw={iw:.2f} {sid:26s} {name:22s} surv={1 - d.crashed.mean():.2f} rmse={d.rmse.mean():.3f} "
              f"inj={'ok' if d.inj_ok.all() else 'FAIL'} freq={'ok' if d.freq_ok.all() else 'FAIL'} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
D = pd.DataFrame(rows); D.to_csv(out_csv, index=False)
D["fam"] = D.ctrl.map(bs.fam_of)
if a.shard:
    print("shard تمام شد؛ بعد از همه‌ی shardها:  --aggregate --out", a.out)
    early = []
else:
    early = summarize(D, a.out, a.boot)
json.dump(dict(args=vars(a), file_sha256=files, early_failure_seeds=early, warnings=warns,
               sac_sources={k: v.get("src") for k, v in ctrls.items() if v["kind"] == "sac"},
               hocbf_state=a.hocbf_state, decision_k=env.decision_k, max_steps=env.MAX_STEPS),
          open(a.out + (f"_shard{a.shard.split('/')[0]}" if a.shard else "") + "_config.json", "w"), indent=1)
