"""
final_fault_benchmark.py — benchmark نهایی روی سناریوهای خرابی (همان جدول eval_ftc_all_methods_v2.py)
برای Neural MPC / Residual SAC / (اختیاری) Original MPC، بدون آموزش مجدد.

قواعد تزریق (برای همه‌ی کنترلرها یکسان):
  sensor   : فقط «اندازه‌گیری» کنترلر خراب می‌شود (φ و φ̇؛ err از φ خراب حساب می‌شود)، plant از state واقعی می‌رود.
             اندازه‌گیری خراب تابعِ «شماره‌ی گام» است و یک بار در هر گام ساخته می‌شود ⇒ نویز/dropout
             برای همه‌ی کنترلرها روی یک seed دقیقاً یکسان است.
  actuator : روی «گشتاور نهاییِ اعمال‌شده به plant» (u_NN + residual) اعمال می‌شود، نه فقط خروجی residual.
  param    : فقط plant واقعی عوض می‌شود؛ Neural MPC و مدل MPC نامی ماندند.
  همه‌ی متریک‌ها از سیگنال‌های واقعی plant حساب می‌شوند.

    python final_fault_benchmark.py --ckpt neural_mpc.pt --sac res_main1m_best res_pilot5_best --episodes 5 --out ff_smoke
    python final_fault_benchmark.py --ckpt neural_mpc.pt --sac res_main1m_best res_pilot5_best --episodes 100 --out ff_final
    (--mpc برای Original MPC؛ کند)
"""
import argparse, hashlib, json, os, pickle, time
import numpy as np

# ═══════════ BEGIN PURE (بدون وابستگی؛ قابل تست مستقل) ═══════════
D2R = np.deg2rad
SCENARIOS = [  # (id, type, params)  — ۳۲ خرابی + healthy، به‌علاوه‌ی ۱ سناریوی تفاضلی (extra)
    ("healthy", "none", {}),
    ("bias_roll_2deg", "bias_phi", {"bias": D2R(2)}), ("bias_roll_5deg", "bias_phi", {"bias": D2R(5)}),
    ("bias_roll_10deg", "bias_phi", {"bias": D2R(10)}),
    ("noise_imu_1deg", "noise_phi", {"sigma": D2R(1)}), ("noise_imu_3deg", "noise_phi", {"sigma": D2R(3)}),
    ("noise_imu_7deg", "noise_phi", {"sigma": D2R(7)}),
    ("drift_phi_0.5dps", "drift_phi", {"rate": D2R(0.5)}), ("drift_phi_2.0dps", "drift_phi", {"rate": D2R(2.0)}),
    ("stuck_phi_0deg", "stuck_phi", {"stuck_val": 0.0, "fault_start": 100}),
    ("stuck_phi_5deg", "stuck_phi", {"stuck_val": D2R(5), "fault_start": 100}),
    ("stuck_phi_15deg", "stuck_phi", {"stuck_val": D2R(15), "fault_start": 100}),
    ("dropout_imu_p0.1", "dropout_phi", {"p": 0.1}), ("dropout_imu_p0.3", "dropout_phi", {"p": 0.3}),
    ("dropout_imu_p0.5", "dropout_phi", {"p": 0.5}),
    ("intermittent_imu_burst5", "intermittent_phi", {"burst": 5, "period": 20}),
    ("intermittent_imu_burst20", "intermittent_phi", {"burst": 20, "period": 50}),
    ("multifault_noise5_bias3", "multifault", {"sigma": D2R(5), "bias": D2R(3)}),
    ("scale_gimbal_0.5", "scale_actuator", {"gain": 0.5}), ("scale_gimbal_0.7", "scale_actuator", {"gain": 0.7}),
    ("scale_gimbal_1.3", "scale_actuator", {"gain": 1.3}),
    ("act_bias_pos", "actuator_bias", {"bias": 0.3, "fault_start": 100}),
    ("act_bias_neg", "actuator_bias", {"bias": -0.5, "fault_start": 100}),
    ("act_sat_50pct", "actuator_saturation", {"max_frac": 0.5}),
    ("act_sat_30pct", "actuator_saturation", {"max_frac": 0.3}),
    ("act_stuck_zero", "actuator_stuck", {"stuck_val": 0.0, "fault_start": 100}),
    ("act_stuck_15Nm", "actuator_stuck", {"stuck_val": 15.0, "fault_start": 100}),
    ("act_delay_3", "actuator_delay", {"delay": 3}), ("act_delay_8", "actuator_delay", {"delay": 8}),
    # نام‌ها اصلاح شد: m_scale هم mgh و هم Iφ=m·h² را ضرب می‌کند ⇒ g/h ثابت، فقط «اختیار کنترل» ∝ 1/m_scale
    ("param_mass_p30", "param_fault", {"m_scale": 1.30}), ("param_mass_m20", "param_fault", {"m_scale": 0.80}),
    ("param_mass_p20_h10", "param_fault", {"m_scale": 1.20, "h_scale": 1.10}),
    ("param_iw_p50", "param_fault", {"Iw_scale": 1.50}),
    ("EXTRA_act_bias_diff_0.5", "actuator_bias_diff", {"bias": 0.5, "fault_start": 100}),
]
SENSOR = {"bias_phi", "noise_phi", "drift_phi", "stuck_phi", "dropout_phi", "intermittent_phi", "multifault"}


def measure(s, k, ft, fp, rng, last_phi):
    """s: state واقعی (6,) → state اندازه‌گیری‌شده؛ فقط خرابی sensor."""
    s = np.array(s, float)
    if ft not in SENSOR or k < fp.get("fault_start", 0):
        return s
    if ft == "bias_phi":
        s[0] += fp["bias"]
    elif ft == "noise_phi":
        s[0] += rng.normal(0, fp["sigma"]); s[1] += rng.normal(0, fp["sigma"])
    elif ft == "drift_phi":
        s[0] += fp["rate"] * (k - fp.get("fault_start", 0)) * 0.01
    elif ft == "stuck_phi":
        s[0] = fp["stuck_val"]; s[1] = 0.0
    elif ft == "dropout_phi":
        if rng.random() < fp["p"]:
            s[0] = last_phi; s[1] = 0.0
    elif ft == "intermittent_phi":
        if k % fp.get("period", 20) < fp.get("burst", 5):
            s[0] = last_phi; s[1] = 0.0
    elif ft == "multifault":
        s[0] += fp.get("bias", 0.0) + rng.normal(0, fp.get("sigma", 0.0))
        s[1] += rng.normal(0, fp.get("sigma", 0.0) * 0.5)
    return s


class FaultedPlant:
    """پوشش dynamics واقعی: خرابی actuator روی u نهایی؛ سایر متدها (linearise/p/...) به dyn واقعی می‌روند.
    last_cmd = گشتاور درخواستی کنترلر ، last_app = گشتاور واقعاً اعمال‌شده به plant (بعد از خرابی و clip)،
    last_limit = سقف مؤثر گشتاور در همین گام."""
    def __init__(self, dyn, ft, fp, tau_max=50.0):
        self.dyn, self.ft, self.fp, self.k, self.buf, self.tau = dyn, ft, fp, 0, [], tau_max
        self.last_cmd = np.zeros(2); self.last_app = np.zeros(2); self.last_limit = tau_max

    def _apply(self, u):
        fp, ft = self.fp, self.ft
        if self.k < fp.get("fault_start", 0):
            return u
        if ft == "scale_actuator":
            return u * fp["gain"]
        if ft == "actuator_bias":
            return u + fp["bias"]
        if ft == "actuator_bias_diff":
            return u + np.array([fp["bias"], -fp["bias"]])
        if ft == "actuator_saturation":
            m = self.tau * fp["max_frac"]; return np.clip(u, -m, m)
        if ft == "actuator_stuck":
            return np.full_like(u, fp["stuck_val"])
        if ft == "actuator_delay":
            self.buf.append(u.copy())
            return self.buf.pop(0) if len(self.buf) > int(fp["delay"]) else np.zeros_like(u)
        return u

    def step(self, x, u):
        u = np.asarray(u, float); self.last_cmd = u.copy()
        active = self.k >= self.fp.get("fault_start", 0)
        self.last_limit = self.tau * self.fp["max_frac"] if (self.ft == "actuator_saturation" and active) else self.tau
        ua = self._apply(u)
        self.last_app = np.clip(ua, -self.tau, self.tau)
        self.k += 1
        return self.dyn.step(x, ua)

    def __getattr__(self, n):
        return getattr(self.dyn, n)
# ═══════════ END PURE ═══════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="neural_mpc.pt")
    ap.add_argument("--sac", nargs="*", default=[], help="stem[@res_scale] ...")
    ap.add_argument("--mpc", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="فقط این scenario idها")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=77000)
    ap.add_argument("--bucket", type=int, default=0)
    ap.add_argument("--selftest", action="store_true", help="آزمون‌های مستقل تزریق خرابی؛ بعد خارج می‌شود")
    ap.add_argument("--out", default="ff")
    a = ap.parse_args()

    from motorcycle_env_residual import MotorcycleEnvResidual
    from motorcycle_dynamics import TAU_MAX, MotorcycleDynamics

    OK_STATUS = {"optimal", "optimal_inaccurate"}

    class FaultEnv(MotorcycleEnvResidual):
        scn = ("none", {})

        def reset(self, seed=None, options=None):
            ft, fp = self.scn
            self._mc, self._mrng, self._last_meas_phi = {}, np.random.default_rng([int(seed), 41]), 0.0
            out = super().reset(seed=seed, options=options)
            dyn = self.dyn
            if ft == "param_fault":                                   # فقط plant واقعی
                p = dict(MotorcycleDynamics().p)
                p["m"] *= fp.get("m_scale", 1.0); p["h"] *= fp.get("h_scale", 1.0)
                dyn = MotorcycleDynamics(params=p, m_scale=1.0, Iw_scale=fp.get("Iw_scale", 1.0))
            self._plant_dyn = FaultedPlant(dyn, ft, fp, TAU_MAX)
            self._tm = 0.0
            Qphi, Qphid, Rv, N = self.decode_action(np.zeros(4, np.float32))
            self._target = np.array([Qphi, Qphid, self._mid4[2], Rv]); self._N_fixed = int(N)
            self._last_ok_u = np.zeros(2)
            return out

        def _obs10(self):                                               # آنچه «کنترلر» می‌بیند
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
            x0 = self._obs10()[:6].astype(float)                        # measured state
            phi_d0 = self._desired_roll()
            self._smoothed_weights = (1 - self.LP_ALPHA) * self._smoothed_weights + self.LP_ALPHA * self._target
            w = self._smoothed_weights
            Qd = np.array([w[0], w[1], w[2], w[2], w[2] * 0.1, w[2] * 0.1]); Rd = np.array([w[3], w[3]])
            N = self._N_fixed
            if N != self.mpc.N:
                up = self.mpc.u_prev.value.copy(); self.mpc = self._solvers[N]
                self.mpc.reset_warm_start(); self.mpc.u_prev.value = up
            xd = np.zeros(6); xd[0] = phi_d0
            A, B, bias = self.dyn.linearise(x0[0])
            self.mpc.update_model(A, B, bias)
            Pc = self._dare_chol(A, B, Q_diag=Qd, R_diag=Rd)
            u = np.clip(self.mpc.solve(x0=x0, xd=xd, Q_diag=Qd, R_diag=Rd, P_chol=Pc), -TAU_MAX, TAU_MAX)
            return u, str(self.mpc.last_solve_status), (time.perf_counter() - t0) * 1e3

        def raw_step(self, u):
            self._steps += 1
            self.state = self._plant_dyn.step(self.state, self._fault_eff * u)
            self.v = float(np.clip(self.v + self.np_random.normal(0.0, self.V_NOISE), 1.0, 8.0))
            self.alpha = float(np.clip(self.alpha + self.np_random.normal(0.0, self.ALPHA_NOISE), -0.20, 0.20))
            c = abs(self.state[0]) > np.pi / 4
            return c, c or self._steps >= self.MAX_STEPS

    class SACPolicy:
        def __init__(self, stem):
            from stable_baselines3 import SAC
            self.model = SAC.load(stem + ".zip", device="cpu")
            with open(stem + "_vecnorm.pkl", "rb") as f:
                vn = pickle.load(f)
            self.m, self.v = np.array(vn.obs_rms.mean, np.float64), np.array(vn.obs_rms.var, np.float64)
            self.eps, self.clip = float(vn.epsilon), float(vn.clip_obs)

        def __call__(self, o):
            o = np.clip((o - self.m) / np.sqrt(self.v + self.eps), -self.clip, self.clip).astype(np.float32)
            return np.asarray(self.model.predict(o, deterministic=True)[0], np.float32)

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for ch in iter(lambda: f.read(1 << 20), b""):
                h.update(ch)
        return h.hexdigest()[:16]

    env = FaultEnv(ckpt=a.ckpt, iw_list=(1.0,))

    def selftest():
        res = []
        def chk(name, ok, msg=""):
            res.append(bool(ok)); print(f"[{'PASS' if ok else 'FAIL'}] {name} {msg}", flush=True)
        Zs = np.zeros(2, np.float32)

        def roll(sid, need=120, T=220, act=Zs, scale=5.0, seeds=None):
            ft, fp = next((f, q) for i, f, q in SCENARIOS if i == sid); env.scn = (ft, fp); env.res_scale = scale
            for sd in (seeds or range(a.seed0, a.seed0 + 8)):
                env.reset(seed=sd, options=dict(iw=1.0, bucket=0))
                R = dict(cmd=[], app=[], meas=[], true=[], uf=[], info_ok=True)
                for k in range(T):
                    R["meas"].append(env._obs10().copy()); R["true"].append(env.state[:6].copy())
                    _, _, te, tr, info = env.step(act)
                    pl = env._plant_dyn
                    R["cmd"].append(pl.last_cmd.copy()); R["app"].append(pl.last_app.copy()); R["uf"].append(info["u_final"].copy())
                    R["info_ok"] &= abs(info["phi"] - env.state[0]) < 1e-12
                    if te or tr:
                        break
                R = {k: (np.array(v) if k != "info_ok" else v) for k, v in R.items()}
                if len(R["cmd"]) >= need:
                    break
            env.res_scale = 5.0
            return R

        R = roll("healthy")
        chk("S1 healthy: applied == commanded == info['u_final']", np.abs(R["app"] - R["cmd"]).max() < 1e-12
            and np.abs(R["uf"] - R["cmd"]).max() < 1e-9, f"(steps={len(R['cmd'])})")
        R = roll("act_stuck_zero")
        k0 = 100
        chk("S2 act_stuck_zero: بعد از گام 100 گشتاور اعمال‌شده صفر، قبلش برابر فرمان",
            len(R["app"]) > k0 + 3 and np.abs(R["app"][k0:]).max() == 0 and np.abs(R["app"][:k0] - R["cmd"][:k0]).max() < 1e-12,
            f"(steps={len(R['app'])})")
        R = roll("act_stuck_15Nm")
        chk("S3 act_stuck_15Nm: اعمال‌شده = 15 بعد از گام 100", len(R["app"]) > k0 + 3 and np.allclose(R["app"][k0:], 15.0))
        R = roll("scale_gimbal_0.5")
        chk("S4 scale_gimbal_0.5: اعمال‌شده = 0.5·فرمان", np.abs(R["app"] - 0.5 * R["cmd"]).max() < 1e-9)
        R = roll("act_sat_30pct", need=8, T=30, act=np.array([1.0, -1.0], np.float32), scale=100.0)
        chk("S5 act_sat_30pct: |اعمال‌شده| ≤ 15 حتی با فرمان ±50", np.abs(R["app"]).max() <= 15 + 1e-9 and np.abs(R["cmd"]).max() > 15,
            f"(max cmd={np.abs(R['cmd']).max():.1f}, max app={np.abs(R['app']).max():.1f})")
        R = roll("act_delay_3")
        c, ap_ = R["cmd"], R["app"]
        chk("S6 act_delay_3: app[k] = cmd[k−3]", len(c) > 10 and np.abs(ap_[:3]).max() == 0 and np.abs(ap_[3:] - c[:-3]).max() < 1e-12)
        R = roll("act_bias_diff_0.5".replace("act", "EXTRA_act"))
        d = R["app"][k0:] - R["cmd"][k0:]
        chk("S7 bias تفاضلی: app − cmd = [+0.5, −0.5] بعد از گام 100", len(R["app"]) > k0 + 3 and np.allclose(d, [0.5, -0.5]))
        R = roll("act_bias_pos"); d = R["app"][k0:] - R["cmd"][k0:]
        chk("S8 act_bias_pos: app − cmd = +0.3 (روی هر دو actuator)", len(R["app"]) > k0 + 3 and np.allclose(d, 0.3))
        R = roll("bias_roll_5deg")
        chk("S9 sensor bias: φ_meas − φ_true = 5°، و info['phi'] همان state واقعی است",
            np.allclose(R["meas"][:, 0] - R["true"][:, 0], D2R(5)) and R["info_ok"])
        R = roll("stuck_phi_5deg")
        chk("S10 sensor stuck: بعد از گام 100 φ_meas = 5° و φ̇_meas = 0 ؛ قبلش برابر واقعی",
            len(R["meas"]) > k0 + 3 and np.allclose(R["meas"][k0:, 0], D2R(5)) and np.abs(R["meas"][k0:, 1]).max() == 0
            and np.abs(R["meas"][:k0, 0] - R["true"][:k0, 0]).max() < 1e-6)
        errs = []
        for sd in range(a.seed0, a.seed0 + 4):
            R = roll("noise_imu_3deg", need=1, T=150, seeds=[sd]); errs.append(R["meas"][:, 0] - R["true"][:, 0])
        errs = np.concatenate(errs)
        chk("S11 noise_imu_3deg: std(φ_meas − φ_true) ≈ 3°", 0.6 * D2R(3) < errs.std() < 1.4 * D2R(3),
            f"(std={np.rad2deg(errs.std()):.2f}°، n={len(errs)})")
        R1 = roll("dropout_imu_p0.3", need=1, T=60, seeds=[a.seed0]); R2 = roll("dropout_imu_p0.3", need=1, T=60, seeds=[a.seed0])
        chk("S12 تعیین‌پذیری اندازه‌گیری خراب (دو اجرا با یک seed)", np.array_equal(R1["meas"], R2["meas"]))
        env.scn = ("param_fault", {"m_scale": 1.30}); env.reset(seed=a.seed0, options=dict(iw=1.0, bucket=0))
        nom = MotorcycleDynamics()
        chk("S13 param_mass_p30: plant واقعی Iφ ×1.30 و مدل کنترلر دست‌نخورده",
            abs(env._plant_dyn.dyn.Iphi / nom.Iphi - 1.30) < 1e-9 and abs(env.dyn.Iphi - nom.Iphi) < 1e-12)
        env.scn = ("param_fault", {"Iw_scale": 1.5}); env.reset(seed=a.seed0, options=dict(iw=1.0, bucket=0))
        chk("S14 param_iw_p50: plant واقعی IwΩ ×1.5 و مدل کنترلر دست‌نخورده",
            abs(env._plant_dyn.dyn.IwO / nom.IwO - 1.5) < 1e-9 and abs(env.dyn.IwO - nom.IwO) < 1e-12)
        base = MotorcycleEnvResidual(ckpt=a.ckpt, iw_list=(1.0,))
        env.scn = ("none", {}); env.reset(seed=a.seed0, options=dict(iw=1.0, bucket=0))
        base.reset(seed=a.seed0, options=dict(iw=1.0, bucket=0)); worst = 0.0
        for _ in range(120):
            o1 = env.step(Zs); o2 = base.step(Zs); worst = max(worst, abs(o1[4]["phi"] - o2[4]["phi"]))
            if o1[2] or o2[2]:
                break
        chk("S15 healthy در FaultEnv ≡ MotorcycleEnvResidual ساده", worst < 1e-9, f"(max|Δφ|={worst:.1e})")
        print(f"\nSELFTEST: {sum(res)}/{len(res)} PASS")
        raise SystemExit(0 if all(res) else 1)

    if a.selftest:
        selftest()
    Z = np.zeros(2, np.float32)
    ctrls, files = {"nmpc": dict(kind="rl", pol=lambda o: Z, scale=5.0)}, {a.ckpt: sha(a.ckpt)}
    for spec in a.sac:
        stem, _, sc = spec.partition("@")
        ctrls[stem] = dict(kind="rl", pol=SACPolicy(stem), scale=float(sc) if sc else 5.0)
        files[stem + ".zip"] = sha(stem + ".zip"); files[stem + "_vecnorm.pkl"] = sha(stem + "_vecnorm.pkl")
    if a.mpc:
        ctrls["mpc"] = dict(kind="mpc")

    def run(name, seed):
        c = ctrls[name]
        env.reset(seed=seed, options=dict(iw=1.0, bucket=a.bucket))
        Ua, Uc, SAT, phi, cms, e2, n = [], [], [], [], [], 0.0, 0
        obs = np.concatenate([env._obs10(), (env._u_nn / env.U_REF).astype(np.float32)])
        while True:
            if c["kind"] == "mpc":
                u, st, ms = env.mpc_control(); cms.append(ms)
                if st in OK_STATUS:
                    env._last_ok_u = u.copy()
                crashed, done = env.raw_step(u)
                p = float(env.state[0]); e2 += (p - env._desired_roll()) ** 2; n += 1; phi.append(p)
            else:
                env.res_scale = c["scale"]
                t0 = time.perf_counter(); act = c["pol"](obs); pm = (time.perf_counter() - t0) * 1e3
                obs, r, crashed, tr, info = env.step(act); done = crashed or tr
                cms.append(pm + getattr(env, "last_nn_ms", 0.0))
                e2 += info["err_sq_sum"]; n += info["n_plant_steps"]; phi.append(info["phi"])
            pl = env._plant_dyn                                   # گشتاور واقعی بعد از خرابی (نه درخواست کنترلر)
            Ua.append(pl.last_app.copy()); Uc.append(pl.last_cmd.copy())
            SAT.append(bool(np.any(np.abs(pl.last_cmd) >= 0.99 * pl.last_limit)))
            if done:
                break
        Ua, Uc, phi, cms = np.array(Ua), np.array(Uc), np.array(phi), np.array(cms)
        dd = lambda U: float(np.linalg.norm(np.diff(U, axis=0), axis=1).mean()) if len(U) > 1 else 0.0
        return dict(crashed=bool(crashed), steps=len(phi), rmse=float(np.sqrt(e2 / max(n, 1))),
                    peak_phi=float(np.rad2deg(np.abs(phi).max())),
                    u_rms=float(np.sqrt(np.mean(Ua ** 2))), u2_sum=float(np.sum(Ua ** 2)), du=dd(Ua),
                    u_rms_cmd=float(np.sqrt(np.mean(Uc ** 2))), du_cmd=dd(Uc),
                    sat=float(np.mean(SAT)), ms=float(cms.mean()))

    import pandas as pd
    scn = [s for s in SCENARIOS if a.only is None or s[0] in a.only]
    rows, t0 = [], time.time()
    for sid, ft, fp in scn:
        env.scn = (ft, fp)
        for name in ctrls:
            for i in range(a.episodes):
                rows.append(dict(scenario=sid, type=ft, ctrl=name, seed=a.seed0 + i, **run(name, a.seed0 + i)))
            d = pd.DataFrame([r for r in rows if r["scenario"] == sid and r["ctrl"] == name])
            print(f"{sid:26s} {name:18s} surv={1 - d.crashed.mean():.2f} rmse={d.rmse.mean():.3f} {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(a.out + "_episodes.csv", index=False)
    D = pd.DataFrame(rows); D.to_csv(a.out + "_episodes.csv", index=False)

    ref = D[(D.scenario == "healthy") & (D.ctrl == "nmpc")]
    early = set(ref[ref.crashed & (ref.steps <= 10)].seed)
    D["early"] = D.seed.isin(early)
    ok = D[~D.early]
    S = ok.groupby(["scenario", "ctrl"]).apply(lambda x: pd.Series(dict(
        n=len(x), surv=1 - x.crashed.mean(), rmse_alive=x[~x.crashed].rmse.mean(), peak_phi=x.peak_phi.mean(),
        u_rms=x.u_rms.mean(), u2_sum=x.u2_sum.mean(), du=x.du.mean(), u_rms_cmd=x.u_rms_cmd.mean(),
        du_cmd=x.du_cmd.mean(), sat=x.sat.mean(), ms=x.ms.mean())),
        include_groups=False)
    grp = lambda sid, ft: ("healthy" if ft == "none" else "sensor" if ft in SENSOR else "param" if ft == "param_fault" else "actuator")
    G = {sid: grp(sid, ft) for sid, ft, _ in scn}
    S["group"] = [G[i] for i in S.index.get_level_values("scenario")]
    S.to_csv(a.out + "_summary.csv")
    GS = S.groupby(["group", "ctrl"])[["surv", "rmse_alive", "u_rms", "du", "ms"]].mean()
    GS.to_csv(a.out + "_by_group.csv")
    print("\n=== میانگین هر گروه خرابی (وزن برابر برای هر سناریو) ===\n", GS.round(3).to_string())
    T = S.surv.unstack("ctrl").reindex([s[0] for s in scn])
    T.to_csv(a.out + "_survival_table.csv")
    pd.set_option("display.width", 220)
    print("\n=== SURVIVAL (بدون شروع‌های غیرقابل‌بازیابی) ===\n", T.round(3).to_string())
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(1.6 * len(T.columns) + 4, 0.28 * len(T) + 2))
        im = ax.imshow(T.values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(T.columns))); ax.set_xticklabels(T.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(T))); ax.set_yticklabels(T.index, fontsize=7)
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                ax.text(j, i, f"{T.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, label="survival"); fig.tight_layout(); fig.savefig(a.out + "_heatmap.png", dpi=200)
        fig2, axs = plt.subplots(1, 4, figsize=(16, 3.4))
        for ax2, col in zip(axs, ["surv", "rmse_alive", "u_rms", "du"]):
            GS[col].unstack("ctrl").plot.bar(ax=ax2, rot=0, legend=(col == "surv")); ax2.set_title(col)
        fig2.tight_layout(); fig2.savefig(a.out + "_by_group.png", dpi=200)
    except Exception as e:
        print("heatmap skipped:", e)
    json.dump(dict(args=vars(a), file_sha256=files, unrecoverable_seeds=sorted(int(s) for s in early)),
              open(a.out + "_config.json", "w"), indent=1)
    print("unrecoverable-start seeds (healthy، nmpc، ≤10 گام) در همین اجرا:", sorted(int(x) for x in early))
    print("\nنکته: param_mass_p30 ≡ اختیار کنترل ×0.77 و param_mass_m20 ≡ ×1.25 (چون g/h ثابت است).")