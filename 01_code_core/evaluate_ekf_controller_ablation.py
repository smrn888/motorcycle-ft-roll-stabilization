"""
EKF integration ablation: MPC / Neural MPC / Residual SAC.

Run from the existing "Three Models" project root and existing project venv.
This script is additive: it does not modify the supplied baseline files or train
any model. It evaluates matched seeds, buckets, and sensor-fault scenarios.

Modes:
  online_mpc            measured-state online MPC (baseline)
  ekf_mpc                EKF-state online MPC
  neural_mpc             measured-state Neural MPC (baseline)
  ekf_neural_mpc          EKF-state Neural MPC
  residual_adaptive       measured-state Residual SAC (baseline)
  ekf_residual_adaptive   EKF-state Residual SAC

The EKF is initialized/corrected from the faulted six-channel sensor stream.
Ground-truth plant state is used only for evaluation metrics and never as EKF
input. This is an exploratory ablation, not a certified distillation-teacher run.
"""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from motorcycle_dynamics import TAU_MAX
from compare_34faults_res_scale import ObservedFaultEnv, Policy
from compare_online_mpc_vs_neural_vs_residual import setup_online
from final_fault_benchmark import SCENARIOS


DEFAULT_SCENARIOS = [
    "healthy", "dropout_imu_p0.3", "dropout_imu_p0.5",
    "intermittent_imu_burst20", "stuck_phi_15deg",
]
MODES = [
    "online_mpc", "ekf_mpc", "neural_mpc", "ekf_neural_mpc",
    "residual_adaptive", "ekf_residual_adaptive",
]
FIELDS = [
    "controller", "scenario", "fault_type", "bucket", "seed", "iw",
    "surv", "rmse", "steps", "u_rms", "u2_sum", "du", "sat", "peak_phi",
    "wall_s", "solve_calls", "solve_failures", "solve_ms_mean",
    "solve_statuses", "ekf_ms_mean", "ekf_rejections", "ekf_roll_rmse",
    "horizon_requested", "horizon_observed",
]


class SixStateEKF:
    """Causal nominal-model EKF: [phi, phi_dot, delta_f, delta_r, ddf, ddr]."""

    def __init__(self, dyn, process_std, measure_std, initial_std, gate):
        self.dyn = dyn
        self.Q = np.diag(np.square(np.asarray(process_std, dtype=float)))
        self.R = np.diag(np.square(np.asarray(measure_std, dtype=float)))
        self.P0 = np.diag(np.square(np.asarray(initial_std, dtype=float)))
        self.gate = float(gate)
        self.x = np.zeros(6, dtype=float)
        self.P = self.P0.copy()
        self.rejected = 0
        self.updates = 0

    def reset(self, y):
        y = np.asarray(y, dtype=float)
        if y.shape != (6,) or not np.all(np.isfinite(y)):
            raise ValueError("EKF initialization requires six finite sensor values")
        self.x = y.copy()
        self.P = self.P0.copy()
        self.rejected = self.updates = 0

    def predict(self, u):
        F = np.asarray(self.dyn.linearise(float(self.x[0]))[0], dtype=float)
        self.x = np.asarray(self.dyn.step(self.x.copy(), np.asarray(u, dtype=float)))
        self.P = F @ self.P @ F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)

    def correct(self, y):
        y = np.asarray(y, dtype=float)
        if y.shape != (6,):
            raise ValueError("EKF correction requires six sensor values")
        # Gate roll and roll-rate; finite gimbal channels remain available.
        accepted = [2, 3, 4, 5]
        roll = [0, 1]
        innovation_roll = y[roll] - self.x[roll]
        Sroll = self.P[np.ix_(roll, roll)] + self.R[np.ix_(roll, roll)]
        try:
            nis = float(innovation_roll @ np.linalg.solve(Sroll, innovation_roll))
            if np.all(np.isfinite(innovation_roll)) and nis <= self.gate:
                accepted = roll + accepted
            else:
                self.rejected += 1
        except np.linalg.LinAlgError:
            self.rejected += 1

        idx = np.asarray([i for i in accepted if np.isfinite(y[i])], dtype=int)
        if idx.size == 0:
            return
        H = np.eye(6)[idx]
        innovation = y[idx] - H @ self.x
        Rk = self.R[np.ix_(idx, idx)]
        S = H @ self.P @ H.T + Rk
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ innovation
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rk @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.updates += 1


def get_obs10(env, ekf=None):
    obs = np.asarray(env._obs10(), dtype=np.float32).copy()
    if ekf is not None:
        obs[:6] = np.asarray(ekf.x, dtype=np.float32)
        obs[9] = float(ekf.x[0] - obs[8])
    if obs.shape != (10,) or not np.all(np.isfinite(obs)):
        raise ValueError("Invalid reconstructed 10-D observation")
    return obs


def get_residual_obs15(env, obs10):
    # Matches MotorcycleEnvResidual._make_obs() feature order and scaling.
    horizon = int(env._horizon)
    nn_input = np.concatenate([
        obs10, np.asarray(env._u_last, dtype=np.float32), [horizon]
    ]).astype(np.float32)
    u_nn = np.asarray(env._nn_u(nn_input), dtype=np.float32)
    h_idx = list(env.HORIZONS).index(horizon)
    h_norm = 2.0 * h_idx / (len(env.HORIZONS) - 1) - 1.0
    obs15 = np.concatenate([
        obs10,
        (np.asarray(env._u_last, dtype=np.float32) / env.U_REF),
        (u_nn / env.U_REF),
        np.asarray([h_norm], dtype=np.float32),
    ]).astype(np.float32)
    if obs15.shape != (15,) or not np.all(np.isfinite(obs15)):
        raise ValueError("Invalid reconstructed 15-D residual observation")
    return obs15


def rollout(env, mode, seed, bucket, iw, policy, horizon, weights, lp_alpha, ekf_cfg):
    env.reset(seed=seed, options={"bucket": bucket, "iw": iw})
    env._u_last = np.zeros(2, dtype=float)
    env._horizon = int(horizon)
    use_ekf = mode.startswith("ekf_")
    ekf = None
    if use_ekf:
        ekf = SixStateEKF(
            env.dyn, ekf_cfg["process_std"], ekf_cfg["measure_std"],
            ekf_cfg["initial_std"], ekf_cfg["gate"]
        )
        # No true-state initialization: use the faulted measurement pathway.
        ekf.reset(np.asarray(env._obs10()[:6], dtype=float))

    is_mpc = mode in ("online_mpc", "ekf_mpc")
    is_neural = mode in ("neural_mpc", "ekf_neural_mpc")
    is_residual = mode in ("residual_adaptive", "ekf_residual_adaptive")
    solver = env._solvers[horizon] if is_mpc else None
    if solver is not None:
        solver.reset_warm_start()

    q = np.array([
        weights[0], weights[1], weights[2], weights[2],
        0.1 * weights[2], 0.1 * weights[2]
    ], dtype=float)
    r = np.array([weights[3], weights[3]], dtype=float)
    statuses = Counter()
    solve_calls = solve_failures = 0
    solve_ms = ekf_ms = 0.0
    estimate_error_sq = 0.0
    applied, err_sq, peak = [], 0.0, 0.0
    crashed = False
    horizons_seen = []
    start = time.perf_counter()

    for k in range(env.MAX_STEPS):
        if ekf is not None and k > 0:
            t = time.perf_counter()
            ekf.correct(np.asarray(env._obs10()[:6], dtype=float))
            ekf_ms += (time.perf_counter() - t) * 1000.0

        obs10 = get_obs10(env, ekf)
        if ekf is not None:
            # Diagnostic only; this value is never passed to a controller.
            estimate_error_sq += float((ekf.x[0] - env.state[0]) ** 2)

        if is_mpc:
            xd = np.zeros(6, dtype=float)
            xd[0] = env._desired_roll()
            try:
                A, B, bias = env.dyn.linearise(float(obs10[0]))
                solver.update_model(A, B, bias)
                pc = env._dare_chol(A, B, Q_diag=q, R_diag=r)
                solver.u_prev.value = env._u_last.copy()
                t = time.perf_counter()
                u = np.asarray(
                    solver.solve(x0=obs10[:6].astype(float), xd=xd,
                                 Q_diag=q, R_diag=r, P_chol=pc), dtype=float
                )
                solve_ms += (time.perf_counter() - t) * 1000.0
                status = str(solver.last_solve_status)
            except Exception as exc:
                u = np.zeros(2, dtype=float)
                status = "pre_solve_exception:" + type(exc).__name__
            solve_calls += 1
            statuses[status] += 1
            if status not in ("optimal", "optimal_inaccurate") or u.shape != (2,) or not np.all(np.isfinite(u)):
                solve_failures += 1
                u = np.zeros(2, dtype=float)
            u = np.clip(u, -TAU_MAX, TAU_MAX)

        elif is_neural:
            nn_input = np.concatenate([
                obs10, np.asarray(env._u_last, dtype=np.float32), [horizon]
            ]).astype(np.float32)
            u = np.clip(env._nn_u(nn_input), -TAU_MAX, TAU_MAX)

        elif is_residual:
            obs15 = get_residual_obs15(env, obs10)
            action = np.asarray(policy(obs15), dtype=float)
            if action.shape != (3,):
                raise ValueError(f"Residual policy returned unexpected shape {action.shape}")
            h_index = min(
                int((np.clip(action[2], -1, 1) + 1.0) * 0.5 * len(env.HORIZONS)),
                len(env.HORIZONS) - 1,
            )
            env._horizon = int(env.HORIZONS[h_index])
            horizons_seen.append(env._horizon)
            nn_input = np.concatenate([
                obs10, np.asarray(env._u_last, dtype=np.float32), [env._horizon]
            ]).astype(np.float32)
            u_base = env._nn_u(nn_input)
            u = np.clip(
                u_base + env.res_scale * np.clip(action[:2], -1, 1),
                -TAU_MAX, TAU_MAX
            )
        else:
            raise ValueError("Unknown mode: " + mode)

        # Advance the physical plant exactly once per control step.
        env._steps += 1
        env.state = env._plant_dyn.step(env.state, u)
        env.v = float(np.clip(env.v + env.np_random.normal(0., env.V_NOISE), 1., 8.))
        env.alpha = float(np.clip(env.alpha + env.np_random.normal(0., env.ALPHA_NOISE), -.20, .20))
        phi = float(env.state[0])
        err = phi - env._desired_roll()
        err_sq += err * err
        peak = max(peak, abs(phi))
        applied.append(env._plant_dyn.last_app.copy())
        env._u_last = np.asarray(u, dtype=float).copy()

        if ekf is not None:
            t = time.perf_counter()
            ekf.predict(u)
            ekf_ms += (time.perf_counter() - t) * 1000.0

        crashed = abs(phi) > np.pi / 4
        if crashed or env._steps >= env.MAX_STEPS:
            break

    ua = np.asarray(applied, dtype=float)
    n = len(ua)
    return {
        "surv": int(not crashed),
        "rmse": float(np.sqrt(err_sq / n)),
        "steps": n,
        "u_rms": float(np.sqrt(np.mean(ua ** 2))),
        "u2_sum": float(np.sum(ua ** 2)),
        "du": float(np.linalg.norm(np.diff(ua, axis=0), axis=1).mean()) if n > 1 else 0.0,
        "sat": float(np.mean(np.any(np.abs(ua) >= .99 * TAU_MAX, axis=1))),
        "peak_phi": float(np.rad2deg(peak)),
        "wall_s": time.perf_counter() - start,
        "solve_calls": solve_calls,
        "solve_failures": solve_failures,
        "solve_ms_mean": solve_ms / solve_calls if solve_calls else 0.0,
        "solve_statuses": json.dumps(dict(statuses), sort_keys=True),
        "ekf_ms_mean": ekf_ms / max(n, 1) if ekf is not None else 0.0,
        "ekf_rejections": ekf.rejected if ekf is not None else 0,
        "ekf_roll_rmse": float(np.sqrt(estimate_error_sq / n)) if ekf is not None else "",
        "horizon_requested": int(horizon),
        "horizon_observed": float(np.mean(horizons_seen)) if horizons_seen else int(horizon),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="neural_mpc_multi13.pt")
    p.add_argument("--res_stem", default="sac_seed404_fromscratch_1m_best_eval")
    p.add_argument("--res_scale", type=float, default=5.0)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed0", type=int, default=78000)
    p.add_argument("--buckets", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--iw", type=float, default=1.0)
    p.add_argument("--only", nargs="+", default=DEFAULT_SCENARIOS)
    p.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    p.add_argument("--ekf_gate", type=float, default=9.21)
    p.add_argument("--ekf_q", type=float, nargs=6, default=[.002, .05, .005, .005, .10, .10])
    p.add_argument("--ekf_r", type=float, nargs=6, default=[.01, .10, .005, .005, .05, .05])
    p.add_argument("--ekf_p0", type=float, nargs=6, default=[.05, .5, .05, .05, .5, .5])
    p.add_argument("--out", default="ekf_controller_ablation")
    p.add_argument("--preflight_only", action="store_true")
    a = p.parse_args()

    if a.episodes < 1 or not set(a.buckets).issubset({0, 1, 2}) or len(set(a.buckets)) != len(a.buckets):
        p.error("Invalid episodes/buckets")
    if a.horizon not in (10, 20, 30, 40):
        p.error("horizon must be one of 10, 20, 30, 40")
    if a.ekf_gate <= 0 or any(x <= 0 for x in a.ekf_q + a.ekf_r + a.ekf_p0):
        p.error("EKF gate and standard deviations must be positive")

    required = [Path(a.ckpt)]
    if any(m in a.modes for m in ("residual_adaptive", "ekf_residual_adaptive")):
        required += [Path(a.res_stem + ".zip"), Path(a.res_stem + "_vecnorm.pkl")]
    for f in required:
        if not f.is_file():
            p.error("Missing required file: " + str(f))

    scenarios = [s for s in SCENARIOS if s[0] in a.only]
    missing = set(a.only) - {s[0] for s in scenarios}
    if missing or not scenarios:
        p.error("Unknown/empty scenario selection: " + str(sorted(missing)))

    env = ObservedFaultEnv(
        ckpt=a.ckpt, decision_k=1, iw_list=(1.0,), res_scale=a.res_scale
    )
    weights, lp_alpha, decoded_n = setup_online(env, a.horizon)
    policy = None
    if any(m in a.modes for m in ("residual_adaptive", "ekf_residual_adaptive")):
        policy = Policy(Path(a.res_stem + ".zip"), Path(a.res_stem + "_vecnorm.pkl"), 15, 3)

    ekf_cfg = {
        "process_std": a.ekf_q, "measure_std": a.ekf_r,
        "initial_std": a.ekf_p0, "gate": a.ekf_gate,
    }
    print("PREFLIGHT: modes=", a.modes, "horizon=", a.horizon,
          "weights=", weights, "decoded_N=", decoded_n,
          "residual_steps=", getattr(policy, "steps", "not loaded"), flush=True)

    # Deterministic reset gate and EKF numerical smoke test.
    for b in a.buckets:
        env.scenario = ("none", {})
        env.reset(seed=a.seed0, options={"bucket": b, "iw": a.iw})
        initial = (env.state.copy(), env.v, env.alpha, env._desired_roll())
        env.reset(seed=a.seed0, options={"bucket": b, "iw": a.iw})
        np.testing.assert_array_equal(initial[0], env.state)
        assert initial[1:] == (env.v, env.alpha, env._desired_roll())
    env.scenario = ("none", {})
    env.reset(seed=a.seed0, options={"bucket": a.buckets[0], "iw": a.iw})
    ekf_test = SixStateEKF(env.dyn, **ekf_cfg)
    ekf_test.reset(np.asarray(env._obs10()[:6], dtype=float))
    ekf_test.predict(np.zeros(2))
    ekf_test.correct(np.asarray(env._obs10()[:6], dtype=float))
    assert np.all(np.isfinite(ekf_test.x))
    assert np.all(np.linalg.eigvalsh(ekf_test.P) > -1e-10)
    print("PREFLIGHT PASS: reset reproducibility and EKF finite/PSD checks", flush=True)
    if a.preflight_only:
        return

    output = Path(a.out + "_episodes.csv")
    summary = Path(a.out + "_summary.csv")
    config_path = Path(a.out + "_config.json")
    cfg = {
        "seed0": a.seed0, "episodes": a.episodes, "buckets": a.buckets,
        "scenarios": a.only, "modes": a.modes, "horizon": a.horizon,
        "iw": a.iw, "res_scale": a.res_scale, "ekf": ekf_cfg,
        "weights": weights.tolist(), "lp_alpha": lp_alpha,
        "checkpoint": str(Path(a.ckpt).resolve()),
        "residual_stem": a.res_stem,
        "caveat": "Exploratory ablation; online MPC is not certified as exact distillation teacher.",
    }
    if output.exists() or summary.exists() or config_path.exists():
        p.error("Output already exists; choose a new --out to avoid mixing runs")
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for sid, fault_type, fault_params in scenarios:
            env.scenario = (fault_type, fault_params)
            for bucket in a.buckets:
                for seed in range(a.seed0, a.seed0 + a.episodes):
                    for mode in a.modes:
                        result = rollout(
                            env, mode, seed, bucket, a.iw, policy, a.horizon,
                            weights, lp_alpha, ekf_cfg
                        )
                        writer.writerow({
                            "controller": mode, "scenario": sid, "fault_type": fault_type,
                            "bucket": bucket, "seed": seed, "iw": a.iw, **result
                        })
                        f.flush()
                        print(
                            sid, "bucket=", bucket, "seed=", seed, mode,
                            "surv=", result["surv"], "steps=", result["steps"],
                            "rmse=", round(result["rmse"], 5),
                            "ekf_rej=", result["ekf_rejections"], flush=True
                        )

    import pandas as pd
    df = pd.read_csv(output)
    expected = len(scenarios) * len(a.buckets) * a.episodes * len(a.modes)
    if len(df) != expected:
        raise RuntimeError(f"Expected {expected} rows, got {len(df)}")
    if df.duplicated(["controller", "scenario", "bucket", "seed"]).any():
        raise RuntimeError("Duplicate episode keys detected")
    agg = df.groupby(["controller", "scenario", "bucket"], as_index=False).agg(
        episodes=("surv", "size"), survival=("surv", "mean"),
        rmse=("rmse", "mean"), steps=("steps", "mean"),
        u_rms=("u_rms", "mean"), du=("du", "mean"),
        solver_failures=("solve_failures", "sum"),
        ekf_rejections=("ekf_rejections", "sum"),
        ekf_ms_mean=("ekf_ms_mean", "mean"),
        ekf_roll_rmse=("ekf_roll_rmse", "mean"),
    )
    agg.to_csv(summary, index=False)
    print("DONE:", len(df), "episode rows")
    print("Episodes:", output)
    print("Summary:", summary)
    print("Config:", config_path)


if __name__ == "__main__":
    main()
