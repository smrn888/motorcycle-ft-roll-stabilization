import argparse, csv, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, os.getcwd())
from motorcycle_dynamics import TAU_MAX
from compare_34faults_res_scale import ObservedFaultEnv, Policy
from final_fault_benchmark import SCENARIOS

SCEN5 = ["healthy", "dropout_imu_p0.3", "dropout_imu_p0.5",
         "intermittent_imu_burst20", "stuck_phi_15deg"]
CFG = dict(process_std=[.002,.05,.005,.005,.10,.10],
           measure_std=[.01,.10,.005,.005,.05,.05],
           initial_std=[.05,.5,.05,.05,.5,.5], gate=9.21,
           stale_phi_tol=1e-10, stale_rate_tol=1e-10,
           stale_run=2, recovery_run=5, recovery_factor=4.0, recovery_cap=1.0)
RES = ["sac_seed101_continued_1m_best", "sac_seed202_continued_1m_best",
       "sac_seed303_continued_1m_best", "sac_seed404_fromscratch_1m_best_eval",
       "sac_seed505_fromscratch_1m_best_eval"]
FIELDS = ["kind","policy","mode","lam","fix_h","scenario","bucket","seed",
          "surv","steps","rmse","n_g150","n_g155"]

def pure_pol(seed):
    return Policy(Path(f"standalone_runs/best_sacplain_seed{seed}.zip"),
                  Path(f"standalone_runs/vecnorm_best_sacplain_seed{seed}.pkl"), 10, 2)

def res_pol(stem):
    return Policy(Path(stem + ".zip"), Path(stem + "_vecnorm.pkl"), 15, 3)

def norm_h(value):
    if value is None or str(value).strip().lower() in ("", "none", "nan"):
        return "None"
    return str(value).strip()

def row_key(r):
    return (str(r.get("kind","")).strip(), str(r.get("policy","")).strip(),
            str(r.get("mode","")).strip(), str(r.get("lam","")).strip(),
            norm_h(r.get("fix_h")), str(r.get("scenario","")).strip(),
            str(r.get("bucket","")).strip(), str(r.get("seed","")).strip())

def rollout(env, kind, pol, seed, bucket, mode, lam, fix_h, EKF):
    env.reset(seed=seed, options={"bucket": bucket, "iw": 1.0})
    env._u_last = np.zeros(2)
    env._horizon = 20
    ekf = None
    if mode != "raw":
        if EKF is None:
            raise RuntimeError("EKF class was not loaded; use --task ekf for EKF modes.")
        ekf_mode = "combined" if mode == "combined" else "baseline"
        ekf = EKF(env.dyn, mode=ekf_mode, **CFG)
        ekf.reset(np.asarray(env._obs10()[:6], dtype=float))

    e2, n, g150, g155 = 0.0, 0, 0, 0
    crashed = False
    for k in range(env.MAX_STEPS):
        if ekf is not None and k:
            ekf.correct(np.asarray(env._obs10()[:6], dtype=float))
        if ekf is not None:
            x = ekf.x.copy()
            pd_ = env._desired_roll()
            o10 = np.array([*x, env.v, env.alpha, pd_, x[0]-pd_], dtype=np.float32)
        else:
            o10 = env._obs10()
        ul = np.asarray(env._u_last, dtype=np.float32)

        if kind == "neural":
            u = env._nn_u(np.concatenate([o10, ul, [20.]]).astype(np.float32))
        elif kind == "pure":
            u = np.clip(np.asarray(pol(o10), dtype=float), -1, 1) * TAU_MAX
        elif kind == "residual":
            un = env._nn_u(np.concatenate([o10, ul, [float(env._horizon)]]).astype(np.float32))
            hi = env.HORIZONS.index(env._horizon)
            hn = 2.0 * hi / (len(env.HORIZONS)-1) - 1.0
            obs = np.concatenate([o10, ul/env.U_REF, (un/env.U_REF).astype(np.float32), [hn]]).astype(np.float32)
            a = np.asarray(pol(obs), dtype=float)
            if fix_h is None:
                ah = float(np.clip(a[2], -1, 1))
                kk = min(int((ah+1)*0.5*len(env.HORIZONS)), len(env.HORIZONS)-1)
                env._horizon = env.HORIZONS[kk]
            else:
                if fix_h not in env.HORIZONS:
                    raise ValueError(f"fix_h={fix_h} not in env.HORIZONS={env.HORIZONS}")
                env._horizon = fix_h
            base = env._nn_u(np.concatenate([o10, ul, [float(env._horizon)]]).astype(np.float32))
            u = base + float(lam) * np.clip(a[:2], -1, 1)
        else:
            raise ValueError(f"Unknown controller kind: {kind}")

        u = np.clip(np.asarray(u, dtype=float), -TAU_MAX, TAU_MAX)
        if ekf is not None:
            ekf.predict(u)
        env._steps += 1
        env.state = env._plant_dyn.step(env.state, u)
        env.v = float(np.clip(env.v + env.np_random.normal(0., env.V_NOISE), 1., 8.))
        env.alpha = float(np.clip(env.alpha + env.np_random.normal(0., env.ALPHA_NOISE), -.20, .20))
        phi = float(env.state[0])
        err = phi - env._desired_roll()
        e2 += err*err
        n += 1
        gm = max(abs(env.state[2]), abs(env.state[3]))
        g150 += int(gm >= 1.50)
        g155 += int(gm >= 1.55)
        env._u_last = u.copy()
        crashed = abs(phi) > np.pi/4
        if crashed or env._steps >= env.MAX_STEPS:
            break
    return {"surv": int(n > 0 and not crashed), "steps": n,
            "rmse": float(np.sqrt(e2/n)) if n else float("nan"),
            "n_g150": g150, "n_g155": g155}

def summarize(path, task):
    import pandas as pd
    if not path.exists() or path.stat().st_size == 0:
        print(f"No results to summarize: {path}")
        return
    d = pd.read_csv(path)
    if d.empty:
        print(f"No result rows to summarize: {path}")
        return
    if task == "gimbal":
        g = d.groupby(["kind","scenario"]).agg(
            episodes=("surv","size"), surv=("surv","mean"), steps=("steps","sum"),
            g150=("n_g150","sum"), g155=("n_g155","sum"))
        g["frac_ge_1.50"] = g.g150 / g.steps.replace(0, np.nan)
        g["frac_ge_1.55"] = g.g155 / g.steps.replace(0, np.nan)
        print(g.round(4).to_string())
        return
    d["variant"] = (d.kind.astype(str)+"|"+d["mode"].astype(str)+"|lam="+
                    d.lam.astype(str)+"|N="+d.fix_h.fillna("None").astype(str))
    t = d.pivot_table(index="variant", columns="scenario", values="surv", aggfunc="mean") * 100
    cols = [c for c in SCEN5 if c in t.columns]
    if cols:
        t["macro"] = t[cols].mean(axis=1)
        print(t[cols+["macro"]].round(2).to_string())
    else:
        print("No expected scenario columns found.")
        print(t.round(2).to_string())

def main():
    ap = argparse.ArgumentParser(description="Run lambda/horizon, EKF, or gimbal evaluation.")
    ap.add_argument("--task", choices=["lambda","ekf","gimbal"], required=True)
    ap.add_argument("--ckpt", default="neural_mpc_multi13.pt")
    ap.add_argument("--seed0", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--buckets", type=int, nargs="+", default=[0,1,2])
    ap.add_argument("--kinds", nargs="+", choices=["neural","residual","pure"],
                    default=["neural","residual","pure"])
    ap.add_argument("--modes", nargs="+", choices=["raw","baseline","combined"], default=["combined"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.seeds is not None and a.seeds < 1:
        ap.error("--seeds must be at least 1")
    if not a.buckets or any(b < 0 for b in a.buckets):
        ap.error("--buckets must be non-negative integers")
    if a.seed0 is None:
        a.seed0 = 500000 if a.task == "gimbal" else 78000
    if a.seeds is None:
        a.seeds = 5 if a.task == "gimbal" else 100
    if not Path(a.ckpt).is_file():
        raise FileNotFoundError(f"Neural MPC checkpoint not found: {a.ckpt}")

    EKF = None
    if a.task == "ekf" and any(m != "raw" for m in a.modes):
        from evaluate_online_mpc_ekf_ablation import RobustEKF
        EKF = RobustEKF

    env = ObservedFaultEnv(ckpt=a.ckpt, decision_k=1, iw_list=(1.,), res_scale=5.)
    scenario_map = {sid: (ft, fp) for sid, ft, fp in SCENARIOS}
    scenarios = ["healthy","dropout_imu_p0.3"] if a.task == "gimbal" else SCEN5
    missing = [s for s in scenarios if s not in scenario_map]
    if missing:
        raise KeyError("Scenario IDs missing from SCENARIOS: " + ", ".join(missing))

    kinds = ["residual","pure"] if a.task == "gimbal" else a.kinds
    if a.task == "lambda":
        kinds = [k for k in kinds if k != "pure"]
    policies = []
    for kind in kinds:
        if kind == "neural":
            candidates = [("nn", None)]
        elif kind == "residual":
            candidates = ([("seed202_final", res_pol("sac_seed202_continued_1m_final"))]
                          if a.task == "gimbal" else [(Path(s).name, res_pol(s)) for s in RES])
        else:
            candidates = ([("seed22", pure_pol(22))] if a.task == "gimbal"
                          else [(f"seed{s}", pure_pol(s)) for s in (11,22,33,44,55)])
        for name, pol in candidates:
            if a.task == "lambda":
                combos = ([("raw",0.0,20)] if kind == "neural" else
                          [("raw",0.0,20),("raw",5.0,20),("raw",0.0,None),("raw",5.0,None)])
            elif a.task == "ekf":
                lam, fh = {"neural":(0.0,20),"pure":(0.0,None),"residual":(5.0,None)}[kind]
                combos = [(m,lam,fh) for m in a.modes]
            else:
                combos = [("raw",5.0 if kind == "residual" else 0.0,None)]
            for mode, lam, fh in combos:
                policies.append((kind,name,pol,mode,lam,fh))

    path = Path(a.out + "_episodes.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and not set(FIELDS).issubset(reader.fieldnames):
                raise ValueError(f"Unexpected columns in existing CSV: {reader.fieldnames}. Use a new --out.")
            for r in reader:
                if r.get("kind"):
                    done.add(row_key(r))

    total = len(policies)*len(scenarios)*len(a.buckets)*a.seeds
    new_count = skipped = 0
    t0 = time.time()
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        for kind,name,pol,mode,lam,fh in policies:
            for sid in scenarios:
                env.scenario = scenario_map[sid]
                for bucket in a.buckets:
                    for seed in range(a.seed0, a.seed0+a.seeds):
                        key = (kind,name,mode,str(lam),norm_h(fh),sid,str(bucket),str(seed))
                        if key in done:
                            skipped += 1
                            continue
                        result = rollout(env,kind,pol,seed,bucket,mode,lam,fh,EKF)
                        w.writerow({"kind":kind,"policy":name,"mode":mode,"lam":lam,
                                    "fix_h":"" if fh is None else fh,"scenario":sid,
                                    "bucket":bucket,"seed":seed,**result})
                        f.flush()
                        done.add(key)
                        new_count += 1
                        if new_count % 50 == 0:
                            print(f"new={new_count}; skipped={skipped}; planned={total}; elapsed={time.time()-t0:.0f}s",
                                  flush=True)
    print(f"DONE: {path}\nnew={new_count}; skipped={skipped}; planned={total}; elapsed={time.time()-t0:.1f}s",
          flush=True)
    summarize(path, a.task)

if __name__ == "__main__":
    main()
