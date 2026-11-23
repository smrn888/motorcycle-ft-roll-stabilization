"""Matched 100-seed Pure SAC evaluation (5 independent training checkpoints).

Execute in project root with compare_34faults_res_scale.py and final_fault_benchmark.py.
Does not train or modify project files. Uses the same ObservedFaultEnv measurement
and plant stepping as evaluate_neural_residual_adaptive_100seeds_patched.py.

Pure SAC observations: env._obs10() (10); policy actions: (2,) normalized,
torque command: TAU_MAX * clip(action, -1, 1).
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from motorcycle_dynamics import TAU_MAX
from compare_34faults_res_scale import ObservedFaultEnv, Policy
from final_fault_benchmark import SCENARIOS

DEFAULT_ONLY = ['healthy','dropout_imu_p0.3','dropout_imu_p0.5','intermittent_imu_burst20','stuck_phi_15deg']
FIELDS = ['controller','training_seed','checkpoint_stem','scenario','fault_type','bucket','seed','iw','horizon',
          'surv','steps','rmse','u_rms','du','sat','peak_phi','mean_horizon','horizon_switches','wall_s']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rollout(env, policy, seed, bucket, iw):
    env.reset(seed=seed, options={'bucket':bucket,'iw':iw})
    env._u_last = np.zeros(2, dtype=float)
    e2 = 0.0; applied = []; peak = 0.0; crashed = False
    start = time.perf_counter()
    for _ in range(env.MAX_STEPS):
        obs = np.asarray(env._obs10(), dtype=np.float32).copy()
        if obs.shape != (10,) or not np.all(np.isfinite(obs)):
            raise RuntimeError(f'Invalid Pure SAC observation {obs.shape}')
        action = np.asarray(policy(obs), dtype=float).reshape(-1)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise RuntimeError(f'Invalid Pure SAC action {action}')
        u = TAU_MAX * np.clip(action, -1., 1.)
        env._steps += 1
        env.state = env._plant_dyn.step(env.state, u)
        env.v = float(np.clip(env.v + env.np_random.normal(0., env.V_NOISE), 1., 8.))
        env.alpha = float(np.clip(env.alpha + env.np_random.normal(0., env.ALPHA_NOISE), -.20, .20))
        phi = float(env.state[0]); err = phi - env._desired_roll()
        e2 += err * err; peak = max(peak, abs(phi))
        applied.append(env._plant_dyn.last_app.copy())
        env._u_last = np.asarray(u, float).copy()
        crashed = abs(phi) > np.pi / 4
        if crashed or env._steps >= env.MAX_STEPS:
            break
    ua = np.asarray(applied, float); n = len(ua)
    return dict(surv=int(not crashed), steps=n, rmse=float(np.sqrt(e2 / n)),
        u_rms=float(np.sqrt(np.mean(ua**2))),
        du=float(np.linalg.norm(np.diff(ua,axis=0),axis=1).mean()) if n > 1 else 0.,
        sat=float(np.mean(np.any(np.abs(ua) >= .99 * TAU_MAX,axis=1))),
        peak_phi=float(np.rad2deg(peak)),mean_horizon=float('nan'),horizon_switches=0,
        wall_s=time.perf_counter()-start)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', default='neural_mpc_multi13.pt', help='Shared env initializer; Pure SAC never calls NN')
    p.add_argument('--pure_stems', nargs=5,
        help='Optional 5 Pure SAC checkpoint stems; requires stem_vecnorm.pkl naming')
    p.add_argument('--pure_dir', default='standalone_runs',
        help='Directory containing best_sacplain_seedXX.zip and vecnorm_best_sacplain_seedXX.pkl')
    p.add_argument('--pure_variant', choices=['best','final'], default='best')
    p.add_argument('--training_seeds', nargs=5, type=int, default=[11,22,33,44,55])
    p.add_argument('--seed0', type=int, default=78000)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--buckets', type=int, nargs='+', default=[0,1,2])
    p.add_argument('--only', nargs='+', default=DEFAULT_ONLY)
    p.add_argument('--iw', type=float, default=1.)
    p.add_argument('--out', default='pure_sac_78000_100seeds')
    p.add_argument('--preflight_only', action='store_true')
    a = p.parse_args()
    if len(set(a.training_seeds)) != 5: p.error('Five training seeds must be distinct')
    if a.episodes < 1 or len(set(a.buckets)) != len(a.buckets) or not set(a.buckets).issubset({0,1,2}):
        p.error('Invalid --episodes/--buckets')
    if not Path(a.ckpt).is_file(): p.error('Missing --ckpt: '+a.ckpt)
    if a.pure_stems:
        pairs = [(Path(s+'.zip'), Path(s+'_vecnorm.pkl')) for s in a.pure_stems]
        stems = list(a.pure_stems)
    else:
        pairs = []
        stems = []
        for training_seed in a.training_seeds:
            tag = f'{a.pure_variant}_sacplain_seed{training_seed}'
            stem = Path(a.pure_dir) / tag
            pairs.append((Path(str(stem)+'.zip'), Path(a.pure_dir)/('vecnorm_'+tag+'.pkl')))
            stems.append(str(stem))
    for pair in pairs:
        for path in pair:
            if not path.is_file(): p.error('Missing checkpoint component: '+str(path))
    scenarios = [s for s in SCENARIOS if s[0] in a.only]
    if set(a.only) != {s[0] for s in scenarios}: p.error('Unknown scenario(s)')
    policies = [Policy(zipf, normf, 10, 2) for zipf,normf in pairs]
    # Intentionally use the same observation/fault wrapper as Neural & Residual
    # The frozen neural reference is initialized but NEVER used to generate Pure actions.
    env = ObservedFaultEnv(ckpt=a.ckpt, decision_k=1, iw_list=(1.,), res_scale=5.)
    for b in a.buckets:
        env.scenario = ('none', {})
        env.reset(seed=a.seed0, options={'bucket':b,'iw':a.iw})
        x1=(env.state.copy(),env.v,env.alpha,env._desired_roll())
        env.reset(seed=a.seed0, options={'bucket':b,'iw':a.iw})
        np.testing.assert_array_equal(x1[0], env.state)
        assert x1[1:] == (env.v,env.alpha,env._desired_roll())
        obs=np.asarray(env._obs10(),dtype=np.float32)
        assert obs.shape==(10,)
        for pol in policies:
            act=np.asarray(pol(obs))
            if act.shape != (2,) or not np.all(np.isfinite(act)):
                raise RuntimeError(f'Pure SAC policy did not return finite (2,) action: {act}')
    print('PREFLIGHT PASS: 10-D obs; 2-D actions; repeatable initial states; Pure SAC no neural baseline action',flush=True)
    print('CHECKPOINTS:',list(zip(a.training_seeds,stems)),flush=True)
    if a.preflight_only: return
    ep=Path(a.out+'_episodes.csv'); summ=Path(a.out+'_summary.csv'); cfgp=Path(a.out+'_config.json')
    cfg = dict(controller='pure_sac', training_seeds=a.training_seeds,
        stems=[str(Path(s).resolve()) for s in stems],
        checkpoints=[dict(zip_sha256=sha(x),vecnorm_sha256=sha(y)) for x,y in pairs],
        shared_ckpt=str(Path(a.ckpt).resolve()), shared_ckpt_sha256=sha(a.ckpt),
        seed0=a.seed0,episodes=a.episodes,buckets=a.buckets,only=a.only,iw=a.iw,
        observation='ObservedFaultEnv._obs10 (10D) including corrupted measurements',
        action='clip(policy(normalized_obs),-1,1)*TAU_MAX',
        metrics='matched run() in evaluate_neural_residual_adaptive_100seeds_patched.py')
    if cfgp.exists():
        if json.loads(cfgp.read_text(encoding='utf-8')) != cfg:
            p.error('Config mismatch; use a new --out')
    elif ep.exists(): p.error('Episodes exist without config; use a new --out')
    else: cfgp.write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf-8')
    done=set()
    if ep.exists():
        with ep.open(newline='',encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
            if rows and list(rows[0]) != FIELDS: raise RuntimeError('Episode CSV schema mismatch')
            for r in rows:
                key=(r['controller'],r['scenario'],int(r['bucket']),int(r['seed']))
                if key in done: raise RuntimeError(f'Duplicate existing key: {key}')
                done.add(key)
    with ep.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS)
        if not done: w.writeheader()
        for sid,ft,fp in scenarios:
            env.scenario = (ft,fp)
            for bucket in a.buckets:
                for seed in range(a.seed0,a.seed0+a.episodes):
                    for training_seed,stem,pol in zip(a.training_seeds,stems,policies):
                        label='pure_sac_seed'+str(training_seed)
                        key=(label,sid,bucket,seed)
                        if key in done: continue
                        r=rollout(env,pol,seed,bucket,a.iw)
                        w.writerow(dict(controller=label,training_seed=training_seed,
                            checkpoint_stem=stem,scenario=sid,fault_type=ft,bucket=bucket,
                            seed=seed,iw=a.iw,horizon='',**r))
                        f.flush(); done.add(key)
                        print(f'{label} {sid} bucket={bucket} seed={seed} survival={r["surv"]} steps={r["steps"]}',flush=True)
    import pandas as pd
    df=pd.read_csv(ep)
    expected=len(scenarios)*len(a.buckets)*a.episodes*5
    if len(df)!=expected or df.duplicated(['controller','scenario','bucket','seed']).any():
        raise RuntimeError(f'Bad result cardinality: expected {expected}, got {len(df)}')
    df.groupby(['controller','scenario','bucket'],as_index=False).agg(
        episodes=('surv','size'),survival=('surv','mean'),steps=('steps','mean'),
        rmse=('rmse','mean'),u_rms=('u_rms','mean'),du=('du','mean'),
        sat=('sat','mean'),peak_phi=('peak_phi','mean'),wall_s=('wall_s','mean')
    ).to_csv(summ,index=False)
    print('DONE',len(df),'episodes;',ep,summ,cfgp,flush=True)

if __name__=='__main__': main()
