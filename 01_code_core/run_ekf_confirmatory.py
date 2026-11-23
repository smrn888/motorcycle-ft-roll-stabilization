"""Frozen EKF Combined experiment launcher (stdlib-only).

Smoke: 1 held-out seed, all five faults, bucket 0, N=20.
Full: 100 held-out seeds, 3 buckets, five faults, N=10/20/30/40.
No EKF/plant/MPC source modifications. Never auto-tunes on held-out outcomes.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = 'evaluate_online_mpc_ekf_ablation.py'
SCENARIOS = ('healthy','dropout_imu_p0.3','dropout_imu_p0.5','intermittent_imu_burst20','stuck_phi_15deg')
HORIZONS = (10,20,30,40)
FROZEN = dict(ekf_gate=9.21,
    ekf_q=[.002,.05,.005,.005,.10,.10],
    ekf_r=[.01,.10,.005,.005,.05,.05],
    ekf_p0=[.05,.5,.05,.05,.5,.5],
    stale_phi_tol=1e-10,stale_rate_tol=1e-10,stale_run=2,
    recovery_run=5,recovery_factor=4.0,recovery_cap=1.0)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def get_specs(base, ckpt):
    return dict(protocol='ekf_combined_current_osqp_holdout_v1',
        source_hash=sha(SOURCE),checkpoint_hash=sha(ckpt),
        evaluator_hash=sha(__file__), seed0=78000, episodes=100,
        buckets=[0,1,2],scenarios=list(SCENARIOS),horizons=list(HORIZONS),
        osqp=dict(eps_abs=1e-4,eps_rel=1e-4,max_iter=4000,warm_start=True,fallback='zero'),
        ekf=FROZEN,smoke=dict(seed=78000,bucket=0,horizon=20,scenarios=list(SCENARIOS)),
        note='Do not adjust controller or estimator after reviewing seed 78000 smoke results; smoke is an integrity check only.')


def verify_csv(path, n, horizon, seed0, buckets, scenarios):
    if not path.exists():raise RuntimeError('Missing output: '+str(path))
    with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    expected={(s,b,seed) for s in scenarios for b in buckets for seed in range(seed0,seed0+n)}
    keys=[]
    for r in rows:
        if r['mode']!='combined' or int(r['horizon'])!=horizon:
            raise RuntimeError('Wrong mode/horizon: '+str(path))
        k=(r['scenario'],int(r['bucket']),int(r['seed']))
        keys.append(k)
        if int(r['steps'])<1 or int(r['steps'])>500 or int(r['surv']) not in (0,1):
            raise RuntimeError('Bad steps/survival: '+str(k))
        if int(r['solver_failures'])<0:raise RuntimeError('Negative solver_failures')
    if len(keys)!=len(set(keys)) or set(keys)!=expected:
        raise RuntimeError(f'Keys mismatch {path}: found {len(keys)} expected {len(expected)}')
    return dict(episodes=len(rows),survival=sum(int(r['surv']) for r in rows)/len(rows),
                solver_failures=sum(int(r['solver_failures']) for r in rows),
                solver_optimal_inaccurate=sum(json.loads(r['solver_statuses']).get('optimal_inaccurate',0) for r in rows))


def command(ckpt, horizon, seed0, episodes, buckets, out):
    cmd=[sys.executable,SOURCE,'--ckpt',str(ckpt),'--horizon',str(horizon),
         '--seed0',str(seed0),'--episodes',str(episodes),
         '--buckets',*map(str,buckets),'--only',*SCENARIOS,
         '--modes','combined','--out',str(out)]
    for key,val in FROZEN.items():
        cmd.append('--'+key)
        cmd.extend(map(str,val)) if isinstance(val,list) else cmd.append(str(val))
    return cmd


def run(cmd, logfile, env):
    print('COMMAND:',subprocess.list2cmdline(cmd),flush=True)
    print('LOG:',logfile,flush=True)
    with open(logfile,'a',encoding='utf-8') as out:
        out.write('\n=== START '+time.strftime('%Y-%m-%d %H:%M:%S')+' ===\n')
        out.write(subprocess.list2cmdline(cmd)+'\n');out.flush()
        p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                           text=True,encoding='utf-8',errors='replace',env=env)
        assert p.stdout
        for line in p.stdout:
            print(line,end='',flush=True);out.write(line);out.flush()
        code=p.wait()
        out.write('=== RETURN CODE '+str(code)+' ===\n')
    if code:raise RuntimeError('Evaluation failed, exit code '+str(code))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=('smoke','full','all'),default='smoke')
    ap.add_argument('--ckpt',default='neural_mpc_multi13.pt')
    ap.add_argument('--out',default='ekf_confirm_78000')
    ap.add_argument('--allow_smoke_failures',action='store_true',help='Allow nonoptimal solver statuses during smoke; not for paper.')
    a=ap.parse_args()
    if not Path(SOURCE).is_file() or not Path(a.ckpt).is_file():
        ap.error('Run inside Three Models with evaluator and checkpoint available.')
    prefix=Path(a.out)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    manifest=Path(str(prefix)+'_frozen_manifest.json')
    settings=get_specs(prefix,a.ckpt)
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8'))!=settings:
            raise RuntimeError('Frozen manifest mismatch; DO NOT resume using changed source/checkpoint/config.')
    else:manifest.write_text(json.dumps(settings,indent=2,ensure_ascii=False),encoding='utf-8')

    # Validate the actual MPCSolver configuration BEFORE evaluating held-out data.
    try:
        import motorcycle_env_mpc17 as mod
        assert float(mod._MPC_EPS)==1e-4 and int(mod._MPC_MAX_ITER)==4000
        assert dict(mod._OSQP_EXTRA)=={} and mod._MPC_FALLBACK=='zero'
    except (ImportError,AttributeError,AssertionError,ValueError) as exc:
        raise RuntimeError('Cannot verify ORIGINAL OSQP configuration; stop before holdout.') from exc
    # Explicitly freeze all available BLAS thread pools for reproducibility.
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key]='1'
    smoke_out=Path(str(prefix)+'_smoke_N20')
    smoke_csv=Path(str(smoke_out)+'_episodes.csv')
    smoke_pass=Path(str(prefix)+'_smoke_gate.json')
    if a.phase in ('smoke','all'):
        run(command(a.ckpt,20,78000,1,[0],smoke_out),Path(str(prefix)+'_smoke.log'),env)
        sm=verify_csv(smoke_csv,1,20,78000,[0],SCENARIOS)
        gate=dict(pass_gate=True,checks=sm,source_hash=settings['source_hash'],ckpt_hash=settings['checkpoint_hash'])
        smoke_pass.write_text(json.dumps(gate,indent=2),encoding='utf-8')
        print('SMOKE PASS: structure and completeness verified. Survival is NOT a smoke gate.',flush=True)
    if a.phase in ('full','all'):
        if not smoke_pass.is_file():raise RuntimeError('Run --phase smoke first.')
        gate=json.loads(smoke_pass.read_text(encoding='utf-8'))
        if not gate.get('pass_gate') or gate['source_hash']!=settings['source_hash'] or gate['ckpt_hash']!=settings['checkpoint_hash']:
            raise RuntimeError('Stale smoke gate; config or checkpoint changed.')
        verify_csv(smoke_csv,1,20,78000,[0],SCENARIOS)
        summaries={}
        for h in HORIZONS:
            out=Path(str(prefix)+f'_N{h}')
            run(command(a.ckpt,h,78000,100,[0,1,2],out),Path(str(prefix)+f'_N{h}.log'),env)
            summaries[str(h)]=verify_csv(Path(str(out)+'_episodes.csv'),100,h,78000,[0,1,2],SCENARIOS)
            print('N',h,'PASS',summaries[str(h)],flush=True)
        Path(str(prefix)+'_completion.json').write_text(json.dumps({'complete':True,'per_horizon':summaries,'frozen_manifest':str(manifest)},indent=2),encoding='utf-8')
        print('FULL BENCHMARK COMPLETE, 6000 verified episode keys.',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
