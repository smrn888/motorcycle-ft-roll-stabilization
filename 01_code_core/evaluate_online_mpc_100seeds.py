"""Online measured-state MPC on matched fault scenarios, evaluation seeds 77000..77099.

Run from Three Models root alongside compare_34faults_res_scale.py and
compare_online_mpc_vs_neural_vs_residual.py. No training or source modification.
Fixed-weight/fixed-horizon exploratory online MPC, NOT certified distillation teacher.
Outputs resumable episode CSV, summary CSV, and immutable config JSON.
"""
import argparse, csv, json, time, hashlib
from collections import Counter
from pathlib import Path
import numpy as np
from motorcycle_dynamics import TAU_MAX
from compare_34faults_res_scale import ObservedFaultEnv
from compare_online_mpc_vs_neural_vs_residual import setup_online
from final_fault_benchmark import SCENARIOS

DEFAULT_SCENARIOS=['healthy','dropout_imu_p0.3','dropout_imu_p0.5','intermittent_imu_burst20','stuck_phi_15deg']
FIELDS=['controller','scenario','fault_type','bucket','seed','iw','surv','steps','rmse','u_rms','du','sat','peak_phi','solver_calls','solver_failures','solver_statuses','solver_ms_mean','wall_s']

def rollout(env,seed,bucket,iw,horizon,weights,oracle=False):
    env.reset(seed=seed,options={'bucket':bucket,'iw':iw})
    env._u_last=np.zeros(2,dtype=float)
    env._horizon=horizon
    solver=env._solvers[horizon]
    solver.reset_warm_start()
    q=np.array([weights[0],weights[1],weights[2],weights[2],0.1*weights[2],0.1*weights[2]],float)
    r=np.array([weights[3],weights[3]],float)
    statuses=Counter();fail=0;ms_sum=0.;applied=[];e2=0.;peak=0.;crashed=False
    start=time.perf_counter()
    for _ in range(env.MAX_STEPS):
        x=np.asarray(env.state[:6] if oracle else env._obs10()[:6],float).copy()
        xd=np.zeros(6);xd[0]=env._desired_roll()
        t0=time.perf_counter()
        try:
            A,B,bias=env.dyn.linearise(float(x[0]))
            solver.update_model(A,B,bias)
            pc=env._dare_chol(A,B,Q_diag=q,R_diag=r)
            solver.u_prev.value=env._u_last.copy()
            u=np.asarray(solver.solve(x0=x,xd=xd,Q_diag=q,R_diag=r,P_chol=pc),float)
            status=str(solver.last_solve_status)
        except Exception as exc:
            u=np.zeros(2);status='pre_solve_exception:'+type(exc).__name__
        ms_sum+=(time.perf_counter()-t0)*1000
        valid=status in ('optimal','optimal_inaccurate') and u.shape==(2,) and np.all(np.isfinite(u))
        statuses[status]+=1
        if not valid:
            fail+=1
            u=np.zeros(2)
        u=np.clip(u,-TAU_MAX,TAU_MAX)
        env._steps+=1
        env.state=env._plant_dyn.step(env.state,u)
        env.v=float(np.clip(env.v+env.np_random.normal(0.,env.V_NOISE),1.,8.))
        env.alpha=float(np.clip(env.alpha+env.np_random.normal(0.,env.ALPHA_NOISE),-.20,.20))
        phi=float(env.state[0]);err=phi-env._desired_roll();e2+=err*err
        peak=max(peak,abs(phi));applied.append(env._plant_dyn.last_app.copy())
        env._u_last=u.copy()
        crashed=abs(phi)>np.pi/4
        if crashed or env._steps>=env.MAX_STEPS:break
    ua=np.asarray(applied,float);calls=len(ua)
    return dict(surv=int(not crashed),steps=calls,rmse=float(np.sqrt(e2/calls)),
        u_rms=float(np.sqrt(np.mean(ua**2))),
        du=float(np.linalg.norm(np.diff(ua,axis=0),axis=1).mean()) if calls>1 else 0.,
        sat=float(np.mean(np.any(np.abs(ua)>=.99*TAU_MAX,axis=1))),
        peak_phi=float(np.rad2deg(peak)),solver_calls=calls,solver_failures=fail,
        solver_statuses=json.dumps(statuses,sort_keys=True),solver_ms_mean=ms_sum/calls,
        wall_s=time.perf_counter()-start)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt',default='neural_mpc_multi13.pt')
    p.add_argument('--seed0',type=int,default=77000)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--buckets',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--only',nargs='+',default=DEFAULT_SCENARIOS)
    p.add_argument('--horizon',type=int,default=20)
    p.add_argument('--iw',type=float,default=1.)
    p.add_argument('--oracle',action='store_true',help='Diagnostic true-state input; NOT deployable')
    p.add_argument('--out',default='online_mpc_77000_100seeds')
    p.add_argument('--preflight_only',action='store_true')
    a=p.parse_args()
    if a.episodes<1 or len(set(a.buckets))!=len(a.buckets) or not set(a.buckets).issubset({0,1,2}):p.error('Invalid episodes/buckets')
    if not Path(a.ckpt).is_file():p.error('Missing neural checkpoint: '+a.ckpt)
    scenarios=[s for s in SCENARIOS if s[0] in a.only]
    missing=set(a.only)-{s[0] for s in scenarios}
    if missing:p.error('Unknown scenario(s): '+str(sorted(missing)))
    env=ObservedFaultEnv(ckpt=a.ckpt,decision_k=1,iw_list=(1.,),res_scale=5.)
    weights,alpha,decoded=setup_online(env,a.horizon)
    print('PREFLIGHT OK; weights=',weights,'fixed N=',a.horizon,'zero-action decoded N=',decoded,'LP_ALPHA=',alpha,flush=True)
    for b in a.buckets:
        env.scenario=('none',{})
        env.reset(seed=a.seed0,options={'bucket':b,'iw':a.iw})
        state=(env.state.copy(),env.v,env.alpha,env._desired_roll())
        env.reset(seed=a.seed0,options={'bucket':b,'iw':a.iw})
        np.testing.assert_array_equal(state[0],env.state)
        assert state[1:]==(env.v,env.alpha,env._desired_roll())
    print('INITIAL STATE GATE PASS',flush=True)
    if a.preflight_only:return
    path=Path(a.out+'_episodes.csv');summary=Path(a.out+'_summary.csv');cfg_path=Path(a.out+'_config.json')
    cfg=dict(seed0=a.seed0,episodes=a.episodes,buckets=a.buckets,only=a.only,horizon=a.horizon,
             iw=a.iw,oracle=a.oracle,ckpt=str(Path(a.ckpt).resolve()),weights=weights.tolist(),
             controller='online_mpc_oracle' if a.oracle else 'online_mpc_measured',fallback='zero',
             caveat='Exploratory fixed-weight, fixed-horizon MPC; not proven exact distillation teacher')
    if cfg_path.exists():
        if json.loads(cfg_path.read_text(encoding='utf-8'))!=cfg:
            p.error('Config mismatch; choose new --out')
    elif path.exists():p.error('Existing episode CSV without config; choose new --out')
    else:cfg_path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    existing=[];done=set()
    if path.exists():
        with path.open(newline='',encoding='utf-8') as f:
            existing=list(csv.DictReader(f))
        for row in existing:
            key=(row['scenario'],int(row['bucket']),int(row['seed']))
            if key in done:raise RuntimeError('Duplicate existing episode key: '+str(key))
            done.add(key)
    label=cfg['controller']
    with path.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS)
        if not existing:w.writeheader()
        for sid,ft,fp in scenarios:
            env.scenario=(ft,fp)
            for bucket in a.buckets:
                for seed in range(a.seed0,a.seed0+a.episodes):
                    key=(sid,bucket,seed)
                    if key in done:continue
                    result=rollout(env,seed,bucket,a.iw,a.horizon,weights,oracle=a.oracle)
                    w.writerow(dict(controller=label,scenario=sid,fault_type=ft,bucket=bucket,seed=seed,iw=a.iw,**result))
                    f.flush();done.add(key)
                    print(f'{sid} bucket={bucket} seed={seed} survival={result["surv"]} steps={result["steps"]} solver_fail={result["solver_failures"]}',flush=True)
    import pandas as pd
    df=pd.read_csv(path)
    expected=len(scenarios)*len(a.buckets)*a.episodes
    if len(df)!=expected:raise RuntimeError(f'Expected {expected} rows, found {len(df)}')
    if df.duplicated(['scenario','bucket','seed']).any():raise RuntimeError('Duplicate keys')
    df.groupby(['controller','scenario','bucket'],as_index=False).agg(
        episodes=('surv','size'),survival=('surv','mean'),steps=('steps','mean'),
        rmse=('rmse','mean'),u_rms=('u_rms','mean'),du=('du','mean'),
        solver_failures=('solver_failures','sum')).to_csv(summary,index=False)
    print('DONE',len(df),'episodes; survival=',df.surv.mean(),'files:',path,summary,cfg_path,flush=True)

if __name__=='__main__':main()
