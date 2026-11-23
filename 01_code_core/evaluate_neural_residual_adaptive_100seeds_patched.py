"""Matched adaptive-horizon evaluation: fixed-N Neural MPC and 5 residual SAC training seeds.
Run from the project root, with compare_34faults_res_scale.py available.
Never retrains or edits project modules. Resume-safe CSV and config.
"""
import argparse, csv, json, time, hashlib
from pathlib import Path
import numpy as np
from motorcycle_dynamics import TAU_MAX
from compare_34faults_res_scale import ObservedFaultEnv, Policy
from final_fault_benchmark import SCENARIOS

DEFAULT_ONLY=['healthy','dropout_imu_p0.3','dropout_imu_p0.5','intermittent_imu_burst20','stuck_phi_15deg']
FIELDS=['controller','training_seed','checkpoint_stem','scenario','fault_type','bucket','seed','iw','horizon','surv','steps','rmse','u_rms','du','sat','peak_phi','mean_horizon','horizon_switches','wall_s']

def find_stem(seed):
    # Prefer converted NumPy1 evaluation checkpoints, when present.
    candidates=[]
    for pattern in [f'sac_seed{seed}*best_eval.zip',f'sac_seed{seed}*best.zip',f'sac_seed{seed}*final_eval.zip',f'sac_seed{seed}*final.zip']:
        candidates.extend(sorted(Path('.').glob(pattern)))
    valid=[]
    for z in candidates:
        stem=str(z)[:-4]
        if Path(stem+'_vecnorm.pkl').is_file():valid.append(stem)
    valid=list(dict.fromkeys(valid))
    if len(valid)==1:return valid[0]
    if not valid:raise FileNotFoundError(f'No matching ZIP + _vecnorm.pkl pair for training seed {seed}; use --res_stems')
    raise RuntimeError(f'Ambiguous checkpoints for seed {seed}: {valid}. Specify --res_stems explicitly.')

def run(env,pol,seed,bucket,iw,horizon,adaptive=False):
    env.reset(seed=seed,options={'bucket':bucket,'iw':iw})
    env._horizon=horizon
    env._u_last=np.zeros(2,dtype=float)
    e2=0.;applied=[];hs=[];peak=0.;crashed=False;start=time.perf_counter()
    for _ in range(env.MAX_STEPS):
        if pol is None:
            xx=np.concatenate([env._obs10(),env._u_last,[horizon]]).astype(np.float32)
            u=np.clip(env._nn_u(xx),-TAU_MAX,TAU_MAX)
        else:
            obs=env._make_obs()
            action=np.asarray(pol(obs),float)
            if action.shape!=(3,):raise RuntimeError(f'Expected residual action (3,), got {action.shape}')
            if adaptive:
                k=min(int((np.clip(action[2],-1,1)+1.)*.5*len(env.HORIZONS)),len(env.HORIZONS)-1)
                env._horizon=env.HORIZONS[k]
            hz=env._horizon if adaptive else horizon
            hs.append(hz)
            xx=np.concatenate([env._obs10(),env._u_last,[hz]]).astype(np.float32)
            base=env._nn_u(xx)
            u=np.clip(base+env.res_scale*np.clip(action[:2],-1,1),-TAU_MAX,TAU_MAX)
        env._steps+=1
        env.state=env._plant_dyn.step(env.state,u)
        env.v=float(np.clip(env.v+env.np_random.normal(0.,env.V_NOISE),1.,8.))
        env.alpha=float(np.clip(env.alpha+env.np_random.normal(0.,env.ALPHA_NOISE),-.20,.20))
        phi=float(env.state[0]);err=phi-env._desired_roll()
        e2+=err*err;peak=max(peak,abs(phi));applied.append(env._plant_dyn.last_app.copy())
        env._u_last=np.asarray(u,float).copy()
        crashed=abs(phi)>np.pi/4
        if crashed or env._steps>=env.MAX_STEPS:break
    ua=np.asarray(applied,float);n=len(ua)
    return dict(surv=int(not crashed),steps=n,rmse=float(np.sqrt(e2/n)),
        u_rms=float(np.sqrt(np.mean(ua**2))),
        du=float(np.linalg.norm(np.diff(ua,axis=0),axis=1).mean()) if n>1 else 0.,
        sat=float(np.mean(np.any(np.abs(ua)>=.99*TAU_MAX,axis=1))),
        peak_phi=float(np.rad2deg(peak)),mean_horizon=float(np.mean(hs)) if hs else float(horizon),
        horizon_switches=int(np.count_nonzero(np.diff(hs))) if len(hs)>1 else 0,
        wall_s=time.perf_counter()-start)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt',default='neural_mpc_multi13.pt')
    p.add_argument('--res_stems',nargs=5,help='Exactly 5 checkpoint stems in order of training seeds 101 202 303 404 505 (no extensions)')
    p.add_argument('--training_seeds',nargs=5,type=int,default=[101,202,303,404,505])
    p.add_argument('--seed0',type=int,default=77000)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--buckets',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--only',nargs='+',default=DEFAULT_ONLY)
    p.add_argument('--horizon',type=int,default=20)
    p.add_argument('--iw',type=float,default=1.)
    p.add_argument('--res_scale',type=float,default=5.)
    p.add_argument('--out',default='neural_residual_adaptive_77000_100seeds')
    p.add_argument('--preflight_only',action='store_true')
    p.add_argument('--adaptive',action='store_true',help='Enable SAC action[2] to select horizon; neural_mpc stays fixed N')
    a=p.parse_args()
    if a.horizon not in (10,20,30,40):p.error('Horizon must be in 10 20 30 40')
    if a.episodes<1 or len(set(a.buckets))!=len(a.buckets) or not set(a.buckets).issubset({0,1,2}):p.error('Invalid episodes/buckets')
    if len(set(a.training_seeds))!=5:p.error('Training seeds must be distinct')
    if not Path(a.ckpt).is_file():p.error('Missing neural checkpoint: '+a.ckpt)
    stems=a.res_stems or [find_stem(s) for s in a.training_seeds]
    for stem in stems:
        for suffix in ('.zip','_vecnorm.pkl'):
            if not Path(stem+suffix).is_file():p.error('Missing: '+stem+suffix)
    scenarios=[s for s in SCENARIOS if s[0] in a.only]
    if set(a.only)!={s[0] for s in scenarios}:p.error('Unknown scenario name(s)')
    policies={str(s):Policy(Path(stem+'.zip'),Path(stem+'_vecnorm.pkl'),15,3)
              for s,stem in zip(a.training_seeds,stems)}
    env=ObservedFaultEnv(ckpt=a.ckpt,decision_k=1,iw_list=(1.,),res_scale=a.res_scale)
    for b in a.buckets:
        env.scenario=('none',{})
        env.reset(seed=a.seed0,options={'bucket':b,'iw':a.iw})
        old=(env.state.copy(),env.v,env.alpha,env._desired_roll())
        env.reset(seed=a.seed0,options={'bucket':b,'iw':a.iw})
        np.testing.assert_array_equal(old[0],env.state)
        assert old[1:]==(env.v,env.alpha,env._desired_roll())
    print('PREFLIGHT OK; INITIAL STATE GATE PASS; N=',a.horizon,'res_scale=',a.res_scale,flush=True)
    print('CHECKPOINTS:',list(zip(a.training_seeds,stems)),flush=True)
    if a.preflight_only:return
    ep=Path(a.out+'_episodes.csv');cfgp=Path(a.out+'_config.json');summ=Path(a.out+'_summary.csv')
    cfg=dict(ckpt=str(Path(a.ckpt).resolve()),training_seeds=a.training_seeds,
        stems=[str(Path(s).resolve()) for s in stems],seed0=a.seed0,episodes=a.episodes,
        buckets=a.buckets,only=a.only,horizon=a.horizon,iw=a.iw,res_scale=a.res_scale,
        adaptive=a.adaptive,method=('frozen residual policy, adaptive horizon from action[2]; neural_mpc fixed N' if a.adaptive else 'frozen residual policy, horizon forced to N; residual action[2] ignored'))
    if cfgp.exists():
        if json.loads(cfgp.read_text(encoding='utf-8'))!=cfg:p.error('Existing config mismatch; choose a new --out')
    elif ep.exists():p.error('Episode CSV exists without config; choose new --out')
    else:cfgp.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    done=set();existing=0
    if ep.exists():
        with ep.open(newline='',encoding='utf-8') as f:
            for r in csv.DictReader(f):
                key=(r['controller'],r['scenario'],int(r['bucket']),int(r['seed']))
                if key in done:raise RuntimeError('Duplicate episode key '+str(key))
                done.add(key);existing+=1
    with ep.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS)
        if not existing:w.writeheader()
        for sid,ft,fp in scenarios:
            env.scenario=(ft,fp)
            for b in a.buckets:
                for seed in range(a.seed0,a.seed0+a.episodes):
                    for label,training_seed,stem,policy in [('neural_mpc', '', '', None)]+[
                        (('residual_adaptive_seed' if a.adaptive else 'residual_fixedN20_seed')+str(s),s,st,policies[str(s)]) for s,st in zip(a.training_seeds,stems)]:
                        key=(label,sid,b,seed)
                        if key in done:continue
                        d=run(env,policy,seed,b,a.iw,a.horizon,adaptive=a.adaptive and policy is not None)
                        w.writerow(dict(controller=label,training_seed=training_seed,checkpoint_stem=stem,
                            scenario=sid,fault_type=ft,bucket=b,seed=seed,iw=a.iw,horizon=a.horizon,**d))
                        f.flush();done.add(key)
                        print(f'{label} {sid} bucket={b} seed={seed} survival={d["surv"]} steps={d["steps"]}',flush=True)
    import pandas as pd
    df=pd.read_csv(ep)
    expected=len(scenarios)*len(a.buckets)*a.episodes*6
    if len(df)!=expected:raise RuntimeError(f'Expected {expected} episodes; found {len(df)}')
    if df.duplicated(['controller','scenario','bucket','seed']).any():raise RuntimeError('Duplicate keys')
    df.groupby(['controller','scenario','bucket'],as_index=False).agg(
        episodes=('surv','size'),survival=('surv','mean'),steps=('steps','mean'),
        rmse=('rmse','mean'),u_rms=('u_rms','mean'),du=('du','mean'),sat=('sat','mean'),
        mean_horizon=('mean_horizon','mean'),horizon_switches=('horizon_switches','mean')).to_csv(summ,index=False)
    print('DONE',len(df),'episodes. Files:',ep,summ,cfgp,flush=True)
if __name__=='__main__':main()
