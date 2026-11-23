"""Paired 34-scenario fault benchmark: Pure SAC vs adaptive-horizon Residual SAC.
Run from Three Models project root. No training; no modification to source files.
"""
import argparse, csv, json, pickle, time
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC
from motorcycle_dynamics import TAU_MAX, MotorcycleDynamics
from motorcycle_env_residual_multi13 import MotorcycleEnvResidual
from final_fault_benchmark import SCENARIOS, measure, FaultedPlant

class ObservedFaultEnv(MotorcycleEnvResidual):
    scenario=('none', {})
    def reset(self, seed=None, options=None):
        self._meas_cache={}
        self._meas_rng=np.random.default_rng([int(seed),41])
        self._last_meas_phi=0.
        obs, info=super().reset(seed=seed,options=options)
        ft,fp=self.scenario
        iw=float(options.get('iw',1.))
        if ft=='param_fault':
            p=dict(MotorcycleDynamics().p)
            p['m']*=fp.get('m_scale',1.)
            p['h']*=fp.get('h_scale',1.)
            dyn=MotorcycleDynamics(params=p,m_scale=1.,Iw_scale=iw*fp.get('Iw_scale',1.))
        else:
            dyn=self._plant_dyn
        self._plant_dyn=RecordingPlant(dyn,ft,fp,TAU_MAX)
        return self._make_obs(),info
    def _obs10(self):
        k=int(self._steps)
        if k not in self._meas_cache:
            ft,fp=self.scenario
            s=measure(self.state[:6],k,ft,fp,self._meas_rng,self._last_meas_phi)
            self._last_meas_phi=float(s[0])
            phi_d=self._desired_roll()
            self._meas_cache[k]=np.array([*s,self.v,self.alpha,phi_d,s[0]-phi_d],np.float32)
        return self._meas_cache[k].copy()

class RecordingPlant(FaultedPlant):
    def __init__(self,*args):
        super().__init__(*args)
        self.last_cmd=np.zeros(2)
        self.last_app=np.zeros(2)
    def step(self,x,u):
        self.last_cmd=np.asarray(u,float).copy()
        self.last_app=np.clip(self._apply(self.last_cmd),-self.tau,self.tau)
        self.k+=1
        return self.dyn.step(x,self.last_app)

class Policy:
    def __init__(self,zip_file,pkl_file,dim,act_dim):
        self.model=SAC.load(str(zip_file),device='cpu')
        assert self.model.observation_space.shape==(dim,), (zip_file,self.model.observation_space)
        assert self.model.action_space.shape==(act_dim,), (zip_file,self.model.action_space)
        with open(pkl_file,'rb') as f: vn=pickle.load(f)
        assert vn.obs_rms.mean.shape==(dim,),pkl_file
        self.mean=np.asarray(vn.obs_rms.mean,float)
        self.var=np.asarray(vn.obs_rms.var,float)
        self.eps=float(vn.epsilon)
        self.clip=float(vn.clip_obs)
        self.count=float(vn.obs_rms.count)
        self.steps=int(self.model.num_timesteps)
    def __call__(self,obs):
        x=np.asarray(obs,float)
        assert x.shape==self.mean.shape
        x=np.clip((x-self.mean)/np.sqrt(self.var+self.eps),-self.clip,self.clip).astype(np.float32)
        return np.asarray(self.model.predict(x,deterministic=True)[0],float)

FIELDS=['controller','scenario','fault_type','bucket','seed','iw','surv','rmse','steps','u_rms','u2_sum','du','sat','peak_phi','wall_s']

def run(env,pol,kind,seed,bucket,iw):
    obs,_=env.reset(seed=seed,options={'bucket':bucket,'iw':iw})
    e2=0.; applied=[]; peak=0.; start=time.perf_counter()
    for _ in range(env.MAX_STEPS):
        if kind=='pure':
            action=pol(env._obs10())
            u=np.clip(action,-1,1)*TAU_MAX
        else:
            action=pol(obs)
            assert len(action)==3
            k=min(int((np.clip(action[2],-1,1)+1.)*.5*len(env.HORIZONS)),len(env.HORIZONS)-1)
            env._horizon=env.HORIZONS[k]
            xx=np.concatenate([env._obs10(),env._u_last,[env._horizon]]).astype(np.float32)
            base=env._nn_u(xx)
            u=np.clip(base+env.res_scale*np.clip(action[:2],-1,1),-TAU_MAX,TAU_MAX)
        env._steps+=1
        env.state=env._plant_dyn.step(env.state,u)
        env.v=float(np.clip(env.v+env.np_random.normal(0.,env.V_NOISE),1.,8.))
        env.alpha=float(np.clip(env.alpha+env.np_random.normal(0.,env.ALPHA_NOISE),-.20,.20))
        phi=float(env.state[0]); err=phi-env._desired_roll()
        e2+=err*err; peak=max(peak,abs(phi))
        applied.append(env._plant_dyn.last_app.copy())
        env._u_last=np.asarray(u,float).copy()
        crashed=abs(phi)>np.pi/4
        if crashed or env._steps>=env.MAX_STEPS: break
        if kind=='residual': obs=env._make_obs()
    ua=np.asarray(applied)
    return dict(surv=int(not crashed),rmse=float(np.sqrt(e2/len(ua))),steps=len(ua),
                u_rms=float(np.sqrt(np.mean(ua**2))),u2_sum=float(np.sum(ua**2)),
                du=float(np.linalg.norm(np.diff(ua,axis=0),axis=1).mean()) if len(ua)>1 else 0.,
                sat=float(np.mean(np.any(np.abs(ua)>=.99*TAU_MAX,axis=1))),
                peak_phi=float(np.rad2deg(peak)),wall_s=time.perf_counter()-start)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ckpt',default='neural_mpc_multi13.pt')
    p.add_argument('--pure_dir',default='standalone_runs')
    p.add_argument('--pure_seed',type=int,default=22)
    p.add_argument('--pure_variant',choices=['best','final'],default='best')
    p.add_argument('--res_stem',default='sac_seed202_continued_1m_final')
    p.add_argument('--res_scale',type=float,default=5.)
    p.add_argument('--skip_pure',action='store_true')
    p.add_argument('--episodes',type=int,default=5)
    p.add_argument('--seed0',type=int,default=400000)
    p.add_argument('--buckets',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--iw',type=float,default=1.)
    p.add_argument('--only',nargs='*',default=None)
    p.add_argument('--out',default='compare_34faults_1m')
    p.add_argument('--preflight_only',action='store_true')
    a=p.parse_args()
    if a.episodes<1 or any(b not in (0,1,2) for b in a.buckets): p.error('Invalid episodes or buckets')
    tag=f'{a.pure_variant}_sacplain_seed{a.pure_seed}'
    pure_zip=Path(a.pure_dir)/(tag+'.zip')
    pure_vn=Path(a.pure_dir)/('vecnorm_'+tag+'.pkl')
    res_zip=Path(a.res_stem+'.zip'); res_vn=Path(a.res_stem+'_vecnorm.pkl')
    for path in [pure_zip,pure_vn,res_zip,res_vn,Path(a.ckpt)]:
        if not path.is_file(): p.error('Missing file: '+str(path))
    pure=Policy(pure_zip,pure_vn,10,2)
    residual=Policy(res_zip,res_vn,15,3)
    print('PREFLIGHT PASS: Pure',pure.steps,'steps; Residual',residual.steps,'steps; VecNorm counts',pure.count,residual.count,flush=True)
    if pure.steps!=1000000 or residual.steps>1000000 or residual.steps<300000:
        print('WARNING: inspect checkpoint step counts before publication',flush=True)
    env=ObservedFaultEnv(ckpt=a.ckpt,decision_k=1,iw_list=(1.,),res_scale=a.res_scale)
    assert env.decision_k==1
    # Gate: exact same physical IC, reference and noise path for both controllers.
    for b in a.buckets:
        env.scenario=('none',{})
        x,_=env.reset(seed=a.seed0,options={'bucket':b,'iw':a.iw})
        initial=(env.state.copy(),env.v,env.alpha,env._desired_roll())
        y,_=env.reset(seed=a.seed0,options={'bucket':b,'iw':a.iw})
        np.testing.assert_allclose(initial[0],env.state,rtol=0,atol=0)
        assert initial[1:]==(env.v,env.alpha,env._desired_roll())
        np.testing.assert_allclose(x,y,rtol=0,atol=0)
    print('INITIAL STATE GATE PASS',flush=True)
    if a.preflight_only: return
    scenarios=[s for s in SCENARIOS if a.only is None or s[0] in a.only]
    assert scenarios
    output=Path(a.out+'_episodes.csv'); config=Path(a.out+'_config.json'); summary=Path(a.out+'_summary.csv')
    cfg={'args':{k:v for k,v in vars(a).items() if k!='preflight_only'},
         'files':{k:str(v.resolve()) for k,v in [('pure_zip',pure_zip),('pure_vn',pure_vn),('res_zip',res_zip),('res_vn',res_vn)]}}
    if config.exists():
        if json.loads(config.read_text(encoding='utf-8'))!=cfg: raise RuntimeError('Existing config differs; use new --out')
    elif output.exists(): raise RuntimeError('Existing episodes CSV without config')
    else: config.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    rows=[]; done=set()
    if output.exists():
        with output.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
        done={(r['controller'],r['scenario'],int(r['bucket']),int(r['seed'])) for r in rows}
    with output.open('a',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        if not done and not rows: writer.writeheader()
        for sid,ft,fp in scenarios:
            env.scenario=(ft,fp)
            for bucket in a.buckets:
                for seed in range(a.seed0,a.seed0+a.episodes):
                    for label,pol,kind in ([('residual_adaptive',residual,'residual')] if a.skip_pure else [('pure_sac',pure,'pure'),('residual_adaptive',residual,'residual')]):
                        key=(label,sid,bucket,seed)
                        if key in done: continue
                        result=run(env,pol,kind,seed,bucket,a.iw)
                        row=dict(controller=label,scenario=sid,fault_type=ft,bucket=bucket,seed=seed,iw=a.iw,**result)
                        writer.writerow(row); f.flush(); rows.append(row); done.add(key)
            sub=[r for r in rows if r['scenario']==sid]
            print(sid, ' '.join(f'{label}: survival={np.mean([int(r["surv"]) for r in sub if r["controller"]==label]):.3f}' for label in ['pure_sac','residual_adaptive']),flush=True)
    import pandas as pd
    df=pd.read_csv(output)
    agg=df.groupby(['controller','scenario'],as_index=False).agg(episodes=('surv','size'),survival=('surv','mean'),rmse=('rmse','mean'),u_rms=('u_rms','mean'),du=('du','mean'))
    agg.to_csv(summary,index=False)
    print('DONE:',len(df),'episodes;',output,summary,flush=True)

if __name__=='__main__': main()

