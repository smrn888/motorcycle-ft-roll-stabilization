"""Generate paired multi-horizon MPC teacher data, excluding N=15.
Run alongside motorcycle_env_mpc17/18/19.py and motorcycle_dynamics.py.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from motorcycle_env_mpc19 import MotorcycleEnvMPC19
from motorcycle_dynamics import TAU_MAX

p=argparse.ArgumentParser()
p.add_argument('--episodes',type=int,default=5,help='Number of base seeds; each seed runs all horizons')
p.add_argument('--seed0',type=int,default=10000)
p.add_argument('--horizons',type=int,nargs='+',default=[10,20,30,40])
p.add_argument('--levels',type=int,nargs='+',default=[8])
p.add_argument('--out',default='multi_sanity.npz')
a=p.parse_args()
assert a.episodes>0 and a.levels and len(a.horizons)==len(set(a.horizons))
assert 15 not in a.horizons and min(a.horizons)>=2 and 20 in a.horizons
H=tuple(sorted(a.horizons))
class Env(MotorcycleEnvMPC19):
    HORIZONS=H
    MPC_N=20

def rollout(env,seed,bucket,level,N,perturb):
    env.set_fault()
    env.set_curriculum_level(level)
    env.reset(seed=seed)
    # Reset the selected horizon solver explicitly, including its u_prev.
    env._solvers[N].reset_warm_start()
    if bucket:
        env.state[0]=float(np.clip(env.state[0]+perturb[0],-.60,.60))
        env.state[1]=float(env.state[1]+perturb[1])
    k=env.HORIZONS.index(N)
    action=np.array([0.,0.,0.,2*(k+.5)/len(H)-1],np.float32)
    assert env.decode_action(action)[-1]==N
    data=[]
    for t in range(env.MAX_STEPS):
        phi_d=env._desired_roll()
        x=np.array([*env.state[:6],env.v,env.alpha,phi_d,env.state[0]-phi_d],np.float32)
        u_prev=env._u_prev_act.copy().astype(np.float32)
        # Teacher uses exact same model/constraints/cost as MPC19.step.
        _,_,terminated,truncated,info=env.step(action)
        u=env._u_prev_act.copy().astype(np.float32)
        # Store raw horizon N as eleventh input, not a normalized categorical index.
        xx=np.concatenate([x,u_prev,np.array([N],np.float32)])
        data.append((xx,u,seed,bucket,level,N,t,str(info['solve_status']),int(terminated)))
        if terminated or truncated:break
    return data

rng=np.random.default_rng(a.seed0)
# One independent environment per horizon, reused across seeds.
envs={N:Env(decision_k=1) for N in H}
rows=[]; metadata=[]; start=time.time()
for i in range(a.episodes):
    seed=a.seed0+i
    bucket=i%3
    level=a.levels[i%len(a.levels)]
    if bucket==0: perturb=(0.,0.)
    else:
        pr,dr={1:(.20,.50),2:(.50,1.20)}[bucket]
        perturb=(rng.uniform(-pr,pr),rng.uniform(-dr,dr))
    for N in H:
        r=rollout(envs[N],seed,bucket,level,N,perturb)
        rows.extend((i,*item) for item in r)
        metadata.append(dict(seed=seed,bucket=bucket,level=level,N=N,steps=len(r),crashed=bool(r[-1][-1])))
    if (i+1)%5==0 or i+1==a.episodes:
        print(f'base seeds {i+1}/{a.episodes}; samples={len(rows)}; {time.time()-start:.1f}s',flush=True)
X=np.stack([r[1] for r in rows]);Y=np.stack([r[2] for r in rows]);
EP=np.array([r[0] for r in rows],np.int32)
SEED=np.array([r[3] for r in rows],np.int32)
BUCKET=np.array([r[4] for r in rows],np.int32)
LEVEL=np.array([r[5] for r in rows],np.int32)
N=np.array([r[6] for r in rows],np.int32)
T=np.array([r[7] for r in rows],np.int32)
STATUS=np.array([r[8] for r in rows]); CRASH=np.array([r[9] for r in rows],np.int8)
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out,X=X,Y=Y,EP=EP,SEED=SEED,BUCKET=BUCKET,LEVEL=LEVEL,N=N,T=T,STATUS=STATUS,CRASH=CRASH)
with open(out.with_suffix('.json'),'w',encoding='utf-8') as f:json.dump(dict(args=vars(a),episodes=metadata),f,indent=2)
print('SANITY',X.shape,Y.shape,'nonfinite',int((~np.isfinite(X)).sum()+(~np.isfinite(Y)).sum()),'status',dict(zip(*np.unique(STATUS,return_counts=True))))
print('horizons',dict(zip(*np.unique(N,return_counts=True))))
assert X.shape[1]==13 and Y.shape[1]==2
print('saved',out)
