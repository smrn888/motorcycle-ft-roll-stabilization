"""Online MPC with nominal-model EKF on matched fault scenarios, evaluation seeds 77000..77099.

Run from Three Models root alongside compare_34faults_res_scale.py and
compare_online_mpc_vs_neural_vs_residual.py. No training or modification to the existing baselines.
Experimental EKF+fixed-weight/fixed-horizon MPC, NOT certified distillation teacher.
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
FIELDS=['controller','scenario','fault_type','bucket','seed','iw','surv','steps','rmse','u_rms','du','sat','peak_phi','solver_calls','solver_failures','solver_statuses','solver_ms_mean','wall_s','ekf_ms_mean','ekf_rejections','ekf_roll_rmse']

class SixStateEKF:
    """Causal, nominal-model EKF. No plant-state access or fault oracle.

    x: [phi, phi_dot, delta_f, delta_r, delta_dot_f, delta_dot_r].
    Sensors are the six channels returned by ObservedFaultEnv._obs10().
    Roll-pair innovation gating is a declared, non-oracle robustness option.
    """
    def __init__(self, dyn, process_std, measure_std, initial_std, gate):
        self.dyn = dyn
        self.Q = np.diag(np.square(np.asarray(process_std,float)))
        self.R = np.diag(np.square(np.asarray(measure_std,float)))
        self.P0 = np.diag(np.square(np.asarray(initial_std,float)))
        self.gate = float(gate)
        self.x = np.zeros(6)
        self.P = self.P0.copy()
        self.rejected = 0
        self.updates = 0

    def reset(self, y):
        # Initial estimate comes from sensor outputs, never from env.state.
        self.x = np.asarray(y,float).copy()
        if self.x.shape != (6,) or not np.all(np.isfinite(self.x)):
            raise ValueError('Invalid six-channel initialization measurement')
        self.P = self.P0.copy()
        self.rejected = self.updates = 0

    def predict(self, u):
        # The nominal discrete Jacobian matches the nominal-model MPC linearization.
        F = self.dyn.linearise(float(self.x[0]))[0]
        self.x = self.dyn.step(self.x.copy(), np.asarray(u,float))
        self.P = F @ self.P @ F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)

    def correct(self, y):
        y = np.asarray(y,float)
        if y.shape != (6,):
            raise ValueError('Expected 6 sensor channels')
        # Roll/roll-rate may be corrupted. Other 4 gimbal channels remain usable.
        accepted = [2,3,4,5]
        roll = [0,1]
        innovation = y[roll] - self.x[roll]
        Sroll = self.P[np.ix_(roll,roll)] + self.R[np.ix_(roll,roll)]
        if np.all(np.isfinite(innovation)):
            try:
                nis = float(innovation @ np.linalg.solve(Sroll, innovation))
                if nis <= self.gate:
                    accepted = roll + accepted
                else:
                    self.rejected += 1
            except np.linalg.LinAlgError:
                self.rejected += 1
        else:
            self.rejected += 1
        idx = np.asarray([i for i in accepted if np.isfinite(y[i])], dtype=int)
        if len(idx) == 0:
            return
        H = np.eye(6)[idx]
        innovation = y[idx] - H @ self.x
        R = self.R[np.ix_(idx,idx)]
        S = H @ self.P @ H.T + R
        # Solve rather than explicitly invert covariance.
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ innovation
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form
        self.P = 0.5 * (self.P + self.P.T)
        self.updates += 1


def rollout(env,seed,bucket,iw,horizon,weights,ekf_cfg):
    env.reset(seed=seed,options={'bucket':bucket,'iw':iw})
    env._u_last=np.zeros(2,dtype=float)
    env._horizon=horizon
    solver=env._solvers[horizon]
    solver.reset_warm_start()
    q=np.array([weights[0],weights[1],weights[2],weights[2],0.1*weights[2],0.1*weights[2]],float)
    r=np.array([weights[3],weights[3]],float)
    ekf=SixStateEKF(env.dyn,ekf_cfg['process_std'],ekf_cfg['measure_std'],ekf_cfg['initial_std'],ekf_cfg['gate'])
    # Deliberately uses the faulted observation path; no truth-state initialization.
    ekf.reset(np.asarray(env._obs10()[:6],float))
    statuses=Counter();fail=0;ms_sum=0.;applied=[];e2=0.;peak=0.;crashed=False
    ekf_ms_sum=0.;esterr2=0.
    start=time.perf_counter()
    for k in range(env.MAX_STEPS):
        if k:
            t_ekf=time.perf_counter()
            ekf.correct(np.asarray(env._obs10()[:6],float))
            ekf_ms_sum+=(time.perf_counter()-t_ekf)*1000.
        x=ekf.x.copy()
        # Ground-truth comparison below is diagnostic ONLY, never given to controller.
        esterr2+=float((x[0]-env.state[0])**2)
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
        t_ekf=time.perf_counter()
        ekf.predict(u)
        ekf_ms_sum+=(time.perf_counter()-t_ekf)*1000.
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
        ekf_ms_mean=ekf_ms_sum/calls, ekf_rejections=ekf.rejected,
        ekf_roll_rmse=float(np.sqrt(esterr2/calls)),
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
    p.add_argument('--ekf_gate',type=float,default=9.21,help='Roll-pair NIS threshold, 9.21 ~= chi-square(2) 99th percentile')
    p.add_argument('--ekf_q',type=float,nargs=6,default=[.002,.05,.005,.005,.10,.10],help='Nominal EKF one-step process std for six state channels')
    p.add_argument('--ekf_r',type=float,nargs=6,default=[.01,.10,.005,.005,.05,.05],help='Observation std for six channels')
    p.add_argument('--ekf_p0',type=float,nargs=6,default=[.05,.5,.05,.05,.5,.5],help='Initial state std for six channels')
    p.add_argument('--out',default='online_mpc_ekf_77000_100seeds')
    p.add_argument('--preflight_only',action='store_true')
    a=p.parse_args()
    if a.ekf_gate<=0 or any(v<=0 for v in (a.ekf_q+a.ekf_r+a.ekf_p0)):
        p.error('EKF std and gate must all be positive')
    ekf_cfg={'process_std':a.ekf_q,'measure_std':a.ekf_r,'initial_std':a.ekf_p0,'gate':a.ekf_gate}
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
    # Basic standalone EKF predict/correct shape, finite value and covariance checks.
    test=SixStateEKF(env.dyn,**ekf_cfg)
    env.scenario=('none',{})
    env.reset(seed=a.seed0,options={'bucket':a.buckets[0],'iw':a.iw})
    test.reset(np.asarray(env._obs10()[:6],float))
    test.predict(np.zeros(2))
    test.correct(np.asarray(env._obs10()[:6],float))
    assert np.all(np.isfinite(test.x)) and np.all(np.linalg.eigvalsh(test.P)>-1e-10)
    print('EKF PREFLIGHT PASS; no true state used in EKF initialization or corrections',flush=True)
    if a.preflight_only:return
    path=Path(a.out+'_episodes.csv');summary=Path(a.out+'_summary.csv');cfg_path=Path(a.out+'_config.json')
    cfg=dict(seed0=a.seed0,episodes=a.episodes,buckets=a.buckets,only=a.only,horizon=a.horizon,
             iw=a.iw,ekf=ekf_cfg,ckpt=str(Path(a.ckpt).resolve()),weights=weights.tolist(),
             controller='online_mpc_ekf',fallback='zero',
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
                    result=rollout(env,seed,bucket,a.iw,a.horizon,weights,ekf_cfg)
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
        solver_failures=('solver_failures','sum'),ekf_rejections=('ekf_rejections','sum'),
        ekf_ms_mean=('ekf_ms_mean','mean'),ekf_roll_rmse=('ekf_roll_rmse','mean')).to_csv(summary,index=False)
    print('DONE',len(df),'episodes; survival=',df.surv.mean(),'files:',path,summary,cfg_path,flush=True)

if __name__=='__main__':main()
