"""Paired EKF ablation for Online MPC: baseline / stale / recovery / combined.

Place next to evaluate_online_mpc_ekf_100seeds.py in the Three Models root.
No fault flags or true plant state are used by the filter; plant truth is used only for metrics.
Each variant resets on the same (scenario, bucket, seed) and uses the original MPC weights.
"""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np

from motorcycle_dynamics import TAU_MAX
from compare_34faults_res_scale import ObservedFaultEnv
from compare_online_mpc_vs_neural_vs_residual import setup_online
from final_fault_benchmark import SCENARIOS
from evaluate_online_mpc_ekf_100seeds import SixStateEKF

MODES = ('baseline', 'stale', 'recovery', 'combined')
DEFAULT_ONLY = ['healthy', 'dropout_imu_p0.5', 'intermittent_imu_burst20']
FIELDS = ['controller','mode','scenario','fault_type','bucket','seed','iw','horizon',
          'surv','steps','rmse','u_rms','du','sat','peak_phi', 'solver_calls',
          'solver_failures','solver_statuses','solver_ms_mean','wall_s',
          'ekf_ms_mean','ekf_rejections','ekf_roll_rmse',
          'stale_detections','recovery_events','max_reject_streak']

class RobustEKF(SixStateEKF):
    def __init__(self, dyn, process_std, measure_std, initial_std, gate,
                 mode='baseline', stale_phi_tol=1e-10, stale_rate_tol=1e-10,
                 stale_run=2, recovery_run=5, recovery_factor=4.0, recovery_cap=1.0):
        super().__init__(dyn, process_std, measure_std, initial_std, gate)
        self.mode = mode
        self.stale_phi_tol = stale_phi_tol
        self.stale_rate_tol = stale_rate_tol
        self.stale_run = stale_run
        self.recovery_run = recovery_run
        self.recovery_factor = recovery_factor
        self.recovery_cap = recovery_cap
        self.last_measurement = None
        self.stale_count = 0
        self.reject_streak = 0
        self.max_reject_streak = 0
        self.stale_detections = 0
        self.recovery_events = 0
        self.last_nis = float('nan')
        self.last_roll_accepted = False

    def reset(self, y):
        super().reset(y)
        self.last_measurement = np.asarray(y, float).copy()
        self.stale_count = self.reject_streak = self.max_reject_streak = 0
        self.stale_detections = self.recovery_events = 0
        self.last_nis = float('nan')
        self.last_roll_accepted = False

    def correct(self, y):
        y = np.asarray(y, float)
        if y.shape != (6,):
            raise ValueError('Expected six sensor channels')
        use_stale = self.mode in ('stale', 'combined')
        use_recovery = self.mode in ('recovery', 'combined')
        # A conservative *candidate* detector, not a perfect dropout detector:
        # exact repeat of roll angle PLUS near-zero reported roll rate.
        prev = self.last_measurement
        suspicious = bool(use_stale and prev is not None and
                          np.isfinite(y[:2]).all() and np.isfinite(prev[0]) and
                          abs(y[0] - prev[0]) <= self.stale_phi_tol and
                          abs(y[1]) <= self.stale_rate_tol)
        self.stale_count = self.stale_count + 1 if suspicious else 0
        stale = bool(use_stale and self.stale_count >= self.stale_run)
        self.last_measurement = y.copy()
        if stale:
            self.stale_detections += 1

        roll = [0, 1]
        accepted = [2, 3, 4, 5]
        innovation = y[roll] - self.x[roll]
        Sroll = self.P[np.ix_(roll, roll)] + self.R[np.ix_(roll, roll)]
        try:
            nis = float(innovation @ np.linalg.solve(Sroll, innovation)) if np.isfinite(innovation).all() else np.inf
        except np.linalg.LinAlgError:
            nis = np.inf
        self.last_nis = nis
        roll_ok = bool(not stale and np.isfinite(nis) and nis <= self.gate)
        self.last_roll_accepted = roll_ok
        if roll_ok:
            accepted = roll + accepted
            self.reject_streak = 0
        else:
            self.rejected += 1
            self.reject_streak += 1
            self.max_reject_streak = max(self.max_reject_streak, self.reject_streak)
            # Inflate uncertainty after repeated rejection; do NOT force acceptance.
            # Increased roll covariance makes eventual statistically consistent
            # reacquisition more likely. Bound to avoid runaway uncertainty.
            if (use_recovery and self.reject_streak >= self.recovery_run and
                self.reject_streak % self.recovery_run == 0):
                # Scale the roll-related rows/columns by sqrt(factor) to preserve PSD.
                factor = self.recovery_factor
                scale = np.ones(6)
                scale[:2] = np.sqrt(factor)
                self.P = (scale[:,None] * self.P) * scale[None,:]
                # Bound roll variances by a congruent rescaling (retains PSD).
                for i in roll:
                    if self.P[i,i] > self.recovery_cap:
                        s = np.sqrt(self.recovery_cap / self.P[i,i])
                        self.P[i,:] *= s
                        self.P[:,i] *= s
                self.P = (self.P + self.P.T) / 2
                self.recovery_events += 1
        idx = np.asarray([i for i in accepted if np.isfinite(y[i])], dtype=int)
        if len(idx) == 0:
            return
        H = np.eye(6)[idx]
        v = y[idx] - H @ self.x
        R = self.R[np.ix_(idx,idx)]
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ v
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = (self.P + self.P.T) / 2
        self.updates += 1


def rollout(env, seed, bucket, iw, horizon, weights, cfg, mode):
    env.reset(seed=seed, options={'bucket':bucket, 'iw':iw})
    env._u_last = np.zeros(2)
    env._horizon = horizon
    solver = env._solvers[horizon]
    solver.reset_warm_start()
    q = np.array([weights[0],weights[1],weights[2],weights[2],.1*weights[2],.1*weights[2]])
    r = np.array([weights[3],weights[3]])
    ekf = RobustEKF(env.dyn, mode=mode, **cfg)
    ekf.reset(np.asarray(env._obs10()[:6],float))
    statuses=Counter(); fail=0; ms_sum=0.; ekf_ms=0.; applied=[]; e2=0.; est_e2=0.; peak=0.; crashed=False
    start=time.perf_counter()
    for k in range(env.MAX_STEPS):
        if k:
            t=time.perf_counter()
            ekf.correct(np.asarray(env._obs10()[:6],float))
            ekf_ms += (time.perf_counter()-t)*1000.
        x = ekf.x.copy()
        est_e2 += float((x[0]-env.state[0])**2) # metrics ONLY
        xd = np.zeros(6); xd[0]=env._desired_roll()
        t=time.perf_counter()
        try:
            A,B,bias=env.dyn.linearise(float(x[0]))
            solver.update_model(A,B,bias)
            pc=env._dare_chol(A,B,Q_diag=q,R_diag=r)
            solver.u_prev.value=env._u_last.copy()
            u=np.asarray(solver.solve(x0=x,xd=xd,Q_diag=q,R_diag=r,P_chol=pc),float)
            status=str(solver.last_solve_status)
        except Exception as exc:
            u=np.zeros(2); status='pre_solve_exception:'+type(exc).__name__
        ms_sum += (time.perf_counter()-t)*1000.
        valid=status in ('optimal','optimal_inaccurate') and u.shape==(2,) and np.all(np.isfinite(u))
        statuses[status]+=1
        if not valid:
            fail+=1;u=np.zeros(2)
        u=np.clip(u,-TAU_MAX,TAU_MAX)
        t=time.perf_counter();ekf.predict(u);ekf_ms+=(time.perf_counter()-t)*1000.
        env._steps+=1
        env.state=env._plant_dyn.step(env.state,u)
        env.v=float(np.clip(env.v+env.np_random.normal(0.,env.V_NOISE),1.,8.))
        env.alpha=float(np.clip(env.alpha+env.np_random.normal(0.,env.ALPHA_NOISE),-.20,.20))
        phi=float(env.state[0]); err=phi-env._desired_roll();e2+=err*err
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
        ekf_ms_mean=ekf_ms/calls,ekf_rejections=ekf.rejected,
        ekf_roll_rmse=float(np.sqrt(est_e2/calls)),wall_s=time.perf_counter()-start,
        stale_detections=ekf.stale_detections,recovery_events=ekf.recovery_events,
        max_reject_streak=ekf.max_reject_streak)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt',default='neural_mpc_multi13.pt')
    p.add_argument('--seed0',type=int,default=77000)
    p.add_argument('--episodes',type=int,default=5)
    p.add_argument('--buckets',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--only',nargs='+',default=DEFAULT_ONLY)
    p.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    p.add_argument('--horizon',type=int,default=20)
    p.add_argument('--iw',type=float,default=1.)
    p.add_argument('--ekf_gate',type=float,default=9.21)
    p.add_argument('--ekf_q',nargs=6,type=float,default=[.002,.05,.005,.005,.10,.10])
    p.add_argument('--ekf_r',nargs=6,type=float,default=[.01,.10,.005,.005,.05,.05])
    p.add_argument('--ekf_p0',nargs=6,type=float,default=[.05,.5,.05,.05,.5,.5])
    p.add_argument('--stale_phi_tol',type=float,default=1e-10)
    p.add_argument('--stale_rate_tol',type=float,default=1e-10)
    p.add_argument('--stale_run',type=int,default=2)
    p.add_argument('--recovery_run',type=int,default=5)
    p.add_argument('--recovery_factor',type=float,default=4.)
    p.add_argument('--recovery_cap',type=float,default=1.)
    p.add_argument('--preflight_only',action='store_true')
    p.add_argument('--out',default='ekf_ablation_pilot_N20')
    a=p.parse_args()
    if (a.episodes<1 or a.horizon not in (10,20,30,40) or not a.buckets or
        not set(a.buckets).issubset({0,1,2}) or len(a.buckets)!=len(set(a.buckets)) or
        len(a.modes)!=len(set(a.modes)) or a.ekf_gate<=0 or
        min(a.ekf_q+a.ekf_r+a.ekf_p0)<=0 or a.stale_run<1 or
        a.recovery_run<1 or a.recovery_factor<=1 or a.recovery_cap<=0 or
        a.stale_phi_tol<0 or a.stale_rate_tol<0):
        p.error('Invalid parameter(s)')
    if not Path(a.ckpt).is_file():p.error('Checkpoint not found: '+a.ckpt)
    scenarios=[s for s in SCENARIOS if s[0] in a.only]
    if set(a.only)!={s[0] for s in scenarios}:p.error('Unknown scenario')
    cfg=dict(process_std=a.ekf_q,measure_std=a.ekf_r,initial_std=a.ekf_p0,gate=a.ekf_gate,
             stale_phi_tol=a.stale_phi_tol,stale_rate_tol=a.stale_rate_tol,
             stale_run=a.stale_run,recovery_run=a.recovery_run,
             recovery_factor=a.recovery_factor,recovery_cap=a.recovery_cap)
    env=ObservedFaultEnv(ckpt=a.ckpt,decision_k=1,iw_list=(1.,),res_scale=5.)
    weights,alpha,decoded=setup_online(env,a.horizon)
    # Offline unit gates; this must not access the true state in EKF.
    t=RobustEKF(env.dyn,mode='combined',**cfg)
    sample=np.zeros(6)
    t.reset(sample)
    t.predict(np.zeros(2));t.correct(sample)
    assert np.isfinite(t.x).all() and np.linalg.eigvalsh(t.P).min()>-1e-9
    for _ in range(a.stale_run):t.correct(sample)
    assert t.stale_detections>=1, 'Stale detector unit gate failed'
    print('PREFLIGHT PASS | fixed N=',a.horizon,'weights=',weights, 'modes=',a.modes,flush=True)
    if a.preflight_only:return
    path=Path(a.out+'_episodes.csv');summary=Path(a.out+'_summary.csv');config=Path(a.out+'_config.json')
    settings=dict(seed0=a.seed0,episodes=a.episodes,buckets=a.buckets,only=a.only,modes=a.modes,
                  horizon=a.horizon,iw=a.iw,ckpt=str(Path(a.ckpt).resolve()),weights=weights.tolist(),ekf=cfg,
                  note='Experimental ablations; no oracle information provided to EKF')
    if config.exists():
        if json.loads(config.read_text(encoding='utf-8')) != settings:
            p.error('Config mismatch: choose a new --out')
    elif path.exists():p.error('Existing CSV without config; choose new --out')
    else:config.write_text(json.dumps(settings,indent=2,ensure_ascii=False),encoding='utf-8')
    done=set()
    if path.exists():
        with path.open(newline='',encoding='utf-8') as f:
            for row in csv.DictReader(f):
                key=(row['mode'],row['scenario'],int(row['bucket']),int(row['seed']))
                if key in done:raise RuntimeError('Duplicate row '+str(key))
                done.add(key)
    with path.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS)
        if not done:w.writeheader()
        for sid,ft,fp in scenarios:
            for bucket in a.buckets:
                for seed in range(a.seed0,a.seed0+a.episodes):
                    for mode in a.modes:
                        key=(mode,sid,bucket,seed)
                        if key in done:continue
                        env.scenario=(ft,fp)
                        result=rollout(env,seed,bucket,a.iw,a.horizon,weights,cfg,mode)
                        w.writerow(dict(controller='online_mpc_ekf_'+mode,mode=mode,
                                        scenario=sid,fault_type=ft,bucket=bucket,seed=seed,iw=a.iw,
                                        horizon=a.horizon,**result))
                        f.flush();done.add(key)
                        print(f'{mode} {sid} bucket={bucket} seed={seed} surv={result["surv"]} steps={result["steps"]} reject={result["ekf_rejections"]} stale={result["stale_detections"]} recovery={result["recovery_events"]}',flush=True)
    import pandas as pd
    df=pd.read_csv(path)
    expected=len(scenarios)*len(a.buckets)*a.episodes*len(a.modes)
    if len(df)!=expected:raise RuntimeError(f'Expected {expected} rows; got {len(df)}')
    if df.duplicated(['mode','scenario','bucket','seed']).any():raise RuntimeError('Duplicate keys')
    df.groupby(['mode','scenario'],as_index=False).agg(
        episodes=('surv','size'),survival=('surv','mean'),steps=('steps','mean'),
        rmse=('rmse','mean'),ekf_roll_rmse=('ekf_roll_rmse','mean'),
        ekf_rejections=('ekf_rejections','mean'),stale_detections=('stale_detections','mean'),
        recovery_events=('recovery_events','mean'),solver_failures=('solver_failures','sum'),
        ekf_ms_mean=('ekf_ms_mean','mean')).to_csv(summary,index=False)
    print('DONE',len(df),'rows:',path,summary,config,flush=True)

if __name__=='__main__':main()
