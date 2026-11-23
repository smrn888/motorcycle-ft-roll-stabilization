"""Exploratory three-controller fault ablation. Run from the Three Models root.

Online MPC uses the nominal fixed-weight/fixed-N=20 setting shown in
final_fault_benchmark.py. This is NOT automatically the exact distillation
teacher; establish training-target provenance before claiming teacher fidelity.
"""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from motorcycle_dynamics import TAU_MAX
from compare_34faults_res_scale import ObservedFaultEnv, Policy
from final_fault_benchmark import SCENARIOS

FIELDS = ['controller','scenario','fault_type','bucket','seed','iw','surv','rmse','steps',
          'u_rms','u2_sum','du','sat','peak_phi','wall_s','solve_calls',
          'solve_failures','solve_ms_total','solve_ms_mean','solve_statuses']


def setup_online(env, horizon):
    if not hasattr(env, '_solvers') or horizon not in env._solvers:
        raise RuntimeError('Online MPC solver pool lacks horizon %d; inspect motorcycle_env_mpc19.py' % horizon)
    if not hasattr(env, '_dare_chol'):
        raise RuntimeError('Missing _dare_chol; inspect the inherited MPC environment')
    if not hasattr(env, 'decode_action'):
        raise RuntimeError('Missing decode_action; cannot obtain baseline MPC weights')
    # Baseline weights follow the zero-action configuration of final_fault_benchmark.
    qphi, qphid, rv, decoded_n = env.decode_action(np.zeros(4, np.float32))
    mid4 = getattr(env, '_mid4', None)
    if mid4 is None or len(mid4) < 3:
        raise RuntimeError('Missing _mid4 baseline weights; inspect final_fault_benchmark.py')
    weights = np.array([qphi, qphid, float(mid4[2]), rv], dtype=float)
    lp_alpha = getattr(env, 'LP_ALPHA', None)
    if lp_alpha is None:
        raise RuntimeError('Missing LP_ALPHA; cannot reproduce online weight smoothing')
    return weights, float(lp_alpha), int(decoded_n)


def rollout(env, kind, seed, bucket, iw, policy, horizon, weights, lp_alpha):
    env.reset(seed=seed, options={'bucket':bucket, 'iw':iw})
    env._u_last = np.zeros(2, dtype=float)
    env._horizon = horizon
    solver = None
    w = weights.copy()
    statuses = {}
    calls = failures = 0
    solve_total = 0.0
    if kind == 'online_mpc':
        solver = env._solvers[horizon]
        solver.reset_warm_start()
    applied = []
    err_sq = 0.0
    peak = 0.0
    crashed = False
    start = time.perf_counter()
    for _ in range(env.MAX_STEPS):
        if kind == 'online_mpc':
            # Use the SAME faulted sensor stream used by the neural controllers.
            x0 = env._obs10()[:6].astype(float)
            xd = np.zeros(6)
            xd[0] = env._desired_roll()
            w = (1.0 - lp_alpha)*w + lp_alpha*weights
            qd = np.array([w[0],w[1],w[2],w[2],0.1*w[2],0.1*w[2]])
            rd = np.array([w[3],w[3]])
            A,B,bias = env.dyn.linearise(x0[0])
            solver.update_model(A,B,bias)
            pc = env._dare_chol(A,B,Q_diag=qd,R_diag=rd)
            # The MPCSolver maintains its own previous-control penalty state.
            solver.u_prev.value = env._u_last.copy()
            t0 = time.perf_counter()
            u = np.clip(solver.solve(x0=x0,xd=xd,Q_diag=qd,R_diag=rd,P_chol=pc),-TAU_MAX,TAU_MAX)
            solve_total += time.perf_counter()-t0
            calls += 1
            status = str(solver.last_solve_status)
            statuses[status] = statuses.get(status,0)+1
            if status not in ('optimal','optimal_inaccurate'):
                failures += 1
        elif kind == 'neural_mpc':
            xx = np.concatenate([env._obs10(),env._u_last,[horizon]]).astype(np.float32)
            u = np.clip(env._nn_u(xx),-TAU_MAX,TAU_MAX)
        elif kind == 'residual_adaptive':
            obs = env._make_obs()
            action = np.asarray(policy(obs),float)
            assert action.shape == (3,)
            k = min(int((np.clip(action[2],-1,1)+1)*0.5*len(env.HORIZONS)),len(env.HORIZONS)-1)
            env._horizon = env.HORIZONS[k]
            xx = np.concatenate([env._obs10(),env._u_last,[env._horizon]]).astype(np.float32)
            u = np.clip(env._nn_u(xx)+env.res_scale*np.clip(action[:2],-1,1),-TAU_MAX,TAU_MAX)
        else:
            raise ValueError(kind)
        env._steps += 1
        env.state = env._plant_dyn.step(env.state,u)
        env.v = float(np.clip(env.v+env.np_random.normal(0.,env.V_NOISE),1.,8.))
        env.alpha = float(np.clip(env.alpha+env.np_random.normal(0.,env.ALPHA_NOISE),-.20,.20))
        phi = float(env.state[0])
        err = phi-env._desired_roll()
        err_sq += err*err
        peak = max(peak,abs(phi))
        applied.append(env._plant_dyn.last_app.copy())
        env._u_last = np.asarray(u,float).copy()
        crashed = abs(phi)>np.pi/4
        if crashed or env._steps>=env.MAX_STEPS:
            break
    ua = np.asarray(applied,float)
    return dict(surv=int(not crashed),rmse=float(np.sqrt(err_sq/len(ua))),steps=len(ua),
                u_rms=float(np.sqrt(np.mean(ua**2))),u2_sum=float(np.sum(ua**2)),
                du=float(np.linalg.norm(np.diff(ua,axis=0),axis=1).mean()) if len(ua)>1 else 0.,
                sat=float(np.mean(np.any(np.abs(ua)>=.99*TAU_MAX,axis=1))),
                peak_phi=float(np.rad2deg(peak)),wall_s=time.perf_counter()-start,
                solve_calls=calls,solve_failures=failures,solve_ms_total=solve_total*1000,
                solve_ms_mean=solve_total*1000/calls if calls else 0.,
                solve_statuses=json.dumps(statuses,sort_keys=True))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ckpt',default='neural_mpc_multi13.pt')
    p.add_argument('--res_stem',default='sac_seed404_fromscratch_1m_best_eval')
    p.add_argument('--res_scale',type=float,default=5.)
    p.add_argument('--horizon',type=int,default=20)
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--seed0',type=int,default=610000)
    p.add_argument('--buckets',type=int,nargs='+',default=[0])
    p.add_argument('--iw',type=float,default=1.)
    p.add_argument('--only',nargs='+',default=['healthy','dropout_imu_p0.3','dropout_imu_p0.5','intermittent_imu_burst20','stuck_phi_15deg'])
    p.add_argument('--out',default='online_neural_residual_pilot')
    p.add_argument('--preflight_only',action='store_true')
    a=p.parse_args()
    if a.episodes<1 or not set(a.buckets).issubset({0,1,2}):p.error('invalid episodes/buckets')
    for f in [a.ckpt,a.res_stem+'.zip',a.res_stem+'_vecnorm.pkl']:
        if not Path(f).is_file():p.error('Missing: '+f)
    env=ObservedFaultEnv(ckpt=a.ckpt,decision_k=1,iw_list=(1.,),res_scale=a.res_scale)
    weights,lp_alpha,decoded_n=setup_online(env,a.horizon)
    residual=Policy(Path(a.res_stem+'.zip'),Path(a.res_stem+'_vecnorm.pkl'),15,3)
    print('PREFLIGHT: online N=',a.horizon,'zero-action decoded N=',decoded_n,
          'weights=',weights,'LP_ALPHA=',lp_alpha,'residual_steps=',residual.steps,flush=True)
    if decoded_n!=a.horizon:
        print('WARNING: zero-action MPC horizon differs from selected N; explicit fixed-N ablation',flush=True)
    for bucket in a.buckets:
        env.scenario=('none',{})
        env.reset(seed=a.seed0,options={'bucket':bucket,'iw':a.iw})
        initial=(env.state.copy(),env.v,env.alpha,env._desired_roll())
        env.reset(seed=a.seed0,options={'bucket':bucket,'iw':a.iw})
        np.testing.assert_array_equal(initial[0],env.state)
        assert initial[1:]==(env.v,env.alpha,env._desired_roll())
    print('INITIAL STATE GATE PASS',flush=True)
    if a.preflight_only:return
    selected=[(sid,ft,fp) for sid,ft,fp in SCENARIOS if a.only is None or sid in a.only]
    missing=set(a.only or [])-{sid for sid,_,_ in SCENARIOS}
    if missing:p.error('Unknown scenarios: '+str(sorted(missing)))
    out=Path(a.out+'_episodes.csv')
    if out.exists():p.error('Output already exists; choose a new --out to avoid mixing runs')
    with out.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
        for sid,ft,fp in selected:
            env.scenario=(ft,fp)
            for bucket in a.buckets:
                for seed in range(a.seed0,a.seed0+a.episodes):
                    for kind in ('online_mpc','neural_mpc','residual_adaptive'):
                        result=rollout(env,kind,seed,bucket,a.iw,residual,a.horizon,weights,lp_alpha)
                        writer.writerow(dict(controller=kind,scenario=sid,fault_type=ft,
                                             bucket=bucket,seed=seed,iw=a.iw,**result))
                        f.flush()
                        print(sid,bucket,seed,kind,'surv',result['surv'],
                              'rmse',round(result['rmse'],4),'solver_fail',result['solve_failures'],flush=True)
    import pandas as pd
    df=pd.read_csv(out)
    agg=df.groupby(['controller','scenario'],as_index=False).agg(
        episodes=('surv','size'),survival=('surv','mean'),rmse=('rmse','mean'),
        u_rms=('u_rms','mean'),du=('du','mean'),solve_failures=('solve_failures','sum'),
        solve_ms_mean=('solve_ms_mean','mean'))
    summary=Path(a.out+'_summary.csv');agg.to_csv(summary,index=False)
    print('DONE',len(df),'rows:',out,summary,flush=True)

if __name__=='__main__':main()
