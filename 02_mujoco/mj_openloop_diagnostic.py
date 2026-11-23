"""Compare reference RK4, reset-each-step Full, and uninterrupted passive Full.
Run from the original Three Models folder, with this file alongside the mj_*.py files.
No learned controllers or checkpoints needed. Outputs a CSV and JSON summary.
"""
import argparse, csv, json, os, sys
from pathlib import Path
sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import mujoco
from motorcycle_dynamics import MotorcycleDynamics
from mj_dynamics import MuJoCoDynamics
from mj_params import P


def passive_full(plant, phi0, spin, steps):
    m, d = plant.m, plant.d
    mujoco.mj_resetData(m, d)
    d.qpos[plant.qa['roll']] = phi0
    if spin:
        d.qvel[plant.va['s_f']] = P['Omega']
        d.qvel[plant.va['s_r']] = -P['Omega']
    mujoco.mj_forward(m, d)
    out = [read_all(plant)]
    for _ in range(steps):
        d.ctrl[:] = 0
        d.qfrc_applied[:] = 0
        for __ in range(plant.substeps):
            mujoco.mj_step(m, d)
        out.append(read_all(plant))
    return out


def read_all(plant):
    d, q, v = plant.d, plant.qa, plant.va
    return dict(phi=float(d.qpos[q['roll']]),phi_dot=float(d.qvel[v['roll']]),
                df=float(d.qpos[q['g_f']]),dr=float(d.qpos[q['g_r']]),
                wgf=float(d.qvel[v['g_f']]),wgr=float(d.qvel[v['g_r']]),
                spin_f=float(d.qvel[v['s_f']]),spin_r=float(d.qvel[v['s_r']]))


def wrapped_full(plant, phi0, steps):
    x = np.zeros(6); x[0] = phi0
    out = [dict(phi=phi0,phi_dot=0.,df=0.,dr=0.,wgf=0.,wgr=0.,spin_f=P['Omega'],spin_r=-P['Omega'])]
    for _ in range(steps):
        x = plant.step(x, np.zeros(2))
        out.append(dict(phi=float(x[0]),phi_dot=float(x[1]),df=float(x[2]),dr=float(x[3]),
                        wgf=float(x[4]),wgr=float(x[5]),spin_f=float('nan'),spin_r=float('nan')))
    return out


def reference(phi0, steps):
    dyn=MotorcycleDynamics()
    x=np.zeros(6);x[0]=phi0
    out=[dict(phi=phi0,phi_dot=0.,df=0.,dr=0.,wgf=0.,wgr=0.,spin_f=float('nan'),spin_r=float('nan'))]
    for _ in range(steps):
        x=dyn.step(x,np.zeros(2))
        out.append(dict(phi=float(x[0]),phi_dot=float(x[1]),df=float(x[2]),dr=float(x[3]),
                        wgf=float(x[4]),wgr=float(x[5]),spin_f=float('nan'),spin_r=float('nan')))
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='mj_openloop_comparison')
    parser.add_argument('--phi0',type=float,default=0.05,help='initial roll, radians')
    parser.add_argument('--seconds',type=float,default=1.0)
    a=parser.parse_args()
    plant=MuJoCoDynamics(mode='full')
    steps=round(a.seconds/plant.Ts)
    if abs(steps*plant.Ts-a.seconds)>1e-9:parser.error('seconds must be multiple of Ts')
    curves={'paper_rk4':reference(a.phi0,steps),
            'full_reset_each_step':wrapped_full(plant,a.phi0,steps),
            'full_continuous_spinning':passive_full(plant,a.phi0,True,steps),
            'full_continuous_no_spin':passive_full(plant,a.phi0,False,steps)}
    fields=['model','time_s','phi','phi_dot','df','dr','wgf','wgr','spin_f','spin_r']
    with open(a.out+'.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for name,rows in curves.items():
            for idx,r in enumerate(rows):writer.writerow(dict(model=name,time_s=round(idx*plant.Ts,10),**r))
    summary={'parameters':{'phi0_rad':a.phi0,'seconds':a.seconds,'Ts':plant.Ts},'traces':{}}
    for name,rows in curves.items():
        z=rows[-1]
        summary['traces'][name]={'last':z,'max_abs_roll':max(abs(r['phi']) for r in rows),
                                 'all_finite_six_states':all(all(np.isfinite(r[k]) for k in ('phi','phi_dot','df','dr','wgf','wgr')) for r in rows),
                                 'first_time_over_45deg_s':next((i*plant.Ts for i,r in enumerate(rows) if abs(r['phi'])>np.pi/4),None)}
    Path(a.out+'.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    print('Saved:',a.out+'.csv',a.out+'.json')

if __name__=='__main__':main()
