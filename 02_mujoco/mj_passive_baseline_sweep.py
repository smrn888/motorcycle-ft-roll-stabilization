"""Phase-2 passive CMG baseline (no controllers or learned policies).
Run from the original project folder; keep this next to mj_dynamics.py and mj_params.py.
Outputs per-trial data, plus a grouped summary. Continuous Full has no post-step
clipping; wrapped Full uses the project-compatible gimbal clamps.
"""
import argparse
import csv
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco
import numpy as np
from motorcycle_dynamics import MotorcycleDynamics
from mj_dynamics import MuJoCoDynamics
from mj_params import P


def trial(name, phi0, seconds, plant):
    step_dt = plant.Ts
    steps = round(seconds / step_dt)
    crash_limit = np.pi / 4
    x = np.zeros(6)
    x[0] = phi0
    history = [float(phi0)]
    gmax = 0.
    wmax = 0.
    spin_drift = 0.

    continuous = name in ('full_continuous_spin', 'full_continuous_no_spin')
    if continuous:
        m, d = plant.m, plant.d
        mujoco.mj_resetData(m, d)
        d.qpos[plant.qa['roll']] = phi0
        if name == 'full_continuous_spin':
            d.qvel[plant.va['s_f']] = P['Omega']
            d.qvel[plant.va['s_r']] = -P['Omega']
        mujoco.mj_forward(m, d)
    for step in range(1, steps + 1):
        if continuous:
            d.ctrl[:] = 0.
            d.qfrc_applied[:] = 0.
            for _ in range(plant.substeps):
                mujoco.mj_step(m, d)
            x = plant._read()
            if name == 'full_continuous_spin':
                spin_drift = max(spin_drift,
                    abs(float(d.qvel[plant.va['s_f']]) - P['Omega']),
                    abs(float(d.qvel[plant.va['s_r']]) + P['Omega']))
        else:
            x = plant.step(x, np.zeros(2))
        if not np.isfinite(x).all():
            return dict(model=name, phi0_deg=float(np.rad2deg(phi0)),survived=False,
                        first_over_45_s=None, finite=False, final_phi_deg=None,
                        max_abs_phi_deg=None, rms_phi_deg=None,
                        max_abs_gimbal_deg=None, max_abs_gimbal_rate=None,
                        max_spin_drift_rad_s=None, recorded_steps=step)
        history.append(float(x[0]))
        gmax = max(gmax, float(np.max(np.abs(x[2:4]))))
        wmax = max(wmax, float(np.max(np.abs(x[4:6]))))
        if abs(float(x[0])) > crash_limit:
            tcrash = round(step*step_dt, 9)
            break
    else:
        tcrash = None
    return dict(model=name, phi0_deg=float(np.rad2deg(phi0)),
                survived=tcrash is None, first_over_45_s=tcrash, finite=True,
                final_phi_deg=float(np.rad2deg(history[-1])),
                max_abs_phi_deg=float(np.rad2deg(max(map(abs, history)))),
                rms_phi_deg=float(np.rad2deg(np.sqrt(np.mean(np.square(history[1:]))))),
                max_abs_gimbal_deg=float(np.rad2deg(gmax)),
                max_abs_gimbal_rate=wmax,
                max_spin_drift_rad_s=spin_drift if name=='full_continuous_spin' else None,
                recorded_steps=len(history)-1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seconds',type=float,default=5.)
    p.add_argument('--angles',nargs='+',type=float,default=[-20,-10,-5,-3,-1,1,3,5,10,20],
                   help='initial roll angles in degrees')
    p.add_argument('--out',default='mj_passive_baseline')
    a=p.parse_args()
    if a.seconds<=0 or any(abs(v)>=45 for v in a.angles):
        p.error('seconds > 0 and all initial |roll| < 45 degrees required')
    plants = {'paper_rk4': MotorcycleDynamics(),
              'full_wrapped_spin': MuJoCoDynamics('full'),
              'full_continuous_spin': MuJoCoDynamics('full'),
              'full_continuous_no_spin': MuJoCoDynamics('full')}
    rows=[trial(name, np.deg2rad(angle), a.seconds, plant)
          for angle in a.angles for name,plant in plants.items()]
    with open(a.out+'.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    sums={name:{'n':len(rr),'survival_fraction':sum(r['survived'] for r in rr)/len(rr),
                 'mean_abs_final_roll_deg':float(np.mean([abs(r['final_phi_deg']) for r in rr if r['finite']])),
                 'mean_max_gimbal_deg':float(np.mean([r['max_abs_gimbal_deg'] for r in rr if r['finite']]))}
          for name in plants for rr in [[r for r in rows if r['model']==name]]}
    out={'metadata':{'seconds':a.seconds,'angles_deg':a.angles,'crash_threshold_deg':45,
                     'note':'Continuous MuJoCo omits project post-step gimbal limits; wrapped version includes them. All tests use zero control.'},
         'summary':sums,'trials':rows}
    Path(a.out+'.json').write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'metadata':out['metadata'],'summary':out['summary']},indent=2))
    print('Saved:',a.out+'.csv',a.out+'.json')

if __name__=='__main__':main()
