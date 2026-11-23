"""Phase 2: instrumented, honest Full-MuJoCo physics audit.
Run in the ORIGINAL project directory:
    python C:/path/to/MuJoCo_Phase2/mj_full_physics_audit.py --out mj_full_physics_audit.json
No RL checkpoints needed. Does not claim model equivalence to the paper.
"""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import mujoco
from mj_dynamics import MuJoCoDynamics
from mj_params import P


def snapshot(plant, x, u=(0., 0.), spin=True):
    m, d = plant.m, plant.d
    mujoco.mj_resetData(m, d)
    plant._write(np.asarray(x, float))
    if not spin and plant.mode == 'full':
        d.qvel[plant.va['s_f']] = 0.
        d.qvel[plant.va['s_r']] = 0.
    d.ctrl[:] = np.asarray(u, float)
    mujoco.mj_forward(m, d)
    return {'roll_accel':float(d.qacc[plant.va['roll']]),
            'gimbal_accel':[float(d.qacc[plant.va[j]]) for j in ('g_f','g_r')],
            'spin_accel':[float(d.qacc[plant.va[j]]) for j in ('s_f','s_r')],
            'spin_rate':[float(d.qvel[plant.va[j]]) for j in ('s_f','s_r')]}


def snapshot_zero_spin(plant,x,u):
    return snapshot(plant,x,u,spin=False)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='mj_full_physics_audit.json')
    ap.add_argument('--timestep',type=float,default=.001)
    a=ap.parse_args()
    plant=MuJoCoDynamics(mode='full',timestep=a.timestep)
    p=P
    z=np.zeros(6)
    expected_H=p['Iw']*p['Omega']
    expected_J=p['m']*p['h']**2
    expected_G=p['m']*p['g']*p['h']
    outcomes={}
    outcomes['parameters']={'Iw_Omega_angular_momentum_each_kgm2_per_s':expected_H,
      'roll_inertia_target_kgm2':expected_J,'gravity_torque_coefficient_Nm':expected_G,
      'gimbal_nominal_inertia_kgm2':p['I_delta'],
      'full_flywheel_transverse_inertia_each_kgm2':p['Iw']/2.,
      'gimbal_gear_ratio':float(plant.m.actuator_gear[0,0]),
      'physical_torque_note':'MuJoCo applied hinge torque = controller torque times actuator gear; not a raw Nm transfer.'}
    x=z.copy();x[0]=.1
    grav_spin=snapshot(plant,x)
    grav_zero=snapshot(plant,x,spin=False)
    outcomes['gravity']={'spinning':grav_spin,'nonspinning':grav_zero,
       'reference_accel_rad_s2':float(expected_G*np.sin(.1)/expected_J),
       'interpretation':'At zero gimbal rates, gravity should destabilize +roll; deviations quantify added inertial coupling.'}
    for name, vf, vr in [('front_positive',1.,0.),('rear_positive',0.,1.),('differential',1.,-1.),('common',1.,1.)]:
        x=z.copy();x[4:6]=[vf,vr]
        outcomes[name]={'spin':snapshot(plant,x),'zero_spin':snapshot(plant,x,spin=False),
           'paper_reference_roll_accel':float(-expected_H*(vf-vr)/expected_J)}
    outcomes['gimbal_torque_10'] = {'spin':snapshot(plant,z,(10.,0.)),
                                    'zero_spin':snapshot_zero_spin(plant,z,(10.,0.)),
                                    'paper_reference_gimbal_accel':10./p['I_delta']}
    x=np.array([.06,.1,.1,-.1,2.,-2.])
    u=np.array([4.,-4.])
    first=plant.step(x,u);plant.step(np.zeros(6),np.array([20.,-10.]));again=plant.step(x,u)
    outcomes['stateless_repetition']={'max_absolute_state_difference':float(np.max(np.abs(first-again))),
                                      'PASS':bool(np.array_equal(first,again))}
    steps=[]; y=z.copy(); y[0]=.05
    for i in range(100):
        y=plant.step(y,np.zeros(2))
        if i in (0,9,49,99): steps.append({'step':i+1,'roll_rad':float(y[0]),'roll_rate_rad_s':float(y[1])})
        if not np.isfinite(y).all():break
    outcomes['open_loop_1s']={'samples':steps,'all_finite':bool(np.isfinite(y).all())}
    # Geometry-and-convention diagnostics must not be confused with a pass of external validity.
    outcomes['sanity_checks']={
      'gravity_roll_positive':bool(grav_spin['roll_accel']>0),
      'gyro_front_response_opposes_rear':bool(outcomes['front_positive']['spin']['roll_accel']*outcomes['rear_positive']['spin']['roll_accel']<0),
      'gyro_spin_effect_nonzero':bool(abs(outcomes['front_positive']['spin']['roll_accel']-outcomes['front_positive']['zero_spin']['roll_accel'])>1e-3),
      'stability_finite':bool(np.isfinite(y).all()),
      'stateless_repetition':outcomes['stateless_repetition']['PASS']}
    outcomes['limitations']=[
      'Flywheel speeds are reinitialized to +Omega/-Omega at each outer 0.01-s step; this is an idealized speed source, not a motor dynamics model.',
      'Full mode actuator gear modifies physical torque authority: controller input Nm cannot be interpreted as actual hinge Nm.',
      'No tire-ground contact, steering, forward speed or full motorcycle dynamics.',
      '6-state interface discards flywheel spin phase/speed after each step; report as assumption.',
      'Sanity checks are necessary, not sufficient, for independent physical validation.' ]
    with open(a.out,'w',encoding='utf-8') as f:json.dump(outcomes,f,indent=2,ensure_ascii=False)
    print(json.dumps(outcomes,indent=2,ensure_ascii=False))
    if not all(outcomes['sanity_checks'].values()):raise SystemExit('PHYSICS SANITY CHECK FAILURE (inspect report)')

if __name__=='__main__':main()
