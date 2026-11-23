"""Non-invasive IMU trajectory logger for unified_fault_benchmark.py.
Uses the benchmark's own controller loading, environment and paired evaluation seeds.
No model training or source modification. Logs actual plant steps, not inferred actions.
"""
import argparse
import atexit
import csv
import os
import runpy
import sys
import time
from collections import Counter

import numpy as np
from final_fault_benchmark import FaultedPlant

P = argparse.ArgumentParser(add_help=False)
P.add_argument('--out', default='imu_trajectory')
opts, _ = P.parse_known_args()
OUT = opts.out + '_trajectory.csv'
rows = []
active = dict(controller='unknown', seed=None, iw=None)
orig_step = FaultedPlant.step


def observed_step(self, x, u):
    # FaultEnv increments _steps immediately before calling FaultedPlant.step.
    # The last cached observation corresponds to the state BEFORE this plant step.
    env = active.get('env')
    if env is None:
        return orig_step(self, x, u)
    k = int(env._steps) - 1
    cached = getattr(env, '_mc', {}).get(k)
    true_phi_before = float(np.asarray(x)[0])
    true_rate_before = float(np.asarray(x)[1])
    meas_phi = float(cached[0]) if cached is not None else float('nan')
    meas_rate = float(cached[1]) if cached is not None else float('nan')
    base = np.asarray(getattr(env, '_u_nn', np.zeros(2)), dtype=float).copy()
    cmd = np.asarray(u, dtype=float).copy()
    # In these runs env._fault_eff is ones; cmd is the pre-fault command as seen by FaultedPlant.
    result = orig_step(self, x, u)
    applied = np.asarray(self.last_app, dtype=float).copy()
    ft, fp = env.scn
    active_fault = k >= int(fp.get('fault_start', 0))
    # A zero roll-rate observation does not, alone, prove a dropout.  For the
    # deterministic burst schedule we know the exact intended fault flag.
    if ft == 'intermittent_phi':
        intended_fault = active_fault and (k % int(fp.get('period', 20)) < int(fp.get('burst', 5)))
    elif ft == 'dropout_phi':
        # Dropout draw is not separately exposed by the benchmark. Do not
        # mislabel equality to the held value as proof of a random draw.
        intended_fault = None
    else:
        intended_fault = None
    is_res = str(active['controller']).startswith('res:')
    # This equals the commanded residual correction only when no final command
    # clipping occurs; keep it labelled 'command_minus_base', not raw RL action.
    correction = cmd - base
    rows.append(dict(training_seed=active.get('training_seed', ''), eval_seed=active['seed'],
        iw=active['iw'], controller=active['controller'], scenario=active.get('scenario', ''),
        step=k, time_s=k * 0.01, fault_type=ft, intended_burst_fault=intended_fault,
        phi_true_before=true_phi_before, phi_meas=meas_phi,
        phidot_true_before=true_rate_before, phidot_meas=meas_rate,
        phi_true_after=float(np.asarray(result)[0]),
        phi_target=float(env._desired_roll()),
        u_nn_0=float(base[0]), u_nn_1=float(base[1]),
        command_0=float(cmd[0]), command_1=float(cmd[1]),
        command_minus_base_0=float(correction[0]) if is_res else float('nan'),
        command_minus_base_1=float(correction[1]) if is_res else float('nan'),
        applied_0=float(applied[0]), applied_1=float(applied[1]),
        crashed_after=int(abs(float(np.asarray(result)[0])) > np.pi / 4)))
    return result


FaultedPlant.step = observed_step


def profiler(frame, event, arg):
    if event == 'call' and frame.f_code.co_name == 'run' and os.path.basename(frame.f_code.co_filename) == 'unified_fault_benchmark.py':
        loc = frame.f_locals
        active['controller'] = loc.get('name', 'unknown')
        active['seed'] = loc.get('seed')
        active['iw'] = loc.get('iw')
        active['env'] = frame.f_globals.get('env')
        active['scenario'] = getattr(active['env'], 'scn', ('unknown', {}))[0] + ':' + str(getattr(active['env'], 'scn', ('unknown', {}))[1])
    return None


def write_report():
    if not rows:
        print('[IMU TRAJECTORY PROBE] No plant steps recorded.')
        return
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    counts = Counter((r['scenario'], r['controller']) for r in rows)
    print('\n========== IMU TRAJECTORY PROBE ==========')
    for (scenario, ctrl), n in sorted(counts.items()):
        print(f'{scenario} | {ctrl}: {n} plant steps')
    print('Total logged steps:', len(rows))
    print('Trajectory CSV:', OUT)
    print('NOTE: random dropout events are NOT directly observable from held measurement alone.')
    print('NOTE: command_minus_base is post-clipping difference, not necessarily raw SAC action.')


atexit.register(write_report)
sys.setprofile(profiler)
try:
    sys.argv[0] = 'unified_fault_benchmark.py'
    runpy.run_path('unified_fault_benchmark.py', run_name='__main__')
finally:
    sys.setprofile(None)
