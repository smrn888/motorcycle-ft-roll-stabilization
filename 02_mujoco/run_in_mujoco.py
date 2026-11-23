"""Run compare_34faults_res_scale.py with the *actual* MuJoCo-backed environment.

Run from the original project root (Three Models). Unlike runpy.run_path,
calling the imported module's main() preserves the monkey patch of its globals.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--mj_mode', choices=['reduced', 'full'], default='full')
ap.add_argument('--mj_servo', action='store_true')
ap.add_argument('--mj_timestep', type=float, default=None)
ap.add_argument('--mj_trace_dir', default=None)
ap.add_argument('--mj_trace_max', type=int, default=5)
ap.add_argument('script')
ap.add_argument('rest', nargs=argparse.REMAINDER)
a = ap.parse_args()
if a.mj_servo and a.mj_mode != 'full':
    ap.error('--mj_servo requires --mj_mode full')

project_root = Path.cwd()
mj_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(mj_dir))
script_path = (project_root / a.script).resolve()
if script_path.name != 'compare_34faults_res_scale.py':
    ap.error('This corrected launcher deliberately supports only compare_34faults_res_scale.py; other evaluation scripts need an explicit integration audit.')
if not script_path.is_file():
    ap.error(f'Script not found: {script_path}. Run from Three Models.')

out = None
for i, token in enumerate(a.rest[:-1]):
    if token == '--out':
        out = a.rest[i+1]
if not out:
    ap.error('Pass --out with a fresh output prefix')
required_prefix = f'mj_{a.mj_mode}_{"servo" if a.mj_servo else "raw"}_'
if not Path(out).name.startswith(required_prefix):
    ap.error(f'--out must start with {required_prefix!r} to prevent mixing runs')
# Never resume output from an older, potentially incorrect launcher.
for suffix in ('_mujoco.json','_episodes.csv','_config.json','_summary.csv'):
    if Path(out + suffix).exists():
        ap.error(f'Output already exists: {out+suffix}; choose a fresh --out')

import mujoco
import compare_34faults_res_scale as c34
from mj_observed_env import make_env_class

original_class = c34.ObservedFaultEnv
patched_class = make_env_class(a.mj_mode, a.mj_trace_dir, a.mj_trace_max,
                               a.mj_timestep, cancel_gimbal_reaction=a.mj_servo)
c34.ObservedFaultEnv = patched_class
assert c34.main.__globals__['ObservedFaultEnv'] is patched_class, 'CRITICAL: benchmark main() cannot see patched environment'
assert issubclass(patched_class, original_class)
assert patched_class.MJ_MODE == a.mj_mode
print(f'MUJOCO PATCH ACTIVE: {patched_class.__name__}, mode={a.mj_mode}, servo={a.mj_servo}', flush=True)

# Runtime guarantee: print and inspect an actual MuJoCo environment once constructed.
_orig_init = patched_class.__init__
def _checked_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    assert isinstance(self, patched_class), 'MuJoCo environment was not instantiated'
    print('MUJOCO ENV INSTANTIATED:', type(self).__name__, flush=True)
patched_class.__init__ = _checked_init

sys.argv = [str(script_path)] + a.rest
metadata = dict(engine='mujoco', mujoco_version=mujoco.__version__, mode=a.mj_mode,
                servo=a.mj_servo, timestep=a.mj_timestep or 0.001, script=script_path.name,
                launcher='patched_module_main_v2')
# metadata is written only once the patched module is selected.
Path(out + '_mujoco.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
# Critical: call the imported module's main; do NOT runpy.run_path(script).
c34.main()
