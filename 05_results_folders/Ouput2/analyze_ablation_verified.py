"""Summarize existing-checkpoint residual-authority experiments; no retraining."""
import argparse, glob, os, re
import numpy as np
import pandas as pd
p = argparse.ArgumentParser()
p.add_argument('--glob', default='ablation_*_*_all_episodes.csv')
p.add_argument('--out', default='ablation_verified')
a = p.parse_args()
rows = []
files = sorted(glob.glob(a.glob))
if not files:
    raise SystemExit('No experiment files found. Run run_ablation_verified.ps1 first.')
for f in files:
    m = re.fullmatch(r'ablation_(iw|fault)_(\d+(?:p\d+)?)_all_episodes\.csv', os.path.basename(f))
    if not m:
        continue
    experiment, tag = m.groups()
    scale = float(tag.replace('p', '.'))
    d = pd.read_csv(f)
    if not {'ctrl','iw','scenario','seed','crashed','rmse','u2_sum','steps','inj_ok','freq_ok'} <= set(d):
        raise ValueError(f'Missing required columns in {f}')
    if not (d.inj_ok.all() and d.freq_ok.all()):
        raise ValueError(f'Fault/frequency integrity check failed: {f}')
    for (iw, scenario, ctrl), g in d.groupby(['iw','scenario','ctrl']):
        alive = g.loc[~g.crashed.astype(bool)]
        rows.append(dict(experiment=experiment, residual_scale=scale, iw=iw, scenario=scenario,
                         controller='Neural MPC' if ctrl == 'nmpc' else 'Residual SAC',
                         n=len(g), survival=float((~g.crashed.astype(bool)).mean()),
                         rmse_alive=float(alive.rmse.mean()) if len(alive) else np.nan,
                         effort_step=float((g.u2_sum / g.steps).mean()),
                         mean_steps=float(g.steps.mean())))
S = pd.DataFrame(rows).sort_values(['experiment','scenario','iw','residual_scale','controller'])
S.to_csv(a.out + '_summary.csv', index=False)
# Matched evaluation seeds: exact paired binary outcomes, plus conditional RMSE on common successes.
pairs = []
for (experiment, scale, iw, scenario), g in pd.concat(
        [pd.read_csv(f).assign(experiment=re.fullmatch(r'ablation_(iw|fault)_(\d+(?:p\d+)?)_all_episodes\.csv', os.path.basename(f)).group(1),
                                      residual_scale=float(re.fullmatch(r'ablation_(iw|fault)_(\d+(?:p\d+)?)_all_episodes\.csv', os.path.basename(f)).group(2).replace('p','.')))
         for f in files if re.fullmatch(r'ablation_(iw|fault)_(\d+(?:p\d+)?)_all_episodes\.csv', os.path.basename(f))], ignore_index=True).groupby(['experiment','residual_scale','iw','scenario']):
    a0 = g[g.ctrl == 'nmpc'].set_index('seed')
    b0 = g[g.ctrl != 'nmpc'].set_index('seed')
    common = a0.index.intersection(b0.index)
    if not len(common):
        continue
    a0,b0 = a0.loc[common],b0.loc[common]
    aa,bb = ~a0.crashed.astype(bool),~b0.crashed.astype(bool)
    both = aa & bb
    pairs.append(dict(experiment=experiment, residual_scale=scale, iw=iw, scenario=scenario,
                      n=len(common), delta_survival=float(bb.mean()-aa.mean()),
                      residual_wins=int((bb & ~aa).sum()), residual_losses=int((aa & ~bb).sum()),
                      n_both_survive=int(both.sum()),
                      delta_rmse_common=float((b0.loc[both,'rmse']-a0.loc[both,'rmse']).mean()) if both.any() else np.nan))
P = pd.DataFrame(pairs).sort_values(['experiment','scenario','iw','residual_scale'])
P.to_csv(a.out + '_paired.csv', index=False)
print('Saved:', a.out + '_summary.csv', a.out + '_paired.csv')
print(S.to_string(index=False, max_rows=30))
