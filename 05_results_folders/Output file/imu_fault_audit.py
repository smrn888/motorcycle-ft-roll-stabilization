"""Offline paired audit of two severe IMU fault scenarios. No models loaded or simulations run."""
import argparse
from pathlib import Path
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('--input',default='three_models_paired_51000.csv')
p.add_argument('--out',default='imu_fault_audit')
a=p.parse_args()
d=pd.read_csv(a.input)
required={'scenario','crashed','steps'}
missing=required-set(d.columns)
if missing: raise SystemExit(f'Missing columns: {sorted(missing)}; available: {list(d.columns)}')
print('Columns:',list(d.columns))
seedcol=next((x for x in ['training_seed','train_seed','model_seed'] if x in d.columns),None)
evalcol=next((x for x in ['eval_seed','seed','episode_seed'] if x in d.columns),None)
if not evalcol: raise SystemExit('Need evaluation seed column (eval_seed or seed).')
print('Evaluation seed column:',evalcol)
ctrlcol=next((x for x in ['controller','ctrl','method'] if x in d.columns),None)
if not seedcol or not ctrlcol: raise SystemExit('Need train_seed and controller columns; inspect printed columns.')

def label(s):
 s=str(s).lower()
 if 'residual' in s or s.startswith('res:'):return 'Residual SAC'
 if 'plain' in s or 'standalone' in s:return 'Standalone SAC'
 if s in ('nmpc','neural mpc','neural_mpc'):return 'Neural MPC'
 return None

d['method_audit']=d[ctrlcol].map(label)
d=d[d.scenario.isin(['dropout_imu_p0.5','intermittent_imu_burst20']) & d.method_audit.notna()].copy()
if d.empty: raise SystemExit('No matching scenarios found')
d['alive']=~d.crashed.astype(str).str.lower().isin(['true','1','1.0'])
key=[seedcol,'scenario',evalcol]
if d.duplicated(key+['method_audit']).any(): raise SystemExit('Duplicate matched rows detected; inspect dataset before analysis')
wide=d.pivot(index=key,columns='method_audit',values='alive')
steps=d.pivot(index=key,columns='method_audit',values='steps')
needed=['Neural MPC','Residual SAC','Standalone SAC']
for c in needed:
 if c not in wide: raise SystemExit(f'Missing method: {c}; labels={d[ctrlcol].unique()}')
wide=wide.dropna(subset=needed)
steps=steps.loc[wide.index]
rows=[]
for (ts,sc),g in wide.groupby(level=[0,1]):
 n=g['Neural MPC'];r=g['Residual SAC'];s=g['Standalone SAC']
 z=steps.loc[g.index]
 rows.append(dict(train_seed=ts,scenario=sc,n=len(g),nmpc_surv_pct=100*n.mean(),res_surv_pct=100*r.mean(),sac_surv_pct=100*s.mean(),res_only_vs_nmpc=int((r&~n).sum()),nmpc_only_vs_res=int((n&~r).sum()),both_survive=int((r&n).sum()),both_fail=int((~r&~n).sum()),res_gain_pp=100*(r.mean()-n.mean()),res_crash_steps_median=float(z.loc[~r,'Residual SAC'].median()) if (~r).any() else float('nan'),nmpc_crash_steps_median=float(z.loc[~n,'Neural MPC'].median()) if (~n).any() else float('nan')))
summary=pd.DataFrame(rows).sort_values(['scenario','train_seed'])
summary.to_csv(a.out+'_by_training_seed.csv',index=False)
print('\nPAIRED AUDIT BY TRAINING SEED\n',summary.round(2).to_string(index=False))
print('\nPOOLED BY SCENARIO (descriptive only; not independent trials across training seeds)')
for sc,g in wide.groupby(level='scenario'):
 n=g['Neural MPC'];r=g['Residual SAC'];s=g['Standalone SAC']
 print(sc,'N=',len(g),'survival%:',*[round(100*x.mean(),2) for x in (n,r,s)],'rescue vs NMPC=',int((r&~n).sum()),'reverse=',int((n&~r).sum()))
print('\nSaved:',a.out+'_by_training_seed.csv')
