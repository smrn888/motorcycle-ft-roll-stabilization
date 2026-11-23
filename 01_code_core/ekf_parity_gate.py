"""Strict paired parity comparison of original EKF vs ablation baseline episode CSVs.
No tolerances are silently relaxed. Exit code 2 means gate failed.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

KEY = ['scenario','bucket','seed']
EXACT = ['surv','steps','solver_failures','ekf_rejections']
FLOAT = ['rmse','peak_phi','u_rms','du','sat','ekf_roll_rmse']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--original',default='ekf_pilot_N20_episodes.csv')
    p.add_argument('--ablation',default='ekf_ablation_pilot_N20_episodes.csv')
    p.add_argument('--atol',type=float,default=1e-6)
    p.add_argument('--out',default='ekf_parity_report')
    a=p.parse_args()
    if a.atol<0:p.error('atol must be nonnegative')
    old=pd.read_csv(a.original); newer=pd.read_csv(a.ablation)
    if 'mode' in newer.columns:newer=newer.loc[newer['mode']=='baseline'].copy()
    for name,df in [('original',old),('ablation baseline',newer)]:
        if df.duplicated(KEY).any():raise ValueError(f'Duplicate keys: {name}')
        if df[KEY].isna().any().any():raise ValueError(f'Null keys: {name}')
    joined=old.merge(newer,on=KEY,how='outer',suffixes=('_old','_new'),indicator=True,validate='one_to_one')
    joined.to_csv(a.out+'_paired.csv',index=False)
    report={'old_rows':int(len(old)),'new_rows':int(len(newer)),
            'matched':int((joined['_merge']=='both').sum()),
            'unmatched':int((joined['_merge']!='both').sum()),'atol':a.atol,'metrics':{}}
    m=joined.loc[joined['_merge']=='both']
    passed=report['unmatched']==0 and len(old)==len(newer) and len(m)>0
    for c in EXACT+FLOAT:
        if c+'_old' not in m or c+'_new' not in m:
            report['metrics'][c]={'error':'missing column'};passed=False;continue
        x=pd.to_numeric(m[c+'_old'],errors='coerce');y=pd.to_numeric(m[c+'_new'],errors='coerce')
        d=(x-y).abs()
        invalid=int((~np.isfinite(x)|~np.isfinite(y)).sum())
        tol=0 if c in EXACT else a.atol
        bad=int((d>tol).sum())+invalid
        report['metrics'][c]={'max_abs_diff':float(d.max()) if len(d) else None,
                              'mismatch_count':bad,'invalid':invalid,'tolerance':tol}
        passed &= bad==0
    report['pass']=bool(passed)
    Path(a.out+'_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    print('PARITY PASS' if passed else 'PARITY FAIL — do not launch confirmatory benchmark')
    return 0 if passed else 2

if __name__=='__main__':raise SystemExit(main())
