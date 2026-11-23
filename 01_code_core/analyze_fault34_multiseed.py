"""Exploratory multi-training-seed comparison, 34-fault CSV outputs.
Uses training seed as bootstrap unit; shared evaluation seeds checked.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

METRICS=['surv','rmse','u_rms','du']

def load_file(path,controller,train_seed):
    if not path.is_file(): raise FileNotFoundError(str(path))
    df=pd.read_csv(path)
    required={'controller','scenario','bucket','seed','iw',*METRICS}
    missing=required-set(df.columns)
    if missing: raise ValueError(f'{path}: missing columns {missing}')
    df=df[df.controller==controller].copy()
    if df.empty: raise ValueError(f'{path}: no {controller} rows')
    key=['scenario','bucket','seed']
    if df.duplicated(key).any(): raise ValueError(f'{path}: duplicate evaluation keys')
    df['train_seed']=train_seed
    return df

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--folder',default='.')
    p.add_argument('--out',default='fault34_multiseed_stats')
    p.add_argument('--bootstrap',type=int,default=20000)
    p.add_argument('--rng_seed',type=int,default=20261016)
    a=p.parse_args()
    folder=Path(a.folder); rng=np.random.default_rng(a.rng_seed)
    allrows=[]; comparisons=[]
    for variant in ['best','final']:
        pure={s:load_file(folder/f'fault34_pure_{s}_{variant}_episodes.csv','pure_sac',s) for s in [11,22,33,44,55]}
        res={s:load_file(folder/f'fault34_{s}_{variant}_scale5_episodes.csv','residual_adaptive',s) for s in [101,202,303]}
        key=['scenario','bucket','seed']
        baseline=set(map(tuple,pure[11][key].to_numpy()))
        for label,group in [('Pure',pure),('Residual',res)]:
            for train_seed,df in group.items():
                if set(map(tuple,df[key].to_numpy()))!=baseline:
                    raise ValueError(f'Unmatched eval keys: {variant} {label} seed {train_seed}')
                if len(df)!=len(baseline): raise ValueError('Unexpected row count')
                if not np.allclose(df.iw,1): raise ValueError('Unexpected iw')
                if df[METRICS].isna().any().any(): raise ValueError(f'NaN metrics: {variant} {label} {train_seed}')
                means=df[METRICS].mean().to_dict()
                allrows.append(dict(variant=variant,method=label,train_seed=train_seed,episodes=len(df),**means))
        sub=pd.DataFrame([r for r in allrows if r['variant']==variant])
        for metric in METRICS:
            x=sub[sub.method=='Pure'][metric].to_numpy()
            y=sub[sub.method=='Residual'][metric].to_numpy()
            # Resampling *training seeds*, not episodes, independently by method.
            draws_x=x[rng.integers(0,len(x),size=(a.bootstrap,len(x)))].mean(axis=1)
            draws_y=y[rng.integers(0,len(y),size=(a.bootstrap,len(y)))].mean(axis=1)
            diff=draws_y-draws_x
            comparisons.append(dict(variant=variant,metric=metric,
                pure_mean=x.mean(),pure_sd=x.std(ddof=1),pure_n=len(x),
                residual_mean=y.mean(),residual_sd=y.std(ddof=1),residual_n=len(y),
                diff_res_minus_pure=y.mean()-x.mean(),
                ci95_low=np.quantile(diff,.025),ci95_high=np.quantile(diff,.975),
                percent_change=100*(y.mean()/x.mean()-1) if x.mean()!=0 else np.nan,
                note='Exploratory seed-level bootstrap; 3 vs 5 training seeds'))
    stem=Path(a.out)
    pd.DataFrame(allrows).to_csv(str(stem)+'_per_training_seed.csv',index=False)
    pd.DataFrame(comparisons).to_csv(str(stem)+'_comparison.csv',index=False)
    print(pd.DataFrame(comparisons).round(5).to_string(index=False))
    print('Saved',str(stem)+'_comparison.csv','and',str(stem)+'_per_training_seed.csv')

if __name__=='__main__': main()
