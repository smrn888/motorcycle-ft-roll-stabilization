"""Freeze EKF ablation configuration and prepare benchmark; forbid premature runs.
Creates a PowerShell run script only if parity passes and an independent
held-out evaluation seed range is selected. No measurements are changed.
"""
import argparse, json, hashlib, subprocess, sys
from pathlib import Path

SRC='evaluate_online_mpc_ekf_ablation.py'
SCENARIOS=['healthy','dropout_imu_p0.3','dropout_imu_p0.5','intermittent_imu_burst20','stuck_phi_15deg']
PARAMS={'ekf_gate':9.21,'ekf_q':[.002,.05,.005,.005,.10,.10],
        'ekf_r':[.01,.10,.005,.005,.05,.05],
        'ekf_p0':[.05,.5,.05,.05,.5,.5],
        'stale_phi_tol':1e-10,'stale_rate_tol':1e-10,'stale_run':2,
        'recovery_run':5,'recovery_factor':4.,'recovery_cap':1.}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--original',default='ekf_pilot_N20_episodes.csv')
    p.add_argument('--ablation',default='ekf_ablation_pilot_N20_episodes.csv')
    p.add_argument('--ckpt',default='neural_mpc_multi13.pt')
    p.add_argument('--seed0',type=int,default=78000,help='Independent holdout, not 77000-77004 pilot')
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--out',default='ekf_frozen_benchmark')
    p.add_argument('--prepare_only',action='store_true',help='Create frozen manifest but no run script')
    a=p.parse_args()
    if a.episodes!=100:p.error('Confirmatory design here requires 100 evaluation seeds')
    if a.seed0<=77004 and a.seed0+a.episodes-1>=77000:p.error('Holdout overlaps pilot tuning seeds 77000..77004')
    for fn in [SRC,'evaluate_online_mpc_ekf_100seeds.py',a.ckpt,a.original,a.ablation]:
        if not Path(fn).exists():p.error('File not found: '+fn)
    prefix=Path(a.out)
    parity=subprocess.run([sys.executable,str(Path(__file__).with_name('ekf_parity_gate.py')),
        '--original',a.original,'--ablation',a.ablation,'--out',str(prefix)+'_parity'],check=False)
    valid=parity.returncode==0
    config={'status':'FROZEN_PENDING_PARITY' if not valid else 'FROZEN',
       'development_seed_range':[77000,77004],
       'evaluation_seed_range':[a.seed0,a.seed0+a.episodes-1],
       'paired_buckets':[0,1,2],'horizons':[10,20,30,40],
       'scenarios':SCENARIOS,'modes':['combined'],
       'parameters':PARAMS,'note':'Params selected after pilot. Holdout needs all competing controllers on same new seeds.',
       'source_sha256':sha(SRC),'ckpt_sha256':sha(a.ckpt),
       'parity_pass':valid,'baseline_parity_file':str(prefix)+'_parity_report.json'}
    (prefix.with_name(prefix.name+'_manifest.json')).write_text(json.dumps(config,indent=2),encoding='utf-8')
    if not valid:
        print('STOP: parity is not established; manifest created, benchmark commands withheld.')
        return 2
    if a.prepare_only:
        print('Parity passed; configuration frozen; --prepare_only used.')
        return 0
    params=['--ekf_gate',str(PARAMS['ekf_gate'])]
    for n in ['ekf_q','ekf_r','ekf_p0']:params += ['--'+n]+list(map(str,PARAMS[n]))
    for n in ['stale_phi_tol','stale_rate_tol','stale_run','recovery_run','recovery_factor','recovery_cap']:
        params+=['--'+n,str(PARAMS[n])]
    def quote(x):return '"'+str(x).replace('"','`"')+'"'
    lines=['# FROZEN configuration. Do not edit parameters after examining holdout results.',
           '# Each horizon shares evaluation keys; main baselines must be run on the same holdout seeds.']
    for n in [10,20,30,40]:
        args=['python','.\\'+SRC,'--ckpt','.\\'+a.ckpt,'--horizon',str(n),'--seed0',str(a.seed0),
              '--episodes','100','--buckets','0','1','2','--only']+SCENARIOS+['--modes','combined','--out',f'{prefix.name}_N{n}']+params
        lines.append(' '.join(quote(z) if ' ' in str(z) else str(z) for z in args))
        lines.append('if ($LASTEXITCODE -ne 0) { throw "N='+str(n)+' failed" }')
    ps=prefix.with_name(prefix.name+'_run.ps1')
    ps.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('PARITY PASS, FROZEN, command file:',ps)
    return 0
if __name__=='__main__':raise SystemExit(main())
