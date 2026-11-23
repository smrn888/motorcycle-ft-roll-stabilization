"""Train Standalone SAC, matched to train_residual_sac.py.
Outputs: final_sacplain_seedN.zip, vecnorm_final_sacplain_seedN.pkl,
         best_sacplain_seedN.zip, vecnorm_best_sacplain_seedN.pkl, eval_sacplain_seedN.csv.
"""
import argparse, csv, os, time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from motorcycle_env_standalone import MotorcycleEnvStandalone

p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, default=11)
p.add_argument('--steps', type=int, default=1_000_000)
p.add_argument('--out_dir', default='standalone_runs')
p.add_argument('--iw', nargs='+', type=float, default=[1., .7, .6, .5, .4])
p.add_argument('--iw_probs', nargs='+', type=float, default=None)
p.add_argument('--bucket_probs', nargs=3, type=float, default=[.4, .3, .3])
p.add_argument('--eval_iw', nargs='+', type=float, default=[1., .6, .5, .4])
p.add_argument('--eval_eps', type=int, default=20)
p.add_argument('--eval_freq', type=int, default=10_000)
p.add_argument('--eval_seed0', type=int, default=74000)
p.add_argument('--lr', type=float, default=3e-4)
p.add_argument('--batch', type=int, default=256)
p.add_argument('--buffer', type=int, default=500_000)
p.add_argument('--learning_starts', type=int, default=2000)
p.add_argument('--rw_u', type=float, default=0.)
p.add_argument('--rw_du', type=float, default=0.)
p.add_argument('--rw_res', type=float, default=0.)
a = p.parse_args()
if a.eval_freq < 1 or a.steps < 1 or a.eval_eps < 1:
    p.error('steps, eval_freq, eval_eps must be positive')
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
tag = f'sacplain_seed{a.seed}'
torch.manual_seed(a.seed); np.random.seed(a.seed)
kw = dict(decision_k=1, rw_u=a.rw_u, rw_du=a.rw_du, rw_res=a.rw_res)
train = DummyVecEnv([lambda: Monitor(MotorcycleEnvStandalone(
    iw_list=a.iw, iw_probs=a.iw_probs, bucket_probs=a.bucket_probs,
    master_seed=a.seed, **kw))])
train = VecNormalize(train, norm_obs=True, norm_reward=True, clip_obs=10., gamma=.99)
eval_env = MotorcycleEnvStandalone(iw_list=(1.,), **kw)

def evaluate(model):
    results = {}
    for iw in a.eval_iw:
        surv, rmses, lengths = [], [], []
        for j in range(a.eval_eps):
            obs, _ = eval_env.reset(seed=a.eval_seed0+j, options={'iw':iw, 'bucket':0})
            e2 = 0.; n = 0
            while True:
                normalized = train.normalize_obs(obs.astype(np.float32))
                action, _ = model.predict(normalized, deterministic=True)
                obs, _, terminated, truncated, info = eval_env.step(action)
                e2 += info['err_sq_sum']; n += info['n_plant_steps']
                if terminated or truncated:
                    break
            surv.append(float(not terminated)); rmses.append(np.sqrt(e2/max(n,1))); lengths.append(n)
        results[iw] = dict(surv=float(np.mean(surv)), rmse=float(np.mean(rmses)), steps=float(np.mean(lengths)))
    return results

def score(results):
    return float(np.mean([v['surv'] for v in results.values()]) -
                 .5*np.mean([v['rmse'] for v in results.values()]))

class Evaluation(BaseCallback):
    def __init__(self):
        super().__init__()
        self.best = -np.inf
        self.start = time.time()
        self.csv_path = out / f'eval_{tag}.csv'
        self.f = None

    def _on_training_start(self):
        self.f = self.csv_path.open('w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.f)
        self.writer.writerow(['step','iw','surv','rmse','steps','score'])

    def _on_step(self):
        if self.n_calls % a.eval_freq:
            return True
        results = evaluate(self.model)
        sc = score(results)
        print(f'[{self.num_timesteps}] seed={a.seed} score={sc:.5f} elapsed={time.time()-self.start:.0f}s', flush=True)
        for iw, v in results.items():
            print(f'  iw={iw}: survival={v["surv"]:.3f} rmse={v["rmse"]:.4f} steps={v["steps"]:.1f}', flush=True)
            self.writer.writerow([self.num_timesteps, iw, v['surv'], v['rmse'], v['steps'], sc])
        self.f.flush()
        if sc > self.best:
            self.best = sc
            self.model.save(str(out/f'best_{tag}'))
            train.save(str(out/f'vecnorm_best_{tag}.pkl'))
            print('  best model + matching normalization saved', flush=True)
        return True

    def _on_training_end(self):
        if self.f:
            self.f.close()

model = SAC('MlpPolicy', train, learning_rate=a.lr, batch_size=a.batch,
            buffer_size=a.buffer, learning_starts=a.learning_starts,
            gamma=.99, tau=.005, train_freq=1, gradient_steps=1,
            policy_kwargs={'net_arch':[256,256]}, seed=a.seed, verbose=0, device='auto')
# No residual zero-init: direct SAC must learn the entire action, not a correction.
model.learn(total_timesteps=a.steps, callback=Evaluation(), progress_bar=False)
model.save(str(out/f'final_{tag}'))
train.save(str(out/f'vecnorm_final_{tag}.pkl'))
print(f'DONE: {out} / final_{tag}.zip and matching vecnorm', flush=True)
