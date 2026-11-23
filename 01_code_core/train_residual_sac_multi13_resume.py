"""
train_residual_sac.py — Phase 5.1: Residual SAC روی Neural MPC فریزشده.

pilot (کوتاه):
    python train_residual_sac.py --ckpt neural_mpc.pt --steps 100000 --out res_pilot
اصلی:
    python train_residual_sac.py --ckpt neural_mpc.pt --steps 1000000 --out res_main

- محیط: MotorcycleEnvResidual (iw_list هر اپیزود، bucket طبق --bucket_probs)، VecNormalize برای obs و reward.
- هر --eval_freq گام: ارزیابی قطعی روی seedهای ثابت (74000..)، bucket=nominal، برای هر iw؛
  baseline = residual صفر (Neural MPC تنها) در ابتدا روی همان seedها حساب و چاپ می‌شود.
- best checkpoint بر اساس score = mean(surv) − 0.5·mean(rmse) روی iwهای ارزیابی.
- خروجی: <out>_best.zip + <out>_best_vecnorm.pkl ، <out>_final.zip + <out>_final_vecnorm.pkl ، <out>_eval.csv
"""
import argparse, csv, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="neural_mpc.pt")
ap.add_argument("--steps", type=int, default=100_000)
ap.add_argument("--res_scale", type=float, default=5.0)
ap.add_argument("--decision_k", type=int, default=1)
ap.add_argument("--iw", nargs="*", type=float, default=[1.0, 0.7, 0.6, 0.5, 0.4])
ap.add_argument("--iw_probs", nargs="*", type=float, default=None)
ap.add_argument("--bucket_probs", nargs=3, type=float, default=[0.4, 0.3, 0.3])
ap.add_argument("--rw_du", type=float, default=0.0)
ap.add_argument("--rw_res", type=float, default=0.0)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--buffer", type=int, default=500_000)
ap.add_argument("--learning_starts", type=int, default=2000)
ap.add_argument("--eval_freq", type=int, default=10_000)
ap.add_argument("--eval_iw", nargs="*", type=float, default=[1.0, 0.6, 0.5, 0.4])
ap.add_argument("--eval_eps", type=int, default=20)
ap.add_argument("--eval_seed0", type=int, default=74000)
ap.add_argument("--no_zero_init", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="res_pilot")
ap.add_argument("--resume_model", default=None, help="Existing SAC .zip checkpoint")
ap.add_argument("--resume_vecnorm", default=None, help="Matching VecNormalize .pkl")
ap.add_argument("--resume_replay", default=None, help="Optional replay buffer .pkl")
a = ap.parse_args()

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from motorcycle_env_residual_multi13 import MotorcycleEnvResidual

torch.manual_seed(a.seed); np.random.seed(a.seed)
KW = dict(ckpt=a.ckpt, res_scale=a.res_scale, decision_k=a.decision_k, rw_du=a.rw_du, rw_res=a.rw_res)

venv = DummyVecEnv([lambda: Monitor(MotorcycleEnvResidual(
    iw_list=a.iw, iw_probs=a.iw_probs, bucket_probs=a.bucket_probs, master_seed=a.seed, **KW))])
if bool(a.resume_model) != bool(a.resume_vecnorm):
    ap.error("--resume_model and --resume_vecnorm must be provided together")
if a.resume_model:
    venv = VecNormalize.load(a.resume_vecnorm, venv)
    venv.training = True
    venv.norm_reward = True
else:
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
eval_env = MotorcycleEnvResidual(iw_list=(1.0,), **KW)


def evaluate(policy):
    """policy(obs_raw)->action. برگشت: {iw: dict} روی seedهای ثابت، bucket=nominal."""
    out = {}
    for iw in a.eval_iw:
        surv, rm, steps, du = [], [], [], []
        for i in range(a.eval_eps):
            obs, _ = eval_env.reset(seed=a.eval_seed0 + i, options=dict(iw=iw, bucket=0))
            e2 = n = 0; dsum = 0.0; k = 0
            while True:
                obs, r, te, tr, info = eval_env.step(policy(obs))
                e2 += info["err_sq_sum"]; n += info["n_plant_steps"]
                dsum += float(np.abs(info["du_rl"]).mean()); k += 1
                if te or tr:
                    break
            surv.append(not te); rm.append(np.sqrt(e2 / max(n, 1))); steps.append(n); du.append(dsum / k)
        out[iw] = dict(surv=float(np.mean(surv)), rmse=float(np.mean(rm)),
                       steps=float(np.mean(steps)), du_rl=float(np.mean(du)))
    return out


def score(res):
    return float(np.mean([v["surv"] for v in res.values()]) - 0.5 * np.mean([v["rmse"] for v in res.values()]))


def show(tag, res):
    print(f"  {tag:10s} " + " | ".join(f"iw{iw}: s={v['surv']:.2f} r={v['rmse']:.3f} L={v['steps']:.0f}"
                                       for iw, v in res.items()) + f" || score={score(res):.4f}", flush=True)


Z = np.array([0.0, 0.0, -1.0/4.0], np.float32)  # zero residual, N=20 (bin midpoint)
print("baseline (residual=0, N=20 ≡ Neural MPC with fixed N=20) روی seedهای ارزیابی:")
base = evaluate(lambda o: Z); show("baseline", base)
base_score = score(base)

if a.resume_model:
    model = SAC.load(a.resume_model, env=venv, device="auto")
    if a.resume_replay:
        model.load_replay_buffer(a.resume_replay)
        print("Replay buffer loaded:", a.resume_replay, flush=True)
    else:
        print("WARNING: no replay buffer supplied; continuing from saved weights and optimizer state "
              "with a fresh replay buffer (fine-tuning, not exact continuation).", flush=True)
    print(f"Loaded checkpoint at {model.num_timesteps} steps", flush=True)
else:
    model = SAC("MlpPolicy", venv, learning_rate=a.lr, batch_size=a.batch, buffer_size=a.buffer,
                learning_starts=a.learning_starts, gamma=0.99, tau=0.005, train_freq=1, gradient_steps=1,
                policy_kwargs=dict(net_arch=[256, 256]), seed=a.seed, verbose=0, device="auto")
    if not a.no_zero_init:
        with torch.no_grad():
            model.actor.mu.weight.mul_(0.01); model.actor.mu.bias.zero_()
            model.actor.mu.bias[2] = -0.25


class EvalCB(BaseCallback):
    def __init__(self):
        super().__init__(); self.best = -1e9; self.t0 = time.time()
        self.f = open(a.out + "_eval.csv", "w", newline=""); self.w = csv.writer(self.f)
        self.w.writerow(["step", "iw", "surv", "rmse", "steps", "du_rl", "score"])

    def _on_step(self):
        if self.n_calls % a.eval_freq:
            return True
        pol = lambda o: self.model.predict(venv.normalize_obs(o.astype(np.float32)), deterministic=True)[0]
        res = evaluate(pol); sc = score(res)
        print(f"[{self.num_timesteps}] {time.time() - self.t0:.0f}s  (baseline score={base_score:.4f})", flush=True)
        show("SAC", res)
        for iw, v in res.items():
            self.w.writerow([self.num_timesteps, iw, v["surv"], v["rmse"], v["steps"], v["du_rl"], sc])
        self.f.flush()
        if sc > self.best:
            self.best = sc; self.model.save(a.out + "_best"); venv.save(a.out + "_best_vecnorm.pkl")
            print(f"  ↑ best saved (score={sc:.4f})", flush=True)
        return True


model.learn(total_timesteps=a.steps, callback=EvalCB(), progress_bar=False,
            reset_num_timesteps=not bool(a.resume_model))
model.save(a.out + "_final"); venv.save(a.out + "_final_vecnorm.pkl")
model.save_replay_buffer(a.out + "_final_replay.pkl")
print("done →", a.out + "_best.zip /", a.out + "_final.zip")
