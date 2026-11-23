"""
test_residual_env.py — Gate Phase 5.0: اول این PASS شود، بعد SAC.

    python test_residual_env.py --ckpt neural_mpc.pt --ref cl_final100_traj.npz --seed0 71000 --n 10
    python test_residual_env.py --ckpt neural_mpc.pt --scan          (اسکن res_scale با action تصادفی)

T1 reproduction : residual=0 ⇒ φ(t) هر seed برابر ref (ستون nmpc در cl_final100_traj.npz)، tol=1e-5 rad
T2 bounds       : حتی با res_scale=100 و a=±1 ⇒ |u_final| ≤ 50
T3 determinism  : دو reset با یک seed و یک action ⇒ obs یکسان
T4 fault        : iw=0.5 ⇒ trajectory با iw=1.0 فرق می‌کند، info['fault'] درست
T5 reward/crash : φ=0.8 ⇒ terminated و r = −CRASH_PEN ؛ reward گام‌های سالم محدود به [0,2]
T6 obs/timing   : shape، finite، میانگین زمان هر گام env
"""
import argparse, os, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="neural_mpc.pt")
ap.add_argument("--ref", default="cl_final100_traj.npz")
ap.add_argument("--seed0", type=int, default=71000)
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--tol", type=float, default=1e-5)
ap.add_argument("--scan", action="store_true")
a = ap.parse_args()
from motorcycle_env_residual import MotorcycleEnvResidual
from motorcycle_dynamics import TAU_MAX

env = MotorcycleEnvResidual(ckpt=a.ckpt)
Z = np.zeros(2, np.float32)
BK = {"nominal": 0, "hard": 2}


def rollout(seed, bucket, iw, act_fn):
    obs, _ = env.reset(seed=seed, options=dict(bucket=bucket, iw=iw))
    phi, U, R = [], [], []
    while True:
        obs, r, te, tr, info = env.step(act_fn(obs))
        phi.append(info["phi"]); U.append(info["u_final"]); R.append(r)
        if te or tr:
            break
    return np.array(phi), np.array(U), np.array(R), te


ok_all = True
def report(name, ok, msg=""):
    global ok_all
    ok_all &= bool(ok); print(f"[{'PASS' if ok else 'FAIL'}] {name} {msg}", flush=True)

if a.scan:
    rng = np.random.default_rng(0)
    print("res_scale | survival | mean ep_len   (action یکنواخت تصادفی، nominal، iw=1.0)")
    for sc in (0.5, 1.0, 2.0, 5.0):
        env.res_scale = sc; s = []; L = []
        for i in range(a.n):
            phi_t, _, _, te = rollout(a.seed0 + i, 0, 1.0, lambda o: rng.uniform(-1, 1, 2).astype(np.float32))
            s.append(not te)
            L.append(len(phi_t))
        print(f"{sc:9.1f} | {np.mean(s):8.2f} | {np.mean(L):.0f}", flush=True)
    raise SystemExit

# T1
if os.path.exists(a.ref):
    ref = np.load(a.ref); worst, bad_len = 0.0, 0
    for cond, b in BK.items():
        for i in range(a.n):
            seed = a.seed0 + i
            phi, _, _, _ = rollout(seed, b, 1.0, lambda o: Z)
            r = ref[f"{cond}|nmpc|{seed}|phi"]
            if len(r) != len(phi):
                bad_len += 1; continue
            worst = max(worst, float(np.abs(r - phi).max()))
    report("T1 reproduction (residual=0 ≡ Neural MPC)", worst <= a.tol and bad_len == 0,
           f"max|Δφ|={worst:.2e} rad، طول‌های نامتناظر={bad_len}")
else:
    report("T1 reproduction", False, f"فایل ref پیدا نشد: {a.ref}")

# T2
env.res_scale = 100.0
_, U, _, _ = rollout(a.seed0, 0, 1.0, lambda o: np.array([1.0, -1.0], np.float32))
report("T2 bounds (res_scale=100)", np.abs(U).max() <= TAU_MAX + 1e-9, f"max|u|={np.abs(U).max():.2f}")
env.res_scale = 5.0

# T3
act = np.array([0.3, -0.2], np.float32)
o1, _ = env.reset(seed=a.seed0 + 3, options=dict(bucket=1, iw=1.0)); s1 = env.step(act)[0]
o2, _ = env.reset(seed=a.seed0 + 3, options=dict(bucket=1, iw=1.0)); s2 = env.step(act)[0]
report("T3 determinism", np.array_equal(o1, o2) and np.array_equal(s1, s2))

# T4
p1, _, _, _ = rollout(a.seed0 + 5, 0, 1.0, lambda o: Z)
p2, _, _, _ = rollout(a.seed0 + 5, 0, 0.5, lambda o: Z)
n = min(len(p1), len(p2)); d = float(np.abs(p1[:n] - p2[:n]).max())
env.reset(seed=1, options=dict(iw=0.5)); _, _, _, _, inf = env.step(Z)
report("T4 fault injection", d > 1e-3 and "0.5" in inf["fault"], f"max|Δφ|(iw1.0 vs 0.5)={d:.3f} rad، fault={inf['fault']}")

# T5
env.reset(seed=a.seed0, options=dict(bucket=0, iw=1.0)); env.state[0] = 0.80
_, r, te, _, _ = env.step(Z)
_, _, R, _ = rollout(a.seed0 + 1, 0, 1.0, lambda o: Z)
report("T5 reward/crash", te and abs(r + env.CRASH_PEN) < 1e-9 and R[:-1].min() >= 0 and R[:-1].max() <= 2,
       f"crash r={r:.1f} (انتظار −{env.CRASH_PEN})، reward سالم mean={R.mean():.3f}")

# T6
obs, _ = env.reset(seed=a.seed0, options=dict(bucket=0, iw=1.0)); t0 = time.perf_counter(); k = 0
for _ in range(200):
    obs, r, te, tr, _ = env.step(Z); k += 1
    if te or tr:
        break
ms = (time.perf_counter() - t0) / k * 1e3
report("T6 obs/timing", obs.shape == (12,) and np.isfinite(obs).all(), f"shape={obs.shape}، {ms:.3f} ms/step")
print("\nGATE Phase 5.0:", "PASS" if ok_all else "FAIL")
