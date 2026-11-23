"""
motorcycle_env_residual.py — Phase 5: Residual SAC روی Neural MPC فریزشده.

    u_final = clip( u_NN(x) + res_scale · a ,  -TAU_MAX, TAU_MAX ),   a ∈ [-1,1]^2

- Neural MPC (neural_mpc.pt) فقط inference می‌شود؛ هیچ MPC/solver در حلقه اجرا نمی‌شود.
- از MotorcycleEnvMPC19 فقط برای reset/سناریو/مرجع/نویز/plant ارث می‌برد (مثل closed_loop_validation).
- با residual=0 و (bucket, iw) یکسان، trajectory دقیقاً همان Neural MPC در closed_loop_validation است
  (Gate: test_residual_env.py).
- Fault: Iw_scale از iw_list (هر اپیزود)؛ plant معیوب، Neural MPC و مدل نامی دست‌نخورده.
- Observation (12): [φ, φ̇, δf, δr, δ̇f, δ̇r, v, α, φd, err] + u_NN/U_REF (۲ عدد).
  (خام؛ در آموزش از VecNormalize استفاده کن.)
- Reward هر گام plant: clip(1 + track − dphi − rw_u‖u/τmax‖² − rw_du‖Δu/2τmax‖² − rw_res·mean(a²), 0, 2)؛
  سقوط: −CRASH_PEN. جمع جمله‌ها در info برای کالیبره‌ی آفلاین.
"""
import numpy as np
import gymnasium as gym
import torch

from motorcycle_env_mpc19 import MotorcycleEnvMPC19
from motorcycle_dynamics import TAU_MAX, MotorcycleDynamics
from train_neural_mpc import load_neural_mpc

torch.set_num_threads(1)


class MotorcycleEnvResidual(MotorcycleEnvMPC19):
    U_REF = 5.0
    PHI_MAX_IC = 0.60
    IC_RANGE = {1: (0.20, 0.50), 2: (0.50, 1.20)}

    def __init__(self, ckpt="neural_mpc.pt", res_scale=5.0, decision_k=1, level=8,
                 iw_list=(1.0,), iw_probs=None, bucket_probs=(0.4, 0.3, 0.3),
                 rw_u=0.0, rw_du=0.0, rw_res=0.0, master_seed=0, **kw):
        super().__init__(**kw)                      # (MPC solverها ساخته می‌شوند ولی استفاده نمی‌شوند)
        self.nn = load_neural_mpc(ckpt, "cpu")
        self.res_scale = float(res_scale)
        self.decision_k = int(decision_k)
        self.level = int(level)
        self.iw_list = np.asarray(iw_list, float)
        self.iw_probs = None if iw_probs is None else np.asarray(iw_probs, float) / np.sum(iw_probs)
        self.bucket_probs = np.asarray(bucket_probs, float) / np.sum(bucket_probs)
        self.rw_u, self.rw_du, self.rw_res = float(rw_u), float(rw_du), float(rw_res)
        self._master = np.random.default_rng(master_seed)
        self._dyn_cache = {}
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)
        self._u_nn = np.zeros(2); self._u_last = np.zeros(2)
        self._iw, self._bucket = 1.0, 0

    # ── helpers ─────────────────────────────────────────────────────────
    def _apply_iw(self, iw):
        self._fault_eff = np.ones(2); self._fault_rate = None
        if iw == 1.0:
            self._plant_dyn = self.dyn
        else:
            if iw not in self._dyn_cache:
                self._dyn_cache[iw] = MotorcycleDynamics(Iw_scale=float(iw))
            self._plant_dyn = self._dyn_cache[iw]
        self._fault_cfg = dict(eff=(1.0, 1.0), iw_scale=float(iw), rate_max=None)
        self._iw = float(iw)

    def _obs10(self):
        phi_d = self._desired_roll()
        return np.array([*self.state[:6], self.v, self.alpha, phi_d, self.state[0] - phi_d], np.float32)

    def _nn_u(self, x):
        with torch.no_grad():
            u = self.nn(torch.from_numpy(x)[None])[0].numpy().astype(np.float64)
        return np.clip(u, -TAU_MAX, TAU_MAX)

    def _make_obs(self):
        x = self._obs10()
        self._u_nn = self._nn_u(x)
        return np.concatenate([x, (self._u_nn / self.U_REF).astype(np.float32)]).astype(np.float32)

    # ── gym API ─────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        options = options or {}
        if seed is None:
            seed = int(self._master.integers(1 << 30))
        frng = np.random.default_rng([int(seed), 29])          # جدا از np_random ⇒ نویز دست‌نخورده
        iw = float(options["iw"]) if "iw" in options else float(frng.choice(self.iw_list, p=self.iw_probs))
        bucket = int(options["bucket"]) if "bucket" in options else int(frng.choice(3, p=self.bucket_probs))
        self._apply_iw(iw); self._bucket = bucket
        self.set_curriculum_level(self.level)
        super().reset(seed=seed)
        if bucket in self.IC_RANGE:                             # همان IC در closed_loop_validation
            rng = np.random.default_rng([int(seed), 17])
            pr, dr = self.IC_RANGE[bucket]
            self.state[0] = float(np.clip(self.state[0] + rng.uniform(-pr, pr), -self.PHI_MAX_IC, self.PHI_MAX_IC))
            self.state[1] = float(self.state[1] + rng.uniform(-dr, dr))
        self._u_last = np.zeros(2)
        obs = self._make_obs()
        return obs, dict(iw=self._iw, bucket=self._bucket, seed=int(seed))

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)
        du_rl = self.res_scale * a
        total_r, terminated, truncated = 0.0, False, False
        n_sub = n_alive = 0
        s_track = s_dphi = s_res2 = 0.0
        err_sq = 0.0
        u = np.zeros(2)
        for _ in range(self.decision_k):
            self._steps += 1; n_sub += 1
            u = np.clip(self._u_nn + du_rl, -TAU_MAX, TAU_MAX)
            self.state = self._plant_dyn.step(self.state, self._fault_eff * u)
            self.v = float(np.clip(self.v + self.np_random.normal(0.0, self.V_NOISE), 1.0, 8.0))
            self.alpha = float(np.clip(self.alpha + self.np_random.normal(0.0, self.ALPHA_NOISE), -0.20, 0.20))
            phi = float(self.state[0]); phi_d = self._desired_roll(); err = phi - phi_d
            err_sq += err ** 2
            du = u - self._u_last; self._u_last = u.copy()
            track = float(np.exp(-20.0 * err ** 2)); dphi = 0.05 * abs(self.state[1])
            res2 = float(np.mean(a ** 2))
            raw = (1.0 + track - dphi - self.rw_u * float(np.sum((u / TAU_MAX) ** 2))
                   - self.rw_du * float(np.sum((du / (2.0 * TAU_MAX)) ** 2)) - self.rw_res * res2)
            r = float(np.clip(raw, 0.0, 2.0))
            if abs(phi) > np.pi / 4:
                r = -self.CRASH_PEN; terminated = True
            else:
                n_alive += 1; s_track += track; s_dphi += dphi; s_res2 += res2
            total_r += r
            if terminated:
                break
            if self._steps >= self.MAX_STEPS:
                truncated = True; break
            self._u_nn = self._nn_u(self._obs10())
        obs = self._make_obs()
        info = dict(phi=phi, phi_d=float(phi_d), err=float(err), u_final=u.copy(), u_nn=self._u_nn.copy(),
                    du_rl=du_rl.copy(), iw=self._iw, bucket=self._bucket, n_plant_steps=n_sub,
                    n_alive=n_alive, err_sq_sum=float(err_sq), sum_track=s_track, sum_dphi=s_dphi,
                    sum_res2=s_res2, fault=str(self._fault_cfg))
        return obs, float(total_r), terminated, truncated, info
