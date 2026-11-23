import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium import Env

stem = "sac_seed404_fromscratch_1m_best"
data = np.load(stem + "_vecnorm_stats.npz", allow_pickle=False)

class DummyEnv(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(15,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(3,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(15, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(15, dtype=np.float32), 0.0, False, False, {}

env = DummyVecEnv([lambda: DummyEnv()])

vn = VecNormalize(
    env,
    training=False,
    norm_obs=bool(data["norm_obs"].item()),
    norm_reward=bool(data["norm_reward"].item()),
    clip_obs=float(data["clip_obs"].item()),
    clip_reward=float(data["clip_reward"].item()),
    gamma=float(data["gamma"].item()),
    epsilon=float(data["epsilon"].item()),
)

vn.obs_rms.mean = data["obs_mean"].copy()
vn.obs_rms.var = data["obs_var"].copy()
vn.obs_rms.count = float(data["obs_count"].item())

vn.ret_rms.mean = data["ret_mean"].copy()
vn.ret_rms.var = data["ret_var"].copy()
vn.ret_rms.count = float(data["ret_count"].item())

output = stem + "_vecnorm_numpy1.pkl"
vn.save(output)

print("VECNORM SAVED:", output)
print("Obs count:", vn.obs_rms.count)
print("Obs shape:", vn.obs_rms.mean.shape)

vn.close()
