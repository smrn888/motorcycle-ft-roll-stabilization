import sys
import numpy as np
from stable_baselines3 import SAC

rng = np.random.default_rng(404)
obs = rng.normal(0, 0.5, size=(100, 15)).astype(np.float32)

if sys.argv[1] == "original":
    model = SAC.load(
        "sac_seed404_fromscratch_1m_best.zip",
        device="cpu",
        custom_objects={"lr_schedule": lambda _: 0.0},
    )
elif sys.argv[1] == "converted":
    from gymnasium import spaces
    model = SAC.load(
        "sac_seed404_fromscratch_1m_best_numpy1.zip",
        device="cpu",
        custom_objects={
            "lr_schedule": lambda _: 0.0,
            "train_freq": (1, "step"),
            "observation_space": spaces.Box(
                -np.inf, np.inf, shape=(15,), dtype=np.float32
            ),
            "action_space": spaces.Box(
                -1.0, 1.0, shape=(3,), dtype=np.float32
            ),
        },
    )
else:
    raise ValueError("Use original or converted")

actions, _ = model.predict(obs, deterministic=True)
np.save("actions_404_" + sys.argv[1] + ".npy", actions)

print("Saved actions:", actions.shape)
print("Min:", actions.min(), "Max:", actions.max())
