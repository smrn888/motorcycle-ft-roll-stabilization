from stable_baselines3 import SAC

src = "sac_seed404_fromscratch_1m_best.zip"
dst = "sac_seed404_fromscratch_1m_best_compat.zip"

model = SAC.load(
    src,
    device="cpu",
    custom_objects={
        "lr_schedule": lambda _: 0.0,
        "_last_obs": None,
        "_last_episode_starts": None,
        "_last_original_obs": None,
        "ep_info_buffer": None,
        "ep_success_buffer": None,
        "replay_buffer_class": None,
    },
)

print("Loaded source checkpoint")
print("Observation:", model.observation_space)
print("Action:", model.action_space)

model.save(dst)
print("Saved:", dst)
