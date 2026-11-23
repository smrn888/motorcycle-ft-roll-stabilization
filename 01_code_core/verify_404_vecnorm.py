import sys
import pickle
import numpy as np

stem = "sac_seed404_fromscratch_1m_best"
mode = sys.argv[1]

path = (
    stem + "_vecnorm.pkl"
    if mode == "original"
    else stem + "_vecnorm_numpy1.pkl"
)

with open(path, "rb") as f:
    vn = pickle.load(f)

rng = np.random.default_rng(404)
obs = rng.normal(0, 0.5, size=(100, 15)).astype(np.float32)

normalized = vn.normalize_obs(obs)

assert normalized.shape == (100, 15)
assert np.isfinite(normalized).all()

np.save("normalized_404_" + mode + ".npy", normalized)

print("Mode:", mode)
print("NumPy:", np.__version__)
print("Normalized shape:", normalized.shape)
print("Saved successfully")
