import json
import zipfile
from pathlib import Path

src = Path("sac_seed404_fromscratch_1m_best_compat.zip")
dst = Path("sac_seed404_fromscratch_1m_best_numpy1.zip")

# These entries are training/runtime metadata, not policy weights.
replace = {
    "lr_schedule": None,
    "_last_obs": None,
    "_last_episode_starts": None,
    "_last_original_obs": None,
    "ep_info_buffer": None,
    "ep_success_buffer": None,
    "replay_buffer_class": None,
    "train_freq": None,
}

with zipfile.ZipFile(src, "r") as zin:
    data = json.loads(zin.read("data").decode("utf-8"))
    print("Serialized keys:")
    for key, value in data.items():
        if isinstance(value, dict) and ":serialized:" in value:
            print(" ", key)

    for key, value in replace.items():
        if key in data:
            data[key] = value

    with zipfile.ZipFile(dst, "w") as zout:
        for item in zin.infolist():
            if item.filename == "data":
                zout.writestr(
                    item,
                    json.dumps(data, ensure_ascii=False)
                )
            else:
                zout.writestr(item, zin.read(item.filename))

print("Created:", dst)
