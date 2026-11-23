import csv
import numpy as np
from pathlib import Path

paths = {
    10: Path("ekf_confirm_78000_N10_episodes.csv"),
    20: Path("ekf_confirm_78000_N20_episodes.csv"),
    30: Path("ekf_confirm_78000_N30_episodes.csv"),
    40: Path("ekf_confirm_78000_N40_candidate_first.csv"),
}

out = Path("ekf_confirm_78000_horizon_comparison.csv")
expected_scenarios = {
    "healthy", "dropout_imu_p0.3", "dropout_imu_p0.5",
    "intermittent_imu_burst20", "stuck_phi_15deg"
}

def avg(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")

all_results = []

for horizon, path in paths.items():
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    key = lambda r: (
        r["mode"], r["scenario"], r["bucket"], r["seed"]
    )

    keys = [key(r) for r in rows]
    if len(rows) != 1500 or len(set(keys)) != 1500:
        raise RuntimeError(f"Invalid episode set: {path}")

    scenarios = set(r["scenario"] for r in rows)
    if scenarios != expected_scenarios:
        raise RuntimeError(f"Unexpected scenarios: {path}")

    if any(int(r["horizon"]) != horizon for r in rows):
        raise RuntimeError(f"Wrong horizon values: {path}")

    for scenario in sorted(scenarios):
        subset = [r for r in rows if r["scenario"] == scenario]
        if len(subset) != 300:
            raise RuntimeError(f"Bad count: N={horizon}, {scenario}")

        alive = [r for r in subset if int(r["surv"]) == 1]

        record = {
            "horizon": horizon,
            "scenario": scenario,
            "episodes": len(subset),
            "survival_pct": 100 * len(alive) / len(subset),
            "rmse_alive": avg([r["rmse"] for r in alive]),
            "u_rms_all": avg([r["u_rms"] for r in subset]),
            "solver_ms_all": avg(
                [r["solver_ms_mean"] for r in subset]
            ),
            "ekf_ms_all": avg(
                [r["ekf_ms_mean"] for r in subset]
            ),
            "mean_rejections": avg(
                [r["ekf_rejections"] for r in subset]
            ),
        }

        all_results.append(record)

fields = list(all_results[0].keys())

with out.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(all_results)

print("\n=== EKF-MPC HORIZON COMPARISON ===")

for horizon in paths:
    sub = [r for r in all_results if r["horizon"] == horizon]

    print(f"\nN={horizon}")
    print("Macro survival:",
          round(avg([r["survival_pct"] for r in sub]), 3), "%")
    print("Macro survivor RMSE:",
          round(avg([r["rmse_alive"] for r in sub]), 6))
    print("Macro control RMS:",
          round(avg([r["u_rms_all"] for r in sub]), 6))
    print("Macro solver time:",
          round(avg([r["solver_ms_all"] for r in sub]), 4), "ms")

print("\nSaved:", out.resolve())
print("Comparison complete.")
