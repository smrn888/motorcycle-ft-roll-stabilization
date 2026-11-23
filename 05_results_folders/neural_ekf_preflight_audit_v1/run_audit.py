import os, sys, json, hashlib, platform, subprocess, importlib
from pathlib import Path

ROOT = Path.cwd()
OUT = ROOT / "neural_ekf_preflight_audit_v1"
OUT.mkdir(exist_ok=True)

report = {
    "root": str(ROOT),
    "python": {
        "version": sys.version,
        "executable": sys.executable,
        "platform": platform.platform()
    },
    "packages": {},
    "files": {},
    "static_audit": {},
    "limitations": []
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def save_text(name, content):
    (OUT / name).write_text(content, encoding="utf-8")

# 1. Runtime and package versions
for name in [
    "numpy", "scipy", "cvxpy", "osqp", "torch",
    "stable_baselines3", "gymnasium", "pandas"
]:
    try:
        mod = importlib.import_module(name)
        report["packages"][name] = getattr(mod, "__version__", "version unavailable")
    except Exception as e:
        report["packages"][name] = {"import_error": repr(e)}

try:
    p = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True, timeout=120
    )
    save_text("pip_freeze.txt", p.stdout + "\n" + p.stderr)
    report["pip_freeze_exit_code"] = p.returncode
except Exception as e:
    report["limitations"].append("pip freeze failed: " + repr(e))

# 2. Identify source files and checkpoints
keywords = [
    "neural_mpc", "online_mpc", "ekf", "motorcycle_env",
    "motorcycle_dynamics", "residual", "sac", "vecnorm",
    "observation", "normaliz", "checkpoint"
]

source_files = sorted(
    p for p in ROOT.glob("*.py")
    if any(k in p.name.lower() for k in keywords)
)

required = [
    "compare_online_mpc_vs_neural_vs_residual.py",
    "evaluate_online_mpc_100seeds.py",
    "evaluate_online_mpc_ekf_100seeds.py",
    "evaluate_online_mpc_ekf_ablation.py",
    "motorcycle_env_mpc17.py",
    "motorcycle_dynamics.py"
]

for name in required:
    p = ROOT / name
    report["files"][name] = {
        "exists": p.is_file(),
        "sha256": sha256(p) if p.is_file() else None
    }

ckpts = []
for pattern in ["*.pt", "*.zip", "*vecnorm*.pkl", "*vecnorm*.pickle"]:
    ckpts.extend(ROOT.glob(pattern))

for p in sorted(set(ckpts)):
    report["files"][str(p.relative_to(ROOT))] = {
        "exists": True,
        "size_bytes": p.stat().st_size,
        "sha256": sha256(p)
    }

# 3. Find definitions and call sites, preserving surrounding source
patterns = [
    "def _nn_u", "def _obs10", "def rollout", "def main",
    "class SixStateEKF", "class RobustEKF",
    "ekf.reset", "ekf.correct", "ekf.predict",
    "ekf.x.copy", "_nn_u(", "_obs10(",
    "VecNormalize", "obs_rms", "observation_space",
    "torch.load", "load_state_dict", "neural_mpc_multi13",
    "res_scale", "TAU_MAX", "np.concatenate",
    "policy.predict", "Policy("
]

source_report = []
for p in source_files:
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        source_report.append(f"\n### {p.name}\nREAD ERROR: {e}")
        continue

    hits = []
    for i, line in enumerate(lines):
        if any(pattern.lower() in line.lower() for pattern in patterns):
            start = max(0, i - 4)
            end = min(len(lines), i + 6)
            hits.append((start, end))

    # Merge overlapping windows
    merged = []
    for start, end in hits:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    if merged:
        source_report.append(f"\n{'='*90}\nFILE: {p.name}\n")
        for start, end in merged:
            source_report.append(
                "\n".join(f"{j+1}: {lines[j]}" for j in range(start, end))
                + "\n" + "-"*50
            )

save_text("source_relevant_lines.txt", "\n".join(source_report))

# 4. Static tests: explicit evidence of EKF -> neural policy data flow
compare_path = ROOT / "compare_online_mpc_vs_neural_vs_residual.py"
ekf_path = ROOT / "evaluate_online_mpc_ekf_100seeds.py"
baseline_path = ROOT / "evaluate_online_mpc_100seeds.py"

def read(p):
    return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""

compare = read(compare_path)
ekf = read(ekf_path)
baseline = read(baseline_path)

checks = {
    "comparison_script_exists": compare_path.is_file(),
    "ekf_evaluator_exists": ekf_path.is_file(),
    "baseline_evaluator_exists": baseline_path.is_file(),
    "comparison_has_neural_policy_call": "_nn_u(" in compare,
    "comparison_mentions_ekf": "ekf" in compare.lower(),
    "ekf_evaluator_defines_six_state_ekf": "class SixStateEKF" in ekf,
    "ekf_evaluator_corrects_from_faulted_observation": (
        "ekf.correct(np.asarray(env._obs10()[:6]" in ekf
    ),
    "ekf_evaluator_uses_estimated_state": "x=ekf.x.copy()" in ekf or "x = ekf.x.copy()" in ekf,
    "baseline_uses_measured_observation": "env._obs10()[:6]" in baseline,
    "comparison_uses_raw_observation_for_neural_input": (
        "env._nn_u(xx)" in compare and "env._obs10()" in compare
    ),
}

# Check if a direct neural + EKF data path is visible in the comparison source.
# This is a screening check, not proof of semantic compatibility.
checks["direct_ekf_to_neural_path_visible"] = any(
    token in compare.lower()
    for token in [
        "ekf.x.copy()", "ekf_state", "estimated_state",
        "state_estimate", "ekf_estimate"
    ]
)

report["static_audit"] = checks
report["limitations"].append(
    "Static text checks cannot prove tensor shape compatibility, observation "
    "normalization semantics, causal correctness, or runtime control behavior."
)
report["limitations"].append(
    "Checkpoint hashes establish file identity only; they do not establish "
    "that a checkpoint was trained with a particular observation layout."
)
report["limitations"].append(
    "No training or long simulation was started by this audit."
)

save_text("audit_report.json", json.dumps(report, indent=2, ensure_ascii=False))

# 5. Human-readable summary
summary = []
summary.append("NEURAL MPC + EKF PRE-RUN AUDIT")
summary.append("=" * 70)
summary.append("Python: " + sys.version.replace("\n", " "))
summary.append("Executable: " + sys.executable)
summary.append("\nPACKAGE VERSIONS")
for k, v in report["packages"].items():
    summary.append(f"{k}: {v}")

summary.append("\nREQUIRED FILES")
for k, v in report["files"].items():
    summary.append(f"{k}: {v}")

summary.append("\nSTATIC CHECKS")
for k, v in checks.items():
    summary.append(f"{k}: {v}")

summary.append("\nIMPORTANT")
summary.append(
    "A PASS in a static text check is not proof that the full controller is correct."
)
summary.append(
    "Review source_relevant_lines.txt to verify how the EKF estimate reaches "
    "the neural network and whether normalization is compatible."
)
summary.append(
    "This script is read-only with respect to source files and existing experiment results."
)

save_text("audit_summary.txt", "\n".join(summary))
print("\n".join(summary))
print("\nREPORTS WRITTEN TO:", OUT)
