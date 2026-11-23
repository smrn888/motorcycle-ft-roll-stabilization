# Reference-aware gravity PD (experimental)

Place `reference_aware_pd.py`, `compare_reference_pd.py`, and `diagnose_reference_pd.py` beside your existing `gravity_pd.py`, `benchmark_classical.py`, `motorcycle_dynamics.py`, `motorcycle_env_residual.py`, `final_fault_benchmark.py`, and neural checkpoint. The comparison reuses `benchmark_classical.run_one` and the existing fault injection; it does **not** modify original files.

PowerShell:

```powershell
python compare_reference_pd.py --only healthy dropout_imu_p0.5 act_delay_3 --episodes 5 --seed0 89000 --out reference_pd_comparison.csv
python -c "import pandas as pd; d=pd.read_csv('reference_pd_comparison.csv'); print(d.groupby(['scenario','variant']).agg(survival=('crashed',lambda s:1-s.mean()),rmse_all=('rmse','mean'),u_rms=('u_rms','mean'),sat=('sat','mean')).to_string())"
python diagnose_reference_pd.py --scenario healthy --seed 89000 --out reference_pd_trace.csv
python diagnose_reference_pd.py --scenario act_delay_3 --seed 89000 --out reference_pd_delay_trace.csv
```

**Method:** Reference velocity is estimated using a causal backward difference on the current/previous roll-reference samples and a first-order low-pass filter. Reference acceleration is estimated from the filtered velocity difference, clipped, and low-pass filtered. The first call initializes the derivative estimates to zero. No future reference is used. The requested roll acceleration is `ref_acc - kd*(phidot-ref_rate) - kp*(phi-ref)`; the corresponding desired differential gimbal rate follows the nonlinear roll equation. Inner rate loop is the same as the previous gravity PD, enabling a controlled A/B comparison. Note that reference derivatives are estimated rather than exact, and can be noisy/bias the result.

**Diagnostic caveat:** Physical roll acceleration is calculated using the actual pre-step state and scenario-adjusted `MotorcycleDynamics.f`; for actuator-delay scenarios the gimbal torque applied to the plant may differ from the issued torque, but instantaneous roll acceleration depends on actual gimbal rate rather than direct torque. `roll_acc_model_sensed` is the controller's estimate from sensed states, not physical ground truth.

**Validation:** Syntax and deterministic toy-model checks are provided, but full environment evaluation requires your local checkpoint/dependencies. Keep validation and final test seeds separate; do not use the 5-episode smoke test as paper evidence.
