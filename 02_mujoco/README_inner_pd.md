# Model-Based Inner-Loop PD — validation package

Copy `model_based_inner_pd.py` and `compare_inner_pd.py` into the project root,
alongside `reference_aware_pd.py`, `benchmark_classical.py`, and the original
motorcycle environment and benchmark modules. Do not replace other files.

## Controller

Differential gimbal speed `r = delta_dot_front - delta_dot_rear` satisfies
`r_dot = u_diff/I_delta - c*r`. The previous reference-aware controller already
used `u_diff = I_delta*(c*r + kr*(r_ref-r))`.

The new variant adds a filtered, causal acceleration feedforward:
`u_diff = I_delta*(ff_gain*r_ref_dot_filtered + c*r + kr*(r_ref-r))`.

The reference acceleration is estimated from current and past reference
commands only, clipped, then low-pass filtered. Both controllers use the same
outer loop and the same physical actuator clipping. This is not an optimal or
proven-stable design.

## Windows PowerShell

```powershell
python compare_inner_pd.py --only healthy dropout_imu_p0.5 act_delay_3 --episodes 5 --seed0 89000 --out inner_pd_validation.csv
python -c "import pandas as pd; d=pd.read_csv('inner_pd_validation.csv'); print(d.groupby(['scenario','variant']).agg(survival=('crashed',lambda s:1-s.mean()),rmse_all=('rmse','mean'),u_rms=('u_rms','mean'),sat=('sat','mean'),mean_steps=('steps','mean')).to_string())"
```

## Important

- Seed 89000 was used for previous diagnosis. These runs are exploratory and
  cannot serve as independent held-out test results.
- The prior PD already compensated gimbal damping. The actual new feature is
  filtered gimbal-reference acceleration feedforward.
- The environment and fault-injection code are required for a full closed-loop
  benchmark. No survival or RMSE outcomes are claimed here.
- For a fair final study, freeze controller design using separate validation
  seeds and run the final test on unseen seeds.
