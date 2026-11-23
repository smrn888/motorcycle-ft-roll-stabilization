# Phase 2 — independent MuJoCo physics audit

## Verified from supplied original plant source (not simulation outcomes)

State order: `[phi, phi_dot, delta_f, delta_r, delta_f_dot, delta_r_dot]`.
Input: `[tau_f, tau_r]` in N m. Nominal `m=85 kg`, `h=0.5 m`, `Iw=0.5 kg m²`, `Omega=200 rad/s`, `Id=0.1 kg m²`, `Ts=0.01 s`, `c=2/s`, `g=9.81 m/s²`.
Reference roll inertia `21.25 kg m²`, gravitational coefficient `416.925 N m`, rotor angular momentum magnitude `100 kg m²/s`; desired reference gyro roll acceleration for `(delta_f_dot,delta_r_dot)=(1,0)` at `phi=0` is `-4.70588 rad/s²` (not guaranteed for the independently coupled full model).

## Known limitations of the original Full MuJoCo construction

1. Gimbal motor `gear≈2.50001` converts an input of 10 into approximately 25 N m mechanical joint effort, not 10 N m. This is a **calibrated torque interface** and must be reported. The parameter is derived from rotor transverse inertia `0.25 kg m²` vs reference `Id=0.1 kg m²`.
2. The paired flywheels carry angular momentum via MuJoCo rigid body joints; no manually imposed roll CMG force is used in Full mode. This *does* establish a distinct physics implementation, but not a detailed motorcycle/tire model.
3. Spin speeds are initialized at every outer step rather than sustained by simulated motor actuators. Spin motor reaction/energy are not physically resolved. Any claim of an autonomous motorized CMG model requires revisiting the plant state and motor design.
4. Geometry has no contact dynamics. Roll is a constrained hinge at the ground; the visual wheels are not dynamically operative.
5. Full-mode mass/COM compensation is an effective calibration, not literal motorcycle mass distribution; compare mass matrix/gravity numerically before asserting identity.
6. **Benchmark integration issue:** `compare_34faults_res_scale.ObservedFaultEnv.reset` wraps `_plant_dyn`; confirm the outer MuJoCo wrapper references the intended dynamics and the controller's internal `env.dyn` remains nominal. `param_fault` not supported by the original MuJoCo wrapper.
7. **Comparison protocol issue:** `final_fault_benchmark.FaultedPlant.step()` logs the clipped `last_app` but evolves the plant using *unclipped* `ua`. `compare_34faults_res_scale.RecordingPlant` instead evolves using `last_app`. This inconsistency must be resolved before drawing paper conclusions. For matched results reproduce the selected benchmark's semantics exactly; do not silently mix them.

## What is checked by mj_full_physics_audit.py

- Roll acceleration with rotor spin on/off, compared with nominal gravity term.
- Sign and magnitude of front/rear gyro roll effects and zero-spin control.
- Gimbal torque mapping, including required torque calibration.
- Determinism under reuse of a single MuJoCoDynamics object.
- Finite one-second open loop from 0.05 rad roll.

This is **diagnostics**, not yet a proof of physical validation. Record MuJoCo version, timestep, JSON report, source/checkpoint hashes, and ideally an ablation with physically limited rotor speed control before publication.

## Run

From original project `Three Models` directory, with this folder's MuJoCo scripts present:

```powershell
pip install mujoco numpy scipy
python "C:\Users\Moein\Desktop\Paper\RL-MPC\MuJoCo\MuJoCo_Phase2\mj_full_physics_audit.py" --out mj_full_physics_audit.json
```

Do not execute an entire controller comparison before checking the audit report.
