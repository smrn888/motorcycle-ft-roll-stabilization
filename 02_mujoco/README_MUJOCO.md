# MuJoCo plant for the CMG motorcycle (same project code, different physics)

Idea: do not re-implement controllers, faults or EKF. `run_in_mujoco.py` runs any existing evaluation script
(`evaluate_*.py`, `compare_34faults_res_scale.py`) after replacing only the plant's `step(x,u)` by MuJoCo.
Initial conditions, v/alpha noise, IMU faults, Neural MPC, SAC, Online MPC and EKF are the project's own code.
Same seed => paired episode with the reference run.

Files (copy into the MuJoCo folder, delete the old scaffold files env.py, ekf.py, controllers.py, run_benchmark.py,
control_oriented_cmg.py, render_video.py, params.py, build_model.py, validate.py and the copy of motorcycle_dynamics.py):
- mj_params.py       parameters imported from the project's motorcycle_dynamics.py
- mj_build_model.py  MJCF: mode reduced (paper equations) / full (real spinning flywheels, independent gyroscopics)
- mj_dynamics.py     MuJoCoDynamics(MotorcycleDynamics): stateless step(x,u), same interface as the project's RK4 step
- mj_observed_env.py ObservedFaultEnv subclass that wraps the MuJoCo plant (+ optional trajectory traces)
- run_in_mujoco.py   launcher; --out must start with mj_<mode>_ (never mix with reference-plant files)
- mj_validate.py     T1-T3 must PASS, T4 is informational (full-mode model-form gap)
- mj_render.py       MP4 from a saved trace

Setup:   pip install mujoco imageio imageio-ffmpeg
Run from the project folder (Three Models):
  python <mujoco folder>\mj_validate.py
  python <mujoco folder>\run_in_mujoco.py --mj_mode reduced <script>.py <its arguments> --out mj_reduced_<name>
  python <mujoco folder>\run_in_mujoco.py --mj_mode full --mj_trace_dir mj_traces <script>.py <its arguments> --out mj_full_<name>
  python <mujoco folder>\mj_render.py mj_traces\<file>.npz --out episode.mp4

Not supported: param_fault scenarios (they build their own dynamics). Actuator faults work (torque is not clipped inside MuJoCo).

## Full Servo integration (2026-10-18)

The new option `--mj_servo` requires `--mj_mode full`. It calls
`make_env_class(..., cancel_gimbal_reaction=True)`, rejects conflicting output prefixes,
and records `servo` in the sidecar JSON. Use distinct `--out` prefixes.

From the **Three Models** directory (replace `<MJ_DIR>` by this extracted folder path):

```
python "<MJ_DIR>\test_servo.py"
python "<MJ_DIR>\run_in_mujoco.py" --mj_mode full compare_34faults_res_scale.py --only healthy dropout_imu_p0.3 --episodes 5 --out mj_full_raw_smoke
python "<MJ_DIR>\run_in_mujoco.py" --mj_mode full --mj_servo compare_34faults_res_scale.py --only healthy dropout_imu_p0.3 --episodes 5 --out mj_full_servo_smoke
```

`test_servo.py` needs `mj_dynamics_orig.py`, provided in this package for comparison.
The benchmark additionally needs controller checkpoint files and dependencies from the original project.

Limitations: actuator gear changes actual physical joint torque; idealized gimbal reaction cancellation is not a physical servo controller model; MuJoCo was not installed in the code-generation environment so no MuJoCo runtime claim is made. This launcher patches `compare_34faults_res_scale.ObservedFaultEnv`; do not assume it patches other benchmark scripts with independently defined environments.
