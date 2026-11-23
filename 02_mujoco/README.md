# Motorcycle–CMG in MuJoCo (generated from the paper's model)
- params.py        all parameters (FILL the marked ones with your paper's theta)
- build_model.py   MJCF generator (roll hinge + 2 gimbals + 2 spinning flywheels; MuJoCo computes gyroscopic coupling)
- env.py           env: x=[phi,dphi,df,dr,ddf,ddr], u=[tau_f,tau_r], dt=0.01, 500 steps, crash pi/4, 5 fault types
- ekf.py           robust EKF (NIS gate 9.21, stale detection, covariance recovery, Joseph form), model identified from MuJoCo
- controllers.py   adapters; PDController is a stand-in; plug in Neural MPC / SAC / online MPC here
- run_benchmark.py survival, RMSE(survivors), effort, ms/step per fault
- render_video.py  MP4 for the paper figure
