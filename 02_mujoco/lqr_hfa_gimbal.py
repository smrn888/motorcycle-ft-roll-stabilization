import os, sys, numpy as np
sys.path.insert(0, os.getcwd())
from scipy.linalg import solve_discrete_are
from motorcycle_dynamics import MotorcycleDynamics
from final_fault_benchmark import SCENARIOS, measure, FaultedPlant
from mj_dynamics import MuJoCoDynamics

nom = MotorcycleDynamics(); A, B, _ = nom.linearise(0.0)
P = solve_discrete_are(A, B, np.diag([100,10,20,20,1,1.]), np.diag([.1,.1]))
K = np.linalg.solve(np.diag([.1,.1]) + B.T @ P @ B, B.T @ P @ A)

def episode(plant, sc, seed):
    ft, fp = next((f, q) for i, f, q in SCENARIOS if i == sc)
    r = np.random.default_rng(seed); mr = np.random.default_rng([seed, 41])
    x = np.zeros(6); x[0] = r.uniform(-.3,.3); x[1] = r.uniform(-.8,.8)
    pl = FaultedPlant(plant, ft, fp, 50.); last = 0.; g = 0.
    for k in range(500):
        s = measure(x, k, ft, fp, mr, last); last = float(s[0])
        x = pl.step(x, np.clip(-K @ s, -50, 50)); g = max(g, abs(x[2]), abs(x[3]))
        if abs(x[0]) > np.pi/4: return 0, g
    return 1, g

for mode in ("reduced", "full"):
    plant = MuJoCoDynamics(mode)
    for sc in ("healthy", "dropout_imu_p0.3"):
        res = [episode(plant, sc, s) for s in range(40)]
        surv = sum(r[0] for r in res); gm = [r[1] for r in res]
        print(f"{mode:8s} {sc:18s} survival {surv}/40  max|delta| over episodes {max(gm):.3f} rad  "
              f"episodes with max|delta|<0.6: {sum(g < 0.6 for g in gm)}/40")