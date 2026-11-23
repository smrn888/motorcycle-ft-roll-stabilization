"""Physics checks for the MuJoCo plant. Needs only the project's motorcycle_dynamics.py and final_fault_benchmark.py
(run from the project folder). Writes mj_validation_report.json.

T1  one-step accuracy of reduced mode vs a 100x finer RK4 of the project equations (project RK4 shown for reference)
T2  step() is stateless / deterministic
T3  closed loop (LQR stand-in on the measured state, project IMU faults): project plant vs MuJoCo reduced must agree
T4  INFORMATION ONLY, full mode: model-form gap (gimbal authority, free-gimbal open loop, gyroscopic coupling, closed loop)
"""
import os, sys
sys.path.insert(0, os.getcwd())          # project modules live in the cwd ("Three Models")
import argparse, json, numpy as np
from scipy.linalg import solve_discrete_are
from motorcycle_dynamics import MotorcycleDynamics
from final_fault_benchmark import SCENARIOS, measure, FaultedPlant
from mj_dynamics import MuJoCoDynamics
from mj_params import P

ap = argparse.ArgumentParser(); ap.add_argument("--n_states", type=int, default=1000); ap.add_argument("--n_eps", type=int, default=20)
ap.add_argument("--out", default="mj_validation_report.json"); a = ap.parse_args()
nom, rng, rep = MotorcycleDynamics(), np.random.default_rng(0), {}


class Fine(MotorcycleDynamics):
    def step(self, x, u, clamp_u=False):
        x = np.asarray(x, float); n = 100; dt = self.Ts / n
        for _ in range(n):
            k1 = self.f(x, u); k2 = self.f(x + .5 * dt * k1, u); k3 = self.f(x + .5 * dt * k2, u); k4 = self.f(x + dt * k3, u)
            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x[2:4] = np.clip(x[2:4], -np.pi / 2, np.pi / 2); x[4:6] = np.clip(x[4:6], -20, 20); return x


mjr, mjf, fine = MuJoCoDynamics("reduced"), MuJoCoDynamics("full"), Fine()

# T1
em, ep = [], []
for _ in range(a.n_states):
    x = np.array([rng.uniform(-.7, .7), rng.uniform(-1.5, 1.5), rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-15, 15), rng.uniform(-15, 15)])
    u = rng.uniform(-50, 50, 2); t = fine.step(x, u)
    em.append(np.abs(mjr.step(x, u) - t).max()); ep.append(np.abs(nom.step(x, u) - t).max())
rep["T1"] = dict(mj_reduced_mean=float(np.mean(em)), mj_reduced_max=float(np.max(em)), project_rk4_mean=float(np.mean(ep)),
                 project_rk4_max=float(np.max(ep)), PASS=bool(np.mean(em) < 5e-6 and np.max(em) < 5e-5))
# T2
x0, u0 = np.array([.1, .2, .1, -.1, 1., -1.]), np.array([3., -3.])
r1 = mjr.step(x0, u0); mjr.step(np.zeros(6), np.array([50., 50.])); r2 = mjr.step(x0, u0)
rep["T2"] = dict(PASS=bool(np.array_equal(r1, r2)))

# LQR stand-in
A, B, _ = nom.linearise(0.0)
Pm = solve_discrete_are(A, B, np.diag([100, 10, 20, 20, 1, 1.]), np.diag([.1, .1])); K = np.linalg.solve(np.diag([.1, .1]) + B.T @ Pm @ B, B.T @ Pm @ A)


def episode(plant, sc, seed):
    ft, fp = next((f, q) for i, f, q in SCENARIOS if i == sc)
    r = np.random.default_rng(seed); mr = np.random.default_rng([seed, 41])
    x = np.zeros(6); x[0] = r.uniform(-.3, .3); x[1] = r.uniform(-.8, .8)
    pl = FaultedPlant(plant, ft, fp, 50.); last, e2 = 0., 0.
    for k in range(500):
        s = measure(x, k, ft, fp, mr, last); last = float(s[0])
        x = pl.step(x, np.clip(-K @ s, -50, 50)); e2 += x[0] ** 2
        if abs(x[0]) > np.pi / 4: return 0, float(np.sqrt(e2 / (k + 1)))
    return 1, float(np.sqrt(e2 / 500))


def cl(plant, sc):
    r = [episode(plant, sc, s) for s in range(a.n_eps)]
    return float(np.mean([q[0] for q in r])), float(np.mean([q[1] for q in r])), r


scs = ["healthy", "dropout_imu_p0.3", "stuck_phi_15deg"]
t3, ok = {}, True
for sc in scs:
    s0, r0, e0 = cl(nom, sc); s1, r1_, e1 = cl(mjr, sc)
    same = all(p[0] == q[0] for p, q in zip(e0, e1)); d = float(np.mean([abs(p[1] - q[1]) for p, q in zip(e0, e1)]))
    t3[sc] = dict(project_surv=s0, mj_reduced_surv=s1, project_rmse=r0, mj_reduced_rmse=r1_, per_episode_survival_identical=bool(same), mean_abs_drmse=d)
    ok &= bool(same and d < 1e-3)
rep["T3"] = dict(per_scenario=t3, PASS=bool(ok))

# T4 (info)
g = lambda p: float(p.step(np.zeros(6), np.array([10., 0.]))[4])
x_free = {}
for nme, p in (("project", nom), ("mj_full", mjf)):
    x = np.array([.05, 0, 0, 0, 0, 0.]); crashed = False
    for _ in range(100):
        x = p.step(x, np.zeros(2))
        if abs(x[0]) > np.pi / 4: crashed = True; break
    x_free[nme] = dict(phi_after_1s=float(x[0]), crashed=crashed)
gyro = P["Iw"] * P["Omega"]
t4 = dict(gimbal_rate_after_one_step_tau10=dict(project=g(nom), mj_reduced=g(mjr), mj_full=g(mjf)),
          unactuated_bike_from_phi0_0p05_for_1s=x_free,
          gyroscopic_reaction_on_gimbal_Nm_per_rad_s=gyro,
          estimated_gyro_spring_rad_s=float(gyro / np.sqrt(P["m"] * P["h"] ** 2 * P["Iw"] / 2)),
          closed_loop_lqr={sc: dict(zip(("surv", "rmse"), cl(mjf, sc)[:2])) for sc in scs})
rep["T4_info_full_mode"] = t4
json.dump(rep, open(a.out, "w"), indent=2)
print(json.dumps(rep, indent=2))
print("\nT1", "PASS" if rep["T1"]["PASS"] else "FAIL", "| T2", "PASS" if rep["T2"]["PASS"] else "FAIL", "| T3", "PASS" if rep["T3"]["PASS"] else "FAIL")
