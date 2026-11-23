"""MuJoCo physics behind the project plant interface.

MuJoCoDynamics subclasses the project's MotorcycleDynamics, so linearise(), f(), p, Ts ... still work and
FaultedPlant / RecordingPlant (final_fault_benchmark.py / compare_34faults_res_scale.py) can wrap it unchanged.
Only step(x, u) is replaced: it is STATELESS (x in -> x_next out), exactly like the project's RK4 step:
    1. write x = [phi, phi_dot, d_f, d_r, dd_f, dd_r] into MuJoCo qpos/qvel
    2. hold u constant (zero-order hold, as in the project) for Ts = substeps * MuJoCo timestep
    3. read x_next, apply the project's gimbal angle/rate limits once per Ts
Torque is NOT clipped here unless clamp_u=True (FaultedPlant passes the already clipped last_app, and
actuator-fault scenarios such as scale_gimbal_1.3 intentionally exceed 50 N m in the project plant).
"""
import numpy as np
import mujoco
from motorcycle_dynamics import MotorcycleDynamics, TAU_MAX, DELTA_MAX, MAX_DELTA_DOT
from mj_params import P, MJ_TIMESTEP
from mj_build_model import build_xml


class MuJoCoDynamics(MotorcycleDynamics):
    def __init__(self, mode="reduced", Iw_scale=1.0, m_scale=1.0, timestep=MJ_TIMESTEP,
                 compensate_flywheel=True, record=False, **kw):
        super().__init__(m_scale=m_scale, Iw_scale=Iw_scale, **kw)
        n = self.Ts / timestep
        if abs(n - round(n)) > 1e-9:
            raise ValueError("MuJoCo timestep must divide Ts exactly")
        self.mode, self.substeps, self.hh = mode, int(round(n)), timestep
        pp = dict(P); pp["m"] = P["m"] * m_scale
        self.m = mujoco.MjModel.from_xml_string(
            build_xml(pp, mode, Iw_scale=Iw_scale, timestep=timestep, compensate_flywheel=compensate_flywheel))
        self.d = mujoco.MjData(self.m)
        J = lambda name: self.m.joint(name)
        self.qa = {k: J(k).qposadr[0] for k in ("roll", "g_f", "g_r")}
        self.va = {k: J(k).dofadr[0] for k in ("roll", "g_f", "g_r")}
        if mode == "full":
            self.va["s_f"], self.va["s_r"] = J("s_f").dofadr[0], J("s_r").dofadr[0]
            self.qa["s_f"], self.qa["s_r"] = J("s_f").qposadr[0], J("s_r").qposadr[0]
        self.IwO_ref = P["Iw"] * P["Omega"] * Iw_scale          # reduced mode only (same as project IwO)
        self.I_delta, self.c_g = P["I_delta"], P["c"]
        self.record, self.trace_x, self.trace_u = record, [], []

    # ------------------------------------------------------------------ helpers
    def _write(self, x):
        d, q, v = self.d, self.qa, self.va
        d.qpos[q["roll"]], d.qvel[v["roll"]] = x[0], x[1]
        d.qpos[q["g_f"]], d.qpos[q["g_r"]] = x[2], x[3]
        d.qvel[v["g_f"]], d.qvel[v["g_r"]] = x[4], x[5]
        if self.mode == "full":                               # spin motors hold speed
            d.qvel[v["s_f"]], d.qvel[v["s_r"]] = +P["Omega"], -P["Omega"]

    def _read(self):
        d, q, v = self.d, self.qa, self.va
        return np.array([d.qpos[q["roll"]], d.qvel[v["roll"]], d.qpos[q["g_f"]], d.qpos[q["g_r"]],
                         d.qvel[v["g_f"]], d.qvel[v["g_r"]]])

    # ------------------------------------------------------------------ interface
    def step(self, x, u, clamp_u=False):
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        if clamp_u:
            u = np.clip(u, -TAU_MAX, TAU_MAX)
        d, m, v, q, hh = self.d, self.m, self.va, self.qa, self.hh
        self._write(x)
        d.ctrl[0], d.ctrl[1] = u[0], u[1]
        d.qfrc_applied[:] = 0.0
        for _ in range(self.substeps):
            if self.mode == "reduced":
                # project equation: phi_dd = (m g h sin(phi) - Iw*Om*(dd_f-dd_r)*cos(phi)) / (m h^2)
                # gravity/inertia come from the MJCF; the CMG term is a generalized force held over the sub-step.
                # Gimbals are linear and decoupled from roll here, so their MEAN rate over the sub-step is known exactly;
                # cos(phi) is evaluated at the sub-step midpoint (2nd-order Taylor).
                cg = self.c_g
                e = (1.0 - np.exp(-cg * hh)) / (cg * hh)
                ss_f, ss_r = u[0] / (cg * self.I_delta), u[1] / (cg * self.I_delta)
                w_f = ss_f + (d.qvel[v["g_f"]] - ss_f) * e
                w_r = ss_r + (d.qvel[v["g_r"]] - ss_r) * e
                ph, pd = d.qpos[q["roll"]], d.qvel[v["roll"]]
                acc = (self.mgh * np.sin(ph) - self.IwO_ref * (d.qvel[v["g_f"]] - d.qvel[v["g_r"]]) * np.cos(ph)) / self.Iphi
                d.qfrc_applied[v["roll"]] = -self.IwO_ref * (w_f - w_r) * np.cos(ph + 0.5 * hh * pd + 0.125 * hh * hh * acc)
            mujoco.mj_step(m, d)
        xn = self._read()
        if self.enforce_limits:
            xn[2:4] = np.clip(xn[2:4], -DELTA_MAX, DELTA_MAX)
            xn[4:6] = np.clip(xn[4:6], -MAX_DELTA_DOT, MAX_DELTA_DOT)
        if self.record:
            self.trace_x.append(xn.copy()); self.trace_u.append(u.copy())
        return xn
