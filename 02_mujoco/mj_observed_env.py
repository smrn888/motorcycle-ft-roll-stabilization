"""Swap the project's physics for MuJoCo WITHOUT touching controllers, EKF, faults, initial conditions or noise.

The project's ObservedFaultEnv (compare_34faults_res_scale.py) wraps its plant in RecordingPlant/FaultedPlant and calls
plant.step(state, u). We replace the wrapped plant by MuJoCoDynamics (same step(x,u) interface). Everything else
(IC buckets, v/alpha noise, IMU fault generator, Neural MPC, SAC, Online MPC, EKF, metrics) is the project's own code, so for
the same seed the episode differs from the reference run ONLY through the physics engine -> paired comparison.
"""
import atexit
import os
import numpy as np
import compare_34faults_res_scale as c34
from motorcycle_dynamics import TAU_MAX
from mj_dynamics import MuJoCoDynamics


def make_env_class(mode="reduced", trace_dir=None, trace_max=5, timestep=None, cancel_gimbal_reaction=False, gimbal_limit=None, supervisor=False):
    Base = c34.ObservedFaultEnv

    class MuJoCoObservedFaultEnv(Base):
        MJ_MODE = mode

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._mj, self._trace_n, self._meta, self._x0, self._plant = {}, 0, None, None, None
            if trace_dir:
                os.makedirs(trace_dir, exist_ok=True)
                atexit.register(self._flush_trace)

        def _flush_trace(self):
            pl = self._plant
            if not trace_dir or pl is None or not pl.trace_x or self._trace_n >= trace_max or self._meta is None:
                return
            ft = self._meta["scenario"]
            name = f"{mode}{'_servo' if cancel_gimbal_reaction else ''}_{ft}_b{self._meta['bucket']}_s{self._meta['seed']}.npz"
            np.savez(os.path.join(trace_dir, name), states=np.vstack([self._x0, np.asarray(pl.trace_x)]),
                     u=np.asarray(pl.trace_u), mode=mode, **{k: v for k, v in self._meta.items()})
            self._trace_n += 1

        def reset(self, seed=None, options=None):
            self._flush_trace()
            out = super().reset(seed=seed, options=options)
            ft, fp = self.scenario
            if ft == "param_fault":
                raise NotImplementedError("param_fault scenarios build their own dynamics; not supported in the MuJoCo wrapper")
            opts = options or {}
            iw = float(opts.get("iw", 1.0))
            if iw not in self._mj:
                kw = {} if timestep is None else {"timestep": timestep}
                if cancel_gimbal_reaction: kw["cancel_gimbal_reaction"] = True
                if gimbal_limit is not None: kw["gimbal_limit"] = gimbal_limit
                if supervisor: kw["supervisor"] = True
                self._mj[iw] = MuJoCoDynamics(mode=mode, Iw_scale=iw, record=bool(trace_dir), **kw)
            plant = self._mj[iw]
            plant.trace_x, plant.trace_u = [], []
            self._plant = plant
            self._x0 = np.array(self.state[:6], float)
            self._meta = dict(scenario=str(self._scenario_id()), bucket=int(opts.get("bucket", 0)),
                              seed=int(seed) if seed is not None else -1, iw=iw)
            self._plant_dyn = c34.RecordingPlant(plant, ft, fp, TAU_MAX)
            return out

        def _scenario_id(self):
            ft, fp = self.scenario
            return ft + "_" + "_".join(f"{k}{v:g}" for k, v in sorted(fp.items()) if isinstance(v, (int, float)))

    return MuJoCoObservedFaultEnv
