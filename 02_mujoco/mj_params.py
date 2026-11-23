"""MuJoCo-side parameters. Everything physical is IMPORTED from the project's motorcycle_dynamics.py (never copied),
so the MuJoCo plant cannot drift away from the paper model. Run from the project folder ("Three Models")."""
import numpy as np
import motorcycle_dynamics as md

P = dict(dt=md.Ts, max_steps=500, crash=np.pi / 4, tau_max=md.TAU_MAX, **md.PARAMS,
         c=md.GIMBAL_DAMP, delta_max=md.DELTA_MAX, ddelta_max=md.MAX_DELTA_DOT,
         x_front=md.PARAMS["L"] / 2, x_rear=-md.PARAMS["L"] / 2)   # x_front/x_rear: geometry only (no dynamic effect)

MJ_TIMESTEP = 0.001   # MuJoCo internal step. Project Ts (0.01) = 10 sub-steps; must divide Ts exactly.
