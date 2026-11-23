"""Replay a trace saved by run_in_mujoco.py (--mj_trace_dir) as MP4 (needs: pip install imageio imageio-ffmpeg).
   python mj_render.py mj_traces\\full_none__b0_s77000.npz --out episode.mp4 [--azimuth 0]
Headless Linux: set MUJOCO_GL=egl. On Windows the default GLFW backend works."""
import os, sys
sys.path.insert(0, os.getcwd())          # project modules live in the cwd ("Three Models")
import argparse, numpy as np, mujoco, imageio
from mj_build_model import build_xml
from mj_params import P

ap = argparse.ArgumentParser(); ap.add_argument("trace"); ap.add_argument("--out", default="episode.mp4")
ap.add_argument("--azimuth", type=float, default=0.)   # 0 = front view (roll lean visible); 90 = side view
ap.add_argument("--every", type=int, default=2); a = ap.parse_args()
z = np.load(a.trace, allow_pickle=True); S = z["states"]; mode = str(z["mode"])
m = mujoco.MjModel.from_xml_string(build_xml(P, mode)); d = mujoco.MjData(m)
q = lambda n: m.joint(n).qposadr[0]
r = mujoco.Renderer(m, 480, 640); cam = mujoco.MjvCamera()
cam.lookat[:] = [0, 0, 0.6]; cam.distance = 4.0; cam.azimuth = a.azimuth; cam.elevation = -5
frames = []
for k, x in enumerate(S):
    if k % a.every: continue
    d.qpos[q("roll")], d.qpos[q("g_f")], d.qpos[q("g_r")] = x[0], x[2], x[3]
    if mode == "full": d.qpos[q("s_f")], d.qpos[q("s_r")] = (P["Omega"] * k * P["dt"]) % (2 * np.pi), (-P["Omega"] * k * P["dt"]) % (2 * np.pi)
    mujoco.mj_forward(m, d); r.update_scene(d, cam); frames.append(r.render())
imageio.mimsave(a.out, frames, fps=int(round(1 / (P["dt"] * a.every)))); print("saved", a.out, len(frames), "frames")
