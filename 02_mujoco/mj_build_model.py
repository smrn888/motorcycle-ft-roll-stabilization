"""MJCF generator for the CMG motorcycle. No 3D asset is needed: every geom is a primitive and purely visual
(contype=0, no contacts, contact disabled).

mode="reduced"  The paper equations inside MuJoCo (numerical-equivalence / regression test of the integrator).
    roll hinge on the ground line = inverted pendulum: gravity + inertia give m g h sin(phi) / (m h^2) exactly.
    gimbals: hinge joints, armature=I_delta, damping=c*I_delta  ->  ddelta = tau/I_delta - c*delta_dot.
    The CMG roll torque  -Iw*Omega*(dd_f - dd_r)*cos(phi)  is added by mj_dynamics.py as a generalized force.
mode="full"     Independent physics: two real spinning flywheels (front +Omega, rear -Omega) on the gimbals.
    MuJoCo itself computes the gyroscopic coupling, including the reaction torque on the gimbal axis (H*phi_dot)
    and the cos(delta) dependence, none of which is in the reduced equations. Two calibrations keep the paper's numbers:
      (a) gimbal authority: a rigid flywheel with spin inertia Iw has transverse inertia of at least Iw/2 (Iz <= Ix + Iy),
          which exceeds I_delta, so the actuator gear is set to I_total/I_delta. The commanded gimbal acceleration is then
          exactly tau/I_delta as in the paper model; damping is scaled the same way.
      (b) roll inertia: the flywheels add inertia about the roll axis, so the bike body mass and COM height are re-solved
          to keep total hinge inertia = m h^2 and gravity torque = m g h at delta = 0 (compensate_flywheel=True).
"""
from mj_params import P, MJ_TIMESTEP


def flywheel_numbers(p, Iw_scale):
    Iw = p["Iw"] * Iw_scale
    I_trans = Iw / 2.0                      # thin disc: transverse inertia Iw/2
    return Iw, I_trans, I_trans + 1e-6      # (spin inertia, transverse inertia, inertia about the gimbal axis)


def build_xml(p=P, mode="reduced", Iw_scale=1.0, timestep=MJ_TIMESTEP, compensate_flywheel=True, visual=True, gimbal_limit=None):
    assert mode in ("reduced", "full")
    m, h, L, c, Id = p["m"], p["h"], p["L"], p["c"], p["I_delta"]
    Iw, I_trans, I_gt = flywheel_numbers(p, Iw_scale)

    if mode == "full" and compensate_flywheel:
        I_fw_roll = 2.0 * I_trans                  # both flywheels about the roll axis at delta = 0
        h_b = h - I_fw_roll / (m * h)              # m_b*h_b = m*h  and  m_b*h_b^2 + I_fw_roll = m*h^2
        if h_b <= 0:
            raise ValueError("Flywheel inertia is too large for positive body COM compensation; reduce Iw_scale or disable compensation")
        m_b = m * h / h_b
    else:
        h_b, m_b = h, m

    if mode == "reduced":
        gear, damp = 1.0, c * Id
    else:
        gear, damp = I_gt / Id, c * I_gt

    vis = (lambda s: s) if visual else (lambda s: "")

    def gimbal(name, x):
        if mode == "reduced":
            return f"""
      <body name="gimbal_{name}" pos="{x} 0 {h}">
        <joint name="g_{name}" type="hinge" axis="0 1 0" armature="{Id}" damping="{damp}"/>
        <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-9 1e-9 1e-9"/>
        {vis('<geom type="cylinder" size="0.12 0.03" euler="1.5708 0 0" rgba="0.95 0.55 0.1 1" contype="0" conaffinity="0"/>')}
        {vis('<geom type="box" pos="0.12 0 0" size="0.03 0.04 0.01" rgba="1 1 1 1" contype="0" conaffinity="0"/>')}
      </body>"""
        lim = "" if gimbal_limit is None else f' limited="true" range="{-gimbal_limit} {gimbal_limit}" solreflimit="0.02 1"'
        return f"""
      <body name="gimbal_{name}" pos="{x} 0 {h}">
        <joint name="g_{name}" type="hinge" axis="0 1 0" damping="{damp}"{lim}/>
        <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-9 1e-9 1e-9"/>
        <body name="fw_{name}">
          <joint name="s_{name}" type="hinge" axis="0 0 1" damping="0"/>
          <inertial pos="0 0 0" mass="1e-6" diaginertia="{I_trans} {I_trans} {Iw}"/>
          {vis('<geom type="cylinder" size="0.12 0.03" rgba="0.95 0.55 0.1 1" contype="0" conaffinity="0"/>')}
          {vis('<geom type="box" pos="0.12 0 0" size="0.03 0.04 0.02" rgba="1 1 1 1" contype="0" conaffinity="0"/>')}
        </body>
      </body>"""

    body_geoms = vis(f"""
      <geom type="cylinder" pos="{L/2} 0 0.3" euler="1.5708 0 0" size="0.3 0.04" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="{-L/2} 0 0.3" euler="1.5708 0 0" size="0.3 0.04" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0"/>
      <geom type="box" pos="0 0 0.45" size="{L/2} 0.04 0.06" rgba="0.2 0.3 0.8 1" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0 0 0.5 0 0 {h+0.35}" size="0.09" rgba="0.8 0.2 0.2 1" contype="0" conaffinity="0"/>
      <geom type="sphere" pos="0 0 {h+0.5}" size="0.1" rgba="0.9 0.75 0.6 1" contype="0" conaffinity="0"/>""")

    return f"""
<mujoco model="motorcycle_cmg_{mode}">
  <compiler angle="radian"/>
  <option timestep="{timestep}" gravity="0 0 -{p['g']}" integrator="RK4">
    <flag contact="disable"/>
  </option>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <worldbody>
    <light pos="0 -3 4" dir="0 0.6 -0.8"/>
    <geom type="plane" size="6 6 .1" rgba=".85 .85 .85 1" contype="0" conaffinity="0"/>
    <body name="bike" pos="0 0 0">
      <joint name="roll" type="hinge" axis="1 0 0" damping="0"/>
      <inertial pos="0 0 {h_b}" mass="{m_b}" diaginertia="1e-6 1e-6 1e-6"/>
      {body_geoms}
      {gimbal('f', p['x_front'])}
      {gimbal('r', p['x_rear'])}
    </body>
  </worldbody>
  <actuator>
    <motor name="tau_f" joint="g_f" gear="{gear}" ctrllimited="false"/>
    <motor name="tau_r" joint="g_r" gear="{gear}" ctrllimited="false"/>
  </actuator>
</mujoco>"""
