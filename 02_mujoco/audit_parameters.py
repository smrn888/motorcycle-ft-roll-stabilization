from pathlib import Path
import importlib.util, csv, math
source = Path(__file__).resolve().parent.parent / 'motorcycle_dynamics(2).py'
spec=importlib.util.spec_from_file_location('original_motorcycle_dynamics',source)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
p=mod.PARAMS
rows=[
 ('total_mass_kg',p['m'],'given','Mass of motorcycle + rider in reduced model; whether CMGs included is unspecified'),
 ('center_of_mass_height_m',p['h'],'given','Reduced-model effective COM height; not a 3D COM measurement'),
 ('wheelbase_m',p['L'],'given','Wheelbase'),
 ('rotor_axial_inertia_each_kg_m2',p['Iw'],'given','Interpretation as each rotor requires confirmation'),
 ('rotor_spin_rad_s',p['Omega'],'given','Magnitude; signs of counter-rotation not specified'),
 ('gimbal_inertia_kg_m2',p['I_delta'],'given','Effective gimbal inertia; not full 3D inertia tensor'),
 ('gimbal_torque_limit_Nm',mod.TAU_MAX,'given','Per actuator'),
 ('gimbal_angle_limit_rad',mod.DELTA_MAX,'given','Per gimbal'),
 ('gimbal_rate_limit_rad_s',mod.MAX_DELTA_DOT,'given','Per gimbal'),
 ('gimbal_damping_per_s',mod.GIMBAL_DAMP,'given','Effective rate damping, not joint viscous coefficient'),
 ('controller_dt_s',mod.Ts,'given','Controller sampling interval'),
 ('target_forward_speed_m_s',3.0,'user_selected','Desired operating speed'),
 ('wheel_radius_m','','missing','Measure actual front/rear radii'),
 ('wheel_mass_kg','','missing','Measure front/rear wheel masses'),
 ('chassis_3d_inertia_kg_m2','','missing','Ixx,Iyy,Izz and off-diagonal terms'),
 ('chassis_geometry','','missing','CAD/physical dimensions and frame orientation'),
 ('steering_geometry','','missing','Steer axis inclination, trail, offsets'),
 ('tire_contact_model','','missing','Lateral slip, friction and stiffness'),
 ('rotor_geometry_and_mass','','missing','Needed to realize Iw=0.5 physically'),
 ('gimbal_axis_3d_orientation','','missing','Must match signs in reduced model'),
 ('rotor_spin_direction','','missing','Need both signed angular velocities'),
 ('motor_actuator_dynamics','','missing','Torque response, speed regulation and delay'),
 ('vehicle_drive_and_speed_control','','missing','Traction / rear-wheel drive parameters'),
]
with open(Path(__file__).with_name('parameter_audit.csv'),'w',newline='',encoding='utf-8-sig') as f:
 w=csv.writer(f);w.writerow(['parameter','value','status','interpretation_or_needed_measurement']);w.writerows(rows)
print('Reduced-model roll inertia =',p['m']*p['h']**2,'kg m^2')
print('Reduced-model gyro angular momentum =',p['Iw']*p['Omega'],'kg m^2/s')
print('Audit written:',Path(__file__).with_name('parameter_audit.csv'))
