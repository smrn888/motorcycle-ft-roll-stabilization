import numpy as np
from motorcycle_dynamics import MotorcycleDynamics
from mj_dynamics import MuJoCoDynamics
import importlib.util,sys,os
raw=MuJoCoDynamics('full'); srv=MuJoCoDynamics('full',cancel_gimbal_reaction=True); ref=MotorcycleDynamics()
# default unchanged vs original file
spec=importlib.util.spec_from_file_location('orig',os.path.join(os.path.dirname(__file__),'mj_dynamics_orig.py')); om=importlib.util.module_from_spec(spec); spec.loader.exec_module(om)
orig=om.MuJoCoDynamics('full'); x=np.array([.06,.1,.1,-.1,2.,-2.]); u=np.array([4.,-4.])
print('default bit-identical to original:',np.array_equal(orig.step(x,u),raw.step(x,u)))
for name,fn,N in [('zero',lambda k:np.zeros(2),100),('pulse1',lambda k:np.array([1.,-1.]) if k<10 else np.zeros(2),100)]:
    xs=[np.array([.05,0,0,0,0,0.])]*3; E={}
    for nm,p in (('ref',ref),('raw',raw),('servo',srv)):
        x=np.array([.05,0,0,0,0,0.]); tr=[x]
        for k in range(N): x=p.step(x,fn(k)); tr.append(x)
        E[nm]=np.array(tr)
    print(name,'max|dphi| raw %.2e servo %.2e'%(abs(E['raw'][:,0]-E['ref'][:,0]).max(),abs(E['servo'][:,0]-E['ref'][:,0]).max()))
