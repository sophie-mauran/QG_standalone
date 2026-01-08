import sys
import numpy as np
from netCDF4 import Dataset

from tools.pre_process import add_noise
from tools.randvars import GaussRV

N = int(sys.argv[1])-1
Nx = int(sys.argv[2])
t_end = sys.argv[3]


order='F'
def f2py(X): return X.flatten(order=order)



E=np.empty([N,Nx])

for k in range(2,N+2):
    file_member_k = "/home/smauran/Developpement/QG_standalone_counillon_assim4pdt/member_"+str(k)+"/run_t_"+str(t_end)+".nc"
    #print(file_member_i)
    with Dataset(file_member_k, 'r') as ncfile_member_k:

        psi_member_k = ncfile_member_k.variables['psi']
        
        member_k = f2py(psi_member_k[-1,:,:])

        
        E[k-2,:] = member_k.data

spread = np.std(E, axis=0)
spread = np.linalg.norm(spread)/(Nx)
print("spread", spread)