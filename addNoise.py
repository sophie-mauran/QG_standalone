import sys
import numpy as np
from netCDF4 import Dataset

from tools.pre_process import add_noise
from tools.randvars import GaussRV

N = int(sys.argv[1])
Nx = int(sys.argv[2])
t_burn_in = int(sys.argv[3])

i=int(np.sqrt(Nx))
j=i

order='F'
def f2py(X): return X.flatten(order=order)



E=np.empty([N,Nx])

for k in range(1,N+1):
    file_member_k = "/home/smauran/Developpement/QG_standalone_counillon_assim4pdt/member_"+str(k)+"/run_t_"+str(t_burn_in)+".nc"
    #print(file_member_i)
    with Dataset(file_member_k, 'r') as ncfile_member_k:

        psi_member_k = ncfile_member_k.variables['psi']
        
        member_k = f2py(psi_member_k[-1,:,:])

        
        E[k-1,:] = member_k.data


noise = GaussRV(C=np.eye(Nx))
E = add_noise(E, t_burn_in, noise, 'Stoch')


for k in range(1,N+1):
    file_member_k = "/home/smauran/Developpement/QG_standalone_counillon_assim4pdt/member_"+str(k)+"/run_t_"+str(t_burn_in)+".nc"
    #print(file_member_i)
    with Dataset(file_member_k, 'a') as ncfile_member_k:

        psi_member_k = ncfile_member_k.variables['psi']
        
        member_k = np.reshape(E[k-1,:], (j,i), 'F')

        psi_member_k[-1, ...] = member_k