import sys
import os
import numpy as np
import numpy.random as rnd
from netCDF4 import Dataset

# Open the NetCDF file in read mode
file_start_ens = sys.argv[1]
N = int(sys.argv[2])

seed = int(os.environ.get("RANDOM_SEED", sys.argv[2]))
#print(seed)

# Provided by numpy file with sample
data   = np.load(file_start_ens)
sample = data['sample']
N0     = len(sample)
if 'w' in data:
    w = data['w']
else:
    w = np.ones(N0)/N0
idx = rnd.choice(N0, N, replace=True, p=w)
E = sample[idx]


for k in range(1,N+1):
    file_member_k = "/home/smauran/Developpement/QG_standalone_counillon_assim4pdt/member_"+str(k)+"/start_run.nc"
    
    with Dataset(file_member_k, 'a') as ncfile_member_k:

        psi_member_k = ncfile_member_k.variables['psi']
        j = psi_member_k.shape[1]
    
        i = psi_member_k.shape[2]
        
        member_k = np.reshape(E[k-1,:], (j,i), 'F')

        psi_member_k[-1, ...] = member_k

