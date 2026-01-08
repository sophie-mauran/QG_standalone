import sys
from netCDF4 import Dataset
import numpy as np
import scipy.linalg as sla
import os
import concurrent.futures
import csv
import random
import statistics as stats
import numpy.random as rnd
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
#import cvxpy as cp
from tools.matrices import genOG_1, genOG, genOG_MN
from collections import Counter

from tools.randvars import GaussRV
from tools.localisation import nd_Id_cluster_loc_state, nd_Id_localization_cluster_obs, nd_Id_localization
from tools.linalg import mldiv, pad0, svd0
from tools.post_process import post_process
from tools.pre_process import add_noise
from tools.subspace_iteration import Covariance_subspace_iteration
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, MaxAbsScaler
from skimage import measure
from matplotlib.path import Path
from math import ceil
from matspy import spy

from scipy import stats as scistats
from scipy import linalg as scila
import matplotlib.pyplot as plt 
#from multiprocessing.pool import ThreadPool as Pool
import multiprocessing as mp

np.set_printoptions(threshold=sys.maxsize)

random.seed(3000)

obs_record = sys.argv[1]
N = int(sys.argv[2])-1
tend = float(sys.argv[3])
dtout = float(sys.argv[4])
n = int(sys.argv[5])
scaler_string="StandardS"

order='F'
def f2py(X): return X.flatten(order=order)

def equi_spaced_integers(m,p):
    """Provide a range of p equispaced integers between 0 and m-1"""
    return np.round(np.linspace(np.floor(m/p/2),np.ceil(m-m/p/2-1),p)).astype(int)


# Get cycle's observation
print("cycle's obs")
with Dataset(obs_record, 'r') as ncfile_obs:
    # Retrieve the last step of the model
    psi_obs = ncfile_obs.variables['psi']
    #print("q",q_obs)
    j_dom = psi_obs.shape[1]
    #print("j",j)
    i_dom = psi_obs.shape[2]
    #print("i",i)
    
    M = i_dom*j_dom
    print(M)
    shape=(j_dom,i_dom)
    # This will look like satellite tracks when plotted in 2D
    Ny = 300
    jj = equi_spaced_integers(M, Ny)

    
    # Want: random_offset(t1)==random_offset(t2) if t1==t2.
    # Solutions: (1) use caching (ensure maxsize=inf) or (2) stream seeding.
    # Either way, use a local random stream to avoid interfering with global stream
    # (and e.g. ensure equal outcomes for 1st and 2nd run of the python session).
    rstream = np.random.RandomState()
    max_offset = min(jj[1]-jj[0],M-jj[-1])


    def random_offset(tend,dtout): 
        rstream.seed(int(tend/dtout*100)) 
        u = rstream.rand()
        return int(np.floor(max_offset * u))

    
    def obs_inds(tend,dtout):
        return jj + random_offset(tend,dtout)


    obs_ind = obs_inds(tend,dtout)
    
    
    obs = f2py(psi_obs[0, ...])
    truth = obs.data
    
    obs = obs[obs_ind]
    obs = obs.data
    obs = obs + GaussRV(C=4*np.eye(Ny)).sample(1)
    obs = obs.reshape((Ny,))
    


#print(obs_ind) 

# Getting ensemble

print("getting ensemble")
E=np.empty([N,M])

for k in range(2,N+2):
    file_member_k = "/home/esimon/MdC/Codes/QG_Cluster/QG_standalone/member_"+str(k)+"/run_t_"+str(tend)+".nc"
    #print(file_member_i)
    with Dataset(file_member_k, 'r') as ncfile_member_k:

        psi_member_k = ncfile_member_k.variables['psi']
        
        member_k = f2py(psi_member_k[-1,:,:])

        
        E[k-2,:] = member_k.data

spread= np.std(E, axis=0)
spread = np.linalg.norm(spread)/(M)

#with open('ens_av_assim.npy', 'rb') as f:
     #E = np.load(f)

#print("E avant assim",E[0,:])


# Kernel definition
def kernel(x,y,params=dict()):
    return np.dot(x,y)

params_kernel = dict()
# Assimilation

def kernel_exp(x,y,params):
    alpha = params["alpha"]
    spread_p = params["spread"]
    eps = alpha*spread_p
    tmp=np.linalg.norm(x-y)
    return np.exp(-(tmp/eps)**2)

params_kernel_exp = dict(alpha=1.0, spread = 0.1)

def kernel_poly(x,y,params):
    d = params["d"]
    return (1+y@x.T)**d

params_kernel_poly = dict(d=2)

def myftanh(x,c):
    tmp=c*np.linalg.norm(x)
    scaling=0.0
    if(tmp>np.finfo(float).eps):
        scaling=np.arctanh(tmp)/tmp
    return scaling*x

def tanh_kernel(x,y,nu):
    return kernel(myftanh(x,np.sqrt(nu)),myftanh(y,np.sqrt(nu)),dict())   
params_kernel_tanh=10e-3#/np.sqrt(M)
    
R_obs=5
R_state=5
batch_shape = [2, 2]
infl=1.04
advanced_sampling=1
#infl=1.0
seuil = 10




def get_clusters(E,loc,rad,taper,j,i):
    # clustering sur les variables
    N, Nx = E.shape
    #print(i,j) 
    
    
    dist_xx, state_taperer = loc(rad,taper)

    nblimit=5
    vec_bouldin=np.zeros((nblimit,1))

    for k in range(1,nblimit):
        nb=np.int64(k+1)
        kmeans = KMeans(n_clusters=nb, random_state=0, n_init = 'auto').fit(E.T)
        vec_bouldin[k]=davies_bouldin_score(E.T,kmeans.labels_)

    #tol = 5*10**-2

    ''' min_s=[]
    for k in range(2,nblimit):
        if abs(vec_bouldin[k]-vec_bouldin[nblimit-1]) <tol:
            min_s.append(kregarder) '''
    #print(min_s) 

    #nb_min=min_s[0]
    nb_min=np.argmin(vec_bouldin[2:])+3
    #print("nb min", nb_min)

    kmeans = KMeans(n_clusters=nb_min, random_state=0, n_init = 'auto').fit(E.T)

    # 2me passe de clustering avec les contours
    
    labels = np.zeros(Nx)
    nb_clusters = 1

    

    # Clustering geometrique via detection de contours
    for c in range(nb_min):
        vec=np.zeros((i*j,1))
        # extraction des elements de la classe i
        inds = [index for (index,cluster) in enumerate(kmeans.labels_) if cluster == c]
        #dist_xx_ii = dist_xx[:,ind]
        #dist_xx_ii = dist_xx_ii[ind,:]

        vec[inds]=1
        Imclusk=np.reshape(vec,(j,i),order='F')


        # detection des contours fermes des classes
        contours = measure.find_contours(Imclusk,0)
        print('Nbre de contours pour la classe', np.int64(c),'=', len(contours))
        for num_contour in range(len(contours)):
                non_vide=False
                l_abs=[]
                l_ord=[]
                for ind in inds: # quels points du cluster sont dans le contour considéré ?
                    abs = ind%j
                    ord = ind//i
                    cluster_contour = np.array(contours[num_contour]).reshape(-1, 2)
                    path = Path(cluster_contour)
                    if path.contains_point((abs,ord)):
                            non_vide=True
                            l_abs.append(abs)
                            l_ord.append(ord)
                            labels[ind] = nb_clusters

                if non_vide:
                    inds_maj = [index for (index,cluster) in enumerate(labels) if cluster == nb_clusters]
                    abs_min = min(l_abs)
                    abs_max = max(l_abs)
                    ord_min = min(l_ord)
                    ord_max = max(l_ord)
                    # Idée pour les clusters trop petits : les mettre à -1 tant qu'on n'a pas trouvé le cluster vide, puis les rajouter au cluster vide
                    if len(inds_maj) < 0.001*i*j:
                        labels[inds_maj] = np.zeros(len(inds_maj))
                        

                    elif abs_max - abs_min > seuil or ord_max - ord_min > seuil :
                            dist_xx_ii = dist_xx[:,inds_maj]
                            dist_xx_ii = dist_xx_ii[inds_maj,:]
                            nc = int(ceil((abs_max - abs_min)/seuil)*ceil((ord_max - ord_min)/seuil))
                            kmeans_dist = KMeans(n_clusters=nc, random_state=0, n_init = 'auto').fit(dist_xx_ii)
                            labels[inds_maj] = nb_clusters*np.ones(len(inds_maj)) + kmeans_dist.labels_
                            nb_clusters += kmeans_dist.n_clusters
                    else:
                            nb_clusters += 1



    ## On s'occupe du contour "exterieur", label 0
    inds_0 = [index for (index,cluster) in enumerate(labels) if cluster == 0]
    l_abs_0=[]
    l_ord_0=[]
    for ind in inds_0:
        abs = ind%i
        ord = ind//j
        l_abs_0.append(abs)
        l_ord_0.append(ord)

    abs_min_0 = min(l_abs_0)
    abs_max_0 = max(l_abs_0)
    ord_min_0 = min(l_abs_0)
    ord_max_0 = max(l_abs_0)
    
    if abs_max_0 - abs_min_0 > seuil or ord_max_0 - ord_min_0 > seuil :

        dist_xx_ii = dist_xx[:,inds_0]
        dist_xx_ii = dist_xx_ii[inds_0,:]
        nc = int(ceil((abs_max_0 - abs_min_0)/seuil)*ceil((ord_max_0 - ord_min_0)/seuil))
        kmeans_dist = KMeans(n_clusters=nc, random_state=0, n_init = 'auto').fit(dist_xx_ii)
        
        labels[inds_0] = nb_clusters*np.ones(len(inds_0)) + kmeans_dist.labels_
        nb_clusters += kmeans_dist.n_clusters

    #print(davies_bouldin_score(E.T,labels))
    print("label max apres prise en compte des contours et redecoupage des gros clusters : ", nb_clusters)



    unique_labels = np.unique(labels)
    print("nb clusters reel", len(unique_labels))
    return(labels, nb_clusters, len(unique_labels),state_taperer)




###############################################################
##############################################################
##### script assimilation


xN = 1.0
g = 0
taper_s='GC'
taper_o='GC'
_map = map

hnoise=GaussRV(C=4*np.eye(Ny))
R, N1 = hnoise.C, N-1
R_tilde = R.sym_sqrt_inv
mu = np.mean(E, 0)
 
#Moderation warm-up EHO
#if(tend<430):
#    R_tilde=R_tilde/np.sqrt(8.0)
#elif(tend<470):
#    R_tilde=R_tilde/2.0
#elif(tend<510):
#    R_tilde=R_tilde/np.sqrt(2.0)
 
A  = E - mu
Eo=E[:,obs_ind]
xo = np.mean(Eo, 0)  # Obs ens mean
y=obs
Y  = Eo-xo           # Obs ens anomalies (HX_f)
dy = (y - xo) #@ R.sym_sqrt_inv.T



scalerString=''
if scalerString == 'NormalizerL1':
    scaler = Normalizer(norm='l1')
elif scalerString == 'NormalizerL2':
    scaler = Normalizer(norm='l2')
elif scalerString == 'NormalizerMAX':
    scaler = Normalizer(norm='max')
elif scalerString == 'StandardS':
    scaler = StandardScaler()
elif scalerString == 'MinMaxS':
    scaler=MinMaxScaler()
elif scalerString == 'MaxAbsS':
    scaler = MaxAbsScaler()
else:
    scaler = None

#scaling prior to clustering
if scaler != None:
    E_scaled = scaler.fit_transform(E)
else:
    E_scaled =E

#clustering
labels,nb_clusters, nb_clusters_reel, state_taperer = get_clusters(E_scaled,nd_Id_cluster_loc_state(shape[::-1], periodic=False),R_state,taper_s,j_dom,i_dom)

def my_state_taperer(ii):
    ind_ii,tapering=state_taperer(ii)
    return ind_ii,tapering
    

print('Clustering done')

loc2=nd_Id_localization_cluster_obs(shape[::-1], obs_ind, periodic=False)
obs_taperer = loc2(R_obs, 'x2y', tend, taper_o)
jjj=[]
def my_obs_taperer(jj):
    ind_jj,tapering =obs_taperer(jj)
    return ind_jj,tapering
    


# recuperation des clusters	
state_batches = []
obs_batches = []
#Index_ech=[]
for c in range(nb_clusters):
    i_cluster = [index for (index,cluster) in enumerate(labels) if cluster == c]
    state_batches.append(i_cluster)
    if len(i_cluster)!=0:
        #Index_ech.append(i_cluster[0])
        #Index_ech.append(i_cluster[-1])
        ind_jj,_= my_obs_taperer(i_cluster)
        obs_batches.append(ind_jj)
#nb_obs_batches=[]
#nb_state_batches=[]
#print("clusters d'obs", obs_batches)
indices_obs = [ind_obs for obs_batch in obs_batches for ind_obs in obs_batch]
occ_obs = Counter(indices_obs)
occ_obs_diag = [1/np.sqrt(occ_obs[i]) for i in range(Ny)]
occ_obs_diag = np.diag(occ_obs_diag)
R_tilde = R_tilde@occ_obs_diag

# For Kernel analysis
H_bar = (Y@R_tilde)
d_bar = (R_tilde@dy)
n_obs=len(H_bar[0,:])
#scaling anomiles
E_aug = np.concatenate((A, H_bar),axis = 1)
if scaler != None:
    E_aug = scaler.fit_transform(E_aug)
    A=E_aug[:,:M]
    H_bar=E_aug[:,M:]
    
#for c in state_batches:
    #if len(c) != 0:
        #ind_jj,_= my_obs_taperer(c)
        #nb_obs_batches.append(len(jj) for jj in jjj)
        #nb_state_batches.append(len(c))
	
#statistics forecast	
mean_av_assim = np.mean(E,0)
rmse_av_assim = np.linalg.norm(mean_av_assim-truth)/len(mean_av_assim)
spread_av_assim = np.std(E, axis=0)
spread_av_assim = np.linalg.norm(spread_av_assim)/(i_dom*j_dom)
print('rmse forecast',rmse_av_assim)
print('spread forecast',spread_av_assim)

def assembling_Kloc(ii,func_Stap, func_Otap, loc_kernel,par_kernel):
    if len(ii) != 0:
        ii_tapered, state_tapering =func_Stap(ii)
	    
        ind_jj,tapering=func_Otap(ii)
        Njj=len(ind_jj)
        Nii=len(ii_tapered)
	     
        #state_tapering[:Nii]=1.0
        #tapering[:Njj]=1.0
	     
        len_vec=int(Nii*(Nii+1)/2+Nii*Njj+Njj*(Njj+1)/2)
        row_loc=np.zeros((len_vec,),dtype=int)
        col_loc=np.zeros((len_vec,),dtype=int)
        Ktmp_loc=np.zeros((len_vec,))
	     
        cpt=0
        for i in range(Nii):
            for j in range(i+1):
                row_loc[cpt]=ii_tapered[i]
                col_loc[cpt]=ii_tapered[j]
                Ktmp_loc[cpt]=loc_kernel(A[:,ii_tapered[i]]*np.sqrt(state_tapering[i]),A[:,ii_tapered[j]]*np.sqrt(state_tapering[j]),par_kernel)  		         
                cpt+=1      
            for j in range(Njj): 
                Ktmp_loc[cpt]=loc_kernel(A[:,ii_tapered[i]]*np.sqrt(state_tapering[i]),H_bar[:,ind_jj[j]]*np.sqrt(tapering[j]),par_kernel)
                row_loc[cpt]=M+ind_jj[j]
                col_loc[cpt]=ii_tapered[i]
                cpt+=1	 	    		     
        for j in range(Njj):   
            for j2 in range(j+1):
                Ktmp_loc[cpt]=loc_kernel(H_bar[:,ind_jj[j2]]*np.sqrt(tapering[j2]),H_bar[:,ind_jj[j]]*np.sqrt(tapering[j]),par_kernel)
                row_loc[cpt]=M+ind_jj[j]
                col_loc[cpt]=M+ind_jj[j2]
                cpt+=1
    else:
        Ktmp_loc=[-1]
        row_loc=[-1]
        col_loc=[-1] 	 
    return row_loc,col_loc,Ktmp_loc

def main_assemble(state_taperer,obs_taperer,kernel,params_kernel):	 
    pool = mp.Pool(processes=7)
    results_assembling = [pool.apply_async(assembling_Kloc, [ii,state_taperer,obs_taperer,kernel,params_kernel])  for _ , ii in enumerate(state_batches)]
    pool.close()
    pool.join()
         
    len_tot=0
    for index, val in enumerate(results_assembling):
        tmp1,_,_=val.get()
        if(tmp1[0]!=-1):
            len_tot+=len(tmp1)
	 
    col=np.zeros((len_tot,),dtype=int)
    row=np.zeros((len_tot,),dtype=int)
    Ktmp=np.zeros((len_tot,))
    #M_ones = np.ones((len_tot,))
         
    beg=0
    for index, val in enumerate(results_assembling):
        tmp1,tmp2,tmp3=val.get()  
        if(tmp1[0]!=-1):
            len_loc=len(tmp1)
            row[beg:beg+len_loc]=tmp1[:]
            col[beg:beg+len_loc]=tmp2[:]
            Ktmp[beg:beg+len_loc]=tmp3[:]
            beg+=len_loc     
    
    unique_indices, counts = np.unique((row, col), axis=1, return_counts=True)
    cooc = { (r, c): count for (r, c), count in zip(unique_indices.T, counts) }
    
    # diagonal_indices = row[row == col]
    # occ_indices, counts = np.unique(diagonal_indices, return_counts=True)
    # occ_counts = {index: count for index, count in zip(occ_indices, counts)}

    rows_occ = []
    cols_occ = []
    values_sim = []

    for (i, j), val in cooc.items():
        if i != j:  # Calcul seulement pour les indices hors diagonale
            diag_i = cooc[(i, i)]
            diag_j = cooc[(j, j)]
            new_val = 2 * val / (diag_i + diag_j)
            rows_occ.append(i)
            cols_occ.append(j)
            values_sim.append(new_val)
    else:  # Si on tombe sur un élément diagonal, le garder tel quel
        rows_occ.append(i)
        cols_occ.append(i)
        values_sim.append(1)
    
    return row,col,Ktmp,rows_occ, cols_occ,values_sim

def local_analysis_mean(ii,obs_taperer,kernel, params_kernel):
    if len(ii) != 0:
        jj=[]
        jj, tapering = obs_taperer(ii)
        jjj.append(jj)
            
        H_bar_jj = H_bar[:,jj]
        d_bar_jj  = d_bar[jj]
	    
	 # Taper
        H_bar_jj  *= np.sqrt(tapering)
        d_bar_jj *= np.sqrt(tapering)    
	    
        Nii=len(ii)
        Njj=len(jj)
        K = np.zeros((Njj,Nii+Njj))       
        for ip in range(Njj):
            for jp in range(Nii):
                K[ip,jp] = kernel(H_bar_jj[:,ip],A[:,ii[jp]],params_kernel) 
            for jp in range(Njj):
                K[ip,Nii+jp] = kernel(H_bar_jj[:,ip],H_bar_jj[:,jp],params_kernel) 
        
	    #calcul xa
        alpha_H = np.linalg.solve((N1*np.eye(Njj)+K[:,Nii:]),d_bar_jj)
        wa = K[:,:Nii].T@alpha_H
        xa = 0*mu[ii] + wa 
    else :
        xa=0*mu[ii]
        
    return ii, xa

def main_la_mean(obs_taperer,kernel, params_kernel):
    pool = mp.Pool(processes=6)
    results_la_mean = [pool.apply_async(local_analysis_mean, [ii,obs_taperer,kernel,params_kernel])  for _ , ii in enumerate(state_batches)]
    pool.close()
    pool.join()
    
    #Ms=len(mu)
    xa=mu[:]
    for index, val in enumerate(results_la_mean):
        tmp1,tmp2=val.get()
        xa[tmp1[:]]+=tmp2[:]
       # xa[index]=tmp[:]
	
    return xa


if __name__ == '__main__':
    #mp.set_start_method('fork')
    xa=main_la_mean(my_obs_taperer,kernel, params_kernel)
    print('Mean computed')
    
    row,col,Ktmp,rows_occ, cols_occ, values_sim = main_assemble(my_state_taperer,my_obs_taperer,kernel,params_kernel)

sig_occ=2
#K_ajust = np.array([Ktmp[i]*np.exp(-(2*cocc[(row[i], col[i])]/(occ[row[i]]+occ[col[i]]))**2/(2*sig_occ**2)) for i in range(len(Ktmp))])
#K_ajust = np.array([Ktmp[i]*np.exp(-(1-2*cocc[(row[i], col[i])]/(occ[row[i]]+occ[col[i]]))/(2*sig_occ)) for i in range(len(Ktmp))])
#K_ajust = np.array([Ktmp[i]*(1+(1-2*cocc[(row[i], col[i])]/(occ[row[i]]+occ[col[i]]))**2) for i in range(len(Ktmp))])
K_cloc=sps.csc_matrix((np.array(Ktmp),(np.array(row),np.array(col))),shape=(M+n_obs,M+n_obs))
M_occ = sps.csc_matrix((np.array(values_sim),(np.array(rows_occ),np.array(cols_occ))),shape=(M+n_obs,M+n_obs))
cooc = M_occ.data
weights = 1-cooc
M_ajust = sps.csc_matrix((weights,M_occ.indices,M_occ.indptr),shape=(M+n_obs,M+n_obs))
M_diag = M_ajust.diagonal()
M_ajust = (M_ajust + M_ajust.T)
I_sparse = sps.identity(M+n_obs,format="csc")
M_adjust = M_ajust + I_sparse

''' M_spd = cp.Variable((M+n_obs, M+n_obs), symmetric=True)

# Contrainte SPD (A' doit être semi-définie positive)
constraints = [M_spd >> 0]

# Fonction objectif (minimiser la distance entre A et A', norme frobenius)
objective = cp.Minimize(cp.norm(M_adjust - M_spd,"fro"))

prob = cp.Problem(objective, constraints)

prob.solve() '''

Sigma_M,U_M = spsla.eigsh(M_ajust)
#print(Sigma_M)
Sigma_M[Sigma_M<0] = 0
M_spd = U_M@np.diag(Sigma_M)@U_M.T

K_cloc = K_cloc.multiply(M_spd)


Kdiag=K_cloc.diagonal()  
K_cloc=K_cloc+K_cloc.T-sps.dia_matrix((Kdiag,0),shape=(M+n_obs,M+n_obs)).tocsc()
print('sparsity rate K_cloc:')
print(100*K_cloc.size/(M+n_obs)**2) 
    
K_cloc_H=K_cloc[M:,M:].toarray()
#U_H,Sigma_H,_=np.linalg.svd(K_cloc_H,full_matrices='False',hermitian='True')
Sigma_H,U_H=np.linalg.eigh(K_cloc_H)
ind_sort=np.flip(np.argsort(Sigma_H))
Sigma_H=Sigma_H[ind_sort]
U_H=U_H[:,ind_sort]
#print(Sigma_H)
    
rg_Pa_X=0
for ll in range(n_obs):
    if(Sigma_H[ll]>np.finfo(float).eps*Sigma_H[0]):
         rg_Pa_X=rg_Pa_X+1
    else:
         break
U_H=U_H[:,:rg_Pa_X]

for ll in range(rg_Pa_X):
    U_H[:,ll]=U_H[:,ll]/np.sqrt(Sigma_H[ll]+N1)
    
Ktmp=K_cloc[:M,M:]@U_H
Pa_X=(K_cloc[:M,:M]-Ktmp@Ktmp.T)/N1   
print('Pa_X computed')
print(rg_Pa_X)

#svd de Pa_X
#nconverged=rg_Pa_X+10	   
#U_H,Sigma_H,loc_converged,_=Covariance_subspace_iteration(Pa_X,nconverged,2,1e-4,0.99,20)
#ind_sort=np.flip(np.argsort(Sigma_H))
#Sigma_H=Sigma_H[ind_sort]
#print(loc_converged)
#U_H=U_H[:,ind_sort]


#U_H,Sigma_H,_=spsla.svds(Pa_X,k=ceil(0.9*rg_Pa_X), tol=1e-6,which='LM',maxiter=40,return_singular_vectors='u')
Sigma_H,U_H=spsla.eigsh(Pa_X,k=ceil(0.9*rg_Pa_X), tol=1e-6,which='LM',maxiter=40,return_eigenvectors=True)
ind_sort=np.flip(np.argsort(Sigma_H))
Sigma_H=Sigma_H[ind_sort]
U_H=U_H[:,ind_sort]
loc_converged=len(Sigma_H)
#print(Sigma_H)

rg_Pa_X=0
trace_trunc=0.0
for ll in range(loc_converged):
    if(Sigma_H[ll]>np.finfo(float).eps*Sigma_H[0]):
         rg_Pa_X=rg_Pa_X+1
         trace_trunc+=Sigma_H[ll]
    else:
         break

trace_Pa=Pa_X.trace()
print(trace_Pa,trace_trunc)
  
U_H=U_H[:,:rg_Pa_X]
Sigma_H=Sigma_H[:rg_Pa_X]#/N1

if(advanced_sampling==0):
   Om=np.transpose(U_H)@genOG_MN(M,N)
   for i in range(rg_Pa_X):
       U_H[:,i]=U_H[:,i]*np.sqrt(Sigma_H[i])
   U_H=U_H@Om  
   U_H=U_H@(np.eye(N)-np.ones((N,N))/N)	
else:
   ratio_samp=2
   Nsamp=ratio_samp*rg_Pa_X
   xi=np.zeros((M,Nsamp))
   for i in range(Nsamp):
       xi[:,i]=np.random.standard_normal(size=M)
       
   xi=np.transpose(U_H)@(xi@(np.eye(Nsamp)-np.ones((Nsamp,Nsamp))/Nsamp))
   for i in range(rg_Pa_X):
       xi[:,i]=xi[:,i]*np.sqrt(Sigma_H[i])
   U_H=U_H@xi

   #SVD de U_H et selection
   U_Samp,Sigma_Samp,_=scila.svd(U_H,full_matrices=False)
       
   Om=genOG(Nsamp)
   Om=np.transpose(Om[:N,:rg_Pa_X])
      
   for i in range(rg_Pa_X):
       U_Samp[:,i]=U_Samp[:,i]*np.sqrt(1/ratio_samp)*Sigma_Samp[i]

   U_H=U_Samp[:,:rg_Pa_X]@(Om@(np.eye(N)-np.ones((N,N))/N))

for ll in range(N):
    #E[ll,:] = xa+np.sqrt(N1)*np.reshape(U_H[:,ll],(M,))#*np.sqrt(N/rg_Pa_X)
    E[ll,:] = xa+np.reshape(U_H[:,ll],(M,))
E = post_process(E,infl,rot=False)
    
#analysis stat
mean_ap_assim = np.mean(E,0)
rmse_ap_assim= np.linalg.norm(mean_ap_assim-truth)/len(mean_ap_assim)
spread_ap_assim = np.std(E, axis=0)
spread_ap_assim = np.linalg.norm(spread_ap_assim)/(i_dom*j_dom)
print("analysis rmse", rmse_ap_assim)
print("analysis std", spread_ap_assim)

for k in range(2,N+2):
    file_member_k = "/home/esimon/MdC/Codes/QG_Cluster/QG_standalone/member_"+str(k)+"/run_t_"+str(tend)+".nc"
    
    with Dataset(file_member_k, 'a') as ncfile_member_k:

        psi_member_k = ncfile_member_k.variables['psi']
        
        member_k = np.reshape(E[k-2,:], (j_dom,i_dom), 'F')
       

        psi_member_k[-1, ...] = member_k
        #psi_member_k[0, ...] = member_k

with open("rmse_kernel_cluster.csv", "a", newline='') as csv_file:
    spamwriter = csv.writer(csv_file)
    spamwriter.writerow([n , rmse_av_assim, rmse_ap_assim, spread_av_assim, spread_ap_assim,])

