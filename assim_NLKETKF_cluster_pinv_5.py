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
from tools.matrices import genOG_1, genOG, genOG_MN

from tools.randvars import GaussRV
from tools.localisation import nd_Id_cluster_loc_state, nd_Id_localization_cluster_obs, nd_Id_localization
from tools.linalg import mldiv, pad0, svd0
from tools.post_process import post_process
from tools.pre_process import add_noise
from tools.subspace_iteration import Covariance_subspace_iteration
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from skimage import measure
from matplotlib.path import Path
from math import ceil
from matspy import spy
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, MaxAbsScaler

from scipy import stats as scistats
from scipy import linalg as scila
import ray
import matplotlib.pyplot as plt 


np.set_printoptions(threshold=sys.maxsize)

ray.init()

random.seed(3000)

obs_record = sys.argv[1]
N = int(sys.argv[2])-1
tend = float(sys.argv[3])
dtout = float(sys.argv[4])
n = int(sys.argv[5])
scaler_string = sys.argv[6]


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
    file_member_k = "/home/smauran/Developpement/QG_standalone_counillon_assim4pdt/member_"+str(k)+"/run_t_"+str(tend)+".nc"
    #print(file_member_i)
    with Dataset(file_member_k, 'r') as ncfile_member_k:

        psi_member_k = ncfile_member_k.variables['psi']
        
        member_k = f2py(psi_member_k[-1,:,:])

        
        E[k-2,:] = member_k.data

spread= np.std(E, axis=0)
spread = np.linalg.norm(spread)/(M)
print(spread)

#with open('ens_av_assim.npy', 'rb') as f:
     #E = np.load(f)

#print("E avant assim",E[0,:])


# Kernel definition
def kernel_lin(x,y,params=dict()):
        return np.dot(x,y)

params_kernel_lin = dict()
# Assimilation

def kernel_exp(x,y,params):
    alpha = params["alpha"]
    spread_p = params["spread"]
    eps = alpha*spread_p
    tmp=np.linalg.norm(x-y)
    return np.linalg.norm(x)*np.linalg.norm(y)*np.exp(-(tmp/eps)**2)



def kernel_poly(self, x,y,params):
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
    return kernel_lin(myftanh(x,np.sqrt(nu)),myftanh(y,np.sqrt(nu)),dict())   

params_kernel_tanh=dict(c=10e-4/M)
    
R_obs=5
R_state=5
batch_shape = [2, 2]
infl=1.04
advanced_sampling=1
#infl=1.0

def get_clusters(E,loc,rad,taper,j,i):
    # clustering sur les variables
    N, Nx = E.shape
    #print(i,j) 
    
    
    dist_xx, state_taperer = loc(rad,taper)

    nblimit=10
    vec_bouldin=np.zeros((nblimit,1))

    for k in range(1,nblimit):
        nb=np.int64(k+1)
        kmeans = KMeans(n_clusters=nb, random_state=0, n_init = 'auto').fit(E.T)
        vec_bouldin[k]=davies_bouldin_score(E.T,kmeans.labels_)

    #tol = 5*10**-2

    ''' min_s=[]
    for k in range(2,nblimit):
        if abs(vec_bouldin[k]-vec_bouldin[nblimit-1]) <tol:
            min_s.append(k) '''
    #print(min_s) 

    #nb_min=min_s[0]
    nb_min=np.argmin(vec_bouldin[2:])+3
    #print("nb min", nb_min)

    kmeans = KMeans(n_clusters=nb_min, random_state=0, n_init = 'auto').fit(E.T)

    # 2me passe de clustering avec les contours
    
    labels = np.zeros(Nx)
    nb_clusters = 1

    seuil = 20

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

def NLKETKF_cluster_pinv(E, t, Eo, y, hnoise, loc1,loc2, rad_obs, rad_state, infl,kernel,scalerString):
    # Decompose ensmeble
    #Constantes
    xN = 1.0
    g = 0
    taper='GC'

    

    _map = map
    #noise=0
    #E = add_noise(E, dtout, noise, 'Stoch')
    R, N1 = hnoise.C, N-1
    R_tilde = R.sym_sqrt_inv
    mu = np.mean(E, 0)
 
    #Moderation warm-up EHO
#    if(t<430):
#        R_tilde=R_tilde/np.sqrt(8.0)
#    elif(t<460):
#        R_tilde=R_tilde/2.0
#    elif(t<510):
#        R_tilde=R_tilde/np.sqrt(2.0)
 
    #print("err relative",np.linalg.norm(mu-truth)/np.linalg.norm(truth))
    
    A  = E - mu
    # Obs space variables
    xo = np.mean(Eo, 0)  # Obs ens mean
    Y  = Eo-xo           # Obs ens anomalies (HX_f)
    # Transform obs space
    Y  = Y        #@ R.sym_sqrt_inv.T
    dy = (y - xo) #@ R.sym_sqrt_inv.T

    # For Kernel analysis
    H_bar = (Y@R_tilde)
    d_bar = (R_tilde@dy)

    labels,nb_clusters, nb_clusters_reel, state_taperer = get_clusters(E,loc1,rad_state,taper,j_dom,i_dom)
    
    E_aug = np.concatenate((A, H_bar),axis = 1)
    

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
    
    if scaler != None:
        E_aug = scaler.fit_transform(E_aug)
    
    print("normalisation faite")

    spread_ap_norm = np.std(E_aug[:,:M], axis=0)
    spread_ap_norm = np.linalg.norm(spread_ap_norm)
    print("spread après normalisation", spread_ap_norm)
    
    obs_taperer = loc2(rad_obs, 'x2y', t, taper)
    jjj=[]
    

    params_kernel_exp = dict(alpha=0.1, spread = spread_ap_norm)
    params_kernel = params_kernel_exp

    @ray.remote
    def local_analysis_mean(ii,E_aug,mu,d_bar):
    #def local_analysis_mean(ii):
        """Do the local analysis.

        Notation:

        - ii: inds for the state batch defining the locality
        - jj: inds for the associated obs
        """
        if len(ii) != 0:
            # Locate local obs
            jj=[]
            jj, tapering = obs_taperer(ii)
            jjj.append(jj)
            
            H_bar_jj = E_aug[:,M+jj]
            d_bar_jj  = d_bar[jj]
	    
            # Taper
            H_bar_jj  *= np.sqrt(tapering)
            d_bar_jj *= np.sqrt(tapering)
            
            Nii=len(ii)
            Njj=len(jj)
            E_aug_loc = np.concatenate((E_aug[:,ii], H_bar_jj),axis = 1)
            

            K = np.zeros((Njj,Nii+Njj))       
            for ip in range(Njj):
                for jp in range(Nii):
                     K[ip,jp] = kernel(E_aug_loc[:,Nii+ip],E_aug_loc[:,jp],params_kernel) 
                for jp in range(Njj):
                     K[ip,Nii+jp] = kernel(E_aug_loc[:,Nii+ip],E_aug_loc[:,Nii+ip],params_kernel) 
        
	    #calcul xa
            alpha_H = np.linalg.solve((N1*np.eye(Njj)+K[:,Nii:]),d_bar_jj)
            wa = K[:,:Nii].T@alpha_H
            xa = mu[ii] + wa 
        else :
            xa=mu[ii]
        
        return xa
    
    # recuperation des clusters	
    state_batches = []
    Index_ech=[]
    for c in range(nb_clusters):
        i_cluster = [index for (index,cluster) in enumerate(labels) if cluster == c]
        state_batches.append(i_cluster)
        if len(i_cluster)!=0:
            Index_ech.append(i_cluster[0])
            Index_ech.append(i_cluster[-1])


    # Run local analyses : xa
    def parmap(la,mEaug,mmu,md,list):
        return [la.remote(ii,mEaug,mmu,md) for ii in list]
    
    
    results_ids=parmap(local_analysis_mean,ray.put(E_aug),ray.put(mu),ray.put(d_bar),state_batches)
    results=ray.get(results_ids)
    #results=zip(*_map(local_analysis_mean, state_batches))
   
    print('Mean computed')
    ######## ensemble mean and associated sparse symmetric matrix K_cloc
    n_obs=len(H_bar[0,:]) 
    K_cloc=sps.csc_matrix((M+n_obs,M+n_obs))
    xa=np.zeros((M,))
    for index, ii in enumerate(state_batches):
        #xa[ii]=results[index]
        if len(ii) != 0:
             xa[ii]=results[index]
             ii_tapered, state_tapering = state_taperer(ii)
             ind_jj,tapering=obs_taperer(ii)
             Njj=len(ind_jj)
             Nii=len(ii_tapered)
             #print('ii et ii_tapered')
             #print(len(ii))
             #print(Nii)
             #print(len(np.unique(ii_tapered)))
             Ktmp=[]
             row=[]
             col=[]
             cpt=0
             for i in range(Nii):
                 for j in range(i+1):
                     row.append(ii_tapered[i])
                     col.append(ii_tapered[j])
                     Ktmp.append(kernel(A[:,ii_tapered[i]]*np.sqrt(state_tapering[i]),A[:,ii_tapered[j]]*np.sqrt(state_tapering[j]),params_kernel))
                     cpt+=1
             K_cloc=K_cloc+sps.csc_matrix((np.array(Ktmp),(np.array(row),np.array(col))),shape=(M+n_obs,M+n_obs))
             Ktmp=[]
             row=[]
             col=[]
             cpt=0	    		         
             for j in range(Njj): 
                  for i in range(Nii):
                     Ktmp.append(kernel(A[:,ii_tapered[i]]*np.sqrt(state_tapering[i]),H_bar[:,ind_jj[j]]*np.sqrt(tapering[j]),params_kernel))
                     row.append(M+ind_jj[j])
                     col.append(ii_tapered[i])
                     cpt=+1	 
             K_cloc=K_cloc+sps.csc_matrix((np.array(Ktmp),(np.array(row),np.array(col))),shape=(M+n_obs,M+n_obs))
             Ktmp=[]
             row=[]
             col=[]
             cpt=0	    		     
             for j in range(Njj):
                 for j2 in range(j+1):
                     Ktmp.append(kernel(H_bar[:,ind_jj[j2]]*np.sqrt(tapering[j2]),H_bar[:,ind_jj[j]]*np.sqrt(tapering[j]),params_kernel))
                     row.append(M+ind_jj[j])
                     col.append(M+ind_jj[j2])
                     cpt+=1 
             K_cloc=K_cloc+sps.csc_matrix((np.array(Ktmp),(np.array(row),np.array(col))),shape=(M+n_obs,M+n_obs))  
    Kdiag=K_cloc.diagonal()  
    K_cloc=K_cloc+K_cloc.T-sps.dia_matrix((Kdiag,0),shape=(M+n_obs,M+n_obs)).tocsc()
    print('sparsity rate K_cloc:')
    print(100*K_cloc.size/(M+n_obs)**2) 
    #spy(K_cloc)
    
    K_cloc_H=K_cloc[M:,M:].toarray()
    U_H,Sigma_H,_=np.linalg.svd(K_cloc_H,full_matrices='False',hermitian='True')
   
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
    Pa_X=K_cloc[:M,:M].copy()
    Pa_X=Pa_X-(Ktmp@Ktmp.T)
   # plt.imshow(np.log(np.abs(Pa_X))) 
   # plt.show() 
       
    print('begining svds Pa_X')  
    print(rg_Pa_X) 
    #ncV=rg_Pa_X+20
    nconverged=rg_Pa_X+10
    print("nconverged =", nconverged)	   
    #U_H,Sigma_H,_=spsla.svds(Pa_X,k=nconverged,ncv=ncV,tol=1e-3,return_singular_vectors='u',maxiter=1000) 
    #loc_converged=rg_Pa_X
    U_H,Sigma_H,loc_converged,_=Covariance_subspace_iteration(Pa_X,nconverged,2,1e-4,0.2,500)
    ind_sort=np.flip(np.argsort(Sigma_H))
    Sigma_H=Sigma_H[ind_sort]
    print(loc_converged)
    #print(Sigma_H)
    U_H=U_H[:,ind_sort]
    rg_Pa_X=0
    trace_trunc=0.0
    for ll in range(loc_converged):
        if(Sigma_H[ll]>np.finfo(float).eps*Sigma_H[0]):
             rg_Pa_X=rg_Pa_X+1
             trace_trunc+=Sigma_H[ll]
        else:
             break
    print(rg_Pa_X)
  
    U_H=U_H[:,:rg_Pa_X]
    Sigma_H=Sigma_H[:rg_Pa_X]/N1

    if(advanced_sampling==0):
       Om=np.transpose(U_H)@genOG_MN(M,N)
       for i in range(rg_Pa_X):
           U_H[:,i]=U_H[:,i]*np.sqrt(Sigma_H[i])
       U_H=U_H@Om  
       U_H=U_H@(np.eye(N)-np.ones((N,N))/N)	
    else:
       ratio_samp=2
       Nsamp=ratio_samp*rg_Pa_X
       #tmp = np.random.multivariate_normal(np.zeros(rg_Pa_X),np.eye(rg_Pa_X),Nsamp)
       #tmp = np.random.multivariate_normal(np.zeros(M),np.eye(M),size=Nsamp)
       #xi=tmp.T
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
       #U_H=U_H@(np.eye(N)-np.ones((N,N))/N)	 
    
            	   
   # xi = np.random.multivariate_normal(np.zeros(M),Pa_X,N)
    #frzn = scistats.multivariate_normal(np.zeros(M),Pa_X,allow_singular=True)
    #Aa=frzn.rvs(N) 
    for ll in range(N):
        #E[ll,:] = xa+np.sqrt(N1)*np.reshape(U_H[:,ll],(M,))#*np.sqrt(N/rg_Pa_X)
        E[ll,:] = xa+np.reshape(U_H[:,ll],(M,))
    E = post_process(E,infl,rot=False)
    
   # A=E-np.mean(E, 0)
    
    #print(np.linalg.norm(np.transpose(A)@A-Pa_X)/np.linalg.norm(Pa_X))
    return E, nb_clusters_reel

mean_av_assim = np.mean(E,0)
rmse_av_assim = np.linalg.norm(mean_av_assim-truth)/len(mean_av_assim)
spread_av_assim = np.std(E, axis=0)
spread_av_assim = np.linalg.norm(spread_av_assim)/(i_dom*j_dom)


#E = LETKF(E, tend, E[:,obs_ind], y_obs, GaussRV(C=4*np.eye(Ny)), nd_Id_localization(shape[::-1], batch_shape[::-1], obs_ind, periodic=False), R, infl)
#E, nb_c  = NLKETKF_cluster_pinv(E, tend, E[:,obs_ind], obs, GaussRV(C=4*np.eye(Ny)), nd_Id_cluster_loc(shape[::-1], obs_ind, periodic=False),nd_Id_localization_cluster_obs(shape[::-1], obs_ind, periodic=False), R_loc, infl,kernel,params_kernel,obs_ind)
E, nb_c  = NLKETKF_cluster_pinv(E, tend, E[:,obs_ind], obs, GaussRV(C=4*np.eye(Ny)), nd_Id_cluster_loc_state(shape[::-1], periodic=False),nd_Id_localization_cluster_obs(shape[::-1], obs_ind, periodic=False), R_obs, R_state, infl,kernel_exp, scaler_string)
mean_ap_assim = np.mean(E,0)
rmse_ap_assim= np.linalg.norm(mean_ap_assim-truth)/len(mean_ap_assim)
spread_ap_assim = np.std(E, axis=0)
spread_ap_assim = np.linalg.norm(spread_ap_assim)/(i_dom*j_dom)



for k in range(2,N+2):
    file_member_k = "/home/smauran/Developpement/QG_standalone_counillon_assim4pdt/member_"+str(k)+"/run_t_"+str(tend)+".nc"
    
    with Dataset(file_member_k, 'a') as ncfile_member_k:

        psi_member_k = ncfile_member_k.variables['psi']
        
        member_k = np.reshape(E[k-2,:], (j_dom,i_dom), 'F')
       

        psi_member_k[-1, ...] = member_k
        #psi_member_k[0, ...] = member_k

with open("rmse_kernel_exp_NormalizerL2.csv", "a", newline='') as csv_file:
    spamwriter = csv.writer(csv_file)
    spamwriter.writerow([n , rmse_av_assim, rmse_ap_assim, spread_av_assim, spread_ap_assim,])
#print("E après assim", E)
#print(E.shape)
