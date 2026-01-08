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
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from skimage import measure
from matplotlib.path import Path
from math import ceil
#import ray


from tools.randvars import GaussRV
from tools.localisation import nd_Id_localization_cluster_obs, nd_Id_cluster_loc_state
from tools.linalg import mldiv, pad0, svd0
from tools.post_process import post_process
from tools.pre_process import add_noise

np.set_printoptions(threshold=sys.maxsize)
#ray.init()

random.seed(3000)

obs_record = sys.argv[1]
N = int(sys.argv[2])-1
tend = float(sys.argv[3])
dtout = float(sys.argv[4])
n = int(sys.argv[5])


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
    ''' if n>100 and n%5==0 :
        np.save("analyse_R4_seuil20/obs_ind/obs_ind_"+str(n)+".npy", obs_ind)
     '''
    
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

# Assimilation



def Newton_m(fun, deriv, x0, is_inverted=False,
             conf=1.0, xtol=1e-4, ytol=1e-7, itermax=10**2):
    """Find root of `fun`.

    This is a simple (and pretty fast) implementation of Newton's method.
    """
    itr, dx, Jx = 0, np.inf, fun(x0)
    def norm(x): return np.sqrt(np.sum(x**2))
    while ytol < norm(Jx) and xtol < norm(dx) and itr < itermax:
        Dx  = deriv(x0)
        if is_inverted:
            dx  = Dx @ Jx
        elif isinstance(Dx, float):
            dx  = Jx/Dx
        else:
            dx  = mldiv(Dx, Jx)
        dx *= conf
        x0 -= dx
        Jx  = fun(x0)
    return x0

def hyperprior_coeffs(s, N, xN=1, g=0):
    r"""Set EnKF-N inflation hyperparams.

    The EnKF-N prior may be specified by the constants:

    - eN: Effect of unknown mean
    - cL: Coeff in front of log term

    These are trivial constants in the original EnKF-N,
    but are further adjusted (corrected and tuned) for the following reasons.

    - Reason 1: mode correction.
      These parameters bridge the Jeffreys (`xN=1`) and Dirac (`xN=Inf`) hyperpriors
      for the prior covariance, B, as discussed in `bib.bocquet2015expanding`.
      Indeed, mode correction becomes necessary when $$ R \rightarrow \infty $$
      because then there should be no ensemble update (and also no inflation!).
      More specifically, the mode of `l1`'s should be adjusted towards 1
      as a function of $$ I - K H $$ ("prior's weight").
      PS: why do we leave the prior mode below 1 at all?
      Because it sets up "tension" (negative feedback) in the inflation cycle:
      the prior pulls downwards, while the likelihood tends to pull upwards.

    - Reason 2: Boosting the inflation prior's certainty from N to xN*N.
      The aim is to take advantage of the fact that the ensemble may not
      have quite as much sampling error as a fully stochastic sample,
      as illustrated in section 2.1 of `bib.raanes2019adaptive`.

    - Its damping effect is similar to work done by J. Anderson.

    The tuning is controlled by:

    - `xN=1`: is fully agnostic, i.e. assumes the ensemble is generated
      from a highly chaotic or stochastic model.
    - `xN>1`: increases the certainty of the hyper-prior,
      which is appropriate for more linear and deterministic systems.
    - `xN<1`: yields a more (than 'fully') agnostic hyper-prior,
      as if N were smaller than it truly is.
    - `xN<=0` is not meaningful.
    """
    N1 = N-1

    eN = (N+1)/N
    cL = (N+g)/N1

    # Mode correction (almost) as in eqn 36 of `bib.bocquet2015expanding`
    prior_mode = eN/cL                        # Mode of l1 (before correction)
    diagonal   = pad0(s**2, N) + N1           # diag of Y@R.inv@Y + N1*I
    #                                           (Hessian of J)
    I_KH       = np.mean(diagonal**(-1))*N1   # ≈ 1/(1 + HBH/R)
    # I_KH      = 1/(1 + (s**2).sum()/N1)     # Scalar alternative: use tr(HBH/R).
    mc         = np.sqrt(prior_mode**I_KH)       # Correction coeff

    # Apply correction
    eN /= mc
    cL *= mc

    # Boost by xN
    eN *= xN
    cL *= xN

    return eN, cL

def effective_N(YR, dyR, xN, g):
    """Effective ensemble size N.

    As measured by the finite-size EnKF-N
    """
    N, Ny = YR.shape
    N1   = N-1

    V, s, UT = svd0(YR)
    du     = UT @ dyR

    eN, cL = hyperprior_coeffs(s, N, xN, g)

    def pad_rk(arr): return pad0(arr, min(N, Ny))
    def dgn_rk(l1): return pad_rk((l1*s)**2) + N1

    # Make dual cost function (in terms of l1)
    def J(l1):
        val = np.sum(du**2/dgn_rk(l1)) \
            + eN/l1**2 \
            + cL*np.log(l1**2)
        return val

    # Derivatives (not required with minimize_scalar):
    def Jp(l1):
        val = -2*l1   * np.sum(pad_rk(s**2) * du**2/dgn_rk(l1)**2) \
            + -2*eN/l1**3 \
            + 2*cL/l1
        return val

    def Jpp(l1):
        val = 8*l1**2 * np.sum(pad_rk(s**4) * du**2/dgn_rk(l1)**3) \
            + 6*eN/l1**4 \
            + -2*cL/l1**2
        return val

    # Find inflation factor (optimize)
    l1 = Newton_m(Jp, Jpp, 1.0)
    # l1 = fmin_bfgs(J, x0=[1], gtol=1e-4, disp=0)
    # l1 = minimize_scalar(J, bracket=(sqrt(prior_mode), 1e2), tol=1e-4).x

    za = N1/l1**2
    return za

#t=tend
#Eo = E[:,obs_inds]
#hnoise=GaussRV(C=4*np.eye(Ny))
#y=obs
#loc = nd_Id_localization(shape[::-1], batch_shape[::-1], obs_inds, periodic=False)

R = 5
seuil =10
batch_shape = [2, 2]
infl=1.02
#infl=1.0
save_nbs_min = "nb_min_history.txt"

def get_clusters(E,loc,rad,taper,j,i):
    # clustering sur les variables
    N, Nx = E.shape
    #print(N,Nx) 
    
    
    dist_xx, state_taperer = loc(rad,taper)

    if n <= 150: # 150 cycles de rôdage
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
                min_s.append(k) '''
        #print(min_s) 

        #nb_min=min_s[0]
        nb_min=np.argmin(vec_bouldin[2:])+3
        with open(save_nbs_min, "a") as f:
            f.write(f"{nb_min}\n")
    #print("nb min", nb_min)
    else :
        with open(save_nbs_min, "r") as f:
            nbs_min = [float(line.strip()) for line in f.readlines()] #commande pour ne lire que la denière ligne du fichier, on pourrait rajouter la moyenne comme ça ?
        nb_min = int(np.floor(sum(nbs_min) / len(nbs_min)))



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
    return(labels, nb_clusters, len(unique_labels),state_taperer,nb_min)


def LETKF(E, t, Eo, y, hnoise, loc1, loc2, rad, infl):
    # Decompose ensemble
    #Constantes
    xN = 1.0
    g = 0
    taper='GC'

    _map = map
    #noise=0
    #E = add_noise(E, dtout, noise, 'Stoch')
    R, N1 = hnoise.C, N-1
    mu = np.mean(E, 0)
    
    #print("err relative",np.linalg.norm(mu-truth)/np.linalg.norm(truth))
    
    A  = E - mu
    # Obs space variables
    xo = np.mean(Eo, 0)  # Obs ens mean
    Y  = Eo-xo           # Obs ens anomalies (HX_f)
    # Transform obs space
    Y  = Y        @ R.sym_sqrt_inv.T
    dy = (y - xo) @ R.sym_sqrt_inv.T
    

    # Local analyses
    # Get localization configuration
    
    labels,nb_clusters, nb_clusters_reel, state_taperer,nb_min = get_clusters(E,loc1,rad,taper,j_dom,i_dom)

    obs_taperer = loc2(rad, 'x2y', t, taper)
    jjj=[]

    #@ray.remote
    def local_analysis(ii):
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
            if len(jj) == 0:
                return E[:, ii], N1  # no update
            Y_jj   = Y[:, jj]
            dy_jj  = dy[jj]

            # Adaptive inflation
            za = effective_N(Y_jj, dy_jj, xN, g) if infl == '-N' else N1
            # Taper
            Y_jj  *= np.sqrt(tapering)
            dy_jj *= np.sqrt(tapering)

            # Compute ETKF update
            if len(jj) < N:
                # SVD version
                V, sd, _ = svd0(Y_jj)
                d      = pad0(sd**2, N) + za
                Pw     = (V * d**(-1.0)) @ V.T
                T      = (V * d**(-0.5)) @ V.T * np.sqrt(za)
            else:
                # EVD version
                d, V   = sla.eigh(Y_jj@Y_jj.T + za*np.eye(N))
                T     = V@np.diag(d**(-0.5))@V.T * np.sqrt(za)
                Pw    = V@np.diag(d**(-1.0))@V.T
            AT  = T @ A[:, ii]
            dmu = dy_jj @ Y_jj.T @ Pw @ A[:, ii]
            Eii = mu[ii] + dmu + AT
            return Eii, za
        else :
            return E[:, ii], N1  # no update


    state_batches = []
    for c in range(nb_clusters):
        i_cluster = [index for (index,cluster) in enumerate(labels) if cluster == c]
        state_batches.append(i_cluster)
    ''' nb_obs_batches=[]
    nb_state_batches=[]
    for c in state_batches:
        if len(c) != 0:
            #obs_c,_= obs_taperer(c)
            nb_obs_batches.append(len(jj) for jj in jjj)
            nb_state_batches.append(len(c))
 '''
    #print(state_batches)
    # Run local analyses
    # def parmap(la,list,mY,mE, mA):
    #     return [la.remote(ii,mY, mE, mA) for ii in list]
    
    # results_ids=parmap(local_analysis, state_batches, ray.put(Y), ray.put(E), ray.put(A))
    # results=ray.get(results_ids)

    
    # for index,ii in enumerate(state_batches):  
    #     #print(results[cpt][0])
    #     E[:, ii] = results[index][0]
    EE, za = zip(*_map(local_analysis, state_batches))
    for ii, Eii in zip(state_batches, EE):
        E[:, ii] = Eii
    #print("E après assim",E[0,:])
    E = post_process(E,infl,rot=True)
    return E, nb_clusters_reel,nb_min


mean_av_assim = np.mean(E,0)
rmse_av_assim = np.linalg.norm(mean_av_assim-truth)/len(mean_av_assim)
spread_av_assim = np.std(E, axis=0)
spread_av_assim = np.linalg.norm(spread_av_assim)/(i_dom*j_dom)
A_pre = E-mean_av_assim
Pf = 1/(N-1)*A_pre.T@A_pre

if n>=150 and n%5==0 :
    np.save("analyse_eco_R5_seuil10_infl102/E_av/E_av_"+str(n)+".npy", E)
    #np.save("analyse_R4_seuil20/Pf/Pf_"+str(n)+".npy", Pf)


#E = LETKF(E, tend, E[:,obs_ind], y_obs, GaussRV(C=4*np.eye(Ny)), nd_Id_localization(shape[::-1], batch_shape[::-1], obs_ind, periodic=False), R, infl)
E, nb_c, nb_min = LETKF(E, tend, E[:,obs_ind], obs, GaussRV(C=4*np.eye(Ny)), nd_Id_cluster_loc_state(shape[::-1], periodic=False),nd_Id_localization_cluster_obs(shape[::-1], obs_ind, periodic=False), R, infl)



#np.save("Pa_30_LETKF.npy",Pa)

mean_ap_assim = np.mean(E,0)
rmse_ap_assim = np.linalg.norm(mean_ap_assim-truth)/len(mean_ap_assim)
spread_ap_assim = np.std(E, axis=0)
spread_ap_assim = np.linalg.norm(spread_ap_assim)/(i_dom*j_dom)
A_post = E-mean_ap_assim
Pa = 1/(N-1)*A_post.T@A_post

if n>=150 and n%5==0 :
    np.save("analyse_eco_R5_seuil10_infl102/E_ap/E_ap_"+str(n)+".npy", E)
    np.save("analyse_eco_R5_seuil10_infl102/truth/truth_"+str(n)+".npy", truth)
    #np.save("analyse_R4_seuil20/Pa/Pa_"+str(n)+".npy", Pa)

def SRF_global(Pf,Pa,H,R_inv):
    SF = H@Pf@H.T@R_inv
    SA = H@Pa@H.T@R_inv
    return(np.sqrt(np.trace(SF)/np.trace(SA))-1)



for k in range(2,N+2):
    file_member_k = "/home/smauran/Developpement/QG_standalone_counillon_assim4pdt/member_"+str(k)+"/run_t_"+str(tend)+".nc"
    
    with Dataset(file_member_k, 'a') as ncfile_member_k:

        psi_member_k = ncfile_member_k.variables['psi']
        
        member_k = np.reshape(E[k-2,:], (j_dom,i_dom), 'F')

        psi_member_k[-1, ...] = member_k

fichier_csv = "resultats_rmse/rmse_cluster_eco_min_R"+str(R)+"_nblimit5_seuil"+str(seuil)+"_infl_"+str(infl)+".csv"
#fichier_csv = "analyse_davies_bouldin/davies_bouldin_R"+str(R)+"_nblimit30_seuil"+str(seuil)+".csv"
with open(fichier_csv, "a", newline='') as csv_file:
    spamwriter = csv.writer(csv_file)
    #spamwriter.writerow([n , nb_min])
    spamwriter.writerow([n , rmse_av_assim, rmse_ap_assim, spread_av_assim, spread_ap_assim ])
#print("E après assim", E)
#print(E.shape)
