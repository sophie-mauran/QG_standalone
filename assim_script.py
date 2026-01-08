import sys
from netCDF4 import Dataset
import numpy as np
import os
import concurrent.futures

from tools.randvars import GaussRV

obs_record = sys.argv[1]
N = int(sys.argv[2])
tend = int(sys.argv[3])
dtout = int(sys.argv[4])

order='F'
def f2py(X): return X.flatten(order=order)

def equi_spaced_integers(m,p):
    """Provide a range of p equispaced integers between 0 and m-1"""
    return np.round(np.linspace(np.floor(m/p/2),np.ceil(m-m/p/2-1),p)).astype(int)


# Get cycle's observation
with Dataset(obs_record, 'r') as ncfile_obs:
    # Retrieve the last step of the model
    q_obs = ncfile_obs.variables['q']
    #print("q",q_obs)
    j = q_obs.shape[1]
    #print("j",j)
    i = q_obs.shape[2]
    #print("i",i)
    
    M = i*j
    shape=(j,i)
    # This will look like satellite tracks when plotted in 2D
    Ny = 300
    jj = equi_spaced_integers(M, Ny)

    # Want: random_offset(t1)==random_offset(t2) if t1==t2.
    # Solutions: (1) use caching (ensure maxsize=inf) or (2) stream seeding.
    # Either way, use a local random stream to avoid interfering with global stream
    # (and e.g. ensure equal outcomes for 1st and 2nd run of the python session).
    rstream = np.random.RandomState()
    max_offset = jj[1]-jj[0]


    def random_offset(tend,dtout): # Mettre le record à la place de t
        rstream.seed(int(tend/dtout*100)) 
        u = rstream.rand()
        return int(np.floor(max_offset * u))


    def obs_inds(tend,dtout):
        return jj + random_offset(tend,dtout)


    obs_ind = obs_inds(tend,dtout)
    #print("obs_inds",obs_ind)
    
    
    
    obs = f2py(q_obs[0, ...])
    obs = obs[obs_ind]
    #print("obs", obs)
    print(obs.shape)
    

    

# Récupérer l'ensemble


E=np.empty([N,M])

for i in range(1,N+1):
    file_member_i = "/home/smauran/Developpement/QG_standalone_counillon_assim4pdt/member_"+str(i)+"/run_t_"+str(tend)+".nc"
    with Dataset(file_member_i, 'r') as ncfile_member_i:

        q_member_i = ncfile_member_i.variables['q']
        member_i = f2py(q_member_i[-1, ...])
        E[i-1,:] = member_i

print(E.shape)
# Assimilation


def pairwise_distances(A, B=None, domain=None):
    """Euclidian distance (not squared) between pts. in `A` and `B`.

    Parameters
    ----------
    A: array of shape `(nPoints, nDims)`.
        A collection of points.

    B:
        Same as `A`, but `nPoints` can differ.

    domain: tuple
        Assume the domain is a **periodic** hyper-rectangle whose
        edges along dimension `i` span from 0 to `domain[i]`.
        NB: Behaviour not defined if `any(A.max(0) > domain)`, and likewise for `B`.

    Returns
    -------
    Array of of shape `(nPointsA, nPointsB)`.

    Examples
    --------
    >>> A = [[0, 0], [0, 1], [1, 0], [1, 1]]
    >>> with np.printoptions(precision=2):
    ...     print(pairwise_distances(A))
    [[0.   1.   1.   1.41]
     [1.   0.   1.41 1.  ]
     [1.   1.41 0.   1.  ]
     [1.41 1.   1.   0.  ]]

    The function matches `pdist(..., metric='euclidean')`, but is faster:
    >>> from scipy.spatial.distance import pdist, squareform
    >>> (pairwise_distances(A) == squareform(pdist(A))).all()
    True

    As opposed to `pdist`, it also allows comparing `A` to a different set of points,
    `B`, without the augmentation/block tricks needed for pdist.

    >>> A = np.arange(4)[:, None]
    >>> pairwise_distances(A, [[2]]).T
    array([[2., 1., 0., 1.]])

    Illustration of periodicity:
    >>> pairwise_distances(A, domain=(4, ))
    array([[0., 1., 2., 1.],
           [1., 0., 1., 2.],
           [2., 1., 0., 1.],
           [1., 2., 1., 0.]])

    NB: If an input array is 1-dim, it is seen as a single point.
    >>> pairwise_distances(np.arange(4))
    array([[0.]])
    """
    if B is None:
        B = A

    # Prep
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    mA, nA = A.shape
    mB, nB = B.shape
    assert nA == nB, "The last axis of A and B must have equal length."

    # Diff
    d = A[:, None] - B  # shape: (mA, mB, nDims)

    # Make periodic
    if domain:
        domain = np.reshape(domain, (1, 1, -1))  # for broadcasting
        d = abs(d)
        d = np.minimum(d, domain-d)

    distances = np.sqrt((d * d).sum(axis=-1))  # == sla.norm(d, axis=-1)

    return distances.reshape(mA, mB)

# NB: Don't try to put the time-dependence of obs_inds inside obs_taperer().
# That would require calling ind2sub len(batches) times per analysis,
# and the result cannot be easily cached, because of multiprocessing.
def safe_eval(fun, t):
    try:
        return fun(t)
    except TypeError:
        return fun



def nd_Id_kernel_loc(shape, obs_inds = None, periodic = True):
    M = np.prod(shape)

    if obs_inds is None:
        obs_inds = np.arange(M)

    def ind2sub(ind):
        return np.asarray(np.unravel_index(ind, shape)).T
    
    state_coord = ind2sub(np.arange(M))
    dist_xx = pairwise_distances(state_coord,domain = shape if periodic else None)

    def y2x_distances(t):
        obs_coord = ind2sub(safe_eval(obs_inds, t))
        return pairwise_distances(obs_coord, state_coord, shape if periodic else None)

    def y2y_distances(t):
        obs_coord = ind2sub(safe_eval(obs_inds, t))
        return pairwise_distances(obs_coord, domain = shape if periodic else None)
    
    return (dist_xx, y2x_distances, y2y_distances)

def gaussian_kernel_d(d,sigma):
    return np.exp(-d**2/(sigma**2))


def kernel(x,y,params):
        return y@x.T

#t=tend
#Eo = E[obs_inds]
#hnoise=modelling.GaussRV(C=4*np.eye(Ny))
#kernel_loc = nd_Id_kernel_loc(shape[::-1],jj,periodic=False)
#y=obs

R=17





def LETKF_kern_analysis(E, t, Eo, hnoise, kernel_loc, y, kernel, rad, params_kernel):
    R     = hnoise.C     # Obs noise cov
    N, Nx = E.shape      # Dimensionality
    N1    = N-1          # Ens size - 1

    mu = np.mean(E, 0)   # Ens mean
    A  = E - mu          # Ens anomalies (X_f)

    xo = np.mean(Eo, 0)  # Obs ens mean
    Y  = Eo-xo           # Obs ens anomalies (HX_f)
    Np = Y.shape[1]
    dy = y - xo          # Mean "innovation" (y-Hx_bar)

    R_full = R.full
    R_tilde = hnoise.C.sym_sqrt_inv
    #R_demi = sla.sqrtm(R_full)
    
    H_bar = Y@R_tilde
    
    E_aug = np.concatenate((A, H_bar),axis = 1)
    

    K = np.zeros((Nx+Np,Nx+Np))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range(Nx + Np):
            print(i)
            for j in range(Nx + Np):
                K[i, j] = executor.submit(kernel, E_aug[:, i], E_aug[:, j], params_kernel).result()
    
    
    #print(K)
    rg_K = np.linalg.matrix_rank(K)
    #print(np.shape(K))

    # # Localisation par noyaux
    # # Utilisation d'un noyau gaussien pour négliger les valeurs associées aux indices 
    # # trop éloignés les uns des autres. Valeur de seuil : rad

    dist_xx, y2x_distances, y2y_distances = kernel_loc
    dist_xy = y2x_distances(t)
    dist_yy = y2y_distances(t)
    #print(dist_xy)
    
    
    L_s = np.zeros((Nx,Nx))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range (Nx):
            for j in range (Nx):
                L_s[i,j] =  executor.submit(gaussian_kernel_d,dist_xx[i,j],rad).result()

    L_o_rect = np.zeros((Nx,Np))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range (Nx):
            for j in range (Np):
                L_o_rect[i,j] =  executor.submit(gaussian_kernel_d,dist_xy[j,i],rad).result()
    
    L_o = np.zeros((Np,Np))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range (Np):
            for j in range (Np):
                L_o[i,j] =  executor.submit(gaussian_kernel_d,dist_yy[i,j],rad).result()
    
    L = np.zeros((Nx+Np,Nx+Np))

    L[:Nx,:Nx] = L_s
    #L[:Nx,:Nx] = np.eye(Nx)
    L[:Nx,Nx:] = L_o_rect
    #L[:Nx,Nx:] = np.eye(Nx,Np)
    L[Nx:,:Nx] = L_o_rect.T
    #L[Nx:,:Nx] = np.eye(Np,Nx)
    L[Nx:,Nx:] = L_o
    #L[Nx:,Nx:] = np.eye(Np)

    
    K = L*K # produit terme à terme avec le noyau gaussien
    # print(np.cond(K))
    #print(rg_K)
    
        #calcul xa
    d_bar = R_tilde@dy
    alpha_H = np.linalg.solve((N1*np.eye(Np)+K[Nx:,Nx:]),d_bar)
    wa = K[Nx:,:Nx].T@alpha_H
    xa = mu + wa
    
    rg_K = np.linalg.matrix_rank(K)
    
       
    #Calcul de Pa
    if rg_K == Nx+Np:
        #print('rang plein')
       
        [V_H,Sigma_H,U_H]=np.linalg.svd(K[Nx:,Nx:],full_matrices = False)
        #En vrai, Pa_X pas Pa
        Pa = K[:Nx,:Nx] - K[:Nx,Nx:]@V_H@np.diag(1/((N1+Sigma_H)))@U_H@K[:Nx,Nx:].T
		#factorisation racine carree de Pa_X
        [U_Pa,Sigma_Pa,V_Pa]=np.linalg.svd(Pa[:Nx,:Nx], full_matrices = False)
    else :
        #Matrice de projection sur l'espace des obs
        Pi_H = np.zeros((Nx+Np,Nx+Np))
        Pi_H[Nx:,Nx:] = np.eye(Np)
	
        # SVD de K
        hess_alpha = N1*K + (K@Pi_H)@Pi_H@K 
        [U,Sigma,V]=np.linalg.svd(hess_alpha)
        
        rg_Hess=0
        for ll in range(Nx+Np):#range(min(N,Nx+Np)):
            if(Sigma[ll]>np.finfo(float).eps*Sigma[0]):
                rg_Hess=rg_Hess+1
            else:
                break	
        
        #Valeurs propres de Pa
        Sigma_Pa=1.0/Sigma[:rg_Hess]
  
        #Vecteurs propres de Pa
        U1 = U[:,:rg_Hess]
        Pi_X = np.zeros((Nx,Nx+Np))
        Pi_X[:,:Nx] = np.eye(Nx)
        U_Pa=Pi_X@K@U1
        
        
    
        
    #rang effectif
    rg_Pa=0
    for ll in range(rg_Hess): #range(min(N,Nx)):
        if(Sigma_Pa[ll]>np.finfo(float).eps*Sigma_Pa[0]):
            rg_Pa=rg_Pa+1
        else:
            break       
    if(rg_Pa==0):
        rg_Pa=1
     
    Sigma_Pa_demi=np.eye(rg_Pa)
    for ll in range(rg_Pa):
        Sigma_Pa_demi[ll,ll]=np.sqrt(Sigma_Pa[ll])
    proj_X_Pa_demi = U_Pa[:,:rg_Pa]@Sigma_Pa_demi
    
    
    for i in range(N-rg_Pa):
        
        # Centrage
        # de la matrice d'anomalies en conservant la covariance (Farchi et Bocquet 2019)
        eps = 1.0
        c = rg_Pa+i+1
        teta = np.sqrt(c)/(np.sqrt(c)-eps)

        
        
        Q_eps = -teta/c*np.ones([c,c])
        Q_eps[0,:] = eps/np.sqrt(c)
        Q_eps[:,0] = eps/np.sqrt(c)
        for j in range(1,c):
            Q_eps[j,j] = 1-teta/c

        W = np.zeros([Nx,c])
        W [:,1:c] = proj_X_Pa_demi
        proj_X_Pa_demi = W@Q_eps

    # Mise à jour de l'ensemble
    E=np.zeros((N,Nx))
    def fill_row_E(ll):
        return xa + np.sqrt(N1) * proj_X_Pa_demi[:, ll]


    with concurrent.futures.ProcessPoolExecutor() as executor:
        for ll, result in enumerate(executor.map(fill_row_E, range(N))):
            E[ll, :] = result
    
            
    return E


E=LETKF_kern_analysis(E, tend, E[:,obs_ind], GaussRV(C=4*np.eye(Ny)), nd_Id_kernel_loc(shape[::-1],jj,periodic=False), obs, kernel, R, dict())

print(E)
print(E.shape)