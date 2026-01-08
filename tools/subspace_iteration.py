import numpy as np
import scipy.linalg as scpl
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import sparseqr

def Kernel_subspace_iteration(A,D,m,p,rmax,tol,niter,mode, full_rank):

    #K=A*D*transpose(A)
    # m: rankf of subspace V
    # tol: relative errors of eigenvalues
    # niter: itermax
    #p: block strategy
    # rmax :expected maximum rank
    #mode : ADAT ou inverse
    # full_rank : svd or lu
    
    
    condition=0
    cpt_iter=0
    nx=len(A[:,0])
    V=np.random.randn(nx,m)
    V,_=np.linalg.qr(V,mode='reduced')
    converged=0
    rlambda=np.max(np.abs(D))*np.max(np.abs(A))**2
    
    if(mode==1):
       A,R=np.linalg.qr(A,mode='reduced') 
       D=R@(D@np.transpose(R))
       if(full_rank==1):
          p_index,D,U=scpl.lu(D,p_indices=True)
       else:
          U,Sigma,_=np.linalg.svd(D,hermitian='True')
          rg_D=0
          nech=len(Sigma)
          for ll in range(nech):
             if(Sigma[ll]>np.finfo(float).eps*Sigma[0]):
                rg_D=rg_D+1
             else:
                break
          U=U[:,:rg_D]
          Sigma=1/Sigma[:rg_D]
	  
    while(condition==0):
       #matvec products
       if(mode==1):
          for pp in range(p):
             V=np.transpose(A)@V
             if(full_rank==1):
                V=scpl.solve_triangular(D,V[p_index,:],lower=True)
                V=A@(scpl.solve_triangular(U,V,lower=False))
             else:
                V=np.diag(Sigma)@(np.transpose(U)@V)
                V=A@(U@V)
       else:
          for pp in range(p):
             V=np.transpose(A)@V
             V=A@(D@V)
       
       #orthonormalization	 
       V,_=np.linalg.qr(V,mode='reduced')
       
       #Ritz projection 
       if(mode==1):
          tmp=np.transpose(A)@V
          if(full_rank==1):
             H=scpl.solve_triangular(D,tmp[p_index,:],lower=True)
             H=scpl.solve_triangular(U,H,lower=False)
          else:
             H=np.diag(Sigma)@(np.transpose(U)@tmp)
             H=U@H
       else:
          H=D@(np.transpose(A)@V)
       H=np.transpose(V)@(A@H)
       
       # eigenpairs
       U_l,Sig_l,_=np.linalg.svd(H,full_matrices='False',hermitian='True')     
       V=V@U_l
       
       if(mode==1):
           for i in np.arange(m-1-converged,0,-1):
               Avi=np.transpose(A)@V[:,i]
               if(full_rank==1):
                  Avi=scpl.solve_triangular(D,Avi[p_index],lower=True)
                  Avi=A@scpl.solve_triangular(U,Avi,lower=False)
               else:
                  Avi=np.diag(Sigma)@(np.transpose(U)@Avi)
                  Avi=A@(U@Avi)
               if(np.linalg.norm(Avi-Sig_l[i]*V[:,i])<tol*rlambda):
                  converged=converged+1
       else:
           for i in np.arange(converged,m,1):
               Avi=A@(D@(np.transpose(A)@V[:,i]))
               if(np.linalg.norm(Avi-Sig_l[i]*V[:,i])<tol*rlambda):
                  converged=converged+1

       cpt_iter+=1
       if ((cpt_iter>niter) or (converged>=rmax)):
           condition=1
	
    return V, Sig_l,converged


def Kernel_sis(A,nx,maxA, m,p,rmax,tol,niter):
    
    condition=0
    cpt_iter=0
    V=np.random.randn(nx,m)
    V,_=np.linalg.qr(V,mode='reduced')
    converged=0
    
    #facto LU de A, plus solveur
    print('Factorization..')	       
    invA=spsla.spilu(A)	 
    print('..Done')
    while(condition==0):
       #block strategy 
       for pp in range(p):
           Y=invA.solve(V)
           #for i in range(m):
           #    V[:,i],_=spsla.cg(A, V[:,i], rtol=1e-4, maxiter=100)
       #orthonormalization	 
       V,_=np.linalg.qr(Y,mode='reduced')
       
       #Rayleigh-Ritz
       H=V@invA.solve(V)
       
       # eigenpairs
       U_l,Sig_l,_=np.linalg.svd(H,full_matrices='False',hermitian='True')     
       V=V@U_l
       
       #stopping criterion
       for i in np.arange(m-1-converged,0,-1):
           Avi=invA.solve(V[:,i])
           #Avi,_=spsla.cg(A, V[:,i], rtol=1e-4, maxiter=100)
       #orthonormalization	 
           if(np.linalg.norm(Avi-Sig_l[i]*V[:,i])<tol*maxA):
                  converged=converged+1
       print(converged)	  
       cpt_iter+=1
       if ((cpt_iter>niter) or (converged>=rmax)):
           condition=1
	   
    return V, Sig_l,converged    

def Kernel_QR_subspace(A,nx,maxA, m,p,rmax,tol,niter):
    
    condition=0
    cpt_iter=0
    V=np.random.randn(nx,m)
    V,_=np.linalg.qr(V,mode='reduced')
    converged=0
    
    print('Factorization..')	
    Q, R, E, rank = sparseqr.qr(A, economy=True )
    Q=Q.tocsc()[:,:rank]
    R=R.tocsc()[:rank,:rank]
    print(rank)
    print('..Done')
    
    
    while(condition==0):
       for pp in range(p):
           V=Q.T@V
           tmp=spsla.spsolve(R,sps.csc_matrix(V))
           V=np.zeros((nx,m))
           V[:rank,:]=tmp.todense()
           V[E,:]=V.copy()
       #orthonormalization	 
       V,_=np.linalg.qr(V,mode='reduced')
       
       #Rayleigh-Ritz
       H=Q.T@V
       tmp=spsla.spsolve(R,sps.csc_matrix(H))
       H=np.zeros((nx,m))
       H[:rank,:]=tmp.todense()
       H[E,:]=H.copy()
       H=V.T@H
       
       # eigenpairs
       U_l,Sig_l,_=np.linalg.svd(H,full_matrices='False',hermitian='True')     
       V=V@U_l
       
       #stopping criterion
       for i in np.arange(converged,m,1):
           Avi=Q.T@V[:,i]
           tmp=spsla.spsolve(R,Avi)
           Avi=np.zeros((nx,))
           Avi[:rank]=tmp
           Avi[E]=Avi.copy()
           #Avi,_=spsla.cg(A, V[:,i], rtol=1e-4, maxiter=100)
       #orthonormalization	 
           if(np.linalg.norm(Avi-Sig_l[i]*V[:,i])<tol*maxA):
                  converged=converged+1
       #print(converged)	  
       cpt_iter+=1
       if ((cpt_iter>niter) or (converged>=min(rank,rmax))):
           condition=1
	   
    return V, Sig_l,converged    


def Covariance_subspace_iteration(Pa,m,p,tol_vp,tol_app, niter):

    # Pa v
    # m: rankf of subspace V
    # tol: relative errors of eigenvalues
    # niter: itermax
    #p: block strategy
    # var_tot : variance 
    
    trace_Pa=Pa.trace()
    tot_var=trace_Pa#[0,0]

    condition=0
    cpt_iter=0
    nx=Pa.shape[0]
    V=np.random.randn(nx,m)
    V,_=np.linalg.qr(V,mode='reduced')
    converged=0
    acc_var=0
    rlambda=np.linalg.norm(Pa,ord='fro')
   
    while(condition==0):
        #matvec products
        for pp in range(p):
           V=Pa@V
       
        #orthonormalization	 
        V,_=np.linalg.qr(V,mode='reduced')
       
        #Ritz projection 
        H=Pa@V
        H=np.transpose(V)@H
       
        # eigenpairs
        U_l,Sig_l,_=np.linalg.svd(H,full_matrices='False',hermitian='True')     
        V=V@U_l
       

        for i in np.arange(converged,m,1):
            Avi=Pa@V[:,i]
            if(np.linalg.norm(Avi-Sig_l[i]*V[:,i])<tol_vp*rlambda):
                converged=converged+1
                acc_var+=Sig_l[i]
                #print(acc_var)
        print(converged)       
        cpt_iter+=1
        if ((cpt_iter>niter) or (np.sqrt(tot_var-acc_var)<tol_app*np.sqrt(tot_var))):
            condition=1
            print('accuracy approc')
            print(acc_var/tot_var)
        elif(converged==m):
            condition=1
            print('m too small')
            print(acc_var/tot_var)
    return V,Sig_l,converged,acc_var



#nx=100
#ns=30
#m=40
#rmax=ns
#A=np.random.randn(nx,ns)
#S=np.random.randn(ns,ns)
#S=np.dot(S,np.transpose(S))
#A=sps.rand(nx,ns)
#A=A@A.T

#B=A.todense()
#K=A@(S@np.transpose(A))

#U_l,Sig_l,V_l=np.linalg.svd(B,full_matrices='False',hermitian='True')
#myV, mySig,converged,acc_var =Covariance_subspace_iteration(A,m,2,1e-8,0.2,1000)

#print(np.shape(mySig))
#rmse_Sig=np.zeros((min(m,ns),))
#for i in range(min(m,ns)):
#    rmse_Sig[i]=np.abs(Sig_l[i]-mySig[i])/np.abs(Sig_l[i])
#print(rmse_Sig)
#print(Sig_l[:m])
#print(converged)
#print(mySig)

