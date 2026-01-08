import numpy as np
from tools.linalg import mldiv, mrdiv, pad0, svd0, svdi, tinv, tsvd
from tools.matrices import funm_psd, center
import numpy.random as rnd

def add_noise(E, dt, noise, method):
    """Treatment of additive noise for ensembles.

    Refs: `bib.raanes2014ext`
    """
    if noise.C == 0:
        return E

    N, Nx = E.shape
    A, mu = center(E)
    Q12   = noise.C.Left
    Q     = noise.C.full

    def sqrt_core():
        T    = np.nan    # cause error if used
        Qa12 = np.nan    # cause error if used
        A2   = A.copy()  # Instead of using (the implicitly nonlocal) A,
        # which changes A outside as well. NB: This is a bug in Datum!
        if N <= Nx:
            Ainv = tinv(A2.T)
            Qa12 = Ainv@Q12
            T    = funm_psd(np.eye(N) + dt*(N-1)*(Qa12@Qa12.T), np.sqrt)
            A2   = T@A2
        else:  # "Left-multiplying" form
            P  = A2.T @ A2 / (N-1)
            L  = funm_psd(np.eye(Nx) + dt*mrdiv(Q, P), np.sqrt)
            A2 = A2 @ L.T
        E = mu + A2
        return E, T, Qa12

    if method == 'Stoch':
        # In-place addition works (also) for empty [] noise sample.
        E += np.sqrt(dt)*noise.sample(N)

    elif method == 'none':
        pass

    elif method == 'Mult-1':
        varE   = np.var(E, axis=0, ddof=1).sum()
        ratio  = (varE + dt*np.diag(Q).sum())/varE
        E      = mu + np.sqrt(ratio)*A
        E      = svdi(*tsvd(E, 0.999))  # Explained in Datum

    elif method == 'Mult-M':
        varE   = np.var(E, axis=0)
        ratios = np.sqrt((varE + dt*np.diag(Q))/varE)
        E      = mu + A*ratios
        E      = svdi(*tsvd(E, 0.999))  # Explained in Datum

    elif method == 'Sqrt-Core':
        E = sqrt_core()[0]

    elif method == 'Sqrt-Mult-1':
        varE0 = np.var(E, axis=0, ddof=1).sum()
        varE2 = (varE0 + dt*np.diag(Q).sum())
        E, _, Qa12 = sqrt_core()
        if N <= Nx:
            A, mu   = center(E)
            varE1   = np.var(E, axis=0, ddof=1).sum()
            ratio   = varE2/varE1
            E       = mu + np.sqrt(ratio)*A
            E       = svdi(*tsvd(E, 0.999))  # Explained in Datum

    elif method == 'Sqrt-Add-Z':
        E, _, Qa12 = sqrt_core()
        if N <= Nx:
            Z  = Q12 - A.T@Qa12
            E += np.sqrt(dt)*(Z@rnd.randn(Z.shape[1], N)).T

    elif method == 'Sqrt-Dep':
        E, T, Qa12 = sqrt_core()
        if N <= Nx:
            # Q_hat12: reuse svd for both inversion and projection.
            Q_hat12      = A.T @ Qa12
            U, s, VT     = tsvd(Q_hat12, 0.99)
            Q_hat12_inv  = (VT.T * s**(-1.0)) @ U.T
            Q_hat12_proj = VT.T@VT
            rQ = Q12.shape[1]
            # Calc D_til
            Z      = Q12 - Q_hat12
            D_hat  = A.T@(T-np.eye(N))
            Xi_hat = Q_hat12_inv @ D_hat
            Xi_til = (np.eye(rQ) - Q_hat12_proj)@rnd.randn(rQ, N)
            D_til  = Z@(Xi_hat + np.sqrt(dt)*Xi_til)
            E     += D_til.T

    else:
        raise KeyError('No such method')

    return E