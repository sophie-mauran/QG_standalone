import numpy as np
from tools.matrices import center, genOG_1


def post_process(E, infl, rot):
    """Inflate, Rotate.

    To avoid recomputing/recombining anomalies,
    this should have been inside :func:`EnKF_analysis`

    But it is kept as a separate function

    - for readability;
    - to avoid inflating/rotationg smoothed states (for the :func:`EnKS`).
    """
    do_infl = infl != 1.0 and infl != '-N'

    if do_infl or rot:
        A, mu  = center(E)
        N, Nx  = E.shape
        T      = np.eye(N)

        if do_infl:
            T = infl * T

        if rot:
            T = genOG_1(N, rot) @ T

        E = mu + T@A
    return E