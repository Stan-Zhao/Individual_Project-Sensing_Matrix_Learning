import numpy as np
from numpy.linalg import norm, lstsq

def cosamp(Phi, y, s, max_iter=10, tol=1e-6):
    """
    CoSaMP algorithm
    
    Parameters
    ----------
    Phi : ndarray, shape (m, n)
        Measurement matrix (columns normalized)
    y : ndarray, shape (m,)
        Measurement vector
    s : int
        Sparsity level
    max_iter : int
        Maximum iterations
    tol : float
        Residual tolerance
        
    Returns
    -------
    x_hat : ndarray, shape (n,)
        Recovered sparse signal
    """

    m, n = Phi.shape
    x_hat = np.zeros(n)
    residual = y.copy()
    support = np.array([], dtype=int)

    for it in range(max_iter):

        # ---- 1. Proxy step ----
        proxy = Phi.T @ residual
        omega = np.argsort(np.abs(proxy))[-2*s:]

        # ---- 2. Merge ----
        T = np.union1d(support, omega)

        # 防止欠定
        if len(T) > m:
            T = T[:m]

        # ---- 3. Least Squares on merged support ----
        Phi_T = Phi[:, T]
        b_T, _, _, _ = lstsq(Phi_T, y, rcond=None)

        # ---- 4. Prune ----
        idx = np.argsort(np.abs(b_T))[-s:]
        support = T[idx]

        # ---- 5. Update signal and residual ----
        x_hat = np.zeros(n)
        Phi_S = Phi[:, support]
        x_S, _, _, _ = lstsq(Phi_S, y, rcond=None)
        x_hat[support] = x_S

        new_residual = y - Phi @ x_hat

        # ---- 6. Stopping rule ----
        if norm(new_residual) < tol:
            break

        residual = new_residual

    return x_hat
