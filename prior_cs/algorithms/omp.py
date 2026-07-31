import numpy as np
from numpy.linalg import pinv, norm

def omp(Phi, y, k, tol=1e-6):
    m, n = Phi.shape
    r = y.copy()
    omega = []
    x_hat = np.zeros(n)
    
    for _ in range(k):
        correlations = np.abs(Phi.T @ r)
        correlations[omega] = 0
        i = np.argmax(correlations)
        if correlations[i] < tol:
            break
        omega.append(i)
        x_hat[omega] = pinv(Phi[:, omega]) @ y
        r = y - Phi[:, omega] @ x_hat[omega]
    return x_hat
