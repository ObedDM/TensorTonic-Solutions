import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.array(A)
    (N, M) = A.shape
    T = np.zeros((M, N))
    
    for n in range(N):
        for m in range(M):
            T[m, n] = A[n, m]
    
    return T
