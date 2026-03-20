import numpy as np
def matrix_transpose(A):
    # Get dimensions
    n = len(A)
    m = len(A[0])
    # Create result matrix (m x n)
    result = np.zeros((m, n), dtype=type(A[0][0]))
    # Fill transpose
    for i in range(n):
        for j in range(m):
            result[j][i] = A[i][j]
    
    return result