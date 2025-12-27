import numpy as np
import numba as nb

@nb.njit
def get_multiplicity_2(n):
    """Generates the int8 multiplicity array for Covariance (N^2)"""
    size = n * (n + 1) // 2  # n choose 2 with replacement
    mults = np.zeros(size, dtype=np.int8)
    idx = 0
    for i in range(n):
        for j in range(i, n):
            if i == j:
                m = 1  # (i,i)
            else:
                m = 2  # (i,j) and (j,i)
            mults[idx] = m
            idx += 1
    return mults

@nb.njit
def get_multiplicity_3(n):
    """Generates the int8 multiplicity array for Skewness (N^3)"""
    size = n * (n + 1) * (n + 2) // 6 # n choose 3 with replacement (n+k+1 choose k, which is equivalet to putting k equivalent balls in n boxes)
    mults = np.zeros(size, dtype=np.int8)
    idx = 0
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                if i == k: m = 1 #(i.i,i)
                elif i == j or j == k: m = 3 # (i, i, k) or (i, j, j), each has 3 permutations
                else: m = 6 # all different (i,j,k), 6 permutations
                mults[idx] = m
                idx += 1
    return mults

@nb.njit
def get_multiplicity_4(n):
    """Generates the int8 multiplicity array for Kurtosis (N^4)"""
    size = n * (n + 1) * (n + 2) * (n + 3) // 24 # n choose 4 with replacement, similar logic as above
    mults = np.zeros(size, dtype=np.int8)
    idx = 0
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                for l in range(k, n):
                    if i == l: m = 1 #(i,i,i,i)
                    elif i == k or j == l: m = 4 # (i,i,i,l) or (i,j,j,j), each has 4 permutations
                    elif i == j and k == l: m = 6 # (i,i,k,k), 6 permutations, (K groups choose, N!/(N_1!*N_2!*...*N_k!))
                    elif i == j or j == k or k == l: m = 12 # (i,i,k,l), (i,j,j,l), (i,j,k,k), each has 12 permutations (K groups choose again)
                    else: m = 24 # all different (i,j,k,l), 24 permutations
                    mults[idx] = m
                    idx += 1
    return mults

@nb.njit
def compute_flat_moments(returns):
    """
    Computes the mean vector, covariance matrix, co-skewness tensor, and co-kurtosis tensor DIRECTLY into flattened arrays.
    Input: returns (T x N)
    Output: m1_flat (N,), m2_flat (N*(N+1)/2,), m3_flat (N*(N+1)*(N+2)/6,), m4_flat (N*(N+1)*(N+2)*(N+3)/24,)
    """
    T, N = returns.shape
    
    # Calculate sizes
    size2 = N * (N + 1) // 2
    size3 = N * (N + 1) * (N + 2) // 6
    size4 = N * (N + 1) * (N + 2) * (N + 3) // 24
    
    m1_flat = np.zeros(N, dtype=np.float64)
    m2_flat = np.zeros(size2, dtype=np.float64)
    m3_flat = np.zeros(size3, dtype=np.float64)
    m4_flat = np.zeros(size4, dtype=np.float64)
    
    # Pre-calculate normalization factor (1/T), could use unbiased versions but for large T this is negligible 
    # also, with 1/T we obtain maximum likelihood estimators which is theoretically easier to work with
    if T < 1:
        raise ValueError("There should be at least one time step to compute moments.")
    
    inv_T = 1.0 / T

    # mean loop
    for i in range(N):
        col_i = returns[:, i]
        val = np.sum(col_i)  # sum over time dimension
        m1_flat[i] = val * inv_T

    # Covariance loop
    idx2 = 0
    for i in range(N):
        col_i = returns[:, i]
        for j in range(i, N):
            col_j = returns[:, j]
            # Sum over time dimension (The contraction)
            # sum(r_it * r_jt)
            val = np.dot(col_i, col_j)
            m2_flat[idx2] = val * inv_T
            idx2 += 1

    # TODO: validate that the skewness and kurtosis calculations are correct, specifically that they represent the access values and normalized correctly to the variances if needed
    # Skewness loop
    idx3 = 0
    for i in range(N):
        col_i = returns[:, i]
        for j in range(i, N):
            col_j = returns[:, j]
            # Pre-multiply columns to save time in inner loop
            col_ij = col_i * col_j
            
            for k in range(j, N):
                col_k = returns[:, k]
                
                # Sum over time dimension (The contraction)
                # sum(r_it * r_jt * r_kt)
                val = np.dot(col_ij, col_k)
                m3_flat[idx3] = val * inv_T
                idx3 += 1

    # Kurtosis loop
    idx4 = 0
    for i in range(N):
        col_i = returns[:, i]
        for j in range(i, N):
            col_j = returns[:, j]
            col_ij = col_i * col_j
            
            for k in range(j, N):
                col_k = returns[:, k]
                col_ijk = col_ij * col_k
                
                for l in range(k, N):
                    col_l = returns[:, l]
                    
                    # Sum over time dimension
                    val = np.dot(col_ijk, col_l)
                    m4_flat[idx4] = val * inv_T
                    idx4 += 1

    # fix multiplicities
    mult2 = get_multiplicity_2(N)
    mult3 = get_multiplicity_3(N)
    mult4 = get_multiplicity_4(N)

    m2_flat *= mult2.astype(np.float64)
    m3_flat *= mult3.astype(np.float64)
    m4_flat *= mult4.astype(np.float64)
                    
    return m1_flat, m2_flat, m3_flat, m4_flat
