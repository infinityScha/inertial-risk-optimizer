import numpy as np
import numba as nb
from scipy.stats import norm


def compute_z_alpha(confidence_level):
    """
    Computes the z-score for a given confidence level using the inverse CDF of the standard normal distribution.
    Input: confidence_level (float)
    Output: z_alpha (float)
    """
    z_alpha = norm.ppf(confidence_level)
    return z_alpha


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
    Computes the centralized moments (except the first moment) DIRECTLY into flattened arrays.
    Input: returns (T x N)
    Output: m1_flat (N), m2_flat (N*(N+1)/2), m3_flat (N*(N+1)*(N+2)/6), m4_flat (N*(N+1)*(N+2)*(N+3)/24)
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
    # also, with 1/T we obtain maximum likelihood estimators which are theoretically "well behaved" so when we use them 
    # in further calculations (e.g., portfolio optimization) they will be easier to handle and interpret.
    if T < 1:
        raise ValueError("There should be at least one time step to compute moments.")
    
    inv_T = 1.0 / T

    # mean loop
    for i in range(N):
        col_i = returns[:, i]
        val = np.sum(col_i)  # sum over time dimension
        m1_flat[i] = val * inv_T


    # centralize the returns for higher moments
    centralized_returns = returns.copy()
    for i in range(N):
        centralized_returns[:, i] -= m1_flat[i]

    # Covariance loop
    idx2 = 0
    for i in range(N):
        col_i = centralized_returns[:, i]
        for j in range(i, N):
            col_j = centralized_returns[:, j]
            # Sum over time dimension (The contraction)
            # sum(r_it * r_jt)
            val = np.dot(col_i, col_j)
            m2_flat[idx2] = val * inv_T
            idx2 += 1

    # TODO: validate that the skewness and kurtosis calculations are correct, specifically that they represent the access values and normalized correctly to the variances if needed - I'm not sure it is needed as we should probably normalize in after the summation.
    # Skewness loop
    idx3 = 0
    for i in range(N):
        col_i = centralized_returns[:, i]
        for j in range(i, N):
            col_j = centralized_returns[:, j]
            # Pre-multiply columns to save time in inner loop
            col_ij = col_i * col_j
            
            for k in range(j, N):
                col_k = centralized_returns[:, k]
                
                # Sum over time dimension (The contraction)
                # sum(r_it * r_jt * r_kt)
                val = np.dot(col_ij, col_k)
                m3_flat[idx3] = val * inv_T
                idx3 += 1

    # Kurtosis loop
    idx4 = 0
    for i in range(N):
        col_i = centralized_returns[:, i]
        for j in range(i, N):
            col_j = centralized_returns[:, j]
            col_ij = col_i * col_j
            
            for k in range(j, N):
                col_k = centralized_returns[:, k]
                col_ijk = col_ij * col_k
                
                for l in range(k, N):
                    col_l = centralized_returns[:, l]
                    
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


@nb.njit(fastmath=True)
def compute_flat_moments_gradients(w, m1_flat, m2_flat, m3_flat, m4_flat):
    """
    Computes the gradients of the centralized moments of the portfolio DIRECTLY into flattened arrays.
    Input: w (N), m1_flat (N), m2_flat (N*(N+1)/2), m3_flat (N*(N+1)*(N+2)/6), m4_flat (N*(N+1)*(N+2)*(N+3)/24)
    Output: grad_m1_flat (N), grad_m2_flat (N), grad_m3_flat (N), grad_m4_flat (N)
    """
    N = w.shape[0]

    grad_m1_flat = m1_flat.copy()
    grad_m2_flat = np.zeros((N), dtype=np.float64)
    grad_m3_flat = np.zeros((N), dtype=np.float64)
    grad_m4_flat = np.zeros((N), dtype=np.float64)

    # Gradient of the second moment (covariance)
    idx2 = 0
    for i in range(N):
        wi = w[i]
        for j in range(i, N):
            mij = m2_flat[idx2]
            # d/dwi
            grad_m2_flat[i] += mij * w[j]
            # d/dwj
            grad_m2_flat[j] += mij * wi
            idx2 += 1
    
    # Precompute outer product of weights for avoiding redundant calculations
    w_ij = np.outer(w, w)

    # Gradient of the third moment (skewness)
    idx3 = 0
    for i in range(N):
        for j in range(i, N):
            for k in range(j, N):
                mijk = m3_flat[idx3]
                # d/dwi
                grad_m3_flat[i] += mijk * w_ij[j, k]
                # d/dwj
                grad_m3_flat[j] += mijk * w_ij[i, k]
                # d/dwk
                grad_m3_flat[k] += mijk * w_ij[i, j]
                idx3 += 1
    
    # Gradient of the fourth moment (kurtosis)
    idx4 = 0
    for i in range(N):
        for j in range(i, N):
            for k in range(j, N):
                for l in range(k, N):
                    mijkl = m4_flat[idx4]
                    # d/dwi
                    grad_m4_flat[i] += mijkl * w_ij[j, k] * w[l]
                    # d/dwj
                    grad_m4_flat[j] += mijkl * w_ij[i, k] * w[l]
                    # d/dwk
                    grad_m4_flat[k] += mijkl * w_ij[i, j] * w[l]
                    # d/dwl
                    grad_m4_flat[l] += mijkl * w_ij[i, j] * w[k]
                    idx4 += 1

    return grad_m1_flat, grad_m2_flat, grad_m3_flat, grad_m4_flat


@nb.njit(fastmath=True)
def compute_portfolio_mean(w, m1_flat):
    """
    Computes the mean of the portfolio.
    Input: w (N), m1_flat (N)
    Output: mu_p (float)
    """
    mu_p = np.dot(w, m1_flat)
    return mu_p


@nb.njit(fastmath=True)
def compute_portfolio_variance(w, m2_flat):
    """
    Computes the variance of the portfolio.
    Input: w (N), m2_flat (N*(N+1)/2)
    Output: sigma2_p (float)
    """
    sigma2_p = 0.0
    idx2 = 0
    N = w.shape[0]
    for i in range(N):
        wi = w[i]
        for j in range(i, N):
            mij = m2_flat[idx2]
            sigma2_p += wi * w[j] * mij
            idx2 += 1
    return sigma2_p


@nb.njit(fastmath=True)
def compute_portfolio_skewness(w, m3_flat, sigma_p):
    """
    Computes the skewness of the portfolio.
    Input: w (N), m3_flat (N*(N+1)*(N+2)/6), sigma_p (float)
    Output: skew_p (float)
    """
    skew_p = 0.0
    idx3 = 0
    N = w.shape[0]
    for i in range(N):
        wi = w[i]
        for j in range(i, N):
            wi_wj = wi * w[j]
            for k in range(j, N):
                mijk = m3_flat[idx3]
                skew_p += wi_wj * w[k] * mijk
                idx3 += 1
    skew_p /= sigma_p**3
    return skew_p


@nb.njit(fastmath=True)
def compute_portfolio_kurtosis(w, m4_flat, sigma_p):
    """
    Computes the kurtosis of the portfolio.
    Input: w (N), m4_flat (N*(N+1)*(N+2)*(N+3)/24), sigma_p (float)
    Output: kurt_p (float)
    """
    kurt_p = 0.0
    idx4 = 0
    N = w.shape[0]
    for i in range(N):
        wi = w[i]
        for j in range(i, N):
            wi_wj = wi * w[j]
            for k in range(j, N):
                wi_wj_wk = wi_wj * w[k]
                for l in range(k, N):
                    mijkl = m4_flat[idx4]
                    kurt_p += wi_wj_wk * w[l] * mijkl
                    idx4 += 1
    kurt_p /= sigma_p**4
    return kurt_p


@nb.njit(fastmath=True)
def compute_modified_VaR(w, m1_flat, m2_flat, m3_flat, m4_flat, z_alpha):
    """
    Computes the modified VaR of the portfolio using the centralized moments.
    Input: w (N), m1_flat (N), m2_flat (N*(N+1)/2), m3_flat (N*(N+1)*(N+2)/6), m4_flat (N*(N+1)*(N+2)*(N+3)/24), z_alpha (float)
    Output: modified_VaR (float)
    """
    
    # compute portfolio mean, variance, skewness, kurtosis
    mu_p = compute_portfolio_mean(w, m1_flat)
    sigma2_p = compute_portfolio_variance(w, m2_flat)
    sigma_p = np.sqrt(sigma2_p)
    skew_p = compute_portfolio_skewness(w, m3_flat, sigma_p)
    kurt_p = compute_portfolio_kurtosis(w, m4_flat, sigma_p)

    # Modified VaR calculation
    # TODO: check if I need to use the -3
    z_mod = (z_alpha +
             (1/6) * (z_alpha**2 - 1) * skew_p +
             (1/24) * (z_alpha**3 - 3*z_alpha) * (kurt_p - 3) -
             (1/36) * (2*z_alpha**3 - 5*z_alpha) * skew_p**2)
    
    modified_VaR = -(mu_p + z_mod * sigma_p)

    return modified_VaR


@nb.njit(fastmath=True)
def compute_modified_VaR_gradient(w, m1_flat, m2_flat, m3_flat, m4_flat, z_alpha):
    """
    Computes the gradient of the modified VaR of the portfolio using the centralized moments.
    Input: w (N), m1_flat (N), m2_flat (N*(N+1)/2), m3_flat (N*(N+1)*(N+2)/6), m4_flat (N*(N+1)*(N+2)*(N+3)/24), z_alpha (float)
    Output: grad_modified_VaR (N)
    """
    N = w.shape[0]
    
    # compute portfolio mean, variance, skewness, kurtosis
    sigma2_p = compute_portfolio_variance(w, m2_flat)
    sigma_p = np.sqrt(sigma2_p)
    skew_p = compute_portfolio_skewness(w, m3_flat, sigma_p)
    kurt_p = compute_portfolio_kurtosis(w, m4_flat, sigma_p)

    # compute gradients of moments
    grad_m1_flat, grad_m2_flat, grad_m3_flat, grad_m4_flat = compute_flat_moments_gradients(w, m1_flat, m2_flat, m3_flat, m4_flat)

    # Precompute common terms for gradient calculation
    inv_sigma_p = 1.0 / sigma_p
    inv_sigma2_p = inv_sigma_p * inv_sigma_p
    inv_sigma3_p = inv_sigma2_p * inv_sigma_p
    inv_sigma4_p = inv_sigma3_p * inv_sigma_p
    inv_sigma5_p = inv_sigma4_p * inv_sigma_p

    # Gradient of modified VaR
    grad_modified_VaR = np.zeros(N, dtype=np.float64)

    z_mod = (z_alpha +
                (1/6) * (z_alpha**2 - 1) * skew_p +
                (1/24) * (z_alpha**3 - 3*z_alpha) * (kurt_p - 3) -
                (1/36) * (2*z_alpha**3 - 5*z_alpha) * skew_p**2)

    # Compute gradient components
    for i in range(N):
        dmu_dw = grad_m1_flat[i]
        dsigma2_dw = grad_m2_flat[i]
        dsigma_dw = 0.5 * dsigma2_dw * inv_sigma_p
        dskew_dw = grad_m3_flat[i] * inv_sigma3_p - 3 * skew_p * dsigma_dw * inv_sigma_p
        dkurt_dw = grad_m4_flat[i] * inv_sigma4_p - 4 * kurt_p * dsigma_dw * inv_sigma_p
        # todo: check if I need to use the -3
        dzmod_dw = ((1/6) * (z_alpha**2 - 1) * dskew_dw +
                    (1/24) * (z_alpha**3 - 3*z_alpha) * dkurt_dw -
                    (1/18) * (2*z_alpha**3 - 5*z_alpha) * skew_p * dskew_dw)
        grad_modified_VaR[i] = -(dmu_dw + dzmod_dw * sigma_p + z_mod * dsigma_dw)
    
    return grad_modified_VaR
