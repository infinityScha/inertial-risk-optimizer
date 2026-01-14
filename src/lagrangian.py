import numpy as np
import numba as nb
from src.objectives import compute_modified_VaR, compute_modified_VaR_gradient


@nb.njit
def compute_abs_unity_constraint(w):
    """
    Computes absolute value constraint for weights to sum to one.
    Input: w (array)
    Output: penalty value (float)
    """
    s = np.sum(w) - 1.0
    return np.abs(s)


@nb.njit
def compute_unity_constraint(w, rho1, lam1):
    """
    Computes constraint for weights to sum to one.
    Input: w (array), rho1 (float), lam1 (float)
    Output: penalty value (float)
    """
    s = np.sum(w) - 1.0
    return 0.5 * rho1 * s * s + lam1 * s


@nb.njit
def compute_unity_constraint_gradient(w, rho1, lam1):
    """
    Computes gradient of the unity constraint with respect to weights.
    Input: w (array), rho1 (float), lam1 (float)
    Output: gradient array
    """
    s = np.sum(w) - 1.0
    ones = np.ones_like(w)
    return (rho1 * s + lam1) * ones


@nb.njit
def compute_abs_positivity_constraint(w):
    """
    Computes absolute value constraint for weights to be non-negative.
    Input: w (array)
    Output: penalty value (float)
    """
    return -np.minimum(w, 0.0)


@nb.njit
def compute_positivity_constraint(w, rho2, lams2):
    shifted_violation = -w + (lams2 / rho2)
    shifted_violation = shifted_violation[shifted_violation > 0.0]
    
    return 0.5 * rho2 * np.sum(shifted_violation**2)


@nb.njit
def compute_positivity_constraint_gradient(w, rho2, lams2):
    shifted_violation = -w + (lams2 / rho2)

    grad = np.zeros_like(w)
    for i in range(len(w)):
        if shifted_violation[i] > 0.0:
            grad[i] = rho2 * w[i] - lams2[i]
    
    return grad


@nb.njit
def compute_lagrangian(w, param_tuple):
    """
    Computes total Lagrangian penalty for weights.
    Input: w (array), param_tuple (tuple): m1_flat (array), m2_flat (array), m3_flat (array), m4_flat (array), z_alpha (float), include_positivity (bool), rho1 (float), lam1 (float), rho2 (float), lams2 (array)
    Output: total value (float)
    """
    m1_flat, m2_flat, m3_flat, m4_flat, z_alpha, include_positivity, rho1, lam1, rho2, lams2 = param_tuple
    modified_VaR = compute_modified_VaR(w, m1_flat, m2_flat, m3_flat, m4_flat, z_alpha)
    unity_penalty = compute_unity_constraint(w, rho1, lam1)
    positivity_penalty = compute_positivity_constraint(w, rho2, lams2) if include_positivity else 0.0
    return modified_VaR + unity_penalty + positivity_penalty


@nb.njit
def compute_lagrangian_gradient(w, param_tuple):
    """
    Computes gradient of total Lagrangian penalty for weights.
    Input: w (array), param_tuple (tuple): m1_flat (array), m2_flat (array), m3_flat (array), m4_flat (array), z_alpha (float), include_positivity (bool), rho1 (float), lam1 (float), rho2 (float), lams2 (array)
    Output: gradient array
    """
    m1_flat, m2_flat, m3_flat, m4_flat, z_alpha, include_positivity, rho1, lam1, rho2, lams2 = param_tuple
    modified_VaR_grad = compute_modified_VaR_gradient(w, m1_flat, m2_flat, m3_flat, m4_flat, z_alpha)
    unity_grad = compute_unity_constraint_gradient(w, rho1, lam1)
    positivity_grad = compute_positivity_constraint_gradient(w, rho2, lams2) if include_positivity else np.zeros_like(w)
    return modified_VaR_grad + unity_grad + positivity_grad
