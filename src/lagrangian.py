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
    """
    Computes constraint for weights to be non-negative.
    Input: w (array), rho2 (float), lams2 (array)
    Output: penalty value (float)
    """
    neg_w = np.minimum(w, 0.0)
    sum_neg2 = np.sum(neg_w*neg_w)
    return 0.5 * rho2 * sum_neg2 + np.dot(lams2, neg_w)


@nb.njit
def compute_positivity_constraint_gradient(w, rho2, lams2):
    """
    Computes gradient of the positivity constraint with respect to weights.
    Input: w (array), rho2 (float), lams2 (array)
    Output: gradient array
    """
    neg_mark = w < 0.0
    neg_w = np.where(neg_mark, w, 0.0)
    rel_lams2 = np.where(neg_mark, lams2, 0.0)
    return rho2 * neg_w + rel_lams2


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
