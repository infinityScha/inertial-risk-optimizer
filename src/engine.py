import numba as nb
import numpy as np

@nb.njit
def fire2(x0, f, alpha0=0.25, alphashrink=0.99, tmin=0.02, tmax=10.0, dtgrow=1.1, dtshrink=0.5, delaystep=20, Npmax=2000, initialdelay=True, Nmax=20000, f_tol=1e-6):
    """
    A Numba-optimized implementation of the FIRE2.0 (Fast Inertial Relaxation Engine 2.0) algorithm
    for energy minimization in molecular dynamics simulations. Based on the method described in https://doi.org/10.1016/j.commatsci.2020.109584

    Parameters:
    x0 : array_like
        Initial positions.
    f : callable
        Forces function that computes forces given positions.
    alpha0 : float
        Initial mixing parameter for velocity adjustment.
    alphashrink : float
        Factor by which to shrink alpha when conditions are met.
    tmin : float
        Minimum time step size.
    tmax : float
        Maximum time step size.
    dtgrow : float
        Factor by which to grow the time step when conditions are met.
    dtshrink : float
        Factor by which to shrink the time step when conditions are met.
    delaystep : int
        Number of steps to wait before applying certain adjustments.
    Npmax : int
        Maximum number of positive power steps before adjustments.
    initialdelay : bool
        Whether to apply an initial delay before starting adjustments.
    Nmax : int
        Maximum number of iterations.
    f_tol : float
        Force tolerance for convergence.

    Returns:
    x : array_like
        Updated positions of particles after applying the FIRE algorithm.
    converged : bool
        Whether the algorithm converged within the maximum number of iterations.
    """

    x = x0.copy()
    v = np.zeros_like(x0)
    alpha = alpha0
    dt = tmin
    Np_plus = 0
    Np_minus = 0
    converged = False

    fx = f(x)
    for step in range(Nmax):
        P = np.dot(fx, v)
        if P > 0:
            Np_plus += 1
            Np_minus = 0
            if Np_plus > delaystep:
                dt = min(dt * dtgrow, tmax)
                alpha0 *= alphashrink
        else:
            Np_plus = 0
            Np_minus += 1
            if Np_minus > Npmax:
                break
            if not (initialdelay and step < delaystep):
                dt = max(dt * dtshrink, tmin)
                alpha = alpha0
            x -= 0.5 * v * dt
            v[:] = 0.0
        # integration step, semi-implicit Euler (https://doi.org/10.1016/j.commatsci.2018.09.049)
        x += v * dt
        fx = f(x)
        norm_fx = np.linalg.norm(fx)
        if norm_fx < f_tol:
            converged = True
            break
        v = (1 - alpha) * v + alpha * fx / norm_fx * np.linalg.norm(v)
    return x, converged
