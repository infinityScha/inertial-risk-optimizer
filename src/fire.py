import numpy as np
from src.engine import fire2
from src.objectives import compute_modified_VaR, compute_z_alpha, compute_flat_moments
from src.lagrangian import compute_lagrangian, compute_lagrangian_gradient, compute_abs_unity_constraint, compute_abs_positivity_constraint
from tqdm import tqdm

class FireOptimizer():
    """
    A Numba-optimized implementation of the FIRE2.0 (Fast Inertial Relaxation Engine 2.0) algorithm
    for energy minimization in molecular dynamics simulations. Based on the method described in https://doi.org/10.1016/j.commatsci.2020.109584
    """

    def __init__(self, returns, long_only = True, rho1=0.5, rho2=4.0, f_tol=1e-6, c_tol=1e-6, verbose=False, max_outer_iterations=1e6):
        if returns is None:
            raise ValueError("Returns data must be provided for optimization.")
        self.long_only = long_only
        self.rho1 = rho1
        self.rho2 = rho2
        self.f_tol = f_tol
        self.c_tol = c_tol
        self.verbose = verbose
        self.max_outer_iterations = int(max_outer_iterations)

        # calculate and store moments
        self.m1_flat, self.m2_flat, self.m3_flat, self.m4_flat = compute_flat_moments(returns)


    def optimize(self, confidence=0.99, x0=None):
        if x0 is None:
            # exponential distribution ensures a uniform distribution over the simplex
            x0 = np.random.exponential(1.0, size=self.m1_flat.shape[0])
            x0 /= np.sum(x0)

        z_alpha = compute_z_alpha(confidence)

        converged = False
        w = x0.copy()
        lam1 = 0.0
        lams2 = np.zeros_like(w)
        counter = 0
        with tqdm(total=self.max_outer_iterations, desc="FIRE Optimizer", disable=not self.verbose) as pbar:
            while not converged and counter < self.max_outer_iterations:
                param_tuple = (self.m1_flat, self.m2_flat, self.m3_flat, self.m4_flat,
                                z_alpha, self.long_only,
                                self.rho1, lam1,
                                self.rho2, lams2)

                # Run the FIRE inner loop (10000 steps tops, should converge earlier)
                w, converged = fire2(w, compute_lagrangian_gradient, param_tuple, Nmax=1000000, f_tol=self.f_tol)
                
                # Check constraints, if satisfied accept convergence
                converged = converged and self._check_feasibility(w, lams2) 
                
                if converged:
                    break

                # Update multipliers
                lam1, lams2 = self._update_lagrange_multipliers(w, self.rho1, lam1, self.rho2, lams2)
                
                # Update progress bar
                grad_norm = np.linalg.norm(compute_lagrangian_gradient(w, param_tuple))
                lagrangian_value = compute_lagrangian(w, param_tuple)
                unity_violation = compute_abs_unity_constraint(w)
                mVaR = compute_modified_VaR(w, self.m1_flat, self.m2_flat, self.m3_flat, self.m4_flat, z_alpha)
                sigma_ineq = np.maximum(-w, -lams2 / self.rho2)
                max_ineq_violation = np.max(np.abs(sigma_ineq))
                pbar.set_description(f"FIRE [G_norm: {grad_norm:.2e} | Lagrangian: {lagrangian_value:.2e} | mVaR: {mVaR:.2e} | Unity_Viol: {unity_violation:.2e} | Positivity_Viol: {max_ineq_violation:.2e}]")
                pbar.update(1)
                
                counter += 1

                if counter >= self.max_outer_iterations:
                    raise RuntimeError("FIRE Optimizer did not converge within the maximum number of outer iterations.")
        
        mVaR = compute_modified_VaR(w, self.m1_flat, self.m2_flat, self.m3_flat, self.m4_flat, z_alpha)
        if self.verbose:
            print(f"Optimization converged in {counter} outer iterations. Minimal mVaR: {mVaR}")
        return w
        
    def _update_lagrange_multipliers(self, w, rho1, lam1, rho2, lams2):
        """
        Updates Lagrange multipliers based on current weights.
        See https://epubs.siam.org/doi/10.1137/060654797 for details.
        Input: w (array), rho1 (float), lam1 (float), rho2 (float), lams2 (array)
        """
        updated_lam1 = lam1 + rho1 * (np.sum(w) - 1.0)
        updated_lams2 = np.maximum(0.0, lams2 + rho2 * (-w)) if self.long_only else lams2
        
        return updated_lam1, updated_lams2
    
    def _check_feasibility(self, w, lams2):
        """
        Checks if the solution is feasible and complementary slackness conditions are met (within tolerance).
        See https://epubs.siam.org/doi/10.1137/060654797 for details.
        Input: w (array)
        Output: tuple (unity_violation (float), positivity_violation (float))
        """
        unity_violation = abs(np.sum(w) - 1.0)

        if not self.long_only:
            return unity_violation < self.c_tol
        
        sigma_ineq = np.maximum(-w, -lams2 / self.rho2)

        max_ineq_violation = np.max(np.abs(sigma_ineq))

        return (unity_violation < self.c_tol) and (max_ineq_violation < self.c_tol)