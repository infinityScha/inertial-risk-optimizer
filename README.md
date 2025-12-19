# Anharmonic Portfolio Optimization: The FIRE Engine

This project applies the FIRE (Fast Inertial Relaxation Engine) algorithm—conventionally used in computational physics for atomic structure relaxation—to the problem of non-convex portfolio optimization.
Standard solvers (like SLSQP) often struggle with the "rugged" error landscapes created by heavy-tailed market data. I built a custom engine that combines inertial dynamics with an Aggressive Augmented Lagrangian method. This allows the solver to escape local minima while strictly enforcing portfolio constraints, offering a robust alternative to traditional gradient-based methods.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Numba](https://img.shields.io/badge/HPC-Numba-green)

## 1. Background & Motivation
During my PhD in Computational Biophysics, I built elastic models to simulate biological membranes. These systems minimize complex bending energy functionals, resulting in rugged landscapes filled with metastable states (local minima).
I noticed a direct mathematical parallel between this and quantitative risk management:

* Membranes minimize elastic energy subject to geometric constraints.
* Portfolios minimize Tail Risk (mVaR) subject to capital constraints.

This project ports the relaxation algorithms I used for biophysics directly into a financial context, testing whether molecular dynamics tools can outperform standard financial solvers on their own turf.

## 2. The Core Problem
Modern Portfolio Theory often assumes Gaussian returns, making risk minimization a simple convex problem (Variance). However, real markets exhibit Skewness and Kurtosis.
When optimizing for Modified Value-at-Risk (mVaR) using the Cornish-Fisher expansion, the objective function includes cubic and quartic terms. This creates a non-convex surface where greedy solvers like `scipy.optimize.SLSQP` frequently get trapped in suboptimal local minima.

## 3. Technical Architecture

### A. The Solver: FIRE (Fast Inertial Relaxation Engine)
I implemented a custom solver based on Newtonian dynamics. The system has "mass" and "velocity."

* **Inertia:** Momentum helps the solver "fly over" small ripples in the objective function that would trap a standard gradient descent algorithm.
* **Adaptive Time-Step:** The engine dynamically adjusts Δt. It accelerates when moving downhill and instantly freezes/resets velocity when moving against the gradient.

### B. The Constraints: Aggressive Augmented Lagrangian
Since FIRE is an unconstrained optimization method, I wrap it in a custom Augmented Lagrangian framework to enforce:

* Unity Constraint: $\sum w_i = 1$
* (optional) Long-only Constraint: $w_i \ge 0$
* Risk Constraint (Modified VaR via Cornish-Fisher):
  $$mVaR_\alpha(w) = - \left( \mu_p + \sigma_p \left[ z_\alpha + \frac{1}{6}(z_\alpha^2 - 1)S_p + \frac{1}{24}(z_\alpha^3 - 3z_\alpha)(K - 3) - \frac{1}{36}(2z_\alpha^3 - 5z_\alpha)S_p^2 \right] \right)$$

  **Where:**
    * $\mu_p, \sigma_p$ -  Portfolio mean and volatility.
    * $z_\alpha$ - Critical value from the normal distribution for confidence $\alpha$.
    * $S_p$ - Skewness
    * $K$ - Kurtosis
**Why "Aggressive"?**
Standard Lagrangians increase penalties slowly. I implemented a schedule that rapidly ramps up the penalty parameters (ρ) and Lagrange multipliers (λ) when constraint violation stalls. This forces the inertial engine to "crash" into the valid region quickly, prioritizing feasibility without losing the kinetic energy needed to find global optima.

### C. Performance (Numba)
To make this computationally viable, I avoided Python loops entirely.

* **JIT Compilation:** All numerical routines are compiled using `numba.jit(nopython=True)`.
* **Tensor Calculus:** The $O(N^3)$ Skewness and $O(N^4)$ Kurtosis tensor contractions are parallelized, achieving performance comparable to C++.

## 4. Benchmarking Strategy
TBA

## 5. Installation & Usage

Run the following commands in a terminal:

```bash
# Clone the repository
git clone [https://github.com/yourusername/anharmonic-portfolio.git](https://github.com/yourusername/anharmonic-portfolio.git)
cd anharmonic-portfolio

# Install dependencies
pip install -r requirements.txt
```

Use inside a python script:

```python
from src.fire import FireOptimizer
from src.data import generate_nasty_data

# 1. Generate heavy-tailed data
returns, _, _ = generate_nasty_data(n_assets=50, nu=3.0)

# 2. Initialize Solver
solver = FireOptimizer(returns)

# 3. Optimize for 99% mVaR
weights = solver.optimize(target='mVaR', confidence=0.99)
```

## 6. Disclaimer
*This project is for research and educational purposes only. It does not constitute investment advice. The "Curse of Dimensionality" implies that applying* $N^4$ *tensor optimization to real-world daily data requires strict regularization techniques not covered in this prototype.*
