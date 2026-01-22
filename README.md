# Anharmonic Portfolio Optimization: The FIRE2 Engine

This project applies the FIRE2 (Fast Inertial Relaxation Engine 2) algorithm—conventionally used in computational physics for atomic structure relaxation—to the problem of non-convex portfolio optimization.

While I initially built this to escape local minima in "rugged" landscapes, I discovered that at the limit of 1% VaR, the market becomes relatively "predictable." Because assets tend to crash together, the problem collapses from 45 variables to effectively 5, making the surface nearly convex. Standard solvers like SLSQP perform just as well as FIRE2 in this regime and are 1000x faster, but the high-quality gradient engine developed here remains the "top notch" backbone for the optimization. As most of the development time involved writing increasingly optimized code for the gradient, I am still quite pleased with the outcome.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Numba](https://img.shields.io/badge/HPC-Numba-green)

## 1. The Core Problem
Modern Portfolio Theory often assumes Gaussian returns, making risk minimization a simple convex problem (Variance). However, real markets exhibit Skewness and Kurtosis.
When optimizing for Modified Value-at-Risk (mVaR) using the Cornish-Fisher expansion, the objective function includes cubic and quartic terms. My assumption was that this creates a non-convex surface where greedy solvers like scipy.optimize.SLSQP frequently get trapped in suboptimal local minima. However, the systemic nature of market crashes simplifies the landscape, allowing gradient-based methods to converge efficiently - provided the gradients themselves are precise.

## 2. Background & Motivation
During my PhD in Computational Biophysics, I built continuum elastic models to simulate biological membranes. These systems find equilibrium by minimizing complex bending energy functionals, resulting sometimes in rugged landscapes.

I noticed a direct mathematical parallel between this physics problem and quantitative risk management:

* Membrane Physics: Minimizes Elastic Energy subject to Geometric Constraints (e.g., fixed area or volume).
* Portfolio Optimization: Minimize Modified Value-at-Risk (mVaR) subject to Constraints (e.g., capital allocation).
  
This project ports the relaxation algorithms I used for biophysics directly into a financial context, testing whether molecular dynamics tools can outperform standard financial solvers on their own turf.

## 3. Technical Architecture

### A. The Solver: FIRE (Fast Inertial Relaxation Engine)
I implemented a custom solver based on Newtonian dynamics. The system has "mass" and "velocity."

* **Inertia:** Momentum helps the solver "fly over" small ripples in the objective function that would trap a standard gradient descent algorithm.
* **Adaptive Time-Step:** The engine dynamically adjusts Δt. It accelerates when moving downhill and instantly freezes/resets velocity when moving against the gradient.

### B. The Constraints: Augmented Lagrangian
Since FIRE is an unconstrained optimization method, I wrap it in an Augmented Lagrangian framework to enforce:

* Unity Constraint: $\sum w_i = 1$
* (optional) Long-only Constraint: $w_i \ge 0$

### C. Performance (Numba)
The $O(N^3)$ skewness and $O(N^4)$ kurtosis calculations are the primary bottlenecks. To handle this:

* **No Python Loops:** All core numerical routines are moved into `@njit` functions. This compiles the code to machine instructions, removing Python's execution overhead.
* **Vectorization:** I structured the tensor contractions to be "vector-friendly." This allows the CPU to use SIMD instructions to process multiple data points at once, which is necessary to keep the gradient calculation fast enough for an iterative solver.

## 4. Observations & Benchmarking

compared the converged portfolios of multiple (N>1000) runs from random initial portfolios for a set of 45 large firms from the S&P 500, using returns from the volatile 2007–2009 period.

Key Findings:

* Both SLSQP and FIRE2 converged to the same results.
* SLSQP is significantly faster (~$x1000$) for this specific use case because the 1% VaR landscape is effectively convex.

## 5. Installation & Usage

Run the following commands in a terminal:

```bash
# Clone the repository
git clone git@github.com:infinityScha/inertial-risk-optimizer.git
cd inertial-risk-optimizer

# Install dependencies
pip install -r requirements.txt
```

Use inside a python script:

```python
from src.fire import FireOptimizer
from src.data import get_snp500_top

# 1. Generate heavy-tailed data
returns, _, _ = get_blue_chip_returns(years=15, n_assets=50)

# 2. Initialize Solver
solver = FireOptimizer(returns)

# 3. Optimize for 99% mVaR
weights = solver.optimize(confidence=0.99)
```

## 6. Disclaimer
*This project is for research and educational purposes only. It does not constitute investment advice. The "Curse of Dimensionality" implies that applying* $N^4$ *tensor optimization to real-world daily data requires strict regularization techniques not covered in this prototype.*
