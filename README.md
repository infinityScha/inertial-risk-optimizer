# Anharmonic Portfolio Optimization: The FIRE2 Engine

This project applies the FIRE2 (Fast Inertial Relaxation Engine 2) algorithm—conventionally used in computational physics for atomic structure relaxation—to the problem of non-convex portfolio optimization.
Standard solvers (like SLSQP) often struggle with the "rugged" error landscapes created by heavy-tailed market data. I built a custom engine that combines inertial dynamics with an Aggressive Augmented Lagrangian method. This allows the solver to escape local minima while strictly enforcing portfolio constraints, offering a better alternative to traditional gradient-based methods.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Numba](https://img.shields.io/badge/HPC-Numba-green)

## 1. The Core Problem
Modern Portfolio Theory often assumes Gaussian returns, making risk minimization a simple convex problem (Variance). However, real markets exhibit Skewness and Kurtosis.
When optimizing for Modified Value-at-Risk (mVaR) using the Cornish-Fisher expansion, the objective function includes cubic and quartic terms. This creates a non-convex surface where greedy solvers like `scipy.optimize.SLSQP` frequently get trapped in suboptimal local minima.

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

### B. The Constraints: Aggressive Augmented Lagrangian
Since FIRE is an unconstrained optimization method, I wrap it in a custom Augmented Lagrangian framework to enforce:

* Unity Constraint: $\sum w_i = 1$
* (optional) Long-only Constraint: $w_i \ge 0$

**Why "Aggressive"?**
Standard methods slowly ramp up penalties. I start with high stiffness to force immediate feasibility ("crash" into the valid region), then relax the penalty parameter (ρ) while refining Lagrange multipliers (λ). I relax the penalty stiffness to stop the solver from wasting time ping-pong between tight constraint walls, allowing it to flow smoothly along the valid path toward the optimum.

### C. Performance (Numba)
The $O(N^3)$ skewness and $O(N^4)$ kurtosis calculations are the primary bottlenecks. To handle this:

* **No Python Loops:** All core numerical routines are moved into `@njit` functions. This compiles the code to machine instructions, removing Python's execution overhead.
* **Vectorization:** I structured the tensor contractions to be "vector-friendly." This allows the CPU to use SIMD instructions to process multiple data points at once, which is necessary to keep the gradient calculation fast enough for an iterative solver.

## 4. Benchmarking Strategy
TBA

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
from src.data import generate_nasty_data

# 1. Generate heavy-tailed data
returns, _, _ = generate_nasty_data(n_assets=50, nu=3.0)

# 2. Initialize Solver
solver = FireOptimizer(returns)

# 3. Optimize for 99% mVaR
weights = solver.optimize(confidence=0.99)
```

## 6. Disclaimer
*This project is for research and educational purposes only. It does not constitute investment advice. The "Curse of Dimensionality" implies that applying* $N^4$ *tensor optimization to real-world daily data requires strict regularization techniques not covered in this prototype.*
