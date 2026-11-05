# ==============================================================================
# demo_mlm_ridge_01.py
# ==============================================================================
#
# Demonstration of RidgeSolver usage.
#
# Mirrors cpp/ml_methods/ridge_regression/demo_mlm_ridge_01/demo_mlm_ridge_01.cpp.
#
# Three demos which illustrate the ridge solver:
#
#   Demo 1 | Primal path  : n = 500, p = 50, lambda = 1.0 — recover a dense
#                           coefficient vector from a known linear model.
#
#   Demo 2 | Dual path    : n = 80, p = 300, lambda = 10.0 — high-dimensional
#                           setting with a sparse ground truth; n < p routes
#                           through the n x n dual system.
#
#   Demo 3 | Lambda sweep : n = 200, p = 40 — coefficient error and residual
#                           across lambda in {1e-4, ..., 1e3}, illustrating
#                           the shrinkage bias-variance trade-off (plus a
#                           monotone-shrinkage check, a Python-side addition).
#
# Notes:
#   - RidgeSolver.solve(X, y, lambda_val) is a static method solving
#     min_beta ||y - X beta||_2^2 + lambda * ||beta||_2^2. The L2 penalty
#     shrinks coefficients toward zero: larger lambda means more shrinkage
#     (more bias, less variance) and a larger residual on the training data.
#   - The C++ demo seeds std::mt19937; here numpy's default_rng is seeded with
#     the same values. The RNG streams differ, so the printed numbers match
#     the C++ output qualitatively, not bitwise.
#
# ==============================================================================

import numpy as np

from trexselector.ml_methods import RidgeSolver


def print_section_header(title):
    print("=" * 78)
    print(f" {title}")
    print("=" * 78 + "\n")


# ==============================================================================
# Demo 1: Basic Solve (n >= p, primal path)
# ==============================================================================

def demo_ridge_primal():
    print_section_header("RidgeSolver - Primal Path (n >= p)")

    # Step 1. Generate data from a known linear model: y = X * beta_true + noise
    n, p = 500, 50
    lam = 1.0
    print(f"Matrix size : {n} x {p}")
    print(f"Lambda      : {lam:g}\n")

    rng = np.random.default_rng(42)
    # The solver accesses the matrix through Eigen and therefore requires a
    # Fortran-ordered (column-major) float64 array.
    X = np.asfortranarray(rng.standard_normal((n, p)))

    beta_true = np.ones(p)
    y = X @ beta_true + 0.1 * rng.standard_normal(n)

    # Step 2. Solve
    beta_hat = RidgeSolver.solve(X, y, lam)

    # Step 3. Report — with plenty of samples (n >> p), low noise, and mild
    #         regularization the estimate lands very close to the ground truth.
    coef_error = np.linalg.norm(beta_hat - beta_true)
    residual = np.linalg.norm(y - X @ beta_hat)
    print(f"||beta_hat - beta_true||_2 : {coef_error:.4e}")
    print(f"Residual ||y - X*beta||_2  : {residual:.4e}")
    print("\n")


# ==============================================================================
# Demo 2: Dual Path (n < p)
# ==============================================================================

def demo_ridge_dual():
    print_section_header("RidgeSolver - Dual Path (n < p)")

    # Step 1. Generate high-dimensional data
    n, p = 80, 300
    lam = 10.0
    print(f"Matrix size : {n} x {p}")
    print(f"Lambda      : {lam:g}\n")

    rng = np.random.default_rng(7)
    X = np.asfortranarray(rng.standard_normal((n, p)))

    # Sparse ground truth: only the first 10 coefficients are nonzero. With
    # n < p the least-squares problem is underdetermined; the ridge penalty
    # makes it well-posed, and the solver switches to the cheaper n x n dual
    # formulation.
    beta_true = np.zeros(p)
    beta_true[:10] = 1.0
    y = X @ beta_true + 0.1 * rng.standard_normal(n)

    # Step 2. Solve
    beta_hat = RidgeSolver.solve(X, y, lam)

    # Step 3. Report
    coef_error = np.linalg.norm(beta_hat - beta_true)
    residual = np.linalg.norm(y - X @ beta_hat)
    print(f"||beta_hat - beta_true||_2 : {coef_error:.4e}")
    print(f"Residual ||y - X*beta||_2  : {residual:.4e}")
    print("\n")


# ==============================================================================
# Demo 3: Lambda Sweep
# ==============================================================================

def demo_ridge_lambda_sweep():
    print_section_header("RidgeSolver - Lambda Sweep")

    n, p = 200, 40
    print(f"Matrix size : {n} x {p}\n")

    rng = np.random.default_rng(99)
    X = np.asfortranarray(rng.standard_normal((n, p)))

    beta_true = np.ones(p)
    y = X @ beta_true + 0.5 * rng.standard_normal(n)

    # Sweep over a range of lambda values. As lambda grows, the penalty
    # dominates the fit: coefficients shrink toward zero and the training
    # residual grows.
    lambdas = [1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]

    print(f"{'lambda':>12}{'||beta_hat - true||_2':>22}{'||residual||_2':>22}")
    print("-" * 56)

    beta_norms = []
    residuals = []
    for lam in lambdas:
        beta_hat = RidgeSolver.solve(X, y, lam)
        coef_error = np.linalg.norm(beta_hat - beta_true)
        residual = np.linalg.norm(y - X @ beta_hat)
        beta_norms.append(np.linalg.norm(beta_hat))
        residuals.append(residual)
        print(f"{lam:>12.1e}{coef_error:>22.4e}{residual:>22.4e}")

    # Python-side sanity check (not in the cpp demo): monotone shrinkage.
    assert np.all(np.diff(beta_norms) < 0), "||beta_hat||_2 must shrink as lambda grows"
    assert np.all(np.diff(residuals) > 0), "training residual must grow as lambda grows"
    print("\nShrinkage check passed: ||beta_hat||_2 decreases and the residual")
    print("increases monotonically in lambda.")
    print("\n")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    demo_ridge_primal()
    demo_ridge_dual()
    demo_ridge_lambda_sweep()
