# ==============================================================================
# demo_mlm_ms_01_ridge_cv_svd.py
# ==============================================================================
#
# Demo of Ridge Regression K-Fold Cross-Validation via the SVD-based RidgeCV
# (per-fold centering and column L2-normalization).
#
# Mirrors cpp/ml_methods/model_selection/demo_mlm_ms_01_ridge_cv_svd/
#         demo_mlm_ms_01_ridge_cv_svd.cpp.
#
# Scenarios:
#
#   Demo 1 | Low-dimensional  : n=1000, p=500  (n > p,  10-fold)
#   Demo 2 | High-dimensional : n=300,  p=500  (p > n,  10-fold)
#   Demo 3 | Very high-dim    : n=300,  p=1000 (p >> n,  5-fold)
#   Demo 4 | Memory-mapped    : n=5000, p=500  — dataset lives on disk; fit()
#                               reads it through the zero-copy to_numpy() view
#                               of a MemoryMappedMatrix; grid widened to a 1e5
#                               range (n >> p). (cpp re-opens the file
#                               read-only via ConstMapType; the Python
#                               zero-copy view requires ReadWrite mode.)
#
# Notes:
#   - Same n / p / fold values and data seeds as the C++ and R demos; NumPy's
#     PCG64 stream differs from std::mt19937 and from R's Mersenne-Twister,
#     so results match qualitatively, not bitwise.
#
#   - RidgeCV requires float64, Fortran-ordered (column-major) arrays;
#     np.asfortranarray() provides the required layout.
#
# ==============================================================================

import os
import tempfile

import numpy as np

from trexselector.ml_methods import RidgeCV
from trexselector.utils import AccessMode, MemoryMappedMatrix, numpy_to_memmap


def print_section_header(title):
    print("=" * 78)
    print(f" {title}")
    print("=" * 78 + "\n")


# ==============================================================================
# Shared data-generation helper
# ==============================================================================

def generate_regression_data(n, p, sigma, seed, n_active=10):
    """Synthetic regression data with known sparse signal.

    True coefficients: n_active entries equal to 1.0, at evenly spaced column
    indices. Observations: y = X * beta_true + N(0, sigma^2). Mirrors the C++
    generate_regression_data() helper.
    """
    rng = np.random.default_rng(seed)
    X = np.asfortranarray(rng.standard_normal((n, p)))
    beta_true = np.zeros(p)
    stride = max(1, p // n_active)
    beta_true[np.arange(n_active) * stride] = 1.0
    y = X @ beta_true + sigma * rng.standard_normal(n)
    return X, y, beta_true


# ==============================================================================
# Shared print helper
# ==============================================================================

def print_cv_summary(cv, step_size=10):
    """Condensed summary of K-fold CV results from a RidgeCV fit:
      - lambda.min: grid value minimizing the CV mean-squared error
      - lambda.1se: largest lambda whose CV-MSE is within one standard error
        of the minimum (more parsimonious / more heavily regularized choice)
    """
    lambdas = np.asarray(cv.lambdas())
    mse = np.asarray(cv.cv_mse())
    sem = np.asarray(cv.cv_sem())
    K = len(lambdas)
    i_min = cv.index_min()
    i_1se = cv.index_1se()

    print(f"  lambda.min : {cv.cv_min():.4f}  "
          f"(CV-MSE={mse[i_min]:.4f} +/- {sem[i_min]:.4f})")
    print(f"  lambda.1se : {cv.cv_1se():.4f}  "
          f"(CV-MSE={mse[i_1se]:.4f} +/- {sem[i_1se]:.4f})\n")

    print(f"  CV-MSE curve (every {step_size} steps):")
    print(f"    {'lambda':<14}{'CV-MSE':<12}{'SEM':<12}marker")

    def print_row(k, marker):
        print(f"    {lambdas[k]:<14.4f}{mse[k]:<12.4f}{sem[k]:<12.4f}{marker}")

    shown = range(0, K, step_size)
    for k in shown:
        marker = "  <-- min" if k == i_min else "  <-- 1se" if k == i_1se else ""
        print_row(k, marker)
    if i_min not in shown:
        print_row(i_min, "  <-- min")
    if i_1se not in shown and i_1se != i_min:
        print_row(i_1se, "  <-- 1se")
    print()


def fit_and_report(X, y, n_folds, seed, lambda_ratio=1000.0):
    """Fit RidgeCV, print the summary, and run the two checks.

    The grid mirrors the C++ fit(): 100 lambdas anchored at the largest useful
    penalty, spanning [lambda_max / lambda_ratio, lambda_max]; `seed` fixes the
    fold assignment.
    """
    cv = RidgeCV()
    cv.fit(X, y, n_folds=n_folds, n_lambda=100, lambda_ratio=lambda_ratio, seed=seed)

    print_cv_summary(cv, 10)

    order_ok = cv.cv_min() <= cv.cv_1se()
    print(f"  Ordering check (lambda.min <= lambda.1se): "
          f"{'PASS' if order_ok else 'FAIL'}")
    # A pick on the first or last grid point is the grid boundary, not a minimum
    print(f"  Grid-edge check (both picks inside the grid): "
          f"{'FAIL' if cv.selection_at_grid_edge() else 'PASS'}\n")
    return cv


# ==============================================================================
# Demo 1: Low-dimensional scenario (n=1000, p=500, n > p, 10-fold)
# ==============================================================================

def demo_kfold_low_dim():
    print_section_header(
        "Demo MS-01 | ridge_cv_svd | Low-Dimensional (n=1000, p=500, 10-fold)")

    n, p, sigma, n_folds = 1_000, 500, 1.0, 10

    X, y, beta_true = generate_regression_data(n, p, sigma, seed=111)

    print(f"Data: n={n}, p={p}, folds={n_folds}, "
          f"active coefs={int(np.sum(beta_true != 0))}\n")

    fit_and_report(X, y, n_folds, seed=111)


# ==============================================================================
# Demo 2: High-dimensional scenario (n=300, p=500, p > n, 10-fold)
# ==============================================================================

def demo_kfold_high_dim():
    print_section_header(
        "Demo MS-01 | ridge_cv_svd | High-Dimensional (n=300, p=500, 10-fold)")

    n, p, sigma, n_folds = 300, 500, 1.0, 10

    X, y, beta_true = generate_regression_data(n, p, sigma, seed=222)

    print(f"Data: n={n}, p={p}, folds={n_folds}, "
          f"active coefs={int(np.sum(beta_true != 0))}\n")

    fit_and_report(X, y, n_folds, seed=222)


# ==============================================================================
# Demo 3: Very high-dimensional scenario (n=300, p=1000, p >> n, 5-fold)
# ==============================================================================

def demo_kfold_very_high_dim():
    print_section_header(
        "Demo MS-01 | ridge_cv_svd | Very High-Dimensional (n=300, p=1000, 5-fold)")

    n, p, sigma, n_folds = 300, 1_000, 1.0, 5

    X, y, beta_true = generate_regression_data(n, p, sigma, seed=333)

    print(f"Data: n={n}, p={p}, folds={n_folds}, "
          f"active coefs={int(np.sum(beta_true != 0))}")
    print("  Note: 5 folds used to ensure each fold's training set is large "
          "enough relative to n.\n")

    fit_and_report(X, y, n_folds, seed=333)


# ==============================================================================
# Demo 4: Memory-mapped scenario (n=5000, p=500, 10-fold)
# ==============================================================================

def demo_kfold_mmap():
    """K-fold CV when the design matrix is backed by a mmap file.

    RidgeCV.fit() accepts any Fortran-ordered float64 array, so the zero-copy
    to_numpy() view of a MemoryMappedMatrix binds without copying the full
    matrix. Internally each fold materializes dense X_train / X_test
    sub-matrices in RAM, so there is no out-of-core streaming path.
    """
    print_section_header(
        "Demo MS-01 | ridge_cv_svd | Memory-Mapped Dataset (n=5000, p=500, 10-fold)")

    n, p, sigma, n_folds = 5_000, 500, 1.0, 10
    # n >> p: the CV optimum lies below lambda_max / 1000, so the grid is widened
    lambda_ratio = 1e5

    fd, mmap_path = tempfile.mkstemp(prefix="trex_mlm_demo_ms01_svd_mmap_",
                                     suffix=".bin")
    os.close(fd)

    X_dense, y, beta_true = generate_regression_data(n, p, sigma, seed=444)

    file_bytes = n * p * 8
    print(f"Writing X to mmap file: <tmp>/{os.path.basename(mmap_path)}")
    print(f"  File size: {file_bytes / (1 << 20):.2f} MB")
    print(f"  Active coefs: {int(np.sum(beta_true != 0))}\n")

    try:
        mmap_w = numpy_to_memmap(mmap_path, X_dense)
        del mmap_w  # flush + unmap the writer before re-opening

        # cpp re-opens the file ReadOnly and binds a ConstMapType; the Python
        # zero-copy view requires ReadWrite mode, so ReadWrite is used here.
        print(f"Re-opening mmap and fitting ridge CV ({n_folds}-fold)...\n")
        mmap_r = MemoryMappedMatrix(mmap_path, n, p, AccessMode.ReadWrite)
        X_ro = mmap_r.to_numpy()

        fit_and_report(X_ro, y, n_folds, seed=444, lambda_ratio=lambda_ratio)
        del X_ro, mmap_r  # unmap before the file is removed
    finally:
        try:
            os.unlink(mmap_path)
            print(f"Cleaned up mmap file: <tmp>/{os.path.basename(mmap_path)}")
        except OSError as e:
            print(f"[WARN] Could not remove mmap file: {e}")
    print()


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print_section_header(
        "ridge_cv_svd Demo Suite  (demo_mlm_ms_01_ridge_cv_svd)")

    demo_kfold_low_dim()
    demo_kfold_high_dim()
    demo_kfold_very_high_dim()
    demo_kfold_mmap()

    print("\nAll ridge_cv_svd demos complete.\n")
