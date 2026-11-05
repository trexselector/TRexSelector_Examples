# ==============================================================================
# demo_mlm_pca_01.py
# ==============================================================================
#
# Demonstration of PCA usage.
#
# Mirrors cpp/ml_methods/pca/demo_mlm_pca_01/demo_mlm_pca_01.cpp.
#
# Three demos which illustrate the PCA interface:
#
#   Demo 1 | Basic fit     : n = 500, p = 200, M = 10 — score matrix Z,
#                            loading matrix V, and explained variance per
#                            retained component.
#
#   Demo 2 | Restore       : n = 300, p = 100, M = 5 — fit() preprocesses X
#                            in place; restore() must return it to its
#                            original state.
#
#   Demo 3 | Transform     : n_train = 400, n_test = 50, p = 80, M = 8 —
#                            project unseen observations onto the learned
#                            components and verify transform() reproduces the
#                            fit scores on the training data.
#
# Notes:
#   - PCA.fit(X, M) preprocesses X in place (centering / normalization), so X
#     must be a writeable, Fortran-ordered float64 array; pass a copy if the
#     original values are needed afterwards. PCA.restore() undoes the in-place
#     preprocessing of the fitted matrix (the cpp API passes the map
#     explicitly: pca.restore(X_map)).
#
#   - The C++ demo seeds std::mt19937; here numpy's default_rng is seeded with
#     the same values. The RNG streams differ, so the printed numbers match
#     the C++ output qualitatively, not bitwise.
#
# ==============================================================================

import numpy as np

from trexselector.ml_methods import PCA


def print_section_header(title):
    print("=" * 78)
    print(f" {title}")
    print("=" * 78 + "\n")


# ==============================================================================
# Demo 1: Basic PCA Fit
# ==============================================================================

def demo_pca_fit():
    print_section_header("PCA Basic Fit")

    # Step 1. Generate random data
    n, p, M = 500, 200, 10
    print(f"Matrix size  : {n} x {p}")
    print(f"Components M : {M}\n")

    rng = np.random.default_rng(42)
    # PCA operates on the matrix in place through Eigen and therefore requires
    # a writeable, Fortran-ordered (column-major) float64 array.
    X = np.asfortranarray(rng.standard_normal((n, p)))

    # Step 2. Construct and fit
    pca = PCA(center=True, normalize=True)
    res = pca.fit(X, M)

    # Step 3. Print result dimensions. Z holds the principal-component scores:
    #         each row is an observation expressed in the M-dimensional
    #         component space. V holds the loadings: each column is a
    #         unit-length direction in the original p-dimensional space.
    print(f"Z dimensions : {res.Z.shape[0]} x {res.Z.shape[1]}")
    print(f"V dimensions : {res.V.shape[0]} x {res.V.shape[1]}")
    print(f"explained_variance size: {res.explained_variance.size}\n")

    # Step 4. Explained variance as a share of the total variance of the
    #         preprocessed X (fit() centered and scaled X in place)
    ev = res.explained_variance
    total_var = float(np.sum(X * X)) / (n - 1)
    cumulative = 0.0
    print("Explained variance per component (% of total):")
    for k in range(M):
        pct = 100.0 * ev[k] / total_var
        cumulative += pct
        print(f"  PC{k + 1}: {pct:.2f}%")
    print(f"  Cumulative ({M} of {p}): {cumulative:.2f}%")

    # Restore X to its original (pre-preprocessing) state
    pca.restore()
    print("\n")


# ==============================================================================
# Demo 2: Restore Round-Trip Accuracy: Data -> PCA -> Restore -> Data
# ==============================================================================

def demo_pca_restore():
    print_section_header("PCA Restore Round-Trip Accuracy")

    # Step 1. Generate random data
    n, p, M = 300, 100, 5
    print(f"Matrix size  : {n} x {p}")
    print(f"Components M : {M}\n")

    rng = np.random.default_rng(123)
    X_orig = np.asfortranarray(rng.standard_normal((n, p)))

    # Step 2. Work on a copy so we can compare to original
    X = X_orig.copy(order="F")

    pca = PCA()
    pca.fit(X, M)

    # Step 3. Restore and measure reconstruction error. fit() has centered /
    #         normalized X in place; restore() undoes that. The point is
    #         interface correctness (a round-trip on the preprocessed training
    #         matrix), not low-rank approximation error.
    pca.restore()

    max_error = np.max(np.abs(X - X_orig))
    mean_error = np.mean(np.abs(X - X_orig))

    print("Reconstruction error after restore():")
    print(f"  Max Error  : {max_error:.4e}")
    print(f"  Mean Error : {mean_error:.4e}\n")

    if max_error < 1e-10:
        print("✓ Restore round-trip passed")
    else:
        print("✗ Restore round-trip FAILED")
    print("\n")


# ==============================================================================
# Demo 3: Transform New Data: Data -> PCA -> Transform -> Scores
# ==============================================================================

def demo_pca_transform():
    print_section_header("PCA Transform New Data")

    # Step 1. Generate train and test data
    n_train, n_test, p, M = 400, 50, 80, 8
    print(f"Train size   : {n_train} x {p}")
    print(f"Test size    : {n_test} x {p}")
    print(f"Components M : {M}\n")

    rng = np.random.default_rng(7)
    X_train = np.asfortranarray(rng.standard_normal((n_train, p)))
    # Keep a pristine copy before PCA preprocesses X_train in place
    X_train_orig = X_train.copy(order="F")
    X_test = np.asfortranarray(rng.standard_normal((n_test, p)))

    # Step 2. Fit PCA on training data
    pca = PCA()
    res = pca.fit(X_train, M)

    print(f"Train scores Z dimensions: {res.Z.shape[0]} x {res.Z.shape[1]}")

    # Step 3. Transform test data — transform() applies the training-set
    #         centering / normalization to new data and projects it onto the
    #         learned loadings; no refitting takes place.
    Z_test = pca.transform(X_test)

    print(f"Test  scores Z dimensions: {Z_test.shape[0]} x {Z_test.shape[1]}\n")

    if Z_test.shape == (n_test, M):
        print(f"✓ Transform output shape correct ({n_test} x {M})")
    else:
        print("✗ Transform output shape FAILED")

    # Step 4. Verify transform() on the original training data reproduces the
    #         fit scores
    Z_train_check = pca.transform(X_train_orig)
    max_diff = np.max(np.abs(Z_train_check - res.Z))
    print(f"Max diff (fit scores vs transform on same data): {max_diff:.4e}")

    if max_diff < 1e-10:
        print("✓ transform() reproduces fit scores")
    else:
        print("✗ transform() mismatch")

    # Restore X_train
    pca.restore()
    print("\n")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    demo_pca_fit()
    demo_pca_restore()
    demo_pca_transform()
