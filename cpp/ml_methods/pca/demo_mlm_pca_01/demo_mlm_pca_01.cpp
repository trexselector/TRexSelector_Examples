// ===================================================================================
// demo_mlm_pca_01.cpp
// ===================================================================================
/**
 * @file demo_mlm_pca_01.cpp
 *
 * @brief Demonstration of PCA usage.
 *
 * Organisation:
 * -------------
 * |-- ml_methods/
 *     |-- pca/
 *         |-- demo_mlm_pca_01.cpp
 */
// ===================================================================================

// std includes
#include <iomanip>
#include <iostream>
#include <random>

// Eigen includes
#include <Eigen/Dense>

// TRex Selector includes
#include <ml_methods/pca/pca.hpp>
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>

// ============================================================================
// Namespace aliases
// ============================================================================

namespace pca_ns      = trex::ml_methods::pca;
namespace cdiagnostics = trex::utils::eval::cdiagnostics;


// ============================================================================
// Demo 1: Basic PCA Fit
// ============================================================================

void demo_pca_fit() {
    cdiagnostics::print_section_header("PCA Basic Fit");

    // Step 1. Generate random data
    const Eigen::Index n = 500, p = 200, M = 10;
    std::cout << "Matrix size  : " << n << " x " << p << "\n";
    std::cout << "Components M : " << M << "\n\n";

    std::mt19937 rng(42);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    Eigen::MatrixXd X(n, p);
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < p; ++j) {
            X(i, j) = norm_dist(rng);
        }
    }

    // Step 2. Construct and fit
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), n, p);
    pca_ns::PCA pca(X_map, M);
    pca_ns::PCAResult res = pca.fit();

    // Step 3. Print result dimensions
    std::cout << "Z dimensions : " << res.Z.rows() << " x " << res.Z.cols() << "\n";
    std::cout << "V dimensions : " << res.V.rows() << " x " << res.V.cols() << "\n";
    std::cout << "explained_variance size: " << res.explained_variance.size() << "\n\n";

    // Step 4. Explained variance as a share of the total variance of the preprocessed X
    const double total_var = X_map.squaredNorm() / static_cast<double>(n - 1);
    double cumulative = 0.0;
    std::cout << "Explained variance per component (% of total):\n"
              << std::fixed << std::setprecision(2);
    for (Eigen::Index k = 0; k < M; ++k) {
        const double pct = 100.0 * res.explained_variance(k) / total_var;
        cumulative += pct;
        std::cout << "  PC" << (k + 1) << ": " << pct << "%\n";
    }
    std::cout << "  Cumulative (" << M << " of " << p << "): " << cumulative << "%\n";

    // Restore X
    pca.restore(X_map);
    std::cout << "\n\n";
}


// ============================================================================
// Demo 2: Restore Round-Trip Accuracy: Data -> PCA -> Restore -> Data
// ============================================================================

void demo_pca_restore() {
    cdiagnostics::print_section_header("PCA Restore Round-Trip Accuracy");

    // Step 1. Generate random data
    const Eigen::Index n = 300, p = 100, M = 5;
    std::cout << "Matrix size  : " << n << " x " << p << "\n";
    std::cout << "Components M : " << M << "\n\n";

    std::mt19937 rng(123);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    Eigen::MatrixXd X_orig(n, p);
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < p; ++j) {
            X_orig(i, j) = norm_dist(rng);
        }
    }

    // Step 2. Work on a copy so we can compare to original
    Eigen::MatrixXd X = X_orig;
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), n, p);

    pca_ns::PCA pca(X_map, M);
    pca.fit();

    // Step 3. Restore and measure reconstruction error
    pca.restore(X_map);

    double max_error  = (X - X_orig).cwiseAbs().maxCoeff();
    double mean_error = (X - X_orig).cwiseAbs().mean();

    std::cout << "Reconstruction error after restore():\n"
              << std::scientific << std::setprecision(4)
              << "  Max Error  : " << max_error  << "\n"
              << "  Mean Error : " << mean_error << "\n\n";

    if (max_error < 1e-10) {
        std::cout << "✓ Restore round-trip passed\n";
    } else {
        std::cout << "✗ Restore round-trip FAILED\n";
    }

    std::cout << "\n\n";
}


// ============================================================================
// Demo 3: Transform New Data: Data -> PCA -> Transform -> Scores
// ============================================================================

void demo_pca_transform() {
    cdiagnostics::print_section_header("PCA Transform New Data");

    // Step 1. Generate train and test data
    const Eigen::Index n_train = 400, n_test = 50, p = 80, M = 8;
    std::cout << "Train size   : " << n_train << " x " << p << "\n";
    std::cout << "Test size    : " << n_test  << " x " << p << "\n";
    std::cout << "Components M : " << M << "\n\n";

    std::mt19937 rng(7);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    Eigen::MatrixXd X_train(n_train, p);
    for (Eigen::Index i = 0; i < n_train; ++i) {
        for (Eigen::Index j = 0; j < p; ++j) {
            X_train(i, j) = norm_dist(rng);
        }
    }
    // Keep a pristine copy before PCA preprocesses X_train in-place
    const Eigen::MatrixXd X_train_orig = X_train;

    Eigen::MatrixXd X_test(n_test, p);
    for (Eigen::Index i = 0; i < n_test; ++i) {
        for (Eigen::Index j = 0; j < p; ++j) {
            X_test(i, j) = norm_dist(rng);
        }
    }

    // Step 2. Fit PCA on training data
    Eigen::Map<Eigen::MatrixXd> X_train_map(X_train.data(), n_train, p);
    pca_ns::PCA pca(X_train_map, M);
    pca_ns::PCAResult res = pca.fit();

    std::cout << "Train scores Z dimensions: " << res.Z.rows() << " x " << res.Z.cols() << "\n";

    // Step 3. Transform test data
    Eigen::MatrixXd Z_test = pca.transform(X_test);

    std::cout << "Test  scores Z dimensions: " << Z_test.rows() << " x " << Z_test.cols() << "\n\n";

    if (Z_test.rows() == n_test && Z_test.cols() == M) {
        std::cout << "✓ Transform output shape correct (" << n_test << " x " << M << ")\n";
    } else {
        std::cout << "✗ Transform output shape FAILED\n";
    }

    // Step 4. Verify transform() on the original training data reproduces fit scores
    Eigen::MatrixXd Z_train_check = pca.transform(X_train_orig);
    double max_diff = (Z_train_check - res.Z).cwiseAbs().maxCoeff();
    std::cout << "Max diff (fit scores vs transform on same data): "
              << std::scientific << std::setprecision(4) << max_diff << "\n";

    if (max_diff < 1e-10) {
        std::cout << "✓ transform() reproduces fit scores\n";
    } else {
        std::cout << "✗ transform() mismatch\n";
    }

    // Restore X_train
    pca.restore(X_train_map);

    std::cout << "\n\n";
}


// ============================================================================
// main
// ============================================================================

int main() {

    demo_pca_fit();
    demo_pca_restore();
    demo_pca_transform();

    return 0;
}
