// ===================================================================================
// demo_mlm_svd_01.cpp
// ===================================================================================
/**
 * @file demo_mlm_svd_01.cpp
 *
 * @brief Demonstration of SVDSolver usage.
 * Organisation:
 * -------------
 * |-- ml_methods/
 *     |-- svd/
 *         |-- demo_mlm_svd_01.cpp
 */
// ===================================================================================

// std includes
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

// Eigen includes
#include <Eigen/Dense>

// TRex Selector includes
#include <ml_methods/svd/svd.hpp>
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>

// ============================================================================
// Namespace aliases
// ============================================================================

namespace svd_ns       = trex::ml_methods::svd;
namespace cdiagnostics = trex::utils::eval::cdiagnostics;


// ============================================================================
// Shared check helper
// ============================================================================

/**
 * @brief Compare the truncation error with the Eckart-Young optimum.
 *
 * @details The best rank-M approximation in the Frobenius norm leaves exactly the
 *          energy of the discarded singular values; a full Eigen SVD provides them.
 *
 * @param X         Input matrix.
 * @param M         Retained rank.
 * @param rel_error Relative Frobenius error ||X_M - X||_F / ||X||_F of the solver.
 */
void print_eckart_young_check(const Eigen::MatrixXd& X, Eigen::Index M, double rel_error) {
    const Eigen::VectorXd s_all = Eigen::JacobiSVD<Eigen::MatrixXd>(X).singularValues();
    const double optimum = std::sqrt(s_all.tail(s_all.size() - M).squaredNorm()) / X.norm();
    const double diff    = std::abs(rel_error - optimum);

    std::cout << std::scientific << std::setprecision(4)
              << "Eckart-Young optimum sqrt(sum_{k>M} s_k^2) / ||X||_F  : " << optimum << "\n"
              << "Eckart-Young check (|difference| = " << diff << " < 1e-10): "
              << (diff < 1e-10 ? "PASS" : "FAIL") << "\n";
}


// ============================================================================
// Demo 1: Direct / Thin-Jacobi Path (n >= p)
// ============================================================================

void demo_svd_direct() {
    cdiagnostics::print_section_header("SVDSolver - Direct Path (n >= p)");

    // Step 1. Generate random data
    const Eigen::Index n = 300, p = 80, M = 10;
    std::cout << "Matrix size : " << n << " x " << p << "\n";
    std::cout << "M           : " << M << "\n\n";

    std::mt19937 rng(42);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    Eigen::MatrixXd X(n, p);
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < p; ++j) {
            X(i, j) = norm_dist(rng);
        }
    }

    // Step 2. Compute SVD
    svd_ns::SVDSolver solver;
    svd_ns::SVDResult res = solver.compute(X, M);

    // Step 3. Print dimensions
    std::cout << "U dimensions : " << res.U.rows() << " x " << res.U.cols() << "\n";
    std::cout << "S size       : " << res.S.size() << "\n";
    std::cout << "V dimensions : " << res.V.rows() << " x " << res.V.cols() << "\n\n";

    // Step 4. Reconstruction error: ||X_approx - X||_F / ||X||_F
    Eigen::MatrixXd X_approx = res.U * res.S.asDiagonal() * res.V.transpose();
    double rel_error = (X_approx - X).norm() / X.norm();
    std::cout << "Relative reconstruction error ||X_M - X||_F / ||X||_F : "
              << std::scientific << std::setprecision(4) << rel_error << "\n";
    print_eckart_young_check(X, M, rel_error);

    // Step 5. Singular values
    std::cout << "\nTop " << M << " singular values:\n";
    for (Eigen::Index k = 0; k < M; ++k) {
        std::cout << "  s" << (k + 1) << " = " << std::fixed << std::setprecision(4)
                  << res.S(k) << "\n";
    }
    std::cout << "\n\n";
}


// ============================================================================
// Demo 2: Gram / Kernel Path (p > 2*n)
// ============================================================================

void demo_svd_gram() {
    cdiagnostics::print_section_header("SVDSolver - Gram Path (p > 2*n)");

    // Step 1. Generate wide matrix
    const Eigen::Index n = 100, p = 500, M = 8;
    std::cout << "Matrix size : " << n << " x " << p << "  (p > 2*n → Gram path)\n";
    std::cout << "M           : " << M << "\n\n";

    std::mt19937 rng(7);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    Eigen::MatrixXd X(n, p);
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < p; ++j) {
            X(i, j) = norm_dist(rng);
        }
    }

    // Step 2. Compute SVD
    svd_ns::SVDSolver solver;
    svd_ns::SVDResult res = solver.compute(X, M);

    // Step 3. Print dimensions
    std::cout << "U dimensions : " << res.U.rows() << " x " << res.U.cols() << "\n";
    std::cout << "S size       : " << res.S.size() << "\n";
    std::cout << "V dimensions : " << res.V.rows() << " x " << res.V.cols() << "\n\n";

    // Step 4. Reconstruction error
    Eigen::MatrixXd X_approx = res.U * res.S.asDiagonal() * res.V.transpose();
    double rel_error = (X_approx - X).norm() / X.norm();
    std::cout << "Relative reconstruction error ||X_M - X||_F / ||X||_F : "
              << std::scientific << std::setprecision(4) << rel_error << "\n";
    print_eckart_young_check(X, M, rel_error);
    std::cout << "\n\n";
}


// ============================================================================
// Demo 3: Orthogonality Check
// ============================================================================

void demo_svd_orthogonality() {
    cdiagnostics::print_section_header("SVDSolver - Orthogonality Check");

    const Eigen::Index n = 200, p = 150, M = 20;
    std::cout << "Matrix size : " << n << " x " << p << "\n";
    std::cout << "M           : " << M << "\n\n";

    std::mt19937 rng(99);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    Eigen::MatrixXd X(n, p);
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < p; ++j) {
            X(i, j) = norm_dist(rng);
        }
    }

    svd_ns::SVDSolver solver;
    svd_ns::SVDResult res = solver.compute(X, M);

    // ||U^T U - I||_F
    Eigen::MatrixXd UtU = res.U.transpose() * res.U;
    double UtU_error = (UtU - Eigen::MatrixXd::Identity(M, M)).norm();

    // ||V^T V - I||_F
    Eigen::MatrixXd VtV = res.V.transpose() * res.V;
    double VtV_error = (VtV - Eigen::MatrixXd::Identity(M, M)).norm();

    std::cout << std::scientific << std::setprecision(4);
    std::cout << "||U^T U - I||_F : " << UtU_error << "\n";
    std::cout << "||V^T V - I||_F : " << VtV_error << "\n\n";

    if (UtU_error < 1e-10 && VtV_error < 1e-10) {
        std::cout << "✓ U and V are orthonormal\n";
    } else {
        std::cout << "✗ Orthonormality check FAILED\n";
    }

    std::cout << "\n\n";
}


// ============================================================================
// main
// ============================================================================

int main() {
    demo_svd_direct();
    demo_svd_gram();
    demo_svd_orthogonality();
    return 0;
}
