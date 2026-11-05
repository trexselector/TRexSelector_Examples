// ===================================================================================
// demo_mlm_ridge_01.cpp
// ===================================================================================
/**
 * @file demo_mlm_ridge_01.cpp
 *
 * @brief Demonstration of RidgeSolver usage.
 *
 * Organisation:
 * -------------
 * |-- ml_methods/
 *     |-- ridge_regression/
 *         |-- demo_mlm_ridge_01.cpp
 */
// ===================================================================================

// std includes
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// TRex Selector includes
#include <ml_methods/ridge_regression/ridge.hpp>
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>

// ============================================================================
// Namespace aliases
// ============================================================================

namespace ridge        = trex::ml_methods::ridge;
namespace cdiagnostics = trex::utils::eval::cdiagnostics;


// ============================================================================
// Demo 1: Basic Solve (n >= p, primal path)
// ============================================================================

void demo_ridge_primal() {
    cdiagnostics::print_section_header("RidgeSolver - Primal Path (n >= p)");

    // Step 1. Generate data from a known linear model: y = X * beta_true + noise
    const Eigen::Index n = 500, p = 50;
    const double lambda = 1.0;
    std::cout << "Matrix size : " << n << " x " << p << "\n";
    std::cout << "Lambda      : " << std::defaultfloat << lambda << "\n\n";

    std::mt19937 rng(42);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    Eigen::MatrixXd X(n, p);
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < p; ++j) {
            X(i, j) = norm_dist(rng);
        }
    }

    Eigen::VectorXd beta_true = Eigen::VectorXd::Ones(p);
    Eigen::VectorXd noise(n);
    for (Eigen::Index i = 0; i < n; ++i) { noise(i) = norm_dist(rng) * 0.1; }
    Eigen::VectorXd y = X * beta_true + noise;

    // Step 2. Solve
    Eigen::VectorXd beta_hat = ridge::RidgeSolver::solve(X, y, lambda);

    // Step 3. Report
    double coef_error = (beta_hat - beta_true).norm();
    double residual   = (y - X * beta_hat).norm();
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "||beta_hat - beta_true||_2 : " << coef_error << "\n";
    std::cout << "Residual ||y - X*beta||_2  : " << residual   << "\n";
    std::cout << "\n\n";
}


// ============================================================================
// Demo 2: Dual Path (n < p)
// ============================================================================

void demo_ridge_dual() {
    cdiagnostics::print_section_header("RidgeSolver - Dual Path (n < p)");

    // Step 1. Generate high-dimensional data
    const Eigen::Index n = 80, p = 300;
    const double lambda = 10.0;
    std::cout << "Matrix size : " << n << " x " << p << "\n";
    std::cout << "Lambda      : " << std::defaultfloat << lambda << "\n\n";

    std::mt19937 rng(7);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    Eigen::MatrixXd X(n, p);
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < p; ++j) {
            X(i, j) = norm_dist(rng);
        }
    }

    // Sparse ground truth: only first 10 coefficients are nonzero
    Eigen::VectorXd beta_true = Eigen::VectorXd::Zero(p);
    beta_true.head(10).setOnes();

    Eigen::VectorXd noise(n);
    for (Eigen::Index i = 0; i < n; ++i) { noise(i) = norm_dist(rng) * 0.1; }
    Eigen::VectorXd y = X * beta_true + noise;

    // Step 2. Solve
    Eigen::VectorXd beta_hat = ridge::RidgeSolver::solve(X, y, lambda);

    // Step 3. Report
    double coef_error = (beta_hat - beta_true).norm();
    double residual   = (y - X * beta_hat).norm();
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "||beta_hat - beta_true||_2 : " << coef_error << "\n";
    std::cout << "Residual ||y - X*beta||_2  : " << residual   << "\n";
    std::cout << "\n\n";
}


// ============================================================================
// Demo 3: Lambda Sweep
// ============================================================================

void demo_ridge_lambda_sweep() {
    cdiagnostics::print_section_header("RidgeSolver - Lambda Sweep");

    const Eigen::Index n = 200, p = 40;
    std::cout << "Matrix size : " << n << " x " << p << "\n\n";

    std::mt19937 rng(99);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    Eigen::MatrixXd X(n, p);
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < p; ++j) {
            X(i, j) = norm_dist(rng);
        }
    }

    Eigen::VectorXd beta_true = Eigen::VectorXd::Ones(p);
    Eigen::VectorXd noise(n);
    for (Eigen::Index i = 0; i < n; ++i) { noise(i) = norm_dist(rng) * 0.5; }
    Eigen::VectorXd y = X * beta_true + noise;

    // Sweep over a range of lambda values
    const std::vector<double> lambdas = {1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0};

    std::cout << std::setw(12) << "lambda"
              << std::setw(22) << "||beta_hat - true||_2"
              << std::setw(22) << "||residual||_2" << "\n";
    std::cout << std::string(56, '-') << "\n";

    for (double lam : lambdas) {
        Eigen::VectorXd beta_hat = ridge::RidgeSolver::solve(X, y, lam);
        double coef_error = (beta_hat - beta_true).norm();
        double residual   = (y - X * beta_hat).norm();
        std::cout << std::scientific
                  << std::setw(12) << std::setprecision(1) << lam << std::setprecision(4)
                  << std::setw(22) << coef_error
                  << std::setw(22) << residual << "\n";
    }

    std::cout << "\n\n";
}


// ============================================================================
// main
// ============================================================================

int main() {

    demo_ridge_primal();
    demo_ridge_dual();
    demo_ridge_lambda_sweep();

    return 0;
}
