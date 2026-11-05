// ===================================================================================
// demo_mlm_ms_02_enet_cv_ccd.cpp
// ===================================================================================
/**
 * @file demo_mlm_ms_02_enet_cv_ccd.cpp
 *
 * @brief Demo of coordinate-descent elastic-net path fitting (`enet_gaussian`)
 *        and K-fold cross-validation (`enet_cv_ccd`).
 *
 * Organization:
 * -------------
 * |-- ml_methods/
 * |   |-- model_selection/
 * |       |-- demo_mlm_ms_02_enet_cv_ccd.cpp
 *
 * Scenarios covered:
 * ------------------
 *  Part A — enet_gaussian (path fit):
 *    A1. Auto-grid path for alpha = {0.0 (ridge), 0.5 (EN), 1.0 (lasso)},
 *        low-dimensional (n=300, p=100, n > p).
 *    A2. High-dimensional path (n=200, p=500, p > n) for alpha = 0.0 and 1.0.
 *
 *  Part B — enet_cv_ccd (K-fold CV):
 *    B1. CV lambda selection (lambda.min / lambda.1se) for each alpha,
 *        low-dimensional (n=300, p=100, 10-fold).
 *    B2. CV for pure ridge (alpha=0) as used by TRexGVS; shows the
 *        lambda_2_lars = lambda.{min,1se} * p / 2 conversion for both selections.
 *
 * Algorithm notes:
 * ----------------
 *   `enet_gaussian` replicates glmnet's pathwise CCD with the sequential strong
 *   rule and glmnet's fdev/devmax early-termination.  `enet_cv_ccd` builds the
 *   full-data lambda grid once and reuses it across folds (glmnet semantics).
 *   For alpha=0 (pure ridge) the returned lambdas are on glmnet's reported scale;
 *   the T-Rex conversion is:  lambda_2_lars = lambda.{min,1se} * p / 2.
 */
// ===================================================================================

// std includes
#include <iomanip>
#include <iostream>
#include <random>

// Eigen includes
#include <Eigen/Dense>

// TRex Selector includes
#include <ml_methods/model_selection/enet_cv_ccd.hpp>
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>


// ===================================================================================
// Namespace aliases
// ===================================================================================

namespace ms = trex::ml_methods::model_selection;
namespace cdiagnostics = trex::utils::eval::cdiagnostics;


// ===================================================================================
// Shared data-generation helper
// ===================================================================================

/**
 * @brief Generate a synthetic regression dataset with known sparse signal.
 */
void generate_data(
    Eigen::Index     n,
    Eigen::Index     p,
    double           sigma,
    unsigned int     seed,
    Eigen::MatrixXd& X_out,
    Eigen::VectorXd& y_out,
    Eigen::VectorXd& beta_true,
    int              n_active = 10)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);

    X_out.resize(n, p);
    for (Eigen::Index j = 0; j < p; ++j)
        for (Eigen::Index i = 0; i < n; ++i)
            X_out(i, j) = nd(rng);

    beta_true.setZero(p);
    const Eigen::Index stride = std::max(Eigen::Index{1}, p / n_active);
    for (int k = 0; k < n_active; ++k)
        beta_true(k * stride) = 1.0;

    y_out = X_out * beta_true;
    for (Eigen::Index i = 0; i < n; ++i)
        y_out(i) += sigma * nd(rng);
}


// ===================================================================================
// Print helpers
// ===================================================================================

/**
 * @brief Print a condensed coefficient path from an enet_gaussian fit.
 *
 * Shows every @p step-th lambda along with the number of non-zero coefficients
 * (l0-norm), L1-norm of the coefficient vector, and the deviance ratio.
 */
void print_path_summary(const ms::enet_gaussian& en, Eigen::Index step = 10) {
    const Eigen::VectorXd& lam = en.lambdas();
    const Eigen::MatrixXd& B   = en.coef();
    const Eigen::VectorXd& dev = en.dev_ratio();
    const Eigen::Index     K   = lam.size();

    std::cout << std::fixed << std::setprecision(5);
    std::cout << "  Path: " << K << " lambdas fitted"
              << (en.converged() ? "" : " [" + std::to_string(en.n_nonconverged()) + " non-converged]")
              << "\n";
    std::cout << "  " << std::left
              << std::setw(12) << "lambda"
              << std::setw(8)  << "nnz"
              << std::setw(12) << "L1(beta)"
              << std::setw(10) << "dev.ratio"
              << "\n";

    for (Eigen::Index k = 0; k < K; k += step) {
        const Eigen::Index nnz = (B.col(k).array().abs() > 1e-9).count();
        std::cout << "  "
                  << std::setw(12) << lam(k)
                  << std::setw(8)  << nnz
                  << std::setw(12) << B.col(k).cwiseAbs().sum()
                  << std::setw(10) << dev(k)
                  << "\n";
    }
    // Always print the last point
    {
        const Eigen::Index last = K - 1;
        if (last % step != 0) {
            const Eigen::Index nnz = (B.col(last).array().abs() > 1e-9).count();
            std::cout << "  "
                      << std::setw(12) << lam(last)
                      << std::setw(8)  << nnz
                      << std::setw(12) << B.col(last).cwiseAbs().sum()
                      << std::setw(10) << dev(last)
                      << "\n";
        }
    }
    std::cout << "\n";
}

/**
 * @brief Print a condensed CV summary from an enet_cv_ccd fit.
 */
void print_cv_summary(const ms::enet_cv_ccd& cv, Eigen::Index step = 10) {
    const Eigen::VectorXd& lam = cv.lambdas();
    const Eigen::VectorXd& mse = cv.cv_mse();
    const Eigen::VectorXd& sem = cv.cv_sem();
    const Eigen::Index     K   = lam.size();
    const Eigen::Index     imin = cv.index_min();
    const Eigen::Index     i1se = cv.index_1se();

    std::cout << std::fixed << std::setprecision(5);
    std::cout << "  lambda.min : " << cv.cv_min()
              << "  (CV-MSE=" << mse(imin) << " +/- " << sem(imin) << ")\n";
    std::cout << "  lambda.1se : " << cv.cv_1se()
              << "  (CV-MSE=" << mse(i1se) << " +/- " << sem(i1se) << ")\n\n";

    std::cout << "  CV-MSE curve (every " << step << " steps):\n";
    std::cout << "    " << std::left
              << std::setw(14) << "lambda"
              << std::setw(12) << "CV-MSE"
              << std::setw(12) << "SEM"
              << "marker\n";

    for (Eigen::Index k = 0; k < K; k += step) {
        const char* marker =
            (k == imin) ? "  <-- min" :
            (k == i1se) ? "  <-- 1se" : "";
        std::cout << "    "
                  << std::setw(14) << lam(k)
                  << std::setw(12) << mse(k)
                  << std::setw(12) << sem(k)
                  << marker << "\n";
    }
    if (imin % step != 0) {
        std::cout << "    "
                  << std::setw(14) << lam(imin)
                  << std::setw(12) << mse(imin)
                  << std::setw(12) << sem(imin)
                  << "  <-- min\n";
    }
    if (i1se % step != 0 && i1se != imin) {
        std::cout << "    "
                  << std::setw(14) << lam(i1se)
                  << std::setw(12) << mse(i1se)
                  << std::setw(12) << sem(i1se)
                  << "  <-- 1se\n";
    }
    std::cout << "\n";
}


// ===================================================================================
// Part A — enet_gaussian path demos
// ===================================================================================

/**
 * @brief A1: Auto-grid paths for alpha = 0.0 / 0.5 / 1.0, low-dim (n=300, p=100).
 */
void demo_path_low_dim() {
    cdiagnostics::print_section_header(
        "Demo MS-02 | enet_gaussian path | Low-Dimensional (n=300, p=100)");

    constexpr Eigen::Index n = 300;
    constexpr Eigen::Index p = 100;
    constexpr double sigma = 1.0;

    Eigen::MatrixXd X;
    Eigen::VectorXd y, beta_true;
    generate_data(n, p, sigma, /*seed=*/101, X, y, beta_true);

    std::cout << "Data: n=" << n << ", p=" << p
              << ", active coefs=" << (beta_true.array() != 0.0).count() << "\n\n";

    for (double alpha : {0.0, 0.5, 1.0}) {
        std::cout << std::fixed << std::setprecision(1) << "--- alpha=" << alpha << " ---\n";
        ms::enet_gaussian en;
        en.fit(X, y, alpha);
        print_path_summary(en, 10);
    }
}

/**
 * @brief A2: Auto-grid path, high-dimensional (n=200, p=500, p > n).
 */
void demo_path_high_dim() {
    cdiagnostics::print_section_header(
        "Demo MS-02 | enet_gaussian path | High-Dimensional (n=200, p=500)");

    constexpr Eigen::Index n = 200;
    constexpr Eigen::Index p = 500;
    constexpr double sigma = 1.0;

    Eigen::MatrixXd X;
    Eigen::VectorXd y, beta_true;
    generate_data(n, p, sigma, /*seed=*/202, X, y, beta_true);

    std::cout << "Data: n=" << n << ", p=" << p
              << ", active coefs=" << (beta_true.array() != 0.0).count() << "\n\n";

    for (double alpha : {0.0, 1.0}) {
        std::cout << std::fixed << std::setprecision(1) << "--- alpha=" << alpha << " ---\n";
        ms::enet_gaussian en;
        en.fit(X, y, alpha);
        print_path_summary(en, 10);
    }
}


// ===================================================================================
// Part B — enet_cv_ccd CV demos
// ===================================================================================

/**
 * @brief B1: CV lambda selection for alpha = 0.0 / 0.5 / 1.0, low-dim (n=300, p=100).
 */
void demo_cv_low_dim() {
    cdiagnostics::print_section_header(
        "Demo MS-02 | enet_cv_ccd | Low-Dimensional CV (n=300, p=100, 10-fold)");

    constexpr Eigen::Index n = 300;
    constexpr Eigen::Index p = 100;
    constexpr double sigma = 1.0;
    constexpr int n_folds = 10;

    Eigen::MatrixXd X;
    Eigen::VectorXd y, beta_true;
    generate_data(n, p, sigma, /*seed=*/303, X, y, beta_true);

    std::cout << "Data: n=" << n << ", p=" << p
              << ", folds=" << n_folds
              << ", active coefs=" << (beta_true.array() != 0.0).count() << "\n\n";

    for (double alpha : {0.0, 0.5, 1.0}) {
        std::cout << std::fixed << std::setprecision(1) << "--- alpha=" << alpha << " ---\n";
        ms::enet_cv_ccd cv;
        cv.fit(X, y, alpha, n_folds);
        print_cv_summary(cv, 10);

        const bool order_ok = cv.cv_min() <= cv.cv_1se();
        std::cout << "  Ordering check (lambda.min <= lambda.1se): "
                  << (order_ok ? "PASS" : "FAIL") << "\n\n";
    }
}

/**
 * @brief B2: Pure-ridge CV (alpha=0) with TRexGVS lambda_2 conversion.
 *
 * @details TRexGVS converts the glmnet-scale lambda via:
 *          lambda_2_lars = lambda.1se * p / 2
 *          This mirrors R's `lm_dummy` and `cv.glmnet(alpha=0)`.
 */
void demo_cv_ridge_lars_conversion() {
    cdiagnostics::print_section_header(
        "Demo MS-02 | enet_cv_ccd | Ridge CV + lambda_2_lars Conversion");

    constexpr Eigen::Index n = 300;
    constexpr Eigen::Index p = 200;
    constexpr double sigma = 1.0;
    constexpr int n_folds = 10;

    Eigen::MatrixXd X;
    Eigen::VectorXd y, beta_true;
    generate_data(n, p, sigma, /*seed=*/404, X, y, beta_true);

    std::cout << "Data: n=" << n << ", p=" << p
              << ", folds=" << n_folds
              << ", active coefs=" << (beta_true.array() != 0.0).count() << "\n\n";

    ms::enet_cv_ccd cv;
    cv.fit(X, y, /*alpha=*/0.0, n_folds);
    print_cv_summary(cv, 10);

    const double lambda2_min  = cv.cv_min() * static_cast<double>(p) / 2.0;
    const double lambda2_1se  = cv.cv_1se() * static_cast<double>(p) / 2.0;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  lambda_2_lars (min) = cv_min  * p/2 = "
              << cv.cv_min() << " * " << p << "/2 = " << lambda2_min << "\n";
    std::cout << "  lambda_2_lars (1se) = cv_1se  * p/2 = "
              << cv.cv_1se() << " * " << p << "/2 = " << lambda2_1se << "\n\n";

    std::cout << "  (This is the value TRexGVS uses internally via CV_1SE_CCD.)\n\n";
}


// ===================================================================================
// Main
// ===================================================================================

int main() {
    std::cout << std::unitbuf;

    cdiagnostics::print_section_header(
        "enet_cv_ccd Demo Suite  (demo_mlm_ms_02_enet_cv_ccd)");

    // Part A: path fitting
    demo_path_low_dim();
    demo_path_high_dim();

    // Part B: K-fold CV
    demo_cv_low_dim();
    demo_cv_ridge_lars_conversion();

    std::cout << "\nAll enet_cv_ccd demos complete.\n\n";
    return 0;
}
