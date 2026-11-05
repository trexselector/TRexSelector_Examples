// ===================================================================================
// demo_mlm_ms_01_ridge_cv_svd.cpp
// ===================================================================================
/**
 * @file demo_mlm_ms_01_ridge_cv_svd.cpp
 *
 * @brief Demo of Ridge Regression K-Fold Cross-Validation via JacobiSVD
 *        (`ridge_cv_svd`).
 *
 * Organization:
 * -------------
 * |-- ml_methods/
 * |   |-- model_selection/
 * |       |-- demo_mlm_ms_01_ridge_cv_svd.cpp
 *
 * Scenarios covered:
 * ------------------
 *  1. Low-dimensional  : n=1000, p=500  (n > p,  10-fold)
 *  2. High-dimensional : n=300,  p=500  (p > n,  10-fold)
 *  3. Very high-dim    : n=300,  p=1000 (p >> n,  5-fold)
 *  4. Memory-mapped    : n=5000, p=500  — dataset lives on disk; fit() reads
 *                        it via ConstMapType (zero-copy Eigen::Ref binding);
 *                        grid widened to a 1e5 range (n >> p).
 *
 * Algorithm notes:
 * ----------------
 *   `ridge_cv_svd` uses `Eigen::JacobiSVD` per fold with per-fold centering
 *   and column L2-normalisation.  The lambda grid is anchored at
 *   c_max = max_j |X_{s,c}^T y_c|_j (unit-L2 scale); 100 lambdas, 1e3 range.
 *   JacobiSVD is chosen over BDCSVD for numerical safety under -O3/-march=native.
 */
// ===================================================================================

// std includes
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

// Eigen includes
#include <Eigen/Dense>

// TRex Selector includes
#include <ml_methods/model_selection/ridge_cv_svd.hpp>
#include <utils/memmap/memory_mapped_matrix.hpp>
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>


// ===================================================================================
// Namespace aliases
// ===================================================================================

namespace ms          = trex::ml_methods::model_selection;
namespace memmap      = trex::utils::memmap;
namespace cdiagnostics = trex::utils::eval::cdiagnostics;

using MmapMatrix    = memmap::MemoryMappedMatrix<double>;
using EigenMap      = MmapMatrix::MapType;
using ConstEigenMap = MmapMatrix::ConstMapType;


// ===================================================================================
// Shared data-generation helper
// ===================================================================================

/**
 * @brief Generate a synthetic regression dataset with known sparse signal.
 *
 * @details True coefficients: @p n_active entries equal to 1.0, at evenly-spaced
 *          column indices. Observations: y = X * beta_true + N(0, sigma^2).
 */
void generate_regression_data(
    Eigen::Index        n,
    Eigen::Index        p,
    double              sigma,
    unsigned int        seed,
    Eigen::MatrixXd&    X_out,
    Eigen::VectorXd&    y_out,
    Eigen::VectorXd&    beta_true,
    int                 n_active = 10)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);

    X_out.resize(n, p);
    for (Eigen::Index j = 0; j < p; ++j)
        for (Eigen::Index i = 0; i < n; ++i)
            X_out(i, j) = nd(rng);

    beta_true.setZero(p);
    Eigen::Index stride = std::max(Eigen::Index{1}, p / n_active);
    for (int k = 0; k < n_active; ++k)
        beta_true(k * stride) = 1.0;

    y_out = X_out * beta_true;
    for (Eigen::Index i = 0; i < n; ++i)
        y_out(i) += sigma * nd(rng);
}


// ===================================================================================
// Shared print helper
// ===================================================================================

/**
 * @brief Print a condensed summary of K-fold CV results from a ridge_cv_svd fit.
 */
void print_cv_summary(const ms::ridge_cv_svd& cv, Eigen::Index step_size = 10) {
    const Eigen::VectorXd& lambdas = cv.lambdas();
    const Eigen::VectorXd& mse     = cv.cv_mse();
    const Eigen::VectorXd& sem     = cv.cv_sem();
    const Eigen::Index     K       = lambdas.size();
    const Eigen::Index     i_min   = cv.index_min();
    const Eigen::Index     i_1se   = cv.index_1se();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  lambda.min : " << cv.cv_min()
              << "  (CV-MSE=" << mse(i_min) << " +/- " << sem(i_min) << ")\n";
    std::cout << "  lambda.1se : " << cv.cv_1se()
              << "  (CV-MSE=" << mse(i_1se) << " +/- " << sem(i_1se) << ")\n\n";

    std::cout << "  CV-MSE curve (every " << step_size << " steps):\n";
    std::cout << "    " << std::left
              << std::setw(14) << "lambda"
              << std::setw(12) << "CV-MSE"
              << std::setw(12) << "SEM"
              << "marker\n";

    for (Eigen::Index k = 0; k < K; k += step_size) {
        const char* marker =
            (k == i_min) ? "  <-- min" :
            (k == i_1se) ? "  <-- 1se" : "";
        std::cout << "    "
                  << std::setw(14) << lambdas(k)
                  << std::setw(12) << mse(k)
                  << std::setw(12) << sem(k)
                  << marker << "\n";
    }
    if (i_min % step_size != 0) {
        std::cout << "    "
                  << std::setw(14) << lambdas(i_min)
                  << std::setw(12) << mse(i_min)
                  << std::setw(12) << sem(i_min)
                  << "  <-- min\n";
    }
    if (i_1se % step_size != 0 && i_1se != i_min) {
        std::cout << "    "
                  << std::setw(14) << lambdas(i_1se)
                  << std::setw(12) << mse(i_1se)
                  << std::setw(12) << sem(i_1se)
                  << "  <-- 1se\n";
    }
    std::cout << "\n";
}

/**
 * @brief Fit ridge_cv_svd, print the CV summary and run the two checks.
 *
 * @param X            Design matrix (n x p); a mapped matrix binds without a copy.
 * @param y            Response vector (n).
 * @param n_folds      Number of CV folds.
 * @param lambda_ratio Grid range factor: the grid spans [lambda_max / lambda_ratio, lambda_max].
 */
void fit_and_report(const Eigen::Ref<const Eigen::MatrixXd>& X,
                    const Eigen::Ref<const Eigen::VectorXd>& y,
                    int n_folds,
                    double lambda_ratio = 1000.0)
{
    // 100-lambda grid, as in the Python and R demos.
    constexpr Eigen::Index n_lambda = 100;

    ms::ridge_cv_svd cv;
    cv.fit(X, y, n_folds, n_lambda, lambda_ratio);

    print_cv_summary(cv, 10);

    const bool order_ok = cv.cv_min() <= cv.cv_1se();
    std::cout << "  Ordering check (lambda.min <= lambda.1se): "
              << (order_ok ? "PASS" : "FAIL") << "\n";
    // A pick on the first or last grid point is the grid boundary, not a minimum.
    std::cout << "  Grid-edge check (both picks inside the grid): "
              << (cv.selection_at_grid_edge() ? "FAIL" : "PASS") << "\n\n";
}


// ===================================================================================
// Demo 1: Low-dimensional scenario (n=1000, p=500, n > p, 10-fold)
// ===================================================================================

void demo_kfold_low_dim() {
    cdiagnostics::print_section_header(
        "Demo MS-01 | ridge_cv_svd | Low-Dimensional (n=1000, p=500, 10-fold)");

    constexpr Eigen::Index n = 1'000;
    constexpr Eigen::Index p = 500;
    constexpr double sigma = 1.0;
    constexpr int n_folds = 10;

    Eigen::MatrixXd X;
    Eigen::VectorXd y, beta_true;
    generate_regression_data(n, p, sigma, /*seed=*/111, X, y, beta_true);

    std::cout << "Data: n=" << n << ", p=" << p
              << ", folds=" << n_folds
              << ", active coefs=" << (beta_true.array() != 0.0).count() << "\n\n";

    fit_and_report(X, y, n_folds);
}


// ===================================================================================
// Demo 2: High-dimensional scenario (n=300, p=500, p > n, 10-fold)
// ===================================================================================

void demo_kfold_high_dim() {
    cdiagnostics::print_section_header(
        "Demo MS-01 | ridge_cv_svd | High-Dimensional (n=300, p=500, 10-fold)");

    constexpr Eigen::Index n = 300;
    constexpr Eigen::Index p = 500;
    constexpr double sigma = 1.0;
    constexpr int n_folds = 10;

    Eigen::MatrixXd X;
    Eigen::VectorXd y, beta_true;
    generate_regression_data(n, p, sigma, /*seed=*/222, X, y, beta_true);

    std::cout << "Data: n=" << n << ", p=" << p
              << ", folds=" << n_folds
              << ", active coefs=" << (beta_true.array() != 0.0).count() << "\n\n";

    fit_and_report(X, y, n_folds);
}


// ===================================================================================
// Demo 3: Very high-dimensional scenario (n=300, p=1000, p >> n, 5-fold)
// ===================================================================================

void demo_kfold_very_high_dim() {
    cdiagnostics::print_section_header(
        "Demo MS-01 | ridge_cv_svd | Very High-Dimensional (n=300, p=1000, 5-fold)");

    constexpr Eigen::Index n = 300;
    constexpr Eigen::Index p = 1000;
    constexpr double sigma = 1.0;
    constexpr int n_folds = 5;

    Eigen::MatrixXd X;
    Eigen::VectorXd y, beta_true;
    generate_regression_data(n, p, sigma, /*seed=*/333, X, y, beta_true);

    std::cout << "Data: n=" << n << ", p=" << p
              << ", folds=" << n_folds
              << ", active coefs=" << (beta_true.array() != 0.0).count() << "\n";
    std::cout << "  Note: 5 folds used to ensure each fold's training set is large "
                 "enough relative to n.\n\n";

    fit_and_report(X, y, n_folds);
}


// ===================================================================================
// Demo 4: Memory-mapped scenario (n=5000, p=500, 10-fold)
// ===================================================================================

/**
 * @brief Path for console output without the machine-specific prefix.
 * @param path Absolute or relative file path.
 * @return ".../" plus the part from "cpp/" on, else the bare file name.
 */
static std::string display_path(const std::string& path) {
    const auto pos = path.find("/cpp/");
    if (pos != std::string::npos)
        return ".../" + path.substr(pos + 1);
    const auto slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

/**
 * @brief Demonstrates K-fold CV when the design matrix is backed by a mmap file.
 *
 * @details `ridge_cv_svd::fit()` accepts `Eigen::Ref<const MatrixXd>`, which binds
 *          to `MemoryMappedMatrix::ConstMapType` without copying the full matrix.
 *          Internally each fold materialises dense X_train / X_test sub-matrices in
 *          RAM, so there is no out-of-core streaming path.
 */
void demo_kfold_mmap() {
    cdiagnostics::print_section_header(
        "Demo MS-01 | ridge_cv_svd | Memory-Mapped Dataset (n=5000, p=500, 10-fold)");

    constexpr Eigen::Index n = 5000;
    constexpr Eigen::Index p = 500;
    constexpr double sigma = 1.0;
    constexpr int n_folds = 10;
    // n >> p: the CV optimum lies below lambda_max / 1000, so the grid is widened.
    constexpr double lambda_ratio = 1e5;

#ifdef TREX_DEMO_MMAP_PATH
    const std::string MMAP_PATH_X = TREX_DEMO_MMAP_PATH;
#else
    const std::string MMAP_PATH_X = "trex_mlm_demo_ms01_svd_mmap.bin";
#endif

    Eigen::MatrixXd X_dense;
    Eigen::VectorXd y, beta_true;
    generate_regression_data(n, p, sigma, /*seed=*/444, X_dense, y, beta_true);

    const std::size_t file_bytes =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(p) * sizeof(double);

    std::cout << "Writing X to mmap file: " << display_path(MMAP_PATH_X) << "\n";
    std::cout << "  File size: " << std::fixed << std::setprecision(2)
              << static_cast<double>(file_bytes) / (1 << 20) << " MB\n";
    std::cout << "  Active coefs: " << (beta_true.array() != 0.0).count() << "\n\n";

    {
        MmapMatrix mmap_w(MMAP_PATH_X,
                          static_cast<std::size_t>(n),
                          static_cast<std::size_t>(p),
                          memmap::AccessMode::ReadWrite);
        EigenMap X_rw = mmap_w.getMap();
        X_rw = X_dense;
    }

    std::cout << "Re-opening mmap read-only and fitting ridge_cv_svd ("
              << n_folds << "-fold)...\n\n";

    {
        const MmapMatrix mmap_r(MMAP_PATH_X,
                                static_cast<std::size_t>(n),
                                static_cast<std::size_t>(p),
                                memmap::AccessMode::ReadOnly);
        const ConstEigenMap X_ro = mmap_r.getMap();

        fit_and_report(X_ro, y, n_folds, lambda_ratio);
    }

    {
        namespace fs = std::filesystem;
        std::error_code ec;
        if (fs::remove(MMAP_PATH_X, ec))
            std::cout << "Cleaned up mmap file: " << display_path(MMAP_PATH_X) << "\n";
        else
            std::cerr << "[WARN] Could not remove mmap file: " << ec.message() << "\n";
    }

    std::cout << "\n";
}


// ===================================================================================
// Main
// ===================================================================================

int main() {
    std::cout << std::unitbuf;

    cdiagnostics::print_section_header(
        "ridge_cv_svd Demo Suite  (demo_mlm_ms_01_ridge_cv_svd)");

    demo_kfold_low_dim();
    demo_kfold_high_dim();
    demo_kfold_very_high_dim();
    demo_kfold_mmap();

    std::cout << "\nAll ridge_cv_svd demos complete.\n\n";
    return 0;
}
