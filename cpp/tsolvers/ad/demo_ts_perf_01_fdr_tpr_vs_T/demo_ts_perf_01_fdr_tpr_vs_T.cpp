// ============================================================================
// demo_ts_perf_01_fdr_tpr_vs_T.cpp
// ============================================================================
/**
 * @file demo_ts_perf_01_fdr_tpr_vs_T.cpp
 *
 * @brief FDR and TPR of the terminated solution paths vs. the number of
 *        included dummies T, for all demo T-solvers (simulation study of the
 *        R tlars vignette).
 *
 * @details n = 100, p = 300, 10 true coefficients equal to 3 on a fixed random
 *          support, noise sigma about 1, L = p fresh standard normal dummies
 *          per run. Each run scales X and D once and hands them to all solvers
 *          as readers; every solver is warm-started through the T grid
 *          {1, 2, 5, 10, 20, 50, 100}. FDR and TPR are the mean FDP and TPP
 *          over the runs; a variable counts as selected when its coefficient
 *          is nonzero, as in the T-Rex selector. After each T the selected
 *          original variables (dummies left out) are refit by least squares,
 *          and the refit is scored on a fresh test set of n_test rows.
 *
 *          Usage: demo_ts_perf_01_fdr_tpr_vs_T [num_MC] [threads]   (default 500, 6)
 */
// ============================================================================

// std includes
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// OpenMP includes
#include <omp.h>

// Eigen includes
#include <Eigen/Dense>

// ml_methods includes
#include <ml_methods/scaler_methods/lp_norm_scaler.hpp>

// utils includes
#include <utils/datageneration/utils_datagen.hpp>
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>
#include <utils/eval_metrics/utils_eval_rates.hpp>
#include <utils/logging/logger.hpp>
#include <utils/openmp/utils_openmp.hpp>

// demo suite includes
#include "ts_perf_solvers.hpp"
#include "ts_perf_utils.hpp"

// ============================================================================
// Namespace aliases
// ============================================================================

namespace datagen = trex::utils::datageneration::datagen;
namespace scaler = trex::ml_methods::scaler_methods;
namespace omp_utils = trex::utils::openmp;
namespace cdiagnost = trex::utils::eval::cdiagnostics;
namespace rates = trex::utils::eval::rates;


// ============================================================================
// Setup
// ============================================================================

namespace {

constexpr std::size_t n = 100;
constexpr std::size_t p = 300;
constexpr std::size_t num_dummies = p;
constexpr std::size_t num_active = 10;
constexpr double beta_active = 3.0;
constexpr double snr = 90.0;             // signal variance 10 * 3^2 over noise variance 1
constexpr unsigned support_seed = 12345;
constexpr int data_seed_offset = 1000;   // run mc uses seed data_seed_offset + mc
constexpr std::size_t n_test = 1000;     // rows of the test set of every run
constexpr int test_seed_offset = 2'000'000; // test set of run mc: seed test_seed_offset + mc
const std::vector<std::size_t> T_grid = {1, 2, 5, 10, 20, 50, 100};

/**
 * @brief Fixed random support: num_active of the p indices, sorted.
 * @return 0-based indices of the true coefficients.
 */
std::vector<std::size_t> draw_support() {
    std::vector<std::size_t> all(p);
    std::iota(all.begin(), all.end(), 0);
    std::vector<std::size_t> support;
    std::mt19937 rng(support_seed);
    std::sample(all.begin(), all.end(), std::back_inserter(support), num_active, rng);
    std::sort(support.begin(), support.end());
    return support;
}

/**
 * @brief Parse a positive integer command-line argument.
 * @param text     Argument text.
 * @param fallback Value when the argument is missing.
 * @throws std::invalid_argument if @p text is not a positive integer.
 */
long parse_positive(const char* text, long fallback) {
    if (text == nullptr) return fallback;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 1)
        throw std::invalid_argument(std::string("expected a positive integer, got: ") + text);
    return value;
}

/**
 * @brief Errors of the least-squares refit with intercept on the selected original variables.
 *
 * @details The dummies are left out. The minimum-norm solution also covers
 *          selections with as many variables as rows, or more.
 *
 * @param X        Training design on its original scale (n x p).
 * @param y        Training response (n).
 * @param X_test   Test design (n_test x p).
 * @param y_test   Test response (n_test).
 * @param selected 0-based indices of the selected original variables.
 * @param beta     True coefficients (p).
 * @return Prediction RMSE on the test set and the coefficient error ||beta_hat - beta||_2.
 */
std::pair<double, double> refit_errors(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                                       const Eigen::MatrixXd& X_test, const Eigen::VectorXd& y_test,
                                       const std::vector<std::size_t>& selected, const Eigen::VectorXd& beta) {
    const double y_mean = y.mean();
    Eigen::VectorXd beta_hat = Eigen::VectorXd::Zero(beta.size());
    double intercept = y_mean;
    if (!selected.empty()) {
        const auto k = static_cast<Eigen::Index>(selected.size());
        Eigen::MatrixXd Xs(X.rows(), k);
        for (Eigen::Index j = 0; j < k; ++j) Xs.col(j) = X.col(static_cast<Eigen::Index>(selected[j]));
        const Eigen::RowVectorXd x_mean = Xs.colwise().mean();
        Xs.rowwise() -= x_mean;
        const Eigen::VectorXd yc = (y.array() - y_mean).matrix();
        const Eigen::VectorXd b = Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>(Xs).solve(yc);
        for (Eigen::Index j = 0; j < k; ++j) beta_hat(static_cast<Eigen::Index>(selected[j])) = b(j);
        intercept = y_mean - x_mean.dot(b);
    }
    const Eigen::VectorXd prediction = (X_test * beta_hat).array() + intercept;
    const double rmse = std::sqrt((y_test - prediction).squaredNorm() / static_cast<double>(y_test.size()));
    return {rmse, (beta_hat - beta).norm()};
}

/** @brief Per-run results of one solver at every T of the grid (ms: time from construction to T). */
struct RunResult {
    std::vector<double> fdp, tpp, selected, steps, reached, ms, rmse, coef_err;
};

/** @brief Mean of @p v. */
double mean(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

/** @brief Standard error of the mean of @p v (Bessel-corrected sd over sqrt(size)). */
double std_error(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    const double m = mean(v);
    double ssq = 0.0;
    for (const double x : v) ssq += (x - m) * (x - m);
    return std::sqrt(ssq / static_cast<double>(v.size() - 1)) / std::sqrt(static_cast<double>(v.size()));
}

} // namespace


// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {

    std::cout << "\n";
    cdiagnost::print_section_header("T-Solver Performance: FDR and TPR vs. T");

    try {
        const auto num_MC = static_cast<std::size_t>(parse_positive(argc > 1 ? argv[1] : nullptr, 500));
        const int threads = static_cast<int>(parse_positive(argc > 2 ? argv[2] : nullptr, 6));

        // Parallel over the runs on the demo's team; the solvers' own OpenMP loops stay single-threaded
        omp_set_num_threads(threads);
        omp_utils::set_num_threads(threads);
        omp_set_max_active_levels(1);
        trex::utils::logging::setWarningsEnabled(false);

        const auto support = draw_support();
        const std::vector<double> coefs(num_active, beta_active);
        const auto solvers = ts_perf::demo_solvers();
        const std::size_t S = solvers.size(), K = T_grid.size();

        std::cout << "n = " << n << ", p = " << p << ", L = p = " << num_dummies
                  << ", true coefficients: " << num_active << " x " << beta_active
                  << ", SNR = " << snr << "\n";
        std::cout << "True support:";
        for (const auto j : support) std::cout << " " << j;
        std::cout << "\nT grid:";
        for (const auto T : T_grid) std::cout << " " << T;
        std::cout << "\n" << S << " solvers, " << num_MC << " runs on " << threads << " threads\n\n";

        // results[s][mc]; a solver that throws counts as a failure and is left out of its means
        std::vector<std::vector<RunResult>> results(S, std::vector<RunResult>(num_MC));
        std::vector<double> oracle_rmse(num_MC), oracle_coef_err(num_MC);  // refit on the true support
        std::atomic<int> failures{0};
        Eigen::VectorXd beta_true = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(p));
        for (const auto j : support) beta_true(static_cast<Eigen::Index>(j)) = beta_active;
        const auto t_start = std::chrono::steady_clock::now();

        #pragma omp parallel for schedule(dynamic)
        for (long mc = 0; mc < static_cast<long>(num_MC); ++mc) {
            datagen::SyntheticData data(
                static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(p),
                static_cast<Eigen::Index>(num_dummies), support, coefs, snr,
                /*seed=*/data_seed_offset + static_cast<int>(mc), /*X_seed=*/-1, /*dummy_seed=*/-1,
                datagen::predictor_policy::Normal(), datagen::dummygen::Distribution::Normal(),
                datagen::noisegen::noise_policy::Normal());

            Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(), data.rows(), data.cols());
            Eigen::Map<Eigen::MatrixXd> D_map(data.getD().data(), data.rows(), data.num_dummies());
            Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

            // Original-scale copies for the refit, a fresh test set, and the oracle refit
            const Eigen::MatrixXd X_raw = data.getX();
            const Eigen::VectorXd y_raw = data.getY();
            datagen::SyntheticData test(
                static_cast<Eigen::Index>(n_test), static_cast<Eigen::Index>(p), /*num_dummies=*/0,
                support, coefs, snr, /*seed=*/test_seed_offset + static_cast<int>(mc), /*X_seed=*/-1,
                /*dummy_seed=*/-1, datagen::predictor_policy::Normal(), datagen::dummygen::Distribution::Normal(),
                datagen::noisegen::noise_policy::Normal());
            std::tie(oracle_rmse[static_cast<std::size_t>(mc)], oracle_coef_err[static_cast<std::size_t>(mc)]) =
                refit_errors(X_raw, y_raw, test.getX(), test.getY(), support, beta_true);

            // Center and scale once in front; every solver below is a reader on
            // these buffers (precentered: active sets stop at the rank n - 1)
            scaler::LpNormScaler scaler_X(scaler::LpNormScaler::NormType::L2, true);
            scaler_X.fit_transform_inplace(X_map);
            scaler::LpNormScaler scaler_D(scaler::LpNormScaler::NormType::L2, true);
            scaler_D.fit_transform_inplace(D_map);
            y_map.array() -= y_map.mean();

            for (std::size_t s = 0; s < S; ++s) {
                RunResult& r = results[s][static_cast<std::size_t>(mc)];
                const auto t0 = std::chrono::steady_clock::now();
                std::chrono::duration<double, std::milli> untimed{0.0};  // reached count, kept out of the times
                std::vector<std::vector<std::size_t>> selections;
                try {
                    auto solver = solvers[s].make(X_map, D_map, y_map);
                    // Seed the tie-break engine per run (default: random_device, not reproducible)
                    solver->setTieSeed(static_cast<std::uint32_t>(data_seed_offset + mc));
                    for (const std::size_t T : T_grid) {
                        solver->executeStep(T, /*early_stop=*/true); // warm start from the previous T
                        const auto selected = ts_perf::selected_predictors(*solver);
                        r.fdp.push_back(rates::compute_fdp(selected, support));
                        r.tpp.push_back(rates::compute_tpp(selected, support));
                        r.selected.push_back(static_cast<double>(selected.size()));
                        r.steps.push_back(static_cast<double>(solver->getNumSteps()));
                        const auto c0 = std::chrono::steady_clock::now();
                        r.reached.push_back(ts_perf::included_dummies(*solver) >= T ? 1.0 : 0.0);
                        untimed += std::chrono::steady_clock::now() - c0;
                        r.ms.push_back((std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0) - untimed).count());
                        selections.push_back(selected);
                    }
                    // Refits after the timed path, so they stay out of the solver times
                    for (const auto& selected : selections) {
                        const auto [rmse, coef_err] = refit_errors(X_raw, y_raw, test.getX(), test.getY(),
                                                                   selected, beta_true);
                        r.rmse.push_back(rmse);
                        r.coef_err.push_back(coef_err);
                    }
                } catch (const std::exception&) {
                    r = RunResult{};
                    ++failures;
                }
            }
        }
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

        // Aggregate over the runs that completed
        auto column = [&](std::size_t s, std::size_t k, auto member) {
            std::vector<double> v;
            for (std::size_t mc = 0; mc < num_MC; ++mc)
                if (!(results[s][mc].*member).empty()) v.push_back((results[s][mc].*member)[k]);
            return v;
        };

        std::ostringstream out;
        auto table = [&](const std::string& title, auto member, double scale, int precision) {
            out << title << "\n" << std::left << std::setw(22) << "solver" << std::right;
            for (const auto T : T_grid) out << std::setw(8) << ("T=" + std::to_string(T));
            out << "\n" << std::string(22 + 8 * K, '-') << "\n" << std::fixed << std::setprecision(precision);
            for (std::size_t s = 0; s < S; ++s) {
                out << std::left << std::setw(22) << solvers[s].name << std::right;
                for (std::size_t k = 0; k < K; ++k) out << std::setw(8) << scale * mean(column(s, k, member));
                out << "\n";
            }
            out << "\n";
        };
        table("FDR (%)", &RunResult::fdp, 100.0, 1);
        table("TPR (%)", &RunResult::tpp, 100.0, 1);
        table("Runs that reached T dummies (%)", &RunResult::reached, 100.0, 0);
        table("Average number of selected variables", &RunResult::selected, 1.0, 1);
        table("Average number of steps", &RunResult::steps, 1.0, 1);
        table("Average time from construction to T (ms, cumulative)", &RunResult::ms, 1.0, 2);
        table("Prediction RMSE of the least-squares refit on the selected variables (test set)",
              &RunResult::rmse, 1.0, 2);
        table("Coefficient error of the refit, ||beta_hat - beta||_2", &RunResult::coef_err, 1.0, 2);
        out << std::setprecision(2) << "Least-squares refit on the true support: prediction RMSE "
            << mean(oracle_rmse) << ", coefficient error " << mean(oracle_coef_err) << "\n\n";

        out << "Failed solver runs: " << failures.load() << "\n";
        out << "Wall time: " << std::setprecision(1) << wall << " s\n";
        std::cout << out.str();

        // Result files, with a suffix for a TREX_SOLVERS selection
        const std::string stem = "fdr_tpr_vs_T_mc" + std::to_string(num_MC) + ts_perf::file_suffix();
        const auto csv_path = ts_perf::output_dir() / (stem + ".csv");
        std::ofstream csv(csv_path);
        csv << "solver,family,T,FDR,FDR_se,TPR,TPR_se,avg_selected,avg_steps,reached,avg_ms,"
               "rmse,rmse_se,coef_err,coef_err_se,oracle_rmse,oracle_coef_err\n"
            << std::setprecision(6);
        for (std::size_t s = 0; s < S; ++s) {
            for (std::size_t k = 0; k < K; ++k) {
                const auto fdp = column(s, k, &RunResult::fdp);
                const auto tpp = column(s, k, &RunResult::tpp);
                csv << solvers[s].name << ',' << solvers[s].family << ',' << T_grid[k] << ','
                    << mean(fdp) << ',' << std_error(fdp) << ',' << mean(tpp) << ',' << std_error(tpp) << ','
                    << mean(column(s, k, &RunResult::selected)) << ','
                    << mean(column(s, k, &RunResult::steps)) << ','
                    << mean(column(s, k, &RunResult::reached)) << ','
                    << mean(column(s, k, &RunResult::ms)) << ','
                    << mean(column(s, k, &RunResult::rmse)) << ',' << std_error(column(s, k, &RunResult::rmse)) << ','
                    << mean(column(s, k, &RunResult::coef_err)) << ','
                    << std_error(column(s, k, &RunResult::coef_err)) << ','
                    << mean(oracle_rmse) << ',' << mean(oracle_coef_err) << '\n';
            }
        }
        const auto txt_path = ts_perf::output_dir() / (stem + ".txt");
        std::ofstream(txt_path) << out.str();

        std::cout << "\nResults written to:\n  " << ts_perf::display_path(csv_path)
                  << "\n  " << ts_perf::display_path(txt_path) << "\n\n";

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n"
                  << "Usage: demo_ts_perf_01_fdr_tpr_vs_T [num_MC] [threads]\n\n";
        return 1;
    }

    return 0;
}
