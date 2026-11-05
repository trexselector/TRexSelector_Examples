// ============================================================================
// demo_ts_perf_02_runtime_vs_dummies.cpp
// ============================================================================
/**
 * @file demo_ts_perf_02_runtime_vs_dummies.cpp
 *
 * @brief Runtime of the terminated solution paths vs. the number of dummies L,
 *        for all demo T-solvers.
 *
 * @details n = 500, p = 1,000, 10 true coefficients equal to 3 on a fixed random
 *          support, noise sigma about 1, L in {1, 2, 5, 10, 20} x p. Every run
 *          draws 20 p standard normal dummies once and each L uses the first L
 *          of them, so the dummy sets are nested across the grid. X and D are
 *          scaled once in front, and every solver runs as a single-threaded
 *          reader, while the runs are spread over the threads. Per run, L and
 *          solver: the time and the steps to T = 10, the extra time of a warm
 *          start from T = 10 to 11, and the time of a fresh run to T = 11.
 *
 *          Usage: demo_ts_perf_02_runtime_vs_dummies [num_MC] [threads]   (default 500, 6)
 */
// ============================================================================

// std includes
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
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


// ============================================================================
// Setup
// ============================================================================

namespace {

constexpr std::size_t n = 500;
constexpr std::size_t p = 1'000;
constexpr std::size_t num_active = 10;
constexpr double beta_active = 3.0;
constexpr double snr = 90.0;             // signal variance 10 * 3^2 over noise variance 1
constexpr std::size_t T_stop = 10;
constexpr unsigned support_seed = 12345;
constexpr int data_seed_offset = 1000;   // run mc uses seed data_seed_offset + mc
const std::vector<std::size_t> L_factors = {1, 2, 5, 10, 20};

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

/** @brief Per-run measurements of one solver at every L of the grid. */
struct RunResult {
    std::vector<double> ms, steps, warm_ms, fresh_ms;
    bool failed = false;  ///< The solver threw at some L; the whole run is left out.
};

/** @brief Milliseconds since @p t0. */
double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace


// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {

    std::cout << "\n";
    cdiagnost::print_section_header("T-Solver Performance: Runtime vs. Number of Dummies");

    try {
        const auto num_MC = static_cast<std::size_t>(ts_perf::parse_positive(argc > 1 ? argv[1] : nullptr, 500));
        const int threads = static_cast<int>(ts_perf::parse_positive(argc > 2 ? argv[2] : nullptr, 6));

        // Parallel over the runs on the demo's team; the solvers' own OpenMP loops stay single-threaded
        omp_set_num_threads(threads);
        omp_utils::set_num_threads(threads);
        omp_set_max_active_levels(1);
        trex::utils::logging::setWarningsEnabled(false);

        const auto support = draw_support();
        const std::vector<double> coefs(num_active, beta_active);
        const auto solvers = ts_perf::demo_solvers();
        const std::size_t S = solvers.size(), K = L_factors.size();
        const std::size_t L_max = L_factors.back() * p;

        std::cout << "n = " << n << ", p = " << p << ", L in {1, 2, 5, 10, 20} x p, T = " << T_stop
                  << ", true coefficients: " << num_active << " x " << beta_active << ", SNR = " << snr << "\n";
        std::cout << S << " solvers, " << num_MC << " runs on " << threads
                  << " threads (single-threaded solvers)\n\n";

        // results[s][mc]; a solver that throws counts as a failure and is left out of its medians
        std::vector<std::vector<RunResult>> results(S, std::vector<RunResult>(num_MC));
        std::atomic<int> failures{0};
        const auto t_start = std::chrono::steady_clock::now();

        #pragma omp parallel for schedule(dynamic)
        for (long mc = 0; mc < static_cast<long>(num_MC); ++mc) {
            datagen::SyntheticData data(
                static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(L_max),
                support, coefs, snr, /*seed=*/data_seed_offset + static_cast<int>(mc), /*X_seed=*/-1,
                /*dummy_seed=*/-1, datagen::predictor_policy::Normal(), datagen::dummygen::Distribution::Normal(),
                datagen::noisegen::noise_policy::Normal());

            Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(), data.rows(), data.cols());
            Eigen::Map<Eigen::MatrixXd> D_all(data.getD().data(), data.rows(), data.num_dummies());
            Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

            // Center and scale once in front; the solvers are precentered readers
            scaler::LpNormScaler scaler_X(scaler::LpNormScaler::NormType::L2, true);
            scaler_X.fit_transform_inplace(X_map);
            scaler::LpNormScaler scaler_D(scaler::LpNormScaler::NormType::L2, true);
            scaler_D.fit_transform_inplace(D_all);
            y_map.array() -= y_map.mean();

            for (std::size_t k = 0; k < K; ++k) {
                // The first L dummy columns (column-major, so one contiguous block)
                const auto L = static_cast<Eigen::Index>(L_factors[k] * p);
                Eigen::Map<Eigen::MatrixXd> D_map(D_all.data(), data.rows(), L);

                for (std::size_t s = 0; s < S; ++s) {
                    RunResult& r = results[s][static_cast<std::size_t>(mc)];
                    if (r.failed) continue;
                    try {
                        const auto seed = static_cast<std::uint32_t>(data_seed_offset + mc);

                        // Path to T, then a warm start to T + 1
                        const auto t0 = std::chrono::steady_clock::now();
                        auto solver = solvers[s].make(X_map, D_map, y_map);
                        solver->setTieSeed(seed);
                        solver->executeStep(T_stop, /*early_stop=*/true);
                        const double to_T = ms_since(t0);
                        const auto steps = static_cast<double>(solver->getNumSteps());
                        const auto t1 = std::chrono::steady_clock::now();
                        solver->executeStep(T_stop + 1, /*early_stop=*/true);
                        const double warm = ms_since(t1);

                        // A fresh run to T + 1 for comparison
                        const auto t2 = std::chrono::steady_clock::now();
                        auto fresh = solvers[s].make(X_map, D_map, y_map);
                        fresh->setTieSeed(seed);
                        fresh->executeStep(T_stop + 1, /*early_stop=*/true);
                        const double fresh_ms = ms_since(t2);

                        r.ms.push_back(to_T);
                        r.steps.push_back(steps);
                        r.warm_ms.push_back(warm);
                        r.fresh_ms.push_back(fresh_ms);
                    } catch (const std::exception&) {
                        r = RunResult{};
                        r.failed = true;
                        ++failures;
                    }
                }
            }
        }
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

        // Medians over the runs that completed
        auto column = [&](std::size_t s, std::size_t k, auto member) {
            std::vector<double> v;
            for (std::size_t mc = 0; mc < num_MC; ++mc)
                if (!results[s][mc].failed && (results[s][mc].*member).size() == K)
                    v.push_back((results[s][mc].*member)[k]);
            return v;
        };
        auto per_step = [&](std::size_t s, std::size_t k) {
            const auto ms = column(s, k, &RunResult::ms);
            const auto steps = column(s, k, &RunResult::steps);
            std::vector<double> v(ms.size());
            for (std::size_t i = 0; i < ms.size(); ++i) v[i] = 1e3 * ms[i] / std::max(1.0, steps[i]);
            return v;
        };
        auto speedup = [&](std::size_t s, std::size_t k) {
            const auto warm = column(s, k, &RunResult::warm_ms);
            const auto fresh = column(s, k, &RunResult::fresh_ms);
            std::vector<double> v(warm.size());
            for (std::size_t i = 0; i < warm.size(); ++i) v[i] = fresh[i] / std::max(1e-6, warm[i]);
            return v;
        };

        std::ostringstream out;
        auto table = [&](const std::string& title, auto values, int precision) {
            out << title << "\n" << std::left << std::setw(22) << "solver" << std::right;
            for (const auto f : L_factors) out << std::setw(9) << ("L=" + (f == 1 ? std::string() : std::to_string(f)) + "p");
            out << "\n" << std::string(22 + 9 * K, '-') << "\n" << std::fixed << std::setprecision(precision);
            for (std::size_t s = 0; s < S; ++s) {
                out << std::left << std::setw(22) << solvers[s].name << std::right;
                for (std::size_t k = 0; k < K; ++k) out << std::setw(9) << ts_perf::median(values(s, k));
                out << "\n";
            }
            out << "\n";
        };
        table("Median time to T = 10 (ms)", [&](std::size_t s, std::size_t k) { return column(s, k, &RunResult::ms); }, 2);
        table("Median steps to T = 10", [&](std::size_t s, std::size_t k) { return column(s, k, &RunResult::steps); }, 1);
        table("Median time per step (us)", per_step, 1);
        table("Median warm start T = 10 -> 11 (ms)",
              [&](std::size_t s, std::size_t k) { return column(s, k, &RunResult::warm_ms); }, 3);
        table("Median fresh run to T = 11 over warm start 10 -> 11 (ratio)", speedup, 1);

        out << "Failed solver runs: " << failures.load() << "\n";
        out << "Wall time: " << std::setprecision(1) << wall << " s\n";
        std::cout << out.str();

        // Result files, with a suffix for a TREX_SOLVERS selection
        const std::string stem = "runtime_vs_dummies_mc" + std::to_string(num_MC) + ts_perf::file_suffix();
        const auto csv_path = ts_perf::output_dir() / (stem + ".csv");
        std::ofstream csv(csv_path);
        csv << "solver,family,L_factor,L,ms,steps,us_per_step,warm_ms,fresh_ms,fresh_over_warm\n"
            << std::setprecision(6);
        for (std::size_t s = 0; s < S; ++s) {
            for (std::size_t k = 0; k < K; ++k) {
                csv << solvers[s].name << ',' << solvers[s].family << ',' << L_factors[k] << ','
                    << L_factors[k] * p << ',' << ts_perf::median(column(s, k, &RunResult::ms)) << ','
                    << ts_perf::median(column(s, k, &RunResult::steps)) << ','
                    << ts_perf::median(per_step(s, k)) << ','
                    << ts_perf::median(column(s, k, &RunResult::warm_ms)) << ','
                    << ts_perf::median(column(s, k, &RunResult::fresh_ms)) << ','
                    << ts_perf::median(speedup(s, k)) << '\n';
            }
        }
        const auto txt_path = ts_perf::output_dir() / (stem + ".txt");
        std::ofstream(txt_path) << out.str();

        std::cout << "\nResults written to:\n  " << ts_perf::display_path(csv_path)
                  << "\n  " << ts_perf::display_path(txt_path) << "\n\n";

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n"
                  << "Usage: demo_ts_perf_02_runtime_vs_dummies [num_MC] [threads]\n\n";
        return 1;
    }

    return 0;
}
