// ============================================================================
// demo_ts_perf_03_memory_vs_np.cpp
// ============================================================================
/**
 * @file demo_ts_perf_03_memory_vs_np.cpp
 *
 * @brief Solver memory vs. the ratio n/p, for all demo T-solvers.
 *
 * @details p = 1,000, L = p, n in {50, 100, 200, 500, 1,000, 2,000}, 10 true
 *          coefficients equal to 3 on a fixed random support, noise sigma
 *          about 1, T = 10. Per run and n the data are drawn and scaled once,
 *          and every solver runs on them as a precentered reader. The solver
 *          memory is the library's memory report memoryBytes() after T = 10,
 *          the heap bytes the solver holds without the borrowed X and D. Unlike
 *          the peak resident set of a forked child, the report is portable (no
 *          fork, no getrusage()) and deterministic for given data, so the runs
 *          share OpenMP threads with single-threaded solvers.
 *
 *          Usage: demo_ts_perf_03_memory_vs_np [num_MC] [threads]   (default 500, 6)
 */
// ============================================================================

// std includes
#include <algorithm>
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

constexpr std::size_t p = 1'000;
constexpr std::size_t num_dummies = p;
constexpr std::size_t num_active = 10;
constexpr double beta_active = 3.0;
constexpr double snr = 90.0;             // signal variance 10 * 3^2 over noise variance 1
constexpr std::size_t T_stop = 10;
constexpr unsigned support_seed = 12345;
constexpr int data_seed_offset = 1000;   // run mc uses seed data_seed_offset + mc
constexpr double bytes_per_mib = 1024.0 * 1024.0;
const std::vector<std::size_t> n_grid = {50, 100, 200, 500, 1'000, 2'000};

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

/** @brief Memory report and steps of one solver on the data of one run and n. */
struct Record {
    double solver_mib = 0.0;  ///< memoryBytes() after T_stop, in MiB.
    double steps = 0.0;       ///< Steps to T_stop.
    bool ok = false;          ///< True when the solver finished.
};

} // namespace


// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {

    std::cout << std::unitbuf << "\n";
    cdiagnost::print_section_header("T-Solver Performance: Memory vs. n/p");

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
        const std::size_t S = solvers.size(), K = n_grid.size();

        std::cout << "p = " << p << ", L = p, n in {50, 100, 200, 500, 1000, 2000}, T = " << T_stop
                  << ", true coefficients: " << num_active << " x " << beta_active << ", SNR = " << snr << "\n";
        std::cout << "Solver memory: held heap bytes after T = " << T_stop
                  << " (memoryBytes; X, D borrowed, not counted)\n";
        std::cout << S << " solvers, " << num_MC << " runs on " << threads << " threads\n\n";

        // records[(s * K + k) * num_MC + mc]; every run writes its own entries
        std::vector<Record> records(S * K * num_MC);
        const auto t_start = std::chrono::steady_clock::now();

        #pragma omp parallel for schedule(dynamic)
        for (long mc = 0; mc < static_cast<long>(num_MC); ++mc) {
            const auto run = static_cast<std::size_t>(mc);
            for (std::size_t k = 0; k < K; ++k) {
                const std::size_t n = n_grid[k];
                datagen::SyntheticData data(
                    static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(num_dummies),
                    support, coefs, snr, /*seed=*/data_seed_offset + static_cast<int>(mc), /*X_seed=*/-1,
                    /*dummy_seed=*/-1, datagen::predictor_policy::Normal(), datagen::dummygen::Distribution::Normal(),
                    datagen::noisegen::noise_policy::Normal());
                Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(), data.rows(), data.cols());
                Eigen::Map<Eigen::MatrixXd> D_map(data.getD().data(), data.rows(), data.num_dummies());
                Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

                // Center and scale once in front; the solvers are precentered readers
                scaler::LpNormScaler scaler_X(scaler::LpNormScaler::NormType::L2, true);
                scaler_X.fit_transform_inplace(X_map);
                scaler::LpNormScaler scaler_D(scaler::LpNormScaler::NormType::L2, true);
                scaler_D.fit_transform_inplace(D_map);
                y_map.array() -= y_map.mean();

                for (std::size_t s = 0; s < S; ++s) {
                    Record& r = records[(s * K + k) * num_MC + run];
                    try {
                        auto solver = solvers[s].make(X_map, D_map, y_map);
                        // Seed the tie-break engine per run (default: random_device, not reproducible)
                        solver->setTieSeed(static_cast<std::uint32_t>(data_seed_offset + mc));
                        solver->executeStep(T_stop, /*early_stop=*/true);
                        r.solver_mib = static_cast<double>(solver->memoryBytes()) / bytes_per_mib;
                        r.steps = static_cast<double>(solver->getNumSteps());
                        r.ok = true;
                    } catch (const std::exception&) {
                        r = Record{};
                    }
                }
            }
        }
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

        // memory[s][k] and steps[s][k] collect the finished runs in run order
        std::vector<std::vector<std::vector<double>>> memory(S, std::vector<std::vector<double>>(K));
        std::vector<std::vector<std::vector<double>>> steps(S, std::vector<std::vector<double>>(K));
        std::size_t failures = 0;
        for (std::size_t s = 0; s < S; ++s) {
            for (std::size_t k = 0; k < K; ++k) {
                for (std::size_t mc = 0; mc < num_MC; ++mc) {
                    const Record& r = records[(s * K + k) * num_MC + mc];
                    if (!r.ok) {
                        ++failures;
                        continue;
                    }
                    memory[s][k].push_back(r.solver_mib);
                    steps[s][k].push_back(r.steps);
                }
            }
        }

        // Data size per n: X, D and y as doubles
        std::vector<double> data_mib(K);
        for (std::size_t k = 0; k < K; ++k)
            data_mib[k] = 8.0 * static_cast<double>(n_grid[k] * (p + num_dummies + 1)) / bytes_per_mib;

        std::ostringstream out;
        out << "\n";
        auto header = [&](const std::string& title) {
            out << title << "\n" << std::left << std::setw(22) << "solver" << std::right;
            for (const auto n : n_grid) out << std::setw(9) << ("n=" + std::to_string(n));
            out << "\n" << std::string(22 + 9 * K, '-') << "\n";
        };
        header("Data X, D and y (MiB)");
        out << std::left << std::setw(22) << "all solvers" << std::right << std::fixed << std::setprecision(2);
        for (std::size_t k = 0; k < K; ++k) out << std::setw(9) << data_mib[k];
        out << "\n\n";
        auto table = [&](const std::string& title, auto value, int precision) {
            header(title);
            out << std::fixed << std::setprecision(precision);
            for (std::size_t s = 0; s < S; ++s) {
                out << std::left << std::setw(22) << solvers[s].name << std::right;
                for (std::size_t k = 0; k < K; ++k) out << std::setw(9) << value(s, k);
                out << "\n";
            }
            out << "\n";
        };
        auto largest = [&](std::size_t s, std::size_t k) {
            return memory[s][k].empty() ? 0.0 : *std::max_element(memory[s][k].begin(), memory[s][k].end());
        };
        table("Median solver memory: held heap after T = 10 (MiB)",
              [&](std::size_t s, std::size_t k) { return ts_perf::median(memory[s][k]); }, 2);
        table("Largest solver memory over the runs: held heap after T = 10 (MiB)", largest, 2);
        table("Median solver memory relative to the data X, D and y (%)",
              [&](std::size_t s, std::size_t k) { return 100.0 * ts_perf::median(memory[s][k]) / data_mib[k]; }, 1);
        table("Median steps to T = 10", [&](std::size_t s, std::size_t k) { return ts_perf::median(steps[s][k]); }, 1);

        out << "Failed solver runs: " << failures << "\n";
        out << "Wall time: " << std::setprecision(1) << wall << " s\n";
        std::cout << out.str();

        // Result files, with a suffix for a TREX_SOLVERS selection
        const std::string stem = "memory_vs_np_mc" + std::to_string(num_MC) + ts_perf::file_suffix();
        const auto csv_path = ts_perf::output_dir() / (stem + ".csv");
        std::ofstream csv(csv_path);
        csv << "solver,family,n,n_over_p,data_mib,solver_mib,solver_max_mib,steps\n" << std::setprecision(6);
        for (std::size_t s = 0; s < S; ++s) {
            for (std::size_t k = 0; k < K; ++k) {
                csv << solvers[s].name << ',' << solvers[s].family << ',' << n_grid[k] << ','
                    << static_cast<double>(n_grid[k]) / static_cast<double>(p) << ',' << data_mib[k] << ','
                    << ts_perf::median(memory[s][k]) << ',' << largest(s, k) << ','
                    << ts_perf::median(steps[s][k]) << '\n';
            }
        }
        const auto txt_path = ts_perf::output_dir() / (stem + ".txt");
        std::ofstream(txt_path) << out.str();

        std::cout << "\nResults written to:\n  " << ts_perf::display_path(csv_path)
                  << "\n  " << ts_perf::display_path(txt_path) << "\n\n";

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n"
                  << "Usage: demo_ts_perf_03_memory_vs_np [num_MC] [threads]\n\n";
        return 1;
    }

    return 0;
}
