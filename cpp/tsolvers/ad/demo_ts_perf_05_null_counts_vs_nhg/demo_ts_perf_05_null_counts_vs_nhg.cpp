// ============================================================================
// demo_ts_perf_05_null_counts_vs_nhg.cpp
// ============================================================================
/**
 * @file demo_ts_perf_05_null_counts_vs_nhg.cpp
 *
 * @brief Number of included and selected null variables vs. the negative
 *        hypergeometric (NHG) distribution, for all demo T-solvers (numerical
 *        verification of Corollary 2, A-I and A-II of the T-Rex supplement,
 *        Figures 10 and 11).
 *
 * @details n = 150, p = 300, 5 true coefficients equal to 1 on a fixed random
 *          support, SNR = 1. Every run draws one data set and K = 20 random
 *          experiments, each with L = p fresh standard normal dummies, and
 *          runs every solver to T = 20 included dummies. The included null
 *          variables of one experiment follow NHG(p0 + L, p0, T) when nulls
 *          and dummies are exchangeable. Their relative occurrences over the
 *          K experiments give the selected null variables V(v) at the voting
 *          levels v -> 1, 0.75 and 0.5, which the NHG should dominate.
 *
 *          Usage: demo_ts_perf_05_null_counts_vs_nhg [num_MC] [threads]   (default 500, 6)
 */
// ============================================================================

// std includes
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
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

constexpr std::size_t n = 150;
constexpr std::size_t p = 300;
constexpr std::size_t num_active = 5;
constexpr std::size_t p0 = p - num_active;      // null variables
constexpr std::size_t num_dummies = p;          // L = p per random experiment
constexpr std::size_t num_experiments = 20;     // K
constexpr std::size_t T_stop = 20;              // included dummies
constexpr double beta_active = 1.0;
constexpr double snr = 1.0;
constexpr unsigned support_seed = 12345;
constexpr int data_seed_offset = 1000;          // run mc uses seed data_seed_offset + mc

/** @brief Voting levels: v -> 1 (all K experiments), 0.75 and 0.5, as minimum occurrence counts. */
const std::vector<std::string> level_names = {"v1", "v0.75", "v0.5"};
const std::vector<std::size_t> level_min_counts = {num_experiments, 16, 11};  // Phi > v

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
 * @brief PMF of the number of nulls drawn before the T-th dummy, NHG(p0 + L, p0, T).
 * @return P(V = v) for v = 0, ..., p0.
 */
std::vector<double> nhg_pmf() {
    const double N = static_cast<double>(p0 + num_dummies), K0 = static_cast<double>(p0);
    const double T = static_cast<double>(T_stop);
    auto log_choose = [](double a, double b) {
        return std::lgamma(a + 1) - std::lgamma(b + 1) - std::lgamma(a - b + 1);
    };
    std::vector<double> pmf(p0 + 1);
    for (std::size_t v = 0; v <= p0; ++v) {
        const double x = static_cast<double>(v);
        pmf[v] = std::exp(log_choose(x + T - 1, x) + log_choose(N - x - T, K0 - x) - log_choose(N, K0));
    }
    return pmf;
}

/** @brief Cumulative sums of @p pmf. */
std::vector<double> cdf_of(const std::vector<double>& pmf) {
    std::vector<double> cdf(pmf.size());
    std::partial_sum(pmf.begin(), pmf.end(), cdf.begin());
    return cdf;
}

/** @brief Frequencies of the values 0, ..., p0 in @p values. */
std::vector<std::size_t> histogram(const std::vector<int>& values) {
    std::vector<std::size_t> freq(p0 + 1, 0);
    for (const int v : values) ++freq[static_cast<std::size_t>(v)];
    return freq;
}

/** @brief Mean and standard deviation of a histogram over 0, ..., p0. */
std::pair<double, double> moments(const std::vector<double>& weights) {
    double total = 0.0, m1 = 0.0, m2 = 0.0;
    for (std::size_t v = 0; v < weights.size(); ++v) {
        const double x = static_cast<double>(v);
        total += weights[v];
        m1 += weights[v] * x;
        m2 += weights[v] * x * x;
    }
    if (total <= 0.0) return {std::nan(""), std::nan("")};
    const double mean = m1 / total;
    return {mean, std::sqrt(std::max(0.0, m2 / total - mean * mean))};
}

/** @brief Empirical CDF over 0, ..., p0 from the frequencies @p freq. */
std::vector<double> ecdf(const std::vector<std::size_t>& freq) {
    const double total = static_cast<double>(std::accumulate(freq.begin(), freq.end(), std::size_t{0}));
    std::vector<double> cdf(freq.size(), std::nan(""));
    if (total == 0.0) return cdf;
    double sum = 0.0;
    for (std::size_t v = 0; v < freq.size(); ++v) cdf[v] = (sum += static_cast<double>(freq[v])) / total;
    return cdf;
}

/** @brief Per-run results of one solver: included nulls per experiment, selected nulls per level. */
struct RunResult {
    std::vector<int> included;  ///< K counts, one per random experiment.
    std::vector<int> selected;  ///< One count per voting level.
    double reached = 0.0;       ///< Share of the experiments that reached T dummies.
    bool ok = false;            ///< False when the solver threw.
};

} // namespace


// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {

    std::cout << "\n";
    cdiagnost::print_section_header("T-Solver Performance: Null Counts vs. NHG");

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
        const std::size_t S = solvers.size(), K = num_experiments, V = level_names.size();
        const auto pmf = nhg_pmf();
        const auto nhg_cdf = cdf_of(pmf);
        const auto [nhg_mean, nhg_sd] = moments(pmf);

        std::cout << "n = " << n << ", p = " << p << ", p0 = " << p0 << ", L = p = " << num_dummies
                  << ", true coefficients: " << num_active << " x " << beta_active << ", SNR = " << snr << "\n"
                  << "K = " << K << " random experiments, T = " << T_stop << ", voting levels v -> 1, 0.75, 0.5\n"
                  << std::fixed << std::setprecision(2) << "NHG(p0 + L, p0, T): mean T p0 / (L + 1) = " << nhg_mean
                  << ", sd = " << nhg_sd << "\n";
        std::cout << "True support:";
        for (const auto j : support) std::cout << " " << j;
        std::cout << "\n" << S << " solvers, " << num_MC << " runs on " << threads << " threads\n\n";

        // results[s][mc]; a solver that throws in one experiment leaves the whole run out
        std::vector<std::vector<RunResult>> results(S, std::vector<RunResult>(num_MC));
        std::atomic<int> failures{0};
        const auto t_start = std::chrono::steady_clock::now();

        #pragma omp parallel for schedule(dynamic)
        for (long mc = 0; mc < static_cast<long>(num_MC); ++mc) {
            // One data set with K * L dummies; experiment k reads the k-th block of L columns
            const int seed = data_seed_offset + static_cast<int>(mc);
            datagen::SyntheticData data(
                static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(p),
                static_cast<Eigen::Index>(K * num_dummies), support, coefs, snr, seed, /*X_seed=*/-1,
                /*dummy_seed=*/-1, datagen::predictor_policy::Normal(), datagen::dummygen::Distribution::Normal(),
                datagen::noisegen::noise_policy::Normal());

            Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(), data.rows(), data.cols());
            Eigen::Map<Eigen::MatrixXd> D_all(data.getD().data(), data.rows(), data.num_dummies());
            Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

            // Center and scale once in front (column-wise, so the blocks stay independent)
            scaler::LpNormScaler scaler_X(scaler::LpNormScaler::NormType::L2, true);
            scaler_X.fit_transform_inplace(X_map);
            scaler::LpNormScaler scaler_D(scaler::LpNormScaler::NormType::L2, true);
            scaler_D.fit_transform_inplace(D_all);
            y_map.array() -= y_map.mean();

            for (std::size_t s = 0; s < S; ++s) {
                RunResult& r = results[s][static_cast<std::size_t>(mc)];
                std::vector<std::size_t> occurrences(p, 0);
                try {
                    for (std::size_t k = 0; k < K; ++k) {
                        Eigen::Map<Eigen::MatrixXd> D_map(D_all.data() + static_cast<Eigen::Index>(k * num_dummies * n),
                                                          data.rows(), static_cast<Eigen::Index>(num_dummies));
                        auto solver = solvers[s].make(X_map, D_map, y_map);
                        solver->setTieSeed(static_cast<std::uint32_t>(seed) * static_cast<std::uint32_t>(K)
                                           + static_cast<std::uint32_t>(k));
                        solver->executeStep(T_stop, /*early_stop=*/true);
                        int nulls = 0;
                        for (const auto j : ts_perf::selected_predictors(*solver)) {
                            ++occurrences[j];
                            if (!std::binary_search(support.begin(), support.end(), j)) ++nulls;
                        }
                        r.included.push_back(nulls);
                        r.reached += ts_perf::included_dummies(*solver) >= T_stop ? 1.0 : 0.0;
                    }
                    r.reached /= static_cast<double>(K);
                    for (const std::size_t min_count : level_min_counts) {
                        int selected = 0;
                        for (std::size_t j = 0; j < p; ++j)
                            if (occurrences[j] >= min_count && !std::binary_search(support.begin(), support.end(), j))
                                ++selected;
                        r.selected.push_back(selected);
                    }
                    r.ok = true;
                } catch (const std::exception&) {
                    r = RunResult{};
                    ++failures;
                }
            }
        }
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

        // Pool the runs that completed: included counts over runs and experiments, selected counts over runs
        std::vector<std::vector<std::size_t>> freq_included(S);
        std::vector<std::vector<std::vector<std::size_t>>> freq_selected(S, std::vector<std::vector<std::size_t>>(V));
        std::vector<double> reached(S, 0.0);
        for (std::size_t s = 0; s < S; ++s) {
            std::vector<int> included;
            std::vector<std::vector<int>> selected(V);
            std::size_t ok_runs = 0;
            for (const auto& r : results[s]) {
                if (!r.ok) continue;
                ++ok_runs;
                reached[s] += r.reached;
                included.insert(included.end(), r.included.begin(), r.included.end());
                for (std::size_t l = 0; l < V; ++l) selected[l].push_back(r.selected[l]);
            }
            reached[s] = ok_runs > 0 ? reached[s] / static_cast<double>(ok_runs) : std::nan("");
            freq_included[s] = histogram(included);
            for (std::size_t l = 0; l < V; ++l) freq_selected[s][l] = histogram(selected[l]);
        }
        auto as_weights = [](const std::vector<std::size_t>& freq) {
            return std::vector<double>(freq.begin(), freq.end());
        };

        std::ostringstream out;
        out << std::fixed;
        out << "Included null variables at T = " << T_stop << " (pooled over runs and experiments)\n"
            << std::left << std::setw(22) << "solver" << std::right << std::setw(8) << "mean" << std::setw(8) << "sd"
            << std::setw(8) << "KS D" << std::setw(10) << "reached" << "\n"
            << std::string(22 + 8 * 3 + 10, '-') << "\n";
        out << std::left << std::setw(22) << "NHG" << std::right << std::setprecision(2) << std::setw(8) << nhg_mean
            << std::setw(8) << nhg_sd << "\n";
        for (std::size_t s = 0; s < S; ++s) {
            const auto [mean, sd] = moments(as_weights(freq_included[s]));
            const auto cdf = ecdf(freq_included[s]);
            double ks = 0.0;
            for (std::size_t v = 0; v <= p0; ++v) ks = std::max(ks, std::abs(cdf[v] - nhg_cdf[v]));
            out << std::left << std::setw(22) << solvers[s].name << std::right << std::setprecision(2) << std::setw(8)
                << mean << std::setw(8) << sd << std::setprecision(3) << std::setw(8) << ks << std::setprecision(1)
                << std::setw(8) << 100.0 * reached[s] << " %\n";
        }
        out << "\n";

        // Stochastic dominance: excess = max over x of F_NHG(x) - F_V(x), positive where the NHG CDF lies above
        out << "Selected null variables V(v) over the runs: mean and largest excess of the NHG CDF over the ECDF\n"
            << std::left << std::setw(22) << "solver" << std::right;
        for (const auto& name : level_names) out << std::setw(13) << ("mean " + name);
        for (const auto& name : level_names) out << std::setw(13) << ("excess " + name);
        out << "\n" << std::string(22 + 13 * 2 * V, '-') << "\n";
        for (std::size_t s = 0; s < S; ++s) {
            out << std::left << std::setw(22) << solvers[s].name << std::right << std::setprecision(2);
            for (std::size_t l = 0; l < V; ++l) out << std::setw(13) << moments(as_weights(freq_selected[s][l])).first;
            out << std::setprecision(3);
            for (std::size_t l = 0; l < V; ++l) {
                const auto cdf = ecdf(freq_selected[s][l]);
                double excess = 0.0;
                for (std::size_t v = 0; v <= p0; ++v) excess = std::max(excess, nhg_cdf[v] - cdf[v]);
                out << std::setw(13) << excess;
            }
            out << "\n";
        }
        out << "\n";

        out << "Failed solver runs: " << failures.load() << "\n";
        out << "Wall time: " << std::setprecision(1) << wall << " s\n";
        std::cout << out.str();

        // Result files: frequencies of the counts, the NHG reference is computed by the plotter
        // A TREX_SOLVERS selection writes suffixed tables
        const std::string stem = "null_counts_vs_nhg_mc" + std::to_string(num_MC) + ts_perf::file_suffix();
        const auto csv_path = ts_perf::output_dir() / (stem + ".csv");
        std::ofstream csv(csv_path);
        csv << "solver,family,p0,L,T,K,statistic,count,frequency\n";
        auto write_rows = [&](std::size_t s, const std::string& statistic, const std::vector<std::size_t>& freq) {
            for (std::size_t v = 0; v <= p0; ++v) {
                if (freq[v] == 0) continue;
                csv << solvers[s].name << ',' << solvers[s].family << ',' << p0 << ',' << num_dummies << ','
                    << T_stop << ',' << K << ',' << statistic << ',' << v << ',' << freq[v] << '\n';
            }
        };
        for (std::size_t s = 0; s < S; ++s) {
            write_rows(s, "included", freq_included[s]);
            for (std::size_t l = 0; l < V; ++l) write_rows(s, "selected_" + level_names[l], freq_selected[s][l]);
        }
        const auto txt_path = ts_perf::output_dir() / (stem + ".txt");
        std::ofstream(txt_path) << out.str();

        std::cout << "\nResults written to:\n  " << ts_perf::display_path(csv_path)
                  << "\n  " << ts_perf::display_path(txt_path) << "\n\n";

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n"
                  << "Usage: demo_ts_perf_05_null_counts_vs_nhg [num_MC] [threads]\n\n";
        return 1;
    }

    return 0;
}
