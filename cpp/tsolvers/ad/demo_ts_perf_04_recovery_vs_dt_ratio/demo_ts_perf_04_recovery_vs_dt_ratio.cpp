// ============================================================================
// demo_ts_perf_04_recovery_vs_dt_ratio.cpp
// ============================================================================
/**
 * @file demo_ts_perf_04_recovery_vs_dt_ratio.cpp
 *
 * @brief Support recovery of the T-solvers vs. the ratio n/(s log p), and the
 *        Donoho-Tanner phase transition of T-LARS, T-OMP and T-AFS.
 *
 * @details Exact recovery: the first s distinct variables that enter the path
 *          are the true support (a dummy among them counts as a miss). Every
 *          solver runs as a precentered reader to the first dummy (T = 1),
 *          which also gives TPR and FDR at T = 1. L = p, coefficients equal
 *          to 3 on a fixed random support, noise sigma about 1.
 *
 *          Part 1: p in {250, 500, 1,000, 2,000, 5,000}, s = 10, n = n/(s log p) * s log p
 *                  for n/(s log p) in {0.25, 0.5, 1, 1.5, 2, 3, 4}, all
 *                  demo solvers.
 *          Part 2: p = 400, 20 x 20 grid of n/p and s/n in {0.05, ..., 1},
 *                  T-LARS, T-OMP and T-AFS (rho 0.3).
 *
 *          Usage: demo_ts_perf_04_recovery_vs_dt_ratio [num_MC] [threads] [parts...]   (default 500, 6, all)
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
#include <set>
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

constexpr double beta_active = 3.0;
constexpr unsigned support_seed = 12345;
constexpr int data_seed_offset = 1000;         // Part 1, run mc: seed data_seed_offset + mc
constexpr int phase_seed_offset = 3'000'000;   // Part 2, run mc and cell c: seed offset + mc * cells + c

// Part 1
constexpr std::size_t s_curves = 10;
const std::vector<std::size_t> p_grid = {250, 500, 1'000, 2'000, 5'000};
const std::vector<double> ratio_grid = {0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0};

// Part 2
constexpr std::size_t p_phase = 400;
constexpr std::size_t phase_steps = 20;  // n/p and s/n in {1, ..., 20} / 20
const std::vector<std::string> phase_solvers = {"T-LARS", "T-OMP", "T-AFS (rho 0.3)"};

/**
 * @brief Fixed random support: @p s of the @p p indices, sorted.
 * @return 0-based indices of the true coefficients.
 */
std::vector<std::size_t> draw_support(std::size_t p, std::size_t s, unsigned seed) {
    std::vector<std::size_t> all(p);
    std::iota(all.begin(), all.end(), 0);
    std::vector<std::size_t> support;
    std::mt19937 rng(seed);
    std::sample(all.begin(), all.end(), std::back_inserter(support), s, rng);
    std::sort(support.begin(), support.end());
    return support;
}

/**
 * @brief True when the first distinct variables entering the path are the support.
 * @param solver  Solver after executeStep().
 * @param support Sorted 0-based true support.
 */
bool first_entries_exact(const trex::tsolvers::ad::TSolver_Base& solver, const std::vector<std::size_t>& support) {
    std::set<std::size_t> seen;
    for (const auto& step : solver.getActions()) {
        for (const int a : step) {
            if (a <= 0) continue;  // a drop
            const auto index = static_cast<std::size_t>(a) - 1;
            if (!seen.insert(index).second) continue;
            if (!std::binary_search(support.begin(), support.end(), index)) return false;
            if (seen.size() == support.size()) return true;
        }
    }
    return false;  // the path ended before s variables entered
}

/** @brief Outcome of one solver on one data set. */
struct Outcome {
    double exact = 0.0;    ///< 1 for exact recovery by the first s entries.
    double tpp = 0.0;      ///< True positive proportion at T = 1.
    double fdp = 0.0;      ///< False discovery proportion at T = 1.
    double reached = 0.0;  ///< 1 when the first dummy entered.
    bool ok = false;       ///< False when the solver threw.
};

/**
 * @brief Draw one data set, scale it in place and run @p specs to the first dummy.
 * @return One outcome per spec.
 */
std::vector<Outcome> run_problem(std::size_t n, std::size_t p, const std::vector<std::size_t>& support,
                                 int seed, const std::vector<const ts_perf::SolverSpec*>& specs) {
    const std::vector<double> coefs(support.size(), beta_active);
    const double snr = static_cast<double>(support.size()) * beta_active * beta_active;  // noise sigma 1
    datagen::SyntheticData data(
        static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(p), support, coefs,
        snr, seed, /*X_seed=*/-1, /*dummy_seed=*/-1, datagen::predictor_policy::Normal(),
        datagen::dummygen::Distribution::Normal(), datagen::noisegen::noise_policy::Normal());
    Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(), data.rows(), data.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(data.getD().data(), data.rows(), data.num_dummies());
    Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

    // Center and scale once in front; the solvers are precentered readers
    scaler::LpNormScaler scaler_X(scaler::LpNormScaler::NormType::L2, true);
    scaler_X.fit_transform_inplace(X_map);
    scaler::LpNormScaler scaler_D(scaler::LpNormScaler::NormType::L2, true);
    scaler_D.fit_transform_inplace(D_map);
    y_map.array() -= y_map.mean();

    std::vector<Outcome> outcomes(specs.size());
    for (std::size_t k = 0; k < specs.size(); ++k) {
        try {
            auto solver = specs[k]->make(X_map, D_map, y_map);
            solver->setTieSeed(static_cast<std::uint32_t>(seed));
            solver->executeStep(1, /*early_stop=*/true);
            const auto selected = ts_perf::selected_predictors(*solver);
            outcomes[k] = {first_entries_exact(*solver, support) ? 1.0 : 0.0, rates::compute_tpp(selected, support),
                           rates::compute_fdp(selected, support),
                           ts_perf::included_dummies(*solver) == 0 ? 0.0 : 1.0, true};
        } catch (const std::exception&) {
            outcomes[k] = Outcome{};
        }
    }
    return outcomes;
}

/** @brief Mean of @p v, NaN when empty. */
double mean(const std::vector<double>& v) {
    if (v.empty()) return std::nan("");
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

/** @brief Number of observations for p and the ratio n/(s log p), natural log. */
std::size_t n_for(std::size_t p, double ratio) {
    return static_cast<std::size_t>(std::lround(ratio * static_cast<double>(s_curves) * std::log(static_cast<double>(p))));
}

/**
 * @brief Part 1: exact recovery, TPR and FDR at T = 1 vs. n/(s log p) for all demo solvers.
 * @return Number of failed solver runs.
 */
int run_part1(std::size_t num_MC) {
    cdiagnost::print_section_header("Part 1: Recovery vs. n/(s log p)");
    const auto solvers = ts_perf::demo_solvers();
    std::vector<const ts_perf::SolverSpec*> specs;
    for (const auto& spec : solvers) specs.push_back(&spec);
    const std::size_t S = solvers.size(), P = p_grid.size(), R = ratio_grid.size(), C = P * R;

    std::cout << "p in {250, 500, 1000, 2000, 5000}, L = p, s = " << s_curves << ", n/(s log p) in {0.25, 0.5, 1, 1.5, 2, 3, 4}, "
              << "T = 1\n" << S << " solvers, " << num_MC << " runs\n\n";

    std::vector<std::vector<std::size_t>> supports(P);
    for (std::size_t i = 0; i < P; ++i) supports[i] = draw_support(p_grid[i], s_curves, support_seed);

    // outcomes[c][mc] holds one outcome per solver, c = i * R + j for p_grid[i] and ratio_grid[j]
    std::vector<std::vector<std::vector<Outcome>>> outcomes(C, std::vector<std::vector<Outcome>>(num_MC));
    const auto t_start = std::chrono::steady_clock::now();

    #pragma omp parallel for schedule(dynamic)
    for (long task = 0; task < static_cast<long>(C * num_MC); ++task) {
        const auto c = static_cast<std::size_t>(task) / num_MC, mc = static_cast<std::size_t>(task) % num_MC;
        const std::size_t i = c / R, j = c % R;
        outcomes[c][mc] = run_problem(n_for(p_grid[i], ratio_grid[j]), p_grid[i], supports[i],
                                      data_seed_offset + static_cast<int>(mc), specs);
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

    // Values of one solver and configuration over the runs that completed
    int failures = 0;
    for (const auto& per_c : outcomes)
        for (const auto& per_mc : per_c)
            for (const auto& o : per_mc) failures += o.ok ? 0 : 1;
    auto column = [&](std::size_t c, std::size_t s, double Outcome::*member) {
        std::vector<double> v;
        for (const auto& per_mc : outcomes[c])
            if (per_mc[s].ok) v.push_back(per_mc[s].*member);
        return v;
    };

    std::ostringstream out;
    auto table = [&](const std::string& title, std::size_t i, double Outcome::*member) {
        out << title << ", p = " << p_grid[i] << "\n" << std::left << std::setw(22) << "n/(s log p)" << std::right;
        for (const double r : ratio_grid) {
            std::ostringstream label;
            label << r;
            out << std::setw(8) << label.str();
        }
        out << "\n" << std::left << std::setw(22) << "n" << std::right;
        for (const double r : ratio_grid) out << std::setw(8) << n_for(p_grid[i], r);
        out << "\n" << std::string(22 + 8 * R, '-') << "\n" << std::fixed << std::setprecision(1);
        for (std::size_t s = 0; s < S; ++s) {
            out << std::left << std::setw(22) << solvers[s].name << std::right;
            for (std::size_t j = 0; j < R; ++j) out << std::setw(8) << 100.0 * mean(column(i * R + j, s, member));
            out << "\n";
        }
        out << "\n";
    };
    for (std::size_t i = 0; i < P; ++i) {
        table("Exact recovery by the first s = 10 entries (%)", i, &Outcome::exact);
        table("TPR at T = 1 (%)", i, &Outcome::tpp);
        table("FDR at T = 1 (%)", i, &Outcome::fdp);
    }
    out << "Failed solver runs: " << failures << "\n";
    out << "Wall time: " << std::setprecision(1) << wall << " s\n";
    std::cout << out.str();

    // A TREX_SOLVERS selection writes suffixed tables
    const std::string stem = "recovery_vs_dt_ratio_mc" + std::to_string(num_MC) + ts_perf::file_suffix();
    const auto csv_path = ts_perf::output_dir() / (stem + ".csv");
    std::ofstream csv(csv_path);
    csv << "solver,family,p,s,n,ratio,exact,exact_se,TPR,TPR_se,FDR,FDR_se,reached\n" << std::setprecision(6);
    for (std::size_t s = 0; s < S; ++s) {
        for (std::size_t i = 0; i < P; ++i) {
            for (std::size_t j = 0; j < R; ++j) {
                const std::size_t c = i * R + j;
                const auto exact = column(c, s, &Outcome::exact);
                const auto tpp = column(c, s, &Outcome::tpp);
                const auto fdp = column(c, s, &Outcome::fdp);
                csv << solvers[s].name << ',' << solvers[s].family << ',' << p_grid[i] << ',' << s_curves << ','
                    << n_for(p_grid[i], ratio_grid[j]) << ',' << ratio_grid[j] << ',' << mean(exact) << ','
                    << std_error(exact) << ',' << mean(tpp) << ',' << std_error(tpp) << ',' << mean(fdp) << ','
                    << std_error(fdp) << ',' << mean(column(c, s, &Outcome::reached)) << '\n';
            }
        }
    }
    const auto txt_path = ts_perf::output_dir() / (stem + ".txt");
    std::ofstream(txt_path) << out.str();
    std::cout << "\nResults written to:\n  " << ts_perf::display_path(csv_path)
              << "\n  " << ts_perf::display_path(txt_path) << "\n\n";
    return failures;
}

/**
 * @brief Part 2: exact recovery on the 20 x 20 grid of n/p and s/n (Donoho-Tanner diagram).
 * @return Number of failed solver runs.
 */
int run_part2(std::size_t num_MC) {
    cdiagnost::print_section_header("Part 2: Donoho-Tanner phase transition");
    const auto solvers = ts_perf::demo_solvers();
    std::vector<const ts_perf::SolverSpec*> specs;
    for (const auto& name : phase_solvers) {
        if (!ts_perf::selected(name)) continue;
        const auto it = std::find_if(solvers.begin(), solvers.end(), [&](const auto& spec) { return spec.name == name; });
        if (it == solvers.end()) throw std::runtime_error("unknown solver: " + name);
        specs.push_back(&*it);
    }
    if (specs.empty()) {
        std::cout << "TREX_SOLVERS selects none of its solvers, Part 2 skipped\n\n";
        return 0;
    }
    const std::size_t S = specs.size(), G = phase_steps, cells = G * G;

    std::cout << "p = " << p_phase << ", L = p, n/p and s/n in {0.05, 0.10, ..., 1}, T = 1\n"
              << "T-LARS, T-OMP and T-AFS (rho 0.3), " << num_MC << " runs per cell\n\n";

    // Cell c = a * G + b: n = (a + 1) / G * p and s = (b + 1) / G * n, both whole numbers
    auto n_of = [&](std::size_t a) { return (a + 1) * p_phase / G; };
    auto s_of = [&](std::size_t a, std::size_t b) { return (b + 1) * n_of(a) / G; };
    std::vector<std::vector<std::size_t>> supports(cells);
    for (std::size_t c = 0; c < cells; ++c)
        supports[c] = draw_support(p_phase, s_of(c / G, c % G), support_seed + static_cast<unsigned>(c));

    // exact[(c * S + k) * num_MC + mc]: 1 for exact recovery, 0 for a miss, -1 for a failed run
    std::vector<signed char> exact(cells * S * num_MC, 0);
    const auto t_start = std::chrono::steady_clock::now();

    #pragma omp parallel for schedule(dynamic)
    for (long task = 0; task < static_cast<long>(cells * num_MC); ++task) {
        const auto mc = static_cast<std::size_t>(task) / cells, c = static_cast<std::size_t>(task) % cells;
        const int seed = phase_seed_offset + static_cast<int>(mc * cells + c);
        const auto outcomes = run_problem(n_of(c / G), p_phase, supports[c], seed, specs);
        for (std::size_t k = 0; k < S; ++k)
            exact[(c * S + k) * num_MC + mc] = outcomes[k].ok ? static_cast<signed char>(outcomes[k].exact) : -1;
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

    const auto failures = static_cast<int>(std::count(exact.begin(), exact.end(), -1));
    auto cell_values = [&](std::size_t c, std::size_t k) {
        std::vector<double> v;
        for (std::size_t mc = 0; mc < num_MC; ++mc) {
            const signed char e = exact[(c * S + k) * num_MC + mc];
            if (e >= 0) v.push_back(static_cast<double>(e));
        }
        return v;
    };

    // One table per solver: rows s/n from 1 down to 0.05, columns n/p
    std::ostringstream out;
    for (std::size_t k = 0; k < S; ++k) {
        out << "Exact recovery by the first s entries (%), " << specs[k]->name << "\n" << std::setw(10) << "s/n \\ n/p";
        for (std::size_t a = 0; a < G; ++a) out << std::setw(5) << std::fixed << std::setprecision(2)
                                               << static_cast<double>(a + 1) / static_cast<double>(G);
        out << "\n" << std::string(10 + 5 * G, '-') << "\n";
        for (std::size_t b = G; b-- > 0;) {
            out << std::setw(10) << std::setprecision(2) << static_cast<double>(b + 1) / static_cast<double>(G);
            for (std::size_t a = 0; a < G; ++a)
                out << std::setw(5) << std::setprecision(0) << 100.0 * mean(cell_values(a * G + b, k));
            out << "\n";
        }
        out << "\n";
    }
    out << "Failed solver runs: " << failures << "\n";
    out << "Wall time: " << std::setprecision(1) << wall << " s\n";
    std::cout << out.str();

    // A TREX_SOLVERS selection writes suffixed tables
    const std::string stem = "phase_transition_mc" + std::to_string(num_MC) + ts_perf::file_suffix();
    const auto csv_path = ts_perf::output_dir() / (stem + ".csv");
    std::ofstream csv(csv_path);
    csv << "solver,family,p,n,s,n_over_p,s_over_n,exact,exact_se\n" << std::setprecision(6);
    for (std::size_t k = 0; k < S; ++k) {
        for (std::size_t c = 0; c < cells; ++c) {
            const std::size_t a = c / G, b = c % G;
            const auto v = cell_values(c, k);
            csv << specs[k]->name << ',' << specs[k]->family << ',' << p_phase << ',' << n_of(a) << ',' << s_of(a, b)
                << ',' << static_cast<double>(a + 1) / static_cast<double>(G) << ','
                << static_cast<double>(b + 1) / static_cast<double>(G) << ',' << mean(v) << ',' << std_error(v) << '\n';
        }
    }
    const auto txt_path = ts_perf::output_dir() / (stem + ".txt");
    std::ofstream(txt_path) << out.str();
    std::cout << "\nResults written to:\n  " << ts_perf::display_path(csv_path)
              << "\n  " << ts_perf::display_path(txt_path) << "\n\n";
    return failures;
}

} // namespace


// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {

    std::cout << "\n";
    cdiagnost::print_section_header("T-Solver Performance: Recovery vs. n/(s log p)");

    try {
        const auto num_MC = static_cast<std::size_t>(ts_perf::parse_positive(argc > 1 ? argv[1] : nullptr, 500));
        const int threads = static_cast<int>(ts_perf::parse_positive(argc > 2 ? argv[2] : nullptr, 6));
        // Parts after num_MC and threads; argv[2] takes the place of the program name
        const ts_perf::PartSelection run(argc > 2 ? argc - 2 : 1, argc > 2 ? argv + 2 : argv);

        // Parallel over the runs on the demo's team; the solvers' own OpenMP loops stay single-threaded
        omp_set_num_threads(threads);
        omp_utils::set_num_threads(threads);
        omp_set_max_active_levels(1);
        trex::utils::logging::setWarningsEnabled(false);
        std::cout << num_MC << " runs on " << threads << " threads\n\n";

        if (run(1)) run_part1(num_MC);
        if (run(2)) run_part2(num_MC);

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n"
                  << "Usage: demo_ts_perf_04_recovery_vs_dt_ratio [num_MC] [threads] [parts...]\n\n";
        return 1;
    }

    return 0;
}
