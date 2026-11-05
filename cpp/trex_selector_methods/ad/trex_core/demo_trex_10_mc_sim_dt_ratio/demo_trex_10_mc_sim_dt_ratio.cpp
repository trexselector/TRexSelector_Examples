// ==============================================================================
// demo_trex_10_mc_sim_dt_ratio.cpp
// ==============================================================================
/**
 * @file demo_trex_10_mc_sim_dt_ratio.cpp
 *
 * @brief Monte Carlo simulation of the FDR and TPR of the T-Rex selector
 *        against the sample-size ratio c = n / (s ln p), for two numbers of
 *        predictors p, three signal-to-noise ratios and three base solvers
 *        (TLARS, TOMP, TAFS).
 *
 * @details Sparse recovery needs n to grow like s log p (the Donoho-Tanner
 *  scaling). This demo sets n = round(c * s * ln p) for a grid of c and two
 *  values of p, so curves of the two p that fall onto each other on the c axis
 *  confirm that c governs the power of the FDR-controlled selection. Demo 08
 *  already shows the drop of the TPR between c of about 2 and 8 at SNR = 1,
 *  and the tsolvers demo demo_ts_perf_04 shows the path-level counterpart
 *  (exact support recovery at T = 1, without FDR control).
 *
 *  Data as in demo 03 except for n: s = 10 unit coefficients on a support
 *  redrawn per trial, Normal predictors and noise.
 *  Selector as in demo 05 at tFDR = 0.1: K = 20, max_dummy_multiplier = 10,
 *  Normal dummies, STANDARD L-loop, use_max_T_stop, stagnation stop at the
 *  library default (AUTO: off for TLARS, on for TOMP and TAFS), rho_afs = 0.3
 *  for TAFS.
 *
 *  Grid: p in {1,000; 10,000} x c in {1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10} x
 *  SNR in {0.5, 1, 2}, num_MC = 200 trials per cell (first command-line
 *  argument overrides it, e.g. 2 for a smoke run).
 *
 *  Seeds: trial mc of a (p, c, SNR) cell uses the data seed
 *  24 + 1000 * snr_index + 100000 * c_index + 10000000 * p_index + mc for every
 *  solver, and the selector is seeded with the same trial seed (reproducible).
 *
 *  Output (simulation_results/data/): one table per solver and SNR in
 *  demo_trex_10_mc_sim_dt_ratio.txt and the tidy CSV
 *  demo_trex_10_mc_sim_dt_ratio.csv with the columns
 *  solver, snr, p, n, c, metric, mean, sd (metrics FDR, TPR, AvgL, AvgT; sd for
 *  FDR and TPR only). Every finished cell is cached, so a stopped run resumes.
 */
// ==============================================================================

// std includes
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// OpenMP compatibility layer
#include <utils/openmp/utils_openmp.hpp>

// T-Rex Selector includes
#include <trex_selector_methods/ad/trex_core/trex.hpp>
#include <utils/datageneration/utils_datagen.hpp>
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>

// Demo utilities
#include "trex_sim_utils.hpp"
using namespace trex_sim;


// ==============================================================================
// Namespace aliases
// ==============================================================================

namespace cdiagnostics = trex::utils::eval::cdiagnostics;
namespace datagen      = trex::utils::datageneration::datagen;
namespace dummygen     = trex::utils::datageneration::dummygen;

// T-Rex types
using trex::trex_selector_methods::ad::trex_core::TRexControlParameter;
using trex::trex_selector_methods::ad::trex_core::LLoopStrategy;
using trex::trex_selector_methods::ad::utils::solver_dispatch::SolverTypeForTRex;


// ==============================================================================
// Simulation settings
// ==============================================================================

static constexpr std::size_t S_TRUE      = 10;     // true support cardinality
static constexpr double      TFDR        = 0.1;    // target FDR
static constexpr std::size_t NUM_MC      = 200;    // trials per cell (default)
static constexpr unsigned    SEED_OFFSET = 24u;    // as demos 02-05

static const std::vector<Eigen::Index> P_VALUES   = {1'000, 10'000};
static const std::vector<double>       C_VALUES   = {1.0, 1.5, 2.0, 2.5, 3.0, 4.0,
                                                     5.0, 6.0, 8.0, 10.0};
static const std::vector<double>       SNR_VALUES = {0.5, 1.0, 2.0};

/** @brief Sample size of a grid point: n = round(c * s * ln p). */
static Eigen::Index sample_size(double c, Eigen::Index p) {
    return static_cast<Eigen::Index>(
        std::lround(c * static_cast<double>(S_TRUE) * std::log(static_cast<double>(p))));
}


// ==============================================================================
// Configuration helpers
// ==============================================================================

/** @brief T-Rex control of one base solver (demo 05 configuration). */
static TRexControlParameter make_trex_control(const DemoSolverInfo& solver) {
    TRexControlParameter ctrl;
    ctrl.solver_type           = solver.solver_type;
    ctrl.solver_params.rho_afs = solver.rho_afs;
    ctrl.K                     = 20;
    ctrl.max_dummy_multiplier  = 10;
    ctrl.use_max_T_stop        = true;
    ctrl.dummy_distribution    = dummygen::Distribution::Normal();
    ctrl.lloop_strategy        = LLoopStrategy::STANDARD;
    return ctrl;
}

/** @brief Data factory of one (n, p, SNR) cell: support redrawn per trial. */
static TrexDGPFactory make_data_factory(Eigen::Index n, Eigen::Index p, double snr) {
    return [n, p, snr](unsigned seed) -> TrexDGPData {
        std::mt19937 rng_sup(seed + 500000u);
        std::vector<std::size_t> support(static_cast<std::size_t>(p));
        std::iota(support.begin(), support.end(), std::size_t{0});
        std::shuffle(support.begin(), support.end(), rng_sup);
        support.resize(S_TRUE);
        std::sort(support.begin(), support.end());

        const std::vector<double> coefs(S_TRUE, 1.0);
        datagen::SyntheticData data(n, p, support, coefs, snr,
                                    static_cast<int>(seed),
                                    datagen::predictor_policy::Normal(),
                                    datagen::noisegen::noise_policy::Normal());
        return {data.getX(), data.getY(), support};
    };
}


// ==============================================================================
// Output: per-solver tables (console + .txt) and tidy CSV
// ==============================================================================

/** @brief Result of one solver over the grid, indexed [snr][p][c]. */
using RatioResults = std::vector<std::vector<std::vector<TrexGridPointResult>>>;

/** @brief Print the per-solver tables and write the .txt and .csv files. */
static void save_results(const std::vector<DemoSolverInfo>&        solvers,
                         const std::map<std::string, RatioResults>& results,
                         std::size_t                                 num_MC,
                         const std::string&                          file_stem)
{
    const std::string folder = DEMO_OUTPUT_DIR;
    std::filesystem::create_directories(folder);

    std::ofstream txt(folder + file_stem + ".txt");
    const auto print_dual = [&](const std::string& text) {
        std::cout << text;
        if (txt.is_open()) txt << text;
    };

    std::ostringstream head;
    head << "\n" << std::string(78, '=') << "\n"
         << "=== T-Rex FDR / TPR vs. c = n / (s ln p) (mean over " << num_MC
         << " trials) ===\n"
         << "    s = " << S_TRUE << ", tFDR = " << TFDR
         << ", K = 20, STANDARD L-loop, Normal dummies\n"
         << std::string(78, '=') << "\n";
    print_dual(head.str());

    constexpr int label_w = 18, col_w = 8;
    for (const auto& solver : solvers) {
        const auto& res = results.at(solver.solver_name);
        for (std::size_t si = 0; si < SNR_VALUES.size(); ++si) {
            std::ostringstream os;
            os << "\n" << solver.solver_name << "  |  SNR = " << SNR_VALUES[si] << "\n"
               << std::left << std::setw(label_w) << "  c";
            for (const double c : C_VALUES)
                os << std::right << std::setw(col_w) << std::fixed << std::setprecision(1) << c;
            os << "\n";
            for (std::size_t pi = 0; pi < P_VALUES.size(); ++pi) {
                os << std::left << std::setw(label_w) << ("  n  p=" + std::to_string(P_VALUES[pi]));
                for (const double c : C_VALUES)
                    os << std::right << std::setw(col_w) << sample_size(c, P_VALUES[pi]);
                os << "\n";
                const auto row = [&](const std::string& label, auto get) {
                    os << std::left << std::setw(label_w)
                       << ("  " + label + " p=" + std::to_string(P_VALUES[pi]));
                    for (std::size_t ci = 0; ci < C_VALUES.size(); ++ci)
                        os << std::right << std::setw(col_w) << std::fixed << std::setprecision(3)
                           << get(res[si][pi][ci]);
                    os << "\n";
                };
                row("FDR", [](const TrexGridPointResult& r) { return r.avg_fdr; });
                row("TPR", [](const TrexGridPointResult& r) { return r.avg_tpr; });
            }
            print_dual(os.str());
        }
    }
    print_dual("\n");
    if (txt.is_open())
        std::cout << "[Info] TXT saved to: " << folder + file_stem + ".txt\n";

    std::ofstream csv(folder + file_stem + ".csv");
    if (!csv.is_open()) {
        std::cout << "[Warning] Could not open CSV file: " << folder + file_stem + ".csv\n";
        return;
    }
    csv << "solver,snr,p,n,c,metric,mean,sd\n" << std::fixed << std::setprecision(6);
    for (const auto& solver : solvers) {
        const auto& res = results.at(solver.solver_name);
        for (std::size_t si = 0; si < SNR_VALUES.size(); ++si)
            for (std::size_t pi = 0; pi < P_VALUES.size(); ++pi)
                for (std::size_t ci = 0; ci < C_VALUES.size(); ++ci) {
                    const auto& r = res[si][pi][ci];
                    const std::string key = solver.solver_name + "," + std::to_string(SNR_VALUES[si])
                        + "," + std::to_string(P_VALUES[pi])
                        + "," + std::to_string(sample_size(C_VALUES[ci], P_VALUES[pi]))
                        + "," + std::to_string(C_VALUES[ci]) + ",";
                    csv << key << "FDR,"  << r.avg_fdr << "," << r.sd_fdr << "\n"
                        << key << "TPR,"  << r.avg_tpr << "," << r.sd_tpr << "\n"
                        << key << "AvgL," << r.avg_L   << ",\n"
                        << key << "AvgT," << r.avg_T   << ",\n";
                }
    }
    std::cout << "[Info] CSV saved to: " << folder + file_stem + ".csv\n\n";
}


// ==============================================================================
// Monte Carlo simulation — FDR / TPR vs. c = n / (s ln p)
// ==============================================================================

/** @brief Sweep p x c x SNR for TLARS, TOMP and TAFS at tFDR = 0.1. */
static void demo_TRexSelector_dt_ratio(std::size_t num_MC)
{
    cdiagnostics::print_section_header(
        "Demo: T-Rex Selector MC — FDR / TPR vs. n / (s ln p)  |  TLARS / TOMP / TAFS");
    std::cout << "s = " << S_TRUE << ",  tFDR = " << TFDR << ",  num_MC = " << num_MC
              << "  [support redrawn per trial]\n";
    for (const auto p : P_VALUES) {
        std::cout << "p = " << p << ":  n =";
        for (const double c : C_VALUES) std::cout << " " << sample_size(c, p);
        std::cout << "\n";
    }
    std::cout << "\n";

    // Stagnation stop at the library default (AUTO), as in demo 05
    const std::vector<DemoSolverInfo> solvers = {
        {SolverTypeForTRex::TLARS, "TLARS"},
        {SolverTypeForTRex::TOMP,  "TOMP"},
        {SolverTypeForTRex::TAFS,  "TAFS", 0.0, 0.3}
    };

    std::map<std::string, RatioResults> results;
    for (const auto& solver : solvers) {
        cdiagnostics::print_section_header("Solver: " + solver.solver_name);
        const TRexControlParameter ctrl = make_trex_control(solver);
        auto& res = results[solver.solver_name];
        res.assign(SNR_VALUES.size(),
                   std::vector<std::vector<TrexGridPointResult>>(
                       P_VALUES.size(), std::vector<TrexGridPointResult>(C_VALUES.size())));

        for (std::size_t si = 0; si < SNR_VALUES.size(); ++si) {
            for (std::size_t pi = 0; pi < P_VALUES.size(); ++pi) {
                for (std::size_t ci = 0; ci < C_VALUES.size(); ++ci) {
                    const Eigen::Index p = P_VALUES[pi];
                    const Eigen::Index n = sample_size(C_VALUES[ci], p);
                    const auto make_data = make_data_factory(n, p, SNR_VALUES[si]);
                    const unsigned base_seed = SEED_OFFSET + static_cast<unsigned>(si) * 1000u
                                             + static_cast<unsigned>(ci) * 100000u
                                             + static_cast<unsigned>(pi) * 10000000u;
                    std::ostringstream lbl;
                    lbl << "SNR=" << std::fixed << std::setprecision(2) << SNR_VALUES[si]
                        << " p=" << p << " c=" << std::setprecision(1) << C_VALUES[ci]
                        << " n=" << n << " [" << solver.solver_name << "]";
                    res[si][pi][ci] = run_mc_trials_trex(num_MC, lbl.str(), make_data, TFDR, ctrl,
                                                         base_seed, /*seeded_selector=*/true);
                }
            }
        }
    }

    save_results(solvers, results, num_MC, "demo_trex_10_mc_sim_dt_ratio");
}


// ==============================================================================
// Main
// ==============================================================================

int main(int argc, char** argv) {

    std::cout.setf(std::ios::unitbuf);
    omp_set_num_threads(trex_mc_workers::count());
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    // Optional first argument: num_MC (e.g. 2 for a smoke run)
    std::size_t num_MC = NUM_MC;
    if (argc > 1) {
        const long v = std::strtol(argv[1], nullptr, 10);
        if (v > 0) num_MC = static_cast<std::size_t>(v);
    }

    demo_TRexSelector_dt_ratio(num_MC);
    return 0;
}
