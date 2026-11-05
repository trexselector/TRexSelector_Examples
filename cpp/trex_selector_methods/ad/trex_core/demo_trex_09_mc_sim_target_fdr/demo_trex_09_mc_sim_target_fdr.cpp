// ==============================================================================
// demo_trex_09_mc_sim_target_fdr.cpp
// ==============================================================================
/**
 * @file demo_trex_09_mc_sim_target_fdr.cpp
 *
 * @brief Monte Carlo simulation of the achieved FDR and TPR of the T-Rex
 *        selector as a function of the target FDR, for three base solvers
 *        (TLARS, TOMP, TAFS) and four signal-to-noise ratios.
 *
 * @details The T-Rex selector controls the FDR at the target level tFDR. This
 *  demo sweeps tFDR and records the achieved FDR (the mean false discovery
 *  proportion) and TPR, so FDR control shows as the FDR curve staying on or
 *  below the diagonal FDR = tFDR, and the TPR curve shows what each level of
 *  control costs in power.
 *
 *  Data as in demo 03: n = 300, p = 1,000, s = 10 unit coefficients on a
 *  support redrawn per trial, Normal predictors and noise.
 *  Selector as in demo 05: K = 20, max_dummy_multiplier = 10, Normal dummies,
 *  STANDARD L-loop, use_max_T_stop, stagnation stop at the library default
 *  (AUTO: off for TLARS, on for TOMP and TAFS), rho_afs = 0.3 for TAFS.
 *
 *  Grid: SNR in {0.5, 1, 2, 5} x tFDR in {0.01, 0.025, 0.05, 0.075, 0.1, 0.15,
 *  0.2, 0.25, 0.3, 0.4, 0.5}, num_MC = 200 trials per cell (first command-line
 *  argument overrides it, e.g. 2 for a smoke run).
 *
 *  Seeds: trial mc of an SNR uses the data seed 24 + 1000 * snr_index + mc for
 *  every tFDR and every solver, so the curves along tFDR compare the same data
 *  sets, and the selector is seeded with the same trial seed (reproducible).
 *
 *  Output (simulation_results/data/): one table per solver in
 *  demo_trex_09_mc_sim_target_fdr.txt and the tidy CSV
 *  demo_trex_09_mc_sim_target_fdr.csv with the columns
 *  solver, snr, tfdr, metric, mean, sd (metrics FDR, TPR, AvgL, AvgT; sd for
 *  FDR and TPR only). Every finished cell is cached, so a stopped run resumes.
 */
// ==============================================================================

// std includes
#include <algorithm>
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

static constexpr Eigen::Index N_OBS       = 300;    // observations
static constexpr Eigen::Index P_VARS      = 1'000;  // predictors
static constexpr std::size_t  S_TRUE      = 10;     // true support cardinality
static constexpr std::size_t  NUM_MC      = 200;    // trials per cell (default)
static constexpr unsigned     SEED_OFFSET = 24u;    // as demos 02-05

static const std::vector<double> SNR_VALUES  = {0.5, 1.0, 2.0, 5.0};
static const std::vector<double> TFDR_VALUES = {0.01, 0.025, 0.05, 0.075, 0.1, 0.15,
                                                0.2, 0.25, 0.3, 0.4, 0.5};


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

/** @brief Data factory of one SNR: demo-03 design, support redrawn per trial. */
static TrexDGPFactory make_data_factory(double snr) {
    return [snr](unsigned seed) -> TrexDGPData {
        std::mt19937 rng_sup(seed + 500000u);
        std::vector<std::size_t> support(static_cast<std::size_t>(P_VARS));
        std::iota(support.begin(), support.end(), std::size_t{0});
        std::shuffle(support.begin(), support.end(), rng_sup);
        support.resize(S_TRUE);
        std::sort(support.begin(), support.end());

        const std::vector<double> coefs(S_TRUE, 1.0);
        datagen::SyntheticData data(N_OBS, P_VARS, support, coefs, snr,
                                    static_cast<int>(seed),
                                    datagen::predictor_policy::Normal(),
                                    datagen::noisegen::noise_policy::Normal());
        return {data.getX(), data.getY(), support};
    };
}


// ==============================================================================
// Output: per-solver table (console + .txt) and tidy CSV
// ==============================================================================

/** @brief Result of one solver over the SNR x tFDR grid. */
using SweepResults = std::vector<std::vector<TrexGridPointResult>>;  // [snr][tfdr]

/** @brief Print the per-solver tables and write the .txt and .csv files. */
static void save_results(const std::vector<DemoSolverInfo>&       solvers,
                         const std::map<std::string, SweepResults>& results,
                         std::size_t                                num_MC,
                         const std::string&                         file_stem)
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
         << "=== T-Rex achieved FDR / TPR vs. target FDR (mean over " << num_MC
         << " trials) ===\n"
         << "    n = " << N_OBS << ", p = " << P_VARS << ", s = " << S_TRUE
         << ", K = 20, STANDARD L-loop, Normal dummies\n"
         << std::string(78, '=') << "\n";
    print_dual(head.str());

    constexpr int label_w = 16, col_w = 8;
    for (const auto& solver : solvers) {
        const auto& res = results.at(solver.solver_name);
        std::ostringstream os;
        os << "\n" << solver.solver_name << "\n" << std::left << std::setw(label_w) << "  tFDR";
        for (const double t : TFDR_VALUES)
            os << std::right << std::setw(col_w) << std::fixed << std::setprecision(3) << t;
        os << "\n";
        const auto row = [&](const std::string& label, std::size_t si, auto get) {
            os << std::left << std::setw(label_w) << ("  " + label);
            for (std::size_t ti = 0; ti < TFDR_VALUES.size(); ++ti)
                os << std::right << std::setw(col_w) << std::fixed << std::setprecision(3)
                   << get(res[si][ti]);
            os << "\n";
        };
        for (std::size_t si = 0; si < SNR_VALUES.size(); ++si) {
            std::ostringstream snr_lbl;
            snr_lbl << "SNR " << SNR_VALUES[si];
            row("FDR " + snr_lbl.str(), si, [](const TrexGridPointResult& r) { return r.avg_fdr; });
            row("TPR " + snr_lbl.str(), si, [](const TrexGridPointResult& r) { return r.avg_tpr; });
        }
        print_dual(os.str());
    }
    print_dual("\n");
    if (txt.is_open())
        std::cout << "[Info] TXT saved to: " << folder + file_stem + ".txt\n";

    std::ofstream csv(folder + file_stem + ".csv");
    if (!csv.is_open()) {
        std::cout << "[Warning] Could not open CSV file: " << folder + file_stem + ".csv\n";
        return;
    }
    csv << "solver,snr,tfdr,metric,mean,sd\n" << std::fixed << std::setprecision(6);
    for (const auto& solver : solvers) {
        const auto& res = results.at(solver.solver_name);
        for (std::size_t si = 0; si < SNR_VALUES.size(); ++si) {
            for (std::size_t ti = 0; ti < TFDR_VALUES.size(); ++ti) {
                const auto& r = res[si][ti];
                const std::string key = solver.solver_name + "," + std::to_string(SNR_VALUES[si])
                                        + "," + std::to_string(TFDR_VALUES[ti]) + ",";
                csv << key << "FDR,"  << r.avg_fdr << "," << r.sd_fdr << "\n"
                    << key << "TPR,"  << r.avg_tpr << "," << r.sd_tpr << "\n"
                    << key << "AvgL," << r.avg_L   << ",\n"
                    << key << "AvgT," << r.avg_T   << ",\n";
            }
        }
    }
    std::cout << "[Info] CSV saved to: " << folder + file_stem + ".csv\n\n";
}


// ==============================================================================
// Monte Carlo simulation — achieved FDR / TPR vs. target FDR
// ==============================================================================

/** @brief Sweep tFDR x SNR for TLARS, TOMP and TAFS. */
static void demo_TRexSelector_target_fdr(std::size_t num_MC)
{
    cdiagnostics::print_section_header(
        "Demo: T-Rex Selector MC — achieved FDR / TPR vs. target FDR  |  TLARS / TOMP / TAFS");
    std::cout << "n = " << N_OBS << ",  p = " << P_VARS << ",  s = " << S_TRUE
              << ",  num_MC = " << num_MC << "  [support redrawn per trial]\n"
              << "SNR grid: " << SNR_VALUES.size() << " values,  tFDR grid: "
              << TFDR_VALUES.size() << " values\n\n";

    // Stagnation stop at the library default (AUTO), as in demo 05
    const std::vector<DemoSolverInfo> solvers = {
        {SolverTypeForTRex::TLARS, "TLARS"},
        {SolverTypeForTRex::TOMP,  "TOMP"},
        {SolverTypeForTRex::TAFS,  "TAFS", 0.0, 0.3}
    };

    std::map<std::string, SweepResults> results;
    for (const auto& solver : solvers) {
        cdiagnostics::print_section_header("Solver: " + solver.solver_name);
        const TRexControlParameter ctrl = make_trex_control(solver);
        auto& res = results[solver.solver_name];
        res.assign(SNR_VALUES.size(), std::vector<TrexGridPointResult>(TFDR_VALUES.size()));

        for (std::size_t si = 0; si < SNR_VALUES.size(); ++si) {
            const auto make_data = make_data_factory(SNR_VALUES[si]);
            const unsigned base_seed = SEED_OFFSET + static_cast<unsigned>(si) * 1000u;
            for (std::size_t ti = 0; ti < TFDR_VALUES.size(); ++ti) {
                std::ostringstream lbl;
                lbl << "SNR=" << std::fixed << std::setprecision(2) << SNR_VALUES[si]
                    << " tFDR=" << std::setprecision(3) << TFDR_VALUES[ti]
                    << " [" << solver.solver_name << "]";
                res[si][ti] = run_mc_trials_trex(num_MC, lbl.str(), make_data, TFDR_VALUES[ti],
                                                 ctrl, base_seed, /*seeded_selector=*/true);
            }
        }
    }

    save_results(solvers, results, num_MC, "demo_trex_09_mc_sim_target_fdr");
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

    demo_TRexSelector_target_fdr(num_MC);
    return 0;
}
