// ==============================================================================
// demo_trex_02_mc_sim_fixed_support.cpp
// ==============================================================================
/**
 * @file demo_trex_02_mc_sim_fixed_support.cpp
 *
 * @brief Monte Carlo simulation for classical T-Rex Selector with fixed support
 *        indices and same cardinality across all Monte Carlo trials.
 *
 */
// ==============================================================================

// std includes
#include <iostream>
#include <iomanip>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// OpenMP compatibility layer
#include <utils/openmp/utils_openmp.hpp>

// T-Rex Selector includes
#include <trex_selector_methods/ad/trex_core/trex.hpp>
#include <utils/datageneration/utils_datagen.hpp>
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>
#include <utils/eval_metrics/utils_eval_rates.hpp>

// Demo utilities
#include "trex_sim_utils.hpp"
using namespace trex_sim;


// ==============================================================================
// Namespace aliases
// ==============================================================================

namespace cdianostics = trex::utils::eval::cdiagnostics;
namespace datagen = trex::utils::datageneration::datagen;
namespace dummygen= trex::utils::datageneration::dummygen;

// T-Rex types
using trex::trex_selector_methods::ad::trex_core::TRexControlParameter;
using trex::trex_selector_methods::ad::trex_core::LLoopStrategy;
using trex::trex_selector_methods::ad::utils::solver_dispatch::SolverTypeForTRex;

// ==============================================================================
// Configuration helpers
// ==============================================================================

static TRexControlParameter make_base_trex_control() {
    TRexControlParameter ctrl;
    ctrl.K = 20;
    ctrl.max_dummy_multiplier = 10;
    ctrl.use_max_T_stop = true;
    ctrl.dummy_distribution = dummygen::Distribution::Normal();
    ctrl.lloop_strategy = LLoopStrategy::STANDARD;
    ctrl.tloop_stagnation_stop = true;
    ctrl.tloop_max_stagnant_steps = 7;
    ctrl.parallel_rnd_experiments = false;
    ctrl.opt_threshold = 0.75;
    ctrl.use_memory_mapping = false;
    return ctrl;
}


// ==============================================================================
// Monte Carlo Simulation — Fixed Active Support
// ==============================================================================

void demo_TRexSelector_MonteCarlo(std::size_t num_MC, bool high_dim, bool rnd_coef) {

    std::cout.setf(std::ios::unitbuf); // Flush output after each std::endl

    cdianostics::print_section_header("Demo: T-Rex Selector Monte Carlo Simulation");
    std::cout << (high_dim ? "High-dimensional (p > n)" : "Low-dimensional (n > p)") << "\n";

    const Eigen::Index n = high_dim ? 300 : 1000;
    const Eigen::Index p = high_dim ? 1000 : 300;
    std::cout << "n = " << n << ", p = " << p << "\n";
    const std::size_t cardinality_true_support = 10;
    const std::vector<double> snr_values = {0.1, 0.5, 1.0, 2.0, 5.0};
    const double tFDR = 0.1;

    // ===================================================================
    // Define solvers to test & T-Rex control parameters
    // ===================================================================

    // TREX_SOLVERS selects a subset (solver_filter.hpp), its tables get a suffix
    const std::vector<DemoSolverInfo> solvers_to_test = selected_solvers(make_default_solvers_to_test());

    // Results: dim = solver x SNR
    std::map<std::string, Eigen::VectorXd> fdr_results_map;
    std::map<std::string, Eigen::VectorXd> tpr_results_map;

    for (const auto& solver_config : solvers_to_test) {
        fdr_results_map[solver_config.solver_name] = Eigen::VectorXd(snr_values.size());
        tpr_results_map[solver_config.solver_name] = Eigen::VectorXd(snr_values.size());
    }

    // Setup TRex Parameters
    TRexControlParameter trex_ctrl = make_base_trex_control();

    // ===================================================================
    // Generate true support once and shared across all Monte Carlo runs
    // ===================================================================
    std::mt19937 rng(24);
    std::uniform_int_distribution<std::size_t> uniform_dist(0, p - 1);
    std::normal_distribution<double> normal(0.0, 1.0);

    // Generate unique support indices
    std::unordered_set<std::size_t> support_set;
    while (support_set.size() < cardinality_true_support) {
        support_set.insert(uniform_dist(rng));
    }
    std::vector<std::size_t> true_support(support_set.begin(), support_set.end());

    // Generate coefficient values
    std::vector<double> true_coefs;
    true_coefs.reserve(cardinality_true_support);
    if (rnd_coef) {
        // Random (heterogeneous) coefficients
        for (std::size_t i = 0; i < cardinality_true_support; ++i) {
            true_coefs.push_back(normal(rng));
        }
    } else {
        // Fixed (deterministic/homogeneous) coefficients
        for (std::size_t i = 0; i < cardinality_true_support; ++i) {
            true_coefs.push_back(1);
        }
    }

    std::cout << "True support (cardinality " << cardinality_true_support << "): ";
    for (const auto idx : true_support) {
        std::cout << idx << " ";
    }
    std::cout << "\n\n";

    // ===================================================================
    // Solver loop
    // ===================================================================
    for (const auto& current_solver : solvers_to_test) {

        std::cout << "===================================================\n";
        std::cout << "Solver: " << current_solver.solver_name << "\n";
        std::cout << "===================================================\n\n";

        // ===================================================================
        // Loop over SNR values
        // ===================================================================
        for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {

            const double snr = snr_values[snr_idx];
            std::cout << "SNR = " << std::fixed << std::setprecision(1) << snr << "\n";

            // Setup per-SNR solver control and DGP factory
            trex_ctrl.solver_type                 = current_solver.solver_type;
            trex_ctrl.solver_params.lambda2       = current_solver.lambda2;
            trex_ctrl.solver_params.rho_afs       = current_solver.rho_afs;  // TAFS only
            trex_ctrl.solver_params.ncgmp_variant = current_solver.ncgmp_variant;  // TNCGMP only

            const auto make_data = [=](unsigned seed) -> trex_sim::TrexDGPData {
                datagen::SyntheticData data(
                    n,
                    p,
                    true_support,
                    true_coefs, snr,
                    static_cast<int>(seed));
                return {data.getX(), data.getY(), true_support};
            };

            std::ostringstream lbl;
            lbl << "SNR=" << std::fixed << std::setprecision(2) << snr
                << " [" << current_solver.solver_name << "]";
            const unsigned base_seed = 24u + static_cast<unsigned>(snr_idx) * 1000u;
            const auto result = trex_sim::run_mc_trials_trex(
                num_MC,
                lbl.str(),
                make_data, tFDR,
                trex_ctrl,
                base_seed);

            const auto ei = static_cast<Eigen::Index>(snr_idx);
            fdr_results_map[current_solver.solver_name](ei) = result.avg_fdr;
            tpr_results_map[current_solver.solver_name](ei) = result.avg_tpr;
        }

        std::cout << "\n";
    }

    // ===================================================================
    // Print and save results
    // ===================================================================
    save_and_print_results(
        num_MC,
        n,
        p,
        trex_ctrl.tloop_max_stagnant_steps,
        snr_values,
        solvers_to_test,
        fdr_results_map,
        tpr_results_map,
        {}, // average_L_results_map not computed in this demo
        {}, // average_T_results_map not computed in this demo
        "demo_trex_02_mc_sim_fixed_support_"
    );

    std::cout << "\n\n";
}



// ==============================================================================
// Main
// ==============================================================================

int main() {

    std::cout.setf(std::ios::unitbuf);
    omp_set_num_threads(trex_mc_workers::count());
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    // ============================================================
    // Monte Carlo simulation: fixed active support
    // ============================================================
    if (true)
        demo_TRexSelector_MonteCarlo(/*num_MC=*/num_mc(200), /*high_dim=*/true, /*rnd_coef=*/false);

    if (false)
        demo_TRexSelector_MonteCarlo(/*num_MC=*/200, /*high_dim=*/false, /*rnd_coef=*/false);

    return 0;
}
