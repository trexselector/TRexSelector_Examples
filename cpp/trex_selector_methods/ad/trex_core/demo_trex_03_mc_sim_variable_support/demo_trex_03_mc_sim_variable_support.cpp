// ==============================================================================
// demo_trex_03_mc_sim_variable_support.cpp
// ==============================================================================
/**
 * @file demo_trex_03_mc_sim_variable_support.cpp
 *
 * @brief Monte Carlo simulations for classical T-Rex Selector with variable
 *        support indices but same cardinality across all Monte Carlo trials.
 *
 */
// ==============================================================================

// std includes
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <map>
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
    ctrl.lloop_strategy = LLoopStrategy::HCONCAT;
    ctrl.tloop_stagnation_stop = true;
    ctrl.tloop_max_stagnant_steps = 5;
    return ctrl;
}

// ===============================================================================
// Monte Carlo Simulation — Variable Active Support
// ===============================================================================

void demo_TRexSelector_varMonteCarlo(std::size_t num_MC, bool high_dim, bool rnd_coef) {

    std::cout.setf(std::ios::unitbuf); // Flush output after each std::endl

    cdianostics::print_section_header("Demo: T-Rex Selector Monte Carlo Simulation");
    std::cout << (high_dim ? "High-dimensional (p > n)" : "Low-dimensional (n > p)") << "\n";

    const Eigen::Index n = high_dim ? 300 : 1000;
    const Eigen::Index p = high_dim ? 1000 : 300;
    std::cout << "n = " << n << ", p = " << p << "\n";
    const std::size_t cardinality_true_support = 10;

    // SNR values: 0.1, 0.2, ..., 2.0, 5.0
    std::vector<double> snr_values(20);
    // Setup 1 ... 20
    std::iota(snr_values.begin(), snr_values.end(), 1);
    // Change to 0.1 ... 2.0
    for (auto& x : snr_values) x *= 0.1;
    snr_values.push_back(5.0);

    // Target FDR level
    const double tFDR = 0.1;

    // TREX_SOLVERS selects a subset (solver_filter.hpp), its tables get a suffix
    const std::vector<DemoSolverInfo> solvers_to_test = selected_solvers(make_default_solvers_to_test());

    // Results: solver x SNR
    std::map<std::string, Eigen::VectorXd> fdr_results_map;
    std::map<std::string, Eigen::VectorXd> tpr_results_map;
    std::map<std::string, Eigen::VectorXd> average_L_results_map;
    std::map<std::string, Eigen::VectorXd> average_T_results_map;

    for (const auto& solver_config : solvers_to_test) {
        fdr_results_map[solver_config.solver_name] = Eigen::VectorXd(snr_values.size());
        tpr_results_map[solver_config.solver_name] = Eigen::VectorXd(snr_values.size());
        average_L_results_map[solver_config.solver_name] = Eigen::VectorXd(snr_values.size());
        average_T_results_map[solver_config.solver_name] = Eigen::VectorXd(snr_values.size());
    }

    // Setup TRex Parameters
    TRexControlParameter trex_ctrl = make_base_trex_control();

    for (const auto& current_solver : solvers_to_test) {

        cdianostics::print_section_header("Solver: " + current_solver.solver_name);

        for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {

            const double snr = snr_values[snr_idx];
            std::cout << "SNR = " << std::fixed << std::setprecision(1) << snr << "\n";

            trex_ctrl.solver_type                 = current_solver.solver_type;
            trex_ctrl.solver_params.lambda2       = current_solver.lambda2;
            trex_ctrl.solver_params.rho_afs       = current_solver.rho_afs;  // TAFS only
            trex_ctrl.solver_params.ncgmp_variant = current_solver.ncgmp_variant;  // TNCGMP only

            const auto make_data = [=](unsigned seed) -> trex_sim::TrexDGPData {
                std::mt19937 rng_sup(seed + 500000u);
                std::vector<std::size_t> pool(static_cast<std::size_t>(p));
                std::iota(pool.begin(), pool.end(), std::size_t{0});
                std::shuffle(pool.begin(), pool.end(), rng_sup);
                pool.resize(cardinality_true_support);
                std::sort(pool.begin(), pool.end());
                std::vector<std::size_t> sup = pool;

                std::vector<double> coefs;
                coefs.reserve(cardinality_true_support);
                if (rnd_coef) {
                    std::mt19937 rng_coef(seed + 600000u);
                    std::normal_distribution<double> nd(0.0, 1.0);
                    for (std::size_t i = 0; i < cardinality_true_support; ++i)
                        coefs.push_back(nd(rng_coef));
                } else {
                    for (std::size_t i = 0; i < cardinality_true_support; ++i)
                        coefs.push_back(1.0);
                }

                datagen::SyntheticData data(
                    n, p,
                    sup, coefs, snr,
                    static_cast<int>(seed),
                    datagen::predictor_policy::Normal(),
                    datagen::noisegen::noise_policy::Normal());

                return {data.getX(), data.getY(), sup};
            };

            std::ostringstream lbl;
            lbl << "SNR=" << std::fixed << std::setprecision(2) << snr
                << " [" << current_solver.solver_name << "]";
            const unsigned base_seed = 24u + static_cast<unsigned>(snr_idx) * 1000u;
            const auto result = trex_sim::run_mc_trials_trex(
                num_MC,
                lbl.str(),
                make_data,
                tFDR,
                trex_ctrl,
                base_seed);

            const auto ei = static_cast<Eigen::Index>(snr_idx);
            fdr_results_map[current_solver.solver_name](ei)       = result.avg_fdr;
            tpr_results_map[current_solver.solver_name](ei)       = result.avg_tpr;
            average_L_results_map[current_solver.solver_name](ei) = result.avg_L;
            average_T_results_map[current_solver.solver_name](ei) = result.avg_T;
        }
        std::cout << "\n";
    }

    save_and_print_results(
        num_MC,
        n,
        p,
        trex_ctrl.tloop_max_stagnant_steps,
        snr_values,
        solvers_to_test,
        fdr_results_map,
        tpr_results_map,
        average_L_results_map,
        average_T_results_map,
        "demo_trex_03_mc_sim_variable_support_"
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
    // Monte Carlo simulation: variable active support
    // ============================================================
    if (true)
        demo_TRexSelector_varMonteCarlo(/*num_MC=*/num_mc(200), /*high_dim=*/true, /*rnd_coef=*/false);

    return 0;
}
