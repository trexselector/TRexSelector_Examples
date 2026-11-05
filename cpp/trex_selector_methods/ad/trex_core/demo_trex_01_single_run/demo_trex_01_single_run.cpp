// ==============================================================================
// demo_trex_01_single_run.cpp
// ==============================================================================
/**
 * @file demo_trex_01_single_run.cpp
 *
 * @brief Demonstration of classical T-Rex Selector.
 *
 * @details Shows basic usage of the classical T-Rex Selector for both low- and
 *          high-dimensional settings in a single-run scenario.
 */
// ==============================================================================

// std includes
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// OpenMP compatibility layer
#include <utils/openmp/utils_openmp.hpp>

// T-Rex Selector includes
#include <trex_selector_methods/ad/trex_core/trex.hpp>
#include <utils/datageneration/utils_datagen.hpp>

// Demo utilities (sparse printers for the per-variable structures)
#include "../trex_sim_utils.hpp"
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>
#include <utils/eval_metrics/utils_eval_rates.hpp>


// ==============================================================================
// Namespace aliases
// ==============================================================================

namespace cdianostics = trex::utils::eval::cdiagnostics;
namespace datagen = trex::utils::datageneration::datagen;
namespace dummygen= trex::utils::datageneration::dummygen;
namespace rates = trex::utils::eval::rates;


// T-Rex types
using trex::trex_selector_methods::ad::trex_core::TRexSelector;
using trex::trex_selector_methods::ad::trex_core::TRexControlParameter;
using trex::trex_selector_methods::ad::trex_core::LLoopStrategy;
using trex::trex_selector_methods::ad::utils::solver_dispatch::SolverTypeForTRex;

// ==============================================================================

// ==============================================================================
// Demo Single-Run: Basic T-Rex Selector functionality
// ==============================================================================

void demo_TRexSelector(bool high_dim, bool rnd_coef) {

    // Print Demo Header
    cdianostics::print_section_header("Demo: T-Rex Selector Basic Functionality");

    // Setup
    const Eigen::Index n = high_dim ? 150: 5000;
    const Eigen::Index p = high_dim ? 300 : 1000;

    const std::vector<std::size_t> true_support = {
        27, 149, 43 , 128, 42, 4
    };
    const std::vector<double> true_coefs = rnd_coef ?
        std::vector<double>{-0.4, -0.25, -0.8, 1.1, 2.5, -1.2} :
        std::vector<double>{1, 1, 1, 1, 1, 1};
    const double snr = 1.0;
    const double tFDR = 0.1;

    // Setup dual output (console + file)
    const std::string folder = DEMO_OUTPUT_DIR;
    std::filesystem::create_directories(folder);
    const std::string filename = "d01_trex_basic_n" + std::to_string(n) +
                                 "_p" + std::to_string(p) + ".txt";
    std::ofstream out_file(folder + filename);
    auto print_dual = [&](const std::string& text) {
        std::cout << text;
        if (out_file.is_open()) out_file << text;
    };

    print_dual(std::string(high_dim ? "High-dimensional (p > n)"
                                    : "Low-dimensional (n > p)") + "\n");

    // Generate synthetic data
    print_dual("Generating synthetic data...\n");
    datagen::SyntheticData data(n, p, true_support, true_coefs, snr, /*seed=*/58);

    // Create mapped views as lvalues (required by TRexSelector constructor)
    print_dual("Creating maps of data...\n");
    Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(),
                                      data.rows(),
                                      data.cols());
    Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(),
                                      data.rows());

    // Setup Control Structures
    TRexControlParameter trex_ctrl;
    trex_ctrl.K = 20;
    trex_ctrl.max_dummy_multiplier = 10;
    trex_ctrl.use_max_T_stop = true;
    trex_ctrl.dummy_distribution = dummygen::Distribution::Normal();
    trex_ctrl.lloop_strategy = LLoopStrategy::STANDARD;
    trex_ctrl.tloop_stagnation_stop = false;
    trex_ctrl.tloop_max_stagnant_steps = 5;
    trex_ctrl.parallel_rnd_experiments = false;
    trex_ctrl.solver_type = SolverTypeForTRex::TLARS;

    // Create T-Rex Selector instance
    print_dual("Creating T-Rex Selector instance...\n");
    TRexSelector trex(X_map, y_map, tFDR, trex_ctrl, 42, true);

    // Execute T-Rex Selector
    print_dual("Executing T-Rex Selector...\n");
    trex.select();

    const auto selected_indices = trex.getSelectedIndices();
    {
        std::ostringstream ss;
        ss << "Selected indices: ";
        for (const auto& idx : selected_indices) ss << idx << " ";
        ss << "\n";
        print_dual(ss.str());
    }

    const double fdp = rates::compute_fdp(selected_indices, true_support);
    const double tpp = rates::compute_tpp(selected_indices, true_support);
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4);
        ss << "False Discovery Proportion (FDP): " << fdp << "\n";
        ss << "True Positive Proportion (TPP):   " << tpp << "\n";
        print_dual(ss.str());
    }

    // Phi_prime and Phi carry one entry per variable, so they are printed
    // sparsely (see trex_sim::print_sparse_vector / print_sparse_matrix).
    trex_sim::print_sparse_vector("Adjusted Relative Occurrences (Phi_prime)",
                                  trex.getPhiPrime(), print_dual);

    trex_sim::print_sparse_matrix("Phi Matrix (Phi)", trex.getPhiMat(),
                                  print_dual);

    {
        const auto& fdp_hat_mat = trex.getFDPHatMat();
        std::ostringstream ss;
        ss << "\nEstimated FDP Matrix (FDP_hat):\n" << fdp_hat_mat << "\n";
        print_dual(ss.str());
    }

    {
        const auto& r_mat = trex.getRMat();
        std::ostringstream ss;
        ss << "\nR Matrix (R):\n" << r_mat << "\n";
        print_dual(ss.str());
    }

    {
        const auto& voting_grid = trex.getVotingGrid();
        std::ostringstream ss;
        ss << "\nVoting Grid:\n" << voting_grid.transpose() << "\n";
        print_dual(ss.str());
    }

    print_dual("\n\n");

    if (out_file.is_open()) {
        std::cout << "[Info] Results saved to: " << folder + filename << "\n";
        out_file.close();
    }
}

// ==============================================================================
// Main
// ==============================================================================

int main() {

    std::cout.setf(std::ios::unitbuf);

    // Run basic T-Rex Selector demo without Monte Carlo simulation
    // --------------------------------------------------------------------------------------
    if (true) demo_TRexSelector(/*high_dim=*/true, /*rnd_coef=*/false);

    if (true) demo_TRexSelector(/*high_dim=*/false, /*rnd_coef=*/false);

    return 0;
}
