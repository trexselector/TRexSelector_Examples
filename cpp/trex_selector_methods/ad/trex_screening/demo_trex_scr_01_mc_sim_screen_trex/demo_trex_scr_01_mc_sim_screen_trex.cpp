// ==============================================================================
// demo_trex_scr_01_mc_sim_screen_trex.cpp
// ==============================================================================
/**
 * @file demo_trex_scr_01_mc_sim_screen_trex.cpp
 *
 * @brief Demonstrations of the Screen-TRex Selector.
 *
 * @details
 *  Monte Carlo comparison: Ordinary vs. Bootstrap-CI Screen-TRex over an SNR
 *  sweep. Simulations are parallelised via OpenMP (run_mc_trials_screen_trex).
 *
 *  In-memory variant. See demo_trex_scr_02_mc_sim_screen_trex_mmap.cpp for the
 *  memory-mapped workflow.
 */
// ==============================================================================

// std includes
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// TRex includes
#include <utils/datageneration/utils_datagen.hpp>

// Screen-TRex demo simulation utilities
#include "trex_screening_simulation_utils.hpp"

// ==============================================================================
// Namespace aliases
// ==============================================================================

namespace datagen = trex::utils::datageneration::datagen;
using namespace scr_demo;

// ==============================================================================
// Monte Carlo simulation — Ordinary vs. Bootstrap Screen-TRex
// ==============================================================================

/**
 * @brief SNR-sweep Monte Carlo comparison of Screen-TRex variants.
 *
 * @details Compares Ordinary Screen-TRex and Bootstrap-CI Screen-TRex over a
 *          range of SNR values. Reports FDR, TPR, and estimated FDR averaged
 *          over num_MC Monte Carlo runs. Parallelised via OpenMP.
 *
 * @param num_MC   Number of Monte Carlo runs per (method, SNR) pair.
 * @param high_dim If true, use a high-dimensional setting (n=300, p=1000).
 * @param rnd_coef If true, draw support coefficients from N(0,1) each run.
 */
void demo_ScreenTRex_MonteCarlo(std::size_t num_MC,
                                bool        high_dim,
                                bool        rnd_coef)
{
    std::cout.setf(std::ios::unitbuf);
    cdiag::print_section_header("Demo: Screen-TRex Monte Carlo Comparison");

    const std::size_t n = high_dim ? 300  : 1000;
    const std::size_t p = high_dim ? 1000 : 300;
    std::cout << (high_dim ? "High-dimensional (p > n)" : "Low-dimensional (n > p)") << "\n";

    const std::size_t support_size = 10;
    const std::vector<double> snr_values =
        {0.01, 0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0};

    const auto methods = default_methods();

    // Storage: method → SNR-indexed vectors
    std::map<std::string, Eigen::VectorXd> est_fdr_map, fdr_map, tpr_map;
    for (const auto& m : methods) {
        const auto len = static_cast<Eigen::Index>(snr_values.size());
        est_fdr_map[m.name] = Eigen::VectorXd::Zero(len);
        fdr_map[m.name]     = Eigen::VectorXd::Zero(len);
        tpr_map[m.name]     = Eigen::VectorXd::Zero(len);
    }

    auto trex_ctrl = make_trex_control();

    // -----------------------------------------------------------------------
    // Method loop
    // -----------------------------------------------------------------------
    for (const auto& method : methods) {
        std::cout << std::string(70, '=') << "\n";
        std::cout << "Method: " << method.name << "\n";
        std::cout << std::string(70, '=') << "\n";

        auto screen_ctrl = make_screen_control(method.method, method.bootstrap);

        // SNR loop -----------------------------------------------------------
        for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {
            const double snr = snr_values[snr_idx];

            const auto make_data = [=](unsigned seed) -> ScrDGPData {
                return make_iid_dgp(n, p, support_size, snr, rnd_coef, seed);
            };

            const unsigned base_seed = 24u + static_cast<unsigned>(snr_idx) * 1000u;

            std::ostringstream lbl;
            lbl << method.name << "  SNR=" << std::fixed << std::setprecision(2) << snr;
            const auto result = run_mc_trials_screen_trex(
                num_MC, lbl.str(), make_data, trex_ctrl, screen_ctrl, base_seed);

            const auto ei = static_cast<Eigen::Index>(snr_idx);
            est_fdr_map[method.name](ei) = result.avg_est_fdr;
            fdr_map    [method.name](ei) = result.avg_fdr;
            tpr_map    [method.name](ei) = result.avg_tpr;
        }
    }

    const std::string file_stem =
        "scr_screen_trex_snr_n" + std::to_string(n)
        + "_p" + std::to_string(p);
    save_and_print_mc_results(num_MC, file_stem, snr_values, methods,
                              fdr_map, tpr_map, est_fdr_map);
}


// ==============================================================================
// main
// ==============================================================================

int main() {
    std::cout.setf(std::ios::unitbuf);
    omp_set_num_threads(trex_mc_workers::count());

    // -------------------------------------------------------------------------
    // Monte Carlo SNR sweep (Ordinary vs. Bootstrap-CI)
    // -------------------------------------------------------------------------
    demo_ScreenTRex_MonteCarlo(
        200,
        true,
        false
    );

    return 0;
}
