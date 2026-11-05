// ==============================================================================
// demo_trex_scr_03_mc_sim_correlated.cpp
// ==============================================================================
/**
 * @file demo_trex_scr_03_mc_sim_correlated.cpp
 *
 * @brief Screen-TRex on correlated predictor structures (AR1 and equi).
 *
 * @details
 *    1. Generate X with AR(1) or equicorrelation structure.
 *    2. Run Screen-TRex with the matching DA variant.
 *    3. Report FDP, TPP, estimated FDR, and estimated ρ.
 *
 *  Part 1 — SNR sweep: Ordinary vs. Bootstrap vs. DA-AR1 on AR(1) data.
 *  Part 2 — SNR sweep: Ordinary vs. Bootstrap vs. DA-EQUI on equicorrelated data.
 *  Part 3 — rho sweep: Ordinary vs. Bootstrap vs. DA-AR1 on AR(1) data.
 *  Part 4 — rho sweep: Ordinary vs. Bootstrap vs. DA-EQUI on equicorrelated data.
 *  Part 5 — rho sweep: Ordinary vs. Bootstrap vs. DA-BLOCK-EQUI on block-equicorr data.
 *
 *  All parts are Monte Carlo studies parallelised via OpenMP.
 */
// ==============================================================================

// std includes
#include <map>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// Screen-TRex demo simulation utilities
#include "trex_screening_simulation_utils.hpp"

using namespace scr_demo;



// ==============================================================================
// Part 1: SNR sweep — Ordinary vs. Bootstrap vs. DA-AR1 on AR(1) data
// ==============================================================================

/**
 * @brief SNR-sweep Monte Carlo: Ordinary vs. Bootstrap DA-AR1 Screen-TRex
 *        on AR(1)-correlated data.  Parallelised via OpenMP.
 *
 * @param num_MC    Number of Monte Carlo runs per (method, SNR) pair.
 * @param rho       AR(1) lag-1 correlation.
 * @param beta_val  Coefficient magnitude for active variables.
 */
void demo_DA_AR1_MonteCarlo(std::size_t num_MC,
                             double     rho      = 0.4,
                             double     beta_val = 1.0)
{
    std::cout.setf(std::ios::unitbuf);
    cdiag::print_section_header(
        "Part 1: DA-AR1 Screen-TRex Monte Carlo on AR(1) Data"
    );

    const Eigen::Index n = 300, p = 1000, p1 = 10;
    std::cout << "DGP: AR(1), n=" << n << ", p=" << p << ", p1=" << p1
              << ", rho=" << rho << "\n";

    const std::vector<double> snr_values =
        {0.01, 0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0};

    // Methods to compare
    const std::vector<ScreenMethodInfo> methods = {
        {"Screen-TRex Ordinary: TLARS", ScreenTRexMethod::TREX, false},
        {"Screen-TRex Bootstrap-CI: TLARS", ScreenTRexMethod::TREX, true},
        {"Screen-TRex Ordinary+DA-AR1: TLARS", ScreenTRexMethod::TREX_DA_AR1,
            false},
        {"Screen-TRex Bootstrap-CI+DA-AR1: TLARS", ScreenTRexMethod::TREX_DA_AR1,
            true}
    };

    std::map<std::string, Eigen::VectorXd> est_fdr_map, fdr_map, tpr_map;
    for (const auto& m : methods) {
        const auto len = static_cast<Eigen::Index>(snr_values.size());
        est_fdr_map[m.name] = Eigen::VectorXd::Zero(len);
        fdr_map[m.name]     = Eigen::VectorXd::Zero(len);
        tpr_map[m.name]     = Eigen::VectorXd::Zero(len);
    }

    auto trex_ctrl = make_trex_control();

    // -------------------------------------------------------------------
    // Method loop
    // -------------------------------------------------------------------
    for (const auto& method : methods) {
        std::cout << std::string(70, '=') << "\n";
        std::cout << "Method: " << method.name << "\n";
        std::cout << std::string(70, '=') << "\n";

        auto screen_ctrl = make_screen_control(method.method, method.bootstrap);

        // SNR loop -------------------------------------------------------
        for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {
            const double snr = snr_values[snr_idx];

            const auto make_data = [=](unsigned seed) -> ScrDGPData {
                return make_ar1_dgp(n, p, p1, rho, snr, beta_val, seed);
            };

            const unsigned base_seed = 42u + static_cast<unsigned>(snr_idx) * 1000u;

            std::ostringstream lbl;
            lbl << method.name << "  SNR=" << std::fixed << std::setprecision(2) << snr;
            const auto result = run_mc_trials_screen_trex(
                num_MC, lbl.str(), make_data, trex_ctrl, screen_ctrl, base_seed);

            const auto idx = static_cast<Eigen::Index>(snr_idx);
            est_fdr_map[method.name](idx) = result.avg_est_fdr;
            fdr_map[method.name](idx)     = result.avg_fdr;
            tpr_map[method.name](idx)     = result.avg_tpr;
        }
    }

    std::ostringstream stem;
    stem << "scr_ar1_snr_n" << n << "_p" << p
         << "_rho" << std::fixed << std::setprecision(2) << rho;
    save_and_print_mc_results(num_MC, stem.str(), snr_values, methods,
                              fdr_map, tpr_map, est_fdr_map);
}


// ==============================================================================
// Part 2: SNR sweep — Ordinary vs. Bootstrap vs. DA-EQUI on equicorrelated data
// ==============================================================================

/**
 * @brief SNR-sweep Monte Carlo: Ordinary vs. Bootstrap DA-EQUI Screen-TRex
 *        on equicorrelated data.  Parallelised via OpenMP.
 *
 * @param num_MC    Number of Monte Carlo runs per (method, SNR) pair.
 * @param rho       Equicorrelation coefficient.
 * @param beta_val  Coefficient magnitude for active variables.
 */
void demo_DA_Equi_MonteCarlo(std::size_t num_MC,
                              double     rho      = 0.4,
                              double     beta_val = 1.0)
{
    std::cout.setf(std::ios::unitbuf);
    cdiag::print_section_header(
        "Part 2: DA-EQUI Screen-TRex Monte Carlo on Equicorrelated Data"
    );

    const Eigen::Index n = 300, p = 1000, p1 = 10;
    std::cout << "DGP: Equicorrelated, n=" << n << ", p=" << p << ", p1=" << p1
              << ", rho=" << rho << "\n";

    const std::vector<double> snr_values =
        {0.01, 0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0};

    // Methods to compare
    const std::vector<ScreenMethodInfo> methods = {
        {"Screen-TRex Ordinary: TLARS", ScreenTRexMethod::TREX, false},
        {"Screen-TRex Bootstrap-CI: TLARS", ScreenTRexMethod::TREX, true},
        {"Screen-TRex Ordinary+DA-EQUI: TLARS", ScreenTRexMethod::TREX_DA_EQUI,
             false},
        {"Screen-TRex Bootstrap-CI+DA-EQUI: TLARS", ScreenTRexMethod::TREX_DA_EQUI,
            true}
    };

    std::map<std::string, Eigen::VectorXd> est_fdr_map, fdr_map, tpr_map;
    for (const auto& m : methods) {
        const auto len = static_cast<Eigen::Index>(snr_values.size());
        est_fdr_map[m.name] = Eigen::VectorXd::Zero(len);
        fdr_map[m.name]     = Eigen::VectorXd::Zero(len);
        tpr_map[m.name]     = Eigen::VectorXd::Zero(len);
    }

    auto trex_ctrl = make_trex_control();

    // -------------------------------------------------------------------
    // Method loop
    // -------------------------------------------------------------------
    for (const auto& method : methods) {
        std::cout << std::string(70, '=') << "\n";
        std::cout << "Method: " << method.name << "\n";
        std::cout << std::string(70, '=') << "\n";

        auto screen_ctrl = make_screen_control(method.method, method.bootstrap);

        // SNR loop -------------------------------------------------------
        for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {
            const double snr = snr_values[snr_idx];

            const auto make_data = [=](unsigned seed) -> ScrDGPData {
                return make_equi_dgp(n, p, p1, rho, snr, beta_val, seed);
            };

            const unsigned base_seed = 42u + static_cast<unsigned>(snr_idx) * 1000u;

            std::ostringstream lbl;
            lbl << method.name << "  SNR=" << std::fixed << std::setprecision(2) << snr;
            const auto result = run_mc_trials_screen_trex(
                num_MC, lbl.str(), make_data, trex_ctrl, screen_ctrl, base_seed);

            const auto idx = static_cast<Eigen::Index>(snr_idx);
            est_fdr_map[method.name](idx) = result.avg_est_fdr;
            fdr_map[method.name](idx)     = result.avg_fdr;
            tpr_map[method.name](idx)     = result.avg_tpr;
        }
    }

    std::ostringstream stem;
    stem << "scr_equi_snr_n" << n << "_p" << p
         << "_rho" << std::fixed << std::setprecision(2) << rho;
    save_and_print_mc_results(num_MC, stem.str(), snr_values, methods,
                              fdr_map, tpr_map, est_fdr_map);
}


// ==============================================================================
// Part 3: rho sweep — DA-AR1 on AR(1)-correlated data
// ==============================================================================

/**
 * @brief rho-sweep Monte Carlo: Ordinary vs. Bootstrap vs. DA-AR1 Screen-TRex
 *        on AR(1)-correlated data at fixed SNR.  Parallelised via OpenMP.
 *
 * @param num_MC    Number of Monte Carlo runs per (method, rho) pair.
 * @param snr       Fixed signal-to-noise ratio.
 * @param beta_val  Coefficient magnitude for active variables.
 */
void demo_DA_AR1_RhoSweep_MonteCarlo(std::size_t num_MC,
                                     double      snr      = 1.0,
                                     double      beta_val = 1.0)
{
    std::cout.setf(std::ios::unitbuf);
    cdiag::print_section_header(
        "Part 3: DA-AR1 Screen-TRex rho Sweep on AR(1) Data"
    );

    const Eigen::Index n = 300, p = 1000, p1 = 10;
    std::cout << "DGP: AR(1), n=" << n << ", p=" << p << ", p1=" << p1
              << ", SNR=" << snr << "\n";

    const std::vector<double> rho_values =
        {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    const std::vector<ScreenMethodInfo> methods = {
        {"Screen-TRex Ordinary: TLARS",         ScreenTRexMethod::TREX,
            false},
        {"Screen-TRex Bootstrap-CI: TLARS",        ScreenTRexMethod::TREX,
            true },
        {"Screen-TRex Ordinary+DA-AR1: TLARS",  ScreenTRexMethod::TREX_DA_AR1,
            false},
        {"Screen-TRex Bootstrap-CI+DA-AR1: TLARS", ScreenTRexMethod::TREX_DA_AR1,
            true }
    };

    std::map<std::string, Eigen::VectorXd> est_fdr_map, fdr_map, tpr_map;
    for (const auto& m : methods) {
        const auto len = static_cast<Eigen::Index>(rho_values.size());
        est_fdr_map[m.name] = Eigen::VectorXd::Zero(len);
        fdr_map[m.name]     = Eigen::VectorXd::Zero(len);
        tpr_map[m.name]     = Eigen::VectorXd::Zero(len);
    }

    auto trex_ctrl = make_trex_control();

    for (const auto& method : methods) {
        std::cout << std::string(70, '=') << "\n";
        std::cout << "Method: " << method.name << "\n";
        std::cout << std::string(70, '=') << "\n";

        auto screen_ctrl = make_screen_control(method.method, method.bootstrap);

        for (std::size_t rho_idx = 0; rho_idx < rho_values.size(); ++rho_idx) {
            const double rho = rho_values[rho_idx];

            const auto make_data = [=](unsigned seed) -> ScrDGPData {
                return make_ar1_dgp(n, p, p1, rho, snr, beta_val, seed);
            };

            const unsigned base_seed = 42u + static_cast<unsigned>(rho_idx) * 1000u;

            std::ostringstream lbl;
            lbl << method.name << "  rho=" << std::fixed << std::setprecision(2) << rho;
            const auto result = run_mc_trials_screen_trex(
                num_MC, lbl.str(), make_data, trex_ctrl, screen_ctrl, base_seed);

            const auto idx = static_cast<Eigen::Index>(rho_idx);
            est_fdr_map[method.name](idx) = result.avg_est_fdr;
            fdr_map[method.name](idx)     = result.avg_fdr;
            tpr_map[method.name](idx)     = result.avg_tpr;
        }
    }

    std::ostringstream stem;
    stem << "scr_ar1_rho_n" << n << "_p" << p
         << "_snr" << std::fixed << std::setprecision(2) << snr;
    save_and_print_mc_results(num_MC, stem.str(), rho_values, methods,
                              fdr_map, tpr_map, est_fdr_map, "rho");
}


// ==============================================================================
// Part 4: rho sweep — DA-EQUI on equicorrelated data
// ==============================================================================

/**
 * @brief rho-sweep Monte Carlo: Ordinary vs. Bootstrap vs. DA-EQUI Screen-TRex
 *        on equicorrelated data at fixed SNR.  Parallelised via OpenMP.
 *
 * @param num_MC    Number of Monte Carlo runs per (method, rho) pair.
 * @param snr       Fixed signal-to-noise ratio.
 * @param beta_val  Coefficient magnitude for active variables.
 */
void demo_DA_Equi_RhoSweep_MonteCarlo(
    std::size_t num_MC,
    double      snr      = 1.0,
    double      beta_val = 1.0)
{
    std::cout.setf(std::ios::unitbuf);
    cdiag::print_section_header(
        "Part 4: DA-EQUI Screen-TRex rho Sweep on Equicorrelated Data"
    );

    const Eigen::Index n = 300, p = 1000, p1 = 10;
    std::cout << "DGP: Equicorrelated, n=" << n << ", p=" << p << ", p1=" << p1
              << ", SNR=" << snr << "\n";

    const std::vector<double> rho_values =
        {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    const std::vector<ScreenMethodInfo> methods = {
        {"Screen-TRex Ordinary: TLARS",          ScreenTRexMethod::TREX,
            false},
        {"Screen-TRex Bootstrap-CI: TLARS",         ScreenTRexMethod::TREX,
            true },
        {"Screen-TRex Ordinary+DA-EQUI: TLARS",  ScreenTRexMethod::TREX_DA_EQUI,
             false},
        {"Screen-TRex Bootstrap-CI+DA-EQUI: TLARS", ScreenTRexMethod::TREX_DA_EQUI,
             true }
    };

    std::map<std::string, Eigen::VectorXd> est_fdr_map, fdr_map, tpr_map;
    for (const auto& m : methods) {
        const auto len = static_cast<Eigen::Index>(rho_values.size());
        est_fdr_map[m.name] = Eigen::VectorXd::Zero(len);
        fdr_map[m.name]     = Eigen::VectorXd::Zero(len);
        tpr_map[m.name]     = Eigen::VectorXd::Zero(len);
    }

    auto trex_ctrl = make_trex_control();

    for (const auto& method : methods) {
        std::cout << std::string(70, '=') << "\n";
        std::cout << "Method: " << method.name << "\n";
        std::cout << std::string(70, '=') << "\n";

        auto screen_ctrl = make_screen_control(method.method, method.bootstrap);

        for (std::size_t rho_idx = 0; rho_idx < rho_values.size(); ++rho_idx) {
            const double rho = rho_values[rho_idx];

            const auto make_data = [=](unsigned seed) -> ScrDGPData {
                return make_equi_dgp(n, p, p1, rho, snr, beta_val, seed);
            };

            const unsigned base_seed = 42u + static_cast<unsigned>(rho_idx) * 1000u;

            std::ostringstream lbl;
            lbl << method.name << "  rho=" << std::fixed << std::setprecision(2) << rho;
            const auto result = run_mc_trials_screen_trex(
                num_MC, lbl.str(), make_data, trex_ctrl, screen_ctrl, base_seed);

            const auto idx = static_cast<Eigen::Index>(rho_idx);
            est_fdr_map[method.name](idx) = result.avg_est_fdr;
            fdr_map[method.name](idx)     = result.avg_fdr;
            tpr_map[method.name](idx)     = result.avg_tpr;
        }
    }

    std::ostringstream stem;
    stem << "scr_equi_rho_n" << n << "_p" << p
         << "_snr" << std::fixed << std::setprecision(2) << snr;
    save_and_print_mc_results(num_MC, stem.str(), rho_values, methods,
                              fdr_map, tpr_map, est_fdr_map, "rho");
}


// ==============================================================================
// Part 5: rho sweep — DA-BLOCK-EQUI on block-equicorrelated data
// ==============================================================================

/**
 * @brief rho-sweep Monte Carlo: Ordinary vs. Bootstrap vs. DA-BLOCK-EQUI
 *        Screen-TRex on block-equicorrelated data at fixed SNR.
 *        Parallelised via OpenMP.
 *
 * @param num_MC    Number of Monte Carlo runs per (method, rho) pair.
 * @param n_blocks  Number of blocks in DGP.
 * @param snr       Fixed signal-to-noise ratio.
 * @param beta_val  Coefficient magnitude for active variables.
 */
void demo_DA_BlockEqui_RhoSweep_MonteCarlo(
    std::size_t num_MC,
    std::size_t n_blocks = 5,
    double      snr      = 1.0,
    double      beta_val = 1.0)
{
    std::cout.setf(std::ios::unitbuf);
    cdiag::print_section_header(
        "Part 5: DA-BLOCK-EQUI Screen-TRex rho Sweep on Block-Equicorr Data"
    );

    const Eigen::Index n = 300, p = 1000, p1 = 10;
    std::cout << "DGP: Block-Equicorr, n=" << n << ", p=" << p << ", p1=" << p1
              << ", n_blocks=" << n_blocks << ", SNR=" << snr << "\n";

    const std::vector<double> rho_values =
        {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    const std::vector<ScreenMethodInfo> methods = {
        {"Screen-TRex Ordinary: TLARS",          ScreenTRexMethod::TREX,
            false},
        {"Screen-TRex Bootstrap-CI: TLARS",         ScreenTRexMethod::TREX,
            true },
        {"Screen-TRex Ordinary+DA-BLOCK-EQUI: TLARS", ScreenTRexMethod::TREX_DA_BLOCK_EQUI,
            false},
        {"Screen-TRex Bootstrap-CI+DA-BLOCK-EQUI: TLARS", ScreenTRexMethod::TREX_DA_BLOCK_EQUI,
            true }
    };

    std::map<std::string, Eigen::VectorXd> est_fdr_map, fdr_map, tpr_map;
    for (const auto& m : methods) {
        const auto len = static_cast<Eigen::Index>(rho_values.size());
        est_fdr_map[m.name] = Eigen::VectorXd::Zero(len);
        fdr_map[m.name]     = Eigen::VectorXd::Zero(len);
        tpr_map[m.name]     = Eigen::VectorXd::Zero(len);
    }

    auto trex_ctrl = make_trex_control();

    for (const auto& method : methods) {
        std::cout << std::string(70, '=') << "\n";
        std::cout << "Method: " << method.name << "\n";
        std::cout << std::string(70, '=') << "\n";

        auto screen_ctrl = make_screen_control(method.method, method.bootstrap,
                                               n_blocks);

        for (std::size_t rho_idx = 0; rho_idx < rho_values.size(); ++rho_idx) {
            const double rho = rho_values[rho_idx];

            const auto make_data = [=](unsigned seed) -> ScrDGPData {
                return make_block_equi_dgp(n, p, p1, rho, n_blocks, snr, beta_val, seed);
            };

            const unsigned base_seed = 42u + static_cast<unsigned>(rho_idx) * 1000u;

            std::ostringstream lbl;
            lbl << method.name << "  rho=" << std::fixed << std::setprecision(2) << rho;
            const auto result = run_mc_trials_screen_trex(
                num_MC, lbl.str(), make_data, trex_ctrl, screen_ctrl, base_seed);

            const auto idx = static_cast<Eigen::Index>(rho_idx);
            est_fdr_map[method.name](idx) = result.avg_est_fdr;
            fdr_map[method.name](idx)     = result.avg_fdr;
            tpr_map[method.name](idx)     = result.avg_tpr;
        }
    }

    std::ostringstream stem;
    stem << "scr_block_equi_rho_n" << n << "_p" << p
         << "_blocks" << n_blocks
         << "_snr" << std::fixed << std::setprecision(2) << snr;
    save_and_print_mc_results(num_MC, stem.str(), rho_values, methods,
                              fdr_map, tpr_map, est_fdr_map, "rho");
}


// ==============================================================================
// Main
// ==============================================================================

int main()
{
    std::cout.setf(std::ios::unitbuf);
    omp_set_num_threads(trex_mc_workers::count());

    // Part 1: SNR sweep — Ordinary vs. Bootstrap vs. DA-AR1 on AR(1) data
    demo_DA_AR1_MonteCarlo(200, 0.5, 1.0);

    // Part 2: SNR sweep — Ordinary vs. Bootstrap vs. DA-EQUI on equicorrelated data
    demo_DA_Equi_MonteCarlo(200, 0.4, 3.0);

    // Part 3: rho sweep — Ordinary vs. Bootstrap vs. DA-AR1 on AR(1) data
    demo_DA_AR1_RhoSweep_MonteCarlo(200, 1.0, 1.0);

    // Part 4: rho sweep — Ordinary vs. Bootstrap vs. DA-EQUI on equicorrelated data
    demo_DA_Equi_RhoSweep_MonteCarlo(200, 1.0, 3.0);

    // Part 5: rho sweep — Ordinary vs. Bootstrap vs. DA-BLOCK-EQUI on block-equicorr data
    demo_DA_BlockEqui_RhoSweep_MonteCarlo(200, 5, 1.0, 3.0);

    return 0;
}
