// ==============================================================================
// demo_trex_da_08_mc_sim_equi_and_bt.cpp
// ==============================================================================
/**
 * @file demo_trex_da_08_mc_sim_equi_and_bt.cpp
 *
 * @brief DA-TRex+EQUI Monte Carlo 2D sweep (SNR x rho) for the equicorrelated DGP.
 *
 * @details
 *
 *  One 2D sweep: for each SNR column in {0.5, 1, 2, 5}, run a full rho sweep
 *  over {0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.7, 0.9} (dense near 0 to locate the
 *  DA suppression cliff) with the Equi DGP (n=300, p=1000, Random support),
 *  3 DA solvers + the 3 base T-Rex comparison rows.
 *
 *  Rationale: equicorrelation ties every predictor to one shared factor, so
 *  DA-EQUI suppresses all candidates as soon as rho is noticeably above 0,
 *  independent of signal strength, while base T-Rex loses FDR control at the
 *  same tiny rho. The 2D layout shows the cliff location, its SNR
 *  independence, and the rho=0 working anchor in a single figure.
 *
 *  Earlier revisions of this demo also carried BT (hierarchical-block) SNR
 *  and linkage sweeps; those were dropped as redundant — the BT method is
 *  covered on better-suited block designs in Demos 02-05.
 *
 * Results are saved to simulation_results/data/ as TXT and CSV files
 * (one file pair per SNR column).
 */
// ==============================================================================

#include "trex_da_sim_common.hpp"

// ==============================================================================

using namespace da_sim;

// ==============================================================================
//  2D sweep: SNR (outer, one output file per value) x rho (inner sweep)
// ==============================================================================

void demo_equi_mc_2d_snr_rho_sweep()
{
    constexpr int    n         = 300;
    constexpr int    p         = 1000;
    constexpr int    s         = 10;
    constexpr int    K         = 20;
    constexpr int    num_MC    = 200;
    constexpr double tFDR      = 0.2;
    constexpr int    base_seed = 2026;
    const auto       policy    = SupportPolicy::Random;

    const std::vector<double> snr_grid = {0.5, 1.0, 2.0, 5.0};
    const std::vector<double> rho_grid = {
        0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.7, 0.9
    };

    TRexDAControlParameter da_ctrl;
    da_ctrl.method = DAMethod::EQUI;

    for (std::size_t si = 0; si < snr_grid.size(); ++si) {
        const double snr = snr_grid[si];

        // Filesystem-safe SNR tag: 0.5 -> "0p5", 2.0 -> "2".
        std::string snr_tag = fmt_num(snr);
        std::replace(snr_tag.begin(), snr_tag.end(), '.', 'p');

        // Disjoint seed blocks per SNR column (inner sweep spans
        // gi * 10000), so the columns are statistically independent.
        const int col_seed = base_seed + static_cast<int>(si) * 1000000;

        run_param_sweep(
            "da_equi_rho_snr" + snr_tag,
            "Rho",
            rho_grid,
            num_MC,
            tFDR,
            col_seed,
            default_solvers("+EQUI"),
            [&](double rho, unsigned seed) {
                return dgp_equi(n, p,
                                make_support(policy, s, p, 0, seed),
                                /*amplitude=*/3.0,
                                rho, snr, seed);
            },
            da_ctrl,
            make_trex_control(K),
            "Equi 2D sweep, SNR=" + fmt_num(snr)
            + ", n=" + std::to_string(n)
            + ", p=" + std::to_string(p)
            + ", support=" + support_policy_label(policy),
            /*include_base_trex=*/true);
    }
}


// ==============================================================================
//  main
// ==============================================================================

int main() {

    std::cout.setf(std::ios::unitbuf);
    omp_set_num_threads(trex_mc_workers::count());
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    demo_equi_mc_2d_snr_rho_sweep();

    std::cout << "\nEqui 2D (SNR x rho) MC simulation complete.\n";
    return 0;
}
