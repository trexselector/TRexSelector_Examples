// ==============================================================================
// demo_trex_da_06_mc_sim_groups.cpp
// ==============================================================================
/**
 * @file demo_trex_da_06_mc_sim_groups.cpp
 *
 * @brief DA-TRex MC simulation for the prior-groups DGP.
 *
 * @details
 *  Functions:
 *   demo_groups_mc_snr_sweep() -- SNR sweep (n=300, p=1000, s=10, 200 MC, 3 solvers).
 */
// ==============================================================================

#include "trex_da_sim_common.hpp"

using namespace da_sim;


// ==============================================================================
// Group structure builder
// ==============================================================================

/**
 * @brief Build a 3-level hierarchical group structure for p predictors.
 *
 * Level 1: groups of 10   (p/10 groups)
 * Level 2: groups of 50   (p/50 groups)
 * Level 3: groups of 250  (p/250 groups)
 */
std::vector<std::vector<Eigen::Index>> make_group_structure(int p) {
    std::vector<std::vector<Eigen::Index>> groups(3);
    for (Eigen::Index j = 0; j < p; ++j) {
        groups[0].push_back(j / 10);
        groups[1].push_back(j / 50);
        groups[2].push_back(j / 250);
    }
    return groups;
}

namespace {
    constexpr int max_gap = 20;
}


// ==============================================================================
//  MC SNR sweep
//  DGP: dgp_groups_toeplitz_leaf() with 3-level hierarchy, Random support.
//  Fixed: n=300, p=1000, s=10 (SimConfig defaults), tFDR=0.2, K=20, num_MC=200.
//  Swept: SNR in {0.1, 0.2, 0.5, 1.0, 2.0, 5.0}.
// ==============================================================================
void demo_groups_mc_snr_sweep()
{
    SimConfig cfg;  // n=300, p=1000, s=10
    const auto policy = SupportPolicy::Random;

    auto groups = make_group_structure(cfg.p);

     // cumulative variance masses, fine -> coarse
    const std::vector<double> rho_levels = {0.55, 0.25, 0.10};
    constexpr double phi_leaf = 0.50;

    // The prior groups act as merge CONSTRAINTS: the selector sub-clusters
    // within their finest common refinement (the groups of 10) and calibrates
    // over a data-driven rho grid ending at the conservative rho = 1 anchor.
    TRexDAControlParameter da_ctrl;
    da_ctrl.prior_groups = groups;

    run_snr_sweep(
        "da_groups_toeplitz_leaf",
        {0.1, 0.2, 0.5, 1.0, 2.0, 5.0},
        cfg.num_MC,
        cfg.tFDR,
        cfg.base_seed,
        default_solvers(),
        [&](double snr, unsigned seed) {
            return dgp_groups_toeplitz_leaf(
                cfg.n,
                cfg.p,
                make_support(policy, cfg.s, cfg.p, max_gap, seed),
                cfg.amplitude,
                groups,
                rho_levels,
                phi_leaf,
                snr,
                seed
            );
        },
        da_ctrl,
        make_trex_control(cfg.K),
        "Prior Groups + Toeplitz leaf, n=" + std::to_string(cfg.n)
        + ", p=" + std::to_string(cfg.p)
        + ", support=" + support_policy_label(policy)
        + ", phi_leaf=" + fmt_num(phi_leaf)
        + ", max_gap=" + std::to_string(max_gap));
}


// ==============================================================================
// main
// ==============================================================================
int main() {
    std::cout.setf(std::ios::unitbuf);
    omp_set_num_threads(trex_mc_workers::count());
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    if (true) demo_groups_mc_snr_sweep();

    std::cout << "\nPrior Groups MC simulations complete.\n";
    return 0;
}
