// ==============================================================================
// demo_trex_gvs_04_mc_sim_neg_traps.cpp
// ==============================================================================
/**
 * @file demo_trex_gvs_04_mc_sim_neg_traps.cpp
 *
 * @brief MC simulation: T-Rex+GVS on the negative-traps DGP.
 *  This scenario evaluates recovery of an active negatively correlated group
 *  in the presence of two spatially separated inactive trap groups with the
 *  same sign-flipped correlation structure.
 *
 * @details
 *  One part:
 *    sim_neg_traps_part1() — 2-D SNR × rho grid, sd_x = sqrt((1 - rho)/rho)
 *    (subsumes the former 1-D SNR and rho sweeps as its rho = 0.99 column and
 *     SNR = 2 row)
 *
 *  DGP: active group (0–99) + Trap1 (100–199) + noise (200–299) +
 *       Trap2 (300–359) + noise (360+); s = 100.
 *  In each correlated group, one half loads positively and the other half
 *  negatively on a latent factor, inducing strong within-group sign structure.
 *  Selector: K=20, tFDR=0.1, corr_max=0.98, lambda2=CV_1SE_CCD (EN), CV_1SE_IEN_CCD (IEN).
 *  Methods compared: TENET, TENET_AUG, TIENET_AUG.
 *  MC: 200 trials per grid point.
 */
// ==============================================================================

// project includes
#include "trex_gvs_dgps.hpp"
#include "trex_gvs_simulation_utils.hpp"

// ==============================================================================

using namespace gvs_demo;

// ==============================================================================
//  Part 1 — 2-D SNR × rho sweep
// ==============================================================================

/**
 * @brief Run a 2-D MC sweep over SNR and rho on the negative-traps DGP.
 *
 * Evaluates TENET, TENET_AUG, and TIENET_AUG on an SNR × rho grid, then
 * prints and saves matrix summaries for mean TPP and mean FDP.
 */

static void sim_neg_traps_part1(const GVSSimConfig& cfg, const std::string& tag)
{
    const std::vector<double> snr_grid_2d = {0.2, 0.5, 1.0, 2.0, 5.0};
    const std::vector<double> rho_grid_2d = {
        0.30, 0.50, 0.70, 0.90, 0.95, 0.99};

    GVS2DDGPFactory dgp_2d = [&cfg](double snr, double rho, unsigned seed) {
        const double sd_x = std::sqrt((1.0 - rho) / rho);
        return make_neg_traps_dgp(cfg.n, cfg.p, snr, sd_x, seed);
    };

    auto trex_ctrl = make_gvs_trex_control(cfg.K);

    TRexGVSControlParameter gvs_ctrl_en;
    gvs_ctrl_en.gvs_type       = GVSType::EN;
    gvs_ctrl_en.lambda2_method = LambdaSelectionMethod::CV_1SE_CCD;
    gvs_ctrl_en.corr_max       = cfg.corr_max;
    gvs_ctrl_en.hc_linkage     = hac::LinkageMethod::Single;
    gvs_ctrl_en.en_solver      = ENSolverType::TENET;

    TRexGVSControlParameter gvs_ctrl_en_aug;
    gvs_ctrl_en_aug.gvs_type       = GVSType::EN;
    gvs_ctrl_en_aug.lambda2_method = LambdaSelectionMethod::CV_1SE_CCD;
    gvs_ctrl_en_aug.corr_max       = cfg.corr_max;
    gvs_ctrl_en_aug.hc_linkage     = hac::LinkageMethod::Single;
    gvs_ctrl_en_aug.en_solver      = ENSolverType::TENET_AUG;

    TRexGVSControlParameter gvs_ctrl_ien;
    gvs_ctrl_ien.gvs_type       = GVSType::IEN;
    gvs_ctrl_ien.lambda2_method = LambdaSelectionMethod::CV_1SE_IEN_CCD;
    gvs_ctrl_ien.cv_dfmax_ratio = 1.0;  // support cap 180 as in the s = 150 demos (library default 0.8 gives 144)
    gvs_ctrl_ien.corr_max       = cfg.corr_max;
    gvs_ctrl_ien.hc_linkage     = hac::LinkageMethod::Single;

    cdiag::print_section_header("Part 1: 2-D SNR x rho Sweep  |  " + tag);
    std::cout << "  n=" << cfg.n << "  p=" << cfg.p << "  s=100"
              << "  MC=" << cfg.num_MC << "  tFDR=" << fmt_num(cfg.tFDR)
              << "  K=" << cfg.K << "  corr_max=" << fmt_num(cfg.corr_max) << "\n"
              << "  methods: TENET / TENET_AUG / TIENET_AUG"
              << "  |  lambda2=CV_1SE_CCD (EN), CV_1SE_IEN_CCD (IEN)  linkage=Single\n"
              << "  snr_grid: {";
    for (std::size_t i = 0; i < snr_grid_2d.size(); ++i)
        std::cout << (i ? ", " : "") << fmt_num(snr_grid_2d[i]);
    std::cout << "}\n  rho_grid: {";
    for (std::size_t i = 0; i < rho_grid_2d.size(); ++i)
        std::cout << (i ? ", " : "") << fmt_num(rho_grid_2d[i]);
    std::cout << "}\n";

    auto en_2d     = run_gvs_2d_sweep(dgp_2d, snr_grid_2d, rho_grid_2d, cfg,
                                       gvs_ctrl_en,     trex_ctrl, "TENET");
    auto en_aug_2d = run_gvs_2d_sweep(dgp_2d, snr_grid_2d, rho_grid_2d, cfg,
                                       gvs_ctrl_en_aug, trex_ctrl, "TENET_AUG");
    auto ien_2d    = run_gvs_2d_sweep(dgp_2d, snr_grid_2d, rho_grid_2d, cfg,
                                       gvs_ctrl_ien,    trex_ctrl, "TIENET_AUG");

    std::vector<std::string> snr_labels, rho_labels;
    for (double s : snr_grid_2d)
        snr_labels.push_back("snr=" + fmt_num(s));
    for (double r : rho_grid_2d)
        rho_labels.push_back("rho=" + fmt_num(r));

    print_mc_matrix("mean_TPP [TENET]",     snr_labels,
                    rho_labels, en_2d,     true);
    print_mc_matrix("mean_FDP [TENET]",     snr_labels,
                    rho_labels, en_2d,     false);
    print_mc_matrix("mean_TPP [TENET_AUG]", snr_labels,
                    rho_labels, en_aug_2d, true);
    print_mc_matrix("mean_FDP [TENET_AUG]", snr_labels,
                    rho_labels, en_aug_2d, false);
    print_mc_matrix("mean_TPP [TIENET_AUG]",    snr_labels,
                    rho_labels, ien_2d,    true);
    print_mc_matrix("mean_FDP [TIENET_AUG]",    snr_labels,
                    rho_labels, ien_2d,    false);
    save_mc_2d_tables(tag, snr_labels, rho_labels,
                      en_2d, en_aug_2d, ien_2d, cfg, "gvs_" + tag + "_2d");
}


// ==============================================================================
//  main
// ==============================================================================

int main()
{
    std::cout.setf(std::ios::unitbuf);
    omp_set_num_threads(trex_mc_workers::count());
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    // ==========================================================================
    //  Simulation parameters
    // ==========================================================================

    GVSSimConfig cfg;
    cfg.n         = 200;
    cfg.p         = 500;
    cfg.tFDR      = 0.1;
    cfg.K         = 20;
    cfg.num_MC    = num_mc(200);
    cfg.base_seed = 2026;
    cfg.corr_max  = 0.98;

    const std::string scenario_tag = "Neg-Traps";

    // ==========================================================================
    //  Run simulations
    // ==========================================================================

    if (true) sim_neg_traps_part1(cfg, scenario_tag);

    std::cout << scenario_tag << " GVS MC simulations complete.\n";
    return 0;
}
