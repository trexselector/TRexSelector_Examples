// ==============================================================================
// demo_trex_gvs_07_mc_sim_arma_blocks.cpp
// ==============================================================================
/**
 * @file demo_trex_gvs_07_mc_sim_arma_blocks.cpp
 *
 * @brief MC simulation: T-Rex+GVS on heterogeneous ARMA blocks.
 *
 * @details
 * One part:
 *   1. 2-D SNR × ar_coef sweep
 *      (subsumes the former 1-D SNR and ar_coef sweeps as its ar_coef = 0.8
 *       column and SNR = 2 row)
 *
 * DGP:
 *   make_arma_blocks_dgp()
 *   - 4 blocks of sizes {20, 50, 80, 65}
 *   - heterogeneous within-block ARMA structure
 *   - blocks 0, 1, 2 active with beta = 3
 *   - block 3 is an inactive AR(1) trap
 *   - support size s = 150
 *
 * Shared selector settings:
 *   - K = 20
 *   - tFDR = 0.1
 *   - corr_max = 0.98
 *   - single-linkage HAC
 *   - lambda2 = CV_1SE_CCD (EN rows), CV_1SE_IEN_CCD (IEN rows)
 *
 * Methods compared:
 *   - TENET
 *   - TENET_AUG
 *   - TIENET_AUG
 *
 * MC:
 *   - 200 trials per grid point
 */
// ==============================================================================

#include "trex_gvs_dgps.hpp"
#include "trex_gvs_simulation_utils.hpp"

// ==============================================================================

using namespace gvs_demo;

// =============================================================================
// Preset and method bundle
// =============================================================================

struct ARMABlocksPreset {
    std::string scenario_tag;
    std::string file_stem_prefix;

    std::vector<double> snr_grid_2d;
    std::vector<double> ar_coef_grid_2d;
};

struct ARMABlocksMethodSet {
    TRexControlParameter trex_ctrl;
    TRexGVSControlParameter en;
    TRexGVSControlParameter en_aug;
    TRexGVSControlParameter ien;
};

// =============================================================================
// Helpers
// =============================================================================

static ARMABlocksPreset make_arma_blocks_preset()
{
    ARMABlocksPreset p;
    p.scenario_tag = "ARMA-Blocks";
    p.file_stem_prefix = "gvs_arma_blocks";

    p.snr_grid_2d = {0.2, 0.5, 1.0, 2.0, 5.0};

    p.ar_coef_grid_2d = {
        0.30, 0.50, 0.70, 0.80,
        0.90, 0.95
    };

    return p;
}

static ARMABlocksMethodSet make_method_set(const GVSSimConfig& cfg)
{
    ARMABlocksMethodSet ms;
    ms.trex_ctrl = make_gvs_trex_control(cfg.K);

    ms.en.gvs_type = GVSType::EN;
    ms.en.lambda2_method = LambdaSelectionMethod::CV_1SE_CCD;
    ms.en.corr_max = cfg.corr_max;
    ms.en.hc_linkage = hac::LinkageMethod::Single;
    ms.en.en_solver = ENSolverType::TENET;

    ms.en_aug = ms.en;
    ms.en_aug.en_solver = ENSolverType::TENET_AUG;

    ms.ien.gvs_type = GVSType::IEN;
    ms.ien.lambda2_method = LambdaSelectionMethod::CV_1SE_IEN_CCD;
    ms.ien.cv_dfmax_ratio = 1.0;  // support cap 180 > s = 150 (library default 0.8 gives 144)
    ms.ien.corr_max = cfg.corr_max;
    ms.ien.hc_linkage = hac::LinkageMethod::Single;

    return ms;
}

// =============================================================================
// Part 1 — 2-D SNR × ar_coef sweep
// =============================================================================

/**
 * @brief Run a 2-D MC sweep over SNR and ar_coef on the ARMA-blocks DGP.
 */
static void run_arma_part1_2d_sweep(
    const GVSSimConfig& cfg,
    const ARMABlocksPreset& preset,
    const ARMABlocksMethodSet& ms)
{
    GVS2DDGPFactory dgp_2d = [&cfg](double snr, double ar_coef, unsigned seed) {
        return make_arma_blocks_dgp(cfg.n, cfg.p, snr, ar_coef, seed);
    };

    cdiag::print_section_header(
        "Part 1: 2-D SNR x ar_coef Sweep | " + preset.scenario_tag);
    std::cout << "  n=" << cfg.n << "  p=" << cfg.p << "  s=150"
              << "  MC=" << cfg.num_MC << "  tFDR=" << fmt_num(cfg.tFDR)
              << "  K=" << cfg.K << "  corr_max=" << fmt_num(cfg.corr_max) << "\n"
              << "  methods: TENET / TENET_AUG / TIENET_AUG"
              << "  |  lambda2=CV_1SE_CCD (EN), CV_1SE_IEN_CCD (IEN)  linkage=Single\n"
              << "  snr_grid: {";
    for (std::size_t i = 0; i < preset.snr_grid_2d.size(); ++i)
        std::cout << (i ? ", " : "") << fmt_num(preset.snr_grid_2d[i]);
    std::cout << "}\n  ar_coef_grid: {";
    for (std::size_t i = 0; i < preset.ar_coef_grid_2d.size(); ++i)
        std::cout << (i ? ", " : "") << fmt_num(preset.ar_coef_grid_2d[i]);
    std::cout << "}\n";

    auto en = run_gvs_2d_sweep(
        dgp_2d,
        preset.snr_grid_2d,
        preset.ar_coef_grid_2d,
        cfg,
        ms.en,
        ms.trex_ctrl,
        "TENET");

    auto en_aug = run_gvs_2d_sweep(
        dgp_2d,
        preset.snr_grid_2d,
        preset.ar_coef_grid_2d,
        cfg,
        ms.en_aug,
        ms.trex_ctrl,
        "TENET_AUG");

    auto ien = run_gvs_2d_sweep(
        dgp_2d,
        preset.snr_grid_2d,
        preset.ar_coef_grid_2d,
        cfg,
        ms.ien,
        ms.trex_ctrl,
        "TIENET_AUG");

    std::vector<std::string> snr_labels, ar_labels;
    for (double s : preset.snr_grid_2d)
        snr_labels.push_back("snr=" + fmt_num(s));
    for (double a : preset.ar_coef_grid_2d)
        ar_labels.push_back("ac=" + fmt_num(a));

    print_mc_matrix("mean_FDP [TENET]",     snr_labels, ar_labels, en,     false);
    print_mc_matrix("mean_TPP [TENET]",     snr_labels, ar_labels, en,     true);
    print_mc_matrix("mean_FDP [TENET_AUG]", snr_labels, ar_labels, en_aug, false);
    print_mc_matrix("mean_TPP [TENET_AUG]", snr_labels, ar_labels, en_aug, true);
    print_mc_matrix("mean_FDP [TIENET_AUG]",    snr_labels, ar_labels, ien,    false);
    print_mc_matrix("mean_TPP [TIENET_AUG]",    snr_labels, ar_labels, ien,    true);

    save_mc_2d_tables(
        preset.scenario_tag,
        snr_labels,
        ar_labels,
        en,
        en_aug,
        ien,
        cfg,
        preset.file_stem_prefix + "_2d");
}

// =============================================================================
// Top-level driver
// =============================================================================

/**
 * @brief Run the heterogeneous ARMA-blocks benchmark (2-D SNR × ar_coef sweep).
 */
static void run_arma_blocks_benchmark(
    const GVSSimConfig& cfg,
    const ARMABlocksPreset& preset)
{
    const auto ms = make_method_set(cfg);

    cdiag::print_section_header(
        "ARMA-blocks benchmark | " + preset.scenario_tag +
        "\nDGP = make_arma_blocks_dgp"
        "\nblocks = {20, 50, 80, 65}, s = 150"
        "\nactive blocks = {0, 1, 2}, trap block = 3");

    run_arma_part1_2d_sweep(cfg, preset, ms);

    std::cout << preset.scenario_tag << " GVS MC simulations complete.\n";
}

// =============================================================================
// main
// =============================================================================

int main()
{

    // ==========================================================================
    //  Simulation parameters
    // ==========================================================================

    std::cout.setf(std::ios::unitbuf);
    omp_set_num_threads(trex_mc_workers::count());
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    GVSSimConfig cfg;
    cfg.n         = 200;
    cfg.p         = 500;
    cfg.tFDR      = 0.1;
    cfg.K         = 20;
    cfg.num_MC    = num_mc(200);
    cfg.base_seed = 2026;
    cfg.corr_max  = 0.98;


    // ==========================================================================
    //  Run simulations
    // ==========================================================================

    if (true) run_arma_blocks_benchmark(cfg, make_arma_blocks_preset());

    std::cout << "All ARMA-block benchmark simulations complete.\n";
    return 0;
}
