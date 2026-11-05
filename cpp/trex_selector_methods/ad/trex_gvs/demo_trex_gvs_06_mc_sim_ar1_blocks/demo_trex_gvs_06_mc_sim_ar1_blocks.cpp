// ==============================================================================
// demo_trex_gvs_06_mc_sim_ar1_blocks.cpp
// ==============================================================================
/**
 * @file demo_trex_gvs_06_mc_sim_ar1_blocks.cpp
 *
 * @brief MC simulation: T-Rex+GVS on AR(1)-within-block DGP.
 *
 * @details
 * This benchmark uses four grouped blocks of sizes {20, 50, 80, 65} with
 * AR(1) covariance structure within each block. Blocks 0, 1, and 2 are active
 * with beta = 3, while block 3 is an inactive AR(1) trap block.
 *
 * One part:
 * 1. 2-D SNR × rho sweep
 *    (subsumes the former 1-D SNR and rho sweeps as its rho = 0.85 column and
 *     SNR = 2 row)
 *
 * DGP:
 * - make_ar1_blocks_dgp()
 * - within-block covariance Sigma(i,j) = rho^|i-j|
 * - block sizes = {20, 50, 80, 65}
 * - active blocks = {0, 1, 2}
 * - inactive trap block = {3}
 * - support size s = 150
 *
 * Shared selector settings:
 * - K = 20
 * - tFDR = 0.1
 * - corr_max = 0.98
 * - single-linkage HAC
 * - lambda2 = CV_1SE_CCD (EN rows), CV_1SE_IEN_CCD (IEN rows)
 *
 * Methods compared:
 * - TENET
 * - TENET_AUG
 * - TIENET_AUG
 *
 * MC:
 * - 200 trials per grid point
 */
// ==============================================================================

#include "trex_gvs_dgps.hpp"
#include "trex_gvs_simulation_utils.hpp"

// ==============================================================================

using namespace gvs_demo;

// =============================================================================
// Preset and method bundle
// =============================================================================

struct AR1BlocksPreset {
  std::string scenario_tag;
  std::string file_stem_prefix;

  std::vector<double> snr_grid_2d;
  std::vector<double> rho_grid_2d;
};

struct AR1BlocksMethodSet {
  TRexControlParameter trex_ctrl;
  TRexGVSControlParameter en;
  TRexGVSControlParameter en_aug;
  TRexGVSControlParameter ien;
};

// =============================================================================
// Preset builder
// =============================================================================

static AR1BlocksPreset make_ar1_blocks_preset() {
  AR1BlocksPreset p;
  p.scenario_tag = "AR1-Blocks";
  p.file_stem_prefix = "gvs_ar1_blocks";

  p.snr_grid_2d = {0.2, 0.5, 1.0, 2.0, 5.0};
  p.rho_grid_2d = {0.30, 0.50, 0.70, 0.85, 0.90, 0.99};

  return p;
}

// =============================================================================
// Shared method controls
// =============================================================================

static AR1BlocksMethodSet make_method_set(const GVSSimConfig& cfg) {
  AR1BlocksMethodSet ms;
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
// Part 1 — 2-D SNR × rho sweep
// =============================================================================

/**
 * @brief Run a 2-D MC sweep over SNR and rho on the AR(1)-blocks DGP.
 */
static void run_ar1_part1_2d_sweep(const GVSSimConfig& cfg,
                                   const AR1BlocksPreset& preset,
                                   const AR1BlocksMethodSet& ms) {
  GVS2DDGPFactory dgp_2d = [&cfg](double snr, double rho, unsigned seed) {
    return make_ar1_blocks_dgp(cfg.n, cfg.p, snr, rho, seed);
  };

  cdiag::print_section_header(
      "Part 1: 2-D SNR x rho Sweep | " + preset.scenario_tag);
  std::cout << "  n=" << cfg.n << "  p=" << cfg.p << "  s=150"
            << "  MC=" << cfg.num_MC << "  tFDR=" << fmt_num(cfg.tFDR)
            << "  K=" << cfg.K << "  corr_max=" << fmt_num(cfg.corr_max) << "\n"
            << "  methods: TENET / TENET_AUG / TIENET_AUG"
            << "  |  lambda2=CV_1SE_CCD (EN), CV_1SE_IEN_CCD (IEN)  linkage=Single\n"
            << "  snr_grid: {";
  for (std::size_t i = 0; i < preset.snr_grid_2d.size(); ++i)
    std::cout << (i ? ", " : "") << fmt_num(preset.snr_grid_2d[i]);
  std::cout << "}\n  rho_grid: {";
  for (std::size_t i = 0; i < preset.rho_grid_2d.size(); ++i)
    std::cout << (i ? ", " : "") << fmt_num(preset.rho_grid_2d[i]);
  std::cout << "}\n";

  auto en = run_gvs_2d_sweep(
      dgp_2d, preset.snr_grid_2d, preset.rho_grid_2d, cfg,
      ms.en, ms.trex_ctrl, "TENET");
  auto en_aug = run_gvs_2d_sweep(
      dgp_2d, preset.snr_grid_2d, preset.rho_grid_2d, cfg,
      ms.en_aug, ms.trex_ctrl, "TENET_AUG");
  auto ien = run_gvs_2d_sweep(
      dgp_2d, preset.snr_grid_2d, preset.rho_grid_2d, cfg,
      ms.ien, ms.trex_ctrl, "TIENET_AUG");

  std::vector<std::string> snr_labels, rho_labels;
  for (double s : preset.snr_grid_2d) {
    snr_labels.push_back("snr=" + fmt_num(s));
  }
  for (double r : preset.rho_grid_2d) {
    rho_labels.push_back("rho=" + fmt_num(r));
  }

  print_mc_matrix("mean_FDP [TENET]", snr_labels, rho_labels, en, false);
  print_mc_matrix("mean_TPP [TENET]", snr_labels, rho_labels, en, true);
  print_mc_matrix("mean_FDP [TENET_AUG]", snr_labels, rho_labels, en_aug, false);
  print_mc_matrix("mean_TPP [TENET_AUG]", snr_labels, rho_labels, en_aug, true);
  print_mc_matrix("mean_FDP [TIENET_AUG]", snr_labels, rho_labels, ien, false);
  print_mc_matrix("mean_TPP [TIENET_AUG]", snr_labels, rho_labels, ien, true);

  save_mc_2d_tables(
      preset.scenario_tag,
      snr_labels,
      rho_labels,
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
 * @brief Run the AR(1)-blocks benchmark (2-D SNR × rho sweep).
 */
static void run_ar1_blocks_benchmark(const GVSSimConfig& cfg,
                                     const AR1BlocksPreset& preset) {
  const auto ms = make_method_set(cfg);

  cdiag::print_section_header(
      "AR(1)-within-block benchmark | " + preset.scenario_tag +
      "\nDGP = make_ar1_blocks_dgp"
      "\nblocks = {20, 50, 80, 65}, active = {0,1,2}, inactive trap = {3}");

  run_ar1_part1_2d_sweep(cfg, preset, ms);

  std::cout << preset.scenario_tag << " GVS MC simulations complete.\n";
}

// =============================================================================
// main
// =============================================================================

int main() {

  // ==========================================================================
  //  Simulation parameters
  // ==========================================================================

  std::cout.setf(std::ios::unitbuf);
  omp_set_num_threads(trex_mc_workers::count());
  std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

  GVSSimConfig cfg;
  cfg.n = 200;
  cfg.p = 500;
  cfg.tFDR = 0.1;
  cfg.K = 20;
  cfg.num_MC = num_mc(200);
  cfg.base_seed = 2026;
  cfg.corr_max = 0.98;

  // ==========================================================================
  //  Run simulations
  // ==========================================================================

  if (true) run_ar1_blocks_benchmark(cfg, make_ar1_blocks_preset());

  std::cout << "All AR(1)-blocks benchmark simulations complete.\n";
  return 0;
}
