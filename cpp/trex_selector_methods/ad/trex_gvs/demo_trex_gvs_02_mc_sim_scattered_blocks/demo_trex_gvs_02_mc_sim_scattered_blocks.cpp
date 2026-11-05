// ==============================================================================
// demo_trex_gvs_02_mc_sim_scattered_blocks.cpp
// ==============================================================================
/**
 * @file demo_trex_gvs_02_mc_sim_scattered_blocks.cpp
 *
 * @brief MC simulation: T-Rex+GVS on the scattered-grouped DGP.
 *
 * @details
 * This scenario evaluates grouped-signal recovery when active variables are
 * randomly scattered across the design rather than arranged in contiguous
 * blocks. Within-group correlation is controlled through sd_x via
 * rho = 1 / (1 + sd_x^2).
 *
 * One part:
 * 1. 2-D SNR × rho sweep, using sd_x = sqrt((1 - rho)/rho)
 *    (subsumes the former 1-D SNR and rho sweeps as its rho = 0.99 column and
 *     SNR = 2 row)
 *
 * DGP:
 * - make_scattered_grouped_dgp()
 * - n_groups = 3
 * - group_size = 50
 * - all grouped variables active
 * - support size s = 150
 * - p = 500 total variables, with remaining columns i.i.d. noise
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

struct ScatteredGroupedPreset {
  std::string scenario_tag;
  std::string file_stem_prefix;

  int n_groups = 3;
  int group_size = 50;

  std::vector<double> snr_grid_2d;
  std::vector<double> rho_grid_2d;
};

struct ScatteredGroupedMethodSet {
  TRexControlParameter trex_ctrl;
  TRexGVSControlParameter en;
  TRexGVSControlParameter en_aug;
  TRexGVSControlParameter ien;
};

// =============================================================================
// Helpers
// =============================================================================

static double rho_to_sd_x(double rho) {
  return std::sqrt((1.0 - rho) / rho);
}

static ScatteredGroupedPreset make_scattered_grouped_preset() {
  ScatteredGroupedPreset p;
  p.scenario_tag = "Scattered-Grouped";
  p.file_stem_prefix = "gvs_scattered_grouped";

  p.n_groups = 3;
  p.group_size = 50;

  p.snr_grid_2d = {0.2, 0.5, 1.0, 2.0, 5.0};
  p.rho_grid_2d = {0.30, 0.50, 0.70, 0.90, 0.95, 0.99};

  return p;
}

static ScatteredGroupedMethodSet make_method_set(const GVSSimConfig& cfg) {
  ScatteredGroupedMethodSet ms;
  ms.trex_ctrl = make_gvs_trex_control(cfg.K);

  ms.en.gvs_type = GVSType::EN;
  ms.en.lambda2_method = LambdaSelectionMethod::CV_1SE_CCD;
  ms.en.corr_max = cfg.corr_max;
  ms.en.hc_linkage = hac::LinkageMethod::Single;
  ms.en.en_solver = ENSolverType::TENET;

  ms.en_aug = ms.en; // copy and overwrite solver type
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
 * @brief Run a 2-D MC sweep over SNR and rho on the scattered-grouped DGP.
 */
static void run_scattered_part1_2d_sweep(
    const GVSSimConfig& cfg,
    const ScatteredGroupedPreset& preset,
    const ScatteredGroupedMethodSet& ms) {
  const int n_groups = preset.n_groups;
  const int group_size = preset.group_size;

  GVS2DDGPFactory dgp_2d = [&cfg, n_groups, group_size](
                               double snr, double rho, unsigned seed) {
    const double sd_x = rho_to_sd_x(rho);
    return make_scattered_grouped_dgp(
        cfg.n, cfg.p, n_groups, group_size, snr, sd_x, seed);
  };

  cdiag::print_section_header(
      "Part 1: 2-D SNR x rho Sweep | " + preset.scenario_tag);
  std::cout << "  n=" << cfg.n << "  p=" << cfg.p
            << "  s=" << (n_groups * group_size)
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

  print_mc_matrix("mean_TPP [TENET]", snr_labels, rho_labels, en, true);
  print_mc_matrix("mean_FDP [TENET]", snr_labels, rho_labels, en, false);
  print_mc_matrix("mean_TPP [TENET_AUG]", snr_labels, rho_labels, en_aug, true);
  print_mc_matrix("mean_FDP [TENET_AUG]", snr_labels, rho_labels, en_aug, false);
  print_mc_matrix("mean_TPP [TIENET_AUG]", snr_labels, rho_labels, ien, true);
  print_mc_matrix("mean_FDP [TIENET_AUG]", snr_labels, rho_labels, ien, false);

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
 * @brief Run the scattered-grouped benchmark (2-D SNR × rho sweep).
 */
static void run_scattered_grouped_benchmark(
    const GVSSimConfig& cfg,
    const ScatteredGroupedPreset& preset) {
  const auto ms = make_method_set(cfg);

  cdiag::print_section_header(
      "Scattered-grouped benchmark | " + preset.scenario_tag +
      "\nDGP = make_scattered_grouped_dgp"
      "\nn_groups = " + std::to_string(preset.n_groups) +
      ", group_size = " + std::to_string(preset.group_size) +
      ", s = " + std::to_string(preset.n_groups * preset.group_size));

  run_scattered_part1_2d_sweep(cfg, preset, ms);

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

  if (true) run_scattered_grouped_benchmark(cfg, make_scattered_grouped_preset());

  std::cout << "All scattered-grouped benchmark simulations complete.\n";
  return 0;
}
