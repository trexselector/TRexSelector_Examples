// ==============================================================================
// demo_trex_da_01_mc_sim_ar1.cpp
// ==============================================================================
/**
 * @file demo_trex_da_01_mc_sim_ar1.cpp
 *
 * @brief DA-TRex+AR1 Monte Carlo simulations for the AR(1) Toeplitz DGP.
 *
 * @details
 *  Part 1: MC simulation sweeping SNR (n=300, p=1000, s=10, rho=0.7, 200 MC,
 *          3 solvers: TLARS, TAFS, TOMP).
 *
 *  Part 2: MC simulation sweeping rho (n=300, p=1000, s=10, SNR=2.0, 200 MC,
 *          3 solvers: TLARS, TAFS, TOMP).
 *
 *  Part 3: 2D Gap × Rho sweep. Explores the kappa-boundary where
 *          gap < kappa collapses TPP. Support is generated via
 *          CappedSpread(gap) and Random policies. 3 solvers: TLARS, TAFS, TOMP.
 *
 *          gap_grid : {100, 50, 20, 15, 10, 5, 1}
 *          rho_grid : {0.0, 0.1, ..., 0.9}
 *
 * Results are saved to simulation_results/ as TXT and CSV files.
 */
// ==============================================================================

#include "trex_da_sim_common.hpp"

// ==============================================================================

using namespace da_sim;

// ==============================================================================
//  Part 1: MC simulation — SNR sweep, 2 support types (CappedSpread + Random)
// ==============================================================================

void demo_ar1_mc_snr_sweep()
{
    // Simulation config
    SimConfig cfg;
    cfg.n = 300;
    cfg.p = 1000;
    cfg.s = 10;
    cfg.amplitude = 3.0;
    cfg.tFDR = 0.2;
    constexpr int max_gap = 20;
    constexpr double rho = 0.7;

    TRexDAControlParameter da_ctrl;
    da_ctrl.method = DAMethod::AR1;


    const std::vector<double> snr_grid = {0.1, 0.2, 0.5, 1.0, 2.0, 5.0};
    const std::string hdr_base =
        "AR(1) rho=" + fmt_num(rho) +
        ", n=" + std::to_string(cfg.n) +
        ", p=" + std::to_string(cfg.p);


    // --- Run A: CappedSpread(max_gap=20) ---
    run_snr_sweep(
        "da_ar1_snr_capped",
        snr_grid,
        cfg.num_MC,
        cfg.tFDR,
        cfg.base_seed,
        default_solvers("+AR1"),
        [&](double snr, unsigned seed) {
            return dgp_ar1(cfg.n,
                           cfg.p,
                           make_support(
                               SupportPolicy::CappedSpread,
                               cfg.s, cfg.p, max_gap, seed),
                           cfg.amplitude, rho, snr, seed);
        },
        da_ctrl,
        make_trex_control(cfg.K),
        hdr_base + ", support=CappedSpread(max_gap=" +
                      std::to_string(max_gap) + ")",
        /*include_base_trex=*/true);


    // --- Run B: Random support (redrawn per trial) ---
    run_snr_sweep(
        "da_ar1_snr_random",
        snr_grid,
        cfg.num_MC,
        cfg.tFDR,
        cfg.base_seed,
        default_solvers("+AR1"),
        [&](double snr, unsigned seed) {
            return dgp_ar1(cfg.n,
                           cfg.p,
                           make_support(
                               SupportPolicy::Random,
                               cfg.s, cfg.p, 0, seed),
                           cfg.amplitude, rho, snr, seed);
        },
        da_ctrl,
        make_trex_control(cfg.K),
        hdr_base + ", support=Random (redrawn per trial)",
        /*include_base_trex=*/true);
}


// ==============================================================================
//  Part 2: MC simulation — Rho sweep, 2 support types (CappedSpread + Random)
// ==============================================================================

void demo_ar1_mc_rho_sweep()
{
    // Simulation config
    SimConfig cfg;
    cfg.n = 300;
    cfg.p = 1000;
    cfg.s = 10;
    cfg.amplitude = 3.0;
    cfg.tFDR = 0.2;
    constexpr int    max_gap   = 20;
    constexpr double snr_fixed = 2.0;

    TRexDAControlParameter da_ctrl;
    da_ctrl.method = DAMethod::AR1;

    const std::vector<double> rho_grid = {
        0.0, 0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8, 0.9
    };
    const std::string hdr_base =
        "AR(1) SNR=" + fmt_num(snr_fixed) +
        ", n=" + std::to_string(cfg.n) +
        ", p=" + std::to_string(cfg.p);


    // --- Run A: CappedSpread(max_gap=20) ---
    run_param_sweep(
        "da_ar1_rho_capped",
        "Rho",
        rho_grid,
        cfg.num_MC,
        cfg.tFDR,
        cfg.base_seed,
        default_solvers("+AR1"),
        [&](double rho_val, unsigned seed) {
            return dgp_ar1(cfg.n,
                           cfg.p,
                           make_support(
                               SupportPolicy::CappedSpread,
                               cfg.s, cfg.p, max_gap, seed),
                           cfg.amplitude, rho_val, snr_fixed, seed);
        },
        da_ctrl,
        make_trex_control(cfg.K),
        hdr_base + ", support=CappedSpread(max_gap=" +
                     std::to_string(max_gap) + ")",
        /*include_base_trex=*/true);

    // --- Run B: Random support (redrawn per trial) ---
    run_param_sweep(
        "da_ar1_rho_random",
        "Rho",
        rho_grid,
        cfg.num_MC,
        cfg.tFDR,
        cfg.base_seed,
        default_solvers("+AR1"),
        [&](double rho_val, unsigned seed) {
            return dgp_ar1(cfg.n, cfg.p,
                           make_support(SupportPolicy::Random,
                                        cfg.s, cfg.p, 0, seed),
                           cfg.amplitude, rho_val, snr_fixed, seed);
        },
        da_ctrl,
        make_trex_control(cfg.K),
        hdr_base + ", support=Random (redrawn per trial)",
        /*include_base_trex=*/true);
}


// ==============================================================================
//  Part 3: 2D Gap × Rho sweep
//
//  Runs num_MC trials per (solver, gap, rho) cell for two support types:
//    A) CappedSpread(gap) — deterministic, evenly-spaced support.
//    B) Random           — support redrawn each trial (not gap-dependent).
//
//  The kappa boundary (kappa = ceiling(log(0.02) / log(rho))) is the DA+AR1
//  correction window half-width.  When gap < kappa, active predictors fall
//  inside each other's correction windows and TPP collapses.
//
//  gap_grid : {100, 50, 20, 15, 10, 5, 1}
//  rho_grid : {0.0, 0.1, ..., 0.9}
// ==============================================================================

void demo_ar1_mc_gap_rho_sweep()
{
    SimConfig cfg;
    cfg.n      = 300;
    cfg.p      = 1000;
    cfg.s      = 10;
    cfg.num_MC = 200;
    cfg.amplitude = 3.0;
    cfg.tFDR = 0.2;
    constexpr double snr_fixed = 2.0;

    const std::vector<int> gap_grid = {100, 50, 20, 15, 10, 5, 1};
    const std::vector<double> rho_grid = {
        0.0, 0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8, 0.9
    };

    const auto n_gap = gap_grid.size();
    const auto n_rho = rho_grid.size();
    const auto n_rho_idx = static_cast<Eigen::Index>(n_rho);
    const auto n_gap_idx = static_cast<Eigen::Index>(n_gap);

    // Kappa boundary: ceil(log(0.02) / log(rho)).
    // For rho=0 (independent predictors) kappa=0 → gap always sufficient.
    std::vector<int> kappa_boundary(n_rho);
    for (std::size_t ir = 0; ir < n_rho; ++ir) {
        if (rho_grid[ir] <= 0.0)
            kappa_boundary[ir] = 0;
        else
            kappa_boundary[ir] = static_cast<int>(
                std::ceil(std::log(0.02) / std::log(rho_grid[ir])));
    }

    // Per-solver result maps
    std::map<std::string, Eigen::MatrixXd> tpp_cs_map,    fdp_cs_map,
                                           sd_tpp_cs_map,  sd_fdp_cs_map;
    std::map<std::string, Eigen::VectorXd> tpp_rand_map,   fdp_rand_map,
                                           sd_tpp_rand_map, sd_fdp_rand_map;

    const auto solvers = default_solvers("+AR1");
    for (const auto& sv : solvers) {
        tpp_cs_map[sv.name]     = Eigen::MatrixXd::Zero(n_rho_idx, n_gap_idx);
        fdp_cs_map[sv.name]     = Eigen::MatrixXd::Zero(n_rho_idx, n_gap_idx);
        sd_tpp_cs_map[sv.name]  = Eigen::MatrixXd::Zero(n_rho_idx, n_gap_idx);
        sd_fdp_cs_map[sv.name]  = Eigen::MatrixXd::Zero(n_rho_idx, n_gap_idx);
        tpp_rand_map[sv.name]    = Eigen::VectorXd::Zero(n_rho_idx);
        fdp_rand_map[sv.name]    = Eigen::VectorXd::Zero(n_rho_idx);
        sd_tpp_rand_map[sv.name] = Eigen::VectorXd::Zero(n_rho_idx);
        sd_fdp_rand_map[sv.name] = Eigen::VectorXd::Zero(n_rho_idx);
    }

    TRexDAControlParameter da_ctrl;
    da_ctrl.method = DAMethod::AR1;

    const unsigned base_offset = static_cast<unsigned>(cfg.base_seed);

    // --- Header ---
    cdianostics::print_section_header("AR(1) Part 3: Gap × Rho sweep");
    std::cout << "  n=" << cfg.n << "  p=" << cfg.p << "  s=" << cfg.s
              << "  SNR=" << snr_fixed << "  MC=" << cfg.num_MC
              << "  tFDR=" << cfg.tFDR
              << "  amplitude=" << cfg.amplitude << "\n";
    std::cout << "  gap_grid: {";
    for (std::size_t i = 0; i < n_gap; ++i)
        std::cout << (i ? ", " : "") << gap_grid[i];
    std::cout << "}\n  rho_grid: {";
    for (std::size_t i = 0; i < n_rho; ++i)
        std::cout << (i ? ", " : "")
                  << std::fixed << std::setprecision(1) << rho_grid[i];
    std::cout << "}\n\n";

    // ---- Loop over solvers ----
    for (const auto& sv : solvers) {
        auto trex_ctrl = make_trex_control(cfg.K);
        trex_ctrl.solver_type           = sv.type;
        trex_ctrl.solver_params.rho_afs = sv.rho_afs;
        trex_ctrl.solver_params.exch_tie_alpha = sv.exch_tie_alpha;
        trex_ctrl.tloop_stagnation_stop = sv.use_stagnation;

        std::cout << "\n  Solver: " << sv.name << "\n\n";

        // ---- Section A: CappedSpread(gap) × rho ----
        std::cout << "  [A] CappedSpread(gap) sweep ...\n\n";
        int cell_idx = 0;
        const int total_cells = static_cast<int>(n_rho * n_gap);

        for (std::size_t ir = 0; ir < n_rho; ++ir) {
            const double rho_val = rho_grid[ir];
            for (std::size_t ig = 0; ig < n_gap; ++ig) {
                ++cell_idx;
                const int gap_val = gap_grid[ig];
                auto support = capped_spread_support(cfg.s, cfg.p, gap_val);

                std::ostringstream lbl;
                lbl << "[" << cell_idx << "/" << total_cells << "] "
                    << sv.name
                    << "  rho=" << std::fixed << std::setprecision(1) << rho_val
                    << "  gap=" << std::setw(3) << gap_val
                    << "  kappa=" << kappa_boundary[ir];

                auto res = run_mc_trials(
                    cfg.num_MC, lbl.str(),
                    [&](unsigned seed) {
                        return dgp_ar1(cfg.n, cfg.p, support,
                                       cfg.amplitude, rho_val, snr_fixed, seed);
                    },
                    cfg.tFDR, trex_ctrl, da_ctrl, base_offset);

                const auto ri = static_cast<Eigen::Index>(ir);
                const auto gi = static_cast<Eigen::Index>(ig);
                tpp_cs_map[sv.name](ri, gi)    = res.avg_tpr;
                fdp_cs_map[sv.name](ri, gi)    = res.avg_fdr;
                sd_tpp_cs_map[sv.name](ri, gi) = res.sd_tpr;
                sd_fdp_cs_map[sv.name](ri, gi) = res.sd_fdr;
            }
        }

        // ---- Section B: Random support sweep (rho only) ----
        std::cout << "\n  [B] Random support sweep ...\n\n";
        for (std::size_t ir = 0; ir < n_rho; ++ir) {
            const double rho_val = rho_grid[ir];

            std::ostringstream lbl;
            lbl << "[" << (ir + 1) << "/" << n_rho << "] "
                << sv.name
                << "  rho=" << std::fixed << std::setprecision(1) << rho_val
                << "  kappa=" << kappa_boundary[ir] << "  Random";

            auto res = run_mc_trials(
                cfg.num_MC, lbl.str(),
                [&](unsigned seed) {
                    return dgp_ar1(cfg.n, cfg.p,
                                   make_support(SupportPolicy::Random,
                                                cfg.s, cfg.p, 0, seed),
                                   cfg.amplitude, rho_val, snr_fixed, seed);
                },
                cfg.tFDR, trex_ctrl, da_ctrl, base_offset);

            const auto ri = static_cast<Eigen::Index>(ir);
            tpp_rand_map[sv.name](ri)    = res.avg_tpr;
            fdp_rand_map[sv.name](ri)    = res.avg_fdr;
            sd_tpp_rand_map[sv.name](ri) = res.sd_tpr;
            sd_fdp_rand_map[sv.name](ri) = res.sd_fdr;
        }
    }

    // ---- Save results ----
    const std::string header_extra =
        "n=" + std::to_string(cfg.n)  +
        ", p=" + std::to_string(cfg.p) +
        ", s=" + std::to_string(cfg.s) +
        ", SNR=" + fmt_num(snr_fixed) +
        ", tFDR=" + fmt_num(cfg.tFDR) +
        ", amplitude=" + fmt_num(cfg.amplitude);

    save_and_print_2d_gap_rho_results(
        "da_ar1_gap_rho",
        gap_grid,
        rho_grid,
        kappa_boundary,
        cfg.num_MC,
        solvers,
        tpp_cs_map,    fdp_cs_map,    sd_tpp_cs_map,  sd_fdp_cs_map,
        tpp_rand_map,  fdp_rand_map,  sd_tpp_rand_map, sd_fdp_rand_map,
        header_extra);
}


// ==============================================================================
//  main
// ==============================================================================

int main() {

    std::cout.setf(std::ios::unitbuf);
    omp_set_num_threads(trex_mc_workers::count());
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    if (true) demo_ar1_mc_snr_sweep();
    if (true) demo_ar1_mc_rho_sweep();
    if (true) demo_ar1_mc_gap_rho_sweep();

    std::cout << "\nAR(1) MC simulation complete.\n";
    return 0;
}
