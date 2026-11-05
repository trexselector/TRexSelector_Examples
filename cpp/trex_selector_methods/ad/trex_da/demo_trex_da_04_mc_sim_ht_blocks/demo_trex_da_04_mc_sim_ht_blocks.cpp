// ==============================================================================
// demo_trex_da_04_mc_sim_ht_blocks.cpp
// ==============================================================================
/**
 * @file demo_trex_da_04_mc_sim_ht_blocks.cpp
 *
 * @brief DA-TRex+BT MC simulations for the heavy-tailed block DGP (BT aggregation).
 *
 * @details
 *  Corresponds to R reference: demo_trex_da_07_bt_heavy_tailed_sweeps.R.
 *
 *  DGP: dgp_block_toeplitz_hvt() -- block-diagonal Toeplitz covariance, predictors
 *  and optionally noise are t(nu=3)-distributed. G = M blocks of size g = Q each.
 *  Support is determined internally (OnePerBlock); no support/amplitude args.
 *
 *  Base params: n=150, M=5, Q=5 (p=25, s=5), rho=0.8, nu=3.0,
 *               tFDR=0.2, K=20, num_MC=200, seed=2026.
 *
 *  Outer loops: h_noise in {false ("s1_Gauss"), true ("s2_Heavy")}
 *               linkage in {Single, Complete, Average}
 *  All 3 solvers (TLARS, TAFS, TOMP) run inside each linkage iteration.
 *
 *  Functions:
 *   demo_ht_block_mc_snr_sweep()     -- Section 1: SNR in {0.1,...,5.0}
 *   demo_ht_block_mc_rho_sweep()     -- Section 2: rho in {0.0,...,0.9}
 *   demo_ht_block_mc_q_sweep()       -- Section 3: Q in {5,...,50}
 *   demo_ht_block_mc_m_sweep()       -- Section 4: M in {1,...,30}
 *   demo_ht_block_mc_tfdr_sweep()    -- Section 5: tFDR in {0.05,...,0.50}
 *   demo_ht_block_mc_linkage_sweep() -- Section 6: linkage {Single,Complete,Average}
 */
// ==============================================================================

#include "trex_da_sim_common.hpp"

using namespace da_sim;

namespace {
    constexpr int    n         = 150;
    constexpr int    M         = 5;
    constexpr int    Q         = 5;
    constexpr double rho       = 0.8;
    constexpr double nu        = 3.0;
    constexpr double snr       = 2.0;
    constexpr double tFDR      = 0.2;
    constexpr int    K         = 20;
    constexpr int    num_MC    = 200;
    constexpr int    base_seed = 2026;
    constexpr int    base_p    = M * Q;   // 25
}

using ScenarioList = std::vector<std::pair<std::string, bool>>;
using LinkageList  = std::vector<std::pair<std::string, hac::LinkageMethod>>;

inline ScenarioList all_scenarios() {
    return {{"s1_Gauss", false}, {"s2_Heavy", true}};
}

inline LinkageList all_linkages() {
    return {
        {"Single",   hac::LinkageMethod::Single},
        {"Complete", hac::LinkageMethod::Complete},
        {"Average",  hac::LinkageMethod::Average}
    };
}


// ==============================================================================
//  Section 1: MC SNR sweep
//  Fixed: n=150, M=5, Q=5 (p=25, s=5), rho=0.8, nu=3.0.
//  Swept: SNR in {0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0}.
//  Outer loops: h_noise scenario x linkage.
// ==============================================================================
void demo_ht_block_mc_snr_sweep()
{
    const std::vector<double> snr_grid = {0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0};

    for (const auto& [scen_str, h_noise] : all_scenarios()) {
        for (const auto& [lnk_str, lnk_val] : all_linkages()) {
            auto da_ctrl = make_da_bt_control();
            da_ctrl.hc_linkage = lnk_val;

            std::string tag = "da_ht_blocks_snr_";
            tag += scen_str; tag += "_"; tag += lnk_str;
            std::string title = "HT-block SNR sweep  n=" + std::to_string(n)
                + "  M=" + std::to_string(M) + "  Q=" + std::to_string(Q)
                + "  p=" + std::to_string(base_p) + "  s=" + std::to_string(M)
                + "  rho=" + fmt_num(rho) + "  nu=" + fmt_num(nu)
                + "  noise=";
            title += scen_str; title += "  linkage="; title += lnk_str;

            run_snr_sweep(
                tag,
                snr_grid,
                num_MC,
                tFDR,
                base_seed,
                default_solvers("+BT"),
                [&](double snr_val, unsigned seed) {
                    return dgp_block_toeplitz_hvt(
                        n, base_p, M, Q, rho, snr_val, seed,
                        /*heavy_X=*/true, h_noise, nu, /*random_support=*/false);
                },
                da_ctrl,
                make_trex_control(K),
                title,
                /*include_base_trex=*/false);
        }
    }
}


// ==============================================================================
//  Section 2: MC rho sweep
//  Fixed: n=150, M=5, Q=5 (p=25, s=5), SNR=2, nu=3.0.
//  Swept: rho in {0.0, 0.1, ..., 0.9}.
//  Outer loops: h_noise scenario x linkage.
// ==============================================================================
void demo_ht_block_mc_rho_sweep()
{
    const std::vector<double> rho_grid = {
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    for (const auto& [scen_str, h_noise] : all_scenarios()) {
        for (const auto& [lnk_str, lnk_val] : all_linkages()) {
            auto da_ctrl = make_da_bt_control();
            da_ctrl.hc_linkage = lnk_val;

            std::string tag = "da_ht_blocks_rho_";
            tag += scen_str; tag += "_"; tag += lnk_str;
            std::string title = "HT-block rho sweep  n=" + std::to_string(n)
                + "  M=" + std::to_string(M) + "  Q=" + std::to_string(Q)
                + "  s=" + std::to_string(M) + "  SNR=" + fmt_num(snr)
                + "  nu=" + fmt_num(nu) + "  noise=";
            title += scen_str; title += "  linkage="; title += lnk_str;

            run_param_sweep(
                tag,
                "Rho",
                rho_grid,
                num_MC,
                tFDR,
                base_seed,
                default_solvers("+BT"),
                [&](double rho_val, unsigned seed) {
                    return dgp_block_toeplitz_hvt(
                        n, base_p, M, Q, rho_val, snr, seed,
                        /*heavy_X=*/true, h_noise, nu, /*random_support=*/false);
                },
                da_ctrl,
                make_trex_control(K),
                title,
                /*include_base_trex=*/false);
        }
    }
}


// ==============================================================================
//  Section 3: MC Q sweep (block size)
//  Fixed: n=150, M=5, rho=0.8, SNR=2, nu=3.0.  s = M = 5 throughout.
//  Swept: Q in {5, 10, ..., 50}; p = M*Q varies.
//  Outer loops: h_noise scenario x linkage.
// ==============================================================================
void demo_ht_block_mc_q_sweep()
{
    const std::vector<double> q_grid = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50};

    for (const auto& [scen_str, h_noise] : all_scenarios()) {
        for (const auto& [lnk_str, lnk_val] : all_linkages()) {
            auto da_ctrl = make_da_bt_control();
            da_ctrl.hc_linkage = lnk_val;

            std::string tag = "da_ht_blocks_Q_";
            tag += scen_str; tag += "_"; tag += lnk_str;
            std::string title = "HT-block Q sweep  n=" + std::to_string(n)
                + "  M=" + std::to_string(M) + "  s=" + std::to_string(M)
                + "  rho=" + fmt_num(rho) + "  SNR=" + fmt_num(snr)
                + "  nu=" + fmt_num(nu) + "  p=M*Q varies  noise=";
            title += scen_str; title += "  linkage="; title += lnk_str;

            run_param_sweep(
                tag,
                "Q",
                q_grid,
                num_MC,
                tFDR,
                base_seed,
                default_solvers("+BT"),
                [&](double q_val, unsigned seed) {
                    int q_int = static_cast<int>(q_val);
                    int p_val = M * q_int;
                    return dgp_block_toeplitz_hvt(
                        n, p_val, M, q_int, rho, snr, seed,
                        /*heavy_X=*/true, h_noise, nu, /*random_support=*/false);
                },
                da_ctrl,
                make_trex_control(K),
                title,
                /*include_base_trex=*/false);
        }
    }
}


// ==============================================================================
//  Section 4: MC M sweep (number of blocks)
//  Fixed: n=150, Q=5, rho=0.8, SNR=2, nu=3.0.
//  Swept: M in {1, 3, 5, 10, 15, 20, 30}; p = M*Q and s = M both vary.
//  Outer loops: h_noise scenario x linkage.
// ==============================================================================
void demo_ht_block_mc_m_sweep()
{
    const std::vector<double> m_grid = {1, 3, 5, 10, 15, 20, 30};

    for (const auto& [scen_str, h_noise] : all_scenarios()) {
        for (const auto& [lnk_str, lnk_val] : all_linkages()) {
            auto da_ctrl = make_da_bt_control();
            da_ctrl.hc_linkage = lnk_val;

            std::string tag = "da_ht_blocks_M_";
            tag += scen_str; tag += "_"; tag += lnk_str;
            std::string title = "HT-block M sweep  n=" + std::to_string(n)
                + "  Q=" + std::to_string(Q)
                + "  rho=" + fmt_num(rho) + "  SNR=" + fmt_num(snr)
                + "  nu=" + fmt_num(nu) + "  p=M*Q and s=M vary  noise=";
            title += scen_str; title += "  linkage="; title += lnk_str;

            run_param_sweep(
                tag,
                "M",
                m_grid,
                num_MC,
                tFDR,
                base_seed,
                default_solvers("+BT"),
                [&](double m_val, unsigned seed) {
                    int m_int = static_cast<int>(m_val);
                    int p_val = m_int * Q;
                    return dgp_block_toeplitz_hvt(
                        n, p_val, m_int, Q, rho, snr, seed,
                        /*heavy_X=*/true, h_noise, nu, /*random_support=*/false);
                },
                da_ctrl,
                make_trex_control(K),
                title,
                /*include_base_trex=*/false);
        }
    }
}


// ==============================================================================
//  Section 5: MC tFDR sweep
//  Fixed: n=150, M=5, Q=5 (p=25, s=5), rho=0.8, SNR=2, nu=3.0.
//  Swept: tFDR in {0.05, 0.10, ..., 0.50}.
//  Outer loops: h_noise scenario x linkage.
//  (tFDR is a selector parameter -- run_mc_trials called per solver x grid point.)
// ==============================================================================
void demo_ht_block_mc_tfdr_sweep()
{
    std::vector<double> tfdr_grid;
    for (int i = 1; i <= 10; ++i) tfdr_grid.push_back(0.05 * i);

    const auto ngrid = static_cast<Eigen::Index>(tfdr_grid.size());

    for (const auto& [scen_str, h_noise] : all_scenarios()) {
        for (const auto& [lnk_str, lnk_val] : all_linkages()) {
            auto da_ctrl = make_da_bt_control();
            da_ctrl.hc_linkage = lnk_val;

            auto solvers = default_solvers("+BT");
            std::map<std::string, Eigen::VectorXd> fdr_map, tpr_map,
                                                   sd_fdr_map, sd_tpr_map,
                                                   avg_L_map, avg_T_map;

            for (const auto& sol : solvers) {
                fdr_map[sol.name]    = Eigen::VectorXd::Zero(ngrid);
                tpr_map[sol.name]    = Eigen::VectorXd::Zero(ngrid);
                sd_fdr_map[sol.name] = Eigen::VectorXd::Zero(ngrid);
                sd_tpr_map[sol.name] = Eigen::VectorXd::Zero(ngrid);
                avg_L_map[sol.name]  = Eigen::VectorXd::Zero(ngrid);
                avg_T_map[sol.name]  = Eigen::VectorXd::Zero(ngrid);
            }

            auto trex_ctrl = make_trex_control(K);

            for (const auto& sol : solvers) {
                trex_ctrl.solver_type           = sol.type;
                trex_ctrl.solver_params.rho_afs = sol.rho_afs;
                trex_ctrl.solver_params.exch_tie_alpha = sol.exch_tie_alpha;
                trex_ctrl.tloop_stagnation_stop = sol.use_stagnation;

                for (Eigen::Index gi = 0; gi < ngrid; ++gi) {
                    double tfdr_val = tfdr_grid[static_cast<std::size_t>(gi)];
                    std::string lbl = "tFDR = " + fmt_num(tfdr_val);

                    auto dgp_fn = [&](unsigned seed) {
                        return dgp_block_toeplitz_hvt(
                            n, base_p, M, Q, rho, snr, seed,
                            /*heavy_X=*/true, h_noise, nu, /*random_support=*/false);
                    };

                    unsigned seed_offset = static_cast<unsigned>(base_seed)
                                         + static_cast<unsigned>(gi) * 10000u;

                    auto res = run_mc_trials(
                        num_MC, lbl, dgp_fn, tfdr_val, trex_ctrl, da_ctrl, seed_offset);

                    fdr_map[sol.name](gi)    = res.avg_fdr;
                    tpr_map[sol.name](gi)    = res.avg_tpr;
                    sd_fdr_map[sol.name](gi) = res.sd_fdr;
                    sd_tpr_map[sol.name](gi) = res.sd_tpr;
                    avg_L_map[sol.name](gi)  = res.avg_L;
                    avg_T_map[sol.name](gi)  = res.avg_T;
                }
            }

            std::string tag = "da_ht_blocks_tFDR_";
            tag += scen_str; tag += "_"; tag += lnk_str;
            std::string title = "HT-block tFDR sweep  n=" + std::to_string(n)
                + "  M=" + std::to_string(M) + "  Q=" + std::to_string(Q)
                + "  p=" + std::to_string(base_p) + "  s=" + std::to_string(M)
                + "  rho=" + fmt_num(rho) + "  SNR=" + fmt_num(snr)
                + "  nu=" + fmt_num(nu) + "  noise=";
            title += scen_str; title += "  linkage="; title += lnk_str;
            save_and_print_grid_results(
                tag,
                "tFDR",
                tfdr_grid,
                num_MC,
                solvers,
                fdr_map, tpr_map, sd_fdr_map, sd_tpr_map, avg_L_map, avg_T_map,
                title);
        }
    }
}


// ==============================================================================
//  Section 6: Linkage sweep
//  Fixed: n=150, M=5, Q=5 (p=25, s=5), rho=0.8, SNR=2, nu=3.0, tFDR=0.2.
//  Swept: linkage encoded as {1.0=Single, 2.0=Complete, 3.0=Average}.
//  Outer loop: h_noise scenario.
// ==============================================================================
void demo_ht_block_mc_linkage_sweep()
{
    const auto lnk_list  = all_linkages();
    const std::vector<double> lnk_vals_d = {1.0, 2.0, 3.0};
    const auto ngrid = static_cast<Eigen::Index>(lnk_vals_d.size());

    for (const auto& [scen_str, h_noise] : all_scenarios()) {
        auto solvers = default_solvers("+BT");
        std::map<std::string, Eigen::VectorXd> fdr_map, tpr_map,
                                               sd_fdr_map, sd_tpr_map,
                                               avg_L_map, avg_T_map;

        for (const auto& sol : solvers) {
            fdr_map[sol.name]    = Eigen::VectorXd::Zero(ngrid);
            tpr_map[sol.name]    = Eigen::VectorXd::Zero(ngrid);
            sd_fdr_map[sol.name] = Eigen::VectorXd::Zero(ngrid);
            sd_tpr_map[sol.name] = Eigen::VectorXd::Zero(ngrid);
            avg_L_map[sol.name]  = Eigen::VectorXd::Zero(ngrid);
            avg_T_map[sol.name]  = Eigen::VectorXd::Zero(ngrid);
        }

        auto trex_ctrl = make_trex_control(K);

        for (const auto& sol : solvers) {
            trex_ctrl.solver_type           = sol.type;
            trex_ctrl.solver_params.rho_afs = sol.rho_afs;
            trex_ctrl.solver_params.exch_tie_alpha = sol.exch_tie_alpha;
            trex_ctrl.tloop_stagnation_stop = sol.use_stagnation;

            for (Eigen::Index gi = 0; gi < ngrid; ++gi) {
                const auto& [lnk_str, lnk_val] = lnk_list[static_cast<std::size_t>(gi)];
                auto da_ctrl = make_da_bt_control();
                da_ctrl.hc_linkage = lnk_val;

                std::string lbl = "Linkage = " + lnk_str;
                unsigned seed_offset = static_cast<unsigned>(base_seed)
                                     + static_cast<unsigned>(gi) * 10000u;

                auto dgp_fn = [&](unsigned seed) {
                    return dgp_block_toeplitz_hvt(
                        n, base_p, M, Q, rho, snr, seed,
                        /*heavy_X=*/true, h_noise, nu, /*random_support=*/false);
                };

                auto res = run_mc_trials(
                    num_MC, lbl, dgp_fn, tFDR, trex_ctrl, da_ctrl, seed_offset);

                fdr_map[sol.name](gi)    = res.avg_fdr;
                tpr_map[sol.name](gi)    = res.avg_tpr;
                sd_fdr_map[sol.name](gi) = res.sd_fdr;
                sd_tpr_map[sol.name](gi) = res.sd_tpr;
                avg_L_map[sol.name](gi)  = res.avg_L;
                avg_T_map[sol.name](gi)  = res.avg_T;
            }
        }

        std::string tag = "da_ht_blocks_linkage_"; tag += scen_str;
        std::string title = "HT-block Linkage sweep  n=" + std::to_string(n)
            + "  M=" + std::to_string(M) + "  Q=" + std::to_string(Q)
            + "  p=" + std::to_string(base_p) + "  s=" + std::to_string(M)
            + "  rho=" + fmt_num(rho) + "  SNR=" + fmt_num(snr)
            + "  nu=" + fmt_num(nu)
            + "  tFDR=" + fmt_num(tFDR) + "  noise=";
        title += scen_str;
        title += "\n  Linkage codes: 1=Single  2=Complete  3=Average";
        save_and_print_grid_results(
            tag,
            "Linkage",
            lnk_vals_d,
            num_MC,
            solvers,
            fdr_map, tpr_map, sd_fdr_map, sd_tpr_map, avg_L_map, avg_T_map,
            title);
    }
}


// ==============================================================================
// main
// ==============================================================================
int main() {

    std::cout.setf(std::ios::unitbuf);
    omp_set_num_threads(trex_mc_workers::count());
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    if (true) demo_ht_block_mc_snr_sweep();
    if (true) demo_ht_block_mc_rho_sweep();
    if (true) demo_ht_block_mc_q_sweep();
    if (true) demo_ht_block_mc_m_sweep();
    if (true) demo_ht_block_mc_tfdr_sweep();
    if (true) demo_ht_block_mc_linkage_sweep();

    std::cout << "\nHeavy-tailed block BT MC simulations complete.\n";
    return 0;
}
