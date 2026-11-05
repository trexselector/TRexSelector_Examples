// ==============================================================================
// demo_trex_gvs_01_mc_sim_hastie_en_blocks.cpp
// ==============================================================================
/**
 * @file demo_trex_gvs_01_mc_sim_hastie_en_blocks.cpp
 *
 * @brief MC simulation: T-Rex+GVS on the Hastie DGP.
 *  This scenario evaluates grouped-signal recovery under strong equicorrelation,
 *  not sparse within-group support identification.
 *  Within-group correlation is controlled through sd_x, with rho = 1 / (1 + sd_x^2).
 *
 *
 * @details
 *  Four parts, each in its own function:
 *    sim_hastie_part_1() — 2-D SNR × rho grid
 *    sim_hastie_part_2() — lambda2_method comparison: SVD / CCD (EN), TIK_SVD / IEN_CCD (IEN)
 *    sim_hastie_part_3() — hc_linkage comparison: Single / Complete / Average / WPGMA
 *    sim_hastie_part_4() — clustering cut: corr_max 0.98 / noise floor / oracle groups
 *
 * DGP: three equicorrelated groups of 50 variables each, all active (s = 150),
 *      plus 350 null white-noise variables; p = 500 and n = 200.
 *  Selector: K=20, tFDR=0.1, corr_max=0.98, lambda2=CV_1SE_CCD on the EN track and the IEN-geometry
 *  CV_1SE_IEN_CCD on the IEN track (Part 2 compares it with CV_1SE_TIK_SVD).
 *  Methods compared in Parts 1–3: TENET, TENET_AUG, TIENET_AUG (Part 4: TENET, TIENET_AUG).
 *  MC: 200 trials per grid point.
 *
 *  Part selection: TREX_GVS_PARTS (default "1234"), e.g. TREX_GVS_PARTS=4
 *  runs only the cut comparison.
 */
// ==============================================================================

#include "trex_gvs_dgps.hpp"
#include "trex_gvs_simulation_utils.hpp"


// ==============================================================================

// Namespace usage
using namespace gvs_demo;

// ==============================================================================
//  Part 1 — 2-D SNR × rho sweep
// ==============================================================================

/**
 * @brief Run a 2-D MC sweep over SNR and rho on the Hastie DGP.
 *
 * Evaluates TENET, TENET_AUG, and TIENET_AUG on an SNR × rho grid, then
 * prints and saves matrix summaries for mean TPP and mean FDP.
 */

static void sim_hastie_part_1(const GVSSimConfig& cfg, const std::string& tag)
{
    const std::vector<double> snr_grid_2d = {
        0.2, 0.5, 1.0, 2.0, 5.0
    };
    const std::vector<double> rho_grid_2d = {
        0.30, 0.50, 0.70, 0.90, 0.95, 0.99
    };

    GVS2DDGPFactory dgp_2d = [&cfg](double snr, double rho, unsigned seed) {
        const double sd_x = std::sqrt((1.0 - rho) / rho);
        return make_hastie_dgp(cfg.n, cfg.p, snr, sd_x, seed);
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
    gvs_ctrl_ien.cv_dfmax_ratio = 1.0;  // support cap 180 > s = 150 (library default 0.8 gives 144)
    gvs_ctrl_ien.corr_max       = cfg.corr_max;
    gvs_ctrl_ien.hc_linkage     = hac::LinkageMethod::Single;

    cdiag::print_section_header("Part 1/4: 2-D SNR x rho Sweep  |  " + tag);
    std::cout << "  n=" << cfg.n << "  p=" << cfg.p << "  s=150"
              << "  MC=" << cfg.num_MC << "  tFDR=" << fmt_num(cfg.tFDR)
              << "  K=" << cfg.K << "  corr_max=" << fmt_num(cfg.corr_max) << "\n"
              << "  methods: TENET / TENET_AUG / TIENET_AUG  |  lambda2=CV_1SE_CCD (EN), CV_1SE_IEN_CCD (IEN)"
              << "  linkage=Single\n"
              << "  snr_grid: {";
    for (std::size_t i = 0; i < snr_grid_2d.size(); ++i)
        std::cout << (i ? ", " : "") << fmt_num(snr_grid_2d[i]);
    std::cout << "}\n  rho_grid: {";
    for (std::size_t i = 0; i < rho_grid_2d.size(); ++i)
        std::cout << (i ? ", " : "") << fmt_num(rho_grid_2d[i]);
    std::cout << "}\n\n";

    auto en_2d     = run_gvs_2d_sweep(
        dgp_2d, snr_grid_2d, rho_grid_2d, cfg,
        gvs_ctrl_en,     trex_ctrl, "TENET");

    auto en_aug_2d = run_gvs_2d_sweep(
        dgp_2d, snr_grid_2d, rho_grid_2d, cfg,
        gvs_ctrl_en_aug, trex_ctrl, "TENET_AUG");

    auto ien_2d    = run_gvs_2d_sweep(
        dgp_2d, snr_grid_2d, rho_grid_2d, cfg,
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
    save_mc_2d_tables(tag, snr_labels, rho_labels, en_2d,
                      en_aug_2d, ien_2d, cfg, "gvs_" + tag + "_2d");
}



// ==============================================================================
//  Part 2 — lambda2_method comparison
// ==============================================================================

/**
 * @brief Compare lambda2 selection methods at a fixed Hastie operating point.
 *
 * Runs TENET and TIENET_AUG at SNR = 2.0 and rho = 0.7 with two lambda2 rules per track:
 * CV_1SE_SVD vs CV_1SE_CCD for TENET, and the two IEN-geometry tuners CV_1SE_TIK_SVD vs
 * CV_1SE_IEN_CCD for TIENET_AUG (the EN track rejects them, and an EN tuner on the IEN track
 * would fit the wrong objective). With cv_dfmax_ratio = 1.0, CV_1SE_IEN_CCD caps its lambda1 path
 * at n_train = 180 variables, above this design's s = 150 (the library default 0.8 gives 144).
 * CV_1SE_TIK_SVD ignores the cap and needs p - M < n_train (M clusters, so p - M within-group
 * contrasts). The library refuses it otherwise, e.g. on the trap blocks of demo 04 at rho = 0.99,
 * so the sweeps of demos 01-07 use CV_1SE_IEN_CCD.
 */

static void sim_hastie_part_2(const GVSSimConfig& cfg, const std::string& tag)
{
    // Fixed operating point: SNR=2.0, rho=0.7.
    // sd_x = sqrt((1 - rho) / rho) = sqrt(0.3 / 0.7)
    constexpr double fixed_snr = 2.0;
    constexpr double fixed_rho = 0.7;
    const double     fixed_sdx = std::sqrt((1.0 - fixed_rho) / fixed_rho);

    GVSDGPFactory dgp_fn = [&cfg, fixed_snr, fixed_sdx](double /*unused*/, unsigned seed) {
        return make_hastie_dgp(cfg.n, cfg.p, fixed_snr, fixed_sdx, seed);
    };

    auto trex_ctrl = make_gvs_trex_control(cfg.K);

    // Rules per track: the IEN-geometry tuners are valid on the IEN track only.
    const std::vector<LambdaSelectionMethod> en_methods = {
        LambdaSelectionMethod::CV_1SE_SVD,
        LambdaSelectionMethod::CV_1SE_CCD
    };
    const std::vector<std::string> en_labels = {"CV_1SE_SVD", "CV_1SE_CCD"};
    const std::vector<LambdaSelectionMethod> ien_methods = {
        LambdaSelectionMethod::CV_1SE_TIK_SVD,
        LambdaSelectionMethod::CV_1SE_IEN_CCD
    };
    const std::vector<std::string> ien_labels = {"CV_1SE_TIK_SVD", "CV_1SE_IEN_CCD"};

    cdiag::print_section_header("Part 2/4: lambda2_method Comparison  |  " + tag);
    std::cout << "  n=" << cfg.n << "  p=" << cfg.p << "  s=150"
              << "  MC=" << cfg.num_MC << "  tFDR=" << fmt_num(cfg.tFDR)
              << "  K=" << cfg.K << "\n"
              << "  fixed operating point: SNR=" << fmt_num(fixed_snr)
              << "  rho=" << fmt_num(fixed_rho) << "\n"
              << "  lambda2 methods: TENET CV_1SE_SVD / CV_1SE_CCD,"
              << " TIENET_AUG CV_1SE_TIK_SVD / CV_1SE_IEN_CCD  (4 runs)\n\n";

    std::vector<GVSGridPointResult> en_results, ien_results;
    en_results.reserve(en_methods.size());
    ien_results.reserve(ien_methods.size());

    for (std::size_t i = 0; i < en_methods.size(); ++i) {
        TRexGVSControlParameter gvs_ctrl_en;
        gvs_ctrl_en.gvs_type       = GVSType::EN;
        gvs_ctrl_en.lambda2_method = en_methods[i];
        gvs_ctrl_en.corr_max       = cfg.corr_max;
        gvs_ctrl_en.hc_linkage     = hac::LinkageMethod::Single;
        gvs_ctrl_en.en_solver      = ENSolverType::TENET;

        en_results.push_back(
            run_gvs_mc_trials(dgp_fn, 0.0, cfg.num_MC,
                              "TENET+" + en_labels[i],
                              cfg.tFDR, gvs_ctrl_en, trex_ctrl,
                              static_cast<unsigned>(cfg.base_seed)));

        TRexGVSControlParameter gvs_ctrl_ien;
        gvs_ctrl_ien.gvs_type       = GVSType::IEN;
        gvs_ctrl_ien.lambda2_method = ien_methods[i];
        gvs_ctrl_ien.cv_dfmax_ratio = 1.0;
        gvs_ctrl_ien.corr_max       = cfg.corr_max;
        gvs_ctrl_ien.hc_linkage     = hac::LinkageMethod::Single;

        ien_results.push_back(
            run_gvs_mc_trials(dgp_fn, 0.0, cfg.num_MC,
                              "TIENET_AUG+" + ien_labels[i],
                              cfg.tFDR,
                              gvs_ctrl_ien, trex_ctrl,
                              static_cast<unsigned>(cfg.base_seed)));
    }

    {
        const std::string dir(DEMO_OUTPUT_DIR);
        const std::string stem = "gvs_" + tag + "_lambda2_method";
        std::filesystem::create_directories(dir);
        std::ofstream txt_file(dir + out_stem(stem) + ".txt");
        std::ofstream* fout = txt_file.is_open() ? &txt_file : nullptr;

        auto write = [&](const std::string& s) {
            std::cout << s;
            if (fout) *fout << s;
        };

        {
            std::ostringstream hdr;
            hdr << "\n" << std::string(78, '=') << "\n"
                << "  T-Rex+GVS MC: " << tag << " — lambda2_method comparison"
                << "  (SNR=" << fixed_snr << ", rho=" << fixed_rho
                << ", MC=" << cfg.num_MC << ", tFDR=" << cfg.tFDR << ")\n"
                << std::string(78, '=') << "\n";
            write(hdr.str());
        }

        print_mc_1d_method_table("TENET",      en_labels,  en_results,  fout);
        print_mc_1d_method_table("TIENET_AUG", ien_labels, ien_results, fout);

        write(std::string(78, '=') + "\n\n");

        if (txt_file.is_open()) {
            txt_file.close();
            std::cout << "[Info] TXT saved to: " << dir + out_stem(stem) + ".txt\n";
        }
        std::ofstream csv(dir + out_stem(stem) + ".csv");
        if (csv.is_open()) {
            csv << "method,metric,lambda2_method,mean,sd\n"
                << std::fixed << std::setprecision(6);
            for (std::size_t i = 0; i < en_labels.size(); ++i) {
                const std::string& lbl = en_labels[i];
                csv << "TENET,mean_FDP,"  << lbl << "," << en_results[i].mean_fdp  << "," << en_results[i].sd_fdp  << "\n";
                csv << "TENET,mean_TPP,"  << lbl << "," << en_results[i].mean_tpp  << "," << en_results[i].sd_tpp  << "\n";
            }
            for (std::size_t i = 0; i < ien_labels.size(); ++i) {
                const std::string& lbl = ien_labels[i];
                csv << "TIENET_AUG,mean_FDP," << lbl << "," << ien_results[i].mean_fdp << "," << ien_results[i].sd_fdp << "\n";
                csv << "TIENET_AUG,mean_TPP," << lbl << "," << ien_results[i].mean_tpp << "," << ien_results[i].sd_tpp << "\n";
            }
            std::cout << "[Info] CSV saved to: " << dir + out_stem(stem) + ".csv\n\n";
        }
    }
}


// ==============================================================================
//  Part 3 — hc_linkage comparison
// ==============================================================================

/**
 * @brief Compare HAC linkage rules at a fixed Hastie operating point.
 *
 * Runs TENET and TIENET_AUG at SNR = 2.0 and rho = 0.7, comparing Single, Complete,
 * Average, and WPGMA linkage with lambda2 fixed at CV_1SE_CCD (EN) and CV_1SE_IEN_CCD (IEN).
 */

static void sim_hastie_part_3(const GVSSimConfig& cfg, const std::string& tag)
{
    // Same fixed operating point as Part 2: SNR=2.0, rho=0.7.
    // One lambda2 rule per track throughout, to isolate the linkage effect from lambda selection.
    constexpr double fixed_snr = 2.0;
    constexpr double fixed_rho = 0.7;
    const double     fixed_sdx = std::sqrt((1.0 - fixed_rho) / fixed_rho);

    GVSDGPFactory dgp_fn = [&cfg, fixed_snr, fixed_sdx](double /*unused*/, unsigned seed) {
        return make_hastie_dgp(cfg.n, cfg.p, fixed_snr, fixed_sdx, seed);
    };

    auto trex_ctrl = make_gvs_trex_control(cfg.K);

    const std::vector<hac::LinkageMethod> linkages = {
        hac::LinkageMethod::Single,
        hac::LinkageMethod::Complete,
        hac::LinkageMethod::Average,
        hac::LinkageMethod::WPGMA
    };
    const std::vector<std::string> linkage_labels = {
        "Single", "Complete", "Average", "WPGMA"};

    cdiag::print_section_header("Part 3/4: hc_linkage Comparison  |  " + tag);
    std::cout << "  n=" << cfg.n << "  p=" << cfg.p << "  s=150"
              << "  MC=" << cfg.num_MC << "  tFDR=" << fmt_num(cfg.tFDR)
              << "  K=" << cfg.K << "  lambda2=CV_1SE_CCD (EN), CV_1SE_IEN_CCD (IEN)\n"
              << "  fixed operating point: SNR=" << fmt_num(fixed_snr)
              << "  rho=" << fmt_num(fixed_rho) << "\n"
              << "  linkages: Single / Complete / Average / WPGMA"
              << "  |  selectors: TENET, TIENET_AUG  (8 runs)\n\n";

    std::vector<GVSGridPointResult> en_results, ien_results;
    en_results.reserve(linkages.size());
    ien_results.reserve(linkages.size());

    for (std::size_t i = 0; i < linkages.size(); ++i) {
        TRexGVSControlParameter gvs_ctrl_en;
        gvs_ctrl_en.gvs_type       = GVSType::EN;
        gvs_ctrl_en.lambda2_method = LambdaSelectionMethod::CV_1SE_CCD;
        gvs_ctrl_en.corr_max       = cfg.corr_max;
        gvs_ctrl_en.hc_linkage     = linkages[i];
        gvs_ctrl_en.en_solver      = ENSolverType::TENET;

        en_results.push_back(
            run_gvs_mc_trials(dgp_fn, 0.0, cfg.num_MC,
                              "TENET+" + linkage_labels[i],
                              cfg.tFDR, gvs_ctrl_en, trex_ctrl,
                              static_cast<unsigned>(cfg.base_seed)));

        TRexGVSControlParameter gvs_ctrl_ien;
        gvs_ctrl_ien.gvs_type       = GVSType::IEN;
        gvs_ctrl_ien.lambda2_method = LambdaSelectionMethod::CV_1SE_IEN_CCD;
        gvs_ctrl_ien.cv_dfmax_ratio = 1.0;
        gvs_ctrl_ien.corr_max       = cfg.corr_max;
        gvs_ctrl_ien.hc_linkage     = linkages[i];

        ien_results.push_back(
            run_gvs_mc_trials(dgp_fn, 0.0, cfg.num_MC,
                              "TIENET_AUG+" + linkage_labels[i],
                              cfg.tFDR, gvs_ctrl_ien, trex_ctrl,
                              static_cast<unsigned>(cfg.base_seed)));
    }

    {
        const std::string dir(DEMO_OUTPUT_DIR);
        const std::string stem = "gvs_" + tag + "_hc_linkage";
        std::filesystem::create_directories(dir);
        std::ofstream txt_file(dir + out_stem(stem) + ".txt");
        std::ofstream* fout = txt_file.is_open() ? &txt_file : nullptr;

        auto write = [&](const std::string& s) {
            std::cout << s;
            if (fout) *fout << s;
        };

        {
            std::ostringstream hdr;
            hdr << "\n" << std::string(78, '=') << "\n"
                << "  T-Rex+GVS MC: " << tag << " — hc_linkage comparison"
                << "  (SNR=" << fixed_snr << ", rho=" << fixed_rho
                << ", MC=" << cfg.num_MC << ", tFDR=" << cfg.tFDR << ")\n"
                << std::string(78, '=') << "\n";
            write(hdr.str());
        }

        print_mc_1d_method_table("TENET",  linkage_labels, en_results,  fout);
        print_mc_1d_method_table("TIENET_AUG", linkage_labels, ien_results, fout);

        write(std::string(78, '=') + "\n\n");

        if (txt_file.is_open()) {
            txt_file.close();
            std::cout << "[Info] TXT saved to: " << dir + out_stem(stem) + ".txt\n";
        }
        std::ofstream csv(dir + out_stem(stem) + ".csv");
        if (csv.is_open()) {
            csv << "method,metric,hc_linkage,mean,sd\n"
                << std::fixed << std::setprecision(6);
            for (std::size_t i = 0; i < linkage_labels.size(); ++i) {
                const std::string& lbl = linkage_labels[i];
                csv << "TENET,mean_FDP,"  << lbl << "," << en_results[i].mean_fdp  << "," << en_results[i].sd_fdp  << "\n";
                csv << "TENET,mean_TPP,"  << lbl << "," << en_results[i].mean_tpp  << "," << en_results[i].sd_tpp  << "\n";
                csv << "TIENET_AUG,mean_FDP," << lbl << "," << ien_results[i].mean_fdp << "," << ien_results[i].sd_fdp << "\n";
                csv << "TIENET_AUG,mean_TPP," << lbl << "," << ien_results[i].mean_tpp << "," << ien_results[i].sd_tpp << "\n";
            }
            std::cout << "[Info] CSV saved to: " << dir + out_stem(stem) + ".csv\n\n";
        }
    }
}


// ==============================================================================
//  Part 4 — clustering-cut comparison
// ==============================================================================

/**
 * @brief Compare the clustering cut at a fixed Hastie operating point.
 *
 * Runs TENET and TIENET_AUG at SNR = 2.0 and rho = 0.7 with the demos' fixed
 * cut corr_max = 0.98 (no groups below rho = 0.99), the Bonferroni noise floor
 * (noise_floor_corr_max(), about 0.35 here) and the DGP's oracle groups, and
 * records variable-level FDP / TPP next to group-level rates.
 */
static void sim_hastie_part_4(const GVSSimConfig& cfg, const std::string& tag)
{
    // Same fixed operating point as Parts 2-3, lambda2 rules as in Parts 1 and 3
    constexpr double fixed_snr = 2.0;
    constexpr double fixed_rho = 0.7;
    const double     fixed_sdx = std::sqrt((1.0 - fixed_rho) / fixed_rho);
    const double     floor_cut = noise_floor_corr_max(cfg.n, cfg.p);

    GVSDGPFactory dgp_fn = [&cfg, fixed_snr, fixed_sdx](double /*unused*/, unsigned seed) {
        return make_hastie_dgp(cfg.n, cfg.p, fixed_snr, fixed_sdx, seed);
    };
    auto trex_ctrl = make_gvs_trex_control(cfg.K);

    struct Cut { std::string label; double corr_max; bool oracle; };
    const std::vector<Cut> cuts = {
        {"fixed_0.98", cfg.corr_max, false},
        {"noise_floor", floor_cut, false},
        {"oracle", cfg.corr_max, true}};

    cdiag::print_section_header("Part 4/4: Clustering-cut Comparison  |  " + tag);
    std::cout << "  n=" << cfg.n << "  p=" << cfg.p << "  s=150"
              << "  MC=" << cfg.num_MC << "  tFDR=" << fmt_num(cfg.tFDR)
              << "  K=" << cfg.K << "  lambda2=CV_1SE_CCD (EN), CV_1SE_IEN_CCD (IEN)  linkage=Single\n"
              << "  fixed operating point: SNR=" << fmt_num(fixed_snr)
              << "  rho=" << fmt_num(fixed_rho) << "\n"
              << "  cuts: corr_max " << fmt_num(cfg.corr_max) << " / noise floor "
              << std::fixed << std::setprecision(3) << floor_cut
              << " / oracle groups  |  selectors: TENET, TIENET_AUG  (6 runs)\n\n";

    std::vector<GVSCutPointResult> en_results, ien_results;
    for (const auto& cut : cuts) {
        TRexGVSControlParameter gvs_ctrl_en;
        gvs_ctrl_en.gvs_type       = GVSType::EN;
        gvs_ctrl_en.lambda2_method = LambdaSelectionMethod::CV_1SE_CCD;
        gvs_ctrl_en.corr_max       = cut.corr_max;
        gvs_ctrl_en.hc_linkage     = hac::LinkageMethod::Single;
        gvs_ctrl_en.en_solver      = ENSolverType::TENET;
        en_results.push_back(run_gvs_mc_trials_cut(
            dgp_fn, cfg.num_MC, "TENET+" + cut.label, cfg.tFDR, gvs_ctrl_en, trex_ctrl,
            static_cast<unsigned>(cfg.base_seed), cut.oracle));

        TRexGVSControlParameter gvs_ctrl_ien;
        gvs_ctrl_ien.gvs_type       = GVSType::IEN;
        gvs_ctrl_ien.lambda2_method = LambdaSelectionMethod::CV_1SE_IEN_CCD;
        gvs_ctrl_ien.cv_dfmax_ratio = 1.0;
        gvs_ctrl_ien.corr_max       = cut.corr_max;
        gvs_ctrl_ien.hc_linkage     = hac::LinkageMethod::Single;
        ien_results.push_back(run_gvs_mc_trials_cut(
            dgp_fn, cfg.num_MC, "TIENET_AUG+" + cut.label, cfg.tFDR, gvs_ctrl_ien, trex_ctrl,
            static_cast<unsigned>(cfg.base_seed), cut.oracle));
    }

    const std::string dir(DEMO_OUTPUT_DIR);
    const std::string stem = "gvs_" + tag + "_cut";
    std::filesystem::create_directories(dir);
    std::ofstream txt_file(dir + out_stem(stem) + ".txt");
    auto write = [&](const std::string& s) {
        std::cout << s;
        if (txt_file.is_open()) txt_file << s;
    };

    std::ostringstream os;
    os << "\n" << std::string(78, '=') << "\n"
       << "  T-Rex+GVS MC: " << tag << " — clustering-cut comparison"
       << "  (SNR=" << fixed_snr << ", rho=" << fixed_rho
       << ", MC=" << cfg.num_MC << ", tFDR=" << cfg.tFDR << ")\n"
       << "  noise floor corr_max = " << std::fixed << std::setprecision(4) << floor_cut
       << " (Bonferroni, alpha = 0.05)\n"
       << std::string(78, '=') << "\n";
    const auto block = [&](const std::string& name, const std::vector<GVSCutPointResult>& rs) {
        os << "\n" << name << "\n" << std::left << std::setw(16) << "  metric";
        for (const auto& cut : cuts) os << std::right << std::setw(14) << cut.label;
        os << "\n";
        const auto row = [&](const std::string& m, auto get) {
            os << std::left << std::setw(16) << ("  " + m);
            for (const auto& r : rs) os << std::right << std::setw(14) << std::fixed
                                        << std::setprecision(4) << get(r);
            os << "\n";
        };
        row("FDR",        [](const GVSCutPointResult& r) { return r.mean_fdp; });
        row("TPR",        [](const GVSCutPointResult& r) { return r.mean_tpp; });
        row("group hit",  [](const GVSCutPointResult& r) { return r.grp_hit; });
        row("group full", [](const GVSCutPointResult& r) { return r.grp_full; });
        row("group FDR",  [](const GVSCutPointResult& r) { return r.grp_fdp; });
        row("groups",     [](const GVSCutPointResult& r) { return r.n_groups; });
    };
    block("TENET", en_results);
    block("TIENET_AUG", ien_results);
    os << std::string(78, '=') << "\n\n";
    write(os.str());
    if (txt_file.is_open()) {
        txt_file.close();
        std::cout << "[Info] TXT saved to: " << dir + out_stem(stem) + ".txt\n";
    }

    std::ofstream csv(dir + out_stem(stem) + ".csv");
    if (csv.is_open()) {
        csv << "method,metric,cut,mean,sd\n" << std::fixed << std::setprecision(6);
        const auto rows = [&](const std::string& name, const std::vector<GVSCutPointResult>& rs) {
            for (std::size_t i = 0; i < cuts.size(); ++i) {
                const auto& r = rs[i];
                const std::string& c = cuts[i].label;
                csv << name << ",mean_FDP," << c << "," << r.mean_fdp << "," << r.sd_fdp << "\n"
                    << name << ",mean_TPP," << c << "," << r.mean_tpp << "," << r.sd_tpp << "\n"
                    << name << ",grp_hit,"  << c << "," << r.grp_hit  << "," << r.sd_grp_hit  << "\n"
                    << name << ",grp_full," << c << "," << r.grp_full << "," << r.sd_grp_full << "\n"
                    << name << ",grp_FDP,"  << c << "," << r.grp_fdp  << "," << r.sd_grp_fdp  << "\n"
                    << name << ",n_groups," << c << "," << r.n_groups << "," << r.sd_n_groups << "\n";
            }
        };
        rows("TENET", en_results);
        rows("TIENET_AUG", ien_results);
        std::cout << "[Info] CSV saved to: " << dir + out_stem(stem) + ".csv\n\n";
    }
}


// ==============================================================================
//  main
// ==============================================================================

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
    cfg.sd_x      = 0.1;    // sqrt(0.01) => rho ≈ 0.99
    cfg.tFDR      = 0.1;
    cfg.K         = 20;
    cfg.num_MC    = num_mc(200);
    cfg.base_seed = 2026;
    cfg.corr_max  = 0.98;
    cfg.snr       = 2.0;    // unused here; Parts 2-3 fix SNR = 2.0 locally

    const std::string scenario_tag = "Hastie";

    // ==========================================================================
    //  Run simulations
    // ==========================================================================

    // TREX_GVS_PARTS selects parts, e.g. "4" (default "1234"); not part of the cache fingerprint
    const char* parts_env = std::getenv("TREX_GVS_PARTS");
    const std::string parts = (parts_env && *parts_env) ? parts_env : "1234";
    const auto run = [&parts](char p) { return parts.find(p) != std::string::npos; };

    if (run('1')) sim_hastie_part_1(cfg, scenario_tag);
    if (run('2')) sim_hastie_part_2(cfg, scenario_tag);
    if (run('3')) sim_hastie_part_3(cfg, scenario_tag);
    if (run('4')) sim_hastie_part_4(cfg, scenario_tag);

    std::cout << "\n" << scenario_tag << " GVS MC simulations complete.\n";
    return 0;
}
