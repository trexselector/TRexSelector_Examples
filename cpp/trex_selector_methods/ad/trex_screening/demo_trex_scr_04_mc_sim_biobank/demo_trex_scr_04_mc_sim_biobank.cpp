// ==============================================================================
// demo_trex_scr_04_mc_sim_biobank.cpp
// ==============================================================================
/**
 * @file demo_trex_scr_04_mc_sim_biobank.cpp
 *
 * @brief Biobank Screen-TRex Selector (Algorithm 1) — in-memory D-file variant.
 *
 * @details Dummy matrices are stored in memory (use_memory_mapping = false).
 *          See demo_trex_scr_05_mc_sim_biobank_mmap.cpp for the
 *          memory-mapped workflow.
 *
 *  Part 1 — Monte Carlo SNR sweep (single phenotype): per-method Usage %, FDR, TPR, Est. FDR.
 *  Part 2 — Monte Carlo SNR sweep (multiple phenotypes): same per-method metrics.
 */
// ==============================================================================

// std includes
#include <random>
#include <unordered_set>

// TRex includes
#include <utils/datageneration/utils_datagen.hpp>

// Biobank Screen-TRex demo infrastructure
#include "trex_screening_simulation_utils.hpp"

// ==============================================================================
// Namespace aliases
// ==============================================================================

namespace datagen = trex::utils::datageneration::datagen;
using namespace scr_demo;


// ==============================================================================
// Part 1 — Monte Carlo SNR sweep
// ==============================================================================

void demo_Biobank_MonteCarlo(std::size_t num_MC)
{
    std::cout.setf(std::ios::unitbuf);
    cdiag::print_section_header(
        "Part 1: Biobank Screen-TRex — MC SNR Sweep (In-Memory)");

    const std::size_t n            = 300;
    const std::size_t p            = 1000;
    const std::size_t support_size = 10;
    const std::vector<double> snr_values = {0.1, 0.5, 1.0, 2.0, 5.0};
    // Map keys are fixed by the library (BiobankScreenTRexResult::method_used);
    // display_names is what the saved tables and figure legends show.
    const std::vector<std::string> method_names = {
        "Screen-TRex (ordinary)",
        "Screen-TRex (bootstrap-CI)",
        "T-Rex (fallback)"
    };
    const std::vector<std::string> display_names = {
        "Screen-TRex Ordinary: TLARS",
        "Screen-TRex Bootstrap-CI: TLARS",
        "T-Rex fallback: TLARS"
    };
    const Eigen::Index ns = static_cast<Eigen::Index>(snr_values.size());

    // Accumulators: fdp/tpp are conditional on method being used
    std::map<std::string, Eigen::VectorXd> fdp_map, tpp_map, est_fdr_map, usage_map;
    for (const auto& nm : method_names) {
        fdp_map[nm]     = Eigen::VectorXd::Zero(ns);
        tpp_map[nm]     = Eigen::VectorXd::Zero(ns);
        est_fdr_map[nm] = Eigen::VectorXd::Zero(ns);
        usage_map[nm]   = Eigen::VectorXd::Zero(ns);
    }

    for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {
        const double       snr = snr_values[snr_idx];
        const Eigen::Index si  = static_cast<Eigen::Index>(snr_idx);

        // One SNR cell, restored from the cache after a restart. Its supports
        // come from an RNG seeded per cell, so the cells are independent.
        const auto compute_cell = [&]() -> trex_cell_cache::Cell {
            std::mt19937 rng(24u + 1000u * static_cast<unsigned>(snr_idx));
            std::uniform_int_distribution<std::size_t> idx_dist(0, p - 1);
            BiobankCellSums acc;
            for (std::size_t mc = 0; mc < num_MC; ++mc) {
                // Random sparse support
                std::unordered_set<std::size_t> sup_set;
                while (sup_set.size() < support_size) { sup_set.insert(idx_dist(rng)); }
                const std::vector<std::size_t> true_support(sup_set.begin(), sup_set.end());
                const std::vector<double>      true_coefs(support_size, 1.0);

                datagen::SyntheticData data(
                    static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(p),
                    true_support, true_coefs, snr,
                    /*seed=*/static_cast<int>(1000 * snr_idx + mc));

                Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(),
                                                  data.rows(),
                                                  data.cols());
                Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), data.rows());

                auto ctrl = makeBiobankControl(/*use_mmap=*/false);
                tbs::BiobankScreenTRex biosctrex(X_map,
                                                 y_map,
                                                 ctrl,
                                                 /*seed=*/-1,
                                                 /*verbose=*/false);
                const auto result = biosctrex.screenPhenotype();

                // Accumulate FDP/TPP only for the method that was actually used
                acc.fdp[result.method_used]  += rates::compute_fdp(
                    result.selected_indices, true_support);
                acc.tpp[result.method_used]  += rates::compute_tpp(
                    result.selected_indices, true_support);
                acc.usage[result.method_used] += 1.0;

                // Estimated FDR values accumulated unconditionally for all methods
                acc.est_fdr["Screen-TRex (ordinary)"]     +=
                                                            result.estimated_FDR_screen_ordinary;
                acc.est_fdr["Screen-TRex (bootstrap-CI)"] +=
                                                            result.estimated_FDR_screen_bootstrap;
                acc.est_fdr["T-Rex (fallback)"]           += ctrl.target_FDR_trex;

                print_mc_progress(snr, mc, num_MC);
            }
            print_mc_done(snr, num_MC);
            return acc.to_cell(method_names);
        };
        std::ostringstream key;
        key << "biobank1|snr=" << snr << "|mc=" << num_MC;
        BiobankCellSums::add_to(TREX_CELL_CACHED(key.str(), compute_cell), method_names, si,
                                fdp_map, tpp_map, est_fdr_map, usage_map);
    }

    // ── Normalise ──────────────────────────────────────────────────────────
    for (const auto& nm : method_names) {
        for (Eigen::Index i = 0; i < ns; ++i) {
            if (usage_map[nm](i) > 0.0) {
                fdp_map[nm](i) /= usage_map[nm](i);   // conditional mean FDP
                tpp_map[nm](i) /= usage_map[nm](i);   // conditional mean TPR
            }
            est_fdr_map[nm](i) /= static_cast<double>(num_MC);   // unconditional mean
            usage_map[nm](i)   /= static_cast<double>(num_MC);   // fraction in [0, 1]
        }
    }

    // ── Build file stem and save results ───────────────────────────────────
    std::ostringstream stem;
    stem << "scr_biobank_snr_n" << n << "_p" << p
         << "_s" << support_size;

    save_and_print_biobank_mc_results(
        num_MC, stem.str(), snr_values, display_names,
        rekey_for_display(method_names, display_names, fdp_map),
        rekey_for_display(method_names, display_names, tpp_map),
        rekey_for_display(method_names, display_names, est_fdr_map),
        rekey_for_display(method_names, display_names, usage_map));
}


// ==============================================================================
// Part 2 — MC SNR sweep — multiple phenotypes
// ==============================================================================

void demo_Biobank_MultiplePhenotypesMC(std::size_t num_MC)
{
    std::cout.setf(std::ios::unitbuf);
    cdiag::print_section_header(
        "Part 2: Biobank Screen-TRex — MC SNR Sweep, Multiple Phenotypes (In-Memory)");

    const std::size_t n            = 300;
    const std::size_t p            = 1000;
    const std::size_t q            = 5;    // phenotypes per MC run
    const std::size_t support_size = 5;
    const std::vector<double> snr_values = {0.1, 0.5, 1.0, 2.0, 5.0};
    // Map keys are fixed by the library (BiobankScreenTRexResult::method_used);
    // display_names is what the saved tables and figure legends show.
    const std::vector<std::string> method_names = {
        "Screen-TRex (ordinary)",
        "Screen-TRex (bootstrap-CI)",
        "T-Rex (fallback)"
    };
    const std::vector<std::string> display_names = {
        "Screen-TRex Ordinary: TLARS",
        "Screen-TRex Bootstrap-CI: TLARS",
        "T-Rex fallback: TLARS"
    };
    const Eigen::Index ns = static_cast<Eigen::Index>(snr_values.size());

    // Accumulators over (q × num_MC) total phenotype screenings per SNR level
    std::map<std::string, Eigen::VectorXd> fdp_map, tpp_map, est_fdr_map, usage_map;
    for (const auto& nm : method_names) {
        fdp_map[nm]     = Eigen::VectorXd::Zero(ns);
        tpp_map[nm]     = Eigen::VectorXd::Zero(ns);
        est_fdr_map[nm] = Eigen::VectorXd::Zero(ns);
        usage_map[nm]   = Eigen::VectorXd::Zero(ns);
    }

    const std::vector<double> coefs(support_size, 1.0);

    for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {
        const double       snr = snr_values[snr_idx];
        const Eigen::Index si  = static_cast<Eigen::Index>(snr_idx);

        // One SNR cell, restored from the cache after a restart. Its supports
        // come from an RNG seeded per cell, so the cells are independent.
        const auto compute_cell = [&]() -> trex_cell_cache::Cell {
            std::mt19937 rng(77u + 1000u * static_cast<unsigned>(snr_idx));
            std::uniform_int_distribution<std::size_t> idx_dist(0, p - 1);
            BiobankCellSums acc;
            for (std::size_t mc = 0; mc < num_MC; ++mc) {
                // All phenotypes in this run share the same X via a fixed X_seed
                const int x_seed = static_cast<int>(5000 * snr_idx + mc + 1);

                // Draw random sparse supports for all q phenotypes
                std::vector<std::vector<std::size_t>> true_supports(q);
                for (std::size_t k = 0; k < q; ++k) {
                    std::unordered_set<std::size_t> sup_set;
                    while (sup_set.size() < support_size) { sup_set.insert(idx_dist(rng)); }
                    true_supports[k] = std::vector<std::size_t>(sup_set.begin(), sup_set.end());
                }

                // Generate all SyntheticData with the same X_seed → shared design matrix
                std::vector<datagen::SyntheticData> pheno_data;
                pheno_data.reserve(q);
                for (std::size_t k = 0; k < q; ++k) {
                    pheno_data.emplace_back(
                        static_cast<Eigen::Index>(n),
                        static_cast<Eigen::Index>(p),
                        /*num_dummies=*/0,
                        true_supports[k], coefs, snr,
                        /*seed=*/static_cast<int>(q * (num_MC * snr_idx + mc) + k + 1),
                        /*X_seed=*/x_seed,
                        /*dummy_seed=*/-1);
                }

                // X is identical for all phenotypes — taken from phenotype 0
                Eigen::Map<Eigen::MatrixXd> X_map(
                    pheno_data[0].getX().data(),
                    pheno_data[0].rows(),
                    pheno_data[0].cols());

                // Stack Y columns
                Eigen::MatrixXd Y(static_cast<Eigen::Index>(n),
                                  static_cast<Eigen::Index>(q));
                for (std::size_t k = 0; k < q; ++k)
                    Y.col(static_cast<Eigen::Index>(k)) = pheno_data[k].getY();
                Eigen::Map<Eigen::MatrixXd> Y_map(Y.data(), Y.rows(), Y.cols());

                auto ctrl = makeBiobankControl(/*use_mmap=*/false);
                tbs::BiobankScreenTRex biosctrex(X_map,
                                                 Y_map,
                                                 ctrl,
                                                 /*seed=*/-1,
                                                 /*verbose=*/false);
                const auto results = biosctrex.screenPhenotypes();

                // Accumulate per phenotype
                for (std::size_t k = 0; k < q; ++k) {
                    const auto& r = results[k];
                    acc.fdp[r.method_used]  += rates::compute_fdp(r.selected_indices,
                                                                       true_supports[k]);
                    acc.tpp[r.method_used]  += rates::compute_tpp(r.selected_indices,
                                                                       true_supports[k]);
                    acc.usage[r.method_used] += 1.0;
                    acc.est_fdr["Screen-TRex (ordinary)"]     += r.estimated_FDR_screen_ordinary;
                    acc.est_fdr["Screen-TRex (bootstrap-CI)"] += r.estimated_FDR_screen_bootstrap;
                    acc.est_fdr["T-Rex (fallback)"]           += ctrl.target_FDR_trex;
                }

                print_mc_progress(snr, mc, num_MC);
            }
            print_mc_done(snr, num_MC);
            return acc.to_cell(method_names);
        };
        std::ostringstream key;
        key << "biobank2|snr=" << snr << "|q=" << q << "|mc=" << num_MC;
        BiobankCellSums::add_to(TREX_CELL_CACHED(key.str(), compute_cell), method_names, si,
                                fdp_map, tpp_map, est_fdr_map, usage_map);
    }

    // ── Normalise over (q × num_MC) total phenotype screenings ────────────
    const double total = static_cast<double>(q) * static_cast<double>(num_MC);
    for (const auto& nm : method_names) {
        for (Eigen::Index i = 0; i < ns; ++i) {
            if (usage_map[nm](i) > 0.0) {
                fdp_map[nm](i) /= usage_map[nm](i);   // conditional mean FDP
                tpp_map[nm](i) /= usage_map[nm](i);   // conditional mean TPR
            }
            est_fdr_map[nm](i) /= total;   // unconditional mean
            usage_map[nm](i)   /= total;   // fraction in [0, 1]
        }
    }

    // ── Build file stem and save results ───────────────────────────────────
    std::ostringstream stem;
    stem << "scr_biobank_multi_n" << n << "_p" << p
         << "_q" << q << "_s" << support_size;

    save_and_print_biobank_mc_results(
        num_MC, stem.str(), snr_values, display_names,
        rekey_for_display(method_names, display_names, fdp_map),
        rekey_for_display(method_names, display_names, tpp_map),
        rekey_for_display(method_names, display_names, est_fdr_map),
        rekey_for_display(method_names, display_names, usage_map));
}


// ==============================================================================
// main
// ==============================================================================

int main()
{
    std::cout.setf(std::ios::unitbuf);

    // Part 1: Monte Carlo SNR sweep — single phenotype
    demo_Biobank_MonteCarlo(/*num_MC=*/200);

    // Part 2: Monte Carlo SNR sweep — multiple phenotypes
    demo_Biobank_MultiplePhenotypesMC(/*num_MC=*/200);

    return 0;
}
// ==============================================================================
