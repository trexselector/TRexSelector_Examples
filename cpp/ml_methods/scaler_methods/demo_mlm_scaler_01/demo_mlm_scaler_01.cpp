// ===================================================================================
// demo_mlm_scaler_01.cpp
// ===================================================================================
/**
 * @file demo_mlm_scaler_01.cpp
 *
 * @brief Demonstration of ZScoreScaler and LpNormScaler usage.
 *
 * Organization:
 * -------------
 * |-- ml_methods/
 *     |-- scaler_methods/
 *         |-- demo_mlm_scaler_01.cpp
 */
// ===================================================================================

// std includes
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

// Eigen includes
#include <Eigen/Dense>

// TRex Selector includes
#include <ml_methods/scaler_methods/lp_norm_scaler.hpp>
#include <ml_methods/scaler_methods/z_score_scaler.hpp>
#include <utils/openmp/utils_openmp.hpp>
#include <utils/memmap/memory_mapped_matrix.hpp>
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>

// ============================================================================
// Namespace aliases
// ============================================================================

namespace scaler = trex::ml_methods::scaler_methods;
namespace omp_utils = trex::utils::openmp;
namespace memmap = trex::utils::memmap;
namespace cdiagnostics = trex::utils::eval::cdiagnostics;


// ============================================================================
// Helper: aggregate column statistics
// ============================================================================

/**
 * @brief Prints compact per-column statistics, averaged over all columns.
 *
 * Reveals the effect of the center / scale switches at a glance:
 *   - centering drives the average absolute column mean toward 0,
 *   - z-score scaling drives the average column SD toward 1,
 *   - L2 normalization drives the average column L2 norm toward 1.
 */
void print_column_stats(const std::string& label, const Eigen::MatrixXd& X) {
    const Eigen::Index n = X.rows();
    const Eigen::RowVectorXd col_mean = X.colwise().mean();
    const Eigen::RowVectorXd col_sd =
        ((X.rowwise() - col_mean).colwise().squaredNorm()
             / static_cast<double>(n - 1)).cwiseSqrt();
    const Eigen::RowVectorXd col_l2 = X.colwise().norm();

    std::cout << "  " << std::left << std::setw(36) << label
              << "avg|mean| = " << std::scientific << std::setprecision(2)
              << col_mean.cwiseAbs().mean()
              << "   avg SD = " << std::fixed << std::setprecision(4) << col_sd.mean()
              << "   avg L2 = " << std::setprecision(2) << col_l2.mean() << "\n";
}


// ============================================================================
// Demo 1: center / scale switches (R scale() semantics)
// ============================================================================

void demo_scaler_comparison() {
    cdiagnostics::print_section_header("Scaler Comparison - center / scale switches");

    // Step 1. Generate data with a deliberately non-zero mean (5.0) and
    //         non-unit spread (SD 3.0), so the effect of each switch is visible.
    const Eigen::Index n = 1'000, p = 500;
    std::mt19937 rng(123);
    std::normal_distribution<double> norm_dist(5.0, 3.0);

    Eigen::MatrixXd X(n, p);
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < p; ++j) {
            X(i, j) = norm_dist(rng);
        }
    }

    std::cout << "Data: " << n << " x " << p << " drawn from N(mean=5, sd=3)\n\n";
    print_column_stats("original (untransformed)", X);
    std::cout << "\n";

    // Step 2. ZScoreScaler under all four center / scale combinations.
    //         The (center, scale) booleans follow R's scale(): centering drives
    //         avg|mean| -> 0, scaling drives avg SD -> 1. With center=false the
    //         scale statistic is the root-mean-square around 0, not the SD.
    struct ZCase {
        bool center;
        bool scale;
        const char* label;
    };

    const ZCase z_cases[] = {
        {true,  true,  "ZScore(center=true,  scale=true )"},
        {false, true,  "ZScore(center=false, scale=true )"},
        {true,  false, "ZScore(center=true,  scale=false)"},
        {false, false, "ZScore(center=false, scale=false)"},
    };
    for (const auto& c : z_cases) {
        Eigen::MatrixXd Xt = X;
        Eigen::Map<Eigen::MatrixXd> Xt_map(Xt.data(), n, p);
        scaler::ZScoreScaler zscaler(c.center, c.scale);
        zscaler.fit(Xt_map);
        zscaler.transform_inplace(Xt_map);
        print_column_stats(c.label, Xt);
    }
    std::cout << "\n";

    // Step 3. LpNormScaler: L2 vs L1, with centering on and off (scale kept on).
    //         With scale=true each column is divided by its Lp norm around the
    //         applied center, so the average column L2 norm collapses toward 1
    //         for the L2 variant.
    struct LCase {
        scaler::LpNormScaler::NormType norm;
        bool center;
        const char* label;
    };

    const LCase l_cases[] = {
        {scaler::LpNormScaler::NormType::L2, true,  "LpNorm(L2, center=true )"},
        {scaler::LpNormScaler::NormType::L2, false, "LpNorm(L2, center=false)"},
        {scaler::LpNormScaler::NormType::L1, true,  "LpNorm(L1, center=true )"},
    };
    for (const auto& c : l_cases) {
        Eigen::MatrixXd Xt = X;
        Eigen::Map<Eigen::MatrixXd> Xt_map(Xt.data(), n, p);
        scaler::LpNormScaler lscaler(c.norm, c.center, /*scale=*/true);
        lscaler.fit(Xt_map);
        lscaler.transform_inplace(Xt_map);
        print_column_stats(c.label, Xt);
    }

    std::cout << "\n\n";
}


// =============================================================================
// Demo 2: Transform and Inverse Transform Test
// =============================================================================

void demo_scaler_inverse_transform() {
    cdiagnostics::print_section_header("Scaler Inverse Transform Accuracy");

    // Step 1. Generate random data
    const Eigen::Index n = 1'000, p = 5'000;
    std::cout << "Matrix size: " << n << " x " << p << "\n\n";

    std::mt19937 rng(456);
    std::normal_distribution<double> norm_dist(0.0, 1.0);
    Eigen::MatrixXd X_orig(n, p);
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < p; ++j) {
            X_orig(i, j) = norm_dist(rng);
        }
    }

    // Step 2. Test ZScoreScaler
    {
        Eigen::MatrixXd X = X_orig;
        Eigen::Map<Eigen::MatrixXd> X_map(X.data(), n, p);
        scaler::ZScoreScaler zscaler;

        zscaler.fit(X_map);
        zscaler.transform_inplace(X_map);
        zscaler.inverse_transform_inplace(X_map);

        double max_error = (X - X_orig).cwiseAbs().maxCoeff();
        double mean_error = (X - X_orig).cwiseAbs().mean();

        std::cout << "ZScoreScaler:\n"
                  << "  Max Error: " << std::scientific << max_error << "\n"
                  << "  Mean Error: " << mean_error << "\n\n";
    }

    // Step 3. Test LpNormScaler L2
    {
        Eigen::MatrixXd X = X_orig;
        Eigen::Map<Eigen::MatrixXd> X_map(X.data(), n, p);
        scaler::LpNormScaler l2scaler(scaler::LpNormScaler::NormType::L2, true);


        l2scaler.fit(X_map);
        l2scaler.transform_inplace(X_map);
        l2scaler.inverse_transform_inplace(X_map);

        double max_error = (X - X_orig).cwiseAbs().maxCoeff();
        double mean_error = (X - X_orig).cwiseAbs().mean();

        std::cout << "LpNormScaler L2:\n"
                  << "  Max Error: " << std::scientific << max_error << "\n"
                  << "  Mean Error: " << mean_error << "\n\n";
    }
    std::cout << "\n\n";
}


// =============================================================================
// Demo 3: Serialization
// =============================================================================

/**
 * @brief Path for console output without the machine-specific prefix.
 * @param path Absolute or relative file path.
 * @return ".../" plus the part from "cpp/" on, else the bare file name.
 */
static std::string display_path(const std::string& path) {
    const auto pos = path.find("/cpp/");
    if (pos != std::string::npos)
        return ".../" + path.substr(pos + 1);
    const auto slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

void demo_scaler_serialization() {
    cdiagnostics::print_section_header("Scaler Serialization");

    // Step 1. Generate data
    const Eigen::Index n = 1'000, p = 500;

    std::mt19937 rng(999);
    std::normal_distribution<double> norm_dist(0.0, 1.0);
    Eigen::MatrixXd X(n, p);
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < p; ++j) {
            X(i, j) = norm_dist(rng);
        }
    }

    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), n, p);

    // Step 2. Fit and save
#ifdef TREX_DEMO_SCALER_SER_PATH
    const std::string filename = TREX_DEMO_SCALER_SER_PATH;
#else
    const std::string filename = "scaler_test.bin";
#endif
    scaler::ZScoreScaler scaler;
    scaler.fit(X_map);
    scaler.save(filename);
    std::cout << "✓ Saved ZScoreScaler to '" << display_path(filename) << "'\n";

    // Step 3. Load and verify
    scaler::ZScoreScaler loaded_scaler;
    loaded_scaler.load(filename);
    std::cout << "✓ Loaded ZScoreScaler from '" << display_path(filename) << "'\n\n";

    // Verify means and scales match
    double mean_diff = (scaler.get_centers() - loaded_scaler.get_centers()).norm();
    double scale_diff = (scaler.get_scales() - loaded_scaler.get_scales()).norm();

    std::cout << "Verification:\n"
              << "  Mean difference: " << std::scientific << mean_diff << "\n"
              << "  Scale difference: " << scale_diff << "\n\n";

    if (mean_diff < 1e-10 && scale_diff < 1e-10) {
        std::cout << "✓ Serialization perfect!\n\n";
    }

    // Step 4. Clean up
    std::remove(filename.c_str());
    std::cout << "\n\n";
}


// ============================================================================
// Demo 04: Memory-Mapped Matrix
// ============================================================================

void demo_scaler_on_mmap_matrix() {
    cdiagnostics::print_section_header("Scaler on Memory-Mapped Matrix");

    // Step 1. Create memory-mapped matrix
    std::cout << "Creating memory-mapped matrix...\n";
    const Eigen::Index n = 20'000, p = 100'000;
    std::cout << "Matrix size: " << n << " x " << p << "\n";
#ifdef TREX_DEMO_MMAP_PATH
    const std::string mmap_filename = TREX_DEMO_MMAP_PATH;
#else
    const std::string mmap_filename = "mmap_matrix.bin";
#endif
    memmap::MemoryMappedMatrix<double> mmap_matrix(
        mmap_filename, n, p, memmap::AccessMode::ReadWrite);
    auto X_map = mmap_matrix.getMap();

    // Step 2. Populate matrix
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (Eigen::Index j = 0; j < p; ++j) {
            for (Eigen::Index i = 0; i < n; ++i) {
                X_map(i, j) = static_cast<double>(2.0 * static_cast<double>(i) +
                                                           static_cast<double>(j) / 100);
            }
        }
    }

    // Step 3. Fit and run LpNormScaler
    scaler::LpNormScaler l2scaler(scaler::LpNormScaler::NormType::L2, true);
    std::cout << "Fitting L2 LpNormScaler on memory-mapped matrix...\n";
    l2scaler.fit(X_map);
    std::cout << "✓ LpNormScaler fitted\n";
    l2scaler.transform_inplace(X_map);
    std::cout << "✓ Data transformed\n";
    l2scaler.inverse_transform_inplace(X_map);
    std::cout << "✓ Data inverse transformed\n";

    // Step 4. Clean up
    std::remove(mmap_filename.c_str());
    std::cout << "✓ Memory-mapped file removed\n";
    std::cout << "\n\n";
}



// ============================================================================
// Main
// ============================================================================

void init_omp_environment(int threads = 6) {
    omp_set_num_threads(threads);              // the demo's own loops
    omp_utils::set_num_threads(threads);       // the library's scalers and clustering
    omp_set_schedule(omp_sched_static, 0);
}


int main() {

    std::cout << "\n";
    cdiagnostics::print_section_header("DataTransformer Demo Suite");

    // Print OpenMP info
    omp_utils::print_info();
    init_omp_environment(6);
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    try {
        demo_scaler_comparison();
        demo_scaler_inverse_transform();
        demo_scaler_serialization();
        demo_scaler_on_mmap_matrix();

        std::cout << "\n";
        cdiagnostics::print_section_header("All Demos Completed Successfully");

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n\n";
        return 1;
    }

    return 0;
}
