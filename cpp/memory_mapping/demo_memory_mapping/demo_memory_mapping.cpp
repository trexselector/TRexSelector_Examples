// ============================================================================
// demo_memory_mapping.cpp
// ============================================================================
/**
 * @file demo_memory_mapping.cpp
 *
 * @brief Demonstration of MemoryMappedMatrix usage.
 *
 * @details Four demos which illustrate the capabilities of MemoryMappedMatrix,
 * including exploiting C++ RAII, zero-copy Eigen::Map access, and OpenMP parallelism.
 *
 *   Demo 1 | Basics       : Create, write, inspect, extract a block, full
 *                           roundtrip, and re-open in read-only mode.
 *
 *   Demo 2 | Streaming    : Out-of-core serial column generation. Working
 *                           memory cost is O(n_rows) per column.
 *
 *   Demo 3 | Parallel     : Out-of-core parallel column generation with OpenMP.
 *                           C++-exclusive: per-column isolated RNG seeds keep
 *                           the output deterministic regardless of thread
 *                           scheduling order. Includes wall-time measurement.
 *
 *   Demo 4 | Element-wise : Exercises the Eigen-like operator()(row, col)
 *                           interface: scalar read (const overload), scalar
 *                           write (mutable reference), compound assignment,
 *                           bounds-check exception, and ReadOnly mode guard.
 */
// ============================================================================

// std includes
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <utility>

// Eigen includes
#include <Eigen/Dense>

// utils includes
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>
#include <utils/memmap/memory_mapped_matrix.hpp>
#include <utils/openmp/utils_openmp.hpp>

// ============================================================================
// Namespace aliases
// ============================================================================

namespace memmap       = trex::utils::memmap;
namespace omp_utils    = trex::utils::openmp;
namespace cdiagnostics = trex::utils::eval::cdiagnostics;
namespace fs           = std::filesystem;

// Type aliases for the memory-mapped matrix and its Eigen views
using MmapMatrix    = memmap::MemoryMappedMatrix<double>;
using EigenMap      = MmapMatrix::MapType;        ///< Eigen::Map<MatrixXd> (mutable)
using ConstEigenMap = MmapMatrix::ConstMapType;   ///< Eigen::Map<const MatrixXd> (read-only)


// ============================================================================
// RAII file guard
// ============================================================================
/**
 * @brief Removes the backing file on destruction.
 *
 * Declare the MemoryMappedMatrix object in a nested scope that ends BEFORE
 * the MmapFileGuard goes out of scope. This guarantees that the OS unmap
 * (triggered by MmapMatrix destructor) happens before the file removal,
 * which is required by Boost.Interprocess on macOS.
 *
 * Example:
 * @code
 *   MmapFileGuard guard(mmap_temp_path("demo"));
 *   {
 *       MmapMatrix mmap(guard.path, rows, cols);
 *       // ... use mmap ...
 *   } // <- unmap here
 *   // guard destructor removes file here
 * @endcode
 */
struct MmapFileGuard {
    fs::path path;

    explicit MmapFileGuard(fs::path p) : path(std::move(p)) {}

    ~MmapFileGuard() {
        std::error_code ec;
        fs::remove(path, ec);   // noexcept: swallows errors silently on unwind
    }

    MmapFileGuard(const MmapFileGuard&)            = delete;
    MmapFileGuard& operator=(const MmapFileGuard&) = delete;
};


// ============================================================================
// Helper: runtime temp path
// ============================================================================
/**
 * @brief Returns a unique temporary file path for a demo.
 *
 * Uses the system temp directory at runtime rather than a compile-time
 * definition; appropriate for small demo files (unlike biobank-scale
 * scenarios that need an explicit filesystem path).
 */
static std::string mmap_temp_path(std::string_view tag) {
    return (fs::temp_directory_path() /
            ("trex_mmap_demo_" + std::string(tag) + ".bin")).string();
}


// ============================================================================
// Demo 1: Basics — create, write, inspect, extract, roundtrip, read-only
// ============================================================================
/**
 * @brief Demonstrates basic usage of MemoryMappedMatrix.
 *
 * Content:
 *   - Creating a MemoryMappedMatrix and writing an Eigen matrix into it.
 *   - Inspecting metadata (rows, cols, size, data pointer).
 *   - Extracting a sub-block via zero-copy Eigen::Map::block().
 *   - Full roundtrip verification (max absolute difference should be 0.0 —
 *     bit-exact, since no type conversion occurs).
 *   - Re-opening an existing file in AccessMode::ReadOnly.
 *     The returned ConstMapType prevents accidental writes at compile time.
 */
void demo_mmap_basics() {

    cdiagnostics::print_section_header("Demo 1: Basics");

    // -----------------------------------------------------------------------
    // 1. Create a sample in-memory matrix
    // -----------------------------------------------------------------------
    const std::size_t n_rows = 100;
    const std::size_t n_cols = 50;

    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    // create and populate
    Eigen::MatrixXd original(static_cast<Eigen::Index>(n_rows),
                              static_cast<Eigen::Index>(n_cols));
    for (Eigen::Index j = 0; j < original.cols(); ++j)
        for (Eigen::Index i = 0; i < original.rows(); ++i)
            original(i, j) = dist(rng);

    std::cout << "1. Created a " << n_rows << " x " << n_cols
              << " in-memory matrix.\n\n";

    // -----------------------------------------------------------------------
    // 2. Temporary file path
    // -----------------------------------------------------------------------
    MmapFileGuard guard(mmap_temp_path("basics"));
    std::cout << "2. Temporary file: " << guard.path << "\n\n";

    // -----------------------------------------------------------------------
    // 3. Write the matrix to a MemoryMappedMatrix
    //    Eigen assignment (ColMajor) copies data into the mmap backing store.
    //    The nested scope ensures unmap happens before MmapFileGuard fires.
    // -----------------------------------------------------------------------
    {
        MmapMatrix mmap(guard.path.string(), n_rows, n_cols,
                        memmap::AccessMode::ReadWrite);
        mmap.getMap() = original;   // zero-extra-copy Eigen ColMajor assignment
        std::cout << "3. Matrix written to MemoryMappedMatrix.\n\n";
    }

    // -----------------------------------------------------------------------
    // 4. Inspect metadata
    // -----------------------------------------------------------------------
    {
        MmapMatrix mmap(guard.path.string(), n_rows, n_cols,
                        memmap::AccessMode::ReadWrite);

        std::cout << "4. Metadata:\n"
                  << "   rows   = " << mmap.rows()  << "\n"
                  << "   cols   = " << mmap.cols()  << "\n"
                  << "   size   = " << mmap.size()  << "\n"
                  << "   data() = " << static_cast<const void*>(mmap.data()) << "\n\n";
    }

    // -----------------------------------------------------------------------
    // 5. Extract a 3x3 block (zero-copy — Eigen::Map::block() returns a view)
    // -----------------------------------------------------------------------
    {
        MmapMatrix mmap(guard.path.string(), n_rows, n_cols,
                        memmap::AccessMode::ReadWrite);

        auto block_mmap  = mmap.getMap().block(0, 0, 3, 3);
        auto block_orig  = original.block(0, 0, 3, 3);
        double max_diff  = (block_mmap - block_orig).cwiseAbs().maxCoeff();

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "5. 3x3 block (rows 0-2, cols 0-2):\n"
                  << block_mmap << "\n\n"
                  << "   Matches original block? "
                  << (max_diff == 0.0 ? "yes" : "NO") << "  (max|diff| = "
                  << max_diff << ")\n\n";
    }

    // -----------------------------------------------------------------------
    // 6. Full roundtrip
    // -----------------------------------------------------------------------
    {
        MmapMatrix mmap(guard.path.string(), n_rows, n_cols,
                        memmap::AccessMode::ReadWrite);

        Eigen::MatrixXd restored = mmap.getMap();  // copy into RAM
        double max_diff = (restored - original).cwiseAbs().maxCoeff();

        std::cout << "6. Full roundtrip max|diff| = " << max_diff
                  << "  (expected: 0.0 — bit-exact, no type conversion)\n\n";
    }

    // -----------------------------------------------------------------------
    // 7. Re-open as read-only
    //    getMap() returns ConstMapType — any assignment to it is a compile-time
    //    error, so accidental writes are caught before the binary is even built.
    // -----------------------------------------------------------------------
    {
        MmapMatrix mmap_ro(guard.path.string(), n_rows, n_cols,
                           memmap::AccessMode::ReadOnly);
        ConstEigenMap map_ro = std::as_const(mmap_ro).getMap();

        std::cout << "7. Re-opened in ReadOnly mode: "
                  << map_ro.rows() << " x " << map_ro.cols() << "  "
                  << "(ConstMapType — writes blocked at compile time)\n\n";
    }

    // MmapFileGuard destructor removes the file here.
    std::cout << "✓ Temporary file removed by RAII guard.\n\n";
}


// ============================================================================
// Demo 2: Out-of-core streaming (serial)
// ============================================================================
/**
 * @brief Serial column-by-column generation into a memory-mapped file.
 *
 * Content:
 * C++ advantage: direct pointer arithmetic over the mmap region requires only
 * O(n_rows) working RAM per column, independent of n_cols.
 * The OS page cache handles I/O transparently through the mmap pointer.
 */
void demo_out_of_core_streaming() {

    cdiagnostics::print_section_header("Demo 2: Out-of-Core Streaming (Serial)");

    const std::size_t n_rows = 100;
    const std::size_t n_cols = 50;

    MmapFileGuard guard(mmap_temp_path("streaming"));
    std::cout << "Temporary file: " << guard.path << "\n\n";

    // -----------------------------------------------------------------------
    // Create the backing file by constructing MmapMatrix (allocates n * p * 8 bytes)
    // then immediately destroying it to release the mapping before the write pass.
    // -----------------------------------------------------------------------
    {
        MmapMatrix mmap(guard.path.string(), n_rows, n_cols,
                        memmap::AccessMode::ReadWrite);
        // File allocated; no data written yet.
    }

    // -----------------------------------------------------------------------
    // Stream columns one-by-one into the mmap region via raw pointer arithmetic.
    // Each column uses an independent RNG seeded with (1000 + column_index),
    // making the generated data deterministic and independently verifiable.
    // -----------------------------------------------------------------------
    {
        MmapMatrix mmap(guard.path.string(), n_rows, n_cols,
                        memmap::AccessMode::ReadWrite);

        double* p = mmap.data();

        std::cout << "Writing " << n_cols << " columns serially "
                  << "(O(n_rows) = " << n_rows << " doubles of working RAM per column)...\n";

        for (std::size_t c = 0; c < n_cols; ++c) {
            std::mt19937_64          col_rng(1000 + c);
            std::normal_distribution<double> col_dist(0.0, 1.0);

            double* col_ptr = p + c * n_rows;
            for (std::size_t r = 0; r < n_rows; ++r)
                col_ptr[r] = col_dist(col_rng);
        }

        std::cout << "✓ All columns written.\n\n";
    }

    // -----------------------------------------------------------------------
    // Verify column 0: re-generate with seed 1000 and compare.
    // -----------------------------------------------------------------------
    {
        MmapMatrix mmap(guard.path.string(), n_rows, n_cols,
                        memmap::AccessMode::ReadOnly);

        // Reconstruct column 0 from seed 1000
        Eigen::VectorXd col0_ref(static_cast<Eigen::Index>(n_rows));
        {
            std::mt19937_64 ref_rng(1000);
            std::normal_distribution<double> ref_dist(0.0, 1.0);
            for (Eigen::Index r = 0; r < static_cast<Eigen::Index>(n_rows); ++r)
                col0_ref(r) = ref_dist(ref_rng);
        }

        Eigen::VectorXd col0_mmap = std::as_const(mmap).getMap().col(0);
        double max_diff = (col0_mmap - col0_ref).cwiseAbs().maxCoeff();

        std::cout << "Spot-check column 0 (seed 1000): max|diff| = " << max_diff
                  << "  (expected: 0.0)\n\n";
    }

    std::cout << "C++ advantage: no full-matrix allocation; "
              << "O(n_rows) working memory per column.\n";
    std::cout << "✓ Temporary file removed by RAII guard.\n\n";
}


// ============================================================================
// Demo 3: Out-of-core parallel generation (C++ exclusive)
// ============================================================================
/**
 * @brief Parallel column-by-column generation using OpenMP.
 *
 * Each thread writes to a non-overlapping column slice of the mmap region,
 * so there are no data races. Per-column RNG seeds (base 3000 + column_index)
 * make the output bitwise-identical regardless of thread scheduling order —
 * something impossible to guarantee with a global or per-thread shared PRNG.
 *
 * Wall time is measured with std::chrono::steady_clock to demonstrate the
 * benefit of parallelism for large out-of-core datasets.
 */
void demo_parallel_out_of_core() {

    cdiagnostics::print_section_header("Demo 3: Out-of-Core Parallel Generation (C++ Exclusive)");

    const std::size_t n_rows = 100;
    const std::size_t n_cols = 50;

    MmapFileGuard guard(mmap_temp_path("parallel"));
    std::cout << "Temporary file: " << guard.path << "\n"
              << "Threads available: " << omp_utils::get_max_threads() << "\n\n";

    // Create backing file
    {
        MmapMatrix mmap(guard.path.string(), n_rows, n_cols,
                        memmap::AccessMode::ReadWrite);
    }

    // -----------------------------------------------------------------------
    // Parallel fill — one column per loop iteration.
    //
    // Key correctness properties:
    //   1. std::mt19937_64 and std::normal_distribution are constructed INSIDE
    //      the loop body (stack-local) — not shared across threads.
    //   2. Each column writes to an independent, non-overlapping memory region.
    //   3. Per-column seed (3000 + c) is deterministic regardless of the order
    //      in which OpenMP schedules threads.
    // -----------------------------------------------------------------------
    {
        MmapMatrix mmap(guard.path.string(), n_rows, n_cols,
                        memmap::AccessMode::ReadWrite);

        double* p = mmap.data();

        std::cout << "Filling " << n_cols << " columns in parallel "
                  << "(#pragma omp parallel for, schedule(static))...\n";

        const auto t0 = std::chrono::steady_clock::now();

        #pragma omp parallel for schedule(static)
        for (int c = 0; c < static_cast<int>(n_cols); ++c) {

            // Both objects are stack-local — no shared mutable state
            std::mt19937_64 col_rng(3000 + static_cast<unsigned long>(c));
            std::normal_distribution<double> col_dist(0.0, 1.0);

            double* col_ptr = p + static_cast<std::ptrdiff_t>(c) *
                                  static_cast<std::ptrdiff_t>(n_rows);
            for (std::size_t r = 0; r < n_rows; ++r)
                col_ptr[r] = col_dist(col_rng);
        }

        const auto t1 = std::chrono::steady_clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        std::cout << "✓ Parallel fill complete in " << ms << " ms.\n\n";
    }

    // -----------------------------------------------------------------------
    // Verify two spot-check columns: first and last.
    // -----------------------------------------------------------------------
    {
        MmapMatrix mmap(guard.path.string(), n_rows, n_cols,
                        memmap::AccessMode::ReadOnly);

        ConstEigenMap map = std::as_const(mmap).getMap();

        auto verify_col = [&](std::size_t c) {
            Eigen::VectorXd col_ref(static_cast<Eigen::Index>(n_rows));
            {
                std::mt19937_64 ref_rng(3000 + c);
                std::normal_distribution<double> ref_dist(0.0, 1.0);
                for (Eigen::Index r = 0; r < static_cast<Eigen::Index>(n_rows); ++r)
                    col_ref(r) = ref_dist(ref_rng);
            }
            double max_diff = (map.col(static_cast<Eigen::Index>(c)) - col_ref)
                              .cwiseAbs().maxCoeff();
            std::cout << "  Column " << std::setw(2) << c
                      << " (seed " << (3000 + c) << "): max|diff| = " << max_diff
                      << "  (expected: 0.0)\n";
        };

        std::cout << "Spot-check:\n";
        verify_col(0);
        verify_col(n_cols - 1);
        std::cout << "\n";
    }

    std::cout << "C++ advantage: per-column seeds guarantee bitwise-identical\n"
              << "results regardless of OpenMP thread scheduling order.\n";
    std::cout << "✓ Temporary file removed by RAII guard.\n\n";
}


// ============================================================================
// Demo 4: Element-wise access via operator()(row, col)
// ============================================================================
/**
 * @brief Exercises the Eigen-like scalar access interface.
 *
 * Demonstrates:
 *   - Scalar read  : const operator()(row, col) returns a value copy.
 *   - Scalar write : operator()(row, col) returns a mutable reference,
 *                    enabling direct assignment and compound expressions.
 *   - Compound     : operator()() used with += without an intermediate copy.
 *   - Bounds guard : std::out_of_range thrown for out-of-bounds indices.
 *   - Mode guard   : std::runtime_error thrown when writing via ReadOnly map.
 */
void demo_element_wise_access() {

    cdiagnostics::print_section_header("Demo 4: Element-wise access via operator()(row, col)");

    const std::size_t n_rows = 10;
    const std::size_t n_cols =  8;

    MmapFileGuard guard(mmap_temp_path("element_access"));
    std::cout << "Temporary file: " << guard.path << "\n\n";

    // -----------------------------------------------------------------------
    // 1. Create and populate with seeded data (same pattern as Demo 1)
    // -----------------------------------------------------------------------
    Eigen::MatrixXd original(static_cast<Eigen::Index>(n_rows),
                              static_cast<Eigen::Index>(n_cols));
    {
        std::mt19937_64 rng(42);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (Eigen::Index j = 0; j < original.cols(); ++j)
            for (Eigen::Index i = 0; i < original.rows(); ++i)
                original(i, j) = dist(rng);
    }
    {
        MmapMatrix mmap(guard.path.string(), n_rows, n_cols,
                        memmap::AccessMode::ReadWrite);
        mmap.getMap() = original;
    }
    std::cout << "1. Created " << n_rows << " x " << n_cols
              << " matrix with seeded data (seed 42).\n\n";

    // -----------------------------------------------------------------------
    // 2. Scalar read — const operator()(row, col)
    // -----------------------------------------------------------------------
    {
        const std::size_t r = 2, c = 3;
        MmapMatrix mmap(guard.path.string(), n_rows, n_cols,
                        memmap::AccessMode::ReadWrite);

        const double val = std::as_const(mmap)(r, c);   // const overload
        const double ref = original(static_cast<Eigen::Index>(r),
                                    static_cast<Eigen::Index>(c));
        std::cout << "2. Scalar read  mmap(" << r << ", " << c << ") = "
                  << std::fixed << std::setprecision(6) << val
                  << "   (matches original: " << (val == ref ? "yes" : "NO") << ")\n\n";
    }

    // -----------------------------------------------------------------------
    // 3. Scalar write — mutable reference from operator()(row, col)
    // -----------------------------------------------------------------------
    {
        const std::size_t r = 2, c = 3;
        MmapMatrix mmap(guard.path.string(), n_rows, n_cols,
                        memmap::AccessMode::ReadWrite);

        mmap(r, c) = 99.0;                               // write via mutable ref
        const double rb = std::as_const(mmap)(r, c);    // read back via const ref

        std::cout << "3. Scalar write mmap(" << r << ", " << c << ") = 99.0"
                  << "   -> read back = " << rb
                  << "  (" << (rb == 99.0 ? "correct" : "MISMATCH") << ")\n\n";
    }

    // -----------------------------------------------------------------------
    // 4. Compound assignment — mutable reference enables in-place expressions
    // -----------------------------------------------------------------------
    {
        const std::size_t r = 0, c = 0;
        MmapMatrix mmap(guard.path.string(), n_rows, n_cols,
                        memmap::AccessMode::ReadWrite);

        const double before = std::as_const(mmap)(r, c);
        mmap(r, c) += 10.0;                              // compound assignment via ref
        const double after  = std::as_const(mmap)(r, c);

        std::cout << "4. Compound +=  mmap(" << r << ", " << c << "): "
                  << std::fixed << std::setprecision(6) << before
                  << " + 10.0 -> " << after
                  << "  (" << (after == before + 10.0 ? "correct" : "MISMATCH") << ")\n\n";
    }

    // -----------------------------------------------------------------------
    // 5. Bounds guard — out_of_range for indices >= dimension
    // -----------------------------------------------------------------------
    {
        MmapMatrix mmap(guard.path.string(), n_rows, n_cols,
                        memmap::AccessMode::ReadWrite);

        std::cout << "5. Bounds check (row index " << n_rows << " on a "
                  << n_rows << "-row matrix):\n";
        try {
            [[maybe_unused]] const double v = std::as_const(mmap)(n_rows, 0);
        } catch (const std::out_of_range& e) {
            std::cout << "   Caught std::out_of_range: " << e.what() << "\n\n";
        }
    }

    // -----------------------------------------------------------------------
    // 6. Mode guard — runtime_error when writing in ReadOnly mode
    // -----------------------------------------------------------------------
    {
        MmapMatrix mmap_ro(guard.path.string(), n_rows, n_cols,
                           memmap::AccessMode::ReadOnly);

        std::cout << "6. Write attempt on ReadOnly matrix:\n";
        try {
            mmap_ro(0, 0) = 1.0;                         // should throw
        } catch (const std::runtime_error& e) {
            std::cout << "   Caught std::runtime_error: " << e.what() << "\n\n";
        }
    }

    std::cout << "✓ Temporary file removed by RAII guard.\n\n";
}


// ============================================================================
// Main
// ============================================================================

int main() {

    std::cout << "\n";
    cdiagnostics::print_section_header("Memory Mapping Demo Suite");

    omp_utils::print_info();
    // Six threads for the demo's own loops and as the library's budget
    omp_set_num_threads(6);
    omp_utils::set_num_threads(omp_utils::get_max_threads());
    std::cout << "Running with " << omp_utils::get_num_threads() << " threads\n\n";

    try {

        demo_mmap_basics();
        demo_out_of_core_streaming();
        demo_parallel_out_of_core();
        demo_element_wise_access();

        cdiagnostics::print_section_header("✓ All demos completed successfully");

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n\n";
        return 1;
    }

    return 0;
}
