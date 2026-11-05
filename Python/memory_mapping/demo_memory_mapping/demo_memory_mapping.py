# ==============================================================================
# demo_memory_mapping.py
# ==============================================================================
#
# Demonstration of MemoryMappedMatrix usage from Python.
#
# Mirrors cpp/memory_mapping/demo_memory_mapping/demo_memory_mapping.cpp.
#
# Four demos which illustrate the capabilities of MemoryMappedMatrix,
# including zero-copy NumPy access and out-of-core workflows:
#
#   Demo 1 | Basics       : Create, write, inspect, extract a block, full
#                           roundtrip, and re-open in read-only mode.
#
#   Demo 2 | Streaming    : Out-of-core serial column generation. Working
#                           memory cost is O(n_rows) per column.
#
#   Demo 3 | Per-column seeds : Deterministic per-column seeded generation
#                           through the zero-copy view. (The cpp demo fills
#                           the columns in parallel with OpenMP — that part
#                           is C++-exclusive; the per-column seeding pattern
#                           that makes parallel filling deterministic is
#                           reproduced here serially.)
#
#   Demo 4 | Element-wise : Exercises the mmap[row, col] interface: scalar
#                           read, scalar write, compound assignment, bounds
#                           guard, ReadOnly mode guard, and the Python
#                           extensions (slice __setitem__, write_block()).
#
# ==============================================================================

import os
import tempfile

import numpy as np

from trexselector.utils import AccessMode, MemoryMappedMatrix, numpy_to_memmap


def print_section_header(title):
    print("=" * 78)
    print(f" {title}")
    print("=" * 78 + "\n")


def mmap_temp_path(tag):
    """Return a unique temporary file path for a demo (system temp directory)."""
    fd, path = tempfile.mkstemp(prefix=f"trex_mmap_demo_{tag}_", suffix=".bin")
    os.close(fd)  # release the OS descriptor; MemoryMappedMatrix re-opens the path
    return path


def remove_file(path):
    try:
        os.unlink(path)
        print("✓ Temporary file removed.\n")
    except OSError:
        pass


# ==============================================================================
# Demo 1: Basics — create, write, inspect, extract, roundtrip, read-only
# ==============================================================================

def demo_mmap_basics():
    print_section_header("Demo 1: Basics")

    # 1. Create a sample in-memory matrix
    n_rows, n_cols = 100, 50
    rng = np.random.default_rng(42)
    original = rng.standard_normal((n_rows, n_cols))
    print(f"1. Created a {n_rows} x {n_cols} in-memory matrix.\n")

    # 2. Temporary file path
    path = mmap_temp_path("basics")
    print(f"2. Temporary file: <tmp>/{os.path.basename(path)}\n")

    try:
        # 3. Write the matrix to a MemoryMappedMatrix
        #    numpy_to_memmap copies the array into the file-backed buffer
        #    (ColMajor, matching Eigen storage on the C++ side).
        mmap_mat = numpy_to_memmap(path, original)
        print("3. Matrix written to MemoryMappedMatrix.\n")

        # 4. Inspect metadata
        print("4. Metadata:")
        print(f"   rows() = {mmap_mat.rows()}")
        print(f"   cols() = {mmap_mat.cols()}")
        print(f"   size() = {mmap_mat.size()}"
              f"   (rows * cols = {mmap_mat.rows() * mmap_mat.cols()})\n")

        # 5. Extract a 3x3 block (zero-copy — to_numpy() returns a view)
        block = mmap_mat.to_numpy()[0:3, 0:3]
        max_diff = np.abs(block - original[0:3, 0:3]).max()
        print("5. 3x3 block (rows 0-2, cols 0-2):")
        print(np.round(block, 4))
        print(f"   Matches original block? {'yes' if max_diff == 0.0 else 'NO'}"
              f"  (max|diff| = {max_diff})\n")

        # 6. Full roundtrip — .copy() gives an independent in-RAM array
        restored = mmap_mat.to_numpy().copy()
        max_diff = np.abs(restored - original).max()
        print(f"6. Full roundtrip max|diff| = {max_diff}"
              "  (expected: 0.0 — bit-exact, no type conversion)\n")

        # 7. Re-open as read-only. Writes through the object raise at runtime
        #    (Python has no compile-time const guard — see Demo 4).
        mmap_ro = MemoryMappedMatrix(path, n_rows, n_cols, AccessMode.ReadOnly)
        print(f"7. Re-opened in ReadOnly mode: {mmap_ro.rows()} x {mmap_ro.cols()}"
              "  (writes raise RuntimeError)\n")
        del mmap_ro, mmap_mat  # unmap before the file is removed
    finally:
        remove_file(path)


# ==============================================================================
# Demo 2: Out-of-core streaming (serial)
# ==============================================================================

def demo_out_of_core_streaming():
    print_section_header("Demo 2: Out-of-Core Streaming (Serial)")

    n_rows, n_cols = 100, 50

    path = mmap_temp_path("streaming")
    print(f"Temporary file: <tmp>/{os.path.basename(path)}\n")

    try:
        # Stream columns one-by-one into the file. Each column uses an
        # independent RNG seeded with (1000 + column_index), making the
        # generated data deterministic and independently verifiable. Only
        # O(n_rows) doubles of working RAM are needed per column; the full
        # matrix is never allocated in memory.
        print(f"Writing {n_cols} columns serially "
              f"(O(n_rows) = {n_rows} doubles of working RAM per column)...")
        with open(path, "wb") as fh:
            for c in range(n_cols):
                col_rng = np.random.default_rng(1000 + c)
                col = col_rng.standard_normal(n_rows).astype(np.float64)
                fh.write(col.tobytes())  # contiguous float64 block == one column
        print("✓ All columns written.\n")

        # Verify column 0: re-generate with seed 1000 and compare.
        mmap_mat = MemoryMappedMatrix(path, n_rows, n_cols, AccessMode.ReadOnly)
        col0_ref = np.random.default_rng(1000).standard_normal(n_rows)
        col0_mmap = np.array([mmap_mat[r, 0] for r in range(n_rows)])
        max_diff = np.abs(col0_mmap - col0_ref).max()
        print(f"Spot-check column 0 (seed 1000): max|diff| = {max_diff}"
              "  (expected: 0.0)\n")
        del mmap_mat
    finally:
        remove_file(path)


# ==============================================================================
# Demo 3: Deterministic per-column seeded generation
# ==============================================================================

def demo_per_column_seeded_fill():
    print_section_header("Demo 3: Per-Column Seeded Fill "
                         "(OpenMP parallel fill is C++-exclusive)")

    n_rows, n_cols = 100, 50

    path = mmap_temp_path("seeded_fill")
    print(f"Temporary file: <tmp>/{os.path.basename(path)}\n")

    try:
        # Fill columns through the zero-copy view, one independent RNG per
        # column (base seed 3000 + column_index). In the cpp demo the same
        # loop runs under "#pragma omp parallel for": because every column
        # has its own seed and its own non-overlapping memory region, the
        # result is bitwise-identical regardless of thread scheduling order.
        mmap_mat = MemoryMappedMatrix(path, n_rows, n_cols, AccessMode.ReadWrite)
        view = mmap_mat.to_numpy()  # zero-copy, writes go straight to the file

        print(f"Filling {n_cols} columns with per-column seeds (3000 + c)...")
        for c in range(n_cols):
            col_rng = np.random.default_rng(3000 + c)
            view[:, c] = col_rng.standard_normal(n_rows)
        print("✓ Fill complete.\n")

        # Verify two spot-check columns: first and last.
        print("Spot-check:")
        for c in (0, n_cols - 1):
            col_ref = np.random.default_rng(3000 + c).standard_normal(n_rows)
            max_diff = np.abs(view[:, c] - col_ref).max()
            print(f"  Column {c:2d} (seed {3000 + c}): max|diff| = {max_diff}"
                  "  (expected: 0.0)")
        print()
        del view, mmap_mat
    finally:
        remove_file(path)


# ==============================================================================
# Demo 4: Element-wise access via mmap[row, col]
# ==============================================================================

def demo_element_wise_access():
    print_section_header("Demo 4: Element-wise access via mmap[row, col]")

    n_rows, n_cols = 10, 8

    path = mmap_temp_path("element_access")
    print(f"Temporary file: <tmp>/{os.path.basename(path)}\n")

    try:
        # 1. Create and populate with seeded data (same pattern as Demo 1)
        rng = np.random.default_rng(42)
        original = rng.standard_normal((n_rows, n_cols))
        mmap_mat = numpy_to_memmap(path, original)
        print(f"1. Created {n_rows} x {n_cols} matrix with seeded data (seed 42).\n")

        # 2. Scalar read — returns a plain float
        r, c = 2, 3
        val = mmap_mat[r, c]
        print(f"2. Scalar read  mmap[{r}, {c}] = {val:.6f}"
              f"   (matches original: {'yes' if val == original[r, c] else 'NO'})\n")

        # 3. Scalar write
        mmap_mat[r, c] = 99.0
        rb = mmap_mat[r, c]
        print(f"3. Scalar write mmap[{r}, {c}] = 99.0   -> read back = {rb}"
              f"  ({'correct' if rb == 99.0 else 'MISMATCH'})\n")

        # 4. Compound assignment (read-modify-write through __getitem__/__setitem__)
        before = mmap_mat[0, 0]
        mmap_mat[0, 0] += 10.0
        after = mmap_mat[0, 0]
        print(f"4. Compound +=  mmap[0, 0]: {before:.6f} + 10.0 -> {after:.6f}"
              f"  ({'correct' if after == before + 10.0 else 'MISMATCH'})\n")

        # 5. Bounds guard — IndexError for indices >= dimension
        print(f"5. Bounds check (row index {n_rows} on a {n_rows}-row matrix):")
        try:
            _ = mmap_mat[n_rows, 0]
        except IndexError as e:
            print(f"   Caught IndexError: {e}\n")

        # 6. Mode guard — RuntimeError when writing in ReadOnly mode
        mmap_ro = MemoryMappedMatrix(path, n_rows, n_cols, AccessMode.ReadOnly)
        print("6. Write attempt on ReadOnly matrix:")
        try:
            mmap_ro[0, 0] = 1.0
        except RuntimeError as e:
            print(f"   Caught RuntimeError: {e}\n")

        # --- Python extensions beyond the cpp scalar interface ---

        # 7. Block write via slice __setitem__
        new_block = np.arange(12.0).reshape(3, 4)
        original_block = mmap_mat.to_numpy()[0:3, 0:4].copy()
        mmap_mat[0:3, 0:4] = new_block
        block_match = np.allclose(mmap_mat.to_numpy()[0:3, 0:4], new_block)
        mmap_mat[0:3, 0:4] = original_block  # revert
        print("7. Block write mmap[0:3, 0:4] = arange(12).reshape(3, 4)"
              f"   -> round-trip match: {block_match}  (reverted)\n")

        # 8. Explicit write_block() method
        fill_block = np.full((3, 4), 7.0)
        mmap_mat.write_block(0, 3, 0, 4, fill_block)
        block_match = np.allclose(mmap_mat.to_numpy()[0:3, 0:4], 7.0)
        mmap_mat.write_block(0, 3, 0, 4, original_block)  # revert
        print(f"8. write_block(0, 3, 0, 4, fill=7.0)   -> round-trip match: "
              f"{block_match}  (reverted)\n")

        del mmap_ro, mmap_mat
    finally:
        remove_file(path)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print()
    print_section_header("Memory Mapping Demo Suite")

    demo_mmap_basics()
    demo_out_of_core_streaming()
    demo_per_column_seeded_fill()
    demo_element_wise_access()

    print_section_header("✓ All demos completed successfully")
