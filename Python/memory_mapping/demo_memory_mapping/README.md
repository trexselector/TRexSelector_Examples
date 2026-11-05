# Demo: Memory-Mapped Matrix Usage

## Purpose

This demo shows `MemoryMappedMatrix` in Python, a file-backed matrix that is read and written like a NumPy array, so
data larger than the random-access memory (RAM) can be written, read and streamed column by column. It mirrors the C++
and R twins (`cpp/memory_mapping/demo_memory_mapping`, `R/memory_mapping/demo_memory_mapping`) and replaces the
C++-specific features with their Python counterparts. Cleanup by Resource Acquisition Is Initialization (RAII) becomes
a `try/finally` block, `Eigen::Map` interoperability becomes the zero-copy `to_numpy()` view, and the
Open Multi-Processing (OpenMP) parallel fill becomes a serial fill with the same per-column seeds. On top of the C++
scalar interface, the Python demo also writes blocks through slices and `write_block()`.

---

## Running the demo

From the repository root, with the package installed in the repository's virtual environment:

```bash
.venv/bin/python Python/memory_mapping/demo_memory_mapping/demo_memory_mapping.py
```

The demo prints to the console only, so no `simulation_results/` folder is created. The backing files live in the
system temporary directory and are removed when each part ends. The output below is a run against library b95b984. It
used the package TRexSelector 1.0.0 and finished within seconds. The random suffixes of the temporary file names
differ per run, while everything else is deterministic. The demo itself prints the temporary folder as `<tmp>/`.

---

## Console walk-through

### Demo 1: Basics

A 100 x 50 matrix is drawn in memory from `np.random.default_rng(42)`, written into a memory-mapped file by
`numpy_to_memmap()`, and read back. The metadata shows `size()` as the product of the two dimensions. The 3 x 3 block
comes from the zero-copy `to_numpy()` view, and both the block and the full roundtrip, a `.copy()` of that view, are
bit-exact because the mapped file stores the same `float64` values without any conversion. The values differ from the
C++ run (1.2938 there, 0.3047 here) because NumPy's permuted congruential generator (PCG64) and `std::mt19937` draw
different numbers from the same seed. Reopening in `ReadOnly` mode works, but Python has no compile-time guard, so a
write through that object raises at runtime (Demo 4 shows it). The `✓` line at the end comes from the `finally` block
that removes the backing file after the map is released.

```text
==============================================================================
 Memory Mapping Demo Suite
==============================================================================

==============================================================================
 Demo 1: Basics
==============================================================================

1. Created a 100 x 50 in-memory matrix.

2. Temporary file: <tmp>/trex_mmap_demo_basics_jcj_5s3c.bin

3. Matrix written to MemoryMappedMatrix.

4. Metadata:
   rows() = 100
   cols() = 50
   size() = 5000   (rows * cols = 5000)

5. 3x3 block (rows 0-2, cols 0-2):
[[ 0.3047 -1.04    0.7505]
 [ 0.2891  0.6313 -1.4572]
 [-0.3782  1.2992 -0.3563]]
   Matches original block? yes  (max|diff| = 0.0)

6. Full roundtrip max|diff| = 0.0  (expected: 0.0 — bit-exact, no type conversion)

7. Re-opened in ReadOnly mode: 100 x 50  (writes raise RuntimeError)

✓ Temporary file removed.
```

### Demo 2: Out-of-core streaming (serial)

The matrix is never allocated as a whole. Each column is drawn from its own seed (`1000` + column index) into a buffer
of 100 doubles and appended to the file as one contiguous `float64` block, which is exactly one column of the
column-major layout that `MemoryMappedMatrix` maps. The writes go through a plain file handle, and the map is opened
afterwards in `ReadOnly` mode. Because `to_numpy()` needs `ReadWrite` mode, the spot check reads column 0 element by
element through `mmap[r, 0]`, regenerates it from seed `1000` and compares the two. This is how a reader can verify any
column later without the full matrix.

```text
==============================================================================
 Demo 2: Out-of-Core Streaming (Serial)
==============================================================================

Temporary file: <tmp>/trex_mmap_demo_streaming_2ft_o5z1.bin

Writing 50 columns serially (O(n_rows) = 100 doubles of working RAM per column)...
✓ All columns written.

Spot-check column 0 (seed 1000): max|diff| = 0.0  (expected: 0.0)

✓ Temporary file removed.
```

### Demo 3: Per-column seeded fill

The C++ demo fills the columns with `#pragma omp parallel for`, which is C++-only. The Python demo keeps the part that
makes the parallel fill deterministic and runs it serially. It opens the file in `ReadWrite` mode and writes every
column through the zero-copy view straight into the file, each from its own seed (`3000` + column index). Because no
column depends on another one, the result does not depend on the order in which the columns are filled. The two spot
checks at both ends of the matrix regenerate columns 0 and 49 from their seeds and match exactly.

```text
==============================================================================
 Demo 3: Per-Column Seeded Fill (OpenMP parallel fill is C++-exclusive)
==============================================================================

Temporary file: <tmp>/trex_mmap_demo_seeded_fill_d1ssb12b.bin

Filling 50 columns with per-column seeds (3000 + c)...
✓ Fill complete.

Spot-check:
  Column  0 (seed 3000): max|diff| = 0.0  (expected: 0.0)
  Column 49 (seed 3049): max|diff| = 0.0  (expected: 0.0)

✓ Temporary file removed.
```

### Demo 4: Element-wise access

`mmap[row, col]` gives scalar reads and writes on the mapped file, and compound assignment works as a read followed by
a write. The 10 x 8 matrix again comes from seed 42, so its entry (0, 0) is the first draw of that generator, the same
0.3047 that opens the block of Demo 1. Steps 5 and 6 are the guard rails. An index past the last row raises
`IndexError` with the offending element and the matrix shape, and a write on a matrix opened in `ReadOnly` mode raises
`RuntimeError`. These are the C++ `std::out_of_range` and `std::runtime_error` with their messages, translated by the
binding. Both are caught and printed, and the demo continues. Steps 7 and 8 go beyond the C++ demo. A slice assignment
and `write_block(row_start, row_count, col_start, col_count, values)` each write a 3 x 4 block, read it back through
the view, and restore the original values.

```text
==============================================================================
 Demo 4: Element-wise access via mmap[row, col]
==============================================================================

Temporary file: <tmp>/trex_mmap_demo_element_access_ecr08idy.bin

1. Created 10 x 8 matrix with seeded data (seed 42).

2. Scalar read  mmap[2, 3] = -0.049926   (matches original: yes)

3. Scalar write mmap[2, 3] = 99.0   -> read back = 99.0  (correct)

4. Compound +=  mmap[0, 0]: 0.304717 + 10.0 -> 10.304717  (correct)

5. Bounds check (row index 10 on a 10-row matrix):
   Caught IndexError: MemoryMappedMatrix: element (10, 0) out of bounds [10 x 8]

6. Write attempt on ReadOnly matrix:
   Caught RuntimeError: Cannot write element: matrix is in ReadOnly mode

7. Block write mmap[0:3, 0:4] = arange(12).reshape(3, 4)   -> round-trip match: True  (reverted)

8. write_block(0, 3, 0, 4, fill=7.0)   -> round-trip match: True  (reverted)

✓ Temporary file removed.

==============================================================================
 ✓ All demos completed successfully
==============================================================================
```

---

## Technical notes

- The backing files come from `tempfile.mkstemp()` in the system temporary directory, and the demo prints their names
  as `<tmp>/...` itself. Cleanup runs in `try/finally` blocks, and the mapped objects are deleted (unmapped) before
  `os.unlink()` removes the file.
- A `to_numpy()` view keeps its `MemoryMappedMatrix` alive, so the demo deletes the view together with the matrix
  before it removes the file.
- `to_numpy()` returns a zero-copy, column-major view and requires `ReadWrite` mode, because it hands out a writable
  map. A matrix opened in `ReadOnly` mode is read element-wise through `mmap[i, j]`.
- The files store the columns as contiguous `float64` blocks, the Eigen column-major layout of the C++ backend, so a
  file written with plain Python I/O as in Demo 2 can be mapped directly.
- All random data is drawn from a deterministic seed, per column in Demos 2 and 3, so any column can be regenerated for
  a check. The Python demo runs single-threaded, and the per-column seeds are what would let a parallel fill produce the
  same bytes.
