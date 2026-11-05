# Demo: Memory-Mapped Matrix Usage

## Purpose

This demo shows `MemoryMappedMatrix<Scalar>` in C++: a file-backed matrix that behaves like an Eigen object, so
data larger than the random-access memory (RAM) can be written, read and streamed column by column. It mirrors the R
and Python twins (`R/memory_mapping/demo_memory_mapping`, `Python/memory_mapping/demo_memory_mapping`) and adds what
only C++ offers: cleanup of the backing file by Resource Acquisition Is Initialization (RAII), direct `Eigen::Map`
interoperability, Open Multi-Processing (OpenMP) parallel generation and element-wise access checks.

---

## Running the demo

From the repository root, after `cmake --preset release && cmake --build --preset release-build` in `cpp/`:

```bash
cpp/build/release/bin/memory_mapping/demo_memory_mapping/demo_memory_mapping
```

The demo prints to the console only, so no `simulation_results/` folder is created. The backing files live in the
system temporary directory and are removed when each part ends. The output below is a run against library
b95b984 on a 12-core Apple Silicon machine. The temporary paths, the `data()` pointer and the thread counts differ
per machine, while everything else is deterministic. The temporary folder is shown as `<tmp>/`.

---

## Console walk-through

The suite reports the OpenMP setup first. The demos cap themselves at 6 threads regardless of the core count.

```text
============================================================
Memory Mapping Demo Suite
============================================================
OpenMP version: 202011 (5.1+)
Max threads: 12
Num processors: 12
Custom reductions: Supported
Running with 6 threads
```

### Demo 1: Basics

A 100 x 50 matrix is generated in memory, written into a memory-mapped file, and read back. The block extraction
and the full roundtrip are bit-exact because the mapped file stores the same `double`s, no conversion happens.
Reopening in `ReadOnly` mode yields a `ConstMapType`, so a write would not even compile. The `✓` line at the end is
the RAII guard deleting the backing file after the map is released.

```text
============================================================
Demo 1: Basics
============================================================
1. Created a 100 x 50 in-memory matrix.

2. Temporary file: "<tmp>/trex_mmap_demo_basics.bin"

3. Matrix written to MemoryMappedMatrix.

4. Metadata:
   rows   = 100
   cols   = 50
   size   = 5000
   data() = 0x104354000

5. 3x3 block (rows 0-2, cols 0-2):
 1.2938 -0.4731  0.2402
 0.7050 -0.1912  0.7400
 0.3980 -1.6661 -0.3649

   Matches original block? yes  (max|diff| = 0.0000)

6. Full roundtrip max|diff| = 0.0000  (expected: 0.0 — bit-exact, no type conversion)

7. Re-opened in ReadOnly mode: 100 x 50  (ConstMapType — writes blocked at compile time)

✓ Temporary file removed by RAII guard.
```

### Demo 2: Out-of-core streaming (serial)

The matrix is never allocated as a whole. Each column is drawn from its own seed (`1000` + column index) into a
buffer of `n_rows` doubles and written straight into the mapped file. The spot check regenerates column 0 from
seed `1000` and compares it with the file, which is how a reader can verify any column later without the full
matrix.

```text
============================================================
Demo 2: Out-of-Core Streaming (Serial)
============================================================
Temporary file: "<tmp>/trex_mmap_demo_streaming.bin"

Writing 50 columns serially (O(n_rows) = 100 doubles of working RAM per column)...
✓ All columns written.

Spot-check column 0 (seed 1000): max|diff| = 0.0000  (expected: 0.0)

C++ advantage: no full-matrix allocation; O(n_rows) working memory per column.
✓ Temporary file removed by RAII guard.
```

### Demo 3: Out-of-core parallel generation (C++ only)

Same idea with `#pragma omp parallel for`, so the columns are filled by 6 threads at once. Because every column
derives its random stream from its own seed (`3000` + column index), the result does not depend on which thread
handled which column or in what order. The two spot checks at both ends of the matrix confirm this. The fill
time rounds to 0 ms at this size. The point is the mechanism, not the speed.

```text
============================================================
Demo 3: Out-of-Core Parallel Generation (C++ Exclusive)
============================================================
Temporary file: "<tmp>/trex_mmap_demo_parallel.bin"
Threads available: 6

Filling 50 columns in parallel (#pragma omp parallel for, schedule(static))...
✓ Parallel fill complete in 0 ms.

Spot-check:
  Column  0 (seed 3000): max|diff| = 0.0000  (expected: 0.0)
  Column 49 (seed 3049): max|diff| = 0.0000  (expected: 0.0)

C++ advantage: per-column seeds guarantee bitwise-identical
results regardless of OpenMP thread scheduling order.
✓ Temporary file removed by RAII guard.
```

### Demo 4: Element-wise access

`operator()(row, col)` gives scalar reads and writes on the mapped file, including compound assignment. The last
two steps are the guard rails. An index past the last row throws `std::out_of_range` with the offending element
and the matrix shape, and a write on a matrix opened in `ReadOnly` mode throws `std::runtime_error`. Both are
caught and printed, and the demo continues.

```text
============================================================
Demo 4: Element-wise access via operator()(row, col)
============================================================
Temporary file: "<tmp>/trex_mmap_demo_element_access.bin"

1. Created 10 x 8 matrix with seeded data (seed 42).

2. Scalar read  mmap(2, 3) = -1.306686   (matches original: yes)

3. Scalar write mmap(2, 3) = 99.0   -> read back = 99.000000  (correct)

4. Compound +=  mmap(0, 0): 1.293820 + 10.0 -> 11.293820  (correct)

5. Bounds check (row index 10 on a 10-row matrix):
   Caught std::out_of_range: MemoryMappedMatrix: element (10, 0) out of bounds [10 x 8]

6. Write attempt on ReadOnly matrix:
   Caught std::runtime_error: Cannot write element: matrix is in ReadOnly mode

✓ Temporary file removed by RAII guard.


============================================================
✓ All demos completed successfully
============================================================
```

---

## Technical notes

- The backing files are created in the system temporary directory and deleted by an RAII guard, which unmaps
  before it removes the file.
- All random data is drawn per column from a deterministic seed, so serial and parallel generation produce the
  same bytes and any column can be regenerated for a check.
- The parallel part uses OpenMP with `schedule(static)`, and the thread count is capped at 6 by the demo.
