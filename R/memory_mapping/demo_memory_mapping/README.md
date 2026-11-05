# Demo: Memory-Mapped Matrix Usage

## Purpose

This demo shows the memory-mapped matrix of TRexSelector in R, a file-backed matrix that is read and written like
an R matrix, so data larger than the random-access memory (RAM) can be written, read and streamed column by column. It
mirrors the C++ and Python twins (`cpp/memory_mapping/demo_memory_mapping`, `Python/memory_mapping/demo_memory_mapping`)
and replaces the C++-specific features with their R counterparts. Cleanup by Resource Acquisition Is Initialization
(RAII) becomes an `on.exit()` handler, `Eigen::Map` interoperability becomes the S3 methods of the `mmap_matrix`
handle (`print()`, `dim()`, `[`, `[<-`, `as.matrix()`), and the Open Multi-Processing (OpenMP) parallel fill becomes a
serial fill with the same per-column seeds. On top of the C++ scalar interface, the R demo also writes a block through
`[<-` and copies the whole matrix into R memory with `as.matrix()`.

The entry points, all from TRexSelector, are:

- `mem_matrix(filename, rows, cols, mode)`, which opens or creates a mapped file in `"readwrite"` (default) or
  `"readonly"` mode and returns an `mmap_matrix` handle,
- `mem_matrix_from(mat, filename)`, which writes an R matrix into a file and returns a `"readwrite"` handle over it,
- the S3 methods `print()`, `dim()`, `[`, `[<-` and `as.matrix()` on the handle.

---

## Running the demo

From the repository root, with the TRexSelector R package 2.0.0 built from library commit b95b984:

```bash
Rscript R/memory_mapping/demo_memory_mapping/demo_memory_mapping.R
```

If the package is installed in a separate library, prefix the command with `R_LIBS=<R library>`. The demo prints to
the console only, so no `simulation_results/` folder is created. The backing files live in the session temporary
directory of R and are removed when each part ends. The output below is a run against library b95b984. It finished
within a few seconds. The random suffixes of the temporary file names differ per run, while everything else is
deterministic. The demo itself prints the temporary folder as `<tmp>/`, and for the path that the S3 `print()` method
shows, it captures the printed lines and replaces the folder before writing them.

---

## Console walk-through

### Demo 1: Basics

A 100 x 50 matrix is drawn in memory after `set.seed(42)`, written into a memory-mapped file by `mem_matrix_from()`,
and read back. The metadata come from the S3 methods. `print()` shows the dimensions, the file and the mode, `dim()`
the two dimensions, and the size is their product. The 3 x 3 block comes from the `[` method with the 1-based ranges
`1:3`, and both the block and the full roundtrip, an `as.matrix()` copy of the whole file, are bit-exact because the
mapped file stores the same `double` values without any conversion. The values differ from the C++ and Python runs
(1.2938 and 0.3047 there, 1.3710 here), because R's Mersenne-Twister with inversion normals, `std::mt19937` and
NumPy's generator draw different numbers from the same seed. Reopening in `"readonly"` mode works, but R has no
compile-time guard, so a write through that handle raises an error at runtime (Demo 4 shows it). The `✓` line at the
end comes from the `on.exit()` handler that removes the backing file.

```text
==============================================================================
 Memory Mapping Demo Suite
==============================================================================

==============================================================================
 Demo 1: Basics
==============================================================================

1. Created a 100 x 50 in-memory matrix.

2. Temporary file: <tmp>/trex_mmap_demo_basics_cd7e34096d11.bin

3. Matrix written to MemoryMappedMatrix.

4. Metadata:
MemoryMappedMatrix [100 x 50]
  File: <tmp>/trex_mmap_demo_basics_cd7e34096d11.bin
  Mode: readwrite
   dim()  = 100 x 50
   size   = 5000   (rows * cols)

5. 3x3 block (rows 1-3, cols 1-3):
        [,1]    [,2]    [,3]
[1,]  1.3710  1.2010 -2.0009
[2,] -0.5647  1.0448  0.3338
[3,]  0.3631 -1.0032  1.1713
   Matches original block? yes  (max|diff| = 0)

6. Full roundtrip max|diff| = 0  (expected: 0 — bit-exact, no type conversion)

7. Re-opened in ReadOnly mode: 100 x 50  (writes raise an error)

✓ Temporary file removed.
```

### Demo 2: Out-of-core streaming (serial)

The matrix is never allocated as a whole. Each column is drawn from its own seed (`1000` + column index - 1, so column
1 uses seed `1000` like column 0 in C++) into a vector of 100 doubles and appended to the file by `writeBin()` as one
contiguous `double` block, which is exactly one column of the column-major layout that the handle maps. The writes go
through a plain file connection, and the map is opened afterwards in `"readonly"` mode. Reading needs no write access
in R, so the spot check reads column 1 as one block through `mmap[1:100, 1]`, regenerates it from seed `1000` and
compares the two. This is how a reader can verify any column later without the full matrix. The two closing lines
state the point of the exercise, a remark the Python demo leaves out.

```text
==============================================================================
 Demo 2: Out-of-Core Streaming (Serial)
==============================================================================

Temporary file: <tmp>/trex_mmap_demo_streaming_cd7e56d791c7.bin

Writing 50 columns serially (O(n_rows) = 100 doubles of working RAM per column)...
✓ All columns written.

Spot-check column 1 (seed 1000): max|diff| = 0  (expected: 0)

Out-of-core advantage: writeBin() streams one column at a time;
the full matrix is never held in R memory.
✓ Temporary file removed.
```

### Demo 3: Per-column seeded fill

The C++ demo fills the columns with `#pragma omp parallel for`, which is C++-only. The R demo keeps the part that makes
the parallel fill deterministic and runs it serially. It opens a new file in `"readwrite"` mode and writes every column
through the `[<-` method straight into the file, each from its own seed (`3000` + column index - 1). Because no
column depends on another one, the result does not depend on the order in which the columns are filled. The two spot
checks at both ends of the matrix regenerate columns 1 and 50 from their seeds `3000` and `3049` and match exactly.

```text
==============================================================================
 Demo 3: Per-Column Seeded Fill (OpenMP parallel fill is C++-exclusive)
==============================================================================

Temporary file: <tmp>/trex_mmap_demo_seeded_fill_cd7e62a3cc5b.bin

Filling 50 columns with per-column seeds (3000 + c)...
✓ Fill complete.

Spot-check:
  Column  1 (seed 3000): max|diff| = 0  (expected: 0)
  Column 50 (seed 3049): max|diff| = 0  (expected: 0)

✓ Temporary file removed.
```

### Demo 4: Element-wise access

`mmap[row, col]` gives scalar reads and writes on the mapped file, and compound assignment works as a read followed by
a write. A scalar read returns a plain `double`, not a 1 x 1 matrix, which step 2 checks. The 10 x 8 matrix again comes
from seed 42, so its entry `[1, 1]` is the first draw of that generator, the same 1.3710 that opens the block of
Demo 1. Entry `[3, 4]` is the element (2, 3) that the C++ and Python demos read with 0-based indices. Steps 5 and 6 are
the guard rails. A row index past the last row raises an error, and so does a write on a handle opened in `"readonly"`
mode. Both checks live in the R methods `[` and `[<-`, which stop before the C++ layer is called, so the messages are
R's own rather than the translated C++ exceptions of the Python binding. Both are caught with `tryCatch()` and
printed, and the demo continues. Steps 7 and 8 go beyond the C++ demo. A block assignment through `[<-` writes a
3 x 4 block, reads it back and restores the original values, and `as.matrix()` copies the whole mapped matrix into an
R matrix.

```text
==============================================================================
 Demo 4: Element-wise access via mmap[row, col]
==============================================================================

Temporary file: <tmp>/trex_mmap_demo_element_access_cd7e6fbd9cf4.bin

1. Created 10 x 8 matrix with seeded data (seed 42).

2. Scalar read  mmap[3, 4] = 1.035104   (matches original: yes)
   Is a plain scalar (not a 1x1 matrix): TRUE

3. Scalar write mmap[3, 4] <- 99.0   -> read back = 99  (correct)

4. Compound +=  mmap[1, 1]: 1.370958 + 10.0 -> 11.370958  (correct)

5. Bounds check (row index 11 on a 10-row matrix):
   Caught error: Row indices out of bounds

6. Write attempt on ReadOnly matrix:
   Caught error: Cannot write to a read-only mmap_matrix.

7. Block write mmap[1:3, 1:4] <- matrix(1:12, 3, 4)   -> round-trip match: TRUE  (reverted)

8. as.matrix(mmap) -> 10 x 8 R matrix   (block copy into R memory)

✓ Temporary file removed.

==============================================================================
 ✓ All demos completed successfully
==============================================================================
```

---

## Technical notes

- The backing files come from `tempfile()` in the session temporary directory of R, and the demo prints their names as
  `<tmp>/...` itself. Each demo function registers `on.exit()` cleanup, which removes its file when the function
  returns, also after an error. The mappings are released later, when R garbage-collects the handles or the process
  ends, which macOS and Linux allow because a removed file stays valid while it is mapped.
- All indices are **1-based**, following the R convention of the repository, while the C++ and Python counterparts are
  0-based. The per-column seeds are therefore `1000 + (c - 1)` and `3000 + (c - 1)`, so column 1 in R uses the same
  seed as column 0 in C++.
- `[` and `[<-` accept **contiguous** index ranges only, because they map onto the block interface of the C++ class,
  and a non-contiguous slice raises an error. Unlike the Python `to_numpy()` view, they work on a `"readonly"` handle
  for reading.
- The files store the columns as contiguous `double` blocks, the Eigen column-major layout of the C++ backend and the
  native layout of an R matrix, so a file written with `writeBin()` as in Demo 2 can be mapped directly.
- The handle passes the mapped file to the C++ backend as a pointer, so solvers and selectors read it without a copy.
  The R-level accessors `[` and `as.matrix()`, in contrast, copy the requested block into R memory.
- All random data is drawn from a deterministic seed, per column in Demos 2 and 3, so any column can be regenerated for
  a check. The R demo runs single-threaded, and the per-column seeds are what would let a parallel fill produce the
  same bytes.
