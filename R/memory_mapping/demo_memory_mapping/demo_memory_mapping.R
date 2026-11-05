# ==============================================================================
# demo_memory_mapping.R
# ==============================================================================
#
# Demonstration of MemoryMappedMatrix usage from R.
#
# Mirrors cpp/memory_mapping/demo_memory_mapping/demo_memory_mapping.cpp.
#
# Four demos which illustrate the capabilities of the memory-mapped matrix,
# including the S3 matrix interface and out-of-core workflows:
#
#   Demo 1 | Basics       : Create, write, inspect, extract a block, full
#                           roundtrip, and re-open in read-only mode.
#
#   Demo 2 | Streaming    : Out-of-core serial column generation. Working
#                           memory cost is O(n_rows) per column.
#
#   Demo 3 | Per-column seeds : Deterministic per-column seeded generation
#                           written straight into the mapped file. (The cpp
#                           demo fills the columns in parallel with OpenMP —
#                           that part is C++-exclusive; the per-column seeding
#                           pattern that makes parallel filling deterministic
#                           is reproduced here serially.)
#
#   Demo 4 | Element-wise : Exercises the mmap[row, col] interface: scalar
#                           read, scalar write, compound assignment, bounds
#                           guard, read-only mode guard, and the R extensions
#                           (block read / block write through `[` and `[<-`).
#
# Notes on the port:
#   - All indices are 1-based (the C++ and Python counterparts are 0-based).
#   - The mapped file is handed to the C++ backend as a pointer, so solvers and
#     selectors never materialize it in R's heap. The R-level accessors
#     (`[`, `as.matrix()`) do copy the requested block into R memory.
#
# ==============================================================================

library(TRexSelector)


print_section_header <- function(title) {
  cat(strrep("=", 78), "\n", sep = "")
  cat(" ", title, "\n", sep = "")
  cat(strrep("=", 78), "\n\n", sep = "")
}

# Unique temporary file path for a demo (system temp directory).
mmap_temp_path <- function(tag) {
  tempfile(pattern = paste0("trex_mmap_demo_", tag, "_"), fileext = ".bin")
}

remove_file <- function(path) {
  if (file.exists(path)) {
    unlink(path)
    cat("✓ Temporary file removed.\n\n")
  }
}


# ==============================================================================
# Demo 1: Basics — create, write, inspect, extract, roundtrip, read-only
# ==============================================================================

demo_mmap_basics <- function() {
  print_section_header("Demo 1: Basics")

  # 1. Create a sample in-memory matrix
  n_rows <- 100L
  n_cols <- 50L
  set.seed(42)
  original <- matrix(rnorm(n_rows * n_cols), nrow = n_rows, ncol = n_cols)
  cat(sprintf("1. Created a %d x %d in-memory matrix.\n\n", n_rows, n_cols))

  # 2. Temporary file path
  path <- mmap_temp_path("basics")
  cat(sprintf("2. Temporary file: <tmp>/%s\n\n", basename(path)))

  on.exit(remove_file(path), add = TRUE)

  # 3. Write the matrix to a MemoryMappedMatrix. mem_matrix_from()
  #    copies the R matrix into the file-backed buffer (column-major, matching
  #    the Eigen ColMajor storage used by the C++ backend) and returns a handle.
  mmap_mat <- mem_matrix_from(original, path)
  cat("3. Matrix written to MemoryMappedMatrix.\n\n")

  # 4. Inspect metadata through the S3 methods
  cat("4. Metadata:\n")
  # The S3 print method shows the file path; the temporary folder is shown as <tmp>/
  printed <- capture.output(print(mmap_mat))
  cat(sub(path, file.path("<tmp>", basename(path)), printed, fixed = TRUE), sep = "\n")
  cat(sprintf("   dim()  = %d x %d\n", dim(mmap_mat)[1], dim(mmap_mat)[2]))
  cat(sprintf("   size   = %d   (rows * cols)\n\n",
              dim(mmap_mat)[1] * dim(mmap_mat)[2]))

  # 5. Extract a 3x3 block through the `[` operator (1-based indices)
  block <- mmap_mat[1:3, 1:3]
  max_diff <- max(abs(block - original[1:3, 1:3]))
  cat("5. 3x3 block (rows 1-3, cols 1-3):\n")
  print(round(block, 4))
  cat(sprintf("   Matches original block? %s  (max|diff| = %g)\n\n",
              if (max_diff == 0) "yes" else "NO", max_diff))

  # 6. Full roundtrip — as.matrix() copies the mapped data back into R memory
  restored <- as.matrix(mmap_mat)
  max_diff <- max(abs(restored - original))
  cat(sprintf(paste0("6. Full roundtrip max|diff| = %g",
                     "  (expected: 0 — bit-exact, no type conversion)\n\n"),
              max_diff))

  # 7. Re-open as read-only. Writes through the handle raise at runtime
  #    (R has no compile-time const guard — see Demo 4).
  mmap_ro <- mem_matrix(path, rows = n_rows, cols = n_cols, mode = "readonly")
  cat(sprintf(paste0("7. Re-opened in ReadOnly mode: %d x %d",
                     "  (writes raise an error)\n\n"),
              dim(mmap_ro)[1], dim(mmap_ro)[2]))

  invisible(NULL)
}


# ==============================================================================
# Demo 2: Out-of-core streaming (serial)
# ==============================================================================

demo_out_of_core_streaming <- function() {
  print_section_header("Demo 2: Out-of-Core Streaming (Serial)")

  n_rows <- 100L
  n_cols <- 50L

  path <- mmap_temp_path("streaming")
  cat(sprintf("Temporary file: <tmp>/%s\n\n", basename(path)))

  on.exit(remove_file(path), add = TRUE)

  # Stream columns one-by-one into the file with writeBin(). Each column uses
  # an independent RNG seeded with (1000 + column index), making the generated
  # data deterministic and independently verifiable. Only O(n_rows) doubles of
  # working RAM are needed per column; the full matrix is never allocated.
  cat(sprintf(paste0("Writing %d columns serially ",
                     "(O(n_rows) = %d doubles of working RAM per column)...\n"),
              n_cols, n_rows))
  con <- file(path, "wb")
  for (c in seq_len(n_cols)) {
    set.seed(1000L + c - 1L)
    writeBin(as.double(rnorm(n_rows)), con)  # one contiguous column block
  }
  close(con)
  cat("✓ All columns written.\n\n")

  # Verify column 1: re-generate with seed 1000 and compare.
  mmap_mat <- mem_matrix(path, rows = n_rows, cols = n_cols, mode = "readonly")
  set.seed(1000L)
  col1_ref  <- rnorm(n_rows)
  col1_mmap <- mmap_mat[1:n_rows, 1]
  max_diff  <- max(abs(col1_mmap - col1_ref))
  cat(sprintf("Spot-check column 1 (seed 1000): max|diff| = %g  (expected: 0)\n\n",
              max_diff))

  cat("Out-of-core advantage: writeBin() streams one column at a time;\n")
  cat("the full matrix is never held in R memory.\n")

  invisible(NULL)
}


# ==============================================================================
# Demo 3: Deterministic per-column seeded generation
# ==============================================================================

demo_per_column_seeded_fill <- function() {
  print_section_header(paste("Demo 3: Per-Column Seeded Fill",
                             "(OpenMP parallel fill is C++-exclusive)"))

  n_rows <- 100L
  n_cols <- 50L

  path <- mmap_temp_path("seeded_fill")
  cat(sprintf("Temporary file: <tmp>/%s\n\n", basename(path)))

  on.exit(remove_file(path), add = TRUE)

  # Fill columns through the `[<-` block-assignment interface, one independent
  # RNG per column (base seed 3000 + column index). In the cpp demo the same
  # loop runs under "#pragma omp parallel for": because every column has its
  # own seed and its own non-overlapping memory region, the result is
  # bitwise-identical regardless of thread scheduling order.
  mmap_mat <- mem_matrix(path, rows = n_rows, cols = n_cols)

  cat(sprintf("Filling %d columns with per-column seeds (3000 + c)...\n", n_cols))
  for (c in seq_len(n_cols)) {
    set.seed(3000L + c - 1L)
    mmap_mat[1:n_rows, c] <- rnorm(n_rows)
  }
  cat("✓ Fill complete.\n\n")

  # Verify two spot-check columns: first and last.
  cat("Spot-check:\n")
  for (c in c(1L, n_cols)) {
    set.seed(3000L + c - 1L)
    col_ref  <- rnorm(n_rows)
    max_diff <- max(abs(mmap_mat[1:n_rows, c] - col_ref))
    cat(sprintf("  Column %2d (seed %d): max|diff| = %g  (expected: 0)\n",
                c, 3000L + c - 1L, max_diff))
  }
  cat("\n")

  invisible(NULL)
}


# ==============================================================================
# Demo 4: Element-wise access via mmap[row, col]
# ==============================================================================

demo_element_wise_access <- function() {
  print_section_header("Demo 4: Element-wise access via mmap[row, col]")

  n_rows <- 10L
  n_cols <- 8L

  path <- mmap_temp_path("element_access")
  cat(sprintf("Temporary file: <tmp>/%s\n\n", basename(path)))

  on.exit(remove_file(path), add = TRUE)

  # 1. Create and populate with seeded data (same pattern as Demo 1)
  set.seed(42)
  original <- matrix(rnorm(n_rows * n_cols), nrow = n_rows, ncol = n_cols)
  mmap_mat <- mem_matrix_from(original, path)
  cat(sprintf("1. Created %d x %d matrix with seeded data (seed 42).\n\n",
              n_rows, n_cols))

  # 2. Scalar read — returns a plain double, not a 1x1 matrix
  r  <- 3L
  cc <- 4L
  val <- mmap_mat[r, cc]
  cat(sprintf("2. Scalar read  mmap[%d, %d] = %.6f   (matches original: %s)\n",
              r, cc, val,
              if (isTRUE(all.equal(val, original[r, cc]))) "yes" else "NO"))
  cat(sprintf("   Is a plain scalar (not a 1x1 matrix): %s\n\n", !is.matrix(val)))

  # 3. Scalar write
  mmap_mat[r, cc] <- 99.0
  rb <- mmap_mat[r, cc]
  cat(sprintf("3. Scalar write mmap[%d, %d] <- 99.0   -> read back = %g  (%s)\n\n",
              r, cc, rb, if (rb == 99.0) "correct" else "MISMATCH"))

  # 4. Compound assignment (read-modify-write through `[` / `[<-`)
  before <- mmap_mat[1, 1]
  mmap_mat[1, 1] <- mmap_mat[1, 1] + 10.0
  after <- mmap_mat[1, 1]
  cat(sprintf("4. Compound +=  mmap[1, 1]: %.6f + 10.0 -> %.6f  (%s)\n\n",
              before, after,
              if (isTRUE(all.equal(after, before + 10.0))) "correct" else "MISMATCH"))

  # 5. Bounds guard — an error is raised for indices outside the dimensions
  cat(sprintf("5. Bounds check (row index %d on a %d-row matrix):\n",
              n_rows + 1L, n_rows))
  tryCatch(
    invisible(mmap_mat[n_rows + 1L, 1L]),
    error = function(e) cat(sprintf("   Caught error: %s\n\n", conditionMessage(e)))
  )

  # 6. Mode guard — writing through a read-only handle raises
  mmap_ro <- mem_matrix(path, rows = n_rows, cols = n_cols, mode = "readonly")
  cat("6. Write attempt on ReadOnly matrix:\n")
  tryCatch(
    mmap_ro[1, 1] <- 1.0,
    error = function(e) cat(sprintf("   Caught error: %s\n\n", conditionMessage(e)))
  )

  # --- R extensions beyond the cpp scalar interface ---

  # 7. Block write through `[<-` (contiguous index ranges only)
  new_block      <- matrix(as.double(1:12), nrow = 3, ncol = 4)
  original_block <- mmap_mat[1:3, 1:4]
  mmap_mat[1:3, 1:4] <- new_block
  block_match <- isTRUE(all.equal(mmap_mat[1:3, 1:4], new_block))
  mmap_mat[1:3, 1:4] <- original_block  # revert
  cat(sprintf(paste0("7. Block write mmap[1:3, 1:4] <- matrix(1:12, 3, 4)",
                     "   -> round-trip match: %s  (reverted)\n\n"),
              block_match))

  # 8. Whole-matrix read through as.matrix()
  full <- as.matrix(mmap_mat)
  cat(sprintf(paste0("8. as.matrix(mmap) -> %d x %d R matrix",
                     "   (block copy into R memory)\n\n"),
              nrow(full), ncol(full)))

  invisible(NULL)
}


# ==============================================================================
# Main
# ==============================================================================

cat("\n")
print_section_header("Memory Mapping Demo Suite")

demo_mmap_basics()
demo_out_of_core_streaming()
demo_per_column_seeded_fill()
demo_element_wise_access()

print_section_header("✓ All demos completed successfully")
