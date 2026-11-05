# Memory-Mapped Matrix Utilities: Demonstration Suite

## Overview

This folder holds the R examples for the memory-mapped matrix of the TRexSelector package. The `mmap_matrix`
handle stores a matrix in a file on disk and exposes it through S3 methods like an ordinary R matrix:

```math
\boldsymbol{X}_{\mathrm{mmap}} \in \mathbb{R}^{n \times p}.
```

The data is indexed like an R matrix while its storage is file-backed. The R binding maps `double` matrices, stored
column-major like an R or Eigen matrix, while the C++ class also takes `float` and complex scalar types.

In TRexSelector this matters for design matrices and dummy matrices that exceed the available random-access memory
(RAM) in solver and selector workflows, for intermediate data that should outlive a single object, and for handing
mapped data to the C++ solvers without a copy, because the handle passes a pointer to the mapped file straight to the
backend. R replaces the C++ cleanup by Resource Acquisition Is Initialization (RAII) with `on.exit()` handlers, while
the Open Multi-Processing (OpenMP) parallel generation of the C++ demo stays C++-only. The C++ and Python packages offer
the same functionality (see [cpp/memory_mapping/](../../cpp/memory_mapping/README.md) and
[Python/memory_mapping/](../../Python/memory_mapping/README.md)).

The entry points, all from TRexSelector, are:

- `mem_matrix(filename, rows, cols, mode)`, which opens or creates a mapped file in `"readwrite"` (default) or
  `"readonly"` mode and returns an `mmap_matrix` handle,
- `mem_matrix_from(mat, filename)`, which writes an R matrix into a file and returns a handle over it,
- the S3 methods `print()`, `dim()`, `[`, `[<-` and `as.matrix()` on the handle, with 1-based indices and contiguous
  index ranges only.

The same handles feed the out-of-core examples
[demo_mlm_hac_02_mmap](../ml_methods/hac_clustering/demo_mlm_hac_02_mmap/README.md),
[demo_mlm_scaler_01](../ml_methods/scaler_methods/demo_mlm_scaler_01/README.md) and
[demo_mlm_ms_01_ridge_cv_svd](../ml_methods/model_selection/demo_mlm_ms_01_ridge_cv_svd/README.md), the
[T-solver demos](../tsolvers/README.md) and the [T-Rex demos](../trex_selector_methods/ad/trex_core/README.md).

---

## Demos

| Folder | Purpose |
| ------ | ------- |
| [`demo_memory_mapping/`](demo_memory_mapping/) | Create and access a mapped matrix, stream columns out of core, fill columns from per-column seeds, element-wise access with bounds and read-only checks, block writes and a full copy into R memory. The README walks through the console output. |

---

## Running

From the repository root, with the TRexSelector R package 2.0.0 built from library commit b95b984:

```bash
Rscript R/memory_mapping/demo_memory_mapping/demo_memory_mapping.R
```

The demo prints to the console only and finishes within a few seconds. Its backing files live in the session temporary
directory of R and are removed when each part ends. There is no build step.
