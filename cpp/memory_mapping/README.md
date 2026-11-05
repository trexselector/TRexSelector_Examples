# Memory-Mapped Matrix Utilities: Demonstration Suite

## Overview

This folder holds the C++ examples for `MemoryMappedMatrix<Scalar>` from `utils/memmap/`. The class stores a matrix
in a file on disk and exposes it through `Eigen::Map`:

```math
\boldsymbol{X}_{\mathrm{mmap}} \in \mathbb{R}^{n \times p}.
```

The data is used like an Eigen matrix while its storage is file-backed. Any fixed-size scalar type works, real
(`float`, `double`) and complex (`std::complex<float>`, `std::complex<double>`). The demos use `double`.

In TRexSelector this matters for design matrices and dummy matrices that exceed the available random-access memory
(RAM) in solver and selector workflows, for intermediate data that should outlive a single object, and for handing
mapped data to the solvers through their Eigen interfaces without a copy. The C++ side adds cleanup of the backing file
by Resource Acquisition Is Initialization (RAII), zero-copy `Eigen::Map` access and Open Multi-Processing (OpenMP)
parallel generation. The Python and R packages offer the same functionality (see
`Python/memory_mapping/` and `R/memory_mapping/`).

---

## Demos

| Folder | Purpose |
| ------ | ------- |
| [`demo_memory_mapping/`](demo_memory_mapping/) | Create and access a mapped matrix, stream columns out of core serially and in parallel, element-wise access with bounds and read-only checks. The README walks through the console output. |

---

## Building and running

From `cpp/`, with the library's software development kit (SDK) installed at the path the presets point to:

```bash
cmake --preset release && cmake --build --preset release-build          # all demos
cmake --build --preset release-build --target demo_memory_mapping       # this demo only
```

Run it from the repository root:

```bash
cpp/build/release/bin/memory_mapping/demo_memory_mapping/demo_memory_mapping
```

The demo prints to the console only. Its backing files live in the system temporary directory and are removed
when each part ends.
