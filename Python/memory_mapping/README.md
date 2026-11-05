# Memory-Mapped Matrix Utilities: Demonstration Suite

## Overview

This folder holds the Python examples for `MemoryMappedMatrix` from `trexselector.utils`. The class stores a
matrix in a file on disk and exposes it through a zero-copy NumPy view:

```math
\boldsymbol{X}_{\mathrm{mmap}} \in \mathbb{R}^{n \times p}.
```

The data is used like a NumPy array while its storage is file-backed. The Python binding maps `double` (`float64`)
matrices, stored column-major like an Eigen matrix, while the C++ class also takes `float` and complex scalar types.

In TRexSelector this matters for design matrices and dummy matrices that exceed the available random-access memory
(RAM) in solver and selector workflows, for intermediate data that should outlive a single object, and for handing
mapped data to the C++ solvers without a copy, because the `to_numpy()` view points straight into the mapped file.
Python replaces the C++ cleanup by Resource Acquisition Is Initialization (RAII) with `try/finally` blocks, while the
Open Multi-Processing (OpenMP) parallel generation of the C++ demo stays C++-only. The C++ and R packages offer the same
functionality (see [cpp/memory_mapping/](../../cpp/memory_mapping/README.md) and
[R/memory_mapping/](../../R/memory_mapping/README.md)).

The entry points, all from `trexselector.utils`, are:

- `MemoryMappedMatrix(filename, rows, cols, mode)` with `rows()`, `cols()`, `size()`, `to_numpy()`, `write_block()` and
  `mmap[row, col]` reads and writes (block writes through slices as well),
- `numpy_to_memmap(path, X)`, which writes a NumPy array into a file and returns a `MemoryMappedMatrix` over it,
- `AccessMode`, with the open modes `ReadOnly` and `ReadWrite`.

---

## Demos

| Folder | Purpose |
| ------ | ------- |
| [`demo_memory_mapping/`](demo_memory_mapping/) | Create and access a mapped matrix, stream columns out of core, fill columns from per-column seeds, element-wise access with bounds and read-only checks, block writes. The README walks through the console output. |

---

## Running

From the repository root, with the package installed in the repository's virtual environment:

```bash
.venv/bin/python Python/memory_mapping/demo_memory_mapping/demo_memory_mapping.py
```

The demo prints to the console only and finishes within seconds. Its backing files live in the system temporary
directory and are removed when each part ends. There is no build step.
