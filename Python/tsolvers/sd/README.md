# T-Solvers: Sparse Dummies (SD)

This folder is reserved for Python demos of the sparse-dummy (SD) terminating solvers (T-solvers). They run the race of
the Terminating Random Experiments (T-Rex) selector against sparse balanced Rademacher dummies instead of a dense dummy
matrix. Each dummy is stored only as the index sets of its $+1$ and $-1$ entries, so its inner product with the residual
reduces to sums of residual entries.

The library implements them in C++ (`cpp/src/tsolvers/sd/` of the TRexSelector library, `SD_TLARS_Solver`,
`SD_TOMP_Solver`, `SD_TAFS_Solver` and their `SD2_` pair variants), and the Python bindings do not expose them yet, so
no demo exists. The C++ counterpart of this folder is [cpp/tsolvers/sd/](../../../cpp/tsolvers/sd/).

The demos of the augmented-dummy (AD) solvers are in [ad/](../ad/), and the [family README](../README.md) describes the
layout.
