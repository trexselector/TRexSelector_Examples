# T-Rex Selector Methods: Sparse Dummies (SD)

This folder is reserved for R demos of the sparse-dummy (SD) variant of the Terminating Random Experiments (T-Rex)
selector. It calibrates the selector on the SD terminating solvers of [tsolvers/sd/](../../tsolvers/sd/), which race the
variables against sparse balanced Rademacher dummies instead of a dense dummy matrix.

The library implements it in C++ (`cpp/src/trex_selector_methods/sd/trex_sd/` of the TRexSelector library, `TRexSD`),
and the R bindings do not expose it yet, so no demo exists. The C++ counterpart of this folder is
[cpp/trex_selector_methods/sd/](../../../cpp/trex_selector_methods/sd/).

The demos of the augmented-dummy (AD) selectors are in [ad/](../ad/), and the [family README](../README.md) describes
the layout.
