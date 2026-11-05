# T-Rex Selector Methods: Virtual Dummies (VD)

This folder is reserved for Python demos of the virtual-dummy (VD) variant of the Terminating Random Experiments (T-Rex)
selector. It runs the VD forward selectors of [tsolvers/vd/](../../tsolvers/vd/), which realize a dummy only when it
enters the active set, and it can calibrate the number of dummies $L$ and the stopping point $T$ as well.

The library implements it in C++ (`cpp/src/trex_selector_methods/vd/trex_vd/` of the TRexSelector library, `TRexVD`),
and the Python bindings do not expose it yet, so no demo exists. The C++ counterpart of this folder is
[cpp/trex_selector_methods/vd/](../../../cpp/trex_selector_methods/vd/).

The demos of the augmented-dummy (AD) selectors are in [ad/](../ad/), and the [family README](../README.md) describes
the layout.
