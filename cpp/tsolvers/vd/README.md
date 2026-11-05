# T-Solvers: Virtual Dummies (VD)

This folder is reserved for demos of the virtual-dummy (VD) forward selectors. Like the terminating solvers (T-solvers),
they race the variables against the dummies of the Terminating Random Experiments (T-Rex) selector, but they never hold
a dummy as a column. Each dummy is represented by its projection onto the growing basis of the selected columns and by
its remaining mass under the spherical law, and the solver realizes it only when it enters the active set.

The library implements them in `cpp/src/tsolvers/vd/` of the TRexSelector library (`VD_LARS`, `VD_OMP`, `VD_AFS` and
`VD_AFS_Logistic`). No demo exists yet.

The demos of the augmented-dummy (AD) solvers are in [ad/](../ad/), and the [family README](../README.md) describes the
layout.
