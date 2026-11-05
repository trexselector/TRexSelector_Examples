# HAC Clustering: Demonstration Suite

## Overview

This folder holds the Python examples for hierarchical agglomerative clustering (HAC) in the
`trexselector.ml_methods.clustering` module. HAC builds a tree of successive merges and recovers cluster
assignments by cutting that tree at a chosen level. In TRexSelector the Grouped Variable Selection (GVS) workflow uses
it to form variable groups from a dendrogram cut.

The folder has two demos, each in its own subfolder with the script and a README that walks through the console output.
`demo_mlm_hac_01/` clusters samples and variables in memory, compares distance and linkage choices and shows the
locality-sensitive hashing (LSH) approximation on problems of at most 800 samples and 1,000 variables, while
`demo_mlm_hac_02_mmap/` repeats the LSH linkage comparison on a memory-mapped matrix of n = 2,000 x p = 4,000 that is
never copied into random-access memory (RAM). Both mirror their C++ counterparts at sizes scaled down by a factor of 50
to 100 in the variable count, so each finishes within seconds. Both print to the console only and run on the
Open Multi-Processing (OpenMP) threads of the C++ backend, 6 or fewer if the machine has fewer cores. The C++ and R
packages have the same demos, the C++ ones at full benchmark size (see
[cpp/ml_methods/hac_clustering/](../../../cpp/ml_methods/hac_clustering/README.md) and
[R/ml_methods/hac_clustering/](../../../R/ml_methods/hac_clustering/README.md)).
Cross-language correctness checks against R references live in the library's test suite (`cpp/tests/validation/`), not
in these examples.

The public entry points are `agglomerative_cluster(X, method, metric, use_mmap)`, which clusters the **rows** of `X` and
returns a SciPy-style linkage matrix, and `cut_tree(linkage, num_orig_objs, num_clusters)`, together with the
`LinkageMethod` and `DistanceMetric` enums.

---

## Demos

| Folder | Purpose |
| ------ | ------- |
| [`demo_mlm_hac_01/`](demo_mlm_hac_01/) | In-memory HAC suite: sample clustering, correlation-distance variable clustering, LSH approximation, linkage-method comparisons, first-order autoregressive (AR(1)) Toeplitz blocks (six subproblems) |
| [`demo_mlm_hac_02_mmap/`](demo_mlm_hac_02_mmap/) | Memory-mapped HAC: generation, in-place standardization and the LSH linkage comparison on a 2,000 x 4,000 matrix |

---

## Running

From the repository root, with the package installed in the repository's virtual environment:

```bash
.venv/bin/python Python/ml_methods/hac_clustering/demo_mlm_hac_01/demo_mlm_hac_01.py
.venv/bin/python Python/ml_methods/hac_clustering/demo_mlm_hac_02_mmap/demo_mlm_hac_02_mmap.py
```

There is no build step and no launch script. The scripts take no arguments and set the thread count themselves.
