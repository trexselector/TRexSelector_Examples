# HAC Clustering: Demonstration Suite

## Overview

This folder holds the R examples for hierarchical agglomerative clustering (HAC) in the TRexSelector package. HAC
builds a tree of successive merges and recovers cluster assignments by cutting that tree at a chosen level. In
TRexSelector the Grouped Variable Selection (GVS) workflow uses it to form variable groups from a dendrogram cut.

The folder has two demos, each in its own subfolder with the script and a README that walks through the console output.
`demo_mlm_hac_01/` clusters samples and variables in memory, compares distance and linkage choices and shows the
locality-sensitive hashing (LSH) approximation on problems of at most 800 samples and 1,000 variables, while
`demo_mlm_hac_02_mmap/` repeats the LSH linkage comparison on a memory-mapped matrix of n = 2,000 x p = 4,000 that is
never copied into random-access memory (RAM). Both mirror their C++ counterparts at sizes scaled down by a factor of 50
to 100 in the variable count, the same sizes as in the Python demos, so each finishes within a few seconds. Both print
to the console only and run on the Open Multi-Processing (OpenMP) threads of the C++ backend, 6 or fewer if the
machine has fewer cores. The C++ and Python packages have the same demos, the C++ ones at full benchmark
size (see [cpp/ml_methods/hac_clustering/](../../../cpp/ml_methods/hac_clustering/README.md) and
[Python/ml_methods/hac_clustering/](../../../Python/ml_methods/hac_clustering/README.md)).
Cross-language correctness checks against R references live in the library's test suite (`cpp/tests/validation/`), not
in these examples.

The public entry points are `ml_agglomerative_cluster(data, method, metric, use_mmap)`, which clusters the **rows** of a
numeric matrix or an `mmap_matrix` and returns a linkage matrix, and
`ml_cut_tree(linkage, num_orig_objs, num_clusters)`, together with the lists `ml_linkage_method` (`Ward`, `Average`,
`Complete`, `Single`, `WPGMA`, `Median`, `Centroid`) and `ml_distance_metric` (`Euclidean`, `Correlation`, `Manhattan`,
`Correlation_LSH_Filter`, `Correlation_LSH_Approx`). Merge ids and cluster labels are 1-based, following the R
convention of the repository. The R wrapper rejects Ward, median and centroid linkage for the correlation and Manhattan
metrics, with the one exception of Ward under `Correlation_LSH_Approx`, which runs in a 64-dimensional projected
Euclidean space. Both scripts are self-contained, with their helpers (the adjusted Rand index, the data generators, the
standardization) inlined.

---

## Demos

| Folder | Purpose |
| ------ | ------- |
| [`demo_mlm_hac_01/`](demo_mlm_hac_01/) | In-memory HAC suite: sample clustering, correlation-distance variable clustering, LSH approximation, linkage-method comparisons, first-order autoregressive (AR(1)) Toeplitz blocks (six subproblems) |
| [`demo_mlm_hac_02_mmap/`](demo_mlm_hac_02_mmap/) | Memory-mapped HAC: generation, in-place standardization and the LSH linkage comparison on a 2,000 x 4,000 matrix |

---

## Running

From the repository root, with the TRexSelector R package 2.0.0 built from library commit b95b984:

```bash
Rscript R/ml_methods/hac_clustering/demo_mlm_hac_01/demo_mlm_hac_01.R
Rscript R/ml_methods/hac_clustering/demo_mlm_hac_02_mmap/demo_mlm_hac_02_mmap.R
```

There is no build step and no launch script. The scripts take no arguments.
