# HAC Clustering: Demonstration Suite

## Overview

This folder holds the C++ examples for hierarchical agglomerative clustering (HAC) in the `ml_methods` module. HAC
builds a tree of successive merges and recovers cluster assignments by cutting that tree at a chosen level. In
TRexSelector the Grouped Variable Selection (GVS) workflow uses it to form variable groups from a dendrogram cut.

The folder has two demos, each in its own subfolder with the source file and a README that walks through the
console output. `demo_mlm_hac_01/` clusters samples and variables in memory, compares distance and linkage choices
and shows the locality-sensitive hashing (LSH) approximation on problems up to n = 50,000 x p = 100,000, while
`demo_mlm_hac_02_mmap/` repeats the LSH linkage comparison at biobank scale (n = 100,000 x p = 400,000) on a
memory-mapped matrix that is never held in random-access memory (RAM). Both are benchmarks with runtimes of hours,
print to the console only and take an optional first argument, the Open Multi-Processing (OpenMP) thread count
(default 6). The Python and R packages have the same demos (see
[Python/ml_methods/hac_clustering/](../../../Python/ml_methods/hac_clustering/README.md) and
[R/ml_methods/hac_clustering/](../../../R/ml_methods/hac_clustering/README.md)). Cross-language correctness checks
against R references live in the library's test suite (`cpp/tests/validation/`), not in these examples.

---

## Demos

| Folder | Purpose |
| ------ | ------- |
| [`demo_mlm_hac_01/`](demo_mlm_hac_01/) | In-memory HAC suite: sample clustering, correlation-distance variable clustering, LSH approximation, linkage-method comparisons, first-order autoregressive (AR(1)) Toeplitz blocks (six subproblems) |
| [`demo_mlm_hac_02_mmap/`](demo_mlm_hac_02_mmap/) | Memory-mapped large-scale HAC: generation, in-place standardization and four linkage rules on a 100,000 x 400,000 matrix |

---

## Running

`run_hac_demos.sh` rebuilds the selected targets, runs them in order with per-demo logs under `logs/hac_<timestamp>/`
(gitignored) and prints a summary with wall-clock times:

```bash
cd cpp/ml_methods/hac_clustering
THREADS=8 ./run_hac_demos.sh          # both demos, 8 OpenMP threads (default 6)
THREADS=8 ./run_hac_demos.sh 01       # only the in-memory suite
tail -f logs/hac_*/demo_01.log        # follow a run from a second terminal
```

The binaries can also be started directly, the thread count as the first argument:

```bash
cpp/build/release/bin/ml_methods/hac_clustering/demo_mlm_hac_01/demo_mlm_hac_01 8
cpp/build/release/bin/ml_methods/hac_clustering/demo_mlm_hac_02_mmap/demo_mlm_hac_02_mmap 8
```
