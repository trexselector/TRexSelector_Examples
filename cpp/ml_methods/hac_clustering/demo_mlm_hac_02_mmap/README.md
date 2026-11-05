# Demo: Large-Scale HAC with Memory-Mapped I/O

## Purpose

This demo illustrates how hierarchical agglomerative clustering (HAC) can be applied using memory-mapped storage.

It extends the block-correlated variable-clustering setting from the in-memory HAC examples to a setting where the full
matrix is too large to allocate in random-access memory (RAM). The demo combines large-scale synthetic data
generation, in-place preprocessing, and clustering under a locality-sensitive hashing (LSH) approximation of the
correlation.

The workflow is organized into three phases:

1. **Generation**: write a large correlated matrix directly into a memory-mapped file.
2. **Standardization**: mean-center the columns and scale them to unit $\ell_2$-norm in place.
3. **Clustering**: benchmark selected HAC linkage strategies under an LSH-based correlation distance approximation.

Each phase section below ends with the console output of the recorded run (library 44cf1ab, 8 Open Multi-Processing
(OpenMP) threads, 12-core Apple Silicon, 64 GB RAM, 1 h 16 min in total).

---

## Statistical model and notation

Let

- $n$ the number of samples,
- $p$ the number of variables,
- $K$ the number of true variable blocks,
- $\boldsymbol{X} \in \mathbb{R}^{n \times p}$ denote the data matrix,
- $b(j) \in \lbrace1,\dots,K\rbrace$ the true block label of variable $j$.

In this demo, the dimensions are

```math
n = 100{,}000, \qquad p = 400{,}000, \qquad K = 10.
```

The quality of the recovered clustering is assessed with the Adjusted Rand Index (ARI), where $\mathrm{ARI} = 1$
indicates perfect recovery of the true block structure.

---

## Data model

The synthetic variables are generated from a blockwise latent-factor model. For each variable $j$ in block $b(j)$, the
entries of $\boldsymbol{X}$ are generated according to

```math
X_{ij} =
\mu_{b(j)}
+
\sigma_{b(j)}
\left[
\alpha L_{i,b(j)} + \gamma \varepsilon_{ij}
\right],
```

where

- $L_{i,b(j)} \sim \mathcal{N}(0,1)$ is the latent signal shared within block $b(j)$,
- $\varepsilon_{ij} \sim \mathcal{N}(0,1)$ is independent noise,
- $\mu_{b(j)}$ is a block-specific mean,
- $\sigma_{b(j)}$ is a block-specific scale.

In the implementation, the within-block signal and noise weights are fixed to

```math
\alpha = 0.85,
\qquad
\gamma = 0.15,
```

which produces strongly correlated variable blocks while preserving some within-block heterogeneity. The ten blocks are
imbalanced and sum to exactly 400,000 variables.

The true labels are determined deterministically from the block sizes, so the clustering target is known exactly before
the HAC stage begins.

---

## Phase 1: Memory-mapped generation

The full matrix is not assembled in RAM. Instead, the entries of $\boldsymbol{X}$ are written column by column into a
memory-mapped file, while only the latent signal matrix

```math
\boldsymbol{L} \in \mathbb{R}^{n \times K}
```

is held in memory.

This changes the storage pattern from a RAM-bound dense allocation to an out-of-core workflow. The file holds
$n \times p$ doubles, 320 GB (298 GiB as the demo prints it), whereas the latent $n \times K$ matrix is 7.6 MB.

## Phase 2: In-place standardization

Before correlation-based clustering, each variable is standardized in place by

```math
\widetilde{\boldsymbol{x}}_j =
\frac{\boldsymbol{x}_j - \bar{x}_j \mathbf{1}}
{\lVert \boldsymbol{x}_j - \bar{x}_j \mathbf{1} \rVert_2}.
```

This preprocessing step is required for correlation-based distance calculations and is performed directly on the
memory-mapped matrix rather than on a copied in-memory version.

### Phases 1 and 2: Console output

Both phases stream through the file once, column by column, with 8 threads. Each takes about six minutes, so the
320 GB are written at roughly 0.9 GB/s and rewritten at 0.8 GB/s. The time goes to the solid-state drive (SSD), not
to the arithmetic.

```text
============================================================
HAC Memory-Mapped Demo Suite (N=100k x P=400k)
============================================================
OpenMP version: 202011 (5.1+)
Max threads: 12
Num processors: 12
Custom reductions: Supported
Running with 8 threads


============================================================
Demo 03: LSH Linkage Comparison via Memory-Mapped I/O (N=100k, P=400k, K=10)
============================================================
Memory-mapped file : .../cpp/ml_methods/hac_clustering/demo_mlm_hac_02_mmap/trex_mlm_demo_02_hac_mmap.bin
File size          : 298.02 GB
Latent RAM (N x K)   : 7.63 MB

=== Phase 1: Generating correlated variables into mmap ===
    N=100000, P=400000, K=10
    Done. (335.5 s)

=== Phase 2: Standardising columns (mean=0, L2=1) in-place ===
    Done. (390.3 s)
```

---

## Phase 3: Clustering

After standardization, the demo clusters variables under the **LSH correlation approximation**,

```math
d_{\mathrm{LSH}}(j,\ell)
\approx
1 - \left| \mathrm{corr}(\boldsymbol{x}_j, \boldsymbol{x}_\ell) \right|.
```

The source describes this as a pure $O(1)$ correlation approximation by random-hyperplane similarity hashing
(SimHash), used to avoid repeatedly evaluating full high-dimensional exact correlations at this scale. Every variable
is reduced to a 64-bit signature from 64 random hyperplanes, so a distance is a Hamming distance between two words.

### Subproblem 1: Single linkage

The first clustering task applies **single linkage** under the LSH approximation. Algorithmically, this is dispatched to
the single-linkage HAC path and evaluated by cutting the dendrogram into $K$ groups and comparing the result against the
known block labels via ARI.

### Subproblem 2: Complete linkage and WPGMA feasibility

The second subproblem is a **feasibility question** rather than a full benchmark run. In the current implementation,
complete linkage and the weighted pair group method with arithmetic mean (WPGMA) are skipped because they require the
$O(p^2)$ Lance-Williams matrix machinery and are therefore treated as computationally intractable at $p = 400{,}000$ in
this setting.

### Subproblem 3: Average linkage feasibility

The third subproblem is skipped for the same reason as the second. The dispatcher routes **average linkage**, the
unweighted pair group method with arithmetic mean (UPGMA), to the exact Lance-Williams matrix engine for every metric,
because the geometric centroid update of the projected engine does not implement the UPGMA formula and would break the
reducibility the nearest-neighbor chain (NN-chain) relies on. At $p = 400{,}000$
that matrix is 1.28 TB, in RAM or on disk. An earlier version of this demo ran UPGMA here in the belief that it used the
projected engine. The RAM matrix engine zero-fills its buffer before it reports, and the process was killed by memory
without a message.

### Subproblem 4: Ward linkage

The fourth subproblem applies **Ward’s minimum-variance linkage** under the same large-scale setup. Ward with the LSH
approximation is the one route the dispatcher sends to the projected 64-dimensional engine, so the clustering runs on a
$64 \times p$ matrix of doubles, about 195 MB here, instead of the $p \times p$ distance matrix.

### Phase 3: Console output

Single linkage over all 400,000 variables takes 26 minutes. Sibson's single-linkage algorithm (SLINK) evaluates each
pair once, but with the $O(1)$ Hamming distance the 8 x 10^10 evaluations stay affordable, while the exact-correlation
equivalent would read the 320 GB file 80 billion times over. Ward projects the file once into the 64-dimensional
space (the 195 MB) and then runs its NN-chain in RAM for 38 minutes. Both recover the ten imbalanced blocks exactly.
The demo unmaps and deletes the file at the end.

```text
=== Phase 3: Reopening mmap for clustering ===
    Distance policy: Correlation LSH Approximation (O(1) SimHash)

[Test A] Single Linkage (SLINK + LSH):
    Time : 1548.6098 s
    ARI  : 1.0000 / 1.0000

[Test B] Complete Linkage & WPGMA:
    [SKIPPED] These methods lack a geometric center of mass.
    They strictly require the O(P^2) Lance-Williams matrix.
    At P=400,000, this requires 1.2 TB of NVMe I/O, which is intractable.

[Test C] UPGMA Average Linkage:
    [SKIPPED] Average linkage runs on the O(P^2) Lance-Williams matrix as well;
    the geometric centroid update of the projected engine does not implement
    the UPGMA formula. At P=400,000 the matrix is 1.28 TB, which is intractable.

[Test D] Ward's Minimum Variance (Projected 64D RAM Engine + LSH):
  -> [Backend] Projecting data to 64D (195 MB RAM)...
    Time : 2292.2481 s
    ARI  : 1.0000 / 1.0000

[Demo 03 Complete]

Cleaned up mmap file: .../cpp/ml_methods/hac_clustering/demo_mlm_hac_02_mmap/trex_mlm_demo_02_hac_mmap.bin

============================================================
Demo 03 completed successfully
============================================================
```

---

## Results at a glance

| Phase | Work | Time | Result |
| --- | --- | --- | --- |
| 1 Generation | 400,000 columns x 100,000 rows into the mapped file | 336 s | 320 GB file |
| 2 Standardization | mean 0, unit L2 norm, in place | 390 s | |
| 3 Test A | single linkage, SLINK + LSH | 1,549 s | ARI 1.000 |
| 3 Test B | complete linkage, WPGMA | skipped | $O(p^2)$ matrix, 1.2 TB |
| 3 Test C | average linkage (UPGMA) | skipped | $O(p^2)$ matrix, 1.28 TB |
| 3 Test D | Ward, projected 64-D engine + LSH | 2,292 s | ARI 1.000 |

You can compare the large-scale behavior with the in-memory HAC examples in
[demo_mlm_hac_01](../demo_mlm_hac_01/README.md), where Subproblem 5 runs the same LSH linkage comparison on
50,000 x 100,000 in RAM. The same memory-mapped clustering technique is demonstrated at a smaller, illustrative scale in
[Python](../../../../Python/ml_methods/hac_clustering/demo_mlm_hac_02_mmap/) and
[R](../../../../R/ml_methods/hac_clustering/demo_mlm_hac_02_mmap/).

---

## Running the demo

Through the folder's launch script, which rebuilds the target, keeps the console output under `logs/` and reports the
wall-clock time:

```bash
cd cpp/ml_methods/hac_clustering
THREADS=8 ./run_hac_demos.sh 02
```

Or directly, the OpenMP thread count as the optional first argument (default 6):

```bash
cpp/build/release/bin/ml_methods/hac_clustering/demo_mlm_hac_02_mmap/demo_mlm_hac_02_mmap 8
```

The demo prints to the console only. It needs 320 GB of free disk for the duration of the run and about 1 h 20 min on
8 threads.

---

## Technical notes

- The memory-mapped file path is injected at compile time via `TREX_DEMO_MMAP_PATH`, with a relative fallback if not
  provided. The pattern `trex_mlm_demo_*_mmap.bin` is gitignored.
- If a file with the expected size already exists, the generation and standardization phases are skipped and the demo
  reuses the existing mapped matrix for clustering. Because the file is pre-allocated to its full size before any data
  is written, a run that is interrupted mid-generation can leave behind a same-sized but incomplete file that this
  check cannot distinguish from a valid one.
- Cleanup is handled automatically on normal exit by unmapping first and then removing the backing file. A run that is
  killed leaves the file behind, so move it to the Trash or let the next run reuse it.
- Resident memory during the run reaches the size of the machine, because the operating system keeps the touched pages
  of the mapped file in the page cache. Those pages are file-backed and reclaimable, while the process's own allocations
  are the latent matrix (7.6 MB), the LSH signatures and planes, and Ward's 64 x p projection (195 MB).
