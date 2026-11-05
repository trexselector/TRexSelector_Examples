# Demo: Large-Scale HAC with Memory-Mapped I/O

## Purpose

This demo illustrates how hierarchical agglomerative clustering (HAC) can be applied using memory-mapped storage.

It extends the block-correlated variable-clustering setting from the in-memory HAC examples to an out-of-core workflow,
where the data matrix lives in a file rather than in random-access memory (RAM). The demo combines synthetic data
generation directly into the mapped file, in-place preprocessing, and clustering under a locality-sensitive hashing
(LSH) approximation of the correlation. It is the R twin of the
[C++ demo](../../../../cpp/ml_methods/hac_clustering/demo_mlm_hac_02_mmap/README.md), which runs the workflow at
biobank scale on a 320 GB file. The R demo keeps the three phases, the block layout and the test sequence at an
illustrative scale with a 64 MB file, the same scale as the Python demo.

The workflow is organized into three phases:

1. **Generation**: write a correlated matrix directly into a memory-mapped file.
2. **Standardization**: mean-center the variables and scale them to unit $\ell_2$-norm in place.
3. **Clustering**: benchmark selected HAC linkage strategies under an LSH-based correlation distance approximation.

Each phase section below ends with the console output of the recorded run. The output below is a run against library
b95b984. It used the R package TRexSelector 2.0.0 and 6 Open Multi-Processing (OpenMP) threads in the C++ backend,
and it finished within a few seconds.

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
n = 2{,}000, \qquad p = 4{,}000, \qquad K = 10.
```

The C++ demo runs $n = 100{,}000$ and $p = 400{,}000$ with the same $K$. The ten R block sizes are the C++ ones
divided by 100, while $n$ is divided by 50.

The quality of the recovered clustering is assessed with the adjusted Rand index (ARI), where $\mathrm{ARI} = 1$
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
imbalanced, with 330 to 500 variables, and sum to exactly 4,000 variables.

The true labels are determined deterministically from the block sizes, so the clustering target is known exactly before
the HAC stage begins.

---

## Phase 1: Memory-mapped generation

The full matrix is not assembled in RAM. Instead, the entries of $\boldsymbol{X}$ are written variable by variable into
a memory-mapped file, while only the latent signal matrix

```math
\boldsymbol{L} \in \mathbb{R}^{n \times K}
```

is held in memory, together with the one variable being drawn.

This changes the storage pattern from a RAM-bound dense allocation to an out-of-core workflow. Because
`ml_agglomerative_cluster()` clusters the **rows** of its input, the file stores the matrix **transposed** relative to
the C++ demo, as a $p \times n$ matrix with one variable per row. The demo opens it with
`mem_matrix(path, rows = p, cols = n)` in the default `"readwrite"` mode and writes each variable through the `[<-`
method of the handle. The same `mmap_matrix` handle later goes straight into the clustering calls. The file holds
$n \times p$ doubles, 64,000,000 bytes, which the demo prints as 61.04 MB (in units of $2^{20}$ bytes), whereas the
latent $n \times K$ matrix takes 0.15 MB.

## Phase 2: In-place standardization

Before correlation-based clustering, each variable is standardized in place by

```math
\widetilde{\boldsymbol{x}}_j =
\frac{\boldsymbol{x}_j - \bar{x}_j \mathbf{1}}
{\lVert \boldsymbol{x}_j - \bar{x}_j \mathbf{1} \rVert_2}.
```

This preprocessing step is required for correlation-based distance calculations and is performed directly on the
memory-mapped matrix, one variable (a row of the handle) at a time through `[` and `[<-`, rather than on a copied
in-memory version.

### Phases 1 and 2: Console output

Both phases loop once over the 4,000 variables in plain R, with block reads and writes of 2,000 values per variable,
and take 0.4 s each, against 0.1 s in the Python run. The C++ demo parallelizes the same loops with OpenMP and needs
about six minutes per phase for its 320 GB file. Unlike the C++ binary, the R demo prints only the thread count and
not the OpenMP version block, and its file lives in the session temporary directory of R, shown as `<tmp>/`.

```text
==============================================================================
 HAC Memory-Mapped Demo Suite (N=2k x P=4k)
==============================================================================

Running with 6 threads

==============================================================================
 Demo 03: LSH Linkage Comparison via Memory-Mapped I/O (N=2k, P=4k, K=10)
==============================================================================

Memory-mapped file : <tmp>/trex_demo03_hac_mmap_f30d217431f7.bin
File size          : 61.04 MB
Latent RAM (N x K) : 0.15 MB

=== Phase 1: Generating correlated variables into mmap ===
    N=2000, P=4000, K=10
    Done. (0.4 s)

=== Phase 2: Standardizing variables (mean=0, L2=1) in-place ===
    Done. (0.4 s)
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
(SimHash), used to avoid repeatedly evaluating full high-dimensional exact correlations at scale. Every variable is
reduced to a 64-bit signature from 64 random hyperplanes, so a distance is a Hamming distance between two words.
Passing the `mmap_matrix` handle routes each call through the memory-mapped path of the binding, which reads the
variables from the file without a copy and always sets `use_mmap`, so a distance matrix, where a linkage rule needs
one, would also go into a memory-mapped temporary file.

### Subproblem 1: Single linkage

The first clustering task applies **single linkage** under the LSH approximation. Algorithmically, this is dispatched to
the single-linkage HAC path and evaluated by cutting the dendrogram into $K$ groups and comparing the result against the
known block labels via ARI.

### Subproblem 2: Complete linkage and WPGMA feasibility

The second subproblem is a **feasibility question** rather than a benchmark run. Complete linkage and the weighted pair
group method with arithmetic mean (WPGMA) require the $O(p^2)$ Lance-Williams matrix machinery, which is intractable at
the C++ scale of $p = 400{,}000$. At $p = 4{,}000$ the matrix would be about 128 MB and tractable, but the R demo keeps
the skip and its message to mirror the C++ console output.

### Subproblem 3: Average linkage

The dispatcher routes **average linkage**, the unweighted pair group method with arithmetic mean (UPGMA), to the exact
Lance-Williams matrix engine for every metric, because the geometric centroid update of the projected engine does not
implement the UPGMA formula and would break the reducibility the nearest-neighbor chain (NN-chain) relies on. At
$p = 400{,}000$ that matrix is 1.28 TB, so the C++ demo skips the test. The R demo keeps the skip and the C++ message,
although at $p = 4{,}000$ the matrix would take about 128 MB.

### Subproblem 4: Ward linkage

The fourth subproblem applies **Ward’s minimum-variance linkage** under the same setup. The R wrapper normally rejects
Ward for non-Euclidean metrics, but Ward with the LSH approximation is the one route the dispatcher sends to the
projected 64-dimensional engine, so it is allowed. The clustering then runs on a $64 \times p$ matrix of doubles, about
2 MB here, instead of the $p \times p$ distance matrix.

### Phase 3: Console output

Single linkage over all 4,000 variables takes 0.18 s. Sibson's single-linkage algorithm (SLINK) evaluates each of the
about $8 \times 10^6$ pairs once with the $O(1)$ Hamming distance. Ward projects the variables once into the
64-dimensional space and runs its NN-chain in RAM in 0.21 s. Both recover the ten imbalanced blocks exactly. The
Python run measured 0.18 s for each of the two calls.

The backend line under Test D is Ward's projection. `Projecting data to 64D (1 MB RAM)` reports its 2,048,000 bytes,
1.95 MB in units of $2^{20}$ bytes, truncated to a whole number. The line appears in place, under its test, because
the R build of the library writes it through `Rcpp::Rcout`, the same console stream that `cat()` uses. The Python
log, in contrast, shows it after the closing banner. The cleanup line comes from the `on.exit()` handler of the demo
function, which deletes the data file when the function returns, so it precedes the closing banner.

```text
=== Phase 3: Clustering the memory-mapped variables ===
    Distance policy: Correlation LSH Approximation (O(1) SimHash)

[Test A] Single Linkage (SLINK + LSH):
    Time : 0.1817 s
    ARI  : 1.0000 / 1.0000

[Test B] Complete Linkage & WPGMA:
    [SKIPPED] These methods lack a geometric center of mass.
    They strictly require the O(P^2) Lance-Williams matrix.
    At the cpp scale (P=400,000) this requires 1.2 TB of NVMe I/O, which is intractable.

[Test C] UPGMA Average Linkage:
    [SKIPPED] Average linkage runs on the O(P^2) Lance-Williams matrix as well;
    the geometric centroid update of the projected engine does not implement
    the UPGMA formula. At the cpp scale (P=400,000) the matrix is 1.28 TB, which
    is intractable.

[Test D] Ward's Minimum Variance (Projected 64D RAM Engine + LSH):
  -> [Backend] Projecting data to 64D (1 MB RAM)...
    Time : 0.2086 s
    ARI  : 1.0000 / 1.0000

[Demo 03 Complete]

Cleaned up mmap file: <tmp>/trex_demo03_hac_mmap_f30d217431f7.bin
==============================================================================
 Demo 03 completed successfully
==============================================================================
```

---

## Results at a glance

| Phase | Work | Time | Result |
| --- | --- | --- | --- |
| 1 Generation | 4,000 variables x 2,000 samples into the mapped file | 0.4 s | 64 MB file |
| 2 Standardization | mean 0, unit $\ell_2$-norm, in place | 0.4 s | |
| 3 Test A | single linkage, SLINK + LSH | 0.182 s | ARI 1.000 |
| 3 Test B | complete linkage, WPGMA | skipped | kept to mirror the C++ skip |
| 3 Test C | average linkage (UPGMA) | skipped | kept to mirror the C++ skip |
| 3 Test D | Ward, projected 64-dimensional engine + LSH | 0.209 s | ARI 1.000 |

You can compare the out-of-core behavior with the in-memory HAC examples in
[demo_mlm_hac_01](../demo_mlm_hac_01/README.md), where Subproblem 5 runs the same LSH linkage comparison on
500 x 1,000 in RAM. The C++ original runs this demo on 100,000 x 400,000, where Tests A and D take 1,568 s and 2,306 s.
The Python twin is in
[Python/ml_methods/hac_clustering/](../../../../Python/ml_methods/hac_clustering/demo_mlm_hac_02_mmap/).

---

## Running the demo

From the repository root, with the TRexSelector R package 2.0.0 built from library commit b95b984:

```bash
Rscript R/ml_methods/hac_clustering/demo_mlm_hac_02_mmap/demo_mlm_hac_02_mmap.R
```

If the package is installed in a separate library, prefix the command with `R_LIBS=<R library>`. The demo prints to
the console only and finishes within a few seconds. It takes no arguments and needs about 64 MB of free space in the
temporary directory for the data file.

---

## Technical notes

- The data file comes from `tempfile()` in the session temporary directory of R, with a random hexadecimal suffix, and
  the demo prints it as `<tmp>/...` itself. Every run creates and fills a fresh file. The C++ demo instead injects the
  path at compile time via `TREX_DEMO_MMAP_PATH` and reuses an existing file of the expected size, skipping Phases 1
  and 2. When the package loads, it points the temporary directory of the C++ backend to the same session directory,
  so any memory-mapped buffer of the backend would land there too.
- The demo keeps its one `"readwrite"` mapping for all three phases instead of reopening the file for clustering as
  the C++ demo does. The clustering binding only reads, so a `"readonly"` handle would work as well.
- In the $p \times n$ column-major file one variable, a row of the handle, is spread over the whole file with a stride
  of $p$ doubles, so Phases 1 and 2 touch pages across the whole file for each variable. The clustering binding reads
  the variables the same way, through a lazy transpose of the mapped matrix. At 64 MB this costs 0.4 s per phase,
  while at biobank scale the C++ layout, with one variable per contiguous column, is the one that streams.
- `[` and `[<-` on an `mmap_matrix` accept contiguous 1-based index ranges only and copy the block between the file and
  R memory, which is why each variable passes through R as a vector of 2,000 doubles.
- Cleanup runs through `on.exit()`, which removes the file when the demo function returns, also after an error. The
  handle's mapping is released later, when R garbage-collects the handle or the process ends, which macOS and Linux
  allow because a removed file stays valid while it is mapped. A run that is killed hard skips the handler and leaves
  the file in the session temporary directory, so move it to the Trash.
- Neither clustering call allocates a distance matrix, although the memory-mapped path always sets `use_mmap`. SLINK
  needs none, and Ward's projection lives in RAM.
- `set.seed(42)` seeds R's default Mersenne-Twister generator with inversion normals, which draws the latent matrix
  first and then the noise variable by variable. The C++ demo uses `std::mt19937` seeded with 42 for the latent matrix
  and per-thread `std::mt19937` streams for the noise, so the two runs share the structure of the data but not its
  random numbers.
- The labels of `ml_cut_tree()` are 1-based, following the R convention of the repository, and the ARI does not depend
  on the label encoding.
- The script pins six threads right after the suite banner with `trex_set_num_threads(min(6L, trex_get_max_threads()))`,
  as the C++ and Python demos do, and prints the count. The threads govern the C++ backend, while generation and
  standardization are serial R loops.
