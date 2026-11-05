# Demo: In-Memory HAC Clustering Suite

## Purpose

This demo illustrates several in-memory uses of hierarchical agglomerative clustering (HAC) that are relevant for the
TRexSelector project and, in particular, for grouped variable selection workflows. It is the R twin of
[cpp/ml_methods/hac_clustering/demo_mlm_hac_01/](../../../../cpp/ml_methods/hac_clustering/demo_mlm_hac_01/README.md)
and keeps its six scenarios, generators, block layouts and evaluation, while the sizes are scaled down so that the whole
suite finishes within a few seconds. The variable counts shrink by a factor of 50 to 100, and each console section
below names the C++ size next to the R one. The R sizes are the same as in the Python demo.

Six subproblems, SP1 to SP6 in the section headings below, run in the main block:

1. clustering samples with Euclidean distance,
2. clustering variables with correlation-based distance,
3. accelerating correlation-based clustering with locality-sensitive hashing (LSH),
4. comparing linkage strategies on structured synthetic data,
5. comparing linkage strategies under the pure $O(1)$ LSH correlation approximation,
6. clustering first-order autoregressive (AR(1)) Toeplitz block structures, where correlation decays with distance.

Throughout the demo, clustering quality is assessed by the **adjusted Rand index (ARI)**, which compares the recovered
cluster labels against the known ground-truth labels from the synthetic data generator. Each subproblem section below
ends with the console output of the recorded run. The output below is a run against library b95b984. It used the R
package TRexSelector 2.0.0 and 6 Open Multi-Processing (OpenMP) threads in the C++ backend.

---

## Statistical model and notation

Let

- $n$ be the number of samples,
- $p$ be the number of variables,
- $K$ be the number of true clusters or blocks,
- $\boldsymbol{X} \in \mathbb{R}^{n \times p}$ denote the data matrix,
- $z_i \in \lbrace1,\dots,K\rbrace$ the true cluster label of sample $i$,
- $b(j) \in \lbrace1,\dots,K\rbrace$ the true block label of variable $j$.

The recovered clustering is evaluated with the adjusted Rand index

```math
\mathrm{ARI} :=
\frac{\mathrm{Index} - \mathbb{E}\lbrack\mathrm{Index}\rbrack}
{\mathrm{MaxIndex} - \mathbb{E}\lbrack\mathrm{Index}\rbrack},
```

where $\mathrm{ARI} = 1$ indicates perfect agreement between true and predicted labels.

---

## Subproblem 1: Sample clustering

### SP1 Data model

In the sample-clustering scenario, each row belongs to one of $K$ clusters and is generated according to

```math
X_{ij} = \mu_{z_i} + \sigma_{z_i} \varepsilon_{ij},
\qquad
\varepsilon_{ij} \sim \mathcal{N}(0,1).
```

Here, $\mu_{z_i}$ is the cluster-specific mean and $\sigma_{z_i}$ is the cluster-specific scale.

### SP1 Mathematical approach

The goal is to cluster the rows $\boldsymbol x_1, \dots, \boldsymbol x_n$ using Euclidean distance,

```math
d(i,i') = \lVert \boldsymbol{x}_i - \boldsymbol{x}_{i'} \rVert_2 .
```

A **single-linkage** dendrogram is built with Sibson's single-linkage algorithm (SLINK). The dendrogram is then cut to
recover $K$ clusters, and the predicted labels are compared with the true sample labels using ARI.

### SP1 Console output

300 samples in 1,000 dimensions, three clusters of 100 around the centers 0, 3 and 20 with unit scale (C++ runs
1,500 x 50,000). `ml_agglomerative_cluster()` clusters the rows of its input and the samples are the rows, so `X` goes
in untransposed, while the variable-clustering subproblems below pass `t(X)`. The tree takes 0.011 s, and the cut
recovers the three clusters exactly. R numbers the clusters 1 to 3, where the C++ and Python demos print 0 to 2. The
block opens with the suite banner and the thread count.

```text
==============================================================================
 Dendrograms Demo Suite
==============================================================================

Running with 6 threads

==============================================================================
 Demo 1: SLINK Sample Clustering (Euclidean)
==============================================================================

Generating data (N=300, P=1000)...
Building Single Linkage tree (SLINK)...
✓ Tree built successfully in 0.011137 seconds.

Cutting tree to form 3 clusters...
Resulting Sample Clustering:
  Predicted Cluster 1: 100 samples
  Predicted Cluster 2: 100 samples
  Predicted Cluster 3: 100 samples

--- CLUSTERING PERFORMANCE ---
Adjusted Rand Index (ARI): 1.0000 / 1.0000

[Demo 1 Complete]
```

---

## Subproblem 2: Variable clustering with correlation distance

### SP2 Data model

In the variable-clustering scenario, variables are generated in correlated blocks.
Each variable $j$ belongs to a block $b(j)$, and the entries follow a latent-factor model of the form

```math
X_{ij} =
\mu_{b(j)} +
\sigma_{b(j)}
\left(
\alpha L_{i,b(j)} + \gamma \varepsilon_{ij}
\right)
```

where

- $L_{i,b(j)} \sim \mathcal{N}(0,1)$ is the latent block signal,
- $\varepsilon_{ij} \sim \mathcal{N}(0,1)$ is independent noise,
- $\alpha = 0.85$ controls the shared within-block signal,
- $\gamma = 0.15$ controls the noise contribution.

This creates blockwise collinearity among variables.

### SP2 Standardization

Before correlation-based clustering, each column is standardized by

```math
\widetilde{\boldsymbol{x}}_j =
\frac{\boldsymbol{x}_j - \bar{x}_j \mathbf{1}}
{\lVert \boldsymbol{x}_j - \bar{x}_j \mathbf{1} \rVert_2}
```

so that variables are mean-centered and scaled to unit $\ell_2$-norm.

### SP2 Mathematical approach

After standardization, variables are clustered using the correlation distance

```math
d(j,\ell) = 1 - \left| \mathrm{corr}(\boldsymbol{x}_j, \boldsymbol{x}_\ell) \right|.
```

Thus, variables with either strong positive or strong negative correlation are treated as close. The demo applies SLINK
to this distance structure, builds a single-linkage dendrogram on the variables, and cuts the tree into $K$ groups. The
resulting block assignments are compared against the known true block labels.

### SP2 Console output

500 variables in five imbalanced blocks (200 down to 50) over 500 samples, exact Pearson correlation (C++ runs 50,000
variables over 10,000 samples). SLINK evaluates every pair once with the exact distance, which is the cost that the LSH
variants of Subproblem 3 cut. Here it takes 0.017 s, and the cut returns the five block sizes exactly.

```text
==============================================================================
 Demo 2: SLINK Variable Clustering (Correlation)
==============================================================================

Generating Correlated Variables (N=500, P=500)...
Standardizing variables (Mean=0, L2=1) for correlation...
Building Single Linkage tree for Variables...
✓ Tree built successfully in 0.017015 seconds.

Cutting tree to form 5 clusters...
Resulting Variable Cluster Sizes:
  Predicted Cluster 1: 200 variables
  Predicted Cluster 2: 100 variables
  Predicted Cluster 3: 80 variables
  Predicted Cluster 4: 70 variables
  Predicted Cluster 5: 50 variables

--- CLUSTERING PERFORMANCE ---
Adjusted Rand Index (ARI): 1.0000 / 1.0000

[Demo 2 Complete]
```

---

## Subproblem 3: Variable clustering with LSH approximation

### SP3 Data model

The data model is the same block-correlated latent-factor model as in Subproblem 2, but the focus now shifts from exact
clustering to **approximate similarity search** for large variable sets.

### SP3 Mathematical approach

This subproblem compares three correlation-based strategies:

1. **Exact correlation**, using the full correlation geometry.
2. **LSH filter**, where hashing acts as a gatekeeper before more exact comparisons.
3. **LSH approximation**, where correlation is approximated through an angular sketch by random-hyperplane
   similarity hashing (SimHash).

Conceptually, the LSH approximation replaces expensive high-dimensional similarity comparisons by fast hash-based
surrogates:

```math
\mathrm{corr}(\boldsymbol{x}_j, \boldsymbol{x}_\ell)
\approx
\widehat{\mathrm{corr}}_{\mathrm{LSH}}(\boldsymbol{x}_j, \boldsymbol{x}_\ell).
```

The purpose is to study whether the block structure is still recovered accurately when correlation distances are
approximated rather than computed exactly.

### SP3 Console output

Same block model, now 800 samples, which makes every exact dot product 1.6 times as expensive as in Subproblem 2 (C++
runs 40,000 samples and 50,000 variables). The three strategies recover the blocks perfectly and differ only in cost.
The exact baseline needs 0.019 s, the LSH filter (Hamming gate, exact recheck of the survivors) 0.016 s and the pure
$O(1)$ SimHash estimate 0.014 s. The Python run of the same sizes measured 0.036 s, 0.028 s and 0.012 s. The gaps are
much smaller than in the C++ run (3,162 s, 1,211 s and 16 s), where 40,000 samples make each exact dot product 50
times as long as here.

```text
==============================================================================
 Demo 3: SLINK Variable Clustering (LSH Benchmark)
==============================================================================

Generating Imbalanced Correlated Variables (N=800, P=500)...
Standardizing variables (Mean=0, L2=1) for correlation...

------------------------------------------------------------
[Test A] Exact Pearson Correlation (The Baseline)
✓ Tree built successfully in 0.019083 seconds.
ARI: 1.0000 / 1.0000

------------------------------------------------------------
[Test B] LSH Filter (Gatekeeper to Exact Dot Product)
✓ Tree built successfully in 0.015957 seconds.
ARI: 1.0000 / 1.0000

------------------------------------------------------------
[Test C] LSH Approx (Pure O(1) SimHash Angular Estimation)
✓ Tree built successfully in 0.013772 seconds.

Resulting Variable Cluster Sizes (Approx):
  Predicted Cluster 1: 200 variables
  Predicted Cluster 2: 100 variables
  Predicted Cluster 3: 80 variables
  Predicted Cluster 4: 70 variables
  Predicted Cluster 5: 50 variables
ARI: 1.0000 / 1.0000

[Demo 3 Complete]
```

---

## Subproblem 4: Linkage-method comparison

### SP4 Data model

This comparison is again carried out on structured synthetic data with known cluster labels, so that different linkage
rules can be judged by their ARI and runtime.

### SP4 Mathematical approach

The main question is how clustering changes when the dendrogram update rule changes. The demo compares several linkage
principles, including:

- **Single linkage**, based on nearest-neighbor inter-cluster distance,
- **Complete linkage**, based on farthest-neighbor inter-cluster distance,
- **Average linkage**, the unweighted pair group method with arithmetic mean (UPGMA), based on average inter-cluster
  distance,
- **Weighted average linkage**, the weighted pair group method with arithmetic mean (WPGMA), based on weighted average
  inter-cluster distance,
- **Ward linkage**, based on minimum increase in within-cluster variance,
- and additional centroid/median-style variants, which don't have the reducibility property.

In abstract form, HAC iteratively merges two clusters $A$ and $B$ minimizing a linkage criterion

```math
d(A,B),
```

where the definition of $d(A,B)$ depends on the chosen linkage rule. This subproblem studies how that choice affects
both accuracy and computational behavior on the same synthetic structure. Single, complete, UPGMA and WPGMA linkage run
under the **exact** Pearson-correlation distance, while Ward, centroid and median linkage run under the squared
Euclidean distance that their update rules require. On the standardized variables the squared Euclidean distance is a
function of the correlation, so all rules see the same block geometry:

```math
\lVert \widetilde{\boldsymbol{x}}_j - \widetilde{\boldsymbol{x}}_\ell \rVert_2^2
= 2 \left( 1 - \mathrm{corr}(\boldsymbol{x}_j, \boldsymbol{x}_\ell) \right).
```

The C++ demo runs single linkage twice, directly through the SLINK class and through the dispatcher. The R binding has
the single entry point `ml_agglomerative_cluster()`, so single linkage appears once and the tests are lettered A to G
instead of A to H.

### SP4 Console output

Clean blocks, 300 samples, 500 variables (C++ runs 1,500 x 50,000). All seven rules recover the five blocks. SLINK
takes 0.015 s and the six matrix-based rules 0.0040 to 0.0050 s, so in this run SLINK is three to four times as slow
as the rules that fill a distance matrix first. Centroid and median, which take the generic linkage path, are the two
slowest of the six (0.0050 s and 0.0047 s), while the other four need 0.0040 to 0.0042 s. The order differs from the
C++ run, where SLINK and the matrix-based rules with the reducibility property all need about two minutes and centroid
and median take up to twice as long. Each matrix-based rule fills a 500 x 500 distance matrix in random-access memory
(RAM), and its 2 MB buffer shows up as a backend line under the test that allocates it. Ward switches to squared
Euclidean distance, as its update rule requires.

```text
==============================================================================
 Demo 4: Linkage Comparison
==============================================================================

1. Generating Clean Correlated Variables (N=300, P=500)...
2. Standardizing variables (Mean=0, L2=1) for correlation...
3. Setting distance policy (Correlation)...

[Test A] Single Linkage (SLINK):
Time: 0.014947 seconds
ARI:  1.0000 / 1.0000

[Test B] Complete Linkage (Disk/Matrix Bound):
  -> [Backend] RAM allocated 2 MB.

Time: 0.004092 seconds
ARI:  1.0000 / 1.0000

[Test C] UPGMA (RAM/Geometric Bound):
  -> [Backend] RAM allocated 2 MB.

Time: 0.004018 seconds
ARI:  1.0000 / 1.0000

[Test D] WPGMA (Disk/Matrix Bound):
  -> [Backend] RAM allocated 2 MB.

Time: 0.003989 seconds
ARI:  1.0000 / 1.0000

[Test E] WARD (RAM/Geometric Bound):
 -> Note: Switching to Squared Euclidean Distance!
  -> [Backend] RAM allocated 2 MB.

Time: 0.004167 seconds
ARI:  1.0000 / 1.0000

[Test F] Centroid Linkage (Generic Linkage):
  -> [Backend] RAM allocated 2 MB.

Time: 0.005002 seconds
ARI:  1.0000 / 1.0000

[Test G] Median Linkage (Generic Linkage):
  -> [Backend] RAM allocated 2 MB.

Time: 0.004746 seconds
ARI:  1.0000 / 1.0000

[Demo 4 Complete]
```

---

## Subproblem 5: LSH-approximate linkage comparison

### SP5 Data model

This subproblem uses the same block-correlated latent-factor model as Subproblems 2–4, but on a larger, imbalanced
configuration. Because the latent factors create dense, well-separated blocks in correlation space, the geometry is
favorable for approximate hashing.

### SP5 Mathematical approach

The comparison repeats the linkage study of Subproblem 4, but replaces the exact correlation distance with the pure
$O(1)$ SimHash-style LSH approximation

```math
d_{\mathrm{LSH}}(j,\ell)
\approx
1 - \left| \mathrm{corr}(\boldsymbol{x}_j, \boldsymbol{x}_\ell) \right|.
```

Single (SLINK), complete, average (UPGMA), and weighted average (WPGMA) linkage are each run under this approximation.
The purpose is to show that, when the block geometry is well separated, the approximate distance still recovers the
correct structure at a substantially reduced per-distance cost.

### SP5 Console output

The largest case of the R suite: 500 samples and 1,000 variables in five imbalanced blocks of 150 to 300 (C++ runs
50,000 x 100,000). With the $O(1)$ distance, single linkage over all pairs takes 0.028 s, while the three matrix-based
rules take 0.011 to 0.012 s, each on a 1,000 x 1,000 buffer of 8 MB (the three 8 MB backend lines). All four rules
recover the blocks. Memory plays no role at this size, in contrast to the C++ run, whose 76,440 MB buffers exceeded
the 64 GB of its machine.

```text
==============================================================================
 Demo 5: LSH Approx Linkage Comparison
==============================================================================

1. Generating Imbalanced Correlated Variables (N=500, P=1000)...
2. Standardizing variables (Mean=0, L2=1) for correlation...
3. Setting distance policy (Pure O(1) Correlation LSH Approx)...

[Test A] Single Linkage (SLINK with LSH):
Time: 0.027883 seconds
ARI:  1.0000 / 1.0000

[Test B] Complete Linkage (Disk/Matrix Bound):
  -> [Backend] RAM allocated 8 MB.

Time: 0.011872 seconds
ARI:  1.0000 / 1.0000

[Test C] UPGMA (RAM/Geometric Bound):
  -> [Backend] RAM allocated 8 MB.

Time: 0.011314 seconds
ARI:  1.0000 / 1.0000

[Test D] WPGMA (Disk/Matrix Bound):
  -> [Backend] RAM allocated 8 MB.

Time: 0.011370 seconds
ARI:  1.0000 / 1.0000

[Demo 5 Complete]
```

---

## Subproblem 6: AR(1) Toeplitz block clustering

### SP6 Data model

Here the variables are generated with a **block-diagonal AR(1) Toeplitz** correlation structure. Each block $k$ is built
by the stationary recursion
$\boldsymbol x_j = \rho_k\,\boldsymbol x_{j-1} + \sqrt{1 - \rho_k^2}\,\boldsymbol\varepsilon_j$ with
$\boldsymbol\varepsilon_j$ standard normal, which leaves every column at unit variance. The correlation therefore
decays *exactly* with the index separation inside a block, and vanishes between blocks:

```math
\mathrm{corr}(\boldsymbol{x}_{j}, \boldsymbol{x}_{j + m}) =
\begin{cases}
\rho_{k}^{m}, & b(j) = b(j+m) = k, \\
0, & \text{otherwise},
\end{cases}
```

so a variable is strongly correlated with its immediate neighbors but nearly uncorrelated with distant variables in the
same block. Each block carries its own decay rate, $\rho_{k} \in \lbrace 0.95,\, 0.90,\, 0.85,\, 0.80,\, 0.75 \rbrace$,
so even the slowest-decaying block, which has 200 variables, is down to $0.95^{100} \approx 0.006$ after a hundred
steps.

### SP6 Mathematical approach

This chaining geometry is used to contrast linkage rules under the **exact** Pearson-correlation distance. Single
linkage (SLINK) follows the correlation gradient and chains each block together, whereas complete and average linkage
tend to fragment the gradually decaying blocks. LSH is deliberately not used here, because the coarse angular
quantization destroys the continuous correlation gradient. In the block-factor models of Subproblems 3 and 5 the
variables of a block share one latent direction, so a 64-bit SimHash sketch separates the blocks as well as the exact
distance does. In this temporal AR(1) structure the correlation lives in the ordering of small angles between
neighbors, which the sketch quantizes away.

### SP6 Console output

This is the one subproblem where the linkage rule decides the outcome. 300 samples and 500 variables in blocks of 200
down to 50 (C++ runs 3,000 x 50,000 with blocks 100 times as long). Single linkage follows the chain of neighbors and
recovers all five blocks. Complete linkage judges two groups by their most distant members, which in a decaying chain
are nearly uncorrelated, and scores ARI 0.244. UPGMA averages over the whole chain and comes close to the structure at
0.965. The three runs take 0.015 s, 0.0036 s and 0.0035 s. The ranking of the three rules is the same in all three
languages, while the scores depend strongly on the draws. The Python run of the same sizes scored 0.151 and 0.626, and
the C++ run, with its longer blocks and larger sample, 0.005 and 0.109.

```text
==============================================================================
 Demo 6: AR(1) Block Linkage Comparison
==============================================================================

1. Generating AR(1) Toeplitz Variables (N=300, P=500)...
2. Standardizing variables (Mean=0, L2=1) for correlation...
3. Setting distance policy (Exact Pearson Correlation)...

[Test A] Single Linkage (SLINK):
Time: 0.014936 seconds
ARI:  1.0000 / 1.0000

[Test B] Complete Linkage (Disk/Matrix Bound):
  -> [Backend] RAM allocated 2 MB.

Time: 0.003571 seconds
ARI:  0.2436 / 1.0000

[Test C] UPGMA (RAM/Geometric Bound):
  -> [Backend] RAM allocated 2 MB.

Time: 0.003467 seconds
ARI:  0.9650 / 1.0000

[Demo 6 Complete]
```

---

## End of the run

The closing banner ends the log. The eleven backend lines from the C++ library, one per matrix-based run, already
appeared in place under the tests that triggered them. These are the six 2 MB buffers of Subproblem 4 ($p = 500$), the
three 8 MB buffers of Subproblem 5 ($p = 1{,}000$) and the two 2 MB buffers of Subproblem 6. The R build of the
library writes these lines through `Rcpp::Rcout`, the same console stream that `cat()` uses, so they stay in order even
though the log was written to a file. The Python log, in contrast, collects them after its closing banner.

```text
==============================================================================
 All clustering demos completed successfully
==============================================================================
```

---

## Results at a glance

| Subproblem | Size (n x p) | Distance | Rule | Time | ARI |
| --- | --- | --- | --- | --- | --- |
| 1 Sample clustering | 300 x 1,000 | Euclidean | SLINK | 0.011 s | 1.000 |
| 2 Variable clustering | 500 x 500 | exact correlation | SLINK | 0.017 s | 1.000 |
| 3 LSH benchmark | 800 x 500 | exact / LSH filter / LSH approx | SLINK | 0.019 s / 0.016 s / 0.014 s | 1.000 each |
| 4 Linkage comparison | 300 x 500 | exact correlation, squared Euclidean for Ward, centroid, median | 7 rules | 0.004 to 0.015 s | 1.000 each |
| 5 LSH linkage comparison | 500 x 1,000 | LSH approx | SLINK, complete, UPGMA, WPGMA | 0.028 / 0.012 / 0.011 / 0.011 s | 1.000 each |
| 6 AR(1) Toeplitz blocks | 300 x 500 | exact correlation | SLINK / complete / UPGMA | 0.015 / 0.0036 / 0.0035 s | 1.000 / 0.244 / 0.965 |

The whole suite finishes within a few seconds on 6 threads. The same scenarios run at full size in
[C++](../../../../cpp/ml_methods/hac_clustering/demo_mlm_hac_01/), where the suite takes 2 h 20 min on 8 threads, and
at the R sizes in [Python](../../../../Python/ml_methods/hac_clustering/demo_mlm_hac_01/).

---

## Running the demo

From the repository root, with the TRexSelector R package 2.0.0 built from library commit b95b984:

```bash
Rscript R/ml_methods/hac_clustering/demo_mlm_hac_01/demo_mlm_hac_01.R
```

If the package is installed in a separate library, prefix the command with `R_LIBS=<R library>`. The demo prints to
the console only and finishes within a few seconds. It takes no arguments, and there is no build step and no launch
script, which only the C++ folder has.

---

## Technical notes

- `ml_agglomerative_cluster()` clusters the **rows** of its input and returns a linkage matrix, and `ml_cut_tree()`
  recovers flat labels. Sample clustering passes `X`, variable clustering passes `t(X)`. The binding copies its input
  once into a transposed Eigen matrix, because the C++ engine clusters columns and reads each object as a contiguous
  column. Variable clustering thus transposes twice, once by `t()` before the timed call and once inside it.
- The merge ids of the linkage matrix and the labels of `ml_cut_tree()` are **1-based**, following the R convention of
  the repository. The two R wrappers shift them by one around the C++ calls, while the C++ and Python ports are
  0-based. The ARI does not depend on the label encoding.
- The R wrapper rejects Ward, median and centroid linkage with an error for the correlation and Manhattan metrics,
  because their update rules are defined in Euclidean geometry. The one exception is Ward with
  `Correlation_LSH_Approx`, which the dispatcher runs in a 64-dimensional SimHash-projected Euclidean space.
  Subproblem 4 therefore passes `ml_distance_metric$Euclidean` for these three rules.
- All six subproblems are enabled in the main block, and each is guarded by its own `if (TRUE)` toggle so an individual
  part can be switched off while iterating on another, as in the C++ `main()`. The helpers (`compute_ari()`, the three
  generators, `standardize_for_correlation()`) are inlined in the script.
- The matrix-based linkage rules (complete, UPGMA, WPGMA, Ward, centroid, median) fill the full $p \times p$ distance
  matrix as a `double` buffer, 2 MB at $p = 500$ and 8 MB at $p = 1{,}000$. The `use_mmap` argument of
  `ml_agglomerative_cluster()` would place that buffer in a memory-mapped file instead, but the demo leaves it at its
  default `FALSE` in every call, including those whose labels read "Disk/Matrix Bound". Every buffer therefore lives in
  RAM, as the backend lines say.
- Each generator calls `set.seed(42)` and draws with R's default Mersenne-Twister generator and inversion normals, in
  vectorized form (the latent-factor model draws the latent matrix first and then the whole noise matrix). The C++
  demo fills its matrices from per-thread `std::mt19937` streams and the Python demo from NumPy's default generator,
  so the three languages draw different random numbers and agree qualitatively, not bitwise.
- The script pins six threads right after the suite banner with `trex_set_num_threads(min(6L, trex_get_max_threads()))`,
  as the C++ and Python demos do, and prints the count, so a machine with fewer cores runs on all of them. The threads
  govern the parallel distance computations of the C++ backend, while generation and standardization run in R.
