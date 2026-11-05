#!/usr/bin/env Rscript
# ==============================================================================
# corr_plt_utils.R
# ==============================================================================
#
# Shared plotter for the design-matrix *correlation heatmaps* embedded in the
# mc_sim demo READMEs — ggplot2 port of the Python plotter of the same name
# (corr_plt_utils.py). It renders the Pearson correlation matrix of an
# exemplary design X drawn from each demo's own DGP, in NATURAL variable order
# (no hierarchical clustering / no reordering) so the raw dependency structure
# the demo studies is visible as-is.
#
# The correlation of X depends only on the DGP's covariance parameters
# (n, p, rho, block layout, ...) and the RNG seed — never on the support,
# coefficients, SNR or the response y — so the exemplar builders below fix a
# seed and pass a nominal support purely to satisfy the DGP signatures.
#
# Usage (normally invoked via a suite's generate_corr_plots.sh):
#   Rscript corr_plt_utils.R <demo_key> [--out PATH.png] [--dpi N]
#   Rscript corr_plt_utils.R --list
#
# Each sub-suite's DGP module lives two levels down (ad/trex_core/, ad/trex_da/, ...); the
# builders source only the DGP file actually needed, into a private
# environment. All 22 Python keys are registered: da_* (8), trex_gauss,
# gvs_* (8), scr_* (4) and spca_factor.
# ==============================================================================

suppressMessages(library(ggplot2))

.HERE <- local({
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(a) > 0) dirname(normalizePath(sub("^--file=", "", a[[1L]]))) else normalizePath(".")
})

.SEED <- 2026L
.TITLE <- "Design-matrix correlation (natural order, no clustering)"

# Leading-block crop sizes for narrow-band designs whose structure is
# sub-pixel at full p (mirrors the Python module; unused by trex_gauss).
.NB <- 150L   # narrow-band crop
.SP <- 75L    # small structured-prefix crop


#' Source one file of a sub-suite (below ad/) into a private environment and return it.
.suite_dgp <- function(...) {
  path <- file.path(.HERE, "ad", ...)
  env <- new.env(parent = globalenv())
  sys.source(path, envir = env)
  env
}


# ==============================================================================
# Rendering
# ==============================================================================

#' Render corr(X) as a diverging heatmap in natural order; save to out_path.
#'
#' X        : (n x p) design matrix.
#' title    : bold heading (what the plot is).
#' subtitle : DGP description line (which dataset).
#' crop     : if set, show only the leading crop x crop sub-block (used for
#'            narrow-band designs whose structure is sub-pixel at full p); the
#'            crop is noted in the caption. corr of a column subset equals the
#'            leading block of the full correlation, so this stays faithful.
plot_correlation_matrix <- function(X, out_path, title, subtitle = NULL,
                                    crop = NULL, dpi = 150) {
  p_full <- ncol(X)
  if (!is.null(crop) && crop < p_full) {
    X <- X[, seq_len(crop), drop = FALSE]
    note <- sprintf("leading %dx%d block of the p=%d design",
                    crop, crop, p_full)
    subtitle <- if (is.null(subtitle)) note else sprintf("%s  |  %s", subtitle, note)
  }

  C <- stats::cor(X)
  p <- ncol(C)

  # Long form for geom_raster: as.vector() is column-major, matching
  # expand.grid()'s first-index-fastest order.
  df <- expand.grid(i = seq_len(p), j = seq_len(p))
  df$value <- as.vector(C)

  # Sparse, integer ticks (variable 1 in the top-left corner, as imshow's
  # origin="upper").
  n_ticks <- min(6L, p)
  ticks <- round(seq(1L, p, length.out = n_ticks))

  gg <- ggplot(df, aes(x = j, y = i, fill = value)) +
    geom_raster() +
    # RdBu reversed (matplotlib "RdBu_r"): -1 blue, 0 white, +1 red.
    scale_fill_distiller(palette = "RdBu", direction = -1,
                         limits = c(-1.0, 1.0),
                         breaks = c(-1.0, -0.5, 0.0, 0.5, 1.0),
                         name = "Pearson correlation") +
    scale_x_continuous(breaks = ticks, expand = c(0, 0)) +
    scale_y_reverse(breaks = ticks, expand = c(0, 0)) +
    coord_fixed() +
    labs(title = title, subtitle = subtitle,
         x = "variable index", y = "variable index") +
    guides(fill = guide_colorbar(barheight = grid::unit(0.62, "npc"),
                                 title.position = "right")) +
    theme_minimal(base_size = 11) +
    theme(panel.grid = element_blank(),
          axis.ticks = element_line(colour = "grey30", linewidth = 0.3),
          plot.title = element_text(size = 11, hjust = 0.5),
          plot.subtitle = element_text(size = 9, hjust = 0.5),
          legend.title = element_text(size = 9, angle = 270, hjust = 0.5))

  dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
  ggsave(out_path, plot = gg, width = 7.3, height = 6.4, dpi = dpi,
         bg = "white")
  out_path
}


# ==============================================================================
# Exemplar builders  ->  list(X, title, subtitle, crop)
#
# One entry per mc_sim demo. Keep the (n, p, covariance) params in sync with
# the demo's own DGP call *and* with the Python corr_plt_utils.py exemplars;
# support/SNR are irrelevant to corr(X) and only satisfy the DGP signatures.
# (R supports are 1-based: 1:10 mirrors Python's np.arange(10).)
# ==============================================================================

# --- trex_da: 8 DGPs (ad/trex_da/trex_da_dgps.R) ------------------------------

.da <- function() .suite_dgp("trex_da", "trex_da_dgps.R")

.ex_da_ar1 <- function() {
  d <- .da()$dgp_ar1_snr(n = 300L, p = 1000L, support = 1:10, rho = 0.7,
                         seed = .SEED)
  list(X = d$X, title = .TITLE, crop = .NB,
       subtitle = "AR(1) Toeplitz, rho=0.7  |  n=300, p=1000")
}

.ex_da_equi <- function() {
  d <- .da()$dgp_equi_snr(n = 300L, p = 1000L, support = 1:10, rho = 0.25,
                          seed = .SEED)
  list(X = d$X, title = .TITLE, crop = NULL,
       subtitle = "Equicorrelation, rho=0.25  |  n=300, p=1000")
}

.ex_da_nn <- function() {
  d <- .da()$dgp_nn_snr(n = 300L, p = 1000L, support = 1:10, kappa = 3L,
                        rho = 0.7, seed = .SEED)
  list(X = d$X, title = .TITLE, crop = .NB,
       subtitle = "Nearest-neighbour, kappa=3, rho=0.7  |  n=300, p=1000")
}

.ex_da_ar1_block <- function() {
  d <- .da()$dgp_ar1_block_snr(n = 300L, p = 1000L, support = 1:10,
                               n_blocks = 5L, rho = 0.7, seed = .SEED)
  list(X = d$X, title = .TITLE, crop = NULL,
       subtitle = "Block-diagonal AR(1), 5 blocks, rho=0.7  |  n=300, p=1000")
}

.ex_da_ar1_block_white <- function() {
  d <- .da()$dgp_ar1_block_white_snr(n = 300L, p_ar = 25L, p_white = 975L,
                                     support = 1:5, n_blocks = 5L, rho = 0.7,
                                     seed = .SEED)
  list(X = d$X, title = .TITLE, crop = .SP,
       subtitle = "AR(1) blocks (25) + white (975), rho=0.7  |  n=300, p=1000")
}

.ex_da_block_toeplitz_hvt <- function() {
  d <- .da()$dgp_block_toeplitz_hvt_snr(n = 300L, p = 1000L, support = 1:10,
                                        n_blocks = 5L, rho = 0.8, nu = 3.0,
                                        seed = .SEED)
  list(X = d$X, title = .TITLE, crop = NULL,
       subtitle = paste0("Heavy-tailed block Toeplitz, 5 blocks, rho=0.8, ",
                         "nu=3  |  n=300, p=1000"))
}

.ex_da_ht_block_white <- function() {
  d <- .da()$dgp_ht_block_white_snr(n = 300L, p_ar = 25L, p_white = 475L,
                                    support = 1:5, n_blocks = 5L, rho = 0.8,
                                    nu = 3.0, seed = .SEED)
  list(X = d$X, title = .TITLE, crop = .SP,
       subtitle = paste0("Heavy-tailed AR blocks (25) + t3 white (475), ",
                         "rho=0.8  |  n=300, p=500"))
}

.ex_da_groups <- function() {
  groups <- list((seq_len(1000L) - 1L) %/% 10L,
                 (seq_len(1000L) - 1L) %/% 50L,
                 (seq_len(1000L) - 1L) %/% 250L)
  d <- .da()$dgp_groups_toeplitz_leaf(n = 300L, p = 1000L, support = 1:10,
                                      amplitude = 3.0, groups = groups,
                                      rho_levels = c(0.55, 0.25, 0.10),
                                      phi = 0.50, seed = .SEED)
  list(X = d$X, title = .TITLE, crop = NULL,
       subtitle = "Prior groups (10/50/250) + Toeplitz leaf, phi=0.5  |  n=300, p=1000")
}


# --- trex_core: iid Gaussian (ad/trex_core/trex_sim_utils.R dgp_gauss_snr) ----

.ex_trex_gauss <- function() {
  g <- .suite_dgp("trex_core", "trex_sim_utils.R")
  d <- g$dgp_gauss_snr(n = 300L, p = 1000L, support = 1:10,
                       coefs = rep(1.0, 10L), snr = 1.0, seed = .SEED)
  list(X = d$X, title = .TITLE, crop = NULL,
       subtitle = paste0("i.i.d. Gaussian, independent columns ",
                         "(corr ~ identity)  |  n=300, p=1000"))
}

# --- trex_gvs: 8 DGPs (ad/trex_gvs/trex_gvs_dgps.R) ---------------------------
# The gvs DGPs require an snr argument; a nominal snr = 1 is passed — corr(X)
# never depends on it.

.gvs <- function() .suite_dgp("trex_gvs", "trex_gvs_dgps.R")

.ex_gvs_hastie <- function() {
  d <- .gvs()$dgp_hastie_snr(n = 200L, p = 500L, snr = 1.0, seed = .SEED)
  list(X = d$X, title = .TITLE, crop = NULL,
       subtitle = "Hastie grouped elastic-net design  |  n=200, p=500")
}

.ex_gvs_scattered <- function() {
  d <- .gvs()$dgp_scattered_grouped_snr(n = 200L, p = 500L, snr = 1.0,
                                        seed = .SEED)
  list(X = d$X, title = .TITLE, crop = NULL,
       subtitle = "Scattered correlated groups (gaps randomised)  |  n=200, p=500")
}

.ex_gvs_mixed_blocks <- function() {
  d <- .gvs()$dgp_mixed_blocks_snr(n = 200L, p = 500L, snr = 1.0, seed = .SEED)
  list(X = d$X, title = .TITLE, crop = NULL,
       subtitle = "Mixed-size scattered blocks (20/50/80/65)  |  n=200, p=500")
}

.ex_gvs_neg_traps <- function() {
  d <- .gvs()$dgp_neg_traps_snr(n = 200L, p = 500L, snr = 1.0, seed = .SEED)
  list(X = d$X, title = .TITLE, crop = NULL,
       subtitle = "Negative-correlation traps  |  n=200, p=500")
}

.ex_gvs_t3_blocks <- function() {
  d <- .gvs()$dgp_t3_equi_blocks_snr(n = 200L, p = 500L, snr = 1.0,
                                     rho = 0.75, df = 3, seed = .SEED)
  list(X = d$X, title = .TITLE, crop = NULL,
       subtitle = "Student-t(3) equicorrelated blocks, rho=0.75  |  n=200, p=500")
}

.ex_gvs_ar1_blocks <- function() {
  d <- .gvs()$dgp_ar1_blocks_snr(n = 200L, p = 500L, snr = 1.0, rho = 0.85,
                                 seed = .SEED)
  list(X = d$X, title = .TITLE, crop = NULL,
       subtitle = "Block-diagonal AR(1), rho=0.85  |  n=200, p=500")
}

.ex_gvs_arma_blocks <- function() {
  d <- .gvs()$dgp_arma_blocks_snr(n = 200L, p = 500L, snr = 1.0, seed = .SEED)
  list(X = d$X, title = .TITLE, crop = NULL,
       subtitle = "Block-diagonal ARMA (ar=0.8, ma=0.5/0.3/0.1)  |  n=200, p=500")
}

.ex_gvs_block_equicorr <- function() {
  d <- .gvs()$dgp_block_equicorr(n = 200L, p = 500L, G = 100L, block_size = 5L,
                                 n_active_blocks = 10L, rho = 0.9, snr = 1.0,
                                 seed = .SEED)
  # 100 blocks of 5 columns are sub-pixel at full p — crop so the tiny
  # 5 x 5 diagonal blocks stay legible.
  list(X = d$X, title = .TITLE, crop = .SP,
       subtitle = "Equicorrelated blocks (100 x size-5), rho=0.9  |  n=200, p=500")
}


# --- trex_screening: 4 DGPs (ad/trex_screening/trex_scr_sim_utils.R) ----------

.scr <- function() .suite_dgp("trex_screening", "trex_scr_sim_utils.R")

.ex_scr_iid <- function() {
  d <- .scr()$dgp_iid_snr(n = 300L, p = 1000L, true_support = 1:10,
                          coefs = rep(1.0, 10L), snr = 1.0, seed = .SEED)
  list(X = d$X, title = .TITLE, crop = NULL,
       subtitle = "i.i.d. Gaussian, independent columns  |  n=300, p=1000")
}

.ex_scr_ar1 <- function() {
  d <- .scr()$dgp_ar1(n = 300L, p = 1000L, p1 = 10L, rho = 0.5, snr = 1.0,
                      beta_val = 1.0, seed = .SEED)
  list(X = d$X, title = .TITLE, crop = .NB,
       subtitle = "AR(1) column correlation, rho=0.5  |  n=300, p=1000")
}

.ex_scr_equi <- function() {
  d <- .scr()$dgp_equi(n = 300L, p = 1000L, p1 = 10L, rho = 0.4, snr = 1.0,
                       beta_val = 1.0, seed = .SEED)
  list(X = d$X, title = .TITLE, crop = NULL,
       subtitle = "Equicorrelation (one shared factor), rho=0.4  |  n=300, p=1000")
}

.ex_scr_block_equi <- function() {
  d <- .scr()$dgp_block_equi(n = 300L, p = 1000L, p1 = 10L, rho = 0.9,
                             n_blocks = 5L, snr = 1.0, beta_val = 1.0,
                             seed = .SEED)
  list(X = d$X, title = .TITLE, crop = NULL,
       subtitle = "Block-equicorrelation, 5 blocks, rho=0.9  |  n=300, p=1000")
}


# --- trex_spca: sparse factor model (ad/trex_spca/trex_spca_sim_utils.R) ------

.ex_spca_factor <- function() {
  g <- .suite_dgp("trex_spca", "trex_spca_sim_utils.R")
  d <- g$generate_sparse_factor_model(n = 50L, p = 100L, p1 = 5L, M = 3L,
                                      overlap_pool_size = 30L, seed = .SEED)
  list(X = d$X, title = .TITLE, crop = NULL,
       subtitle = "Sparse factor model (M=3, p1=5, pool=30)  |  n=50, p=100")
}


# key -> builder. Keys are suite-namespaced DGP names; the output file is
# <suite>/dgp_corr_plots/<key>.png (the per-suite generate_corr_plots.sh
# sets it).
EXEMPLARS <- list(
  # trex_da
  da_ar1                = .ex_da_ar1,
  da_equi               = .ex_da_equi,
  da_nn                 = .ex_da_nn,
  da_ar1_block          = .ex_da_ar1_block,
  da_ar1_block_white    = .ex_da_ar1_block_white,
  da_block_toeplitz_hvt = .ex_da_block_toeplitz_hvt,
  da_ht_block_white     = .ex_da_ht_block_white,
  da_groups             = .ex_da_groups,
  # trex
  trex_gauss         = .ex_trex_gauss,
  # trex_gvs
  gvs_hastie         = .ex_gvs_hastie,
  gvs_scattered      = .ex_gvs_scattered,
  gvs_mixed_blocks   = .ex_gvs_mixed_blocks,
  gvs_neg_traps      = .ex_gvs_neg_traps,
  gvs_t3_blocks      = .ex_gvs_t3_blocks,
  gvs_ar1_blocks     = .ex_gvs_ar1_blocks,
  gvs_arma_blocks    = .ex_gvs_arma_blocks,
  gvs_block_equicorr = .ex_gvs_block_equicorr,
  # trex_screening
  scr_iid            = .ex_scr_iid,
  scr_ar1            = .ex_scr_ar1,
  scr_equi           = .ex_scr_equi,
  scr_block_equi     = .ex_scr_block_equi,
  # trex_spca
  spca_factor        = .ex_spca_factor
)


# ==============================================================================
# CLI
# ==============================================================================

.default_out <- function(demo_key) {
  # The per-suite generate_corr_plots.sh sets --out to
  # <suite>/dgp_corr_plots/<key>.png; this is only a CWD-relative fallback.
  file.path("dgp_corr_plots", paste0(demo_key, ".png"))
}

main <- function() {
  argv <- commandArgs(trailingOnly = TRUE)
  key <- NULL
  out <- NULL
  dpi <- 150
  list_only <- FALSE
  i <- 1L
  while (i <= length(argv)) {
    a <- argv[[i]]
    flag <- function() {
      i <<- i + 1L
      argv[[i]]
    }
    switch(a,
           "--out"  = {
             out <- flag()
           },
           "--dpi"  = {
             dpi <- as.numeric(flag())
           },
           "--list" = {
             list_only <- TRUE
           },
           {
             key <- a
           })
    i <- i + 1L
  }

  if (list_only || is.null(key)) {
    cat(sort(names(EXEMPLARS)), sep = "\n")
    return(invisible(0L))
  }
  if (!key %in% names(EXEMPLARS))
    stop("unknown demo key '", key, "'; try --list", call. = FALSE)

  ex <- EXEMPLARS[[key]]()
  if (is.null(out)) out <- .default_out(key)
  path <- plot_correlation_matrix(ex$X, out, title = ex$title,
                                  subtitle = ex$subtitle, crop = ex$crop,
                                  dpi = dpi)
  cat(sprintf("[corr] wrote %s  (p=%d, crop=%s)\n", path, ncol(ex$X),
              if (is.null(ex$crop)) "NULL" else as.character(ex$crop)))
  invisible(0L)
}

if (sys.nframe() == 0L) main()
