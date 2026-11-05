#!/usr/bin/env Rscript
# ==============================================================================
# trex_plt_utils.R
# ==============================================================================
#
# Shared plotting utilities for the classical T-Rex demo suite — ggplot2 port
# of the Python plotter of the same name (trex_plt_utils.py), kept visually
# aligned with it: tab10 (+ Dark2 overflow) row colours, hollow per-row
# markers, black dotted tFDR reference leading the legend, PNG + PDF output
# into the sibling simulation_results/plots/ folder.
#
# One module hosting the plot patterns for every demo in this folder whose
# results use the tidy CSV schema
#
#     solver,metric,snr,value
#
# (the "solver" column holds whatever the demo sweeps: base solvers in demos
# 02/03, L-loop strategies in demo 04, dummy distributions in demo 05, storage
# scenarios in demo 07's overlay, ...). Extra metrics (AvgL, AvgT) are noted
# and skipped; only FDR and TPR are plotted.
#
# Plot patterns hosted here:
#     plot_fdr_tpr_vs_snr()          — side-by-side FDR | TPR panels.
#     plot_fdr_tpr_vs_snr_grouped()  — de-cluttered 2x2 grid, rows split in half.
#     plot_comparison_grid()         — cross-solver 2xN grid: FDR/TPR rows x one
#                                      column per base solver, every sweep row as
#                                      a line (demo 04 L-strategies, demo 05
#                                      dummy distributions).
#     plot_scalability_dashboard()   — demo 08 runtime / memory / FDR vs problem
#                                      size (its own wider CSV schema,
#                                      n,p,is_barrier,solver,num_mc,snr,metric,
#                                      value — read via read_scalability()).
#     plot_sweep_grid()              — demos 09 / 10 FDR / TPR grid against a
#                                      swept column (tfdr, c), one line per value
#                                      of another (snr, p); long schema
#                                      solver,<grid columns>,metric,mean,sd.
#
# CLI modes (all typically driven from a demo's generate_plots.sh):
#     # default — per-CSV overview + grouped figure
#     Rscript trex_plt_utils.R <csv> --title "..." --legend-title "L-strategy"
#     # storage-scenario overlay (demo 07): rewrite each CSV's 'solver' column
#     # to the paired label and overlay everything on one FDR|TPR pair
#     Rscript trex_plt_utils.R <csv_c> <csv_d> --overlay "Label C" "Label D" \
#         --legend-title Scenario --stem my_overlay --no-grouped
#     # cross-solver comparison grid from several per-solver CSVs
#     Rscript trex_plt_utils.R grid --labels TLARS TOMP TAFS \
#         --csvs a_tlars.csv a_tomp.csv a_tafs.csv \
#         --legend-title "L-strategy" --stem my_grid --outdir <plots>
#     # demo-08 scalability dashboard from one or more wide CSVs (concatenated)
#     Rscript trex_plt_utils.R scalability <wide_csv> [<wide_csv> ...] \
#         --stem my_dash --outdir <plots>
#     # demo-09 / demo-10 sweep grid from one long CSV
#     Rscript trex_plt_utils.R sweep <csv> --x tfdr --lines snr --diagonal \
#         --stem my_sweep [--filter snr=1] [--outdir <plots>]
#
# Flags: --title T, --legend-title T, --tfdr F, --xscale {linear,sqrt,log},
#        --outdir DIR, --stem S, --formats png,pdf, --groups N, --no-grouped,
#        --overlay L1 L2. Unknown flags are ignored with a note (a demo's
#        generate_plots.sh forwards its "$@" to every plotter call).
#
# Output location: figures default to the simulation_results/plots/ sibling
# when the CSV lives in simulation_results/data/ (the suite convention);
# otherwise they are written next to the CSV. Override with --outdir. (The
# grid and scalability modes require an explicit --outdir, since they are not
# tied to a single CSV's location.)
#
# DIVERGENCE from the Python plotter: no interactive Plotly HTML output —
# this port renders the static ggplot2 png + pdf figure set only (the
# --no-plotly / --plotly-cdn flags are accepted and ignored so shared
# invocations keep working).
# ==============================================================================

suppressMessages({
  library(ggplot2)
  library(patchwork)
})

# ------------------------------------------------------------------------------
# Defaults (mirroring the Python module's constants)
# ------------------------------------------------------------------------------

STEM_SUFFIX <- "_fdr_tpr_vs_snr"
DEFAULT_TFDR <- 0.10          # target FDR control level (tFDR)
DEFAULT_LEGEND_TITLE <- "Method"
Y_HEADROOM <- 1.25            # budget above the largest FDR value / tFDR line
MAX_XTICKS <- 6               # thin x-axis ticks when the SNR grid is dense

METRICS <- c("FDR", "TPR")
METRIC_LABELS <- c(FDR = "False discovery rate", TPR = "True positive rate")

# matplotlib "tab10" + "Dark2" — 18 hand-picked qualitative colours, matching
# the Python plotter's row_colors(). Black is reserved for the tFDR line.
TAB10 <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")
DARK2 <- c("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
           "#e6ab02", "#a6761d", "#666666")
PALETTE <- c(TAB10, DARK2)

# Hollow per-row markers (o, s, ^, D, v, then line shapes) — overlapping curves
# stay distinguishable; markers cycle independently of colour. ggplot2 shapes
# 21-25 honour fill (drawn hollow via fill = NA), the rest are stroke-only.
ROW_SHAPES <- c(21L, 22L, 24L, 23L, 25L, 3L, 4L, 8L, 10L, 12L, 7L, 13L, 14L, 9L)

XSCALES <- c("linear", "sqrt", "log")
DEFAULT_XSCALE <- "linear"
# The demo-08 dashboard's x-axis is a problem size p spanning decades, not a
# sweep grid, so it keeps a logarithmic default.
SCAL_XSCALE <- "log"


# ------------------------------------------------------------------------------
# Shared helpers (house conventions)
# ------------------------------------------------------------------------------

#' Natural-sort key: digit runs are zero-padded so numbered labels order
#' numerically (SKIPL_5p < SKIPL_10p < SKIPL_20p); text compares
#' case-insensitively. Mirrors natural_sort_key() of the Python plotter.
.natural_key <- function(x) {
  vapply(as.character(x), function(s) {
    m <- gregexpr("[0-9]+", s)
    regmatches(s, m) <- lapply(regmatches(s, m),
                               function(d) sprintf("%020.0f", as.numeric(d)))
    tolower(s)
  }, character(1L), USE.NAMES = FALSE)
}

.natural_sort <- function(x) x[order(.natural_key(x), method = "radix")]

#' Load and validate a tidy Monte Carlo-results CSV (solver,metric,snr,value).
#' Rows are ordered by a natural sort of the 'solver' column (flowing straight
#' into the legend and the colour assignment), and only FDR/TPR rows are kept.
read_mc_results <- function(path) {
  if (!file.exists(path))
    stop("Input file not found:\n  ", path,
         "\nPass the results CSV path as the first argument.", call. = FALSE)

  df <- utils::read.csv(path, strip.white = TRUE, stringsAsFactors = FALSE)
  names(df) <- tolower(trimws(names(df)))

  required <- c("solver", "metric", "snr", "value")
  missing <- setdiff(required, names(df))
  if (length(missing) > 0)
    stop("CSV is missing required columns: ", paste(missing, collapse = ", "),
         "\nFound columns: ", paste(names(df), collapse = ", "), call. = FALSE)

  df <- df[, required]
  df$solver <- trimws(as.character(df$solver))
  df$metric <- toupper(trimws(as.character(df$metric)))
  df$snr <- as.numeric(df$snr)
  df$value <- as.numeric(df$value)

  missing_metrics <- setdiff(METRICS, unique(df$metric))
  if (length(missing_metrics) > 0)
    stop("CSV is missing required metric(s): ",
         paste(missing_metrics, collapse = ", "),
         "\nFound metrics: ", paste(sort(unique(df$metric)), collapse = ", "),
         call. = FALSE)

  # Some demos also record extra metrics (e.g. AvgL, AvgT); keep only the
  # metrics this module plots and note the rest.
  extra <- setdiff(unique(df$metric), METRICS)
  if (length(extra) > 0)
    cat(sprintf("note: ignoring non-plotted metric(s): %s\n",
                paste(sort(extra), collapse = ", ")))
  df <- df[df$metric %in% METRICS, ]

  rows <- .natural_sort(unique(df$solver))
  df$solver <- factor(df$solver, levels = rows)
  df[order(df$metric, as.integer(df$solver), df$snr), ]
}

#' `.../simulation_results/data/x.csv` -> `.../simulation_results/plots`;
#' falls back to the CSV's own directory outside that layout.
default_plots_dir <- function(csv_path) {
  parent <- dirname(csv_path)
  if (basename(parent) == "data") file.path(dirname(parent), "plots") else parent
}

#' Stable colour / hollow-marker assignment keyed by row order.
row_colors <- function(rows) {
  setNames(PALETTE[(seq_along(rows) - 1L) %% length(PALETTE) + 1L], rows)
}

row_shapes <- function(rows) {
  setNames(ROW_SHAPES[(seq_along(rows) - 1L) %% length(ROW_SHAPES) + 1L], rows)
}

#' Split data rows into n_groups contiguous, roughly equal groups.
split_rows <- function(rows, n_groups = 2L) {
  size <- ceiling(length(rows) / n_groups)
  split(rows, (seq_along(rows) - 1L) %/% size)
}

#' Upper FDR y-limit: the larger of the data max and the tFDR line, + headroom.
fdr_axis_top <- function(df, tfdr) {
  fdr_max <- max(df$value[df$metric == "FDR"], na.rm = TRUE)
  max(fdr_max, if (tfdr > 0) tfdr else 0.0) * Y_HEADROOM
}

#' Readable subset of x-tick positions for a log-scaled sweep grid: small grids
#' keep every value; dense grids are thinned to ~max_ticks values spaced evenly
#' in log space (endpoints retained). Mirrors choose_ticks() in Python.
choose_ticks <- function(values, max_ticks = MAX_XTICKS) {
  values <- sort(unique(values))
  if (length(values) <= max_ticks) return(values)
  lo <- log10(values[1L])
  hi <- log10(values[length(values)])
  ticks <- numeric(0)
  for (i in seq_len(max_ticks) - 1L) {
    target <- lo + (hi - lo) * i / (max_ticks - 1L)
    nearest <- values[which.min(abs(log10(values) - target))]
    if (!nearest %in% ticks) ticks <- c(ticks, nearest)
  }
  ticks
}

#' Show at most ~10 markers per curve so dense grids stay legible
#' (the ggplot2 equivalent of matplotlib's markevery).
marker_stride <- function(values) max(1L, length(values) %/% 10L)

#' Thin a frame's rows to every mev-th swept x value (for geom_point only;
#' geom_line always uses the full grid).
.strided <- function(df, values, mev, col = "snr") {
  if (mev <= 1L) return(df)
  keep <- sort(unique(values))[seq(1L, length(unique(values)), by = mev)]
  df[df[[col]] %in% keep, ]
}

#' Sweep x-axis scale: 'linear' (default) and 'sqrt' take <= MAX_XTICKS majors
#' from an extended-breaks locator so labels cannot collide; 'log' restores the
#' logarithmic axis tick-marked at choose_ticks(). ticks + force_ticks pin the
#' majors to fixed positions (the scalability p-axis). Mirrors
#' apply_sweep_xaxis() of the Python plotter.
.sweep_x_scale <- function(values, xscale = DEFAULT_XSCALE, ticks = NULL,
                           force_ticks = FALSE, angle = 0) {
  values <- sort(unique(values))
  lo <- min(values)
  hi <- max(values)
  fmt <- function(b) {
    format(b, big.mark = ",", trim = TRUE,
           scientific = FALSE, drop0trailing = TRUE)
  }
  guide <- if (angle != 0) guide_axis(angle = angle) else guide_axis()

  if (xscale == "log") {
    brk <- if (!is.null(ticks)) ticks else choose_ticks(values)
    return(scale_x_log10(breaks = brk, labels = fmt, guide = guide,
                         limits = c(lo * 0.82, hi * 1.18),
                         expand = expansion(0)))
  }

  brk <- if (force_ticks && !is.null(ticks)) ticks else scales::breaks_extended(MAX_XTICKS)
  if (xscale == "sqrt") {
    span_sqrt <- sqrt(hi) - sqrt(max(lo, 0.0))
    pad <- if (span_sqrt > 0) (span_sqrt * 0.04)^2 else 0.5
    if (!(force_ticks && !is.null(ticks)))
      brk <- scales::breaks_extended(MAX_XTICKS)(c(lo, hi))
    return(scale_x_sqrt(breaks = brk, labels = fmt, guide = guide,
                        limits = c(lo, hi + pad), expand = expansion(0)))
  }

  span <- hi - lo
  pad <- if (span > 0) span * 0.04 else 0.5
  scale_x_continuous(breaks = brk, labels = fmt, guide = guide,
                     limits = c(lo - pad, hi + pad), expand = expansion(0))
}

#' The shared panel look: recessive grid, bold panel title.
#'
#' Grid convention (family-wide): major gridlines solid, minor gridlines
#' dashed, both faint enough to stay behind the data — the curves carry the
#' figure, the grid only helps read values off it.
.panel_theme <- function() {
  theme_bw(base_size = 11) +
    theme(
      panel.grid.major = element_line(colour = "grey88", linewidth = 0.3),
      panel.grid.minor = element_line(colour = "grey92", linewidth = 0.25,
                                      linetype = "dashed"),
      plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
      legend.title = element_text(size = 10, face = "bold"),
      legend.text = element_text(size = 9),
      legend.background = element_rect(colour = "grey55", linewidth = 0.3)
    )
}

#' Rate-axis breaks: FDR panels resolve to 0.02 (minor 0.01), the 0-1 rate
#' panels (TPR) to 0.2 (minor 0.1). Wide FDR ranges fall back to a coarser
#' step so the axis never crowds.
.rate_y_scale <- function(metric, top = NULL) {
  if (identical(metric, "FDR")) {
    step <- if (!is.null(top) && top > 0.3) 0.05 else 0.02
  } else {
    step <- 0.2
  }
  scale_y_continuous(breaks = scales::breaks_width(step),
                     minor_breaks = scales::breaks_width(step / 2))
}

#' Place a panel's own legend inside the plotting area (grouped figure /
#' scalability dashboard, matching the matplotlib per-axes legends).
.legend_inside <- function(x, y) {
  theme(legend.position = "inside",
        legend.position.inside = c(x, y),
        legend.justification.inside = c(round(x), round(y)),
        legend.title = element_blank(),
        legend.text = element_text(size = 8),
        legend.key.size = grid::unit(0.9, "lines"),
        legend.margin = margin(2, 4, 2, 4))
}

#' The tFDR reference layers: black dotted hline + a linetype legend entry that
#' leads the collected legend (guide order 1 < the colour/shape guides' 2).
.tfdr_layers <- function(tfdr) {
  lbl <- sprintf("tFDR = %g", tfdr)
  list(
    geom_hline(aes(yintercept = tfdr, linetype = lbl),
               color = "black", linewidth = 0.8),
    scale_linetype_manual(name = NULL, values = setNames("dotted", lbl),
                          guide = guide_legend(order = 1))
  )
}

save_figure <- function(plot, outdir, stem, formats = c("png", "pdf"),
                        width = 13.0, height = 5.6) {
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  for (ext in formats) {
    path <- file.path(outdir, paste0(stem, ".", ext))
    ggsave(path, plot = plot, width = width, height = height,
           dpi = 300, bg = "white")
    cat(sprintf("Wrote: %s\n", path))
  }
}


# ------------------------------------------------------------------------------
# Pattern 1 — side-by-side FDR and TPR versus SNR panels
# ------------------------------------------------------------------------------

#' One FDR-or-TPR panel: all `rows` as coloured hollow-marker lines vs SNR.
.mc_panel <- function(df, metric, rows, colors, shapes, tfdr, legend_title,
                      xscale, snr_values, fdr_top, panel_title = metric,
                      show_ylab = TRUE, show_xlab = TRUE) {
  sub <- df[df$metric == metric & df$solver %in% rows, ]
  sub$solver <- factor(as.character(sub$solver), levels = rows)
  mev <- marker_stride(snr_values)

  p <- ggplot(sub, aes(x = snr, y = value, color = solver, shape = solver))
  if (metric == "FDR" && tfdr > 0) p <- p + .tfdr_layers(tfdr)
  p <- p +
    geom_line(linewidth = 0.8) +
    geom_point(data = .strided(sub, snr_values, mev),
               size = 2.4, fill = NA, stroke = 0.9) +
    scale_color_manual(name = legend_title, values = colors, drop = FALSE) +
    scale_shape_manual(name = legend_title, values = shapes, drop = FALSE) +
    guides(color = guide_legend(order = 2), shape = guide_legend(order = 2)) +
    labs(title = panel_title,
         x = if (show_xlab) "Signal-to-noise ratio (SNR)" else NULL,
         y = if (show_ylab) unname(METRIC_LABELS[metric]) else NULL) +
    .sweep_x_scale(snr_values, xscale, ticks = choose_ticks(snr_values)) +
    .rate_y_scale(metric, fdr_top) +
    .panel_theme()
  ylim <- if (metric == "FDR") c(0.0, fdr_top) else c(-0.02, 1.02)
  p + coord_cartesian(ylim = ylim)
}

#' Pattern 1 — FDR | TPR overview with the legend collected outside right.
#' When tfdr > 0 a black dotted reference marks the target FDR level on the
#' FDR panel, and the FDR y-axis is scaled to include it.
plot_fdr_tpr_vs_snr <- function(df, title, tfdr = DEFAULT_TFDR,
                                legend_title = DEFAULT_LEGEND_TITLE,
                                xscale = DEFAULT_XSCALE) {
  rows <- levels(df$solver)
  snr_values <- sort(unique(df$snr))
  colors <- row_colors(rows)
  shapes <- row_shapes(rows)
  fdr_top <- fdr_axis_top(df, tfdr)

  panels <- lapply(METRICS, function(metric) {
    .mc_panel(df, metric, rows, colors, shapes, tfdr, legend_title,
              xscale, snr_values, fdr_top)
  })

  (panels[[1L]] | panels[[2L]]) +
    plot_layout(guides = "collect") +
    plot_annotation(title = title,
                    theme = theme(plot.title = element_text(
                      size = 14, face = "bold", hjust = 0.5
                    )))
}


# ------------------------------------------------------------------------------
# Pattern 2 — grouped 2xN layout (rows = FDR/TPR, columns = row groups)
# ------------------------------------------------------------------------------

#' Each panel carries only a fraction of the curves, so overlapping regions are
#' easier to read than in the combined overview. Colours and markers are
#' identical to the overview figure, all FDR panels share one y-scale, and each
#' panel keeps its own inside legend (FDR upper-left, TPR lower-right).
plot_fdr_tpr_vs_snr_grouped <- function(df, title, tfdr = DEFAULT_TFDR,
                                        legend_title = DEFAULT_LEGEND_TITLE,
                                        n_groups = 2L,
                                        xscale = DEFAULT_XSCALE) {
  rows <- levels(df$solver)
  snr_values <- sort(unique(df$snr))
  colors <- row_colors(rows)
  shapes <- row_shapes(rows)
  fdr_top <- fdr_axis_top(df, tfdr)
  groups <- split_rows(rows, n_groups)

  # Row index ranges for the column titles (1-based, inclusive).
  ends <- cumsum(lengths(groups))
  starts <- c(1L, head(ends, -1L) + 1L)

  panels <- list()
  for (ri in seq_along(METRICS)) {
    metric <- METRICS[ri]
    for (gi in seq_along(groups)) {
      p <- .mc_panel(df, metric, groups[[gi]], colors[groups[[gi]]],
                     shapes[groups[[gi]]], tfdr, legend_title, xscale,
                     snr_values, fdr_top,
                     panel_title = if (ri == 1L)
                       sprintf("%s %d-%d", legend_title,
                               starts[gi], ends[gi]) else NULL,
                     show_ylab = gi == 1L,
                     show_xlab = ri == length(METRICS))
      p <- p + if (metric == "FDR") .legend_inside(0.03, 0.97) else .legend_inside(0.97, 0.03)
      panels[[length(panels) + 1L]] <- p
    }
  }

  wrap_plots(panels, nrow = length(METRICS), byrow = TRUE) +
    plot_annotation(title = title,
                    theme = theme(plot.title = element_text(
                      size = 14, face = "bold", hjust = 0.5
                    )))
}


# ------------------------------------------------------------------------------
# Pattern 4 — cross-solver comparison grid (demo 04 / 05 `grid` mode)
# ------------------------------------------------------------------------------

#' Union of the 'solver'-column values across per-column frames, in first-seen
#' order (each frame is already natural-sorted by read_mc_results). One ordered
#' list keeps the colour / marker assignment identical in every column.
ordered_grid_rows <- function(frames) {
  seen <- character(0)
  for (df in frames)
    seen <- c(seen, setdiff(levels(df$solver), seen))
  seen
}

#' `frames` is an *ordered* named list (one tidy frame per base solver, e.g.
#' TLARS / TOMP / TAFS). 2xN grid: rows = metric (FDR top, TPR below), columns
#' = the frames' names; every panel carries all swept rows, colour/marker keyed
#' to the row so a given row matches across columns. All FDR panels share one
#' y-scale, all TPR panels share [0, 1]; one legend right of the grid.
plot_comparison_grid <- function(frames, title,
                                 legend_title = DEFAULT_LEGEND_TITLE,
                                 tfdr = DEFAULT_TFDR,
                                 xscale = DEFAULT_XSCALE) {
  rows <- ordered_grid_rows(frames)
  colors <- row_colors(rows)
  shapes <- row_shapes(rows)

  # A shared SNR grid / y-top across panels keeps the columns comparable.
  all_snr <- sort(unique(unlist(lapply(frames, function(df) df$snr))))
  fdr_top <- max(vapply(frames, fdr_axis_top, numeric(1L), tfdr = tfdr))

  panels <- list()
  for (ri in seq_along(METRICS)) {
    metric <- METRICS[ri]
    for (ci in seq_along(frames)) {
      panels[[length(panels) + 1L]] <-
        .mc_panel(frames[[ci]], metric, rows, colors, shapes, tfdr,
                  legend_title, xscale, all_snr, fdr_top,
                  panel_title = if (ri == 1L) names(frames)[ci] else NULL,
                  show_ylab = ci == 1L,
                  show_xlab = ri == length(METRICS))
    }
  }

  wrap_plots(panels, nrow = length(METRICS), byrow = TRUE) +
    plot_layout(guides = "collect") +
    plot_annotation(title = title,
                    theme = theme(plot.title = element_text(
                      size = 14, face = "bold", hjust = 0.5
                    )))
}


# ------------------------------------------------------------------------------
# Pattern 5 — sweep grid (demos 09 and 10, long schema with a mean column)
# ------------------------------------------------------------------------------
# Demos 09 and 10 write the long schema
#     solver,<grid columns>,metric,mean,sd
# (grid columns snr,tfdr for demo 09 and snr,p,n,c for demo 10).

SWEEP_X_LABELS <- c(tfdr = "Target FDR (tFDR)", c = "n / (s \u00b7 ln p)")
SWEEP_LINE_LABELS <- c(snr = "SNR", p = "p", n = "n")

#' Load a long sweep CSV (solver, <grid columns>, metric, mean, sd).
read_sweep <- function(path) {
  df <- utils::read.csv(path, stringsAsFactors = FALSE, strip.white = TRUE)
  names(df) <- tolower(trimws(names(df)))
  missing <- setdiff(c("solver", "metric", "mean"), names(df))
  if (length(missing) > 0L)
    stop(sprintf("Sweep CSV is missing columns %s: %s",
                 paste(missing, collapse = ", "), path), call. = FALSE)
  df$solver <- trimws(as.character(df$solver))
  df$metric <- trimws(as.character(df$metric))
  df
}

#' 2 x N grid: FDR (top) and TPR (bottom) against `x`, one column per solver
#' and one line per value of `lines` (mean over the trials). The FDR panels
#' carry the diagonal FDR = tFDR when `diagonal` is set, else the target line
#' at `tfdr`. Mirrors plot_sweep_grid() of the Python plotter.
plot_sweep_grid <- function(df, x, lines, title, diagonal = FALSE,
                            tfdr = DEFAULT_TFDR, xscale = DEFAULT_XSCALE) {
  solvers <- unique(df$solver)
  line_vals <- sort(unique(df[[lines]]))
  line_name <- if (lines %in% names(SWEEP_LINE_LABELS)) SWEEP_LINE_LABELS[[lines]] else lines
  line_lbls <- sprintf("%s = %s", line_name, format(line_vals, big.mark = ",", trim = TRUE,
                                                    drop0trailing = TRUE))
  # viridis per line value, with orange instead of its yellow end
  cols <- viridisLite::viridis(length(line_vals))
  pos <- if (length(line_vals) > 1L) seq(0, 1, length.out = length(line_vals)) else 0
  cols[pos > 0.85] <- "#FF7F0E"
  cols <- setNames(cols, line_lbls)
  shapes <- setNames(ROW_SHAPES[seq_along(line_vals)], line_lbls)
  x_vals <- sort(unique(df[[x]]))
  x_label <- if (x %in% names(SWEEP_X_LABELS)) SWEEP_X_LABELS[[x]] else x

  fdr_vals <- df$mean[df$metric == "FDR"]
  fdr_top <- max(max(fdr_vals) * 1.1, (if (diagonal) max(x_vals) else tfdr) * 1.1)
  df$line_f <- factor(line_lbls[match(df[[lines]], line_vals)], levels = line_lbls)
  df$x_val <- df[[x]]

  panels <- list()
  for (metric in METRICS) {
    for (ci in seq_along(solvers)) {
      sub <- df[df$solver == solvers[[ci]] & df$metric == metric, ]
      sub <- sub[order(sub$line_f, sub$x_val), ]
      p <- ggplot(sub, aes(x = x_val, y = mean, color = line_f, shape = line_f))
      if (metric == "FDR") {
        if (diagonal) {
          ref <- data.frame(x0 = 0, y0 = 0, x1 = max(x_vals), y1 = max(x_vals), lbl = "FDR = tFDR")
          p <- p +
            geom_segment(data = ref, inherit.aes = FALSE,
                         aes(x = x0, y = y0, xend = x1, yend = y1, linetype = lbl),
                         color = "black", linewidth = 0.8) +
            scale_linetype_manual(name = NULL, values = c("FDR = tFDR" = "dotted"),
                                  guide = guide_legend(order = 1))
        } else {
          p <- p + .tfdr_layers(tfdr)
        }
      }
      p <- p +
        geom_line(linewidth = 0.9) +
        geom_point(size = 2.2, fill = NA, stroke = 0.9) +
        scale_color_manual(name = NULL, values = cols, drop = FALSE) +
        scale_shape_manual(name = NULL, values = shapes, drop = FALSE) +
        guides(color = guide_legend(order = 2), shape = guide_legend(order = 2)) +
        labs(title = if (metric == "FDR") scal_solver_label(solvers[[ci]]) else NULL,
             x = if (metric == "TPR") x_label else NULL,
             y = if (ci == 1L) sprintf("%s (%s)", metric, tolower(METRIC_LABELS[[metric]])) else NULL) +
        .sweep_x_scale(x_vals, xscale) +
        .rate_y_scale(metric, fdr_top) +
        .panel_theme() +
        coord_cartesian(ylim = if (metric == "FDR") c(0.0, fdr_top) else c(0.0, 1.05))
      panels[[length(panels) + 1L]] <- p
    }
  }

  wrap_plots(panels, nrow = length(METRICS), byrow = TRUE) +
    plot_layout(guides = "collect") +
    plot_annotation(title = title,
                    theme = theme(plot.title = element_text(
                      size = 14, face = "bold", hjust = 0.5
                    )))
}


# ------------------------------------------------------------------------------
# Demo 08 — scalability dashboard (wider CSV schema than the rest)
# ------------------------------------------------------------------------------
# Demo 08's CSVs (base block plus large-p block, one CSV each) use the wide
# n x p benchmark schema
#     n,p,is_barrier,solver,num_mc,snr,metric,value
# read (and concatenated) via read_scalability() and rendered by
# plot_scalability_dashboard().

# Base solvers, in a fixed display order with stable colours / markers, as the
# demo writes them into the 'solver' column.
SCAL_SOLVERS <- c("TLARS", "TOMP", "TAFS")
SCAL_SOLVER_COLORS <- c(TLARS = "#1f77b4", TOMP = "#d62728", TAFS = "#2ca02c")
SCAL_SOLVER_SHAPES <- c(TLARS = 21L, TOMP = 22L, TAFS = 24L)
SCAL_SOLVER_LINES <- c(TLARS = "solid", TOMP = "dashed", TAFS = "dotted")
# TAFS is the only solver run with a non-zero AFS correlation parameter
# (rho_afs = 0.3 in the demo's solver table) — named in the legend since it is
# a defining part of the method; lookups stay keyed on the raw CSV name.
SCAL_SOLVER_RHO_AFS <- c(TAFS = 0.3)

scal_solver_label <- function(solver) {
  rho <- SCAL_SOLVER_RHO_AFS[solver]
  ifelse(is.na(rho), solver, sprintf("%s (rho = %g)", solver, rho))
}

# Column labels as they appear in the CSV's 'metric' field.
M_SELMEAN <- "SelMean_s"  # the runtime line (older CSVs only carry the median)
M_SELMED <- "SelMed_s"
M_SELMIN <- "SelMin_s"
M_SELMAX <- "SelMax_s"
M_PEAKRSS <- "PeakRSS_MiB"
M_RSS <- "RSS_MiB"     # resident set size after select() (MiB), per point
M_FOOT <- "Foot_MiB"   # physical footprint after select() (MiB), the anonymous heap
M_XY <- "X+y_MiB"
M_VOL <- "VolGiB"             # per T-iteration volume X + K*L*X at mean L (GiB)
M_VOL_LEGACY <- "DesignGiB"   # older CSVs: logical [X|D] at the dummy cap (GiB)
M_FDR <- "FDR"

#' Physical RAM in GiB (for the out-of-core reference line); NA on failure.
.scal_ram_gib <- function() {
  tryCatch({
    if (Sys.info()[["sysname"]] == "Darwin") {
      as.numeric(system("sysctl -n hw.memsize", intern = TRUE)) / 1024^3
    } else {
      line <- grep("^MemTotal:", readLines("/proc/meminfo"), value = TRUE)
      as.numeric(strsplit(trimws(line), "\\s+")[[1L]][2L]) / 1024^2
    }
  }, error = function(e) NA_real_)
}

#' Load one or more wide demo-08 CSVs (concatenated), normalising column and
#' metric whitespace. Each row is tagged with its block, "large_p" for a file
#' whose stem contains "large_p" and "base" otherwise.
read_scalability <- function(paths) {
  frames <- lapply(paths, function(path) {
    if (!file.exists(path))
      stop("Input file not found:\n  ", path,
           "\nPass the results CSV path(s) as the first argument(s).",
           call. = FALSE)
    f <- utils::read.csv(path, strip.white = TRUE, stringsAsFactors = FALSE)
    names(f) <- tolower(trimws(names(f)))
    f$block <- if (grepl("large_p", tools::file_path_sans_ext(basename(path)), fixed = TRUE)) "large_p" else "base"
    f
  })
  df <- do.call(rbind, frames)

  required <- c("n", "p", "solver", "metric", "value")
  missing <- setdiff(required, names(df))
  if (length(missing) > 0)
    stop("CSV is missing required columns: ", paste(missing, collapse = ", "),
         "\nFound columns: ", paste(names(df), collapse = ", "), call. = FALSE)
  if (!"is_barrier" %in% names(df)) df$is_barrier <- 0L

  df$solver <- trimws(as.character(df$solver))
  df$metric <- trimws(as.character(df$metric))
  for (col in c("n", "p", "value", "is_barrier"))
    df[[col]] <- as.numeric(df[[col]])
  df
}

#' Wide table: one row per (n, p, solver, is_barrier, block), a column per metric.
scal_pivot <- function(df) {
  ids <- c("n", "p", "solver", "is_barrier", if ("block" %in% names(df)) "block")
  w <- stats::reshape(df[, c(ids, "metric", "value")],
                      idvar = ids, timevar = "metric", direction = "wide")
  names(w) <- sub("^value\\.", "", names(w))
  w[order(w$solver, w$n, w$p), ]
}

#' The p-axis of the dashboard: log by default (p spans decades), tick-marked
#' at the measured p grid with comma labels rotated 45 degrees.
.scal_p_scale <- function(p_values, xscale = SCAL_XSCALE) {
  if (xscale == "log") {
    fmt <- function(b) {
      format(b, big.mark = ",", trim = TRUE,
             scientific = FALSE, drop0trailing = TRUE)
    }
    scale_x_log10(breaks = p_values, labels = fmt,
                  guide = guide_axis(angle = 45),
                  limits = c(min(p_values) / 1.6, max(p_values) * 1.6),
                  expand = expansion(0))
  } else {
    .sweep_x_scale(p_values, xscale, ticks = p_values, force_ticks = TRUE,
                   angle = 45)
  }
}

#' demo-08 scaling dashboard for the n x p memory-mapped benchmark.
#' 2x2: runtime vs p (colour = n, linetype = solver, min-max band), memory vs p
#' (peak RSS per n + X/y backing reference + RAM line), the out-of-core plot
#' (resident peak vs per T-iteration volume with the RAM line; out-of-core
#' points ringed), and FDR control at scale. Panels degrade gracefully
#' (NA-filter) when a metric was not recorded on this platform.
plot_scalability_dashboard <- function(df, title, tfdr = DEFAULT_TFDR,
                                       xscale = SCAL_XSCALE) {
  w <- scal_pivot(df)
  # Large-p points at an (n, p) of the base block (the cap comparison) get their
  # own markers, all other points join the lines
  w_dup <- w[0L, ]
  if ("block" %in% names(w)) {
    dup <- w$block != "base" &
      paste(w$n, w$p) %in% paste(w$n[w$block == "base"], w$p[w$block == "base"])
    w_dup <- w[dup, ]
    w <- w[!dup, ]
  }
  m_vol <- if (M_VOL %in% names(w)) M_VOL else M_VOL_LEGACY
  solvers <- SCAL_SOLVERS[SCAL_SOLVERS %in% unique(w$solver)]
  n_values <- sort(unique(w$n))
  p_values <- sort(unique(w$p))
  n_labels <- sprintf("n = %s", format(n_values, big.mark = ",", trim = TRUE))
  w$n_f <- factor(w$n, levels = n_values, labels = n_labels)
  # viridis per n, with orange instead of its yellow end (hard to read on white)
  n_cols <- viridisLite::viridis(length(n_values))
  n_pos <- if (length(n_values) > 1L) seq(0, 1, length.out = length(n_values)) else 0
  n_cols[n_pos > 0.85] <- "#FF7F0E"
  n_cols <- setNames(n_cols, n_labels)
  ram <- .scal_ram_gib()
  slv_lines <- SCAL_SOLVER_LINES[solvers]
  names(slv_lines) <- scal_solver_label(solvers)
  w$solver_lbl <- factor(scal_solver_label(w$solver),
                         levels = scal_solver_label(solvers))
  w_dup$n_f <- factor(w_dup$n, levels = n_values, labels = n_labels)
  w_dup$solver_lbl <- factor(scal_solver_label(w_dup$solver),
                             levels = scal_solver_label(solvers))
  # Points of the large-p block, drawn with white-filled markers
  large_p <- function(d) rbind(d[d$block != "base", ], w_dup[w_dup$solver %in% unique(d$solver), ])
  has_large_p <- "block" %in% names(w) && (any(w$block != "base") || nrow(w_dup) > 0)

  dash_theme <- .panel_theme() +
    theme(plot.title = element_text(size = 11, face = "bold", hjust = 0.5),
          axis.title = element_text(size = 9.5),
          axis.text = element_text(size = 8))

  # --- (0,0) Runtime vs p: colour = n, linetype = solver; median + band ------
  m_sel <- if (M_SELMEAN %in% names(w)) M_SELMEAN else M_SELMED
  rt <- w[!is.na(w[[m_sel]]), ]
  band <- rt[rt$solver == solvers[1L] &
               !is.na(rt[[M_SELMIN]]) & !is.na(rt[[M_SELMAX]]), ]
  p_rt <- ggplot(rt, aes(x = p, y = .data[[m_sel]], color = n_f,
                         linetype = solver_lbl,
                         group = interaction(n_f, solver_lbl)))
  if (nrow(band) > 0)
    p_rt <- p_rt + geom_ribbon(
      data = band,
      aes(x = p, ymin = .data[[M_SELMIN]], ymax = .data[[M_SELMAX]],
          fill = n_f, group = n_f),
      alpha = 0.12, colour = NA, inherit.aes = FALSE, show.legend = FALSE
    )
  p_rt <- p_rt +
    geom_line(linewidth = 0.7) +
    geom_point(size = 1.4, shape = 16, show.legend = FALSE)
  if (has_large_p) {
    lp_rt <- large_p(rt)
    lp_rt <- lp_rt[!is.na(lp_rt[[m_sel]]), ]
    p_rt <- p_rt + geom_point(data = lp_rt, aes(x = p, y = .data[[m_sel]], color = n_f),
                              shape = 21, fill = "white", size = 1.8, stroke = 0.8,
                              inherit.aes = FALSE, show.legend = FALSE)
  }
  p_rt <- p_rt +
    scale_color_manual(name = NULL, values = n_cols) +
    scale_fill_manual(values = n_cols, guide = "none") +
    scale_linetype_manual(name = NULL, values = slv_lines) +
    scale_y_log10() +
    .scal_p_scale(p_values, xscale) +
    labs(title = "Runtime scaling",
         subtitle = if (has_large_p) "white-filled points = large-p block" else NULL,
         x = "Number of features p",
         y = sprintf("select() per trial [s]  (%s, band = min-max)",
                     if (m_sel == M_SELMEAN) "mean" else "median")) +
    dash_theme + .legend_inside(0.03, 0.97) +
    theme(plot.subtitle = element_text(size = 8, hjust = 0.5)) +
    guides(color = guide_legend(order = 1, ncol = 2),
           linetype = guide_legend(order = 2))

  # --- (0,1) Memory vs p: per-point resident and heap per n + X/y + RAM -----
  # (PeakRSS is the running maximum of the whole process, no per-point value)
  mem <- w[w$solver == solvers[1L] & !is.na(w[[M_RSS]]), ]
  ref <- stats::aggregate(w[[M_XY]], by = list(p = w$p),
                          FUN = mean, na.rm = TRUE)
  names(ref)[2L] <- "xy"
  ref <- ref[!is.na(ref$xy), ]
  lt_vals <- c("resident after select()" = "solid", "heap after select()" = "dotted",
               "X/y mmap backing" = "dashed")
  p_mem <- ggplot(mem, aes(x = p, y = .data[[M_RSS]], color = n_f)) +
    geom_line(aes(linetype = "resident after select()"), linewidth = 0.8) +
    geom_point(size = 1.8, shape = 16)
  if (M_FOOT %in% names(mem) && any(!is.na(mem[[M_FOOT]])))
    p_mem <- p_mem +
      geom_line(aes(y = .data[[M_FOOT]], linetype = "heap after select()"), linewidth = 0.5) +
      geom_point(aes(y = .data[[M_FOOT]]), size = 1.0, shape = 15)
  if (has_large_p) {
    lp_mem <- large_p(mem)
    lp_mem <- lp_mem[!is.na(lp_mem[[M_RSS]]), ]
    p_mem <- p_mem + geom_point(data = lp_mem, aes(x = p, y = .data[[M_RSS]], color = n_f),
                                shape = 21, fill = "white", size = 2.2, stroke = 0.9,
                                inherit.aes = FALSE, show.legend = FALSE)
  }
  if (nrow(ref) > 0)
    p_mem <- p_mem +
      geom_line(data = ref, aes(x = p, y = xy, linetype = "X/y mmap backing"),
                color = "grey35", linewidth = 0.7, inherit.aes = FALSE) +
      geom_point(data = ref, aes(x = p, y = xy), shape = 23, size = 2.0,
                 color = "grey35", fill = "white", inherit.aes = FALSE)
  if (is.finite(ram) && ram > 0) {
    ram_lbl <- sprintf("physical RAM = %.0f GiB", ram)
    p_mem <- p_mem +
      geom_hline(aes(yintercept = ram * 1024.0, linetype = ram_lbl),
                 color = "#DC143C", linewidth = 0.6)
    lt_vals <- c(lt_vals, setNames("dotdash", ram_lbl))
  }
  p_mem <- p_mem +
    scale_color_manual(name = NULL, values = n_cols) +
    scale_linetype_manual(name = NULL, values = lt_vals) +
    scale_y_log10() +
    .scal_p_scale(p_values, xscale) +
    labs(title = "Memory scaling: resident follows X/y, the heap stays small",
         x = "Number of features p", y = "Memory after select() [MiB]") +
    dash_theme + .legend_inside(0.03, 0.97) +
    guides(color = guide_legend(order = 1), linetype = guide_legend(order = 2))

  # --- (1,0) Out-of-core: heap per point vs volume, the run's peak as a line
  oc_df <- rbind(w, w_dup)
  # The heap where it was measured (none from R on macOS), else the resident
  has_col <- function(col) col %in% names(oc_df) && any(!is.na(oc_df[[col]]))
  m_oc <- if (has_col(M_FOOT)) M_FOOT else if (has_col(M_RSS)) M_RSS else M_PEAKRSS
  oc_df <- oc_df[!is.na(oc_df[[m_vol]]) & !is.na(oc_df[[m_oc]]), ]
  barrier <- oc_df[oc_df$is_barrier == 1, ]
  slv_shapes <- SCAL_SOLVER_SHAPES[solvers]
  names(slv_shapes) <- scal_solver_label(solvers)
  p_oc <- ggplot(oc_df, aes(x = .data[[m_vol]],
                            y = .data[[m_oc]] / 1024.0,
                            color = n_f, shape = solver_lbl)) +
    geom_point(size = 2.2, stroke = 0.8, fill = NA)
  if (nrow(barrier) > 0)
    p_oc <- p_oc + geom_point(
      data = barrier,
      aes(x = .data[[m_vol]], y = .data[[m_oc]] / 1024.0),
      shape = 21, size = 5.5, color = "#DC143C", fill = NA, stroke = 1.0,
      inherit.aes = FALSE, show.legend = FALSE
    )
  oc_lt <- character(0)
  if (m_oc != M_PEAKRSS && M_PEAKRSS %in% names(oc_df)) {
    peak <- max(oc_df[[M_PEAKRSS]], na.rm = TRUE) / 1024.0
    peak_lbl <- sprintf("process peak RSS of the run = %.0f GiB", peak)
    p_oc <- p_oc + geom_hline(aes(yintercept = peak, linetype = peak_lbl),
                              color = "grey25", linewidth = 0.6)
    oc_lt <- setNames("dashed", peak_lbl)
  }
  if (is.finite(ram) && ram > 0) {
    ram_lbl <- sprintf("physical RAM = %.0f GiB", ram)
    p_oc <- p_oc +
      geom_vline(aes(xintercept = ram, linetype = ram_lbl),
                 color = "#DC143C", linewidth = 0.7)
    if (m_oc != M_PEAKRSS)
      p_oc <- p_oc + geom_hline(yintercept = ram, color = "#DC143C",
                                linewidth = 0.7, linetype = "dotdash")
    oc_lt <- c(oc_lt, setNames("dotdash", ram_lbl))
  }
  if (length(oc_lt) > 0)
    p_oc <- p_oc + scale_linetype_manual(name = NULL, values = oc_lt)
  p_oc <- p_oc +
    scale_color_manual(name = NULL, values = n_cols) +
    scale_shape_manual(name = NULL, values = slv_shapes) +
    scale_x_log10(labels = function(b) {
      format(b, trim = TRUE,
             scientific = FALSE,
             drop0trailing = TRUE)
    }) +
    labs(title = "Out-of-core: volume crosses RAM, memory stays bounded",
         subtitle = if (nrow(barrier) > 0)
           "ringed = out-of-core point (volume > RAM)" else NULL,
         x = if (m_vol == M_VOL) "Volume per T-iteration X + K · L · X [GiB]  (log)"
         else "Total design volume [X|D] [GiB]  (log)",
         y = switch(m_oc, Foot_MiB = "Heap after select() [GiB]",
                    RSS_MiB = "Resident after select() [GiB]", "Process peak RSS [GiB]")) +
    dash_theme + .legend_inside(0.03, 0.97) +
    theme(plot.subtitle = element_text(size = 8, hjust = 0.5,
                                       colour = "#DC143C")) +
    guides(color = guide_legend(order = 1), shape = guide_legend(order = 2),
           linetype = guide_legend(order = 3))

  if (m_oc != M_PEAKRSS) p_oc <- p_oc + scale_y_log10()

  # --- (1,1) FDR control at scale: per solver, mean over n ------------------
  fdr <- w[!is.na(w[[M_FDR]]), ]
  g <- stats::aggregate(fdr[[M_FDR]],
                        by = list(solver = fdr$solver, p = fdr$p), FUN = mean)
  names(g)[3L] <- "FDR"
  g$solver_lbl <- factor(scal_solver_label(g$solver),
                         levels = scal_solver_label(solvers))
  fdr_top <- max(if (tfdr > 0) tfdr * 1.5 else 0.1, max(g$FDR) * 1.15)
  slv_cols <- SCAL_SOLVER_COLORS[solvers]
  names(slv_cols) <- scal_solver_label(solvers)
  p_fdr <- ggplot(g, aes(x = p, y = FDR, color = solver_lbl,
                         shape = solver_lbl))
  if (tfdr > 0) p_fdr <- p_fdr + .tfdr_layers(tfdr)
  p_fdr <- p_fdr +
    geom_line(linewidth = 0.8) +
    geom_point(size = 2.4, fill = NA, stroke = 0.9) +
    scale_color_manual(name = NULL, values = slv_cols) +
    scale_shape_manual(name = NULL, values = slv_shapes) +
    .scal_p_scale(p_values, xscale) +
    .rate_y_scale("FDR", fdr_top) +
    coord_cartesian(ylim = c(0.0, fdr_top)) +
    labs(title = "FDR control at scale",
         x = "Number of features p",
         y = "False discovery rate  (mean over n)") +
    dash_theme + .legend_inside(0.03, 0.97) +
    guides(linetype = guide_legend(order = 1),
           color = guide_legend(order = 2), shape = guide_legend(order = 2))

  ((p_rt | p_mem) / (p_oc | p_fdr)) +
    plot_annotation(title = title,
                    theme = theme(plot.title = element_text(
                      size = 14, face = "bold", hjust = 0.5
                    )))
}


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

.split_formats <- function(x) unlist(strsplit(x, ","))

#' Greedy nargs="+"-style consumption: gather tokens until the next --flag.
.consume_multi <- function(argv, i) {
  vals <- character(0)
  while (i + 1L <= length(argv) && !startsWith(argv[[i + 1L]], "--")) {
    i <- i + 1L
    vals <- c(vals, argv[[i]])
  }
  list(values = vals, i = i)
}

.parse_default_args <- function(argv) {
  args <- list(csv = character(0), title = NULL,
               legend_title = DEFAULT_LEGEND_TITLE, outdir = NULL,
               stem = NULL, formats = c("png", "pdf"), tfdr = DEFAULT_TFDR,
               groups = 2L, xscale = DEFAULT_XSCALE, no_grouped = FALSE,
               overlay = NULL)
  i <- 1L
  while (i <= length(argv)) {
    a <- argv[[i]]
    flag <- function() {
      i <<- i + 1L
      argv[[i]]
    }
    switch(a,
           "--title"        = {
             args$title <- flag()
           },
           "--legend-title" = {
             args$legend_title <- flag()
           },
           "--outdir"       = {
             args$outdir <- flag()
           },
           "--stem"         = {
             args$stem <- flag()
           },
           "--formats"      = {
             args$formats <- .split_formats(flag())
           },
           "--tfdr"         = {
             args$tfdr <- as.numeric(flag())
           },
           "--groups"       = {
             args$groups <- as.integer(flag())
           },
           "--xscale"       = {
             args$xscale <- match.arg(flag(), XSCALES)
           },
           "--no-grouped"   = {
             args$no_grouped <- TRUE
           },
           "--overlay"      = {
             r <- .consume_multi(argv, i)
             args$overlay <- r$values
             i <- r$i
           },
           "--no-plotly"    = { },  # accepted for Python-CLI parity; no HTML here
           "--plotly-cdn"   = { },
           {
             if (startsWith(a, "--")) {
               cat(sprintf("note: ignoring unknown flag %s\n", a))
             } else {
               args$csv <- c(args$csv, a)
             }
           })
    i <- i + 1L
  }
  args
}

#' Default mode: per-CSV overview + grouped figure. With --overlay, the
#' 'solver' column of each CSV is rewritten to the paired scenario label and
#' the frames are overlaid on one FDR | TPR pair (demo 07's headline figure).
main_fdr_tpr <- function(argv) {
  args <- .parse_default_args(argv)
  if (length(args$csv) == 0)
    stop("Usage: trex_plt_utils.R <results.csv> [--title T] ",
         "[--legend-title T] [--tfdr F] [--xscale linear|sqrt|log] ",
         "[--outdir DIR] [--stem S] [--formats png,pdf] [--groups N] ",
         "[--no-grouped] [--overlay L1 L2 (with matching CSVs)]",
         call. = FALSE)

  if (!is.null(args$overlay)) {
    if (length(args$csv) != length(args$overlay))
      stop("--overlay needs exactly one label per input CSV (got ",
           length(args$overlay), " labels for ", length(args$csv), " CSVs).",
           call. = FALSE)
    frames <- Map(function(csv, label) {
      df <- read_mc_results(csv)
      df$solver <- label
      df
    }, args$csv, args$overlay)
    df <- do.call(rbind, c(frames, make.row.names = FALSE))
    df$solver <- factor(df$solver, levels = args$overlay)  # file order
    if (is.null(args$stem))
      stop("--overlay requires --stem (the merged figure is not tied to one ",
           "CSV's name).", call. = FALSE)
  } else {
    if (length(args$csv) != 1L)
      stop("Default mode takes exactly one CSV (got ", length(args$csv),
           "); pass --overlay labels to merge several.", call. = FALSE)
    df <- read_mc_results(args$csv[[1L]])
  }

  csv_path <- args$csv[[1L]]
  csv_stem <- sub("[.]csv$", "", basename(csv_path))
  title <- if (!is.null(args$title)) args$title else sprintf("T-Rex Monte Carlo simulation (%s)", csv_stem)
  outdir <- if (!is.null(args$outdir)) args$outdir else default_plots_dir(csv_path)
  stem <- if (!is.null(args$stem)) args$stem else paste0(csv_stem, STEM_SUFFIX)

  # 1. Combined overview (FDR | TPR).
  fig <- plot_fdr_tpr_vs_snr(df, title, tfdr = args$tfdr,
                             legend_title = args$legend_title,
                             xscale = args$xscale)
  save_figure(fig, outdir, stem, args$formats, width = 13.0, height = 5.6)

  # 2. De-cluttered grouped view.
  if (!args$no_grouped) {
    grouped <- plot_fdr_tpr_vs_snr_grouped(
      df, title, tfdr = args$tfdr, legend_title = args$legend_title,
      n_groups = args$groups, xscale = args$xscale
    )
    save_figure(grouped, outdir, paste0(stem, "_grouped"), args$formats,
                width = 6.8 * args$groups, height = 9.0)
  }
  invisible(0L)
}

#' `grid` mode: cross-solver comparison grid from several per-solver CSVs.
#' --labels and --csvs are parallel lists (one column per pair, in order); a
#' listed CSV that does not exist yet is skipped with a note.
main_grid <- function(argv) {
  args <- list(labels = character(0), csvs = character(0),
               title = "T-Rex cross-solver comparison",
               legend_title = DEFAULT_LEGEND_TITLE, stem = NULL,
               outdir = NULL, formats = c("png", "pdf"),
               tfdr = DEFAULT_TFDR, xscale = DEFAULT_XSCALE)
  i <- 1L
  while (i <= length(argv)) {
    a <- argv[[i]]
    flag <- function() {
      i <<- i + 1L
      argv[[i]]
    }
    switch(a,
           "--labels"       = {
             r <- .consume_multi(argv, i)
             args$labels <- r$values
             i <- r$i
           },
           "--csvs"         = {
             r <- .consume_multi(argv, i)
             args$csvs <- r$values
             i <- r$i
           },
           "--title"        = {
             args$title <- flag()
           },
           "--legend-title" = {
             args$legend_title <- flag()
           },
           "--stem"         = {
             args$stem <- flag()
           },
           "--outdir"       = {
             args$outdir <- flag()
           },
           "--formats"      = {
             args$formats <- .split_formats(flag())
           },
           "--tfdr"         = {
             args$tfdr <- as.numeric(flag())
           },
           "--xscale"       = {
             args$xscale <- match.arg(flag(), XSCALES)
           },
           { })  # ignore flags meant for the default mode (--no-grouped, ...)
    i <- i + 1L
  }
  if (is.null(args$stem) || is.null(args$outdir))
    stop("grid mode requires --stem and --outdir.", call. = FALSE)
  if (length(args$labels) != length(args$csvs))
    stop("--labels (", length(args$labels), ") and --csvs (",
         length(args$csvs), ") must have the same length.", call. = FALSE)

  frames <- list()
  for (k in seq_along(args$labels)) {
    if (!file.exists(args$csvs[[k]])) {
      cat(sprintf("note: skipping %s - CSV not found:\n  %s\n",
                  args$labels[[k]], args$csvs[[k]]))
      next
    }
    frames[[args$labels[[k]]]] <- read_mc_results(args$csvs[[k]])
  }
  if (length(frames) == 0)
    stop("No input CSVs found - run the demo first, then re-run this plotter.",
         call. = FALSE)

  fig <- plot_comparison_grid(frames, args$title,
                              legend_title = args$legend_title,
                              tfdr = args$tfdr, xscale = args$xscale)
  save_figure(fig, args$outdir, args$stem, args$formats,
              width = 4.9 * length(frames) + 2.4, height = 8.6)
  invisible(0L)
}

#' `scalability` mode: demo-08 runtime / memory / quality dashboard.
main_scalability <- function(argv) {
  args <- list(csv = character(0),
               title = "T-Rex scalability on the memory-mapped pipeline",
               stem = NULL, outdir = NULL, formats = c("png", "pdf"),
               tfdr = DEFAULT_TFDR, xscale = SCAL_XSCALE)
  i <- 1L
  while (i <= length(argv)) {
    a <- argv[[i]]
    flag <- function() {
      i <<- i + 1L
      argv[[i]]
    }
    switch(a,
           "--title"   = {
             args$title <- flag()
           },
           "--stem"    = {
             args$stem <- flag()
           },
           "--outdir"  = {
             args$outdir <- flag()
           },
           "--formats" = {
             args$formats <- .split_formats(flag())
           },
           "--tfdr"    = {
             args$tfdr <- as.numeric(flag())
           },
           "--xscale"  = {
             args$xscale <- match.arg(flag(), XSCALES)
           },
           {
             if (!startsWith(a, "--")) args$csv <- c(args$csv, a)
           })  # unknown flags meant for other modes are ignored
    i <- i + 1L
  }
  if (length(args$csv) == 0L || is.null(args$stem) || is.null(args$outdir))
    stop("Usage: trex_plt_utils.R scalability <wide_csv> [<wide_csv> ...] ",
         "--stem S --outdir DIR [--title T] [--tfdr F] [--formats png,pdf] ",
         "[--xscale linear|sqrt|log]", call. = FALSE)

  df <- read_scalability(args$csv)
  fig <- plot_scalability_dashboard(df, args$title, tfdr = args$tfdr,
                                    xscale = args$xscale)
  save_figure(fig, args$outdir, args$stem, args$formats,
              width = 13.5, height = 10.0)
  invisible(0L)
}

#' `sweep` mode: FDR / TPR grid of a long sweep CSV (demos 09 and 10).
main_sweep <- function(argv) {
  args <- list(csv = character(0), x = NULL, lines = NULL, diagonal = FALSE,
               filter = character(0), title = "T-Rex sweep", stem = NULL, outdir = NULL,
               formats = c("png", "pdf"), tfdr = DEFAULT_TFDR, xscale = DEFAULT_XSCALE)
  i <- 1L
  while (i <= length(argv)) {
    a <- argv[[i]]
    flag <- function() {
      i <<- i + 1L
      argv[[i]]
    }
    switch(a,
           "--x"        = {
             args$x <- flag()
           },
           "--lines"    = {
             args$lines <- flag()
           },
           "--diagonal" = {
             args$diagonal <- TRUE
           },
           "--filter"   = {
             r <- .consume_multi(argv, i)
             args$filter <- c(args$filter, r$values)
             i <- r$i
           },
           "--title"    = {
             args$title <- flag()
           },
           "--stem"     = {
             args$stem <- flag()
           },
           "--outdir"   = {
             args$outdir <- flag()
           },
           "--formats"  = {
             args$formats <- .split_formats(flag())
           },
           "--tfdr"     = {
             args$tfdr <- as.numeric(flag())
           },
           "--xscale"   = {
             args$xscale <- match.arg(flag(), XSCALES)
           },
           {
             if (!startsWith(a, "--")) args$csv <- c(args$csv, a)
           })  # unknown flags meant for other modes are ignored
    i <- i + 1L
  }
  if (length(args$csv) != 1L || is.null(args$x) || is.null(args$lines) || is.null(args$stem))
    stop("Usage: trex_plt_utils.R sweep <csv> --x COL --lines COL --stem S ",
         "[--diagonal] [--filter col=val ...] [--title T] [--outdir DIR] [--tfdr F] ",
         "[--formats png,pdf] [--xscale linear|sqrt|log]", call. = FALSE)

  csv <- normalizePath(args$csv)
  df <- read_sweep(csv)
  for (f in args$filter) {
    kv <- strsplit(f, "=", fixed = TRUE)[[1L]]
    df <- df[abs(suppressWarnings(as.numeric(df[[kv[[1L]]]])) - as.numeric(kv[[2L]])) < 1e-9, ]
  }
  fig <- plot_sweep_grid(df, args$x, args$lines, args$title, diagonal = args$diagonal,
                         tfdr = args$tfdr, xscale = args$xscale)
  outdir <- if (is.null(args$outdir)) default_plots_dir(csv) else args$outdir
  save_figure(fig, outdir, args$stem, args$formats,
              width = 5.0 * length(unique(df$solver)) + 2.4, height = 8.6)
  invisible(0L)
}

#' Dispatch on the first token: `grid` / `scalability` / `sweep` sub-modes,
#' else the default per-CSV FDR/TPR mode.
main <- function() {
  argv <- commandArgs(trailingOnly = TRUE)
  if (length(argv) > 0 && argv[[1L]] == "grid") {
    main_grid(argv[-1L])
  } else if (length(argv) > 0 && argv[[1L]] == "scalability") {
    main_scalability(argv[-1L])
  } else if (length(argv) > 0 && argv[[1L]] == "sweep") {
    main_sweep(argv[-1L])
  } else {
    main_fdr_tpr(argv)
  }
}

if (sys.nframe() == 0L) main()
