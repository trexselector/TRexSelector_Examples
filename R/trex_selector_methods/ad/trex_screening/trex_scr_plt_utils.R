#!/usr/bin/env Rscript
# ==============================================================================
# trex_scr_plt_utils.R
# ==============================================================================
#
# Plotting utilities for the Screen-TRex demonstration suite — ggplot2 port of
# the Python plotter of the same name (trex_scr_plt_utils.py), kept visually
# aligned with it: tab10 method colours, hollow per-method markers, ONE legend
# outside the panels, PNG + PDF output into the sibling
# simulation_results/plots/ folder. Static ggplot2 output only — no
# plotly/HTML figures (divergence from nothing: the Python plotter is static
# matplotlib as well; noted for the record).
#
# Reads the tidy result CSVs written by the demos and auto-detects two shapes
# from their columns (same rule as the Python plotter):
#
#   * Sweep   — method,metric,<SNR|rho>,value with metrics FDR / TPR /
#               Est. FDR (demos 01-03, 06). Rendered as a TPR panel and an
#               FDR panel vs. the sweep variable; the FDR panel overlays the
#               procedure's own ESTIMATED FDR as dashed lines and marks the
#               tFDR target as a dotted reference line (the tFDR line is an
#               R-side addition over the Python figure).
#   * Biobank — method,metric,snr,value with an additional Usage metric
#               (demos 04-05, "Algorithm 1" adaptive routing). Rendered as
#               three panels: TPR, FDR (realized solid / estimated dashed,
#               with the routing window [0.05, 0.15] shaded), and Usage %.
#
# Usage:
#   Rscript trex_scr_plt_utils.R <results.csv> [--outdir DIR] [--title T]
#                                [--tfdr F] [--formats png,pdf] [--stem S]
#                                [--xscale linear|sqrt]
#
# --xscale sqrt switches the x-axis to a square-root scale, which spreads a
# sweep grid that is dense near zero while keeping 0 on-axis (default: linear).
# Divergences from the Python CLI: --formats is comma-separated (png,pdf) and
# --tfdr controls the dotted target-FDR reference of the sweep figure.
# ==============================================================================

suppressMessages({
  library(ggplot2)
  library(patchwork)
})

# Target FDR of the screening demos (target_FDR_trex = 0.10); 0 disables.
DEFAULT_TFDR <- 0.1

# Routing FDR window of the biobank workflow (BiobankScreenTRexControl).
ROUTING_FDR_WINDOW <- c(0.05, 0.15)

# matplotlib "tab10" — the colour convention shared by every suite's figures.
TAB10 <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")

# Hollow per-method markers (o, s, ^, D, v, P) — overlapping curves stay
# distinguishable. ggplot2 shapes 21-25 honour fill; 3 is the plus.
METHOD_SHAPES <- c(21L, 22L, 24L, 23L, 25L, 3L)


# ------------------------------------------------------------------------------
# Shared helpers (house conventions)
# ------------------------------------------------------------------------------

save_figure <- function(plot, outdir, stem, formats = c("png", "pdf"),
                        width = 12.4, height = 4.8) {
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  for (ext in formats) {
    path <- file.path(outdir, paste0(stem, ".", ext))
    if (ext == "pdf" && isTRUE(capabilities("cairo"))) {
      # cairo_pdf embeds UTF-8 (the em dashes in the demo titles); the base
      # pdf device would substitute '-' for them.
      ggsave(path, plot = plot, width = width, height = height,
             dpi = 300, bg = "white", device = grDevices::cairo_pdf)
    } else {
      ggsave(path, plot = plot, width = width, height = height,
             dpi = 300, bg = "white")
    }
    cat(sprintf("Wrote: %s\n", path))
  }
}

#' `.../simulation_results/data/x.csv` -> `.../simulation_results/plots`.
default_plots_dir <- function(csv_path) {
  parent <- dirname(csv_path)
  if (basename(parent) == "data") file.path(dirname(parent), "plots") else file.path(parent, "plots")
}

load_csv <- function(path) {
  if (!file.exists(path)) stop("Input CSV not found: ", path)
  df <- utils::read.csv(path, strip.white = TRUE, stringsAsFactors = FALSE)
  names(df) <- trimws(names(df))
  df
}

#' The sweep-variable column is whatever is not method/metric/value.
.sweep_column <- function(df) {
  extras <- setdiff(names(df), c("method", "metric", "value"))
  if (length(extras) != 1L)
    stop("Expected one sweep column, got: ", paste(extras, collapse = ", "))
  extras
}

#' Human-readable axis label for the sweep variable.
.axis_label <- function(sweep) {
  if (tolower(sweep) == "snr") return("SNR")
  if (tolower(sweep) == "rho") return(expression(rho))
  sweep
}

#' Round-number tick step: the smallest of {1, 2, 2.5, 5, 10} x 10^k that
#' keeps <= max_intervals major intervals (mirrors MaxNLocator's steps).
.nice_step <- function(span, max_intervals = 6) {
  if (span <= 0) return(1)
  raw <- span / max_intervals
  mag <- 10^floor(log10(raw))
  for (m in c(1, 2, 2.5, 5, 10)) {
    s <- m * mag
    if (s + 1e-12 >= raw) return(s)
  }
  10 * mag
}

#' Linear (default) or square-root x-axis with round-number major ticks and
#' half-step minor gridlines, ~4% padding. With xscale = "sqrt" the padding is
#' computed in sqrt space so the margin is visually the same, and the lower
#' limit stays at the data (keeps 0 on-axis) — mirrors the Python helper
#' apply_linear_minor_xaxis().
.sweep_x_scale <- function(values, xscale = "linear") {
  lo <- min(values)
  hi <- max(values)
  step <- .nice_step(hi - lo)
  breaks <- seq(ceiling(lo / step) * step, hi + 1e-12, by = step)
  minor <- seq(ceiling(lo / (step / 2)) * step / 2, hi + 1e-12, by = step / 2)
  if (identical(xscale, "sqrt")) {
    span_sqrt <- sqrt(hi) - sqrt(max(lo, 0))
    pad <- if (span_sqrt > 0) (span_sqrt * 0.04)^2 else 0.5
    scale_x_continuous(
      transform = "sqrt", breaks = breaks, minor_breaks = minor,
      limits = c(lo, hi + pad), expand = expansion(0)
    )
  } else {
    span <- hi - lo
    pad <- if (span > 0) span * 0.04 else 0.5
    scale_x_continuous(
      breaks = breaks, minor_breaks = minor,
      limits = c(lo - pad, hi + pad), expand = expansion(0)
    )
  }
}

#' The shared panel look: light major/minor grid, bold panel title.
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

#' Method colour/fill/shape scales over a fixed ordered set (CSV order — the
#' Python plotter keeps first-appearance order, no reordering).
.method_scales <- function(methods, group_label = "Method") {
  cols <- setNames(TAB10[(seq_along(methods) - 1L) %% length(TAB10) + 1L],
                   methods)
  shps <- setNames(METHOD_SHAPES[(seq_along(methods) - 1L) %%
                                   length(METHOD_SHAPES) + 1L], methods)
  list(
    scale_color_manual(name = group_label, values = cols),
    scale_shape_manual(name = group_label, values = shps)
  )
}

#' Subset one metric, method factor in first-appearance order.
.metric_df <- function(df, sweep, methods, metric) {
  sub <- df[df$metric == metric, , drop = FALSE]
  sub$method <- factor(sub$method, levels = methods)
  sub[order(sub$method, sub[[sweep]]), , drop = FALSE]
}


# ------------------------------------------------------------------------------
# Panels
# ------------------------------------------------------------------------------

#' Solid line + hollow markers per method for one metric (TPR / Usage panels).
.lines_panel <- function(df, sweep, methods, metric, panel_title,
                         xs, xscale, ylim = NULL, percent = FALSE,
                         method_legend = TRUE) {
  sub <- .metric_df(df, sweep, methods, metric)
  if (percent) sub$value <- 100 * sub$value
  guide <- if (method_legend) guide_legend(order = 1) else "none"
  p <- ggplot(sub, aes(x = .data[[sweep]], y = value,
                       color = method, shape = method)) +
    geom_line(linewidth = 0.6) +
    geom_point(size = 2.4, fill = NA, stroke = 0.9) +
    .method_scales(methods) +
    guides(color = guide, shape = guide) +
    labs(title = panel_title, x = .axis_label(sweep), y = NULL) +
    .sweep_x_scale(xs, xscale) +
    .panel_theme()
  if (!is.null(ylim)) p <- p + coord_cartesian(ylim = ylim)
  p
}

#' FDR panel: realized FDR solid with markers, estimated FDR dashed without
#' markers; optional dotted tFDR reference and shaded routing window.
.fdr_panel <- function(df, sweep, methods, xs, xscale,
                       tfdr = 0, routing_window = NULL) {
  realized <- .metric_df(df, sweep, methods, "FDR")
  estimated <- .metric_df(df, sweep, methods, "Est. FDR")
  has_est <- nrow(estimated) > 0

  lt_values <- c("realized FDR" = "solid")
  lt_colors <- c("grey30")

  p <- ggplot(realized, aes(x = .data[[sweep]], y = value,
                            color = method, shape = method))

  if (!is.null(routing_window)) {
    win <- data.frame(lo = routing_window[1], hi = routing_window[2])
    win_lbl <- sprintf("routing window [%g, %g]", win$lo, win$hi)
    p <- p +
      geom_rect(data = win, aes(xmin = -Inf, xmax = Inf,
                                ymin = lo, ymax = hi, fill = win_lbl),
                alpha = 0.6, inherit.aes = FALSE) +
      scale_fill_manual(
        name = NULL, values = setNames("grey75", win_lbl),
        guide = guide_legend(order = 3,
                             override.aes = list(alpha = 0.6))
      )
  }

  p <- p +
    geom_line(aes(linetype = "realized FDR"), linewidth = 0.6) +
    geom_point(size = 2.4, fill = NA, stroke = 0.9)

  if (has_est) {
    p <- p + geom_line(data = estimated,
                       aes(linetype = "estimated FDR"), linewidth = 0.6)
    lt_values <- c(lt_values, "estimated FDR" = "dashed")
    lt_colors <- c(lt_colors, "grey30")
  }
  if (tfdr > 0) {
    tfdr_lbl <- sprintf("tFDR = %g", tfdr)
    p <- p + geom_hline(aes(yintercept = tfdr, linetype = tfdr_lbl),
                        color = "black", linewidth = 0.8)
    lt_values <- c(lt_values, setNames("dotted", tfdr_lbl))
    lt_colors <- c(lt_colors, "black")
  }

  p +
    scale_linetype_manual(
      name = NULL, values = lt_values, breaks = names(lt_values),
      guide = guide_legend(
        order = 2,
        override.aes = list(color = lt_colors, shape = NA)
      )
    ) +
    .method_scales(methods) +
    # The method legend is carried by the TPR panel alone so patchwork's
    # guide collection shows exactly ONE Method box (this panel's colour
    # keys carry linetype info and would not dedupe against it).
    guides(color = "none", shape = "none") +
    labs(title = "FDR: realized (solid) vs. estimated (dashed)",
         x = .axis_label(sweep), y = NULL) +
    .sweep_x_scale(xs, xscale) +
    .panel_theme()
}

.collect <- function(panels, title) {
  panels +
    plot_layout(guides = "collect") +
    plot_annotation(title = title,
                    theme = theme(plot.title = element_text(
                      size = 14, face = "bold", hjust = 0.5
                    )))
}


# ------------------------------------------------------------------------------
# Sweep figure (demos 01-03, 06): TPR + FDR/Est.FDR panels
# ------------------------------------------------------------------------------

plot_sweep <- function(df, title, tfdr = DEFAULT_TFDR, xscale = "linear") {
  sweep <- .sweep_column(df)
  xs <- sort(unique(df[[sweep]]))
  methods <- unique(df$method)

  p_tpr <- .lines_panel(df, sweep, methods, "TPR",
                        "TPR (true positive rate)", xs, xscale,
                        ylim = c(0, 1.05))
  p_fdr <- .fdr_panel(df, sweep, methods, xs, xscale, tfdr = tfdr)
  .collect(p_tpr | p_fdr, title)
}


# ------------------------------------------------------------------------------
# Biobank figure (demos 04-05): TPR + FDR + Usage panels
# ------------------------------------------------------------------------------

plot_biobank <- function(df, title, xscale = "linear") {
  sweep <- .sweep_column(df)
  xs <- sort(unique(df[[sweep]]))
  methods <- unique(df$method)

  p_tpr <- .lines_panel(df, sweep, methods, "TPR",
                        "TPR (conditional on routing)", xs, xscale,
                        ylim = c(0, 1.05))
  p_fdr <- .fdr_panel(df, sweep, methods, xs, xscale,
                      routing_window = ROUTING_FDR_WINDOW)
  p_use <- .lines_panel(df, sweep, methods, "Usage",
                        "Usage (% of screenings routed)", xs, xscale,
                        ylim = c(0, 105), percent = TRUE,
                        method_legend = FALSE)
  .collect(p_tpr | p_fdr | p_use, title)
}


# ------------------------------------------------------------------------------
# Dispatch + CLI
# ------------------------------------------------------------------------------

detect_kind <- function(df) {
  if (!all(c("method", "metric", "value") %in% names(df)))
    stop("Unrecognized CSV schema: ", paste(sort(names(df)), collapse = ", "))
  if ("Usage" %in% df$metric) "biobank" else "sweep"
}

.parse_args <- function(argv) {
  args <- list(csv = character(0), outdir = NULL, title = NULL,
               tfdr = DEFAULT_TFDR, formats = c("png", "pdf"),
               stem = NULL, xscale = "linear")
  i <- 1L
  while (i <= length(argv)) {
    a <- argv[[i]]
    flag <- function() {
      i <<- i + 1L
      argv[[i]]
    }
    switch(a,
           "--outdir"  = {
             args$outdir <- flag()
           },
           "--title"   = {
             args$title <- flag()
           },
           "--tfdr"    = {
             args$tfdr <- as.numeric(flag())
           },
           "--formats" = {
             args$formats <- strsplit(flag(), ",")[[1]]
           },
           "--stem"    = {
             args$stem <- flag()
           },
           "--xscale"  = {
             args$xscale <- flag()
           },
           {
             args$csv <- c(args$csv, a)
           })
    i <- i + 1L
  }
  if (length(args$csv) != 1L)
    stop("Usage: trex_scr_plt_utils.R <results.csv> [--outdir DIR] ",
         "[--title T] [--tfdr F] [--formats png,pdf] [--stem S] ",
         "[--xscale linear|sqrt]")
  if (!args$xscale %in% c("linear", "sqrt"))
    stop("--xscale must be 'linear' or 'sqrt', got: ", args$xscale)
  args
}

main <- function() {
  args <- .parse_args(commandArgs(trailingOnly = TRUE))
  csv_path <- args$csv[[1]]
  df <- load_csv(csv_path)

  outdir <- if (is.null(args$outdir)) default_plots_dir(csv_path) else args$outdir
  stem <- if (is.null(args$stem)) sub("[.]csv$", "", basename(csv_path)) else args$stem
  title <- if (is.null(args$title)) sub("[.]csv$", "", basename(csv_path)) else args$title

  kind <- detect_kind(df)
  if (kind == "biobank") {
    fig <- plot_biobank(df, title, args$xscale)
    save_figure(fig, outdir, stem, args$formats, width = 16.8, height = 4.8)
  } else {
    fig <- plot_sweep(df, title, args$tfdr, args$xscale)
    save_figure(fig, outdir, stem, args$formats, width = 12.4, height = 4.8)
  }
  invisible(0L)
}

if (sys.nframe() == 0L) main()
