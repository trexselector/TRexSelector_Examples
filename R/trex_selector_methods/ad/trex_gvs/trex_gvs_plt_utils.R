#!/usr/bin/env Rscript
# ==============================================================================
# trex_gvs_plt_utils.R
# ==============================================================================
#
# Plotting utilities for the T-Rex+GVS demonstration suite — ggplot2 port of
# the Python plotter of the same name (trex_gvs_plt_utils.py), kept visually
# aligned with it: viridis for TPR, reversed magma for FDR, cyan outlines
# marking realized FDR above the tFDR target, tab10 method colours, hollow
# per-method markers, ONE legend outside the panels, PNG + PDF output into the
# sibling simulation_results/plots/ folder.
#
# Three CSV shapes are auto-detected from their columns:
#
#   * 2-D sweep — method,metric,row_label,col_label,mean,sd (the *_2d.csv
#     files). Rendered as a 2 x 3 heatmap grid: rows = TPR / FDR, columns =
#     the three solvers (TENET / TENET_AUG / TIENET_AUG); row axis = SNR,
#     column axis = the swept correlation (rho or ar_coef). Cell values are
#     annotated with luminance-adaptive text colour.
#
#   * Method comparison — method,metric,<param>,mean,sd (demo 01's
#     lambda2_method, hc_linkage and cut studies). Rendered as grouped bar
#     charts (TPR and FDR panels, bars grouped by parameter level, one colour
#     per solver, +/- 1 sd error bars, tFDR reference line on the FDR panel).
#     The cut study adds full-group recovery and group FDR panels (2 x 2 grid).
#
#   * Block benchmark — scenario,rho,snr,method,... (demo 08). Rendered as
#     TPR / FDR line plots, one line per method, tFDR reference on the FDR
#     row. --vs snr (default) puts SNR on the x-axis with one panel per rho;
#     --vs rho puts rho on the x-axis with one panel per SNR (the demo 08
#     headline view). --panels restricts which panel values are drawn.
#
# Usage:
#   Rscript trex_gvs_plt_utils.R <results.csv> [--outdir DIR] [--title T]
#                                [--tfdr F] [--formats png,pdf]
#                                [--param-label L] [--vs snr|rho]
#                                [--panels V ...] [--stem S]
#
# Divergences from the Python plotter:
#   * Output is PNG + PDF only (no HTML output in either port).
#   * --xscale is not ported: the block-benchmark x-axis is always linear,
#     which is what every committed invocation uses.
#   * The FDR colorbar does not carry the cyan tick line at tFDR; the cyan
#     cell outlines and the figure footnote are reproduced.
#   * --panels values may be space-separated (as in Python) or one
#     comma-separated token.
# ==============================================================================

suppressMessages({
  library(ggplot2)
  library(patchwork)
})

DEFAULT_TFDR <- 0.1
SOLVER_ORDER <- c("TENET", "TENET_AUG", "TIENET_AUG")

# Block-benchmark method tags (demo 08 CSVs) -> legend labels.
METHOD_LABELS <- c(
  "M1(EN-C)" = "EN + HAC groups",
  "M2(EN-O)" = "EN + oracle groups",
  "M3(IE-C)" = "IEN + HAC groups",
  "M4(IE-O)" = "IEN + oracle groups"
)

# Hollow per-method markers (o, s, ^, D) — overlapping curves (e.g. M1/M2 at
# high rho) stay distinguishable by marker shape. ggplot2 shapes 21-25.
METHOD_SHAPES <- c(21L, 22L, 24L, 23L)

# matplotlib "tab10" — the colour convention shared by every suite's figures.
TAB10 <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")

# Axis-prefix -> human label for the swept quantity.
AXIS_LABEL <- c(snr = "SNR", rho = "ρ", ac = "AR coefficient",
                ar_coef = "AR coefficient")
VIOLATION <- "cyan"


# ------------------------------------------------------------------------------
# Shared helpers (house conventions)
# ------------------------------------------------------------------------------

#' 'snr=0.2' -> axis 'snr', numeric value 0.2 (labels without '=' pass).
.split_axis <- function(labels) {
  s <- as.character(labels)
  has_eq <- grepl("=", s, fixed = TRUE)
  axis <- if (any(has_eq)) sub("=.*$", "", s[has_eq][[1L]]) else ""
  vals <- ifelse(has_eq, sub("^[^=]*=", "", s), s)
  list(axis = axis, values = as.numeric(vals))
}

.axis_label <- function(prefix) {
  if (prefix %in% names(AXIS_LABEL)) AXIS_LABEL[[prefix]] else prefix
}

.fmt_g <- function(v) sprintf("%g", v)

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

#' Linear sweep x-axis with 4% padding (block-benchmark line plots).
.sweep_x_scale <- function(values) {
  lo <- min(values)
  hi <- max(values)
  span <- hi - lo
  pad <- if (span > 0) span * 0.04 else 0.5
  scale_x_continuous(limits = c(lo - pad, hi + pad),
                     breaks = scales::breaks_extended(6),
                     expand = expansion(0))
}

#' Elementwise np.isclose (default rtol/atol) against a set of targets.
.is_close_any <- function(v, targets) {
  vapply(v, function(x) any(abs(x - targets) <= 1e-8 + 1e-5 * abs(targets)),
         logical(1))
}

save_figure <- function(plot, outdir, stem, formats = c("png", "pdf"),
                        width = 12.4, height = 4.8) {
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  for (ext in formats) {
    path <- file.path(outdir, paste0(stem, ".", ext))
    # cairo_pdf: the titles carry Unicode (ρ, λ₂, ×) the base pdf device
    # cannot encode.
    if (ext == "pdf") {
      ggsave(path, plot = plot, width = width, height = height,
             device = grDevices::cairo_pdf, bg = "white")
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


# ------------------------------------------------------------------------------
# 2-D sweep heatmap grid
# ------------------------------------------------------------------------------

#' Per-cell annotation colour: white on dark cells, near-black on light ones
#' (luminance threshold 0.5 on the mapped colormap colour — mirrors the
#' Python plotter's _annotate_cells).
.annotation_colors <- function(vals, pal, vmin, vmax) {
  span <- vmax - vmin
  if (!is.finite(span) || span <= 0) span <- 1
  idx <- round((vals - vmin) / span * (length(pal) - 1)) + 1L
  idx <- pmin(pmax(idx, 1L), length(pal))
  rgb <- grDevices::col2rgb(pal[idx]) / 255
  lum <- 0.299 * rgb[1, ] + 0.587 * rgb[2, ] + 0.114 * rgb[3, ]
  ifelse(lum < 0.5, "white", "grey15")
}

#' One heatmap panel: solver x metric, discrete SNR / rho axes, annotated.
.heat_panel <- function(sub, pal, vmin, vmax, panel_title, x_lab, y_lab,
                        tfdr_outline = 0, show_y = TRUE) {
  sub$x_f <- factor(.fmt_g(sub$col_val), levels = .fmt_g(sort(unique(sub$col_val))))
  sub$y_f <- factor(.fmt_g(sub$row_val), levels = .fmt_g(sort(unique(sub$row_val))))
  sub$ann_col <- .annotation_colors(sub$mean, pal, vmin, vmax)

  p <- ggplot(sub, aes(x = x_f, y = y_f)) +
    geom_tile(aes(fill = mean)) +
    geom_text(aes(label = sprintf("%.2f", mean), colour = ann_col),
              size = 2.5, show.legend = FALSE) +
    scale_colour_identity() +
    scale_fill_gradientn(colours = pal, limits = c(vmin, vmax), name = NULL) +
    scale_x_discrete(expand = expansion(0)) +
    scale_y_discrete(expand = expansion(0)) +
    labs(title = panel_title, x = x_lab, y = if (show_y) y_lab else NULL) +
    theme_bw(base_size = 11) +
    theme(panel.grid = element_blank(),
          plot.title = element_text(size = 11, face = "bold", hjust = 0.5),
          axis.title = element_text(size = 11))

  viol <- sub[tfdr_outline > 0 & is.finite(sub$mean) & sub$mean > tfdr_outline, ]
  if (nrow(viol) > 0) {
    p <- p + geom_tile(data = viol, fill = NA, colour = VIOLATION,
                       linewidth = 0.9)
  }
  if (!show_y) p <- p + theme(axis.text.y = element_blank(),
                              axis.ticks.y = element_blank())
  p
}

plot_2d_heatmaps <- function(df, title, tfdr = DEFAULT_TFDR) {
  row_ax <- .split_axis(df$row_label)
  col_ax <- .split_axis(df$col_label)
  df$row_val <- row_ax$values
  df$col_val <- col_ax$values
  solvers <- SOLVER_ORDER[SOLVER_ORDER %in% unique(df$method)]

  fdr_top <- max(df$mean[df$metric == "mean_FDP"], tfdr, na.rm = TRUE)
  n_col <- length(unique(df$col_val))
  fig_w <- max(11.0, 1.6 + length(solvers) * (1.4 + 0.5 * n_col))

  pal_tpr <- viridisLite::viridis(256)
  pal_fdr <- viridisLite::magma(256, direction = -1)
  specs <- list(
    list(metric = "mean_TPP", mlabel = "TPR (true positive rate)",
         pal = pal_tpr, vmin = 0, vmax = 1),
    list(metric = "mean_FDP", mlabel = "FDR (false discovery rate)",
         pal = pal_fdr, vmin = 0, vmax = fdr_top)
  )

  rows <- lapply(seq_along(specs), function(mi) {
    sp <- specs[[mi]]
    panels <- lapply(seq_along(solvers), function(si) {
      sub <- df[df$method == solvers[si] & df$metric == sp$metric, ]
      head_txt <- if (mi == 1) paste0(solvers[si], "\n", sp$mlabel) else sp$mlabel
      .heat_panel(sub, sp$pal, sp$vmin, sp$vmax, head_txt,
                  x_lab = .axis_label(col_ax$axis),
                  y_lab = .axis_label(row_ax$axis),
                  tfdr_outline = if (sp$metric == "mean_FDP") tfdr else 0,
                  show_y = si == 1)
    })
    # One shared colorbar per metric row (all panels share the fill scale).
    wrap_plots(panels, nrow = 1) + plot_layout(guides = "collect")
  })

  caption <- if (tfdr > 0)
    sprintf("cyan boxes: realized FDR above tFDR = %g", tfdr) else NULL
  fig <- (rows[[1]] / rows[[2]]) +
    plot_annotation(
      title = title, caption = caption,
      theme = theme(
        plot.title = element_text(size = 15, face = "bold", hjust = 0.5),
        plot.caption = element_text(size = 10, colour = "darkcyan",
                                    hjust = 0.5)
      )
    )
  list(fig = fig, width = fig_w, height = 8.6)
}


# ------------------------------------------------------------------------------
# Method-comparison grouped bars (demo 01 lambda2_method / hc_linkage / cut)
# ------------------------------------------------------------------------------

.bar_panel <- function(df, param_col, metric, mlabel, solvers, levels,
                       param_label, tfdr = 0) {
  sub <- df[df$metric == metric & df$method %in% solvers, ]
  sub$method <- factor(sub$method, levels = solvers)
  sub$level <- factor(sub[[param_col]], levels = levels)
  p <- ggplot(sub, aes(x = level, y = mean, fill = method)) +
    geom_col(width = 0.8, position = position_dodge(width = 0.8)) +
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd),
                  position = position_dodge(width = 0.8),
                  width = 0.25, linewidth = 0.4, colour = "black",
                  alpha = 0.6)

  if (tfdr > 0) {
    p <- p + geom_hline(
      aes(yintercept = tfdr, linetype = sprintf("tFDR = %g", tfdr)),
      color = "black", linewidth = 0.8
    ) +
      scale_linetype_manual(name = NULL, values = "dotted",
                            guide = guide_legend(order = 1))
  }

  p <- p +
    scale_fill_manual(name = "Method",
                      values = setNames(TAB10[seq_along(solvers)], solvers)) +
    guides(fill = guide_legend(order = 2)) +
    labs(title = mlabel, x = param_label, y = NULL) +
    .panel_theme() +
    theme(panel.grid.major.x = element_blank(),
          panel.grid.minor.x = element_blank())
  if (metric %in% c("mean_TPP", "grp_full", "grp_FDP"))
    p <- p + coord_cartesian(ylim = c(0, 1.05))
  if (max(nchar(as.character(levels))) > 6)
    p <- p + theme(axis.text.x = element_text(angle = 15, hjust = 1))
  p
}

plot_method_bars <- function(df, param_col, title, param_label,
                             tfdr = DEFAULT_TFDR) {
  levels <- unique(df[[param_col]])
  solvers <- SOLVER_ORDER[SOLVER_ORDER %in% unique(df$method)]

  p_tpr <- .bar_panel(df, param_col, "mean_TPP", "TPR (true positive rate)",
                      solvers, levels, param_label)
  p_fdr <- .bar_panel(df, param_col, "mean_FDP", "FDR (false discovery rate)",
                      solvers, levels, param_label, tfdr = tfdr)

  # Cut study: group-level panels below the variable-level ones
  grouped <- "grp_full" %in% df$metric
  layout <- if (grouped) {
    p_full <- .bar_panel(df, param_col, "grp_full", "Full-group recovery",
                         solvers, levels, param_label)
    p_gfdr <- .bar_panel(df, param_col, "grp_FDP",
                         "Group FDR (groups without signal)", solvers, levels,
                         param_label)
    (p_tpr | p_fdr) / (p_full | p_gfdr)
  } else {
    p_tpr | p_fdr
  }
  fig <- layout +
    plot_layout(guides = "collect") +
    plot_annotation(title = title,
                    theme = theme(plot.title = element_text(
                      size = 14, face = "bold", hjust = 0.5
                    )))
  list(fig = fig, width = max(11.0, 2.2 * length(levels) + 5),
       height = if (grouped) 8.6 else 4.8)
}


# ------------------------------------------------------------------------------
# Block-benchmark line plots (demo 08)
# ------------------------------------------------------------------------------

.bench_panel <- function(sub, x_col, metric, methods, x_values, panel_title,
                         x_lab, y_lab, ylim = NULL, tfdr = 0) {
  sub <- sub[order(sub[[x_col]]), ]
  sub$mlab <- factor(
    ifelse(sub$method %in% names(METHOD_LABELS),
           METHOD_LABELS[sub$method], sub$method),
    levels = ifelse(methods %in% names(METHOD_LABELS),
                    METHOD_LABELS[methods], methods)
  )
  sub$x <- sub[[x_col]]
  sub$y <- sub[[metric]]

  p <- ggplot(sub, aes(x = x, y = y, colour = mlab, shape = mlab)) +
    geom_line(linewidth = 0.6) +
    geom_point(size = 2.4, fill = NA, stroke = 0.9)

  if (tfdr > 0) {
    p <- p + geom_hline(
      aes(yintercept = tfdr, linetype = sprintf("tFDR = %g", tfdr)),
      color = "black", linewidth = 0.8
    ) +
      scale_linetype_manual(name = NULL, values = "dotted",
                            guide = guide_legend(order = 1))
  }

  lvls <- levels(sub$mlab)
  p <- p +
    scale_color_manual(name = "Method",
                       values = setNames(TAB10[seq_along(lvls)], lvls)) +
    scale_shape_manual(name = "Method",
                       values = setNames(
                         METHOD_SHAPES[(seq_along(lvls) - 1L) %%
                                         length(METHOD_SHAPES) + 1L], lvls
                       )) +
    guides(color = guide_legend(order = 2), shape = guide_legend(order = 2)) +
    labs(title = panel_title, x = x_lab, y = y_lab) +
    .sweep_x_scale(x_values) +
    .panel_theme()
  if (!is.null(ylim)) p <- p + coord_cartesian(ylim = ylim)
  p
}

plot_block_bench <- function(df, title, tfdr = DEFAULT_TFDR, vs = "snr",
                             panels = NULL) {
  x_col <- vs                                     # x-axis quantity
  panel_col <- if (vs == "snr") "rho" else "snr"  # one panel per value
  panel_vals <- sort(unique(df[[panel_col]]))
  if (!is.null(panels)) {
    panel_vals <- panel_vals[.is_close_any(panel_vals, panels)]
    if (length(panel_vals) == 0)
      stop(sprintf("--panels {%s} matches no %s value in the CSV ({%s})",
                   paste(panels, collapse = ", "), panel_col,
                   paste(sort(unique(df[[panel_col]])), collapse = ", ")))
  }
  methods <- unique(df$method)
  # Shared across panels, so the axis is configured from the full swept grid.
  x_values <- sort(unique(df[[x_col]]))

  metric_rows <- list(
    list(metric = "mean_coord_tpr", mlabel = "TPR (true positive rate)",
         ylim = c(0, 1.05), tfdr = 0),
    list(metric = "mean_coord_fdp", mlabel = "FDR (false discovery rate)",
         ylim = NULL, tfdr = tfdr)
  )

  panels_lst <- list()
  for (mi in seq_along(metric_rows)) {
    mr <- metric_rows[[mi]]
    for (ri in seq_along(panel_vals)) {
      pv <- panel_vals[ri]
      sub <- df[.is_close_any(df[[panel_col]], pv), ]
      head_txt <- if (mi == 1) {
        if (panel_col == "rho") sprintf("ρ = %g", pv) else sprintf("SNR = %g", pv)
      } else {
        NULL
      }
      panels_lst[[length(panels_lst) + 1L]] <- .bench_panel(
        sub, x_col, mr$metric, methods, x_values,
        panel_title = head_txt,
        x_lab = if (mi == 2) (if (x_col == "snr") "SNR" else "ρ") else NULL,
        y_lab = if (ri == 1) mr$mlabel else NULL,
        ylim = mr$ylim, tfdr = mr$tfdr
      )
    }
  }

  fig <- wrap_plots(panels_lst, nrow = 2) +
    plot_layout(guides = "collect") +
    plot_annotation(title = title,
                    theme = theme(plot.title = element_text(
                      size = 15, face = "bold", hjust = 0.5
                    )))
  list(fig = fig, width = 5.2 * length(panel_vals) + 2.0, height = 8.4)
}


# ------------------------------------------------------------------------------
# Dispatch + CLI
# ------------------------------------------------------------------------------

detect_kind <- function(df) {
  cols <- names(df)
  if (all(c("row_label", "col_label") %in% cols)) return("2d")
  if (all(c("scenario", "rho", "snr", "method") %in% cols)) return("block")
  if (all(c("method", "metric", "mean", "sd") %in% cols) && ncol(df) == 5)
    return("method")
  stop("Unrecognized CSV schema: columns = ", paste(sort(cols), collapse = ", "))
}

.parse_args <- function(argv) {
  args <- list(csv = character(0), outdir = NULL, title = NULL,
               tfdr = DEFAULT_TFDR, formats = c("png", "pdf"),
               param_label = NULL, stem = NULL, vs = "snr", panels = NULL)
  i <- 1L
  while (i <= length(argv)) {
    a <- argv[[i]]
    flag <- function() {
      i <<- i + 1L
      argv[[i]]
    }
    switch(a,
           "--outdir"      = {
             args$outdir <- flag()
           },
           "--title"       = {
             args$title <- flag()
           },
           "--tfdr"        = {
             args$tfdr <- as.numeric(flag())
           },
           "--formats"     = {
             # Consume all following extension tokens (Python nargs='+' style);
             # a single comma-separated token also works.
             fmts <- character(0)
             while (i < length(argv) &&
                      grepl("^(png|pdf|svg)(,(png|pdf|svg))*$", argv[[i + 1L]])) {
               fmts <- c(fmts, strsplit(argv[[i + 1L]], ",")[[1]])
               i <- i + 1L
             }
             if (length(fmts) == 0) stop("--formats requires png/pdf value(s)")
             args$formats <- fmts
           },
           "--param-label" = {
             args$param_label <- flag()
           },
           "--stem"        = {
             args$stem <- flag()
           },
           "--vs"          = {
             args$vs <- flag()
           },
           "--panels"      = {
             # Consume all following numeric tokens (Python nargs='+' style);
             # a single comma-separated token also works.
             vals <- numeric(0)
             while (i < length(argv) && !startsWith(argv[[i + 1L]], "--")) {
               tok <- strsplit(argv[[i + 1L]], ",")[[1]]
               num <- suppressWarnings(as.numeric(tok))
               if (any(is.na(num))) break
               vals <- c(vals, num)
               i <- i + 1L
             }
             if (length(vals) == 0) stop("--panels requires numeric value(s)")
             args$panels <- vals
           },
           {
             args$csv <- c(args$csv, a)
           })
    i <- i + 1L
  }
  if (length(args$csv) != 1L)
    stop("Usage: trex_gvs_plt_utils.R <results.csv> [--outdir DIR] ",
         "[--title T] [--tfdr F] [--formats png,pdf] [--param-label L] ",
         "[--vs snr|rho] [--panels V ...] [--stem S]")
  if (!args$vs %in% c("snr", "rho"))
    stop("--vs must be 'snr' or 'rho', got: ", args$vs)
  args
}

main <- function() {
  args <- .parse_args(commandArgs(trailingOnly = TRUE))
  csv_path <- args$csv[[1]]
  df <- load_csv(csv_path)

  kind <- detect_kind(df)
  outdir <- if (is.null(args$outdir)) default_plots_dir(csv_path) else args$outdir
  stem <- if (is.null(args$stem)) sub("[.]csv$", "", basename(csv_path)) else args$stem
  title <- if (is.null(args$title)) sub("[.]csv$", "", basename(csv_path)) else args$title

  res <- if (kind == "2d") {
    plot_2d_heatmaps(df, title, args$tfdr)
  } else if (kind == "method") {
    param_col <- setdiff(names(df), c("method", "metric", "mean", "sd"))[[1]]
    label <- if (is.null(args$param_label)) param_col else args$param_label
    plot_method_bars(df, param_col, title, label, args$tfdr)
  } else {  # block
    plot_block_bench(df, title, args$tfdr, args$vs, args$panels)
  }

  save_figure(res$fig, outdir, stem, args$formats,
              width = res$width, height = res$height)
  invisible(0L)
}

if (sys.nframe() == 0L) main()
