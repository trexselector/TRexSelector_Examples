#!/usr/bin/env Rscript
# ==============================================================================
# trex_da_plt_utils.R
# ==============================================================================
#
# Shared plotting utilities for the DA-TRex demo suite — ggplot2 port of the
# Python plotter of the same name (trex_da_plt_utils.py), kept visually aligned
# with it: tab10 (+ Dark2 overflow) method colours, hollow per-method markers,
# black dotted tFDR reference, viridis TPR / reversed-magma FDR heatmaps with
# cyan outlines marking realized FDR above the tFDR target, ONE collected
# legend outside the panels, PNG + PDF output into the sibling
# simulation_results/plots/ folder.
#
# The Monte Carlo results use the tidy CSV schema
#
#     solver,metric,<SWEEP>,value
#
# where <SWEEP> is whatever the demo sweeps — one of SNR, Rho, M (blocks), Q
# (block size), tFDR, or a Linkage index — and the 'solver' column holds the
# methods compared. Each demo records the metrics FDR, sd_FDR, TPR, sd_TPR
# (plus bookkeeping Avg_L / Avg_T which are ignored here); the shared reader
# auto-detects the sweep column, so no per-demo configuration is required
# beyond the CSV path and titles supplied by each demo's generate_plots.sh.
#
# Four CLI modes (all typically driven from a demo's generate_plots.sh):
#   # default — per-CSV FDR | TPR overview, sweep auto-detected
#   Rscript trex_da_plt_utils.R <csv> --title "..." --legend-title Method --tfdr 0.2
#   # cross-CSV comparison grid (columns = linkage / scenario / support / SNR)
#   Rscript trex_da_plt_utils.R grid --labels Single Complete Average \
#       --csvs a_single.csv a_complete.csv a_average.csv \
#       --title "..." --stem my_grid --outdir <plots>
#   # demo-01 gap x rho 2D study from the wide CSV
#   Rscript trex_da_plt_utils.R gaprho <gap_rho.csv> --stem my_gaprho --outdir <plots>
#   # generic 2D sweep heatmaps (one figure per support placement)
#   Rscript trex_da_plt_utils.R sweep2d <kappa_rho.csv> --stem my_2d --outdir <plots>
#
# Output location: figures default to the simulation_results/plots/ sibling
# when the CSV lives in simulation_results/data/ (the suite convention);
# otherwise they are written next to the CSV. Override with --outdir. (The
# grid / gaprho / sweep2d modes require an explicit --outdir.)
#
# Divergences from the Python plotter:
#   * Output is PNG + PDF only — no interactive Plotly HTML in the R port
#     (--no-plotly / --plotly-cdn are accepted and ignored for CLI parity).
#   * The legend is the house ggplot2 form: the tFDR reference as a linetype
#     entry above a titled Method colour/shape legend, instead of Python's
#     in-legend "Method:" header row.
#   * Every point carries its marker (no marker striding; the DA sweep grids
#     have at most 10 values).
#   * Heatmap axes are discrete factor axes (equal-width cells), matching the
#     house heatmap style; Python places rho ticks at their numeric positions.
#   * The demo-01 kappa-boundary overlay is explained in the figure caption
#     rather than an in-panel legend, and the gap x rho / sweep2d FDR
#     colorbars do not carry the cyan tick line at tFDR (the cyan cell
#     outlines and the footnote are reproduced).
# ==============================================================================

suppressMessages({
  library(ggplot2)
  library(patchwork)
})

STEM_SUFFIX <- "_fdr_tpr"
DEFAULT_TFDR <- 0.20          # target FDR control level (tFDR); demo-specific
DEFAULT_LEGEND_TITLE <- "Method"
Y_HEADROOM <- 1.25            # budget above the largest FDR value / tFDR line
MAX_XTICKS <- 10L             # keep all ticks on the rho grid; thin denser sweeps

# TAFS is the only solver run with a non-zero AFS correlation parameter, and it
# is a defining part of the method rather than a tuning detail — so it is shown
# in every legend. Mirrors rho_afs in default_solvers() (trex_da_sim_common);
# keep the two in sync. Applied at plot time rather than baked into the CSVs.
TAFS_RHO <- 0.3

METRICS <- c("FDR", "TPR")
METRIC_LABELS <- c(FDR = "False discovery rate", TPR = "True positive rate")
# Each plotted metric's paired standard-deviation metric (upper-cased).
SD_OF <- c(FDR = "SD_FDR", TPR = "SD_TPR")

# matplotlib "tab10" + "Dark2" (18 hand-picked, high-contrast colours) — the
# colour convention shared by every suite's figures; black is reserved for the
# tFDR reference line, so no data row uses it.
TAB10 <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")
DARK2 <- c("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
           "#e6ab02", "#a6761d", "#666666")
PALETTE <- c(TAB10, DARK2)

# Hollow per-method markers (o, s, ^, D, v) first — overlapping curves stay
# distinguishable. ggplot2 shapes 21-25 honour fill; the rest cycle.
METHOD_SHAPES <- c(21L, 22L, 24L, 23L, 25L, 3L, 4L, 8L, 7L, 9L, 12L, 13L, 14L, 10L)

VIOLATION <- "cyan"           # FDR-above-tFDR outline / kappa-boundary colour

# ------------------------------------------------------------------------------
# Sweep-axis configuration
# ------------------------------------------------------------------------------
# The DA demos sweep different quantities; the x-axis label, scale, and tick
# behaviour are chosen from the detected sweep column (CSV's 3rd column).
SWEEP_CONFIG <- list(
  snr     = list(label = "Signal-to-noise ratio (SNR)", scale = "log"),
  rho     = list(label = "AR(1) correlation ρ", scale = "linear"),
  m       = list(label = "Number of blocks M", scale = "linear"),
  q       = list(label = "Block size Q", scale = "linear"),
  tfdr    = list(label = "Target FDR level (tFDR)", scale = "linear",
                 is_tfdr_sweep = TRUE),
  kappa   = list(label = "NN bandwidth κ", scale = "linear"),
  linkage = list(label = "HAC linkage method", scale = "linear",
                 categorical = c("1" = "Single", "2" = "Complete",
                                 "3" = "Average"))
)
DEFAULT_SWEEP_CONFIG <- list(label = "Sweep value", scale = "linear")

#' Normalise a CSV sweep-column name to a SWEEP_CONFIG key ('SNR' -> 'snr',
#' the verbose 'Linkage_(1=Single,...)' header -> 'linkage').
sweep_key_of <- function(column) {
  name <- tolower(trimws(as.character(column)))
  if (startsWith(name, "linkage")) return("linkage")
  name
}

#' Look up the axis configuration for a sweep key (with a safe default).
sweep_config <- function(key) {
  if (key %in% names(SWEEP_CONFIG)) SWEEP_CONFIG[[key]] else DEFAULT_SWEEP_CONFIG
}


# ------------------------------------------------------------------------------
# Shared helpers (house conventions)
# ------------------------------------------------------------------------------

`%||%` <- function(a, b) if (is.null(a)) b else a

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

# A "<solver>_STAG" row is <solver> with the T-loop stagnation stop on (the
# TLARS_STAG comparison row of the suite). It shares its sibling's colour and
# is drawn dashed, so each pair reads as one solver with and without the stop.
STAG_SUFFIX <- "_STAG"

#' Sibling row of each "_STAG" row (NA for the other rows).
.stag_sibling <- function(methods) {
  methods <- as.character(methods)
  ifelse(endsWith(methods, STAG_SUFFIX),
         substr(methods, 1L, nchar(methods) - nchar(STAG_SUFFIX)),
         NA_character_)
}

#' Stable colour / shape maps keyed by position in the method list, so a
#' given method is the same colour in every panel and every figure. A "_STAG"
#' row takes its sibling's colour and no palette slot of its own.
.method_cols <- function(methods) {
  sib <- .stag_sibling(methods)
  paired <- !is.na(sib) & sib %in% methods
  own <- methods[!paired]
  cols <- setNames(PALETTE[(seq_along(own) - 1L) %% length(PALETTE) + 1L], own)
  for (i in which(paired)) cols[methods[i]] <- cols[[sib[i]]]
  cols[methods]
}

#' Method lines: solid, dashed for "_STAG" rows. The dashed layer draws a key
#' for every method, and .method_ltys() sets each key's line type.
.method_lines <- function(sub) {
  stag <- !is.na(.stag_sibling(sub$solver))
  list(geom_line(data = sub[!stag, ], linewidth = 0.7),
       geom_line(data = sub[stag, ], linewidth = 0.7, linetype = "dashed",
                 show.legend = TRUE))
}
.method_ltys <- function(methods) {
  ifelse(is.na(.stag_sibling(methods)), "solid", "dashed")
}
.method_shps <- function(methods) {
  setNames(METHOD_SHAPES[(seq_along(methods) - 1L) %%
                           length(METHOD_SHAPES) + 1L], methods)
}

save_figure <- function(plot, outdir, stem, formats = c("png", "pdf"),
                        width = 12.4, height = 4.8) {
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  for (ext in formats) {
    path <- file.path(outdir, paste0(stem, ".", ext))
    # cairo_pdf: the titles / labels carry Unicode (rho, kappa) the base pdf
    # device cannot encode.
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
  if (basename(parent) == "data") file.path(dirname(parent), "plots") else parent
}


# ------------------------------------------------------------------------------
# Reading the tidy Monte Carlo CSV
# ------------------------------------------------------------------------------

#' Load a tidy DA-TRex results CSV (solver,metric,<sweep>,value).
#'
#' The third column is the swept quantity and its name varies between demos, so
#' it is detected positionally and renamed to `sweep`; its original name is
#' normalised to a SWEEP_CONFIG key and returned alongside the frame. Only the
#' plotted metrics (FDR/TPR and their sd_ partners) are kept; other recorded
#' metrics (Avg_L, Avg_T, ...) are noted and dropped.
#'
#' Returns list(df, sweep_key) with df columns solver, metric, sweep, value.
read_mc_results <- function(path) {
  if (!file.exists(path))
    stop("Input file not found:\n  ", path,
         "\nPass the results CSV path as the first argument.", call. = FALSE)
  df <- utils::read.csv(path, check.names = FALSE, strip.white = TRUE,
                        stringsAsFactors = FALSE)
  if (ncol(df) < 4)
    stop("CSV must have at least 4 columns (solver,metric,<sweep>,value); ",
         "found columns: ", paste(names(df), collapse = ", "), call. = FALSE)

  # Columns are positional: solver, metric, <sweep>, value. A linkage-sweep
  # header carrying an unescaped comma ("Linkage_(1=Single,2=Complete,
  # 3=Average)") over-splits into extra header columns while the 4-field data
  # rows stay aligned to positions 1-4 — so read the data positionally and
  # rebuild the true sweep name from the header fragments.
  sweep_name <- if (ncol(df) > 4) {
    paste(names(df)[3:(ncol(df) - 1L)], collapse = ",")
  } else {
    names(df)[3]
  }
  sweep_key <- sweep_key_of(sweep_name)

  df <- df[, 1:4]
  names(df) <- c("solver", "metric", "sweep", "value")
  df$solver <- trimws(as.character(df$solver))
  # Normalise metric names: 'sd_FDR' -> 'SD_FDR', 'FDR' -> 'FDR'.
  df$metric <- toupper(trimws(as.character(df$metric)))
  df$sweep <- as.numeric(df$sweep)
  df$value <- as.numeric(df$value)

  plotted <- c(METRICS, unname(SD_OF))
  present <- unique(df$metric)
  missing_metrics <- setdiff(METRICS, present)
  if (length(missing_metrics) > 0)
    stop("CSV is missing required metric(s): ",
         paste(sort(missing_metrics), collapse = ", "),
         "\nFound metrics: ", paste(sort(present), collapse = ", "),
         call. = FALSE)
  extra <- setdiff(present, plotted)
  if (length(extra) > 0)
    cat("note: ignoring non-plotted metric(s):",
        paste(sort(extra), collapse = ", "), "\n")

  df <- df[df$metric %in% plotted, ]
  df <- annotate_solver_params(df)
  # C-locale (radix) sort mirrors Python's sort_values, so the first-seen
  # method order — and with it the colour assignment — matches the Python
  # figures byte for byte.
  df <- df[order(df$metric, df$solver, df$sweep, method = "radix"), ]
  list(df = df, sweep_key = sweep_key)
}

#' Append TAFS' AFS correlation parameter to its display label:
#' 'TREX-DA+BT: TAFS' -> 'TREX-DA+BT: TAFS (rho = 0.3)'. Idempotent.
annotate_solver_params <- function(df) {
  suffix <- sprintf(" (rho = %g)", TAFS_RHO)
  df$solver <- gsub("(?<![\\w()])TAFS(?!\\s*\\(rho)",
                    paste0("TAFS", suffix), df$solver, perl = TRUE)
  df
}

#' Rename method labels in the 'solver' column for display. Each spec is
#' OLD=NEW; OLD is replaced as a substring, in order. Purely cosmetic.
apply_relabel <- function(df, specs) {
  for (spec in specs) {
    parts <- regmatches(spec, regexpr("=", spec), invert = TRUE)[[1]]
    if (length(parts) != 2L)
      stop("--relabel expects OLD=NEW, got: ", spec, call. = FALSE)
    df$solver <- gsub(parts[[1]], parts[[2]], df$solver, fixed = TRUE)
  }
  df
}

#' Upper FDR y-limit: the larger of the data max and the tFDR reference, plus
#' headroom. For a tFDR *sweep* the reference is the largest target on the
#' x-axis (the identity line's top), not a single fixed tFDR.
fdr_axis_top <- function(df, cfg, tfdr) {
  fdr_max <- suppressWarnings(max(df$value[df$metric == "FDR"], na.rm = TRUE))
  if (!is.finite(fdr_max)) fdr_max <- 0
  ref <- if (isTRUE(cfg$is_tfdr_sweep)) max(df$sweep) else if (tfdr > 0) tfdr else 0
  max(fdr_max, ref) * Y_HEADROOM
}

#' A readable subset of x-tick positions for a sweep grid: small grids keep
#' every value; dense grids are thinned to ~max_ticks values — evenly in log
#' space for a log axis, evenly in index order otherwise — endpoints retained.
choose_ticks <- function(values, scale, max_ticks = MAX_XTICKS) {
  values <- sort(unique(values))
  if (length(values) <= max_ticks) return(values)
  if (scale == "log") {
    lo <- log10(values[[1]])
    hi <- log10(values[[length(values)]])
    ticks <- numeric(0)
    for (i in seq_len(max_ticks)) {
      target <- lo + (hi - lo) * (i - 1) / (max_ticks - 1)
      nearest <- values[[which.min(abs(log10(values) - target))]]
      if (!nearest %in% ticks) ticks <- c(ticks, nearest)
    }
    return(ticks)
  }
  idx <- unique(round(seq(1, length(values), length.out = max_ticks)))
  values[idx]
}

#' scale_x_continuous whose transform argument works on both pre- and
#' post-3.5 ggplot2 ('trans' was renamed to 'transform').
.x_cont <- function(..., transform = NULL) {
  if (is.null(transform)) return(scale_x_continuous(...))
  if ("transform" %in% names(formals(ggplot2::scale_x_continuous)))
    scale_x_continuous(..., transform = transform) else
    scale_x_continuous(..., trans = transform)
}

#' Layers configuring an axis' scale, ticks, labels and limits for the sweep.
#'
#' `xscale` overrides the sweep-config scale:
#'   * "sqrt" spreads a grid that is dense near zero (e.g. the equi demo's rho
#'     values) while keeping 0 on-axis; tick labels are slanted 45 degrees.
#'   * "linear-minor" is a plain linear axis tick-marked at round numbers with
#'     minor ticks in between, instead of one label per swept value — for
#'     grids with near-neighbours (e.g. the demo-02 SNR grid's 0.5 / 0.6).
.sweep_x_layers <- function(cfg, values, xscale = NULL) {
  values <- sort(unique(values))
  categorical <- cfg$categorical
  if (!is.null(categorical)) {
    present <- values[values %in% as.numeric(names(categorical))]
    return(list(scale_x_continuous(
      breaks = present,
      labels = unname(categorical[as.character(as.integer(present))]),
      limits = c(min(present) - 0.5, max(present) + 0.5),
      expand = expansion(0)
    )))
  }

  scale <- xscale %||% cfg$scale
  span <- max(values) - min(values)
  pad <- if (span > 0) span * 0.04 else 0.5

  if (scale == "linear-minor") {
    # Ticks come from the break algorithm, not the sweep grid, so labels
    # cannot collide however tightly the swept values are spaced.
    return(list(scale_x_continuous(
      breaks = scales::breaks_extended(7),
      limits = c(min(values) - pad, max(values) + pad),
      expand = expansion(0)
    )))
  }
  if (scale == "log") {
    return(list(scale_x_log10(
      breaks = choose_ticks(values, "log"), labels = .fmt_g,
      limits = c(min(values) * 0.82, max(values) * 1.18),
      expand = expansion(0)
    )))
  }
  if (scale == "sqrt") {
    span_sqrt <- sqrt(max(values)) - sqrt(max(min(values), 0))
    pad_sq <- if (span_sqrt > 0) (span_sqrt * 0.04)^2 else 0.5
    return(list(
      .x_cont(breaks = choose_ticks(values, "linear"), labels = .fmt_g,
              limits = c(min(values), max(values) + pad_sq),
              expand = expansion(0), transform = "sqrt"),
      # Grid values cluster near zero even after the sqrt stretch — slant
      # the labels so neighbouring ticks cannot collide.
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    ))
  }
  list(scale_x_continuous(
    breaks = choose_ticks(values, "linear"), labels = .fmt_g,
    limits = c(min(values) - pad, max(values) + pad),
    expand = expansion(0)
  ))
}

#' The FDR target reference for an FDR panel: a horizontal dotted tFDR line,
#' or the identity line (realized FDR == target) for a tFDR *sweep*. Returns
#' an empty list if no reference applies (tfdr == 0 on a non-tFDR sweep).
.fdr_ref_layers <- function(cfg, tfdr, values) {
  if (isTRUE(cfg$is_tfdr_sweep)) {
    ident <- data.frame(x = sort(unique(values)))
    return(list(
      geom_line(data = ident, aes(x = x, y = x, linetype = "realized = target"),
                colour = "black", linewidth = 0.8, inherit.aes = FALSE),
      scale_linetype_manual(name = NULL,
                            values = c("realized = target" = "dotted"),
                            guide = guide_legend(order = 1))
    ))
  }
  if (tfdr > 0) {
    lab <- sprintf("tFDR = %g", tfdr)
    return(list(
      geom_hline(aes(yintercept = tfdr, linetype = lab),
                 color = "black", linewidth = 0.8),
      scale_linetype_manual(name = NULL, values = setNames("dotted", lab),
                            guide = guide_legend(order = 1))
    ))
  }
  list()
}


# ------------------------------------------------------------------------------
# Pattern 1 + 3 — FDR / TPR line panels vs the swept quantity
# ------------------------------------------------------------------------------

#' One metric panel: all methods as lines vs the sweep, optional +/- sd bands,
#' tFDR reference on FDR panels. `methods` fixes the colour/shape assignment
#' (shared across the panels of a comparison grid via limits + drop=FALSE).
.line_panel <- function(df, methods, metric, cfg, all_values, legend_title,
                        tfdr, bands, xscale, panel_title, x_lab, y_lab, ylim) {
  cols <- .method_cols(methods)
  shps <- .method_shps(methods)
  sub <- df[df$metric == metric, ]
  sub <- sub[order(match(sub$solver, methods), sub$sweep), ]
  sub$solver <- factor(sub$solver, levels = methods)

  p <- ggplot(sub, aes(x = sweep, y = value, colour = solver, shape = solver))

  if (bands) {
    sd_sub <- df[df$metric == SD_OF[[metric]], c("solver", "sweep", "value")]
    names(sd_sub)[3] <- "sd"
    band <- merge(sub, sd_sub, by = c("solver", "sweep"))
    if (nrow(band) > 0) {
      band$lo <- pmax(band$value - band$sd, 0)
      band$hi <- band$value + band$sd
      if (metric == "TPR") band$hi <- pmin(band$hi, 1)
      p <- p +
        geom_ribbon(data = band,
                    aes(x = sweep, ymin = lo, ymax = hi, fill = solver),
                    alpha = 0.12, colour = NA, inherit.aes = FALSE) +
        scale_fill_manual(values = cols, limits = methods, guide = "none")
    }
  }

  if (metric == "FDR")
    p <- p + .fdr_ref_layers(cfg, tfdr, all_values)

  key_ltys <- list(linetype = .method_ltys(methods))
  p <- p +
    .method_lines(sub) +
    geom_point(size = 2.4, fill = NA, stroke = 0.9) +
    scale_color_manual(name = legend_title, values = cols, limits = methods,
                       drop = FALSE) +
    scale_shape_manual(name = legend_title, values = shps, limits = methods,
                       drop = FALSE) +
    guides(colour = guide_legend(order = 2, override.aes = key_ltys),
           shape = guide_legend(order = 2)) +
    labs(title = panel_title, x = x_lab, y = y_lab) +
    # Panel theme first: the sqrt-scale layers carry a theme tweak (slanted
    # tick labels) that must land on top of the full theme_bw reset.
    .panel_theme() +
    .sweep_x_layers(cfg, all_values, xscale) +
    coord_cartesian(ylim = ylim)
  p
}

#' Pattern 1 — side-by-side FDR and TPR panels versus the swept quantity.
#'
#' One line per method, colours/markers keyed to the method. The FDR panel
#' carries the tFDR reference (a horizontal line, or the identity line for a
#' tFDR sweep) and is scaled to include it. With bands=TRUE each curve gets a
#' translucent +/- sd envelope from the CSV's sd_ metrics. `xlabel` overrides
#' the sweep-config axis label (e.g. the 'rho' default names AR(1), which is
#' wrong for the equi demo).
plot_fdr_tpr_vs_sweep <- function(df, sweep_key, title,
                                  tfdr = DEFAULT_TFDR,
                                  legend_title = DEFAULT_LEGEND_TITLE,
                                  bands = FALSE, xlabel = NULL,
                                  xscale = NULL) {
  cfg <- sweep_config(sweep_key)
  if (!is.null(xlabel)) cfg$label <- xlabel
  methods <- unique(df$solver)
  all_values <- sort(unique(df$sweep))
  fdr_top <- fdr_axis_top(df, cfg, tfdr)

  p_fdr <- .line_panel(df, methods, "FDR", cfg, all_values, legend_title,
                       tfdr, bands, xscale, "FDR", cfg$label,
                       METRIC_LABELS[["FDR"]], c(0, fdr_top))
  p_tpr <- .line_panel(df, methods, "TPR", cfg, all_values, legend_title,
                       0, bands, xscale, "TPR", cfg$label,
                       METRIC_LABELS[["TPR"]], c(-0.02, 1.02))

  fig <- (p_fdr | p_tpr) +
    plot_layout(guides = "collect") +
    plot_annotation(title = title,
                    theme = theme(plot.title = element_text(
                      size = 14, face = "bold", hjust = 0.5
                    )))
  list(fig = fig, width = 13.6, height = 5.6)
}

#' Pattern 3 — cross-CSV comparison grid.
#'
#' `frames` is an *ordered* named list {column_label: df} (all sharing one
#' sweep_key). The figure is a 2xN grid: rows = metric (FDR top, TPR below),
#' columns = the frames' names (HAC linkage in the BT demos 02-05, noise
#' scenario, support placement, or SNR in the equi demo). Every panel carries
#' all methods as lines vs the sweep, colours/markers keyed to the method so a
#' given method is the same colour in every column. All FDR panels share one
#' y-scale and all TPR panels share [0, 1], so columns are directly comparable.
plot_comparison_grid <- function(frames, sweep_key, title,
                                 legend_title = DEFAULT_LEGEND_TITLE,
                                 tfdr = DEFAULT_TFDR, column_title = "",
                                 bands = FALSE, xlabel = NULL,
                                 xscale = NULL) {
  cfg <- sweep_config(sweep_key)
  if (!is.null(xlabel)) cfg$label <- xlabel
  labels <- names(frames)
  # Union of the methods across per-column frames, in first-seen order, so the
  # colour / marker assignment is identical in every column of the grid.
  methods <- unique(unlist(lapply(frames, function(d) unique(d$solver)),
                           use.names = FALSE))
  all_values <- sort(unique(unlist(lapply(frames, function(d) d$sweep),
                                   use.names = FALSE)))
  fdr_top <- max(vapply(frames, function(d) fdr_axis_top(d, cfg, tfdr),
                        numeric(1)))

  panels <- list()
  for (mi in seq_along(METRICS)) {
    metric <- METRICS[[mi]]
    for (ci in seq_along(labels)) {
      head_txt <- if (mi == 1) {
        if (nzchar(column_title)) paste0(column_title, ": ", labels[[ci]]) else labels[[ci]]
      } else {
        NULL
      }
      panels[[length(panels) + 1L]] <- .line_panel(
        frames[[ci]], methods, metric, cfg, all_values, legend_title,
        if (metric == "FDR") tfdr else 0, bands, xscale,
        panel_title = head_txt,
        x_lab = if (mi == length(METRICS)) cfg$label else NULL,
        y_lab = if (ci == 1) METRIC_LABELS[[metric]] else NULL,
        ylim = if (metric == "FDR") c(0, fdr_top) else c(-0.02, 1.02)
      )
    }
  }

  fig <- wrap_plots(panels, nrow = length(METRICS)) +
    plot_layout(guides = "collect") +
    plot_annotation(title = title,
                    theme = theme(plot.title = element_text(
                      size = 15, face = "bold", hjust = 0.5
                    )))
  list(fig = fig, width = 5.0 * length(labels) + 2.8, height = 9.0)
}


# ------------------------------------------------------------------------------
# Heatmap building blocks (gap x rho and sweep2d)
# ------------------------------------------------------------------------------

#' Per-cell annotation colour: white on dark cells, near-black on light ones
#' (luminance threshold 0.5 on the mapped colormap colour).
.annotation_colors <- function(vals, pal, vmin, vmax) {
  span <- vmax - vmin
  if (!is.finite(span) || span <= 0) span <- 1
  idx <- round((vals - vmin) / span * (length(pal) - 1)) + 1L
  idx <- pmin(pmax(idx, 1L), length(pal))
  rgb <- grDevices::col2rgb(pal[idx]) / 255
  lum <- 0.299 * rgb[1, ] + 0.587 * rgb[2, ] + 0.114 * rgb[3, ]
  ifelse(lum < 0.5, "white", "grey15")
}

#' One annotated heatmap panel over discrete row/col axes; `sub` must carry
#' numeric columns row, col, value. FDR cells above `tfdr_outline` are
#' outlined in cyan; `overlay` (a data.frame x, y in axis positions) draws
#' the demo-01 kappa-boundary curve.
.heat_panel <- function(sub, xvals, yvals, pal, vmin, vmax, panel_title,
                        x_lab, y_lab, fill_name = NULL, tfdr_outline = 0,
                        show_y = TRUE, overlay = NULL) {
  sub$x_f <- factor(.fmt_g(sub$col), levels = .fmt_g(xvals))
  sub$y_f <- factor(.fmt_g(sub$row), levels = .fmt_g(yvals))
  sub$ann_col <- .annotation_colors(sub$value, pal, vmin, vmax)

  p <- ggplot(sub, aes(x = x_f, y = y_f)) +
    geom_tile(aes(fill = value)) +
    geom_text(aes(label = sprintf("%.2f", value), colour = ann_col),
              size = 2.3, show.legend = FALSE) +
    scale_colour_identity() +
    scale_fill_gradientn(colours = pal, limits = c(vmin, vmax),
                         name = fill_name) +
    scale_x_discrete(expand = expansion(0)) +
    scale_y_discrete(expand = expansion(0)) +
    labs(title = panel_title, x = x_lab, y = if (show_y) y_lab else NULL) +
    theme_bw(base_size = 11) +
    theme(panel.grid = element_blank(),
          plot.title = element_text(size = 11, face = "bold", hjust = 0.5),
          axis.title = element_text(size = 11))

  viol <- sub[tfdr_outline > 0 & is.finite(sub$value) &
                sub$value > tfdr_outline, ]
  if (nrow(viol) > 0)
    p <- p + geom_tile(data = viol, fill = NA, colour = VIOLATION,
                       linewidth = 0.9)
  if (!is.null(overlay)) {
    overlay <- overlay[is.finite(overlay$x) & is.finite(overlay$y), ]
    if (nrow(overlay) > 0)
      p <- p + geom_line(data = overlay, aes(x = x, y = y),
                         colour = VIOLATION, linetype = "dashed",
                         linewidth = 1.0, inherit.aes = FALSE)
  }
  if (!show_y)
    p <- p + theme(axis.text.y = element_blank(),
                   axis.ticks.y = element_blank())
  p
}


# ------------------------------------------------------------------------------
# Pattern 4 — demo-01 gap x rho 2D study (wider CSV schema)
# ------------------------------------------------------------------------------
# The AR(1) demo's Part-3 output uses a wider schema,
#     solver,metric,rho,support_type,gap,value
# with metrics mean_TPP / sd_TPP / mean_FDP / sd_FDP over a 2D (gap x rho) grid
# under CappedSpread support, plus a Random support baseline (gap column empty).

GAP_RHO_EPS <- 0.02  # correlation cut defining the DA+AR1 window half-width

#' Load the wide demo-01 gap x rho CSV, normalising column/metric whitespace.
read_gap_rho <- function(path) {
  if (!file.exists(path))
    stop("Input file not found:\n  ", path,
         "\nPass the gap_rho results CSV path as the first argument.",
         call. = FALSE)
  df <- utils::read.csv(path, check.names = FALSE, strip.white = TRUE,
                        stringsAsFactors = FALSE)
  names(df) <- tolower(trimws(names(df)))
  required <- c("solver", "metric", "rho", "support_type", "gap", "value")
  missing <- setdiff(required, names(df))
  if (length(missing) > 0)
    stop("CSV is missing required columns: ",
         paste(sort(missing), collapse = ", "),
         "\nFound columns: ", paste(names(df), collapse = ", "),
         call. = FALSE)
  df$solver <- trimws(as.character(df$solver))
  df$metric <- trimws(as.character(df$metric))
  df$support_type <- trimws(as.character(df$support_type))
  df$rho <- as.numeric(df$rho)
  df$value <- as.numeric(df$value)
  df$gap <- suppressWarnings(as.numeric(df$gap))  # empty for Random
  annotate_solver_params(df)
}

#' DA+AR1 correction-window half-width kappa = ceil(log(eps)/log(rho)).
#' Undefined at rho <= 0 (no correlation, so no window) -> NA.
kappa_boundary <- function(rho) {
  kap <- suppressWarnings(ceiling(log(GAP_RHO_EPS) / log(rho)))
  kap[!is.finite(kap) | rho <= 0] <- NA_real_
  kap
}

#' Pattern 4 — demo-01 gap x rho study.
#'
#' Top row: TPR and FDR heatmaps (MC means over the DA solvers) over (rho on
#' x, gap on y) for CappedSpread support, with the kappa-boundary curve
#' gap = kappa(rho) overlaid — below it, active variables share correction
#' windows and TPP is expected to drop. Bottom row: the Random support
#' baseline (TPR / FDR vs rho, one line per solver), the reference against
#' which the deterministic capped spacing is read.
plot_gap_rho <- function(df, title, tfdr = DEFAULT_TFDR) {
  solvers <- sort(unique(df$solver), method = "radix")
  method_txt <- if (length(solvers) > 1)
    paste0("mean over ", paste(solvers, collapse = " / ")) else solvers[[1]]

  capped <- df[df$support_type == "CappedSpread", ]
  rho_vals <- sort(unique(capped$rho))
  gap_vals <- sort(unique(capped$gap))

  # kappa-boundary curve in discrete-axis positions: tile i sits at position
  # i, so rho / gap values are mapped onto their level indices.
  fine_rho <- seq(max(min(rho_vals), 1e-3), max(rho_vals), length.out = 200)
  kap <- kappa_boundary(fine_rho)
  overlay <- data.frame(
    x = stats::approx(rho_vals, seq_along(rho_vals), xout = fine_rho,
                      rule = 2)$y,
    y = stats::approx(gap_vals, seq_along(gap_vals), xout = kap,
                      rule = 1)$y
  )

  heat <- function(metric, pal, vmin, vmax, panel_title, fill_name) {
    # Multiple solvers share each (gap, rho) cell; the heatmaps show the mean
    # over the DA solvers (mirrors the Python pivot_table aggfunc="mean").
    sub <- capped[capped$metric == metric, ]
    agg <- stats::aggregate(value ~ gap + rho, data = sub, FUN = mean)
    names(agg) <- c("row", "col", "value")
    if (is.null(vmax)) vmax <- max(agg$value, na.rm = TRUE)
    .heat_panel(agg, rho_vals, gap_vals, pal, vmin, vmax, panel_title,
                x_lab = "AR(1) correlation ρ",
                y_lab = "CappedSpread gap between active vars",
                fill_name = fill_name, overlay = overlay)
  }
  h_tpp <- heat("mean_TPP", viridisLite::viridis(256), 0, 1,
                "TPR (true positive rate)", "TPR")
  h_fdp <- heat("mean_FDP", viridisLite::magma(256, direction = -1), 0, NULL,
                "FDR (false discovery rate)", "FDR")

  # --- Random support baseline (1D vs rho) --------------------------------
  rnd <- df[df$support_type == "Random", ]
  rnd_rho <- sort(unique(rnd$rho))
  cols <- .method_cols(solvers)
  shps <- .method_shps(solvers)
  base_panel <- function(metric, panel_title, y_lab, ylim, tfdr0) {
    sub <- rnd[rnd$metric == metric, ]
    sub <- sub[order(match(sub$solver, solvers), sub$rho), ]
    sub$solver <- factor(sub$solver, levels = solvers)
    p <- ggplot(sub, aes(x = rho, y = value, colour = solver, shape = solver))
    if (tfdr0 > 0) {
      lab <- sprintf("tFDR = %g", tfdr0)
      p <- p +
        geom_hline(aes(yintercept = tfdr0, linetype = lab),
                   color = "black", linewidth = 0.8) +
        scale_linetype_manual(name = NULL, values = setNames("dotted", lab),
                              guide = guide_legend(order = 1))
    }
    span <- max(rnd_rho) - min(rnd_rho)
    pad <- if (span > 0) span * 0.04 else 0.5
    key_ltys <- list(linetype = .method_ltys(solvers))
    p +
      .method_lines(sub) +
      geom_point(size = 2.4, fill = NA, stroke = 0.9) +
      scale_color_manual(name = "Method", values = cols) +
      scale_shape_manual(name = "Method", values = shps) +
      guides(colour = guide_legend(order = 2, override.aes = key_ltys),
             shape = guide_legend(order = 2)) +
      labs(title = panel_title, x = "AR(1) correlation ρ", y = y_lab) +
      scale_x_continuous(breaks = rnd_rho, labels = .fmt_g,
                         limits = c(min(rnd_rho) - pad, max(rnd_rho) + pad),
                         expand = expansion(0)) +
      .panel_theme() +
      coord_cartesian(ylim = ylim)
  }
  fdp_max <- suppressWarnings(
    max(rnd$value[rnd$metric == "mean_FDP"], na.rm = TRUE)
  )
  if (!is.finite(fdp_max)) fdp_max <- 0
  b_tpr <- base_panel("mean_TPP", "Random support baseline — TPR",
                      "True positive rate", c(-0.02, 1.02), 0)
  b_fdp <- base_panel("mean_FDP", "Random support baseline — FDR",
                      "False discovery rate",
                      c(0, max(fdp_max, tfdr) * Y_HEADROOM), tfdr)

  caption <- paste0(
    "cyan dashed curve: κ-boundary  gap = ⌈log ε / log ρ",
    "⌉, ε = ", .fmt_g(GAP_RHO_EPS),
    " (DA window, same for all solvers)"
  )
  fig <- ((h_tpp | h_fdp) / (b_tpr | b_fdp)) +
    plot_layout(guides = "collect") +
    plot_annotation(
      title = paste0(title, "\n(", method_txt, ")"),
      caption = caption,
      theme = theme(
        plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        plot.caption = element_text(size = 10, colour = "darkcyan",
                                    hjust = 0.5)
      )
    )
  list(fig = fig, width = 14.0, height = 10.5)
}


# ------------------------------------------------------------------------------
# Pattern 5 — generic two-support 2D sweep (demo-07 NN studies)
# ------------------------------------------------------------------------------
# Demo 07 records its 2D Monte Carlo studies (kappa x rho on NN data, and the
# SNR x rho method-mismatch study on AR(1) data) in the tidy two-support schema
#     solver,support_type,metric,<ROW>,<COL>,value
# with metrics mean_TPP / sd_TPP / mean_FDP / sd_FDP on a full 2D grid for
# every (solver, support_type) pair. One figure per support placement:
# TPR / FDR heatmap rows x one column per solver, FDR cells that exceed the
# tFDR target outlined in cyan.

SUPPORT_SUFFIX <- c(cappedspread = "capped", random = "random")

#' Load a tidy two-support 2D sweep CSV. The 4th and 5th columns are the two
#' swept quantities (row / column axes of the heatmaps); their names vary
#' between studies, so they are detected positionally and returned alongside
#' the frame. Returns list(df, row_name, col_name) with df columns
#' solver, support_type, metric, row, col, value.
read_sweep2d <- function(path) {
  if (!file.exists(path))
    stop("Input file not found:\n  ", path,
         "\nPass the 2D sweep results CSV path as the first argument.",
         call. = FALSE)
  df <- utils::read.csv(path, check.names = FALSE, strip.white = TRUE,
                        stringsAsFactors = FALSE)
  if (ncol(df) != 6)
    stop("CSV must have 6 columns (solver,support_type,metric,<ROW>,<COL>,",
         "value); found columns: ", paste(names(df), collapse = ", "),
         call. = FALSE)
  row_name <- names(df)[4]
  col_name <- names(df)[5]
  names(df) <- c("solver", "support_type", "metric", "row", "col", "value")
  for (colnm in c("solver", "support_type", "metric"))
    df[[colnm]] <- trimws(as.character(df[[colnm]]))
  for (colnm in c("row", "col", "value"))
    df[[colnm]] <- as.numeric(df[[colnm]])
  required <- c("mean_TPP", "mean_FDP")
  missing <- setdiff(required, unique(df$metric))
  if (length(missing) > 0)
    stop("CSV is missing required metric(s): ",
         paste(sort(missing), collapse = ", "),
         "\nFound metrics: ", paste(sort(unique(df$metric)), collapse = ", "),
         call. = FALSE)
  list(df = annotate_solver_params(df), row_name = row_name,
       col_name = col_name)
}

#' Pattern 5 — one support placement of a two-support 2D sweep.
#'
#' A 2 x n_solvers panel grid: rows = TPR / FDR heatmaps over the 2D sweep
#' (row quantity on y, column quantity on x), one column per solver, on shared
#' colour scales (TPR fixed to [0, 1]; FDR shared across the whole CSV so the
#' CappedSpread and Random figures are directly comparable). FDR cells above
#' the tFDR target are outlined in cyan.
plot_sweep2d <- function(df, support, title, row_label, col_label,
                         tfdr = DEFAULT_TFDR) {
  solvers <- unique(df$solver)
  sub_all <- df[df$support_type == support, ]
  # Shared FDR scale across supports and solvers (never below tFDR, so the
  # violation outlines sit inside the mapped range).
  fdr_top <- max(df$value[df$metric == "mean_FDP"], tfdr, na.rm = TRUE)

  n_col <- length(unique(sub_all$col))
  fig_w <- max(11.5, 1.8 + length(solvers) * (1.6 + 0.42 * n_col))

  specs <- list(
    list(metric = "mean_TPP", mlabel = "TPR (true positive rate)",
         pal = viridisLite::viridis(256), vmin = 0, vmax = 1),
    list(metric = "mean_FDP", mlabel = "FDR (false discovery rate)",
         pal = viridisLite::magma(256, direction = -1), vmin = 0,
         vmax = fdr_top)
  )

  rows <- lapply(seq_along(specs), function(mi) {
    sp <- specs[[mi]]
    panels <- lapply(seq_along(solvers), function(si) {
      sub <- sub_all[sub_all$solver == solvers[[si]] &
                       sub_all$metric == sp$metric, ]
      sub <- sub[, c("row", "col", "value")]
      # The metric heads every panel of its row; the solver tops the figure
      # columns. The y-axis stays reserved for the row quantity.
      head_txt <- if (mi == 1) paste0(solvers[[si]], "\n", sp$mlabel) else sp$mlabel
      .heat_panel(sub, sort(unique(sub$col)), sort(unique(sub$row)),
                  sp$pal, sp$vmin, sp$vmax, head_txt,
                  x_lab = col_label, y_lab = row_label,
                  tfdr_outline = if (sp$metric == "mean_FDP") tfdr else 0,
                  show_y = si == 1)
    })
    # One shared colorbar per metric row (all panels share the fill scale).
    wrap_plots(panels, nrow = 1) + plot_layout(guides = "collect")
  })

  caption <- if (tfdr > 0)
    sprintf("cyan boxes: realized FDR above the tFDR = %g target", tfdr) else NULL
  fig <- (rows[[1]] / rows[[2]]) +
    plot_annotation(
      title = paste0(title, "\n(", support, " support)"),
      caption = caption,
      theme = theme(
        plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        plot.caption = element_text(size = 10, colour = "darkcyan",
                                    hjust = 0.5)
      )
    )
  list(fig = fig, width = fig_w, height = 9.0)
}


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

.USAGE <- paste0(
  "Usage:\n",
  "  Rscript trex_da_plt_utils.R <csv> [--title T] [--legend-title L]\n",
  "      [--tfdr F] [--bands] [--xlabel L] [--xscale linear|linear-minor|",
  "sqrt|log]\n",
  "      [--relabel OLD=NEW ...] [--outdir DIR] [--stem S] [--formats png pdf]\n",
  "  Rscript trex_da_plt_utils.R grid --labels L1 L2 ... --csvs C1 C2 ...\n",
  "      --stem S --outdir DIR [--title T] [--legend-title L]\n",
  "      [--column-title C] [--tfdr F] [--bands] [--xlabel L] [--xscale ...]\n",
  "      [--relabel OLD=NEW ...] [--formats png pdf]\n",
  "  Rscript trex_da_plt_utils.R gaprho <csv> --stem S --outdir DIR\n",
  "      [--title T] [--tfdr F] [--formats png pdf]\n",
  "  Rscript trex_da_plt_utils.R sweep2d <csv> --stem S --outdir DIR\n",
  "      [--title T] [--xlabel L] [--ylabel L] [--tfdr F] [--formats png pdf]"
)

.parse_args <- function(argv) {
  args <- list(csv = character(0), title = NULL,
               legend_title = DEFAULT_LEGEND_TITLE, outdir = NULL,
               stem = NULL, formats = c("png", "pdf"), tfdr = DEFAULT_TFDR,
               bands = FALSE, xlabel = NULL, ylabel = NULL, xscale = NULL,
               relabel = character(0), labels = NULL, csvs = NULL,
               column_title = "")
  i <- 1L
  acc <- environment()  # the argument cursor i that take() and take_multi() advance
  take <- function() {
    acc$i <- i + 1L
    if (i > length(argv)) stop("missing value for ", argv[[i - 1L]],
                               call. = FALSE)
    argv[[i]]
  }
  # Python nargs='+' style: consume all following non-flag tokens.
  take_multi <- function(flag_name) {
    vals <- character(0)
    while (i < length(argv) && !startsWith(argv[[i + 1L]], "--")) {
      vals <- c(vals, argv[[i + 1L]])
      acc$i <- i + 1L
    }
    if (length(vals) == 0)
      stop(flag_name, " requires at least one value", call. = FALSE)
    vals
  }
  while (i <= length(argv)) {
    a <- argv[[i]]
    switch(a,
           "--title"        = {
             args$title <- take()
           },
           "--legend-title" = {
             args$legend_title <- take()
           },
           "--outdir"       = {
             args$outdir <- take()
           },
           "--stem"         = {
             args$stem <- take()
           },
           "--formats"      = {
             args$formats <- unlist(strsplit(take_multi("--formats"), ","))
           },
           "--tfdr"         = {
             args$tfdr <- as.numeric(take())
           },
           "--bands"        = {
             args$bands <- TRUE
           },
           "--xlabel"       = {
             args$xlabel <- take()
           },
           "--ylabel"       = {
             args$ylabel <- take()
           },
           "--xscale"       = {
             args$xscale <- take()
           },
           "--relabel"      = {
             args$relabel <- take_multi("--relabel")
           },
           "--labels"       = {
             args$labels <- take_multi("--labels")
           },
           "--csvs"         = {
             args$csvs <- take_multi("--csvs")
           },
           "--column-title" = {
             args$column_title <- take()
           },
           # Accepted for CLI parity with the Python plotter; the R port never
           # produces the interactive Plotly HTML.
           "--no-plotly"    = { },
           "--plotly-cdn"   = { },
           {
             if (startsWith(a, "--"))
               stop("Unknown flag: ", a, "\n", .USAGE, call. = FALSE)
             args$csv <- c(args$csv, a)
           })
    i <- i + 1L
  }
  if (!is.null(args$xscale) &&
        !args$xscale %in% c("linear", "linear-minor", "sqrt", "log"))
    stop("--xscale must be one of linear, linear-minor, sqrt, log; got: ",
         args$xscale, call. = FALSE)
  if (any(is.na(args$tfdr)))
    stop("--tfdr must be numeric", call. = FALSE)
  args
}

.require_one_csv <- function(args, mode) {
  if (length(args$csv) != 1L)
    stop(mode, " mode expects exactly one CSV path.\n", .USAGE, call. = FALSE)
  args$csv[[1]]
}

.require_out <- function(args, mode) {
  if (is.null(args$stem) || is.null(args$outdir))
    stop(mode, " mode requires --stem and --outdir.\n", .USAGE, call. = FALSE)
}

main_fdr_tpr <- function(args) {
  csv_path <- .require_one_csv(args, "default")
  res <- read_mc_results(csv_path)
  df <- apply_relabel(res$df, args$relabel)
  csv_stem <- sub("[.]csv$", "", basename(csv_path))
  title <- args$title %||%
    sprintf("DA-TRex Monte Carlo simulation (%s)", csv_stem)
  outdir <- args$outdir %||% default_plots_dir(csv_path)
  stem <- args$stem %||% paste0(csv_stem, STEM_SUFFIX)

  out <- plot_fdr_tpr_vs_sweep(df, res$sweep_key, title, tfdr = args$tfdr,
                               legend_title = args$legend_title,
                               bands = args$bands, xlabel = args$xlabel,
                               xscale = args$xscale)
  save_figure(out$fig, outdir, stem, args$formats, out$width, out$height)
}

main_grid <- function(args) {
  if (is.null(args$labels) || is.null(args$csvs))
    stop("grid mode requires --labels and --csvs.\n", .USAGE, call. = FALSE)
  if (length(args$labels) != length(args$csvs))
    stop("--labels (", length(args$labels), ") and --csvs (",
         length(args$csvs), ") must have the same length.", call. = FALSE)
  .require_out(args, "grid")

  frames <- list()
  sweep_keys <- character(0)
  for (k in seq_along(args$labels)) {
    csv <- args$csvs[[k]]
    if (!file.exists(csv)) {
      # A listed CSV that does not exist yet is skipped with a note, so the
      # grid renders whatever runs are available.
      cat(sprintf("note: skipping %s - CSV not found:\n  %s\n",
                  args$labels[[k]], csv))
      next
    }
    res <- read_mc_results(csv)
    frames[[args$labels[[k]]]] <- apply_relabel(res$df, args$relabel)
    sweep_keys <- union(sweep_keys, res$sweep_key)
  }
  if (length(frames) == 0)
    stop("No input CSVs found - run the demo first, then re-run this plotter.",
         call. = FALSE)
  if (length(sweep_keys) > 1)
    stop("Grid columns sweep different quantities (",
         paste(sort(sweep_keys), collapse = ", "),
         "); all --csvs must share one sweep axis.", call. = FALSE)

  out <- plot_comparison_grid(
    frames, sweep_keys[[1]], args$title %||% "DA-TRex comparison",
    legend_title = args$legend_title, tfdr = args$tfdr,
    column_title = args$column_title, bands = args$bands,
    xlabel = args$xlabel, xscale = args$xscale
  )
  save_figure(out$fig, args$outdir, args$stem, args$formats,
              out$width, out$height)
}

main_gaprho <- function(args) {
  csv_path <- .require_one_csv(args, "gaprho")
  .require_out(args, "gaprho")
  df <- read_gap_rho(csv_path)
  title <- args$title %||%
    "DA-TRex AR(1): support spacing (gap) vs correlation (ρ)"
  out <- plot_gap_rho(df, title, tfdr = args$tfdr)
  save_figure(out$fig, args$outdir, args$stem, args$formats,
              out$width, out$height)
}

main_sweep2d <- function(args) {
  csv_path <- .require_one_csv(args, "sweep2d")
  .require_out(args, "sweep2d")
  res <- read_sweep2d(csv_path)
  row_label <- args$ylabel %||% sweep_config(sweep_key_of(res$row_name))$label
  col_label <- args$xlabel %||% sweep_config(sweep_key_of(res$col_name))$label
  title <- args$title %||% "DA-TRex 2D sweep"

  for (support in unique(res$df$support_type)) {
    out <- plot_sweep2d(res$df, support, title, row_label, col_label,
                        tfdr = args$tfdr)
    suffix <- SUPPORT_SUFFIX[tolower(support)]
    if (is.na(suffix)) suffix <- tolower(support)
    save_figure(out$fig, args$outdir, paste0(args$stem, "_", suffix),
                args$formats, out$width, out$height)
  }
}

main <- function() {
  argv <- commandArgs(trailingOnly = TRUE)
  # Dispatch on the first token: grid / gaprho / sweep2d sub-modes, else the
  # default per-CSV FDR/TPR mode (so bare `trex_da_plt_utils.R <csv> ...`
  # works unchanged for the single-CSV demos).
  mode <- "fdr_tpr"
  if (length(argv) >= 1 && argv[[1]] %in% c("grid", "gaprho", "sweep2d")) {
    mode <- argv[[1]]
    argv <- argv[-1]
  }
  args <- .parse_args(argv)
  switch(mode,
         fdr_tpr = main_fdr_tpr(args),
         grid    = main_grid(args),
         gaprho  = main_gaprho(args),
         sweep2d = main_sweep2d(args))
  invisible(0L)
}

if (sys.nframe() == 0L) main()
