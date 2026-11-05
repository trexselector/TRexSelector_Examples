#!/usr/bin/env Rscript
# ==============================================================================
# ts_plt_utils.R
# ==============================================================================
#
# Shared plotting utilities for the T-solver demo suite, the ggplot2 port of
# cpp/tsolvers/ts_plt_utils.py and kept visually aligned with it. Each demo's
# generate_plots.sh supplies its CSV files and titles, so no per-demo
# configuration is needed.
#
# Plot patterns hosted here:
#     plot_terminated_paths()  — coefficient paths vs. number of included dummies,
#                                one panel per T_stop of a warm-started solver
#                                (the figure of the R tlars vignette).
#     plot_path_comparison()   — the same paths for several variants of one
#                                solver (T-AFS rho values, T-NCGMP variants), one
#                                panel per variant at its largest T_stop.
#     plot_fdr_tpr_vs_T()      — FDR and TPR vs. T per solver, one column per
#                                solver family (demo_ts_perf_01).
#     plot_tpr_vs_fdr()        — TPR vs. FDR along the T grid, same layout.
#     plot_path_stats()        — selected variables, steps and cumulative time
#                                vs. T, same layout, with the exchangeability
#                                line and the step limit as references.
#     plot_refit_errors()      — prediction RMSE and coefficient error of the
#                                least-squares refit on the selected variables
#                                vs. T, with the refit on the true support.
#     plot_runtime()           — time to T, steps and time per step vs. the
#                                number of dummies L (demo_ts_perf_02), with a
#                                reference line proportional to p + L.
#     plot_warm_start()        — fresh run to T + 1 over the warm start T -> T + 1.
#     plot_memory()            — solver memory and its share of the data vs. n/p
#                                (demo_ts_perf_03), with the data size.
#     plot_recovery()          — exact recovery, TPR or FDR at T = 1 vs.
#                                n/(s log p), one panel per solver, one curve
#                                per p (demo_ts_perf_04, Part 1).
#     plot_phase_transition()  — exact recovery over n/p and s/n, one heatmap
#                                per solver, with the l1 phase transition and
#                                the LARS s-step bound (demo_ts_perf_04, Part 2).
#     plot_null_pmf()          — histogram of the included null variables at T
#                                per solver with the negative hypergeometric
#                                (NHG) PMF and both means (demo_ts_perf_05).
#     plot_null_qq()           — Q-Q plot of the same counts against the NHG.
#     plot_selected_ecdf()     — ECDFs of the selected null variables V(v) at
#                                the voting levels, with the NHG CDF.
#
# CLI modes (typically driven from a demo's generate_plots.sh):
#     Rscript ts_plt_utils.R paths <stem>_path.csv --solver T-LARS --title "..."
#     Rscript ts_plt_utils.R compare --csvs a_path.csv b_path.csv \
#         --labels '$\rho = 0.1$' '$\rho = 1$' --title "..." --stem my_stem
#     Rscript ts_plt_utils.R fdr-tpr fdr_tpr_vs_T_mc500.csv --title "..." \
#         [--rank 99 --p 300 --L 300 --s 10]
#     Rscript ts_plt_utils.R runtime runtime_vs_dummies_mc500.csv --title "..." --p 1000
#     Rscript ts_plt_utils.R memory memory_vs_np_mc500.csv --title "..."
#     Rscript ts_plt_utils.R recovery recovery_vs_dt_ratio_mc500.csv \
#         --phase phase_transition_mc500.csv --title "..."
#     Rscript ts_plt_utils.R nhg null_counts_vs_nhg_mc500.csv --title "..."
#
# Flags: --title T, --stem S, --outdir DIR, --formats png pdf (or png,pdf).
# Titles and labels take the $...$ math of the Python plotter, translated to
# plotmath by latex_title() (letters italic, Greek letters, subscripts, \to,
# \propto, norms \|...\|, accents such as \hat{...}, function names like \log).
#
# The path modes read <stem>_path.csv (T_stop,step,index,is_dummy,beta) and
# the sibling <stem>_actions.csv (T_stop,step,index,is_dummy,added) written by
# path_recorder() of ts_demo_utils.R, with 1-based indices. The performance
# modes read the result CSVs of demo_ts_perf_01 to _05, whose columns match the
# C++ and Python demos, and write <csv stem>_<figure> (the phase CSV stem for
# the heatmaps). Figures default to the simulation_results/plots/ sibling of
# the data folder; override with --outdir. PDFs carry no creation date, so an
# unchanged figure stays byte-identical.
# ==============================================================================

suppressMessages({
  library(ggplot2)
  library(patchwork)
})

# Colours and line styles of the tlars vignette path plot
COLOR_ORIGINAL <- "black"
COLOR_DUMMY <- "#d62728"
COLOR_STEP <- "grey65"
MAX_TOP_LABELS <- 40L  # beyond this, the top axis labels get small and rotated

# Line widths of the Python plotter in points, as ggplot2 linewidths (grid lwd is 1/96 inch)
.lw <- function(points) points * 96 / 72 / ggplot2::.pt


# ==============================================================================
# Shared helpers
# ==============================================================================

#' simulation_results/plots/ next to simulation_results/data/, else the CSV's folder.
default_plots_dir <- function(csv_path) {
  parent <- dirname(normalizePath(csv_path, mustWork = FALSE))
  if (basename(parent) == "data" && basename(dirname(parent)) == "simulation_results")
    file.path(dirname(parent), "plots")
  else parent
}

#' PDF device without a creation date, so an unchanged figure stays byte-identical.
.pdf_device <- function(filename, ...) {
  grDevices::pdf(file = filename, ..., timestamp = FALSE, useDingbats = FALSE)
}

#' Write the figure to outdir/<stem>.<ext> for each requested format.
save_figure <- function(plot, outdir, stem, formats = c("png", "pdf"), width = 10, height = 6) {
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  for (ext in formats) {
    path <- file.path(outdir, paste0(stem, ".", ext))
    if (ext == "pdf") {
      ggsave(path, plot = plot, width = width, height = height, device = .pdf_device, bg = "white")
    } else {
      ggsave(path, plot = plot, width = width, height = height, dpi = 300, bg = "white")
    }
    cat(sprintf("Wrote: %s\n", path))
  }
}


# ==============================================================================
# $...$ math to plotmath
# ==============================================================================

GREEK <- c("alpha", "beta", "gamma", "delta", "epsilon", "eta", "theta", "kappa", "lambda",
           "mu", "nu", "pi", "rho", "sigma", "tau", "phi", "chi", "psi", "omega")
# Binary operators of the math, as plotmath infix operators
MATH_OPERATORS <- c(to = "%->%", le = "<=", leq = "<=", ge = ">=", geq = ">=",
                    times = "%*%", cdot = "%.%", approx = "%~~%", propto = "%prop%")
# Function names, set upright with a space on both sides, and accents over a {...} argument
MATH_FUNCTIONS <- c("log", "ln", "exp")
MATH_ACCENTS <- c("hat", "bar", "tilde")

#' One plotmath expression of a $...$ segment (a single piece stays as it is).
.math_group <- function(math) {
  pieces <- .math_pieces(math)
  if (length(pieces) == 1L) pieces[[1L]] else as.call(c(as.name("paste"), pieces))
}

#' Plotmath pieces of one $...$ segment: letters italic, Greek letters and
#' subscripts as symbols, numbers and operators upright, relations spaced.
.math_pieces <- function(math) {
  math <- gsub("{,}", ",", math, fixed = TRUE)
  pattern <- paste0(
    "\\\\\\|.*?\\\\\\||\\\\(?:", paste(MATH_ACCENTS, collapse = "|"), ")\\{[^}]*\\}|",
    "\\\\[A-Za-z]+|[A-Za-z]|[0-9.,]+|_\\{[^}]*\\}|_[A-Za-z0-9]|\\S"
  )
  tokens <- regmatches(math, gregexpr(pattern, math, perl = TRUE))[[1]]
  pieces <- list()
  ops <- character()  # operator name before the piece at the same position
  pending <- ""
  for (tok in tokens) {
    if (startsWith(tok, "_")) {
      sub <- gsub("^_\\{?|\\}$", "", tok)
      pieces[[length(pieces)]] <- call("[", pieces[[length(pieces)]], sub)
      next
    }
    if (startsWith(tok, "\\|")) {
      # \|...\| as a norm, its double bars nested single ones (plotmath draws "||" as one bar)
      inner <- call("group", "|", .math_group(substring(tok, 3L, nchar(tok) - 2L)), "|")
      piece <- call("group", "|", inner, "|")
    } else if (grepl("^\\\\[A-Za-z]+\\{", tok)) {
      piece <- call(sub("^\\\\([A-Za-z]+)\\{.*$", "\\1", tok), .math_group(sub("^[^{]*\\{(.*)\\}$", "\\1", tok)))
    } else if (startsWith(tok, "\\")) {
      name <- substring(tok, 2L)
      if (name %in% names(MATH_OPERATORS)) {
        pending <- MATH_OPERATORS[[name]]
        next
      }
      # Greek letters as symbols, \ell as an italic l, \log and other names upright
      piece <- if (name %in% GREEK) {
        as.name(name)
      } else if (name %in% MATH_FUNCTIONS) {
        paste0(" ", name, " ")
      } else if (name == "ell") {
        call("italic", "l")
      } else {
        name
      }
    } else if (grepl("^[A-Za-z]$", tok)) {
      piece <- call("italic", tok)
    } else if (tok %in% c("=", "<", ">", "+", "-")) {
      piece <- paste0(" ", tok, " ")
    } else {
      piece <- tok
    }
    pieces[[length(pieces) + 1L]] <- piece
    ops[length(pieces)] <- pending
    pending <- ""
  }
  # Fold the binary operators left to right: a \to b \to c -> (a %->% b) %->% c,
  # and a leading one (\propto p) gets an empty left side
  out <- list()
  for (k in seq_along(pieces)) {
    if (!is.na(ops[k]) && nzchar(ops[k])) {
      left <- if (length(out) > 0L) out[[length(out)]] else ""
      out[[max(1L, length(out))]] <- call(ops[k], left, pieces[[k]])
    } else {
      out[[length(out) + 1L]] <- pieces[[k]]
    }
  }
  out
}

#' A title with $...$ math as a plotmath expression, a plain string without math.
latex_title <- function(text) {
  if (is.null(text) || !grepl("$", text, fixed = TRUE)) return(text)
  parts <- strsplit(text, "$", fixed = TRUE)[[1]]
  pieces <- list()
  for (k in seq_along(parts)) {
    if (k %% 2L == 0L) {
      pieces <- c(pieces, .math_pieces(parts[[k]]))
    } else if (nzchar(parts[[k]])) {
      pieces <- c(pieces, list(parts[[k]]))
    }
  }
  as.call(c(as.name("paste"), pieces))
}


# ==============================================================================
# Terminated solution paths
# ==============================================================================

#' Read <stem>_path.csv and its sibling <stem>_actions.csv.
read_paths <- function(path_csv) {
  actions_csv <- sub("_path\\.csv$", "_actions.csv", path_csv)
  if (actions_csv == path_csv || !file.exists(actions_csv))
    stop("no actions file next to ", path_csv)
  list(path = utils::read.csv(path_csv), actions = utils::read.csv(actions_csv))
}

#' Theme of one path panel: box, dashed horizontal grid, title on the left.
.path_theme <- function(small_top) {
  top_text <- if (small_top) {
    element_text(size = 6, angle = 90, vjust = 0.5, hjust = 0)
  } else {
    element_text(size = 8)
  }
  theme_bw(base_size = 11) +
    theme(
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(colour = "grey80", linewidth = .lw(0.8), linetype = "dashed"),
      plot.title = element_text(size = 11, hjust = 0),
      axis.title = element_text(size = 10),
      axis.text = element_text(size = 9),
      axis.title.x.top = element_text(size = 9),
      axis.text.x.top = top_text,
      legend.title = element_blank(),
      legend.background = element_rect(colour = "grey70", linewidth = .lw(0.8)),
      legend.key.width = grid::unit(2.2, "lines")
    )
}

#' One panel: coefficient paths up to the T_stop-th included dummy.
.path_panel <- function(path, actions, T_stop, label = NULL) {
  num_steps <- max(actions$step)

  # Every variable that is ever nonzero, as a dense line over all steps
  indices <- unique(path$index)
  wide <- expand.grid(step = 0:num_steps, index = indices)
  wide <- merge(wide, path[, c("step", "index", "beta")], all.x = TRUE)
  wide$beta[is.na(wide$beta)] <- 0
  dummy_of <- tapply(path$is_dummy, path$index, `[`, 1L)
  wide$kind <- factor(ifelse(dummy_of[as.character(wide$index)] == 1L, "Dummies", "Original variables"),
                      levels = c("Original variables", "Dummies"))

  dummy_steps <- actions$step[actions$is_dummy == 1L & actions$added == 1L]
  added <- actions[actions$added == 1L, ]
  top_labels <- ifelse(added$is_dummy == 1L, "D", as.character(added$index))

  # Margin on both sides, so the curves visibly start from 0 at step 0
  margin <- max(0.5, 0.02 * num_steps)
  num_dummies <- length(dummy_steps)
  heading <- sprintf(" = %d: %d steps, %d %s included", T_stop, num_steps, num_dummies,
                     if (num_dummies == 1L) "dummy" else "dummies")
  title <- as.call(c(as.name("paste"), if (!is.null(label)) list(label, ", "),
                     list(quote(italic("T")), heading)))

  ggplot(wide, aes(step, beta, group = index, colour = kind, linetype = kind)) +
    geom_vline(xintercept = seq_len(num_steps), colour = COLOR_STEP, linetype = "5212",
               linewidth = .lw(0.6)) +
    geom_vline(xintercept = dummy_steps, colour = COLOR_DUMMY, linewidth = .lw(1.3)) +
    geom_hline(yintercept = 0, colour = COLOR_ORIGINAL, linewidth = .lw(0.8)) +
    geom_line(data = wide[wide$kind == "Original variables", ], linewidth = .lw(1.2)) +
    geom_line(data = wide[wide$kind == "Dummies", ], linewidth = .lw(1.2)) +
    scale_colour_manual(values = c("Original variables" = COLOR_ORIGINAL, "Dummies" = COLOR_DUMMY),
                        drop = FALSE) +
    scale_linetype_manual(values = c("Original variables" = "solid", "Dummies" = "42"), drop = FALSE) +
    scale_x_continuous(
      limits = c(-margin, num_steps + margin), expand = c(0, 0),
      breaks = dummy_steps, labels = seq_along(dummy_steps),
      sec.axis = dup_axis(name = "Index of the entering variable (1-based, D = dummy)",
                          breaks = added$step, labels = top_labels)
    ) +
    labs(title = title, x = "Number of included dummies", y = "Coefficients") +
    .path_theme(length(top_labels) > MAX_TOP_LABELS)
}

#' Panels stacked, with the title and the shared legend above them, so the
#' legend never covers a curve.
.finish_figure <- function(panels, title) {
  heights <- grid::unit(c(0.9, rep(1, length(panels))), c("cm", rep("null", length(panels))))
  wrap_plots(c(list(guide_area()), panels), ncol = 1L) +
    plot_layout(guides = "collect", heights = heights) +
    plot_annotation(title = latex_title(title),
                    theme = theme(plot.title = element_text(size = 13, hjust = 0.5))) &
    theme(legend.position = "top")
}

#' Coefficient paths vs. number of included dummies, one panel per T_stop.
plot_terminated_paths <- function(path, actions, solver, title = NULL) {
  T_values <- sort(unique(actions$T_stop))
  panels <- lapply(T_values, function(T_stop) {
    .path_panel(path[path$T_stop == T_stop, ], actions[actions$T_stop == T_stop, ], T_stop)
  })
  .finish_figure(panels, if (is.null(title)) paste0(solver, ": terminated solution paths") else title)
}

#' One panel per (label, path, actions) variant, each at its largest T_stop.
plot_path_comparison <- function(variants, title) {
  panels <- lapply(variants, function(v) {
    T_stop <- max(v$actions$T_stop)
    .path_panel(v$path[v$path$T_stop == T_stop, ], v$actions[v$actions$T_stop == T_stop, ], T_stop,
                latex_title(v$label))
  })
  .finish_figure(panels, title)
}


# ==============================================================================
# Solver styles, matplotlib markers and legends (demo_ts_perf_01 to _05)
# ==============================================================================

FAMILIES <- c("LARS-based", "OMP-based", "AFS-based")
MARKERS <- c("o", "s", "^", "D", "v", "P", "X")
# matplotlib's tab10 colour cycle
TAB10 <- c(
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
)
COLOR_REF <- "#595959"  # grey 0.35 of the reference lines
COLOR_GRID <- "#dfdfdf"  # matplotlib's grid colour #b0b0b0 at alpha 0.4 on white
COLOR_NOTE <- "#4d4d4d"  # grey 0.3 of the open-marker note
COLOR_FRAME <- "#b3b3b3"  # grey 0.7 of the legend frames
DASHED <- "42"  # matplotlib's "--" and ":" as dash patterns in multiples of the line width
DOTTED <- "12"
OPEN_MARKER_NOTE <- "Open markers: some runs ended before $T$ dummies entered (active set at its size limit)."
LEGEND_LOCS <- c(
  "upper right", "upper left", "lower left", "lower right", "right", "center left", "center right",
  "lower center", "upper center", "center"
)

#' Solver label with its parameter as math: "T-AFS (rho 0.1)" -> T-AFS (rho = 0.1), rho as a symbol.
display_name <- function(name) {
  name <- gsub("(rho ", "($\\rho$ = ", name, fixed = TRUE)
  latex_title(gsub("(lambda2 ", "($\\lambda_2$ = ", name, fixed = TRUE))
}

#' Colour and marker per solver, cycling within each family in file order.
.solver_styles <- function(df) {
  styles <- list()
  for (family in FAMILIES) {
    names <- unique(df$solver[df$family == family])
    for (k in seq_along(names)) {
      styles[[names[k]]] <- list(
        colour = TAB10[(k - 1L) %% length(TAB10) + 1L],
        marker = MARKERS[(k - 1L) %% length(MARKERS) + 1L]
      )
    }
  }
  styles
}

#' Outline of a matplotlib marker (o s ^ D v P X), in units of the marker size.
.marker_outline <- function(marker) {
  angle <- seq(0, 2 * pi, length.out = 49L)[-49L]
  switch(
    marker,
    o = cbind(cos(angle), sin(angle)) / 2,
    s = cbind(c(-1, 1, 1, -1), c(-1, -1, 1, 1)) / 2,
    "^" = cbind(c(0, -1, 1), c(1, -1, -1)) / 2,
    v = cbind(c(0, -1, 1), c(-1, 1, 1)) / 2,
    D = cbind(c(0, 1, 0, -1), c(-1, 0, 1, 0)) / sqrt(2),
    P = cbind(c(-1, 1, 1, 3, 3, 1, 1, -1, -1, -3, -3, -1), c(-3, -3, -1, -1, 1, 1, 3, 3, 1, 1, -1, -1)) / 6,
    X = cbind(c(-1, 0, 1, 2, 1, 2, 1, 0, -1, -2, -1, -2), c(-2, -1, -2, -1, 0, 1, 2, 1, 2, 1, 0, -1)) / 4,
    stop("unknown marker ", marker)
  )
}

#' Markers centred at the units x, y, with size and edge width in points.
.marker_grob <- function(x, y, marker, size, colour, fill, stroke) {
  if (length(x) == 0L) return(grid::nullGrob())
  outlines <- Map(function(m, s) .marker_outline(m) * s, rep_len(marker, length(x)), rep_len(size, length(x)))
  counts <- vapply(outlines, nrow, integer(1L))
  offsets <- do.call(rbind, outlines)
  grid::polygonGrob(
    x = rep(x, counts) + grid::unit(offsets[, 1L], "bigpts"),
    y = rep(y, counts) + grid::unit(offsets[, 2L], "bigpts"),
    id = rep(seq_along(x), counts),
    gp = grid::gpar(col = colour, fill = fill, lwd = stroke * 96 / 72, linejoin = "mitre")
  )
}

#' ggplot2 geom of matplotlib markers, so the plus (P) and x (X) markers can be open as well.
GeomMarker <- ggproto(
  "GeomMarker", Geom,
  required_aes = c("x", "y"),
  default_aes = aes(colour = "black", fill = "black", marker = "o", size = 6, stroke = 1),
  draw_panel = function(data, panel_params, coord) {
    coords <- coord$transform(data, panel_params)
    .marker_grob(
      grid::unit(coords$x, "npc"), grid::unit(coords$y, "npc"), coords$marker, coords$size, coords$colour,
      coords$fill, coords$stroke
    )
  }
)

#' Layer of matplotlib markers at the columns x and y of data.
geom_marker <- function(data, ...) {
  layer(
    geom = GeomMarker, stat = "identity", position = "identity", data = data, mapping = aes(x, y),
    params = list(...), inherit.aes = FALSE, show.legend = FALSE
  )
}

#' ggplot2 geom drawing a fixed grob over the whole panel, on log axes as well (the in-panel legends).
GeomPanelGrob <- ggproto(
  "GeomPanelGrob", Geom,
  draw_panel = function(data, panel_params, coord, grob) grob
)

#' Layer of a grob over the whole panel.
panel_grob <- function(grob) {
  layer(
    geom = GeomPanelGrob, stat = "identity", position = "identity", data = data.frame(panel = 1L),
    mapping = aes(), params = list(grob = grob), inherit.aes = FALSE, show.legend = FALSE
  )
}

#' Legend entry: a line of the given colour, dash pattern and width (points), with an optional marker.
.entry <- function(label, colour, linetype = "solid", linewidth = 1.6, marker = NA, size = 6) {
  list(label = label, colour = colour, linetype = linetype, linewidth = linewidth, marker = marker, size = size)
}

#' Legend in matplotlib's layout at loc of the current viewport. "best" takes the first location
#' with the fewest points and line vertices inside plus lines crossing it (npc points and lines).
.legend_grob <- function(entries, loc = "best", points = NULL, lines = list(), fontsize = 8, title = NULL,
                         fill = "white", text_colour = "black", alpha = 0.8) {
  grid::gTree(
    entries = entries, loc = loc, points = points, lines = lines, fontsize = fontsize, heading = title,
    fill = fill, text_colour = text_colour, alpha = alpha, cl = "mpl_legend"
  )
}

#' Label as grid draws it: plotmath calls as expressions.
.as_label <- function(label) if (is.language(label)) as.expression(label) else label

#' Width of a label in points.
.text_width <- function(label, gp) {
  grid::convertWidth(grid::grobWidth(grid::textGrob(.as_label(label), gp = gp)), "bigpts", valueOnly = TRUE)
}

#' Whether the polyline x, y enters the box (xmin, ymin, xmax, ymax), by Liang-Barsky clipping.
.line_hits_box <- function(x, y, box) {
  if (length(x) < 2L) return(FALSE)
  x0 <- x[-length(x)]
  y0 <- y[-length(y)]
  p <- list(-diff(x), diff(x), -diff(y), diff(y))
  q <- list(x0 - box[1L], box[3L] - x0, y0 - box[2L], box[4L] - y0)
  t0 <- rep(0, length(x0))
  t1 <- rep(1, length(x0))
  ok <- rep(TRUE, length(x0))
  for (k in seq_len(4L)) {
    ok <- ok & !(p[[k]] == 0 & q[[k]] < 0)
    r <- q[[k]] / p[[k]]
    t0 <- ifelse(p[[k]] < 0, pmax(t0, r), t0)
    t1 <- ifelse(p[[k]] > 0, pmin(t1, r), t1)
  }
  any(ok & t0 <= t1)
}

#' Lower-left corner (points) of a w x h legend in a panel_w x panel_h viewport, as matplotlib places it.
.legend_origin <- function(x, w, h, panel_w, panel_h) {
  pad <- 0.5 * x$fontsize  # borderaxespad
  xs <- c(right = panel_w - pad - w, left = pad, center = (panel_w - w) / 2)
  ys <- c(upper = panel_h - pad - h, lower = pad, center = (panel_h - h) / 2)
  horizontal <- c("right", "left", "left", "right", "right", "left", "right", "center", "center", "center")
  vertical <- c("upper", "upper", "lower", "lower", "center", "center", "center", "lower", "upper", "center")
  corners <- cbind(xs[horizontal], ys[vertical])
  if (x$loc != "best") return(corners[match(x$loc, LEGEND_LOCS), ])
  points <- if (is.null(x$points)) NULL else cbind(x$points$x * panel_w, x$points$y * panel_h)
  lines <- lapply(x$lines, function(l) cbind(l$x * panel_w, l$y * panel_h))
  badness <- apply(corners, 1L, function(corner) {
    box <- c(corner, corner + c(w, h))
    inside <- function(v) sum(v[, 1L] > box[1L] & v[, 1L] < box[3L] & v[, 2L] > box[2L] & v[, 2L] < box[4L])
    hits <- vapply(lines, function(l) inside(l) + .line_hits_box(l[, 1L], l[, 2L], box), numeric(1L))
    (if (is.null(points)) 0 else inside(points)) + sum(hits)
  })
  corners[which.min(badness), ]
}

#' Draw-time layout of a legend: frame, optional title, and one handle and label per entry.
.legend_content <- function(x) {
  fs <- x$fontsize
  pad <- 0.4 * fs  # borderpad
  row <- fs
  gap <- 0.5 * fs  # labelspacing
  handle <- 2 * fs
  text_pad <- 0.8 * fs
  gp_text <- grid::gpar(fontsize = fs, col = x$text_colour)
  widths <- vapply(x$entries, function(e) .text_width(e$label, gp_text), numeric(1L))
  title_w <- if (is.null(x$heading)) 0 else .text_width(x$heading, gp_text)
  title_h <- if (is.null(x$heading)) 0 else row + gap
  n <- length(x$entries)
  w <- 2 * pad + max(handle + text_pad + max(widths), title_w)
  h <- 2 * pad + title_h + n * row + (n - 1) * gap
  panel_w <- grid::convertWidth(grid::unit(1, "npc"), "bigpts", valueOnly = TRUE)
  panel_h <- grid::convertHeight(grid::unit(1, "npc"), "bigpts", valueOnly = TRUE)
  origin <- .legend_origin(x, w, h, panel_w, panel_h)
  u <- function(v) grid::unit(v, "bigpts")

  kids <- list(grid::roundrectGrob(
    u(origin[1L]), u(origin[2L]), u(w), u(h), just = c(0, 0), r = u(0.2 * fs),
    gp = grid::gpar(col = COLOR_FRAME, fill = x$fill, alpha = x$alpha, lwd = 96 / 72)
  ))
  top <- origin[2L] + h - pad
  if (!is.null(x$heading)) {
    kids <- c(kids, list(grid::textGrob(.as_label(x$heading), u(origin[1L] + w / 2), u(top - row / 2), gp = gp_text)))
    top <- top - row - gap
  }
  left <- origin[1L] + pad
  for (k in seq_len(n)) {
    e <- x$entries[[k]]
    mid <- top - (k - 1) * (row + gap) - row / 2
    gp_line <- grid::gpar(col = e$colour, lty = e$linetype, lwd = e$linewidth * 96 / 72, lineend = "butt")
    kids <- c(kids, list(grid::segmentsGrob(u(left), u(mid), u(left + handle), u(mid), gp = gp_line)))
    if (!is.na(e$marker)) {
      kids <- c(kids, list(.marker_grob(u(left + handle / 2), u(mid), e$marker, e$size, e$colour, e$colour, 1)))
    }
    label <- grid::textGrob(.as_label(e$label), u(left + handle + text_pad), u(mid), just = c(0, 0.5), gp = gp_text)
    kids <- c(kids, list(label))
  }
  grid::setChildren(x, do.call(grid::gList, kids))
}
registerS3method("makeContent", "mpl_legend", .legend_content, envir = asNamespace("grid"))


# ==============================================================================
# Axes and panels in the matplotlib style
# ==============================================================================

#' Axis of a panel: fixed range, ticks with labels, log or linear, optional top axis.
.axis <- function(limits, breaks, labels, log = FALSE, top = NULL) {
  list(limits = limits, breaks = breaks, labels = labels, log = log, top = top)
}

#' Data range plus matplotlib's 5% margins, taken in log10 space on a log axis.
.auto_limits <- function(values, log = FALSE) {
  v <- range(if (log) log10(values[values > 0]) else values, na.rm = TRUE)
  pad <- if (diff(v) > 0) 0.05 * diff(v) else 0.5
  v <- v + c(-pad, pad)
  if (log) 10^v else v
}

#' Ticks of matplotlib's MaxNLocator (steps 1, 2, 2.5, 5, 10) inside the range lim.
.linear_ticks <- function(lim, nbins = 9L) {
  raw <- diff(lim) / nbins
  steps <- c(0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20) * 10^floor(log10(raw))
  ticks <- numeric()
  for (step in rev(steps[seq_len(which(steps >= raw)[1L])])) {
    first <- ceiling(lim[1L] / step - 1e-10)
    last <- floor(lim[2L] / step + 1e-10)
    ticks <- if (last >= first) round(seq(first, last) * step, 10L) else numeric()
    if (length(ticks) >= 2L) break
  }
  ticks
}

#' Linear axis over lim with matplotlib's ticks.
.linear_axis <- function(lim) {
  ticks <- .linear_ticks(lim)
  .axis(lim, ticks, format(ticks, trim = TRUE))
}

#' Plain tick labels of a log axis: 20, 300, 1,000 instead of 2 x 10^1.
.plain_number <- function(v) {
  ifelse(v >= 1, formatC(v, format = "f", digits = 0L, big.mark = ","), sprintf("%g", v))
}

#' Log axis over lim with matplotlib's ticks. "sci": decades labelled 10^k. "plain": plain labels and
#' minor ticks at 2 to 9 times a decade (linear ticks if at most one is in range), labelled at 2, 3, 5.
.log_axis <- function(lim, style = c("sci", "plain")) {
  style <- match.arg(style)
  inside <- function(v) v[v >= lim[1L] * (1 - 1e-9) & v <= lim[2L] * (1 + 1e-9)]
  decades <- 10^seq(floor(log10(lim[1L])), ceiling(log10(lim[2L])))
  major <- inside(decades)
  if (style == "sci") {
    return(.axis(lim, major, parse(text = sprintf("10^%d", as.integer(round(log10(major))))), log = TRUE))
  }
  minor <- inside(as.vector(outer(2:9, decades)))
  if (length(minor) <= 1L) minor <- .linear_ticks(lim)
  minor <- minor[!vapply(minor, function(v) any(abs(v - major) < 1e-9 * v), logical(1L))]
  mantissa <- round(minor / 10^floor(log10(minor)))
  breaks <- c(major, minor)
  labels <- c(.plain_number(major), ifelse(mantissa %in% c(2, 3, 5), .plain_number(minor), ""))
  .axis(lim, breaks[order(breaks)], labels[order(breaks)], log = TRUE)
}

#' Positions of values on an axis in npc (0 to 1 across the panel).
.npc <- function(v, axis) {
  t <- if (axis$log) log10 else identity
  (t(v) - t(axis$limits[1L])) / (t(axis$limits[2L]) - t(axis$limits[1L]))
}

#' Position scale of an axis: its range without expansion, its ticks and labels, and the top axis.
.axis_scale <- function(axis, aesthetic) {
  scale <- if (aesthetic == "x") scale_x_continuous else scale_y_continuous
  top <- if (is.null(axis$top)) {
    waiver()
  } else {
    dup_axis(name = axis$top$title, breaks = axis$top$breaks, labels = axis$top$labels)
  }
  scale(
    transform = if (axis$log) "log10" else "identity", limits = axis$limits, breaks = axis$breaks,
    labels = axis$labels, expand = c(0, 0), oob = scales::oob_keep, sec.axis = top
  )
}

#' Panel theme of the matplotlib figures: black box, dashed grid, tick labels only where requested.
.series_theme <- function(show_x_text = TRUE, show_y_text = TRUE, grid = TRUE) {
  th <- theme_bw(base_size = 10) +
    theme(
      # The panel clips the outer half of its border, so 1.6 shows the 0.8 pt spines of matplotlib
      panel.border = element_rect(colour = "black", fill = NA, linewidth = .lw(1.6)),
      panel.grid.major = element_line(colour = COLOR_GRID, linewidth = .lw(0.8), linetype = DASHED),
      panel.grid.minor = element_blank(),
      axis.ticks = element_line(colour = "black", linewidth = .lw(0.8)),
      axis.ticks.length = grid::unit(3.5, "bigpts"),
      axis.text = element_text(size = 10, colour = "black"),
      axis.text.x.top = element_text(size = 8, colour = "black"),
      axis.title = element_text(size = 10),
      axis.title.x.top = element_text(size = 9),
      plot.title = element_text(size = 12, hjust = 0.5)
    )
  if (!grid) th <- th + theme(panel.grid.major = element_blank())
  if (!show_x_text) th <- th + theme(axis.text.x.bottom = element_blank())
  if (!show_y_text) th <- th + theme(axis.text.y.left = element_blank())
  th
}

#' Solver curves: lines, then markers filled where every run reached T ("reached") or always ("line").
.curve_layers <- function(curves, markers = "reached", linewidth = 1.6,
                          size = if (markers == "line") 6 else sqrt(34),
                          stroke = if (markers == "line") 1 else 1.2) {
  if (is.null(curves) || nrow(curves) == 0L) return(list())
  marker_layer <- function(pts) {
    geom_marker(
      pts, colour = pts$colour, fill = ifelse(pts$filled, pts$colour, "white"), marker = pts$marker,
      size = size, stroke = stroke
    )
  }
  groups <- unique(curves$solver)
  layers <- list()
  for (name in groups) {
    sub <- curves[curves$solver == name, ]
    line <- geom_path(data = sub, aes(x, y), colour = sub$colour[1L], linewidth = .lw(linewidth), lineend = "square")
    layers <- c(layers, list(line), if (markers == "line") list(marker_layer(sub)))
  }
  # Scatter markers above all lines, the filled ones of a solver before its open ones
  if (markers == "reached") {
    layers <- c(layers, list(marker_layer(curves[order(match(curves$solver, groups), !curves$filled), ])))
  }
  layers
}

#' Reference lines below the curves: a line through x, y, or a horizontal line at y without x.
.ref_layers <- function(refs) {
  lapply(refs, function(r) {
    colour <- r$colour %||% COLOR_REF
    linewidth <- .lw(r$linewidth %||% 1.2)
    if (is.null(r$x)) {
      geom_hline(yintercept = r$y, colour = colour, linetype = r$linetype, linewidth = linewidth)
    } else {
      geom_path(data = data.frame(x = r$x, y = r$y), aes(x, y), colour = colour, linetype = r$linetype,
                linewidth = linewidth)
    }
  })
}

#' Legend entries of the solvers in names, with their line and filled marker.
.solver_entries <- function(names, styles) {
  lapply(names, function(n) .entry(display_name(n), styles[[n]]$colour, marker = styles[[n]]$marker))
}

#' Legend entries of the labelled reference lines.
.ref_entries <- function(refs) {
  labelled <- Filter(function(r) !is.null(r$label), refs)
  lapply(labelled, function(r) {
    .entry(latex_title(r$label), r$colour %||% COLOR_REF, r$linetype, r$linewidth %||% 1.2)
  })
}

#' Legend layer of a panel at loc; for "best" it avoids the curves and reference lines of the panel.
.panel_legend <- function(entries, loc, curves, refs, x_axis, y_axis, markers = "reached", ...) {
  lines <- list()
  points <- NULL
  if (!is.null(curves) && nrow(curves) > 0L) {
    for (name in unique(curves$solver)) {
      sub <- curves[curves$solver == name, ]
      lines <- c(lines, list(list(x = .npc(sub$x, x_axis), y = .npc(sub$y, y_axis))))
    }
    if (markers == "reached") points <- list(x = .npc(curves$x, x_axis), y = .npc(curves$y, y_axis))
  }
  for (r in refs) {
    # A horizontal line spans the panel from edge to edge
    ref <- if (is.null(r$x)) {
      list(x = c(0, 1), y = rep(.npc(r$y, y_axis), 2L))
    } else {
      list(x = .npc(r$x, x_axis), y = .npc(r$y, y_axis))
    }
    lines <- c(lines, list(ref))
  }
  panel_grob(.legend_grob(entries, loc, points, lines, ...))
}

#' Figure of panels in a grid under the title, with the open-marker note below when note is TRUE;
#' the figure size in inches rides along as the attribute figure_size.
.assemble <- function(panels, ncol, title, size, note = FALSE, widths = NULL) {
  fig <- wrap_plots(panels, ncol = ncol, widths = widths) +
    plot_annotation(
      title = latex_title(title),
      caption = if (note) latex_title(OPEN_MARKER_NOTE),
      theme = theme(
        plot.title = element_text(size = 13, hjust = 0.5),
        plot.caption = element_text(size = 9, colour = COLOR_NOTE, hjust = 0.5)
      )
    )
  attr(fig, "figure_size") <- size
  fig
}


# ==============================================================================
# One column per solver family (demo_ts_perf_01 to _03)
# ==============================================================================

#' Curves of the solvers in fam, one row per point sorted by the column x$order, with the solver style.
.family_curves <- function(fam, x, row, styles) {
  parts <- lapply(unique(fam$solver), function(name) {
    sub <- fam[fam$solver == name, , drop = FALSE]
    sub <- sub[order(sub[[x$order]]), , drop = FALSE]
    data.frame(
      solver = name, x = x$value(sub), y = row$value(sub),
      filled = if ("reached" %in% names(sub)) sub$reached >= 0.99 else TRUE,
      colour = styles[[name]]$colour, marker = styles[[name]]$marker
    )
  })
  do.call(rbind, parts)
}

#' Panels over the families: a row per entry of rows, x shared by all, y shared along each row.
#' x: value(), order column, title, axis. Row: value(), axis (fixed linear range, "sci" or "plain"),
#' label, optional refs(fam) (list of lines), legend (loc, solvers) and top (axis on top).
.family_panels <- function(df, x, rows, markers = "reached") {
  styles <- .solver_styles(df)
  fams <- lapply(FAMILIES, function(f) df[df$family == f, , drop = FALSE])
  panels <- list()
  for (i in seq_along(rows)) {
    row <- rows[[i]]
    curves <- lapply(fams, .family_curves, x = x, row = row, styles = styles)
    refs <- lapply(fams, function(fam) if (is.null(row$refs)) list() else row$refs(fam))
    y_axis <- if (is.numeric(row$axis)) {
      .linear_axis(row$axis)
    } else {
      values <- c(unlist(lapply(curves, `[[`, "y")), unlist(lapply(refs, function(r) lapply(r, `[[`, "y"))))
      .log_axis(.auto_limits(values, log = TRUE), row$axis)
    }
    x_axis <- x$axis
    x_axis$top <- row$top
    last <- i == length(rows)
    for (j in seq_along(FAMILIES)) {
      p <- ggplot() + .ref_layers(refs[[j]]) + .curve_layers(curves[[j]], markers) +
        .axis_scale(x_axis, "x") + .axis_scale(y_axis, "y") +
        labs(
          title = if (i == 1L) FAMILIES[j], x = if (last) latex_title(x$title),
          y = if (j == 1L) latex_title(row$label)
        ) +
        .series_theme(show_x_text = last, show_y_text = j == 1L)
      if (!is.null(row$legend)) {
        entries <- c(
          if (!isFALSE(row$legend$solvers)) .solver_entries(unique(fams[[j]]$solver), styles),
          .ref_entries(refs[[j]])
        )
        p <- p + .panel_legend(entries, row$legend$loc, curves[[j]], refs[[j]], x_axis, y_axis, markers)
      }
      panels[[length(panels) + 1L]] <- p
    }
  }
  panels
}

#' Log x-axis over the values of T, labelled as integers.
.T_axis <- function(df) {
  values <- sort(unique(df[["T"]]))
  list(
    value = function(d) d[["T"]], order = "T", title = "$T$ (number of included dummies)",
    axis = .axis(.auto_limits(values, log = TRUE), values, as.character(as.integer(values)), log = TRUE)
  )
}

#' FDR (top row) and TPR (bottom row) in % vs. T, one column per solver family.
plot_fdr_tpr_vs_T <- function(df, title) {
  rows <- lapply(c("FDR", "TPR"), function(metric) {
    list(value = function(d) 100 * d[[metric]], axis = c(-3, 103), label = paste0(metric, " (%)"))
  })
  rows[[2L]]$legend <- list(loc = "best")
  panels <- .family_panels(df, .T_axis(df), rows)
  .assemble(panels, 3L, paste0(title, ": FDR and TPR vs. $T$"), c(15, 9), note = TRUE)
}

#' TPR vs. FDR in % along the T grid, one panel per solver family.
plot_tpr_vs_fdr <- function(df, title) {
  x <- list(value = function(d) 100 * d$FDR, order = "T", title = "FDR (%)", axis = .linear_axis(c(-3, 103)))
  rows <- list(list(value = function(d) 100 * d$TPR, axis = c(-3, 103), label = "TPR (%)", legend = list(loc = "best")))
  panels <- .family_panels(df, x, rows)
  .assemble(panels, 3L, paste0(title, ": TPR vs. FDR along the $T$ grid"), c(15, 5.2), note = TRUE)
}

#' Selected variables, steps and cumulative time (ms) vs. T, one column per solver family.
#' With rank, p, L and s, the first row shows s + (p - s) / (L + 1) * T (true variables plus the
#' expected nulls before the T-th dummy under exchangeability) and the second row the step limit
#' 8 * min(p + L, rank) of the capped solvers, rank being the active-set cap (n - 1 when centred).
plot_path_stats <- function(df, title, rank = NULL, p = NULL, L = NULL, s = NULL) {
  rows <- list(
    list(value = function(d) d$avg_selected, axis = "sci", label = "Selected variables",
         legend = list(loc = "upper left")),
    list(value = function(d) d$avg_steps, axis = "sci", label = "Steps"),
    list(value = function(d) d$avg_ms, axis = "sci", label = "Time from construction (ms)")
  )
  if (!is.null(rank) && !is.null(p) && !is.null(L) && !is.null(s)) {
    grid_T <- sort(unique(df[["T"]]))
    rate <- (p - s) / (L + 1)
    limit <- 8L * min(p + L, rank)
    expected <- list(x = grid_T, y = s + rate * grid_T, linetype = DASHED,
                     label = sprintf("expected: %d true + %.2f $T$ null", s, rate))
    rows[[1L]]$refs <- function(fam) list(expected)
    rows[[2L]]$refs <- function(fam) list(list(y = limit, linetype = DOTTED, label = sprintf("step limit %d", limit)))
    rows[[2L]]$legend <- list(loc = "upper left", solvers = FALSE)
  }
  panels <- .family_panels(df, .T_axis(df), rows)
  .assemble(panels, 3L, paste0(title, ": selected variables, steps and time vs. $T$"), c(15, 12), note = TRUE)
}

#' Prediction RMSE and coefficient error of the post-selection refit vs. T, one column per family,
#' with the refit on the true support as the reference.
plot_refit_errors <- function(df, title) {
  oracle <- function(column, label = NULL) {
    function(fam) list(list(y = df[[column]][1L], linetype = DASHED, label = label))
  }
  rows <- list(
    list(value = function(d) d$rmse, axis = "sci", label = "Prediction RMSE (test set)",
         refs = oracle("oracle_rmse", "least squares on the true support"), legend = list(loc = "best")),
    list(value = function(d) d$coef_err, axis = "sci", label = "Coefficient error $\\|\\hat{\\beta} - \\beta\\|_2$",
         refs = oracle("oracle_coef_err"))
  )
  panels <- .family_panels(df, .T_axis(df), rows)
  title <- paste0(title, ": least-squares refit on the selected variables vs. $T$")
  .assemble(panels, 3L, title, c(15, 8.5), note = TRUE)
}

#' Log x-axis over the dummy factors L/p, labelled p, 2p, 5p, ...
.L_axis <- function(df) {
  factors <- sort(unique(as.integer(df$L_factor)))
  labels <- lapply(factors, function(f) latex_title(if (f == 1L) "$p$" else sprintf("$%dp$", f)))
  list(
    value = function(d) d$L_factor, order = "L_factor", title = "$L$ (number of dummies)",
    axis = .axis(.auto_limits(factors, log = TRUE), factors, as.expression(labels), log = TRUE)
  )
}

#' Time to T, steps to T and time per step vs. L/p, one column per solver family, with a
#' reference proportional to the number of columns p + L, anchored at L = p.
plot_runtime <- function(df, title, p) {
  x <- .L_axis(df)
  first <- min(df$L_factor)
  scaling <- function(fam) {
    start <- stats::median(fam$us_per_step[fam$L_factor == first])
    factors <- x$axis$breaks
    list(list(x = factors, y = start * (p + factors * p) / (2 * p), linetype = DASHED, label = "$\\propto p + L$"))
  }
  rows <- list(
    list(value = function(d) d$ms, axis = "plain", label = "Time to $T$ (ms)", legend = list(loc = "upper left")),
    list(value = function(d) d$steps, axis = "plain", label = "Steps to $T$"),
    list(value = function(d) d$us_per_step, axis = "plain", label = "Time per step ($\\mu$s)", refs = scaling,
         legend = list(loc = "upper left", solvers = FALSE))
  )
  panels <- .family_panels(df, x, rows, markers = "line")
  .assemble(panels, 3L, paste0(title, ": runtime vs. the number of dummies"), c(15, 12))
}

#' Fresh run to T + 1 over the warm start T -> T + 1 vs. L/p, one panel per solver family.
plot_warm_start <- function(df, title) {
  rows <- list(list(
    value = function(d) d$fresh_over_warm, axis = "plain",
    label = "Fresh run to $T + 1$ / warm start $T \\to T + 1$", legend = list(loc = "best")
  ))
  panels <- .family_panels(df, .L_axis(df), rows, markers = "line")
  .assemble(panels, 3L, paste0(title, ": gain of the warm start"), c(15, 5))
}

#' Held solver heap (MiB, with the data size) and its share of the data (%) vs. n/p, n on the top axis.
plot_memory <- function(df, title) {
  ratios <- sort(unique(df$n_over_p))
  n_values <- sort(unique(as.integer(df$n)))
  data <- df[!duplicated(df$n), , drop = FALSE]
  data <- data[order(data$n), , drop = FALSE]
  x <- list(
    value = function(d) d$n_over_p, order = "n", title = "$n/p$",
    axis = .axis(.auto_limits(ratios, log = TRUE), ratios, sprintf("%g", ratios), log = TRUE)
  )
  reference <- list(x = data$n_over_p, y = data$data_mib, linetype = DASHED, label = "data $X$, $D$ and $y$")
  top <- list(breaks = ratios, labels = formatC(n_values, format = "d", big.mark = ","), title = latex_title("$n$"))
  rows <- list(
    list(value = function(d) d$solver_mib, axis = "plain", label = "Held solver heap (MiB)",
         refs = function(fam) list(reference), legend = list(loc = "upper left"), top = top),
    list(value = function(d) 100 * d$solver_mib / d$data_mib, axis = "plain", label = "Held solver heap / data (%)")
  )
  panels <- .family_panels(df, x, rows, markers = "line")
  .assemble(panels, 3L, paste0(title, ": memory vs. $n/p$"), c(15, 8.5))
}


# ==============================================================================
# Recovery vs. n/(s log p) and the phase transition (demo_ts_perf_04)
# ==============================================================================

RECOVERY_METRICS <- list(
  exact = c("Exact recovery (%)", "exact recovery by the first $s$ entries"),
  TPR = c("TPR at $T = 1$ (%)", "TPR at the first dummy"),
  FDR = c("FDR at $T = 1$ (%)", "FDR at the first dummy")
)

#' Colours of matplotlib's 256-entry viridis map at positions v in [0, 1].
.viridis <- function(v) viridisLite::viridis(256L)[pmin(floor(v * 256), 255) + 1L]

#' One panel per solver from panel(name, row, col, bottom), a row per family, spacers in the unused slots.
#' The legend grob goes to the lower right, into the empty slot there if the last family has fewer solvers.
.solver_grid <- function(df, panel, legend = NULL) {
  per_family <- lapply(FAMILIES, function(f) unique(df$solver[df$family == f]))
  cols <- max(lengths(per_family))
  panels <- list()
  for (row in seq_along(FAMILIES)) {
    names <- per_family[[row]]
    for (col in seq_len(cols)) {
      if (col > length(names)) {
        panels[[length(panels) + 1L]] <- plot_spacer()
        next
      }
      bottom <- row == length(FAMILIES) || col > length(per_family[[row + 1L]])
      panels[[length(panels) + 1L]] <- panel(names[col], row, col, bottom)
    }
  }
  if (!is.null(legend)) {
    last <- length(panels)
    panels[[last]] <- if (length(per_family[[length(FAMILIES)]]) < cols) {
      wrap_elements(panel = legend)
    } else {
      panels[[last]] + panel_grob(legend)
    }
  }
  list(panels = panels, cols = cols)
}

#' Title, axis labels and theme of a solver panel: the family and y title in the first column, the x title
#' and tick labels at the bottom of every column.
.grid_labels <- function(name, row, col, bottom, x_title, y_title, x_text_size = 10) {
  list(
    labs(
      title = display_name(name), x = if (bottom) x_title,
      y = if (col == 1L) call("atop", FAMILIES[row], y_title)
    ),
    .series_theme(show_x_text = bottom, show_y_text = col == 1L),
    theme(
      plot.title = element_text(size = 10, hjust = 0.5), axis.title = element_text(size = 9),
      axis.text.x.bottom = if (bottom) element_text(size = x_text_size, colour = "black") else element_blank()
    )
  )
}

#' One metric in % vs. n/(s log p): a panel per solver (rows = families), a curve per p.
plot_recovery <- function(df, metric, title) {
  label <- latex_title(RECOVERY_METRICS[[metric]][1L])
  ratios <- sort(unique(df$ratio))
  p_values <- sort(unique(as.integer(df$p)))
  colours <- .viridis(0.9 * (seq_along(p_values) - 1) / max(1L, length(p_values) - 1L))
  x_axis <- .axis(.auto_limits(ratios, log = TRUE), ratios, sprintf("%g", ratios), log = TRUE)
  y_axis <- .linear_axis(c(-3, 103))
  x_title <- latex_title("$n/(s \\log p)$")
  panel <- function(name, row, col, bottom) {
    sub <- df[df$solver == name, , drop = FALSE]
    curves <- do.call(rbind, lapply(seq_along(p_values), function(k) {
      curve <- sub[sub$p == p_values[k], , drop = FALSE]
      curve <- curve[order(curve$ratio), , drop = FALSE]
      data.frame(solver = p_values[k], x = curve$ratio, y = 100 * curve[[metric]], filled = curve$reached >= 0.99,
                 colour = colours[k], marker = "o")
    }))
    ggplot() +
      .curve_layers(curves, "reached", linewidth = 1.5, size = sqrt(22), stroke = 1.1) +
      .axis_scale(x_axis, "x") + .axis_scale(y_axis, "y") +
      .grid_labels(name, row, col, bottom, x_title, label, x_text_size = 8)
  }

  entries <- Map(function(p, colour) {
    .entry(latex_title(sprintf("$p = %s$", formatC(p, format = "d", big.mark = "{,}"))), colour, linewidth = 1.5,
           marker = "o", size = 5)
  }, p_values, colours)
  legend <- .legend_grob(unname(entries), "lower right", fontsize = 10, title = latex_title("$L = p$, $s = 10$"))
  grid <- .solver_grid(df, panel, legend)
  .assemble(
    grid$panels, grid$cols, sprintf("%s: %s vs. $n/(s \\log p)$", title, RECOVERY_METRICS[[metric]][2L]),
    c(3.1 * grid$cols, 3.0 * length(FAMILIES)), note = any(df$reached < 0.99)
  )
}

#' Upper tail of the standard normal, 0.5 * erfc(t / sqrt(2)).
.gauss_tail <- function(t) stats::pnorm(t, lower.tail = FALSE)

#' Statistical dimension over N of the l1 descent cone at an eps N-sparse vector (Amelunxen et al., 2014).
.l1_statistical_dimension <- function(eps) {
  bound <- function(t) {
    density <- exp(-t * t / 2) / sqrt(2 * pi)
    eps * (1 + t * t) + 2 * (1 - eps) * ((1 + t * t) * .gauss_tail(t) - t * density)
  }
  lo <- 0
  hi <- 12
  golden <- (sqrt(5) - 1) / 2
  for (k in seq_len(100L)) {  # the bound is convex in t
    a <- hi - golden * (hi - lo)
    b <- lo + golden * (hi - lo)
    if (bound(a) < bound(b)) hi <- b else lo <- a
  }
  bound((lo + hi) / 2)
}

#' Noiseless l1 phase transition in the coordinates n/p and s/n for a design with columns_over_p * p columns.
l1_phase_transition <- function(columns_over_p) {
  eps <- 0.5 * seq_len(399L) / 400
  delta <- vapply(eps, .l1_statistical_dimension, numeric(1L))  # n / N at the transition
  keep <- columns_over_p * delta <= 1
  data.frame(x = columns_over_p * delta[keep], y = eps[keep] / delta[keep])
}

#' Colour bar of the viridis scale from 0 to 100, as a narrow panel with its axis on the right.
.colour_bar <- function(title) {
  bar <- data.frame(x = 0, y = (seq_len(256L) - 0.5) * 100 / 256)
  ggplot(bar, aes(x, y, fill = y)) +
    geom_raster() +
    scale_fill_viridis_c(limits = c(0, 100), guide = "none") +
    scale_x_continuous(expand = c(0, 0), breaks = NULL) +
    scale_y_continuous(limits = c(0, 100), breaks = .linear_ticks(c(0, 100)), expand = c(0, 0), position = "right") +
    labs(x = NULL, y = title) +
    .series_theme(grid = FALSE) +
    theme(axis.title.y.right = element_text(angle = 90))
}

#' Exact recovery (%) over n/p and s/n, one heatmap per solver, with the 50% contour, the l1 phase
#' transition and the LARS s-step bound, and a shared colour bar.
plot_phase_transition <- function(df, title) {
  solvers <- unique(df$solver)
  x_values <- sort(unique(df$n_over_p))
  y_values <- sort(unique(df$s_over_n))
  columns <- 2 * df$p[1L]  # L = p
  step_x <- x_values[2L] - x_values[1L]
  step_y <- y_values[2L] - y_values[1L]
  x_axis <- .linear_axis(c(x_values[1L] - step_x / 2, x_values[length(x_values)] + step_x / 2))
  y_axis <- .linear_axis(c(y_values[1L] - step_y / 2, y_values[length(y_values)] + step_y / 2))
  l1 <- l1_phase_transition(columns / df$p[1L])
  refs <- list(
    list(x = l1$x, y = l1$y, colour = "#d62728", linetype = DASHED, linewidth = 1.6,
         label = "$\\ell_1$ phase transition, $N = p + L$ (noiseless)"),
    list(y = 1 / (2 * log(columns)), colour = "#ff7f0e", linetype = DOTTED, linewidth = 1.8,
         label = "$s/n = 1/(2 \\ln N)$ (LARS $s$-step bound)")
  )
  panels <- lapply(seq_along(solvers), function(k) {
    grid <- df[df$solver == solvers[k], , drop = FALSE]
    grid$z <- 100 * grid$exact
    p <- ggplot(grid, aes(n_over_p, s_over_n)) +
      geom_raster(aes(fill = z)) +
      geom_contour(aes(z = z), breaks = 50, colour = "white", linewidth = .lw(1.4)) +
      .ref_layers(refs) +
      scale_fill_viridis_c(limits = c(0, 100), guide = "none") +
      .axis_scale(x_axis, "x") + .axis_scale(y_axis, "y") +
      labs(title = display_name(solvers[k]), x = latex_title("$n/p$"), y = if (k == 1L) latex_title("$s/n$")) +
      .series_theme(show_y_text = k == 1L, grid = FALSE)
    if (k > 1L) return(p)
    entries <- c(list(.entry("50% contour", "white", linewidth = 1.4)), .ref_entries(refs))
    legend <- .legend_grob(entries, "upper left", fill = "#4d4d4d", text_colour = "white", alpha = 0.85)
    p + panel_grob(legend)
  })
  .assemble(
    c(panels, list(.colour_bar("Exact recovery (%)"))), length(solvers) + 1L,
    paste0(title, ": exact recovery by the first $s$ entries"), c(5.2 * length(solvers), 5),
    widths = c(rep(1, length(solvers)), 0.045)
  )
}


# ==============================================================================
# Null counts vs. the negative hypergeometric distribution (demo_ts_perf_05)
# ==============================================================================

SELECTED_LEVELS <- list(
  list(statistic = "selected_v1", label = "$v \\to 1$", colour = "#1b9e77"),
  list(statistic = "selected_v0.75", label = "$v = 0.75$", colour = "#ff7f0e"),
  list(statistic = "selected_v0.5", label = "$v = 0.5$", colour = "#7570b3")
)

#' PMF of the number of nulls drawn before the T-th dummy, NHG(p0 + L, p0, T), for 0, ..., p0.
nhg_pmf <- function(p0, L, T_stop) {
  v <- 0:p0
  exp(lchoose(v + T_stop - 1, v) + lchoose(p0 + L - v - T_stop, p0 - v) - lchoose(p0 + L, p0))
}

#' p0, L, T and K of the demo, constant over the CSV.
.counts_setup <- function(df) lapply(df[1L, c("p0", "L", "T", "K")], as.integer)

#' Frequencies of the counts 0, ..., size - 1 of one solver and statistic.
.frequencies <- function(df, solver, statistic, size) {
  sub <- df[df$solver == solver & df$statistic == statistic, , drop = FALSE]
  freq <- numeric(size)
  freq[sub$count + 1L] <- sub$frequency
  freq
}

#' CDF of weights over 0, ..., size - 1.
.cumulative <- function(weights) cumsum(weights) / sum(weights)

#' Smallest count whose CDF reaches prob.
.count_quantile <- function(cdf, prob) {
  hit <- which(cdf >= prob - 1e-12)
  if (length(hit) == 0L) length(cdf) - 1L else hit[1L] - 1L
}

#' Histogram of the included null variables at T per solver, with the NHG PMF and both means.
plot_null_pmf <- function(df, title) {
  setup <- .counts_setup(df)
  pmf <- nhg_pmf(setup$p0, setup$L, setup$T)
  nhg_mean <- setup$T * setup$p0 / (setup$L + 1)
  xs <- 0:(.count_quantile(.cumulative(pmf), 0.9999) + 2L)
  densities <- lapply(unique(df$solver), function(name) {
    freq <- .frequencies(df, name, "included", setup$p0 + 1L)
    freq / sum(freq)
  })
  names(densities) <- unique(df$solver)
  y_top <- 1.05 * max(pmf[xs + 1L], unlist(lapply(densities, `[`, xs + 1L)))
  y_ticks <- .linear_ticks(c(0, y_top), 5L)
  x_axis <- .linear_axis(c(-0.5, max(xs) + 0.5))
  y_axis <- .axis(c(0, y_top), y_ticks, format(y_ticks, trim = TRUE))
  panel <- function(name, row, col, bottom) {
    density <- densities[[name]]
    ggplot() +
      geom_col(data = data.frame(x = xs, y = density[xs + 1L]), aes(x, y), width = 1, fill = "#fde0b8",
               colour = "grey25", linewidth = .lw(0.4)) +
      geom_vline(xintercept = nhg_mean, colour = "black", linewidth = .lw(1.4)) +
      geom_vline(xintercept = sum((seq_along(density) - 1) * density), colour = COLOR_DUMMY, linetype = DASHED,
                 linewidth = .lw(1.4)) +
      geom_marker(data.frame(x = xs, y = pmf[xs + 1L]), colour = "black", fill = "black", size = sqrt(6)) +
      .axis_scale(x_axis, "x") + .axis_scale(y_axis, "y") +
      .grid_labels(name, row, col, bottom, "Included null variables", "Density")
  }
  entries <- list(
    .entry("NHG PMF", "black", linetype = "blank", marker = "o", size = 3),
    .entry(latex_title(sprintf("NHG mean $T p_0/(L+1) = %.2f$", nhg_mean)), "black", linewidth = 1.4),
    .entry("Mean of the histogram", COLOR_DUMMY, DASHED, linewidth = 1.4)
  )
  grid <- .solver_grid(df, panel, .legend_grob(entries, "lower right", fontsize = 10))
  title <- sprintf("%s: included null variables at $T = %d$ vs. NHG($p_0 + L$, $p_0$, $T$)", title, setup$T)
  .assemble(grid$panels, grid$cols, title, c(3.1 * grid$cols, 3.0 * length(FAMILIES)))
}

#' Q-Q plot of the included null variables at T against the NHG, one panel per solver.
plot_null_qq <- function(df, title) {
  setup <- .counts_setup(df)
  nhg_cdf <- .cumulative(nhg_pmf(setup$p0, setup$L, setup$T))
  probs <- (seq_len(200L) - 0.5) / 200
  theory <- vapply(probs, .count_quantile, integer(1L), cdf = nhg_cdf)
  samples <- lapply(unique(df$solver), function(name) {
    cdf <- .cumulative(.frequencies(df, name, "included", setup$p0 + 1L))
    vapply(probs, .count_quantile, integer(1L), cdf = cdf)
  })
  names(samples) <- unique(df$solver)
  axis <- .linear_axis(c(0, max(theory, unlist(samples)) + 2))
  panel <- function(name, row, col, bottom) {
    ggplot() +
      geom_abline(slope = 1, intercept = 0, colour = "#666666", linewidth = .lw(1.0)) +
      geom_marker(data.frame(x = theory, y = samples[[name]]), colour = "black", fill = "white", size = sqrt(12),
                  stroke = 0.7) +
      .axis_scale(axis, "x") + .axis_scale(axis, "y") +
      .grid_labels(name, row, col, bottom, "NHG quantiles", "Sample quantiles")
  }
  grid <- .solver_grid(df, panel)
  title <- sprintf("%s: Q-Q plot of the included null variables at $T = %d$", title, setup$T)
  .assemble(grid$panels, grid$cols, title, c(3.1 * grid$cols, 3.0 * length(FAMILIES)))
}

#' ECDFs of the selected null variables V(v) at the voting levels, with the NHG CDF, one panel per solver.
plot_selected_ecdf <- function(df, title) {
  setup <- .counts_setup(df)
  nhg_cdf <- .cumulative(nhg_pmf(setup$p0, setup$L, setup$T))
  xs <- 0:(.count_quantile(nhg_cdf, 0.9999) + 2L)
  x_axis <- .linear_axis(c(0, max(xs)))
  y_axis <- .linear_axis(c(-0.02, 1.02))
  step <- function(cdf, colour) {
    geom_step(data = data.frame(x = xs, y = cdf[xs + 1L]), aes(x, y), direction = "hv", colour = colour,
              linewidth = .lw(1.3))
  }
  panel <- function(name, row, col, bottom) {
    levels <- lapply(SELECTED_LEVELS, function(level) {
      step(.cumulative(.frequencies(df, name, level$statistic, setup$p0 + 1L)), level$colour)
    })
    ggplot() + levels + step(nhg_cdf, "black") +
      .axis_scale(x_axis, "x") + .axis_scale(y_axis, "y") +
      .grid_labels(name, row, col, bottom, "Selected null variables", "Empirical CDF")
  }
  entries <- c(
    lapply(SELECTED_LEVELS, function(level) .entry(latex_title(level$label), level$colour, linewidth = 1.3)),
    list(.entry("NHG", "black", linewidth = 1.3))
  )
  legend <- .legend_grob(entries, "lower right", fontsize = 10,
                         title = latex_title(sprintf("$K = %d$, $T = %d$", setup$K, setup$T)))
  grid <- .solver_grid(df, panel, legend)
  title <- paste0(title, ": selected null variables $V_{T,L}(v)$ vs. NHG")
  .assemble(grid$panels, grid$cols, title, c(3.1 * grid$cols, 3.0 * length(FAMILIES)))
}


# ==============================================================================
# CLI
# ==============================================================================

#' Values of a multi-value flag: tokens up to the next --flag.
.consume_multi <- function(argv, i) {
  values <- character()
  while (i + 1L <= length(argv) && !startsWith(argv[[i + 1L]], "--")) {
    i <- i + 1L
    values <- c(values, argv[[i]])
  }
  list(values = values, i = i)
}

# Flags with one value and flags with several values, by their field in the arguments
SINGLE_FLAGS <- c(
  "--solver" = "solver", "--title" = "title", "--stem" = "stem", "--outdir" = "outdir", "--rank" = "rank",
  "--p" = "p", "--L" = "L", "--s" = "s", "--phase" = "phase"
)
MULTI_FLAGS <- c("--formats" = "formats", "--csvs" = "csvs", "--labels" = "labels")

#' Parse the arguments after the mode: positional CSVs and the flags.
parse_args <- function(argv) {
  args <- list(csv = character(), csvs = character(), labels = character(), solver = NULL,
               title = NULL, stem = NULL, outdir = NULL, formats = c("png", "pdf"), rank = NULL,
               p = NULL, L = NULL, s = NULL, phase = NULL)
  i <- 1L
  while (i <= length(argv)) {
    a <- argv[[i]]
    value <- function() {
      if (i + 1L > length(argv)) stop("missing value for ", a)
      i <<- i + 1L
      argv[[i]]
    }
    multi <- function() {
      r <- .consume_multi(argv, i)
      i <<- r$i
      r$values
    }
    if (a %in% names(SINGLE_FLAGS)) {
      args[[SINGLE_FLAGS[[a]]]] <- value()
    } else if (a %in% names(MULTI_FLAGS)) {
      args[[MULTI_FLAGS[[a]]]] <- multi()
    } else if (startsWith(a, "--")) {
      stop("unknown flag ", a)
    } else {
      args$csv <- c(args$csv, a)
    }
    i <- i + 1L
  }
  args$formats <- unlist(strsplit(args$formats, ","))
  args
}

#' Integer value of a numeric flag, NULL when the flag is not given.
.int_flag <- function(value) if (is.null(value)) NULL else as.integer(value)

#' File name of a CSV without its folder and the .csv suffix.
.csv_stem <- function(csv) sub("\\.csv$", "", basename(csv))

#' Save figures to outdir/<stem>_<name>, each at its figure_size.
.save_figures <- function(figs, outdir, stem, formats) {
  for (name in names(figs)) {
    size <- attr(figs[[name]], "figure_size")
    save_figure(figs[[name]], outdir, paste0(stem, "_", name), formats, width = size[1L], height = size[2L])
  }
}

main <- function(argv = commandArgs(trailingOnly = TRUE)) {
  usage <- paste("Usage: ts_plt_utils.R paths <stem>_path.csv --solver S [--title T]",
                 "| compare --csvs A B --labels LA LB --title T --stem S",
                 "| fdr-tpr <csv> --title T [--rank R --p P --L L --s S]",
                 "| runtime <csv> --title T --p P | memory <csv> --title T",
                 "| recovery [<csv>] [--phase <csv>] --title T | nhg <csv> --title T",
                 "[--outdir DIR] [--formats png pdf]")
  if (length(argv) == 0L) stop(usage)
  mode <- argv[[1L]]
  args <- parse_args(argv[-1L])
  perf_modes <- c("fdr-tpr", "runtime", "memory", "nhg")
  if (mode %in% perf_modes) {
    if (length(args$csv) != 1L || is.null(args$title)) stop(usage)
    df <- utils::read.csv(args$csv, check.names = FALSE)
    outdir <- if (is.null(args$outdir)) default_plots_dir(args$csv) else args$outdir
  }
  if (mode == "paths") {
    if (length(args$csv) != 1L || is.null(args$solver)) stop(usage)
    data <- read_paths(args$csv)
    fig <- plot_terminated_paths(data$path, data$actions, args$solver, args$title)
    stem <- if (is.null(args$stem)) sub("_path\\.csv$", "", basename(args$csv)) else args$stem
    outdir <- if (is.null(args$outdir)) default_plots_dir(args$csv) else args$outdir
    save_figure(fig, outdir, stem, args$formats, height = 4.2 * length(unique(data$actions$T_stop)))
  } else if (mode == "compare") {
    if (length(args$csvs) == 0L || is.null(args$title) || is.null(args$stem)) stop(usage)
    if (length(args$csvs) != length(args$labels))
      stop("--csvs and --labels need the same number of entries")
    variants <- Map(function(label, csv) c(list(label = label), read_paths(csv)), args$labels, args$csvs)
    fig <- plot_path_comparison(unname(variants), args$title)
    outdir <- if (is.null(args$outdir)) default_plots_dir(args$csvs[[1L]]) else args$outdir
    save_figure(fig, outdir, args$stem, args$formats, height = 4.2 * length(variants))
  } else if (mode == "fdr-tpr") {
    figs <- list(
      fdr_tpr_vs_T = plot_fdr_tpr_vs_T(df, args$title),
      tpr_vs_fdr = plot_tpr_vs_fdr(df, args$title),
      path_stats = plot_path_stats(
        df, args$title, .int_flag(args$rank), .int_flag(args$p), .int_flag(args$L), .int_flag(args$s)
      ),
      refit_errors = plot_refit_errors(df, args$title)
    )
    .save_figures(figs, outdir, .csv_stem(args$csv), args$formats)
  } else if (mode == "runtime") {
    if (is.null(args$p)) stop(usage)
    figs <- list(
      runtime = plot_runtime(df, args$title, .int_flag(args$p)),
      warm_start = plot_warm_start(df, args$title)
    )
    .save_figures(figs, outdir, .csv_stem(args$csv), args$formats)
  } else if (mode == "memory") {
    .save_figures(list(memory = plot_memory(df, args$title)), outdir, .csv_stem(args$csv), args$formats)
  } else if (mode == "recovery") {
    if (length(args$csv) > 1L || is.null(args$title)) stop(usage)
    if (length(args$csv) == 0L && is.null(args$phase)) stop("recovery needs the Part 1 CSV, --phase, or both")
    if (length(args$csv) == 1L) {
      df <- utils::read.csv(args$csv, check.names = FALSE)
      figs <- lapply(names(RECOVERY_METRICS), function(metric) plot_recovery(df, metric, args$title))
      names(figs) <- tolower(names(RECOVERY_METRICS))
      outdir <- if (is.null(args$outdir)) default_plots_dir(args$csv) else args$outdir
      .save_figures(figs, outdir, .csv_stem(args$csv), args$formats)
    }
    if (!is.null(args$phase)) {
      fig <- plot_phase_transition(utils::read.csv(args$phase, check.names = FALSE), args$title)
      size <- attr(fig, "figure_size")
      outdir <- if (is.null(args$outdir)) default_plots_dir(args$phase) else args$outdir
      save_figure(fig, outdir, .csv_stem(args$phase), args$formats, width = size[1L], height = size[2L])
    }
  } else if (mode == "nhg") {
    figs <- list(
      pmf = plot_null_pmf(df, args$title), qq = plot_null_qq(df, args$title), ecdf = plot_selected_ecdf(df, args$title)
    )
    .save_figures(figs, outdir, .csv_stem(args$csv), args$formats)
  } else {
    stop(usage)
  }
  invisible(0L)
}

if (sys.nframe() == 0L) main()
