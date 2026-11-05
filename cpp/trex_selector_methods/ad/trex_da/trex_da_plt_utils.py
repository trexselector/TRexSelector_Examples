#!/usr/bin/env python3
"""
trex_da_plt_utils.py — shared plotting utilities for the DA-TRex demo suite.

Suite-level companion of trex_da_sim_common.hpp: one module hosting the plot
patterns for every DA-TRex demo in this folder whose Monte Carlo results use the
tidy CSV schema

    solver,metric,<SWEEP>,value

where <SWEEP> is whatever the demo sweeps -- one of SNR, Rho, M (blocks), Q
(block size), tFDR, or a Linkage index -- and the 'solver' column holds the
methods compared (``TREX-DA: TLARS`` / ``TAFS`` / ``TOMP`` and, in the AR(1)
demo, the classical ``TREX: TLARS`` baseline). Each demo records the metrics
``FDR, sd_FDR, TPR, sd_TPR`` (plus bookkeeping ``Avg_L`` / ``Avg_T`` which are
ignored here); the shared reader auto-detects the sweep column, so no per-demo
configuration is required beyond the CSV path and titles supplied by each
demo's generate_plots.sh.

Unlike the classical suite (../trex/trex_plt_utils.py), which always sweeps SNR,
the DA demos sweep several different quantities, so the x-axis (label, log vs.
linear scale, tick handling, categorical linkage labels) is chosen from the
detected sweep column via SWEEP_CONFIG.

Plot patterns hosted here:
    plot_fdr_tpr_vs_sweep()        — side-by-side FDR | TPR panels vs the swept
                                     quantity, one line per method, optional
                                     +/- sd bands (--bands).
    write_fdr_tpr_vs_sweep_html()  — interactive Plotly HTML (hover, toggles).
    plot_comparison_grid()         — 2xN grid: FDR/TPR rows x one column per
                                     input CSV, all methods per panel on a
                                     shared scale (columns = HAC linkage in
                                     the BT demos 02-05, or noise
                                     scenario / support placement -- whatever
                                     the caller passes).
    plot_gap_rho()                 — demo-01 2D gap x rho study: mean_TPP /
                                     mean_FDP heatmaps for CappedSpread support
                                     with the DA+AR1 correction-window
                                     (kappa-boundary) overlaid, plus the Random
                                     support baseline (its own wider CSV schema,
                                     read via read_gap_rho()).
    plot_sweep2d()                 — generic 2D sweep (demo-07 kappa x rho and
                                     SNR x rho studies): per-solver TPR / FDR
                                     heatmaps over the 2D grid, one figure per
                                     support placement, tFDR-violating FDR
                                     cells outlined (tidy two-support schema
                                     solver,support_type,metric,<ROW>,<COL>,
                                     value, read via read_sweep2d()).

Four CLI modes (all typically driven from a demo's generate_plots.sh):
    # default -- per-CSV overview (--plotly adds the interactive html), sweep auto-detected
    python trex_da_plt_utils.py <csv> --title "..." --legend-title "Method" --tfdr 0.2
    # cross-CSV comparison grid (columns = linkage / scenario / support)
    python trex_da_plt_utils.py grid --labels Single Complete Average \
        --csvs a_single.csv a_complete.csv a_average.csv \
        --title "..." --stem my_grid --outdir <plots>
    # demo-01 gap x rho 2D study from the wide CSV
    python trex_da_plt_utils.py gaprho <gap_rho.csv> --stem my_gaprho --outdir <plots>
    # generic 2D sweep heatmaps (one figure per support placement)
    python trex_da_plt_utils.py sweep2d <kappa_rho.csv> --stem my_2d --outdir <plots>

Output location: figures default to the simulation_results/plots/ sibling when
the CSV lives in simulation_results/data/ (the suite convention); otherwise they
are written next to the CSV. Override with --outdir. (The grid and gaprho modes
require an explicit --outdir, since they are not tied to a single CSV's
location.)
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")  # headless / batch-safe: no display required

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, ScalarFormatter

# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
STEM_SUFFIX = "_fdr_tpr"
DEFAULT_TFDR = 0.20          # target FDR control level (tFDR); demo-specific
DEFAULT_LEGEND_TITLE = "Method"
Y_HEADROOM = 1.25            # budget above the largest FDR value / the tFDR line
MAX_XTICKS = 10              # keep all ticks on the rho grid (10 values); thin only denser sweeps

# TAFS is the only solver run with a non-zero AFS correlation parameter, and it
# is a defining part of the method rather than a tuning detail -- so it is shown
# in every legend. Mirrors rho_afs in default_solvers() (trex_da_sim_common.hpp);
# keep the two in sync. Applied at plot time rather than baked into the CSVs, so
# the annotation appears without re-running any simulation.
TAFS_RHO = 0.3

METRICS = ("FDR", "TPR")
METRIC_LABELS = {
    "FDR": "False discovery rate",
    "TPR": "True positive rate",
}
# Each plotted metric's paired standard-deviation metric (as it appears in the
# CSV's 'metric' column, upper-cased). Used to shade +/- sd bands with --bands.
SD_OF = {"FDR": "SD_FDR", "TPR": "SD_TPR"}

# Black is reserved for the tFDR reference line, so no data row uses it.
TFDR_COLOR = "black"

# Markers cycle independently of colour so many rows stay distinguishable.
MARKERS = [
    "o", "s", "^", "D", "v", "P", "X",
    "<", ">", "h", "H", "*", "8", "p",
]

# ---------------------------------------------------------------------
# Sweep-axis configuration
# ---------------------------------------------------------------------
# The DA demos sweep different quantities; the x-axis label, scale, and tick
# behaviour are chosen from the detected sweep column (CSV's 3rd column). Keys
# are the normalised sweep name (lower-cased column name, or 'linkage' for the
# "Linkage_(1=Single,2=Complete,3=Average)" column).
SWEEP_CONFIG: dict[str, dict[str, Any]] = {
    "snr": dict(label="Signal-to-noise ratio (SNR)", scale="log"),
    "rho": dict(label=r"AR(1) correlation $\rho$", scale="linear"),
    "m": dict(label="Number of blocks $M$", scale="linear"),
    "q": dict(label="Block size $Q$", scale="linear"),
    "tfdr": dict(label="Target FDR level (tFDR)", scale="linear",
                 is_tfdr_sweep=True),
    "kappa": dict(label=r"NN bandwidth $\kappa$", scale="linear"),
    "linkage": dict(label="HAC linkage method", scale="linear",
                    categorical={1: "Single", 2: "Complete", 3: "Average"}),
}
DEFAULT_SWEEP_CONFIG: dict[str, Any] = dict(label="Sweep value", scale="linear")


def sweep_key_of(column: str) -> str:
    """Normalise a CSV sweep-column name to a SWEEP_CONFIG key.

    ``SNR`` -> ``snr``, ``Rho`` -> ``rho``, and the verbose
    ``Linkage_(1=Single,2=Complete,3=Average)`` header -> ``linkage``. Unknown
    columns fall back to their lower-cased name (handled by DEFAULT_SWEEP_CONFIG).
    """
    name = str(column).strip().lower()
    if name.startswith("linkage"):
        return "linkage"
    return name


def sweep_config(key: str) -> dict[str, Any]:
    """Look up the axis configuration for a sweep key (with a safe default)."""
    return SWEEP_CONFIG.get(key, DEFAULT_SWEEP_CONFIG)


# ---------------------------------------------------------------------
# Reading the tidy Monte Carlo CSV
# ---------------------------------------------------------------------
def read_mc_results(path: Path) -> tuple[pd.DataFrame, str]:
    """Load a tidy DA-TRex results CSV (solver,metric,<sweep>,value).

    The third column is the swept quantity and its name varies between demos, so
    it is detected positionally and renamed to ``sweep``; its original name is
    normalised to a SWEEP_CONFIG key and returned alongside the frame. Only the
    plotted metrics (FDR/TPR and their sd_ partners) are kept; other recorded
    metrics (Avg_L, Avg_T, ...) are noted and dropped.

    Returns ``(df, sweep_key)`` with df columns ``solver, metric, sweep, value``.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n  {path}\n"
            "Pass the results CSV path as the first argument."
        )

    df = pd.read_csv(path)
    if df.shape[1] < 4:
        raise ValueError(
            "CSV must have at least 4 columns (solver,metric,<sweep>,value); "
            f"found columns: {df.columns.tolist()}"
        )

    # Columns are positional: solver, metric, <sweep>, value. The linkage-sweep
    # files carry an unescaped comma in their header ("Linkage_(1=Single,
    # 2=Complete,3=Average)"), which pandas over-splits into extra header
    # columns while the 4-field data rows stay aligned to positions 0-3. So read
    # the data positionally (first 4 columns) and, for sweep detection, rebuild
    # the true sweep name by re-joining any header fragments between metric and
    # the trailing 'value' column.
    if df.shape[1] > 4:
        sweep_name = ",".join(str(c) for c in df.columns[2:-1])
    else:
        sweep_name = str(df.columns[2])
    sweep_key = sweep_key_of(sweep_name)

    df = df.iloc[:, :4].copy()
    df.columns = ["solver", "metric", "sweep", "value"]

    df["solver"] = df["solver"].astype(str).str.strip()
    # Normalise metric names: 'sd_FDR' -> 'SD_FDR', 'FDR' -> 'FDR'.
    df["metric"] = df["metric"].astype(str).str.strip().str.upper()
    df["sweep"] = pd.to_numeric(df["sweep"], errors="raise")
    df["value"] = pd.to_numeric(df["value"], errors="raise")

    plotted = set(METRICS) | set(SD_OF.values())
    present = set(df["metric"])
    missing_metrics = set(METRICS) - present
    if missing_metrics:
        raise ValueError(
            f"CSV is missing required metric(s): {sorted(missing_metrics)}\n"
            f"Found metrics: {sorted(present)}"
        )
    extra = present - plotted
    if extra:
        print(f"note: ignoring non-plotted metric(s): {sorted(extra)}")

    df = df.loc[df["metric"].isin(plotted)]
    df = annotate_solver_params(df)
    return df.sort_values(["metric", "solver", "sweep"]), sweep_key


def annotate_solver_params(df: pd.DataFrame) -> pd.DataFrame:
    """Append TAFS' AFS correlation parameter to its display label.

    ``TREX-DA+BT: TAFS`` -> ``TREX-DA+BT: TAFS (rho = 0.3)``, so the setting that
    distinguishes TAFS from the other solvers is visible in the figure itself and
    cannot be lost when a plot is read apart from its README. Matches the label
    with or without a ``TREX-DA...: `` prefix, and is idempotent (re-annotating
    an already-annotated frame is a no-op). Plain ASCII rather than mathtext:
    these labels are rendered by both matplotlib and plotly, and plotly does not
    typeset ``$\\rho$``.
    """
    suffix = f" (rho = {TAFS_RHO:g})"
    df["solver"] = df["solver"].astype(str).str.replace(
        r"(?<![\w()])TAFS(?!\s*\(rho)", f"TAFS{suffix}", regex=True
    )
    return df


def apply_relabel(df: pd.DataFrame, specs: list) -> pd.DataFrame:
    """Rename method labels in the 'solver' column for display.

    Each spec is ``OLD=NEW``; OLD is replaced as a substring, in order. Purely
    cosmetic — the CSVs keep the labels the demo binary wrote (e.g. relabel
    ``TREX-DA`` to ``TREX-DA+BT`` in a figure without touching the data).
    """
    for spec in specs:
        old, sep, new = spec.partition("=")
        if not sep:
            raise SystemExit(f"--relabel expects OLD=NEW, got: {spec!r}")
        df["solver"] = df["solver"].str.replace(old, new, regex=False)
    return df


def default_plots_dir(csv_path: Path) -> Path:
    """Suite convention: CSVs live in simulation_results/data/, figures in the
    sibling simulation_results/plots/. Falls back to the CSV's own directory."""
    if csv_path.parent.name == "data":
        return csv_path.parent.parent / "plots"
    return csv_path.parent


def method_order(df: pd.DataFrame) -> list[str]:
    """Methods (the 'solver' column) in first-seen order -- stable colours."""
    return list(dict.fromkeys(df["solver"]))


def qualitative_colors(name: str) -> list:
    """The hand-picked colour list behind a qualitative colormap.

    ``get_cmap`` is typed as returning the base ``Colormap``, but ``.colors``
    only exists on ``ListedColormap`` -- which is what the qualitative maps
    (tab10, Dark2) actually are. Narrowing here keeps the read type-safe and
    fails loudly on a colormap that carries no discrete colour list at all
    (a ``LinearSegmentedColormap`` such as jet).
    """
    cmap = plt.get_cmap(name)
    if not isinstance(cmap, ListedColormap):
        raise TypeError(
            f"{name!r} is a {type(cmap).__name__}, which has no discrete "
            "colour list; pass a ListedColormap such as tab10 or Dark2."
        )
    # `.colors` is declared as the broad ArrayLike (which admits scalars); on a
    # ListedColormap it is concretely the sequence of RGB tuples.
    return list(cast(Sequence[Any], cmap.colors))


# A "<solver>_STAG" row is <solver> with the T-loop stagnation stop on (the
# TLARS_STAG comparison row of the suite). It shares its sibling's colour and is
# drawn dashed, so each pair reads as one solver with and without the stop.
STAG_SUFFIX = "_STAG"


def stag_sibling(row: str) -> str | None:
    """The row a "_STAG" row is compared with, e.g. "TREX: TLARS" for "TREX: TLARS_STAG"."""
    return row[: -len(STAG_SUFFIX)] if row.endswith(STAG_SUFFIX) else None


def line_style(row: str) -> str:
    """Dashed for a "_STAG" row, solid otherwise (matplotlib linestyle)."""
    return "--" if stag_sibling(row) is not None else "-"


def row_colors(rows: list[str]) -> dict[str, tuple]:
    """Assign a stable, clearly distinct colour to each method.

    Concatenates two qualitative colormaps (tab10 + Dark2 = 18 hand-picked,
    high-contrast colours) rather than sampling a gradient, so adjacent curves
    are easy to tell apart. Keyed by first-seen position, so the mapping is
    deterministic and identical across the panels of a comparison grid. A
    "_STAG" row takes its sibling's colour and no palette slot of its own, so
    the other rows keep the colours they have without it.
    """
    palette = qualitative_colors("tab10") + qualitative_colors("Dark2")
    own = [row for row in rows if stag_sibling(row) not in rows]
    colors = {row: palette[i % len(palette)] for i, row in enumerate(own)}
    for row in rows:
        sibling = stag_sibling(row)
        if sibling in colors:
            colors[row] = colors[sibling]
    return colors


def rgba_to_css(color: tuple) -> str:
    """Convert a matplotlib RGB(A) tuple in [0, 1] to a CSS ``rgb(...)`` string."""
    r, g, b = (int(round(channel * 255)) for channel in color[:3])
    return f"rgb({r},{g},{b})"


def fdr_axis_top(df: pd.DataFrame, sweep_key: str, tfdr: float) -> float:
    """Upper FDR y-limit: the larger of the data max and the tFDR reference,
    plus headroom. For a tFDR *sweep* the reference is the largest target on the
    x-axis (the identity line's top), not a single fixed tFDR."""
    fdr_max = df.loc[df["metric"] == "FDR", "value"].max()
    if sweep_config(sweep_key).get("is_tfdr_sweep"):
        ref = df["sweep"].max()
    else:
        ref = tfdr if tfdr > 0 else 0.0
    return max(fdr_max, ref) * Y_HEADROOM


def legend_tfdr_lead(handles: list, labels: list, group_label: str):
    """Order legend entries as: tFDR reference first, then a group-header row
    (e.g. "Method:"), then the method/solver entries below it.

    Replaces a legend *title*: a title always sits at the very top of the
    legend box, which would file the tFDR reference under the group heading
    it does not belong to. Returns (handles, labels) for a title-less legend.
    """
    pairs = list(zip(handles, labels))
    ref = [pr for pr in pairs
           if str(pr[1]).startswith("tFDR") or str(pr[1]).startswith("realized")]
    methods = [pr for pr in pairs if pr not in ref]
    ordered = list(ref)
    if methods:
        header = plt.Line2D([], [], linestyle="None", alpha=0.0)
        ordered.append((header, f"{group_label}:"))
        ordered.extend(methods)
    return [h for h, _ in ordered], [l for _, l in ordered]


def choose_ticks(values: list[float], scale: str,
                 max_ticks: int = MAX_XTICKS) -> list[float]:
    """A readable subset of x-tick positions for a sweep grid.

    Small grids keep every value. Dense grids are thinned to ~``max_ticks``
    values -- evenly in log space for a log axis (matching the SNR axis), evenly
    in index order for a linear axis -- with the endpoints retained.
    """
    values = sorted(values)
    if len(values) <= max_ticks:
        return values
    if scale == "log":
        import math

        lo, hi = math.log10(values[0]), math.log10(values[-1])
        ticks: list[float] = []
        for i in range(max_ticks):
            target = lo + (hi - lo) * i / (max_ticks - 1)
            nearest = min(values, key=lambda v: abs(math.log10(v) - target))
            if nearest not in ticks:
                ticks.append(nearest)
        return ticks
    # Linear: evenly spaced by index, endpoints retained.
    idx = np.linspace(0, len(values) - 1, max_ticks).round().astype(int)
    return [values[i] for i in sorted(set(idx))]


def marker_stride(values: list[float]) -> int:
    """Show at most ~10 markers per curve so dense grids stay legible."""
    return max(1, len(values) // 10)


def _apply_sweep_xaxis(ax, sweep_key: str, sweep_values: list[float],
                       xscale: str | None = None) -> None:
    """Configure an axis' scale, ticks, labels and limits for the sweep.

    ``xscale`` overrides the sweep-config scale:
      * "sqrt" spreads a grid that is dense near zero (e.g. the equi demo's rho
        values) while keeping 0 on-axis.
      * "linear-minor" is a plain linear axis tick-marked at round numbers with
        minor ticks in between, instead of one label per swept value. Use it when
        the grid has near-neighbours whose labels would collide when placed at the
        values themselves -- e.g. the demo-02 SNR grid, where 0.5 and 0.6 overlap
        on the default log axis.
    """
    cfg = sweep_config(sweep_key)
    categorical = cfg.get("categorical")

    if categorical:
        ticks = sorted(v for v in sweep_values if v in categorical)
        ax.set_xticks(ticks)
        ax.set_xticklabels([categorical[int(v)] for v in ticks])
        ax.set_xlim(min(ticks) - 0.5, max(ticks) + 0.5)
        return

    scale = xscale if xscale is not None else cfg["scale"]
    if scale == "linear-minor":
        span = max(sweep_values) - min(sweep_values)
        pad = span * 0.04 if span > 0 else 0.5
        ax.set_xlim(min(sweep_values) - pad, max(sweep_values) + pad)
        # Ticks come from the locators, not the sweep grid, so labels cannot
        # collide however tightly the swept values are spaced.
        ax.xaxis.set_major_locator(MaxNLocator(nbins="auto", steps=[1, 2, 2.5, 5, 10]))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.grid(which="minor", axis="x", alpha=0.25, linewidth=0.5)
        return

    if scale == "log":
        ax.set_xscale("log")
        ax.set_xlim(min(sweep_values) * 0.82, max(sweep_values) * 1.18)
    elif scale == "sqrt":
        ax.set_xscale(
            "function",
            functions=(
                lambda x: np.sqrt(np.maximum(x, 0.0)),
                lambda x: np.square(x),
            ),
        )
        span_sqrt = np.sqrt(max(sweep_values)) - np.sqrt(max(min(sweep_values), 0.0))
        pad = (span_sqrt * 0.04) ** 2 if span_sqrt > 0 else 0.5
        ax.set_xlim(min(sweep_values), max(sweep_values) + pad)
    else:
        span = max(sweep_values) - min(sweep_values)
        pad = span * 0.04 if span > 0 else 0.5
        ax.set_xlim(min(sweep_values) - pad, max(sweep_values) + pad)

    ax.set_xticks(choose_ticks(sweep_values, scale if scale != "sqrt" else "linear"))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    if scale == "sqrt":
        # Grid values cluster near zero even after the sqrt stretch — slant
        # the labels so neighbouring ticks cannot collide.
        ax.tick_params(axis="x", labelrotation=45)
        for tick_label in ax.get_xticklabels():
            tick_label.set_horizontalalignment("right")


def _draw_fdr_reference(ax, df: pd.DataFrame, sweep_key: str, tfdr: float,
                        sweep_values: list[float]):
    """Draw the FDR target reference on an FDR panel and return (handle, label).

    For a plain sweep this is a horizontal dotted tFDR line; for a tFDR *sweep*
    it is the identity line (realized FDR == target), against which the curves'
    over/under-shoot is read directly. Returns ``(None, None)`` if no reference
    applies (tfdr == 0 on a non-tFDR sweep).
    """
    if sweep_config(sweep_key).get("is_tfdr_sweep"):
        xs = sorted(sweep_values)
        line, = ax.plot(
            xs, xs, color=TFDR_COLOR, linestyle=":", linewidth=2.0, zorder=1,
            label="realized = target",
        )
        return line, "realized = target"
    if tfdr > 0:
        line = ax.axhline(
            tfdr, color=TFDR_COLOR, linestyle=":", linewidth=2.0, zorder=1,
            label=f"tFDR = {tfdr:g}",
        )
        return line, f"tFDR = {tfdr:g}"
    return None, None


def _plot_method_curve(ax, df, method, metric, index, color, mev, bands):
    """Plot one method's curve for one metric, optionally with a +/- sd band.

    Returns the Line2D handle (for building a shared legend), or None if the
    method has no data for this metric.
    """
    sub = df[(df["solver"] == method) & (df["metric"] == metric)].sort_values(
        "sweep"
    )
    if sub.empty:
        return None

    if bands:
        sd_sub = df[
            (df["solver"] == method) & (df["metric"] == SD_OF[metric])
        ].sort_values("sweep")
        if not sd_sub.empty and sd_sub["sweep"].tolist() == sub["sweep"].tolist():
            lo = (sub["value"].values - sd_sub["value"].values).clip(min=0.0)
            hi = sub["value"].values + sd_sub["value"].values
            if metric == "TPR":
                hi = hi.clip(max=1.0)
            ax.fill_between(sub["sweep"], lo, hi, color=color, alpha=0.12,
                            linewidth=0, zorder=1)

    line, = ax.plot(
        sub["sweep"], sub["value"],
        color=color, marker=MARKERS[index % len(MARKERS)],
        linestyle=line_style(method),
        markersize=6.5, markevery=mev, markeredgecolor="white",
        markeredgewidth=0.7, linewidth=2.0, label=method, zorder=3,
    )
    return line


def plot_fdr_tpr_vs_sweep(
    df: pd.DataFrame,
    sweep_key: str,
    title: str,
    tfdr: float = DEFAULT_TFDR,
    legend_title: str = DEFAULT_LEGEND_TITLE,
    bands: bool = False,
    xlabel: str | None = None,
    xscale: str | None = None,
) -> plt.Figure:
    """Pattern 1 — side-by-side FDR and TPR panels versus the swept quantity.

    One line per method, colours/markers keyed to the method. The FDR panel
    carries the tFDR reference (a horizontal line, or the identity line for a
    tFDR sweep) and is scaled to include it. With ``bands=True`` each curve gets
    a translucent +/- sd envelope from the CSV's sd_ metrics. ``xlabel``
    overrides the sweep-config axis label (e.g. the 'rho' default names AR(1),
    which is wrong for the equi demo).
    """
    methods = method_order(df)
    sweep_values = sorted(df["sweep"].unique())
    colors = row_colors(methods)
    cfg = sweep_config(sweep_key)
    if xlabel is not None:
        cfg = dict(cfg, label=xlabel)
    mev = marker_stride(sweep_values)

    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(15.0, 6.0), constrained_layout=False,
    )

    legend_handles: list = []
    legend_labels: list = []

    for ax, metric in zip(axes, METRICS):
        if metric == "FDR":
            ref_handle, ref_label = _draw_fdr_reference(
                ax, df, sweep_key, tfdr, sweep_values
            )
            if ref_handle is not None:
                legend_handles.append(ref_handle)
                legend_labels.append(ref_label)

        for index, method in enumerate(methods):
            line = _plot_method_curve(
                ax, df, method, metric, index, colors[method], mev, bands
            )
            if line is not None and metric == "FDR":
                legend_handles.append(line)
                legend_labels.append(method)

        _apply_sweep_xaxis(ax, sweep_key, sweep_values, xscale=xscale)
        ax.set_xlabel(cfg["label"], fontsize=12)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=12)
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.45)
        ax.tick_params(axis="both", labelsize=11)

        if metric == "FDR":
            ax.set_ylim(0.0, fdr_axis_top(df, sweep_key, tfdr))
        else:
            ax.set_ylim(-0.02, 1.02)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    legend_handles, legend_labels = legend_tfdr_lead(
        legend_handles, legend_labels, legend_title
    )
    legend = fig.legend(
        legend_handles, legend_labels,
        loc="center left", bbox_to_anchor=(0.81, 0.5),
        frameon=True, framealpha=1.0, facecolor="white", edgecolor="0.7",
        fontsize=10, ncol=1,
    )
    legend.get_frame().set_linewidth(0.8)

    fig.subplots_adjust(
        left=0.08, right=0.80,
        bottom=0.16 if xscale == "sqrt" else 0.12,
        top=0.86, wspace=0.28,
    )
    return fig


def write_fdr_tpr_vs_sweep_html(
    df: pd.DataFrame,
    sweep_key: str,
    title: str,
    out_html: Path,
    tfdr: float = DEFAULT_TFDR,
    legend_title: str = DEFAULT_LEGEND_TITLE,
    inline_js: bool = True,
    xlabel: str | None = None,
    xscale: str | None = None,
) -> bool:
    """Pattern 2 — self-contained interactive HTML (FDR/TPR subplots, hover, toggles).

    Clicking a method in the legend hides/shows its curve in *both* panels.
    ``xscale`` overrides the sweep-derived scale as in the static figures, so the
    html and the png/pdf of the same figure share an x-axis. Returns False (with
    a note) if plotly is not installed.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print(
            "note: plotly not installed - skipping interactive HTML "
            "(pip install plotly, or drop --plotly)."
        )
        return False

    methods = method_order(df)
    sweep_values = sorted(df["sweep"].unique())
    colors = row_colors(methods)
    cfg = sweep_config(sweep_key)
    if xlabel is not None:
        cfg = dict(cfg, label=xlabel)

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("FDR", "TPR"), horizontal_spacing=0.08
    )

    for metric, col in (("FDR", 1), ("TPR", 2)):
        for method in methods:
            sub = df[
                (df["solver"] == method) & (df["metric"] == metric)
            ].sort_values("sweep")
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["sweep"], y=sub["value"],
                    mode="lines+markers", name=method,
                    legendgroup=method, showlegend=(metric == "FDR"),
                    line=dict(color=rgba_to_css(colors[method]), width=2,
                              dash="dash" if stag_sibling(method) else "solid"),
                    marker=dict(size=7),
                    hovertemplate=(
                        f"<b>{method}</b><br>{cfg['label']}=%{{x}}<br>"
                        f"{metric}=%{{y:.4f}}<extra></extra>"
                    ),
                ),
                row=1, col=col,
            )

    if cfg.get("is_tfdr_sweep"):
        xs = sorted(sweep_values)
        fig.add_trace(
            go.Scatter(
                x=xs, y=xs, mode="lines", name="realized = target",
                line=dict(color="black", dash="dot", width=2),
                hoverinfo="skip", showlegend=True, legendrank=1,
            ),
            row=1, col=1,
        )
    elif tfdr > 0:
        fig.add_hline(
            y=tfdr, line=dict(color="black", dash="dot", width=2),
            annotation_text=f"tFDR = {tfdr:g}", annotation_position="top left",
            row=1, col=1,
        )

    scale = xscale if xscale is not None else cfg["scale"]
    xaxis_kwargs = dict(title_text=cfg["label"])
    if scale == "log":
        xaxis_kwargs.update(type="log", tickvals=choose_ticks(sweep_values, "log"))
    elif scale == "linear-minor":
        # Round-number ticks from plotly's autorange, plus minor ticks; no
        # per-value tickvals, so near-neighbours cannot overlap.
        xaxis_kwargs.update(type="linear", minor=dict(ticklen=4, showgrid=True))
    elif cfg.get("categorical"):
        cat = cfg["categorical"]
        present = sorted(v for v in sweep_values if v in cat)
        xaxis_kwargs.update(tickvals=present,
                            ticktext=[cat[int(v)] for v in present])
    for col in (1, 2):
        fig.update_xaxes(row=1, col=col, **xaxis_kwargs)
    fig.update_yaxes(title_text="False discovery rate", row=1, col=1)
    fig.update_yaxes(
        title_text="True positive rate", range=[-0.02, 1.02], row=1, col=2
    )

    # Full frame (box) around every panel, matching the static figures.
    fig.update_xaxes(
        showline=True, linewidth=1, linecolor="rgb(82,82,82)", mirror=True
    )
    fig.update_yaxes(
        showline=True, linewidth=1, linecolor="rgb(82,82,82)", mirror=True
    )

    fig.update_layout(
        title=title.replace("$", ""),  # plain text: avoid raw LaTeX in HTML
        template="plotly_white", height=600,
        legend=dict(title=f"{legend_title} (click to toggle)", traceorder="grouped"),
        hovermode="closest",
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs=(True if inline_js else "cdn"))
    print(f"Wrote: {out_html}")
    return True


def ordered_grid_rows(frames: dict) -> list[str]:
    """Union of the methods across per-column frames, in first-seen order, so
    the colour / marker assignment is identical in every column of the grid."""
    seen: list[str] = []
    for df in frames.values():
        for name in df["solver"]:
            if name not in seen:
                seen.append(name)
    return seen


def plot_comparison_grid(
    frames: dict,
    sweep_key: str,
    title: str,
    legend_title: str = DEFAULT_LEGEND_TITLE,
    tfdr: float = DEFAULT_TFDR,
    column_title: str = "",
    bands: bool = False,
    xlabel: str | None = None,
    xscale: str | None = None,
) -> plt.Figure:
    """Pattern 3 — cross-CSV comparison grid.

    ``frames`` is an *ordered* mapping ``{column_label: (df, sweep_key)}`` ... no:
    ``{column_label: df}`` (all sharing one sweep_key). The figure is a 2xN grid:
    rows = metric (FDR top, TPR below), columns = the frames' keys (HAC linkage
    in the BT demos 02-05, or noise scenario / support placement). Every panel carries
    all methods as lines vs the sweep, colours/markers keyed to the method so a
    given method is the same colour in every column. All FDR panels share one
    y-scale and all TPR panels share [0, 1], so columns are directly comparable.
    """
    columns = list(frames.keys())
    methods = ordered_grid_rows(frames)
    colors = row_colors(methods)
    markers = {m: MARKERS[i % len(MARKERS)] for i, m in enumerate(methods)}
    cfg = sweep_config(sweep_key)
    if xlabel is not None:
        cfg = dict(cfg, label=xlabel)

    all_sweep = sorted(
        {v for df in frames.values() for v in df["sweep"].unique()}
    )
    mev = marker_stride(all_sweep)
    fdr_top = max(fdr_axis_top(df, sweep_key, tfdr) for df in frames.values())

    fig, axes = plt.subplots(
        nrows=2, ncols=len(columns),
        figsize=(5.4 * len(columns) + 2.4, 9.0), squeeze=False,
    )

    handles: list = []
    labels: list = []

    for row_idx, metric in enumerate(METRICS):
        for col, column in enumerate(columns):
            ax = axes[row_idx, col]
            cdf = frames[column]

            if metric == "FDR":
                ref_handle, ref_label = _draw_fdr_reference(
                    ax, cdf, sweep_key, tfdr, all_sweep
                )
                if ref_handle is not None and not handles:
                    handles.append(ref_handle)
                    labels.append(ref_label)

            for index, method in enumerate(methods):
                sub = cdf[
                    (cdf["solver"] == method) & (cdf["metric"] == metric)
                ].sort_values("sweep")
                if sub.empty:
                    continue
                if bands:
                    sd_sub = cdf[
                        (cdf["solver"] == method)
                        & (cdf["metric"] == SD_OF[metric])
                    ].sort_values("sweep")
                    if (not sd_sub.empty
                            and sd_sub["sweep"].tolist() == sub["sweep"].tolist()):
                        lo = (sub["value"].values
                              - sd_sub["value"].values).clip(min=0.0)
                        hi = sub["value"].values + sd_sub["value"].values
                        if metric == "TPR":
                            hi = hi.clip(max=1.0)
                        ax.fill_between(sub["sweep"], lo, hi,
                                        color=colors[method], alpha=0.12,
                                        linewidth=0, zorder=1)
                line, = ax.plot(
                    sub["sweep"], sub["value"],
                    color=colors[method], marker=markers[method],
                    linestyle=line_style(method),
                    markersize=6.0, markevery=mev, markeredgecolor="white",
                    markeredgewidth=0.7, linewidth=1.9, label=method, zorder=3,
                )
                if row_idx == 0 and col == 0:
                    handles.append(line)
                    labels.append(method)

            _apply_sweep_xaxis(ax, sweep_key, all_sweep, xscale=xscale)
            ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.45)
            ax.tick_params(axis="both", labelsize=11)

            if metric == "FDR":
                ax.set_ylim(0.0, fdr_top)
            else:
                ax.set_ylim(-0.02, 1.02)

            if col == 0:
                ax.set_ylabel(METRIC_LABELS[metric], fontsize=12)
            if row_idx == 0:
                head = f"{column_title}: {column}" if column_title else column
                ax.set_title(head, fontsize=14, fontweight="bold")
            if row_idx == len(METRICS) - 1:
                ax.set_xlabel(cfg["label"], fontsize=12)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)

    handles, labels = legend_tfdr_lead(handles, labels, legend_title)
    legend = fig.legend(
        handles, labels, loc="center left",
        bbox_to_anchor=(0.845, 0.5), frameon=True, framealpha=1.0,
        facecolor="white", edgecolor="0.7", fontsize=10,
        ncol=1,
    )
    legend.get_frame().set_linewidth(0.8)

    # Slanted tick labels (sqrt scale) need extra room between the metric
    # rows and below the bottom row.
    rotated = xscale == "sqrt"
    fig.subplots_adjust(
        left=0.07, right=0.84,
        bottom=0.12 if rotated else 0.08, top=0.90,
        wspace=0.18, hspace=0.30 if rotated else 0.12,
    )
    return fig


# ---------------------------------------------------------------------
# Demo 01 — gap x rho 2D study (wider CSV schema than the tidy sweeps)
# ---------------------------------------------------------------------
# The AR(1) demo's Part-3 output uses a wider schema,
#     solver,metric,rho,support_type,gap,value
# with metrics mean_TPP / sd_TPP / mean_FDP / sd_FDP over a 2D (gap x rho) grid
# under CappedSpread support, plus a Random support baseline (gap column empty).
# It is read via read_gap_rho() and rendered by plot_gap_rho().

GAP_RHO_EPS = 0.02  # correlation cut used to define the DA+AR1 window half-width


def read_gap_rho(path: Path) -> pd.DataFrame:
    """Load the wide demo-01 gap x rho CSV, normalising column/metric whitespace."""
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n  {path}\n"
            "Pass the gap_rho results CSV path as the first argument."
        )
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    required = {"solver", "metric", "rho", "support_type", "gap", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}\n"
            f"Found columns: {df.columns.tolist()}"
        )
    df["metric"] = df["metric"].astype(str).str.strip()
    df["support_type"] = df["support_type"].astype(str).str.strip()
    for col in ("rho", "value"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    df["gap"] = pd.to_numeric(df["gap"], errors="coerce")  # empty for Random
    return annotate_solver_params(df)


def kappa_boundary(rho_values: np.ndarray) -> np.ndarray:
    """DA+AR1 correction-window half-width kappa = ceil(log(eps)/log(rho)).

    Active variables spaced closer than kappa fall inside each other's window,
    where TPR is expected to collapse. Undefined at rho=0 (no correlation, so no
    window) -> NaN, so it is simply not drawn there.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.ceil(np.log(GAP_RHO_EPS) / np.log(rho_values))
    kappa[~np.isfinite(kappa)] = np.nan
    kappa[rho_values <= 0] = np.nan
    return kappa


def _gap_rho_grid(df: pd.DataFrame, metric: str):
    """Pivot the CappedSpread rows of one metric into a gap x rho grid.

    Returns (rho_values, gap_values, Z) with Z[i,j] = metric at
    (gap=gap_values[i], rho=rho_values[j]); rows sorted by ascending gap.
    """
    sub = df[(df["support_type"] == "CappedSpread") & (df["metric"] == metric)]
    # Note: multiple solvers share each (gap, rho) cell; pivot_table averages
    # them, so the heatmaps show the mean over the DA solvers.
    grid = sub.pivot_table(index="gap", columns="rho", values="value", aggfunc="mean")
    grid = grid.sort_index(ascending=True)
    return grid.columns.values, grid.index.values, grid.values


def plot_gap_rho(df: pd.DataFrame, title: str, tfdr: float = DEFAULT_TFDR) -> plt.Figure:
    """Pattern 4 — demo-01 gap x rho study.

    Top row: TPR and FDR heatmaps (MC means) over (rho on x, gap on y) for
    CappedSpread support, with the kappa-boundary curve gap = kappa(rho)
    overlaid -- below it, active variables share correction windows and TPP is
    expected to drop. Bottom row: the Random support baseline (TPR / FDR vs
    rho, one line per solver), the reference against which the deterministic capped
    spacing is read.
    """
    solvers = sorted(df["solver"].astype(str).str.strip().unique())
    method = "mean over " + " / ".join(solvers) if len(solvers) > 1 else solvers[0]
    rho_vals, gap_vals, tpp = _gap_rho_grid(df, "mean_TPP")
    _, _, fdp = _gap_rho_grid(df, "mean_FDP")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14.0, 10.5))
    (ax_tpp, ax_fdp), (ax_rt, ax_rf) = axes

    # --- Heatmaps (CappedSpread) with the kappa-boundary overlaid ------------
    heat_specs = [
        (ax_tpp, tpp, "TPR (true positive rate)", "viridis", 0.0, 1.0),
        (ax_fdp, fdp, "FDR (false discovery rate)", "magma_r", 0.0, None),
    ]
    for ax, Z, subtitle, cmap, vmin, vmax in heat_specs:
        im = ax.imshow(
            Z, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
            extent=[rho_vals.min(), rho_vals.max(),
                    0, len(gap_vals)],
        )
        # Discrete gap rows: label the y centres with the actual gap values.
        ax.set_yticks(np.arange(len(gap_vals)) + 0.5)
        ax.set_yticklabels([f"{int(g)}" for g in gap_vals])
        ax.set_xticks(rho_vals)
        ax.set_xticklabels([f"{r:g}" for r in rho_vals])
        # Annotate each cell with its value.
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if np.isfinite(Z[i, j]):
                    ax.text(rho_vals[j], i + 0.5, f"{Z[i, j]:.2f}",
                            ha="center", va="center", fontsize=7,
                            color="white" if cmap == "viridis" and Z[i, j] < 0.6
                            else "0.15")
        # kappa-boundary: map kappa(rho) onto the discrete gap-row axis.
        fine_rho = np.linspace(max(rho_vals.min(), 1e-3), rho_vals.max(), 200)
        kap = kappa_boundary(fine_rho)
        gsorted = np.sort(gap_vals)
        krow = np.interp(kap, gsorted, np.arange(len(gsorted)) + 0.5,
                         left=np.nan, right=np.nan)
        ax.plot(fine_rho, krow, color="cyan", linewidth=2.2, linestyle="--",
                label=(r"$\kappa$-boundary  gap $=\lceil\log\varepsilon/\log\rho\rceil$,"
                       r" $\varepsilon=0.02$ (DA window, same for all solvers)"))
        ax.set_xlabel(r"AR(1) correlation $\rho$", fontsize=12)
        ax.set_ylabel("CappedSpread gap between active vars", fontsize=12)
        ax.set_title(subtitle, fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Random support baseline (1D vs rho) --------------------------------
    rnd = df[df["support_type"] == "Random"]
    solver_markers = ("o", "s", "^", "D", "v")
    solver_colors = row_colors(list(solvers))
    for ax, metric, abbrev, ylabel, ylim in (
        (ax_rt, "mean_TPP", "TPR", "True positive rate", (-0.02, 1.02)),
        (ax_rf, "mean_FDP", "FDR", "False discovery rate", None),
    ):
        for si, solver in enumerate(solvers):
            sub = (rnd[(rnd["metric"] == metric) & (rnd["solver"] == solver)]
                   .sort_values("rho"))
            ax.plot(sub["rho"], sub["value"],
                    color=solver_colors[solver], linestyle=line_style(solver),
                    marker=solver_markers[si % len(solver_markers)],
                    markersize=7, markeredgecolor="white", linewidth=2.2,
                    label=solver)
        if metric == "mean_FDP" and tfdr > 0:
            # tFDR reference + headroom above it, suite convention.
            ax.axhline(tfdr, color=TFDR_COLOR, linestyle=":", linewidth=2.0,
                       zorder=1, label=f"tFDR = {tfdr:g}")
            fdp_max = rnd.loc[rnd["metric"] == "mean_FDP", "value"].max()
            ylim = (0.0, max(fdp_max, tfdr) * Y_HEADROOM)
        ax.set_xticks(sorted(rnd["rho"].unique()))
        ax.set_xlabel(r"AR(1) correlation $\rho$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"Random support baseline — {abbrev}", fontsize=13,
                     fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
        handles, labels = ax.get_legend_handles_labels()
        lh, ll = legend_tfdr_lead(handles, labels, "Method")
        ax.legend(lh, ll, loc="best", fontsize=10, framealpha=0.9)
        if ylim:
            ax.set_ylim(*ylim)

    fig.suptitle(f"{title}\n({method})", fontsize=15, fontweight="bold", y=0.995)
    fig.subplots_adjust(
        left=0.07, right=0.97, bottom=0.07, top=0.90, wspace=0.22, hspace=0.32
    )
    return fig


# ---------------------------------------------------------------------
# Pattern 5 — generic two-support 2D sweep (demo-07 NN studies)
# ---------------------------------------------------------------------
# Demo 07 records its 2D Monte Carlo studies (kappa x rho on NN data, and the
# SNR x rho method-mismatch study on AR(1) data) in the tidy two-support schema
#     solver,support_type,metric,<ROW>,<COL>,value
# with metrics mean_TPP / sd_TPP / mean_FDP / sd_FDP on a full 2D grid for every
# (solver, support_type) pair. Unlike the demo-01 gap x rho study (which has a
# CappedSpread-only grid plus a 1D Random baseline and a DA-window overlay),
# both support placements cover the full grid here, so the pattern is generic:
# one figure per support placement, TPR / FDR heatmap rows x one column per
# solver, FDR cells that exceed the tFDR target outlined in cyan.

SUPPORT_SUFFIX = {"cappedspread": "capped", "random": "random"}


def read_sweep2d(path: Path) -> tuple[pd.DataFrame, str, str]:
    """Load a tidy two-support 2D sweep CSV.

    The 4th and 5th columns are the two swept quantities (row axis and column
    axis of the heatmaps); their names vary between studies, so they are
    detected positionally and returned alongside the frame.

    Returns ``(df, row_name, col_name)`` with df columns
    ``solver, support_type, metric, row, col, value``.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n  {path}\n"
            "Pass the 2D sweep results CSV path as the first argument."
        )
    df = pd.read_csv(path)
    if df.shape[1] != 6:
        raise ValueError(
            "CSV must have 6 columns (solver,support_type,metric,<ROW>,<COL>,"
            f"value); found columns: {df.columns.tolist()}"
        )
    row_name, col_name = str(df.columns[3]), str(df.columns[4])
    df.columns = ["solver", "support_type", "metric", "row", "col", "value"]
    for col in ("solver", "support_type", "metric"):
        df[col] = df[col].astype(str).str.strip()
    for col in ("row", "col", "value"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    required = {"mean_TPP", "mean_FDP"}
    missing = required - set(df["metric"])
    if missing:
        raise ValueError(
            f"CSV is missing required metric(s): {sorted(missing)}\n"
            f"Found metrics: {sorted(set(df['metric']))}"
        )
    return annotate_solver_params(df), row_name, col_name


def _sweep2d_grid(df: pd.DataFrame, solver: str, metric: str):
    """Pivot one (solver, metric) slice into a row x col grid.

    Returns (row_values, col_values, Z) with Z[i,j] = metric at
    (row=row_values[i], col=col_values[j]), both axes ascending.
    """
    sub = df[(df["solver"] == solver) & (df["metric"] == metric)]
    grid = sub.pivot_table(index="row", columns="col", values="value")
    grid = grid.sort_index(ascending=True)
    return grid.index.values, grid.columns.values, grid.values


def _annotate_heatmap(ax, Z, cmap, vmin, vmax):
    """Write each finite cell's value, in white on dark cells / dark on light."""
    span = (vmax - vmin) or 1.0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if not np.isfinite(Z[i, j]):
                continue
            r, g, b = cmap((Z[i, j] - vmin) / span)[:3]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if luminance < 0.5 else "0.15")


def plot_sweep2d(
    df: pd.DataFrame,
    support: str,
    title: str,
    row_label: str,
    col_label: str,
    tfdr: float = DEFAULT_TFDR,
) -> plt.Figure:
    """Pattern 5 — one support placement of a two-support 2D sweep.

    A 2 x n_solvers panel grid: rows = TPR / FDR heatmaps over the 2D sweep
    (row quantity on y, column quantity on x), one column per solver, on shared
    colour scales (TPR fixed to [0, 1]; FDR shared across the whole CSV so the
    CappedSpread and Random figures are directly comparable). FDR cells above
    the tFDR target are outlined in cyan.
    """
    solvers = list(dict.fromkeys(df["solver"]))
    sub = df[df["support_type"] == support]
    # Shared FDR scale across supports and solvers (never below tFDR, so the
    # violation outlines sit inside the mapped range).
    fdr_top = max(df.loc[df["metric"] == "mean_FDP", "value"].max(), tfdr)

    n_cols = len(sub["col"].unique())
    fig_w = max(11.5, 1.8 + len(solvers) * (1.6 + 0.42 * n_cols))
    # Every panel carries its own x-axis (ticks + label); only y is shared,
    # keeping the left axis reserved for the row quantity.
    fig, axes = plt.subplots(
        nrows=2, ncols=len(solvers), figsize=(fig_w, 9.0),
        squeeze=False, sharey=True, gridspec_kw={"hspace": 0.35},
    )

    metric_specs = [
        ("mean_TPP", "TPR (true positive rate)", plt.get_cmap("viridis"), 0.0, 1.0),
        ("mean_FDP", "FDR (false discovery rate)", plt.get_cmap("magma_r"), 0.0, fdr_top),
    ]
    # Violation outline: cyan reads clearly on both ends of the magma_r FDR
    # palette (light orange and dark purple), where red would blend in.
    violation_color = "cyan"
    for mi, (metric, metric_label, cmap, vmin, vmax) in enumerate(metric_specs):
        im = None
        for si, solver in enumerate(solvers):
            ax = axes[mi][si]
            row_vals, col_vals, Z = _sweep2d_grid(sub, solver, metric)
            im = ax.imshow(Z, aspect="auto", origin="lower", cmap=cmap,
                           vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(len(col_vals)))
            ax.set_xticklabels([f"{v:g}" for v in col_vals])
            ax.set_yticks(np.arange(len(row_vals)))
            ax.set_yticklabels([f"{v:g}" for v in row_vals])
            _annotate_heatmap(ax, Z, cmap, vmin, vmax)
            if metric == "mean_FDP" and tfdr > 0:
                for i in range(Z.shape[0]):
                    for j in range(Z.shape[1]):
                        if np.isfinite(Z[i, j]) and Z[i, j] > tfdr:
                            ax.add_patch(plt.Rectangle(
                                (j - 0.5, i - 0.5), 1.0, 1.0, fill=False,
                                edgecolor=violation_color, linewidth=1.8))
            # The metric heads every panel of its row; the solver tops the
            # figure columns. The y-axis stays reserved for the row quantity.
            if mi == 0:
                ax.set_title(f"{solver}\n{metric_label}", fontsize=12,
                             fontweight="bold")
            else:
                ax.set_title(metric_label, fontsize=11, fontweight="bold")
            ax.set_xlabel(col_label, fontsize=11)
            if si == 0:
                ax.set_ylabel(row_label, fontsize=11)
        # One shared colorbar per metric row (identical scale across solvers).
        # `im` is unset only if the solver list was empty, in which case the row
        # has no panels to attach a colorbar to.
        if im is not None:
            cbar = fig.colorbar(im, ax=list(axes[mi]), fraction=0.025, pad=0.015)
            if metric == "mean_FDP" and tfdr > 0:
                cbar.ax.axhline(tfdr, color=violation_color, linewidth=1.8)
    if tfdr > 0:
        fig.text(0.5, 0.015,
                 f"cyan boxes / colorbar line: realized FDR above the "
                 f"tFDR = {tfdr:g} target",
                 ha="center", va="bottom", fontsize=10, color="darkcyan")
    fig.suptitle(f"{title}\n({support} support)", fontsize=15,
                 fontweight="bold", y=0.98)
    return fig


# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------
def save_figure(
    fig: plt.Figure, outdir: Path, stem: str, formats: list[str]
) -> None:
    """Write the figure to outdir/<stem>.<ext> for each requested format."""
    outdir.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        output_path = outdir / f"{stem}.{extension}"
        fig.savefig(
            output_path, dpi=300 if extension == "png" else None,
            bbox_inches="tight", facecolor="white",
        )
        print(f"Wrote: {output_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "csv_path", type=Path,
        help="Path to the tidy results CSV (solver,metric,<sweep>,value).",
    )
    parser.add_argument(
        "--title", default=None,
        help="Figure super-title (default: derived from the CSV file name).",
    )
    parser.add_argument(
        "--legend-title", default=DEFAULT_LEGEND_TITLE,
        help=(
            "Legend title naming what the CSV's 'solver' column holds "
            f"(default: {DEFAULT_LEGEND_TITLE!r}; e.g. 'Method' for the DA-vs-"
            "base comparison, 'Solver' for the linkage-sweep demos)."
        ),
    )
    parser.add_argument(
        "--outdir", type=Path, default=None,
        help=(
            "Output directory (default: the simulation_results/plots/ sibling "
            "when the CSV lives in simulation_results/data/, else the CSV's own "
            "directory)."
        ),
    )
    parser.add_argument(
        "--stem", default=None,
        help="Output filename stem (default: <csv-stem>" + STEM_SUFFIX + ").",
    )
    parser.add_argument(
        "--formats", nargs="+", default=["png", "pdf"],
        help="Output formats to write (default: png pdf).",
    )
    parser.add_argument(
        "--tfdr", type=float, default=DEFAULT_TFDR,
        help=(
            "Target FDR level drawn as a dotted reference on the FDR panel "
            f"(default: {DEFAULT_TFDR:g}; demo-specific -- e.g. 0.1 for demo 02)."
            " Pass 0 to omit. Ignored for a tFDR sweep (identity line used)."
        ),
    )
    parser.add_argument(
        "--bands", action="store_true",
        help="Shade +/- sd bands (from the CSV's sd_FDR / sd_TPR metrics).",
    )
    parser.add_argument(
        "--xlabel", default=None,
        help=(
            "Override the x-axis label derived from the sweep column "
            "(e.g. the 'rho' default is the AR(1) wording; pass "
            "'Equicorrelation $\\rho$' for the equi demo)."
        ),
    )
    parser.add_argument(
        "--xscale", default=None,
        choices=["linear", "linear-minor", "sqrt", "log"],
        help=(
            "Override the sweep-derived x-axis scale. 'sqrt' spreads grids dense "
            "near zero, e.g. the equi rho sweep. 'linear-minor' ticks at round "
            "numbers with minor ticks in between instead of at each swept value, "
            "for grids whose labels would otherwise collide (e.g. the demo-02 SNR "
            "grid's 0.5 / 0.6)."
        ),
    )
    parser.add_argument(
        "--relabel", nargs="+", default=[], metavar="OLD=NEW",
        help=(
            "Rename method labels in the 'solver' column for display "
            "(substring replacement, applied in order; e.g. "
            "'TREX-DA=TREX-DA+BT'). Cosmetic only — the CSV is unchanged."
        ),
    )
    parser.add_argument(
        "--plotly", action="store_true",
        help=(
            "Also write the interactive Plotly HTML (<stem>.html, about 5 MB, "
            "gitignored). Off by default."
        ),
    )
    parser.add_argument(
        "--plotly-cdn", action="store_true",
        help=(
            "Write the interactive HTML with plotly.js loaded from a CDN: a tiny "
            "file that needs internet to render (implies --plotly)."
        ),
    )
    # Former switch from the time the HTML was written by default; now a no-op.
    parser.add_argument("--no-plotly", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main_fdr_tpr() -> None:
    """Default mode: per-CSV overview (+ interactive HTML with --plotly)."""
    args = parse_args()
    csv_path = args.csv_path.resolve()

    df, sweep_key = read_mc_results(csv_path)
    df = apply_relabel(df, args.relabel)
    title = (
        args.title if args.title is not None
        else f"DA-TRex Monte Carlo simulation ({csv_path.stem})"
    )
    outdir = args.outdir if args.outdir is not None else default_plots_dir(csv_path)
    stem = args.stem if args.stem is not None else csv_path.stem + STEM_SUFFIX
    outdir = outdir.resolve()

    fig = plot_fdr_tpr_vs_sweep(
        df, sweep_key, title, tfdr=args.tfdr,
        legend_title=args.legend_title, bands=args.bands,
        xlabel=args.xlabel, xscale=args.xscale,
    )
    save_figure(fig, outdir, stem, args.formats)
    plt.close(fig)

    if args.plotly or args.plotly_cdn:   # interactive HTML only on request
        write_fdr_tpr_vs_sweep_html(
            df, sweep_key, title, outdir / f"{stem}.html",
            tfdr=args.tfdr, legend_title=args.legend_title,
            inline_js=not args.plotly_cdn,
            xlabel=args.xlabel, xscale=args.xscale,
        )


def main_grid(argv: list) -> None:
    """`grid` mode: cross-CSV comparison grid (columns = linkage / scenario /
    support). ``--labels`` and ``--csvs`` are parallel lists; a listed CSV that
    does not exist yet is skipped with a note, so the grid renders whatever runs
    are available."""
    parser = argparse.ArgumentParser(
        prog="trex_da_plt_utils.py grid",
        description="Cross-CSV FDR/TPR comparison grid.",
    )
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Column labels, one per CSV (e.g. Single Complete Average).")
    parser.add_argument("--csvs", nargs="+", required=True, type=Path,
                        help="Tidy CSVs, parallel to --labels.")
    parser.add_argument("--title", default="DA-TRex comparison")
    parser.add_argument("--legend-title", default=DEFAULT_LEGEND_TITLE)
    parser.add_argument("--column-title", default="",
                        help="Prefix for each column header (e.g. 'Linkage').")
    parser.add_argument("--stem", required=True, help="Output filename stem.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--tfdr", type=float, default=DEFAULT_TFDR)
    parser.add_argument("--bands", action="store_true")
    parser.add_argument("--xlabel", default=None,
                        help="Override the sweep-derived x-axis label.")
    parser.add_argument("--xscale", default=None,
                        choices=["linear", "linear-minor", "sqrt", "log"],
                        help="Override the sweep-derived x-axis scale.")
    parser.add_argument("--relabel", nargs="+", default=[], metavar="OLD=NEW",
                        help="Rename method labels for display (see default mode).")
    args, _ignored = parser.parse_known_args(argv)

    if len(args.labels) != len(args.csvs):
        raise SystemExit(
            f"--labels ({len(args.labels)}) and --csvs ({len(args.csvs)}) "
            "must have the same length."
        )

    frames = {}
    sweep_keys = set()
    for label, csv in zip(args.labels, args.csvs):
        csv = csv.resolve()
        if not csv.exists():
            print(f"note: skipping {label} - CSV not found:\n  {csv}")
            continue
        df, sweep_key = read_mc_results(csv)
        frames[label] = apply_relabel(df, args.relabel)
        sweep_keys.add(sweep_key)

    if not frames:
        raise SystemExit(
            "No input CSVs found - run the demo first, then re-run this plotter."
        )
    if len(sweep_keys) > 1:
        raise SystemExit(
            f"Grid columns sweep different quantities {sorted(sweep_keys)}; "
            "all --csvs must share one sweep axis."
        )
    sweep_key = sweep_keys.pop()

    fig = plot_comparison_grid(
        frames, sweep_key, args.title, legend_title=args.legend_title,
        tfdr=args.tfdr, column_title=args.column_title, bands=args.bands,
        xlabel=args.xlabel, xscale=args.xscale,
    )
    save_figure(fig, args.outdir.resolve(), args.stem, args.formats)
    plt.close(fig)


def main_gaprho(argv: list) -> None:
    """`gaprho` mode: demo-01 gap x rho 2D study."""
    parser = argparse.ArgumentParser(
        prog="trex_da_plt_utils.py gaprho",
        description="Demo-01 gap x rho study (wide CSV schema).",
    )
    parser.add_argument(
        "csv_path", type=Path,
        help="Wide gap_rho CSV (solver,metric,rho,support_type,gap,value).",
    )
    parser.add_argument(
        "--title",
        default="DA-TRex AR(1): support spacing (gap) vs correlation ($\\rho$)",
    )
    parser.add_argument("--stem", required=True, help="Output filename stem.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument(
        "--tfdr", type=float, default=DEFAULT_TFDR,
        help="Target FDR reference for the Random-baseline FDP panel "
             "(0 disables the line).",
    )
    args, _ignored = parser.parse_known_args(argv)

    df = read_gap_rho(args.csv_path.resolve())
    fig = plot_gap_rho(df, args.title, tfdr=args.tfdr)
    save_figure(fig, args.outdir.resolve(), args.stem, args.formats)
    plt.close(fig)


def main_sweep2d(argv: list) -> None:
    """`sweep2d` mode: generic two-support 2D sweep heatmaps (demo 07)."""
    parser = argparse.ArgumentParser(
        prog="trex_da_plt_utils.py sweep2d",
        description="Per-solver TPR/FDR heatmaps over a 2D sweep, one figure "
                    "per support placement (tidy two-support CSV schema).",
    )
    parser.add_argument(
        "csv_path", type=Path,
        help="Tidy 2D sweep CSV (solver,support_type,metric,<ROW>,<COL>,value).",
    )
    parser.add_argument("--title", default="DA-TRex 2D sweep")
    parser.add_argument("--xlabel", default=None,
                        help="Heatmap x-axis label (default: derived from the "
                             "CSV's 5th column via the sweep-axis config).")
    parser.add_argument("--ylabel", default=None,
                        help="Heatmap y-axis label (default: derived from the "
                             "CSV's 4th column via the sweep-axis config).")
    parser.add_argument("--stem", required=True,
                        help="Output filename stem; the support placement is "
                             "appended (<stem>_capped / <stem>_random).")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument(
        "--tfdr", type=float, default=DEFAULT_TFDR,
        help="Target FDR level; FDR cells above it are outlined in cyan "
             "(0 disables the outlines).",
    )
    args, _ignored = parser.parse_known_args(argv)

    df, row_name, col_name = read_sweep2d(args.csv_path.resolve())
    row_label = (args.ylabel if args.ylabel is not None
                 else sweep_config(sweep_key_of(row_name))["label"])
    col_label = (args.xlabel if args.xlabel is not None
                 else sweep_config(sweep_key_of(col_name))["label"])

    for support in dict.fromkeys(df["support_type"]):
        fig = plot_sweep2d(df, support, args.title, row_label, col_label,
                           tfdr=args.tfdr)
        suffix = SUPPORT_SUFFIX.get(support.lower(), support.lower())
        save_figure(fig, args.outdir.resolve(), f"{args.stem}_{suffix}",
                    args.formats)
        plt.close(fig)


def main() -> None:
    """Dispatch on the first token: `grid` / `gaprho` / `sweep2d` sub-modes,
    else the default per-CSV FDR/TPR mode (so bare `trex_da_plt_utils.py <csv>
    ...` works unchanged for the single-CSV demos)."""
    argv = sys.argv[1:]
    if argv and argv[0] == "grid":
        main_grid(argv[1:])
    elif argv and argv[0] == "gaprho":
        main_gaprho(argv[1:])
    elif argv and argv[0] == "sweep2d":
        main_sweep2d(argv[1:])
    else:
        main_fdr_tpr()


if __name__ == "__main__":
    main()
