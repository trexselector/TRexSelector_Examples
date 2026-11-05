#!/usr/bin/env python3
"""Plotting utilities for the T-Rex+GVS demonstration suite.

Reads the tidy result CSVs written by the demos (see each demo's
``simulation_results/data/``) and renders publication-style figures into the
sibling ``simulation_results/plots/`` folder. The visual conventions mirror the
DA-TRex suite (``../trex_da/trex_da_plt_utils.py``): viridis for TPR, magma_r
for FDR, cyan outlines / colorbar ticks marking realized FDR above the tFDR
target, PNG + PDF output.

Three CSV shapes are auto-detected from their columns:

* **2-D sweep** — ``method,metric,row_label,col_label,mean,sd`` (the ``*_2d.csv``
  files). Rendered as a 2 x 3 heatmap grid: rows = TPR / FDR, columns = the
  three solvers (TENET / TENET_AUG / TIENET_AUG); row axis = SNR, column axis =
  the swept correlation (rho or ar_coef).

* **Method comparison** — ``method,metric,<param>,mean,sd`` (demo 01's
  ``lambda2_method``, ``hc_linkage`` and ``cut`` studies). Rendered as grouped
  bar charts (TPR and FDR panels, bars grouped by parameter level, one colour
  per solver, +/- 1 sd error bars, tFDR reference line on the FDR panel). The
  cut study adds full-group recovery and group FDR panels (2 x 2 grid).

* **Block benchmark** — ``scenario,rho,snr,method,...`` (demo 08). Rendered as
  TPR / FDR line plots, one line per method, tFDR reference on the FDR row.
  ``--vs snr`` (default) puts SNR on the x-axis with one panel per rho;
  ``--vs rho`` puts rho on the x-axis with one panel per SNR (the demo 08
  headline view: the oracle-vs-clustered gap as a function of rho).
  ``--panels`` restricts which panel values are drawn.

Usage:
    trex_gvs_plt_utils.py <results.csv> [--outdir DIR] [--title T]
                          [--tfdr F] [--formats png pdf] [--param-label L]
                          [--vs snr|rho] [--panels V ...] [--stem S]
                          [--xscale {linear,sqrt,log}]

``--xscale`` sets the x-axis scale of the block-benchmark line plots (the
heatmap and bar renderers are categorical and unaffected). The default is
``linear`` with minor ticks, matching the rest of the suites; ``sqrt`` spreads
a grid that is dense near zero while keeping 0 on-axis, and ``log`` restores a
logarithmic axis.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import (  # noqa: E402
    AutoMinorLocator, MaxNLocator, ScalarFormatter,
)

DEFAULT_TFDR = 0.1
SOLVER_ORDER = ["TENET", "TENET_AUG", "TIENET_AUG"]
# Block-benchmark method tags (demo 08 CSVs) -> legend labels.
METHOD_LABELS = {
    "M1(EN-C)": "EN + HAC groups",
    "M2(EN-O)": "EN + oracle groups",
    "M3(IE-C)": "IEN + HAC groups",
    "M4(IE-O)": "IEN + oracle groups",
}
# Per-method marker cycle for the block-benchmark line plots — overlapping
# curves (e.g. M1/M2 at high rho) stay distinguishable by marker shape.
METHOD_MARKERS = ["o", "s", "^", "D"]
# Axis-prefix -> human label for the swept quantity.
AXIS_LABEL = {"snr": "SNR", "rho": r"$\rho$", "ac": "AR coefficient",
              "ar_coef": "AR coefficient"}
VIOLATION = "cyan"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _split_axis(labels: pd.Series) -> tuple[str, pd.Series]:
    """'snr=0.2' -> axis 'snr', numeric value 0.2 (labels without '=' pass)."""
    axis = ""
    values = []
    for lbl in labels:
        s = str(lbl)
        if "=" in s:
            name, _, val = s.partition("=")
            axis = axis or name
            values.append(float(val))
        else:
            values.append(float(s))
    return axis, pd.Series(values, index=labels.index)


def _axis_label(prefix: str) -> str:
    return AXIS_LABEL.get(prefix, prefix)


def legend_tfdr_lead(handles: list, labels: list, group_label: str = "Method"):
    """Order legend entries as: tFDR reference first, then a group-header row
    (e.g. "Method:"), then the method entries below it.

    Replaces a legend *title*: a title always sits at the very top of the legend
    box, which would file the tFDR reference under the group heading it does not
    belong to. Returns (handles, labels) for a title-less legend. Mirrors the
    DA-TRex suite helper of the same name.
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


def place_figure_legend(fig, handles, labels, group_label: str = "Method",
                        anchor_x: float = 0.81, fontsize: int = 10):
    """Attach the single, figure-level legend just outside the right panel.

    One legend per figure, placed outside the axes so it can never obscure the
    data, in an opaque white box (``framealpha=1.0`` — no see-through
    background). Matches the T-Rex / DA-TRex suite convention; the caller is
    responsible for reserving the right margin via ``fig.subplots_adjust``.
    """
    handles, labels = legend_tfdr_lead(handles, labels, group_label)
    legend = fig.legend(
        handles, labels,
        loc="center left", bbox_to_anchor=(anchor_x, 0.5),
        frameon=True, framealpha=1.0, facecolor="white", edgecolor="0.7",
        fontsize=fontsize, ncol=1,
    )
    legend.get_frame().set_linewidth(0.8)
    return legend


def _annotate_cells(ax, Z, cmap, vmin, vmax):
    """Write each finite cell's value, white on dark / dark on light."""
    span = (vmax - vmin) or 1.0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if not np.isfinite(Z[i, j]):
                continue
            r, g, b = cmap((Z[i, j] - vmin) / span)[:3]
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if lum < 0.5 else "0.15")


def save_figure(fig: plt.Figure, outdir: Path, stem: str,
                formats: list[str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        path = outdir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
        print(f"Wrote: {path}")
    plt.close(fig)


def default_plots_dir(csv_path: Path) -> Path:
    """`.../simulation_results/data/x.csv` -> `.../simulation_results/plots`."""
    if csv_path.parent.name == "data":
        return csv_path.parent.parent / "plots"
    return csv_path.parent / "plots"


def apply_sweep_xaxis(ax, values: list[float], xscale: str = "linear") -> None:
    """Configure a line plot's x-axis scale, ticks and limits for a sweep grid.

    ``linear`` (default) and ``sqrt`` both take their major ticks from the
    locators rather than from the swept values, so labels cannot collide however
    tightly the grid is spaced, and both carry minor ticks with a faint minor
    x-grid. ``sqrt`` additionally stretches the region near zero (keeping 0
    on-axis), which is useful when the grid clusters there. ``log`` keeps the
    logarithmic axis this module used to hard-code for the SNR sweeps.

    Mirrors ``../trex_screening/trex_scr_plt_utils.apply_linear_minor_xaxis``
    and the DA-TRex suite's ``_apply_sweep_xaxis``.
    """
    lo, hi = min(values), max(values)
    if xscale == "log":
        ax.set_xscale("log")
        ax.set_xlim(lo * 0.82, hi * 1.18)
        return
    if xscale == "sqrt":
        ax.set_xscale(
            "function",
            functions=(
                lambda x: np.sqrt(np.maximum(x, 0.0)),
                lambda x: np.square(x),
            ),
        )
        span_sqrt = np.sqrt(hi) - np.sqrt(max(lo, 0.0))
        pad = (span_sqrt * 0.04) ** 2 if span_sqrt > 0 else 0.5
        ax.set_xlim(lo, hi + pad)
    else:
        span = hi - lo
        pad = span * 0.04 if span > 0 else 0.5
        ax.set_xlim(lo - pad, hi + pad)
    ax.xaxis.set_major_locator(MaxNLocator(nbins="auto", steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.grid(which="minor", axis="x", alpha=0.25, linewidth=0.5)


# ---------------------------------------------------------------------------
# 2-D sweep heatmap grid
# ---------------------------------------------------------------------------
def plot_2d_heatmaps(df: pd.DataFrame, title: str,
                     tfdr: float = DEFAULT_TFDR) -> plt.Figure:
    row_axis, df["_row"] = _split_axis(df["row_label"])
    col_axis, df["_col"] = _split_axis(df["col_label"])
    solvers = [s for s in SOLVER_ORDER if s in set(df["method"])]

    fdr_top = max(df.loc[df["metric"] == "mean_FDP", "mean"].max(), tfdr)
    n_col = df["_col"].nunique()
    fig_w = max(11.0, 1.6 + len(solvers) * (1.4 + 0.5 * n_col))
    fig, axes = plt.subplots(nrows=2, ncols=len(solvers),
                             figsize=(fig_w, 8.6), squeeze=False,
                             sharey=True, gridspec_kw={"hspace": 0.32})

    specs = [
        ("mean_TPP", "TPR (true positive rate)", plt.get_cmap("viridis"), 0.0, 1.0),
        ("mean_FDP", "FDR (false discovery rate)", plt.get_cmap("magma_r"), 0.0, fdr_top),
    ]
    for mi, (metric, mlabel, cmap, vmin, vmax) in enumerate(specs):
        im = None
        for si, solver in enumerate(solvers):
            ax = axes[mi][si]
            sub = df[(df["method"] == solver) & (df["metric"] == metric)]
            grid = sub.pivot_table(index="_row", columns="_col", values="mean")
            grid = grid.sort_index(ascending=True)
            row_vals, col_vals, Z = grid.index.values, grid.columns.values, grid.values
            im = ax.imshow(Z, aspect="auto", origin="lower", cmap=cmap,
                           vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(len(col_vals)))
            ax.set_xticklabels([f"{v:g}" for v in col_vals])
            ax.set_yticks(np.arange(len(row_vals)))
            ax.set_yticklabels([f"{v:g}" for v in row_vals])
            _annotate_cells(ax, Z, cmap, vmin, vmax)
            if metric == "mean_FDP" and tfdr > 0:
                for i in range(Z.shape[0]):
                    for j in range(Z.shape[1]):
                        if np.isfinite(Z[i, j]) and Z[i, j] > tfdr:
                            ax.add_patch(plt.Rectangle(
                                (j - 0.5, i - 0.5), 1.0, 1.0, fill=False,
                                edgecolor=VIOLATION, linewidth=1.8))
            if mi == 0:
                ax.set_title(f"{solver}\n{mlabel}", fontsize=12, fontweight="bold")
            else:
                ax.set_title(mlabel, fontsize=11, fontweight="bold")
            ax.set_xlabel(_axis_label(col_axis), fontsize=11)
            if si == 0:
                ax.set_ylabel(_axis_label(row_axis), fontsize=11)
        if im is not None:
            cbar = fig.colorbar(im, ax=list(axes[mi]), fraction=0.025, pad=0.015)
            if metric == "mean_FDP" and tfdr > 0:
                cbar.ax.axhline(tfdr, color=VIOLATION, linewidth=1.8)
    if tfdr > 0:
        fig.text(0.5, 0.012,
                 f"cyan boxes / colorbar line: realized FDR above tFDR = {tfdr:g}",
                 ha="center", va="bottom", fontsize=10, color="darkcyan")
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)
    return fig


# ---------------------------------------------------------------------------
# Method-comparison grouped bars (demo 01 lambda2_method / hc_linkage / cut)
# ---------------------------------------------------------------------------
def plot_method_bars(df: pd.DataFrame, param_col: str, title: str,
                     param_label: str, tfdr: float = DEFAULT_TFDR) -> plt.Figure:
    levels = list(dict.fromkeys(df[param_col]))
    solvers = [s for s in SOLVER_ORDER if s in set(df["method"])]
    colors = plt.get_cmap("tab10").colors

    panels = [("mean_TPP", "TPR (true positive rate)"),
              ("mean_FDP", "FDR (false discovery rate)")]
    # Cut study: group-level panels below the variable-level ones
    grouped = "grp_full" in set(df["metric"])
    if grouped:
        panels += [("grp_full", "Full-group recovery"),
                   ("grp_FDP", "Group FDR (groups without signal)")]
    width_in = max(11.0, 2.2 * len(levels) + 5)
    fig, axes = plt.subplots(2 if grouped else 1, 2,
                             figsize=(width_in, 8.6 if grouped else 4.8))
    axes = np.ravel(axes)
    x = np.arange(len(levels))
    width = 0.8 / max(len(solvers), 1)

    # Collected once across panels so the figure carries a single legend.
    legend_handles: list = []
    legend_labels: list[str] = []

    for ai, (metric, mlabel) in enumerate(panels):
        ax = axes[ai]
        for si, solver in enumerate(solvers):
            sub = df[(df["method"] == solver) & (df["metric"] == metric)]
            # A level the method was not run at (e.g. an IEN-only lambda2 rule) gets no bar.
            rows = [sub[sub[param_col] == lv] for lv in levels]
            means = [float(r["mean"].iloc[0]) if len(r) else np.nan for r in rows]
            sds = [float(r["sd"].iloc[0]) if len(r) else np.nan for r in rows]
            bars = ax.bar(x + (si - (len(solvers) - 1) / 2) * width, means, width,
                          yerr=sds, capsize=3, label=solver, color=colors[si],
                          error_kw={"elinewidth": 0.8, "alpha": 0.6})
            if ai == 0:
                legend_handles.append(bars)
                legend_labels.append(solver)
        ax.set_xticks(x)
        ax.set_xticklabels(levels, rotation=15 if max(map(len, map(str, levels))) > 6 else 0)
        ax.set_xlabel(param_label, fontsize=11)
        ax.set_title(mlabel, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        if metric in ("mean_TPP", "grp_full", "grp_FDP"):
            ax.set_ylim(0, 1.05)
        if metric == "mean_FDP" and tfdr > 0:
            ref = ax.axhline(tfdr, color="black", linestyle=":", linewidth=2.0)
            legend_handles.insert(0, ref)
            legend_labels.insert(0, f"tFDR = {tfdr:g}")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)
    place_figure_legend(fig, legend_handles, legend_labels, "Method",
                        anchor_x=0.815, fontsize=9)
    if grouped:
        fig.subplots_adjust(left=0.07, right=0.80, bottom=0.09, top=0.89,
                            wspace=0.22, hspace=0.38)
    else:
        fig.subplots_adjust(left=0.07, right=0.80, bottom=0.18, top=0.83, wspace=0.22)
    return fig


# ---------------------------------------------------------------------------
# Block-benchmark line plots (demo 08)
# ---------------------------------------------------------------------------
def plot_block_bench(df: pd.DataFrame, title: str,
                     tfdr: float = DEFAULT_TFDR, vs: str = "snr",
                     panels: list[float] | None = None,
                     xscale: str = "linear") -> plt.Figure:
    x_col = vs                              # x-axis quantity
    panel_col = "rho" if vs == "snr" else "snr"   # one panel per value
    panel_vals = sorted(df[panel_col].unique())
    if panels:
        panel_vals = [v for v in panel_vals
                      if any(np.isclose(v, w) for w in panels)]
        if not panel_vals:
            raise ValueError(f"--panels {panels} matches no {panel_col} value "
                             f"in the CSV ({sorted(df[panel_col].unique())})")
    methods = list(dict.fromkeys(df["method"]))
    colors = plt.get_cmap("tab10").colors
    # Shared across panels (sharex=True), so the axis is configured from the
    # full swept grid rather than one panel's slice.
    x_values = sorted(df[x_col].unique())

    fig, axes = plt.subplots(2, len(panel_vals),
                             figsize=(5.2 * len(panel_vals) + 2.0, 8.4),
                             squeeze=False, sharex=True)
    metric_rows = [("mean_coord_tpr", "TPR (true positive rate)", (0, 1.05)),
                   ("mean_coord_fdp", "FDR (false discovery rate)", None)]

    # Collected once across panels so the figure carries a single legend.
    legend_handles: list = []
    legend_labels: list[str] = []

    for mi, (metric, mlabel, ylim) in enumerate(metric_rows):
        for ri, pv in enumerate(panel_vals):
            ax = axes[mi][ri]
            sub = df[np.isclose(df[panel_col], pv)]
            for si, method in enumerate(methods):
                m = sub[sub["method"] == method].sort_values(x_col)
                line, = ax.plot(m[x_col], m[metric],
                                marker=METHOD_MARKERS[si % len(METHOD_MARKERS)],
                                markersize=5, markerfacecolor="none",
                                markeredgewidth=1.2, color=colors[si],
                                label=METHOD_LABELS.get(method, method))
                if mi == 0 and ri == 0:
                    legend_handles.append(line)
                    legend_labels.append(METHOD_LABELS.get(method, method))
            apply_sweep_xaxis(ax, x_values, xscale)
            ax.grid(alpha=0.3)
            if ylim:
                ax.set_ylim(*ylim)
            if metric == "mean_coord_fdp" and tfdr > 0:
                ref = ax.axhline(tfdr, color="black", linestyle=":", linewidth=2.0)
                if ri == 0:
                    legend_handles.insert(0, ref)
                    legend_labels.insert(0, f"tFDR = {tfdr:g}")
            if mi == 0:
                head = (rf"$\rho = {pv:g}$" if panel_col == "rho"
                        else rf"$\mathrm{{SNR}} = {pv:g}$")
                ax.set_title(head, fontsize=12, fontweight="bold")
            if mi == 1:
                ax.set_xlabel("SNR" if x_col == "snr" else r"$\rho$",
                              fontsize=11)
            if ri == 0:
                ax.set_ylabel(mlabel, fontsize=11)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.97)
    place_figure_legend(fig, legend_handles, legend_labels, "Method",
                        anchor_x=0.825, fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.81, bottom=0.08, top=0.88,
                        wspace=0.18, hspace=0.16)
    return fig


# ---------------------------------------------------------------------------
# Dispatch + CLI
# ---------------------------------------------------------------------------
def detect_kind(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if {"row_label", "col_label"} <= cols:
        return "2d"
    if {"scenario", "rho", "snr", "method"} <= cols:
        return "block"
    if {"method", "metric", "mean", "sd"} <= cols and len(df.columns) == 5:
        return "method"
    raise ValueError(f"Unrecognized CSV schema: columns = {sorted(cols)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render T-Rex+GVS result figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path, help="Result CSV (schema auto-detected).")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Output dir (default: sibling simulation_results/plots).")
    p.add_argument("--title", default=None, help="Figure title.")
    p.add_argument("--tfdr", type=float, default=DEFAULT_TFDR)
    p.add_argument("--formats", nargs="+", default=["png", "pdf"])
    p.add_argument("--param-label", default=None,
                   help="Axis label for method-comparison bar charts.")
    p.add_argument("--stem", default=None,
                   help="Output filename stem (default: CSV stem).")
    p.add_argument("--vs", choices=["snr", "rho"], default="snr",
                   help="Block benchmark only: x-axis quantity (panels are "
                        "formed over the other one).")
    p.add_argument("--panels", nargs="+", type=float, default=None,
                   help="Block benchmark only: restrict panels to these "
                        "values of the non-x quantity.")
    p.add_argument("--xscale", choices=("linear", "sqrt", "log"),
                   default="linear",
                   help="Block benchmark only: x-axis scale (default: linear). "
                        "'sqrt' spreads a grid that is dense near zero.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.csv}")
    df = pd.read_csv(args.csv, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()

    kind = detect_kind(df)
    outdir = args.outdir or default_plots_dir(args.csv)
    stem = args.stem or args.csv.stem
    title = args.title or args.csv.stem

    if kind == "2d":
        fig = plot_2d_heatmaps(df, title, args.tfdr)
    elif kind == "method":
        param_col = [c for c in df.columns
                     if c not in ("method", "metric", "mean", "sd")][0]
        label = args.param_label or param_col
        fig = plot_method_bars(df, param_col, title, label, args.tfdr)
    else:  # block
        fig = plot_block_bench(df, title, args.tfdr, args.vs, args.panels,
                               args.xscale)

    save_figure(fig, outdir, stem, args.formats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
