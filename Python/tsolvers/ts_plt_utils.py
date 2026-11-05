#!/usr/bin/env python3
"""
ts_plt_utils.py — shared plotting utilities for the T-solver demo suite.

Suite-level companion of ts_perf_utils.hpp. Each demo's generate_plots.sh
supplies its CSV files and titles, so no per-demo configuration is needed.

Plot patterns hosted here:
    plot_terminated_paths()  — coefficient paths vs. number of included dummies,
                               one panel per T_stop of a warm-started solver
                               (the figure of the R tlars vignette).
    plot_path_comparison()   — the same paths for several variants of one
                               solver (T-AFS rho values, T-NCGMP variants), one
                               panel per variant at its largest T_stop.
    plot_fdr_tpr_vs_T()      — FDR and TPR vs. T per solver, one column per
                               solver family (demo_ts_perf_01).
    plot_tpr_vs_fdr()        — TPR vs. FDR along the T grid, same layout.
    plot_path_stats()        — selected variables, steps and cumulative time
                               vs. T, same layout, with the exchangeability
                               line and the step limit as references.
    plot_refit_errors()      — prediction RMSE and coefficient error of the
                               least-squares refit on the selected variables
                               vs. T, with the refit on the true support.
    plot_runtime()           — time to T, steps and time per step vs. the
                               number of dummies L (demo_ts_perf_02), with a
                               reference line proportional to p + L.
    plot_warm_start()        — fresh run to T + 1 over the warm start T -> T + 1.
    plot_memory()            — solver memory and its share of the data vs. n/p
                               (demo_ts_perf_03), with the data size.
    plot_recovery()          — exact recovery, TPR or FDR at T = 1 vs.
                               n/(s log p), one panel per solver, one curve
                               per p (demo_ts_perf_04, Part 1).
    plot_phase_transition()  — exact recovery over n/p and s/n, one heatmap
                               per solver, with the l1 phase transition and
                               the LARS s-step bound (demo_ts_perf_04, Part 2).
    plot_null_pmf()          — histogram of the included null variables at T
                               with the negative hypergeometric (NHG) PMF,
                               one panel per solver (demo_ts_perf_05).
    plot_null_qq()           — Q-Q plot of the same counts against the NHG.
    plot_selected_ecdf()     — ECDFs of the selected null variables V(v) at
                               the voting levels, with the NHG CDF.

CLI modes (typically driven from a demo's generate_plots.sh):
    python ts_plt_utils.py paths <stem>_path.csv --solver T-LARS --title "..."
    python ts_plt_utils.py compare --csvs a_path.csv b_path.csv \
        --labels 'rho = 0.1' 'rho = 1' --title "..." --stem my_stem
    python ts_plt_utils.py fdr-tpr fdr_tpr_vs_T_mc500.csv --title "..."
    python ts_plt_utils.py runtime runtime_vs_dummies_mc20.csv --title "..." --p 1000
    python ts_plt_utils.py memory memory_vs_np_mc500.csv --title "..."
    python ts_plt_utils.py recovery recovery_vs_dt_ratio_mc500.csv \
        --phase phase_transition_mc500.csv --title "..."
    python ts_plt_utils.py nhg null_counts_vs_nhg_mc500.csv --title "..."

The path mode reads <stem>_path.csv (T_stop,step,index,is_dummy,beta) and the
sibling <stem>_actions.csv (T_stop,step,index,is_dummy,added) written by
ts_perf::PathRecorder. Figures default to the simulation_results/plots/
sibling of the data folder; override with --outdir.
"""

import argparse
import math
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")  # headless / batch-safe: no display required

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# Colours and line styles of the tlars vignette path plot
COLOR_ORIGINAL = "black"
COLOR_DUMMY = "#d62728"
COLOR_STEP = "0.65"
MAX_TOP_LABELS = 40  # beyond this, the top axis labels get small and rotated


# ==============================================================================
# Shared helpers
# ==============================================================================

def default_plots_dir(csv_path: Path) -> Path:
    """simulation_results/plots/ next to simulation_results/data/, else the CSV's folder."""
    parent = csv_path.resolve().parent
    if parent.name == "data" and parent.parent.name == "simulation_results":
        return parent.parent / "plots"
    return parent


def save_figure(fig: plt.Figure, outdir: Path, stem: str, formats: list[str]) -> None:
    """Write the figure to outdir/<stem>.<ext> for each requested format."""
    outdir.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        output_path = outdir / f"{stem}.{extension}"
        fig.savefig(
            output_path,
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
            # No creation date in the PDF, so an unchanged figure stays byte-identical
            metadata={"CreationDate": None} if extension == "pdf" else None,
        )
        print(f"Wrote: {output_path}")


# ==============================================================================
# Terminated solution paths
# ==============================================================================

def read_paths(path_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read <stem>_path.csv and its sibling <stem>_actions.csv."""
    actions_csv = path_csv.with_name(path_csv.name.replace("_path.csv", "_actions.csv"))
    if actions_csv == path_csv or not actions_csv.exists():
        raise FileNotFoundError(f"no actions file next to {path_csv}")
    return pd.read_csv(path_csv), pd.read_csv(actions_csv)


def _draw_path_panel(ax: plt.Axes, path: pd.DataFrame, actions: pd.DataFrame, T_stop: int,
                     label: str | None = None) -> None:
    """One panel: coefficient paths up to the T_stop-th included dummy."""
    num_steps = int(actions["step"].max())
    steps = range(num_steps + 1)

    # Every variable that is ever nonzero, as a dense line over all steps
    wide = path.pivot_table(index="step", columns="index", values="beta", aggfunc="first")
    wide = wide.reindex(steps).fillna(0.0)
    dummy_of = path.groupby("index")["is_dummy"].first()

    dummy_steps = actions.loc[(actions["is_dummy"] == 1) & (actions["added"] == 1), "step"].tolist()

    for step in range(1, num_steps + 1):
        ax.axvline(step, color=COLOR_STEP, linestyle=(0, (5, 2, 1, 2)), linewidth=0.6, zorder=0)
    for step in dummy_steps:
        ax.axvline(step, color=COLOR_DUMMY, linewidth=1.3, zorder=1)
    ax.axhline(0.0, color=COLOR_ORIGINAL, linewidth=0.8, zorder=1)

    for index in wide.columns:
        is_dummy = bool(dummy_of[index])
        ax.plot(
            wide.index,
            wide[index],
            color=COLOR_DUMMY if is_dummy else COLOR_ORIGINAL,
            linestyle="--" if is_dummy else "-",
            linewidth=1.2,
            zorder=3 if is_dummy else 2,
        )

    # Margin on both sides, so the curves visibly start from 0 at step 0
    margin = max(0.5, 0.02 * num_steps)
    ax.set_xlim(-margin, num_steps + margin)
    ax.set_xticks(dummy_steps)
    ax.set_xticklabels([str(k) for k in range(1, len(dummy_steps) + 1)])
    ax.set_xlabel("Number of included dummies")
    ax.set_ylabel("Coefficients")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Top axis: the variable (index from the CSV) or dummy (D) entering at each step
    added = actions[actions["added"] == 1]
    top = ax.secondary_xaxis("top")
    top.set_xticks(added["step"].tolist())
    labels = ["D" if d else str(i) for i, d in zip(added["index"], added["is_dummy"])]
    if len(labels) > MAX_TOP_LABELS:
        top.set_xticklabels(labels, fontsize=6, rotation=90)
    else:
        top.set_xticklabels(labels, fontsize=8)
    top.set_xlabel("Index of the entering variable (0-based, D = dummy)", fontsize=9)

    num_dummies = len(dummy_steps)
    dummy_word = "dummy" if num_dummies == 1 else "dummies"
    prefix = f"{label}, " if label else ""
    ax.set_title(f"{prefix}$T = {T_stop}$: {num_steps} steps, {num_dummies} {dummy_word} included",
                 fontsize=11, loc="left")


def _finish_figure(fig: plt.Figure, title: str) -> None:
    """Title and legend above the panels, so the legend never covers a curve."""
    handles = [
        Line2D([], [], color=COLOR_ORIGINAL, linestyle="-", label="Original variables"),
        Line2D([], [], color=COLOR_DUMMY, linestyle="--", label="Dummies"),
    ]
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=2,
               frameon=True, facecolor="white", edgecolor="0.7")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.962))


def plot_terminated_paths(path: pd.DataFrame, actions: pd.DataFrame, solver: str,
                          title: str | None = None) -> plt.Figure:
    """Coefficient paths vs. number of included dummies, one panel per T_stop."""
    T_values = sorted(actions["T_stop"].unique())
    fig, axes = plt.subplots(len(T_values), 1, figsize=(10, 4.2 * len(T_values)), squeeze=False)

    for ax, T_stop in zip(axes[:, 0], T_values):
        _draw_path_panel(ax, path[path["T_stop"] == T_stop], actions[actions["T_stop"] == T_stop],
                         int(T_stop))

    _finish_figure(fig, title or f"{solver}: terminated solution paths")
    return fig


def plot_path_comparison(variants: list[tuple[str, pd.DataFrame, pd.DataFrame]],
                         title: str) -> plt.Figure:
    """One panel per (label, path, actions) variant, each at its largest T_stop."""
    fig, axes = plt.subplots(len(variants), 1, figsize=(10, 4.2 * len(variants)), squeeze=False)
    for ax, (label, path, actions) in zip(axes[:, 0], variants):
        T_stop = int(actions["T_stop"].max())
        _draw_path_panel(ax, path[path["T_stop"] == T_stop], actions[actions["T_stop"] == T_stop],
                         T_stop, label)
    _finish_figure(fig, title)
    return fig


# ==============================================================================
# FDR and TPR vs. T (demo_ts_perf_01)
# ==============================================================================

FAMILIES = ["LARS-based", "OMP-based", "AFS-based"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def display_name(name: str) -> str:
    """Solver label for legends, with its parameter as math: "T-AFS (rho 0.1)" -> "T-AFS ($\\rho$ = 0.1)"."""
    return name.replace("(rho ", "($\\rho$ = ").replace("(lambda2 ", "($\\lambda_2$ = ")


def _solver_styles(df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Colour (hex) and marker per solver, cycling within each family in file order."""
    cmap = plt.get_cmap("tab10")
    styles: dict[str, tuple[str, str]] = {}
    for family in FAMILIES:
        names = list(dict.fromkeys(df.loc[df["family"] == family, "solver"]))
        for k, name in enumerate(names):
            styles[name] = (to_hex(cmap(k % 10)), MARKERS[k % len(MARKERS)])
    return styles


def _plot_series(ax: plt.Axes, sub: pd.DataFrame, x: str, y: str, color: str,
                 marker: str, label: str) -> None:
    """Line with filled markers where every run reached T, open markers elsewhere."""
    ax.plot(100 * sub[x] if x != "T" else sub[x], 100 * sub[y], color=color, linewidth=1.6, label=label)
    for reached, face in ((True, color), (False, "white")):
        part = sub[(sub["reached"] >= 0.99) == reached]
        ax.scatter(100 * part[x] if x != "T" else part[x], 100 * part[y], marker=marker, s=34,
                   facecolors=[face] * len(part), edgecolors=[color] * len(part), linewidths=1.2, zorder=3)


def _marker_legend(ax: plt.Axes, family_df: pd.DataFrame, styles: dict[str, tuple[str, str]],
                   loc: Literal["best", "upper left", "upper right"] = "best",
                   extra: list[Line2D] | None = None) -> None:
    """Legend with the line and the filled marker of every solver in the panel."""
    handles = [Line2D([], [], color=styles[name][0], marker=styles[name][1], markersize=6, linewidth=1.6,
                      label=display_name(name)) for name in dict.fromkeys(family_df["solver"])]
    ax.legend(handles=handles + (extra or []), loc=loc, fontsize=8, frameon=True, facecolor="white",
              edgecolor="0.7")


def _open_marker_note(fig: plt.Figure) -> None:
    """Explain the open markers below the panels and lay out the figure around the note."""
    fig.text(0.5, 0.005, "Open markers: some runs ended before $T$ dummies entered (active set at its size limit).",
             ha="center", fontsize=9, color="0.3")
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 1.0))


def plot_fdr_tpr_vs_T(df: pd.DataFrame, title: str) -> plt.Figure:
    """FDR (top row) and TPR (bottom row) in % vs. T, one column per solver family."""
    styles = _solver_styles(df)
    T_values = sorted(df["T"].unique())
    fig, axes = plt.subplots(2, len(FAMILIES), figsize=(15, 9), sharex=True, sharey=True, squeeze=False)
    for col, family in enumerate(FAMILIES):
        fam = df[df["family"] == family]
        for name in dict.fromkeys(fam["solver"]):
            sub = fam[fam["solver"] == name].sort_values("T")
            color, marker = styles[name]
            for row, metric in enumerate(("FDR", "TPR")):
                _plot_series(axes[row, col], sub, "T", metric, color, marker, name)
        axes[0, col].set_title(family, fontsize=12)
        for row, metric in enumerate(("FDR", "TPR")):
            ax = axes[row, col]
            ax.set_xscale("log")
            ax.set_xticks(T_values)
            ax.set_xticklabels([str(int(v)) for v in T_values])
            ax.minorticks_off()
            ax.set_ylim(-3, 103)
            ax.grid(linestyle="--", alpha=0.4)
            if col == 0:
                ax.set_ylabel(f"{metric} (%)")
            if row == 1:
                ax.set_xlabel("$T$ (number of included dummies)")
    for col, family in enumerate(FAMILIES):
        _marker_legend(axes[1, col], df[df["family"] == family], styles)
    fig.suptitle(f"{title}: FDR and TPR vs. $T$", fontsize=13)
    _open_marker_note(fig)
    return fig


def plot_tpr_vs_fdr(df: pd.DataFrame, title: str) -> plt.Figure:
    """TPR vs. FDR in % along the T grid, one panel per solver family."""
    styles = _solver_styles(df)
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(15, 5.2), sharex=True, sharey=True, squeeze=False)
    for col, family in enumerate(FAMILIES):
        ax = axes[0, col]
        fam = df[df["family"] == family]
        for name in dict.fromkeys(fam["solver"]):
            sub = fam[fam["solver"] == name].sort_values("T")
            color, marker = styles[name]
            _plot_series(ax, sub, "FDR", "TPR", color, marker, name)
        ax.set_title(family, fontsize=12)
        ax.set_xlim(-3, 103)
        ax.set_ylim(-3, 103)
        ax.set_xlabel("FDR (%)")
        if col == 0:
            ax.set_ylabel("TPR (%)")
        ax.grid(linestyle="--", alpha=0.4)
    for col, family in enumerate(FAMILIES):
        _marker_legend(axes[0, col], df[df["family"] == family], styles)
    fig.suptitle(f"{title}: TPR vs. FDR along the $T$ grid", fontsize=13)
    _open_marker_note(fig)
    return fig


def plot_path_stats(df: pd.DataFrame, title: str, rank: int | None = None, p: int | None = None,
                    L: int | None = None, s: int | None = None) -> plt.Figure:
    """Selected variables, steps and cumulative time (ms) vs. T, one column per solver family.

    With rank, p, L and s given, the selected-variables row shows s + (p - s) / (L + 1) * T:
    the true variables plus the expected number of null variables that enter before the
    T-th dummy when nulls and dummies are exchangeable (negative hypergeometric mean). The
    steps row shows the step limit 8 * min(p + L, rank) of the capped solvers, with rank the
    active-set cap (n - 1 for centered data).
    """
    rows = [("avg_selected", "Selected variables"), ("avg_steps", "Steps"),
            ("avg_ms", "Time from construction (ms)")]
    styles = _solver_styles(df)
    T_values = sorted(df["T"].unique())
    fig, axes = plt.subplots(len(rows), len(FAMILIES), figsize=(15, 12), sharex=True, sharey="row",
                             squeeze=False)
    for col, family in enumerate(FAMILIES):
        fam = df[df["family"] == family]
        for name in dict.fromkeys(fam["solver"]):
            sub = fam[fam["solver"] == name].sort_values("T")
            color, marker = styles[name]
            for row, (metric, _) in enumerate(rows):
                ax = axes[row, col]
                ax.plot(sub["T"], sub[metric], color=color, linewidth=1.6, label=name)
                for reached, face in ((True, color), (False, "white")):
                    part = sub[(sub["reached"] >= 0.99) == reached]
                    ax.scatter(part["T"], part[metric], marker=marker, s=34, facecolors=[face] * len(part),
                               edgecolors=[color] * len(part), linewidths=1.2, zorder=3)
        axes[0, col].set_title(family, fontsize=12)

    expected: list[Line2D] = []
    limit: list[Line2D] = []
    if rank is not None and p is not None and L is not None and s is not None:
        grid = pd.Series([float(v) for v in T_values])
        rate = (p - s) / (L + 1)
        expected = [Line2D([], [], color="0.35", linestyle="--", linewidth=1.2,
                           label=f"expected: {s} true + {rate:.2f} $T$ null")]
        limit = [Line2D([], [], color="0.35", linestyle=":", linewidth=1.2,
                        label=f"step limit {8 * min(p + L, rank)}")]
        for col in range(len(FAMILIES)):
            axes[0, col].plot(grid, s + rate * grid, color="0.35", linestyle="--", linewidth=1.2, zorder=1)
            axes[1, col].axhline(8 * min(p + L, rank), color="0.35", linestyle=":", linewidth=1.2, zorder=1)
    for col in range(len(FAMILIES)):
        for row, (_, label) in enumerate(rows):
            ax = axes[row, col]
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(T_values)
            ax.set_xticklabels([str(int(v)) for v in T_values])
            ax.minorticks_off()
            ax.grid(linestyle="--", alpha=0.4, which="both")
            if col == 0:
                ax.set_ylabel(label)
            if row == len(rows) - 1:
                ax.set_xlabel("$T$ (number of included dummies)")
    for col, family in enumerate(FAMILIES):
        _marker_legend(axes[0, col], df[df["family"] == family], styles, loc="upper left", extra=expected)
        if limit:
            axes[1, col].legend(handles=limit, loc="upper left", fontsize=8, frameon=True, facecolor="white",
                                edgecolor="0.7")
    fig.suptitle(f"{title}: selected variables, steps and time vs. $T$", fontsize=13)
    _open_marker_note(fig)
    return fig


def plot_refit_errors(df: pd.DataFrame, title: str) -> plt.Figure:
    """Prediction RMSE and coefficient error of the post-selection refit vs. T, one column per family."""
    rows = [("rmse", "oracle_rmse", "Prediction RMSE (test set)"),
            ("coef_err", "oracle_coef_err", "Coefficient error $\\|\\hat{\\beta} - \\beta\\|_2$")]
    styles = _solver_styles(df)
    T_values = sorted(df["T"].unique())
    fig, axes = plt.subplots(len(rows), len(FAMILIES), figsize=(15, 8.5), sharex=True, sharey="row",
                             squeeze=False)
    oracle = [Line2D([], [], color="0.35", linestyle="--", linewidth=1.2,
                     label="least squares on the true support")]
    for col, family in enumerate(FAMILIES):
        fam = df[df["family"] == family]
        for name in dict.fromkeys(fam["solver"]):
            sub = fam[fam["solver"] == name].sort_values("T")
            color, marker = styles[name]
            for row, (metric, _, _) in enumerate(rows):
                ax = axes[row, col]
                ax.plot(sub["T"], sub[metric], color=color, linewidth=1.6)
                for reached, face in ((True, color), (False, "white")):
                    part = sub[(sub["reached"] >= 0.99) == reached]
                    ax.scatter(part["T"], part[metric], marker=marker, s=34, facecolors=[face] * len(part),
                               edgecolors=[color] * len(part), linewidths=1.2, zorder=3)
        axes[0, col].set_title(family, fontsize=12)
        for row, (_, ref, label) in enumerate(rows):
            ax = axes[row, col]
            ax.axhline(float(df[ref].iloc[0]), color="0.35", linestyle="--", linewidth=1.2, zorder=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(T_values)
            ax.set_xticklabels([str(int(v)) for v in T_values])
            ax.minorticks_off()
            ax.grid(linestyle="--", alpha=0.4, which="both")
            if col == 0:
                ax.set_ylabel(label)
            if row == len(rows) - 1:
                ax.set_xlabel("$T$ (number of included dummies)")
        _marker_legend(axes[0, col], fam, styles, loc="best", extra=oracle)
    fig.suptitle(f"{title}: least-squares refit on the selected variables vs. $T$", fontsize=13)
    _open_marker_note(fig)
    return fig


# ==============================================================================
# Runtime vs. number of dummies (demo_ts_perf_02)
# ==============================================================================

def _plain_log_labels(ax: plt.Axes) -> None:
    """Plain tick labels on a log y-axis (20, 300, 1,000) instead of 2 x 10^1.

    Minor ticks are labelled only at 2, 3 and 5 times a power of ten, so the axis stays readable.
    """
    def plain(v: float) -> str:
        return f"{v:,.0f}" if v >= 1 else f"{v:g}"

    def minor(v: float, _: object) -> str:
        mantissa = round(v / 10 ** math.floor(math.log10(v))) if v > 0 else 0
        return plain(v) if mantissa in (2, 3, 5) else ""

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: plain(v)))
    ax.yaxis.set_minor_formatter(FuncFormatter(minor))


def _L_axis(ax: plt.Axes, factors: list[int]) -> None:
    """Log x-axis over the dummy factors, labelled p, 2p, 5p, ..."""
    ax.set_xscale("log")
    ax.set_xticks(factors)
    ax.set_xticklabels(["$p$" if f == 1 else f"${f}p$" for f in factors])
    ax.minorticks_off()


def plot_runtime(df: pd.DataFrame, title: str, p: int) -> plt.Figure:
    """Time to T, steps to T and time per step vs. L/p, one column per solver family."""
    rows = [("ms", "Time to $T$ (ms)"), ("steps", "Steps to $T$"), ("us_per_step", "Time per step ($\\mu$s)")]
    styles = _solver_styles(df)
    factors = sorted(int(f) for f in df["L_factor"].unique())
    fig, axes = plt.subplots(len(rows), len(FAMILIES), figsize=(15, 12), sharex=True, sharey="row", squeeze=False)
    scaling = [Line2D([], [], color="0.35", linestyle="--", linewidth=1.2, label="$\\propto p + L$")]
    for col, family in enumerate(FAMILIES):
        fam = df[df["family"] == family]
        for name in dict.fromkeys(fam["solver"]):
            sub = fam[fam["solver"] == name].sort_values("L_factor")
            color, marker = styles[name]
            for row, (metric, _) in enumerate(rows):
                axes[row, col].plot(sub["L_factor"], sub[metric], color=color, marker=marker, markersize=6,
                                    linewidth=1.6)
        # Reference: cost per step proportional to the number of columns p + L, anchored at L = p
        start = float(fam.loc[fam["L_factor"] == factors[0], "us_per_step"].median())
        grid = pd.Series([float(f) for f in factors])
        axes[2, col].plot(grid, start * (p + grid * p) / (2 * p), color="0.35", linestyle="--", linewidth=1.2,
                          zorder=1)
        axes[0, col].set_title(family, fontsize=12)
        for row, (_, label) in enumerate(rows):
            ax = axes[row, col]
            _L_axis(ax, factors)
            ax.set_yscale("log")
            _plain_log_labels(ax)
            ax.grid(linestyle="--", alpha=0.4, which="both")
            if col == 0:
                ax.set_ylabel(label)
            if row == len(rows) - 1:
                ax.set_xlabel("$L$ (number of dummies)")
        _marker_legend(axes[0, col], fam, styles, loc="upper left")
        axes[2, col].legend(handles=scaling, loc="upper left", fontsize=8, frameon=True, facecolor="white",
                            edgecolor="0.7")
    fig.suptitle(f"{title}: runtime vs. the number of dummies", fontsize=13)
    fig.tight_layout()
    return fig


def plot_warm_start(df: pd.DataFrame, title: str) -> plt.Figure:
    """Fresh run to T + 1 over the warm start T -> T + 1 vs. L/p, one panel per solver family."""
    styles = _solver_styles(df)
    factors = sorted(int(f) for f in df["L_factor"].unique())
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(15, 5), sharex=True, sharey=True, squeeze=False)
    for col, family in enumerate(FAMILIES):
        ax = axes[0, col]
        fam = df[df["family"] == family]
        for name in dict.fromkeys(fam["solver"]):
            sub = fam[fam["solver"] == name].sort_values("L_factor")
            color, marker = styles[name]
            ax.plot(sub["L_factor"], sub["fresh_over_warm"], color=color, marker=marker, markersize=6,
                    linewidth=1.6)
        ax.set_title(family, fontsize=12)
        _L_axis(ax, factors)
        ax.set_yscale("log")
        _plain_log_labels(ax)
        ax.grid(linestyle="--", alpha=0.4, which="both")
        ax.set_xlabel("$L$ (number of dummies)")
        if col == 0:
            ax.set_ylabel("Fresh run to $T + 1$ / warm start $T \\to T + 1$")
        _marker_legend(ax, fam, styles)
    fig.suptitle(f"{title}: gain of the warm start", fontsize=13)
    fig.tight_layout()
    return fig


# ==============================================================================
# Memory vs. n/p (demo_ts_perf_03)
# ==============================================================================

def plot_memory(df: pd.DataFrame, title: str) -> plt.Figure:
    """Held solver heap (MiB, with the data size) and its share of the data (%) vs. n/p."""
    styles = _solver_styles(df)
    ratios = sorted(float(r) for r in df["n_over_p"].unique())
    n_values = sorted(int(n) for n in df["n"].unique())
    data = df.drop_duplicates("n").sort_values("n")
    fig, axes = plt.subplots(2, len(FAMILIES), figsize=(15, 8.5), sharex=True, sharey="row", squeeze=False)
    reference = [Line2D([], [], color="0.35", linestyle="--", linewidth=1.2, label="data $X$, $D$ and $y$")]
    for col, family in enumerate(FAMILIES):
        fam = df[df["family"] == family]
        for name in dict.fromkeys(fam["solver"]):
            sub = fam[fam["solver"] == name].sort_values("n")
            color, marker = styles[name]
            axes[0, col].plot(sub["n_over_p"], sub["solver_mib"], color=color, marker=marker, markersize=6,
                              linewidth=1.6)
            axes[1, col].plot(sub["n_over_p"], 100 * sub["solver_mib"] / sub["data_mib"], color=color,
                              marker=marker, markersize=6, linewidth=1.6)
        axes[0, col].plot(data["n_over_p"], data["data_mib"], color="0.35", linestyle="--", linewidth=1.2, zorder=1)
        axes[0, col].set_title(family, fontsize=12)
        for row, label in enumerate(("Held solver heap (MiB)", "Held solver heap / data (%)")):
            ax = axes[row, col]
            ax.set_xscale("log")
            ax.set_xticks(ratios)
            ax.set_xticklabels([f"{r:g}" for r in ratios])
            ax.minorticks_off()
            ax.set_yscale("log")
            _plain_log_labels(ax)
            ax.grid(linestyle="--", alpha=0.4, which="both")
            if col == 0:
                ax.set_ylabel(label)
            if row == 1:
                ax.set_xlabel("$n/p$")
        top = axes[0, col].secondary_xaxis("top")
        top.set_xticks(ratios)
        top.set_xticklabels([f"{n:,}" for n in n_values], fontsize=8)
        top.minorticks_off()
        top.set_xlabel("$n$", fontsize=9)
        _marker_legend(axes[0, col], fam, styles, loc="upper left", extra=reference)
    fig.suptitle(f"{title}: memory vs. $n/p$", fontsize=13)
    fig.tight_layout()
    return fig


# ==============================================================================
# Recovery vs. n/(s log p) and the phase transition (demo_ts_perf_04)
# ==============================================================================

RECOVERY_METRICS = {
    "exact": ("Exact recovery (%)", "exact recovery by the first $s$ entries"),
    "TPR": ("TPR at $T = 1$ (%)", "TPR at the first dummy"),
    "FDR": ("FDR at $T = 1$ (%)", "FDR at the first dummy"),
}


def plot_recovery(df: pd.DataFrame, metric: str, title: str) -> plt.Figure:
    """One metric in % vs. n/(s log p), one panel per solver (rows = families), one curve per p."""
    label, subject = RECOVERY_METRICS[metric]
    ratios = sorted(float(r) for r in df["ratio"].unique())
    p_values = sorted(int(p) for p in df["p"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {p: to_hex(cmap(0.9 * k / max(1, len(p_values) - 1))) for k, p in enumerate(p_values)}
    per_family = [list(dict.fromkeys(df.loc[df["family"] == f, "solver"])) for f in FAMILIES]
    cols = max(len(names) for names in per_family)
    fig, axes = plt.subplots(len(FAMILIES), cols, figsize=(3.1 * cols, 3.0 * len(FAMILIES)), sharex=True,
                             sharey=True, squeeze=False)
    for row, names in enumerate(per_family):
        for col in range(cols):
            ax = axes[row, col]
            if col >= len(names):
                ax.set_visible(False)
                continue
            sub = df[df["solver"] == names[col]]
            for p in p_values:
                curve = sub[sub["p"] == p].sort_values("ratio")
                color = colors[p]
                ax.plot(curve["ratio"], 100 * curve[metric], color=color, linewidth=1.5)
                for reached, face in ((True, color), (False, "white")):
                    part = curve[(curve["reached"] >= 0.99) == reached]
                    ax.scatter(part["ratio"], 100 * part[metric], s=22, facecolors=[face] * len(part),
                               edgecolors=[color] * len(part), linewidths=1.1, zorder=3)
            ax.set_title(display_name(names[col]), fontsize=10)
            ax.set_xscale("log")
            ax.set_xticks(ratios)
            ax.set_xticklabels([f"{r:g}" for r in ratios], fontsize=8)
            ax.minorticks_off()
            ax.set_ylim(-3, 103)
            ax.grid(linestyle="--", alpha=0.4)
            if col == 0:
                ax.set_ylabel(f"{FAMILIES[row]}\n{label}", fontsize=9)
            if row == len(FAMILIES) - 1 or col >= len(per_family[row + 1]):
                ax.set_xlabel("$n/(s \\log p)$", fontsize=9)
                ax.xaxis.set_tick_params(labelbottom=True)
    handles = [Line2D([], [], color=colors[p], marker="o", markersize=5, linewidth=1.5, label=f"$p = {p:,}$".replace(
        ",", "{,}")) for p in p_values]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.98, 0.06), fontsize=10, frameon=True,
               facecolor="white", edgecolor="0.7", title="$L = p$, $s = 10$")
    fig.suptitle(f"{title}: {subject} vs. $n/(s \\log p)$", fontsize=13)
    if (df["reached"] < 0.99).any():
        _open_marker_note(fig)
    else:
        fig.tight_layout()
    return fig


def _gauss_tail(t: float) -> float:
    return 0.5 * math.erfc(t / math.sqrt(2.0))


def _l1_statistical_dimension(eps: float) -> float:
    """Statistical dimension over N of the l1 descent cone at an eps N-sparse vector (Amelunxen et al., 2014)."""
    def bound(t: float) -> float:
        density = math.exp(-t * t / 2.0) / math.sqrt(2.0 * math.pi)
        return eps * (1 + t * t) + 2 * (1 - eps) * ((1 + t * t) * _gauss_tail(t) - t * density)

    lo, hi = 0.0, 12.0
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(100):  # the bound is convex in t
        a, b = hi - golden * (hi - lo), lo + golden * (hi - lo)
        if bound(a) < bound(b):
            hi = b
        else:
            lo = a
    return bound((lo + hi) / 2.0)


def l1_phase_transition(columns_over_p: float) -> tuple[list[float], list[float]]:
    """Noiseless l1 phase transition in the coordinates n/p and s/n for a design with columns_over_p * p columns."""
    xs, ys = [], []
    for k in range(1, 400):
        eps = 0.5 * k / 400
        delta = _l1_statistical_dimension(eps)  # n / N at the transition
        if columns_over_p * delta <= 1.0:
            xs.append(columns_over_p * delta)
            ys.append(eps / delta)
    return xs, ys


def plot_phase_transition(df: pd.DataFrame, title: str) -> plt.Figure:
    """Exact recovery (%) over n/p and s/n, one heatmap per solver, with the l1 and k-step references."""
    solvers = list(dict.fromkeys(df["solver"]))
    x_values = sorted(float(v) for v in df["n_over_p"].unique())
    y_values = sorted(float(v) for v in df["s_over_n"].unique())
    p = int(df["p"].iloc[0])
    columns = 2 * p  # L = p
    step_x, step_y = x_values[1] - x_values[0], y_values[1] - y_values[0]
    edges_x = [x_values[0] - step_x / 2] + [v + step_x / 2 for v in x_values]
    edges_y = [y_values[0] - step_y / 2] + [v + step_y / 2 for v in y_values]
    l1_x, l1_y = l1_phase_transition(columns / p)
    kstep = 1.0 / (2.0 * math.log(columns))
    fig, axes = plt.subplots(1, len(solvers), figsize=(5.2 * len(solvers), 5.0), sharex=True, sharey=True,
                             squeeze=False)
    mesh = None
    for col, name in enumerate(solvers):
        ax = axes[0, col]
        grid = df[df["solver"] == name].pivot(index="s_over_n", columns="n_over_p", values="exact").sort_index()
        grid = grid[sorted(grid.columns)]
        mesh = ax.pcolormesh(edges_x, edges_y, 100 * grid.to_numpy(), cmap="viridis", vmin=0, vmax=100)
        ax.contour(x_values, y_values, 100 * grid.to_numpy(), levels=[50], colors="white", linewidths=1.4)
        ax.plot(l1_x, l1_y, color="#d62728", linestyle="--", linewidth=1.6)
        ax.axhline(kstep, color="#ff7f0e", linestyle=":", linewidth=1.8)
        ax.set_title(display_name(name), fontsize=12)
        ax.set_xlim(edges_x[0], edges_x[-1])
        ax.set_ylim(edges_y[0], edges_y[-1])
        ax.set_xlabel("$n/p$")
        if col == 0:
            ax.set_ylabel("$s/n$")
    handles = [Line2D([], [], color="white", linewidth=1.4, label="50% contour"),
               Line2D([], [], color="#d62728", linestyle="--", linewidth=1.6,
                      label="$\\ell_1$ phase transition, $N = p + L$ (noiseless)"),
               Line2D([], [], color="#ff7f0e", linestyle=":", linewidth=1.8,
                      label="$s/n = 1/(2 \\ln N)$ (LARS $s$-step bound)")]
    legend = axes[0, 0].legend(handles=handles, loc="upper left", fontsize=8, frameon=True, facecolor="0.3",
                               edgecolor="0.7", labelcolor="white")
    legend.get_frame().set_alpha(0.85)
    fig.suptitle(f"{title}: exact recovery by the first $s$ entries", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 0.93, 1.0))
    if mesh is not None:
        bar = fig.colorbar(mesh, ax=axes[0, :].tolist(), fraction=0.025, pad=0.02)
        bar.set_label("Exact recovery (%)")
    return fig


# ==============================================================================
# Null counts vs. the negative hypergeometric distribution (demo_ts_perf_05)
# ==============================================================================

SELECTED_LEVELS = [("selected_v1", "$v \\to 1$", "#1b9e77"), ("selected_v0.75", "$v = 0.75$", "#ff7f0e"),
                   ("selected_v0.5", "$v = 0.5$", "#7570b3")]


def nhg_pmf(p0: int, L: int, T: int) -> list[float]:
    """PMF of the number of nulls drawn before the T-th dummy, NHG(p0 + L, p0, T), for 0, ..., p0."""
    total = math.comb(p0 + L, p0)
    return [math.comb(v + T - 1, v) * math.comb(p0 + L - v - T, p0 - v) / total for v in range(p0 + 1)]


def _counts_setup(df: pd.DataFrame) -> tuple[int, int, int, int]:
    """p0, L, T and K of the demo, constant over the CSV."""
    return tuple(int(df[c].iloc[0]) for c in ("p0", "L", "T", "K"))


def _frequencies(df: pd.DataFrame, solver: str, statistic: str, size: int) -> list[float]:
    """Frequencies of the counts 0, ..., size - 1 of one solver and statistic."""
    sub = df[(df["solver"] == solver) & (df["statistic"] == statistic)]
    freq = [0.0] * size
    for count, frequency in zip(sub["count"], sub["frequency"]):
        freq[int(count)] += float(frequency)
    return freq


def _cumulative(weights: list[float]) -> list[float]:
    total = sum(weights)
    out, running = [], 0.0
    for w in weights:
        running += w
        out.append(running / total if total > 0 else float("nan"))
    return out


def _quantile(cdf: list[float], prob: float) -> int:
    """Smallest count whose CDF reaches prob."""
    return next((v for v, c in enumerate(cdf) if c >= prob - 1e-12), len(cdf) - 1)


def _solver_grid(df: pd.DataFrame) -> tuple[plt.Figure, list[list[str]], object]:
    """One panel per solver, one row per family; unused panels hidden."""
    per_family = [list(dict.fromkeys(df.loc[df["family"] == f, "solver"])) for f in FAMILIES]
    cols = max(len(names) for names in per_family)
    fig, axes = plt.subplots(len(FAMILIES), cols, figsize=(3.1 * cols, 3.0 * len(FAMILIES)), sharex=True,
                             sharey=True, squeeze=False)
    for row, names in enumerate(per_family):
        for col in range(len(names), cols):
            axes[row, col].set_visible(False)
    return fig, per_family, axes


def _grid_labels(axes, per_family: list[list[str]], row: int, col: int, xlabel: str, ylabel: str) -> None:
    """Panel title, the y label in the first column and the x label at the bottom of every column."""
    ax = axes[row, col]
    ax.set_title(display_name(per_family[row][col]), fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    if col == 0:
        ax.set_ylabel(f"{FAMILIES[row]}\n{ylabel}", fontsize=9)
    if row == len(FAMILIES) - 1 or col >= len(per_family[row + 1]):
        ax.set_xlabel(xlabel, fontsize=9)
        ax.xaxis.set_tick_params(labelbottom=True)


def _grid_legend(fig: plt.Figure, handles: list, title: str | None = None) -> None:
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.98, 0.06), fontsize=10, frameon=True,
               facecolor="white", edgecolor="0.7", title=title)


def plot_null_pmf(df: pd.DataFrame, title: str) -> plt.Figure:
    """Histogram of the included null variables at T per solver, with the NHG PMF and both means."""
    p0, L, T, _ = _counts_setup(df)
    pmf = nhg_pmf(p0, L, T)
    nhg_mean = T * p0 / (L + 1)
    x_max = _quantile(_cumulative(pmf), 0.9999) + 2
    fig, per_family, axes = _solver_grid(df)
    for row, names in enumerate(per_family):
        for col, name in enumerate(names):
            ax = axes[row, col]
            freq = _frequencies(df, name, "included", p0 + 1)
            total = sum(freq)
            density = [f / total for f in freq]
            mean = sum(v * d for v, d in enumerate(density))
            ax.bar(range(x_max + 1), density[:x_max + 1], width=1.0, color="#fde0b8", edgecolor="0.25",
                   linewidth=0.4)
            ax.scatter(range(x_max + 1), pmf[:x_max + 1], s=6, color="black", zorder=3)
            ax.axvline(nhg_mean, color="black", linewidth=1.4)
            ax.axvline(mean, color="#d62728", linestyle="--", linewidth=1.4)
            ax.set_xlim(-0.5, x_max + 0.5)
            _grid_labels(axes, per_family, row, col, "Included null variables", "Density")
    handles = [Line2D([], [], color="black", marker="o", markersize=3, linestyle="", label="NHG PMF"),
               Line2D([], [], color="black", linewidth=1.4, label=f"NHG mean $T p_0/(L+1) = {nhg_mean:.2f}$"),
               Line2D([], [], color="#d62728", linestyle="--", linewidth=1.4, label="Mean of the histogram")]
    _grid_legend(fig, handles)
    fig.suptitle(f"{title}: included null variables at $T = {T}$ vs. NHG($p_0 + L$, $p_0$, $T$)", fontsize=13)
    fig.tight_layout()
    return fig


def plot_null_qq(df: pd.DataFrame, title: str) -> plt.Figure:
    """Q-Q plot of the included null variables at T against the NHG, one panel per solver."""
    p0, L, T, _ = _counts_setup(df)
    nhg_cdf = _cumulative(nhg_pmf(p0, L, T))
    probs = [(i - 0.5) / 200 for i in range(1, 201)]
    theory = [_quantile(nhg_cdf, q) for q in probs]
    fig, per_family, axes = _solver_grid(df)
    upper = 0
    for row, names in enumerate(per_family):
        for col, name in enumerate(names):
            ax = axes[row, col]
            cdf = _cumulative(_frequencies(df, name, "included", p0 + 1))
            sample = [_quantile(cdf, q) for q in probs]
            upper = max(upper, max(sample), max(theory))
            ax.scatter(theory, sample, s=12, facecolors="white", edgecolors="black", linewidths=0.7, zorder=3)
            _grid_labels(axes, per_family, row, col, "NHG quantiles", "Sample quantiles")
    for row, names in enumerate(per_family):
        for col in range(len(names)):
            axes[row, col].plot([0, upper + 2], [0, upper + 2], color="0.4", linewidth=1.0)
    axes[0, 0].set_xlim(0, upper + 2)
    axes[0, 0].set_ylim(0, upper + 2)
    fig.suptitle(f"{title}: Q-Q plot of the included null variables at $T = {T}$", fontsize=13)
    fig.tight_layout()
    return fig


def plot_selected_ecdf(df: pd.DataFrame, title: str) -> plt.Figure:
    """ECDFs of the selected null variables V(v) at the voting levels, with the NHG CDF, one panel per solver."""
    p0, L, T, K = _counts_setup(df)
    nhg_cdf = _cumulative(nhg_pmf(p0, L, T))
    x_max = _quantile(nhg_cdf, 0.9999) + 2
    xs = list(range(x_max + 1))
    fig, per_family, axes = _solver_grid(df)
    for row, names in enumerate(per_family):
        for col, name in enumerate(names):
            ax = axes[row, col]
            for statistic, _, color in SELECTED_LEVELS:
                cdf = _cumulative(_frequencies(df, name, statistic, p0 + 1))
                ax.step(xs, cdf[:x_max + 1], where="post", color=color, linewidth=1.3)
            ax.step(xs, nhg_cdf[:x_max + 1], where="post", color="black", linewidth=1.3)
            ax.set_xlim(0, x_max)
            ax.set_ylim(-0.02, 1.02)
            _grid_labels(axes, per_family, row, col, "Selected null variables", "Empirical CDF")
    handles = [Line2D([], [], color=color, linewidth=1.3, label=label) for _, label, color in SELECTED_LEVELS]
    handles.append(Line2D([], [], color="black", linewidth=1.3, label="NHG"))
    _grid_legend(fig, handles, title=f"$K = {K}$, $T = {T}$")
    fig.suptitle(f"{title}: selected null variables $V_{{T,L}}(v)$ vs. NHG", fontsize=13)
    fig.tight_layout()
    return fig


# ==============================================================================
# CLI
# ==============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    paths = sub.add_parser("paths", help="terminated solution paths of one solver")
    paths.add_argument("path_csv", type=Path, help="<stem>_path.csv written by PathRecorder")
    paths.add_argument("--solver", required=True, help="solver name for the default title")
    paths.add_argument("--title", default=None, help="figure title")
    paths.add_argument("--stem", default=None, help="output stem (default: CSV stem without _path)")
    paths.add_argument("--outdir", type=Path, default=None, help="output folder")
    paths.add_argument("--formats", nargs="+", default=["png", "pdf"], help="output formats")

    compare = sub.add_parser("compare", help="paths of several variants, one panel each")
    compare.add_argument("--csvs", type=Path, nargs="+", required=True, help="<stem>_path.csv files")
    compare.add_argument("--labels", nargs="+", required=True, help="one panel label per CSV")
    compare.add_argument("--title", required=True, help="figure title")
    compare.add_argument("--stem", required=True, help="output stem")
    compare.add_argument("--outdir", type=Path, default=None, help="output folder")
    compare.add_argument("--formats", nargs="+", default=["png", "pdf"], help="output formats")
    fdr = sub.add_parser("fdr-tpr", help="FDR and TPR vs. T of all solvers (demo_ts_perf_01)")
    fdr.add_argument("csv", type=Path, help="fdr_tpr_vs_T_mc<N>.csv")
    fdr.add_argument("--title", required=True, help="base title; each figure adds its subject")
    fdr.add_argument("--outdir", type=Path, default=None, help="output folder")
    fdr.add_argument("--formats", nargs="+", default=["png", "pdf"], help="output formats")
    fdr.add_argument("--rank", type=int, default=None, help="active-set cap, n - 1 for centered data (step limit)")
    fdr.add_argument("--p", type=int, default=None, help="original variables (reference lines)")
    fdr.add_argument("--L", type=int, default=None, help="dummies (reference lines)")
    fdr.add_argument("--s", type=int, default=None, help="true variables (reference lines)")

    run = sub.add_parser("runtime", help="runtime vs. number of dummies (demo_ts_perf_02)")
    run.add_argument("csv", type=Path, help="runtime_vs_dummies_mc<N>.csv")
    run.add_argument("--title", required=True, help="base title; each figure adds its subject")
    run.add_argument("--p", type=int, required=True, help="original variables (x-axis in multiples of p)")
    run.add_argument("--outdir", type=Path, default=None, help="output folder")
    run.add_argument("--formats", nargs="+", default=["png", "pdf"], help="output formats")

    mem = sub.add_parser("memory", help="solver memory vs. n/p (demo_ts_perf_03)")
    mem.add_argument("csv", type=Path, help="memory_vs_np_mc<N>.csv")
    mem.add_argument("--title", required=True, help="base title; the figure adds its subject")
    mem.add_argument("--outdir", type=Path, default=None, help="output folder")
    mem.add_argument("--formats", nargs="+", default=["png", "pdf"], help="output formats")

    rec = sub.add_parser("recovery", help="recovery vs. n/(s log p) and the phase transition (demo_ts_perf_04)")
    rec.add_argument("csv", type=Path, nargs="?", default=None, help="recovery_vs_dt_ratio_mc<N>.csv (Part 1)")
    rec.add_argument("--phase", type=Path, default=None, help="phase_transition_mc<N>.csv (Part 2)")
    rec.add_argument("--title", required=True, help="base title; each figure adds its subject")
    rec.add_argument("--outdir", type=Path, default=None, help="output folder")
    rec.add_argument("--formats", nargs="+", default=["png", "pdf"], help="output formats")

    nhg = sub.add_parser("nhg", help="null counts vs. the negative hypergeometric distribution (demo_ts_perf_05)")
    nhg.add_argument("csv", type=Path, help="null_counts_vs_nhg_mc<N>.csv")
    nhg.add_argument("--title", required=True, help="base title; each figure adds its subject")
    nhg.add_argument("--outdir", type=Path, default=None, help="output folder")
    nhg.add_argument("--formats", nargs="+", default=["png", "pdf"], help="output formats")

    args = parser.parse_args(argv)
    if args.mode == "recovery" and args.csv is None and args.phase is None:
        parser.error("recovery needs the Part 1 CSV, --phase, or both")
    if args.mode == "compare" and len(args.csvs) != len(args.labels):
        parser.error("--csvs and --labels need the same number of entries")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "paths":
        path, actions = read_paths(args.path_csv)
        fig = plot_terminated_paths(path, actions, args.solver, args.title)
        stem = args.stem or args.path_csv.stem.removesuffix("_path")
        save_figure(fig, args.outdir or default_plots_dir(args.path_csv), stem, args.formats)
        plt.close(fig)
    elif args.mode == "compare":
        variants = [(label, *read_paths(csv)) for label, csv in zip(args.labels, args.csvs)]
        fig = plot_path_comparison(variants, args.title)
        save_figure(fig, args.outdir or default_plots_dir(args.csvs[0]), args.stem, args.formats)
        plt.close(fig)
    elif args.mode == "fdr-tpr":
        df = pd.read_csv(args.csv)
        outdir = args.outdir or default_plots_dir(args.csv)
        for fig, suffix in ((plot_fdr_tpr_vs_T(df, args.title), "fdr_tpr_vs_T"),
                            (plot_tpr_vs_fdr(df, args.title), "tpr_vs_fdr"),
                            (plot_path_stats(df, args.title, args.rank, args.p, args.L, args.s), "path_stats"),
                            (plot_refit_errors(df, args.title), "refit_errors")):
            save_figure(fig, outdir, f"{args.csv.stem}_{suffix}", args.formats)
            plt.close(fig)
    elif args.mode == "runtime":
        df = pd.read_csv(args.csv)
        outdir = args.outdir or default_plots_dir(args.csv)
        for fig, suffix in ((plot_runtime(df, args.title, args.p), "runtime"),
                            (plot_warm_start(df, args.title), "warm_start")):
            save_figure(fig, outdir, f"{args.csv.stem}_{suffix}", args.formats)
            plt.close(fig)
    elif args.mode == "memory":
        df = pd.read_csv(args.csv)
        fig = plot_memory(df, args.title)
        save_figure(fig, args.outdir or default_plots_dir(args.csv), f"{args.csv.stem}_memory", args.formats)
        plt.close(fig)
    elif args.mode == "recovery":
        if args.csv is not None:
            df = pd.read_csv(args.csv)
            for metric in RECOVERY_METRICS:
                fig = plot_recovery(df, metric, args.title)
                save_figure(fig, args.outdir or default_plots_dir(args.csv), f"{args.csv.stem}_{metric.lower()}",
                            args.formats)
                plt.close(fig)
        if args.phase is not None:
            fig = plot_phase_transition(pd.read_csv(args.phase), args.title)
            save_figure(fig, args.outdir or default_plots_dir(args.phase), args.phase.stem, args.formats)
            plt.close(fig)
    elif args.mode == "nhg":
        df = pd.read_csv(args.csv)
        outdir = args.outdir or default_plots_dir(args.csv)
        for fig, suffix in ((plot_null_pmf(df, args.title), "pmf"), (plot_null_qq(df, args.title), "qq"),
                            (plot_selected_ecdf(df, args.title), "ecdf")):
            save_figure(fig, outdir, f"{args.csv.stem}_{suffix}", args.formats)
            plt.close(fig)


if __name__ == "__main__":
    main()
