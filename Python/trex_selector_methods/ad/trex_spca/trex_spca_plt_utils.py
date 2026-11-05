#!/usr/bin/env python3
"""Plotting utilities for the T-Rex SPCA demonstration suite.

Reads the tidy result CSV written by the demo (see
``demo_trex_spca_01_mc_sim/simulation_results/data/``) and renders a
publication-style figure into the sibling ``simulation_results/plots/`` folder.
The visual conventions mirror the other suites: tab10 method colours, hollow
per-method markers, ONE figure-level legend outside the axes, PNG + PDF output.

CSV shape (``method,metric,snr_db,value``): the sweep axis is the
signal-to-noise ratio **in decibels**, and it is plotted as such — the grid is
linear in dB (-10 … 10) and is never converted back to a linear SNR. Three
metrics are reported per method:

* **FDR** — false discovery rate of the estimated loading support (PC1),
  against the ``tfdr`` target drawn as a dashed reference line. Together with
  TPR it makes up the main sweep figure; the PEV metrics are shown only in
  the dedicated two-normalization figure.
* **TPR** — true positive rate of that support.
* **PEV** — cumulative explained variance as a share of the **total** variance
  of X, the quantity the sparsity is bought against. PCA baselines lead here by
  construction, so the panel shows what the FDR control costs.
* **PEVsig** — the same adjusted EV normalized by the **Signal + Mixed EV of
  the data**: the variance carried by the true active variables, a denominator
  fixed per dataset and shared by all methods (Definition 1 of the reference
  paper, in the active-column convention of the legacy R reference
  demo_trex_spca_03_fig3.R). Uncapped: overshooting 100 %
  marks a method that is explaining null variance. Not present in older CSVs;
  when it is present a second figure is rendered that puts the two
  normalizations side by side.
* **PEVsigmix** — the method's own Signal + Mixed part over the same
  denominator. Read for OrdPCA it is the paper's "Ordinary PCA (Sig + Mix)"
  reference curve, drawn as a dashed line in the Definition-1 panels.

Passing SEVERAL sweep CSVs together with ``--layout panels`` renders the
Fig.-3-style panel figures instead: one panel per sweep axis (# PCs, SNR,
# true active loadings, target FDR), once under each PEV normalization. The
sweep axis of each file is auto-detected from its column name.

Usage:
    trex_spca_plt_utils.py <results.csv> [--outdir DIR] [--title T]
                           [--pev-title T] [--tfdr F] [--formats png pdf]
                           [--stem S] [--xscale {linear,sqrt}]
    trex_spca_plt_utils.py <a.csv> <b.csv> ... --layout panels [--stem S]
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
    AutoMinorLocator, MultipleLocator, ScalarFormatter,
)

DEFAULT_TFDR = 0.1
# Per-method marker cycle — overlapping curves stay distinguishable.
METHOD_MARKERS = ["o", "s", "^", "D", "v", "P"]
# Baselines first, then the T-Rex SPCA variants, so the legend reads top-down.
METHOD_ORDER = ["OrdPCA", "OraclePCA",
                "TRexSPCA-EN-Act", "TRexSPCA-EN-Thr",
                "TRexSPCA-ENaug-Act", "TRexSPCA-ENaug-Thr"]
# Panels of the main sweep figure: support recovery only. The explained
# variance is deliberately NOT shown here — it has its own dedicated
# two-normalization figure (see plot_spca_pev_normalizations), and duplicating
# it as a third panel added nothing.
METRICS = [("TPR", "TPR (true positive rate)", (0, 1.05)),
           ("FDR", "FDR (false discovery rate)", None)]

# Legend entries whose label starts with one of these is a reference line, not a
# method, and is hoisted above the "Method:" group header.
REFERENCE_PREFIXES = ("tFDR", "Signal", "OrdPCA (Sig")


# ---------------------------------------------------------------------------
# Shared helpers (house conventions)
# ---------------------------------------------------------------------------
def legend_tfdr_lead(handles: list, labels: list, group_label: str = "Method"):
    """Order legend entries as: tFDR reference first, then a group-header row,
    then the method entries below it.

    A legend *title* would sit at the very top and file the tFDR reference under
    the group heading it does not belong to. Mirrors the helper of the same name
    in the T-Rex+GVS and DA-TRex suites.
    """
    pairs = list(zip(handles, labels))
    ref = [pr for pr in pairs if str(pr[1]).startswith(REFERENCE_PREFIXES)]
    methods = [pr for pr in pairs if pr not in ref]
    ordered = list(ref)
    if methods:
        header = plt.Line2D([], [], linestyle="None", alpha=0.0)
        ordered.append((header, f"{group_label}:"))
        ordered.extend(methods)
    return [h for h, _ in ordered], [l for _, l in ordered]


def place_figure_legend(fig, handles, labels, group_label: str = "Method",
                        anchor_x: float = 0.845, fontsize: int = 9):
    """One figure-level legend outside the right panel, opaque white box."""
    handles, labels = legend_tfdr_lead(handles, labels, group_label)
    legend = fig.legend(
        handles, labels,
        loc="center left", bbox_to_anchor=(anchor_x, 0.5),
        frameon=True, framealpha=1.0, facecolor="white", edgecolor="0.7",
        fontsize=fontsize, ncol=1,
    )
    legend.get_frame().set_linewidth(0.8)
    return legend


def apply_db_xaxis(ax, values: list[float], xscale: str = "linear") -> None:
    """Configure the decibel sweep axis.

    The grid is already logarithmic in nature — decibels *are* a log scale — so
    it is plotted linearly in dB and never re-mapped. Major ticks are placed at
    a **whole number** of decibels and minor ticks every 1 dB in between: the
    sweeps are specified in integer dB, so a fractional label (``-7.5``) would
    name a level the grid never visits. ``sqrt`` is offered for parity with the
    other suites but is rarely useful on an axis that already spans evenly.
    """
    lo, hi = min(values), max(values)
    span = hi - lo
    pad = span * 0.04 if span > 0 else 0.5
    if xscale == "sqrt":
        shift = -lo + 1.0 if lo <= 0 else 0.0    # keep the transform defined
        ax.set_xscale("function",
                      functions=(lambda x: np.sqrt(np.maximum(x + shift, 0.0)),
                                 lambda x: np.square(x) - shift))
    ax.set_xlim(lo - pad, hi + pad)
    # Coarsest integer step that keeps the axis at <= 8 major intervals, so the
    # narrow demo-02 panels and the wide demo-01 figures stay equally readable.
    step = next((s for s in (1, 2, 5, 10, 20, 25, 50, 100) if span <= 8 * s),
                max(1, round(span / 8)))
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_minor_locator(AutoMinorLocator(step))   # one minor tick per dB
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.grid(which="minor", axis="x", alpha=0.25, linewidth=0.5)


def apply_rate_yaxis(ax, step: float) -> None:
    """Fixed-step y major ticks with halved minor ticks and a faint minor grid.

    Demo 01 resolves its rate/PEV axes in 0.1 steps (10 percentage points on
    the percent-scaled panels); the minor ticks land halfway between.
    """
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which="minor", axis="y", alpha=0.25, linewidth=0.5)


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


# ---------------------------------------------------------------------------
# Figure: TPR / FDR / PEV vs. SNR (dB)
# ---------------------------------------------------------------------------
def plot_spca_sweep(df: pd.DataFrame, title: str, tfdr: float = DEFAULT_TFDR,
                    xscale: str = "linear") -> plt.Figure:
    present = list(dict.fromkeys(df["method"]))
    methods = [m for m in METHOD_ORDER if m in present]
    methods += [m for m in present if m not in methods]   # keep any newcomers
    xs = sorted(df["snr_db"].unique())
    colors = matplotlib.color_sequences["tab10"]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(6.2 * len(METRICS), 4.8))
    legend_handles: list = []
    legend_labels: list[str] = []

    for ai, (metric, mlabel, ylim) in enumerate(METRICS):
        ax = axes[ai]
        for mi, method in enumerate(methods):
            sub = df[(df["method"] == method) & (df["metric"] == metric)]
            sub = sub.sort_values("snr_db")
            line, = ax.plot(
                sub["snr_db"], sub["value"],
                marker=METHOD_MARKERS[mi % len(METHOD_MARKERS)],
                markersize=5, markerfacecolor="none", markeredgewidth=1.2,
                color=colors[mi], label=method)
            if ai == 0:
                legend_handles.append(line)
                legend_labels.append(method)
        if ylim:
            ax.set_ylim(*ylim)
        apply_rate_yaxis(ax, 0.1)
        if metric == "FDR" and tfdr > 0:
            ref = ax.axhline(tfdr, color="black", linestyle=":", linewidth=2.0)
            legend_handles.append(ref)
            legend_labels.append(f"tFDR = {tfdr:g}")
        ax.set_title(mlabel, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        apply_db_xaxis(ax, xs, xscale)
        ax.set_xlabel("SNR (dB)", fontsize=11)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    place_figure_legend(fig, legend_handles, legend_labels, "Method",
                        anchor_x=0.815)
    fig.subplots_adjust(left=0.07, right=0.80, bottom=0.13, top=0.83,
                        wspace=0.26)
    return fig


# ---------------------------------------------------------------------------
# Extra figure: the two PEV normalizations side by side
# ---------------------------------------------------------------------------
def plot_spca_pev_normalizations(df: pd.DataFrame, title: str,
                                 xscale: str = "linear") -> plt.Figure:
    """Cumulative explained variance under both normalizations.

    Left panel: EV divided by the **total** variance of X. Every method is
    bounded by 1, so the panel reads as "how much of the data did we reproduce".

    Right panel: EV divided by the **Signal + Mixed EV of the data** — the
    variance carried by the true active variables, fixed per dataset and
    shared by all methods (Definition 1 of the reference paper, active-column
    convention of the legacy R reference). The point of the panel is that this
    ratio is *not* capped: loadings spent on null variables accumulate
    variance the denominator does not contain, so the curve climbs past
    100 %. Per the paper, exceeding 100 % "indicates an inferior performance
    of the respective method".

    Reproduces the reading of Fig. 3 in the reference paper on this suite's own
    sweep axis. When the CSV carries PEVsigmix, OrdPCA's Sig + Mix part is
    drawn as the paper's dashed reference curve.
    """
    present = list(dict.fromkeys(df["method"]))
    methods = [m for m in METHOD_ORDER if m in present]
    methods += [m for m in present if m not in methods]
    xs = sorted(df["snr_db"].unique())
    colors = matplotlib.color_sequences["tab10"]

    panels = [("PEV", "Normalized by the total variance of $X$"),
              ("PEVsig", "Normalized by Signal + Mixed EV (Definition 1)")]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    legend_handles: list = []
    legend_labels: list[str] = []

    for ai, (metric, panel_title) in enumerate(panels):
        ax = axes[ai]
        for mi, method in enumerate(methods):
            sub = df[(df["method"] == method) & (df["metric"] == metric)]
            sub = sub.sort_values("snr_db")
            line, = ax.plot(
                sub["snr_db"], 100.0 * sub["value"],
                marker=METHOD_MARKERS[mi % len(METHOD_MARKERS)],
                markersize=5, markerfacecolor="none", markeredgewidth=1.2,
                color=colors[mi], label=method)
            if ai == 0:
                legend_handles.append(line)
                legend_labels.append(method)
        if metric == "PEVsig":
            # Everything above this line is Null EV — variance harvested from
            # null loadings and reported as if it were explained signal.
            ref = ax.axhline(100.0, color="black", linestyle=":", linewidth=2.0)
            legend_handles.append(ref)
            legend_labels.append("Signal + Mixed EV = 100%")

            ref_sub = df[(df["method"] == "OrdPCA")
                         & (df["metric"] == "PEVsigmix")].sort_values("snr_db")
            if not ref_sub.empty:
                refline, = ax.plot(
                    ref_sub["snr_db"], 100.0 * ref_sub["value"],
                    linestyle="--", linewidth=1.6, color="0.35",
                    marker="x", markersize=4)
                legend_handles.append(refline)
                legend_labels.append("OrdPCA (Sig + Mix part)")
        ax.set_title(panel_title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Cumulative PEV (%)", fontsize=11)
        ax.grid(alpha=0.3)
        # 10 % steps = 0.1 in fraction units, matching the rate panels.
        apply_rate_yaxis(ax, 10.0)
        apply_db_xaxis(ax, xs, xscale)
        ax.set_xlabel("SNR (dB)", fontsize=11)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    place_figure_legend(fig, legend_handles, legend_labels, "Method",
                        anchor_x=0.815)
    fig.subplots_adjust(left=0.07, right=0.80, bottom=0.13, top=0.83,
                        wspace=0.26)
    return fig


# ---------------------------------------------------------------------------
# Multi-sweep panel figure (the Fig. 3 reproduction)
# ---------------------------------------------------------------------------
# Sweep column -> (axis label, categorical?, value scale). "Categorical" means
# the grid points are drawn evenly spaced rather than to scale: the PC-count
# grid runs 1 … 49 with most of its points crammed below 10, so a linear axis
# would pile them onto the left edge and waste the panel.
SWEEP_AXES = {
    "n_pcs":  ("# PCs",                   True,  1.0),
    "snr_db": ("SNR (dB)",                False, 1.0),
    "p1":     ("# True active loadings",  False, 1.0),
    "tfdr":   ("Target FDR (%)",          False, 100.0),
}

# Panel order of the reference paper's Fig. 3.
PANEL_ORDER = ["n_pcs", "snr_db", "p1", "tfdr"]


def detect_sweep_col(df: pd.DataFrame) -> str:
    """Return the sweep column of a tidy CSV (the one that is not fixed)."""
    for col in SWEEP_AXES:
        if col in df.columns:
            return col
    raise ValueError(f"No known sweep column in {sorted(df.columns)}; "
                     f"expected one of {sorted(SWEEP_AXES)}")


def plot_spca_pev_panels(frames: dict, metric: str, title: str,
                         ylabel: str, ref_100: bool,
                         xscale: str = "linear") -> plt.Figure:
    """One panel per sweep, all showing the same cumulative-PEV metric.

    ``frames`` maps sweep column -> DataFrame. Panels are laid out in the order
    of the reference paper's Fig. 3; missing sweeps are simply skipped, so the
    same helper renders a partial set while a long run is still in progress.

    ``ref_100`` draws the 100 % line — meaningful only under the Definition 1
    normalization, where crossing it means a method is explaining Null EV.
    """
    cols = [c for c in PANEL_ORDER if c in frames]
    if not cols:
        raise ValueError("No sweeps to plot.")

    colors = matplotlib.color_sequences["tab10"]
    fig, axes = plt.subplots(1, len(cols), figsize=(4.4 * len(cols), 4.8))
    if len(cols) == 1:
        axes = [axes]

    legend_handles: list = []
    legend_labels: list[str] = []

    for ai, col in enumerate(cols):
        ax = axes[ai]
        df = frames[col]
        label, categorical, scale = SWEEP_AXES[col]

        present = list(dict.fromkeys(df["method"]))
        methods = [m for m in METHOD_ORDER if m in present]
        methods += [m for m in present if m not in methods]

        xs = sorted(df[col].unique())
        positions = list(range(len(xs))) if categorical else [x * scale for x in xs]

        for mi, method in enumerate(methods):
            sub = df[(df["method"] == method) & (df["metric"] == metric)]
            sub = sub.sort_values(col)
            if sub.empty:
                continue
            ys = [100.0 * v for v in sub["value"]]
            line, = ax.plot(
                positions[:len(ys)], ys,
                marker=METHOD_MARKERS[mi % len(METHOD_MARKERS)],
                markersize=5, markerfacecolor="none", markeredgewidth=1.2,
                color=colors[mi], label=method)
            if ai == 0:
                legend_handles.append(line)
                legend_labels.append(method)

        if ref_100:
            ref = ax.axhline(100.0, color="black", linestyle=":", linewidth=2.0)
            if ai == 0:
                legend_handles.append(ref)
                legend_labels.append("Signal + Mixed EV = 100%")

        # The paper's "Ordinary PCA (Sig + Mix)" reference: OrdPCA's own
        # Signal + Mixed part over the shared Definition-1 denominator. Only
        # meaningful on the Definition-1 panels and only in newer CSVs.
        if metric == "PEVsig":
            ref_sub = df[(df["method"] == "OrdPCA")
                         & (df["metric"] == "PEVsigmix")].sort_values(col)
            if not ref_sub.empty:
                ys = [100.0 * v for v in ref_sub["value"]]
                refline, = ax.plot(
                    positions[:len(ys)], ys,
                    linestyle="--", linewidth=1.6, color="0.35",
                    marker="x", markersize=4)
                if ai == 0:
                    legend_handles.append(refline)
                    legend_labels.append("OrdPCA (Sig + Mix part)")

        if categorical:
            ax.set_xticks(positions)
            ax.set_xticklabels([f"{x:g}" for x in xs])
            ax.set_xlim(-0.4, len(xs) - 0.6)
        else:
            apply_db_xaxis(ax, positions, xscale)

        ax.set_xlabel(label, fontsize=11)
        # The #PCs panel accumulates over the swept count; the other sweeps
        # hold the extraction at the M = 3 true factors, and their y-labels
        # say so — matching the reference paper's "PCs 1 - 3" annotation.
        ax.set_ylabel(ylabel if col == "n_pcs"
                      else ylabel.replace("PEV", "PEV (PCs 1–3)"),
                      fontsize=11)
        ax.set_title(f"({chr(ord('a') + ai)}) vs. {label}",
                     fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    anchor = 1.0 - 0.14 / len(cols) * 4.0 * 0.42
    place_figure_legend(fig, legend_handles, legend_labels, "Method",
                        anchor_x=min(0.86, anchor))
    fig.subplots_adjust(left=0.05, right=min(0.85, anchor - 0.01),
                        bottom=0.13, top=0.82, wspace=0.24)
    return fig


# ---------------------------------------------------------------------------
# Figure: union-support FDR per PC (demo 01, section 2)
# ---------------------------------------------------------------------------
def plot_union_fdr(df: pd.DataFrame, title: str, tfdr: float = DEFAULT_TFDR,
                   xscale: str = "linear") -> plt.Figure:
    """Two panels from the union-FDR CSV (pc,metric,snr_db,value).

    Left: realized FDR of each PC's selected support scored against the UNION
    of the true factor supports (the only well-posed per-PC truth beyond PC1),
    with the tFDR target as a dotted line. Right: mean selected support size
    per PC. One line per component.
    """
    pcs = sorted(df["pc"].unique())
    xs = sorted(df["snr_db"].unique())
    colors = matplotlib.color_sequences["tab10"]

    panels = [("FDRunion", "Union-support FDR", "FDR (vs. union support)"),
              ("k", "Selected support size", "mean $|A_m|$")]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    legend_handles: list = []
    legend_labels: list[str] = []

    for ai, (metric, panel_title, ylabel) in enumerate(panels):
        ax = axes[ai]
        for pi, pc in enumerate(pcs):
            sub = df[(df["pc"] == pc) & (df["metric"] == metric)]
            sub = sub.sort_values("snr_db")
            line, = ax.plot(
                sub["snr_db"], sub["value"],
                marker=METHOD_MARKERS[pi % len(METHOD_MARKERS)],
                markersize=5, markerfacecolor="none", markeredgewidth=1.2,
                color=colors[pi], label=f"PC{pc:g}")
            if ai == 0:
                legend_handles.append(line)
                legend_labels.append(f"PC{pc:g}")
        if metric == "FDRunion":
            ax.set_ylim(0, 1.0)
            apply_rate_yaxis(ax, 0.1)
            if tfdr > 0:
                ref = ax.axhline(tfdr, color="black", linestyle=":", linewidth=2.0)
                legend_handles.append(ref)
                legend_labels.append(f"tFDR = {tfdr:g}")
        else:
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.grid(which="minor", axis="y", alpha=0.25, linewidth=0.5)
        ax.set_title(panel_title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(alpha=0.3)
        apply_db_xaxis(ax, xs, xscale)
        ax.set_xlabel("SNR (dB)", fontsize=11)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    place_figure_legend(fig, legend_handles, legend_labels, "Component",
                        anchor_x=0.845)
    fig.subplots_adjust(left=0.07, right=0.83, bottom=0.13, top=0.83,
                        wspace=0.26)
    return fig


# ---------------------------------------------------------------------------
# Figure: realized FDR / TPR heatmaps over the (target FDR x SNR) grid
# ---------------------------------------------------------------------------
def plot_fdr_tpr_heatmaps(df: pd.DataFrame, title: str) -> plt.Figure:
    """2x2 heatmap figure from the part-5 CSV (response,metric,tfdr,snr_db,value).

    Rows: selector response — "plugin" (y = X v1, what T-Rex SPCA regresses on)
    and "oracle" (y = z1, the true factor scores). Columns: realized FDR and
    TPR of PC1's support.

    The FDR panels are colored by the realized FDR itself on a sequential red
    scale from 0; the TPR panels by the TPR. Annotations show the values.
    """
    tfdrs = sorted(df["tfdr"].unique())
    snrs = sorted(df["snr_db"].unique())
    responses = [("plugin", "plug-in response $y = X\\,v_1$"),
                 ("oracle", "oracle response $y = z_1$ (true factor)")]

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8))

    for ri, (resp, rlabel) in enumerate(responses):
        for ci, metric in enumerate(["FDR", "TPR"]):
            ax = axes[ri][ci]
            sub = df[(df["response"] == resp) & (df["metric"] == metric)]
            M = np.full((len(snrs), len(tfdrs)), np.nan)
            for _, row in sub.iterrows():
                M[snrs.index(row["snr_db"]), tfdrs.index(row["tfdr"])] = row["value"]

            if metric == "FDR":
                vmax = max(0.10, float(np.nanmax(M)))
                im = ax.imshow(M, cmap="Reds", vmin=0.0, vmax=vmax,
                               aspect="auto", origin="lower")
                cbar_label = "FDR"
                shown = M
            else:
                vmin = min(0.5, float(np.nanmin(M)))
                im = ax.imshow(M, cmap="viridis", vmin=vmin, vmax=1.0,
                               aspect="auto", origin="lower")
                cbar_label = "TPR"
                shown = M

            for yi in range(len(snrs)):
                for xi in range(len(tfdrs)):
                    r, g, b, _ = im.cmap(im.norm(shown[yi, xi]))
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    ax.text(xi, yi, f"{M[yi, xi]:.2f}",
                            ha="center", va="center", fontsize=6.5,
                            color="black" if lum > 0.5 else "white")

            ax.set_xticks(range(len(tfdrs)))
            ax.set_xticklabels([f"{t * 100:g}" for t in tfdrs], fontsize=8)
            ax.set_yticks(range(len(snrs)))
            ax.set_yticklabels([f"{s:g}" for s in snrs], fontsize=8)
            ax.set_xlabel("Target FDR (%)", fontsize=10)
            ax.set_ylabel("SNR (dB)", fontsize=10)
            ax.set_title(f"Realized {metric} — {rlabel}",
                         fontsize=11, fontweight="bold")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.set_label(cbar_label, fontsize=9)
            cbar.ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.07, top=0.90,
                        wspace=0.22, hspace=0.34)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render T-Rex SPCA result figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path, nargs="+",
                   help="Result CSV(s). One CSV renders the sweep figures; "
                        "several with --layout panels render the combined "
                        "Fig.-3-style panel figures.")
    p.add_argument("--layout", choices=["auto", "panels"], default="auto",
                   help="'panels' combines several sweep CSVs into one figure.")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Output dir (default: sibling simulation_results/plots).")
    p.add_argument("--title", default=None, help="Sweep figure title.")
    p.add_argument("--pev-title", default=None,
                   help="Title of the extra PEV-normalization figure.")
    p.add_argument("--tfdr", type=float, default=DEFAULT_TFDR)
    p.add_argument("--formats", nargs="+", default=["png", "pdf"])
    p.add_argument("--stem", default=None,
                   help="Output filename stem (default: CSV stem).")
    p.add_argument("--xscale", choices=["linear", "sqrt"], default="linear",
                   help="Sweep-axis scale; the axis is in dB either way.")
    return p.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    """Read a tidy result CSV and strip whitespace from headers and cells."""
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()
    return df


def main() -> int:
    args = parse_args()

    # --- combined panel figures over several sweeps ------------------------
    if args.layout == "panels" or len(args.csv) > 1:
        frames = {}
        for path in args.csv:
            df = load_csv(path)
            frames[detect_sweep_col(df)] = df

        outdir = args.outdir or default_plots_dir(args.csv[0])
        stem = args.stem or "demo_trex_spca_02_mc_sim_pev"

        fig = plot_spca_pev_panels(
            frames, "PEVsig",
            args.title or "T-Rex SPCA: cumulative PEV (Definition 1)",
            "Cumulative PEV (%)", ref_100=True, xscale=args.xscale)
        save_figure(fig, outdir, f"{stem}_signal", args.formats)

        fig2 = plot_spca_pev_panels(
            frames, "PEV",
            args.pev_title or "T-Rex SPCA: cumulative PEV "
                              "(share of the total variance)",
            "Cumulative PEV (%)", ref_100=False, xscale=args.xscale)
        save_figure(fig2, outdir, f"{stem}_total", args.formats)
        return 0

    # --- single-sweep figures ---------------------------------------------
    csv_path = args.csv[0]
    df = load_csv(csv_path)

    # Union-FDR CSV (pc,metric,snr_db,value) gets its own two-panel figure.
    if "pc" in df.columns:
        outdir = args.outdir or default_plots_dir(csv_path)
        stem = args.stem or csv_path.stem
        fig = plot_union_fdr(
            df, args.title or "T-Rex SPCA: union-support FDR per PC",
            args.tfdr, args.xscale)
        save_figure(fig, outdir, stem, args.formats)
        return 0

    # Part-5 heatmap CSV (response,metric,tfdr,snr_db,value) gets its own figure.
    if "response" in df.columns:
        outdir = args.outdir or default_plots_dir(csv_path)
        stem = args.stem or csv_path.stem
        fig = plot_fdr_tpr_heatmaps(
            df, args.title or "T-Rex SPCA: realized FDR / TPR over "
                              "(target FDR × SNR), PC1 support")
        save_figure(fig, outdir, stem, args.formats)
        return 0

    missing = {"method", "metric", "snr_db", "value"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    outdir = args.outdir or default_plots_dir(csv_path)
    stem = args.stem or csv_path.stem

    fig = plot_spca_sweep(df, args.title or csv_path.stem, args.tfdr,
                          args.xscale)
    save_figure(fig, outdir, stem, args.formats)

    # Extra figure — only if the demo reported the second normalization.
    if "PEVsig" in set(df["metric"]):
        fig2 = plot_spca_pev_normalizations(
            df, args.pev_title or "T-Rex SPCA: cumulative PEV under both "
                                  "normalizations", args.xscale)
        save_figure(fig2, outdir, f"{stem}_pev", args.formats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
