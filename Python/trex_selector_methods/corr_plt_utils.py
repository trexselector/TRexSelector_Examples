# ==============================================================================
# corr_plt_utils.py
# ==============================================================================
#
# Shared plotter for the design-matrix *correlation heatmaps* embedded in the
# mc_sim demo READMEs. It renders the Pearson correlation matrix of an
# exemplary design X drawn from each demo's own DGP, in NATURAL variable order
# (no hierarchical clustering / no reordering) so the raw dependency structure
# the demo studies is visible as-is.
#
# The correlation of X depends only on the DGP's covariance parameters
# (n, p, rho, block layout, ...) and the RNG seed -- never on the support,
# coefficients, SNR or the response y -- so the exemplar builders below fix a
# seed and pass a nominal support purely to satisfy the DGP signatures.
#
# Usage (normally invoked via a demo's generate_corr_plot.sh):
#   python corr_plt_utils.py <demo_key> [--out PATH.png] [--dpi N]
#   python corr_plt_utils.py --list
#
# Each sub-suite's DGP module lives two levels down (ad/trex_da/, ad/trex_gvs/, ...);
# the builders add the relevant suite dir to sys.path lazily so only the DGP
# actually needed is imported.
# ==============================================================================

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))


def _suite_dgp(suite: str):
    """Add ad/<suite>/ to sys.path and return its module namespace helpers."""
    d = os.path.join(_HERE, "ad", suite)
    if d not in sys.path:
        sys.path.insert(0, d)


# ==============================================================================
# Rendering
# ==============================================================================

def plot_correlation_matrix(X, out_path, *, title, subtitle=None, crop=None,
                            dpi=150):
    """Render corr(X) as a diverging heatmap in natural order; save to out_path.

    X        : (n, p) design matrix.
    title    : bold heading (what the plot is).
    subtitle : DGP description line (which dataset), e.g. "AR(1), rho=0.7 ...".
    crop     : if set, show only the leading crop x crop sub-block (used for
               narrow-band designs whose structure is sub-pixel at full p); the
               crop is noted in the caption. corr of a column subset equals the
               leading block of the full correlation, so this stays faithful.
    """
    p_full = X.shape[1]
    if crop is not None and crop < p_full:
        X = X[:, :crop]
        note = f"leading {crop}x{crop} block of the p={p_full} design"
        subtitle = note if subtitle is None else f"{subtitle}  |  {note}"

    C = np.corrcoef(X, rowvar=False)
    p = C.shape[0]

    # Square figure that scales gently with p but stays README-friendly.
    side = 6.4
    fig, ax = plt.subplots(figsize=(side + 0.9, side))  # +width for the colorbar

    im = ax.imshow(C, cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                   interpolation="nearest", origin="upper")

    # Sparse, integer ticks (1-based labels read naturally for "variable j").
    n_ticks = min(6, p)
    ticks = np.linspace(0, p - 1, n_ticks).round().astype(int)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(t + 1) for t in ticks])
    ax.set_yticklabels([str(t + 1) for t in ticks])
    ax.set_xlabel("variable index")
    ax.set_ylabel("variable index")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation", rotation=270, labelpad=15)
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    heading = title if subtitle is None else f"{title}\n{subtitle}"
    ax.set_title(heading, fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ==============================================================================
# Exemplar builders  ->  (X, title, subtitle)
#
# One entry per mc_sim demo. Keep the (n, p, covariance) params in sync with the
# demo's own DGP call; support/SNR are irrelevant to corr(X) and only satisfy
# the DGP signatures.
# ==============================================================================

_SEED = 2026
_TITLE = "Design-matrix correlation (natural order, no clustering)"

# Leading-block crop for narrow-band (AR(1)/nearest-neighbour) designs whose
# structure is sub-pixel at full p; block/equicorrelation/iid designs use full p.
_NB = 150   # narrow-band crop
_SP = 75    # small structured-prefix crop (few AR cols before a white tail)


def _da():
    _suite_dgp("trex_da")
    import dgp_generators as g
    return g


def _sup(k=10):
    return np.arange(k)


# --- trex_da: 8 DGPs (dgp_generators.py) ---------------------------------------

def _ex_da_ar1():
    d = _da().dgp_ar1_snr(n=300, p=1000, support=_sup(), rho=0.7, seed=_SEED)
    return dict(X=d["X"], title=_TITLE, crop=_NB,
                subtitle="AR(1) Toeplitz, rho=0.7  |  n=300, p=1000")


def _ex_da_equi():
    d = _da().dgp_equi_snr(n=300, p=1000, support=_sup(), rho=0.25, seed=_SEED)
    return dict(X=d["X"], title=_TITLE, crop=None,
                subtitle="Equicorrelation, rho=0.25  |  n=300, p=1000")


def _ex_da_nn():
    d = _da().dgp_nn_snr(n=300, p=1000, support=_sup(), kappa=3, rho=0.7, seed=_SEED)
    return dict(X=d["X"], title=_TITLE, crop=_NB,
                subtitle="Nearest-neighbour, kappa=3, rho=0.7  |  n=300, p=1000")


def _ex_da_ar1_block():
    d = _da().dgp_ar1_block_snr(n=300, p=1000, support=_sup(), n_blocks=5,
                                rho=0.7, seed=_SEED)
    return dict(X=d["X"], title=_TITLE, crop=None,
                subtitle="Block-diagonal AR(1), 5 blocks, rho=0.7  |  n=300, p=1000")


def _ex_da_ar1_block_white():
    d = _da().dgp_ar1_block_white_snr(n=300, p=1000, support=_sup(5), p_ar=25,
                                      p_white=975, n_blocks=5, rho=0.7, seed=_SEED)
    return dict(X=d["X"], title=_TITLE, crop=_SP,
                subtitle="AR(1) blocks (25) + white (975), rho=0.7  |  n=300, p=1000")


def _ex_da_block_toeplitz_hvt():
    d = _da().dgp_block_toeplitz_hvt_snr(n=300, p=1000, support=_sup(), n_blocks=5,
                                         rho=0.8, nu=3.0, seed=_SEED)
    return dict(X=d["X"], title=_TITLE, crop=None,
                subtitle="Heavy-tailed block Toeplitz, 5 blocks, rho=0.8, nu=3  |  n=300, p=1000")


def _ex_da_ht_block_white():
    d = _da().dgp_ht_block_white_snr(n=300, p=500, support=_sup(5), p_ar=25,
                                     p_white=475, n_blocks=5, rho=0.8, nu=3.0,
                                     seed=_SEED)
    return dict(X=d["X"], title=_TITLE, crop=_SP,
                subtitle="Heavy-tailed AR blocks (25) + t3 white (475), rho=0.8  |  n=300, p=500")


def _ex_da_groups():
    groups = [[j // 10 for j in range(1000)],
              [j // 50 for j in range(1000)],
              [j // 250 for j in range(1000)]]
    d = _da().dgp_groups_toeplitz_leaf(n=300, p=1000, support=_sup(), groups=groups,
                                       rho_levels=[0.55, 0.25, 0.10], phi=0.50,
                                       seed=_SEED)
    return dict(X=d["X"], title=_TITLE, crop=None,
                subtitle="Prior groups (10/50/250) + Toeplitz leaf, phi=0.5  |  n=300, p=1000")


# --- trex: iid Gaussian (dgp_gauss_snr) ---------------------------------------

def _ex_trex_gauss():
    _suite_dgp("trex_core")
    from dgp_gauss_snr import dgp_gauss_snr
    d = dgp_gauss_snr(n=300, p=1000, support=_sup(), coefs=np.ones(10), snr=1.0,
                      seed=_SEED)
    return dict(X=d["X"], title=_TITLE, crop=None,
                subtitle="i.i.d. Gaussian, independent columns (corr ~ identity)"
                         "  |  n=300, p=1000")


# --- trex_gvs: 8 DGPs (trex_gvs_dgps.py) --------------------------------------

def _gvs():
    _suite_dgp("trex_gvs")
    import trex_gvs_dgps as g
    return g


def _ex_gvs_hastie():
    d = _gvs().dgp_hastie_snr(n=200, p=500, seed=_SEED)
    return dict(X=d["X"], title=_TITLE, crop=None,
                subtitle="Hastie grouped elastic-net design  |  n=200, p=500")


def _ex_gvs_scattered():
    d = _gvs().dgp_scattered_grouped_snr(n=200, p=500, seed=_SEED)
    return dict(X=d["X"], title=_TITLE, crop=None,
                subtitle="Scattered correlated groups (gaps randomised)  |  n=200, p=500")


def _ex_gvs_mixed_blocks():
    d = _gvs().dgp_mixed_blocks_snr(n=200, p=500, seed=_SEED)
    return dict(X=d["X"], title=_TITLE, crop=None,
                subtitle="Mixed-size scattered blocks (20/50/80/65)  |  n=200, p=500")


def _ex_gvs_neg_traps():
    d = _gvs().dgp_neg_traps_snr(n=200, p=500, seed=_SEED)
    return dict(X=d["X"], title=_TITLE, crop=None,
                subtitle="Negative-correlation traps  |  n=200, p=500")


def _ex_gvs_t3_blocks():
    d = _gvs().dgp_t3_equi_blocks_snr(n=200, p=500, rho=0.75, df=3, seed=_SEED)
    return dict(X=d["X"], title=_TITLE, crop=None,
                subtitle="Student-t(3) equicorrelated blocks, rho=0.75  |  n=200, p=500")


def _ex_gvs_ar1_blocks():
    d = _gvs().dgp_ar1_blocks_snr(n=200, p=500, rho=0.85, seed=_SEED)
    return dict(X=d["X"], title=_TITLE, crop=None,
                subtitle="Block-diagonal AR(1), rho=0.85  |  n=200, p=500")


def _ex_gvs_arma_blocks():
    d = _gvs().dgp_arma_blocks_snr(n=200, p=500, seed=_SEED)
    return dict(X=d["X"], title=_TITLE, crop=None,
                subtitle="Block-diagonal ARMA (ar=0.8, ma=0.5/0.3/0.1)  |  n=200, p=500")


def _ex_gvs_block_equicorr():
    d = _gvs().dgp_block_equicorr(n=200, p=500, G=100, block_size=5,
                                  n_active_blocks=10, rho=0.9, seed=_SEED)
    # 100 blocks of 5 columns are sub-pixel at full p -- crop so the tiny
    # 5 x 5 diagonal blocks stay legible.
    return dict(X=d["X"], title=_TITLE, crop=_SP,
                subtitle="Equicorrelated blocks (100 x size-5), rho=0.9  |  n=200, p=500")


# --- trex_screening: 4 DGPs (trex_scr_common.py) ------------------------------

def _scr():
    _suite_dgp("trex_screening")
    import trex_scr_common as g
    return g


def _ex_scr_iid():
    d = _scr().dgp_iid_snr(n=300, p=1000, true_support=_sup(), coefs=np.ones(10),
                           snr=1.0, seed=_SEED)
    return dict(X=d["X"], title=_TITLE, crop=None,
                subtitle="i.i.d. Gaussian, independent columns  |  n=300, p=1000")


def _ex_scr_ar1():
    d = _scr().dgp_ar1(n=300, p=1000, p1=10, rho=0.5, snr=1.0, beta_val=1.0,
                       seed=_SEED)
    return dict(X=d["X"], title=_TITLE, crop=_NB,
                subtitle="AR(1) column correlation, rho=0.5  |  n=300, p=1000")


def _ex_scr_equi():
    d = _scr().dgp_equi(n=300, p=1000, p1=10, rho=0.4, snr=1.0, beta_val=1.0,
                        seed=_SEED)
    return dict(X=d["X"], title=_TITLE, crop=None,
                subtitle="Equicorrelation (one shared factor), rho=0.4  |  n=300, p=1000")


def _ex_scr_block_equi():
    d = _scr().dgp_block_equi(n=300, p=1000, p1=10, rho=0.9, n_blocks=5, snr=1.0,
                              beta_val=1.0, seed=_SEED)
    return dict(X=d["X"], title=_TITLE, crop=None,
                subtitle="Block-equicorrelation, 5 blocks, rho=0.9  |  n=300, p=1000")


# --- trex_spca: sparse factor model (trex_spca_sim_utils.py) -------------------

def _ex_spca_factor():
    _suite_dgp("trex_spca")
    from trex_spca_sim_utils import generate_sparse_factor_model
    d = generate_sparse_factor_model(n=50, p=100, p1=5, M=3, overlap_pool_size=30,
                                     seed=_SEED)
    return dict(X=d["X"], title=_TITLE, crop=None,
                subtitle="Sparse factor model (M=3, p1=5, pool=30)  |  n=50, p=100")


# key -> builder. Keys are suite-namespaced DGP names; the output file is
# <suite>/dgp_corr_plots/<key>.png (the per-suite generate_corr_plots.sh sets it).
EXEMPLARS = {
    # trex_da
    "da_ar1":                _ex_da_ar1,
    "da_equi":               _ex_da_equi,
    "da_nn":                 _ex_da_nn,
    "da_ar1_block":          _ex_da_ar1_block,
    "da_ar1_block_white":    _ex_da_ar1_block_white,
    "da_block_toeplitz_hvt": _ex_da_block_toeplitz_hvt,
    "da_ht_block_white":     _ex_da_ht_block_white,
    "da_groups":             _ex_da_groups,
    # trex
    "trex_gauss":            _ex_trex_gauss,
    # trex_gvs
    "gvs_hastie":            _ex_gvs_hastie,
    "gvs_scattered":         _ex_gvs_scattered,
    "gvs_mixed_blocks":      _ex_gvs_mixed_blocks,
    "gvs_neg_traps":         _ex_gvs_neg_traps,
    "gvs_t3_blocks":         _ex_gvs_t3_blocks,
    "gvs_ar1_blocks":        _ex_gvs_ar1_blocks,
    "gvs_arma_blocks":       _ex_gvs_arma_blocks,
    "gvs_block_equicorr":    _ex_gvs_block_equicorr,
    # trex_screening
    "scr_iid":               _ex_scr_iid,
    "scr_ar1":               _ex_scr_ar1,
    "scr_equi":              _ex_scr_equi,
    "scr_block_equi":        _ex_scr_block_equi,
    # trex_spca
    "spca_factor":           _ex_spca_factor,
}


# ==============================================================================
# CLI
# ==============================================================================

def _default_out(demo_key: str) -> str:
    # The per-suite generate_corr_plots.sh sets --out to
    # <suite>/dgp_corr_plots/<key>.png; this is only a CWD-relative fallback.
    return os.path.join("dgp_corr_plots", f"{demo_key}.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render a demo's design-matrix "
                                             "correlation heatmap.")
    ap.add_argument("demo_key", nargs="?", help="e.g. trex_da_01")
    ap.add_argument("--out", default=None, help="output PNG path")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--list", action="store_true", help="list demo keys and exit")
    args = ap.parse_args(argv)

    if args.list or not args.demo_key:
        for k in sorted(EXEMPLARS):
            print(k)
        return 0

    if args.demo_key not in EXEMPLARS:
        ap.error(f"unknown demo key {args.demo_key!r}; try --list")

    ex = EXEMPLARS[args.demo_key]()
    out = args.out or _default_out(args.demo_key)
    path = plot_correlation_matrix(ex["X"], out, title=ex["title"],
                                   subtitle=ex.get("subtitle"),
                                   crop=ex.get("crop"), dpi=args.dpi)
    print(f"[corr] wrote {path}  (p={ex['X'].shape[1]}, "
          f"crop={ex.get('crop')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
