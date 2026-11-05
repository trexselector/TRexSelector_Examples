#!/usr/bin/env bash
#
# Regenerate this demo's DA-TRex+BT heavy-tailed figures via the suite plotting
# module
#   ../trex_da_plt_utils.py
# using the repo's local .venv. Figures are written to
# simulation_results/plots/.
#
# Demo 05 (BT on heavy-tailed block Toeplitz + i.i.d. Student-t white noise) sweeps SNR, rho, Q, M, and tFDR,
# each under two noise scenarios {s1_Gauss, s2_Heavy} x HAC linkage
# {Single, Complete, Average}, with the AR blocks embedded in a
# larger design padded with heavy-tailed white-noise columns. This
# script produces:
#   * a per-CSV FDR/TPR overview (+ interactive html with --plotly) for every sweep x scenario
#     x linkage;
#   * one linkage-comparison grid per (sweep x scenario);
#   * one Gaussian-vs-heavy scenario grid per sweep (at Average linkage) -- the
#     figure for this demo's core question, how much heavy tails cost in FDR
#     control and power.
#
# Usage:
#   ./generate_plots.sh                 # this demo's full figure set
#   ./generate_plots.sh <other.csv>     # just that CSV (default FDR/TPR mode)
#   ./generate_plots.sh --formats png   # extra args pass through to the plotter
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find the repo-root .venv by walking up the directory tree.
dir="$here"
venv_python=""
while [[ "$dir" != "/" ]]; do
  if [[ -x "$dir/.venv/bin/python" ]]; then
    venv_python="$dir/.venv/bin/python"
    break
  fi
  dir="$(dirname "$dir")"
done

if [[ -z "$venv_python" ]]; then
  echo "error: no .venv/bin/python found above $here" >&2
  echo "       create it and 'pip install matplotlib pandas plotly', then re-run." >&2
  exit 1
fi

plotter="$here/../trex_da_plt_utils.py"
data="$here/simulation_results/data"
plots="$here/simulation_results/plots"
prefix="da_trex_mc_da_ht_blocks_plus_white"
tfdr=0.2

# First non-flag argument: plot just that CSV instead of the full set.
if [[ $# -gt 0 && "$1" != -* ]]; then
  csv="$1"
  shift
  exec "$venv_python" "$plotter" "$csv" --legend-title 'Method' --tfdr "$tfdr" "$@"
fi

# Human-readable labels for the figure titles (functions rather than associative
# arrays, so this stays compatible with macOS's stock bash 3.2).
sweep_label() {
  case "$1" in
    snr)  echo 'SNR sweep' ;;
    rho)  echo '$\rho$ sweep' ;;
    Q)    echo 'block-size $Q$ sweep' ;;
    M)    echo 'number-of-blocks $M$ sweep' ;;
    tFDR) echo 'tFDR sweep' ;;
  esac
}
scen_label() {
  case "$1" in
    s1_Gauss) echo 'Gaussian noise' ;;
    s2_Heavy) echo 'heavy-tailed noise' ;;
  esac
}

# The SNR grid {0.1, 0.2, 0.5, 0.6, 1, 2, 5} carries the near-neighbours 0.5/0.6,
# whose labels collide on the default log axis. A linear axis ticked at round
# numbers with minor ticks in between keeps every label readable; the other
# sweeps are evenly spaced and tick fine at their swept values.
sweep_xscale() {
  case "$1" in
    snr) echo '--xscale linear-minor' ;;
    *)   echo '' ;;
  esac
}

for sweep in snr rho Q M tFDR; do
  label="$(sweep_label "$sweep")"
  # Unquoted on purpose below: expands to zero or two words.
  xscale=$(sweep_xscale "$sweep")

  for scen in s1_Gauss s2_Heavy; do
    slab="$(scen_label "$scen")"

    # Per-linkage overview (+ interactive html with --plotly).
    for linkage in Single Complete Average; do
      csv="$data/${prefix}_${sweep}_${scen}_${linkage}.csv"
      "$venv_python" "$plotter" "$csv" \
        --title "DA-TRex+BT, heavy-tailed + white: ${label}, ${slab}, ${linkage} linkage" \
        --legend-title 'Method' --tfdr "$tfdr" $xscale "$@"
    done

    # Linkage-comparison grid for this (sweep, scenario).
    "$venv_python" "$plotter" grid \
      --labels Single Complete Average \
      --csvs "$data/${prefix}_${sweep}_${scen}_Single.csv" \
             "$data/${prefix}_${sweep}_${scen}_Complete.csv" \
             "$data/${prefix}_${sweep}_${scen}_Average.csv" \
      --title "DA-TRex+BT, heavy-tailed + white: ${label} across HAC linkage (${slab})" \
      --legend-title 'Method' --column-title 'Linkage' \
      --stem "${prefix}_${sweep}_${scen}_linkage_grid" --outdir "$plots" \
      --tfdr "$tfdr" $xscale "$@"
  done

  # Gaussian-vs-heavy scenario grid (at Average linkage) for this sweep.
  "$venv_python" "$plotter" grid \
    --labels 'Gaussian' 'Heavy-tailed' \
    --csvs "$data/${prefix}_${sweep}_s1_Gauss_Average.csv" \
           "$data/${prefix}_${sweep}_s2_Heavy_Average.csv" \
    --title "DA-TRex+BT + white: ${label}, Gaussian vs heavy-tailed noise (Average linkage)" \
    --legend-title 'Method' --column-title 'Noise' \
    --stem "${prefix}_${sweep}_scenario_grid" --outdir "$plots" --tfdr "$tfdr" \
    $xscale "$@"
done
