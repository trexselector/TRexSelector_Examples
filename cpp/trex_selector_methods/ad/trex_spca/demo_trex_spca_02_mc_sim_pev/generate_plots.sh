#!/usr/bin/env bash
#
# Regenerate this demo's T-Rex SPCA figures via the suite plotting module
#   ../trex_spca_plt_utils.py
# using the repo's local .venv. Figures are written to simulation_results/plots/.
#
# The four sweep CSVs are combined into ONE panel figure per normalization —
# the Fig.-3-style reproduction — rather than plotted one file at a time.
#
# Usage:
#   ./generate_plots.sh                 # this demo's full figure set
#   ./generate_plots.sh --formats png   # extra args pass through to the plotter
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find the repo-root .venv by walking up the directory tree.
dir="$here"; venv_python=""
while [[ "$dir" != "/" ]]; do
  if [[ -x "$dir/.venv/bin/python" ]]; then venv_python="$dir/.venv/bin/python"; break; fi
  dir="$(dirname "$dir")"
done
if [[ -z "$venv_python" ]]; then
  echo "error: no .venv/bin/python found above $here" >&2
  echo "       create it and 'pip install matplotlib pandas numpy', then re-run." >&2
  exit 1
fi

plotter="$here/../trex_spca_plt_utils.py"
data="$here/simulation_results/data"

# Panel order follows the reference paper's Fig. 3: PCs, SNR, loadings, tFDR.
# Missing files are skipped, so a partial run still renders what it has.
csvs=()
for part in pcs snr loadings tfdr; do
  f="$data/demo_trex_spca_02_mc_sim_pev_${part}.csv"
  [[ -f "$f" ]] && csvs+=("$f")
done

if [[ ${#csvs[@]} -eq 0 ]]; then
  echo "error: no sweep CSVs in $data — run the demo first." >&2
  exit 1
fi

"$venv_python" "$plotter" "${csvs[@]}" \
  --layout panels \
  --stem demo_trex_spca_02_mc_sim_pev \
  --title 'T-Rex SPCA: cumulative PEV (Definition 1 — Signal + Mixed EV)' \
  --pev-title 'T-Rex SPCA: cumulative PEV (share of the total variance)' \
  "$@"

# Part-5 FDR/TPR heatmaps (rendered only if the sweep has been run).
heatmap_csv="$data/demo_trex_spca_02_mc_sim_pev_fdr_heatmap.csv"
if [[ -f "$heatmap_csv" ]]; then
  "$venv_python" "$plotter" "$heatmap_csv" \
    --title 'T-Rex SPCA: realized FDR / TPR over (target FDR × SNR), PC1 support' \
    "$@"
fi
