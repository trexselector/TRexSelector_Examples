#!/usr/bin/env bash
#
# Regenerate this demo's scalability dashboard via the suite plotting module
#   ../trex_plt_utils.R  (ggplot2, 'scalability' mode)
# Figures are written to simulation_results/plots/.
#
# Demo 08's CSVs use the wider n,p,is_barrier,solver,num_mc,snr,metric,value
# schema rather than the suite's tidy solver,metric,snr,value, so the plotter
# handles them in a dedicated 'scalability' mode: a 2x2 dashboard of runtime,
# memory, the out-of-core money plot and FDR control vs problem size. The base
# block and the large-p block are concatenated into one dashboard, so the
# n = 300 line runs from p = 1,000 to 1,000,000. Large-p points are drawn
# white-filled, and its cap-5 point at p = 100,000 sits next to the base one.
#
# Usage:
#   ./generate_plots.sh                          # this demo's default CSVs
#   ./generate_plots.sh <a.csv> [<b.csv> ...]    # explicit CSVs (e.g. smoke)
#   ./generate_plots.sh --formats png            # extra args pass through to
#   ./generate_plots.sh --tfdr 0.05              # trex_plt_utils.R
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rscript="${RSCRIPT:-$(command -v Rscript || true)}"
if [[ -z "$rscript" ]]; then
  echo "error: no Rscript found on PATH (set RSCRIPT=/path/to/Rscript)" >&2
  exit 1
fi

data="$here/simulation_results/data"

# Leading non-flag arguments override this demo's default CSVs; fall back to
# the smoke-run CSV when no benchmark block has been recorded yet.
csvs=()
while [[ $# -gt 0 && "$1" != -* ]]; do
  csvs+=("$1")
  shift
done
stem="demo_trex_08_scalability_dashboard"
if [[ ${#csvs[@]} -eq 0 ]]; then
  for s in demo_trex_08_scalability_results demo_trex_08_scalability_large_p; do
    [[ -f "$data/$s.csv" ]] && csvs+=("$data/$s.csv")
  done
fi
if [[ ${#csvs[@]} -eq 0 && -f "$data/demo_trex_08_scalability_smoke.csv" ]]; then
  echo "note: no benchmark CSV found - plotting the smoke-run CSV."
  csvs=("$data/demo_trex_08_scalability_smoke.csv")
  stem="demo_trex_08_scalability_smoke_dashboard"
fi
if [[ ${#csvs[@]} -eq 0 ]]; then
  echo "error: no result CSV found in $data" >&2
  exit 1
fi

exec "$rscript" "$here/../trex_plt_utils.R" scalability "${csvs[@]}" \
  --title 'T-Rex scalability on the memory-mapped pipeline (TLARS / TOMP / TAFS)' \
  --stem "$stem" \
  --outdir "$here/simulation_results/plots" \
  "$@"
