#!/usr/bin/env bash
#
# Regenerate this demo's FDR/TPR figures via the plotting module:
#   ../trex_plt_utils.R  (ggplot2)
# One figure set is rendered per base solver (TLARS / TOMP / TAFS); figures
# are written to simulation_results/plots/.
#
# Usage:
#   ./generate_plots.sh                          # all three solver CSVs
#   ./generate_plots.sh <other.csv>              # one explicit CSV
#   ./generate_plots.sh --formats png            # extra args pass through to
#   ./generate_plots.sh --tfdr 0.05              # trex_plt_utils.R
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rscript="${RSCRIPT:-$(command -v Rscript || true)}"
if [[ -z "$rscript" ]]; then
  echo "error: no Rscript found on PATH (set RSCRIPT=/path/to/Rscript)" >&2
  exit 1
fi

# First non-flag argument: plot just that CSV instead of the three defaults.
if [[ $# -gt 0 && "$1" != -* ]]; then
  csv="$1"
  shift
  exec "$rscript" "$here/../trex_plt_utils.R" "$csv" \
    --legend-title 'Dummy distribution' \
    "$@"
fi

# TAFS is the only base solver here run with a non-zero AFS correlation
# parameter (rho_afs = 0.3 in this demo's solver table; TLARS/TOMP take 0), so
# the figures name it -- otherwise the setting is lost when a plot is read
# apart from this demo's README. The loop variable stays the raw name: it also
# builds the CSV filename token.
solver_label() {
  case "$1" in
    TAFS) echo 'TAFS (rho = 0.3)' ;;
    *)    echo "$1" ;;
  esac
}

for solver in TLARS TOMP TAFS; do
  token="$(echo "$solver" | tr '[:upper:]' '[:lower:]')"
  label="$(solver_label "$solver")"
  csv="$here/simulation_results/data/demo_trex_05_dummy_distributions_results_n300_p1000_${token}_random_support.csv"
  "$rscript" "$here/../trex_plt_utils.R" "$csv" \
    --title 'T-Rex dummy distributions (n = 300, p = 1000, random support, '"$label"' + STANDARD)' \
    --legend-title 'Dummy distribution' \
    "$@"
done

# Cross-solver comparison grid (2x3: FDR/TPR rows x TLARS/TOMP/TAFS cols, all
# twelve dummy distributions per panel on a shared scale) -- the suite
# plotter's 'grid' mode. This is the figure for the demo's core question:
# whether the dummy distribution's effect on FDR/TPR depends on the base
# solver family.
data="$here/simulation_results/data"
"$rscript" "$here/../trex_plt_utils.R" grid \
  --labels TLARS TOMP "$(solver_label TAFS)" \
  --csvs \
    "$data/demo_trex_05_dummy_distributions_results_n300_p1000_tlars_random_support.csv" \
    "$data/demo_trex_05_dummy_distributions_results_n300_p1000_tomp_random_support.csv" \
    "$data/demo_trex_05_dummy_distributions_results_n300_p1000_tafs_random_support.csv" \
  --title 'T-Rex dummy distributions across base solvers (n = 300, p = 1000, random support, STANDARD L-loop)' \
  --legend-title 'Dummy distribution' \
  --stem demo_trex_05_dummy_distributions_grid \
  --outdir "$here/simulation_results/plots" \
  "$@"
