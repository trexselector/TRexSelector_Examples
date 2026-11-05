#!/usr/bin/env bash
#
# Regenerate this demo's FDR/TPR figures via the plotting module:
#   ../trex_plt_utils.R  (ggplot2)
# Figures are written to simulation_results/plots/.
#
# Usage:
#   ./generate_plots.sh                          # this demo's default CSVs
#   ./generate_plots.sh <other.csv>              # explicit CSV
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
    --legend-title 'L-strategy' \
    "$@"
fi

# TAFS is the only base solver here run with a non-zero AFS correlation
# parameter (rho_afs = 0.3 in this demo's solver table; TLARS/TOMP take 0), so
# the figures name it -- otherwise the setting is lost when a plot is read
# apart from this demo's README. The loop variable stays the raw name: it also
# builds the CSV filename token.
solver_label() {
  case "$1" in
    TAFS)       echo 'TAFS (rho = 0.3)' ;;
    TLARS_STAG) echo 'TLARS, stagnation stop' ;;
    *)          echo "$1" ;;
  esac
}

# One per-solver figure set (overview + grouped) per base solver, via the
# suite-level plotter. TLARS_STAG (TLARS with the stagnation stop on) joins the
# figures once its CSV exists.
data="$here/simulation_results/data"
grid_labels=()
grid_csvs=()
for solver in TLARS TOMP TAFS TLARS_STAG; do
  token="$(echo "$solver" | tr '[:upper:]' '[:lower:]')"
  label="$(solver_label "$solver")"
  csv="$data/demo_trex_04_lloop_strategies_results_n300_p1000_${token}_random_support.csv"
  [[ -f "$csv" ]] || continue
  grid_labels+=("$label")
  grid_csvs+=("$csv")
  "$rscript" "$here/../trex_plt_utils.R" "$csv" \
    --title 'T-Rex L-loop strategies (n = 300, p = 1000, random support, '"$label"')' \
    --legend-title 'L-strategy' \
    "$@"
done

# Cross-solver comparison grid (FDR/TPR rows x one column per base solver, all
# eight L-loop strategies per panel on a shared scale) -- the suite plotter's
# 'grid' mode. This is the figure for the demo's core question: whether the
# SKIPL FDR overshoot is specific to the LARS path (TLARS) or persists under
# the greedy solvers (TOMP / TAFS), and whether the stagnation stop removes it
# (TLARS_STAG).
"$rscript" "$here/../trex_plt_utils.R" grid \
  --labels "${grid_labels[@]}" \
  --csvs "${grid_csvs[@]}" \
  --title 'T-Rex L-loop strategies across base solvers (n = 300, p = 1000, random support)' \
  --legend-title 'L-strategy' \
  --stem demo_trex_04_lloop_strategies_grid \
  --outdir "$here/simulation_results/plots" \
  "$@"
