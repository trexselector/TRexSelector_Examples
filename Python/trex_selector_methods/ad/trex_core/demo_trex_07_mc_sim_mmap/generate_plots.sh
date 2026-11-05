#!/usr/bin/env bash
#
# Regenerate this demo's memory-mapping figures via the suite plotting module
#   ../trex_plt_utils.py
# using the repo's local .venv. Figures are written to
# simulation_results/plots/.
#
# Demo 07 runs the same MC sweep (TLARS, SNR in {0.1, 0.5, 1, 2, 5}, MC=200)
# under two storage scenarios: (C) X in RAM with the dummy matrices D memory-
# mapped, and (D) a fully memory-mapped X + D pipeline. The demo's question is
# whether the storage medium is transparent to the algorithm, so the headline
# figure overlays both scenarios on one pair of axes -- the curves should
# coincide up to MC noise. This script produces:
#   * the C-vs-D scenario overlay (+ interactive html with --plotly) -- the core figure;
#   * a per-scenario FDR/TPR overview (+ html) for each CSV, for reading exact
#     values off a single run.
#
# The overlay is built by merging the two per-scenario CSVs into one tidy frame
# whose 'solver' column carries the scenario name. That column is the plotting
# module's generic "what this demo varies" dimension (base solvers in demos
# 02/03, L-loop strategies in demo 04, dummy distributions in demo 05 -- storage
# scenarios here). The merge is a temporary file: simulation_results/data/ holds
# only what the demo binary writes.
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

plotter="$here/../trex_plt_utils.py"
data="$here/simulation_results/data"
plots="$here/simulation_results/plots"
prefix="d03_trex_mmap_demo"
tfdr=0.1

csv_c="$data/${prefix}_c_n300_p1000_sw7.csv"
csv_d="$data/${prefix}_d_n300_p1000_sw7.csv"

# Scenario labels. No commas: they become a CSV field below.
label_c='Demo C: X in RAM + D mmap'
label_d='Demo D: fully mmapped X + D'

# First non-flag argument: plot just that CSV instead of the full set. A CSV
# passed here is a per-scenario file, whose rows are solvers -- 'Scenario' only
# names the rows of the merged overlay built further down.
if [[ $# -gt 0 && "$1" != -* ]]; then
  csv="$1"
  shift
  exec "$venv_python" "$plotter" "$csv" --legend-title 'Method' \
    --tfdr "$tfdr" --no-grouped "$@"
fi

mkdir -p "$plots"

# --- Headline figure: both storage scenarios on one pair of axes -------------
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
merged="$tmpdir/mmap_c_vs_d.csv"

# Rewrite the 'solver' column (always TLARS here) to the scenario name and
# concatenate. Fields 2-4 (metric, snr, value) are carried over untouched.
{
  echo "solver,metric,snr,value"
  awk -F, -v lbl="$label_c" 'NR > 1 {print lbl "," $2 "," $3 "," $4}' "$csv_c"
  awk -F, -v lbl="$label_d" 'NR > 1 {print lbl "," $2 "," $3 "," $4}' "$csv_d"
} > "$merged"

"$venv_python" "$plotter" "$merged" \
  --title 'T-Rex MC under memory-mapped I/O ($n=300$, $p=1000$, TLARS, MC=200): storage scenarios compared' \
  --legend-title 'Scenario' --tfdr "$tfdr" --no-grouped \
  --stem "${prefix}_c_vs_d" --outdir "$plots" "$@"

# --- Per-scenario overviews --------------------------------------------------
# The grouped view splits the row set in half; with a single row per scenario
# there is nothing to de-clutter, so it is skipped.
"$venv_python" "$plotter" "$csv_c" \
  --title 'T-Rex MC, Demo C: in-memory $\mathbf{X}$ + memory-mapped $\mathbf{D}$ ($n=300$, $p=1000$)' \
  --legend-title 'Method' --tfdr "$tfdr" --no-grouped "$@"

"$venv_python" "$plotter" "$csv_d" \
  --title 'T-Rex MC, Demo D: fully memory-mapped $\mathbf{X}$ + $\mathbf{D}$ ($n=300$, $p=1000$)' \
  --legend-title 'Method' --tfdr "$tfdr" --no-grouped "$@"
