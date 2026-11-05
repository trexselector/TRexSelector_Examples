#!/usr/bin/env bash
#
# Overnight: (re)generate the T-Rex SPCA suite's result data.
#
# Command:
#   ./run_overnight.sh              # run everything (default)
#   SKIP_HEAVY=1 ./run_overnight.sh # skip demo 02 (49-PC sweep + heatmap grid)
#
# Demo 01 sweeps the 9-point SNR grid, demo 02 runs the four Fig.-3 sweeps and
# the 99-cell heatmap grid, at num_MC=200 with 8 PSOCK workers each. Both
# demos are resumable (finished cells are cached). Figures are regenerated
# separately via each demo's generate_plots.sh.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/../../_overnight_lib.sh"

export HEAVY_DEMOS="demo_trex_spca_02_mc_sim_pev"

run_suite "$here" \
  demo_trex_spca_01_mc_sim \
  demo_trex_spca_02_mc_sim_pev
