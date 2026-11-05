#!/usr/bin/env bash
#
# Overnight: (re)generate the T-Rex SPCA suite's mc_sim result data, then render
# correlation heatmaps. (This suite already has a results-plotting layer:
# per-demo generate_plots.sh renders FDR/TPR/PEV figures from the CSVs.)
#
#   ./run_overnight.sh              # run everything (default)
#   SKIP_HEAVY=1 ./run_overnight.sh # skip demo 02 (the heavy PEV sweep)
#
# Both demos now run at full cpp-reference fidelity — the full legacy CRAN SNR
# grid {-10, -7, -5, -3, 0, 3, 5, 7, 10} dB and num_MC = 200, matching the C++
# suite exactly. TREX_SPCA_NUM_MC / TREX_SPCA_PARTS remain available to the
# demos for pilot runs but are left unset here so the overnight reproduces the
# reference in full.
#
# HEAVY: demo_trex_spca_02_mc_sim_pev is the RAM/time hog — Part 1 extracts up
# to 49 PCs per trial and Part 5 is a ~40k-run (tFDR x SNR) x 2-response heatmap.
# SKIP_HEAVY=1 skips it, leaving the (still multi-hour) demo 01 SNR sweep.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/../../_overnight_lib.sh"

export HEAVY_DEMOS="demo_trex_spca_02_mc_sim_pev"

run_suite "$here" \
  demo_trex_spca_01_mc_sim \
  demo_trex_spca_02_mc_sim_pev
