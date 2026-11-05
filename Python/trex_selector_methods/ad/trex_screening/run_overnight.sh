#!/usr/bin/env bash
#
# Overnight: (re)generate the T-Rex screening suite's mc_sim result data, then
# render correlation heatmaps.
#
#   ./run_overnight.sh              # run everything (default)
#   SKIP_HEAVY=1 ./run_overnight.sh # skip the biobank demos (04, 05)
#
# The demos use the cpp-reference Monte Carlo counts (num_MC = 200, biobank
# num_MC_multi = 200, R_boot = 1000), so this run reproduces the C++ suite's
# fidelity. Set SCR_NUM_WORKERS to change the parallel worker-process count
# (default 8); it does not affect results.
#
# HEAVY: demo_trex_scr_04 / _05 are the (synthetic) biobank studies — heaviest.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/../../_overnight_lib.sh"

export HEAVY_DEMOS="demo_trex_scr_04_mc_sim_biobank demo_trex_scr_05_mc_sim_biobank_mmap"

run_suite "$here" \
  demo_trex_scr_01_mc_sim_screen_trex \
  demo_trex_scr_02_mc_sim_screen_trex_mmap \
  demo_trex_scr_03_mc_sim_correlated \
  demo_trex_scr_04_mc_sim_biobank \
  demo_trex_scr_05_mc_sim_biobank_mmap \
  demo_trex_scr_06_mc_sim_solvers
