#!/usr/bin/env bash
#
# Overnight: (re)generate the DA-TRex suite's mc_sim result data, then render
# correlation heatmaps.
#
#   ./run_overnight.sh              # run everything (default)
#   SKIP_HEAVY=1 ./run_overnight.sh # skip the heavy demos
#
# HEAVY: the heavy-tailed block demos (04, 05) are the longest / most memory-
# intensive in the suite — each sweeps 5-6 parts over 2 noise scenarios x 3
# linkages x 3 solvers with t(nu=3) sampling.
# SKIP_HEAVY=1 omits them.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/../../_overnight_lib.sh"

export HEAVY_DEMOS="demo_trex_da_04_mc_sim_ht_blocks demo_trex_da_05_mc_sim_ht_blocks_plus_white"

run_suite "$here" \
  demo_trex_da_01_mc_sim_ar1 \
  demo_trex_da_02_mc_sim_ar1_blocks_plus_white \
  demo_trex_da_03_mc_sim_ar1_blocks \
  demo_trex_da_04_mc_sim_ht_blocks \
  demo_trex_da_05_mc_sim_ht_blocks_plus_white \
  demo_trex_da_06_mc_sim_groups \
  demo_trex_da_07_mc_sim_nn \
  demo_trex_da_08_mc_sim_equi_and_bt
