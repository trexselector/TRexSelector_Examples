#!/usr/bin/env bash
#
# Overnight: (re)generate the DA-TRex suite's result data.
#
# Command:
#   ./run_overnight.sh              # run everything (default)
#   SKIP_HEAVY=1 ./run_overnight.sh # skip the time-heavy HT block sweeps
#
# HEAVY: demos 04 and 05 each run 6 sweeps x 2 scenarios x 3 linkages at
# num_MC=200 on heavy-tailed block designs (time, not RAM).
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
