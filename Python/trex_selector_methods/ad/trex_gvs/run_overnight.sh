#!/usr/bin/env bash
#
# Overnight: (re)generate the T-Rex GVS suite's mc_sim result data, then render
# correlation heatmaps.
#
#   ./run_overnight.sh              # run everything (default)
#   SKIP_HEAVY=1 ./run_overnight.sh # skip the heavy demos
#
# HEAVY: demo_trex_gvs_08 is the block benchmark (largest sweep in the suite).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/../../_overnight_lib.sh"

export HEAVY_DEMOS="demo_trex_gvs_08_mc_sim_block_bench"

run_suite "$here" \
  demo_trex_gvs_01_mc_sim_hastie_en_blocks \
  demo_trex_gvs_02_mc_sim_scattered_blocks \
  demo_trex_gvs_03_mc_sim_mixed_blocks \
  demo_trex_gvs_04_mc_sim_neg_traps \
  demo_trex_gvs_05_mc_sim_t3_blocks \
  demo_trex_gvs_06_mc_sim_ar1_blocks \
  demo_trex_gvs_07_mc_sim_arma_blocks \
  demo_trex_gvs_08_mc_sim_block_bench
