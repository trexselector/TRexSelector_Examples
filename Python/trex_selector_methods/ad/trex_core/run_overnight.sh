#!/usr/bin/env bash
#
# Overnight: (re)generate the classical T-Rex suite's result data, then render the
# design-correlation heatmap.
#
# Command:
#   ./run_overnight.sh              # run everything (default)
#   SKIP_HEAVY=1 ./run_overnight.sh # skip the RAM/time-heavy demos
#
# demo_trex_08_mc_sim_scalability is intentionally NOT run here: it is a heavy
# out-of-core benchmark with its own num_MC CLI + per-trial cache (running it at
# the default num_MC=10 over the full n x p grid OOMs).
#
# Run it separately, e.g.:
#   demo_trex_08_mc_sim_scalability/demo_trex_08_mc_sim_scalability.py 2
#
# HEAVY: demo_trex_07 is the num_MC=200 mmap MC sweep and demo_trex_10 the
# n / (s ln p) sweep up to p = 10,000 (time, not RAM).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/../../_overnight_lib.sh"

export HEAVY_DEMOS="demo_trex_07_mc_sim_mmap demo_trex_10_mc_sim_dt_ratio"

run_suite "$here" \
  demo_trex_01_single_run \
  demo_trex_02_mc_sim_fixed_support \
  demo_trex_03_mc_sim_variable_support \
  demo_trex_04_mc_sim_lloop_strategies \
  demo_trex_05_mc_sim_dummy_distributions \
  demo_trex_06_mmap \
  demo_trex_07_mc_sim_mmap \
  demo_trex_09_mc_sim_target_fdr \
  demo_trex_10_mc_sim_dt_ratio
