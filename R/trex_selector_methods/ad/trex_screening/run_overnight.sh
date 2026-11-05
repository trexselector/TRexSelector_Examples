#!/usr/bin/env bash
#
# Overnight: (re)generate the Screen-TRex suite's result data.
#
# Command:
#   ./run_overnight.sh              # run everything (default)
#   SKIP_HEAVY=1 ./run_overnight.sh # skip the time-heavy biobank demos
#
# HEAVY: the two biobank demos each run a single-phenotype MC sweep plus a
# q=5 multi-phenotype MC sweep at num_MC=200 with R_boot=1000 bootstrap
# resamples (time, not RAM).
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
