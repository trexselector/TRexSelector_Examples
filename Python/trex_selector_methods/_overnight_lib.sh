#!/usr/bin/env bash
# ==============================================================================
# _overnight_lib.sh  —  shared runner for the per-suite overnight scripts.
# ==============================================================================
#
# Sourced by each suite's run_overnight.sh. Runs a list of demos end-to-end to
# (re)generate their simulation_results/ data, then renders any correlation
# heatmaps present in the suite. Designed for unattended overnight use:
#
#   * continue-on-error — one failing demo never aborts the batch;
#   * per-demo timestamped logs + a run summary under <suite>/_overnight_logs/;
#   * SKIP_HEAVY=1 skips the demos listed in HEAVY_DEMOS (RAM/time hogs);
#   * no leaked workers — each demo runs in its own process group, which is
#     reaped when the demo exits (see _reap_group) and on runner INT/TERM/EXIT;
#   * resumable — see "Run tag and resume" below.
#
# Run tag and resume
# ------------------
# TREX_RUN_TAG names the library state the results come from: the pinned
# commit of LIBRARY_TARGET.txt unless the caller sets it. A demo that exits 0
# is recorded in <suite>/_overnight_logs/done_<tag>.txt, and later runs with
# the same tag skip it, so a stopped batch restarts at the demo it was in. The
# demos cache their finished grid cells under the same tag, so the demo in
# flight continues where it stopped too. FRESH=1 ignores the record and makes
# the demos set their caches aside, which reruns everything.
#
# Worker reaping
# --------------
# The demos parallelise with ProcessPoolExecutor. A clean exit shuts the pool
# down, but a hard kill / crash / OOM of the demo leaves its spawned workers
# behind: reparented to init (PPID 1), idle, each still holding its share of
# the interpreter + design matrices (~1 GB for a 6-worker pool was observed).
#
# Each demo is therefore started as a background job with job control enabled
# (set -m), which puts it in a NEW process group with PGID == the job's PID.
# After the demo returns, anything left in that group is a stray worker and
# gets TERM, then KILL. A process group survives reparenting to init, so this
# catches orphans that PPID-based cleanup cannot.
#
# This is deliberately not tied to the controlling terminal: no `jobs`, no
# SIGHUP, no tty. It works identically under nohup, disown, a detached
# ssh session, cron, or a CI runner. (macOS has no setsid(1), hence set -m.)
#
# Not sourced by anything type-checked; pure bash — kept 3.2-compatible so it
# runs under the stock /bin/bash on macOS.
# ==============================================================================

_find_venv() {
  local d="$1"
  while [[ "$d" != "/" ]]; do
    if [[ -x "$d/.venv/bin/python" ]]; then echo "$d/.venv/bin/python"; return 0; fi
    d="$(dirname "$d")"
  done
  return 1
}

# _run_tag <dir>
# TREX_RUN_TAG if set, else the pinned commit of the nearest LIBRARY_TARGET.txt
# above <dir>, else "untagged".
_run_tag() {
  if [[ -n "${TREX_RUN_TAG:-}" ]]; then echo "$TREX_RUN_TAG"; return 0; fi
  local d="$1"
  while [[ "$d" != "/" ]]; do
    if [[ -f "$d/LIBRARY_TARGET.txt" ]]; then
      awk '/^Pinned commit:/ { print $3; exit }' "$d/LIBRARY_TARGET.txt"
      return 0
    fi
    d="$(dirname "$d")"
  done
  echo "untagged"
}

# _reap_group <pgid>
# Kill every process remaining in process group <pgid>: TERM, up to 5s grace,
# then KILL. Silent no-op when the group is already empty. Echoes a one-line
# note only when it actually had to reap something, so a clean run stays quiet.
#
# Refuses to touch its own process group — `kill -- -$$` would otherwise take
# the runner (and the user's shell) down with it.
_reap_group() {
  local pgid="$1"
  [[ -z "$pgid" ]] && return 0

  local self_pgid; self_pgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"
  if [[ -z "$pgid" || "$pgid" == "$self_pgid" || "$pgid" == "0" || "$pgid" == "1" ]]; then
    return 0
  fi

  # kill -0 on a negative pid succeeds while ANY process in the group lives.
  kill -0 -- "-$pgid" 2>/dev/null || return 0

  local n; n="$(pgrep -g "$pgid" 2>/dev/null | wc -l | tr -d ' ')"
  kill -TERM -- "-$pgid" 2>/dev/null

  local i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 -- "-$pgid" 2>/dev/null || break
    sleep 0.5
  done

  if kill -0 -- "-$pgid" 2>/dev/null; then
    kill -KILL -- "-$pgid" 2>/dev/null
    sleep 0.2
  fi
  echo "[reap ] pgid $pgid — cleaned up $n stray process(es)"
  return 0
}

# Set while a demo is in flight so the signal/EXIT trap knows what to reap.
_CURRENT_PGID=""

_overnight_trap() {
  local sig="$1"
  if [[ -n "$_CURRENT_PGID" ]]; then
    echo "" >&2
    echo "[$sig ] runner interrupted — reaping demo process group $_CURRENT_PGID" >&2
    _reap_group "$_CURRENT_PGID" >&2
    _CURRENT_PGID=""
  fi
  if [[ "$sig" != "EXIT" ]]; then
    trap - EXIT
    exit 130
  fi
}

# run_suite <suite_dir> <demo_name> [<demo_name> ...]
# Each <demo_name> is a subdir containing <demo_name>.py.
run_suite() {
  local suite_dir="$1"; shift
  local suite_name; suite_name="$(basename "$suite_dir")"

  local venv; venv="$(_find_venv "$suite_dir")" || {
    echo "error: no .venv/bin/python found above $suite_dir" >&2
    echo "       create it and install the demo deps, then re-run." >&2
    return 1
  }

  # Reap the in-flight demo's workers if the runner itself is interrupted or
  # exits early. Installed here so a sourcing script gets it for free.
  trap '_overnight_trap INT'  INT
  trap '_overnight_trap TERM' TERM
  trap '_overnight_trap HUP'  HUP
  trap '_overnight_trap EXIT' EXIT

  local logdir="$suite_dir/_overnight_logs"; mkdir -p "$logdir"
  local stamp; stamp="$(date +%Y%m%d_%H%M%S)"
  local summary="$logdir/summary_${stamp}.txt"
  local skip_heavy="${SKIP_HEAVY:-0}"

  # The demos read the tag (cell-cache fingerprint) and FRESH (set caches aside).
  TREX_RUN_TAG="$(_run_tag "$suite_dir")"; export TREX_RUN_TAG
  export TREX_FRESH="${FRESH:-0}"
  export TREX_NUM_WORKERS="${TREX_NUM_WORKERS:-8}"   # worker processes per demo (0 = all CPUs)
  local done_file="$logdir/done_${TREX_RUN_TAG}.txt"

  {
    echo "=============================================================="
    echo " overnight run: $suite_name"
    echo " started : $(date)"
    echo " venv    : $venv"
    echo " run tag : $TREX_RUN_TAG   FRESH=$TREX_FRESH   (record: $(basename "$done_file"))"
    echo " SKIP_HEAVY=$skip_heavy   HEAVY_DEMOS='${HEAVY_DEMOS:-}'"
    echo "=============================================================="
    echo
  } | tee "$summary"

  local name dir log t0 rc demo_pid corr_pid ok=0 fail=0 skip=0 done_n=0
  for name in "$@"; do
    dir="$suite_dir/$name"
    if [[ ! -f "$dir/$name.py" ]]; then
      echo "[MISS ] $name  (no $name.py)" | tee -a "$summary"; continue
    fi
    if [[ "$TREX_FRESH" != "1" ]] && grep -qx "$name" "$done_file" 2>/dev/null; then
      echo "[DONE ] $name  (finished earlier on $TREX_RUN_TAG, FRESH=1 reruns it)" | tee -a "$summary"
      done_n=$((done_n + 1)); continue
    fi
    if [[ "$skip_heavy" == "1" && " ${HEAVY_DEMOS:-} " == *" $name "* ]]; then
      echo "[SKIP ] $name  (HEAVY, SKIP_HEAVY=1)" | tee -a "$summary"
      skip=$((skip + 1)); continue
    fi
    log="$logdir/${name}_${stamp}.log"
    echo "[START] $name  $(date +%H:%M:%S)" | tee -a "$summary"
    t0=$SECONDS

    # set -m gives the background job its own process group (PGID == its PID),
    # so the reap below can never reach the runner or the user's shell. `exec`
    # keeps the PID, making the python process itself the group leader.
    set -m
    ( cd "$dir" && exec "$venv" "$dir/$name.py" ) >"$log" 2>&1 &
    demo_pid=$!
    set +m
    _CURRENT_PGID="$demo_pid"
    wait "$demo_pid"; rc=$?
    _reap_group "$demo_pid" | tee -a "$summary"
    _CURRENT_PGID=""

    if [[ $rc -eq 0 ]]; then
      echo "[ OK  ] $name  $((SECONDS - t0))s  -> $(basename "$log")" | tee -a "$summary"
      grep -qx "$name" "$done_file" 2>/dev/null || echo "$name" >> "$done_file"
      ok=$((ok + 1))
    else
      echo "[FAIL ] $name  $((SECONDS - t0))s  rc=$rc  -> $(basename "$log")" | tee -a "$summary"
      tail -n 4 "$log" 2>/dev/null | sed 's/^/         | /' | tee -a "$summary"
      fail=$((fail + 1))
    fi
  done

  echo | tee -a "$summary"
  echo "--- design-correlation heatmaps (one per DGP) ---" | tee -a "$summary"
  local corr_sh="$suite_dir/generate_corr_plots.sh"
  if [[ -f "$corr_sh" ]]; then
    # Same own-process-group treatment: the plotters import the extension too.
    set -m
    bash "$corr_sh" >>"$summary" 2>&1 &
    corr_pid=$!
    set +m
    _CURRENT_PGID="$corr_pid"
    wait "$corr_pid"; rc=$?
    _reap_group "$corr_pid" | tee -a "$summary"
    _CURRENT_PGID=""
    if [[ $rc -eq 0 ]]; then
      echo "[corr ] $(basename "$suite_dir") DGP heatmaps rendered" | tee -a "$summary"
    else
      echo "[corr!] $(basename "$suite_dir") (failed — see summary)" | tee -a "$summary"
    fi
  else
    echo "(no generate_corr_plots.sh in this suite yet)" | tee -a "$summary"
  fi

  {
    echo
    echo "=== $suite_name done: $ok ok, $fail failed, $skip skipped, $done_n done earlier ==="
    echo " finished: $(date)"
    echo " summary : $summary"
  } | tee -a "$summary"

  # Nothing left in flight — drop the traps so a later exit stays silent.
  trap - INT TERM HUP EXIT
  return 0
}
