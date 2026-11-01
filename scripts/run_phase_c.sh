#!/usr/bin/env bash
# ============================================================================
# Phase C driver (TASK 45 Part B) -- RUN 2026-10-10 at HEAD 3f0b382 (logs/phase_c/).
# Groups 0-3 ran (about 1,409 solves). Group 1 hit a hash collision: see the NOTE
# under OUTPUTS and Group 4 below (TASK 48).
# ============================================================================
#
# PURPOSE
#   Re-runs that TASK 44 found missing at the current epoch, as three
#   independently runnable groups:
#     Group 1  (2 solves)  O&M arms: 01b_run_baseline_sliced.py at the
#                          central arm set with --om-case om_low, then om_high.
#                          F-3's O&M-band paragraph (storage fixed while O&M
#                          varies across the ATB band) has no current-epoch run.
#     Group 2  (5 solves)  F-21's decline arm: 02_run_ndc_caps.py
#                          --include-decline-arm. The runner always solves the
#                          four flat caps too (the flag is additive), so the
#                          flat comparator is re-solved in the same invocation
#                          and the decline-vs-flat comparison is within-epoch.
#     Group 3  (~1,405 solves)  the runners that inherit load_scenario's capped
#                          solar default, re-run at --solar-build-case
#                          deployment_unconstrained into <results dir>_uncapped
#                          (TASK 46). The capped results stay where they are, as
#                          the NIRP-trajectory policy arm. Refuses to run until
#                          Group 0 has passed on the CURRENT code.
#     Group 0  (regression gate, ~14 min)  the same runners at their DEFAULT
#                          design into <results dir>_gate, then
#                          scripts/compare_phase_c_gate.py requires byte-identity
#                          against the stored results. Writes a fingerprint of the
#                          touched code to logs/phase_c/.gate_passed on success.
#                          Run with --only 0; it is not part of the default range.
#
# CONVENTIONS (matched from scripts/run_phase_b.sh, read at HEAD, not retyped):
#   - The central C_* arm constants, the NO_RESERVE_MARGIN_RUNNERS array, the
#     per-runner exclusion table excluded_keys_for() and run() are EXTRACTED
#     from run_phase_b.sh at start-up (awk/grep over that file), so this driver
#     can never carry a different central arm string from Phase B. If any block
#     cannot be found the driver aborts.
#   - Only one local addition to the exclusion table: 02_run_ndc_caps.py is not
#     a Phase B runner. Its argparse (read directly) declares every standard
#     flag except --reserve-margin (it hardcodes reserve_margin="nirp_spinning"),
#     and adds --include-decline-arm. Hence the single entry below, plus a
#     literal runtime assert in _execute that neither 01b_ nor 02_run_ndc_caps.py
#     ever receives --reserve-margin.
#   - Logs under logs/phase_c/, header block, free -h before/after, time, tee,
#     abort on non-zero exit. Same --dry-run / --allow-dirty / --only / --from /
#     --to / --resume flags. No deletion, no overwrite beyond what each runner
#     writes under its own name, no commits, no tagging.
#
# OUTPUTS
#   Group 1 was written on the premise that the om arm is part of the hashed arm
#   tag, so the central summary__02c17f55.json would not be touched. THAT PREMISE
#   WAS FALSE at 3f0b382: om_case was not in the tag, om_low and om_high both wrote
#   summary__02c17f55.json, the om_low result was overwritten, and the file now
#   holds om_high. Fixed in 5aad6ff (non-central om_case is now in the tag, so
#   om_low/om_high get their own hashes and om_central keeps 02c17f55). Group 4
#   below re-solves the three O&M arms at the fixed HEAD.
#   Group 2 rewrites results/ndc/ndc{2,3}_{un,}conditional_<hash8>/ -- the same
#   names as the current-epoch flat results, and ndc3_conditional_decline_2cfa636a,
#   which holds the stored 2026-09-14 result. Before running, the script copies
#   that directory to results/_pre_phase_c_<UTC>/ so the old value survives.
#
# USAGE
#   bash scripts/run_phase_c.sh --dry-run             # print the plan, run nothing
#   bash scripts/run_phase_c.sh --only 1              # one group
#   bash scripts/run_phase_c.sh --from 1 --to 2
#   bash scripts/run_phase_c.sh --resume
#   bash scripts/run_phase_c.sh --allow-dirty ...
# ============================================================================

set -u
cd "$(dirname "$0")/.."

PHASE_B="scripts/run_phase_b.sh"
[ -f "$PHASE_B" ] || { echo "ABORT: $PHASE_B not found."; exit 1; }

mkdir -p logs/phase_c
CHECKPOINT="logs/phase_c/.checkpoint"
MAX_GROUP=3

DRY_RUN=0
ALLOW_DIRTY=0
RESUME=0
ONLY=""
FROM=""
TO=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)     DRY_RUN=1; shift ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    --resume)      RESUME=1; shift ;;
    --only)        ONLY="${2:?--only needs a group number}"; shift 2 ;;
    --from)        FROM="${2:?--from needs a group number}"; shift 2 ;;
    --to)          TO="${2:?--to needs a group number}"; shift 2 ;;
    *) echo "ABORT: unknown flag: $1"; exit 1 ;;
  esac
done

if [ -n "$ONLY" ] && { [ -n "$FROM" ] || [ -n "$TO" ]; }; then
  echo "ABORT: --only cannot be combined with --from/--to."; exit 1
fi
if [ "$RESUME" -eq 1 ] && [ -n "$FROM" ]; then
  echo "ABORT: --resume cannot be combined with --from (ambiguous start point)."; exit 1
fi

START=${FROM:-1}
END=${TO:-$MAX_GROUP}
if [ -n "$ONLY" ]; then START=$ONLY; END=$ONLY; fi
if [ "$RESUME" -eq 1 ] && [ -f "$CHECKPOINT" ]; then
  START=$(( $(cat "$CHECKPOINT") + 1 ))
  echo "Resuming from group $START (checkpoint: $(cat "$CHECKPOINT"))."
fi

# ---- header block -------------------------------------------------------
UTC_NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
BRANCH=$(git branch --show-current)
HEAD_SHA=$(git rev-parse HEAD)
DESCRIBE=$(git describe --tags 2>/dev/null || echo "(no tag reachable)")
DIRTY_STATUS=$(git status --porcelain)

HEADER_TEXT=$(cat <<EOF
================ Phase C driver run header ================
UTC timestamp : $UTC_NOW
Branch        : $BRANCH
HEAD          : $HEAD_SHA
git describe  : $DESCRIBE
Working tree  : $([ -z "$DIRTY_STATUS" ] && echo "clean" || echo "DIRTY")
==============================================================
EOF
)
echo "$HEADER_TEXT"

if [ -n "$DIRTY_STATUS" ] && [ "$DRY_RUN" -eq 0 ]; then
  if [ "$ALLOW_DIRTY" -eq 0 ]; then
    echo
    echo "!! ABORT: working tree is DIRTY. Phase C results must be attributable"
    echo "!! to a committed tree. Commit or stash, or pass --allow-dirty."
    echo
    git status --short
    exit 1
  fi
  echo "WARNING: working tree is dirty; proceeding because --allow-dirty was passed."
fi

# ---- take the central arm string, the exclusion table and run() from Phase B
# _extract <start-regex> <end-regex>: print the first block of $PHASE_B that
# starts at a line matching the first regex and ends at the first following
# line matching the second (inclusive). Empty output is a hard error.
_extract () {
  local out
  out=$(awk -v s="$1" -v e="$2" '
    !on && $0 ~ s { on=1 }
    on { print }
    on && $0 ~ e { exit }
  ' "$PHASE_B")
  [ -n "$out" ] || { echo "ABORT: could not extract '$1' from $PHASE_B." >&2; return 1; }
  printf '%s\n' "$out"
}

# central constants: every ^C_NAME="value" line, verbatim
C_LINES=$(grep -E '^C_[A-Z_]+="[^"]*"$' "$PHASE_B")
[ -n "$C_LINES" ] || { echo "ABORT: no C_* central constants found in $PHASE_B."; exit 1; }
eval "$C_LINES"

ARR_B=$(_extract '^readonly -a NO_RESERVE_MARGIN_RUNNERS=[(]' '^[)]$') || exit 1
eval "$ARR_B"

EXCL_B=$(_extract '^excluded_keys_for [(][)] [{]' '^[}]$') || exit 1
eval "${EXCL_B/excluded_keys_for ()/_excluded_keys_for_b ()}"

RUN_B=$(_extract '^run [(][)] [{]' '^[}]$') || exit 1
eval "$RUN_B"

# every C_* variable run() dereferences must exist (derived from run()'s own text)
for v in $(printf '%s\n' "$RUN_B" | grep -o '\$C_[A-Z_]*' | sort -u | tr -d '$'); do
  [ -n "${!v:-}" ] || { echo "ABORT: run() uses \$$v but $PHASE_B did not define it."; exit 1; }
done

# The central output hash, derived from the SAME C_* values by 01b's own --print-hash8 (no build, no solve),
# as run_phase_b.sh's central_diag_hash8 does. Never a pasted constant (TASK 53 2.5 moved it 02c17f55 -> 6046f759).
central_h8 () {
  python3 scripts/01b_run_baseline_sliced.py \
    --peak-case "$C_PEAK" --storage-cc-case "$C_STORAGE_CC" --gas-build-case "$C_GAS_BUILD" \
    --new-gas-deliverability "$C_NEW_GAS_DELIV" --gas-availability-case "$C_GAS_AVAIL" \
    --gas-retirement-case "$C_GAS_RETIRE" --gas-capex-case "$C_GAS_CAPEX" \
    --ccgt-efficiency-case "$C_CCGT" --capital-recovery-case "$C_CR" --dscr-case "$C_DSCR" \
    --us-inflation "$C_INFL" --capital-treatment-case "$C_TREAT" \
    --concessional-envelope-case "$C_ENV" --gas-deliverability-case "$C_GAS_DELIV" \
    --conversion-case "$C_CONV" --hydro-case "$C_HYDRO" --tariff-case "$C_TARIFF" \
    --om-case "$C_OM" --print-hash8 | tail -1
}

# Local exclusion entry for the one non-Phase-B runner (argparse read directly,
# 2026-10-10: every standard flag except --reserve-margin, plus --include-decline-arm).
excluded_keys_for () {
  case "$1" in
    # 02_ and 03_: argparse read directly (TASK 45/46): every standard flag except
    # --reserve-margin (hardcoded "nirp_spinning"); 02_ adds --include-decline-arm.
    02_run_ndc_caps.py|03_run_ndc_eaas.py) echo " reserve_margin " ;;
    # 26_run_f3_identification.py: argparse read directly (TASK 46): every
    # standard flag except --reserve-margin (hardcoded) and --om-case (hardcoded
    # om_central); adds --capital-case and --tol (both required).
    26_run_f3_identification.py) echo " reserve_margin om_case " ;;
    *) _excluded_keys_for_b "$1" ;;
  esac
}

# ---- execution primitive (Phase B's, with logs/phase_c and a literal assert)
_execute () {
  local log_label="$1"; shift
  local log="logs/phase_c/${log_label}.log"

  # literal runtime assert: these runners hardcode reserve_margin internally
  case " $* " in
    *" scripts/01b_run_baseline_sliced.py "*|*" scripts/02_run_ndc_caps.py "*|*" scripts/03_run_ndc_eaas.py "*|*" scripts/26_run_f3_identification.py "*)
      for a in "$@"; do
        if [ "$a" = "--reserve-margin" ]; then
          echo "ABORT: $* must never receive --reserve-margin (hardcodes nirp_spinning internally)."
          exit 1
        fi
      done ;;
  esac

  # capture mode (name resolution only): run a no-solve, no-write query such as
  # 02_run_ndc_caps.py --print-case-dirs and keep its stdout, even in a dry run.
  if [ -n "${CAPTURE_TO:-}" ]; then
    "$@" > "$CAPTURE_TO"
    return $?
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    printf '  [%s] ' "$log_label"
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi

  {
    echo "$HEADER_TEXT"
    echo "item log    : $log_label"
    echo
    echo "=== free -h (before) ==="
    free -h
    echo
    echo "=== invocation ==="
    printf '%q ' "$@"; printf '\n'
    echo
  } > "$log"

  { time "$@"; } 2>&1 | tee -a "$log"
  local code=${PIPESTATUS[0]}

  {
    echo
    echo "=== free -h (after) ==="
    free -h
    echo
    echo "=== exit status: $code ==="
  } >> "$log"

  if [ "$code" -ne 0 ]; then
    echo
    echo "ABORT: [$log_label] exited $code. See $log."
    echo "Phase C stops here -- no continuing past a failure."
    exit 1
  fi
  printf '  [%s] ok -> %s\n' "$log_label" "$log"
}

banner () { echo; echo "════════════════════════════════════════════════════"; echo "$1"; echo "════════════════════════════════════════════════════"; }
should_run () { local n="$1"; [ "$n" -ge "$START" ] && [ "$n" -le "$END" ]; }
checkpoint () { [ "$DRY_RUN" -eq 1 ] || echo "$1" > "$CHECKPOINT"; }

# ============================================================================
# GROUPS
# ============================================================================

group_1 () {
  banner "GROUP 1 -- 01b_run_baseline_sliced.py x2 (2 solves: om_low, om_high at the central arm set)"
  run 01b_run_baseline_sliced.py 1_A_om_low  om=om_low
  run 01b_run_baseline_sliced.py 1_B_om_high om=om_high
  if [ "$DRY_RUN" -eq 0 ]; then
    local h8; h8=$(central_h8)
    echo "CHECK: compare with results/baseline_sliced/summary__${h8}.json (central, om_central; post-epoch anchor"
    echo "       24,986,620,949.18 USD at TASK 53 -- hash 6046f759 -- 2054 storage in storage_capacity_2054_mwh)."
    python3 - "$h8" <<'PY'
import json, sys
f = f"results/baseline_sliced/summary__{sys.argv[1]}.json"
try:
    d = json.load(open(f))
    print(f"       on disk now: npv_total_cost_usd {d['npv_total_cost_usd']:,.2f}, "
          f"storage_capacity_2054_mwh {d['decision_variables']['storage_capacity_2054_mwh']:,.5f}")
except Exception as e:
    print(f"       WARNING: could not read {f}: {e}")
PY
    echo "       F-3 predicts a different objective under om_low/om_high and the SAME 2054 storage capacity."
  fi
}

group_2 () {
  banner "GROUP 2 -- 02_run_ndc_caps.py --include-decline-arm (5 solves: 4 flat caps + ndc3_conditional_decline)"
  # Resolve the directory names THIS invocation will write, without a solve (02_ --print-case-dirs).
  # Post-TASK 53 the names carry the completed arm tag, so the pre-epoch name (ndc3_conditional_decline_2cfa636a)
  # is not what gets written. Failing to resolve is fatal: the snapshot below must never be skipped by accident.
  local names="logs/phase_c/.group2_case_dirs.txt"
  CAPTURE_TO="$names"
  run 02_run_ndc_caps.py 2_resolve_case_dirs -- --include-decline-arm --print-case-dirs
  local rc=$?
  CAPTURE_TO=""
  [ "$rc" -eq 0 ] || { echo "ABORT: Group 2 could not resolve its case directory names (02_ --print-case-dirs exited $rc)."; exit 1; }
  local case dir n=0
  for case in ndc2_unconditional ndc2_conditional ndc3_unconditional ndc3_conditional ndc3_conditional_decline; do
    dir=$(awk -v c="$case" '$1 == c { print $2 }' "$names")
    [ -n "$dir" ] || { echo "ABORT: Group 2 name resolution produced no directory for $case ($names)."; exit 1; }
    n=$((n+1))
  done
  [ "$n" -eq 5 ] || { echo "ABORT: expected 5 resolved case directories, got $n."; exit 1; }
  local D F
  D=$(awk '$1 == "ndc3_conditional_decline" { print $2 }' "$names")
  F=$(awk '$1 == "ndc3_conditional" { print $2 }' "$names")
  echo "  resolved: decline arm -> results/ndc/$D ; flat comparator -> results/ndc/$F"

  # Snapshot any of the five that already exists (it would be overwritten); say so when none does.
  local snap="results/_pre_phase_c_${UTC_NOW//[:-]/}"
  local existing=0
  for case in ndc2_unconditional ndc2_conditional ndc3_unconditional ndc3_conditional ndc3_conditional_decline; do
    dir=$(awk -v c="$case" '$1 == c { print $2 }' "$names")
    if [ -d "results/ndc/$dir" ]; then
      existing=$((existing+1))
      if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [snapshot] would copy results/ndc/$dir -> $snap/ndc/ (it exists and would be overwritten)"
      else
        mkdir -p "$snap/ndc" && cp -a "results/ndc/$dir" "$snap/ndc/" \
          || { echo "ABORT: could not preserve results/ndc/$dir in $snap/ndc/."; exit 1; }
        echo "  preserved results/ndc/$dir in $snap/ndc/"
      fi
    else
      echo "  results/ndc/$dir does not exist yet: new epoch name, nothing to preserve"
    fi
  done
  echo "  $existing of 5 target directories already existed."

  run 02_run_ndc_caps.py 2_ndc_caps_with_decline -- --include-decline-arm
  if [ "$DRY_RUN" -eq 0 ]; then
    [ -d "results/ndc/$D" ] && [ -d "results/ndc/$F" ] \
      || { echo "ABORT: expected outputs results/ndc/$D and results/ndc/$F are missing after the run."; exit 1; }
    echo "CHECK: results/ndc/$D vs results/ndc/$F"
    echo "       (same invocation, same epoch; capped solar design, as 02_run_ndc_caps.py passes no"
    echo "       solar_build_case). The four flat results should reproduce the Phase B files for the same arms."
  fi
}

# ---- Group 4: the three O&M arms re-solved after the hash fix (TASK 48 B.2) ----
group_4 () {
  banner "GROUP 4 -- 01b_run_baseline_sliced.py x3 (om_central, om_low, om_high at the central arm set)"
  run 01b_run_baseline_sliced.py 4_A_om_central om=om_central
  run 01b_run_baseline_sliced.py 4_B_om_low     om=om_low
  run 01b_run_baseline_sliced.py 4_C_om_high    om=om_high
  if [ "$DRY_RUN" -eq 0 ]; then
    echo "CHECK: python3 scripts/check_om_arms.py   (three distinct files, three distinct om_case values)"
  fi
}

# ---- Group 5: the om_none arm (TASK 48 B.2 addendum; ONE solve, separate from Group 4) ----
# TASK 47 C14 and C22 (half 2) need a current-epoch om_none run; Group 4's three arms
# (central, om_low, om_high) do not supply it. Kept out of Group 4 so that Group 4 stays
# the three solves TASK 48 specified.
group_5 () {
  banner "GROUP 5 -- 01b_run_baseline_sliced.py x1 (om_none at the central arm set)"
  run 01b_run_baseline_sliced.py 5_A_om_none om=om_none
  if [ "$DRY_RUN" -eq 0 ]; then
    echo "CHECK: python3 scripts/check_om_arms.py --with-none"
  fi
}

# ---- Groups 0 and 3 share this list of touched code ------------------------
GATE_CODE_FILES="src/cli_arms.py src/optimize_experiments.py \
  scripts/02_run_ndc_caps.py scripts/03_run_ndc_eaas.py scripts/04_run_fin2_tariff_sweep.py \
  scripts/05_run_fin3_capital_sweep.py scripts/06_run_gas1_shadow_benchmarks.py \
  scripts/07_run_gas2_eaas_gas_relief.py scripts/08_run_gas3_regime_ndc_feasibility.py \
  scripts/10_run_rel2_marginal_cost.py scripts/11_run_rel3_financing_frontier.py \
  scripts/26_run_f3_identification.py scripts/compare_phase_c_gate.py"
gate_fingerprint () { cat $GATE_CODE_FILES | sha256sum | cut -d' ' -f1; }
GATE_MARKER="logs/phase_c/.gate_passed"

# The ten runners, once per design. $1 = label prefix, then the extra flags that
# select the design/output directory ("$@" after shift), identical for every runner.
_ndc_runners () {
  local pre="$1"; shift
  run 02_run_ndc_caps.py                 "${pre}_02_ndc_caps"        -- "$@"
  run 03_run_ndc_eaas.py                 "${pre}_03_ndc_eaas"        -- "$@"
  run 04_run_fin2_tariff_sweep.py        "${pre}_04_fin2"            -- "$@"
  run 05_run_fin3_capital_sweep.py       "${pre}_05_fin3"            -- "$@"
  run 06_run_gas1_shadow_benchmarks.py   "${pre}_06_gas1"            -- "$@"
  run 07_run_gas2_eaas_gas_relief.py     "${pre}_07_gas2"            -- "$@"
  run 08_run_gas3_regime_ndc_feasibility.py "${pre}_08_gas3"         -- "$@"
  run 10_run_rel2_marginal_cost.py       "${pre}_10_rel2_nameplate"  gas_avail=gas_avail_nameplate -- "$@"
  run 10_run_rel2_marginal_cost.py       "${pre}_10_rel2_observed"   gas_avail=gas_avail_observed  -- "$@"
  run 11_run_rel3_financing_frontier.py  "${pre}_11_rel3_nameplate"  gas_avail=gas_avail_nameplate -- "$@"
  run 11_run_rel3_financing_frontier.py  "${pre}_11_rel3_observed"   gas_avail=gas_avail_observed  -- "$@"
  # 26_ (F-12/F-13 identification): the seven configurations behind results/f3
  run 26_run_f3_identification.py "${pre}_26_f3_moderate"      -- --capital-case moderate      --tol 1e-6 "$@"
  run 26_run_f3_identification.py "${pre}_26_f3_tight"         -- --capital-case tight         --tol 1e-6 "$@"
  run 26_run_f3_identification.py "${pre}_26_f3_adequacy"      -- --capital-case adequacy      --tol 1e-6 "$@"
  run 26_run_f3_identification.py "${pre}_26_f3_expansion"     -- --capital-case expansion     --tol 1e-6 "$@"
  run 26_run_f3_identification.py "${pre}_26_f3_unconstrained" -- --capital-case unconstrained --tol 1e-6 "$@"
  run 26_run_f3_identification.py "${pre}_26_f3_tariff_low"  tariff=tariff_low  -- --capital-case moderate --tol 1e-6 "$@"
  run 26_run_f3_identification.py "${pre}_26_f3_tariff_high" tariff=tariff_high -- --capital-case moderate --tol 1e-6 "$@"
}

group_0 () {
  banner "GROUP 0 -- regression gate: one configuration per touched runner at the DEFAULT design into *_gate (841 solves, about 14 min); compared with Phase B's results for the same arms"
  local g="--results-suffix _gate"
  run 02_run_ndc_caps.py                    0_02_ndc_caps        -- $g
  run 03_run_ndc_eaas.py                    0_03_ndc_eaas        -- $g
  run 04_run_fin2_tariff_sweep.py           0_04_fin2            -- $g
  run 05_run_fin3_capital_sweep.py          0_05_fin3            -- $g
  run 06_run_gas1_shadow_benchmarks.py      0_06_gas1            -- $g
  run 07_run_gas2_eaas_gas_relief.py        0_07_gas2            -- $g
  run 08_run_gas3_regime_ndc_feasibility.py 0_08_gas3            -- $g
  run 10_run_rel2_marginal_cost.py          0_10_rel2_nameplate  gas_avail=gas_avail_nameplate -- $g
  run 11_run_rel3_financing_frontier.py     0_11_rel3_nameplate  gas_avail=gas_avail_nameplate -- $g
  run 26_run_f3_identification.py           0_26_f3_moderate     -- --capital-case moderate --tol 1e-6 $g
  run 26_run_f3_identification.py           0_26_f3_tight        -- --capital-case tight    --tol 1e-6 $g
  if [ "$DRY_RUN" -eq 0 ]; then
    # Post-epoch gate (TASK 54 follow-up): each *_gate run is compared with Phase B's result for the SAME
    # arm set (arms_<hash8>; hash-named entries for 02_/03_/26_), not with the pre-epoch stored files. A gate
    # arm set Phase B did not produce FAILS ("no reference"). Run Phase B first.
    if python3 scripts/compare_phase_c_gate.py; then
      gate_fingerprint > "$GATE_MARKER"
      echo "GATE PASSED on the current code; fingerprint written to $GATE_MARKER."
    else
      rm -f "$GATE_MARKER"
      echo "GATE NOT PASSED: Group 3 will refuse to run. Read the differences above."
      exit 1
    fi
  fi
}

group_3 () {
  banner "GROUP 3 -- the same runners at deployment_unconstrained into *_uncapped (about 1,400 solves)"
  if [ "$DRY_RUN" -eq 0 ]; then
    if [ ! -f "$GATE_MARKER" ] || [ "$(cat "$GATE_MARKER")" != "$(gate_fingerprint)" ]; then
      echo "ABORT: no passing Group 0 gate for the current code ($GATE_MARKER missing or stale)."
      echo "Run: bash scripts/run_phase_c.sh --only 0"
      exit 1
    fi
  fi
  _ndc_runners 3 --solar-build-case deployment_unconstrained --results-suffix _uncapped
  # F-21's decline arm under the uncapped design too (5 solves; the four flat caps are re-solved)
  run 02_run_ndc_caps.py 3_02_ndc_caps_decline -- --include-decline-arm \
    --solar-build-case deployment_unconstrained --results-suffix _uncapped
}

# ============================================================================
# DISPATCH
# ============================================================================

echo "Phase C driver: groups $START..$END$( [ "$DRY_RUN" -eq 1 ] && echo " (DRY RUN -- nothing will execute)" )"

# Groups 4 and 5 are outside the default range (END defaults to MAX_GROUP=3): --only 4 / --only 5.
for n in $(seq 0 5); do
  should_run "$n" || continue
  "group_$n"
  [ "$n" -le "$MAX_GROUP" ] && checkpoint "$n"
done

banner "PHASE C DRIVER: requested range complete ($START..$END)"
echo "Logs in logs/phase_c/. Checkpoint: $( [ -f "$CHECKPOINT" ] && cat "$CHECKPOINT" || echo "(none written -- dry run)" )"
