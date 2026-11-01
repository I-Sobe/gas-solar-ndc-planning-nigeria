#!/usr/bin/env bash
# ============================================================================
# Phase B driver
# ============================================================================
#
# PURPOSE
#   Phase B produces the citable result set for Chapters 4 and 5. Driving
#   it from shell variables typed at an interactive prompt is not
#   reproducible: the arm string is not versioned, a typo in one invocation
#   silently changes one runner's scenario, and nothing records which
#   commit produced which result. This script is that committed driver,
#   matching the pattern scripts/run_central_sweep.sh and
#   scripts/run_verification_sweep.sh already establish for a committed,
#   reproducible sweep (TASK 17, Part B).
#
# CONVENTIONS MATCHED FROM scripts/run_central_sweep.sh (read in full
# before writing this file, per TASK 17 B.1):
#   - Central arm values as named C_* shell constants, committed once.
#   - A run()-style function taking a runner, a label, and key=value
#     overrides for the standard arms.
#   - --om-case gated per runner (that script gates it IN because most of
#     its own runners need it and 26_ does not; this script gates it IN
#     for every item because every item in ITS OWN sequence needs it --
#     verified runner by runner, B.3).
#   - banner()-style section markers, logs under a dedicated logs/ subdir.
#
# DELIBERATE DIFFERENCES, reported rather than silently introduced:
#   - run_central_sweep.sh uses `set -uo pipefail`, NOT `-e`, because one
#     failed arm must not kill the sweep. THIS script's whole purpose is
#     the opposite: Phase B's items are dependency-ordered (item 18
#     consumes item 17's draws file; several share
#     run_gas_regime_ndc_matrix()/run_fin2_blended_bankability_sweep(),
#     so a defect surfacing cheaply at the first exerciser must stop the
#     run before the same defect burns solve time in a more expensive
#     caller downstream). Failure handling is therefore explicit
#     (check-and-exit after every invocation), not a global `-e`/
#     `pipefail`, so the abort always happens at a controlled point with
#     a clear message, never from `-e` firing mid-pipeline.
#   - run_central_sweep.sh's `run()` always builds the SAME 17-flag
#     invocation (the one runner family it drives takes an identical
#     surface). Phase B's 19 runners do not share one surface -- TASK 17
#     B.3 required verifying each one's own argparse directly, and the
#     result is that six distinct exclude-patterns exist across them (see
#     `excluded_keys_for()` below), plus `31_compute_vss_bound.py`, which
#     takes NONE of the 19 standard flags at all (a pure post-processor,
#     confirmed by reading it: no add_standard_arms call, no
#     load_scenario()/build_model call of any kind). This script's run()
#     is therefore driven by a per-runner exclude table, not a fixed flag
#     list, with an UNKNOWN-RUNNER abort if a name is passed that the
#     table has no entry for -- a silently-wrong flag set must never run.
#   - run_central_sweep.sh does not expose --reserve-margin at all, because
#     none of ITS runners take it (01b_/02_ndc_caps/03_/26_ all hardcode
#     reserve_margin="nirp_spinning" internally). Several of THIS script's
#     runners (06_/07_/08_/09_/10_/11_/13_/15_/17_/18_/19_/20_/28_/29_/30_)
#     use src/cli_arms.py's add_standard_arms() with no exclusion on
#     reserve_margin, which makes it REQUIRED on their CLI -- confirmed
#     directly against each file's own argparse block (B.3's own gate),
#     not inferred from this file's sibling. --reserve-margin is therefore
#     a committed central value (C_RESERVE) here, passed to those runners
#     and withheld from the seven that hardcode it -- WITH AN EXPLICIT,
#     LITERAL RUNTIME ASSERT (not just the exclude table) that those seven
#     never receive it, per B.2's own instruction not to rely on a correct
#     string being typed.
#   - TASK 18 Parts A/B: the original design resolved three cross-item
#     data dependencies (0->20, 17->18 x2) by RECENCY -- `ls -t
#     <dir>/<pattern> | head -1`, the most recently modified file in the
#     producing runner's own output directory. This broke in two ways,
#     both found by audit rather than by a failed run: (1) item 19
#     (run_central_sweep.sh) drives 01b_run_baseline_sliced.py across its
#     full cell count into results/baseline_sliced/ between item 0 and
#     item 20, so by the time item 20 ran, recency picked a sweep cell,
#     not item 0's central anchor; (2) item 17 is now itself two
#     invocations (17_A --fix-gas fixed, 17_B --fix-gas free -- see
#     below), so even within one item, recency over the whole directory
#     was never the right primitive. ALL THREE are now pinned
#     explicitly instead:
#       - item 20's --baseline-diagnostics: a committed hash8 constant
#         (ITEM0_CENTRAL_DIAG_HASH8), confirmed against
#         results/baseline_sliced/INDEX.md directly (not taken from a
#         report), not computed and not discovered by directory listing.
#       - item 18's --draws-file / --stage1-summary: grep'd from 17_B's
#         OWN log file specifically (logs/phase_b/17_B_*.log), not from
#         `ls -t` over results/stochastic{,_redispatch}/ -- scoped to
#         the one producing invocation, immune to 17_A's own prior write
#         to the same directory or anything else that might land there.
#     Both methods read a filename a Python runner already printed or a
#     hash8 already confirmed by hand against INDEX.md; neither computes
#     or hashes anything in bash (B.4's own restriction) -- deliberately
#     NOT reproducing src/output_naming.py's hash8_of().
#
# USAGE
#   bash scripts/run_phase_b.sh --dry-run              # print the plan, run nothing
#   bash scripts/run_phase_b.sh --only 0                # one item
#   bash scripts/run_phase_b.sh --from 0 --to 5         # a range
#   bash scripts/run_phase_b.sh --resume                # continue after a stop
#   bash scripts/run_phase_b.sh --allow-dirty ...       # proceed on a dirty tree
#
# Items 0-18: the main dependency-ordered sequence (TASK 17 B.3's table).
# Items 19-23: post-sequence, separately selectable (run_central_sweep.sh,
# the two epsilon-constraint frontiers, Sobol, FIN-3).
#
# WHAT THIS SCRIPT DOES NOT DO (B.4): no deletion, no overwrite of
# results/ beyond what each runner writes under its own name, no manifest
# regeneration, no tagging, no commits. It runs runners and writes logs.
# ============================================================================

set -u
cd "$(dirname "$0")/.."

mkdir -p logs/phase_b
CHECKPOINT="logs/phase_b/.checkpoint"
MAX_ITEM=23

DRY_RUN=0
ALLOW_DIRTY=0
RESUME=0
ONLY=""
FROM=""
TO=""

# ---- driver's own CLI -------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)     DRY_RUN=1; shift ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    --resume)      RESUME=1; shift ;;
    --only)        ONLY="${2:?--only needs an item number}"; shift 2 ;;
    --from)        FROM="${2:?--from needs an item number}"; shift 2 ;;
    --to)          TO="${2:?--to needs an item number}"; shift 2 ;;
    *) echo "ABORT: unknown flag: $1"; exit 1 ;;
  esac
done

if [ -n "$ONLY" ] && { [ -n "$FROM" ] || [ -n "$TO" ]; }; then
  echo "ABORT: --only cannot be combined with --from/--to."; exit 1
fi
if [ "$RESUME" -eq 1 ] && [ -n "$FROM" ]; then
  echo "ABORT: --resume cannot be combined with --from (ambiguous start point)."; exit 1
fi

START=${FROM:-0}
END=${TO:-$MAX_ITEM}
if [ -n "$ONLY" ]; then START=$ONLY; END=$ONLY; fi
if [ "$RESUME" -eq 1 ] && [ -f "$CHECKPOINT" ]; then
  START=$(( $(cat "$CHECKPOINT") + 1 ))
  echo "Resuming from item $START (checkpoint: $(cat "$CHECKPOINT"))."
fi

# ---- header block (printed once; prepended to every item's own log,
# so each log is citable on its own without cross-referencing this) -----
UTC_NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
BRANCH=$(git branch --show-current)
HEAD_SHA=$(git rev-parse HEAD)
DESCRIBE=$(git describe --tags 2>/dev/null || echo "(no tag reachable)")
DIRTY_STATUS=$(git status --porcelain)

HEADER_TEXT=$(cat <<EOF
================ Phase B driver run header ================
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
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!! ABORT: working tree is DIRTY. Phase B results must be       "
    echo "!! attributable to a committed tree. Commit or stash, or pass  "
    echo "!! --allow-dirty if this is deliberate.                        "
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo
    git status --short
    exit 1
  fi
  echo "WARNING: working tree is dirty; proceeding because --allow-dirty was passed."
fi

# ---- committed central arm values (TASK 17 B.2) ------------------------
# Copied from scripts/run_central_sweep.sh's own Section A central
# configuration, verbatim -- same values, same reasoning for not sourcing
# them (bash has no clean cross-script constant import; see that script's
# own header and run_verification_sweep.sh's identical note).
C_PEAK="peak_unsuppressed"
C_RESERVE="nirp_spinning"
C_STORAGE_CC="storage_cc_central"
C_GAS_BUILD="gas_unconstrained"
C_NEW_GAS_DELIV="constrained"
C_GAS_AVAIL="gas_avail_nameplate"
C_GAS_RETIRE="gas_retire_vintage_30"
C_GAS_CAPEX="gas_capex_central"
C_CCGT="ccgt_eff_central"
C_CR="cr_central"
C_DSCR="dscr_central"
C_INFL="0.020"
C_TREAT="f1_tranche"
C_ENV="env_dares"
C_GAS_DELIV="baseline"
C_CONV="conv_central"
C_HYDRO="hydro_flat"
C_TARIFF="tariff_central"
C_OM="om_central"

# TASK 18 Part B: item 0's central-baseline diagnostics file is identified by
# hash8, not by directory recency (item 19, run_central_sweep.sh, writes many
# more files to this same directory between item 0 and item 20).
# TASK 53 2.5 (G7-4): the hash8 is no longer a pasted constant. Every arm now
# reaches 01b's output tag (us_inflation, tariff_case, reserve_margin and the
# window and storage-series defaults were absent), which moved the central hash
# from the pre-epoch `02c17f55` to the one `central_diag_hash8` prints. It is
# derived from the SAME C_* values item 0 runs at, by 01b's own `--print-hash8`
# (no build, no solve), so a future tag change cannot silently invalidate it.
central_diag_hash8 () {
  python3 scripts/01b_run_baseline_sliced.py \
    --peak-case "$C_PEAK" --storage-cc-case "$C_STORAGE_CC" --gas-build-case "$C_GAS_BUILD" \
    --new-gas-deliverability "$C_NEW_GAS_DELIV" --gas-availability-case "$C_GAS_AVAIL" \
    --gas-retirement-case "$C_GAS_RETIRE" --gas-capex-case "$C_GAS_CAPEX" \
    --ccgt-efficiency-case "$C_CCGT" --capital-recovery-case "$C_CR" --dscr-case "$C_DSCR" \
    --us-inflation "$C_INFL" --capital-treatment-case "$C_TREAT" \
    --concessional-envelope-case "$C_ENV" --gas-deliverability-case "$C_GAS_DELIV" \
    --conversion-case "$C_CONV" --hydro-case "$C_HYDRO" --tariff-case "$C_TARIFF" \
    --om-case "$C_OM" --print-hash8
}

# Runners that hardcode reserve_margin="nirp_spinning" internally and
# expose NO --reserve-margin flag at all (confirmed by reading each file
# directly, 2026-10-02): the hand-written-argparse group.
readonly -a NO_RESERVE_MARGIN_RUNNERS=(
  01b_run_baseline_sliced.py 01_run_baseline.py 02_run_eaas.py
  24_run_leakage_arm.py 25_run_obligation_trajectory.py
  99c_run_genset_arm_sensitivity.py 04_run_fin2_tariff_sweep.py
)

# ---- per-runner standard-arm exclusions (TASK 17 B.3's own-args gate) --
# Returns a SPACE-PADDED string of load_scenario() kwarg names this
# runner's own argparse does NOT declare, verified by reading each file's
# add_standard_arms()/hand-written parser.add_argument() block directly --
# not inferred from a sibling runner, not assumed from the flag's name.
# "__UNKNOWN__" for anything not in this table: an unverified runner must
# abort, not silently receive a guessed flag set.
excluded_keys_for () {
  case "$1" in
    01b_run_baseline_sliced.py|01_run_baseline.py)
      echo " reserve_margin " ;;
    02_run_eaas.py)
      echo " reserve_margin gas_deliverability_case " ;;
    24_run_leakage_arm.py|25_run_obligation_trajectory.py|99c_run_genset_arm_sensitivity.py)
      echo " reserve_margin gas_deliverability_case " ;;
    04_run_fin2_tariff_sweep.py)
      echo " reserve_margin gas_deliverability_case " ;;
    06_run_gas1_shadow_benchmarks.py|07_run_gas2_eaas_gas_relief.py)
      echo " gas_deliverability_case " ;;
    08_run_gas3_regime_ndc_feasibility.py)
      echo " gas_deliverability_case new_gas_deliverability " ;;
    13_run_dem2_growth_gas_interaction.py)
      echo " gas_deliverability_case new_gas_deliverability tariff_case " ;;
    09_run_rel1_feasibility.py|10_run_rel2_marginal_cost.py)
      echo " gas_deliverability_case new_gas_deliverability " ;;
    11_run_rel3_financing_frontier.py)
      echo " gas_deliverability_case new_gas_deliverability tariff_case " ;;
    17_run_monte_carlo.py)
      echo " gas_deliverability_case " ;;
    31_compute_vss_bound.py)
      echo " peak_case reserve_margin storage_cc_case gas_build_case new_gas_deliverability gas_availability_case gas_retirement_case gas_capex_case ccgt_efficiency_case capital_recovery_case dscr_case us_inflation capital_treatment_case concessional_envelope_case gas_deliverability_case conversion_case hydro_case tariff_case om_case " ;;
    05_run_fin3_capital_sweep.py|15_run_pol1_ndc_comparison.py|20_run_cf1_solar_yield_sensitivity.py|19_run_carbon_price_sweep.py|30_run_stochastic_redispatch.py|18_run_pareto_frontier.py|28_run_capital_frontier.py|29_run_parameter_uncertainty.py)
      echo " " ;;
    *)
      echo " __UNKNOWN__ " ;;
  esac
}

# ---- execution primitive: logging, timing, free -h, tee, abort ---------
_execute () {
  local log_label="$1"; shift
  local log="logs/phase_b/${log_label}.log"

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
    echo "Phase B stops here -- no continuing past a failure."
    exit 1
  fi
  printf '  [%s] ok -> %s\n' "$log_label" "$log"
}

# ---- run <runner> <log_label> [key=value ...] [-- <extra args verbatim>]
run () {
  local runner="$1"; shift
  local log_label="$1"; shift

  local peak=$C_PEAK reserve=$C_RESERVE storage_cc=$C_STORAGE_CC gas_build=$C_GAS_BUILD
  local new_gas_deliv=$C_NEW_GAS_DELIV gas_avail=$C_GAS_AVAIL gas_retire=$C_GAS_RETIRE
  local gas_capex=$C_GAS_CAPEX ccgt=$C_CCGT cr=$C_CR dscr=$C_DSCR infl=$C_INFL
  local treat=$C_TREAT env=$C_ENV gas_deliv=$C_GAS_DELIV conv=$C_CONV hydro=$C_HYDRO
  local tariff=$C_TARIFF om=$C_OM
  local -a extra=()

  while [ $# -gt 0 ]; do
    if [ "$1" = "--" ]; then
      shift; extra=("$@"); break
    fi
    case "${1%%=*}" in
      peak)          peak="${1#*=}" ;;
      reserve)       reserve="${1#*=}" ;;
      storage_cc)    storage_cc="${1#*=}" ;;
      gas_build)     gas_build="${1#*=}" ;;
      new_gas_deliv) new_gas_deliv="${1#*=}" ;;
      gas_avail)     gas_avail="${1#*=}" ;;
      gas_retire)    gas_retire="${1#*=}" ;;
      gas_capex)     gas_capex="${1#*=}" ;;
      ccgt)          ccgt="${1#*=}" ;;
      cr)            cr="${1#*=}" ;;
      dscr)          dscr="${1#*=}" ;;
      infl)          infl="${1#*=}" ;;
      treat)         treat="${1#*=}" ;;
      env)           env="${1#*=}" ;;
      gas_deliv)     gas_deliv="${1#*=}" ;;
      conv)          conv="${1#*=}" ;;
      hydro)         hydro="${1#*=}" ;;
      tariff)        tariff="${1#*=}" ;;
      om)            om="${1#*=}" ;;
      *) echo "ABORT: unknown override key in run(): $1"; exit 1 ;;
    esac
    shift
  done

  local excl
  excl=$(excluded_keys_for "$runner")
  if [[ "$excl" == *" __UNKNOWN__ "* ]]; then
    echo "ABORT: no verified own-args entry for runner '$runner' in excluded_keys_for()."
    echo "Do not guess its flag set -- add a verified entry first."
    exit 1
  fi

  local -a args=()
  [[ "$excl" == *" peak_case "* ]]                 || args+=(--peak-case "$peak")
  [[ "$excl" == *" reserve_margin "* ]]             || args+=(--reserve-margin "$reserve")
  [[ "$excl" == *" storage_cc_case "* ]]            || args+=(--storage-cc-case "$storage_cc")
  [[ "$excl" == *" gas_build_case "* ]]             || args+=(--gas-build-case "$gas_build")
  [[ "$excl" == *" new_gas_deliverability "* ]]     || args+=(--new-gas-deliverability "$new_gas_deliv")
  [[ "$excl" == *" gas_availability_case "* ]]      || args+=(--gas-availability-case "$gas_avail")
  [[ "$excl" == *" gas_retirement_case "* ]]        || args+=(--gas-retirement-case "$gas_retire")
  [[ "$excl" == *" gas_capex_case "* ]]             || args+=(--gas-capex-case "$gas_capex")
  [[ "$excl" == *" ccgt_efficiency_case "* ]]       || args+=(--ccgt-efficiency-case "$ccgt")
  [[ "$excl" == *" capital_recovery_case "* ]]      || args+=(--capital-recovery-case "$cr")
  [[ "$excl" == *" dscr_case "* ]]                  || args+=(--dscr-case "$dscr")
  [[ "$excl" == *" us_inflation "* ]]               || args+=(--us-inflation "$infl")
  [[ "$excl" == *" capital_treatment_case "* ]]     || args+=(--capital-treatment-case "$treat")
  [[ "$excl" == *" concessional_envelope_case "* ]] || args+=(--concessional-envelope-case "$env")
  [[ "$excl" == *" gas_deliverability_case "* ]]    || args+=(--gas-deliverability-case "$gas_deliv")
  [[ "$excl" == *" conversion_case "* ]]            || args+=(--conversion-case "$conv")
  [[ "$excl" == *" hydro_case "* ]]                 || args+=(--hydro-case "$hydro")
  [[ "$excl" == *" tariff_case "* ]]                || args+=(--tariff-case "$tariff")
  [[ "$excl" == *" om_case "* ]]                    || args+=(--om-case "$om")

  # B.2's explicit, literal assert -- independent of excl above, a
  # last-resort check that these seven NEVER receive --reserve-margin,
  # rather than trusting the exclude table alone to have been typed right.
  for nr in "${NO_RESERVE_MARGIN_RUNNERS[@]}"; do
    if [ "$runner" = "$nr" ]; then
      for a in "${args[@]}"; do
        if [ "$a" = "--reserve-margin" ]; then
          echo "ABORT: $runner must never receive --reserve-margin (hardcodes nirp_spinning internally)."
          exit 1
        fi
      done
      break
    fi
  done

  args+=("${extra[@]}")
  _execute "$log_label" python3 "scripts/$runner" "${args[@]}"
}

banner () { echo; echo "════════════════════════════════════════════════════"; echo "$1"; echo "════════════════════════════════════════════════════"; }

should_run () { local n="$1"; [ "$n" -ge "$START" ] && [ "$n" -le "$END" ]; }

checkpoint () { [ "$DRY_RUN" -eq 1 ] || echo "$1" > "$CHECKPOINT"; }

# ============================================================================
# ITEMS 0-18 -- the main dependency-ordered sequence
# ============================================================================

item_0 () {
  banner "ITEM 0 -- 01b_run_baseline_sliced.py x2 (2 solves: central + regression anchor)"
  run 01b_run_baseline_sliced.py 00_A_central_baseline
  run 01b_run_baseline_sliced.py 00_B_regression_anchor \
    conv=conv_none treat=pre_f1_overnight dscr=dscr_regression \
    tariff=tariff_legacy_65m om=om_none
  if [ "$DRY_RUN" -eq 0 ]; then
    echo "CHECK: 00_B_regression_anchor must read 21.328 bn USD / 43,539.97566 MWh"
    echo "       (register L58). If it does not, this run is unattributable."
  fi
}

item_1 ()  { banner "ITEM 1 -- 01_run_baseline.py (1 solve)";                run 01_run_baseline.py 01_baseline_annual; }
item_2 ()  { banner "ITEM 2 -- 02_run_eaas.py (1 solve)";                    run 02_run_eaas.py 02_eaas; }
item_3 ()  { banner "ITEM 3 -- 15_run_pol1_ndc_comparison.py (8 solves)";    run 15_run_pol1_ndc_comparison.py 03_pol1_ndc_comparison; }
item_4 ()  { banner "ITEM 4 -- 20_run_cf1_solar_yield_sensitivity.py (3 solves)"; run 20_run_cf1_solar_yield_sensitivity.py 04_cf1_solar_yield; }
item_5 ()  { banner "ITEM 5 -- 19_run_carbon_price_sweep.py (39 solves, extended by TASK 04)"; run 19_run_carbon_price_sweep.py 05_carbon_price_sweep; }
item_6 ()  { banner "ITEM 6 -- 24_run_leakage_arm.py (6 solves)";            run 24_run_leakage_arm.py 06_leakage_arm; }
item_7 ()  { banner "ITEM 7 -- 25_run_obligation_trajectory.py (3 solves)";  run 25_run_obligation_trajectory.py 07_obligation_trajectory; }
item_8 ()  { banner "ITEM 8 -- 99c_run_genset_arm_sensitivity.py (36 solves)"; run 99c_run_genset_arm_sensitivity.py 08_genset_arm_sensitivity; }
item_9 ()  { banner "ITEM 9 -- 06_run_gas1_shadow_benchmarks.py (8 solves; first exerciser of run_gas_regime_ndc_matrix(sliced))"; run 06_run_gas1_shadow_benchmarks.py 09_gas1_shadow_benchmarks; }
item_10 () { banner "ITEM 10 -- 07_run_gas2_eaas_gas_relief.py (40 solves; same shared function)"; run 07_run_gas2_eaas_gas_relief.py 10_gas2_eaas_gas_relief; }
item_11 () { banner "ITEM 11 -- 08_run_gas3_regime_ndc_feasibility.py (28 solves; same shared function)"; run 08_run_gas3_regime_ndc_feasibility.py 11_gas3_regime_ndc_feasibility; }
item_12 () { banner "ITEM 12 -- 04_run_fin2_tariff_sweep.py (200 solves; first exerciser of run_fin2_blended_bankability_sweep(sliced))"; run 04_run_fin2_tariff_sweep.py 12_fin2_tariff_sweep; }
item_13 () { banner "ITEM 13 -- 13_run_dem2_growth_gas_interaction.py (72 solves; same shared function)"; run 13_run_dem2_growth_gas_interaction.py 13_dem2_growth_gas_interaction; }

# TASK 23, Part B.5 (docs/task_log_23b_rel1_arm_grid.md): item 14 needed
# two more invocations once 09_run_rel1_feasibility.py became
# gas_availability_case-aware (Part B.2) -- two modes x two arms, same
# convention as item 15/16's own A/B pairs, extended to four. Solve
# count: 4 gas_cases x 2 solar_build_cases x (1+11 eps) = 96 per
# invocation (was 56 at the pre-Part-B.1 6-level grid), 384 for all four.
item_14 () {
  banner "ITEM 14 -- 09_run_rel1_feasibility.py x4 (384 solves; first exerciser of run_rel1_feasibility_matrix(sliced))"
  run 09_run_rel1_feasibility.py 14_A_rel1_total_nameplate  gas_avail=gas_avail_nameplate -- --reliability-mode total
  run 09_run_rel1_feasibility.py 14_B_rel1_total_observed   gas_avail=gas_avail_observed  -- --reliability-mode total
  run 09_run_rel1_feasibility.py 14_C_rel1_annual_nameplate gas_avail=gas_avail_nameplate -- --reliability-mode annual
  run 09_run_rel1_feasibility.py 14_D_rel1_annual_observed  gas_avail=gas_avail_observed  -- --reliability-mode annual
}

item_15 () {
  banner "ITEM 15 -- 10_run_rel2_marginal_cost.py x2 (96 solves; first exerciser of run_reliability_sweep(sliced))"
  run 10_run_rel2_marginal_cost.py 15_A_rel2_gas_avail_nameplate gas_avail=gas_avail_nameplate
  run 10_run_rel2_marginal_cost.py 15_B_rel2_gas_avail_observed  gas_avail=gas_avail_observed
}

# TASK 22 (docs/task_log_22_rel3_grid.md): the handover's "728 solves"
# figure for this item was already stale before TASK 22 -- the actual
# formula (len(GAS_CASES)*len(POLICY_CONFIGS)*2*(1+len(EPS_LEVELS))) gives
# 280 at the pre-TASK-22 6-level grid (confirmed independently, and by
# TASK 22's own stated baseline), 480 per invocation after Part A's
# extension to 11 levels.
#
# TASK 23, Part A (docs/task_log_23a_driver_item16.md): TASK 22 made
# 11_run_rel3_financing_frontier.py arm-aware (results/rel3/
# <gas_availability_case>/), but this item still invoked it once --
# Zeus's own item-16 run produced only results/rel3/gas_avail_nameplate/.
# Split into two invocations, same convention as item 15 above: 16_A at
# gas_avail_nameplate, 16_B at gas_avail_observed, identical on every
# other flag. Banner updated to the pair's total (480*2=960).
item_16 () {
  banner "ITEM 16 -- 11_run_rel3_financing_frontier.py x2 (960 solves; largest runner in the project)"
  run 11_run_rel3_financing_frontier.py 16_A_rel3_gas_avail_nameplate gas_avail=gas_avail_nameplate
  run 11_run_rel3_financing_frontier.py 16_B_rel3_gas_avail_observed  gas_avail=gas_avail_observed
}

item_17 () {
  banner "ITEM 17 -- 30_run_stochastic_redispatch.py x2, same draws, two --fix-gas legs (TASK 18 Part A)"
  # TASK 06 (F-32): firm_capacity_constraint carries no priced slack, so
  # under --fix-gas fixed a draw whose peak exceeds the credited fixed
  # capacity is genuinely infeasible (F-32 measured 50/100). An
  # expectation over infeasible draws is infinite and bounds nothing --
  # 31_compute_vss_bound.py's Guard 2 refuses to emit a bound unless the
  # stage-1 run used --fix-gas free. 17_B below is that leg.
  #
  # 17_A (--fix-gas fixed) is NOT redundant with 17_B: it is the only
  # configuration in the project that can observe a chosen plan failing
  # under a draw, which is where F-32's finding comes from. Keep it,
  # and keep it first.
  #
  # Own-arg values shared by BOTH legs: the project's own established
  # central stochastic configuration (parameter_provenance.md rows
  # 16-19; identical to multiple pre-existing entries in
  # results/stochastic_redispatch/INDEX.md, e.g. hash8 ae0a8d24), not
  # invented here. TASK 18 Part A asked whether --demand-sigma 0.01
  # matches demand_growth_prior()'s own sigma -- verified and reported
  # in docs/task_log_18a_driver_fix_gas.md: it does not (0.01 here vs.
  # 0.0120 there, a different, self-described calibration per this
  # runner's own docstring Section 6), and per that part's own
  # instruction the value is reported, not changed.
  run 30_run_stochastic_redispatch.py 17_A_stochastic_redispatch_fixed -- \
    --n-draws 100 --seed 42 --demand-sigma 0.01 --carbon-mu 3.9 \
    --carbon-sigma 0.3 --fix-gas fixed --fix-tolerance 1e-5 --gas-weights sourced
  run 30_run_stochastic_redispatch.py 17_B_stochastic_redispatch_free -- \
    --n-draws 100 --seed 42 --demand-sigma 0.01 --carbon-mu 3.9 \
    --carbon-sigma 0.3 --fix-gas free --fix-tolerance 1e-5 --gas-weights sourced
}

item_18 () {
  banner "ITEM 18 -- 17_run_monte_carlo.py then 31_compute_vss_bound.py (draws + 1; must follow item 17, consumes 17_B specifically)"
  # TASK 18 Part A/B: pinned to 17_B's own log file, not to directory
  # recency -- immune to 17_A's own prior write to the same directories
  # (both legs write results/stochastic_redispatch/; the draws CSV in
  # results/stochastic/ is shared by design, since 30_'s draws-tag hash
  # excludes --fix-gas/--fix-tolerance, but is still read from 17_B's
  # log specifically so this item never depends on which leg ran last).
  local log17b="logs/phase_b/17_B_stochastic_redispatch_free.log"
  local draws_csv stage1_summary
  if [ "$DRY_RUN" -eq 1 ]; then
    draws_csv="<from $log17b, 17_B's own 'Saved:' line>"
    stage1_summary="<from $log17b, 17_B's own 'Saved:' line>"
  else
    if [ ! -f "$log17b" ]; then
      echo "ABORT: item 18 needs item 17_B's own log ($log17b) to resolve its inputs."
      echo "Run item 17 first (--only 17)."
      exit 1
    fi
    draws_csv=$(grep -oE 'Saved: [^ ]*draws__[0-9a-f]+\.csv' "$log17b" | tail -1 | sed 's/^Saved: //')
    stage1_summary=$(grep -oE 'Saved: [^ ]*summary__[0-9a-f]+\.json' "$log17b" | tail -1 | sed 's/^Saved: //')
    if [ -z "$draws_csv" ] || [ -z "$stage1_summary" ]; then
      echo "ABORT: could not parse item 17_B's draws CSV / stage-1 summary paths from $log17b."
      exit 1
    fi
    echo "Using draws file (17_B's own log)      : $draws_csv"
    echo "Using stage-1 summary (17_B's own log) : $stage1_summary"
  fi
  run 17_run_monte_carlo.py 18_A_monte_carlo -- --draws-file "$draws_csv"
  # TASK 53 2.5 (G7-4): 17_ now writes to results/monte_carlo/arms_<hash8>/ (printed as "Results directory: ..."),
  # so 31_'s --mc-results-dir is read from 18_A's own log, the same way draws_csv is read from 17_B's.
  local mc_dir
  if [ "$DRY_RUN" -eq 1 ]; then
    mc_dir="<from logs/phase_b/18_A_monte_carlo.log, 'Results directory:' line>"
  else
    mc_dir=$(grep -oE 'Results directory: .*' logs/phase_b/18_A_monte_carlo.log | tail -1 | sed 's/^Results directory: //')
    [ -n "$mc_dir" ] || { echo "ABORT: could not parse 17_'s results directory from its log."; exit 1; }
    echo "Using Monte Carlo results dir (18_A's own log): $mc_dir"
  fi
  run 31_compute_vss_bound.py 18_B_vss_bound -- \
    --stage1-summary "$stage1_summary" \
    --mc-results-dir "$mc_dir" \
    --ws-case "Matched to 30_ central (no NDC)"
}

# ============================================================================
# ITEMS 19-22 -- post-sequence, separately selectable
# ============================================================================

item_19 () {
  banner "ITEM 19 -- scripts/run_central_sweep.sh (52 invocations as of this commit -- read that file for the current count, do not trust this comment to stay accurate)"
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '  [19_run_central_sweep] bash scripts/run_central_sweep.sh\n'
    return 0
  fi
  _execute "19_run_central_sweep" bash scripts/run_central_sweep.sh
}

item_20 () {
  banner "ITEM 20 -- 18_run_pareto_frontier.py (TASK 26 Part B: per arm 1 unconstrained anchor + 25 grid + up to 6 tight-end extension = 108 to 128 solves over 4 arms; cost-emissions epsilon-frontier)"
  # TASK 18 Part B: pinned by hash8 (ITEM0_CENTRAL_DIAG_HASH8, committed
  # above, confirmed against results/baseline_sliced/INDEX.md directly),
  # not by directory recency -- item 19 (run_central_sweep.sh) writes
  # many more files to this same directory between item 0 and item 20,
  # so recency would pick a sweep cell here, not item 0's central run.
  local ITEM0_CENTRAL_DIAG_HASH8
  ITEM0_CENTRAL_DIAG_HASH8=$(central_diag_hash8) || { echo "ABORT: could not derive the central hash8"; exit 1; }
  local baseline_diag="results/baseline_sliced/diagnostics__${ITEM0_CENTRAL_DIAG_HASH8}.json"
  if [ "$DRY_RUN" -eq 0 ] && [ ! -f "$baseline_diag" ]; then
    echo "ABORT: item 0's central diagnostics file not found at $baseline_diag."
    echo "Run item 0 first (--only 0) before item 20."
    exit 1
  fi
  echo "Using baseline diagnostics (pinned by hash8): $baseline_diag"
  run 18_run_pareto_frontier.py 20_pareto_frontier -- --baseline-diagnostics "$baseline_diag"
}

item_21 () { banner "ITEM 21 -- 28_run_capital_frontier.py (19 points x 2 regimes = 38; cost-capital epsilon-frontier)"; run 28_run_capital_frontier.py 21_capital_frontier; }

item_22 () {
  banner "ITEM 22 -- 29_run_parameter_uncertainty.py (Sobol, N=256*(5+2)=1792 solves)"
  run 29_run_parameter_uncertainty.py 22_sobol_parameter_uncertainty -- --n-base 256 --seed 42
}

# TASK 32 C. FIN-3 (F-31, section 3.5.6's leakage table). Not in any earlier
# Phase B item: its stored output (results/fin3/, 2026-09-26 12:09) predates
# the Phase B re-run. It takes the full standard arm block (its argparse
# declares every flag, checked with --help), writes only to results/fin3/,
# and reads data/cost/processed/emissions_cap.csv (must exist). 15 solves =
# 5 capital cases x 3 collection rates; sliced builder; no forced
# required_margin (public/traditional regime). NOT RUN as of the commit that
# added this item. A re-run renames the carbon-shadow columns in
# capital_sweep_results.csv (TASK 27 A: *_all_year_mean / *_binding_mean).
item_23 () { banner "ITEM 23 -- 05_run_fin3_capital_sweep.py (15 solves = 5 capital cases x 3 collection rates)"; run 05_run_fin3_capital_sweep.py 23_fin3_capital_sweep; }

# ============================================================================
# DISPATCH
# ============================================================================

echo "Phase B driver: items $START..$END$( [ "$DRY_RUN" -eq 1 ] && echo " (DRY RUN -- nothing will execute)" )"

for n in $(seq 0 "$MAX_ITEM"); do
  should_run "$n" || continue
  "item_$n"
  checkpoint "$n"
done

banner "PHASE B DRIVER: requested range complete ($START..$END)"
echo "Logs in logs/phase_b/. Checkpoint: $( [ -f "$CHECKPOINT" ] && cat "$CHECKPOINT" || echo "(none written -- dry run)" )"
