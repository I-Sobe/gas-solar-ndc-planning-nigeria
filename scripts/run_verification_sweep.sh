#!/usr/bin/env bash
set -u
export PYTHONIOENCODING=utf-8
# Snapshot target. A bare name is resolved under ./snapshots/ -- Git Bash
# maps /tmp to %TEMP%, but python is a native Windows binary and reads /tmp
# literally as C:\tmp, so /tmp snapshots are unreadable from Python. Keep
# snapshots repo-relative.
#
# task_audit_fixes_p2 (audit finding A26). Before this fix, every runner
# below was invoked with AT MOST --om-case -- nothing else -- while the
# "healthy" five (01_/01b_/02_run_eaas/02_run_ndc_caps/03_) each require
# 15-18 flags and five more (05_/15_/18_/19_/20_, via src/cli_arms.py)
# require the full standard-arm surface too. Every one of those ten
# rejected the old invocation immediately with argparse's own "the
# following arguments are required" error -- indistinguishable, in the
# old binary pass/fail check below, from a genuine crash in one of the
# remaining, already-dead runners (register L10/L27). The harness could
# not produce a usable signal for anything and would ALWAYS refuse to
# snapshot. Fixed two ways: (1) every invocation now receives the full
# canonical CENTRAL argument set (the same flag names and values
# scripts/run_central_sweep.sh's own C_* constants use -- copied here,
# not sourced, since bash has no clean cross-script constant import; if
# the central configuration ever changes, both files must be updated
# together, same as before this fix for the two om_central call sites
# that already existed); (2) outcomes are now PASS / NEEDS-ARGS / FAIL,
# using the EXACT detection scripts/smoke_all.sh already uses (exit code
# 2 AND argparse's specific "error: the following arguments are
# required" text on stderr, matched TOGETHER -- reused verbatim rather
# than re-derived, per scripts/smoke_all.sh's own header comment on why
# exit code 2 alone is not specific enough). This does not make every
# runner pass -- the runners with no argparse at all (register L10/L27:
# 06_, 07_, 08_, 09_, 10_, 11_, 13_, 14_) still fail, now correctly
# bucketed as FAIL rather than conflated with a healthy refusal, because
# their failure does NOT match the NEEDS-ARGS signature (no argparse
# means no "the following arguments are required" text and no exit code
# 2 -- they fail with a plain Python TypeError from a missing
# load_scenario() argument instead, exit code 1). That is the correct,
# informative outcome for scripts already declared dead; this harness
# does not attempt to repair them.
SNAPSHOT_ARG="${1:-}"
SNAPSHOT_DIR=""
if [ -n "$SNAPSHOT_ARG" ]; then
    case "$SNAPSHOT_ARG" in
        /*|[A-Za-z]:*) SNAPSHOT_DIR="$SNAPSHOT_ARG" ;;
        *)             SNAPSHOT_DIR="snapshots/$SNAPSHOT_ARG"; mkdir -p snapshots ;;
    esac
fi
LOG_DIR="/tmp/sweep_logs"
mkdir -p "$LOG_DIR"; rm -f "$LOG_DIR"/*.log
RUNNERS="
scripts/01_run_baseline.py
scripts/01b_run_baseline_sliced.py
scripts/02_run_eaas.py
scripts/02_run_ndc_caps.py
scripts/03_run_ndc_eaas.py
scripts/05_run_fin3_capital_sweep.py
scripts/06_run_gas1_shadow_benchmarks.py
scripts/07_run_gas2_eaas_gas_relief.py
scripts/08_run_gas3_regime_ndc_feasibility.py
scripts/09_run_rel1_feasibility.py
scripts/10_run_rel2_marginal_cost.py
scripts/11_run_rel3_financing_frontier.py
scripts/13_run_dem2_growth_gas_interaction.py
scripts/14_run_str1_storage_role.py
scripts/15_run_pol1_ndc_comparison.py
scripts/18_run_pareto_frontier.py
scripts/19_run_carbon_price_sweep.py
scripts/20_run_cf1_solar_yield_sensitivity.py
scripts/99c_run_genset_arm_sensitivity.py
"
# task_audit_fixes_p2, Part 3 step 5: audited every entry above against
# the files that actually exist on disk (`[ -f "$f" ]` below does this on
# every run, not just once at write time) -- all nineteen are present
# under their current names; none renamed or missing. Nothing to report
# here beyond confirming the check itself still runs.
MISSING=""
for f in $RUNNERS; do [ -f "$f" ] || MISSING="$MISSING $f"; done
if [ -n "$MISSING" ]; then echo "ABORT -- not found:$MISSING"; exit 1; fi

# ---- canonical central configuration ---------------------------------
# Copied from scripts/run_central_sweep.sh's own C_* constants (its
# Section A central run) -- same flag names, same values. Kept as a
# separate literal copy rather than sourced, since run_central_sweep.sh
# defines these as shell variables inside its own script body, not as a
# separately-sourceable file; if the central configuration is ever
# revised there, this block needs the same edit.
CENTRAL_ARGS=(
  --peak-case peak_unsuppressed
  --storage-cc-case storage_cc_central
  --gas-build-case gas_unconstrained
  --new-gas-deliverability constrained
  --gas-availability-case gas_avail_nameplate
  --gas-retirement-case gas_retire_vintage_30
  --gas-capex-case gas_capex_central
  --ccgt-efficiency-case ccgt_eff_central
  --gas-deliverability-case baseline
  --conversion-case conv_central
  --hydro-case hydro_flat
  --tariff-case tariff_central
  --capital-treatment-case f1_tranche
  --dscr-case dscr_central
  --capital-recovery-case cr_central
  --us-inflation 0.020
  --concessional-envelope-case env_dares
)

PASS=0; NEEDS_ARGS=0; FAIL=0
FAILED=""; N=0
for f in $RUNNERS; do
  name=$(basename "$f" .py); N=$((N+1))
  # task_om_implementation: --om-case is required=True on these five
  # runners only; every other runner in RUNNERS above does not expose
  # the flag (26_-style internal resolution, or no argparse at all).
  om_args=()
  case "$name" in
    01_run_baseline|01b_run_baseline_sliced|02_run_eaas|02_run_ndc_caps|03_run_ndc_eaas)
      om_args=(--om-case om_central) ;;
  esac
  log="$LOG_DIR/$name.log"
  python "$f" "${CENTRAL_ARGS[@]}" "${om_args[@]}" > "$log" 2>&1
  code=$?
  # Reused verbatim from scripts/smoke_all.sh's own three-outcome check
  # (same exit code, same grep pattern, same rationale for requiring
  # both together -- see that script's header for why exit code 2 alone
  # is not specific enough).
  if [ $code -eq 0 ]; then
    outcome="PASS"; PASS=$((PASS+1))
  elif [ $code -eq 2 ] && grep -q "error: the following arguments are required" "$log"; then
    outcome="NEEDS-ARGS"; NEEDS_ARGS=$((NEEDS_ARGS+1))
  else
    outcome="FAIL"; FAIL=$((FAIL+1)); FAILED="$FAILED $name"
  fi
  printf "%-12s %-44s exit=%d\n" "$outcome" "$name" "$code"
done
echo
echo "ran $N runners: $PASS passed, $NEEDS_ARGS needs-args, $FAIL failed"
echo "FAILED (genuine, not needs-args):${FAILED:- none}"
echo; echo "--- tracebacks ---"; grep -il "traceback" "$LOG_DIR"/*.log || echo "  none"
echo; echo "--- plan 2.5 guard ---"
grep -h "plan 2.5" "$LOG_DIR"/*.log 2>/dev/null | sed 's/^ *//' | sort | uniq -c || echo "  none"
if [ -n "$SNAPSHOT_DIR" ]; then
  # task_audit_fixes_p2: snapshotting is now gated on genuine FAILures
  # only, not on NEEDS-ARGS. Under the old all-or-nothing gate (any
  # non-zero exit refuses the snapshot), and given this RUNNERS list
  # contains several already-declared-dead runners (register L10/L27)
  # that will ALWAYS report something other than PASS, the gate would
  # never once have allowed a snapshot, regardless of whether the
  # healthy runners' results were good. A NEEDS-ARGS outcome is an
  # already-known, expected state, not a new regression; it must not
  # block capturing the runners that DID produce output.
  if [ -n "$FAILED" ]; then echo; echo "REFUSING snapshot: runners failed."; exit 1; fi
  rm -rf "$SNAPSHOT_DIR"; cp -r results "$SNAPSHOT_DIR"; echo; echo "snapshot -> $SNAPSHOT_DIR"
fi
[ -n "$FAILED" ] && exit 1; exit 0
