#!/usr/bin/env bash
#
# Phase 2.6 commit-A regression.
#
# Runs both baselines at the PRE-PATCH commit in a detached git worktree, then
# at HEAD, and compares. With genset_case=None passed, selfgen is
# priced at VoLL with a zero emission factor and is an exact duplicate of
# unserved, so the optimum must not move.
#
# Usage:
#   bash scripts/verify_2_6_commit_a.sh <pre_patch_ref>
# e.g.
#   bash scripts/verify_2_6_commit_a.sh HEAD~1
#
# Uses a worktree rather than git stash: the working tree is never touched, so
# a failure part-way through cannot lose uncommitted work.
#
# task_audit_fixes_p2 (audit finding A27). Before this fix, steps 2 and 3
# invoked scripts/01_run_baseline.py and scripts/01b_run_baseline_
# sliced.py with, at most, --om-case -- both scripts now require 15-18
# flags. Under `set -euo pipefail`, the FIRST such failure aborted the
# whole script immediately, before the comparison step ever ran, with no
# per-runner status reported at all. Fixed two ways: (1) all four runner
# invocations (pre-patch annual/sliced, post-patch annual/sliced) now
# receive the REGRESSION arm set -- the SAME configuration scripts/
# run_central_sweep.sh's own A_REGRESSION_ANCHOR invocation uses (the
# central set with conversion/capital-treatment/DSCR/tariff/O&M
# overridden to their regression values) -- chosen because this script's
# own stated purpose is a REGRESSION comparison ("the optimum must not
# move"), not a central-configuration health check; (2) each of the four
# invocations is now captured individually (exit code recorded, script
# continues) rather than left to `set -e`, using the SAME PASS/NEEDS-ARGS/
# FAIL detection scripts/smoke_all.sh already uses, so a failure is
# REPORTED per runner instead of silently aborting the whole harness.
# `set -euo pipefail` is KEPT for every OTHER step (the dirty-tree guard,
# the worktree setup, the file copies) -- those genuinely cannot proceed
# on failure and losing that protection would be a new defect, not a fix.
#
# KNOWN LIMITATION, stated rather than solved: PRE_REF is a caller-
# supplied, arbitrary commit. This script cannot know in advance whether
# that commit's own copy of these two runners recognises the regression
# flag set below -- older commits may predate some of these arms
# entirely, or may have used a different flag surface, or none at all.
# If the pre-patch invocation fails for that reason, it is now REPORTED
# (as FAIL, since an unrecognised-argument error does not match the
# NEEDS-ARGS signature) rather than silently aborting -- but no flag
# combination this script could choose is guaranteed to work against an
# arbitrary historical commit, and this fix does not attempt to solve
# that; it only stops one failure from hiding all four outcomes.

set -uo pipefail   # NOT -e for the block below -- see note above and at each runner call

PRE_REF="${1:-HEAD~1}"
ROOT="$(git rev-parse --show-toplevel)"
WT="${TMPDIR:-/tmp}/nigeria_2_6_prepatch"
OUT="${ROOT}/results/_regression_2_6"

cd "${ROOT}"

# ------------------------------------------------------------------
# 0. Preconditions -- genuinely fatal, kept under -e semantics manually
# ------------------------------------------------------------------
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: working tree is dirty. Commit the 2.6 plumbing first."
    echo "       This script compares HEAD against ${PRE_REF}; uncommitted"
    echo "       changes would not be included and the test would be a lie."
    exit 1
fi

command -v python >/dev/null || { echo "ERROR: python not on PATH. conda activate nigeria-energy"; exit 1; }
python -c "import pyomo" 2>/dev/null || { echo "ERROR: pyomo not importable. conda activate nigeria-energy"; exit 1; }

mkdir -p "${OUT}" || { echo "ERROR: could not create ${OUT}"; exit 1; }

# ------------------------------------------------------------------
# Regression arm set -- copied from scripts/run_central_sweep.sh's own
# A_REGRESSION_ANCHOR invocation (the central set with these five
# overridden). Kept as a literal copy for the same reason run_
# verification_sweep.sh's CENTRAL_ARGS is: no cross-script constant
# import in bash without sourcing the whole file.
# ------------------------------------------------------------------
REGRESSION_ARGS=(
  --peak-case peak_unsuppressed
  --storage-cc-case storage_cc_central
  --gas-build-case gas_unconstrained
  --new-gas-deliverability constrained
  --gas-availability-case gas_avail_nameplate
  --gas-retirement-case gas_retire_vintage_30
  --gas-capex-case gas_capex_central
  --ccgt-efficiency-case ccgt_eff_central
  --gas-deliverability-case baseline
  --conversion-case conv_none
  --hydro-case hydro_flat
  --tariff-case tariff_legacy_65m
  --capital-treatment-case pre_f1_overnight
  --dscr-case dscr_regression
  --capital-recovery-case cr_central
  --us-inflation 0.020
  --concessional-envelope-case env_dares
  --om-case om_none
)

# Reused verbatim from scripts/smoke_all.sh: PASS / NEEDS-ARGS / FAIL,
# by exit code plus argparse's own required-argument error text matched
# together. Records the outcome and exit code without aborting the
# script -- the caller decides what to do with a non-PASS outcome.
run_and_classify () {
  local label="$1"; shift
  local log="$1"; shift
  "$@" > "$log" 2>&1
  local code=$?
  if [ $code -eq 0 ]; then
    echo "  ${label}: PASS"
  elif [ $code -eq 2 ] && grep -q "error: the following arguments are required" "$log"; then
    echo "  ${label}: NEEDS-ARGS (exit=$code) -- see $log"
  else
    echo "  ${label}: FAIL (exit=$code) -- see $log"
  fi
  return $code
}

# ------------------------------------------------------------------
# 1. Smoke test at HEAD -- catches AttributeError/NameError in seconds
#    rather than after a full solve. Builds scenario/econ directly in
#    Python, bypassing argparse entirely -- unaffected by, and does not
#    test for, the CLI required-argument surface the rest of this
#    script now has to account for.
# ------------------------------------------------------------------
echo "=== [1/4] Smoke test: do both builders construct? ==="
python - <<'PY'
import sys
from pathlib import Path
ROOT = Path.cwd()
sys.path.append(str(ROOT))
from src.scenarios import load_scenario, solar_min_build_default, MODEL_END_YEAR
from src.optimize_model import build_model
from src.optimize_model_sliced import build_model_sliced
from src.io import load_econ, load_solar_capex_by_year

sc = load_scenario(
    demand_level_case="served", demand_case="organic_central",
    capital_case="unconstrained", gas_deliverability_case="baseline",
    solar_build_case="deployment_unconstrained", land_case="loose",
    carbon_case="no_policy", start_year=2025, end_year=MODEL_END_YEAR,
)
econ = load_econ("voll_mid", genset_case=None)             # EXPLICIT fallback path
assert "GENSET_COST_PER_TWH_E" not in econ, "fallback broken: genset keys leaked in"
yrs = sc["years"]
capex = load_solar_capex_by_year("solar_low", int(yrs[0]), int(yrs[-1]))
sc["solar_min_build_mw_per_year"] = solar_min_build_default()

for name, fn in (("annual", build_model), ("sliced", build_model_sliced)):
    m = fn(scenario=sc, econ=econ, emissions_cap=1e18,
           emissions_cap_by_year=None, solar_capex_by_year=capex)
    assert hasattr(m, "selfgen"), f"{name}: m.selfgen missing"
    assert hasattr(m, "cost_selfgen_npv"), f"{name}: cost_selfgen_npv missing"
    assert hasattr(m, "emissions_grid_only"), f"{name}: emissions_grid_only missing"
    print(f"  {name}: builds, selfgen present")
print("  OK")
PY
if [ $? -ne 0 ]; then echo "ABORT: smoke test failed -- fix before running the full comparison."; exit 1; fi

# ------------------------------------------------------------------
# 2. Pre-patch run in an isolated worktree
# ------------------------------------------------------------------
echo
echo "=== [2/4] Pre-patch baselines at ${PRE_REF} ==="
rm -rf "${WT}"
if ! git worktree add --detach "${WT}" "${PRE_REF}" >/dev/null; then
  echo "ERROR: could not create worktree at ${PRE_REF}. Does that ref exist?"
  exit 1
fi
trap 'git worktree remove --force "${WT}" >/dev/null 2>&1 || true' EXIT

PREPATCH_OK=1
(
  cd "${WT}"
  run_and_classify "pre-patch annual" "${OUT}/prepatch_annual.log" \
    python scripts/01_run_baseline.py "${REGRESSION_ARGS[@]}"
) || PREPATCH_OK=0
(
  cd "${WT}"
  run_and_classify "pre-patch sliced" "${OUT}/prepatch_sliced.log" \
    python scripts/01b_run_baseline_sliced.py "${REGRESSION_ARGS[@]}"
) || PREPATCH_OK=0

if [ "$PREPATCH_OK" -eq 1 ]; then
  cp "${WT}/results/baseline/summary.json"          "${OUT}/prepatch_annual_summary.json"
  cp "${WT}/results/baseline/timeseries.csv"        "${OUT}/prepatch_annual_ts.csv"
  cp "${WT}/results/baseline_sliced/summary.json"   "${OUT}/prepatch_sliced_summary.json"
  cp "${WT}/results/baseline_sliced/timeseries.csv" "${OUT}/prepatch_sliced_ts.csv"
  echo "  done"
else
  echo "  SKIPPED file copy: pre-patch run(s) did not PASS -- see logs above."
  echo "  NOTE: ${PRE_REF} may predate some of the regression flags above"
  echo "        (this script cannot know in advance -- see header)."
fi

# ------------------------------------------------------------------
# 3. Post-patch run at HEAD
# ------------------------------------------------------------------
echo
echo "=== [3/4] Post-patch baselines at HEAD ==="
POSTPATCH_OK=1
run_and_classify "post-patch annual" "${OUT}/postpatch_annual.log" \
  python scripts/01_run_baseline.py "${REGRESSION_ARGS[@]}" || POSTPATCH_OK=0
run_and_classify "post-patch sliced" "${OUT}/postpatch_sliced.log" \
  python scripts/01b_run_baseline_sliced.py "${REGRESSION_ARGS[@]}" || POSTPATCH_OK=0

if [ "$POSTPATCH_OK" -eq 1 ]; then
  cp results/baseline/summary.json          "${OUT}/postpatch_annual_summary.json"
  cp results/baseline/timeseries.csv        "${OUT}/postpatch_annual_ts.csv"
  cp results/baseline_sliced/summary.json   "${OUT}/postpatch_sliced_summary.json"
  cp results/baseline_sliced/timeseries.csv "${OUT}/postpatch_sliced_ts.csv"
  echo "  done"
else
  echo "  SKIPPED file copy: post-patch run(s) did not PASS -- see logs above."
fi

# ------------------------------------------------------------------
# 4. Compare -- only if both sides actually produced output
# ------------------------------------------------------------------
echo
echo "=== [4/4] Regression comparison ==="
if [ "$PREPATCH_OK" -eq 1 ] && [ "$POSTPATCH_OK" -eq 1 ]; then
  python scripts/_compare_2_6.py "${OUT}"
else
  echo "SKIPPED: cannot compare -- pre-patch_ok=${PREPATCH_OK} post-patch_ok=${POSTPATCH_OK}."
  echo "This is a reported outcome, not a silent pass. Fix the failing side and re-run."
  exit 1
fi
