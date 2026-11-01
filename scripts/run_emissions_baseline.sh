#!/usr/bin/env bash
# ============================================================================
# TASK 53 section 4 driver -- rebuild the emissions baseline BEFORE the re-run.
# ============================================================================
# E_base (15.745 MtCO2/yr) is an input of the NDC cap trajectory and every capped
# scenario inherits it; a re-run that went first would use a stale cap. In order:
#   0. snapshot the committed cap CSV and the bare baseline diagnostics aside
#   1. 01_run_baseline.py at the central arms with --conversion-case conv_none
#      (sliced since the A5 port; the arm set that produced the current value;
#      the corrected model: arrears + declining storage)               [1 solve]
#   2. copy its diagnostics to the bare path 00_build_emissions_cap.py reads
#   3. python3 scripts/00_build_emissions_cap.py: OVERWRITES the committed
#      data/cost/processed/emissions_cap.csv (review `git diff`, commit it alone)
#      and print old vs new E_base and the largest relative cap change
#   4. scripts/57_emissions_baseline_builder_agreement.py: section 3.1.8's
#      builder-inheritance test, annual vs sliced on the corrected model, E_base
#      and all 150 cap values                                          [2 solves]
# 3 solves, minutes. Nothing here commits. Zeus runs this, then reports the
# step-3 and step-4 figures (section 3.1.8's sentence depends on step 4's).
#
# Usage: bash scripts/run_emissions_baseline.sh [--dry-run] [--allow-dirty]
# ============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1
DRY_RUN=0; ALLOW_DIRTY=0; WITH_FLAT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;; --allow-dirty) ALLOW_DIRTY=1 ;; --with-flat) WITH_FLAT=1 ;;
    *) echo "unknown flag: $1"; exit 2 ;;
  esac; shift
done
PHASE_B=scripts/run_phase_b.sh
_const () {
  local v
  v=$(grep -E "^$1=" "$PHASE_B" | head -1 | sed -E 's/^[A-Z_]+="?([^"]*)"?$/\1/')
  [ -n "$v" ] || { echo "ABORT: $1 not found in $PHASE_B" >&2; exit 1; }
  printf '%s' "$v"
}
PEAK=$(_const C_PEAK); RESERVE=$(_const C_RESERVE); STORAGE_CC=$(_const C_STORAGE_CC)
GAS_BUILD=$(_const C_GAS_BUILD); NEW_GAS_DELIV=$(_const C_NEW_GAS_DELIV)
GAS_AVAIL=$(_const C_GAS_AVAIL); GAS_RETIRE=$(_const C_GAS_RETIRE)
GAS_CAPEX=$(_const C_GAS_CAPEX); CCGT=$(_const C_CCGT); CR=$(_const C_CR)
DSCR=$(_const C_DSCR); INFL=$(_const C_INFL); TREAT=$(_const C_TREAT)
ENV_=$(_const C_ENV); GAS_DELIV=$(_const C_GAS_DELIV); CONV=$(_const C_CONV)
HYDRO=$(_const C_HYDRO); TARIFF=$(_const C_TARIFF); OM=$(_const C_OM)

if [ "$DRY_RUN" -eq 0 ] && [ "$ALLOW_DIRTY" -eq 0 ]; then
  if [ -n "$(git status --porcelain --untracked-files=no | grep -v '^A ')" ]; then
    echo "ABORT: tracked tree is dirty. Commit or stash, or pass --allow-dirty."; exit 1
  fi
fi
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=logs/emissions_baseline; mkdir -p "$LOG"
PRE=results/_pre_task53_emissions_$STAMP
CAP=data/cost/processed/emissions_cap.csv
ARMS=(--peak-case "$PEAK" --storage-cc-case "$STORAGE_CC"
  --gas-build-case "$GAS_BUILD" --new-gas-deliverability "$NEW_GAS_DELIV"
  --gas-availability-case "$GAS_AVAIL" --gas-retirement-case "$GAS_RETIRE"
  --gas-capex-case "$GAS_CAPEX" --ccgt-efficiency-case "$CCGT"
  --capital-recovery-case "$CR" --dscr-case "$DSCR" --us-inflation "$INFL"
  --capital-treatment-case "$TREAT" --concessional-envelope-case "$ENV_"
  --gas-deliverability-case "$GAS_DELIV" --conversion-case conv_none
  --hydro-case "$HYDRO" --tariff-case "$TARIFF" --om-case "$OM")
CMD1=(python3 scripts/01_run_baseline.py "${ARMS[@]}")                       # 01_ has no --reserve-margin flag
CMD4=(python3 scripts/57_emissions_baseline_builder_agreement.py "${ARMS[@]}" --reserve-margin "$RESERVE")
if [ "$DRY_RUN" -eq 1 ]; then
  echo "step 0: snapshot $CAP and results/baseline/diagnostics.json into $PRE"
  echo "step 1: ${CMD1[*]}"
  echo "step 2: cp <diagnostics__hash8.json printed by step 1> results/baseline/diagnostics.json"
  echo "step 3: python3 scripts/00_build_emissions_cap.py        (overwrites $CAP)"
  echo "step 4: ${CMD4[*]}"
  exit 0
fi
set -e
mkdir -p "$PRE"
cp "$CAP" "$PRE/emissions_cap.csv"; md5sum "$CAP" | tee "$PRE/emissions_cap.md5"
[ -f results/baseline/diagnostics.json ] && cp results/baseline/diagnostics.json "$PRE/diagnostics.json"
echo "== step 1: conv_none baseline (corrected model)"
"${CMD1[@]}" 2>&1 | tee "$LOG/step1_$STAMP.log"
DIAG=$(grep -oE 'Saved diagnostics: [^ ]*diagnostics__[0-9a-f]+\.json' "$LOG/step1_$STAMP.log" | tail -1 | sed 's/^Saved diagnostics: //')
[ -n "$DIAG" ] && [ -f "$DIAG" ] || { echo "ABORT: could not find step 1's diagnostics file in its log."; exit 1; }
echo "== step 2: bare path"
cp "$DIAG" results/baseline/diagnostics.json
echo "$DIAG  (copied $(date -u +%FT%TZ) by run_emissions_baseline.sh)" > results/baseline/diagnostics.json.SOURCE
echo "== step 3: rebuild the cap trajectory (overwrites $CAP)"
python3 scripts/00_build_emissions_cap.py 2>&1 | tee "$LOG/step3_$STAMP.log" | tail -15
python3 - "$PRE" "$CAP" "$PRE/diagnostics.json" results/baseline/diagnostics.json <<'PY' | tee "$LOG/step3_compare_$STAMP.txt"
import json, sys, pandas as pd, os
pre, cap, old_d, new_d = sys.argv[1:5]
old = pd.read_csv(os.path.join(pre, "emissions_cap.csv")); new = pd.read_csv(cap)
key = [c for c in old.columns if c != "cap_tco2"]
mg = old.merge(new, on=key, suffixes=("_old", "_new"))
den = mg[["cap_tco2_old", "cap_tco2_new"]].abs().max(axis=1).clip(lower=1e-300)
rel = ((mg.cap_tco2_old - mg.cap_tco2_new).abs() / den).where(~((mg.cap_tco2_old == 0) & (mg.cap_tco2_new == 0)), 0.0)
EF = 182300.0
def eb(p):
    d = json.load(open(p)); return d["fleet_gas_to_power_twh_th_by_year"]["2025"] * EF / 1e6
print(f"rows compared: {len(mg)} of {len(old)} old / {len(new)} new")
if os.path.exists(old_d): print(f"E_base old {eb(old_d):.10f}  new {eb(new_d):.10f}  rel diff {abs(eb(old_d)-eb(new_d))/eb(old_d):.3e}")
print(f"largest relative cap change {rel.max():.3e}; largest absolute {(mg.cap_tco2_old-mg.cap_tco2_new).abs().max():.4f} tCO2")
PY
git diff --stat -- "$CAP" || true
echo "== step 4: builder-inheritance test (annual vs sliced), 2 solves"
"${CMD4[@]}" 2>&1 | tee "$LOG/step4_$STAMP.log" | tail -16
echo
echo "DONE. Report: $LOG/step3_compare_$STAMP.txt and the agreement.json printed above."
echo "Review 'git diff $CAP' and commit it ALONE; then the re-run (run_phase_b.sh)."
