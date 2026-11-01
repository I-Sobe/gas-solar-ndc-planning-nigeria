#!/usr/bin/env bash
# ============================================================================
# TASK 52 gate item 6 driver -- Sobol parameter uncertainty into a CLEAN file
# (register L77: results/parameter_uncertainty/draws_n256_seed42.jsonl holds
# three appended runs, 5,376 rows against N(d+2) = 1,792).
# ============================================================================
# TASK 53 2.7: the guards this driver used to carry are now IN the runner
# (scripts/29_run_parameter_uncertainty.py): it writes to
# results/parameter_uncertainty/arms_<hash8>/ (keyed by every arm, n_base and
# seed), ABORTS if that directory already holds anything but ARMS.json, opens the
# draws log exclusively (never appends to an existing file), and raises at the end
# if the log does not hold exactly N(d+2) = 1,792 rows. This driver only supplies
# the central arms from run_phase_b.sh's C_* block. 1,792 solves, about 15 minutes.
# The old results/parameter_uncertainty/draws_n256_seed42.jsonl is not touched.
# Zeus runs this.
#
# Usage: bash scripts/run_sobol_clean.sh [--dry-run] [--allow-dirty]
# ============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1
DRY_RUN=0; ALLOW_DIRTY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;; --allow-dirty) ALLOW_DIRTY=1 ;;
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
mkdir -p logs/sobol_clean
CMD=(python3 scripts/29_run_parameter_uncertainty.py
  --peak-case "$PEAK" --reserve-margin "$RESERVE" --storage-cc-case "$STORAGE_CC"
  --gas-build-case "$GAS_BUILD" --new-gas-deliverability "$NEW_GAS_DELIV"
  --gas-availability-case "$GAS_AVAIL" --gas-retirement-case "$GAS_RETIRE"
  --gas-capex-case "$GAS_CAPEX" --ccgt-efficiency-case "$CCGT"
  --capital-recovery-case "$CR" --dscr-case "$DSCR" --us-inflation "$INFL"
  --capital-treatment-case "$TREAT" --concessional-envelope-case "$ENV_"
  --gas-deliverability-case "$GAS_DELIV" --conversion-case "$CONV"
  --hydro-case "$HYDRO" --tariff-case "$TARIFF" --om-case "$OM"
  --n-base 256 --seed 42)
if [ "$DRY_RUN" -eq 1 ]; then echo "${CMD[*]}"; exit 0; fi
{
  echo "UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)  HEAD: $(git rev-parse HEAD)"
  echo "invocation: ${CMD[*]}"
  ( time "${CMD[@]}" ) 2>&1
} 2>&1 | tee logs/sobol_clean/run.log
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ABORT: failed. See logs/sobol_clean/run.log."; exit 1; }
