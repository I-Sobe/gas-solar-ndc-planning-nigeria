#!/usr/bin/env bash
# ============================================================================
# TASK 52 section 2.1 driver -- storage_capex_case measurement (register L90).
# ============================================================================
# Default: ONE solve, storage_capex_case=declining_nrel (the central case since
# TASK 53) at the anchor configuration (unconstrained, arrears, central arms from
# run_phase_b.sh's C_* block); it must reproduce the post-epoch anchor
# (anchor_check in its summary). --with-flat adds the 'flat' cell, the declared
# sensitivity reported in section 3.7; TASK 53 section 5 runs both. Output:
# results/storage_capex/summary__<hash8>.json. Zeus runs this.
#
# Usage: bash scripts/run_storage_capex.sh [--dry-run] [--allow-dirty] [--with-flat]
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
mkdir -p logs/storage_capex

_cmd () {
  CMD=(python3 scripts/54_run_storage_capex_case.py
    --peak-case "$PEAK" --reserve-margin "$RESERVE" --storage-cc-case "$STORAGE_CC"
    --gas-build-case "$GAS_BUILD" --new-gas-deliverability "$NEW_GAS_DELIV"
    --gas-availability-case "$GAS_AVAIL" --gas-retirement-case "$GAS_RETIRE"
    --gas-capex-case "$GAS_CAPEX" --ccgt-efficiency-case "$CCGT"
    --capital-recovery-case "$CR" --dscr-case "$DSCR" --us-inflation "$INFL"
    --capital-treatment-case "$TREAT" --concessional-envelope-case "$ENV_"
    --gas-deliverability-case "$GAS_DELIV" --conversion-case "$CONV"
    --hydro-case "$HYDRO" --tariff-case "$TARIFF" --om-case "$OM"
    --storage-capex-case "$1" "${@:2}")
}
CASES=(declining_nrel); [ "$WITH_FLAT" -eq 1 ] && CASES=(flat declining_nrel)

PATHS=()
for c in "${CASES[@]}"; do
  _cmd "$c" --print-path-only; p=$("${CMD[@]}" 2>/dev/null | tail -1)
  [ -n "$p" ] || { echo "ABORT: no output path for $c"; exit 1; }; PATHS+=("$p")
done
[ "$(printf '%s\n' "${PATHS[@]}" | sort -u | wc -l)" -eq "${#CASES[@]}" ] || { echo "ABORT: output paths collide"; exit 1; }
printf 'output path: %s\n' "${PATHS[@]}"

for c in "${CASES[@]}"; do
  _cmd "$c"; log="logs/storage_capex/${c}.log"
  if [ "$DRY_RUN" -eq 1 ]; then printf '  [%s] %s\n' "$c" "${CMD[*]}"; continue; fi
  {
    echo "UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)  HEAD: $(git rev-parse HEAD)"
    echo "invocation: ${CMD[*]}"
    ( time "${CMD[@]}" ) 2>&1
  } 2>&1 | tee "$log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ABORT: $c failed. See $log."; exit 1; }
done
