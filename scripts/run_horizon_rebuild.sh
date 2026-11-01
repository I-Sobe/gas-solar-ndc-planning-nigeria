#!/usr/bin/env bash
# ============================================================================
# TASK 52 B-3 driver -- horizon / in-horizon-rebuild isolation (8 solves).
# Cells (end_year, storage_life): (2045,15) (2050,15) (2051,15) (2052,15) (2053,15)
# (2054,15) | (2045,30) (2054,30). Each is one invocation of
# scripts/53_run_horizon_rebuild_isolation.py -> results/horizon_rebuild/summary__<hash8>.json.
# Central arm constants are grepped from scripts/run_phase_b.sh's C_* block.
# Aborts on a dirty tracked tree unless --allow-dirty; asserts eight distinct output
# paths before solving. Zeus runs this. Usage: bash scripts/run_horizon_rebuild.sh [--dry-run] [--allow-dirty] [--only N]
# Then: python scripts/53_report_horizon_rebuild.py
# NOTE: the (2054,15) cell is the central anchor configuration; its result is a
# citable-runner reproduction of the anchor (anchor_check in the summary).
# ============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1
DRY_RUN=0; ALLOW_DIRTY=0; ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;; --allow-dirty) ALLOW_DIRTY=1 ;; --only) shift; ONLY="$1" ;;
    *) echo "unknown flag: $1"; exit 2 ;;
  esac; shift
done
PHASE_B=scripts/run_phase_b.sh
_const () { local v; v=$(grep -E "^$1=" "$PHASE_B" | head -1 | sed -E 's/^[A-Z_]+="?([^"]*)"?$/\1/'); [ -n "$v" ] || { echo "ABORT: $1 not found in $PHASE_B" >&2; exit 1; }; printf '%s' "$v"; }
PEAK=$(_const C_PEAK); RESERVE=$(_const C_RESERVE); STORAGE_CC=$(_const C_STORAGE_CC)
GAS_BUILD=$(_const C_GAS_BUILD); NEW_GAS_DELIV=$(_const C_NEW_GAS_DELIV); GAS_AVAIL=$(_const C_GAS_AVAIL)
GAS_RETIRE=$(_const C_GAS_RETIRE); GAS_CAPEX=$(_const C_GAS_CAPEX); CCGT=$(_const C_CCGT); CR=$(_const C_CR)
DSCR=$(_const C_DSCR); INFL=$(_const C_INFL); TREAT=$(_const C_TREAT); ENV_=$(_const C_ENV)
GAS_DELIV=$(_const C_GAS_DELIV); CONV=$(_const C_CONV); HYDRO=$(_const C_HYDRO); TARIFF=$(_const C_TARIFF); OM=$(_const C_OM)
if [ "$DRY_RUN" -eq 0 ] && [ "$ALLOW_DIRTY" -eq 0 ]; then
  if [ -n "$(git status --porcelain --untracked-files=no | grep -v '^A ')" ]; then
    echo "ABORT: tracked tree is dirty (staged-add review files are ignored)."; git status --porcelain --untracked-files=no | grep -v '^A '; exit 1
  fi
fi
mkdir -p logs/horizon_rebuild
CELLS=("2045 15" "2050 15" "2051 15" "2052 15" "2053 15" "2054 15" "2045 30" "2054 30")
_cmd () {
  local ey="$1" sl="$2"; shift 2
  CMD=(python3 scripts/53_run_horizon_rebuild_isolation.py
    --peak-case "$PEAK" --reserve-margin "$RESERVE" --storage-cc-case "$STORAGE_CC"
    --gas-build-case "$GAS_BUILD" --new-gas-deliverability "$NEW_GAS_DELIV"
    --gas-availability-case "$GAS_AVAIL" --gas-retirement-case "$GAS_RETIRE"
    --gas-capex-case "$GAS_CAPEX" --ccgt-efficiency-case "$CCGT"
    --capital-recovery-case "$CR" --dscr-case "$DSCR" --us-inflation "$INFL"
    --capital-treatment-case "$TREAT" --concessional-envelope-case "$ENV_"
    --gas-deliverability-case "$GAS_DELIV" --conversion-case "$CONV"
    --hydro-case "$HYDRO" --tariff-case "$TARIFF" --om-case "$OM"
    --end-year "$ey" --storage-life "$sl" "$@")
}
PATHS=()
for c in "${CELLS[@]}"; do set -- $c; _cmd "$1" "$2" --print-path-only; p=$("${CMD[@]}" 2>/dev/null | tail -1); [ -n "$p" ] || { echo "ABORT: no path for cell $c"; exit 1; }; PATHS+=("$p"); done
NDIST=$(printf '%s\n' "${PATHS[@]}" | sort -u | wc -l)
[ "$NDIST" -eq 8 ] || { echo "ABORT: ${#CELLS[@]} cells map to $NDIST distinct paths"; printf '  %s\n' "${PATHS[@]}"; exit 1; }
echo "Pre-solve check ok: eight distinct output paths:"; printf '  %s\n' "${PATHS[@]}"
n=0
for c in "${CELLS[@]}"; do
  n=$((n+1)); set -- $c; [ -n "$ONLY" ] && [ "$ONLY" != "$n" ] && continue
  _cmd "$1" "$2"; label="${n}_end$1_life$2"; log="logs/horizon_rebuild/$label.log"
  if [ "$DRY_RUN" -eq 1 ]; then printf '  [%s] %s\n' "$label" "${CMD[*]}"; continue; fi
  { echo "=== horizon-rebuild header ==="; echo "UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "HEAD: $(git rev-parse HEAD)"; echo "cell: $label"; echo "invocation: ${CMD[*]}"; echo "=============================="; ( time "${CMD[@]}" ) 2>&1; } 2>&1 | tee "$log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ABORT: cell $label failed. See $log."; exit 1; }
done
echo "Done. Next: python scripts/53_report_horizon_rebuild.py"
