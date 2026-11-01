#!/usr/bin/env bash
# ============================================================================
# TASK 51 M1 driver -- the capital-window measurement (6 solves).
# ============================================================================
# Cells (capital_case x capital_window_case), in this order:
#   1 unconstrained advance (regression: must equal the central anchor)
#   2 unconstrained arrears   3 unconstrained due_crf
#   4 moderate advance        5 moderate arrears        6 moderate due_crf
# Each cell is one invocation of scripts/51_run_capital_window_sensitivity.py,
# written to results/capital_window/summary__<hash8>.json. The central arm
# constants are read from scripts/run_phase_b.sh's own C_* block (grep, not
# retyped); if any is missing the driver aborts. Aborts on a dirty tracked
# tree unless --allow-dirty (so the log header's HEAD describes the code that
# ran). No deletion, no commit, no tag. Zeus runs this.
#
# Usage: bash scripts/run_capital_window.sh [--dry-run] [--allow-dirty] [--only N]
#
# After the six cells: python scripts/51_report_capital_window.py
# ============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1

DRY_RUN=0; ALLOW_DIRTY=0; ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --allow-dirty) ALLOW_DIRTY=1 ;;
    --only) shift; ONLY="$1" ;;
    *) echo "unknown flag: $1"; exit 2 ;;
  esac
  shift
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
    echo "ABORT: tracked tree is dirty (staged-add review files are ignored)."
    git status --porcelain --untracked-files=no | grep -v '^A '
    echo "Commit or stash, or pass --allow-dirty."
    exit 1
  fi
fi

mkdir -p logs/capital_window

CELLS=("unconstrained advance" "unconstrained arrears" "unconstrained due_crf"
       "moderate advance" "moderate arrears" "moderate due_crf")

_cmd () {   # _cmd <capital_case> <window> [extra flags...]  -> fills the array CMD
  local cc="$1" cw="$2"; shift 2
  CMD=(python3 scripts/51_run_capital_window_sensitivity.py
    --peak-case "$PEAK" --reserve-margin "$RESERVE" --storage-cc-case "$STORAGE_CC"
    --gas-build-case "$GAS_BUILD" --new-gas-deliverability "$NEW_GAS_DELIV"
    --gas-availability-case "$GAS_AVAIL" --gas-retirement-case "$GAS_RETIRE"
    --gas-capex-case "$GAS_CAPEX" --ccgt-efficiency-case "$CCGT"
    --capital-recovery-case "$CR" --dscr-case "$DSCR" --us-inflation "$INFL"
    --capital-treatment-case "$TREAT" --concessional-envelope-case "$ENV_"
    --gas-deliverability-case "$GAS_DELIV" --conversion-case "$CONV"
    --hydro-case "$HYDRO" --tariff-case "$TARIFF" --om-case "$OM"
    --capital-case "$cc" --capital-window-case "$cw" "$@")
}

# TASK 51 amendment 1.2: before any solve, build the six output paths and
# assert they are six DISTINCT paths (the om_case collision of TASK 48 must
# not recur). No build, no solve: --print-path-only exits right after argparse.
PATHS=()
for c in "${CELLS[@]}"; do
  set -- $c
  _cmd "$1" "$2" --print-path-only
  p=$("${CMD[@]}" 2>/dev/null | tail -1)
  [ -n "$p" ] || { echo "ABORT: could not compute the output path for cell $c"; exit 1; }
  PATHS+=("$p")
done
NDIST=$(printf '%s\n' "${PATHS[@]}" | sort -u | wc -l)
if [ "$NDIST" -ne 6 ]; then
  echo "ABORT: the six cells map to $NDIST distinct output paths, not 6:"; printf '  %s\n' "${PATHS[@]}"; exit 1
fi
echo "Pre-solve check ok: six distinct output paths:"; printf '  %s\n' "${PATHS[@]}"

cell () {
  local n="$1" cc="$2" cw="$3"
  [ -n "$ONLY" ] && [ "$ONLY" != "$n" ] && return 0
  local label="${n}_${cc}_${cw}" log="logs/capital_window/${n}_${cc}_${cw}.log"
  _cmd "$cc" "$cw"
  local cmd=("${CMD[@]}")
  if [ "$DRY_RUN" -eq 1 ]; then printf '  [%s] %s\n' "$label" "${cmd[*]}"; return 0; fi
  {
    echo "================ capital-window driver run header ================"
    echo "UTC timestamp : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Branch        : $(git rev-parse --abbrev-ref HEAD)"
    echo "HEAD          : $(git rev-parse HEAD)"
    echo "Working tree  : $(git status --porcelain --untracked-files=no | grep -vc '^A ' | sed 's/^0$/clean/')"
    echo "cell          : $label"
    echo "invocation    : ${cmd[*]}"
    echo "=================================================================="
    ( time "${cmd[@]}" ) 2>&1
  } 2>&1 | tee "$log"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "ABORT: cell $label failed. See $log."; exit 1; fi
}

cell 1 unconstrained advance
cell 2 unconstrained arrears
cell 3 unconstrained due_crf
cell 4 moderate advance
cell 5 moderate arrears
cell 6 moderate due_crf
echo "Done. Next: python scripts/51_report_capital_window.py"
