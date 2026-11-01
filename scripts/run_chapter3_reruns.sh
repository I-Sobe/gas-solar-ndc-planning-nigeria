#!/usr/bin/env bash
# ============================================================================
# Chapter 3 re-runs driver (TASK 54 follow-up) -- NOT YET RUN.
# ============================================================================
#
# PURPOSE
#   Re-runs on the corrected model (TASK 53 fix epoch: arrears window,
#   declining_nrel storage capex, arm-keyed output names), for results whose
#   pre-epoch files are gone or never had a driver:
#
#     Cell 1  (3 solves)  27_run_hydro_minflow_sensitivity.py --om-case om_central
#                         -- hydro_min_flow_fraction 0.30 / 0.50 / 0.70 (the
#                         values register L20 records, fixed inside the runner).
#                         This is the configuration M-3 rests on going forward.
#     Cell 2  (3 solves)  the same, --om-case om_none: reproduces L20's
#                         configuration (its objectives were measured pre-O&M)
#                         on the corrected model. Both bands are reported at
#                         the end of the cell.
#     Cell 3  (5 solves)  the social-discount-rate sweep, r_s = 4, 6, 8, 10,
#                         12%, at the central arm set, through
#                         01b_run_baseline_sliced.py --social-discount-rate.
#
#   Both runners now carry the varied parameter in their output key (G7-4):
#   27_ writes results/hydro_minflow/arms_<hash8>/ (om_case is a flag and part
#   of the key); 01b's tag gains __social_discount_rate=<r> whenever the flag is
#   GIVEN (including 0.08), so no cell can overwrite the central anchor and
#   nothing needs snapshotting or restoring. Omitting the flag leaves every
#   stored hash unchanged (central stays 6046f759).
#
#   (c) -- does run_phase_c.sh need a change -- is a separate task block; see
#   docs/task_log_54_rederivation.md.
#
# WHAT PRODUCED results/_sdr_sweep_v3/ (no runner name was recorded)
#   Nothing in the repository. It was an interactive shell loop (kept only in
#   ~/.bash_history, entered twice, 2026-09-24/25): for each r_s it ran
#     sed -i 's/"social_discount_rate": 0.08,/"social_discount_rate": $r,/' src/scenarios.py
#     python3 scripts/01b_run_baseline_sliced.py <central arms> > central_r$r.log
#     git checkout src/scenarios.py
#     cp the two newest results/baseline_sliced/*.json  results/_sdr_sweep*/r${r}_<name>
#   Order 0.04 0.06 0.10 0.12 0.08, ending on 0.08 because r_s was NOT in the
#   output hash, so every run overwrote the central anchor and the last had to
#   restore it (a crash mid-loop left the anchor corrupted: the G7-4 gap).
#   v3 kept only the five logs; the copied summaries survive only for v1.
#
# CONVENTIONS (matched from scripts/run_phase_c.sh and run_phase_b.sh)
#   - The central C_* arm constants are EXTRACTED from run_phase_b.sh at
#     start-up (grep over that file); never retyped. Abort if none found.
#   - 27_'s central configuration is a dict inside the runner (it has no
#     per-arm flags). The driver CHECKS that dict against the C_* values -- all
#     19 arms, om_case included -- and aborts on any disagreement. Cell 2 then
#     overrides om_case on the command line, which is the arm, not a mismatch.
#   - Logs under logs/chapter3_reruns/, header block, free -h before/after,
#     time, tee, abort on non-zero exit. --dry-run, --only N, --from/--to.
#   - Aborts on a dirty TRACKED tree (untracked files are ignored); there is
#     no --allow-dirty. (--dry-run still prints the plan on a dirty tree.)
#   - Cell 1 first copies the existing results/hydro_minflow/ to
#     results/_pre_chapter3_reruns_<UTC>/ (the flat files TASK 53's smoke run
#     left); the arm-keyed directories are new paths.
#   - Cell 3 checks that the r_s = 8% point reproduces the stored central
#     objective (summary__6046f759.json) to 1e-6, and that r_s changes the
#     objective and not the build (L59's claim), both read from the new files.
#
# USAGE
#   bash scripts/run_chapter3_reruns.sh --dry-run
#   bash scripts/run_chapter3_reruns.sh --only 1      # or 2, or 3
#   bash scripts/run_chapter3_reruns.sh               # all three cells
# ============================================================================

set -u
cd "$(dirname "$0")/.."

PHASE_B="scripts/run_phase_b.sh"
[ -f "$PHASE_B" ] || { echo "ABORT: $PHASE_B not found."; exit 1; }

LOGDIR="logs/chapter3_reruns"
SDR_VALUES="0.04 0.06 0.08 0.10 0.12"

DRY_RUN=0
ONLY=""
FROM=""
TO=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --only)    ONLY="${2:?--only needs a cell number}"; shift 2 ;;
    --from)    FROM="${2:?--from needs a cell number}"; shift 2 ;;
    --to)      TO="${2:?--to needs a cell number}"; shift 2 ;;
    *) echo "ABORT: unknown flag: $1"; exit 1 ;;
  esac
done
if [ -n "$ONLY" ] && { [ -n "$FROM" ] || [ -n "$TO" ]; }; then
  echo "ABORT: --only cannot be combined with --from/--to."; exit 1
fi
START=${FROM:-1}; END=${TO:-3}
if [ -n "$ONLY" ]; then START=$ONLY; END=$ONLY; fi
case "$START$END" in *[!123]*) echo "ABORT: cells are 1, 2 and 3."; exit 1 ;; esac

# ---- header block -------------------------------------------------------
UTC_NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
UTC_TAG=${UTC_NOW//[:-]/}
BRANCH=$(git branch --show-current)
HEAD_SHA=$(git rev-parse HEAD)
DESCRIBE=$(git describe --tags 2>/dev/null || echo "(no tag reachable)")
DIRTY_TRACKED=$(git status --porcelain --untracked-files=no)

HEADER_TEXT=$(cat <<EOF
============ Chapter 3 re-runs driver run header ============
UTC timestamp : $UTC_NOW
Branch        : $BRANCH
HEAD          : $HEAD_SHA
git describe  : $DESCRIBE
Tracked tree  : $([ -z "$DIRTY_TRACKED" ] && echo "clean" || echo "DIRTY")
==============================================================
EOF
)
echo "$HEADER_TEXT"

if [ -n "$DIRTY_TRACKED" ]; then
  if [ "$DRY_RUN" -eq 0 ]; then
    echo
    echo "!! ABORT: a tracked file is modified. These results must be attributable"
    echo "!! to a committed tree. Commit or stash (no --allow-dirty exists)."
    echo
    echo "$DIRTY_TRACKED"
    exit 1
  fi
  echo "NOTE (dry run): tracked tree is dirty; a real run would abort here."
fi

# ---- central arm string, taken from Phase B (never retyped) ---------------
C_LINES=$(grep -E '^C_[A-Z_]+="[^"]*"$' "$PHASE_B")
[ -n "$C_LINES" ] || { echo "ABORT: no C_* central constants found in $PHASE_B."; exit 1; }
eval "$C_LINES"
for v in C_PEAK C_STORAGE_CC C_GAS_BUILD C_NEW_GAS_DELIV C_GAS_AVAIL C_GAS_RETIRE C_GAS_CAPEX \
         C_CCGT C_CR C_DSCR C_INFL C_TREAT C_ENV C_GAS_DELIV C_CONV C_HYDRO C_TARIFF C_OM; do
  [ -n "${!v:-}" ] || { echo "ABORT: $PHASE_B did not define $v."; exit 1; }
done

# 01b's central argument vector: the same flag set run_phase_b.sh's
# central_diag_hash8 passes (01b hardcodes reserve_margin; it takes no such flag).
CENTRAL_01B_ARGS=(
  --peak-case "$C_PEAK" --storage-cc-case "$C_STORAGE_CC" --gas-build-case "$C_GAS_BUILD"
  --new-gas-deliverability "$C_NEW_GAS_DELIV" --gas-availability-case "$C_GAS_AVAIL"
  --gas-retirement-case "$C_GAS_RETIRE" --gas-capex-case "$C_GAS_CAPEX"
  --ccgt-efficiency-case "$C_CCGT" --capital-recovery-case "$C_CR" --dscr-case "$C_DSCR"
  --us-inflation "$C_INFL" --capital-treatment-case "$C_TREAT"
  --concessional-envelope-case "$C_ENV" --gas-deliverability-case "$C_GAS_DELIV"
  --conversion-case "$C_CONV" --hydro-case "$C_HYDRO" --tariff-case "$C_TARIFF"
  --om-case "$C_OM"
)

# ---- execution primitive ---------------------------------------------------
_execute () {
  local log_label="$1"; shift
  local log="$LOGDIR/${log_label}.log"

  for a in "$@"; do
    if [ "$a" = "--reserve-margin" ]; then
      echo "ABORT: $* must never receive --reserve-margin (01b and 27_ hardcode nirp_spinning)."; exit 1
    fi
  done

  if [ "$DRY_RUN" -eq 1 ]; then
    printf '  [%s] ' "$log_label"; printf '%q ' "$@"; printf '\n'
    return 0
  fi
  mkdir -p "$LOGDIR"
  {
    echo "$HEADER_TEXT"; echo "item log    : $log_label"; echo
    echo "=== free -h (before) ==="; free -h; echo
    echo "=== invocation ==="; printf '%q ' "$@"; printf '\n'; echo
  } > "$log"
  { time "$@"; } 2>&1 | tee -a "$log"
  local code=${PIPESTATUS[0]}
  { echo; echo "=== free -h (after) ==="; free -h; echo; echo "=== exit status: $code ==="; } >> "$log"
  if [ "$code" -ne 0 ]; then
    echo; echo "ABORT: [$log_label] exited $code. See $log."; exit 1
  fi
  printf '  [%s] ok -> %s\n' "$log_label" "$log"
}

banner () { echo; echo "════════════════════════════════════════════════════"; echo "$1"; echo "════════════════════════════════════════════════════"; }
should_run () { [ "$1" -ge "$START" ] && [ "$1" -le "$END" ]; }

# 01b's own hash8 for the central arms (no build, no solve), as Phase B derives it
central_hash8 () { python3 scripts/01b_run_baseline_sliced.py "${CENTRAL_01B_ARGS[@]}" --print-hash8; }

# ============================================================================
# Check 27_'s hard-coded central configuration against Phase B's C_* values
# (all 18 arms, om_case included). Run once, before any hydro cell.
# ============================================================================
check_27_arms () {
  local want="peak_multiple_case=$C_PEAK storage_cc_case=$C_STORAGE_CC gas_build_case=$C_GAS_BUILD \
new_gas_deliverability=$C_NEW_GAS_DELIV gas_availability_case=$C_GAS_AVAIL gas_retirement_case=$C_GAS_RETIRE \
gas_capex_case=$C_GAS_CAPEX ccgt_efficiency_case=$C_CCGT capital_recovery_case=$C_CR dscr_case=$C_DSCR \
us_inflation=$C_INFL capital_treatment_case=$C_TREAT concessional_envelope_case=$C_ENV \
gas_deliverability_case=$C_GAS_DELIV conversion_case=$C_CONV hydro_case=$C_HYDRO tariff_case=$C_TARIFF \
om_case=$C_OM"
  python3 - $want <<'PY' || exit 1
import ast, sys
want = dict(a.split("=", 1) for a in sys.argv[1:])
tree = ast.parse(open("scripts/27_run_hydro_minflow_sensitivity.py").read())
args = None
for n in tree.body:
    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CENTRAL_ARGS":
        args = {}
        for k in n.value.keywords:
            try: args[k.arg] = ast.literal_eval(k.value)
            except ValueError: pass          # e.g. end_year=MODEL_END_YEAR: not an arm
if args is None:
    print("ABORT: CENTRAL_ARGS not found in 27_."); sys.exit(1)
bad = []
for k, v in want.items():
    if k not in args: bad.append(f"{k}: absent from 27_ CENTRAL_ARGS")
    elif k == "us_inflation":
        if abs(float(args[k]) - float(v)) > 1e-12: bad.append(f"{k}: 27_={args[k]} phase_b={v}")
    elif str(args[k]) != v: bad.append(f"{k}: 27_={args[k]} phase_b={v}")
if bad:
    print("ABORT: 27_ CENTRAL_ARGS disagrees with run_phase_b.sh's central arm string:")
    print("\n".join("  " + b for b in bad)); sys.exit(1)
print(f"  27_ CENTRAL_ARGS agrees with run_phase_b.sh on all {len(want)} arms (om_case included).")
PY
}

_snapshot_flat_hydro () {
  local old="results/hydro_minflow"
  local snap="results/_pre_chapter3_reruns_${UTC_TAG}"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "  [snapshot] would copy the flat files of $old -> $snap/hydro_minflow/ before the run (if present)"
  elif [ -d "$old" ] && [ ! -d "$snap/hydro_minflow" ]; then
    mkdir -p "$snap/hydro_minflow" && find "$old" -maxdepth 1 -type f -exec cp -a {} "$snap/hydro_minflow/" \; \
      && echo "  existing flat files of $old preserved in $snap/hydro_minflow/"
  fi
}

# one hydro cell: $1 label, $2 om_case
_hydro_cell () {
  local label="$1" om="$2"
  check_27_arms
  _snapshot_flat_hydro
  _execute "$label" python3 scripts/27_run_hydro_minflow_sensitivity.py --om-case "$om"
  if [ "$DRY_RUN" -eq 0 ]; then
    local dir
    dir=$(sed -n 's/^Results directory: //p' "$LOGDIR/$label.log" | tail -1)
    [ -d "$dir" ] || { echo "ABORT: could not read the results directory from $LOGDIR/$label.log."; exit 1; }
    echo "$dir" > "$LOGDIR/.hydro_dir_$om"
    echo "  results: $dir"
  fi
}

cell_1 () {
  banner "CELL 1 -- 27_run_hydro_minflow_sensitivity.py --om-case om_central (3 solves: hydro_min_flow_fraction 0.30 / 0.50 / 0.70)"
  _hydro_cell 1_hydro_minflow_om_central om_central
  if [ "$DRY_RUN" -eq 0 ]; then
    echo "CHECK: the 0.50 arm is the central configuration; the script prints whether it reproduces"
    echo "       results/baseline_sliced/summary__6046f759.json (npv_total_cost_usd) to 1e-6."
  fi
}

cell_2 () {
  banner "CELL 2 -- 27_run_hydro_minflow_sensitivity.py --om-case om_none (3 solves; L20's configuration on the corrected model)"
  _hydro_cell 2_hydro_minflow_om_none om_none
  if [ "$DRY_RUN" -eq 0 ]; then report_hydro_bands; fi
}

# Both bands, side by side, from the two arms-keyed directories (whichever exist).
report_hydro_bands () {
  python3 - <<'PY'
import json, os
print("\n  Hydro min-flow bands (thresholds fixed before any solve: objective 0.5%, storage@2045 2%)")
print("  L20 (pre-epoch, om_none): objective band 0.141%, storage@2045 band 0.000%, IMMATERIAL")
for om in ("om_central", "om_none"):
    f = f"logs/chapter3_reruns/.hydro_dir_{om}"
    if not os.path.exists(f):
        print(f"  {om:11s} no run recorded"); continue
    d = open(f).read().strip()
    v = json.load(open(f"{d}/hydro_minflow_verdict_0.30_0.50_0.70.json"))
    print(f"  {om:11s} objective band {100*v['objective_band_pct_of_central']:.3f}%   "
          f"storage@2045 band {100*v['storage_2045_band_pct_of_central']:.3f}%   "
          f"{'MATERIAL' if v['material'] else 'IMMATERIAL'}   ({d})")
PY
}

# ============================================================================
# CELL 3 -- social discount rate sweep (01b x 5, --social-discount-rate)
# ============================================================================
cell_3 () {
  banner "CELL 3 -- 01b_run_baseline_sliced.py x5 at the central arm set, --social-discount-rate 0.04 0.06 0.08 0.10 0.12"
  local r h8
  for r in $SDR_VALUES; do
    if [ "$DRY_RUN" -eq 1 ]; then
      h8="(derived)"
    else
      h8=$(python3 scripts/01b_run_baseline_sliced.py "${CENTRAL_01B_ARGS[@]}" --social-discount-rate "$r" --print-hash8 | tail -1)
      [[ "$h8" =~ ^[0-9a-f]{8}$ ]] || { echo "ABORT: could not derive the hash8 for r_s=$r."; exit 1; }
      [ "$h8" != "$(central_hash8 | tail -1)" ] || { echo "ABORT: r_s=$r maps to the central hash: the flag is not in the key."; exit 1; }
    fi
    _execute "3_sdr_r${r}" python3 scripts/01b_run_baseline_sliced.py "${CENTRAL_01B_ARGS[@]}" --social-discount-rate "$r"
    [ "$DRY_RUN" -eq 1 ] || echo "$r $h8" >> "$LOGDIR/.sdr_hashes_$UTC_TAG"
  done
  if [ "$DRY_RUN" -eq 0 ]; then
    local c8; c8=$(central_hash8 | tail -1)
    python3 - "$c8" "$LOGDIR/.sdr_hashes_$UTC_TAG" <<'PY' || { echo "ABORT: sweep checks failed (above)."; exit 1; }
import json, sys
c8, hf = sys.argv[1], sys.argv[2]
B = "results/baseline_sliced"
S = lambda h: json.load(open(f"{B}/summary__{h}.json"))
central = S(c8)["npv_total_cost_usd"]
rows = [l.split() for l in open(hf)]
ok = True
print(f"\n  stored central anchor ({c8}): {central:,.2f}")
print("  r_s     objective (bn)   solar to 2054 (MW)   storage 2054 (MWh)")
sol, sto = [], []
for r, h in rows:
    d = S(h); dv = d["decision_variables"]
    s_mw = dv["solar_cumulative_built_2054_mw"]; s_mwh = dv["storage_capacity_2054_mwh"]
    sol.append(s_mw); sto.append(s_mwh)
    print(f"  {float(r):.2f}  {d['npv_total_cost_usd']/1e9:14.3f}  {s_mw:19,.1f}  {s_mwh:18,.1f}   used={d['social_discount_rate_used']}")
    if abs(d["social_discount_rate_used"] - float(r)) > 1e-12: print("  !! rate used differs from requested"); ok = False
    if float(r) == 0.08:
        rel = abs(d["npv_total_cost_usd"] - central) / central
        print(f"  8% point vs stored central: relative difference {rel:.2e}")
        if rel > 1e-6: print("  !! the 8% point does not reproduce the stored central objective"); ok = False
print(f"  cumulative solar spread across r_s: {max(sol)-min(sol):.4f} MW; storage 2054 spread: {max(sto)-min(sto):.4f} MWh   (L59 pre-epoch: 0.0003 MW, 0)")
sys.exit(0 if ok else 1)
PY
  fi
}

# ============================================================================
echo "Chapter 3 re-runs: cells $START..$END$( [ "$DRY_RUN" -eq 1 ] && echo " (DRY RUN -- nothing will execute)" )"
should_run 1 && cell_1
should_run 2 && cell_2
should_run 3 && cell_3
echo
echo "Done$( [ "$DRY_RUN" -eq 1 ] && echo " (dry run)" )."
