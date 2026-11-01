#!/usr/bin/env bash
# ============================================================================
# Leakage-increment driver (TASK 55 follow-up for F-31 and section 3.5.6) -- NOT YET RUN.
# ============================================================================
#
# PURPOSE
#   F-31's out-of-boundary shares are LEVELS. The quantity that corresponds to leakage is the INCREMENT over the same
#   configuration solved with no cap. All 15 FIN-3 cells apply the NDC-3 unconditional cap, so the uncapped counterpart does
#   not exist in the post-epoch results. This driver produces it, and one attribution row.
#
#   Step 0  (2 solves)  GATE. 05_run_fin3_capital_sweep.py at its DEFAULT behaviour (cap applied), cells tight:no_friction and
#                       unconstrained:no_friction. Every numeric field must equal the stored Phase B row
#                       (results/fin3/arms_<default key>/capital_sweep_results.csv) EXACTLY. The new path (--no-cap) is not used
#                       unless this passes.
#   Step 1  (13 solves) the same runner with --no-cap (NDC_CAP_SCENARIO bypassed: emissions_cap_by_year=None), the 13 DISTINCT
#                       configurations: tight, moderate, adequacy, expansion at each of no_friction / reform_target / current
#                       (12) plus unconstrained at no_friction (1).
#   Step 2  (1 solve)   F-31 attribution row C': the capped cell tight:no_friction at --conversion-case conv_none.
#
#   DUPLICATES, stated rather than solved: `unconstrained` has no budget, so its collection rate cannot enter the model;
#   unconstrained:reform_target and unconstrained:current are the SAME configuration as unconstrained:no_friction and are
#   copied from it in the report. That is 2 of the 15 cells. The capped sweep's `expansion` cells at no_friction and
#   reform_target coincide with `unconstrained` in OUTCOME (the budget is slack there), which brings the capped table to 11
#   distinct results; that equivalence is an outcome, not a property of the configuration, and is not assumed for the uncapped
#   run: the 12 budgeted cells are all solved and the report states whether the expansion cells equal the unconstrained cell.
#
#   Output keys (G7-4). --no-cap and --only-cell enter the output key (scripts/05_run_fin3_capital_sweep.py), so no run here can
#   write to the standard sweep's directory. Before any solve the driver prints the key of the standard sweep and of each run
#   below (05_ ... --print-key: no build, no solve, nothing written) and ABORTS unless all are distinct.
#
# CONVENTIONS (matched from scripts/run_phase_c.sh and run_chapter3_reruns.sh)
#   - The central C_* arm constants, NO_RESERVE_MARGIN_RUNNERS, excluded_keys_for() and run() are EXTRACTED from run_phase_b.sh
#     at start-up; this driver cannot carry a different central arm string from Phase B.
#   - Logs under logs/leakage_increment/, header block, free -h before/after, time, tee, abort on non-zero exit.
#   - Aborts on a dirty TRACKED tree; there is no --allow-dirty. --dry-run prints the plan, runs no solve.
#
# USAGE
#   bash scripts/run_leakage_increment.sh --dry-run
#   bash scripts/run_leakage_increment.sh                 # steps 0, 1, 2 (stops after step 0 if the gate fails)
#   bash scripts/run_leakage_increment.sh --only 0|1|2
# ============================================================================

set -u
cd "$(dirname "$0")/.."

PHASE_B="scripts/run_phase_b.sh"
[ -f "$PHASE_B" ] || { echo "ABORT: $PHASE_B not found."; exit 1; }
RUNNER="05_run_fin3_capital_sweep.py"
LOGDIR="logs/leakage_increment"
STD_DIR="results/fin3"

DRY_RUN=0
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --only)    ONLY="${2:?--only needs 0, 1 or 2}"; shift 2 ;;
    *) echo "ABORT: unknown flag: $1"; exit 1 ;;
  esac
done
case "$ONLY" in ""|0|1|2) ;; *) echo "ABORT: --only takes 0, 1 or 2."; exit 1 ;; esac

UTC_NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
BRANCH=$(git branch --show-current)
HEAD_SHA=$(git rev-parse HEAD)
DESCRIBE=$(git describe --tags 2>/dev/null || echo "(no tag reachable)")
DIRTY_TRACKED=$(git status --porcelain --untracked-files=no)
HEADER_TEXT=$(cat <<EOF
=========== Leakage-increment driver run header ===========
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
    echo; echo "!! ABORT: a tracked file is modified. These results are citable and must be attributable"
    echo "!! to a committed tree. Commit or stash (no --allow-dirty exists)."; echo
    echo "$DIRTY_TRACKED"; exit 1
  fi
  echo "NOTE (dry run): tracked tree is dirty; a real run would abort here."
fi

# ---- the central arm string, the exclusion table and run(), from Phase B ----------------------------------
_extract () {
  local out
  out=$(awk -v s="$1" -v e="$2" '!on && $0 ~ s { on=1 } on { print } on && $0 ~ e { exit }' "$PHASE_B")
  [ -n "$out" ] || { echo "ABORT: could not extract '$1' from $PHASE_B." >&2; return 1; }
  printf '%s\n' "$out"
}
C_LINES=$(grep -E '^C_[A-Z_]+="[^"]*"$' "$PHASE_B")
[ -n "$C_LINES" ] || { echo "ABORT: no C_* central constants found in $PHASE_B."; exit 1; }
eval "$C_LINES"
ARR_B=$(_extract '^readonly -a NO_RESERVE_MARGIN_RUNNERS=[(]' '^[)]$') || exit 1; eval "$ARR_B"
EXCL_B=$(_extract '^excluded_keys_for [(][)] [{]' '^[}]$') || exit 1; eval "$EXCL_B"
RUN_B=$(_extract '^run [(][)] [{]' '^[}]$') || exit 1; eval "$RUN_B"
for v in $(printf '%s\n' "$RUN_B" | grep -o '\$C_[A-Z_]*' | sort -u | tr -d '$'); do
  [ -n "${!v:-}" ] || { echo "ABORT: run() uses \$$v but $PHASE_B did not define it."; exit 1; }
done

# ---- execution primitive ----------------------------------------------------------------------------------------
# CAPTURE_TO: run a no-solve query (--print-key) and keep its stdout, even in a dry run.
_execute () {
  local log_label="$1"; shift
  local log="$LOGDIR/${log_label}.log"
  if [ -n "${CAPTURE_TO:-}" ]; then "$@" > "$CAPTURE_TO"; return $?; fi
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '  [%s] ' "$log_label"; printf '%q ' "$@"; printf '\n'; return 0
  fi
  mkdir -p "$LOGDIR"
  { echo "$HEADER_TEXT"; echo "item log    : $log_label"; echo; echo "=== free -h (before) ==="; free -h; echo
    echo "=== invocation ==="; printf '%q ' "$@"; printf '\n'; echo; } > "$log"
  { time "$@"; } 2>&1 | tee -a "$log"
  local code=${PIPESTATUS[0]}
  { echo; echo "=== free -h (after) ==="; free -h; echo; echo "=== exit status: $code ==="; } >> "$log"
  [ "$code" -eq 0 ] || { echo; echo "ABORT: [$log_label] exited $code. See $log."; exit 1; }
  printf '  [%s] ok -> %s\n' "$log_label" "$log"
}
banner () { echo; echo "════════════════════════════════════════════════════"; echo "$1"; echo "════════════════════════════════════════════════════"; }

# ---- the cells -----------------------------------------------------------------------------------------------------
GATE_CELLS="tight:no_friction,unconstrained:no_friction"
NOCAP_CELLS=""
for cap in tight moderate adequacy expansion; do
  for cr in no_friction reform_target current; do NOCAP_CELLS="${NOCAP_CELLS:+$NOCAP_CELLS,}$cap:$cr"; done
done
NOCAP_CELLS="$NOCAP_CELLS,unconstrained:no_friction"          # 13 distinct configurations
CPRIME_CELL="tight:no_friction"

key_of () {   # key_of <label> [run overrides and runner flags, as for run()]
  local out; out=$(mktemp)
  CAPTURE_TO="$out" run "$RUNNER" "key_$1" "${@:2}" || { rm -f "$out"; echo "ABORT: key query failed for $1."; exit 1; }
  CAPTURE_TO=""
  tail -1 "$out" | tr -d '[:space:]'; rm -f "$out"
}

# ---- before any solve: build every output key and assert they are distinct -------------------------------------------
banner "KEYS -- built before any solve (05_ --print-key: no build, no solve, nothing written)"
K_STD=$(key_of standard -- --print-key)
K_GATE=$(key_of gate -- --only-cell "$GATE_CELLS" --print-key)
K_NOCAP=$(key_of nocap -- --no-cap --only-cell "$NOCAP_CELLS" --print-key)
K_CPRIME=$(key_of cprime conv=conv_none -- --only-cell "$CPRIME_CELL" --print-key)
printf '  standard sweep (existing, capped, 15 cells)  %s  -> %s/arms_%s\n' "$K_STD" "$STD_DIR" "$K_STD"
printf '  step 0  gate  (capped, 2 cells)               %s  -> %s/arms_%s\n' "$K_GATE" "$STD_DIR" "$K_GATE"
printf '  step 1  --no-cap (13 cells)                   %s  -> %s/arms_%s\n' "$K_NOCAP" "$STD_DIR" "$K_NOCAP"
printf '  step 2  C-prime (conv_none, 1 cell)           %s  -> %s/arms_%s\n' "$K_CPRIME" "$STD_DIR" "$K_CPRIME"
for k in "$K_STD" "$K_GATE" "$K_NOCAP" "$K_CPRIME"; do
  [[ "$k" =~ ^[0-9a-f]{8}$ ]] || { echo "ABORT: a key is not a hash8 ('$k')."; exit 1; }
done
if [ "$(printf '%s\n' "$K_STD" "$K_GATE" "$K_NOCAP" "$K_CPRIME" | sort -u | wc -l)" -ne 4 ]; then
  echo "ABORT: two runs share an output key; a no-cap or single-cell run would overwrite another. Not solving."; exit 1
fi
echo "  all four keys are distinct."
[ "$DRY_RUN" -eq 1 ] || [ -d "$STD_DIR/arms_$K_STD" ] || { echo "ABORT: the standard sweep $STD_DIR/arms_$K_STD does not exist; nothing to compare against."; exit 1; }

# ---- step 0: gate ------------------------------------------------------------------------------------------------------
step_0 () {
  banner "STEP 0 -- gate: default behaviour must reproduce the stored cells exactly (2 solves)"
  run "$RUNNER" 0_gate_default -- --only-cell "$GATE_CELLS"
  if [ "$DRY_RUN" -eq 0 ]; then
    python3 - "$STD_DIR/arms_$K_STD/capital_sweep_results.csv" "$STD_DIR/arms_$K_GATE/capital_sweep_results.csv" <<'PY' || { echo "GATE FAILED: the default path no longer reproduces the stored FIN-3 cells. Steps 1 and 2 are not run."; exit 1; }
import sys, pandas as pd
a = pd.read_csv(sys.argv[1]); b = pd.read_csv(sys.argv[2])
key = ["capital_case", "collection_rate_scenario"]
m = a.merge(b, on=key, suffixes=("_stored", "_gate"))
if len(m) != len(b):
    print(f"gate cells missing from the stored sweep: {len(b)} gate rows, {len(m)} matched"); sys.exit(1)
bad = []
for c in a.columns:
    if c in key: continue
    x, y = m[c + "_stored"], m[c + "_gate"]
    same = (x == y) | (x.isna() & y.isna())
    if not same.all(): bad.append((c, float((x - y).abs().max()) if x.dtype.kind in "fi" else "non-numeric"))
print(f"  gate: {len(b)} cells, {len(a.columns) - len(key)} columns compared, exact equality: {'YES' if not bad else 'NO'}")
for c, d in bad: print("   differs:", c, d)
sys.exit(0 if not bad else 1)
PY
    echo "GATE PASSED -- the default path reproduces the stored cells bit-identically."
  fi
}

# ---- step 1: the no-cap set ----------------------------------------------------------------------------------------
step_1 () {
  banner "STEP 1 -- the same cells with NO emissions cap (13 distinct configurations)"
  run "$RUNNER" 1_nocap -- --no-cap --only-cell "$NOCAP_CELLS"
}

# ---- step 2: attribution row C' -------------------------------------------------------------------------------------
step_2 () {
  banner "STEP 2 -- F-31 attribution row C': conv_none at tight:no_friction, cap applied (1 solve)"
  run "$RUNNER" 2_cprime conv=conv_none -- --only-cell "$CPRIME_CELL"
}

report () {
  python3 - "$STD_DIR/arms_$K_STD/capital_sweep_results.csv" "$STD_DIR/arms_$K_NOCAP/capital_sweep_results.csv" "$STD_DIR/arms_$K_CPRIME/capital_sweep_results.csv" <<'PY'
import sys, pandas as pd
cap = pd.read_csv(sys.argv[1]); nc = pd.read_csv(sys.argv[2])
key = ["capital_case", "collection_rate_scenario"]
# duplicates by configuration: unconstrained has no budget, so its collection rate cannot enter
u = nc[(nc.capital_case == "unconstrained") & (nc.collection_rate_scenario == "no_friction")]
for cr in ("reform_target", "current"):
    d = u.copy(); d["collection_rate_scenario"] = cr; d["copied_from"] = "unconstrained:no_friction"; nc = pd.concat([nc, d], ignore_index=True)
m = cap.merge(nc, on=key, suffixes=("_capped", "_uncapped"))
m["increment_pp"] = 100 * (m.leakage_share_capped - m.leakage_share_uncapped)
m["increment_mt"] = (m.leakage_tco2_capped - m.leakage_tco2_uncapped) / 1e6
order = {"tight": 0, "moderate": 1, "adequacy": 2, "expansion": 3, "unconstrained": 4}
m = m.sort_values(["capital_case", "collection_rate_scenario"], key=lambda s: s.map(order) if s.name == "capital_case" else s)
print("\nLEVEL (capped, as F-31) | uncapped level | INCREMENT (pp, Mt)   -- uncapped share = self-generation emissions / total")
for _, r in m.iterrows():
    cp = "  (copied: same configuration)" if isinstance(r.get("copied_from"), str) else ""
    print(f"  {r.capital_case:14s}{r.collection_rate_scenario:14s} {100*r.leakage_share_capped:6.2f}% | {100*r.leakage_share_uncapped:6.2f}% | {r.increment_pp:+7.2f} pp  {r.increment_mt:+8.1f} Mt{cp}")
uc = m[m.capital_case == "unconstrained"].iloc[0]
print(f"\n7.4% floor: capped unconstrained {100*uc.leakage_share_capped:.2f}%; uncapped unconstrained {100*uc.leakage_share_uncapped:.2f}% -> the floor "
      + ("PERSISTS without a cap." if uc.leakage_share_uncapped > 0.05 else "COLLAPSES without a cap: it is cap-induced."))
ex = m[m.capital_case == "expansion"]; ue = uc.leakage_share_uncapped
print("expansion cells equal the unconstrained cell (uncapped)?", {r.collection_rate_scenario: bool(abs(r.leakage_share_uncapped - ue) < 1e-9) for _, r in ex.iterrows()})
try:
    c = pd.read_csv(sys.argv[3]).iloc[0]
    print(f"\nC-prime (conv_none, tight, no_friction, capped): leakage {100*c.leakage_share:.1f}%, budget shadow {c.budget_shadow_usd_per_usd:.2f}, "
          f"carbon shadow binding mean PV {c.carbon_shadow_pv_binding_mean_usd_per_tco2:.0f} / year-t {c.carbon_shadow_yr_binding_mean_usd_per_tco2:.0f}, binding years {int(c.carbon_binding_years)}"
          f"   (F-31 row C': 55.7%, 8.79, 264, 30)")
except Exception as e:
    print("C-prime not available:", e)
PY
}

echo; echo "Leakage-increment driver$([ "$DRY_RUN" -eq 1 ] && echo " (DRY RUN -- nothing will be solved)")"
if [ -z "$ONLY" ] || [ "$ONLY" = 0 ]; then step_0; fi
if [ -z "$ONLY" ] || [ "$ONLY" = 1 ]; then step_1; fi
if [ -z "$ONLY" ] || [ "$ONLY" = 2 ]; then step_2; fi
if [ "$DRY_RUN" -eq 0 ] && [ -z "$ONLY" ]; then banner "REPORT"; report; fi
echo; echo "Done$([ "$DRY_RUN" -eq 1 ] && echo " (dry run)")."
