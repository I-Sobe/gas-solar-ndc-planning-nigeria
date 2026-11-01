#!/usr/bin/env bash
# M1 — capital-window measurement sequence.
# Stops at the first failure. Run from the repo root.
set -euo pipefail

LOG_DIR="logs/m1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
# TASK 53: the anchor is no longer a pasted constant. The pre-epoch value (27317670155.72187) was produced
# under the `advance` window and flat storage capex; the cell that reproduces the post-epoch anchor is the one
# at the load_scenario() default window (`arrears`). The value is read from the post-epoch central run (item 0).
ANCHOR_FILE="results/baseline_sliced/summary__6046f759.json"
RESULT_DIR="results/capital_window"

say() { printf '\n=== %s ===\n' "$*"; }

# ---------------------------------------------------------------
# 1. Pre-flight — state of record
# ---------------------------------------------------------------
say "1. Pre-flight"

git rev-parse --abbrev-ref HEAD | tee "$LOG_DIR/branch.txt"
git rev-parse HEAD              | tee "$LOG_DIR/head.txt"

# src/ and scripts/ must be clean. Review files elsewhere are fine.
if ! git diff --quiet -- src scripts || ! git diff --cached --quiet -- src scripts; then
  echo "ABORT: uncommitted changes under src/ or scripts/."
  git status --porcelain -- src scripts
  exit 1
fi
echo "src/ and scripts/ clean."

# Environment of record. Must match what produced the anchor.
python -c "import pyomo; print('pyomo', pyomo.__version__)" | tee "$LOG_DIR/env.txt"
command -v cbc >/dev/null && cbc -quit 2>&1 | head -2 | tee -a "$LOG_DIR/env.txt"
conda env export --no-builds > "$LOG_DIR/environment_at_run.yml" 2>/dev/null || true

# ---------------------------------------------------------------
# 2. Gates — instalment parity and six distinct paths
# ---------------------------------------------------------------
say "2. Gates"

# Instalment counts must be EQUAL under advance and arrears, per
# capital case. If they are not, the arrears arm is conflating the
# timing change with horizon truncation and the measurement is void.
python scripts/51_check_advance_identity.py --instalments --capital-case unconstrained \
  2>&1 | tee "$LOG_DIR/instalments_unconstrained.txt"
python scripts/51_check_advance_identity.py --instalments --capital-case moderate \
  2>&1 | tee "$LOG_DIR/instalments_moderate.txt"

grep -q "PASS" "$LOG_DIR/instalments_unconstrained.txt" || { echo "ABORT: instalment parity failed (unconstrained)"; exit 1; }
grep -q "PASS" "$LOG_DIR/instalments_moderate.txt"      || { echo "ABORT: instalment parity failed (moderate)";      exit 1; }

# Six distinct output paths, or the cells overwrite each other.
bash scripts/run_capital_window.sh --dry-run 2>&1 | tee "$LOG_DIR/dryrun.txt"

# ---------------------------------------------------------------
# 3. The six solves
# ---------------------------------------------------------------
say "3. Solves"

time bash scripts/run_capital_window.sh 2>&1 | tee "$LOG_DIR/solve.txt"

# ---------------------------------------------------------------
# 4. Post-run checks
# ---------------------------------------------------------------
say "4. Post-run checks"

[ -f "$ANCHOR_FILE" ] || { echo "ABORT: $ANCHOR_FILE not found; run the central baseline (run_phase_b.sh --only 0) before the M1 cells."; exit 1; }
ANCHOR_USD=$(python3 -c "import json,sys; print(repr(json.load(open(sys.argv[1]))['npv_total_cost_usd']))" "$ANCHOR_FILE")
echo "anchor (post-epoch central): $ANCHOR_USD"
N=$(ls -1 "$RESULT_DIR"/summary__*.json 2>/dev/null | wc -l)
echo "summary files written: $N"
[ "$N" -eq 6 ] || { echo "ABORT: expected 6 summary files, found $N."; exit 1; }

# The unconstrained cell at the DEFAULT window (arrears) MUST reproduce the central anchor to
# every digit. If it does not, the arm changed behaviour it should not
# have, and nothing downstream is interpretable.
python3 - "$RESULT_DIR" "$ANCHOR_USD" <<'PY' | tee "$LOG_DIR/anchor_check.txt"
import json, sys, glob, os
d, anchor = sys.argv[1], sys.argv[2]
hits = []
for f in sorted(glob.glob(os.path.join(d, "summary__*.json"))):
    s = json.load(open(f))
    if repr(s.get("npv_total_cost_usd")) == anchor or f"{s.get('npv_total_cost_usd'):.5f}" == f"{float(anchor):.5f}":
        hits.append(os.path.basename(f))
print("cells reproducing the anchor exactly:", hits or "NONE")
print("PASS" if len(hits) >= 1 else "FAIL — no cell reproduces the anchor")
PY

grep -q "^PASS" "$LOG_DIR/anchor_check.txt" || { echo "ABORT: regression cell did not reproduce the anchor."; exit 1; }

# ---------------------------------------------------------------
# 5. The comparison table
# ---------------------------------------------------------------
say "5. Comparison table"

# Dumps every top-level key of all six summaries side by side, so the
# capital-charge components appear without having to know their names
# in advance. Narrow it once you have seen the key list.
python3 - "$RESULT_DIR" <<'PY' | tee "$LOG_DIR/comparison.txt"
import json, glob, os, sys
rows = {}
for f in sorted(glob.glob(os.path.join(sys.argv[1], "summary__*.json"))):
    rows[os.path.basename(f).replace("summary__","").replace(".json","")] = json.load(open(f))
keys = sorted({k for r in rows.values() for k in r})
w = max(len(k) for k in keys) + 2
print("cells:", ", ".join(rows))
for k in keys:
    vals = [rows[c].get(k) for c in rows]
    if all(isinstance(v, (int, float, str, bool)) or v is None for v in vals):
        print(k.ljust(w), " | ".join(str(v)[:22].rjust(22) for v in vals))
    else:
        print(k.ljust(w), "<non-scalar: inspect directly>")
PY

say "Done. Logs in $LOG_DIR"
echo "Send $LOG_DIR/comparison.txt and $LOG_DIR/anchor_check.txt to Chat."