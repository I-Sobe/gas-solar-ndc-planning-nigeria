#!/usr/bin/env bash
# scripts/diagnose_f33_window.sh  (v2 — explicit config columns)
# DIAGNOSTIC ONLY. No solve. F-33 Part 1 on the 2025-2045 window.
set -uo pipefail
cd "${THESIS_ROOT:-$HOME/thesis-code}" || { echo "FATAL: repo not found"; exit 1; }
OUT=docs/task_log_49_f33_window.md

python3 - "$OUT" <<'PY' | tee "$OUT"
import os, re, csv, json

REPORT_END, STANDARD = 2045, 0.10
CONFIG = ["gas_case", "solar_build_case", "capital_case", "eps", "reliability_mode"]
out = []
def w(s=""): out.append(s)

def num(v):
    try: return float(v)
    except (TypeError, ValueError): return None

def find(cols, *pats):
    for p in pats:
        for c in cols:
            if re.fullmatch(p, c, re.I): return c
    for p in pats:
        for c in cols:
            if re.search(p, c, re.I): return c
    return None

w("# TASK 49 — F-33 on the 2025-2045 reporting window\n")
w("Diagnostic. No solve. Grouped by an explicit config key.\n")

for form in ("total", "annual"):
    for arm in ("gas_avail_nameplate", "gas_avail_observed"):
        path = f"results/rel1/{form}/{arm}/shortfall_by_year.csv"
        if not os.path.isfile(path):
            w(f"### `{path}` — ABSENT\n"); continue
        rows = list(csv.DictReader(open(path, newline="")))
        if not rows:
            w(f"### `{path}` — EMPTY\n"); continue
        cols = list(rows[0])
        ycol = find(cols, r"year", r".*year.*")
        fcol = find(cols, r".*shortfall.*frac.*", r".*uns.*frac.*", r"fraction")
        if not fcol:                      # derive it if absent
            st, dt = find(cols, r"shortfall_twh"), find(cols, r"demand_twh")
            if not (st and dt):
                w(f"### `{path}` — ABORT, columns = {cols}\n"); continue
            for r in rows:
                s, d = num(r[st]), num(r[dt])
                r["_frac"] = s / d if (s is not None and d) else None
            fcol = "_frac"
        keys = [c for c in CONFIG if c in cols]
        w(f"### `{path}`")
        w(f"rows {len(rows)} · year=`{ycol}` · fraction=`{fcol}` · grouped by {keys}\n")

        groups = {}
        for r in rows:
            y, f = num(r[ycol]), num(r[fcol])
            if y is None or f is None: continue
            groups.setdefault(tuple(r.get(k, "") for k in keys), []).append((int(y), f))

        w("| " + " | ".join(keys) + " | yrs>10% full | worst full | yrs>10% ≤2045 | worst ≤2045 |")
        w("|" + "---|" * (len(keys) + 4))
        changed = 0
        for key, ser in sorted(groups.items()):
            ser.sort()
            win = [(y, f) for y, f in ser if y <= REPORT_END]
            nf = sum(1 for _, f in ser if f > STANDARD)
            nw = sum(1 for _, f in win if f > STANDARD)
            wf = max(ser, key=lambda t: t[1]) if ser else (0, 0.0)
            ww = max(win, key=lambda t: t[1]) if win else (0, 0.0)
            flag = " **← changes**" if nf != nw else ""
            if flag: changed += 1
            w(f"| {' | '.join(str(k) for k in key)} | {nf} | {wf[1]:.4f} ({wf[0]}) "
              f"| {nw}{flag} | {ww[1]:.4f} ({ww[0]}) |")
        w(f"\n**{changed} of {len(groups)} configurations change.** "
          f"Years present: {min(y for s in groups.values() for y,_ in s)}"
          f"–{max(y for s in groups.values() for y,_ in s)}\n")

w("## Part 2 — restrictable?\n")
for form in ("total", "annual"):
    for arm in ("gas_avail_nameplate", "gas_avail_observed"):
        p = f"results/rel1/{form}/{arm}/rel1_summary.json"
        if not os.path.isfile(p): continue
        try: d = json.load(open(p))
        except Exception as e: w(f"- `{p}` unreadable: {e}"); continue
        ks = list(d) if isinstance(d, dict) else []
        per = [k for k in ks if re.search(r"by_year|per_year|series|annual", k, re.I)]
        w(f"- `{p}` → keys {ks[:25]}")
        w(f"  - **restrictable without a re-run: {'YES' if per else 'NO'}**")

print("\n".join(out))
PY
echo; echo "Written to $OUT"