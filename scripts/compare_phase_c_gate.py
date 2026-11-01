#!/usr/bin/env python3
"""
Phase C regression gate comparator (TASK 46 Part A.2; redefined after TASK 53 2.5, TASK 54).

What the gate asks. Each gated runner is run once more at the DEFAULT design with
`--results-suffix _gate`, at the same arm set Phase B ran it at. The gate passes for a
runner iff every file it wrote is byte-identical to the file Phase B wrote for the SAME
ARM SET. It no longer compares against the pre-epoch stored results: the fix epoch changed
the defaults on purpose (window, storage capex), so byte-identity with pre-epoch files is
neither achievable nor wanted; what the gate guards is that the `--results-suffix` /
`--solar-build-case` plumbing leaves a default-design run unchanged.

How runs are paired. NOT by relative path. Runners that go through resolve_results_dir()
write `<dir>/arms_<hash8>/...` (TASK 53 2.5); `<dir>_gate/arms_<h>` is paired with
`<dir>/arms_<h>` for the same hash8 (hash8 = every arm, so equal hash = equal arm set;
the suffix and a default solar_build_case are not part of it). A gate arm set with no Phase B
counterpart FAILS the gate ("no reference"), it is never skipped. Runners that key their own
file names with complete_arm_tag (02_, 03_, 26_: keyed=False) have no arms_ level; their
hash-named entries are paired by relative path as before, which is the same arm-set pairing.

For a file that differs, the differing fields are reported (JSON: dotted paths; CSV: column
names with the largest absolute difference), so a label-only difference can be told from a
numeric one. ARMS.json is compared on `arm_tag` only (its `invocation` differs by design).

    python scripts/compare_phase_c_gate.py [--runner NAME ...] [--results-root DIR]

Reads results only; writes nothing; no solve.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# runner -> list of (stored dir, gate dir), relative to results/
PAIRS = {
    "02_run_ndc_caps": [("ndc", "ndc_gate")],
    "03_run_ndc_eaas": [("ndc_eaas", "ndc_eaas_gate")],
    "04_run_fin2_tariff_sweep": [("fin2", "fin2_gate")],
    "05_run_fin3_capital_sweep": [("fin3", "fin3_gate")],
    "06_run_gas1_shadow_benchmarks": [("gas1", "gas1_gate")],
    "07_run_gas2_eaas_gas_relief": [("gas2", "gas2_gate")],
    "08_run_gas3_regime_ndc_feasibility": [("gas3", "gas3_gate")],
    "10_run_rel2_marginal_cost": [("rel2", "rel2_gate")],
    "11_run_rel3_financing_frontier": [("rel3", "rel3_gate")],
    "26_run_f3_identification": [("f3", "f3_gate")],
}
SKIP = {"DESIGN.txt"}   # ARMS.json is handled separately (arm_tag only)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def json_diffs(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: present in one side only")
            else:
                out += json_diffs(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += json_diffs(x, y, f"{path}[{i}]")
    elif a != b:
        if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
            return out
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def csv_diffs(pa, pb):
    a, b = pd.read_csv(pa), pd.read_csv(pb)
    if list(a.columns) != list(b.columns):
        return [f"columns differ: {list(a.columns)} vs {list(b.columns)}"]
    if len(a) != len(b):
        return [f"row count {len(a)} vs {len(b)}"]
    out = []
    for c in a.columns:
        x, y = a[c], b[c]
        same = (x == y) | (x.isna() & y.isna())
        if same.all():
            continue
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            out.append(f"{c}: {int((~same).sum())} rows, max abs diff {float((x - y).abs().max()):.6g}")
        else:
            out.append(f"{c}: {int((~same).sum())} rows differ (non-numeric)")
    return out


def _compare_files(stored, gate, files, label=""):
    """Compare the listed files of `gate` against the same relative paths under `stored`."""
    n_same, notes = 0, []
    for g in files:
        rel = g.relative_to(gate)
        name = f"{label}{rel}"
        s = stored / rel
        if not s.exists():
            notes.append(f"{name}: no reference file")
            continue
        if g.name == "ARMS.json":
            if json.load(open(s)).get("arm_tag") == json.load(open(g)).get("arm_tag"):
                n_same += 1
            else:
                notes.append(f"{name}: arm_tag differs")
            continue
        if sha(s) == sha(g):
            n_same += 1
            continue
        if g.name == "INDEX.md":
            # the gate directory indexes only its own run; the reference index lists every arm
            missing = set(g.read_text().splitlines()) - set(s.read_text().splitlines())
            if not missing:
                n_same += 1
            else:
                notes.append(f"{name}: {len(missing)} gate index line(s) not in the reference index")
            continue
        if g.suffix == ".json":
            d = json_diffs(json.load(open(s)), json.load(open(g)))
            notes.append(f"{name}: JSON differs at {len(d)} path(s): " + "; ".join(d[:6]))
        elif g.suffix == ".csv":
            d = csv_diffs(s, g)
            notes.append(f"{name}: CSV differs: " + "; ".join(d[:6]) if d else f"{name}: bytes differ, parsed values equal")
        else:
            notes.append(f"{name}: bytes differ")
    return n_same, notes


def compare(stored, gate):
    """(identical, total), notes. Arm-keyed gate directories are paired by arms_<hash8>; others by relative path."""
    if not gate.exists():
        return None, ["gate directory does not exist (not run)"]
    arm_dirs = sorted(d for d in gate.iterdir() if d.is_dir() and d.name.startswith("arms_"))
    if not arm_dirs:
        files = sorted(p for p in gate.rglob("*") if p.is_file() and p.name not in SKIP)
        same, notes = _compare_files(stored, gate, files)
        return (same, len(files)), notes
    same_all, tot_all, notes_all = 0, 0, []
    # the gate directory's own top-level files (INDEX.md), then each arm set
    top = sorted(p for p in gate.iterdir() if p.is_file() and p.name not in SKIP)
    n, nt = _compare_files(stored, gate, top)
    same_all += n; tot_all += len(top); notes_all += nt
    for ad in arm_dirs:
        files = sorted(p for p in ad.rglob("*") if p.is_file() and p.name not in SKIP)
        tot_all += len(files)
        ref = stored / ad.name
        if not ref.is_dir():
            tag = ""
            try:
                tag = json.load(open(ad / "ARMS.json")).get("arm_tag", "")
            except Exception:
                pass
            notes_all.append(f"{ad.name}: NO REFERENCE -- Phase B has no {stored.name}/{ad.name} for this arm set"
                             + (f" (tag {tag[:90]}...)" if tag else ""))
            continue
        n, nt = _compare_files(ref, ad, files, label=f"{ad.name}/")
        same_all += n; notes_all += nt
    return (same_all, tot_all), notes_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", nargs="*", default=list(PAIRS))
    ap.add_argument("--results-root", default=str(ROOT / "results"),
                    help="the results directory holding both Phase B's directories and the *_gate siblings "
                         "(default: this repository's results/; a test points it elsewhere).")
    args = ap.parse_args()
    root = Path(args.results_root)
    ok_all = True
    print(f"{'runner':<40} {'identical':>12}  verdict")
    for r in args.runner:
        tot_same = tot = 0
        allnotes = []
        ran = True
        for st, ga in PAIRS[r]:
            res, notes = compare(root / st, root / ga)
            if res is None:
                ran = False
                allnotes += [f"{ga}: {notes[0]}"]
                continue
            tot_same += res[0]
            tot += res[1]
            allnotes += [f"[{ga}] {n}" for n in notes]
        if not ran and tot == 0:
            print(f"{r:<40} {'-':>12}  NOT RUN")
            continue
        passed = (tot_same == tot) and tot > 0 and not allnotes
        ok_all &= passed
        print(f"{r:<40} {f'{tot_same}/{tot}':>12}  {'PASS' if passed else 'DIFFERS'}")
        for n in allnotes:
            print("      " + n)
    print("\nGATE:", "ALL PASS" if ok_all else "NOT PASSED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
