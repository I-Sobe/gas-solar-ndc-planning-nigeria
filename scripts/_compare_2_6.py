"""
Commit-A regression comparator for correction plan 2.6.

Three tests, in increasing order of how much they tell you.

TEST 1 -- OBJECTIVE (tolerance 1e-9 relative)
    Not exact equality. Adding m.selfgen adds a column to the LP, so the
    simplex path differs even when the optimum does not, and CBC may return a
    value differing in the last few ULPs. A relative move above 1e-9 means the
    optimum genuinely moved and the plumbing is wrong.

TEST 2 -- SHORTFALL CONSERVATION (exact, to 1e-6 TWh)
    sum(unserved + selfgen) post-patch must equal sum(unserved) pre-patch.
    This is the strong test. It is invariant under LP degeneracy, and it is
    the one that catches a wrong busbar coefficient on the selfgen term --
    which would otherwise shift energy in a way TEST 1 could mask.

TEST 3 -- BUILD PLAN (tolerance 1e-6 relative)
    Solar and storage build must not move. If TESTS 1 and 2 pass but this
    fails, the objective is right but the model is choosing a different plan
    at the same cost -- i.e. it was already degenerate before 2.6, which is a
    finding worth chasing separately.

DEGENERACY NOTE: the split between unserved and selfgen at commit A is
arbitrary and is NOT tested. Only their sum is meaningful.
"""

import json
import sys
from pathlib import Path

import pandas as pd

OUT = Path(sys.argv[1])
OBJ_RTOL = 1e-9
ENERGY_ATOL = 1e-6
PLAN_RTOL = 1e-6

failures = []


def load(name):
    return json.loads((OUT / name).read_text())


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


for tag in ("annual", "sliced"):
    print(f"\n{'=' * 62}\n  {tag.upper()} BASELINE\n{'=' * 62}")
    pre = load(f"prepatch_{tag}_summary.json")
    post = load(f"postpatch_{tag}_summary.json")
    pre_ts = pd.read_csv(OUT / f"prepatch_{tag}_ts.csv")
    post_ts = pd.read_csv(OUT / f"postpatch_{tag}_ts.csv")

    # ---- TEST 1: objective -------------------------------------------------
    o_pre = float(pre["npv_total_cost_usd"])
    o_post = float(post["npv_total_cost_usd"])
    r = rel(o_post, o_pre)
    ok = r < OBJ_RTOL
    print(f"\n  TEST 1  objective")
    print(f"    pre  : {o_pre:,.6f} USD")
    print(f"    post : {o_post:,.6f} USD")
    print(f"    rel  : {r:.3e}   (tol {OBJ_RTOL:.0e})   {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"{tag}: objective moved by {r:.3e}")

    # ---- TEST 2: shortfall conservation ------------------------------------
    pre_short = float(pre_ts["unserved_twh"].sum())
    post_short = float(post_ts["unserved_twh"].sum())
    if "selfgen_twh" in post_ts.columns:
        post_short += float(post_ts["selfgen_twh"].sum())
    else:
        failures.append(f"{tag}: selfgen_twh column missing from timeseries")
    d = abs(post_short - pre_short)
    ok = d < ENERGY_ATOL
    print(f"\n  TEST 2  shortfall conservation")
    print(f"    pre  sum(unserved)           : {pre_short:.9f} TWh")
    print(f"    post sum(unserved + selfgen) : {post_short:.9f} TWh")
    print(f"    diff : {d:.3e} TWh   (tol {ENERGY_ATOL:.0e})   {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"{tag}: shortfall not conserved, diff {d:.3e} TWh")

    if "selfgen_twh" in post_ts.columns:
        sg = float(post_ts["selfgen_twh"].sum())
        us = float(post_ts["unserved_twh"].sum())
        tot = sg + us
        print(f"    split (arbitrary, NOT a test): "
              f"selfgen {sg:.6f} / unserved {us:.6f}"
              + (f"  -> {sg / tot:.1%} selfgen" if tot > ENERGY_ATOL else "  -> both zero"))

    # ---- TEST 3: build plan ------------------------------------------------
    print(f"\n  TEST 3  build plan")
    for key, label in (("solar_total_built_mw", "solar built (MW)"),
                       ("final_storage_capacity_mwh", "final storage (MWh)")):
        a = float(pre["decision_variables"][key])
        b = float(post["decision_variables"][key])
        r = rel(b, a)
        ok = r < PLAN_RTOL
        print(f"    {label:22} pre {a:14,.3f}  post {b:14,.3f}  "
              f"rel {r:.3e}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"{tag}: {label} moved by {r:.3e}")

    # ---- context -----------------------------------------------------------
    e_pre = float(pre["actual_emissions_tco2_total"])
    e_post = float(post["actual_emissions_tco2_total"])
    print(f"\n  context  emissions pre {e_pre:,.1f} / post {e_post:,.1f} tCO2")
    print(f"           (must also be unchanged at commit A: selfgen EF is 0.0)")
    if rel(e_post, e_pre) >= OBJ_RTOL:
        failures.append(f"{tag}: emissions moved -- selfgen EF fallback is not 0.0")

print(f"\n{'=' * 62}")
if failures:
    print(f"  COMMIT A FAILED -- {len(failures)} issue(s). DO NOT PROCEED TO COMMIT B.\n")
    for f in failures:
        print(f"    - {f}")
    print()
    sys.exit(1)

print("  COMMIT A CLEAN. Plumbing is inert. Safe to revalue.")
print(f"{'=' * 62}\n")
