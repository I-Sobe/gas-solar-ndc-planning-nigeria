#!/usr/bin/env python3
"""
F1 quantity diagnostic -- omnibus, read-only, no solve.

WHY THIS EXISTS
---------------
Under f1_tranche, terminal storage is 75,481 MWh and terminal gas is
5,313 MW across EVERY capital_recovery_case tested (cr_central,
cr_comm_low L=15, cr_comm_high L=7) and every envelope arm (env_dares,
env_none), while the objective moves 22.0 -> 26.4 bn. Quantities are
completely insensitive to cost of capital; only price moves.

In an LP that means the quantities are pinned by constraints, not sitting
at an interior cost optimum. Three hypotheses remain live and this script
tests ALL of them in one pass rather than one guess per round:

  H1  New gas build is zero under F1 at every commercial rate tested, so
      gas capacity is just the retirement schedule and is not on a margin.
  H2  Storage is a RESIDUAL of the firm-capacity constraint:
      storage_firm_contribution == requirement - gas_firm_contribution,
      year by year. If so it is not an optimised outcome, which the
      project's own standing principle says it must be.
  H3  Terminal truncation arbitrage relocated from solar to storage.
      Tested by whether the storage ramp start year MOVES with L_comm.
      (It should, if H3 is live. Prior evidence says it does not.)

Also checks a reporting defect: the runner prints "storage 2045" but
appears to emit TERMINAL capacity, not the 2045 value.

WHAT IT READS
-------------
Only results/baseline_sliced/*.csv and *.json already on disk. It does
NOT solve, does not import the model, does not touch src/. Run it after
the arms it compares have been run.
"""

import glob
import json
import os
import sys

import pandas as pd

RESULTS = "results/baseline_sliced"

# Arm tag fragments -> label. Add rows here if more arms are run.
ARMS = {
    "pre_f1_overnight__env_dares":            "regression",
    "cr_central__dscr_regression__f1_tranche__env_dares": "f1_central_dares",
    "cr_central__dscr_regression__f1_tranche__env_none":  "f1_central_none",
    "cr_comm_low__dscr_regression__f1_tranche__env_none": "f1_commlow_L15",
    "cr_comm_high__dscr_regression__f1_tranche__env_none": "f1_commhigh_L7",
}

# L_comm per arm, for the H3 truncation test.
L_COMM = {
    "f1_central_dares": 10,
    "f1_central_none": 10,
    "f1_commlow_L15": 15,
    "f1_commhigh_L7": 7,
}


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def find(pattern_fragment, kind):
    """Locate one results file by arm fragment and kind (timeseries/summary)."""
    hits = [f for f in glob.glob(f"{RESULTS}/{kind}*")
            if pattern_fragment in os.path.basename(f)]
    if not hits:
        return None
    return sorted(hits, key=os.path.getmtime)[-1]


def load_ts(frag):
    f = find(frag, "timeseries")
    if f is None:
        return None, None
    d = pd.read_csv(f)
    ycols = [c for c in d.columns if c.lower() in ("year", "t")]
    if ycols:
        d = d.set_index(ycols[0])
    return d, f


def col(d, *cands):
    """First matching column name, else None."""
    for c in cands:
        if c in d.columns:
            return c
    return None


def main():
    if not os.path.isdir(RESULTS):
        sys.exit(f"Not found: {RESULTS}. Run from the repo root.")

    frames, files, missing = {}, {}, []
    for frag, label in ARMS.items():
        d, f = load_ts(frag)
        if d is None:
            missing.append(label)
        else:
            frames[label], files[label] = d, f

    if missing:
        print(f"NOTE: no timeseries found for: {', '.join(missing)}")
        print("      (run those arms first, or ignore if not yet run)\n")
    if not frames:
        sys.exit("No arm results found at all.")

    print("Arms loaded:")
    for k, f in files.items():
        print(f"  {k:<20} {os.path.basename(f)[:70]}")

    # ---------------------------------------------------------------
    banner("0. COLUMNS AVAILABLE (so nothing below is guessed)")
    any_df = next(iter(frames.values()))
    print(sorted(any_df.columns.tolist()))

    # ---------------------------------------------------------------
    banner("H1. GAS CAPACITY BY YEAR -- is new gas build zero under F1?")
    gcol = col(any_df, "gas_capacity_mw")
    if gcol:
        g = pd.DataFrame({k: v[gcol] for k, v in frames.items()})
        print(g.round(1).to_string())
        print("\nYear-on-year CHANGE in gas capacity (negative = retirement only,")
        print("positive = new build). If F1 columns are never positive, H1 holds.")
        print(g.diff().round(1).to_string())
        for k in g.columns:
            adds = g[k].diff()
            pos = adds[adds > 1.0]
            print(f"  {k:<20} years with net gas ADD > 1 MW: "
                  f"{list(pos.index) if len(pos) else 'NONE'}")
    else:
        print("gas_capacity_mw not present.")

    # ---------------------------------------------------------------
    banner("H2. STORAGE -- residual of firm capacity, or optimised?")
    scol = col(any_df, "storage_capacity_mwh")
    if scol:
        s = pd.DataFrame({k: v[scol] for k, v in frames.items()})
        print("Storage capacity (MWh):")
        print(s.round(0).to_string())
        print("\nAnnual storage BUILD (MWh):")
        print(s.diff().round(0).to_string())

        print("\nIdentical-to-last-digit check across F1 arms:")
        f1arms = [k for k in s.columns if k != "regression"]
        base = f1arms[0] if f1arms else None
        for k in f1arms[1:]:
            print(f"  {base} vs {k}: "
                  f"{s[base].round(6).equals(s[k].round(6))}")
        print("  -> if ALL True, storage is insensitive to BOTH rate and")
        print("     tenor, which rules out a cost-driven interior optimum.")
    else:
        print("storage_capacity_mwh not present.")

    # ---------------------------------------------------------------
    banner("H2b. FIRM CAPACITY -- slack, dual, and the residual identity")
    fcols = [c for c in any_df.columns
             if "firm" in c.lower() or "reserve" in c.lower()
             or "slack" in c.lower() or "dual" in c.lower()]
    if fcols:
        print(f"Firm-capacity-related columns found: {fcols}\n")
        for k, d in frames.items():
            print(f"--- {k}")
            print(d[fcols].round(2).to_string())
            print()
        print("READ: if slack is ~0 in the same years storage is being built,")
        print("      and the dual is positive, storage is being set BY the")
        print("      constraint, not chosen against cost. That is H2.")
    else:
        print("No firm-capacity columns in the timeseries CSV.")
        print("Check the summary JSON instead:")
        for frag, label in ARMS.items():
            f = find(frag, "summary")
            if f:
                with open(f) as fh:
                    j = json.load(fh)
                keys = [k for k in j
                        if any(w in k.lower()
                               for w in ("firm", "slack", "dual", "reserve"))]
                print(f"  {label}: {keys if keys else 'none'}")

    # ---------------------------------------------------------------
    banner("H3. TRUNCATION ARBITRAGE -- does the ramp move with L_comm?")
    if scol:
        print("If terminal truncation drives late storage, the year the build")
        print("ramp accelerates should shift EARLIER as L_comm grows, because")
        print("a longer window means more escaped payments at the horizon.\n")
        print(f"{'arm':<20}{'L_comm':>8}{'terminal MWh':>15}"
              f"{'first yr build>2000':>22}")
        for k in s.columns:
            builds = s[k].diff()
            big = builds[builds > 2000]
            first = big.index[0] if len(big) else "never"
            print(f"{k:<20}{L_COMM.get(k, '-'):>8}"
                  f"{s[k].iloc[-1]:>15,.0f}{str(first):>22}")
        print("\nREAD: if terminal MWh and ramp year are IDENTICAL across")
        print("      L_comm = 7, 10, 15, truncation length is NOT the driver")
        print("      and H3 is dead.")

    # ---------------------------------------------------------------
    banner("H4. CONCESSIONAL DRAW -- is the envelope actually binding?")
    for frag, label in ARMS.items():
        f = find(frag, "summary")
        if not f:
            continue
        with open(f) as fh:
            j = json.load(fh)
        keys = [k for k in j if "conc" in k.lower() or "envelope" in k.lower()
                or "comm_vol" in k.lower() or "tranche" in k.lower()]
        if keys:
            print(f"--- {label}")
            for k in keys:
                v = j[k]
                if isinstance(v, (int, float)):
                    print(f"    {k:<45} {v:,.2f}")
                else:
                    print(f"    {k:<45} {str(v)[:60]}")
    print("\nREAD: with a $750M envelope against billions of capex, the draw")
    print("      should equal the envelope exactly (binding) in env_dares and")
    print("      be zero in env_none. If the draw is BELOW the envelope in")
    print("      env_dares, the envelope is NOT binding and something else")
    print("      limits concessional uptake.")

    # ---------------------------------------------------------------
    banner("REPORTING DEFECT -- 'storage 2045' label check")
    if scol:
        for k, d in frames.items():
            v2045 = d[scol].get(2045, float("nan"))
            print(f"  {k:<20} 2045 = {v2045:>12,.0f}   "
                  f"terminal = {d[scol].iloc[-1]:>12,.0f}")
        print("\nThe runner prints 'storage 2045'. If that print matches the")
        print("TERMINAL column rather than the 2045 column, the summary block")
        print("is mislabeled and other lines in it should be checked too.")

    banner("SUMMARY OF WHAT TO CONCLUDE")
    print("""
  H1 confirmed  -> new gas is uneconomic under F1 at every commercial rate
                   tested; gas capacity is the retirement schedule alone.
                   That is a reportable finding, not a bug.

  H2 confirmed  -> storage is a firm-capacity residual, not an optimised
                   outcome. This CONTRADICTS the project's standing
                   principle and must be fixed or declared before any
                   storage number is reported.

  H3 dead       -> truncation length is not the driver (already strongly
                   suggested by the identical terminal MWh across L=7/10/15).

  If H1 and H2 both hold, the storage quantity carries no cost-optimality
  content and the +70% against the regression arm is an artifact of gas
  exiting the firm-capacity market, not a financing result.
""")


if __name__ == "__main__":
    main()
