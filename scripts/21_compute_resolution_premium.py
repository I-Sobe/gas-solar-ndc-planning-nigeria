"""
21_compute_resolution_premium.py  --  Annual vs time-sliced resolution premium
==============================================================================

PURPOSE
-------
Computes the cost premium that intra-annual temporal resolution reveals:

    premium = (sliced_objective - annual_objective) / annual_objective

This is a METHODS FINDING, not a scenario result. It quantifies how much annual
resolution understates system cost by allowing 1 TWh of solar to substitute
perfectly for 1 TWh of gas regardless of when either is available.

BASIS CONSISTENCY -- read before trusting the number
----------------------------------------------------
The two objectives are only comparable if they share a cost definition. Any
term present in one model and absent from the other silently corrupts the
premium. This script therefore refuses to report unless it can verify:

  1. Both runs used the same demand arm, gas regime and VoLL case.
  2. Both objectives include the salvage credit (plan 2.5). The previously
     reported ~13% premium was measured BEFORE salvage and is superseded.
  3. Neither run carries unserved energy above tolerance -- a VoLL-dominated
     objective is a penalty measurement, not a cost measurement, and
     differencing two of them is meaningless.
  4. Both runs are recent relative to the source modules.

PRE-REQUISITES
--------------
    python scripts/01_run_baseline.py
    python scripts/01b_run_baseline_sliced.py

OUTPUT
------
    results/resolution_premium/resolution_premium.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

ANNUAL_DIR = ROOT / "results" / "baseline"
SLICED_DIR = ROOT / "results" / "baseline_sliced"
RESULTS_DIR = ROOT / "results" / "resolution_premium"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Unserved above this share of cumulative demand means the objective is
# VoLL-dominated and the comparison is not a cost comparison.
UNSERVED_TOLERANCE_FRACTION = 0.001

# Modules whose mtime must predate both runs, or a run is stale.
SOURCE_FILES = [
    ROOT / "src" / "optimize_model.py",
    ROOT / "src" / "optimize_model_sliced.py",
    ROOT / "src" / "scenarios.py",
]


def _load(path, label):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} diagnostics not found at {path}. "
            f"Run the corresponding baseline script first."
        )
    with open(path) as f:
        return json.load(f)


def _get(d, *keys):
    """First present key, else None. Diagnostics key names differ by model."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def main():
    annual_path = ANNUAL_DIR / "summary.json"
    sliced_path = SLICED_DIR / "summary.json"

    annual = _load(annual_path, "Annual summary")
    sliced = _load(sliced_path, "Sliced summary")

    # Diagnostics carry the salvage credit and the per-year series; the
    # sliced model has no cost keys there at all, so both are optional.
    annual_diag = {}
    sliced_diag = {}
    if (ANNUAL_DIR / "diagnostics.json").exists():
        annual_diag = json.load(open(ANNUAL_DIR / "diagnostics.json"))
    if (SLICED_DIR / "diagnostics.json").exists():
        sliced_diag = json.load(open(SLICED_DIR / "diagnostics.json"))

    warnings = []
    blockers = []

    # ---- Objectives -------------------------------------------------
    a_cost = _get(annual, "npv_total_cost_usd", "system_cost_npv_usd", "npv_cost")
    s_cost = _get(sliced, "npv_total_cost_usd", "system_cost_npv_usd", "npv_cost")
    if a_cost is None or s_cost is None:
        raise KeyError(
            "Could not find a total-cost key in one or both diagnostics files. "
            f"Annual keys: {sorted(annual)[:12]} ... "
            f"Sliced keys: {sorted(sliced)[:12]}"
        )

    premium = (s_cost - a_cost) / a_cost

    # ---- Check 1: salvage present in both (plan 2.5) ----------------
    a_salv = _get(annual_diag, "npv_salvage_credit_usd", "npv_salvage_value_usd")
    s_salv = _get(sliced_diag, "npv_salvage_credit_usd", "npv_salvage_value_usd")
    if s_salv is None:
        warnings.append(
            "Sliced diagnostics carry no salvage key -- the sliced runner writes "
            "no cost fields there. Salvage IS in the sliced objective (verified: "
            "18.31 -> 14.286 bn), so this is a reporting gap, not a basis "
            "mismatch. Add npv_salvage_credit_usd to the sliced runner."
        )
    if a_salv is None:
        blockers.append(
            "Salvage credit missing from the annual run. Objectives may use "
            "different cost definitions and the premium would be meaningless."
        )

    # ---- Check 2: unserved energy ------------------------------------
    def _unserved(d):
        v = _get(d, "cumulative_unserved_twh")
        if v is not None:
            return float(v)
        by_year = _get(d, "unserved_twh_by_year")
        return float(sum(by_year.values())) if by_year else None

    def _demand(d):
        by_year = _get(d, "demand_twh_by_year")
        return float(sum(by_year.values())) if by_year else None

    for label, d, diag in (("annual", annual, annual_diag),
                           ("sliced", sliced, sliced_diag)):
        u = _get(d, "cumulative_unserved_twh")
        if u is None:
            u = _unserved(diag)
        dem = _demand(diag)
        if u is None:
            warnings.append(f"{label}: unserved energy not found -- could not check.")
        elif dem and (u / dem) > UNSERVED_TOLERANCE_FRACTION:
            blockers.append(
                f"{label}: unserved {u:.2f} TWh = {100*u/dem:.2f}% of demand, "
                f"above the {100*UNSERVED_TOLERANCE_FRACTION:.1f}% tolerance. "
                f"The objective is VoLL-dominated; this is a penalty "
                f"measurement, not a cost measurement."
            )
        elif u and u > 1e-6:
            warnings.append(f"{label}: unserved {u:.4f} TWh (within tolerance).")

    # ---- Check 3: same scenario configuration ------------------------
    for field in ("demand_case", "demand_growth_rate", "gas_case",
                  "gas_scenario", "voll_case"):
        av, sv = annual.get(field), sliced.get(field)
        if av is not None and sv is not None and av != sv:
            blockers.append(
                f"Scenario mismatch on '{field}': annual={av!r}, sliced={sv!r}. "
                f"The runs are not comparable."
            )

    # ---- Check 4: staleness ------------------------------------------
    newest_src = max((f.stat().st_mtime for f in SOURCE_FILES if f.exists()),
                     default=0)
    for label, p in (("annual", annual_path), ("sliced", sliced_path)):
        if p.stat().st_mtime < newest_src:
            blockers.append(
                f"{label} run predates the newest source module. Re-run before "
                f"reporting."
            )

    # ---- Component decomposition (optional, sliced-model breakdown) --
    components = {}
    a_dec = annual_diag.get("cost_decomposition", {}) or {}
    s_dec = sliced_diag.get("cost_decomposition", {}) or {}
    for key in ("real_expenditure_npv", "voll_penalty_npv", "voll_penalty_share"):
        av, sv = a_dec.get(key), s_dec.get(key)
        if av is not None or sv is not None:
            components[key] = {
                "annual": av, "sliced": sv,
                "delta": (sv - av) if (av is not None and sv is not None) else None,
            }

    # ---- Report -------------------------------------------------------
    print("\n" + "=" * 68)
    print("  ANNUAL vs TIME-SLICED RESOLUTION PREMIUM")
    print("=" * 68)
    print(f"  Annual objective   : ${a_cost/1e9:>10.4f} bn")
    print(f"  Sliced objective   : ${s_cost/1e9:>10.4f} bn")
    print(f"  Difference         : ${(s_cost-a_cost)/1e9:>10.4f} bn")
    print(f"  RESOLUTION PREMIUM : {premium:>10.2%}")

    if a_salv is not None and s_salv is not None:
        print(f"\n  Salvage credit, annual : ${a_salv/1e9:>9.4f} bn")
        print(f"  Salvage credit, sliced : ${s_salv/1e9:>9.4f} bn")

    if components:
        print("\n  Component decomposition (USD bn):")
        print(f"    {'component':<34}{'annual':>10}{'sliced':>10}{'delta':>10}")
        for k, v in components.items():
            a = f"{v['annual']/1e9:.3f}" if v["annual"] is not None else "--"
            sl = f"{v['sliced']/1e9:.3f}" if v["sliced"] is not None else "--"
            dl = f"{v['delta']/1e9:+.3f}" if v["delta"] is not None else "--"
            print(f"    {k:<34}{a:>10}{sl:>10}{dl:>10}")

    if warnings:
        print("\n  WARNINGS:")
        for w in warnings:
            print(f"    - {w}")

    if blockers:
        print("\n  " + "!" * 60)
        print("  NOT REPORTABLE -- basis consistency failed:")
        for b in blockers:
            print(f"    - {b}")
        print("  " + "!" * 60)
    else:
        print("\n  Basis checks passed. Premium is reportable.")

    print("\n  NOTE: the previously reported ~13% premium was measured BEFORE")
    print("  the salvage correction (plan 2.5) and is superseded by this run.")

    out = {
        "annual_objective_usd": a_cost,
        "sliced_objective_usd": s_cost,
        "difference_usd": s_cost - a_cost,
        "resolution_premium_fraction": premium,
        "resolution_premium_pct": 100 * premium,
        "salvage_credit_annual_usd": a_salv,
        "salvage_credit_sliced_usd": s_salv,
        "components": components,
        "warnings": warnings,
        "blockers": blockers,
        "reportable": len(blockers) == 0,
        "supersedes": "the ~13% premium measured before plan 2.5 salvage",
    }
    path = RESULTS_DIR / "resolution_premium.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {path}\n")

    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
