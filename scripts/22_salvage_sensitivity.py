"""
22_salvage_sensitivity.py  --  Is the investment trajectory a residual?
=======================================================================

PURPOSE
-------
Tests whether the capacity trajectory RESPONDS to cost, or is determined by the
energy balance regardless of cost. This is the scheduled test written into
correction plan 2.5.

THE FINDING BEING TESTED
------------------------
When salvage was added (plan 2.5), system cost fell 17.1% with the credit
concentrated entirely on late vintages -- and the solar trajectory was IDENTICAL
to five decimal places, while sliced storage was identical to four. Zero MW of
movement from a large, timing-skewed cost change.

The interpretation: with demand exogenous, hydro exogenous, gas at its
deliverability ceiling, and unserved priced out by VoLL, the energy balance has
one unknown per year and the model solves for it. Solar -- and in the sliced
model, storage -- is an ACCOUNTING RESIDUAL, not an optimisation outcome.

WHEN TO RE-RUN THIS
-------------------
After each change that could open a degree of freedom:
  * 2.5 step 3b   removing the min-build floor
  * 2.5 step 4    relaxing or ramping the max-build cap
  * 2.6           genset backstop (serve-vs-pay becomes economic, not penal)
  * 5.3           freeing gas_add

If the trajectory MOVES, the degrees of freedom have opened and the residual
finding must be restated as "was determined, now is not."
If it does NOT move, the constraint structure is a property of Nigeria's system
rather than of the formulation -- the stronger version of the claim.

Either result belongs in the methods chapter. Record which one you got.

METHOD
------
Salvage is switched off by setting every asset life to 1.0 year: the salvage
fraction is max(0, life - years_served)/life, and years_served >= 1 for every
vintage, so the credit is exactly zero. No model change is needed.

Expect the plan 2.5 in-horizon retirement guard to fire loudly in the
no-salvage arm. That is correct -- a 1-year life retires immediately -- and is
suppressed in the printed summary.

OUTPUT
------
    results/salvage_sensitivity/salvage_sensitivity.json
    results/salvage_sensitivity/trajectories.csv
"""

import json
import sys
import contextlib
import io as _io
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io import load_econ, load_solar_capex_by_year
from src.scenarios import load_scenario, asset_lifetimes, solar_min_build_default, MODEL_END_YEAR
from src.optimize_model import build_model, solve_model
from src.optimize_model_sliced import build_model_sliced

RESULTS_DIR = ROOT / "results" / "salvage_sensitivity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_VOLL = "voll_mid"

# MW / MWh below which two trajectories count as identical.
TRAJECTORY_TOL = 1e-4


def lifetime_arms():
    """
    Lifetime arms. 'no_salvage' sets every life to 1.0 yr so the salvage
    fraction is zero for every vintage -- an exact switch-off with no code
    change. The remainder bracket the sourced central values.
    """
    central = asset_lifetimes()
    return {
        "no_salvage":     {"solar": 1.0,  "storage": 1.0,  "gas": 1.0},
        "solar_25":       {**central, "solar": 25.0},
        "central":        dict(central),
        "solar_35":       {**central, "solar": 35.0},
        "storage_12":     {**central, "storage": 12.0},
        "storage_18":     {**central, "storage": 18.0},
    }


def run_arm(lives, sliced, econ, capex_tv):
    """Solve one arm; return objective and capacity trajectories."""
    scenario = load_scenario(
        demand_level_case="served",
        demand_case="organic_central",
        capital_case="unconstrained",
        gas_deliverability_case="baseline",
        solar_build_case="deployment_unconstrained",
        land_case="loose",
        carbon_case="no_policy",
        start_year=2025,
        end_year=MODEL_END_YEAR,
    )
    scenario["solar_min_build_mw_per_year"] = solar_min_build_default()
    scenario["asset_lifetimes"] = dict(lives)

    builder = build_model_sliced if sliced else build_model
    m = builder(scenario=scenario, econ=econ,
                emissions_cap=1e18, solar_capex_by_year=capex_tv)

    # Solver chatter and the (expected) retirement warnings are captured so the
    # summary table stays readable. Failures still raise.
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        status = solve_model(m, scenario=scenario)
    if not status["optimal"]:
        raise RuntimeError(f"non-optimal: {status['status']}")

    years = list(scenario["years"])
    T = range(len(years))

    solar_add = {
        int(years[t]): float(pyo.value(m.solar_public_add[t]))
                       + float(pyo.value(m.solar_eaas_add[t]))
        for t in T
    }
    try:
        storage_cap = {int(years[t]): float(pyo.value(m.storage_capacity_mwh[t]))
                       for t in T}
    except (AttributeError, KeyError):
        storage_cap = {int(years[t]): 0.0 for t in T}

    return {
        "objective_usd": float(pyo.value(m.system_cost_npv)),
        "salvage_usd": (float(pyo.value(m.salvage_npv))
                        if hasattr(m, "salvage_npv") else None),
        "solar_add_mw": solar_add,
        "storage_capacity_mwh": storage_cap,
        "solar_total_mw": sum(solar_add.values()),
        "storage_final_mwh": storage_cap[int(years[-1])],
    }


def max_abs_diff(a, b):
    return max(abs(a[k] - b[k]) for k in a)


def analyse(arms, label):
    ref = arms["central"]
    print(f"\n{'=' * 74}")
    print(f"  SALVAGE SENSITIVITY -- {label.upper()} MODEL")
    print(f"{'=' * 74}")
    print(f"  {'arm':<14}{'objective $bn':>15}{'salvage $bn':>13}"
          f"{'solar dMW':>12}{'storage dMWh':>14}")
    print(f"  {'-' * 68}")

    responsive = False
    rows = []
    for name, r in arms.items():
        d_solar = max_abs_diff(r["solar_add_mw"], ref["solar_add_mw"])
        d_stor = max_abs_diff(r["storage_capacity_mwh"], ref["storage_capacity_mwh"])
        if name != "central" and (d_solar > TRAJECTORY_TOL or d_stor > TRAJECTORY_TOL):
            responsive = True
        salv = f"{r['salvage_usd']/1e9:.4f}" if r["salvage_usd"] is not None else "--"
        print(f"  {name:<14}{r['objective_usd']/1e9:>15.4f}{salv:>13}"
              f"{d_solar:>12.5f}{d_stor:>14.4f}")
        rows.append({
            "model": label, "arm": name,
            "objective_usd": r["objective_usd"], "salvage_usd": r["salvage_usd"],
            "solar_total_mw": r["solar_total_mw"],
            "storage_final_mwh": r["storage_final_mwh"],
            "max_solar_deviation_mw": d_solar,
            "max_storage_deviation_mwh": d_stor,
        })

    ns, ce = arms["no_salvage"], arms["central"]
    cost_swing = (ce["objective_usd"] - ns["objective_usd"]) / ns["objective_usd"]

    print(f"\n  Cost swing, no_salvage -> central: {cost_swing:+.2%}")
    if responsive:
        print("  VERDICT: trajectory RESPONDS to cost. The degrees of freedom")
        print("           have opened -- restate the residual finding.")
    else:
        print("  VERDICT: trajectory IDENTICAL across every arm. Capacity is an")
        print("           ACCOUNTING RESIDUAL -- the energy balance determines it")
        print("           and cost cannot move it.")
        print(f"           A {abs(cost_swing):.1%} cost change moved zero MW.")
    return rows, responsive, cost_swing


def main():
    econ = load_econ(CANONICAL_VOLL)
    capex_tv = load_solar_capex_by_year(scenario_name="solar_low",
                                        start_year=2025, end_year=MODEL_END_YEAR)
    arms_def = lifetime_arms()
    out = {"tolerance_mw": TRAJECTORY_TOL, "models": {}}
    all_rows = []

    for label, sliced in (("annual", False), ("sliced", True)):
        print(f"\nSolving {label} model, {len(arms_def)} arms ...")
        arms = {}
        for name, lives in arms_def.items():
            try:
                arms[name] = run_arm(lives, sliced, econ, capex_tv)
                print(f"  {name:<14} ok")
            except Exception as e:
                print(f"  {name:<14} FAILED: {e}")
        if "central" not in arms:
            print(f"  central arm failed -- cannot analyse {label}")
            continue

        rows, responsive, swing = analyse(arms, label)
        all_rows.extend(rows)
        out["models"][label] = {
            "trajectory_responds_to_cost": responsive,
            "cost_swing_no_salvage_to_central": swing,
            "arms": {k: {kk: vv for kk, vv in v.items()
                         if kk not in ("solar_add_mw", "storage_capacity_mwh")}
                     for k, v in arms.items()},
            "solar_trajectory_central": arms["central"]["solar_add_mw"],
            "storage_trajectory_central": arms["central"]["storage_capacity_mwh"],
        }

    pd.DataFrame(all_rows).to_csv(RESULTS_DIR / "trajectories.csv", index=False)
    with open(RESULTS_DIR / "salvage_sensitivity.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'=' * 74}")
    print("  RECORD THE VERDICT IN THE CORRECTION PLAN (item 2.5).")
    print("  Re-run after 2.5 step 3b, 2.5 step 4, 2.6 and 5.3.")
    print(f"  Saved: {RESULTS_DIR}")
    print(f"{'=' * 74}\n")


if __name__ == "__main__":
    main()
