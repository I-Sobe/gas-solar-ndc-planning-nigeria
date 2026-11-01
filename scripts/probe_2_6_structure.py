"""
Structural probe for the selfgen wiring (correction plan 2.6).

WHY THIS EXISTS
---------------
The commit-A regression passed with sum(unserved) = 0.000 in both builders.
That is a real pass, but it proves less than it appears to:

  TEST 1 (objective)  -- meaningful. The plumbing perturbed nothing.
  TEST 3 (build plan) -- meaningful. Same plan, same cost.
  TEST 2 (shortfall)  -- proved 0 == 0. Nothing was reallocated, because
                         there was no shortfall to reallocate.

selfgen never entered the basis, so its BALANCE COEFFICIENT and its OBJECTIVE
PRICE were never exercised. A wrong busbar factor, a dropped selfgen term in
one builder's balance, or a sign error on selfgen_npv would all be invisible
to that regression.

This script tests those structurally instead of by simulation: it perturbs
selfgen by one unit and reads the response out of the constraint body and the
objective expression. No solve required. Runs in seconds.

USAGE
-----
    python scripts/probe_2_6_structure.py
    python scripts/probe_2_6_structure.py --busbar 1.0752688172043012

Exit code 0 = all probes pass.
"""

import argparse
import sys
from pathlib import Path

import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.scenarios import load_scenario, solar_min_build_default, MODEL_END_YEAR  # noqa: E402
from src.optimize_model import build_model  # noqa: E402
from src.optimize_model_sliced import build_model_sliced, SLICES  # noqa: E402
from src.io import load_econ, load_solar_capex_by_year  # noqa: E402
from src.cli_arms import add_standard_arms, standard_arms_kwargs  # noqa: E402

RTOL = 1e-9
failures = []


def zero_all_vars(m):
    """Give every variable a concrete value so expressions can be evaluated."""
    for v in m.component_data_objects(pyo.Var, active=True):
        v.set_value(0.0)


def probe(label, expr_fn, var, index, expected, unit=""):
    """Perturb `var[index]` by one unit; compare the response to `expected`."""
    var[index].set_value(0.0)
    base = float(pyo.value(expr_fn()))
    var[index].set_value(1.0)
    bumped = float(pyo.value(expr_fn()))
    var[index].set_value(0.0)

    got = bumped - base
    rel = abs(got - expected) / max(abs(expected), 1e-30)
    ok = rel < RTOL
    print(f"    {label:42} got {got:>18,.6f}  expected {expected:>18,.6f} "
          f"{unit}  {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"{label}: got {got:.9g}, expected {expected:.9g}")
    return got


def run(tag, builder, econ, scenario, capex, sliced):
    print(f"\n{'=' * 78}\n  {tag}\n{'=' * 78}")
    # Plan 0.4, item C: 1e18 sentinel -> None ("no cap", already the
    # builder's own guarded meaning for this argument).
    m = builder(scenario=scenario, econ=econ, emissions_cap=None,
                emissions_cap_by_year=None, solar_capex_by_year=capex, allow_unserved=True)
    zero_all_vars(m)

    busbar = float(scenario.get("selfgen_busbar_factor", 1.0))
    price = econ.get("GENSET_COST_PER_TWH_E", econ["UNSERVED_ENERGY_PENALTY"])
    if "GENSET_ETA_HHV" in econ:
        ef_e = econ["GENSET_EF_TH_HHV"] / econ["GENSET_ETA_HHV"]
    else:
        ef_e = 0.0
    df0 = float(pyo.value(m.DF[0]))

    idx = 0 if not sliced else (0, SLICES[0])
    bidx = 0 if not sliced else (0, *SLICES[0])

    # ---- 1. BALANCE COEFFICIENT ------------------------------------------
    # This is the probe the regression could not run. One TWh of selfgen must
    # relieve the balance by exactly `busbar` TWh -- no more, no less.
    print("\n  [1] balance coefficient on selfgen")
    probe("d(balance body)/d(selfgen)",
          lambda: m.balance[bidx].body, m.selfgen, idx, busbar, "TWh")

    # Sanity anchor: unserved must have coefficient exactly 1.0. If this
    # fails, the probe itself is reading the wrong constraint.
    probe("d(balance body)/d(unserved)  [anchor]",
          lambda: m.balance[bidx].body, m.unserved, idx, 1.0, "TWh")

    # ---- 2. OBJECTIVE PRICE ----------------------------------------------
    print("\n  [2] objective price on selfgen")
    probe("d(system_cost_npv)/d(selfgen)",
          lambda: m.system_cost_npv, m.selfgen, idx, df0 * price, "USD")
    probe("d(system_cost_npv)/d(unserved)  [anchor]",
          lambda: m.system_cost_npv, m.unserved, idx,
          df0 * econ["UNSERVED_ENERGY_PENALTY"], "USD")

    # ---- 3. EMISSIONS BOUNDARY -------------------------------------------
    print("\n  [3] emissions boundary")
    probe("d(emissions_by_year[0])/d(selfgen)",
          lambda: m.emissions_by_year[0], m.selfgen, idx, 1e6 * ef_e, "tCO2")
    probe("d(emissions_grid_only_by_year[0])/d(selfgen)",
          lambda: m.emissions_grid_only_by_year[0], m.selfgen, idx, 0.0, "tCO2")

    # ---- 4. PUBLIC BUDGET ISOLATION --------------------------------------
    # selfgen is PRIVATE expenditure. If it has leaked into the public budget
    # constraint the EaaS comparison is corrupted at the root.
    print("\n  [4] public budget isolation")
    if hasattr(m, "public_budget_constraint"):
        probe("d(public_budget body)/d(selfgen)  [must be 0]",
              lambda: m.public_budget_constraint.body, m.selfgen, idx, 0.0, "USD")
    else:
        print("    public_budget_constraint absent (capital_case unconstrained) "
              "-- rerun with a constrained case to exercise this probe")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--busbar", type=float, default=None,
                    help="selfgen_busbar_factor to test, e.g. 1.0752688 for TLF=7%%")
    ap.add_argument("--genset-case", default=None,
                    help="e.g. genset_central. Omit to probe the commit-A fallback.")
    ap.add_argument("--voll", default="voll_mid")
    ap.add_argument("--capital-case", default="unconstrained",
                    help="tight | moderate | adequacy | expansion | unconstrained")
    # task_runner_revival, Part 2A: this probe's load_scenario() call
    # supplied only om_case of the 17 sentinel-required keys -- 16 missing
    # (tariff_case, peak_multiple_case, reserve_margin, storage_cc_case,
    # gas_build_case, gas_availability_case, gas_retirement_case, gas_
    # capex_case, ccgt_efficiency_case, capital_recovery_case, dscr_case,
    # us_inflation, capital_treatment_case, concessional_envelope_case,
    # conversion_case, hydro_case). Added via add_standard_arms(ap),
    # excluding gas_deliverability_case only, since this call already
    # hardcodes gas_deliverability_case="baseline" below -- a pre-existing
    # constant this revival does not change.
    add_standard_arms(ap, exclude=["gas_deliverability_case"])
    args = ap.parse_args()
    standard_kwargs = standard_arms_kwargs(args)

    scenario = load_scenario(
        demand_level_case="served", demand_case="organic_central",
        capital_case=args.capital_case, gas_deliverability_case="baseline",
        solar_build_case="deployment_unconstrained", land_case="loose",
        carbon_case="no_policy", start_year=2025, end_year=MODEL_END_YEAR,
        **standard_kwargs,
    )
    if args.busbar is not None:
        scenario["selfgen_busbar_factor"] = args.busbar
    scenario["solar_min_build_mw_per_year"] = solar_min_build_default()

    econ = load_econ(args.voll, genset_case=args.genset_case)
    yrs = scenario["years"]
    capex = load_solar_capex_by_year("solar_low", int(yrs[0]), int(yrs[-1]))

    mode = args.genset_case or "FALLBACK (priced at VoLL, EF 0)"
    print(f"\nprobing: genset_case = {mode}")
    print(f"         busbar factor = {scenario.get('selfgen_busbar_factor', 1.0)}")

    run("ANNUAL", build_model, econ, scenario, capex, sliced=False)
    run("SLICED", build_model_sliced, econ, scenario, capex, sliced=True)

    print(f"\n{'=' * 78}")
    if failures:
        print(f"  {len(failures)} STRUCTURAL FAILURE(S)\n")
        for f in failures:
            print(f"    - {f}")
        print()
        sys.exit(1)
    print("  STRUCTURE OK. selfgen enters balance, objective and emissions "
          "with the intended coefficients.")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
