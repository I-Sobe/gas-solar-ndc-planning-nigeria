"""
54_run_storage_capex_case.py -- TASK 52 section 2.1 / TASK 53: one cell of the storage
capex measurement (register L90).

Solves the sliced baseline once at the standard arms, capital_case
'unconstrained' and capital_window_case 'arrears' (the anchor configuration since
TASK 53), varying ONLY storage_capex_case ('declining_nrel' | 'flat'). Writes ONE
summary JSON to results/storage_capex/summary__<hash8>.json.

TASK 53 made 'declining_nrel' the central case and 'flat' the declared sensitivity
(this runner's pair of cells is that sensitivity, reported in section 3.7).
The output key covers the standard arms, capital_case, capital_window_case
AND storage_capex_case, so the two cells cannot overwrite each other.

Under 'declining_nrel' (the default) the objective and decision variables are
compared with results/baseline_sliced/summary__6046f759.json (`anchor_check`); the
no-solve check is scripts/51_check_advance_identity.py (identical LP file).

Usage (Zeus):
    python scripts/54_run_storage_capex_case.py --storage-capex-case declining_nrel   # reproduces the anchor
    python scripts/54_run_storage_capex_case.py --storage-capex-case flat             # the declared sensitivity
"""

import argparse
import json
import sys
from pathlib import Path

import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cli_arms import add_standard_arms, standard_arms_kwargs
from src.io import load_econ, load_solar_capex_by_year
from src.optimize_model import solve_model
from src.optimize_model_sliced import build_model_sliced
from src.output_naming import append_index, hash8_of
from src.summary_years import labelled_decision_totals
from src.scenarios import (
    BASELINE_CENTRAL_ARM,
    STORAGE_CAPEX_CASE_CHOICES,
    load_scenario,
)

RESULTS_DIR = ROOT / "results" / "storage_capex"
ANCHOR_PATH = ROOT / "results" / "baseline_sliced" / "summary__6046f759.json"
MODEL_END_YEAR = 2054
REPORT_END_YEAR = 2045


def _v(expr):
    return float(pyo.value(expr))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_standard_arms(parser)
    parser.add_argument("--storage-capex-case", required=True,
                        choices=STORAGE_CAPEX_CASE_CHOICES,
                        help="scenarios.storage_capex_scenarios(). No default here; "
                             "'declining_nrel' is load_scenario()'s default since TASK 53.")
    parser.add_argument("--print-path-only", action="store_true")
    args = parser.parse_args()

    std = standard_arms_kwargs(args)
    tag_parts = [f"{k}={v}" for k, v in sorted(std.items())]
    tag_parts += ["capital_case=unconstrained", "capital_window_case=arrears",
                  f"storage_capex_case={args.storage_capex_case}"]
    arm_tag = "stcapex__" + "__".join(tag_parts)
    h8 = hash8_of(arm_tag)
    path = RESULTS_DIR / f"summary__{h8}.json"
    if args.print_path_only:
        print(path)
        return

    scenario = load_scenario(
        demand_level_case="served", demand_case="organic_central",
        capital_case="unconstrained", solar_build_case="deployment_unconstrained",
        land_case="loose", carbon_case="no_policy",
        start_year=2025, end_year=MODEL_END_YEAR,
        capital_window_case="arrears",
        storage_capex_case=args.storage_capex_case, **std)
    econ = load_econ("voll_mid", genset_case="genset_central")
    years = [int(y) for y in scenario["years"]]
    T = range(len(years))
    solar_capex_tv = load_solar_capex_by_year(
        scenario_name="solar_low", start_year=years[0], end_year=years[-1])

    m = build_model_sliced(
        scenario=scenario, econ=econ, emissions_cap=None,
        emissions_cap_by_year=None, solar_capex_by_year=solar_capex_tv)
    status = solve_model(m, scenario=scenario)
    if not status["optimal"]:
        raise RuntimeError(f"storage-capex solve failed: {status}")

    npv = _v(m.system_cost_npv)
    solar = {y: _v(m.solar_public_add[t]) + _v(m.solar_eaas_add[t]) for t, y in enumerate(years)}
    storage_add = {y: _v(m.storage_add[t]) for t, y in enumerate(years)}
    i45 = years.index(REPORT_END_YEAR)
    dv = {
        "solar_add_mw_by_year": solar,
        "gas_add_mw_by_year": {y: _v(m.gas_add[t]) for t, y in enumerate(years)},
        "storage_add_mwh_by_year": storage_add,
    }
    # TASK 53 2.4 (G1-2): the same year-labelled aggregates 01b writes.
    dv.update(labelled_decision_totals(
        years, solar, {y: _v(m.storage_capacity_mwh[t]) for t, y in enumerate(years)},
        gas_add_by_year=dv["gas_add_mw_by_year"]))

    anchor_check = None
    if args.storage_capex_case == "declining_nrel" and ANCHOR_PATH.exists():     # the default since TASK 53
        anchor = json.loads(ANCHOR_PATH.read_text())
        anchor_check = {
            "anchor_file": str(ANCHOR_PATH.relative_to(ROOT)),
            "arms_match_central": all(anchor.get(k) == v for k, v in BASELINE_CENTRAL_ARM.items()
                                      if k in anchor),
            "npv_this_run": npv,
            "npv_anchor": anchor["npv_total_cost_usd"],
            "npv_bit_identical": npv == anchor["npv_total_cost_usd"],
            "decision_variables_identical": json.loads(json.dumps(dv)) == anchor["decision_variables"],
            "scope": "npv_total_cost_usd and every key of decision_variables only",
        }

    out = {
        "arm_tag": arm_tag,
        "storage_capex_case": args.storage_capex_case,
        "standard_arms": std,
        "storage_capex_usd_per_mwh_by_year": {y: float(m.storage_capex_param[t])
                                              for t, y in enumerate(years)},
        "npv_total_cost_usd": npv,
        "decision_variables": dv,
        "solar_cumulative_to_2045_mw": float(sum(v for y, v in solar.items() if y <= REPORT_END_YEAR)),
        "solar_cumulative_to_2054_mw": float(sum(solar.values())),
        "storage_capacity_2045_mwh": _v(m.storage_capacity_mwh[i45]),
        "storage_capacity_2054_mwh": dv["storage_capacity_2054_mwh"],
        "storage_capex_npv_usd": _v(m.cost_storage_capex_npv) if hasattr(m, "cost_storage_capex_npv") else None,
        "anchor_check": anchor_check,
        "solver_status": {k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
                          for k, v in status.items()},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True))
    append_index(RESULTS_DIR, h8, arm_tag)

    print(f"storage_capex_case={args.storage_capex_case} hash8={h8}")
    print(f"  npv_total_cost_usd        {npv:.2f}")
    print(f"  solar to 2045 / 2054 (MW) {out['solar_cumulative_to_2045_mw']:.3f} / {out['solar_cumulative_to_2054_mw']:.3f}")
    print(f"  storage 2045 / 2054 (MWh) {out['storage_capacity_2045_mwh']:.2f} / {out['storage_capacity_2054_mwh']:.2f}")
    if anchor_check:
        print(f"  anchor_check              {anchor_check}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
