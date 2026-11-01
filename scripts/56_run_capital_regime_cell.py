"""
56_run_capital_regime_cell.py -- TASK 53 2.8: one cell at a stated capital envelope AND a stated financing
regime. This is the runner S3 (the proposed second regression signature, TASK 52 F3) needs and none had:
`01b_run_baseline_sliced.py` exposes neither flag, and `51_run_capital_window_sensitivity.py` has
--capital-case but forces the traditional regime.

S3 = tight capital, EaaS regime, central arms, 2025-2054, at the load_scenario() defaults
(capital_window_case arrears, storage_capex_case declining_nrel since TASK 53):

    python scripts/56_run_capital_regime_cell.py <standard arms> --capital-case tight --financing-regime eaas

(`bash scripts/run_signature_s3.sh` supplies the central arms.)

WHAT IS RECORDED AS THE SIGNATURE: the objective and the concessional volume. NEVER the EaaS megawatts: the
EaaS/public split is not identified on the flat optimal face (F-12, register L48), so an EaaS-MW signature
would fail on a correct model. The EaaS total is written for context under a key that says so
(`eaas_solar_total_mw_UNIDENTIFIED_F12`). DSCR is objective-flat and cannot be covered by any objective-valued
signature (TASK 53 section 5).

Output: results/capital_regime/summary__<hash8>.json, the hash covering every standard arm, capital_case,
financing_regime, capital_window_case and storage_capex_case.
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
from src.scenarios import (
    CAPITAL_WINDOW_CASE_CHOICES,
    STORAGE_CAPEX_CASE_CHOICES,
    capital_envelope_scenarios,
    load_scenario,
)
from src.summary_years import labelled_decision_totals

RESULTS_DIR = ROOT / "results" / "capital_regime"
MODEL_END_YEAR = 2054


def _v(expr):
    return float(pyo.value(expr))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_standard_arms(ap)
    ap.add_argument("--capital-case", required=True, choices=sorted(capital_envelope_scenarios()),
                    help="load_scenario(capital_case=...). No default: the envelope is the point of this runner.")
    ap.add_argument("--financing-regime", required=True, choices=("traditional", "eaas"),
                    help="scenario['financing_regime'], set after load_scenario() as 02_run_eaas.py does. No default.")
    ap.add_argument("--capital-window-case", default=None, choices=CAPITAL_WINDOW_CASE_CHOICES,
                    help="absent -> load_scenario()'s default (arrears)")
    ap.add_argument("--storage-capex-case", default=None, choices=STORAGE_CAPEX_CASE_CHOICES,
                    help="absent -> load_scenario()'s default (declining_nrel)")
    ap.add_argument("--print-path-only", action="store_true")
    args = ap.parse_args()

    std = standard_arms_kwargs(args)
    opt = {}
    if args.capital_window_case is not None:
        opt["capital_window_case"] = args.capital_window_case
    if args.storage_capex_case is not None:
        opt["storage_capex_case"] = args.storage_capex_case

    scenario = load_scenario(
        demand_level_case="served", demand_case="organic_central",
        capital_case=args.capital_case, solar_build_case="deployment_unconstrained",
        land_case="loose", carbon_case="no_policy",
        start_year=2025, end_year=MODEL_END_YEAR, **opt, **std)
    # the key records the arms as RESOLVED (the defaults in force), not as typed
    tag_parts = [f"{k}={v}" for k, v in sorted(std.items())]
    tag_parts += [f"capital_case={args.capital_case}", f"financing_regime={args.financing_regime}",
                  f"capital_window_case={scenario['capital_window_case']}",
                  f"storage_capex_case={scenario['storage_capex_case']}"]
    arm_tag = "regime__" + "__".join(tag_parts)
    h8 = hash8_of(arm_tag)
    path = RESULTS_DIR / f"summary__{h8}.json"
    if args.print_path_only:
        print(path)
        return

    scenario["financing_regime"] = args.financing_regime
    econ = load_econ("voll_mid", genset_case="genset_central")
    years = [int(y) for y in scenario["years"]]
    T = range(len(years))
    solar_capex_tv = load_solar_capex_by_year(scenario_name="solar_low", start_year=years[0], end_year=years[-1])
    m = build_model_sliced(scenario=scenario, econ=econ, emissions_cap=None, emissions_cap_by_year=None,
                           solar_capex_by_year=solar_capex_tv)
    status = solve_model(m, scenario=scenario)
    if not status["optimal"]:
        raise RuntimeError(f"capital-regime solve failed: {status}")

    npv = _v(m.system_cost_npv)
    solar = {y: _v(m.solar_public_add[t]) + _v(m.solar_eaas_add[t]) for t, y in enumerate(years)}
    solar_eaas = sum(_v(m.solar_eaas_add[t]) for t in T)
    dv = {"solar_add_mw_by_year": solar,
          "storage_add_mwh_by_year": {y: _v(m.storage_add[t]) for t, y in enumerate(years)},
          "gas_add_mw_by_year": {y: _v(m.gas_add[t]) for t, y in enumerate(years)}}
    dv.update(labelled_decision_totals(years, solar, {y: _v(m.storage_capacity_mwh[t]) for t, y in enumerate(years)},
                                       gas_add_by_year=dv["gas_add_mw_by_year"]))
    conc_total = float(sum(_v(m.concessional_volume_by_vintage[t]) for t in T))
    out = {
        "arm_tag": arm_tag,
        "capital_case": args.capital_case,
        "financing_regime": args.financing_regime,
        "capital_window_case": scenario["capital_window_case"],
        "storage_capex_case": scenario["storage_capex_case"],
        "standard_arms": std,
        # the signature: objective and concessional volume only (F-12)
        "signature": {"npv_total_cost_usd": npv, "concessional_volume_total_usd": conc_total},
        "npv_total_cost_window": f"{years[0]}-{years[-1]}, discounted to {years[0]}",
        "capital_charge_total_npv_usd": _v(m.capital_charge_total_npv),
        "decision_variables": dv,
        "eaas_solar_total_mw_UNIDENTIFIED_F12": solar_eaas,
        "solver_status": {k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v)) for k, v in status.items()},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True))
    append_index(RESULTS_DIR, h8, arm_tag)
    print(f"capital_case={args.capital_case} regime={args.financing_regime} window={scenario['capital_window_case']} storage={scenario['storage_capex_case']} hash8={h8}")
    print(f"  signature: npv_total_cost_usd {npv:.2f}   concessional_volume_total_usd {conc_total:.2f}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
