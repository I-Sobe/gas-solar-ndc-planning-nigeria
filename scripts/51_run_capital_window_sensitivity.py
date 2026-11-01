"""
51_run_capital_window_sensitivity.py -- TASK 51 M1: one cell of the
capital-window measurement.

Solves the sliced baseline once at the standard arm set plus a
capital_case and a capital_window_case, and writes ONE summary JSON to
results/capital_window/summary__<hash8>.json carrying everything the TASK 51
report needs per cell: objective, the capital charge as a level and as a
share of the objective (conversion charge separately), the solar and
storage plan, the concessional volume by vintage, and the public-budget
and concessional-envelope status.

This is a MEASUREMENT runner. It does not migrate any other runner and it
does not change the default of capital_window_case ('advance') anywhere.
Driven by scripts/run_capital_window.sh (six cells); not for ad-hoc use.

The hash8 of the output covers the standard arms, capital_case AND the
window arm in every case (unlike the suffix-only scheme om_tag_suffix and
capital_window_tag_suffix use for the shared runners), because every cell
of this runner must have its own file.

Regression check (cell 1): when capital_case == 'unconstrained' and the window
is 'advance', the objective and the decision variables are compared with
results/baseline_sliced/summary__6046f759.json and the outcome is written
into the summary as `anchor_check`. The comparison covers
`npv_total_cost_usd` and every key of `decision_variables`; the anchor
file's other keys are produced by 01b_run_baseline_sliced.py's own
diagnostics and are not recomputed here. The stronger no-solve check is
scripts/51_check_advance_identity.py (identical LP file).
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
    CAPITAL_WINDOW_CASE_CHOICES,
    BASELINE_CENTRAL_ARM,
    load_scenario,
)

RESULTS_DIR = ROOT / "results" / "capital_window"
ANCHOR_PATH = ROOT / "results" / "baseline_sliced" / "summary__6046f759.json"
CANONICAL_VOLL = "voll_mid"
CANONICAL_GENSET = "genset_central"
MODEL_END_YEAR = 2054
REPORT_END_YEAR = 2045


def _v(expr):
    return float(pyo.value(expr))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_standard_arms(parser)
    parser.add_argument("--capital-case", required=True,
                        choices=("unconstrained", "moderate"),
                        help="load_scenario(capital_case=...). 'moderate' adds the "
                             "public solar budget (public_solar_budget_npv). No default.")
    parser.add_argument("--capital-window-case", required=True,
                        choices=CAPITAL_WINDOW_CASE_CHOICES,
                        help="Timing convention of the per-tranche capital charge "
                             "(scenarios.capital_window_scenarios()). No default here; "
                             "'arrears' is load_scenario()'s default since TASK 53.")
    parser.add_argument("--print-path-only", action="store_true",
                        help="print the output path for these arguments and exit (no build, no solve); "
                             "the driver uses it to assert six distinct cell paths before solving.")
    args = parser.parse_args()

    std = standard_arms_kwargs(args)
    # Output key: standard arms + capital_case + capital_window_case, ALWAYS (TASK 51
    # amendment 1.2) -- independent of the shared-runner hash, so the six cells
    # cannot overwrite each other.
    tag_parts = [f"{k}={v}" for k, v in sorted(std.items())]
    from src.cli_arms import _scenario_default_arms
    tag_parts += [f"capital_case={args.capital_case}",
                  f"capital_window_case={args.capital_window_case}",
                  f"storage_capex_case={_scenario_default_arms()['storage_capex_case']}"]   # TASK 53 2.5: the series default joins the key
    arm_tag = "capwin__" + "__".join(tag_parts)
    h8 = hash8_of(arm_tag)
    path = RESULTS_DIR / f"summary__{h8}.json"
    if args.print_path_only:
        print(path)
        return
    scenario = load_scenario(
        demand_level_case="served",
        demand_case="organic_central",
        capital_case=args.capital_case,
        solar_build_case="deployment_unconstrained",
        land_case="loose",
        carbon_case="no_policy",
        start_year=2025,
        end_year=MODEL_END_YEAR,
        capital_window_case=args.capital_window_case,
        **std,
    )
    econ = load_econ(CANONICAL_VOLL, genset_case=CANONICAL_GENSET)
    years = [int(y) for y in scenario["years"]]
    T = range(len(years))
    solar_capex_tv = load_solar_capex_by_year(
        scenario_name="solar_low", start_year=years[0], end_year=years[-1])

    m = build_model_sliced(
        scenario=scenario, econ=econ, emissions_cap=None,
        emissions_cap_by_year=None, solar_capex_by_year=solar_capex_tv)
    status = solve_model(m, scenario=scenario)
    if not status["optimal"]:
        raise RuntimeError(f"capital-window solve failed: {status}")

    npv = _v(m.system_cost_npv)
    ccap = _v(m.capital_charge_total_npv)
    conv_charge = (_v(m.cost_conversion_capital_charge_npv)
                   if hasattr(m, "cost_conversion_capital_charge_npv") else 0.0)
    components = {
        "public_solar": _v(m.cost_public_solar_capital_charge_npv),
        "eaas_solar": _v(m.cost_eaas_solar_capital_charge_npv),
        "gas": _v(m.cost_gas_capital_charge_npv),
        "storage": _v(m.cost_storage_capital_charge_npv),
        "rebuild": _v(m.cost_rebuild_capital_charge_npv),
        "conversion": conv_charge,
    }

    solar = {y: _v(m.solar_public_add[t]) + _v(m.solar_eaas_add[t])
             for t, y in enumerate(years)}
    storage_add = {y: _v(m.storage_add[t]) for t, y in enumerate(years)}
    i45 = years.index(REPORT_END_YEAR)
    dv = {
        "solar_add_mw_by_year": solar,
        "gas_add_mw_by_year": {y: _v(m.gas_add[t]) for t, y in enumerate(years)},
        "storage_add_mwh_by_year": storage_add,
    }
    # TASK 53 2.4 (G1-2): the same year-labelled aggregates 01b writes, so the anchor comparison is like for like.
    dv.update(labelled_decision_totals(
        years, solar, {y: _v(m.storage_capacity_mwh[t]) for t, y in enumerate(years)},
        gas_add_by_year=dv["gas_add_mw_by_year"]))

    conc_vol = {y: _v(m.concessional_volume_by_vintage[t])
                for t, y in enumerate(years)}
    conc_share = {}
    for t, y in enumerate(years):
        elig = (_v(m.solar_public_conc_vol[t]) + _v(m.solar_public_comm_vol[t])
                + _v(m.solar_eaas_conc_vol[t]) + _v(m.solar_eaas_comm_vol[t])
                + _v(m.storage_conc_vol[t]) + _v(m.storage_comm_vol[t]))
        conc_share[y] = conc_vol[y] / elig if elig > 0 else 0.0
    total_conc = float(sum(conc_vol.values()))

    def constraint_status(con):
        if con is None:
            return None
        body = _v(con.body)
        upper = _v(con.upper) if con.upper is not None else None
        d = m.dual.get(con, None)
        return {
            "body": body, "upper": upper,
            "slack": (upper - body) if upper is not None else None,
            "binding": (upper is not None and (upper - body) <= 1e-6 * max(1.0, abs(upper))),
            "dual": float(d) if d is not None else None,
        }

    budget = (constraint_status(m.public_budget_constraint)
              if hasattr(m, "public_budget_constraint") else None)
    envelope = constraint_status(m.f1_concessional_envelope_constraint)

    anchor_check = None
    if args.capital_case == "unconstrained" and args.capital_window_case == "arrears":   # the load_scenario() default since TASK 53
        if ANCHOR_PATH.exists():
            anchor = json.loads(ANCHOR_PATH.read_text())
            same_arms = all(anchor.get(k) == v for k, v in BASELINE_CENTRAL_ARM.items()
                            if k in anchor)
            dv_json = json.loads(json.dumps(dv))
            dv_equal = dv_json == anchor["decision_variables"]
            anchor_check = {
                "anchor_file": str(ANCHOR_PATH.relative_to(ROOT)),
                "arms_match_central": same_arms,
                "npv_this_run": npv,
                "npv_anchor": anchor["npv_total_cost_usd"],
                "npv_bit_identical": npv == anchor["npv_total_cost_usd"],
                "npv_difference": npv - anchor["npv_total_cost_usd"],
                "decision_variables_identical": dv_equal,
                "scope": "npv_total_cost_usd and every key of decision_variables only",
            }
        else:
            anchor_check = {"anchor_file_missing": str(ANCHOR_PATH)}

    out = {
        "arm_tag": arm_tag,
        "capital_case": args.capital_case,
        "capital_window_case": args.capital_window_case,
        "standard_arms": std,
        "npv_total_cost_usd": npv,
        "capital_charge_total_npv_usd": ccap,
        "capital_charge_share_of_objective": ccap / npv,
        "capital_charge_components_npv_usd": components,
        "conversion_charge_share_of_objective": conv_charge / npv,
        "decision_variables": dv,
        "solar_cumulative_to_2045_mw": float(sum(v for y, v in solar.items() if y <= REPORT_END_YEAR)),
        "solar_cumulative_to_2054_mw": float(sum(solar.values())),
        "storage_capacity_2045_mwh": _v(m.storage_capacity_mwh[i45]),
        "storage_capacity_2054_mwh": dv["storage_capacity_2054_mwh"],
        "concessional_volume_by_vintage_usd": conc_vol,
        "concessional_share_by_vintage": conc_share,
        "concessional_volume_total_usd": total_conc,
        "public_budget_constraint": budget,
        "concessional_envelope_constraint": envelope,
        "anchor_check": anchor_check,
        "solver_status": {k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
                          for k, v in status.items()},
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True))
    append_index(RESULTS_DIR, h8, arm_tag)

    print(f"capital_case={args.capital_case} window={args.capital_window_case} hash8={h8}")
    print(f"  npv_total_cost_usd           {npv:.2f}")
    print(f"  capital charge               {ccap:.2f}  ({100*ccap/npv:.2f}% of objective)")
    print(f"  conversion charge            {conv_charge:.2f}")
    print(f"  solar to 2045 / 2054 (MW)    {out['solar_cumulative_to_2045_mw']:.3f} / {out['solar_cumulative_to_2054_mw']:.3f}")
    print(f"  storage 2045 / 2054 (MWh)    {out['storage_capacity_2045_mwh']:.2f} / {out['storage_capacity_2054_mwh']:.2f}")
    print(f"  concessional volume total    {total_conc:.2f}")
    print(f"  public budget                {budget}")
    if anchor_check:
        print(f"  anchor_check                 {anchor_check}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
