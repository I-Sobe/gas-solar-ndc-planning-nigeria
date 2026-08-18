"""
20_run_cf1_solar_yield_sensitivity.py  —  CF-1
==============================================

RESEARCH QUESTION (CF-1)
------------------------
How sensitive are system cost and solar deployment to the assumed solar
capacity factor? Solar CF is a swept physical parameter because Nigeria's
delivered yield is genuinely uncertain: it spans the north-south irradiance
gradient and is reduced below clean PVOUT by soiling, high-temperature
derating, and availability.

CF is the dominant scalar on solar energy per MW, so an optimistic CF
inflates the solar-displaces-gas result. This runner isolates that effect.

CF BAND (fixed-tilt; matches NREL ATB fixed-tilt CAPEX)
-------------------------------------------------------
  low     0.18  — southern sites and/or full soiling/thermal/availability derate
  central 0.20  — Global Solar Atlas PVOUT for named N. Nigeria sites / 8760,
                  net of realistic operating derate   [SOURCE: name sites]
  high    0.22  — clean high-irradiance northern yield

METHOD
------
Everything except solar_cf is held at the baseline configuration used in
01_run_baseline.py (served demand, baseline growth, baseline gas, aggressive
solar build, no carbon policy, non-binding scalar cap). Only solar_cf varies.
Convention follows the rest of the codebase: load_scenario(...) then mutate
the scenario dict for keys that are not load_scenario arguments.

OUTPUT
------
results/cf1/cf_sensitivity.csv  — one row per CF value:
  cf, objective_npv_usd, public_solar_capex_npv_usd,
  total_solar_built_mw, cumulative_unserved_twh, total_emissions_tco2
"""

from pathlib import Path
import csv
import sys

import pyomo.environ as pyo

# ── Repo root ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from src.scenarios import load_scenario, asset_lifetime_sweep, solar_min_build_default, MODEL_END_YEAR
from src.optimize_model import build_model, solve_model
from src.optimize_experiments import extract_planning_diagnostics
from src.io import load_econ, load_solar_capex_by_year

# Resolve repo root the same way the other runners do.
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "cf1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_VOLL = "voll_mid"
CF_SWEEP = [0.18, 0.20, 0.22]


def main():
    econ = load_econ(CANONICAL_VOLL)

    print("CF-1: solar capacity-factor sensitivity")
    print(f"  CF sweep: {CF_SWEEP}")
    print(f"  total solves: {len(CF_SWEEP)}")

    rows = []
    for cf in CF_SWEEP:
        # Baseline configuration (identical to 01_run_baseline.py) ...
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
        # ... with only solar_cf varied (mutate convention, as elsewhere).
        scenario["solar_cf"] = cf
        scenario["solar_min_build_mw_per_year"] = solar_min_build_default()

        years = list(scenario["years"])
        solar_capex_tv = load_solar_capex_by_year(
            scenario_name="solar_low",
            start_year=int(years[0]),
            end_year=int(years[-1]),
        )

        m = build_model(
            scenario=scenario,
            econ=econ,
            emissions_cap=1e18,
            emissions_cap_by_year=None,
            solar_capex_by_year=solar_capex_tv,
        )
        status = solve_model(m, scenario=scenario)
        if not status["optimal"]:
            raise RuntimeError(f"CF-1 solve failed at cf={cf}: {status}")

        diag = extract_planning_diagnostics(m, scenario, econ)

        total_solar_mw = sum(
            float(pyo.value(m.solar_public_add[t]))
            + float(pyo.value(m.solar_eaas_add[t]))
            for t in range(len(years))
        )
        cumulative_unserved = sum(diag["unserved_twh_by_year"].values())

        row = {
            "cf": cf,
            "objective_npv_usd": float(pyo.value(m.system_cost_npv)),
            "public_solar_capex_npv_usd": sum(
                float(pyo.value(m.DF[t]))
                * float(pyo.value(m.solar_public_add[t]))
                * float(pyo.value(m.solar_capex_param[t]))
                for t in range(len(years))
            ),
            "total_solar_built_mw": total_solar_mw,
            "cumulative_unserved_twh": cumulative_unserved,
            "total_emissions_tco2": float(pyo.value(m.emissions)),
        }
        rows.append(row)
        print(
            f"  cf={cf:.2f}  NPV={row['objective_npv_usd']/1e9:.2f}B  "
            f"solar={total_solar_mw/1000:.1f}GW  "
            f"unserved={cumulative_unserved:.2f}TWh  "
            f"emis={row['total_emissions_tco2']/1e6:.2f}Mt"
        )

    out_path = RESULTS_DIR / "cf_sensitivity.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
