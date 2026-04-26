import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.append(str(ROOT))

from src.scenarios import load_scenario
import pyomo.environ as pyo
from src.optimize_model import build_model, solve_model
from src.optimize_experiments import extract_planning_diagnostics
from src.optimize_experiments import run_deterministic_scenario
from src.io import load_econ, load_solar_capex_by_year
from src.utils import json_safe

CANONICAL_VOLL = "voll_mid"


# ============================================================
# PATH SETUP
# ============================================================

RESULTS_DIR = ROOT / "results" / "baseline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MAIN
# ============================================================

def main():

    scenario = load_scenario(
        demand_level_case="served",
        demand_case="baseline",
        capital_case="unconstrained",
        gas_deliverability_case="baseline",
        solar_build_case="aggressive",
        land_case="loose",
        carbon_case="no_policy",
        start_year=2025,
        end_year=2045,
    )

    econ = load_econ(CANONICAL_VOLL)

    years = scenario["years"]

    # Load time-varying solar CAPEX from NREL ATB (solar_low scenario).
    # solar_low declines from $1,456k/MW (2025) to $603k/MW (2045).
    solar_capex_tv = load_solar_capex_by_year(
        scenario_name="solar_low",
        start_year=int(years[0]),
        end_year=int(years[-1]),
    )

    # Activate minimum build floor when time-varying CAPEX is in use.
    # This prevents, the optimizer from delaying all solar to the cheapest years
    # (2040-2045) creating unrealistic 2025-2030 supply gaps.
    scenario["solar_min_build_mw_per_year"] = 100.0

    # Build and solve baseline model with an effectively non-binding cumulative cap
    m = build_model(
        scenario=scenario,
        econ=econ,
        emissions_cap=1e18,          # scalar cumulative cap
        emissions_cap_by_year=None,  # ensure this is scalar case
        solar_capex_by_year=solar_capex_tv,
    )

    status = solve_model(m)
    if not status["optimal"]:
        raise RuntimeError(f"Baseline solve failed: {status}")

    diag = extract_planning_diagnostics(m, scenario, econ)

    capacity_paths = {
        "solar_mw": [
            float(pyo.value(m.solar_capacity_mw[t]))
            for t in range(len(years))
        ],
        "storage_mwh": [
            float(pyo.value(m.storage_capacity_mwh[t]))
            for t in range(len(years))
        ],
        "gas_mw": [
            float(pyo.value(m.gas_capacity_mw[t]))
            for t in range(len(years))
        ],
    }

    det_output = run_deterministic_scenario(
        scenario=scenario,
        econ=econ,
        capacity_paths=capacity_paths,
    )

    # Decision variables
    dv = {
        "solar_add_mw_by_year":
            {int(y):
                float(pyo.value(m.solar_public_add[t]))
                + float(pyo.value(m.solar_eaas_add[t]))
            for t,y in enumerate(years)
        },
        # gas_add removed (Fix 9): gas generation is fuel-constrained,
        # new gas capacity additions are never optimal and the variable
        # was removed from the model.
        "storage_add_mwh_by_year": {
            int(y): float(pyo.value(m.storage_add[t]))
            for t, y in enumerate(years)
        },
        "solar_total_built_mw": sum(
            float(pyo.value(m.solar_public_add[t]))
            + float(pyo.value(m.solar_eaas_add[t]))
            for t in range(len(years))
        ),
        "final_storage_capacity_mwh":
            float(pyo.value(m.storage_capacity_mwh[len(years)-1])),
    }

    npv_total_cost_usd = float(pyo.value(m.system_cost_npv))
    cumulative_unserved_twh = sum(diag["unserved_twh_by_year"].values())
    actual_emissions_tco2_total = float(pyo.value(m.emissions))

    print("Storage discharge (TWh by year):", diag["storage_discharge_twh_e_by_year"])
    print("Storage binding constraint by year:", diag["storage_binding_by_year"])

    # Cross-validation of LP storage formulation against stateful dispatch
    det_unserved = float(np.sum(det_output["unserved"]))
    lp_unserved = cumulative_unserved_twh
    print(f"LP unserved (TWh): {lp_unserved:.4f}")
    print(f"Deterministic dispatch unserved (TWh): {det_unserved:.4f}")
    if abs(det_unserved - lp_unserved) > 0.1:
        print(f"WARNING: dispatch mismatch = {abs(det_unserved - lp_unserved):.3f} TWh")

    # ------------------------------------------------------------
    # Save diagnostics (json)
    # ------------------------------------------------------------
    with open(RESULTS_DIR / "diagnostics.json", "w") as f:
        json.dump(json_safe(diag), f, indent=2)

    # Compute actual discounted public solar CAPEX used.
    # The model uses econ["SOLAR_CAPEX_PER_MW"] — a fixed 2025 scalar from
    # the solar_low NREL ATB scenario. There is no time-varying solar_capex_param
    # Pyomo object in the current model; that was a planned extension that was
    # not implemented. Using the scalar is consistent with how the objective
    # function prices solar additions.
    solar_public_npv_spend = sum(
        float(pyo.value(m.DF[t]))
        * float(pyo.value(m.solar_public_add[t]))
        * float(pyo.value(m.solar_capex_param[t])) #time-varying NREL ATB value
        for t in range(len(years))
    )

    print("Discounted public solar CAPEX used (USD):", solar_public_npv_spend)


    # ------------------------------------------------------------
    # Save summary (json)
    # ------------------------------------------------------------
    summary = {
        "cap_scenario": "baseline_no_policy",
        "decision_variables": dv,
        "npv_total_cost_usd": npv_total_cost_usd,
        "cumulative_unserved_twh": cumulative_unserved_twh,
        "actual_emissions_tco2_total": actual_emissions_tco2_total,
        "solar_public_npv_spend": solar_public_npv_spend,
        "notes": "Baseline = no binding emissions cap (cap set to 1e18 tCO2).",
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------
    # Save yearly timeseries (csv)
    # ------------------------------------------------------------
    years = scenario["years"]

    ts = pd.DataFrame(
        {
            "year": [int(y) for y in years],
            "demand_twh": [diag["demand_twh_by_year"][int(y)] for y in years],
            "gas_avail_twh_th": [diag["gas_avail_twh_th_by_year"][int(y)] for y in years],
            "gas_to_power_twh_th": [diag["gas_to_power_twh_th_by_year"][int(y)] for y in years],
            "gas_generation_twh_e": [diag["gas_generation_twh_e_by_year"][int(y)] for y in years],
            "solar_generation_twh_e": [diag["solar_generation_twh_e_by_year"][int(y)] for y in years],
            "storage_discharge_twh_e": [diag["storage_discharge_twh_e_by_year"][int(y)] for y in years],
            "unserved_twh": [diag["unserved_twh_by_year"][int(y)] for y in years],
            "emissions_tco2": [diag["emissions_tco2_by_year"][int(y)] for y in years],
            "gas_shadow_usd_per_twh_th": [diag["gas_shadow_price_usd_per_twh_th_by_year"][int(y)] for y in years],
            "carbon_shadow_usd_per_tco2": [diag["carbon_shadow_price_usd_per_tco2_by_year"][int(y)] for y in years],
            "discount_factor": [diag["discount_factor_by_year"][int(y)] for y in years],
            # gas_add_mw removed (Fix 9): variable no longer exists in model
            "solar_capacity_mw": [float(pyo.value(m.solar_capacity_mw[t])) for t in range(len(years))],
            "gas_capacity_mw": [float(pyo.value(m.gas_capacity_mw[t])) for t in range(len(years))],
            "storage_capacity_mwh": [float(pyo.value(m.storage_capacity_mwh[t])) for t in range(len(years))],
            "subsidy_per_mw_usd": [diag["subsidy_per_mw_usd_by_year"][y] for y in years],
        }
    )

    ts.to_csv(RESULTS_DIR / "timeseries.csv", index=False)

    print("--- Baseline run saved ---")
    print("Saved diagnostics:", RESULTS_DIR / "diagnostics.json")
    print("Saved summary:", RESULTS_DIR / "summary.json")
    print("Saved timeseries:", RESULTS_DIR / "timeseries.csv")
    print("Solar addition (MW/year):", dv["solar_add_mw_by_year"])
    print("Final storage capacity (MWh):", dv["final_storage_capacity_mwh"])

if __name__ == "__main__":
    main()
