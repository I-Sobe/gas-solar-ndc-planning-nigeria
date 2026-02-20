import json
import sys
from pathlib import Path

import pandas as pd
ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.append(str(ROOT))

from src.scenarios import load_scenario
import pyomo.environ as pyo
from src.optimize_model import build_model, solve_model
from src.optimize_experiments import extract_planning_diagnostics


# ============================================================
# PATH SETUP
# ============================================================

#ROOT = Path(__file__).resolve().parents[1]  # repo root
#sys.path.append(str(ROOT))

RESULTS_DIR = ROOT / "results" / "baseline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def to_float(x) -> float:
    return float(str(x).replace(",", "").replace("$", "").strip())


def load_econ() -> dict:
    econ = {}

    gas_cost_df = pd.read_csv(
        ROOT / "data" / "cost" / "processed" / "gas_cost.csv",
        thousands=",",
    )
    gas_low_row = gas_cost_df[gas_cost_df["Scenario"] == "gas_low"].iloc[0]
    econ["GAS_COST_PER_TWH_TH"] = to_float(gas_low_row["total_usd_per_twh_th"])

    solar_df = pd.read_csv(
        ROOT / "data" / "cost" / "processed" / "solar_capex.csv",
        thousands=",",
    )
    solar_row = solar_df[
        (solar_df["Scenario"] == "solar_low") & (solar_df["Year"] == 2025)
    ].iloc[0]
    econ["SOLAR_CAPEX_PER_MW"] = to_float(solar_row["Solar_capex_usd_per_mw"])

    storage_df = pd.read_csv(
        ROOT / "data" / "cost" / "processed" / "storage_capex.csv",
        thousands=",",
    )
    storage_row = storage_df[
        (storage_df["Scenario"] == "Storage_low") & (storage_df["Year"] == 2025)
    ].iloc[0]
    # NOTE: ensure your header truly is Storage_capex_usd_per_mwh
    econ["STORAGE_COST_PER_MWH"] = to_float(storage_row["Storage_capex_usd_per_mwh"])

    voll_df = pd.read_csv(
        ROOT / "data" / "cost" / "processed" / "unserved_energy_penalty.csv",
        thousands=",",
    )
    voll_row = voll_df[
        (voll_df["scenario"] == "voll_low") & (voll_df["year"] == 2025)
    ].iloc[0]
    econ["UNSERVED_ENERGY_PENALTY"] = to_float(voll_row["voll_usd_per_twh"])

    # Emissions accounting ON (policy can still be "no_policy")
    econ["CARBON_EMISSION_FACTOR"] = 0.421  # tCO2/MWh_e

    return econ


# ============================================================
# MAIN
# ============================================================

def main():

    scenario = load_scenario(
        demand_level_case="served",
        demand_case="baseline",
        gas_case="baseline",
        gas_deliverability_case="baseline",
        solar_case="baseline",
        carbon_case="no_policy",
        start_year=2025,
        end_year=2045,
    )

    econ = load_econ()

    years = scenario["years"]

    # Build and solve baseline model with an effectively non-binding cumulative cap
    m = build_model(
        scenario=scenario,
        econ=econ,
        emissions_cap=1e18,          # scalar cumulative cap
        emissions_cap_by_year=None,  # ensure this is scalar case
    )

    status = solve_model(m)
    if not status["optimal"]:
        raise RuntimeError(f"Baseline solve failed: {status}")

    diag = extract_planning_diagnostics(m, scenario)

    # Decision variables
    dv = {
        "solar_add_mw_by_year": {int(y): float(pyo.value(m.solar_add[t])) for t, y in enumerate(years)},
        "storage_capacity_mwh": float(pyo.value(m.storage_capacity)),
        "solar_total_built_mw": sum(float(pyo.value(m.solar_add[t])) for t in range(len(years))),
    }

    npv_total_cost_usd = float(pyo.value(m.system_cost_npv))
    cumulative_unserved_twh = sum(diag["unserved_twh_by_year"].values())
    actual_emissions_tco2_total = float(pyo.value(m.emissions))

    print("Storage discharge (TWh by year):", diag["storage_discharge_twh_e_by_year"])
    print("Storage binding constraint by year:", diag["storage_binding_by_year"])

    # ------------------------------------------------------------
    # Save diagnostics (json)
    # ------------------------------------------------------------
    with open(RESULTS_DIR / "diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)

    # ------------------------------------------------------------
    # Save summary (json)
    # ------------------------------------------------------------
    summary = {
        "cap_scenario": "baseline_no_policy",
        "decision_variables": dv,
        "npv_total_cost_usd": npv_total_cost_usd,
        "cumulative_unserved_twh": cumulative_unserved_twh,
        "actual_emissions_tco2_total": actual_emissions_tco2_total,
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
        }
    )

    ts.to_csv(RESULTS_DIR / "timeseries.csv", index=False)

    print("--- Baseline run saved ---")
    print("Saved diagnostics:", RESULTS_DIR / "diagnostics.json")
    print("Saved summary:", RESULTS_DIR / "summary.json")
    print("Saved timeseries:", RESULTS_DIR / "timeseries.csv")
    print("Solar addition (MW/year):", dv["solar_add_mw_by_year"])
    print("Storage capacity (MWh):", dv["storage_capacity_mwh"])


if __name__ == "__main__":
    main()
