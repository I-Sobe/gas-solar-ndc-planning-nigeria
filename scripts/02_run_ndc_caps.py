import json
import sys
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.append(str(ROOT))


from src.scenarios import load_scenario
from src.optimize_model import build_model, solve_model
from src.optimize_experiments import extract_planning_diagnostics


# ============================================================
# PATH SETUP
# ============================================================


RESULTS_DIR = ROOT / "results" / "ndc"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAP_PATH = ROOT / "data" / "cost" / "processed" / "emissions_cap.csv"


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
    econ["STORAGE_COST_PER_MWH"] = to_float(storage_row["Storage_capex_usd_per_mwh"])

    voll_df = pd.read_csv(
        ROOT / "data" / "cost" / "processed" / "unserved_energy_penalty.csv",
        thousands=",",
    )
    voll_row = voll_df[
        (voll_df["scenario"] == "voll_low") & (voll_df["year"] == 2025)
    ].iloc[0]
    econ["UNSERVED_ENERGY_PENALTY"] = to_float(voll_row["voll_usd_per_twh"])

    # Emissions accounting ON
    econ["CARBON_EMISSION_FACTOR"] = 0.421  # tCO2/MWh_e

    return econ


def load_annual_caps(scenario_name: str, years: list[int]) -> list[float]:
    """
    Load annual emissions caps from the processed cap file, aligned to model years.
    """
    if not CAP_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CAP_PATH}. Run scripts/00_build_emissions_cap.py first."
        )

    df = pd.read_csv(CAP_PATH)
    df = df[df["scenario"] == scenario_name].copy()
    df = df.sort_values("year")

    # Align exactly to model years
    df = df[df["year"].isin([int(y) for y in years])]
    df = df.sort_values("year")

    caps = df["cap_tco2"].astype(float).tolist()

    if len(caps) != len(years):
        raise ValueError(
            f"Cap length {len(caps)} != number of model years {len(years)}. "
            f"Check missing years in emissions_cap.csv for scenario={scenario_name}."
        )

    return caps


# ============================================================
# RUNNERS
# ============================================================

def run_case(cap_scenario_name: str, scenario: dict, econ: dict) -> dict:
    years = [int(y) for y in scenario["years"]]
    caps = load_annual_caps(cap_scenario_name, years)

    m = build_model(
        scenario=scenario,
        econ=econ,
        emissions_cap_by_year=caps,  # annual cap trajectory
    )

    status = solve_model(m)
    if not status["optimal"]:
        raise RuntimeError(f"Optimization failed for {cap_scenario_name}: {status}")

    diag = extract_planning_diagnostics(m, scenario)

    out = {
        "cap_scenario": cap_scenario_name,
        "decision_variables": {
            "solar_addition_mw_per_year": float(pyo.value(m.solar_addition)),
            "storage_capacity_mwh": float(pyo.value(m.storage_capacity)),
        },
        "actual_emissions_tco2_total": float(pyo.value(m.emissions)),
        "diagnostics": diag,
    }

    print("Storage discharge (TWh by year):", diag["storage_discharge_twh_e_by_year"])
    print("Storage binding constraint by year:", diag["storage_binding_by_year"])

    return out


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

    cases = ["ndc_unconditional_20", "ndc_conditional_47"]

    for c in cases:
        out = run_case(c, scenario, econ)

        # ------------------------------------------------------------
        # Save outputs
        # ------------------------------------------------------------
        case_dir = RESULTS_DIR / c
        case_dir.mkdir(parents=True, exist_ok=True)

        with open(case_dir / "summary.json", "w") as f:
            json.dump(
                {
                    "cap_scenario": c,
                    "decision_variables": out["decision_variables"],
                    "actual_emissions_tco2_total": out["actual_emissions_tco2_total"],
                },
                f,
                indent=2,
            )

        with open(case_dir / "diagnostics.json", "w") as f:
            json.dump(out["diagnostics"], f, indent=2)

        # ------------------------------------------------------------
        # Save timeseries (csv)
        # ------------------------------------------------------------
        years = [int(y) for y in scenario["years"]]
        diag = out["diagnostics"]

        ts = pd.DataFrame(
            {
                "year": years,
                "gas_avail_twh_th": [diag["gas_avail_twh_th_by_year"][y] for y in years],
                "gas_to_power_twh_th": [diag["gas_to_power_twh_th_by_year"][y] for y in years],
                "gas_generation_twh_e": [diag["gas_generation_twh_e_by_year"][y] for y in years],
                "unserved_twh": [diag["unserved_twh_by_year"][y] for y in years],
                "gas_shadow_price_usd_per_twh_th": [
                    diag["gas_shadow_price_usd_per_twh_th_by_year"][y] for y in years
                ],
            }
        )

        ts.to_csv(case_dir / "timeseries.csv", index=False)

        print(f"--- {c} saved ---")
        print("Solar addition (MW/year):", out["decision_variables"]["solar_addition_mw_per_year"])
        print("Storage capacity (MWh):", out["decision_variables"]["storage_capacity_mwh"])
        print("Total emissions (MtCO2):", out["actual_emissions_tco2_total"] / 1e6)
        print("Unserved 2025 (TWh):", out["diagnostics"]["unserved_twh_by_year"][2025])


if __name__ == "__main__":
    main()
