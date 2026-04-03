import json
import numpy as np
import sys
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.append(str(ROOT))

from src.io import (load_econ, load_solar_capex_by_year)
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
    
    # Load time-varying solar CAPEX from NREL ATB (solar_low scenario).
    # solar_low declines from $1,456k/MW (2025) to $603k/MW (2045).
    solar_capex_tv = load_solar_capex_by_year(
        scenario_name="solar_low",
        start_year=2025, end_year=2045,
    )
    # Activate minimum build floor when time-varying CAPEX is in use.
    # This prevents, the optimizer from delaying all solar to the cheapest years
    # (2040-2045) creating unrealistic 2025-2030 supply gaps.
    scenario["solar_min_build_mw_per_year"] = 100.0

    m = build_model(
        scenario=scenario,
        econ=econ,
        emissions_cap_by_year=caps,  # annual cap trajectory
        solar_capex_by_year=solar_capex_tv,
    )

    status = solve_model(m)
    if not status["optimal"]:
        raise RuntimeError(f"Optimization failed for {cap_scenario_name}: {status}")

    diag = extract_planning_diagnostics(m, scenario, econ)
    npv_total_cost_usd = float(pyo.value(m.system_cost_npv))
    cumulative_unserved_twh = sum(diag["unserved_twh_by_year"].values())
    
    avg_gas_shadow = np.mean(
        [v for v in diag["gas_shadow_price_usd_per_twh_th_by_year"].values()
         if v is not None]
    )
    solar_total_built_mw = sum(
        float(pyo.value(m.solar_public_add[t]))
        + float(pyo.value(m.solar_eaas_add[t]))
        for t in range(len(years))
    )

    out = {
        "cap_scenario": cap_scenario_name,
        "decision_variables": {
            "solar_add_mw_by_year": {
                int(y):
                    float(pyo.value(m.solar_public_add[t]))
                    + float(pyo.value(m.solar_eaas_add[t]))
                for t, y in enumerate(years)
                },
                "storage_capacity_mwh":
                    float(pyo.value(m.storage_capacity_mwh[len(years)-1])),
                "solar_total_built_mw": solar_total_built_mw,
            },
        "npv_total_cost_usd": npv_total_cost_usd,
        "cumulative_unserved_twh": cumulative_unserved_twh,
        "actual_emissions_tco2_total": float(pyo.value(m.emissions)),
        "avg_gas_shadow_usd_per_twh_th": avg_gas_shadow,
        "diagnostics": diag,
        }

    print("Storage discharge (TWh by year):", diag["storage_discharge_twh_e_by_year"])
    # binding constraint by year:", diag["storage_binding_by_year"])

    return out


# ============================================================
# MAIN
# ============================================================

def main():

    CANONICAL_VOLL = "voll_mid"

    # ----------------------------------------------------------------
    # NDC 3.0 scenario definitions
    #
    # Each case carries two parameters that differ between unconditional
    # and conditional:
    #
    #   capital_case — public capital ceiling passed to load_scenario().
    #     unconditional: "moderate" (0.85 × B* = 5.2B USD)
    #       → domestically self-funded floor, budget is binding.
    #     conditional:   "expansion" (1.20 × B* = 7.4B USD)
    #       → international finance supplements domestic capital;
    #         apportioned power-sector share of $270bn is ~$19.8bn,
    #         expansion (7.4B) sits within that envelope while keeping
    #         the constraint economically meaningful (not unconstrained).
    #         Source: NDC 3.0 p.21, same 12.7% electricity-gen share
    #         used to derive the emissions cap.
    #
    # Note: 02_run_ndc_caps.py has NO EaaS. Public capital is the only
    # deployment channel, so capital_case is the dominant constraint.
    # ----------------------------------------------------------------
    NDC_CASES = {
        "ndc3_unconditional": {"capital_case": "moderate"},
        "ndc3_conditional":   {"capital_case": "expansion"},
    }

    for c, cfg in NDC_CASES.items():
        scenario = load_scenario(
            demand_level_case="served",
            demand_case="baseline",
            gas_deliverability_case="baseline",
            capital_case=cfg["capital_case"],
            carbon_case="no_policy",
            start_year=2025,
            end_year=2045,
        )

        voll_cases = [CANONICAL_VOLL]

        for voll_case in voll_cases:

            print(f"\nRunning {c} with {voll_case}")
            econ = load_econ(voll_case)
            out = run_case(c, scenario, econ)

            # Save outputs
            case_dir = RESULTS_DIR / f"{c}_{voll_case}"
            case_dir.mkdir(parents=True, exist_ok=True)

            with open(case_dir / "summary.json", "w") as f:
                json.dump(
                    {
                        "cap_scenario": c,
                        "capital_case": cfg["capital_case"],
                        "voll_case": voll_case,
                        "voll_value_usd_per_twh": econ["UNSERVED_ENERGY_PENALTY"],
                        "decision_variables": out["decision_variables"],
                        "npv_total_cost_usd": out["npv_total_cost_usd"],
                        "cumulative_unserved_twh": out["cumulative_unserved_twh"],
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
                    "demand_twh": [diag["demand_twh_by_year"][y] for y in years],
                    "gas_avail_twh_th": [diag["gas_avail_twh_th_by_year"][y] for y in years],
                    "gas_to_power_twh_th": [diag["gas_to_power_twh_th_by_year"][y] for y in years],
                    "gas_generation_twh_e": [diag["gas_generation_twh_e_by_year"][y] for y in years],
                    "solar_generation_twh_e": [diag["solar_generation_twh_e_by_year"][y] for y in years],
                    "storage_discharge_twh_e": [diag["storage_discharge_twh_e_by_year"][y] for y in years],
                    "unserved_twh": [diag["unserved_twh_by_year"][y] for y in years],
                    "emissions_tco2": [diag["emissions_tco2_by_year"][y] for y in years],
                    "gas_shadow_price_usd_per_twh_th": [diag["gas_shadow_price_usd_per_twh_th_by_year"][y] for y in years],
                    "carbon_shadow_usd_per_tco2": [diag["carbon_shadow_price_usd_per_tco2_by_year"][y] for y in years],
                    "discount_factor": [diag["discount_factor_by_year"][y] for y in years],
                    "subsidy_per_mw_usd": [diag["subsidy_per_mw_usd_by_year"][y] for y in years],
                }
            )

            ts.to_csv(case_dir / "timeseries.csv", index=False)

            print(f"--- {c} saved ---")
            print("Solar addition (MW/year):", out["decision_variables"]["solar_add_mw_by_year"])
            print("Storage capacity (MWh):", out["decision_variables"]["storage_capacity_mwh"])
            print("Total emissions (MtCO2):", out["actual_emissions_tco2_total"] / 1e6)
            print("Unserved 2025 (TWh):", out["diagnostics"]["unserved_twh_by_year"][2025])


if __name__ == "__main__":
    main()
