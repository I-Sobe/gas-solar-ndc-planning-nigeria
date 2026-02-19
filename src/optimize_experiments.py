"""
Optimization experiments and orchestration.

Scope
-----
- Deterministic operational scenario evaluation
- Pareto frontier generation (weighted-sum, ε-constraint)

Notes
-----
This module orchestrates experiments only.
Planning optimization and operational simulation intentionally
use different storage abstractions and are not numerically
comparable at the component level.
"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from src.gas_supply import gas_generation_cap, gas_available_power
from src.optimize_model import build_model, solve_model
from src.economics import (
    gas_cost,
    solar_capex,
    carbon_cost,
    unserved_energy_cost,
    total_system_cost,
)
from src.solar import solar_generation, solar_capacity_trajectory
from src.storage import BatteryStorage
from src.dispatch import dispatch_energy
from src.demand import project_baseline_demand
from src.scenarios import (
    load_scenario,
    demand_level_scenarios,
    demand_growth_scenarios,
    gas_decline_scenarios,
    solar_capacity_scenarios,
    carbon_policy_scenarios,
)


# ============================================================
# ANNUAL CAP OPTIMIZATION
# ============================================================

def run_annual_cap_case(
    scenario,
    econ,
    cap_scenario_name,
    cap_path="data/cost/processed/emissions_cap.csv",
):

    years = scenario["years"]
    start_year = int(years[0])
    end_year = int(years[-1])

    cap_df = pd.read_csv(cap_path)

    cap_df = cap_df[
        (cap_df["scenario"] == cap_scenario_name)
        & (cap_df["year"] >= start_year)
        & (cap_df["year"] <= end_year)
    ].copy()

    cap_df = cap_df.sort_values("year")

    emissions_cap_by_year = cap_df["cap_tco2"].astype(float).tolist()

    if len(emissions_cap_by_year) != len(years):
        raise ValueError(
            f"Cap series length {len(emissions_cap_by_year)} "
            f"!= number of model years {len(years)}"
        )

    m = build_model(
        scenario=scenario,
        econ=econ,
        emissions_cap_by_year=emissions_cap_by_year,
    )

    status = solve_model(m)

    if not status["optimal"]:
        raise RuntimeError(
            f"Optimization failed for annual cap scenario {cap_scenario_name}"
        )

    diagnostics = extract_planning_diagnostics(m, scenario)

    return {
        "cap_scenario": cap_scenario_name,
        "decision_variables": {
            "solar_addition_mw_per_year": float(pyo.value(m.solar_addition)),
            "storage_capacity_mwh": float(pyo.value(m.storage_capacity)),
        },
        "actual_emissions_tco2_total": float(pyo.value(m.emissions)),
        "diagnostics": diagnostics,
    }


# ============================================================
# DETERMINISTIC OPERATIONAL SIMULATION
# ============================================================

def run_deterministic_scenario(scenario, econ):

    years = scenario["years"]
    n_years = len(years)

    demand = project_baseline_demand(
        base_demand=scenario["base_demand_twh"],
        growth_rate=scenario["demand_growth"],
        start_year=years[0],
        end_year=years[-1],
    )["demand"]

    gas_gen = gas_generation_cap(
        q0=scenario["gas_q0_twh"],
        decline_rate=scenario["gas_decline"],
        start_year=years[0],
        end_year=years[-1],
    )["generation"]

    solar_cap = solar_capacity_trajectory(
        initial_capacity_mw=scenario["solar_baseline_mw"],
        annual_addition_mw=scenario["solar_addition_mw"],
        start_year=years[0],
        end_year=years[-1],
    )

    solar_gen = solar_generation(
        capacity_mw=solar_cap["capacity_mw"],
        capacity_factor=scenario["solar_cf"],
    )

    storage = BatteryStorage(
        energy_capacity_mwh=scenario["storage_mwh"],
        power_capacity_mw=scenario["storage_mw"],
        round_trip_efficiency=scenario["storage_eff"],
    )

    dispatch = dispatch_energy(
        years=years,
        demand=demand,
        gas_generation=gas_gen,
        solar_generation=solar_gen,
        storage=storage,
    )

    costs = {
        "gas": gas_cost(gas_gen, econ["GAS_COST_PER_TWH"]),
        "solar": solar_capex(
            solar_cap["capacity_mw"],
            econ["SOLAR_CAPEX_PER_MW"],
            n_years,
        ),
        "unserved": unserved_energy_cost(
            dispatch["unserved"],
            econ["UNSERVED_ENERGY_PENALTY"],
        ),
    }

    if scenario["carbon_active"]:
        costs["carbon"] = carbon_cost(
            gas_gen,
            scenario["carbon_price"],
            econ["CARBON_EMISSION_FACTOR"],
        )
    else:
        costs["carbon"] = 0.0

    costs["total"] = total_system_cost(costs)

    return {
        "served": dispatch["served"],
        "unserved": dispatch["unserved"],
        "costs": costs,
        "units": {
            "annual_energy": "TWh/year",
            "cumulative_energy": "TWh",
            "cost": "USD",
        },
    }


# ============================================================
# DIAGNOSTICS EXTRACTION
# ============================================================

def _series_dict_by_year(values, years):
    return {int(y): float(values[t]) for t, y in enumerate(years)}


def extract_gas_shadow_prices_usd_per_twh_th(m, years):

    out = {}

    for t, y in enumerate(years):
        val = m.dual.get(m.gas_balance[t], None)
        out[int(y)] = None if val is None else float(-val)

    return out


def extract_planning_diagnostics(m, scenario):

    years = scenario["years"]

    gas_avail = gas_available_power(
        start_year=int(years[0]),
        end_year=int(years[-1]),
        scenario_name=scenario["gas_scenario"],
    )["available_twh_th"]

    return {
        "gas_shadow_price_usd_per_twh_th_by_year":
            extract_gas_shadow_prices_usd_per_twh_th(m, years),

        "gas_avail_twh_th_by_year":
            _series_dict_by_year(gas_avail, years),

        "gas_to_power_twh_th_by_year":
            {int(y): float(pyo.value(m.gas_to_power[t]))
             for t, y in enumerate(years)},

        "gas_generation_twh_e_by_year":
            {int(y): float(pyo.value(m.gas_generation[t]))
             for t, y in enumerate(years)},

        "unserved_twh_by_year":
            {int(y): float(pyo.value(m.unserved[t]))
             for t, y in enumerate(years)},

        "storage_discharge_twh_e_by_year":
            {int(y): float(pyo.value(m.storage_discharge[t]))
             for t, y in enumerate(years)},

        "storage_binding_by_year":
            {
                int(y):
                    "cycle_limit"
                    if abs(
                        pyo.value(m.storage_cycle_limit[t].body)
                        - pyo.value(m.storage_cycle_limit[t].upper)
                    ) <= 1e-6
                    else "solar_limit"
                    if abs(
                        pyo.value(m.storage_solar_limit[t].body)
                        - pyo.value(m.storage_solar_limit[t].upper)
                    ) <= 1e-6
                    else "power_limit"
                    if abs(
                        pyo.value(m.storage_power_limit[t].body)
                        - pyo.value(m.storage_power_limit[t].upper)
                    ) <= 1e-6
                    else "none"
                for t, y in enumerate(years)
            },
    }


# ============================================================
# PARETO FRONTIERS
# ============================================================

def generate_weighted_pareto(scenario, econ, weight_grid):

    results = []

    for w_cost, w_em in weight_grid:

        if w_cost == 0.0 and w_em == 0.0:
            raise ValueError("Invalid weight pair (0,0).")

        m = build_model(
            scenario=scenario,
            econ=econ,
            weight_cost=w_cost,
            weight_emissions=w_em,
        )

        status = solve_model(m)

        if not status["optimal"]:
            raise RuntimeError("Optimization failed.")

        diagnostics = extract_planning_diagnostics(m, scenario)

        results.append({
            "weights": {"cost": w_cost, "emissions": w_em},
            "decision_variables": {
                "solar_addition_mw_per_year":
                    float(pyo.value(m.solar_addition)),
                "storage_capacity_mwh":
                    float(pyo.value(m.storage_capacity)),
            },
            "objectives": {
                "emissions_tco2":
                    float(pyo.value(m.emissions)),
            },
            "diagnostics": diagnostics,
        })

    return {"results": results}


def generate_epsilon_pareto(scenario, econ, emissions_caps):

    results = []

    for eps in emissions_caps:

        m = build_model(
            scenario=scenario,
            econ=econ,
            emissions_cap=eps,
        )

        status = solve_model(m)

        if not status["optimal"]:
            raise RuntimeError(
                f"Optimization failed for emissions cap {eps}"
            )

        diagnostics = extract_planning_diagnostics(m, scenario)

        results.append({
            "emissions_cap_tco2": float(eps),
            "actual_emissions_tco2":
                float(pyo.value(m.emissions)),
            "decision_variables": {
                "solar_addition_mw_per_year":
                    float(pyo.value(m.solar_addition)),
                "storage_capacity_mwh":
                    float(pyo.value(m.storage_capacity)),
            },
            "diagnostics": diagnostics,
        })

    return {"results": results}


# ============================================================
# BATCH DETERMINISTIC EXECUTION
# ============================================================

def run_all_deterministic_scenarios(
    econ,
    start_year=2025,
    end_year=2045,
):

    results = []

    for demand_level_case in demand_level_scenarios():
        for demand_case in demand_growth_scenarios():
            for gas_case in gas_decline_scenarios():
                for solar_case in solar_capacity_scenarios():
                    for carbon_case in carbon_policy_scenarios():

                        scenario = load_scenario(
                            demand_level_case=demand_level_case,
                            demand_case=demand_case,
                            gas_case=gas_case,
                            gas_deliverability_case="baseline",
                            solar_case=solar_case,
                            carbon_case=carbon_case,
                            start_year=start_year,
                            end_year=end_year,
                        )

                        output = run_deterministic_scenario(
                            scenario=scenario,
                            econ=econ,
                        )

                        results.append({
                            "scenario_labels": {
                                "demand_level": demand_level_case,
                                "demand": demand_case,
                                "gas": gas_case,
                                "solar": solar_case,
                                "carbon": carbon_case,
                            },
                            "total_cost_usd": output["costs"]["total"],
                            "gas_cost_usd": output["costs"]["gas"],
                            "solar_cost_usd": output["costs"]["solar"],
                            "carbon_cost_usd": output["costs"]["carbon"],
                            "unserved_cost_usd": output["costs"]["unserved"],
                            "total_unserved_energy_twh":
                                float(np.sum(output["unserved"])),
                        })

    return results
