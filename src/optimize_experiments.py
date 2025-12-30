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
import pyomo.environ as pyo

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
from src.gas_supply import gas_generation_cap
from src.scenarios import (
    load_scenario,
    demand_growth_scenarios,
    gas_decline_scenarios,
    solar_capacity_scenarios,
    carbon_policy_scenarios,
)
# ============================================================
# DETERMINISTIC OPERATIONAL SCENARIO
# ============================================================

def run_deterministic_scenario(scenario, econ):
    """
    Run a deterministic operational simulation (no optimization).
    Important
    ---------
    - Storage is modeled explicitly using BatteryStorage
      (power + energy + efficiency).
    - A fresh storage instance is created per run.
    - Results are operational diagnostics and are NOT directly
      comparable to planning optimization outputs.
    """

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

    # Planning-level solar capacity trajectory (deterministic, exogenous)
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

    # Fresh storage instance per scenario run (stateful)
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

    # ---- Cost aggregation (shared economics)
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
        "served": dispatch["served"],            # TWh/year
        "unserved": dispatch["unserved"],        # TWh/year
        "costs": costs,
        "units": {
            "annual_energy": "TWh/year",
            "cumulative_energy": "TWh",
            "cost": "USD",
        },
    }
# ============================================================
# PARETO FRONT GENERATION
# ============================================================

def generate_weighted_pareto(scenario, econ, weight_grid):
    """
    Generate Pareto frontier using weighted-sum formulation.
    """

    results = []

    for w_cost, w_em in weight_grid:
        if w_cost == 0.0 and w_em == 0.0:
            raise ValueError(
                "Invalid weight pair (0,0): objective is undefined."
            )

        m = build_model(
            scenario,
            econ,
            weight_cost=w_cost,
            weight_emissions=w_em,
        )

        status = solve_model(m)

        if not status["optimal"]:
            raise RuntimeError(
                f"Optimization failed for weights "
                f"(cost={w_cost}, emissions={w_em})"
            )

        results.append({
            "weights": {
                "cost": w_cost,
                "emissions": w_em,
            },
            "decision_variables": {
                "solar_addition_mw_per_year": pyo.value(m.solar_addition),
                "storage_capacity_mwh": pyo.value(m.storage_capacity),
            },
            "objectives": {
                "emissions_tco2": pyo.value(m.emissions),
            },
        })

    return {
        "results": results,
        "units": {
            "solar_addition_mw_per_year": "MW/year",
            "storage_capacity_mwh": "MWh",
            "emissions_tco2": "tCO2",
        },
    }


def generate_epsilon_pareto(scenario, econ, emissions_caps):
    """
    Generate Pareto frontier using ε-constraint formulation.
    """

    results = []

    for eps in emissions_caps:
        m = build_model(
            scenario,
            econ,
            emissions_cap=eps,
        )

        status = solve_model(m)

        if not status["optimal"]:
            raise RuntimeError(
                f"Optimization failed for emissions cap {eps}"
            )

        results.append({
            "emissions_cap_tco2": eps,
            "actual_emissions_tco2": pyo.value(m.emissions),
            "decision_variables": {
                "solar_addition_mw_per_year": pyo.value(m.solar_addition),
                "storage_capacity_mwh": pyo.value(m.storage_capacity),
            },
        })

    return {
        "results": results,
        "units": {
            "solar_addition_mw_per_year": "MW/year",
            "storage_capacity_mwh": "MWh",
            "emissions_tco2": "tCO2",
        },
    }
# ============================================================
# DETERMINISTIC BATCH EXECUTION (SCENARIO MATRIX)
# ============================================================

def run_all_deterministic_scenarios(
    econ,
    start_year=2025,
    end_year=2045,
):
    """
    Execute deterministic operational simulations across all
    combinations of predefined planning scenarios.

    Notes
    -----
    - Results are OPERATIONAL diagnostics.
    - Energy values aggregated over time represent cumulative
      energy over the planning horizon (TWh), not annual rates.
    """

    results = []

    for demand_case in demand_growth_scenarios():
        for gas_case in gas_decline_scenarios():
            for solar_case in solar_capacity_scenarios():
                for carbon_case in carbon_policy_scenarios():

                    scenario = load_scenario(
                        demand_case=demand_case,
                        gas_case=gas_case,
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
                        "total_unserved_energy_twh": float(
                            np.sum(output["unserved"])
                        ),
                    })

    return results
