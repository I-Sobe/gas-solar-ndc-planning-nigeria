"""
Optimization Module

Constructs and solves the multi-objective optimization problem
using ε-constraint or weighted-sum approaches. Produces
Pareto-optimal solutions for cost, emissions, and reliability.

Functions:
    build_model(inputs)
    solve_model(model, solver='cbc')
    generate_pareto_front(...)
"""

"""
optimize.py
Deterministic system evaluation and cost calculation
"""

import numpy as np

from src.demand import project_baseline_demand
from src.gas_supply import gas_generation_cap
from src.solar import solar_generation, solar_capacity_trajectory
from src.storage import BatteryStorage
from src.dispatch import dispatch_energy


# ----------------------------
# COST PARAMETERS (BASELINE)
# ----------------------------
GAS_COST_PER_TWH = 50e6        # $/TWh
SOLAR_CAPEX_PER_MW = 800_000   # $/MW
CARBON_EMISSION_FACTOR = 0.4  # tCO2/MWh
UNSERVED_ENERGY_PENALTY = 1e9 # $/TWh (Value of Lost Load)


def run_deterministic_model(scenario):
    """
    Run deterministic system model for a given scenario.

    Parameters
    ----------
    scenario : dict
        Output of scenarios.build_scenario()

    Returns
    -------
    dict
        System outputs and cost breakdown
    """

    years = scenario["years"]
    n_years = len(years)

    # ----------------------------
    # DEMAND
    # ----------------------------
    demand_out = project_baseline_demand(
        base_demand=30.0,  # TWh (calibration point)
        growth_rate=scenario["demand_growth"],
        start_year=years[0],
        end_year=years[-1],
    )
    demand = demand_out["demand"]

    # ----------------------------
    # GAS SUPPLY
    # ----------------------------
    gas_out = gas_generation_cap(
        q0=40.0,  # TWh initial gas generation
        decline_rate=scenario["gas_decline"],
        start_year=years[0],
        end_year=years[-1],
    )
    gas_gen = gas_out["generation"]

    # ----------------------------
    # SOLAR SUPPLY
    # ----------------------------
    solar_cfg = scenario["solar"]

    solar_cap = solar_capacity_trajectory(
        initial_capacity_mw=solar_cfg["initial_capacity_mw"],
        annual_addition_mw=solar_cfg["annual_addition_mw"],
        start_year=years[0],
        end_year=years[-1],
    )

    # Assume fixed capacity factor (from PVGIS, Nigeria ~22%)
    solar_gen = solar_generation(
        capacity_mw=solar_cap["capacity_mw"],
        capacity_factor=0.22,
    )

    # ----------------------------
    # STORAGE
    # ----------------------------
    storage = BatteryStorage(
        energy_capacity_mwh=20_000,  # 20 GWh
        power_capacity_mw=2_000,
        round_trip_efficiency=0.9,
    )

    # ----------------------------
    # DISPATCH
    # ----------------------------
    dispatch = dispatch_energy(
        years=years,
        demand=demand,
        gas_generation=gas_gen,
        solar_generation=solar_gen,
        storage=storage,
    )

    # ----------------------------
    # COST CALCULATIONS
    # ----------------------------
    gas_cost = np.sum(gas_gen) * GAS_COST_PER_TWH

    solar_cost = (
        np.sum(solar_cap["capacity_mw"]) * SOLAR_CAPEX_PER_MW / n_years
    )

    if scenario["carbon_policy"]["active"]:
        carbon_price = 50.0  # $/tCO2 (baseline placeholder)
        carbon_cost = (
            np.sum(gas_gen) * 1e6
            * CARBON_EMISSION_FACTOR
            * carbon_price
        )
    else:
        carbon_cost = 0.0

    unserved_cost = (
        np.sum(dispatch["unserved"]) * UNSERVED_ENERGY_PENALTY
    )

    total_cost = gas_cost + solar_cost + carbon_cost + unserved_cost

    return {
        "years": years,
        "demand": demand,
        "gas_generation": gas_gen,
        "solar_generation": solar_gen,
        "served": dispatch["served"],
        "unserved": dispatch["unserved"],
        "costs": {
            "gas": gas_cost,
            "solar": solar_cost,
            "carbon": carbon_cost,
            "unserved": unserved_cost,
            "total": total_cost,
        },
    }
