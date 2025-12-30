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

Deterministic system evaluation and planning-level optimization
for Nigeria’s gas–solar electricity system.

This module serves two purposes:

1. Deterministic system evaluation (no solver)
2. Multi-objective planning optimization (Pyomo)

Objectives supported:
- Cost minimization
- Emissions minimization
- Reliability enforcement via unserved energy penalty

Storage is modeled as an annual energy buffer.
"""

import numpy as np
import pyomo.environ as pyo

from src.demand import project_baseline_demand
from src.gas_supply import gas_generation_cap
from src.solar import solar_generation, solar_capacity_trajectory
from src.storage import BatteryStorage
from src.dispatch import dispatch_energy


# ============================================================
# GLOBAL COST & EMISSIONS PARAMETERS
# ============================================================

GAS_COST_PER_TWH = 50e6              # USD / TWh
SOLAR_CAPEX_PER_MW = 800_000         # USD / MW
STORAGE_COST_PER_MWH = 300_000       # USD / MWh (planning-level)
UNSERVED_ENERGY_PENALTY = 1e9        # USD / TWh (Value of Lost Load)
CARBON_EMISSION_FACTOR = 0.4         # tCO2 / MWh


# ============================================================
# DETERMINISTIC SYSTEM EVALUATION (NO OPTIMIZATION)
# ============================================================

def run_deterministic_model(scenario):
    """
    Evaluate system performance for a fixed scenario.

    Parameters
    ----------
    scenario : dict
        Scenario configuration from scenarios.load_scenario()

    Returns
    -------
    dict
        Time series outputs and aggregated cost metrics
    """

    years = scenario["years"]
    n_years = len(years)

    # ---- Demand projection
    demand = project_baseline_demand(
        base_demand=30.0,
        growth_rate=scenario["demand_growth"],
        start_year=years[0],
        end_year=years[-1],
    )["demand"]

    # ---- Gas supply
    gas_gen = gas_generation_cap(
        q0=40.0,
        decline_rate=scenario["gas_decline"],
        start_year=years[0],
        end_year=years[-1],
    )["generation"]

    # ---- Solar supply
    solar_cfg = scenario["solar"]
    solar_cap = solar_capacity_trajectory(
        initial_capacity_mw=solar_cfg["initial_capacity_mw"],
        annual_addition_mw=solar_cfg["annual_addition_mw"],
        start_year=years[0],
        end_year=years[-1],
    )

    solar_gen = solar_generation(
        capacity_mw=solar_cap["capacity_mw"],
        capacity_factor=0.22,
    )

    # ---- Storage (fixed in deterministic mode)
    storage = BatteryStorage(
        energy_capacity_mwh=20_000,
        power_capacity_mw=2_000,
        round_trip_efficiency=0.9,
    )

    # ---- Dispatch
    dispatch = dispatch_energy(
        years=years,
        demand=demand,
        gas_generation=gas_gen,
        solar_generation=solar_gen,
        storage=storage,
    )

    # ---- Costs
    gas_cost = np.sum(gas_gen) * GAS_COST_PER_TWH
    solar_cost = np.sum(solar_cap["capacity_mw"]) * SOLAR_CAPEX_PER_MW / n_years

    carbon_cost = 0.0
    if scenario["carbon_policy"]["active"]:
        carbon_cost = (
            np.sum(gas_gen) * 1e6
            * CARBON_EMISSION_FACTOR
            * 50.0  # baseline carbon price
        )

    unserved_cost = np.sum(dispatch["unserved"]) * UNSERVED_ENERGY_PENALTY
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


# ============================================================
# MULTI-OBJECTIVE PLANNING OPTIMIZATION (PYOMO)
# ============================================================

def build_model(
    scenario,
    weight_cost=1.0,
    weight_emissions=0.0,
):
    """
    Build a weighted-sum planning optimization model.

    Decision variables:
    - Annual solar capacity addition (MW/year)
    - Storage energy capacity (MWh)
    - Unserved energy (slack variable)

    Storage is treated as an annual energy buffer.
    """

    m = pyo.ConcreteModel()

    years = scenario["years"]
    T = range(len(years))

    # ---- Decision variables
    m.solar_addition = pyo.Var(domain=pyo.NonNegativeReals)
    m.storage_capacity = pyo.Var(domain=pyo.NonNegativeReals)
    m.unserved = pyo.Var(T, domain=pyo.NonNegativeReals)

    # ---- Fixed trajectories
    demand = project_baseline_demand(
        base_demand=30.0,
        growth_rate=scenario["demand_growth"],
        start_year=years[0],
        end_year=years[-1],
    )["demand"]

    gas_gen = gas_generation_cap(
        q0=40.0,
        decline_rate=scenario["gas_decline"],
        start_year=years[0],
        end_year=years[-1],
    )["generation"]

    solar_energy_per_mw = 0.22 * 8760 / 1e6  # TWh per MW

    # ---- Expressions
    m.solar_generation = pyo.Expression(
        T,
        rule=lambda m, t: (500 + m.solar_addition * t) * solar_energy_per_mw,
    )

    # ---- Power balance with storage & slack
    def balance_rule(m, t):
        return (
            gas_gen[t]
            + m.solar_generation[t]
            + m.storage_capacity / len(T)
            + m.unserved[t]
            >= demand[t]
        )

    m.balance = pyo.Constraint(T, rule=balance_rule)

    # ---- Objective components
    system_cost = (
        pyo.quicksum(gas_gen[t] * GAS_COST_PER_TWH for t in T)
        + m.solar_addition * SOLAR_CAPEX_PER_MW * len(T)
        + m.storage_capacity * STORAGE_COST_PER_MWH
        + pyo.quicksum(m.unserved[t] for t in T) * UNSERVED_ENERGY_PENALTY
    )

    emissions = pyo.quicksum(
        gas_gen[t] * 1e6 * CARBON_EMISSION_FACTOR for t in T
    )

    # ---- Weighted-sum objective
    m.objective = pyo.Objective(
        expr=(
            weight_cost * system_cost
            + weight_emissions * emissions
        ),
        sense=pyo.minimize,
    )

    return m


def solve_model(model, solver="cbc"):
    """Solve a Pyomo optimization model."""
    return pyo.SolverFactory(solver).solve(model, tee=False)


def generate_pareto_front(scenario, weight_grid):
    """
    Generate Pareto-optimal solutions using weighted sums.
    """

    results = []

    for w_cost, w_em in weight_grid:
        model = build_model(
            scenario,
            weight_cost=w_cost,
            weight_emissions=w_em,
        )
        solve_model(model)

        results.append({
            "w_cost": w_cost,
            "w_emissions": w_em,
            "solar_addition": pyo.value(model.solar_addition),
            "storage_capacity": pyo.value(model.storage_capacity),
        })

    return results
