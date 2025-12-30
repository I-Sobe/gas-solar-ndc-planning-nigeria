"""
Optimization model definition and solution utilities.

Scope:
- Planning-level Pyomo model construction
- Weighted-sum and ε-constraint formulations
- Solver invocation with termination checks

Non-scope:
- Scenario scripting
- Deterministic evaluation
- Pareto orchestration
"""

import pyomo.environ as pyo
from src.demand import project_baseline_demand
from src.gas_supply import gas_generation_cap


def build_model(
    scenario,
    econ,
    weight_cost=1.0,
    weight_emissions=0.0,
    emissions_cap=None,
):
    """
    Build a planning optimization model.

    Decision variables
    ------------------
    solar_addition : MW/year
        Slope of linear solar capacity build-out.
    storage_capacity : MWh
        Annualized energy buffer (perfectly dispatchable).
    unserved[t] : TWh
        Energy not served (slack variable).

    Modes
    -----
    - Weighted-sum (cost + emissions)
    - ε-constraint (min cost s.t. emissions ≤ ε)
    """

    # ---- Defensive objective check
    if emissions_cap is None and (weight_cost == 0.0 and weight_emissions == 0.0):
        raise ValueError(
            "Invalid objective: at least one of weight_cost or "
            "weight_emissions must be non-zero when emissions_cap is None."
        )

    m = pyo.ConcreteModel()

    years = scenario["years"]
    T = range(len(years))

    # ---- Decision variables
    m.solar_addition = pyo.Var(domain=pyo.NonNegativeReals)
    m.storage_capacity = pyo.Var(domain=pyo.NonNegativeReals)
    m.unserved = pyo.Var(T, domain=pyo.NonNegativeReals)

    # ---- Fixed trajectories
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

    # ---- Solar generation (linear capacity trajectory)
    solar_energy_per_mw = scenario["solar_cf"] * 8760 / 1e6  # TWh per MW
    baseline_mw = scenario["solar_baseline_mw"]

    def solar_rule(m, t):
        return (baseline_mw + m.solar_addition * t) * solar_energy_per_mw

    m.solar_generation = pyo.Expression(T, rule=solar_rule)

    # ---- Balance constraint
    # Storage is modeled as a perfectly dispatchable annual energy buffer,
    # uniformly spread across the planning horizon (no power constraint,
    # no efficiency losses, planning-level approximation).
    def balance_rule(m, t):
        return (
            gas_gen[t]
            + m.solar_generation[t]
            + m.storage_capacity / len(T)
            + m.unserved[t]
            >= demand[t]
        )

    m.balance = pyo.Constraint(T, rule=balance_rule)

    # ---- System cost
    system_cost = (
        pyo.quicksum(gas_gen[t] * econ["GAS_COST_PER_TWH"] for t in T)
        + m.solar_addition * econ["SOLAR_CAPEX_PER_MW"] * len(T)
        + m.storage_capacity * econ["STORAGE_COST_PER_MWH"]
        + pyo.quicksum(m.unserved[t] for t in T) * econ["UNSERVED_ENERGY_PENALTY"]
    )

    # ---- Emissions
    m.emissions = pyo.Expression(
        expr=pyo.quicksum(
            gas_gen[t] * 1e6 * econ["CARBON_EMISSION_FACTOR"] for t in T
        )
    )

    # ---- Objective
    if emissions_cap is not None:
        m.emissions_constraint = pyo.Constraint(
            expr=m.emissions <= emissions_cap
        )
        m.objective = pyo.Objective(expr=system_cost, sense=pyo.minimize)
    else:
        m.objective = pyo.Objective(
            expr=weight_cost * system_cost + weight_emissions * m.emissions,
            sense=pyo.minimize,
        )

    return m


def solve_model(model, solver="cbc"):
    """
    Solve a Pyomo model and return a minimal status dictionary.

    Raises
    ------
    RuntimeError
        If solver is unavailable or solution is infeasible.
    """
    solver_obj = pyo.SolverFactory(solver)
    if not solver_obj.available():
        raise RuntimeError(f"Solver not available: {solver}")

    res = solver_obj.solve(model, tee=False)
    tc = res.solver.termination_condition

    if tc not in (
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
    ):
        raise RuntimeError(f"Solver failed with termination condition: {tc}")

    return {
        "status": str(tc),
        "optimal": tc == pyo.TerminationCondition.optimal,
    }
