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
from src.gas_supply import gas_available_power


# ============================================================
# MODEL CONSTRUCTION
# ============================================================

def build_model(
    scenario,
    econ,
    weight_cost=1.0,
    weight_emissions=0.0,
    emissions_cap=None,
    emissions_cap_by_year=None,
):
    """
    Build a planning optimization model.
    """

    # ------------------------------------------------------------
    # Defensive objective check
    # ------------------------------------------------------------
    if (
        emissions_cap is None
        and emissions_cap_by_year is None
        and (weight_cost == 0.0 and weight_emissions == 0.0)
    ):
        raise ValueError(
            "Invalid objective: at least one of weight_cost or "
            "weight_emissions must be non-zero when no emissions cap is provided."
        )

    m = pyo.ConcreteModel()
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    years = scenario["years"]
    T = range(len(years))

    # --------------------------
    # Discounting 
    # --------------------------
    r = float(scenario.get("discount_rate", 0.10))
    # t=0 at start year
    df = {t: 1.0 / ((1.0 + r) ** t) for t in T}
    m.DF = pyo.Param(T, initialize=df, within=pyo.PositiveReals)

    # ------------------------------------------------------------
    # Gas parameters
    # ------------------------------------------------------------
    eta = scenario.get("gas_eta", 0.43)

    # ------------------------------------------------------------
    # Storage parameters
    # ------------------------------------------------------------
    storage_cycles_per_year = scenario.get("storage_cycles_per_year", 250.0)
    storage_solar_surplus_frac = scenario.get("storage_solar_surplus_frac", 0.20)
    storage_round_trip_eff = scenario.get("storage_round_trip_eff", 0.90)
    storage_duration_hours = scenario.get("storage_duration_hours", 4.0)

    if storage_duration_hours <= 0:
        raise ValueError("storage_duration_hours must be > 0")
    if storage_cycles_per_year <= 0:
        raise ValueError("storage_cycles_per_year must be > 0")
    if not (0.0 <= storage_solar_surplus_frac <= 1.0):
        raise ValueError("storage_solar_surplus_frac must be in [0, 1]")
    if not (0.0 < storage_round_trip_eff <= 1.0):
        raise ValueError("storage_round_trip_eff must be in (0, 1]")

    # ------------------------------------------------------------
    # Decision Variables
    # ------------------------------------------------------------
    # Solar additions (MW built each year)
    m.solar_add = pyo.Var(T, domain=pyo.NonNegativeReals)
    # Storage: keep as one-shot capacity decision (built in 2025) for simplicity
    m.storage_capacity = pyo.Var(domain=pyo.NonNegativeReals)
    
    m.unserved = pyo.Var(T, domain=pyo.NonNegativeReals)
    m.gas_to_power = pyo.Var(T, domain=pyo.NonNegativeReals)
    m.gas_curtail = pyo.Var(T, domain=pyo.NonNegativeReals)
    m.storage_discharge = pyo.Var(T, domain=pyo.NonNegativeReals)

    # storage power proxy unchanged
    storage_duration_hours = scenario.get("storage_duration_hours", 4.0)
    m.storage_power_mw = pyo.Expression(
        expr=m.storage_capacity / storage_duration_hours
    )

    # ------------------------------------------------------------
    # Demand and Gas Availability
    # ------------------------------------------------------------
    demand = project_baseline_demand(
        base_demand=scenario["base_demand_twh"],
        growth_rate=scenario["demand_growth"],
        start_year=years[0],
        end_year=years[-1],
    )["demand"]

    gas_avail = gas_available_power(
        start_year=int(years[0]),
        end_year=int(years[-1]),
        scenario_name=scenario["gas_scenario"],
    )["available_twh_th"]

    # ------------------------------------------------------------
    # Generation Expressions
    # ------------------------------------------------------------
    m.gas_generation = pyo.Expression(
        T, rule=lambda m, t: eta * m.gas_to_power[t]
    )

    m.gas_balance = pyo.Constraint(
        T,
        rule=lambda m, t:
            m.gas_to_power[t] + m.gas_curtail[t] <= gas_avail[t],
    )

    solar_energy_per_mw = scenario["solar_cf"] * 8760 / 1e6
    baseline_mw = scenario["solar_baseline_mw"]

    # Solar capacity = baseline + cumulative sum of annual additions
    def solar_cap_rule(m, t):
        return baseline_mw + sum(m.solar_add[k] for k in range(0, t + 1))

    m.solar_capacity_mw = pyo.Expression(T, rule=solar_cap_rule)

    # Solar generation
    m.solar_generation = pyo.Expression(
        T, rule=lambda m, t: m.solar_capacity_mw[t] * solar_energy_per_mw)

    # ------------------------------------------------------------
    # Storage Constraints
    # ------------------------------------------------------------
    m.storage_cycle_limit = pyo.Constraint(
        T,
        rule=lambda m, t:
            m.storage_discharge[t]
            <= (m.storage_capacity / 1e6) * storage_cycles_per_year,
    )

    m.storage_solar_limit = pyo.Constraint(
        T,
        rule=lambda m, t:
            m.storage_discharge[t]
            <= storage_round_trip_eff
            * storage_solar_surplus_frac
            * m.solar_generation[t],
    )

    storage_critical_hours = scenario.get(
        "storage_critical_hours_per_year", 1460.0
    )

    if storage_critical_hours <= 0 or storage_critical_hours > 8760:
        raise ValueError(
            "storage_critical_hours_per_year must be in (0, 8760]"
        )

    m.storage_power_limit = pyo.Constraint(
        T,
        rule=lambda m, t:
            m.storage_discharge[t]
            <= (m.storage_power_mw * storage_critical_hours) / 1e6,
    )

    #------------------------------
    # Deployment Constraint
    #------------------------------
    max_build = scenario.get("solar_max_build_mw_per_year", None)
    if max_build is not None:
        m.solar_build_cap = pyo.Constraint(
            T, rule=lambda m, t: m.solar_add[t] <= max_build
        )

    # ------------------------------------------------------------
    # Electricity Balance
    # ------------------------------------------------------------
    m.balance = pyo.Constraint(
        T,
        rule=lambda m, t:
            m.gas_generation[t]
            + m.solar_generation[t]
            + m.storage_discharge[t]
            + m.unserved[t]
            >= demand[t],
    )

    # -------------------------
    # System Cost (Discounted NPV)
    # -------------------------
    # Gas cost is in USD per TWh_th
    gas_opex_npv = pyo.quicksum(
        m.DF[t] * m.gas_to_power[t] * econ["GAS_COST_PER_TWH_TH"]
        for t in T
    )

    # Solar CAPEX: pay when built (year t)
    solar_capex_npv = pyo.quicksum(
        m.DF[t] * m.solar_add[t] * econ["SOLAR_CAPEX_PER_MW"]
        for t in T
    )

    # Storage CAPEX: assume built in 2025 (t=0) unless you model staged build
    storage_capex_npv = m.DF[0] * m.storage_capacity * econ["STORAGE_COST_PER_MWH"]

    # Unserved energy penalty (VoLL in USD/TWh)
    unserved_npv = pyo.quicksum(
        m.DF[t] * m.unserved[t] * econ["UNSERVED_ENERGY_PENALTY"]
        for t in T
    )

    # Carbon price cost (only when carbon_active is True and you're treating it as a price, not a cap)
    carbon_cost_npv = 0.0
    if scenario.get("carbon_active", False):
        carbon_price = float(scenario["carbon_price"])
        # emissions_by_year is tCO2; multiply by USD/tCO2
        carbon_cost_npv = pyo.quicksum(
            m.DF[t] * m.emissions_by_year[t] * carbon_price
            for t in T
        )

    system_cost_npv = gas_opex_npv + solar_capex_npv + storage_capex_npv + unserved_npv + carbon_cost_npv

    # ------------------------------------------------------------
    # Emissions
    # ------------------------------------------------------------
    m.emissions_by_year = pyo.Expression(
        T,
        rule=lambda m, t:
            m.gas_generation[t] * 1e6
            * econ["CARBON_EMISSION_FACTOR"],
    )

    m.emissions = pyo.Expression(
        expr=pyo.quicksum(m.emissions_by_year[t] for t in T)
    )

    # ------------------------------------------------------------
    # Objective & Emissions Constraints
    # ------------------------------------------------------------
    if emissions_cap_by_year is not None:

        if len(emissions_cap_by_year) != len(list(T)):
            raise ValueError(
                "emissions_cap_by_year length mismatch."
            )

        m.emissions_constraint = pyo.Constraint(
            T,
            rule=lambda m, t:
                m.emissions_by_year[t] <= emissions_cap_by_year[t],
        )

        m.objective = pyo.Objective(
            expr=system_cost,
            sense=pyo.minimize,
        )

    elif emissions_cap is not None:

        m.emissions_constraint = pyo.Constraint(
            expr=m.emissions <= emissions_cap
        )

        m.objective = pyo.Objective(
            expr=system_cost,
            sense=pyo.minimize,
        )

    else:

        m.objective = pyo.Objective(
            expr=weight_cost * system_cost
            + weight_emissions * m.emissions,
            sense=pyo.minimize,
        )

    return m


# ============================================================
# SOLVER WRAPPER
# ============================================================

def solve_model(model, solver="cbc"):
    """
    Solve a Pyomo model and return a minimal status dictionary.
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
        raise RuntimeError(
            f"Solver failed with termination condition: {tc}"
        )

    return {
        "status": str(tc),
        "optimal": tc == pyo.TerminationCondition.optimal,
    }
