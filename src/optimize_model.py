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

import math
import pyomo.environ as pyo
from src.demand import project_baseline_demand
from src.gas_supply import gas_available_power
from src.io import load_solar_capex_by_year, load_storage_capex_by_year

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
    reliability_max_unserved_fraction=None,
    reliability_mode="annual"
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
    df = {t: 1.0 / ((1.0 + r) ** t) for t in T}
    
    m.DF = pyo.Param(T, initialize=df, within=pyo.PositiveReals)
    
    # ----------------------------------------------------------------
    # Time-varying CAPEX from NREL ATB (solar_low / Storage_low).
    # The optimizer sees actual year-t costs rather than frozen 2025 prices.
    # ----------------------------------------------------------------
    _solar_capex_scenario   = scenario.get("solar_capex_scenario",   "solar_low")
    _storage_capex_scenario = scenario.get("storage_capex_scenario", "Storage_low")

    _solar_capex_dict = load_solar_capex_by_year(
        scenario_name=_solar_capex_scenario,
        start_year=int(years[0]),
        end_year=int(years[-1]),
    )
    _storage_capex_dict = load_storage_capex_by_year(
        scenario_name=_storage_capex_scenario,
        start_year=int(years[0]),
        end_year=int(years[-1]),
    )

    solar_capex_t   = {t: _solar_capex_dict[int(years[t])]   for t in T}
    storage_capex_t = {t: _storage_capex_dict[int(years[t])] for t in T}

    m.solar_capex_param   = pyo.Param(T, initialize=solar_capex_t,   within=pyo.PositiveReals)
    m.storage_capex_param = pyo.Param(T, initialize=storage_capex_t, within=pyo.PositiveReals)
    
    # Fixed-tenor annuity factor for EaaS bankability constraint.
    # A fixed tenor decouples the investor's revenue horizon from the planning model's
    # end year - a solar built in 2040 under a 20-year PPA earns revenue through 2060
    # not just to 2045.
    tenor = int(scenario.get("eaas_contract_tenor_years", 20))
    fixed_tenor_npv = {
        t: sum(1.0 / (1.0 + r) ** j for j in range(tenor))
        for t in T
    }
    m.fixed_tenor_npv = pyo.Param(T, initialize=fixed_tenor_npv, within=pyo.PositiveReals)
    # ------------------------------------------------------------
    # Gas parameters
    # ------------------------------------------------------------
    eta = scenario.get("gas_eta", 0.43)

    # ------------------------------------------------------------
    # Storage parameters
    # ------------------------------------------------------------
    storage_deployable_hours = scenario.get("storage_deployable_hours_per_year", 700.0)
    storage_solar_surplus_frac = scenario.get("storage_solar_surplus_frac", 0.20)
    storage_round_trip_eff = scenario.get("storage_round_trip_eff", 0.90)
    storage_duration_hours = scenario.get("storage_duration_hours", 4.0)  # kept for storage_power_mw only

    tariff = scenario.get("solar_service_tariff_usd_per_twh", None)
    financing_regime = scenario.get("financing_regime", "traditional")
    required_margin = scenario.get("required_margin", 1.0)

    voll = econ["UNSERVED_ENERGY_PENALTY"]

    if storage_deployable_hours <= 0 or storage_deployable_hours > 8760:
        raise ValueError("storage_deployable_hours_per_year must be in (0, 8760]")
    if not (0.0 <= storage_solar_surplus_frac <= 1.0):
        raise ValueError("storage_solar_surplus_frac must be in [0, 1]")
    if not (0.0 < storage_round_trip_eff <= 1.0):
        raise ValueError("storage_round_trip_eff must be in (0, 1]")

    # ------------------------------------------------------------
    # Decision Variables
    # ------------------------------------------------------------ 
    m.solar_public_add = pyo.Var(T, domain=pyo.NonNegativeReals)
    m.solar_eaas_add = pyo.Var(T, domain=pyo.NonNegativeReals)
    m.eaas_subsidy = pyo.Var(T, domain=pyo.NonNegativeReals)
    # Storage:     
    m.storage_add = pyo.Var(T, domain=pyo.NonNegativeReals)
    
    gas_baseline = scenario["gas_baseline_mw"]

    retirement_start = scenario.get("gas_baseline_retirement_start_year", None)
    retirement_rate = scenario.get("gas_baseline_retirement_mw_per_year", 0.0)

    def gas_cap_rule(m, t):
        year_t = int(years[t])
        # No new gas capacity additions modelled. Gas generation is fuel-
        # constrained (gas_avail < gas_capacity) throughout the horizon,
        # so gas_add = 0 at optimality in all scenarios.
        if retirement_start is not None and year_t >= retirement_start:
            retired = retirement_rate * (year_t - retirement_start + 1)
            retired = min(retired, gas_baseline)
        else:
            retired = 0.0
        return gas_baseline - retired

    m.gas_capacity_mw = pyo.Expression(T, rule=gas_cap_rule)

    storage_baseline = scenario.get("storage_baseline_mwh", 0.0)

    def storage_cap_rule(m, t):
        return storage_baseline + sum(m.storage_add[k] for k in range(0, t + 1))

    m.storage_capacity_mwh = pyo.Expression(T, rule=storage_cap_rule)
    m.unserved = pyo.Var(T, domain=pyo.NonNegativeReals)
    m.gas_to_power = pyo.Var(T, domain=pyo.NonNegativeReals)
    m.gas_curtail = pyo.Var(T, domain=pyo.NonNegativeReals)
    m.storage_discharge = pyo.Var(T, domain=pyo.NonNegativeReals)
    m.storage_charge = pyo.Var(T, domain=pyo.NonNegativeReals)

    m.storage_power_mw = pyo.Expression(
        T,
        rule=lambda m, t:
            m.storage_capacity_mwh[t] / storage_duration_hours
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

    solar_energy_per_mw = scenario["solar_cf"] * 8760 / 1_000_000
    baseline_mw = scenario["solar_baseline_mw"]
    # Total-horizon NPV of 1 MW solar generation (used in financing gap calc).
    npv_energy = sum(df[t] * solar_energy_per_mw for t in T)

    if financing_regime == "eaas":
        if tariff is None:
            raise ValueError("Tariff must be defined under EaaS regime.")
    public_solar_capex_npv = pyo.quicksum(
        m.DF[t] * m.solar_public_add[t] * m.solar_capex_param[t]
        for t in T
    )

    eaas_solar_capex_npv = pyo.quicksum(
        m.DF[t] * m.solar_eaas_add[t] * m.solar_capex_param[t]
        for t in T
    )
        
    # Solar capacity = baseline + cumulative sum of annual additions
    def solar_cap_rule(m, t):
        return (
            baseline_mw + sum(m.solar_public_add[k] + m.solar_eaas_add[k] 
                              for k in range(0, t + 1))
        )
    m.solar_capacity_mw = pyo.Expression(T, rule=solar_cap_rule)

    # Solar generation
    m.solar_generation = pyo.Expression(
        T, rule=lambda m, t: m.solar_capacity_mw[t] * solar_energy_per_mw)
    
    # ============================================================
    # EaaS BANKABILITY CONSTRAINT (lifetime NPV formulation)
    # make Public capital lower effective CAPEX, making projects bankable
    # ============================================================
    if tariff is not None:
        def eaas_bankability_rule(m, t):
            # Investor cost: actual year-t CAPEX (time-varying via Fix 6).
            capex = m.solar_capex_param[t] * m.solar_eaas_add[t]

            effective_private_cost = capex - m.eaas_subsidy[t]

            # Bankable revenue: tariff × fixed 20-yr annuity × energy_per_mw × MW.
            # fixed_tenor_npv[t] is the annuity factor (sum of discount factors
            # over the contract life), the same for every build year because the
            # contract tenor is fixed, not the remaining model horizon.
            bankable_revenue = (
                tariff
                * m.fixed_tenor_npv[t]
                * solar_energy_per_mw
                * m.solar_eaas_add[t]
            ) / required_margin

            return effective_private_cost <= bankable_revenue

        m.eaas_bankability = pyo.Constraint(T, rule=eaas_bankability_rule)   
    
    # ============================================================
    # RELIABILITY-TRIGGERED EAAS INVESTMENT
    # ============================================================

    def eaas_regime_gate(m, t):
        if financing_regime != "eaas":
            return m.solar_eaas_add[t] == 0
        return pyo.Constraint.Skip

    m.eaas_regime_gate = pyo.Constraint(T, rule=eaas_regime_gate)
    
    # ============================================================
    # SUBSIDY LIMITED TO FINANCING GAP
    # ============================================================
    if tariff is not None:
        # Financing gap is year-specific because both CAPEX and the bankable
        # revenue (fixed_tenor_npv * tariff * energy_per_mw) are time-varying.
        # gap[t] = max(0, actual_capex[t] - bankable_capex[t])
        financing_gap_by_year = {
            t: max(
                0.0,
                solar_capex_t[t]
                - (tariff * fixed_tenor_npv[t] * solar_energy_per_mw / required_margin)
            )
            for t in T
        }
    else:
        financing_gap_by_year = {t: 0.0 for t in T}

    def subsidy_limit_rule(m, t):
        # Subsidy cannot exceed the financing gap per MW of EaaS solar built.
        # financing_gap_by_year is a dict {t: float}; index with [t].
        return m.eaas_subsidy[t] <= financing_gap_by_year[t] * m.solar_eaas_add[t]

    m.eaas_subsidy_limit = pyo.Constraint(T, rule=subsidy_limit_rule)

    # ------------------------------------------------------------
    # Land Constraint (Baseline + Additions Count)
    # ------------------------------------------------------------

    A_max = scenario["land_available_km2"]
    l_sol = scenario["land_intensity_solar_km2_per_mw"]
    l_gas = scenario["land_intensity_gas_km2_per_mw"]
    l_sto = scenario["land_intensity_storage_km2_per_mwh"]

    m.land_constraint = pyo.Constraint(
        T,
        rule=lambda m, t:
            m.solar_capacity_mw[t] * l_sol
            + m.gas_capacity_mw[t] * l_gas
            + m.storage_capacity_mwh[t] * l_sto
            <= A_max
    )

    # ------------------------------------------------------------
    # Storage Constraints
    # ------------------------------------------------------------
    # Storage Constraints (Annual Energy-Throughput Model)
    # Interpretation: storage is an annual energy-shifting device.
    # We do not model intra-annual SOC. Instead, we impose:
    #   (i)  charging is limited to a fixed fraction of solar surplus
    #  (ii)  discharging cannot exceed what was charged (with RTE)
    # (iii)  both flows are bounded by annualized power capacity
    #
    # Deployable hours (H_d): the number of hours per year over which
    # the battery can deliver at rated power. For a 4-hour BESS
    # dispatched during ~6 peak-deficit hours/day across the dry season
    # (approx. 180 days), H_d = 4 × 180 = 720 h/yr is a conservative
    # anchor.
    # Default here: 700 h/yr (explicit, citable, conservative).
    # ------------------------------------------------------------
    
    # Charge limited to solar surplus fraction
    m.storage_charge_limit = pyo.Constraint(
        T,
        rule=lambda m, t:
            m.storage_charge[t]
            <= storage_solar_surplus_frac * m.solar_generation[t],
    )

    # Charge bounded by annualized power capacity (charge side)
    m.storage_charge_power_limit = pyo.Constraint(
        T,
        rule=lambda m, t:
            m.storage_charge[t]
            <= (m.storage_power_mw[t] * storage_deployable_hours) / 1e6,
    )

    # Discharge bounded by energy charged (after RTE)
    m.storage_energy_balance = pyo.Constraint(
        T,
        rule=lambda m, t:
            m.storage_discharge[t]
            <= storage_round_trip_eff * m.storage_charge[t],
    )

    # Discharge bounded by annualized power capacity (discharge side)
    m.storage_power_limit = pyo.Constraint(
        T,
        rule=lambda m, t:
            m.storage_discharge[t]
            <= (m.storage_power_mw[t] * storage_deployable_hours) / 1e6,
    )

    #------------------------------
    # Deployment Constraint
    #------------------------------
    max_build = scenario.get("solar_max_build_mw_per_year", None)
    if max_build is not None:
        m.solar_build_cap = pyo.Constraint(
            T, rule=lambda m, t: m.solar_public_add[t] + m.solar_eaas_add[t] 
            <= max_build
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

    # ------------------------------------------------------------  
    # Reliability Constraint
    # ------------------------------------------------------------
    if reliability_max_unserved_fraction is not None:

        if reliability_mode == "annual":
            # Year-by-year adequacy
            m.reliability_constraint = pyo.Constraint(
                T,
                rule=lambda m, t:
                    m.unserved[t] <= reliability_max_unserved_fraction * demand[t]
            )

        elif reliability_mode == "total":
            # Horizon-wide adequacy
            total_demand = sum(demand[t] for t in T)

            m.reliability_constraint = pyo.Constraint(
                expr=sum(m.unserved[t] for t in T)
                <= reliability_max_unserved_fraction * total_demand
            )

        else:
            raise ValueError("reliability_mode must be 'annual' or 'total'")

    # -------------------------
    # System Cost (Discounted NPV)
    # -------------------------
    gas_opex_npv = pyo.quicksum(
        m.DF[t] * m.gas_to_power[t] * econ["GAS_COST_PER_TWH_TH"]
        for t in T
    )

    # Storage CAPEX: staged build, each year's addition discounted to present.
    storage_capex_npv = pyo.quicksum(
        m.DF[t] * m.storage_add[t] * m.storage_capex_param[t]
        for t in T
    )   
    
    # Storage annual O&M applied to cumulative installed capacity.
    # Breaks LP degeneracy: without this, any storage capacity beyond the charge
    # limit (~10 GWh) has zero marginal cost and the solver picks an arbitrary
    # large value. The 2,000 USD/MWh-yr rate is intentionally conservative
    # (see io.py for rationale).
    storage_om_npv = pyo.quicksum(
        m.DF[t] * m.storage_capacity_mwh[t] * econ.get("STORAGE_OM_PER_MWH_YR", 0.0)
        for t in T
    )

    # Unserved energy penalty (VoLL in USD/TWh)
    unserved_npv = pyo.quicksum(
        m.DF[t] * m.unserved[t] * econ["UNSERVED_ENERGY_PENALTY"]
        for t in T
    )

    m.emissions_by_year = pyo.Expression(
        T,
        rule=lambda m, t:
            m.gas_generation[t] * 1e6
            * econ["CARBON_EMISSION_FACTOR"],
    )


    m.emissions = pyo.Expression(
        expr=pyo.quicksum(m.emissions_by_year[t] for t in T)
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

    system_cost_npv = (
        gas_opex_npv
        + public_solar_capex_npv
        + eaas_solar_capex_npv
        + storage_capex_npv
        + storage_om_npv
        + unserved_npv
        + carbon_cost_npv
    )
    m.system_cost_npv = pyo.Expression(expr=system_cost_npv)
    
    public_budget_npv = scenario.get("public_solar_budget_npv", None)

    if public_budget_npv is not None:
        m.public_budget_constraint = pyo.Constraint(
            expr=pyo.quicksum(
                m.DF[t] * (
                    m.solar_public_add[t] * m.solar_capex_param[t]
                    + m.eaas_subsidy[t]
                )
                for t in T
            ) <= public_budget_npv
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
            expr=system_cost_npv,
            sense=pyo.minimize,
        )

    elif emissions_cap is not None:

        m.emissions_constraint = pyo.Constraint(
            expr=m.emissions <= emissions_cap
        )

        m.objective = pyo.Objective(
            expr=system_cost_npv,
            sense=pyo.minimize,
        )

    else:

        m.objective = pyo.Objective(
            expr=weight_cost * system_cost_npv
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

    res = solver_obj.solve(model, tee=True)
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
