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
    reliability_mode="annual",
    solar_capex_by_year=None,
):
    """
    Build a planning optimization model.

    Parameters
    ----------
    solar_capex_by_year : dict {year: usd_per_mw} or None
        Time-varying solar CAPEX trajectory from NREL ATB.
        If provided, each year's additions are priced at that year's CAPEX,
        reflecting actual hardware cost declines (not discounting — these are
        separate effects: discounting is time preference, CAPEX decline is
        a real market price trajectory).
        If None, falls back to the fixed econ["SOLAR_CAPEX_PER_MW"] scalar
        (2025 value), which is conservative and overstates post-2030 solar cost.
        Use load_solar_capex_by_year() from io.py to construct this dict.
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
    # Remaining-horizon NPV factor for each year t
    remaining_npv = {}
    for t in T:
        remaining_npv[t] = sum(df[k] / df[t] for k in range(t, len(T)))

    m.remaining_npv_factor = pyo.Param(T, initialize=remaining_npv, within=pyo.PositiveReals)
    m.DF = pyo.Param(T, initialize=df, within=pyo.PositiveReals)

    # ------------------------------------------------------------
    # Solar CAPEX parameter (time-varying or fixed scalar)
    # ------------------------------------------------------------
    # If solar_capex_by_year is provided, build a time-indexed Pyomo Param
    # so each year's additions are priced at that year's actual market cost.
    # This correctly reflects NREL ATB hardware cost trajectories — which fall
    # by ~59% from 2025 to 2045 under solar_low — without double-discounting:
    # discounting (DF) = time preference of money; CAPEX decline = real price path.
    # If not provided, fall back to the fixed 2025 scalar (conservative).
    if solar_capex_by_year is not None:
        capex_init = {
            t: float(solar_capex_by_year[int(years[t])])
            for t in T
        }
    else:
        capex_init = {t: float(econ["SOLAR_CAPEX_PER_MW"]) for t in T}

    m.solar_capex_param = pyo.Param(T, initialize=capex_init, within=pyo.PositiveReals)

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

    
    m.solar_public_add = pyo.Var(T, domain=pyo.NonNegativeReals)
    m.solar_eaas_add = pyo.Var(T, domain=pyo.NonNegativeReals)
    m.eaas_subsidy = pyo.Var(T, domain=pyo.NonNegativeReals)
     
    # No new gas plant construction: fuel availability is always the binding constraint.
    # New gas capacity is structurally non-optimal across all scenarios (verified).
    # gas_add is fixed at zero to make this assumption explicit.
    m.gas_add = pyo.Param(T, initialize=0.0, within=pyo.NonNegativeReals)
    # Storage:
    m.storage_add = pyo.Var(T, domain=pyo.NonNegativeReals)
    
    gas_baseline = scenario["gas_baseline_mw"]

    retirement_start = scenario.get("gas_baseline_retirement_start_year", None)
    retirement_rate = scenario.get("gas_baseline_retirement_mw_per_year", 0.0)

    def gas_cap_rule(m, t):
        year_t = int(years[t])
        # Cumulative retirement of baseline fleet
        if retirement_start is not None and year_t >= retirement_start:
            retired = retirement_rate * (year_t - retirement_start + 1)
            retired = min(retired, gas_baseline)  # cannot retire more than exists
        else:
            retired = 0.0
        return gas_baseline - retired + sum(m.gas_add[k] for k in range(0, t + 1))

    m.gas_capacity_mw = pyo.Expression(T, rule=gas_cap_rule)

    storage_baseline = scenario.get("storage_baseline_mwh", 0.0)

    # Storage always starts from zero — no brownfield BESS baseline in Nigeria (2025).
    # storage_baseline_mwh is set to 0.0 in all scenarios and is intentionally excluded.
    def storage_cap_rule(m, t):
        return sum(m.storage_add[k] for k in range(0, t+1))

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
    m.gas_capacity_constraint = pyo.Constraint(
        T,
        rule=lambda m, t:
            m.gas_generation[t]
            <= m.gas_capacity_mw[t] * 8760 / 1e6

    )
    
    m.gas_balance = pyo.Constraint(
        T,
        rule=lambda m, t:
            m.gas_to_power[t] + m.gas_curtail[t] <= gas_avail[t],
    )

    solar_energy_per_mw = scenario["solar_cf"] * 8760 / 1_000_000
    npv_energy_per_mw = sum(
        pyo.value(m.DF[t]) * solar_energy_per_mw
        for t in T
    )
    baseline_mw = scenario["solar_baseline_mw"]
    # NPV of 1 MW solar generation over horizon
    npv_energy = sum(df[t] * solar_energy_per_mw for t in T)

    solar_lcoe = econ["SOLAR_CAPEX_PER_MW"] / npv_energy
    # Maximum CAPEX investors can support given tariff
    # max_bankable_capex = tariff * npv_energy / required_margin    
    # choose between private or public based on voll > tariff
    if tariff is None:
        eaas_trigger_strength = 0.0
    else:
        eaas_trigger_strength = max(0.0, voll - tariff)
    # Financing Regime Adjustment (EaaS)
    # effective_solar_capex = econ["SOLAR_CAPEX_PER_MW"]

    if financing_regime == "eaas":
        if tariff is None:
            raise ValueError("Tariff must be defined under EaaS regime.")
        
    # Public solar CAPEX — priced at each year's NREL ATB value if available
    public_solar_capex_npv = pyo.quicksum(
        m.DF[t] * m.solar_public_add[t]
        * m.solar_capex_param[t]
        for t in T
    )

    # Private (EaaS) solar CAPEX with premium
    private_capex_multiplier = 1.0
    if financing_regime == "eaas":
        private_capex_multiplier = required_margin

    eaas_solar_capex_npv = pyo.quicksum(
        m.DF[t] * m.solar_eaas_add[t]
        * m.solar_capex_param[t]
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

            capex = m.solar_capex_param[t] * m.solar_eaas_add[t]

            effective_private_cost = capex - m.eaas_subsidy[t]
            bankable_revenue = (tariff * m.remaining_npv_factor[t] * m.solar_eaas_add[t]) / required_margin

            return effective_private_cost <= bankable_revenue

        m.eaas_bankability = pyo.Constraint(T, rule=eaas_bankability_rule)   
    
    # ============================================================
    # EaaS deployment gating (non-EaaS regimes only)
    # ============================================================
    # When financing_regime is not "eaas", solar_eaas_add is structurally
    # disabled. Economic feasibility under EaaS regimes is governed by
    # eaas_bankability and eaas_subsidy_limit constraints below.
    #
    # Reliability-based EaaS triggering is evaluated post-solve as a
    # diagnostic (see extract_planning_diagnostics) — the LP decides
    # deployment optimally without artificial gating.
    # ============================================================

    if financing_regime != "eaas":
        def eaas_disable_rule(m, t):
            return m.solar_eaas_add[t] == 0
        m.eaas_disable = pyo.Constraint(T, rule=eaas_disable_rule)
    
    # ============================================================
    # SUBSIDY LIMITED TO FINANCING GAP
    # ============================================================
    # Subsidy bounded above by financing gap (optimizer chooses how much to deploy)
    # This makes eaas_subsidy a genuine decision variable, not a derived quantity.
    # SUBSIDY LIMITED TO FINANCING GAP
    # The financing gap is year-specific when CAPEX varies: cheaper future solar
    # may be fully bankable even at lower tariffs.
    # financing_gap_per_mw[t] = max(0, CAPEX[t] - max_bankable_capex_at_t)
    # Note: max_bankable_capex uses remaining_npv_factor[t] (horizon-from-t NPV)
    # so the bankability test is consistent per year.
    if tariff is not None:
        def subsidy_limit_rule(m, t):
            max_bankable_at_t = tariff * m.remaining_npv_factor[t] / required_margin
            gap_at_t = pyo.value(m.solar_capex_param[t]) - max_bankable_at_t
            gap_at_t = max(0.0, gap_at_t)
            return m.eaas_subsidy[t] <= gap_at_t * m.solar_eaas_add[t]
    else:
        def subsidy_limit_rule(m, t):
            return m.eaas_subsidy[t] <= 0.0

    m.eaas_subsidy_limit = pyo.Constraint(T, rule=subsidy_limit_rule)

    # Subsidy non-negativity is already enforced by domain=NonNegativeReals on eaas_subsidy
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
    # ------------------------------------------------------------
    # Storage Constraints (Annual Energy-Throughput Model)
    #
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
    # anchor. Calibrate to your system context and cite it.
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

    # Minimum annual solar build rate.
    # Required when time-varying CAPEX is active: the optimizer sees 2040-2045
    # solar as ~2.4x cheaper than 2025 (discounting × price decline combined),
    # creating a strong incentive to delay all builds. A minimum floor prevents
    # pathological all-delay solutions that leave 2025-2030 with no solar.
    # Default 0 MW/yr preserves backward-compatibility when solar_capex_by_year
    # is not passed. Set to 100 MW/yr when using time-varying CAPEX.
    min_build = scenario.get("solar_min_build_mw_per_year", 0.0)
    if min_build > 0:
        m.solar_min_build_constraint = pyo.Constraint(
            T,
            rule=lambda m, t:
                m.solar_public_add[t] + m.solar_eaas_add[t] >= min_build
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
    # Gas cost is in USD per TWh_th
    gas_capex_npv = pyo.quicksum(
        m.DF[t] * m.gas_add[t] * scenario["gas_capex_per_mw"]
        for t in T
    )

    gas_opex_npv = pyo.quicksum(
        m.DF[t] * m.gas_to_power[t] * econ["GAS_COST_PER_TWH_TH"]
        for t in T
    )

    # Storage CAPEX: assume built in 2025 (t=0) unless you model staged build
    storage_capex_npv = pyo.quicksum(
    m.DF[t] * m.storage_add[t] * econ["STORAGE_COST_PER_MWH"]
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
        + gas_capex_npv
        + storage_capex_npv
        + unserved_npv
        + carbon_cost_npv
    )
    m.system_cost_npv = pyo.Expression(expr=system_cost_npv)
    
    # ============================================================
    # COST COMPONENT EXPRESSIONS (for decomposition diagnostics)
    # ============================================================
    m.cost_gas_opex_npv          = pyo.Expression(expr=gas_opex_npv)
    m.cost_gas_capex_npv         = pyo.Expression(expr=gas_capex_npv)
    m.cost_public_solar_capex_npv = pyo.Expression(expr=public_solar_capex_npv)
    m.cost_eaas_solar_capex_npv   = pyo.Expression(expr=eaas_solar_capex_npv)
    m.cost_storage_capex_npv      = pyo.Expression(expr=storage_capex_npv)
    m.cost_unserved_voll_npv      = pyo.Expression(expr=unserved_npv)
    m.cost_carbon_npv             = pyo.Expression(expr=carbon_cost_npv)
    
    public_budget_npv = scenario.get("public_solar_budget_npv", None)

    if public_budget_npv is not None:
        # ============================================================
        # PUBLIC CAPITAL BUDGET CONSTRAINT
        # ============================================================
        # The public capital envelope covers all generation-side CAPEX:
        #   - Public solar CAPEX (utility-scale solar additions)
        #   - Storage CAPEX (battery storage additions)
        #   - EaaS subsidy (public gap-funding for private solar)
        #
        # This reflects Nigerian fiscal reality where solar and storage
        # compete for the same infrastructure capital allocation from
        # federal budget and international development finance.
        # ============================================================
        m.public_budget_constraint = pyo.Constraint(
            expr=pyo.quicksum(
                m.DF[t] * (
                    m.solar_public_add[t] * m.solar_capex_param[t]
                    + m.storage_add[t] * econ["STORAGE_COST_PER_MWH"]
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
