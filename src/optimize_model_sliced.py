"""
Time-sliced planning model (Phase 2.1).

Parallel implementation of build_model() with intra-annual resolution.
The annual model in optimize_model.py is retained unchanged so that:
  (a) existing runners keep working during migration, and
  (b) the annual-vs-sliced comparison is available as a methods result.

WHAT SLICING FIXES
------------------
At annual resolution the LP treats energy as fungible within a year, so
1 TWh of midday solar can "serve" the evening peak. That is physically false
and it is the single largest source of solar over-credit in the annual model.
It also makes storage structurally incapable of value (there is no intra-annual
time for it to shift energy across), and it hides the dry-season adequacy
problem behind an annual average.

SLICE STRUCTURE — 6 slices, {dry, wet} x {night, day, peak}
------------------------------------------------------------
Seasons  : dry = Nov-Apr (181 d), wet = May-Oct (184 d)
Periods  : night 22:00-05:00 (7 h), day 05:00-18:00 (13 h), peak 18:00-22:00 (4 h)

    slice        hours/yr   share
    dry_night        1267   14.5%
    dry_day          2353   26.9%
    dry_peak          724    8.3%
    wet_night        1288   14.7%
    wet_day          2392   27.3%
    wet_peak          736    8.4%
                     8760

Period boundaries follow the Nigerian daily load shape (night trough, morning
rise from 05:00, evening peak 18:00-22:00). The day/peak boundary is set by
SOLAR AVAILABILITY, not by the demand curve: solar generates in day slices only.

WHAT IS AND IS NOT SLICED
-------------------------
Sliced [T,S]      : gas_to_power, gas_curtail, unserved, storage_charge,
                    storage_discharge, hydro_gen, and the electricity balance.
Annual [T]        : all capacity additions, all financial constraints
                    (bankability, subsidy, concessional envelope, public
                    budget), land, and build caps. Capacity is built once a
                    year, not per slice.
Annual, summed    : gas fuel budget, emissions, reliability.

Gas fuel becomes an ANNUAL BUDGET allocated across slices, so the optimiser
can now save scarce gas for the dry-season evening peak instead of spreading
it evenly. That is real dispatch logic the annual model cannot express.
"""

import pyomo.environ as pyo
from src.demand import project_baseline_demand
from src.gas_supply import gas_available_power
from src.blended_finance import resolve_private_rate


# ============================================================
# SLICE DEFINITION
# ============================================================

SEASON_DAYS = {"dry": 181, "wet": 184}      # dry = Nov-Apr, wet = May-Oct
PERIOD_HOURS = {"night": 7, "day": 13, "peak": 4}   # 22-05, 05-18, 18-22
SLICES = [(s, p) for s in ("dry", "wet") for p in ("night", "day", "peak")]

SOLAR_PERIODS = ("day",)   # solar generates in day slices ONLY


def slice_hours():
    """Hours per year represented by each slice. Sums to 8760."""
    h = {(s, p): SEASON_DAYS[s] * PERIOD_HOURS[p] for (s, p) in SLICES}
    total = sum(h.values())
    if total != 8760:
        raise ValueError(f"Slice hours sum to {total}, expected 8760")
    return h


# ============================================================
# DEFAULT SHAPE PARAMETERS  --  ALL REQUIRE SOURCING
# ============================================================
# These are DECLARED ASSUMPTIONS, not measurements. Each is exposed as a
# scenario key so it can be swept. Do not report a sliced result without
# either sourcing these or reporting the sweep.

# Relative average POWER by season (normalised internally, so only the ratio
# matters). Dry season (Nov-Apr) demand is higher: hot dry months, peak
# Jan-Apr per the cited Nigerian load research.
# [SOURCE NEEDED: quantitative seasonal demand ratio]
DEFAULT_SEASON_DEMAND_FACTOR = {"dry": 1.08, "wet": 0.92}

# Relative average POWER by period. Deliberately FLATTER than a residential
# profile: in aggregate national load, commercial and industrial daytime
# consumption offsets the residential midday dip. Nigeria has little nocturnal
# economy (22:00-05:00), so night is the genuine trough.
# NOTE: observed grid data CANNOT settle this — load shedding truncates the
# peak, so the observed load-duration curve is flat-topped by construction.
# Sweep this; do not present it as measured.
# [SOURCE NEEDED: quantitative intra-day load shape]
DEFAULT_PERIOD_DEMAND_FACTOR = {"night": 0.65, "day": 1.05, "peak": 1.20}

# Relative solar yield by season, applied to day slices only.
# Nigerian PVOUT is higher in the dry season (clear skies) than the wet season
# (monsoon cloud), partially offset by harmattan dust Dec-Feb.
# CAVEAT: this ratio is a reasoned placeholder, NOT sourced. It MUST be
# replaced with Global Solar Atlas MONTHLY PVOUT for the named candidate sites,
# aggregated to the Nov-Apr / May-Oct split. Until then, treat any
# season-resolved solar result as provisional.
# [SOURCE NEEDED: Global Solar Atlas monthly PVOUT, named sites]
DEFAULT_SEASON_SOLAR_FACTOR = {"dry": 1.15, "wet": 0.85}

# Hydro seasonal energy split. The 2025 quarterly data cannot establish this:
# absolute hydro is near-flat Q1-Q3 (3.9% spread) but Zungeru was ramping
# through the year, so a genuine seasonal decline may be masked by a
# commissioning trend. Large reservoir plants (Kainji, Jebba, Shiroro) do
# buffer inflow, so partial flatness is physically plausible.
# Central case: dry-season hydro at 0.90 of wet-season average power.
# Sweep {0.75, 0.90, 1.00} = "meaningful decline" .. "fully buffered".
# [SOURCE NEEDED: pre-Zungeru quarterly generation levels, 2022-2023]
DEFAULT_HYDRO_SEASON_FACTOR = {"dry": 0.90, "wet": 1.10}

# Minimum hydro output per slice as a fraction of its proportional share,
# representing environmental flow, irrigation and downstream obligations.
# Without a floor the LP concentrates all hydro into peak slices, which
# reservoir operating constraints do not permit.
# [SOURCE NEEDED: minimum flow obligations for Kainji/Jebba/Shiroro]
DEFAULT_HYDRO_MIN_FLOW_FRACTION = 0.50


def _normalised_demand_weights(scenario):
    """
    Fraction of annual demand energy falling in each slice.

    Built from relative average power by season and period, weighted by slice
    hours, then normalised so the fractions sum to 1.
    """
    h = slice_hours()
    season_f = scenario.get("season_demand_factor", DEFAULT_SEASON_DEMAND_FACTOR)
    period_f = scenario.get("period_demand_factor", DEFAULT_PERIOD_DEMAND_FACTOR)

    raw = {
        (s, p): h[(s, p)] * season_f[s] * period_f[p]
        for (s, p) in SLICES
    }
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def _solar_energy_per_mw_by_slice(scenario):
    """
    Solar energy (TWh per MW installed) in each slice.

    The annual capacity factor is redistributed across day slices only, so the
    total annual yield per MW is preserved but the intra-annual placement is
    physically correct: zero output at night and in the evening peak.
    """
    h = slice_hours()
    cf_annual = scenario["solar_cf"]
    season_f = scenario.get("season_solar_factor", DEFAULT_SEASON_SOLAR_FACTOR)

    annual_mwh_per_mw = cf_annual * 8760

    # Weight day slices by hours x seasonal yield factor.
    day_raw = {
        (s, p): h[(s, p)] * season_f[s]
        for (s, p) in SLICES if p in SOLAR_PERIODS
    }
    day_total = sum(day_raw.values())

    out = {}
    for (s, p) in SLICES:
        if p in SOLAR_PERIODS:
            share = day_raw[(s, p)] / day_total
            out[(s, p)] = annual_mwh_per_mw * share / 1_000_000   # TWh per MW
        else:
            out[(s, p)] = 0.0
    return out


def _hydro_season_energy_share(scenario):
    """Fraction of annual hydro energy in each season."""
    h = slice_hours()
    f = scenario.get("hydro_season_factor", DEFAULT_HYDRO_SEASON_FACTOR)
    season_hours = {s: sum(h[(ss, p)] for (ss, p) in SLICES if ss == s)
                    for s in ("dry", "wet")}
    raw = {s: season_hours[s] * f[s] for s in ("dry", "wet")}
    total = sum(raw.values())
    return {s: v / total for s, v in raw.items()}


# ============================================================
# MODEL CONSTRUCTION
# ============================================================

def build_model_sliced(
    scenario,
    econ,
    weight_cost=1.0,
    weight_emissions=0.0,
    emissions_cap=None,
    emissions_cap_by_year=None,
    reliability_max_unserved_fraction=None,
    reliability_mode="annual",
    fixed_plan=None,
    solar_capex_by_year=None,
):
    """
    Build a time-sliced planning optimisation model.

    Signature matches build_model() so runners can swap with one import change.
    """

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
    S = SLICES
    H = slice_hours()

    m.slice_hours = dict(H)   # plain dict, for diagnostics

    # --------------------------
    # Discounting
    # --------------------------
    r_social = float(scenario.get("social_discount_rate", 0.08))
    df = {t: 1.0 / ((1.0 + r_social) ** t) for t in T}
    r_private = resolve_private_rate(scenario)
    remaining_npv_priv = {
        t: sum((1.0 / ((1.0 + r_private) ** (k - t))) for k in range(t, len(T)))
        for t in T
    }
    m.remaining_npv_factor = pyo.Param(T, initialize=remaining_npv_priv,
                                       within=pyo.PositiveReals)
    m.DF = pyo.Param(T, initialize=df, within=pyo.PositiveReals)

    concessional_share = scenario.get("concessional_share", 0.0)
    concessional_envelope_npv = scenario.get("concessional_envelope_npv", None)

    # --------------------------
    # Solar CAPEX trajectory
    # --------------------------
    if solar_capex_by_year is not None:
        capex_init = {t: float(solar_capex_by_year[int(years[t])]) for t in T}
    else:
        capex_init = {t: float(econ["SOLAR_CAPEX_PER_MW"]) for t in T}
    m.solar_capex_param = pyo.Param(T, initialize=capex_init,
                                    within=pyo.PositiveReals)

    # --------------------------
    # Scalar parameters
    # --------------------------
    eta = scenario.get("gas_eta", None)
    if eta is None:
        raise ValueError("gas_eta is unset. Provide the measured fleet efficiency.")

    storage_round_trip_eff = scenario.get("storage_round_trip_eff", 0.90)
    storage_duration_hours = scenario.get("storage_duration_hours", 4.0)
    tariff = scenario.get("solar_service_tariff_usd_per_twh", None)
    financing_regime = scenario.get("financing_regime", "traditional")
    required_margin = scenario.get("required_margin", 1.0)

    if not (0.0 < storage_round_trip_eff <= 1.0):
        raise ValueError("storage_round_trip_eff must be in (0, 1]")

    gas_availability_factor = scenario.get("gas_availability_factor", None)
    if gas_availability_factor is None:
        raise ValueError(
            "gas_availability_factor is unset. Provide FUEL-AGNOSTIC gas fleet "
            "availability (mechanical + atmospheric only; exclude feedstock, "
            "which the gas fuel budget already carries)."
        )
    if not (0.0 < gas_availability_factor <= 1.0):
        raise ValueError("gas_availability_factor must be in (0, 1]")

    # ------------------------------------------------------------
    # ANNUAL decision variables (capacity and finance)
    # ------------------------------------------------------------
    m.solar_public_add = pyo.Var(T, domain=pyo.NonNegativeReals)
    m.solar_eaas_add = pyo.Var(T, domain=pyo.NonNegativeReals)
    m.eaas_subsidy = pyo.Var(T, domain=pyo.NonNegativeReals)
    m.gas_add = pyo.Param(T, initialize=0.0, within=pyo.NonNegativeReals)
    m.storage_add = pyo.Var(T, domain=pyo.NonNegativeReals)

    if fixed_plan is not None:
        for t in T:
            m.solar_public_add[t].fix(fixed_plan["solar_public_add"][t])
            m.solar_eaas_add[t].fix(fixed_plan["solar_eaas_add"][t])
            m.storage_add[t].fix(fixed_plan["storage_add"][t])

    # ------------------------------------------------------------
    # SLICED operational variables
    # ------------------------------------------------------------
    m.S = pyo.Set(initialize=S, dimen=2, ordered=True)
    m.unserved = pyo.Var(T, m.S, domain=pyo.NonNegativeReals)
    m.gas_to_power = pyo.Var(T, m.S, domain=pyo.NonNegativeReals)
    m.gas_curtail = pyo.Var(T, domain=pyo.NonNegativeReals)
    m.storage_discharge = pyo.Var(T, m.S, domain=pyo.NonNegativeReals)
    m.storage_charge = pyo.Var(T, m.S, domain=pyo.NonNegativeReals)
    m.hydro_gen = pyo.Var(T, m.S, domain=pyo.NonNegativeReals)

    # ------------------------------------------------------------
    # Capacity expressions (annual)
    # ------------------------------------------------------------
    gas_baseline = scenario["gas_baseline_mw"]
    retirement_start = scenario.get("gas_baseline_retirement_start_year", None)
    retirement_rate = scenario.get("gas_baseline_retirement_mw_per_year", 0.0)

    def gas_cap_rule(m, t):
        year_t = int(years[t])
        if retirement_start is not None and year_t >= retirement_start:
            retired = min(retirement_rate * (year_t - retirement_start + 1),
                          gas_baseline)
        else:
            retired = 0.0
        return gas_baseline - retired + sum(m.gas_add[k] for k in range(0, t + 1))

    m.gas_capacity_mw = pyo.Expression(T, rule=gas_cap_rule)

    def storage_cap_rule(m, t):
        return sum(m.storage_add[k] for k in range(0, t + 1))

    m.storage_capacity_mwh = pyo.Expression(T, rule=storage_cap_rule)
    m.storage_power_mw = pyo.Expression(
        T, rule=lambda m, t: m.storage_capacity_mwh[t] / storage_duration_hours
    )

    baseline_mw = scenario["solar_baseline_mw"]

    def solar_cap_rule(m, t):
        return baseline_mw + sum(
            m.solar_public_add[k] + m.solar_eaas_add[k] for k in range(0, t + 1)
        )

    m.solar_capacity_mw = pyo.Expression(T, rule=solar_cap_rule)

    # ------------------------------------------------------------
    # Demand, gas availability, hydro
    # ------------------------------------------------------------
    demand_annual = project_baseline_demand(
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

    hydro_baseline_twh = scenario.get("hydro_baseline_twh", None)
    if hydro_baseline_twh is None:
        raise ValueError(
            "hydro_baseline_twh is unset. Provide a verified Nigerian grid "
            "hydro figure (TWh_e/year) in the scenario."
        )
    hydro_growth = scenario.get("hydro_growth", 0.0)
    hydro_annual = {t: hydro_baseline_twh * ((1.0 + hydro_growth) ** t) for t in T}

    hydro_capacity_mw = scenario.get("hydro_capacity_mw", None)
    if hydro_capacity_mw is None:
        raise ValueError(
            "hydro_capacity_mw is unset. Required for the sliced model: hydro "
            "is dispatchable within a seasonal energy budget and needs a power "
            "limit. Nigerian installed hydro is ~2,678 MW (Kainji, Jebba, "
            "Shiroro, Zungeru, Dadin-Kowa)."
        )

    # Slice shapes
    dw = _normalised_demand_weights(scenario)
    solar_per_mw = _solar_energy_per_mw_by_slice(scenario)
    hydro_season_share = _hydro_season_energy_share(scenario)
    hydro_min_frac = scenario.get("hydro_min_flow_fraction",
                                  DEFAULT_HYDRO_MIN_FLOW_FRACTION)

    demand = {(t, s, p): demand_annual[t] * dw[(s, p)]
              for t in T for (s, p) in S}

    # ------------------------------------------------------------
    # Generation expressions (sliced)
    # ------------------------------------------------------------
    m.gas_generation = pyo.Expression(
        T, m.S, rule=lambda m, t, s, p: eta * m.gas_to_power[t, (s, p)]
    )

    m.solar_generation = pyo.Expression(
        T, m.S,
        rule=lambda m, t, s, p: m.solar_capacity_mw[t] * solar_per_mw[(s, p)]
    )

    # ------------------------------------------------------------
    # Gas: per-slice POWER limit, annual FUEL budget
    # ------------------------------------------------------------
    # Capacity limits electrical output within each slice's hours.
    m.gas_capacity_constraint = pyo.Constraint(
        T, m.S,
        rule=lambda m, t, s, p:
            m.gas_generation[t, (s, p)]
            <= m.gas_capacity_mw[t] * H[(s, p)] * gas_availability_factor / 1e6
    )

    # Fuel is an ANNUAL budget the optimiser allocates across slices. This is
    # what lets scarce gas be saved for the dry-season evening peak.
    m.gas_balance = pyo.Constraint(
        T,
        rule=lambda m, t:
            pyo.quicksum(m.gas_to_power[t, (s, p)] for (s, p) in S)
            + m.gas_curtail[t] <= gas_avail[t]
    )

    # ------------------------------------------------------------
    # Hydro: dispatchable within a seasonal energy budget
    # ------------------------------------------------------------
    # Reservoir plants (Kainji, Jebba, Shiroro) buffer inflow and are dispatched
    # preferentially to peak. Modelling hydro as flat across slices would
    # understate it badly and make peak adequacy artificially pessimistic.
    m.hydro_season_budget = pyo.Constraint(
        T, ["dry", "wet"],
        rule=lambda m, t, season:
            pyo.quicksum(m.hydro_gen[t, (s, p)] for (s, p) in S if s == season)
            == hydro_annual[t] * hydro_season_share[season]
    )

    # Power limit per slice.
    m.hydro_power_limit = pyo.Constraint(
        T, m.S,
        rule=lambda m, t, s, p:
            m.hydro_gen[t, (s, p)] <= hydro_capacity_mw * H[(s, p)] / 1e6
    )

    # Minimum flow floor: without it the LP concentrates all hydro into peak
    # slices, which reservoir operating obligations do not permit.
    season_hours = {s: sum(H[(ss, p)] for (ss, p) in S if ss == s)
                    for s in ("dry", "wet")}

    m.hydro_min_flow = pyo.Constraint(
        T, m.S,
        rule=lambda m, t, s, p:
            m.hydro_gen[t, (s, p)]
            >= hydro_min_frac * hydro_annual[t] * hydro_season_share[s]
            * (H[(s, p)] / season_hours[s])
    )

    # ------------------------------------------------------------
    # Storage: charges in any slice, discharges within the same SEASON
    # ------------------------------------------------------------
    # The seasonal energy balance is what gives storage its purpose: charge
    # from midday solar in the day slice, discharge into the evening peak of
    # the same season. Note that storage_deployable_hours is NO LONGER a free
    # parameter -- throughput now falls out of the slice structure.
    m.storage_season_balance = pyo.Constraint(
        T, ["dry", "wet"],
        rule=lambda m, t, season:
            pyo.quicksum(m.storage_discharge[t, (s, p)] for (s, p) in S if s == season)
            <= storage_round_trip_eff
            * pyo.quicksum(m.storage_charge[t, (s, p)] for (s, p) in S if s == season)
    )

    m.storage_charge_power_limit = pyo.Constraint(
        T, m.S,
        rule=lambda m, t, s, p:
            m.storage_charge[t, (s, p)]
            <= m.storage_power_mw[t] * H[(s, p)] / 1e6
    )

    m.storage_discharge_power_limit = pyo.Constraint(
        T, m.S,
        rule=lambda m, t, s, p:
            m.storage_discharge[t, (s, p)]
            <= m.storage_power_mw[t] * H[(s, p)] / 1e6
    )

    # ------------------------------------------------------------
    # ELECTRICITY BALANCE (per slice)
    # ------------------------------------------------------------
    m.balance = pyo.Constraint(
        T, m.S,
        rule=lambda m, t, s, p:
            m.gas_generation[t, (s, p)]
            + m.solar_generation[t, (s, p)]
            + m.hydro_gen[t, (s, p)]
            + m.storage_discharge[t, (s, p)]
            - m.storage_charge[t, (s, p)]
            + m.unserved[t, (s, p)]
            >= demand[(t, s, p)]
    )

    # ------------------------------------------------------------
    # Financial constraints (ANNUAL -- unchanged from the annual model)
    # ------------------------------------------------------------
    solar_energy_per_mw_annual = scenario["solar_cf"] * 8760 / 1_000_000

    if financing_regime == "eaas" and tariff is None:
        raise ValueError("Tariff must be defined under EaaS regime.")

    public_solar_capex_npv = pyo.quicksum(
        m.DF[t] * m.solar_public_add[t] * m.solar_capex_param[t] for t in T
    )
    eaas_solar_capex_npv = pyo.quicksum(
        m.DF[t] * m.solar_eaas_add[t] * m.solar_capex_param[t] for t in T
    )

    if (concessional_envelope_npv is not None
            and concessional_share > 0.0
            and financing_regime == "eaas"):
        m.concessional_drawdown_npv = pyo.Expression(
            expr=concessional_share * pyo.quicksum(
                m.DF[t] * m.solar_eaas_add[t] * m.solar_capex_param[t] for t in T
            )
        )
        m.concessional_envelope_constraint = pyo.Constraint(
            expr=m.concessional_drawdown_npv <= concessional_envelope_npv
        )

    if tariff is not None:
        def eaas_bankability_rule(m, t):
            capex = m.solar_capex_param[t] * m.solar_eaas_add[t]
            effective_private_cost = capex - m.eaas_subsidy[t]
            bankable_revenue = (
                tariff * solar_energy_per_mw_annual
                * m.remaining_npv_factor[t] * m.solar_eaas_add[t]
            ) / required_margin
            return effective_private_cost <= bankable_revenue

        m.eaas_bankability = pyo.Constraint(T, rule=eaas_bankability_rule)

    if financing_regime != "eaas":
        m.eaas_disable = pyo.Constraint(
            T, rule=lambda m, t: m.solar_eaas_add[t] <= 0.0
        )

    if tariff is not None:
        def subsidy_limit_rule(m, t):
            max_bankable_at_t = (
                tariff * solar_energy_per_mw_annual
                * m.remaining_npv_factor[t] / required_margin
            )
            gap_at_t = max(0.0, capex_init[t] - max_bankable_at_t)
            return m.eaas_subsidy[t] <= gap_at_t * m.solar_eaas_add[t]
    else:
        def subsidy_limit_rule(m, t):
            return m.eaas_subsidy[t] <= 0.0

    m.eaas_subsidy_limit = pyo.Constraint(T, rule=subsidy_limit_rule)

    # Land (annual)
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

    # Build caps (annual)
        # ------------------------------------------------------------
    # Solar deployment-capability cap (plan 2.5 step 4)
    # ------------------------------------------------------------
    # Constrains CUMULATIVE ADDITIONS to a sourced national deployment
    # trajectory (NIRP 2024) rather than a flat annual rate. Cumulative is the
    # right object: annual capability is a stock that accumulates, and a flat
    # annual rate against compounding demand binds eventually at any level.
    # See scenarios.nirp_solar_cumulative_mw() for provenance and the coherence
    # caveat on pairing this with gas regimes.
    solar_cum_cap = scenario.get("solar_cumulative_cap_mw", None)
    if solar_cum_cap is not None:
        if len(solar_cum_cap) != len(T):
            raise ValueError(
                f"solar_cumulative_cap_mw has {len(solar_cum_cap)} entries for "
                f"{len(T)} model years. Refusing to build with a misaligned cap."
            )

        def solar_cum_cap_rule(m, t):
            return (
                sum(m.solar_public_add[k] + m.solar_eaas_add[k]
                    for k in range(0, t + 1))
                <= solar_cum_cap[t]
            )

        m.solar_cumulative_cap = pyo.Constraint(T, rule=solar_cum_cap_rule)

    min_build = scenario.get("solar_min_build_mw_per_year", 0.0)
    if min_build > 0:
        m.solar_min_build_constraint = pyo.Constraint(
            T,
            rule=lambda m, t:
                m.solar_public_add[t] + m.solar_eaas_add[t] >= min_build
        )

    # ------------------------------------------------------------
    # Reliability (annual, summed over slices)
    # ------------------------------------------------------------
    if reliability_max_unserved_fraction is not None:
        if reliability_mode == "annual":
            m.reliability_constraint = pyo.Constraint(
                T,
                rule=lambda m, t:
                    pyo.quicksum(m.unserved[t, (s, p)] for (s, p) in S)
                    <= reliability_max_unserved_fraction * demand_annual[t]
            )
        elif reliability_mode == "total":
            total_demand = sum(demand_annual[t] for t in T)
            m.reliability_constraint = pyo.Constraint(
                expr=pyo.quicksum(m.unserved[t, (s, p)] for t in T for (s, p) in S)
                <= reliability_max_unserved_fraction * total_demand
            )
        else:
            raise ValueError("reliability_mode must be 'annual' or 'total'")

    # ------------------------------------------------------------
    # Costs
    # ------------------------------------------------------------
    gas_capex_npv = pyo.quicksum(
        m.DF[t] * m.gas_add[t] * scenario["gas_capex_per_mw"] for t in T
    )
    gas_opex_npv = pyo.quicksum(
        m.DF[t] * m.gas_to_power[t, (s, p)] * econ["GAS_COST_PER_TWH_TH"]
        for t in T for (s, p) in S
    )
    storage_capex_npv = pyo.quicksum(
        m.DF[t] * m.storage_add[t] * econ["STORAGE_COST_PER_MWH"] for t in T
    )
    # ------------------------------------------------------------
    # End-of-horizon salvage value (correction plan 2.5)
    # ------------------------------------------------------------
    # The horizon (21 yr) is shorter than every asset life, so without this
    # term assets with 9-29 years of remaining service are fully expensed and
    # system NPV is overstated. Straight-line residual, credited at T_last.
    #
    # Salvage is computed on RAW capex for BOTH arms: required_margin is a
    # financing premium, not asset value. It is subtracted from the objective
    # ONLY -- the public budget constraint below continues to see gross capex,
    # because the envelope constrains cash out the door.
    lives = scenario.get("asset_lifetimes", None)
    if lives is None:
        raise ValueError(
            "scenario['asset_lifetimes'] is required. build_model computes an "
            "end-of-horizon salvage credit and will not silently assume a "
            "lifetime. See scenarios.asset_lifetimes()."
        )
    for _k in ("solar", "storage", "gas"):
        if lives.get(_k) is None:
            raise ValueError(
                f"asset_lifetimes['{_k}'] is None -- [SOURCE NEEDED]. "
                "Refusing to compute salvage from a placeholder."
            )

    T_last = len(T) - 1

    def _salvage_fraction(t, life):
        """Straight-line residual fraction at end of horizon for vintage t.
        Vintage t operates years t..T_last inclusive = (T_last - t + 1) years.
        Returns 0.0 for vintages that retire within the horizon."""
        served = T_last - t + 1
        return max(0.0, float(life) - served) / float(life)

    m.salvage_npv = pyo.Expression(
        expr=m.DF[T_last] * (
            pyo.quicksum(
                (m.solar_public_add[t] + m.solar_eaas_add[t])
                * m.solar_capex_param[t]
                * _salvage_fraction(t, lives["solar"])
                for t in T
            )
            + pyo.quicksum(
                m.storage_add[t] * econ["STORAGE_COST_PER_MWH"]
                * _salvage_fraction(t, lives["storage"])
                for t in T
            )
            + pyo.quicksum(
                m.gas_add[t] * scenario["gas_capex_per_mw"]
                * _salvage_fraction(t, lives["gas"])
                for t in T
            )
        )
    )

    unserved_npv = pyo.quicksum(
        m.DF[t] * m.unserved[t, (s, p)] * econ["UNSERVED_ENERGY_PENALTY"]
        for t in T for (s, p) in S
    )

    ef_e = econ["EF_TCO2_PER_MWH_TH"] / eta
    m.emissions_by_year = pyo.Expression(
        T,
        rule=lambda m, t: pyo.quicksum(
            m.gas_generation[t, (s, p)] for (s, p) in S
        ) * 1e6 * ef_e
    )
    m.emissions = pyo.Expression(
        expr=pyo.quicksum(m.emissions_by_year[t] for t in T)
    )

    carbon_cost_npv = 0.0
    if scenario.get("carbon_active", False):
        carbon_price = float(scenario["carbon_price"])
        carbon_cost_npv = pyo.quicksum(
            m.DF[t] * m.emissions_by_year[t] * carbon_price for t in T
        )

    system_cost_npv = (
        gas_opex_npv
        + public_solar_capex_npv
        + eaas_solar_capex_npv
        + gas_capex_npv
        + storage_capex_npv
        + unserved_npv
        + carbon_cost_npv
        - m.salvage_npv
    )
    m.system_cost_npv = pyo.Expression(expr=system_cost_npv)

    m.cost_gas_opex_npv = pyo.Expression(expr=gas_opex_npv)
    m.cost_gas_capex_npv = pyo.Expression(expr=gas_capex_npv)
    m.cost_public_solar_capex_npv = pyo.Expression(expr=public_solar_capex_npv)
    m.cost_eaas_solar_capex_npv = pyo.Expression(expr=eaas_solar_capex_npv)
    m.cost_storage_capex_npv = pyo.Expression(expr=storage_capex_npv)
    m.cost_unserved_voll_npv = pyo.Expression(expr=unserved_npv)
    m.cost_carbon_npv = pyo.Expression(expr=carbon_cost_npv)
    m.cost_salvage_npv = pyo.Expression(expr=m.salvage_npv)

    # Public budget (annual)
    public_budget_npv = scenario.get("public_solar_budget_npv", None)
    if public_budget_npv is not None:
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
        m.has_public_budget_constraint = True

    # ------------------------------------------------------------
    # Objective and emissions constraints
    # ------------------------------------------------------------
    if emissions_cap_by_year is not None:
        if len(emissions_cap_by_year) != len(list(T)):
            raise ValueError("emissions_cap_by_year length mismatch.")
        m.emissions_constraint = pyo.Constraint(
            T, rule=lambda m, t: m.emissions_by_year[t] <= emissions_cap_by_year[t]
        )
        m.objective = pyo.Objective(expr=system_cost_npv, sense=pyo.minimize)

    elif emissions_cap is not None:
        m.emissions_constraint = pyo.Constraint(expr=m.emissions <= emissions_cap)
        m.objective = pyo.Objective(expr=system_cost_npv, sense=pyo.minimize)

    else:
        m.objective = pyo.Objective(
            expr=weight_cost * system_cost_npv + weight_emissions * m.emissions,
            sense=pyo.minimize,
        )

    return m


# ============================================================
# AGGREGATION HELPERS (for annual-vs-sliced comparison)
# ============================================================

def annual_totals(m, years):
    """Aggregate sliced results to annual totals for comparison and reporting."""
    T = range(len(years))
    out = {}
    for name, var in (
        ("gas_generation", m.gas_generation),
        ("solar_generation", m.solar_generation),
        ("hydro_gen", m.hydro_gen),
        ("storage_discharge", m.storage_discharge),
        ("storage_charge", m.storage_charge),
        ("unserved", m.unserved),
    ):
        out[name] = {
            int(years[t]): sum(float(pyo.value(var[t, (s, p)])) for (s, p) in SLICES)
            for t in T
        }
    out["gas_to_power"] = {
        int(years[t]): sum(float(pyo.value(m.gas_to_power[t, (s, p)]))
                           for (s, p) in SLICES)
        for t in T
    }
    return out


def slice_detail(m, years, t_index):
    """Per-slice breakdown for a single year — used to inspect peak adequacy."""
    rows = []
    for (s, p) in SLICES:
        rows.append({
            "slice": f"{s}_{p}",
            "hours": slice_hours()[(s, p)],
            "gas": float(pyo.value(m.gas_generation[t_index, (s, p)])),
            "solar": float(pyo.value(m.solar_generation[t_index, (s, p)])),
            "hydro": float(pyo.value(m.hydro_gen[t_index, (s, p)])),
            "storage_out": float(pyo.value(m.storage_discharge[t_index, (s, p)])),
            "storage_in": float(pyo.value(m.storage_charge[t_index, (s, p)])),
            "unserved": float(pyo.value(m.unserved[t_index, (s, p)])),
        })
    return rows
