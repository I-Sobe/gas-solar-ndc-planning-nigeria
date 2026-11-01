"""
Optimization model definition and solution utilities.

Scope:
- Planning-level Pyomo model construction
- ε-constraint formulation (emissions_cap / emissions_cap_by_year)
- Solver invocation with termination checks

The weighted-sum formulation (weight_cost / weight_emissions) was removed
(Plan 0.4, item B, docs/numerical_hygiene_0_4.md): it was dead code, never
set by any caller. The objective is unconditionally system_cost_npv,
minimised.

Non-scope:
- Scenario scripting
- Deterministic evaluation
- Pareto orchestration
"""

import pyomo.environ as pyo
from src.demand import project_baseline_demand
from src.gas_supply import gas_available_power
from src.blended_finance import resolve_remaining_npv_factor
from src.io import compute_om_cost_npv

# ============================================================
# MODEL CONSTRUCTION
# ============================================================

def build_model(
    scenario,
    econ,
    emissions_cap=None,
    emissions_cap_by_year=None,
    max_offgrid_fraction=None,
    grid_service_mode="annual",
    reliability_max_unserved_fraction=None,  # DEPRECATED alias for max_offgrid_fraction
    reliability_mode=None,                    # DEPRECATED alias for grid_service_mode
    fixed_plan = None,
    solar_capex_by_year=None,
    selfgen_in_emissions_cap=False,
    allow_unserved=None,
):

    # ------------------------------------------------------------------
    # Grid service obligation (renamed from "reliability" in 2.6 F0).
    #
    # This constraint bounds (selfgen + unserved) -- the GRID SERVICE
    # SHORTFALL -- not unserved alone. A household running its own genset was
    # not served by the grid, but no capacity was short, so this is a service
    # obligation, NOT an adequacy/reliability standard: LOLE and EENS do not
    # describe it. The old names are retained as deprecated aliases so the
    # pre-2.6 REL and VoLL runners keep working unchanged.
    # ------------------------------------------------------------------
    if reliability_max_unserved_fraction is not None:
        if max_offgrid_fraction is not None:
            raise ValueError(
                "Pass either max_offgrid_fraction or its deprecated alias "
                "reliability_max_unserved_fraction, not both."
            )
        max_offgrid_fraction = reliability_max_unserved_fraction
    if reliability_mode is not None:
        grid_service_mode = reliability_mode

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
    
        selfgen_in_emissions_cap : bool, default False
        Whether private self-generation CO2 counts against the NDC cap.

        FALSE is the central case. Nigeria's NDC commits the gas-power slice;
        the cap here was derived as 7.24% of the energy-sector abatement
        target. Private diesel sits in the national inventory but outside that
        slice, so charging it against this budget is an accounting error, and
        it charges diesel at a shadow price that is an artefact of how tightly
        the slice was drawn.

        WHAT DISCIPLINES DIESEL INSTEAD: the grid service obligation
        (max_offgrid_fraction). It is quantity-based, corresponds to a real
        policy instrument, and requires no carbon price -- which matters,
        because Nigeria has none (carbon_case='no_policy' -> price 0.0), so a
        price-based discipline would be assuming the conclusion.

        THE RESIDUAL RISK: with the cap on grid-only, a slack obligation lets
        the model meet the NDC by pushing load onto diesel. Call
        leakage_diagnostic() after every capped solve; a flagged run is not an
        NDC-compliant plan and must not be reported as one.

        TRUE widens the cap to total emissions. That is the alternative-
        boundary arm, run to quantify the difference -- not the central case.
    """

    # Plan 0.4, item B: the "defensive objective check" that stood here
    # (raising when no cap was provided AND both weight_cost/weight_emissions
    # were zero) is DELETED along with the weighted-sum branch it guarded --
    # see the objective construction below. No caller anywhere in src/ or
    # scripts/ ever set either weight (grep confirmed nothing), so the
    # weighted-sum term was dead code and this guard existed only to catch
    # its one degenerate combination.

    m = pyo.ConcreteModel()
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    years = scenario["years"]
    T = range(len(years))

    # ============================================================
    # Firm capacity / genset guard (2.6, F0 Step 2 -- phase_2_4_inversion)
    # ============================================================
    # AT THE TOP, before any other scenario/econ processing: 22 of 33
    # runner scripts call load_scenario() (directly or via
    # src/optimize_experiments.py), which already guards peak_multiple_case
    # and reserve_margin as required, no-default arguments -- see
    # scenarios.py. That guard is the EARLY, FRIENDLIER failure and is kept
    # exactly as-is. But build_model() accepts ANY scenario/econ dict a
    # caller constructs directly (tests/test_run.py hand-builds econ this
    # way today, bypassing load_econ() entirely), so nothing upstream can be
    # relied on. This block is the guard that actually closes the bypass,
    # placed first so it fires before any other processing gives a
    # confusing, unrelated error instead.
    if scenario.get("peak_multiple_case", None) is None:
        raise ValueError(
            "scenario['peak_multiple_case'] is missing or None (2.6, F0 "
            "Step 2.4). Required so the firm-capacity constraint's PAR has "
            "a traceable, named justification -- see "
            "scenarios.peak_multiple_scenarios()."
        )
    peak_mw = scenario.get("peak_mw", None)
    if peak_mw is None:
        raise ValueError(
            "scenario['peak_mw'] is required (2.6, F0 Step 2.3). Derived by "
            "load_scenario() from peak_multiple_case (Step 2.4); build_model() "
            "will not silently assume a peak-demand trajectory."
        )
    if len(peak_mw) != len(T):
        raise ValueError(
            f"scenario['peak_mw'] has {len(peak_mw)} entries for {len(T)} "
            "model years. Refusing to build with a misaligned peak trajectory."
        )
    reserve_margin = scenario.get("reserve_margin", None)
    if reserve_margin is None:
        raise ValueError(
            "scenario['reserve_margin'] is required (2.6, F0 Step 2.3). No "
            "default: an omitted reserve margin would silently build an "
            "unenforced firm-capacity constraint."
        )
    capacity_credit = scenario.get("capacity_credit", None)
    if capacity_credit is None:
        raise ValueError(
            "scenario['capacity_credit'] is required (2.6, F0 Step 2.3). "
            "Provide {'solar':..., 'gas':..., 'storage':..., 'hydro':...} "
            "(see scenarios.py)."
        )
    for _tech in ("solar", "gas", "storage", "hydro"):
        if _tech not in capacity_credit:
            raise ValueError(
                f"scenario['capacity_credit'] is missing '{_tech}'."
            )
    # storage_cc_case (2.6, phase_2_3b Step 2): closes the same
    # hand-built-scenario bypass as the peak_multiple_case guard above, for
    # a caller that sets capacity_credit['storage'] directly without going
    # through load_scenario()'s storage_cc_case validation.
    if scenario.get("storage_cc_case", None) is None:
        raise ValueError(
            "scenario['storage_cc_case'] is missing or None (2.6, "
            "phase_2_3b Step 2). Required so storage's firm-capacity "
            "credit has a traceable, sourced justification -- see "
            "scenarios.storage_cc_scenarios()."
        )
    # gas_build_case (2.9, Step 2/4): closes the same hand-built-scenario
    # bypass as the storage_cc_case guard above, for a caller that leaves
    # new gas capacity's build policy unstated. The deep validation (arm
    # membership, trajectory/capex/efficiency sourcing) lives in
    # load_scenario() -- this is presence-only, matching the existing
    # pattern.
    if scenario.get("gas_build_case", None) is None:
        raise ValueError(
            "scenario['gas_build_case'] is missing or None (2.9, Step 2). "
            "Required so new gas capacity has a traceable, named choice-set "
            "-- see scenarios.GAS_BUILD_CASE_CHOICES."
        )
    # gas_availability_case (2.10, Step 2/4): closes the same hand-built-
    # scenario bypass as the gas_build_case guard above, for a caller that
    # leaves the gas fleet's availability arm unstated. The deep validation
    # (arm membership, SOURCE: [PENDING] raise for gas_avail_central) lives
    # in load_scenario() -- this is presence-only, matching the existing
    # pattern.
    if scenario.get("gas_availability_case", None) is None:
        raise ValueError(
            "scenario['gas_availability_case'] is missing or None (2.10, "
            "Step 2). Required so the gas fleet's availability figure has "
            "a traceable, named choice-set -- see "
            "scenarios.gas_availability_scenarios()."
        )
    # gas_retirement_case (2.11, Step 2/3): closes the same hand-built-
    # scenario bypass as the gas_availability_case guard above, for a
    # caller that leaves the gas fleet's retirement schedule unstated. The
    # deep validation (arm membership; the removed-name raise for
    # 'gas_retire_sourced', 2.12) lives in load_scenario() -- this is
    # presence-only, matching the existing pattern.
    if scenario.get("gas_retirement_case", None) is None:
        raise ValueError(
            "scenario['gas_retirement_case'] is missing or None (2.11, "
            "Step 2). Required so the gas fleet's retirement schedule has "
            "a traceable, named choice-set -- see "
            "scenarios.gas_retirement_scenarios()."
        )
    # gas_capex_case (2.13, Step 1/3): closes the same hand-built-scenario
    # bypass as the gas_retirement_case guard above, for a caller that
    # leaves new-build gas's unit cost unstated. The deep validation (arm
    # membership) lives in load_scenario() -- this is presence-only,
    # matching the existing pattern.
    if scenario.get("gas_capex_case", None) is None:
        raise ValueError(
            "scenario['gas_capex_case'] is missing or None (2.13, Step "
            "1). Required so new-build gas capacity's unit cost has a "
            "traceable, named choice-set -- see "
            "scenarios.gas_capex_scenarios()."
        )
    # ccgt_efficiency_case (2.13, Step 2/3): same pattern, for a caller
    # that leaves new-build CCGT efficiency unstated. gas_new_efficiency
    # itself is read directly (not via .get(..., None) + a raise) at its
    # one functional use below (new_gas_deliverability='exempt') -- this
    # presence guard on the ARM is what now guarantees it is set, since
    # gas_new_efficiency is no longer a free, independently-settable
    # kwarg (2.13, Step 2).
    if scenario.get("ccgt_efficiency_case", None) is None:
        raise ValueError(
            "scenario['ccgt_efficiency_case'] is missing or None (2.13, "
            "Step 2). Required so new-build CCGT efficiency has a "
            "traceable, named choice-set -- see "
            "scenarios.ccgt_efficiency_scenarios()."
        )
    # TASK 77 C.2: vintage-differentiated gas dispatch (register L123), mirroring build_model_sliced() with the slice
    # index removed. Validated at new_gas_deliverability='constrained' only, as in the sliced builder.
    gas_vintage_dispatch = bool(scenario.get("gas_vintage_dispatch", False))
    if gas_vintage_dispatch:
        if scenario.get("new_gas_deliverability", "constrained") != "constrained":
            raise ValueError("gas_vintage_dispatch is validated at new_gas_deliverability='constrained' only.")
        gas_new_eff_vintage = float(scenario["gas_new_efficiency"])
    # capital_recovery_case (F1, Step 2/7): same pattern, for a caller
    # that leaves the per-tranche rate/tenor pair unstated.
    if scenario.get("capital_recovery_case", None) is None:
        raise ValueError(
            "scenario['capital_recovery_case'] is missing or None (F1, "
            "Step 2). Required so the CRF() capital charge has a "
            "traceable, named recovery period -- see "
            "scenarios.capital_recovery_scenarios()."
        )
    # dscr_case (F1, Step 3/7): same pattern. required_margin itself is
    # read directly (scenario["required_margin"], no .get(..., 1.0)
    # fallback) below -- this presence guard on the ARM is what now
    # guarantees it is set, replacing the removed 1.0 builder fallback.
    if scenario.get("dscr_case", None) is None:
        raise ValueError(
            "scenario['dscr_case'] is missing or None (F1, Step 3). "
            "Required so the EaaS bankability coverage cushion has a "
            "traceable, named choice-set -- see scenarios.dscr_scenarios()."
        )
    # us_inflation (F1, Step 1/7): checked by key PRESENCE, not truthiness
    # -- 0.0 is a legitimate (if unusual) inflation assumption and must
    # not be treated as "missing".
    if "us_inflation" not in scenario or scenario["us_inflation"] is None:
        raise ValueError(
            "scenario['us_inflation'] is missing or None (F1, Step 1). "
            "Required to deflate capital_recovery_case's nominal rates at "
            "the CRF() call site."
        )
    # capital_treatment_case (F1, Step 6/7): closes the same hand-built-
    # scenario bypass, for a caller that leaves the objective's F1-vs-
    # pre-F1 form unstated -- NOT one of the task's three named required
    # keys, added because Step 6 has no other way to be a traceable,
    # named choice. See scenarios.CAPITAL_TREATMENT_CASE_CHOICES.
    if scenario.get("capital_treatment_case", None) is None:
        raise ValueError(
            "scenario['capital_treatment_case'] is missing or None (F1, "
            "Step 6). Pass 'pre_f1_overnight' or 'f1_tranche' -- see "
            "scenarios.CAPITAL_TREATMENT_CASE_CHOICES."
        )
    if scenario["capital_treatment_case"] not in ("pre_f1_overnight", "f1_tranche"):
        raise ValueError(
            f"Unknown capital_treatment_case: {scenario['capital_treatment_case']}"
        )
    # Phase 2.8: OCGT-to-CCGT conversion is SLICE-RESOLVED (Decision 3 --
    # m.convert_to_power[tier, t, (s,p)]) and has no annual form: an
    # annual converted-energy block would add capacity the slice-resolved
    # dispatch and peak adequacy cannot see. It is implemented in
    # build_model_sliced() ONLY. conv_none leaves conversion_available
    # False and this builder is unaffected (regression arm).
    if scenario.get("conversion_available", False):
        raise ValueError(
            "scenario['conversion_available'] is True but build_model() "
            "(annual) does not implement OCGT-to-CCGT conversion -- it is "
            "slice-resolved (Phase 2.8, Decision 3) and lives in "
            "build_model_sliced() only. Pass conversion_case='conv_none', "
            "or build with build_model_sliced()."
        )
    # hydro_capacity_mw was previously read only by the sliced builder
    # (optimize_model_sliced.py), which dispatches hydro against a power
    # limit. The annual builder had no installed-MW figure for hydro because
    # it only needed the TWh energy series -- the firm capacity constraint
    # below is the first thing in this file that needs hydro's MW rating.
    #
    # Correction plan 1.5: hydro_capacity_mw is now a PER-YEAR series
    # (hydro_capacity_mw_by_year), set by load_scenario() from the REQUIRED
    # hydro_case arm and moving with the hydro energy series. Under
    # hydro_flat it is 2,678 in every year (byte-identical to the pre-1.5
    # scalar).
    hydro_capacity_mw_by_year = scenario.get("hydro_capacity_mw_by_year", None)
    if hydro_capacity_mw_by_year is None:
        raise ValueError(
            "scenario['hydro_capacity_mw_by_year'] is required for the firm "
            "capacity constraint (2.6, F0 Step 2.3; correction plan 1.5). It "
            "is set by load_scenario() from the REQUIRED hydro_case arm -- "
            "see scenarios.hydro_trajectory(). A hand-built scenario must "
            "provide the per-year installed-capacity series."
        )
    # genset_case: checked on ECON, not scenario. load_econ() (src/io.py)
    # ALWAYS sets econ["GENSET_CASE"] now (to a case-name string, or to None
    # for a deliberate, considered opt-out -- the commit-A regression test).
    # Checking KEY PRESENCE (not truthiness) is deliberate: None is a
    # legitimate value here, unlike peak_mw/reserve_margin/capacity_credit
    # above. Only the key's ABSENCE means econ was hand-built, bypassing
    # load_econ() entirely -- exactly what tests/test_run.py does today.
    if "GENSET_CASE" not in econ:
        raise ValueError(
            "econ['GENSET_CASE'] is missing (2.6, F0 Step 2). This means "
            "econ was not built via load_econ() (src/io.py) -- genset_case "
            "was never decided, not even explicitly declined. Build econ "
            "via load_econ(genset_case=...) (pass None explicitly for the "
            "commit-A regression test) rather than constructing it by hand."
        )

    # --------------------------
    # Discounting
    # --------------------------
    r_social = float(scenario.get("social_discount_rate", 0.08))
    # t=0 at start year
    df = {t: 1.0 / ((1.0 + r_social) ** t) for t in T}
    # bank_tenor_case (rate-coherence task, Part 1): resolves to the
    # commercial-basis factor by default, the pre-correction blended-WACC
    # factor under 'bank_tenor_legacy' -- see resolve_remaining_npv_factor()
    # docstring and docs/task_log_rate_coherence.md. Called here, before any
    # capital_treatment_case branch, so it is built identically under both
    # treatments.
    remaining_npv_priv = resolve_remaining_npv_factor(scenario, T)
    m.remaining_npv_factor = pyo.Param(T, initialize=remaining_npv_priv, within=pyo.PositiveReals)
    
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

    # Storage CAPEX trajectory (TASK 52 section 2.1 / TASK 53 2.2): mirrors
    # build_model_sliced exactly. scenario["storage_capex_case"] has NO fallback.
    _sc_case = scenario["storage_capex_case"]
    if _sc_case == "flat":
        _sc_init = {t: float(econ["STORAGE_COST_PER_MWH"]) for t in T}
    elif _sc_case == "declining_nrel":
        _sc_series = scenario["storage_capex_by_year"]
        _sc_init = {t: float(_sc_series[int(years[t])]) for t in T}
    else:
        raise ValueError(f"Unknown storage_capex_case: {_sc_case!r}")
    m.storage_capex_param = pyo.Param(T, initialize=_sc_init, within=pyo.PositiveReals)

    # ------------------------------------------------------------
    # Gas parameters
    # ------------------------------------------------------------
    eta = scenario.get("gas_eta", None)
    if eta is None:
        raise ValueError("gas_eta is unset. Provide the measured fleet efficiency")

    # ------------------------------------------------------------
    # Storage parameters
    # ------------------------------------------------------------
    storage_deployable_hours = scenario.get("storage_deployable_hours_per_year", 700.0)
    #storage_solar_surplus_frac = scenario.get("storage_solar_surplus_frac", 0.20)
    storage_round_trip_eff = scenario.get("storage_round_trip_eff", 0.90)
    storage_duration_hours = scenario["storage_duration_hours"]  # kept for storage_power_mw only
    
    tariff = scenario.get("solar_service_tariff_usd_per_twh", None)
    financing_regime = scenario.get("financing_regime", "traditional")
    # F1, Step 3: the old scenario.get("required_margin", 1.0) fallback is
    # REMOVED -- dscr_case (validated above by presence guard, resolved in
    # load_scenario()) now always supplies this. Direct key access so a
    # hand-built scenario that skips dscr_case fails loudly here, not
    # silently at 1.0.
    required_margin = scenario["required_margin"]
    # F1, Step 6: presence/membership already guarded above; direct read.
    capital_treatment_case = scenario["capital_treatment_case"]

    voll = econ["UNSERVED_ENERGY_PENALTY"]

    # ------------------------------------------------------------
    # Private self-generation backstop (correction plan 2.6)
    # ------------------------------------------------------------
    # Fallback: price selfgen at VoLL with zero EF. That makes it an exact
    # duplicate of unserved and leaves the objective unchanged -- the commit-A
    # regression test. Pass genset_case to load_econ() to activate real prices.
    selfgen_price = econ.get("GENSET_COST_PER_TWH_E", voll)
    if "GENSET_ETA_HHV" in econ:
        # EF_e DERIVED, never stored (same rule as the gas fleet, 1.2).
        selfgen_ef_e = econ["GENSET_EF_TH_HHV"] / econ["GENSET_ETA_HHV"]
    else:
        selfgen_ef_e = 0.0

    # Busbar-equivalence factor for self-generation.
    #
    # The balance is metered at busbar. A genset delivers at the point of use,
    # downstream of the transmission loss factor, so 1 TWh of genset output
    # displaces MORE than 1 TWh of busbar generation. Setting this to
    # 1/(1 - TLF) credits that; leaving it at 1.0 does not.
    #
    # DEFAULT IS 1.0 -- a declared conservative simplification, not an
    # oversight. It understates the genset by roughly the TLF (~7%). Set
    # scenario["selfgen_busbar_factor"] = 1/(1-0.07) to switch it on. Whichever
    # you choose, state it in the methods chapter.
    # TASK 69 B.2 (register L96): read directly; load_scenario() sets it (default 1.0).
    selfgen_busbar_factor = float(scenario["selfgen_busbar_factor"])
    if selfgen_busbar_factor < 1.0:
        raise ValueError(
            "selfgen_busbar_factor < 1.0 would penalise a genset for avoiding "
            "transmission losses, which is backwards."
        )
        # ------------------------------------------------------------------
    # Is load-shedding an available ACTION? (correction plan 2.6, D11)
    #
    # Default None = auto: unserved is DISALLOWED whenever a genset backstop
    # is priced, ALLOWED otherwise (pre-2.6 behaviour preserved).
    #
    # Why: with genset CO2 inside the cap, m.unserved became a zero-emission
    # abatement technology. The model paid VoLL ($10,000/MWh) to shed load
    # rather than $277/MWh to self-generate, purely to buy cap headroom --
    # 344.8 TWh of it. That is physically false. In Nigeria unmet grid demand
    # does not vanish; it becomes self-generation (86% of MSMEs own a genset;
    # 14-40 GW installed). Shedding is a CONSEQUENCE, not a planner's choice.
    #
    # With unserved disallowed there is exactly one shortfall channel, VoLL
    # leaves the objective entirely, and an unmeetable cap returns INFEASIBLE
    # rather than a trillion-dollar penalty. Infeasible is the honest answer
    # and the stronger finding.
    #
    # Pass allow_unserved=True explicitly to restore the shed option -- needed
    # by scripts/probe_2_6_structure.py, whose unserved anchor probe requires
    # a free variable to perturb.
    # ------------------------------------------------------------------
    
    if allow_unserved is None:
        allow_unserved = "GENSET_COST_PER_TWH_E" not in econ
    if storage_deployable_hours <= 0 or storage_deployable_hours > 8760:
        raise ValueError("storage_deployable_hours_per_year must be in (0, 8760]")
    if not (0.0 < storage_round_trip_eff <= 1.0):
        raise ValueError("storage_round_trip_eff must be in (0, 1]")

    m.solar_public_add = pyo.Var(T, domain=pyo.NonNegativeReals)
    m.solar_eaas_add = pyo.Var(T, domain=pyo.NonNegativeReals)
    m.eaas_subsidy = pyo.Var(T, domain=pyo.NonNegativeReals)
    # New gas plant construction (2.9, Step 1): gas_add is now a free
    # decision variable, gated by gas_build_case (2.9, Step 2) -- see below,
    # after gas_capacity_mw is defined, for how each arm constrains it.
    # Previously a pyo.Param fixed at 0.0 (pre-2.9: "New gas capacity is
    # structurally non-optimal across all scenarios (verified)" -- true
    # only because gas_add could not move at all).
    m.gas_add = pyo.Var(T, domain=pyo.NonNegativeReals)
    gas_build_case = scenario["gas_build_case"]
    # Storage:
    m.storage_add = pyo.Var(T, domain=pyo.NonNegativeReals)
    if fixed_plan is not None:
        # eaas_subsidy is excluded 
        for t in T:
            m.solar_public_add[t].fix(fixed_plan["solar_public_add"][t])
            m.solar_eaas_add[t].fix(fixed_plan["solar_eaas_add"][t])
            m.storage_add[t].fix(fixed_plan["storage_add"][t]) 
    
    gas_baseline = scenario["gas_baseline_mw"]

    retirement_start = scenario.get("gas_baseline_retirement_start_year", None)
    retirement_rate = scenario.get("gas_baseline_retirement_mw_per_year", 0.0)
    # gas_baseline_retirement_series_mw (2.12, Step 1): an optional
    # per-model-year CUMULATIVE retired-MW list (e.g. from
    # scenarios.gas_vintage_retirement_series_mw()), for retirement
    # schedules too lumpy for the flat rate * (year - start + 1) formula
    # below to represent (see docs/phase_2_12_vintage_retirement.md, Step
    # 0). None for gas_retire_none/gas_retire_proxy, which are unchanged
    # and fall through to the flat formula exactly as in 2.11.
    retirement_series = scenario.get("gas_baseline_retirement_series_mw", None)
    if retirement_series is not None and len(retirement_series) != len(T):
        raise ValueError(
            f"scenario['gas_baseline_retirement_series_mw'] has "
            f"{len(retirement_series)} entries for {len(T)} model years. "
            "Refusing to build with a misaligned retirement series."
        )

    def gas_cap_rule(m, t):
        year_t = int(years[t])
        # Cumulative retirement of baseline fleet
        if retirement_series is not None:
            retired = min(retirement_series[t], gas_baseline)  # cannot retire more than exists
        elif retirement_start is not None and year_t >= retirement_start:
            retired = retirement_rate * (year_t - retirement_start + 1)
            retired = min(retired, gas_baseline)  # cannot retire more than exists
        else:
            retired = 0.0
        return gas_baseline - retired + sum(m.gas_add[k] for k in range(0, t + 1))

    m.gas_capacity_mw = pyo.Expression(T, rule=gas_cap_rule)

    # ------------------------------------------------------------
    # gas_build_case (2.9, Step 2): wires each arm's treatment of gas_add
    # onto the now-free Var. The cap, where it exists, is on CUMULATIVE gas
    # capacity (m.gas_capacity_mw), not on annual additions (m.gas_add),
    # per the task's explicit instruction.
    # ------------------------------------------------------------
    if gas_build_case == "gas_none":
        for t in T:
            m.gas_add[t].fix(0.0)
    elif gas_build_case == "gas_ndc_17gw":
        gas_new_capacity_cap_mw = scenario.get("gas_new_capacity_cap_mw", None)
        if gas_new_capacity_cap_mw is None:
            raise ValueError(
                "gas_build_case='gas_ndc_17gw' selected, but "
                "scenario['gas_new_capacity_cap_mw'] is unset (2.9, Step 2 "
                "-- SOURCE: [PENDING], see "
                "scenarios.gas_ndc_17gw_trajectory_mw()). A hand-built "
                "scenario cannot bypass this by setting gas_build_case "
                "directly without the trajectory."
            )
        m.gas_new_capacity_cap_constraint = pyo.Constraint(
            T,
            rule=lambda m, t: m.gas_capacity_mw[t] <= gas_new_capacity_cap_mw[t]
        )
    elif gas_build_case == "gas_unconstrained":
        pass  # no cap; the LP builds whatever is cost-optimal

    # Storage always starts from zero -- no brownfield BESS baseline in Nigeria
    # (2025). storage_baseline_mwh is 0.0 in all scenarios and is intentionally
    # excluded from the cumulative-capacity rule below.
    def storage_cap_rule(m, t):
        return sum(m.storage_add[k] for k in range(0, t+1))

    m.storage_capacity_mwh = pyo.Expression(T, rule=storage_cap_rule)
    m.unserved = pyo.Var(T, domain=pyo.NonNegativeReals)
    # Private self-generation backstop (correction plan 2.6). Deliberately
    # UNBOUNDED: Nigeria's installed self-generation stock is roughly 8x peak
    # realised grid capacity (ZE-Gen 2026), so no capacity limit binds within
    # this horizon. m.unserved is retained above it as the true scarcity tail
    # and is expected to solve to zero once the backstop is priced.
    m.selfgen = pyo.Var(T, domain=pyo.NonNegativeReals)
    if not allow_unserved:
        # Retained in the model (balance, obligation and diagnostics all
        # reference it) but pinned to zero, so presolve removes the columns.
        for t in T:
            m.unserved[t].fix(0.0)

    m.gas_to_power = pyo.Var(T, domain=pyo.NonNegativeReals)
    if gas_vintage_dispatch:
        # part of the TOTAL thermal draw burned by capacity ADDED in the model (see build_model_sliced())
        m.gas_to_power_new = pyo.Var(T, domain=pyo.NonNegativeReals)
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
    # Hydro generation (exogenous fixed series, TWh_e/year)
    # ------------------------------------------------------------
    
    # Correction plan 1.5: the hydro energy series is a REQUIRED per-year
    # trajectory (hydro_energy_twh_by_year), set by load_scenario() from
    # the hydro_case arm. Under hydro_flat every entry is 11.47 (identical
    # to the pre-1.5 hydro_baseline_twh * (1+0)^t formula).
    hydro_energy_twh_by_year = scenario.get("hydro_energy_twh_by_year", None)
    if hydro_energy_twh_by_year is None:
        raise ValueError(
            "hydro_energy_twh_by_year is unset (correction plan 1.5). It is "
            "set by load_scenario() from the REQUIRED hydro_case arm -- see "
            "scenarios.hydro_trajectory(). A hand-built scenario must provide "
            "the annual hydro energy series (TWh_e/year); a numeric default "
            "is deliberately not supplied to avoid a mis-scaled hydro slab "
            "silently corrupting gas/emissions/NDC results."
        )
    if len(hydro_energy_twh_by_year) != len(list(T)):
        raise ValueError(
            f"hydro_energy_twh_by_year has {len(hydro_energy_twh_by_year)} "
            f"entries for {len(list(T))} model years."
        )
    hydro = {t: float(hydro_energy_twh_by_year[t]) for t in T}

    # ------------------------------------------------------------
    # Generation Expressions
    # ------------------------------------------------------------
    if gas_vintage_dispatch:
        # total thermal draw x blended eta, plus the uplift on the added-capacity stream
        m.gas_generation = pyo.Expression(
            T, rule=lambda m, t: eta * m.gas_to_power[t] + (gas_new_eff_vintage - eta) * m.gas_to_power_new[t]
        )
    else:
        m.gas_generation = pyo.Expression(
            T, rule=lambda m, t: eta * m.gas_to_power[t]
        )
    # Gas capacity constraint: annual electrical energy from gas cannot exceed
    # installed gas capacity operating at its availability factor.
    gas_availability_factor = scenario.get("gas_availability_factor", None)
    if gas_availability_factor is None:
        raise ValueError(
            "gas_availability_factor is unset. Provide FUEL-AGNOSTIC gas fleet "
            "availability (mechanical + atmospheric only; exclude feedstock, "
            "which gas_balance already carries). Do NOT use NERC Plant Load "
            "Factor or the headline Availability Factor. See scenarios.py."
        )
    if not (0.0 < gas_availability_factor <= 1.0):
        raise ValueError("gas_availability_factor must be in (0, 1]")
    m.gas_capacity_constraint = pyo.Constraint(
        T,
        rule=lambda m, t:
            m.gas_generation[t]
            <= m.gas_capacity_mw[t] * 8760 * gas_availability_factor / 1e6
    )
    if gas_vintage_dispatch:
        # Same three vintage rows as build_model_sliced() (TASK 66, tightened TASK 74): (i) the added stream within
        # the total draw; (ii) the added stream within the capacity added in the model; (iii) the LEGACY stream
        # within surviving legacy capacity (m.gas_capacity_mw less cumulative additions). No conversion here
        # (the annual builder cannot represent it), so the legacy row needs no overlap caveat.
        m.gas_vintage_new_capacity_mw = pyo.Expression(
            T, rule=lambda m, t: sum(m.gas_add[k] for k in range(0, t + 1))
        )
        m.gas_new_stream_within_total = pyo.Constraint(
            T, rule=lambda m, t: m.gas_to_power_new[t] <= m.gas_to_power[t]
        )
        m.gas_new_stream_capacity = pyo.Constraint(
            T,
            rule=lambda m, t:
                gas_new_eff_vintage * m.gas_to_power_new[t]
                <= m.gas_vintage_new_capacity_mw[t] * 8760 * gas_availability_factor / 1e6
        )
        m.gas_legacy_stream_capacity = pyo.Constraint(
            T,
            rule=lambda m, t:
                eta * (m.gas_to_power[t] - m.gas_to_power_new[t])
                <= (m.gas_capacity_mw[t] - m.gas_vintage_new_capacity_mw[t]) * 8760 * gas_availability_factor / 1e6
        )

    # ------------------------------------------------------------
    # new_gas_deliverability (2.9, Step 3): 'constrained' (default) leaves
    # gas_balance exactly as before -- the whole fleet, including new build,
    # draws on the SAME shared fuel budget. 'exempt' assumes new capacity
    # is sited at gas supply or on dedicated commercial contracts: it adds
    # a fuel-equivalent allowance, sized to new capacity's OWN
    # capacity-and-efficiency ceiling, to the shared budget -- so new
    # capacity is never fuel-constrained by the existing fleet's scarcity,
    # without introducing a second, vintage-tagged dispatch variable.
    # gas_generation, emissions, and dispatch OPEX all remain on the single
    # shared m.gas_to_power and the blended fleet eta (2.9 does not split
    # dispatch by vintage -- see docs/phase_2_9_gas_addition.md).
    # ------------------------------------------------------------
    new_gas_deliverability = scenario.get("new_gas_deliverability", "constrained")
    if new_gas_deliverability not in ("constrained", "exempt"):
        raise ValueError(f"Unknown new_gas_deliverability: {new_gas_deliverability}")

    if new_gas_deliverability == "exempt":
        # TASK 66 (Option 1 of TASK 64 Part B, register L120): the allowance is
        # sized at the DISPATCH efficiency `eta` (the blended fleet value that
        # m.gas_generation and the emissions expression convert at), so it equals
        # the thermal fuel the added capacity burns running at its availability
        # ceiling. It was sized at scenario["gas_new_efficiency"] (ccgt_efficiency_case),
        # which supplied only eta / efficiency (55.5% at the central arm) of that fuel
        # and made a more efficient plant claim LESS fuel. gas_new_efficiency is
        # therefore read nowhere in this builder; ccgt_efficiency_case is still
        # validated by the presence guard above.
        m.gas_new_capacity_mw = pyo.Expression(
            T, rule=lambda m, t: sum(m.gas_add[k] for k in range(0, t + 1))
        )
        m.gas_exempt_fuel_allowance_twh_th = pyo.Expression(
            T,
            rule=lambda m, t:
                m.gas_new_capacity_mw[t] * 8760 * gas_availability_factor
                / eta / 1e6
        )
        m.gas_balance = pyo.Constraint(
            T,
            rule=lambda m, t:
                m.gas_to_power[t] + m.gas_curtail[t]
                <= gas_avail[t] + m.gas_exempt_fuel_allowance_twh_th[t],
        )
    else:
        m.gas_balance = pyo.Constraint(
            T,
            rule=lambda m, t:
                m.gas_to_power[t] + m.gas_curtail[t] <= gas_avail[t],
        )

    solar_energy_per_mw = scenario["solar_cf"] * 8760 / 1_000_000
    baseline_mw = scenario["solar_baseline_mw"]

    # EaaS regime requires a defined service tariff (used by the bankability test).
    if financing_regime == "eaas":
        if tariff is None:
            raise ValueError("Tariff must be defined under EaaS regime.")

    # F1, Step 5/6: the four overnight capex terms below (public solar here;
    # EaaS solar right after; gas and storage further down, alongside
    # salvage/replacement) are built ONLY under capital_treatment_case==
    # 'pre_f1_overnight' -- the live objective must contain EXACTLY ONE of
    # {F1, salvage, replacement} (system_cost_npv assembly, below). Under
    # 'f1_tranche' these Pyomo components do not exist on the model at all
    # (not merely zeroed), so a stale reader fails loudly rather than
    # silently reading an overnight-capex number the objective no longer
    # uses -- see the F1 tranche mechanism after the replacement-capex
    # block for the 'f1_tranche' equivalents.
    if capital_treatment_case == "pre_f1_overnight":
        # Public solar CAPEX -- priced at each year's NREL ATB value (time-varying).
        public_solar_capex_npv = pyo.quicksum(
            m.DF[t] * m.solar_public_add[t]
            * m.solar_capex_param[t]
            for t in T
        )

        # EaaS (private) solar CAPEX.
        # MODELLING CHOICE (state explicitly in the thesis): EaaS-financed and
        # publicly-financed solar are priced at the SAME unit CAPEX -- they are the
        # same physical assets. The private cost of capital is represented NOT by an
        # objective-side CAPEX premium but by the bankability constraint, which
        # discounts EaaS revenue at the private/blended rate via remaining_npv_factor.
        # Applying a required_margin multiplier here would double-count: required_margin
        # is the bankability coverage cushion, a distinct object from the cost of
        # capital. The EaaS mechanism's effect in this model is therefore structural
        # (budget decoupling + a harder bankability test), not a cheaper sticker price.
        eaas_solar_capex_npv = pyo.quicksum(
            m.DF[t] * m.solar_eaas_add[t]
            * m.solar_capex_param[t]
            for t in T
        )
    # ============================================================
    # CONCESSIONAL CAPITAL ENVELOPE (scarcity constraint) -- RETIRED.
    # ============================================================
    # The pre-existing NPV-discounted, EaaS-solar-only mechanism
    # (concessional_envelope_npv scenario key, m.concessional_drawdown_npv,
    # m.concessional_envelope_constraint) is retired per envelope
    # resolution Decision 2 (docs/f1_implementation.md, "Envelope
    # resolution"): one envelope, one quantity. F1's OWN concessional
    # envelope -- undiscounted commitment stock, spanning solar
    # (public + EaaS) and storage, gas excluded -- replaces it; see the
    # 'f1_tranche' branch below (m.f1_concessional_envelope_constraint).

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
            bankable_revenue = (tariff * solar_energy_per_mw * m.remaining_npv_factor[t] * m.solar_eaas_add[t]) / required_margin

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
            # remaining_npv_factor is a fixed Param, so this arithmetic resolves
            # to a numeric constant; capex_init[t] is read as a plain float to
            # avoid any dependence on Param evaluation inside the rule.
            max_bankable_at_t = tariff * solar_energy_per_mw * m.remaining_npv_factor[t] / required_margin
            gap_at_t = capex_init[t] - max_bankable_at_t
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

    
    # NOTE: the solar-surplus charge-coupling constraint
    # (storage_charge <= surplus_frac * solar_generation) was intentionally
    # DROPPED. Storage is now governed by power and throughput limits only. The
    # surplus fraction was an unsourced 0.20 that could not be justified at
    # annual resolution; if reinstated it must be derived from an 8760
    # solar-vs-load surplus calculation. With the charge now debited in the
    # electricity balance, storage must "pay" for what it stores, so the coupling
    # is no longer needed to prevent phantom-energy arbitrage.

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
            + hydro[t]
            + m.storage_discharge[t]
            - m.storage_charge[t]
            + m.selfgen[t] * selfgen_busbar_factor
            + m.unserved[t]
            >= demand[t],
    )

    # ------------------------------------------------------------  
    # Reliability Constraint
    # ------------------------------------------------------------
    # new
    # Reliability now bounds GRID SHORTFALL (selfgen + unserved), not
    # unserved alone. With a backstop present, unserved solves to zero and
    # a constraint written on unserved alone would stop binding SILENTLY.
    # Reliability means the grid served the load -- not that someone,
    # somewhere, kept the lights on with a diesel set.
    if max_offgrid_fraction is not None:
        # 2.6, D9: accept either a scalar (flat obligation) or a per-year
        # sequence (obligation TRAJECTORY). A trajectory is what tests
        # front- vs back-loading, which a flat bound cannot: under a fixed NPV
        # envelope with declining capex, a tighter NEAR-TERM obligation forces
        # continuous early build and buys fewer lifetime MW.
        if isinstance(max_offgrid_fraction, (list, tuple)):
            if len(max_offgrid_fraction) != len(T):
                raise ValueError(
                    f"max_offgrid_fraction has {len(max_offgrid_fraction)} "
                    f"entries but the horizon has {len(T)} years."
                )
            _eps = {t: float(max_offgrid_fraction[t]) for t in T}
        else:
            _eps = {t: float(max_offgrid_fraction) for t in T}

        if grid_service_mode == "annual":
            # Year-by-year grid service obligation
            m.grid_service_obligation = pyo.Constraint(
                T,
                rule=lambda m, t:
                    m.selfgen[t] + m.unserved[t]
                    <= _eps[t] * demand[t]
            )

        elif grid_service_mode == "total":
            # Horizon-wide grid service obligation
            m.grid_service_obligation = pyo.Constraint(
                expr=sum(m.selfgen[t] + m.unserved[t] for t in T)
                <= sum(_eps[t] * demand[t] for t in T)
            )

        else:
            raise ValueError("grid_service_mode must be 'annual' or 'total'")
        # Backward-compatible component alias. optimize_experiments.py and
        # 99c_run_voll_sensitivity.py read m.reliability_constraint by name for
        # dual extraction. Pyomo lets one Constraint be referenced under a
        # second attribute without re-adding it to the model, so this is a
        # pointer, not a duplicate constraint.
        # m.reliability_constraint = m.grid_service_obligation

    # ------------------------------------------------------------
    # Firm Capacity Constraint (2.6, F0 Step 2.3)
    # ------------------------------------------------------------
    # Sigma_tech capacity_credit[tech] x cumulative_capacity[tech,t]
    #   >= peak_mw[t] * within_block_peaking_factor * (1 + reserve_margin[t])
    #
    # Unlike grid_service_obligation (a quantity-based service-obligation
    # bound on selfgen+unserved), this is a genuine capacity-adequacy
    # constraint: it tests whether CREDITED installed capacity covers a
    # peak-demand forecast with a reserve margin, independent of what the
    # LP chooses to dispatch. No such constraint existed anywhere in this
    # file before 2.6, F0 (see docs/F0_audit.md, Step B6 of this same
    # correction round).
    #
    # capacity_credit and peak_mw come from load_scenario() (2.6, F0 Step
    # 2.4) -- see scenarios.py: capacity_credit (solar/gas/storage/hydro
    # fractions, ALL SOURCE: [PENDING]), peak_multiple_case (drives
    # peak_mw/par_block), reserve_margin. Storage's credited capacity is its
    # POWER rating (m.storage_power_mw, MW), not its energy rating
    # (m.storage_capacity_mwh, MWh) -- the constraint is dimensioned in MW
    # throughout.
    #
    # reserve_margin: accepts either a flat scalar or a per-year sequence --
    # same dual-form pattern already used for max_offgrid_fraction above.
    # 'nirp_spinning' mode (load_scenario()) resolves to a per-year list
    # before it ever reaches here; build_model() just handles whichever
    # shape it receives.
    if isinstance(reserve_margin, (list, tuple)):
        if len(reserve_margin) != len(T):
            raise ValueError(
                f"scenario['reserve_margin'] has {len(reserve_margin)} "
                f"entries but the horizon has {len(T)} years."
            )
        _reserve_margin_by_year = {t: float(reserve_margin[t]) for t in T}
    else:
        _reserve_margin_by_year = {t: float(reserve_margin) for t in T}

    # within_block_peaking_factor (2.6, F0 Step 2.3): the ratio of the
    # INSTANTANEOUS system peak to par_block's 4-hour BLOCK-AVERAGE peak --
    # see scenarios.py, the module comment above peak_multiple_scenarios().
    # SOURCE: [PENDING]; DEFAULT 1.0 is a declared conservative
    # simplification (assumes no within-block peaking above the block
    # average), not an oversight -- same pattern as selfgen_busbar_factor
    # above. Set scenario["within_block_peaking_factor"] once sourced.
    # DELIBERATELY SEPARATE from reserve_margin -- do not merge the two:
    # one corrects a measurement-resolution gap (block-average vs
    # instantaneous), the other is a policy/engineering adequacy margin.
    within_block_peaking_factor = float(
        scenario.get("within_block_peaking_factor", 1.3448)
    )
    if within_block_peaking_factor < 1.0:
        raise ValueError(
            "within_block_peaking_factor < 1.0 would claim the instantaneous "
            "peak is LOWER than its own 4-hour block average, which is "
            "backwards."
        )

    m.firm_capacity_constraint = pyo.Constraint(
        T,
        rule=lambda m, t:
            capacity_credit["solar"] * m.solar_capacity_mw[t]
            + capacity_credit["gas"] * m.gas_capacity_mw[t]
            + capacity_credit["storage"] * m.storage_power_mw[t]
            + capacity_credit["hydro"] * hydro_capacity_mw_by_year[t]
            >= peak_mw[t] * within_block_peaking_factor
               * (1.0 + _reserve_margin_by_year[t])
    )

    # -------------------------
    # System Cost (Discounted NPV)
    # -------------------------
    # Gas cost is in USD per TWh_th. Fuel OPEX is unconditional -- it is
    # untouched by F1 (F1 only replaces the CAPEX/financing side).
    gas_opex_npv = pyo.quicksum(
        m.DF[t] * m.gas_to_power[t] * econ["GAS_COST_PER_TWH_TH"]
        for t in T
    )

    # ------------------------------------------------------------
    # Fixed + variable O&M (task_om_implementation, Part 3.3)
    # ------------------------------------------------------------
    # STORAGE_OM_PER_MWH_YR (io.py) never reached the objective for any
    # technology -- see docs/task_storage_om.md. om_cost_by_year is set by
    # load_scenario() from the REQUIRED om_case arm. Shared with
    # optimize_model_sliced.py (plan 2.5: "the two must share a cost
    # definition or the annual-vs-sliced premium compares different
    # things") via io.compute_om_cost_npv.
    #
    # This builder never models OCGT->CCGT conversion (Phase 2.8, Decision
    # 3 -- conversion_available=True raises above). All gas capacity and
    # generation here is therefore UNCONVERTED (gas_ct); the converted
    # (gas_cc) arguments are passed as zero for every t.
    om_cost_npv = compute_om_cost_npv(
        T=T,
        years=years,
        DF=m.DF,
        om_cost_by_year=scenario["om_cost_by_year"],
        storage_duration_hours=storage_duration_hours,
        solar_capacity_mw=m.solar_capacity_mw,
        storage_capacity_mwh=m.storage_capacity_mwh,
        gas_unconverted_capacity_mw=m.gas_capacity_mw,
        gas_converted_capacity_mw={t: 0.0 for t in T},
        gas_unconverted_generation_twh=m.gas_generation,
        gas_converted_generation_twh={t: 0.0 for t in T},
        # TASK 69 B.1 (register L95): hydro FOM term, OFF by default (see io.compute_om_cost_npv).
        **(dict(hydro_capacity_mw=hydro_capacity_mw_by_year,
                hydro_fom_usd_per_kw_yr=float(scenario.get("hydro_fom_usd_per_kw_yr", 0.0)))
           if scenario.get("hydro_om_term", False) else {}),
    )

    # F1, Step 5/6: matching comment above public_solar_capex_npv --
    # gas/storage overnight capex and the salvage/replacement mechanism
    # are built ONLY under 'pre_f1_overnight'. See the 'f1_tranche' branch
    # below for the replacement mechanism.
    if capital_treatment_case == "pre_f1_overnight":
        gas_capex_npv = pyo.quicksum(
            m.DF[t] * m.gas_add[t] * scenario["gas_capex_per_mw"]
            for t in T
        )

        # Storage CAPEX: assume built in 2025 (t=0) unless you model staged build
        storage_capex_npv = pyo.quicksum(
            m.DF[t] * m.storage_add[t] * m.storage_capex_param[t]
            for t in T
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

        # ------------------------------------------------------------
        # Replacement capex on in-horizon retirement (2.6, phase_2_3b Step 1).
        # A vintage built at build_t with technical life `life` reaches end of
        # life at build_t + life; if that falls inside the horizon, the asset
        # is rebuilt -- full capex charged again, discounted at m.DF[rebuild_t]
        # (the social rate in the year the rebuild occurs). Repeats for as many
        # rebuild cycles as fit before T_last. Generic across solar/storage/gas
        # so it stays correct if the horizon changes or gas_add is ever freed
        # (plan 5.3) -- under CURRENT lifetimes (solar/gas 30 yr == the 30-yr
        # horizon, gas_add pinned at 0) only storage actually rebuilds; see
        # docs/phase_2_3b_replacement_capex.md Step 0.2.
        #
        # SUPERSEDED (F1): this rebuild-on-retirement treatment and the
        # salvage credit above are the pre-F1 path only, selected by
        # capital_treatment_case='pre_f1_overnight'. 'f1_tranche' (below)
        # replaces both with the per-tranche CRF annuity truncated at the
        # horizon -- truncation IS the salvage treatment there.
        # ------------------------------------------------------------
        def _rebuild_years(build_t, life):
            """rebuild_t for every in-horizon replacement of the vintage built
            at build_t, `life` years apart, for as many cycles as fit <= T_last."""
            life_int = int(round(float(life)))
            k = 1
            while build_t + k * life_int <= T_last:
                yield build_t + k * life_int
                k += 1

        m.salvage_npv = pyo.Expression(
            expr=m.DF[T_last] * (
                pyo.quicksum(
                    (m.solar_public_add[t] + m.solar_eaas_add[t])
                    * m.solar_capex_param[t]
                    * _salvage_fraction(t, lives["solar"])
                    for t in T
                )
                + pyo.quicksum(
                    m.storage_add[t] * m.storage_capex_param[t]
                    * _salvage_fraction(t, lives["storage"])
                    for t in T
                )
                + pyo.quicksum(
                    m.gas_add[t] * scenario["gas_capex_per_mw"]
                    * _salvage_fraction(t, lives["gas"])
                    for t in T
                )
                # Replacement vintages get the SAME straight-line salvage
                # treatment as original builds, credited at T_last (Step 1).
                + pyo.quicksum(
                    (m.solar_public_add[build_t] + m.solar_eaas_add[build_t])
                    * m.solar_capex_param[rebuild_t]
                    * _salvage_fraction(rebuild_t, lives["solar"])
                    for build_t in T
                    for rebuild_t in _rebuild_years(build_t, lives["solar"])
                )
                + pyo.quicksum(
                    m.storage_add[build_t] * m.storage_capex_param[rebuild_t]
                    * _salvage_fraction(rebuild_t, lives["storage"])
                    for build_t in T
                    for rebuild_t in _rebuild_years(build_t, lives["storage"])
                )
                + pyo.quicksum(
                    m.gas_add[build_t] * scenario["gas_capex_per_mw"]
                    * _salvage_fraction(rebuild_t, lives["gas"])
                    for build_t in T
                    for rebuild_t in _rebuild_years(build_t, lives["gas"])
                )
            )
        )

        # Replacement capex itself: full capex again at the rebuild year,
        # discounted at m.DF[rebuild_t]. A SEPARATE, visible cost -- not folded
        # into gas/storage/solar capex_npv above (2.6, phase_2_3b Step 1). Read
        # independently via m.cost_replacement_capex_npv (mirrors the
        # m.cost_<x>_npv reporting pattern used for the other cost components
        # below).
        m.cost_replacement_capex_npv = pyo.Expression(
            expr=(
                pyo.quicksum(
                    m.DF[rebuild_t]
                    * (m.solar_public_add[build_t] + m.solar_eaas_add[build_t])
                    * m.solar_capex_param[rebuild_t]
                    for build_t in T
                    for rebuild_t in _rebuild_years(build_t, lives["solar"])
                )
                + pyo.quicksum(
                    m.DF[rebuild_t] * m.storage_add[build_t] * m.storage_capex_param[rebuild_t]
                    for build_t in T
                    for rebuild_t in _rebuild_years(build_t, lives["storage"])
                )
                + pyo.quicksum(
                    m.DF[rebuild_t] * m.gas_add[build_t] * scenario["gas_capex_per_mw"]
                    for build_t in T
                    for rebuild_t in _rebuild_years(build_t, lives["gas"])
                )
            )
        )

    else:
        # ============================================================
        # F1 -- per-tranche CRF capital charge (capital_treatment_case
        # == 'f1_tranche'; Steps 2-5)
        # ============================================================
        # Replaces ALL FOUR overnight capex terms, salvage, AND
        # replacement with one mechanism: each channel's build in year v
        # is split into a concessional and a commercial VOLUME (new
        # decision variables, below), each amortised at its OWN tranche's
        # capital recovery factor over its OWN tenor, truncated at the
        # horizon. Truncation IS the salvage treatment (a tranche that
        # would still be paying down principal past T_last simply stops
        # being charged there) -- there is no separate residual credit or
        # rebuild-on-retirement charge under this arm. Both CRFs are
        # CONSTANTS (data, not decision variables), so every objective
        # coefficient below is exogenous and the objective stays LINEAR.
        def _crf(r, L):
            """
            Capital recovery factor: CRF(r, L) = r(1+r)^L / ((1+r)^L - 1).

            Negative-real-rate floor (F1, declared not incidental): if
            r <= 0, CRF := 1/L. This floor makes the model INSENSITIVE to
            concessional rate variation below inflation (every r <= 0
            collapses to the same 1/L annuity factor) and BIASES AGAINST
            concessionality: the true CRF falls without bound as r -> 0
            from above and keeps falling for r < 0, so flooring at 1/L
            OVERSTATES the annuity charge on a below-inflation
            concessional rate rather than rewarding it.
            """
            if r <= 0:
                return 1.0 / L
            return r * (1.0 + r) ** L / ((1.0 + r) ** L - 1.0)

        _L_conc = int(round(float(scenario["L_conc"])))
        _L_comm = int(round(float(scenario["L_comm"])))
        _r_conc_nom = float(scenario["r_conc_nominal"])
        _r_comm_nom = float(scenario["r_comm_nominal"])
        _us_inflation = float(scenario["us_inflation"])

        # rates SOURCED nominal USD; model is real-USD (see register);
        # deflated here, not at source.
        _r_conc_real = (1.0 + _r_conc_nom) / (1.0 + _us_inflation) - 1.0
        _r_comm_real = (1.0 + _r_comm_nom) / (1.0 + _us_inflation) - 1.0

        # F2 (docs/f1_implementation.md, "F2: asset lifetime restored",
        # Decision 1): technical-life CEILING on the recovery window.
        # asset_lifetimes() is used here ONLY as a ceiling on the recovery
        # period, NEVER as the recovery period itself -- consistent with
        # its own docstring's insistence that technical service life is
        # not a capital recovery period. Capital cannot be amortised over
        # longer than the asset physically survives:
        #   L_eff(channel, tranche) = min(L_tranche, asset_lifetimes[channel]).
        # A SEPARATE read of scenario["asset_lifetimes"] from the one in
        # the 'pre_f1_overnight' branch above -- not shared with it, since
        # that branch is the regression instrument and must not be
        # touched, and its `_rebuild_years` closure is local to it and
        # inaccessible here under Python scoping regardless.
        _lives = scenario.get("asset_lifetimes", None)
        if _lives is None:
            raise ValueError(
                "scenario['asset_lifetimes'] is required. build_model computes "
                "the F1/F2 per-channel recovery-window ceiling from it and "
                "will not silently assume a lifetime. See scenarios.asset_lifetimes()."
            )
        for _k in ("solar", "storage", "gas"):
            if _lives.get(_k) is None:
                raise ValueError(
                    f"asset_lifetimes['{_k}'] is None -- [SOURCE NEEDED]. "
                    "Refusing to compute the F2 recovery-window ceiling from "
                    "a placeholder."
                )

        _L_eff_conc = {
            ch: min(_L_conc, int(round(float(_lives[ch]))))
            for ch in ("solar", "storage", "gas")
        }
        _L_eff_comm = {
            ch: min(_L_comm, int(round(float(_lives[ch]))))
            for ch in ("solar", "storage", "gas")
        }

        # Both CRFs remain plain Python floats computed from L_eff (data,
        # not decision variables), so every objective coefficient below
        # stays exogenous and the LP stays LINEAR (Linearity invariant,
        # restated here per F2 Step 1 -- unchanged in kind from the F1
        # invariant above, now technology-specific via L_eff).
        # TASK 51 M1: timing convention of the capital charge. REQUIRED key,
        # no fallback (load_scenario() defaults it to 'advance', the
        # behaviour of every result produced before the arm existed).
        # 'advance': window starts in the build year, CRF as is.
        # 'arrears': window starts one year later, CRF as is.
        # 'due_crf': window as 'advance', CRF / (1 + r) at its own real rate.
        # See scenarios.capital_window_scenarios().
        # local import: src.scenarios imports slice_hours from this module.
        from src.scenarios import capital_window_scenarios
        _cw_case = scenario["capital_window_case"]
        _cw = capital_window_scenarios()
        if _cw_case not in _cw:
            raise ValueError(f"Unknown capital_window_case: {_cw_case}")
        _win_offset = int(_cw[_cw_case]["window_offset"])
        _crf_due = bool(_cw[_cw_case]["crf_due"])

        def _crf_tc(r, L):
            c = _crf(r, L)
            return c / (1.0 + r) if _crf_due else c

        _crf_conc = {ch: _crf_tc(_r_conc_real, _L_eff_conc[ch]) for ch in ("solar", "storage", "gas")}
        _crf_comm = {ch: _crf_tc(_r_comm_real, _L_eff_comm[ch]) for ch in ("solar", "storage", "gas")}

        _T_last = len(T) - 1
        # Per-vintage, per-tranche, per-CHANNEL discounted-years-in-window
        # coefficient (F2: was a single tranche-wide pair; now
        # technology-specific via L_eff): sum(DF[t] for t in {v, ...,
        # min(v+L_eff-1, T_last)}). Exogenous data (DF and L_eff are both
        # data, fixed before the solve).
        # TASK 51 amendment 1.1: under 'arrears' the window is shifted one
        # year right AND the horizon cut-off is shifted with it
        # (upper bound T_last + 1 + offset), with the discount-factor series
        # extended one year (index T_last + 1, i.e. 2055) for window
        # purposes only, so every vintage is charged the SAME NUMBER of
        # instalments as under 'advance', each discounted one year further.
        # The advance/arrears delta is then the pure timing effect, not
        # timing plus a deeper horizon truncation. m.DF itself is not
        # extended. Same expression as the model's own DF (line ~468).
        def _df_ext(t):
            if t <= _T_last:
                return float(pyo.value(m.DF[t]))
            return 1.0 / ((1.0 + r_social) ** t)

        def _win_range(v, L):
            return range(v + _win_offset, min(v + _win_offset + L, _T_last + 1 + _win_offset))

        def _win_range_advance(v, L):
            return range(v, min(v + L, _T_last + 1))

        _df_window_conc = {
            ch: {v: sum(_df_ext(t) for t in _win_range(v, _L_eff_conc[ch])) for v in T}
            for ch in ("solar", "storage", "gas")
        }
        _df_window_comm = {
            ch: {v: sum(_df_ext(t) for t in _win_range(v, _L_eff_comm[ch])) for v in T}
            for ch in ("solar", "storage", "gas")
        }
        # Instalment counts per tranche, channel and vintage, under the
        # chosen convention and under 'advance'. They MUST be equal; if not
        # the extension is not working and the measurement is void.
        capital_window_instalments = {
            tr: {
                ch: {v: (len(_win_range(v, Ld[ch])), len(_win_range_advance(v, Ld[ch]))) for v in T}
                for ch in ("solar", "storage", "gas")
            }
            for tr, Ld in (("conc", _L_eff_conc), ("comm", _L_eff_comm))
        }
        for _tr, _byc in capital_window_instalments.items():
            for _ch, _byv in _byc.items():
                for _v, (_n_chosen, _n_adv) in _byv.items():
                    if _n_chosen != _n_adv:
                        raise AssertionError(
                            f"capital_window_case={_cw_case!r}: {_tr}/{_ch} vintage {_v} "
                            f"has {_n_chosen} instalments, 'advance' has {_n_adv}; the "
                            "DF extension is not working. Measurement void."
                        )
        m.capital_window_instalments = capital_window_instalments
        m.capital_window_W = {"conc": _df_window_conc, "comm": _df_window_comm}

        # F2 Step 2: reuse of the EXISTING _rebuild_years(build_t, life)
        # helper (see docs/f1_implementation.md Step 0 inventory for why
        # this is a verbatim duplicate rather than the same object --
        # the 'pre_f1_overnight' branch's copy is local to that branch and
        # off-limits to edit).
        def _rebuild_years(build_t, life):
            """rebuild_t for every in-horizon replacement of the vintage built
            at build_t, `life` years apart, for as many cycles as fit <= T_last."""
            life_int = int(round(float(life)))
            k = 1
            while build_t + k * life_int <= _T_last:
                yield build_t + k * life_int
                k += 1

        # ---- Tranche Vars + split identity, one pair per buildable
        # channel (public solar, EaaS solar, gas, storage).
        #
        # Linearity invariant: the coefficient on each build variable
        # (unit capex x CRF x df_window) must be exogenous data. It may
        # vary over vintage and technology; it may NOT depend on any
        # decision variable, including cumulative deployment.
        # Time-varying capex trajectories (m.solar_capex_param, from
        # load_solar_capex_by_year) are fine -- exogenous data indexed by
        # t, fixed before the solve. Learning curves, volume discounts,
        # and endogenous FX are NOT: none are present here.
        m.solar_public_conc_vol = pyo.Var(T, domain=pyo.NonNegativeReals)
        m.solar_public_comm_vol = pyo.Var(T, domain=pyo.NonNegativeReals)
        m.solar_public_split_identity = pyo.Constraint(
            T, rule=lambda m, t:
                m.solar_public_conc_vol[t] + m.solar_public_comm_vol[t]
                == m.solar_capex_param[t] * m.solar_public_add[t]
        )

        m.solar_eaas_conc_vol = pyo.Var(T, domain=pyo.NonNegativeReals)
        m.solar_eaas_comm_vol = pyo.Var(T, domain=pyo.NonNegativeReals)
        m.solar_eaas_split_identity = pyo.Constraint(
            T, rule=lambda m, t:
                m.solar_eaas_conc_vol[t] + m.solar_eaas_comm_vol[t]
                == m.solar_capex_param[t] * m.solar_eaas_add[t]
        )

        # Gas is COMMERCIAL-ONLY (envelope resolution Decision 1): DARES is
        # a distributed-renewables facility, and IDA concessional credit
        # for a CCGT is not a defensible claim. gas_conc_vol is declared
        # (the split-identity Constraint and the capital-charge Expression
        # below still reference it uniformly across all four channels) but
        # FIXED at 0.0 for every vintage, so gas_split_identity forces
        # gas_comm_vol[t] == gas_capex_per_mw * gas_add[t] -- every dollar
        # of gas capex is commercial.
        m.gas_conc_vol = pyo.Var(T, domain=pyo.NonNegativeReals)
        m.gas_comm_vol = pyo.Var(T, domain=pyo.NonNegativeReals)
        for _t in T:
            m.gas_conc_vol[_t].fix(0.0)
        m.gas_split_identity = pyo.Constraint(
            T, rule=lambda m, t:
                m.gas_conc_vol[t] + m.gas_comm_vol[t]
                == scenario["gas_capex_per_mw"] * m.gas_add[t]
        )

        m.storage_conc_vol = pyo.Var(T, domain=pyo.NonNegativeReals)
        m.storage_comm_vol = pyo.Var(T, domain=pyo.NonNegativeReals)
        m.storage_split_identity = pyo.Constraint(
            T, rule=lambda m, t:
                m.storage_conc_vol[t] + m.storage_comm_vol[t]
                == m.storage_capex_param[t] * m.storage_add[t]
        )

        # ---- Concessional envelope (envelope resolution): commitment
        # STOCK, UNDISCOUNTED, spanning solar (public + EaaS) and storage
        # ONLY -- gas is excluded per Decision 1 above. This is the ONLY
        # concessional envelope mechanism left in the model: the
        # pre-existing NPV-discounted, EaaS-solar-only
        # concessional_envelope_npv/m.concessional_drawdown_npv/
        # m.concessional_envelope_constraint mechanism is RETIRED (one
        # envelope, one quantity -- see the removed block above this
        # function's CONCESSIONAL CAPITAL ENVELOPE section).
        # concessional_envelope_case is REQUIRED, no default (see
        # scenarios.py) -- this constraint therefore ALWAYS builds under
        # 'f1_tranche', with no `if ... is not None` guard: a missing
        # envelope is an error at load_scenario(), not an unconstrained
        # default reachable here.
        _concessional_envelope_usd = scenario["concessional_envelope_usd"]
        m.f1_concessional_envelope_constraint = pyo.Constraint(
            expr=(
                pyo.quicksum(m.solar_public_conc_vol[t] for t in T)
                + pyo.quicksum(m.solar_eaas_conc_vol[t] for t in T)
                + pyo.quicksum(m.storage_conc_vol[t] for t in T)
            ) <= _concessional_envelope_usd
        )

        # TASK 98: supplied concessional-commercial division (Chapter 1 Question 2's counterfactual). Inert unless
        # scenario["conc_share_rule"] is set; see src/supplied_division.py. The envelope constraint above remains.
        _conc_share_rule = scenario.get("conc_share_rule")
        if _conc_share_rule is not None:
            from src.supplied_division import add_supplied_division_constraints
            add_supplied_division_constraints(m, scenario, T, _conc_share_rule)

        # ---- Reporting (Step 5): concessional volume drawn BY VINTAGE
        # (undiscounted), aggregated across all four channels. A
        # long-tenor tranche truncated at the horizon escapes more of its
        # annuity than a short-tenor one, so late vintages may show a
        # concessional spike analogous to the 2054 salvage arbitrage
        # (2.5-era finding) -- reported here, NOT corrected.
        m.concessional_volume_by_vintage = pyo.Expression(
            T, rule=lambda m, t:
                m.solar_public_conc_vol[t] + m.solar_eaas_conc_vol[t]
                + m.storage_conc_vol[t]
        )

        # ---- Per-channel capital charge (NPV): CRF x volume, held
        # constant over the tranche's OWN window, discounted year-by-year,
        # truncated at the horizon via the precomputed _df_window_*
        # coefficients. NOT named m.cost_<channel>_capex_npv (the
        # pre-F1 names, built only under 'pre_f1_overnight' above) --
        # deliberately renamed so a stale reader (e.g.
        # extract_planning_diagnostics, which reads the old names
        # unconditionally) fails loudly under 'f1_tranche' instead of
        # silently reading a number that no longer means overnight capex.
        m.cost_public_solar_capital_charge_npv = pyo.Expression(
            expr=pyo.quicksum(
                _crf_conc["solar"] * _df_window_conc["solar"][v] * m.solar_public_conc_vol[v]
                + _crf_comm["solar"] * _df_window_comm["solar"][v] * m.solar_public_comm_vol[v]
                for v in T
            )
        )
        m.cost_eaas_solar_capital_charge_npv = pyo.Expression(
            expr=pyo.quicksum(
                _crf_conc["solar"] * _df_window_conc["solar"][v] * m.solar_eaas_conc_vol[v]
                + _crf_comm["solar"] * _df_window_comm["solar"][v] * m.solar_eaas_comm_vol[v]
                for v in T
            )
        )
        m.cost_gas_capital_charge_npv = pyo.Expression(
            expr=pyo.quicksum(
                _crf_conc["gas"] * _df_window_conc["gas"][v] * m.gas_conc_vol[v]
                + _crf_comm["gas"] * _df_window_comm["gas"][v] * m.gas_comm_vol[v]
                for v in T
            )
        )
        m.cost_storage_capital_charge_npv = pyo.Expression(
            expr=pyo.quicksum(
                _crf_conc["storage"] * _df_window_conc["storage"][v] * m.storage_conc_vol[v]
                + _crf_comm["storage"] * _df_window_comm["storage"][v] * m.storage_comm_vol[v]
                for v in T
            )
        )

        # F2 Step 2: replacement capex restored under 'f1_tranche'. A
        # vintage whose TECHNICAL life ends in-horizon is rebuilt and
        # recharged -- truncation at the horizon (above) handles
        # END-OF-HORIZON RESIDUAL VALUE only; it does not handle
        # MID-HORIZON ASSET DEATH. These are orthogonal, not alternatives
        # (Decision, docs/f1_implementation.md "F2: asset lifetime
        # restored"). F2 Decision 3: rebuilt vintages are COMMERCIAL-ONLY
        # -- DARES is a distributed-renewables ACCESS facility, not an
        # asset-churn facility; an IDA credit funds new access, not
        # battery replacement. Mechanically: no *_conc_vol variables are
        # created for a rebuild, the charge uses CRF(r_comm, L_eff_comm)
        # directly, and it does NOT enter m.f1_concessional_envelope_constraint's
        # sum above (that sum only ever referenced the original *_conc_vol
        # Vars and is unmodified by this addition).
        m.cost_rebuild_capital_charge_npv = pyo.Expression(
            expr=(
                pyo.quicksum(
                    _crf_comm["solar"] * _df_window_comm["solar"][rebuild_t]
                    * (m.solar_public_add[build_t] + m.solar_eaas_add[build_t])
                    * m.solar_capex_param[rebuild_t]
                    for build_t in T
                    for rebuild_t in _rebuild_years(build_t, _lives["solar"])
                )
                + pyo.quicksum(
                    _crf_comm["storage"] * _df_window_comm["storage"][rebuild_t]
                    * m.storage_add[build_t] * m.storage_capex_param[rebuild_t]
                    for build_t in T
                    for rebuild_t in _rebuild_years(build_t, _lives["storage"])
                )
                + pyo.quicksum(
                    _crf_comm["gas"] * _df_window_comm["gas"][rebuild_t]
                    * m.gas_add[build_t] * scenario["gas_capex_per_mw"]
                    for build_t in T
                    for rebuild_t in _rebuild_years(build_t, _lives["gas"])
                )
            )
        )

        m.capital_charge_total_npv = pyo.Expression(
            expr=(
                m.cost_public_solar_capital_charge_npv
                + m.cost_eaas_solar_capital_charge_npv
                + m.cost_gas_capital_charge_npv
                + m.cost_storage_capital_charge_npv
                + m.cost_rebuild_capital_charge_npv
            )
        )

    # Unserved energy penalty (VoLL in USD/TWh)
    unserved_npv = pyo.quicksum(
        m.DF[t] * m.unserved[t] * econ["UNSERVED_ENERGY_PENALTY"]
        for t in T
    )

    # Private self-generation cost. This is a REAL RESOURCE COST and belongs in
    # system_cost_npv. It is NOT public expenditure and must stay out of the
    # public budget constraint -- see 2.8 below, which is a check, not an edit.
    selfgen_npv = pyo.quicksum(
        m.DF[t] * m.selfgen[t] * selfgen_price
        for t in T
    )

    # EF per MWh_e derived from the thermal factor and THIS scenario's gas_eta,
    # so emissions accounting uses the same efficiency as dispatch. Sweeping
    # gas_eta automatically and consistently sweeps the electrical emission factor.
    ef_e = econ["EF_TCO2_PER_MWH_TH"] / scenario["gas_eta"]

    # --------------------------------------------------------------
    # THREE emissions quantities, deliberately distinct (2.6, D3).
    #
    # The earlier single-switch design conflated two different things:
    # what the NDC COMMITS, and what is PHYSICALLY EMITTED. They are not
    # the same set, and the cap and the carbon price must attach to
    # different ones.
    #
    #   emissions_grid_only_by_year  gas-fired grid generation. This is the
    #                                NDC commitment boundary: the cap was
    #                                derived as the gas-power share (7.24%)
    #                                of the energy-sector abatement target.
    #   emissions_selfgen_by_year    private diesel. Inside Nigeria's
    #                                economy-wide inventory, but NOT inside
    #                                the gas-power slice the cap was sized
    #                                for. Charging it against that budget
    #                                would be an accounting error.
    #   emissions_by_year            TOTAL. Always both. This is physical
    #                                truth and is what a carbon PRICE must
    #                                see, and what gets reported.
    # --------------------------------------------------------------
    if gas_vintage_dispatch:
        # emissions follow the THERMAL draw (as build_model_sliced()): tCO2 = TWh_th x 1e6 x tCO2/MWh_th.
        # At gas_new_efficiency == gas_eta this equals the generation-based expression below.
        m.emissions_grid_only_by_year = pyo.Expression(
            T,
            rule=lambda m, t:
                m.gas_to_power[t] * 1e6 * econ["EF_TCO2_PER_MWH_TH"],
        )
    else:
        m.emissions_grid_only_by_year = pyo.Expression(
            T,
            rule=lambda m, t:
                m.gas_generation[t] * 1e6 * ef_e,
        )
    m.emissions_selfgen_by_year = pyo.Expression(
        T,
        rule=lambda m, t:
            m.selfgen[t] * 1e6 * selfgen_ef_e,
    )
    # ALWAYS total -- no longer switched by selfgen_in_emissions_cap.
    m.emissions_by_year = pyo.Expression(
        T,
        rule=lambda m, t:
            m.emissions_grid_only_by_year[t]
            + m.emissions_selfgen_by_year[t],
    )
    # What the CAP constrains. Default is grid-only (correct NDC boundary).
    # Setting selfgen_in_emissions_cap=True widens it to total, which is the
    # alternative-boundary arm, not the central case.
    if selfgen_in_emissions_cap:
        m.emissions_capped_by_year = pyo.Expression(
            T, rule=lambda m, t: m.emissions_by_year[t])
    else:
        m.emissions_capped_by_year = pyo.Expression(
            T, rule=lambda m, t: m.emissions_grid_only_by_year[t])
    m.emissions_capped = pyo.Expression(
        expr=pyo.quicksum(m.emissions_capped_by_year[t] for t in T))


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

    # F1, Step 5/6: the live objective contains EXACTLY ONE of {F1 tranche
    # charge, overnight capex + salvage + replacement}. The
    # 'pre_f1_overnight' branch below is written with the IDENTICAL term
    # order as the pre-F1 system_cost_npv formula (not refactored through
    # an intermediate variable) so the expression TREE -- not just its
    # value -- is unchanged: Python's + is left-associative and floating-
    # point addition is not strictly associative, so even a
    # mathematically-equivalent regrouping could shift the last bit of
    # the regression value. This preserves bit-exact reproduction.
    if capital_treatment_case == "pre_f1_overnight":
        system_cost_npv = (
            gas_opex_npv
            + public_solar_capex_npv
            + eaas_solar_capex_npv
            + gas_capex_npv
            + storage_capex_npv
            + m.cost_replacement_capex_npv
            + selfgen_npv
            + unserved_npv
            + carbon_cost_npv
            - m.salvage_npv
            + om_cost_npv
        )
    else:
        system_cost_npv = (
            gas_opex_npv
            + m.capital_charge_total_npv
            + selfgen_npv
            + unserved_npv
            + carbon_cost_npv
            + om_cost_npv
        )
    m.system_cost_npv = pyo.Expression(expr=system_cost_npv)
    
    # ============================================================
    # COST COMPONENT EXPRESSIONS (for decomposition diagnostics)
    # ============================================================
    m.cost_gas_opex_npv          = pyo.Expression(expr=gas_opex_npv)
    # F1, Step 5/6: these four ONLY exist under 'pre_f1_overnight' -- see
    # m.cost_<channel>_capital_charge_npv for the 'f1_tranche' equivalents,
    # built above. Deliberately NOT built here under 'f1_tranche' so a
    # stale reader (e.g. extract_planning_diagnostics, which reads these
    # names unconditionally) fails loudly rather than silently reading an
    # overnight-capex number the objective no longer uses.
    if capital_treatment_case == "pre_f1_overnight":
        m.cost_gas_capex_npv         = pyo.Expression(expr=gas_capex_npv)
        m.cost_public_solar_capex_npv = pyo.Expression(expr=public_solar_capex_npv)
        m.cost_eaas_solar_capex_npv   = pyo.Expression(expr=eaas_solar_capex_npv)
        m.cost_storage_capex_npv      = pyo.Expression(expr=storage_capex_npv)
    m.cost_unserved_voll_npv      = pyo.Expression(expr=unserved_npv)
    m.cost_selfgen_npv            = pyo.Expression(expr=selfgen_npv)
    # Avoided-genset metrics. The EaaS case is measured against these: every
    # TWh the grid serves is a TWh of private diesel spend and private diesel
    # CO2 that does not happen. This is the strongest quantitative channel for
    # EaaS in the model and must be reported, not left implicit.
    m.selfgen_twh_by_year = pyo.Expression(T, rule=lambda m, t: m.selfgen[t])
    m.selfgen_cost_by_year = pyo.Expression(
        T, rule=lambda m, t: m.selfgen[t] * selfgen_price)
    m.emissions_selfgen = pyo.Expression(
        expr=pyo.quicksum(m.emissions_selfgen_by_year[t] for t in T))
    m.emissions_grid_only = pyo.Expression(
        expr=pyo.quicksum(m.emissions_grid_only_by_year[t] for t in T))
    m.cost_carbon_npv = pyo.Expression(expr=carbon_cost_npv)
    m.cost_om_npv = pyo.Expression(expr=om_cost_npv)

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
                    + m.storage_add[t] * m.storage_capex_param[t]
                    + m.eaas_subsidy[t]
                )
                for t in T
            ) <= public_budget_npv
        )
        # Explicit flag so diagnostics can detect the constraint's presence
        # without relying solely on hasattr. (This flag was previously read by
        # extract_planning_diagnostics but never set, so the FIN-3 shadow price
        # always returned None.)
        m.has_public_budget_constraint = True

    # ------------------------------------------------------------
    # Objective & Emissions Constraints
    # ------------------------------------------------------------
    # Plan 0.4, item B: the objective is UNCONDITIONALLY system_cost_npv,
    # minimised -- the weighted-sum branch (weight_cost * system_cost_npv +
    # weight_emissions * m.emissions) is DELETED. It was dead code: no
    # caller anywhere in src/ or scripts/ ever set weight_cost or
    # weight_emissions away from their old defaults (1.0 / 0.0), so the
    # branch below never executed in any run. Only whether an emissions
    # constraint is ALSO built varies with the cap arguments.
    if emissions_cap_by_year is not None:

        if len(emissions_cap_by_year) != len(list(T)):
            raise ValueError(
                "emissions_cap_by_year length mismatch."
            )

        m.emissions_constraint = pyo.Constraint(
            T,
            rule=lambda m, t:
                m.emissions_capped_by_year[t] <= emissions_cap_by_year[t],
        )

    elif emissions_cap is not None:

        m.emissions_constraint = pyo.Constraint(
            expr=m.emissions_capped <= emissions_cap
        )

    m.objective = pyo.Objective(
        expr=system_cost_npv,
        sense=pyo.minimize,
    )

    return m


def leakage_diagnostic(m, threshold=0.10):
    """Post-solve guard on carbon leakage past the cap boundary (2.6, D3).

    With the cap on grid-only emissions, the model can satisfy an NDC target
    by shedding grid load onto private diesel: the cap is met, reported
    compliance is achieved, and NATIONAL emissions rise. Genset EF_e (0.729 to
    4.638 tCO2/MWh across the arms) exceeds grid EF_e (0.635), so the swap is
    always emissions-negative in reality.

    The grid service obligation is the primary defence. This is the check that
    it worked. A flagged run is not an NDC-compliant plan.

    threshold : declared, not derived. 0.10 says that if more than a tenth of
    physical emissions escape the commitment boundary, the plan's compliance
    is an artefact of the boundary. State the value used in the methods
    chapter; it is a judgement, not a result.
    """
    total = float(pyo.value(m.emissions))
    selfgen = float(pyo.value(m.emissions_selfgen))
    grid = float(pyo.value(m.emissions_grid_only))
    capped = float(pyo.value(m.emissions_capped))
    share = selfgen / total if total > 0 else 0.0
    return {
        "emissions_total_tco2": total,
        "emissions_grid_only_tco2": grid,
        "emissions_selfgen_tco2": selfgen,
        "emissions_capped_tco2": capped,
        "leakage_tco2": total - capped,
        "leakage_share": share,
        "threshold": threshold,
        "flagged": share > threshold,
    }


def firm_capacity_diagnostic(m, years):
    """Post-solve slack and dual on m.firm_capacity_constraint, per year
    (2.6, F0 Step 2.3).

    slack_mw : credited capacity in excess of peak_mw[t]*(1+reserve_margin).
               0 (to solver tolerance) means the constraint is binding.
    dual     : shadow price (USD, NPV-discounted, per additional MW of
               required firm capacity) if the solver returned one for this
               constraint, else None.
    """
    out = {}
    for t, y in enumerate(years):
        con = m.firm_capacity_constraint[t]
        out[int(y)] = {
            "slack_mw": float(pyo.value(con.body) - pyo.value(con.lower)),
            "dual": m.dual.get(con, None),
        }
    return out


# ============================================================
# SOLVER WRAPPER
# ============================================================

def solve_model(model, scenario=None, solver="cbc", tee=False):
    """
    Solve a Pyomo model and return a minimal status dictionary.

    Plan 0.4, item A (docs/numerical_hygiene_0_4.md): every regression arm
    in this project rests on bit-identical reproduction, so the solver's
    numerical tolerances and version are RECORDED explicitly rather than
    left to whatever a given solver build happens to ship as its
    compiled-in default -- an upgrade could otherwise silently change a
    tolerance with no way to attribute a result difference. The values
    below are each solver's own DOCUMENTED DEFAULT, set explicitly but NOT
    tightened; the point is recording, not changing behaviour.

    tee defaults to False (was hardcoded True): every solve no longer
    floods stdout with solver iteration output unless a caller opts in
    with tee=True. No caller in src/ or scripts/ currently passes tee, and
    grepping the repo shows nothing parses the solver's printed log (no
    "Optimal objective" or similar text match anywhere) -- so this is a
    pure logging-verbosity change, not a functional one. A runner's own
    summary print (via the returned status dict / diagnostics) is
    unaffected.
    """
    solver_obj = pyo.SolverFactory(solver)

    if not solver_obj.available():
        raise RuntimeError(f"Solver not available: {solver}")

    # Solver version: recorded so a silent solver upgrade is attributable.
    # Captured before solving since it does not depend on a solution.
    try:
        solver_version = solver_obj.version()
    except Exception:
        solver_version = None

    if solver == "cbc":
        # CBC's own documented/compiled-in defaults (confirmed against the
        # installed build via `cbc` interactive: `primalTolerance`,
        # `dualTolerance`, `integerTolerance` all report 1e-07). Set
        # EXPLICITLY so a future CBC upgrade that ships a different
        # compiled-in default is caught by diffing this function, not by
        # an unattributable change in results.
        solver_obj.options["primalTolerance"] = 1e-7
        solver_obj.options["dualTolerance"] = 1e-7
        solver_obj.options["integerTolerance"] = 1e-7
    elif solver in ("highs", "appsi_highs"):
        # HiGHS's documented option defaults (HiGHS options reference):
        # primal_feasibility_tolerance / dual_feasibility_tolerance are the
        # LP feasibility tolerances (1e-7 each); mip_feasibility_tolerance
        # is the integrality-equivalent tolerance (1e-6). Sourced from
        # HiGHS's published documentation -- NOT independently re-verified
        # against an installed HiGHS build in this environment, since
        # HiGHS is not installed here (only CBC is; see
        # docs/numerical_hygiene_0_4.md).
        solver_obj.options["primal_feasibility_tolerance"] = 1e-7
        solver_obj.options["dual_feasibility_tolerance"] = 1e-7
        solver_obj.options["mip_feasibility_tolerance"] = 1e-6
    else:
        print(
            f"  NOTE: solve_model() has no documented-default tolerance "
            f"set recorded for solver={solver!r}; none applied. Add one "
            "here before using this solver for a regression-bearing run."
        )

    res = solver_obj.solve(model, tee=tee)
    tc = res.solver.termination_condition

    if tc not in (
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
    ):
        raise RuntimeError(
            f"Solver failed with termination condition: {tc}"
        )

    # ---- Post-solve guard: in-horizon retirement IS rebuilt (plan 2.5;
    # D6 register CLOSED -- docs/limitations_register.md,
    # docs/d6_storage_guard.md)
    # ------------------------------------------------------------
    # D6, as originally recorded, claimed a vintage retiring within the
    # horizon gets NO replacement capex and is dispatched for free. That
    # was true before the rebuild mechanism existed, but has been false
    # since Phase 2.6 (pre_f1_overnight: m.cost_replacement_capex_npv,
    # entering system_cost_npv) and since F2 (f1_tranche:
    # m.cost_rebuild_capital_charge_npv, entering
    # m.capital_charge_total_npv). Both sum a rebuild at every
    # build_t + k*life <= T_last via the builder's local _rebuild_years()
    # closure. The OLD guard fired on this ordinary, now-compensated
    # event on every run (storage life 15 yr < the horizon) with a
    # message asserting a defect that no longer exists.
    #
    # This guard now tests the ACTUAL invariant: for every vintage whose
    # remaining service span exceeds its technical life (build variable
    # non-zero), at least one rebuild year is generated -- i.e.
    # replacement capex IS charged somewhere in the objective.
    # _rebuild_years() is a closure local to build_model()/
    # build_model_sliced() and not reachable from here, so its arithmetic
    # is REPLICATED inline below. Keep this in sync with all four copies
    # if that helper's arithmetic ever changes: optimize_model.py:1137
    # (pre_f1_overnight), optimize_model.py:1331 (f1_tranche),
    # optimize_model_sliced.py:1366 (pre_f1_overnight),
    # optimize_model_sliced.py:1552 (f1_tranche) -- all four are
    # byte-identical today (docs/d6_storage_guard.md, step 0).
    _rebuild_invariant_violations = []
    try:
        _lives = scenario.get("asset_lifetimes", {})
        _T = list(range(len(scenario["years"])))
        _T_last = _T[-1]
        _checks = [
            ("storage", _lives.get("storage"), lambda t: pyo.value(model.storage_add[t])),
            ("solar",   _lives.get("solar"),
             lambda t: pyo.value(model.solar_public_add[t]) + pyo.value(model.solar_eaas_add[t])),
            ("gas",     _lives.get("gas"),   lambda t: pyo.value(model.gas_add[t])),
        ]
        for _name, _life, _getter in _checks:
            if not _life:
                continue
            for t in _T:
                if (_T_last - t + 1) > float(_life) and _getter(t) > 1e-6:
                    # Replicated from _rebuild_years(build_t, life).
                    _life_int = int(round(float(_life)))
                    _rebuild_ts = []
                    _k = 1
                    while t + _k * _life_int <= _T_last:
                        _rebuild_ts.append(t + _k * _life_int)
                        _k += 1
                    if not _rebuild_ts:
                        # The real D6 defect, actually occurring: a
                        # vintage retires in-horizon with NO compensating
                        # rebuild. Collected and raised after the
                        # try/except below so it is never swallowed by
                        # the diagnostic's own catch-all.
                        _rebuild_invariant_violations.append(
                            f"{_name} vintage {scenario['years'][t]} retires "
                            f"within the horizon (life {_life:.0f} yr < "
                            f"{_T_last - t + 1} yr served) but no in-horizon "
                            f"rebuild year exists -- replacement capex is NOT "
                            f"charged for it."
                        )
                    else:
                        _rebuild_years_str = ", ".join(
                            str(scenario["years"][rt]) for rt in _rebuild_ts
                        )
                        print(
                            f"  NOTE [plan 2.5]: {_name} vintage {scenario['years'][t]} "
                            f"retires within the horizon (life {_life:.0f} yr < "
                            f"{_T_last - t + 1} yr served) and is rebuilt at "
                            f"{_rebuild_years_str}, with replacement capex charged "
                            f"there (m.cost_replacement_capex_npv under "
                            f"pre_f1_overnight / m.cost_rebuild_capital_charge_npv "
                            f"under f1_tranche)."
                        )
                    break
    except AttributeError as _e:
        print(f"  NOTE [plan 2.5]: retirement check skipped -- scenario not passed to "
              f"solve_model, or model lacks an expected variable ({_e})")
    except Exception as _e:
        print(f"  WARNING [plan 2.5]: retirement check FAILED unexpectedly ({_e}). "
              f"This is a bug, not a skip.")

    if _rebuild_invariant_violations:
        raise RuntimeError(
            "D6 rebuild invariant violated -- replacement capex is not "
            "charged for: " + "; ".join(_rebuild_invariant_violations) +
            " This is a real defect, not a stale message; see "
            "docs/d6_storage_guard.md."
        )

    return {
        "status": str(tc),
        "optimal": tc == pyo.TerminationCondition.optimal,
        "solver_version": solver_version,
    }
