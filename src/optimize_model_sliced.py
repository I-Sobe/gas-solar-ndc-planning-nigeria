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
from src.blended_finance import resolve_remaining_npv_factor
from src.io import compute_om_cost_npv


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
# STILL REFERENCED -- do not delete without also checking scenarios.py.
# _normalised_demand_weights() below no longer reads this (direct
# subscript now, docs/task_hydro_gate_bucketA.md Part 2 Edit 4), but
# scenarios.py:2277 does: an arm whose own PEAK_*_SEASON_FACTOR is None
# (peak_nirp only, itself blocked) falls back to this constant when
# building scenario["season_demand_factor"]. DEFAULT_PERIOD_DEMAND_FACTOR
# had no such second reference and was removed; this one was kept for
# exactly that reason.
DEFAULT_SEASON_DEMAND_FACTOR = {"dry": 1.08, "wet": 0.92}

# Relative solar yield by season, applied to day slices only.
# SOURCED (Global Solar Atlas, retrieved 2026-09-05) -- full provenance
# at the canonical definition site, src/scenarios.py, the
# "season_solar_factor" scenario-dict entry (KEEP IN SYNC with it; this
# is a fallback for a hand-built scenario dict that omits the key --
# load_scenario() always sets it, so this default is not exercised in
# normal operation). Dry = clearer skies (offset by harmattan dust); wet
# = monsoon cloud.
DEFAULT_SEASON_SOLAR_FACTOR = {"dry": 1.076, "wet": 0.925}

# Hydro seasonal energy split.
# SOURCED (two independent NERC quarterly datasets, both excluding
# Zungeru) -- full provenance at the canonical definition site,
# src/scenarios.py, the "hydro_season_factor" scenario-dict entry (KEEP
# IN SYNC with it; this is a fallback for a hand-built scenario dict that
# omits the key -- load_scenario() always sets it, so this default is
# not exercised in normal operation). The RETIRED value's SIGN was wrong
# (dry LOWER, 0.90/1.10): it assumed run-of-river hydrology, but Kainji
# and Jebba are reservoir schemes whose output follows demand, which
# peaks in the dry season. Correct direction: dry HIGHER.
#
# SEASONAL-SCISSOR CAUTION (merged_correction_plan_v2.md:320-324) --
# STILL RELEVANT, now consistent with the corrected direction: 2025
# quarterly hydro SHARES rise 29.9 -> 38.3% but absolute LEVELS are
# near-flat Q1-Q3; NERC attributes the Q4 jump to UTILISATION
# improvements at Zungeru/Shiroro/Dadin-Kowa/Jebba, not hydrology. That
# 2025-only series could not by itself establish "dry season = high
# hydro" (the commissioning ramp confounds it) -- the sourcing above uses
# the PRE-ZUNGERU 2023 dataset instead, precisely to avoid that
# confound, cross-checked against a 2025 ratio with Zungeru excluded.
#
# KNOWN OVERLAP, DECLARED (correction plan 1.5): under hydro_committed /
# hydro_ndc (scenarios.hydro_trajectory()) the growth trajectory and this
# seasonal factor are BOTH modelling the same underlying commissioning
# programme -- the growth trajectory as a rising annual total, the
# seasonal factor as a Q4-weighted intra-year shape. Sourcing the
# seasonal factor from the pre-Zungeru 2023 data (above) decouples it
# from that overlap for THIS number; capacity_credit["hydro"] (which
# uses this factor) is likewise now sourced (see src/scenarios.py).
DEFAULT_HYDRO_SEASON_FACTOR = {"dry": 1.045, "wet": 0.956}

# Minimum hydro output per slice as a fraction of its proportional share,
# representing environmental flow, irrigation and downstream obligations.
# Without a floor the LP concentrates all hydro into peak slices, which
# reservoir operating constraints do not permit.
# Classification: ASSUMED, not SOURCED. Full documented search lives at
# the canonical site, src/scenarios.py's "hydro_min_flow_fraction"
# scenario-dict entry -- one canonical statement, not duplicated here.
DEFAULT_HYDRO_MIN_FLOW_FRACTION = 0.50


def _normalised_demand_weights(scenario):
    """
    Fraction of annual demand energy falling in each slice.

    Built from relative average power by season and period, weighted by slice
    hours, then normalised so the fractions sum to 1.
    """
    h = slice_hours()
    # Direct subscript, not .get(key, DEFAULT): load_scenario() always
    # sets both keys (docs/task_hydro_gate_bucketA.md, Part 1C), and the
    # retired DEFAULT_* fallbacks above (module top) held the UNSOURCED
    # pre-2.6 values -- a silent substitution converted into a loud
    # KeyError on any future caller that omits the key, rather than a
    # wrong number.
    season_f = scenario["season_demand_factor"]
    period_f = scenario["period_demand_factor"]

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
    emissions_cap=None,
    emissions_cap_by_year=None,
    max_offgrid_fraction=None,
    grid_service_mode="annual",
    reliability_max_unserved_fraction=None,  # DEPRECATED alias for max_offgrid_fraction
    reliability_mode=None,                    # DEPRECATED alias for grid_service_mode
    selfgen_in_emissions_cap=False,
    fixed_plan=None,
    solar_capex_by_year=None,
    allow_unserved=None,
):
    """
    Build a time-sliced planning optimisation model.

    Signature matches build_model() so runners can swap with one import change.
    """
    # ------------------------------------------------------------------
    # Grid service obligation (renamed from "reliability" in 2.6 F0).
    #
    # This constraint bounds (selfgen + unserved) -- the GRID SERVICE
    # SHORTFALL -- not unserved alone. A household running its own genset was
    # not served by the grid, but no capacity was short, so this is a service
    # obligation, NOT an adequacy/reliability standard: LOLE and EENS do not
    # describe it. The old names are retained as deprecated aliases so the
    # pre-2.6 REL and VoLL runners keep working unchanged.
    #
    # task_audit_fixes_p1 (audit finding A15): this translation was present
    # in build_model() (optimize_model.py) but silently absent here -- the
    # two DEPRECATED aliases above were accepted into the signature and
    # never read again anywhere in this function. Ported verbatim from
    # optimize_model.py's own working block (same names, same guard, same
    # error text) rather than reimplemented, so the two builders cannot
    # drift apart on this translation a second time.
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
    S = SLICES
    H = slice_hours()

    m.slice_hours = dict(H)   # plain dict, for diagnostics

    # ============================================================
    # Firm capacity / genset guard (2.6, F0 Step 2 -- phase_2_4_inversion)
    # ============================================================
    # AT THE TOP, before any other scenario/econ processing. Mirrors
    # optimize_model.py exactly -- see that file for the full rationale
    # (22 of 33 runner scripts already get the early, friendlier failure
    # inside load_scenario(); this is the guard that actually closes the
    # bypass for a hand-built scenario/econ dict, e.g. tests/test_run.py's
    # hand-built econ today).
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
            "load_scenario() from peak_multiple_case (Step 2.4); "
            "build_model_sliced() will not silently assume a peak-demand "
            "trajectory."
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
    # genset_case: checked on ECON, not scenario -- see optimize_model.py
    # for the full rationale on why key PRESENCE (not truthiness) is the
    # right check.
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
    df = {t: 1.0 / ((1.0 + r_social) ** t) for t in T}
    # bank_tenor_case (rate-coherence task, Part 1): resolves to the
    # commercial-basis factor by default, the pre-correction blended-WACC
    # factor under 'bank_tenor_legacy' -- see resolve_remaining_npv_factor()
    # docstring and docs/task_log_rate_coherence.md. Called here, before any
    # capital_treatment_case branch, so it is built identically under both
    # treatments -- mirrors optimize_model.py exactly.
    remaining_npv_priv = resolve_remaining_npv_factor(scenario, T)
    m.remaining_npv_factor = pyo.Param(T, initialize=remaining_npv_priv,
                                       within=pyo.PositiveReals)
    m.DF = pyo.Param(T, initialize=df, within=pyo.PositiveReals)

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
    # Storage CAPEX trajectory (TASK 52 section 2.1). 'flat' = the 2025
    # Storage_low scalar at every vintage and every rebuild (all results
    # before this arm); 'declining_nrel' = the Storage_low series, indexed by
    # the year of the build or rebuild exactly as solar_capex_param is.
    # scenario["storage_capex_case"] is read with NO fallback.
    # --------------------------
    _sc_case = scenario["storage_capex_case"]
    if _sc_case == "flat":
        _sc_init = {t: float(econ["STORAGE_COST_PER_MWH"]) for t in T}
    elif _sc_case == "declining_nrel":
        _sc_series = scenario["storage_capex_by_year"]
        _sc_init = {t: float(_sc_series[int(years[t])]) for t in T}
    else:
        raise ValueError(f"Unknown storage_capex_case: {_sc_case!r}")
    m.storage_capex_param = pyo.Param(T, initialize=_sc_init,
                                      within=pyo.PositiveReals)

    # --------------------------
    # Scalar parameters
    # --------------------------
    eta = scenario.get("gas_eta", None)
    if eta is None:
        raise ValueError("gas_eta is unset. Provide the measured fleet efficiency.")

    storage_round_trip_eff = scenario.get("storage_round_trip_eff", 0.90)
    storage_duration_hours = scenario["storage_duration_hours"]
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

    # ------------------------------------------------------------
    # OCGT-to-CCGT conversion (Phase 2.8)
    # ------------------------------------------------------------
    # conversion_available gates the ENTIRE mechanism. When False
    # (conv_none, or a hand-built scenario that omits the key) this
    # builder is byte-identical to pre-2.8: no conversion Var, and the
    # gas capacity ceiling / gas fuel budget / electricity balance /
    # gas OPEX / grid emissions expressions are their original form.
    _conv = bool(scenario.get("conversion_available", False))
    if _conv:
        # Mirror the load_scenario() cross-key guard for a hand-built
        # scenario that sets conversion_available directly. Conversion
        # capex is wired into the f1_tranche capital charge only; the
        # pre_f1_overnight branch is not modified (Phase 2.8, Do NOT).
        if capital_treatment_case == "pre_f1_overnight":
            raise ValueError(
                "conversion_available=True requires "
                "capital_treatment_case='f1_tranche' (Phase 2.8): "
                "conversion capex enters the F1 per-tranche capital "
                "charge only. Pass conv_none with pre_f1_overnight."
            )
        _conv_tiers = scenario["conversion_tiers"]        # {"A"/"B"/"C": {...}}
        _CONV_KEYS = ("A", "B", "C")
        _conv_uplift = float(scenario["conversion_uplift"])
        _eta_retrofit = float(scenario["conversion_retrofit_efficiency"])
        _conv_capex_per_kw = float(scenario["conversion_capex_per_kw"])
        if not (0.0 < _eta_retrofit <= 1.0):
            raise ValueError("conversion_retrofit_efficiency must be in (0, 1]")
        for _tk in _CONV_KEYS:
            if _tk not in _conv_tiers:
                raise ValueError(f"conversion_tiers is missing tier '{_tk}'")

    # ------------------------------------------------------------
    # Private self-generation backstop (correction plan 2.6)
    # ------------------------------------------------------------
    # NOTE: unlike the annual builder, this module has no local `voll`;
    # UNSERVED_ENERGY_PENALTY is used inline in the objective. Define it here
    # so the commit-A fallback below has something to fall back to.
    voll = econ["UNSERVED_ENERGY_PENALTY"]

    # Fallback: price selfgen at VoLL with zero EF. That makes it an exact
    # duplicate of unserved and leaves the objective unchanged -- the commit-A
    # regression test. Pass genset_case to load_econ() to activate real prices.
    selfgen_price = econ.get("GENSET_COST_PER_TWH_E", voll)
    if "GENSET_ETA_HHV" in econ:
        # EF_e DERIVED, never stored (same rule as the gas fleet, 1.2).
        selfgen_ef_e = econ["GENSET_EF_TH_HHV"] / econ["GENSET_ETA_HHV"]
    else:
        selfgen_ef_e = 0.0

    # Busbar-equivalence factor. The balance is metered at busbar; a genset
    # delivers at the point of use, downstream of the transmission loss factor.
    # DEFAULT 1.0 is a declared conservative simplification, not an oversight.
    # Set scenario["selfgen_busbar_factor"] = 1/(1-0.07) to credit the avoided
    # TLF. Whichever you choose, state it in the methods chapter.
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
    # New gas plant construction (2.9, Step 1): gas_add is now a free
    # decision variable, gated by gas_build_case (2.9, Step 2) -- see below,
    # after gas_capacity_mw is defined, for how each arm constrains it.
    m.gas_add = pyo.Var(T, domain=pyo.NonNegativeReals)
    gas_build_case = scenario["gas_build_case"]
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
    # See the annual model for the rationale. Sliced indexing matters here:
    # the backstop is available in EVERY slice, so it competes directly with
    # storage in (dry, night) and (dry, peak) and caps storage's marginal value
    # at genset cost. Storage build is EXPECTED TO FALL. That is a verification
    # test, and a declared threat to the "NIRP under-provisions storage" claim.
    m.selfgen = pyo.Var(T, m.S, domain=pyo.NonNegativeReals)
    if not allow_unserved:
        for t in T:
            for s in m.S:
                m.unserved[t, s].fix(0.0)

    m.gas_to_power = pyo.Var(T, m.S, domain=pyo.NonNegativeReals)
    # TASK 66 / TASK 65 Part B (register L123): OPTIONAL vintage-differentiated gas dispatch, a
    # sensitivity, OFF by default. When scenario["gas_vintage_dispatch"] is true, m.gas_to_power
    # stays the TOTAL thermal draw (so the fuel balance, the fuel cost and the thermal diagnostics
    # are untouched) and m.gas_to_power_new is the part of it burned by capacity ADDED in the model,
    # which converts at scenario["gas_new_efficiency"]; the rest converts at the blended gas_eta.
    # With the flag off nothing below this comment is built or changed.
    gas_vintage_dispatch = bool(scenario.get("gas_vintage_dispatch", False))
    if gas_vintage_dispatch:
        if scenario.get("new_gas_deliverability", "constrained") != "constrained":
            raise ValueError("gas_vintage_dispatch is validated at new_gas_deliverability='constrained' only.")
        gas_new_eff_vintage = float(scenario["gas_new_efficiency"])
        m.gas_to_power_new = pyo.Var(T, m.S, domain=pyo.NonNegativeReals)
    m.gas_curtail = pyo.Var(T, domain=pyo.NonNegativeReals)
    m.storage_discharge = pyo.Var(T, m.S, domain=pyo.NonNegativeReals)
    m.storage_charge = pyo.Var(T, m.S, domain=pyo.NonNegativeReals)
    m.hydro_gen = pyo.Var(T, m.S, domain=pyo.NonNegativeReals)

    # ------------------------------------------------------------
    # Conversion decision + converted-gas dispatch stream (Phase 2.8)
    # ------------------------------------------------------------
    # convert_mw[tier]: MW of OCGT NAMEPLATE converted, per tier. ONE-OFF
    # (time-invariant -- Decision 5), each bounded above by its tier's
    # nameplate. convert_to_power[tier, t, (s,p)]: SLICE-RESOLVED thermal
    # dispatch of the converted plant (Decision 3 -- 3 x 30 x 6 = 540
    # vars), so converted plant participates in the per-slice electricity
    # balance and can serve the dry-season evening peak like the rest of
    # gas. Parallel to m.gas_to_power (Decision 4), NOT a replacement.
    if _conv:
        m.convert_mw = pyo.Var(
            _CONV_KEYS,
            domain=pyo.NonNegativeReals,
            bounds=lambda m, tier: (0.0, _conv_tiers[tier]["nameplate_mw"]),
        )
        m.convert_to_power = pyo.Var(
            _CONV_KEYS, T, m.S, domain=pyo.NonNegativeReals
        )   # thermal, TWh_th

    # ------------------------------------------------------------
    # Capacity expressions (annual)
    # ------------------------------------------------------------
    gas_baseline = scenario["gas_baseline_mw"]
    retirement_start = scenario.get("gas_baseline_retirement_start_year", None)
    retirement_rate = scenario.get("gas_baseline_retirement_mw_per_year", 0.0)
    # gas_baseline_retirement_series_mw (2.12, Step 1): an optional
    # per-model-year CUMULATIVE retired-MW list (e.g. from
    # scenarios.gas_vintage_retirement_series_mw()), for retirement
    # schedules too lumpy for the flat rate * (year - start + 1) formula
    # below to represent (see docs/phase_2_12_vintage_retirement.md, Step
    # 0). None for gas_retire_none/gas_retire_proxy, which are unchanged
    # and fall through to the flat formula exactly as in 2.11. Mirrors
    # optimize_model.py exactly.
    retirement_series = scenario.get("gas_baseline_retirement_series_mw", None)
    if retirement_series is not None and len(retirement_series) != len(T):
        raise ValueError(
            f"scenario['gas_baseline_retirement_series_mw'] has "
            f"{len(retirement_series)} entries for {len(T)} model years. "
            "Refusing to build with a misaligned retirement series."
        )

    def gas_cap_rule(m, t):
        year_t = int(years[t])
        if retirement_series is not None:
            retired = min(retirement_series[t], gas_baseline)
        elif retirement_start is not None and year_t >= retirement_start:
            retired = min(retirement_rate * (year_t - retirement_start + 1),
                          gas_baseline)
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

    # Correction plan 1.5: the hydro energy series and the matching
    # installed-capacity series are REQUIRED per-year trajectories, set by
    # load_scenario() from the hydro_case arm. Under hydro_flat every entry
    # is 11.47 / 2,678 (byte-identical to the pre-1.5
    # hydro_baseline_twh * (1+0)^t formula and the scalar hydro_capacity_mw).
    hydro_energy_twh_by_year = scenario.get("hydro_energy_twh_by_year", None)
    if hydro_energy_twh_by_year is None:
        raise ValueError(
            "hydro_energy_twh_by_year is unset (correction plan 1.5). It is "
            "set by load_scenario() from the REQUIRED hydro_case arm -- see "
            "scenarios.hydro_trajectory(). A hand-built scenario must provide "
            "the annual hydro energy series (TWh_e/year)."
        )
    hydro_capacity_mw_by_year = scenario.get("hydro_capacity_mw_by_year", None)
    if hydro_capacity_mw_by_year is None:
        raise ValueError(
            "hydro_capacity_mw_by_year is unset (correction plan 1.5). The "
            "sliced model dispatches hydro within a seasonal energy budget "
            "subject to a power limit and needs the per-year installed-MW "
            "series (moving with the energy series so capacity_credit['hydro'] "
            "stays coherent). Set by load_scenario() from hydro_case."
        )
    if len(hydro_energy_twh_by_year) != len(list(T)) or \
            len(hydro_capacity_mw_by_year) != len(list(T)):
        raise ValueError("hydro_*_by_year length mismatch with model years.")
    hydro_annual = {t: float(hydro_energy_twh_by_year[t]) for t in T}

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
    if gas_vintage_dispatch:
        # total thermal draw x blended eta, plus the uplift on the added-capacity stream
        m.gas_generation = pyo.Expression(
            T, m.S,
            rule=lambda m, t, s, p: eta * m.gas_to_power[t, (s, p)]
            + (gas_new_eff_vintage - eta) * m.gas_to_power_new[t, (s, p)]
        )
    else:
        m.gas_generation = pyo.Expression(
            T, m.S, rule=lambda m, t, s, p: eta * m.gas_to_power[t, (s, p)]
        )

    # Converted plant generates at eta_retrofit (Decision 3 -- the second
    # coexisting gas efficiency). Same slice indexing as m.gas_generation.
    if _conv:
        m.convert_generation = pyo.Expression(
            _CONV_KEYS, T, m.S,
            rule=lambda m, tier, t, s, p:
                _eta_retrofit * m.convert_to_power[tier, t, (s, p)]
        )

        def _converted_thermal_year(m, t):
            """Total converted thermal draw in year t (TWh_th), all tiers/slices."""
            return pyo.quicksum(
                m.convert_to_power[tier, t, (s, p)]
                for tier in _CONV_KEYS for (s, p) in S
            )

        def _converted_gen_year(m, t):
            """Total converted electrical generation in year t (TWh_e)."""
            return pyo.quicksum(
                m.convert_generation[tier, t, (s, p)]
                for tier in _CONV_KEYS for (s, p) in S
            )

    m.solar_generation = pyo.Expression(
        T, m.S,
        rule=lambda m, t, s, p: m.solar_capacity_mw[t] * solar_per_mw[(s, p)]
    )

    # ------------------------------------------------------------
    # Gas: per-slice POWER limit, annual FUEL budget
    # ------------------------------------------------------------
    # Capacity limits electrical output within each slice's hours.
    if _conv:
        # Phase 2.8: converted OCGT nameplate LEAVES the fleet-scalar
        # bucket -- those plants now run combined-cycle at eta_retrofit and
        # tier PAF (m.convert_capacity_constraint below), not at the fleet
        # eta and gas_availability_factor. Netting it out of the RHS here
        # (not out of m.gas_capacity_mw itself, which still feeds land /
        # firm capacity unchanged) is the Decision-4 amendment. convert_mw
        # is bounded by tier nameplate (total <= 6,482 MW); if exogenous
        # retirement plus conversion ever exceed gas_baseline the RHS goes
        # negative and the model returns INFEASIBLE -- a loud failure, not
        # a silent wrong answer (a plant cannot be both retired and
        # converted; the model does not currently prevent that overlap).
        _converted_ocgt_nameplate = pyo.quicksum(
            m.convert_mw[tier] for tier in _CONV_KEYS
        )
        m.gas_capacity_constraint = pyo.Constraint(
            T, m.S,
            rule=lambda m, t, s, p:
                m.gas_generation[t, (s, p)]
                <= (m.gas_capacity_mw[t] - _converted_ocgt_nameplate)
                   * H[(s, p)] * gas_availability_factor / 1e6
        )
        # Converted energy ceiling: available converted capacity is
        # (1 + uplift) * convert_mw[tier] -- the original OCGT plus the
        # bottoming cycle (Decision 1) -- running at the TIER's PAF, NOT
        # the fleet scalar (Decision 4: a blended availability is forbidden).
        m.convert_capacity_constraint = pyo.Constraint(
            _CONV_KEYS, T, m.S,
            rule=lambda m, tier, t, s, p:
                m.convert_generation[tier, t, (s, p)]
                <= (1.0 + _conv_uplift) * m.convert_mw[tier]
                   * H[(s, p)] * _conv_tiers[tier]["paf"] / 1e6
        )
    else:
        m.gas_capacity_constraint = pyo.Constraint(
            T, m.S,
            rule=lambda m, t, s, p:
                m.gas_generation[t, (s, p)]
                <= m.gas_capacity_mw[t] * H[(s, p)] * gas_availability_factor / 1e6
        )

    if gas_vintage_dispatch:
        # Capacity accounting (TASK 66, tightened TASK 74 D.2). The total-capacity row above (which nets
        # converted OCGT nameplate under conversion, and can be met by added capacity) bounds total
        # electrical output. Three vintage rows sit on top of it: (i) the added stream cannot exceed the
        # total thermal draw; (ii) the added stream cannot run beyond the capacity added in the model;
        # (iii) [TASK 74] the LEGACY stream, eta * (draw - added stream), cannot exceed the surviving legacy
        # capacity (gas_baseline minus retirements, i.e. m.gas_capacity_mw less cumulative additions) at the
        # fleet availability. Row (iii) removes the TASK 66 relaxation in which idle added capacity could
        # carry legacy-efficiency output beyond the legacy fleet. It deliberately does NOT net converted
        # nameplate: a strict legacy row (surviving legacy capacity minus converted nameplate) was tried in
        # TASK 66 and rejected, because converted nameplate (3,786 MW at the central arm) exceeds surviving
        # legacy capacity (3,666 MW in 2054), an overlap the baseline's pooled row tolerates (the model does
        # not retire converted plants with their vintage) and a strict row forbids. Row (iii) is implied
        # by the baseline's feasible set whenever the total row holds, so it collapses at
        # gas_new_efficiency == gas_eta (demonstrated: task_log_74 D.2, D.3).
        m.gas_vintage_new_capacity_mw = pyo.Expression(
            T, rule=lambda m, t: sum(m.gas_add[k] for k in range(0, t + 1))
        )
        m.gas_new_stream_within_total = pyo.Constraint(
            T, m.S,
            rule=lambda m, t, s, p: m.gas_to_power_new[t, (s, p)] <= m.gas_to_power[t, (s, p)]
        )
        m.gas_new_stream_capacity = pyo.Constraint(
            T, m.S,
            rule=lambda m, t, s, p:
                gas_new_eff_vintage * m.gas_to_power_new[t, (s, p)]
                <= m.gas_vintage_new_capacity_mw[t] * H[(s, p)] * gas_availability_factor / 1e6
        )
        m.gas_legacy_stream_capacity = pyo.Constraint(
            T, m.S,
            rule=lambda m, t, s, p:
                eta * (m.gas_to_power[t, (s, p)] - m.gas_to_power_new[t, (s, p)])
                <= (m.gas_capacity_mw[t] - m.gas_vintage_new_capacity_mw[t]) * H[(s, p)] * gas_availability_factor / 1e6
        )

    # Fuel is an ANNUAL budget the optimiser allocates across slices. This is
    # what lets scarce gas be saved for the dry-season evening peak.
    #
    # new_gas_deliverability (2.9, Step 3): 'constrained' (default) leaves
    # this exactly as before -- the whole fleet, including new build, draws
    # on the SAME shared fuel budget. 'exempt' assumes new capacity is
    # sited at gas supply or on dedicated commercial contracts: it adds a
    # fuel-equivalent allowance, sized to new capacity's OWN
    # capacity-and-efficiency ceiling, to the shared budget -- so new
    # capacity is never fuel-constrained by the existing fleet's scarcity,
    # without introducing a second, vintage-tagged dispatch variable.
    # gas_generation, emissions, and dispatch OPEX all remain on the single
    # shared m.gas_to_power and the blended fleet eta (2.9 does not split
    # dispatch by vintage -- see docs/phase_2_9_gas_addition.md).
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
        if _conv:
            m.gas_balance = pyo.Constraint(
                T,
                rule=lambda m, t:
                    pyo.quicksum(m.gas_to_power[t, (s, p)] for (s, p) in S)
                    + _converted_thermal_year(m, t)
                    + m.gas_curtail[t]
                    <= gas_avail[t] + m.gas_exempt_fuel_allowance_twh_th[t]
            )
        else:
            m.gas_balance = pyo.Constraint(
                T,
                rule=lambda m, t:
                    pyo.quicksum(m.gas_to_power[t, (s, p)] for (s, p) in S)
                    + m.gas_curtail[t]
                    <= gas_avail[t] + m.gas_exempt_fuel_allowance_twh_th[t]
            )
    else:
        if _conv:
            # Phase 2.8: converted plant draws the SAME scarce fuel
            # envelope as the unconverted fleet (Decision 4).
            m.gas_balance = pyo.Constraint(
                T,
                rule=lambda m, t:
                    pyo.quicksum(m.gas_to_power[t, (s, p)] for (s, p) in S)
                    + _converted_thermal_year(m, t)
                    + m.gas_curtail[t] <= gas_avail[t]
            )
        else:
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

    # Power limit per slice. Correction plan 1.5: hydro_capacity_mw is now
    # a per-year series that moves with the energy series, so a growth
    # trajectory does not make the dry-season budget exceed what the
    # slices can physically pass.
    m.hydro_power_limit = pyo.Constraint(
        T, m.S,
        rule=lambda m, t, s, p:
            m.hydro_gen[t, (s, p)]
            <= hydro_capacity_mw_by_year[t] * H[(s, p)] / 1e6
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
    # Phase 2.8: converted plant serves load in the same slice it
    # generates -- its output MUST enter the per-slice balance, or the
    # LP would burn fuel and emit CO2 for generation that meets no
    # demand (the mechanism would be inert and never chosen). This is a
    # fourth amendment beyond Decision 4's named three; without it the
    # feature does not function. See docs/phase_2_8_conversion.md.
    if _conv:
        m.balance = pyo.Constraint(
            T, m.S,
            rule=lambda m, t, s, p:
                m.gas_generation[t, (s, p)]
                + pyo.quicksum(m.convert_generation[tier, t, (s, p)]
                               for tier in _CONV_KEYS)
                + m.solar_generation[t, (s, p)]
                + m.hydro_gen[t, (s, p)]
                + m.storage_discharge[t, (s, p)]
                - m.storage_charge[t, (s, p)]
                + m.selfgen[t, (s, p)] * selfgen_busbar_factor
                + m.unserved[t, (s, p)]
                >= demand[(t, s, p)]
        )
    else:
        m.balance = pyo.Constraint(
            T, m.S,
            rule=lambda m, t, s, p:
                m.gas_generation[t, (s, p)]
                + m.solar_generation[t, (s, p)]
                + m.hydro_gen[t, (s, p)]
                + m.storage_discharge[t, (s, p)]
                - m.storage_charge[t, (s, p)]
                + m.selfgen[t, (s, p)] * selfgen_busbar_factor
                + m.unserved[t, (s, p)]
                >= demand[(t, s, p)]
        )

    # ------------------------------------------------------------
    # Financial constraints (ANNUAL -- unchanged from the annual model)
    # ------------------------------------------------------------
    solar_energy_per_mw_annual = scenario["solar_cf"] * 8760 / 1_000_000

    if financing_regime == "eaas" and tariff is None:
        raise ValueError("Tariff must be defined under EaaS regime.")

    # F1, Step 5/6: the four overnight capex terms (public solar here;
    # EaaS solar right after; gas and storage further down, alongside
    # salvage/replacement) are built ONLY under capital_treatment_case==
    # 'pre_f1_overnight' -- the live objective must contain EXACTLY ONE of
    # {F1, salvage, replacement} (system_cost_npv assembly, below). Under
    # 'f1_tranche' these Pyomo components do not exist on the model at
    # all -- see the F1 tranche mechanism after the replacement-capex
    # block for the 'f1_tranche' equivalents. Mirrors optimize_model.py.
    if capital_treatment_case == "pre_f1_overnight":
        public_solar_capex_npv = pyo.quicksum(
            m.DF[t] * m.solar_public_add[t] * m.solar_capex_param[t] for t in T
        )
        # EaaS (private) solar CAPEX.
        # MODELLING CHOICE (state explicitly in the thesis): EaaS-financed and
        # publicly-financed solar are priced at the SAME unit CAPEX -- they are
        # the same physical assets. The private cost of capital is represented
        # NOT by an objective-side CAPEX premium but by the bankability
        # constraint, which discounts EaaS revenue at the private/blended rate
        # via remaining_npv_factor. Applying a required_margin multiplier here
        # would double-count: required_margin is the bankability coverage
        # cushion, a distinct object from the cost of capital. The EaaS
        # mechanism's effect in this model is therefore structural (budget
        # decoupling + a harder bankability test), not a cheaper sticker
        # price. (F1, Step 5: written here to match optimize_model.py:628-637,
        # which had this comment and this file did not -- same rationale
        # applies verbatim; F1's per-tranche split, when active, replaces
        # this whole overnight-capex term, not just the margin question.)
        eaas_solar_capex_npv = pyo.quicksum(
            m.DF[t] * m.solar_eaas_add[t] * m.solar_capex_param[t] for t in T
        )

    # CONCESSIONAL CAPITAL ENVELOPE (scarcity constraint) -- RETIRED.
    # The pre-existing NPV-discounted, EaaS-solar-only mechanism
    # (concessional_envelope_npv scenario key, m.concessional_drawdown_npv,
    # m.concessional_envelope_constraint) is retired per envelope
    # resolution Decision 2 (docs/f1_implementation.md, "Envelope
    # resolution"): one envelope, one quantity. F1's OWN concessional
    # envelope -- undiscounted commitment stock, spanning solar
    # (public + EaaS) and storage, gas excluded -- replaces it; see the
    # 'f1_tranche' branch below (m.f1_concessional_envelope_constraint).

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
    if max_offgrid_fraction is not None:
        # 2.6, D9: accept either a scalar (flat obligation) or a per-year
        # sequence (obligation trajectory), matching optimize_model.py.
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
            m.grid_service_obligation = pyo.Constraint(
                T,
                rule=lambda m, t:
                    pyo.quicksum(
                        m.selfgen[t, (s, p)] + m.unserved[t, (s, p)]
                        for (s, p) in S
                    )
                    <= _eps[t] * demand_annual[t]
            )
        elif grid_service_mode == "total":
            m.grid_service_obligation = pyo.Constraint(
                expr=pyo.quicksum(
                    m.selfgen[t, (s, p)] + m.unserved[t, (s, p)]
                    for t in T for (s, p) in S
                )
                <= sum(_eps[t] * demand_annual[t] for t in T)
            )
        else:
            raise ValueError("grid_service_mode must be 'annual' or 'total'")

        # m.reliability_constraint = m.grid_service_obligation

    # ------------------------------------------------------------
    # Firm Capacity Constraint (2.6, F0 Step 2.3)
    # ------------------------------------------------------------
    # Sigma_tech capacity_credit[tech] x cumulative_capacity[tech,t]
    #   >= peak_mw[t] * within_block_peaking_factor * (1 + reserve_margin[t])
    #
    # Mirrors optimize_model.py exactly. Unlike grid_service_obligation (a
    # quantity-based bound on selfgen+unserved), this tests whether CREDITED
    # installed capacity covers a peak-demand forecast with a reserve margin,
    # independent of dispatch. No such constraint existed anywhere in this
    # file before 2.6, F0 (see docs/F0_audit.md, Step B6).
    #
    # ANNUAL, like all other capacity/financial constraints in this builder
    # (see the module docstring): capacity is built once a year, not per
    # slice, so this is not sliced either. Storage's credited capacity is
    # its POWER rating (m.storage_power_mw, MW), not its energy rating
    # (m.storage_capacity_mwh, MWh).
    #
    # reserve_margin: flat scalar or per-year sequence (same dual-form
    # pattern already used for max_offgrid_fraction above). 'nirp_spinning'
    # mode (load_scenario()) resolves to a per-year list before it reaches
    # here.
    if isinstance(reserve_margin, (list, tuple)):
        if len(reserve_margin) != len(T):
            raise ValueError(
                f"scenario['reserve_margin'] has {len(reserve_margin)} "
                f"entries but the horizon has {len(T)} years."
            )
        _reserve_margin_by_year = {t: float(reserve_margin[t]) for t in T}
    else:
        _reserve_margin_by_year = {t: float(reserve_margin) for t in T}

    # within_block_peaking_factor (2.6, F0 Step 2.3): ratio of the
    # INSTANTANEOUS system peak to par_block's 4-hour BLOCK-AVERAGE peak --
    # see scenarios.py, the module comment above peak_multiple_scenarios().
    # SOURCE: [PENDING]; DEFAULT 1.0 is a declared conservative
    # simplification, not an oversight -- same pattern as
    # selfgen_busbar_factor above. DELIBERATELY SEPARATE from reserve_margin
    # -- do not merge the two.
    within_block_peaking_factor = float(
        scenario.get("within_block_peaking_factor", 1.3448)
    )
    if within_block_peaking_factor < 1.0:
        raise ValueError(
            "within_block_peaking_factor < 1.0 would claim the instantaneous "
            "peak is LOWER than its own 4-hour block average, which is "
            "backwards."
        )

    if _conv:
        # Phase 2.8 addendum -- firm-capacity credit for converted capacity.
        # The uplift * convert_mw[tier] bottoming-cycle capacity must enter
        # the firm-capacity constraint; without it conversion delivers the
        # efficiency benefit only, which contradicts Decision 1 and voids
        # Decision 3's rationale. The OCGT nameplate convert_mw[tier] is
        # unchanged -- it still counts through m.gas_capacity_mw at
        # capacity_credit["gas"] (as does land); only the bottoming-cycle
        # uplift is credited here, at the tier rate.
        #
        # capacity_credit["gas"] and gas_availability_factor were assigned
        # from the same local variable in Phase 2.10 (both 0.80) -- the
        # fleet's firm-capacity credit is currently DEFINED as its
        # availability. Scaling by tier_PAF / fleet availability therefore
        # applies the model's own existing convention at tier resolution
        # rather than introducing a new assumption. Tier A 0.96, B 0.345,
        # C 0.106. Inherits the PENDING source status of
        # capacity_credit["gas"] (register L8, phase_2_10 line 103).
        _firm_credit = {
            tier: capacity_credit["gas"] * _conv_tiers[tier]["paf"]
                  / gas_availability_factor
            for tier in _CONV_KEYS
        }
        m.firm_capacity_constraint = pyo.Constraint(
            T,
            rule=lambda m, t:
                capacity_credit["solar"] * m.solar_capacity_mw[t]
                + capacity_credit["gas"] * m.gas_capacity_mw[t]
                + capacity_credit["storage"] * m.storage_power_mw[t]
                + capacity_credit["hydro"] * hydro_capacity_mw_by_year[t]
                + pyo.quicksum(
                    _firm_credit[tier] * _conv_uplift * m.convert_mw[tier]
                    for tier in _CONV_KEYS
                )
                >= peak_mw[t] * within_block_peaking_factor
                   * (1.0 + _reserve_margin_by_year[t])
        )
    else:
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

    # ------------------------------------------------------------
    # Costs
    # ------------------------------------------------------------
    # Fuel OPEX is unconditional -- untouched by F1 (F1 only replaces the
    # CAPEX/financing side).
    gas_opex_npv = pyo.quicksum(
        m.DF[t] * m.gas_to_power[t, (s, p)] * econ["GAS_COST_PER_TWH_TH"]
        for t in T for (s, p) in S
    )
    if _conv:
        # Phase 2.8: converted plant burns the same fuel at the same
        # thermal price. Omitting this term would give converted
        # generation free fuel and the LP would idle unconverted gas.
        # A fifth amendment beyond Decision 4's named three -- economically
        # mandatory, same category as the gas_balance amendment (both are
        # "the converted stream draws/pays fuel thermally"; see the step-0
        # inventory's (a) note). See docs/phase_2_8_conversion.md.
        gas_opex_npv = gas_opex_npv + pyo.quicksum(
            m.DF[t] * m.convert_to_power[tier, t, (s, p)]
            * econ["GAS_COST_PER_TWH_TH"]
            for tier in _CONV_KEYS for t in T for (s, p) in S
        )

    # ------------------------------------------------------------
    # Fixed + variable O&M (task_om_implementation, Part 3.3)
    # ------------------------------------------------------------
    # STORAGE_OM_PER_MWH_YR (io.py) never reached the objective for any
    # technology -- see docs/task_storage_om.md. om_cost_by_year is set by
    # load_scenario() from the REQUIRED om_case arm. Shared with
    # optimize_model.py (plan 2.5: "the two must share a cost definition
    # or the annual-vs-sliced premium compares different things") via
    # io.compute_om_cost_npv.
    #
    # Converted (gas_cc) capacity is _converted_ocgt_nameplate -- the
    # converted OCGT nameplate, time-invariant (Decision 5) -- and
    # converted generation is the electrical (TWh_e) sum _converted_gen_year
    # already defines. Unconverted (gas_ct) capacity is the remainder of
    # m.gas_capacity_mw (which still carries the full nameplate, converted
    # included, per the Decision-4 amendment above); unconverted
    # generation is m.gas_generation summed over slices. Under conv_none
    # (_conv is False) there is no conversion mechanism at all, so
    # everything is unconverted and the converted arguments are zero.
    om_unconverted_gas_gen_twh = {
        t: pyo.quicksum(m.gas_generation[t, (s, p)] for (s, p) in S)
        for t in T
    }
    if _conv:
        om_converted_gas_capacity_mw = {t: _converted_ocgt_nameplate for t in T}
        om_converted_gas_gen_twh = {t: _converted_gen_year(m, t) for t in T}
    else:
        om_converted_gas_capacity_mw = {t: 0.0 for t in T}
        om_converted_gas_gen_twh = {t: 0.0 for t in T}
    om_unconverted_gas_capacity_mw = {
        t: m.gas_capacity_mw[t] - om_converted_gas_capacity_mw[t] for t in T
    }

    om_cost_npv = compute_om_cost_npv(
        T=T,
        years=years,
        DF=m.DF,
        om_cost_by_year=scenario["om_cost_by_year"],
        storage_duration_hours=storage_duration_hours,
        solar_capacity_mw=m.solar_capacity_mw,
        storage_capacity_mwh=m.storage_capacity_mwh,
        gas_unconverted_capacity_mw=om_unconverted_gas_capacity_mw,
        gas_converted_capacity_mw=om_converted_gas_capacity_mw,
        gas_unconverted_generation_twh=om_unconverted_gas_gen_twh,
        gas_converted_generation_twh=om_converted_gas_gen_twh,
        # TASK 69 B.1 (register L95): hydro FOM term, OFF by default. With scenario["hydro_om_term"]
        # absent or false nothing is passed and the term does not exist (byte-identical to before).
        **(dict(hydro_capacity_mw=hydro_capacity_mw_by_year,
                hydro_fom_usd_per_kw_yr=float(scenario.get("hydro_fom_usd_per_kw_yr", 0.0)))
           if scenario.get("hydro_om_term", False) else {}),
    )

    # F1, Step 5/6: matching comment above public_solar_capex_npv -- gas/
    # storage overnight capex and the salvage/replacement mechanism are
    # built ONLY under 'pre_f1_overnight'. Mirrors optimize_model.py.
    if capital_treatment_case == "pre_f1_overnight":
        gas_capex_npv = pyo.quicksum(
            m.DF[t] * m.gas_add[t] * scenario["gas_capex_per_mw"] for t in T
        )
        storage_capex_npv = pyo.quicksum(
            m.DF[t] * m.storage_add[t] * m.storage_capex_param[t] for t in T
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
        # == 'f1_tranche'; Steps 2-5). Mirrors optimize_model.py exactly
        # -- capacity-build variables (solar_public_add, solar_eaas_add,
        # gas_add, storage_add) and their per-unit capex data are the
        # SAME objects/keys in both builders; only dispatch is sliced.
        # ============================================================
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
        # channel. Linearity invariant: the coefficient on each build
        # variable (unit capex x CRF x df_window) must be exogenous data
        # -- may vary over vintage/technology, may NOT depend on any
        # decision variable including cumulative deployment. Time-varying
        # capex trajectories (m.solar_capex_param) are fine; learning
        # curves, volume discounts, endogenous FX are not present here.
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

        # ---- Per-channel capital charge (NPV). NOT named
        # m.cost_<channel>_capex_npv (the pre-F1 names, built only under
        # 'pre_f1_overnight' above) -- deliberately renamed so a stale
        # reader (e.g. extract_planning_diagnostics, which reads the old
        # names unconditionally) fails loudly under 'f1_tranche' instead
        # of silently reading a number that no longer means overnight capex.
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

        # Phase 2.8: conversion capex on the SAME basis as gas_add
        # (Decision, step 3) -- commercial-only (gas is not
        # concessional-eligible, envelope resolution Decision 1),
        # _crf_comm["gas"] and _df_window_comm["gas"][0] (convert_mw is a
        # one-off decision assigned to vintage 0, Decision 5). NOT in
        # m.f1_concessional_envelope_constraint and NOT in
        # m.concessional_volume_by_vintage. Capex is on OCGT NAMEPLATE
        # converted (Decision 2): capex_per_kw * 1000 USD/MW * convert_mw.
        _capital_charge_total = (
            m.cost_public_solar_capital_charge_npv
            + m.cost_eaas_solar_capital_charge_npv
            + m.cost_gas_capital_charge_npv
            + m.cost_storage_capital_charge_npv
            + m.cost_rebuild_capital_charge_npv
        )
        if _conv:
            m.convert_comm_vol = pyo.Var(_CONV_KEYS, domain=pyo.NonNegativeReals)
            m.convert_split_identity = pyo.Constraint(
                _CONV_KEYS,
                rule=lambda m, tier:
                    m.convert_comm_vol[tier]
                    == _conv_capex_per_kw * 1000.0 * m.convert_mw[tier]
            )
            m.cost_conversion_capital_charge_npv = pyo.Expression(
                expr=pyo.quicksum(
                    _crf_comm["gas"] * _df_window_comm["gas"][0]
                    * m.convert_comm_vol[tier]
                    for tier in _CONV_KEYS
                )
            )
            _capital_charge_total = (
                _capital_charge_total + m.cost_conversion_capital_charge_npv
            )
        m.capital_charge_total_npv = pyo.Expression(expr=_capital_charge_total)

    unserved_npv = pyo.quicksum(
        m.DF[t] * m.unserved[t, (s, p)] * econ["UNSERVED_ENERGY_PENALTY"]
        for t in T for (s, p) in S
    )

    selfgen_npv = pyo.quicksum(
        m.DF[t] * m.selfgen[t, (s, p)] * selfgen_price
        for t in T for (s, p) in S
    )

    ef_e = econ["EF_TCO2_PER_MWH_TH"] / eta
    # Phase 2.8: converted plant emits at the retrofit efficiency -- a
    # more efficient plant emits LESS CO2 per MWh_e for the same fuel.
    if _conv:
        _ef_e_retrofit = econ["EF_TCO2_PER_MWH_TH"] / _eta_retrofit
    # --------------------------------------------------------------
    # THREE emissions quantities, deliberately distinct (2.6, D3).
    # Mirrors optimize_model.py. The earlier single-switch design
    # conflated what the NDC COMMITS with what is PHYSICALLY EMITTED;
    # the cap and the carbon price must attach to different ones.
    #
    #   emissions_grid_only_by_year  gas-fired grid generation = the NDC
    #                                commitment boundary (7.24% gas-power
    #                                share of the energy-sector target)
    #   emissions_selfgen_by_year    private diesel: in the national
    #                                inventory, outside that slice
    #   emissions_by_year            TOTAL. Physical truth. What a carbon
    #                                PRICE sees and what gets reported.
    # --------------------------------------------------------------
    if gas_vintage_dispatch:
        # TASK 66 / TASK 65: emissions follow the THERMAL draw, so the added-capacity stream emits
        # less per MWh_e because it converts more efficiently: tCO2 = TWh_th x 1e6 x tCO2/MWh_th.
        # At gas_new_efficiency == gas_eta this equals the generation-based expression below.
        _ef_th = econ["EF_TCO2_PER_MWH_TH"]
        if _conv:
            m.emissions_grid_only_by_year = pyo.Expression(
                T,
                rule=lambda m, t:
                    pyo.quicksum(m.gas_to_power[t, (s, p)] for (s, p) in S)
                    * 1e6 * _ef_th
                    + _converted_gen_year(m, t) * 1e6 * _ef_e_retrofit
            )
        else:
            m.emissions_grid_only_by_year = pyo.Expression(
                T,
                rule=lambda m, t: pyo.quicksum(
                    m.gas_to_power[t, (s, p)] for (s, p) in S
                ) * 1e6 * _ef_th
            )
    elif _conv:
        m.emissions_grid_only_by_year = pyo.Expression(
            T,
            rule=lambda m, t:
                pyo.quicksum(m.gas_generation[t, (s, p)] for (s, p) in S)
                * 1e6 * ef_e
                + _converted_gen_year(m, t) * 1e6 * _ef_e_retrofit
        )
    else:
        m.emissions_grid_only_by_year = pyo.Expression(
            T,
            rule=lambda m, t: pyo.quicksum(
                m.gas_generation[t, (s, p)] for (s, p) in S
            ) * 1e6 * ef_e
        )
    m.emissions_selfgen_by_year = pyo.Expression(
        T,
        rule=lambda m, t: pyo.quicksum(
            m.selfgen[t, (s, p)] for (s, p) in S
        ) * 1e6 * selfgen_ef_e
    )
    # ALWAYS total -- no longer switched by selfgen_in_emissions_cap.
    m.emissions_by_year = pyo.Expression(
        T,
        rule=lambda m, t:
            m.emissions_grid_only_by_year[t]
            + m.emissions_selfgen_by_year[t]
    )
    # What the CAP constrains. Default grid-only (correct NDC boundary);
    # True widens it to total = the alternative-boundary arm.
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

    carbon_cost_npv = 0.0
    if scenario.get("carbon_active", False):
        carbon_price = float(scenario["carbon_price"])
        carbon_cost_npv = pyo.quicksum(
            m.DF[t] * m.emissions_by_year[t] * carbon_price for t in T
        )

    # F1, Step 5/6: the live objective contains EXACTLY ONE of {F1 tranche
    # charge, overnight capex + salvage + replacement}. The
    # 'pre_f1_overnight' branch below is written with the IDENTICAL term
    # order as the pre-F1 system_cost_npv formula (not refactored through
    # an intermediate variable) so the expression TREE -- not just its
    # value -- is unchanged: floating-point addition is not strictly
    # associative, so even a mathematically-equivalent regrouping could
    # shift the last bit of the regression value. Mirrors optimize_model.py.
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

    m.cost_gas_opex_npv = pyo.Expression(expr=gas_opex_npv)
    m.cost_om_npv = pyo.Expression(expr=om_cost_npv)
    if capital_treatment_case == "pre_f1_overnight":
        m.cost_gas_capex_npv = pyo.Expression(expr=gas_capex_npv)
        m.cost_public_solar_capex_npv = pyo.Expression(expr=public_solar_capex_npv)
        m.cost_eaas_solar_capex_npv = pyo.Expression(expr=eaas_solar_capex_npv)
        m.cost_storage_capex_npv = pyo.Expression(expr=storage_capex_npv)
        m.cost_salvage_npv = pyo.Expression(expr=m.salvage_npv)
    m.cost_unserved_voll_npv = pyo.Expression(expr=unserved_npv)
    m.cost_selfgen_npv = pyo.Expression(expr=selfgen_npv)
    m.emissions_selfgen = pyo.Expression(
        expr=pyo.quicksum(m.emissions_selfgen_by_year[t] for t in T))
    m.emissions_grid_only = pyo.Expression(
        expr=pyo.quicksum(m.emissions_grid_only_by_year[t] for t in T))
    m.cost_carbon_npv = pyo.Expression(expr=carbon_cost_npv)
    # Phase 2.8 diagnostic (task_conversion_capex_exposure, Part 1B). RAW
    # overnight OCGT-to-CCGT conversion capital expenditure, per year --
    # BEFORE any annuity or discounting, reading m.convert_mw[tier] (the
    # raw MW decision) and _conv_capex_per_kw (the same USD/kW read by
    # convert_split_identity above) directly, not via m.convert_comm_vol.
    # convert_mw is a one-off decision assigned to vintage 0 (Decision 5),
    # so this is 0.0 for every t except t == 0, where it equals the full
    # retrofit spend across all three tiers.
    #
    # Referenced by NO constraint and NO objective term (grepped this
    # entire file for the name below: the only occurrence is its own
    # definition) -- a pure diagnostic, read only by extract_planning_
    # diagnostics()'s _capex_at() in src/optimize_experiments.py.
    #
    # Deliberately given its OWN name rather than reusing m.convert_comm_
    # vol or m.cost_conversion_capital_charge_npv: this Expression and
    # convert_comm_vol are numerically identical per tier (convert_comm_
    # vol[tier] is CONSTRAINED, by convert_split_identity above, to equal
    # exactly conversion_capex_per_kw * 1000 * convert_mw[tier] -- the same
    # raw product, no annuity or discounting applied to convert_comm_vol
    # itself; the annuity is the separate m.cost_conversion_capital_charge_
    # npv above). But convert_comm_vol's own name implies a commercial-
    # tranche financing split it does not carry (task_conversion_
    # diagnostics background), and a diagnostic consumer reading it as "the
    # raw spend" has to re-derive that fact from this comment every time.
    # This Expression exists so a reader never has to.
    if _conv:
        m.conversion_overnight_capex_usd_by_year = pyo.Expression(
            T,
            rule=lambda m, t:
                pyo.quicksum(
                    _conv_capex_per_kw * 1000.0 * m.convert_mw[tier]
                    for tier in _CONV_KEYS
                ) if t == 0 else 0.0
        )

    # Public budget (annual)
    public_budget_npv = scenario.get("public_solar_budget_npv", None)
    if public_budget_npv is not None:
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
        m.has_public_budget_constraint = True

    # ------------------------------------------------------------
    # Objective and emissions constraints
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
            raise ValueError("emissions_cap_by_year length mismatch.")
        m.emissions_constraint = pyo.Constraint(
            T, rule=lambda m, t: m.emissions_capped_by_year[t] <= emissions_cap_by_year[t]
        )

    elif emissions_cap is not None:
        m.emissions_constraint = pyo.Constraint(expr=m.emissions_capped <= emissions_cap)

    m.objective = pyo.Objective(expr=system_cost_npv, sense=pyo.minimize)

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
        ("selfgen", m.selfgen),
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


def firm_capacity_diagnostic(m, years):
    """Post-solve slack and dual on m.firm_capacity_constraint, per year
    (2.6, F0 Step 2.3). Mirrors optimize_model.py's version exactly.

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
            "selfgen": float(pyo.value(m.selfgen[t_index, (s, p)])),
        })
    return rows


def conversion_diagnostic(m, years, scenario):
    """
    Per-tier OCGT-to-CCGT conversion outcome (Phase 2.8, Step 4).

    Returns {} when the model carries no conversion mechanism -- the
    annual builder, or build_model_sliced() under conv_none -- so callers
    can attach it unconditionally.

    Per tier {A, B, C}:
      convert_mw                : MW of OCGT nameplate converted
      nameplate_mw              : that tier's convertible nameplate (bound)
      share_of_nameplate        : convert_mw / nameplate_mw
      tier_paf                  : nameplate-weighted availability of the tier
      converted_generation_twh_e_by_year : converted electrical output,
                                 slice-summed to annual TWh_e -- needed to
                                 tell "built and running" from "built and
                                 idle" (Step 4).
    Plus:
      total_convert_mw          : sum over tiers
      implied_conversion_capex_usd : capex_per_kw * 1000 * total_convert_mw
    """
    if not hasattr(m, "convert_mw"):
        return {}

    tiers = scenario["conversion_tiers"]
    T = range(len(years))
    out = {}
    for tier in ("A", "B", "C"):
        mw = float(pyo.value(m.convert_mw[tier]))
        nameplate = float(tiers[tier]["nameplate_mw"])
        out[tier] = {
            "convert_mw": mw,
            "nameplate_mw": nameplate,
            "share_of_nameplate": (mw / nameplate) if nameplate > 0 else 0.0,
            "tier_paf": float(tiers[tier]["paf"]),
            "converted_generation_twh_e_by_year": {
                int(years[t]): sum(
                    float(pyo.value(m.convert_generation[tier, t, (s, p)]))
                    for (s, p) in SLICES
                )
                for t in T
            },
        }

    total_mw = sum(out[tier]["convert_mw"] for tier in ("A", "B", "C"))
    capex_per_kw = scenario.get("conversion_capex_per_kw")
    out["total_convert_mw"] = total_mw
    out["implied_conversion_capex_usd"] = (
        float(capex_per_kw) * 1000.0 * total_mw if capex_per_kw else 0.0
    )
    return out
