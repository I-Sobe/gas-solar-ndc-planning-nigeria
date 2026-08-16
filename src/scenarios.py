"""
Scenario Definitions Module (Configuration Only)

Scope
-----
Defines deterministic planning scenarios as structured parameter dictionaries
for use in system evaluation and optimization studies.

This module provides scenario DEFINITIONS ONLY.
It does NOT execute models, run experiments, or aggregate results.

Demand framing (examiner-safe)
------------------------------
- Demand is treated as an EXOGENOUS planning input, not statistically forecast
  from historical time series.
- The base-year demand level is Tier 1: observed gross generation at busbar.
  Reconstructed latent-demand bases (latent_low/latent_high) are DEPRECATED
  and raise in load_scenario; they are superseded by the tiered demand
  architecture (Tier 2 self-generation, Tier 3 access-adjusted), Phase 2.
- Growth-rate scenarios remain separate from demand-level scenarios.

Base-year demand anchor (2024) — GROSS GENERATION at busbar
------------------------------------------------------------
Demand is metered at gross generation sent out. This is the natural boundary
for a generation capacity-expansion model: capacity serves generation, and the
NDC emissions caps account against actual grid output.

  [SOURCE: NERC 2024 — gross generation 37,093.70 GWh; hydro share 30.92%]
    grid gross generation   37.09 TWh   <- Tier 1, the model's demand base
      of which gas          25.62 TWh
      of which hydro        11.47 TWh

Loss ladder (NERC 2024) — for reference, NOT applied in the energy balance:
    gross generation        37.09 TWh
      -7% TLF            -> 34.50 TWh   delivered to DisCos
      -36.4% ATC&C       -> 21.97 TWh   collected (billed and paid)

The previous base (23.08 TWh — NBS Q1 2024 served energy of 5.770 TWh,
annualised as Q1 x 4) sat at the bottom of that ladder: it was COLLECTED
energy, excluding electricity consumed but stolen, unmetered, or unpaid — all
of which is real demand of real users. The Q1 x 4 annualisation is separately
unsound, ignoring seasonality, and is not used anywhere in the current model.

UNITS CONVENTION (read before editing any value)
------------------------------------------------
- Demand, hydro, solar/gas generation entering the electricity balance are in
  TWh_e (electrical energy) per year.
- Gas *deliverability* (gas_scenario data) is in TWh_th (thermal); it is
  converted to electrical via gas_eta inside build_model. Hydro is ALREADY
  electrical and must NOT be multiplied by gas_eta.
"""

from __future__ import annotations

import numpy as np


# ============================================================
# BLENDED CONCESSIONAL FINANCE (single source of truth)
# ============================================================

def blended_finance_scenarios():
    """
    Blended concessional finance capital-structure scenarios for EaaS solar.

    The blended WACC fed to the bankability test is DERIVED from these tuples,
    not asserted. 'blended_central' is the adopted working case.

    Tuple order: (concessional_share, r_concessional, r_commercial)

    ALL RATES ARE PLACEHOLDERS pending primary-source verification:
      - r_commercial   : Nigerian commercial power-sector WACC.
                         Anchor: IEA / CCSI 'Africa power-sector WACC > 18%
                         (2023)'. VERIFY a Nigeria-specific value from the
                         Nature (2025) global CoC dataset (s41597-025-05912-x).
      - r_concessional : DFI / climate-fund concessional senior debt. Sits below
                         the Kenya/Senegal *blended* 8.5-9% because it is the
                         concessional tranche alone. SOURCE to a named facility.
      - concessional_share : from an actual blend / mobilisation ratio. SOURCE.

    NOTE: blended_central here is the SINGLE definition of the adopted capital
    structure. load_scenario() pulls its defaults from this entry so the whole
    codebase has one source of truth for the central blend. Do not hard-code
    these three numbers anywhere else.
    """
    return {
        # label:              (conc_share, r_conc, r_comm)   blended WACC
        "commercial_only":    (0.00, 0.06, 0.18),   # 18.0% (unblended benchmark)
        "blended_central":    (0.30, 0.06, 0.18),   # 14.4% (ADOPTED working case)
        "blended_optimistic": (0.50, 0.06, 0.18),   # 12.0% (deeper concessional)
        "concessional_heavy": (0.75, 0.06, 0.18),   #  9.0% (near-intervention)
    }


# ============================================================
# PLANNING HORIZON
# ============================================================

def planning_horizon(start_year: int = 2025, end_year: int = 2045) -> np.ndarray:
    """Define the planning horizon (annual resolution)."""
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")
    return np.arange(start_year, end_year + 1)


# ============================================================
# SCENARIO REGISTRIES
# ============================================================

def demand_level_scenarios() -> dict[str, float]:
    """
    Base-year demand level scenarios (TWh/year).

    - served     : Tier 1 base — observed GROSS GENERATION at busbar, 2024.
                   [SOURCE: NERC 2024 — 37,093.70 GWh]. Not a bound; measured.
                   NOTE: the key name is a misnomer pending rename (plan 1.6
                   Step 5). Served demand is 21.97 TWh; see loss ladder above.
    - latent_low : DEPRECATED (raises in load_scenario) — was lambda = 0.60
                   applied to the collected base.
    - latent_high: DEPRECATED (raises in load_scenario) — was lambda = 0.30.

    Traceable to: NERC 2024 annual gross generation (37,093.70 GWh).
    NOTE: data/demand/demand_base_annualized_2024.csv.xlsx is STALE — it holds the
    superseded 23.08 collected-energy base and is not read by this module.
    """
    return {
        # WARNING: latent_low/latent_high are retained for provenance only and
        # raise in load_scenario(). Do not iterate this dict without filtering.
        # Tier 1 (calibration anchor): observed GROSS GENERATION at busbar, 2024.
        # Corrected from 23.08, which was COLLECTED energy (a revenue quantity),
        # not demand. See docstring for the loss ladder.
        "served": 37.09,
        # DEPRECATED — wrong base (built as multiples of collected energy).
        # Rebuilt as the tiered demand architecture in Phase 2. DEM-1 suspended.
        "latent_low": 38.47,
        "latent_high": 76.93,
    }


def demand_growth_scenarios() -> dict[str, float]:
    """
    Annual demand growth arms (fraction/year), applied to the Tier 1 base
    (37.09 TWh gross generation at busbar, NERC 2024).

    SOURCING: correction plan Phase 1.6, Steps 1-4. Every value below is
    triangulated from at least two independent routes. Workbook and full
    provenance: data/demand/step1_per_capita_diagnostic.xlsx and
    data/demand/README_step1_per_capita.md.

    WHY NOT HISTORICAL GENERATION CAGR
    ----------------------------------
    Nigerian grid generation is a SUPPLY series, not a demand series. Over
    2010-2025 it grew 3.14%/yr while population grew 2.39%/yr — i.e. 76% of
    observed growth is demographic. Per CONNECTED person, generation FELL
    1.66%/yr (327 -> 258 kWh, 2010-2024). Calibrating demand growth to this
    series would assume the supply constraint persists and then report modest
    capacity needs because demand was assumed not to grow. Same error class as
    the superseded 23.08 TWh base. NIRP 2024 (Table 14) reaches the same
    conclusion and forecasts bottom-up from drivers instead.

    THE THREE SOURCING ROUTES
    -------------------------
    Route A - driver decomposition.  g = eps*g_GDP + (1-eps)*g_pop
        eps    = 0.76-0.90, long-run income elasticity of electricity demand,
                 middle-income NON-SSA panel [Liddle, Smyth & Zhang, Energy
                 Economics]. SSA-estimated elasticities (0.60-0.76) are
                 rejected as a central source: they are estimated on
                 supply-rationed systems and inherit the same contamination
                 as the Nigerian series. Retained as a lower bound only.
        g_pop  = 1.77%/yr, 2024-2045 average [UN WPP: 232.68m -> 336.66m].
                 NOTE: g is INSENSITIVE to this. The population coefficient is
                 (1-eps) ~ 0.15, so a 0.3pp change in g_pop moves g by <0.05pp.
                 g is approximately eps x GDP growth. The GDP assumption is the
                 load-bearing choice, not the population assumption.
        g_GDP  = declared arms, not forecast. See table below.

    Route B - per-capita convergence.  Nigeria 2024 = 161 kWh/cap.
        Nigeria is ~15 years behind Kenya (Kenya crossed 161 kWh/cap in 2009).
        Ghana is unusable as a target: Nigeria's 2024 level is BELOW Ghana's
        worst year since 2000, and converging to Ghana's 2024 level (705
        kWh/cap) would require 9.24%/yr.

    Route C - NIRP 2024, Nigeria's official Integrated Resource Plan.
        Total demand (grid + off-grid) 59.6 TWh (2022) -> 328.6 (2045) = 7.70%.
        This is TIER 2 SCOPE, not Tier 1: the base already includes
        self-generation. NIRP's grid-only rate (11.0%) is organic growth PLUS
        absorption of self-generation and must NOT be applied here — in this
        architecture suppression closure is the Tier1->Tier2 gap, not a growth
        rate. Applying it would double-count.
        NIRP's implied elasticity is 1.19 (= (7.7-2.1)/(6.8-2.1)), ~35% above
        the credible literature; its GDP path (6.8%/yr to 2045) is roughly the
        pre-2015 oil-boom rate. Aggressive on both terms simultaneously.

    THE ARMS
    --------
    constrained_continuation = 0.0314
        Ember 2010-2025 grid generation CAGR. This is a CONSTRAINT-PERSISTENCE
        arm ("what if the constraints hold"), NOT organic demand growth, and
        must never be presented as such. Window range 1.53%-3.22%.
        Clears the floor test (>1.77% population growth) and implies 1.35%/yr
        per-capita growth — above the 0.73% actually observed, so even the
        pessimistic arm is mildly optimistic against the record.

    organic_central = 0.040
        Route A: eps 0.83, GDP 4.5% -> 4.04%.
        Route B: converge to Kenya's 2024 level by 2045 -> 3.67%.
        GDP 4.5% is a DECLARED ASSUMPTION: above the post-2015 average
        (two recessions, weak diversification), materially below the oil-boom
        rate. It does NOT assume a commodity boom returns. Verify the two
        historical averages against WDI NY.GDP.MKTP.KD; IMF WEO covers only
        ~5 years, so beyond 2030 this is an assumption, not a forecast.

    organic_high = 0.060
        Route A: eps 0.90, GDP 6.8% -> 6.30%.
        Route B: converge to Ghana's 2015 level by 2045 -> 6.40%.
        Route C: NIRP low case -> 5.9%.

    NOT AN ARM: NIRP's 7.7% base case. See demand_growth_benchmark().

    KNOWN LIMITATIONS (must be disclosed in the thesis)
    ---------------------------------------------------
    1. AMPLIFICATION. Gas is deliverability-bound and hydro is exogenous, so
       solar is an accounting residual: solar_TWh(2045) ~ D(2045) - 39.9. The
       amplification factor D/(D-39.9) is ~2x. A 1pp error in g moves the
       headline solar build by roughly 20%.
    2. CONSTANT g. NIRP's trajectory is front-loaded (10.8%/yr to 2030, then
       6.9%). A constant g reaching the same D(2045) understates near-term
       build pressure and therefore near-term financing stress. Piecewise g
       (break at 2030) requires changing demand.project_baseline_demand, which
       takes a scalar. Deferred deliberately — one structural change at a time.
    3. CONSTRAINT-RELIEF ASYMMETRY. These arms are sourced from UNCONSTRAINED
       drivers while gas deliverability is anchored to OBSERVED CONSTRAINED
       throughput (89.27 TWh_th, a commercial not a geological constraint).
       That asymmetry is what generates the solar residual. It is a scenario
       design choice and must be declared, not inherited. See the demand-arm x
       gas-regime coherence cross-tab (plan 1.6 Step 8).
    4. GDP IS EXOGENOUS. Electricity supply constrains GDP in reality; that
       feedback is not modelled. High-GDP arms paired with high unserved energy
       are internally incoherent. Excluding those cells is CONSERVATIVE (it
       removes demand, not adds it). NIRP has the identical omission and does
       not flag it.
    """
    return {
        "constrained_continuation": 0.0314,
        "organic_central":          0.040,
        "organic_high":             0.060,
    }


def demand_growth_benchmark() -> dict[str, float]:
    """
    External demand-growth benchmarks from other models. NOT scenario arms.

    Deliberately kept OUT of demand_growth_scenarios() so that consumers
    iterating that registry (optimize_experiments.run_all_deterministic_
    scenarios, scripts/13_) are unaffected, and so the three-arm structure
    stays intact.

    nirp_2024_base = 0.077
        NIRP 2024 total-demand CAGR, 59.6 TWh (2022) -> 328.6 TWh (2045).
        Tier 2 scope. Rests on eps = 1.19 and GDP 6.8%/yr, both above the
        credible ranges used for the arms above. Used only for the NIRP
        benchmark comparison run.
    """
    return {"nirp_2024_base": 0.077}


def demand_growth_prior() -> dict[str, float]:
    """
    Prior over annual demand growth for Monte Carlo sampling (scripts/17_).

    SOURCED — correction plan Phase 1.6 Step 5b. Supersedes the previous
    N(0.04, 0.01), in which BOTH parameters were unsourced: the mean was
    inherited from a placeholder of unknown provenance, and sigma was assumed.

    MEAN = demand_growth_scenarios()["organic_central"] = 0.040
        Triangulated from three independent routes; see
        demand_growth_scenarios() for the full derivation and sources. The key
        is referenced, not hardcoded, so the prior cannot drift from the arms.

    SIGMA = 0.0120
        Derived from the SPREAD BETWEEN THE ARMS rather than assumed. It is the
        equal-weighted standard deviation of the three sourced arms:

            arms          : 0.0314, 0.0400, 0.0600
            arm mean      : 0.0438
            deviations    : -0.0124, -0.0038, +0.0162
            sigma         : sqrt(mean of squared deviations) = 0.01198 -> 0.0120

        This is the defensible construction: the dispersion now reflects the
        observed disagreement between the driver decomposition, the per-capita
        convergence path and the NIRP forecast, rather than an analyst's guess
        about how uncertain the parameter feels.

    KNOWN LIMITATION — THE PRIOR IS SYMMETRIC BUT THE ARMS ARE NOT
    ---------------------------------------------------------------
    The arms are right-skewed around organic_central:

        constrained_continuation  0.0314   = central - 0.86 pp
        organic_central           0.0400   = central
        organic_high              0.0600   = central + 2.00 pp

    The upside arm is 2.3x further from centre than the downside arm, because
    the downside is bounded by the observed historical record (a
    constraint-persistence scenario cannot fall far below what actually
    happened) while the upside is bounded only by the GDP assumption, which is
    genuinely open. The equal-weighted arm mean is 0.0438, ABOVE
    organic_central.

    A symmetric normal therefore:
      - understates the probability of high-growth outcomes, and
      - overstates the probability of growth below the historical record.

    Because of the ~2x amplification into the solar build (see
    demand_growth_scenarios(), limitation 1), this is not cosmetic: it biases
    the cost and infeasibility distributions DOWNWARD. MC results are
    conservative in this respect, which is the safer direction, but the
    asymmetry must be disclosed.

    Candidate replacements, deferred to plan 5.1 rather than fixed here:
      (a) an explicit three-point discrete prior over the arms with declared
          weights — simplest, fully transparent, and matches how the arms were
          actually constructed;
      (b) a lognormal calibrated to the arm quantiles.
    Option (a) is preferred: it makes the prior an exact restatement of the
    sourcing work rather than a smooth approximation to it.

    ALSO UNRESOLVED (plan 1.6 Step 8): scripts/17_ samples this prior
    INDEPENDENTLY of the gas regime. Both are driven by the same underlying
    reform dynamic, so independent sampling generates incoherent worlds and
    inflates both tails. No MC tail statistic — including the headline
    infeasibility rate — is reportable until the correlation structure is
    imposed.
    """
    return {
        "mean":  demand_growth_scenarios()["organic_central"],
        "sigma": 0.0120,
    }


def solar_build_scenarios():
    """Annual solar build-rate caps (MW/year)."""
    return {
        "conservative": 500,
        "baseline": 1000,
        "aggressive": 2000,
    }


def carbon_policy_scenarios() -> dict[str, dict[str, float | bool]]:
    """Carbon policy stances (deterministic). Price in USD/tCO2."""
    return {
        "no_policy": {
            "carbon_active": False,
            "carbon_price": 0.0,
        },
        "moderate_policy": {
            "carbon_active": True,
            "carbon_price": 50.0,
        },
        "stringent_policy": {
            "carbon_active": True,
            "carbon_price": 75.0,
        },
    }


def gas_deliverability_scenarios() -> dict[str, str]:
    """
    Gas deliverability-to-power scenario labels.

    Must match the 'scenario' column in:
    data/gas/processed/gas_available_power_annual_twh_th.csv

    Structural (all RQs): downside, baseline, upside, shock_recovery
    Level-equivalent flat controls (GAS-3 only): flat_downside, flat_upside,
      flat_shock_recovery -- identical cumulative supply, no shape variation,
      to isolate SHAPE vs LEVEL effects.
    """
    return {
        "downside":            "downside",
        "baseline":            "baseline",
        "upside":              "upside",
        "shock_recovery":      "shock_recovery",
        "flat_downside":       "flat_downside",
        "flat_upside":         "flat_upside",
        "flat_shock_recovery": "flat_shock_recovery",
    }


def land_scenarios():
    """Land availability envelopes (km2)."""
    return {
        "tight": 1000,
        "moderate": 3000,
        "loose": 6000,
    }


def capital_envelope_scenarios():
    """
    Public capital envelopes (NPV over full planning horizon, USD).

    Fractions are applied to B_star, the adequacy capital requirement.

    STALE-DOC: an earlier docstring quoted 'B* ~ 6.13B USD'. The coded value is
    9.104B. Reconcile and restate B_star from a single documented derivation in
    the final calibration pass. Do not cite 6.13B in the thesis until resolved.
    """
    B_star = 9_104_000_000

    return {
        "tight": int(0.50 * B_star),
        "moderate": int(0.85 * B_star),
        "adequacy": int(1.00 * B_star),
        "expansion": int(1.20 * B_star),
        "unconstrained": None,
    }


def solar_tariff_scenarios():
    """
    EaaS service tariff scenarios (USD per TWh). Named levels low/baseline/high.

    STALE-DOC (deferred to final calibration pass): the previous threshold note
    ('discount rate 4%, T* = 44.3-46.4M') is NO LONGER VALID. The model now uses
    a social discount rate for the objective and a private/blended rate for the
    bankability test, under which T* is materially higher and is a FUNCTION of
    the blended-finance scenario (one T* per blend). Recompute T*(blend) and the
    tariff grid once the financing rates are frozen. Until then, treat the grid
    below as provisional.
    """
    return {
        "low":      45_000_000,
        "baseline": 65_000_000,
        "high":     85_000_000,
    }


# Tariff sweep grid for FIN-2 bankability analysis (USD per TWh).
# STALE-DOC: fine resolution was placed around the old 44-46M crossing. Under
# the blended-finance rates the crossing moves (higher, and blend-dependent).
# Widen and re-centre this grid in the final pass once rates are frozen.
TARIFF_SWEEP_GRID = [
    30_000_000,
    35_000_000,
    40_000_000,
    44_000_000,
    46_000_000,
    50_000_000,
    60_000_000,
    75_000_000,
    95_000_000,
    110_000_000,
]


def gas_probability_weights():
    """
    Prior over gas deliverability regimes for Monte Carlo robustness diagnostics.

    Not a forecast -- a weighted sensitivity prior. Report headline results under
    symmetric weights (0.25 each) as a robustness check as well.

    SOURCING NOTE: the justifications reference NGC Decade of Gas, IEA Nigeria
    Outlook 2023, Nigerian Gas Master Plan / PIA 2021. Confirm each citation
    against a primary source before submission (standing verification rule).
    """
    return {
        "baseline":       0.50,
        "downside":       0.25,
        "upside":         0.20,
        "shock_recovery": 0.05,
    }


# ============================================================
# SCENARIO CONSTRUCTOR
# ============================================================

def load_scenario(
    demand_level_case: str = "served",
    demand_case: str = "organic_central",
    land_case: str = "moderate",
    capital_case: str = "moderate",
    gas_deliverability_case: str = "baseline",
    solar_build_case: str = "baseline",
    solar_tariff_case: str = "baseline",
    carbon_case: str = "no_policy",
    blend_case: str = "blended_central",
    start_year: int = 2025,
    end_year: int = 2045,
) -> dict:
    """
    Build a scenario parameter dictionary consumed by the optimization model.

    Parameters
    ----------
    demand_level_case : {"served"} — the only accepted value. "latent_low" and
                        "latent_high" are retained in the registry for
                        provenance but raise (see validation below).
    demand_case       : key of demand_growth_scenarios() —
                        {"constrained_continuation", "organic_central",
                         "organic_high"}. NIRP's 7.7% benchmark is NOT here;
                        see demand_growth_benchmark().
    land_case         : key of land_scenarios()
    capital_case      : key of capital_envelope_scenarios()
    gas_deliverability_case : key of gas_deliverability_scenarios()
    solar_build_case  : key of solar_build_scenarios()
    solar_tariff_case : key of solar_tariff_scenarios()
    carbon_case       : key of carbon_policy_scenarios()
    blend_case        : key of blended_finance_scenarios() (default central).
                        Sets the EaaS capital structure -> blended private rate.
    start_year, end_year : planning horizon bounds.
    """

    # ---- Validate labels
    if demand_level_case not in demand_level_scenarios():
        raise ValueError(f"Unknown demand_level_case: {demand_level_case}")
    _DEPRECATED_DEMAND_LEVELS = {"latent_low", "latent_high"}
    if demand_level_case in _DEPRECATED_DEMAND_LEVELS:
        raise ValueError(
            f"demand_level_case='{demand_level_case}' is deprecated and must "
            "not be used. These values were constructed as multiples of "
            "COLLECTED energy (23.08 TWh, a revenue quantity), not demand. "
            "Superseded by the tiered demand architecture (correction plan "
            "2.7). DEM-1 is suspended pending that rebuild."
        )
    if demand_case not in demand_growth_scenarios():
        raise ValueError(f"Unknown demand_case: {demand_case}")
    if carbon_case not in carbon_policy_scenarios():
        raise ValueError(f"Unknown carbon_case: {carbon_case}")
    if gas_deliverability_case not in gas_deliverability_scenarios():
        raise ValueError(f"Unknown gas_deliverability_case: {gas_deliverability_case}")
    if land_case not in land_scenarios():
        raise ValueError(f"Unknown land_case: {land_case}")
    if capital_case not in capital_envelope_scenarios():
        raise ValueError(f"Unknown capital_case: {capital_case}")
    if solar_build_case not in solar_build_scenarios():
        raise ValueError(f"Unknown solar_build_case: {solar_build_case}")
    if solar_tariff_case not in solar_tariff_scenarios():
        raise ValueError(f"Unknown solar_tariff_case: {solar_tariff_case}")
    if blend_case not in blended_finance_scenarios():
        raise ValueError(f"Unknown blend_case: {blend_case}")

    years = planning_horizon(start_year, end_year)

    # ---- Blended finance: single source of truth (no hard-coded duplicates)
    conc_share, r_conc, r_comm = blended_finance_scenarios()[blend_case]

    scenario = {
        # ---- Temporal
        "years": years,

        # ---- Demand (TWh_e, annual)
        "base_demand_twh": demand_level_scenarios()[demand_level_case],
        "demand_growth": demand_growth_scenarios()[demand_case],

        # ---- Gas deliverability to power (TWh_th; converted by gas_eta)
        "gas_scenario": gas_deliverability_scenarios()[gas_deliverability_case],
        "gas_eta": 0.287,

        # ---- Gas capacity baseline (gas-only fleet, MW)
        "gas_baseline_mw": 11000,
        # Brownfield retirement proxy: linear retirement from 2035.
        # (Comment corrected: 680 MW/yr x 11 years = 7,480 MW retired by 2045
        #  from an 11,000 MW gas-only base -- NOT the whole fleet.)
        "gas_baseline_retirement_start_year": 2035,
        "gas_baseline_retirement_mw_per_year": 680.0,
        "gas_capex_per_mw": 900000,

        # ---- Gas fleet availability ceiling (fraction of installed capacity) ---
        # This scales INSTALLED gas capacity to a realistic ENERGY ceiling. It is
        # a capacity ceiling, not a dispatch prediction: the optimiser sets actual
        # utilisation below it, subject to fuel (gas_balance) and demand.
        #
        # CRITICAL -- this must be FUEL-AGNOSTIC availability: mechanical +
        # atmospheric availability only (NERC Availability Factor clauses (i) and
        # (ii)). It must EXCLUDE clause (iii) feedstock availability, because gas-
        # supply scarcity is already carried by gas_balance / the deliverability
        # scenarios. Using NERC's headline Availability Factor here would double-
        # count fuel scarcity and contaminate the gas shadow prices.
        
        # Source: fuel-agnostic mechanical/maintenance availability, reconstructed
        # from NERC outage cause-decomposition, or an OCGT/CCGT engineering
        # benchmark derated for Nigerian plant condition (~0.75-0.85). REPLACE.
        "gas_availability_factor": 0.80,   # <-- set from fuel-agnostic availability

        # ---- Hydro (exogenous must-run, TWh_e/year, already electrical)
        # -------------------------------------------------------------------
        # Measured 2024: 11.47 TWh (NERC — 30.92% of 37,093.70 GWh).
        # 2025 ~12.8 TWh as Zungeru (700 MW) ramps.
        # TODO Phase 2: replace with explicit committed-vs-NDC-aligned trajectory.
        "hydro_baseline_twh": 11.47,        "hydro_growth": 0.0,          # flat unless modelling new hydro (e.g. Zungeru ramp)
        # Installed hydro capacity (MW) — required by the sliced model, which
        # dispatches hydro within a seasonal energy budget subject to a power
        # limit. Kainji + Jebba + Shiroro + Zungeru + Dadin-Kowa.
        # [SOURCE: NERC — sum of installed hydro plant ratings]
        "hydro_capacity_mw": 2678,

        # ---- Land policy
        "land_available_km2": land_scenarios()[land_case],
        "land_intensity_solar_km2_per_mw": 0.025,
        "land_intensity_gas_km2_per_mw": 0.001,
        "land_intensity_storage_km2_per_mwh": 0.00001,
        "storage_baseline_mwh": 0.0,

        # ---- Solar
        "solar_cf": 0.20,   # fixed-tilt CF, central. Global Solar Atlas PVOUT for
                            # named N. Nigeria sites / 8760, net of soiling/thermal/
                            # availability derating. Sweep {0.18, 0.20, 0.22}.
                            # [SOURCE: Global Solar Atlas, sites: <name them>]
        "solar_baseline_mw": 500,
        "solar_max_build_mw_per_year": solar_build_scenarios()[solar_build_case],
        # Min annual build floor: set to 100 MW/yr in runners when time-varying
        # CAPEX is active (prevents pathological all-delay). 0 = disabled here.
        "solar_min_build_mw_per_year": 0.0,
        "solar_capex_scenario": "solar_low",

        # ---- Intra-annual shape parameters (sliced model only) --------------
        # ALL are declared assumptions pending sourcing. Each is swept.
        # Relative average power by season (ratio only; normalised internally).
        # Dry (Nov-Apr) demand higher: heat, peak Jan-Apr per cited load research.
        # [SOURCE NEEDED: quantitative seasonal demand ratio]
        "season_demand_factor": {"dry": 1.08, "wet": 0.92},
        # Relative average power by period. Flatter than residential: national
        # aggregate load has C&I daytime offset; night is the genuine trough.
        # Observed grid data CANNOT settle this (shedding truncates the peak).
        # [SOURCE NEEDED: quantitative intra-day load shape]
        "period_demand_factor": {"night": 0.65, "day": 1.05, "peak": 1.20},
        # Relative solar yield by season, day slices only. Dry = clearer skies
        # (offset by harmattan dust); wet = monsoon cloud.
        # [SOURCE NEEDED: Global Solar Atlas MONTHLY PVOUT, named sites]
        "season_solar_factor": {"dry": 1.15, "wet": 0.85},
        # Hydro seasonal energy split. 2025 data cannot establish this (Zungeru
        # ramp masks any seasonal signal). Central 0.90; sweep {0.75,0.90,1.00}.
        # [SOURCE NEEDED: pre-Zungeru quarterly generation, 2022-2023]
        "hydro_season_factor": {"dry": 0.90, "wet": 1.10},
        # Minimum hydro per slice as a fraction of proportional share
        # (environmental flow, irrigation, downstream obligations).
        # [SOURCE NEEDED: minimum flow obligations, Kainji/Jebba/Shiroro]
        "hydro_min_flow_fraction": 0.50,
        # ---- Public capital constraint (NPV, USD)
        "public_solar_budget_npv": capital_envelope_scenarios()[capital_case],

        # ---- Storage reduced-form (annual)
        "storage_deployable_hours_per_year": 700.0,
        # storage_solar_surplus_frac: coupling cap intentionally DROPPED.
        # Storage is governed by power + throughput limits only. If reinstated,
        # it must be sourced from an 8760 solar-vs-load surplus calculation, not
        # a bare 0.20 default. Left commented as an explicit record of the choice.
        # "storage_solar_surplus_frac": 0.20,
        "storage_round_trip_eff": 0.90,

        # ---- Financing regime
        "financing_regime": "traditional",  # or "eaas"

        # ---- EaaS service tariff (USD per TWh)
        "solar_service_tariff_usd_per_twh":
            solar_tariff_scenarios()[solar_tariff_case],

        # ---- Discounting
        "social_discount_rate": 0.08,   # objective DF (society's time preference)

        # ---- EaaS capital structure (blended concessional finance)
        # Pulled from blended_finance_scenarios()[blend_case] above -- single
        # source of truth. build_model derives the private/bankability rate from
        # these via resolve_private_rate(). private_discount_rate is the fallback
        # used ONLY if the three keys below are absent.
        "blend_case": blend_case,
        "concessional_share": conc_share,
        "r_concessional": r_conc,
        "r_commercial": r_comm,
        "private_discount_rate": r_comm,   # fallback == unblended commercial rate

        # Optional Level-2 concessional envelope (NPV, USD). None => no scarcity
        # constraint (Level 1: blended rate only). Set in FIN runners to activate
        # the concessional shadow-price diagnostic.
        "concessional_envelope_npv": None,

        # ---- Required NPV margin (bankability coverage cushion, distinct from WACC)
        "required_margin": 1.10,

        # ---- Diagnostic-only inputs (consumed by specific runners, not the LP)
        # Confirm these are read where intended; they do not affect the core solve.
        "disco_collection_rate": 1.0,   # used by FIN-3 runner
        "peak_demand_multiple": 2.5,    # used by peak-adequacy diagnostic

        # ---- Carbon policy (spreads carbon_active / carbon_price)
        **carbon_policy_scenarios()[carbon_case],

        # ---- Labels (reporting)
        "labels": {
            "demand_level": demand_level_case,
            "demand": demand_case,
            "carbon": carbon_case,
            "gas_deliverability": gas_deliverability_case,
            "land": land_case,
            "solar_build": solar_build_case,
            "blend": blend_case,
        },
    }

    return scenario
