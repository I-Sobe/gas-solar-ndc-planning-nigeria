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
    Annual demand growth arms (fraction/year), applied to the Tier 1 base.

    KEYS RENAMED (plan 1.6 Step 5a). Old keys low/baseline/high are NOT
    aliased: load_scenario raises on unknown keys, so any missed call site
    fails loudly rather than silently running the wrong arm.

    VALUES IN THIS COMMIT ARE THE LEGACY UNSOURCED SET. They are replaced
    with the triangulated values in Step 5b. Do not cite anything from a run
    made at this commit.
    """
    return {
        "constrained_continuation": 0.025,   # legacy value — replaced in 5b
        "organic_central":          0.04,    # legacy value — replaced in 5b
        "organic_high":             0.06,    # legacy value — replaced in 5b
    }


def demand_growth_prior() -> dict[str, float]:
    """
    Prior over annual demand growth for Monte Carlo sampling.

    STATUS: [SOURCE NEEDED] — correction plan 1.6.

    mean  : provenance unknown (audited Aug 2026). Inherited from the
            'organic_central' demand_growth_scenarios() key, itself unsourced.
    sigma : assumed, not estimated. Intended replacement is the observed
            spread across independent sourcing routes (driver decomposition,
            per-capita convergence, NIRP 2024 forecast).

    The 'organic_central' key is hardcoded here deliberately: Step 5's rename must
    update this line, and will fail loudly rather than silently drift.

    No Monte Carlo result is reportable until both are sourced.
    """
    return {
        "mean":  demand_growth_scenarios()["organic_central"],
        "sigma": 0.01,
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
