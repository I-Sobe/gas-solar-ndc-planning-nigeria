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
- The base-year demand level is scenario-based and may represent:
    (i) served demand (lower bound), or
    (ii) reconstructed latent demand (suppressed-demand bounds).
- Growth-rate scenarios remain separate from demand-level scenarios.

Base-year demand anchors (2024)
-------------------------------
Anchored from NBS Q1 2024 served energy:
- Q1 served energy = 5,770 GWh = 5.770 TWh
- Annualized served demand = 4 × 5.770 = 23.08 TWh/year

Latent demand reconstruction via suppression factor λ:
- latent_low  = served / 0.60 = 38.47 TWh/year
- latent_high = served / 0.30 = 76.93 TWh/year
"""

from __future__ import annotations

import numpy as np


# ============================================================
# PLANNING HORIZON
# ============================================================

def planning_horizon(start_year: int = 2025, end_year: int = 2045) -> np.ndarray:
    """
    Define the planning horizon.

    Returns
    -------
    np.ndarray
        Array of simulation years (annual resolution).
    """
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")

    return np.arange(start_year, end_year + 1)


# ============================================================
# SCENARIO REGISTRIES
# ============================================================

def demand_level_scenarios() -> dict[str, float]:
    """
    Base-year demand level scenarios (TWh/year).

    Interpretation
    --------------
    - served: observed served demand (lower bound), annualized from NBS Q1 2024.
    - latent_low: reconstructed latent demand using λ = 0.60.
    - latent_high: reconstructed latent demand using λ = 0.30.

    Notes
    -----
    These values are intended to be traceable to the data layer:
    data/demand/demand_base_annualized_2024.csv
    """
    return {
        "served": 23.08,
        "latent_low": 38.47,
        "latent_high": 76.93,
    }


def demand_growth_scenarios() -> dict[str, float]:
    """
    Annual electricity demand growth assumptions (fraction per year).

    Notes
    -----
    These are scenario envelopes for long-term planning. They are not
    econometrically estimated from historical consumption.
    """
    return {
        "low": 0.025,
        "baseline": 0.04,
        "high": 0.06,
    }


def solar_build_scenarios():
    return {
        "conservative": 500,
        "baseline": 1000,
        "aggressive": 2000,
    }


def carbon_policy_scenarios() -> dict[str, dict[str, float | bool]]:
    """
    Carbon policy stances (deterministic).

    Carbon price anchor is applied elsewhere via economics.
    """
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

    These labels must match the 'scenario' column in:
    data/gas/processed/gas_available_power_annual_twh_th.csv

    Structural scenarios (used in all RQs):
      downside, baseline, upside, shock_recovery

    Level-equivalent flat scenarios (GAS-3 only):
      flat_downside, flat_upside, flat_shock_recovery
      These hold cumulative gas supply identical to the paired structural
      scenario but remove shape variation. Used to isolate whether cost
      and feasibility differences are driven by SHAPE or LEVEL.
    """
    return {
        "downside":            "downside",
        "baseline":            "baseline",
        "upside":              "upside",
        "shock_recovery":      "shock_recovery",
        # Level-equivalent controls for GAS-3 shape isolation
        "flat_downside":       "flat_downside",
        "flat_upside":         "flat_upside",
        "flat_shock_recovery": "flat_shock_recovery",
    }

def land_scenarios():
    return {
        "tight": 1000,
        "moderate": 3000,
        "loose": 6000,
    }

def capital_envelope_scenarios():
    """
    Public capital envelopes (NPV over full planning horizon).

    Calibrated from unconstrained adequacy requirement:
    B* ≈ 6.13B USD
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
    EaaS service tariff scenarios (USD per TWh).

    Three named levels for general use (low/baseline/high).

    Self-financing threshold analysis (FIN-2):
      At solar_low CAPEX (1,456,000 USD/MW), discount rate 4%, 21-year horizon:
        NPV energy per MW = 0.03451 TWh
        T* (unconditional, required_margin=1.10) = 46.4 M USD/TWh
        T* (conditional,   required_margin=1.05) = 44.3 M USD/TWh

      Below T*: financing_gap_per_mw > 0  → eaas_subsidy is required
      Above T*: financing_gap_per_mw = 0  → EaaS is fully self-financing

      The named levels are used by load_scenario() via solar_tariff_case.
      The FIN-2 sweep (run_tariff_bankability_sweep) uses the TARIFF_SWEEP_GRID
      constant defined below and bypasses load_scenario() directly.
    """
    return {
        "low":      45_000_000,
        "baseline": 65_000_000,
        "high":     85_000_000,
    }


# Tariff sweep grid for FIN-2 bankability analysis.
# Spans sub-threshold (30M) through well-above-threshold (110M),
# with fine resolution around the T* = 44-46M crossing point.
# Units: USD per TWh.
TARIFF_SWEEP_GRID = [
    30_000_000,   # well below threshold — large subsidy required
    35_000_000,
    40_000_000,
    44_000_000,   # just below T* (conditional threshold ~44.3M)
    46_000_000,   # just above T* (unconditional threshold ~46.4M)
    50_000_000,
    60_000_000,
    75_000_000,
    95_000_000,   # existing canonical run value
    110_000_000,  # well above threshold — no subsidy required
]

def gas_probability_weights():
    """
    Probabilistic prior over gas deliverability regimes for Monte Carlo analysis.

    These weights represent a plausible scenario prior consistent with Nigerian
    gas-to-power sector risk assessments. They are not probabilistic forecasts;
    they are weighted sensitivity priors used for robustness diagnostics.

    Justification:
    --------------
    baseline (0.50) — Central expectation aligned with NGC's Decade of Gas
        roadmap and the NDC 3.0 power-sector planning assumption of sustained
        current deliverability levels.

    downside (0.25) — Reflects the historical pattern of gas-to-power supply
        underperforming nameplate deliverability since 2015 (IEA Nigeria
        Energy Outlook 2023). Vandalism, pipeline integrity failures, and
        upstream pricing disputes have repeatedly suppressed available gas.

    upside (0.20) — Reflects the probability of successful execution of
        upstream gas investments and flare-capture under the 2060 net-zero
        framework (Nigerian Gas Master Plan, PIA 2021).

    shock_recovery (0.05) — Reflects tail-risk events such as the 2022
        vandalism-driven gas supply crash, where deliverability fell sharply
        before recovering. Low probability, high consequence.

    Sensitivity:
    ------------
    Headline results in this study should also be reported under symmetric
    weights (0.25 each) as a robustness check. Where findings are sensitive
    to weight specification, both are reported.
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
    demand_case: str = "baseline",
    land_case: str = "moderate",
    capital_case: str = "moderate", 
    gas_deliverability_case: str = "baseline",
    #solar_case: str = "baseline",
    solar_build_case: str = "baseline",
    solar_tariff_case= "baseline",
    carbon_case: str = "no_policy",
    start_year: int = 2025,
    end_year: int = 2045,
) -> dict:
    """
    Parameters
    ----------
    demand_level_case : str
        Base-year demand level. One of {"served", "latent_low", "latent_high"}.
        Anchored from NBS Q1 2024 served energy (23.08 TWh/year).
    demand_case : str
        Annual demand growth rate. One of demand_growth_scenarios() keys.
    land_case : str
        Land availability constraint. One of land_scenarios() keys.
    capital_case : str
        Public capital budget envelope. One of capital_envelope_scenarios() keys.
    gas_deliverability_case : str
        Gas deliverability-to-power scenario. One of gas_deliverability_scenarios() keys.
    solar_build_case : str
        Annual solar build rate cap. One of solar_build_scenarios() keys.
    solar_tariff_case : str
        EaaS service tariff level. One of solar_tariff_scenarios() keys.
    carbon_case : str
        Carbon policy stance. One of carbon_policy_scenarios() keys.
    start_year : int
        First year of planning horizon (default 2025).
    end_year : int
        Final year of planning horizon (default 2045).

    Returns
    -------
    dict
        Scenario parameter dictionary consumed by downstream evaluation
        and optimization modules.
    """

    # ---- Validate labels
    if demand_level_case not in demand_level_scenarios():
        raise ValueError(f"Unknown demand_level_case: {demand_level_case}")
    if demand_case not in demand_growth_scenarios():
        raise ValueError(f"Unknown demand_case: {demand_case}")
    #if solar_case not in solar_capacity_scenarios():
        #raise ValueError(f"Unknown solar_case: {solar_case}")
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
    
    years = planning_horizon(start_year, end_year)

    scenario = {
        # ---- Temporal
        "years": years,

        # ---- Demand (planning-level, annual)
        "base_demand_twh": demand_level_scenarios()[demand_level_case],
        "demand_growth": demand_growth_scenarios()[demand_case],
    
        # ---- Gas deliverability to power (optimization Phase 1: thermal energy)
        "gas_scenario": gas_deliverability_scenarios()[gas_deliverability_case],
        "gas_eta": 0.43,
        
        # ---- Gas capacity baseline
        "gas_baseline_mw": 13600,
        # ---- Gas fleet derating (brownfield retirement proxy)
        # Baseline fleet avg commissioning ~2005–2015. Assume linear retirement
        # begins 2035 (year 10 of horizon), 680 MW/yr through 2045.
        # Set to 0.0 to disable (e.g., upside gas scenarios with new builds).
        "gas_baseline_retirement_start_year": 2035,
        "gas_baseline_retirement_mw_per_year": 680.0,  # ~13600/20 years remaining
        # ---- Gas CAPEX
        "gas_capex_per_mw": 900000,

        # ---- Land policy
        "land_available_km2": land_scenarios()[land_case],
        "land_intensity_solar_km2_per_mw": 0.025,
        "land_intensity_gas_km2_per_mw": 0.001,
        "land_intensity_storage_km2_per_mwh": 0.00001,
        "storage_baseline_mwh": 0.0,  # No existing utility-scale BESS (brownfield baseline)
        # ---- Solar
        "solar_cf": 0.27,
        "solar_baseline_mw": 500,  # 2025 installed base (MW); brownfield anchor
        "solar_max_build_mw_per_year": solar_build_scenarios()[solar_build_case],
        # Minimum annual solar build rate (MW/yr).
        # Prevents pathological all-delay when time-varying CAPEX is active.
        # Default 100 MW/yr when time-varying CAPEX used; 0 otherwise.
        # Override in runner scripts: scenario["solar_min_build_mw_per_year"] = 100.
        "solar_min_build_mw_per_year": 0.0,
        "solar_capex_scenario": "solar_low",  # NREL ATB scenario for LCOE diagnostic
        # ---- Public capital constraint (NPV, USD)
        "public_solar_budget_npv": capital_envelope_scenarios()[capital_case],
        "disco_collection_rate": 1.0,
        # ---- Storage reduced-form (optimization; annual, energy-neutral)
        # Equivalent full cycles/year (dimensionless). Typical planning proxy: 150–350.
        "storage_deployable_hours_per_year": 700.0,
        # Fraction of solar generation available for charging storage
        "storage_solar_surplus_frac": 0.20,
        # Round-trip efficiency used as a limiter on usable discharge in reduced-form constraint.
        "storage_round_trip_eff": 0.90,

        # Financing regime
        "financing_regime": "traditional",  # or "eaas"

        # Tariff (USD per TWh)
        "solar_service_tariff_usd_per_twh":
            solar_tariff_scenarios()[solar_tariff_case],

        # Required NPV margin
        "required_margin": 1.10,
        "peak_demand_multiple": 2.5,
        # ---- Carbon policy
        **carbon_policy_scenarios()[carbon_case],

        # ---- Labels (for reporting)
        "labels": {
            "demand_level": demand_level_case,
            "demand": demand_case,
            "carbon": carbon_case,
            "gas_deliverability": gas_deliverability_case,
            "land": land_case,
            "solar_build": solar_build_case,
        },
    }

    return scenario
