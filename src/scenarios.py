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
def solar_capacity_scenarios() -> dict[str, dict[str, float]]:
    """
    Solar PV capacity expansion assumptions.
    """
    return {
        "slow": {
            "solar_baseline_mw": 500,
        },
        "baseline": {
            "solar_baseline_mw": 500,
        },
        "accelerated": {
            "solar_baseline_mw": 500,
        },
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
    """
    return {
        "downside": "downside",
        "baseline": "baseline",
        "upside": "upside",
        "shock_recovery": "shock_recovery",
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
    B_star = 6_130_600_000

    return {
        "tight": int(0.50 * B_star),
        "moderate": int(0.85 * B_star),
        "adequacy": int(1.00 * B_star),
        "expansion": int(1.20 * B_star),
        "unconstrained": None,
    }

def solar_tariff_scenarios():

    return {
        "low": 45_000_000,
        "baseline": 65_000_000,
        "high": 85_000_000,
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
    solar_case: str = "baseline",
    solar_build_case: str = "baseline",
    solar_tariff_case= "baseline",
    carbon_case: str = "no_policy",
    start_year: int = 2025,
    end_year: int = 2045,
) -> dict:
    """
    Construct a deterministic planning scenario.

    Parameters
    ----------
    demand_level_case : str
        Base-year demand level label. One of:
        {"served", "latent_low", "latent_high"}.
    demand_case : str
        Demand growth label. One of demand_growth_scenarios() keys.
    gas_case : str
        Gas decline label. One of gas_decline_scenarios() keys.
    gas_deliverability_case : str
        Gas deliverability label. One of gas_deliverability_scenarios() keys.
    solar_case : str
        Solar capacity pathway label. One of solar_capacity_scenarios() keys.
    carbon_case : str
        Carbon policy label. One of carbon_policy_scenarios() keys.
    start_year : int
    end_year : int

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
    if solar_case not in solar_capacity_scenarios():
        raise ValueError(f"Unknown solar_case: {solar_case}")
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
        
        # ---- Gas CAPEX
        "gas_capex_per_mw": 900000,

        # ---- Land policy
        "land_available_km2": land_scenarios()[land_case],
        "land_intensity_solar_km2_per_mw": 0.025,
        "land_intensity_gas_km2_per_mw": 0.001,
        "land_intensity_storage_km2_per_mwh": 0.00001,
        # ---- Solar
        "solar_cf": 0.27,
        **solar_capacity_scenarios()[solar_case],
        "solar_max_build_mw_per_year": solar_build_scenarios()[solar_build_case],
        # ---- Public capital constraint (NPV, USD)
        "public_solar_budget_npv": capital_envelope_scenarios()[capital_case],
        
        # ---- Storage reduced-form (optimization; annual, energy-neutral)
        # Equivalent full cycles/year (dimensionless). Typical planning proxy: 150–350.
        "storage_cycles_per_year": 250.0,
        # Fraction of annual solar energy that is "surplus" and can be shifted by storage.
        # Conservative: 0.10, baseline: 0.20, optimistic: 0.30
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

        # ---- Carbon policy
        **carbon_policy_scenarios()[carbon_case],

        # ---- Labels (for reporting)
        "labels": {
            "demand_level": demand_level_case,
            "demand": demand_case,
            "solar": solar_case,
            "carbon": carbon_case,
            "gas_deliverability": gas_deliverability_case,
            "land": land_case,
            "solar_build": solar_build_case,
        },
    }

    return scenario
