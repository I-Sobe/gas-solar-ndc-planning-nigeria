"""
Scenario Definitions Module (Configuration Only)

Scope
-----
Defines deterministic planning scenarios as structured parameter
dictionaries for use in system evaluation and optimization studies.

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


def gas_decline_scenarios() -> dict[str, float]:
    """Gas-field decline rate assumptions (physical depletion only)."""
    return {
        "low_decline": 0.03,
        "baseline": 0.06,
        "high_decline": 0.09,
    }


def solar_capacity_scenarios() -> dict[str, dict[str, float]]:
    """Solar PV capacity expansion assumptions."""
    return {
        "slow": {
            "solar_baseline_mw": 500,
            "solar_addition_mw": 200,
        },
        "baseline": {
            "solar_baseline_mw": 500,
            "solar_addition_mw": 400,
        },
        "accelerated": {
            "solar_baseline_mw": 500,
            "solar_addition_mw": 700,
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


# ============================================================
# SCENARIO CONSTRUCTOR
# ============================================================

def load_scenario(
    demand_level_case: str = "served",
    demand_case: str = "baseline",
    gas_case: str = "baseline",
    solar_case: str = "baseline",
    carbon_case: str = "moderate_policy",
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
    if gas_case not in gas_decline_scenarios():
        raise ValueError(f"Unknown gas_case: {gas_case}")
    if solar_case not in solar_capacity_scenarios():
        raise ValueError(f"Unknown solar_case: {solar_case}")
    if carbon_case not in carbon_policy_scenarios():
        raise ValueError(f"Unknown carbon_case: {carbon_case}")

    years = planning_horizon(start_year, end_year)

    scenario = {
        # ---- Temporal
        "years": years,

        # ---- Demand (planning-level, annual)
        "base_demand_twh": demand_level_scenarios()[demand_level_case],
        "demand_growth": demand_growth_scenarios()[demand_case],

        # ---- Gas supply
        "gas_q0_twh": 40.0,
        "gas_decline": gas_decline_scenarios()[gas_case],

        # ---- Solar
        "solar_cf": 0.22,
        **solar_capacity_scenarios()[solar_case],

        # ---- Storage (operational diagnostics)
        "storage_mwh": 20_000,
        "storage_mw": 2_000,
        "storage_eff": 0.9,

        # ---- Carbon policy
        **carbon_policy_scenarios()[carbon_case],

        # ---- Labels (for reporting)
        "labels": {
            "demand_level": demand_level_case,
            "demand": demand_case,
            "gas": gas_case,
            "solar": solar_case,
            "carbon": carbon_case,
        },
    }

    return scenario
