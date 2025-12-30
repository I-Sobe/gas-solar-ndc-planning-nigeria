"""
Scenario Definitions Module (Configuration Only)

Scope
-----
Defines deterministic planning scenarios as structured parameter
dictionaries for use in system evaluation and optimization studies.

This module provides scenario DEFINITIONS ONLY.
It does NOT execute models, run experiments, or aggregate results.

"""

import numpy as np


# ============================================================
# PLANNING HORIZON
# ============================================================

def planning_horizon(start_year=2025, end_year=2045):
    """
    Define the planning horizon.

    Returns
    -------
    np.ndarray
        Array of simulation years (annual resolution)
    """
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")
    return np.arange(start_year, end_year + 1)


# ============================================================
# SCENARIO REGISTRIES
# ============================================================

def demand_growth_scenarios():
    """Annual electricity demand growth assumptions."""
    return {
        "low": 0.025,
        "baseline": 0.04,
        "high": 0.06,
    }


def gas_decline_scenarios():
    """Gas-field decline rate assumptions (physical depletion only)."""
    return {
        "low_decline": 0.03,
        "baseline": 0.06,
        "high_decline": 0.09,
    }


def solar_capacity_scenarios():
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


def carbon_policy_scenarios():
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
    demand_case="baseline",
    gas_case="baseline",
    solar_case="baseline",
    carbon_case="moderate_policy",
    start_year=2025,
    end_year=2045,
):
    """
    Construct a deterministic planning scenario.

    Returns
    -------
    dict
        Scenario parameter dictionary consumed by downstream
        evaluation and optimization modules.
    """

    # ---- Validate labels
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

        # ---- Demand
        "base_demand_twh": 30.0,
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
            "demand": demand_case,
            "gas": gas_case,
            "solar": solar_case,
            "carbon": carbon_case,
        },
    }

    return scenario
