"""
Scenario Manager

Defines scenario dictionaries and automates execution of deterministic
scenario matrices (e.g., decline levels, carbon prices, EaaS cases).

Functions:
    load_scenario(name)
    run_all_scenarios()
"""
"""
scenarios.py
Centralized scenario and assumption definitions
"""

import numpy as np


# ----------------------------
# TIME HORIZON
# ----------------------------
def planning_horizon(
    start_year=2025,
    end_year=2045
):
    """
    Define planning horizon.

    Returns
    -------
    np.ndarray
        Array of simulation years
    """
    return np.arange(start_year, end_year + 1)


# ----------------------------
# DEMAND SCENARIOS
# ----------------------------
def demand_growth_scenarios():
    """
    Annual electricity demand growth assumptions.

    Returns
    -------
    dict
    """
    return {
        "low": 0.025,
        "baseline": 0.04,
        "high": 0.06,
    }


# ----------------------------
# GAS SUPPLY SCENARIOS
# ----------------------------
def gas_decline_scenarios():
    """
    Gas-field decline rate assumptions.

    Returns
    -------
    dict
    """
    return {
        "low_decline": 0.03,
        "baseline": 0.06,
        "high_decline": 0.09,
    }


# ----------------------------
# SOLAR BUILD-OUT SCENARIOS
# ----------------------------
def solar_capacity_scenarios():
    """
    Solar PV capacity expansion assumptions.

    Returns
    -------
    dict
        initial_capacity_mw, annual_addition_mw
    """
    return {
        "slow": {
            "initial_capacity_mw": 500,
            "annual_addition_mw": 200,
        },
        "baseline": {
            "initial_capacity_mw": 500,
            "annual_addition_mw": 400,
        },
        "accelerated": {
            "initial_capacity_mw": 500,
            "annual_addition_mw": 700,
        },
    }


# ----------------------------
# CARBON PRICE SCENARIOS
# ----------------------------
def carbon_price_cases():
    """
    Carbon policy stance definitions.

    These map to stochastic distributions later.

    Returns
    -------
    dict
    """
    return {
        "no_policy": {
            "active": False,
        },
        "moderate_policy": {
            "active": True,
            "multiplier": 1.0,
        },
        "stringent_policy": {
            "active": True,
            "multiplier": 1.5,
        },
    }


