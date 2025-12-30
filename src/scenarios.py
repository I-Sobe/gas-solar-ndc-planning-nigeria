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


