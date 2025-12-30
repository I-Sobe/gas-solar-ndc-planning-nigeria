"""
Demand Module

Handles national baseline demand projections and integrates
Energy-as-a-Service (EaaS) deployment scenarios as demand-side
modifiers. Produces net demand time series for system optimization.

Functions:
    load_baseline_demand(filepath)
    apply_eaas_intervention(demand, scenario_params)
    get_net_demand(profile, eaas_params)
"""
"""
demand.py
Electricity demand projection model
"""

import numpy as np
from src.utils import assert_non_negative



def project_baseline_demand(
    base_demand,
    growth_rate,
    start_year,
    end_year
):
    """
    Project baseline electricity demand using constant growth.

    Parameters
    ----------
    base_demand : float
        Electricity demand in start_year (TWh)
    growth_rate : float
        Annual demand growth rate (fraction, e.g. 0.04)
    start_year : int
    end_year : int

    Returns
    -------
    dict
        {
            "years": np.ndarray,
            "demand": np.ndarray
        }
    """

    assert_non_negative([base_demand], "base_demand")

    years = np.arange(start_year, end_year + 1)
    t = years - start_year

    demand = base_demand * (1 + growth_rate) ** t

    assert_non_negative(demand, "demand")

    return {
        "years": years,
        "demand": demand
    }


def apply_access_adjustment(
    demand,
    access_series,
    base_access
):
    """
    Adjust demand based on electricity access.

    Parameters
    ----------
    demand : array-like
        Baseline demand (TWh)
    access_series : array-like
        Electricity access over time (%)
    base_access : float
        Electricity access in base year (%)

    Returns
    -------
    np.ndarray
        Access-adjusted demand (TWh)
    """

    assert_non_negative(demand, "demand")
    assert_non_negative(access_series, "access_series")

    adjustment_factor = np.array(access_series) / base_access

    return np.array(demand) * adjustment_factor
