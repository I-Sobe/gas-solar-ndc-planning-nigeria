"""
Solar PV Module

Processes solar irradiance datasets and converts them into
capacity factor arrays or generation time series for the
optimization model.

Functions:
    load_solar_data(filepath)
    compute_capacity_factor(irradiance, derate=0.8)
    solar_generation(capacity, capacity_factor)
"""
"""
solar.py
Solar photovoltaic electricity generation model
"""

import numpy as np
from src.utils import validate_non_negative


HOURS_PER_YEAR = 8760


def solar_generation(
    capacity_mw,
    capacity_factor
):
    """
    Compute annual solar electricity generation.

    Parameters
    ----------
    capacity_mw : float or array-like
        Installed solar capacity (MW)
    capacity_factor : float or array-like
        Capacity factor (0–1)

    Returns
    -------
    float or np.ndarray
        Annual electricity generation (TWh)
    """

    capacity_mw = np.array(capacity_mw)
    capacity_factor = np.array(capacity_factor)

    validate_non_negative(capacity_mw, "capacity_mw")
    validate_non_negative(capacity_factor, "capacity_factor")

    generation_mwh = capacity_mw * capacity_factor * HOURS_PER_YEAR
    generation_twh = generation_mwh / 1e6

    return generation_twh


def solar_capacity_trajectory(
    initial_capacity_mw,
    annual_addition_mw,
    start_year,
    end_year
):
    """
    Construct a linear solar capacity build-out trajectory.

    Parameters
    ----------
    initial_capacity_mw : float
        Installed capacity in start_year (MW)
    annual_addition_mw : float
        New capacity added per year (MW/year)
    start_year : int
    end_year : int

    Returns
    -------
    dict
        {
            "years": np.ndarray,
            "capacity_mw": np.ndarray
        }
    """

    validate_non_negative([initial_capacity_mw], "initial_capacity_mw")
    validate_non_negative([annual_addition_mw], "annual_addition_mw")

    years = np.arange(start_year, end_year + 1)
    t = years - start_year

    capacity = initial_capacity_mw + annual_addition_mw * t

    validate_non_negative(capacity, "capacity")

    return {
        "years": years,
        "capacity_mw": capacity
    }
