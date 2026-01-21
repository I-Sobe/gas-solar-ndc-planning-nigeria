"""
Electricity Demand Module (Planning-Level)

Scope
-----
Provides national-scale baseline electricity demand projections and
a simple electrification/access-based adjustment suitable for
long-term planning studies with annual time-step resolution.

This module DOES NOT model Energy-as-a-Service (EaaS), demand response,
or behavioral demand reduction. EaaS effects are handled elsewhere
via supply substitution, financing, and deployment pathways.

Modeling assumptions
--------------------
- Annual time-step resolution
- Constant exponential growth for baseline demand
- Electricity demand scales linearly with access percentage
- No price elasticity or sectoral disaggregation
- Base-year demand is scenario input and may represent served demand or reconstructed latent demand; 
    this module does not infer demand from historical time series.

Non-scope
---------
- Demand response or efficiency programs
- Customer-level behavioral modeling
- Explicit EaaS demand-side interventions
"""

import numpy as np
from src.utils import assert_non_negative


def project_baseline_demand(
    base_demand,
    growth_rate,
    start_year,
    end_year,
):
    """
    Project baseline electricity demand using constant growth.
        Parameters
    ----------
    base_demand : float
        Base-year demand input (TWh/year). 
            This may be served demand (lower bound) or reconstructed latent demand, depending on the selected demand scenario.
    growth_rate : float
        Annual demand growth rate (fraction, e.g. 0.04).
        Must be >= -1.0.
    start_year : int
    end_year : int
    Returns
    -------
    dict
        {
            "years": np.ndarray,
            "demand": np.ndarray  # TWh/year
        }
    """

    assert_non_negative([base_demand], "base_demand")

    if growth_rate < -1.0:
        raise ValueError(
            "growth_rate < -1.0 implies negative demand and is invalid"
        )

    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")

    years = np.arange(start_year, end_year + 1)
    t = years - start_year

    demand = base_demand * (1 + growth_rate) ** t

    assert_non_negative(demand, "demand")

    return {
        "years": years,
        "demand": demand,
    }


def apply_access_adjustment(
    demand,
    access_series,
    base_access,
):
    """
    Adjust electricity demand based on changes in electricity access.
    Interpretation
    --------------
    Demand is assumed to scale linearly with the proportion of the
    population (or load) that has electricity access. This function
    represents electrification-driven demand growth, not efficiency
    improvements or demand reduction.
    IMPORTANT:
    ----------
    - `access_series` and `base_access` must use the SAME units:
        * both fractions (0–1), or
        * both percentages (0–100)
    - No saturation, rebound, or elasticity effects are modeled.
    Parameters
    ----------
    demand : array-like
        Baseline electricity demand (TWh/year)
    access_series : array-like
        Electricity access over time (fraction or %)
    base_access : float
        Electricity access level in the base year
        (same units as access_series)
    Returns
    -------
    np.ndarray
        Access-adjusted electricity demand (TWh/year)
    """

    demand = np.asarray(demand, dtype=float)
    access_series = np.asarray(access_series, dtype=float)

    # ---- Dimensional validation
    if len(demand) != len(access_series):
        raise ValueError(
            "demand and access_series must have identical length"
        )

    assert_non_negative(demand, "demand")
    assert_non_negative(access_series, "access_series")

    if base_access <= 0:
        raise ValueError("base_access must be positive")

    # ---- Unit consistency check
    if np.max(access_series) <= 1.0 and base_access > 1.0:
        raise ValueError(
            "access_series appears to be fractional but base_access is not"
        )
    if np.max(access_series) > 1.0 and base_access <= 1.0:
        raise ValueError(
            "access_series appears to be percentage but base_access is fractional"
        )

    adjustment_factor = access_series / base_access

    return demand * adjustment_factor
