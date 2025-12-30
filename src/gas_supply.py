"""
Gas Supply Module (Planning-Level Physical Decline)

Scope
-----
Implements Arps decline-curve models to generate annual gas-to-power
supply availability profiles under physical depletion constraints.
This module represents physical decline only.

Modeling assumptions
--------------------
- Annual time-step resolution
- Representative maturing gas field(s)
- Decline applies to electricity-equivalent gas supply (TWh/year)
- No new field development, infill drilling, or compression
- No policy, allocation, pricing, or uncertainty logic (handled elsewhere)

Supported decline forms
-----------------------
- Exponential (b = 0)
- Hyperbolic (0 < b < 1)

Notes
-----
- Decline rates are assumed non-negative.
- Hyperbolic decline asymptotically approaches zero production.

Non-scope
---------
- Field development optimization
- Domestic gas allocation or export trade-offs
- Price formation or stochastic sampling
"""

import numpy as np
from src.utils import assert_non_negative


def arps_exponential_decline(q0, decline_rate, t):
    """
    Arps exponential decline model.

    q(t) = q0 * exp(-D * t)

    Parameters
    ----------
    q0 : float
        Initial gas-to-power supply (TWh/year)
    decline_rate : float
        Annual decline rate D (fraction, >= 0)
    t : array-like
        Time index (years since start)

    Returns
    -------
    np.ndarray
        Annual gas supply (TWh/year)
    """

    assert_non_negative([q0], "q0")
    assert_non_negative([decline_rate], "decline_rate")

    t = np.asarray(t, dtype=float)
    generation = q0 * np.exp(-decline_rate * t)

    assert_non_negative(generation, "gas generation")
    return generation


def arps_hyperbolic_decline(q0, decline_rate, b, t):
    """
    Arps hyperbolic decline model.

    q(t) = q0 / (1 + b * D * t)^(1/b)

    Parameters
    ----------
    q0 : float
        Initial gas-to-power supply (TWh/year)
    decline_rate : float
        Initial decline rate D (fraction, >= 0)
    b : float
        Hyperbolic exponent (0 < b < 1)
    t : array-like
        Time index (years since start)

    Returns
    -------
    np.ndarray
        Annual gas supply (TWh/year)

    Notes
    -----
    Production declines asymptotically toward zero as t increases.
    """

    assert_non_negative([q0], "q0")
    assert_non_negative([decline_rate], "decline_rate")

    if not (0 < b < 1):
        raise ValueError("Hyperbolic exponent b must be in (0, 1)")

    t = np.asarray(t, dtype=float)
    generation = q0 / (1.0 + b * decline_rate * t) ** (1.0 / b)

    assert_non_negative(generation, "gas generation")
    return generation


def generate_gas_supply_profile(
    q0,
    decline_rate,
    start_year,
    end_year,
    model="exponential",
    b=None,
):
    """
    Generate annual gas supply availability profile.

    Parameters
    ----------
    q0 : float
        Initial gas-to-power supply (TWh/year)
    decline_rate : float
        Decline rate parameter (>= 0)
    start_year : int
    end_year : int
    model : str
        Decline model: 'exponential' or 'hyperbolic'
    b : float or None
        Hyperbolic exponent (required if model='hyperbolic')

    Returns
    -------
    dict
        {
            "years": np.ndarray,
            "generation": np.ndarray  # TWh/year
        }
    """

    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")

    years = np.arange(start_year, end_year + 1)
    t = years - start_year

    if model == "exponential":
        generation = arps_exponential_decline(
            q0=q0,
            decline_rate=decline_rate,
            t=t,
        )
    elif model == "hyperbolic":
        if b is None:
            raise ValueError("Parameter b must be provided for hyperbolic decline")
        generation = arps_hyperbolic_decline(
            q0=q0,
            decline_rate=decline_rate,
            b=b,
            t=t,
        )
    else:
        raise ValueError("model must be 'exponential' or 'hyperbolic'")

    return {
        "years": years,
        "generation": generation,
    }


# ---- Backward-compatible convenience wrapper
def gas_generation_cap(q0, decline_rate, start_year, end_year):
    """
    Backward-compatible wrapper for exponential Arps decline.

    NOTE
    ----
    This function returns an annual gas SUPPLY PROFILE,
    not a capacity constraint. Prefer generate_gas_supply_profile(...)
    for new development.
    """

    return generate_gas_supply_profile(
        q0=q0,
        decline_rate=decline_rate,
        start_year=start_year,
        end_year=end_year,
        model="exponential",
    )
