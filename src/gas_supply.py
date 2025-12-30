"""
Gas Supply Module

Implements reservoir decline modeling using Arps equations
(exponential and hyperbolic) to generate annual gas availability
profiles for system-level optimization under supply constraints.

Functions:
    arps_exponential(qi, Di, t)
    arps_hyperbolic(qi, Di, b, t)
    generate_gas_supply_profile(params, years)
"""

"""
gas_supply.py
Gas-fired electricity supply under decline constraints
"""

import numpy as np
from src.utils import assert_non_negative


#from utils import assert_non_negative

def arps_exponential_decline(
    q0,
    decline_rate,
    years
):
    """
    Arps exponential decline model.

    Parameters
    ----------
    q0 : float
        Initial gas-based electricity generation (TWh)
    decline_rate : float
        Annual decline rate (fraction, e.g. 0.06)
    years : array-like
        Year indices (e.g. [0,1,2,...])

    Returns
    -------
    np.ndarray
        Gas generation per year (TWh)
    """

    assert_non_negative([q0], "q0")

    t = np.array(years)
    generation = q0 * np.exp(-decline_rate * t)

    assert_non_negative(generation, "gas generation")

    return generation


def gas_generation_cap(
    q0,
    decline_rate,
    start_year,
    end_year
):
    """
    Convenience wrapper to generate gas supply trajectory.

    Parameters
    ----------
    q0 : float
        Initial generation (TWh)
    decline_rate : float
    start_year : int
    end_year : int

    Returns
    -------
    dict
        {
            "years": [...],
            "generation": [...]
        }
    """

    years = np.arange(start_year, end_year + 1)
    t = years - start_year

    generation = arps_exponential_decline(
        q0=q0,
        decline_rate=decline_rate,
        years=t
    )

    return {
        "years": years,
        "generation": generation
    }
