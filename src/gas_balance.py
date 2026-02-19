"""
Gas Accounting Utilities (Planning-Level)

Purpose
-------
Implements annual gas availability (energy-in-gas) projection and
gas-to-power conversion to electricity.

Internal gas unit
-----------------
- TWh_th (thermal energy content of gas)

Interfaces
----------
- project_gas_availability_th(): q0_th + decline -> annual gas availability (TWh_th/year)
- gas_to_power_electricity_twh(): converts gas thermal energy -> electricity energy (TWh/year)

Scope discipline
----------------
- No field-level reservoir modeling here.
- No export pricing here (Phase 2).
- Pure numeric utilities (no Pyomo constructs).
"""

import numpy as np
from src.utils import assert_non_negative


def project_gas_availability_th(
    q0_th,
    decline_rate,
    start_year,
    end_year,
):
    """
    Project annual gas availability in TWh_th/year via exponential decline.

    Parameters
    ----------
    q0_th : float
        Base-year gas availability (TWh_th/year)
    decline_rate : float
        Annual decline rate (fraction). Must be >= -1.0.
    start_year : int
    end_year : int

    Returns
    -------
    dict
        {
            "years": np.ndarray,
            "gas_available_th": np.ndarray  # TWh_th/year
        }
    """
    assert_non_negative([q0_th], "q0_th")

    if decline_rate < -1.0:
        raise ValueError("decline_rate < -1.0 implies negative availability")

    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")

    years = np.arange(start_year, end_year + 1)
    t = years - start_year
    gas_available_th = q0_th * (1.0 + decline_rate) ** t

    assert_non_negative(gas_available_th, "gas_available_th")

    return {"years": years, "gas_available_th": gas_available_th}


def gas_to_power_electricity_twh(gas_to_power_th, eta_gas_to_power):
    """
    Convert gas thermal energy used for power to electricity energy.

    Parameters
    ----------
    gas_to_power_th : array-like
        Gas allocated to power (TWh_th/year)
    eta_gas_to_power : float
        Gas-to-power conversion efficiency (0 < eta <= 1)

    Returns
    -------
    np.ndarray
        Electricity from gas (TWh/year)
    """
    if not (0.0 < eta_gas_to_power <= 1.0):
        raise ValueError("eta_gas_to_power must be in (0, 1]")

    gas_to_power_th = np.asarray(gas_to_power_th, dtype=float)
    assert_non_negative(gas_to_power_th, "gas_to_power_th")

    return eta_gas_to_power * gas_to_power_th
