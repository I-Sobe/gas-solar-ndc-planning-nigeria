"""
Dispatch Module

Defines power balance constraints, gas generation limits,
solar dispatch, storage charge/discharge, and unmet load.

Used by the Pyomo/PyPSA optimization framework to ensure
system feasibility at each timestep.

Functions:
    power_balance(...)
    gas_generation_constraints(...)
    build_dispatch_model(params)
"""
"""
dispatch.py
System-level energy dispatch and balance
"""

import numpy as np
from src.utils import validate_non_negative


def dispatch_energy(
    years,
    demand,
    gas_generation,
    solar_generation,
    storage=None
):
    """
    Dispatch electricity supply to meet demand.

    Parameters
    ----------
    years : array-like
        Simulation years
    demand : array-like
        Electricity demand (TWh)
    gas_generation : array-like
        Gas-fired generation (TWh)
    solar_generation : array-like
        Solar generation (TWh)
    storage : BatteryStorage or None
        Optional storage system

    Returns
    -------
    dict
        {
            "served": np.ndarray,
            "unserved": np.ndarray,
            "storage_soc": np.ndarray,
            "excess": np.ndarray
        }
    """

    demand = np.array(demand)
    gas_generation = np.array(gas_generation)
    solar_generation = np.array(solar_generation)

    validate_non_negative(demand, "demand")
    validate_non_negative(gas_generation, "gas_generation")
    validate_non_negative(solar_generation, "solar_generation")

    n = len(years)

    served = np.zeros(n)
    unserved = np.zeros(n)
    excess = np.zeros(n)
    storage_soc = np.zeros(n)

    for t in range(n):
        supply = gas_generation[t] + solar_generation[t]

        if supply >= demand[t]:
            # All demand served
            served[t] = demand[t]
            excess_energy = supply - demand[t]

            if storage is not None:
                # Convert TWh → MWh for storage interface
                stored = storage.charge(excess_energy * 1e6)
                excess[t] = excess_energy - stored / 1e6
                storage_soc[t] = storage.soc
            else:
                excess[t] = excess_energy

        else:
            # Demand shortfall
            served[t] = supply
            deficit = demand[t] - supply

            if storage is not None:
                delivered = storage.discharge(deficit * 1e6)
                served[t] += delivered / 1e6
                unserved[t] = demand[t] - served[t]
                storage_soc[t] = storage.soc
            else:
                unserved[t] = deficit

    return {
        "served": served,
        "unserved": unserved,
        "storage_soc": storage_soc,
        "excess": excess,
    }
