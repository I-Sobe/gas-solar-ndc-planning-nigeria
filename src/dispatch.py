"""
Deterministic Energy-Balance Dispatch (Accounting-Level)

Scope
-----
Implements a deterministic, greedy, annual energy-balance accounting
routine used to evaluate supply–demand feasibility and unserved energy
after capacity planning decisions are made.

This is NOT an optimization dispatch model and does NOT construct Pyomo
constraints or decision variables.

Modeling assumptions
--------------------
- Annual time-step resolution (no intra-annual dispatch)
- Merit-order supply: gas + solar → storage → unmet load
- Storage modeled as a stateful energy buffer with internal SOC
- No ramping, reserve, or power-flow constraints
- Intended for ex post feasibility and reliability assessment
- Energy-balance is annual; storage operations represent simplified intertemporal shifting and are not chronological dispatch.

Boundary discipline
-------------------
- The `storage` object is MUTATED during execution (SOC evolves).
- Callers must provide a fresh storage instance per scenario run.
- This function is order-dependent and not safe for parallel reuse
  of the same storage object.

Non-scope
---------
- Chronological (hourly) dispatch
- Unit commitment or operational optimization
- Network or transmission modeling
"""

import numpy as np
from src.utils import assert_non_negative


def dispatch_energy(
    years,
    demand,
    gas_generation,
    solar_generation,
    storage=None,
):
    """
    Perform deterministic energy-balance dispatch.

    Parameters
    ----------
    years : array-like
        Simulation years (annual index)
    demand : array-like
        Electricity demand (TWh/year)
    gas_generation : array-like
        Available gas-fired generation (TWh/year)
    solar_generation : array-like
        Available solar generation (TWh/year)
    storage : BatteryStorage or None, optional
        Optional storage system (stateful, mutated in-place)

    Returns
    -------
    dict
        {
            "served": np.ndarray,       # TWh/year
            "unserved": np.ndarray,     # TWh/year
            "excess": np.ndarray,       # TWh/year
            "storage_soc": np.ndarray,  # MWh (end-of-year SOC)
            "units": dict
        }

    Notes
    -----
    - Storage state-of-charge (SOC) is tracked end-of-year.
    - If storage is None, SOC is returned as zeros.
    """

    demand = np.asarray(demand, dtype=float)
    gas_generation = np.asarray(gas_generation, dtype=float)
    solar_generation = np.asarray(solar_generation, dtype=float)

    # ---- Defensive validation
    assert_non_negative(demand, "demand")
    assert_non_negative(gas_generation, "gas_generation")
    assert_non_negative(solar_generation, "solar_generation")

    n = len(years)
    if not (len(demand) == len(gas_generation) == len(solar_generation) == n):
        raise ValueError(
            "years, demand, gas_generation, and solar_generation "
            "must have the same length"
        )

    served = np.zeros(n)
    unserved = np.zeros(n)
    excess = np.zeros(n)
    storage_soc = np.zeros(n)

    for t in range(n):
        supply = gas_generation[t] + solar_generation[t]

        if supply >= demand[t]:
            served[t] = demand[t]
            excess_energy = supply - demand[t]

            if storage is not None:
                # Convert TWh → MWh for storage interface
                stored_mwh = storage.charge(excess_energy * 1e6)
                excess[t] = excess_energy - stored_mwh / 1e6
                storage_soc[t] = storage.soc
            else:
                excess[t] = excess_energy

        else:
            served[t] = supply
            deficit = demand[t] - supply

            if storage is not None:
                delivered_mwh = storage.discharge(deficit * 1e6)
                served[t] += delivered_mwh / 1e6
                unserved[t] = demand[t] - served[t]
                storage_soc[t] = storage.soc
            else:
                unserved[t] = deficit

    return {
        "served": served,
        "unserved": unserved,
        "excess": excess,
        "storage_soc": storage_soc,
        "units": {
            "energy": "TWh",
            "storage_soc": "MWh",
        },
    }
