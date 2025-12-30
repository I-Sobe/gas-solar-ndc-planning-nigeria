"""
Economic Evaluation Utilities

Contains:
- financial discounting and Net Present Value calculations.
- deterministic and planning-level cost calculations.

All functions are stateless and scenario-driven.

All functions here represent explicit economic assumptions
used in techno-economic evaluation.
"""

import numpy as np

def gas_cost(gas_gen_twh, gas_cost_per_twh):
    return np.sum(gas_gen_twh) * gas_cost_per_twh


def solar_capex(capacity_mw, solar_capex_per_mw, n_years):
    """
    Planning-level annualized solar capital cost.
    """
    return np.sum(capacity_mw) * solar_capex_per_mw / n_years


def carbon_cost(gas_gen_twh, carbon_price, emission_factor):
    """
    Carbon cost from gas-fired generation.
    """
    return (
        np.sum(gas_gen_twh)
        * 1e6
        * emission_factor
        * carbon_price
    )


def unserved_energy_cost(unserved_twh, penalty_per_twh):
    return np.sum(unserved_twh) * penalty_per_twh


def total_system_cost(cost_components):
    """
    Aggregate system cost from a dict of components.
    """
    return sum(cost_components.values())

def discount_factor(rate, year):
    """
    Compute discount factor for a given year.

    Parameters
    ----------
    rate : float
        Discount rate (e.g. 0.1 for 10%)
    year : int
        Year index (0-based)

    Returns
    -------
    float
    """
    return 1.0 / ((1.0 + rate) ** year)


def npv(cashflows, discount_rate):
    """
    Compute Net Present Value of a cashflow stream.

    Parameters
    ----------
    cashflows : array-like
        Cashflows indexed by year
    discount_rate : float

    Returns
    -------
    float
    """
    return sum(
        cf * discount_factor(discount_rate, t)
        for t, cf in enumerate(cashflows)
    )
