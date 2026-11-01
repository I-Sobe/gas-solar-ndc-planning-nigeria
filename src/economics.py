"""
Economic Evaluation Utilities

Contains:
- Financial discounting and Net Present Value calculations.
- Deterministic and planning-level cost calculations.

All functions are stateless and scenario-driven.

All functions here represent explicit economic assumptions
used in techno-economic evaluation.
"""

import numpy as np


# ============================================================
# Cost Components
# ============================================================

def gas_cost(gas_gen_twh, gas_cost_per_twh):
    """
    Total gas fuel cost.

    Parameters
    ----------
    gas_gen_twh : array-like
        Gas generation in TWh.
    gas_cost_per_twh : float
        Gas cost in USD per TWh.

    Returns
    -------
    float
    """
    return np.sum(gas_gen_twh) * gas_cost_per_twh


def solar_capex(capacity_mw, solar_capex_per_mw, n_years):
    """
    Undiscounted average annual solar capital cost (total CAPEX / n_years).

    DOC-CODE PARITY (plan 5.5): this is a straight arithmetic mean, not
    an annuity -- there is no discount rate or time-value-of-money
    treatment, so "annualized" (the prior wording) overstated what the
    arithmetic does. Not called by either Pyomo LP builder
    (`src/optimize_model.py`, `src/optimize_model_sliced.py`); the only
    call site is `src/optimize_experiments.py`'s
    `run_deterministic_scenario()`, a separate non-Pyomo dispatch-
    simulation helper. The LP builders compute solar capital cost via
    their own CRF/overnight-capex machinery under `capital_treatment_case`,
    not this function.

    Parameters
    ----------
    capacity_mw : array-like
        Installed solar capacity in MW.
    solar_capex_per_mw : float
        CAPEX in USD per MW.
    n_years : int
        Planning horizon length.

    Returns
    -------
    float
    """
    return np.sum(capacity_mw) * solar_capex_per_mw / n_years


def carbon_cost(gas_gen_twh, carbon_price, emission_factor):
    """
    Carbon cost from gas-fired generation.

    Parameters
    ----------
    gas_gen_twh : array-like
        Gas generation in TWh.
    carbon_price : float
        Carbon price in USD per tCO2.
    emission_factor : float
        Emission factor in tCO2 per MWh_e.

    Returns
    -------
    float
    """
    return (
        np.sum(gas_gen_twh)
        * 1e6  # Convert TWh to MWh
        * emission_factor
        * carbon_price
    )


def unserved_energy_cost(unserved_twh, penalty_per_twh):
    """
    Cost of unserved energy.

    Parameters
    ----------
    unserved_twh : array-like
        Unserved energy in TWh.
    penalty_per_twh : float
        Penalty in USD per TWh.

    Returns
    -------
    float
    """
    return np.sum(unserved_twh) * penalty_per_twh


def total_system_cost(cost_components):
    """
    Aggregate system cost from a dictionary of components.

    Parameters
    ----------
    cost_components : dict
        Dictionary of cost components.

    Returns
    -------
    float
    """
    return sum(cost_components.values())


# ============================================================
# Financial Utilities
# ============================================================

def discount_factor(rate, year):
    """
    Compute discount factor for a given year.

    Parameters
    ----------
    rate : float
        Discount rate (e.g. 0.10 for 10%).
    year : int
        Year index (0-based).

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
        Cashflows indexed by year.
    discount_rate : float
        Discount rate.

    Returns
    -------
    float
    """
    return sum(
        cf * discount_factor(discount_rate, t)
        for t, cf in enumerate(cashflows)
    )
