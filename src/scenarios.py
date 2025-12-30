"""
Scenario Manager

Defines scenario dictionaries and automates execution of deterministic
scenario matrices (e.g., decline levels, carbon prices, EaaS cases).

Functions:
    load_scenario(name)
    run_all_scenarios()
"""
"""
scenarios.py
Scenario Manager
Defines scenario dictionaries and automates execution of deterministic
scenario matrices (decline levels, carbon policies, solar build-out cases).
"""

import itertools
import pandas as pd
import numpy as np


from optimize_model import run_deterministic_model


# ----------------------------
# BASE SCENARIO DEFINITIONS
# ----------------------------
def planning_horizon(start_year=2025, end_year=2045):
    """
    Define planning horizon.
    Returns
    -------
    np.ndarray
        Array of simulation years
    """
    return np.arange(start_year, end_year + 1)


def demand_growth_scenarios():
    """
    Annual electricity demand growth assumptions.
    Returns
    -------
    dict
    """
    return {
        "low": 0.025,
        "baseline": 0.04,
        "high": 0.06,
    }


def gas_decline_scenarios():
    """
    Gas-field decline rate assumptions.
    Returns
    -------
    dict
    """
    return {
        "low_decline": 0.03,
        "baseline": 0.06,
        "high_decline": 0.09,
    }


def solar_capacity_scenarios():
    """
    Solar PV capacity expansion assumptions.
    Returns
    -------
    dict
        initial_capacity_mw, annual_addition_mw
    """
    return {
        "slow": {
            "initial_capacity_mw": 500,
            "annual_addition_mw": 200,
        },
        "baseline": {
            "initial_capacity_mw": 500,
            "annual_addition_mw": 400,
        },
        "accelerated": {
            "initial_capacity_mw": 500,
            "annual_addition_mw": 700,
        },
    }


def carbon_price_cases():
    """
    Carbon policy stance definitions.
    These map to stochastic distributions later.
    Returns
    -------
    dict
    """
    return {
        "no_policy": {
            "active": False,
        },
        "moderate_policy": {
            "active": True,
            "multiplier": 1.0,
        },
        "stringent_policy": {
            "active": True,
            "multiplier": 1.5,
        },
    }


# ----------------------------
# SCENARIO BUILDER / LOADER
# ----------------------------
def load_scenario(
    demand_case="baseline",
    gas_case="baseline",
    solar_case="baseline",
    carbon_case="moderate_policy",
    start_year=2025,
    end_year=2045,
):
    """
    Load a single deterministic scenario configuration.
    """

    years = planning_horizon(start_year, end_year)

    scenario = {
        "years": years,
        "demand_growth": demand_growth_scenarios()[demand_case],
        "gas_decline": gas_decline_scenarios()[gas_case],
        "solar": solar_capacity_scenarios()[solar_case],
        "carbon_policy": carbon_price_cases()[carbon_case],
        "labels": {
            "demand": demand_case,
            "gas": gas_case,
            "solar": solar_case,
            "carbon": carbon_case,
        },
    }

    return scenario



# -------------------------------------------------
# SCENARIO MATRIX EXECUTION
# -------------------------------------------------
def run_all_scenarios(
    start_year=2025,
    end_year=2045,
):
    """
    Execute deterministic model across all scenario combinations.

    Returns
    -------
    pd.DataFrame
        Summary results for all scenarios
    """

    results = []

    demand_cases = demand_growth_scenarios().keys()
    gas_cases = gas_decline_scenarios().keys()
    solar_cases = solar_capacity_scenarios().keys()
    carbon_cases = carbon_price_cases().keys()

    for (
        demand_case,
        gas_case,
        solar_case,
        carbon_case,
    ) in itertools.product(
        demand_cases,
        gas_cases,
        solar_cases,
        carbon_cases,
    ):

        scenario = load_scenario(
            demand_case=demand_case,
            gas_case=gas_case,
            solar_case=solar_case,
            carbon_case=carbon_case,
            start_year=start_year,
            end_year=end_year,
        )

        output = run_deterministic_model(scenario)

        results.append({
            "demand_case": demand_case,
            "gas_case": gas_case,
            "solar_case": solar_case,
            "carbon_case": carbon_case,
            "total_cost": output["costs"]["total"],
            "gas_cost": output["costs"]["gas"],
            "solar_cost": output["costs"]["solar"],
            "carbon_cost": output["costs"]["carbon"],
            "unserved_cost": output["costs"]["unserved"],
            "total_unserved_energy": np.sum(output["unserved"]),
        })

    return pd.DataFrame(results)
