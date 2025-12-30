"""
Stochastic Simulation Module

Implements Monte Carlo sampling for uncertain variables including
gas price, carbon price, solar CAPEX, and demand growth. Wraps the
deterministic optimization model to generate risk-adjusted metrics.

Functions:
    sample_uncertainties(N, distributions)
    run_stochastic_optimization(base_model, samples)
    compute_risk_metrics(results)
"""
"""
stochastic.py
Carbon price stochastic modeling
"""

import numpy as np
import pandas as pd
from src.utils import validate_non_negative


"""
stochastic.py
Monte Carlo simulation and risk analysis wrapper
"""

import numpy as np
import pandas as pd

from src.optimize import run_deterministic_model


# -------------------------------------------------
# UNCERTAINTY SAMPLING
# -------------------------------------------------
def sample_uncertainties(
    N,
    base_scenario,
    carbon_mu,
    carbon_sigma,
    demand_sigma=0.01,
    gas_sigma=0.01,
    seed=None,
):
    """
    Sample uncertain parameters for Monte Carlo simulation.
    """

    if seed is not None:
        np.random.seed(seed)

    samples = []

    for _ in range(N):
        scenario = base_scenario.copy()

        # Demand growth uncertainty
        scenario["demand_growth"] = max(
            0.0,
            np.random.normal(
                base_scenario["demand_growth"],
                demand_sigma,
            ),
        )

        # Gas decline uncertainty
        scenario["gas_decline"] = max(
            0.0,
            np.random.normal(
                base_scenario["gas_decline"],
                gas_sigma,
            ),
        )

        # Carbon price uncertainty
        if scenario["carbon_policy"]["active"]:
            scenario["carbon_policy"] = scenario["carbon_policy"].copy()
            scenario["carbon_policy"]["price"] = np.random.lognormal(
                mean=carbon_mu,
                sigma=carbon_sigma,
            )
        else:
            scenario["carbon_policy"]["price"] = 0.0

        samples.append(scenario)

    return samples


