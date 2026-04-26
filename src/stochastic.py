"""
Stochastic Sensitivity Analysis Module (Exploratory)

Scope
-----
Implements exploratory Monte Carlo sensitivity analysis for a
limited set of exogenous uncertainties in the national gas–solar
planning model.

This module perturbs selected scenario parameters and repeatedly
runs the deterministic system evaluation to assess robustness of
aggregate outcomes (e.g. total system cost).

Modeled uncertainties
---------------------
- Electricity demand growth rate
- Gas supply regime (discrete deliverability scenarios)
- Carbon price level (exogenous)

Non-scope
---------
- Stochastic optimization or chance constraints
- Decision-variable distributions
- Endogenous policy response
- Correlated uncertainties
- Energy-as-a-Service (EaaS) logic

Important
---------
Results are intended as robustness and risk diagnostics, not as
probabilistic forecasts or optimal stochastic decisions.
"""

import copy
import numpy as np

from src.optimize_experiments import run_deterministic_scenario
from src.scenarios import gas_probability_weights

# ============================================================
# UNCERTAINTY SAMPLING
# ============================================================

def sample_uncertainties(
    N,
    base_scenario,
    carbon_mu,
    carbon_sigma,
    demand_sigma=0.01,
    seed=None,
):
    """
    Generate Monte Carlo samples of exogenous uncertainties.

    Parameters
    ----------
    N : int
        Number of Monte Carlo samples
    base_scenario : dict
        Deterministic scenario dictionary
    carbon_mu : float
        Mean of lognormal carbon price distribution (log-space)
    carbon_sigma : float
        Standard deviation of lognormal carbon price distribution
    demand_sigma : float, optional
        Standard deviation of demand growth perturbation
    seed : int or None

    Returns
    -------
    list of dict
        Independent scenario realizations
    """

    if seed is not None:
        np.random.seed(seed)

    samples = []

    for _ in range(N):
        # ---- Deep copy to avoid cross-sample contamination
        scenario = copy.deepcopy(base_scenario)

        # ---- Demand growth uncertainty (truncated normal)
        scenario["demand_growth"] = max(
            0.0,
            np.random.normal(
                base_scenario["demand_growth"],
                demand_sigma,
            ),
        )

        # ---- Gas supply uncertainty (scenario regime sampling)
        gas_probs = gas_probability_weights()

        labels = list(gas_probs.keys())
        weights = list(gas_probs.values())

        scenario["gas_scenario"] = np.random.choice(labels, p=weights)        
        
        # ---- Carbon price uncertainty (lognormal, exogenous)
        if scenario["carbon_active"]:
            scenario["carbon_price"] = np.random.lognormal(
                mean=carbon_mu,
                sigma=carbon_sigma,
            )

        samples.append(scenario)

    return samples


# ============================================================
# STOCHASTIC EXECUTION (DETERMINISTIC WRAPPER)
# ============================================================

def run_stochastic_simulation(
    base_scenario,
    econ,
    capacity_paths,
    carbon_mu,
    carbon_sigma,
    N=500,
    seed=None,
):
    """
    Run Monte Carlo sensitivity analysis using deterministic evaluation.

    Parameters
    ----------
    base_scenario : dict
        Baseline deterministic scenario
    econ : dict
        Economic parameters passed through unchanged
    carbon_mu : float
    carbon_sigma : float
    N : int
        Number of Monte Carlo samples
    seed : int or None

    Returns
    -------
    np.ndarray
        Total system cost outcomes (USD)
    """

    samples = sample_uncertainties(
        N=N,
        base_scenario=base_scenario,
        carbon_mu=carbon_mu,
        carbon_sigma=carbon_sigma,
        seed=seed,
    )

    outcomes = []
    
    for i, scenario in enumerate(samples):
        output = run_deterministic_scenario(
            scenario=scenario,
            econ=econ,
            capacity_paths=capacity_paths,
        )
        outcomes.append(
            (scenario["gas_scenario"], output["costs"]["total"])
        )
        
    return outcomes


# ============================================================
# RISK METRICS
# ============================================================

def compute_risk_metrics(values, alpha=0.95):
    """
    Compute basic risk metrics from Monte Carlo outcomes.

    Parameters
    ----------
    values : array-like
        Scalar outcomes (e.g. total cost)
    alpha : float
        Confidence level for VaR / CVaR

    Returns
    -------
    dict
        {
            "expected": float,
            "variance": float,
            "VaR": float,
            "CVaR": float
        }
    """

    values = np.asarray(values, dtype=float)

    expected = np.mean(values)
    variance = np.var(values)
    var = np.quantile(values, alpha)
    cvar = values[values >= var].mean()

    return {
        "expected": expected,
        "variance": variance,
        "VaR": var,
        "CVaR": cvar,
    }
