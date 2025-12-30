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


def load_carbon_price_data(filepath):
    """
    Load cleaned carbon price reference data.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["carbon_price"])
    validate_non_negative(df["carbon_price"], "carbon_price")
    return df


def fit_lognormal(carbon_prices):
    """
    Fit lognormal distribution parameters.

    Parameters
    ----------
    carbon_prices : array-like

    Returns
    -------
    tuple
        (mu, sigma)
    """
    prices = np.array(carbon_prices)
    validate_non_negative(prices, "carbon_prices")

    log_prices = np.log(prices)
    mu = np.mean(log_prices)
    sigma = np.std(log_prices)

    return mu, sigma


def simulate_carbon_prices(
    mu,
    sigma,
    years,
    n_sims=1000,
    seed=None
):
    """
    Generate Monte Carlo carbon price paths.

    Parameters
    ----------
    mu : float
        Mean of log prices
    sigma : float
        Std dev of log prices
    years : array-like
        Simulation years
    n_sims : int
        Number of Monte Carlo simulations
    seed : int or None

    Returns
    -------
    np.ndarray
        Shape (n_sims, n_years)
    """

    if seed is not None:
        np.random.seed(seed)

    n_years = len(years)

    prices = np.random.lognormal(
        mean=mu,
        sigma=sigma,
        size=(n_sims, n_years)
    )

    return prices



