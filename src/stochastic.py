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



