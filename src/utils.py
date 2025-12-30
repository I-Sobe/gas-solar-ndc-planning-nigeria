"""
Utility Functions

Shared helpers for data processing, interpolation, logging, and
plot formatting across modules.

Functions:
    load_yaml(filepath)
    interpolate_series(series, method='linear')
    save_results(output, filepath)
"""
"""
utils.py
Shared utilities and helper functions
"""

import numpy as np
import yaml
import os


def load_yaml(filepath):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    filepath : str
        Path to YAML file

    Returns
    -------
    dict
        Parsed YAML contents
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"YAML file not found: {filepath}")

    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def interpolate_series(series, method="linear"):
    """
    Interpolate missing values in a 1D numeric series.

    Parameters
    ----------
    series : array-like
        Input data with possible NaNs
    method : str
        Interpolation method ('linear' only supported)

    Returns
    -------
    np.ndarray
        Interpolated series
    """
    series = np.asarray(series, dtype=float)

    if method != "linear":
        raise NotImplementedError(
            "Only linear interpolation is supported for thesis scope"
        )

    if np.all(np.isfinite(series)):
        return series

    x = np.arange(len(series))
    mask = np.isfinite(series)

    if mask.sum() < 2:
        raise ValueError("Insufficient data points for interpolation")

    return np.interp(x, x[mask], series[mask])


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


def mw_to_twh(power_mw, hours=8760):
    """
    Convert MW to TWh.

    Parameters
    ----------
    power_mw : float or array
    hours : int

    Returns
    -------
    float or array
    """
    return power_mw * hours / 1e6


def validate_non_negative(array, name="value"):
    """
    Raise error if any element is negative.
    """
    if np.any(np.array(array) < 0):
        raise ValueError(f"{name} contains negative values")


def year_index(start_year, end_year):
    """
    Generate year index array.

    Returns
    -------
    np.ndarray
    """
    return np.arange(start_year, end_year + 1)


def save_results(output, filepath):
    """
    Save model results to disk.

    Parameters
    ----------
    output : dict or array-like
        Model results to persist
    filepath : str
        Output file path ('.npz' or '.npy')

    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if isinstance(output, dict):
        np.savez(filepath, **output)
    else:
        np.save(filepath, output)
