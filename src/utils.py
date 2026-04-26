"""
Utility Functions

Lightweight, dependency-safe helper functions shared across modules.

Scope:
- Numerical validation
- Simple interpolation
- Unit conversion
- Index helpers

Non-scope (by design):
- Economic logic
- Optimization logic
- Scenario configuration
- High-level I/O orchestration
"""

import numpy as np


def interpolate_series(series, method="linear"):
    """
    Interpolate missing values in a 1D numeric series.

    Notes
    -----
    - Only linear interpolation is supported.
    - No extrapolation beyond data bounds is performed.

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


def mw_to_twh(power_mw, hours=8760):
    """
    Convert average power (MW) to energy (TWh).

    Parameters
    ----------
    power_mw : float or array-like
        Average power level in MW
    hours : int, optional
        Number of hours (default: 8760)

    Returns
    -------
    float or array-like
        Energy in TWh
    """
    return power_mw * hours / 1e6


def assert_non_negative(array, name="value"):
    """
    Assert that all elements in an array are non-negative.

    Parameters
    ----------
    array : array-like
    name : str

    Raises
    ------
    ValueError
        If any element is negative
    """
    arr = np.asarray(array)
    if np.any(arr < 0):
        raise ValueError(f"{name} contains negative values")


def year_index(start_year, end_year):
    """
    Generate an inclusive year index.

    Parameters
    ----------
    start_year : int
    end_year : int

    Returns
    -------
    np.ndarray
    """
    return np.arange(start_year, end_year + 1)


def json_safe(obj):
    """Recursively convert NumPy types to Python natives for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_safe(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj