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


