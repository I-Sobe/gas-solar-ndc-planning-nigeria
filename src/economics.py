"""
Economic Evaluation Utilities

Contains financial discounting and Net Present Value calculations.

All functions here represent explicit economic assumptions
used in techno-economic evaluation.
"""


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
