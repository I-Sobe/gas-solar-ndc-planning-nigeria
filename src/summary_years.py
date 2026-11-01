"""
Year-labelled summary quantities (TASK 53 2.4; invariant G1-2, register entry on
unlabelled anchors, TASK 50 V12).

The anchor's headline numbers were written under names that carry no year
(`final_storage_capacity_mwh` is the 2054 value, `solar_total_built_mw` and
`cumulative_unserved_twh` run to 2054, i.e. include the end-effect buffer). Every
such quantity now carries the year or window it is taken from, and every value
taken at the last model year carries its REPORT_END_YEAR (2045) counterpart
beside it. The last model year is read from `years`, never hard-coded in a name
that could go stale, so a shorter horizon changes the key rather than lying under
the old one.
"""

REPORT_END_YEAR = 2045


def _upto(series_by_year, last_year):
    return float(sum(v for y, v in series_by_year.items() if int(y) <= last_year))


def labelled_decision_totals(years, solar_add_by_year, storage_capacity_by_year, gas_add_by_year=None):
    """
    years: iterable of model years. *_by_year: {int year: value}.
    Returns {name: value} with the last model year and REPORT_END_YEAR versions of
    cumulative solar, (optionally) cumulative gas and the storage capacity stock.
    """
    ys = [int(y) for y in years]
    end = ys[-1]
    out = {}
    for tag, last in ((str(end), end), (str(REPORT_END_YEAR), REPORT_END_YEAR)):
        if last > end:
            continue
        out[f"solar_cumulative_built_{tag}_mw"] = _upto(solar_add_by_year, last)
        if gas_add_by_year is not None:
            out[f"gas_cumulative_built_{tag}_mw"] = _upto(gas_add_by_year, last)
        out[f"storage_capacity_{tag}_mwh"] = float(storage_capacity_by_year[last])
    return out


def labelled_period_totals(years, unserved_by_year, emissions_by_year):
    """Cumulative unserved energy (TWh) and emissions (tCO2) over 2025..end and 2025..REPORT_END_YEAR."""
    ys = [int(y) for y in years]
    start, end = ys[0], ys[-1]
    out = {}
    for last in sorted({end, min(end, REPORT_END_YEAR)}):
        out[f"cumulative_unserved_{start}_{last}_twh"] = _upto(unserved_by_year, last)
        out[f"cumulative_emissions_{start}_{last}_tco2"] = _upto(emissions_by_year, last)
    return out
