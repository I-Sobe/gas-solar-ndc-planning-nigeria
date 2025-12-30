"""
Energy-as-a-Service (EaaS) System Transformation Module

Purpose
-------
Implements Energy-as-a-Service (EaaS) as a system-level coordination
and financing paradigm for national-scale energy planning.

EaaS is modeled as a transformation layer that:
- introduces alternative deployment pathways for distributed solar
  and storage assets, and
- converts upfront capital investment into annual service payments.

EaaS is treated as an interpretive and structural extension to the
core gas–solar system model, consistent with a planning-level
Energy-as-a-Service ecosystem framing.

Implemented Phases
------------------
Phase 1: Deployment lever (capacity rollout)
Phase 2: Financing lever (CAPEX → service payments)

Deferred Phases
---------------
Phase 3: Reliability substitution (explicitly not modeled)
"""

import numpy as np
from src.utils import assert_non_negative


# ============================================================
# CONSTANTS (PLANNING-LEVEL)
# ============================================================

HOURS_PER_YEAR = 8760


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _capital_recovery_factor(rate: float, n_years: int) -> float:
    """
    Capital Recovery Factor (CRF).

    CRF = r (1+r)^n / ((1+r)^n - 1)

    Notes
    -----
    - rate = 0 implies straight-line amortization (1 / n_years)
    - Negative rates are not permitted
    """
    if rate < 0:
        raise ValueError("service_wacc must be non-negative")
    if n_years <= 0:
        raise ValueError("n_years must be positive")

    if rate == 0.0:
        return 1.0 / n_years

    return (rate * (1 + rate) ** n_years) / ((1 + rate) ** n_years - 1)


def _validate_eaas_deployment_params(eaas_params: dict) -> None:
    """Validate EaaS deployment parameters."""
    penetration_target = eaas_params["penetration_target"]
    solar_share = eaas_params["solar_share"]
    solar_cf = eaas_params["solar_cf"]

    if not (0.0 <= penetration_target <= 1.0):
        raise ValueError("penetration_target must be in [0, 1]")
    if not (0.0 <= solar_share <= 1.0):
        raise ValueError("solar_share must be in [0, 1]")
    if not (0.0 < solar_cf <= 1.0):
        raise ValueError("solar_cf must be in (0, 1]")

    assert_non_negative(
        [eaas_params["ramp_years"], eaas_params["storage_hours"]],
        "EaaS deployment parameters",
    )


# ============================================================
# PHASE 1: DEPLOYMENT LEVER
# ============================================================

def compute_eaas_deployment(
    years,
    demand_twh,
    eaas_params,
):
    """
    Compute EaaS-driven deployment of distributed solar and storage.

    Interpretation
    --------------
    EaaS penetration is used as a proxy to SIZE distributed capacity.
    It does NOT reduce total electricity demand and does NOT imply
    demand-side efficiency or substitution.

    Parameters
    ----------
    years : array-like
        Planning horizon (annual resolution)
    demand_twh : array-like
        Baseline electricity demand (TWh/year)
    eaas_params : dict
        Required keys:
            - penetration_target : float (0–1)
            - ramp_years : int
            - start_year : int
            - solar_share : float (0–1)
            - storage_hours : float
            - solar_cf : float (0–1)

    Returns
    -------
    dict
        Deployment trajectories and units
    """

    _validate_eaas_deployment_params(eaas_params)

    years = np.asarray(years)
    demand_twh = np.asarray(demand_twh, dtype=float)

    if years.shape[0] != demand_twh.shape[0]:
        raise ValueError("years and demand_twh must have identical length")

    n = len(years)

    penetration_target = eaas_params["penetration_target"]
    ramp_years = eaas_params["ramp_years"]
    start_year = eaas_params["start_year"]
    solar_share = eaas_params["solar_share"]
    storage_hours = eaas_params["storage_hours"]
    solar_cf = eaas_params["solar_cf"]

    # ---- Coverage trajectory (piecewise linear ramp)
    coverage_fraction = np.zeros(n)
    for i, year in enumerate(years):
        if year < start_year:
            coverage_fraction[i] = 0.0
        elif year >= start_year + ramp_years:
            coverage_fraction[i] = penetration_target
        else:
            coverage_fraction[i] = (
                penetration_target * (year - start_year + 1) / ramp_years
            )

    # ---- Energy proxy used for sizing (TWh/year)
    eaas_energy_twh = coverage_fraction * demand_twh

    # ---- Solar sizing
    solar_energy_twh = eaas_energy_twh * solar_share
    solar_capacity_mw = (
        solar_energy_twh * 1e6 / (solar_cf * HOURS_PER_YEAR)
    )

    # ---- Storage sizing (energy-based)
    storage_capacity_mwh = solar_capacity_mw * storage_hours

    return {
        "additional_solar_mw": solar_capacity_mw,
        "additional_storage_mwh": storage_capacity_mwh,
        "coverage_fraction": coverage_fraction,
        "units": {
            "additional_solar_mw": "MW",
            "additional_storage_mwh": "MWh",
            "coverage_fraction": "fraction",
        },
    }


# ============================================================
# PHASE 2: FINANCING LEVER
# ============================================================

def compute_eaas_financing(
    years,
    eaas_deployment,
    eaas_params,
    system_costs,
):
    """
    Compute annual EaaS service payments from deployed capacity.

    Interpretation
    --------------
    Converts incremental EaaS CAPEX into annual service payments
    using a capital recovery formulation.

    Accounting mode is retained as metadata only and does NOT
    affect numerical results.
    """

    years = np.asarray(years)
    n = len(years)

    solar_mw = np.asarray(eaas_deployment["additional_solar_mw"])
    storage_mwh = np.asarray(eaas_deployment["additional_storage_mwh"])

    if not (len(solar_mw) == len(storage_mwh) == n):
        raise ValueError("EaaS deployment arrays must match horizon length")

    service_wacc = eaas_params["service_wacc"]
    tenor = eaas_params["contract_tenor_years"]
    accounting_mode = eaas_params["accounting_mode"]

    if tenor <= 0:
        raise ValueError("contract_tenor_years must be positive")
    if accounting_mode not in ("off_balance", "on_balance"):
        raise ValueError("accounting_mode must be 'off_balance' or 'on_balance'")

    solar_capex_per_mw = system_costs["solar_capex_per_mw"]
    storage_capex_per_mwh = system_costs["storage_capex_per_mwh"]

    assert_non_negative(
        [solar_capex_per_mw, storage_capex_per_mwh],
        "EaaS CAPEX parameters",
    )

    # ---- Incremental capacity additions
    delta_solar_mw = np.diff(
        np.insert(solar_mw, 0, 0.0)
    )
    delta_storage_mwh = np.diff(
        np.insert(storage_mwh, 0, 0.0)
    )

    delta_solar_mw = np.maximum(delta_solar_mw, 0.0)
    delta_storage_mwh = np.maximum(delta_storage_mwh, 0.0)

    delta_capex = (
        delta_solar_mw * solar_capex_per_mw
        + delta_storage_mwh * storage_capex_per_mwh
    )

    # ---- Service payment stream
    crf = _capital_recovery_factor(service_wacc, tenor)
    annual_service_cost = np.zeros(n)

    for t in range(n):
        if delta_capex[t] <= 0:
            continue

        annual_payment = delta_capex[t] * crf
        annual_service_cost[t : min(t + tenor, n)] += annual_payment

    return {
        "annual_service_cost_usd": annual_service_cost,
        "accounting_mode": accounting_mode,
        "units": {
            "annual_service_cost_usd": "USD/year",
        },
    }
