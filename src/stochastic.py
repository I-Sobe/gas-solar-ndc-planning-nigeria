"""
Stochastic Sensitivity Analysis Module (Fixed-Plan Re-Dispatch)

Scope
-----
Monte Carlo robustness analysis of the LP planning model under exogenous
uncertainty. The method is FIXED-PLAN RE-DISPATCH (two-stage logic):

  1. Solve the LP once at the base scenario to obtain the optimal capacity
     plan (the here-and-now decision).
  2. Freeze that plan and re-solve the LP under each Monte Carlo draw, so only
     the operational/dispatch variables adjust (the recourse).

Because every draw is evaluated with the SAME objective, discounting, and cost
stack as the deterministic runs, the resulting cost distribution is directly
comparable to the deterministic NPV. (This supersedes the earlier approach that
routed draws through a separate undiscounted simulator.)

Modeled uncertainties
---------------------
- Electricity demand growth rate (truncated normal)
- Gas deliverability regime (discrete, sampled from gas_probability_weights)
- Carbon price level (lognormal, only if carbon_active)

Scope note on EaaS
------------------
EaaS is representable: if the base scenario uses financing_regime="eaas", the
fixed plan carries EaaS solar, and its robustness is evaluated like any other
plan. (The earlier "EaaS out of scope" note no longer applies.)

Limitations
-----------
- This measures the robustness of a plan optimized for the BASE case; it does
  NOT find the plan that is optimal under uncertainty (that would require
  stochastic programming with recourse). State this explicitly in the thesis.
- Regime probabilities are a sensitivity prior, not a forecast (see
  gas_probability_weights). Report symmetric-weight results alongside.
"""

import copy
import numpy as np
import pyomo.environ as pyo

from src.scenarios import gas_probability_weights
from src.optimize_model import build_model, solve_model


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
    Generate N independent Monte Carlo scenario realizations.

    Parameters
    ----------
    N : int
        Number of samples.
    base_scenario : dict
        Base scenario dictionary (deep-copied per sample).
    carbon_mu, carbon_sigma : float
        Log-space mean and sigma of the lognormal carbon price (applied only
        when the base scenario has carbon_active=True).
    demand_sigma : float, optional
        Std dev of the demand-growth perturbation (absolute, on the growth
        fraction). Surface this to the runner so it is reportable; default 0.01
        (i.e. +/- ~1 percentage point on the growth rate).
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    list[dict]
    """
    if seed is not None:
        np.random.seed(seed)

    gas_probs = gas_probability_weights()
    labels = list(gas_probs.keys())
    weights = list(gas_probs.values())

    samples = []
    for _ in range(N):
        scenario = copy.deepcopy(base_scenario)

        # Demand growth (truncated at 0)
        scenario["demand_growth"] = max(
            0.0,
            np.random.normal(base_scenario["demand_growth"], demand_sigma),
        )

        # Gas deliverability regime
        scenario["gas_scenario"] = np.random.choice(labels, p=weights)

        # Carbon price (only if active)
        if scenario.get("carbon_active", False):
            scenario["carbon_price"] = np.random.lognormal(
                mean=carbon_mu, sigma=carbon_sigma,
            )

        samples.append(scenario)

    return samples


# ============================================================
# STOCHASTIC EXECUTION (FIXED-PLAN RE-DISPATCH)
# ============================================================

def run_stochastic_simulation(
    base_scenario,
    econ,
    carbon_mu,
    carbon_sigma,
    solar_capex_tv,
    caps=None,
    N=500,
    seed=None,
    demand_sigma=0.01,
):
    """
    Fixed-plan re-dispatch Monte Carlo.

    Returns
    -------
    list[tuple(str, float)]
        (gas_scenario_label, system_cost_npv_usd) for each optimal draw.
        Infeasible draws are dropped and counted (a warning is printed);
        infeasibility under a fixed plan is itself a reliability-risk finding.
    """
    # ---- 1. Base solve -> optimal plan (guard against infeasible base)
    m_base = build_model(
        scenario=base_scenario, econ=econ,
        emissions_cap_by_year=caps,
        solar_capex_by_year=solar_capex_tv,
    )
    base_status = solve_model(m_base)
    if not base_status["optimal"]:
        raise RuntimeError(
            "Base scenario solve was not optimal "
            f"(status={base_status['status']}); cannot extract a fixed plan. "
            "Fix base feasibility before running the Monte Carlo."
        )

    T = range(len(base_scenario["years"]))
    fixed_plan = {
        "solar_public_add": {t: pyo.value(m_base.solar_public_add[t]) for t in T},
        "solar_eaas_add":   {t: pyo.value(m_base.solar_eaas_add[t]) for t in T},
        "storage_add":      {t: pyo.value(m_base.storage_add[t]) for t in T},
    }

    # ---- 2. Sample uncertainties
    samples = sample_uncertainties(
        N, base_scenario, carbon_mu, carbon_sigma,
        demand_sigma=demand_sigma, seed=seed,
    )

    # ---- 3. Re-dispatch the fixed plan under each draw
    outcomes = []
    n_infeasible = 0
    for scenario in samples:
        m = build_model(
            scenario=scenario, econ=econ,
            emissions_cap_by_year=caps,
            solar_capex_by_year=solar_capex_tv,
            fixed_plan=fixed_plan,
        )
        status = solve_model(m)
        if status["optimal"]:
            outcomes.append(
                (scenario["gas_scenario"], float(pyo.value(m.system_cost_npv)))
            )
        else:
            n_infeasible += 1

    if n_infeasible > 0:
        print(
            f"WARNING: {n_infeasible}/{N} draws infeasible under the fixed plan "
            "-- risk metrics computed on survivors only. The infeasible share "
            "is itself a reliability-risk result worth reporting."
        )

    return outcomes


# ============================================================
# RISK METRICS
# ============================================================

def compute_risk_metrics(values, alpha=0.95):
    """
    Basic risk metrics from Monte Carlo cost outcomes.

    Parameters
    ----------
    values : array-like of float
    alpha  : float
        Confidence level for VaR / CVaR (upper tail, since higher cost = worse).

    Returns
    -------
    dict with keys: expected, variance, VaR, CVaR.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("No outcomes to compute risk metrics on (all draws infeasible?).")

    expected = float(np.mean(values))
    variance = float(np.var(values))
    var = float(np.quantile(values, alpha))
    tail = values[values >= var]
    cvar = float(tail.mean()) if tail.size else var

    return {
        "expected": expected,
        "variance": variance,
        "VaR": var,
        "CVaR": cvar,
    }
