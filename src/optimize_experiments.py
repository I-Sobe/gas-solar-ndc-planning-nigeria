"""
Optimization experiments and orchestration.

Scope
-----
- Deterministic operational scenario evaluation
- Pareto frontier generation (weighted-sum, ε-constraint)

Notes
-----
This module orchestrates experiments only.
Planning optimization and operational simulation intentionally
use different storage abstractions and are not numerically
comparable at the component level.
"""
import json
import itertools
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import copy

from src.gas_supply import gas_available_power
from src.optimize_model import build_model, solve_model
from src.io import load_solar_capex_by_year # Load time-varying solar CAPEX 
from src.economics import (
    gas_cost,
    solar_capex,
    carbon_cost,
    unserved_energy_cost,
    total_system_cost,
)
from src.solar import solar_generation
from src.storage import BatteryStorage
from src.dispatch import dispatch_energy
from src.demand import project_baseline_demand
from src.scenarios import (
    load_scenario,
    demand_level_scenarios,
    demand_growth_scenarios,
    gas_deliverability_scenarios,
    solar_build_scenarios,
    carbon_policy_scenarios,
)
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def write_run_outputs(run_id, diagnostics, summary):
    """
    Writes:
      - results/<run_id>_timeseries.csv
      - results/<run_id>_summary.json
    """
    ts_df = diagnostics_to_timeseries_df(diagnostics)
    ts_path = RESULTS_DIR / f"{run_id}_timeseries.csv"
    ts_df.to_csv(ts_path, index=False)

    js_path = RESULTS_DIR / f"{run_id}_summary.json"
    with open(js_path, "w") as f:
        json.dump(summary, f, indent=2)


def assert_duals_present(m, scenario, require_gas_duals=True, require_carbon_duals=False):
    """
    Fail fast if expected duals are missing.
    This is thesis-grade hygiene: you either get interpretable duals or you do not report them.
    """
    years = scenario["years"]

    if require_gas_duals:
        missing = []
        for t, y in enumerate(years):
            if m.dual.get(m.gas_balance[t], None) is None:
                missing.append(int(y))
        if missing:
            raise RuntimeError(
                f"Missing gas_balance duals for years: {missing}. "
                "Solver did not return duals; cannot report gas scarcity shadow prices."
            )

    if require_carbon_duals and hasattr(m, "emissions_constraint"):
        # Annual caps (indexed) vs single cap (scalar)
        if m.emissions_constraint.is_indexed():
            missing = []
            for t, y in enumerate(years):
                if m.dual.get(m.emissions_constraint[t], None) is None:
                    missing.append(int(y))
            if missing:
                raise RuntimeError(
                    f"Missing emissions_constraint duals for years: {missing}. "
                    "Cannot report carbon shadow prices."
                )
        else:
            # Scalar cap
            if m.dual.get(m.emissions_constraint, None) is None:
                raise RuntimeError(
                    "Missing scalar emissions_constraint dual. Cannot report carbon shadow price."
                )
    return None


def diagnostics_to_timeseries_df(diagnostics):
    """
    Convert diagnostics dict into a flat timeseries DataFrame.
    """
    years = sorted(diagnostics["demand_twh_by_year"].keys())

    rows = []
    for y in years:
        rows.append({
            "year": y,
            "demand_twh": diagnostics["demand_twh_by_year"][y],
            "gas_to_power_twh_th": diagnostics["gas_to_power_twh_th_by_year"][y],
            "gas_generation_twh_e": diagnostics["gas_generation_twh_e_by_year"][y],
            "solar_generation_twh_e": diagnostics["solar_generation_twh_e_by_year"][y],
            "storage_discharge_twh_e": diagnostics["storage_discharge_twh_e_by_year"][y],
            "unserved_twh": diagnostics["unserved_twh_by_year"][y],
            "emissions_tco2": diagnostics["emissions_tco2_by_year"][y], 
            "gas_shadow_usd_per_twh_th": diagnostics["gas_shadow_price_usd_per_twh_th_by_year"][y],
            "carbon_shadow_usd_per_tco2": diagnostics["carbon_shadow_price_usd_per_tco2_by_year"][y],
            "discount_factor": diagnostics["discount_factor_by_year"][y],
        })

    return pd.DataFrame(rows)


def summarize_run(m, diagnostics, scenario):
    years = scenario["years"]

    cumulative_unserved = sum(diagnostics["unserved_twh_by_year"].values())
    cumulative_emissions = float(pyo.value(m.emissions))
    gas_shadow = diagnostics["gas_shadow_price_usd_per_twh_th_by_year"]
    positive_shadow = [v for v in gas_shadow.values() if v and v > 1e-6]

    return {
        "gas_binding_share": len(positive_shadow) / len(gas_shadow),
        "gas_shadow_avg_usd_per_twh_th":
            float(np.mean(positive_shadow)) if positive_shadow else 0.0,
        "gas_shadow_max_usd_per_twh_th":
            float(max(positive_shadow)) if positive_shadow else 0.0,
        "npv_total_cost_usd": float(pyo.value(m.system_cost_npv)),
        "cumulative_unserved_twh": cumulative_unserved,
        "cumulative_emissions_tco2": cumulative_emissions,
        "solar_total_built_mw":
            sum(
                float(pyo.value(m.solar_public_add[t]))
                + float(pyo.value(m.solar_eaas_add[t]))
                for t in range(len(years))
            ),
        "final_storage_capacity_mwh":
            float(pyo.value(m.storage_capacity_mwh[len(years)-1])),
        "solar_public_total_mw":
        sum(float(pyo.value(m.solar_public_add[t])) for t in range(len(years))),
        "gas_shadow_last_year":
        diagnostics["gas_shadow_price_usd_per_twh_th_by_year"][int(years[-1])],
        "solar_eaas_total_mw":
        sum(float(pyo.value(m.solar_eaas_add[t])) for t in range(len(years))),
    }
# ============================================================
# ANNUAL CAP OPTIMIZATION
# ============================================================

def run_annual_cap_case(
    scenario,
    econ,
    cap_scenario_name,
    cap_path="data/cost/processed/emissions_cap.csv",
):

    years = scenario["years"]
    start_year = int(years[0])
    end_year = int(years[-1])

    cap_df = pd.read_csv(cap_path)

    cap_df = cap_df[
        (cap_df["scenario"] == cap_scenario_name)
        & (cap_df["year"] >= start_year)
        & (cap_df["year"] <= end_year)
    ].copy()

    cap_df = cap_df.sort_values("year")

    emissions_cap_by_year = cap_df["cap_tco2"].astype(float).tolist()

    if len(emissions_cap_by_year) != len(years):
        raise ValueError(
            f"Cap series length {len(emissions_cap_by_year)} "
            f"!= number of model years {len(years)}"
        )

    m = build_model(
        scenario=scenario,
        econ=econ,
        emissions_cap_by_year=emissions_cap_by_year,
    )

    status = solve_model(m)
    
    if not status["optimal"]:
        raise RuntimeError(
            f"Optimization failed for annual cap scenario {cap_scenario_name}"
        )

    assert_duals_present(m, scenario,require_gas_duals=True, require_carbon_duals=True)

    diagnostics = extract_planning_diagnostics(m, scenario, econ)
    summary = summarize_run(m, diagnostics, scenario)

    run_id = f"annualcap_{cap_scenario_name}"
    write_run_outputs(run_id, diagnostics, summary)

    return {
        "cap_scenario": cap_scenario_name,
        "decision_variables": {
            "solar_public_add_by_year": {
                int(y): float(pyo.value(m.solar_public_add[t])) 
                for t,y in enumerate(years)
                },
            "solar_eaas_add_by_year": {
                int(y): float(pyo.value(m.solar_eaas_add[t]))
                for t, y in enumerate(years)
            },
            "solar_total_built_mw": sum(
                float(pyo.value(m.solar_public_add[t]))
                + float(pyo.value(m.solar_eaas_add[t]))
                for t in range(len(years))
            ),
            "final_storage_capacity_mwh": float(pyo.value(m.storage_capacity_mwh[len(years)-1])),
        },
        "actual_emissions_tco2_total": float(pyo.value(m.emissions)),
        "diagnostics": diagnostics,
    }

    
# ============================================================
# DETERMINISTIC OPERATIONAL SIMULATION
# ============================================================

def run_deterministic_scenario(scenario, econ, capacity_paths):

    years = scenario["years"]
    n_years = len(years)

    demand = project_baseline_demand(
        base_demand=scenario["base_demand_twh"],
        growth_rate=scenario["demand_growth"],
        start_year=years[0],
        end_year=years[-1],
    )["demand"]

    gas_profile = gas_available_power(
        start_year=years[0],
        end_year=years[-1],
        scenario_name=scenario["gas_scenario"],
    )

    eta =scenario.get("gas_eta", 0.43)
    gas_gen = gas_profile["available_twh_th"] * eta

    solar_gen = solar_generation(
        capacity_mw=capacity_paths["solar_mw"],
        capacity_factor=scenario["solar_cf"],
    )

    storage_duration_hours = 4.0

    storage = BatteryStorage(
        energy_capacity_mwh=capacity_paths["storage_mwh"][-1],
        power_capacity_mw=capacity_paths["storage_mwh"][-1] / storage_duration_hours,
        round_trip_efficiency=scenario["storage_round_trip_eff"],
    )

    dispatch = dispatch_energy(
        years=years,
        demand=demand,
        gas_generation=gas_gen,
        solar_generation=solar_gen,
        storage=storage,
    )

    costs = {
        "gas": gas_cost(gas_gen, econ["GAS_COST_PER_TWH_TH"]),
        "solar": solar_capex(
            capacity_paths["solar_mw"],
            econ["SOLAR_CAPEX_PER_MW"],
            n_years,
        ),
        "unserved": unserved_energy_cost(
            dispatch["unserved"],
            econ["UNSERVED_ENERGY_PENALTY"],
        ),
    }

    if scenario["carbon_active"]:
        costs["carbon"] = carbon_cost(
            gas_gen,
            scenario["carbon_price"],
            econ["CARBON_EMISSION_FACTOR"],
        )
    else:
        costs["carbon"] = 0.0

    costs["total"] = total_system_cost(costs)

    return {
        "served": dispatch["served"],
        "unserved": dispatch["unserved"],
        "costs": costs,
        "units": {
            "annual_energy": "TWh/year",
            "cumulative_energy": "TWh",
            "cost": "USD",
        },
    }

# ------------------------------------------------------------
# Reliability Constraint Extraction
# ------------------------------------------------------------
def reliability_levels_log():
    """
    Log-tightening reliability levels (fraction of unserved energy allowed).
    """
    return [0.20, 0.15, 0.10, 0.08, 0.06, 0.05]


def extract_reliability_dual(m, diag=None):
    """
    Returns the dual on the reliability constraint (USD per TWh of
    unserved energy allowed) and a binding flag.

    The dual is only economically meaningful when the constraint is
    binding (primal slack ≈ 0). A dual of zero with a slack constraint
    means the system meets the reliability target without being forced to
    — the constraint is redundant at that eps level.

    Parameters
    ----------
    m    : solved Pyomo model
    diag : diagnostics dict from extract_planning_diagnostics(), optional.
           If provided, used to compute primal slack for binding detection.

    Returns
    -------
    dict with keys:
        dual_usd_per_twh  : float or None — raw dual (USD per TWh unserved allowed)
        binding           : bool or None  — True if primal slack < 1e-4 TWh
        primal_slack_twh  : float or None — RHS - LHS of constraint
    """
    if not hasattr(m, "reliability_constraint"):
        return {
            "dual_usd_per_twh": None,
            "binding":          None,
            "primal_slack_twh": None,
        }

    # Extract raw dual
    if not m.reliability_constraint.is_indexed():
        # Scalar (total mode) — the standard REL-2 case
        raw = m.dual.get(m.reliability_constraint, None)
        dual_val = None if raw is None else max(0.0, float(-raw))

        # Primal slack: compute from diagnostics if available
        primal_slack = None
        if diag is not None:
            lhs = sum(diag["unserved_twh_by_year"].values())
            # RHS: eps * total_demand. Infer from constraint expression.
            # Access via Pyomo: m.reliability_constraint.expr
            try:
                rhs = float(pyo.value(m.reliability_constraint.ubound))
            except Exception:
                rhs = None
            if rhs is not None:
                primal_slack = rhs - lhs

        binding = None if primal_slack is None else (primal_slack < 1e-4)

    else:
        # Indexed (annual mode) — average across binding years
        vals = []
        for k in m.reliability_constraint:
            raw = m.dual.get(m.reliability_constraint[k], None)
            if raw is not None:
                vals.append(max(0.0, float(-raw)))
        dual_val = sum(vals) / len(vals) if vals else None
        primal_slack = None
        binding = None

    return {
        "dual_usd_per_twh": dual_val,
        "binding":          binding,
        "primal_slack_twh": primal_slack,
    }


def run_reliability_sweep(
    scenario,
    econ,
    gas_case=None,
    financing_label=None,
    policy_label=None,
    emissions_cap_by_year=None,
):
    """
    Runs:
    - Baseline (VoLL only, no reliability constraint)
    - Constrained reliability cases across eps_levels

    Parameters
    ----------
    scenario             : dict — fully configured scenario (includes financing_regime,
                           required_margin, tariff, capital_case, etc.)
    econ                 : dict
    gas_case             : str or None — gas regime label for tagging rows
    financing_label      : str or None — e.g. "public_only" or "eaas"
                           for tagging rows in REL-3 cross-arm comparisons
    policy_label         : str or None — e.g. "no_policy", "ndc3_unconditional"
    emissions_cap_by_year: list[float] or None — annual NDC emissions caps (tCO2).
                           If None, no emissions constraint is applied (no_policy).
                           Required for REL-3 NDC arm runs.

    Returns a list of result dicts, one per eps level (plus voll_only baseline).
    """
    eps_levels = reliability_levels_log()

    if gas_case is None:
        gas_case = scenario.get("labels", {}).get("gas_deliverability", "unknown")
    if financing_label is None:
        financing_label = scenario.get("financing_regime", "traditional")
    if policy_label is None:
        policy_label = "ndc" if emissions_cap_by_year is not None else "no_policy"

    results = []

    # Total horizon demand — for normalised dual and primal slack
    from src.demand import project_baseline_demand
    demand_arr = project_baseline_demand(
        base_demand=scenario["base_demand_twh"],
        growth_rate=scenario["demand_growth"],
        start_year=scenario["years"][0],
        end_year=scenario["years"][-1],
    )["demand"]
    total_demand_twh = float(sum(demand_arr))

    solar_capex_tv = load_solar_capex_by_year(
        scenario_name="solar_low",
        start_year=2025,
        end_year=2045,
    )

    def _build_and_solve(eps):
        """Build model with optional NDC cap and reliability constraint."""
        kwargs = dict(
            scenario=scenario,
            econ=econ,
        )
        if emissions_cap_by_year is not None:
            kwargs["emissions_cap_by_year"] = emissions_cap_by_year
        else:
            kwargs["emissions_cap"] = 1e18   # non-binding

        if eps is not None:
            kwargs["reliability_max_unserved_fraction"] = eps
            kwargs["reliability_mode"] = "total"

        kwargs["solar_capex_by_year"] = solar_capex_tv
        m = build_model(**kwargs)
        status = solve_model(m)
        if not status["optimal"]:
            raise RuntimeError("non-optimal termination")
        return m

    def _make_row(eps, m, diag, status_str):
        """Build a result row from solved model and diagnostics."""
        unserved = sum(diag["unserved_twh_by_year"].values())
        dual_info = extract_reliability_dual(m, diag=diag)
        dual_raw  = dual_info["dual_usd_per_twh"]
        dual_norm = (
            dual_raw * total_demand_twh / 100.0
            if dual_raw is not None else None
        )
        # Primal-side binding detection (authoritative over dual-side)
        rhs_twh = eps * total_demand_twh if eps is not None else None
        primal_slack = rhs_twh - unserved if rhs_twh is not None else None
        binding = primal_slack < 1e-4 if primal_slack is not None else None

        gs_vals = [
            v for v in diag["gas_shadow_price_usd_per_twh_th_by_year"].values()
            if v is not None
        ]

        return {
            "gas_case":            gas_case,
            "financing_label":     financing_label,
            "policy_label":        policy_label,
            "eps":                 eps,
            "eps_label":           "voll_only" if eps is None else f"eps_{eps:.2f}",
            "npv_cost":            float(pyo.value(m.system_cost_npv)),
            "unserved_total":      unserved,
            "horizon_reliability": diag["horizon_reliability"],
            "dual_usd_per_twh":    dual_raw,
            "dual_usd_per_pct_pt": dual_norm,
            "binding":             binding,
            "primal_slack_twh":    primal_slack,
            "total_demand_twh":    total_demand_twh,
            "gas_shadow_mean":     float(np.mean(gs_vals)) if gs_vals else 0.0,
            "solar_eaas_total_mw": sum(
                float(pyo.value(m.solar_eaas_add[t]))
                for t in range(len(scenario["years"]))
            ),
            "solar_public_total_mw": sum(
                float(pyo.value(m.solar_public_add[t]))
                for t in range(len(scenario["years"]))
            ),
            "status":              status_str,
        }
    
    # ── 1. VoLL-only baseline ──────────────────────────────────────────────
    try:
        m_base = _build_and_solve(eps=None)
        diag_base = extract_planning_diagnostics(m_base, scenario, econ)
        results.append(_make_row(None, m_base, diag_base, "feasible"))
    except RuntimeError as e:
        results.append({
            "gas_case": gas_case, "financing_label": financing_label,
            "policy_label": policy_label, "eps": None,
            "eps_label": "voll_only", "status": f"infeasible: {e}",
            "total_demand_twh": total_demand_twh,
        })

    # ── 2. Constrained sweep ───────────────────────────────────────────────
    for eps in eps_levels:
        try:
            m = _build_and_solve(eps=eps)
            diag = extract_planning_diagnostics(m, scenario, econ)
            results.append(_make_row(eps, m, diag, "feasible"))
        except RuntimeError:
            results.append({
                "gas_case": gas_case, "financing_label": financing_label,
                "policy_label": policy_label, "eps": eps,
                "eps_label": f"eps_{eps:.2f}", "status": "infeasible",
                "total_demand_twh": total_demand_twh,
                "npv_cost": None, "unserved_total": None,
                "horizon_reliability": None, "dual_usd_per_twh": None,
                "dual_usd_per_pct_pt": None, "binding": None,
                "primal_slack_twh": None, "gas_shadow_mean": None,
                "solar_eaas_total_mw": None, "solar_public_total_mw": None,
            })

    return results


def identify_bottleneck(diag):

    gas_shadow = diag["gas_shadow_price_usd_per_twh_th_by_year"]
    land_shadow = diag["land_shadow_price_by_year"]
    # storage_binding = diag["storage_binding_by_year"]

    gas_binding_years = sum(1 for v in gas_shadow.values() if v and v > 1e-6)
    land_binding_years = sum(1 for v in land_shadow.values() if v and v > 1e-6)
    
    # storage_modes = list(storage_binding.values())
    # dominant_storage = max(set(storage_modes), key=storage_modes.count)

    if gas_binding_years > 0.3 * len(gas_shadow):
        return "gas_constraint"

    if land_binding_years > 0.3 * len(land_shadow):
        return "land_constraint"

    #if dominant_storage != "none":
    #    return f"storage_{dominant_storage}"

    return "none"


def run_bottleneck_sensitivity(scenario, econ):

    gas_cases = ["baseline", "upside", "downside", "shock_recovery"]

    results = []

    for g in gas_cases:

        scenario_mod = load_scenario(
            demand_level_case="served",
            demand_case="baseline",
            gas_deliverability_case=g,
            carbon_case="no_policy",
            start_year=int(scenario["years"][0]),
            end_year=int(scenario["years"][-1]),
        )

        m = build_model(
            scenario=scenario_mod,
            econ=econ,
            reliability_max_unserved_fraction=0.10,
            reliability_mode="total"
        )

        try:
            solve_model(m)
            diag = extract_planning_diagnostics(m, scenario_mod, econ)

            results.append({
                "gas_case": g,
                "reliability": diag["horizon_reliability"],
                "unserved": sum(diag["unserved_twh_by_year"].values())
            })

        except:
            results.append({
                "gas_case": g,
                "reliability": None,
                "unserved": None
            })

    return results

def reliability_results_to_df(results):
    """
    Convert reliability sweep results into a flat DataFrame.
    Handles both old-format results (with 'case' key) and new-format
    results (with 'gas_case' key) for backward compatibility.
    """
    rows = []
    for r in results:
        # Support both old format (case=) and new format (gas_case=)
        gas_case = r.get("gas_case") or r.get("case", "unknown")
        rows.append({
            "gas_case":            gas_case,
            "eps":                 r.get("eps"),
            "eps_label":           r.get("eps_label", ""),
            "reliability":         None if r.get("eps") is None else 1 - r["eps"],
            "npv_cost":            r.get("npv_cost"),
            "unserved_total":      r.get("unserved_total"),
            "horizon_reliability": r.get("horizon_reliability"),
            # REL-2 core columns
            "dual_usd_per_twh":    r.get("dual_usd_per_twh") or r.get("dual_reliability"),
            "dual_usd_per_pct_pt": r.get("dual_usd_per_pct_pt"),
            "binding":             r.get("binding"),
            "primal_slack_twh":    r.get("primal_slack_twh"),
            "total_demand_twh":    r.get("total_demand_twh"),
            "gas_shadow_mean":     r.get("gas_shadow_mean"),
            "status":              r.get("status", "feasible"),
        })
    return pd.DataFrame(rows)


def run_rel3_financing_frontier(
    econ,
    gas_cases,
    policy_configs,
    eps_levels=None,
    cap_path="data/cost/processed/emissions_cap.csv",
):
    """
    REL-3: Financing arm comparison of the reliability feasibility frontier.

    For every combination of (gas_case, policy_arm), runs the eps sweep
    under BOTH financing arms (public_only and eaas) and returns all rows
    tagged for direct comparison.

    The key output is:
      frontier_shift_eps = threshold(public_only) - threshold(eaas)
      Positive: EaaS achieves tighter reliability (lower eps still feasible)
      Zero:     EaaS does not improve the feasibility frontier
      Negative: Shouldn't happen (EaaS weakly dominates public_only)

    Parameters
    ----------
    econ          : dict
    gas_cases     : list[str]
    policy_configs: list[dict], each with keys:
        "policy_label"   : str (e.g. "no_policy", "ndc3_unconditional")
        "ndc_scenario"   : str or None (cap CSV scenario name)
        "public_capital" : str — capital_case for public_only arm
        "eaas_capital"   : str — capital_case for EaaS arm
        "eaas_margin"    : float — required_margin for EaaS arm
        "eaas_tariff"    : float — solar_service_tariff_usd_per_twh
    eps_levels    : list[float] or None
    cap_path      : str

    Returns
    -------
    list[dict] — all rows from all combinations, tagged by gas_case,
                 financing_label, policy_label
    """
    if eps_levels is None:
        eps_levels = reliability_levels_log()

    # Load cap CSV once
    cap_df = pd.read_csv(cap_path)

    all_rows = []

    for policy_cfg in policy_configs:
        policy_label  = policy_cfg["policy_label"]
        ndc_scenario  = policy_cfg.get("ndc_scenario")   # None = no_policy
        pub_capital   = policy_cfg["public_capital"]
        eaas_capital  = policy_cfg["eaas_capital"]
        eaas_margin   = policy_cfg["eaas_margin"]
        eaas_tariff   = policy_cfg["eaas_tariff"]

        for gas_case in gas_cases:

            # Load annual NDC caps if policy requires them
            caps = None
            if ndc_scenario is not None:
                years_list = list(range(2025, 2046))
                cap_sub = cap_df[
                    (cap_df["scenario"] == ndc_scenario)
                    & (cap_df["year"].isin(years_list))
                ].sort_values("year")
                if len(cap_sub) != len(years_list):
                    raise ValueError(
                        f"Cap length {len(cap_sub)} != {len(years_list)} "
                        f"for scenario='{ndc_scenario}'"
                    )
                caps = cap_sub["cap_tco2"].astype(float).tolist()

            # ── Arm A: Public-only (traditional financing) ─────────────────
            scenario_pub = load_scenario(
                demand_level_case="served",
                demand_case="baseline",
                gas_deliverability_case=gas_case,
                capital_case=pub_capital,
                carbon_case="no_policy",
                start_year=2025,
                end_year=2045,
            )
            # financing_regime defaults to "traditional" from load_scenario

            rows_pub = run_reliability_sweep(
                scenario=scenario_pub,
                econ=econ,
                gas_case=gas_case,
                financing_label="public_only",
                policy_label=policy_label,
                emissions_cap_by_year=caps,
            )
            all_rows.extend(rows_pub)

            # ── Arm B: EaaS ────────────────────────────────────────────────
            scenario_eaas = load_scenario(
                demand_level_case="served",
                demand_case="baseline",
                gas_deliverability_case=gas_case,
                capital_case=eaas_capital,
                carbon_case="no_policy",
                start_year=2025,
                end_year=2045,
            )
            scenario_eaas["financing_regime"] = "eaas"
            scenario_eaas["required_margin"]  = eaas_margin
            scenario_eaas["solar_service_tariff_usd_per_twh"] = eaas_tariff

            rows_eaas = run_reliability_sweep(
                scenario=scenario_eaas,
                econ=econ,
                gas_case=gas_case,
                financing_label="eaas",
                policy_label=policy_label,
                emissions_cap_by_year=caps,
            )
            all_rows.extend(rows_eaas)

    return all_rows


def compute_frontier_shift(df):
    """
    From the output of run_rel3_financing_frontier(), compute:
      - feasibility_threshold_eps per (gas_case, financing_label, policy_label)
      - frontier_shift_eps = threshold(public_only) - threshold(eaas)
        per (gas_case, policy_label)

    Returns two DataFrames: (threshold_df, shift_df)
    """
    df_constrained = df[df["eps"].notna()].copy()

    # Threshold per combination
    threshold_rows = []
    for (gas, fin, pol), grp in df_constrained.groupby(
        ["gas_case", "financing_label", "policy_label"]
    ):
        feasible = grp[grp["status"] == "feasible"]
        threshold = float(feasible["eps"].min()) if len(feasible) > 0 else None
        infeasible_all = threshold is None

        cost_at_thresh = None
        eaas_mw_at_thresh = None
        if threshold is not None:
            row_at = feasible[feasible["eps"] == threshold].iloc[0]
            cost_at_thresh = row_at["npv_cost"]
            eaas_mw_at_thresh = row_at.get("solar_eaas_total_mw")

        threshold_rows.append({
            "gas_case":                  gas,
            "financing_label":           fin,
            "policy_label":              pol,
            "feasibility_threshold_eps": threshold,
            "fully_infeasible":          infeasible_all,
            "npv_cost_at_threshold":     cost_at_thresh,
            "solar_eaas_mw_at_threshold": eaas_mw_at_thresh,
        })

    threshold_df = pd.DataFrame(threshold_rows)

    # Frontier shift per (gas_case, policy_label)
    shift_rows = []
    for (gas, pol), grp in threshold_df.groupby(["gas_case", "policy_label"]):
        pub  = grp[grp["financing_label"] == "public_only"]
        eaas = grp[grp["financing_label"] == "eaas"]

        pub_thresh  = pub["feasibility_threshold_eps"].values[0]  if len(pub)  else None
        eaas_thresh = eaas["feasibility_threshold_eps"].values[0] if len(eaas) else None

        # frontier_shift > 0 means EaaS achieves tighter reliability
        shift = (
            float(pub_thresh - eaas_thresh)
            if (pub_thresh is not None and eaas_thresh is not None)
            else None
        )

        # Does EaaS make a previously infeasible standard feasible?
        necessity = (
            pub_thresh is None and eaas_thresh is not None
        )

        shift_rows.append({
            "gas_case":              gas,
            "policy_label":          pol,
            "pub_threshold_eps":     pub_thresh,
            "eaas_threshold_eps":    eaas_thresh,
            "frontier_shift_eps":    shift,
            "eaas_strictly_better": (shift is not None and shift > 1e-4),
            "eaas_necessary":        necessity,  # EaaS rescues otherwise infeasible
            "interpretation": (
                "EaaS is NECESSARY: makes feasible what public capital cannot."
                if necessity
                else (
                    f"EaaS improves frontier by {shift:.2f} eps units."
                    if (shift is not None and shift > 1e-4)
                    else "EaaS provides no feasibility improvement at this gas/policy combination."
                )
            ),
        })

    return threshold_df, pd.DataFrame(shift_rows)


def run_dem1_demand_sensitivity(
    econ,
    demand_level_cases,
    ndc_configs,
    cap_path="data/cost/processed/emissions_cap.csv",
):
    """
    DEM-1: Demand level sensitivity of NDC compliance cost and EaaS deployment.

    For every combination of (demand_level_case, ndc_scenario, financing_arm),
    solves the NDC-constrained optimisation and records cost, access, capital
    bottleneck intensity, and EaaS deployment.

    The experiment answers two sub-questions:

    Sub-Q1 — Cost sensitivity:
      Does NDC compliance cost rise under latent demand vs served demand?
      Mechanism: higher demand requires more solar under the same emissions cap.
      The gas supply ceiling is unchanged (supply-constrained), so gas emissions
      are identical across demand levels. The additional solar needed to serve
      latent demand must be financed from the same capital envelope.

    Sub-Q2 — Financing bottleneck diagnosis:
      Does the public capital bottleneck intensify under latent demand?
      Measured by: public_budget_shadow (dual on budget constraint),
      public_budget_utilisation (fraction of budget spent), and
      solar_eaas_total_mw (private capital mobilised to fill the gap).
      If EaaS deployment grows with demand level, EaaS shifts from
      convenient to necessary as demand assumptions tighten.

    NOTE ON EMISSIONS CAPS:
      The emissions caps are built from the baseline (served demand) gas_to_power
      path. Because gas deliverability is supply-constrained (fixed at 40.66 TWh_th
      regardless of demand), gas_to_power and therefore emissions are independent
      of the demand level. The existing caps are valid across all demand levels
      and do not need to be rebuilt.

    Parameters
    ----------
    econ               : dict
    demand_level_cases : list[str] — e.g. ["served", "latent_low", "latent_high"]
    ndc_configs        : list[dict], each with:
        "ndc_label"       : str — display label
        "ndc_scenario"    : str — key in emissions_cap.csv
        "public_capital"  : str — capital_case for public_only arm
        "eaas_capital"    : str — capital_case for EaaS arm
        "eaas_margin"     : float — required_margin for EaaS
        "eaas_tariff"     : float — solar_service_tariff_usd_per_twh
    cap_path           : str

    Returns
    -------
    list[dict] — one row per (demand_level, ndc_scenario, financing_arm)
    """
    cap_df = pd.read_csv(cap_path)
    years_list = list(range(2025, 2046))

    results = []

    for ndc_cfg in ndc_configs:
        ndc_label    = ndc_cfg["ndc_label"]
        ndc_scenario = ndc_cfg["ndc_scenario"]
        pub_capital  = ndc_cfg["public_capital"]
        eaas_capital = ndc_cfg["eaas_capital"]
        eaas_margin  = ndc_cfg["eaas_margin"]
        eaas_tariff  = ndc_cfg["eaas_tariff"]

        # Load annual caps once per NDC scenario
        cap_sub = cap_df[
            (cap_df["scenario"] == ndc_scenario)
            & (cap_df["year"].isin(years_list))
        ].sort_values("year")
        if len(cap_sub) != len(years_list):
            raise ValueError(
                f"Cap length {len(cap_sub)} != {len(years_list)} "
                f"for scenario='{ndc_scenario}'"
            )
        caps = cap_sub["cap_tco2"].astype(float).tolist()

        for demand_level in demand_level_cases:

            for fin_label, fin_params in [
                ("public_only", {
                    "financing_regime": "traditional",
                    "capital_case":     pub_capital,
                }),
                ("eaas", {
                    "financing_regime": "eaas",
                    "capital_case":     eaas_capital,
                    "required_margin":  eaas_margin,
                    "solar_service_tariff_usd_per_twh": eaas_tariff,
                }),
            ]:
                scenario = load_scenario(
                    demand_level_case=demand_level,
                    demand_case="baseline",
                    gas_deliverability_case="baseline",
                    capital_case=fin_params["capital_case"],
                    solar_build_case="aggressive",
                    carbon_case="no_policy",
                    start_year=2025,
                    end_year=2045,
                )
                # Apply financing parameters
                scenario["financing_regime"] = fin_params["financing_regime"]
                if "required_margin" in fin_params:
                    scenario["required_margin"] = fin_params["required_margin"]
                if "solar_service_tariff_usd_per_twh" in fin_params:
                    scenario["solar_service_tariff_usd_per_twh"] = (
                        fin_params["solar_service_tariff_usd_per_twh"]
                    )

                years = list(scenario["years"])
                # Load time-varying solar CAPEX from NREL ATB (solar_low scenario).
                # solar_low declines from $1,456k/MW (2025) to $603k/MW (2045).
                solar_capex_tv = load_solar_capex_by_year(
                    scenario_name="solar_low",
                    start_year=int(years[0]),
                    end_year=int(years[-1]),
                )

                # Activate minimum build floor when time-varying CAPEX is in use.
                # This prevents, the optimizer from delaying all solar to the cheapest years
                # (2040-2045) creating unrealistic 2025-2030 supply gaps.
                scenario["solar_min_build_mw_per_year"] = 100.0
                m = build_model(
                    scenario=scenario,
                    econ=econ,
                    emissions_cap_by_year=caps,
                    solar_capex_by_year=solar_capex_tv,
                )

                try:
                    status = solve_model(m)
                    if not status["optimal"]:
                        raise RuntimeError("non-optimal")

                    diag = extract_planning_diagnostics(m, scenario, econ)

                    cs_vals = [
                        v for v in
                        diag["carbon_shadow_price_usd_per_tco2_by_year"].values()
                        if v is not None
                    ]
                    gs_vals = [
                        v for v in
                        diag["gas_shadow_price_usd_per_twh_th_by_year"].values()
                        if v is not None
                    ]

                    results.append({
                        "demand_level":              demand_level,
                        "ndc_label":                 ndc_label,
                        "ndc_scenario":              ndc_scenario,
                        "financing_arm":             fin_label,
                        "status":                    "optimal",
                        # Sub-Q1: cost and access
                        "npv_total_cost_usd":        float(pyo.value(m.system_cost_npv)),
                        "cumulative_unserved_twh":   sum(
                            diag["unserved_twh_by_year"].values()
                        ),
                        "cumulative_emissions_tco2": float(pyo.value(m.emissions)),
                        "horizon_reliability":       diag["horizon_reliability"],
                        "solar_total_mw":            sum(
                            float(pyo.value(m.solar_public_add[t]))
                            + float(pyo.value(m.solar_eaas_add[t]))
                            for t in range(len(years))
                        ),
                        # Sub-Q2: financing bottleneck
                        "budget_shadow_usd_per_usd": diag.get(
                            "public_budget_shadow_usd_per_usd"
                        ),
                        "budget_utilisation":        diag.get(
                            "public_budget_utilisation"
                        ),
                        "budget_realised_spend_usd": diag.get(
                            "public_budget_realised_spend_usd"
                        ),
                        "solar_eaas_total_mw":       sum(
                            float(pyo.value(m.solar_eaas_add[t]))
                            for t in range(len(years))
                        ),
                        "solar_public_total_mw":     sum(
                            float(pyo.value(m.solar_public_add[t]))
                            for t in range(len(years))
                        ),
                        # Constraint diagnostics
                        "carbon_shadow_mean":        (
                            float(np.mean(cs_vals)) if cs_vals else 0.0
                        ),
                        "carbon_binding_years":      sum(
                            1 for v in cs_vals if v > 1e-6
                        ),
                        "gas_shadow_mean":           (
                            float(np.mean(gs_vals)) if gs_vals else 0.0
                        ),
                        # Model parameters for traceability
                        "base_demand_twh":           scenario["base_demand_twh"],
                        "capital_case":              fin_params["capital_case"],
                        "required_margin":           scenario.get("required_margin"),
                    })

                except RuntimeError as e:
                    results.append({
                        "demand_level":    demand_level,
                        "ndc_label":       ndc_label,
                        "ndc_scenario":    ndc_scenario,
                        "financing_arm":   fin_label,
                        "status":          f"infeasible: {e}",
                        "base_demand_twh": scenario["base_demand_twh"],
                    })

    return results


def run_dem2_growth_gas_matrix(
    econ,
    demand_cases,
    gas_cases,
    financing_configs,
    policy_configs,
    cap_path="data/cost/processed/emissions_cap.csv",
):
    """
    DEM-2: Demand growth × gas deliverability interaction matrix.

    Runs a full LP re-optimisation for every combination of:
      (demand_case, gas_case, financing_arm, policy_arm)

    This is NOT a fixed-capacity sensitivity (that is what stochastic.py does).
    Each combination re-solves the capacity planning problem from scratch,
    producing the optimal investment mix for that specific demand × gas context.

    The interaction effect is measured by:
      interaction = [cost(high_growth, gas_case) - cost(high_growth, baseline_gas)]
                  - [cost(low_growth,  gas_case) - cost(low_growth,  baseline_gas)]

    Positive: demand growth AMPLIFIES gas scarcity cost (super-additive effect).
    Zero:     demand growth and gas regime are independent (additive only).
    Negative: higher demand growth REDUCES gas scarcity premium (unusual).

    EaaS value across growth rates:
      eaas_value = cost(public_only) - cost(eaas)  for each (demand, gas, policy)
    If eaas_value rises monotonically with demand growth: EaaS becomes more
    valuable as demand pressure increases.

    Parameters
    ----------
    econ            : dict
    demand_cases    : list[str] — e.g. ["low", "baseline", "high"]
    gas_cases       : list[str] — e.g. ["baseline", "upside", "downside", "shock_recovery"]
    financing_configs : list[dict], each with:
        "label"              : str — "public_only" or "eaas"
        "financing_regime"   : str — "traditional" or "eaas"
        "capital_case"       : str
        "required_margin"    : float (EaaS only)
        "tariff"             : float (EaaS only)
    policy_configs  : list[dict], each with:
        "policy_label"  : str
        "ndc_scenario"  : str or None
    cap_path        : str

    Returns
    -------
    list[dict] — one row per (demand_case, gas_case, financing_arm, policy_arm)
    """
    cap_df = pd.read_csv(cap_path)
    years_list = list(range(2025, 2046))

    results = []

    for policy_cfg in policy_configs:
        policy_label = policy_cfg["policy_label"]
        ndc_scenario = policy_cfg.get("ndc_scenario")

        # Load caps once per policy
        caps = None
        if ndc_scenario is not None:
            cap_sub = cap_df[
                (cap_df["scenario"] == ndc_scenario)
                & (cap_df["year"].isin(years_list))
            ].sort_values("year")
            if len(cap_sub) != len(years_list):
                raise ValueError(
                    f"Cap length mismatch for scenario='{ndc_scenario}'"
                )
            caps = cap_sub["cap_tco2"].astype(float).tolist()
        
        solar_capex_tv = load_solar_capex_by_year(
            scenario_name="solar_low",
            start_year=2025,
            end_year=2045,
        )
        for demand_case in demand_cases:
            for gas_case in gas_cases:
                for fin_cfg in financing_configs:
                    fin_label = fin_cfg["label"]
                    capital   = fin_cfg["capital_case"]

                    scenario = load_scenario(
                        demand_level_case="served",
                        demand_case=demand_case,
                        gas_deliverability_case=gas_case,
                        capital_case=capital,
                        solar_build_case="aggressive",
                        carbon_case="no_policy",
                        start_year=2025,
                        end_year=2045,
                    )

                    scenario["financing_regime"] = fin_cfg["financing_regime"]
                    if fin_cfg["financing_regime"] == "eaas":
                        scenario["required_margin"] = fin_cfg["required_margin"]
                        scenario["solar_service_tariff_usd_per_twh"] = fin_cfg["tariff"]

                    build_kwargs = dict(scenario=scenario, econ=econ)
                    if caps is not None:
                        build_kwargs["emissions_cap_by_year"] = caps
                    else:
                        build_kwargs["emissions_cap"] = 1e18

                    build_kwargs["solar_capex_by_year"] = solar_capex_tv
                    m = build_model(**build_kwargs)

                    try:
                        status = solve_model(m)
                        if not status["optimal"]:
                            raise RuntimeError("non-optimal")

                        diag = extract_planning_diagnostics(m, scenario, econ)
                        T = range(len(scenario["years"]))

                        cs_vals = [
                            v for v in
                            diag["carbon_shadow_price_usd_per_tco2_by_year"].values()
                            if v is not None
                        ]
                        gs_vals = [
                            v for v in
                            diag["gas_shadow_price_usd_per_twh_th_by_year"].values()
                            if v is not None
                        ]

                        results.append({
                            "demand_case":           demand_case,
                            "demand_growth_rate":    scenario["demand_growth"],
                            "gas_case":              gas_case,
                            "financing_arm":         fin_label,
                            "policy_label":          policy_label,
                            "status":                "optimal",
                            # Core outcomes
                            "npv_total_cost_usd":    float(pyo.value(m.system_cost_npv)),
                            "cumulative_unserved_twh": sum(
                                diag["unserved_twh_by_year"].values()
                            ),
                            "cumulative_emissions_tco2": float(pyo.value(m.emissions)),
                            "horizon_reliability":   diag["horizon_reliability"],
                            # Investment decisions
                            "solar_total_mw":        sum(
                                float(pyo.value(m.solar_public_add[t]))
                                + float(pyo.value(m.solar_eaas_add[t]))
                                for t in T
                            ),
                            "solar_eaas_total_mw":   sum(
                                float(pyo.value(m.solar_eaas_add[t])) for t in T
                            ),
                            "solar_public_total_mw": sum(
                                float(pyo.value(m.solar_public_add[t])) for t in T
                            ),
                            # Constraint diagnostics
                            "gas_shadow_mean":       (
                                float(np.mean(gs_vals)) if gs_vals else 0.0
                            ),
                            "gas_shadow_binding_years": sum(
                                1 for v in gs_vals if v > 1e-6
                            ),
                            "carbon_shadow_mean":    (
                                float(np.mean(cs_vals)) if cs_vals else 0.0
                            ),
                            "carbon_binding_years":  sum(
                                1 for v in cs_vals if v > 1e-6
                            ),
                            "budget_shadow":         diag.get(
                                "public_budget_shadow_usd_per_usd"
                            ),
                            "budget_utilisation":    diag.get(
                                "public_budget_utilisation"
                            ),
                        })

                    except RuntimeError as e:
                        results.append({
                            "demand_case":     demand_case,
                            "demand_growth_rate": scenario["demand_growth"],
                            "gas_case":        gas_case,
                            "financing_arm":   fin_label,
                            "policy_label":    policy_label,
                            "status":          f"infeasible: {e}",
                        })

    return results


def run_str1_storage_parameter_sweep(
    econ,
    hd_values,
    surplus_frac_values,
    financing_configs,
    policy_configs,
    cap_path="data/cost/processed/emissions_cap.csv",
):
    """
    STR-1: Storage parameter sweep across H_d × surplus_frac × financing × policy.

    Answers whether battery storage COMPLEMENTS or SUBSTITUTES for EaaS solar
    deployment, and how that role depends on the storage operational parameters:
      H_d (storage_deployable_hours_per_year): how many hours/year the battery
        can charge and discharge at rated power. Controls throughput capacity.
      surplus_frac (storage_solar_surplus_frac): what fraction of solar generation
        is available for storage charging. Controls the solar-storage coupling.

    COMPLEMENT vs SUBSTITUTE detection:
      For each (H_d, surplus_frac) combination, compare the EaaS arm and
      public_only arm. Then compute:
        complement_ratio = storage_final_capacity_mwh / solar_eaas_total_mw
      Rising complement_ratio as H_d increases => storage and EaaS solar grow
        together (complement: each makes the other more valuable).
      Falling complement_ratio as H_d increases => storage replaces EaaS solar
        (substitute: more efficient storage deployment reduces solar requirement).

    Storage utilisation interpretation:
      storage_utilisation_rate ≈ 1.0 => H_d is binding (more hours would help)
      storage_utilisation_rate << 1.0 => H_d is slack; solar surplus is the limit
        => increasing H_d won't help; need more solar or higher surplus_frac

    Parameters
    ----------
    econ               : dict
    hd_values          : list[float] — H_d levels to sweep
    surplus_frac_values: list[float] — surplus_frac levels to sweep
    financing_configs  : list[dict] with keys: label, financing_regime,
                         capital_case, required_margin (EaaS), tariff (EaaS)
    policy_configs     : list[dict] with keys: policy_label, ndc_scenario
    cap_path           : str

    Returns
    -------
    list[dict] — one row per (H_d, surplus_frac, financing_arm, policy_arm)
    """
    cap_df = pd.read_csv(cap_path)
    years_list = list(range(2025, 2046))

    results = []

    for policy_cfg in policy_configs:
        policy_label = policy_cfg["policy_label"]
        ndc_scenario = policy_cfg.get("ndc_scenario")

        caps = None
        if ndc_scenario is not None:
            cap_sub = cap_df[
                (cap_df["scenario"] == ndc_scenario)
                & (cap_df["year"].isin(years_list))
            ].sort_values("year")
            if len(cap_sub) != len(years_list):
                raise ValueError(
                    f"Cap length mismatch for scenario='{ndc_scenario}'"
                )
            caps = cap_sub["cap_tco2"].astype(float).tolist()

        solar_capex_tv = load_solar_capex_by_year(
            scenario_name="solar_low",
            start_year=2025,
            end_year=2045,
        )

        for hd in hd_values:
            for sfrac in surplus_frac_values:
                for fin_cfg in financing_configs:
                    fin_label = fin_cfg["label"]

                    scenario = load_scenario(
                        demand_level_case="served",
                        demand_case="baseline",
                        gas_deliverability_case="baseline",
                        capital_case=fin_cfg["capital_case"],
                        solar_build_case="aggressive",
                        carbon_case="no_policy",
                        start_year=2025,
                        end_year=2045,
                    )

                    # Override storage parameters
                    scenario["storage_deployable_hours_per_year"] = hd
                    scenario["storage_solar_surplus_frac"]        = sfrac
                    scenario["solar_min_build_mw_per_year"] = 100.0

                    scenario["financing_regime"] = fin_cfg["financing_regime"]
                    if fin_cfg["financing_regime"] == "eaas":
                        scenario["required_margin"] = fin_cfg["required_margin"]
                        scenario["solar_service_tariff_usd_per_twh"] = fin_cfg["tariff"]

                    build_kwargs = dict(scenario=scenario, econ=econ)
                    if caps is not None:
                        build_kwargs["emissions_cap_by_year"] = caps
                    else:
                        build_kwargs["emissions_cap"] = 1e18
                    
                    build_kwargs["solar_capex_by_year"] = solar_capex_tv
                    
                    m = build_model(**build_kwargs)

                    try:
                        status = solve_model(m)
                        if not status["optimal"]:
                            raise RuntimeError("non-optimal")

                        diag = extract_planning_diagnostics(m, scenario, econ)
                        T = range(len(scenario["years"]))

                        solar_eaas_mw = sum(
                            float(pyo.value(m.solar_eaas_add[t])) for t in T
                        )
                        storage_mwh = diag["storage_final_capacity_mwh"]

                        results.append({
                            "hd":                    hd,
                            "surplus_frac":          sfrac,
                            "financing_arm":         fin_label,
                            "policy_label":          policy_label,
                            "status":                "optimal",
                            # Core outcomes
                            "npv_total_cost_usd":    float(pyo.value(m.system_cost_npv)),
                            "cumulative_unserved_twh": sum(
                                diag["unserved_twh_by_year"].values()
                            ),
                            "horizon_reliability":   diag["horizon_reliability"],
                            # Storage outcomes
                            "storage_final_mwh":     storage_mwh,
                            "storage_throughput_twh": diag["storage_throughput_twh"],
                            "storage_utilisation":   diag["storage_utilisation_rate"],
                            "power_limit_years":     diag["storage_power_limit_years"],
                            "charge_limit_years":    diag["storage_charge_limit_years"],
                            # Solar outcomes
                            "solar_total_mw":        sum(
                                float(pyo.value(m.solar_public_add[t]))
                                + float(pyo.value(m.solar_eaas_add[t]))
                                for t in T
                            ),
                            "solar_eaas_total_mw":   solar_eaas_mw,
                            "solar_public_total_mw": sum(
                                float(pyo.value(m.solar_public_add[t])) for t in T
                            ),
                            # STR-1 core metric: complement ratio
                            # Rising with H_d => complement
                            # Falling with H_d => substitute
                            "complement_ratio": (
                                storage_mwh / solar_eaas_mw
                                if solar_eaas_mw > 1.0
                                else None
                            ),
                            # Storage cost share of total NPV
                            "storage_cost_npv": sum(
                                float(pyo.value(m.DF[t]))
                                * float(pyo.value(m.storage_add[t]))
                                * econ["STORAGE_COST_PER_MWH"]
                                for t in T
                            ),
                        })

                    except RuntimeError as e:
                        results.append({
                            "hd":           hd,
                            "surplus_frac": sfrac,
                            "financing_arm": fin_label,
                            "policy_label": policy_label,
                            "status":       f"infeasible: {e}",
                        })

    return results


def run_rel2_marginal_cost_curves(
    econ,
    gas_cases,
    capital_case="moderate",
    eps_levels=None,
):
    """
    REL-2: Marginal cost of reliability across gas deliverability regimes.

    For each gas regime, runs the full eps sweep and returns the dual on
    the reliability constraint at each eps level, plus binding detection
    and normalised cost per percentage point.

    The key output is the COST CURVE per gas regime: how steeply does the
    marginal cost of reliability rise as eps tightens, and does this curve
    shift LEFT (binding earlier, at looser eps) under adverse gas regimes?

    Parameters
    ----------
    econ         : dict
    gas_cases    : list[str] — gas regime names
    capital_case : str — fixed capital envelope
    eps_levels   : list[float] or None

    Returns
    -------
    list[dict]  — all rows from all gas regime sweeps combined.
                  Use reliability_results_to_df() to convert to DataFrame.
    """
    if eps_levels is None:
        eps_levels = reliability_levels_log()

    all_rows = []

    for gas_case in gas_cases:

        scenario = load_scenario(
            demand_level_case="served",
            demand_case="baseline",
            gas_deliverability_case=gas_case,
            capital_case=capital_case,
            carbon_case="no_policy",
            start_year=2025,
            end_year=2045,
        )

        rows = run_reliability_sweep(
            scenario=scenario,
            econ=econ,
            gas_case=gas_case,
        )
        all_rows.extend(rows)

    return all_rows


# ============================================================
# DIAGNOSTICS EXTRACTION
# ============================================================

def _series_dict_by_year(values, years):
    return {int(y): float(values[t]) for t, y in enumerate(years)}


def extract_gas_shadow_prices_usd_per_twh_th(m, years):
    out = {}
    for t, y in enumerate(years):
        dual = m.dual.get(m.gas_balance[t], None)
        val = None if dual is None else float(-dual)   # sign convention
        out[int(y)] = None if val is None else max(0.0, val)  # clamp solver noise
    return out


def extract_carbon_shadow_prices_usd_per_tco2(m, years):
    """
    Returns year -> USD/tCO2 (non-negative by convention).
    Handles both:
      - annual caps: indexed m.emissions_constraint[t]
      - single cap: scalar m.emissions_constraint
    """
    years = list(years)

    if not hasattr(m, "emissions_constraint"):
        return {int(y): None for y in years}

    # Indexed constraint (annual caps)
    if m.emissions_constraint.is_indexed():
        out = {}
        for t, y in enumerate(years):
            dual = m.dual.get(m.emissions_constraint[t], None)
            val = None if dual is None else float(-dual)
            out[int(y)] = None if val is None else max(0.0, val)
        return out

    # Scalar constraint (single cap)
    dual = m.dual.get(m.emissions_constraint, None)
    val = None if dual is None else float(-dual)
    val = None if val is None else max(0.0, val)
    return {int(y): val for y in years}


def compute_reliability_metrics(demand_by_year, unserved_by_year):

    years = sorted(demand_by_year.keys())

    # ---------- Annual reliability ----------
    annual_reliability = {}
    for y in years:
        d = demand_by_year[y]
        u = unserved_by_year[y]

        if d <= 0:
            annual_reliability[y] = None
        else:
            annual_reliability[y] = 1.0 - (u / d)

    # ---------- Worst year reliability ----------
    worst_year_reliability = min(
        r for r in annual_reliability.values() if r is not None
    )

    # ---------- Horizon reliability ----------
    total_demand = sum(demand_by_year.values())
    total_unserved = sum(unserved_by_year.values())

    if total_demand > 0:
        horizon_reliability = 1.0 - (total_unserved / total_demand)
    else:
        horizon_reliability = None

    # ---------- Expected Unserved Energy ----------
    expected_unserved_energy = total_unserved

    return {
        "annual_reliability_by_year": annual_reliability,
        "worst_year_reliability": worst_year_reliability,
        "horizon_reliability": horizon_reliability,
        "expected_unserved_energy_twh": expected_unserved_energy,
    }


def extract_planning_diagnostics(m, scenario, econ=None, solar_capex_series=None):

    years = scenario["years"]
    T = range(len(years))

    demand = project_baseline_demand(
        base_demand=scenario["base_demand_twh"],
        growth_rate=scenario["demand_growth"],
        start_year=years[0],
        end_year=years[-1],
    )["demand"]
    demand_dict = {int(years[t]): float(demand[t]) for t in range(len(years))}

    gas_avail = gas_available_power(
        start_year=int(years[0]),
        end_year=int(years[-1]),
        scenario_name=scenario["gas_scenario"],
    )["available_twh_th"]

    carbon_shadow = extract_carbon_shadow_prices_usd_per_tco2(m, years)
    
    # ------------------------------------------------------------
    # Shortage diagnostics
    # ------------------------------------------------------------

    unserved_by_year = {
        int(y): float(pyo.value(m.unserved[t]))
        for t, y in enumerate(years)
    }

    years_with_unserved = [
        y for y, v in unserved_by_year.items() if v > 1e-6
    ]

    max_unserved_year = (
        max(unserved_by_year, key=unserved_by_year.get)
        if years_with_unserved else None
    )

    # ------------------------------------------------------------
    # Gas scarcity economic impact
    # ------------------------------------------------------------

    gas_shadow = extract_gas_shadow_prices_usd_per_twh_th(m, years)

    gas_scarcity_value = 0.0

    for t, y in enumerate(years):

        shadow = gas_shadow[int(y)]
        gas_shortfall = gas_avail[t] - pyo.value(m.gas_to_power[t])

        if shadow is not None and shadow > 0:
            gas_scarcity_value += shadow * gas_shortfall

    # ------------------------------------------------------------
    # Value of additional gas supply to electricity reliability
    # ------------------------------------------------------------

    gas_value_electricity = {}

    eta = scenario["gas_eta"]

    for y, shadow in gas_shadow.items():

        if shadow is None:
            gas_value_electricity[y] = None
        else:
            gas_value_electricity[y] = shadow / eta
    
    
    #Solar share of generation
    total_generation = sum(
        pyo.value(m.gas_generation[t])
        + pyo.value(m.solar_generation[t])
        + pyo.value(m.storage_discharge[t])
        for t in T
    )

    solar_total = sum(
        pyo.value(m.solar_generation[t]) for t in T
    )

    solar_share = solar_total / total_generation if total_generation > 0 else 0
    # ------------------------------------------------------------
    # Year solar becomes cheaper than gas
    # ------------------------------------------------------------

    solar_energy_per_mw = scenario["solar_cf"] * 8760 / 1_000_000

    npv_energy = sum(
        pyo.value(m.DF[t]) * solar_energy_per_mw
        for t in range(len(years))
    )

    # solar_lcoe = econ["SOLAR_CAPEX_PER_MW"] / npv_energy if npv_energy > 0 else None
    # Solar LCOE: annualized over remaining horizon from each year t
    # Using a "remaining horizon" LCOE — CAPEX recovered over years remaining
    # from t onwards. This reflects the effective cost of a MW built in year t.
    gas_mc = econ["GAS_COST_PER_TWH_TH"] / scenario["gas_eta"]  # USD/TWh_e

    
    if solar_capex_series is None:
        # Infer CAPEX scenario from econ or scenario dict; default to solar_low
        capex_scenario = scenario.get("solar_capex_scenario", "solar_low")
        solar_capex_by_year = load_solar_capex_by_year(
            scenario_name=capex_scenario,
            start_year=int(years[0]),
            end_year=int(years[-1])
        )
    else:
        solar_capex_by_year = solar_capex_series
        

    solar_beats_gas_year = None
    for t, y in enumerate(years):
        capex_at_t = solar_capex_by_year.get(int(y), econ["SOLAR_CAPEX_PER_MW"])
        remaining_energy_npv = sum(
            pyo.value(m.DF[k]) * solar_energy_per_mw
            for k in range(t, len(years))
        )
        if remaining_energy_npv <= 0:
            continue
        lcoe_at_t = capex_at_t / remaining_energy_npv
        if lcoe_at_t < gas_mc:
            solar_beats_gas_year = int(y)
            break
    
    # ------------------------------------------------------------
    # Public capital multiplier of EaaS
    # ------------------------------------------------------------
    public_spending = 0.0
    private_investment = 0.0

    for t in T:

        public_spending += (
            pyo.value(m.DF[t])
            * pyo.value(m.eaas_subsidy[t])
        )

        private_investment += (
            pyo.value(m.DF[t])
            * pyo.value(m.solar_eaas_add[t])
            * pyo.value(m.solar_capex_param[t])
        )

    eaas_multiplier = None

    if public_spending > 0:
        eaas_multiplier = private_investment / public_spending

    # ------------------------------------------------------------
    # NDC conditional finance utilisation diagnostic
    # ------------------------------------------------------------
    # Tracks what fraction of the NDC conditional finance envelope
    # is absorbed by EaaS gap-funding (eaas_subsidy) and public solar
    # CAPEX. Only meaningful when scenario["ndc_finance_budget_usd"]
    # is set (conditional NDC runs). Safe to call on all runs —
    # returns None when no budget is defined.
    ndc_finance_budget = scenario.get("ndc_finance_budget_usd", None)

    # Discounted public solar CAPEX (parallel to eaas_subsidy tracking)
    public_solar_npv_spend = sum(
        pyo.value(m.DF[t])
        * pyo.value(m.solar_public_add[t])
        * pyo.value(m.solar_capex_param[t])
        for t in T
    )
    # eaas_subsidy spend (already computed above as public_spending)
    eaas_subsidy_npv_spend = public_spending   # reuse — same quantity

    ndc_finance_total_spend = public_solar_npv_spend + eaas_subsidy_npv_spend
    ndc_finance_utilisation = (
        ndc_finance_total_spend / ndc_finance_budget
        if ndc_finance_budget and ndc_finance_budget > 0
        else None
    )

    unserved_dict = {
        int(y): float(pyo.value(m.unserved[t]))
        for t, y in enumerate(years)
    }

    reliability_metrics = compute_reliability_metrics(
        demand_dict,
        unserved_dict
    )

    storage_binding = {}

    subsidy_per_mw_by_year = {}
    for t, y in enumerate(years):
        solar_add = pyo.value(m.solar_eaas_add[t])
        if solar_add > 1e-6:
            subsidy_per_mw_by_year[int(y)] = pyo.value(m.eaas_subsidy[t]) / solar_add
        else:
            subsidy_per_mw_by_year[int(y)] = 0.0

    for t, y in enumerate(years):
        power_bind = abs(
            pyo.value(m.storage_discharge[t])
            - (pyo.value(m.storage_power_mw[t]) * scenario.get("storage_deployable_hours_per_year", 700.0)) / 1e6
        ) <= 1e-6

        energy_bind = abs(
            pyo.value(m.storage_discharge[t])
            - scenario["storage_round_trip_eff"] * pyo.value(m.storage_charge[t])
        ) <= 1e-6
        # Note: binding is checked by primal slack < 1e-6 TWh (≈ 1 MWh).
        # This is a primal check, not a dual check. For LP post-processing,
        # this is for diagnostic purposes.
        charge_bind = abs(
            pyo.value(m.storage_charge[t])
            - scenario["storage_solar_surplus_frac"] * pyo.value(m.solar_generation[t])
        ) <= 1e-6

        if power_bind:
            storage_binding[int(y)] = "power_limit"
        elif energy_bind:
            storage_binding[int(y)] = "energy_limit"
        elif charge_bind:
            storage_binding[int(y)] = "charge_limit"
        else:
            storage_binding[int(y)] = "none"

    # ---- Peak adequacy heuristic (must be computed before return dict)
    # Peak-to-average demand ratio. Nigerian grid load factor ≈ 0.55
    # (Nigerian System Operator statistics, 2022-2023). 1/0.55 ≈ 1.82.
    # A higher value (2.0-2.5) would reflect more peaky demand patterns
    # characteristic of residential-dominated systems.
    peak_demand_multiple = scenario.get("peak_demand_multiple", 1.82)
    peak_adequacy_by_year = {}
    for t, y in enumerate(years):
        avg_demand_twh = demand[t]
        avg_mw = avg_demand_twh * 1e6 / 8760
        peak_mw = avg_mw * peak_demand_multiple
        firm_capacity_mw = (
            pyo.value(m.gas_capacity_mw[t])
            + pyo.value(m.storage_power_mw[t])
        )
        peak_adequacy_by_year[int(y)] = {
            "peak_demand_mw": round(peak_mw, 1),
            "firm_capacity_mw": round(firm_capacity_mw, 1),
            "reserve_margin": round((firm_capacity_mw - peak_mw) / peak_mw, 4),
        }

    # ---- Land shadow price (must be computed before return dict)
    land_shadow = {}
    for t, y in enumerate(years):
        raw = m.dual.get(m.land_constraint[t], None)
        if raw is None:
            slack = (
                scenario["land_available_km2"]
                - pyo.value(m.solar_capacity_mw[t]) * scenario["land_intensity_solar_km2_per_mw"]
                - pyo.value(m.gas_capacity_mw[t]) * scenario["land_intensity_gas_km2_per_mw"]
                - pyo.value(m.storage_capacity_mwh[t]) * scenario["land_intensity_storage_km2_per_mwh"]
            )
            if slack < -1e-4:
                raise RuntimeError(
                    f"Land constraint binding in year {int(y)} but dual is missing. "
                    "Solver did not return dual. Cannot report land shadow price."
                )
            land_shadow[int(y)] = 0.0
        else:
            land_shadow[int(y)] = max(0.0, float(-raw))

    # ------------------------------------------------------------
    # Public capital budget shadow price (FIN-3)
    # Dual on public_budget_constraint: the marginal NPV cost reduction
    # per additional USD of public capital made available.
    # Positive → constraint is binding (capital is the bottleneck).
    # Zero     → constraint is slack (budget not exhausted).
    # None     → no budget constraint was applied this run.
    # ------------------------------------------------------------
    public_budget_shadow = None
    public_budget_utilisation = None
    realised_public_spend = None

    if getattr(m, "has_public_budget_constraint", False):
        raw_dual = m.dual.get(m.public_budget_constraint, None)
        if raw_dual is not None:
            public_budget_shadow = max(0.0, float(-raw_dual))

        realised_public_spend = sum(
            float(pyo.value(m.DF[t])) * (
                float(pyo.value(m.solar_public_add[t])) * float(pyo.value(m.solar_capex_param[t]))
                + float(pyo.value(m.eaas_subsidy[t]))
            )
            for t in T
        )
        budget_npv = scenario.get("public_solar_budget_npv", None)
        if budget_npv and budget_npv > 0:
            public_budget_utilisation = realised_public_spend / budget_npv
    
    
    # ============================================================
    # EaaS reliability trigger diagnostic (post-solve)
    # ============================================================
    # Evaluates whether EaaS deployment was reliability-motivated:
    # compares the marginal value of unserved energy (VoLL dual or
    # penalty) against the effective cost of EaaS-financed solar.
    #
    # If unserved_shadow_price > eaas_effective_lcoe in a given year,
    # EaaS deployment in that year was reliability-justified — it was
    # cheaper to deploy private solar than to accept blackouts.
    #
    # This is a diagnostic, not a constraint. The LP decides EaaS
    # deployment optimally; this metric explains *why* it deployed.
    # ============================================================

    eaas_reliability_trigger_by_year = {}
    for t in T:
        yr = years[t]
        eaas_add_t = float(pyo.value(m.solar_eaas_add[t]))

        # Effective LCOE of EaaS solar at year t (USD per MWh)
        capex_t = float(pyo.value(m.solar_capex_param[t]))  # USD/MW
        cf_t = scenario.get("solar_cf", 0.27)
        hours_per_year = 8760
        eaas_lcoe_mwh = capex_t / (cf_t * hours_per_year * 20)  # simple 20-yr levelization

        # Marginal value of unserved energy (USD per MWh)
        # This is the VoLL penalty from the objective — represents
        # the planner's valuation of avoided blackouts.
        voll_usd_per_mwh = econ["UNSERVED_ENERGY_PENALTY"] * 1e6  # convert from USD/TWh to USD/MWh

        # Was EaaS deployment reliability-motivated at year t?
        reliability_motivated = (
            eaas_add_t > 0.01 and  # non-trivial deployment
            voll_usd_per_mwh > eaas_lcoe_mwh  # cheaper to build than to blackout
        )

        eaas_reliability_trigger_by_year[yr] = {
            "eaas_add_mw":           round(eaas_add_t, 1),
            "eaas_lcoe_usd_per_mwh": round(eaas_lcoe_mwh, 2),
            "voll_usd_per_mwh":      round(voll_usd_per_mwh, 2),
            "reliability_motivated": reliability_motivated,
        }

    n_reliability_motivated = sum(
        1 for v in eaas_reliability_trigger_by_year.values()
        if v["reliability_motivated"]
    )

    # ============================================================
    # COST DECOMPOSITION (real expenditure vs VoLL penalty)
    # ============================================================
    cost_gas_opex     = float(pyo.value(m.cost_gas_opex_npv))
    cost_gas_capex    = float(pyo.value(m.cost_gas_capex_npv))
    cost_pub_solar    = float(pyo.value(m.cost_public_solar_capex_npv))
    cost_eaas_solar   = float(pyo.value(m.cost_eaas_solar_capex_npv))
    cost_storage      = float(pyo.value(m.cost_storage_capex_npv))
    cost_voll_penalty = float(pyo.value(m.cost_unserved_voll_npv))
    cost_carbon       = float(pyo.value(m.cost_carbon_npv))

    cost_real_expenditure = (
        cost_gas_opex + cost_gas_capex + cost_pub_solar
        + cost_eaas_solar + cost_storage + cost_carbon
    )
    cost_total = cost_real_expenditure + cost_voll_penalty

    cost_decomposition = {
        "real_expenditure_npv":        round(cost_real_expenditure, 0),
        "voll_penalty_npv":            round(cost_voll_penalty, 0),
        "voll_penalty_share":          round(cost_voll_penalty / cost_total, 4) if cost_total > 0 else 0,
        "components": {
            "gas_opex_npv":            round(cost_gas_opex, 0),
            "gas_capex_npv":           round(cost_gas_capex, 0),
            "public_solar_capex_npv":  round(cost_pub_solar, 0),
            "eaas_solar_capex_npv":    round(cost_eaas_solar, 0),
            "storage_capex_npv":       round(cost_storage, 0),
            "voll_penalty_npv":        round(cost_voll_penalty, 0),
            "carbon_cost_npv":         round(cost_carbon, 0),
        },
    }

    return {
        # Cost decomposition
        "cost_decomposition": cost_decomposition,
        # EaaS reliability trigger
        "eaas_reliability_trigger_by_year": eaas_reliability_trigger_by_year,
        "eaas_reliability_motivated_years": n_reliability_motivated,
        # Peak and storage diagnostics
        "peak_adequacy_by_year": peak_adequacy_by_year,
        "storage_binding_by_year": storage_binding,
        "subsidy_per_mw_usd_by_year": subsidy_per_mw_by_year,
        "solar_capex_usd_per_mw": float(pyo.value(m.solar_capex_param[0])),
        "solar_tariff_usd_per_twh": scenario["solar_service_tariff_usd_per_twh"],
        "solar_beats_gas_year": solar_beats_gas_year,
        # Gas and energy value diagnostics
        "value_of_gas_to_power_usd_per_twh_e_by_year": gas_value_electricity,
        "storage_charge_twh_e_by_year":
            {int(y): float(pyo.value(m.storage_charge[t]))
             for t, y in enumerate(years)},
        "bankability_threshold":
            scenario["required_margin"] *
            (
                float(pyo.value(m.solar_capex_param[0])) /
                sum(pyo.value(m.DF[t]) for t in range(len(years)))
            ),
        "solar_generation_share": solar_share,
        "gas_scarcity_value_usd": gas_scarcity_value,
        "eaas_public_spending_usd": public_spending,
        "eaas_private_investment_usd": private_investment,
        "eaas_capital_multiplier": eaas_multiplier,
        "ndc_finance_public_solar_npv_usd": public_solar_npv_spend,
        "ndc_finance_eaas_subsidy_npv_usd": eaas_subsidy_npv_spend,
        "ndc_finance_total_spend_usd": ndc_finance_total_spend,
        "ndc_finance_utilisation": ndc_finance_utilisation,
        "years_with_unserved_energy": years_with_unserved,
        "max_unserved_year": max_unserved_year,
        "gas_shadow_price_usd_per_twh_th_by_year":
            extract_gas_shadow_prices_usd_per_twh_th(m, years),
        "land_shadow_price_by_year": land_shadow,
        "solar_capacity_mw_by_year":
            {int(y): float(pyo.value(m.solar_capacity_mw[t])) for t, y in enumerate(years)},
        "solar_generation_twh_e_by_year":
            {int(y): float(pyo.value(m.solar_generation[t])) for t, y in enumerate(years)},
        "discount_factor_by_year":
            {int(y): float(pyo.value(m.DF[t])) for t, y in enumerate(years)},
        "emissions_tco2_by_year":
            {int(y): float(pyo.value(m.emissions_by_year[t])) for t, y in enumerate(years)},
        "carbon_shadow_price_usd_per_tco2_by_year": carbon_shadow,
        "gas_avail_twh_th_by_year": _series_dict_by_year(gas_avail, years),
        "bankable_capex_per_mw":
            scenario["solar_service_tariff_usd_per_twh"] *
            npv_energy
            / scenario["required_margin"],
        "gas_to_power_twh_th_by_year":
            {int(y): float(pyo.value(m.gas_to_power[t])) for t, y in enumerate(years)},
        "gas_generation_twh_e_by_year":
            {int(y): float(pyo.value(m.gas_generation[t])) for t, y in enumerate(years)},
        "demand_twh_by_year": demand_dict,
        "unserved_twh_by_year": unserved_dict,
        "storage_discharge_twh_e_by_year":
            {int(y): float(pyo.value(m.storage_discharge[t])) for t, y in enumerate(years)},
        "solar_public_add_mw_by_year":
            {int(y): float(pyo.value(m.solar_public_add[t])) for t, y in enumerate(years)},
        "solar_eaas_add_mw_by_year":
            {int(y): float(pyo.value(m.solar_eaas_add[t])) for t, y in enumerate(years)},
        "annual_reliability_by_year": reliability_metrics["annual_reliability_by_year"],
        "worst_year_reliability": reliability_metrics["worst_year_reliability"],
        "horizon_reliability": reliability_metrics["horizon_reliability"],
        "expected_unserved_energy_twh": reliability_metrics["expected_unserved_energy_twh"],
        # FIN-3: public capital budget diagnostics
        "public_budget_shadow_usd_per_usd": public_budget_shadow,
        "public_budget_utilisation": public_budget_utilisation,
        "public_budget_realised_spend_usd": realised_public_spend,
        # STR-1: storage utilisation and constraint binding diagnostics
        # storage_throughput_twh: total energy discharged over the horizon.
        #   Measures how much work storage actually does in the optimal plan.
        "storage_throughput_twh": sum(
            float(pyo.value(m.storage_discharge[t])) for t in T
        ),
        # storage_final_capacity_mwh: end-of-horizon installed capacity.
        "storage_final_capacity_mwh": float(
            pyo.value(m.storage_capacity_mwh[len(years) - 1])
        ),
        # storage_utilisation_rate: fraction of theoretical maximum throughput used.
        #   = discharge_total / (final_capacity_mwh × H_d/1e6 × n_years)
        #   < 1.0: H_d is NOT the binding constraint (storage has unused capacity)
        #   ≈ 1.0: H_d IS binding (more deployable hours would increase throughput)
        "storage_utilisation_rate": (
            sum(float(pyo.value(m.storage_discharge[t])) for t in T)
            / max(
                float(pyo.value(m.storage_capacity_mwh[len(years) - 1]))
                * scenario.get("storage_deployable_hours_per_year", 700.0)
                / 1e6
                * len(years),
                1e-9,
            )
        ),
        # power_limit_years: years where H_d (power throughput) is the binding
        #   storage constraint — implies more deployable hours would help.
        # charge_limit_years: years where solar surplus fraction is the binding
        #   constraint — implies more solar or higher surplus_frac would help.
        "storage_power_limit_years": sum(
            1 for v in storage_binding.values() if v == "power_limit"
        ),
        "storage_charge_limit_years": sum(
            1 for v in storage_binding.values() if v == "charge_limit"
        ),
        "storage_energy_limit_years": sum(
            1 for v in storage_binding.values() if v == "energy_limit"
        ),
    }    

# ============================================================
# BATCH DETERMINISTIC EXECUTION
# ============================================================
# REL-1: Reliability feasibility matrix
# ============================================================

def run_rel1_feasibility_matrix(
    econ,
    gas_cases,
    solar_build_cases,
    capital_case="moderate",
    eps_levels=None,
):
    """
    REL-1: Reliability feasibility matrix.

    For every combination of (gas_deliverability_case, solar_build_case),
    runs the reliability sweep across eps_levels and records:
      - whether each eps level is feasible
      - the feasibility threshold (tightest feasible eps)
      - the NPV cost and reliability dual at each feasible eps

    The capital envelope (capital_case) and both the gas regime and solar
    build rate are explicit parameters so the experiment is fully reproducible.

    The RQ is answered by comparing feasibility_threshold across:
      - gas_cases   (primary hypothesis: gas regime drives threshold)
      - solar_build_cases  (control: solar level should NOT drive threshold)

    Parameters
    ----------
    econ : dict
    gas_cases : list[str]
        e.g. ["baseline", "upside", "downside", "shock_recovery"]
    solar_build_cases : list[str]
        e.g. ["aggressive", "conservative"]
        Must be keys in solar_build_scenarios().
    capital_case : str
        Fixed capital envelope. Default "moderate" (0.85 × B*).
    eps_levels : list[float] or None
        Unserved fraction tolerance levels to sweep (0=perfect reliability).
        If None, uses reliability_levels_log() = [0.20, 0.15, 0.10, 0.08, 0.06, 0.05].

    Returns
    -------
    list[dict]  — one row per (gas_case, solar_build_case, eps)
                  plus the feasibility_threshold summary rows
    """
    if eps_levels is None:
        eps_levels = reliability_levels_log()

    all_rows = []

    for gas_case in gas_cases:
        for solar_build_case in solar_build_cases:

            scenario = load_scenario(
                demand_level_case="served",
                demand_case="baseline",
                gas_deliverability_case=gas_case,
                capital_case=capital_case,
                solar_build_case=solar_build_case,
                carbon_case="no_policy",
                start_year=2025,
                end_year=2045,
            )
            years = scenario["years"]
            # Load time-varying solar CAPEX from NREL ATB (solar_low scenario).
            # solar_low declines from $1,456k/MW (2025) to $603k/MW (2045).
            solar_capex_tv = load_solar_capex_by_year(
                scenario_name="solar_low",
                start_year=int(years[0]),
                end_year=int(years[-1]),
            )

            # Activate minimum build floor when time-varying CAPEX is in use.
            # This prevents, the optimizer from delaying all solar to the cheapest years
            # (2040-2045) creating unrealistic 2025-2030 supply gaps.
            scenario["solar_min_build_mw_per_year"] = 100.0

            # --- VoLL-only baseline (no reliability constraint)
            m_base = build_model(
                scenario=scenario,
                econ=econ,
                reliability_max_unserved_fraction=None,
                solar_capex_by_year=solar_capex_tv,
            )
            solve_model(m_base)
            diag_base = extract_planning_diagnostics(m_base, scenario, econ)

            all_rows.append({
                "gas_case":         gas_case,
                "solar_build_case": solar_build_case,
                "capital_case":     capital_case,
                "eps":              None,
                "eps_label":        "voll_only",
                "status":           "feasible",
                "npv_cost":         float(pyo.value(m_base.system_cost_npv)),
                "unserved_total":   sum(diag_base["unserved_twh_by_year"].values()),
                "horizon_reliability": diag_base["horizon_reliability"],
                "dual_reliability": None,
                "gas_shadow_mean":  float(np.mean(
                    [v for v in diag_base["gas_shadow_price_usd_per_twh_th_by_year"].values()
                     if v is not None]
                )),
                "solar_total_mw":   sum(
                    float(pyo.value(m_base.solar_public_add[t]))
                    + float(pyo.value(m_base.solar_eaas_add[t]))
                    for t in range(len(scenario["years"]))
                ),
            })

            # --- Reliability-constrained sweep
            for eps in eps_levels:

                m = build_model(
                    scenario=scenario,
                    econ=econ,
                    reliability_max_unserved_fraction=eps,
                    reliability_mode="total",
                    solar_capex_by_year=solar_capex_tv,
                )

                try:
                    status = solve_model(m)
                    if not status["optimal"]:
                        raise RuntimeError("non-optimal")

                    diag = extract_planning_diagnostics(m, scenario, econ)
                    dual_rel = extract_reliability_dual(m)

                    all_rows.append({
                        "gas_case":         gas_case,
                        "solar_build_case": solar_build_case,
                        "capital_case":     capital_case,
                        "eps":              eps,
                        "eps_label":        f"eps_{eps:.2f}",
                        "status":           "feasible",
                        "npv_cost":         float(pyo.value(m.system_cost_npv)),
                        "unserved_total":   sum(diag["unserved_twh_by_year"].values()),
                        "horizon_reliability": diag["horizon_reliability"],
                        "dual_reliability": dual_rel,
                        "gas_shadow_mean":  float(np.mean(
                            [v for v in diag["gas_shadow_price_usd_per_twh_th_by_year"].values()
                             if v is not None]
                        )),
                        "solar_total_mw":   sum(
                            float(pyo.value(m.solar_public_add[t]))
                            + float(pyo.value(m.solar_eaas_add[t]))
                            for t in range(len(scenario["years"]))
                        ),
                    })

                except RuntimeError:
                    all_rows.append({
                        "gas_case":         gas_case,
                        "solar_build_case": solar_build_case,
                        "capital_case":     capital_case,
                        "eps":              eps,
                        "eps_label":        f"eps_{eps:.2f}",
                        "status":           "infeasible",
                        "npv_cost":         None,
                        "unserved_total":   None,
                        "horizon_reliability": None,
                        "dual_reliability": None,
                        "gas_shadow_mean":  None,
                        "solar_total_mw":   None,
                    })

    return all_rows


def compute_feasibility_threshold(rows):
    """
    From the output of run_rel1_feasibility_matrix(), compute the
    feasibility threshold per (gas_case, solar_build_case):
    the tightest eps (smallest unserved fraction allowed) that is
    still feasible.

    Returns a DataFrame with one row per (gas_case, solar_build_case).
    The key column is 'feasibility_threshold_eps':
      - None  => not even eps=0.05 is feasible (fully infeasible)
      - 0.05  => tightest standard (5% unserved) is still achievable
      - 0.20  => only the loosest standard (20%) is achievable

    This is the primary output for REL-1.
    """
    df = pd.DataFrame(rows)
    df_constrained = df[df["eps"].notnull()].copy()

    summary = []
    for (gas_case, solar_build_case), grp in df_constrained.groupby(
        ["gas_case", "solar_build_case"]
    ):
        feasible = grp[grp["status"] == "feasible"].copy()
        infeasible = grp[grp["status"] == "infeasible"].copy()

        # Tightest feasible eps = smallest eps value that is feasible
        threshold = float(feasible["eps"].min()) if len(feasible) > 0 else None

        # Cost at threshold
        if threshold is not None:
            threshold_row = feasible[feasible["eps"] == threshold].iloc[0]
            cost_at_threshold = threshold_row["npv_cost"]
            dual_at_threshold = threshold_row["dual_reliability"]
            solar_at_threshold = threshold_row["solar_total_mw"]
        else:
            cost_at_threshold = None
            dual_at_threshold = None
            solar_at_threshold = None

        summary.append({
            "gas_case":                  gas_case,
            "solar_build_case":          solar_build_case,
            "feasibility_threshold_eps": threshold,
            # None means fully infeasible at all eps levels
            "fully_infeasible":          threshold is None,
            "n_feasible_eps":            len(feasible),
            "n_infeasible_eps":          len(infeasible),
            "npv_cost_at_threshold":     cost_at_threshold,
            "dual_at_threshold":         dual_at_threshold,
            "solar_mw_at_threshold":     solar_at_threshold,
        })

    return pd.DataFrame(summary)


# ============================================================
# GAS-1 / GAS-3: Gas regime × NDC matrix runner
# ============================================================

def run_gas_regime_ndc_matrix(
    econ,
    ndc_cap_scenario,
    gas_cases,
    financing_regime="traditional",
    required_margin=1.10,
    capital_case="moderate",
    cap_path="data/cost/processed/emissions_cap.csv",
):
    """
    GAS-1 + GAS-3 shared runner.

    Solves the NDC-constrained optimisation for every gas_deliverability_case
    in gas_cases, holding the NDC cap scenario, financing regime, and capital
    envelope constant. Returns one result dict per gas case with full
    diagnostics needed for:

      GAS-1: gas shadow price vs export and domestic benchmarks
      GAS-3: feasibility, cost, and carbon shadow across gas regime shapes

    Parameters
    ----------
    econ : dict
    ndc_cap_scenario : str
        e.g. "ndc3_unconditional"
    gas_cases : list[str]
        Subset of ["baseline","upside","downside","shock_recovery"]
    financing_regime : str
        "traditional" (GAS-1/GAS-3) or "eaas" (GAS-2 comparison arm)
    required_margin : float
    capital_case : str
    cap_path : str

    Returns
    -------
    list[dict]
    """
    import copy as _copy

    cap_df = pd.read_csv(cap_path)

    results = []

    for gas_case in gas_cases:

        scenario = load_scenario(
            demand_level_case="served",
            demand_case="baseline",
            gas_deliverability_case=gas_case,
            capital_case=capital_case,
            carbon_case="no_policy",
            start_year=2025,
            end_year=2045,
        )

        scenario["financing_regime"] = financing_regime
        if financing_regime == "eaas":
            scenario["required_margin"] = required_margin

        years = [int(y) for y in scenario["years"]]

        # Load NDC annual caps
        cap_sub = cap_df[
            (cap_df["scenario"] == ndc_cap_scenario)
            & (cap_df["year"].isin(years))
        ].sort_values("year")

        if len(cap_sub) != len(years):
            raise ValueError(
                f"Cap length {len(cap_sub)} != {len(years)} for "
                f"scenario='{ndc_cap_scenario}'"
            )
        caps = cap_sub["cap_tco2"].astype(float).tolist()

        # Load time-varying solar CAPEX from NREL ATB (solar_low scenario).
        # solar_low declines from $1,456k/MW (2025) to $603k/MW (2045).
        solar_capex_tv = load_solar_capex_by_year(
            scenario_name="solar_low",
            start_year=int(years[0]),
            end_year=int(years[-1]),
        )

        # Activate minimum build floor when time-varying CAPEX is in use.
        # This prevents, the optimizer from delaying all solar to the cheapest years
        # (2040-2045) creating unrealistic 2025-2030 supply gaps.
        scenario["solar_min_build_mw_per_year"] = 100.0

        # Build and solve baseline model with an effectively non-binding cumulative cap

        m = build_model(
            scenario=scenario,
            econ=econ,
            emissions_cap_by_year=caps,
            solar_capex_by_year=solar_capex_tv,
        )

        try:
            status = solve_model(m)
            if not status["optimal"]:
                raise RuntimeError("non-optimal termination")

            diag = extract_planning_diagnostics(m, scenario, econ)

            gas_shadow = diag["gas_shadow_price_usd_per_twh_th_by_year"]
            carbon_shadow = diag["carbon_shadow_price_usd_per_tco2_by_year"]

            gs_vals = [v for v in gas_shadow.values() if v is not None]
            cs_vals = [v for v in carbon_shadow.values() if v is not None]

            # Export/domestic benchmark comparison (USD/TWh_th)
            # Conversion: 1 MMBtu = 293.071 kWh_th => 1 TWh_th = 3,412,142 MMBtu
            MMBTU_PER_TWH_TH = 1e9 / 293.071
            BENCHMARKS = {
                "dom_power_2p4":    2.4  * MMBTU_PER_TWH_TH,
                "dom_commercial_2p9": 2.9 * MMBTU_PER_TWH_TH,
                "dom_high_3p3":     3.3  * MMBTU_PER_TWH_TH,
                "opp_3":            3.0  * MMBTU_PER_TWH_TH,
                "opp_5":            5.0  * MMBTU_PER_TWH_TH,
                "opp_7":            7.0  * MMBTU_PER_TWH_TH,
                "lng_export_10":   10.0  * MMBTU_PER_TWH_TH,
            }

            benchmark_comparison = {}
            for bench_name, bench_val in BENCHMARKS.items():
                exceeding = [v for v in gs_vals if v > bench_val]
                benchmark_comparison[bench_name] = {
                    "benchmark_usd_per_twh_th": bench_val,
                    "share_years_shadow_gt_benchmark": (
                        len(exceeding) / len(gs_vals) if gs_vals else None
                    ),
                    "max_gap_usd_per_twh_th": (
                        float(max(v - bench_val for v in gs_vals)) if gs_vals else None
                    ),
                    "mean_gap_usd_per_twh_th": (
                        float(np.mean([v - bench_val for v in gs_vals])) if gs_vals else None
                    ),
                }

            results.append({
                "gas_case":             gas_case,
                "ndc_scenario":         ndc_cap_scenario,
                "financing_regime":     financing_regime,
                "status":               "optimal",
                # GAS-1: scarcity rent
                "gas_shadow_mean":      float(np.mean(gs_vals)) if gs_vals else 0.0,
                "gas_shadow_max":       float(max(gs_vals)) if gs_vals else 0.0,
                "gas_shadow_binding_years": sum(1 for v in gs_vals if v > 1e-6),
                "gas_shadow_by_year":   gas_shadow,
                "benchmark_comparison": benchmark_comparison,
                # GAS-2/GAS-3: constraint interaction
                "carbon_shadow_mean":   float(np.mean(cs_vals)) if cs_vals else 0.0,
                "carbon_shadow_max":    float(max(cs_vals)) if cs_vals else 0.0,
                "carbon_binding_years": sum(1 for v in cs_vals if v > 1e-6),
                # GAS-3: cost and access outcomes
                "npv_total_cost_usd":   float(pyo.value(m.system_cost_npv)),
                "cumulative_unserved_twh": sum(diag["unserved_twh_by_year"].values()),
                "cumulative_emissions_tco2": float(pyo.value(m.emissions)),
                "horizon_reliability":  diag["horizon_reliability"],
                "solar_total_mw":       sum(
                    float(pyo.value(m.solar_public_add[t]))
                    + float(pyo.value(m.solar_eaas_add[t]))
                    for t in range(len(years))
                ),
                # GAS-3: annual NDC binding diagnostic
                # Records year-by-year gas availability, actual emissions,
                # annual cap, cap slack, and carbon shadow price.
                # This is needed to answer whether the NDC cap binds in the
                # same years as gas disruption (shock_recovery) or progressively
                # tightens (downside). A scalar mean cannot distinguish these.
                "annual_ndc_timeseries": [
                    {
                        "year":              int(years[t]),
                        "gas_avail_twh_th":  diag["gas_avail_twh_th_by_year"][int(years[t])],
                        "gas_to_power_twh_th": diag["gas_to_power_twh_th_by_year"][int(years[t])],
                        "emissions_tco2":    diag["emissions_tco2_by_year"][int(years[t])],
                        "annual_cap_tco2":   caps[t],
                        "cap_slack_tco2":    caps[t] - diag["emissions_tco2_by_year"][int(years[t])],
                        "carbon_shadow":     diag["carbon_shadow_price_usd_per_tco2_by_year"][int(years[t])],
                        "unserved_twh":      diag["unserved_twh_by_year"][int(years[t])],
                    }
                    for t in range(len(years))
                ],
            })

        except RuntimeError as e:
            results.append({
                "gas_case":         gas_case,
                "ndc_scenario":     ndc_cap_scenario,
                "financing_regime": financing_regime,
                "status":           f"infeasible: {e}",
            })

    return results


# ============================================================

def run_all_deterministic_scenarios(
    econ,
    start_year=2025,
    end_year=2045,
):

    results = []

    for demand_level_case in demand_level_scenarios():
        for demand_case in demand_growth_scenarios():
            for gas_case in gas_deliverability_scenarios():
                # for solar_case in solar_capacity_scenarios():
                for carbon_case in carbon_policy_scenarios():

                    scenario = load_scenario(
                        demand_level_case=demand_level_case,
                        demand_case=demand_case,
                        gas_deliverability_case=gas_case,
                        #solar_case=solar_case,
                        carbon_case=carbon_case,
                        start_year=start_year,
                        end_year=end_year,
                    )
                    output = run_deterministic_scenario(
                        scenario=scenario,
                        econ=econ,
                        capacity_paths={
                            "solar_mw": [scenario["solar_baseline_mw"]] * len(scenario["years"]),
                            "storage_mwh": [0.0] * len(scenario["years"]),
                        }
                    )
                        
                    results.append({
                        "scenario_labels": {
                            "demand_level": demand_level_case,
                            "demand": demand_case,
                            "gas": gas_case,
                            "carbon": carbon_case,
                        },
                        "total_cost_usd": output["costs"]["total"],
                        "gas_cost_usd": output["costs"]["gas"],
                        "solar_cost_usd": output["costs"]["solar"],
                        "carbon_cost_usd": output["costs"]["carbon"],
                        "unserved_cost_usd": output["costs"]["unserved"],
                        "total_unserved_energy_twh":
                            float(np.sum(output["unserved"])),
                    })

    return results


# frontier Experiment
def run_tariff_bankability_sweep(
    base_scenario,
    econ,
    tariff_grid,
    ndc_cap_scenario,
    cap_path="data/cost/processed/emissions_cap.csv",
):
    """
    FIN-2: Tariff bankability sweep.

    For each tariff level in tariff_grid, solves the NDC-constrained EaaS
    optimisation and records:
      - whether EaaS is self-financing (financing_gap_per_mw == 0)
      - total eaas_subsidy deployed (USD NPV)
      - subsidy per MW of EaaS solar built
      - total EaaS solar deployed (MW)
      - system NPV cost
      - cumulative unserved energy

    Called once per NDC scenario (unconditional / conditional) so that
    the required_margin and capital_case already embedded in base_scenario
    are respected — the sweep varies ONLY the tariff.

    Parameters
    ----------
    base_scenario : dict
        Fully configured scenario dict with financing_regime="eaas",
        required_margin, capital_case, etc. already set.
        solar_service_tariff_usd_per_twh will be overwritten per tariff.
    econ : dict
        Economics dict from load_econ().
    tariff_grid : list[float]
        Tariff levels to sweep (USD/TWh). Use TARIFF_SWEEP_GRID from scenarios.py.
    ndc_cap_scenario : str
        Scenario name in emissions_cap.csv, e.g. "ndc3_unconditional".
    cap_path : str
        Path to emissions_cap.csv.

    Returns
    -------
    list[dict]  — one row per tariff level.
    """
    import copy as _copy

    years = [int(y) for y in base_scenario["years"]]

    # Load annual caps — if ndc_cap_scenario is None, no emissions cap is applied
    # (baseline_no_policy arm). Otherwise load from CSV.
    if ndc_cap_scenario is None:
        caps = None
    else:
        cap_df = pd.read_csv(cap_path)
        cap_df = cap_df[
            (cap_df["scenario"] == ndc_cap_scenario)
            & (cap_df["year"].isin(years))
        ].sort_values("year")

        if len(cap_df) != len(years):
            raise ValueError(
                f"Cap length {len(cap_df)} != model years {len(years)} "
                f"for scenario '{ndc_cap_scenario}'. Run 00_build_emissions_cap.py first."
            )

        caps = cap_df["cap_tco2"].astype(float).tolist()

    # Pre-compute the analytical self-financing threshold for this scenario
    # so we can annotate each row without re-solving.
    solar_cf = base_scenario["solar_cf"]
    solar_energy_per_mw = solar_cf * 8760 / 1_000_000
    r = float(base_scenario.get("discount_rate", 0.10))
    npv_energy = sum(
        solar_energy_per_mw / (1.0 + r) ** t
        for t in range(len(years))
    )
    required_margin = base_scenario["required_margin"]
    capex = econ["SOLAR_CAPEX_PER_MW"]
    threshold_usd_per_twh = capex * required_margin / npv_energy

    results = []

    # Load time-varying solar CAPEX from NREL ATB (solar_low scenario).
    # solar_low declines from $1,456k/MW (2025) to $603k/MW (2045).
    solar_capex_tv = load_solar_capex_by_year(
        scenario_name="solar_low",
        start_year=int(years[0]),
        end_year=int(years[-1]),
    )

    for tariff in tariff_grid:

        scenario_t = _copy.deepcopy(base_scenario)
        scenario_t["solar_service_tariff_usd_per_twh"] = tariff
        
        # Activate minimum build floor when time-varying CAPEX is in use.
        # This prevents, the optimizer from delaying all solar to the cheapest years
        # (2040-2045) creating unrealistic 2025-2030 supply gaps.
        scenario_t["solar_min_build_mw_per_year"] = 100.0
        # Analytical financing gap for this tariff (does not require solving)
        max_bankable = tariff * npv_energy / required_margin
        financing_gap_per_mw = max(0.0, capex - max_bankable)
        is_self_financing = financing_gap_per_mw < 1.0   # < 1 USD/MW = effectively zero

        if caps is not None:
            m = build_model(
                scenario=scenario_t,
                econ=econ,
                emissions_cap_by_year=caps,
                solar_capex_by_year=solar_capex_tv,
            )
        else:
            m = build_model(
                scenario=scenario_t,
                econ=econ,
                emissions_cap=1e18,
                solar_capex_by_year=solar_capex_tv
            )

        status = solve_model(m)

        if not status["optimal"]:
            results.append({
                "ndc_scenario":           ndc_cap_scenario if ndc_cap_scenario is not None else "baseline_no_policy",
                "tariff_usd_per_twh":     tariff,
                "tariff_m_usd_per_twh":   tariff / 1e6,
                "required_margin":        required_margin,
                "threshold_usd_per_twh":  threshold_usd_per_twh,
                "financing_gap_per_mw":   financing_gap_per_mw,
                "is_self_financing":      is_self_financing,
                "status": "infeasible",
            })
            continue

        diag = extract_planning_diagnostics(m, scenario_t, econ)

        # Total discounted eaas_subsidy actually deployed by the optimizer
        eaas_subsidy_npv = sum(
            float(pyo.value(m.DF[t])) * float(pyo.value(m.eaas_subsidy[t]))
            for t in range(len(years))
        )

        solar_eaas_total_mw = sum(
            float(pyo.value(m.solar_eaas_add[t]))
            for t in range(len(years))
        )

        # Realised subsidy per MW (zero if no EaaS built)
        realised_subsidy_per_mw = (
            eaas_subsidy_npv / solar_eaas_total_mw
            if solar_eaas_total_mw > 1e-3
            else 0.0
        )

        results.append({
            "ndc_scenario":                ndc_cap_scenario,
            "tariff_usd_per_twh":          tariff,
            "tariff_m_usd_per_twh":        tariff / 1e6,
            "required_margin":             required_margin,
            # Analytical fields (no solve needed — here for cross-check)
            "threshold_usd_per_twh":       threshold_usd_per_twh,
            "financing_gap_per_mw":        financing_gap_per_mw,
            "is_self_financing":           is_self_financing,
            # Solved results
            "npv_total_cost_usd":          float(pyo.value(m.system_cost_npv)),
            "cumulative_unserved_twh":     sum(diag["unserved_twh_by_year"].values()),
            "cumulative_emissions_tco2":   float(pyo.value(m.emissions)),
            "solar_eaas_total_mw":         solar_eaas_total_mw,
            "solar_public_total_mw":       sum(
                float(pyo.value(m.solar_public_add[t])) for t in range(len(years))
            ),
            "eaas_subsidy_npv_usd":        eaas_subsidy_npv,
            "realised_subsidy_per_mw_usd": realised_subsidy_per_mw,
            "horizon_reliability":         diag["horizon_reliability"],
            "gas_shadow_avg_usd_per_twh_th": float(np.mean(
                [v for v in diag["gas_shadow_price_usd_per_twh_th_by_year"].values()
                 if v is not None]
            )),
            "status": "optimal",
        })

    return results


def run_tariff_public_capital_frontier(
    scenario,
    econ,
    tariffs,
    reliability_target=0.995,
):

    frontier = []

    # candidate capital envelopes (USD NPV)
    capital_grid = [
        0.0,
        0.5e9,
        1.0e9,
        2.0e9,
        3.0e9,
        4.0e9,
        5.0e9,
        6.0e9,
    ]

    for tariff in tariffs:

        scenario_copy = copy.deepcopy(scenario)
        scenario_copy["solar_service_tariff_usd_per_twh"] = tariff
        scenario_copy["financing_regime"] = "eaas"

        required_public_capital = None
        achieved_reliability = None

        for cap in capital_grid:

            scenario_copy["public_solar_budget_npv"] = cap

            m = build_model(
                scenario=scenario_copy,
                econ=econ,
                emissions_cap=1e18,
                emissions_cap_by_year=None,
            )

            solve_model(m)

            diag = extract_planning_diagnostics(m, scenario_copy, econ)

            achieved_reliability = diag["worst_year_reliability"]

            if achieved_reliability >= reliability_target:

                required_public_capital = cap
                break

        frontier.append({
            "tariff": tariff,
            "required_public_capital": required_public_capital,
            "reliability": achieved_reliability
        })

    return frontier


def run_financing_vs_resource_test(econ):

    cases = [
        {"name": "A_highGas_highCapital",
         "gas": "upside",
         "capital": "expansion"},

        {"name": "B_highGas_lowCapital",
         "gas": "upside",
         "capital": "tight"},

        {"name": "C_lowGas_highCapital",
         "gas": "downside",
         "capital": "expansion"},

        {"name": "D_lowGas_lowCapital",
         "gas": "downside",
         "capital": "tight"},
    ]
    results = []

    for case in cases:
    
        for build_case in solar_build_scenarios().keys():

            scenario = load_scenario(
                demand_level_case="served",
                demand_case="baseline",
                gas_deliverability_case=case["gas"],
                capital_case=case["capital"],
                solar_tariff_case="low",
                solar_build_case=build_case
            )

            scenario["financing_regime"] = "eaas"

            m = build_model(
                scenario=scenario,
                econ=econ,
                emissions_cap=1e18
            )

            solve_model(m)

            diagnostics = extract_planning_diagnostics(m, scenario, econ)
            summary = summarize_run(m, diagnostics, scenario)

            run_id = f'{case["name"]}_{build_case}'
            write_run_outputs(run_id, diagnostics, summary)

            results.append({
                "case": case["name"],
                "solar_build_case": build_case,
                "gas_scenario": case["gas"],
                "capital_scenario": case["capital"],

                "horizon_reliability": diagnostics["horizon_reliability"],
                "worst_year_reliability": diagnostics["worst_year_reliability"],
                "expected_unserved_energy_twh": diagnostics["expected_unserved_energy_twh"],

                "solar_total_mw": summary["solar_total_built_mw"],
                "gas_shadow_avg": summary["gas_shadow_avg_usd_per_twh_th"],
            })

    return results