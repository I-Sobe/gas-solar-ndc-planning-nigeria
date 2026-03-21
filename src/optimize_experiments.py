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


def extract_reliability_dual(m):
    """
    Returns marginal cost of tightening reliability (USD per TWh).
    Only valid if constraint exists and is binding.
    """

    if not hasattr(m, "reliability_constraint"):
        return None

    # Case 1: scalar constraint (total mode)
    if not m.reliability_constraint.is_indexed():
        dual = m.dual.get(m.reliability_constraint, None)
        if dual is None:
            return None
        return max(0.0, float(-dual))  # sign convention

    # Case 2: indexed (annual mode)
    vals = []
    for k in m.reliability_constraint:
        dual = m.dual.get(m.reliability_constraint[k], None)
        if dual is not None:
            vals.append(max(0.0, float(-dual)))

    return sum(vals)/len(vals) if vals else None


def run_reliability_sweep(scenario, econ):
    """
    Runs:
    - Baseline (VoLL only)
    - Constrained reliability cases

    Returns comparable outputs
    """
    eps_levels = reliability_levels_log()
    results = []

    # -----------------------------
    # 1. Baseline (VoLL only)
    # -----------------------------
    m_base = build_model(
        scenario=scenario,
        econ=econ,
        reliability_max_unserved_fraction=None
    )
        
    solve_model(m_base)
    diag_base = extract_planning_diagnostics(m_base, scenario, econ)
    print("\n=== BASELINE DIAGNOSTICS ===")
    print("Baseline reliability:", diag_base["horizon_reliability"])
    print("Total unserved (TWh):", sum(diag_base["unserved_twh_by_year"].values()))

    results.append({
        "case": "voll_only",
        "eps": None,
        "npv_cost": float(pyo.value(m_base.system_cost_npv)),
        "unserved_total": sum(diag_base["unserved_twh_by_year"].values()),
        "diagnostics": diag_base,
        "dual_reliability": None,
        "status": "feasible"
    })

    # -----------------------------
    # 2. Reliability-constrained runs
    # -----------------------------
    for eps in eps_levels:

        m = build_model(
            scenario=scenario,
            econ=econ,
            reliability_max_unserved_fraction=eps,
            reliability_mode="total"
        )

        try:
            solve_model(m)

            diag = extract_planning_diagnostics(m, scenario, econ)

            if eps == 0.10:
                print("\n=== BOTTLENECK DIAGNOSTICS (eps=0.10) ===")

                print("\nGas shadow prices:")
                print(diag["gas_shadow_price_usd_per_twh_th_by_year"])

                print("\nLand shadow prices:")
                print(diag["land_shadow_price_by_year"])

                print("\nStorage binding:")
                print(diag["storage_binding_by_year"])

                bottleneck = identify_bottleneck(diag)
                print("\n>>> IDENTIFIED BOTTLENECK:", bottleneck)

            dual_rel = extract_reliability_dual(m)

            results.append({
                "case": "reliability_constrained",
                "eps": eps,
                "npv_cost": float(pyo.value(m.system_cost_npv)),
                "unserved_total": sum(diag["unserved_twh_by_year"].values()),
                "diagnostics": diag,
                "dual_reliability": dual_rel,
                "status": "feasible"
            })

        except RuntimeError:
            results.append({
                "case": "reliability_constrained",
                "eps": eps,
                "npv_cost": None,
                "unserved_total": None,
                "diagnostics": None,
                "dual_reliability": None,
                "status": "infeasible"
            })
    
    return results


def identify_bottleneck(diag):

    gas_shadow = diag["gas_shadow_price_usd_per_twh_th_by_year"]
    land_shadow = diag["land_shadow_price_by_year"]
    
    gas_binding_years = sum(1 for v in gas_shadow.values() if v and v > 1e-6)
    land_binding_years = sum(1 for v in land_shadow.values() if v and v > 1e-6)
    
    
    if gas_binding_years > 0.3 * len(gas_shadow):
        return "gas_constraint"

    if land_binding_years > 0.3 * len(land_shadow):
        return "land_constraint"

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
    Convert reliability sweep results into a flat dataframe
    for plotting and export.
    """

    rows = []

    for r in results:
        rows.append({
            "eps": r["eps"],
            "reliability": None if r["eps"] is None else 1 - r["eps"],
            "npv_cost": r["npv_cost"],
            "unserved_total": r["unserved_total"],
            "dual_reliability": r["dual_reliability"],
            "case": r["case"],
            "status": r.get("status", "feasible")
        })

    return pd.DataFrame(rows)

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

    solar_lcoe = econ["SOLAR_CAPEX_PER_MW"] / npv_energy if npv_energy > 0 else None
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
            * econ["SOLAR_CAPEX_PER_MW"]
        )

    eaas_multiplier = None

    if public_spending > 0:
        eaas_multiplier = private_investment / public_spending
    
    demand_dict = {int(y): float(demand[t]) for t, y in enumerate(years)}

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
    peak_demand_multiple = scenario.get("peak_demand_multiple", 2.5)
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

    return {
        "peak_adequacy_by_year": peak_adequacy_by_year,
        "storage_binding_by_year": storage_binding,
        "subsidy_per_mw_usd_by_year": subsidy_per_mw_by_year,
        "solar_lcoe_usd_per_twh": solar_lcoe,
        "solar_capex_usd_per_mw": econ["SOLAR_CAPEX_PER_MW"],
        "solar_tariff_usd_per_twh": scenario["solar_service_tariff_usd_per_twh"],
        "solar_beats_gas_year": solar_beats_gas_year,
        "value_of_gas_to_power_usd_per_twh_e_by_year": gas_value_electricity,
        "storage_charge_twh_e_by_year":
            {int(y): float(pyo.value(m.storage_charge[t]))
             for t, y in enumerate(years)},
        "bankability_threshold":
            scenario["required_margin"] *
            (
                econ["SOLAR_CAPEX_PER_MW"] /
                sum(pyo.value(m.DF[t]) for t in range(len(years)))
            ),
        "solar_generation_share": solar_share,
        "gas_scarcity_value_usd": gas_scarcity_value,
        "eaas_public_spending_usd": public_spending,
        "eaas_private_investment_usd": private_investment,
        "eaas_capital_multiplier": eaas_multiplier,
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
    }

# ============================================================
# PARETO FRONTIERS
# ============================================================

def generate_weighted_pareto(scenario, econ, weight_grid):

    results = []
    years = scenario["years"]

    for w_cost, w_em in weight_grid:

        if w_cost == 0.0 and w_em == 0.0:
            raise ValueError("Invalid weight pair (0,0).")

        m = build_model(
            scenario=scenario,
            econ=econ,
            weight_cost=w_cost,
            weight_emissions=w_em,
        )

        status = solve_model(m)

        if not status["optimal"]:
            raise RuntimeError("Optimization failed.")

        assert_duals_present(m, scenario, require_gas_duals=True, require_carbon_duals=False)
        
        diagnostics = extract_planning_diagnostics(m, scenario, econ)
        summary = summarize_run(m, diagnostics, scenario)

        run_id = f"pareto_wcost{w_cost:.3f}_wem{w_em:.3f}"
        write_run_outputs(run_id, diagnostics, summary)

        results.append({
            "weights": {"cost": w_cost, "emissions": w_em},
            "decision_variables": {
            "solar_add_mw_by_year":
                {
                    int(y):
                        float(pyo.value(m.solar_public_add[t]))
                        + float(pyo.value(m.solar_eaas_add[t]))
                    for t,y in enumerate(years)
                },
                "final_storage_capacity_mwh": float(pyo.value(m.storage_capacity_mwh[len(years)-1])),
                "solar_total_built_mw": sum(
                    float(pyo.value(m.solar_public_add[t])) 
                    + float(pyo.value(m.solar_eaas_add[t])) for t in range(len(years))),
            },
            "objectives": {
                "emissions_tco2":
                    float(pyo.value(m.emissions)),
            },
            "diagnostics": diagnostics,
        })
        
    return {"results": results}


def generate_epsilon_pareto(scenario, econ, emissions_caps):

    results = []
    years = scenario["years"]
    
    for eps in emissions_caps:

        m = build_model(
            scenario=scenario,
            econ=econ,
            emissions_cap=eps,
        )

        status = solve_model(m)

        if not status["optimal"]:
            raise RuntimeError(
                f"Optimization failed for emissions cap {eps}"
            )

        # If epsilon is intentionally non-binding (baseline proxy), don't require carbon duals.
        require_carbon = float(eps) < 1e15  # threshold: anything huge is treated as "no cap"
        assert_duals_present(m, scenario, require_gas_duals=True, require_carbon_duals=require_carbon)
        
        diagnostics = extract_planning_diagnostics(m, scenario, econ)
        summary = summarize_run(m, diagnostics, scenario)
        run_id = f"pareto_eps{float(eps):.3e}"
        write_run_outputs(run_id, diagnostics, summary)
        
        results.append({
            "emissions_cap_tco2": float(eps),
            "actual_emissions_tco2":
                float(pyo.value(m.emissions)),
            "decision_variables": {
            "solar_add_mw_by_year":
                {
                    int(y):
                        float(pyo.value(m.solar_public_add[t]))
                        + float(pyo.value(m.solar_eaas_add[t]))
                    for t,y in enumerate(years)
                },
                "final_storage_capacity_mwh": float(pyo.value(m.storage_capacity_mwh[len(years)-1])),
                "solar_total_built_mw": sum(
                    float(pyo.value(m.solar_public_add[t])) 
                    + float(pyo.value(m.solar_eaas_add[t])) for t in range(len(years))),
            },
            "diagnostics": diagnostics,
        })
        

    return {"results": results}

# ===========================================================
# Structured sensitivity runner
# ===========================================================
def run_sensitivity_matrix(base_scenario, econ_sets, storage_sets, cap_sets, out_csv):
    rows = []
    
    for econ_case in econ_sets:      # gas price + VoLL combos (controlled)
        for storage_case in storage_sets:
            for cap_case in cap_sets:
                scenario = dict(base_scenario)
                years = base_scenario["years"]


                # Apply storage proxy settings
                scenario["storage_deployable_hours_per_year"] = storage_case["deployable_hours"]

                # Cap settings handled by passing emissions_cap_by_year to build_model (if any)
                emissions_caps = cap_case.get("emissions_cap_by_year", None)

                m = build_model(
                    scenario=scenario,
                    econ=econ_case["econ"],
                    emissions_cap_by_year=emissions_caps,
                )
                status = solve_model(m)
                if not status["optimal"]:
                    raise RuntimeError(f"Sensitivity run failed: {econ_case['name']}, {storage_case['name']}, {cap_case['name']}")

                assert_duals_present(
                    m, scenario,
                    require_gas_duals=True,
                    require_carbon_duals=cap_case.get("emissions_cap_by_year") is not None
                )
                diag = extract_planning_diagnostics(m, scenario, econ=econ_case["econ"])
                reliability = compute_reliability_metrics(
                    diag["demand_twh_by_year"],
                    diag["unserved_twh_by_year"]
                )
                # reliability = compute_reliability_metrics(diag)
                summary = summarize_run(m, diag, scenario)
                run_id = f"sens_{econ_case['name']}_{storage_case['name']}_{cap_case['name']}"
                write_run_outputs(run_id, diag, summary)
                
                rows.append({
                    "econ_case": econ_case["name"],
                    "storage_case": storage_case["name"],
                    "cap_case": cap_case["name"],
                    "npv_total_cost_usd": float(pyo.value(m.system_cost_npv)),
                    "cumulative_emissions_tco2": float(pyo.value(m.emissions)),
                    "cumulative_unserved_twh": sum(diag["unserved_twh_by_year"].values()),
                    "solar_total_built_mw": sum(
                        float(pyo.value(m.solar_public_add[t])) 
                        + float(pyo.value(m.solar_eaas_add[t])) for t in range(len(years))),
                    "final_storage_capacity_mwh": float(pyo.value(m.storage_capacity_mwh[len(years)-1])),
                    "avg_gas_shadow_usd_per_twh_th": np.mean(
                        [v for v in diag["gas_shadow_price_usd_per_twh_th_by_year"].values() if v is not None]
                    ),
                    "avg_carbon_shadow_usd_per_tco2": np.mean(
                        [v for v in diag["carbon_shadow_price_usd_per_tco2_by_year"].values()if v is not None]
                    ) if emissions_caps is not None else None,
                })

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return rows

# ============================================================
# BATCH DETERMINISTIC EXECUTION
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
                for carbon_case in carbon_policy_scenarios():

                    scenario = load_scenario(
                        demand_level_case=demand_level_case,
                        demand_case=demand_case,
                        gas_deliverability_case=gas_case,
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