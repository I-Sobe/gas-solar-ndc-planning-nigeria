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

from src.gas_supply import gas_available_power
from src.optimize_model import build_model, solve_model
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
    solar_capacity_scenarios,
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

    diagnostics = extract_planning_diagnostics(m, scenario)
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

    gas_gen = gas_profile["available_twh_th"]

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

def extract_planning_diagnostics(m, scenario, econ=None):

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

    solar_lcoe = econ["SOLAR_CAPEX_PER_MW"] / npv_energy

    gas_mc = econ["GAS_COST_PER_TWH_TH"] / scenario["gas_eta"]

    solar_beats_gas_year = None

    if solar_lcoe < gas_mc:
        solar_beats_gas_year = int(years[0])
    
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

    return {
        "solar_lcoe_usd_per_twh": solar_lcoe,

        "solar_capex_usd_per_mw":
            econ["SOLAR_CAPEX_PER_MW"],

        "solar_tariff_usd_per_twh":
            scenario["solar_service_tariff_usd_per_twh"],
        "solar_beats_gas_year": solar_beats_gas_year,
        "value_of_gas_to_power_usd_per_twh_e_by_year": gas_value_electricity,
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
        "land_shadow_price_by_year": {int(y): float(-m.dual.get(m.land_constraint[t], 0.0))for t, y in enumerate(years)},
        
        "solar_capacity_mw_by_year": {int(y): float(pyo.value(m.solar_capacity_mw[t])) for t, y in enumerate(years)},
        "solar_generation_twh_e_by_year": {int(y): float(pyo.value(m.solar_generation[t])) for t, y in enumerate(years)},
        "discount_factor_by_year": {int(y): float(pyo.value(m.DF[t])) for t, y in enumerate(years)},
        "emissions_tco2_by_year": {
            int(y): float(pyo.value(m.emissions_by_year[t])) for t, y in enumerate(years)
        },
        "carbon_shadow_price_usd_per_tco2_by_year": carbon_shadow, 
        
        "gas_avail_twh_th_by_year":
            _series_dict_by_year(gas_avail, years),

        "bankable_capex_per_mw":
            scenario["solar_service_tariff_usd_per_twh"] *
            npv_energy
            / scenario["required_margin"],

        "gas_to_power_twh_th_by_year":
            {int(y): float(pyo.value(m.gas_to_power[t]))
             for t, y in enumerate(years)},

        "gas_generation_twh_e_by_year":
            {int(y): float(pyo.value(m.gas_generation[t]))
             for t, y in enumerate(years)},
        "demand_twh_by_year": demand_dict,
        "unserved_twh_by_year": unserved_dict,
        
        "storage_discharge_twh_e_by_year":
            {int(y): float(pyo.value(m.storage_discharge[t]))
             for t, y in enumerate(years)},

        "solar_public_add_mw_by_year":
            {int(y): float(pyo.value(m.solar_public_add[t]))
            for t, y in enumerate(years)},

        "solar_eaas_add_mw_by_year":
            {int(y): float(pyo.value(m.solar_eaas_add[t]))
            for t, y in enumerate(years)},
        
        "storage_binding_by_year":
            {
                int(y):
                    "cycle_limit"
                    if abs(
                        pyo.value(m.storage_cycle_limit[t].body)
                        - pyo.value(m.storage_cycle_limit[t].upper)
                    ) <= 1e-6
                    else "solar_limit"
                    if abs(
                        pyo.value(m.storage_solar_limit[t].body)
                        - pyo.value(m.storage_solar_limit[t].upper)
                    ) <= 1e-6
                    else "power_limit"
                    if abs(
                        pyo.value(m.storage_power_limit[t].body)
                        - pyo.value(m.storage_power_limit[t].upper)
                    ) <= 1e-6
                    else "none"
                for t, y in enumerate(years)
            },
        # reliability diagnostics
        "annual_reliability_by_year":
            reliability_metrics["annual_reliability_by_year"],

        "worst_year_reliability":
            reliability_metrics["worst_year_reliability"],

        "horizon_reliability":
            reliability_metrics["horizon_reliability"],

        "expected_unserved_energy_twh":
            reliability_metrics["expected_unserved_energy_twh"],
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
        
        diagnostics = extract_planning_diagnostics(m, scenario)
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
        
        diagnostics = extract_planning_diagnostics(m, scenario)
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

                # Apply storage proxy settings
                scenario["storage_cycles_per_year"] = storage_case["cycles"]
                scenario["storage_critical_hours_per_year"] = storage_case["critical_hours"]

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
                diag = extract_planning_diagnostics(m, scenario)
                reliability = compute_reliability_metrics(diag)
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
                for solar_case in solar_capacity_scenarios():
                    for carbon_case in carbon_policy_scenarios():

                        scenario = load_scenario(
                            demand_level_case=demand_level_case,
                            demand_case=demand_case,
                            gas_deliverability_case=gas_case,
                            solar_case=solar_case,
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
                                "solar": solar_case,
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

        scenario_copy = scenario.copy()
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