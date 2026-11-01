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
from src.gas_benchmarks import BENCHMARKS, MMBTU_PER_TWH_TH
from src.optimize_model import build_model, solve_model, leakage_diagnostic
from src.optimize_model_sliced import build_model_sliced  # A5: run_gas_regime_ndc_matrix() (Group C); run_fin2_blended_bankability_sweep(), run_dem2_growth_gas_matrix() (addendum Part 2); run_reliability_sweep(), run_rel1_feasibility_matrix() (addendum Part 3) -- see each function's own required builder= parameter
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
    MODEL_START_YEAR,
    load_scenario,
    demand_level_scenarios,
    demand_growth_scenarios,
    gas_deliverability_scenarios,
    solar_build_scenarios,
    carbon_policy_scenarios,
    MODEL_END_YEAR,
    REPORT_END_YEAR,
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
            "gas_to_power_twh_th": diagnostics["fleet_gas_to_power_twh_th_by_year"][y],
            "gas_generation_twh_e": diagnostics["fleet_gas_generation_twh_e_by_year"][y],
            "solar_generation_twh_e": diagnostics["solar_generation_twh_e_by_year"][y],
            "storage_discharge_twh_e": diagnostics["storage_discharge_twh_e_by_year"][y],
            "unserved_twh": diagnostics["unserved_twh_by_year"][y],
            "emissions_tco2": diagnostics["emissions_tco2_by_year"][y], 
            "gas_shadow_pv_usd_per_twh_th": diagnostics["gas_shadow_pv_usd_per_twh_th_by_year"][y],
            "gas_shadow_yr_usd_per_twh_th": diagnostics["gas_shadow_yr_usd_per_twh_th_by_year"][y],
            "carbon_shadow_pv_usd_per_tco2": diagnostics["carbon_shadow_pv_usd_per_tco2_by_year"][y],
            "carbon_shadow_yr_usd_per_tco2": diagnostics["carbon_shadow_yr_usd_per_tco2_by_year"][y],
            "discount_factor": diagnostics["discount_factor_by_year"][y],
        })

    return pd.DataFrame(rows)


SHADOW_BINDING_TOL = 1e-6


def shadow_mean_fields(prefix, vals_pv, vals_yr, tol=SHADOW_BINDING_TOL):
    """
    TASK 27 Part A.2 (docs/task_log_27a_gas_shadow_fix.md; register L70),
    renamed and split by TASK 53 2.3 (register L106).

    A cross-year mean of a shadow price taken over EVERY year with a dual
    is diluted by the non-binding years (whose dual is ~0): it equals
    sum_of_binding_duals / N_evaluable, not / N_binding. That is the same
    defect class TASK 02 fixed in extract_reliability_dual by emitting the
    binding-year mean and the all-year mean as SEPARATE named fields.

    TASK 53 2.3: the dual of a constraint inherits the objective's
    present-value units, so every mean is now emitted on BOTH bases and the
    basis is in the name: `<prefix>_shadow_pv_*` (the solver's number,
    year-0 USD) and `<prefix>_shadow_yr_*` (divided by m.DF[t], year-t USD).
    The old basis-less names `<prefix>_shadow_all_year_mean` and
    `<prefix>_shadow_binding_mean` are REMOVED, not kept with a changed
    meaning, so a stale reader fails loudly. Counts are basis-free and keep
    their names.

    prefix: "gas" or "carbon". vals_pv / vals_yr: per-year duals on the two
    bases, same years, None = no dual. Binding = pv dual > tol (the sign and
    zero set are the same on both bases).
    """
    pairs = [(a, b) for a, b in zip(list(vals_pv), list(vals_yr)) if a is not None]
    pv = [a for a, _ in pairs]
    yr = [b for _, b in pairs]
    bpv = [a for a, _ in pairs if a > tol]
    byr = [b for a, b in pairs if a > tol]
    return {
        f"{prefix}_shadow_pv_all_year_mean": float(np.mean(pv)) if pv else 0.0,
        f"{prefix}_shadow_yr_all_year_mean": float(np.mean(yr)) if yr else 0.0,
        f"{prefix}_shadow_pv_binding_mean": float(np.mean(bpv)) if bpv else None,
        f"{prefix}_shadow_yr_binding_mean": float(np.mean(byr)) if byr else None,
        f"{prefix}_shadow_binding_years": len(bpv),
        f"{prefix}_shadow_evaluable_years": len(pv),
    }


def none_shadow_fields(prefix):
    """All-None counterpart of shadow_mean_fields(), for failed-solve rows."""
    return {
        f"{prefix}_shadow_pv_all_year_mean": None,
        f"{prefix}_shadow_yr_all_year_mean": None,
        f"{prefix}_shadow_pv_binding_mean": None,
        f"{prefix}_shadow_yr_binding_mean": None,
        f"{prefix}_shadow_binding_years": None,
        f"{prefix}_shadow_evaluable_years": None,
    }


def passthrough_shadow_fields(prefix, r):
    """Copy the six shadow_mean_fields() keys from row dict r."""
    return {k: r.get(k) for k in none_shadow_fields(prefix)}


def shadow_series(diag, which):
    """(pv_values, yr_values) of the gas or carbon shadow in year order; `which` in {'gas', 'carbon'}."""
    unit = {"gas": "usd_per_twh_th", "carbon": "usd_per_tco2"}[which]
    pv = diag[f"{which}_shadow_pv_{unit}_by_year"]
    yr = diag[f"{which}_shadow_yr_{unit}_by_year"]
    ys = sorted(pv)
    return [pv[y] for y in ys], [yr[y] for y in ys]


def summarize_run(m, diagnostics, scenario):
    years = scenario["years"]

    cumulative_unserved = sum(diagnostics["unserved_twh_by_year"].values())
    cumulative_emissions = float(pyo.value(m.emissions))
    gas_shadow = diagnostics["gas_shadow_pv_usd_per_twh_th_by_year"]
    gas_shadow_yr = diagnostics["gas_shadow_yr_usd_per_twh_th_by_year"]
    positive_shadow = [v for v in gas_shadow.values() if v and v > 1e-6]
    positive_shadow_yr = [gas_shadow_yr[y] for y, v in gas_shadow.items() if v and v > 1e-6]

    return {
        "gas_binding_share": len(positive_shadow) / len(gas_shadow),
        "gas_shadow_pv_avg_usd_per_twh_th":
            float(np.mean(positive_shadow)) if positive_shadow else 0.0,
        "gas_shadow_pv_max_usd_per_twh_th":
            float(max(positive_shadow)) if positive_shadow else 0.0,
        "gas_shadow_yr_avg_usd_per_twh_th":
            float(np.mean(positive_shadow_yr)) if positive_shadow_yr else 0.0,
        "gas_shadow_yr_max_usd_per_twh_th":
            float(max(positive_shadow_yr)) if positive_shadow_yr else 0.0,
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
        "gas_shadow_pv_last_year":
        diagnostics["gas_shadow_pv_usd_per_twh_th_by_year"][int(years[-1])],
        "gas_shadow_yr_last_year":
        diagnostics["gas_shadow_yr_usd_per_twh_th_by_year"][int(years[-1])],
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

    status = solve_model(m, scenario=scenario)
    
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
    """
    LEGACY NON-LP SIMULATOR — NOT for thesis result tables.

    This evaluates a HAND-SUPPLIED capacity path with must-run gas, undiscounted
    costs, and the fixed 2025 solar CAPEX scalar. Its cost basis is therefore
    NOT comparable to build_model()'s discounted NPV. It is retained only for
    quick feasibility sketches. Any headline cost/emissions/reliability result
    must come from build_model()/extract_planning_diagnostics(), not from here.
    The Monte Carlo path no longer uses this function (see stochastic.py, which
    now uses fixed-plan re-dispatch through build_model()).
    """
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

    # task_audit_fixes_p1 (audit finding A8): was `scenario.get("gas_eta",
    # 0.43)` -- 0.43 is not an arbitrary placeholder, it is the retired
    # CCGT-on-an-OCGT-fleet figure this project's sourcing ledger records
    # as wrong and replaced by the measured 0.287
    # (docs/merged_correction_plan_v2.md, sourcing ledger, eta row). A
    # fallback that ever fired would silently overstate fleet efficiency
    # by 50%. Matches optimize_model_sliced.py's own gas_eta guard
    # (:491-493) instead: this function takes a bare scenario dict, and
    # nothing in its signature requires load_scenario() to have built it.
    eta = scenario.get("gas_eta", None)
    if eta is None:
        raise ValueError("gas_eta is unset. Provide the measured fleet efficiency.")
    gas_gen = gas_profile["available_twh_th"] * eta

    solar_gen = solar_generation(
        capacity_mw=capacity_paths["solar_mw"],
        capacity_factor=scenario["solar_cf"],
    )

    # Canonical site: scenarios.py:2666 (_storage_duration_hours = 4.0),
    # surfaced on the scenario dict at scenarios.py:2970. Read from the
    # scenario in scope here instead of a second, independent 4.0 literal
    # that could silently drift from it -- same discipline as the
    # builders' own scenario.get("storage_duration_hours", 4.0).
    storage_duration_hours = scenario.get("storage_duration_hours", 4.0)

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
    Returns the dual(s) on the reliability constraint (USD per TWh of
    selfgen+unserved shortfall allowed), per-year detail, and binding
    flags.

    TASK 02 rewrite (docs/task_log_02_dual_fix.md) -- retires the single
    collapsed `dual_usd_per_twh` field this function used to return, no
    alias kept, because it silently conflated three separate defects:

    1. The indexed branch averaged over EVERY year with a recorded dual,
       not binding years only, despite its own former comment claiming
       "average across binding years." Non-binding years return a dual
       of 0 under LP complementary slackness, which dragged the mean
       toward zero in proportion to how many years were slack. Both an
       all-years mean and a binding-years-only mean are now returned,
       explicitly named, so the two are never conflated again.
    2. The objective (`m.system_cost_npv`) is entirely NPV-denominated
       (every cost term carries `m.DF[t]`), so the dual the solver
       returns is a PRESENT-VALUE marginal, never a same-year one -- and
       this was never declared or convertible. Both bases are now
       returned; the year-basis series divides the present-value dual by
       `m.DF[t]`, READ FROM THE SOLVED MODEL (never reconstructed from
       `r_social`), so the two can never diverge.
    3. The scalar ("total" mode) branch read `_gso.ubound`, which does
       not exist on a Pyomo `ScalarConstraint` at the installed version
       (6.10.1 -- confirmed empirically, docs/task_log_02_dual_fix.md:
       `hasattr(constraint, "ubound")` is `False`; `AttributeError` on
       access). The bare `except Exception:` around it silently
       swallowed that error on every call, so `binding`/
       `primal_slack_twh` have been `None` from the scalar branch on
       every call regardless of the model's actual primal slack. Fixed
       to `.upper`, matching the indexed branch's own (already-correct)
       attribute, with the exception narrowed to `AttributeError` so a
       genuinely new failure is no longer silent.

    The reliability constraint is `m.grid_service_obligation` (deprecated
    alias `m.reliability_constraint`): SCALAR under
    `grid_service_mode="total"` (one constraint for the whole horizon) or
    INDEXED over calendar years under `grid_service_mode="annual"` (one
    constraint per year). `09_run_rel1_feasibility.py` reaches only the
    scalar branch (docs/task_log_01_omnibus_inventory.md item 6);
    REL-2/REL-3 (`run_reliability_sweep`) reach only the indexed branch
    (same report) -- correcting this function's own former comment,
    which called the scalar branch "the standard REL-2 case." It is not,
    at HEAD.

    Parameters
    ----------
    m    : solved Pyomo model
    diag : diagnostics dict from extract_planning_diagnostics(), optional.
           Used ONLY for the two legacy fields `binding`/`primal_slack_
           twh` below -- UNCHANGED semantics from before this rewrite,
           including still being `None` when `diag` is not supplied.
           Every other field is computed directly from the constraint
           object itself (`.upper`, `.body`, `m.dual`, `m.DF`) and does
           NOT need `diag`. `09_`'s own call site passes no `diag` at all
           (`extract_reliability_dual(m)`,
           `src/optimize_experiments.py:3599` at the time of TASK 01/02 --
           verified, not assumed), so for that caller the new per-year/
           per-basis fields below are populated while the two legacy
           fields remain `None`, exactly as `09_`'s current, unfixed
           behaviour already is. `09_`/`run_rel1_feasibility_matrix` are
           NOT edited here -- that is TASK 03's file; this is reported,
           not changed.

    Returns
    -------
    dict with keys:
        dual_pv_usd_per_twh_binding_mean   : float|None -- present-value
            dual averaged over years whose OWN primal slack < 1e-4 TWh.
            None if no binding year has a recorded dual (an empty mean is
            None, never 0.0).
        dual_pv_usd_per_twh_all_mean       : float|None -- present-value
            dual averaged over every year with a recorded dual, binding
            or not. This is the OLD `dual_usd_per_twh`'s exact arithmetic,
            renamed -- see the regression check in
            docs/task_log_02_dual_fix.md.
        dual_yr_usd_per_twh_binding_mean   : float|None -- as above, each
            year's present-value dual divided by that year's `m.DF[t]`
            before averaging.
        dual_yr_usd_per_twh_all_mean       : float|None -- as above.
        dual_pv_usd_per_twh_binding_median : float|None
        dual_pv_usd_per_twh_binding_min    : float|None
        dual_pv_usd_per_twh_binding_max    : float|None
        binding_years_n   : int|None -- count of years with primal slack
            < 1e-4 TWh.
        years_with_dual_n : int|None -- count of years with a recorded
            dual (binding or not).
        dual_by_year_pv   : dict[int, float]|None -- present-value dual
            per year, years with a recorded dual only.
        binding_by_year   : dict[int, bool]|None -- per-year binding flag
            for EVERY year in the constraint's index set, independent of
            whether that year has a recorded dual (binding is a
            primal-side fact, read from `.upper`/`.body` directly, not
            from the dual).
        binding           : bool|None -- UNCHANGED semantics: True iff the
            MINIMUM per-year primal slack (indexed branch) / the single
            slack (scalar branch) is < 1e-4 TWh. Still `None` when `diag`
            is not supplied.
        primal_slack_twh  : float|None -- UNCHANGED semantics: the
            minimum per-year slack (indexed) / the single slack (scalar).
            Still `None` when `diag` is not supplied.

    In the SCALAR branch, every year-basis field
    (`dual_yr_usd_per_twh_binding_mean`, `dual_yr_usd_per_twh_all_mean`)
    and every per-year field (`dual_by_year_pv`, `binding_by_year`,
    `binding_years_n`, `years_with_dual_n`) is `None` -- the horizon-total
    constraint has no single year to divide a discount factor by, and
    none is invented here by dividing by an average discount factor. The
    PV-basis mean/median/min/max fields still apply there, treating the
    one scalar constraint as a single-element set (populated if binding,
    `None` if not or if no dual was recorded).
    """
    _gso = getattr(m, "grid_service_obligation",
                   getattr(m, "reliability_constraint", None))
    _none_result = {
        "dual_pv_usd_per_twh_binding_mean":   None,
        "dual_pv_usd_per_twh_all_mean":       None,
        "dual_yr_usd_per_twh_binding_mean":   None,
        "dual_yr_usd_per_twh_all_mean":       None,
        "dual_pv_usd_per_twh_binding_median": None,
        "dual_pv_usd_per_twh_binding_min":    None,
        "dual_pv_usd_per_twh_binding_max":    None,
        "binding_years_n":                    None,
        "years_with_dual_n":                  None,
        "dual_by_year_pv":                    None,
        "binding_by_year":                    None,
        "binding":                            None,
        "primal_slack_twh":                   None,
    }
    if _gso is None:
        return dict(_none_result)

    if not _gso.is_indexed():
        # Scalar (total mode) -- 09_run_rel1_feasibility.py's own path.
        # No per-year structure exists: every year-basis and per-year
        # field stays None, by this function's own contract (above).
        raw = m.dual.get(_gso, None)
        dual_val = None if raw is None else max(0.0, float(-raw))

        primal_slack = None
        if diag is not None:
            # 2.6, D4: the obligation bounds selfgen + unserved. unserved alone
            # is ~0 by construction under a genset case (D11), so slack must be
            # computed on the SHORTFALL or the constraint always reads slack.
            lhs = (sum(diag["unserved_twh_by_year"].values())
                   + sum(diag["selfgen_twh_by_year"].values()))
            try:
                # TASK 02, Defect 3: was `_gso.ubound`, which does not
                # exist on a Pyomo ScalarConstraint (confirmed empirically
                # at the installed version) -- `.upper` is the correct
                # attribute, matching the indexed branch below. Narrowed
                # from a bare `except Exception:` so an unrelated future
                # failure is no longer silent.
                rhs = float(pyo.value(_gso.upper))
            except AttributeError:
                rhs = None
            if rhs is not None:
                primal_slack = rhs - lhs

        binding = None if primal_slack is None else (primal_slack < 1e-4)

        result = dict(_none_result)
        result["binding"] = binding
        result["primal_slack_twh"] = primal_slack
        if dual_val is not None:
            result["dual_pv_usd_per_twh_all_mean"] = dual_val
            if binding:
                result["dual_pv_usd_per_twh_binding_mean"]   = dual_val
                result["dual_pv_usd_per_twh_binding_median"] = dual_val
                result["dual_pv_usd_per_twh_binding_min"]    = dual_val
                result["dual_pv_usd_per_twh_binding_max"]    = dual_val
        return result

    # Indexed (annual mode).
    dual_by_year_pv = {}
    for t in _gso:
        raw = m.dual.get(_gso[t], None)
        if raw is not None:
            dual_by_year_pv[t] = max(0.0, float(-raw))

    # Per-year primal slack and binding, read directly from each year's
    # own constraint instance (TASK 02, Defect 1) -- does not depend on
    # `diag` at all, unlike the legacy `binding`/`primal_slack_twh` pair
    # below, which is kept diag-gated for unchanged semantics.
    binding_by_year = {}
    slacks_by_year = {}
    for t in _gso:
        c = _gso[t]
        slack_t = float(pyo.value(c.upper)) - float(pyo.value(c.body))
        slacks_by_year[t] = slack_t
        binding_by_year[t] = slack_t < 1e-4

    binding_years_n = sum(1 for v in binding_by_year.values() if v)
    years_with_dual_n = len(dual_by_year_pv)

    all_years_with_dual = list(dual_by_year_pv.keys())
    binding_years_with_dual = [t for t in all_years_with_dual if binding_by_year.get(t, False)]

    def _mean(xs):
        return (sum(xs) / len(xs)) if xs else None

    def _yr_vals(years_subset):
        return [dual_by_year_pv[t] / float(pyo.value(m.DF[t])) for t in years_subset]

    all_vals = [dual_by_year_pv[t] for t in all_years_with_dual]
    binding_vals = [dual_by_year_pv[t] for t in binding_years_with_dual]

    dual_pv_all_mean       = _mean(all_vals)
    dual_pv_binding_mean   = _mean(binding_vals)
    dual_pv_binding_median = float(np.median(binding_vals)) if binding_vals else None
    dual_pv_binding_min    = min(binding_vals) if binding_vals else None
    dual_pv_binding_max    = max(binding_vals) if binding_vals else None

    dual_yr_all_mean     = _mean(_yr_vals(all_years_with_dual))
    dual_yr_binding_mean = _mean(_yr_vals(binding_years_with_dual))

    # Legacy fields, UNCHANGED semantics: min slack across years / binding
    # iff at least one year is pinned. Still gated on `diag` exactly as
    # before this rewrite -- no current indexed-branch caller omits
    # `diag` (REL-2/REL-3 always pass it), so this gate is inert in
    # practice, but is preserved rather than removed, per this task's
    # scope.
    primal_slack = None
    binding = None
    if diag is not None:
        try:
            primal_slack = min(slacks_by_year.values()) if slacks_by_year else None
            binding = (primal_slack < 1e-4) if primal_slack is not None else None
        except Exception:
            pass

    return {
        "dual_pv_usd_per_twh_binding_mean":   dual_pv_binding_mean,
        "dual_pv_usd_per_twh_all_mean":       dual_pv_all_mean,
        "dual_yr_usd_per_twh_binding_mean":   dual_yr_binding_mean,
        "dual_yr_usd_per_twh_all_mean":       dual_yr_all_mean,
        "dual_pv_usd_per_twh_binding_median": dual_pv_binding_median,
        "dual_pv_usd_per_twh_binding_min":    dual_pv_binding_min,
        "dual_pv_usd_per_twh_binding_max":    dual_pv_binding_max,
        "binding_years_n":                    binding_years_n,
        "years_with_dual_n":                  years_with_dual_n,
        "dual_by_year_pv":                    dual_by_year_pv,
        "binding_by_year":                    binding_by_year,
        "binding":                            binding,
        "primal_slack_twh":                   primal_slack,
    }


# TASK 03 (docs/task_log_03_rel1_mode.md), Section C: the flat-numeric
# subset of extract_reliability_dual()'s own return contract, shared by
# run_rel1_feasibility_matrix() and compute_feasibility_threshold() so
# the two cannot drift apart on which keys are unwrapped. Excludes
# dual_by_year_pv/binding_by_year -- dict-valued, not flat numeric, same
# reason TASK 02 kept them out of run_reliability_sweep's own flat row.
DUAL_FLAT_KEYS = [
    "dual_pv_usd_per_twh_binding_mean",
    "dual_pv_usd_per_twh_all_mean",
    "dual_yr_usd_per_twh_binding_mean",
    "dual_yr_usd_per_twh_all_mean",
    "dual_pv_usd_per_twh_binding_median",
    "dual_pv_usd_per_twh_binding_min",
    "dual_pv_usd_per_twh_binding_max",
    "binding_years_n",
    "years_with_dual_n",
    "binding",
    "primal_slack_twh",
]


def bankability_slack_per_mw_by_year(m, scenario: dict, years: list) -> dict:
    """
    Per-year slack in m.eaas_bankability, in USD per MW of EaaS solar
    (docs/tariff_arm.md, Step 3).

    task_defect_closure, D17: reused verbatim from
    `bankability_slack_per_mw_by_year` in `scripts/03_run_ndc_eaas.py`
    (TASK.md's own instruction: "read that implementation and reuse its
    approach rather than inventing one") rather than sharing that
    function directly, because `03_run_ndc_eaas` is not a valid Python
    module name (identifiers cannot start with a digit) -- no runner in
    this project imports another runner, for the same reason. Duplicated
    here, in `src/`, which is where every OTHER shared diagnostic in this
    file already lives; `03_`'s own copy is untouched (out of this
    task's whitelist). See the report's closing list for the follow-on
    this leaves: point `03_`'s copy at this one.

        slack_per_mw[t] = tariff * solar_energy_per_mw * remaining_npv_factor[t]
                          / required_margin
                          - (capex[t] - eaas_subsidy[t] / solar_eaas_add[t])

    Positive slack ⇒ the tariff more than covers the effective private
    cost at year t: the constraint does not bind and EaaS deployment is
    not tariff-limited there. ~Zero ⇒ it binds.

    Guard: when solar_eaas_add[t] == 0 the constraint is 0 <= 0 and
    carries no information — emit None, not zero. (Tolerance 1e-6 MW
    absorbs solver noise around a true zero.)
    """
    tariff = scenario["solar_service_tariff_usd_per_twh"]
    solar_energy_per_mw = scenario["solar_cf"] * 8760 / 1_000_000
    required_margin = scenario["required_margin"]

    out = {}
    for t, y in enumerate(years):
        eaas_add = float(pyo.value(m.solar_eaas_add[t]))
        if eaas_add <= 1e-6:
            out[int(y)] = None
            continue
        rnf = float(pyo.value(m.remaining_npv_factor[t]))
        capex = float(pyo.value(m.solar_capex_param[t]))
        subsidy = float(pyo.value(m.eaas_subsidy[t]))
        bankable_revenue_per_mw = tariff * solar_energy_per_mw * rnf / required_margin
        effective_private_cost_per_mw = capex - subsidy / eaas_add
        out[int(y)] = bankable_revenue_per_mw - effective_private_cost_per_mw
    return out


def eaas_bankability_binding_years(m, scenario: dict, years: list) -> dict:
    """
    task_defect_closure, D17: count of years m.eaas_bankability actually
    binds, built on bankability_slack_per_mw_by_year() above.

    "Binds" reuses the threshold already established for this exact
    quantity in docs/findings_provisional_pre_sweep.md's F-29 ("binding
    years (slack < 1 USD/MW)") — not a new tolerance invented here, so
    this diagnostic reports the SAME thing that finding already named,
    now measurable per run rather than asserted from one historical
    sweep (which is what D17 asks for: whether the EaaS arm's capital
    constraint binds at all is "now measurable per run, not assumed").

    n_evaluable_years excludes years where solar_eaas_add ~ 0 (no EaaS
    build that year carries no bankability information — see the guard
    in bankability_slack_per_mw_by_year). Report n_binding_years OUT OF
    n_evaluable_years, not out of len(years) — a scenario with little
    EaaS build has few evaluable years, and quoting a fixed 30-year
    denominator regardless would overstate the sample the count is
    drawn from.
    """
    slack = bankability_slack_per_mw_by_year(m, scenario, years)
    evaluable = [v for v in slack.values() if v is not None]
    binding = [v for v in evaluable if v < 1.0]
    return {
        "n_binding_years":   len(binding),
        "n_evaluable_years": len(evaluable),
    }


def run_reliability_sweep(
    scenario,
    econ,
    builder,
    gas_case=None,
    financing_label=None,
    policy_label=None,
    emissions_cap_by_year=None,
    eps_levels=None,
):
    """
    Runs:
    - Baseline (VoLL only, no reliability constraint)
    - Constrained reliability cases across eps_levels

    builder : {"annual", "sliced"} -- REQUIRED, NO DEFAULT (A5 addendum,
        Part 3). Selects build_model (annual) or build_model_sliced, same
        pattern as run_gas_regime_ndc_matrix's builder= (Group C). Shared
        by REL-2 (run_rel2_marginal_cost_curves) and REL-3
        (run_rel3_financing_frontier), both of which must now accept and
        thread their own builder= parameter through to this call. No
        default is given deliberately, for the same reason Group C gave
        none. Every existing call site is keyword-only (confirmed before
        this change), so this breaks no positional call.

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
    eps_levels            : list[float] or None — task_defect_closure, D23: this
                           parameter previously did not exist on this function at
                           all, so every caller's own eps_levels (itself correctly
                           resolved from a caller parameter one level up) was
                           silently discarded here in favour of
                           reliability_levels_log(), unconditionally. If None,
                           defaults to reliability_levels_log(), same as before;
                           if supplied, used as-is.

    Returns a list of result dicts, one per eps level (plus voll_only baseline).
    """
    _BUILDER_FNS = {"annual": build_model, "sliced": build_model_sliced}
    if builder not in _BUILDER_FNS:
        raise ValueError(
            f"run_reliability_sweep: builder must be one of "
            f"{sorted(_BUILDER_FNS)}, got {builder!r}."
        )
    _build_fn = _BUILDER_FNS[builder]

    if eps_levels is None:
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
        end_year=MODEL_END_YEAR,
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
            # Plan 0.4, item C: 1e18 sentinel -> None ("no cap", already
            # the builder's own guarded meaning for this argument).
            kwargs["emissions_cap"] = None

        if eps is not None:
            # 2.6, D6: 'annual' not 'total'. Under a monotonically rising
            # deficit, a horizon-total bound is satisfied by early-years slack
            # and stops discriminating between service standards -- 09b showed
            # shortfall identical at eps 0.20/0.15/0.10/0.08 in total mode.
            kwargs["max_offgrid_fraction"] = eps
            kwargs["grid_service_mode"] = "annual"

        kwargs["solar_capex_by_year"] = solar_capex_tv
        m = _build_fn(**kwargs)
        status = solve_model(m, scenario=scenario)
        if not status["optimal"]:
            raise RuntimeError("non-optimal termination")
        return m

    def _make_row(eps, m, diag, status_str):
        """Build a result row from solved model and diagnostics."""
        unserved = sum(diag["unserved_twh_by_year"].values())
        # 2.6, D4: the obligation bounds selfgen + unserved. unserved alone is
        # ~0 by construction under a genset case (D11), so slack and binding
        # must be computed on the SHORTFALL or the constraint reads as
        # permanently slack. TASK 53 2.6: this definition is now the canonical one
        # (diag["grid_shortfall_total_twh"], computed once in extract_planning_diagnostics).
        selfgen = sum(diag["selfgen_twh_by_year"].values())
        shortfall = diag["grid_shortfall_total_twh"]
        assert abs(shortfall - (unserved + selfgen)) <= 1e-9 * max(1.0, abs(shortfall))
        dual_info = extract_reliability_dual(m, diag=diag)

        def _norm(x):
            # TASK 02, Section B: normalisation constant and source line
            # unchanged (total_demand_twh / 100.0) -- mirrored across
            # every pv/yr x binding/all variant instead of the single
            # collapsed dual_usd_per_pct_pt this replaces.
            return x * total_demand_twh / 100.0 if x is not None else None

        # TASK 15, Part A.2 (docs/task_log_15a_binding_criterion.md):
        # RENAMED from "binding" -- this is a HORIZON-AGGREGATE comparison
        # (eps * total_demand_twh, a 30-year sum, against shortfall_total,
        # also a 30-year sum), not a per-year test, even though the
        # constraint this project actually imposes (grid_service_mode=
        # "annual") is per-year. One binding year in thirty is invisible
        # to a thirty-year sum -- TASK 13 found exactly this at
        # downside/eps=0.20 (binding_years_n=1, this field False). The
        # computation is UNCHANGED; only the name is now honest about what
        # it measures. Superseded for onset detection by binding_any_year
        # below, sourced from extract_reliability_dual's own per-year
        # result (dual_info, already in scope two lines above -- no
        # signature change needed anywhere).
        rhs_twh = eps * total_demand_twh if eps is not None else None
        primal_slack = rhs_twh - shortfall if rhs_twh is not None else None
        binding_horizon_aggregate = primal_slack < 1e-4 if primal_slack is not None else None
        binding_any_year = (
            dual_info["binding_years_n"] >= 1
            if dual_info["binding_years_n"] is not None else None
        )

        gs_pv, gs_yr = shadow_series(diag, "gas")

        # task_defect_closure, D17: only meaningful for the EaaS financing
        # arm -- m.eaas_bankability is structurally absent (not merely
        # zero) whenever financing_regime != "eaas" (optimize_model.py:
        # 689-764), so this must not be read unconditionally the way the
        # rest of this row is. None for every non-EaaS row (REL-2's
        # public-only-only sweeps, and REL-3's own public_only arm).
        if financing_label == "eaas" and hasattr(m, "eaas_bankability"):
            _bank = eaas_bankability_binding_years(m, scenario, scenario["years"])
            eaas_bankability_n_binding = _bank["n_binding_years"]
            eaas_bankability_n_evaluable = _bank["n_evaluable_years"]
        else:
            eaas_bankability_n_binding = None
            eaas_bankability_n_evaluable = None

        return {
            "gas_case":            gas_case,
            "financing_label":     financing_label,
            "policy_label":        policy_label,
            "eps":                 eps,
            "eps_label":           "voll_only" if eps is None else f"eps_{eps:.2f}",
            "npv_cost":            float(pyo.value(m.system_cost_npv)),
            "unserved_total":      unserved,
            # 2.6, D4: unserved alone is ~0 by construction under a genset case
            # (D11). The obligation bounds selfgen + unserved, so a frontier
            # built on unserved plots a constraint it cannot see.
            "selfgen_total":       selfgen,
            "shortfall_total":     shortfall,
            "unserved_only_reliability": diag["unserved_only_reliability"],
            "grid_service_reliability_horizon": diag["grid_service_reliability_horizon"],
            # TASK 02 (docs/task_log_02_dual_fix.md): dual_usd_per_twh and
            # dual_usd_per_pct_pt are RETIRED, no alias kept -- a stale
            # consumer reading either name now gets a KeyError, not a
            # silently-still-diluted number.
            "dual_pv_usd_per_twh_binding_mean":   dual_info["dual_pv_usd_per_twh_binding_mean"],
            "dual_pv_usd_per_twh_all_mean":       dual_info["dual_pv_usd_per_twh_all_mean"],
            "dual_pv_usd_per_twh_binding_median": dual_info["dual_pv_usd_per_twh_binding_median"],
            "dual_pv_usd_per_twh_binding_min":    dual_info["dual_pv_usd_per_twh_binding_min"],
            "dual_pv_usd_per_twh_binding_max":    dual_info["dual_pv_usd_per_twh_binding_max"],
            "dual_yr_usd_per_twh_binding_mean":   dual_info["dual_yr_usd_per_twh_binding_mean"],
            "dual_yr_usd_per_twh_all_mean":       dual_info["dual_yr_usd_per_twh_all_mean"],
            "binding_years_n":                    dual_info["binding_years_n"],
            "years_with_dual_n":                  dual_info["years_with_dual_n"],
            "dual_pv_usd_per_pct_pt_binding_mean": _norm(dual_info["dual_pv_usd_per_twh_binding_mean"]),
            "dual_pv_usd_per_pct_pt_all_mean":     _norm(dual_info["dual_pv_usd_per_twh_all_mean"]),
            "dual_yr_usd_per_pct_pt_binding_mean": _norm(dual_info["dual_yr_usd_per_twh_binding_mean"]),
            "dual_yr_usd_per_pct_pt_all_mean":     _norm(dual_info["dual_yr_usd_per_twh_all_mean"]),
            # TASK 15, Part A.2: binding_years_n was already present here
            # (unchanged below) -- only binding_any_year and the rename are
            # new.
            "binding_horizon_aggregate": binding_horizon_aggregate,
            "binding_any_year":    binding_any_year,
            "primal_slack_twh":    primal_slack,
            "total_demand_twh":    total_demand_twh,
            **shadow_mean_fields("gas", gs_pv, gs_yr),
            "solar_eaas_total_mw": sum(
                float(pyo.value(m.solar_eaas_add[t]))
                for t in range(len(scenario["years"]))
            ),
            "solar_public_total_mw": sum(
                float(pyo.value(m.solar_public_add[t]))
                for t in range(len(scenario["years"]))
            ),
            # task_defect_closure, D17: "the number of years the EaaS
            # capital/bankability constraint actually binds" -- None on
            # every non-EaaS row (see guard above).
            "eaas_bankability_n_binding_years":   eaas_bankability_n_binding,
            "eaas_bankability_n_evaluable_years": eaas_bankability_n_evaluable,
            "status":              status_str,
            **leakage_flat(diag),
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
                "selfgen_total": None, "shortfall_total": None,
                "unserved_only_reliability": None,
                "grid_service_reliability_horizon": None,
                "dual_pv_usd_per_twh_binding_mean": None,
                "dual_pv_usd_per_twh_all_mean": None,
                "dual_pv_usd_per_twh_binding_median": None,
                "dual_pv_usd_per_twh_binding_min": None,
                "dual_pv_usd_per_twh_binding_max": None,
                "dual_yr_usd_per_twh_binding_mean": None,
                "dual_yr_usd_per_twh_all_mean": None,
                "binding_years_n": None, "years_with_dual_n": None,
                "dual_pv_usd_per_pct_pt_binding_mean": None,
                "dual_pv_usd_per_pct_pt_all_mean": None,
                "dual_yr_usd_per_pct_pt_binding_mean": None,
                "dual_yr_usd_per_pct_pt_all_mean": None,
                "binding_horizon_aggregate": None,
                "binding_any_year": None,
                "primal_slack_twh": None, **none_shadow_fields("gas"),
                "solar_eaas_total_mw": None, "solar_public_total_mw": None,
            })

    return results


def identify_bottleneck(diag):

    gas_shadow = diag["gas_shadow_pv_usd_per_twh_th_by_year"]    # binding detection: basis-free
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
            demand_case="organic_central",
            gas_deliverability_case=g,
            carbon_case="no_policy",
            start_year=int(scenario["years"][0]),
            end_year=int(scenario["years"][-1]),
            om_case="om_central",
        )

        m = build_model(
            scenario=scenario_mod,
            econ=econ,
            max_offgrid_fraction=0.10,
            grid_service_mode="annual"
        )

        try:
            solve_model(m, scenario=scenario_mod)
            diag = extract_planning_diagnostics(m, scenario_mod, econ)

            results.append({
                "gas_case": g,
                "grid_service_reliability_horizon": diag["grid_service_reliability_horizon"],
                "unserved": sum(diag["unserved_twh_by_year"].values())
            })

        except:
            results.append({
                "gas_case": g,
                "grid_service_reliability_horizon": None,
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
            # 2.6, D4: unserved alone is ~0 by construction under a genset case
            # (D11). The obligation bounds selfgen + unserved, so the frontier
            # must carry the shortfall or it plots a constraint it cannot see.
            "selfgen_total":       r.get("selfgen_total"),
            "shortfall_total":     r.get("shortfall_total"),
            "unserved_only_reliability": r.get("unserved_only_reliability"),
            "grid_service_reliability_horizon": r.get("grid_service_reliability_horizon"),
            # REL-2 core columns. TASK 02: dual_usd_per_twh/dual_usd_per_
            # pct_pt are retired (docs/task_log_02_dual_fix.md) -- no
            # alias kept, so a stale reader of either old name gets
            # KeyError via .get() returning None, not a silently-still-
            # diluted number. The old `r.get("dual_reliability")` fallback
            # (for 09_'s own nested-dict format) is dropped: 09_ never
            # calls this function (confirmed, docs/task_log_01_omnibus_
            # inventory.md item 5) and is out of scope for this task
            # (TASK 03's file).
            "dual_pv_usd_per_twh_binding_mean":   r.get("dual_pv_usd_per_twh_binding_mean"),
            "dual_pv_usd_per_twh_all_mean":       r.get("dual_pv_usd_per_twh_all_mean"),
            "dual_pv_usd_per_twh_binding_median": r.get("dual_pv_usd_per_twh_binding_median"),
            "dual_pv_usd_per_twh_binding_min":    r.get("dual_pv_usd_per_twh_binding_min"),
            "dual_pv_usd_per_twh_binding_max":    r.get("dual_pv_usd_per_twh_binding_max"),
            "dual_yr_usd_per_twh_binding_mean":   r.get("dual_yr_usd_per_twh_binding_mean"),
            "dual_yr_usd_per_twh_all_mean":       r.get("dual_yr_usd_per_twh_all_mean"),
            "binding_years_n":                    r.get("binding_years_n"),
            "years_with_dual_n":                  r.get("years_with_dual_n"),
            "dual_pv_usd_per_pct_pt_binding_mean": r.get("dual_pv_usd_per_pct_pt_binding_mean"),
            "dual_pv_usd_per_pct_pt_all_mean":     r.get("dual_pv_usd_per_pct_pt_all_mean"),
            "dual_yr_usd_per_pct_pt_binding_mean": r.get("dual_yr_usd_per_pct_pt_binding_mean"),
            "dual_yr_usd_per_pct_pt_all_mean":     r.get("dual_yr_usd_per_pct_pt_all_mean"),
            # TASK 15, Part A.2: RENAMED from "binding" -- no alias kept
            # (same discipline as TASK 02's retirements above: a stale
            # reader of the old name now gets None via .get(), not a
            # silently-still-horizon-diluted value). This whitelist is the
            # THIRD pass-through site for this field, found only while
            # making this edit -- _make_row's own dict and
            # compute_curve_summary's/main()'s two filters in
            # 10_run_rel2_marginal_cost.py were the other two; without
            # this one, the rename would have broken the field (every
            # row's "binding" reading None forever) rather than renaming
            # it, since this function is what actually builds the
            # DataFrame both 10_ and 11_ consume.
            "binding_horizon_aggregate": r.get("binding_horizon_aggregate"),
            "binding_any_year":    r.get("binding_any_year"),
            "primal_slack_twh":    r.get("primal_slack_twh"),
            "total_demand_twh":    r.get("total_demand_twh"),
            **passthrough_shadow_fields("gas", r),
            # task_defect_closure, D17: threaded through from _make_row's
            # row dict -- None on every non-EaaS row (see the guard there).
            "eaas_bankability_n_binding_years":
                r.get("eaas_bankability_n_binding_years"),
            "eaas_bankability_n_evaluable_years":
                r.get("eaas_bankability_n_evaluable_years"),
            "status":              r.get("status", "feasible"),
        })
    return pd.DataFrame(rows)


def run_rel3_financing_frontier(
    econ,
    gas_cases,
    policy_configs,
    peak_multiple_case,
    reserve_margin,
    storage_cc_case,
    gas_build_case,
    gas_availability_case,
    gas_retirement_case,
    gas_capex_case,
    ccgt_efficiency_case,
    capital_recovery_case,
    dscr_case,
    us_inflation,
    capital_treatment_case,
    concessional_envelope_case,
    conversion_case,
    hydro_case,
    om_case,
    builder,
    eps_levels=None,
    cap_path="data/cost/processed/emissions_cap.csv",
    solar_build_case=None,
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
        (no margin key: required_margin is dscr_case-derived for BOTH arms --
         TASK 29 removed the former "eaas_margin" literal; register L34)
        "tariff_case"    : str — key of tariff_scenarios(), fed to
                           load_scenario(tariff_case=...) (docs/tariff_arm.md)
    peak_multiple_case, reserve_margin, storage_cc_case, gas_build_case,
    gas_availability_case, gas_retirement_case, gas_capex_case,
    ccgt_efficiency_case, capital_recovery_case, dscr_case, us_inflation,
    capital_treatment_case, concessional_envelope_case, conversion_case,
    hydro_case, om_case
                  : REQUIRED, no default -- threaded straight to both
                    load_scenario() calls below. Previously omitted
                    (docs/task_shared_fn_repair.md, Step One); the
                    caller must now supply the same value it would pass
                    to load_scenario() directly. om_case in particular
                    must be stated explicitly -- it is no longer
                    hardcoded to "om_central".
    builder       : {"annual", "sliced"} -- REQUIRED, NO DEFAULT (A5
                    addendum, Part 3). Threaded straight to
                    run_reliability_sweep()'s own required builder=
                    parameter at both call sites below. No default is
                    given deliberately, for the same reason Group C gave
                    none.
    eps_levels    : list[float] or None
    cap_path      : str

    Returns
    -------
    list[dict] — all rows from all combinations, tagged by gas_case,
                 financing_label, policy_label
    """
    # TASK 46: opt-in solar build arm; None = pass nothing, so load_scenario's own
    # default (nirp_trajectory) applies exactly as before.
    _solar_kw = {} if solar_build_case is None else {"solar_build_case": solar_build_case}
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
        # docs/tariff_arm.md, Steps 1-3: config dict now carries a named
        # tariff_case, threaded into load_scenario() below, not a bare
        # USD/TWh number applied post-call.
        tariff_case   = policy_cfg["tariff_case"]

        for gas_case in gas_cases:

            # Load annual NDC caps if policy requires them
            caps = None
            if ndc_scenario is not None:
                years_list = list(range(MODEL_START_YEAR, MODEL_END_YEAR + 1))
                cap_sub = cap_df[
                    (cap_df["scenario"] == ndc_scenario)
                    & (cap_df["year"].isin(years_list))
                ].sort_values("year")
                if len(cap_sub) != len(years_list):
                    raise ValueError(
                        f"Cap length {len(cap_sub)} != {len(years_list)} "
                        f"for scenario='{ndc_scenario}'"
                    )
                _s = cap_sub.set_index("year")["cap_tco2"].astype(float)
                caps = [float(_s[int(y)]) for y in years_list]

            # ── Arm A: Public-only (traditional financing) ─────────────────
            scenario_pub = load_scenario(
                **_solar_kw,
                demand_level_case="served",
                demand_case="organic_central",
                gas_deliverability_case=gas_case,
                capital_case=pub_capital,
                carbon_case="no_policy",
                start_year=2025,
                end_year=MODEL_END_YEAR,
                tariff_case=tariff_case,
                peak_multiple_case=peak_multiple_case,
                reserve_margin=reserve_margin,
                storage_cc_case=storage_cc_case,
                gas_build_case=gas_build_case,
                gas_availability_case=gas_availability_case,
                gas_retirement_case=gas_retirement_case,
                gas_capex_case=gas_capex_case,
                ccgt_efficiency_case=ccgt_efficiency_case,
                capital_recovery_case=capital_recovery_case,
                dscr_case=dscr_case,
                us_inflation=us_inflation,
                capital_treatment_case=capital_treatment_case,
                concessional_envelope_case=concessional_envelope_case,
                conversion_case=conversion_case,
                hydro_case=hydro_case,
                om_case=om_case,
            )
            # financing_regime defaults to "traditional" from load_scenario

            rows_pub = run_reliability_sweep(
                scenario=scenario_pub,
                econ=econ,
                builder=builder,  # A5 addendum, Part 3: threaded from this function's own required builder=
                gas_case=gas_case,
                financing_label="public_only",
                policy_label=policy_label,
                emissions_cap_by_year=caps,
                # task_defect_closure, D23: previously dropped -- this
                # function's own eps_levels (resolved above) never
                # reached run_reliability_sweep, which has its own
                # eps_levels parameter now.
                eps_levels=eps_levels,
            )
            all_rows.extend(rows_pub)

            # ── Arm B: EaaS ────────────────────────────────────────────────
            scenario_eaas = load_scenario(
                **_solar_kw,
                demand_level_case="served",
                demand_case="organic_central",
                gas_deliverability_case=gas_case,
                capital_case=eaas_capital,
                carbon_case="no_policy",
                start_year=2025,
                end_year=MODEL_END_YEAR,
                tariff_case=tariff_case,
                peak_multiple_case=peak_multiple_case,
                reserve_margin=reserve_margin,
                storage_cc_case=storage_cc_case,
                gas_build_case=gas_build_case,
                gas_availability_case=gas_availability_case,
                gas_retirement_case=gas_retirement_case,
                gas_capex_case=gas_capex_case,
                ccgt_efficiency_case=ccgt_efficiency_case,
                capital_recovery_case=capital_recovery_case,
                dscr_case=dscr_case,
                us_inflation=us_inflation,
                capital_treatment_case=capital_treatment_case,
                concessional_envelope_case=concessional_envelope_case,
                conversion_case=conversion_case,
                hydro_case=hydro_case,
                om_case=om_case,
            )
            scenario_eaas["financing_regime"] = "eaas"
            # TASK 29 (TASK 28 A.2): required_margin is NOT set here. It comes from
            # load_scenario()'s dscr_case on the same path every arm uses
            # (dscr_scenarios(): a SINGLE cushion, constant across financing
            # regimes; the pre-F1 1.05/1.10 split is RETIRED -- register L34/L43).
            # tariff now set by load_scenario(tariff_case=...) above.

            rows_eaas = run_reliability_sweep(
                scenario=scenario_eaas,
                econ=econ,
                builder=builder,  # A5 addendum, Part 3: threaded from this function's own required builder=
                gas_case=gas_case,
                financing_label="eaas",
                policy_label=policy_label,
                emissions_cap_by_year=caps,
                # task_defect_closure, D23: see the "public_only" call above.
                eps_levels=eps_levels,
            )
            all_rows.extend(rows_eaas)

    return all_rows


def run_fin2_blended_bankability_sweep(
    base_scenario,
    econ,
    tariff_grid,
    blend_scenarios,           # dict from blended_finance_scenarios()
    ndc_cap_scenario,
    builder,
    concessional_envelope_npv=None,   # None => Level 1 (rate only, no scarcity)
    cap_path="data/cost/processed/emissions_cap.csv",
):
    """
    FIN-2 (blended): bankability as a function of BOTH tariff and capital
    structure (concessional share -> blended WACC).

    For each (blend_scenario, tariff) cell, derives the blended private rate,
    solves the NDC-constrained EaaS LP, and records self-financing status,
    realised subsidy, EaaS deployment, and — if a concessional envelope is
    supplied (Level 2) — the concessional shadow price.

    builder : {"annual", "sliced"} -- REQUIRED, NO DEFAULT (A5 addendum,
        Part 2). Selects build_model (annual) or build_model_sliced, same
        pattern as run_gas_regime_ndc_matrix's builder= (Group C). Previously
        hardwired to the annual builder with no way for its sole caller
        (04_run_fin2_tariff_sweep.py) to state otherwise. No default is
        given deliberately, for the same reason Group C gave none: a silent
        default would let a caller's builder choice go unstated. The
        function's one call site is keyword-only (confirmed before this
        change), so this breaks no positional call.

    Returns list[dict], one row per (blend, tariff) cell.
    """
    import copy as _copy
    from src.blended_finance import blended_wacc

    _BUILDER_FNS = {"annual": build_model, "sliced": build_model_sliced}
    if builder not in _BUILDER_FNS:
        raise ValueError(
            f"run_fin2_blended_bankability_sweep: builder must be one of "
            f"{sorted(_BUILDER_FNS)}, got {builder!r}."
        )
    _build_fn = _BUILDER_FNS[builder]

    years = [int(y) for y in base_scenario["years"]]

    # Caps
    if ndc_cap_scenario is None:
        caps = None
    else:
        cap_df = pd.read_csv(cap_path)
        cap_df = cap_df[(cap_df["scenario"] == ndc_cap_scenario)
                        & (cap_df["year"].isin(years))].sort_values("year")
        if len(cap_df) != len(years):
            raise ValueError(
                f"Cap length {len(cap_df)} != model years {len(years)} "
                f"for '{ndc_cap_scenario}'. Run 00_build_emissions_cap.py first."
            )
        _s = cap_df.set_index("year")["cap_tco2"].astype(float)
        caps = [float(_s[int(y)]) for y in years]

    solar_capex_tv = load_solar_capex_by_year(
        scenario_name="solar_low",
        start_year=years[0], end_year=years[-1],
    )
    solar_cf = base_scenario["solar_cf"]
    solar_energy_per_mw = solar_cf * 8760 / 1_000_000
    required_margin = base_scenario["required_margin"]
    # Use the 2025 time-varying CAPEX value for the analytical threshold, to
    # match what the LP prices in the first build year (fixes stale-scalar bug).
    capex_2025 = float(solar_capex_tv[years[0]])

    results = []

    for blend_label, (share, r_conc, r_comm) in blend_scenarios.items():

        r_private = blended_wacc(share, r_conc, r_comm)

        # Analytical threshold at THIS blended rate
        npv_energy = sum(
            solar_energy_per_mw / (1.0 + r_private) ** t
            for t in range(len(years))
        )
        threshold_usd_per_twh = capex_2025 * required_margin / (npv_energy)
        # (threshold in USD/TWh: capex[USD/MW]*margin / (npv_energy[TWh/MW]))

        for tariff in tariff_grid:
            sc = _copy.deepcopy(base_scenario)
            sc["financing_regime"] = "eaas"
            sc["solar_service_tariff_usd_per_twh"] = tariff
            # solar_min_build_default() is [UNSOURCED] (see src/scenarios.py) --
            # forcing it here would contaminate the tariff signal this sweep is
            # meant to isolate, so the floor is pinned to 0.0 for this runner only.
            sc["solar_min_build_mw_per_year"] = 0.0
            # Inject the blended-finance structure so build_model derives r_private
            sc["concessional_share"] = share
            sc["r_concessional"] = r_conc
            sc["r_commercial"] = r_comm
            # Rate-coherence task, Part 1: PIN to the legacy basis, explicitly,
            # rather than let the LP pick up load_scenario()'s new
            # 'bank_tenor_comm' default. This sweep's own analytical threshold
            # (npv_energy / threshold_usd_per_twh above, and max_bankable
            # below) is computed from blended_wacc(share, r_conc, r_comm) --
            # the SAME blend this pins the LP to via resolve_private_rate().
            # Under 'bank_tenor_comm' the LP would discount bankability at the
            # objective's commercial tenor while these analytics kept using
            # the blended WACC: the two would diverge and the "is_self_financing"
            # / "financing_gap_per_mw" figures below would no longer describe
            # what the LP actually solved. FIN-2's analytics are NOT rebuilt in
            # this task (Zeus, rate-coherence Part 1 amendment 2) -- pinning
            # here is a declared scope condition, not an oversight. See
            # docs/task_log_rate_coherence.md.
            sc["bank_tenor_case"] = "bank_tenor_legacy"
            if concessional_envelope_npv is not None:
                sc["concessional_envelope_npv"] = concessional_envelope_npv

            max_bankable = tariff * npv_energy / required_margin
            financing_gap_per_mw = max(0.0, capex_2025 - max_bankable)
            is_self_financing = financing_gap_per_mw < 1.0

            build_kwargs = dict(scenario=sc, econ=econ,
                                solar_capex_by_year=solar_capex_tv)
            if caps is not None:
                build_kwargs["emissions_cap_by_year"] = caps
            else:
                # Plan 0.4, item C: 1e18 sentinel -> None ("no cap",
                # already the builder's own guarded meaning here).
                build_kwargs["emissions_cap"] = None
            m = _build_fn(**build_kwargs)
            status = solve_model(m, scenario=sc)

            row = {
                "blend_scenario": blend_label,
                "concessional_share": share,
                "r_concessional": r_conc,
                "r_commercial": r_comm,
                "blended_wacc": r_private,
                "tariff_usd_per_twh": tariff,
                "tariff_m_usd_per_twh": tariff / 1e6,
                "threshold_usd_per_twh": threshold_usd_per_twh,
                "threshold_m_usd_per_twh": threshold_usd_per_twh / 1e6,
                "financing_gap_per_mw": financing_gap_per_mw,
                "is_self_financing": is_self_financing,
            }

            if not status["optimal"]:
                row["status"] = "infeasible"
                row["zero_subsidy_reason"] = "n/a"
                results.append(row)
                continue

            diag = extract_planning_diagnostics(m, sc, econ,
                                                solar_capex_series=solar_capex_tv)
            eaas_subsidy_npv = sum(
                float(pyo.value(m.DF[t])) * float(pyo.value(m.eaas_subsidy[t]))
                for t in range(len(years))
            )
            solar_eaas_total_mw = sum(
                float(pyo.value(m.solar_eaas_add[t])) for t in range(len(years))
            )

            # Why did eaas_subsidy_npv land at (near) zero, if it did?
            # eaas_subsidy[t] is pinned to gap_at_t * solar_eaas_add[t] by the
            # eaas_bankability_rule / subsidy_limit_rule constraint pair in
            # src/optimize_model.py (~line 490): gap_at_t = max(0,
            # solar_capex_param[t] - tariff*solar_energy_per_mw*
            # remaining_npv_factor[t]/required_margin). A zero aggregate
            # subsidy means EITHER the per-year gap closed (tariff covers
            # CAPEX in every year solar was built) OR nothing was built at
            # all. Reconstructed from the solved model's own Params here,
            # NOT from financing_gap_per_mw above, which is anchored to
            # capex_2025 only and misrepresents later build years under the
            # time-varying CAPEX trajectory.
            SUBSIDY_EPS_USD = 1000.0
            MW_EPS = 1e-3
            GAP_EPS_USD_PER_MW = 1.0
            if eaas_subsidy_npv >= SUBSIDY_EPS_USD:
                zero_subsidy_reason = "n/a"
            elif solar_eaas_total_mw < MW_EPS:
                zero_subsidy_reason = "zero_deployment"
            else:
                built_with_open_gap = any(
                    float(pyo.value(m.solar_eaas_add[t])) > MW_EPS
                    and max(
                        0.0,
                        float(pyo.value(m.solar_capex_param[t]))
                        - tariff * solar_energy_per_mw
                        * float(pyo.value(m.remaining_npv_factor[t])) / required_margin,
                    ) > GAP_EPS_USD_PER_MW
                    for t in range(len(years))
                )
                zero_subsidy_reason = "unexplained" if built_with_open_gap else "gap_closed"

            row.update({
                "status": "optimal",
                "eaas_subsidy_npv_usd": eaas_subsidy_npv,
                "solar_eaas_total_mw": solar_eaas_total_mw,
                "realised_subsidy_per_mw": (
                    eaas_subsidy_npv / solar_eaas_total_mw
                    if solar_eaas_total_mw > 1e-3 else 0.0
                ),
                "system_cost_npv_usd": float(pyo.value(m.system_cost_npv)),
                "zero_subsidy_reason": zero_subsidy_reason,
                # Level 2 shadow price (None if no envelope was set)
                "concessional_shadow_usd_per_usd":
                    diag.get("concessional_shadow_usd_per_usd"),
                "concessional_utilisation":
                    diag.get("concessional_utilisation"),
                **leakage_flat(diag),
            })
            results.append(row)

    return results

def compute_frontier_shift(df, gas_availability_case=None):
    """
    From the output of run_rel3_financing_frontier(), compute:
      - feasibility_threshold_eps per (gas_case, financing_label, policy_label)
      - frontier_shift_eps = threshold(public_only) - threshold(eaas)
        per (gas_case, policy_label)

    gas_availability_case : str or None (TASK 22, Part B). Stamped onto
        every row of both returned DataFrames, exactly as TASK 13 threaded
        gas_availability_case into compute_curve_summary()'s own row
        construction for REL-2 (this function's exclusive caller is
        11_run_rel3_financing_frontier.py, confirmed by grep -- unlike
        run_reliability_sweep/_make_row above, which ARE shared with
        REL-2, so threading it here does not touch REL-2's output).
        None (the default) preserves this function's own pre-TASK-22
        signature for any caller that does not pass it.

    TASK 22, Part C (docs/task_log_22_rel3_grid.md): "floor-bound" fields
    make visible what was previously silent. `eps_floor` is the tightest
    eps value actually present in the data (NOT assumed to be
    EPS_LEVELS[-1] -- derived from df_constrained itself, so this stays
    correct regardless of which grid produced it). A threshold is
    floor-bound when the system was STILL FEASIBLE at the tightest eps
    tested -- the reported threshold is where testing stopped, not where
    feasibility actually ends. frontier_shift_eps and eaas_necessary are
    both computed from these same thresholds, so a floor-bound threshold
    on either side of a (gas_case, policy_label) cell makes BOTH derived
    quantities for that cell unreliable in the same way: "no_improvement"
    may really be "untested," and a non-necessary reading may hide a
    necessity that only appears at a tighter eps than was run.

    TASK 24: `eaas_necessary` means ONLY "public financing infeasible at
    every tested eps". It is NOT "public cannot reach a standard EaaS can";
    that wider case is `eaas_strictly_better`, and the standards it covers
    are given by `eaas_sole_route_eps_range`, the interval
    (eaas_threshold_eps, pub_threshold_eps] (string, e.g. "(0.08, 0.25]").

    Returns two DataFrames: (threshold_df, shift_df)
    """
    df_constrained = df[df["eps"].notna()].copy()
    eps_floor = float(df_constrained["eps"].min()) if len(df_constrained) else None

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
        # task_defect_closure, D17: the EaaS bankability binding-year
        # count AT THE THRESHOLD ROW specifically -- the eps level where
        # the reported feasibility_threshold_eps was actually achieved,
        # which is the row frontier_shift_eps is derived from. None for
        # financing_label != "eaas" (see the guard in _make_row).
        bankability_n_binding_at_thresh = None
        bankability_n_evaluable_at_thresh = None
        if threshold is not None:
            row_at = feasible[feasible["eps"] == threshold].iloc[0]
            cost_at_thresh = row_at["npv_cost"]
            eaas_mw_at_thresh = row_at.get("solar_eaas_total_mw")
            bankability_n_binding_at_thresh = row_at.get(
                "eaas_bankability_n_binding_years"
            )
            bankability_n_evaluable_at_thresh = row_at.get(
                "eaas_bankability_n_evaluable_years"
            )

        threshold_rows.append({
            "gas_case":                  gas,
            "financing_label":           fin,
            "policy_label":              pol,
            "gas_availability_case":     gas_availability_case,
            "feasibility_threshold_eps": threshold,
            "fully_infeasible":          infeasible_all,
            # TASK 22, Part C: True when the system was still feasible at
            # the tightest tested eps -- the threshold above is where
            # testing stopped, not a confirmed feasibility boundary.
            "threshold_is_floor_bound": (
                bool(np.isclose(threshold, eps_floor))
                if (threshold is not None and eps_floor is not None)
                else False
            ),
            "npv_cost_at_threshold":     cost_at_thresh,
            "solar_eaas_mw_at_threshold": eaas_mw_at_thresh,
            "eaas_bankability_n_binding_years_at_threshold":
                bankability_n_binding_at_thresh,
            "eaas_bankability_n_evaluable_years_at_threshold":
                bankability_n_evaluable_at_thresh,
        })

    threshold_df = pd.DataFrame(threshold_rows)

    # Frontier shift per (gas_case, policy_label)
    shift_rows = []
    for (gas, pol), grp in threshold_df.groupby(["gas_case", "policy_label"]):
        pub  = grp[grp["financing_label"] == "public_only"]
        eaas = grp[grp["financing_label"] == "eaas"]

        pub_thresh  = pub["feasibility_threshold_eps"].values[0]  if len(pub)  else None
        eaas_thresh = eaas["feasibility_threshold_eps"].values[0] if len(eaas) else None

        # TASK 22, Part C: per-cell degeneracy. Either arm's threshold
        # being floor-bound means frontier_shift_eps and eaas_necessary
        # below are both drawn from an unconfirmed boundary on that side.
        pub_floor_bound = (
            bool(pub["threshold_is_floor_bound"].values[0]) if len(pub) else False
        )
        eaas_floor_bound = (
            bool(eaas["threshold_is_floor_bound"].values[0]) if len(eaas) else False
        )
        cell_is_floor_bound = pub_floor_bound or eaas_floor_bound
        # task_defect_closure, D24: a SECOND, deeper defect underlying the
        # one TASK.md named -- found by this fix's own synthetic
        # verification (a fully-infeasible public_only arm, no run
        # performed), not by inspection. `threshold_df` is built by
        # `pd.DataFrame(threshold_rows)`; a column mixing `None` (no
        # feasible eps) with `float` values is coerced to float64 by
        # pandas, silently turning every missing threshold into `NaN`,
        # NOT `None`. The pre-fix `pub_thresh is None` test (and this
        # fix's own first draft) is therefore FALSE whenever `pub_thresh`
        # came from an actual DataFrame column with at least one real
        # float in it -- which is every real run. `pub_thresh is None`
        # could only ever fire when the public_only row was missing from
        # the group ENTIRELY (`len(pub) == 0`, e.g. a data gap), never
        # when it was present and correctly recorded as infeasible. So
        # `eaas_necessary` was effectively unreachable in practice, not
        # merely mis-classified when it did fire -- the exact "EaaS is
        # NECESSARY... the strongest possible finding" case this runner's
        # docstring names may never have been detected on any real run.
        # pd.isna() (True for both NaN and None) replaces every `is
        # None`/`is not None` test below.
        pub_missing  = pd.isna(pub_thresh)
        eaas_missing = pd.isna(eaas_thresh)

        # frontier_shift > 0 means EaaS achieves tighter reliability
        shift = (
            float(pub_thresh - eaas_thresh)
            if (not pub_missing and not eaas_missing)
            else None
        )

        # task_defect_closure, D24: ONE classification, computed once,
        # drives eaas_strictly_better, eaas_necessary, and interpretation.
        # The pre-fix code computed "necessity" (pub_thresh is None and
        # eaas_thresh is not None) and eaas_strictly_better (shift > 1e-4)
        # as two INDEPENDENT tests. Because `shift` is undefined whenever
        # pub_thresh is None, a row with eaas_necessary=True ALWAYS had
        # eaas_strictly_better=False -- the strongest possible finding
        # (this runner's own docstring: "Special case... This is the
        # strongest possible finding") was, in the very same row, flagged
        # as "not better". Same shape as D1: one row, two disagreeing
        # verdicts from two independently-thresholded fields. These are
        # NOT measuring genuinely different things -- both purport to
        # answer "is EaaS better than public financing here", so this is
        # a logic defect, not an intentional-divergence documentation one
        # (TASK.md Part 3 Sec 4's fallback does not apply).
        #
        # Five states, not two: necessity is treated as the STRONGEST form
        # of "strictly better" (feasible where nothing was), not a
        # separate, disagreeing category. A new "eaas_worse" state also
        # surfaces the case this runner's own docstring calls out as an
        # assumption, not a certainty ("Negative: Shouldn't happen (EaaS
        # weakly dominates public_only)") -- if it ever occurs, the
        # pre-fix code silently reported it identically to "no
        # improvement" (both False); per the same "must not be discarded
        # as noise" principle D1 applied to the sub-additive case, a
        # violation of the model's own dominance expectation is flagged,
        # not hidden.
        if pub_missing and eaas_missing:
            classification = "cannot_compute"
        elif pub_missing:
            classification = "eaas_necessary"
        elif eaas_missing or shift < -1e-4:
            classification = "eaas_worse"
        elif shift > 1e-4:
            classification = "eaas_strictly_better"
        else:
            classification = "no_improvement"

        eaas_strictly_better = classification in (
            "eaas_necessary", "eaas_strictly_better"
        )
        eaas_necessary = classification == "eaas_necessary"

        if classification == "cannot_compute":
            interpretation = (
                "Cannot compute — neither financing arm achieves "
                "feasibility at any tested eps level."
            )
        elif classification == "eaas_necessary":
            interpretation = (
                "EaaS is NECESSARY: makes feasible what public capital cannot."
            )
        elif classification == "eaas_worse":
            interpretation = (
                f"ANOMALY: EaaS performs WORSE than public-only here "
                f"(pub_threshold_eps={pub_thresh}, "
                f"eaas_threshold_eps={eaas_thresh}) — violates this "
                f"model's own weak-dominance expectation that EaaS should "
                f"never do worse than public financing. Flagged, not "
                f"discarded as noise."
            )
        elif classification == "eaas_strictly_better":
            interpretation = f"EaaS improves frontier by {shift:.2f} eps units."
        else:
            interpretation = (
                "EaaS provides no feasibility improvement at this "
                "gas/policy combination."
            )

        # task_defect_closure, D17: this runner's own docstring records
        # the caveat verbatim -- "at the default tariff the EaaS arm has
        # no binding capital constraint... frontier_shift_eps therefore
        # measures constrained-vs-unconstrained, not a financing effect."
        # tariff_case is now a required, sourced arm (docs/tariff_arm.md),
        # so whether that caveat applies is MEASURABLE per run rather
        # than assumed: read the EaaS bankability binding-year count AT
        # the threshold row (populated in threshold_df above). Zero
        # evaluable years (no EaaS build at all at the threshold eps) or
        # zero binding years among the evaluable ones both mean the same
        # thing here -- the EaaS arm was unconstrained at the eps level
        # this shift is computed from, so frontier_shift_eps should be
        # read as constrained-vs-unconstrained, not as a financing
        # result. None when there is no threshold row to check at all
        # (e.g. classification == "cannot_compute").
        eaas_row = eaas.iloc[0] if len(eaas) else None
        if eaas_row is None or pd.isna(
            eaas_row.get("eaas_bankability_n_evaluable_years_at_threshold")
        ):
            eaas_bankability_n_binding_at_thresh = None
            eaas_not_a_financing_effect = None
        else:
            eaas_bankability_n_binding_at_thresh = int(
                eaas_row["eaas_bankability_n_binding_years_at_threshold"]
            )
            eaas_not_a_financing_effect = (
                eaas_bankability_n_binding_at_thresh == 0
            )

        # TASK 24: eaas_necessary is deliberately narrow -- public financing
        # infeasible at EVERY tested eps. The wider "public cannot reach a
        # standard EaaS can" case (public threshold looser than EaaS's) is
        # eaas_strictly_better; this field names the standards for which
        # EaaS is the sole route: eps in (eaas_threshold, pub_threshold].
        # Flat string (CSV runners need flat fields), None otherwise. Null
        # when pub is infeasible everywhere (eaas_necessary: unbounded
        # above, nothing to bound with).
        if eaas_strictly_better and not pub_missing and not eaas_missing:
            eaas_sole_route_eps_range = f"({eaas_thresh:g}, {pub_thresh:g}]"
        else:
            eaas_sole_route_eps_range = None

        shift_rows.append({
            "gas_case":              gas,
            "policy_label":          pol,
            "gas_availability_case": gas_availability_case,
            "pub_threshold_eps":     pub_thresh,
            "eaas_threshold_eps":    eaas_thresh,
            "pub_threshold_is_floor_bound":  pub_floor_bound,
            "eaas_threshold_is_floor_bound": eaas_floor_bound,
            # TASK 22, Part C: true if EITHER arm's threshold is
            # floor-bound -- frontier_shift_eps and eaas_necessary below
            # are both derived from these same two thresholds, so either
            # one being untested-past-the-floor makes both readings for
            # this cell unreliable in the same way.
            "cell_is_floor_bound":   cell_is_floor_bound,
            "frontier_shift_eps":    shift,
            "frontier_classification": classification,
            "eaas_strictly_better": eaas_strictly_better,
            "eaas_necessary":        eaas_necessary,  # EaaS rescues otherwise infeasible
            "eaas_sole_route_eps_range": eaas_sole_route_eps_range,
            "eaas_bankability_n_binding_years_at_threshold":
                eaas_bankability_n_binding_at_thresh,
            "eaas_not_a_financing_effect": eaas_not_a_financing_effect,
            "interpretation": interpretation,
        })

    return threshold_df, pd.DataFrame(shift_rows)


def run_dem1_demand_sensitivity(
    econ,
    demand_level_cases,
    ndc_configs,
    peak_multiple_case,
    reserve_margin,
    storage_cc_case,
    gas_build_case,
    gas_availability_case,
    gas_retirement_case,
    gas_capex_case,
    ccgt_efficiency_case,
    capital_recovery_case,
    dscr_case,
    us_inflation,
    capital_treatment_case,
    concessional_envelope_case,
    conversion_case,
    hydro_case,
    om_case,
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
        (no margin key: required_margin is dscr_case-derived -- TASK 29)
        "tariff_case"     : str — key of tariff_scenarios(), fed to
                            load_scenario(tariff_case=...) (docs/tariff_arm.md)
    peak_multiple_case, reserve_margin, storage_cc_case, gas_build_case,
    gas_availability_case, gas_retirement_case, gas_capex_case,
    ccgt_efficiency_case, capital_recovery_case, dscr_case, us_inflation,
    capital_treatment_case, concessional_envelope_case, conversion_case,
    hydro_case, om_case
                       : REQUIRED, no default -- threaded straight to
                         load_scenario() below. Previously omitted
                         (docs/task_shared_fn_repair.md, Step One). om_case
                         in particular must be stated explicitly -- it is
                         no longer hardcoded to "om_central".
    cap_path           : str

    Returns
    -------
    list[dict] — one row per (demand_level, ndc_scenario, financing_arm)
    """
    cap_df = pd.read_csv(cap_path)
    years_list = list(range(MODEL_START_YEAR, MODEL_END_YEAR + 1))

    results = []

    for ndc_cfg in ndc_configs:
        ndc_label    = ndc_cfg["ndc_label"]
        ndc_scenario = ndc_cfg["ndc_scenario"]
        pub_capital  = ndc_cfg["public_capital"]
        eaas_capital = ndc_cfg["eaas_capital"]
        # docs/tariff_arm.md, Steps 1-3: config dict carries a named
        # tariff_case, threaded into load_scenario() below, not a bare
        # USD/TWh number applied post-call.
        tariff_case  = ndc_cfg["tariff_case"]

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
        _s = cap_sub.set_index("year")["cap_tco2"].astype(float)
        caps = [float(_s[int(y)]) for y in years_list]

        for demand_level in demand_level_cases:

            for fin_label, fin_params in [
                ("public_only", {
                    "financing_regime": "traditional",
                    "capital_case":     pub_capital,
                }),
                ("eaas", {
                    "financing_regime": "eaas",
                    "capital_case":     eaas_capital,
                }),
            ]:
                scenario = load_scenario(
                    demand_level_case=demand_level,
                    demand_case="organic_central",
                    gas_deliverability_case="baseline",
                    capital_case=fin_params["capital_case"],
                    solar_build_case="deployment_unconstrained",
                    carbon_case="no_policy",
                    start_year=2025,
                    end_year=MODEL_END_YEAR,
                    tariff_case=tariff_case,
                    peak_multiple_case=peak_multiple_case,
                    reserve_margin=reserve_margin,
                    storage_cc_case=storage_cc_case,
                    gas_build_case=gas_build_case,
                    gas_availability_case=gas_availability_case,
                    gas_retirement_case=gas_retirement_case,
                    gas_capex_case=gas_capex_case,
                    ccgt_efficiency_case=ccgt_efficiency_case,
                    capital_recovery_case=capital_recovery_case,
                    dscr_case=dscr_case,
                    us_inflation=us_inflation,
                    capital_treatment_case=capital_treatment_case,
                    concessional_envelope_case=concessional_envelope_case,
                    conversion_case=conversion_case,
                    hydro_case=hydro_case,
                    om_case=om_case,
                )
                # Apply financing parameters
                scenario["financing_regime"] = fin_params["financing_regime"]
                # TASK 29: no required_margin override; dscr_case-derived.
                # tariff now set by load_scenario(tariff_case=...) above.

                years = list(scenario["years"])
                # Load time-varying solar CAPEX from NREL ATB (solar_low scenario).
                # solar_low declines from $1,456k/MW (2025) to $603k/MW (2045).
                solar_capex_tv = load_solar_capex_by_year(
                    scenario_name="solar_low",
                    start_year=int(years[0]),
                    end_year=int(years[-1]),
                )

                # solar_min_build_mw_per_year override to solar_min_build_default()
                # REMOVED (2.11, Step 1) -- [UNSOURCED] (plan 2.5); load_scenario()'s
                # own 0.0 default now applies. This load_scenario() call previously
                # omitted peak_multiple_case/reserve_margin/storage_cc_case/
                # gas_build_case/gas_availability_case and nine further required
                # keys, and raised before reaching here -- repaired by threading
                # them through as required parameters above
                # (docs/task_shared_fn_repair.md, Part 3).
                m = build_model(
                    scenario=scenario,
                    econ=econ,
                    emissions_cap_by_year=caps,
                    solar_capex_by_year=solar_capex_tv,
                )

                try:
                    status = solve_model(m, scenario=scenario)
                    if not status["optimal"]:
                        raise RuntimeError("non-optimal")

                    diag = extract_planning_diagnostics(m, scenario, econ)

                    cs_pv, cs_yr = shadow_series(diag, "carbon")
                    gs_pv, gs_yr = shadow_series(diag, "gas")
                    cs_pv_f = [v for v in cs_pv if v is not None]
                    # LEVEL 2: concessional envelope diagnostics now come from
                    # extract_planning_diagnostics (single source of truth).
                    concessional_shadow = diag.get("concessional_shadow_usd_per_usd")
                    concessional_drawdown = diag.get("concessional_drawdown_npv_usd")
                    concessional_utilisation = diag.get("concessional_utilisation")
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
                        "concessional_shadow_usd_per_usd": concessional_shadow,
                        "concessional_drawdown_npv_usd": concessional_drawdown,
                        "concessional_utilisation": concessional_utilisation,
                        "cumulative_emissions_tco2": float(pyo.value(m.emissions)),
                        "unserved_only_reliability": diag["unserved_only_reliability"],
                        "grid_service_reliability_horizon": diag["grid_service_reliability_horizon"],
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
                        "budget_realised_spend_pv_usd": diag.get(
                            "public_budget_realised_spend_pv_usd"
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
                        **shadow_mean_fields("carbon", cs_pv, cs_yr),
                        "carbon_binding_years":      sum(
                            1 for v in cs_pv_f if v > 1e-6
                        ),
                        **shadow_mean_fields("gas", gs_pv, gs_yr),
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
    peak_multiple_case,
    reserve_margin,
    storage_cc_case,
    gas_build_case,
    gas_availability_case,
    gas_retirement_case,
    gas_capex_case,
    ccgt_efficiency_case,
    capital_recovery_case,
    dscr_case,
    us_inflation,
    capital_treatment_case,
    concessional_envelope_case,
    conversion_case,
    hydro_case,
    om_case,
    builder,
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
    demand_cases    : list[str] —keys of scenarios.demand_growth_scenarios()
    gas_cases       : list[str] — e.g. ["baseline", "upside", "downside", "shock_recovery"]
    financing_configs : list[dict], each with:
        "label"              : str — "public_only" or "eaas"
        "financing_regime"   : str — "traditional" or "eaas"
        "capital_case"       : str
        (no required_margin key: dscr_case-derived for every arm -- TASK 29;
         a stale key in a config dict is IGNORED, not applied)
        "tariff_case"        : str — key of tariff_scenarios(), fed to
                               load_scenario(tariff_case=...). REQUIRED in
                               every config dict, including public_only --
                               previously silently defaulted to
                               tariff_legacy_95m when omitted
                               (docs/task_shared_fn_repair.md, Step One);
                               an omitted key now raises KeyError instead.
    policy_configs  : list[dict], each with:
        "policy_label"  : str
        "ndc_scenario"  : str or None
    peak_multiple_case, reserve_margin, storage_cc_case, gas_build_case,
    gas_availability_case, gas_retirement_case, gas_capex_case,
    ccgt_efficiency_case, capital_recovery_case, dscr_case, us_inflation,
    capital_treatment_case, concessional_envelope_case, conversion_case,
    hydro_case, om_case
                    : REQUIRED, no default -- threaded straight to
                      load_scenario() below. Previously omitted
                      (docs/task_shared_fn_repair.md, Step One). om_case
                      in particular must be stated explicitly -- it is no
                      longer hardcoded to "om_central".
    cap_path        : str
    builder         : {"annual", "sliced"} -- REQUIRED, NO DEFAULT (A5
                      addendum, Part 2). Selects build_model (annual) or
                      build_model_sliced, same pattern as
                      run_gas_regime_ndc_matrix's builder= (Group C).
                      Previously hardwired to the annual builder with no
                      way for its sole caller
                      (13_run_dem2_growth_gas_interaction.py) to state
                      otherwise. No default is given deliberately, for the
                      same reason Group C gave none. The function's one
                      call site is keyword-only (confirmed before this
                      change), so this breaks no positional call.

    Returns
    -------
    list[dict] — one row per (demand_case, gas_case, financing_arm, policy_arm)
    """
    _BUILDER_FNS = {"annual": build_model, "sliced": build_model_sliced}
    if builder not in _BUILDER_FNS:
        raise ValueError(
            f"run_dem2_growth_gas_matrix: builder must be one of "
            f"{sorted(_BUILDER_FNS)}, got {builder!r}."
        )
    _build_fn = _BUILDER_FNS[builder]

    cap_df = pd.read_csv(cap_path)
    years_list = list(range(MODEL_START_YEAR, MODEL_END_YEAR + 1))

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
            _s = cap_sub.set_index("year")["cap_tco2"].astype(float)
            caps = [float(_s[int(y)]) for y in years_list]
        
        solar_capex_tv = load_solar_capex_by_year(
            scenario_name="solar_low",
            start_year=2025,
            end_year=MODEL_END_YEAR,
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
                        solar_build_case="deployment_unconstrained",
                        carbon_case="no_policy",
                        start_year=2025,
                        end_year=MODEL_END_YEAR,
                        # docs/tariff_arm.md, Steps 1-3: named tariff_case
                        # threaded in, not a bare number applied post-call.
                        # REQUIRED in every fin_cfg, including public_only
                        # (tariff is unused there but must still be stated
                        # explicitly -- docs/task_shared_fn_repair.md, Step
                        # One: a silent fallback to tariff_legacy_95m was
                        # the same defect class this repair closes).
                        tariff_case=fin_cfg["tariff_case"],
                        peak_multiple_case=peak_multiple_case,
                        reserve_margin=reserve_margin,
                        storage_cc_case=storage_cc_case,
                        gas_build_case=gas_build_case,
                        gas_availability_case=gas_availability_case,
                        gas_retirement_case=gas_retirement_case,
                        gas_capex_case=gas_capex_case,
                        ccgt_efficiency_case=ccgt_efficiency_case,
                        capital_recovery_case=capital_recovery_case,
                        dscr_case=dscr_case,
                        us_inflation=us_inflation,
                        capital_treatment_case=capital_treatment_case,
                        concessional_envelope_case=concessional_envelope_case,
                        conversion_case=conversion_case,
                        hydro_case=hydro_case,
                        om_case=om_case,
                    )

                    scenario["financing_regime"] = fin_cfg["financing_regime"]
                    # TASK 29: required_margin is NOT overridden here; it is
                    # the dscr_case-derived value load_scenario() set above.

                    build_kwargs = dict(scenario=scenario, econ=econ)
                    if caps is not None:
                        build_kwargs["emissions_cap_by_year"] = caps
                    else:
                        # Plan 0.4, item C: 1e18 sentinel -> None ("no
                        # cap", already the builder's own guarded meaning
                        # here). carbon_shadow_pv_usd_per_tco2_by_year
                        # values below are None wherever no
                        # emissions_constraint exists; cs_vals already
                        # filters those out and the carbon shadow means (carbon_shadow_pv_all_year_mean, carbon_shadow_pv_binding_mean) already
                        # falls back to 0.0 on an empty list (see below),
                        # so this is numerically unchanged.
                        build_kwargs["emissions_cap"] = None

                    build_kwargs["solar_capex_by_year"] = solar_capex_tv
                    m = _build_fn(**build_kwargs)

                    try:
                        status = solve_model(m, scenario=scenario)
                        if not status["optimal"]:
                            raise RuntimeError("non-optimal")

                        diag = extract_planning_diagnostics(m, scenario, econ)
                        T = range(len(scenario["years"]))

                        cs_pv, cs_yr = shadow_series(diag, "carbon")
                        gs_pv, gs_yr = shadow_series(diag, "gas")
                        cs_pv_f = [v for v in cs_pv if v is not None]

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
                            "unserved_only_reliability": diag["unserved_only_reliability"],
                            "grid_service_reliability_horizon": diag["grid_service_reliability_horizon"],
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
                            **shadow_mean_fields("gas", gs_pv, gs_yr),
                            **shadow_mean_fields("carbon", cs_pv, cs_yr),
                            "carbon_binding_years":  sum(
                                1 for v in cs_pv_f if v > 1e-6
                            ),
                            "budget_shadow":         diag.get(
                                "public_budget_shadow_usd_per_usd"
                            ),
                            "budget_utilisation":    diag.get(
                                "public_budget_utilisation"
                            ),
                            **leakage_flat(diag),
                            **annual_emissions_flat(m, scenario["years"]),
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
                         capital_case, tariff (EaaS) (required_margin: dscr_case-derived, TASK 29)
    policy_configs     : list[dict] with keys: policy_label, ndc_scenario
    cap_path           : str

    Returns
    -------
    list[dict] — one row per (H_d, surplus_frac, financing_arm, policy_arm)
    """
    cap_df = pd.read_csv(cap_path)
    years_list = list(range(MODEL_START_YEAR, MODEL_END_YEAR + 1))

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
            _s = cap_sub.set_index("year")["cap_tco2"].astype(float)
            caps = [float(_s[int(y)]) for y in years_list]

        solar_capex_tv = load_solar_capex_by_year(
            scenario_name="solar_low",
            start_year=2025,
            end_year=MODEL_END_YEAR,
        )

        for hd in hd_values:
            for sfrac in surplus_frac_values:
                for fin_cfg in financing_configs:
                    fin_label = fin_cfg["label"]

                    scenario = load_scenario(
                        demand_level_case="served",
                        demand_case="organic_central",
                        gas_deliverability_case="baseline",
                        capital_case=fin_cfg["capital_case"],
                        solar_build_case="deployment_unconstrained",
                        carbon_case="no_policy",
                        start_year=2025,
                        end_year=MODEL_END_YEAR,
                        # docs/tariff_arm.md, Steps 1-3: see the sibling
                        # load_scenario() call above.
                        tariff_case=fin_cfg.get("tariff_case", "tariff_legacy_95m"),
                        om_case="om_central",
                    )

                    # Override storage parameters
                    scenario["storage_deployable_hours_per_year"] = hd
                    scenario["storage_solar_surplus_frac"]        = sfrac
                    # solar_min_build_mw_per_year override to solar_min_build_default
                    # REMOVED (2.11, Step 1) -- also fixes a missing-parentheses bug
                    # (this assigned the function object, not its return value; see
                    # docs/phase_2_11_minbuild_retirement.md, Step 0). [UNSOURCED]
                    # (plan 2.5); load_scenario()'s own 0.0 default now applies. This
                    # load_scenario() call is also already missing peak_multiple_case/
                    # reserve_margin/storage_cc_case/gas_build_case/
                    # gas_availability_case (required since 2.6/2.9/2.10) and would
                    # raise before reaching here regardless.

                    scenario["financing_regime"] = fin_cfg["financing_regime"]
                    # TASK 29: no required_margin override (dscr_case-derived).
                    # tariff set by load_scenario(tariff_case=...) above.

                    build_kwargs = dict(scenario=scenario, econ=econ)
                    if caps is not None:
                        build_kwargs["emissions_cap_by_year"] = caps
                    else:
                        # Plan 0.4, item C: 1e18 sentinel -> None ("no
                        # cap", already the builder's own guarded meaning
                        # here).
                        build_kwargs["emissions_cap"] = None

                    build_kwargs["solar_capex_by_year"] = solar_capex_tv
                    
                    m = build_model(**build_kwargs)

                    try:
                        status = solve_model(m, scenario=scenario)
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
                            "unserved_only_reliability": diag["unserved_only_reliability"],
                            "grid_service_reliability_horizon": diag["grid_service_reliability_horizon"],
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
                                * _storage_capex_at(m, econ, t)
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
    peak_multiple_case,
    reserve_margin,
    storage_cc_case,
    gas_build_case,
    gas_availability_case,
    gas_retirement_case,
    gas_capex_case,
    ccgt_efficiency_case,
    capital_recovery_case,
    dscr_case,
    us_inflation,
    capital_treatment_case,
    concessional_envelope_case,
    conversion_case,
    hydro_case,
    tariff_case,
    om_case,
    builder,
    capital_case="moderate",
    eps_levels=None,
    solar_build_case=None,
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
    peak_multiple_case, reserve_margin, storage_cc_case, gas_build_case,
    gas_availability_case, gas_retirement_case, gas_capex_case,
    ccgt_efficiency_case, capital_recovery_case, dscr_case, us_inflation,
    capital_treatment_case, concessional_envelope_case, conversion_case,
    hydro_case, tariff_case, om_case
                 : REQUIRED, no default -- threaded straight to
                   load_scenario() below. Previously omitted entirely,
                   including tariff_case (docs/task_shared_fn_repair.md,
                   Step One). om_case in particular must be stated
                   explicitly -- it is no longer hardcoded to "om_central".
    builder      : {"annual", "sliced"} -- REQUIRED, NO DEFAULT (A5
                   addendum, Part 3). Threaded straight to
                   run_reliability_sweep()'s own required builder=
                   parameter below. No default is given deliberately, for
                   the same reason Group C gave none.
    capital_case : str — fixed capital envelope
    eps_levels   : list[float] or None

    Returns
    -------
    list[dict]  — all rows from all gas regime sweeps combined.
                  Use reliability_results_to_df() to convert to DataFrame.
    """
    # TASK 46: opt-in solar build arm; None = pass nothing, so load_scenario's own
    # default (nirp_trajectory) applies exactly as before.
    _solar_kw = {} if solar_build_case is None else {"solar_build_case": solar_build_case}
    if eps_levels is None:
        eps_levels = reliability_levels_log()

    all_rows = []

    for gas_case in gas_cases:

        scenario = load_scenario(
            **_solar_kw,
            demand_level_case="served",
            demand_case="organic_central",
            gas_deliverability_case=gas_case,
            capital_case=capital_case,
            carbon_case="no_policy",
            start_year=2025,
            end_year=MODEL_END_YEAR,
            peak_multiple_case=peak_multiple_case,
            reserve_margin=reserve_margin,
            storage_cc_case=storage_cc_case,
            gas_build_case=gas_build_case,
            gas_availability_case=gas_availability_case,
            gas_retirement_case=gas_retirement_case,
            gas_capex_case=gas_capex_case,
            ccgt_efficiency_case=ccgt_efficiency_case,
            capital_recovery_case=capital_recovery_case,
            dscr_case=dscr_case,
            us_inflation=us_inflation,
            capital_treatment_case=capital_treatment_case,
            concessional_envelope_case=concessional_envelope_case,
            conversion_case=conversion_case,
            hydro_case=hydro_case,
            tariff_case=tariff_case,
            om_case=om_case,
        )

        rows = run_reliability_sweep(
            scenario=scenario,
            econ=econ,
            builder=builder,  # A5 addendum, Part 3: threaded from this function's own required builder=
            gas_case=gas_case,
            # task_defect_closure, D23: this function's own eps_levels
            # (resolved above) previously never reached
            # run_reliability_sweep, which unconditionally used
            # reliability_levels_log() regardless of what was passed here.
            eps_levels=eps_levels,
        )
        all_rows.extend(rows)

    return all_rows


# ============================================================
# DIAGNOSTICS EXTRACTION
# ============================================================

def _series_dict_by_year(values, years):
    return {int(y): float(values[t]) for t, y in enumerate(years)}


def _storage_capex_at(m, econ, t):
    """Storage capex (USD per MWh) of the vintage built, or rebuilt, in model
    index t: the builder's own m.storage_capex_param[t] (TASK 53 2.2: the
    default is the declining NREL series), falling back to the flat econ scalar
    only for a model that carries no such parameter."""
    sc = getattr(m, "storage_capex_param", None)
    return float(sc[t]) if sc is not None else float(econ["STORAGE_COST_PER_MWH"])


# Plan 0.3, defect 3: tolerance for the dual-sign-convention assertions
# below. A value more negative than -DUAL_SIGN_TOL after sign conversion
# means the sign convention is wrong, not that the solver was noisy -- a
# bare max(0.0, val) clamp would silently zero that signal. Small
# negatives within tolerance still clamp to 0.0 as before.
DUAL_SIGN_TOL = 1e-6


def _df_list(m, years):
    return [float(pyo.value(m.DF[t])) for t in range(len(list(years)))]


def extract_gas_shadow_pv_usd_per_twh_th(m, years):
    """
    Gas deliverability shadow price, PRESENT-VALUE basis: -dual of
    m.gas_balance[t], in year-0 USD per TWh_th. The objective is entirely
    present-value (every cost term carries m.DF[t]), so the dual of a
    constraint written in undiscounted TWh_th inherits that discounting
    (TASK 52 G8-3, E4; register L101, L106). Year-t USD is the _yr_ series.
    """
    out = {}
    for t, y in enumerate(years):
        dual = m.dual.get(m.gas_balance[t], None)
        val = None if dual is None else float(-dual)   # sign convention
        if val is not None and val < -DUAL_SIGN_TOL:
            raise ValueError(
                f"gas shadow price dual is negative after sign conversion: "
                f"{val} in year {int(y)}. This indicates a wrong sign "
                "convention, not solver noise."
            )
        out[int(y)] = None if val is None else max(0.0, val)  # clamp solver noise
    return out


def extract_gas_shadow_yr_usd_per_twh_th(m, years):
    """The _pv_ series divided by m.DF[t] (read from the solved model): year-t USD per TWh_th."""
    pv = extract_gas_shadow_pv_usd_per_twh_th(m, years)
    d = _df_list(m, years)
    return {int(y): (None if pv[int(y)] is None else pv[int(y)] / d[t]) for t, y in enumerate(years)}


def extract_carbon_shadow_pv_usd_per_tco2(m, years):
    """
    Carbon shadow price, ANNUAL-CAP branch, PRESENT-VALUE basis (year-0 USD
    per tCO2): -dual of m.emissions_constraint[t]. Returns year -> value, or
    None for every year when there is no constraint or when the cap is a
    single scalar: a scalar cap has no per-year value (see
    extract_carbon_shadow_capwide_pv_usd_per_tco2), and TASK 53 2.3 stops
    writing one scalar across every year.
    """
    years = list(years)
    if not hasattr(m, "emissions_constraint") or not m.emissions_constraint.is_indexed():
        return {int(y): None for y in years}
    out = {}
    for t, y in enumerate(years):
        dual = m.dual.get(m.emissions_constraint[t], None)
        val = None if dual is None else float(-dual)
        if val is not None and val < -DUAL_SIGN_TOL:
            raise ValueError(
                f"carbon shadow price dual is negative after sign "
                f"conversion: {val} in year {int(y)}. This indicates a "
                "wrong sign convention, not solver noise."
            )
        out[int(y)] = None if val is None else max(0.0, val)
    return out


def extract_carbon_shadow_yr_usd_per_tco2(m, years):
    """The annual-cap _pv_ series divided by m.DF[t]: year-t USD per tCO2."""
    pv = extract_carbon_shadow_pv_usd_per_tco2(m, years)
    d = _df_list(m, years)
    return {int(y): (None if pv[int(y)] is None else pv[int(y)] / d[t]) for t, y in enumerate(years)}


def extract_carbon_shadow_capwide_pv_usd_per_tco2(m):
    """
    Single-cap branch: the PRESENT-VALUE marginal cost (year-0 USD) of one
    more tonne of the cap-wide total, -dual of the scalar m.emissions_constraint.
    It has no per-year value (the marginal tonne is abated in whichever year
    is cheapest, which is endogenous), so it is returned as one number and
    never written to a year. None when the cap is annual or absent. No
    runner uses this branch (TASK 52 section 3.1).
    """
    if not hasattr(m, "emissions_constraint") or m.emissions_constraint.is_indexed():
        return None
    dual = m.dual.get(m.emissions_constraint, None)
    val = None if dual is None else float(-dual)
    if val is not None and val < -DUAL_SIGN_TOL:
        raise ValueError(
            f"carbon shadow price dual is negative after sign conversion: "
            f"{val} (scalar cap). This indicates a wrong sign convention, "
            "not solver noise."
        )
    return None if val is None else max(0.0, val)


def compute_unserved_only_metrics(demand_by_year, unserved_by_year):
    """
    UNSERVED-ONLY reliability (TASK 53 2.6; register L103). 1 - unserved/demand.

    Under a priced self-generation case the builder fixes unserved energy at zero
    (D11), so every value here is exactly 1.0 and says nothing about the
    reliability the model enforces. The quantity the model enforces is the
    grid-service SHORTFALL, unserved + self-generation (see
    compute_grid_service_metrics). These fields are renamed `unserved_only_*` so
    that nothing reads them as reliability; they are kept because they are the
    right quantity when unserved energy is genuinely the only shortfall (a
    VoLL-priced run with no self-generation).
    """
    years = sorted(demand_by_year.keys())
    annual = {}
    for y in years:
        d = demand_by_year[y]
        annual[y] = None if d <= 0 else 1.0 - (unserved_by_year[y] / d)
    worst = min(r for r in annual.values() if r is not None)
    total_demand = sum(demand_by_year.values())
    total_unserved = sum(unserved_by_year.values())
    return {
        "unserved_only_annual_reliability_by_year": annual,
        "unserved_only_worst_year_reliability": worst,
        "unserved_only_reliability": (1.0 - total_unserved / total_demand) if total_demand > 0 else None,
        "unserved_only_expected_unserved_energy_twh": total_unserved,
    }


def compute_grid_service_metrics(demand_by_year, unserved_by_year, selfgen_by_year):
    """
    CANONICAL reliability (TASK 53 2.6): the grid-service shortfall the model
    constrains, unserved + self-generation, as the REL runners always computed it
    (`shortfall = unserved + selfgen`, run_reliability_sweep._make_row). Not made
    to agree with the unserved-only fields: they are different quantities and
    carry different names.
    """
    years = sorted(demand_by_year.keys())
    short = {y: unserved_by_year[y] + selfgen_by_year[y] for y in years}
    frac, rel = {}, {}
    for y in years:
        d = demand_by_year[y]
        frac[y] = None if d <= 0 else short[y] / d
        rel[y] = None if d <= 0 else 1.0 - frac[y]
    total_demand = sum(demand_by_year.values())
    total_short = sum(short.values())
    valid = [r for r in rel.values() if r is not None]
    return {
        "grid_shortfall_twh_by_year": short,
        "grid_shortfall_total_twh": total_short,
        "grid_service_reliability_by_year": rel,
        "grid_service_reliability_worst_year": min(valid) if valid else None,
        "grid_service_reliability_horizon": (1.0 - total_short / total_demand) if total_demand > 0 else None,
    }


def annual_emissions_flat(m, years):
    """TASK 80 D: the ANNUAL emissions series of a solved model as FLAT row fields (CSV runners need flat fields):
    emissions_total_tco2_<year> (physical, grid + self-generation), emissions_capped_tco2_<year> (the quantity the cap
    constrains) and emissions_selfgen_tco2_<year>. Grid-only is total minus self-generation. Before this the capped runners
    stored cumulative totals only, which is why F-43's reporting-window turning point could not be located."""
    out = {}
    for t, y in enumerate(years):
        out[f"emissions_total_tco2_{int(y)}"] = float(pyo.value(m.emissions_by_year[t]))
        out[f"emissions_capped_tco2_{int(y)}"] = float(pyo.value(m.emissions_capped_by_year[t]))
        out[f"emissions_selfgen_tco2_{int(y)}"] = float(pyo.value(m.emissions_selfgen_by_year[t]))
    return out


def annual_emissions_nested(m, years):
    """TASK 80 D: the same series as a nested dict {series: {year: tCO2}} for JSON runners."""
    return {
        "emissions_total_tco2_by_year": {int(y): float(pyo.value(m.emissions_by_year[t])) for t, y in enumerate(years)},
        "emissions_grid_only_tco2_by_year": {int(y): float(pyo.value(m.emissions_grid_only_by_year[t])) for t, y in enumerate(years)},
        "emissions_capped_tco2_by_year": {int(y): float(pyo.value(m.emissions_capped_by_year[t])) for t, y in enumerate(years)},
        "emissions_selfgen_tco2_by_year": {int(y): float(pyo.value(m.emissions_selfgen_by_year[t])) for t, y in enumerate(years)},
    }


def leakage_flat(diag):
    """TASK 80 C: the leakage diagnostic (2.6, D3) as FLAT row fields, from diag["leakage"] (CSV runners need flat fields).
    Every runner that goes through extract_planning_diagnostics() now carries the leakage share; before this, the shared
    functions in this module never referenced it."""
    lk = diag["leakage"]
    return {
        "leakage_share": lk["leakage_share"],
        "leakage_flagged": lk["flagged"],
        "leakage_tco2": lk["leakage_tco2"],
        "emissions_selfgen_tco2": lk["emissions_selfgen_tco2"],
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

    # ------------------------------------------------------------
    # Builder-agnostic slice aggregation (Tier port, step 0)
    # ------------------------------------------------------------
    # build_model_sliced() indexes the operational quantities read below
    # over (year, slice); build_model() indexes them over (year) only.
    # Branch on the component's OWN dimensionality -- Var(T) -> 1,
    # Var(T, m.S) -> 3 -- not on a flag. For the sliced builder, recover
    # the annual total by summing over the 6 slices: these quantities are
    # additive energy (TWh) -- the per-slice electricity balance is an
    # energy balance with slice weights normalised to sum to 1 -- so the
    # plain slice sum IS the annual total. This is exactly
    # optimize_model_sliced.annual_totals()'s own aggregation (duplicated
    # inline so the annual branch stays a single unbranched expression).
    # In the annual case _slice_total(comp, t) == pyo.value(comp[t]), so
    # the annual code path is byte-identical -- it is the regression
    # instrument for every existing result.
    _sliced = m.unserved.dim() > 1
    if _sliced:
        _slice_index = list(m.S)

        def _slice_total(comp, t):
            return sum(pyo.value(comp[t, sp]) for sp in _slice_index)
    else:
        def _slice_total(comp, t):
            return pyo.value(comp[t])

    gas_avail = gas_available_power(
        start_year=int(years[0]),
        end_year=int(years[-1]),
        scenario_name=scenario["gas_scenario"],
    )["available_twh_th"]

    carbon_shadow = extract_carbon_shadow_pv_usd_per_tco2(m, years)
    carbon_shadow_yr = extract_carbon_shadow_yr_usd_per_tco2(m, years)
    carbon_shadow_capwide = extract_carbon_shadow_capwide_pv_usd_per_tco2(m)
    
    # ------------------------------------------------------------
    # Shortage diagnostics
    # ------------------------------------------------------------

    unserved_by_year = {
        int(y): float(_slice_total(m.unserved, t))
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
    # Plan 0.3, defect 2: gas_scarcity_value_usd REMOVED. It was zero by
    # construction -- when the fuel constraint binds, gas_to_power ==
    # gas_avail, so gas_shortfall == 0 and shadow * gas_shortfall == 0
    # exactly when gas is scarcest; when the constraint is slack the dual
    # is zero, so the product is zero too. There is no regime in which it
    # is non-trivially non-zero. It was also wrong in a second way:
    # converted plant draws fuel through m.convert_to_power, absent from
    # gas_shortfall's computation, so the (already-zero-by-construction)
    # quantity additionally overstated unused gas by the entire converted
    # draw wherever it wasn't exactly zero. No consumer of
    # gas_scarcity_value_usd exists anywhere in src/ or scripts/ (grep
    # confirmed) -- removal has no downstream effect. Report the annual
    # dual directly instead: gas_shadow_pv_usd_per_twh_th_by_year
    # (returned below) already carries the same information without the
    # zero-by-construction wrapper.

    gas_shadow = extract_gas_shadow_pv_usd_per_twh_th(m, years)
    gas_shadow_yr = extract_gas_shadow_yr_usd_per_twh_th(m, years)

    # ------------------------------------------------------------
    # Value of additional gas supply to electricity reliability
    # ------------------------------------------------------------

    # PV and year-t bases, as the shadow they derive from (TASK 53 2.3).
    gas_value_electricity = {}
    gas_value_electricity_yr = {}

    eta = scenario["gas_eta"]

    for y, shadow in gas_shadow.items():

        if shadow is None:
            gas_value_electricity[y] = None
            gas_value_electricity_yr[y] = None
        else:
            gas_value_electricity[y] = shadow / eta
            gas_value_electricity_yr[y] = gas_shadow_yr[y] / eta
    
    
    # ------------------------------------------------------------
    # Solar share of GRID generation (Plan 0.3, defect 1)
    # ------------------------------------------------------------
    # Denominator = gas fleet generation + converted gas generation +
    # solar generation + hydro generation.
    #
    # storage_discharge is EXCLUDED: that energy was already counted once
    # in solar_generation when it was charged, so including it
    # double-counts.
    #
    # Converted gas generation (Phase 2.8) is a SEPARATE stream from the
    # unconverted fleet's gas_generation and must be added explicitly, or
    # the denominator misses most of gas-side output under an active
    # conversion arm -- under conv_central, 21.76 TWh converted vs only
    # 2.98 TWh unconverted fleet. Read m.convert_generation if present
    # (sliced, f1_tranche, with an available conversion arm), else zero --
    # guarded with hasattr because the annual builder raises on any
    # available conversion arm and so never has this component.
    #
    # Hydro (Phase 1.5) is exogenous in BOTH builders and is read from the
    # scenario series directly rather than from a model component: the
    # annual builder has no m.hydro_gen Var at all (hydro enters its
    # balance constraint as a plain float), while the sliced builder's
    # m.hydro_season_budget constrains sum_slices(m.hydro_gen[t, .]) to
    # equal this same series EXACTLY (an equality, not a bound) -- so the
    # scenario series is the correct value under either builder.
    #
    # BOUNDARY (implemented as stated, not derived): this is a share of
    # GRID generation, matching the model's declared busbar metering
    # point. selfgen (private off-grid diesel/generator supply) is
    # EXCLUDED from both numerator and denominator -- under NDC caps it is
    # 34-58% of national emissions (finding F-10), so a national-basis
    # share would differ from this grid-basis share by roughly a factor of
    # two. selfgen_twh_by_year is reported separately (see below); use it
    # alongside this field rather than folding it in here.
    hydro_energy_twh_by_year = scenario["hydro_energy_twh_by_year"]

    def _converted_generation_total(t):
        if not hasattr(m, "convert_generation"):
            return 0.0
        return sum(
            pyo.value(v) for k, v in m.convert_generation.items() if k[1] == t
        )

    total_generation = sum(
        _slice_total(m.gas_generation, t)
        + _converted_generation_total(t)
        + _slice_total(m.solar_generation, t)
        + float(hydro_energy_twh_by_year[t])
        for t in T
    )

    solar_total = sum(
        _slice_total(m.solar_generation, t) for t in T
    )

    solar_share = solar_total / total_generation if total_generation > 0 else 0
    # ------------------------------------------------------------
    # Year solar becomes cheaper than gas
    # ------------------------------------------------------------

    solar_energy_per_mw = scenario["solar_cf"] * 8760 / 1_000_000

    # Social-rate energy NPV: use ONLY for social-cost diagnostics
    # (e.g. solar_beats_gas_year, a system-cost comparison).
    npv_energy = sum(
        pyo.value(m.DF[t]) * solar_energy_per_mw
        for t in range(len(years))
    )

    # Private/blended-rate energy NPV: use for ALL reported bankability
    # quantities so the reporting layer matches what the LP actually enforces
    # in the bankability constraint (which discounts at the private rate via
    # m.remaining_npv_factor). m.remaining_npv_factor[0] is the horizon-from-0
    # discount sum at the private/blended rate.
    remaining_npv_factor_0 = float(pyo.value(m.remaining_npv_factor[0]))
    npv_energy_priv = remaining_npv_factor_0 * solar_energy_per_mw

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
    # EaaS capital multiplier (Plan 0.3, defect 4)
    # ------------------------------------------------------------
    # public_spending here is the discounted EaaS SUBSIDY only (NOT total
    # public spending -- public solar CAPEX is a separate channel, see
    # public_solar_npv_spend below). Kept as a local name because
    # eaas_subsidy_npv_spend reuses it a few lines down; the field exposed
    # in the returned dict is named eaas_subsidy_npv_usd, not
    # eaas_public_spending_usd, so a stale reader of the old name fails
    # loudly instead of silently reading a value that was never "total
    # public spending".
    public_spending = 0.0
    private_investment = 0.0
    private_investment_undiscounted = 0.0       # same flows without DF[t] (TASK 53 2.3)

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
        private_investment_undiscounted += (
            pyo.value(m.solar_eaas_add[t])
            * pyo.value(m.solar_capex_param[t])
        )

    # eaas_multiplier_vs_subsidy: private_investment / subsidy NPV. F3
    # established that m.eaas_subsidy is forced to zero whenever the
    # tariff covers CAPEX, which it does at every SOURCED tariff arm
    # (tariff_low/central/high) -- so public_spending is normally zero and
    # this ratio is normally None. That is the expected central-case
    # result, not a bug: it measures a channel (subsidy leverage) that is
    # inactive at every sourced tariff. See eaas_multiplier_vs_public_budget
    # below for the mechanism findings F-26/F-27 actually attribute EaaS's
    # leverage to -- decoupling from the public capital ceiling, not
    # subsidy leverage.
    eaas_multiplier_vs_subsidy = None

    if public_spending > 0:
        eaas_multiplier_vs_subsidy = private_investment / public_spending

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

    # eaas_multiplier_vs_public_budget (Plan 0.3, defect 4, ADDED metric):
    # private_investment divided by the discounted public solar CAPEX plus
    # storage CAPEX plus EaaS subsidy -- the same sum m.public_budget_
    # constraint bounds (src/optimize_model.py / optimize_model_sliced.py,
    # "PUBLIC CAPITAL BUDGET CONSTRAINT"). Guarded on that constraint's
    # existence, since the multiplier is meaningless when nothing bounds
    # public spending. Findings F-26/F-27 establish that EaaS's mechanism
    # is decoupling deployment from the PUBLIC CAPITAL CEILING, not
    # subsidy leverage: EaaS solar capex does not enter the public budget
    # constraint, so total solar and system cost are unchanged while that
    # constraint is slack. How a given plan is split between public and
    # EaaS financing is NOT identified, in any capital case (F-12 as
    # corrected 2026-10-08): both private_investment and the public spend
    # below are split-derived, so this multiplier is a point inside F-12's
    # interval, not a model result. eaas_multiplier_vs_subsidy above
    # measures a channel that is inactive at every sourced tariff.
    eaas_multiplier_vs_public_budget = None
    if hasattr(m, "public_budget_constraint") and econ is not None:
        storage_capex_npv_spend = sum(
            float(pyo.value(m.DF[t])) * float(pyo.value(m.storage_add[t]))
            * _storage_capex_at(m, econ, t)
            for t in T
        )
        public_budget_bound_spend = (
            public_solar_npv_spend + storage_capex_npv_spend + eaas_subsidy_npv_spend
        )
        if public_budget_bound_spend > 0:
            eaas_multiplier_vs_public_budget = (
                private_investment / public_budget_bound_spend
            )

    ndc_finance_total_spend = public_solar_npv_spend + eaas_subsidy_npv_spend
    ndc_finance_utilisation = (
        ndc_finance_total_spend / ndc_finance_budget
        if ndc_finance_budget and ndc_finance_budget > 0
        else None
    )

    unserved_dict = {
        int(y): float(_slice_total(m.unserved, t))
        for t, y in enumerate(years)
    }

    selfgen_dict = {
        int(y): float(_slice_total(m.selfgen, t))
        for t, y in enumerate(years)
    }
    reliability_metrics = compute_unserved_only_metrics(demand_dict, unserved_dict)
    grid_service_metrics = compute_grid_service_metrics(demand_dict, unserved_dict, selfgen_dict)

    storage_binding = {}

    subsidy_per_mw_by_year = {}
    for t, y in enumerate(years):
        solar_add = pyo.value(m.solar_eaas_add[t])
        if solar_add > 1e-6:
            subsidy_per_mw_by_year[int(y)] = pyo.value(m.eaas_subsidy[t]) / solar_add
        else:
            subsidy_per_mw_by_year[int(y)] = 0.0

    for t, y in enumerate(years):
        # NOTE (Tier port, step 0): under the sliced builder there is no
        # storage_deployable_hours_per_year constraint, so this power_bind
        # test compares the annual discharge total against a quantity the
        # model never constrained -- the "power_limit" classification is
        # not meaningful for sliced runs. Aggregated here only so the
        # function does not raise; a slice-native storage-binding
        # diagnostic is deferred to the port task. See
        # docs/tier_port_inventory.md 0.4.
        power_bind = abs(
            _slice_total(m.storage_discharge, t)
            - (pyo.value(m.storage_power_mw[t]) * scenario.get("storage_deployable_hours_per_year", 700.0)) / 1e6
        ) <= 1e-6

        energy_bind = abs(
            _slice_total(m.storage_discharge, t)
            - scenario["storage_round_trip_eff"] * _slice_total(m.storage_charge, t)
        ) <= 1e-6
        
        # Guard: when no storage is built, no constraint is meaningfully binding.
        # Without this, 0 <= 0 slack tests register as "power_limit" on a
        # zero-capacity battery, producing a misleading diagnostic.
        if pyo.value(m.storage_capacity_mwh[t]) <= 1e-6:
            storage_binding[int(y)] = "none"
        elif power_bind:
            storage_binding[int(y)] = "power_limit"
        elif energy_bind:
            storage_binding[int(y)] = "energy_limit"
        else:
            storage_binding[int(y)] = "none"
        # charge_limit branch removed: the solar-surplus coupling constraint was
        # dropped from build_model, so scenario["storage_solar_surplus_frac"]
        # no longer exists and this diagnostic had no constraint to report on.
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
            _land_val = float(-raw)
            if _land_val < -DUAL_SIGN_TOL:
                raise ValueError(
                    f"land shadow price dual is negative after sign "
                    f"conversion: {_land_val} in year {int(y)}. This "
                    "indicates a wrong sign convention, not solver noise."
                )
            land_shadow[int(y)] = max(0.0, _land_val)

    # ------------------------------------------------------------
    # Public capital budget shadow price (FIN-3)
    # Dual on public_budget_constraint: the marginal NPV cost reduction
    # per additional USD of public capital made available.
    # Positive -> constraint is binding (capital is the bottleneck).
    # Zero     -> constraint is slack (budget not exhausted).
    # None     -> no budget constraint was applied this run.
    # ------------------------------------------------------------
    public_budget_shadow = None
    public_budget_utilisation = None
    realised_public_spend = None
    realised_public_spend_undiscounted = None

    # Gate on the CONSTRAINT's existence, not on a flag attribute that the
    # model never sets. (The prior `has_public_budget_constraint` flag was
    # never assigned in build_model, so this dual was always None.)
    if hasattr(m, "public_budget_constraint"):
        raw_dual = m.dual.get(m.public_budget_constraint, None)
        if raw_dual is not None:
            _pub_budget_val = float(-raw_dual)
            if _pub_budget_val < -DUAL_SIGN_TOL:
                raise ValueError(
                    f"public budget shadow price dual is negative after "
                    f"sign conversion: {_pub_budget_val}. This indicates a "
                    "wrong sign convention, not solver noise."
                )
            public_budget_shadow = max(0.0, _pub_budget_val)

        realised_public_spend_undiscounted = sum(
            (
                float(pyo.value(m.solar_public_add[t])) * float(pyo.value(m.solar_capex_param[t]))
                + (float(pyo.value(m.storage_add[t])) * _storage_capex_at(m, econ, t)
                   if hasattr(m, "storage_add") and econ is not None else 0.0)
                + float(pyo.value(m.eaas_subsidy[t]))
            )
            for t in T
        )
        realised_public_spend = sum(
            float(pyo.value(m.DF[t])) * (
                float(pyo.value(m.solar_public_add[t])) * float(pyo.value(m.solar_capex_param[t]))
                + (float(pyo.value(m.storage_add[t])) * _storage_capex_at(m, econ, t)
                   if hasattr(m, "storage_add") and econ is not None else 0.0)
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
        cf_t = scenario["solar_cf"] # direct index: no silent fallback to a stale default
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
    # Envelope resolution, Defect 2 (docs/f1_implementation.md): the four
    # cost_<channel>_capex_npv Expressions are structurally ABSENT under
    # 'f1_tranche' (by design -- see Step 5/6 of the Steps 1-7 report), so
    # reading them unconditionally crashed every f1_tranche solve here.
    # Branch on capital_treatment_case and read the corresponding
    # Expressions for each regime. Component dict KEYS are also renamed
    # under 'f1_tranche' (capex_npv -> capital_charge_npv) so a stale
    # reader comparing e.g. "gas_capex_npv" across the two regimes gets a
    # KeyError -- fail loud -- instead of silently comparing an annuity to
    # an overnight cost.
    capital_treatment_case = scenario["capital_treatment_case"]
    cost_gas_opex     = float(pyo.value(m.cost_gas_opex_npv))
    cost_voll_penalty = float(pyo.value(m.cost_unserved_voll_npv))
    cost_selfgen      = float(pyo.value(m.cost_selfgen_npv))
    cost_carbon       = float(pyo.value(m.cost_carbon_npv))
    # task_wrong_output_repair, Part 1: m.cost_om_npv (docs/task_om_
    # implementation.md) is built unconditionally in BOTH capital-
    # treatment branches of BOTH builders, immediately beside
    # m.cost_carbon_npv (optimize_model.py:1689, optimize_model_sliced.py
    # :1924) and enters system_cost_npv unconditionally in both branches
    # too (optimize_model.py:1647/1656, optimize_model_sliced.py:1910/
    # 1919) -- read the same way as cost_carbon above, no hasattr guard
    # needed (unlike cost_salvage, which IS conditionally absent).
    cost_om           = float(pyo.value(m.cost_om_npv))
    cost_salvage      = float(pyo.value(m.salvage_npv)) if hasattr(m, "salvage_npv") else 0.0

    if capital_treatment_case == "pre_f1_overnight":
        cost_gas_capital       = float(pyo.value(m.cost_gas_capex_npv))
        cost_pub_solar_capital = float(pyo.value(m.cost_public_solar_capex_npv))
        cost_eaas_solar_capital = float(pyo.value(m.cost_eaas_solar_capex_npv))
        cost_storage_capital   = float(pyo.value(m.cost_storage_capex_npv))
        # task_wrong_output_repair, Part 4: m.cost_replacement_capex_npv
        # is built ONLY inside this branch, in both builders
        # (optimize_model.py:1223, optimize_model_sliced.py:1482 -- it
        # sits inside the same `if capital_treatment_case ==
        # "pre_f1_overnight":` block as m.salvage_npv, confirmed by
        # direct reading), so it needs the identical hasattr guard
        # cost_salvage already uses -- reading it unconditionally would
        # raise AttributeError under 'f1_tranche'. It IS a real, charged
        # objective term whenever this branch is selected: register L18
        # confirms a rebuild is generated for every lifetime value this
        # project supplies anywhere, so this is never a zero-by-
        # construction placeholder. MAGNITUDE CLASS, determined
        # structurally (not run): under the CENTRAL lifetimes
        # (asset_lifetimes(): solar 30 / storage 15 / gas 30, against a
        # 30-year horizon), solar and gas life exactly equal the horizon
        # length, so build_t + life always exceeds T_last and neither
        # ever rebuilds -- stated directly in optimize_model.py's own
        # comment above _rebuild_years(). Only STORAGE, this project's
        # smallest capital channel, rebuilds once for vintages built
        # before roughly the horizon's midpoint. Structurally a small
        # charge relative to the objective, unlike O&M's 30.9% -- folded
        # into cost_real_expenditure below (capital expenditure by
        # nature), not given a top-level cost_decomposition key, but
        # still named explicitly in _capital_components for visibility.
        cost_replacement_capex = (
            float(pyo.value(m.cost_replacement_capex_npv))
            if hasattr(m, "cost_replacement_capex_npv") else 0.0
        )
        cost_capital_total = (
            cost_gas_capital + cost_pub_solar_capital + cost_eaas_solar_capital
            + cost_storage_capital + cost_replacement_capex
        )
        _capital_components = {
            "gas_capex_npv":          round(cost_gas_capital, 0),
            "public_solar_capex_npv": round(cost_pub_solar_capital, 0),
            "eaas_solar_capex_npv":   round(cost_eaas_solar_capital, 0),
            "storage_capex_npv":      round(cost_storage_capital, 0),
            "replacement_capex_npv":  round(cost_replacement_capex, 0),
        }
    else:
        cost_gas_capital       = float(pyo.value(m.cost_gas_capital_charge_npv))
        cost_pub_solar_capital = float(pyo.value(m.cost_public_solar_capital_charge_npv))
        cost_eaas_solar_capital = float(pyo.value(m.cost_eaas_solar_capital_charge_npv))
        cost_storage_capital   = float(pyo.value(m.cost_storage_capital_charge_npv))
        # task_wrong_output_repair, Part 4: m.capital_charge_total_npv
        # (optimize_model.py:1533-1539) is READ DIRECTLY, not re-derived
        # by summing the four channel charges above, because it already
        # equals their sum PLUS m.cost_rebuild_capital_charge_npv (the
        # f1_tranche equivalent of replacement_capex_npv above, built
        # only in this branch, confirmed by direct reading) --
        # optimize_model.py:1533-1539 shows capital_charge_total_npv =
        # public_solar + eaas_solar + gas + storage capital charges +
        # cost_rebuild_capital_charge_npv, exactly. Summing only the four
        # channels here, as the pre-Part-4 code did, would silently drop
        # the rebuild charge -- the defect this Part fixes; do NOT ALSO
        # add cost_rebuild_capital_charge_npv on top of this total, or it
        # double-counts. Reading the model's own total (rather than
        # re-listing its parts) also means this line cannot go stale the
        # way a hand-copied four-channel list could: the SLICED builder's
        # own capital_charge_total_npv Expression folds in a further
        # conversion-capex term this annual-only function never lists --
        # confirmed this function is called only on build_model() models,
        # never build_model_sliced(), by checking every one of its ~25
        # call sites. Reading the total directly means a term like that
        # is captured automatically if this function is ever pointed at
        # a different builder, rather than requiring this list to be
        # found and extended again.
        cost_rebuild_capital = float(pyo.value(m.cost_rebuild_capital_charge_npv))
        # task_conversion_diagnostics, Part 2B: m.cost_conversion_capital_
        # charge_npv (Phase 2.8, optimize_model_sliced.py:1800-1806) is
        # ALREADY folded into m.capital_charge_total_npv (read directly
        # below, unchanged) whenever an available conversion arm is active
        # -- hasattr-guarded like cost_salvage/cost_replacement_capex above,
        # since it is structurally absent under conv_none. Itemised here so
        # the five named channels plus this one sum to the sixth
        # (capital_charge_total_npv) entry on a conversion-active run; before
        # this line they summed to capital_charge_total_npv MINUS the
        # conversion charge, unlabelled.
        cost_conversion_capital = (
            float(pyo.value(m.cost_conversion_capital_charge_npv))
            if hasattr(m, "cost_conversion_capital_charge_npv") else 0.0
        )
        cost_capital_total = float(pyo.value(m.capital_charge_total_npv))
        _capital_components = {
            "gas_capital_charge_npv":          round(cost_gas_capital, 0),
            "public_solar_capital_charge_npv": round(cost_pub_solar_capital, 0),
            "eaas_solar_capital_charge_npv":   round(cost_eaas_solar_capital, 0),
            "storage_capital_charge_npv":      round(cost_storage_capital, 0),
            "rebuild_capital_charge_npv":      round(cost_rebuild_capital, 0),
            "conversion_capital_charge_npv":   round(cost_conversion_capital, 0),
            "capital_charge_total_npv":        round(cost_capital_total, 0),
        }

    cost_real_expenditure = (
        cost_gas_opex + cost_capital_total + cost_carbon
    )
    # Salvage is a CREDIT (subtracted in system_cost_npv), so it enters
    # negatively. Without it the decomposition overstates the objective by the
    # terminal-value credit -- ~2.5bn on the unconstrained baseline (2.6, D2).
    # Under 'f1_tranche' cost_salvage is 0.0 (hasattr guard above -- salvage
    # is structurally absent, truncation at the horizon plays its role
    # instead), so this line is a no-op subtraction in that regime.
    # task_wrong_output_repair, Part 1: O&M (+6.602 bn, +30.9% at central
    # -- the single largest correction in this project, task_om_
    # implementation.md) was omitted here entirely -- found by
    # 16_plot_cost_decomposition.py's OWN reconciliation guard flagging
    # MISMATCH on every om_central result, not by inspection. Added as
    # its OWN top-level line, not folded into cost_real_expenditure: O&M
    # is a real resource cost like fuel, so lumping it in would be
    # defensible on cost-accounting grounds alone -- but this dict feeds
    # a STACKED decomposition a reader interprets visually, and burying a
    # 6.6bn / 30.9%-of-objective term inside an existing category, where
    # it would inflate "real expenditure" without a visible line of its
    # own, is a presentation choice with real consequences for what a
    # reader takes away, not a neutral cost-accounting one. Given O&M is
    # exactly this large, it earns the same visibility carbon and salvage
    # already have as separate lines.
    cost_total = (cost_real_expenditure + cost_selfgen
                  + cost_voll_penalty - cost_salvage + cost_om)

    cost_decomposition = {
        "real_expenditure_npv":        round(cost_real_expenditure, 0),
        "voll_penalty_npv":            round(cost_voll_penalty, 0),
        "voll_penalty_share":          round(cost_voll_penalty / cost_total, 4) if cost_total > 0 else 0,
        "selfgen_npv":                 round(cost_selfgen, 0),
        "selfgen_share":               round(cost_selfgen / cost_total, 4) if cost_total > 0 else 0,
        "om_npv":                      round(cost_om, 0),
        "om_share":                    round(cost_om / cost_total, 4) if cost_total > 0 else 0,
        "salvage_npv":                 round(cost_salvage, 0),
        "cost_total_npv":              round(cost_total, 0),
        "components": {
            "gas_opex_npv":            round(cost_gas_opex, 0),
            **_capital_components,
            "selfgen_npv":             round(cost_selfgen, 0),
            "voll_penalty_npv":        round(cost_voll_penalty, 0),
            "carbon_cost_npv":         round(cost_carbon, 0),
            "om_cost_npv":             round(cost_om, 0),
            "salvage_npv":             round(cost_salvage, 0),
        },
    }

    # ------------------------------------------------------------
    # F1 concessional draw, by vintage (envelope resolution, Step 5)
    # ------------------------------------------------------------
    # Only meaningful under 'f1_tranche' -- m.concessional_volume_by_vintage
    # and the four *_conc_vol/*_comm_vol Var pairs are structurally absent
    # under 'pre_f1_overnight'. Shows the draw order (which vintages went
    # concessional first) and the point of envelope exhaustion (where the
    # realised share drops toward 0 as the envelope binds).
    # Denominator sums ONLY concessional-eligible channels -- solar public,
    # solar EaaS, storage -- matching m.concessional_volume_by_vintage's own
    # numerator (gas excluded there since gas_conc_vol is fixed to 0.0 and
    # gas is not eligible; see envelope resolution Decision 1). Including
    # gas_comm_vol here would dilute the realised share with an ineligible
    # channel's build, understating how much of the ELIGIBLE build went
    # concessional.
    if capital_treatment_case == "f1_tranche":
        concessional_volume_by_vintage_usd = {
            int(years[t]): float(pyo.value(m.concessional_volume_by_vintage[t]))
            for t in T
        }
        concessional_share_by_vintage = {}
        for t in T:
            _eligible_vol_t = (
                float(pyo.value(m.solar_public_conc_vol[t])) + float(pyo.value(m.solar_public_comm_vol[t]))
                + float(pyo.value(m.solar_eaas_conc_vol[t])) + float(pyo.value(m.solar_eaas_comm_vol[t]))
                + float(pyo.value(m.storage_conc_vol[t])) + float(pyo.value(m.storage_comm_vol[t]))
            )
            _conc_vol_t = concessional_volume_by_vintage_usd[int(years[t])]
            # Guarded zero (not None): a vintage with no eligible build has
            # no concessional draw either, so 0.0 is the correct share, not
            # an undefined one.
            concessional_share_by_vintage[int(years[t])] = (
                _conc_vol_t / _eligible_vol_t if _eligible_vol_t > 0 else 0.0
            )
    else:
        concessional_volume_by_vintage_usd = None
        concessional_share_by_vintage = None

    # ============================================================
    # LEVEL 2: Concessional envelope shadow price
    # ============================================================
    # Marginal value of concessional capital (USD per USD of envelope).
    # Extracted HERE so every consumer of the diagnostics dict (FIN-2 blended
    # sweep, DEM-1, etc.) gets it. None when no concessional envelope was set.
    concessional_shadow = None
    concessional_drawdown = None
    concessional_utilisation = None
    if hasattr(m, "concessional_envelope_constraint"):
        raw_conc_dual = m.dual.get(m.concessional_envelope_constraint, None)
        if raw_conc_dual is not None:
            # <= constraint: dual sign convention gives a non-positive raw value;
            # report magnitude as the marginal value of relaxing the envelope.
            concessional_shadow = abs(float(raw_conc_dual))
        concessional_drawdown = float(pyo.value(m.concessional_drawdown_npv))
        env = scenario.get("concessional_envelope_npv", None)
        if env and env > 0:
            concessional_utilisation = concessional_drawdown / env
    # ------------------------------------------------------------
    # Reporting-window gross CAPEX (plan 2.5 step 4)
    # ------------------------------------------------------------
    # The model optimises over 2025-MODEL_END_YEAR so terminal-year salvage
    # effects fall outside the results. system_cost_npv therefore spans the
    # full model horizon: it is an OPTIMISATION OBJECTIVE, not a headline.
    #
    # This is the financing-relevant figure: discounted GROSS capital
    # expenditure over the reporting window only, salvage EXCLUDED. Gross and
    # window-limited for the same reason the public budget constraint is --
    # the envelope constrains cash out the door, and an investor cannot fund a
    # 2045 build with residual value realised in 2045 or later.
    #
    # CAPEX only. Fuel and O&M are paid from operating revenue, not mobilised
    # capital, so they do not belong in a capital-mobilisation figure.
    #
    # Uses RAW capex for both arms: the EaaS required_margin is a financing
    # premium, not capital deployed into assets. Consistent with the salvage
    # treatment.
    def _capex_at(t):
        v = 0.0
        if hasattr(m, "solar_public_add"):
            v += float(pyo.value(m.solar_public_add[t])) * float(pyo.value(m.solar_capex_param[t]))
        if hasattr(m, "solar_eaas_add"):
            v += float(pyo.value(m.solar_eaas_add[t])) * float(pyo.value(m.solar_capex_param[t]))
        if hasattr(m, "storage_add") and econ is not None:
            v += float(pyo.value(m.storage_add[t])) * _storage_capex_at(m, econ, t)
        if hasattr(m, "gas_add") and "gas_capex_per_mw" in scenario:
            v += float(pyo.value(m.gas_add[t])) * float(scenario["gas_capex_per_mw"])
        # Phase 2.8: OCGT-to-CCGT conversion capex (task_conversion_capex_
        # exposure, Part 2). Reads m.conversion_overnight_capex_usd_by_year
        # [t] -- a dedicated diagnostic Expression added to build_model_
        # sliced() (Part 1B) giving RAW overnight conversion capex per year,
        # BEFORE any annuity or discounting, from the same m.convert_mw[tier]
        # and the same unit cost the objective's convert_split_identity
        # constraint reads. Zero for every t except t == 0 (convert_mw is a
        # one-off decision assigned to vintage 0, Decision 5), so this
        # naturally contributes only there. hasattr guard: the Expression is
        # built only under an active conversion arm (absent under conv_none
        # and under the annual builder, which raises on any available
        # conversion arm), so this branch is a no-op and npv_capex_report /
        # npv_capex_full are byte-identical to pre-fix on any such run.
        if hasattr(m, "conversion_overnight_capex_usd_by_year"):
            v += float(pyo.value(m.conversion_overnight_capex_usd_by_year[t]))
        return v

    _report_T = [t for t, y in enumerate(years) if int(y) <= REPORT_END_YEAR]
    npv_capex_report = sum(float(pyo.value(m.DF[t])) * _capex_at(t) for t in _report_T)
    npv_capex_full = sum(float(pyo.value(m.DF[t])) * _capex_at(t) for t in range(len(years)))

    return {
        # Leakage guard (2.6, D3), TASK 80 C: computed on every solved model that reaches this extractor
        "leakage": leakage_diagnostic(m),
        # TASK 80 D: annual emissions series by boundary (total / grid-only / capped / self-generation)
        **annual_emissions_nested(m, years),
        # Cost decomposition
        "cost_decomposition": cost_decomposition,
        # F1 concessional draw by vintage (envelope resolution, Step 5).
        # None under 'pre_f1_overnight' (mechanism structurally absent).
        "concessional_volume_by_vintage_usd": concessional_volume_by_vintage_usd,
        "concessional_share_by_vintage": concessional_share_by_vintage,
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
        "value_of_gas_to_power_pv_usd_per_twh_e_by_year": gas_value_electricity,
        "value_of_gas_to_power_yr_usd_per_twh_e_by_year": gas_value_electricity_yr,
        "storage_charge_twh_e_by_year":
            {int(y): float(_slice_total(m.storage_charge, t))
             for t, y in enumerate(years)},
        # Reported at the PRIVATE/blended rate (consistent with the LP's
        # bankability constraint). Units: USD/TWh.
        "bankability_threshold":
            scenario["required_margin"] *
            (
                float(pyo.value(m.solar_capex_param[0])) /
                npv_energy_priv
            ),
        # RENAMED from solar_generation_share (Plan 0.3, defect 1): the
        # denominator changed (now gas fleet + converted gas + solar +
        # hydro, storage_discharge excluded, selfgen excluded -- see the
        # comment at this field's computation above), so the numeric value
        # is not comparable to the old field. Renamed so a stale reader of
        # that key fails loudly instead of silently reading a differently
        # defined share under the old name.
        "solar_share_of_grid_generation": solar_share,
        # Plan 0.3, defect 4: eaas_public_spending_usd RENAMED to
        # eaas_subsidy_npv_usd -- this is the discounted EaaS SUBSIDY
        # only, not total public spending (public solar CAPEX is the
        # separate ndc_finance_public_solar_npv_usd field below). The old
        # name read as "total public spending"; a stale reader of it fails
        # loudly instead of silently under-counting.
        "eaas_subsidy_npv_usd": public_spending,
        # TASK 53 2.3 (register L107): the old name `eaas_private_investment_usd`
        # held a PRESENT-VALUE sum under a name that reads as nominal USD. Removed,
        # not kept: both bases are named.
        "eaas_private_investment_pv_usd": private_investment,
        "eaas_private_investment_undiscounted_usd": private_investment_undiscounted,
        # eaas_capital_multiplier RENAMED/split into two fields: the plan
        # asked for (capex - subsidy) / subsidy, but F3 established
        # m.eaas_subsidy is forced to zero whenever the tariff covers
        # CAPEX -- true at every SOURCED tariff arm -- so that ratio
        # divides by zero in the central configuration. Report both ends
        # instead: the subsidy-relative ratio (None in the normal case)
        # and the public-budget-relative ratio (the mechanism F-26/F-27
        # actually attribute EaaS's leverage to). See the comments at
        # their computation above.
        "eaas_multiplier_vs_subsidy": eaas_multiplier_vs_subsidy,
        "eaas_multiplier_vs_public_budget": eaas_multiplier_vs_public_budget,
        "npv_salvage_credit_usd": (
            float(pyo.value(m.salvage_npv)) if hasattr(m, "salvage_npv") else None
        ),
        "ndc_finance_public_solar_npv_usd": public_solar_npv_spend,
        "ndc_finance_eaas_subsidy_npv_usd": eaas_subsidy_npv_spend,
        "ndc_finance_total_spend_usd": ndc_finance_total_spend,
        "ndc_finance_utilisation": ndc_finance_utilisation,
        "years_with_unserved_energy": years_with_unserved,
        "max_unserved_year": max_unserved_year,
        "gas_shadow_pv_usd_per_twh_th_by_year": gas_shadow,
        "gas_shadow_yr_usd_per_twh_th_by_year": gas_shadow_yr,
        "land_shadow_price_by_year": land_shadow,
        "solar_capacity_mw_by_year":
            {int(y): float(pyo.value(m.solar_capacity_mw[t])) for t, y in enumerate(years)},
        "solar_generation_twh_e_by_year":
            {int(y): float(_slice_total(m.solar_generation, t)) for t, y in enumerate(years)},
        # Private self-generation (2.6, D1). Added so consumers stop reading
        # m.selfgen[t] directly -- 01_run_baseline.py carried that workaround.
        "selfgen_twh_by_year":
            {int(y): float(_slice_total(m.selfgen, t)) for t, y in enumerate(years)},
        "discount_factor_by_year":
            {int(y): float(pyo.value(m.DF[t])) for t, y in enumerate(years)},
        "emissions_tco2_by_year":
            {int(y): float(pyo.value(m.emissions_by_year[t])) for t, y in enumerate(years)},
        "carbon_shadow_pv_usd_per_tco2_by_year": carbon_shadow,
        "carbon_shadow_yr_usd_per_tco2_by_year": carbon_shadow_yr,
        "carbon_shadow_capwide_pv_usd_per_tco2": carbon_shadow_capwide,
        "gas_avail_twh_th_by_year": _series_dict_by_year(gas_avail, years),
        # Bankable CAPEX headroom at the PRIVATE/blended rate (matches the LP).
        "bankable_capex_per_mw":
            scenario["solar_service_tariff_usd_per_twh"] *
            npv_energy_priv
            / scenario["required_margin"],
        # RENAMED from gas_to_power_twh_th_by_year / gas_generation_twh_e_by_year
        # (task_conversion_diagnostics, Part 2C). Both read ONLY m.gas_to_power /
        # m.gas_generation -- the fleet's own thermal draw / electrical output --
        # never m.convert_to_power / m.convert_generation (Phase 2.8). Under an
        # active conversion arm the converted plant supplies most of gas-fired
        # electricity (88% at central, task background) and draws its OWN
        # thermal input, absent from both fields below. There is no single
        # "total gas" quantity in the model to add them into (thermal input via
        # the two paths is metered by two structurally separate variables at
        # two different efficiencies, m.gas_to_power at fleet eta and
        # m.convert_to_power at conversion_retrofit_efficiency), so the fields
        # are renamed to say what they measure rather than fixed to be
        # complete. Deliberately NOT aliased under the old name -- a consumer
        # missed by this rename must KeyError, not silently read a fleet-only
        # figure under a name that claims to be total.
        "fleet_gas_to_power_twh_th_by_year":
            {int(y): float(_slice_total(m.gas_to_power, t)) for t, y in enumerate(years)},
        "fleet_gas_generation_twh_e_by_year":
            {int(y): float(_slice_total(m.gas_generation, t)) for t, y in enumerate(years)},
        "demand_twh_by_year": demand_dict,
        "unserved_twh_by_year": unserved_dict,
        "storage_discharge_twh_e_by_year":
            {int(y): float(_slice_total(m.storage_discharge, t)) for t, y in enumerate(years)},
        "solar_public_add_mw_by_year":
            {int(y): float(pyo.value(m.solar_public_add[t])) for t, y in enumerate(years)},
        "solar_eaas_add_mw_by_year":
            {int(y): float(pyo.value(m.solar_eaas_add[t])) for t, y in enumerate(years)},
        # TASK 53 2.6 (G8-1b, L103): the four fields below were computed from unserved energy alone,
        # which is identically zero whenever self-generation is priced, so they read 1.0 in every such
        # result. RENAMED unserved_only_* (no alias: a stale reader fails loudly); the canonical
        # reliability is the grid-service shortfall (unserved + self-generation), exposed beside them.
        **reliability_metrics,
        **grid_service_metrics,
        # FIN-3: public capital budget diagnostics
        "public_budget_shadow_usd_per_usd": public_budget_shadow,
        "public_budget_utilisation": public_budget_utilisation,
        # TASK 53 2.3 (register L107): old name `public_budget_realised_spend_usd`
        # was the budget constraint's own PV left-hand side. Removed; both bases named.
        "public_budget_realised_spend_pv_usd": realised_public_spend,
        "public_budget_realised_spend_undiscounted_usd": realised_public_spend_undiscounted,
        # LEVEL 2: concessional envelope diagnostics
        "concessional_shadow_usd_per_usd": concessional_shadow,
        "concessional_drawdown_npv_usd": concessional_drawdown,
        "concessional_utilisation": concessional_utilisation,
        # STR-1: storage utilisation and constraint binding diagnostics
        # storage_throughput_twh: total energy discharged over the horizon.
        #   Measures how much work storage actually does in the optimal plan.
        "storage_throughput_twh": sum(
            float(_slice_total(m.storage_discharge, t)) for t in T
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
            sum(float(_slice_total(m.storage_discharge, t)) for t in T)
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
        "npv_gross_capex_report_window_usd": npv_capex_report,
        "npv_gross_capex_full_horizon_usd":  npv_capex_full,
        "report_window_end_year":            int(REPORT_END_YEAR),
        "model_horizon_end_year":            int(years[-1]),        
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
    peak_multiple_case,
    reserve_margin,
    storage_cc_case,
    gas_build_case,
    gas_availability_case,
    gas_retirement_case,
    gas_capex_case,
    ccgt_efficiency_case,
    capital_recovery_case,
    dscr_case,
    us_inflation,
    capital_treatment_case,
    concessional_envelope_case,
    conversion_case,
    hydro_case,
    tariff_case,
    om_case,
    builder,
    reliability_mode="total",
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
    peak_multiple_case, reserve_margin, storage_cc_case, gas_build_case,
    gas_availability_case, gas_retirement_case, gas_capex_case,
    ccgt_efficiency_case, capital_recovery_case, dscr_case, us_inflation,
    capital_treatment_case, concessional_envelope_case, conversion_case,
    hydro_case, tariff_case, om_case
        : REQUIRED, no default -- threaded straight to load_scenario()
          below. Previously omitted entirely, including tariff_case
          (docs/task_shared_fn_repair.md, Step One). om_case in
          particular must be stated explicitly -- it is no longer
          hardcoded to "om_central".
    builder : {"annual", "sliced"} -- REQUIRED, NO DEFAULT (A5 addendum,
        Part 3). Selects build_model (annual) or build_model_sliced, same
        pattern as run_gas_regime_ndc_matrix's builder= (Group C).
        Previously hardwired to the annual builder with no way for its
        sole caller (09_run_rel1_feasibility.py) to state otherwise. No
        default is given deliberately, for the same reason Group C gave
        none. Both of this function's own build_model() call sites are
        keyword-only (confirmed before this change), so this breaks no
        positional call.
    reliability_mode : {"total", "annual"} -- TASK 03
        (docs/task_log_03_rel1_mode.md). Threaded to the eps-constrained
        build call's own `reliability_mode=` kwarg (a deprecated alias
        for `grid_service_mode`, unchanged here). Previously hardwired to
        the literal `"total"` with no way for `09_run_rel1_feasibility.py`
        to request `"annual"` instead. **Default is `"total"`,
        deliberately** -- unlike `builder`, which has no default, this
        one does, so that a call made with no new argument (e.g. any
        pre-TASK-03 test or script) reproduces current behaviour exactly;
        the no-default discipline is enforced instead on `09_`'s own CLI
        flag (`--reliability-mode`, required, no default), not on this
        function. Validated against `{"total", "annual"}`; any other
        value raises before any solve.
    capital_case : str
        Fixed capital envelope. Default "moderate" (0.85 × B*).
    eps_levels : list[float] or None
        Unserved fraction tolerance levels to sweep (0=perfect reliability).
        If None, uses reliability_levels_log() = [0.20, 0.15, 0.10, 0.08, 0.06, 0.05].

    Returns
    -------
    (list[dict], list[dict]) -- TASK 03 changed this from a single
    list[dict] to a 2-tuple:
        all_rows  : one row per (gas_case, solar_build_case, eps), plus
                    the voll_only baseline row, as before.
        year_rows : one row per (gas_case, solar_build_case, capital_case,
                    eps, year) -- NEW. Empty for any (cell, eps) that was
                    infeasible (no solved model to read a per-year series
                    from). See docs/task_log_03_rel1_mode.md, Section B.
    """
    _BUILDER_FNS = {"annual": build_model, "sliced": build_model_sliced}
    if builder not in _BUILDER_FNS:
        raise ValueError(
            f"run_rel1_feasibility_matrix: builder must be one of "
            f"{sorted(_BUILDER_FNS)}, got {builder!r}."
        )
    _build_fn = _BUILDER_FNS[builder]

    if reliability_mode not in {"total", "annual"}:
        raise ValueError(
            f"run_rel1_feasibility_matrix: reliability_mode must be one "
            f"of {{'total', 'annual'}}, got {reliability_mode!r}."
        )

    if eps_levels is None:
        eps_levels = reliability_levels_log()

    # TASK 03, Section C: flat numeric replacement for the single nested
    # extract_reliability_dual() dict this function used to carry
    # wholesale under "dual_reliability"/"dual_at_threshold". Uses TASK
    # 02's field names directly (docs/task_log_02_dual_fix.md landed
    # before this task) via the shared DUAL_FLAT_KEYS list (module level,
    # above) so this function and compute_feasibility_threshold() cannot
    # drift apart on which keys are unwrapped.
    def _dual_fields(model, diagnostics):
        d = extract_reliability_dual(model, diag=diagnostics)
        return {k: d[k] for k in DUAL_FLAT_KEYS}

    def _none_dual_fields():
        return {k: None for k in DUAL_FLAT_KEYS}

    def _year_rows(gas_case, solar_build_case, eps, diagnostics):
        # TASK 03, Section B. diag["demand_twh_by_year"] is read, not
        # recomputed -- it is the SAME project_baseline_demand() series
        # the builder itself used (extract_planning_diagnostics calls it
        # with the identical scenario["base_demand_twh"]/["demand_growth"]
        # the builder was given), so this is the builder's own series,
        # not an independent re-derivation from a growth rate.
        rows = []
        for y in sorted(diagnostics["demand_twh_by_year"].keys()):
            unserved_twh = diagnostics["unserved_twh_by_year"][y]
            selfgen_twh = diagnostics["selfgen_twh_by_year"][y]
            demand_twh = diagnostics["demand_twh_by_year"][y]
            shortfall_twh = unserved_twh + selfgen_twh
            rows.append({
                "gas_case":           gas_case,
                "solar_build_case":   solar_build_case,
                "capital_case":       capital_case,
                "eps":                eps,
                "year":               y,
                "unserved_twh":       unserved_twh,
                "selfgen_twh":        selfgen_twh,
                "shortfall_twh":      shortfall_twh,
                "demand_twh":         demand_twh,
                "shortfall_fraction": shortfall_twh / demand_twh,
                "reliability_mode":   reliability_mode,
            })
        return rows

    all_rows = []
    year_rows = []

    for gas_case in gas_cases:
        for solar_build_case in solar_build_cases:

            scenario = load_scenario(
                demand_level_case="served",
                demand_case="organic_central",
                gas_deliverability_case=gas_case,
                capital_case=capital_case,
                solar_build_case=solar_build_case,
                carbon_case="no_policy",
                start_year=2025,
                end_year=MODEL_END_YEAR,
                peak_multiple_case=peak_multiple_case,
                reserve_margin=reserve_margin,
                storage_cc_case=storage_cc_case,
                gas_build_case=gas_build_case,
                gas_availability_case=gas_availability_case,
                gas_retirement_case=gas_retirement_case,
                gas_capex_case=gas_capex_case,
                ccgt_efficiency_case=ccgt_efficiency_case,
                capital_recovery_case=capital_recovery_case,
                dscr_case=dscr_case,
                us_inflation=us_inflation,
                capital_treatment_case=capital_treatment_case,
                concessional_envelope_case=concessional_envelope_case,
                conversion_case=conversion_case,
                hydro_case=hydro_case,
                tariff_case=tariff_case,
                om_case=om_case,
            )
            years = scenario["years"]
            # Load time-varying solar CAPEX from NREL ATB (solar_low scenario).
            # solar_low declines from $1,456k/MW (2025) to $603k/MW (2045).
            solar_capex_tv = load_solar_capex_by_year(
                scenario_name="solar_low",
                start_year=int(years[0]),
                end_year=int(years[-1]),
            )

            # solar_min_build_mw_per_year override to solar_min_build_default()
            # REMOVED (2.11, Step 1) -- [UNSOURCED] (plan 2.5); load_scenario()'s
            # own 0.0 default now applies.
            #
            # This load_scenario() call previously omitted
            # peak_multiple_case/reserve_margin/storage_cc_case/
            # gas_build_case/gas_availability_case and nine further
            # required keys, and raised before reaching this point --
            # repaired by threading them through as required parameters
            # above (docs/task_shared_fn_repair.md, Part 3). TASK 03
            # (docs/task_log_03_rel1_mode.md): the eps-constrained build
            # call below now passes this function's own reliability_mode
            # parameter instead of the literal "total" -- "annual" is
            # reachable from 09_'s new required --reliability-mode arm.

            # --- VoLL-only baseline (no reliability constraint)
            m_base = _build_fn(
                scenario=scenario,
                econ=econ,
                max_offgrid_fraction=None,
                solar_capex_by_year=solar_capex_tv,
            )
            solve_model(m_base, scenario=scenario)
            diag_base = extract_planning_diagnostics(m_base, scenario, econ)

            all_rows.append({
                "gas_case":         gas_case,
                "solar_build_case": solar_build_case,
                "capital_case":     capital_case,
                "eps":              None,
                "eps_label":        "voll_only",
                "status":           "feasible",
                **leakage_flat(diag_base),
                "reliability_mode": reliability_mode,
                "npv_cost":         float(pyo.value(m_base.system_cost_npv)),
                "unserved_total":   sum(diag_base["unserved_twh_by_year"].values()),
                "selfgen_total":    sum(diag_base["selfgen_twh_by_year"].values()),
                "shortfall_total":  (sum(diag_base["unserved_twh_by_year"].values())
                                     + sum(diag_base["selfgen_twh_by_year"].values())),
                "unserved_only_reliability": diag_base["unserved_only_reliability"],
                "grid_service_reliability_horizon": diag_base["grid_service_reliability_horizon"],
                # TASK 03, Section C: no reliability constraint exists on
                # this model at all (max_offgrid_fraction=None), so
                # extract_reliability_dual's own _gso-is-None branch
                # returns all-None immediately -- called anyway (rather
                # than hand-writing the same all-None dict again) so this
                # row has the identical set of flat columns every other
                # row has, with no ragged schema between them.
                **_dual_fields(m_base, diag_base),
                **shadow_mean_fields("gas", *shadow_series(diag_base, "gas")),
                "solar_total_mw":   sum(
                    float(pyo.value(m_base.solar_public_add[t]))
                    + float(pyo.value(m_base.solar_eaas_add[t]))
                    for t in range(len(scenario["years"]))
                ),
            })
            year_rows.extend(_year_rows(gas_case, solar_build_case, None, diag_base))

            # --- Reliability-constrained sweep
            for eps in eps_levels:

                m = _build_fn(
                    scenario=scenario,
                    econ=econ,
                    reliability_max_unserved_fraction=eps,
                    reliability_mode=reliability_mode,
                    solar_capex_by_year=solar_capex_tv,
                )

                try:
                    status = solve_model(m, scenario=scenario)
                    if not status["optimal"]:
                        raise RuntimeError("non-optimal")

                    diag = extract_planning_diagnostics(m, scenario, econ)

                    all_rows.append({
                        "gas_case":         gas_case,
                        "solar_build_case": solar_build_case,
                        "capital_case":     capital_case,
                        "eps":              eps,
                        "eps_label":        f"eps_{eps:.2f}",
                        "status":           "feasible",
                        **leakage_flat(diag),
                        "reliability_mode": reliability_mode,
                        "npv_cost":         float(pyo.value(m.system_cost_npv)),
                        "unserved_total":   sum(diag["unserved_twh_by_year"].values()),
                        "selfgen_total":    sum(diag["selfgen_twh_by_year"].values()),
                        "shortfall_total":  (sum(diag["unserved_twh_by_year"].values())
                                             + sum(diag["selfgen_twh_by_year"].values())),
                        "unserved_only_reliability": diag["unserved_only_reliability"],
            "grid_service_reliability_horizon": diag["grid_service_reliability_horizon"],
                        # TASK 03, Section C: flat columns, diag now
                        # passed through (TASK 02 confirmed it was
                        # previously absent here, which is why REL-1's
                        # `binding` has been None on every prior run --
                        # see docs/task_log_03_rel1_mode.md).
                        **_dual_fields(m, diag),
                        **shadow_mean_fields("gas", *shadow_series(diag, "gas")),
                        "solar_total_mw":   sum(
                            float(pyo.value(m.solar_public_add[t]))
                            + float(pyo.value(m.solar_eaas_add[t]))
                            for t in range(len(scenario["years"]))
                        ),
                    })
                    year_rows.extend(_year_rows(gas_case, solar_build_case, eps, diag))

                except RuntimeError:
                    # TASK 03, Section B: an infeasible (gas_case,
                    # solar_build_case, eps) cell has no solved model, so
                    # no diag and no year rows -- contributes none to
                    # year_rows, per this task's own instruction.
                    all_rows.append({
                        "gas_case":         gas_case,
                        "solar_build_case": solar_build_case,
                        "capital_case":     capital_case,
                        "eps":              eps,
                        "eps_label":        f"eps_{eps:.2f}",
                        "status":           "infeasible",
                        "reliability_mode": reliability_mode,
                        "npv_cost":         None,
                        "unserved_total":   None,
                        "selfgen_total":    None,
                        "shortfall_total":  None,
                        "unserved_only_reliability": None,
                        "grid_service_reliability_horizon": None,
                        **_none_dual_fields(),
                        **none_shadow_fields("gas"),
                        "solar_total_mw":   None,
                    })

    return all_rows, year_rows


def compute_feasibility_threshold(rows, gas_availability_case=None):
    """
    From the output of run_rel1_feasibility_matrix(), compute the
    feasibility threshold per (gas_case, solar_build_case):
    the tightest eps (smallest unserved fraction allowed) that is
    still feasible.

    Returns a DataFrame with one row per (gas_case, solar_build_case).
    The key column is 'feasibility_threshold_eps':
      - None  => not even the tightest tested eps is feasible (fully infeasible)
      - a value equal to the tightest tested eps => still achievable there,
        but see threshold_is_floor_bound below
      - a looser value => a confirmed interior threshold

    gas_availability_case : str or None (TASK 23, Part B). Stamped onto
        every row, exactly as TASK 22 threaded it into
        compute_frontier_shift()'s own row construction for REL-3 (this
        function's exclusive caller is 09_run_rel1_feasibility.py,
        confirmed by grep -- unlike run_rel1_feasibility_matrix's own
        load_scenario() calls, which this does not touch). None (the
        default) preserves this function's own pre-TASK-23 signature.

    TASK 23, Part B.3 (docs/task_log_23b_rel1_arm_grid.md): "floor-bound"
    fields, same construction as TASK 22's Part C for REL-3.
    `eps_floor` is the tightest eps value actually present in `rows`
    (not assumed from EPS_LEVELS), so this stays correct regardless of
    which grid produced it. A threshold is floor-bound when the system
    was still feasible at the tightest eps tested -- the reported
    threshold is where testing stopped, not a confirmed feasibility
    boundary. REL-1's own PRIMARY FINDING (gas regime vs. solar
    investment as the primary determinant) is computed from the RANGE
    of these thresholds across gas regimes and across solar arms; a
    floor-bound threshold on either side of that comparison means the
    measured range is a lower bound on the true effect, not the effect
    itself -- it may be compressed by clipping at the floor, never
    inflated by it.

    This is the primary output for REL-1.
    """
    df = pd.DataFrame(rows)
    df_constrained = df[df["eps"].notnull()].copy()
    eps_floor = float(df_constrained["eps"].min()) if len(df_constrained) else None

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
            # TASK 03, Section C: flat numeric columns, suffixed
            # "_at_threshold" to match this function's own existing
            # convention (npv_cost_at_threshold, solar_mw_at_threshold)
            # -- replaces the single nested dual_at_threshold dict
            # (DUAL_FLAT_KEYS, module level, shared with
            # run_rel1_feasibility_matrix so the two cannot drift apart).
            dual_at_threshold = {
                f"{k}_at_threshold": threshold_row[k] for k in DUAL_FLAT_KEYS
            }
            solar_at_threshold = threshold_row["solar_total_mw"]
        else:
            cost_at_threshold = None
            dual_at_threshold = {f"{k}_at_threshold": None for k in DUAL_FLAT_KEYS}
            solar_at_threshold = None

        summary.append({
            "gas_case":                  gas_case,
            "solar_build_case":          solar_build_case,
            "reliability_mode":          grp["reliability_mode"].iloc[0],
            "gas_availability_case":     gas_availability_case,
            "feasibility_threshold_eps": threshold,
            # None means fully infeasible at all eps levels
            "fully_infeasible":          threshold is None,
            # TASK 23, Part B.3: True when still feasible at the tightest
            # tested eps -- the threshold above is where testing stopped,
            # not a confirmed feasibility boundary.
            "threshold_is_floor_bound": (
                bool(np.isclose(threshold, eps_floor))
                if (threshold is not None and eps_floor is not None)
                else False
            ),
            "n_feasible_eps":            len(feasible),
            "n_infeasible_eps":          len(infeasible),
            "npv_cost_at_threshold":     cost_at_threshold,
            **dual_at_threshold,
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
    peak_multiple_case,
    reserve_margin,
    storage_cc_case,
    gas_build_case,
    gas_availability_case,
    gas_retirement_case,
    gas_capex_case,
    ccgt_efficiency_case,
    capital_recovery_case,
    dscr_case,
    us_inflation,
    capital_treatment_case,
    concessional_envelope_case,
    conversion_case,
    hydro_case,
    tariff_case,
    om_case,
    builder,
    financing_regime="traditional",
    capital_case="moderate",
    cap_path="data/cost/processed/emissions_cap.csv",
    solar_build_case=None,
):
    """
    GAS-1 + GAS-2 + GAS-3 shared runner.

    builder : {"annual", "sliced"} -- REQUIRED, NO DEFAULT (A5, Group C).
        Selects build_model (annual) or build_model_sliced. Previously
        hardwired to the annual builder with no way for a caller to state
        otherwise -- this blocked every caller (06_run_gas1_shadow_
        benchmarks.py, 07_run_gas2_eaas_gas_relief.py,
        08_run_gas3_regime_ndc_feasibility.py) from ever reaching the
        sliced builder through this function, regardless of what those
        scripts' own build_model() calls elsewhere were changed to. No
        default is given deliberately: a silent default here would let a
        caller's builder choice go unstated, exactly the failure mode the
        model's own required-key discipline (dscr_case, storage_cc_case,
        etc.) exists to prevent. Every existing call site is keyword-only
        (confirmed at HEAD before this change), so adding a required
        parameter here breaks no positional call.

    F1, Step 3: the required_margin=1.10 kwarg this function used to carry
    is REMOVED -- it was one of the three pre-F1 required_margin defaults
    named for removal (builder 1.0 fallback; scenarios.py's unconditional
    1.10; this kwarg). dscr_case now supplies a single cushion, constant
    across financing regimes, via load_scenario() -- see dscr_scenarios().

    REPAIRED (docs/task_shared_fn_repair.md, Part 3): this function's own
    load_scenario() call previously omitted peak_multiple_case,
    reserve_margin, storage_cc_case, gas_build_case, gas_availability_case,
    gas_retirement_case, gas_capex_case, ccgt_efficiency_case,
    capital_recovery_case, dscr_case, us_inflation, capital_treatment_case,
    concessional_envelope_case, conversion_case, hydro_case, and
    tariff_case, and raised before reaching this point for every caller
    (documented in docs/phase_2_10_gas_availability.md and later phase
    reports). All sixteen are now required parameters, threaded straight
    through below.

    Solves the NDC-constrained optimisation for every gas_deliverability_case
    in gas_cases, holding the NDC cap scenario, financing regime, and capital
    envelope constant. Returns one result dict per gas case with full
    diagnostics needed for:

      GAS-1: gas shadow price vs export and domestic benchmarks
      GAS-2: EaaS gas-scarcity relief (financing_regime="eaas" arm)
      GAS-3: feasibility, cost, and carbon shadow across gas regime shapes

    Parameters
    ----------
    econ : dict
    ndc_cap_scenario : str
        e.g. "ndc3_unconditional"
    gas_cases : list[str]
        Subset of ["baseline","upside","downside","shock_recovery"]
    peak_multiple_case, reserve_margin, storage_cc_case, gas_build_case,
    gas_availability_case, gas_retirement_case, gas_capex_case,
    ccgt_efficiency_case, capital_recovery_case, dscr_case, us_inflation,
    capital_treatment_case, concessional_envelope_case, conversion_case,
    hydro_case, tariff_case, om_case
        : REQUIRED, no default -- threaded straight to load_scenario()
          below. Previously omitted entirely, including tariff_case
          (docs/task_shared_fn_repair.md, Step One) -- needed even under
          financing_regime="traditional" because load_scenario() requires
          it regardless of whether the EaaS channel is active. om_case in
          particular must be stated explicitly -- it is no longer
          hardcoded to "om_central".
    financing_regime : str
        "traditional" (GAS-1/GAS-3) or "eaas" (GAS-2 comparison arm)
    capital_case : str
    cap_path : str

    Returns
    -------
    list[dict]
    """
    # TASK 46: opt-in solar build arm; None = pass nothing, so load_scenario's own
    # default (nirp_trajectory) applies exactly as before.
    _solar_kw = {} if solar_build_case is None else {"solar_build_case": solar_build_case}
    import copy as _copy

    # A5, Group C: builder is REQUIRED, no default -- validated once, up
    # front, rather than per gas_case, so a bad value fails before any
    # solve rather than partway through the sweep.
    _BUILDER_FNS = {"annual": build_model, "sliced": build_model_sliced}
    if builder not in _BUILDER_FNS:
        raise ValueError(
            f"run_gas_regime_ndc_matrix: builder must be one of "
            f"{sorted(_BUILDER_FNS)}, got {builder!r}."
        )
    _build_fn = _BUILDER_FNS[builder]

    cap_df = pd.read_csv(cap_path)

    results = []

    for gas_case in gas_cases:

        scenario = load_scenario(
            **_solar_kw,
            demand_level_case="served",
            demand_case="organic_central",
            gas_deliverability_case=gas_case,
            capital_case=capital_case,
            carbon_case="no_policy",
            start_year=2025,
            end_year=MODEL_END_YEAR,
            peak_multiple_case=peak_multiple_case,
            reserve_margin=reserve_margin,
            storage_cc_case=storage_cc_case,
            gas_build_case=gas_build_case,
            gas_availability_case=gas_availability_case,
            gas_retirement_case=gas_retirement_case,
            gas_capex_case=gas_capex_case,
            ccgt_efficiency_case=ccgt_efficiency_case,
            capital_recovery_case=capital_recovery_case,
            dscr_case=dscr_case,
            us_inflation=us_inflation,
            capital_treatment_case=capital_treatment_case,
            concessional_envelope_case=concessional_envelope_case,
            conversion_case=conversion_case,
            hydro_case=hydro_case,
            tariff_case=tariff_case,
            om_case=om_case,
        )

        scenario["financing_regime"] = financing_regime

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
        _s = cap_sub.set_index("year")["cap_tco2"].astype(float)
        caps = [float(_s[int(y)]) for y in years]

        # Load time-varying solar CAPEX from NREL ATB (solar_low scenario).
        # solar_low declines from $1,456k/MW (2025) to $603k/MW (2045).
        solar_capex_tv = load_solar_capex_by_year(
            scenario_name="solar_low",
            start_year=int(years[0]),
            end_year=int(years[-1]),
        )

        # solar_min_build_mw_per_year override to solar_min_build_default()
        # REMOVED (2.11, Step 1) -- [UNSOURCED] (plan 2.5); load_scenario()'s
        # own 0.0 default now applies. This load_scenario() call previously
        # omitted peak_multiple_case/reserve_margin/storage_cc_case/
        # gas_build_case/gas_availability_case and eleven further required
        # keys, and raised before reaching here -- repaired by threading
        # them through as required parameters above
        # (docs/task_shared_fn_repair.md, Part 3).

        # Build and solve baseline model with an effectively non-binding cumulative cap

        m = _build_fn(
            scenario=scenario,
            econ=econ,
            emissions_cap_by_year=caps,
            solar_capex_by_year=solar_capex_tv,
        )

        try:
            status = solve_model(m, scenario=scenario)
            if not status["optimal"]:
                raise RuntimeError("non-optimal termination")

            diag = extract_planning_diagnostics(m, scenario, econ)

            gas_shadow = diag["gas_shadow_pv_usd_per_twh_th_by_year"]
            gas_shadow_yr = diag["gas_shadow_yr_usd_per_twh_th_by_year"]
            gs_pv, gs_yr = shadow_series(diag, "gas")
            cs_pv, cs_yr = shadow_series(diag, "carbon")
            gs_vals = [v for v in gs_yr if v is not None]       # YEAR-t basis: the benchmarks are year-t USD/TWh_th (TASK 53 2.3, was G8-3)
            cs_vals = [v for v in cs_pv if v is not None]

            # Export/domestic benchmark comparison (USD/TWh_th)
            # BENCHMARKS / MMBTU_PER_TWH_TH: single source, src/gas_benchmarks.py
            # (imported at module top; register L35 item 1).

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
                **leakage_flat(diag),
                **annual_emissions_flat(m, scenario["years"]),
                # GAS-1: scarcity rent
                **shadow_mean_fields("gas", gs_pv, gs_yr),
                "gas_shadow_pv_max":    float(max(v for v in gs_pv if v is not None)) if any(v is not None for v in gs_pv) else 0.0,
                "gas_shadow_yr_max":    float(max(gs_vals)) if gs_vals else 0.0,
                # TASK 20 Part B: the denominator this numerator is actually
                # drawn from -- len(gs_vals), NOT a hardcoded horizon length.
                # Mirrors eaas_bankability_binding_years()'s own
                # n_binding_years/n_evaluable_years pair (same file, above):
                # "evaluable" years exclude any year whose dual came back
                # None, so a fixed denominator would overstate the sample a
                # scenario with fewer evaluable years is actually drawn from.
                "gas_shadow_pv_by_year": gas_shadow,
                "gas_shadow_yr_by_year": gas_shadow_yr,
                "benchmark_comparison": benchmark_comparison,
                # GAS-2/GAS-3: constraint interaction
                **shadow_mean_fields("carbon", cs_pv, cs_yr),
                "carbon_shadow_pv_max": float(max(cs_vals)) if cs_vals else 0.0,
                "carbon_shadow_yr_max": float(max(v for v in cs_yr if v is not None)) if any(v is not None for v in cs_yr) else 0.0,
                "carbon_binding_years": sum(1 for v in cs_vals if v > 1e-6),
                "carbon_shadow_evaluable_years": len(cs_vals),
                # GAS-3: cost and access outcomes
                "npv_total_cost_usd":   float(pyo.value(m.system_cost_npv)),
                "cumulative_unserved_twh": sum(diag["unserved_twh_by_year"].values()),
                "cumulative_emissions_tco2": float(pyo.value(m.emissions)),
                "unserved_only_reliability": diag["unserved_only_reliability"],
                "grid_service_reliability_horizon": diag["grid_service_reliability_horizon"],
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
                        "gas_to_power_twh_th": diag["fleet_gas_to_power_twh_th_by_year"][int(years[t])],
                        "emissions_tco2":    diag["emissions_tco2_by_year"][int(years[t])],
                        "annual_cap_tco2":   caps[t],
                        "cap_slack_tco2":    caps[t] - diag["emissions_tco2_by_year"][int(years[t])],
                        "carbon_shadow_pv":  diag["carbon_shadow_pv_usd_per_tco2_by_year"][int(years[t])],
                        "carbon_shadow_yr":  diag["carbon_shadow_yr_usd_per_tco2_by_year"][int(years[t])],
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
    end_year=MODEL_END_YEAR,
):
    """
    LEGACY batch wrapper around run_deterministic_scenario (non-LP simulator).
    Outputs are undiscounted and use must-run gas + fixed 2025 CAPEX; they are
    NOT comparable to LP results and must NOT populate thesis tables. Use the
    LP-based runners (build_model + extract_planning_diagnostics) instead.
    """
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
                        om_case="om_central",
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
def run_tariff_bankability_sweep(*args, **kwargs):
    """
    DEPRECATED — superseded by run_fin2_blended_bankability_sweep.

    The old single-rate FIN-2 sweep priced its analytical self-financing
    threshold off econ["SOLAR_CAPEX_PER_MW"] (fixed 2025 scalar) while the LP
    prices solar off the time-varying NREL ATB trajectory, and it predates the
    blended-finance capital structure. Use run_fin2_blended_bankability_sweep,
    which derives the private rate from the blend and uses the 2025 CAPEX from
    the time-varying trajectory consistently.
    """
    raise NotImplementedError(
        "run_tariff_bankability_sweep is deprecated. "
        "Use run_fin2_blended_bankability_sweep instead."
    )


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

            # Plan 0.4, item C: 1e18 sentinel -> None ("no cap", already
            # the builder's own guarded meaning for this argument).
            m = build_model(
                scenario=scenario_copy,
                econ=econ,
                emissions_cap=None,
                emissions_cap_by_year=None,
            )

            solve_model(m, scenario=scenario_copy)

            diag = extract_planning_diagnostics(m, scenario_copy, econ)

            achieved_reliability = diag["grid_service_reliability_worst_year"]      # canonical: unserved + self-generation (TASK 53 2.6)

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
                demand_case="organic_central",
                gas_deliverability_case=case["gas"],
                capital_case=case["capital"],
                tariff_case="tariff_low",
                solar_build_case=build_case,
                om_case="om_central",
            )

            scenario["financing_regime"] = "eaas"

            # Plan 0.4, item C: 1e18 sentinel -> None ("no cap", already
            # the builder's own guarded meaning for this argument).
            m = build_model(
                scenario=scenario,
                econ=econ,
                emissions_cap=None
            )

            solve_model(m, scenario=scenario)

            diagnostics = extract_planning_diagnostics(m, scenario, econ)
            summary = summarize_run(m, diagnostics, scenario)

            run_id = f'{case["name"]}_{build_case}'
            write_run_outputs(run_id, diagnostics, summary)

            results.append({
                "case": case["name"],
                "solar_build_case": build_case,
                "gas_scenario": case["gas"],
                "capital_scenario": case["capital"],

                "unserved_only_reliability": diagnostics["unserved_only_reliability"],
                "unserved_only_worst_year_reliability": diagnostics["unserved_only_worst_year_reliability"],
                "unserved_only_expected_unserved_energy_twh": diagnostics["unserved_only_expected_unserved_energy_twh"],
                "grid_service_reliability_horizon": diagnostics["grid_service_reliability_horizon"],
                "grid_service_reliability_worst_year": diagnostics["grid_service_reliability_worst_year"],

                "solar_total_mw": summary["solar_total_built_mw"],
                "gas_shadow_pv_avg": summary["gas_shadow_pv_avg_usd_per_twh_th"],
                "gas_shadow_yr_avg": summary["gas_shadow_yr_avg_usd_per_twh_th"],
            })

    return results