"""
07_run_gas2_eaas_gas_relief.py  —  GAS-2
=========================================

RESEARCH QUESTION (GAS-2)
--------------------------
How does EaaS solar deployment change the scarcity rent of gas — and does
private capital deployment functionally relieve the gas constraint that
public-only investment cannot?

WHAT THIS ANSWERS
-----------------
The gas shadow price (dual on gas_balance[t]) measures how binding the
gas deliverability constraint is. If EaaS causes the shadow price to fall
— i.e., the gas constraint becomes less binding when private capital builds
more solar — then EaaS is not merely a financing tool: it is providing
a functional energy security service by substituting for constrained gas.

The question has two parts:
  (a) Does EaaS reduce gas shadow prices relative to no-EaaS (public-only)?
  (b) Is the reduction larger under capital-constrained public investment
      (tight/moderate budget) than under unconstrained?
      This would confirm EaaS provides relief that public capital alone cannot.

NDC3.0 CONTEXT
--------------
The previous phase8 comparison (ndc_unconditional_20_eaas vs
ndc_unconditional_20) used NDC2.0 caps and a single gas scenario (baseline).
Updated here to use ndc3_unconditional and ndc3_conditional, and to run
across the full gas regime matrix so the EaaS relief effect can be assessed
under both favourable and adverse gas supply conditions.

The conditional arm (ndc3_conditional + EaaS) is
the most policy-relevant: it asks whether concessional-finance-enabled EaaS
can relieve the gas constraint under Nigeria's most ambitious NDC target.

OUTPUTS
--------
  results/gas2/eaas_vs_noeaas_gas_shadow.csv  — shadow comparison per case
  results/gas2/gas2_summary.json              — relief magnitude and direction
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
import pyomo.environ as pyo
from src.cli_arms import add_solar_build_args, solar_build_kwargs, resolve_results_dir  # TASK 46
from src.scenarios import (
    load_scenario, asset_lifetime_sweep, tariff_scenarios, MODEL_END_YEAR,
    dscr_scenarios,
)
from src.optimize_model import solve_model
from src.optimize_model_sliced import build_model_sliced
from src.optimize_experiments import extract_planning_diagnostics, shadow_mean_fields, shadow_series, leakage_flat

from src.io import (
    load_econ, load_solar_capex_by_year)
from src.io import load_econ
from src.optimize_experiments import run_gas_regime_ndc_matrix
from src.cli_arms import add_standard_arms, standard_arms_kwargs
from src.gas_benchmarks import BENCHMARKS, MMBTU_PER_TWH_TH

# ============================================================
# CONFIG
# ============================================================

RESULTS_DIR = ROOT / "results" / "gas2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAP_PATH = ROOT / "data" / "cost" / "processed" / "emissions_cap.csv"
CANONICAL_VOLL = "voll_mid"

# task_runner_revival, Part 2A: this runner's two internal load_scenario()
# calls (in run_no_policy_gas_matrix_eaas / run_no_policy_gas_matrix) and
# its two run_gas_regime_ndc_matrix() calls (in main()) omitted the
# standard arm set outright -- no argparse existed at all, and both
# hardcoded a bare tariff_case=CANONICAL_TARIFF_CASE literal instead of a
# flag. Revived via add_standard_arms(parser)/standard_arms_kwargs(args),
# following 05_run_fin3_capital_sweep.py's pattern: the CANONICAL_TARIFF_
# CASE literal is retired (superseded by the --tariff-case flag, same as
# docs/tariff_arm.md's retirement of the bare 95_000_000 literal
# elsewhere) and capital_case stays a module constant (NDC_CASES' own
# swept subject), not a flag, matching 05_'s/15_'s convention.

# Gas regimes to compare — baseline is the primary; downside is the
# stress test where gas relief from EaaS matters most.
GAS_CASES = ["baseline", "downside", "upside", "shock_recovery"]

# NDC scenarios; both are run with and without EaaS. TASK 29: NDC_CASES no
# longer carries a "required_margin" field. It held the pre-F1 1.10/1.05
# literals, which were NOT applied to any solve (removed in task_runner_
# revival) but WERE written to this runner's CSV and summary as
# "required_margin" -- reporting 1.10/1.05 for solves that used the
# dscr_case-derived value (1.30 at dscr_central). The outputs now record the
# value actually used (REQUIRED_MARGIN_USED, resolved from --dscr-case).
REQUIRED_MARGIN_USED = None   # set in main() from --dscr-case (TASK 29)

NDC_CASES = {
    "baseline_no_policy": {"capital_case": "unconstrained",
                           "ndc_cap_scenario": None},
    "ndc2_unconditional": {"capital_case": "tight",
                           "ndc_cap_scenario": "ndc2_unconditional"},
    "ndc2_conditional":   {"capital_case": "moderate",
                           "ndc_cap_scenario": "ndc2_conditional"},
    "ndc3_unconditional": {"capital_case": "tight",
                           "ndc_cap_scenario": "ndc3_unconditional"},
    "ndc3_conditional":   {"capital_case": "tight",
                           "ndc_cap_scenario": "ndc3_conditional"},
}

def run_no_policy_gas_matrix_eaas(econ, gas_cases, cap_path, standard_kwargs):
    """
    Run gas regime matrix WITHOUT any emissions cap (baseline_no_policy).
    This is structurally different from run_gas_regime_ndc_matrix because
    it passes emissions_cap=None instead of annual caps.
    """

    # BENCHMARKS / MMBTU_PER_TWH_TH: single source, src/gas_benchmarks.py
    # (imported at module top; register L35 item 1).

    results = []

    for gas_case in gas_cases:
        scenario = load_scenario(
            demand_level_case="served",
            demand_case="organic_central",
            gas_deliverability_case=gas_case,
            capital_case="moderate",
            carbon_case="no_policy",
            start_year=2025,
            end_year=MODEL_END_YEAR,
            **standard_kwargs,
        )

        years = scenario["years"]

        # Load time-varying solar CAPEX from NREL ATB (solar_low scenario).
        # solar_low declines from $1,456k/MW (2025) to $603k/MW (2045).
        solar_capex_tv = load_solar_capex_by_year(
            scenario_name="solar_low",
            start_year=int(years[0]),
            end_year=int(years[-1]),
        )
        scenario["financing_regime"] = "eaas"
        # task_runner_revival, Part 2A: no per-case required_margin override
        # here anymore -- this was the same retired conditional/
        # unconditional double-count already fixed in 02_run_eaas.py/
        # 03_run_ndc_eaas.py/04_run_fin2_tariff_sweep.py/15_run_pol1_ndc_
        # comparison.py ("F1, Decision 2"), found here mid-revival: it
        # overwrote the correctly dscr_case-sourced required_margin
        # load_scenario() had just set two lines above. required_margin is
        # now sourced solely from load_scenario(dscr_case=...) via
        # standard_kwargs.
        # docs/tariff_arm.md, Steps 1-3: tariff threaded via
        # load_scenario(tariff_case=...) above; post-call literal retired.
        # solar_min_build_mw_per_year override to solar_min_build_default()
        # REMOVED (2.11, Step 1) -- [UNSOURCED] (plan 2.5); load_scenario()'s
        # own 0.0 default now applies (see docs/phase_2_11_minbuild_retirement.md).

        m = build_model_sliced(
            scenario=scenario,
            econ=econ,
            # Plan 0.4, item C: 1e18 sentinel -> None ("no cap", already
            # the builder's own guarded meaning for this argument).
            emissions_cap=None,
            solar_capex_by_year=solar_capex_tv,
            )
        status = solve_model(m, scenario=scenario)

        if not status["optimal"]:
            results.append({
                "gas_case": gas_case,
                "ndc_scenario": "baseline_no_policy",
                "status": "infeasible",
            })
            continue

        diag = extract_planning_diagnostics(m, scenario, econ)
        gas_shadow = diag["gas_shadow_pv_usd_per_twh_th_by_year"]
        gas_shadow_yr = diag["gas_shadow_yr_usd_per_twh_th_by_year"]
        gs_pv, gs_yr = shadow_series(diag, "gas")
        gs_vals = [v for v in gs_pv if v is not None]

        benchmark_comparison = {}
        for bench_name, bench_val in BENCHMARKS.items():
            exceeding = [v for v in gs_vals if v > bench_val]
            benchmark_comparison[bench_name] = {
                "benchmark_usd_per_twh_th": bench_val,
                "share_years_shadow_gt_benchmark": len(exceeding) / len(gs_vals) if gs_vals else None,
                "max_gap_usd_per_twh_th": float(max(v - bench_val for v in gs_vals)) if gs_vals else None,
                "mean_gap_usd_per_twh_th": float(np.mean([v - bench_val for v in gs_vals])) if gs_vals else None,
            }

        results.append({
            "gas_case":                gas_case,
            "ndc_scenario":            "baseline_no_policy",
            # task_runner_revival, Part 2A / audit A37: this was an unedited
            # copy-paste of run_no_policy_gas_matrix()'s own (correct)
            # result dict below -- "traditional"/"traditional" here, in the
            # EaaS-on function (scenario["financing_regime"] = "eaas" is set
            # above), silently forced every downstream `status == "optimal"`
            # check to fail for baseline_no_policy's EaaS arm specifically
            # (the cleanest, no-cap-confound test case), collapsing every
            # relief metric to None with no crash and no warning.
            "financing_regime":        "eaas",
            "status":                  "optimal",
            **leakage_flat(diag),
            **shadow_mean_fields("gas", gs_pv, gs_yr),
            "gas_shadow_pv_max":       float(max(gs_vals)) if gs_vals else 0.0,
            "gas_shadow_yr_max":       float(max(v for v in gs_yr if v is not None)) if gs_vals else 0.0,
            "gas_shadow_pv_by_year":   gas_shadow,
            "gas_shadow_yr_by_year":   gas_shadow_yr,
            "benchmark_comparison":    benchmark_comparison,
            "carbon_shadow_pv_all_year_mean": 0.0,
            "carbon_shadow_yr_all_year_mean": 0.0,
            "carbon_shadow_pv_binding_mean":  None,
            "carbon_shadow_yr_binding_mean":  None,
            "carbon_shadow_pv_max":    0.0,
            "carbon_shadow_yr_max":    0.0,
            "carbon_binding_years":    0,
            "npv_total_cost_usd":      float(pyo.value(m.system_cost_npv)),
            "cumulative_unserved_twh": sum(diag["unserved_twh_by_year"].values()),
            "cumulative_emissions_tco2": float(pyo.value(m.emissions)),
            "unserved_only_reliability": diag["unserved_only_reliability"],
            "grid_service_reliability_horizon": diag["grid_service_reliability_horizon"],
        })

    return results


def run_no_policy_gas_matrix(econ, gas_cases, cap_path, standard_kwargs):
    """
    Run gas regime matrix WITHOUT any emissions cap (baseline_no_policy).
    This is structurally different from run_gas_regime_ndc_matrix because
    it passes emissions_cap=None instead of annual caps.
    """

    # BENCHMARKS / MMBTU_PER_TWH_TH: single source, src/gas_benchmarks.py
    # (imported at module top; register L35 item 1).

    results = []

    for gas_case in gas_cases:
        scenario = load_scenario(
            demand_level_case="served",
            demand_case="organic_central",
            gas_deliverability_case=gas_case,
            capital_case="moderate",
            carbon_case="no_policy",
            start_year=2025,
            end_year=MODEL_END_YEAR,
            **standard_kwargs,
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
        # own 0.0 default now applies (see docs/phase_2_11_minbuild_retirement.md).

        m = build_model_sliced(
            scenario=scenario,
            econ=econ,
            # Plan 0.4, item C: 1e18 sentinel -> None ("no cap", already
            # the builder's own guarded meaning for this argument).
            emissions_cap=None,
            solar_capex_by_year=solar_capex_tv,
            )
        status = solve_model(m, scenario=scenario)

        if not status["optimal"]:
            results.append({
                "gas_case": gas_case,
                "ndc_scenario": "baseline_no_policy",
                "status": "infeasible",
            })
            continue

        diag = extract_planning_diagnostics(m, scenario, econ)
        gas_shadow = diag["gas_shadow_pv_usd_per_twh_th_by_year"]
        gas_shadow_yr = diag["gas_shadow_yr_usd_per_twh_th_by_year"]
        gs_pv, gs_yr = shadow_series(diag, "gas")
        gs_vals = [v for v in gs_pv if v is not None]

        benchmark_comparison = {}
        for bench_name, bench_val in BENCHMARKS.items():
            exceeding = [v for v in gs_vals if v > bench_val]
            benchmark_comparison[bench_name] = {
                "benchmark_usd_per_twh_th": bench_val,
                "share_years_shadow_gt_benchmark": len(exceeding) / len(gs_vals) if gs_vals else None,
                "max_gap_usd_per_twh_th": float(max(v - bench_val for v in gs_vals)) if gs_vals else None,
                "mean_gap_usd_per_twh_th": float(np.mean([v - bench_val for v in gs_vals])) if gs_vals else None,
            }

        results.append({
            "gas_case":                gas_case,
            "ndc_scenario":            "baseline_no_policy",
            "financing_regime":        "traditional",
            "status":                  "optimal",
            **leakage_flat(diag),
            **shadow_mean_fields("gas", gs_pv, gs_yr),
            "gas_shadow_pv_max":       float(max(gs_vals)) if gs_vals else 0.0,
            "gas_shadow_yr_max":       float(max(v for v in gs_yr if v is not None)) if gs_vals else 0.0,
            "gas_shadow_pv_by_year":   gas_shadow,
            "gas_shadow_yr_by_year":   gas_shadow_yr,
            "benchmark_comparison":    benchmark_comparison,
            "carbon_shadow_pv_all_year_mean": 0.0,
            "carbon_shadow_yr_all_year_mean": 0.0,
            "carbon_shadow_pv_binding_mean":  None,
            "carbon_shadow_yr_binding_mean":  None,
            "carbon_shadow_pv_max":    0.0,
            "carbon_shadow_yr_max":    0.0,
            "carbon_binding_years":    0,
            "npv_total_cost_usd":      float(pyo.value(m.system_cost_npv)),
            "cumulative_unserved_twh": sum(diag["unserved_twh_by_year"].values()),
            "cumulative_emissions_tco2": float(pyo.value(m.emissions)),
            "unserved_only_reliability": diag["unserved_only_reliability"],
            "grid_service_reliability_horizon": diag["grid_service_reliability_horizon"],
        })

    return results

CANONICAL_GENSET = "genset_central"

def parse_args():
    # task_runner_revival, Part 2A: this runner had NO argparse at all
    # before this edit -- every load_scenario()/run_gas_regime_ndc_matrix()
    # call below either sat on a hardcoded literal (tariff_case, om_case)
    # or omitted the remaining ~15 sentinel-required keys outright. Built
    # fresh, add_standard_arms(parser) for the full set, following
    # 05_run_fin3_capital_sweep.py's pattern -- EXCEPT gas_deliverability_
    # case, which this runner sweeps itself (GAS_CASES above) and must
    # therefore stay excluded from the flag set, exactly as capital_case
    # stays a module constant per NDC_CASES rather than a flag.
    parser = argparse.ArgumentParser(
        description="07_run_gas2_eaas_gas_relief: GAS-2 EaaS-vs-traditional gas-shadow relief"
    )
    add_standard_arms(parser, exclude=["gas_deliverability_case"])
    add_solar_build_args(parser)  # TASK 46: opt-in; absent = inherited default
    return parser.parse_args()


def main():

    args = parse_args()
    standard_kwargs = standard_arms_kwargs(args)
    # TASK 46: opt-in design flags. With neither given, nothing changes: no
    # solar_build_case keyword is added, so load_scenario's own default applies.
    # When given, it travels in standard_kwargs to every load_scenario() call
    # and (via shared_fn_kwargs / **standard_kwargs) to the shared function.
    global RESULTS_DIR
    RESULTS_DIR = resolve_results_dir(RESULTS_DIR, args)
    standard_kwargs.update(solar_build_kwargs(args))
    # TASK 29: the margin the solves actually use (dscr_case-derived), for the
    # output labels; NOT passed to any solve (load_scenario() sets it).
    global REQUIRED_MARGIN_USED
    REQUIRED_MARGIN_USED = dscr_scenarios()[args.dscr_case]
    # run_gas_regime_ndc_matrix() (src/optimize_experiments.py) does not
    # accept new_gas_deliverability at all (confirmed via inspect.
    # signature) -- unlike load_scenario(), which does. Stripped only for
    # the two run_gas_regime_ndc_matrix() calls below; the two internal
    # load_scenario() calls inside run_no_policy_gas_matrix[_eaas] use the
    # full standard_kwargs, unmodified.
    shared_fn_kwargs = {k: v for k, v in standard_kwargs.items() if k != "new_gas_deliverability"}

    all_rows = []

    for ndc_name, cfg in NDC_CASES.items():

        econ = load_econ(CANONICAL_VOLL, genset_case=CANONICAL_GENSET)
        ndc_cap = cfg["ndc_cap_scenario"]

        # --- Arm A: no EaaS (traditional financing)
        print(f"\n=== GAS-2: {ndc_name} × traditional (no EaaS) ===")
        if ndc_cap is None:
            no_eaas = run_no_policy_gas_matrix(
                econ=econ,
                gas_cases=GAS_CASES,
                cap_path=str(CAP_PATH),
                standard_kwargs=standard_kwargs,
            )
        else:
            no_eaas = run_gas_regime_ndc_matrix(
                econ=econ,
                ndc_cap_scenario = ndc_cap,
                gas_cases=GAS_CASES,
                builder="sliced",  # A5, Group C: was hardwired to the annual builder
                financing_regime="traditional",
                capital_case=cfg["capital_case"],
                cap_path=str(CAP_PATH),
                **shared_fn_kwargs,
            )

        # --- Arm B: EaaS active
        print(f"\n=== GAS-2: {ndc_name} × EaaS (required_margin="
              f"{REQUIRED_MARGIN_USED}, dscr_case-derived) ===")
        if ndc_cap is None:
            eaas_results = run_no_policy_gas_matrix_eaas(
                econ=econ,
                gas_cases=GAS_CASES,
                cap_path=str(CAP_PATH),
                standard_kwargs=standard_kwargs,
            )

        else:
            # required_margin=cfg["required_margin"] REMOVED here: run_gas_
            # regime_ndc_matrix() no longer accepts this parameter at all
            # (src/optimize_experiments.py's own docstring: "F1, Step 3:
            # the required_margin=1.10 kwarg this function used to carry is
            # REMOVED... dscr_case now supplies a single cushion"). Passing
            # it, as this call site still did, would raise TypeError:
            # unexpected keyword argument 'required_margin' -- a second,
            # independent execution blocker beyond the missing standard
            # arms, found and fixed in the same revival.
            eaas_results = run_gas_regime_ndc_matrix(
                econ=econ,
                ndc_cap_scenario=ndc_cap,
                gas_cases=GAS_CASES,
                builder="sliced",  # A5, Group C: was hardwired to the annual builder
                financing_regime="eaas",
                capital_case=cfg["capital_case"],
                cap_path=str(CAP_PATH),
                **shared_fn_kwargs,
            )

        # --- Build comparison rows
        no_eaas_by_gas = {r["gas_case"]: r for r in no_eaas}
        eaas_by_gas    = {r["gas_case"]: r for r in eaas_results}

        for gas_case in GAS_CASES:
            r_no  = no_eaas_by_gas.get(gas_case, {})
            r_yes = eaas_by_gas.get(gas_case, {})

            # TASK 27 A.2: relief on BOTH bases, named for what they are.
            # all_year: ratio of total rent over the (shared) horizon -- free
            # of binding-set composition. binding_mean: ratio of binding-year
            # means -- equals the all-year ratio iff the two arms bind in the
            # same NUMBER of years (true in all 8 priced pairs of the
            # 2026-10-02 run, task_log_26a), otherwise composition-confounded
            # (register L70). `binding_counts_equal` makes that visible.
            def _rel(a, b):
                ab = (a - b) if (a is not None and b is not None) else None
                pc = (ab / a * 100) if (ab is not None and a and a > 0) else None
                return ab, pc
            ok_no, ok_yes = r_no.get("status") == "optimal", r_yes.get("status") == "optimal"
            # TASK 53 2.3: relief is computed on BOTH bases (pv = the solver's year-0 USD, yr = year-t USD).
            def _g(r, ok, k):
                return r.get(k) if ok else None
            no_shadow_pv  = _g(r_no, ok_no, "gas_shadow_pv_all_year_mean");   yes_shadow_pv = _g(r_yes, ok_yes, "gas_shadow_pv_all_year_mean")
            no_shadow_yr  = _g(r_no, ok_no, "gas_shadow_yr_all_year_mean");   yes_shadow_yr = _g(r_yes, ok_yes, "gas_shadow_yr_all_year_mean")
            no_bshadow_pv = _g(r_no, ok_no, "gas_shadow_pv_binding_mean");    yes_bshadow_pv = _g(r_yes, ok_yes, "gas_shadow_pv_binding_mean")
            no_bshadow_yr = _g(r_no, ok_no, "gas_shadow_yr_binding_mean");    yes_bshadow_yr = _g(r_yes, ok_yes, "gas_shadow_yr_binding_mean")
            relief_abs_pv, relief_pct_pv = _rel(no_shadow_pv, yes_shadow_pv)
            relief_abs_yr, relief_pct_yr = _rel(no_shadow_yr, yes_shadow_yr)
            relief_abs_b_pv, relief_pct_b_pv = _rel(no_bshadow_pv, yes_bshadow_pv)
            relief_abs_b_yr, relief_pct_b_yr = _rel(no_bshadow_yr, yes_bshadow_yr)
            n_bind_no  = r_no.get("gas_shadow_binding_years")  if ok_no else None
            n_bind_yes = r_yes.get("gas_shadow_binding_years") if ok_yes else None
            counts_equal = (n_bind_no == n_bind_yes) if (n_bind_no is not None and n_bind_yes is not None) else None

            row = {
                "ndc_scenario":               ndc_name,
                "gas_case":                   gas_case,
                "required_margin":            REQUIRED_MARGIN_USED,
                # No-EaaS arm
                "no_eaas_gas_shadow_pv_all_year_mean": no_shadow_pv,
                "no_eaas_gas_shadow_yr_all_year_mean": no_shadow_yr,
                "no_eaas_gas_shadow_pv_binding_mean":  no_bshadow_pv,
                "no_eaas_gas_shadow_yr_binding_mean":  no_bshadow_yr,
                "no_eaas_binding_years":      r_no.get("gas_shadow_binding_years") if r_no.get("status") == "optimal" else None,
                "no_eaas_unserved_twh":       r_no.get("cumulative_unserved_twh")  if r_no.get("status") == "optimal" else None,
                "no_eaas_npv_cost":           r_no.get("npv_total_cost_usd")       if r_no.get("status") == "optimal" else None,
                "no_eaas_status":             r_no.get("status", "missing"),
                # EaaS arm
                "eaas_gas_shadow_pv_all_year_mean":    yes_shadow_pv,
                "eaas_gas_shadow_yr_all_year_mean":    yes_shadow_yr,
                "eaas_gas_shadow_pv_binding_mean":     yes_bshadow_pv,
                "eaas_gas_shadow_yr_binding_mean":     yes_bshadow_yr,
                "eaas_binding_years":         r_yes.get("gas_shadow_binding_years") if r_yes.get("status") == "optimal" else None,
                "eaas_unserved_twh":          r_yes.get("cumulative_unserved_twh")  if r_yes.get("status") == "optimal" else None,
                "eaas_solar_total_mw":        r_yes.get("solar_total_mw")           if r_yes.get("status") == "optimal" else None,
                "eaas_npv_cost":              r_yes.get("npv_total_cost_usd")        if r_yes.get("status") == "optimal" else None,
                "eaas_status":                r_yes.get("status", "missing"),
                # Relief metrics
                "gas_shadow_relief_abs_all_year_pv":      relief_abs_pv,
                "gas_shadow_relief_pct_all_year_pv":      relief_pct_pv,
                "gas_shadow_relief_abs_all_year_yr":      relief_abs_yr,
                "gas_shadow_relief_pct_all_year_yr":      relief_pct_yr,
                "gas_shadow_relief_abs_binding_mean_pv":  relief_abs_b_pv,
                "gas_shadow_relief_pct_binding_mean_pv":  relief_pct_b_pv,
                "gas_shadow_relief_abs_binding_mean_yr":  relief_abs_b_yr,
                "gas_shadow_relief_pct_binding_mean_yr":  relief_pct_b_yr,
                "binding_counts_equal":                counts_equal,
            }
            all_rows.append(row)

            if relief_pct_pv is not None:
                print(f"  {gas_case:<15} relief pv={relief_pct_pv:.1f}% yr={relief_pct_yr:.1f}%  "
                      f"(no_eaas pv={no_shadow_pv:.0f} -> eaas pv={yes_shadow_pv:.0f})")

    df = pd.DataFrame(all_rows)
    csv_path = RESULTS_DIR / "eaas_vs_noeaas_gas_shadow.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    summary = {}
    for ndc_name in NDC_CASES:
        sub = df[df["ndc_scenario"] == ndc_name]
        optimal = sub[
            (sub["no_eaas_status"] == "optimal") &
            (sub["eaas_status"] == "optimal")
        ]

        # Does EaaS always reduce gas shadow (monotone relief)?
        always_relieves = bool((optimal["gas_shadow_relief_abs_all_year_yr"] > 0).all()) if len(optimal) else None
        max_relief = float(optimal["gas_shadow_relief_pct_all_year_yr"].max()) if len(optimal) else None
        downside_relief = optimal[optimal["gas_case"] == "downside"]["gas_shadow_relief_pct_all_year_yr"].values
        downside_relief_val = float(downside_relief[0]) if len(downside_relief) else None

        summary[ndc_name] = {
            "required_margin":            REQUIRED_MARGIN_USED,
            "eaas_always_relieves_gas":   always_relieves,
            # TASK 27 A.2: relief percentages are on the ALL-YEAR basis
            # (ratio of total rent); the binding-mean basis is in the CSV and
            # equals this iff binding_counts_equal (see _rel() above).
            "max_relief_pct_all_year":    max_relief,
            "downside_gas_relief_pct_all_year": downside_relief_val,
            "binding_counts_equal_all_priced_pairs": bool(
                optimal[(optimal["no_eaas_binding_years"] > 0)
                        | (optimal["eaas_binding_years"] > 0)]["binding_counts_equal"].all()
            ) if len(optimal) else None,
            "interpretation": (
                "Positive relief means EaaS reduces gas scarcity rent. "
                "Relief in downside scenario is the key thesis claim: "
                "private capital provides gas relief that public-only cannot."
            ),
        }

    with open(RESULTS_DIR / "gas2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {RESULTS_DIR / 'gas2_summary.json'}")

    print("\n=== GAS-2 HEADLINE ===")
    for ndc, s in summary.items():
        print(f"  {ndc}:")
        print(f"    EaaS always relieves gas: {s['eaas_always_relieves_gas']}")
        max_relief_str = f"{s['max_relief_pct_all_year']:.1f}" if s['max_relief_pct_all_year'] is not None else 'N/A'
        downside_str = f"{s['downside_gas_relief_pct_all_year']:.1f}" if s['downside_gas_relief_pct_all_year'] is not None else 'N/A'
        print(f"    Max relief: {max_relief_str}%")
        print(f"    Downside gas relief: {downside_str}%")

if __name__ == "__main__":
    main()
