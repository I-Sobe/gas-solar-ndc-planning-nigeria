"""
06_run_gas1_shadow_benchmarks.py  —  GAS-1
==========================================

RESEARCH QUESTION (GAS-1)
--------------------------
What is the shadow price of gas deliverability to Nigeria's power sector
across planning scenarios, and does it structurally exceed the opportunity
cost of gas exports and domestic regulated benchmarks?

WHAT THIS ANSWERS
-----------------
The shadow price on gas_balance[t] is the marginal value of an additional
TWh_th of gas delivered to the power sector — i.e., what the system would
pay for one more unit of fuel. If this exceeds:
  - Domestic regulated prices (2.4–3.3 USD/MMBtu): gas is underpriced for
    power and misallocated relative to its system value.
  - LNG export NETBACK (lng_netback_{des,hub}_{low,central,high} in
    src/gas_benchmarks.py: delivered price less liquefaction, shipping and,
    on the HUB basis, regas; delivered anchor 10.05 = SOURCED (interim),
    Japan landed LNG spot Jan 2024, DES basis): domestic power use
    outcompetes exports in scarcity terms — the standard resource
    allocation argument for domestic gas reservation. The ~10 USD/MMBtu
    `lng_export_10` is a DELIVERED price, not a netback, and is retained
    only for reference/continuity; it is not comparable to a wellhead
    shadow price.

The question is whether this relationship HOLDS STRUCTURALLY across gas
regime shapes and NDC ambition levels — not just in one baseline run.

ROBUSTNESS GAP BEING CLOSED
-----------------------------
Previous phase8 runs used baseline gas only. The old scarcity_stats.csv
recorded 1/21 binding years for NDC cases and read as suspiciously low; the
horizon is 30 years, so the denominator should have been /30 -- the author's
own "corrected by the author" note here only fixed the no-policy arm's
console print (to a bare `/30`), not the NDC arm's below, which kept
printing a bare `/21` literal until TASK 20 Part B
(`docs/task_log_20b_binding_denominator.md`) found it, confirmed the true
denominator is `len(gs_vals)` (evaluable years, not a fixed horizon count --
same reasoning as `eaas_bankability_binding_years()`'s own
n_binding_years/n_evaluable_years pair, `src/optimize_experiments.py:679`),
and made both arms report it dynamically rather than guessing a literal
either way. The low count is not an anomaly: under an NDC cap the emissions cap binds
before gas deliverability does, so the gas shadow price is zero in those
years (F-9 shows the same from the objective side: gas deliverability is
worth exactly zero under a cap). Under the baseline gas scenario (flat, zero
drift) the model additionally substitutes solar for gas, reducing scarcity
rent.

To answer GAS-1 properly, the shadow price must be computed across all
four gas regime shapes so we can distinguish:
  (a) whether scarcity is driven by the gas regime (supply-side)
  (b) whether it is driven by the NDC cap (demand-side squeeze on gas)
  (c) whether the gas scarcity value systematically exceeds export benchmarks

NDC3.0 NOTE
-----------
GAS-1 uses ndc3_unconditional (the tighter cap) as the primary policy case.
The comparison arm is baseline_no_policy to isolate the cap effect on shadow
prices. The old NDC2.0 names (ndc_unconditional_20) are no longer used.

OUTPUTS
--------
  results/gas1/gas_shadow_matrix.csv     — shadow stats per (gas_case, ndc_scenario)
  results/gas1/benchmark_comparison.csv  — share of years shadow > each benchmark
  results/gas1/gas_shadow_by_year.csv    — full timeseries of gas shadow per case
  results/gas1/gas1_summary.json         — headline findings
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
import pyomo.environ as pyo
from src.cli_arms import add_solar_build_args, solar_build_kwargs, resolve_results_dir  # TASK 46
from src.scenarios import load_scenario, asset_lifetime_sweep, solar_min_build_default, MODEL_END_YEAR
from src.optimize_model import solve_model
from src.optimize_model_sliced import build_model_sliced
from src.optimize_experiments import extract_planning_diagnostics, shadow_mean_fields, shadow_series, leakage_flat

from src.io import (
    load_econ, load_solar_capex_by_year)
from src.optimize_experiments import run_gas_regime_ndc_matrix
from src.cli_arms import add_standard_arms, standard_arms_kwargs
from src.gas_benchmarks import BENCHMARKS, MMBTU_PER_TWH_TH

# ============================================================
# CONFIG
# ============================================================

RESULTS_DIR = ROOT / "results" / "gas1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAP_PATH = ROOT / "data" / "cost" / "processed" / "emissions_cap.csv"
CANONICAL_VOLL = "voll_mid"

GAS_CASES = ["baseline", "upside", "downside", "shock_recovery"]

# Two policy cases: unconstrained baseline and NDC3.0 unconditional
# Baseline has no emissions cap — gas shadow is pure scarcity rent.
# NDC cap adds a second demand-side squeeze — shadow reflects both.
NDC_CASES = {
    "baseline_no_policy": {
        "ndc_cap_scenario": None,           # no cap — handled separately below
        "capital_case":     "unconstrained",
    },
    "ndc2_unconditional": {
        "ndc_cap_scenario": "ndc2_unconditional",
        "capital_case":     "tight",
    },
    "ndc3_unconditional": {
        "ndc_cap_scenario": "ndc3_unconditional",
        "capital_case":     "tight",
    },
}


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

        # Activate minimum build floor when time-varying CAPEX is in use.
        # This prevents, the optimizer from delaying all solar to the cheapest years
        # (2040-2045) creating unrealistic 2025-2030 supply gaps.
        scenario["solar_min_build_mw_per_year"] = solar_min_build_default()
        
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
        # TASK 53 2.3 (was G8-3): the benchmarks are year-t USD per TWh_th, so the comparison
        # is made on the YEAR-t series; the present-value series is kept beside it.
        gs_vals = [v for v in gs_yr if v is not None]

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
            # TASK 27 A.2: all-year mean, binding-year mean and counts as
            # separate named fields (src.optimize_experiments.shadow_mean_fields);
            # the former single `gas_shadow_mean` was an all-year mean.
            **shadow_mean_fields("gas", gs_pv, gs_yr),
            "gas_shadow_pv_max":       float(max(v for v in gs_pv if v is not None)) if gs_vals else 0.0,
            "gas_shadow_yr_max":       float(max(gs_vals)) if gs_vals else 0.0,
            "gas_shadow_pv_by_year":   gas_shadow,
            "gas_shadow_yr_by_year":   gas_shadow_yr,
            "benchmark_comparison":    benchmark_comparison,
            "carbon_shadow_pv_all_year_mean":  0.0,
            "carbon_shadow_yr_all_year_mean":  0.0,
            "carbon_shadow_pv_binding_mean":   None,
            "carbon_shadow_yr_binding_mean":   None,
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
    # before this edit -- its load_scenario() call and its run_gas_regime_
    # ndc_matrix() call both omitted the standard arm set outright. Built
    # fresh, add_standard_arms(parser) for the full set, following 05_run_
    # fin3_capital_sweep.py's pattern -- EXCEPT gas_deliverability_case,
    # which this runner sweeps itself (GAS_CASES above) and must therefore
    # stay excluded, matching 07_run_gas2_eaas_gas_relief.py's identical
    # revival.
    parser = argparse.ArgumentParser(
        description="06_run_gas1_shadow_benchmarks: GAS-1 gas-shadow-price vs benchmark comparison"
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
    # run_gas_regime_ndc_matrix() does not accept new_gas_deliverability at
    # all (confirmed via inspect.signature) -- unlike load_scenario(),
    # which does. Stripped only for that call below.
    shared_fn_kwargs = {k: v for k, v in standard_kwargs.items() if k != "new_gas_deliverability"}

    econ = load_econ(CANONICAL_VOLL, genset_case=CANONICAL_GENSET)
    all_results = []

    # --- Arm 1: baseline no policy (no emissions cap)
    print("\n=== GAS-1: baseline_no_policy × gas regimes ===")
    no_policy_results = run_no_policy_gas_matrix(econ, GAS_CASES, str(CAP_PATH), standard_kwargs)
    all_results.extend(no_policy_results)
    for r in no_policy_results:
        print(f"  {r['gas_case']:<15} yr all_year_mean={r.get('gas_shadow_yr_all_year_mean',0):.0f}  "
              f"yr binding_mean={(r.get('gas_shadow_yr_binding_mean') or 0):.0f} (pv {(r.get('gas_shadow_pv_binding_mean') or 0):.0f})  "
              f"binding={r.get('gas_shadow_binding_years',0)}/{r.get('gas_shadow_evaluable_years','?')} evaluable yrs  [{r['status']}]")

    # --- Arm 2: ndc3_unconditional × gas regimes
    print("\n=== GAS-1: ndc3_unconditional × gas regimes ===")
    ndc_results = run_gas_regime_ndc_matrix(
        econ=econ,
        ndc_cap_scenario="ndc3_unconditional",
        gas_cases=GAS_CASES,
        builder="sliced",  # A5, Group C: was hardwired to the annual builder
        financing_regime="traditional",
        capital_case="moderate",
        cap_path=str(CAP_PATH),
        **shared_fn_kwargs,
    )
    all_results.extend(ndc_results)
    for r in ndc_results:
        print(f"  {r['gas_case']:<15} yr all_year_mean={r.get('gas_shadow_yr_all_year_mean',0):.0f}  "
              f"yr binding_mean={(r.get('gas_shadow_yr_binding_mean') or 0):.0f} (pv {(r.get('gas_shadow_pv_binding_mean') or 0):.0f})  "
              f"binding={r.get('gas_shadow_binding_years',0)}/{r.get('gas_shadow_evaluable_years','?')} evaluable yrs  [{r['status']}]")

    # ── Save gas shadow matrix ─────────────────────────────────────────────
    matrix_rows = []
    for r in all_results:
        if r["status"] != "optimal":
            continue
        matrix_rows.append({
            "gas_case":                   r["gas_case"],
            "ndc_scenario":               r["ndc_scenario"],
            "gas_shadow_pv_all_year_mean":   r["gas_shadow_pv_all_year_mean"],
            "gas_shadow_yr_all_year_mean":   r["gas_shadow_yr_all_year_mean"],
            "gas_shadow_pv_binding_mean":    r["gas_shadow_pv_binding_mean"],
            "gas_shadow_yr_binding_mean":    r["gas_shadow_yr_binding_mean"],
            "gas_shadow_pv_max":             r["gas_shadow_pv_max"],
            "gas_shadow_yr_max":             r["gas_shadow_yr_max"],
            "gas_shadow_binding_years":   r["gas_shadow_binding_years"],
            "gas_shadow_evaluable_years": r.get("gas_shadow_evaluable_years"),
            "carbon_shadow_pv_all_year_mean": r.get("carbon_shadow_pv_all_year_mean", 0),
            "carbon_shadow_yr_all_year_mean": r.get("carbon_shadow_yr_all_year_mean", 0),
            "carbon_shadow_pv_binding_mean":  r.get("carbon_shadow_pv_binding_mean"),
            "carbon_shadow_yr_binding_mean":  r.get("carbon_shadow_yr_binding_mean"),
            "carbon_binding_years":       r.get("carbon_binding_years", 0),
            "carbon_shadow_evaluable_years": r.get("carbon_shadow_evaluable_years"),
            "npv_total_cost_usd":         r.get("npv_total_cost_usd"),
            "cumulative_unserved_twh":    r.get("cumulative_unserved_twh"),
        })

    pd.DataFrame(matrix_rows).to_csv(
        RESULTS_DIR / "gas_shadow_matrix.csv", index=False
    )

    # ── Save benchmark comparison ──────────────────────────────────────────
    bench_rows = []
    for r in all_results:
        if r["status"] != "optimal":
            continue
        for bench_name, bench_data in r.get("benchmark_comparison", {}).items():
            bench_rows.append({
                "gas_case":       r["gas_case"],
                "ndc_scenario":   r["ndc_scenario"],
                "benchmark":      bench_name,
                **bench_data,
            })

    pd.DataFrame(bench_rows).to_csv(
        RESULTS_DIR / "benchmark_comparison.csv", index=False
    )

    # ── Save full timeseries of gas shadow by year ─────────────────────────
    ts_rows = []
    for r in all_results:
        if r["status"] != "optimal":
            continue
        for year, shadow in r.get("gas_shadow_pv_by_year", {}).items():
            ts_rows.append({
                "gas_case":     r["gas_case"],
                "ndc_scenario": r["ndc_scenario"],
                "year":         year,
                "gas_shadow_pv_usd_per_twh_th": shadow,
                "gas_shadow_yr_usd_per_twh_th": r["gas_shadow_yr_by_year"][year],
            })

    pd.DataFrame(ts_rows).to_csv(
        RESULTS_DIR / "gas_shadow_by_year.csv", index=False
    )

    # ── Summary: does shadow structurally exceed benchmarks? ──────────────
    # Key thesis claim: gas shadow > export benchmarks in majority of years
    # under baseline_no_policy (pure scarcity). Under NDC cap, solar
    # substitution reduces gas shadow — so the claim should be strongest
    # in baseline.
    summary = {}
    for r in all_results:
        if r["status"] != "optimal":
            continue
        key = f"{r['ndc_scenario']}__{r['gas_case']}"
        bench = r.get("benchmark_comparison", {})
        summary[key] = {
            "gas_shadow_pv_all_year_mean": r["gas_shadow_pv_all_year_mean"],
            "gas_shadow_yr_all_year_mean": r["gas_shadow_yr_all_year_mean"],
            "gas_shadow_pv_binding_mean": r["gas_shadow_pv_binding_mean"],
            "gas_shadow_yr_binding_mean": r["gas_shadow_yr_binding_mean"],
            "gas_shadow_binding_years": r["gas_shadow_binding_years"],
            "exceeds_dom_high_3p3": bench.get("dom_high_3p3", {}).get("share_years_shadow_gt_benchmark"),
            "exceeds_opp_7": bench.get("opp_7", {}).get("share_years_shadow_gt_benchmark"),
            "exceeds_lng_10": bench.get("lng_export_10", {}).get("share_years_shadow_gt_benchmark"),
            # exceeds_lng_10 is against a DELIVERED price (not comparable);
            # the netbacks are the comparable exceedances. Both are kept so
            # the liquefaction/shipping/regas wedge is visible.
            "exceeds_lng_netback_des_low": bench.get("lng_netback_des_low", {}).get("share_years_shadow_gt_benchmark"),
            "exceeds_lng_netback_des_central": bench.get("lng_netback_des_central", {}).get("share_years_shadow_gt_benchmark"),
            "exceeds_lng_netback_des_high": bench.get("lng_netback_des_high", {}).get("share_years_shadow_gt_benchmark"),
            "exceeds_lng_netback_hub_low": bench.get("lng_netback_hub_low", {}).get("share_years_shadow_gt_benchmark"),
            "exceeds_lng_netback_hub_central": bench.get("lng_netback_hub_central", {}).get("share_years_shadow_gt_benchmark"),
            "exceeds_lng_netback_hub_high": bench.get("lng_netback_hub_high", {}).get("share_years_shadow_gt_benchmark"),
        }

    with open(RESULTS_DIR / "gas1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to: {RESULTS_DIR}")
    print("\n=== GAS-1 HEADLINE ===")
    print(f"{'Case':<45} {'Bind-yr mean':>12} {'> opp_7':>8} {'> LNG_10':>9} "
          f"{'> DES_cen':>10} {'> HUB_cen':>10}   (full DES/HUB low/central/high in gas1_summary.json)")
    print("-" * 120)
    for key, s in summary.items():
        ndc, gas = key.split("__")
        print(f"  {ndc[:20]:<20} {gas:<20} "
              f"{(s['gas_shadow_yr_binding_mean'] or 0):>12.0f} "
              f"{str(s['exceeds_opp_7']):>8} "
              f"{str(s['exceeds_lng_10']):>9} "
              f"{str(s['exceeds_lng_netback_des_central']):>10} "
              f"{str(s['exceeds_lng_netback_hub_central']):>10}")


if __name__ == "__main__":
    main()
