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
  - LNG export netback (~10 USD/MMBtu): domestic power use outcompetes
    exports in scarcity terms — the standard resource allocation argument
    for domestic gas reservation.

The question is whether this relationship HOLDS STRUCTURALLY across gas
regime shapes and NDC ambition levels — not just in one baseline run.

ROBUSTNESS GAP BEING CLOSED
-----------------------------
Previous phase8 runs used baseline gas only. The old scarcity_stats.csv
shows 1/21 binding years for NDC cases — suspiciously low. This is because
the baseline gas scenario is flat (zero drift), and under the NDC cap the
model substitutes solar for gas, reducing gas scarcity rent.

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

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
import pyomo.environ as pyo
from src.scenarios import load_scenario
from src.optimize_model import build_model, solve_model
from src.optimize_experiments import extract_planning_diagnostics

from src.io import (
    load_econ, load_solar_capex_by_year)
from src.optimize_experiments import run_gas_regime_ndc_matrix

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
        "capital_case":     "moderate",
    },
    "ndc3_unconditional": {
        "ndc_cap_scenario": "ndc3_unconditional",
        "capital_case":     "moderate",
    },
}


def run_no_policy_gas_matrix(econ, gas_cases, cap_path):
    """
    Run gas regime matrix WITHOUT any emissions cap (baseline_no_policy).
    This is structurally different from run_gas_regime_ndc_matrix because
    it passes emissions_cap=1e18 instead of annual caps.
    """
    
    MMBTU_PER_TWH_TH = 1e9 / 293.071
    BENCHMARKS = {
        "dom_power_2p4":      2.4  * MMBTU_PER_TWH_TH,
        "dom_commercial_2p9": 2.9  * MMBTU_PER_TWH_TH,
        "dom_high_3p3":       3.3  * MMBTU_PER_TWH_TH,
        "opp_3":              3.0  * MMBTU_PER_TWH_TH,
        "opp_5":              5.0  * MMBTU_PER_TWH_TH,
        "opp_7":              7.0  * MMBTU_PER_TWH_TH,
        "lng_export_10":     10.0  * MMBTU_PER_TWH_TH,
    }

    results = []

    for gas_case in gas_cases:
        scenario = load_scenario(
            demand_level_case="served",
            demand_case="baseline",
            gas_deliverability_case=gas_case,
            capital_case="moderate",
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
        
        m = build_model(
            scenario=scenario, 
            econ=econ, 
            emissions_cap=1e18, 
            solar_capex_by_year=solar_capex_tv,
            )
        status = solve_model(m)

        if not status["optimal"]:
            results.append({
                "gas_case": gas_case,
                "ndc_scenario": "baseline_no_policy",
                "status": "infeasible",
            })
            continue

        diag = extract_planning_diagnostics(m, scenario, econ)
        gas_shadow = diag["gas_shadow_price_usd_per_twh_th_by_year"]
        gs_vals = [v for v in gas_shadow.values() if v is not None]

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
            "gas_shadow_mean":         float(np.mean(gs_vals)) if gs_vals else 0.0,
            "gas_shadow_max":          float(max(gs_vals)) if gs_vals else 0.0,
            "gas_shadow_binding_years": sum(1 for v in gs_vals if v > 1e-6),
            "gas_shadow_by_year":      gas_shadow,
            "benchmark_comparison":    benchmark_comparison,
            "carbon_shadow_mean":      0.0,
            "carbon_shadow_max":       0.0,
            "carbon_binding_years":    0,
            "npv_total_cost_usd":      float(pyo.value(m.system_cost_npv)),
            "cumulative_unserved_twh": sum(diag["unserved_twh_by_year"].values()),
            "cumulative_emissions_tco2": float(pyo.value(m.emissions)),
            "horizon_reliability":     diag["horizon_reliability"],
        })

    return results


def main():

    econ = load_econ(CANONICAL_VOLL)
    all_results = []

    # --- Arm 1: baseline no policy (no emissions cap)
    print("\n=== GAS-1: baseline_no_policy × gas regimes ===")
    no_policy_results = run_no_policy_gas_matrix(econ, GAS_CASES, str(CAP_PATH))
    all_results.extend(no_policy_results)
    for r in no_policy_results:
        print(f"  {r['gas_case']:<15} shadow_mean={r.get('gas_shadow_mean',0):.0f}  "
              f"binding={r.get('gas_shadow_binding_years',0)}/21  [{r['status']}]")

    # --- Arm 2: ndc3_unconditional × gas regimes
    print("\n=== GAS-1: ndc3_unconditional × gas regimes ===")
    ndc_results = run_gas_regime_ndc_matrix(
        econ=econ,
        ndc_cap_scenario="ndc3_unconditional",
        gas_cases=GAS_CASES,
        financing_regime="traditional",
        capital_case="moderate",
        cap_path=str(CAP_PATH),
    )
    all_results.extend(ndc_results)
    for r in ndc_results:
        print(f"  {r['gas_case']:<15} shadow_mean={r.get('gas_shadow_mean',0):.0f}  "
              f"binding={r.get('gas_shadow_binding_years',0)}/21  [{r['status']}]")

    # ── Save gas shadow matrix ─────────────────────────────────────────────
    matrix_rows = []
    for r in all_results:
        if r["status"] != "optimal":
            continue
        matrix_rows.append({
            "gas_case":                   r["gas_case"],
            "ndc_scenario":               r["ndc_scenario"],
            "gas_shadow_mean":            r["gas_shadow_mean"],
            "gas_shadow_max":             r["gas_shadow_max"],
            "gas_shadow_binding_years":   r["gas_shadow_binding_years"],
            "carbon_shadow_mean":         r.get("carbon_shadow_mean", 0),
            "carbon_binding_years":       r.get("carbon_binding_years", 0),
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
        for year, shadow in r.get("gas_shadow_by_year", {}).items():
            ts_rows.append({
                "gas_case":     r["gas_case"],
                "ndc_scenario": r["ndc_scenario"],
                "year":         year,
                "gas_shadow_usd_per_twh_th": shadow,
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
            "gas_shadow_mean": r["gas_shadow_mean"],
            "gas_shadow_binding_years": r["gas_shadow_binding_years"],
            "exceeds_dom_high_3p3": bench.get("dom_high_3p3", {}).get("share_years_shadow_gt_benchmark"),
            "exceeds_opp_7": bench.get("opp_7", {}).get("share_years_shadow_gt_benchmark"),
            "exceeds_lng_10": bench.get("lng_export_10", {}).get("share_years_shadow_gt_benchmark"),
        }

    with open(RESULTS_DIR / "gas1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to: {RESULTS_DIR}")
    print("\n=== GAS-1 HEADLINE ===")
    print(f"{'Case':<45} {'Mean shadow':>12} {'> opp_7':>8} {'> LNG_10':>9}")
    print("-" * 78)
    for key, s in summary.items():
        ndc, gas = key.split("__")
        print(f"  {ndc[:20]:<20} {gas:<20} "
              f"{s['gas_shadow_mean']:>12.0f} "
              f"{str(s['exceeds_opp_7']):>8} "
              f"{str(s['exceeds_lng_10']):>9}")


if __name__ == "__main__":
    main()
