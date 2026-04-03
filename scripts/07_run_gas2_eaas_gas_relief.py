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

The conditional arm (ndc3_conditional + EaaS, required_margin=1.05) is
the most policy-relevant: it asks whether concessional-finance-enabled EaaS
can relieve the gas constraint under Nigeria's most ambitious NDC target.

OUTPUTS
--------
  results/gas2/eaas_vs_noeaas_gas_shadow.csv  — shadow comparison per case
  results/gas2/gas2_summary.json              — relief magnitude and direction
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
import pyomo.environ as pyo
from src.scenarios import load_scenario
from src.optimize_model import build_model, solve_model
from src.optimize_experiments import extract_planning_diagnostics

from src.io import (
    load_econ, load_solar_capex_by_year)
from src.io import load_econ
from src.optimize_experiments import run_gas_regime_ndc_matrix

# ============================================================
# CONFIG
# ============================================================

RESULTS_DIR = ROOT / "results" / "gas2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAP_PATH = ROOT / "data" / "cost" / "processed" / "emissions_cap.csv"
CANONICAL_VOLL = "voll_mid"

# Gas regimes to compare — baseline is the primary; downside is the
# stress test where gas relief from EaaS matters most.
GAS_CASES = ["baseline", "downside", "upside", "shock_recovery"]

# NDC scenarios: unconditional uses commercial finance (rm=1.10),
# conditional uses concessional (rm=1.05). Both are run with and without EaaS.
NDC_CASES = {
    "baseline_no_policy": {"required_margin": 1.10, "capital_case": "moderate",
                           "ndc_cap_scenario": None},
    "ndc3_unconditional": {"required_margin": 1.10, "capital_case": "moderate",
                           "ndc_cap_scenario": "ndc3_unconditional"},
    "ndc3_conditional":   {"required_margin": 1.05, "capital_case": "expansion",
                           "ndc_cap_scenario": "ndc3_conditional"},
}

def run_no_policy_gas_matrix_eaas(econ, gas_cases, required_margin, cap_path):
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
        scenario["financing_regime"] = "eaas"
        scenario["required_margin"] = required_margin
        scenario["solar_service_tariff_usd_per_twh"] = 95_000_000
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
            "status":                  "traditional",
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

    all_rows = []

    for ndc_name, cfg in NDC_CASES.items():

        econ = load_econ(CANONICAL_VOLL)
        ndc_cap = cfg["ndc_cap_scenario"]

        # --- Arm A: no EaaS (traditional financing)
        print(f"\n=== GAS-2: {ndc_name} × traditional (no EaaS) ===")
        if ndc_cap is None:
            no_eaas = run_no_policy_gas_matrix(
                econ=econ,
                gas_cases=GAS_CASES,
                cap_path=str(CAP_PATH),
            )
        else:
            no_eaas = run_gas_regime_ndc_matrix(
                econ=econ,
                ndc_cap_scenario = ndc_cap,
                gas_cases=GAS_CASES,
                financing_regime="traditional",
                capital_case=cfg["capital_case"],
                cap_path=str(CAP_PATH),
            )

        # --- Arm B: EaaS active
        print(f"\n=== GAS-2: {ndc_name} × EaaS (rm={cfg['required_margin']}) ===")
        if ndc_cap is None:
            eaas_results = run_no_policy_gas_matrix_eaas(
                econ=econ,
                gas_cases=GAS_CASES,
                required_margin=cfg["required_margin"],
                cap_path=str(CAP_PATH),
            )

        else:
            eaas_results = run_gas_regime_ndc_matrix(
                econ=econ,
                ndc_cap_scenario=ndc_cap,
                gas_cases=GAS_CASES,
                financing_regime="eaas",
                required_margin=cfg["required_margin"],
                capital_case=cfg["capital_case"],
                cap_path=str(CAP_PATH),
            )

        # --- Build comparison rows
        no_eaas_by_gas = {r["gas_case"]: r for r in no_eaas}
        eaas_by_gas    = {r["gas_case"]: r for r in eaas_results}

        for gas_case in GAS_CASES:
            r_no  = no_eaas_by_gas.get(gas_case, {})
            r_yes = eaas_by_gas.get(gas_case, {})

            no_shadow  = r_no.get("gas_shadow_mean",  None) if r_no.get("status") == "optimal" else None
            yes_shadow = r_yes.get("gas_shadow_mean", None) if r_yes.get("status") == "optimal" else None

            relief_abs = (no_shadow - yes_shadow) if (no_shadow is not None and yes_shadow is not None) else None
            relief_pct = (relief_abs / no_shadow * 100) if (relief_abs is not None and no_shadow and no_shadow > 0) else None

            row = {
                "ndc_scenario":               ndc_name,
                "gas_case":                   gas_case,
                "required_margin":            cfg["required_margin"],
                # No-EaaS arm
                "no_eaas_gas_shadow_mean":    no_shadow,
                "no_eaas_binding_years":      r_no.get("gas_shadow_binding_years") if r_no.get("status") == "optimal" else None,
                "no_eaas_unserved_twh":       r_no.get("cumulative_unserved_twh")  if r_no.get("status") == "optimal" else None,
                "no_eaas_npv_cost":           r_no.get("npv_total_cost_usd")       if r_no.get("status") == "optimal" else None,
                "no_eaas_status":             r_no.get("status", "missing"),
                # EaaS arm
                "eaas_gas_shadow_mean":       yes_shadow,
                "eaas_binding_years":         r_yes.get("gas_shadow_binding_years") if r_yes.get("status") == "optimal" else None,
                "eaas_unserved_twh":          r_yes.get("cumulative_unserved_twh")  if r_yes.get("status") == "optimal" else None,
                "eaas_solar_total_mw":        r_yes.get("solar_total_mw")           if r_yes.get("status") == "optimal" else None,
                "eaas_npv_cost":              r_yes.get("npv_total_cost_usd")        if r_yes.get("status") == "optimal" else None,
                "eaas_status":                r_yes.get("status", "missing"),
                # Relief metrics
                "gas_shadow_relief_abs":      relief_abs,
                "gas_shadow_relief_pct":      relief_pct,
            }
            all_rows.append(row)

            if relief_pct is not None:
                print(f"  {gas_case:<15} relief={relief_pct:.1f}%  "
                      f"(no_eaas={no_shadow:.0f} -> eaas={yes_shadow:.0f})")

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
        always_relieves = bool((optimal["gas_shadow_relief_abs"] > 0).all()) if len(optimal) else None
        max_relief = float(optimal["gas_shadow_relief_pct"].max()) if len(optimal) else None
        downside_relief = optimal[optimal["gas_case"] == "downside"]["gas_shadow_relief_pct"].values
        downside_relief_val = float(downside_relief[0]) if len(downside_relief) else None

        summary[ndc_name] = {
            "required_margin":            NDC_CASES[ndc_name]["required_margin"],
            "eaas_always_relieves_gas":   always_relieves,
            "max_relief_pct":             max_relief,
            "downside_gas_relief_pct":    downside_relief_val,
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
        max_relief_str = f"{s['max_relief_pct']:.1f}" if s['max_relief_pct'] is not None else 'N/A'
        downside_str = f"{s['downside_gas_relief_pct']:.1f}" if s['downside_gas_relief_pct'] is not None else 'N/A'
        print(f"    Max relief: {max_relief_str}%")
        print(f"    Downside gas relief: {downside_str}%")

if __name__ == "__main__":
    main()
