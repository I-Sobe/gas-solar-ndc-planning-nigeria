"""
09b_run_rel1_retirement_sensitivity.py  —  REL-1 Robustness Check
==================================================================

PURPOSE
-------
Tests whether the REL-1 finding (gas regime is the primary determinant
of reliability feasibility) is sensitive to the assumed gas fleet
retirement rate.

The base model assumes 680 MW/yr retirement starting 2035 (~13,600/20).
This script sweeps:
    0 MW/yr       — fleet extended indefinitely (upper bound on gas capacity)
    340 MW/yr     — half the default rate
    680 MW/yr     — model default
    1,360 MW/yr   — double rate (aggressive decommissioning)

For each retirement rate, runs the REL-1 feasibility matrix across
all four gas regimes under the aggressive solar build case only
(the case that gives solar the maximum chance to compensate).

PRE-REQUISITES
--------------
    01_run_baseline.py  — baseline results

OUTPUTS
-------
    results/rel1_retirement/retirement_sensitivity.csv
    results/rel1_retirement/retirement_summary.json

RUN SEQUENCE
------------
    python scripts/01_run_baseline.py
    python scripts/09b_run_rel1_retirement_sensitivity.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io import load_econ, load_solar_capex_by_year
from src.scenarios import load_scenario
from src.optimize_model import build_model, solve_model
from src.optimize_experiments import extract_planning_diagnostics

# ============================================================
# CONFIG
# ============================================================

RESULTS_DIR = ROOT / "results" / "rel1_retirement"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_VOLL = "voll_mid"
CAPITAL_CASE   = "moderate"

RETIREMENT_CASES = {
    "no_retirement":   0.0,
    "slow":            340.0,
    "default":         680.0,
    "fast":            1360.0,
}

GAS_CASES  = ["baseline", "upside", "downside", "shock_recovery"]
EPS_LEVELS = [0.20, 0.15, 0.10, 0.08, 0.06, 0.05]

# Use aggressive solar only — gives solar the best chance to compensate.
# If the threshold still shifts with gas regime under aggressive solar,
# gas regime dominance is robust to retirement rate.
SOLAR_BUILD_CASE = "aggressive"


# ============================================================
# MAIN
# ============================================================

def main():
    econ = load_econ(CANONICAL_VOLL)

    total_solves = (
        len(RETIREMENT_CASES)
        * len(GAS_CASES)
        * (1 + len(EPS_LEVELS))  # +1 for VoLL-only baseline
    )

    print(f"\nREL-1 Retirement Sensitivity")
    print(f"  retirement_rates:  {list(RETIREMENT_CASES.values())} MW/yr")
    print(f"  gas_cases:         {GAS_CASES}")
    print(f"  solar_build_case:  {SOLAR_BUILD_CASE}")
    print(f"  eps_levels:        {EPS_LEVELS}")
    print(f"  total solves:      {total_solves}")

    all_rows = []

    for ret_label, ret_rate in RETIREMENT_CASES.items():
        for gas_case in GAS_CASES:

            scenario = load_scenario(
                demand_level_case="served",
                demand_case="baseline",
                gas_deliverability_case=gas_case,
                capital_case=CAPITAL_CASE,
                solar_build_case=SOLAR_BUILD_CASE,
                carbon_case="no_policy",
                start_year=2025,
                end_year=2045,
            )

            # Override retirement rate
            scenario["gas_baseline_retirement_mw_per_year"] = ret_rate

            years = scenario["years"]
            solar_capex_tv = load_solar_capex_by_year(
                scenario_name="solar_low",
                start_year=int(years[0]),
                end_year=int(years[-1]),
            )
            scenario["solar_min_build_mw_per_year"] = 100.0

            # ---- VoLL-only baseline (no reliability constraint) ----
            try:
                m_base = build_model(
                    scenario=scenario,
                    econ=econ,
                    reliability_max_unserved_fraction=None,
                    solar_capex_by_year=solar_capex_tv,
                )
                status = solve_model(m_base)
                if not status["optimal"]:
                    raise RuntimeError("non-optimal")

                diag = extract_planning_diagnostics(m_base, scenario, econ)
                all_rows.append({
                    "retirement_label":    ret_label,
                    "retirement_mw_yr":    ret_rate,
                    "gas_case":            gas_case,
                    "eps":                 None,
                    "eps_label":           "voll_only",
                    "status":              "feasible",
                    "npv_cost":            float(pyo.value(m_base.system_cost_npv)),
                    "unserved_total_twh":  sum(diag["unserved_twh_by_year"].values()),
                    "horizon_reliability": diag["horizon_reliability"],
                    "solar_total_mw":      sum(
                        float(pyo.value(m_base.solar_public_add[t]))
                        + float(pyo.value(m_base.solar_eaas_add[t]))
                        for t in range(len(years))
                    ),
                })
            except Exception as e:
                all_rows.append({
                    "retirement_label":    ret_label,
                    "retirement_mw_yr":    ret_rate,
                    "gas_case":            gas_case,
                    "eps":                 None,
                    "eps_label":           "voll_only",
                    "status":              f"error: {e}",
                    "npv_cost":            None,
                    "unserved_total_twh":  None,
                    "horizon_reliability": None,
                    "solar_total_mw":      None,
                })

            # ---- Reliability-constrained sweep ----
            for eps in EPS_LEVELS:
                try:
                    m = build_model(
                        scenario=scenario,
                        econ=econ,
                        reliability_max_unserved_fraction=eps,
                        reliability_mode="total",
                        solar_capex_by_year=solar_capex_tv,
                    )
                    status = solve_model(m)
                    if not status["optimal"]:
                        raise RuntimeError("non-optimal")

                    diag = extract_planning_diagnostics(m, scenario, econ)
                    all_rows.append({
                        "retirement_label":    ret_label,
                        "retirement_mw_yr":    ret_rate,
                        "gas_case":            gas_case,
                        "eps":                 eps,
                        "eps_label":           f"eps_{eps:.2f}",
                        "status":              "feasible",
                        "npv_cost":            float(pyo.value(m.system_cost_npv)),
                        "unserved_total_twh":  sum(diag["unserved_twh_by_year"].values()),
                        "horizon_reliability": diag["horizon_reliability"],
                        "solar_total_mw":      sum(
                            float(pyo.value(m.solar_public_add[t]))
                            + float(pyo.value(m.solar_eaas_add[t]))
                            for t in range(len(years))
                        ),
                    })

                except RuntimeError:
                    all_rows.append({
                        "retirement_label":    ret_label,
                        "retirement_mw_yr":    ret_rate,
                        "gas_case":            gas_case,
                        "eps":                 eps,
                        "eps_label":           f"eps_{eps:.2f}",
                        "status":              "infeasible",
                        "npv_cost":            None,
                        "unserved_total_twh":  None,
                        "horizon_reliability": None,
                        "solar_total_mw":      None,
                    })

            print(f"  Done: retirement={ret_label}, gas={gas_case}")

    # ── Save full results ──────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "retirement_sensitivity.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR / 'retirement_sensitivity.csv'}")

    # ── Compute feasibility threshold per (retirement, gas_case) ──
    feasible_df = df[(df["status"] == "feasible") & (df["eps"].notna())].copy()

    threshold_rows = []
    for (ret_label, gas_case), grp in feasible_df.groupby(
        ["retirement_label", "gas_case"]
    ):
        # Tightest feasible eps = smallest eps that is still feasible
        tightest_eps = grp["eps"].min()
        cost_at_threshold = grp.loc[
            grp["eps"] == tightest_eps, "npv_cost"
        ].values[0]

        threshold_rows.append({
            "retirement_label":      ret_label,
            "retirement_mw_yr":      RETIREMENT_CASES[ret_label],
            "gas_case":              gas_case,
            "tightest_feasible_eps": tightest_eps,
            "npv_cost_at_threshold": cost_at_threshold,
        })

    # Add fully-infeasible cases (no eps was feasible)
    for ret_label in RETIREMENT_CASES:
        for gas_case in GAS_CASES:
            if not any(
                r["retirement_label"] == ret_label and r["gas_case"] == gas_case
                for r in threshold_rows
            ):
                threshold_rows.append({
                    "retirement_label":      ret_label,
                    "retirement_mw_yr":      RETIREMENT_CASES[ret_label],
                    "gas_case":              gas_case,
                    "tightest_feasible_eps": None,
                    "npv_cost_at_threshold": None,
                })

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(RESULTS_DIR / "retirement_threshold.csv", index=False)

    # ── Summary: does the REL-1 finding hold across retirement rates? ──
    summary = {
        "experiment": "REL-1 retirement rate sensitivity",
        "retirement_rates_mw_yr": list(RETIREMENT_CASES.values()),
        "solar_build_case": SOLAR_BUILD_CASE,
        "threshold_by_retirement_and_gas": {},
    }

    for ret_label in RETIREMENT_CASES:
        sub = threshold_df[threshold_df["retirement_label"] == ret_label]
        thresholds = sub["tightest_feasible_eps"].dropna()
        gas_range = (
            float(thresholds.max() - thresholds.min())
            if len(thresholds) >= 2 else None
        )
        summary["threshold_by_retirement_and_gas"][ret_label] = {
            "thresholds_by_gas": {
                row["gas_case"]: row["tightest_feasible_eps"]
                for _, row in sub.iterrows()
            },
            "threshold_range_across_gas": gas_range,
        }

    # Overall: is gas range consistent across retirement rates?
    ranges = [
        v["threshold_range_across_gas"]
        for v in summary["threshold_by_retirement_and_gas"].values()
        if v["threshold_range_across_gas"] is not None
    ]
    summary["gas_regime_drives_threshold_across_all_retirement_rates"] = (
        all(r > 0 for r in ranges) if ranges else None
    )
    summary["interpretation"] = (
        "The gas regime feasibility threshold range is positive under all "
        "retirement rate assumptions. The REL-1 finding (gas regime is the "
        "primary determinant of reliability feasibility) is robust to "
        "retirement rate uncertainty."
        if summary["gas_regime_drives_threshold_across_all_retirement_rates"]
        else
        "The gas regime threshold range collapses under some retirement "
        "rate assumptions. The REL-1 finding may be partially driven by "
        "retirement rate, not purely by gas deliverability."
    )

    with open(RESULTS_DIR / "retirement_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Console output ──────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  REL-1 RETIREMENT SENSITIVITY: FEASIBILITY THRESHOLDS")
    print(f"{'='*65}")
    print(f"  {'Retirement':>14}  {'Gas case':<18}  {'Threshold eps':>14}")
    print(f"  {'-'*50}")
    for _, row in threshold_df.sort_values(
        ["retirement_mw_yr", "gas_case"]
    ).iterrows():
        thresh = (
            f"{row['tightest_feasible_eps']:.2f}"
            if pd.notna(row["tightest_feasible_eps"])
            else "ALL INFEASIBLE"
        )
        print(f"  {row['retirement_label']:>14}  {row['gas_case']:<18}  {thresh:>14}")

    print(f"\n  Gas regime dominates across all retirement rates: "
          f"{summary['gas_regime_drives_threshold_across_all_retirement_rates']}")
    print(f"  {summary['interpretation']}")
    print(f"\nSaved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()