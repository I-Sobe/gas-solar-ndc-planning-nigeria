"""
15_run_pol1_ndc_comparison.py  —  POL-1
========================================

RESEARCH QUESTION (POL-1)
--------------------------
Does NDC 3.0's absolute emission reduction target (29% by 2030, 32% by 2035)
require a materially different investment trajectory than the NDC 2.0
BAU-relative targets — and does EaaS change the answer?

THE FUNDAMENTAL DIFFERENCE BETWEEN NDC 2.0 AND NDC 3.0
-------------------------------------------------------
NDC 2.0 (BAU-relative):
  cap_2030 = multiplier × model_baseline_gas_emissions_2030
  Unconditional: 80% of BAU (20% below BAU)
  Conditional:   53% of BAU (47% below BAU)
  Gas continues at reduced but non-zero level throughout the horizon.
  The cap is calibrated to the model's own projected emissions trajectory.

NDC 3.0 (absolute economy-wide):
  Power sector share = economy_reduction × electricity_generation_share (13.95%)
  Unconditional: 114.7 MtCO2e economy-wide × 13.95% = ~16 MtCO2 power abatement
  Conditional:   168.2 MtCO2e economy-wide × 13.95% = ~23.5 MtCO2 power abatement
  BUT: model baseline power emissions = ~5.4 MtCO2/yr
  => Required abatement >> available emissions => cap floors at ZERO
  => Gas-fired power must cease almost entirely from 2030 onwards

THIS IS A QUALITATIVE REGIME SHIFT — not a quantitative adjustment:
  NDC 2.0: gas at 80% of BAU (significant gas use continues)
  NDC 3.0: gas at ~0% from 2030 (power sector must be solar-dominant)

INVESTMENT TRAJECTORY IMPLICATIONS
------------------------------------
NDC 2.0: Moderate solar build, gas shadow price limited, EaaS convenient.
NDC 3.0: Aggressive solar build required, feasibility may fail under public
          capital alone, EaaS transitions from convenient to necessary.

Under NDC 3.0, the PUBLIC CAPITAL budget (moderate = 5.21B USD) may be
insufficient to finance the solar deployment needed to:
  (a) replace near-zero gas with solar for power supply, AND
  (b) maintain acceptable energy access (low unserved energy)
  simultaneously within the annual NDC cap.

EaaS sub-question: Does EaaS rescue NDC 3.0 compliance where public
capital alone fails?

EXPERIMENT STRUCTURE
--------------------
  NDC versions: [ndc2_unconditional, ndc2_conditional,
                 ndc3_unconditional, ndc3_conditional]
  Financing arms: [public_only, eaas]
  Total solves: 4 × 2 = 8 LP solves

MATCHED AMBITION COMPARISON
----------------------------
The comparison at matched ambition levels:
  ndc2_unconditional vs ndc3_unconditional  (both "unconditional" commitment)
  ndc2_conditional   vs ndc3_conditional    (both "conditional" commitment)

OUTPUTS
--------
  results/pol1/pol1_results.csv          — all 8 runs
  results/pol1/trajectory_comparison.csv — solar_built, cost, carbon_shadow per run
  results/pol1/ndc_version_diff.csv      — NDC3 - NDC2 trajectory difference
  results/pol1/pol1_summary.json         — headline findings

PRE-REQUISITE
-------------
Run 00_build_emissions_cap.py first — this version now outputs all four
NDC scenarios (ndc2_unconditional, ndc2_conditional, ndc3_unconditional,
ndc3_conditional) in a single emissions_cap.csv.
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

RESULTS_DIR    = ROOT / "results" / "pol1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAP_PATH       = ROOT / "data" / "cost" / "processed" / "emissions_cap.csv"
CANONICAL_VOLL = "voll_mid"
CANONICAL_TARIFF = 95_000_000   # USD/TWh

# NDC version configurations — matched ambition pairs for comparison
NDC_CONFIGS = [
    {
        "ndc_label":     "ndc2_unconditional",
        "ndc_version":   "NDC 2.0",
        "ambition_level": "unconditional",
        "public_capital": "moderate",
        "eaas_capital":   "moderate",
        "eaas_margin":    1.10,
    },
    {
        "ndc_label":     "ndc2_conditional",
        "ndc_version":   "NDC 2.0",
        "ambition_level": "conditional",
        "public_capital": "expansion",
        "eaas_capital":   "expansion",
        "eaas_margin":    1.05,
    },
    {
        "ndc_label":     "ndc3_unconditional",
        "ndc_version":   "NDC 3.0",
        "ambition_level": "unconditional",
        "public_capital": "moderate",
        "eaas_capital":   "moderate",
        "eaas_margin":    1.10,
    },
    {
        "ndc_label":     "ndc3_conditional",
        "ndc_version":   "NDC 3.0",
        "ambition_level": "conditional",
        "public_capital": "expansion",
        "eaas_capital":   "expansion",
        "eaas_margin":    1.05,
    },
]

FINANCING_CONFIGS = [
    {
        "label":            "public_only",
        "financing_regime": "traditional",
    },
    {
        "label":            "eaas",
        "financing_regime": "eaas",
    },
]


# ============================================================
# HELPERS
# ============================================================

def load_annual_caps(scenario_name, years_list):
    cap_df = pd.read_csv(CAP_PATH)
    sub = cap_df[
        (cap_df["scenario"] == scenario_name)
        & (cap_df["year"].isin(years_list))
    ].sort_values("year")
    if len(sub) != len(years_list):
        raise ValueError(
            f"Cap length {len(sub)} != {len(years_list)} for '{scenario_name}'. "
            f"Run 00_build_emissions_cap.py first."
        )
    return sub["cap_tco2"].astype(float).tolist()


def run_one(ndc_cfg, fin_cfg, econ):
    """Solve one (NDC version, financing arm) combination."""
    ndc_label = ndc_cfg["ndc_label"]
    fin_label  = fin_cfg["label"]
    capital    = (
        ndc_cfg["eaas_capital"]
        if fin_label == "eaas"
        else ndc_cfg["public_capital"]
    )

    scenario = load_scenario(
        demand_level_case="served",
        demand_case="baseline",
        gas_deliverability_case="baseline",
        capital_case=capital,
        solar_build_case="aggressive",
        carbon_case="no_policy",
        start_year=2025,
        end_year=2045,
    )

    scenario["financing_regime"] = fin_cfg["financing_regime"]
    if fin_label == "eaas":
        scenario["required_margin"] = ndc_cfg["eaas_margin"]
        scenario["solar_service_tariff_usd_per_twh"] = CANONICAL_TARIFF

    years_list = list(range(2025, 2046))
    caps = load_annual_caps(ndc_label, years_list)
    T    = range(len(scenario["years"]))
    # Load time-varying solar CAPEX from NREL ATB (solar_low scenario).
    # solar_low declines from $1,456k/MW (2025) to $603k/MW (2045).
    solar_capex_tv = load_solar_capex_by_year(
        scenario_name="solar_low",
        start_year=int(years_list[0]),
        end_year=int(years_list[-1]),
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
    except RuntimeError as e:
        return {
            "ndc_label":      ndc_label,
            "ndc_version":    ndc_cfg["ndc_version"],
            "ambition_level": ndc_cfg["ambition_level"],
            "financing_arm":  fin_label,
            "status":         f"infeasible: {e}",
        }

    diag = extract_planning_diagnostics(m, scenario, econ)

    cs_vals = [
        v for v in diag["carbon_shadow_price_usd_per_tco2_by_year"].values()
        if v is not None
    ]
    gs_vals = [
        v for v in diag["gas_shadow_price_usd_per_twh_th_by_year"].values()
        if v is not None
    ]

    solar_public_mw = sum(float(pyo.value(m.solar_public_add[t])) for t in T)
    solar_eaas_mw   = sum(float(pyo.value(m.solar_eaas_add[t]))   for t in T)

    # Year-by-year investment trajectory (solar additions per year)
    solar_add_by_year = {
        int(scenario["years"][t]):
            float(pyo.value(m.solar_public_add[t]))
            + float(pyo.value(m.solar_eaas_add[t]))
        for t in T
    }

    return {
        "ndc_label":              ndc_label,
        "ndc_version":            ndc_cfg["ndc_version"],
        "ambition_level":         ndc_cfg["ambition_level"],
        "financing_arm":          fin_label,
        "status":                 "optimal",
        "npv_total_cost_usd":     float(pyo.value(m.system_cost_npv)),
        "cumulative_unserved_twh": sum(diag["unserved_twh_by_year"].values()),
        "cumulative_emissions_tco2": float(pyo.value(m.emissions)),
        "horizon_reliability":    diag["horizon_reliability"],
        "solar_total_mw":         solar_public_mw + solar_eaas_mw,
        "solar_public_total_mw":  solar_public_mw,
        "solar_eaas_total_mw":    solar_eaas_mw,
        "storage_final_mwh":      float(pyo.value(m.storage_capacity_mwh[len(list(T))-1])),
        "carbon_shadow_mean":     float(np.mean(cs_vals)) if cs_vals else 0.0,
        "carbon_shadow_max":      float(max(cs_vals))     if cs_vals else 0.0,
        "carbon_binding_years":   sum(1 for v in cs_vals if v > 1e-6),
        "gas_shadow_mean":        float(np.mean(gs_vals)) if gs_vals else 0.0,
        "gas_binding_years":      sum(1 for v in gs_vals if v > 1e-6),
        "budget_shadow":          diag.get("public_budget_shadow_usd_per_usd"),
        "budget_utilisation":     diag.get("public_budget_utilisation"),
        # Annual trajectory for plotting — serialised as JSON string
        "solar_add_by_year_json": json.dumps(solar_add_by_year),
        "cap_2030_tco2":          caps[5],   # index 5 = year 2030 (2025+5)
        "capital_case":           capital,
    }


# ============================================================
# ANALYSIS HELPERS
# ============================================================

def compute_trajectory_diff(df):
    """
    For each (ambition_level, financing_arm), compute:
      trajectory_diff = NDC3 metric - NDC2 metric

    Positive diff on solar_total_mw: NDC 3.0 requires MORE solar
    Positive diff on npv_cost: NDC 3.0 is MORE expensive
    Positive diff on carbon_binding_years: NDC 3.0 cap is tighter
    """
    rows = []
    for (ambition, fin), grp in df.groupby(["ambition_level", "financing_arm"]):
        ndc2 = grp[grp["ndc_version"] == "NDC 2.0"]
        ndc3 = grp[grp["ndc_version"] == "NDC 3.0"]

        if not len(ndc2) or not len(ndc3):
            continue

        r2 = ndc2.iloc[0]
        r3 = ndc3.iloc[0]

        both_optimal = (r2["status"] == "optimal" and r3["status"] == "optimal")

        rows.append({
            "ambition_level":       ambition,
            "financing_arm":        fin,
            "ndc2_label":           r2["ndc_label"],
            "ndc3_label":           r3["ndc_label"],
            "ndc2_status":          r2["status"],
            "ndc3_status":          r3["status"],
            "ndc2_cap_2030_tco2":   r2.get("cap_2030_tco2"),
            "ndc3_cap_2030_tco2":   r3.get("cap_2030_tco2"),
            "cap_tightening_tco2": (
                float(r2["cap_2030_tco2"] - r3["cap_2030_tco2"])
                if both_optimal else None
            ),
            # Solar trajectory difference
            "diff_solar_total_mw": (
                float(r3["solar_total_mw"] - r2["solar_total_mw"])
                if both_optimal else None
            ),
            "diff_solar_eaas_mw": (
                float(r3["solar_eaas_total_mw"] - r2["solar_eaas_total_mw"])
                if both_optimal else None
            ),
            # Cost difference
            "diff_npv_cost_usd": (
                float(r3["npv_total_cost_usd"] - r2["npv_total_cost_usd"])
                if both_optimal else None
            ),
            "diff_npv_cost_b_usd": (
                float(r3["npv_total_cost_usd"] - r2["npv_total_cost_usd"]) / 1e9
                if both_optimal else None
            ),
            # Access difference
            "diff_unserved_twh": (
                float(r3["cumulative_unserved_twh"] - r2["cumulative_unserved_twh"])
                if both_optimal else None
            ),
            # Constraint pressure
            "diff_carbon_binding_years": (
                int(r3["carbon_binding_years"] - r2["carbon_binding_years"])
                if both_optimal else None
            ),
            # Is NDC 3.0 infeasible where NDC 2.0 is feasible?
            "ndc3_infeasible_ndc2_feasible": (
                r2["status"] == "optimal" and r3["status"] != "optimal"
            ),
            # Does EaaS rescue NDC 3.0 infeasibility?
            "interpretation": None,   # filled below
        })

    result_df = pd.DataFrame(rows)

    # Add interpretation
    for i, row in result_df.iterrows():
        if row["ndc3_infeasible_ndc2_feasible"]:
            interp = (
                "NDC 3.0 is INFEASIBLE under this financing arm. "
                "The absolute target cannot be met with the current capital envelope. "
            )
        elif row["diff_solar_total_mw"] is not None and row["diff_solar_total_mw"] > 500:
            interp = (
                f"NDC 3.0 requires {row['diff_solar_total_mw']:.0f} MW more solar "
                f"than NDC 2.0 — a materially different investment trajectory."
            )
        elif row["diff_solar_total_mw"] is not None:
            interp = (
                f"NDC 3.0 and NDC 2.0 require similar solar deployment "
                f"(difference: {row['diff_solar_total_mw']:.0f} MW). "
                "Investment trajectories are not materially different."
            )
        else:
            interp = "Cannot compare — missing data."
        result_df.at[i, "interpretation"] = interp

    return result_df


def assess_eaas_rescue(df, diff_df):
    """
    For each ambition level, check whether EaaS rescues NDC 3.0 compliance
    that is infeasible under public-only financing.

    EaaS is NECESSARY if:
      public_only arm: NDC 3.0 infeasible
      eaas arm:        NDC 3.0 feasible
    """
    rows = []
    for ambition in df["ambition_level"].unique():
        pub_ndc3 = df[
            (df["ambition_level"] == ambition)
            & (df["financing_arm"] == "public_only")
            & (df["ndc_version"] == "NDC 3.0")
        ]
        eaas_ndc3 = df[
            (df["ambition_level"] == ambition)
            & (df["financing_arm"] == "eaas")
            & (df["ndc_version"] == "NDC 3.0")
        ]

        pub_feasible  = (
            len(pub_ndc3) > 0 and pub_ndc3["status"].values[0] == "optimal"
        )
        eaas_feasible = (
            len(eaas_ndc3) > 0 and eaas_ndc3["status"].values[0] == "optimal"
        )

        eaas_necessary = not pub_feasible and eaas_feasible

        # Also check cost improvement when both feasible
        cost_improvement = None
        if pub_feasible and eaas_feasible:
            cost_improvement = float(
                pub_ndc3["npv_total_cost_usd"].values[0]
                - eaas_ndc3["npv_total_cost_usd"].values[0]
            )

        rows.append({
            "ambition_level":           ambition,
            "ndc3_pub_status":          pub_ndc3["status"].values[0] if len(pub_ndc3) else "missing",
            "ndc3_eaas_status":         eaas_ndc3["status"].values[0] if len(eaas_ndc3) else "missing",
            "eaas_necessary_for_ndc3":  eaas_necessary,
            "eaas_cost_improvement_usd": cost_improvement,
            "interpretation": (
                f"EaaS is NECESSARY for NDC 3.0 {ambition} compliance — "
                "public capital alone cannot achieve the target."
                if eaas_necessary
                else (
                    f"EaaS saves {cost_improvement/1e9:.2f}B USD vs public-only "
                    f"under NDC 3.0 {ambition}."
                    if cost_improvement is not None
                    else f"NDC 3.0 {ambition} not feasible under either arm."
                    if not pub_feasible and not eaas_feasible
                    else "EaaS not needed for feasibility but may reduce cost."
                )
            ),
        })
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    econ = load_econ(CANONICAL_VOLL)

    print(f"\nPOL-1: NDC 2.0 vs NDC 3.0 investment trajectory comparison")
    print(f"  NDC scenarios: {[c['ndc_label'] for c in NDC_CONFIGS]}")
    print(f"  financing_arms: {[f['label'] for f in FINANCING_CONFIGS]}")
    print(f"  total solves: {len(NDC_CONFIGS) * len(FINANCING_CONFIGS)}")
    print()

    # ── Check cap CSV has all four scenarios ───────────────────────────────
    cap_df = pd.read_csv(CAP_PATH)
    available = cap_df["scenario"].unique().tolist()
    required  = [c["ndc_label"] for c in NDC_CONFIGS]
    missing   = [s for s in required if s not in available]
    if missing:
        raise FileNotFoundError(
            f"Missing NDC scenarios in emissions_cap.csv: {missing}\n"
            f"Run 00_build_emissions_cap.py first (this version outputs all four).\n"
            f"Available: {available}"
        )

    # ── Run all combinations ───────────────────────────────────────────────
    rows = []
    for ndc_cfg in NDC_CONFIGS:
        for fin_cfg in FINANCING_CONFIGS:
            label = f"{ndc_cfg['ndc_label']} × {fin_cfg['label']}"
            print(f"  Running: {label}")
            result = run_one(ndc_cfg, fin_cfg, econ)
            rows.append(result)
            status_str = result["status"]
            if status_str == "optimal":
                print(
                    f"    cost={result['npv_total_cost_usd']/1e9:.2f}B  "
                    f"solar={result['solar_total_mw']:.0f}MW  "
                    f"carbon_bind={result['carbon_binding_years']}/21  "
                    f"[{status_str}]"
                )
            else:
                print(f"    [{status_str}]")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "pol1_results.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR / 'pol1_results.csv'}")

    # ── Trajectory comparison ──────────────────────────────────────────────
    optimal = df[df["status"] == "optimal"].copy()

    # Trajectory table — key metrics per run
    traj_cols = [
        "ndc_label", "ndc_version", "ambition_level", "financing_arm",
        "npv_total_cost_usd", "solar_total_mw", "solar_eaas_total_mw",
        "solar_public_total_mw", "storage_final_mwh",
        "cumulative_unserved_twh", "horizon_reliability",
        "carbon_binding_years", "carbon_shadow_mean",
        "gas_binding_years", "cap_2030_tco2", "status",
    ]
    traj_df = df[[c for c in traj_cols if c in df.columns]].copy()
    traj_df.to_csv(RESULTS_DIR / "trajectory_comparison.csv", index=False)

    # NDC version diff
    diff_df = compute_trajectory_diff(df)
    diff_df.to_csv(RESULTS_DIR / "ndc_version_diff.csv", index=False)

    # EaaS rescue assessment
    eaas_df = assess_eaas_rescue(df, diff_df)
    eaas_df.to_csv(RESULTS_DIR / "eaas_rescue.csv", index=False)

    # ── Summary JSON ──────────────────────────────────────────────────────
    summary = {
        "experiment": {
            "ndc_scenarios":    [c["ndc_label"] for c in NDC_CONFIGS],
            "financing_arms":   [f["label"] for f in FINANCING_CONFIGS],
            "voll":             CANONICAL_VOLL,
        },
        "cap_comparison": {},
        "trajectory_findings": {},
        "eaas_findings": {},
    }

    # Cap level comparison (what the two frameworks actually impose)
    for cfg in NDC_CONFIGS:
        label = cfg["ndc_label"]
        sub = cap_df[cap_df["scenario"] == label]
        if len(sub):
            sub = sub.sort_values("year")
            summary["cap_comparison"][label] = {
                "ndc_version":    cfg["ndc_version"],
                "ambition_level": cfg["ambition_level"],
                "cap_2025_tco2":  float(sub[sub["year"] == 2025]["cap_tco2"].values[0]),
                "cap_2030_tco2":  float(sub[sub["year"] == 2030]["cap_tco2"].values[0]),
                "cap_2035_tco2":  float(sub[sub["year"] == 2035]["cap_tco2"].values[0]),
                "cap_2045_tco2":  float(sub[sub["year"] == 2045]["cap_tco2"].values[0]),
                "effectively_zero_gas_from_2030": float(
                    sub[sub["year"] == 2030]["cap_tco2"].values[0]
                ) < 1e5,   # < 100 tCO2 = effectively zero
            }

    # Trajectory findings
    for _, row in diff_df.iterrows():
        key = f"{row['ambition_level']}__{row['financing_arm']}"
        summary["trajectory_findings"][key] = {
            "ambition_level":      row["ambition_level"],
            "financing_arm":       row["financing_arm"],
            "cap_tightening_tco2": row.get("cap_tightening_tco2"),
            "additional_solar_mw": row.get("diff_solar_total_mw"),
            "additional_cost_b_usd": row.get("diff_npv_cost_b_usd"),
            "ndc3_infeasible":     row.get("ndc3_infeasible_ndc2_feasible"),
            "interpretation":      row.get("interpretation"),
        }

    # EaaS findings
    for _, row in eaas_df.iterrows():
        summary["eaas_findings"][row["ambition_level"]] = {
            "eaas_necessary_for_ndc3": row["eaas_necessary_for_ndc3"],
            "eaas_cost_improvement_b_usd": (
                row["eaas_cost_improvement_usd"] / 1e9
                if row["eaas_cost_improvement_usd"] is not None else None
            ),
            "interpretation": row["interpretation"],
        }

    with open(RESULTS_DIR / "pol1_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {RESULTS_DIR / 'pol1_summary.json'}")

    # ── Console output ─────────────────────────────────────────────────────
    print("\n=== POL-1: CAP LEVELS ===")
    print(f"  {'NDC scenario':<22} {'2025 cap':>11} {'2030 cap':>11} "
          f"{'2045 cap':>11} {'Near-zero?':>11}")
    print("  " + "-" * 68)
    for label, caps_info in summary["cap_comparison"].items():
        c25 = f"{caps_info['cap_2025_tco2']/1e6:.2f}M"
        c30 = f"{caps_info['cap_2030_tco2']/1e6:.2f}M"
        c45 = f"{caps_info['cap_2045_tco2']/1e6:.2f}M"
        nz  = "YES" if caps_info["effectively_zero_gas_from_2030"] else "no"
        print(f"  {label:<22} {c25:>11} {c30:>11} {c45:>11} {nz:>11}")

    print("\n=== POL-1: TRAJECTORY DIFFERENCE (NDC3 - NDC2) ===")
    print(f"  {'Ambition':<16} {'Arm':<12} {'Cap tight (tCO2)':>18} "
          f"{'ΔSolar (MW)':>12} {'ΔCost (B$)':>11} {'NDC3 infeas':>12}")
    print("  " + "-" * 76)
    for _, row in diff_df.iterrows():
        ct  = f"{row['cap_tightening_tco2']:.0f}" if pd.notna(row.get("cap_tightening_tco2")) else "—"
        ds  = f"{row['diff_solar_total_mw']:+.0f}" if pd.notna(row.get("diff_solar_total_mw")) else "—"
        dc  = f"{row['diff_npv_cost_b_usd']:+.2f}" if pd.notna(row.get("diff_npv_cost_b_usd")) else "—"
        inf = "YES" if row["ndc3_infeasible_ndc2_feasible"] else "no"
        print(f"  {row['ambition_level']:<16} {row['financing_arm']:<12} "
              f"{ct:>18} {ds:>12} {dc:>11} {inf:>12}")

    print("\n=== POL-1: EAAS ASSESSMENT ===")
    for _, row in eaas_df.iterrows():
        print(f"\n  {row['ambition_level']}:")
        print(f"    EaaS necessary for NDC 3.0: {row['eaas_necessary_for_ndc3']}")
        print(f"    {row['interpretation']}")


if __name__ == "__main__":
    main()
