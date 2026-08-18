"""
09_run_rel1_feasibility.py  —  REL-1
=====================================

RESEARCH QUESTION (REL-1)
--------------------------
Under a fixed capital envelope and solar build rate, does the feasibility
of minimum reliability standards depend primarily on gas deliverability
regime rather than on the level of solar investment?

WHAT THIS ANSWERS
-----------------
The question has a specific logical structure. It is not asking which
scenario achieves the best reliability — it is asking what DETERMINES
whether a given reliability standard is achievable at all.

The experiment holds capital and solar rate fixed, then asks:
  (a) Does the feasibility threshold shift across gas regimes?
      (gas regime = primary determinant hypothesis)
  (b) Does the feasibility threshold shift across solar build rates?
      (solar investment = alternative determinant hypothesis)

If (a) is large and (b) is small, the primary determinant is gas regime.
If both are large, both matter and the claim must be qualified.
If (b) is large and (a) is small, solar investment is the driver —
  this would falsify the primary hypothesis and is a valid finding.

EXPERIMENT DESIGN
-----------------
Fixed:
  capital_case = "moderate"  
  This is the baseline public capital assumption used throughout.
  It constrains solar deployment without being so tight that the
  system is trivially infeasible under all gas regimes.

Varied (gas — primary dimension):
  baseline, upside, downside, shock_recovery

Varied (solar deployment capability — control dimension):
  deployment_unconstrained  — no cap; maximum solar flexibility
  nirp_trajectory           — NIRP 2024 cumulative capacity (61.9 GW by 2045)

Reliability epsilon grid:
  [0.20, 0.15, 0.10, 0.08, 0.06, 0.05]
  — VoLL-only baseline also recorded (eps=None)

Total solves: 4 gas × 2 solar × 7 points = 56 LP solves

WHY UNCONSTRAINED vs NIRP TRAJECTORY
-------------------------------------
The comparison must span conditions where solar deployment COULD plausibly
matter. The previous arms (500 vs 2000 MW/yr flat) were two unsourced
numbers; both were also wrong in shape, permitting more early build than
NIRP judges achievable and less late build.

The replacement spans a SOURCED national deployment trajectory against no
constraint at all. NIRP's cumulative path binds this model from 2027 to 2037
(shortfall peaking at 4,505 MW in 2032), so the arms are genuinely distinct.
If the feasibility threshold is unchanged across them, the claim that gas
regime rather than solar deployment is the primary determinant is supported
against a real deployment constraint rather than an invented one.

CAVEAT: under nirp_trajectory the model cannot serve demand through the 2030s
and the shortfall is priced at VoLL. Read FEASIBILITY and UNSERVED ENERGY from
that arm, not NPV, until the genset backstop lands (plan 2.6).

OUTPUTS
--------
  results/rel1/feasibility_matrix.csv    — all (gas, solar, eps) rows
  results/rel1/feasibility_threshold.csv — summary: threshold per (gas, solar)
  results/rel1/rel1_summary.json         — headline finding + interpretation
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io import load_econ
from src.optimize_experiments import (
    run_rel1_feasibility_matrix,
    compute_feasibility_threshold,
)

# ============================================================
# CONFIG
# ============================================================

RESULTS_DIR = ROOT / "results" / "rel1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_VOLL  = "voll_mid"
CAPITAL_CASE    = "moderate"

GAS_CASES        = ["baseline", "upside", "downside", "shock_recovery"]
SOLAR_BUILD_CASES = ["deployment_unconstrained", "nirp_trajectory"]

# Eps grid — ordered from loose to tight.
# run_rel1_feasibility_matrix will use reliability_levels_log() by default,
# but explicit here for transparency and reproducibility.
EPS_LEVELS = [0.20, 0.15, 0.10, 0.08, 0.06, 0.05]


# ============================================================
# MAIN
# ============================================================

def main():

    econ = load_econ(CANONICAL_VOLL)

    print(f"\nREL-1: Reliability feasibility matrix")
    print(f"  capital_case:      {CAPITAL_CASE}")
    print(f"  gas_cases:         {GAS_CASES}")
    print(f"  solar_build_cases: {SOLAR_BUILD_CASES}")
    print(f"  eps_levels:        {EPS_LEVELS}")
    print(f"  total solves:      {len(GAS_CASES) * len(SOLAR_BUILD_CASES) * (1 + len(EPS_LEVELS))}")

    rows = run_rel1_feasibility_matrix(
        econ=econ,
        gas_cases=GAS_CASES,
        solar_build_cases=SOLAR_BUILD_CASES,
        capital_case=CAPITAL_CASE,
        eps_levels=EPS_LEVELS,
    )

    # ── Save full matrix ───────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    matrix_path = RESULTS_DIR / "feasibility_matrix.csv"
    df.to_csv(matrix_path, index=False)
    print(f"\nSaved: {matrix_path}")

    # ── Compute and save feasibility thresholds ────────────────────────────
    threshold_df = compute_feasibility_threshold(rows)
    threshold_path = RESULTS_DIR / "feasibility_threshold.csv"
    threshold_df.to_csv(threshold_path, index=False)
    print(f"Saved: {threshold_path}")

    # ── Build summary ──────────────────────────────────────────────────────
    # Variation across gas regimes (within each solar arm)
    # Variation across solar arms (within each gas regime)
    # If gas variation >> solar variation => gas is primary determinant

    threshold_df["threshold_numeric"] = threshold_df["feasibility_threshold_eps"].astype(float)

    gas_variation = {}
    for solar_case in SOLAR_BUILD_CASES:
        sub = threshold_df[threshold_df["solar_build_case"] == solar_case]
        thresholds = sub["threshold_numeric"].dropna()
        gas_variation[solar_case] = {
            "min_threshold": float(thresholds.min()) if len(thresholds) else None,
            "max_threshold": float(thresholds.max()) if len(thresholds) else None,
            "range":         float(thresholds.max() - thresholds.min()) if len(thresholds) else None,
            "n_infeasible":  int(sub["fully_infeasible"].sum()),
        }

    solar_variation = {}
    for gas_case in GAS_CASES:
        sub = threshold_df[threshold_df["gas_case"] == gas_case]
        solar_variation[gas_case] = {
            "unconstrained_threshold": (
                sub[sub["solar_build_case"] == "deployment_unconstrained"]["feasibility_threshold_eps"].values[0]
                if len(sub[sub["solar_build_case"] == "deployment_unconstrained"]) else None
            ),
            "nirp_threshold": (
                sub[sub["solar_build_case"] == "nirp_trajectory"]["feasibility_threshold_eps"].values[0]
                if len(sub[sub["solar_build_case"] == "nirp_trajectory"]) else None
            ),
            "solar_arm_difference": (
                float(
                    sub[sub["solar_build_case"] == "deployment_unconstrained"]["threshold_numeric"].values[0]
                    - sub[sub["solar_build_case"] == "nirp_trajectory"]["threshold_numeric"].values[0]
                )
                if (
                    len(sub[sub["solar_build_case"] == "deployment_unconstrained"]) and
                    len(sub[sub["solar_build_case"] == "nirp_trajectory"]) and
                    sub["threshold_numeric"].notna().all()
                )
                else None
            ),
        }

    # Primary finding: does gas regime shift threshold more than solar level?
    max_gas_range   = max(
        (v["range"] for v in gas_variation.values() if v["range"] is not None),
        default=None
    )
    max_solar_diff  = max(
        (abs(v["solar_arm_difference"]) for v in solar_variation.values()
         if v["solar_arm_difference"] is not None),
        default=None
    )

    if max_gas_range is not None and max_solar_diff is not None:
        if max_gas_range > max_solar_diff:
            primary_determinant = "gas_regime"
            interpretation = (
                f"Gas regime shifts the feasibility threshold by up to {max_gas_range:.2f} "
                f"(in unserved fraction units), while varying solar deployment capability shifts it by "
                f"at most {max_solar_diff:.2f}. Gas deliverability is the primary determinant."
            )
        elif max_solar_diff > max_gas_range:
            primary_determinant = "solar_investment"
            interpretation = (
                f"Solar deployment capability shifts the feasibility threshold by up to {max_solar_diff:.2f}, "
                f"exceeding the gas regime effect of {max_gas_range:.2f}. "
                f"Solar deployment capability is the primary determinant — the hypothesis is not supported."
            )
        else:
            primary_determinant = "indeterminate"
            interpretation = "Gas regime and solar deployment capability shift the threshold by equal amounts."
    else:
        primary_determinant = "indeterminate"
        interpretation = "Insufficient feasible solutions to determine primary driver."

    summary = {
        "experiment": {
            "capital_case":      CAPITAL_CASE,
            "gas_cases":         GAS_CASES,
            "solar_build_cases": SOLAR_BUILD_CASES,
            "eps_levels":        EPS_LEVELS,
            "voll":              CANONICAL_VOLL,
        },
        "primary_determinant":   primary_determinant,
        "interpretation":        interpretation,
        "max_gas_range_eps":     max_gas_range,
        "max_solar_diff_eps":    max_solar_diff,
        "gas_variation_by_solar_arm":  gas_variation,
        "solar_variation_by_gas_case": solar_variation,
        "threshold_table": threshold_df[[
            "gas_case", "solar_build_case",
            "feasibility_threshold_eps", "fully_infeasible",
            "npv_cost_at_threshold",
        ]].to_dict(orient="records"),
    }

    json_path = RESULTS_DIR / "rel1_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {json_path}")

    # ── Console output ─────────────────────────────────────────────────────
    print("\n=== REL-1 FEASIBILITY THRESHOLD TABLE ===")
    print(f"  Fixed: capital_case={CAPITAL_CASE}")
    print()
    print(f"  {'Gas case':<18} {'Solar build':<14} {'Threshold eps':>14} {'Infeasible?':>12}")
    print("  " + "-" * 62)
    for _, row in threshold_df.sort_values(
        ["solar_build_case", "gas_case"]
    ).iterrows():
        thresh = (
            f"{row['feasibility_threshold_eps']:.2f}"
            if pd.notna(row["feasibility_threshold_eps"])
            else "NONE"
        )
        infeas = "YES" if row["fully_infeasible"] else "no"
        print(f"  {row['gas_case']:<18} {row['solar_build_case']:<14} {thresh:>14} {infeas:>12}")

    print()
    print(f"=== PRIMARY FINDING ===")
    print(f"  Primary determinant: {primary_determinant.upper()}")
    print(f"  {interpretation}")
    print()
    print(f"  Max threshold range across gas regimes: {max_gas_range}")
    print(f"  Max threshold shift across solar arms:  {max_solar_diff}")


if __name__ == "__main__":
    main()
