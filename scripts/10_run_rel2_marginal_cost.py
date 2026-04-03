"""
10_run_rel2_marginal_cost.py  —  REL-2
=======================================

RESEARCH QUESTION (REL-2)
--------------------------
What is the marginal cost of tightening the reliability standard
(dual_reliability), and how does this cost curve change across gas
deliverability regimes?

WHAT THE DUAL MEANS
-------------------
The reliability constraint (total mode) is:

    sum(unserved[t] for t in T) <= eps * sum(demand[t] for t in T)

where eps is the maximum allowed unserved fraction (e.g. 0.10 = 10%).

In a minimisation LP with a <= constraint, the dual is non-positive.
The sign is negated to get the shadow price as a positive number:

    dual_usd_per_twh = -dual_pyomo

This is the marginal system cost reduction per additional TWh of
unserved energy ALLOWED (i.e., per unit of RHS relaxation).

Equivalently: the marginal cost of TIGHTENING the reliability standard
by 1 TWh (requiring 1 TWh less unserved) = dual_usd_per_twh.

The NORMALISED form is more interpretable for the thesis:

    dual_usd_per_pct_pt = dual_usd_per_twh * total_demand_twh / 100

This is the marginal cost of improving reliability by 1 percentage point
(e.g. from 90% to 91% served). Units: USD NPV per pp.

WHAT THE COST CURVE LOOKS LIKE
-------------------------------
As eps tightens from 0.20 → 0.05:
  - At loose eps levels, the constraint may be non-binding (slack).
    The system achieves better reliability than required WITHOUT the
    constraint forcing it. Dual = 0, slack > 0.
  - Once eps crosses below the system's natural reliability level,
    the constraint becomes binding. Dual > 0, slack ≈ 0.
  - As eps tightens further, the dual rises — each additional pp of
    reliability requires increasingly expensive measures (more solar,
    storage, or accepting lower utilisation of gas plants).

The REL-2 thesis claim is that this curve SHIFTS LEFT across gas regimes:
  - Under downside gas, the constraint becomes binding at a LOOSER eps
    (higher unserved fraction) because the system's natural reliability
    is lower. The curve starts rising earlier.
  - Under upside gas, the constraint stays slack until eps is very tight.
  - The curve's STEEPNESS also differs: downside may produce a steeper
    curve (each pp is more expensive when gas is scarce).

The "left shift" and steepness difference together quantify how much
gas deliverability regime drives the cost of reliability.

WHAT IS DIFFERENT FROM REL-1
------------------------------
REL-1 asked: does gas regime determine FEASIBILITY (can the standard
be met at all)?  → binary: feasible/infeasible at each eps.

REL-2 asks: what is the MARGINAL COST of meeting the standard?
→ continuous: how steeply does cost rise as eps tightens?

REL-1 uses the primal (feasibility) as its signal.
REL-2 uses the dual (shadow price) as its signal.
Both are needed: REL-1 establishes that gas regime determines
WHETHER reliability is achievable; REL-2 establishes at what COST.

BINDING DETECTION
-----------------
A dual of zero can mean either:
  (a) constraint is slack (system achieves the target without being forced)
  (b) degenerate LP solution (constraint binding but dual is zero)

To distinguish, primal slack is computed directly:
    primal_slack = eps * total_demand - sum(unserved)

If primal_slack < 1e-4 TWh: constraint is binding (dual is meaningful)
If primal_slack >= 1e-4 TWh: constraint is slack (dual of zero is uninformative)

Only binding-region duals are used to construct the cost curve.

OUTPUTS
--------
  results/rel2/cost_curve_all.csv      — all rows (all gas regimes, all eps)
  results/rel2/cost_curve_binding.csv  — binding region only (dual meaningful)
  results/rel2/curve_summary.csv       — onset eps + steepness per gas regime
  results/rel2/rel2_summary.json       — left-shift metric + interpretation
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
    run_rel2_marginal_cost_curves,
    reliability_results_to_df,
)

# ============================================================
# CONFIG
# ============================================================

RESULTS_DIR    = ROOT / "results" / "rel2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_VOLL = "voll_mid"
CAPITAL_CASE   = "moderate"    # Fixed — same as REL-1 for comparability

GAS_CASES = ["baseline", "upside", "downside", "shock_recovery"]

# Eps grid: 0.20 → 0.05 (loose to tight)
EPS_LEVELS = [0.20, 0.15, 0.10, 0.08, 0.06, 0.05]


# ============================================================
# ANALYSIS HELPERS
# ============================================================

def compute_curve_summary(df):
    """
    For each gas_case, compute:
      onset_eps:  the loosest eps at which the constraint is binding
                  (where the cost curve starts rising)
      onset_reliability: 1 - onset_eps
      steepness:  slope of dual_usd_per_pct_pt vs reliability in binding region
                  (positive = steeper curve = more expensive reliability)
      max_dual:   peak dual_usd_per_pct_pt in binding region

    Returns a DataFrame with one row per gas_case.
    """
    rows = []
    for gas_case, grp in df.groupby("gas_case"):
        feasible = grp[grp["status"] == "feasible"].copy()
        binding  = feasible[feasible["binding"] == True].copy()

        # Onset: loosest eps where binding (largest eps value in binding region)
        onset_eps = float(binding["eps"].max()) if len(binding) > 0 else None
        onset_rel = 1 - onset_eps if onset_eps is not None else None

        # Max dual in binding region
        max_dual = float(binding["dual_usd_per_pct_pt"].max()) if len(binding) > 0 else None

        # Steepness: slope of (dual vs reliability) in binding region
        # Using simple linear regression across the binding eps levels
        steepness = None
        if len(binding) >= 2:
            x = binding["eps"].values           # x = eps (decreasing = tighter)
            y = binding["dual_usd_per_pct_pt"].values
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() >= 2:
                # As eps decreases (tighter standard), dual should increase
                # Steepness = |d(dual)/d(eps)| — larger = steeper curve
                slope = np.polyfit(x[valid], y[valid], 1)[0]
                steepness = float(-slope)  # negate: we want rate of increase as eps falls

        rows.append({
            "gas_case":          gas_case,
            "onset_eps":         onset_eps,
            "onset_reliability": onset_rel,
            "max_dual_usd_per_pct_pt": max_dual,
            "steepness_usd_per_pct_pt_per_eps": steepness,
            "n_binding_eps":     len(binding),
            "n_infeasible_eps":  int((grp["status"] == "infeasible").sum()),
        })

    return pd.DataFrame(rows)


def compute_left_shift(summary_df):
    """
    Quantify the left shift of the cost curve across gas regimes.

    Left shift = how much earlier (at a looser eps) does the cost curve
    start rising compared to the best-case (upside) gas regime?

    onset_eps(upside) - onset_eps(gas_case):
      Positive = gas_case curve starts rising earlier (left-shifted)
      Zero     = same onset as upside
      Negative = gas_case curve starts rising later (right-shifted, cheaper reliability)
    """
    upside_row = summary_df[summary_df["gas_case"] == "upside"]
    upside_onset = (
        float(upside_row["onset_eps"].values[0])
        if len(upside_row) and upside_row["onset_eps"].values[0] is not None
        and not pd.isna(upside_row["onset_eps"].values[0])
        else None
    )

    rows = []
    for _, r in summary_df.iterrows():
        onset = r["onset_eps"]
        left_shift = (
            float(upside_onset - onset)
            if (upside_onset is not None and onset is not None)
            else None
        )
        rows.append({
            "gas_case":       r["gas_case"],
            "onset_eps":      onset,
            "left_shift_eps": left_shift,
            # Positive left_shift means constraint binds earlier under this regime
            "curve_is_left_shifted": (
                left_shift > 1e-3 if left_shift is not None else None
            ),
        })
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    econ = load_econ(CANONICAL_VOLL)

    print(f"\nREL-2: Marginal cost of reliability across gas regimes")
    print(f"  capital_case:  {CAPITAL_CASE}")
    print(f"  gas_cases:     {GAS_CASES}")
    print(f"  eps_levels:    {EPS_LEVELS}")
    print(f"  total solves:  {len(GAS_CASES) * (1 + len(EPS_LEVELS))}")

    # ── Run all sweeps ─────────────────────────────────────────────────────
    all_rows = run_rel2_marginal_cost_curves(
        econ=econ,
        gas_cases=GAS_CASES,
        capital_case=CAPITAL_CASE,
        eps_levels=EPS_LEVELS,
    )

    df = reliability_results_to_df(all_rows)

    # ── Save full results ──────────────────────────────────────────────────
    df.to_csv(RESULTS_DIR / "cost_curve_all.csv", index=False)

    # Binding-region only (where dual is economically meaningful)
    binding_df = df[
        (df["binding"] == True) &
        (df["status"] == "feasible") &
        (df["eps"].notna())
    ].copy()
    binding_df.to_csv(RESULTS_DIR / "cost_curve_binding.csv", index=False)

    print(f"\nSaved: {RESULTS_DIR / 'cost_curve_all.csv'}")
    print(f"Saved: {RESULTS_DIR / 'cost_curve_binding.csv'}")

    # ── Curve summary ──────────────────────────────────────────────────────
    summary_df = compute_curve_summary(df)
    summary_df.to_csv(RESULTS_DIR / "curve_summary.csv", index=False)

    # ── Left-shift analysis ────────────────────────────────────────────────
    shift_df = compute_left_shift(summary_df)

    # ── Build summary JSON ─────────────────────────────────────────────────
    summary = {
        "experiment": {
            "capital_case":  CAPITAL_CASE,
            "gas_cases":     GAS_CASES,
            "eps_levels":    EPS_LEVELS,
            "voll":          CANONICAL_VOLL,
        },
        "curve_by_gas_case": {},
        "left_shift_analysis": {},
        "interpretation": None,
    }

    for _, row in summary_df.iterrows():
        gc = row["gas_case"]
        shift_row = shift_df[shift_df["gas_case"] == gc].iloc[0]
        summary["curve_by_gas_case"][gc] = {
            "onset_eps":                 row["onset_eps"],
            "onset_reliability":         row["onset_reliability"],
            "max_dual_usd_per_pct_pt":   row["max_dual_usd_per_pct_pt"],
            "steepness":                 row["steepness_usd_per_pct_pt_per_eps"],
            "n_binding_eps":             int(row["n_binding_eps"]),
            "n_infeasible_eps":          int(row["n_infeasible_eps"]),
        }
        summary["left_shift_analysis"][gc] = {
            "onset_eps":          row["onset_eps"],
            "left_shift_eps":     shift_row["left_shift_eps"],
            "curve_left_shifted": shift_row["curve_is_left_shifted"],
        }

    # Identify the regime with the most left-shifted curve (earliest onset)
    onset_vals = {
        gc: d["onset_eps"]
        for gc, d in summary["left_shift_analysis"].items()
        if d["onset_eps"] is not None
    }
    if onset_vals:
        most_expensive = max(onset_vals, key=lambda k: onset_vals[k])
        least_expensive = min(onset_vals, key=lambda k: onset_vals[k])
        summary["interpretation"] = (
            f"Under {most_expensive} gas, the reliability cost curve starts "
            f"binding at eps={onset_vals[most_expensive]:.2f} (reliability="
            f"{1-onset_vals[most_expensive]:.2f}), meaning reliability becomes "
            f"costly at lower standards. Under {least_expensive} gas, the curve "
            f"only starts binding at eps={onset_vals[least_expensive]:.2f} "
            f"(reliability={1-onset_vals[least_expensive]:.2f}). "
            f"The left shift between these extremes is "
            f"{onset_vals[most_expensive] - onset_vals[least_expensive]:.2f} "
            f"in eps units."
        )

    with open(RESULTS_DIR / "rel2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {RESULTS_DIR / 'rel2_summary.json'}")

    # ── Console output ─────────────────────────────────────────────────────
    print("\n=== REL-2 COST CURVE SUMMARY ===")
    print(f"\n  {'Gas regime':<18} {'Onset eps':>10} {'Onset rel':>10} "
          f"{'Max dual (M$/pp)':>17} {'Steepness':>10} {'Left shift':>11}")
    print("  " + "-" * 80)

    for _, row in summary_df.sort_values("onset_eps", ascending=False).iterrows():
        gc    = row["gas_case"]
        onset = f"{row['onset_eps']:.2f}" if pd.notna(row["onset_eps"]) else "—"
        orel  = f"{row['onset_reliability']:.2f}" if pd.notna(row["onset_reliability"]) else "—"
        maxd  = (
            f"{row['max_dual_usd_per_pct_pt']/1e6:.1f}"
            if pd.notna(row["max_dual_usd_per_pct_pt"]) else "—"
        )
        steep = (
            f"{row['steepness_usd_per_pct_pt_per_eps']/1e6:.1f}"
            if pd.notna(row["steepness_usd_per_pct_pt_per_eps"]) else "—"
        )
        shift_row = shift_df[shift_df["gas_case"] == gc]
        ls = (
            f"{shift_row['left_shift_eps'].values[0]:.2f}"
            if len(shift_row) and pd.notna(shift_row["left_shift_eps"].values[0])
            else "—"
        )
        print(f"  {gc:<18} {onset:>10} {orel:>10} {maxd:>17} {steep:>10} {ls:>11}")

    print()
    if summary["interpretation"]:
        print("  " + summary["interpretation"])

    print("\n=== BINDING REGION DUALS ===")
    print("  (Only rows where constraint is binding — dual is economically meaningful)")
    print()
    print(f"  {'Gas regime':<18} {'eps':>6} {'Dual (M$/TWh)':>14} "
          f"{'Dual (M$/pp)':>13} {'Slack TWh':>10}")
    print("  " + "-" * 66)
    for _, row in binding_df.sort_values(
        ["gas_case", "eps"], ascending=[True, False]
    ).iterrows():
        d_twh = (
            f"{row['dual_usd_per_twh']/1e6:.2f}"
            if pd.notna(row["dual_usd_per_twh"]) else "—"
        )
        d_pp = (
            f"{row['dual_usd_per_pct_pt']/1e6:.2f}"
            if pd.notna(row["dual_usd_per_pct_pt"]) else "—"
        )
        slack = (
            f"{row['primal_slack_twh']:.4f}"
            if pd.notna(row["primal_slack_twh"]) else "—"
        )
        print(f"  {row['gas_case']:<18} {row['eps']:>6.2f} "
              f"{d_twh:>14} {d_pp:>13} {slack:>10}")


if __name__ == "__main__":
    main()
