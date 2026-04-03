"""
11_run_rel3_financing_frontier.py  —  REL-3
============================================

RESEARCH QUESTION (REL-3)
--------------------------
Does EaaS financing improve reliability outcomes (measured by the
feasibility frontier) relative to public-capital-only financing under
the same gas constraint? Is this still important under NDC3.0?

WHY REL-3 IS MORE IMPORTANT UNDER NDC3.0
-----------------------------------------
Under NDC3.0, two constraints interact simultaneously:

  (A) reliability_constraint: sum(unserved) <= eps × total_demand
  (B) emissions_constraint[t]: emissions[t] <= cap_t  (annual NDC cap)

Under public-capital-only financing, the public budget caps solar deployment.
If the budget is exhausted before sufficient solar is built to both (a) meet
the reliability target AND (b) substitute for gas emissions, the system is
infeasible — both constraints cannot be satisfied simultaneously.

EaaS adds private solar deployment beyond the public budget. This additional
solar serves BOTH objectives at once:
  - Reduces unserved energy (improving reliability)
  - Substitutes for gas generation (reducing emissions)

The NDC3.0 conditional scenario (47% reduction, required_margin=1.05) is
the most stringent test: concessional finance makes more EaaS bankable,
but the emissions cap is tighter. The interaction is non-trivial.

WHAT "FEASIBILITY FRONTIER IMPROVEMENT" MEANS
----------------------------------------------
The feasibility frontier is the set of (eps, cost) pairs where the
system CAN achieve reliability standard eps at total cost C.

EaaS improves the frontier if:
  threshold_eps(EaaS) < threshold_eps(public_only)
  i.e. EaaS achieves tighter standards (smaller eps still feasible)

The frontier_shift_eps metric quantifies this:
  frontier_shift = threshold(public_only) - threshold(eaas)
  Positive: EaaS achieves tighter reliability
  Zero:     EaaS provides no feasibility benefit at this gas/policy combo
  Null:     Cannot compute — one or both arms fully infeasible

Special case (EaaS necessary, not just beneficial):
  If public_only is fully infeasible (cannot achieve ANY eps) but EaaS
  is feasible, EaaS is NECESSARY for any reliability standard to be met.
  This is the strongest possible finding.

EXPERIMENT STRUCTURE
--------------------
  gas_cases:     4 regimes (baseline, upside, downside, shock_recovery)
  policy_arms:   3 levels (no_policy, ndc3_unconditional, ndc3_conditional)
  financing_arms: 2 (public_only, eaas)
  eps_levels:    [0.20, 0.15, 0.10, 0.08, 0.06, 0.05]

  Total solves: 4 × 3 × 2 × 7 = 168

FINANCING ARM PARAMETERS
------------------------
  public_only:
    financing_regime = "traditional"
    capital_case = "moderate"  (for no_policy + unconditional)
    capital_case = "expansion" (for conditional — matches 02/03 scripts)

  eaas:
    financing_regime = "eaas"
    solar_service_tariff_usd_per_twh = 95_000_000 (canonical tariff)
    required_margin = 1.10 for no_policy + unconditional (commercial)
    required_margin = 1.05 for conditional (concessional)
    capital_case matches public_only arm per policy

OUTPUTS
--------
  results/rel3/frontier_all.csv          — all (gas, policy, financing, eps) rows
  results/rel3/frontier_threshold.csv    — threshold per combination
  results/rel3/frontier_shift.csv        — shift metric per (gas, policy)
  results/rel3/rel3_summary.json         — structured findings
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
    run_rel3_financing_frontier,
    compute_frontier_shift,
    reliability_results_to_df,
)

# ============================================================
# CONFIG
# ============================================================

RESULTS_DIR    = ROOT / "results" / "rel3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAP_PATH       = ROOT / "data" / "cost" / "processed" / "emissions_cap.csv"
CANONICAL_VOLL = "voll_mid"
CANONICAL_TARIFF = 95_000_000   # USD/TWh — above self-financing threshold

GAS_CASES = ["baseline", "upside", "downside", "shock_recovery"]
EPS_LEVELS = [0.20, 0.15, 0.10, 0.08, 0.06, 0.05]

# Policy configurations — one dict per policy arm
# Parameters set the financing arm details for that policy context
POLICY_CONFIGS = [
    {
        "policy_label":   "no_policy",
        "ndc_scenario":   None,              # no emissions cap
        "public_capital": "moderate",
        "eaas_capital":   "moderate",
        "eaas_margin":    1.10,
        "eaas_tariff":    CANONICAL_TARIFF,
    },
    {
        "policy_label":   "ndc3_unconditional",
        "ndc_scenario":   "ndc3_unconditional",
        "public_capital": "moderate",        # matches 02_run_ndc_caps.py
        "eaas_capital":   "moderate",
        "eaas_margin":    1.10,              # commercial finance
        "eaas_tariff":    CANONICAL_TARIFF,
    },
    {
        "policy_label":   "ndc3_conditional",
        "ndc_scenario":   "ndc3_conditional",
        "public_capital": "expansion",       # matches 02_run_ndc_caps.py
        "eaas_capital":   "expansion",
        "eaas_margin":    1.05,              # concessional finance
        "eaas_tariff":    CANONICAL_TARIFF,
    },
]


# ============================================================
# MAIN
# ============================================================

def main():

    econ = load_econ(CANONICAL_VOLL)

    n_solves = len(GAS_CASES) * len(POLICY_CONFIGS) * 2 * (1 + len(EPS_LEVELS))
    print(f"\nREL-3: EaaS vs public-only financing frontier")
    print(f"  gas_cases:    {GAS_CASES}")
    print(f"  policy_arms:  {[p['policy_label'] for p in POLICY_CONFIGS]}")
    print(f"  eps_levels:   {EPS_LEVELS}")
    print(f"  total solves: {n_solves}")

    # ── Run all combinations ───────────────────────────────────────────────
    all_rows = run_rel3_financing_frontier(
        econ=econ,
        gas_cases=GAS_CASES,
        policy_configs=POLICY_CONFIGS,
        eps_levels=EPS_LEVELS,
        cap_path=str(CAP_PATH),
    )

    df = reliability_results_to_df(all_rows)

    # Tag financing_label and policy_label from the new row fields
    # (reliability_results_to_df may not carry these through — add them)
    for col in ["financing_label", "policy_label"]:
        if col not in df.columns:
            df[col] = [r.get(col, "") for r in all_rows]

    df.to_csv(RESULTS_DIR / "frontier_all.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR / 'frontier_all.csv'}")

    # ── Compute thresholds and shifts ──────────────────────────────────────
    threshold_df, shift_df = compute_frontier_shift(df)

    threshold_df.to_csv(RESULTS_DIR / "frontier_threshold.csv", index=False)
    shift_df.to_csv(RESULTS_DIR / "frontier_shift.csv", index=False)
    print(f"Saved: {RESULTS_DIR / 'frontier_threshold.csv'}")
    print(f"Saved: {RESULTS_DIR / 'frontier_shift.csv'}")

    # ── Build summary JSON ─────────────────────────────────────────────────
    summary = {
        "experiment": {
            "gas_cases":     GAS_CASES,
            "policy_arms":   [p["policy_label"] for p in POLICY_CONFIGS],
            "eps_levels":    EPS_LEVELS,
            "canonical_tariff_usd_per_twh": CANONICAL_TARIFF,
            "voll":          CANONICAL_VOLL,
        },
        "findings_by_policy": {},
        "ndc_relevance": {},
    }

    for pol in [p["policy_label"] for p in POLICY_CONFIGS]:
        pol_shift = shift_df[shift_df["policy_label"] == pol]

        # Cases where EaaS is strictly better (positive shift)
        better = pol_shift[pol_shift["eaas_strictly_better"] == True]
        necessary = pol_shift[pol_shift["eaas_necessary"] == True]

        summary["findings_by_policy"][pol] = {
            "n_gas_cases_where_eaas_strictly_better": int(len(better)),
            "n_gas_cases_where_eaas_necessary":       int(len(necessary)),
            "max_frontier_shift_eps": (
                float(pol_shift["frontier_shift_eps"].max())
                if pol_shift["frontier_shift_eps"].notna().any() else None
            ),
            "gas_cases_eaas_necessary":  necessary["gas_case"].tolist(),
            "gas_cases_eaas_better":     better["gas_case"].tolist(),
            "shift_by_gas_case": pol_shift[
                ["gas_case", "pub_threshold_eps", "eaas_threshold_eps",
                 "frontier_shift_eps", "eaas_strictly_better", "eaas_necessary",
                 "interpretation"]
            ].to_dict(orient="records"),
        }

    # NDC relevance: does EaaS improvement grow under tighter NDC caps?
    no_pol  = summary["findings_by_policy"].get("no_policy", {})
    unc_pol = summary["findings_by_policy"].get("ndc3_unconditional", {})
    con_pol = summary["findings_by_policy"].get("ndc3_conditional", {})

    max_shift_no_pol = no_pol.get("max_frontier_shift_eps")
    max_shift_unc    = unc_pol.get("max_frontier_shift_eps")
    max_shift_con    = con_pol.get("max_frontier_shift_eps")

    summary["ndc_relevance"] = {
        "max_shift_no_policy":        max_shift_no_pol,
        "max_shift_ndc_unconditional": max_shift_unc,
        "max_shift_ndc_conditional":   max_shift_con,
        "eaas_improvement_grows_with_ndc_ambition": (
            (max_shift_con is not None and max_shift_no_pol is not None
             and max_shift_con > max_shift_no_pol)
            if (max_shift_con is not None and max_shift_no_pol is not None)
            else None
        ),
        "interpretation": (
            "EaaS frontier improvement grows as NDC ambition increases, confirming "
            "that EaaS becomes MORE important (not just convenient) under NDC3.0."
            if (max_shift_con is not None and max_shift_no_pol is not None
                and max_shift_con > max_shift_no_pol)
            else (
                "EaaS frontier improvement does not increase with NDC ambition. "
                "EaaS may be sufficient without NDC pressure at the current gas regime."
                if (max_shift_con is not None and max_shift_no_pol is not None)
                else "Insufficient data to assess NDC relevance."
            )
        ),
    }

    with open(RESULTS_DIR / "rel3_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {RESULTS_DIR / 'rel3_summary.json'}")

    # ── Console output ─────────────────────────────────────────────────────
    print("\n=== REL-3 FRONTIER SHIFT TABLE ===")
    print("  (frontier_shift > 0 means EaaS achieves tighter reliability)")
    print()

    for pol in [p["policy_label"] for p in POLICY_CONFIGS]:
        print(f"\n  Policy: {pol}")
        print(f"  {'Gas regime':<18} {'Pub thresh':>11} {'EaaS thresh':>12} "
              f"{'Shift eps':>10} {'EaaS better':>12} {'Necessary':>10}")
        print("  " + "-" * 76)

        pol_rows = shift_df[shift_df["policy_label"] == pol].sort_values("gas_case")
        for _, row in pol_rows.iterrows():
            pt  = f"{row['pub_threshold_eps']:.2f}"  if pd.notna(row["pub_threshold_eps"])  else "infeas"
            et  = f"{row['eaas_threshold_eps']:.2f}" if pd.notna(row["eaas_threshold_eps"]) else "infeas"
            sh  = f"{row['frontier_shift_eps']:+.2f}" if pd.notna(row["frontier_shift_eps"]) else "—"
            bet = "YES" if row["eaas_strictly_better"] else "no"
            nec = "YES" if row["eaas_necessary"] else "no"
            print(f"  {row['gas_case']:<18} {pt:>11} {et:>12} {sh:>10} {bet:>12} {nec:>10}")

    print()
    print("=== NDC RELEVANCE ===")
    ndc = summary["ndc_relevance"]
    print(f"  Max shift — no_policy:        {ndc['max_shift_no_policy']}")
    print(f"  Max shift — ndc_unconditional: {ndc['max_shift_ndc_unconditional']}")
    print(f"  Max shift — ndc_conditional:   {ndc['max_shift_ndc_conditional']}")
    print(f"  EaaS grows with NDC ambition: {ndc['eaas_improvement_grows_with_ndc_ambition']}")
    print(f"  {ndc['interpretation']}")


if __name__ == "__main__":
    main()
