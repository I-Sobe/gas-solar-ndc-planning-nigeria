"""
13_run_dem2_growth_gas_interaction.py  —  DEM-2
================================================

RESEARCH QUESTION (DEM-2)
--------------------------
How does demand growth rate uncertainty interact with gas deliverability
constraints to determine system cost risk — and does higher demand growth
make the EaaS mechanism more or less valuable?

WHY THIS IS NOT A MONTE CARLO PROBLEM
--------------------------------------
The existing stochastic.py evaluates cost risk by perturbing parameters
around a FIXED capacity investment (solar MW and storage MWh from one
baseline LP solve). This correctly answers: "given the optimal investment
for 4% growth, what does cost look like if actual growth is 6%?"

DEM-2 asks a different question: "what is the OPTIMAL investment for each
growth rate, and how does the optimal cost vary?" This requires
re-optimising under each growth rate — not evaluating a fixed plan under
perturbed conditions. A fixed-capacity sensitivity systematically
underestimates system adaptability and overestimates cost risk because it
ignores the planner's ability to choose a different capacity mix.

The distinction matters for the thesis: DEM-2's finding is about the
COST OF PLANNING under different growth assumptions, not the cost of
being surprised by growth after investment decisions are locked in.

THE INTERACTION EFFECT
-----------------------
The interaction between demand growth and gas deliverability is measured by:

  gas_premium(growth) = cost(gas_case, growth) - cost(baseline_gas, growth)

  interaction_effect = gas_premium(high_growth) - gas_premium(low_growth)

If positive and large: demand growth AMPLIFIES gas scarcity — every additional
TWh of demand growth must be served by solar (gas is capped), and solar is
more expensive when gas is scarce because the gas shadow price is higher.

If near zero: demand growth and gas scarcity affect cost independently —
they add together but do not compound. The system can adapt to each
pressure separately.

EAAS VALUE ACROSS GROWTH RATES
--------------------------------
EaaS value = cost(public_only) - cost(eaas) for each (growth, gas, policy)

The expected finding is monotone: EaaS becomes more valuable at higher
growth rates because:
  (a) More solar is needed to serve growing demand
  (b) The public budget cap becomes more binding as solar requirements grow
  (c) EaaS fills the gap that public capital cannot finance

A non-monotone finding would indicate that build-rate constraints (the
solar_max_build_mw_per_year cap) bind before the capital budget does at
very high growth — in which case EaaS provides no additional deployment
because both arms hit the same physical build constraint.

NDC INTERACTION WITH DEMAND GROWTH
------------------------------------
Under ndc3_unconditional: solar is needed for BOTH access (serving demand)
AND emissions (substituting gas under the NDC cap). At high demand growth,
both pressures compound. The NDC cost premium under high growth is not simply
the sum of the NDC premium (at baseline growth) plus the high-growth premium
(under no policy) — they interact because the same solar investment serves
both objectives simultaneously.

EXPERIMENT STRUCTURE
--------------------
  demand_cases:    [low (2.5%), baseline (4%), high (6%)]
  gas_cases:       [baseline, upside, downside, shock_recovery]
  financing_arms:  [public_only, eaas]
  policy_arms:     [no_policy, ndc3_unconditional]

  Total solves: 3 × 4 × 2 × 2 = 48 LP solves

OUTPUTS
--------
  results/dem2/dem2_results.csv         — all 48 runs
  results/dem2/gas_premium.csv          — gas scarcity premium per growth rate
  results/dem2/interaction_matrix.csv   — interaction effects (amplification)
  results/dem2/eaas_value.csv           — EaaS value by growth rate and policy
  results/dem2/dem2_summary.json        — headline findings
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io import load_econ
from src.optimize_experiments import run_dem2_growth_gas_matrix

# ============================================================
# CONFIG
# ============================================================

RESULTS_DIR    = ROOT / "results" / "dem2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAP_PATH       = ROOT / "data" / "cost" / "processed" / "emissions_cap.csv"
CANONICAL_VOLL = "voll_mid"
CANONICAL_TARIFF = 95_000_000   # USD/TWh

DEMAND_CASES = ["low", "baseline", "high"]   # 2.5%, 4%, 6% growth
GAS_CASES    = ["baseline", "upside", "downside", "shock_recovery"]

FINANCING_CONFIGS = [
    {
        "label":            "public_only",
        "financing_regime": "traditional",
        "capital_case":     "moderate",
    },
    {
        "label":            "eaas",
        "financing_regime": "eaas",
        "capital_case":     "moderate",
        "required_margin":  1.10,
        "tariff":           CANONICAL_TARIFF,
    },
]

POLICY_CONFIGS = [
    {
        "policy_label": "no_policy",
        "ndc_scenario": None,
    },
    {
        "policy_label": "ndc2_unconditional",
        "ndc_scenario": "ndc2_unconditional",
    },
    {
        "policy_label": "ndc3_unconditional",
        "ndc_scenario": "ndc3_unconditional",
    },
]

# Demand growth rates from scenarios.py (for display)
GROWTH_RATE_MAP = {"low": 0.025, "baseline": 0.04, "high": 0.06}


# ============================================================
# ANALYSIS HELPERS
# ============================================================

def compute_gas_premium(df):
    """
    For each (demand_case, financing_arm, policy_label), compute:
      gas_premium = cost(gas_case) - cost(baseline_gas)

    This is the additional cost imposed by each gas regime above
    the baseline gas scenario at the same demand growth rate.

    Returns a DataFrame with one row per
    (demand_case, gas_case, financing_arm, policy_label).
    """
    rows = []
    for (demand, fin, pol), grp in df.groupby(
        ["demand_case", "financing_arm", "policy_label"]
    ):
        base_row = grp[
            (grp["gas_case"] == "baseline") & (grp["status"] == "optimal")
        ]
        base_cost = (
            float(base_row["npv_total_cost_usd"].values[0])
            if len(base_row) else None
        )

        for _, r in grp.iterrows():
            cost = r["npv_total_cost_usd"] if r["status"] == "optimal" else None
            premium = (
                float(cost - base_cost)
                if (cost is not None and base_cost is not None)
                else None
            )
            rows.append({
                "demand_case":       demand,
                "demand_growth_rate": r.get("demand_growth_rate"),
                "gas_case":          r["gas_case"],
                "financing_arm":     fin,
                "policy_label":      pol,
                "npv_cost_usd":      cost,
                "gas_premium_usd":   premium,
                "gas_premium_pct": (
                    premium / base_cost * 100
                    if (premium is not None and base_cost) else None
                ),
                "status":            r["status"],
            })
    return pd.DataFrame(rows)


def compute_interaction_matrix(gas_premium_df):
    """
    Compute the interaction effect between demand growth and gas regime:

      interaction = gas_premium(high_growth, gas_case)
                  - gas_premium(low_growth,  gas_case)

    Positive: demand growth AMPLIFIES gas scarcity cost
              (super-additive: high growth + bad gas > sum of parts)
    Zero:     independent effects (additive only)

    Returns a DataFrame with one row per (gas_case, financing_arm, policy_label).
    """
    rows = []
    for (gas, fin, pol), grp in gas_premium_df.groupby(
        ["gas_case", "financing_arm", "policy_label"]
    ):
        low_row  = grp[grp["demand_case"] == "low"]
        high_row = grp[grp["demand_case"] == "high"]

        low_prem  = low_row["gas_premium_usd"].values[0]  if len(low_row)  else None
        high_prem = high_row["gas_premium_usd"].values[0] if len(high_row) else None

        interaction = (
            float(high_prem - low_prem)
            if (high_prem is not None and low_prem is not None)
            else None
        )

        rows.append({
            "gas_case":           gas,
            "financing_arm":      fin,
            "policy_label":       pol,
            "gas_premium_low_growth":  low_prem,
            "gas_premium_high_growth": high_prem,
            "interaction_effect_usd":  interaction,
            "interaction_amplifies": (
                interaction > 0 if interaction is not None else None
            ),
            "interpretation": (
                "Demand growth AMPLIFIES gas scarcity cost (super-additive)."
                if (interaction is not None and interaction > 1e6)
                else (
                    "Demand growth and gas scarcity are approximately independent."
                    if interaction is not None
                    else "Cannot compute — missing data."
                )
            ),
        })
    return pd.DataFrame(rows)


def compute_eaas_value(df):
    """
    Compute EaaS value = cost(public_only) - cost(eaas) for each
    (demand_case, gas_case, policy_label).

    Also flags whether EaaS value is monotone increasing with demand growth.
    """
    rows = []
    for (demand, gas, pol), grp in df.groupby(
        ["demand_case", "gas_case", "policy_label"]
    ):
        pub_row  = grp[
            (grp["financing_arm"] == "public_only") & (grp["status"] == "optimal")
        ]
        eaas_row = grp[
            (grp["financing_arm"] == "eaas") & (grp["status"] == "optimal")
        ]

        pub_cost  = pub_row["npv_total_cost_usd"].values[0]  if len(pub_row)  else None
        eaas_cost = eaas_row["npv_total_cost_usd"].values[0] if len(eaas_row) else None

        value = (
            float(pub_cost - eaas_cost)
            if (pub_cost is not None and eaas_cost is not None)
            else None
        )
        value_pct = (
            value / pub_cost * 100
            if (value is not None and pub_cost) else None
        )

        eaas_mw = (
            eaas_row["solar_eaas_total_mw"].values[0]
            if len(eaas_row) else None
        )

        rows.append({
            "demand_case":          demand,
            "demand_growth_rate":   GROWTH_RATE_MAP.get(demand),
            "gas_case":             gas,
            "policy_label":         pol,
            "pub_cost_usd":         pub_cost,
            "eaas_cost_usd":        eaas_cost,
            "eaas_value_usd":       value,
            "eaas_value_pct":       value_pct,
            "eaas_solar_mw":        eaas_mw,
            "eaas_strictly_better": (
                value > 1e6 if value is not None else None
            ),
        })

    eaas_df = pd.DataFrame(rows)

    # Add monotonicity flag per (gas_case, policy_label)
    mono_flags = {}
    for (gas, pol), grp in eaas_df.groupby(["gas_case", "policy_label"]):
        vals = grp.sort_values("demand_growth_rate")["eaas_value_usd"].dropna()
        is_monotone = bool((vals.diff().dropna() >= 0).all()) if len(vals) >= 2 else None
        mono_flags[(gas, pol)] = is_monotone

    eaas_df["eaas_value_monotone_increasing"] = eaas_df.apply(
        lambda r: mono_flags.get((r["gas_case"], r["policy_label"])), axis=1
    )

    return eaas_df


# ============================================================
# MAIN
# ============================================================

def main():

    econ = load_econ(CANONICAL_VOLL)

    n_solves = (
        len(DEMAND_CASES) * len(GAS_CASES)
        * len(FINANCING_CONFIGS) * len(POLICY_CONFIGS)
    )
    print(f"\nDEM-2: Demand growth × gas deliverability interaction")
    print(f"  demand_cases:    {DEMAND_CASES}  "
          f"(growth: {[GROWTH_RATE_MAP[d] for d in DEMAND_CASES]})")
    print(f"  gas_cases:       {GAS_CASES}")
    print(f"  financing_arms:  {[f['label'] for f in FINANCING_CONFIGS]}")
    print(f"  policy_arms:     {[p['policy_label'] for p in POLICY_CONFIGS]}")
    print(f"  total solves:    {n_solves}")

    # ── Run all combinations ───────────────────────────────────────────────
    rows = run_dem2_growth_gas_matrix(
        econ=econ,
        demand_cases=DEMAND_CASES,
        gas_cases=GAS_CASES,
        financing_configs=FINANCING_CONFIGS,
        policy_configs=POLICY_CONFIGS,
        cap_path=str(CAP_PATH),
    )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "dem2_results.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR / 'dem2_results.csv'}")

    optimal = df[df["status"] == "optimal"].copy()

    # ── Gas premium table ──────────────────────────────────────────────────
    gas_prem_df = compute_gas_premium(optimal)
    gas_prem_df.to_csv(RESULTS_DIR / "gas_premium.csv", index=False)

    # ── Interaction matrix ─────────────────────────────────────────────────
    interaction_df = compute_interaction_matrix(gas_prem_df)
    interaction_df.to_csv(RESULTS_DIR / "interaction_matrix.csv", index=False)

    # ── EaaS value ────────────────────────────────────────────────────────
    eaas_df = compute_eaas_value(optimal)
    eaas_df.to_csv(RESULTS_DIR / "eaas_value.csv", index=False)

    # ── Summary JSON ──────────────────────────────────────────────────────
    summary = {
        "experiment": {
            "demand_cases":   DEMAND_CASES,
            "growth_rates":   GROWTH_RATE_MAP,
            "gas_cases":      GAS_CASES,
            "policy_arms":    [p["policy_label"] for p in POLICY_CONFIGS],
            "voll":           CANONICAL_VOLL,
        },
        "interaction_findings": {},
        "eaas_value_findings":  {},
    }

    # Interaction by policy and gas case
    for pol in [p["policy_label"] for p in POLICY_CONFIGS]:
        sub = interaction_df[
            (interaction_df["policy_label"] == pol)
            & (interaction_df["financing_arm"] == "public_only")
        ]
        summary["interaction_findings"][pol] = {
            gc: {
                "gas_premium_low":  row["gas_premium_low_growth"],
                "gas_premium_high": row["gas_premium_high_growth"],
                "interaction":      row["interaction_effect_usd"],
                "amplifies":        row["interaction_amplifies"],
            }
            for _, row in sub.iterrows()
            for gc in [row["gas_case"]]
        }

    # EaaS value monotonicity by policy
    for pol in [p["policy_label"] for p in POLICY_CONFIGS]:
        sub = eaas_df[eaas_df["policy_label"] == pol]
        summary["eaas_value_findings"][pol] = {
            "eaas_value_monotone_by_gas_case": {
                gc: bool(
                    grp["eaas_value_monotone_increasing"].values[0]
                )
                for gc, grp in sub.groupby("gas_case")
                if len(grp) and grp["eaas_value_monotone_increasing"].notna().any()
            },
            "max_eaas_value_usd": (
                float(sub["eaas_value_usd"].max())
                if sub["eaas_value_usd"].notna().any() else None
            ),
            "max_eaas_value_at_demand_case": (
                sub.loc[sub["eaas_value_usd"].idxmax(), "demand_case"]
                if sub["eaas_value_usd"].notna().any() else None
            ),
        }

    with open(RESULTS_DIR / "dem2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {RESULTS_DIR / 'dem2_summary.json'}")

    # ── Console output ─────────────────────────────────────────────────────
    print("\n=== DEM-2: GAS SCARCITY PREMIUM BY DEMAND GROWTH ===")
    print("  (public_only, no_policy — shows raw interaction)")
    print()
    print(f"  {'Gas case':<18} {'Low (2.5%)':>11} {'Base (4%)':>10} "
          f"{'High (6%)':>10} {'Interaction':>13} {'Amplifies':>10}")
    print("  " + "-" * 76)

    inter_sub = interaction_df[
        (interaction_df["financing_arm"] == "public_only")
        & (interaction_df["policy_label"] == "no_policy")
    ]
    prem_sub = gas_prem_df[
        (gas_prem_df["financing_arm"] == "public_only")
        & (gas_prem_df["policy_label"] == "no_policy")
    ]

    for gc in GAS_CASES:
        if gc == "baseline":
            continue
        i_row = inter_sub[inter_sub["gas_case"] == gc]
        low_p = prem_sub[
            (prem_sub["gas_case"] == gc) & (prem_sub["demand_case"] == "low")
        ]["gas_premium_usd"].values
        base_p = prem_sub[
            (prem_sub["gas_case"] == gc) & (prem_sub["demand_case"] == "baseline")
        ]["gas_premium_usd"].values
        high_p = prem_sub[
            (prem_sub["gas_case"] == gc) & (prem_sub["demand_case"] == "high")
        ]["gas_premium_usd"].values

        lv = f"{low_p[0]/1e9:.2f}B"   if len(low_p)  and pd.notna(low_p[0])  else "—"
        bv = f"{base_p[0]/1e9:.2f}B"  if len(base_p) and pd.notna(base_p[0]) else "—"
        hv = f"{high_p[0]/1e9:.2f}B"  if len(high_p) and pd.notna(high_p[0]) else "—"

        if len(i_row):
            inter = i_row["interaction_effect_usd"].values[0]
            iv = f"{inter/1e9:.2f}B" if pd.notna(inter) else "—"
            amp = "YES" if i_row["interaction_amplifies"].values[0] else "no"
        else:
            iv, amp = "—", "—"

        print(f"  {gc:<18} {lv:>11} {bv:>10} {hv:>10} {iv:>13} {amp:>10}")

    print("\n=== DEM-2: EAAS VALUE BY DEMAND GROWTH (no_policy) ===")
    print()
    print(f"  {'Gas case':<18} {'Low':>9} {'Baseline':>9} {'High':>9} "
          f"{'Monotone':>9}")
    print("  " + "-" * 58)

    eaas_sub = eaas_df[eaas_df["policy_label"] == "no_policy"]
    for gc in GAS_CASES:
        gc_rows = eaas_sub[eaas_sub["gas_case"] == gc].sort_values(
            "demand_growth_rate"
        )
        vals = []
        for dc in DEMAND_CASES:
            r = gc_rows[gc_rows["demand_case"] == dc]
            v = f"{r['eaas_value_usd'].values[0]/1e9:.2f}B" if len(r) and pd.notna(r["eaas_value_usd"].values[0]) else "—"
            vals.append(v)
        mono = gc_rows["eaas_value_monotone_increasing"].values[0] if len(gc_rows) else None
        mono_str = "YES" if mono else ("no" if mono is False else "—")
        print(f"  {gc:<18} {vals[0]:>9} {vals[1]:>9} {vals[2]:>9} {mono_str:>9}")


if __name__ == "__main__":
    main()
