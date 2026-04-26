"""
12_run_dem1_demand_sensitivity.py  —  DEM-1
============================================

RESEARCH QUESTION (DEM-1)
--------------------------
How sensitive are the NDC compliance cost and EaaS deployment findings
to the choice of served vs latent demand as the planning base? Does the
financing bottleneck diagnosis change when latent demand is used?

WHAT SERVED VS LATENT MEANS
----------------------------
Nigeria's reported electricity consumption is supply-constrained — not all
demand is being met. Official statistics capture SERVED demand (what was
actually delivered), not LATENT demand (what would be consumed under
reliable, unconstrained supply).

Three planning bases:
  served:      23.08 TWh/yr — NBS Q1 2024 annualised (observed lower bound)
  latent_low:  38.47 TWh/yr — reconstructed via suppression factor λ = 0.60
  latent_high: 76.93 TWh/yr — reconstructed via suppression factor λ = 0.30

The suppression factor λ represents the fraction of true demand currently
being served. λ = 0.60 means 60% of true demand is served; λ = 0.30 means
only 30% is served, implying much higher latent demand.

WHY THE CHOICE OF DEMAND BASE MATTERS
--------------------------------------
The demand base affects:

1. SOLAR REQUIREMENT:
   Gas electricity output is supply-capped at ~17.5 TWh_e/yr (40.66 TWh_th × η=0.43).
   Under served demand (23 TWh_e): solar must cover 24% of demand.
   Under latent_low (38 TWh_e):   solar must cover 55% of demand.
   Under latent_high (77 TWh_e):  solar must cover 77% of demand.
   More solar requires more capital → capital bottleneck worsens.

2. NDC COMPLIANCE COST:
   The emissions cap (built from the baseline served-demand run) applies the
   SAME absolute cap regardless of demand level. Under latent demand, the model
   must build more solar to meet higher demand under the same emissions ceiling.
   This increases the NPV cost of compliance, potentially substantially.

3. CAPITAL BOTTLENECK INTENSITY:
   Under served demand, the moderate capital budget (5.21B) may be sufficient.
   Under latent demand, the same budget may be severely binding — the shadow
   price on the budget constraint rises and EaaS becomes necessary rather
   than convenient.

4. EAAS DEPLOYMENT:
   If the public budget is exhausted under latent demand, EaaS is the only
   mechanism to deploy additional solar. The EaaS deployment volume should
   increase monotonically with demand level — this is the test of whether
   EaaS transitions from a financing tool to an energy access enabler.

NOTE ON EMISSIONS CAPS
----------------------
The emissions caps were built from the baseline (served demand) gas_to_power path.
Because gas deliverability is supply-constrained (fixed at 40.66 TWh_th regardless
of demand level), gas_to_power and therefore emissions are identical across demand
levels. The existing caps from 00_build_emissions_cap.py are valid here without
rebuilding. A higher demand assumption does NOT increase emissions — it only
increases unserved energy and the solar requirement to serve additional demand.

EXPERIMENT STRUCTURE
--------------------
  demand_level_cases: [served, latent_low, latent_high]
  ndc_scenarios:      [ndc3_unconditional, ndc3_conditional]
  financing_arms:     [public_only, eaas]

  Total solves: 3 × 2 × 2 = 12 LP solves

OUTPUTS
--------
  results/dem1/dem1_results.csv       — all 12 runs
  results/dem1/cost_sensitivity.csv   — cost premium of latent vs served
  results/dem1/bottleneck_shift.csv   — budget shadow change across demand levels
  results/dem1/eaas_deployment.csv    — EaaS MW by demand level and policy
  results/dem1/dem1_summary.json      — headline findings
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io import load_econ
from src.optimize_experiments import run_dem1_demand_sensitivity

# ============================================================
# CONFIG
# ============================================================

RESULTS_DIR    = ROOT / "results" / "dem1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAP_PATH       = ROOT / "data" / "cost" / "processed" / "emissions_cap.csv"
CANONICAL_VOLL = "voll_mid"
CANONICAL_TARIFF = 95_000_000   # USD/TWh

DEMAND_LEVEL_CASES = ["served", "latent_low", "latent_high"]

# NDC configurations — consistent with 02_run_ndc_caps.py and 03_run_ndc_eaas.py
NDC_CONFIGS = [
    {
        "ndc_label":    "ndc2_unconditional",
        "ndc_scenario": "ndc2_unconditional",
        "public_capital": "moderate",    # 0.85 × B* = 5.21B
        "eaas_capital":   "moderate",
        "eaas_margin":    1.10,          # commercial finance
        "eaas_tariff":    CANONICAL_TARIFF,
    },
    {
        "ndc_label":    "ndc2_conditional",
        "ndc_scenario": "ndc2_conditional",
        "public_capital": "expansion",   # 1.20 × B* = 7.36B
        "eaas_capital":   "expansion",
        "eaas_margin":    1.05,          # concessional finance
        "eaas_tariff":    CANONICAL_TARIFF,
    },
    {
        "ndc_label":    "ndc3_unconditional",
        "ndc_scenario": "ndc3_unconditional",
        "public_capital": "moderate",    # 0.85 × B* = 5.21B
        "eaas_capital":   "moderate",
        "eaas_margin":    1.10,          # commercial finance
        "eaas_tariff":    CANONICAL_TARIFF,
    },
    {
        "ndc_label":    "ndc3_conditional",
        "ndc_scenario": "ndc3_conditional",
        "public_capital": "expansion",   # 1.20 × B* = 7.36B
        "eaas_capital":   "expansion",
        "eaas_margin":    1.05,          # concessional finance
        "eaas_tariff":    CANONICAL_TARIFF,
    },
]


# ============================================================
# ANALYSIS HELPERS
# ============================================================

def compute_cost_sensitivity(df):
    """
    For each (ndc_label, financing_arm), compute the cost premium of
    latent demand relative to served demand:
      cost_premium_pct = (cost(latent) - cost(served)) / cost(served) × 100
    """
    rows = []
    for (ndc, fin), grp in df.groupby(["ndc_label", "financing_arm"]):
        served_row = grp[grp["demand_level"] == "served"]
        served_cost = (
            float(served_row["npv_total_cost_usd"].values[0])
            if len(served_row) and served_row["status"].values[0] == "optimal"
            else None
        )

        for _, r in grp.iterrows():
            if r["status"] != "optimal":
                rows.append({
                    "demand_level": r["demand_level"],
                    "ndc_label":    ndc,
                    "financing_arm": fin,
                    "npv_cost_usd": None,
                    "cost_premium_vs_served_pct": None,
                    "status": r["status"],
                })
                continue

            cost = r["npv_total_cost_usd"]
            premium = (
                (cost - served_cost) / served_cost * 100
                if served_cost else None
            )
            rows.append({
                "demand_level":               r["demand_level"],
                "ndc_label":                  ndc,
                "financing_arm":              fin,
                "npv_cost_b_usd":             cost / 1e9,
                "cost_premium_vs_served_pct": premium,
                "cumulative_unserved_twh":    r["cumulative_unserved_twh"],
                "horizon_reliability":        r["horizon_reliability"],
                "solar_total_mw":             r["solar_total_mw"],
                "status":                     r["status"],
            })
    return pd.DataFrame(rows)


def compute_bottleneck_shift(df):
    """
    For each (ndc_label, financing_arm), show how the budget shadow price
    and utilisation change across demand levels.

    A rising budget_shadow as demand increases confirms that the capital
    bottleneck intensifies under latent demand assumptions.
    """
    rows = []
    for (ndc, fin), grp in df.groupby(["ndc_label", "financing_arm"]):
        served_shadow = None
        served_row = grp[
            (grp["demand_level"] == "served") & (grp["status"] == "optimal")
        ]
        if len(served_row):
            served_shadow = served_row["budget_shadow_usd_per_usd"].values[0]

        for _, r in grp.iterrows():
            shadow = r.get("budget_shadow_usd_per_usd")
            shift = (
                float(shadow - served_shadow)
                if (shadow is not None and served_shadow is not None)
                else None
            )
            rows.append({
                "demand_level":          r["demand_level"],
                "ndc_label":             ndc,
                "financing_arm":         fin,
                "base_demand_twh":       r.get("base_demand_twh"),
                "budget_shadow":         shadow,
                "budget_utilisation":    r.get("budget_utilisation"),
                "shadow_shift_vs_served": shift,
                "bottleneck_worsens":    (
                    shift > 1e-6 if shift is not None else None
                ),
                "status":                r.get("status"),
            })
    return pd.DataFrame(rows)


def compute_eaas_deployment(df):
    """
    Show EaaS solar deployment and eaas_improvement across demand levels.
    The eaas_improvement is the reduction in unserved energy from
    switching to the EaaS arm at the same demand level.
    """
    rows = []
    for (ndc, demand), grp in df.groupby(["ndc_label", "demand_level"]):
        pub_row  = grp[
            (grp["financing_arm"] == "public_only") & (grp["status"] == "optimal")
        ]
        eaas_row = grp[
            (grp["financing_arm"] == "eaas") & (grp["status"] == "optimal")
        ]

        pub_unserved  = pub_row["cumulative_unserved_twh"].values[0]  if len(pub_row)  else None
        eaas_unserved = eaas_row["cumulative_unserved_twh"].values[0] if len(eaas_row) else None
        eaas_mw       = eaas_row["solar_eaas_total_mw"].values[0]     if len(eaas_row) else None

        access_improvement = (
            float(pub_unserved - eaas_unserved)
            if (pub_unserved is not None and eaas_unserved is not None)
            else None
        )

        rows.append({
            "ndc_label":                   ndc,
            "demand_level":                demand,
            "base_demand_twh":             df[
                (df["ndc_label"] == ndc) & (df["demand_level"] == demand)
            ]["base_demand_twh"].values[0] if len(grp) else None,
            "pub_unserved_twh":            pub_unserved,
            "eaas_unserved_twh":           eaas_unserved,
            "eaas_solar_mw":               eaas_mw,
            "access_improvement_twh":      access_improvement,
            "eaas_necessary": (
                pub_unserved is not None and pub_unserved > 0.01
                and eaas_unserved is not None and eaas_unserved < 0.01
            ),
        })
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    econ = load_econ(CANONICAL_VOLL)

    print(f"\nDEM-1: Demand level sensitivity")
    print(f"  demand_levels:  {DEMAND_LEVEL_CASES}")
    print(f"  ndc_scenarios:  {[c['ndc_label'] for c in NDC_CONFIGS]}")
    print(f"  financing_arms: [public_only, eaas]")
    print(f"  total solves:   {len(DEMAND_LEVEL_CASES) * len(NDC_CONFIGS) * 2}")

    rows = run_dem1_demand_sensitivity(
        econ=econ,
        demand_level_cases=DEMAND_LEVEL_CASES,
        ndc_configs=NDC_CONFIGS,
        cap_path=str(CAP_PATH),
    )

    df = pd.DataFrame(rows)

    # ── Save raw results ───────────────────────────────────────────────────
    df.to_csv(RESULTS_DIR / "dem1_results.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR / 'dem1_results.csv'}")

    # ── Derived tables ─────────────────────────────────────────────────────
    optimal = df[df["status"] == "optimal"].copy()

    cost_df = compute_cost_sensitivity(optimal)
    cost_df.to_csv(RESULTS_DIR / "cost_sensitivity.csv", index=False)

    bottleneck_df = compute_bottleneck_shift(optimal)
    bottleneck_df.to_csv(RESULTS_DIR / "bottleneck_shift.csv", index=False)

    eaas_df = compute_eaas_deployment(optimal)
    eaas_df.to_csv(RESULTS_DIR / "eaas_deployment.csv", index=False)

    # ── Summary JSON ──────────────────────────────────────────────────────
    summary = {"findings": {}, "bottleneck_diagnosis": {}, "eaas_necessity": {}}

    for ndc_label in [c["ndc_label"] for c in NDC_CONFIGS]:
        sub = cost_df[cost_df["ndc_label"] == ndc_label]

        # Cost premium at latent_high vs served
        lh_pub = sub[
            (sub["demand_level"] == "latent_high")
            & (sub["financing_arm"] == "public_only")
        ]
        cost_premium_lh = (
            float(lh_pub["cost_premium_vs_served_pct"].values[0])
            if len(lh_pub) and pd.notna(lh_pub["cost_premium_vs_served_pct"].values[0])
            else None
        )

        summary["findings"][ndc_label] = {
            "cost_premium_latent_high_vs_served_pct": cost_premium_lh,
            "latent_high_feasible_public_only": bool(
                len(df[
                    (df["ndc_label"] == ndc_label)
                    & (df["demand_level"] == "latent_high")
                    & (df["financing_arm"] == "public_only")
                    & (df["status"] == "optimal")
                ]) > 0
            ),
            "latent_high_feasible_eaas": bool(
                len(df[
                    (df["ndc_label"] == ndc_label)
                    & (df["demand_level"] == "latent_high")
                    & (df["financing_arm"] == "eaas")
                    & (df["status"] == "optimal")
                ]) > 0
            ),
        }

        # Bottleneck: does shadow price rise with demand?
        bt_sub = bottleneck_df[bottleneck_df["ndc_label"] == ndc_label]
        pub_bt = bt_sub[bt_sub["financing_arm"] == "public_only"]
        shadows = pub_bt.set_index("demand_level")["budget_shadow"].to_dict()
        summary["bottleneck_diagnosis"][ndc_label] = {
            "budget_shadow_served":      shadows.get("served"),
            "budget_shadow_latent_low":  shadows.get("latent_low"),
            "budget_shadow_latent_high": shadows.get("latent_high"),
            "bottleneck_intensifies":    (
                shadows.get("latent_high", 0) is not None
                and shadows.get("served", 0) is not None
                and shadows.get("latent_high", 0) > shadows.get("served", 0)
            ),
        }

        # EaaS necessity
        eaas_sub = eaas_df[eaas_df["ndc_label"] == ndc_label]
        summary["eaas_necessity"][ndc_label] = {
            dl: bool(
                eaas_sub[eaas_sub["demand_level"] == dl]["eaas_necessary"].values[0]
            )
            for dl in DEMAND_LEVEL_CASES
            if len(eaas_sub[eaas_sub["demand_level"] == dl]) > 0
        }

    with open(RESULTS_DIR / "dem1_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Console output ─────────────────────────────────────────────────────
    print("\n=== DEM-1: COST SENSITIVITY ===")
    print(f"  {'NDC':<22} {'Demand level':<14} {'Arm':<12} "
          f"{'Cost (B$)':>10} {'Premium':>9} {'Unserved':>9}")
    print("  " + "-" * 80)
    for _, r in cost_df.sort_values(
        ["ndc_label", "financing_arm", "demand_level"]
    ).iterrows():
        cost  = f"{r['npv_cost_b_usd']:.2f}" if pd.notna(r.get("npv_cost_b_usd")) else "infeas"
        prem  = (
            f"{r['cost_premium_vs_served_pct']:+.1f}%"
            if pd.notna(r.get("cost_premium_vs_served_pct")) else "—"
        )
        unser = (
            f"{r['cumulative_unserved_twh']:.2f}"
            if pd.notna(r.get("cumulative_unserved_twh")) else "—"
        )
        print(f"  {r['ndc_label']:<22} {r['demand_level']:<14} "
              f"{r['financing_arm']:<12} {cost:>10} {prem:>9} {unser:>9}")

    print("\n=== DEM-1: BOTTLENECK DIAGNOSIS ===")
    print("  (budget_shadow = USD system cost saving per USD additional public capital)")
    print(f"  {'NDC':<22} {'Demand level':<14} {'Arm':<12} "
          f"{'Budget shadow':>14} {'Utilisation':>12} {'Worsens?':>9}")
    print("  " + "-" * 85)
    for _, r in bottleneck_df.sort_values(
        ["ndc_label", "financing_arm", "demand_level"]
    ).iterrows():
        sh   = f"{r['budget_shadow']:.4f}"   if pd.notna(r.get("budget_shadow"))   else "—"
        util = f"{r['budget_utilisation']:.2f}" if pd.notna(r.get("budget_utilisation")) else "—"
        wors = "YES" if r.get("bottleneck_worsens") else "no"
        print(f"  {r['ndc_label']:<22} {r['demand_level']:<14} "
              f"{r['financing_arm']:<12} {sh:>14} {util:>12} {wors:>9}")

    print("\n=== DEM-1: EAAS NECESSITY ===")
    print(f"  {'NDC':<22} {'Demand level':<14} "
          f"{'EaaS MW':>9} {'Access gain TWh':>16} {'Necessary?':>11}")
    print("  " + "-" * 76)
    for _, r in eaas_df.sort_values(["ndc_label", "demand_level"]).iterrows():
        mw  = f"{r['eaas_solar_mw']:.0f}"             if pd.notna(r.get("eaas_solar_mw")) else "—"
        ag  = f"{r['access_improvement_twh']:.3f}"    if pd.notna(r.get("access_improvement_twh")) else "—"
        nec = "YES" if r.get("eaas_necessary") else "no"
        print(f"  {r['ndc_label']:<22} {r['demand_level']:<14} "
              f"{mw:>9} {ag:>16} {nec:>11}")

    print(f"\nSaved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
