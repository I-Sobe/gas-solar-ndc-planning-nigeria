"""
17_run_monte_carlo.py  —  LP-Based Monte Carlo Uncertainty Analysis
=====================================================================

For each of four headline cases, re-solves the LP under N draws from
the joint uncertainty space (demand growth × gas regime). Produces
properly discounted cost, LCOE, unserved energy, and solar deployment
distributions.

Unlike the deterministic replay approach, this script re-optimises the
full LP for each draw, ensuring:
    - Capacity plans adapt to each gas/demand realisation
    - Cost accounting uses discounted NPV (consistent with deterministic results)
    - LCOE is properly computed as NPV_cost / NPV_generation
    - VoLL decomposition is available per draw

UNCERTAINTY DIMENSIONS
-----------------------
    Demand growth:   Normal(mu=0.04, sigma=0.01), truncated at 0.005
    Gas regime:      Categorical draw from gas_probability_weights()
                     {baseline:0.50, downside:0.25, upside:0.20, shock_recovery:0.05}

PRE-REQUISITES
--------------
    01_run_baseline.py
    00_build_emissions_cap.py

RUN SEQUENCE
------------
    python scripts/01_run_baseline.py
    python scripts/00_build_emissions_cap.py
    python scripts/17_run_monte_carlo.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io import load_econ, load_solar_capex_by_year
from src.scenarios import load_scenario, gas_probability_weights
from src.optimize_model import build_model, solve_model
from src.optimize_experiments import extract_planning_diagnostics
from src.utils import json_safe

CANONICAL_VOLL = "voll_mid"
N_DRAWS = 200       # 200 draws × 4 cases = 800 LP solves (~30 seconds)
SEED = 42

RESULTS_DIR = ROOT / "results" / "monte_carlo"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAP_PATH = ROOT / "data" / "cost" / "processed" / "emissions_cap.csv"


def load_annual_caps(scenario_name, years):
    if not CAP_PATH.exists():
        raise FileNotFoundError(f"Missing {CAP_PATH}.")
    df = pd.read_csv(CAP_PATH)
    df = df[df["scenario"] == scenario_name].copy().sort_values("year")
    df = df[df["year"].isin([int(y) for y in years])].sort_values("year")
    caps = df["cap_tco2"].astype(float).tolist()
    if len(caps) != len(years):
        raise ValueError(f"Cap length {len(caps)} != years {len(years)}.")
    return caps


CASES = [
    {"label": "NDC3 uncond (public)", "ndc_scenario": "ndc3_unconditional",
     "capital_case": "tight", "financing_regime": "public", "required_margin": 1.10},
    {"label": "NDC3 uncond (EaaS)", "ndc_scenario": "ndc3_unconditional",
     "capital_case": "tight", "financing_regime": "eaas", "required_margin": 1.10},
    {"label": "NDC3 cond (public)", "ndc_scenario": "ndc3_conditional",
     "capital_case": "moderate", "financing_regime": "public", "required_margin": 1.05},
    {"label": "NDC3 cond (EaaS)", "ndc_scenario": "ndc3_conditional",
     "capital_case": "moderate", "financing_regime": "eaas", "required_margin": 1.05},
]


def generate_draws(N, seed=42):
    rng = np.random.RandomState(seed)
    gas_probs = gas_probability_weights()
    gas_labels = list(gas_probs.keys())
    gas_weights = list(gas_probs.values())
    draws = []
    for _ in range(N):
        dg = max(0.005, rng.normal(0.04, 0.01))
        gr = rng.choice(gas_labels, p=gas_weights)
        draws.append({"demand_growth": dg, "gas_regime": gr})
    return draws


def main():
    draws = generate_draws(N_DRAWS, seed=SEED)
    econ = load_econ(CANONICAL_VOLL)

    total_solves = N_DRAWS * len(CASES)
    print(f"\nMonte Carlo LP-Based Uncertainty Analysis")
    print(f"  Draws:  {N_DRAWS}")
    print(f"  Cases:  {len(CASES)}")
    print(f"  Total LP solves: {total_solves}")
    print(f"  Seed:   {SEED}\n")

    all_rows = []

    for case in CASES:
        print(f"  == {case['label']} ==")

        for i, draw in enumerate(draws):
            scenario = load_scenario(
                demand_level_case="served",
                demand_case="baseline",
                gas_deliverability_case=draw["gas_regime"],
                capital_case=case["capital_case"],
                solar_build_case="aggressive",
                land_case="loose",
                carbon_case="no_policy",
                start_year=2025,
                end_year=2045,
            )
            scenario["demand_growth"] = draw["demand_growth"]
            scenario["solar_min_build_mw_per_year"] = 100.0
            scenario["financing_regime"] = case["financing_regime"]
            scenario["required_margin"] = case["required_margin"]

            years = scenario["years"]
            caps = load_annual_caps(case["ndc_scenario"], years)
            solar_capex_tv = load_solar_capex_by_year(
                scenario_name="solar_low",
                start_year=int(years[0]),
                end_year=int(years[-1]),
            )

            try:
                m = build_model(
                    scenario=scenario,
                    econ=econ,
                    emissions_cap_by_year=caps,
                    solar_capex_by_year=solar_capex_tv,
                )
                status = solve_model(m)

                if not status["optimal"]:
                    all_rows.append({
                        "case": case["label"], "draw": i,
                        "demand_growth": draw["demand_growth"],
                        "gas_regime": draw["gas_regime"],
                        "status": "infeasible",
                        "npv_cost": None, "real_exp": None,
                        "voll_penalty": None, "unserved_twh": None,
                        "solar_total_mw": None, "storage_mwh": None,
                        "system_lcoe": None,
                    })
                    continue

                diag = extract_planning_diagnostics(m, scenario, econ)
                decomp = diag.get("cost_decomposition", {})

                npv_cost = float(pyo.value(m.system_cost_npv))
                unserved = sum(diag["unserved_twh_by_year"].values())
                solar_total = sum(
                    float(pyo.value(m.solar_public_add[t]))
                    + float(pyo.value(m.solar_eaas_add[t]))
                    for t in range(len(years))
                )
                storage = float(pyo.value(m.storage_capacity_mwh[len(years) - 1]))

                # Proper discounted LCOE
                disc_gen = sum(
                    float(pyo.value(m.DF[t])) * (
                        float(pyo.value(m.solar_generation[t]))
                        + float(pyo.value(m.gas_generation[t]))
                        + float(pyo.value(m.storage_discharge[t]))
                    )
                    for t in range(len(years))
                )
                # disc_gen is in TWh (discounted), cost is in USD
                system_lcoe = npv_cost / (disc_gen * 1e6) if disc_gen > 1e-9 else None

                all_rows.append({
                    "case": case["label"], "draw": i,
                    "demand_growth": draw["demand_growth"],
                    "gas_regime": draw["gas_regime"],
                    "status": "optimal",
                    "npv_cost": npv_cost,
                    "real_exp": decomp.get("real_expenditure_npv", None),
                    "voll_penalty": decomp.get("voll_penalty_npv", None),
                    "unserved_twh": unserved,
                    "solar_total_mw": solar_total,
                    "storage_mwh": storage,
                    "system_lcoe": system_lcoe,
                })

            except Exception as e:
                all_rows.append({
                    "case": case["label"], "draw": i,
                    "demand_growth": draw["demand_growth"],
                    "gas_regime": draw["gas_regime"],
                    "status": f"error: {e}",
                    "npv_cost": None, "real_exp": None,
                    "voll_penalty": None, "unserved_twh": None,
                    "solar_total_mw": None, "storage_mwh": None,
                    "system_lcoe": None,
                })

            if (i + 1) % 50 == 0:
                print(f"    ... {i+1}/{N_DRAWS} draws")

        print(f"    Done: {case['label']}")

    # ── Save raw results ──────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "monte_carlo_raw.csv", index=False)

    optimal = df[df["status"] == "optimal"].copy()

    # ── Summary statistics ────────────────────────────────────
    summary_rows = []
    for case_label in [c["label"] for c in CASES]:
        sub = optimal[optimal["case"] == case_label]
        if len(sub) == 0:
            continue

        costs = sub["npv_cost"].values
        unserved = sub["unserved_twh"].values
        lcoes = sub["system_lcoe"].dropna().values

        summary_rows.append({
            "case":          case_label,
            "n_optimal":     len(sub),
            "n_infeasible":  len(df[(df["case"] == case_label) & (df["status"] != "optimal")]),
            "cost_mean":     np.mean(costs),
            "cost_std":      np.std(costs),
            "cost_p10":      np.percentile(costs, 10),
            "cost_p50":      np.percentile(costs, 50),
            "cost_p90":      np.percentile(costs, 90),
            "unserved_mean": np.mean(unserved),
            "unserved_p10":  np.percentile(unserved, 10),
            "unserved_p50":  np.percentile(unserved, 50),
            "unserved_p90":  np.percentile(unserved, 90),
            "lcoe_mean":     np.mean(lcoes) if len(lcoes) else None,
            "lcoe_p10":      np.percentile(lcoes, 10) if len(lcoes) else None,
            "lcoe_p50":      np.percentile(lcoes, 50) if len(lcoes) else None,
            "lcoe_p90":      np.percentile(lcoes, 90) if len(lcoes) else None,
            "real_exp_mean": np.mean(sub["real_exp"].dropna().values),
            "voll_mean":     np.mean(sub["voll_penalty"].dropna().values),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "monte_carlo_summary.csv", index=False)

    # ── EaaS necessity ────────────────────────────────────────
    eaas_necessity = {}
    for ndc_label in ["NDC3 uncond", "NDC3 cond"]:
        pub_label = f"{ndc_label} (public)"
        eaas_label = f"{ndc_label} (EaaS)"
        pub_sub = optimal[optimal["case"] == pub_label].set_index("draw")
        eaas_sub = optimal[optimal["case"] == eaas_label].set_index("draw")
        common = pub_sub.index.intersection(eaas_sub.index)
        if len(common) == 0:
            continue
        pub_u = pub_sub.loc[common, "unserved_twh"].values
        eaas_u = eaas_sub.loc[common, "unserved_twh"].values
        pub_c = pub_sub.loc[common, "npv_cost"].values
        eaas_c = eaas_sub.loc[common, "npv_cost"].values
        pub_l = pub_sub.loc[common, "system_lcoe"].dropna().values
        eaas_l = eaas_sub.loc[common, "system_lcoe"].dropna().values

        eaas_necessity[ndc_label] = {
            "prob_eaas_reduces_unserved": round(float(np.mean(eaas_u < pub_u)), 3),
            "prob_eaas_reduces_cost": round(float(np.mean(eaas_c < pub_c)), 3),
            "mean_cost_savings_usd": round(float(np.mean(pub_c - eaas_c)), 0),
            "mean_cost_savings_pct": round(float(np.mean(
                (pub_c - eaas_c) / np.where(pub_c > 0, pub_c, 1)) * 100), 1),
            "mean_lcoe_pub": round(float(np.mean(pub_l)), 2) if len(pub_l) else None,
            "mean_lcoe_eaas": round(float(np.mean(eaas_l)), 2) if len(eaas_l) else None,
            "n_draws": int(len(common)),
        }

    with open(RESULTS_DIR / "eaas_necessity.json", "w") as f:
        json.dump(eaas_necessity, f, indent=2)

    # ── Console summary ───────────────────────────────────────
    print(f"\n{'='*95}")
    print(f"  MONTE CARLO SUMMARY ({N_DRAWS} draws, LP-based)")
    print(f"{'='*95}")
    print(f"  {'Case':<30} {'Mean($B)':>10} {'P10($B)':>10} {'P50($B)':>10} "
          f"{'P90($B)':>10} {'LCOE mean':>10} {'LCOE P90':>10}")
    print(f"  {'-'*92}")
    for _, r in summary_df.iterrows():
        lcoe_m = f"${r['lcoe_mean']:.1f}" if r['lcoe_mean'] else "n/a"
        lcoe_90 = f"${r['lcoe_p90']:.1f}" if r['lcoe_p90'] else "n/a"
        print(f"  {r['case']:<30} ${r['cost_mean']/1e9:>8.1f}B ${r['cost_p10']/1e9:>8.1f}B "
              f"${r['cost_p50']/1e9:>8.1f}B ${r['cost_p90']/1e9:>8.1f}B "
              f"{lcoe_m:>10} {lcoe_90:>10}")

    print(f"\n  UNSERVED ENERGY (TWh):")
    print(f"  {'Case':<30} {'Mean':>8} {'P10':>8} {'P50':>8} {'P90':>8}")
    print(f"  {'-'*60}")
    for _, r in summary_df.iterrows():
        print(f"  {r['case']:<30} {r['unserved_mean']:>8.1f} {r['unserved_p10']:>8.1f} "
              f"{r['unserved_p50']:>8.1f} {r['unserved_p90']:>8.1f}")

    print(f"\n  EaaS NECESSITY:")
    for ndc, met in eaas_necessity.items():
        print(f"    {ndc}:")
        print(f"      P(reduces unserved): {met['prob_eaas_reduces_unserved']*100:.0f}%")
        print(f"      P(reduces cost):     {met['prob_eaas_reduces_cost']*100:.0f}%")
        print(f"      Mean savings:        ${met['mean_cost_savings_usd']/1e9:.1f}B "
              f"({met['mean_cost_savings_pct']:.0f}%)")
        if met['mean_lcoe_pub'] and met['mean_lcoe_eaas']:
            print(f"      LCOE public:         ${met['mean_lcoe_pub']:.1f}/MWh")
            print(f"      LCOE EaaS:           ${met['mean_lcoe_eaas']:.1f}/MWh")

    # ── Box plots ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    labels_short = [c["label"].replace(" (", "\n(") for c in CASES]
    colors = ["#FF5722", "#2196F3", "#FF9800", "#4CAF50"]

    # Panel 1: NPV Cost
    cost_data = [optimal[optimal["case"] == c["label"]]["npv_cost"].dropna().values / 1e9
                 for c in CASES]
    bp1 = axes[0].boxplot(cost_data, tick_labels=labels_short, patch_artist=True)
    for p, col in zip(bp1["boxes"], colors):
        p.set_facecolor(col); p.set_alpha(0.7)
    axes[0].set_ylabel("NPV System Cost (USD Billions)")
    axes[0].set_title("System Cost Distribution")
    axes[0].tick_params(axis="x", rotation=15, labelsize=8)

    # Panel 2: Unserved
    uns_data = [optimal[optimal["case"] == c["label"]]["unserved_twh"].dropna().values
                for c in CASES]
    bp2 = axes[1].boxplot(uns_data, tick_labels=labels_short, patch_artist=True)
    for p, col in zip(bp2["boxes"], colors):
        p.set_facecolor(col); p.set_alpha(0.7)
    axes[1].set_ylabel("Unserved Energy (TWh)")
    axes[1].set_title("Reliability Distribution")
    axes[1].tick_params(axis="x", rotation=15, labelsize=8)

    # Panel 3: LCOE
    lcoe_data = [optimal[optimal["case"] == c["label"]]["system_lcoe"].dropna().values
                 for c in CASES]
    bp3 = axes[2].boxplot(lcoe_data, tick_labels=labels_short, patch_artist=True)
    for p, col in zip(bp3["boxes"], colors):
        p.set_facecolor(col); p.set_alpha(0.7)
    axes[2].set_ylabel("System LCOE (USD/MWh)")
    axes[2].set_title("LCOE Distribution")
    axes[2].tick_params(axis="x", rotation=15, labelsize=8)

    plt.suptitle(f"Monte Carlo Uncertainty Analysis ({N_DRAWS} draws, LP-based)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "monte_carlo_boxplots.png", dpi=200, bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "monte_carlo_boxplots.pdf", bbox_inches="tight")
    plt.show()
    print(f"\nSaved: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
