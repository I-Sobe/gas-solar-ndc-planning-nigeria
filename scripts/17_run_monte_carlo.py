"""
17_run_monte_carlo.py  —  Monte Carlo Uncertainty Analysis
============================================================

For each of four headline cases, solves the LP once to obtain the
optimal capacity plan, then replays that plan under N draws from the
joint uncertainty space (demand growth × gas regime) using deterministic
dispatch. This produces cost, unserved energy, and solar utilisation
distributions.

This approach is appropriate because the capacity plan is budget-
constrained and therefore nearly invariant to demand/gas draws — the
cost variance comes from VoLL penalties on unserved energy, which
the deterministic replay captures correctly.

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

import copy
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
from src.optimize_experiments import (
    extract_planning_diagnostics,
    run_deterministic_scenario,
)
from src.utils import json_safe
from src.stochastic import compute_risk_metrics

CANONICAL_VOLL = "voll_mid"
N_DRAWS = 500
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

    print(f"\nMonte Carlo Uncertainty Analysis (Deterministic Replay)")
    print(f"  Draws:  {N_DRAWS}")
    print(f"  Cases:  {len(CASES)}")
    print(f"  Seed:   {SEED}\n")

    all_rows = []

    for case in CASES:
        print(f"  == {case['label']} ==")

        # ── Step 1: Solve LP once for optimal capacity plan ───────
        print(f"    Solving LP...")
        scenario = load_scenario(
            demand_level_case="served", demand_case="baseline",
            gas_deliverability_case="baseline",
            capital_case=case["capital_case"],
            solar_build_case="aggressive", land_case="loose",
            carbon_case="no_policy", start_year=2025, end_year=2045,
        )
        scenario["solar_min_build_mw_per_year"] = 100.0
        scenario["financing_regime"] = case["financing_regime"]
        scenario["required_margin"] = case["required_margin"]

        years = scenario["years"]
        caps = load_annual_caps(case["ndc_scenario"], years)
        solar_capex_tv = load_solar_capex_by_year(
            scenario_name="solar_low",
            start_year=int(years[0]), end_year=int(years[-1]),
        )

        m = build_model(
            scenario=scenario, econ=econ,
            emissions_cap_by_year=caps,
            solar_capex_by_year=solar_capex_tv,
        )
        status = solve_model(m)
        if not status["optimal"]:
            print(f"    WARNING: LP not optimal. Skipping.")
            continue

        capacity_paths = {
            "solar_mw": [float(pyo.value(m.solar_capacity_mw[t])) for t in range(len(years))],
            "storage_mwh": [float(pyo.value(m.storage_capacity_mwh[t])) for t in range(len(years))],
        }

        diag = extract_planning_diagnostics(m, scenario, econ)
        lp_cost = float(pyo.value(m.system_cost_npv))
        lp_unserved = sum(diag["unserved_twh_by_year"].values())
        lp_solar = sum(
            float(pyo.value(m.solar_public_add[t])) + float(pyo.value(m.solar_eaas_add[t]))
            for t in range(len(years))
        )
        print(f"    LP: cost=${lp_cost/1e9:.1f}B  unserved={lp_unserved:.2f}TWh  solar={lp_solar:.0f}MW")

        # ── Step 2: Replay under N draws ──────────────────────────
        print(f"    Replaying {N_DRAWS} draws...")
        for i, draw in enumerate(draws):
            draw_scenario = load_scenario(
                demand_level_case="served", demand_case="baseline",
                gas_deliverability_case=draw["gas_regime"],
                capital_case=case["capital_case"],
                solar_build_case="aggressive", land_case="loose",
                carbon_case="no_policy", start_year=2025, end_year=2045,
            )
            draw_scenario["demand_growth"] = draw["demand_growth"]
            draw_scenario["solar_min_build_mw_per_year"] = 100.0
            draw_scenario["financing_regime"] = case["financing_regime"]
            draw_scenario["required_margin"] = case["required_margin"]

            try:
                det_output = run_deterministic_scenario(
                    scenario=draw_scenario, econ=econ,
                    capacity_paths=capacity_paths,
                )
                all_rows.append({
                    "case": case["label"], "draw": i,
                    "demand_growth": draw["demand_growth"],
                    "gas_regime": draw["gas_regime"],
                    "status": "optimal",
                    "det_cost": det_output["costs"]["total"],
                    "det_unserved": float(np.sum(det_output["unserved"])),
                    "det_served": float(np.sum(det_output["served"])),
                    "lp_cost": lp_cost,
                    "lp_unserved": lp_unserved,
                    "lp_solar_mw": lp_solar,
                })
            except Exception as e:
                all_rows.append({
                    "case": case["label"], "draw": i,
                    "demand_growth": draw["demand_growth"],
                    "gas_regime": draw["gas_regime"],
                    "status": f"error: {e}",
                    "det_cost": None, "det_unserved": None,
                    "det_served": None,
                    "lp_cost": lp_cost, "lp_unserved": lp_unserved,
                    "lp_solar_mw": lp_solar,
                })

            if (i + 1) % 100 == 0:
                print(f"      ... {i+1}/{N_DRAWS}")

        print(f"    Done.")

    # ── Save ──────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "monte_carlo_raw.csv", index=False)

    optimal = df[df["status"] == "optimal"].copy()

    # ── Summary statistics ────────────────────────────────────────
    summary_rows = []
    for case_label in [c["label"] for c in CASES]:
        sub = optimal[optimal["case"] == case_label]
        if len(sub) == 0:
            continue
        costs = sub["det_cost"].values
        unserved = sub["det_unserved"].values
        risk = compute_risk_metrics(costs, alpha=0.95)
        summary_rows.append({
            "case": case_label, "n_draws": len(sub),
            "lp_cost": sub.iloc[0]["lp_cost"],
            "cost_mean": risk["expected"],
            "cost_std": np.sqrt(risk["variance"]),
            "cost_p10": np.percentile(costs, 10),
            "cost_p50": np.percentile(costs, 50),
            "cost_p90": np.percentile(costs, 90),
            "cost_var95": risk["VaR"], "cost_cvar95": risk["CVaR"],
            "unserved_mean": np.mean(unserved),
            "unserved_p10": np.percentile(unserved, 10),
            "unserved_p50": np.percentile(unserved, 50),
            "unserved_p90": np.percentile(unserved, 90),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "monte_carlo_summary.csv", index=False)

    # ── EaaS necessity ────────────────────────────────────────────
    eaas_necessity = {}
    for ndc_label in ["NDC3 uncond", "NDC3 cond"]:
        pub_label = f"{ndc_label} (public)"
        eaas_label = f"{ndc_label} (EaaS)"
        pub_sub = optimal[optimal["case"] == pub_label].set_index("draw")
        eaas_sub = optimal[optimal["case"] == eaas_label].set_index("draw")
        common = pub_sub.index.intersection(eaas_sub.index)
        if len(common) == 0:
            continue
        pub_u = pub_sub.loc[common, "det_unserved"].values
        eaas_u = eaas_sub.loc[common, "det_unserved"].values
        pub_c = pub_sub.loc[common, "det_cost"].values
        eaas_c = eaas_sub.loc[common, "det_cost"].values
        eaas_necessity[ndc_label] = {
            "prob_eaas_reduces_unserved": round(float(np.mean(eaas_u < pub_u)), 3),
            "prob_eaas_reduces_cost": round(float(np.mean(eaas_c < pub_c)), 3),
            "mean_savings_usd": round(float(np.mean(pub_c - eaas_c)), 0),
            "mean_savings_pct": round(float(np.mean(
                (pub_c - eaas_c) / np.where(pub_c > 0, pub_c, 1)) * 100), 1),
            "n_draws": int(len(common)),
        }
    with open(RESULTS_DIR / "eaas_necessity.json", "w") as f:
        json.dump(eaas_necessity, f, indent=2)

    # ── Console ───────────────────────────────────────────────────
    print(f"\n{'='*85}")
    print(f"  MONTE CARLO SUMMARY ({N_DRAWS} draws)")
    print(f"{'='*85}")
    print(f"  {'Case':<30} {'LP cost':>10} {'MC Mean':>10} {'P10':>10} "
          f"{'P50':>10} {'P90':>10} {'VaR95':>10}")
    print(f"  {'-'*85}")
    for _, r in summary_df.iterrows():
        print(f"  {r['case']:<30} ${r['lp_cost']/1e9:>8.1f}B ${r['cost_mean']/1e9:>8.1f}B "
              f"${r['cost_p10']/1e9:>8.1f}B ${r['cost_p50']/1e9:>8.1f}B "
              f"${r['cost_p90']/1e9:>8.1f}B ${r['cost_var95']/1e9:>8.1f}B")

    print(f"\n  UNSERVED ENERGY (TWh):")
    print(f"  {'Case':<30} {'Mean':>8} {'P10':>8} {'P50':>8} {'P90':>8}")
    print(f"  {'-'*60}")
    for _, r in summary_df.iterrows():
        print(f"  {r['case']:<30} {r['unserved_mean']:>8.1f} {r['unserved_p10']:>8.1f} "
              f"{r['unserved_p50']:>8.1f} {r['unserved_p90']:>8.1f}")

    print(f"\n  EaaS NECESSITY:")
    for ndc, m in eaas_necessity.items():
        print(f"    {ndc}: P(reduces unserved)={m['prob_eaas_reduces_unserved']*100:.0f}%  "
              f"P(reduces cost)={m['prob_eaas_reduces_cost']*100:.0f}%  "
              f"savings=${m['mean_savings_usd']/1e9:.1f}B ({m['mean_savings_pct']:.0f}%)")

    # ── Box plots ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    labels_short = [c["label"].replace(" (", "\n(") for c in CASES]
    colors = ["#FF5722", "#2196F3", "#FF9800", "#4CAF50"]

    cost_data = [optimal[optimal["case"] == c["label"]]["det_cost"].values / 1e9 for c in CASES]
    bp1 = axes[0].boxplot(cost_data, labels=labels_short, patch_artist=True)
    for p, col in zip(bp1["boxes"], colors):
        p.set_facecolor(col); p.set_alpha(0.7)
    axes[0].set_ylabel("Deterministic Cost (USD Billions)")
    axes[0].set_title("System Cost Distribution")
    axes[0].tick_params(axis="x", rotation=15, labelsize=8)

    uns_data = [optimal[optimal["case"] == c["label"]]["det_unserved"].values for c in CASES]
    bp2 = axes[1].boxplot(uns_data, labels=labels_short, patch_artist=True)
    for p, col in zip(bp2["boxes"], colors):
        p.set_facecolor(col); p.set_alpha(0.7)
    axes[1].set_ylabel("Unserved Energy (TWh)")
    axes[1].set_title("Reliability Distribution")
    axes[1].tick_params(axis="x", rotation=15, labelsize=8)

    plt.suptitle(f"Monte Carlo Uncertainty Analysis ({N_DRAWS} draws)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "monte_carlo_boxplots.png", dpi=200, bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "monte_carlo_boxplots.pdf", bbox_inches="tight")
    print(f"\nSaved: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
