"""
18_run_pareto_frontier.py  —  Cost-Emissions Pareto Frontier
==============================================================

Traces the cost-emissions Pareto frontier for Nigeria's power sector
by sweeping the emissions cap from unconstrained (baseline) to very
tight, under both public-only and EaaS financing at tight capital.

This answers MIX-2: "How does the optimal mix change across the
cost-emissions Pareto frontier, and how does EaaS shift it?"

The sweep produces 25 points per financing arm, recording at each:
    - NPV system cost (total, real expenditure, VoLL)
    - Cumulative emissions
    - Solar deployed (MW)
    - Storage deployed (MWh)
    - Unserved energy (TWh)
    - System LCOE (USD/MWh)

PRE-REQUISITES
--------------
    01_run_baseline.py      — baseline diagnostics
    00_build_emissions_cap.py — emissions cap trajectories

RUN SEQUENCE
------------
    python scripts/01_run_baseline.py
    python scripts/00_build_emissions_cap.py
    python scripts/18_run_pareto_frontier.py
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
from src.scenarios import load_scenario
from src.optimize_model import build_model, solve_model
from src.optimize_experiments import extract_planning_diagnostics
from src.utils import json_safe

CANONICAL_VOLL = "voll_mid"
RESULTS_DIR = ROOT / "results" / "pareto_frontier"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================

# Baseline emissions (will be read from baseline diagnostics)
BASELINE_DIAG_PATH = ROOT / "results" / "baseline" / "diagnostics.json"

# Number of Pareto points
N_POINTS = 25

# Financing arms to compare
FINANCING_ARMS = [
    {
        "label":            "public_only_tight",
        "financing_regime": "public",
        "capital_case":     "tight",
        "required_margin":  1.10,
    },
    {
        "label":            "eaas_tight",
        "financing_regime": "eaas",
        "capital_case":     "tight",
        "required_margin":  1.10,
    },
    {
        "label":            "public_only_moderate",
        "financing_regime": "public",
        "capital_case":     "moderate",
        "required_margin":  1.05,
    },
    {
        "label":            "eaas_moderate",
        "financing_regime": "eaas",
        "capital_case":     "moderate",
        "required_margin":  1.05,
    },
]


def main():
    # ── Load baseline emissions ───────────────────────────────
    with open(BASELINE_DIAG_PATH) as f:
        baseline_diag = json.load(f)

    baseline_emissions_by_year = baseline_diag["emissions_tco2_by_year"]
    years_str = sorted(baseline_emissions_by_year.keys())
    baseline_total = sum(baseline_emissions_by_year[y] for y in years_str)
    n_years = len(years_str)

    # Annual baseline emissions (roughly constant)
    baseline_annual = baseline_total / n_years

    print(f"\nPareto Frontier Sweep")
    print(f"  Baseline total emissions: {baseline_total/1e6:.1f} MtCO2")
    print(f"  Baseline annual: {baseline_annual/1e6:.3f} MtCO2/yr")
    print(f"  Points: {N_POINTS}")
    print(f"  Arms: {len(FINANCING_ARMS)}")
    print(f"  Total solves: {N_POINTS * len(FINANCING_ARMS)}")

    # ── Generate cap levels ───────────────────────────────────
    # Sweep from 100% of baseline (unconstrained) to 10% of baseline
    cap_fractions = np.linspace(1.0, 0.10, N_POINTS)

    econ = load_econ(CANONICAL_VOLL)

    all_rows = []

    for arm in FINANCING_ARMS:
        print(f"\n  == {arm['label']} ==")

        for i, frac in enumerate(cap_fractions):
            # Build uniform annual cap at this fraction of baseline
            annual_cap = baseline_annual * frac
            cap_by_year = [annual_cap] * n_years

            scenario = load_scenario(
                demand_level_case="served",
                demand_case="baseline",
                gas_deliverability_case="baseline",
                capital_case=arm["capital_case"],
                solar_build_case="aggressive",
                land_case="loose",
                carbon_case="no_policy",
                start_year=2025,
                end_year=2045,
            )
            scenario["solar_min_build_mw_per_year"] = 100.0
            scenario["financing_regime"] = arm["financing_regime"]
            scenario["required_margin"] = arm["required_margin"]

            years = scenario["years"]
            solar_capex_tv = load_solar_capex_by_year(
                scenario_name="solar_low",
                start_year=int(years[0]),
                end_year=int(years[-1]),
            )

            try:
                m = build_model(
                    scenario=scenario,
                    econ=econ,
                    emissions_cap_by_year=cap_by_year,
                    solar_capex_by_year=solar_capex_tv,
                )
                status = solve_model(m)

                if not status["optimal"]:
                    all_rows.append({
                        "arm": arm["label"], "cap_fraction": frac,
                        "annual_cap_tco2": annual_cap,
                        "status": "infeasible",
                    })
                    print(f"    {frac:.0%} cap → INFEASIBLE")
                    continue

                diag = extract_planning_diagnostics(m, scenario, econ)
                decomp = diag.get("cost_decomposition", {})

                total_cost = float(pyo.value(m.system_cost_npv))
                cum_emissions = sum(diag["emissions_tco2_by_year"].values())
                cum_unserved = sum(diag["unserved_twh_by_year"].values())
                cum_demand = sum(diag["demand_twh_by_year"].values())
                cum_generation = cum_demand - cum_unserved

                solar_total = sum(
                    float(pyo.value(m.solar_public_add[t]))
                    + float(pyo.value(m.solar_eaas_add[t]))
                    for t in range(len(years))
                )
                storage_final = float(pyo.value(m.storage_capacity_mwh[len(years) - 1]))

                # System LCOE: total discounted cost / total discounted generation
                disc_generation = sum(
                    float(pyo.value(m.DF[t])) * (
                        float(pyo.value(m.solar_generation[t]))
                        + float(pyo.value(m.gas_generation[t]))
                        + float(pyo.value(m.storage_discharge[t]))
                    )
                    for t in range(len(years))
                )
                system_lcoe = total_cost / (disc_generation * 1e6) if disc_generation > 0 else None

                all_rows.append({
                    "arm":              arm["label"],
                    "cap_fraction":     frac,
                    "annual_cap_tco2":  annual_cap,
                    "status":           "optimal",
                    "npv_cost":         total_cost,
                    "real_expenditure": decomp.get("real_expenditure_npv", None),
                    "voll_penalty":     decomp.get("voll_penalty_npv", None),
                    "cum_emissions":    cum_emissions,
                    "cum_unserved":     cum_unserved,
                    "solar_total_mw":   solar_total,
                    "storage_mwh":      storage_final,
                    "system_lcoe":      system_lcoe,
                    "gas_share_2030":   (
                        float(pyo.value(m.gas_generation[5]))
                        / max(float(pyo.value(m.solar_generation[5]))
                              + float(pyo.value(m.gas_generation[5]))
                              + float(pyo.value(m.storage_discharge[5])), 1e-9)
                    ),
                    "solar_share_2045": (
                        float(pyo.value(m.solar_generation[len(years)-1]))
                        / max(float(pyo.value(m.solar_generation[len(years)-1]))
                              + float(pyo.value(m.gas_generation[len(years)-1]))
                              + float(pyo.value(m.storage_discharge[len(years)-1])), 1e-9)
                    ),
                })

                print(f"    {frac:>4.0%} cap → cost=${total_cost/1e9:.1f}B  "
                      f"emit={cum_emissions/1e6:.1f}Mt  "
                      f"unserved={cum_unserved:.1f}TWh  "
                      f"solar={solar_total/1e3:.1f}GW")

            except Exception as e:
                all_rows.append({
                    "arm": arm["label"], "cap_fraction": frac,
                    "annual_cap_tco2": annual_cap, "status": f"error: {e}",
                })
                print(f"    {frac:.0%} cap → ERROR: {e}")

    # ── Save ──────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "pareto_raw.csv", index=False)

    optimal = df[df["status"] == "optimal"].copy()

    # ── Pareto frontier figure ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    arm_styles = {
        "public_only_tight":    ("#FF5722", "o", "Public (tight capital)"),
        "eaas_tight":           ("#2196F3", "s", "EaaS (tight capital)"),
        "public_only_moderate": ("#FF9800", "^", "Public (moderate capital)"),
        "eaas_moderate":        ("#4CAF50", "D", "EaaS (moderate capital)"),
    }

    # Panel 1: Cost vs Emissions
    ax = axes[0]
    for arm_label, (color, marker, legend_label) in arm_styles.items():
        sub = optimal[optimal["arm"] == arm_label].sort_values("cum_emissions")
        if len(sub):
            ax.plot(sub["cum_emissions"] / 1e6, sub["npv_cost"] / 1e9,
                    f"{marker}-", color=color, label=legend_label,
                    linewidth=2, markersize=6, alpha=0.8)
    ax.set_xlabel("Cumulative Emissions (MtCO2)")
    ax.set_ylabel("NPV System Cost (USD Billions)")
    ax.set_title("Cost-Emissions Pareto Frontier")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: Unserved vs Emissions
    ax = axes[1]
    for arm_label, (color, marker, legend_label) in arm_styles.items():
        sub = optimal[optimal["arm"] == arm_label].sort_values("cum_emissions")
        if len(sub):
            ax.plot(sub["cum_emissions"] / 1e6, sub["cum_unserved"],
                    f"{marker}-", color=color, label=legend_label,
                    linewidth=2, markersize=6, alpha=0.8)
    ax.set_xlabel("Cumulative Emissions (MtCO2)")
    ax.set_ylabel("Cumulative Unserved Energy (TWh)")
    ax.set_title("Reliability-Emissions Frontier")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle("Pareto Frontier: How Financing Structure Reshapes the Feasibility Space",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "pareto_frontier.png", dpi=200, bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "pareto_frontier.pdf", bbox_inches="tight")
    plt.show()

    # ── Optimal mix table at key cap levels ───────────────────
    key_fracs = [1.0, 0.80, 0.60, 0.40, 0.20]
    print(f"\n{'='*90}")
    print(f"  OPTIMAL GENERATION MIX AT KEY EMISSIONS LEVELS")
    print(f"{'='*90}")
    print(f"  {'Arm':<25} {'Cap':>5} {'Cost($B)':>10} {'Emit(Mt)':>10} "
          f"{'Solar(GW)':>10} {'Stor(GWh)':>10} {'Unserved':>10} {'LCOE':>8}")
    print(f"  {'-'*88}")

    for arm_label in arm_styles:
        for frac in key_fracs:
            row = optimal[
                (optimal["arm"] == arm_label)
                & (np.abs(optimal["cap_fraction"] - frac) < 0.02)
            ]
            if len(row):
                r = row.iloc[0]
                lcoe_str = f"${r['system_lcoe']:.1f}" if r['system_lcoe'] else "n/a"
                print(f"  {arm_label:<25} {frac:>4.0%} "
                      f"${r['npv_cost']/1e9:>8.1f}B "
                      f"{r['cum_emissions']/1e6:>9.1f} "
                      f"{r['solar_total_mw']/1e3:>9.1f} "
                      f"{r['storage_mwh']/1e3:>9.1f} "
                      f"{r['cum_unserved']:>9.1f} "
                      f"{lcoe_str:>8}")

    print(f"\nSaved: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
