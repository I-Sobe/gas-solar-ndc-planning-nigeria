"""
19_run_carbon_price_sweep.py  —  Carbon Price Tipping Point
=============================================================

Sweeps carbon price from $0 to $150/tCO2 to identify the price at
which solar becomes cost-dominant over gas WITHOUT any emissions cap.

This answers MIX-3: "At what carbon price does solar become
cost-dominant, and how does EaaS affect this threshold?"

The sweep runs under NO emissions cap — the carbon price enters the
objective function as a cost on gas-fired emissions. As the price
rises, the optimizer shifts from gas to solar purely on cost grounds.

The "tipping point" is defined as the carbon price at which:
    - Solar provides >50% of total generation in 2035
    - OR gas generation falls below 50% of baseline gas generation

PRE-REQUISITES
--------------
    01_run_baseline.py

RUN SEQUENCE
------------
    python scripts/01_run_baseline.py
    python scripts/19_run_carbon_price_sweep.py
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
RESULTS_DIR = ROOT / "results" / "carbon_price_sweep"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================

# Carbon prices to sweep (USD/tCO2)
CARBON_PRICES = [0, 5, 10, 15, 20, 30, 40, 50, 60, 75, 100, 125, 150]

FINANCING_ARMS = [
    {
        "label":            "public_unconstrained",
        "financing_regime": "public",
        "capital_case":     "unconstrained",
        "required_margin":  1.10,
    },
    {
        "label":            "public_tight",
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
]


def main():
    econ = load_econ(CANONICAL_VOLL)

    total_solves = len(CARBON_PRICES) * len(FINANCING_ARMS)
    print(f"\nCarbon Price Sweep")
    print(f"  Prices: {CARBON_PRICES} USD/tCO2")
    print(f"  Arms: {len(FINANCING_ARMS)}")
    print(f"  Total solves: {total_solves}")

    all_rows = []

    for arm in FINANCING_ARMS:
        print(f"\n  == {arm['label']} ==")

        for cp in CARBON_PRICES:
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

            # Activate carbon pricing
            if cp > 0:
                scenario["carbon_active"] = True
                scenario["carbon_price"] = float(cp)
            else:
                scenario["carbon_active"] = False
                scenario["carbon_price"] = 0.0

            years = scenario["years"]
            solar_capex_tv = load_solar_capex_by_year(
                scenario_name="solar_low",
                start_year=int(years[0]),
                end_year=int(years[-1]),
            )

            try:
                # NO emissions cap — carbon price drives the transition
                m = build_model(
                    scenario=scenario,
                    econ=econ,
                    emissions_cap=1e18,
                    solar_capex_by_year=solar_capex_tv,
                )
                status = solve_model(m)

                if not status["optimal"]:
                    all_rows.append({
                        "arm": arm["label"], "carbon_price": cp,
                        "status": "infeasible",
                    })
                    print(f"    ${cp:>3}/tCO2 → INFEASIBLE")
                    continue

                diag = extract_planning_diagnostics(m, scenario, econ)
                decomp = diag.get("cost_decomposition", {})

                total_cost = float(pyo.value(m.system_cost_npv))
                cum_emissions = sum(diag["emissions_tco2_by_year"].values())
                cum_unserved = sum(diag["unserved_twh_by_year"].values())

                solar_total = sum(
                    float(pyo.value(m.solar_public_add[t]))
                    + float(pyo.value(m.solar_eaas_add[t]))
                    for t in range(len(years))
                )
                storage_final = float(pyo.value(m.storage_capacity_mwh[len(years) - 1]))

                # Generation shares at 2030 (index 5) and 2035 (index 10)
                def gen_shares(t_idx):
                    solar_g = float(pyo.value(m.solar_generation[t_idx]))
                    gas_g = float(pyo.value(m.gas_generation[t_idx]))
                    stor_g = float(pyo.value(m.storage_discharge[t_idx]))
                    total_g = solar_g + gas_g + stor_g
                    if total_g < 1e-9:
                        return 0, 0, 0
                    return solar_g / total_g, gas_g / total_g, stor_g / total_g

                solar_share_2030, gas_share_2030, _ = gen_shares(5)
                solar_share_2035, gas_share_2035, _ = gen_shares(10)
                solar_share_2045, gas_share_2045, _ = gen_shares(len(years) - 1)

                # System LCOE
                disc_gen = sum(
                    float(pyo.value(m.DF[t])) * (
                        float(pyo.value(m.solar_generation[t]))
                        + float(pyo.value(m.gas_generation[t]))
                        + float(pyo.value(m.storage_discharge[t]))
                    )
                    for t in range(len(years))
                )
                system_lcoe = total_cost / (disc_gen * 1e6) if disc_gen > 0 else None

                all_rows.append({
                    "arm":              arm["label"],
                    "carbon_price":     cp,
                    "status":           "optimal",
                    "npv_cost":         total_cost,
                    "real_expenditure": decomp.get("real_expenditure_npv", None),
                    "carbon_cost":      decomp.get("components", {}).get("carbon_cost_npv", 0),
                    "cum_emissions":    cum_emissions,
                    "cum_unserved":     cum_unserved,
                    "solar_total_mw":   solar_total,
                    "storage_mwh":      storage_final,
                    "system_lcoe":      system_lcoe,
                    "solar_share_2030": solar_share_2030,
                    "gas_share_2030":   gas_share_2030,
                    "solar_share_2035": solar_share_2035,
                    "gas_share_2035":   gas_share_2035,
                    "solar_share_2045": solar_share_2045,
                    "gas_share_2045":   gas_share_2045,
                })

                print(f"    ${cp:>3}/tCO2 → cost=${total_cost/1e9:.1f}B  "
                      f"emit={cum_emissions/1e6:.1f}Mt  "
                      f"solar2035={solar_share_2035:.0%}  "
                      f"gas2035={gas_share_2035:.0%}")

            except Exception as e:
                all_rows.append({
                    "arm": arm["label"], "carbon_price": cp,
                    "status": f"error: {e}",
                })
                print(f"    ${cp:>3}/tCO2 → ERROR: {e}")

    # ── Save ──────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "carbon_price_raw.csv", index=False)

    optimal = df[df["status"] == "optimal"].copy()

    # ── Identify tipping points ───────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CARBON PRICE TIPPING POINTS")
    print(f"{'='*70}")

    for arm_label in [a["label"] for a in FINANCING_ARMS]:
        sub = optimal[optimal["arm"] == arm_label].sort_values("carbon_price")
        if len(sub) == 0:
            continue

        # Tipping point: first carbon price where solar > 50% in 2035
        tipping = sub[sub["solar_share_2035"] > 0.50]
        if len(tipping):
            tp = tipping.iloc[0]["carbon_price"]
            print(f"  {arm_label:<25} Solar >50% in 2035 at ${tp:.0f}/tCO2")
        else:
            print(f"  {arm_label:<25} Solar never reaches 50% in 2035 at tested prices")

    # ── Figures ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    arm_styles = {
        "public_unconstrained": ("#9E9E9E", "o", "Public (unconstrained)"),
        "public_tight":         ("#FF5722", "s", "Public (tight capital)"),
        "eaas_tight":           ("#2196F3", "^", "EaaS (tight capital)"),
    }

    # Panel 1: Cost vs carbon price
    ax = axes[0, 0]
    for arm_label, (color, marker, legend) in arm_styles.items():
        sub = optimal[optimal["arm"] == arm_label].sort_values("carbon_price")
        if len(sub):
            ax.plot(sub["carbon_price"], sub["npv_cost"] / 1e9,
                    f"{marker}-", color=color, label=legend, linewidth=2, markersize=6)
    ax.set_xlabel("Carbon Price (USD/tCO2)")
    ax.set_ylabel("NPV System Cost ($B)")
    ax.set_title("System Cost vs Carbon Price")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: Emissions vs carbon price
    ax = axes[0, 1]
    for arm_label, (color, marker, legend) in arm_styles.items():
        sub = optimal[optimal["arm"] == arm_label].sort_values("carbon_price")
        if len(sub):
            ax.plot(sub["carbon_price"], sub["cum_emissions"] / 1e6,
                    f"{marker}-", color=color, label=legend, linewidth=2, markersize=6)
    ax.set_xlabel("Carbon Price (USD/tCO2)")
    ax.set_ylabel("Cumulative Emissions (MtCO2)")
    ax.set_title("Emissions vs Carbon Price")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 3: Solar share 2035 vs carbon price
    ax = axes[1, 0]
    for arm_label, (color, marker, legend) in arm_styles.items():
        sub = optimal[optimal["arm"] == arm_label].sort_values("carbon_price")
        if len(sub):
            ax.plot(sub["carbon_price"], sub["solar_share_2035"] * 100,
                    f"{marker}-", color=color, label=legend, linewidth=2, markersize=6)
    ax.axhline(y=50, color="black", linestyle="--", alpha=0.5, label="50% threshold")
    ax.set_xlabel("Carbon Price (USD/tCO2)")
    ax.set_ylabel("Solar Share of Generation 2035 (%)")
    ax.set_title("Solar Penetration vs Carbon Price (2035)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 4: LCOE vs carbon price
    ax = axes[1, 1]
    for arm_label, (color, marker, legend) in arm_styles.items():
        sub = optimal[optimal["arm"] == arm_label].sort_values("carbon_price")
        sub_valid = sub[sub["system_lcoe"].notna()]
        if len(sub_valid):
            ax.plot(sub_valid["carbon_price"], sub_valid["system_lcoe"],
                    f"{marker}-", color=color, label=legend, linewidth=2, markersize=6)
    ax.set_xlabel("Carbon Price (USD/tCO2)")
    ax.set_ylabel("System LCOE (USD/MWh)")
    ax.set_title("System LCOE vs Carbon Price")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle("Carbon Price Sweep: Tipping Points for Nigeria's Gas-to-Solar Transition",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "carbon_price_sweep.png", dpi=200, bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "carbon_price_sweep.pdf", bbox_inches="tight")
    plt.show()

    # ── Console summary table ─────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  GENERATION MIX BY CARBON PRICE")
    print(f"{'='*90}")
    print(f"  {'Arm':<25} {'CP':>4} {'Cost($B)':>10} {'Emit(Mt)':>10} "
          f"{'Solar35':>8} {'Gas35':>8} {'LCOE':>8}")
    print(f"  {'-'*78}")

    for arm_label in arm_styles:
        sub = optimal[optimal["arm"] == arm_label].sort_values("carbon_price")
        for _, r in sub.iterrows():
            lcoe = f"${r['system_lcoe']:.1f}" if r['system_lcoe'] else "n/a"
            print(f"  {arm_label:<25} ${r['carbon_price']:>3} "
                  f"${r['npv_cost']/1e9:>8.1f}B "
                  f"{r['cum_emissions']/1e6:>9.1f} "
                  f"{r['solar_share_2035']:>7.0%} "
                  f"{r['gas_share_2035']:>7.0%} "
                  f"{lcoe:>8}")

    print(f"\nSaved: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
