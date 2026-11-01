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

**This MIX-3 tipping-point definition is unchanged by TASK 04** (docs/
task_log_04_carbon_selfgen.md) -- the grid generation-mix test above is
the script's research question and stays exactly as it is. TASK 04 adds
observables the tipping-point test cannot see: self-generation (diesel)
volume and emissions, which `gen_shares()` and `cum_emissions` excluded
or conflated respectively. See `cum_selfgen_twh`, `selfgen_twh_2035`,
`cum_emissions_grid`, `cum_emissions_selfgen`, `selfgen_by_year.csv`, and
the displacement-price summary below.

PRE-REQUISITES
--------------
    01_run_baseline.py

RUN SEQUENCE
------------
    python scripts/01_run_baseline.py
    python scripts/19_run_carbon_price_sweep.py
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io import load_econ, load_solar_capex_by_year
from src.scenarios import load_scenario, asset_lifetime_sweep, solar_min_build_default, MODEL_END_YEAR
from src.optimize_model import solve_model
from src.optimize_model_sliced import build_model_sliced
from src.optimize_experiments import extract_planning_diagnostics
from src.utils import json_safe
from src.cli_arms import add_standard_arms, standard_arms_kwargs, resolve_results_dir

CANONICAL_VOLL = "voll_mid"
CANONICAL_GENSET = "genset_central"
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
    },
    {
        "label":            "public_tight",
        "financing_regime": "public",
        "capital_case":     "tight",
    },
    {
        "label":            "eaas_tight",
        "financing_regime": "eaas",
        "capital_case":     "tight",
    },
]


def parse_args():
    # docs/task_runner_repair.md, Step Three: this script had NO argparse
    # at all before this edit -- its load_scenario() call below either sat
    # on a hardcoded literal (gas_deliverability_case="baseline",
    # om_case="om_central") or omitted a sentinel-required key outright,
    # which would have raised a TypeError before ever reaching
    # build_model(). Built fresh, add_standard_arms(parser) for the full
    # set; capital_case stays sourced from FINANCING_ARMS (the runner's
    # own subject), not a flag, matching 01b_'s own convention.
    parser = argparse.ArgumentParser(
        description="19_run_carbon_price_sweep: carbon price tipping point (MIX-3)"
    )
    add_standard_arms(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    global RESULTS_DIR
    RESULTS_DIR = resolve_results_dir(RESULTS_DIR, args)   # TASK 53 2.5: arm-keyed output directory
    standard_kwargs = standard_arms_kwargs(args)

    econ = load_econ(CANONICAL_VOLL, genset_case=CANONICAL_GENSET)

    total_solves = len(CARBON_PRICES) * len(FINANCING_ARMS)
    print(f"\nCarbon Price Sweep")
    print(f"  Prices: {CARBON_PRICES} USD/tCO2")
    print(f"  Arms: {len(FINANCING_ARMS)}")
    print(f"  Total solves: {total_solves}")

    all_rows = []
    year_rows = []  # TASK 04, Section A: selfgen_by_year.csv

    for arm in FINANCING_ARMS:
        print(f"\n  == {arm['label']} ==")

        for cp in CARBON_PRICES:
            scenario = load_scenario(
                demand_level_case="served",
                demand_case="organic_central",
                capital_case=arm["capital_case"],
                solar_build_case="deployment_unconstrained",
                land_case="loose",
                carbon_case="no_policy",
                start_year=2025,
                end_year=MODEL_END_YEAR,
                **standard_kwargs,
            )
            scenario["solar_min_build_mw_per_year"] = solar_min_build_default()
            scenario["financing_regime"] = arm["financing_regime"]
            # TASK 29: no required_margin override (dscr_case-derived; L34).

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
                # NO emissions cap — carbon price drives the transition.
                # Plan 0.4, item C: 1e18 sentinel -> None ("no cap",
                # already the builder's own guarded meaning here).
                m = build_model_sliced(
                    scenario=scenario,
                    econ=econ,
                    emissions_cap=None,
                    solar_capex_by_year=solar_capex_tv,
                )
                status = solve_model(m, scenario=scenario)

                if not status["optimal"]:
                    all_rows.append({
                        "arm": arm["label"], "carbon_price": cp,
                        "status": "infeasible",
                    })
                    print(f"    ${cp:>3}/tCO2 -> INFEASIBLE")
                    continue

                diag = extract_planning_diagnostics(m, scenario, econ)
                decomp = diag.get("cost_decomposition", {})

                total_cost = float(pyo.value(m.system_cost_npv))
                cum_emissions = sum(diag["emissions_tco2_by_year"].values())
                cum_unserved = sum(diag["unserved_twh_by_year"].values())
                # TASK 04, Section A (docs/task_log_04_carbon_selfgen.md):
                # direct counterpart to cum_unserved above -- same route
                # (diag's own by-year dict), same style.
                cum_selfgen_twh = sum(diag["selfgen_twh_by_year"].values())

                # TASK 04, Section B: emissions decomposition. Route 1 --
                # the builder already exposes grid-only and selfgen-only
                # emissions as their own Expressions (src/optimize_model_
                # sliced.py: m.emissions_grid_only, m.emissions_selfgen,
                # both summed from m.emissions_{grid_only,selfgen}_by_year;
                # m.emissions_by_year[t] is defined AS their per-year sum,
                # "ALWAYS total -- no longer switched by
                # selfgen_in_emissions_cap"). No derivation, no new
                # parameter, no hardcoded factor -- both read directly.
                cum_emissions_grid = float(pyo.value(m.emissions_grid_only))
                cum_emissions_selfgen = float(pyo.value(m.emissions_selfgen))
                # Assertion: confirms cum_emissions (independently summed
                # from diag["emissions_tco2_by_year"], itself read from
                # m.emissions_by_year) agrees with the two components read
                # from their OWN separate Expressions. Since m.emissions_
                # by_year[t] is defined as grid_only_by_year[t] +
                # selfgen_by_year[t] in the builder, this must hold to
                # floating-point summation-order tolerance, not as a
                # derived check of a fitted factor (Route 1 has none).
                assert math.isclose(
                    cum_emissions, cum_emissions_grid + cum_emissions_selfgen,
                    rel_tol=1e-9,
                ), (
                    f"TASK 04 assertion failed: cum_emissions ({cum_emissions}) "
                    f"!= cum_emissions_grid + cum_emissions_selfgen "
                    f"({cum_emissions_grid} + {cum_emissions_selfgen} = "
                    f"{cum_emissions_grid + cum_emissions_selfgen}) at "
                    f"carbon_price={cp}, arm={arm['label']!r}."
                )

                solar_total = sum(
                    float(pyo.value(m.solar_public_add[t]))
                    + float(pyo.value(m.solar_eaas_add[t]))
                    for t in range(len(years))
                )
                storage_final = float(pyo.value(m.storage_capacity_mwh[len(years) - 1]))

                # Generation shares at 2030 (index 5) and 2035 (index 10)
                def gen_shares(t_idx):
                    # Sliced port (A5, Group B): m.solar_generation/gas_generation/
                    # storage_discharge are [T, m.S]-indexed under build_model_sliced
                    # (zero at night/peak for solar; per-slice for the others), not
                    # [T] as under the annual builder. Summing over m.S reconstructs
                    # the SAME annual total the [t]-only read gave before -- these
                    # are additive across slices by construction (the 6 slices
                    # partition the year's 8760 hours exactly), so the quantity's
                    # meaning is unchanged, only its representation.
                    solar_g = sum(float(pyo.value(m.solar_generation[t_idx, (s, p)])) for (s, p) in m.S)
                    gas_g = sum(float(pyo.value(m.gas_generation[t_idx, (s, p)])) for (s, p) in m.S)
                    stor_g = sum(float(pyo.value(m.storage_discharge[t_idx, (s, p)])) for (s, p) in m.S)
                    total_g = solar_g + gas_g + stor_g
                    if total_g < 1e-9:
                        return 0, 0, 0
                    return solar_g / total_g, gas_g / total_g, stor_g / total_g

                solar_share_2030, gas_share_2030, _ = gen_shares(5)
                solar_share_2035, gas_share_2035, _ = gen_shares(10)
                solar_share_2045, gas_share_2045, _ = gen_shares(len(years) - 1)

                # TASK 04, Section A: selfgen volume at the SAME year index
                # (10 = 2035) gen_shares()'s own tipping test reads, so the
                # two metrics are read at a common point. Direct per-slice
                # model summation, matching gen_shares()'s own style (not
                # diag, which is keyed by calendar year rather than index --
                # same quantity either way, since diag's own selfgen_twh_
                # by_year is itself built from m.selfgen the same way).
                selfgen_twh_2035 = sum(
                    float(pyo.value(m.selfgen[10, (s, p)])) for (s, p) in m.S
                )

                # System LCOE
                disc_gen = sum(
                    float(pyo.value(m.DF[t])) * (
                        sum(float(pyo.value(m.solar_generation[t, (s, p)])) for (s, p) in m.S)
                        + sum(float(pyo.value(m.gas_generation[t, (s, p)])) for (s, p) in m.S)
                        + sum(float(pyo.value(m.storage_discharge[t, (s, p)])) for (s, p) in m.S)
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
                    # TASK 04, Section B.
                    "cum_emissions_grid":    cum_emissions_grid,
                    "cum_emissions_selfgen": cum_emissions_selfgen,
                    "cum_unserved":     cum_unserved,
                    # TASK 04, Section A.
                    "cum_selfgen_twh":  cum_selfgen_twh,
                    "selfgen_twh_2035": selfgen_twh_2035,
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

                # TASK 04, Section A: selfgen_by_year.csv. "arm" is added
                # beyond the task's literal (carbon_price, year) key --
                # 19_ sweeps 3 financing arms per carbon price, and without
                # it, rows from different arms at the same carbon_price
                # would collide/overwrite rather than compose into a valid
                # long-format table. demand_twh read from the builder's own
                # diagnostics series (diag["demand_twh_by_year"]), not
                # recomputed from a growth rate, per this task's own
                # instruction and TASK 03's precedent.
                for y in sorted(diag["demand_twh_by_year"].keys()):
                    year_rows.append({
                        "arm":           arm["label"],
                        "carbon_price":  cp,
                        "year":          y,
                        "selfgen_twh":   diag["selfgen_twh_by_year"][y],
                        "unserved_twh":  diag["unserved_twh_by_year"][y],
                        "demand_twh":    diag["demand_twh_by_year"][y],
                    })

                print(f"    ${cp:>3}/tCO2 -> cost=${total_cost/1e9:.1f}B  "
                      f"emit={cum_emissions/1e6:.1f}Mt  "
                      f"solar2035={solar_share_2035:.0%}  "
                      f"gas2035={gas_share_2035:.0%}")

            except Exception as e:
                all_rows.append({
                    "arm": arm["label"], "carbon_price": cp,
                    "status": f"error: {e}",
                })
                print(f"    ${cp:>3}/tCO2 -> ERROR: {e}")

    # ── Save ──────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "carbon_price_raw.csv", index=False)

    # TASK 04, Section A: long-format per-year selfgen/unserved/demand.
    # Path note (docs/task_log_04_carbon_selfgen.md): the task text names
    # "results/carbon_sweep/selfgen_by_year.csv"; no such directory exists
    # anywhere in this repository (checked), and this script's own
    # RESULTS_DIR has always been results/carbon_price_sweep/ (every other
    # output above lives there). Read as shorthand for the existing
    # directory, not an instruction to fragment this runner's output
    # across two sibling directories with no stated reason -- saved
    # alongside carbon_price_raw.csv, not in a new results/carbon_sweep/.
    year_df = pd.DataFrame(year_rows)
    year_df.to_csv(RESULTS_DIR / "selfgen_by_year.csv", index=False)

    optimal = df[df["status"] == "optimal"].copy()

    # ── Identify tipping points ───────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CARBON PRICE TIPPING POINTS")
    print(f"{'='*70}")

    tipping_points = {}
    for arm_label in [a["label"] for a in FINANCING_ARMS]:
        sub = optimal[optimal["arm"] == arm_label].sort_values("carbon_price")
        if len(sub) == 0:
            continue

        # Tipping point: first carbon price where solar > 50% in 2035
        tipping = sub[sub["solar_share_2035"] > 0.50]
        if len(tipping):
            tp = tipping.iloc[0]["carbon_price"]
            print(f"  {arm_label:<25} Solar >50% in 2035 at ${tp:.0f}/tCO2")
            tipping_points[arm_label] = {"solar_50pct_2035_price_usd_per_tco2": float(tp)}
        else:
            print(f"  {arm_label:<25} Solar never reaches 50% in 2035 at tested prices")
            tipping_points[arm_label] = {"solar_50pct_2035_price_usd_per_tco2": None}

    # ── Self-generation displacement metric (TASK 04, Section C) ───────
    # docs/task_log_04_carbon_selfgen.md. Does NOT redefine MIX-3's own
    # tipping-point test above (unchanged) -- a SEPARATE, added observable:
    # the lowest SAMPLED carbon price at which cum_selfgen_twh has fallen
    # to 50% / 90% below its own carbon_price=0 value, per financing arm.
    GRID_RESOLUTION_NOTE = (
        "grid-resolution limited: the lowest SAMPLED price (of the 13 "
        "tested) meeting the threshold, not the infimum"
    )
    print(f"\n{'='*70}")
    print(f"  SELF-GENERATION DISPLACEMENT (grid-resolution limited)")
    print(f"{'='*70}")

    displacement = {}
    for arm_label in [a["label"] for a in FINANCING_ARMS]:
        sub = optimal[optimal["arm"] == arm_label].sort_values("carbon_price")
        if len(sub) == 0:
            displacement[arm_label] = None
            continue

        zero_rows = sub[sub["carbon_price"] == 0]
        if len(zero_rows) == 0:
            print(f"  {arm_label:<25} carbon_price=0 not optimal -- no baseline to compare against")
            displacement[arm_label] = {
                "note": "carbon_price=0 was not optimal for this arm; "
                        "no baseline selfgen volume to measure displacement from",
            }
            continue

        selfgen_0 = float(zero_rows.iloc[0]["cum_selfgen_twh"])

        def _find_threshold(fraction_remaining):
            # Lowest sampled price at which cum_selfgen_twh <= fraction_remaining * selfgen_0.
            hits = sub[sub["cum_selfgen_twh"] <= fraction_remaining * selfgen_0]
            hits = hits[hits["carbon_price"] > 0]
            if len(hits) == 0:
                return {
                    "price_usd_per_tco2": None,
                    "realised_fraction_of_baseline": None,
                    "reached_within_swept_range": False,
                }
            row = hits.sort_values("carbon_price").iloc[0]
            realised = (
                float(row["cum_selfgen_twh"]) / selfgen_0 if selfgen_0 > 1e-9 else None
            )
            return {
                "price_usd_per_tco2": float(row["carbon_price"]),
                "realised_fraction_of_baseline": realised,
                "reached_within_swept_range": True,
            }

        if selfgen_0 <= 1e-9:
            print(f"  {arm_label:<25} cum_selfgen_twh at $0/tCO2 is ~0 -- "
                  f"no baseline self-generation to displace")
            displacement[arm_label] = {
                "baseline_selfgen_twh": selfgen_0,
                "note": "baseline self-generation at carbon_price=0 is "
                        "approximately zero; a percentage displacement "
                        "from a near-zero baseline is not meaningful",
            }
            continue

        d50 = _find_threshold(0.50)
        d90 = _find_threshold(0.10)
        displacement[arm_label] = {
            "baseline_selfgen_twh": selfgen_0,
            "displaced_50pct": d50,
            "displaced_90pct": d90,
        }

        def _fmt(d):
            if d["reached_within_swept_range"]:
                return (f"${d['price_usd_per_tco2']:.0f}/tCO2 "
                        f"(realised fraction {d['realised_fraction_of_baseline']:.2f})")
            return "not reached within swept range ($0-150/tCO2)"

        print(f"  {arm_label:<25} baseline selfgen (CP=0): {selfgen_0:.2f} TWh")
        print(f"  {'':<25} 50% displaced: {_fmt(d50)}")
        print(f"  {'':<25} 90% displaced: {_fmt(d90)}")

    # ── Summary JSON (TASK 04: new -- none existed before this task) ───
    summary_path = RESULTS_DIR / "carbon_price_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(json_safe({
            "carbon_prices_usd_per_tco2": CARBON_PRICES,
            "financing_arms": [a["label"] for a in FINANCING_ARMS],
            "mix3_tipping_points": tipping_points,
            "selfgen_displacement": {
                "note": GRID_RESOLUTION_NOTE,
                "by_arm": displacement,
            },
        }), f, indent=2)
    print(f"\nSaved: {summary_path}")

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
