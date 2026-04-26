"""
00_build_emissions_cap.py  —  NDC 3.0 (proportional top-down) + NDC 2.0 (BAU-relative)
========================================================================================

TWO NDC FORMULATIONS
---------------------
This file generates annual power-sector emissions caps under two distinct
NDC apportionment methodologies:

    NDC 2.0 (BAU-relative):
        Cap is a percentage of the model's own baseline power-sector emissions.
        Unconditional: 20% below BAU 2030.  Conditional: 47% below BAU 2030.

    NDC 3.0 (proportional top-down):
        Cap is derived from NDC 3.0's energy-sector abatement target, apportioned
        to gas-fired power generation by its share of energy-sector emissions.

NDC 3.0 PROPORTIONAL METHODOLOGY
----------------------------------
NDC 3.0 (September 2025) defines economy-wide absolute reduction targets:
    - 168.2 MtCO2e by 2030  (29.3% of 2018 baseline of 573.5 MtCO2e)
    - 184.9 MtCO2e by 2035  (32.2% of 2018 baseline)
    Source: NDC 3.0, Section 3, p.14

The energy sector's share of economy-wide abatement (NDC 3.0, p.15):
    - 2030: 22.0%  →  37.0 MtCO2e energy-sector abatement
    - 2035: 26.3%  →  48.6 MtCO2e energy-sector abatement

Gas-fired power generation's share of energy-sector emissions:
    The model's baseline gas-to-power emissions (E_base) divided by total
    energy-sector emissions (~217 MtCO2e, ~38% of 573.5 MtCO2e economy-wide).
    E_base ≈ 7.364 MtCO2/yr → gas-power share ≈ 3.39%.

Gas-power abatement = gas_power_share × energy_sector_abatement
Power-sector cap    = E_base - gas_power_abatement.

This method is scope-appropriate: it asks gas-fired power to carry only its
proportional share of energy-sector abatement, not the share attributable to
hydro, DSM, or turbine efficiency (which are outside the model's scope).


UNCONDITIONAL vs CONDITIONAL (NDC 3.0, p.14)
---------------------------------------------
"Nigeria maintains its 20% unconditional objective and views the conditional
 reduction at 80% of the 29% and 32% reduction."

    Unconditional: 20% of 573.5 = 114.7 MtCO2e economy-wide reduction.
    Conditional:   Full 168.2 / 184.9 MtCO2e targets (requires int'l finance).

The conditional-to-unconditional ratio scales the gas-power abatement:
    Conditional multiplier 2030 = 168.2 / 114.7 ≈ 1.467
    Conditional multiplier 2035 = 184.9 / 114.7 ≈ 1.612

RESEARCH QUESTIONS ADDRESSED
    RQ-POL-1: What is the investment trajectory and cost differential between 
                unconditional and conditional NDC 3.0 targets in the power sector?"
    RQ-POL-2: What is the implied MAC of moving from unconditional to conditional
              NDC 3.0 target in the power sector — and how does EaaS change that?

"""
import json
import sys
import pandas as pd
from pathlib import Path

# ── Repo root ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# ── Emission factor (consistent with optimize_model.py) ───────────────────────
EF_TCO2_PER_TWH_TH = 181_100.0   # tCO2 per TWh_th

# ── Time horizon ──────────────────────────────────────────────────────────────
START_YEAR    = 2025
TARGET_YEAR   = 2030
TARGET_YEAR_2 = 2035
END_YEAR      = 2045

# ── NDC 3.0 economy-wide numbers (NDC 3.0 Section 3, pp.2, 14) ───────────────
NDC3_BASELINE_2018_MTCO2E          = 573.5
NDC3_ECONOMY_REDUCTION_2030_MTCO2E = 168.2   # 29.3% of baseline
NDC3_ECONOMY_REDUCTION_2035_MTCO2E = 184.9   # 32.2% of baseline

# ── Energy sector share of economy-wide abatement (NDC 3.0, p.15) ─────────────
ENERGY_SECTOR_SHARE_2030 = 0.220   # 22.0%
ENERGY_SECTOR_SHARE_2035 = 0.263   # 26.3%

# ── Energy sector total emissions 2018 (derived from NDC 3.0) ─────────────────
# NDC 3.0 reports energy sector as ~38% of economy-wide emissions.
# 0.38 × 573.5 = ~217.9 MtCO2e. We use 217.0 as a round conservative estimate.
ENERGY_SECTOR_TOTAL_EMISSIONS_2018_MTCO2E = 217.0

# ── Unconditional reduction (NDC 3.0, p.14) ───────────────────────────────────
UNCONDITIONAL_ABS_REDUCTION_MTCO2E = 0.20 * NDC3_BASELINE_2018_MTCO2E  # 114.7

# ── Conditional-to-unconditional scaling ──────────────────────────────────────
CONDITIONAL_MULTIPLIER_2030 = NDC3_ECONOMY_REDUCTION_2030_MTCO2E / UNCONDITIONAL_ABS_REDUCTION_MTCO2E
CONDITIONAL_MULTIPLIER_2035 = NDC3_ECONOMY_REDUCTION_2035_MTCO2E / UNCONDITIONAL_ABS_REDUCTION_MTCO2E

# ── File paths ────────────────────────────────────────────────────────────────
BASELINE_DIAG_PATH = ROOT / "results" / "baseline" / "diagnostics.json"
OUT_PATH           = ROOT / "data" / "cost" / "processed" / "emissions_cap.csv"


# ── Helper: linear interpolation ─────────────────────────────────────────────
def _lerp(y, y0, v0, y1, v1):
    if y1 == y0:
        return v1
    frac = (y - y0) / (y1 - y0)
    return (1.0 - frac) * v0 + frac * v1


def main():
    # ── 1. Load baseline diagnostics ──────────────────────────────────────────
    if not BASELINE_DIAG_PATH.exists():
        raise FileNotFoundError(
            f"Missing baseline diagnostics at {BASELINE_DIAG_PATH}. "
            f"Run scripts/01_run_baseline.py first."
        )

    with open(BASELINE_DIAG_PATH, "r") as f:
        diag = json.load(f)

    gas_to_power = diag["gas_to_power_twh_th_by_year"]

    def _get(year):
        key_str = str(year)
        if key_str in gas_to_power:
            return float(gas_to_power[key_str])
        if year in gas_to_power:
            return float(gas_to_power[year])
        raise KeyError(f"Year {year} not found in gas_to_power_twh_th_by_year")

    # ── 2. Modelled power-sector baseline emissions (tCO2/year) ───────────────
    Ebase_2025 = _get(START_YEAR)    * EF_TCO2_PER_TWH_TH
    Ebase_2030 = _get(TARGET_YEAR)   * EF_TCO2_PER_TWH_TH
    Ebase_2035 = _get(TARGET_YEAR_2) * EF_TCO2_PER_TWH_TH

    E_BASE_MTCO2 = Ebase_2025 / 1e6   # annual baseline in MtCO2

    # ── 3. Gas-power share of energy-sector emissions ─────────────────────────
    GAS_POWER_SHARE = E_BASE_MTCO2 / ENERGY_SECTOR_TOTAL_EMISSIONS_2018_MTCO2E

    # ── 4. NDC 3.0 energy-sector abatement targets ───────────────────────────
    energy_abatement_2030 = ENERGY_SECTOR_SHARE_2030 * NDC3_ECONOMY_REDUCTION_2030_MTCO2E
    energy_abatement_2035 = ENERGY_SECTOR_SHARE_2035 * NDC3_ECONOMY_REDUCTION_2035_MTCO2E

    # ── 5. Gas-power abatement (unconditional) ────────────────────────────────
    gas_abatement_uncond_2030 = GAS_POWER_SHARE * energy_abatement_2030
    gas_abatement_uncond_2035 = GAS_POWER_SHARE * energy_abatement_2035

    # ── 6. Gas-power abatement (conditional — scaled by ambition ratio) ───────
    gas_abatement_cond_2030 = gas_abatement_uncond_2030 * CONDITIONAL_MULTIPLIER_2030
    gas_abatement_cond_2035 = gas_abatement_uncond_2035 * CONDITIONAL_MULTIPLIER_2035

    # ── 7. Power-sector caps ──────────────────────────────────────────────────
    ndc3_uncond_cap_2030 = max(E_BASE_MTCO2 - gas_abatement_uncond_2030, 0.0) * 1e6
    ndc3_uncond_cap_2035 = max(E_BASE_MTCO2 - gas_abatement_uncond_2035, 0.0) * 1e6
    ndc3_cond_cap_2030   = max(E_BASE_MTCO2 - gas_abatement_cond_2030, 0.0) * 1e6
    ndc3_cond_cap_2035   = max(E_BASE_MTCO2 - gas_abatement_cond_2035, 0.0) * 1e6

    # ── 8. Build scenario definitions ─────────────────────────────────────────
    SCENARIOS = {
        # ── NDC 3.0 (proportional top-down) ────────────────────────────────
        "ndc3_unconditional": {
            "type":        "ndc3_proportional",
            "cap_2030":    ndc3_uncond_cap_2030,
            "cap_2035":    ndc3_uncond_cap_2035,
            "description": (
                f"NDC 3.0 proportional: gas-power share ({GAS_POWER_SHARE*100:.2f}%) "
                f"of energy-sector abatement (self-funded)"
            ),
        },
        "ndc3_conditional": {
            "type":        "ndc3_proportional",
            "cap_2030":    ndc3_cond_cap_2030,
            "cap_2035":    ndc3_cond_cap_2035,
            "description": (
                f"NDC 3.0 proportional: gas-power share ({GAS_POWER_SHARE*100:.2f}%) "
                f"of energy-sector abatement (conditional, int'l finance)"
            ),
        },
        # ── NDC 2.0 (BAU-relative) ─────────────────────────────────────────
        "ndc2_unconditional": {
            "type":          "bau_relative",
            "bau_mult_2030": 0.80,
            "description":   "NDC 2.0 unconditional: -20% vs projected 2030 power-sector BAU",
        },
        "ndc2_conditional": {
            "type":          "bau_relative",
            "bau_mult_2030": 0.53,
            "description":   "NDC 2.0 conditional: -47% vs projected 2030 power-sector BAU",
        },
    }

    # ── 9. Build annual cap trajectories ──────────────────────────────────────
    rows = []

    for scen_name, scen_cfg in SCENARIOS.items():
        scen_type = scen_cfg["type"]

        # ── NDC 2.0: BAU-relative ─────────────────────────────────────────
        if scen_type == "bau_relative":
            bau_mult = scen_cfg["bau_mult_2030"]
            cap_2030 = bau_mult * Ebase_2030

            for y in range(START_YEAR, END_YEAR + 1):
                if y <= START_YEAR:
                    cap_y = Ebase_2025
                elif y <= TARGET_YEAR:
                    cap_y = _lerp(y, START_YEAR, Ebase_2025, TARGET_YEAR, cap_2030)
                else:
                    cap_y = cap_2030

                rows.append({
                    "year":     y,
                    "scenario": scen_name,
                    "cap_tco2": max(cap_y, 0.0),
                })
            continue

        # ── NDC 3.0: proportional top-down ────────────────────────────────
        if scen_type == "ndc3_proportional":
            cap_2030 = scen_cfg["cap_2030"]
            cap_2035 = scen_cfg["cap_2035"]

            for y in range(START_YEAR, END_YEAR + 1):
                if y <= START_YEAR:
                    cap_y = Ebase_2025   # unconstrained in year 0
                elif y <= TARGET_YEAR:
                    cap_y = _lerp(y, START_YEAR, Ebase_2025, TARGET_YEAR, cap_2030)
                elif y <= TARGET_YEAR_2:
                    cap_y = _lerp(y, TARGET_YEAR, cap_2030, TARGET_YEAR_2, cap_2035)
                else:
                    cap_y = cap_2035     # held flat after 2035

                rows.append({
                    "year":     y,
                    "scenario": scen_name,
                    "cap_tco2": max(cap_y, 0.0),
                })
            continue

    # ── 10. Write output ──────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)

    # ── 11. Console summary ───────────────────────────────────────────────────
    print("=" * 70)
    print("  Emissions cap written — NDC 3.0 (proportional) + NDC 2.0 (BAU-relative)")
    print("=" * 70)

    print(f"\nNDC 3.0 economy-wide 2018 baseline:    {NDC3_BASELINE_2018_MTCO2E:.1f} MtCO2e")
    print(f"Economy-wide reduction target 2030:     {NDC3_ECONOMY_REDUCTION_2030_MTCO2E:.1f} MtCO2e "
          f"({NDC3_ECONOMY_REDUCTION_2030_MTCO2E/NDC3_BASELINE_2018_MTCO2E*100:.1f}%)")
    print(f"Economy-wide reduction target 2035:     {NDC3_ECONOMY_REDUCTION_2035_MTCO2E:.1f} MtCO2e "
          f"({NDC3_ECONOMY_REDUCTION_2035_MTCO2E/NDC3_BASELINE_2018_MTCO2E*100:.1f}%)")

    print(f"\nEnergy sector abatement share (NDC 3.0 p.15):")
    print(f"  2030: {ENERGY_SECTOR_SHARE_2030*100:.1f}% → {energy_abatement_2030:.1f} MtCO2e")
    print(f"  2035: {ENERGY_SECTOR_SHARE_2035*100:.1f}% → {energy_abatement_2035:.1f} MtCO2e")

    print(f"\nGas-power share of energy-sector emissions:")
    print(f"  E_base = {E_BASE_MTCO2:.3f} MtCO2/yr")
    print(f"  E_energy_sector = {ENERGY_SECTOR_TOTAL_EMISSIONS_2018_MTCO2E:.0f} MtCO2e")
    print(f"  Gas-power share = {GAS_POWER_SHARE*100:.2f}%")

    print(f"\nGas-power abatement targets:")
    print(f"  Unconditional 2030: {gas_abatement_uncond_2030:.3f} MtCO2e "
          f"→ cap = {ndc3_uncond_cap_2030/1e6:.3f} MtCO2/yr")
    print(f"  Unconditional 2035: {gas_abatement_uncond_2035:.3f} MtCO2e "
          f"→ cap = {ndc3_uncond_cap_2035/1e6:.3f} MtCO2/yr")
    print(f"  Conditional 2030:   {gas_abatement_cond_2030:.3f} MtCO2e "
          f"→ cap = {ndc3_cond_cap_2030/1e6:.3f} MtCO2/yr")
    print(f"  Conditional 2035:   {gas_abatement_cond_2035:.3f} MtCO2e "
          f"→ cap = {ndc3_cond_cap_2035/1e6:.3f} MtCO2/yr")

    print(f"\nConditional multipliers (ambition scaling):")
    print(f"  2030: {CONDITIONAL_MULTIPLIER_2030:.3f}×  "
          f"(168.2 / 114.7)")
    print(f"  2035: {CONDITIONAL_MULTIPLIER_2035:.3f}×  "
          f"(184.9 / 114.7)")

    print(f"\nModelled power-sector baseline (gas-fired generation):")
    print(f"  Ebase 2025: {Ebase_2025/1e6:.3f} MtCO2/yr")
    print(f"  Ebase 2030: {Ebase_2030/1e6:.3f} MtCO2/yr")
    print(f"  Ebase 2035: {Ebase_2035/1e6:.3f} MtCO2/yr")

    print()
    for scen_name, scen_cfg in SCENARIOS.items():
        subset = df[df["scenario"] == scen_name]
        cap_2030 = subset.loc[subset["year"] == TARGET_YEAR,   "cap_tco2"].values[0]
        cap_2035 = subset.loc[subset["year"] == TARGET_YEAR_2, "cap_tco2"].values[0]
        abate_2030 = Ebase_2030 - cap_2030
        abate_2035 = Ebase_2035 - cap_2035
        print(f"  Scenario: {scen_name}")
        print(f"    {scen_cfg['description']}")
        print(f"    Power cap 2030: {cap_2030/1e6:.3f} MtCO2/yr  "
              f"(abatement: {abate_2030/1e6:.3f} MtCO2/yr)")
        print(f"    Power cap 2035: {cap_2035/1e6:.3f} MtCO2/yr  "
              f"(abatement: {abate_2035/1e6:.3f} MtCO2/yr)")
        print()

    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
