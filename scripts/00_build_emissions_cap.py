"""
00_build_emissions_cap.py  —  NDC 3.0 edition 
==========================================================

WHAT CHANGED FROM NDC 2.0 VERSION
-----------------------------------
NDC 2.0 used BAU-relative percentage targets:
    - Unconditional: -20% vs projected 2030 BAU emissions
    - Conditional:   -47% vs projected 2030 BAU emissions

NDC 3.0 (September 2025) shifts to economy-wide ABSOLUTE emissions reduction
targets relative to the 2018 base year:
    - 168.2 MtCO2e absolute reduction by 2030  (29.3% of 2018 baseline)
    - 184.9 MtCO2e absolute reduction by 2035  (32.2% of 2018 baseline)

Source: Nigeria NDC 3.0, Section 3 (Contribution to the Paris Agreement), p.14
    "Intermediate targets are absolute emissions reductions of 168.2 Mt CO2e in
     2030 and 184.9 Mt CO2e in 2035 from base year 2018 towards net zero by 2060."

2018 NATIONAL BASELINE
    Total economy-wide net emissions 2018 = 573.5 MtCO2e (NDC 3.0, p.2 / Table 1)

SCOPE OF THE MODEL vs NDC 3.0 SCOPE
-------------------------------------
IMPORTANT: The optimization model (optimize_model.py) imposes emissions_cap_by_year
as an ANNUAL constraint on gas-fired electricity generation ONLY:

    emissions_by_year[t] = gas_generation_TWhe * 1e6 * CARBON_EMISSION_FACTOR

This covers gas-to-power emissions exclusively. It does NOT cover transport,
buildings, manufacturing, agriculture, LULUCF, or waste — which together
constitute the bulk of Nigeria's NDC 3.0 economy-wide commitment.

NDC 3.0 does NOT define a standalone power-sector cap. To derive a power-sector
cap consistent with NDC 3.0 we apportion the economy-wide target using the
electricity-generation-specific mitigation potential stated in the NDC 3.0.

ELECTRICITY GENERATION SHARE (NDC 3.0, Table 2, p.16)
-------------------------------------------------------
NDC 3.0 Table 2 lists mitigation potential for fuel combustion measures:
    Measure 1 (renewables share):         12.220 MtCO2e by 2035
    Measure 2 (energy efficiency):
        Action 1 (turbine upgrades):       5.018 MtCO2e
        Action 2 (appliance standards):    6.229 MtCO2e
    Subtotal electricity generation:      ~23.467 MtCO2e by 2035

Fraction of economy-wide reduction attributable to electricity generation:
    2030: ~23.467 / 168.2 = ~13.95%  (applying same measures with partial deployment)
    2035: ~23.467 / 184.9 = ~12.69%

NOTE: NDC 3.0 p.15 cites "Energy sector (including buildings and transport) = 22.0%
in 2030 and 26.3% in 2035" — this is the FULL ENERGY SECTOR share, NOT just
electricity generation. Using that broader share would impose a cap approximately
2x tighter than the NDC 3.0 actually requires for gas-fired power. This file
uses the electricity-generation-specific share derived from Table 2.

UNCONDITIONAL vs CONDITIONAL (NDC 3.0, p.14)
---------------------------------------------
"Nigeria maintains its 20% unconditional objective and views the conditional
 reduction at 80% of the 29% and 32% reduction."

Interpretation:
    - "20% unconditional objective" = the NDC 2.0 commitment carried forward.
      In NDC 2.0 terms: 20% reduction from projected 2030 BAU.
      In NDC 3.0 absolute terms: 20% of the 573.5 MtCO2e 2018 baseline
      = 114.7 MtCO2e unconditional absolute reduction.

    - Conditional = the full 168.2 MtCO2e (2030) / 184.9 MtCO2e (2035) target.
      Conditional finance required = US$270bn of the total US$337bn (NDC 3.0, p.21).

    - The "80% of the 29% and 32%" language means ~80% of the ADDITIONAL ambition
      beyond the NDC 2.0 floor requires international finance. It does NOT mean
      the unconditional target is 20% × 168.2 = 33.6 MtCO2e (which would be
      only ~5.9% of the baseline — far below the stated 20% objective).

SCENARIO DEFINITIONS
---------------------
    ndc3_unconditional:  Domestically self-funded floor (20% of 573.5 MtCO2e baseline)
    ndc3_conditional:    Full NDC 3.0 target (requires international finance)

NOTE ON SCENARIO NAME COMPATIBILITY
-------------------------------------
Downstream scripts (02_run_ndc_caps.py, 03_run_ndc_eaas.py, 08_phase8_analysis.py)
currently reference "ndc_unconditional_20" and "ndc_conditional_47" (NDC 2.0 names).
This file writes ndc3_unconditional and ndc3_conditional (NDC 3.0 names).
You must update case name references in those downstream scripts before running.
Alternatively, add NDC 3.0 rows to the SAME CSV alongside NDC 2.0 rows
(which this file does NOT do — it overwrites the CSV).

RESEARCH QUESTIONS ADDRESSED
    RQ-POL-1: Does NDC 3.0's absolute-reduction target require a materially
              different investment trajectory than NDC 2.0's BAU-relative targets?
    RQ-POL-2: What is the implied MAC of moving from unconditional to conditional
              NDC 3.0 target in the power sector — and how does EaaS change that?
"""

import json
import os
import pandas as pd
import sys
from pathlib import Path

# ── Repo root ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# ── Emission factor (consistent with optimize_model.py and original 00_build) ──
# Derived from: eta=0.43 * 1e6 * CARBON_EMISSION_FACTOR=0.421 = 181,030 tCO2/TWh_th
# Original file uses 181,100 — retained for consistency across all scripts.
EF_TCO2_PER_TWH_TH = 181_100.0   # tCO2 per TWh_th  (0.04% difference to model, immaterial)

# ── Time horizon ───────────────────────────────────────────────────────────────
START_YEAR    = 2025   # first year caps are applied in the model
TARGET_YEAR   = 2030   # first NDC 3.0 milestone
TARGET_YEAR_2 = 2035   # second NDC 3.0 milestone
END_YEAR      = 2045   # model horizon (caps held flat after 2035)

# ── NDC 3.0 economy-wide numbers ───────────────────────────────────────────────
# Source: NDC 3.0 Section 3, pp.2, 14
NDC3_BASELINE_2018_MTCO2E         = 573.5   # economy-wide net emissions, 2018 (MtCO2e)
NDC3_ECONOMY_REDUCTION_2030_MTCO2E = 168.2  # 29.3% of baseline
NDC3_ECONOMY_REDUCTION_2035_MTCO2E = 184.9  # 32.2% of baseline

# ── Electricity-generation-specific share of economy-wide abatement ────────────
# Derived from NDC 3.0 Table 2 (p.16): Measures 1 and 2 for electricity generation
#   Measure 1 (renewables):  12.220 MtCO2e
#   Measure 2 (efficiency):   5.018 + 6.229 = 11.247 MtCO2e
#   Total:                   23.467 MtCO2e by 2035
#
# Shares derived:
#   ELEC_SHARE_2030: applying same measures at 2030 interim deployment level.
#   NDC 3.0 does not separate electricity and non-electricity energy targets by
#   year, so we apportion proportionally using the same 23.467 MtCO2e numerator
#   against the year-specific economy-wide denominator.
#
#   2030: 23.467 / 168.2 = 0.1395  (13.95%)
#   2035: 23.467 / 184.9 = 0.1269  (12.69%)
#
# These are the fractions of the economy-wide NDC 3.0 absolute reduction that
# the MODEL'S scope (gas-fired electricity) is responsible for.
ELEC_SHARE_2030 = 23.467 / NDC3_ECONOMY_REDUCTION_2030_MTCO2E   # ~13.95%
ELEC_SHARE_2035 = 23.467 / NDC3_ECONOMY_REDUCTION_2035_MTCO2E   # ~12.69%

# ── Unconditional reduction: 20% of 2018 baseline (NDC 3.0 p.14) ──────────────
# "Nigeria maintains its 20% unconditional objective" = NDC 2.0 commitment.
# In absolute terms: 20% × 573.5 MtCO2e = 114.7 MtCO2e.
# This is the domestically self-funded floor. The remaining ambition is
# conditional on international finance.
UNCONDITIONAL_ABS_REDUCTION_MTCO2E = 0.20 * NDC3_BASELINE_2018_MTCO2E  # 114.7 MtCO2e

# ── Scenario definitions ───────────────────────────────────────────────────────
# NDC 3.0 scenarios: absolute-reduction basis (this file's primary purpose)
# NDC 2.0 scenarios: BAU-relative basis (added for POL-1 comparison)
#
# POL-1 requires running the model under BOTH NDC versions with identical
# model parameters. The NDC 2.0 scenarios use a different methodology:
# they define a cap as a MULTIPLIER on the model's own baseline gas emissions
# at the target year (BAU-relative), not as an apportioned absolute reduction.
#
# NDC 2.0 unconditional: cap_2030 = 0.80 × Ebase_2030  (20% below BAU)
# NDC 2.0 conditional:   cap_2030 = 0.53 × Ebase_2030  (47% below BAU)
# These are linearly interpolated from Ebase_2025 to cap_2030, then held flat.
#
# NDC 2.0 entries use a "type": "bau_relative" key to signal the different
# computation path inside main().

SCENARIOS = {
    # ── NDC 3.0 (absolute-reduction basis) ────────────────────────────────
    "ndc3_unconditional": {
        "type":          "absolute",
        "eco_red_2030":  UNCONDITIONAL_ABS_REDUCTION_MTCO2E,   # 114.7 MtCO2e
        "eco_red_2035":  UNCONDITIONAL_ABS_REDUCTION_MTCO2E,   # held flat
        "description":   "NDC 3.0 unconditional: 20% of 573.5 MtCO2e 2018 baseline (self-funded)",
    },
    "ndc3_conditional": {
        "type":          "absolute",
        "eco_red_2030":  NDC3_ECONOMY_REDUCTION_2030_MTCO2E,   # 168.2 MtCO2e
        "eco_red_2035":  NDC3_ECONOMY_REDUCTION_2035_MTCO2E,   # 184.9 MtCO2e
        "description":   "NDC 3.0 conditional: full 29%/32% NDC 3.0 target (requires int'l finance)",
    },
    # ── NDC 2.0 (BAU-relative basis — added for POL-1 comparison) ──────────
    # These replicate the original 00_build_emissions_cap.py methodology
    # (multiplier on the model's modelled 2030 baseline power-sector emissions)
    # so that both NDC versions are computed from the same baseline run.
    # Scenario names use ndc2_ prefix to distinguish from NDC 3.0 scenarios.
    "ndc2_unconditional": {
        "type":          "bau_relative",
        "bau_mult_2030": 0.80,   # 20% reduction below BAU 2030 power emissions
        "description":   "NDC 2.0 unconditional: -20% vs projected 2030 power-sector BAU",
    },
    "ndc2_conditional": {
        "type":          "bau_relative",
        "bau_mult_2030": 0.53,   # 47% reduction below BAU 2030 power emissions
        "description":   "NDC 2.0 conditional: -47% vs projected 2030 power-sector BAU",
    },
}

# ── File paths (same structure as NDC 2.0 version) ────────────────────────────
BASELINE_DIAG_PATH = ROOT / "results" / "baseline" / "diagnostics.json"
OUT_PATH           = ROOT / "data" / "cost" / "processed" / "emissions_cap.csv"


# ── Helper: linear interpolation ──────────────────────────────────────────────
def _lerp(y, y0, v0, y1, v1):
    """Linear interpolation between (y0, v0) and (y1, v1) at year y."""
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

    gas_to_power = diag["gas_to_power_twh_th_by_year"]  # dict year -> TWh_th

    def _get(year):
        """Look up a year in gas_to_power (keys may be int or str)."""
        key_str = str(year)
        if key_str in gas_to_power:
            return float(gas_to_power[key_str])
        if year in gas_to_power:
            return float(gas_to_power[year])
        raise KeyError(f"Year {year} not found in gas_to_power_twh_th_by_year")

    # ── 2. Modelled power-sector baseline emissions (tCO2/year) ───────────────
    # These are the annual gas-fired electricity emissions in the unconstrained
    # baseline run. They define the reference against which caps are applied.
    Ebase_2025 = _get(START_YEAR)    * EF_TCO2_PER_TWH_TH
    Ebase_2030 = _get(TARGET_YEAR)   * EF_TCO2_PER_TWH_TH
    Ebase_2035 = _get(TARGET_YEAR_2) * EF_TCO2_PER_TWH_TH

    # ── 3. Build annual cap trajectories ──────────────────────────────────────
    #
    # For each year y, the power-sector annual cap is:
    #
    #   cap(y) = baseline(y) - power_abatement(y)
    #
    # where:
    #   power_abatement(y) = economy_reduction(y) [tCO2]
    #                        × electricity_share(y)
    #
    # economy_reduction is linearly interpolated from:
    #   0 tCO2  in 2025  (model starts with unconstrained baseline)
    #   eco_red_2030 in 2030
    #   eco_red_2035 in 2035
    #   eco_red_2035 held flat from 2035 to END_YEAR
    #
    # electricity_share is linearly interpolated between ELEC_SHARE_2030
    # and ELEC_SHARE_2035 (held constant outside that range).
    #
    # baseline(y) is linearly interpolated between anchor years using the
    # modelled gas-to-power trajectory from the baseline diagnostics.

    rows = []

    for scen_name, scen_cfg in SCENARIOS.items():

        scen_type = scen_cfg.get("type", "absolute")

        # ── NDC 2.0 BAU-relative scenarios ────────────────────────────────────
        if scen_type == "bau_relative":
            # cap_2030 = multiplier × Ebase_2030 (model's own baseline at 2030)
            # Linearly interpolated from Ebase_2025 (unconstrained) to cap_2030,
            # then held flat from 2030 to END_YEAR.
            bau_mult = scen_cfg["bau_mult_2030"]
            cap_2030 = bau_mult * Ebase_2030

            for y in range(START_YEAR, END_YEAR + 1):
                if y <= START_YEAR:
                    cap_y = Ebase_2025
                elif y <= TARGET_YEAR:
                    cap_y = _lerp(y, START_YEAR, Ebase_2025, TARGET_YEAR, cap_2030)
                else:
                    cap_y = cap_2030   # held flat after 2030

                rows.append({
                    "year":     y,
                    "scenario": scen_name,
                    "cap_tco2": max(cap_y, 0.0),
                })
            continue   # skip the absolute-scenario logic below

        # ── NDC 3.0 absolute-reduction scenarios ──────────────────────────────
        eco_red_2030 = scen_cfg["eco_red_2030"] * 1e6   # MtCO2e → tCO2
        eco_red_2035 = scen_cfg["eco_red_2035"] * 1e6

        for y in range(START_YEAR, END_YEAR + 1):

            # --- economy-wide reduction required in year y (tCO2) ---
            if y <= START_YEAR:
                eco_red_y = 0.0
            elif y <= TARGET_YEAR:
                eco_red_y = _lerp(y, START_YEAR, 0.0, TARGET_YEAR, eco_red_2030)
            elif y <= TARGET_YEAR_2:
                eco_red_y = _lerp(y, TARGET_YEAR, eco_red_2030, TARGET_YEAR_2, eco_red_2035)
            else:
                eco_red_y = eco_red_2035   # held flat after 2035

            # --- electricity-generation share interpolated between 2030 and 2035 values ---
            if y <= TARGET_YEAR:
                elec_share_y = ELEC_SHARE_2030
            elif y <= TARGET_YEAR_2:
                elec_share_y = _lerp(y, TARGET_YEAR, ELEC_SHARE_2030,
                                        TARGET_YEAR_2, ELEC_SHARE_2035)
            else:
                elec_share_y = ELEC_SHARE_2035

            # --- modelled baseline power emissions for year y (tCO2) ---
            if y <= TARGET_YEAR:
                baseline_y = _lerp(y, START_YEAR, Ebase_2025, TARGET_YEAR, Ebase_2030)
            elif y <= TARGET_YEAR_2:
                baseline_y = _lerp(y, TARGET_YEAR, Ebase_2030, TARGET_YEAR_2, Ebase_2035)
            else:
                baseline_y = Ebase_2035

            # --- power-sector annual cap (tCO2/year) ---
            power_abatement_y = eco_red_y * elec_share_y
            cap_y = baseline_y - power_abatement_y

            # Floor at zero — cap cannot require negative emissions from gas plant
            cap_y = max(cap_y, 0.0)

            rows.append({
                "year":    y,
                "scenario": scen_name,
                "cap_tco2": cap_y,
            })
            # Note: diagnostic columns removed to maintain compatibility with
            # downstream load_annual_caps() which reads only year/scenario/cap_tco2.

    # ── 4. Write output ────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)

    # ── 5. Console summary ────────────────────────────────────────────────────
    print("=" * 65)
    print("  Emissions cap written — NDC 3.0 (absolute-reduction basis)")
    print("=" * 65)
    print(f"\nNDC 3.0 economy-wide 2018 baseline:  {NDC3_BASELINE_2018_MTCO2E:.1f} MtCO2e")
    print(f"Economy-wide reduction target 2030:   {NDC3_ECONOMY_REDUCTION_2030_MTCO2E:.1f} MtCO2e "
          f"({NDC3_ECONOMY_REDUCTION_2030_MTCO2E/NDC3_BASELINE_2018_MTCO2E*100:.1f}%)")
    print(f"Economy-wide reduction target 2035:   {NDC3_ECONOMY_REDUCTION_2035_MTCO2E:.1f} MtCO2e "
          f"({NDC3_ECONOMY_REDUCTION_2035_MTCO2E/NDC3_BASELINE_2018_MTCO2E*100:.1f}%)")
    print(f"\nElectricity-generation share (NDC 3.0 Table 2):")
    print(f"  2030: {ELEC_SHARE_2030*100:.2f}%   (23.467 / {NDC3_ECONOMY_REDUCTION_2030_MTCO2E})")
    print(f"  2035: {ELEC_SHARE_2035*100:.2f}%   (23.467 / {NDC3_ECONOMY_REDUCTION_2035_MTCO2E})")
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
    print()
    print("DOWNSTREAM SCRIPT UPDATE REQUIRED:")
    print("  02_run_ndc_caps.py   — update cases list to:")
    print("    [\"ndc3_unconditional\", \"ndc3_conditional\"]")
    print("  03_run_ndc_eaas.py   — same update")
    print("  08_phase8_analysis.py — update CASES paths to ndc3_ result directories")
    print()
    print("FOR RQ-POL-2 (Marginal Abatement Cost):")
    print("  MAC = (cost_ndc3_conditional - cost_ndc3_unconditional) /")
    print("        (cumulative_abatement_conditional - cumulative_abatement_unconditional)")
    print("  NDC 3.0 p.21: US$337bn total; US$270bn (80%) from int'l finance.")


if __name__ == "__main__":
    main()
