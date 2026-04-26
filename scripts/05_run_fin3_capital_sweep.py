"""
05_run_fin3_capital_sweep.py  —  FIN-3 Capital Budget Sweep
============================================================

RESEARCH QUESTION (FIN-3)
--------------------------
How does the public capital budget ceiling (B*) independently determine
the energy access outcome under NDC constraints — and at what budget level
does the public-capital bottleneck disappear without EaaS?

ROBUSTNESS GAP BEING CLOSED
-----------------------------
Previous runs used a single capital_case per NDC scenario:
    02_run_ndc_caps.py: moderate (unconditional) / expansion (conditional)

This fixed point cannot answer FIN-3. The question requires isolating the
capital budget as the sole varying dimension while holding all else constant:
    - NDC cap: ndc3_unconditional (the tighter, harder-to-achieve target)
    - financing_regime: "traditional" (NO EaaS — isolates capital effect)
    - gas_case: baseline (one variable at a time)
    - voll: voll_mid (canonical)
    - tariff_grid: not applicable (EaaS off)

By running across all five capital levels without EaaS, we can show:
    (a) At what B* does unserved energy drop to zero?
        → This is the budget level where the capital bottleneck disappears.
    (b) Does the carbon shadow price remain positive above that B*?
        → If yes, the NDC constraint (not capital) becomes the binding limit.
    (c) What is the shadow price of the budget constraint at each level?
        → Positive shadow = budget is binding; zero = budget is slack.

ANSWERING THE "WHAT IF YOU JUST SPENT MORE?" EXAMINER QUESTION
---------------------------------------------------------------
This experiment directly tests the claim:
    "Without EaaS, NDC compliance worsens energy access because the
     public capital ceiling cannot sustain the solar deployment needed
     to compensate for constrained gas."

If the claim is only true at moderate/tight budgets but breaks at
adequacy or unconstrained, then EaaS is a CONVENIENCE, not a NECESSITY.
If the claim holds even at unconstrained budget (i.e. even with unlimited
public capital, NDC compliance forces emissions reductions that crowd out
gas, worsening access), then EaaS is NECESSARY.
This is the distinction FIN-3 settles.
FIN-3 is explicitly scoped to NDC 3.0 because the capital frontier question is most stringent there;
the NDC 2.0 equivalent would show a less-binding constraint.

OUTPUTS
--------
  results/fin3/capital_sweep_results.csv   — one row per capital_case
  results/fin3/capital_sweep_summary.json  — bottleneck disappearance point

RUN ORDER
----------
  00_build_emissions_cap.py  (must exist)
  01_run_baseline.py         (must exist)
  05_run_fin3_capital_sweep.py  ← this script
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io import (load_econ, load_solar_capex_by_year)
from src.scenarios import load_scenario, capital_envelope_scenarios
from src.optimize_model import build_model, solve_model
from src.optimize_experiments import extract_planning_diagnostics

# ============================================================
# CONFIG
# ============================================================

RESULTS_DIR = ROOT / "results" / "fin3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAP_PATH = ROOT / "data" / "cost" / "processed" / "emissions_cap.csv"

CANONICAL_VOLL = "voll_mid"

# NDC scenario: unconditional is the harder test.
# The conditional cap is looser; if capital is the bottleneck it will
# manifest most clearly under the tighter unconditional cap.
NDC_CAP_SCENARIO = "ndc3_unconditional"

# Capital levels to sweep — all five from capital_envelope_scenarios()
# in order from most constrained to unconstrained.
CAPITAL_SWEEP_ORDER = ["tight", "moderate", "adequacy", "expansion", "unconstrained"]

# ── DisCo Collection Rate Sensitivity (institutional financing friction) ──────
# domestic_share: fraction of B* that is domestically financed
# (domestic fiscal + CBN intervention). Remainder = external DFI finance.
# Calibrated from Nigeria power sector financing mix: ~35% domestic, ~65% external.
# Source: World Bank Nigeria DPO P164307 (2022), AfDB PSRP appraisal (2021).
DOMESTIC_SHARE = 0.35

COLLECTION_RATE_SCENARIOS = {
    "no_friction":   1.00,   # planning ideal — original model assumption
    "reform_target": 0.60,   # World Bank target (Nigeria 2030 aspiration)
    "current":       0.35,   # NERC 3-year average 2021-2023 (Nigeria today)
}

# ============================================================
# HELPERS
# ============================================================

def load_annual_caps(scenario_name: str, years: list) -> list:
    if not CAP_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CAP_PATH}. Run 00_build_emissions_cap.py first."
        )
    df = pd.read_csv(CAP_PATH)
    df = df[df["scenario"] == scenario_name].copy()
    df = df[df["year"].isin([int(y) for y in years])].sort_values("year")
    caps = df["cap_tco2"].astype(float).tolist()
    if len(caps) != len(years):
        raise ValueError(
            f"Cap length {len(caps)} != model years {len(years)} "
            f"for scenario '{scenario_name}'."
        )
    return caps


# ============================================================
# MAIN
# ============================================================

def main():

    econ = load_econ(CANONICAL_VOLL)
    envelopes = capital_envelope_scenarios()

    print(f"\nFIN-3: Capital budget sweep × DisCo collection rate sensitivity")
    print(f"  NDC cap:           {NDC_CAP_SCENARIO}")
    print(f"  Financing regime:  traditional (no EaaS)")
    print(f"  VoLL:              {CANONICAL_VOLL}")
    print(f"  Capital levels:    {CAPITAL_SWEEP_ORDER}")
    print(f"  Collection rates:  {COLLECTION_RATE_SCENARIOS}")
    print(f"  Total solves:      {len(CAPITAL_SWEEP_ORDER) * len(COLLECTION_RATE_SCENARIOS)}")

    rows = []   # accumulates all rows across all collection rates

    for cr_label, cr_value in COLLECTION_RATE_SCENARIOS.items():

        print(f"\n{'='*65}")
        print(f"  Collection rate scenario: {cr_label} ({cr_value:.0%})")
        print(f"{'='*65}")

        for capital_case in CAPITAL_SWEEP_ORDER:

            budget = envelopes[capital_case]
            budget_label = f"{budget/1e9:.2f}B" if budget is not None else "unconstrained"
            print(f"\n  Running capital_case={capital_case} ({budget_label} USD NPV)")

            scenario = load_scenario(
                demand_level_case="served",
                demand_case="baseline",
                gas_deliverability_case="baseline",
                capital_case=capital_case,
                carbon_case="no_policy",   # no carbon price — cap only
                start_year=2025,
                end_year=2045,
            )

            # NO EaaS — this is the critical isolation condition for FIN-3.
            # financing_regime defaults to "traditional" in load_scenario().
            # Explicitly assert it here for clarity.
            assert scenario["financing_regime"] == "traditional", (
                "FIN-3 requires financing_regime='traditional'. "
                "Check load_scenario() defaults."
            )

            years = [int(y) for y in scenario["years"]]
            caps = load_annual_caps(NDC_CAP_SCENARIO, years)

            # ── Apply DisCo collection rate to public budget ───────────────────
            # Only the domestic portion of the budget is affected by collection
            # efficiency. External DFI finance (World Bank, AfDB) is independent
            # of DisCo revenue and is held at its nominal value.
            # domestic_budget = nominal_budget × DOMESTIC_SHARE × cr_value
            # external_budget = nominal_budget × (1 - DOMESTIC_SHARE)
            # effective_budget = domestic_budget + external_budget
            base_budget = scenario.get("public_solar_budget_npv")
            if base_budget is not None:
                domestic_budget  = base_budget * DOMESTIC_SHARE * cr_value
                external_budget  = base_budget * (1.0 - DOMESTIC_SHARE)
                effective_budget = domestic_budget + external_budget
                scenario["public_solar_budget_npv"] = effective_budget
            # If base_budget is None (unconstrained), no modification needed.

            # Load time-varying solar CAPEX from NREL ATB (solar_low scenario).
            # solar_low declines from $1,456k/MW (2025) to $603k/MW (2045).
            solar_capex_tv = load_solar_capex_by_year(
                scenario_name="solar_low",
                start_year=int(years[0]),
                end_year=int(years[-1]),
            )

            # Activate minimum build floor when time-varying CAPEX is in use.
            # This prevents, the optimizer from delaying all solar to the cheapest years
            # (2040-2045) creating unrealistic 2025-2030 supply gaps.
            scenario["solar_min_build_mw_per_year"] = 100.0
            scenario["disco_collection_rate"] = cr_value

            # ------------------------------------------
            # Build + Solve
            # ------------------------------------------
            m = build_model(
                scenario=scenario,
                econ=econ,
                emissions_cap_by_year=caps,
                solar_capex_by_year=solar_capex_tv,
            )

            status = solve_model(m)
            # -------------------------------------------
            # Collect results 
            # ------------------------------------------
            if not status["optimal"]:
                print(f"    INFEASIBLE at capital_case={capital_case}")
                rows.append({
                    "collection_rate_scenario": cr_label,
                    "collection_rate":          cr_value,
                    "capital_case":             capital_case,
                    "budget_usd":               budget,
                    "budget_b_usd":             budget / 1e9 if budget else None,
                    "budget_fraction_bstar":    budget / envelopes["adequacy"] if budget else None,
                    "status":                   "infeasible",
                })
                continue

            diag = extract_planning_diagnostics(m, scenario, econ)

            npv_cost       = float(pyo.value(m.system_cost_npv))
            cum_unserved   = sum(diag["unserved_twh_by_year"].values())
            cum_emissions  = float(pyo.value(m.emissions))
            horizon_rel    = diag["horizon_reliability"]

            solar_public_mw = sum(
                float(pyo.value(m.solar_public_add[t])) for t in range(len(years))
            )
            # EaaS is off — solar_eaas_add will be zero; confirm
            solar_eaas_mw = sum(
                float(pyo.value(m.solar_eaas_add[t])) for t in range(len(years))
            )

            # Carbon shadow: mean of binding-year values (proxy for constraint pressure)
            c_shadow_vals = [
                v for v in diag["carbon_shadow_price_usd_per_tco2_by_year"].values()
                if v is not None
            ]
            carbon_shadow_mean = float(np.mean(c_shadow_vals)) if c_shadow_vals else 0.0
            carbon_shadow_max  = float(max(c_shadow_vals))     if c_shadow_vals else 0.0
            carbon_binding_years = sum(1 for v in c_shadow_vals if v > 1e-6)

            # Budget shadow and utilisation from the new FIN-3 diagnostics
            budget_shadow       = diag.get("public_budget_shadow_usd_per_usd")
            budget_utilisation  = diag.get("public_budget_utilisation")
            realised_spend      = diag.get("public_budget_realised_spend_usd")

            row = {
                "capital_case":              capital_case,
                "budget_usd":                effective_budget if base_budget is not None else None,
                "budget_b_usd":              budget / 1e9 if budget else None,
                "budget_fraction_bstar":     budget / envelopes["adequacy"] if budget else None,
                "status":                    "optimal",
                # Collections
                "collection_rate_scenario":  cr_label,
                "collection_rate":           cr_value,
                # Energy access
                "cumulative_unserved_twh":   cum_unserved,
                "horizon_reliability":       horizon_rel,
                # Emissions
                "cumulative_emissions_tco2": cum_emissions,
                "carbon_shadow_mean_usd_per_tco2": carbon_shadow_mean,
                "carbon_shadow_max_usd_per_tco2":  carbon_shadow_max,
                "carbon_binding_years":      carbon_binding_years,
                # Cost
                "npv_total_cost_usd":        npv_cost,
                # Solar investment
                "solar_public_total_mw":     solar_public_mw,
                "solar_eaas_total_mw":       solar_eaas_mw,   # should be 0
                # Budget constraint diagnostics (FIN-3 core)
                "budget_shadow_usd_per_usd": budget_shadow,
                "budget_utilisation":        budget_utilisation,
                "budget_realised_spend_usd": realised_spend,
            }
            rows.append(row)

            print(
                f"    unserved={cum_unserved:.3f} TWh  "
                f"carbon_shadow_mean={carbon_shadow_mean:.0f} USD/tCO2  "
                f"budget_shadow={budget_shadow:.4f}" if budget_shadow is not None
                else f"    unserved={cum_unserved:.3f} TWh  "
                    f"carbon_shadow_mean={carbon_shadow_mean:.0f} USD/tCO2  "
                    f"budget_shadow=None (unconstrained)"
            )

    # ── Save CSV ───────────────────────────────────────────────────────────────
    df_out = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "capital_sweep_results.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ── Summary: identify bottleneck disappearance point ──────────────────────
    no_friction_rows = df_out[
        (df_out["status"] == "optimal") &
        (df_out["collection_rate_scenario"] == "no_friction")
    ].copy()

    # Bottleneck disappears when unserved drops to ~zero (< 0.01 TWh)
    adequate = no_friction_rows[no_friction_rows["cumulative_unserved_twh"] < 0.01]
    bottleneck_disappears_at = (
        adequate["capital_case"].iloc[0] if len(adequate) > 0 else "not_reached"
    )

    # NDC constraint becomes the binding limit when carbon shadow > 0
    # even after unserved drops to zero
    ndc_binding_after_capital = (
        adequate[adequate["carbon_shadow_mean_usd_per_tco2"] > 1.0]["capital_case"].tolist()
        if len(adequate) > 0 else []
    )

    summary = {
        "ndc_cap_scenario":           NDC_CAP_SCENARIO,
        "financing_regime":           "traditional (no EaaS)",
        "voll":                       CANONICAL_VOLL,
        "b_star_usd":                 envelopes["adequacy"],
        "bottleneck_disappears_at":   bottleneck_disappears_at,
        "ndc_binding_after_capital_removed": ndc_binding_after_capital,
        "interpretation": (
            "If bottleneck_disappears_at is 'not_reached', public capital alone "
            "cannot eliminate unserved energy under this NDC cap — EaaS is necessary, "
            "not merely convenient. "
            "If it IS reached, note carbon_shadow at that level: if positive, the NDC "
            "constraint (not capital) becomes the next binding limit."
        ),
        "capital_levels": [
            {
                "capital_case":          r["capital_case"],
                "budget_b_usd":          r.get("budget_b_usd"),
                "cumulative_unserved_twh": r.get("cumulative_unserved_twh"),
                "budget_shadow":         r.get("budget_shadow_usd_per_usd"),
                "carbon_shadow_mean":    r.get("carbon_shadow_mean_usd_per_tco2"),
                "status":                r["status"],
            }
            for r in rows
        ],
    }

    json_path = RESULTS_DIR / "capital_sweep_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")

    # ── Three-way comparison: one table per collection rate ────────────────────

    for cr_label, cr_value in COLLECTION_RATE_SCENARIOS.items():
        subset = df_out[df_out["collection_rate_scenario"] == cr_label].copy()
        optimal = subset[subset["status"] == "optimal"]

        # Bottleneck disappearance for this collection rate
        adequate = optimal[optimal["cumulative_unserved_twh"] < 0.01]
        bottleneck_at = (
            adequate["capital_case"].iloc[0] if len(adequate) > 0
            else "not_reached"
        )

        print(f"\n=== FIN-3: {cr_label.upper()} (collection rate = {cr_value:.0%}) ===")
        print(f"  Effective budget = {cr_value:.0%} domestic × {DOMESTIC_SHARE:.0%} share "
              f"+ {(1-DOMESTIC_SHARE):.0%} external DFI (fixed)")
        print()
        print(f"  {'Capital case':<14} {'Nominal B$':>11} {'Effective B$':>13} "
              f"{'Unserved TWh':>13} {'Bgt shadow':>11} {'C shadow':>10}")
        print("  " + "-" * 76)

        for _, r in subset.sort_values("capital_case",
                key=lambda s: s.map({c: i for i, c in enumerate(CAPITAL_SWEEP_ORDER)})
            ).iterrows():

            nom_b  = envelopes[r["capital_case"]]
            nom_str = f"{nom_b/1e9:.2f}" if nom_b else "∞"

            eff_b  = r.get("budget_usd")
            eff_str = f"{eff_b/1e9:.2f}" if eff_b else "∞"

            unser = (f"{r['cumulative_unserved_twh']:.3f}"
                     if r.get("cumulative_unserved_twh") is not None
                     else "infeasible")
            bshdw = (f"{r['budget_shadow_usd_per_usd']:.4f}"
                     if r.get("budget_shadow_usd_per_usd") is not None
                     else "—")
            cshdw = (f"{r['carbon_shadow_mean_usd_per_tco2']:.0f}"
                     if r.get("carbon_shadow_mean_usd_per_tco2") is not None
                     else "—")

            print(f"  {r['capital_case']:<14} {nom_str:>11} {eff_str:>13} "
                  f"{unser:>13} {bshdw:>11} {cshdw:>10}")

        print(f"\n  Bottleneck disappears at: {bottleneck_at}")
        if bottleneck_at == "not_reached":
            print("  → Public capital alone CANNOT eliminate unserved energy.")
            print("    EaaS is NECESSARY under this collection rate scenario.")
        else:
            print("  → Public capital sufficient at this collection rate.")
            print("    EaaS is CONVENIENT but not strictly necessary.")

    # ── Cross-scenario comparison: does bottleneck point shift? ───────────────
    print(f"\n=== COLLECTION RATE IMPACT ON EaaS NECESSITY ===")
    print(f"  {'Scenario':<16} {'Rate':>6}  {'Bottleneck disappears at'}")
    print("  " + "-" * 50)
    for cr_label, cr_value in COLLECTION_RATE_SCENARIOS.items():
        subset = df_out[df_out["collection_rate_scenario"] == cr_label]
        optimal = subset[subset["status"] == "optimal"]
        adequate = optimal[optimal["cumulative_unserved_twh"] < 0.01]
        b_at = adequate["capital_case"].iloc[0] if len(adequate) > 0 else "not_reached"
        print(f"  {cr_label:<16} {cr_value:>5.0%}  {b_at}")

    print(f"\n  Interpretation: if bottleneck_at shifts to a higher capital case")
    print(f"  as collection rate falls, DisCo revenue friction directly")
    print(f"  strengthens the EaaS necessity argument.")

if __name__ == "__main__":
    main()
