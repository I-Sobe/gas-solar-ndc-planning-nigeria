"""
99c_run_voll_sensitivity.py  —  Value of Lost Load Sensitivity
================================================================

Tests the robustness of headline findings to VoLL specification.

The canonical VoLL ($10,000/MWh) drives the optimizer to minimise
blackouts so aggressively that the explicit reliability constraint
(eps) never binds. This sensitivity sweeps VoLL from $1,000 to
$15,000/MWh to identify:

    1. The VoLL threshold at which the reliability constraint
       transitions from non-binding to binding.
    2. Whether the EaaS cost-saving finding holds across the range.
    3. How the cost decomposition (real expenditure vs VoLL penalty)
       shifts with VoLL specification.

PRE-REQUISITES
--------------
    01_run_baseline.py
    00_build_emissions_cap.py

RUN SEQUENCE
------------
    python scripts/01_run_baseline.py
    python scripts/00_build_emissions_cap.py
    python scripts/99c_run_voll_sensitivity.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io import load_econ, load_solar_capex_by_year
from src.scenarios import load_scenario
from src.optimize_model import build_model, solve_model
from src.optimize_experiments import extract_planning_diagnostics
from src.utils import json_safe

RESULTS_DIR = ROOT / "results" / "voll_sensitivity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CAP_PATH = ROOT / "data" / "cost" / "processed" / "emissions_cap.csv"

# ============================================================
# HELPERS
# ============================================================

def load_annual_caps(scenario_name: str, years: list[int]) -> list[float]:
    """
    Load annual emissions caps from the processed cap file, aligned to model years.
    """
    if not CAP_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CAP_PATH}. Run scripts/00_build_emissions_cap.py first."
        )

    df = pd.read_csv(CAP_PATH)
    df = df[df["scenario"] == scenario_name].copy()
    df = df.sort_values("year")

    # Align exactly to model years
    df = df[df["year"].isin([int(y) for y in years])]
    df = df.sort_values("year")

    caps = df["cap_tco2"].astype(float).tolist()

    if len(caps) != len(years):
        raise ValueError(
            f"Cap length {len(caps)} != number of model years {len(years)}. "
            f"Check missing years in emissions_cap.csv for scenario={scenario_name}."
        )

    return caps

# ============================================================
# CONFIG
# ============================================================

# VoLL levels to sweep (USD/MWh)
VOLL_LEVELS = [1_000, 2_000, 3_000, 5_000, 7_500, 10_000, 12_500, 15_000]

# Reliability constraint eps (applied at all VoLL levels)
RELIABILITY_EPS = 0.10

# Scenarios: NDC3 unconditional at tight capital, with and without EaaS
FINANCING_ARMS = [
    {"label": "public_only", "financing_regime": "public",  "capital_case": "tight",    "required_margin": 1.10},
    {"label": "eaas",        "financing_regime": "eaas",    "capital_case": "tight",    "required_margin": 1.10},
]

NDC_SCENARIO = "ndc3_unconditional"


def main():
    rows = []

    total_solves = len(VOLL_LEVELS) * len(FINANCING_ARMS)
    print(f"\nVoLL Sensitivity: {len(VOLL_LEVELS)} levels × {len(FINANCING_ARMS)} arms = {total_solves} solves")
    print(f"VoLL levels: {VOLL_LEVELS} USD/MWh")
    print(f"NDC scenario: {NDC_SCENARIO}")
    print(f"Reliability eps: {RELIABILITY_EPS}")

    for voll in VOLL_LEVELS:
        for arm in FINANCING_ARMS:
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

            # Load econ and override VoLL
            econ = load_econ("voll_mid")
            econ["UNSERVED_ENERGY_PENALTY"] = voll * 1e6  # convert USD/MWh to USD/TWh

            years = scenario["years"]
            caps = load_annual_caps(NDC_SCENARIO, years)
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
                    reliability_max_unserved_fraction=RELIABILITY_EPS,
                    reliability_mode="total",
                    solar_capex_by_year=solar_capex_tv,
                )
                status = solve_model(m)

                if not status["optimal"]:
                    rows.append({
                        "voll_usd_per_mwh": voll,
                        "financing_arm":    arm["label"],
                        "status":           "infeasible",
                    })
                    print(f"  VoLL={voll:>6}  {arm['label']:>12}  INFEASIBLE")
                    continue

                diag = extract_planning_diagnostics(m, scenario, econ)
                decomp = diag["cost_decomposition"]

                # Check if reliability constraint is binding
                # (if it has a dual/shadow price > 0)
                rel_binding = False
                try:
                    if hasattr(m, "reliability_constraint"):
                        rel_dual = abs(float(m.dual[m.reliability_constraint]))
                        rel_binding = rel_dual > 1e-6
                except:
                    rel_binding = False

                unserved_total = sum(diag["unserved_twh_by_year"].values())
                demand_total = sum(diag["demand_twh_by_year"].values())
                unserved_fraction = unserved_total / demand_total if demand_total > 0 else 0

                rows.append({
                    "voll_usd_per_mwh":     voll,
                    "financing_arm":        arm["label"],
                    "status":               "optimal",
                    "npv_cost_total":       float(pyo.value(m.system_cost_npv)),
                    "real_expenditure":     decomp["real_expenditure_npv"],
                    "voll_penalty":         decomp["voll_penalty_npv"],
                    "voll_share":           decomp["voll_penalty_share"],
                    "unserved_total_twh":   unserved_total,
                    "unserved_fraction":    unserved_fraction,
                    "reliability_binding":  rel_binding,
                    "solar_total_mw":       sum(
                        float(pyo.value(m.solar_public_add[t]))
                        + float(pyo.value(m.solar_eaas_add[t]))
                        for t in range(len(years))
                    ),
                    "storage_mwh":          float(pyo.value(m.storage_capacity_mwh[len(years)-1])),
                })

                cost_b = float(pyo.value(m.system_cost_npv)) / 1e9
                voll_pct = decomp["voll_penalty_share"] * 100
                print(f"  VoLL={voll:>6}  {arm['label']:>12}  "
                      f"cost=${cost_b:.1f}B  voll_share={voll_pct:.0f}%  "
                      f"unserved={unserved_total:.2f}TWh  "
                      f"rel_binding={'YES' if rel_binding else 'no'}")

            except Exception as e:
                rows.append({
                    "voll_usd_per_mwh": voll,
                    "financing_arm":    arm["label"],
                    "status":           f"error: {e}",
                })
                print(f"  VoLL={voll:>6}  {arm['label']:>12}  ERROR: {e}")

    # ── Save results ──────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "voll_sensitivity.csv", index=False)

    # ── Console summary ───────────────────────────────────────────
    optimal = df[df["status"] == "optimal"].copy()

    print(f"\n{'='*70}")
    print(f"  VOLL SENSITIVITY: HEADLINE SUMMARY")
    print(f"{'='*70}")
    print(f"  {'VoLL':>8}  {'Arm':>12}  {'Cost($B)':>10}  {'Real($B)':>10}  "
          f"{'VoLL($B)':>10}  {'VoLL%':>6}  {'Unserved':>8}  {'Rel bind':>9}")
    print(f"  {'-'*80}")

    for _, row in optimal.sort_values(["voll_usd_per_mwh", "financing_arm"]).iterrows():
        print(f"  {row['voll_usd_per_mwh']:>8.0f}  {row['financing_arm']:>12}  "
              f"{row['npv_cost_total']/1e9:>10.1f}  {row['real_expenditure']/1e9:>10.1f}  "
              f"{row['voll_penalty']/1e9:>10.1f}  {row['voll_share']*100:>5.0f}%  "
              f"{row['unserved_total_twh']:>7.2f}  "
              f"{'YES' if row['reliability_binding'] else 'no':>9}")

    # ── EaaS savings by VoLL level ────────────────────────────────
    print(f"\n  {'VoLL':>8}  {'Public cost':>12}  {'EaaS cost':>12}  {'Savings':>12}  {'Savings %':>10}")
    print(f"  {'-'*60}")
    for voll_val in VOLL_LEVELS:
        pub = optimal[(optimal["voll_usd_per_mwh"] == voll_val) & (optimal["financing_arm"] == "public_only")]
        eaas = optimal[(optimal["voll_usd_per_mwh"] == voll_val) & (optimal["financing_arm"] == "eaas")]
        if len(pub) and len(eaas):
            pub_cost = pub.iloc[0]["npv_cost_total"]
            eaas_cost = eaas.iloc[0]["npv_cost_total"]
            saving = pub_cost - eaas_cost
            saving_pct = saving / pub_cost * 100 if pub_cost > 0 else 0
            print(f"  {voll_val:>8.0f}  ${pub_cost/1e9:>10.1f}B  ${eaas_cost/1e9:>10.1f}B  "
                  f"${saving/1e9:>10.1f}B  {saving_pct:>9.1f}%")

    # ── Transition point ──────────────────────────────────────────
    if "reliability_binding" in optimal.columns:
        binding_rows = optimal[optimal["reliability_binding"] == True]
    else:
        binding_rows = pd.DataFrame()
    if len(binding_rows):
        min_binding_voll = binding_rows["voll_usd_per_mwh"].min()
        print(f"\n  Reliability constraint becomes binding at VoLL ≤ ${min_binding_voll:,.0f}/MWh")
    else:
        print(f"\n  Reliability constraint is NON-BINDING at all tested VoLL levels")
        print(f"  (VoLL penalty alone drives reliability even at ${min(VOLL_LEVELS):,}/MWh)")

    print(f"\nSaved: {RESULTS_DIR / 'voll_sensitivity.csv'}")


if __name__ == "__main__":
    main()
