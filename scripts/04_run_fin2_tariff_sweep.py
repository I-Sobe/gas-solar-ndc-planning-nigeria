"""
04_run_fin2_tariff_sweep.py  —  FIN-2 Tariff Bankability Sweep
===============================================================

RESEARCH QUESTION (FIN-2)
--------------------------
What is the minimum EaaS tariff at which private capital becomes
self-financing — and what is the tariff threshold below which public
subsidy (eaas_subsidy) is required?

The self-financing threshold T* is analytically:

    T* = SOLAR_CAPEX_PER_MW × required_margin / npv_energy_per_mw

where npv_energy_per_mw is the horizon-discounted TWh per MW.

This script verifies T* empirically by solving the full NDC-constrained
optimisation across a tariff grid, for both NDC scenarios:

    ndc3_unconditional  (required_margin=1.10,  capital_case="moderate")
    ndc3_conditional    (required_margin=1.05,  capital_case="expansion")

The conditional scenario has a strictly lower T* (concessional finance
reduces the hurdle rate), so private capital becomes bankable at a lower
tariff. This is the quantitative expression of the conditional NDC
finance mechanism.

WHAT THIS ADDS OVER EXISTING RUNS
-----------------------------------
The canonical runs in 03_run_ndc_eaas.py use tariff=95M USD/TWh, which
is well above T* for both scenarios. This sweep varies the tariff from
below T* to above T* to:
  1. Confirm T* empirically (where eaas_subsidy_npv drops to zero)
  2. Quantify the subsidy required at each below-threshold tariff
  3. Show that the conditional T* is lower than unconditional T*
     — the direct modelled effect of concessional finance

OUTPUTS
--------
  results/fin2/tariff_sweep_results.csv   — one row per (ndc_scenario, tariff)
  results/fin2/tariff_sweep_summary.json  — threshold crossings and key stats

RUN ORDER
----------
  00_build_emissions_cap.py  (must exist)
  01_run_baseline.py         (must exist)
  04_run_fin2_tariff_sweep.py  ← this script
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io import load_econ
from src.scenarios import load_scenario, TARIFF_SWEEP_GRID
from src.optimize_experiments import run_tariff_bankability_sweep

# ============================================================
# CONFIG
# ============================================================

RESULTS_DIR = ROOT / "results" / "fin2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAP_PATH = ROOT / "data" / "cost" / "processed" / "emissions_cap.csv"

# Canonical VoLL for all FIN-2 runs — consistent with 02 and 03.
CANONICAL_VOLL = "voll_mid"

# NDC scenarios to sweep — each has its own required_margin and capital_case,
# which is the core of the conditional vs unconditional finance comparison.
NDC_CASES = {
    "baseline_no_policy": {
        "capital_case":    "moderate",
        "required_margin": 1.10,
        "ndc_cap_scenario": None,
    },
    "ndc3_unconditional": {
        "capital_case":    "moderate",
        "required_margin": 1.10,
        "ndc_cap_scenario": "ndc3_unconditional",
    },
    "ndc3_conditional": {
        "capital_case":    "expansion",
        "required_margin": 1.05,
        "ndc_cap_scenario": "ndc3_conditional",
    },
}


# ============================================================
# MAIN
# ============================================================

def main():

    econ = load_econ(CANONICAL_VOLL)
    all_rows = []

    for ndc_name, cfg in NDC_CASES.items():

        print(f"\n{'='*60}")
        print(f"  FIN-2 sweep: {ndc_name}")
        print(f"  capital_case={cfg['capital_case']}  "
              f"required_margin={cfg['required_margin']}")
        print(f"  tariff levels: {[t//1_000_000 for t in TARIFF_SWEEP_GRID]}M USD/TWh")
        print(f"{'='*60}")

        scenario = load_scenario(
            demand_level_case="served",
            demand_case="baseline",
            gas_deliverability_case="baseline",
            capital_case=cfg["capital_case"],
            carbon_case="no_policy",
            start_year=2025,
            end_year=2045,
        )

        # Activate EaaS — tariff will be overwritten per point in the sweep.
        # required_margin is the scenario-specific concessional/commercial value.
        scenario["financing_regime"] = "eaas"
        scenario["required_margin"] = cfg["required_margin"]

        # Public budget is unconstrained for the sweep — the question being
        # answered is about tariff-driven private bankability, not public capital.
        # If public budget were binding it would confound the tariff signal.
        scenario["public_solar_budget_npv"] = None

        rows = run_tariff_bankability_sweep(
            base_scenario=scenario,
            econ=econ,
            tariff_grid=TARIFF_SWEEP_GRID,
            ndc_cap_scenario=cfg["ndc_cap_scenario"],
            cap_path=str(CAP_PATH),
        )

        for r in rows:
            t_m = r["tariff_m_usd_per_twh"]
            sf = r["is_self_financing"]
            sub = r.get("eaas_subsidy_npv_usd", float("nan"))
            status = r["status"]
            print(
                f"  tariff={t_m:5.0f}M  "
                f"self-financing={str(sf):5}  "
                f"subsidy_npv={sub/1e6:8.2f}M  "
                f"[{status}]"
            )

        all_rows.extend(rows)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    csv_path = RESULTS_DIR / "tariff_sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ── Summary: threshold crossing per NDC scenario ───────────────────────────
    summary = {}
    for ndc_name in NDC_CASES:
        subset = df[df["ndc_scenario"] == ndc_name]
        optimal = subset[subset["status"] == "optimal"]

        # Analytical threshold (same for all rows within a scenario)
        threshold = optimal["threshold_usd_per_twh"].iloc[0] if len(optimal) else None

        # Empirical crossing: lowest tariff where eaas_subsidy_npv_usd == 0
        # (or effectively zero — below 1000 USD NPV)
        self_fin = optimal[optimal["eaas_subsidy_npv_usd"] < 1000.0]
        empirical_threshold = (
            float(self_fin["tariff_usd_per_twh"].min())
            if len(self_fin) > 0
            else None
        )

        # Max subsidy (at lowest tariff)
        max_subsidy_row = (
            optimal.loc[optimal["eaas_subsidy_npv_usd"].idxmax()].to_dict()
            if len(optimal) > 0 else {}
        )

        summary[ndc_name] = {
            "required_margin":              NDC_CASES[ndc_name]["required_margin"],
            "analytical_threshold_usd_per_twh":  threshold,
            "analytical_threshold_m_usd_per_twh": threshold / 1e6 if threshold else None,
            "empirical_threshold_usd_per_twh":   empirical_threshold,
            "empirical_threshold_m_usd_per_twh": empirical_threshold / 1e6 if empirical_threshold else None,
            "max_subsidy_npv_usd":          max_subsidy_row.get("eaas_subsidy_npv_usd"),
            "max_subsidy_at_tariff_m":      max_subsidy_row.get("tariff_m_usd_per_twh"),
        }

        print(f"\n{ndc_name}:")
        print(f"  Analytical T*:   {threshold/1e6:.3f} M USD/TWh" if threshold else "  T*: N/A")
        print(f"  Empirical T*:    {empirical_threshold/1e6:.3f} M USD/TWh" if empirical_threshold else "  Empirical T*: not crossed in sweep")
        print(f"  Max subsidy NPV: {max_subsidy_row.get('eaas_subsidy_npv_usd', 0)/1e6:.2f} M USD "
              f"at tariff={max_subsidy_row.get('tariff_m_usd_per_twh', '?')}M")

    json_path = RESULTS_DIR / "tariff_sweep_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {json_path}")

    print("\n=== FIN-2 sweep complete ===")
    print("Key thesis results to report:")
    for ndc_name, s in summary.items():
        t_star = s.get("analytical_threshold_m_usd_per_twh")
        print(f"  {ndc_name}: T* = {f'{t_star:.2f}' if t_star is not None else 'N/A'} M USD/TWh  ")
    if len(summary) == 2:
        vals = [s["analytical_threshold_m_usd_per_twh"]
                for s in summary.values() if s.get("analytical_threshold_m_usd_per_twh")]
        if len(vals) == 2:
            delta = abs(vals[0] - vals[1])
            print(f"\n  Delta T* (conditional vs unconditional) = {delta:.3f} M USD/TWh")
            print("  This is the quantified effect of concessional NDC finance")
            print("  on private EaaS bankability.")


if __name__ == "__main__":
    main()
