import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.append(str(ROOT))

from src.scenarios import load_scenario
import pyomo.environ as pyo
from src.optimize_model import solve_model
from src.optimize_model_sliced import (
    build_model_sliced, annual_totals, slice_detail, SLICES
)
from src.io import load_econ, load_solar_capex_by_year
from src.utils import json_safe

CANONICAL_VOLL = "voll_mid"

RESULTS_DIR = ROOT / "results" / "baseline_sliced"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():

    scenario = load_scenario(
        demand_level_case="served",
        demand_case="baseline",
        capital_case="unconstrained",
        gas_deliverability_case="baseline",
        solar_build_case="aggressive",
        land_case="loose",
        carbon_case="no_policy",
        start_year=2025,
        end_year=2045,
    )

    econ = load_econ(CANONICAL_VOLL)
    years = scenario["years"]

    solar_capex_tv = load_solar_capex_by_year(
        scenario_name="solar_low",
        start_year=int(years[0]),
        end_year=int(years[-1]),
    )

    scenario["solar_min_build_mw_per_year"] = 100.0

    # ------------------------------------------------------------
    # Build and solve the SLICED baseline
    # ------------------------------------------------------------
    m = build_model_sliced(
        scenario=scenario,
        econ=econ,
        emissions_cap=1e18,
        emissions_cap_by_year=None,
        solar_capex_by_year=solar_capex_tv,
    )

    status = solve_model(m)
    if not status["optimal"]:
        raise RuntimeError(f"Sliced baseline solve failed: {status}")

    # ------------------------------------------------------------
    # Aggregate sliced operational variables to annual totals.
    # NOTE: extract_planning_diagnostics is annual-only and would crash on the
    # sliced [t,(s,p)] indexing, so the sliced runner builds its diagnostics
    # directly from annual_totals plus the (unchanged) annual variables.
    # ------------------------------------------------------------
    tot = annual_totals(m, years)  # dict: name -> {year: TWh}

    def annual_series(name):
        return {int(y): tot[name][int(y)] for y in years}

    # Duals (annual constraints, unchanged from the annual model).
    gas_shadow = {}
    for t, y in enumerate(years):
        d = m.dual.get(m.gas_balance[t], None)
        gas_shadow[int(y)] = abs(float(d)) if d is not None else None

    # Emissions per year (Expression, annual).
    emissions_by_year = {
        int(y): float(pyo.value(m.emissions_by_year[t]))
        for t, y in enumerate(years)
    }

    # ------------------------------------------------------------
    # Decision variables (all annual — unchanged)
    # ------------------------------------------------------------
    dv = {
        "solar_add_mw_by_year": {
            int(y): float(pyo.value(m.solar_public_add[t]))
                    + float(pyo.value(m.solar_eaas_add[t]))
            for t, y in enumerate(years)
        },
        "storage_add_mwh_by_year": {
            int(y): float(pyo.value(m.storage_add[t]))
            for t, y in enumerate(years)
        },
        "solar_total_built_mw": sum(
            float(pyo.value(m.solar_public_add[t]))
            + float(pyo.value(m.solar_eaas_add[t]))
            for t in range(len(years))
        ),
        "final_storage_capacity_mwh":
            float(pyo.value(m.storage_capacity_mwh[len(years) - 1])),
    }

    npv_total_cost_usd = float(pyo.value(m.system_cost_npv))
    cumulative_unserved_twh = sum(annual_series("unserved").values())
    actual_emissions_tco2_total = float(pyo.value(m.emissions))

    # ------------------------------------------------------------
    # BACKCAST CHECK — printed prominently
    # ------------------------------------------------------------
    gas_2025 = annual_series("gas_generation")[2025]
    print("=" * 55)
    print("SLICED BASELINE — BACKCAST CHECK (2025 vs observed 2024)")
    print(f"  gas generation : {gas_2025:6.2f} TWh_e   (observed 25.62)")
    print(f"  deviation      : {(gas_2025 - 25.62) / 25.62:+6.1%}")
    print(f"  unserved 2025  : {annual_series('unserved')[2025]:.4f} TWh")
    print(f"  objective      : {npv_total_cost_usd / 1e9:.3f} bn USD")
    print(f"  storage 2045   : {dv['final_storage_capacity_mwh']:,.0f} MWh")
    print("=" * 55)

    # ------------------------------------------------------------
    # Per-slice detail for 2025 and 2045 (peak-adequacy inspection)
    # ------------------------------------------------------------
    slice_2025 = slice_detail(m, years, 0)
    slice_2045 = slice_detail(m, years, len(years) - 1)

    # ------------------------------------------------------------
    # Save summary + diagnostics
    # ------------------------------------------------------------
    summary = {
        "cap_scenario": "baseline_no_policy_SLICED",
        "decision_variables": dv,
        "npv_total_cost_usd": npv_total_cost_usd,
        "cumulative_unserved_twh": cumulative_unserved_twh,
        "actual_emissions_tco2_total": actual_emissions_tco2_total,
        "backcast_gas_2025_twh_e": gas_2025,
        "notes": "Sliced baseline (6 slices). No binding emissions cap (1e18).",
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(json_safe(summary), f, indent=2)

    diag_out = {
        "gas_generation_twh_e_by_year": annual_series("gas_generation"),
        "solar_generation_twh_e_by_year": annual_series("solar_generation"),
        "hydro_generation_twh_e_by_year": annual_series("hydro_gen"),
        "storage_discharge_twh_e_by_year": annual_series("storage_discharge"),
        "storage_charge_twh_e_by_year": annual_series("storage_charge"),
        "unserved_twh_by_year": annual_series("unserved"),
        "gas_to_power_twh_th_by_year": annual_series("gas_to_power"),
        "emissions_tco2_by_year": emissions_by_year,
        "gas_shadow_price_usd_per_twh_th_by_year": gas_shadow,
        "slice_detail_2025": slice_2025,
        "slice_detail_2045": slice_2045,
    }
    with open(RESULTS_DIR / "diagnostics.json", "w") as f:
        json.dump(json_safe(diag_out), f, indent=2)

    # ------------------------------------------------------------
    # Yearly timeseries (annual aggregates + annual capacity)
    # ------------------------------------------------------------
    ts = pd.DataFrame({
        "year": [int(y) for y in years],
        "gas_to_power_twh_th": [annual_series("gas_to_power")[int(y)] for y in years],
        "gas_generation_twh_e": [annual_series("gas_generation")[int(y)] for y in years],
        "solar_generation_twh_e": [annual_series("solar_generation")[int(y)] for y in years],
        "hydro_generation_twh_e": [annual_series("hydro_gen")[int(y)] for y in years],
        "storage_discharge_twh_e": [annual_series("storage_discharge")[int(y)] for y in years],
        "storage_charge_twh_e": [annual_series("storage_charge")[int(y)] for y in years],
        "unserved_twh": [annual_series("unserved")[int(y)] for y in years],
        "emissions_tco2": [emissions_by_year[int(y)] for y in years],
        "gas_shadow_usd_per_twh_th": [gas_shadow[int(y)] for y in years],
        "solar_capacity_mw": [float(pyo.value(m.solar_capacity_mw[t])) for t in range(len(years))],
        "gas_capacity_mw": [float(pyo.value(m.gas_capacity_mw[t])) for t in range(len(years))],
        "storage_capacity_mwh": [float(pyo.value(m.storage_capacity_mwh[t])) for t in range(len(years))],
    })
    ts.to_csv(RESULTS_DIR / "timeseries.csv", index=False)

    # Per-slice CSV for the two inspection years
    slice_rows = []
    for label, rows in (("2025", slice_2025), ("2045", slice_2045)):
        for r in rows:
            r2 = dict(r); r2["year"] = label
            slice_rows.append(r2)
    pd.DataFrame(slice_rows).to_csv(RESULTS_DIR / "slice_detail.csv", index=False)

    print("--- Sliced baseline saved ---")
    print("Solar addition (MW/year):", dv["solar_add_mw_by_year"])
    print("Final storage capacity (MWh):", dv["final_storage_capacity_mwh"])


if __name__ == "__main__":
    main()
