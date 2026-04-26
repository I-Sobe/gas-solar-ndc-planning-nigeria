"""
REL-1 sensitivity: reliability_mode = 'annual' vs 'total'.

Compares investment trajectories and feasibility thresholds
under year-by-year vs horizon-wide unserved energy constraints.

Expected finding: 'annual' mode is stricter, builds more solar
earlier in horizon, and has a higher (less feasible) threshold
eps at which the system becomes infeasible.
"""

import json
import sys
from pathlib import Path
import pyomo.environ as pyo
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io import load_econ, load_solar_capex_by_year
from src.scenarios import load_scenario
from src.optimize_model import build_model, solve_model
from src.optimize_experiments import (
    extract_planning_diagnostics, reliability_levels_log,
)

#ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "rel1_mode_sensitivity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

econ = load_econ("voll_mid")
solar_capex_tv = load_solar_capex_by_year(scenario_name="solar_low",
                                           start_year=2025, end_year=2045)

rows = []
for mode in ["annual", "total"]:
    for eps in reliability_levels_log():  # [0.20, 0.15, 0.10, 0.08, 0.06, 0.05]
        scenario = load_scenario(
            demand_level_case="served", demand_case="baseline",
            gas_deliverability_case="baseline", capital_case="moderate",
            carbon_case="no_policy", start_year=2025, end_year=2045,
        )
        scenario["solar_min_build_mw_per_year"] = 100.0
        try:
            m = build_model(
                scenario=scenario, econ=econ,
                emissions_cap=1e18,
                reliability_max_unserved_fraction=eps,
                reliability_mode=mode,
                solar_capex_by_year=solar_capex_tv,
            )
            status = solve_model(m)
            if not status["optimal"]:
                rows.append({"mode": mode, "eps": eps, "status": "infeasible"})
                continue
            diag = extract_planning_diagnostics(m, scenario, econ)
            solar_total = sum(
                float(pyo.value(m.solar_public_add[t])) +
                float(pyo.value(m.solar_eaas_add[t]))
                for t in range(len(scenario["years"]))
            )
            rows.append({
                "mode": mode, "eps": eps, "status": "feasible",
                "npv_cost": float(pyo.value(m.system_cost_npv)),
                "solar_total_mw": solar_total,
                "horizon_reliability": diag["horizon_reliability"],
                "worst_year_reliability": diag["worst_year_reliability"],
                "cumulative_unserved_twh": sum(diag["unserved_twh_by_year"].values()),
            })
        except RuntimeError as e:
            rows.append({"mode": mode, "eps": eps, "status": f"error: {e}"})

import pandas as pd
pd.DataFrame(rows).to_csv(RESULTS_DIR / "mode_sensitivity.csv", index=False)
print(pd.DataFrame(rows))