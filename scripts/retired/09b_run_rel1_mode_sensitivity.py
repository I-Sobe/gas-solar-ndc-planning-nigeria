"""
RETIRED (Tier work A, step 1) — moved to scripts/retired/, not deleted.
This script sweeps `grid_service_mode` over ['annual', 'total']. That
axis no longer has two live options: the 2.6 docket (item D6) switched
the grid-service obligation from 'total' to 'annual' permanently, and
docs/phase_2_6_handoff.md §3 records it as a standing convention —
"Mode `annual`, never `total` — `total` is slack under a rising deficit"
(under a growing deficit 'total' permits ~95% service in the 2030s and
~30% in the 2050s while still reporting horizon compliance). The
sibling script's own header records the same switch and notes that
results before and after it are not comparable. A sweep whose sole
purpose is to compare 'annual' against a retired mode cannot produce a
reportable result. This script also calls the ANNUAL `build_model`
(storage structurally zero, register L5) and predates the required
pre-F1 scenario keys (register L10). Kept for provenance only.

--- original docstring ---
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
from src.scenarios import load_scenario, asset_lifetime_sweep, MODEL_END_YEAR
from src.optimize_model import build_model, solve_model
from src.optimize_experiments import (
    extract_planning_diagnostics, reliability_levels_log,
)

#ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "rel1_mode_sensitivity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CANONICAL_VOLL = "voll_mid"
CANONICAL_GENSET = "genset_central"


econ = load_econ(CANONICAL_VOLL, genset_case=CANONICAL_GENSET)
solar_capex_tv = load_solar_capex_by_year(scenario_name="solar_low",
                                           start_year=2025, end_year=MODEL_END_YEAR)

rows = []
for mode in ["annual", "total"]:
    for eps in reliability_levels_log():  # [0.20, 0.15, 0.10, 0.08, 0.06, 0.05]
        scenario = load_scenario(
            demand_level_case="served", demand_case="organic_central",
            gas_deliverability_case="baseline", capital_case="moderate",
            carbon_case="no_policy", start_year=2025, end_year=MODEL_END_YEAR,
            om_case="om_central",
        )
        # solar_min_build_mw_per_year override to solar_min_build_default()
        # REMOVED (2.11, Step 1) -- [UNSOURCED] (plan 2.5); load_scenario()'s
        # own 0.0 default now applies (see docs/phase_2_11_minbuild_retirement.md).
        try:
            m = build_model(
                scenario=scenario, econ=econ,
                emissions_cap=1e18,
                max_offgrid_fraction=eps,
                grid_service_mode=mode,
                solar_capex_by_year=solar_capex_tv,
            )
            status = solve_model(m, scenario=scenario)
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
                # unserved is ~0 by construction under a genset case (D11);
                # shortfall is what the obligation actually bounds.
                "cumulative_selfgen_twh": sum(diag["selfgen_twh_by_year"].values()),
                "cumulative_shortfall_twh": (
                    sum(diag["unserved_twh_by_year"].values())
                    + sum(diag["selfgen_twh_by_year"].values())),
            })
        except RuntimeError as e:
            rows.append({"mode": mode, "eps": eps, "status": f"error: {e}"})

import pandas as pd
pd.DataFrame(rows).to_csv(RESULTS_DIR / "mode_sensitivity.csv", index=False)
print(pd.DataFrame(rows))