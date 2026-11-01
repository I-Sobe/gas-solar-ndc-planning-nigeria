"""
20_run_cf1_solar_yield_sensitivity.py  —  CF-1
==============================================

RESEARCH QUESTION (CF-1)
------------------------
How sensitive are system cost and solar deployment to the assumed solar
capacity factor? Solar CF is a swept physical parameter because Nigeria's
delivered yield is genuinely uncertain: it spans the north-south irradiance
gradient and is reduced below clean PVOUT by soiling, high-temperature
derating, and availability.

CF is the dominant scalar on solar energy per MW, so an optimistic CF
inflates the solar-displaces-gas result. This runner isolates that effect.

CF BAND (fixed-tilt; matches NREL ATB fixed-tilt CAPEX)
-------------------------------------------------------
SOURCED (Global Solar Atlas annual PVOUT / 8760, ground-mounted large
scale at each site's own optimum fixed tilt; see src/scenarios.py's
"solar_cf" entry for the full site table and provenance):
  low     0.16  — Anambra (southern) measured yield
  central 0.19  — national-plan-consistent (matches NIRP 2024's own 0.19
                  falling to 0.18, Annex F Table 19 with Table 8 CF)
  high    0.20  — Kano (Kumbotso), the national maximum measured site
The retired band (0.18/0.20/0.22) had its central arm (0.20) AT the
national maximum and its high arm (0.22) ABOVE every measured Nigerian
site -- unattainable, and retired rather than kept as an unreachable
bound.

METHOD
------
Everything except solar_cf is held at the baseline configuration used in
01_run_baseline.py (served demand, baseline growth, baseline gas, aggressive
solar build, no carbon policy, non-binding scalar cap). Only solar_cf varies.
Convention follows the rest of the codebase: load_scenario(...) then mutate
the scenario dict for keys that are not load_scenario arguments.

OUTPUT
------
results/cf1/cf_sensitivity.csv  — one row per CF value:
  cf, objective_npv_usd, public_solar_capex_npv_usd,
  total_solar_built_mw, cumulative_unserved_twh, total_emissions_tco2
"""

from pathlib import Path
import argparse
import csv
import sys

import pyomo.environ as pyo

# ── Repo root ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from src.scenarios import load_scenario, asset_lifetime_sweep, solar_min_build_default, MODEL_END_YEAR
from src.optimize_model import solve_model
from src.optimize_model_sliced import build_model_sliced
from src.optimize_experiments import extract_planning_diagnostics
from src.io import load_econ, load_solar_capex_by_year
from src.cli_arms import add_standard_arms, standard_arms_kwargs, resolve_results_dir

# Resolve repo root the same way the other runners do.
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "cf1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_GENSET = "genset_central"
CANONICAL_VOLL = "voll_mid"
# Sourced band (docs/shape_factors_and_dist5.md, Part A3); replaces the
# retired, unsourced [0.18, 0.20, 0.22] -- 0.20 was the national maximum
# and 0.22 was above every measured Nigerian site.
CF_SWEEP = [0.16, 0.19, 0.20]


def parse_args():
    # docs/task_runner_repair.md, Step Three: this script had NO argparse
    # at all before this edit -- its load_scenario() call below either sat
    # on a hardcoded literal (gas_deliverability_case="baseline",
    # om_case="om_central") or omitted a sentinel-required key outright,
    # which would have raised a TypeError before ever reaching
    # build_model(). Built fresh, add_standard_arms(parser) for the full
    # set; capital_case="unconstrained" stays a module-level literal here,
    # matching 01b_'s own convention (capital_case is never a CLI arm) and
    # this script's own stated design ("baseline configuration used in
    # 01_run_baseline.py", module docstring).
    parser = argparse.ArgumentParser(
        description="20_run_cf1_solar_yield_sensitivity: solar capacity-factor sensitivity (CF-1)"
    )
    add_standard_arms(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    global RESULTS_DIR
    RESULTS_DIR = resolve_results_dir(RESULTS_DIR, args)   # TASK 53 2.5: arm-keyed output directory
    standard_kwargs = standard_arms_kwargs(args)

    econ = load_econ(CANONICAL_VOLL, genset_case=CANONICAL_GENSET)

    print("CF-1: solar capacity-factor sensitivity")
    print(f"  CF sweep: {CF_SWEEP}")
    print(f"  total solves: {len(CF_SWEEP)}")

    rows = []
    for cf in CF_SWEEP:
        # Baseline configuration (identical to 01_run_baseline.py) ...
        scenario = load_scenario(
            demand_level_case="served",
            demand_case="organic_central",
            capital_case="unconstrained",
            solar_build_case="deployment_unconstrained",
            land_case="loose",
            carbon_case="no_policy",
            start_year=2025,
            end_year=MODEL_END_YEAR,
            **standard_kwargs,
        )
        # ... with only solar_cf varied (mutate convention, as elsewhere).
        scenario["solar_cf"] = cf
        scenario["solar_min_build_mw_per_year"] = solar_min_build_default()

        years = list(scenario["years"])
        solar_capex_tv = load_solar_capex_by_year(
            scenario_name="solar_low",
            start_year=int(years[0]),
            end_year=int(years[-1]),
        )

        # Plan 0.4, item C: 1e18 sentinel -> None ("no cap", already the
        # builder's own guarded meaning for this argument).
        m = build_model_sliced(
            scenario=scenario,
            econ=econ,
            emissions_cap=None,
            emissions_cap_by_year=None,
            solar_capex_by_year=solar_capex_tv,
        )
        status = solve_model(m, scenario=scenario)
        if not status["optimal"]:
            raise RuntimeError(f"CF-1 solve failed at cf={cf}: {status}")

        diag = extract_planning_diagnostics(m, scenario, econ)

        total_solar_mw = sum(
            float(pyo.value(m.solar_public_add[t]))
            + float(pyo.value(m.solar_eaas_add[t]))
            for t in range(len(years))
        )
        cumulative_unserved = sum(diag["unserved_twh_by_year"].values())

        row = {
            "cf": cf,
            "objective_npv_usd": float(pyo.value(m.system_cost_npv)),
            "public_solar_capex_npv_usd": sum(
                float(pyo.value(m.DF[t]))
                * float(pyo.value(m.solar_public_add[t]))
                * float(pyo.value(m.solar_capex_param[t]))
                for t in range(len(years))
            ),
            "total_solar_built_mw": total_solar_mw,
            "cumulative_unserved_twh": cumulative_unserved,
            "total_emissions_tco2": float(pyo.value(m.emissions)),
        }
        rows.append(row)
        print(
            f"  cf={cf:.2f}  NPV={row['objective_npv_usd']/1e9:.2f}B  "
            f"solar={total_solar_mw/1000:.1f}GW  "
            f"unserved={cumulative_unserved:.2f}TWh  "
            f"emis={row['total_emissions_tco2']/1e6:.2f}Mt"
        )

    out_path = RESULTS_DIR / "cf_sensitivity.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
