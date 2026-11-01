"""
27_run_hydro_minflow_sensitivity.py -- how much does the binding hydro
                                        min-flow floor move the system?
=====================================================================

BACKGROUND
----------
`docs/task_hydro_gate_bucketA.md` verified `m.hydro_min_flow` BINDS at
the central configuration (`hydro_flat`): `wet_night` sits exactly on
its floor in 2025, and both day slices sit exactly on theirs in the
terminal year. `hydro_min_flow_fraction` (currently 0.50) is classified
ASSUMED, not SOURCED (`src/scenarios.py`, full documented search there).
An assumed, load-bearing parameter needs a sensitivity band around it --
this script provides it.

METHOD
------
Three solves at the central sliced-baseline configuration (verbatim from
`results/baseline_sliced/summary__02c17f55.json`), varying ONLY
`hydro_min_flow_fraction` in {0.30, 0.50, 0.70}. Pattern follows
`22_salvage_sensitivity.py`: `load_scenario()` is called ONCE, then each
arm is a `copy.deepcopy()` of that single base scenario with one key
mutated -- not a fresh `load_scenario()` call per arm -- so the internal
control (assert only that one key differs) is meaningful rather than
trivially true.

hydro_min_flow_fraction is NOT a required key, is NOT in any arm
registry, and this script does not make it one -- it is mutated directly
on the loaded scenario dict, exactly as `22_salvage_sensitivity.py`
mutates `scenario["asset_lifetimes"]`.

SWEEP BAND -- DECLARED, NOT SOURCED
------------------------------------
0.30 / 0.50 (central) / 0.70. This is a DECLARED band around an ASSUMED
central value, not a sourced range -- see src/scenarios.py's
"hydro_min_flow_fraction" entry for why no sourced range exists (a
documented search found none; the nearest analogue, JICA's Shiroro
release rule, is not convertible into this parameter). +/-0.20 around
0.50 was chosen to be a materially wide band without leaving the [0, 1]
domain the fraction is defined on.

MATERIALITY THRESHOLD -- set before any result is read
-------------------------------------------------------
The sweep is MATERIAL if the objective (npv_total_cost_usd) moves more
than 0.5% across the 0.30-0.70 band, or if storage at 2045
(storage_capacity_mwh_2045) moves more than 2% of its central (0.50)
value. Below both thresholds, hydro_min_flow_fraction is declared
IMMATERIAL at the central configuration -- fixed here, before solving,
so the verdict cannot be read backward from the result.

OUTPUT
------
    results/hydro_minflow/hydro_minflow_<value>.json   -- one per arm
    results/hydro_minflow/hydro_minflow_sweep_0.30_0.50_0.70.csv
                                                         -- all three, long format

Zeus runs this script. It is not run here.
"""

import argparse
import copy
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.append(str(ROOT))

from src.scenarios import load_scenario, MODEL_END_YEAR, om_scenarios
from src.cli_arms import _arm_specs, resolve_results_dir
from src.optimize_model import solve_model
from src.optimize_model_sliced import (
    build_model_sliced, annual_totals, conversion_diagnostic,
    slice_hours, SLICES, _hydro_season_energy_share,
)
from src.io import load_econ, load_solar_capex_by_year
from src.utils import json_safe

# TASK 54 follow-up: outputs go to RESULTS_BASE/arms_<hash8>/ (G7-4: the O&M arm, now a flag, must reach the
# key, else the om_central and om_none runs would overwrite each other). RESULTS_DIR is set in main().
RESULTS_BASE = ROOT / "results" / "hydro_minflow"
RESULTS_BASE.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = RESULTS_BASE

# Reference run for the reproduction guard -- must not be written to.
BASELINE_SLICED_DIR = ROOT / "results" / "baseline_sliced"
REFERENCE_SUMMARY_PATH = BASELINE_SLICED_DIR / "summary__6046f759.json"   # TASK 53: the post-epoch central anchor (hash moved by 2.5)

CANONICAL_VOLL = "voll_mid"
CANONICAL_GENSET = "genset_central"

# ============================================================
# CENTRAL CONFIGURATION -- read verbatim from
# results/baseline_sliced/summary__02c17f55.json (its own top-level
# fields name every load_scenario() arm the central sliced baseline
# uses), mapped onto load_scenario()'s actual parameter names via
# scripts/01b_run_baseline_sliced.py's own load_scenario() call (the
# JSON's "peak_case" is load_scenario()'s "peak_multiple_case", etc.).
# Fixed, non-swept load_scenario() arguments (demand_level_case,
# demand_case, capital_case, solar_build_case, land_case, carbon_case,
# start_year, end_year, reserve_margin) are 01b's own hardcoded values,
# confirmed by reading that runner's load_scenario() call directly --
# not retyped from TASK.md.
# ============================================================
CENTRAL_ARGS = dict(
    demand_level_case="served",
    demand_case="organic_central",
    capital_case="unconstrained",
    gas_deliverability_case="baseline",
    solar_build_case="deployment_unconstrained",
    land_case="loose",
    carbon_case="no_policy",
    start_year=2025,
    end_year=MODEL_END_YEAR,
    peak_multiple_case="peak_unsuppressed",
    reserve_margin="nirp_spinning",
    storage_cc_case="storage_cc_central",
    gas_build_case="gas_unconstrained",
    new_gas_deliverability="constrained",
    gas_availability_case="gas_avail_nameplate",
    gas_retirement_case="gas_retire_vintage_30",
    gas_capex_case="gas_capex_central",
    ccgt_efficiency_case="ccgt_eff_central",
    capital_recovery_case="cr_central",
    dscr_case="dscr_central",
    us_inflation=0.02,
    capital_treatment_case="f1_tranche",
    concessional_envelope_case="env_dares",
    conversion_case="conv_central",
    hydro_case="hydro_flat",
    tariff_case="tariff_central",
    # TASK 54 follow-up: om_case is now a proper arm (--om-case), default om_central = the central
    # configuration everywhere else. Until then this script was hard-wired to om_none (a pre-O&M reference
    # kept so a guard hardcoded to the pre-O&M anchor still held); register L20's three objectives are om_none.
    # `--om-case om_none` reproduces that configuration on the corrected model. This value is the DEFAULT; a
    # driver that checks this dict against run_phase_b.sh's C_* values compares it as written.
    om_case="om_central",
)

# Reference objective for the reproduction guard (below). Read AT RUN TIME from the post-epoch central
# anchor (REFERENCE_SUMMARY_PATH) when --om-case is om_central, because the 0.50 arm of this sweep IS the
# central configuration. For om_none no reference exists on the corrected model (the pre-epoch 21,351,610,495.35
# was produced under the `advance` window and flat storage capex); the guard then only prints.
REFERENCE_NPV_TOTAL_COST_USD = None

# Declared band, not sourced -- see module docstring.
HYDRO_MIN_FLOW_VALUES = [0.30, 0.50, 0.70]
CENTRAL_HYDRO_MIN_FLOW = 0.50

# Materiality thresholds, fixed before any result exists -- see module docstring.
OBJECTIVE_MATERIALITY_PCT = 0.005      # 0.5%
STORAGE_2045_MATERIALITY_PCT = 0.02    # 2%

# Relative tolerance for the at-floor comparison.
AT_FLOOR_REL_TOL = 1e-6

# Reported years for per-slice hydro detail (index into `years` is
# resolved per arm from the model's own year list -- never hardcoded).
REPORTED_YEARS = (2025, 2045, "terminal")

OBSERVED_GAS_2025_TWH_E = 25.62  # same backcast reference 01b prints


def _assert_structurally_equal(a, b, path):
    """
    Raise immediately if `a` and `b` are not structurally identical.
    Replaces a bare `!=`, which raises `ValueError` on a numpy ndarray
    (ambiguous truth value) instead of comparing it -- see "Guard fix" in
    docs/task_hydro_minflow_sensitivity.md for the type inventory this is
    built from (every type actually present in load_scenario()'s output).

    Handles, in order: numpy ndarray (`np.array_equal`); dict (key sets
    compared first, then recurse per key); list (lengths compared first,
    then recurse elementwise); `None` (identity); plain scalars
    (float/int/str/bool, via `==`).

    No pandas Series/DataFrame branch exists: Step 1's inventory found
    none anywhere in load_scenario()'s output (`src/scenarios.py` imports
    numpy only, never pandas), so speculative, untested handling for a
    type that cannot currently occur is deliberately NOT added. If one
    ever appears, it falls through to the final branch below and raises
    -- it does NOT fall through to a permissive "assume equal" default.

    `path` is the key/index path accumulated so far, used only to name
    the exact offending location in the raised error. No broad
    try/except anywhere in this function or its caller: every mismatch,
    at any depth, propagates as an uncaught exception.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            raise AssertionError(
                f"{path}: type mismatch, {type(a).__name__} vs {type(b).__name__}"
            )
        if not np.array_equal(a, b):
            raise AssertionError(f"{path}: ndarray values differ")
        return

    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            raise AssertionError(
                f"{path}: type mismatch, {type(a).__name__} vs {type(b).__name__}"
            )
        a_keys, b_keys = set(a.keys()), set(b.keys())
        if a_keys != b_keys:
            raise AssertionError(f"{path}: dict key sets differ: {a_keys ^ b_keys}")
        for k in a_keys:
            _assert_structurally_equal(a[k], b[k], f"{path}.{k}")
        return

    if isinstance(a, list) or isinstance(b, list):
        if not (isinstance(a, list) and isinstance(b, list)):
            raise AssertionError(
                f"{path}: type mismatch, {type(a).__name__} vs {type(b).__name__}"
            )
        if len(a) != len(b):
            raise AssertionError(f"{path}: list lengths differ: {len(a)} vs {len(b)}")
        for i, (av, bv) in enumerate(zip(a, b)):
            _assert_structurally_equal(av, bv, f"{path}[{i}]")
        return

    if a is None or b is None:
        if a is not b:
            raise AssertionError(f"{path}: values differ: {a!r} != {b!r}")
        return

    if isinstance(a, (float, int, str, bool)) and isinstance(b, (float, int, str, bool)):
        if a != b:
            raise AssertionError(f"{path}: values differ: {a!r} != {b!r}")
        return

    raise TypeError(
        f"{path}: unrecognised type(s) {type(a).__name__} / {type(b).__name__} "
        "in scenario-dict comparison. No handling exists for this type "
        "(numpy ndarray, dict, list, None, and float/int/str/bool are the "
        "only types Step 1's inventory found in load_scenario()'s output) "
        "-- refusing to fall through to a permissive default. Add explicit "
        "handling above before trusting this comparison."
    )


def assert_scenarios_differ_only_in_hydro_min_flow(base_scenario, scenario, arm_label):
    """
    Internal control (TASK.md guard 1): the three arms must be identical
    except for hydro_min_flow_fraction. Compares the deepcopied+mutated
    `scenario` against the unmutated `base_scenario`, key by key,
    structurally (`_assert_structurally_equal`) rather than with a bare
    `!=` -- the original form of this guard raised `ValueError` on
    `scenario["years"]` (a numpy ndarray) before any solve ran.
    """
    base_keys = set(base_scenario.keys())
    scenario_keys = set(scenario.keys())
    if base_keys != scenario_keys:
        raise AssertionError(
            f"[{arm_label}] scenario key sets differ from the base scenario: "
            f"{base_keys ^ scenario_keys}"
        )
    for key in base_keys:
        if key == "hydro_min_flow_fraction":
            continue
        _assert_structurally_equal(
            base_scenario[key], scenario[key],
            f"[{arm_label}] scenario differs from the base scenario beyond "
            f"the intended hydro_min_flow_fraction mutation, at key '{key}'"
        )


def hydro_min_flow_rhs(scenario, t, s, p, H, season_hours):
    """
    Exact replica of the RHS in `m.hydro_min_flow`
    (`src/optimize_model_sliced.py`):

        hydro_min_frac * hydro_annual[t] * hydro_season_share[s]
        * (H[(s, p)] / season_hours[s])

    Computed independently from the scenario dict and calendar
    constants, not read off the model -- this IS the comparison the
    at-floor check needs (model value vs. independently recomputed
    bound).
    """
    hydro_min_frac = scenario["hydro_min_flow_fraction"]
    hydro_annual_t = float(scenario["hydro_energy_twh_by_year"][t])
    season_share = _hydro_season_energy_share(scenario)
    return (
        hydro_min_frac * hydro_annual_t * season_share[s]
        * (H[(s, p)] / season_hours[s])
    )


def per_slice_hydro_detail(m, scenario, t, H, season_hours):
    """One row per slice at year-index t: actual hydro_gen, the floor
    RHS, their ratio, and the at_floor flag (relative tolerance
    AT_FLOOR_REL_TOL)."""
    rows = []
    for (s, p) in SLICES:
        actual = float(pyo.value(m.hydro_gen[t, (s, p)]))
        rhs = hydro_min_flow_rhs(scenario, t, s, p, H, season_hours)
        ratio = (actual / rhs) if rhs != 0 else None
        at_floor = (
            abs(actual - rhs) <= AT_FLOOR_REL_TOL * abs(rhs)
            if rhs != 0 else (actual == 0)
        )
        rows.append({
            "slice": f"{s}_{p}",
            "hydro_gen_actual_twh_e": actual,
            "floor_rhs_twh_e": rhs,
            "ratio_actual_over_floor": ratio,
            "at_floor": bool(at_floor),
        })
    return rows


def run_arm(hydro_min_flow_fraction, base_scenario, econ, capex_tv, arm_label):
    """Deepcopy the base scenario, mutate hydro_min_flow_fraction only,
    build, solve, and extract every required field. No re-solve."""
    scenario = copy.deepcopy(base_scenario)
    scenario["hydro_min_flow_fraction"] = hydro_min_flow_fraction

    assert_scenarios_differ_only_in_hydro_min_flow(base_scenario, scenario, arm_label)

    m = build_model_sliced(
        scenario=scenario,
        econ=econ,
        emissions_cap=None,
        solar_capex_by_year=capex_tv,
    )
    status = solve_model(m, scenario=scenario)
    if not status["optimal"]:
        raise RuntimeError(f"[{arm_label}] non-optimal: {status}")

    years = [int(y) for y in scenario["years"]]
    T = range(len(years))

    # 2045 index computed from the model's own year list -- never hardcoded.
    t_2025 = years.index(2025)
    t_2045 = years.index(2045)
    t_terminal = len(years) - 1

    H = slice_hours()
    season_hours = {
        s: sum(H[(ss, p)] for (ss, p) in SLICES if ss == s)
        for s in ("dry", "wet")
    }

    hydro_slice_detail = {
        "2025": per_slice_hydro_detail(m, scenario, t_2025, H, season_hours),
        "2045": per_slice_hydro_detail(m, scenario, t_2045, H, season_hours),
        f"terminal_{years[t_terminal]}": per_slice_hydro_detail(
            m, scenario, t_terminal, H, season_hours
        ),
    }

    tot = annual_totals(m, years)
    cumulative_unserved_twh = float(sum(tot["unserved"].values()))

    gas_fleet_2025 = float(tot["gas_generation"][2025])
    _conv_diag = conversion_diagnostic(m, years, scenario)
    gas_conv_2025 = sum(
        _conv_diag[tier]["converted_generation_twh_e_by_year"].get(2025, 0.0)
        for tier in ("A", "B", "C")
    ) if _conv_diag else 0.0
    gas_2025 = gas_fleet_2025 + gas_conv_2025

    npv_total_cost_usd = float(pyo.value(m.system_cost_npv))
    solar_total_built_mw = sum(
        float(pyo.value(m.solar_public_add[t])) + float(pyo.value(m.solar_eaas_add[t]))
        for t in T
    )

    return {
        "hydro_min_flow_fraction": hydro_min_flow_fraction,
        "npv_total_cost_usd": npv_total_cost_usd,
        "final_storage_capacity_mwh": float(pyo.value(m.storage_capacity_mwh[t_terminal])),
        "storage_capacity_mwh_2045": float(pyo.value(m.storage_capacity_mwh[t_2045])),
        "solar_total_built_mw": solar_total_built_mw,
        "cumulative_unserved_twh": cumulative_unserved_twh,
        "backcast_gas_2025_twh_e": gas_2025,
        "backcast_gas_2025_fleet_twh_e": gas_fleet_2025,
        "backcast_gas_2025_converted_twh_e": gas_conv_2025,
        "backcast_deviation_from_observed_pct":
            (gas_2025 - OBSERVED_GAS_2025_TWH_E) / OBSERVED_GAS_2025_TWH_E,
        "hydro_slice_detail": hydro_slice_detail,
        "terminal_year": years[t_terminal],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Hydro min-flow sensitivity at the central sliced configuration.")
    parser.add_argument(
        "--om-case", default=CENTRAL_ARGS["om_case"], choices=tuple(om_scenarios()),
        help="O&M arm fed to load_scenario(om_case=...). Default om_central. om_none reproduces register L20's "
             "configuration (its three objectives were measured pre-O&M). Part of the output key.")
    return parser.parse_args()


def _output_dir(central_args):
    """RESULTS_BASE/arms_<hash8>/ keyed by every arm this run used (src.cli_arms.resolve_results_dir)."""
    ns = SimpleNamespace()
    for kwarg, _flag, dest, *_ in _arm_specs():
        if kwarg in central_args:
            setattr(ns, dest, central_args[kwarg])
    ns.capital_case = central_args["capital_case"]
    ns.solar_build_case = central_args["solar_build_case"]
    return resolve_results_dir(RESULTS_BASE, ns, extra={"hydro_min_flow_values": "_".join(f"{v:.2f}" for v in HYDRO_MIN_FLOW_VALUES)})


def main():
    global RESULTS_DIR, REFERENCE_NPV_TOTAL_COST_USD
    args = parse_args()
    central_args = dict(CENTRAL_ARGS, om_case=args.om_case)
    RESULTS_DIR = _output_dir(central_args)
    print(f"om_case = {args.om_case}")
    if args.om_case == "om_central" and REFERENCE_SUMMARY_PATH.exists():
        REFERENCE_NPV_TOTAL_COST_USD = float(json.load(open(REFERENCE_SUMMARY_PATH))["npv_total_cost_usd"])
    econ = load_econ(CANONICAL_VOLL, genset_case=CANONICAL_GENSET)
    capex_tv = load_solar_capex_by_year(
        scenario_name="solar_low",
        start_year=CENTRAL_ARGS["start_year"],
        end_year=int(CENTRAL_ARGS["end_year"]),
    )

    # load_scenario() called ONCE -- every arm is a deepcopy of this.
    base_scenario = load_scenario(**central_args)

    results = {}
    for hmf in HYDRO_MIN_FLOW_VALUES:
        arm_label = f"{hmf:.2f}"
        print(f"Solving hydro_min_flow_fraction={arm_label} ...")
        results[hmf] = run_arm(hmf, base_scenario, econ, capex_tv, arm_label)
        print(f"  done: npv_total_cost_usd={results[hmf]['npv_total_cost_usd']:,.2f}")

    # ---- Reproduction guard (TASK.md guard 2) --------------------------
    central = results[CENTRAL_HYDRO_MIN_FLOW]
    central_npv = central["npv_total_cost_usd"]
    if REFERENCE_NPV_TOTAL_COST_USD is None:
        print(f"  Central-arm objective {central_npv:,.2f}: no reference value exists on the corrected model "
              f"(TASK 53); record this figure against a central om_none run if one is made.")
    elif abs(central_npv - REFERENCE_NPV_TOTAL_COST_USD) > 1e-6 * abs(REFERENCE_NPV_TOTAL_COST_USD):
        print(
            f"  WARNING: central arm (hydro_min_flow_fraction="
            f"{CENTRAL_HYDRO_MIN_FLOW}) objective {central_npv:,.2f} does "
            f"NOT reproduce the reference {REFERENCE_NPV_TOTAL_COST_USD:,.2f} "
            f"({REFERENCE_SUMMARY_PATH.relative_to(ROOT)}, "
            f"\"npv_total_cost_usd\"). This means the scenario this script "
            f"builds is NOT identical to that run's, for a reason not "
            f"diagnosed here -- it does NOT by itself say which figure is "
            f"right. Continuing."
        )
    else:
        print(f"  Central-arm objective reproduces the reference to within 1e-6 relative.")

    # ---- Deltas against the central (0.50) arm -------------------------
    for hmf, r in results.items():
        r["npv_delta_usd_vs_central"] = r["npv_total_cost_usd"] - central_npv
        r["npv_pct_delta_vs_central"] = (
            (r["npv_total_cost_usd"] - central_npv) / central_npv if central_npv else None
        )
        r["storage_2045_delta_mwh_vs_central"] = (
            r["storage_capacity_mwh_2045"] - central["storage_capacity_mwh_2045"]
        )
        r["storage_2045_pct_delta_vs_central"] = (
            (r["storage_capacity_mwh_2045"] - central["storage_capacity_mwh_2045"])
            / central["storage_capacity_mwh_2045"]
            if central["storage_capacity_mwh_2045"] else None
        )

    # ---- Materiality verdict, thresholds fixed before this point -------
    npv_values = [r["npv_total_cost_usd"] for r in results.values()]
    npv_band_pct = (max(npv_values) - min(npv_values)) / central_npv if central_npv else None
    storage_values = [r["storage_capacity_mwh_2045"] for r in results.values()]
    storage_2045_central = central["storage_capacity_mwh_2045"]
    storage_band_pct = (
        (max(storage_values) - min(storage_values)) / storage_2045_central
        if storage_2045_central else None
    )
    material = (
        (npv_band_pct is not None and npv_band_pct > OBJECTIVE_MATERIALITY_PCT)
        or (storage_band_pct is not None and storage_band_pct > STORAGE_2045_MATERIALITY_PCT)
    )
    verdict = {
        "objective_materiality_threshold_pct": OBJECTIVE_MATERIALITY_PCT,
        "storage_2045_materiality_threshold_pct": STORAGE_2045_MATERIALITY_PCT,
        "objective_band_pct_of_central": npv_band_pct,
        "storage_2045_band_pct_of_central": storage_band_pct,
        "material": material,
        "om_case": args.om_case,
    }
    print(f"\nMateriality verdict: {'MATERIAL' if material else 'IMMATERIAL'}")
    print(f"  objective band: {npv_band_pct:.3%} of central (threshold {OBJECTIVE_MATERIALITY_PCT:.1%})")
    print(f"  storage@2045 band: {storage_band_pct:.3%} of central (threshold {STORAGE_2045_MATERIALITY_PCT:.1%})")

    # ---- Write outputs. Never into results/baseline_sliced/. -----------
    assert RESULTS_DIR != BASELINE_SLICED_DIR
    csv_rows = []
    for hmf, r in results.items():
        for year_label, slice_rows in r["hydro_slice_detail"].items():
            for slice_row in slice_rows:
                csv_rows.append({
                    "hydro_min_flow_fraction": hmf,
                    "npv_total_cost_usd": r["npv_total_cost_usd"],
                    "npv_delta_usd_vs_central": r["npv_delta_usd_vs_central"],
                    "npv_pct_delta_vs_central": r["npv_pct_delta_vs_central"],
                    "final_storage_capacity_mwh": r["final_storage_capacity_mwh"],
                    "storage_capacity_mwh_2045": r["storage_capacity_mwh_2045"],
                    "solar_total_built_mw": r["solar_total_built_mw"],
                    "cumulative_unserved_twh": r["cumulative_unserved_twh"],
                    "backcast_gas_2025_twh_e": r["backcast_gas_2025_twh_e"],
                    "backcast_deviation_from_observed_pct": r["backcast_deviation_from_observed_pct"],
                    "year_label": year_label,
                    **slice_row,
                })

        arm_path = RESULTS_DIR / f"hydro_minflow_{hmf:.2f}.json"
        with open(arm_path, "w") as f:
            json.dump(json_safe(r), f, indent=2)
        print(f"saved -> {arm_path}")

    csv_path = RESULTS_DIR / "hydro_minflow_sweep_0.30_0.50_0.70.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"saved -> {csv_path}")

    verdict_path = RESULTS_DIR / "hydro_minflow_verdict_0.30_0.50_0.70.json"
    with open(verdict_path, "w") as f:
        json.dump(json_safe(verdict), f, indent=2)
    print(f"saved -> {verdict_path}")


if __name__ == "__main__":
    main()
