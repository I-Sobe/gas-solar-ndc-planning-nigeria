"""
17_run_monte_carlo.py — REVIVED (TASK 06, docs/task_log_06_vss_bound.md).
Phase B item 18. Moved from scripts/deferred/, where it sat as a
structurally-sound-but-blocked script. Three things changed it from
blocked to live; a fourth and fifth remain found-but-NOT-fixed by this
task, declared below rather than silently carried forward.

FIXED BY TASK 06
-----------------
1. DRAWS ARE NOW CONSUMED, NOT GENERATED. This script's own internal
   sampler (`generate_draws()`, `N_DRAWS`/`SEED` module constants, the
   rejection-sampling loop, and the `demand_growth_prior()` call) is
   DELETED entirely. `--draws-file` (required, no default) points at a
   CSV `30_run_stochastic_redispatch.py` writes (`results/stochastic/
   draws__<hash8>.csv`) -- draw_id/demand_growth/gas_scenario/
   carbon_price, one row per draw. TASK 01 item 17 established that this
   script's old sampler and `30_`'s differed (this script rejection-
   sampled the demand normal at zero; `30_` truncates with `max(0, ·)`;
   `30_` carries a carbon branch and selectable gas weights, this script
   had neither) -- identical seeds would NOT have produced identical
   draws, so sharing the realisation, not matching the samplers, is the
   fix. Consequence: the old Defect B (`demand_growth_prior()`'s stale
   sigma and symmetric-vs-skewed-arms bias, both still-open §3.7 items)
   is now MOOT for any bound built from this script's output -- sigma
   only ever determined the draws, and the draws are now supplied
   externally, from whatever process generated the shared file.

2. LCOE IS FIXED (three sub-fixes, Defect A). Numerator: `m.system_
   cost_npv` less `m.cost_unserved_voll_npv` (the VoLL term is a
   scarcity shadow price, not a cost of supply, and does not belong in
   a cost-per-MWh figure). Denominator: hydro added (read from
   `scenario["hydro_energy_twh_by_year"]`, the same exogenous series
   `build_model()` itself reads -- previously omitted entirely) and
   `m.storage_charge[t]` subtracted (previously only `storage_discharge`
   was added, double-counting every MWh cycled through storage: once
   when generated, again when discharged, with the charging withdrawal
   never netted out). **This fix does not touch `npv_cost`
   (`m.system_cost_npv`) itself, and must not** -- `npv_cost`, VoLL term
   included, is what feeds the VSS bound construction (`31_compute_vss_
   bound.py`), because that term is exactly the recourse price a
   wait-and-see policy is allowed to pay. Only `system_lcoe`, a separate
   diagnostic column, changes.

3. FEASIBILITY IS NO LONGER SILENTLY AVERAGED AWAY. This script
   re-optimises the full LP per draw (capacity free, not fixed), so
   infeasible draws should be rare -- but `WS` (the wait-and-see
   expectation this script computes) is INFINITE if even one draw is
   infeasible, and a feasible-only mean would misreport that as a finite
   number. Per-case feasibility is now an explicit, printed, and
   JSON-recorded field (`ws_valid`), separate from the feasible-draw
   summary statistics, which are not relabelled as "the expectation"
   when infeasible draws exist.

FOUND BY TASK 06, NOT FIXED -- OUT OF ITS NAMED SCOPE
-------------------------------------------------------
4. **`peak_mw` is never recomputed per draw.** `30_run_stochastic_
   redispatch.py` needed an explicit fix for exactly this (register
   L66): a demand-growth draw raises energy demand but, left alone,
   leaves the firm-capacity requirement at its base-case value, so
   demand draws test energy adequacy only (always achievable at a cost,
   since self-generation is unbounded and priced) and never capacity
   adequacy. This script has NEVER applied that fix (grep for
   "peak_mw" in this file before TASK 06: zero hits) -- every draw here
   builds against a STALE peak, unlike `30_`'s post-L66 draws. TASK 06's
   own Part 2 scope (A/B/C above) does not name this defect, and it is
   not fixed here. Consequence: `WS`, as this script currently computes
   it, may understate the true wait-and-see cost, in a direction and
   magnitude not established, because capacity-adequacy pressure from a
   high-demand draw is not being tested at all.

5. **This script still calls the ANNUAL `build_model`, not `build_model_
   sliced`** -- `30_` (the `ECP_free` leg) uses the sliced builder. The
   two builders differ in representable scope (no conversion mechanism,
   no hydro dispatch, and storage cannot value energy arbitrage under
   the annual builder -- established throughout this project, most
   recently docs/task_log_01_omnibus_inventory.md). A `WS` computed on
   the annual builder and an `ECP_free` computed on the sliced builder
   are not directly comparable: any gap between them would conflate the
   VALUE OF INFORMATION this task's bound is meant to isolate with a
   BUILDER-CAPABILITY difference that has nothing to do with
   information. Porting this script's builder is out of this task's
   named scope (TASK 06 Part 2 A/B/C does not mention it, and A5 — the
   dedicated builder-port task — treats a builder change as its own,
   separate risk class). Reported, not changed.

6. **This script's four CASES all apply an NDC emissions cap; `30_`'s
   central/base plan applies none.** `30_run_stochastic_redispatch.py`
   builds its base scenario with `emissions_cap_by_year=None` and
   `carbon_case="no_policy"` -- no cap of any kind. Every one of this
   script's own four `CASES` (below) calls `load_annual_caps(case[
   "ndc_scenario"], years)` and passes a real `emissions_cap_by_year`.
   These are not the same stochastic program: a WS computed under an
   NDC cap and an ECP_free computed under no cap answer different
   questions regardless of any other fix. TASK 06's own Part 3 guards
   (`31_compute_vss_bound.py`) check neither this nor finding 5 above --
   both are reported here because they condition whether ANY current
   pairing of this script's output with `30_`'s is a valid bound input,
   which `31_`'s own four named guards do not catch.

WHAT THIS SCRIPT DOES
-----------------------
For each of four headline cases, re-solves the LP under every draw in
the supplied `--draws-file` (demand growth × gas regime; carbon price is
read but inert, since `carbon_case="no_policy"` is hardcoded below, same
as `30_`'s own convention). Produces properly discounted cost, LCOE,
unserved energy, and solar deployment distributions. Re-optimises the
full LP per draw (capacity plans adapt to each realisation), unlike a
fixed-plan re-dispatch.

PRE-REQUISITES
--------------
    00_build_emissions_cap.py (for the NDC cap CSV the four CASES read)
    A draws file from 30_run_stochastic_redispatch.py
    (results/stochastic/draws__<hash8>.csv)

RUN SEQUENCE
------------
    python scripts/00_build_emissions_cap.py
    python scripts/30_run_stochastic_redispatch.py --n-draws ... (produces the draws file)
    python scripts/17_run_monte_carlo.py --draws-file results/stochastic/draws__<hash8>.csv ...
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io import load_econ, load_solar_capex_by_year
from src.scenarios import (
    load_scenario,
    demand_growth_scenarios,
    asset_lifetime_sweep,
    MODEL_END_YEAR,
)
from src.optimize_model import solve_model, leakage_diagnostic
from src.optimize_model_sliced import build_model_sliced
from src.optimize_experiments import extract_planning_diagnostics, annual_emissions_flat
from src.utils import json_safe
from src.cli_arms import add_standard_arms, standard_arms_kwargs, resolve_results_dir
from src.demand import project_baseline_demand

CANONICAL_VOLL = "voll_mid"
CANONICAL_GENSET = "genset_central"

RESULTS_DIR = ROOT / "results" / "monte_carlo"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAP_PATH = ROOT / "data" / "cost" / "processed" / "emissions_cap.csv"
# load_scenario requires a valid demand_case key, but scenario["demand_growth"]
# is overwritten with the draw immediately after. Derived from the registry so
# the Phase 1.6 rename cannot break this call.
MC_DEMAND_CASE_PLACEHOLDER = next(iter(demand_growth_scenarios()))


def load_annual_caps(scenario_name, years):
    if not CAP_PATH.exists():
        raise FileNotFoundError(f"Missing {CAP_PATH}.")
    df = pd.read_csv(CAP_PATH)
    df = df[df["scenario"] == scenario_name].copy().sort_values("year")
    df = df[df["year"].isin([int(y) for y in years])].sort_values("year")
    caps = df["cap_tco2"].astype(float).tolist()
    if len(caps) != len(years):
        raise ValueError(f"Cap length {len(caps)} != years {len(years)}.")
    return caps


def _recompute_peak_mw(scenario):
    """
    TASK 08, Part C. Copied VERBATIM from scripts/30_run_stochastic_
    redispatch.py's own _recompute_peak_mw() (register L66) -- same
    source (scenarios.py:3036-3048's derivation), not a second,
    independent implementation. Recomputes scenario["peak_mw"] from
    THIS scenario's own demand_growth: a demand-growth draw raises
    energy demand but, left alone, leaves the firm-capacity requirement
    at its base-case value, so draws test energy adequacy only (always
    achievable at a cost, since self-generation is unbounded and priced)
    and never capacity adequacy. This script never applied this fix
    before TASK 08 (zero "peak_mw" hits pre-port, per TASK 06's own
    finding 4) -- every draw here built against a stale peak, unlike
    30_'s post-L66 draws.

    par_block (the block-average peaking ratio) depends only on
    peak_multiple_case's slice weights, not on demand_growth -- read
    unchanged from the scenario dict, never recomputed, never perturbed
    by any draw (same justification as 30_'s own copy).

    Sharing opportunity, not acted on here (TASK 08 Part C: "do not move
    it into src/ in this task"): this function is now duplicated
    byte-for-byte in two runners. It touches no builder and depends only
    on scenario[...] keys both runners already populate identically via
    load_scenario() -- a plain, builder-agnostic helper in src/scenarios.py
    or src/optimize_experiments.py (next to extract_planning_diagnostics,
    which already contains a comparable builder-agnostic helper,
    _slice_total) could hold a single copy for both callers. Proposed,
    not made.
    """
    demand_proj = project_baseline_demand(
        base_demand=scenario["base_demand_twh"],
        growth_rate=scenario["demand_growth"],
        start_year=int(scenario["years"][0]),
        end_year=int(scenario["years"][-1]),
    )["demand"]
    par_block = scenario["par_block"]
    return [
        float(demand_proj[t]) * 1e6 / 8760.0 * par_block
        for t in range(len(scenario["years"]))
    ]


CASES = [
    {"label": "NDC3 uncond (public)", "ndc_scenario": "ndc3_unconditional",
     "capital_case": "tight", "financing_regime": "traditional"},
    {"label": "NDC3 uncond (EaaS)", "ndc_scenario": "ndc3_unconditional",
     "capital_case": "tight", "financing_regime": "eaas"},
    {"label": "NDC3 cond (public)", "ndc_scenario": "ndc3_conditional",
     "capital_case": "moderate", "financing_regime": "traditional"},
    {"label": "NDC3 cond (EaaS)", "ndc_scenario": "ndc3_conditional",
     "capital_case": "moderate", "financing_regime": "eaas"},
    # TASK 08, Part B (docs/task_log_08_ws_comparability.md): additive
    # fifth case, matched field-by-field to 30_run_stochastic_redispatch.
    # py's own Stage-1 scenario configuration (its base_scenario =
    # load_scenario(...) call plus the summary__<hash8>.json standard_
    # arms/own_args this script's own verification run produced) -- not
    # merely "add carbon_case='no_policy' to an existing case." The four
    # existing cases above are UNCHANGED and remain valid WS values for
    # their own (NDC-capped) scenarios; only this fifth case is eligible
    # as a --ws-case input to 31_compute_vss_bound.py's matched guard.
    # ndc_scenario=None -> emissions_cap_by_year=None below (30_'s base
    # solve passes no cap at all, not merely a different one).
    # financing_regime=None -> not overridden on the scenario dict below,
    # left at load_scenario()'s own default ("traditional"), exactly as 30_'s
    # own base_scenario leaves it. TASK 29: NO case sets required_margin any
    # more -- the four NDC cases formerly carried the pre-F1 1.10/1.05
    # literals (the fork dscr_scenarios()'s docstring says was "removed" when
    # dscr_case replaced it); every case now takes the dscr_case-derived value,
    # as 30_ does, so the public and EaaS cases cannot diverge again.
    {"label": "Matched to 30_ central (no NDC)", "ndc_scenario": None,
     "capital_case": "unconstrained", "financing_regime": None},  # TASK 81 Batch 4: matches 30_ and the central anchor (L92); was "moderate"
]


def parse_args():
    # TASK 06, Part 2A (docs/task_log_06_vss_bound.md): this script had NO
    # argparse at all before this task -- every scenario arm below
    # load_scenario() needs (peak_multiple_case, reserve_margin,
    # storage_cc_case, etc.) was simply omitted, which would raise before
    # any solve (register L10, "missing required pre-F1 scenario keys" --
    # the old docstring's own third, downstream-of-the-two-named-defects
    # blocker). Built fresh, add_standard_arms(parser), following every
    # other revived runner's pattern -- gas_deliverability_case excluded
    # since THIS runner's swept gas dimension comes from --draws-file, not
    # the CLI.
    parser = argparse.ArgumentParser(
        description="17_run_monte_carlo: re-optimising Monte Carlo (wait-and-see)"
    )
    add_standard_arms(parser, exclude=["gas_deliverability_case"])
    parser.add_argument(
        "--draws-file", type=str, required=True,
        help="CSV from 30_run_stochastic_redispatch.py (results/stochastic/"
             "draws__<hash8>.csv): draw_id, demand_growth, gas_scenario, "
             "carbon_price. Required, no default -- TASK 01 item 17 "
             "established that this script's own former sampler and 30_'s "
             "produce different draws even from an identical seed, so the "
             "realised sample must be shared, not regenerated.",
    )
    return parser.parse_args()


def load_draws(path):
    """
    Read a draws CSV written by 30_run_stochastic_redispatch.py (TASK 06,
    Part 1). Returns list[dict] with keys demand_growth (float),
    gas_regime (str, read from the CSV's own "gas_scenario" column --
    renamed here only for this script's own internal naming, matching
    its pre-TASK-06 draw-dict shape, not a different quantity).
    carbon_price is read but not applied anywhere in this script
    (carbon_case="no_policy" is hardcoded below, same convention as
    30_run_stochastic_redispatch.py -- inert either way).
    """
    df = pd.read_csv(path)
    required = {"draw_id", "demand_growth", "gas_scenario"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"--draws-file {path} is missing columns: {sorted(missing)}")
    draws = [
        {
            "demand_growth": float(row["demand_growth"]),
            "gas_regime": str(row["gas_scenario"]),
        }
        for _, row in df.sort_values("draw_id").iterrows()
    ]
    print(f"  Loaded {len(draws)} draws from {path}")
    return draws


def main():
    args = parse_args()
    global RESULTS_DIR
    RESULTS_DIR = resolve_results_dir(RESULTS_DIR, args)   # TASK 53 2.5: arm-keyed output directory
    standard_kwargs = standard_arms_kwargs(args)

    draws = load_draws(args.draws_file)
    econ = load_econ(CANONICAL_VOLL, genset_case=CANONICAL_GENSET)

    total_solves = len(draws) * len(CASES)
    print(f"\nMonte Carlo LP-Based Uncertainty Analysis")
    print(f"  Draws:       {len(draws)}  (from {args.draws_file})")
    print(f"  Cases:       {len(CASES)}")
    print(f"  Total LP solves: {total_solves}\n")

    # TASK 08, Part D prerequisite: 31_compute_vss_bound.py's Guard 5
    # needs each case's RESOLVED financing_regime/required_margin (not
    # the None sentinel some cases pass for "do not override") to compare
    # against 30_'s own stage1_scenario_fields. One load_scenario() call
    # per case (no solve) -- financing_regime/required_margin do not
    # depend on the draw (demand_growth/gas_regime), only on the case
    # and the CLI's standard_kwargs, so this is exact, not a placeholder
    # guess, and costs nothing against the solve budget.
    case_fields = {}
    for case in CASES:
        _s = load_scenario(
            demand_level_case="served",
            demand_case=MC_DEMAND_CASE_PLACEHOLDER,
            gas_deliverability_case=draws[0]["gas_regime"],
            capital_case=case["capital_case"],
            solar_build_case="deployment_unconstrained",
            land_case="loose",
            carbon_case="no_policy",
            start_year=2025,
            end_year=MODEL_END_YEAR,
            **standard_kwargs,
        )
        if case["financing_regime"] is not None:
            _s["financing_regime"] = case["financing_regime"]
        # TASK 29: no required_margin override on any case; _s keeps the
        # dscr_case-derived value load_scenario() set (recorded below).
        case_fields[case["label"]] = {
            "capital_case": case["capital_case"],
            "financing_regime": _s["financing_regime"],
            "required_margin": _s["required_margin"],
            "has_emissions_cap": case["ndc_scenario"] is not None,
        }

    all_rows = []

    for case in CASES:
        print(f"  == {case['label']} ==")

        for i, draw in enumerate(draws):
            scenario = load_scenario(
                demand_level_case="served",
                demand_case=MC_DEMAND_CASE_PLACEHOLDER,
                gas_deliverability_case=draw["gas_regime"],
                capital_case=case["capital_case"],
                solar_build_case="deployment_unconstrained",
                land_case="loose",
                carbon_case="no_policy",
                start_year=2025,
                end_year=MODEL_END_YEAR,
                **standard_kwargs,
            )
            scenario["demand_growth"] = draw["demand_growth"]
            # TASK 08, Part C: recompute peak_mw from THIS draw's own
            # demand_growth -- must come after the demand_growth
            # overwrite immediately above (30_'s own ordering: recompute
            # after the draw is applied, never before).
            scenario["peak_mw"] = _recompute_peak_mw(scenario)
            # solar_min_build_mw_per_year override to solar_min_build_default()
            # REMOVED (2.11, Step 1) -- [UNSOURCED] (plan 2.5); load_scenario()'s
            # own 0.0 default now applies (see docs/phase_2_11_minbuild_retirement.md).
            # TASK 08, Part B: None means "do not override" -- leaves
            # load_scenario()'s own default in place, matching 30_'s
            # base_scenario, which never touches either key. Only the
            # fifth (matched) case passes None; the original four keep
            # their own literal overrides, unchanged.
            if case["financing_regime"] is not None:
                scenario["financing_regime"] = case["financing_regime"]
            # TASK 29: no required_margin override on any case (dscr_case-derived).

            years = scenario["years"]
            # TASK 08, Part B: ndc_scenario=None -> no cap at all (30_'s
            # base solve passes emissions_cap_by_year=None directly), not
            # a call to load_annual_caps with some other scenario name.
            caps = load_annual_caps(case["ndc_scenario"], years) if case["ndc_scenario"] is not None else None
            solar_capex_tv = load_solar_capex_by_year(
                scenario_name="solar_low",
                start_year=int(years[0]),
                end_year=int(years[-1]),
            )

            try:
                # TASK 08, Part A: ported from the annual build_model to
                # build_model_sliced -- 30_ (the ECP_free leg) uses the
                # sliced builder, and the two differ structurally, not in
                # resolution (no conversion mechanism, no hydro dispatch,
                # no storage energy-arbitrage value in the annual builder).
                # Called directly, not through a shared src/optimize_
                # experiments.py function with a required builder=
                # keyword -- this script's own call site was the only
                # place choosing a builder, so the edit is confined to
                # here. Signature matches build_model() exactly (same
                # keyword names; build_model_sliced's own docstring:
                # "Signature matches build_model() so runners can swap
                # with one import change") -- no other call-site change
                # needed. Investment variables (solar_public_add,
                # solar_eaas_add, storage_add, gas_add) remain free, no
                # fixed_plan passed -- this script still re-optimises
                # investment per draw, unchanged by the port.
                m = build_model_sliced(
                    scenario=scenario,
                    econ=econ,
                    emissions_cap_by_year=caps,
                    solar_capex_by_year=solar_capex_tv,
                )
                status = solve_model(m, scenario=scenario)

                if not status["optimal"]:
                    all_rows.append({
                        "case": case["label"], "draw": i,
                        "demand_growth": draw["demand_growth"],
                        "gas_regime": draw["gas_regime"],
                        "status": "infeasible",
                        "npv_cost": None, "real_exp": None,
                        "voll_penalty": None, "unserved_twh": None,
                        "solar_total_mw": None, "storage_mwh": None,
                        "system_lcoe": None,
                        "leakage_share": None, "leakage_flagged": None,
                    })
                    continue

                diag = extract_planning_diagnostics(m, scenario, econ)
                decomp = diag.get("cost_decomposition", {})

                # npv_cost is the BOUND's own input (31_compute_vss_bound.py) --
                # m.system_cost_npv, UNCHANGED, VoLL term included. Do not
                # subtract anything from this value; only system_lcoe,
                # below, is corrected (TASK 06, Part 2B).
                npv_cost = float(pyo.value(m.system_cost_npv))
                unserved = sum(diag["unserved_twh_by_year"].values())
                solar_total = sum(
                    float(pyo.value(m.solar_public_add[t]))
                    + float(pyo.value(m.solar_eaas_add[t]))
                    for t in range(len(years))
                )
                storage = float(pyo.value(m.storage_capacity_mwh[len(years) - 1]))

                # TASK 06, Part 2B: LCOE, fixed (Defect A, three sub-fixes).
                # Numerator: system_cost_npv LESS the VoLL term
                # (m.cost_unserved_voll_npv) -- VoLL is a scarcity shadow
                # price, not a cost of supply, and does not belong in a
                # cost-per-MWh figure. Denominator: hydro ADDED, read from
                # the same exogenous series build_model() itself reads
                # (scenario["hydro_energy_twh_by_year"]) -- previously
                # omitted entirely; storage_charge[t] SUBTRACTED --
                # previously only storage_discharge was added, double-
                # counting every MWh cycled through storage (once
                # generated, again when discharged, charging withdrawal
                # never netted out).
                hydro_by_year = scenario["hydro_energy_twh_by_year"]
                lcoe_numerator = npv_cost - float(pyo.value(m.cost_unserved_voll_npv))
                # TASK 08, Part A: m.solar_generation/m.gas_generation/
                # m.storage_discharge/m.storage_charge are indexed (T, S)
                # under build_model_sliced (one entry per year x slice),
                # not (T) as under the annual builder -- m.solar_
                # generation[t] alone is no longer a valid year total and
                # would raise. Read extract_planning_diagnostics()'s own
                # per-year aggregated series instead (it already branches
                # on m.unserved.dim() to sum over slices correctly for
                # either builder), rather than re-deriving the slice sum
                # here a second time. fleet_gas_generation_twh_e_by_year
                # is FLEET-ONLY by its own name (task_conversion_
                # diagnostics, Part 2C) -- it excludes m.convert_
                # generation, structurally absent from the annual builder
                # and so never included here before this port either;
                # under an active conversion arm this denominator still
                # omits converted output, same scope as every other
                # consumer of this field project-wide, not a new gap this
                # port introduces. npv_cost (the bound's own input) does
                # not use this series at all -- only this diagnostic does.
                disc_gen = sum(
                    float(pyo.value(m.DF[t])) * (
                        diag["solar_generation_twh_e_by_year"][int(years[t])]
                        + diag["fleet_gas_generation_twh_e_by_year"][int(years[t])]
                        + float(hydro_by_year[t])
                        + diag["storage_discharge_twh_e_by_year"][int(years[t])]
                        - diag["storage_charge_twh_e_by_year"][int(years[t])]
                    )
                    for t in range(len(years))
                )
                # disc_gen is in TWh (discounted), cost is in USD
                system_lcoe = lcoe_numerator / (disc_gen * 1e6) if disc_gen > 1e-9 else None

                # Leakage guard (2.6, D15) — per-draw column, flat for CSV.
                # No per-draw print: 2000 solves would flood the log. A summary
                # count is printed after the loop instead.
                leak = leakage_diagnostic(m)

                all_rows.append({
                    **annual_emissions_flat(m, years),
                    "case": case["label"], "draw": i,
                    "demand_growth": draw["demand_growth"],
                    "gas_regime": draw["gas_regime"],
                    "status": "optimal",
                    "npv_cost": npv_cost,
                    "real_exp": decomp.get("real_expenditure_npv", None),
                    "voll_penalty": decomp.get("voll_penalty_npv", None),
                    "unserved_twh": unserved,
                    "solar_total_mw": solar_total,
                    "storage_mwh": storage,
                    "system_lcoe": system_lcoe,
                    "leakage_share": leak["leakage_share"],
                    "leakage_flagged": leak["flagged"],
                })

            except Exception as e:
                all_rows.append({
                    "case": case["label"], "draw": i,
                    "demand_growth": draw["demand_growth"],
                    "gas_regime": draw["gas_regime"],
                    "status": f"error: {e}",
                    "npv_cost": None, "real_exp": None,
                    "voll_penalty": None, "unserved_twh": None,
                    "solar_total_mw": None, "storage_mwh": None,
                    "system_lcoe": None,
                    "leakage_share": None, "leakage_flagged": None,
                })

            if (i + 1) % 50 == 0:
                print(f"    ... {i+1}/{len(draws)} draws")

        print(f"    Done: {case['label']}")

    # ── Save raw results ──────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "monte_carlo_raw.csv", index=False)
    # Leakage summary across all draws (2.6, D15). A high flagged fraction
    # means the NDC target is being met on paper by dumping load to diesel in
    # many draws -- the cap boundary, not the plan, is doing the work.
    opt = df[df["status"] == "optimal"]
    if len(opt):
        n_flag = int(opt["leakage_flagged"].sum())
        print(f"\n  LEAKAGE: {n_flag}/{len(opt)} optimal draws flagged "
              f"({n_flag/len(opt):.1%}); mean leakage share "
              f"{opt['leakage_share'].mean():.1%}")

    optimal = df[df["status"] == "optimal"].copy()

    # ── Summary statistics ────────────────────────────────────
    # TASK 06, Part 2C: feasibility is recorded explicitly, not silently
    # excluded. This script re-optimises the full LP per draw, so
    # infeasible draws should be rare -- but WS (the wait-and-see
    # expectation) is INFINITE if even one draw is infeasible, and
    # "cost_mean" computed only over the feasible subset would misreport
    # an infinite expectation as a finite number if printed or read
    # without the n_infeasible/ws_valid fields alongside it. ws_valid ==
    # False means: do not read cost_mean as WS for this case.
    summary_rows = []
    for case_label in [c["label"] for c in CASES]:
        sub = optimal[optimal["case"] == case_label]
        n_infeasible_case = len(df[(df["case"] == case_label) & (df["status"] != "optimal")])
        n_total_case = len(df[df["case"] == case_label])
        ws_valid = (n_infeasible_case == 0) and (len(sub) > 0)
        if len(sub) == 0:
            summary_rows.append({
                "case": case_label, "n_optimal": 0, "n_infeasible": n_infeasible_case,
                "n_draws_total": n_total_case, "ws_valid": False, "cost_mean": None,
            })
            print(f"  WARNING: case {case_label!r} has ZERO optimal draws "
                  f"({n_infeasible_case}/{n_total_case} infeasible) -- WS undefined.")
            continue

        costs = sub["npv_cost"].values
        unserved = sub["unserved_twh"].values
        lcoes = sub["system_lcoe"].dropna().values

        if not ws_valid:
            print(f"  WARNING: case {case_label!r} has {n_infeasible_case}/"
                  f"{n_total_case} infeasible draws -- WS is INFINITE for "
                  f"this case; cost_mean below is the feasible-subset mean "
                  f"ONLY, not the wait-and-see expectation. Do not read it "
                  f"as WS.")

        summary_rows.append({
            "case":          case_label,
            "n_optimal":     len(sub),
            "n_infeasible":  n_infeasible_case,
            "n_draws_total": n_total_case,
            # ws_valid: True iff zero infeasible draws, i.e. cost_mean
            # below IS the wait-and-see expectation (WS) for this case.
            # False: WS is infinite; cost_mean is a feasible-subset
            # statistic only, printed/recorded for diagnosis, not as WS.
            "ws_valid":      ws_valid,
            "cost_mean":     np.mean(costs),
            "cost_std":      np.std(costs),
            "cost_p10":      np.percentile(costs, 10),
            "cost_p50":      np.percentile(costs, 50),
            "cost_p90":      np.percentile(costs, 90),
            "unserved_mean": np.mean(unserved),
            "unserved_p10":  np.percentile(unserved, 10),
            "unserved_p50":  np.percentile(unserved, 50),
            "unserved_p90":  np.percentile(unserved, 90),
            "lcoe_mean":     np.mean(lcoes) if len(lcoes) else None,
            "lcoe_p10":      np.percentile(lcoes, 10) if len(lcoes) else None,
            "lcoe_p50":      np.percentile(lcoes, 50) if len(lcoes) else None,
            "lcoe_p90":      np.percentile(lcoes, 90) if len(lcoes) else None,
            "real_exp_mean": np.mean(sub["real_exp"].dropna().values),
            "voll_mean":     np.mean(sub["voll_penalty"].dropna().values),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "monte_carlo_summary.csv", index=False)

    # ── EaaS necessity ────────────────────────────────────────
    eaas_necessity = {}
    for ndc_label in ["NDC3 uncond", "NDC3 cond"]:
        pub_label = f"{ndc_label} (public)"
        eaas_label = f"{ndc_label} (EaaS)"
        pub_sub = optimal[optimal["case"] == pub_label].set_index("draw")
        eaas_sub = optimal[optimal["case"] == eaas_label].set_index("draw")
        common = pub_sub.index.intersection(eaas_sub.index)
        if len(common) == 0:
            continue
        pub_u = pub_sub.loc[common, "unserved_twh"].values
        eaas_u = eaas_sub.loc[common, "unserved_twh"].values
        pub_c = pub_sub.loc[common, "npv_cost"].values
        eaas_c = eaas_sub.loc[common, "npv_cost"].values
        pub_l = pub_sub.loc[common, "system_lcoe"].dropna().values
        eaas_l = eaas_sub.loc[common, "system_lcoe"].dropna().values

        eaas_necessity[ndc_label] = {
            "prob_eaas_reduces_unserved": round(float(np.mean(eaas_u < pub_u)), 3),
            "prob_eaas_reduces_cost": round(float(np.mean(eaas_c < pub_c)), 3),
            "mean_cost_savings_usd": round(float(np.mean(pub_c - eaas_c)), 0),
            "mean_cost_savings_pct": round(float(np.mean(
                (pub_c - eaas_c) / np.where(pub_c > 0, pub_c, 1)) * 100), 1),
            "mean_lcoe_pub": round(float(np.mean(pub_l)), 2) if len(pub_l) else None,
            "mean_lcoe_eaas": round(float(np.mean(eaas_l)), 2) if len(eaas_l) else None,
            "n_draws": int(len(common)),
        }

    with open(RESULTS_DIR / "eaas_necessity.json", "w") as f:
        json.dump(eaas_necessity, f, indent=2)

    # ── Provenance sidecar (TASK 06, Part 3 prerequisite) ──────
    # Not one of Part 2's named A/B/C fixes. Added because 31_compute_
    # vss_bound.py's Guard 1 ("both legs reference the same draws file
    # by SHA-256, not filename") is unverifiable without it: this script
    # writes to fixed, unhashed filenames (no arm-tag, unlike 30_), so a
    # second run with a different --draws-file silently overwrites the
    # first with no trace of which draws file either run consumed. This
    # sidecar is the minimum needed for 31_ to check Guard 1 against this
    # script's OWN record of what it read, not merely the caller's say-so.
    draws_path_resolved = Path(args.draws_file).resolve()
    draws_file_sha256 = hashlib.sha256(draws_path_resolved.read_bytes()).hexdigest()
    with open(RESULTS_DIR / "monte_carlo_meta.json", "w") as f:
        json.dump({
            "draws_file": str(draws_path_resolved),
            "draws_file_sha256": draws_file_sha256,
            "n_draws": len(draws),
            "standard_arms": standard_kwargs,
            "cases": [c["label"] for c in CASES],
            "ws_valid_by_case": {
                r["case"]: bool(r["ws_valid"]) for r in summary_rows
            },
            # TASK 08, Part D prerequisite: per-case resolved scenario
            # fields (computed above, no extra solve), for 31_'s Guard 5
            # to compare against 30_'s stage1_scenario_fields.
            "case_fields": case_fields,
        }, f, indent=2, default=str)
    print(f"Saved: {RESULTS_DIR / 'monte_carlo_meta.json'}")

    # ── Console summary ───────────────────────────────────────
    print(f"\n{'='*95}")
    print(f"  MONTE CARLO SUMMARY ({len(draws)} draws, LP-based)")
    print(f"{'='*95}")
    print(f"  {'Case':<30} {'Mean($B)':>10} {'P10($B)':>10} {'P50($B)':>10} "
          f"{'P90($B)':>10} {'LCOE mean':>10} {'LCOE P90':>10}")
    print(f"  {'-'*92}")
    for _, r in summary_df.iterrows():
        lcoe_m = f"${r['lcoe_mean']:.1f}" if r['lcoe_mean'] else "n/a"
        lcoe_90 = f"${r['lcoe_p90']:.1f}" if r['lcoe_p90'] else "n/a"
        print(f"  {r['case']:<30} ${r['cost_mean']/1e9:>8.1f}B ${r['cost_p10']/1e9:>8.1f}B "
              f"${r['cost_p50']/1e9:>8.1f}B ${r['cost_p90']/1e9:>8.1f}B "
              f"{lcoe_m:>10} {lcoe_90:>10}")

    print(f"\n  UNSERVED ENERGY (TWh):")
    print(f"  {'Case':<30} {'Mean':>8} {'P10':>8} {'P50':>8} {'P90':>8}")
    print(f"  {'-'*60}")
    for _, r in summary_df.iterrows():
        print(f"  {r['case']:<30} {r['unserved_mean']:>8.1f} {r['unserved_p10']:>8.1f} "
              f"{r['unserved_p50']:>8.1f} {r['unserved_p90']:>8.1f}")

    print(f"\n  EaaS NECESSITY:")
    for ndc, met in eaas_necessity.items():
        print(f"    {ndc}:")
        print(f"      P(reduces unserved): {met['prob_eaas_reduces_unserved']*100:.0f}%")
        print(f"      P(reduces cost):     {met['prob_eaas_reduces_cost']*100:.0f}%")
        print(f"      Mean savings:        ${met['mean_cost_savings_usd']/1e9:.1f}B "
              f"({met['mean_cost_savings_pct']:.0f}%)")
        if met['mean_lcoe_pub'] and met['mean_lcoe_eaas']:
            print(f"      LCOE public:         ${met['mean_lcoe_pub']:.1f}/MWh")
            print(f"      LCOE EaaS:           ${met['mean_lcoe_eaas']:.1f}/MWh")

    # ── Box plots ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    labels_short = [c["label"].replace(" (", "\n(") for c in CASES]
    colors = ["#FF5722", "#2196F3", "#FF9800", "#4CAF50"]

    # Panel 1: NPV Cost
    cost_data = [optimal[optimal["case"] == c["label"]]["npv_cost"].dropna().values / 1e9
                 for c in CASES]
    bp1 = axes[0].boxplot(cost_data, tick_labels=labels_short, patch_artist=True)
    for p, col in zip(bp1["boxes"], colors):
        p.set_facecolor(col); p.set_alpha(0.7)
    axes[0].set_ylabel("NPV System Cost (USD Billions)")
    axes[0].set_title("System Cost Distribution")
    axes[0].tick_params(axis="x", rotation=15, labelsize=8)

    # Panel 2: Unserved
    uns_data = [optimal[optimal["case"] == c["label"]]["unserved_twh"].dropna().values
                for c in CASES]
    bp2 = axes[1].boxplot(uns_data, tick_labels=labels_short, patch_artist=True)
    for p, col in zip(bp2["boxes"], colors):
        p.set_facecolor(col); p.set_alpha(0.7)
    axes[1].set_ylabel("Unserved Energy (TWh)")
    axes[1].set_title("Reliability Distribution")
    axes[1].tick_params(axis="x", rotation=15, labelsize=8)

    # Panel 3: LCOE
    lcoe_data = [optimal[optimal["case"] == c["label"]]["system_lcoe"].dropna().values
                 for c in CASES]
    bp3 = axes[2].boxplot(lcoe_data, tick_labels=labels_short, patch_artist=True)
    for p, col in zip(bp3["boxes"], colors):
        p.set_facecolor(col); p.set_alpha(0.7)
    axes[2].set_ylabel("System LCOE (USD/MWh)")
    axes[2].set_title("LCOE Distribution")
    axes[2].tick_params(axis="x", rotation=15, labelsize=8)

    plt.suptitle(f"Monte Carlo Uncertainty Analysis ({len(draws)} draws, LP-based)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "monte_carlo_boxplots.png", dpi=200, bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "monte_carlo_boxplots.pdf", bbox_inches="tight")
    plt.show()
    print(f"\nSaved: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
