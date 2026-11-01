"""
29_run_parameter_uncertainty.py -- Sobol variance decomposition over
PARAMETER uncertainty at the central scenario (task_monte_carlo).

PARAMETER uncertainty vs SCENARIO uncertainty -- keep these separate
--------------------------------------------------------------------
This model is DETERMINISTIC with structured scenario analysis. The
project's 38-arm sweep answers SCENARIO uncertainty -- which discrete
world happens (which gas-deliverability trajectory, which financing
regime, and so on) -- and does it better than random sampling would,
because the sweep's own constraint-relief cross-tab rules out
combinations that do not cohere (e.g. a gas-scarcity arm paired with a
demand-growth arm that assumes the scarcity never bound). Naively
SAMPLING across scenario axes independently reproduces the deferred
scripts/deferred/17_run_monte_carlo.py's own defect: it drew demand
growth and gas regime independently despite a shared driver, generating
incoherent worlds that were never real possibilities.

This script answers a DIFFERENT question: PARAMETER uncertainty -- five
physical/cost quantities whose TRUE value is unknown but FIXED (not a
scenario branch), sampled continuously at the CENTRAL scenario
configuration, decomposed by Sobol variance analysis to find which
parameter's uncertainty drives the output band. The two kinds of
uncertainty must not be conflated, and this script does not attempt to
also cover scenario uncertainty -- every non-sampled scenario arm below
is held at one explicit central choice, passed via the standard CLI
arm surface (src/cli_arms.py), same discipline as every other runner.

THE SAME FAILURE MODE, FOUND INSIDE ONE PARAMETER (task_monte_carlo,
Part 1, 3A)
--------------------------------------------------------------------
Gas fleet availability is not one scenario-dict key -- it is TWO,
`scenario["gas_availability_factor"]` (the annual energy ceiling) and
`scenario["capacity_credit"]["gas"]` (the peak-hour firm-capacity
credit), independently settable in principle but always initialised
EQUAL by load_scenario() because no separately sourced peak-hour figure
exists yet. Sampling only one and leaving the other at whatever the
CLI's placeholder --gas-availability-case arm set is EXACTLY the
deferred 17_'s failure mode, reproduced inside a single Sobol
dimension: a shared driver, moved on one axis while the other silently
stays put. This script sets both together on every draw and asserts
they are equal immediately before each build -- see set_five_parameters()
below.

FIVE PARAMETERS, ALL UNIFORM, ALL AT THE CENTRAL SCENARIO
--------------------------------------------------------------------
A uniform prior is a statement of IGNORANCE between two bounds, not a
claim about likelihood. None of the five parameters below has a
sourced distributional SHAPE -- only sourced (or, for #2/#5, DERIVED)
ENDPOINTS. Uniform is the honest default given that, not a modelling
choice being asserted as more than it is.

  1. gas_availability_factor / capacity_credit["gas"] (SAME value,
     set together): U(0.3061, 0.80). Endpoints SOURCED
     (gas_availability_scenarios(): gas_avail_observed / gas_avail_
     nameplate). Distribution ASSUMED -- gas_avail_central is
     unsourced ([PENDING], register P2-9) so there is no sourced
     central value to place a non-uniform prior around. This parameter
     spans the model's single largest lever (F-5, ~8.60 bn) and has no
     sourced centre -- ITS DOMINATING THE DECOMPOSITION, IF IT DOES, IS
     A STATEMENT ABOUT THE PRIOR'S WIDTH, NOT AN EMERGENT MODEL
     PROPERTY. State this plainly in every report of these results.
  2. gas_eta (fleet thermal efficiency): U(0.277, 0.298). DERIVED --
     eta=0.287 is itself derived as observed TWh_e / observed TWh_th
     (NERC 2024, ~805 MMscf/d, "at the low end of the reported 800-860 [NO SOURCE, register L130]
     MMscf/d range" -- docs/merged_correction_plan_v2.md). An earlier
     project phase proposed sweeping exactly {0.277, 0.287, 0.298} for
     this same unresolved measurement boundary and never implemented
     it (task_monte_carlo, Part 1, 3C -- grep confirms no trace of
     0.277/0.298 anywhere in src/ or scripts/ before this file).
     EF_e = EF_th / gas_eta is DERIVED at build time
     (optimize_model_sliced.py:1822) from whatever gas_eta this script
     sets, so sweeping efficiency moves emissions automatically -- no
     separate emissions-factor update exists to forget.
  3. solar_cf: U(0.16, 0.20). SOURCED (Global Solar Atlas, register
     L26 -- the measured Nigerian site range).
  4. O&M level, as a blend fraction U(0, 1) between load_om_cost_by_
     year("om_low", ...) and load_om_cost_by_year("om_high", ...):
     SOURCED endpoints (NREL ATB Advanced/Conservative cost cases).
     NOT a scalar -- see build_om_cost_by_year_blend() below.
  5. conversion_capex_per_kw: U(450, 850) USD/kW. DERIVED (Gas Turbine
     World 2024 greenfield CC-minus-SC increment plus a declared
     brownfield premium -- conversion_scenarios()'s own docstring).
     Sampled ALONE -- see "WHY conversion_capex_per_kw ALONE" below.

GENSET EFFICIENCY IS DELIBERATELY EXCLUDED. Its three published values
(0.3474 / 0.2809 / 0.0511) are different LOAD POINTS on the same
Caterpillar datasheets -- an operating decision (how hard the genset
runs), not a measurement of one uncertain quantity. Sampling between
them would sample an operating choice, not a parameter uncertainty.
Held at econ.load_econ(..., genset_case=CANONICAL_GENSET) = "genset_
central" on every draw, same as every other diagnostic in this
project.

WHY conversion_capex_per_kw ALONE, NOT "somewhere between conv_low and
conv_high" (task_monte_carlo, Part 1, 3F / this task's Part 2, 1.2)
--------------------------------------------------------------------
conversion_scenarios()'s conv_low/conv_high are NOT single-parameter
cost arms -- each BUNDLES three parameters into a pessimistic/
optimistic package:
    conv_central: capex_per_kw=600.0, uplift=0.50, retrofit_efficiency=0.46
    conv_low:     capex_per_kw=850.0, uplift=0.40, retrofit_efficiency=0.42
    conv_high:    capex_per_kw=450.0, uplift=0.50, retrofit_efficiency=0.50
Interpolating "between the arms" would confound capex, uplift and
efficiency onto one Sobol axis -- the same shared-driver-sampled-as-
one-thing failure this script exists to avoid, in the opposite
direction (three parameters compressed into one, instead of one
parameter split across two un-synchronised keys as in gas availability
above). This script samples conversion_capex_per_kw ALONE over the
range the three existing arms already span, [450, 850] USD/kW, while
HOLDING uplift and retrofit_efficiency at their conv_central values
(0.50, 0.46) on every single draw, regardless of the drawn capex. See
set_five_parameters() for exactly how this is written into the
scenario dict.

THE 748,000-VS-600 DISCREPANCY (this task's Part 2, 1.2) -- resolved,
not fixed here
--------------------------------------------------------------------
An earlier TASK.md (not a task report -- see docs/task_monte_carlo.md,
Part 2, for the exact attribution correction) asserted the OCGT-to-CCGT
conversion omission was "1,941 MW x 748,000 USD/MW =~ 1.45 bn". No
"748" figure, in any unit, appears anywhere in src/, scripts/, or any
data file in this repository (grepped exhaustively, twice, across two
separate task sequences). The objective's actual conversion-capex term
(optimize_model_sliced.py, convert_split_identity) reads
scenario["conversion_capex_per_kw"] -- USD per KILOWATT, not megawatt
-- at exactly 600.0 for conv_central. 748 cannot be reconstructed as a
nameplate-weighted, tier-effective, or arm-averaged figure from
{600, 850, 450} by any combination this file's author could find (the
simple mean is 633.33; a nameplate-weighted blend has no meaning here
since capex_per_kw does NOT vary by tier under any single arm). VERDICT:
748,000 is simply wrong -- not an alternate valid quantity, not a unit
variant of a real figure. It never entered any actual computation
(grep confirms zero occurrences in code), so no PUBLISHED capital
number needs correcting because of it -- it only ever lived in a
TASK.md's prose. Recorded here as an aid to a future reader who
encounters the "1.45 bn" figure in project history and needs to know it
was never real. NOT fixed, edited, or removed from any prior document
-- outside this script's whitelist.

MODEL IS DETERMINISTIC -- this is a SENSITIVITY ANALYSIS, not a
stochastic program
--------------------------------------------------------------------
Every draw below builds and solves an ordinary, deterministic LP at one
fixed point in the 5-dimensional parameter space. "Uncertainty" here
means EPISTEMIC uncertainty about which fixed point is the true one,
not aleatory randomness inside the model itself. Sobol variance
decomposition is applied to the MAP from that 5-D input space to two
scalar outputs; it is a sensitivity-analysis technique borrowed for
this purpose, not a claim that the underlying model is now stochastic.

METHOD -- scipy.stats.sobol_indices, not a hand-built Saltelli
estimator (task_monte_carlo, Part 2, 1.3)
--------------------------------------------------------------------
scipy.stats.sobol_indices (scipy 1.15.2, confirmed installed; SALib is
NOT installed and is not used) implements the complete Saltelli 2010
procedure -- sampling, A/B/AB-hybrid matrix construction, and index
computation -- given only a callable and a list of per-parameter
distributions. Chosen over a hand-written Saltelli estimator because:
  - a wrong index formula in a hand-rolled estimator produces plausible
    numbers with no error signature; scipy's implementation is tested
    library code, not this script's own arithmetic;
  - bootstrap confidence intervals come free via .bootstrap() --
    without them, Sobol indices are easy to over-read;
  - multiple QoIs (system cost, capital mobilised) decompose together
    in one call (s=2), rather than needing two separate manual runs;
  - the solve count is identical either way: n*(d+2).
func(x) below receives x of shape (5, n_batch) up to THREE times (once
each for scipy's internal A, B, and concatenated-AB-hybrid batches --
verified empirically, task_monte_carlo Part 1, 3G) and must loop over
its own n_batch columns internally, since a Pyomo solve cannot be
vectorised. Per-draw results are written to disk from INSIDE that loop
(see the append_draw() call site), not from any outer harness --
scipy's own docs 3-call sequence must not be the checkpointing unit, or
draw 2,900 of 1,792 vanishes if the process dies before the 3rd call
returns.

FAILED-SOLVE POLICY
--------------------------------------------------------------------
solve_model() (src/optimize_model.py) RAISES RuntimeError for any
termination condition other than optimal/feasible -- it does not
return a status to check for that case (task_monte_carlo, Part 1, 3E).
scipy.stats.sobol_indices requires every function evaluation to be
FINITE (NaN is not permitted), so a failed draw cannot simply be
skipped or marked NaN. Each failed draw:
  - is caught by a try/except RuntimeError around the build+solve call,
  - is logged to the JSONL output with status="failed" and the
    exception text,
  - contributes FAILED_SOLVE_PENALTY_USD (a fixed, documented, clearly
    out-of-range sentinel -- see below) to BOTH QoI arrays passed to
    scipy, so the run does not abort and every column stays finite.
The summary printed at the end reports the failure COUNT and FRACTION
prominently. Per the task brief: if that fraction is not small, the
printed summary says so explicitly next to the indices, rather than
presenting the indices as though nothing happened -- a handful of
penalty-valued outliers can distort a variance decomposition batch,
and the reader must be told when that risk applies rather than
discovering it by re-deriving the failure count from the JSONL.

WHAT THIS SCRIPT DOES NOT DO
--------------------------------------------------------------------
It does not edit, import privately from, or otherwise touch any
mutable state in src/. It does not run itself -- see the two commands
in docs/task_monte_carlo.md, Part 2, for the pilot and full invocations
Zeus is to run. It does not attempt to resume a partially-completed
run; the JSONL is an append-only log for post-hoc recovery of
completed draws, not a checkpoint/resume mechanism.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyomo.environ as pyo
from scipy.stats import sobol_indices, uniform

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.cli_arms import add_standard_arms, standard_arms_kwargs, resolve_results_dir
from src.scenarios import load_scenario, MODEL_END_YEAR
from src.optimize_model import solve_model
from src.optimize_model_sliced import build_model_sliced
from src.optimize_experiments import extract_planning_diagnostics
from src.io import load_econ, load_om_cost_by_year, load_solar_capex_by_year

# Same canonical constants 01b_run_baseline_sliced.py uses for its own
# central-configuration diagnostics -- kept identical so this script's
# "central scenario" is the same central scenario every other runner
# means by that phrase, not a fresh redefinition.
CANONICAL_VOLL = "voll_mid"
CANONICAL_GENSET = "genset_central"

# conv_central's own uplift / retrofit_efficiency (conversion_scenarios(),
# src/scenarios.py) -- held fixed on every draw while capex_per_kw is
# swept alone. Literal, not re-derived from conversion_scenarios() at
# runtime, so a future change to conv_central's own values cannot
# silently re-point what this script calls "held at central" -- if
# conv_central's uplift/efficiency ever change, this script's own
# docstring and these two constants go stale together and visibly, not
# silently in sync with an import.
CONV_CENTRAL_UPLIFT = 0.50
CONV_CENTRAL_RETROFIT_EFFICIENCY = 0.46

# Parameter bounds (task_monte_carlo.md, Part 1 + this task's Part 2,
# section 2's table). All uniform; see the module docstring for why.
PARAM_NAMES = [
    "gas_availability_factor",
    "gas_eta",
    "solar_cf",
    "om_blend_fraction",
    "conversion_capex_per_kw",
]
PARAM_BOUNDS = {
    "gas_availability_factor": (0.3061, 0.80),
    "gas_eta": (0.277, 0.298),
    "solar_cf": (0.16, 0.20),
    "om_blend_fraction": (0.0, 1.0),
    "conversion_capex_per_kw": (450.0, 850.0),
}

# Documented sentinel for a failed draw's QoI contribution. Fixed,
# clearly out of range of any real system-cost or capex figure this
# model produces (which sit in the 1e10 range) so a failed draw is
# visually obvious in any printed distribution rather than blending
# into the real one -- and finite, since scipy.stats.sobol_indices
# requires every function value to be finite (NaN is rejected).
FAILED_SOLVE_PENALTY_USD = 1e13

# "Small fraction" threshold for the failed-solve warning in the
# printed summary (task brief: "if more than a small fraction fail,
# the indices are not interpretable and the report must say so"). A
# judgment call, stated explicitly rather than left implicit: 5% is
# used here as the line past which even one or two extra failures
# would visibly distort a 1,792-draw or 112-draw batch's mean/variance.
FAILURE_FRACTION_WARN_THRESHOLD = 0.05


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sobol parameter-uncertainty variance decomposition at the "
            "central scenario (task_monte_carlo). See this file's module "
            "docstring for the parameter table, the method, and the "
            "distinction from the project's 38-arm SCENARIO sweep."
        )
    )
    add_standard_arms(parser)
    parser.add_argument(
        "--n-base", type=int, required=True,
        help=(
            "Sobol base sample size N. Total solves = N*(d+2) = N*7 "
            "(d=5). Must be a power of 2 (scipy.stats.sobol_indices "
            "requirement). No default: an unstated N would silently "
            "pick a run length (and therefore an index-estimate "
            "precision) with no visible signature in the output."
        ),
    )
    parser.add_argument(
        "--seed", type=int, required=True,
        help=(
            "RNG seed passed to scipy.stats.sobol_indices(rng=...). No "
            "default: an unstated seed makes a run silently "
            "non-reproducible relative to a reader's expectation that "
            "runs in this project are."
        ),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Directory for the JSONL draw log and the summary JSON. Default (TASK 53 2.5/2.7): "
             "results/parameter_uncertainty/arms_<hash8>/, keyed by every arm AND n_base and seed. "
             "Whatever the directory, the run ABORTS if it already holds anything but ARMS.json: the "
             "draws log is opened exclusively, never appended to an existing file (register L77: "
             "draws_n256_seed42.jsonl once held three runs).",
    )
    args = parser.parse_args()

    if args.n_base <= 0 or (args.n_base & (args.n_base - 1)) != 0:
        parser.error(f"--n-base must be a power of 2, got {args.n_base}.")

    # Conversion capex (parameter 5) is structurally inert under
    # conv_none (build_model_sliced() builds no conversion mechanism at
    # all when conversion_available=False) and conversion is refused
    # outright under pre_f1_overnight (optimize_model_sliced.py,
    # ValueError). Both would make this script silently sample a
    # parameter that has zero effect on every solve, or crash on the
    # first one -- fail at argument-parsing time instead, loudly, with
    # the reason, matching this project's standing discipline against
    # silent no-op arms.
    if args.conversion_case == "conv_none":
        parser.error(
            "--conversion-case conv_none makes parameter 5 "
            "(conversion_capex_per_kw) a no-op: build_model_sliced() "
            "builds no conversion mechanism at all under conv_none, so "
            "every draw would sample a value nothing reads. Pass "
            "conv_central (or conv_low/conv_high, though central is "
            "what this script's docstring holds uplift/efficiency at)."
        )
    if args.capital_treatment_case != "f1_tranche":
        parser.error(
            "--capital-treatment-case must be f1_tranche: conversion "
            "capex is wired into the F1 per-tranche capital charge only "
            "(optimize_model_sliced.py) and an available conversion arm "
            "under pre_f1_overnight raises ValueError at build time -- "
            "every draw would fail identically, for a reason unrelated "
            "to any sampled parameter."
        )

    return args


def build_om_cost_by_year_blend(om_low, om_high, frac):
    """
    Linear per-technology, per-year blend between two
    load_om_cost_by_year() outputs. Both inputs share identical
    {technology: {year: {...}}} key sets (both loaded over the same
    start_year/end_year from the same CSV's technology list), so the
    blend is well-defined with no missing-key risk (task_monte_carlo,
    Part 1, 3D).
    """
    return {
        tech: {
            yr: {
                "fom_usd_per_kw_yr": (
                    (1.0 - frac) * om_low[tech][yr]["fom_usd_per_kw_yr"]
                    + frac * om_high[tech][yr]["fom_usd_per_kw_yr"]
                ),
                "vom_usd_per_mwh": (
                    (1.0 - frac) * om_low[tech][yr]["vom_usd_per_mwh"]
                    + frac * om_high[tech][yr]["vom_usd_per_mwh"]
                ),
            }
            for yr in om_low[tech]
        }
        for tech in om_low
    }


def set_five_parameters(scenario, x, om_low, om_high):
    """
    Sets all five sampled parameters on an already-loaded scenario
    dict, in place. x is a length-5 array in PARAM_NAMES order.

    Gas availability is set at BOTH keys and asserted equal immediately
    -- task_monte_carlo, Part 1 3A / this task's Part 2, 1.1. The
    assertion is deliberately a plain `assert`, not a soft warning: the
    failure mode it guards is silent BY NATURE (both keys read
    successfully either way; only their divergence is the defect), so
    the guard must stop the run, not log and continue.
    """
    gas_avail, gas_eta, solar_cf, om_frac, conv_capex_kw = x

    scenario["gas_availability_factor"] = float(gas_avail)
    scenario["capacity_credit"]["gas"] = float(gas_avail)
    assert scenario["gas_availability_factor"] == scenario["capacity_credit"]["gas"], (
        "gas availability keys diverged: gas_availability_factor="
        f"{scenario['gas_availability_factor']} != capacity_credit['gas']="
        f"{scenario['capacity_credit']['gas']}. This must never happen -- "
        "both are set from the same local variable two lines above."
    )

    scenario["gas_eta"] = float(gas_eta)
    scenario["solar_cf"] = float(solar_cf)

    scenario["om_cost_by_year"] = build_om_cost_by_year_blend(om_low, om_high, float(om_frac))
    # scenario["om_case"] is NOT updated -- it still reads whatever
    # placeholder string --om-case supplied to satisfy load_scenario()'s
    # required argument. task_monte_carlo, Part 1: the per-draw JSONL
    # row (om_blend_fraction) is the source of truth for the O&M level
    # actually used on this draw, not scenario["om_case"].

    # Conversion capex ALONE (task_monte_carlo, Part 1, 3F; this task's
    # Part 2, 1.2) -- uplift and retrofit_efficiency pinned to
    # conv_central on every draw regardless of the drawn capex, so only
    # ONE parameter varies on this Sobol axis.
    scenario["conversion_capex_per_kw"] = float(conv_capex_kw)
    scenario["conversion_uplift"] = CONV_CENTRAL_UPLIFT
    scenario["conversion_retrofit_efficiency"] = CONV_CENTRAL_RETROFIT_EFFICIENCY
    # scenario["conversion_tiers"] (nameplate_mw / paf per tier) is left
    # untouched -- it is CSV-derived plant data, not a sampled cost or
    # efficiency parameter.


def append_draw(out_path, row):
    """Append one draw's record to the JSONL log and flush immediately
    -- 1,792 solves is far too many to lose at draw 1,700 to a crash
    that an end-of-run write would not survive."""
    with open(out_path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def run_one_draw(x_col, draw_id, call_index, args, om_low, om_high, solar_capex_tv, jsonl_path):
    """
    One full build+solve+extract cycle at one point in parameter space.
    Returns (npv_total_cost_usd, npv_gross_capex_report_window_usd) --
    real values on a successful solve, FAILED_SOLVE_PENALTY_USD for
    both on any solve failure. Always appends a row to jsonl_path
    before returning.
    """
    # Fresh load_scenario() EVERY draw (task brief, section 3, step 1):
    # a scenario dict mutated in place across draws would leak one
    # draw's overrides into the next, silently, and no output field
    # would reveal it happened. Matches 01b_run_baseline_sliced.py's own
    # central-configuration call, plus the standard arm surface for
    # every arm this script does not itself sample.
    scenario = load_scenario(
        demand_level_case="served",
        demand_case="organic_central",
        capital_case="unconstrained",
        solar_build_case="deployment_unconstrained",
        land_case="loose",
        carbon_case="no_policy",
        start_year=2025,
        end_year=MODEL_END_YEAR,
        **standard_arms_kwargs(args),
    )
    econ = load_econ(CANONICAL_VOLL, genset_case=CANONICAL_GENSET)

    set_five_parameters(scenario, x_col, om_low, om_high)

    row = {
        "draw_id": draw_id,
        "call_index": call_index,
        "gas_availability_factor": float(x_col[0]),
        "gas_eta": float(x_col[1]),
        "solar_cf": float(x_col[2]),
        "om_blend_fraction": float(x_col[3]),
        "conversion_capex_per_kw": float(x_col[4]),
    }

    try:
        m = build_model_sliced(
            scenario=scenario,
            econ=econ,
            emissions_cap=None,
            emissions_cap_by_year=None,
            solar_capex_by_year=solar_capex_tv,
        )
        # solve_model() RAISES RuntimeError for anything worse than
        # feasible -- task_monte_carlo, Part 1, 3E. Checking a returned
        # status dict alone would never see that case; it must be
        # caught here.
        status = solve_model(m, scenario=scenario)

        diag = extract_planning_diagnostics(m, scenario, econ, solar_capex_series=solar_capex_tv)
        years = scenario["years"]
        npv_total_cost_usd = float(pyo.value(m.system_cost_npv))
        npv_gross_capex_report_window_usd = float(diag["npv_gross_capex_report_window_usd"])
        final_storage_capacity_mwh = float(pyo.value(m.storage_capacity_mwh[len(years) - 1]))
        solar_total_built_mw = sum(
            float(pyo.value(m.solar_public_add[t])) + float(pyo.value(m.solar_eaas_add[t]))
            for t in range(len(years))
        )

        row.update({
            "status": status["status"],
            "optimal": status["optimal"],
            "error": None,
            "npv_total_cost_usd": npv_total_cost_usd,
            "npv_gross_capex_report_window_usd": npv_gross_capex_report_window_usd,
            "final_storage_capacity_mwh": final_storage_capacity_mwh,
            "solar_total_built_mw": solar_total_built_mw,
        })
        append_draw(jsonl_path, row)
        return npv_total_cost_usd, npv_gross_capex_report_window_usd

    except RuntimeError as e:
        row.update({
            "status": "failed",
            "optimal": False,
            "error": str(e),
            "npv_total_cost_usd": FAILED_SOLVE_PENALTY_USD,
            "npv_gross_capex_report_window_usd": FAILED_SOLVE_PENALTY_USD,
            "final_storage_capacity_mwh": None,
            "solar_total_built_mw": None,
        })
        append_draw(jsonl_path, row)
        return FAILED_SOLVE_PENALTY_USD, FAILED_SOLVE_PENALTY_USD


def main():
    args = parse_args()
    if args.out_dir is None:
        # TASK 53 2.5 (G7-4): this runner had no per-arm key at all. n_base and seed join the key.
        args.out_dir = resolve_results_dir(ROOT / "results" / "parameter_uncertainty", args,
                                           extra={"n_base": args.n_base, "seed": args.seed})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # TASK 53 2.7: the guards run_sobol_clean.sh used to supply, folded into the runner. A pre-existing
    # draws file is never appended to (append_draw opens mode "a", which is how L77's three runs came
    # to share one file): abort on anything in the directory but ARMS.json, then create the log exclusively.
    _existing = sorted(p.name for p in args.out_dir.iterdir() if p.name != "ARMS.json")
    if _existing:
        raise SystemExit(f"ABORT: {args.out_dir} already holds {_existing}; a Sobol run needs a clean directory "
                         "(move the old one aside, or pass a different --out-dir). Nothing was run.")
    jsonl_path = args.out_dir / f"draws_n{args.n_base}_seed{args.seed}.jsonl"
    summary_path = args.out_dir / f"summary_n{args.n_base}_seed{args.seed}.json"
    open(jsonl_path, "x").close()

    # Loaded ONCE, outside the per-draw loop -- unlike the scenario
    # dict, these are pure, side-effect-free reads of static CSVs
    # (om_cost.csv, the solar capex-by-year table) with no mutable
    # state a draw could leak into another. Re-loading them 1,792 times
    # would only add redundant disk I/O, not safety.
    om_low = load_om_cost_by_year("om_low", start_year=2025, end_year=MODEL_END_YEAR)
    om_high = load_om_cost_by_year("om_high", start_year=2025, end_year=MODEL_END_YEAR)
    solar_capex_tv = load_solar_capex_by_year(
        scenario_name="solar_low", start_year=2025, end_year=MODEL_END_YEAR
    )

    draw_counter = {"n": 0}

    def func(x):
        # x: shape (5, n_batch). Called up to 3 times by
        # scipy.stats.sobol_indices (A, B, concatenated AB-hybrid) --
        # verified empirically, task_monte_carlo Part 1 3G. Must loop
        # internally: a Pyomo solve cannot be vectorised.
        n_batch = x.shape[1]
        out = np.empty((2, n_batch))
        call_index = func.n_calls
        func.n_calls += 1
        for j in range(n_batch):
            draw_id = draw_counter["n"]
            draw_counter["n"] += 1
            cost, capex = run_one_draw(
                x[:, j], draw_id, call_index, args, om_low, om_high, solar_capex_tv, jsonl_path
            )
            out[0, j] = cost
            out[1, j] = capex
            print(
                f"draw {draw_id:5d} (call {call_index}, col {j:4d}/{n_batch}): "
                f"cost={cost:.3e}  capex={capex:.3e}",
                flush=True,
            )
        return out

    func.n_calls = 0

    dists = [uniform(loc=PARAM_BOUNDS[p][0], scale=PARAM_BOUNDS[p][1] - PARAM_BOUNDS[p][0])
             for p in PARAM_NAMES]

    total_solves = args.n_base * (len(PARAM_NAMES) + 2)
    print(
        f"Starting Sobol run: n_base={args.n_base}, d={len(PARAM_NAMES)}, "
        f"total solves = n_base*(d+2) = {total_solves}.",
        flush=True,
    )

    res = sobol_indices(func=func, n=args.n_base, dists=dists, rng=args.seed)

    # ---- Summary ----------------------------------------------------
    with open(jsonl_path) as f:
        draws = [json.loads(line) for line in f]

    n_failed = sum(1 for d in draws if d["status"] == "failed")
    n_total = len(draws)
    if n_total != total_solves:
        raise RuntimeError(f"draws file has {n_total} rows; N(d+2) = {total_solves}. The run is not "
                           f"quotable (register L77 is exactly this failure).")
    failure_fraction = n_failed / n_total if n_total else float("nan")

    def qoi_stats(values):
        arr = np.array(values, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "p5": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }

    cost_values = [d["npv_total_cost_usd"] for d in draws]
    capex_values = [d["npv_gross_capex_report_window_usd"] for d in draws]
    storage_values = [d["final_storage_capacity_mwh"] for d in draws if d["final_storage_capacity_mwh"] is not None]
    solar_values = [d["solar_total_built_mw"] for d in draws if d["solar_total_built_mw"] is not None]

    # res.bootstrap()'s .confidence_interval is a scipy ConfidenceInterval
    # namedtuple of numpy arrays (shape (s, d)), not JSON-serialisable as
    # returned, and can contain NaN (BCa is documented to fail on
    # degenerate resamples -- observed directly while building this
    # script on a toy function). Converted to plain nested lists with
    # NaN -> null so the summary JSON is valid rather than merely
    # accepted by Python's own lenient (non-standard) NaN literal.
    def _ci_to_list(ci_side):
        return [[None if (isinstance(v, float) and math.isnan(v)) else float(v) for v in row]
                for row in ci_side]

    bootstrap = res.bootstrap(confidence_level=0.95, n_resamples=999)

    summary = {
        "n_base": args.n_base,
        "d": len(PARAM_NAMES),
        "total_solves": total_solves,
        "seed": args.seed,
        "param_names": PARAM_NAMES,
        "param_bounds": PARAM_BOUNDS,
        "feasibility": {
            "n_total": n_total,
            "n_failed": n_failed,
            "failure_fraction": failure_fraction,
            "warning": (
                f"FAILURE FRACTION {failure_fraction:.1%} EXCEEDS THE "
                f"{FAILURE_FRACTION_WARN_THRESHOLD:.0%} INTERPRETABILITY "
                "THRESHOLD -- the Sobol indices below are contaminated "
                "by penalty-valued draws and MUST NOT be reported as a "
                "clean decomposition without this caveat."
                if failure_fraction > FAILURE_FRACTION_WARN_THRESHOLD
                else "Failure fraction within the interpretability threshold."
            ),
        },
        "qoi_summary": {
            "npv_total_cost_usd": qoi_stats(cost_values),
            "npv_gross_capex_report_window_usd": qoi_stats(capex_values),
            "final_storage_capacity_mwh": qoi_stats(storage_values) if storage_values else None,
            "solar_total_built_mw": qoi_stats(solar_values) if solar_values else None,
        },
        "sobol_indices": {
            "qoi_order": ["npv_total_cost_usd", "npv_gross_capex_report_window_usd"],
            "first_order": res.first_order.tolist(),
            "total_order": res.total_order.tolist(),
            "first_order_ci95": {
                "low": _ci_to_list(bootstrap.first_order.confidence_interval.low),
                "high": _ci_to_list(bootstrap.first_order.confidence_interval.high),
            },
            "total_order_ci95": {
                "low": _ci_to_list(bootstrap.total_order.confidence_interval.low),
                "high": _ci_to_list(bootstrap.total_order.confidence_interval.high),
            },
        },
        "note_on_parameter_1": (
            "gas_availability_factor's prior (U(0.3061, 0.80)) spans this "
            "model's single largest lever (F-5, ~8.60 bn) and has no "
            "sourced central value (gas_avail_central is [PENDING]). If "
            "this parameter dominates first_order/total_order below, "
            "that is a property of the PRIOR's width, not a discovery "
            "about the model -- do not report it as the latter."
        ),
        "om_case_caveat": (
            "scenario['om_case'] on every draw is whatever placeholder "
            "string --om-case supplied; the actual O&M level used is "
            "om_blend_fraction in the JSONL, not scenario['om_case']."
        ),
        "conversion_748_verdict": (
            "748,000 USD/MW, cited in an earlier TASK.md as the "
            "conversion-capex omission's implied unit cost, does not "
            "exist anywhere in this codebase or its data files. The "
            "model's real value (conv_central) is 600.0 USD/kW. See "
            "this script's module docstring and docs/task_monte_carlo.md "
            "Part 2 for the full verdict."
        ),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {n_total} draws to {jsonl_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Failed draws: {n_failed}/{n_total} ({failure_fraction:.1%})")
    if failure_fraction > FAILURE_FRACTION_WARN_THRESHOLD:
        print(summary["feasibility"]["warning"])


if __name__ == "__main__":
    main()
