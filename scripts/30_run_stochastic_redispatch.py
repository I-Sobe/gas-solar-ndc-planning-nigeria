"""
30_run_stochastic_redispatch.py  —  Fixed-Plan Monte Carlo Re-Dispatch
==========================================================================

TASK A4 (docs/task_log_stochastic_runner.md). Wires `src/stochastic.py`
(implemented, interface-current, previously called by nothing) as a
runner. This docstring states, in the order TASK.md requires: what the
method is, what it does NOT claim, which builder and its scope, the
three fixed-plan gaps, the gas-weights prior's status, every ASSUMED
parameter this runner introduces, and why `run_stochastic_simulation`
(and, for one branch, `sample_uncertainties`) are not called directly.

1. WHAT THE METHOD IS
----------------------
Fixed-plan re-dispatch: a two-stage EVALUATION, not a two-stage
optimisation. Stage 1 (here-and-now): solve the LP once at the CLI's
own arm set to obtain a capacity plan. Stage 2 (recourse): freeze that
plan's capacity and re-solve under each of `--n-draws` independent
Monte Carlo realisations of demand growth, gas deliverability regime,
and (if active) carbon price, so only dispatch variables adjust. Every
draw uses the SAME objective, discounting, and cost stack as the base
solve, so the resulting cost distribution is directly comparable to
the deterministic NPV.

2. WHAT IT DOES NOT CLAIM
---------------------------
Quoted verbatim from `src/stochastic.py`'s own module docstring:

    "This measures the robustness of a plan optimized for the BASE
    case; it does NOT find the plan that is optimal under uncertainty
    (that would require stochastic programming with recourse)."

This runner reports how the base-case plan performs under uncertainty
(VaR, CVaR, infeasibility risk) — it never searches for a different,
uncertainty-optimal plan, and no output field should be read as one.

3. WHICH BUILDER, AND ITS SCOPE (confirmed against HEAD, not assumed)
------------------------------------------------------------------------
STALE AS OF A5 (docs/task_log_a5_port.md, Group D): this runner's own
two `build_model(` calls (Stage 1 base solve, Stage 2 per-draw re-solve)
never went through `src/stochastic.py` — verified by reading this file,
not by re-quoting the earlier assumption below — so porting THIS
runner's own calls to `build_model_sliced` required no edit to
`src/stochastic.py` at all and was made as a plain Group A port (import
and call name only). `src/stochastic.py` itself is UNCHANGED and its
own functions (`run_stochastic_simulation`, etc., none of which this
runner calls) still import and call only the annual builder — the
paragraph below describes THAT builder's scope, retained for the record
of what Stage 1/2 used to run on, and is no longer a description of
this runner:

The annual builder's scope condition,
confirmed by direct read of both builder files, not by re-quoting
Chapter 3 unchecked:

  - NO CONVERSION MECHANISM AT ALL. `optimize_model.py` has no
    `convert_mw` variable or OCGT-to-CCGT logic of any kind — it is
    not merely "free," it does not exist as a concept in this builder.
  - HYDRO IS A FIXED EXOGENOUS ANNUAL ENERGY SERIES
    (`hydro_energy_twh_by_year`), injected directly into the
    electricity balance every year with no decision variable, no
    seasonal shape, and no minimum-flow constraint. Its installed-MW
    rating feeds only the firm-capacity constraint.
  - STORAGE CANNOT VALUE ITS ENERGY-ARBITRAGE CONTRIBUTION, ONLY ITS
    FIRM-CAPACITY CREDIT. The annual builder does carry
    `storage_charge`/`storage_discharge` Vars (an "Annual
    Energy-Throughput Model" proxy bounded by
    `storage_deployable_hours_per_year`, default 700 h/yr), but with
    only one time step per year and `discharge <= RTE * charge`
    (RTE < 1), any nonzero throughput nets to an energy LOSS with no
    time-differentiated demand to serve — confirmed against the sliced
    builder's own docstring (`optimize_model_sliced.py:9-16`):
    "It also makes storage structurally incapable of value (there is
    no intra-annual time for it to shift energy across)." What storage
    DOES still get here: `capacity_credit["storage"]` enters the
    annual builder's `firm_capacity_constraint` identically to the
    sliced builder, so a fixed `storage_add` still counts toward
    meeting a tightened peak/reserve-margin requirement under a draw,
    even though building it for energy-shifting value alone would
    never be chosen by this builder.

  §3.1.8 and §3.3.5 currently say "cannot value storage" with no
  qualifier — flagged for Chapter 3 in the register (Part 3), not
  edited here.

4. THE FIXED PLAN IS PARTIAL — THREE GAPS, NOT ONE
------------------------------------------------------
`build_model`'s `fixed_plan` argument (`optimize_model.py:476-481`)
`.fix()`es exactly `solar_public_add[t]`, `solar_eaas_add[t]`,
`storage_add[t]`. Three things are NOT frozen by it:

  (a) `gas_add` — new gas capacity construction is a free
      `pyo.Var` (`optimize_model.py:472`), UNTOUCHED by `fixed_plan`.
      Controlled here by `--fix-gas`:
        `free`  (default framing): gas stays a RECOURSE decision — the
                model can build more gas capacity in response to a
                draw. This absorbs most shocks, makes infeasibility
                rare, and turns VaR largely into a measure of gas
                cost. A real, defensible evaluation, but a NARROWER
                question than "does the fixed plan survive."
        `fixed`: this runner ADDITIONALLY constrains `m.gas_add[t]` to
                the base solve's own per-year values — done directly
                on the model object THIS RUNNER builds, after
                `build_model()` returns it and before `solve_model()`
                is called. This is NOT a `fixed_plan` kwarg (confirmed:
                `fixed_plan` reads only the three keys above and would
                silently ignore a `"gas_add"` key if one were passed —
                verified by reading `optimize_model.py:476-481` line by
                line; no edit was made to accept one, per this task's
                explicit instruction to STOP rather than edit the
                builder). The two `--fix-gas` runs answer a genuinely
                different question and the gap between them is itself
                a reportable result: the value of gas flexibility as
                recourse.

                **`--fix-tolerance` controls HOW `gas_add[t]` is
                constrained, and is required whenever `--fix-gas
                fixed` is meaningfully exercised** (see docs/task_log_
                stochastic_runner.md, Items 1 and 4):
                  `0`: the ORIGINAL exact `m.gas_add[t].fix(v)` —
                       kept reproducible, but INFEASIBLE even at zero
                       perturbation, because it removes the LP's own
                       slack cushion at every year the base solve
                       drives `firm_capacity_constraint` to exact zero
                       slack (gas is the model's marginal,
                       cost-minimising capacity instrument, so the
                       solver builds exactly enough and no more — every
                       year `gas_add[t] > 0` in the base solution sits
                       on that knife-edge).
                  `>0`: bounds `gas_add[t]` to `[v*(1-tol), v*(1+tol)]`
                       instead of pinning it exactly — a NUMERICAL
                       ACCOMMODATION of the precision issue above, NOT
                       a modelling choice: the plan remains fixed in
                       every economically meaningful sense, since even
                       a generous `1e-3` relative tolerance on a
                       hundreds-to-thousands-of-MW `gas_add[t]` value
                       is well under 1 MW absolute — nine orders of
                       magnitude below the smallest capacity decision
                       in this model at the RECOMMENDED `1e-5`. A
                       tolerance sweep (`docs/task_log_stochastic_
                       runner.md`, Item 1) found `1e-5` relative the
                       smallest value that restores feasibility at the
                       zero-perturbation reproduction; NOT hardcoded
                       here — pass it explicitly. Years where the base
                       solve's own `gas_add[t] == 0` are UNAFFECTED by
                       `--fix-tolerance`: a relative band around zero
                       is `[0, 0]`, an exact fix regardless of the
                       tolerance chosen.
  (b) `eaas_subsidy` — deliberately excluded from `fixed_plan`, per
      the builder's own code comment, quoted verbatim
      (`optimize_model.py:477`): `# eaas_subsidy is excluded`. The
      subsidy transfer can still adjust per draw even though the EaaS
      solar CAPACITY it finances is frozen. No CLI flag controls this;
      it is a permanent gap in every run.
  (c) `convert_mw` — absent from this builder entirely (§3, above),
      not merely unfixed. Nothing to fix or leave free.

This is a "fixed CAPITAL plan for solar and storage; gas capacity
(unless `--fix-gas fixed`) and the EaaS subsidy remain adjustable"
evaluation — never described here as "the plan is frozen and only
dispatch adjusts."

5. `gas_probability_weights()` — A PRIOR, NOT A FORECAST
------------------------------------------------------------
Quoted verbatim from its own docstring (`src/scenarios.py:1032-1041`):
"Not a forecast -- a weighted sensitivity prior." Its own sourcing
note flags itself as UNVERIFIED against a primary: "Confirm each
citation against a primary source before submission (standing
verification rule)." `--gas-weights sourced` uses this prior as-is
({"baseline": 0.50, "downside": 0.25, "upside": 0.20,
"shock_recovery": 0.05}); `--gas-weights symmetric` uses equal weight
(0.25 each) over the SAME four labels, read live from
`gas_probability_weights().keys()` — never hardcoded — as the
robustness check the module's own docstring calls for.

6. ASSUMED PARAMETERS THIS RUNNER INTRODUCES — ALL REQUIRED, NO DEFAULT
----------------------------------------------------------------------------
Confirmed: zero references to `demand_sigma`, `carbon_mu`, or
`carbon_sigma` exist anywhere in this repository outside
`src/stochastic.py` itself before this runner. Matching the
`dscr_case`/`storage_cc_case` required-key discipline, all three are
required CLI arguments with no default, so the choice is recorded in
every output rather than silently inherited from the module's own
0.01 default.

  --demand-sigma : CALIBRATED to this model's own demand-arm span, NOT
      independently sourced. At sigma=0.01 on a 3.9% central arm
      (`organic_central`), +/-2 sigma spans roughly 1.9%-5.9%, which
      brackets `constrained_continuation` (3.14%) and approaches
      `organic_high` (6.0%) — see `demand_growth_scenarios()`. This
      calibration is reproducible from the model's own named arms and
      is declared as such, not presented as an external estimate.

  --carbon-mu, --carbon-sigma : ASSUMED, no source. This runner's base
      scenario is built with `carbon_case="no_policy"` (matching
      `28_run_capital_frontier.py`'s own convention for a script not
      studying carbon policy) and no CLI flag here changes it, so
      `carbon_active=False` at this runner's central configuration —
      confirmed by reading `carbon_policy_scenarios()["no_policy"]`
      (`src/scenarios.py:842-845`). `sample_uncertainties()` only
      samples/applies `carbon_price` "only if carbon_active"
      (`src/stochastic.py:104-107`), so under THIS runner's fixed
      "no_policy" base configuration, `--carbon-mu`/`--carbon-sigma`
      are INERT — required on the CLI regardless (per this task's own
      required-key discipline for every ASSUMED parameter), but they
      have no effect on any result this runner produces unless a
      future variant exposes `carbon_case` itself as a CLI choice.
      `data/stochastic/carbon_price_reference.csv` (a World Bank
      Carbon Pricing Dashboard extract, 1,249 rows, NOT Nigeria-
      specific) exists as an UNWIRED candidate calibration source —
      it has a generator (`data/make_data.py:process_carbon_prices`)
      but is read by nothing in `src/`/`scripts/`, and is NOT read
      here either: calibrating against it would mean inventing a
      mapping, forbidden by this task's brief.

7. WHY `run_stochastic_simulation` (AND, PARTIALLY, `sample_uncertainties`)
   ARE NOT CALLED
------------------------------------------------------------------------------
`run_stochastic_simulation` returns `list[tuple(gas_scenario, cost)]`
for FEASIBLE draws only, discarding every infeasible draw's
realisation entirely and most of each feasible draw's realisation
(`demand_growth`/`carbon_price` never survive to its return value).
This cannot satisfy the requirement that `draws.jsonl` record every
draw's full realisation, feasibility, and cost. This runner instead
calls `sample_uncertainties`, `build_model`, `solve_model`, and
`compute_risk_metrics` directly — the same four functions
`run_stochastic_simulation` itself calls, in the same order,
reproducing its exact logic — while retaining the full per-draw
record. No line of `src/stochastic.py` changes.

One further gap, found while implementing `--gas-weights symmetric`:
`sample_uncertainties()` has NO parameter to override the gas-regime
weights — it always reads `gas_probability_weights()` internally
(`src/stochastic.py:86`). So for `--gas-weights sourced`, this runner
calls the REAL `sample_uncertainties()` unmodified. For `--gas-weights
symmetric`, calling it is not possible without editing the module
(forbidden), so this runner instead uses a local helper,
`_draw_one_realisation()` below, that reproduces
`sample_uncertainties()`'s three documented distributions EXACTLY
(truncated-at-zero normal on demand growth, discrete draw over gas
regime labels, lognormal carbon price only if active) with the
labels read live from `gas_probability_weights().keys()` and the
weights parameterised by `--gas-weights`. This is flagged here, not
silently done, because it is a second instance of the same
interface-insufficiency class Ruling 1 already approved a workaround
for. **This parallel path exists ONLY because `sample_uncertainties()`
takes no weights override** — it is not a preference for reimplementing
module logic in a runner. Because a parallel path can silently drift
from the source it mirrors, `--gas-weights sourced` runs (Self-check 2,
below) verify on EVERY invocation that this runner's own reading of
the gas-regime labels/weights reproduces `gas_probability_weights()`
exactly, rather than trusting that agreement once at write-time.

8. `peak_mw` IS RECOMPUTED PER DRAW — THE `free`-MODE CONVENTION CHANGED
----------------------------------------------------------------------------
Every draw's `peak_mw` (the firm-capacity requirement, constraint (3.21))
is now recomputed from THAT draw's own `demand_growth`, via
`_recompute_peak_mw()` below — reusing `load_scenario()`'s own
derivation (`project_baseline_demand()` + the scenario's own stored,
demand-growth-INDEPENDENT `par_block`) line for line, confirmed
byte-identical to `load_scenario()`'s own `peak_mw` at zero
perturbation. Previously, `peak_mw` stayed at whatever the CLI's
original `demand_case` produced, for every draw — so a demand-growth
draw raised ENERGY demand (re-derived fresh inside `build_model()`)
but left the CAPACITY requirement untouched.

**Consequence, stated plainly**: with `peak_mw` now moving with the
draw, **some draws under `--fix-gas fixed` will become infeasible when
a draw's peak exceeds what the frozen capacity (plus its credits) can
cover — and THAT infeasibility is the reliability finding this runner
exists to produce, not a defect.** It is precisely what Sobol cannot
observe, since Sobol re-optimises the plan at every draw and so always
finds a new feasible capacity mix if one exists; only a genuinely
FIXED plan can be found wanting against a peak it was not sized for.
This is a DIFFERENT infeasibility mechanism from the one diagnosed in
`docs/task_log_stochastic_runner.md` (numerical: fixing `gas_add`
exactly at a value where `firm_capacity_constraint` is already
essentially zero-slack in the base solve reproduces infeasible under
finite LP precision, even at ZERO perturbation) — that mechanism is
unresolved and orthogonal to this one, which requires an ACTUAL
capacity shortfall from a moved peak. Both can now occur in the same
run and must not be conflated when reading `draws.jsonl`.

**The existing `--fix-gas free` pilot result (`draws__76965a0b.jsonl`,
`summary__76965a0b.json`, 100/100 feasible) is SUPERSEDED by this
change and must be re-run** — it was produced against a static peak
that never tested capacity adequacy at all; its regime-breakdown
figures (quoted in prior analysis: downside mean 68.24 bn, baseline
49.29 bn, upside 40.60 bn) describe ENERGY-adequacy-only outcomes and
should not be read as this runner's finished answer to "does the plan
survive a realisation." Not re-run here — Zeus re-runs, per role
separation.

`selfgen` remains UNBOUNDED, deliberately — it is the backstop by
design, and bounding it would make infeasibility rather than cost the
signal of an inadequate plan; cost is the more informative signal for
the energy dimension. Only the CAPACITY dimension is meant to be able
to bind to infeasibility, via (3.21), which is exactly what recomputing
`peak_mw` per draw restores the ability to observe.

TWO SELF-CHECKS, RUN EVERY TIME, NOT JUST AT WRITE-TIME
------------------------------------------------------------
1. **`--fix-gas fixed` fixation/banding is verified, not assumed.**
   After applying `m.gas_add[t].fix(v)` (`--fix-tolerance 0`) or
   `setlb`/`setub` to `[v*(1-tol), v*(1+tol)]` (`--fix-tolerance > 0`)
   for every `t`, this runner reads the resulting `.fixed` state or
   `.lb`/`.ub` straight back off the model and raises `RuntimeError`
   immediately if either does not match what was requested — a silent
   no-op here would make `free` and `fixed` numerically
   indistinguishable while claiming to differ, or make a chosen
   `--fix-tolerance` silently not apply. The per-year fixed values are
   recorded in `summary.json` (`fixed_plan.gas_add_fixed`) regardless;
   under a nonzero tolerance, `summary.json`'s
   `gas_add_band_diagnostics` additionally reports, across every
   feasible draw's nonzero-band years, how many landed AT a band edge
   versus in the INTERIOR — many-at-edge means the band is doing real
   numerical work and its width matters; none-at-edge means it is pure
   slack (Item 4, `docs/task_log_stochastic_runner.md`).
2. **`--gas-weights sourced` cross-checks the local helper against the
   live source, every run.** Even though `sourced` mode calls the REAL
   `sample_uncertainties()` (not the local helper) for drawing, this
   runner also independently resolves the labels/weights it WOULD hand
   the local helper under `sourced` semantics and asserts that pairing
   reproduces `gas_probability_weights()`'s own dict exactly (same
   labels, same values, same order), raising `RuntimeError` on any
   divergence. This exists so the local helper used by `symmetric`
   mode — whose label set must stay aligned with the live source or
   "symmetric" silently becomes "equal weight over the WRONG labels" —
   is never trusted to have matched at write-time alone.

OUTPUTS — results/stochastic_redispatch/
-------------------------------------------
  draws.jsonl__<hash8>.jsonl  -- one line per draw: realisation
      (demand_growth, gas_scenario, carbon_price), feasibility, cost
      where feasible, and (for a solver-failed draw) `error_message`.
      INFEASIBLE DRAWS ARE RECORDED WITH THEIR REALISATION, never
      dropped: `solve_model()` raises `RuntimeError` DIRECTLY on any
      non-optimal/non-feasible termination (`optimize_model.py:
      1884-1890`) rather than returning a status dict, so the per-draw
      `solve_model()` call is wrapped in `try`/`except RuntimeError` --
      unguarded, the FIRST infeasible draw would otherwise abort the
      whole run. The raised message embeds the literal termination
      condition (`f"Solver failed with termination condition: {tc}"`),
      parsed here to label infeasible vs. unbounded vs. other solver
      failures distinctly; if that exact message format ever changes,
      the label falls back to `"unknown"` but the full raised message
      is ALWAYS recorded verbatim in `error_message` regardless, so
      infeasibility and other solver failures are separable only to
      the extent this coupled parsing continues to hold, never lost
      entirely. Only the BASE solve is left unguarded — a base-solve
      failure is a genuine error (there is no fixed plan to
      re-dispatch), not a per-draw outcome to record.
  summary__<hash8>.json -- full arm set (including this runner's own
      eight), the fixed plan (all three variables, plus gas_add if
      `--fix-gas fixed`), n_draws, seed, demand_sigma, carbon_mu,
      carbon_sigma, fix_gas, fix_tolerance, gas_weights;
      `gas_add_band_diagnostics` (populated only under `--fix-gas
      fixed` with `--fix-tolerance > 0`: n_at_edge vs n_interior across
      every feasible draw's nonzero-band years -- Item 4's "does the
      band do real work" verification, recorded every run rather than
      as a one-off check); n_feasible alongside every risk figure, so a
      metric is never read without its sample size; mean/sd/p5/p50/p95
      of cost; VaR and CVaR at 95%; infeasible count and fraction; and
      the base deterministic cost for comparison.
      **ALL-INFEASIBLE RUNS STILL WRITE THIS FILE.** `compute_risk_
      metrics()` raises `ValueError` on an empty list
      (`src/stochastic.py:213-214`, not edited here) -- if
      `n_feasible == 0`, this runner skips that call entirely and
      writes every risk field (mean/sd/p5/p50/p95/VaR/CVaR) EXPLICITLY
      `null`, not omitted, so a reader can tell "no draw was feasible"
      from "this field was never populated." A 100%-infeasible run
      under `--fix-gas fixed` is itself a reliability-risk result, not
      a crash, and exits 0.
  INDEX.md -- hash8 -> full arm tag mapping (src/output_naming.py,
      same scheme as 01b_/26_).

PRE-REQUISITES
----------------
    none beyond the standard arm choices and this script's own seven
    required flags -- it does not read another runner's output.
"""

import argparse
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io import load_econ, load_solar_capex_by_year
from src.scenarios import load_scenario, gas_probability_weights, MODEL_END_YEAR
from src.optimize_model import solve_model
from src.optimize_model_sliced import build_model_sliced
from src.stochastic import sample_uncertainties, compute_risk_metrics
from src.cli_arms import add_standard_arms, standard_arms_kwargs, complete_arm_tag
from src.output_naming import hash8_of, append_index
from src.demand import project_baseline_demand

# Matches 28_run_capital_frontier.py's own convention exactly: VOLL and
# genset_case are a fixed methodological choice for this family of
# runners, not a swept arm -- neither is part of add_standard_arms()'s
# surface (src/cli_arms.py), so hardcoding them here mirrors the
# closest sibling, not a violation of "no hardcoded arm values" (which
# refers to the scenario arms this runner's own interface exposes).
CANONICAL_VOLL = "voll_mid"
CANONICAL_GENSET = "genset_central"

RESULTS_DIR = ROOT / "results" / "stochastic_redispatch"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="30_run_stochastic_redispatch: fixed-plan Monte Carlo "
                     "re-dispatch (VaR/CVaR, infeasibility-under-a-fixed-plan)"
    )
    add_standard_arms(parser)

    parser.add_argument(
        "--n-draws", type=int, required=True,
        help="Number of Monte Carlo re-dispatch draws. No default: the "
             "precision of every risk metric below depends on it, and an "
             "unstated draw count would leave that precision invisible.",
    )
    parser.add_argument(
        "--seed", type=int, required=True,
        help="RNG seed for sample_uncertainties()/local draw reproducibility. "
             "No default: an unstated seed makes the run unreproducible.",
    )
    parser.add_argument(
        "--demand-sigma", type=float, required=True,
        help="Std dev of the demand-growth perturbation (absolute, on the "
             "growth fraction). No default -- see this file's own docstring "
             "Section 6: 0.01 is a reproducible CALIBRATION against this "
             "model's own demand arms, not an independently sourced figure, "
             "and must be stated explicitly rather than silently inherited "
             "from src/stochastic.py's own 0.01 default.",
    )
    parser.add_argument(
        "--carbon-mu", type=float, required=True,
        help="Log-space mean of the lognormal carbon-price draw. ASSUMED, "
             "no source (docstring Section 6). INERT at this runner's "
             "hardcoded carbon_case='no_policy' base configuration -- "
             "required on the CLI regardless, per the required-key "
             "discipline for every ASSUMED parameter this runner introduces.",
    )
    parser.add_argument(
        "--carbon-sigma", type=float, required=True,
        help="Log-space sigma of the lognormal carbon-price draw. ASSUMED, "
             "no source. Same inertness caveat as --carbon-mu.",
    )
    parser.add_argument(
        "--fix-gas", choices=["free", "fixed"], required=True,
        help="Whether new gas capacity (gas_add) is left as a RECOURSE "
             "decision under re-dispatch ('free', matching stochastic.py's "
             "own fixed_plan exactly) or additionally frozen at the base "
             "solve's per-year values ('fixed', via a direct Pyomo .fix() "
             "call this runner makes on the built model -- NOT a "
             "fixed_plan kwarg; build_model's fixed_plan does not accept a "
             "gas_add key). No default: this is a declared, reportable "
             "modelling choice (docstring Section 4), not a detail to bury "
             "in a default.",
    )
    parser.add_argument(
        "--fix-tolerance", type=float, required=True,
        help="Relative half-width of the band [v*(1-tol), v*(1+tol)] used "
             "to fix gas_add[t] under --fix-gas fixed, in place of an exact "
             ".fix(v). Required, no default -- exact fixation (docs/task_"
             "log_stochastic_runner.md Item 1) is infeasible even at zero "
             "perturbation, because it removes the LP's own slack cushion "
             "at years where the base solve drives firm_capacity_constraint "
             "to exact zero slack; the tolerance sweep behind this flag "
             "found 1e-5 relative as the smallest value that restores "
             "feasibility (RECOMMENDED, not hardcoded -- pass it "
             "explicitly). This is a NUMERICAL accommodation, not a "
             "modelling choice: 1e-5 MW is nine orders of magnitude below "
             "the smallest capacity decision in this model. Pass 0 for the "
             "original exact .fix() behaviour, kept reproducible. Inert "
             "under --fix-gas free.",
    )
    parser.add_argument(
        "--gas-weights", choices=["sourced", "symmetric"], required=True,
        help="'sourced' uses gas_probability_weights() as given (a "
             "self-described, unverified sensitivity prior). 'symmetric' "
             "uses equal weight over the same labels, as the module's own "
             "docstring calls for as a robustness check. No default: the "
             "prior is explicitly unverified, so which one was used must "
             "be recorded, not assumed.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Override the default results/stochastic_redispatch/ output "
             "directory. Optional.",
    )
    return parser.parse_args()


def _draw_one_realisation(base_scenario, carbon_mu, carbon_sigma, demand_sigma,
                           labels, weights):
    """
    Reproduces sample_uncertainties()'s three documented distributions
    EXACTLY (src/stochastic.py:52-111), parameterised by caller-supplied
    gas-regime labels/weights since that module has no such override --
    see this file's own docstring, Section 7. Used only for
    --gas-weights symmetric; the sourced path calls the real
    sample_uncertainties() unmodified (see main()).
    """
    scenario = copy.deepcopy(base_scenario)
    scenario["demand_growth"] = max(
        0.0,
        np.random.normal(base_scenario["demand_growth"], demand_sigma),
    )
    scenario["gas_scenario"] = np.random.choice(labels, p=weights)
    if scenario.get("carbon_active", False):
        scenario["carbon_price"] = np.random.lognormal(
            mean=carbon_mu, sigma=carbon_sigma,
        )
    return scenario


def _resolve_gas_weights(mode, gas_probs):
    """
    Returns (labels, weights) for `_draw_one_realisation()`'s `mode`.
    Exists ONLY because `sample_uncertainties()` (src/stochastic.py) has
    no parameter to override the gas-regime weights -- this is this
    runner's own parallel resolution path, used directly for
    `symmetric` and, via Self-check 2 in main(), cross-checked against
    `gas_probability_weights()` on every run for `sourced` -- see this
    file's own docstring, "TWO SELF-CHECKS".

    mode='sourced'   : labels/weights exactly as gas_probs gives them.
    mode='symmetric' : same labels, equal weight (1/len(labels)) each --
        the module's own docstring's own robustness check.
    """
    labels = list(gas_probs.keys())
    if mode == "sourced":
        weights = list(gas_probs.values())
    elif mode == "symmetric":
        weights = [1.0 / len(labels)] * len(labels)
    else:
        raise ValueError(f"Unknown gas-weights mode: {mode!r}")
    return labels, weights


def _recompute_peak_mw(scenario):
    """
    Recompute scenario["peak_mw"] from THIS scenario's own demand_growth,
    reusing load_scenario()'s own derivation line for line (scenarios.py:
    3036-3048) -- not a parallel formula. `par_block` (the block-average
    peaking ratio) depends only on peak_multiple_case's slice weights,
    NOT on demand_growth -- confirmed by reading its derivation
    (scenarios.py:3015-3034, `_normalised_demand_weights`/`slice_hours`
    applied to `_season_demand_factor`/`_period_demand_factor` only) and
    empirically (recomputing it from a scenario's own stored value
    reproduces load_scenario()'s peak_mw byte-for-byte at zero
    perturbation) -- so it is read unchanged from the scenario dict,
    never recomputed, and never perturbed by any Monte Carlo draw.

    Required per TASK.md (A4 peak_mw item): a draw that raises demand
    must also raise the firm-capacity requirement it is tested against,
    or demand draws test energy adequacy only (unconditionally met by
    the unbounded selfgen backstop) and never capacity adequacy at all.
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


def main():
    args = parse_args()
    standard_kwargs = standard_arms_kwargs(args)

    if args.n_draws <= 0:
        raise ValueError("--n-draws must be a positive integer.")
    if args.fix_tolerance < 0:
        raise ValueError("--fix-tolerance must be >= 0 (0 = exact .fix()).")

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full arm tag: every standard arm plus this runner's own eight,
    # same scheme as 01b_/26_ (src/output_naming.py). Deterministic --
    # re-running the identical configuration overwrites its own output.
    standard_tag = "__".join(
        f"{k}={v}" for k, v in sorted(standard_kwargs.items())
    )
    own_tag = (
        f"n_draws={args.n_draws}__seed={args.seed}"
        f"__demand_sigma={args.demand_sigma}"
        f"__carbon_mu={args.carbon_mu}__carbon_sigma={args.carbon_sigma}"
        f"__fix_gas={args.fix_gas}__fix_tolerance={args.fix_tolerance}"
        f"__gas_weights={args.gas_weights}"
    )
    # TASK 53 2.5 (G7-4): the non-flag arms (capital_window_case, storage_capex_case) join the key.
    _extra = complete_arm_tag("", args, in_tag=set(standard_kwargs))
    standard_tag = standard_tag + _extra
    arm_tag = f"{standard_tag}__{own_tag}"
    hash8 = hash8_of(arm_tag)
    append_index(out_dir, hash8, arm_tag)

    print(f"\nStochastic Fixed-Plan Re-Dispatch")
    print(f"  n_draws={args.n_draws}  seed={args.seed}  "
          f"fix_gas={args.fix_gas}  fix_tolerance={args.fix_tolerance}  "
          f"gas_weights={args.gas_weights}")
    print(f"  demand_sigma={args.demand_sigma}  "
          f"carbon_mu={args.carbon_mu}  carbon_sigma={args.carbon_sigma}")
    print(f"  arm_tag hash8={hash8}")

    econ = load_econ(CANONICAL_VOLL, genset_case=CANONICAL_GENSET)

    # Matches 28_run_capital_frontier.py's own literal choices exactly
    # for every scenario dimension this runner does not itself sweep --
    # "follow 28_'s conventions exactly" (TASK.md). carbon_case=
    # "no_policy" is why --carbon-mu/--carbon-sigma are inert here
    # (docstring Section 6).
    base_scenario = load_scenario(
        demand_level_case="served",
        demand_case="organic_central",
        capital_case="unconstrained",  # TASK 81 Batch 4: the central anchor's own capital_case (01b:294); was "moderate" (L92)
        solar_build_case="deployment_unconstrained",
        land_case="loose",
        carbon_case="no_policy",
        start_year=2025,
        end_year=MODEL_END_YEAR,
        **standard_kwargs,
    )

    years = base_scenario["years"]
    T = range(len(years))
    solar_capex_tv = load_solar_capex_by_year(
        scenario_name="solar_low",
        start_year=int(years[0]),
        end_year=int(years[-1]),
    )

    # ---- Stage 1: base (here-and-now) solve -------------------------
    print(f"\n  Base solve...")
    m_base = build_model_sliced(
        scenario=base_scenario, econ=econ,
        emissions_cap_by_year=None,
        solar_capex_by_year=solar_capex_tv,
    )
    base_status = solve_model(m_base, scenario=base_scenario)
    if not base_status["optimal"]:
        raise RuntimeError(
            "Base scenario solve was not optimal "
            f"(status={base_status['status']}); cannot extract a fixed plan."
        )
    base_cost = float(pyo.value(m_base.system_cost_npv))

    fixed_plan = {
        "solar_public_add": {t: pyo.value(m_base.solar_public_add[t]) for t in T},
        "solar_eaas_add":   {t: pyo.value(m_base.solar_eaas_add[t]) for t in T},
        "storage_add":      {t: pyo.value(m_base.storage_add[t]) for t in T},
    }
    base_gas_add = {t: pyo.value(m_base.gas_add[t]) for t in T}

    print(f"    base cost (NPV, USD): {base_cost:,.2f}")
    print(f"    fixed plan: solar_public_add sum={sum(fixed_plan['solar_public_add'].values()):.2f} MW, "
          f"solar_eaas_add sum={sum(fixed_plan['solar_eaas_add'].values()):.2f} MW, "
          f"storage_add sum={sum(fixed_plan['storage_add'].values()):.2f} MW")
    if args.fix_gas == "fixed":
        print(f"    gas_add (to be additionally fixed): sum={sum(base_gas_add.values()):.2f} MW")

    # ---- Stage 2: sample realisations --------------------------------
    gas_probs = gas_probability_weights()

    # Self-check 2 (docstring "TWO SELF-CHECKS"): whenever sourced weights
    # are in play, verify THIS runner's own label/weight resolution
    # (_resolve_gas_weights, the same function `symmetric` mode uses)
    # reproduces gas_probability_weights() exactly, every run -- not
    # trusted from a one-time write-time check. sample_uncertainties()
    # itself has no weights-override parameter (docstring Section 7), so
    # this runner necessarily maintains a parallel reading of the same
    # source for `symmetric` mode; this assertion is what keeps that
    # parallel reading from silently drifting.
    check_labels, check_weights = _resolve_gas_weights("sourced", gas_probs)
    if (check_labels != list(gas_probs.keys())
            or check_weights != list(gas_probs.values())):
        raise RuntimeError(
            "Self-check 2 FAILED: _resolve_gas_weights('sourced', ...) "
            f"returned labels={check_labels}, weights={check_weights}, "
            f"which does not exactly match gas_probability_weights() = "
            f"{gas_probs}. This runner's local gas-weights resolution "
            "path has diverged from src/scenarios.py's live source -- "
            "refusing to proceed, since --gas-weights symmetric's label "
            "set would then be untrustworthy too."
        )
    print(f"    Self-check 2 passed: local gas-weights resolution matches "
          f"gas_probability_weights() exactly ({check_labels}).")

    if args.gas_weights == "sourced":
        # Real sample_uncertainties(), unmodified (Ruling 1 / docstring
        # Section 7) -- this path has no weights-override need; Self-check
        # 2 above already confirmed this runner's own view of the prior
        # agrees with what sample_uncertainties() reads internally.
        if args.seed is not None:
            np.random.seed(args.seed)
        realisations = sample_uncertainties(
            args.n_draws, base_scenario, args.carbon_mu, args.carbon_sigma,
            demand_sigma=args.demand_sigma, seed=args.seed,
        )
    else:
        # symmetric: sample_uncertainties() cannot express this (no
        # weights parameter) -- local reproduction, docstring Section 7.
        labels, weights = _resolve_gas_weights("symmetric", gas_probs)
        np.random.seed(args.seed)
        realisations = [
            _draw_one_realisation(
                base_scenario, args.carbon_mu, args.carbon_sigma,
                args.demand_sigma, labels, weights,
            )
            for _ in range(args.n_draws)
        ]

    # ---- peak_mw MUST move with the draw's own demand path --------------
    # TASK.md (A4 peak_mw item): sample_uncertainties()/_draw_one_realisation()
    # only overwrite scenario["demand_growth"] -- scenario["peak_mw"] is left
    # at whatever load_scenario() computed from the CLI's ORIGINAL demand_case,
    # a STALE value (confirmed by reading scenarios.py: peak_mw is computed
    # once, inline, at load_scenario() time, and neither src/stochastic.py nor
    # this runner previously touched it again). Left uncorrected, a demand
    # draw raises ENERGY demand (build_model() re-derives that fresh from
    # scenario["demand_growth"], optimize_model.py:577-582) but leaves the
    # FIRM-CAPACITY requirement (3.21) at its base value -- since selfgen is
    # unbounded and priced, energy adequacy is then ALWAYS achievable at a
    # cost, so demand draws would test energy adequacy only, never capacity
    # adequacy, which hollows out the method (the plan's CAPACITY is
    # correctly fixed at the investment stage; the PEAK it must meet is a
    # property of the drawn world, not of the plan, and the two must not be
    # conflated). Corrected here, for every draw regardless of which sampling
    # path produced it, by _recompute_peak_mw() -- reusing load_scenario()'s
    # own derivation exactly (confirmed byte-identical at zero perturbation),
    # never a parallel formula.
    for draw_scenario in realisations:
        draw_scenario["peak_mw"] = _recompute_peak_mw(draw_scenario)

    # ---- TASK 06, Part 1: serialise the realised draws, shared with 17_ --
    # docs/task_log_06_vss_bound.md. Written HERE, before the solve loop,
    # from the realised sample, so the file exists even if the loop below
    # is interrupted -- a crashed re-dispatch run still leaves a usable
    # draws file for 17_ to consume. TASK 01 item 17 established that 30_
    # and 17_ sample differently (30_ truncates the demand normal at zero;
    # 17_ rejection-samples; 30_ carries a carbon branch and selectable gas
    # weights, 17_ neither) -- identical seeds do NOT produce identical
    # draws, so sharing the REALISATION is the fix, not matching the
    # samplers. Columns are exactly the three quantities sample_
    # uncertainties() (or this runner's own symmetric-mode equivalent,
    # _draw_one_realisation -- both produce the identical three fields)
    # actually samples: demand_growth, gas_scenario, carbon_price.
    # peak_mw is NOT included -- it is DERIVED from demand_growth
    # (_recompute_peak_mw, above), not sampled, and is fully reproducible
    # by any consumer from demand_growth alone.
    #
    # Filename is hashed on a DRAWS-ONLY tag -- n_draws/seed/demand_sigma/
    # carbon_mu/carbon_sigma/gas_weights plus the standard arm set, but
    # excluding --fix-gas/--fix-tolerance, which affect only the Stage-2
    # RE-DISPATCH of an already-realised sample, not what was sampled.
    # Two runs that differ only by --fix-gas therefore share one draws
    # file, which is the whole point: 31_'s guard that "both legs
    # reference the same draws file, by SHA-256" is trivially satisfiable
    # by construction when the same draws-tag was used for both.
    draws_dir = ROOT / "results" / "stochastic"
    draws_dir.mkdir(parents=True, exist_ok=True)
    draws_tag = (
        f"{standard_tag}__n_draws={args.n_draws}__seed={args.seed}"
        f"__demand_sigma={args.demand_sigma}__carbon_mu={args.carbon_mu}"
        f"__carbon_sigma={args.carbon_sigma}__gas_weights={args.gas_weights}"
    )
    draws_hash8 = hash8_of(draws_tag)
    draws_csv_path = draws_dir / f"draws__{draws_hash8}.csv"
    with open(draws_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["draw_id", "demand_growth", "gas_scenario", "carbon_price"])
        for i, draw_scenario in enumerate(realisations):
            writer.writerow([
                i,
                draw_scenario["demand_growth"],
                draw_scenario["gas_scenario"],
                draw_scenario.get("carbon_price") if draw_scenario.get("carbon_active", False) else "",
            ])

    # SHA-256 of the file itself -- computed AFTER writing it (a file
    # cannot state its own hash inside itself; same resolution TASK 00
    # used for the pre-phase-b tag). Written to a sidecar, not a header
    # block, for exactly that reason.
    draws_sha256 = hashlib.sha256(draws_csv_path.read_bytes()).hexdigest()
    draws_meta_path = draws_dir / f"draws__{draws_hash8}_meta.json"
    with open(draws_meta_path, "w") as f:
        json.dump({
            "n_draws": args.n_draws,
            "seed": args.seed,
            "demand_sigma": args.demand_sigma,
            "carbon_mu": args.carbon_mu,
            "carbon_sigma": args.carbon_sigma,
            "gas_weights": args.gas_weights,
            "draws_tag": draws_tag,
            "draws_hash8": draws_hash8,
            "csv_path": str(draws_csv_path.relative_to(ROOT)),
            "csv_sha256": draws_sha256,
        }, f, indent=2)
    append_index(draws_dir, draws_hash8, draws_tag)
    print(f"\n  Saved: {draws_csv_path} ({args.n_draws} draws, sha256={draws_sha256[:12]}...)")
    print(f"  Saved: {draws_meta_path}")

    # ---- Stage 2: re-dispatch each draw, recording EVERY one ---------
    print(f"\n  Re-dispatching {args.n_draws} draws...")
    draw_records = []
    feasible_costs = []
    n_infeasible = 0
    # Band-edge diagnostic (Item 4, docs/task_log_stochastic_runner.md):
    # accumulated across every draw and every nonzero-band year, so
    # summary.json can report whether the band is doing real work (many
    # at an edge) or is pure numerical slack (none at an edge). Zero-
    # valued base gas_add years give a zero-width band (v=0 => the
    # relative band [0,0] degenerates to an exact fix regardless of
    # --fix-tolerance) and are excluded from this count as uninformative
    # -- tallied separately instead.
    n_gas_add_at_edge = 0
    n_gas_add_interior = 0
    n_gas_add_zero_width_years = sum(1 for t in T if base_gas_add[t] == 0.0)
    for i, draw_scenario in enumerate(realisations):
        m = build_model_sliced(
            scenario=draw_scenario, econ=econ,
            emissions_cap_by_year=None,
            solar_capex_by_year=solar_capex_tv,
            fixed_plan=fixed_plan,
        )
        gas_add_bounds = {}  # t -> (lb, ub), for the band-edge check after solving
        if args.fix_gas == "fixed":
            # docstring Section 4(a) / Item 4: build_model's fixed_plan does
            # not accept a gas_add key, and the builder is not edited to add
            # one -- fixation is applied directly on the built model, NOT
            # via fixed_plan, exactly as before. --fix-tolerance selects
            # exact .fix() (0, the original, reproducible behaviour -- Item
            # 1's infeasibility) or a banded [v*(1-tol), v*(1+tol)] bound
            # (>0, the numerical accommodation Item 4 adds): exact fixation
            # is infeasible even at zero perturbation because it removes
            # the LP's own slack cushion at years where the base solve
            # drives firm_capacity_constraint to exact zero slack
            # (docs/task_log_stochastic_runner.md, Item 1) -- the band is a
            # NUMERICAL accommodation of that, not a modelling choice: at
            # the recommended 1e-5 relative tolerance, the band's absolute
            # width is nine orders of magnitude below the smallest capacity
            # decision in this model, so the plan remains fixed in every
            # economically meaningful sense.
            for t in T:
                v = base_gas_add[t]
                if args.fix_tolerance == 0:
                    m.gas_add[t].fix(v)
                    gas_add_bounds[t] = (v, v)
                else:
                    eps = args.fix_tolerance * abs(v)
                    lb = max(0.0, v - eps)
                    ub = v + eps
                    m.gas_add[t].setlb(lb)
                    m.gas_add[t].setub(ub)
                    gas_add_bounds[t] = (lb, ub)

                # Self-check 1 (docstring "TWO SELF-CHECKS"): verify the
                # fix/band actually took, every draw, every year -- a
                # silent no-op here would make 'free' and 'fixed'
                # numerically indistinguishable while claiming to differ.
                lb_now, ub_now = gas_add_bounds[t]
                if args.fix_tolerance == 0:
                    ok = m.gas_add[t].fixed
                else:
                    ok = (
                        m.gas_add[t].lb is not None and m.gas_add[t].ub is not None
                        and abs(float(m.gas_add[t].lb) - lb_now) < 1e-12
                        and abs(float(m.gas_add[t].ub) - ub_now) < 1e-12
                    )
                if not ok:
                    raise RuntimeError(
                        f"Self-check 1 FAILED on draw {i}, year index {t}: "
                        f"requested gas_add[{t}] bound/.fix() at "
                        f"(lb={lb_now}, ub={ub_now}, tol={args.fix_tolerance}) "
                        "did not take on the model. Refusing to proceed -- "
                        "--fix-gas fixed would silently be no different "
                        "from --fix-gas free."
                    )

        realisation_fields = {
            "draw": i,
            "demand_growth": float(draw_scenario["demand_growth"]),
            "gas_scenario": draw_scenario["gas_scenario"],
            "carbon_price": (
                float(draw_scenario["carbon_price"])
                if draw_scenario.get("carbon_active", False) else None
            ),
        }

        # solve_model() raises RuntimeError DIRECTLY on any termination
        # condition other than optimal/feasible (optimize_model.py:
        # 1884-1890) instead of returning a status dict -- confirmed by
        # reading that guard, not assumed. Unguarded, the FIRST
        # infeasible draw would abort the whole Monte Carlo loop, making
        # n_infeasible/infeasible_fraction unreachable. Caught here so
        # an infeasible (or otherwise solver-failed) draw is RECORDED,
        # not fatal, matching src/stochastic.py's own stated intent
        # ("infeasibility under a fixed plan is itself a reliability-risk
        # finding") which its own run_stochastic_simulation() cannot
        # actually reach for the same reason. Only the BASE solve above
        # is left unguarded: a base-solve failure is a genuine error
        # (there is no fixed plan to re-dispatch), not a per-draw
        # outcome to record.
        try:
            status = solve_model(m, scenario=draw_scenario)
        except RuntimeError as exc:
            # The raised message embeds the literal termination
            # condition string (f"Solver failed with termination
            # condition: {tc}", optimize_model.py:1888-1890) -- parsed
            # here to distinguish infeasible/unbounded/other. This
            # parsing is coupled to that exact message format: if it
            # ever changes, tc_label falls back to "unknown" but the
            # raw message is ALWAYS recorded verbatim regardless, so no
            # information is lost either way.
            error_message = str(exc)
            _prefix = "Solver failed with termination condition: "
            tc_label = (
                error_message.split(_prefix, 1)[1].strip()
                if _prefix in error_message else "unknown"
            )
            draw_records.append({
                **realisation_fields,
                "feasible": False,
                "status": tc_label,
                "cost_usd": None,
                "error_message": error_message,
            })
            n_infeasible += 1
            continue

        # Unchanged from before this fix: solve_model() only returns (as
        # opposed to raising) for tc in (optimal, feasible), and the
        # ORIGINAL branching here treated the rarer non-optimal-but-
        # feasible case as an infeasible-for-counting-purposes outcome
        # (status["optimal"] gates cost/feasible_costs, not the mere
        # absence of an exception) -- preserved exactly, not touched by
        # this fix.
        if status["optimal"]:
            cost = float(pyo.value(m.system_cost_npv))
            draw_records.append({
                **realisation_fields,
                "feasible": True,
                "status": status["status"],
                "cost_usd": cost,
                "error_message": None,
            })
            feasible_costs.append(cost)

            # Band-edge diagnostic (Item 4): only meaningful under
            # --fix-gas fixed with a nonzero band, and only for years
            # whose band has nonzero width (base_gas_add[t] > 0). "At
            # edge" uses 1% of the band half-width eps as the closeness
            # threshold, so it scales with whatever --fix-tolerance was
            # chosen rather than an arbitrary absolute MW figure.
            if args.fix_gas == "fixed" and args.fix_tolerance > 0:
                for t in T:
                    if base_gas_add[t] == 0.0:
                        continue
                    lb, ub = gas_add_bounds[t]
                    solved = pyo.value(m.gas_add[t])
                    edge_tol = max(1e-12, 0.01 * (ub - lb))
                    if abs(solved - lb) <= edge_tol or abs(solved - ub) <= edge_tol:
                        n_gas_add_at_edge += 1
                    else:
                        n_gas_add_interior += 1
        else:
            draw_records.append({
                **realisation_fields,
                "feasible": False,
                "status": status["status"],
                "cost_usd": None,
                "error_message": None,
            })
            n_infeasible += 1

    if n_infeasible > 0:
        print(f"    WARNING: {n_infeasible}/{args.n_draws} draws infeasible "
              f"under the fixed plan -- the infeasible share is itself a "
              f"reliability-risk result, recorded below with each "
              f"infeasible draw's realisation (draws.jsonl), not dropped.")

    # ---- Write draws.jsonl FIRST, unconditionally -----------------------
    # Moved ahead of risk-metrics computation (TASK.md, "an all-infeasible
    # run must still write its output"): the PRIOR ordering computed risk
    # metrics before writing draws.jsonl, so an empty feasible_costs list
    # (compute_risk_metrics raising ValueError, src/stochastic.py:213-214,
    # NOT edited here) killed the run before EITHER output file existed --
    # the --fix-gas fixed pilot's 100/100-infeasible run left neither
    # draws.jsonl nor summary.json on disk for exactly this reason (see
    # docs/task_log_stochastic_runner.md, Item 3). draws.jsonl depends only
    # on draw_records, which is always fully populated regardless of
    # feasibility, so there is no reason for its write to be gated behind
    # risk metrics at all.
    draws_path = out_dir / f"draws__{hash8}.jsonl"
    with open(draws_path, "w") as f:
        for record in draw_records:
            f.write(json.dumps(record) + "\n")
    print(f"\nSaved: {draws_path}")

    # ---- Risk metrics, guarded against an all-infeasible run ------------
    # An all-infeasible run is a result, not an error (TASK.md): skip
    # compute_risk_metrics() entirely rather than let its ValueError on an
    # empty list (src/stochastic.py:213-214) abort the run, and record
    # every risk field EXPLICITLY None in summary.json below -- so a
    # reader can tell "not computed" (every draw infeasible) from "absent"
    # (a bug). Exits 0 either way: this branch is not an error path.
    if len(feasible_costs) > 0:
        risk = compute_risk_metrics(feasible_costs, alpha=0.95)
        p5, p50, p95 = (
            float(np.percentile(feasible_costs, 5)),
            float(np.percentile(feasible_costs, 50)),
            float(np.percentile(feasible_costs, 95)),
        )
        sd = float(np.std(feasible_costs))
        mean_cost, var_95, cvar_95 = risk["expected"], risk["VaR"], risk["CVaR"]

        print(f"\n{'='*70}")
        print(f"  RISK METRICS (n_feasible={len(feasible_costs)}/{args.n_draws})")
        print(f"{'='*70}")
        print(f"  base deterministic cost : {base_cost:,.2f}")
        print(f"  mean                    : {mean_cost:,.2f}")
        print(f"  sd                      : {sd:,.2f}")
        print(f"  p5 / p50 / p95          : {p5:,.2f} / {p50:,.2f} / {p95:,.2f}")
        print(f"  VaR (95%)               : {var_95:,.2f}")
        print(f"  CVaR (95%)              : {cvar_95:,.2f}")
        print(f"  infeasible              : {n_infeasible}/{args.n_draws} "
              f"({n_infeasible / args.n_draws:.1%})")
    else:
        mean_cost = sd = p5 = p50 = p95 = var_95 = cvar_95 = None
        print(f"\n{'='*70}")
        print(f"  RISK METRICS: NOT COMPUTED — 0/{args.n_draws} draws feasible")
        print(f"{'='*70}")
        print(f"  base deterministic cost : {base_cost:,.2f}")
        print(f"  All {args.n_draws} draws were infeasible under this fixed "
              f"plan (--fix-gas {args.fix_gas}) -- this is itself a "
              f"reliability-risk RESULT, not a failure. Every risk field "
              f"in summary.json is recorded explicitly null, not omitted. "
              f"See draws.jsonl for each draw's realisation and "
              f"error_message.")

    # ---- Write summary.json ----------------------------------------------
    summary = {
        "arm_tag": arm_tag,
        "standard_arms": standard_kwargs,
        "own_args": {
            "n_draws": args.n_draws,
            "seed": args.seed,
            "demand_sigma": args.demand_sigma,
            "carbon_mu": args.carbon_mu,
            "carbon_sigma": args.carbon_sigma,
            "fix_gas": args.fix_gas,
            "fix_tolerance": args.fix_tolerance,
            "gas_weights": args.gas_weights,
        },
        # TASK 08, Part D prerequisite: 31_compute_vss_bound.py's Guard 5
        # (ws_case_scenario_matches_stage1) needs these to compare against
        # -- they are hardcoded literals at the base_scenario =
        # load_scenario(...) call above (capital_case="unconstrained"), not CLI
        # arms, so standard_arms/own_args above never carried them.
        # financing_regime/required_margin are read directly off the
        # solved base_scenario dict (load_scenario()'s own resolved
        # values, "traditional" / dscr_case-derived here since this call
        # never overrides either), not duplicated from scenarios.py's
        # internal default logic. has_emissions_cap is False by
        # construction -- emissions_cap_by_year=None is passed literally
        # to the Stage-1 build_model_sliced() call above, never CLI-driven.
        "stage1_scenario_fields": {
            "capital_case": "unconstrained",  # must match the load_scenario() call above
            "financing_regime": base_scenario["financing_regime"],
            "required_margin": base_scenario["required_margin"],
            "has_emissions_cap": False,
        },
        "fixed_plan": {
            "solar_public_add": fixed_plan["solar_public_add"],
            "solar_eaas_add": fixed_plan["solar_eaas_add"],
            "storage_add": fixed_plan["storage_add"],
            "gas_add_fixed": base_gas_add if args.fix_gas == "fixed" else None,
        },
        # Item 4 (docs/task_log_stochastic_runner.md): whether the band is
        # doing real numerical work. Only populated under --fix-gas fixed
        # with --fix-tolerance > 0; null under exact fixation (tolerance=0,
        # where the concept does not apply -- every nonzero year is an
        # exact pin by construction) or --fix-gas free.
        "gas_add_band_diagnostics": (
            {
                "fix_tolerance": args.fix_tolerance,
                "n_at_edge": n_gas_add_at_edge,
                "n_interior": n_gas_add_interior,
                "n_nonzero_band_years_per_draw": len(T) - n_gas_add_zero_width_years,
                "n_zero_width_years_per_draw": n_gas_add_zero_width_years,
                "note": (
                    "counts are summed across every feasible draw's "
                    "nonzero-band years; zero-width years (base gas_add=0) "
                    "are excluded as uninformative -- a relative band around "
                    "0 is always [0,0], an exact fix regardless of tolerance"
                ),
            }
            if (args.fix_gas == "fixed" and args.fix_tolerance > 0)
            else None
        ),
        "base_deterministic_cost_usd": base_cost,
        "n_draws": args.n_draws,
        "n_feasible": len(feasible_costs),
        "n_infeasible": n_infeasible,
        "infeasible_fraction": n_infeasible / args.n_draws,
        "mean_cost_usd": mean_cost,
        "sd_cost_usd": sd,
        "p5_cost_usd": p5,
        "p50_cost_usd": p50,
        "p95_cost_usd": p95,
        "VaR_95_usd": var_95,
        "CVaR_95_usd": cvar_95,
    }
    summary_path = out_dir / f"summary__{hash8}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {summary_path}")
    print(f"Saved: {out_dir / 'INDEX.md'} (hash8 -> full arm tag)")


if __name__ == "__main__":
    main()
