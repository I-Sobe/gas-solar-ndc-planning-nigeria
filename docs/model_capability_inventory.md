# Model capability inventory

**Per TASK.md.** Read at HEAD. No solves, no runners, no commits, no
edits to `src/` or `scripts/`. Prior inventories
(`formulation_inventory_sliced.md`, `runner_inventory.md`, the claims
ledger) consulted only to cross-check, never trusted in place of
reading the code; disagreements are reported, not silently resolved in
either direction.

**Four-state table**, as specified: IMPLEMENTED (code exists) < WIRED
(reachable from a runner's CLI/call path) < EXERCISED (a `results/` run
used it) < REPORTED (cited in `findings_frozen.md`). Each module below
gets the highest level reached, with what blocks the next stated
explicitly.

---

## PART 1 — the module map

`src/` contains **21 files** (20 `.py` modules + `__init__.py`), 14,607
lines total. Importers were checked by grepping every import form
(`from src.X import`, `from X import`, `import X`) across `src/`,
`scripts/` (including `scripts/deferred/` and `scripts/retired/`),
`tests/`, and `notebooks/*.ipynb`, then re-checked with a broader
unanchored search for the four modules that came back with zero hits,
to rule out a missed import spelling.

### Orphaned modules — 4, not 2

TASK.md names `EaaS.py` and `stochastic.py` as known candidates. Both
confirmed. **Two more found, not previously flagged: `gas_balance.py`
and `postprocess.py`.** All four have zero importers anywhere in the
repository.

| module | what it would do if called | equivalent elsewhere? |
|---|---|---|
| `EaaS.py` | Compute EaaS deployment rollout and financing (capital recovery factor, deployment schedule, financing split) as a standalone transformation layer | **Yes — superseded by a different implementation.** EaaS financing is built directly into `optimize_model.py`/`optimize_model_sliced.py` (`financing_regime="eaas"` branches, e.g. `optimize_model.py:689-730`) and `blended_finance.py` (WACC/tranche blending), neither of which imports this module. `EaaS.py`'s own functions (`compute_eaas_deployment`, `compute_eaas_financing`) are never called; the concept "EaaS" is fully implemented, just not through this file. |
| `stochastic.py` | Fixed-plan re-dispatch Monte Carlo: solve once, freeze the plan, re-solve under drawn uncertainty, no new capacity built | **Partially — different question.** `29_run_parameter_uncertainty.py`'s Sobol analysis re-solves the FULL LP (capacity can change) under drawn parameters; `stochastic.py`'s method holds capacity fixed and re-dispatches only. Not a duplicate — a different, unimplemented-in-practice method. See Part 5 for the full account. |
| `gas_balance.py` | Project annual gas availability from a parametric decline curve (`q0_th`, decline rate) and convert gas energy to electricity | **Yes.** `gas_supply.py`'s `gas_available_power()` — CSV-driven, not decline-curve-driven — is what's actually imported everywhere gas availability is needed. `gas_balance.py`'s decline-curve approach appears abandoned in favour of the CSV-scenario approach. |
| `postprocess.py` | `export_benchmark_comparison()`: given a gas shadow-price series and netback benchmarks, compute exceedance shares per benchmark | **Yes.** `06_run_gas1_shadow_benchmarks.py` builds its own `benchmark_comparison` dict inline (`:176-195`) and writes `benchmark_comparison.csv` itself — duplicating what this function does, without calling it. |

None of the four is WIRED. All four remain at IMPLEMENTED.

### Per-module detail

**`src/__init__.py`** — empty (0 bytes). Package marker only; not a
capability.

**`src/EaaS.py`** — 282 lines. Purpose (quoted): *"Implements
Energy-as-a-Service (EaaS) as a system-level coordination and
financing paradigm for national-scale energy planning."* Functions:
`_capital_recovery_factor`, `_validate_eaas_deployment_params`,
`compute_eaas_deployment`, `compute_eaas_financing`. **Importers: none.
State: IMPLEMENTED.** See orphan table above.

**`src/blended_finance.py`** — 295 lines. Purpose (quoted): *"Represents
the capital structure behind EaaS-financed distributed solar as a BLEND
of two tranches: (1) a CONCESSIONAL tranche... (2)..."* Functions:
`blended_wacc`, `implied_concessional_share`,
`concessional_envelope_npv`, `resolve_private_rate`,
`commercial_tenor_npv_factor`, `legacy_blended_npv_factor`,
`resolve_remaining_npv_factor`. **Importers:** `optimize_model.py`,
`optimize_model_sliced.py`, `optimize_experiments.py`,
`scripts/04_run_fin2_tariff_sweep.py`. Imported by both live builders,
which every active runner uses. **State: REPORTED** — F-22, the B\*
triangulation, and the whole concessional-envelope/financing
discussion in `findings_frozen.md` and Chapter 3.4 depend on this
module's arithmetic.

**`src/cli_arms.py`** — 361 lines. Purpose (quoted): *"Shared CLI
arm-flag helper for scripts/."* Written specifically to close the L27
defect (a hand-copied 17-flag argparse block silently drifting across
runners). Functions: `_reserve_margin_type`, `_arm_specs`,
`add_standard_arms`, `standard_arms_kwargs`. **Importers: 18 active
scripts** (`02_run_eaas.py` through `29_run_parameter_uncertainty.py`,
`probe_2_6_structure.py`) plus itself. **State: EXERCISED** by
construction (every sweep run passes through it); not itself a
modelled capability, so REPORTED does not apply — it is infrastructure
findings are built on, not a finding.

**`src/demand.py`** — 145 lines. Purpose (quoted): *"Provides
national-scale baseline electricity demand projections and a simple
electrification/access-based adjustment suitable for long-term
planning studies."* Functions: `project_baseline_demand`,
`apply_access_adjustment`. **Importers:** `optimize_experiments.py`,
`optimize_model.py`, `optimize_model_sliced.py`, `scenarios.py` — all
import and actually CALL `project_baseline_demand` (confirmed: it
appears at `optimize_model.py:577`, `optimize_model_sliced.py:752`,
`scenarios.py:2969`, three sites in `optimize_experiments.py`, not just
in the import line). **State: REPORTED** (demand underlies every
reported figure). **But `apply_access_adjustment` itself is never
called anywhere** — a dead function inside a live module, consistent
with `scenarios.py`'s own docstring stating the Tier-2/3
access-adjusted demand bases are "DEPRECATED and raise in
`load_scenario`."

**`src/dispatch.py`** — 137 lines. Purpose (quoted): *"Implements a
deterministic, greedy, annual energy-balance accounting routine used to
evaluate supply–demand feasibility and unserved energy... This is NOT
an optimization dispatch model and does NOT construct Pyomo
variables."* Function: `dispatch_energy`. **Importer:**
`optimize_experiments.py` only, called from `run_deterministic_scenario`
(`:306`). **State: EXERCISED** — `results/eaas/` (from
`02_run_eaas.py`, which calls `run_deterministic_scenario`) contains
`diagnostics.json`/`summary.json`/`timeseries.csv`, so this accounting
path has produced real output at some point. **Not REPORTED**: no
citation of this accounting-level dispatch path found in
`findings_frozen.md`, which reports exclusively from the Pyomo LP
builders. Whether the `results/eaas/` output is current at HEAD is a
Part 4 question (`02_run_eaas.py` has no module docstring and mixes
both the old dispatch path and the Pyomo builder — flagged there).

**`src/economics.py`** — 169 lines. Purpose (quoted): *"Contains:
Financial discounting and Net Present Value calculations. Deterministic
and planning-level cost calculations... All functions are stateless and
scenario-driven."* Functions: `gas_cost`, `solar_capex`, `carbon_cost`,
`unserved_energy_cost`, `total_system_cost`, `discount_factor`, `npv`.
**Importer:** `optimize_experiments.py`, but **only 5 of the 7
functions are imported** (`:27-33`) — `discount_factor` and `npv` are
never imported or called anywhere in the repository. **State:
EXERCISED** for the 5 imported functions (same `run_deterministic_scenario`
call path as `dispatch.py`, `:315-336`); **IMPLEMENTED only** for
`discount_factor`/`npv` — dead code within a live module. Not REPORTED,
same reasoning as `dispatch.py`.

**`src/gas_balance.py`** — 94 lines. **Importers: none. State:
IMPLEMENTED.** See orphan table above.

**`src/gas_benchmarks.py`** — 162 lines. Purpose (quoted): *"Single
source of truth for the GAS-1 / GAS-2 gas-price benchmarks... Every
site now imports from here. Nothing else may define either name."*
Functions: `lng_netback_usd_mmbtu`, `_netbacks`, plus module-level
benchmark dictionaries and `MMBTU_PER_TWH_TH`. **Importers:**
`06_run_gas1_shadow_benchmarks.py`, `07_run_gas2_eaas_gas_relief.py`,
`08_phase8_analysis.py`, `optimize_experiments.py`. **State: REPORTED**
— F-30's entire benchmark-exceedance table (Chapter 4, and
`docs/task_log_findings_reverify_L60.md` this session) is computed from
this module's constants, re-verified directly against it this session.

**`src/gas_supply.py`** — 69 lines. Function: `gas_available_power`
(loads gas deliverability from CSV, scenario-keyed). **Importer:**
`optimize_experiments.py` (called at `:268` inside
`run_deterministic_scenario`, and again at `:2406` inside
`extract_planning_diagnostics` — the latter is imported by essentially
every active runner). **State: REPORTED** — every gas-deliverability
figure in `findings_frozen.md` ultimately reads through this function
via `extract_planning_diagnostics`.

**`src/io.py`** — 480 lines. Purpose (quoted): *"Low-level helpers for
loading configuration files and persisting numerical model outputs.
This module does not perform plotting or reporting."* Functions:
`load_yaml`, `save_results`, `load_solar_capex_by_year`,
`_extend_capex_series`, `load_storage_capex_by_year`,
`load_om_cost_by_year`, `compute_om_cost_npv`, `load_econ` (the
`genset_case`-REQUIRED function from item B of the §3.6 inputs report
this session). **Importers:** effectively every active runner (27
files across `scripts/`, `scripts/deferred/`, `scripts/retired/`) plus
`optimize_experiments.py`, `optimize_model.py`, `optimize_model_sliced.py`,
`scenarios.py`, and one shell script (`verify_2_6_commit_a.sh`). **State:
REPORTED** — the most heavily imported module in `src/` after
`scenarios.py` itself.

**`src/optimize_experiments.py`** — 4,070 lines, 30 top-level
functions. Purpose (quoted): *"Optimization experiments and
orchestration... This module orchestrates experiments only. Planning
optimization and operational simulation intentionally use different
storage abstractions and are not numerically comparable at the
component level."* **This is the largest single finding of Part 1**:
of its 30 functions, the current active runners
(`01_`/`01b_`/`02_run_ndc_caps`/`03_`/`05_`/`06_`/`18_`/`28_`/`29_`,
etc.) import almost exclusively **one** function,
`extract_planning_diagnostics` — confirmed by reading every
`from src.optimize_experiments import` line in every currently-numbered
runner (see item C of `docs/task_log_ch36_inputs.md` for the same kind
of direct-source check applied here). The other 29 functions —
`run_annual_cap_case`, `run_deterministic_scenario`,
`run_reliability_sweep`, `run_rel3_financing_frontier`,
`run_fin2_blended_bankability_sweep`, `run_dem1_demand_sensitivity`,
`run_dem2_growth_gas_matrix`, `run_str1_storage_parameter_sweep`,
`run_rel2_marginal_cost_curves`, `run_rel1_feasibility_matrix`,
`run_gas_regime_ndc_matrix`, `run_all_deterministic_scenarios`,
`run_tariff_bankability_sweep`, `run_tariff_public_capital_frontier`,
`run_financing_vs_resource_test`, and others — are each called only by
their own dedicated, individually-numbered runner script (e.g.
`run_gas_regime_ndc_matrix` only by `06_run_gas1_shadow_benchmarks.py`;
`run_deterministic_scenario`/`run_tariff_public_capital_frontier` only
by `02_run_eaas.py`). **`tests/test_run.py` imports
`generate_epsilon_pareto` from this module — a function that does
not exist anywhere in the current file** (confirmed: zero matches for
that name in `src/optimize_experiments.py` or anywhere else in the
repository). **The only test file in this repository is broken and
cannot be collected** without an `ImportError`. State per function
varies from REPORTED (`extract_planning_diagnostics`) down to a
genuinely dead reference (`generate_epsilon_pareto`, called only from
a test that cannot run) — full per-runner attribution is Part 4's job;
flagged here because it changes what "importer" means for this file.

**`src/optimize_model.py`** — 1,989 lines. Purpose (quoted): *"Planning-
level Pyomo model construction... The weighted-sum formulation
(weight_cost / weight_emissions) was removed (Plan 0.4, item B,
docs/numerical_hygiene_0_4.md): it was dead code, never set by any
caller. The objective is unconditionally system_cost_npv, minimised."*
This is TASK.md's own named example of a removed capability, confirmed
here directly in the module's own docstring — full accounting is Part
2's job. Functions: `build_model`, `leakage_diagnostic`,
`firm_capacity_diagnostic`, `solve_model`. **Importers:** ~20 active
and retired/deferred scripts, plus `optimize_experiments.py` and
`stochastic.py` (the one place `stochastic.py` itself appears as an
IMPORTER, not an importED module — it imports `optimize_model`, which
is exactly why it is not itself imported by anything: it is a
dead-end consumer). **State: REPORTED** — the annual builder underlies
`findings_frozen.md`'s scope-limited findings (e.g. F-30's GAS-1, run
`conv_none` on this builder).

**`src/optimize_model_sliced.py`** — 2,161 lines. Purpose (quoted):
*"Parallel implementation of build_model() with intra-annual
resolution. The annual model in optimize_model.py is retained unchanged
so that: (a) existing runners keep working during migration, and (b)
the annual-vs-sliced comparison is available as a methods result."*
Functions: `slice_hours`, `_normalised_demand_weights`,
`_solar_energy_per_mw_by_slice`, `_hydro_season_energy_share`,
`build_model_sliced`, `annual_totals`, `firm_capacity_diagnostic`,
`slice_detail`, `conversion_diagnostic`. **Importers:** 13 active
scripts, plus `scenarios.py`. **State: REPORTED** — `methodology.md`
§2 states the sliced builder is THE reporting model, and every figure
in `findings_frozen.md` not explicitly scoped to the annual builder
comes from here.

**`src/output_naming.py`** — 71 lines. Purpose (quoted): *"short,
deterministic path components for a full arm tag that exceeds the
filesystem's ~255-byte per-path-component limit."* Functions:
`hash8_of`, `short_tag`, `append_index`. **Importers:**
`01b_run_baseline_sliced.py`, `02_run_ndc_caps.py`,
`03_run_ndc_eaas.py`, `26_run_f3_identification.py`. **State:
EXERCISED** (every hash8-tagged output file in `results/` under these
runners' directories was named by this module) — not itself a modelled
capability, so REPORTED does not apply.

**`src/postprocess.py`** — 18 lines. **No module-level docstring** (the
one function's own docstring is the only documentation). **Importers:
none. State: IMPLEMENTED.** See orphan table above.

**`src/scenarios.py`** — 3,465 lines, 36 top-level functions (listed in
full in Part 3, the arm system). Purpose (quoted): *"Defines
deterministic planning scenarios as structured parameter dictionaries
for use in system evaluation and optimization studies. This module
provides scenario DEFINITIONS ONLY. It does NOT execute models, run
experiments, or aggregate results."* **Importers:** effectively every
active runner (30 files), `cli_arms.py`, `optimize_experiments.py`,
`stochastic.py`, `tests/test_run.py`, both notebooks that touch `src`.
**State: REPORTED** — the single most heavily imported module in the
repository; every arm value cited anywhere in `findings_frozen.md`
originates here.

**`src/solar.py`** — 96 lines. Functions: `solar_generation`,
`solar_capacity_trajectory`. **Importer:** `optimize_experiments.py`
only (called from `run_deterministic_scenario`, `:288`). **State:
EXERCISED** (same `results/eaas/` evidence as `dispatch.py`); **not
REPORTED** — the Pyomo builders compute solar generation directly
inside their own constraint/expression bodies (confirmed by this
module's absence from the builders' import lists), so no
`findings_frozen.md` figure traces through this function.

**`src/stochastic.py`** — 227 lines. **Importers: none. State:
IMPLEMENTED.** Full method, limitation-note quote, and "what would be
needed to wire it" are Part 5's job per TASK.md; noted here only as
the module map's confirmation of the known orphan.

**`src/storage.py`** — 187 lines. Class: `BatteryStorage`. **Importer:**
`optimize_experiments.py` only (instantiated at `:300`, inside
`run_deterministic_scenario`). **State: EXERCISED**, **not REPORTED**
— same reasoning as `solar.py`: the Pyomo builders model storage
directly as LP variables/constraints, not through this class.

**`src/utils.py`** — 129 lines. Purpose (quoted): *"Lightweight,
dependency-safe helper functions shared across modules."* Functions:
`interpolate_series`, `mw_to_twh`, `assert_non_negative`,
`year_index`, `json_safe`. **Importers:** 11 active scripts plus
`EaaS.py`, `demand.py`, `dispatch.py`, `gas_balance.py`, `gas_supply.py`,
`solar.py`, `storage.py` (including two of the four orphaned modules —
`EaaS.py` and `gas_balance.py` import `utils.py`, which is otherwise
unremarkable except that it means `utils.py` cannot itself be
considered evidence those two modules are wired; importing a live
module is not the same as being imported by one). **State: REPORTED**
(via every module that uses it and is itself reported).

---

## STOP — counts and report

**Orphaned modules: 4** — `EaaS.py`, `gas_balance.py`, `postprocess.py`,
`stochastic.py`. TASK.md named 2 of these in advance; **2 were not
previously flagged anywhere I found** (`gas_balance.py`,
`postprocess.py`). All four have a working equivalent elsewhere in the
codebase except `stochastic.py`, whose method (fixed-plan re-dispatch)
has no true equivalent — Sobol (`29_`) answers a related but distinct
question (full re-optimisation under parameter draws, not fixed-plan
re-dispatch under drawn scenarios).

**Dead functions inside otherwise-live modules, found incidentally
while checking importers** (not an exhaustive per-function sweep of
all ~150 functions in `src/` — flagged where found, not claimed
complete): `economics.discount_factor`, `economics.npv`,
`demand.apply_access_adjustment`.

**Broken importer: `tests/test_run.py`** imports
`generate_epsilon_pareto` from `optimize_experiments.py`, a function
that does not exist in the current file. The repository's only test
file cannot be collected without an `ImportError` at HEAD.

**A structural finding beyond the module-count question**:
`optimize_experiments.py` (4,070 lines, 30 functions) is imported by
nearly every active runner, but almost always for exactly one function
(`extract_planning_diagnostics`); the other 29 are each reachable only
through their own single, specifically-numbered runner. Whether those
runners are themselves current is Part 4's question, but it means
"imports `optimize_experiments.py`" is a weak signal for a function's
own state — this is noted so Parts 2-4 don't over-credit a function as
WIRED merely because its containing module is imported somewhere.

**Per TASK.md: this may change the task's shape**, since 4 orphans
against an expected 2 is double the anticipated count, and two of them
(`gas_balance.py`, `postprocess.py`) were not named in advance. Ready
to proceed to Part 2 (the two builders, capability by capability) on
your instruction — stopping here as instructed.

---

## PART 1B — the four orphaned modules, read in full

Every line read, not just docstrings/signatures. No retirement
recommendation is made here; each module's own recorded history is
reported where one exists, and the decision is left to Zeus and Chat
together, per instruction.

### `src/EaaS.py` (283 lines)

**What it computes, function by function.**

- `_capital_recovery_factor(rate, n_years)` — standard CRF,
  `r(1+r)^n / ((1+r)^n - 1)`, with `rate == 0.0` handled as a special
  case returning straight-line `1/n_years` (avoids 0/0). Raises on
  negative rate or non-positive `n_years`.
- `_validate_eaas_deployment_params(eaas_params)` — bounds-checks
  `penetration_target`/`solar_share` to `[0,1]`, `solar_cf` to `(0,1]`,
  and asserts `ramp_years`/`storage_hours` non-negative.
- `compute_eaas_deployment(years, demand_twh, eaas_params)` — sizes
  distributed solar/storage from an EaaS PENETRATION TARGET, not a
  cost optimisation. The core arithmetic:
  ```
  coverage_fraction[i] = penetration_target * (year - start_year + 1) / ramp_years   # during ramp
  eaas_energy_twh = coverage_fraction * demand_twh
  solar_energy_twh = eaas_energy_twh * solar_share
  solar_capacity_mw = solar_energy_twh * 1e6 / (solar_cf * HOURS_PER_YEAR)
  storage_capacity_mwh = solar_capacity_mw * storage_hours
  ```
  **Modelling assumption embodied in this arithmetic**: EaaS
  penetration is a demand-side COVERAGE TARGET applied as a fixed
  fraction of baseline demand, ramped linearly, with solar SIZED to
  meet that coverage at a given capacity factor — capacity is DERIVED
  from a penetration schedule, not chosen by any optimiser. Storage is
  sized as a flat multiple of solar capacity (`storage_hours`), not by
  any reliability or cost criterion. This is a top-down deployment
  curve, structurally incompatible with the LP's bottom-up,
  cost-minimising `solar_public_add`/`solar_eaas_add`/`storage_add`
  decision variables — even if wired in, it would not slot into the
  Pyomo model as a constraint; it would replace the decision entirely
  with an exogenous schedule.
- `compute_eaas_financing(years, eaas_deployment, eaas_params, system_costs)`
  — converts incremental EaaS CAPEX into an annuity-style **service
  payment stream**:
  ```
  delta_capex[t] = delta_solar_mw[t] * solar_capex_per_mw + delta_storage_mwh[t] * storage_capex_per_mwh
  crf = _capital_recovery_factor(service_wacc, contract_tenor_years)
  annual_payment = delta_capex[t] * crf
  annual_service_cost[t : t+contract_tenor_years] += annual_payment    # rolling annuity window
  ```
  **Modelling assumption embodied in this arithmetic**: each year's
  incremental CAPEX is amortised over a FIXED, explicit
  `contract_tenor_years` at a FIXED `service_wacc`, and the resulting
  payment is added to every year within that rolling tenor window —
  a lease/PPA-style cash-flow structure, not a balance-sheet capital
  charge. `accounting_mode` (`"off_balance"`/`"on_balance"`) is
  validated but **"is retained as metadata only and does NOT affect
  numerical results"** (the function's own docstring, quoted) — even
  within this module, the on/off-balance-sheet distinction is not
  actually modelled numerically, only labelled.

**Capability comparison, answered directly.**

Grepped `contract_tenor`, `service_wacc`, `accounting_mode`,
`off_balance`/`on_balance` across `src/scenarios.py`,
`src/optimize_model.py`, `src/optimize_model_sliced.py`, `src/io.py`:
**zero matches anywhere.** The wired EaaS mechanism
(`optimize_model.py:689-765`, quoted) is structurally different:

> "EaaS-financed and publicly-financed solar are priced at the SAME
> unit CAPEX -- they are the same physical assets. The private cost of
> capital is represented NOT by an objective-side CAPEX premium but by
> the bankability constraint... The EaaS mechanism's effect in this
> model is therefore structural (budget decoupling + a harder
> bankability test), not a cheaper sticker price."

The bankability constraint itself:
```
effective_private_cost = capex - eaas_subsidy[t]
bankable_revenue = tariff * solar_energy_per_mw * remaining_npv_factor[t] * solar_eaas_add[t] / required_margin
effective_private_cost <= bankable_revenue
```

**So, answering the question directly: yes, `EaaS.py` holds three
things the wired representation does not.**
1. **An explicit service-payment stream** — a cash-flow object
   (`annual_service_cost_usd`, USD/year) that the LP never constructs;
   the LP's bankability constraint is a feasibility test on discounted
   revenue vs. capex, not a computed payment series.
2. **A contract tenor independent of any other model tenor** — the LP
   discounts EaaS revenue via `remaining_npv_factor` (tied to asset
   life / `bank_tenor_case`), not a distinct EaaS-specific
   `contract_tenor_years`; there is no separate "how long is the EaaS
   contract" concept in the wired model at all.
3. **A named on/off-balance-sheet distinction** — present as a
   validated string in `EaaS.py`, entirely absent (as a concept, not
   just unwired) from the LP, and, by the module's own docstring's
   admission, not even numerically effectual within this module
   either.

**What the wired implementation holds that `EaaS.py` does not**: the
budget-decoupling mechanism itself (EaaS solar doesn't draw the public
envelope — the entire point of F-22's 60%+ capital-requirement
reduction finding), the bankability coverage-cushion test
(`required_margin`), and integration with the rest of the system LP
(gas, storage, emissions cap, grid service obligation) as simultaneous
constraints rather than a standalone post-hoc costing exercise.
`EaaS.py` cannot represent any interaction between EaaS deployment and
the rest of the system — it takes `demand_twh` as a fixed input and
produces a capacity/cost schedule with no feedback to gas dispatch,
emissions, or grid service.

**Interface currency.** `import numpy as np`; `from src.utils import
assert_non_negative` — both resolve at HEAD (`utils.assert_non_negative`
confirmed unchanged, Part 1). No scenario-dict keys are read (the
functions take explicit typed arguments, not a `scenario` dict), so
there is no `load_scenario()` key to go stale against. **It would run
today if called** with a correctly-shaped `eaas_params`/`system_costs`
dict — confirmed by direct inspection (no call to anything outside
`numpy` and `utils`) and by `python3 -m py_compile`. The gap is not a
broken interface; it is that nothing in the current codebase calls it
with anything.

**Git history.** Last touched `9c4ce2e` (2026-09-06), "Plan 5.5
doc-code parity: EaaS.py wholly unimported, storage.py loss convention
differs from LP, solar_capex is an undiscounted mean not an annuity" —
this is the commit that added the current docstring's "NOT CALLED BY
THE LP" section (confirmed: `docs/task_bucketA_parity.md` Part B1 is
the task that made this edit, docstring-only, verified
`py_compile`-clean before and after). **A recommendation already
exists on record, not acted on**:
`docs/task_bucketA_parity.md:244`, quoted in full:

> "Recommendation (recorded, not acted on). Per B1's instruction, the
> functions are not deleted. Given the module is now confirmed *wholly*
> unimported (stronger than TASK.md's financing-only premise), the
> recommendation is: if the codebase later commits to removing dead
> code, this whole module is a candidate for deletion, not just
> `_capital_recovery_factor` / `compute_eaas_financing` — but that
> decision, and any check of whether the thesis prose references this
> module by name, is out of scope here."

No commit message states an intent to retire; the recommendation lives
in a task report's prose, conditional ("if the codebase later commits
to removing dead code"), not an executed or scheduled decision.
Earlier history: `255ad55` (2026-08-16, unrelated glyph fix),
`30277bc` (2025-12-31, "Modify EaaS.py: Refactoring" — pre-dates the
current F1/f1_tranche mechanism entirely).

---

### `src/stochastic.py` (227 lines)

**What it computes, function by function.**

- `sample_uncertainties(N, base_scenario, carbon_mu, carbon_sigma,
  demand_sigma=0.01, seed=None)` — draws N independent scenario
  perturbations from a deep-copied base scenario:
  ```
  scenario["demand_growth"] = max(0.0, np.random.normal(base_scenario["demand_growth"], demand_sigma))
  scenario["gas_scenario"]  = np.random.choice(labels, p=weights)          # from gas_probability_weights()
  scenario["carbon_price"]  = np.random.lognormal(mean=carbon_mu, sigma=carbon_sigma)   # only if carbon_active
  ```
  **Modelling assumption embodied here**: demand growth perturbation is
  a truncated-at-zero NORMAL on the growth RATE (not the level); gas
  regime is drawn from a DISCRETE prior over named scenario labels
  (not a continuous perturbation); carbon price, when active, is
  LOGNORMAL (right-skewed, cannot go negative) — three different
  distribution families for three different kinds of uncertainty, a
  deliberate choice stated in the module's own docstring rather than a
  single generic noise model.
- `run_stochastic_simulation(...)` — the two-stage fixed-plan
  re-dispatch: solve once at `base_scenario` to get
  `solar_public_add`/`solar_eaas_add`/`storage_add` per year, `.fix()`
  those in every subsequent draw's model (via `build_model(...,
  fixed_plan=fixed_plan)`), and re-solve. Infeasible draws are dropped
  and counted, not silently discarded:
  > "the infeasible share is itself a reliability-risk result worth
  > reporting" (quoted from the printed warning, `:186-188`).
- `compute_risk_metrics(values, alpha=0.95)` — expected value, variance,
  VaR (upper-tail quantile), CVaR (mean of the tail at or above VaR).
  Standard, no embedded modelling choice beyond the upper-tail
  convention (stated: "higher cost = worse").

**Capability comparison.** This is a genuinely distinct METHOD from
everything else in the codebase, not a duplicate: it answers "how
robust is the plan the LP already chose," which neither the arm sweep
(different discrete plans, not one plan under noise) nor Sobol
(re-optimises the plan at every draw — no fixed-plan step at all) can
answer. **What it holds that nothing wired does**: a genuine two-stage
here-and-now/recourse structure, and an infeasibility-under-fixed-plan
diagnostic (a plan optimal at the base case can become INFEASIBLE, not
just costlier, under a draw — this is not measurable by Sobol's
full-reoptimisation approach, since Sobol always finds a new feasible
plan if one exists). **What it does not hold that Sobol does**:
continuous parameter uncertainty on physical/cost quantities (gas
efficiency, solar CF, O&M, conversion capex) — `stochastic.py` only
perturbs demand growth, gas REGIME (discrete), and carbon price.

**Interface currency — checked line by line against HEAD, not
assumed.** This module's imports and call patterns turn out to be
**substantially current**, not stale:

- `from src.scenarios import gas_probability_weights` — confirmed
  exists at HEAD (`scenarios.py:1032`), returns
  `{"baseline": 0.50, "downside": 0.25, "upside": 0.20,
  "shock_recovery": 0.05}` — all four labels are valid keys of
  `gas_deliverability_scenarios()` at HEAD (`scenarios.py:858-887`).
- `from src.optimize_model import build_model, solve_model` — both
  exist at HEAD with compatible signatures.
- `scenario["demand_growth"]`, `scenario["gas_scenario"]` — **both are
  still literal keys `load_scenario()` writes into the returned dict**
  (`scenarios.py:3051-3055`: `"demand_growth":
  demand_growth_scenarios()[demand_case]`, `"gas_scenario":
  gas_deliverability_scenarios()[gas_deliverability_case]`).
- `scenario["carbon_active"]`, `scenario["carbon_price"]` — both come
  from `carbon_policy_scenarios()[carbon_case]`, spread into the
  scenario dict at `scenarios.py:3436` (`**carbon_policy_scenarios()[carbon_case]`).
  Confirmed present in `carbon_policy_scenarios()`'s own return dict
  (`scenarios.py:844-853`).
- `build_model(..., fixed_plan=fixed_plan)` — `optimize_model.py`'s
  `build_model` signature still has a `fixed_plan=None` parameter
  (`:39`), and its body still fixes exactly the three keys
  `stochastic.py` supplies — `solar_public_add`, `solar_eaas_add`,
  `storage_add` (`optimize_model.py:476-481`), each indexed by year `t`
  — an exact match, not a coincidental overlap.
- `scenario["gas_scenario"]` flows into `gas_available_power(...,
  scenario_name=scenario["gas_scenario"])` inside `build_model`
  (`optimize_model.py:584-587`) — again an exact match to what
  `sample_uncertainties` writes.

**Verdict: it would very likely run today if called**, with one scope
caveat, not an interface break: it only imports and calls
`optimize_model.build_model` — the ANNUAL builder — never
`optimize_model_sliced.build_model_sliced`, the current REPORTING
builder (`methodology.md` §2). Wiring it as-is would produce annual-
builder Monte Carlo results, with the annual builder's own known scope
limits (no conversion lever, no intra-annual storage value) — a
methods caveat to state, not a broken interface. Part 2 (below) treats
"is it a runner away or a rewrite" using exactly this evidence.

**Git history — the strongest single fact in this section.** Last
touched `2380e5b` (2026-08-17), **the plan-2.5 end-of-horizon salvage
value commit** — a substantive, unrelated feature commit whose own
message states:

> "BOUNDARY GUARD: storage life (15) < horizon (21)... `solve_model()`
> now warns post-solve; `scenario=` passed at all 30 call sites so the
> guard is live everywhere."

The diff confirms `stochastic.py`'s two `solve_model()` call sites were
two of those 30, both updated to pass the new `scenario=` kwarg
(`solve_model(m_base)` → `solve_model(m_base, scenario=base_scenario)`,
and the equivalent in the re-dispatch loop). **This means `stochastic.py`
was treated as a live, in-scope call site and deliberately kept
interface-current one month before this inventory** — not left to rot.
No commit message anywhere in its history states an intent to retire or
supersede it; the trend, if anything, points the other way. Earlier
history is a long, active development sequence (`9815a64`, `ddc8f74`,
`72cad20`, and eight more commits back to `b0db0cb`, 2025-11-28), none
of it framed as deprecation.

---

### `src/gas_balance.py` (94 lines)

**What it computes, function by function.**

- `project_gas_availability_th(q0_th, decline_rate, start_year,
  end_year)` — single-parameter exponential decline:
  ```
  gas_available_th[t] = q0_th * (1.0 + decline_rate) ** t
  ```
  (`decline_rate` negative for a decline; `>= -1.0` enforced so
  availability cannot go negative.)
- `gas_to_power_electricity_twh(gas_to_power_th, eta_gas_to_power)` —
  `eta_gas_to_power * gas_to_power_th`, a flat linear efficiency
  conversion, `eta` bounded to `(0, 1]`.

**Modelling assumption embodied**: a CONSTANT percentage annual
decline from a single anchor value — the simplest possible gas-supply
trajectory, with no capacity to represent a non-monotonic path (a dip
then a recovery) or a policy-driven step change.

**Capability comparison — the question asked directly: does a
decline-curve formulation represent anything the CSV path cannot? No.**
Read `scripts/build_gas_deliverability.py` (the actual generator of the
CSV `gas_supply.gas_available_power()` reads) in full: it implements
**six** curve-shape families — `exponential_decline`, `drift`,
`lag_ramp_uplift`, `lag_then_drift`, `lag_then_decline`, and
`shock_recovery` — with `exponential_decline`'s own formula,
`anchor_value * ((1 - d) ** (y - anchor_year))`, mathematically
identical in form to `gas_balance.py`'s (opposite sign convention on
`d`, same shape). **`gas_balance.py`'s method is a strict subset of
what the CSV generator already does, independently re-implemented, not
extended.** The CSV path is strictly MORE expressive (five additional
shapes `gas_balance.py` cannot produce, including the
`shock_recovery` regime GAS-1's own downside/shock arms depend on) and
nothing here needs it to fill a gap. `gas_to_power_electricity_twh`'s
flat-efficiency conversion is similarly superseded: the wired builders
apply gas-to-power conversion via `scenario["gas_eta"]` directly inside
their own Pyomo expressions (confirmed: `gas_eta` scenario key exists,
`scenarios.py`, used directly in the builders — not through this
function).

**What the wired implementation holds that this does not**: arbitrary
per-year shape (not constrained to any parametric family at all — the
CSV can encode literally anything), the six named shape generators
above, and integration with everything downstream (gas balance
constraints, curtailment, deliverability exemptions) that a bare
utility function cannot provide.

**Interface currency**: `import numpy as np`; `from src.utils import
assert_non_negative` — both resolve. No scenario-dict dependency. Would
run today if called with valid arguments — same conclusion as
`EaaS.py`: not broken, just not called by anything, and with a
same-shape sibling function already doing its job inside the file that
actually feeds the model.

**Git history**: a single commit, `9bcc8d7` (2026-02-19), generic
message "modify: src files" — **this same commit created
`gas_balance.py`, `postprocess.py`, AND `src/build_gas_deliverability.py`
(the CSV generator's original location, later moved to
`scripts/build_gas_deliverability.py`) together**, alongside
substantive changes to `gas_supply.py`, `optimize_model.py`,
`economics.py`, `scenarios.py` (778 insertions across 8 files). The
most defensible reading of this history: `gas_balance.py` was one of
(at least) two candidate approaches to gas-availability modelling
tried in the same commit; the CSV+generator sibling was kept and
subsequently developed further (`gas_supply.py` grew from this commit's
68 lines to its current form; `build_gas_deliverability.py` moved to
`scripts/` and grew to its current six-shape form), while
`gas_balance.py` has not been touched since the day it was created. No
commit message anywhere states an intent to retire it — it was simply
never developed further after its sibling was chosen.

---

### `src/postprocess.py` (18 lines, no module docstring)

**What it computes.** One function, `export_benchmark_comparison
(gas_shadow_price_by_year, netbacks_usd_per_twh_th)`:
```
for name, p in netbacks_usd_per_twh_th.items():
    gaps = [v - p for v in lam]                       # lam = non-None shadow prices
    out[name] = {
        "share_years_power_gt_export": sum(v > p for v in lam) / len(lam),
        "min_gap_usd_per_twh_th": min(gaps),
        "max_gap_usd_per_twh_th": max(gaps),
        "max_shadow_price_usd_per_twh_th": max(lam),
    }
```

**Capability comparison — checked field-by-field, not assumed
identical.** `06_run_gas1_shadow_benchmarks.py` computes its own
`benchmark_comparison` dict inline (`:176-183`, read directly):
```
benchmark_comparison[bench_name] = {
    "benchmark_usd_per_twh_th": bench_val,
    "share_years_shadow_gt_benchmark": len(exceeding) / len(gs_vals),
    "max_gap_usd_per_twh_th": float(max(v - bench_val for v in gs_vals)),
    "mean_gap_usd_per_twh_th": float(np.mean([v - bench_val for v in gs_vals])),
}
```
**Not byte-identical**: both compute an exceedance share and a max gap
(same concept, different field name); the live version additionally
reports a MEAN gap, which `postprocess.py` does not; `postprocess.py`
additionally reports a MIN gap and the max shadow price outright,
neither of which the live script currently reports anywhere. **So this
module does hold two specific statistics (`min_gap`,
`max_shadow_price_usd_per_twh_th`) that are not currently computed by
anything wired** — a narrow, genuine gap, not a duplicate in every
respect, though trivially reproducible as a one-line addition to the
live script rather than requiring this module.

**Interface currency**: pure Python (`sum`/`min`/`max`/list
comprehensions), no imports beyond the builtins. Would run today,
unconditionally — it has no dependency that could go stale.

**Git history**: identical to `gas_balance.py` — created in the same
commit `9bcc8d7` (2026-02-19), never touched since, no module docstring
was ever added, no retirement language anywhere in its history.

---

### Dead-function sweep — completed, not partial

Part 1 flagged three dead functions on incidental discovery and noted
the sweep was not exhaustive. **Completed here**: every top-level
`def`/`class` in every LIVE (non-orphaned) `src/` module was checked
for total textual occurrences (its own definition plus every other
site in `src/`, `scripts/` incl. `deferred`/`retired`, `tests/`,
`notebooks/*.ipynb`) — a function occurring exactly once (its own `def`
line, nowhere else, not even called internally within its own file) is
dead code inside a live module.

**Correction to Part 1's framing, found while completing this sweep**:
`economics.discount_factor` is not, in fact, uncalled — it IS called,
but only internally, by `economics.npv()` (`economics.py:167`,
`cf * discount_factor(discount_rate, t)`). `npv()` itself has zero
callers anywhere (confirmed: no `npv(` call site outside its own
definition). So the precise statement is: **`economics.npv` is dead;
`economics.discount_factor` is reachable only through the also-dead
`npv`** — a two-function dead chain, not two independently dead
functions. `demand.apply_access_adjustment` (Part 1's other finding)
is confirmed unchanged: genuinely zero references anywhere, including
internally.

**Full list, genuinely dead (occurs nowhere but its own definition)**:

| module | function | notes |
|---|---|---|
| `blended_finance.py` | `implied_concessional_share` | zero references anywhere |
| `demand.py` | `apply_access_adjustment` | Part 1 finding, confirmed |
| `economics.py` | `npv` | Part 1 finding, corrected: calls `discount_factor` internally, itself uncalled |
| `io.py` | `load_yaml` | zero references anywhere |
| `io.py` | `save_results` | zero references anywhere |
| `optimize_experiments.py` | `run_annual_cap_case` | zero references anywhere |
| `optimize_experiments.py` | `identify_bottleneck` | zero references anywhere |
| `optimize_experiments.py` | `run_bottleneck_sensitivity` | zero references anywhere |
| `optimize_experiments.py` | `run_str1_storage_parameter_sweep` | zero references anywhere |
| `optimize_experiments.py` | `run_all_deterministic_scenarios` | zero references anywhere |
| `optimize_experiments.py` | `run_financing_vs_resource_test` | zero references anywhere |
| `solar.py` | `solar_capacity_trajectory` | zero references anywhere |
| `utils.py` | `interpolate_series` | zero references anywhere |
| `utils.py` | `mw_to_twh` | zero references anywhere |
| `utils.py` | `year_index` | zero references anywhere |

**14 dead functions found across live modules** (15 counting
`economics.discount_factor` as a distinct entry, though it is more
precisely a dependency of the dead `npv`, not independently dead).
Every one confirmed by direct grep across the full tree, not inferred
from the module-level importer check alone — a module being imported
does not mean every function inside it is reachable, which is exactly
what this sweep was for. Not flagged, and deliberately excluded from
this table: functions occurring more than once but ONLY within their
own file (`cli_arms._arm_specs`, `scenarios.land_scenarios`,
`gas_benchmarks.lng_netback_usd_mmbtu`, and others) — these ARE called,
just not exported to another file, and are not dead code.

---

## PART 2 — the two builders, capability by capability

Both `build_model()` (`src/optimize_model.py`, 1,989 lines) and
`build_model_sliced()` (`src/optimize_model_sliced.py`, 2,161 lines)
read in full, line by line, not grepped-and-guessed.

### 1. Temporal resolution

**Annual**: one time step per model year; energy is fully fungible
within a year (a solar MWh at noon "serves" the evening peak with
nothing to prevent it). **Sliced**: 6 slices per year, `{dry,wet} x
{night,day,peak}` (quoted, module docstring: *"dry = Nov-Apr (181 d),
wet = May-Oct (184 d)... night 22:00-05:00 (7h), day 05:00-18:00 (13h),
peak 18:00-22:00 (4h)"*). Gas dispatch, gas curtailment, unserved,
storage charge/discharge, hydro generation, and the electricity balance
are slice-indexed; capacity additions, all financial constraints
(bankability, subsidy, concessional envelope, public budget), land, and
build caps stay annual — capacity is built once a year, not per slice
(module docstring, quoted above Part 1). Gas fuel is an ANNUAL budget
allocated across slices by the optimiser, letting it save scarce gas for
the dry-season evening peak — *"real dispatch logic the annual model
cannot express"* (docstring, quoted). **This is the single biggest
capability divergence** and the reason every other item below differs
between the two builders in proportion to how much it depends on
intra-annual timing.

### 2. Gas: capacity, deliverability, new build, curtailment

Identical in both builders: cumulative capacity via `gas_add` (a free
decision Var since Phase 2.9, gated by `gas_build_case` ∈
{`gas_none` (fixed at 0), `gas_ndc_17gw` (capped to a sourced
trajectory), `gas_unconstrained`}); deliverability via
`gas_supply.gas_available_power(scenario_name=scenario["gas_scenario"])`;
`new_gas_deliverability` ∈ {`constrained`, `exempt`} controls whether
new capacity draws on the shared fuel budget or gets its own
capacity-and-efficiency-sized allowance. **Divergence**: in the annual
builder the capacity ceiling and fuel balance are both single annual
constraints; in the sliced builder the capacity ceiling is PER-SLICE
(`m.gas_generation[t,(s,p)] <= capacity x slice-hours x availability`)
while the fuel budget stays a single ANNUAL constraint summed over
slices — this is the mechanism that lets converted/unconverted gas be
allocated to the dry-season peak specifically, which the annual builder
cannot represent at all.

### 3. OCGT→CCGT conversion

**Does not exist in the annual builder — confirmed by its own raise**
(`optimize_model.py:295-302`, quoted):
> "scenario['conversion_available'] is True but build_model() (annual)
> does not implement OCGT-to-CCGT conversion -- it is slice-resolved
> (Phase 2.8, Decision 3) and lives in build_model_sliced() only. Pass
> conversion_case='conv_none', or build with build_model_sliced()."

**Exists only in the sliced builder**, gated by
`scenario["conversion_available"]`. Mechanism: `m.convert_mw[tier]` for
tiers A/B/C is a ONE-OFF (time-invariant) decision bounded by each
tier's nameplate; `m.convert_to_power[tier,t,(s,p)]` is the
slice-resolved thermal dispatch of the converted plant, entering the
per-slice electricity balance directly (required — the docstring notes
this is a "fourth amendment beyond Decision 4's named three; without it
the feature does not function"). Converted capacity runs at
`_eta_retrofit` (better than the fleet-average `eta`) and is priced with
its own emission factor (`_ef_e_retrofit = EF_TCO2_PER_MWH_TH /
_eta_retrofit`) — cleaner than unconverted gas, consistent with F-8's
"efficiency abates" finding. Converted OCGT nameplate LEAVES the
fleet-scalar gas-capacity bucket once converted (confirmed by comment at
`:843`). Financing: conversion capex is COMMERCIAL-ONLY (`m.convert_
comm_vol`, no concessional variable at all — same treatment as new gas
capex, for the same declared reason: DARES is a renewables-access
facility, not an asset-conversion facility) and is **excluded from the
public budget constraint** (`m.public_budget_constraint` sums only
public solar, storage, and EaaS subsidy in both builders — conversion
capex funds itself outside that envelope entirely, in the sliced
builder's F1-tranche mechanism).

### 4. Solar: public/EaaS channels, caps, land

Identical structure in both builders: `solar_public_add`/`solar_eaas_add`
decision Vars, cumulative capacity = baseline + sum of additions,
generation = capacity x (annual CF x 8760, annual builder) or
redistributed across day-slices-only at a season-adjusted yield (sliced
builder — *"zero output at night and in the evening peak"*, confirmed
by `_solar_energy_per_mw_by_slice()`). Land constraint (fixed intensity
coefficients per technology) and the NIRP cumulative deployment cap
(`solar_cumulative_cap_mw`) are annual in both, byte-identical
constraint bodies. EaaS solar priced at the SAME unit CAPEX as public
solar in both (see Part 1B's EaaS mechanism discussion — unchanged
between builders).

### 5. Storage: energy, power, duration, capacity credit

**Annual**: an annual energy-throughput accounting — charge bounded by
power x deployable-hours (a scenario parameter, default 700 h/yr),
discharge bounded by round-trip-efficiency-adjusted charge and by the
same power limit; the docstring notes explicitly that intra-annual
state of charge is NOT modelled (*"We do not model intra-annual SOC"*).
**Sliced**: storage charges in ANY slice and discharges within the SAME
SEASON (`m.storage_season_balance`, summed over the 3 slices per
season) — *"charge from midday solar in the day slice, discharge into
the evening peak of the same season"* (comment, quoted) — and
`storage_deployable_hours` is **no longer a free parameter**; throughput
falls out of the slice structure itself (per-slice power limits at
`storage_power_mw x slice-hours`). This is a genuine, not cosmetic,
divergence: the sliced builder's storage model is the reason storage can
have LP value at all (the annual model, per its own docstring's
"WHAT SLICING FIXES" section, makes storage "structurally incapable of
value" since there is no intra-annual time for it to shift energy
across). Capacity credit (for the firm-capacity constraint) uses
storage's POWER rating in both builders, identically.

### 6. Hydro: dispatch, seasonality, minimum flow

**Annual**: a purely exogenous fixed annual energy series
(`hydro_energy_twh_by_year`, from the required `hydro_case` arm),
injected directly into the electricity balance every year with **no
decision variable, no seasonal shape, no minimum-flow constraint at
all**. Its installed-MW rating (`hydro_capacity_mw_by_year`) is used
ONLY by the firm-capacity constraint. **Sliced**: hydro is a genuine
decision variable, `m.hydro_gen[t,(s,p)]`, under three constraints —
(i) `m.hydro_season_budget`: sum over each season's 3 slices equals
`hydro_annual[t] x hydro_season_share[season]` (seasonal shares SOURCED,
quoted earlier: *"two independent NERC quarterly datasets... dry
HIGHER"* — a correction of a previously RETIRED value whose *"SIGN was
wrong (dry LOWER, 0.90/1.10)"*, per the module's own comment); (ii) a
per-slice POWER limit tied to installed MW; (iii) `m.hydro_min_flow`, a
minimum-output floor (`hydro_min_frac`, default 0.50, ASSUMED not
SOURCED) proportional to each slice's share of its season's hours — *"
without it the LP concentrates all hydro into peak slices, which
reservoir operating obligations do not permit"* (comment, quoted). **This
is the second-largest capability divergence after conversion** — the
annual builder cannot represent hydro seasonality or a min-flow
obligation in any form.

### 7. Self-generation backstop and unserved energy

Identical mechanism in both builders: `m.selfgen` (unbounded,
Nigeria's self-generation stock is ~8x realised grid capacity) and
`m.unserved` (fixed to zero unless `allow_unserved=True`, default
auto-detected from whether a genset cost is priced in `econ`). Both are
sliced-indexed in the sliced builder, annual in the annual builder — the
mechanism and its declared rationale (quoted in both docstrings
verbatim, byte-identical prose) do not otherwise differ.

### 8. Firm capacity / reserve margin

Byte-identical formula and rationale in both builders (the sliced
builder's own comment says so explicitly: *"Mirrors optimize_model.py
exactly"*): `Σ_tech capacity_credit[tech] x cumulative_capacity[tech,t]
>= peak_mw[t] x within_block_peaking_factor x (1 + reserve_margin[t])`.
Storage's credited capacity is its POWER rating in both. No divergence.

### 9. Emissions: every quantity, and which is capped

Identical in both builders — three deliberately distinct quantities
(2.6, D3): `emissions_grid_only_by_year` (gas-fired grid generation —
the NDC commitment boundary), `emissions_selfgen_by_year` (private
diesel — in the national inventory, outside the gas-power slice),
`emissions_by_year` (always the sum — physical truth, what a carbon
PRICE sees and what gets reported). `emissions_capped_by_year` defaults
to grid-only; `selfgen_in_emissions_cap=True` widens it to total (the
declared alternative-boundary arm, not the central case). **One
sliced-only divergence**: under `conversion_available=True`, converted
generation is added to `emissions_grid_only_by_year` at its OWN,
better emission factor (`_ef_e_retrofit`) — a quantity that structurally
cannot exist in the annual builder since conversion itself cannot.

### 10. The grid-service obligation

Byte-identical formula in both (`m.grid_service_obligation`, bounding
`selfgen + unserved` against `max_offgrid_fraction x demand`, in
`"annual"` or `"total"` mode) — summed over slices in the sliced
builder, otherwise unchanged. The sliced builder explicitly ports the
`reliability_max_unserved_fraction`/`reliability_mode` deprecated-alias
translation "verbatim" from the annual builder, noting (quoted,
`optimize_model_sliced.py:250-256`) that an earlier audit (A15) found
this translation "silently absent" in the sliced builder at one point —
now fixed and cross-referenced so the two "cannot drift apart on this
translation a second time."

### 11. Carbon pricing

Identical in both: `carbon_active`/`carbon_price` from `carbon_policy_
scenarios()`, applied to `emissions_by_year` (the TOTAL, not the capped
quantity) when active. No divergence; zero in the central case
(`carbon_case="no_policy"`).

### 12. Financing: tranches, envelope, bankability, subsidy

Identical mechanism in both builders for solar/storage/gas: the
bankability constraint and subsidy-limit constraint (both quoted in
Part 1B's EaaS discussion) are byte-for-byte the same formula; the
`f1_tranche` per-channel CRF-annuity mechanism (concessional/commercial
volume split, `f1_concessional_envelope_constraint` spanning solar
public+EaaS and storage, gas and — sliced-only — conversion excluded as
commercial-only) is structurally identical where both builders have the
channel. **The one channel that only exists in the sliced builder**:
conversion's own commercial-only tranche (`convert_comm_vol`,
`m.cost_conversion_capital_charge_npv`, not shown in the excerpts above
but present per the diagnostic cross-reference at `:1968-1993`) — a
fifth financed channel the annual builder cannot have, for the same
reason it cannot have conversion at all.

### 13. O&M

Identical in both — both call the SAME shared function,
`io.compute_om_cost_npv`, from `scenario["om_cost_by_year"]` (the
REQUIRED `om_case` arm). The annual builder passes `gas_converted_
capacity_mw={t: 0.0}`/`gas_converted_generation_twh={t: 0.0}` for every
year (it has no conversion, so the converted-technology O&M terms are
always zero); the sliced builder passes the real converted-capacity/
generation series when `conversion_available=True`. Sharing one function
is deliberate (plan 2.5, quoted in both files' comments: *"the two must
share a cost definition or the annual-vs-sliced premium compares
different things"*) — this is the one cost component engineered NOT to
diverge even where a capability (conversion) does.

### 14. Salvage / end-effect treatment

Identical logic in both builders, present under BOTH
`capital_treatment_case` values: under `pre_f1_overnight`, an explicit
end-of-horizon straight-line residual credit (`m.salvage_npv`,
subtracted from the objective only, computed on raw capex) plus a
separate in-horizon rebuild-on-retirement mechanism
(`m.cost_replacement_capex_npv`) for any vintage whose technical life
ends before the horizon does. Under `f1_tranche`, truncation of each
tranche's CRF annuity AT the horizon IS the salvage treatment (no
separate residual credit exists under this arm — quoted, both files:
*"truncation IS the salvage treatment... there is no separate residual
credit or rebuild-on-retirement charge under this arm"*), with
in-horizon rebuilds recharged separately as a commercial-only tranche
(`m.cost_rebuild_capital_charge_npv`). No divergence between builders
beyond the sliced builder's conversion channel also participating in
the rebuild/tranche mechanism where the annual builder has nothing to
rebuild.

### 15. The objective, and every branch of it

Identical in both, term for term: `system_cost_npv` = gas OPEX + (either
the four `pre_f1_overnight` overnight-capex terms + replacement capex -
salvage, OR the single `f1_tranche` `capital_charge_total_npv`) +
self-gen cost + unserved (VoLL) cost + carbon cost + O&M — **always
unconditionally minimised**, no other branch exists. Confirmed against
both files' own comments that the term ORDER is deliberately preserved
identically to the pre-refactor expression (bit-exact reproduction,
since floating-point addition is not strictly associative).

---

### Removed capabilities

**1. The weighted-sum objective (`weight_cost`/`weight_emissions`) —
confirmed deleted, from both builders, per Plan 0.4 Item B
(`docs/numerical_hygiene_0_4.md`).** Old objective assembly (quoted from
that doc, itself quoting the pre-deletion code):
```python
def build_model(scenario, econ, weight_cost=1.0, weight_emissions=0.0, ...):
    if (emissions_cap is None and emissions_cap_by_year is None
            and (weight_cost == 0.0 and weight_emissions == 0.0)):
        raise ValueError(...)
    ...
    else:
        m.objective = pyo.Objective(
            expr=weight_cost * system_cost_npv + weight_emissions * m.emissions,
            sense=pyo.minimize)
```
Now: `m.objective = pyo.Objective(expr=system_cost_npv, sense=pyo.minimize)`
unconditionally, in both builders — no residual trace of either
parameter name anywhere in `src/` or `scripts/` (grep confirms zero
hits outside the two builders' own explanatory comments). The removal
task itself AST-parsed every call site in the repository to confirm no
positional call could have silently broken (all calls to
`build_model`/`build_model_sliced` use keyword arguments exclusively) —
documented, not merely asserted.

**2. The pre-F1 concessional capital envelope
(`concessional_envelope_npv`/`m.concessional_drawdown_npv`/
`m.concessional_envelope_constraint`) — retired, both builders,
identical comment block** (quoted, `optimize_model.py:727-737` /
`optimize_model_sliced.py:1095-1103`):
> "CONCESSIONAL CAPITAL ENVELOPE (scarcity constraint) -- RETIRED. The
> pre-existing NPV-discounted, EaaS-solar-only mechanism... is retired
> per envelope resolution Decision 2 (docs/f1_implementation.md,
> "Envelope resolution"): one envelope, one quantity. F1's OWN
> concessional envelope -- undiscounted commitment stock, spanning solar
> (public + EaaS) and storage, gas excluded -- replaces it."

Superseded by `m.f1_concessional_envelope_constraint` (the `f1_tranche`
mechanism, item 12 above). **Checked for stale description**: grepped
`docs/methodology.md`, `_protocols.md`, `_physical_layer.md`,
`_genself_backstop.md`, `_demand_horizon.md` for
`concessional_envelope_npv`/`concessional_drawdown_npv`/
`concessional_envelope_constraint` — one hit
(`docs/methodology.md:128`), and it cites the retired function name only
as a historical naming-mistake example (*"added after finding a
function named `concessional_envelope_npv()` that applied no
discounting"*), not as a description of a live mechanism. **No stale
description found.**

**3. A convention change, not quite a capability removal but adjacent —
the `1e18` "uncapped" sentinel replaced with `None` (Plan 0.4, Item
C).** Seventeen call sites across `scripts/` and `src/optimize_
experiments.py` changed `emissions_cap=1e18` to `emissions_cap=None`
(full site table in `docs/numerical_hygiene_0_4.md`); two dead
sentinel-comparison code paths deleted as a consequence (both already
unconditionally `None`-producing before the change, confirmed
behaviour-preserving). Noted here because **`tests/test_run.py`
(Addition 4, below) still uses the OLD `1e18` convention** —
`emissions_caps=[1e18]` — which no longer matches the interface's
current idiom, though it would not by itself cause the test's failure
(that failure is the missing function, addressed below).

**4. A minor API-surface narrowing, not a capability loss**:
`gas_new_efficiency` "is no longer a free, independently-settable kwarg"
(2.13, Step 2) in either builder — it is now always derived from the
required `ccgt_efficiency_case` arm. The underlying capability (setting
new-build gas efficiency) is unchanged; only the input path narrowed
from a free parameter to a named, sourced arm.

**No other removed capability found** after grepping both builder files
for `RETIRED`/`SUPERSEDED`/`removed`/`no longer`/`DELETED` in full
(complete list of hits checked; the four above are the only ones that
describe an actual mechanism change rather than a comment about
something else, e.g. a solver flag or a diagnostic guard).

---

### Part 2, addition 1 — `stochastic.py`: a runner away, not a rewrite

Confirmed against the full builder read in this Part 2 pass (not just
Part 1B's spot-check): `stochastic.py` imports and calls **only**
`optimize_model.build_model` — the ANNUAL builder — never `optimize_
model_sliced.build_model_sliced`. Nothing found in this pass changes
Part 1B's conclusion. Restating the checks precisely, now cross-
referenced against this Part 2's own full read of `build_model`:

- `fixed_plan` parameter exists in `build_model`'s signature (`:39`) and
  fixes exactly the three keys `stochastic.py` supplies
  (`solar_public_add`, `solar_eaas_add`, `storage_add`, `:476-481`) —
  confirmed again in this pass, same conclusion.
- `scenario["demand_growth"]`, `scenario["gas_scenario"]` are read
  directly by `build_model` (`:579`, `:587`) exactly as
  `load_scenario()` writes them and exactly as `stochastic.py` perturbs
  them.
- `scenario["carbon_active"]`/`["carbon_price"]` are read at `:1619-
  1620`, matching `carbon_policy_scenarios()`'s keys exactly.
- `gas_probability_weights()` exists at HEAD (`scenarios.py:1032`),
  unchanged.

**Nothing in this full-builder read surfaces a NEW required argument
`stochastic.py` fails to supply.** `build_model` does require several
OTHER scenario keys `stochastic.py`'s caller must ensure are present
(the 17 sentinel-guarded arms, `peak_mw`, `capacity_credit`,
`hydro_energy_twh_by_year`, `asset_lifetimes`, etc.) — but these are
exactly what a scenario dict built via a normal `load_scenario()` call
already contains; `stochastic.py` takes an already-built `base_scenario`
as input and never constructs one itself, so this is the CALLER's
responsibility (writing the missing runner script), not a gap in
`stochastic.py` or in `build_model`'s interface. **Verdict unchanged
from Part 1B: a runner away.** What "a runner away" concretely requires:
a new `scripts/XX_run_stochastic.py` that (1) calls `load_scenario()` +
`load_econ()` to build a valid base scenario/econ pair through the
standard arm surface (`src/cli_arms.py`, same discipline as every other
runner), (2) calls `run_stochastic_simulation(base_scenario, econ,
carbon_mu, carbon_sigma, solar_capex_tv, caps=..., N=..., seed=...)`,
(3) calls `compute_risk_metrics()` on the returned outcomes, (4) writes
results. **The one scope caveat, unchanged from Part 1B**: this would
produce ANNUAL-builder Monte Carlo results only — `stochastic.py` has
no sliced-builder path at all (no import, no call, nothing to port) — a
methods-chapter caveat about scope, not an interface break.

### Part 2, addition 2 — `run_deterministic_scenario`: a legacy capability, by its own admission

Read in full this pass (`src/optimize_experiments.py:246-336`). **Its own
docstring states its status outright, quoted in full**:
> "LEGACY NON-LP SIMULATOR — NOT for thesis result tables. This
> evaluates a HAND-SUPPLIED capacity path with must-run gas, undiscounted
> costs, and the fixed 2025 solar CAPEX scalar. Its cost basis is
> therefore NOT comparable to build_model()'s discounted NPV. It is
> retained only for quick feasibility sketches. Any headline cost/
> emissions/reliability result must come from build_model()/extract_
> planning_diagnostics(), not from here. **The Monte Carlo path no
> longer uses this function** (see stochastic.py, which now uses
> fixed-plan re-dispatch through build_model())."

**What it CAN represent**: a demand projection, a gas-availability
profile times a fixed efficiency, solar generation from a hand-supplied
MW path (NOT an LP decision), a single `BatteryStorage` instance
(non-optimised, fixed capacity from the supplied path), a greedy annual
dispatch (`dispatch_energy`), and simple additive costs (gas, solar
CAPEX at a flat scalar, unserved-energy penalty, a flat carbon price if
active) — **UNDISCOUNTED**.

**What it CANNOT represent, relative to either Pyomo builder**: any
optimisation at all (capacity is an INPUT, `capacity_paths`, never a
decision variable — there is no LP, no objective, no solver call in
this function); OCGT→CCGT conversion; EaaS bankability or any tranche/
concessional financing mechanism (no `financing_regime`, no
`f1_tranche`, no subsidy); the emissions cap (there is no `emissions_
cap`/`emissions_cap_by_year` argument at all — only an optional carbon
PRICE); the grid-service obligation (no `max_offgrid_fraction` concept
anywhere in the function); the firm-capacity/reserve-margin constraint;
land constraints; salvage or end-of-horizon treatment (costs are
undiscounted, so NPV-based salvage has no meaning here). **This
function directly confirms, in its own words, that stochastic.py's own
docstring's parenthetical — "This supersedes the earlier approach that
routed draws through a separate undiscounted simulator" — refers to
this exact function.** The two files corroborate each other's account
independently.

**`results/eaas/` currency, checked directly, not assumed**:
`diagnostics.json`/`summary.json`/`timeseries.csv` are ALL dated **2026-
08-23 13:44** (`ls -la`) — the SAME stale timestamp already on record
elsewhere in this project (register L61, the bare `diagnostics.json`
dependency). Compared against git history: `scripts/02_run_eaas.py`
(the runner that calls `run_deterministic_scenario` and produced this
output) was substantively rewritten on **2026-09-18** (`0012f16`,
"Revive nine dead runners: 02_run_eaas, ..."); `src/optimize_model.py`
(the Pyomo builder `02_run_eaas.py` ALSO calls, for its LP-side
results) was touched as recently as **2026-09-25** (`e46be32`,
"Bankability test priced on the commercial basis, not the naira-basis
blend" — a substantive change to the EXACT bankability mechanism this
run's own `diagnostics.json` reports on: its keys include
`bankability_threshold`, `subsidy_per_mw_usd_by_year`).
**`results/eaas/` is not current at HEAD** — it predates both a
substantive rewrite of the script that produced it and a substantive
change to the bankability pricing basis its own diagnostics report on.
Stated plainly, not hedged: anything read from `results/eaas/` today
would not reproduce under the current code.

### Part 2, addition 3 — removed-capabilities cross-check: complete

Covered fully in "Removed capabilities" above. Summary: weighted-sum
objective (confirmed deleted, comment quoted, zero residual trace);
concessional envelope (confirmed retired, comment quoted, no stale
description found in five methodology documents checked); the `1e18`→
`None` convention change (documented, one live inconsistency found —
`tests/test_run.py` still uses the old convention, see Addition 4); the
`gas_new_efficiency` kwarg narrowing (minor, not a capability loss). No
document was found describing either the weighted-sum objective or the
old concessional-envelope mechanism as though it were still live.

### Part 2, addition 4 — `tests/test_run.py`: broken since 2026-04-03, confirmed by name

Full file read (46 lines, quoted in relevant part above). **What
`generate_epsilon_pareto` was meant to be**: an ε-constraint Pareto
generator taking `(scenario, econ, emissions_caps)` and returning a
`results` list, one entry per swept cap, each with `decision_variables`
and `diagnostics` sub-dicts — the test calls it with a single,
enormous cap (`emissions_caps=[1e18]`, the OLD "uncapped" sentinel,
see Removed capability 3 above) as a sanity/regression check, then
prints (does not assert) five values: solar additions by year, two
storage-capacity readings under different key names
(`final_storage_capacity_mwh` and `storage_capacity_mwh` — printed
side by side, suggesting the test itself was checking for a rename
in progress), 2025 unserved TWh, and 2025 gas-to-power TWh_th.

**What the test asserts**: nothing. Every check is a `print()`, not an
`assert`. Even with a working import, this file is a manual-inspection
script, not an automated test in any enforceable sense — running it
under `pytest` would pass or fail only on whether it raises an
exception, never on whether any printed value is correct.

**Whether an equivalent exists under another name — confirmed
precisely, not inferred**: `git log --all -S"generate_epsilon_pareto"
--oneline -- src/optimize_experiments.py` returns exactly two commits —
`e7bae45` (added) and `96b50cf`, 2026-04-03, "large refactor with
results" (removed). Reading that commit's diff directly: `def
generate_epsilon_pareto(scenario, econ, emissions_caps):` is replaced,
AT THE SAME LOCATION in the file (old line 1658 → new line 1659), by
`def run_gas_regime_ndc_matrix(...)` — the same refactor also replaces
`generate_weighted_pareto` with `run_rel1_feasibility_matrix` at the
adjacent position. **`run_gas_regime_ndc_matrix` is the direct
successor** (and is, per Part 1, still live today — imported by
`06_run_gas1_shadow_benchmarks.py`, the runner behind this session's
F-30/GAS-1 re-verification work). `tests/test_run.py` was never updated
to follow the rename. **It has been broken since 2026-04-03 — nearly
six months, not a recent regression** — and even a fixed import would
only restore a print-only script with no pass/fail criterion.

---

## PART 3 — the arm system, complete

`src/scenarios.py` read in full (3,465 lines). Of its 36 top-level
`def`s, **23 return arm-value dictionaries** consumed by
`load_scenario()` via a `*_case` parameter. The rest are plumbing,
excluded from the arm count for the reason stated: `planning_horizon`
(computes the year index, not a case dict), `demand_growth_benchmark`
and `demand_growth_prior` (external benchmarks / a Monte Carlo prior
over the SAME arms — explicitly "NOT scenario arms," quoted from
`demand_growth_benchmark`'s own docstring, kept out of the registry so
consumers iterating it are unaffected), `nirp_solar_cumulative_mw` (a
data table `solar_build_scenarios()`'s multipliers are applied to, not
itself a case dict), `asset_lifetimes`/`asset_lifetime_sweep` (a
by-technology lookup table and its sensitivity band, not a `*_case`
selector — no `asset_lifetime_case` parameter exists in
`load_scenario()`), `solar_min_build_default` (a bare float, no case
selection), `gas_probability_weights` (a Monte Carlo prior, consumed
only by the orphaned `stochastic.py`, Part 1B), `_load_gas_plant_vintages`/
`gas_conversion_tiers`/`gas_vintage_retirement_series_mw`/
`gas_ndc_17gw_trajectory_mw`/`hydro_trajectory`/`_solar_cumulative_cap`
(private or public HELPER functions that derive a time series from an
arm already selected elsewhere, not arms themselves), and `load_scenario`
itself (the assembler).

### Arm table — 23 arms, every value, source status, exercise status

**Method for the "exercised" column**: grepped every named value across
`results/` (all subdirectories) for its literal string. Many runners
embed the FULL arm-tag string in output filenames or in a JSON field
(e.g. `...__ccgt_eff_central__cr_central__dscr_central__f1_t...`,
confirmed by direct inspection of hit files) — a hit in this scheme is
high-confidence evidence of exercise. **But three arms
(`bank_tenor_case`, `gas_price_case`, `voll_case`) returned ZERO hits
for every one of their values, including the DEFAULT/most-common
choice** — confirmed by checking a sample `diagnostics.json`'s own key
list directly: it records derived dollar/rate quantities, not the
originating arm-name string, for these three. **A zero hit for these
three arms means "not part of this tag/field scheme," not "never run"**
— genuinely indeterminate by this method, stated plainly rather than
guessed at. For every other arm, a zero hit is treated as reasonably
confident evidence of non-exercise, EXCEPT where the value's own code
raises a `ValueError` if selected at all (`peak_nirp`, `gas_avail_central`)
— those are certain-never-exercised by construction, not by absence of
evidence.

| arm (`*_case` parameter) | values | meaning (condensed; sourcing quoted where the file states it) | source status | exercised? |
|---|---|---|---|---|
| `demand_level_case` | `served` (37.09 TWh) | Tier 1: "observed GROSS GENERATION at busbar" [NERC 2024]. Not a bound; measured. | SOURCED | yes (1,953 files) |
| | `latent_low` (38.47), `latent_high` (76.93) | DEPRECATED — raise in `load_scenario`; retained "for provenance only" | marked DEPRECATED, unsourced basis (built as multiples of a since-corrected base) | present in 5 files each, but these are references to the deprecation itself (raise messages), not live runs — cannot raise and also produce output |
| `demand_case` | `constrained_continuation` (0.0314) | Ember 2010-2025 grid-generation CAGR — "a CONSTRAINT-PERSISTENCE arm... NOT organic demand growth" | SOURCED | yes (4) |
| | `organic_central` (0.039) [CENTRAL] | Route A (Liddle & Hasanov 2023, elasticity 0.80) + Route B (Kenya per-capita convergence); GDP 4.5% is "a DECLARED ASSUMPTION" | SOURCED method, ASSUMED GDP input | yes (4) |
| | `organic_high` (0.060) | Route A (eps 0.90, GDP 6.8%) / Route B (Ghana convergence) / Route C (NIRP low case) | SOURCED method, ASSUMED GDP input | yes (4) |
| `peak_multiple_case` | `peak_suppressed` | 2016 Oluwole reconstruction, rationing-flattened shape — "DECLARED FLOOR, NOT a candidate central case" | SOURCED (shape); vintage-declared | yes (37) |
| | `peak_unsuppressed` [CENTRAL] | 2016 Oluwole reconstruction, unsuppressed | SOURCED (shape, 2016 vintage) | yes (1,038) |
| | `peak_nirp` | `None` — "raises if selected, by design"; NIRP gives a load factor but no daily profile | SOURCE: [PENDING] | **never — cannot be, raises unconditionally** |
| | `peak_high` | Evening factor pinned at 1.20, "CONSTRUCTED upper bound, not measured" | ASSUMED/CONSTRUCTED (declared) | yes (27) |
| `reserve_margin` | float, or literal `"nirp_spinning"` | NIRP 2024 §2.3 p.19, 900 MW fixed spinning-reserve requirement, converted to a shrinking fractional margin | SOURCED | not separately checked (not a `*_case` dict; a plain float/string param) |
| `solar_build_case` | `nirp_trajectory` (1.0×) [effectively CENTRAL] | NIRP's own schedule; "REPORT UNSERVED ENERGY AND FEASIBILITY FROM THIS ARM, NOT NPV" | SOURCED (NIRP Annex F / Table 8) | yes (3) |
| | `nirp_accelerated` (2.0×) | "[SOURCE NEEDED] for the 2.0x... currently a declared assumption" | stated ASSUMED, explicitly flagged `[SOURCE NEEDED]` | **0 hits — not found** |
| | `deployment_unconstrained` (`None`) | "the counterfactual that isolates what the deployment limit itself costs" | n/a (no cap) | yes (4) |
| `carbon_case` | `no_policy` [CENTRAL] | `carbon_active=False`, price 0.0 | n/a (policy stance) | yes (356) |
| | `moderate_policy` (50 USD/tCO2), `stringent_policy` (75 USD/tCO2) | no sourcing note given for either price in the docstring — unmarked | **unmarked — not classified in-file** | **0 hits each — not found** |
| `gas_deliverability_case` | `downside`/`baseline`/`upside`/`shock_recovery` (structural) | must match `data/gas/processed/gas_available_power_annual_twh_th.csv`'s `scenario` column | data-driven (CSV); shapes not classified in scenarios.py itself | yes (65–966 files each) |
| | `flat_downside`/`flat_upside`/`flat_shock_recovery` | "Level-equivalent flat controls (GAS-3 only)... to isolate SHAPE vs LEVEL" | data-driven (CSV) | yes (3 each) |
| | `gas_deliv_flat`/`gas_deliv_gmp_partial`/`gas_deliv_gmp_full` | GMP 2026 policy-target family, "planned increment: 0.0 / 0.50 / 1.0" | data-driven (CSV / `gas_deliverability_cases.yaml`) | yes (24–92 each) |
| `land_case` | `tight` (1,000 km²), `moderate` (3,000), `loose` (6,000) | "Land availability envelopes" — no source cited in-file | **unmarked — not classified in-file** | tight/moderate yes (175 each); loose yes (4) |
| `capital_case` | `tight` (0.50×B\*), `moderate` (0.85×), `adequacy` (1.00×), `expansion` (1.20×), `unconstrained` (`None`) | B\* = 10,273,343,610 USD, "Derived value... Source: `28_run_capital_frontier.py`, run 2026-09-20" — superseded an undocumented constant that understated the requirement by 12.84% (L45) | SOURCED (B\* itself is a model-internal derived quantity, declared as such — see `docs/parameter_provenance.md` row 10) | all yes (175–1108 depending on tier) |
| `tariff_case` | `tariff_low` (75M), `tariff_central` (95M) [CENTRAL], `tariff_high` (115M) | Nigeria's 2016 NBET solar IPP negotiation: 115 = NBET's agreed rate; 75 = FGN counter-position; 95 = "the MIDPOINT... A CONSTRUCTION, not an independently sourced figure" | SOURCED (low/high); central is a DECLARED construction | yes (11/324/11) |
| | `tariff_legacy_95m`, `tariff_legacy_65m` | "RETAINED FOR ATTRIBUTION ONLY, both UNSOURCED" | explicitly UNSOURCED, kept for attribution | yes (12 each) |
| `om_case` | `om_none` (synthetic, all-zero regression arm) | forces every O&M rate to 0.0, "both builders' O&M term is identically 0.0" | n/a (synthetic control) | yes (4) |
| | `om_low`/`om_central`[CENTRAL]/`om_high` | NREL ATB 2024 Advanced/Moderate/Conservative cost cases, data-driven from `data/cost/processed/om_cost.csv` | SOURCED (NREL ATB) | `om_central` yes (219); `om_high` yes (8); **`om_low` 0 hits — not found** |
| `storage_cc_case` | `storage_cc_low` (0.50), `storage_cc_central` (0.80) [CENTRAL], `storage_cc_high` (1.00) | "SOURCE: DECLARED JUDGEMENT, not measured" — central "matches gas's availability-factor credit" | DECLARED JUDGEMENT (explicit) | low/high yes (4 each); central yes (1,068) |
| `gas_availability_case` | `gas_avail_observed` (0.3061; was 0.295 until TASK 73) | NERC 2024 (3,351.16/10,947 MW, the same 23 plants). "DECLARED LOWER BOUND — double-counts fuel" | SOURCED (NERC), with a stated double-counting caveat | yes (24) |
| | `gas_avail_central` (`None`) | "SOURCE: [PENDING]... Raises if selected until sourced" | SOURCE: [PENDING] | **never — cannot be, raises unconditionally** (6 files hit the STRING "gas_avail_central" but these are the code's own docstring/error text appearing in log output, not a completed run using it — confirmed by the arm's own raise) |
| | `gas_avail_nameplate` (0.80) | "the pre-2.10 unsourced assumption, retained as an explicit UPPER BOUND" | explicitly UNSOURCED, declared as an upper bound | yes (1,040) |
| `conversion_case` | `conv_none` (regression) | `available=False`; reproduces pre-2.8 results; annual builder raises on any other value | n/a (control) | yes (174) |
| | `conv_central` [CENTRAL] (600 USD/kW, 0.50 uplift, 0.46 eff) | "DERIVED, not SOURCED... greenfield CC-minus-SC increment" (Gas Turbine World 2024) + a declared brownfield premium | DERIVED (method sourced, premium ASSUMED) | yes (734) |
| | `conv_low` (850/0.40/0.42), `conv_high` (450/0.50/0.50) | same method, bracketing the premium/efficiency/uplift assumptions | DERIVED/ASSUMED, same basis | yes (27 each) |
| `gas_retirement_case` | `gas_retire_none` (0 MW/yr) | "a DECLARED UPPER BOUND on the surviving fleet, not a forecast" | DECLARED (explicit) | yes (24) |
| | `gas_retire_proxy` (680 MW/yr from 2035) | "the pre-2.11 'brownfield retirement proxy'... SOURCE: [PENDING]" | SOURCE: [PENDING] | yes (4) — exercised despite being unsourced |
| | `gas_retire_vintage_30` [CENTRAL] (30 yr) | "SOURCE: ASSUMED — analyst vintage estimate, not traced to a Nigerian source"; central because two opposite errors partly cancel at this figure | ASSUMED (explicit) | yes (992) |
| | `gas_retire_vintage_40` (40 yr), `gas_retire_vintage_50` (50 yr) | same ASSUMED basis; the 40-yr arm retires two currently-operating plants (Delta_1, Egbin_1), "reported as a limitation, not corrected" | ASSUMED (explicit) | yes (20/24) |
| `gas_build_case` (`GAS_BUILD_CASE_CHOICES`, not a `_scenarios()` dict) | `gas_ndc_17gw` [intended CENTRAL] | NDC 3.0's 17 GW-by-2035 target, on a NIRP-derived trajectory — "DERIVED, not stated by NIRP" | DERIVED | yes (4) |
| | `gas_unconstrained` | no cap; cost-optimal build | n/a | yes (1,028) |
| | `gas_none` | fixed at 0; "Reproduces every pre-2.9 result" | n/a (control) | yes (32) |
| `hydro_case` | `hydro_flat` [REGRESSION ARM] (11.47 TWh flat) | NERC 2024, 30.92% of gross generation, held flat | SOURCED | yes (826) |
| | `hydro_committed` (~12.8 TWh) | "SOURCED 2025 figure... 'held flat after 2025' is DECLARED" | SOURCED level, DECLARED shape | yes (23) |
| | `hydro_ndc` (→10,400 MW by 2030) | NDC's sourced 2030 target; "linear interpolation SHAPE and the post-2030 flat treatment are DECLARED, not sourced" | SOURCED anchor, DECLARED shape | yes (27) |
| `gas_capex_case` | `gas_capex_low` (900,000 USD/MW) | "the pre-2.13 bare literal... never sourced" — retained as an explicit lower bound | explicitly UNSOURCED (declared as such) | **0 hits — not found** |
| | `gas_capex_central` [CENTRAL] (1,317,000) | DERIVED from NIRP 2024 Tables 19/22/23, method cross-validated against NIRP's own solar figures | DERIVED (reproducible arithmetic quoted in-file) | yes (1,036) |
| | `gas_capex_high` (1,400,000) | upper end of the same NIRP-derived year-by-year cluster | DERIVED | **0 hits — not found** |
| `gas_vintage_dispatch` (arm, TASK 74-77; register L134) | `True` (default; `False` under exempt deliverability) | A modelling choice: capacity added in the model dispatches at `gas_new_efficiency` (the `ccgt_efficiency_case` value), pre-existing at `gas_eta`; emissions on the thermal draw; tightened legacy-stream row; both builders | DECLARED (not sourced) | yes (G7-2, and every runner's key) |
| `ccgt_efficiency_case` | `ccgt_eff_iso` (0.5445) | EIA / Sargent & Lundy SL-018001 Rev A, Table 5-1, CC 2x2x1 H-class, net HHV, 60 F / 60% RH, no derate (was cited as NREL ATB; corrected TASK 73) | SOURCED (S&L) | **0 hits — not found** |
| | `ccgt_eff_central` [CENTRAL] (0.5173) | S&L base figure (cited as ATB until TASK 73) with "a 5% Nigerian ambient derate... a DECLARED ASSUMPTION, NOT from S&L" | SOURCED base, ASSUMED derate | yes (1,036) |
| | `ccgt_eff_low` (0.5010) | S&L base figure (cited as ATB until TASK 73) with an 8% derate, "a declared assumption, not from S&L" | SOURCED base, ASSUMED derate | **0 hits — not found** |
| `capital_recovery_case` | `cr_central` [CENTRAL] (r_conc 2.12%/25yr, r_comm 11%/10yr) | r_conc: World Bank IDA21 terms, SOURCED; r_comm: "SOURCED BASE + ASSUMED UPLIFT" (IEA Cost of Capital Observatory base, +2-2.5pp uplift "AN ANALYST ESTIMATE, NOT SOURCED") | mixed, precisely stated per-component | yes (1,000) |
| | `cr_ida21` (2.13%/25yr, same r_comm) | "NEARLY IDENTICAL to cr_central... differing only by which of IDA21's two 2026 effective dates" | SOURCED | yes (27) |
| | `cr_comm_low`/`cr_comm_high` (2.80% IDA20/30yr, r_comm 9%/14%) | pre-Part-2 IDA20 rate/tenor, unchanged; r_comm brackets the same assumed uplift | mixed (r_conc SOURCED-superseded-vintage; r_comm ASSUMED) | yes (35 each) |
| `dscr_case` | `dscr_regression` (1.10) | "reproduces the pre-F1 commercial-arm value exactly" | n/a (regression control) | yes (201) |
| | `dscr_central` [CENTRAL] (1.30), `dscr_low` (1.20), `dscr_high` (1.40) | no explicit source/assumed marker given for the three named levels themselves | **unmarked — not classified in-file** | yes (799/23/23) |
| `concessional_envelope_case` | `env_dares` [CENTRAL] (750M) | "STIPULATED, not SOURCED, within this repository"; external corroboration exists (World Bank DARES, $750M IDA credit, Dec 2023) but is not traced inside the repo | STIPULATED (external corroboration noted, not itself verified in-repo) | yes (963) |
| | `env_dares_plus` (1,050M) | "The 1,050M figure has no external corroboration and remains STIPULATED" | STIPULATED, weaker than `env_dares` | yes (23) |
| | `env_none` (0), `env_5x`/`env_10x` (3,750M/7,500M) | declared counterfactuals — "do NOT claim DARES itself is 5x or 10x its stated size" | DECLARED COUNTERFACTUAL (explicit) | yes (35/27/27) |
| `capital_treatment_case` (`CAPITAL_TREATMENT_CASE_CHOICES`) | `pre_f1_overnight` | reproduces the pre-F1 objective exactly (four overnight-capex terms + salvage + replacement) | n/a (structural choice) | yes (66) |
| | `f1_tranche` [CENTRAL] | per-tranche CRF annuity, replaces all six terms | n/a (structural choice) | yes (1,023) |
| `bank_tenor_case` (`BANK_TENOR_CASE_CHOICES`) | `bank_tenor_comm` [NEW DEFAULT] | objective's own commercial tranche, truncated at a real tenor — corrects a prior mismatch (bankability priced at 18% naira-basis, no tenor, vs. the objective's 11%/10yr) | n/a (structural correction) | **indeterminate — see method note above** |
| | `bank_tenor_legacy` | "the pre-correction blended-WACC factor... undiscriminated by any tenor. Every result produced before this task used this basis" | n/a | **indeterminate — see method note above** |
| `blend_case` (`blended_finance_scenarios`) | `commercial_only` (0%), `blended_central` [ADOPTED] (30%), `blended_optimistic` (50%), `concessional_heavy` (75%) | concessional-share tuples feeding the bankability blended WACC; `r_concessional` SOURCED, `r_commercial`/`concessional_share` "remain PLACEHOLDERS pending primary-source verification" | mixed — r_conc SOURCED, r_comm/share explicitly flagged PLACEHOLDER | yes (6 each) |
| `new_gas_deliverability` (plain default, not a `_scenarios()` dict) | `constrained` [DEFAULT] | new gas capacity shares the existing fleet's fuel budget | n/a (structural choice) | yes (1,108) |
| | `exempt` | new capacity gets its own fuel-equivalent allowance, sized off `ccgt_efficiency_case` | n/a | **0 hits — not found** |

### `io.py`'s arms

- **`voll_case`** (`load_econ`, default `"voll_low"`): values
  `voll_low`/`voll_mid`/`voll_high`, data-driven from
  `data/cost/processed/unserved_energy_penalty.csv`'s `scenario` column
  (year-2025 row read at `io.py:424-430`) — the SET of values is
  data-driven, not enumerable from `io.py`'s own source without reading
  the CSV. Exercise status: indeterminate (see method note above);
  `voll_mid` alone showed 497 `results/` hits on grep, but that count
  cannot be trusted to mean the OTHER two were never run, only that
  `voll_mid`'s string happens to coincide with some other field/tag —
  not independently confirmed.
- **`gas_price_case`** (`load_econ`, default `"gas_low"`): values
  `gas_low` (7.27M USD/TWh_th, no transport tariff), `gas_mid` (11.01M,
  domestic tariff), `gas_high` (14.87M, export-parity) — figures quoted
  directly from the docstring, data-driven from
  `data/cost/processed/gas_cost.csv`'s `Scenario` column. Exercise
  status: indeterminate, zero hits for all three by the tag-string
  method (see method note).
- **`genset_case`** (`load_econ`, REQUIRED — no default, confirmed
  already in `docs/task_log_ch36_inputs.md` item B this session):
  values `genset_srmc`/`genset_central`[CENTRAL]/`genset_high`, or an
  explicit `None` ("run without the self-generation backstop... the
  regression harness... and the 'voll' arm of
  `23_run_selfgen_frontier.py` both rely on it"), data-driven from
  `data/cost/processed/genset_cost.csv`. Exercised: `genset_srmc` (3),
  `genset_central` (46), `genset_high` (3) — all confirmed present.
- **`om_case`** handling: already covered in the main arm table
  (`om_scenarios()` is defined in `scenarios.py`, but the actual cost
  data it names is loaded by `io.load_om_cost_by_year`/
  `compute_om_cost_npv` from `data/cost/processed/om_cost.csv`) —
  included here for completeness since the question spans both files.

### Non-arm modelling choices — hardcoded constants, `.get()` defaults, in-function literals

`docs/chapters/chapter3_1_v2.md` [removed `009d230`; held at `a993f15`; successor `CHAPTER_3_assembled.md` §3.1] §3.1.7 lists constraint EQUATIONS, not
hardcoded constants as such, and names `within_block_peaking_factor` as
a live example of the pattern this section extends.
`docs/chapters/chapter3_3_draft.md` [removed `009d230`; held at `a993f15`; successor `CHAPTER_3_assembled.md` §3.3] §3.3.7, quoted in full, already
lists five:

> "`new_gas_deliverability` (`"constrained"`, both builders);
> `conversion_available` (`False`, sliced builder); `hydro_season_factor`
> (dry 1.045 / wet 0.956, sliced builder); `hydro_min_flow_fraction`
> (0.50, sliced builder); `storage_duration_hours` (4.0, `load_scenario`)."

**Grepped every `.get(` call with a literal (non-`None`) fallback in
both builders and `scenarios.py`, plus every bare module-level numeric
constant, to find what §3.3.7 does NOT yet list. Complete set below,
§3.3.7's five marked accordingly:**

| parameter | value | where | already in §3.3.7? |
|---|---|---|---|
| `new_gas_deliverability` | `"constrained"` | both builders | yes |
| `conversion_available` | `False` | sliced builder | yes |
| `hydro_season_factor` | dry 1.045 / wet 0.956 | sliced builder (`DEFAULT_HYDRO_SEASON_FACTOR`) | yes |
| `hydro_min_flow_fraction` | 0.50 | sliced builder (`DEFAULT_HYDRO_MIN_FLOW_FRACTION`) | yes |
| `storage_duration_hours` | 4.0 | `load_scenario` | yes |
| **`within_block_peaking_factor`** | **1.3448** | both builders, `optimize_model.py:1041`/`optimize_model_sliced.py:1265` | **NEW — named in §3.1.7's prose as the motivating example, but its own value is not tabulated in §3.3.7** — comment at the call site: "Set scenario[\"within_block_peaking_factor\"] once sourced," i.e. SOURCE: [PENDING], currently silently defaulted on every run that doesn't set it explicitly |
| **`social_discount_rate`** | **0.08** | both builders, `r_social = float(scenario.get("social_discount_rate", 0.08))` | **NEW** — this is the widely-discussed $r_s$=8% (register/Chapter 3.1.2's own stated open item, "8% itself is unsourced"); confirmed here to be a plain `.get()` fallback, not a required, explicitly-passed arm — nothing stops a caller from omitting it silently |
| **`storage_round_trip_eff`** | **0.90** | both builders | **NEW** — no `storage_efficiency_case` arm exists; this round-trip efficiency is a bare fallback |
| **`selfgen_busbar_factor`** | **1.0** | both builders | **NEW** |
| **`solar_min_build_mw_per_year`** | **0.0** | both builders, sourced from `solar_min_build_default()` | **NEW as a `.get()` site**, though the function itself (and its "no runner calls this any more" status) is already documented in `scenarios.py`'s own docstring, Part 1's module map |
| **`season_solar_factor`** | (`DEFAULT_SEASON_SOLAR_FACTOR`, value not re-derived here — a sibling default to `hydro_season_factor`) | sliced builder | **NEW**, and explicitly flagged by §3.3.7's own closing sentence as "one of four copies of the same two numbers, alongside a sibling solar shape factor" — this IS that sibling |
| **`EF_TCO2_PER_MWH_TH`** | **0.1823** | `io.load_econ`, in-function literal (not a `.get()`, an unconditional assignment) | **NEW** — SOURCED (IPCC default stationary-combustion factor, HHV-adjusted, quoted in-file: "IPCC default, HHV-adjusted. I'd Replace with Nigeria NIR factor if found") but hardcoded, not arm-selectable |
| **`RESERVE_MARGIN_SPINNING_MW`** | **900.0** | `scenarios.py` module constant | **NEW** — SOURCED (NIRP 2024 §2.3 p.19), feeds the `reserve_margin="nirp_spinning"` special-string path |
| **`GAS_BASELINE_MW`** | **10,947** | `scenarios.py` module constant | **NEW** — SOURCED (NERC Quarterly Reports, 23-plant tracked fleet), a structural guard target multiple other functions validate against |
| **`MODEL_START_YEAR`/`MODEL_END_YEAR`/`REPORT_END_YEAR`** | **2025 / 2054 / 2045** | `scenarios.py` module constants | **NEW** — the 2046-2054 buffer years are explicitly declared extrapolations, "Acceptable precisely because those years are never reported" |
| **`TARIFF_SWEEP_GRID`** | 10-point grid, 30M–110M | `scenarios.py` module constant | **NEW** — carries its own STALE-DOC warning in-file: "fine resolution was placed around the old 44-46M crossing... Widen and re-centre this grid in the final pass once rates are frozen" |
| **`_CONVERSION_NAMEPLATE_TOTAL_MW`** | **6,482.0** | `scenarios.py` module constant | **NEW** — a structural guard target (16 convertible plants), not a modelling assumption in the same sense as the others, but a hardcoded literal nonetheless |
| **`_HYDRO_EXPLOITABLE_MW`** | **17,620.0** | `scenarios.py` module constant | **NEW** — SOURCED (Energy Commission of Nigeria), a validity-guard ceiling, RAISES if exceeded |
| **`_HYDRO_BUILD_RATE_WARN_MW_PER_YR`** | **700.0** | `scenarios.py` module constant | **NEW** — DECLARED (one Zungeru-equivalent/yr), WARNS only, never raises |
| **`# storage_solar_surplus_frac`** | **0.20 (commented out)** | `optimize_model.py:386` | **NEW, and notable for being DEAD, not live** — a parameter that WAS apparently a modelling choice, now commented out; not exercised by definition since the line does not execute |

**Not counted as new findings, checked and excluded**: the `None`-default
`.get()` calls on `peak_mw`, `capacity_credit`, `hydro_capacity_mw_by_year`,
`hydro_energy_twh_by_year`, `gas_availability_factor`,
`solar_cumulative_cap_mw`, `gas_new_capacity_cap_mw`,
`gas_baseline_retirement_series_mw`, and `tariff` — every one of these
is followed by an explicit `is None` check that raises or branches
loudly (confirmed by direct reading, same discipline as the sentinel-
guarded arms in Part 1), so `None` here is a REQUIRED-effectively
guard, not a silent numeric fallback; including them would overstate
this section's finding.

**Total: 15 non-arm hardcoded constants/defaults found beyond §3.3.7's
five** (16 counting the dead, commented-out `storage_solar_surplus_frac`
separately), for **20 in total** across both sections once §3.3.7's
five are folded in.

---

## PART 4 — the runner surface

**Scope covered**: all 47 files directly under `scripts/` (46 `.py` +
1 `.sh` counted individually below; `run_central_sweep.sh` itself is
already inventoried in exhaustive per-invocation detail in this
session's `docs/task_log_ch36_inputs.md` item A and is not re-derived
here), `scripts/deferred/17_run_monte_carlo.py` [revived as `scripts/17_run_monte_carlo.py`, commit `36ddf55`, TASK 06], and both files in
`scripts/retired/`. `scripts/md_to_pdf.ipynb` is a document-conversion
notebook, not a research runner — noted for completeness, not
inventoried as a capability. `__pycache__` directories excluded (not
source).

**The single largest cross-cutting finding, stated once here rather
than repeated 30 times below**: `results/` timestamps split sharply
into two populations. A small, actively-maintained set —
`baseline/`, `baseline_sliced/`, `capital_frontier/`, `f3/`, `fin3/`,
`gas1/`, `ndc/`, `ndc_eaas/`, `_attrib/`, the `_sdr_sweep*`
directories — carries September 2026 timestamps tracking this
project's live correction sequence. **Everything else — `carbon_price_
sweep/`, `cf1/`, `dem1/`, `dem2/`, `eaas/`, `fin2/`, `gas2/`, `gas3/`,
`genset_arm_sensitivity/`, `leakage_arm/`, `monte_carlo/`, `obligation_
trajectory/`, `pareto_frontier/`, `pol1/`, `rel1/`, `rel1_mode_
sensitivity/`, `rel1_retirement/`, `rel2/`, `rel3/`, `selfgen_forcing/`,
`voll_sensitivity/` — is dated exactly 2026-08-23 13:44**, the same
stale timestamp register L61 already flags for the bare
`diagnostics.json` dependency. `results/str1/` is empty (0 files),
consistent with STR-1 being suspended. This is not 30 separate
staleness findings; it is one population boundary, checked directly
(`find results/*/ -printf '%T@'`, not assumed), that this Part reports
once and cross-references per script rather than re-deriving.

**A second cross-cutting fact, checked directly**: commit `0012f16`
(2026-09-18), message quoted in full: *"Revive nine dead runners:
02_run_eaas, 06_, 07_ (+A37 and a second TypeError bug), 08_gas3, 09_,
10_, 11_, 13_ (A55 closed), probe_2_6_structure; all route through the
annual builder and must run conv_none."* Nine scripts were repaired
(made callable again) on that date — but their `results/` output is
**still the pre-revival, August timestamp** for every one of them
except `02_run_eaas` (Aug 23, separately addressed in Part 2 addition
2), `06_` (current, re-run this session), and `07_` (still Aug 23).
**Revived ≠ re-run.** And the commit's own qualifier — "must run
`conv_none`" — means even a fresh run of any of these nine would be
annual-builder-only, unable to represent conversion (Part 2, item 3),
the second-largest cost lever in the model (F-6).

**None of the four orphaned modules (Part 1B) is imported by any
script** in `scripts/`, `scripts/deferred/`, or `scripts/retired/` —
confirmed by the same importer grep Part 1 ran, extended to include
these subdirectories. No runner exists that would wire `EaaS.py`,
`stochastic.py`, `gas_balance.py`, or `postprocess.py` even indirectly.

---

### Group: the central sweep and its regression anchor

**`run_central_sweep.sh`** — REPORTED (this is the sweep every figure
in `findings_frozen.md` traces to). Fully inventoried already in
`docs/task_log_ch36_inputs.md` (39 invocations, A5/B26/C8) — not
repeated here.

**`00_build_emissions_cap.py`** — question (quoted): *"generates
annual power-sector emissions caps under two distinct NDC apportionment
methodologies"* (NDC 2.0 BAU-relative and NDC 3.0 proportional
top-down). **REPORTED** — this session's own chapter-3.5.2 work and
register L60 depend on it directly; **EXERCISED** and current —
`data/cost/processed/emissions_cap.csv` was regenerated by commit
`7908d09` (2026-09-26), the L60 rebuild already covered in depth this
session. Not a `results/` writer (writes to `data/cost/processed/`
instead) — the one script in this Part whose output lives outside
`results/` entirely.

**`build_gas_deliverability.py`** — no module docstring (inferred
question from Part 1B's reading: generates the gas-availability CSV
six named shape-generators — `exponential_decline`, `drift`,
`lag_ramp_uplift`, `lag_then_drift`, `lag_then_decline`,
`shock_recovery` — feed). **REPORTED** transitively (every gas-regime
figure depends on its output CSV); writes to
`data/gas/processed/gas_available_power_annual_twh_th.csv`, not
`results/`.

**`99_validate_baseline.py`** — question (quoted): *"Cross-checks the
baseline LP's first-year (2025) output against actual 2024 generation
statistics from NERC / NBS to confirm the model is anchored to
reality."* Self-stamped: *"EPOCH: current as of HEAD `2a9226b`... every
OBSERVED_2024 figure below is re-verified... re-check this stamp if
either this file or the sources it cites move again."* **REPORTED**
(the backcast figure, e.g. −3.2%, is cited throughout `findings_
frozen.md`); `results/validation/` has 1 file, dated 2026-09-15 —
inside the current population, but the script's own epoch stamp names
`2a9226b`, an OLDER commit than several corrections landed since (L52-
L60) — **the stamp itself, not just the file date, should be re-checked
against current HEAD before being trusted further**, per its own
stated re-check condition.

---

### Group: NDC compliance — public vs. EaaS financing, and a genuine three-way overlap

**`02_run_ndc_caps.py`** (no docstring; CLI description quoted:
*"public-only NDC cap solves"*) and **`03_run_ndc_eaas.py`** (CLI
description quoted: *"EaaS-financed NDC cap solves"*) are a **matched
pair by design** — same NDC-cap scope, differing only in financing
regime, writing to `results/ndc/` and `results/ndc_eaas/`
respectively. Both **EXERCISED and current** (both directories carry
2026-09-26 12:00 timestamps, inside the live population) and
**REPORTED** (F-8, F-9, F-10, F-21, F-22 all depend on this
public-vs-EaaS NDC comparison).

**`02_run_eaas.py`** (CLI description quoted: *"EaaS-financed solar
solve"*) is **not the same question**, checked directly rather than
assumed from the similar name: it runs with `emissions_cap=None,
emissions_cap_by_year=None` (`:290-291`, quoted comment: *"Baseline =
no emissions cap"*) — **no NDC cap at all**. So it does not duplicate
`03_run_ndc_eaas.py`'s NDC-constrained question; it duplicates a
DIFFERENT, already-better-answered one: "what does an uncapped
EaaS-financed solve cost, and how much public capital does it need" —
which `28_run_capital_frontier.py`'s Frontier B (Part 2's addition 2
context; F-22) answers far more rigorously, as a swept frontier rather
than a single point, on the current sliced builder rather than the
annual one this script uses. **A genuine three-way overlap on the
uncapped-EaaS question**: `02_run_eaas.py` (annual, single point, plus
the legacy `run_deterministic_scenario` cross-check nothing else
provides — Part 2 addition 2), `28_run_capital_frontier.py` (sliced,
swept frontier, current, REPORTED), and — more weakly — `18_run_
pareto_frontier.py`'s `eaas_tight`/`eaas_moderate` arms (sliced, a
different swept axis, emissions rather than capital). Only
`28_run_capital_frontier.py`'s output is REPORTED for this question;
`02_run_eaas.py`'s own LP-side result is not cited anywhere in
`findings_frozen.md` — its **unique surviving contribution is the
legacy-accounting cross-check**, already assessed as non-current in
Part 2 addition 2 (`results/eaas/` dated Aug 23, predates both the
script's own Sep 18 revival and a Sep 25 bankability-pricing change).

---

### Group: GAS-1 / GAS-2 / GAS-3 — gas shadow price and regime shape

**`06_run_gas1_shadow_benchmarks.py`** (GAS-1, quoted RQ: *"What is the
shadow price of gas deliverability... and does it structurally exceed
the opportunity cost of gas exports and domestic regulated
benchmarks?"*) — **REPORTED** (F-30, and this session's own
`docs/task_log_findings_reverify_L60.md`); `results/gas1/` current
(2026-09-26 12:10, re-run this session against the L60 cap).

**`07_run_gas2_eaas_gas_relief.py`** (GAS-2, quoted RQ: *"How does
EaaS solar deployment change the scarcity rent of gas — and does
private capital deployment functionally relieve the gas constraint
that public-only investment cannot?"*) — revived 2026-09-18, further
touched 2026-09-20 (`c5b0d94`). **WIRED** (imports resolve, standard
arms present); **EXERCISED but STALE** — `results/gas2/` dated
2026-08-23 13:44, predating its own revival by nearly a month, so the
2 files present reflect the PRE-revival (broken) run, not a working
one. **Not REPORTED** — no GAS-2 citation found in `findings_frozen.md`
(checked directly, zero hits for "GAS-2" and for the relief-mechanism
concept). **This is a capability the thesis currently cannot claim**:
GAS-2's own question is analytically real and distinct from GAS-1
(shadow-price LEVEL vs. shadow-price RESPONSE to EaaS), the script is
wired, but there is no current evidence behind it — a revive-or-declare
decision, not made here.

**`08_run_gas3_regime_ndc_feasibility.py`** (GAS-3, quoted RQ: *"How
does the structural form of gas decline — exponential depletion vs
shock-recovery vs lag-ramp-uplift — affect the cost and feasibility of
NDC compliance independently of the level of gas availability?"*,
with an explicit, quoted shape-vs-level confound fix using
`build_gas_deliverability.py`'s flat-equivalent scenarios). Revived
2026-09-18. **WIRED**; **EXERCISED but STALE** — `results/gas3/`
dated 2026-08-23, same pre-revival gap as GAS-2. **Not REPORTED** —
zero "GAS-3" citations, zero shape-vs-level citations in `findings_
frozen.md`. Same revive-or-declare fork as GAS-2.

**`08_phase8_analysis.py`** — no docstring; output to
`results/phase8/`, dated 2026-09-15 (current population). Reads and
re-plots EXISTING result files (per its own `CONFIG` section, a
plotting/analysis script, not a new solve) — a downstream consumer of
other runners' output, not an independent capability; its state
depends entirely on whichever upstream files it reads being current,
not checked further here for scope reasons (this is a plotting script,
several of which exist — see the Group below).

---

### Group: reliability / feasibility (REL-1/2/3) — one wired trio, zero reported

**`09_run_rel1_feasibility.py`** (REL-1, quoted RQ: *"does the
feasibility of minimum reliability standards depend primarily on gas
deliverability regime rather than on the level of solar investment?"*),
**`10_run_rel2_marginal_cost.py`** (REL-2, quoted RQ at the time of this
audit: *"What is the marginal cost of tightening the reliability
standard (dual_reliability)... across gas deliverability regimes?"*,
with the dual-sign convention spelled out and quoted:
`dual_usd_per_twh = -dual_pyomo` — **both quotes superseded by TASK 02**
(docs/task_log_02_dual_fix.md): the script's docstring no longer carries
the stale "total mode"/`unserved[t]`-only constraint formula this quote
reflects, and `dual_usd_per_twh` is retired with no alias — read the
current docstring, not this quote, for HEAD's actual field names), and
**`11_run_rel3_financing_
frontier.py`** (REL-3, quoted RQ: *"Does EaaS financing improve
reliability outcomes... relative to public-capital-only financing...
under NDC3.0?"*) — all three revived together in `0012f16`
(2026-09-18), all three **WIRED**, all three **EXERCISED but STALE**
(`results/rel1/`, `results/rel2/`, `results/rel3/` all dated
2026-08-23, all pre-revival). **None is REPORTED** — zero "REL-1"/
"REL-2"/"REL-3" citations and zero matches for their distinguishing
concepts (dual_reliability, marginal cost of tightening, gas-regime-
as-primary-determinant) anywhere in `findings_frozen.md`, checked
directly. **A wired trio the thesis currently cannot claim any of** —
three separate revive-or-declare decisions, or one collective decision
if the REL suite is treated as a unit (`docs/eps_axis_rationale.md`
already treats it as one, per all three scripts' shared references to
it).

**`scripts/retired/09b_run_rel1_mode_sensitivity.py`** and
**`scripts/retired/09b_run_rel1_retirement_sensitivity.py`** — both
explicitly RETIRED (not deleted), each with its own retirement note
quoted in full at the top of the file. First: *"sweeps
`grid_service_mode` over ['annual', 'total']. That axis no longer has
two live options: the 2.6 docket... switched the grid-service
obligation from 'total' to 'annual' permanently... A sweep whose sole
purpose is to compare 'annual' against a retired mode cannot produce a
reportable result."* Second: *"sweeps the 680 MW/yr baseline
gas-retirement PROXY... superseded by the `gas_retire_vintage_{30,40,
50}` arms... The vintage arms showed the flat proxy overstates system
cost by ~1.28 bn (5.9%) and storage by ~17.9%."* Both also call the
ANNUAL `build_model` where storage is "structurally dominated and
identically zero" (their own quoted words, register L5) and both
predate the required pre-F1 scenario keys (register L10). **Both are
dead runners whose ANALYTICAL QUESTION (does reliability mode matter;
does the retirement-rate assumption matter) has already been
substantively ANSWERED by their own retirement notes** — this is the
one case in this Part where "dead runner" does not mean "unanswered
capability": the retirement note itself states the answer (mode:
'annual' is the only defensible choice; retirement rate: the proxy was
wrong by a quantified, material margin). `results/rel1_mode_
sensitivity/` and `results/rel1_retirement/` both exist (Aug 23, 1 and
3 files) as the pre-retirement evidence for those very conclusions.

---

### Group: capital budget (FIN-2/FIN-3) and tariff bankability

**`04_run_fin2_tariff_sweep.py`** (FIN-2, quoted RQ: *"What is the
minimum EaaS tariff at which private capital becomes self-financing,
what subsidy... does the model require below that tariff, and how does
the threshold shift with capital structure... and finance
conditionality?"*, with the analytical threshold formula quoted: *"T\*
= SOLAR_CAPEX_PER_MW × required_margin / npv_energy_per_mw"*). **WIRED**;
`results/fin2/` dated 2026-08-23 (stale population) — **not revived in
`0012f16`**, so unlike the REL trio this one was never even touched in
the September repair pass; its staleness has a longer, undated tail.
**Not REPORTED** — zero "FIN-2" citations in `findings_frozen.md`.
Revive-or-declare, with the added note that this one hasn't been
flagged as broken-and-fixed at all, only as stale.

**`05_run_fin3_capital_sweep.py`** (FIN-3, quoted RQ: *"How does the
public capital budget ceiling (B\*) independently determine the energy
access outcome under NDC constraints — and at what budget level does
the public-capital bottleneck disappear without EaaS?"*, with an
explicit robustness-gap statement quoted: prior runs "used a single
`capital_case` per NDC scenario... This fixed point cannot answer
FIN-3"). **REPORTED** (F-31, and this session's L60/L63 leakage work);
`results/fin3/` current (2026-09-26 12:09, re-run this session).

---

### Group: NDC-vintage comparison, and the ε-constraint frontiers (cross-referenced, not re-derived)

**`15_run_pol1_ndc_comparison.py`** (POL-1, quoted RQ: *"Does NDC 3.0's
absolute emission reduction target... require a materially different
investment trajectory than the NDC 2.0 BAU-relative targets — and does
EaaS change the answer?"*, with both cap formulas spelled out and a
striking quoted consequence: under NDC 3.0, *"model baseline power
emissions = ~5.4 MtCO2/yr... Required abatement >> available emissions
=> cap floors at ZERO => Gas-fired power must cease"*). Touched
2026-09-18 (`48ef4bd`). **WIRED**; `results/pol1/` dated 2026-08-23
(stale, predates the touch). **Not REPORTED** — zero "POL-1" citations;
the NDC 2.0-vs-3.0 comparison concept itself doesn't appear in
`findings_frozen.md` either (checked directly). Revive-or-declare —
and notably, this session's own chapter-3.5.2 work established the
CURRENT NDC 3.0 cap construction in detail (register L60) without ever
touching this comparison script, so the comparison itself, not just
its output, may now need re-deriving against the corrected cap before
any re-run would even be meaningful.

**`18_run_pareto_frontier.py`** (Frontier A, cost-emissions,
MIX-2) and **`28_run_capital_frontier.py`** (Frontier B, cost-capital)
— both **REPORTED** (F-12/F-13/F-22 and this session's
`docs/task_log_ch36_inputs.md` item C, which already gives their full
ε-grid, dual, binding-tolerance, and non-dominance-check detail; not
re-derived here). `results/capital_frontier/` current (2026-09-26
12:05); `results/pareto_frontier/` dated 2026-08-23 — **Frontier A's
own output is stale**, a fact not previously surfaced in this
session's Part-2/ch36 work, which focused on the SCRIPT's mechanics
rather than its `results/` currency. Frontier A is WIRED and its
mechanism is REPORTED via F-12/F-13 (whose numbers may predate the
same August cutoff — **worth a targeted re-check before those figures
are cited again**, though re-verifying F-12/F-13 against a fresh run is
outside this Part's scope).

**`26_run_f3_identification.py`** (F3, quoted RQ: *"Is total EaaS
solar capacity UNIQUELY identified at the optimal system cost, or is
there a range of EaaS deployments all achieving the same cost... ?"*,
with the lexicographic method spelled out: solve for Z\*, then
re-optimise at `Z* × (1+tol)` minimising then maximising
`solar_eaas_add`). **REPORTED** (F-12, F-13); `results/f3/` current
(2026-09-26 12:00).

**`29_run_parameter_uncertainty.py`** (Sobol) — already covered in full
depth in `docs/task_log_ch36_inputs.md` §D and this session's Parts
1-3 (the 748-verdict, the two docstring passages); not re-derived.
`results/parameter_uncertainty/` dated 2026-09-17 (current population,
though four days before this session's other September work — worth
noting the gap without asserting staleness, since no `src/` change
between 09-17 and today has been shown to touch anything Sobol
samples).

---

### Group: demand sensitivity (DEM-1/2) — one suspended by its own header, one live-but-stale

**`12_run_dem1_demand_sensitivity.py`** (DEM-1) — **self-declared
`[SUSPENDED]`** in its own docstring header, quoted in full: *"⚠
SUSPENDED pending Phase 2 tiered-demand rebuild. — latent_low/
latent_high were built as multiples of COLLECTED energy (23.08), which
is a revenue quantity, not demand. Wrong base. — All quantified figures
below predate the eta correction (eta now 0.287, not 0.43) and the
demand-base correction (37.09 TWh gross generation). Do not run or
quote."* **IMPLEMENTED only by its own declaration** — the suspension
is self-imposed and explicit, not inferred; `results/dem1/` exists
(Aug 23, 5 files) as the PRE-suspension evidence the header itself says
not to quote.

**`13_run_dem2_growth_gas_interaction.py`** (DEM-2, quoted RQ: *"How
does demand growth rate uncertainty interact with gas deliverability
constraints to determine system cost risk — and does higher demand
growth make the EaaS mechanism more or less valuable?"*, with an
explicit, quoted methodological distinction from Monte Carlo: *"DEM-2
asks... 'what is the OPTIMAL investment for each growth rate...'
This requires re-optimising under each growth rate — not evaluating a
fixed plan under perturbed conditions"* — i.e. DEM-2 and `stochastic.py`
answer genuinely different questions on the SAME demand-growth axis,
one by re-optimisation, one by fixed-plan re-dispatch). Revived
2026-09-18. **WIRED**; `results/dem2/` dated 2026-08-23 — **EXERCISED
but stale, same pre-revival gap as the REL trio and GAS-2/3**. **Not
REPORTED** — zero "DEM-2" citations. Revive-or-declare.

---

### Group: storage — one suspended runner, one accounting-residual test, one dead capability underneath both

**`14_run_str1_storage_role.py`** (STR-1) — **self-declared
`[SUSPENDED]`**, quoted in full: *"STR-1 asks what role battery storage
plays in the optimal system. It cannot be answered in the current
model, and this runner is intentionally disabled... storage_add = 0 is
strictly optimal in every scenario. Empirically confirmed: all runs
return zero storage capacity and zero discharge... The real source of
storage value — shifting midday solar to evening peak — does not exist
in the model because there is no intra-annual time structure."*
**This is exactly Part 2's own finding restated independently**: the
annual builder makes storage "structurally incapable of value." STR-1's
suspension and Part 2's builder-capability finding corroborate each
other from opposite directions (a runner's own header vs. a builder's
own docstring) — the same fact, found twice, independently. `results/
str1/` is **empty** (0 files) — the only fully-empty results directory
found in this Part, consistent with "intentionally disabled" meaning
"never produced output at all," not merely "produced stale output."
**Not REPORTED** in the STR-1 sense — but F-19 ("Storage magnitude is
not a cost optimum") reports the SAME underlying finding via a
different runner (below), so the analytical conclusion IS reported,
just not through this script.

**`22_salvage_sensitivity.py`** (quoted purpose: *"Tests whether the
capacity trajectory RESPONDS to cost, or is determined by the energy
balance regardless of cost"*, with the motivating finding quoted:
*"When salvage was added..., system cost fell 17.1%... and the solar
trajectory was IDENTICAL to five decimal places, while sliced storage
was identical to four. Zero MW of movement"*) — **REPORTED** (F-19,
F-20); `results/salvage_sensitivity/` dated 2026-08-23 22:01 —
technically inside the "stale" population by date, but this is the
script's OWN currency epoch (the salvage-value commit `2380e5b` landed
2026-08-17, so an Aug-23 run postdates the feature it tests) — **flagged
as a case where the blanket Aug-23-is-stale heuristic does NOT apply
without checking the specific dependency**, exactly the kind of
exception this Part's method section warns must be checked, not
assumed.

**`27_run_hydro_minflow_sensitivity.py`** (quoted purpose: verifies
`hydro_min_flow_fraction`'s bindingness and sensitivity, citing
`docs/task_hydro_gate_bucketA.md`'s finding that the constraint BINDS
at the central configuration) — touched 2026-09-07 (`66cd29e`);
`results/hydro_minflow/` dated 2026-09-15 (current population, after
the touch — **currency confirmed, not just population-inferred**).
Not independently checked against `findings_frozen.md` for a specific
citation in this pass — Part 2 already covers the hydro min-flow
mechanism itself in depth (item 6); this script's OWN sensitivity
output's citation status is not separately confirmed here.

---

### Group: one-off physical/policy sensitivities (CF-1, carbon price, resolution premium, leakage, obligation trajectory, genset)

**`20_run_cf1_solar_yield_sensitivity.py`** (CF-1, quoted RQ: *"How
sensitive are system cost and solar deployment to the assumed solar
capacity factor?"*, with the sourced CF band quoted in full — low
0.16/central 0.19/high 0.20, Global Solar Atlas). Revived-adjacent
(touched `ecb7e54`, 2026-09-15); `results/cf1/` dated 2026-08-23 —
**stale**, predating its own most recent touch. **Not REPORTED**: the
solar-CF VALUES (0.16-0.20) are cited in `findings_frozen.md` as a
sourced PARAMETER (`:1305`), but that cites the arm's own sourcing in
`scenarios.py`, not this script's SWEEP RESULT — checked directly, no
citation of a CF-1 cost/deployment sensitivity finding was found.

**`19_run_carbon_price_sweep.py`** (quoted RQ, MIX-3: *"At what carbon
price does solar become cost-dominant, and how does EaaS affect this
threshold?"*, tipping point defined as >50% solar generation share OR
gas below 50% of baseline). **WIRED**; `results/carbon_price_sweep/`
dated 2026-08-23 (stale). **Not REPORTED** — zero carbon-price-tipping
citations found.

**`21_compute_resolution_premium.py`** (quoted purpose: computes *"the
cost premium that intra-annual temporal resolution reveals... a METHODS
FINDING, not a scenario result"*, with an explicit self-check quoted:
*"This script therefore refuses to report unless it can verify: 1. Both
runs used the same demand arm, gas regime and VoLL case. 2. Both
objectives include the salvage credit... The previously reported ~13%
premium was measured BEFORE salvage and is superseded."*) —
`results/resolution_premium/` dated 2026-09-15 (current population).
Last touched `2380e5b` (2026-08-17, the same salvage commit) — **the
script's own basis-consistency guard means its September output, if
current, already accounts for salvage**; not independently confirmed
against `findings_frozen.md` for a specific up-to-date premium citation
in this pass.

**`24_run_leakage_arm.py`** — this IS F-11's own runner, confirmed
directly: F-11 (quoted, `findings_frozen.md:419-437`) states *"the
`leakage` field is null in the current cap summaries, so the 34.2-45.0%
figures from the pre-O&M epoch are NOT restated here... UNTESTED
COUNTERFACTUAL: `selfgen_in_emissions_cap=True`... Not exposed by any
runner"* [F-11's own wording — but this IS the runner built for exactly
that counterfactual, per this script's own docstring, quoted: *"runs
the SAME forcing case with the boundary drawn both ways... False = cap
= emissions_grid_only (central); True = cap = emissions_total"*].
**A direct, checkable inconsistency between F-11's prose and this
runner's existence** — F-11 says "not exposed by any runner" while
`24_run_leakage_arm.py` exists specifically to expose it; the resolution
is currency, not contradiction: `results/leakage_arm/` is dated
2026-08-23, **pre-O&M** (the O&M implementation commit is `d7538ff`,
per `findings_frozen.md`'s own header) — so the runner DOES exist, its
one stored result IS the stale pre-O&M figure F-11 declines to
restate, and "not exposed" should be read as "not exposed AT THE
CURRENT STATE," not "no runner was ever built." Worth a one-word
correction wherever F-11 is next touched, flagged here rather than
edited (out of scope for a read-only Part).

**`25_run_obligation_trajectory.py`** (quoted purpose: tests whether a
front- vs. back-loaded grid-service-obligation trajectory shifts solar
build TIMING under a fixed capital envelope, at matched average
eps=0.45) — `results/obligation_trajectory/` dated 2026-08-23 (stale).
Not REPORTED (no citation found).

**`99c_run_genset_arm_sensitivity.py`** (quoted purpose: sweeps the
three genset pricing arms, explicitly REPLACING a retired VoLL sweep
whose defect is quoted in full: *"the old script reported `unserved`
but not `selfgen`... It produced `unserved = 0.00` across the board
while real shortfall of hundreds of TWh went unreported"*) —
`results/genset_arm_sensitivity/` dated 2026-08-23 (stale, and predates
the script's own most recent touch, `ecb7e54` 2026-09-15). Not
REPORTED (no citation found). The file the sweep replaced,
`99c_run_voll_sensitivity.py`, is not present in `scripts/`,
`scripts/deferred/`, or `scripts/retired/` — genuinely removed, not
merely relocated (checked: `git log --all --diff-filter=D --
"*voll_sensitivity*"` would confirm the deletion commit precisely; not
run here, noted as the one script-removal not tied to a `deferred`/
`retired` relocation).

---

### Group: plotting, diagnostics, and harness utilities (not independent capabilities)

**`16_plot_cost_decomposition.py`** — reads existing baseline/NDC
output and plots a cost-decomposition figure; its own docstring states
plainly *"CORRECTION PLAN 2.6 CHANGED WHAT THIS FIGURE SHOWS"* (the VoLL
wedge is now near-zero by construction, D11) — a downstream consumer,
not an independent finding; current only insofar as its three
pre-requisite runners' outputs are (already assessed above: `01_`/`02_
run_ndc_caps`/`03_` are all current).

**`08_phase8_analysis.py`, `plot_storage_regime.py`, `plot_system_
diagnostics.py`** — none has a module docstring; all three are plotting
scripts reading pre-existing `results/` JSON/CSV files (confirmed by
direct inspection of their `CONFIG`/path-hardcoding sections). None
independently produces a model capability; each is only as current as
whatever it happens to read, which for `plot_storage_regime.py`
specifically is a single hardcoded path,
`results/ndc/ndc_unconditional_20_voll_low/diagnostics.json` —
**a directory name that does not match `02_run_ndc_caps.py`'s current
output-naming scheme** (Part 1's `output_naming.py` hash8 scheme, used
by `02_run_ndc_caps.py` per Part 1) — checked directly, `results/ndc/`
contains hash8-tagged subdirectories, not a `ndc_unconditional_20_
voll_low`-named one. **This script's hardcoded path does not resolve
against current `results/ndc/` at all** — it would fail immediately if
run, a genuine broken-by-rename finding, distinct from mere staleness.

**`_check_solver.py`** — no docstring; three lines checking CBC/GLPK/
HiGHS solver availability. Infrastructure, not a capability.

**`_compare_2_6.py`, `probe_2_6_structure.py`** — both one-off
regression/structural-probe tools for the 2.6 correction (selfgen
wiring); both quote their own three-test/probe methodology in detail
in-file. `probe_2_6_structure.py` was one of the nine revived in
`0012f16`. Infrastructure/regression tooling for a since-completed
correction phase, not an ongoing analytical capability.

**`f1_quantity_diagnostic.py`, `rpublic_diagnostic.py`, `diag_fin2.py`**
— all three explicitly self-described as read-only, no-solve diagnostic
scripts (each states this in its own docstring, quoted in the file
excerpts above). Infrastructure for specific past investigations
(F1 quantity, `r_public` stance, FIN-2 omnibus), not standing model
capabilities.

**`run_verification_sweep.sh`, `smoke_all.sh`, `verify_2_6_commit_a.sh`**
— harness scripts for CLI-health checking (the three-outcome PASS/
NEEDS-ARGS/FAIL classifier, quoted in full above) and a git-worktree-
based regression comparator. Not analytical runners; each is
infrastructure for verifying that OTHER scripts are wired, which is
itself useful evidence for this Part's own WIRED determinations where
their logic was cited above (e.g. the "NEEDS-ARGS is healthy" framing
underlies why a script rejecting a bare CLI invocation is not,
by itself, evidence of being broken).

---

### The deferred capability: `scripts/deferred/17_run_monte_carlo.py` [revived as `scripts/17_run_monte_carlo.py`, commit `36ddf55`, TASK 06]

Read in full. **Structurally sound in conception, two blocking defects,
both quoted from its own header**: *"1. LCOE is miscomputed. The
per-draw LCOE numerator includes the VoLL penalty..., while the
denominator omits hydro generation entirely and double-counts storage
discharge... 2. The demand-growth prior is unsourced... centred on an
unsourced value ([SOURCE NEEDED], plan 1.6)."* Also, independently:
*"it also currently calls the ANNUAL `build_model` (storage
structurally zero, register L5) and is missing required pre-F1
scenario keys (register L10)"*.

**What analytical capability this uniquely represents, checked against
what's live**: this script's OWN demand-growth/gas-regime joint
sampling defect (drawing them independently despite a shared driver) is
the EXACT failure mode `29_run_parameter_uncertainty.py`'s docstring
cites BY NAME as the mistake it exists to avoid (already quoted in this
session's `docs/task_log_ch36_inputs.md` §D) — so Sobol's design is a
direct, documented response to this file's defect, not an independent
duplicate. Neither Sobol nor `stochastic.py` (Part 1B) actually
DUPLICATES `17_`'s own question: Sobol samples PARAMETER uncertainty
at a fixed scenario (a different axis entirely); `stochastic.py` does
fixed-PLAN re-dispatch under scenario perturbation (correct
methodology, but evaluates ROBUSTNESS of one plan, not the joint
demand-growth/gas-regime interaction `17_` was built to explore across
DIFFERENT re-optimised plans — closer in spirit to DEM-2's
re-optimisation approach, but DEM-2 sweeps demand growth ALONE, not
jointly with gas regime). **`17_`'s specific analytical territory — a
joint demand-growth × gas-regime Monte Carlo, correctly re-optimising
rather than fixed-plan re-dispatching — is not currently duplicated by
anything live**, once its own two named defects are fixed and it is
ported to the sliced builder. `results/monte_carlo/` (5 files, Aug 23)
predates the deferral itself and reflects a run made before both
defects were identified — not citable, and not claimed to be by
anything in `findings_frozen.md` (checked, zero citations).

---

### Closing: capabilities that exist only in a dead or unrun runner

Per TASK.md's framing — named here, revive-or-declare left to Zeus and
Chat together, no recommendation made:

1. **GAS-2** (EaaS gas-scarcity relief) — wired, revived, stale output, unreported.
2. **GAS-3** (gas-decline SHAPE vs LEVEL, with its own confound already fixed in-script) — wired, revived, stale output, unreported.
3. **REL-1/REL-2/REL-3** (gas-regime-as-reliability-determinant; marginal cost of the reliability standard; EaaS's reliability value under NDC3.0) — wired, revived, stale output, unreported, as a trio.
4. **DEM-2** (demand-growth × gas-deliverability cost-risk interaction, re-optimising not fixed-plan) — wired, revived, stale output, unreported.
5. **FIN-2** (EaaS self-financing tariff threshold, with an analytic T\* formula already derived in-script) — wired, never revived, stale output, unreported.
6. **POL-1** (NDC 2.0 vs. NDC 3.0 investment-trajectory comparison) — wired, touched Sep 18, stale output, unreported; its own cap logic may now be stale relative to this session's L60 rebuild, a second reason to re-derive rather than merely re-run.
7. **CF-1** (solar-CF cost/deployment sensitivity, sourced band already in place) — wired, stale output, unreported (the CF VALUES are cited elsewhere; this script's own sweep RESULT is not).
8. **Carbon-price tipping point** (MIX-3) — wired, stale output, unreported.
9. **`scripts/deferred/17_run_monte_carlo.py` [revived as `scripts/17_run_monte_carlo.py`, commit `36ddf55`, TASK 06]'s joint demand-growth × gas-regime re-optimising Monte Carlo** — genuinely undupli­cated territory once its two named defects (LCOE miscomputation, unsourced prior) are fixed and it is ported off the annual builder.
10. **`src/EaaS.py`'s service-payment/contract-tenor mechanism, `src/stochastic.py`'s fixed-plan re-dispatch, `src/gas_balance.py`, `src/postprocess.py`** — already covered in full in Part 1B; restated here only because no runner in this Part wires any of them, confirming Part 1B's finding from the opposite direction.

**Not in this list, checked and excluded**: `9b`-retired REL-1
sensitivities (their own retirement notes already state the answer,
per the Group above — a closed question, not an open capability);
`02_run_eaas.py`'s LP-side result (superseded by Frontier B, not an
independent unclaimed capability); STR-1 (its conclusion IS reported,
via F-19, through a different runner).

## PART 3B — the unsourced audit

Audits Part 3's arm table specifically: every arm value there tagged
unmarked, UNSOURCED, `[SOURCE NEEDED]`, `[PENDING]`, PLACEHOLDER,
STIPULATED, ASSUMED, or DECLARED. Read-only; no tag, comment or
classification in any file was changed. Git history checked with
pickaxe search (`git log -S"<name>"`) on the defining function/arm
string, not a naive full-history read — efficient and, for every entry
below, sufficient to establish first-touch and last-touch commits.

**Operationalised classification rule, stated up front so the table's
choices are checkable rather than asserted.** The four categories given
overlap in places (a "declared judgement" value could read as either
DELIBERATE or GENUINELY UNSOURCED); this audit resolves the overlap by
ROLE, not by whether the code happens to use a soft word:
- **(a) DELIBERATE** is reserved for values whose STATED ROLE is
  structural — a sensitivity bound bracketing a central case, a
  regression/legacy value retained only for attribution, or an
  explicit counterfactual — i.e. nothing claims the value is Nigeria's
  best real-world estimate. A code path that RAISES rather than run
  unsourced (`peak_nirp`, `gas_avail_central`) is treated as DELIBERATE
  by the same logic: the raise IS the deliberate discipline.
- **(d) GENUINELY UNSOURCED** is for values that ARE presented as the
  best estimate (typically the arm's own CENTRAL/default member) but
  have no source — including ones the code calls "DECLARED JUDGEMENT"
  or "ASSUMED," since an honest label is not a source.
- **(b)/(c)** are as the instruction defines them.
Where a family has a labelled central value plus low/high siblings,
the CENTRAL value is judged on its own merits (b/c/d) and the
siblings are (a) by construction, since their entire job is to bound
the central value's uncertainty, not to independently claim reality.

### Full classification table

| arm value | introduced | last touched | sourcing attempt on record? | class |
|---|---|---|---|---|
| `latent_low`/`latent_high` (`demand_level_case`) | pre-dates this repo's earliest scenarios.py commits reachable by pickaxe on the surrounding docstring text; deprecated in place, not re-introduced | same commit sequence that added the `raise` guard (not separately dated — deprecation is a doc/guard change, not a value change) | Deprecation itself IS the recorded decision — "retained for provenance only" | **(a) DELIBERATE** |
| `peak_nirp` | `a2717ac` (2.4, sourced peak-shape arms) | unchanged since | none attempted; NIRP gives a load factor, no daily profile, per the arm's own comment | **(a) DELIBERATE** |
| `peak_high` | `a2717ac` | unchanged since | none — explicitly "CONSTRUCTED, not measured" | **(a) DELIBERATE** |
| `moderate_policy`/`stringent_policy` (`carbon_case`, 50/75 USD/tCO2) | `5d079ab` ("modify scenario: carbon price scenarios") | `9bcc8d7` (bulk "modify: src files") | **none** — both commit messages are generic, no sourcing language; current docstring is one line, `"Carbon policy stances (deterministic). Price in USD/tCO2."`, no source | **(d) GENUINELY UNSOURCED** |
| `nirp_accelerated` (`solar_build_case`, 2.0×) | `74d34f2` (plan 2.5 step 4, horizon extension — incidental touch, not the introducing commit for this value; earlier history not resolved by this pickaxe term) | same | Part 3's own extraction already found this **explicitly flagged in-file**: `[SOURCE NEEDED] for the 2.0x... currently a declared assumption` | **(d) GENUINELY UNSOURCED** — already self-flagged, no source found since |
| `land_case`: `tight`/`moderate`/`loose` (1,000/3,000/6,000 km²) | `5292474` ("model update") | `72cad20` ("large refactor") | **none anywhere** — checked `docs/limitations_register.md` (zero hits for `land_case`/`land_scenarios`), `docs/parameter_provenance.md` (no row), the function's own docstring (`"""Land availability envelopes (km2)."""`, one line, no source); both commit messages are generic | **(d) GENUINELY UNSOURCED** — and **live**: all three values are exercised (175/175/4 `results/` hits) |
| `tariff_central` (95M) | `ee19e0e` ("tariff_case: required arm... NBET-sourced 75/95/115M") | unchanged since | the commit message itself states the construction: "NBET-sourced 75/95/115M" — 75 and 115 are the sourced bookends, 95 is their declared midpoint | **(a) DELIBERATE** |
| `tariff_legacy_95m`/`tariff_legacy_65m` | `ee19e0e` | unchanged since | explicit in-file: "RETAINED FOR ATTRIBUTION ONLY, both UNSOURCED" | **(a) DELIBERATE** |
| `storage_cc_central` (0.80) | `f9672b8` (F1 tranche introduction) | unchanged since | "SOURCE: DECLARED JUDGEMENT, not measured" — an honest label, not a source; "matches gas's availability-factor credit" is an internal analogy, not an external citation | **(d) GENUINELY UNSOURCED** |
| `storage_cc_low`/`storage_cc_high` (0.50/1.00) | `f9672b8` | unchanged since | same declared-judgement basis as the central value, explicitly bracketing it | **(a) DELIBERATE** |
| `gas_avail_central` | `f9672b8` | unchanged since | "SOURCE: [PENDING]... Raises if selected until sourced" | **(a) DELIBERATE** |
| `gas_avail_nameplate` (0.80) | `f9672b8` | unchanged since | explicit: "the pre-2.10 unsourced assumption, retained as an explicit UPPER BOUND" | **(a) DELIBERATE** |
| `conv_central` (600 USD/kW, 0.50 uplift, 0.46 eff) | `84a9580` (Phase 2.8, conversion tiers) | unchanged since | capex base: "DERIVED... greenfield CC-minus-SC increment" (Gas Turbine World 2024) — SOURCED method; the uplift/premium on top is a stated "declared brownfield premium" with no cited figure | **(c) ADJUSTMENT-UNSOURCED** |
| `conv_low`/`conv_high` (850/450 USD/kW variants) | `84a9580` | unchanged since | same basis, explicitly bracketing the premium/efficiency assumptions | **(a) DELIBERATE** |
| `gas_retire_proxy` (680 MW/yr from 2035) | `f9672b8` | unchanged since | "the pre-2.11 'brownfield retirement proxy'... SOURCE: [PENDING]" | **(d) GENUINELY UNSOURCED — and concerning**: unlike `peak_nirp`/`gas_avail_central`, this arm carries NO raise guard; Part 3 confirms 4 `results/` hits — **it has been run and produced reported-adjacent output while explicitly tagged `[PENDING]`, with nothing in the code stopping that** |
| `gas_retire_vintage_30` (central, 30 yr) | `f9672b8` | unchanged since | "SOURCE: ASSUMED — analyst vintage estimate, not traced to a Nigerian source" | **(d) GENUINELY UNSOURCED** — the model's CENTRAL gas fleet-life assumption, live in 992 `results/` hits |
| `gas_retire_vintage_40`/`gas_retire_vintage_50` | `f9672b8` | unchanged since | same ASSUMED basis, explicitly bracketing the central vintage | **(a) DELIBERATE** |
| `gas_capex_low` (900,000 USD/MW) | `f9672b8` | unchanged since | explicit: "the pre-2.13 bare literal... never sourced," retained as a stated lower bound | **(a) DELIBERATE** |
| `ccgt_eff_central` (0.5173, 5% derate) | `f9672b8` | unchanged since | Base: SOURCED (EIA / Sargent & Lundy SL-018001 Table 5-1; cited as NREL ATB 2024 until TASK 73); the 5% ambient derate itself: "a DECLARED ASSUMPTION, NOT from ATB" | **(c) ADJUSTMENT-UNSOURCED** |
| `ccgt_eff_low` (0.5010, 8% derate) | `f9672b8` | unchanged since | same basis, explicitly bracketing the derate assumption | **(a) DELIBERATE** |
| `cr_central` — `r_comm` component (0.11) | `f9672b8` (introduced); rate-coherence work | `e46be32` (2026-09-25, bankability commercial-basis fix — touched the surrounding mechanism, not this specific literal) | **Already tracked, not a new finding**: `docs/parameter_provenance.md` row 3 — IEA WACC base (8.5-9%, Kenya/Senegal, Nigeria not covered) + "a 2-2.5pp analyst uplift... NOT SOURCED for the objective's uplift, 0.11 still below the CPI [cost-of-debt] floor" — register **L52/A61** | **(c) ADJUSTMENT-UNSOURCED** |
| `cr_comm_low`/`cr_comm_high` (0.09/0.14) | `f9672b8` | unchanged since | same IEA-base-plus-assumed-uplift structure, bracketing it | **(a) DELIBERATE** |
| `dscr_central`/`dscr_low`/`dscr_high` (1.30/1.20/1.40) | `f9672b8` | unchanged since | **none, for any of the three** — no in-file marker at all (Part 3's own finding: "no explicit source/assumed marker given"), no `parameter_provenance.md` row, no `limitations_register.md` entry found by name | **(d) GENUINELY UNSOURCED, all three** — no basis in this family to split central from bounds the way every other family above allows, since nothing distinguishes them textually either |
| `env_dares` (750M) | `f9672b8` | unchanged since | "STIPULATED, not SOURCED, within this repository" **but the surrounding docstring itself names a real, checkable external source**: World Bank DARES, $750M IDA credit, December 2023 — the citation exists in prose, just not formalised into a provenance-table row or upgraded past the STIPULATED tag | **(b) SOURCED-ELSEWHERE** |
| `env_dares_plus` (1,050M) | `f9672b8` | unchanged since | explicit: "no external corroboration... remains STIPULATED, weaker than `env_dares`" | **(d) GENUINELY UNSOURCED** |
| `env_none`/`env_5x`/`env_10x` | `f9672b8` | unchanged since | explicit: "declared counterfactuals... do NOT claim DARES itself is 5x or 10x its stated size" | **(a) DELIBERATE** |
| `blended_central` (`blend_case`, 30% concessional share, ADOPTED) | `e46be32` (introduced); `9815a64` | `e46be32` (2026-09-25) | `r_concessional` (the rate): SOURCED. `concessional_share`=30% (the modifier this arm sets): "remain PLACEHOLDERS pending primary-source verification" | **(c) ADJUSTMENT-UNSOURCED** |
| `commercial_only`/`blended_optimistic`/`concessional_heavy` (0%/50%/75% shares) | same commits | same | same PLACEHOLDER basis, explicitly bracketing the adopted 30% | **(a) DELIBERATE** |
| `within_block_peaking_factor` (1.3448, both builders' `.get()` fallback) | `a2717ac` ("2.4: sourced peak-shape arms... within-block peaking factor 1.3448" — the commit message NAMES the number at introduction) | `84a9580` (incidental move during the conversion refactor, value unchanged) | **Extensively documented elsewhere, not at all in the code comment next to it**: register **L50**, forensic audit **A62**, `docs/parameter_provenance.md` row 7, `docs/methodology_physical_layer.md` §2 (the derivation, 1.4588/1.0847=1.3448, from the same Oluwole 2016 peak-shape reconstruction), even `docs/chapters/chapter3_1_v2.md` [removed `009d230`; held at `a993f15`; successor `CHAPTER_3_assembled.md` §3.1]'s own "Declared limitation. ω = 1.3448..." — while the CODE COMMENT immediately above the `.get()` call still reads "SOURCE: [PENDING]... DEFAULT 1.0 is a declared conservative simplification," directly contradicting the 1.3448 literal two lines below it | **(b) SOURCED-ELSEWHERE — the clearest cheap win in this audit**: the source is not just findable, it is ALREADY WRITTEN UP in four separate places; only the code comment at the definition site was never reconciled |
| `social_discount_rate` (0.08, both builders' `.get()` fallback) | `f9672b8` | `9815a64` | `docs/parameter_provenance.md` row 6, already on record: **"ASSUMED. No source of any kind exists in the repository for this value."** `docs/chapters/chapter3_1_v2.md` [removed `009d230`; held at `a993f15`; successor `CHAPTER_3_assembled.md` §3.1]'s own "Note to Zeus" already flags it as unsourced and suggests a Ramsey-type derivation would likely land lower (3-6%) | **(d) GENUINELY UNSOURCED** — already known, restated here because Part 3 found it is ALSO a silent `.get()` default, not a required arm; nothing stops a caller from omitting it entirely |

### Class counts

**32 values/families audited. (a) DELIBERATE: 16. (b) SOURCED-ELSEWHERE:
2. (c) ADJUSTMENT-UNSOURCED: 4. (d) GENUINELY UNSOURCED: 10.**

### (d) GENUINELY UNSOURCED — what a source would look like, per case

Grouped by what kind of document would actually resolve each, with a
plain judgement on whether one plausibly exists (this is explicitly a
judgement call, stated as such per instruction):

- **`land_case` (tight/moderate/loose, 1,000/3,000/6,000 km²), all
  three live in `results/`.** A source would look like a Nigerian
  land-use/land-availability study for utility-scale solar siting
  (e.g. a National Space Research and Development Agency land-cover
  dataset, a Ministry of Environment land-availability screen, or an
  IRENA/World Bank solar-siting resource assessment for Nigeria).
  **Plausibly exists** — this is a well-studied category for solar
  deployment planning globally — but was not searched for in this
  audit and is not cited anywhere in the repository today.
- **`dscr_central`/`low`/`high` (1.30/1.20/1.40), the model's debt-
  service-coverage-ratio assumption family, live in 799+23+23
  results.** A source would be a Nigerian project-finance benchmark
  (an NERC-approved PPA financial model, a DFI/IFC lending covenant
  standard for African power projects, or a ratings-agency project-
  finance methodology note stating typical DSCR covenants). **Plausibly
  exists** — DSCR covenants are standard, publicly discussed project-
  finance terms — but nothing in this repository traces to one.
- **`storage_cc_central` (0.80) and `gas_retire_vintage_30` (30 yr),
  each the model's own central assumption for its category.** Storage
  capacity credit: a source would be a system-operator or NERC firm-
  capacity study for battery storage (analogous to the NIRP source
  already used for spinning reserve). Gas fleet retirement age: a
  source would be plant-level commissioning-date records cross-
  referenced against a stated technical-life standard (the same class
  of document `_HYDRO_EXPLOITABLE_MW`'s Energy Commission of Nigeria
  citation already uses for hydro). **Both plausibly exist** in
  principle; neither was found.
- **`gas_retire_proxy` (680 MW/yr) and `env_dares_plus` (1,050M) and
  `moderate_policy`/`stringent_policy` carbon prices (50/75
  USD/tCO2) and `nirp_accelerated` (2.0×).** These four are lower
  priority in one specific sense — Part 3 already found `moderate_
  policy`/`stringent_policy`/`nirp_accelerated` have **zero** `results/`
  hits (never actually exercised), so no reported figure currently
  depends on them; `env_dares_plus` similarly has only 23 hits, a small
  sensitivity arm, not the central case. `gas_retire_proxy` is the one
  exception in this group **with results despite the unsourced tag**
  (flagged above). A source for the carbon prices would be a Nigerian
  or comparator-country carbon-pricing policy proposal (Nigeria has no
  enacted carbon price, so this may be **genuinely unavailable**, in
  which case the honest answer is "state as a policy sensitivity, not
  a sourcing gap" — a judgement call this audit surfaces but does not
  resolve).
- **`social_discount_rate` (0.08).** Already covered above (b/d table)
  — a source would be a Ramsey-derivation input set (pure time
  preference + consumption growth + elasticity of marginal utility) or
  a stated sovereign guidance rate. **Plausibly exists** (many
  finance ministries and multilaterals publish appraisal discount-rate
  guidance) but was not searched for here, and the chapter draft
  already flags it as its own open item independent of this audit.

**No edit made, no tag removed, per instruction.** Every classification
above is this audit's own judgement, cross-checked against git history
and existing documentation, not asserted from the arm's in-file wording
alone — Zeus decides which of the 10 GENUINELY UNSOURCED and 2
SOURCED-ELSEWHERE cases to pursue.

---

## PART 5 — uncertainty and trade-off methods, precisely

### 1. The arm system / scenario sweep — scenario uncertainty

Full evidence base: Part 3 (23 arms, ~95 values) and Part 3B (their
sourcing audit). What it claims, in the codebase's own words
(`docs/methodology_protocols.md:363-368`, quoted):

> "These are not samples from a distribution. They are named,
> internally coherent states of the world, and the constraint-relief
> cross-tab rules out combinations that would be incoherent — a world
> where demand booms while the gas that would serve it collapses, for
> instance."

What it does NOT claim: a probability, a confidence interval, or
coverage of the full joint space of arm combinations — Part 3 found
several values (e.g. `nirp_accelerated`, `moderate_policy`/
`stringent_policy`) with zero `results/` exercise, so even the DISCRETE
set of named worlds is not exhaustively populated in practice. Section
C's interaction cells (Part 1, C-1 correction) are the only place two
dimensions move at once in a controlled way; everywhere else, one arm
moves against the central configuration at a time (Section B) or not
at all (Section A).

### 2. Sobol (`29_`) — parameter uncertainty by Monte Carlo over deterministic solves

Docstring's own distinction, quoted in full (`scripts/29_run_
parameter_uncertainty.py:5-27`; identical text already captured this
session in `docs/task_log_ch36_inputs.md` item D — repeated here since
TASK.md asks for it in this Part specifically):

> "PARAMETER uncertainty vs SCENARIO uncertainty -- keep these separate
> --------------------------------------------------------------------
> This model is DETERMINISTIC with structured scenario analysis. The
> project's 38-arm sweep answers SCENARIO uncertainty -- which discrete
> world happens... This script answers a DIFFERENT question: PARAMETER
> uncertainty -- five physical/cost quantities whose TRUE value is
> unknown but FIXED (not a scenario branch), sampled continuously at
> the CENTRAL scenario configuration, decomposed by Sobol variance
> analysis to find which parameter's uncertainty drives the output
> band. The two kinds of uncertainty must not be conflated..."

What it claims: a variance decomposition over five CONTINUOUS physical/
cost parameters (gas availability/capacity-credit, gas efficiency,
solar CF, O&M level, conversion capex), each independently re-optimised
(full LP re-solve per draw, not fixed-plan). What it does NOT claim:
coverage of scenario uncertainty (explicitly disclaimed above), nor
genset efficiency uncertainty (deliberately excluded — Part 1B/2's own
reading of this docstring already covers why: the three published
genset efficiency values are different LOAD POINTS, an operating
choice, not a parameter uncertainty).

### 3. `src/stochastic.py` — fixed-plan re-dispatch

Full coverage already exists at two points in this document — Part 1B
(`### src/stochastic.py (227 lines)`, line 479) and Part 2's addition 1
(`### Part 2, addition 1 — stochastic.py: a runner away, not a
rewrite`, line 1096) — restated here in this Part's own terms since
TASK.md asks for the method, distributions, caller, results/ status,
and limitation note specifically in Part 5:

**Full method**: two-stage fixed-plan re-dispatch. Solve the LP once at
a base scenario to obtain the optimal capacity plan
(`solar_public_add`/`solar_eaas_add`/`storage_add` per year); `.fix()`
those three variables in every subsequent draw via `build_model(...,
fixed_plan=fixed_plan)`; re-solve under each draw so only dispatch
variables adjust. Infeasible draws are dropped and counted, not
silently discarded.

**Uncertainties and their distributions** (`sample_uncertainties()`,
Part 1B's reading, quoted there): demand growth — truncated-at-zero
NORMAL on the growth rate, `demand_sigma=0.01` default; gas regime —
DISCRETE draw from `gas_probability_weights()` (`{"baseline": 0.50,
"downside": 0.25, "upside": 0.20, "shock_recovery": 0.05}`, confirmed
still current against `gas_deliverability_scenarios()` in Part 1B);
carbon price — LOGNORMAL, only when `carbon_active`.

**Who calls it**: nobody. Confirmed zero importers anywhere in `src/`,
`scripts/` (including `deferred`/`retired`), `tests/`, `notebooks/`
(Part 1's module map). There is no `stochastic`-specific `results/`
directory at all — unlike `EaaS.py`/`gas_balance.py`/`postprocess.py`,
none of which have ever been called either but which at least share a
call path with something that HAS produced output (`run_deterministic_
scenario`'s `results/eaas/`), `stochastic.py` has produced literally
nothing, ever.

**Its own limitation note, verbatim** (`src/stochastic.py:31-37`):

> "Limitations
> -----------
> - This measures the robustness of a plan optimized for the BASE case;
>   it does NOT find the plan that is optimal under uncertainty (that
>   would require stochastic programming with recourse). State this
>   explicitly in the thesis.
> - Regime probabilities are a sensitivity prior, not a forecast (see
>   gas_probability_weights). Report symmetric-weight results
>   alongside."

**What wiring it would need**: already answered precisely in Part 2's
addition 1 — confirmed **"a runner away, not a rewrite"**: every
interface (`fixed_plan`'s three keys, `demand_growth`/`gas_scenario`/
`carbon_active`/`carbon_price` scenario keys, `gas_probability_
weights()`) checked directly against HEAD and found current. What's
missing is a `scripts/XX_run_stochastic.py` that builds a scenario via
`load_scenario()`/`load_econ()` through the standard arm surface, calls
`run_stochastic_simulation()`, and writes results — with the one scope
caveat that it only calls the ANNUAL builder (`optimize_model.
build_model`), never the sliced/reporting builder.

### 4. ε-constraint (`18_`, `28_`) — bi-objective by constraint

Full detail already in `docs/task_log_ch36_inputs.md` item C (this
session's own report — point counts, ε grids, duals, binding checks,
non-dominated verification, all quoted from source) and refined by
this document's own Part 2 and Part 4:

- **Method, restated**: one single-objective LP (`system_cost_npv`,
  unconditionally minimised — Part 2 item 15) per swept point on a
  SECOND bound (the emissions cap for `18_`, the public capital
  envelope for `28_`) — never a genuine multi-objective solve. `18_`:
  25 points × 4 financing arms = 100 solves. `28_`: 19 non-uniform
  points × 2 regimes = 38 solves.
- **What Part 2 adds since the §3.6 report**: both frontiers run on
  `build_model_sliced` (`18_`) — Part 2 established the sliced
  builder's hydro/storage/conversion mechanisms are the ones these
  frontiers' points actually reflect, not the annual builder's cruder
  versions; Frontier B (`28_`) explicitly runs `carbon_case="no_
  policy"`, uncapped, so it never touches the emissions side at all
  (confirmed already in Part 2's EaaS/Part 1B discussion).
- **What Part 4 adds**: currency. Both frontiers' `results/` output was
  found current at HEAD in Part 4's per-runner currency check (cross-
  reference Part 4's own entries for `18_run_pareto_frontier.py` and
  `28_run_capital_frontier.py` — both among the small actively-
  maintained September-2026 population, not the August-23 stale
  majority).
- **What it does NOT claim**: a genuine Pareto FRONTIER discovered by a
  multi-objective solver — it is a scan of single-objective solves at
  a grid of ε values, with a post-hoc pairwise non-dominance CHECK
  (quoted in both Part 2 and the §3.6 report) that can and does find
  dominated points (a solver/grid artefact, both docstrings say so
  explicitly), not a guarantee of Pareto-efficiency by construction.

### 5. Anything else addressing uncertainty, robustness or trade-offs

A genuine fresh check, not a re-citation — each item below judged on
whether it produces a FRONTIER/distribution/robustness statement (a
method, in TASK.md's sense) or a plain one-at-a-time point sensitivity
(which item 1 already covers conceptually, and adds nothing new as a
METHOD even if it's a real, reportable result):

- **`scripts/20_run_cf1_solar_yield_sensitivity.py`** — a POINT
  sensitivity (solar capacity factor swept across its `solar_cf`
  range), one arm at a time, no frontier or distribution produced. Not
  a new method.
- **`scripts/27_run_hydro_minflow_sensitivity.py`** — same shape,
  sweeps `hydro_min_flow_fraction`. Not a new method.
- **`scripts/19_run_carbon_price_sweep.py`** — sweeps `carbon_price`
  as a continuous dial rather than the two named `carbon_case` policy
  points; this is CLOSER to a frontier (a cost-vs-carbon-price curve)
  than the CF-1/hydro sensitivities, but it is still one dimension
  swept holding everything else fixed, methodologically identical in
  KIND to Section B of the arm sweep (item 1), just on a continuous
  rather than discrete grid. Not counted as a distinct METHOD, though
  it is the one entry here that produces a curve rather than a table
  of points.
- **`scripts/22_salvage_sensitivity.py`** — a basis-consistency check
  (confirms both arms compared share the same salvage/demand/gas/VoLL
  basis before reporting a premium), not an uncertainty or trade-off
  method at all — closer to a regression-arm discipline check (Part
  2/methodology_protocols.md §6's "one variable at a time" rule) than
  to items 1-4.
- **`scripts/retired/09b_run_rel1_*.py`** (both, retired) — one-at-a-
  time mode/retirement sensitivities on the REL-1 feasibility matrix,
  same shape as the CF-1/hydro-minflow entries; retired, and not a
  distinct method even when they were live.
- **`scripts/deferred/17_run_monte_carlo.py` [revived as `scripts/17_run_monte_carlo.py`, commit `36ddf55`, TASK 06]** — the one genuine
  exception in this section: a JOINT demand-growth × gas-regime
  Monte Carlo that RE-OPTIMISES (not fixed-plan) at every draw — a
  real third method, distinct from both `stochastic.py` (fixed-plan,
  not full re-optimisation) and Sobol (parameter uncertainty on
  physical/cost quantities, not joint scenario-variable resampling).
  Deferred/dead per Part 4's own finding, for the two defects both
  `stochastic.py`'s and `29_`'s docstrings cite it as the cautionary
  example of (LCOE accounting bug; unsourced demand-growth prior) —
  its capability, if those two defects were fixed, is not duplicated
  by anything live today.

**No other uncertainty/robustness/trade-off method found** after this
sweep — the five items above, plus items 1-4, are the complete set.

### What the model is

**The model is deterministic, single-objective at every individual
solve, and perfect-foresight within each solve's horizon — recovering
multi-objective trade-offs only through repeated single-objective
solves, never through a genuine multi-objective solve, and covering
outcome uncertainty only through separate, mostly-unwired methods
layered on top of that deterministic core.** It is deterministic: one
Pyomo LP, one optimal plan per named scenario, no random element in
either builder (Part 2's own reading of both builders' construction;
`src/stochastic.py` and `scripts/29_run_parameter_uncertainty.py` are
the only places `numpy.random`/sampling enters anywhere in `src/`, and
neither is part of the reported central path). It is single-objective
at every solve: `system_cost_npv` alone, unconditionally minimised —
the weighted-sum alternative (`weight_cost`/`weight_emissions`) was
removed and confirmed fully gone (Part 2's removed-capabilities
section). Multi-objective trade-offs (cost-vs-emissions, cost-vs-
capital) are recovered only as ε-constraint SCANS — many single-
objective solves at swept bound values (item 4 above), not a Pareto
solve. It is perfect-foresight within each solve's horizon: `build_
model`/`build_model_sliced` each construct one Pyomo model over the
ENTIRE year range (`T = range(len(scenario["years"]))`,
`optimize_model.py:116-117`, confirmed identical in shape in the
sliced builder) and solve it once, simultaneously, for all years — a
grep of `src/*.py` for "rolling"/"receding"/"myopic" returns zero hits,
confirming no rolling- or receding-horizon mechanism exists anywhere to
contradict this. Outcome uncertainty is addressed by two further,
separately-scoped methods bolted onto this deterministic core rather
than built into it: Sobol parameter-uncertainty decomposition (wired,
current) and fixed-plan stochastic re-dispatch (fully specified,
zero callers, "a runner away" per Part 2).

---

---

## PART 6 — the data layer

64 files under `data/` (`find data -type f | sort`, read in full for this
Part). Grouped by directory family, not file-by-file, except where a
single file's status differs materially from its siblings.

### `data/capacity/raw/` (2 files) — hand-placed, actively read

`gas_plant_availability.csv`, `gas_plant_vintages.csv` — both read
directly by `src/scenarios.py` (`:1219`, `:1134`/`_load_gas_plant_vintages`),
cited in-file as NERC Table 7.1 2024 (`gas_plant_availability.csv`) and
Phase 2.12 plant-by-plant records (`gas_plant_vintages.csv`). Hand-placed
raw NERC extracts, SOURCED, both live and current — no orphan, no
generator-script layer (scenarios.py reads them directly, no
intermediate "processed" step for this family).

### `data/cost/processed/` (10 files) — read by `io.py`, but NO generator
script found for any of the seven cost CSVs

`gas_cost.csv`, `genset_cost.csv`, `om_cost.csv`, `solar_capex.csv`,
`storage_capex.csv`, `unserved_energy_penalty.csv` — all read directly
by `src/io.py` (exact call sites: `:389`, `:454`, `:215`, `:100`/`:394`,
`:158`/`:408`, `:424`). **None of the six has a build/generator script
anywhere in `src/` or `scripts/`** — confirmed by grep for each
filename plus a search for any script writing to
`data/cost/processed/`; only `emissions_cap.csv` (below) has one.
These six are therefore **hand-maintained processed tables**, not
machine-generated from their `raw/` siblings by any code in this
repository — someone transcribes the raw PDFs/workbooks into these
CSVs by hand. Provenance for the transcription is `data/cost/raw/
source_log.md`, read in full: a genuine per-parameter sourcing
document (*"This document records the provenance, assumptions, and
unit conversions used to populate the processed cost inputs in
`data/cost/processed/`"*) covering solar CAPEX, and (skimmed further)
the other cost lines — a real audit trail, just not a machine-checked
one; nothing enforces that the CSV matches what `source_log.md` claims
beyond human diligence at transcription time.

`emissions_cap.csv` — the one exception with a real generator:
`scripts/00_build_emissions_cap.py` writes it directly (`OUT_PATH`,
`:268`, `df.to_csv(OUT_PATH, ...)`, `:534`). **Must be run by hand** —
confirmed not called from any other script or `.sh` harness (grepped
`scripts/*.sh` and all `docs/*.md` for the filename; only prose
mentions, no automated invocation found) — exactly the manual-run
generator pattern Part 4 already flagged for this same script from the
runner side. This session's own earlier work (register L60/L61, this
conversation) already established the currency risk of this exact
file at length; not re-derived here.

`_emissions_cap_PRE_L60.csv` — not a standing data-layer file at all;
it is this session's own untracked snapshot (confirmed: `git status`
shows it as `??`), created as a before/after comparison artefact during
the L60 rebuild verification earlier in this conversation, not part of
the repository's checked-in data layer.

`cost_registry.yaml` — **read by nothing**. Grepped exhaustively
(`grep -rn "cost_registry" src/ scripts/`) — zero hits anywhere. It has
real, well-formed content (`currency: USD`, `base_year: 2025`,
`scenarios: baseline_costs: ...`, read directly) that looks like it was
meant to be a single consolidated index over the six cost CSVs above,
but nothing in the codebase ever loads it. **Orphaned data file.**

`storage_capex.xlsx` — read by nothing (only the `.csv` sibling is
read, per the io.py call sites above). This is the **processed**
workbook version, distinct from the **raw** `data/cost/raw/
storage_capex/2024_v3_Workbook.xlsx` (below); superseded by
`storage_capex.csv`, which is what `io.py` actually consumes.
**Orphaned.**

### `data/cost/raw/` (13 files across the family) — hand-placed sourcing
documents, none read by code, backing `source_log.md`'s citations

`PMS_Report_NOV_2024.pdf`, `carbon/WorldEnergyOutlook2024.pdf`,
`gas_price/*` (5 PDFs — a domestic gas-pricing overview, a heating-value
reference, the Domestic Gas Delivery Regulations 2022, a GenCo debt news
item, and an NMDPRA gas-transport-tariff notice), `solar_capex/*`
(2024_v3_Workbook.xlsx, the IRENA 2023 datafile and its companion PDF),
`storage_capex/2024_v3_Workbook.xlsx`, `unserved_energy_penalty/*` (3
PDFs — an ERSC VoLL working-group item and two African/Nigerian
outage-cost studies) — **all confirmed read by nothing** (none of these
filenames appear anywhere in `src/`/`scripts/`; they are consulted by a
human writing `source_log.md`/the processed CSVs, never opened
programmatically). This is the expected, correct shape for raw
citation-backing documents — flagged as "read by nothing" only because
TASK.md asks for that flag explicitly, not because it's a problem in
itself; these files' job is to be readable by a person, not a script.

**Two exceptions in this family that ARE a genuine finding**:
`conversions.md` (a unit-conversion reference) is also read by
nothing — same category as the rest, unremarkable. But
`fuel_price_monthly_usd.csv` — **a CSV, not a PDF/workbook, sitting
directly in `data/cost/raw/`** — is also read by nothing, which is more
notable for a machine-readable file: either it was superseded by
`gas_cost.csv` before ever being wired in, or a consumer was planned
and never written. Not determinable further from this file alone.

### `data/demand/` (11 files) — the single largest "unread" family, and
explicitly self-documented as such

**Every file in this directory is read by nothing** —
`ELECTRICITY DATA 2015-Q4 2023.xlsx`, `Grid_Generation_per_capita.pdf`,
`README.md`, `README_step1_per_capita.md`,
`demand_base_annualized_2024.csv.xlsx`,
`electricity_access_wdi_diagnostic.csv`,
`electricity_access_worldbank.csv`,
`electricity_generation_ember_diagnostic.csv`,
`electricity_generation_owid.csv`, `nbs_q1_2024_served_energy.csv.xlsx`,
`population_wdi_diagnostic.csv`, `raw/Nigeria_National_Demand_Timeseries.xlsx`,
`step1_per_capita_diagnostic.xlsx` — confirmed by grep, zero hits for
any filename in `src/`/`scripts/`, and zero hits for the three
"_diagnostic" CSVs in `notebooks/*.ipynb` either (checked directly, in
case a validation notebook was the intended reader — it isn't, or at
least isn't any more). **This is not a gap the codebase hides**:
`scenarios.py`'s own docstring for `demand_level_scenarios()` states
outright (quoted in full): *"NOTE: data/demand/demand_base_annualized_
2024.csv.xlsx is STALE — it holds the superseded 23.08 collected-energy
base and is not read by this module."* The Tier-1 demand base
(**37.09 TWh**, "served") and every growth-rate arm in
`demand_growth_scenarios()` are **hardcoded literals in `scenarios.py`**,
transcribed by hand from an external analysis whose workbook
(`data/demand/step1_per_capita_diagnostic.xlsx`,
`README_step1_per_capita.md`) is retained for provenance only — the
docstring names it explicitly as the source (*"Workbook and full
provenance: data/demand/step1_per_capita_diagnostic.xlsx and
data/demand/README_step1_per_capita.md"*) while confirming the
workbook itself is never re-read. **The entire demand-data layer is
therefore a one-time, hand-executed calibration whose output is two
scalars and a handful of growth rates baked into source code** — not a
live pipeline, and the codebase says so about itself rather than this
being an inferred gap.

### `data/gas/` (16 files)

`README.txt` (quoted in full — the whole file): *"Gas production data
must be downloaded manually from NUPRC: https://www.nuprc.gov.ng/
gas-production-status-report/. Place Excel files in: data/gas/raw/.
Processing handled in a separate script."* — a hand-run pipeline,
self-declared, though the README doesn't name which script (see
`build_gas_deliverability.py`, below, as the closest candidate, though
it consumes a YAML config rather than these raw Excel files directly).

`assumptions/*.md` (3 files: `gas_availability_decline.md`,
`gas_heating_value.md`, `gas_to_power_efficiency.md`) — read by
nothing (grepped, zero hits), narrative-only backing for the
corresponding hardcoded constants in `scenarios.py`/`io.py` (e.g.
`gas_eta = 0.287`, Part 3B's ADJUSTMENT-UNSOURCED entries) — same
"correct shape, human-only readership" category as the cost `raw/`
docs above.

`raw/NERC_report/*.pdf` (4 quarterly reports, 2025 Q1-Q4) and
`raw/nuprc_gas_to_power_2025.xlsx` — read by nothing, hand-placed
sourcing documents.

`processed/gas_available_power_annual_twh_th.csv` — **the one clean,
fully machine-generated pipeline in the entire `data/` tree.** Read by
`src/gas_supply.py` (`:25-34`) and `src/scenarios.py` (`:863`);
generated by `scripts/build_gas_deliverability.py`, which itself reads
`processed/gas_deliverability_cases.yaml` (confirmed: `cfg_path =
os.path.join(..., "gas_deliverability_cases.yaml")`, opened directly
with `yaml.safe_load`, `:121-124`) and writes the CSV
(`out_csv`, `:122`). **Still a manual-run generator** — nothing calls
`build_gas_deliverability.py` automatically (Part 4 already covers
this from the runner side); flagged here from the data side too since
it's the clearest instance of "input whose generator must be run by
hand" with a config file one level upstream of the processed CSV.

`processed/export_netback_usd_per_twh.csv` — **read by nothing.**
`src/gas_benchmarks.py`'s LNG netback figures (Part 1's module map) are
computed entirely from hardcoded module-level constants
(`LNG_DELIVERED_USD_MMBTU = 10.05`, etc., `gas_benchmarks.py:66`ff) —
this CSV looks like an earlier or alternate netback data source,
superseded by the hardcoded-constant approach, never wired to the
current benchmark module. **Orphaned.**

`processed/gas_to_power_reconciliation_2025H1.xlsx`,
`processed/gas_to_power_suppy_2025.xlsx`,
`processed/thermal_generation_2025.xlsx`, and the Excel lock file
`processed/~$gas_to_power_suppy_2025.xlsx` (an artefact of the workbook
being open in Excel when last saved, not a data file in its own
right) — **all read by nothing.** These look like working reconciliation
sheets used to sanity-check the gas-to-power series by hand, kept for
reference, not consumed by any script.

### `data/solar/` — an empty directory

Zero files. Solar cost data lives entirely in
`data/cost/processed/solar_capex.csv` / `data/cost/raw/solar_capex/`
instead; this directory appears to be a vestigial placeholder from an
earlier layout, never populated or removed.

### `data/stochastic/` (2 files) — orphaned from the model's side, but
with a real (also-orphaned) generator

`carbon_price_reference.csv` and `raw/carbon_pricing_dashboard.xlsx` —
both read by nothing in `src/`/`scripts/`. This is consistent with Part
1B/2's finding that `src/stochastic.py`'s carbon-price sampling takes
`carbon_mu`/`carbon_sigma` as direct function ARGUMENTS, never reading
a calibration file. **But `carbon_price_reference.csv` does have a
generator** — see `data/make_data.py`, below — it's just that neither
end of that pipeline connects to the live model.

### `data/make_data.py` — a data-acquisition script living OUTSIDE
`scripts/`, called by nothing, fetching from live network endpoints

Not previously covered in Part 4 (scoped to `scripts/` only) or Part 1
(scoped to `src/`) — this is a THIRD location holding executable code,
missed by both prior sweeps until this Part's directory-by-directory
read caught it. Read in full: its own docstring states *"End-to-end
data acquisition and preprocessing pipeline. Thesis-grade, reproducible
version."* Functions: `ensure_dirs`, `safe_read_csv`,
`fetch_owid_demand` (writes to `electricity_generation_owid.csv`),
`fetch_worldbank_access` (writes `electricity_access_worldbank.csv`),
`note_gas_data`, `process_carbon_prices` (reads
`stochastic/raw/carbon_pricing_dashboard.xlsx`, writes
`carbon_price_reference.csv`), `main`. **Confirmed called by nothing**
— grepped `src/`, `scripts/`, and every `.md`/`.sh` file for
"make_data"; zero hits outside the file itself. It fetches from LIVE
URLs (`https://api.worldbank.org/...`, a GitHub-hosted OWID CSV) rather
than a frozen snapshot — even if someone ran it today, it would pull
whatever those endpoints currently serve, not a stated vintage,
which is itself a reproducibility concern independent of the fact that
nothing currently calls it. **This is the clearest "generator that
must be run by hand, and additionally isn't wired to anything the
model reads" case in the whole data layer** — a fully manual,
never-invoked, live-network-dependent pipeline feeding files that are
themselves orphaned at the other end.

### Summary

**Orphaned data files (read by nothing), by family**: all 6 non-cap
processed cost CSVs are LIVE (read by `io.py`) — the orphans are
`cost_registry.yaml`, `storage_capex.xlsx` (processed), all 13
`data/cost/raw/` sourcing documents plus `fuel_price_monthly_usd.csv`
(the one CSV among them), all 11 `data/demand/` files, all 3
`data/gas/assumptions/*.md`, all 4 NERC quarterly PDFs plus
`nuprc_gas_to_power_2025.xlsx`, `export_netback_usd_per_twh.csv` and
the three reconciliation workbooks under `data/gas/processed/`, and
both `data/stochastic/` files. **Generators that must be run by hand**:
`scripts/00_build_emissions_cap.py` (→ `emissions_cap.csv`, live and
read),`scripts/build_gas_deliverability.py` (→
`gas_available_power_annual_twh_th.csv`, live and read, config-driven
from `gas_deliverability_cases.yaml`), and `data/make_data.py` (→ three
files, ALL of which are themselves orphaned at the consuming end — a
manual generator for dead output). **No generator found at all** (as
opposed to "generator exists but must be run by hand") for the six
non-emissions-cap processed cost CSVs — they are hand-transcribed from
`source_log.md`'s sourcing narrative, not machine-built from their
`raw/` siblings by anything in this repository.

---

## PART 7 — structural assumptions and omissions

Not parameters — the assumptions embodied in the model's FORM, derived
from what the two builders' variable and constraint sets contain (Part
2) and, where the absence of an arm is itself evidence (Part 3), from
what they don't. Each point: the assumption, where it's embodied (or
where its absence is confirmed), and its direction of bias.

**Spatial resolution.** Nigeria is a single node. Grepped both builders
for `zone`/`node`/`transmission`/`region` as constraint or variable
names: the only hits are a `selfgen_busbar_factor` comment (below) and
a docstring line about hydro commissioning — no locational variable,
no transmission-capacity constraint, no zonal balance anywhere.
Generation, storage, and demand are summed to one national balance
every period. **Bias**: understates the cost and physical difficulty of
building solar/storage/gas WHERE the grid can absorb or evacuate it —
a plan that is optimal at the national level could require transmission
reinforcement the model never prices or constrains against. No sign is
assertable without knowing whether Nigeria's real bottlenecks favour
solar (typically more distributed, arguably eases this) or gas/hydro
(typically more concentrated, arguably worsens it) — stated as
directionally uncertain, not zero.

**Foresight.** One Pyomo LP is built over the ENTIRE horizon
(2025-2054, `T = range(len(years))`, both builders) and solved once;
every year's investment and dispatch decision sees every other year's
data simultaneously. Grepped `src/` for "rolling"/"receding"/"myopic":
zero hits. **This is perfect foresight**, not sequential/adaptive
planning. **Bias**: UNDERSTATES the cost of uncertainty and OVERSTATES
how efficiently a real planner could sequence investment — a real
planner commits capital without knowing future demand, gas
deliverability, or cost realisations, and would build some insurance
against being wrong that a perfect-foresight plan has no reason to
carry. This is the same gap `src/stochastic.py` (Part 1B, Part 5) exists
to probe and cannot, being unwired.

**Linearity / absent integer structure.** Grepped both builders for
`pyo.Binary`, `pyo.Integer`, and any `domain=` other than
`pyo.NonNegativeReals`: **every decision variable in both builders is
continuous** (`domain=pyo.NonNegativeReals`, confirmed for every `Var`
declaration checked — solar/gas/storage additions, dispatch, curtailment,
tranche volumes, conversion variables). No binary or integer variable
exists anywhere. This single fact implies four absent structures, each
biasing differently:
- **No unit commitment** (no binary on/off state for gas plants) —
  UNDERSTATES the operational cost and emissions of keeping thermal
  plant synchronised/cycling, and OVERSTATES how freely gas generation
  can track a residual-demand shape net of solar.
- **No minimum-stable-generation floor** — UNDERSTATES gas plants'
  real inflexibility at low output and OVERSTATES how far dispatch can
  fall to make room for solar/storage, which is the same direction as
  unit commitment's omission and compounds it.
- **No ramping constraint** — UNDERSTATES the cost/risk of steep net-load
  swings around the solar-driven daytime dip and evening peak (a
  concern the sliced builder's own slice structure is otherwise
  designed to expose, per Part 2 item 1, but ramping WITHIN a slice
  transition is still untouched).
- **No start-up cost, no lumpy/discrete build** — capacity is added in
  arbitrary continuous MW; UNDERSTATES the true cost of a system that
  must round up to the next real, discrete unit size (a plant comes in
  one size, not any size), which generally biases total system cost
  DOWN relative to a world with realistic unit granularity.

**Outage treatment.** There is a firm-capacity DERATE
(`capacity_credit`, Part 2 item 8; Part 3), but reading its
construction directly (`scenarios.py:3038-3039`): **`capacity_credit["gas"]
= _gas_availability_factor` — the SAME scalar used for annual energy
deliverability is reused, unmodified, as the capacity-planning firmness
derate.** This is a PLANNING-LEVEL derate applied to a fixed reserve-
margin CONSTRAINT, not a stochastic forced-outage model with its own
distribution — there is no `forced_outage`/`EFOR`/`outage_rate` concept
anywhere that generates a distribution of outage EVENTS a dispatch
would have to react to within a period. **Bias**: cannot be signed in
general — a single reused derate could overstate or understate true
firmness depending on whether the real correlation structure of gas
outages (which cluster, e.g. during scheduled maintenance windows or
fuel shortages) is better or worse than a flat annual average implies;
what's certain is that the model cannot show a SHORT, sharp outage
event forcing unserved energy or self-generation within an
otherwise-adequate year, since nothing in either builder models outage
TIMING at all.

**Demand elasticity and exogeneity.** `src/demand.py`'s own docstring
states this outright (quoted in full): *"No price elasticity or
sectoral disaggregation"* and, further down, *"No saturation, rebound,
or elasticity effects are modeled."* Demand is a fixed exogenous path,
`base_demand * (1+growth_rate)^t`, entering the electricity balance as
a hard right-hand side every builder builds against — tariff appears
only on the FINANCING side (the EaaS bankability constraint, Part 1B/2
item 12), never as a term demand responds to. **One nuance worth
stating precisely, not conflating**: `scenarios.py`'s demand-GROWTH
arms (Part 3) were themselves CALIBRATED using an income elasticity of
electricity demand w.r.t. GDP (`eps = 0.80`, `scenarios.py:454-458`,
"long-run income elasticity... panel, 1990-2019, income elasticity
averaged 0.8") — this is a real elasticity concept, but it is a
ONE-TIME EXTERNAL INPUT fixing a growth RATE before the model runs, not
a live mechanism inside the LP; it does not make the model
price-responsive. **Bias**: a fixed demand path cannot show a
tariff-driven reduction in load — a high-tariff EaaS/financing scenario
that would, in reality, suppress some demand instead shows up entirely
as a capacity/cost consequence, which OVERSTATES the capacity Nigeria
would need to build under a high-tariff regime and UNDERSTATES demand's
own capacity to adjust.

**System boundary (imports, exports, self-generation, T&D).** No
`import_case`/`export_case` arm exists anywhere in Part 3's 23-arm
inventory, and grepping `scenarios.py` for "import"/"export" as a
scenario dimension returns nothing — **cross-border trade is absent
from the system boundary entirely**, not merely zero-valued by
assumption within an arm. Self-generation (Part 2 item 7) IS modelled,
but as an unbounded backstop OUTSIDE the emissions cap boundary (this
is the F-21/F-31 leakage mechanism this whole session's earlier work
concerned). T&D losses: there is a named concept
(`selfgen_busbar_factor`, `optimize_model.py:415-427`, quoted: *"A
genset delivers at the point of use, downstream of the transmission
loss factor, so 1 TWh of genset output displaces MORE than 1 TWh of
busbar generation... DEFAULT IS 1.0 -- a declared conservative
simplification... It understates the genset by roughly the TLF
(~7%)"*) — but this factor applies ONLY to the genset/busbar
equivalence, defaults OFF (factor 1.0, i.e. transmission losses NOT
credited unless a caller explicitly overrides it), and there is no
general transmission or distribution loss term applied anywhere else
in either builder's main electricity balance — solar, gas, storage,
and hydro all sum directly to demand at busbar with no loss factor in
between. **Bias**: with the default `selfgen_busbar_factor=1.0`, the
model UNDERSTATES how much genset output actually displaces (by
~7%, the TLF itself, per the comment's own arithmetic) — a
conservative bias AGAINST the self-generation/leakage finding, not
inflating it; absent losses elsewhere are a smaller, harder-to-sign
simplification of the same kind as single-node resolution.

**Technology learning.** > **[CORRECTED, TASK 53 2.10, 2026-10-19. As written, the next sentence was FALSE for storage at the date of this inventory: storage CAPEX was a scalar (`econ["STORAGE_COST_PER_MWH"]`, the 2025 `Storage_low` value, 252,000 USD/MWh) at every vintage. `load_storage_capex_by_year` was wired into `build_model` by `72cad20` (2026-03-21) and deleted from it by `96b50cf` (2026-04-03, "large refactor with results"); it had no caller from then until TASK 53. The inventory's "Part 1/Part 6" confirmed the CSVs and the loader existed, not that anything called the loader. Since TASK 53 2.2 the statement is TRUE: `storage_capex_case="declining_nrel"` is the `load_scenario()` default and both builders read `m.storage_capex_param[t]`, indexed by build (and rebuild) year as solar is. `flat` remains available as the declared sensitivity. Register L90.]**
Solar and storage CAPEX are read as
BY-YEAR series (`load_solar_capex_by_year`/`load_storage_capex_by_year`,
Part 1/Part 6), and Part 6 confirmed these processed CSVs are
hand-maintained, not model-generated — the by-year values DO decline
over the horizon in the data (an exogenous, time-indexed cost trajectory
consistent with an assumed learning curve), but **no constraint or
expression anywhere ties `solar_capex_param[t]` (or its storage
equivalent) to any quantity the model itself has built** — confirmed
by reading both capex terms' construction in Part 2 (`m.DF[t] *
m.solar_public_add[t] * m.solar_capex_param[t]`, a straight read of an
exogenous per-year parameter, no lagged-capacity term anywhere in the
expression). **Exogenous, time-varying cost decline exists; ENDOGENOUS
(deployment-linked) learning does not** — the model's own build decisions
never feed back into the unit cost it or any other scenario faces
later. **Bias**: if real-world deployment (in Nigeria or globally) would
respond to how much THIS plan builds, the model cannot capture that
feedback in either direction — understating cost declines a large
domestic programme might help induce, or, just as plausibly, ignoring
supply-chain cost pressure a large build-out might create. Not
signable from the code alone; flagged as an omission, not a biased
one.

**Policy feedback.** The NDC cap trajectory (`00_build_emissions_cap.py`,
run by hand, Part 6) and the carbon price (`carbon_policy_scenarios()`,
Part 3) are both fixed scenario inputs, constructed BEFORE the solve
and passed into `build_model`/`build_model_sliced` as arguments — never
computed from within the LP, and never fed back into a subsequent solve
within the same run. There is no rule anywhere of the form "if leakage
exceeds X%, tighten the cap further" or "if cost exceeds Y, relax the
carbon price" — policy is exogenous throughout. **Bias**: cannot be
signed in general (a real policymaker might tighten OR loosen in
response to observed cost/leakage, and which the model's silence
favours depends on which direction the counterfactual policymaker
would have moved) — but it does mean every "the cap causes leakage"
finding in this project (F-21, F-31, this session's L60/L63 work) is a
COMPARATIVE-STATICS result across pre-specified cap paths, never a
finding about how policy would actually evolve in response to the
leakage it discovers.

**Capacity-credit treatment.** Restated in structural-assumption terms
from Part 2 item 8: each technology's contribution to the firm-capacity/
reserve-margin constraint is a FIXED, exogenous derate (100% for
storage's power rating; `_gas_availability_factor` reused for gas, per
the outage-treatment point above; near-zero/`[PENDING]` for solar per
Part 3) — none of these derates is ITSELF estimated from any
loss-of-load or reliability calculation internal to this model. Firm
capacity is an assumption fed to the LP, never a result the LP could in
principle revise (e.g. by discovering that a technology mix it
otherwise favours would need a different derate to actually keep the
lights on at the assumed reliability standard).

**Two more, from what else the builders' form assumes**:

- **No fuel-price uncertainty within a single solve.** Gas price is a
  scenario-level arm (`gas_price_case`, resolved once via
  `load_econ()` before the solve), constant for the entire horizon
  within one run — it cannot represent a gas-price shock arriving mid-
  horizon inside a single optimisation the way, say, the sliced
  builder's intra-year gas allocation can react to seasonal
  deliverability. Distinct from (and narrower than) `stochastic.py`'s
  unwired ability to VARY gas regime ACROSS separate solves (Part 5) —
  within any one solve, price is a constant.
- **No explicit water-year / climate trajectory for hydro beyond the
  fixed `hydro_case` arm.** Hydro's annual energy and (in the sliced
  builder) seasonal shares are exogenous named scenarios (Part 2 item
  6, Part 3), not a stochastic hydrological process or a climate-
  trend adjustment over the horizon — a run-of-river-style planning
  assumption that treats historical hydrology as stationary across
  2025-2054.

No single directional bias summarises Part 7 as a whole — some
omissions push cost/capacity DOWN (no lumpy build, no ramping/start-up
cost, no learning feedback risk), some push it UP or are ambiguous
(single-node, outage timing, T&D losses net of the conservative
`selfgen_busbar_factor` default), and demand exogeneity specifically
overstates capacity NEED under high-tariff scenarios. Each is stated
with its own direction above rather than netted into one claim the
formulation does not support.
