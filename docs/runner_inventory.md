# Runner inventory — what every script asks, and whether it still works

**Read-only inventory. Nothing in `src/` or `scripts/` was edited or
executed.** Model frozen at HEAD `2a9226b`. Written incrementally, one
runner at a time, per TASK.md's instruction.

**Required-key reference** (from `src/scenarios.py:load_scenario()`,
read once, used throughout this report rather than re-derived per
runner): 17 keys are sentinel-required (raise `TypeError` if omitted) —
`tariff_case`, `peak_multiple_case`, `reserve_margin`, `storage_cc_case`,
`gas_build_case`, `gas_availability_case`, `gas_retirement_case`,
`gas_capex_case`, `ccgt_efficiency_case`, `capital_recovery_case`,
`dscr_case`, `us_inflation`, `capital_treatment_case`,
`concessional_envelope_case`, `conversion_case`, `hydro_case`,
`om_case`. Eleven further parameters have real defaults in
`load_scenario()` itself (`demand_level_case`, `demand_case`,
`land_case`, `capital_case`, `gas_deliverability_case`,
`solar_build_case`, `carbon_case`, `blend_case`,
`new_gas_deliverability`, `start_year`, `end_year`) — omitting one of
these does not raise; it silently takes the default, which is a
different, milder defect class than the sentinel set (worth flagging
per-runner but does not make a runner "dead").

---

## GROUP 1 — reported PASSING

### `00_build_emissions_cap.py`

**Question.** No single stated question; module docstring names two
research questions instead: *"RQ-POL-1: What is the investment
trajectory and cost differential between unconditional and conditional
NDC 3.0 targets in the power sector? RQ-POL-2: What is the implied MAC of
moving from unconditional to conditional NDC 3.0 target in the power
sector — and how does EaaS change that?"* This script itself only builds
the cap series those questions are answered against; it does not answer
them.

**Sweeps.** Not CLI-driven. Builds five named cap *scenarios* internally
(a Python dict, not an arm registry): `ndc3_unconditional`,
`ndc3_conditional`, `ndc2_unconditional`, `ndc2_conditional`, and
`ndc3_conditional_decline` (plan 4.2, added post-freeze — see register
L24).

**Fixes.** NDC 3.0/2.0 policy constants hardcoded at module level (2018
baseline 573.5 MtCO2e, 2030/2035 targets 168.2/184.9 MtCO2e, energy-sector
shares 22.0%/26.3%, energy-sector total 217.0 MtCO2e) — these are cited
policy-document figures, not model configuration, so hardcoding them is
appropriate; none is a swept arm elsewhere.

**Solves.** None. Reads `results/baseline/diagnostics.json` for baseline
gas-to-power TWh_th by year; does no optimisation itself.

**Writes.** `data/cost/processed/emissions_cap.csv` — 5 scenarios × 30
years, no arm tag (not a results-summary file, a static input the
downstream runners consume by scenario name).

**State: PASSING**, consistent with the reported group. No CLI, no
required-key surface at all — it fails only if
`results/baseline/diagnostics.json` is missing (raises
`FileNotFoundError` naming `01_run_baseline.py` as the fix).

**Declared defects.** None cite this file by number. Its two-layer
construction (apportionment to gas-power, then post-2035 shape) is the
subject of register **L24** and plan items 4.1/4.2.

**Epoch signals.** None pre-O&M — this file never touches cost, only
emissions. `NET_ZERO_YEAR = 2060` is current-epoch (added for plan 4.2,
after the freeze).

**Dependencies.** Reads `results/baseline/diagnostics.json`, produced by
`01_run_baseline.py`. Written CSV is read by `02_run_ndc_caps.py`
(confirmed) and, by scenario-name convention, likely by
`03_run_ndc_eaas.py`, `08_run_gas3_regime_ndc_feasibility.py`, and
`15_run_pol1_ndc_comparison.py` — not yet confirmed, will check when each
is reached below.

**Epoch check on the dependency (checked directly, disk not gitignored
from reads).** `results/baseline/diagnostics.json` on disk is dated
2026-08-23 and its companion `summary.json` has no `om_case`, no
`conversion_case`, no `capital_treatment_case` key at all — a schema from
before F1 and before conversion existed, let alone O&M. A second,
arm-tagged baseline file dated 2026-08-28 exists
(`diagnostics__..._pre_f1_overnight...json`) and is explicitly
`pre_f1_overnight`, also pre-O&M. **This runner will currently regenerate
the cap series from a stale baseline** unless `01_run_baseline.py` is
rerun first — worth flagging even though this script itself has no
defect.

---

### `08_phase8_analysis.py`

**Question.** No docstring at all — no module comment, no header. The
closest thing to a stated question is the inline comment describing
`CANONICAL_VOLL`: prevents "unserved-energy penalty from dominating total
system cost" so "cross-case comparisons [are] valid." Otherwise the
intent must be inferred from its five pipeline stages (master table,
cost/MAC frontier, mechanism decomposition, scarcity stats, EaaS-vs-carbon
table) — this absence is itself a finding.

**Sweeps.** None — a fixed `CASES` list of 9 named result directories
(baseline, 4×NDC public-only, 4×NDC+EaaS), all at `CANONICAL_VOLL =
"voll_mid"` / `CANONICAL_GENSET = "genset_central"`.

**Fixes.** Every arm implicitly, by reading whatever `summary.json` /
`timeseries.csv` already say at 9 hardcoded flat paths, e.g.
`results/ndc/ndc3_unconditional_voll_mid/summary.json`.

**Solves.** None. Pure aggregation over existing result files.

**Writes.** `results/phase8/` — 5 CSVs (`master_results`,
`frontier_points`, `marginal_abatement_cost`, `eaas_financing_gap`,
`decomposition_table`, `scarcity_stats`, `energy_security_dashboard`,
`subsidy_vs_carbon`) plus 2 plots. No arm tag (paths are the tag).

**State: technically PASSING but on stale, mismatched data.** Two
independent problems, checked directly: (1) `02_run_ndc_caps.py` now
writes to hash8-named directories (`docs/output_naming.md` pattern A),
not the flat `<case>_voll_mid` paths this script hardcodes — though a
flat-named directory still happens to exist on disk for some cases
(`results/ndc/ndc2_unconditional_voll_mid/`), it is not what current runs
produce. (2) That surviving directory's `summary.json` has `om_case:
None`, `arm_tag: None`, `gas_retirement_case: None`, and
`npv_total_cost_usd: 81,374,741,735` — roughly 3× the current central
objective, from an epoch that predates every required-key discipline
currently in force. The script does not error (`if not p.exists():
WARNING ... continue`), so a run today would silently mix whatever stale
directories happen to exist with warnings for the rest, and report
numbers 3× off scale as if current.

**Declared defects.** None cite this file by number.

**Epoch signals.** `results/ndc/ndc2_unconditional_voll_mid/summary.json`
— 81,374,741,734.86 USD, no `om_case` key (see State above).

**Dependencies.** Reads `results/baseline/`, `results/ndc/*_voll_mid/`,
`results/ndc_eaas/*_voll_mid/` — all producer runners' CURRENT output
naming has moved on from what this script expects.

---

### `14_run_str1_storage_role.py`

**Question.** Quoted from the module docstring: *"STR-1 asks what role
battery storage plays in the optimal system."* Immediately answered:
"It cannot be answered in the current model, and this runner is
intentionally disabled."

**Sweeps / Fixes / Solves.** None of any of these — `main()` prints one
message and calls `sys.exit(0)`. No `load_scenario()` call, no builder
import beyond nothing.

**Writes.** Nothing.

**State: PASSING by design** — it is meant to do nothing and exit 0, and
it does. But **its own stated re-enable condition appears to already be
satisfied and unactioned**: the docstring says re-enable "when Phase 2
introduces representative intra-annual slices" — the sliced builder is
now the reporting model (`methodology.md` §2: "the sliced builder is the
reporting model"), and storage now builds a nonzero central quantity
(65,613.0 MWh). This is a descriptive fact about the gap between the
runner's stated condition and the model's current state, not a judgement
about whether STR-1's question is worth re-asking.

**Declared defects.** Names its own prior defects: the retired
`storage_solar_surplus_frac` coupling (unsourced, removed) and
`STORAGE_OM_PER_MWH_YR` as a no-op sweep axis — the same constant
register **L22** documents as retired for having zero readers.

**Epoch signals.** None numeric — purely structural/prose.

**Dependencies.** None.

---

### `16_plot_cost_decomposition.py`

**Question.** Docstring states the figure's purpose directly: decompose
NPV system cost into "Real expenditure," "Private selfgen," and "VoLL
penalty," updated for 2.6 to argue EaaS savings come from avoided diesel
spend, not avoided blackout penalty.

**Sweeps.** None — 5 fixed cases (public/EaaS × unconditional/conditional
NDC3, plus baseline), at `CANONICAL_VOLL = "voll_mid"`.

**Fixes.** Same flat-path convention as `08_phase8_analysis.py`
(`results/ndc/ndc3_unconditional_voll_mid/diagnostics.json`, etc.) — same
naming-mismatch risk against current hash8-named output.

**Solves.** None. Reads `diagnostics.json`'s `cost_decomposition` field
plus each case's `summary.json` for the solved objective.

**Writes.** `results/figures/cost_decomposition.{png,pdf}` plus a console
reconciliation table.

**State: has a real, currently-live defect, not just stale paths.**
Checked the source of `cost_decomposition` (`src/optimize_experiments.py`,
the function building it, read-only): `cost_real_expenditure = gas_opex +
gas_capital + pub_solar_capital + eaas_solar_capital + storage_capital +
carbon` — **O&M is not a term in this sum**, under either capital
treatment branch. This script's own built-in reconciliation check (`total
= real + selfgen + voll − salvage`, compared against the solved
`npv_total_cost_usd`) would therefore **fail with a MISMATCH flag on any
`om_central` or `om_low`/`om_high` result** — the objective includes O&M,
the decomposition it's checked against does not. This is a `src/`-level
defect (not fixable here, read-only), surfaced through this runner's own
guard rather than found independently — the guard is working as designed.

**Declared defects.** None cite this file by number, but the mismatch
above is a live consequence of the O&M implementation not being
back-propagated into `optimize_experiments.py`'s cost decomposition.

**Epoch signals.** `CANONICAL_VOLL = "voll_mid"` (unaffected by O&M).
The O&M-omission above is itself the epoch signal — this function was
last updated before O&M existed and has not been touched since.

**Dependencies.** Same flat-path dependency issue as `08_`, on
`results/{baseline,ndc,ndc_eaas}/*_voll_mid/diagnostics.json`.

---

### `27_run_hydro_minflow_sensitivity.py`

**Question.** Quoted: *"how much does the binding hydro min-flow floor
move the system?"* Full context: `hydro_min_flow_fraction` is ASSUMED,
not SOURCED, and is verified to BIND at the central sliced configuration
— an assumed, load-bearing, binding parameter needs a sensitivity band.

**Sweeps.** `hydro_min_flow_fraction` only, `{0.30, 0.50 (central), 0.70}`
— explicitly NOT a required key or arm-registry entry; mutated directly
on a deep-copied scenario dict.

**Fixes.** A full `CENTRAL_ARGS` dict, all 21 `load_scenario()`
parameters, read verbatim from `results/baseline_sliced/summary__02c17f55.json`
by the script's own account. Notably `capital_case="unconstrained"`,
`solar_build_case="deployment_unconstrained"`, `land_case="loose"` —
deliberately the unconstrained-baseline configuration, not the NDC
capital-tight/moderate configuration. **`om_case="om_none"` is a
documented, deliberate deviation** from `om_central` — the script's own
comment explains its reproduction guard is pinned to the pre-O&M anchor
specifically so this sensitivity isn't confounded by an unrelated O&M
change. This is the correct, self-aware use of an epoch-pinned regression
arm, not an oversight.

**Solves.** 3, `build_model_sliced` (sliced/reporting builder).

**Writes.** `results/hydro_minflow/hydro_minflow_<value>.json` ×3, a
long-format sweep CSV, and a materiality verdict JSON. No single
summary-with-arm-tag file, but every output embeds the full
`hydro_min_flow_fraction` value.

**State: PASSING**, and unusually well-instrumented: an internal control
(`assert_scenarios_differ_only_in_hydro_min_flow`, structural not `!=`)
and a reproduction guard against a hardcoded reference objective, both
asserted before the materiality verdict is computed.

**Declared defects.** This IS the source script for register **L20**
(`hydro_min_flow_fraction` ASSUMED, binding, immaterial) —
materiality thresholds (0.5% objective, 2% storage@2045) fixed in this
script before any solve, exactly as L20 describes.

**Epoch signals.** `REFERENCE_NPV_TOTAL_COST_USD = 21351610495.345654`
— the pre-O&M, post-shape-sourcing central anchor. **Deliberately
pre-O&M** (see Fixes above), not stale.

**Dependencies.** References `results/baseline_sliced/summary__02c17f55.json`
by value only (hardcoded, not read at runtime — no filesystem
dependency).

---

### `99_validate_baseline.py`

**Question.** No explicit question sentence; docstring states purpose:
*"Cross-checks the baseline LP's first-year (2025) output against actual
2024 generation statistics from NERC / NBS to confirm the model is
anchored to reality."*

**Sweeps / Fixes.** None — no `load_scenario()` call at all. Reads a
single hardcoded `OBSERVED_2024` dict and one row of
`results/baseline/timeseries.csv`.

**Solves.** None.

**Writes.** `results/validation/validation_table.json` plus a console
table.

**State: runs without crashing, but validates against two retired
quantities — likely produces a misleading result, not a correct one.**
`OBSERVED_2024["demand_twh"] = 23.08` is the **specifically-retired**
demand base — `methodology_demand_horizon.md` §1.1: "The base was
previously 23.08 TWh... Using it understated system scale by roughly
40%," corrected to 37.09 TWh. This script still validates the model's
current (~37 TWh-scale) demand against the old, rejected 23.08 TWh figure
— the "gap" it reports is now mostly the demand-base correction, not
model error. Separately, `OBSERVED_2024["unserved_twh"] = 3.0` predates
the genset backstop (2.6): `unserved` is pinned at ~0 by construction
under any genset case (`methodology_genself_backstop.md` §1.4), so this
script's unserved check will always report close to a −100% gap against
an "observed" figure that is no longer the right comparison at all.

**Declared defects.** None cite this file by number.

**Epoch signals.** `demand_twh: 23.08` (retired, pre-correction);
`unserved_twh: 3.0` (pre-backstop, 2.6). Both are the two oldest,
most-superseded figures found anywhere in this inventory so far.

**Dependencies.** `results/baseline/timeseries.csv` — same stale baseline
identified under `00_build_emissions_cap.py` above (no `om_case`,
2026-08-23).

---

## GROUP 2 — argparse rejection when called bare

### `01_run_baseline.py`

**Question.** No stated question; `argparse` description is
`"01_run_baseline: unconstrained baseline solve"` — the un-capped,
un-EaaS'd reference case every other runner is checked against.

**Sweeps.** None of its own — 18 required CLI flags, one per
`load_scenario()` required key (all 17 sentinel keys plus
`--gas-deliverability-case`, which has a real default in
`load_scenario()` but is promoted to required here anyway, following the
project's stated policy of requiring some non-sentinel keys too).

**Fixes.** `demand_level_case="served"`, `demand_case="organic_central"`,
`capital_case="unconstrained"`, `solar_build_case="deployment_unconstrained"`,
`land_case="loose"`, `carbon_case="no_policy"`,
`reserve_margin="nirp_spinning"` — all hardcoded, matching register
**L9**'s documented pattern (`reserve_margin` is a canonical constant, not
swept, by deliberate choice).

**Solves.** 1, `build_model` (annual).

**Writes.** `results/baseline/{diagnostics,summary,timeseries}__<tag>.json/csv`,
where `<tag>` is the **raw, unhashed concatenation of 15 arm values** —
no `hash8` shortening.

**State: argparse-only in the reported sense, but carries a live latent
defect.** Computed the central-configuration tag directly:
`diagnostics__peak_unsuppressed__..._hydro_flat.json` is **253
characters** — inside the ~255-byte per-path-component limit
`01b_run_baseline_sliced.py`'s own docstring says it hit (`OSError
[Errno 36]`) with the *same 15-field tag structure*, but only by 2
characters, for this exact arm combination. `01b` was fixed with a
`hash8_of()` shortener after hitting this; `01` was never given the same
fix. Any arm combination with longer-named choices (e.g.
`gas_retire_vintage_50` vs `_30`, or a longer envelope/dscr name) would
overflow it. Not verified by running — reported as a computed risk, per
this task's read-only bound.

**Declared defects.** None cite this file by number, but it is the
producer `00_build_emissions_cap.py`'s error message names when
`results/baseline/diagnostics.json` is missing.

**Epoch signals.** None hardcoded — all cost/objective figures come from
the live solve.

**Dependencies.** None upstream. Downstream: `00_build_emissions_cap.py`
reads `results/baseline/diagnostics.json` (the **unwrapped**, no-tag
path this script does not write directly — see the epoch note under
`00_` above: the file that exists there is dated 2026-08-23 and
pre-dates several required keys entirely, meaning either an even older
version of this script wrote it, or it was written by hand/copied).

---

### `01b_run_baseline_sliced.py`

**Question.** No stated question; description
`"01b_run_baseline_sliced: unconstrained sliced baseline solve"` — the
sliced-builder counterpart to `01_`, and per `methodology.md` §2 this
IS the reporting model.

**Sweeps.** None of its own — same 18 required CLI flags as `01_`, one
addition to the conversion-case help text: "An available arm requires
`--capital-treatment-case f1_tranche`."

**Fixes.** Identical fixed set to `01_run_baseline.py`
(`demand_level_case="served"`, `capital_case="unconstrained"`,
`solar_build_case="deployment_unconstrained"`, `land_case="loose"`,
`reserve_margin="nirp_spinning"`).

**Solves.** 1, `build_model_sliced`.

**Writes.** `results/baseline_sliced/{summary,diagnostics,timeseries,slice_detail}__<hash8>.json/csv`
— **hash8-shortened** (the fix `01_` lacks), full tag preserved as
`arm_tag` inside each JSON and as a CSV header comment, plus an
`INDEX.md` mapping hash8 back to the full tag.

**State: PASSING-shaped, well-built.** Prints an explicit backcast check
(gas 2025 vs observed 25.62 TWh_e) every run, and computes F2
concessional-volume/share fields inline since `extract_planning_diagnostics`
is annual-only and cannot be reused on sliced `[t,(s,p)]` indexing. This
is the runner `27_run_hydro_minflow_sensitivity.py`'s `CENTRAL_ARGS`
dict is built to reproduce.

**Declared defects.** L15 (stale "2045" naming) is corrected here at the
source: `slice_detail_terminal_year` (renamed from `slice_detail_2045`),
with an inline comment citing the correction and `docs/task_hydro_gate_bucketA.md`
as having confirmed nothing else reads the old key.

**Epoch signals.** None hardcoded — `25.62` (observed gas) is a citation
constant, not a stale model output.

**Dependencies.** None upstream. Downstream: this is the producer for
`results/baseline_sliced/summary__02c17f55.json`, the file
`27_run_hydro_minflow_sensitivity.py` reads its `CENTRAL_ARGS` from (by
value, hardcoded — see that entry above).

---

### `02_run_eaas.py`

**Question.** No stated question; description
`"02_run_eaas: EaaS-financed solar solve"`.

**Sweeps.** 17 required CLI flags (16 of the 17 sentinel keys, plus a
17th) — **`--gas-deliverability-case` is conspicuously absent** from the
CLI surface here, unlike `01_`/`01b_`.

**Fixes.** `gas_deliverability_case="baseline"` **hardcoded directly in
the `load_scenario()` call**, not exposed as a flag — the one arm this
sibling runner promotes to required, this one fixes silently. Also fixes
`capital_case="moderate"` (vs. `01_`'s `"unconstrained"` — appropriate,
since this is a capital-constrained EaaS comparison, not the
unconstrained baseline) and `land_case="loose"`, `solar_build_case="deployment_unconstrained"`.
Sets `scenario["financing_regime"] = "eaas"` post-hoc, enabling the EaaS
channel.

**Solves.** 1, `build_model` (annual — not sliced, unlike `01b_`/`02_run_ndc_caps.py`).

**Writes.** `results/eaas/{diagnostics,summary,timeseries}__<tag>.json/csv`,
tag from 13 fields — **no hash8**, same latent length risk as `01_`, not
separately computed here (fewer fields than `01_`'s 15, so likely under
the limit, but not verified).

**State: argparse-only, consistent with the reported group.** One dead
code path noticed: a `tariff_grid` list (`[60M..110M]`) is built at the
very end of `main()` and never used — no sweep, no write references it.

**Declared defects.** None cite this file by number. The docstring for
`--tariff-case` explicitly documents the retirement of the bare
95,000,000 literal this runner used to hardcode (`docs/tariff_arm.md`).

**Epoch signals.** None hardcoded.

**Dependencies.** None found.

---

### `02_run_ndc_caps.py`

**Question.** No stated question; description
`"02_run_ndc_caps: public-only NDC cap solves"`.

**Sweeps.** 18 required CLI flags (all 17 sentinel keys +
`--gas-deliverability-case`), plus one **opt-in, non-required** flag,
`--include-decline-arm` (`store_true`, default off — added for plan 4.2;
see `docs/task_cap_trajectory.md`). Internally iterates a fixed
`NDC_CASES` dict — 4 entries by default (`ndc2_unconditional/conditional`,
`ndc3_unconditional/conditional`), a 5th (`ndc3_conditional_decline`)
appended only if the opt-in flag is passed.

**Fixes.** `demand_level_case="served"`, `demand_case="organic_central"`,
`carbon_case="no_policy"`, `reserve_margin="nirp_spinning"`. Per-case
`capital_case` (`"tight"` for unconditional, `"moderate"` for
conditional, `"moderate"` again for the decline arm) fixed in
`NDC_CASES`, not CLI-swept — documented rationale in an inline comment
(apportioned power-sector share of NDC 3.0's $270bn).

**Solves.** 4 (or 5 with the opt-in flag), `build_model_sliced`, one per
`NDC_CASES` entry.

**Writes.** `results/ndc/<hash8>/{summary,diagnostics,timeseries}.json/csv`
— hash8-shortened, full tag in `arm_tag` + `INDEX.md`. Every entry
records `om_case` and `tariff_case` for auditability though this runner
is public-only (no EaaS channel) so tariff is inert.

**State: PASSING** (Group 2 assumption confirmed) — this is the runner
this session's own prior task edited to add the plan-4.2 decline arm;
re-verified the CLI/`NDC_CASES` wiring matches that edit (`grep` for
`--include-decline-arm` and `NDC_CASES["ndc3_conditional_decline"]`, both
present as expected).

**Declared defects.** None cite this file by number.

**Epoch signals.** None hardcoded — no fixed objective values.

**Dependencies.** Reads `data/cost/processed/emissions_cap.csv`
(`00_build_emissions_cap.py`). Downstream: `08_phase8_analysis.py` and
`16_plot_cost_decomposition.py` expect its output at flat
`ndc*_voll_mid` paths this runner does NOT write (see their entries
above — naming mismatch, not this runner's defect).

---

### `03_run_ndc_eaas.py`

**Question.** No stated question; description
`"03_run_ndc_eaas: EaaS-financed NDC cap solves"` — the EaaS-enabled
counterpart to `02_run_ndc_caps.py`.

**Sweeps.** 17 required CLI flags (all 17 sentinel keys; no separate
`--gas-deliverability-case`, fixed instead — see Fixes). Internally, a
fixed 4-entry `NDC_CASES` dict pairing each cap scenario with a
`capital_case`.

**Fixes.** `gas_deliverability_case="baseline"` — this sibling of `02_`
does NOT expose it as CLI, unlike `02_run_ndc_caps.py`. Per-case
`capital_case` (`ndc2_unconditional_eaas`: `"tight"`,
`ndc2_conditional_eaas`: `"moderate"`, `ndc3_*`: `"tight"`/`"moderate"`).
`scenario["public_solar_budget_npv"] = None` post-hoc — deliberately
unconstrained, documented rationale: at the tariff arms this runner
carries, the financing gap is zero and EaaS is fully self-financing, so
an unconstrained public budget is the correct test, not an oversight.

**Solves.** 4, `build_model_sliced`.

**Writes.** `results/ndc_eaas/<hash8>/{summary,diagnostics,timeseries}.json/csv`
— hash8-shortened (this runner's own comment records hitting the same
`OSError [Errno 36]` `01b_`/`02_` hit, "after solving all four NDC
cases" — i.e. discovered expensively, mid-sweep). Adds
`bankability_slack_per_mw_by_year` — a per-year USD/MW slack on
`m.eaas_bankability`, explicitly new: "No such reader existed anywhere
before this task."

**State: PASSING**, well-instrumented (the bankability-slack diagnostic
makes the "is EaaS tariff-constrained this year" question directly
readable rather than inferred).

**Declared defects.** None cite this file by number.

**Epoch signals.** None hardcoded.

**Dependencies.** Reads `data/cost/processed/emissions_cap.csv`. Same
downstream naming-mismatch exposure as `02_` for `08_`/`16_`.

---

### `04_run_fin2_tariff_sweep.py`

**Question.** Quoted from the docstring (RESEARCH QUESTION, FIN-2):
*"What is the minimum EaaS tariff at which private capital becomes
self-financing, what subsidy (`eaas_subsidy`) does the model require
below that tariff, and how does the threshold shift with capital
structure (blended WACC, via `--concessional-envelope`) and finance
conditionality (NDC cap scenario)?"*

**Sweeps.** `TARIFF_SWEEP_GRID` (imported from `src.scenarios`, not
re-declared here) × `blended_finance_scenarios()` × a fixed 5-entry
`NDC_CASES` dict (`ndc2_*`, `ndc3_*`, `baseline_no_policy`). CLI:
13 required flags (13 of the 17 sentinel keys), plus two **non-required**
flags, `--concessional-envelope` (default `"unbounded"`) and
`--deployable-fraction` (default `1.0`).

**Fixes.** `om_case="om_central"` **hardcoded inside the per-case
`load_scenario()` call — no `--om-case` flag exists on this runner at
all.** `gas_deliverability_case="baseline"` also hardcoded, no flag.

**Solves.** `len(NDC_CASES) × len(TARIFF_SWEEP_GRID) × len(blend_scenarios)`,
via `run_fin2_blended_bankability_sweep()` (`src/optimize_experiments.py`)
— not called directly here, delegated.

**Writes.** `results/fin2/tariff_sweep_results__<tag>.{csv,json}` and
`tariff_sweep_summary__<tag>.json`, tagged by envelope arm +
deployable-fraction + the 13 required arms (no hash8 — not checked for
length here, given the extra 13-field tag this is a real risk, not
verified).

**State: NOT actually healthy — refutes the Group 2 "probably healthy"
premise, confirmed two independent ways.** (1) Its own `load_scenario()`
call (main body, per-`NDC_CASES`-entry) **omits three of the 17
sentinel-required keys entirely** — `conversion_case`, `hydro_case`,
`tariff_case` are never passed, despite this being the tariff-bankability
script. Supplying every one of this script's own 13 required CLI flags
would therefore still raise `TypeError` from inside `load_scenario()` —
this script cannot currently run to completion under any invocation. (2)
Register **L17** independently corroborates, in its own words: *"Known
second instance, same defect. `run_fin2_blended_bankability_sweep`
(`src/optimize_experiments.py:853, 887, 907-908, 981`) discounts at the
same pre-F1 blend in both its analytical T* and its solves. Its runner
`04_run_fin2_tariff_sweep.py` is dead on unrelated required args."` L17's
"dead" classification and this session's direct read of the missing
`load_scenario()` arguments agree. **This runner belongs in a
"argparse-complete but still dead" category TASK.md's four groups don't
have a slot for** — Group 2 assumed argparse rejection implies
"probably healthy"; here it does not.

**Declared defects.** Register **L17** (quoted above) and the module's own
docstring documents a second, separate defect: `--concessional-envelope`
(the original flag) is now **inert** — its underlying mechanism was
retired from both builders, but the flag, its choices, and its
`CONCESSIONAL_ENVELOPE_ARMS` dict remain, silently accepting a value that
does nothing. The similarly-named but mechanically distinct
`--concessional-envelope-case` is the live F1 mechanism.

**Epoch signals.** None hardcoded numerically, but the whole T*/subsidy
analytical apparatus this script exists to test is the subject of L17's
pre-F1-blend defect (see Declared defects).

**Dependencies.** Reads `data/cost/processed/emissions_cap.csv` (via
`run_fin2_blended_bankability_sweep`'s `cap_path` argument).

---

### `22_salvage_sensitivity.py`

**Question.** Quoted from the title: *"Is the investment trajectory a
residual?"* Tests whether the capacity trajectory RESPONDS to cost or is
determined entirely by the energy balance (an "accounting residual") —
"the scheduled test written into correction plan 2.5."

**Sweeps.** 6 hardcoded asset-lifetime arms (`no_salvage`, `solar_25`,
`central`, `solar_35`, `storage_12`, `storage_18`), applied via
`scenario["asset_lifetimes"]` mutation — not a `load_scenario()` arm.
CLI: 17 of the 17 sentinel keys **except `--om-case`**.

**Fixes.** `capital_case="unconstrained"`,
`solar_build_case="deployment_unconstrained"`, `land_case="loose"`,
`om_case="om_central"` hardcoded (no CLI flag — same pattern as `04_`,
smaller in degree: here the key IS passed, just not swept).

**Solves.** 12 — 6 lifetime arms × 2 builders (annual AND sliced, run
side by side deliberately, since the finding differs by builder per
register **L5**).

**Writes.** `results/salvage_sensitivity/{trajectories,salvage_sensitivity}__<tag>.csv/json`,
tag from 15 fields, no hash8 (length risk not computed).

**State: PASSING (Group 2 confirmed), but its own re-run trigger has
fired twice since it last ran.** The docstring names three events that
require re-running it: "2.6 genset backstop," "5.3 freeing `gas_add`,"
and the min/max-build floor changes — all landed. Checked the actual
output on disk: `results/salvage_sensitivity/salvage_sensitivity.json` is
dated **2026-08-23** and is **44 bytes** (i.e. essentially empty — an
old, pre-tag-naming artifact, since current code writes
`salvage_sensitivity__<tag>.json`, never a bare filename). **The
residual-vs-responsive verdict this script exists to produce has not
been recorded since before 2.6, and O&M (a further, even more recent
trigger the docstring couldn't have named) has not been tested against
it either.**

**Declared defects.** Directly implements the test register **L5**
(storage structurally dominated at annual resolution) and
`methodology_demand_horizon.md` §7 ("whether capacity responds to cost")
depend on; its own docstring is explicit that the finding is
re-run-triggered, not permanent.

**Epoch signals.** Docstring narrates a historical result (system cost
fell 17.1%, solar identical to 5 decimal places) explicitly framed as
"the finding being tested," not current — correctly historical, not
stale-as-current. The on-disk artifact (see State) is the actual stale
epoch signal.

**Dependencies.** None upstream.

---

### `23_run_selfgen_frontier.py`

**Question.** No single quoted sentence; docstring states purpose:
sweep "the GRID SERVICE OBLIGATION — the bound on (selfgen + unserved)
as a fraction of Tier 1 demand — and report the cost of the shortfall at
each service level, priced two ways" (VoLL vs. genset).

**Sweeps.** `EPS_GRID = [0.05 ... 0.60]` (12 points, "calibrated to the
model's own projected shortfall fraction," explicitly NOT
`reliability_levels_log()`'s narrower default) × 2 pricings (`voll`,
`genset`) × 2 builders. CLI: 17 of 17 sentinel keys except `--om-case`
(hardcoded, see Fixes).

**Fixes.** `capital_case=FORCING_CAPITAL_CASE="tight"` (deliberately
forcing, named as such), `om_case="om_central"` hardcoded, no CLI flag.
`RELIABILITY_MODE = "annual"` fixed — the docstring gives a reasoned,
non-arbitrary case: a horizon-total mode would let early-year slack
subsidise late-horizon collapse and still report compliance.

**Interesting naming finding, checked and resolved (not a defect).**
This script calls the builders with `reliability_max_unserved_fraction=`
and `reliability_mode=` — the **pre-rename** parameter names its own
docstring documents as a deliberately-deferred TODO ("rename ...
Rename is deliberately NOT bundled with this behaviour change"). Checked
`src/optimize_model.py`/`optimize_model_sliced.py` directly: both
parameters survive as **documented `DEPRECATED alias`s** for
`max_offgrid_fraction`/`grid_service_mode`, wired through correctly
(`if reliability_max_unserved_fraction is not None: max_offgrid_fraction
= reliability_max_unserved_fraction`, and raises if BOTH names are
passed at once). This script still works correctly — a rename executed
with real backward compatibility, not a break.

**Solves.** 2 builders × 2 pricings × 12 eps = 48.

**Writes.** `results/selfgen_frontier/frontier__<tag>.csv/json`, tag from
15 fields, no hash8.

**State: PASSING (Group 2 confirmed), but stale on disk** — same
pattern as `22_`: `results/selfgen_frontier/frontier.json` (bare
filename, pre-tag) is dated **2026-08-23**, predating the current naming
convention and certainly O&M.

**Declared defects.** Its own docstring documents and corrects a prior
defect in an earlier version of itself: "The first version of this
script compared VoLL-priced against genset-priced cost on an
UNCONSTRAINED tight-capital case... miscalibrated... -94.6% objective
delta measured the size of the VoLL assumption, not the economics of
self-generation." Also documents a second, independently-caught error:
an earlier calibration compared eps against NIRP's lambda, which the
docstring states was WRONG (different denominators) and corrects.

**Epoch signals.** `results/selfgen_frontier/frontier.json`, 2026-08-23
(see State).

**Dependencies.** Reads `data/cost/processed/genset_cost.csv` directly
(for avoided-genset rate reporting, not via `load_econ`).

---

### `24_run_leakage_arm.py`

**Question.** No single sentence; docstring states purpose: measure "the
quantity of real national emissions that a grid-boundary carbon cap
FAILS TO SEE" by solving the same forcing case with the emissions
boundary drawn both ways (`selfgen_in_emissions_cap` False/True).

**Sweeps.** 3 genset arms (`genset_srmc`, `genset_central`,
`genset_high`) × 2 boundary settings = 6 solves. CLI: 12 flags (12 of the
17 sentinel keys).

**Fixes.** `NDC_SCENARIO="ndc3_unconditional"`, `EPS=0.40`,
`CAPITAL_CASE="tight"`, `GRID_SERVICE_MODE="annual"`,
`gas_deliverability_case="baseline"`, `om_case="om_central"` — none
CLI-exposed.

**Solves.** 6, `build_model` (annual only).

**Writes.** `results/leakage_arm/leakage_arm__<tag>.csv`, tag from 12
fields, no hash8.

**State: NOT healthy — same defect class as `04_run_fin2_tariff_sweep.py`,
found independently here.** `solve_one()`'s `load_scenario()` call (lines
242–267) **omits `conversion_case`, `hydro_case`, and `tariff_case`
entirely** — all three sentinel-required. Supplying all 12 of this
script's own required CLI flags will still raise `TypeError` inside
`load_scenario()`. Not cited by any register entry found. This script,
`25_`, and `99c_` (below) share this exact defect — flagged as a pattern
in the synthesis section.

**Declared defects.** None cite this file by number (the register does
not yet know about this defect).

**Epoch signals.** None hardcoded numerically as objective figures — the
genset rates (`0.729/0.902/4.638` implied EF_e) are citation constants,
consistent with current sourcing.

**Dependencies.** Reads `data/cost/processed/emissions_cap.csv`.

---

### `25_run_obligation_trajectory.py`

**Question.** No single sentence; docstring states purpose: test whether
a per-year `max_offgrid_fraction` TRAJECTORY (front-loaded vs.
back-loaded vs. flat, all averaging eps=0.45) shifts solar build TIMING
under a fixed capital envelope with declining capex, not just the
objective.

**Sweeps.** 3 trajectory shapes (`front_loaded`, `back_loaded`, `flat`),
all `np.linspace`-generated around `EPS_AVG=0.45`. CLI: 12 of 17
sentinel keys (same 12 as `24_`).

**Fixes.** `NDC_SCENARIO="ndc3_unconditional"`,
`GENSET_ARM="genset_central"`, `CAPITAL_CASE="tight"`,
`GRID_SERVICE_MODE="annual"`, `gas_deliverability_case="baseline"`,
`om_case="om_central"` — none CLI-exposed. Same case family as `24_` by
the docstring's own account.

**Solves.** 3, `build_model` (annual only).

**Writes.** `results/obligation_trajectory/obligation_trajectory_{summary,annual}__<tag>.csv`.

**State: NOT healthy — identical defect to `24_` and `04_`.**
`solve_one()`'s `load_scenario()` call (lines 239–264) **omits
`conversion_case`, `hydro_case`, and `tariff_case`.** Cannot run to
completion under any set of its 12 required CLI flags.

**Declared defects.** None cite this file by number.

**Epoch signals.** None hardcoded.

**Dependencies.** Reads `data/cost/processed/emissions_cap.csv`.

---

### `26_run_f3_identification.py`

**Question.** Quoted directly: *"Is total EaaS solar capacity UNIQUELY
identified at the optimal system cost, or is there a range of EaaS
deployments all achieving the same cost (a degenerate optimum)?"* — F3,
testing whether F1 resolved the pre-F1 degeneracy the F0 audit found.

**Sweeps.** `--capital-case` (moderate/unconstrained, the two F3
configurations, run one per invocation by design) × tolerance
`{1e-6, 1e-5, 1e-4, --tol}` (always-evaluated set unioned with the
required CLI value) — 3 solves per tolerance (normal, min-EaaS,
max-EaaS re-optimisation under a fixed objective ceiling). CLI: **17 of
17 sentinel keys, all present, correctly threaded** — `--conversion-case`,
`--hydro-case`, `--tariff-case` all exposed AND passed to
`load_scenario()` (lines 247–249) — **this runner does NOT share the
missing-three-keys defect found in `04_`/`24_`/`25_`/`99c_`.**

**Fixes.** `NDC_CAP_SCENARIO="ndc3_conditional"` — TASK.md-specified,
documented in the module docstring.
`om_case="om_central"` hardcoded, no CLI flag (same pattern as elsewhere,
milder — not a missing key).

**Solves.** `2 × len(tolerances)` re-solves plus 1 normal solve, all
`build_model_sliced`.

**Writes.** `results/f3/<hash8>/summary.json` — hash8-shortened (16
required arms; docstring cites the same `OSError [Errno 36]` class as
`01b_`/`03_`), full tag preserved, `INDEX.md` appended.

**State: PASSING, and unusually rigorous** — an internal consistency
assertion (`constrained == has_pbc`) checks the scenario dict agrees with
the model's own public-budget-constraint flag before solving anything.
Five SCOPED CONDITIONS are declared in the docstring (tariff arm
superseded, DIST-2/DIST-5 physically-identical channels, B* stale-doc
warning, `gas_avail_nameplate` vs. measured 29.5% [30.6% since TASK 73], NDC cap narrows the
interval) — a model of how to scope a conditional finding.

**Declared defects.** Register **L19** ("`26_run_f3_identification.py`
sets two required keys internally... does not expose `--dist5-case` or
`--solar-cf-case`") — checked against current `parse_args()`: still true,
neither flag exists on this runner today. Docstring's own condition 2
(DIST-2/DIST-5 physically-identical channels) is the same finding L19
references at one remove.

**Epoch signals.** None hardcoded as current-state figures; docstring
cites the historical pre-F1 degeneracy result (F0 audit) explicitly as
motivation, correctly framed as historical.

**Dependencies.** Reads `data/cost/processed/emissions_cap.csv`.

---

### `99c_run_genset_arm_sensitivity.py`

**Question.** No single sentence; docstring states purpose: sweep the
three genset pricing arms crossed with financing regime and obligation
level, replacing a retired VoLL sweep whose premise (a transition from
non-binding to binding) "can no longer occur" once a self-generation
backstop exists.

**Sweeps.** 3 genset arms × 2 financing arms (`public_only`, `eaas`) × 6
`EPS_LEVELS` = 36 solves. CLI: 12 of 17 sentinel keys (same 12-flag
pattern as `24_`/`25_`).

**Fixes.** `NDC_SCENARIO="ndc3_unconditional"`,
`GRID_SERVICE_MODE="annual"`, `gas_deliverability_case="baseline"`,
`om_case="om_central"`, both `FINANCING_ARMS` entries hardcode
`required_margin=1.10` (documented as correct: "no 1.05/1.10 split to
retire" — this is one of the F1 task's declared six no-tariff runners,
out of scope for that retirement).

**Solves.** 36, `build_model` (annual only).

**Writes.** `results/genset_arm_sensitivity/genset_arm_sensitivity__<tag>.csv`.

**State: NOT healthy — fourth instance of the same defect.**
`solve_one()`'s `load_scenario()` call (lines 266–291) **omits
`conversion_case`, `hydro_case`, and `tariff_case`**, identically to
`04_`/`24_`/`25_`. Cannot run to completion under its 12 required CLI
flags alone.

**Declared defects.** None cite this file by number. Its own docstring
documents the defect that retired its predecessor,
`99c_run_voll_sensitivity.py`: loaded `econ` without a `genset_case`, so
`selfgen` silently absorbed the shortfall the script never printed, and
`unserved = 0.00` was reported across the board while "real shortfall of
hundreds of TWh went unreported."

**Epoch signals.** `EPS_LEVELS` capped at 0.30 with a comment citing "the
unconstrained optimum shortfall is ~0.31" as the calibration basis — not
verified against the frozen state, but self-consistently sourced from
this script's own prior runs, not an external stale figure.

**Dependencies.** Reads `data/cost/processed/emissions_cap.csv`.

---

## GROUP 3 — `TypeError` from `load_scenario`, genuinely dead

### `05_run_fin3_capital_sweep.py`

**Question.** Quoted (RESEARCH QUESTION, FIN-3): *"How does the public
capital budget ceiling (B*) independently determine the energy access
outcome under NDC constraints — and at what budget level does the
public-capital bottleneck disappear without EaaS?"*

**Sweeps (as designed).** 5 capital levels (`tight`→`unconstrained`) ×
3 DisCo collection-rate scenarios (`no_friction`/`reform_target`/`current`,
1.00/0.60/0.35, sourced to World Bank DPO P164307 / AfDB PSRP appraisal)
= 15 solves.

**Fixes (as designed).** `NDC_CAP_SCENARIO="ndc3_unconditional"`,
`gas_deliverability_case="baseline"`, `carbon_case="no_policy"`,
`om_case="om_central"`.

**Solves.** 0 in practice — never reaches a solve. `build_model` (annual)
is the intended builder.

**Writes.** Never reached: `results/fin3/capital_sweep_{results.csv,summary.json}`
(no arm tag on either — a second defect independent of the `TypeError`,
since two invocations with different configs would overwrite each
other's output).

**State: DEAD. No CLI/argparse at all — every configuration value is a
Python constant.** `load_scenario()` call (lines 162–171) passes only 8
parameters: `demand_level_case`, `demand_case`, `gas_deliverability_case`,
`capital_case`, `carbon_case`, `start_year`, `end_year`, `om_case`.
**Missing all 16 other sentinel-required keys:** `peak_multiple_case`,
`reserve_margin`, `storage_cc_case`, `gas_build_case`,
`gas_availability_case`, `gas_retirement_case`, `gas_capex_case`,
`ccgt_efficiency_case`, `capital_recovery_case`, `dscr_case`,
`us_inflation`, `capital_treatment_case`, `concessional_envelope_case`,
`conversion_case`, `hydro_case`, `tariff_case`. Raises `TypeError` on
the first of these `load_scenario()` checks internally, before any solve.

**Declared defects.** Matches register **L10**'s dead-runner class
exactly, though L10 does not name this file specifically (L10's list is
`05, 06, 07, 09b (both), 15, 17, 18, 19, 20` — **`05` IS named**, so L10
already covers this one correctly).

**Epoch signals.** None reached (dies before any solve).

**Dependencies.** Would read `data/cost/processed/emissions_cap.csv`
and `results/baseline/` per its docstring's RUN ORDER — never reached.

---

### `06_run_gas1_shadow_benchmarks.py`

**Question.** Quoted (RESEARCH QUESTION, GAS-1): *"What is the shadow
price of gas deliverability to Nigeria's power sector across planning
scenarios, and does it structurally exceed the opportunity cost of gas
exports and domestic regulated benchmarks?"*

**Sweeps (as designed).** 4 gas regimes (`baseline/upside/downside/shock_recovery`)
× effectively 3 policy cases (`baseline_no_policy`,
`ndc2_unconditional`, `ndc3_unconditional`, via a mix of a local function
and the imported `run_gas_regime_ndc_matrix`).

**Fixes (as designed).** `capital_case="moderate"` (no-cap branch) /
per-`NDC_CASES` value, `carbon_case="no_policy"`, `om_case="om_central"`.
7 hardcoded benchmark prices (domestic 2.4–3.3, opportunity 3–7, LNG
export 10 USD/MMBtu).

**Solves.** 0 — dies in `run_no_policy_gas_matrix()`, the first branch
`main()` calls.

**Writes.** Never reached: `results/gas1/{gas_shadow_matrix.csv,
benchmark_comparison.csv, gas_shadow_by_year.csv, gas1_summary.json}`.

**State: DEAD.** `load_scenario()` call (lines 122–131) passes 8
parameters, identical pattern to `05_`: `demand_level_case`,
`demand_case`, `gas_deliverability_case`, `capital_case`, `carbon_case`,
`start_year`, `end_year`, `om_case`. **Missing the same 16 sentinel keys**
as `05_run_fin3_capital_sweep.py`.

**Declared defects.** Register **L10** names this file directly (`06`
in its list).

**Epoch signals.** None reached.

**Dependencies.** Imports `run_gas_regime_ndc_matrix` from
`src/optimize_experiments.py` for the NDC-capped arms — not inspected
here (the no-cap branch dies first regardless).

---

### `07_run_gas2_eaas_gas_relief.py`

**Question.** Quoted (RESEARCH QUESTION, GAS-2): *"How does EaaS solar
deployment change the scarcity rent of gas — and does private capital
deployment functionally relieve the gas constraint that public-only
investment cannot?"*

**Sweeps (as designed).** 4 gas regimes × 5 `NDC_CASES` × 2 financing
arms (traditional/EaaS) — the most ambitious sweep in this group.

**Fixes (as designed).** Per-`NDC_CASES` `required_margin` (1.10/1.05)
and `capital_case`; `tariff_case=CANONICAL_TARIFF_CASE="tariff_legacy_95m"`
— **this one DOES thread a tariff arm**, unlike `05_`/`06_`.

**Solves.** 0.

**Writes.** Never reached: `results/gas2/{eaas_vs_noeaas_gas_shadow.csv,
gas2_summary.json}`.

**State: DEAD, and self-documented as such.** This script's own comment
(lines 74–80) states the diagnosis exactly: *"This runner is
non-functional independent of this task — its `load_scenario()` calls
omit ~15 arms required since phases 2.6-2.13 and raise before
returning."* Confirmed directly: both `load_scenario()` calls (lines
123–133, 227–237) pass 9 parameters (the `05_`/`06_` set plus
`tariff_case`) and **omit 15**: `peak_multiple_case`, `reserve_margin`,
`storage_cc_case`, `gas_build_case`, `gas_availability_case`,
`gas_retirement_case`, `gas_capex_case`, `ccgt_efficiency_case`,
`capital_recovery_case`, `dscr_case`, `us_inflation`,
`capital_treatment_case`, `concessional_envelope_case`,
`conversion_case`, `hydro_case`. Matches the self-diagnosis exactly (15).

**Declared defects.** Register **L10** names this file (`07`). Also
self-documents the tariff-literal retirement (the bare `95_000_000` this
file used to hardcode is now `CANONICAL_TARIFF_CASE`, pinned to the
legacy attribution arm) — a repair made independent of the larger dead-arg
defect, which remains.

**Epoch signals.** None reached.

**Dependencies.** Same `run_gas_regime_ndc_matrix` dependency as `06_`.

---

**Root-cause note, established here and applying to every remaining
Group 3 entry below.** `08_`, `09_`, `10_`, `11_`, `13_` call no
`load_scenario()` directly — they delegate to shared helper functions in
`src/optimize_experiments.py` (`run_gas_regime_ndc_matrix`,
`run_rel1_feasibility_matrix`, `run_rel2_marginal_cost_curves`,
`run_rel3_financing_frontier`, `run_dem2_growth_gas_matrix`). **Every one
of those five shared functions contains its own `load_scenario()` call
that independently omits the same 15-16 sentinel-required keys** — this
is not five separate bugs, it is one bug (a stale calling convention)
replicated by copy-paste across five library functions, each inherited
by the runner(s) that call it. One of the five
(`run_rel1_feasibility_matrix`) is self-documented: an inline comment
reads *"This `load_scenario()` call is also already missing
`peak_multiple_case`/`reserve_margin`/`storage_cc_case`/`gas_build_case`/
`gas_availability_case` (required since 2.6/2.9/2.10) and would raise
before reaching here regardless."* Fixing these five shared functions
once would resurrect every runner below that depends on them — a
materially different repair shape than "12 broken scripts."

### `08_run_gas3_regime_ndc_feasibility.py`

**Question.** Quoted (RESEARCH QUESTION, GAS-3): *"How does the
structural form of gas decline — exponential depletion vs shock-recovery
vs lag-ramp-uplift — affect the cost and feasibility of NDC compliance
independently of the level of gas availability?"*

**Sweeps (as designed).** 4 structural gas regimes + 3 flat
level-equivalent controls (isolating shape from level) × 4 NDC scenarios.

**Fixes (as designed).** Per-scenario `capital_case`
(`moderate`/`expansion`).

**Solves.** 0 — delegates entirely to `run_gas_regime_ndc_matrix()`,
which dies in its own `load_scenario()` call (see root-cause note).

**Writes.** Never reached: `results/gas3/{regime_ndc_matrix.csv,
regime_feasibility.csv, annual_ndc_binding.csv, shape_cost_premium.csv,
gas3_summary.json}`.

**State: DEAD**, via the shared-function root cause. Separately requires
`build_gas_deliverability.py` to have been re-run first to populate
`flat_downside`/`flat_upside`/`flat_shock_recovery` rows — not verified
here, moot given the earlier defect.

**Declared defects.** Register **L10** names this file (`08`) — though
note `08_phase8_analysis.py` (Group 1, unrelated file, same number
prefix) is a different script; L10's "08" plainly means this one
(`08_run_gas3_regime_ndc_feasibility.py`), confirmed by cross-checking
L10's own list against the dead-runner symptom it describes.

**Epoch signals.** None reached.

**Dependencies.** `data/cost/processed/gas_available_power_annual_twh_th.csv`
(via `build_gas_deliverability.py`), `emissions_cap.csv`.

---

### `09_run_rel1_feasibility.py`

**Question.** Quoted (RESEARCH QUESTION, REL-1): *"Under a fixed capital
envelope and solar build rate, does the feasibility of minimum
reliability standards depend primarily on gas deliverability regime
rather than on the level of solar investment?"*

**Sweeps (as designed).** 4 gas regimes × 2 solar-build-cases
(`deployment_unconstrained`/`nirp_trajectory`) × 7 eps points (6 levels +
VoLL-only) = 56 solves.

**Fixes (as designed).** `CAPITAL_CASE="moderate"`.

**Solves.** 0 — delegates to `run_rel1_feasibility_matrix()`, whose own
inline comment already documents the exact defect (quoted in the
root-cause note above).

**Writes.** Never reached: `results/rel1/{feasibility_matrix.csv,
feasibility_threshold.csv, rel1_summary.json}`.

**State: DEAD, self-documented in the library function it calls.**

**Declared defects.** Register **L10** names this file (`09` — distinct
from the retired `09b` pair, both separately listed in L10).

**Epoch signals.** Docstring cites a historical figure — "NIRP's
cumulative path binds this model from 2027 to 2037 (shortfall peaking at
4,505 MW in 2032)" — correctly framed as a prior measurement, not a
current claim; also the docstring's own CAVEAT flags that under
`nirp_trajectory` cost is VoLL-priced "until the genset backstop lands
(plan 2.6)" — 2.6 has since landed (see the register L25/task-methods
discussion elsewhere), so this caveat is itself now stale, though moot
since the script cannot run.

**Dependencies.** None reached.

---

### `10_run_rel2_marginal_cost.py`

**Question.** Quoted (RESEARCH QUESTION, REL-2): *"What is the marginal
cost of tightening the reliability standard (`dual_reliability`), and
how does this cost curve change across gas deliverability regimes?"*

**Sweeps (as designed).** Gas regimes × an eps grid, reading the dual on
the reliability constraint plus a primal-slack check to distinguish
"slack" from "degenerate-zero-dual."

**Fixes (as designed).** `CAPITAL_CASE="moderate"` (matches `09_` "for
comparability").

**Solves.** 0 — delegates to `run_rel2_marginal_cost_curves()`, same
8-parameter `load_scenario()` pattern as the other shared functions.

**Writes.** Never reached: `results/rel2/{cost_curve_all.csv,
cost_curve_binding.csv, curve_summary.csv, rel2_summary.json}`.

**State: DEAD**, shared-function root cause.

**Declared defects.** Not explicitly named in L10's list (`10` is
absent) — **this is a gap in L10's own coverage**, worth flagging: L10
says its list "predates the O&M work and is likely incomplete," and this
is a concrete instance — `10_run_rel2_marginal_cost.py` is dead for the
identical reason as its listed neighbours `09`/`11` but is not itself
named.

**Epoch signals.** None reached.

**Dependencies.** None reached.

---

### `11_run_rel3_financing_frontier.py`

**Question.** Quoted (RESEARCH QUESTION, REL-3): *"Does EaaS financing
improve reliability outcomes (measured by the feasibility frontier)
relative to public-capital-only financing under the same gas
constraint? Is this still important under NDC3.0?"*

**Sweeps (as designed).** 4 gas × 3 policy × 2 financing × 7 eps = 168
solves — the largest sweep in this group.

**Fixes (as designed).** Per-policy `capital_case`/`required_margin`
pairs; EaaS tariff pinned to `tariff_legacy_95m`.

**Solves.** 0 — delegates to `run_rel3_financing_frontier()`, 9-parameter
`load_scenario()` call (adds `tariff_case` to the 8-parameter base),
missing the same 15 keys as `07_`/`13_`/`15_`.

**Writes.** Never reached: `results/rel3/{frontier_all.csv,
frontier_threshold.csv, frontier_shift.csv, rel3_summary.json}`.

**State: DEAD**, shared-function root cause.

**Declared defects.** Register **L10** names this file (`11`). Docstring
itself flags a second, independent defect even if it could run: "CAVEAT
(2.6, D17): at the default tariff the EaaS arm has no binding capital
constraint... `frontier_shift_eps` therefore measures
constrained-vs-unconstrained, not a financing effect. Resolve before
reporting." — i.e. this runner has a declared D17 defect stacked on top
of the dead-args defect.

**Epoch signals.** None reached.

**Dependencies.** `emissions_cap.csv` (via the shared function).

---

### `13_run_dem2_growth_gas_interaction.py`

**Question.** Quoted (RESEARCH QUESTION, DEM-2): *"How does demand
growth rate uncertainty interact with gas deliverability constraints to
determine system cost risk — and does higher demand growth make the
EaaS mechanism more or less valuable?"*

**Sweeps (as designed).** 3 demand cases × 4 gas cases × 2 financing × 3
policy = 72 solves — matches register **L9**'s independently-recorded
"DEM-2 runs 72" correction (the "48 runs" DO-NOT-REPORT entry from the
prior documentation-closure task).

**Fixes (as designed).** None beyond the swept dimensions.

**Solves.** 0 — delegates to `run_dem2_growth_gas_matrix()`, same
9-parameter pattern (adds `tariff_case` and `solar_build_case`, neither
of which is sentinel-required, so still missing the same 15 keys).

**Writes.** Never reached: `results/dem2/{dem2_results.csv,
gas_premium.csv, interaction_matrix.csv, eaas_value.csv,
dem2_summary.json}`.

**State: DEAD**, shared-function root cause.

**Declared defects.** Register **L10** names this file (`13`). Its
docstring explicitly distinguishes itself from a Monte Carlo / fixed-plan
sensitivity — worth preserving as a design note for whoever repairs it,
since the distinction (re-optimise per growth rate vs. perturb a fixed
plan) is easy to collapse by accident during a repair.

**Epoch signals.** None reached.

**Dependencies.** `emissions_cap.csv` (via the shared function).

---

### `15_run_pol1_ndc_comparison.py`

**Question.** Quoted (RESEARCH QUESTION, POL-1): *"Does NDC 3.0's
absolute emission reduction target... require a materially different
investment trajectory than the NDC 2.0 BAU-relative targets — and does
EaaS change the answer?"*

**Sweeps.** 4 NDC versions × 2 financing arms = 8 solves.

**Fixes.** `gas_deliverability_case="baseline"`,
`CANONICAL_TARIFF_CASE` pinned (value not re-inspected here, pattern
matches `07_`'s legacy pin).

**Solves.** 0 — **this is the one Group-3 runner that calls
`load_scenario()` directly** (not via a shared function), at line 191:
10 parameters passed (`demand_level_case`, `demand_case`,
`gas_deliverability_case`, `capital_case`, `solar_build_case`,
`carbon_case`, `start_year`, `end_year`, `tariff_case`, `om_case`).
**Missing 15 sentinel keys**: `peak_multiple_case`, `reserve_margin`,
`storage_cc_case`, `gas_build_case`, `gas_availability_case`,
`gas_retirement_case`, `gas_capex_case`, `ccgt_efficiency_case`,
`capital_recovery_case`, `dscr_case`, `us_inflation`,
`capital_treatment_case`, `concessional_envelope_case`,
`conversion_case`, `hydro_case`.

**Writes.** Never reached: `results/pol1/{pol1_results.csv,
trajectory_comparison.csv, ndc_version_diff.csv, pol1_summary.json}`.

**State: DEAD.** Same defect family as the shared functions, but a
directly-owned instance of it, not inherited.

**Declared defects.** Register **L10** names this file (`15`). Also the
DO-NOT-REPORT entry in `merged_correction_plan_v2.md` ("POL-1 `public_only`
(`15_`) — $220.75B cost difference") independently establishes this
runner's historical output was already invalidated before this dead-args
defect is even considered.

**Epoch signals.** None reached.

**Dependencies.** `emissions_cap.csv` (docstring notes it now needs all
four NDC scenarios in one file, which `00_build_emissions_cap.py`
already provides).

---

### `18_run_pareto_frontier.py`

**Question.** Quoted: answers *"MIX-2: How does the optimal mix change
across the cost-emissions Pareto frontier, and how does EaaS shift it?"*
— traces 25 points per financing arm from unconstrained to a tight
emissions cap.

**Sweeps (as designed).** 25 cap points × 4 financing arms (tight/moderate
× public/EaaS).

**Fixes (as designed).** Per-arm `capital_case`/`required_margin`.

**Solves.** 0 — direct `load_scenario()` call (line 131), 9 parameters,
**missing all 16 sentinel keys** (no `tariff_case` either, unlike `15_`).

**Writes.** Never reached: `results/pareto_frontier/...`.

**State: DEAD.**

**Declared defects.** Register **L10** names `18_run_pareto_frontier.py`
directly, and separately flags it as one of three runners "on the
planned-outputs list" that "must be wired before those analyses can run,
or the analysis dropped" (alongside `17_` and `19_`).

**Epoch signals.** None reached.

**Dependencies.** `results/baseline/diagnostics.json`,
`emissions_cap.csv` (docstring RUN SEQUENCE).

---

### `19_run_carbon_price_sweep.py`

**Question.** Answers *"MIX-3: At what carbon price does solar become
cost-dominant, and how does EaaS affect this threshold?"* — sweeps
$0–150/tCO2 with no emissions cap.

**Sweeps (as designed).** 13 carbon prices × 3 financing arms.

**Fixes (as designed).** Per-arm `capital_case`/`required_margin`.

**Solves.** 0 — direct `load_scenario()` call (line 96), identical
9-parameter/16-missing pattern to `18_`.

**Writes.** Never reached: `results/carbon_price_sweep/...`.

**State: DEAD.**

**Declared defects.** Register **L10** names this file directly, in the
same "planned-outputs, must be wired or dropped" group as `18_`
(quoted under `18_`'s entry above).

**Epoch signals.** None reached.

**Dependencies.** `results/baseline/diagnostics.json` (docstring
PRE-REQUISITE).

---

### `20_run_cf1_solar_yield_sensitivity.py`

**Question.** Quoted (RESEARCH QUESTION, CF-1): *"How sensitive are
system cost and solar deployment to the assumed solar capacity
factor?"*

**Sweeps.** `CF_SWEEP = [0.16, 0.19, 0.20]` — the current sourced band
(matches `methodology_physical_layer.md` §5.3's central/band figures
exactly), replacing a retired `[0.18, 0.20, 0.22]` set the docstring
explains was unsourced and partly unattainable.

**Fixes.** `capital_case="unconstrained"`,
`solar_build_case="deployment_unconstrained"`, `land_case="loose"` —
"identical to `01_run_baseline.py`" by the docstring's own account
(though `01_`'s actual `capital_case` is also `"unconstrained"`,
consistent).

**Solves.** 0 — direct `load_scenario()` call (line 84), 9 parameters,
**missing all 16 sentinel keys**, identical pattern to `18_`/`19_`.

**Writes.** Never reached:
`results/cf1/cf_sensitivity.csv`.

**State: DEAD.**

**Declared defects.** Register **L10** names this file (`20`).

**Epoch signals.** None reached. (The CF band itself — 0.16/0.19/0.20 —
is current-epoch, sourced, and matches the frozen-state methodology
document; only the runner is dead, not the parameter it would sweep.)

**Dependencies.** None reached.

---

## GROUP 4 — other

### `12_run_dem1_demand_sensitivity.py` (NotImplementedError, stub)

**Question.** Quoted (RESEARCH QUESTION, DEM-1): *"How sensitive are the
NDC compliance cost and EaaS deployment findings to the choice of served
vs latent demand as the planning base? Does the financing bottleneck
diagnosis change when latent demand is used?"*

**Sweeps (as designed).** 3 demand-level cases (`served`/`latent_low`/`latent_high`)
× 4 NDC configs × 2 financing arms = 12 solves.

**Fixes (as designed).** Per-NDC `public_capital`/`eaas_capital`/`eaas_margin`;
`tariff_case=CANONICAL_TARIFF_CASE`.

**Solves.** 0 — `main()`'s first executable line is `raise
NotImplementedError("DEM-1 suspended: latent demand scenarios use a
depreciated base...")`. **Doubly dead**: even past that, the backend it
would call, `run_dem1_demand_sensitivity()`
(`src/optimize_experiments.py`), is independently broken by this
script's own inline comment — same missing-~15-keys `load_scenario()`
defect as the Group 3 cohort.

**Writes.** Nothing, ever.

**State: STUB, confirmed via `raise NotImplementedError` as the first
line of `main()`.** Docstring layered with three separate warning
headers written at different times (three module-level string literals
— only the first is the real `__doc__`; the other two are inert
expression statements, a minor hygiene artefact, not a functional
defect): `served`/`latent_low`/`latent_high` use the **retired 23.08 TWh
demand base** (multiples of collected energy) and predate both the η
correction and the demand-base correction.

**Declared defects.** Register **L10**'s dead-runner list does not
include `12` by number (it lists `05, 06, 07, 09b (both), 15, 17, 18,
19, 20`) — `12` is dead for a *different* reason (explicit stub) than
L10's TypeError class, so its absence from L10 is correct, not a gap.

**Epoch signals.** `served: 23.08 TWh/yr`, `latent_low: 38.47 TWh/yr`,
`latent_high: 76.93 TWh/yr` — all built on the retired 23.08 TWh base;
`η=0.43` and "gas cap 17.5 TWh_e" cited in the docstring as explicitly
stale, correctly self-flagged.

**Dependencies.** Would read `emissions_cap.csv` — never reached.

---

### `21_compute_resolution_premium.py` (blank error, unknown)

**Question.** No single sentence; docstring states purpose: compute
`premium = (sliced_objective - annual_objective) / annual_objective`,
"a METHODS FINDING, not a scenario result," quantifying how much annual
resolution understates system cost.

**Sweeps.** None — reads two existing result files, no solve of its own.

**Fixes.** N/A.

**Solves.** 0. Reads `results/baseline/summary.json` and
`results/baseline_sliced/summary.json` (both bare filenames, not
hash8-tagged).

**Writes.** `results/resolution_premium/resolution_premium.json`.

**State: could not confirm "blank error" from a static read — the
opposite is predicted.** This script is unusually careful: it has four
explicit basis-consistency checks (same demand/gas/VoLL config, salvage
present in both objectives, unserved energy below a 0.1% tolerance on
both sides, and a **staleness check** — every source file
(`optimize_model.py`, `optimize_model_sliced.py`, `scenarios.py`) must
predate both result files or a blocker fires). Checked directly: both
`results/baseline/summary.json` and `results/baseline_sliced/summary.json`
on disk are dated **2026-08-23**, while `src/optimize_model.py` and
`src/scenarios.py` are dated **2026-09-07** — the staleness check's own
condition (`p.stat().st_mtime < newest_src`) is true for both, so a run
today should print `NOT REPORTABLE` with at least one blocker and exit
1 (via `sys.exit(main())`, `return 1 if blockers else 0`) — a clear,
labelled failure, not a blank one. **This task cannot execute the
script to confirm**, so this is a prediction from static reading, not a
verified outcome — flagged for Zeus to check directly, since the
"blank error" characterisation and this prediction disagree and the gap
itself might be informative (e.g. an import error occurring before any
of this logic runs, which a static read of this file alone would not
show).

**Declared defects.** None cite this file by number. Docstring itself
documents that the previously-reported ~13% resolution premium "was
measured BEFORE the salvage correction (plan 2.5) and is superseded" —
matches register **D8** (`merged_correction_plan_v2.md`: "the ~13%
figure is superseded and already on the do-not-report list").

**Epoch signals.** The two input files it reads (Aug 23, pre-O&M,
pre-current-schema) are themselves the central epoch signal — see State.

**Dependencies.** `results/baseline/summary.json`,
`results/baseline_sliced/summary.json` — both present but stale (see
State).

---

## OTHER FILES — brief inventory

**`.py` utilities, not part of the 32:**

- `scripts/_check_solver.py` — checks CBC/GLPK/HiGHS availability via
  `SolverFactory(...).available()`; no scenario, no model, pure
  environment check.
- `scripts/_compare_2_6.py` — the Commit-A regression comparator for the
  2.6 genset-backstop patch: objective (1e-9 rtol), shortfall
  conservation (exact), build-plan invariance (1e-6 rtol); reads two
  result JSONs, no solve.
- `scripts/build_gas_deliverability.py` — builds
  `gas_available_power_annual_twh_th.csv` from YAML case configs
  (`exponential_decline`/`drift`/`lag_ramp_uplift`); producer for `08_`'s
  flat-equivalent level-control arms (`flat_downside` etc.).
- `scripts/deferred/17_run_monte_carlo.py` — DEFERRED (not retired,
  explicitly distinguished in its own header). Two self-declared
  blocking defects: LCOE numerator includes the VoLL penalty while the
  denominator omits hydro and double-counts storage discharge; the
  demand-growth prior is unsourced (`[SOURCE NEEDED]`, plan 1.6). Also
  calls the annual `build_model` (register L5) and is missing pre-F1
  keys (register L10) — declared as downstream of the two defects above,
  not the primary issue. Matches register L10's description exactly.
- `scripts/diag_fin2.py` — ad-hoc FIN-2 "H1..H5" diagnostic; reads two
  hardcoded `results/fin2/tariff_sweep_results__*.csv` paths. Depends on
  `04_run_fin2_tariff_sweep.py` output, which cannot currently be
  produced (Group 2 finding above) — these CSVs likely do not exist at
  current code state, not verified by execution.
- `scripts/f1_quantity_diagnostic.py` — omnibus, read-only, no-solve
  diagnostic testing three hypotheses (H1 new gas build always zero, H2
  storage is a firm-capacity residual, H3 terminal-truncation arbitrage
  moved from solar to storage) for why F1 quantities are insensitive to
  cost of capital. Reads existing results, no solve.
- `scripts/plot_storage_regime.py` — plots `storage_binding_by_year`
  from ONE hardcoded path, `results/ndc/ndc_unconditional_20_voll_low/diagnostics.json`
  — an NDC-2.0-only, pre-NDC3.0-split naming convention, the oldest
  naming style found anywhere in this inventory; almost certainly absent
  on disk under any current arm-tag convention.
- `scripts/plot_system_diagnostics.py` — plots 5 cases from flat
  `results/ndc*_voll_mid/` and `results/ndc_eaas*_voll_mid/` timeseries
  paths — the same naming-mismatch defect documented under
  `08_phase8_analysis.py` and `16_plot_cost_decomposition.py` above.
- `scripts/probe_2_6_structure.py` — structural probe for the selfgen
  wiring: perturbs `selfgen` by one unit and reads the constraint-body
  and objective-expression response directly, explicitly because the
  Commit-A regression's shortfall-conservation test proved `0 == 0`
  (selfgen never entered the basis) and could not exercise the real
  coefficients. No solve, runs in seconds.
- `scripts/retired/09b_run_rel1_mode_sensitivity.py` — RETIRED (moved,
  not deleted). Swept `grid_service_mode` over `['annual', 'total']`; D6
  removed `'total'` as a live option permanently, so the sweep's whole
  axis is retired. Register L10's "09b (both)" covers this file.
- `scripts/retired/09b_run_rel1_retirement_sensitivity.py` — RETIRED.
  Swept the flat 680 MW/yr gas-retirement proxy (0/340/680/1360),
  superseded by the vintage arms (register L8). The other L10 "09b"
  instance.
- `scripts/rpublic_diagnostic.py` — standalone (no repo or solver
  dependency) arithmetic script comparing three candidate
  interpretations of `r_public` (tranche cost ~2.25%, opportunity cost
  9-11%, handover placeholder 6%) and how far the F1 capital charge
  moves across them.

**`.sh` files:**

- `scripts/run_central_sweep.sh` — **the frozen sweep artefact**
  (Sections A/B/C, established facts confirm 38 invocations, no Section
  D). Header comment quotes the full central configuration verbatim,
  consistent with `findings_frozen.md`.
- `scripts/run_verification_sweep.sh` — **currently broken, verified
  directly.** Its `RUNNERS` heredoc lists `scripts/99c_run_voll_sensitivity.py`
  — checked: **this file does not exist** (`99c_run_genset_arm_sensitivity.py`
  explicitly replaced it, per that runner's own docstring: "REPLACES
  `99c_run_voll_sensitivity.py`"). The script's own `MISSING`-file guard
  (`for f in $RUNNERS; do [ -f "$f" ] || MISSING="$MISSING $f"; done ...
  echo "ABORT -- not found:$MISSING"; exit 1`) would fire on the very
  first invocation and abort before running anything. Separately, its
  `RUNNERS` list includes most of the Group 3 dead cohort (`05, 06, 07,
  08, 09, 10, 11, 13, 15, 18, 19, 20`) — even with the missing-file
  aborted fixed, most of the list would still fail on the shared
  `load_scenario()` defect documented above. One inline comment shows
  awareness some of these were "already broken before this task."
- `scripts/smoke_all.sh` — the source of the established "6 pass / 26
  fail" figure; globs `scripts/[0-9]*.py` (catches `99c_` too, since it
  starts with a digit) and runs each bare, catching import/structural
  errors only — explicitly "NOT a correctness test."
- `scripts/verify_2_6_commit_a.sh` — pre/post-patch regression harness
  for the 2.6 commit, using a detached git worktree (not `git stash`,
  specifically so a failure mid-run cannot lose uncommitted work);
  refuses to run on a dirty working tree.

---

# SYNTHESIS

## Table

| Runner | State | Question (one line) | Declared defects |
|---|---|---|---|
| `00_build_emissions_cap.py` | PASSING | Build NDC 2.0/3.0 cap trajectories | — (L24 covers the shape it embodies) |
| `08_phase8_analysis.py` | PASSING, on stale/mismatched data | Aggregate Phase-8 cost/frontier/scarcity tables | — |
| `14_run_str1_storage_role.py` | PASSING (suspended by design) | What role does storage play at annual resolution? | L5 |
| `16_plot_cost_decomposition.py` | PASSING, but its decomposition source omits O&M | Decompose NPV cost: real/selfgen/VoLL | — (live `src/` gap found) |
| `27_run_hydro_minflow_sensitivity.py` | PASSING | How much does the binding hydro min-flow floor move the system? | L20 |
| `99_validate_baseline.py` | PASSING, validates against two retired quantities | Cross-check baseline LP vs 2024 observed | — |
| `01_run_baseline.py` | argparse-only, PASSING, latent path-length risk | Unconstrained baseline solve | — |
| `01b_run_baseline_sliced.py` | argparse-only, PASSING | Unconstrained sliced baseline solve | L15 (corrected here) |
| `02_run_eaas.py` | argparse-only, PASSING | EaaS-financed solar solve | — |
| `02_run_ndc_caps.py` | argparse-only, PASSING | Public-only NDC cap solves (+ opt-in decline arm) | — |
| `03_run_ndc_eaas.py` | argparse-only, PASSING | EaaS-financed NDC cap solves | — |
| `04_run_fin2_tariff_sweep.py` | **DEAD** (argparse-complete, `load_scenario` incomplete) | Minimum self-financing EaaS tariff (T*) | L17 |
| `22_salvage_sensitivity.py` | argparse-only, PASSING, stale on disk | Is the investment trajectory a cost residual? | L5 |
| `23_run_selfgen_frontier.py` | argparse-only, PASSING, stale on disk | Grid service obligation frontier, VoLL vs genset pricing | — |
| `24_run_leakage_arm.py` | **DEAD** | Carbon leakage past the grid-only cap boundary | — |
| `25_run_obligation_trajectory.py` | **DEAD** | Does obligation *timing* (front/back-loaded) shift build? | — |
| `26_run_f3_identification.py` | argparse-only, PASSING | Is EaaS solar capacity uniquely identified at optimum? | L19 |
| `99c_run_genset_arm_sensitivity.py` | **DEAD** | Does public-vs-EaaS survive the genset pricing arm? | — |
| `05_run_fin3_capital_sweep.py` | **DEAD** | At what B* does the public-capital bottleneck disappear? | L10 |
| `06_run_gas1_shadow_benchmarks.py` | **DEAD** | Does gas shadow price exceed export/domestic benchmarks? | L10 |
| `07_run_gas2_eaas_gas_relief.py` | **DEAD**, self-documented | Does EaaS relieve gas scarcity rent? | L10 |
| `08_run_gas3_regime_ndc_feasibility.py` | **DEAD** | Does gas decline *shape* matter independent of *level*? | L10 |
| `09_run_rel1_feasibility.py` | **DEAD**, self-documented | Does gas regime primarily determine reliability feasibility? | L10 |
| `10_run_rel2_marginal_cost.py` | **DEAD** | What is the marginal cost of tightening reliability? | not in L10 (gap) |
| `11_run_rel3_financing_frontier.py` | **DEAD** | Does EaaS improve the reliability feasibility frontier? | L10, D17 |
| `13_run_dem2_growth_gas_interaction.py` | **DEAD** | Does demand growth amplify the gas-scarcity cost premium? | L10 |
| `15_run_pol1_ndc_comparison.py` | **DEAD** | Does NDC 3.0 require a different trajectory than NDC 2.0? | L10 |
| `18_run_pareto_frontier.py` | **DEAD** | Cost-emissions Pareto frontier, public vs EaaS | L10 |
| `19_run_carbon_price_sweep.py` | **DEAD** | At what carbon price does solar dominate gas on cost alone? | L10 |
| `20_run_cf1_solar_yield_sensitivity.py` | **DEAD** | How sensitive is the system to solar CF? | L10 |
| `12_run_dem1_demand_sensitivity.py` | STUB (`NotImplementedError`) | Does served-vs-latent demand change the financing diagnosis? | not in L10 (correctly — different reason) |
| `21_compute_resolution_premium.py` | Unconfirmed — static read predicts "NOT REPORTABLE", not a crash | What is the annual-vs-sliced resolution cost premium? | D8 |

## Overlaps

- **`09_run_rel1_feasibility.py` / `10_run_rel2_marginal_cost.py`** —
  explicitly two halves of one question by `10_`'s own docstring: REL-1
  uses the primal (feasible/infeasible) at each eps, REL-2 uses the dual
  (marginal cost) at each eps, same gas-regime × eps design. Not
  duplicative — genuinely complementary — but should be read and,
  eventually, repaired as a pair.
- **`11_run_rel3_financing_frontier.py` / `23_run_selfgen_frontier.py` /
  `99c_run_genset_arm_sensitivity.py`** — all three sweep a grid-service
  eps axis and report a feasibility frontier, but vary a different
  primary dimension: `11_` varies financing arm (public vs EaaS), `23_`
  varies the *pricing* of the shortfall (VoLL vs genset), `99c_` varies
  the *genset arm* itself crossed with financing. Genuinely three
  different questions sharing one method — worth naming as a family
  when triaging repairs, since a fix to the shared eps-sweep scaffolding
  would likely help all three at once.
- **`24_run_leakage_arm.py` / `99c_run_genset_arm_sensitivity.py`** —
  both solve the identical forcing case (`ndc3_unconditional`, tight
  capital, `grid_service_mode="annual"`) swept across the same 3 genset
  arms; `24_` asks about the emissions-boundary leakage consequence,
  `99c_` asks about financing-frontier robustness. Same base case,
  different lens.
- **`15_run_pol1_ndc_comparison.py`** substantially overlaps
  `02_run_ndc_caps.py` + `03_run_ndc_eaas.py` in *coverage* (all four
  NDC scenarios × public/EaaS) but does its own independent solves
  rather than reading `02_`/`03_`'s output — `15_` is dead, `02_`/`03_`
  are healthy, so this is currently a duplicated-and-broken path to data
  the healthy pair can already produce.
- **`08_phase8_analysis.py` / `16_plot_cost_decomposition.py` /
  `plot_system_diagnostics.py`** — all three consume the same
  baseline/NDC/NDC+EaaS result set via the identical stale flat-path
  naming convention (`results/ndc/<case>_voll_mid/...`), for three
  different output products (dashboard tables, a decomposition bar
  chart, a diagnostics panel figure). One naming-convention fix would
  potentially repair all three at once.

## Orphans

Most numbered runners ARE the terminal deliverable for their named
research question (RQ/FIN/GAS/REL/DEM/POL/MIX/CF label) — that is their
purpose, not a defect. Two genuine orphans found, where the *consumer*
script is itself neither a deliverable nor reachable:

- **`scripts/diag_fin2.py`** reads
  `results/fin2/tariff_sweep_results__{unbounded,dares_only}__df1.csv`,
  produced only by `04_run_fin2_tariff_sweep.py` — which is dead (Group
  2). Nothing currently produces this script's input, and the script
  itself is an ad-hoc debugging aid ("paste the whole output"), not a
  reportable deliverable.
- **`scripts/plot_storage_regime.py`** reads a single hardcoded path
  under the NDC-2.0-only, pre-NDC-3.0-split naming convention
  (`results/ndc/ndc_unconditional_20_voll_low/...`) that no current
  runner produces or could produce under today's scenario registry.
  Even repaired, its content (`storage_binding_by_year`) is the
  diagnostic register **L11** declares "not meaningful under the sliced
  builder... inert... do not report it."

## Hardcoded epoch signals (pre-O&M or otherwise pre-correction), file and line

- `scripts/27_run_hydro_minflow_sensitivity.py:146` —
  `REFERENCE_NPV_TOTAL_COST_USD = 21351610495.345654` — pre-O&M, but
  **deliberate**: this arm runs `om_case="om_none"` specifically to
  match it (see that entry's State note).
- `scripts/99_validate_baseline.py:22` — `"demand_twh": 23.08` — the
  specifically-retired demand base (correct value 37.09 TWh).
- `scripts/99_validate_baseline.py:25` — `"unserved_twh": 3.0` —
  pre-genset-backstop (2.6); `unserved` is now ~0 by construction under
  any genset case.
- `scripts/12_run_dem1_demand_sensitivity.py:40-42` — `served: 23.08`,
  `latent_low: 38.47`, `latent_high: 76.93` TWh/yr — all built on the
  retired 23.08 TWh base (self-flagged stale in the same docstring).
- `scripts/12_run_dem1_demand_sensitivity.py:53` — "Gas electricity
  output is supply-capped at ~17.5 TWh_e/yr (40.66 TWh_th × η=0.43)" —
  pre-Phase-1.1 η correction (η now 0.287; self-flagged stale).
- `scripts/plot_storage_regime.py:6` — `results/ndc/ndc_unconditional_20_voll_low/...`
  — NDC-2.0-only naming, predates the NDC 3.0 split entirely; the oldest
  naming convention found anywhere in this inventory.
- **On-disk dependency staleness** (not code-hardcoded, but functionally
  identical — every one of these files is dated **2026-08-23**, while
  `src/optimize_model.py`/`optimize_model_sliced.py`/`scenarios.py` are
  dated **2026-09-07**): `results/baseline/{summary,diagnostics}.json`,
  `results/baseline_sliced/summary.json`,
  `results/ndc/ndc2_unconditional_voll_mid/summary.json` (this last one
  additionally has `om_case: None`, `npv_total_cost_usd: 81,374,741,735`
  — roughly 3× the current central objective, no `arm_tag`),
  `results/salvage_sensitivity/salvage_sensitivity.json` (44 bytes),
  `results/selfgen_frontier/frontier.json`.

## Coverage gaps

- **`om_low` and `om_high` — checked specifically, per instruction.**
  Established fact: only `om_central`/`om_none` have ever been run.
  Structurally confirmed here that this gap is **partly load-bearing,
  not just unexercised**: `22_`, `23_`, `24_`, `25_`, `99c_` all
  hardcode `om_case="om_central"` with **no CLI path to any other
  value** — even once repaired for the missing-keys defect, these five
  cannot exercise `om_low`/`om_high` without a further code change.
  Only `01_`, `01b_`, `02_`, `03_`, `26_` (and `04_`/`05_`-`20_`/`99c`'s
  siblings, if repaired to expose it) exposes `--om-case` as a live CLI
  choice. `om_low`/`om_high` are therefore not merely "not yet run" —
  for most of the sweep surface they are **not currently reachable at
  all**.
- **`genset_high`.** Reachable only via `24_run_leakage_arm.py` and
  `99c_run_genset_arm_sensitivity.py`, both dead. No currently-healthy
  runner sweeps the genset arm at all (`01`–`03`, `26` all pin
  `CANONICAL_GENSET = "genset_central"`).
- **`ndc3_conditional_decline` (plan 4.2).** Reachable only via
  `02_run_ndc_caps.py --include-decline-arm`, opt-in and default-off.
  No established fact or on-disk artefact found indicating it has been
  run yet.
- **Everything else is unverifiable from a static, read-only
  inventory.** `storage_cc_low/high`, `gas_retire_vintage_40/50`, the
  full `gas_deliverability_case` GMP-policy family, and most other
  CLI-exposed arms on the healthy runners (`01`/`01b`/`02`/`03`/`26`)
  are *selectable* — nothing in the code blocks them — but whether they
  have actually been invoked is a question about run history this
  report cannot answer without execution, which is out of scope. Flagged
  as a limit of this method, not a finding either way.

---

# Encountered but not asked about

- **The single highest-leverage repair target found in this inventory:
  six shared functions in `src/optimize_experiments.py`
  (`run_gas_regime_ndc_matrix`, `run_rel1_feasibility_matrix`,
  `run_rel2_marginal_cost_curves`, `run_rel3_financing_frontier`,
  `run_dem1_demand_sensitivity`, `run_dem2_growth_gas_matrix`), each
  with its own stale `load_scenario()` call omitting the same ~15-16
  sentinel-required keys.** Repairing these six functions once would
  resurrect `08_`, `09_`, `10_`, `11_`, `13_` outright, and `12_`
  partially (its `NotImplementedError` stub would still need removing
  separately). `06_` and `07_` call one of the six for their NDC-capped
  arms but *also* carry their own directly-broken `load_scenario()`
  call for the no-cap arm, so need both fixes. Nine further runners own
  the defect directly, with no shared function involved: `05_`, `15_`,
  `18_`, `19_`, `20_` (Group 3) and `04_`, `24_`, `25_`, `99c_` (Group
  2). This was not asked for as a repair — TASK.md is explicitly
  read-only — but the *shape* of the fix is itself a descriptive
  finding worth surfacing before triage: "twelve independently broken
  scripts" and "one bug replicated across six library functions, plus
  nine directly-broken scripts" imply very different repair effort,
  even though both produce the identical symptom list.
- **Register L10 is confirmed incomplete, exactly as TASK.md's
  established facts warned ("predates the O&M work and is likely
  incomplete").** Concretely: L10 does not name `10_run_rel2_marginal_cost.py`
  (dead for the identical reason as its listed neighbours `09`/`11`),
  and does not name `24_run_leakage_arm.py`, `25_run_obligation_trajectory.py`,
  or `99c_run_genset_arm_sensitivity.py` — all four newly found dead in
  this inventory, none previously documented as such anywhere this
  session found. L10 also does not need to name `12_` (stub, different
  defect class) or `21_` (unconfirmed, likely not dead at all) — their
  absence is correct.
- **TASK.md's Group 2 premise ("probably healthy — confirm or refute")
  was refuted for 4 of its 12 members**: `04_`, `24_`, `25_`, `99c_`.
  All four pass argparse cleanly but die inside `load_scenario()` on the
  same missing-three-keys pattern (`conversion_case`, `hydro_case`,
  `tariff_case`) — a defect class TASK.md's own Group 3 was built to
  describe, hiding inside a group assumed healthy. Worth flagging
  because it means "argparse accepts every flag" is not sufficient
  evidence of health anywhere in this codebase, a fact worth carrying
  into any future audit of this kind.
- `results/` is gitignored, so none of the on-disk staleness findings in
  this report (file mtimes, stale `om_case`/`arm_tag` fields, the 44-byte
  `salvage_sensitivity.json`) are visible in `git log` or `git diff` —
  they were found by direct filesystem reads, permitted under this
  task's "read files; search the tree" clause. Worth noting since a
  future read-only inventory of this kind should not assume `git`
  alone gives a complete picture of what is stale.
- `12_run_dem1_demand_sensitivity.py` has **three stacked module-level
  docstrings** (only the first is the real `__doc__`; the other two
  execute as inert expression statements) — a minor hygiene artefact
  from sequential edits, not a functional defect, not fixed (read-only).
- Arithmetic check on the established "6 pass / 26 fail" `smoke_all.sh`
  figure: 6 + 26 = 32, matching the total runner count this inventory
  covers exactly. No discrepancy found.
- This inventory did not open `src/optimize_experiments.py` in full —
  only the specific functions named above, located by grep from each
  runner's import statement. A repair task working from this report
  should re-read those six functions in full before editing, per this
  session's own standing instruction not to edit against grep output
  (inherited from this project's `CLAUDE.md`, itself out of scope to
  re-verify here since it governs edits, and this task made none).

**Then STOP. Nothing edited, nothing committed.**
