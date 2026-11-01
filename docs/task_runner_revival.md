# task_runner_revival — Part 1: INVENTORY

Read-only. Nothing edited, run, staged, or committed. Written
incrementally as each runner's status was established. Ground truth for
every "missing argument" claim below was obtained either by reading the
actual call site against the actual current signature (via
`inspect.signature`, side-effect-free) or, where `load_scenario()`
itself was the callee, by directly invoking `load_scenario()` with
exactly the arguments a call site supplies (also side-effect-free —
`load_scenario()` builds a plain dict from CSVs/YAML, no model, no
solve) and reading the resulting `TypeError`/`ValueError` text. No
script was executed; no model was built via a runner's own `main()`.

**Headline correction, stated first per this task's own instruction not
to trust any prior list.** Three files this session's own prior audits
repeatedly treated as healthy are not:

- **`scripts/02_run_eaas.py`** — cited as one of "the healthy five"
  (alongside `01_/01b_/02_run_ndc_caps/03_`) in at least three prior
  task reports this session, and never re-checked end-to-end. Its
  `load_scenario()` call (`:215-242`) omits **`conversion_case` and
  `hydro_case`** — confirmed by direct invocation:
  `TypeError: load_scenario() requires conversion_case (Phase 2.8, Step
  2)...`. **This script is currently DEAD.** See its full entry below.
- **`scripts/probe_2_6_structure.py`** — given a clean bill twice
  (`docs/forensic_audit.md:2320-2322`, Phase 2 and Phase 3: "no defect
  found") — but that verdict was scoped to the script's own structural-
  probe logic, never to whether its `load_scenario()` call (`:142-148`)
  succeeds. It supplies only `om_case` of the 17 sentinel-required keys.
  **Currently DEAD** — 16 missing arguments.
- **Register L10's own claim, and this task's framing of D6, both
  overstate how dead `grid_service_mode='total'` is.** `'total'` was
  NOT removed from the codebase — `src/optimize_model.py:976` and
  `src/optimize_model_sliced.py:1210` both still implement it, and it is
  actively exercised TODAY by a live call: `run_rel1_feasibility_
  matrix` (`src/optimize_experiments.py:3478`) hardcodes `reliability_
  mode="total"`, and this function is `09_run_rel1_feasibility.py`'s
  sole engine. See `scripts/retired/09b_run_rel1_mode_sensitivity.py`'s
  entry (1E) for the full nuance.

**Also corrected: this task's own characterisation of `17_run_monte_
carlo.py`'s method.** TASK.md and prior session framing call it "fixed-
plan re-dispatch." It is not — `17_` calls `build_model()` fresh and
re-solves for every draw (`:202-207`), i.e. full re-optimisation, the
same method family as `29_run_parameter_uncertainty.py`. The genuinely
"fixed-plan re-dispatch" mechanism is a **third**, separate, wholly
unused module: `src/stochastic.py` (already declared dead in register
L32 / forensic finding A11). See 1F below.

---

## THE DEAD SET IS LARGER THAN "ABOUT TEN" — FLAGGED AS INSTRUCTED

**Fourteen runners are currently dead** on the required-argument/stub
dimension this task is about (excluding `plot_storage_regime.py`,
which is also currently non-executable but for an unrelated reason — a
stale hardcoded output path and a stale vocabulary map, not a missing-
argument defect; it does not fit this task's THREAD-ONLY/CLI-NEEDED
repair mechanism at all, so it is inventoried below but not counted
toward this total):

1. `scripts/02_run_eaas.py`
2. `scripts/06_run_gas1_shadow_benchmarks.py`
3. `scripts/07_run_gas2_eaas_gas_relief.py`
4. `scripts/08_run_gas3_regime_ndc_feasibility.py`
5. `scripts/09_run_rel1_feasibility.py`
6. `scripts/10_run_rel2_marginal_cost.py`
7. `scripts/11_run_rel3_financing_frontier.py`
8. `scripts/12_run_dem1_demand_sensitivity.py` (STUB)
9. `scripts/13_run_dem2_growth_gas_interaction.py`
10. `scripts/14_run_str1_storage_role.py` (STUB)
11. `scripts/probe_2_6_structure.py`
12. `scripts/deferred/17_run_monte_carlo.py`
13. `scripts/retired/09b_run_rel1_mode_sensitivity.py`
14. `scripts/retired/09b_run_rel1_retirement_sensitivity.py`

**This exceeds the "about ten" threshold TASK.md itself names as
changing the task's shape.** Of the fourteen: two are genuinely retired
(SUPERSEDED, no revival recommended — 13, and arguably 12 with strong
caveats on 12's SUPERSEDED-adjacent framing below); two are deliberate
STUBs whose suspension is substantively, not just mechanically,
justified (12, 14); one needs more than plumbing before its output is
reportable regardless of argument-threading (17_run_monte_carlo.py).
That leaves **nine runners that are pure plumbing revivals** (02_run_
eaas, 06, 07, 08_run_gas3, 09, 10, 11, 13, probe_2_6_structure) — close
to, but still slightly over, the original "about ten" estimate once
counted this way. **Recommendation stated per instruction: Part 2
should very likely be split**, at minimum separating (a) the nine pure
CLI/THREAD-ONLY plumbing revivals from (b) the two STUB/SUPERSEDED
dispositions that need a judgement call stated rather than code changed,
from (c) `17_run_monte_carlo.py`'s larger, multi-defect revival. This
report proceeds to sketch Part 2 for all of them below regardless, since
TASK.md asks for a sketch either way, but the split recommendation is
the load-bearing point of this section.

---

## MASTER TABLE — every `.py` under `scripts/`, `scripts/deferred/`, `scripts/retired/`

| File | 1A: executes? | 1C: classification | 1D: known defects |
|---|---|---|---|
| `00_build_emissions_cap.py` | YES (no `load_scenario` dependency at all) | ALREADY WORKS | A18 (fixed, guard added) |
| `01_run_baseline.py` | YES (all 17 keys, own argparse) | ALREADY WORKS | — |
| `01b_run_baseline_sliced.py` | YES (all 17 keys, own argparse; `cli_arms.py`'s own reference implementation) | ALREADY WORKS | A15 (fixed in the builder itself — see below), A19 |
| `02_run_eaas.py` | **NO — missing `conversion_case`, `hydro_case`** | **THREAD-ONLY** (has 15/17 flags already; needs 2 more) | Never previously flagged — new this task |
| `02_run_ndc_caps.py` | YES (all 17 keys, own argparse) | ALREADY WORKS | A20 (hand-typed CHOICES tuples, TRAP), A21 (fixed), A47/L37-comment (fixed) |
| `03_run_ndc_eaas.py` | YES (all 17 keys, own argparse) | ALREADY WORKS | A20 (same as 02_ndc_caps) |
| `04_run_fin2_tariff_sweep.py` | YES (own flags + `cli_arms.py` `include=[...]` for the rest) | ALREADY WORKS | A51 (fixed — no longer a "dead stub"), A52/L33, A53/L33 |
| `05_run_fin3_capital_sweep.py` | YES (`cli_arms.py`, full set) | ALREADY WORKS | A34/L37, A35/L37 |
| `06_run_gas1_shadow_benchmarks.py` | **NO — no argparse; `load_scenario()` missing 16 keys; `run_gas_regime_ndc_matrix` missing 17** | **CLI-NEEDED** | A36/L35 (bites on repair) |
| `07_run_gas2_eaas_gas_relief.py` | **NO — same shape as 06_** | **CLI-NEEDED** | **A37/L35 (copy-paste status bug — must fix as part of revival)**, A38/L35 |
| `08_phase8_analysis.py` | YES (no `load_scenario`/shared-fn call at all; reads pre-existing CSVs/JSON) | ALREADY WORKS (on this dimension) | A39/L44, A40/L44 (NaN in real output — schema mismatch, unrelated to this task's repair mechanism) |
| `08_run_gas3_regime_ndc_feasibility.py` | **NO — no argparse; `run_gas_regime_ndc_matrix` missing 17 keys** | **CLI-NEEDED** | A41/L36 (unlisted dead-runner), A42/L36 (stale anchor), A43/L36 (stale docstring totals) |
| `09_run_rel1_feasibility.py` | **NO — no argparse; `run_rel1_feasibility_matrix` missing 17 keys** | **CLI-NEEDED** | uses deprecated `reliability_mode="total"` internally — report, do not fix (see 1E note) |
| `10_run_rel2_marginal_cost.py` | **NO — no argparse; `run_rel2_marginal_cost_curves` missing 17 keys** | **CLI-NEEDED** | uses correct `"annual"` mode internally — clean on this dimension |
| `11_run_rel3_financing_frontier.py` | **NO — no argparse; `run_rel3_financing_frontier` missing 16 keys** | **CLI-NEEDED** | **A45/L34 (required_margin double-count — will bite on repair; not explicitly named by TASK.md but same shape as A56, flagged for Part 2)** |
| `12_run_dem1_demand_sensitivity.py` | **NO — `main()`'s first statement is `raise NotImplementedError`; underneath, `run_dem1_demand_sensitivity` is also missing 17 keys** | **STUB** | A54/L42 (orphaned docstring, declared not fixed) |
| `13_run_dem2_growth_gas_interaction.py` | **NO — no argparse; `run_dem2_growth_gas_matrix` missing 16 keys** | **CLI-NEEDED** | **A55/L43 (this exact defect), A56/L34 (required_margin double-count — must fix as part of revival)** |
| `14_run_str1_storage_role.py` | **NO — no `load_scenario` call at all; `main()` prints a suspension notice and exits** | **STUB** | A57/L39 (stale "re-enable when" precondition — see below) |
| `15_run_pol1_ndc_comparison.py` | YES (`cli_arms.py`, full set; A46-49 fixed last task) | ALREADY WORKS | A46-49 (fixed), A48/L37-comment (fixed) |
| `16_plot_cost_decomposition.py` | N/A (plotting script; no `load_scenario`/shared-fn call) | ALREADY WORKS (on this dimension) | A50 (fixed last task — reads `om_npv`, has completeness guard) |
| `18_run_pareto_frontier.py` | YES (`cli_arms.py`, full set) | ALREADY WORKS | A23 (register L10 stale for this file — already amended) |
| `19_run_carbon_price_sweep.py` | YES (`cli_arms.py`, full set) | ALREADY WORKS | A23 (same) |
| `20_run_cf1_solar_yield_sensitivity.py` | YES (`cli_arms.py`, full set) | ALREADY WORKS | A23 (same) |
| `21_compute_resolution_premium.py` | YES (no `load_scenario` call at all — pure comparator over existing `summary.json` files) | ALREADY WORKS | A58/L40 (vacuous safety check), A59/L40 (untimestamped figure) |
| `22_salvage_sensitivity.py` | YES (own argparse, full set) | ALREADY WORKS | none (annual builder use is legitimate here — salvage is `pre_f1_overnight`-specific) |
| `23_run_selfgen_frontier.py` | YES (own argparse, full set) | ALREADY WORKS | **A15 — CONFIRMED FIXED in `src/optimize_model_sliced.py:258-266`** (was previously the single most severe Phase-1 finding); existing `results/selfgen_frontier/frontier.csv` sliced rows remain VOID until re-run |
| `24_run_leakage_arm.py` | YES (own flags + `cli_arms.py` `include=[...]`) | ALREADY WORKS, conditional on `--conversion-case conv_none` | A24 (annual builder rejects non-`conv_none`; undocumented in-file) |
| `25_run_obligation_trajectory.py` | YES (same pattern as 24_) | ALREADY WORKS, same conditional | A24 (same) |
| `26_run_f3_identification.py` | YES (own argparse, full set) | ALREADY WORKS | L19 (two keys set internally, not swept — reproducibility gap, not an execution blocker), A22 (no capex-reporting field) |
| `27_run_hydro_minflow_sensitivity.py` | YES (deliberately no CLI — hardcoded `CENTRAL_ARGS` dict, by design) | ALREADY WORKS | A22 (same as 26_); L20 (its own designed output, not a defect) |
| `28_run_capital_frontier.py` | YES (`cli_arms.py`, full set) | ALREADY WORKS | none found |
| `29_run_parameter_uncertainty.py` | YES (`cli_arms.py`, full set; already run successfully, n=256) | ALREADY WORKS | none |
| `99_validate_baseline.py` | YES (no `load_scenario` call; reads existing `timeseries.csv`) | ALREADY WORKS | A19 (25.62 duplicate, already declared) |
| `99c_run_genset_arm_sensitivity.py` | YES (own flags + `cli_arms.py` `include=[...]`) | ALREADY WORKS, same `conv_none` conditional as 24_/25_ | A24 (same) |
| `_check_solver.py` | YES (no repo dependency) | NOT-APPLICABLE | none |
| `_compare_2_6.py` | YES, given a valid output directory | NOT-APPLICABLE | none |
| `build_gas_deliverability.py` | YES (no `load_scenario` dependency) | ALREADY WORKS | A29/L13-correction (register prose, not this file, was wrong — fixed last task) |
| `diag_fin2.py` | YES (reads existing CSVs, both confirmed present) | ALREADY WORKS | none |
| `f1_quantity_diagnostic.py` | YES (globs existing result files, all 5 fragments present) | ALREADY WORKS | A30/L-none (stale suspicion in its own comments, COSMETIC, no register entry) |
| `plot_storage_regime.py` | **NO — hardcoded path does not exist** | not applicable to this task's repair mechanism (see note above) | A31/L41 |
| `plot_system_diagnostics.py` | YES (all 5 hardcoded CSVs present) | ALREADY WORKS | A32/L-none (dead code, COSMETIC, no register entry) |
| `probe_2_6_structure.py` | **NO — `load_scenario()` missing 16 keys** | **CLI-NEEDED** | Newly discovered this task — no prior finding number |
| `rpublic_diagnostic.py` | YES (no repo dependency at all) | ALREADY WORKS | A33/L38 (wrong solar lifetime, 25 vs 30) |
| `deferred/17_run_monte_carlo.py` | **NO — `load_scenario()` missing 16 keys** | **CLI-NEEDED, plus BACKEND-BLOCKED by its own two declared defects** | already declared (its own header; register L10, L32) |
| `retired/09b_run_rel1_mode_sensitivity.py` | **NO — `load_scenario()` missing 16 keys** | **SUPERSEDED, by convention (see 1E nuance)** | already declared (its own header; L5, L10) |
| `retired/09b_run_rel1_retirement_sensitivity.py` | **NO — `load_scenario()` missing 16 keys** | **SUPERSEDED (confirmed, cleanly)** | already declared (its own header; L5, L8, L10) |

**46 files total.** 32 ALREADY WORKS (on the dimension this task checks
— several carry OTHER, unrelated declared defects, noted above but not
blocking execution). 14 dead per the count above. 1 file (`plot_storage_
regime.py`) is separately non-executable for an unrelated reason.

---

## PER-RUNNER DETAIL — dead runners and every file carrying a new or corrected finding

Runners confirmed cleanly ALREADY WORKS with no new wrinkle beyond
already-declared, already-cited findings are covered by the master
table alone, per this task's own emphasis on the dead set — re-deriving
a full 1A-1D for all 32 healthy files here would bury the nine files
that actually need Part 2's attention. Every file in the master table
above IS accounted for; what follows is the detail for the ones that
need it.

### `scripts/02_run_eaas.py` — **NEWLY DISCOVERED DEAD**

**1A.** `main()` (`:210`) calls `load_scenario(...)` at `:215-242` with
16 explicit keyword arguments (`demand_level_case, demand_case, capital_
case, gas_deliverability_case, solar_build_case, land_case, carbon_
case, start_year, end_year, peak_multiple_case, reserve_margin
[hardcoded "nirp_spinning"], storage_cc_case, gas_build_case, new_gas_
deliverability, gas_availability_case, gas_retirement_case, gas_capex_
case, ccgt_efficiency_case, capital_recovery_case, dscr_case, us_
inflation, capital_treatment_case, concessional_envelope_case, tariff_
case, om_case`). **Missing: `conversion_case`, `hydro_case`** — neither
appears anywhere in the file (confirmed by grep — not a `--conversion-
case`/`--hydro-case` flag, not a hardcoded literal, not present at all).
Confirmed by direct invocation with exactly this call's arguments:
```
TypeError: load_scenario() requires conversion_case (Phase 2.8, Step 2).
Pass 'conv_none' / 'conv_central' / 'conv_low' / 'conv_high' (see
conversion_scenarios()). There is no default: omitting it would drop
OCGT-to-CCGT retrofit -- the only fuel-free route to more generation
under a binding gas deliverability constraint -- from the choice set
with no visible signature.
```
**Cannot execute today.**

**1B.** No module docstring. Header comment / purpose inferred from its
own CLI help text and body: an EaaS (Energy-as-a-Service) baseline
runner — same family as `01_run_baseline.py` but with `scenario[
"financing_regime"] = "eaas"` (`:244`) — used as `run_deterministic_
scenario`'s live caller for a console-only `det_unserved` diagnostic
print (A8's context, `forensic_audit.md:280-292`).

**1C. THREAD-ONLY.** The file already has a full, mature argparse
surface (15 of the 17 sentinel flags, `reserve_margin` hardcoded) —
this is the smallest possible fix in the entire dead set: add two more
required flags (`--conversion-case`, `--hydro-case`, following the
file's own existing pattern for every other flag) and pass them
through at the call site. Sibling files `03_run_ndc_eaas.py` (same
family, EaaS-focused) already have both flags correctly wired — this
file can copy that pattern directly rather than invent one.

**1D.** No finding number exists for this in any prior audit — every
mention of `02_run_eaas.py` in `forensic_audit.md` (lines 280-292,
1035, 1319, 1650, 1940, 2847, 3576) either cites it as an example of an
"active" script (never re-verified end-to-end) or a partial read
("required-flag counts only, for A26" — the count itself, 15, was
correct, but 15-of-17 was never flagged as 2-SHORT rather than
complete). This is a genuine gap in three prior audit passes this
session, not a previously-declared-and-missed item.

**Why this cannot move a solved value.** `02_run_eaas.py` has never
successfully produced output under the current `load_scenario()`
signature — there is no existing `results/` directory for it to have
written to under this defect, and nothing in `findings_frozen.md` cites
it. Reviving it produces a NEW result, not a revised one.

---

### `scripts/probe_2_6_structure.py` — **NEWLY DISCOVERED DEAD**

**1A.** `run()` (called twice from `main()`, once per builder) calls
`load_scenario(...)` at `:142-148` with only 9 keyword arguments:
`demand_level_case, demand_case, capital_case, gas_deliverability_case,
solar_build_case, land_case, carbon_case, start_year, end_year, om_
case`. **Missing 16 of the 17 sentinel keys**: `tariff_case, peak_
multiple_case, reserve_margin, storage_cc_case, gas_build_case, gas_
availability_case, gas_retirement_case, gas_capex_case, ccgt_
efficiency_case, capital_recovery_case, dscr_case, us_inflation,
capital_treatment_case, concessional_envelope_case, conversion_case,
hydro_case`. `load_scenario()` checks `tariff_case` first among these
in source order, so the concrete failure today is `TypeError: load_
scenario() requires tariff_case...`. **Cannot execute today.**

**1B.** Docstring (verbatim excerpt): *"Structural probe for the
selfgen wiring (correction plan 2.6). ... selfgen never entered the
basis, so its BALANCE COEFFICIENT and its OBJECTIVE PRICE were never
exercised. ... This script tests those structurally instead of by
simulation: it perturbs selfgen by one unit and reads the response out
of the constraint body and the objective expression. No solve
required. Runs in seconds."*

**1C. CLI-NEEDED.** Has its own small argparse (`--busbar`, `--genset-
case`, `--voll`, `--capital-case` — 4 flags, none required except
implicitly via defaults) but was never given `add_standard_arms`/
`standard_arms_kwargs` the way `29_run_parameter_uncertainty.py`
demonstrates. Needs the full `cli_arms.py` treatment, keeping its own
4 existing flags alongside.

**1D.** `docs/forensic_audit.md:2320-2322` gave this file a clean bill
— *"careful, well-guarded side-effect-free structural probe with its
own internal sanity anchors — read in full, no defect found"* — **but
that check never traced the `load_scenario()` call to completion**. The
probe's own internal LOGIC (the balance-coefficient/objective-price
perturbation tests) remains sound and is not in question; only its
ability to reach that logic at all is newly found broken. No prior
finding number.

**Why this cannot move a solved value.** This is a diagnostic probe,
never a source of a reported number — it prints pass/fail assertions to
console and writes nothing to `results/`.

---

### `scripts/06_run_gas1_shadow_benchmarks.py`

**1A.** No `argparse` anywhere (confirmed: 0 hits). Two separate
broken call sites:
- `load_scenario(...)` (`:122-131`, inside `run_no_policy_gas_matrix`'s
  per-case-arm loop) supplies only `demand_level_case, demand_case,
  gas_deliverability_case, capital_case, carbon_case, start_year, end_
  year, om_case` — **missing 16**: `tariff_case, peak_multiple_case,
  reserve_margin, storage_cc_case, gas_build_case, gas_availability_
  case, gas_retirement_case, gas_capex_case, ccgt_efficiency_case,
  capital_recovery_case, dscr_case, us_inflation, capital_treatment_
  case, concessional_envelope_case, conversion_case, hydro_case`.
- `run_gas_regime_ndc_matrix(...)` (`:217-224`, Arm 2) supplies `econ,
  ndc_cap_scenario, gas_cases, financing_regime, capital_case, cap_
  path` — **missing all 17 of that function's own required keys**
  (`peak_multiple_case` through `om_case`, same list as `run_dem2_
  growth_gas_matrix`'s in the master table).
**Cannot execute today, on either arm.**

**1B.** GAS-1 research question — gas shadow price vs. domestic/
opportunity/export benchmark comparison, no-policy and NDC-capped
variants.

**1C. CLI-NEEDED**, for both arms — a single `add_standard_arms(parser)`
plus `standard_arms_kwargs(args)` fixes both call sites at once, since
both need the identical 17-key set.

**1D.** A36/**L35**: the gas-price benchmark dictionary is hand-typed
independently in four places (this file, `src/optimize_experiments.py`,
and twice in `07_`) — currently consistent, unenforced; would only
matter after this repair makes the file executable again. Not a
blocker to reviving it, a trap to note in the same commit.

---

### `scripts/07_run_gas2_eaas_gas_relief.py`

**1A.** No `argparse` (confirmed: 0 hits) — and the file's own comment
(`:76-80`) already self-diagnoses this: *"its load_scenario() calls
omit ~15 arms required since phases 2.6-2.13 and raise before
returning."* Verified precisely: `load_scenario(...)` (`:123-133`)
supplies `demand_level_case, demand_case, gas_deliverability_case,
capital_case, carbon_case, start_year, end_year, tariff_case, om_case`
— **missing 15**: `peak_multiple_case, reserve_margin, storage_cc_
case, gas_build_case, gas_availability_case, gas_retirement_case, gas_
capex_case, ccgt_efficiency_case, capital_recovery_case, dscr_case, us_
inflation, capital_treatment_case, concessional_envelope_case,
conversion_case, hydro_case`. `run_gas_regime_ndc_matrix(...)` (`:326-
333` and elsewhere) — **missing all 17** of that function's required
keys, same as `06_`'s Arm 2. **Cannot execute today.**

**1B.** GAS-2 — does EaaS financing relieve gas scarcity (shadow
price) relative to traditional financing, across gas regimes and NDC
arms, including a `baseline_no_policy` arm with no emissions cap at
all.

**1C. CLI-NEEDED.**

**1D. — must be fixed as part of this revival, per this task's own
Part 1D instruction.** **A37/L35**: `run_no_policy_gas_matrix_eaas`'s
own result dict (`:187-188`) hardcodes `"financing_regime":
"traditional"` and `"status": "traditional"` — a copy-paste leftover
from the sibling non-EaaS function (correct there, wrong here, since
this IS the EaaS-on function, confirmed by `scenario["financing_
regime"] = "eaas"` at `:144`). Every downstream read gates on
`r_yes.get("status") == "optimal"`, always False today — the moment
this file is revived (argument threading alone), `baseline_no_policy`
(the cleanest, no-cap-confound NDC arm) would silently report nothing
for its EaaS side, with no crash and no warning. **The fix**: at
`:187-188`, change `"financing_regime": "traditional"` → `"eaas"` and
`"status": "traditional"` → `"optimal"`. Also **A38/L35**: the same
function's result dict has no `"solar_total_mw"` key (present in the
other four NDC arms' schema) — `dict.get` silently defaults it to
`None` for this arm only; add the key for schema consistency in the
same commit.

**Why fixing A37 alongside the revival cannot move a solved value.**
`07_` has never successfully produced output under the current
`load_scenario()` signature (confirmed dead before AND after A37 is
fixed) — there is no existing result for A37's fix to change; it only
determines what the FIRST successful run reports.

---

### `scripts/08_run_gas3_regime_ndc_feasibility.py`

**1A.** No `argparse` (confirmed: 0 hits). `run_gas_regime_ndc_
matrix(...)` called twice (`:261-268`, `:292-299`), each supplying only
`econ, ndc_cap_scenario, gas_cases, financing_regime, capital_case,
cap_path` — **missing all 17 required keys** (`peak_multiple_case`
through `om_case`). **Cannot execute today.**

**1B.** GAS-3 — gas regime × NDC-cap feasibility matrix, plus a
structural/flat-level-equivalent shape-vs-level decomposition
(`STRUCTURAL_GAS_CASES` vs `FLAT_GAS_CASES`).

**1C. CLI-NEEDED.**

**1D.** **A41/L36**: this file is dead by the exact same mechanism as
the nine (now effectively twelve, per L27/L35/this report) other dead
runners, but was never itself added to either dead-runner enumeration
(L10's original nine, or L27's corrected list) — flagged here so this
revival doesn't get missed a second time. **A42/L36**: `GAS_ANCHOR_
TWH_TH = 40.66` (`:121`) is the pre-recalibration value; the live
anchor is 89.27 TWh_th — actively gates `"gas_disrupted"`/`"cap_and_
gas_disruption_coincide"`, so this would silently miscalibrate GAS-3's
own disruption-coincidence diagnostic even after the CLI fix, unless
corrected in the same commit (L36's own "What would resolve it"
already recommends exactly this bundling). **A43/L36**: docstring
cumulative-total figures stale by 2.3-3.4x — a text-only fix, cheap to
bundle in the same commit.

**Why fixing A42 alongside the revival cannot move a solved value.**
Same reasoning as `07_`/A37 — no existing output for this constant to
have corrupted; it only affects what the first successful run reports.

---

### `scripts/09_run_rel1_feasibility.py`

**1A.** No `argparse` (confirmed: 0 hits). `run_rel1_feasibility_
matrix(...)` (`:130-136`) supplies `econ, gas_cases, solar_build_cases,
capital_case, eps_levels` — **missing all 17 required keys** (`peak_
multiple_case` through `om_case`, `tariff_case` included — this
function's signature requires it, unlike `run_rel3_financing_
frontier`). **Cannot execute today.**

**1B.** REL-1 — reliability feasibility matrix: at what unserved-
energy threshold (`eps`) does each gas regime × solar-build-pace
combination become infeasible.

**1C. CLI-NEEDED.**

**1D.** No file-specific register entry. **Report, do not fix, per
Part 2's own rule**: `run_rel1_feasibility_matrix` (`src/optimize_
experiments.py:3478`) internally hardcodes `build_model(...,
reliability_mode="total", ...)` — the DEPRECATED alias, superseded
project-wide by `"annual"` (per `docs/phase_2_6_handoff.md`'s standing
convention, and per `09b_run_rel1_mode_sensitivity.py`'s own retirement
rationale below). `src/optimize_experiments.py:3430-3434`'s own comment
confirms this was deliberately left alone by the shared-function repair
as "a computation question, not a plumbing one." Reviving `09_`'s CLI
surface does not touch this — flagged here as a pre-existing,
orthogonal issue Zeus should know about before reading REL-1 output:
every REL-1 result, once produced, will be on the non-standard,
generally-superseded reliability mode, unless a SEPARATE, deliberate
decision is made to change `run_rel1_feasibility_matrix`'s own
hardcoded argument (which would be a `src/`-adjacent change outside
this task's whitelist regardless).

---

### `scripts/10_run_rel2_marginal_cost.py`

**1A.** No `argparse`. `run_rel2_marginal_cost_curves(...)` (`:243-
248`) supplies `econ, gas_cases, capital_case, eps_levels` — **missing
all 17 required keys**. **Cannot execute today.**

**1B.** REL-2 — marginal cost curves: shadow price of the reliability
constraint as a function of `eps`, per gas regime.

**1C. CLI-NEEDED.**

**1D.** No file-specific register entry. Uses the CORRECT `"annual"`
mode internally (confirmed via `run_reliability_sweep`, which this
function delegates to) — clean on the mode question, unlike `09_`.

---

### `scripts/11_run_rel3_financing_frontier.py`

**1A.** No `argparse`. `run_rel3_financing_frontier(...)` (`:215-221`)
supplies `econ, gas_cases, policy_configs, eps_levels, cap_path` —
**missing all 16 required keys** (this function's signature does not
require `tariff_case`, unlike REL-1/REL-2). **Cannot execute today.**

**1B.** REL-3 — financing frontier: reliability threshold vs. public-
capital envelope trade-off, by policy configuration.

**1C. CLI-NEEDED.**

**1D. — flagged for Part 2, same shape as A56, not explicitly named by
TASK.md but should be fixed in the same spirit.** **A45/L34**: `11_`'s
own `POLICY_CONFIGS` (`:161,169,177,185,193`) hardcodes the retired
`1.10`/`1.05` `required_margin` split, overwriting the correctly
`dscr_case`-sourced value inside `run_reliability_sweep`'s own
downstream call — the exact double-count `02_/03_/04_/15_` were all
fixed to stop doing. **Currently dormant only because `11_` is dead**;
reviving the CLI surface without also removing this override would
silently reintroduce a known-retired financing-cost bias in the very
first successful run. Recommended: remove the override in the same
commit as the CLI fix, mirroring the pattern already applied to `02_/
03_/04_/15_`.

**Why fixing A45 alongside the revival cannot move a solved value.**
`11_` has never successfully produced output under the current
signature — nothing existing is affected.

---

### `scripts/12_run_dem1_demand_sensitivity.py`

**1A.** `main()`'s first executable statement (`:313`) is `raise
NotImplementedError(...)` — confirmed via direct read, ~150 lines of
real sweep logic follow but are unreachable. Underneath the stub guard,
`run_dem1_demand_sensitivity(...)` (`:325-330`) supplies only `econ,
demand_level_cases, ndc_configs, cap_path` — **missing all 17 required
keys** — so even removing the stub guard alone would not revive it.
**Cannot execute today, on two independent grounds.**

**1B.** Docstring (live `__doc__`, verbatim): *"12_run_dem1_demand_
sensitivity.py — DEM-1 [SUSPENDED] ⚠ SUSPENDED pending Phase 2 tiered-
demand rebuild. - latent_low/latent_high were built as multiples of
COLLECTED energy (23.08), which is a revenue quantity, not demand.
Wrong base. - All quantified figures below predate the eta correction
... Do not run or quote."*

**1C. STUB.** The suspension is a deliberate, substantive judgement
(wrong demand base, not merely missing plumbing) — TASK.md's own
instruction for STUB is "leave it. Report what implementing it would
require." Implementing it would require: (1) removing the `Not
ImplementedError` guard, a judgement call about whether the suspension
reason still holds (this task's whitelist forbids editing `scenarios.
py`, where the "tiered demand architecture" the suspension notice waits
for would need to land — so this script cannot actually be un-suspended
within this task's scope regardless); (2) the same 17-key CLI threading
every other dead runner needs.

**1D.** A54/**L42** (fixed/declared last task): the file's detailed
rationale is a second, orphaned docstring, declared not merged (see
L42 for why merging was judged unsafe).

---

### `scripts/13_run_dem2_growth_gas_interaction.py`

**1A.** No `argparse`. `run_dem2_growth_gas_matrix(...)` (`:393-400`)
supplies `econ, demand_cases, gas_cases, financing_configs, policy_
configs, cap_path` — **missing all 16 required keys** (`peak_multiple_
case, reserve_margin, storage_cc_case, gas_build_case, gas_
availability_case, gas_retirement_case, gas_capex_case, ccgt_
efficiency_case, capital_recovery_case, dscr_case, us_inflation,
capital_treatment_case, concessional_envelope_case, conversion_case,
hydro_case, om_case`). Confirmed via `inspect.signature(...).bind(...)`:
`TypeError: missing a required argument: 'peak_multiple_case'`.
**Cannot execute today** — this is **A55**, already declared last task.

**1B.** DEM-2 — does demand growth's effect on cost/feasibility
interact with gas regime (super-additive, sub-additive, or
approximately independent)?

**1C. CLI-NEEDED** (this IS the fix A55 calls for).

**1D. — must be fixed as part of this revival, per this task's own
explicit instruction.** **A56/L34**: `run_dem2_growth_gas_matrix`
(`src/optimize_experiments.py:1840-1842`) itself overwrites the
correctly `dscr_case`-sourced `required_margin` with `13_`'s own
`FINANCING_CONFIGS`' hardcoded `1.10` literal for the `eaas` arm — the
same retired double-count as A45/A46. **This is a `src/optimize_
experiments.py` defect, not a `scripts/` one** — and `optimize_
experiments.py` is explicitly OUTSIDE this task's whitelist ("No
`src/` edits except `cli_arms.py`"). **This is therefore a genuine
BACKEND-BLOCKED element bundled with an otherwise CLI-NEEDED file**:
the CLI threading can be done within this task's whitelist; removing
the double-count cannot, since it lives inside the shared function
itself, not in `13_`'s own call to it. TASK.md's Part 2 says "Fix A37,
A55 and A56 as part of the revival" — for A56 specifically, the only
part fixable within this whitelist is `13_`'s OWN `FINANCING_CONFIGS`
literal (`:128-134`, `"required_margin": 1.10`) if `13_` stops passing
it into the shared function's `financing_configs` structure at all —
but `financing_configs` is a required positional argument whose
CONTENTS `run_dem2_growth_gas_matrix` reads and re-applies internally
(`optimize_experiments.py:1840-1842`); `13_` supplying a config dict
without a `required_margin` key would need the shared function to
tolerate that key's absence, which is a `src/` behaviour question, not
a `scripts/` one. **Flagged as a whitelist conflict for Part 2 to
resolve explicitly** (most likely resolution: fix the CLI threading
now, leave the double-count declared per L34, since removing it
requires touching `optimize_experiments.py`).

---

### `scripts/14_run_str1_storage_role.py`

**1A.** No `load_scenario()` call at all — `main()` (`:45-52`) is
exactly a `print(...)` of the suspension notice followed by `sys.
exit(0)`. **Cannot execute (produce results) by construction.**

**1B.** Docstring (verbatim excerpt): *"STR-1 asks what role battery
storage plays in the optimal system. It cannot be answered in the
current model, and this runner is intentionally disabled. ... At
annual resolution the storage energy constraint is discharge[t] <= RTE
* charge[t] ... Storage can only destroy energy at a capital cost, so
storage_add = 0 is strictly optimal in every scenario."*

**1C. STUB**, and substantively so — the suspension's stated technical
reason (storage has no possible role at ANNUAL resolution) does not
even depend on missing arguments; it is an algebraic fact about the
annual builder's own storage constraint.

**1D. A57/L39**: the file's own "RE-ENABLE WHEN" precondition ("Phase
2 introduces representative intra-annual slices... storage then
shifts day->evening") **has already shipped** — `src/optimize_model_
sliced.py` (self-labelled "Phase 2.1") already exists and implements
exactly this (`m.storage_season_balance`). Reviving STR-1 properly
would mean PORTING it to `build_model_sliced()`, not merely threading
CLI arguments through its current annual-builder logic — a genuine
redesign, not a plumbing fix, and whether storage actually wins
economically against the genset backstop under slices remains an open
empirical question L39 states explicitly. **Recommendation for Part 2:
leave this STUB as-is; update its own text to point at the sliced
builder (a `scripts/`-only text edit, within whitelist) but do not
attempt a working revival within a plumbing-only task.**

---

### Retired: `scripts/retired/09b_run_rel1_mode_sensitivity.py`

**1A.** `load_scenario(...)` (`:56-61`) supplies `demand_level_case,
demand_case, gas_deliverability_case, capital_case, carbon_case,
start_year, end_year, om_case` — **missing 16** (same list as `07_`'s
Arm 1, minus `tariff_case` which this one also lacks — 16 total
including `tariff_case`). **Cannot execute today**, independent of its
retirement status.

**1B.** Docstring: *"REL-1 sensitivity: reliability_mode = 'annual' vs
'total'. ... Expected finding: 'annual' mode is stricter, builds more
solar earlier in horizon, and has a higher (less feasible) threshold
eps at which the system becomes infeasible."* Retirement header:
*"RETIRED... This script sweeps `grid_service_mode` over ['annual',
'total']. That axis no longer has two live options: the 2.6 docket
(item D6) switched the grid-service obligation from 'total' to
'annual' permanently..."*

**1C. SUPERSEDED — by convention, with an important nuance (1E).**

**1E. The prior reading is independently checked and found PARTIALLY
WRONG.** `'total'` has NOT been removed from the codebase: `src/
optimize_model.py:976` and `src/optimize_model_sliced.py:1210` both
still implement a live `elif grid_service_mode == "total":` branch, and
it is actively exercised TODAY, not merely as inert legacy code —
`run_rel1_feasibility_matrix` (`src/optimize_experiments.py:3478`)
hardcodes `reliability_mode="total"`, and `09_run_rel1_feasibility.py`
calls this function as its sole engine (see `09_`'s own entry above).
**The retired script's claim that the axis "no longer has two live
options" overstates the case** — the SECOND option was not removed,
only demoted to an undeclared, un-swept internal default in one
surviving caller.

**That said, the substantive conclusion holds, for a different
reason.** The project has an explicit, stated standing convention
(`docs/phase_2_6_handoff.md`: *"Mode `annual`, never `total` — `total`
is slack under a rising deficit"*) against treating `'total'` as a
legitimate co-equal alternative to compare against `'annual'`. A sweep
BUILT to present the two as equally valid options (as this retired
script does) would misrepresent the project's own methodology, even
though the underlying code path both options exercise still exists.
**Verdict: the QUESTION this script asks (does mode strictness matter,
compared as two live alternatives) is not "answered elsewhere" so much
as declared out of scope by a standing convention this project has
committed to** — a materially different, more precise statement than
"D6 permanently removed the axis." Recommend: keep this script
retired, but correct its own header to state the nuance above rather
than the stronger, now-inaccurate "no longer has two live options"
claim — a `scripts/`-only text fix, within this task's whitelist, not
attempted here since Part 1 is inventory-only.

**1D.** `forensic_audit.md:1833-1848, 3431-3444` — already thoroughly
covered, re-confirmed clean each phase; only a directory-path
staleness noted in L10's own prose (not attributable to this file's own
header).

---

### Retired: `scripts/retired/09b_run_rel1_retirement_sensitivity.py`

**1A.** `load_scenario(...)` (`:122-132`) supplies `demand_level_case,
demand_case, gas_deliverability_case, capital_case, solar_build_case,
carbon_case, start_year, end_year, om_case` — **missing 16**. **Cannot
execute today**, independent of retirement status.

**1B.** Docstring: *"Tests whether the REL-1 finding (gas regime is
the primary determinant of reliability feasibility) is sensitive to
the assumed gas fleet retirement rate."* Retirement header: *"This
script sweeps the 680 MW/yr baseline gas-retirement PROXY at 0/340/
680/1360 MW/yr. That proxy is superseded by the `gas_retire_vintage_
{30,40,50}` arms... The vintage arms showed the flat proxy overstates
system cost by ~1.28 bn (5.9%) and storage by ~17.9%."*

**1C. SUPERSEDED — confirmed, cleanly, no nuance needed.**

**1E. Independently checked and CONFIRMED, unlike the sibling file
above.** `gas_retirement_case` is a genuine, current, sentinel-required
`load_scenario()` argument (`src/scenarios.py:2052`, `_GAS_RETIREMENT_
REQUIRED`), and `gas_retire_vintage_30/40/50` genuinely exist in the
live registry (`src/scenarios.py:1381-1383`). Register **L8** (`docs/
limitations_register.md:212-...`) confirms these vintage arms are load-
bearing everywhere, and its own table already reports the flat-proxy-
vs-vintage effect on objective and terminal storage. The vintage arms
are a BROADER and more realistic sensitivity axis (fleet-vintage-based,
not a single flat rate) than this retired script's simple 4-point
sweep. **Verdict: the retirement-sensitivity question is genuinely
answered elsewhere, and this script's specific proxy is a real,
superseded modelling artefact.** Recommend: keep retired, no revival
needed — this is the cleaner of the two retirement dispositions.

**1D.** Same two `forensic_audit.md` citations as the sibling file
above, same "model of doing it right" verdict.

---

### Deferred: `scripts/deferred/17_run_monte_carlo.py`

**1A.** `load_scenario(...)` (`:174-185`) supplies `demand_level_case,
demand_case, gas_deliverability_case, capital_case, solar_build_case,
land_case, carbon_case, start_year, end_year, om_case` — **missing
16**. **Cannot execute today**, independent of its own two self-
declared substantive defects.

**1B.** Docstring (verbatim excerpt): *"DEFERRED... This script is
structurally sound in conception but has two blocking defects, both
declared in its own text below... 1. LCOE is miscomputed... 2. The
demand-growth prior is unsourced... Until both are fixed this script
must not be run for thesis output."*

**1C. CLI-NEEDED, plus BACKEND-BLOCKED by its own declared defects** —
the two do not resolve each other. Threading the 17-key CLI surface
through makes it EXECUTE; it would still not be REPORTABLE without
separately fixing the LCOE formula and sourcing the demand-growth
prior (both `src/`-adjacent changes: the LCOE fix touches this script's
own post-processing, within whitelist; the demand-growth prior lives in
`scenarios.py`'s `demand_growth_prior()`, outside this task's
whitelist).

**1D.** Already declared, own header + register L10, L32; see also
Phase 1's A11 (a second, independent, unrelated module — `src/
stochastic.py` — carries the same shared-driver-independent-sampling
defect this script has, confirming the concern is real and recurring,
not specific to this one file).

**1F. — corrected framing, per this task's own request to check the
"fixed-plan re-dispatch" characterisation.** **Not accurate as stated.**
`17_`'s own docstring (`:29-31`): *"Unlike the deterministic replay
approach, this script re-optimises the full LP for each draw, ensuring:
Capacity plans adapt to each gas/demand realisation."* Confirmed in
code: each draw calls `build_model(...)` fresh (`:202-207`) then
`solve_model`. **This is full re-optimisation, the identical method
family `29_run_parameter_uncertainty.py` uses** (both call a builder +
solver per draw) — NOT the "fixed-plan re-dispatch" this task's own
framing and TASK.md's phrasing both use. The genuinely fixed-plan re-
dispatch mechanism is a wholly separate, third module never wired to
any runner: `src/stochastic.py` (`run_stochastic_simulation`), already
declared dead code (register L32, finding A11).

**What each actually varies**, read directly rather than assumed: `17_`
draws `demand_growth` (continuous, `Normal` from `demand_growth_
prior()`, unsourced) × `gas_regime` (categorical, `gas_probability_
weights()`) — INDEPENDENTLY, its own declared defect. `29_` draws five
CONTINUOUS PHYSICAL/COST parameters at the fixed central scenario
(`gas_availability_factor, gas_eta, solar_cf, om_blend_fraction,
conversion_capex_per_kw`) via Sobol sampling, and explicitly does not
touch demand growth or gas regime at all — its own docstring (`:5-27`)
states the scenario/parameter distinction directly: *"This model is
DETERMINISTIC with structured scenario analysis. The project's 38-arm
sweep answers SCENARIO uncertainty... This script answers a DIFFERENT
question: PARAMETER uncertainty."*

**Verdict: NOT SUPERSEDED.** `29_` does not ask `17_`'s question — a
demand-growth × gas-regime JOINT DISTRIBUTION via re-optimisation
(rather than the 38-arm sweep's discrete cross-tab) is not produced by
anything currently in this codebase. `17_` needs reviving IF that
question is judged worth asking (a Part-2-and-later decision, per this
task's own "revive, run, then decide" ordering) — but reviving it
correctly requires, at minimum: the 17-key CLI threading (in scope),
the LCOE formula fix (in scope, `scripts/`-only), the demand-growth
prior sourced (OUT of scope — `scenarios.py`), and the independent-
sampling defect corrected (a design question, arguably `scripts/`-only
since the sampling call is in `17_` itself, `:174` area — worth
Part 2 confirming precisely where `sample`/`np.random.normal`/`np.
random.choice` for these two draws actually live before promising a
fix).

---

## Files with no wrinkle beyond what the master table already states

Confirmed via the checks summarised in the master table, with no
additional detail needed: `00_build_emissions_cap.py`, `01_run_
baseline.py`, `01b_run_baseline_sliced.py`, `02_run_ndc_caps.py`, `03_
run_ndc_eaas.py`, `04_run_fin2_tariff_sweep.py`, `05_run_fin3_capital_
sweep.py`, `08_phase8_analysis.py`, `15_run_pol1_ndc_comparison.py`,
`16_plot_cost_decomposition.py`, `18_run_pareto_frontier.py`, `19_run_
carbon_price_sweep.py`, `20_run_cf1_solar_yield_sensitivity.py`, `21_
compute_resolution_premium.py`, `22_salvage_sensitivity.py`, `23_run_
selfgen_frontier.py`, `24_run_leakage_arm.py`, `25_run_obligation_
trajectory.py`, `26_run_f3_identification.py`, `27_run_hydro_minflow_
sensitivity.py`, `28_run_capital_frontier.py`, `29_run_parameter_
uncertainty.py`, `99_validate_baseline.py`, `99c_run_genset_arm_
sensitivity.py`, `_check_solver.py`, `_compare_2_6.py`, `build_gas_
deliverability.py`, `diag_fin2.py`, `f1_quantity_diagnostic.py`, `plot_
system_diagnostics.py`, `rpublic_diagnostic.py`.

**`plot_storage_regime.py`** is separately non-executable (A31/L41 —
stale hardcoded path plus a stale vocabulary map) but does not fit
this task's THREAD-ONLY/CLI-NEEDED repair mechanism at all — it is
not a `load_scenario()` caller, and its fix is a path correction plus a
vocabulary correction, not argument threading. Recommend Zeus treat it
as out of scope for this specific revival task, per L41's own already-
recorded remediation notes.

---

## Why none of this Part 1 work can move a solved value

Every check in this Part was one of: reading a call site's own source
text, calling `inspect.signature()` on an already-defined function (no
model built, no solve), or directly invoking `load_scenario()` with the
exact literal arguments a call site supplies (a pure function returning
a plain dict from CSV/YAML reads — no Pyomo model constructed, no
solver invoked, confirmed by reading `load_scenario()`'s own body,
which contains no `import pyomo` and no model-building calls). No
runner's `main()` was executed. No `build_model`/`build_model_sliced`
was called. No `.solve()` was ever reached. The two frozen anchors
(regression 21,811,042,472.821114 / 45,502.9754 MWh; central
27,954,000,000 / 65,612.985 MWh) are produced by `01_run_baseline.py`/
`01b_run_baseline_sliced.py` under specific arm sets — neither file was
touched, and neither is in the dead set.

---

## Encountered but not asked about

- **`04_run_fin2_tariff_sweep.py:697`** in `docs/merged_correction_
  plan_v2.md` (the D17 deferred-item text, distinct from D25) still
  frames this file as a future fix ("until `04_run_fin2_tariff_sweep.
  py` (a stub, D25) imposes a real tariff constraint") — already noted
  as encountered-but-out-of-scope in last task's report; still
  unresolved, still out of this task's named scope.
- **`22_salvage_sensitivity.py`, `23_run_selfgen_frontier.py`, `26_
  run_f3_identification.py`, `27_run_hydro_minflow_sensitivity.py`**
  all maintain their own hand-rolled, non-`cli_arms.py` argparse
  surfaces, fully correct today but a second, independent hand-copy of
  the same 17-flag pattern `cli_arms.py` exists specifically to
  centralise (register L27's own stated purpose). Not a defect — all
  four currently work — but a maintenance-debt observation outside
  this task's "revive the dead" scope.
- **`24_run_leakage_arm.py`, `25_run_obligation_trajectory.py`, `99c_
  run_genset_arm_sensitivity.py`** each still lack an in-file statement
  of the `conv_none`-only restriction A24 already names — cheap doc
  fixes, not attempted here since Part 1 is inventory-only and these
  three files are not in the dead set.
- **`src/stochastic.py`**'s exact relationship to `17_run_monte_
  carlo.py`'s own two declared defects (LCOE, prior) is now doubly
  confirmed distinct — both modules independently carry a correlated-
  sampling defect, and fixing one does not fix the other, since neither
  currently calls the other. Worth a register cross-reference at some
  point; not added here (register edits beyond the existing L1-L44 are
  outside this task's whitelist — "No... register").
- **Exact location of `17_`'s `demand_growth`/`gas_regime` sampling
  calls** was confirmed to be inside `17_run_monte_carlo.py` itself
  (not a shared function), which means — contrary to a first
  assumption — fixing the independent-sampling defect may be within
  this task's `scripts/`-only whitelist after all, not a `src/`-
  adjacent change. Flagged for Part 2 to confirm precisely before
  scoping.

---

## PART 2A — PLUMBING REVIVALS

Read-only verification (`inspect.signature`, direct `load_scenario()`
invocation) plus edits confined to the nine THREAD-ONLY/CLI-NEEDED
runners Part 1 identified, per the accepted split. `src/cli_arms.py`
was not touched — every runner's needs were already met by its
existing `include=`/`exclude=` mechanism. Nothing run, staged, or
committed. Reported incrementally as each runner was finished.

### Critical correction to the run commands, found while verifying Part 2C — flagged before any per-runner detail, since it applies to all nine

**Every one of the nine revived runners uses the ANNUAL builder
(`build_model()`), not the sliced builder, somewhere in its call
chain** — confirmed by tracing each shared function's own body, not
assumed:

| Runner | Calls | Builder used |
|---|---|---|
| `02_run_eaas.py` | `build_model()` directly | annual |
| `06_run_gas1_shadow_benchmarks.py` | `run_gas_regime_ndc_matrix()` | annual (`build_model(` inside it) |
| `07_run_gas2_eaas_gas_relief.py` | `run_gas_regime_ndc_matrix()` + its own `build_model()` calls | annual |
| `08_run_gas3_regime_ndc_feasibility.py` | `run_gas_regime_ndc_matrix()` | annual |
| `09_run_rel1_feasibility.py` | `run_rel1_feasibility_matrix()` | annual |
| `10_run_rel2_marginal_cost.py` | `run_rel2_marginal_cost_curves()` → `run_reliability_sweep()` | annual |
| `11_run_rel3_financing_frontier.py` | `run_rel3_financing_frontier()` | annual |
| `13_run_dem2_growth_gas_interaction.py` | `run_dem2_growth_gas_matrix()` | annual |
| `probe_2_6_structure.py` | `build_model()` AND `build_model_sliced()`, both directly | both (compares them, like `23_`) |

`src/optimize_model.py:295-302` raises `ValueError` unconditionally
whenever `scenario.get("conversion_available", False)` — true for
EVERY `conversion_case` value except `conv_none` (`conversion_
scenarios()`'s own docstring: *"conv_none reproduces pre-2.8 results
exactly: ... build_model() (annual) raises on any available arm"*).

**This is precisely the deviation Part 2C's own example names** — the
same structural reason `23_run_selfgen_frontier.py` must run
`conv_none`, now applying to all nine of these, for the identical
reason, and matching the ALREADY-DECLARED A24 pattern for `24_/25_/
99c_` (three siblings that also use the annual builder and already
carry this exact caveat). **Every run command below uses `--conversion-
case conv_none` in place of the central `conv_central`** — the ONLY
flag on which these nine deviate from the pure central set. Verified
directly for each: attempting `conversion_case="conv_central"` against
any of these nine's builder call would raise the `ValueError` above.

---

### `scripts/02_run_eaas.py`

**Classification:** THREAD-ONLY (15 of 17 flags already present).

**Arguments threaded:** `conversion_case`, `hydro_case` — the only two
of the 17 sentinel-required keys missing from the existing 15-flag CLI.

**Old:**
```python
from src.io import (load_econ, load_solar_capex_by_year)
from src.utils import json_safe
```
```python
        capital_treatment_case=args.capital_treatment_case,
        concessional_envelope_case=args.concessional_envelope_case,
        tariff_case=args.tariff_case,
        om_case=args.om_case,
    )
```
**New:**
```python
from src.io import (load_econ, load_solar_capex_by_year)
from src.utils import json_safe
from src.cli_arms import add_standard_arms
```
```python
    add_standard_arms(parser, include=["conversion_case", "hydro_case"])
    return parser.parse_args()
```
```python
        capital_treatment_case=args.capital_treatment_case,
        concessional_envelope_case=args.concessional_envelope_case,
        tariff_case=args.tariff_case,
        om_case=args.om_case,
        conversion_case=args.conversion_case,
        hydro_case=args.hydro_case,
    )
```
(also recorded in the output `summary` dict, matching the existing
`om_case` "recorded, not tagged" precedent).

**`load_scenario()` verification:** invoked directly with all 17 keys
(central values) — **SUCCESS**, no exception.

**Run command** (central set, `conv_none` per the builder finding
above):
```bash
python scripts/02_run_eaas.py \
  --peak-case peak_unsuppressed --storage-cc-case storage_cc_central \
  --gas-build-case gas_unconstrained --new-gas-deliverability constrained \
  --gas-availability-case gas_avail_nameplate --gas-retirement-case gas_retire_vintage_30 \
  --gas-capex-case gas_capex_central --ccgt-efficiency-case ccgt_eff_central \
  --capital-recovery-case cr_central --dscr-case dscr_central --us-inflation 0.020 \
  --capital-treatment-case f1_tranche --concessional-envelope-case env_dares \
  --tariff-case tariff_central --om-case om_central \
  --conversion-case conv_none --hydro-case hydro_flat
```

---

### `scripts/06_run_gas1_shadow_benchmarks.py`

**Classification:** CLI-NEEDED (no argparse existed).

**Arguments threaded:** all 17 sentinel keys, via `add_standard_arms(
parser, exclude=["gas_deliverability_case"])` (excluded because this
runner sweeps `gas_deliverability_case` itself, via `GAS_CASES`) — 18
of the 19 standard flags added (including `new_gas_deliverability`,
used by `load_scenario()` inside `run_no_policy_gas_matrix`); a derived
`shared_fn_kwargs` (17 keys, `new_gas_deliverability` stripped) used
for the `run_gas_regime_ndc_matrix()` call, since that function accepts
neither `gas_deliverability_case` nor `new_gas_deliverability` at all.

**Old:**
```python
def run_no_policy_gas_matrix(econ, gas_cases, cap_path):
    ...
        scenario = load_scenario(
            demand_level_case="served", demand_case="organic_central",
            gas_deliverability_case=gas_case, capital_case="moderate",
            carbon_case="no_policy", start_year=2025, end_year=MODEL_END_YEAR,
            om_case="om_central",
        )
```
```python
CANONICAL_GENSET = "genset_central"
def main():
    econ = load_econ(CANONICAL_VOLL, genset_case=CANONICAL_GENSET)
    all_results = []
    no_policy_results = run_no_policy_gas_matrix(econ, GAS_CASES, str(CAP_PATH))
    ...
    ndc_results = run_gas_regime_ndc_matrix(
        econ=econ, ndc_cap_scenario="ndc3_unconditional", gas_cases=GAS_CASES,
        financing_regime="traditional", capital_case="moderate", cap_path=str(CAP_PATH),
    )
```
**New:**
```python
def run_no_policy_gas_matrix(econ, gas_cases, cap_path, standard_kwargs):
    ...
        scenario = load_scenario(
            demand_level_case="served", demand_case="organic_central",
            gas_deliverability_case=gas_case, capital_case="moderate",
            carbon_case="no_policy", start_year=2025, end_year=MODEL_END_YEAR,
            **standard_kwargs,
        )
```
```python
CANONICAL_GENSET = "genset_central"

def parse_args():
    parser = argparse.ArgumentParser(...)
    add_standard_arms(parser, exclude=["gas_deliverability_case"])
    return parser.parse_args()

def main():
    args = parse_args()
    standard_kwargs = standard_arms_kwargs(args)
    shared_fn_kwargs = {k: v for k, v in standard_kwargs.items() if k != "new_gas_deliverability"}
    econ = load_econ(CANONICAL_VOLL, genset_case=CANONICAL_GENSET)
    all_results = []
    no_policy_results = run_no_policy_gas_matrix(econ, GAS_CASES, str(CAP_PATH), standard_kwargs)
    ...
    ndc_results = run_gas_regime_ndc_matrix(
        econ=econ, ndc_cap_scenario="ndc3_unconditional", gas_cases=GAS_CASES,
        financing_regime="traditional", capital_case="moderate", cap_path=str(CAP_PATH),
        **shared_fn_kwargs,
    )
```

**Verification:** `load_scenario()` invoked with the 18-key
`standard_kwargs` plus its own explicit `gas_deliverability_case` —
**SUCCESS**. `inspect.signature(run_gas_regime_ndc_matrix).bind(...)`
with the 17-key `shared_fn_kwargs` — **SUCCESS**.

**Run command:**
```bash
python scripts/06_run_gas1_shadow_benchmarks.py \
  --peak-case peak_unsuppressed --storage-cc-case storage_cc_central \
  --gas-build-case gas_unconstrained --new-gas-deliverability constrained \
  --gas-availability-case gas_avail_nameplate --gas-retirement-case gas_retire_vintage_30 \
  --gas-capex-case gas_capex_central --ccgt-efficiency-case ccgt_eff_central \
  --capital-recovery-case cr_central --dscr-case dscr_central --us-inflation 0.020 \
  --capital-treatment-case f1_tranche --concessional-envelope-case env_dares \
  --tariff-case tariff_central --om-case om_central \
  --conversion-case conv_none --hydro-case hydro_flat
```
(no `--gas-deliverability-case`: this runner sweeps it internally via
`GAS_CASES`.)

---

### `scripts/07_run_gas2_eaas_gas_relief.py`

**Classification:** CLI-NEEDED (no argparse existed).

**Arguments threaded:** same 18/17-key split as `06_` above.

**A37 fixed — the copy-paste bug, quoted both forms:**

Wrong (found):
```python
results.append({
    "gas_case":                gas_case,
    "ndc_scenario":            "baseline_no_policy",
    "financing_regime":        "traditional",
    "status":                  "traditional",
    "gas_shadow_mean":         float(np.mean(gs_vals)) if gs_vals else 0.0,
```
— inside `run_no_policy_gas_matrix_eaas`, which sets `scenario[
"financing_regime"] = "eaas"` three lines above building the model.

Right (fixed):
```python
results.append({
    "gas_case":                gas_case,
    "ndc_scenario":            "baseline_no_policy",
    "financing_regime":        "eaas",
    "status":                  "optimal",
    "gas_shadow_mean":         float(np.mean(gs_vals)) if gs_vals else 0.0,
```

**A second, independent execution blocker found and fixed in the same
file, while threading arguments — not itself one of the three named
traps, but the same shape as A45/A56.** `run_gas_regime_ndc_matrix()`
no longer accepts `required_margin` at all (removed under "F1, Step
3" — confirmed via its own docstring and `inspect.signature`). The
original EaaS-branch call still passed it:
```python
eaas_results = run_gas_regime_ndc_matrix(
    econ=econ, ndc_cap_scenario=ndc_cap, gas_cases=GAS_CASES,
    financing_regime="eaas", required_margin=cfg["required_margin"],
    capital_case=cfg["capital_case"], cap_path=str(CAP_PATH),
)
```
— this would have raised `TypeError: unexpected keyword argument
'required_margin'` regardless of the missing-standard-args issue.
Removed (see the file's own new comment at that call site). **Also
removed**, for the same reason and in the same spirit as every sibling
fix this session (`02_/03_/04_/15_`): `run_no_policy_gas_matrix_eaas`'s
own `scenario["required_margin"] = required_margin` line, which
overwrote the correctly `dscr_case`-sourced value `load_scenario(
**standard_kwargs)` had just set two lines above — the identical
retired double-count pattern, found here mid-revival. The now-unused
`required_margin` parameter was removed from `run_no_policy_gas_
matrix_eaas`'s own signature; `NDC_CASES`' own `required_margin` field
is left untouched (still used for the console `rm=` print label only).

**Verification:** `load_scenario()` (18-key `standard_kwargs` +
`gas_deliverability_case`) — **SUCCESS**. `inspect.signature(run_gas_
regime_ndc_matrix).bind(...)` for BOTH the `"traditional"` and
`"eaas"` branches (17-key `shared_fn_kwargs`, no `required_margin`) —
**SUCCESS**, both.

**Run command:**
```bash
python scripts/07_run_gas2_eaas_gas_relief.py \
  --peak-case peak_unsuppressed --storage-cc-case storage_cc_central \
  --gas-build-case gas_unconstrained --new-gas-deliverability constrained \
  --gas-availability-case gas_avail_nameplate --gas-retirement-case gas_retire_vintage_30 \
  --gas-capex-case gas_capex_central --ccgt-efficiency-case ccgt_eff_central \
  --capital-recovery-case cr_central --dscr-case dscr_central --us-inflation 0.020 \
  --capital-treatment-case f1_tranche --concessional-envelope-case env_dares \
  --tariff-case tariff_central --om-case om_central \
  --conversion-case conv_none --hydro-case hydro_flat
```

---

### `scripts/08_run_gas3_regime_ndc_feasibility.py`

**Classification:** CLI-NEEDED (no argparse existed).

**Arguments threaded:** all 17 keys `run_gas_regime_ndc_matrix` needs,
via `add_standard_arms(parser, exclude=["gas_deliverability_case",
"new_gas_deliverability"])` — this file never calls `load_scenario()`
directly (only the shared function, twice), so no split kwargs dict is
needed here, unlike `06_`/`07_`.

**Old:**
```python
from src.io import load_econ
from src.optimize_experiments import run_gas_regime_ndc_matrix
```
```python
def main():
    econ = load_econ(CANONICAL_VOLL, genset_case=CANONICAL_GENSET)
    ...
        structural = run_gas_regime_ndc_matrix(
            econ=econ, ndc_cap_scenario=ndc_name, gas_cases=STRUCTURAL_GAS_CASES,
            financing_regime="traditional", capital_case=ndc_cfg["capital_case"],
            cap_path=str(CAP_PATH),
        )
        ...
        flat = run_gas_regime_ndc_matrix(
            econ=econ, ndc_cap_scenario=ndc_name, gas_cases=FLAT_GAS_CASES,
            financing_regime="traditional", capital_case=ndc_cfg["capital_case"],
            cap_path=str(CAP_PATH),
        )
```
**New:**
```python
from src.io import load_econ
from src.optimize_experiments import run_gas_regime_ndc_matrix
from src.cli_arms import add_standard_arms, standard_arms_kwargs
```
```python
def parse_args():
    parser = argparse.ArgumentParser(...)
    add_standard_arms(parser, exclude=["gas_deliverability_case", "new_gas_deliverability"])
    return parser.parse_args()

def main():
    args = parse_args()
    standard_kwargs = standard_arms_kwargs(args)
    econ = load_econ(CANONICAL_VOLL, genset_case=CANONICAL_GENSET)
    ...
        structural = run_gas_regime_ndc_matrix(
            econ=econ, ndc_cap_scenario=ndc_name, gas_cases=STRUCTURAL_GAS_CASES,
            financing_regime="traditional", capital_case=ndc_cfg["capital_case"],
            cap_path=str(CAP_PATH), **standard_kwargs,
        )
        ...
        flat = run_gas_regime_ndc_matrix(
            econ=econ, ndc_cap_scenario=ndc_name, gas_cases=FLAT_GAS_CASES,
            financing_regime="traditional", capital_case=ndc_cfg["capital_case"],
            cap_path=str(CAP_PATH), **standard_kwargs,
        )
```

**`GAS_ANCHOR_TWH_TH = 40.66` (A42) and the docstring's stale
cumulative totals (A43) were left untouched** — both are constants/
documentation, out of scope for a plumbing-only revival per "change no
constant." Both remain exactly as Part 1 and the prior audit found
them; see the closing list.

**Verification:** `inspect.signature(run_gas_regime_ndc_matrix).bind(
...)` with the 17-key `standard_kwargs` — **SUCCESS**.

**Run command:**
```bash
python scripts/08_run_gas3_regime_ndc_feasibility.py \
  --peak-case peak_unsuppressed --storage-cc-case storage_cc_central \
  --gas-build-case gas_unconstrained --gas-availability-case gas_avail_nameplate \
  --gas-retirement-case gas_retire_vintage_30 --gas-capex-case gas_capex_central \
  --ccgt-efficiency-case ccgt_eff_central --capital-recovery-case cr_central \
  --dscr-case dscr_central --us-inflation 0.020 --capital-treatment-case f1_tranche \
  --concessional-envelope-case env_dares --tariff-case tariff_central --om-case om_central \
  --conversion-case conv_none --hydro-case hydro_flat
```
(no `--gas-deliverability-case`/`--new-gas-deliverability`: this
runner sweeps gas shape internally via `STRUCTURAL_GAS_CASES`/
`FLAT_GAS_CASES`, and the shared function accepts neither flag.)

---

### `scripts/09_run_rel1_feasibility.py`

**Classification:** CLI-NEEDED (no argparse existed).

**Arguments threaded:** all 17 keys `run_rel1_feasibility_matrix`
needs, via `add_standard_arms(parser, exclude=["gas_deliverability_
case", "new_gas_deliverability"])`.

**Old:**
```python
    rows = run_rel1_feasibility_matrix(
        econ=econ, gas_cases=GAS_CASES, solar_build_cases=SOLAR_BUILD_CASES,
        capital_case=CAPITAL_CASE, eps_levels=EPS_LEVELS,
    )
```
**New:**
```python
    rows = run_rel1_feasibility_matrix(
        econ=econ, gas_cases=GAS_CASES, solar_build_cases=SOLAR_BUILD_CASES,
        capital_case=CAPITAL_CASE, eps_levels=EPS_LEVELS, **standard_kwargs,
    )
```

**Reported, not fixed, per Part 2A's own rule ("if a runner's logic
looks wrong, report it and leave it"):** `run_rel1_feasibility_matrix`
(`src/optimize_experiments.py:3478`) internally hardcodes `build_
model(..., reliability_mode="total", ...)` — the deprecated alias,
project-wide superseded by `"annual"`. This lives inside `optimize_
experiments.py`, outside this task's whitelist regardless. Every
REL-1 result this runner now produces will be on the non-standard
mode unless a separate, deliberate `src/`-edit task changes it.

**Verification:** `inspect.signature(run_rel1_feasibility_matrix).
bind(...)` with the 17-key `standard_kwargs` — **SUCCESS**.

**Run command:**
```bash
python scripts/09_run_rel1_feasibility.py \
  --peak-case peak_unsuppressed --storage-cc-case storage_cc_central \
  --gas-build-case gas_unconstrained --gas-availability-case gas_avail_nameplate \
  --gas-retirement-case gas_retire_vintage_30 --gas-capex-case gas_capex_central \
  --ccgt-efficiency-case ccgt_eff_central --capital-recovery-case cr_central \
  --dscr-case dscr_central --us-inflation 0.020 --capital-treatment-case f1_tranche \
  --concessional-envelope-case env_dares --tariff-case tariff_central --om-case om_central \
  --conversion-case conv_none --hydro-case hydro_flat
```

---

### `scripts/10_run_rel2_marginal_cost.py`

**Classification:** CLI-NEEDED (no argparse existed).

**Arguments threaded:** all 17 keys `run_rel2_marginal_cost_curves`
needs, via the same exclusion pattern as `09_`.

**Old:**
```python
    all_rows = run_rel2_marginal_cost_curves(
        econ=econ, gas_cases=GAS_CASES, capital_case=CAPITAL_CASE, eps_levels=EPS_LEVELS,
    )
```
**New:**
```python
    all_rows = run_rel2_marginal_cost_curves(
        econ=econ, gas_cases=GAS_CASES, capital_case=CAPITAL_CASE, eps_levels=EPS_LEVELS,
        **standard_kwargs,
    )
```

**Confirmed clean on the reliability-mode question** (unlike `09_`):
`run_rel2_marginal_cost_curves` delegates to `run_reliability_sweep`,
which uses the correct `"annual"` mode.

**Verification:** `inspect.signature(run_rel2_marginal_cost_curves).
bind(...)` — **SUCCESS**.

**Run command:**
```bash
python scripts/10_run_rel2_marginal_cost.py \
  --peak-case peak_unsuppressed --storage-cc-case storage_cc_central \
  --gas-build-case gas_unconstrained --gas-availability-case gas_avail_nameplate \
  --gas-retirement-case gas_retire_vintage_30 --gas-capex-case gas_capex_central \
  --ccgt-efficiency-case ccgt_eff_central --capital-recovery-case cr_central \
  --dscr-case dscr_central --us-inflation 0.020 --capital-treatment-case f1_tranche \
  --concessional-envelope-case env_dares --tariff-case tariff_central --om-case om_central \
  --conversion-case conv_none --hydro-case hydro_flat
```

---

### `scripts/11_run_rel3_financing_frontier.py`

**Classification:** CLI-NEEDED (no argparse existed).

**Arguments threaded:** 16 keys `run_rel3_financing_frontier` needs,
via `add_standard_arms(parser, exclude=["gas_deliverability_case",
"new_gas_deliverability", "tariff_case"])` — `tariff_case` additionally
excluded because this function takes no top-level `tariff_case`
parameter at all; each `POLICY_CONFIGS` entry already carries its own
`"tariff_case": CANONICAL_TARIFF_CASE`, a per-arm constant left
untouched.

**Old:**
```python
    all_rows = run_rel3_financing_frontier(
        econ=econ, gas_cases=GAS_CASES, policy_configs=POLICY_CONFIGS,
        eps_levels=EPS_LEVELS, cap_path=str(CAP_PATH),
    )
```
**New:**
```python
    all_rows = run_rel3_financing_frontier(
        econ=econ, gas_cases=GAS_CASES, policy_configs=POLICY_CONFIGS,
        eps_levels=EPS_LEVELS, cap_path=str(CAP_PATH), **standard_kwargs,
    )
```

**Reported, not fixed — same shape as A45/A56, found independently
here.** `POLICY_CONFIGS` (`:161-193`) hardcodes the retired `1.10`/
`1.05` `eaas_margin` split, which `run_reliability_sweep`'s own
downstream handling applies as an override on top of the correctly
`dscr_case`-sourced `required_margin` — the identical double-count
`02_/03_/04_/15_/07_` were all fixed to stop doing. **Not fixed here**:
TASK.md names only A37/A55/A56 as required fixes this task; this is a
newly-confirmed FOURTH instance of the same shape, left alone per
"report it, do not fix it," and already recorded in register **L34**
(Part 3 of the prior task) as exactly this class of defect. Threading
this runner's CLI does not trigger it any differently than before —
it was already going to apply the moment `11_` could execute at all,
which is now.

**Also observed, not fixed:** `POLICY_CONFIGS`' `public_capital`/
`eaas_capital` tiers (`tight`/`moderate` by ambition) DO match the
canonical `02_/03_` convention — no capital-tier defect in this file,
unlike `07_` and `08_` below.

**Verification:** `inspect.signature(run_rel3_financing_frontier).
bind(...)` with the 16-key `standard_kwargs` — **SUCCESS**.

**Run command:**
```bash
python scripts/11_run_rel3_financing_frontier.py \
  --peak-case peak_unsuppressed --storage-cc-case storage_cc_central \
  --gas-build-case gas_unconstrained --gas-availability-case gas_avail_nameplate \
  --gas-retirement-case gas_retire_vintage_30 --gas-capex-case gas_capex_central \
  --ccgt-efficiency-case ccgt_eff_central --capital-recovery-case cr_central \
  --dscr-case dscr_central --us-inflation 0.020 --capital-treatment-case f1_tranche \
  --concessional-envelope-case env_dares --om-case om_central \
  --conversion-case conv_none --hydro-case hydro_flat
```
(no `--tariff-case`: supplied per policy arm internally.)

---

### `scripts/13_run_dem2_growth_gas_interaction.py`

**Classification:** CLI-NEEDED (no argparse existed) — **this is A55**.

**Arguments threaded (A55 fix):** 16 keys `run_dem2_growth_gas_matrix`
needs, via `add_standard_arms(parser, exclude=["gas_deliverability_
case", "new_gas_deliverability", "tariff_case"])` (`tariff_case`
excluded for the same reason as `11_` — supplied per `FINANCING_
CONFIGS` entry).

**Old:**
```python
    rows = run_dem2_growth_gas_matrix(
        econ=econ, demand_cases=DEMAND_CASES, gas_cases=GAS_CASES,
        financing_configs=FINANCING_CONFIGS, policy_configs=POLICY_CONFIGS,
        cap_path=str(CAP_PATH),
    )
```
**New:**
```python
    rows = run_dem2_growth_gas_matrix(
        econ=econ, demand_cases=DEMAND_CASES, gas_cases=GAS_CASES,
        financing_configs=FINANCING_CONFIGS, policy_configs=POLICY_CONFIGS,
        cap_path=str(CAP_PATH), **standard_kwargs,
    )
```

**A56 — reported prominently, per instruction, NOT fixed (root cause
outside this whitelist).**

`run_dem2_growth_gas_matrix` (`src/optimize_experiments.py:1839-1841`):
```python
scenario["financing_regime"] = fin_cfg["financing_regime"]
if fin_cfg["financing_regime"] == "eaas":
    scenario["required_margin"] = fin_cfg["required_margin"]
```
runs immediately after `load_scenario(dscr_case=dscr_case, ...)` has
already set `scenario["required_margin"]` correctly (now that A55
threads `dscr_case` through). `13_`'s own `FINANCING_CONFIGS["eaas"]`
(`:129-134`) hardcodes `"required_margin": 1.10`. **This line lives
inside `optimize_experiments.py`, outside this task's whitelist ("No
`src/` edits except `cli_arms.py`") — it cannot be fixed here.**

**What precisely would happen if `13_` is run now, post-A55:** it
would EXECUTE — no crash, no exception, all 72 solves complete — but
**every one of the 36 rows where `financing_arm == "eaas"` would carry
a `required_margin` of 1.10 regardless of whatever `--dscr-case` value
Zeus passes on the command line**, silently overriding the correctly-
sourced value the new CLI flag just supplied. The other 36 rows
(`financing_arm == "public_only"`, which never triggers the `if`
branch above) are unaffected and trustworthy from this specific angle.
**This is wrong output, not an error — stated prominently, per
instruction, because it determines whether `13_` may be run at all:**
`13_` should NOT be run for a figure spanning its `eaas` arm (any
`gas_premium`/`interaction_effect`/`eaas_value` figure touching
`financing_arm="eaas"`) until A56 is fixed in `optimize_experiments.
py` — a task this whitelist does not permit. The `public_only` half of
the matrix (36 of 72 rows) is not affected by A56 and may be read once
produced, independent of this caveat.

**Verification:** `inspect.signature(run_dem2_growth_gas_matrix).bind(
...)` with the 16-key `standard_kwargs` — **SUCCESS** (the function
now BINDS correctly — A55 is closed on the argument-threading
dimension; A56 is a silent-wrong-value issue `bind()` cannot detect,
which is exactly why it needed tracing by hand, not signature-checking
alone).

**Run command** (see the A56 warning above before using this for
anything beyond a mechanical execution check):
```bash
python scripts/13_run_dem2_growth_gas_interaction.py \
  --peak-case peak_unsuppressed --storage-cc-case storage_cc_central \
  --gas-build-case gas_unconstrained --gas-availability-case gas_avail_nameplate \
  --gas-retirement-case gas_retire_vintage_30 --gas-capex-case gas_capex_central \
  --ccgt-efficiency-case ccgt_eff_central --capital-recovery-case cr_central \
  --dscr-case dscr_central --us-inflation 0.020 --capital-treatment-case f1_tranche \
  --concessional-envelope-case env_dares --om-case om_central \
  --conversion-case conv_none --hydro-case hydro_flat
```

---

### `scripts/probe_2_6_structure.py`

**Classification:** CLI-NEEDED (had 4 of its own flags, none of the 17
standard ones).

**Arguments threaded:** all 17 sentinel keys, via `add_standard_arms(
ap, exclude=["gas_deliverability_case"])` — excluded because this
probe's own `load_scenario()` call already hardcodes `gas_
deliverability_case="baseline"`, a pre-existing constant left
untouched.

**Old:**
```python
    ap.add_argument("--capital-case", default="unconstrained",
                    help="tight | moderate | adequacy | expansion | unconstrained")
    args = ap.parse_args()

    scenario = load_scenario(
        demand_level_case="served", demand_case="organic_central",
        capital_case=args.capital_case, gas_deliverability_case="baseline",
        solar_build_case="deployment_unconstrained", land_case="loose",
        carbon_case="no_policy", start_year=2025, end_year=MODEL_END_YEAR,
        om_case="om_central",
    )
```
**New:**
```python
    ap.add_argument("--capital-case", default="unconstrained",
                    help="tight | moderate | adequacy | expansion | unconstrained")
    add_standard_arms(ap, exclude=["gas_deliverability_case"])
    args = ap.parse_args()
    standard_kwargs = standard_arms_kwargs(args)

    scenario = load_scenario(
        demand_level_case="served", demand_case="organic_central",
        capital_case=args.capital_case, gas_deliverability_case="baseline",
        solar_build_case="deployment_unconstrained", land_case="loose",
        carbon_case="no_policy", start_year=2025, end_year=MODEL_END_YEAR,
        **standard_kwargs,
    )
```
(`om_case="om_central"` — previously the only sentinel key present, as
a hardcoded literal — is now sourced from the new `--om-case` flag
instead, same value, matching the run command below.)

**Verification:** `load_scenario()` invoked directly with the 17-key
`standard_kwargs` plus the probe's own explicit `capital_case`/`gas_
deliverability_case` — **SUCCESS**.

**Run command** (this probe builds BOTH builders itself, exactly like
`23_` — `conv_none` required for the same reason):
```bash
python scripts/probe_2_6_structure.py \
  --capital-case unconstrained \
  --peak-case peak_unsuppressed --storage-cc-case storage_cc_central \
  --gas-build-case gas_unconstrained --new-gas-deliverability constrained \
  --gas-availability-case gas_avail_nameplate --gas-retirement-case gas_retire_vintage_30 \
  --gas-capex-case gas_capex_central --ccgt-efficiency-case ccgt_eff_central \
  --capital-recovery-case cr_central --dscr-case dscr_central --us-inflation 0.020 \
  --capital-treatment-case f1_tranche --concessional-envelope-case env_dares \
  --tariff-case tariff_central --om-case om_central \
  --conversion-case conv_none --hydro-case hydro_flat
```
(`--busbar`/`--genset-case`/`--voll` left at their own defaults, per
the probe's own documented purpose of testing the commit-A VoLL
fallback; pass `--genset-case genset_central` instead to probe the
priced-genset path if that is what Zeus wants to check.)

---

## PART 2B — `02_run_eaas.py`, in full

**When did it break?** `conversion_case` became sentinel-required in
commit `84a9580` ("Phase 2.8: OCGT-to-CCGT conversion tiers...",
2026-08-29 18:05:11); `hydro_case` became sentinel-required in `f43a54c`
("Phase 1.5: hydro_case arms...", 2026-08-29 19:48:57), roughly 1h43m
later, same day. **`02_run_eaas.py` has been dead since 2026-08-29
19:48:57** (the later of the two).

**It was touched twice since breaking, by edits that did not notice.**
`66cd29e` ("Thread om_case through all load_scenario call sites...",
2026-09-07) edited this EXACT `load_scenario()` call site — adding
`om_case` — nine days after both keys had already become required, and
did not add `conversion_case`/`hydro_case`. `8a5b094` ("Thread the
fleet_gas_* rename through 01_ and 02_...", 2026-09-18) touched this
file again, eleven days after that, for an unrelated rename, again
without noticing. **A commit whose entire stated purpose was "thread
om_case through ALL load_scenario call sites" passed directly over
this exact defect.**

**Has anything consumed its output since?** `results/eaas/` holds
exactly three files: `summary.json`, `diagnostics.json`, `timeseries.
csv` — all bare-named (the pre-tag-scheme convention; current code
writes `summary__{tag}.json` etc.), all dated **2026-08-23 13:44** —
six days BEFORE the break, and from before either `conversion_case` or
`hydro_case` existed as concepts at all. **Confirmed nothing reads
these bare paths**: `grep -rn` for `results.*eaas.*summary\.json` (and
the diagnostics/timeseries equivalents) across every `.py`/`.md` file
in the repo returns zero hits outside this script's own writer.
Confirmed not cited: `docs/findings_frozen.md`, `docs/frozen_state_
2026_09_10.md`, `docs/merged_correction_plan_v2.md` all return zero
hits for `results/eaas` or `02_run_eaas`.

**The stale output, named as instructed:**
- `results/eaas/summary.json`
- `results/eaas/diagnostics.json`
- `results/eaas/timeseries.csv`

All three predate this file's own break by six days and predate the
current tag-based output-naming convention entirely. **Flagged, not
deleted** (deletion is not in this task's whitelist) — a future reader
should not treat these as current EaaS output; they are a different,
older model epoch under a superseded file-naming scheme.

**Why this matters, confirmed:** `03_run_ndc_eaas.py` — the OTHER,
NDC-capped EaaS runner — was never dead (its `load_scenario()` call
has carried both `conversion_case` and `hydro_case` correctly since
before this task began, confirmed in Part 1). **No currently-published
EaaS figure traces to `02_run_eaas.py`** — checked directly above; the
uncapped-EaaS question this script alone answers has produced no
citable output since 2026-08-23, and none since the code that would
make a fresh run meaningful (`89a9580`/`f43a54c`) existed at all.

---

## PART 2C — additional flags, beyond the conv_none finding

**Central arm set differs necessarily for all nine — see the
conv_none finding above, which subsumes this section's own instruction
("flag any runner whose central arm set differs")**: every one of the
nine revived runners requires `--conversion-case conv_none` rather
than the central `conv_central`, for the structural reason given at
the top of Part 2A. No other flag deviates from the pure central set
for any of the nine.

**Signature-satisfaction verification, summarised** (detail per runner
above): all nine `inspect.signature(...).bind(...)` / direct `load_
scenario()` checks passed on first attempt after the edits above — no
runner required a second round of fixes to satisfy its target
function's signature.

---

## Closing disposition table

| Runner | Disposition | One-line reason |
|---|---|---|
| `02_run_eaas.py` | **REVIVED** | 2 missing keys threaded (`conversion_case`, `hydro_case`); `load_scenario()` verified. |
| `06_run_gas1_shadow_benchmarks.py` | **REVIVED** | Full CLI added; both call sites verified. |
| `07_run_gas2_eaas_gas_relief.py` | **REVIVED** | Full CLI added; A37 fixed; a second, independent `required_margin`-parameter-removal blocker found and fixed; both call sites verified. |
| `08_run_gas3_regime_ndc_feasibility.py` | **REVIVED** | Full CLI added; both call sites verified; `GAS_ANCHOR_TWH_TH`/docstring left untouched (constants, out of scope). |
| `09_run_rel1_feasibility.py` | **REVIVED** | Full CLI added; verified; stale `reliability_mode="total"` reported, not fixed (outside whitelist). |
| `10_run_rel2_marginal_cost.py` | **REVIVED** | Full CLI added; verified; confirmed clean on reliability mode. |
| `11_run_rel3_financing_frontier.py` | **REVIVED** | Full CLI added; verified; A45-shaped `required_margin` double-count reported, not fixed (outside whitelist). |
| `13_run_dem2_growth_gas_interaction.py` | **REVIVED (A55 closed)** | Full CLI added; verified executable — **but see A56: the `eaas`-arm half of its output is wrong, not merely unverified, until a `src/`-level fix lands.** |
| `probe_2_6_structure.py` | **REVIVED** | Full CLI added; verified against both `build_model`/`build_model_sliced` paths. |

**All nine: REVIVED on the execution dimension this task addresses.**
None is BLOCKED or NOT ATTEMPTED. `13_`'s revival carries an explicit,
prominent caveat (A56) that Zeus must read before using any `eaas`-arm
figure from it.

---

## Why none of this can move a solved value

Every edit in this task added or rerouted CLI arguments and fixed two
call-site bugs (A37; the `required_margin`-parameter-removal blocker in
`07_`) that previously prevented these nine scripts from reaching
`build_model()`/`build_model_sliced()` at all. None of the nine dead
runners has ever successfully produced output under its current,
now-fixed calling convention — each was either missing arguments
outright (a `TypeError`/`ValueError` before any model was built) or, for
`02_run_eaas.py`, producing output from an epoch six days before the
break that nothing currently reads or cites. There is no existing,
currently-relied-upon result for any of today's edits to change: fixing
a call site that previously could not run at all cannot retroactively
alter a number that was never produced under it. `01_run_baseline.py`
and `01b_run_baseline_sliced.py` — the two scripts that reproduce the
frozen regression and central anchors — were not touched, are not in
the dead set, and their own `load_scenario()` calls are unrelated to
any edit made this task. `src/optimize_model.py`, `src/optimize_model_
sliced.py`, `src/scenarios.py`, and `src/optimize_experiments.py` were
read for verification only — confirmed via `git status --porcelain`,
none was written to; the only files edited are the nine listed above,
none of which is on the anchors' own call path.

---

## Encountered but not asked about

- **A38** (`07_run_gas2_eaas_gas_relief.py`'s `run_no_policy_gas_
  matrix_eaas` still has no `"solar_total_mw"` key, unlike the shared-
  function schema the other four NDC arms use) — confirmed still
  present; not fixed, since it does not block execution and was not
  named among the three required fixes.
- **A newly-found capital-envelope tier discrepancy in `07_`,** the
  same shape as the already-declared/fixed A47: `NDC_CASES["ndc3_
  conditional"]["capital_case"]` is `"tight"` (`:98-99`), where the
  canonical convention (`02_run_ndc_caps.py`, `03_run_ndc_eaas.py`,
  `11_run_rel3_financing_frontier.py`) uses `"moderate"` for every
  conditional arm. Reported, not fixed — this is a "constant that
  looks wrong," explicitly out of scope for a plumbing-only revival.
- **A newly-found capital-envelope tier discrepancy in `08_run_gas3_
  regime_ndc_feasibility.py`,** more severe: `NDC_SCENARIOS` (`:113-
  118`) uses `"moderate"`/`"expansion"` for unconditional/conditional
  throughout — the EXACT stale pair A47 already found and fixed in
  `15_run_pol1_ndc_comparison.py` last task, here in a THIRD file,
  previously invisible because `08_` could not execute at all until
  today. Reported, not fixed, for the same reason as above.
- **`13_run_dem2_growth_gas_interaction.py`'s `FINANCING_CONFIGS`
  uses a single, flat `capital_case="moderate"` for both financing
  arms, independent of which `POLICY_CONFIGS` (NDC ambition) arm is
  active** — unlike every sibling script's ambition-dependent tight/
  moderate split. This may be a deliberate DEM-2 design choice (holding
  capital fixed to isolate the demand/gas interaction cleanly) rather
  than an oversight; noted without a verdict, since judging it is
  explicitly this task's later phase, not this one.
- `probe_2_6_structure.py`'s own `--capital-case` default
  (`"unconstrained"`) and `--voll`/`--genset-case` defaults were left
  exactly as they were — not part of the sentinel-required set, and
  changing them would be a computation choice, not a plumbing one.
- Whether `13_run_dem2_growth_gas_interaction.py`'s own demand-growth/
  gas-regime sampling (Part 1's 1F note about `17_run_monte_carlo.py`)
  has any relationship to this runner was re-checked: none — `13_`
  re-optimises over a fixed, discrete `DEMAND_CASES`/`GAS_CASES` grid
  (not a Monte Carlo draw), unrelated to `17_`'s independent-sampling
  defect. Confirmed distinct, mentioned only to close the loop from
  Part 1.
- `results/eaas/`'s three stale files (Part 2B) were flagged but not
  moved, renamed, or deleted — outside this task's whitelist (no file
  operations beyond the nine runners and the `cli_arms.py` option were
  authorised).

**Then STOP, per instruction. Nothing committed, nothing run this
task** — every verification above was `inspect.signature`, a direct,
side-effect-free `load_scenario()` invocation (no model, no solve), or
a read. `git status --porcelain` confirms exactly the nine runner files
modified, plus this report; nothing staged.
