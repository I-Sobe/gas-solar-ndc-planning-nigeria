# Forensic audit — Phase 1 (`src/`)

Adversarial-reader pass over every file in `src/`, looking for defects
NOT already declared in `docs/limitations_register.md` (L1-L28, read in
full), `docs/findings_frozen.md`'s scoped-conditions list (read in
full), and `docs/merged_correction_plan_v2.md`'s DO-NOT-REPORT table
and DEFERRED DEFECTS (D1-D8, read in full). Nothing edited. Nothing
run. Nothing committed. Written incrementally, file by file, per
instruction.

Frozen-state anchors used to check every hardcoded figure against:
central **27.954 bn** / storage **65,613.0 MWh**, regression
**21,811,042,472.821114** / **45,502.9754 MWh**, backcast **−3.2%**,
demand base **37.09 TWh**, η **0.287**.

---

## `src/utils.py` — read in full

### A1 — Three of five module functions have zero callers anywhere

**Fingerprint:** F1 (defined, plausible, never read).

**Location:** `src/utils.py:22-59` (`interpolate_series`), `:62-78`
(`mw_to_twh`), `:100-113` (`year_index`).

**What is wrong.** `grep -rn "interpolate_series\|mw_to_twh\|year_index"
src/ scripts/` (excluding their own `def` lines) returns nothing. All
three are plausible, correctly-implemented, dependency-free helpers
that no file in this project imports or calls. The other two functions
in the same file (`assert_non_negative`, `json_safe`) are used
extensively (7 files, 20+ modules/scripts respectively) — confirming
this is not a stale-file situation, just three specific dead functions
sitting beside two live ones.

**Why it matters.** Nothing reported depends on any of the three — they
cannot silently produce a wrong number because nothing reads their
output. The risk is purely a maintenance/trust one: a future author
sees `interpolate_series` and may reasonably assume some part of the
pipeline fills gaps in a demand or capex series with linear
interpolation, when in fact no series in this project is ever passed
through it.

**Severity:** LIVE-LATENT (zero current consumers; would only matter if
something started calling one of these three).

**Declared?** No. Not in the register, not in the O&M/`extract_
planning_diagnostics`/reporting-field "defined, plausible, never read"
instances the task's own F1 fingerprint cites as precedent — those are
three different objects (`STORAGE_OM_PER_MWH_YR`, `extract_planning_
diagnostics`, `npv_gross_capex_report_window_usd`), none of which is in
this file.

**Read in full. No other findings in this file.**

---

## `src/postprocess.py` — read in full

### A3 — The entire module is dead code

**Fingerprint:** F1 (defined, plausible, never read).

**Location:** `src/postprocess.py:1-19`, `export_benchmark_comparison`
— the only function this file defines.

**What is wrong.** `grep -rn "export_benchmark_comparison" src/
scripts/` returns only its own definition. This is not a helper
buried in a large file (like A1) — it is a dedicated module, with its
own file, that has never had a caller. The logic itself (comparing a
gas shadow price series against export/domestic netback benchmarks) is
plausible and reads as though it were built to support GAS-1
("gas shadow price vs. export/domestic benchmarks," per L27's own
description of that research question) but nothing wires it in even
where `06_run_gas1_shadow_benchmarks.py` (the script whose name matches
this exact comparison) is concerned — that script computes its own
benchmark comparison inline rather than importing this module (checked
by the same grep: `06_` does not appear in the caller list at all).

**Why it matters.** Nothing reported depends on it — it cannot produce
a wrong number because it never runs. Named for completeness of the
"zero callers" pattern class, and because a whole unused file is a
stronger instance of F1 than an unused function inside an otherwise-
live file.

**Severity:** LIVE-LATENT.

**Declared?** No.

**Read in full. No other findings in this file.**

---

## `src/gas_supply.py` — read in full

No findings. `gas_available_power()` (the function actually used
throughout the model for `gas_available_twh_th`/`gas_scenario`) is
correctly guarded: raises on a missing CSV, a missing required column,
or missing years for the requested scenario/range; no silent fallback.
Sole reader confirmed to be `extract_planning_diagnostics` and its
various runner callers.

---

## `src/gas_balance.py` — read in full

### A4 — Second entire dead module: a superseded gas-availability model

**Fingerprint:** F1 (defined, plausible, never read).

**Location:** `src/gas_balance.py:29-69` (`project_gas_availability_th`)
and `:72-94` (`gas_to_power_electricity_twh`) — the only two functions
this file defines.

**What is wrong.** `grep -rn "project_gas_availability_th\|gas_to_
power_electricity_twh" src/ scripts/` returns only their own
definitions. Zero callers anywhere. Unlike `postprocess.py` (A3, a
standalone comparison utility that was apparently never wired in), this
file's docstring describes it as THE gas-availability projection
mechanism ("Implements annual gas availability... projection")
via a simple exponential-decline model from a single base year
(`q0_th * (1+decline_rate)^t`). The model actually in use throughout
this project reads gas deliverability from a CSV via a completely
different function, `gas_available_power()` in the sibling file
`src/gas_supply.py`, keyed by a `gas_deliverability_case` scenario arm
with named trajectory shapes (downside/baseline/upside/shock_recovery,
plus the GMP policy-target family — register L13). This file appears to
be an earlier, single-parameter decline-curve approach to the same
question, superseded by the CSV/arm-based system, never removed.

**Why it matters.** Nothing reported depends on it — confirmed zero
callers, so it cannot silently produce a wrong number. Flagged because
a reader of `src/` (an examiner, or a future contributor) encountering
this file's docstring — which reads as a confident, current
description of "the" gas-availability mechanism — would form an
incorrect model of how gas supply is actually handled in this project
unless they separately discovered `gas_supply.py` and noticed this
one is never invoked. That is a different, more misleading failure
mode than an unused one-line helper: the docstring actively describes
capability the codebase no longer uses this way.

**Severity:** LIVE-LATENT (no current consumer; the misleading-
documentation aspect is COSMETIC on its own, but paired with a fully
dead, non-trivial exponential-decline implementation it is closer to a
stale-architecture trap for a future reader than a pure style issue).

**Declared?** No.

**Read in full. No other findings in this file.**

---

## `src/solar.py` — read in full

### A5 — Module docstring names two functions that do not exist in the file

**Fingerprint:** F7 (documentation that contradicts the code), reported
as "other" since it is a docstring-vs-its-own-file mismatch rather than
a separate `docs/` file contradicting `src/`.

**Location:** `src/solar.py:8-11`:
```python
Functions:
    load_solar_data(filepath)
    compute_capacity_factor(irradiance, derate=0.8)
    solar_generation(capacity, capacity_factor)
```
The file defines exactly two functions: `solar_generation` and
`solar_capacity_trajectory` (`:24-53`, `:56-96`). `load_solar_data` and
`compute_capacity_factor` do not exist anywhere in this file, `src/`,
or `scripts/` (`grep -rn "load_solar_data\|compute_capacity_factor"`
returns only this docstring line).

**Why it matters.** Nothing reported depends on the docstring text
itself. A reader relying on this module's own header to understand
its interface would look for a raw-irradiance-to-capacity-factor
pipeline that does not exist — this project's actual solar capacity
factor (`solar_cf`, register L26) is a single sourced scalar set in
`scenarios.py`, not computed from an irradiance file anywhere in this
codebase.

**Severity:** COSMETIC (no failure mode — nothing calls or relies on
the claimed functions existing).

**Declared?** No.

### A6 — `solar_capacity_trajectory` has zero callers

**Fingerprint:** F1.

**Location:** `src/solar.py:56-96`.

**What is wrong.** `grep -rn "solar_capacity_trajectory" src/
scripts/` returns only its own definition. A linear capacity build-out
helper with no caller — this project's actual solar capacity path is
an LP decision variable (`solar_public_add`/`solar_eaas_add`), not a
precomputed linear trajectory.

**Why it matters.** Nothing reported depends on it.

**Severity:** LIVE-LATENT.

**Declared?** No.

**`solar_generation`** (the plain function, distinct from the Pyomo
Expression `m.solar_generation` used throughout both builders) IS
called once, `src/optimize_experiments.py:276` — confirmed a live
consumer, no finding there.

**No other findings in this file.**

---

## `src/demand.py` — read in full

### A7 — `apply_access_adjustment` has zero callers

**Fingerprint:** F1.

**Location:** `src/demand.py:84-145`.

**What is wrong.** `grep -rn "apply_access_adjustment" src/ scripts/`
returns only its own definition. A fully-implemented, carefully unit-
guarded (checks fraction-vs-percentage consistency between
`access_series` and `base_access`, raises on length mismatch) function
with no caller anywhere.

**Why it matters.** Nothing reported depends on it. `project_baseline_
demand` (the other function in this file) is heavily used (`scenarios.
py`, both builders, `optimize_experiments.py`) — confirming this is a
single unused function in an otherwise fully-wired file, not a stale
module.

**Severity:** LIVE-LATENT.

**Declared?** No.

**No other findings in this file.**

---

## `src/dispatch.py` — read in full

No findings in this file itself — `dispatch_energy` is correctly
guarded (length checks, non-negativity, explicit units in its return
dict) and is a live consumer's dependency (see A8 below, found via this
file's one caller).

### A8 — A live, called function silently falls back to the RETIRED, explicitly-wrong gas efficiency value

**Fingerprint:** F2 (silent default on a load-bearing value) — and a
particularly sharp instance, since the fallback is not merely
"unsourced" but is the SPECIFIC number this project's own history
records as wrong and replaced.

**Location:** `src/optimize_experiments.py:273`, inside
`run_deterministic_scenario()`:
```python
eta =scenario.get("gas_eta", 0.43)
```

**What is wrong.** Every other gas-efficiency read in this codebase
uses `scenario["gas_eta"]` directly (a required, unconditionally-set
key, `scenarios.py:2836`, value 0.287) or raises if absent
(`optimize_model_sliced.py:463-465`: `eta = scenario.get("gas_eta",
None); if eta is None: raise ValueError(...)`). This one site falls
back to **0.43** — not an arbitrary placeholder, but the EXACT figure
`docs/merged_correction_plan_v2.md`'s own sourcing ledger records as
retired for being wrong: *"η = 0.287 from the identity observed TWh_e ÷
observed TWh_th... Replaced 0.43 (CCGT figure on an OCGT-dominated
fleet)"* (line 1.1). If this fallback ever fires, it does not merely
use an unsourced number — it silently resurrects a specific,
by-name-retired figure that overstates the fleet's true efficiency by
50% (0.43 vs 0.287).

**Is it live?** Checked, not assumed: `run_deterministic_scenario` is
explicitly labelled *"LEGACY NON-LP SIMULATOR — NOT for thesis result
tables... The Monte Carlo path no longer uses this function"*
(`optimize_experiments.py:246-256`), and a naive read might conclude it
is dead. It is not: `grep -rn "run_deterministic_scenario"` shows it is
still called by `scripts/02_run_eaas.py:301` — an active script whose
OTHER outputs feed reported EaaS findings (F-10, F-12, F-13). The
deterministic-dispatch output itself (`det_output`) is used only to
print `det_unserved` to the console (`02_run_eaas.py:337-338`) — it is
not written into any persisted `diagnostics.json`/`summary.json`, so no
currently-published NUMBER depends on this specific line.

**Why the fallback does not currently fire.** `02_run_eaas.py` builds
its scenario via `load_scenario()`, which unconditionally sets
`scenario["gas_eta"] = 0.287` in its returned dict — so `.get("gas_eta",
0.43)` finds the key present and never reaches the fallback in this
call path.

**Why it matters anyway.** `run_deterministic_scenario(scenario, econ,
capacity_paths)` takes a bare `scenario` dict as a parameter — nothing
in its signature or body enforces that the caller built it via
`load_scenario()`. A future caller passing a hand-built or partially-
stripped scenario dict (exactly the kind of "quick feasibility sketch"
this function's own docstring says it is for) that omits `gas_eta`
would silently get 0.43 instead of an error — the opposite of this
project's otherwise-consistent no-silent-defaults discipline (register
L9, L19, L21 all name this exact failure shape for other parameters),
and specifically the RETIRED value, not a merely-unstated one.

**Severity:** TRAP — correct today (the one live call path always
supplies `gas_eta`), but positioned so a plausible future caller
(anyone using this function per its own documented purpose — a quick,
hand-supplied-capacity-path sketch) breaks silently, reintroducing a
50%-efficiency error the project's own sourcing ledger already
identified and fixed everywhere else.

**Declared?** No. Not in the register. `merged_correction_plan_v2.md`
records that 0.43 WAS the old figure and was replaced (sourcing ledger,
η row) but does not know this fallback literal still exists.

---

## `src/storage.py` — read in full

No findings. The file already carries its own accurate "DOC-CODE
PARITY" declaration (`:19-36`) stating plainly that this class's
symmetric-√η charge/discharge convention is NOT what either LP builder
uses, and naming its one real consumer (`run_deterministic_scenario`,
already flagged at A8). Nothing further to add — the class is
internally consistent and correctly guarded (SOC invariants checked
after every charge/discharge).

---

## `src/economics.py` — read in full

### A9 — `npv()` and `discount_factor()` are unreachable from any live code path

**Fingerprint:** F1.

**Location:** `src/economics.py:133-148` (`discount_factor`), `:151-169`
(`npv`).

**What is wrong.** `npv()` has zero callers anywhere in `src/` or
`scripts/`. `discount_factor()` has exactly one caller — `npv()`
itself (`economics.py:167`) — and no caller outside this file. Since
`npv` is itself never called, `discount_factor` is unreachable from any
live code path even though it is technically "called once" in a purely
syntactic sense. The other four functions in this file (`gas_cost`,
`solar_capex`, `carbon_cost`, `unserved_energy_cost`,
`total_system_cost`) are all live, called from `run_deterministic_
scenario` (already flagged at A8 as a legacy-but-live function) —
confirming this is two specific dead functions in an otherwise-used
file, not a wholesale-dead module.

**Why it matters.** Nothing reported depends on either function — a
simple, generic NPV/discount-factor pair that every LP builder
reimplements independently via its own `m.DF[t]` Pyomo Param rather
than calling this. Named for completeness of the pattern; the risk is
purely that a future author might assume this project's canonical NPV
convention lives here and use it for a new script, when the actual
discounting convention (and its various regime-specific variants —
`m.DF`, `m.remaining_npv_factor`, the F1 tranche CRF machinery) lives
entirely in the two builders instead.

**Severity:** LIVE-LATENT.

**Declared?** No.

**No other findings in this file.**

---

## `src/blended_finance.py` — read in full

This file is the machinery register **L17** already extensively
covers (`resolve_private_rate()`, the pre-F1 blended-rate/wrong-horizon
defect reaching `m.eaas_bankability`/`m.eaas_subsidy_limit`) and its
"known second instance" (`run_fin2_blended_bankability_sweep`,
`optimize_experiments.py`, dead runner `04_run_fin2_tariff_sweep.py`
— **correction, task_audit_fixes_p3 (A51): no longer a dead runner.**
`04_` was repaired weeks after this Phase 1 text was written (`b496d77`
built it out, `ecb7e54` repaired its CLI-arm surface) and carries no
suspension guard today — confirmed directly. This sentence's use of it
was purely as a caller-count citation for `run_fin2_blended_bankability_
sweep`, which is unaffected either way, so nothing else in this
paragraph or in Phase 1's findings changes).
Confirmed `resolve_private_rate()` is called by both LP builders
(`optimize_model.py:345`, `optimize_model_sliced.py:441`) exactly as
L17 describes, and that `blend_case`'s silent default
(`"blended_central"`, `scenarios.py:2046`) and its being unoverridden
by any runner are already stated explicitly in L17's own text. Not
re-reported.

### A10 — `implied_concessional_share` has zero callers

**Fingerprint:** F1.

**Location:** `src/blended_finance.py:116-132`.

**What is wrong.** `grep -rn "implied_concessional_share" src/
scripts/` returns only its own definition. A correctly-implemented
inverse of `blended_wacc()` — its own docstring describes it as a
tool to "STRESS-TEST a quoted blended figure" — with no caller
anywhere. `blended_wacc` itself (the function this one inverts) IS
called (`optimize_experiments.py:1075`, inside the L17-covered second
instance), confirming this is one specific unused function in an
otherwise-live file, not a dead module.

**Why it matters.** Nothing reported depends on it.

**Severity:** LIVE-LATENT.

**Declared?** No.

**No other findings in this file** — `resolve_private_rate()`'s own
`.get("private_discount_rate", 0.18)` fallback branch is, per the trace
above, never reached for any `load_scenario()`-built scenario (the
three blended-finance keys are always present since `blend_case` — not
one of the required, no-default arms — always resolves to a value),
same shape as A8's gas-efficiency fallback but lower risk: `0.18`
matches `r_commercial`'s own typical sourced value rather than a
retired/wrong figure, and the whole surrounding mechanism is already
the subject of L17's much larger, already-declared finding. Not
reported as a separate item to avoid diluting L17's existing coverage
with a minor variant of the same shape.

---

## `src/stochastic.py` — read in full

### A11 — A third, entirely dead Monte Carlo module carries the same independent-sampling defect the project explicitly flags elsewhere

**Fingerprint:** F1 (whole-module dead code) combined with a
substantive design issue that would matter if it were ever revived —
reported as both.

**Location:** `src/stochastic.py:52-111` (`sample_uncertainties`),
`:118-191` (`run_stochastic_simulation`), `:198-227`
(`compute_risk_metrics`) — every function this file defines.

**What is wrong (F1 half).** `grep -rn "run_stochastic_simulation\|
sample_uncertainties\|compute_risk_metrics" src/ scripts/` returns only
their own definitions in this file. The entire module — a carefully
documented, two-stage "fixed-plan re-dispatch" Monte Carlo mechanism,
with its own risk-metrics helper (VaR/CVaR) — has zero callers anywhere
in this project. This is a THIRD Monte Carlo implementation
attempt distinct from `scripts/deferred/17_run_monte_carlo.py` (dead on
required arguments, register L10, plus its own two declared defects)
and from the new `scripts/29_run_parameter_uncertainty.py` (Sobol
parameter-uncertainty sweep, unrelated method) — this one was never
wired to any runner at all, so it never had the chance to be found
broken.

**What is wrong (substantive half, would matter only if revived).**
`sample_uncertainties` (`:94-101`) draws `demand_growth` (a truncated
normal perturbation) and `gas_scenario` (a discrete draw from
`gas_probability_weights()`) as TWO INDEPENDENT random variables per
sample:
```python
scenario["demand_growth"] = max(0.0, np.random.normal(base_scenario["demand_growth"], demand_sigma))
scenario["gas_scenario"] = np.random.choice(labels, p=weights)
```
This is the SAME defect shape TASK.md's own background attributes to
`scripts/deferred/17_run_monte_carlo.py` — "sampled demand growth and
gas regime independently despite a shared driver" — reproduced here in
a second, entirely separate module. `run_stochastic_simulation` also
calls `build_model` (`:140`, `:170`) — the ANNUAL builder — never
`build_model_sliced`, so a revived version would additionally inherit
register **L5**'s "storage is structurally dominated at annual
resolution... optimal storage is identically zero" for every draw.

**Why it matters.** Nothing reported depends on any function in this
file — confirmed zero callers, so the correlated-sampling defect and
the annual-builder storage issue are both currently inert. Flagged
because this is a second, independent instance of the exact
shared-driver-independent-sampling failure the project already treats
as serious enough to retire a whole script over (L10's dead-runner
list, and this task's own background section), found only because this
audit checked every function's caller count rather than only
`17_run_monte_carlo.py`'s.

**Severity:** LIVE-LATENT (zero current consumers). Would become
LIVE-REPORTED immediately if any future script imported and called
`run_stochastic_simulation` without first fixing the sampling
correlation and switching builders.

**Declared?** No. The register and this task's own background discuss
the identical defect shape for `17_run_monte_carlo.py` specifically;
neither mentions this file.

**No other findings in this file.**

---

## `src/EaaS.py` — read in full

No findings. The entire module is already self-declared dead at the
top of its own docstring ("NOT CALLED BY THE LP (doc-code parity, plan
5.5)"), naming every function and confirming zero external callers,
with a citation to `docs/task_bucketA_parity.md` Part B1 for the
decision on whether to delete it. Re-confirmed the underlying claim
still holds (`grep -rn "from src.EaaS import\|import src.EaaS"`
returns nothing outside the file's own docstring text) — not reported
as new since it is already and thoroughly declared, correctly, in the
file itself.

---

## `src/cli_arms.py` — read in full

No new findings beyond what the module already self-declares in its
own docstring: the "THREE ARMS WITH NO LIVE REGISTRY" section
(`gas_build_case`, `new_gas_deliverability`, `capital_treatment_case`
hardcoded as tuples with no matching `*_scenarios()` function in
`scenarios.py`) is an already-stated, explicit gap, not a new finding.
Traced `add_standard_arms`/`standard_arms_kwargs`'s dest-to-kwarg
bridging for `peak_multiple_case` (the one arm whose flag text and
`dest` differ) end to end — correct, no mismatch. Every function in
this file has confirmed live callers across the runners already
examined in prior tasks this session (`01b_`, `18_`, `28_`, `29_`, and
others).

---

## `src/io.py` — read in full

### A12 — `load_yaml` and `save_results` have zero callers

**Fingerprint:** F1.

**Location:** `src/io.py:21-39` (`load_yaml`), `:42-62` (`save_results`).

**What is wrong.** `grep -rn "load_yaml\|save_results" src/ scripts/`
returns only their own definitions. Every other function in this file
(`load_solar_capex_by_year`, `load_storage_capex_by_year`, `load_om_
cost_by_year`, `compute_om_cost_npv`, `load_econ`) is confirmed live
and heavily used. **[CORRECTED, TASK 53 2.10, 2026-10-19: false for `load_storage_capex_by_year`. It had NO caller in `src/` or `scripts/` from `96b50cf` (2026-04-03) until TASK 53 2.2 reconnected it (via `load_scenario`). The grep recorded here was for `load_yaml`/`save_results` only; "confirmed live and heavily used" for the other functions was asserted without one. `git log -S'load_storage_capex_by_year('` shows the call added in `72cad20` and removed in `96b50cf` and no other commit, so there was no caller at any date this audit could have been written. Register L90.]** YAML configuration in this project is read directly
via `yaml`/pandas in specific loaders (e.g.
`gas_deliverability_cases.yaml` is read elsewhere, not through this
generic helper), and results are persisted via `json.dump`/`pandas.
to_csv` at each runner's own call site, never through this module's
`np.savez`/`np.save` path.

**Why it matters.** Nothing reported depends on either function.

**Severity:** LIVE-LATENT.

**Declared?** No.

### A13 — Three of four CSV row-lookups in `load_econ` raise an opaque pandas error on a bad key, one raises a clear one

**Fingerprint:** F2-adjacent (not a wrong silent default — nothing is
silently substituted — but the same "no visible signature" family in
spirit: a bad input produces an unhelpful failure rather than a
diagnosable one). Reported as other.

**Location:** `src/io.py:361` (`gas_row = gas_df[...].iloc[0]`), `:371`
(`solar_row`), `:383` (`storage_row`), `:399` (`voll_row`) — none of
these four checks whether the filter matched any row before indexing
`.iloc[0]`. Contrast `:427-432` (`g_rows` for `genset_case`), which
DOES check `g_rows.empty` first and raises a clear `ValueError` naming
every available scenario.

**What is wrong.** A mistyped or unsupported `gas_price_case`/
`voll_case` (both are free caller-supplied strings — `load_econ`'s own
signature has no `choices=`-style validation, unlike the scenario-level
arms) filters `gas_df`/`voll_df` to zero rows, and `.iloc[0]` on an
empty frame raises pandas's generic `IndexError: single positional
indexer is out-of-bounds` — reproduced directly while auditing this
file (`load_econ("not_a_real_voll_case", genset_case="genset_central")`
in a prior task's read-only Python check raised exactly this). The
genset branch three lines later shows the fix already exists in this
same function for one of the four lookups.

**Why it matters.** No currently-published number is wrong because of
this — every live caller passes one of the small set of valid case
strings (`voll_low`/`voll_mid`/`voll_high`, `gas_low`/`gas_mid`/
`gas_high`). This is a failure-QUALITY issue, not a failure-EXISTENCE
one: the run still stops, just with a less diagnosable error than the
one genset lookup three lines below already demonstrates is easy to
provide.

**Severity:** COSMETIC (no wrong number, no silent behaviour — a bad
input still crashes the run, just less informatively than it could).

**Declared?** No.

**No other findings in this file.** `SOLAR_CAPEX_PER_MW`'s own comment
(`:373-376`, "Used ONLY as fallback when solar_capex_by_year is not
passed... All runner scripts must pass solar_capex_by_year") is a claim
about EVERY runner's behaviour — checked against the live callers
already confirmed elsewhere in this audit (`01_`, `01b_`, `02_ndc_caps`,
`03_`, `18_`, `27_`, `28_`, `29_` all pass it); not exhaustively
re-checked against every one of the ~19 files that call either builder
(some are already-dead runners per L10/L27) — flagged as a coverage
gap in the closing section rather than asserted complete here.

---

## `src/scenarios.py`, `src/optimize_model.py`, `src/optimize_model_sliced.py`, `src/optimize_experiments.py`

The four largest files (171K/99K/103K/183K characters). Given
proportionate time per the task's own instruction toward `scenarios.py`
as the highest-value target, and substantial prior direct reading of
`extract_planning_diagnostics`, the conversion mechanism, `_capex_at`,
and the cost-decomposition/capex-reporting machinery across earlier
tasks this session (all re-verified fresh where cited below, not
assumed from memory) — findings below, then a coverage statement in
the closing section for what these four files were and were not fully
read against.

### A15 — `build_model_sliced()` silently drops its own declared backward-compatibility aliases; an already-run, disk-persisted sweep is empirically confirmed dead on half its rows

**Fingerprint:** F8 (a parameter accepted and then discarded) combined
with F9 (a feature present in one builder, silently absent from the
other) — reported as both, since each half of the shape is independently
verifiable.

**Location:** `src/optimize_model_sliced.py:221-234`
(`build_model_sliced`'s signature):
```python
def build_model_sliced(
    scenario,
    econ,
    emissions_cap=None,
    emissions_cap_by_year=None,
    max_offgrid_fraction=None,
    grid_service_mode="annual",
    reliability_max_unserved_fraction=None,  # DEPRECATED alias for max_offgrid_fraction
    reliability_mode=None,                    # DEPRECATED alias for grid_service_mode
    selfgen_in_emissions_cap=False,
    fixed_plan=None,
    solar_capex_by_year=None,
    allow_unserved=None,
):
```
Found via an AST scan of every function in `src/` for parameters never
referenced in their own body (side-effect-free, no model built) —
`reliability_max_unserved_fraction` and `reliability_mode` are the only
two parameters in this entire file's function set that are accepted and
never read again anywhere in the ~1,700-line function body that follows
(`grep -c "reliability_max_unserved_fraction\|reliability_mode"
src/optimize_model_sliced.py` — 2, both on the signature line, zero
elsewhere).

**Confirmed as an asymmetry, not a project-wide convention.** The
ANNUAL builder, `src/optimize_model.py:35-38,55-63`, has the IDENTICAL
two parameters, IDENTICAL names, IDENTICAL "DEPRECATED alias" comments
— and CORRECTLY implements the translation:
```python
if reliability_max_unserved_fraction is not None:
    if max_offgrid_fraction is not None:
        raise ValueError("Pass either max_offgrid_fraction or its deprecated alias reliability_max_unserved_fraction, not both.")
    max_offgrid_fraction = reliability_max_unserved_fraction
if reliability_mode is not None:
    grid_service_mode = reliability_mode
```
The sliced builder has no equivalent block anywhere — the two values
are captured into local parameters and never touched again.

**Confirmed LIVE, not hypothetical — a real caller, and a real,
already-computed, disk-persisted result.**
`scripts/23_run_selfgen_frontier.py:448-489` sweeps `EPS_GRID` across
BOTH builders in one script (`for builder_label, builder, sliced in
(("annual", build_model, False), ("sliced", build_model_sliced,
True)):`), calling, for every `eps` in the grid:
```python
m = builder(
    scenario=sc, econ=econ,
    emissions_cap=None, emissions_cap_by_year=None,
    solar_capex_by_year=capex,
    # TODO rename -> max_offgrid_fraction (separate commit)
    reliability_max_unserved_fraction=eps,
    reliability_mode=RELIABILITY_MODE,
)
```
— note the script's OWN unresolved `# TODO rename` comment, evidence
the author already suspected this needed fixing and never circled
back. `max_offgrid_fraction` itself is never passed by this script at
all, under either builder.

**Verified empirically against the actual, already-persisted result
file**, `results/selfgen_frontier/frontier.csv` (48 rows, this script's
last real run, not reproduced or re-run by this audit — read only):
for the `sliced` builder, EVERY ONE of the 12 swept `eps` values
(0.05 through 0.6) for a given `pricing` arm produces **bit-identical**
`objective_usd` and `shortfall_fraction_realised` — e.g.
`voll`/`sliced`: `objective_usd=1132177162475.6655`,
`shortfall_fraction_realised=0.272833252162856`, unchanged to full
float precision across all 12 rows. The `annual` builder half of the
SAME file, same script, same run, varies exactly as expected: infeasible
below `eps=0.3`, then a genuinely different objective and shortfall
fraction at every subsequent `eps` (`0.2644729`→`0.2566531`,
`objective_usd` falling smoothly from `1.4667e12` to `1.1468e12`) —
proving the sweep mechanism and the annual builder both work correctly,
and isolating the defect to the sliced builder's silently-dropped
aliases specifically.

**Why it matters.** `23_run_selfgen_frontier.py`'s entire stated
purpose is to trace how outcomes change as the self-generation/
reliability threshold varies. For its sliced-builder half — the builder
that produces every OTHER headline result in this project — the
"frontier" is twelve copies of the same solve. Anyone reading `results/
selfgen_frontier/frontier.csv`'s `sliced` rows as a real sensitivity
would be reading noise as signal.

**Severity:** Not confirmed cited in `findings_frozen.md` or `frozen_
state_2026_09_10.md` (`grep -rn "selfgen_frontier\|23_run_selfgen"`
across both: no hits) — so classified **LIVE-LATENT** by the task's
strict definition (nothing in the FROZEN FINDINGS currently cites it).
Flagged as the most severe finding in this audit regardless: this is
not a hypothetical future breakage but an ALREADY-RUN, disk-persisted
result file, sitting in `results/`, whose sliced half is confirmed
empirically dead — the gap between "not yet cited in a chapter" and
"safe" is only whether someone has opened this specific CSV since it
was produced.

**Declared?** No. Not in the register, not in the DO-NOT-REPORT table
(which does not mention this script or file), not in either builder's
own module docstring beyond the two inline "DEPRECATED alias" comments
that describe the annual builder's own working behaviour and, by
sitting unchanged in the sliced builder's signature, imply — falsely
— the same behaviour there.

### A16 — `slice_detail`'s `years` parameter is unused (minor)

**Fingerprint:** F8, minor instance.

**Location:** `src/optimize_model_sliced.py:2063-2078`.

**What is wrong.** `slice_detail(m, years, t_index)` never references
`years` in its body — only `t_index` indexes into `m`. Unlike A15,
nothing is silently mis-behaving: the function's only two callers
(`01b_run_baseline_sliced.py`) already have `years` in scope
independently and use it themselves to label output outside this
call — the parameter is vestigial, not misleading.

**Severity:** COSMETIC.

**Declared?** No, but inconsequential.

### A17 — Six top-level library functions in `optimize_experiments.py` have zero callers anywhere

**Fingerprint:** F1.

**Location and method.** An AST scan of every top-level function
definition in the four largest `src/` files (36/4/9/33 functions in
`scenarios.py`/`optimize_model.py`/`optimize_model_sliced.py`/
`optimize_experiments.py` respectively), cross-checked against
`grep -rc` caller counts across `src/`+`scripts/` (side-effect-free,
no model built). `scenarios.py`, `optimize_model.py`, and `optimize_
model_sliced.py` show no top-level function with zero external
references. `optimize_experiments.py` has six:

- `run_annual_cap_case` (`:168-...`)
- `identify_bottleneck` (`:716-...`)
- `run_bottleneck_sensitivity` (`:740-...`)
- `run_str1_storage_parameter_sweep` (`:1929-...`)
- `run_all_deterministic_scenarios` (`:3846-...`) — self-labelled in
  its own docstring, *"LEGACY batch wrapper around run_deterministic_
  scenario (non-LP simulator)"*
- `run_financing_vs_resource_test` (`:3984-...`)

Each confirmed individually: `grep -n "<name>"
src/optimize_experiments.py` returns only the `def` line for all six —
no internal self-reference, no external caller in any script.

**One of the six is consistent with an already-declared suspension,
not new.** `run_str1_storage_parameter_sweep` is the library function
behind STR-1; register **L5** already states `14_run_str1_storage_
role.py` "is suspended for exactly this reason" (storage structurally
dominated at annual resolution). This function's dead status is the
expected consequence of that already-declared suspension, not an
independent finding.

**The other five are not declared anywhere.** `run_annual_cap_case`,
`identify_bottleneck`, `run_bottleneck_sensitivity`, `run_all_
deterministic_scenarios`, and `run_financing_vs_resource_test` read as
an early exploratory layer (annual NDC-cap case running, shadow-price
bottleneck identification, gas-scenario bottleneck sweeps, batch
deterministic-scenario running, financing-vs-resource-availability
testing) predating the current runner-script architecture, with no
register or findings mention.

**Why it matters.** Nothing reported depends on any of the six —
confirmed zero callers. Named together because six dead top-level
functions concentrated in one file (out of 33 total) is a meaningfully
higher dead-code density than the other three large files (zero each),
worth a reader's attention as a place old exploratory code was never
pruned when the current pipeline (`extract_planning_diagnostics` plus
the runner scripts) replaced it.

**Severity:** LIVE-LATENT (all six).

**Declared?** Partially — one of six (`run_str1_storage_parameter_
sweep`) is the expected consequence of L5's already-declared STR-1
suspension. The other five: no.

### A14 — Two structurally different "peak demand" quantities feed a diagnostic and a constraint, and the diagnostic never varies with the arm that changes the constraint

**Fingerprint:** F3 (name misdescribes the quantity) with an F4 flavor
(two representations of "peak demand" a reader would assume move
together, and do not).

**Location:** the constraint, `src/optimize_model_sliced.py:1198-1199,
1234-1235` (mirrored in `optimize_model.py`), reads `peak_mw[t]` —
REQUIRED (`optimize_model_sliced.py:273-279`, raises if absent), and
itself `avg_mw[t] * par_block`, where `par_block` is DERIVED from the
`peak_multiple_case` arm's `season_demand_factor`/`period_demand_
factor` (`scenarios.py:3160-3164`, four arms:
`peak_suppressed`/`peak_unsuppressed`/`peak_nirp`/`peak_high`). The
diagnostic, `src/optimize_experiments.py:2704-2721`
(`extract_planning_diagnostics`'s `peak_adequacy_by_year`), computes an
UNRELATED peak figure:
```python
peak_demand_multiple = scenario.get("peak_demand_multiple", 1.82)
...
avg_mw = avg_demand_twh * 1e6 / 8760
peak_mw = avg_mw * peak_demand_multiple
```
using `scenario["peak_demand_multiple"]`, an UNCONDITIONAL, arm-
independent scalar set to a flat **2.5** in every scenario
(`scenarios.py:3157`, outside any `peak_multiple_case` branch) — never
derived from `season_demand_factor`/`period_demand_factor`, never
varying with which `peak_multiple_case` arm actually built the model.

**Already self-documented in code, not in the register.** `scenarios.
py:3150-3156` carries an explicit comment: *"this is a PRE-EXISTING,
DIFFERENT quantity from peak_multiple_case/peak_mw below. It feeds
only the post-solve heuristic in optimize_experiments.py
(peak_adequacy_by_year), not any LP constraint... flagged here so the
two are not confused."* So the divergence is a known, intentional
design choice, not an oversight the author was unaware of. It does not
appear in the register, `findings_frozen.md`, or `frozen_state_2026_
09_10.md` (`grep -rn "peak_adequacy\|peak_demand_multiple"` across all
three: no hits) — a limitation the code comments concede but that has
no declared direction of bias anywhere a reader of the FINDINGS (rather
than the source) would see it.

**Why it matters.** `peak_adequacy_by_year`'s own field names —
`peak_demand_mw`, `firm_capacity_mw`, `reserve_margin` — read as a
direct report on whether the model's built capacity is adequate
against ITS OWN peak-demand assumption. It is not: running the exact
same configuration under `peak_suppressed` versus `peak_high`
(`period_demand_factor` up to 1.20 at the evening block, register L1)
changes `peak_mw[t]` inside the actual firm-capacity constraint
substantially, but `peak_adequacy_by_year`'s own `peak_demand_mw`
figure is IDENTICAL across both runs (same `avg_demand_twh × 2.5`
regardless of arm) — a reader comparing `reserve_margin` across
`peak_multiple_case` arms would see numbers that look like they measure
the thing that changed, when they measure a flat multiple of average
demand that did not.

**Severity:** LIVE-LATENT — `grep -rn "peak_adequacy_by_year"
scripts/` shows zero consumers, so no currently-published number is
wrong because of it. Would become LIVE-REPORTED the moment any script
or finding cites `peak_adequacy_by_year`/`reserve_margin` across more
than one `peak_multiple_case` arm, since the two are then comparing a
genuinely arm-sensitive built-capacity figure against an arm-blind
peak-demand figure without saying so.

**Declared?** Partially — self-documented in a source comment
(`scenarios.py:3150-3156`) as an intentional, known distinction, but
NOT declared in the register/findings with a stated direction of bias,
which is what this task's own definition of "declared" requires. The
gap is specifically the ABSENCE of a register/findings entry, not the
underlying design choice, which the code itself defends as deliberate.

---

## `src/output_naming.py` — read in full

### A2 — Hash8 output naming has no collision detection

**Fingerprint:** F9 (unreachable/dead branches) is the closest listed
fingerprint but not an exact match — this is closer to "a silent
failure mode with no guard," reported as other.

**Location:** `src/output_naming.py:28-36` (`hash8_of`), `:54-71`
(`append_index`).

**What is wrong.** `hash8_of` truncates a SHA-1 digest to 8 hex
characters (32 bits of hash space). `append_index` deduplicates only
on the EXACT `(hash8, arm_tag)` line — it checks whether the identical
line already exists, never whether `hash8` already maps to a
DIFFERENT `arm_tag`. If two distinct arm tags ever hash to the same
8 characters, nothing in this module detects it: `append_index` simply
appends a second, different line for the same `hash8`, and every
caller that names an output file `f"diagnostics__{hash8}.json"` (`01b_
run_baseline_sliced.py`, `02_run_ndc_caps.py`, `03_run_ndc_eaas.py`,
`26_run_f3_identification.py`) will have the second run's file
silently overwrite the first's, with no error, warning, or visible
signature — the exact "no error or visible signature" phrase this
project's own register uses for its most serious defect class (L19,
L21).

**Why it matters.** At this project's current usage volume (a bounded
set of named arm combinations per sweep script, at most low hundreds
of distinct configurations across the whole project to date), the
birthday-bound collision probability is small (order 1e-4 to 1e-5) but
not zero, and grows quadratically with the number of distinct
configurations any one hash8-naming caller produces. Nothing currently
reported is known to be wrong because of this — no collision is known
to have occurred — but there is no code path that would ever reveal one
if it did.

**Severity:** TRAP — correct today (no known collision), but positioned
so a plausible future change (any script that sweeps a much finer grid
of distinct arm tags through this same naming scheme — the sort of
large-N sweep this project has already built, e.g. `scripts/29_run_
parameter_uncertainty.py`'s Sobol draws, though that particular script
was written to log draws into one JSONL file rather than one file per
hash8 and so does not itself use this module) would silently overwrite
one run's output with another's, with nothing in the code to catch it.

**Declared?** No. Not in the register, not in the DO-NOT-REPORT list.

**Read in full. No other findings in this file.**

---

## CLOSING SECTIONS

A note on ordering before the table: findings are numbered in the
order they were WRITTEN, which does not exactly match the order the
files were finally organised into in this document (`A14`, `A15`,
`A16`, `A17` were found and written while auditing the four large
files, then inserted ahead of where `A2` — found earlier, in
`output_naming.py` — happens to sit in the final file layout). The
table below is the authoritative sequential index; the ID number, not
its position in the document body, is what identifies a finding.

### Findings table

| ID | Fingerprint | File | Severity | Declared? |
|---|---|---|---|---|
| A1 | F1 | `src/utils.py` | LIVE-LATENT | No |
| A2 | other (no collision detection) | `src/output_naming.py` | TRAP | No |
| A3 | F1 | `src/postprocess.py` | LIVE-LATENT | No |
| A4 | F1 | `src/gas_balance.py` | LIVE-LATENT | No |
| A5 | F7 (docstring vs. own file) | `src/solar.py` | COSMETIC | No |
| A6 | F1 | `src/solar.py` | LIVE-LATENT | No |
| A7 | F1 | `src/demand.py` | LIVE-LATENT | No |
| A8 | F2 | `src/optimize_experiments.py` (`run_deterministic_scenario`) | TRAP | No |
| A9 | F1 | `src/economics.py` | LIVE-LATENT | No |
| A10 | F1 | `src/blended_finance.py` | LIVE-LATENT | No |
| A11 | F1 + design issue | `src/stochastic.py` | LIVE-LATENT | No |
| A12 | F1 | `src/io.py` | LIVE-LATENT | No |
| A13 | other (opaque error) | `src/io.py` (`load_econ`) | COSMETIC | No |
| A14 | F3 / F4 | `src/scenarios.py` + `src/optimize_experiments.py` (`peak_adequacy_by_year`) | LIVE-LATENT | Partially (code comment, not register) |
| A15 | F8 + F9 | `src/optimize_model_sliced.py` (`build_model_sliced`) | LIVE-LATENT (see note) | No |
| A16 | F8, minor | `src/optimize_model_sliced.py` (`slice_detail`) | COSMETIC | No |
| A17 | F1 (×6) | `src/optimize_experiments.py` | LIVE-LATENT | Partially (1 of 6 consistent with L5) |

### The LIVE-REPORTED list

**None of the seventeen findings above meets the strict bar of
"affects a number in `findings_frozen.md` or `frozen_state_2026_
09_10.md` today"** — checked explicitly for every finding, not
assumed. Stated first and plainly because a null result here is itself
a claim that needs to be checkable, not a default.

**The two findings closest to that line, named for priority attention
regardless of the strict classification:**

- **A15** (`build_model_sliced()` silently drops its reliability-alias
  parameters): not cited in a frozen finding, but empirically confirmed
  against an ALREADY-COMPUTED, disk-persisted result file (`results/
  selfgen_frontier/frontier.csv`) that exists right now — the sliced
  half of that file's entire `eps` sweep is bit-identical across all
  12 swept values. If any future document cites that file, or if
  `23_run_selfgen_frontier.py` is re-run and its sliced results used
  anywhere, this becomes LIVE-REPORTED immediately.
- **A8** (`run_deterministic_scenario`'s `gas_eta` fallback to the
  RETIRED 0.43): inert today only because every live call path supplies
  `gas_eta` via `load_scenario()`; the function's own documented purpose
  (a hand-suppliable-capacity-path sketch) is exactly the use case that
  would trigger it.

### Coverage statement

**Read in full, this task:** `src/utils.py`, `src/output_naming.py`,
`src/postprocess.py`, `src/gas_supply.py`, `src/gas_balance.py`,
`src/solar.py`, `src/demand.py`, `src/dispatch.py`, `src/storage.py`,
`src/economics.py`, `src/blended_finance.py`, `src/stochastic.py`,
`src/EaaS.py`, `src/io.py` — fourteen files, start to end.

**Read in full in an earlier task this same session, re-verified
(not merely recalled) against the live file where cited above:**
`src/cli_arms.py`.

**NOT read exhaustively line-by-line in this pass — `scenarios.py`
(171K chars, 36 top-level functions), `optimize_model.py` (99K, 4),
`optimize_model_sliced.py` (103K, 9), `optimize_experiments.py` (183K,
33).** These four were covered by a DIFFERENT method, stated honestly
rather than implied to be equivalent to a full read:
- Large, specific sections of all four were read in full, start-to-
  finish, IN EARLIER TASKS this session (not from memory — re-verified
  against the current file, line numbers re-confirmed, in this task)
  — `extract_planning_diagnostics` (its entire ~950-line body),
  `_capex_at()` and the capex-reporting chain, `cost_decomposition`,
  the conversion mechanism (Phase 2.8) end to end in both builders, the
  firm-capacity constraint, the O&M diagnostic-Expression zone, and the
  full `load_scenario()` machinery for gas availability, O&M,
  conversion, and the tranche/blended-finance keys.
- Every one of the 9 fingerprint patterns was searched for ACROSS THE
  FULL TEXT of all four files (not sampled) via `grep`/AST — every
  `scenario.get(`/`econ.get(` call with a default, every top-level
  function's caller count, every `DEPRECATED` marker, every unused
  function parameter (AST-verified, not just grepped). This gives
  complete PATTERN coverage for those specific searches without
  constituting a full manual read of either file.
- What this method CANNOT catch: a defect that doesn't match any of
  the 9 fingerprint shapes and sits in a stretch of these four files
  that was neither pattern-matched nor previously read in full — for
  example, `scenarios.py`'s ~36 arm-registry functions were checked for
  caller existence and default-value consistency, but not every
  registry's OWN cited numbers were re-verified against their sources
  a second time in this pass (many were already re-verified across
  earlier tasks this session, e.g. `conversion_scenarios()`,
  `gas_availability_scenarios()`, `om_scenarios()`'s CSV wiring — those
  are not blind spots — but a systematic re-check of all 36 was not
  performed here). Similarly, `optimize_experiments.py`'s six functions
  covering GAS-1/2/3, REL-1/2/3, DEM-1/2 (register L27's "six shared
  analysis functions") were confirmed to exist and were not re-read in
  full in this pass beyond what L27 itself already documents.

**Not reached at all in `src/`:** none — every file listed in the
task's own file list, plus `cli_arms.py`, `output_naming.py`, and
`utils.py` (present in `src/` but not in that list), was at least
grep/AST-swept for all nine fingerprints, even where not manually read
line by line.

**Read as evidence only (Phase 2 territory, not audited for findings
of their own), per the task's explicit permission:** `docs/limitations_
register.md`, `docs/findings_frozen.md`, `docs/frozen_state_2026_
09_10.md`, `docs/merged_correction_plan_v2.md` (read in full, as
instructed, to establish the known set); `scripts/23_run_selfgen_
frontier.py`, `scripts/02_run_eaas.py` (read in part, to confirm two
`src/` findings were live); `results/selfgen_frontier/frontier.csv`,
`results/baseline_sliced/summary__02c17f55.json` (read, to confirm
findings empirically against real, already-computed output);
`docs/methodology_demand_horizon.md`, `docs/methodology_genself_
backstop.md` (grepped, not read in full, for one F7 spot-check that
came back resolved — not exhaustively checked against `optimize_
model.py`/`optimize_model_sliced.py` beyond that one check).
`docs/methodology.md`, `docs/methodology_physical_layer.md`, `docs/
methodology_protocols.md` were **not checked against the code** in
this pass — a genuine gap, stated plainly rather than left implicit,
and the most likely place a Phase-2-adjacent F7 finding is still
sitting unfound.

**Nothing edited. Nothing committed. Nothing run** — every check above
was read-only file inspection, read-only git (not used this task —
no `git log`/`blame` was needed beyond what earlier tasks already
established), or a side-effect-free Python call (an AST parse, or a
`grep`/`csv`/`json` read of an already-existing file). No script,
solver, or model build was executed.

---

## PHASE 2 — scripts, harnesses, and documentation

Continuing the A-numbering from A18. Nothing edited, run, or committed.
A15 and A8 (Phase 1) confirmed fixed and committed (`git log`:
`2d684e9`). Register status re-verified before treating it as the known
set — see the correction immediately below, since it changes what
counts as "already declared" for this phase.

### Correction to this task's "KNOWN SET" claim, checked before relying on it

TASK.md states *"L1 to L33... L29-L33 are applied."* Re-checked
directly (`git show HEAD:docs/limitations_register.md | grep "^## L"`
vs the current working tree):

- **L29, L30, L31 are committed** (commit `215d53f`, despite that
  commit's own message claiming "L29-L33" — it actually added three
  entries, not five: `git show 215d53f --stat` shows 183 lines
  inserted, matching L29-L31's combined length, not five entries'
  worth).
- **L32 exists only in the current UNCOMMITTED working tree**
  (`git diff --stat docs/limitations_register.md` shows 63 uncommitted
  insertion lines, exactly L32's text) — not yet part of any commit.
- **There is no L33, anywhere, committed or not.** The fifth draft from
  the prior task (4E, a DO-NOT-REPORT addition for `results/selfgen_
  frontier/frontier.csv`) was never a register entry to begin with —
  it targeted `docs/merged_correction_plan_v2.md`'s DO-NOT-REPORT
  table, a different document. Checked directly:
  `grep -n "selfgen_frontier\|frontier.csv" docs/merged_correction_
  plan_v2.md` returns nothing. **That draft was never applied
  anywhere.**

**Consequence for this phase**: `results/selfgen_frontier/frontier.csv`
is NOT currently barred in any document a screening pass would consult
— its sliced rows remain void (per Phase 1's A15) and undeclared as
such in `merged_correction_plan_v2.md`. This is directly relevant to
this phase's own F10 fingerprint, which names this exact file. Treated
below as still-open, not re-derived as a new finding (it is Phase 1's
A15 with an outstanding paperwork gap, not a new defect) — flagged
here so it is not silently treated as closed because a later task's
description says so.

**The rest of the known set was re-confirmed accurate**: `docs/
limitations_register.md` L1-L28 (Phase 1's baseline) plus L29-L31
(committed) and L32 (present, if not yet committed) all read as
described; `docs/forensic_audit.md` A1-A17 and their fixes (A15, A8)
are correctly described; `findings_frozen.md`'s scoped conditions and
`merged_correction_plan_v2.md`'s DO-NOT-REPORT table are unchanged from
Phase 1's reading except for the M-4/methodology-protocols additions
from the most recent Monte Carlo commit (`c8bcdfa`), read fresh below
where relevant.

---

### Audit order, stated per the task's own prioritisation

1. Tier 1 — cited in `findings_frozen.md`: `00_`, `01b_`,
   `02_run_ndc_caps`, `03_`, `26_`, `27_`, `29_`.
2. Tier 2 — produce results, not yet cited: `18_`, `28_`, `23_`, `22_`,
   `19_`, and the rest of the numbered runners.
3. Tier 3 — deferred/retired: brief.
4. Shell harnesses.
5. Documentation vs. code, all seven named documents.

---

## TIER 1 — cited in `findings_frozen.md`

### `scripts/00_build_emissions_cap.py`

### A18 — The rename that fixed A-series conversion-blindness now silently breaks this script's own bare-path input

**Fingerprint:** F10 (output/input path nothing correctly maintains),
overlapping F2 in spirit (a rename's consequence that fires as a hard
crash rather than a silent wrong value — the opposite failure
direction, but the same root cause: no code enforces the bare file and
the reading code stay in sync).

**Location:** `scripts/00_build_emissions_cap.py:131`, `gas_to_power =
diag["fleet_gas_to_power_twh_th_by_year"]` — reads the RENAMED key
(`task_conversion_diagnostics`, Part 2C), against `results/baseline/
diagnostics.json`, an untracked, unregenerated file on disk (mtime
2026-08-23 13:44, unchanged since before that rename).

**What is wrong, verified by direct read, not inference:**
```python
import json
d = json.load(open('results/baseline/diagnostics.json'))
'fleet_gas_to_power_twh_th_by_year' in d   # False
'gas_to_power_twh_th_by_year' in d         # True
```
The bare file still carries the PRE-rename key. Running `00_
build_emissions_cap.py` right now would raise `KeyError:
'fleet_gas_to_power_twh_th_by_year'` at line 131, immediately.

**Why it matters.** `00_` produces `data/cost/processed/emissions_
cap.csv`, which `02_run_ndc_caps.py` consumes for every NDC-capped
result in `findings_frozen.md` (F-8, F-9, F-10, F-21, and the NDC
comparison in F-15/F-21's own text). The CURRENTLY-COMMITTED `emissions_
cap.csv` (git `9700f3f`, unaffected — it was generated before the
rename, from the old key name, and is not regenerated by reading this
script's source) remains numerically valid, per `task_conversion_
diagnostics.md`'s own established argument (the input values are
unchanged; only the key name changed). But the SCRIPT that produces it
is now broken for any future re-run, with no warning anywhere a reader
of the FINDINGS would see before attempting one.

**Already anticipated, not registered.** This exact consequence was
predicted in `docs/task_conversion_diagnostics.md` at the moment the
rename was made ("Running `00_` against the current bare file will now
raise `KeyError`, not silently succeed... flagged so it isn't a
surprise the next time someone runs `00_`"). It does not appear in
`docs/limitations_register.md` (L1-L32), was not added to `docs/
merged_correction_plan_v2.md`'s DO-NOT-REPORT table, and is not stated
in `00_`'s own docstring (which documents the annual/`conv_none`
requirement and the "stale in provenance, not in value" argument, but
was written before this specific rename and does not mention it).

**Severity:** TRAP — correct today (the currently-published
`emissions_cap.csv` is unaffected, per the already-established
provenance argument), but the SCRIPT will crash the moment anyone
tries to refresh it, and nothing announces that before they try.

**Declared?** Partially — anticipated in a task report
(`task_conversion_diagnostics.md`), not in the register or DO-NOT-
REPORT table, which is what this task's "known set" actually checks
against. Recorded here as the gap between "someone wrote it down once"
and "it is where a future maintainer would look."

**No other findings in this file.** Re-read `00_`'s docstring
(including this session's own added "REQUIRED INPUT" section) against
the current `data/gas/processed/gas_available_power_annual_twh_th.csv`
and `results/baseline/diagnostics.json` mtimes — both unchanged since
that section was written, so its claims (the `986f467`/`74d34f2`/
`5b811f2` commit trail, the Aug-23 bare-file mtime) remain accurate.

### `scripts/01b_run_baseline_sliced.py`

### A19 — The observed-2024 backcast anchor (25.62 TWh_e) is hardcoded independently at four sites

**Fingerprint:** F5 (duplicate source of truth) — the exact shape
Phase 1's own worked example names (`storage_duration_hours` at four
sites); this is a second instance of the same count, a different
constant.

**Location:** `scripts/01b_run_baseline_sliced.py:471,473` (the printed
backcast check, `f"observed 25.62"` / `(gas_2025 - 25.62) / 25.62`),
`scripts/99_validate_baseline.py:52` (`"gas_generation_twh": 25.62,` in
its `OBSERVED_2024` dict), `scripts/27_run_hydro_minflow_
sensitivity.py:163` (`OBSERVED_GAS_2025_TWH_E = 25.62  # same backcast
reference 01b prints`), and `src/scenarios.py:30` (the original
citation, in a module docstring: `"of which gas 25.62 TWh"`).

**What is wrong.** Four independent literals, in four files, each
typed separately rather than imported from one canonical constant.
`27_`'s own comment even names `01b_` as "the same backcast reference"
— acknowledging the duplication explicitly while still re-typing the
literal rather than importing it. All four currently agree (checked:
`grep -rn "25\.62" scripts/*.py src/*.py` — every hit is exactly
`25.62`, no drift today), so this is not presently wrong, only
unenforced.

**Why it matters.** If NERC's 2024 gas generation figure is ever
revised (the same kind of update that already happened once — this
figure itself replaced the earlier, retired 23.08 TWh demand base's
gas component per `src/scenarios.py`'s own header history), nothing
would force all four sites to move together. `27_`'s own comment shows
the risk is already understood locally, one file at a time, without a
shared constant closing it.

**Severity:** TRAP — correct today, positioned to silently diverge the
next time this specific empirical anchor is revised, with three of the
four sites (`01b_`, `99_`, `27_`) having no cross-reference to each
other beyond a comment in one of them.

**Declared?** No — not in the register, not in Phase 1 (which was
scoped to `src/` only; `src/scenarios.py`'s own citation is the fourth
site, but the other three are all in `scripts/`, outside Phase 1's
scope).

**No other findings in this file.** Re-read in full against the
current output-field names (`npv_gross_capex_report_window_usd` /
`npv_gross_capex_full_horizon_usd`, added two tasks ago) and the
gas-key/reliability-alias fixes from the immediately preceding
task — `01b_` calls `extract_planning_diagnostics(...)` correctly
(confirmed again: does not pass either reliability alias to `build_
model_sliced`, matching Part 3's own verification in the prior task)
and does not exercise `run_deterministic_scenario` (A8's location) at
all. `F11` does not apply: `01b_` uses `build_model_sliced`, the
builder that CAN represent the full central configuration including
conversion.

### `scripts/02_run_ndc_caps.py` and `scripts/03_run_ndc_eaas.py`

### A20 — Two Tier-1 runners hand-copy arm-choice tuples that `cli_arms.py` exists specifically to unify, instead of using it

**Fingerprint:** F5 (duplicate source of truth), the same shape
register **L27** already names for a DIFFERENT part of this codebase
(six shared library functions' stale `load_scenario()` calls) — this
is a sibling instance in two runner scripts' own CLI layers, not
covered by L27's text.

**Location:** `scripts/02_run_ndc_caps.py:36-68` and `scripts/03_run_
ndc_eaas.py` (identical block, both files define twelve `*_CHOICES`
tuples each). Of the twelve, seven are hand-typed literals with no
connection to the live registry: `PEAK_CASE_CHOICES`, `STORAGE_CC_
CASE_CHOICES`, `GAS_AVAILABILITY_CASE_CHOICES`, `GAS_RETIREMENT_CASE_
CHOICES`, `GAS_CAPEX_CASE_CHOICES`, `CCGT_EFFICIENCY_CASE_CHOICES`,
`CAPITAL_RECOVERY_CASE_CHOICES`, `DSCR_CASE_CHOICES`,
`CAPITAL_TREATMENT_CASE_CHOICES`, `CONCESSIONAL_ENVELOPE_CASE_
CHOICES` (ten, not seven — recounted directly from the file) — e.g.
```python
PEAK_CASE_CHOICES = ("peak_suppressed", "peak_unsuppressed", "peak_nirp", "peak_high")
```
versus the other two/three (`GAS_DELIVERABILITY_CASE_CHOICES`,
`CONVERSION_CASE_CHOICES`, `HYDRO_CASE_CHOICES`, `TARIFF_CASE_CHOICES`,
`OM_CASE_CHOICES`), which ARE pulled live, e.g.
`tuple(gas_deliverability_scenarios().keys())` — the file's own
comments even explain WHY for these: `gas_deliverability_case` was
"hardcoded... which stranded the GMP 2026 policy-target arms," fixed
by making it live. The other ten were never given the same treatment.

**`src/cli_arms.py` exists to solve exactly this**, per its own
docstring: *"This module is the fix for the COPYING, not the arms
themselves... one definition of the standard arm flags, imported by
every runner that needs them, so the next required-key addition to
`load_scenario()` is a one-file change... rather than an N-file
hand-edit that can silently miss a caller."* Its own text names six
callers that already use it (`05_, 15_, 18_, 19_, 20_, 28_`) — `02_run_
ndc_caps.py` and `03_run_ndc_eaas.py` are not among them; both
maintain fully independent, hand-typed `*_CHOICES` tuples instead.

**Currently consistent, checked directly, not assumed.** Compared all
nine of the hardcoded tuples against their live registries
(`peak_multiple_scenarios()`, `storage_cc_scenarios()`, `gas_
availability_scenarios()`, `gas_retirement_scenarios()`, `gas_capex_
scenarios()`, `ccgt_efficiency_scenarios()`, `capital_recovery_
scenarios()`, `dscr_scenarios()`, `concessional_envelope_scenarios()`)
via a side-effect-free Python import — every one matches exactly today.

**Why it matters.** These are two of the seven Tier-1 scripts (cited
directly in `findings_frozen.md` for every NDC-capped result: F-8,
F-9, F-10, F-21). If any registry in `scenarios.py` ever adds, removes,
or renames a choice — exactly the kind of change register L27 and
`cli_arms.py`'s own history (the GMP-arm stranding this file's own
comment describes) show has already happened more than once in this
project — these two files would not notice: a removed choice stays
silently acceptable at the CLI layer until `load_scenario()` itself
rejects it deeper in, and a renamed choice could leave a stale option
listed as valid here while the registry no longer recognises it.

**Severity:** TRAP — correct today (verified), not self-updating.

**Declared?** No. Register L27 documents the identical failure SHAPE
for library functions in `optimize_experiments.py`; it does not cover
these two runners' own CLI argument definitions, and `cli_arms.py`'s
own docstring lists its adopters without flagging these two as
holdouts.

**Scope note, not separately findings-numbered.** The same
`*_CASE_CHOICES = ("...", ...)` hand-typed pattern appears in TEN
scripts total (`grep -l '_CASE_CHOICES = ("' scripts/*.py`):
`01_run_baseline.py`, `01b_run_baseline_sliced.py`, `02_run_eaas.py`,
`02_run_ndc_caps.py`, `03_run_ndc_eaas.py`, `04_run_fin2_tariff_
sweep.py`, `22_salvage_sensitivity.py`, `23_run_selfgen_frontier.py`,
`24_run_leakage_arm.py`, `25_run_obligation_trajectory.py`,
`26_run_f3_identification.py`, `99c_run_genset_arm_sensitivity.py` —
including `01b_`, which `cli_arms.py`'s own docstring names as ITS
reference implementation, not a migration target. This is not
necessarily a defect in every one of the twelve (`cli_arms.py`'s
`add_standard_arms`/`standard_arms_kwargs` pattern is aimed at runners
whose ENTIRE argument surface is the standard set; several of these
twelve have bespoke additional flags a full migration may not suit) —
named here as scope context for A20, not as eleven further findings;
only `02_run_ndc_caps.py`/`03_run_ndc_eaas.py` were verified in full
for THIS finding, since they are the two Tier-1, `findings_frozen.md`-
cited scripts in the list.

### A21 — `02_run_ndc_caps.py`/`03_run_ndc_eaas.py` build the model with one solar-CAPEX series and diagnose it with a second, independently-resolved one

**Fingerprint:** F4 (two representations of one quantity that must
move together, kept in sync only by two literals currently agreeing,
not by construction).

**Location:** `scripts/02_run_ndc_caps.py:311-315` builds the model
with `solar_capex_by_year=solar_capex_tv` from `load_solar_capex_
by_year(scenario_name="solar_low", ...)`, then calls `extract_
planning_diagnostics(m, scenario, econ)` (`:331`, same file) with NO
`solar_capex_series=` argument. `extract_planning_diagnostics`'s own
signature defaults that parameter to `None`, in which case it
INTERNALLY reloads its own copy via `load_solar_capex_by_year(
scenario_name=scenario.get("solar_capex_scenario", "solar_low"), ...)`
(`optimize_experiments.py:2529-2536`) — a second, independent read of
the same CSV, used only for the `solar_beats_gas_year` diagnostic
field (the LCOE-crossing year), not for anything the LP itself solved
with.

**Currently consistent.** `load_scenario()` unconditionally sets
`scenario["solar_capex_scenario"] = "solar_low"` (`scenarios.py:2940`)
— the same literal `02_run_ndc_caps.py`/`03_run_ndc_eaas.py` pass
explicitly to build the model. Both paths resolve to the same CSV
scenario today, confirmed by reading both sites, not assumed.

**Why it matters.** Nothing enforces the two staying equal — a future
runner (or a hand-built scenario) that sets `scenario["solar_capex_
scenario"]` to anything other than `"solar_low"`, or that builds the
model with a different `solar_capex_by_year` argument than
`"solar_low"`, would silently get a `solar_beats_gas_year` diagnostic
computed against CAPEX data the model itself never used. `scripts/29_
run_parameter_uncertainty.py` (written after this audit's Phase 1)
avoids exactly this trap by passing `solar_capex_series=solar_capex_tv`
explicitly — the safer pattern already exists in this project, just
not here.

**Severity:** TRAP — correct today, unenforced.

**Declared?** No.

### `scripts/26_run_f3_identification.py` and `scripts/27_run_hydro_minflow_sensitivity.py`

### A22 — Two more Tier-1 sliced runners still lack the financing-headline field `01b_` now has

**Fingerprint:** F1 (a computed thing — here, a field this project
already fixed one runner to expose — absent from two siblings) /
overlaps F10 in spirit (the reporting rule's field, absent from these
two outputs).

**Location:** neither `scripts/26_run_f3_identification.py` nor
`scripts/27_run_hydro_minflow_sensitivity.py` calls `extract_planning_
diagnostics` at all (`grep -n "extract_planning_diagnostics("` on both:
no hits) — both build their own summary dicts inline from `build_
model_sliced()`'s output directly, the same pattern `01b_run_baseline_
sliced.py` used BEFORE `task_capex_reporting_field` added its two-line
fix. Neither has been given the equivalent fix: `grep -n "npv_gross_
capex"` on both returns nothing.

**Why it matters.** `27_` is Tier-1 — its measured band (register
**L20**: objective 21.337-21.367 bn across the `hydro_min_flow_
fraction` sweep) is a currently-published figure with no reported
capital-mobilisation companion. `26_` underpins F3's identification
ceiling (register L17's "F3's identification ceiling (47,528 MW) is
set by [the bankability] factor"). Neither script can currently answer
"how much capital does this configuration mobilise" the way `01b_` now
can — the reporting rule (`frozen_state_2026_09_10.md` §8) applies to
every sliced run this project treats as reportable, not only the
baseline.

**Severity:** LIVE-LATENT — no number currently published from either
script is WRONG for lacking this field (L20's own table reports
objective and storage only, not capex), but the gap is real and, unlike
`01b_`, unclosed.

**Declared?** No — not in the register, not remediated by the prior
task (`task_capex_reporting_field`'s whitelist was `01b_run_baseline_
sliced.py` only; this was flagged as encountered-but-out-of-scope in
that task's own report and remains so here).

### `scripts/29_run_parameter_uncertainty.py`

No new findings. Written this session with its own extensive self-
documentation (module docstring covers the parameter-vs-scenario-
uncertainty distinction, the 748-vs-600 verdict, and every design
choice). Re-checked against this phase's specific fingerprints: uses
`build_model_sliced` (F11 — can represent the full central
configuration); writes fresh, run-tagged output paths
(`draws_n{N}_seed{S}.jsonl`, `summary_n{N}_seed{S}.json`), no bare-path
collision risk (F10); explicitly passes `solar_capex_series=
solar_capex_tv` to `extract_planning_diagnostics` (avoiding A21's
pattern, confirmed by re-reading the call site). **Both runs now exist
on disk** (`results/parameter_uncertainty/`: `n16` pilot and `n256`
full, both `seed42`) — checked as evidence, not merely inferred: the
full run shows `n_total=1792, n_failed=0`, matching this project's own
commit history (`caf973e`/subsequent commit message: "1,792 solves, 0
failures"), and `gas_availability_factor`'s first-order index (~0.90 on
both QoIs) dominates as this task's own docstring predicted it would,
given its prior's width — no anomaly found in the real output.

---

## TIER 2 — produce results, not yet cited

### A23 — Register L10 is stale: five of its nine named "dead" runners have been repaired and were never marked resolved

**Fingerprint:** other — a DECLARED issue whose stated scope is now
wrong (narrower than claimed, i.e. better news than the register
states, but still a documentation-vs-reality gap the task explicitly
asks to be flagged as its own finding).

**Location:** `docs/limitations_register.md` L10 ("Ten runners are
dead code predating F1"), listing `05, 06, 07, 09b (both), 15, 17, 18,
19, 20` as calling `load_scenario()` "without any of the eight
required keys... None imports argparse. Each raises `TypeError` on the
first missing argument."

**What is wrong.** Checked every one of L10's nine named scripts
directly for `import argparse` and use of `src.cli_arms`'s
`add_standard_arms`/`standard_arms_kwargs` (the shared mechanism that
threads every required `load_scenario()` key, built specifically to
fix this defect class — Phase 1's audit of `cli_arms.py` already noted
its own docstring names "05_, 15_, 18_, 19_, 20_, 28_" as its "six
full-set callers," a claim not cross-checked against L10 at the time).
**Five of L10's nine named scripts have since been repaired**:
```
scripts/05_run_fin3_capital_sweep.py:76,128,157      add_standard_arms + standard_arms_kwargs, both called
scripts/15_run_pol1_ndc_comparison.py:94,191,490     add_standard_arms + standard_arms_kwargs, both called
scripts/18_run_pareto_frontier.py:110,192,276        add_standard_arms + standard_arms_kwargs, both called
scripts/19_run_carbon_price_sweep.py:47,95,101       add_standard_arms + standard_arms_kwargs, both called
scripts/20_run_cf1_solar_yield_sensitivity.py:61,90,96  add_standard_arms + standard_arms_kwargs, both called
```
All five now import `argparse` (via `cli_arms.py`'s own `import
argparse`, re-exported through the shared helper) and thread every
required key `add_standard_arms`/`standard_arms_kwargs` supplies — the
specific failure L10 describes ("None imports argparse... raises
TypeError on the first missing argument") does not hold for these
five. **The other four remain as L10 describes**, checked the same
way: `06_run_gas1_shadow_benchmarks.py` and `07_run_gas2_eaas_gas_
relief.py` both call `load_scenario()` directly with no `argparse`
import at all (`07_`'s own comment self-diagnoses this: *"its load_
scenario() calls omit ~15 arms required since phases..."*) — genuinely
still dead, matching L10 exactly. `09b` (both, in `scripts/retired/`)
and `17_` (`scripts/deferred/17_run_monte_carlo.py`) were not
re-verified in this pass (both already carry additional, separately-
declared defects — L10 itself for 09b, and the deferred script's own
two-defect header for 17_ — so their dead status is unlikely to be in
question regardless of the argparse question specifically).

**Why it matters.** L10's own "Consequence for planned analysis"
section states: *"`17_run_monte_carlo`, `18_run_pareto_frontier` and
`19_run_carbon_price_sweep` are on the planned-outputs list. Each must
be wired before those analyses can run, or the analysis dropped."*
Two of those three (`18_`, `19_`) ARE now wired — this consequence
statement is itself stale. A reader relying on L10 alone would
conclude `18_run_pareto_frontier.py` (the script producing this
project's own epsilon-constraint Pareto frontier, `f4dd48d`'s subject)
is unusable, when it has been running-capable (at least on the
required-arguments dimension checked here) since that commit.

**Severity:** Not a wrong NUMBER (this is a documentation-currency gap,
not a computational defect) — classified as **other/documentation**
rather than LIVE-REPORTED/LIVE-LATENT/TRAP/COSMETIC, since none of
those four categories quite fits "a register entry is stale in the
GOOD direction." Flagged because L10 is exactly the kind of entry a
reader would consult before citing or dismissing a script's output,
and citing it uncorrected would cause someone to wrongly discount five
runners' results as unavailable.

**Declared?** The underlying repair (cli_arms.py, and the commits that
adopted it into these five scripts) is committed and visible in `git
log`; the register entry describing their PRIOR state was never
updated to reflect it. This is the finding: not that the repair is
undeclared, but that the OLD, superseded claim remains the only
register statement about these five scripts' argparse/required-key
status.

### A24 — Three live Tier-2 runners use the annual builder with no stated caveat that it cannot represent conversion (or the central config's other sliced-only mechanics)

**Fingerprint:** F11 (a runner whose builder cannot express the
reporting model).

**Location:** `scripts/24_run_leakage_arm.py:61`, `scripts/25_run_
obligation_trajectory.py:51`, `scripts/99c_run_genset_arm_
sensitivity.py:69` all `from src.optimize_model import build_model,
...` — the ANNUAL builder, never `build_model_sliced`. `src/optimize_
model.py:295-302` raises `ValueError` unconditionally whenever
`scenario.get("conversion_available", False)` — i.e. these three
scripts can only ever run `conv_none`, never the central configuration's
own `conv_central` conversion mechanism, and inherit every other
sliced-vs-annual difference this project already treats as material
elsewhere (storage dynamics, register L5; the six-slice demand/solar
shape).

**Distinguish from `22_salvage_sensitivity.py`**, which also uses
`build_model` — but LEGITIMATELY: salvage credit is structurally
specific to the `pre_f1_overnight` capital-treatment branch (register
L28: "under `f1_tranche` salvage is structurally absent... truncation
at the horizon plays its role instead"), so a salvage-sensitivity
script has no sliced-builder equivalent to run instead. Not a finding
for `22_`.

**Why it matters.** None of the three is currently cited in `findings_
frozen.md` (checked: no hits for any of the three script names) — so
no PUBLISHED number is affected today, consistent with their Tier-2
classification. But nothing in any of the three scripts' own text
states the annual-builder/no-conversion limitation — the only "annual"
references found are to `GRID_SERVICE_MODE = "annual"` (the
reliability-accounting MODE, a `grid_service_mode` argument value,
unrelated to which builder FUNCTION is imported) — a reader could
easily conflate the two and assume these scripts inherit the sliced
builder's full capability because they see "annual" only in that
unrelated context.

**Severity:** LIVE-LATENT.

**Declared?** No.

---

## SHELL HARNESSES

### `scripts/run_central_sweep.sh` — read in full, the highest-stakes file in this phase per the task's own framing

### A25 — The sweep's own header cites a backcast figure the project's DO-NOT-REPORT table already names as superseded

**Fingerprint:** F6 (epoch-stale hardcoded figure).

**Location:** `scripts/run_central_sweep.sh:26`:
```
#   hydro_case                 hydro_flat        (only arm reproducing the
#                                                 2024 backcast at -3.4%)
```

**What is wrong.** `docs/merged_correction_plan_v2.md`'s own DO-NOT-
REPORT table (line 1184) states explicitly: *"Anything quoting the
backcast as −4.1% — STILL BARRED — figure changed again. −4.1% →
−3.4% was correct after the min-build floor was removed, but at the
frozen central configuration the backcast now measures −3.2%...
Neither −4.1% nor −3.4% is current; only −3.2% (central, `2a9226b`)
is."* This project's own highest-stakes harness — the one that
"produced the frozen 38-arm sweep" (TASK.md's own description) — still
carries `-3.4%`, one of the two figures that same project's own
documentation explicitly names as superseded, in its header comment
explaining WHY `hydro_flat` is the central hydro arm.

**Why it matters.** This comment is prose, not an executable
assertion — it does not change what the sweep runs or computes. But it
sits in the one file most likely to be read by anyone trying to
understand or reproduce the central sweep, and states a number the
project's own screening table already flags as wrong to quote. A
reader citing "the backcast reproduces at -3.4%" on this comment's
authority would be citing a figure this project's own tooling would
reject if checked against the DO-NOT-REPORT table.

**Severity:** COSMETIC (a comment, not a computation — nothing the
script DOES depends on this figure) but flagged prominently given the
file's stakes and the task's own instruction to check this file's
"anchor assertions... against the current values."

**Declared?** The correct figure (`-3.2%`) IS declared, in `docs/
frozen_state_2026_09_10.md` §1 and the DO-NOT-REPORT table itself —
this finding is that the sweep script's OWN copy was never updated to
match, not that the correct figure is undeclared anywhere.

**Everything else checked in this file came back clean.** `bash -n`
syntax-checked (`OK`). The regression-anchor invocation
(`A_REGRESSION_ANCHOR`, `conv=conv_none treat=pre_f1_overnight
dscr=dscr_regression tariff=tariff_legacy_65m om=om_none`) uses only
valid arm choices (checked `tariff_legacy_65m` and `dscr_regression`
against their live registries — both present). The two objective-
extraction regexes (`'objective +: +[0-9.]+ bn USD'` for single-case
runners; `'^--- .* saved ---|^saved -> '` for multi-case runners
`02_`/`03_`/`26_`) were checked against what those runners actually
print — `01b_run_baseline_sliced.py:476` prints exactly the format the
first regex expects; `02_run_ndc_caps.py:569`/`03_run_ndc_eaas.py:633`
print `--- {c} saved ---` and `26_run_f3_identification.py:547` prints
`saved -> {path}`, both matched by the fallback regex; `26_` never
prints anything matching the FIRST regex (checked, so it correctly
falls through to the multi-case counting branch rather than
misreporting a single objective). The `om_flag` gating (`26_` does not
receive `--om-case`, since it resolves O&M internally — matches
register L19's description of `26_`'s internal-resolution pattern) is
correctly scoped to the five runners that actually expose the flag.
`$?` is checked immediately after the `python3` invocation with no
intervening command that could clobber it. **F12 does not apply here**
— exit-code handling is correct.

### `scripts/run_verification_sweep.sh` — read in full

### A26 — This project's regression/snapshot harness gives every one of its 19 listed runners zero chance to pass, and cannot tell a healthy refusal from a crash

**Fingerprint:** F12 (a harness that reports its own failure/success
incorrectly) — the SAME defect class `smoke_all.sh`'s own header
documents as already fixed THERE (`"6 pass, 26 fail" ... twelve of
those 'fail' runners were healthy runners correctly refusing to run
without a stated configuration, scored as failures indistinguishably
from a real defect"`), evidently never applied to this file.

**Location:** `scripts/run_verification_sweep.sh:43-57` (the runner
loop) and `:66` (`[ -n "$FAILED" ] && exit 1; exit 0`).

**What is wrong.** The script invokes each of its 19 listed runners
with, at most, `--om-case om_central` — nothing else:
```bash
om_args=""
case "$name" in
  01_run_baseline|01b_run_baseline_sliced|02_run_eaas|02_run_ndc_caps|03_run_ndc_eaas)
    om_args="--om-case om_central" ;;
esac
python "$f" $om_args > "$LOG_DIR/$name.log" 2>&1
code=$?; [ $code -ne 0 ] && FAILED="$FAILED $name"
```
Every runner in the list requires far more than that. Checked directly
(`grep -c "required=True"` on each): `01_run_baseline.py` — 18
required flags; `01b_run_baseline_sliced.py` — 18; `02_run_ndc_caps.py`
— 18; `03_run_ndc_eaas.py` — 18; `02_run_eaas.py` — 15. These are the
"healthy" runners (not on register L10's dead list) — every one of
them will reject this invocation immediately with argparse's own
"error: the following arguments are required" for the 15-18 flags
`om_args` does not supply, before reaching a single line of model
code. The remaining runners in the list (`05_`, `06_`, `07_`, `08_`,
`09_`, `10_`, `11_`, `13_`, `14_`, `15_`, `18_`, `19_`, `20_`, `99c_`)
either have their own required-argument surfaces (several confirmed
via `cli_arms.py` in A23, above) or are already-declared dead (L10,
L27) — either way, none can succeed on `om_args` alone either.

**Consequence, not merely theoretical.** `code -ne 0` is the ONLY test
applied — there is no `smoke_all.sh`-style check for argparse's
specific "error: the following arguments are required" signature. A
run of this script today would report **every one of its 19 runners as
FAILED**, `FAILED` would be non-empty, and (per `:62-64`) the script
would print `"REFUSING snapshot: runners failed"` and exit 1
UNCONDITIONALLY — the harness cannot currently produce a usable
verification snapshot for anything, and a genuine regression in any
one of these 19 scripts would be invisible: it would look exactly like
the other 18 expected argparse rejections in the same run.

**Why it matters.** This is the project's own named REGRESSION
mechanism (`docs/merged_correction_plan_v2.md`'s "VERIFICATION
PROTOCOL THAT WORKED" section names snapshot-based comparison — item 2,
"capture the pre-image on unmodified code" — as a core method this
project relies on for verifying changes don't silently break results).
As currently written, this specific script cannot perform that role
for any of its 19 targets.

**Severity:** Not classified as LIVE-REPORTED (it does not corrupt a
published NUMBER — it simply cannot run) nor cleanly LIVE-LATENT
(nothing "wrong" is silently reported; the tool visibly fails). Closest
to **TRAP**, but stronger: not "positioned to fail on a future change,"
already failing on every invocation today, just never invoked in this
session's history — not found in any `logs/` directory, and not
referenced by any doc checked. Reported plainly as a harness that does
not do what its own name promises, right now.

**Declared?** No — not in the register, not in `merged_correction_
plan_v2.md`'s DO-NOT-REPORT table or DEFERRED DEFECTS (D1-D8), not
referenced anywhere this audit's four known-set documents cover.

### `scripts/verify_2_6_commit_a.sh` — read in full

### A27 — The same required-argument drift also aborts this harness outright, via `set -e`, one step further along than A26

**Fingerprint:** F12, same root cause as A26 (the runner CLI surface
has grown past what older harness scripts supply), a different
consequence in a different file.

**Location:** `scripts/verify_2_6_commit_a.sh:113-114`:
```bash
python scripts/01_run_baseline.py --om-case om_none         > "${OUT}/postpatch_annual.log" 2>&1
python scripts/01b_run_baseline_sliced.py --om-case om_none > "${OUT}/postpatch_sliced.log" 2>&1
```
— the ONLY flag supplied to either runner at HEAD is `--om-case`. Per
A26, both scripts now require 18 flags each; the other 17 are missing.

**What is wrong.** `set -euo pipefail` (`:18`) means the FIRST
non-zero exit anywhere in the script aborts it immediately. Line 113
would fail argparse's required-argument check and exit non-zero before
line 114 or the comparison step (`:126`, `python scripts/_compare_
2_6.py`) ever run. The script's own Step 1 "smoke test" (`:46-77`)
would report `"both builders construct, selfgen present"` successfully
— it calls `load_scenario()`/`build_model`/`build_model_sliced`
directly in Python, bypassing argparse entirely — creating a
misleading impression that the pipeline is healthy immediately before
the actual CLI invocation two steps later fails.

**Why it matters.** This is a regression tool for exactly the kind of
change this project makes often (per its own header: comparing a
pre-patch commit against HEAD to confirm an optimum did not move) —
and per the "VERIFICATION PROTOCOL THAT WORKED" section of `merged_
correction_plan_v2.md`, this class of tool is what the project
considers its strongest defence against a silent regression. As
written, at current HEAD, it cannot complete a single comparison.

**Some awareness already present, not enough.** The script's own
comment at `:104-112` shows the author DID revisit this file to add
`--om-case` when that flag became required — but not the other
required flags added across the F1/2.4-2.13 phases the same runners
now also demand. A partial update, not a full one.

**Severity:** Same classification as A26 — not a wrong published
number, a harness that cannot currently run to completion.

**Declared?** No.

---

## DOCUMENTATION VS. CODE

### `docs/methodology.md` — read in full

### A28 — `methodology.md` restates a claim already disproven twice this session, still uncommitted, still wrong on three counts

**Fingerprint:** F7 (documentation contradicting code).

**Location:** `docs/methodology.md:645-648` (§8, "Open method
decisions", the "Buffer-year capital spend" item):
> *"`npv_capex_report` in `src/optimize_experiments.py` computes a
> FULL-HORIZON figure under a name implying the report window; it is
> not used by any runner, and the naming defect is recorded in the
> register rather than corrected, since the field has no consumers."*

**What is wrong, verified fresh against the live code a fourth time
(not repeated from memory)** — `src/optimize_experiments.py:3127-3129`:
```python
_report_T = [t for t, y in enumerate(years) if int(y) <= REPORT_END_YEAR]
npv_capex_report = sum(float(pyo.value(m.DF[t])) * _capex_at(t) for t in _report_T)
npv_capex_full = sum(float(pyo.value(m.DF[t])) * _capex_at(t) for t in range(len(years)))
```
Wrong on three counts:
1. **The window pairing is backwards.** `npv_capex_report` is the one
   TRUNCATED to `_report_T` (`≤ REPORT_END_YEAR = 2045`) — the REPORT
   WINDOW figure, exactly as its name claims. `npv_capex_full` is the
   untruncated, full-2025-2054-horizon one. `methodology.md` has this
   exactly backwards.
2. **"Not used by any runner" is false.** `grep -rln "npv_gross_capex_
   report_window_usd" scripts/*.py` returns THREE files:
   `01_run_baseline.py` (has read it since before this session's own
   work began — `git log -S` traces the key itself to commit `74d34f2`,
   weeks before any task in this sequence), `01b_run_baseline_
   sliced.py` (added two tasks ago, `task_capex_reporting_field`), and
   `29_run_parameter_uncertainty.py` (this session's own Monte Carlo
   script). The field is read, live, by three runners.
3. **Consequently, the framing itself is wrong** — this is not a
   "naming defect with no consumers" at all; it is a correctly-named,
   correctly-computed, actively-consumed field.

**This is not a fresh discovery — it is a claim this task sequence
already disproved, twice, in prior tasks' own reports
(`task_capex_reporting_field.md`'s "Corrections" section, and flagged
directly to the user mid-session before this task even started: *"I
just re-checked the live code a third time — this is backwards..."*).
Despite that, `docs/limitations_register.md` at the time
`task_capex_reporting_field` ran (2026-09-16) already required no
correction to methodology.md — the whitelist for that task did not
include this file — and the claim was never fixed here. `git status
--porcelain` confirms `docs/methodology.md` is STILL uncommitted right
now, meaning this exact incorrect text has been sitting in the working
tree, unresolved, across at least three subsequent tasks in this
sequence (Monte Carlo Part 1/2, the Phase-1 audit and its fixes, and
now this Phase-2 audit), each of which could have — and did not —
correct it, because none of their whitelists included this file.

**Why it matters.** This is `methodology.md`'s own worked answer to
"what is the reported financing headline and does anything compute
it" — exactly the question this project's reporting rule
(`frozen_state_2026_09_10.md` §8) turns on. A reader of the
methodology chapter would conclude the headline financing figure is
an uncorrected, unused, misnamed field — the opposite of true.

**Severity:** LIVE-REPORTED-adjacent — not a wrong NUMBER in a finding
(the actual computed figures, 12.772 bn / 14.350 bn quoted three
sentences earlier in this SAME section, are correct and consistent
with the code), but a wrong CLAIM about the field's own definition and
liveness, sitting in the methodology document a reader would consult
to understand what those two correct figures actually are.

**Declared?** No — and actively contradicted by this task sequence's
own prior corrections, which were never applied to this specific file.

### The other six documents — sampled, not fully cross-checked; see the coverage statement

`docs/findings_frozen.md`'s M-4 section (the newest content, from the
most recent commit touching this file, `c8bcdfa`) was spot-checked in
full against the actual `results/parameter_uncertainty/summary_
n256_seed42.json` this session produced: every number quoted — the QoI
means/sd/percentiles, the Sobol first-/total-order indices per
parameter, the "central sits below the 5th percentile" observation —
matches the real output file to the decimal place. No discrepancy
found. `frozen_state_2026_09_10.md`, `methodology_demand_horizon.md`,
`methodology_genself_backstop.md`, `methodology_physical_layer.md`,
and `methodology_protocols.md` were NOT read in full this task — see
the coverage statement below for exactly what was and was not checked
in each.

---

## TIER 3 — deferred and retired

`scripts/deferred/17_run_monte_carlo.py`: already thoroughly covered
by Phase 1's A11 (the same independent-sampling defect exists
separately in `src/stochastic.py`) and by this project's own register
(L10) and the deferred script's own two-defect header. Re-read the
header only, to confirm it still self-describes accurately — it does.
No new findings.

`scripts/retired/09b_run_rel1_mode_sensitivity.py` and `scripts/
retired/09b_run_rel1_retirement_sensitivity.py`: both carry thorough,
accurate "RETIRED... Kept for provenance only" headers explaining
exactly why (a superseded `grid_service_mode` sweep axis; a superseded
gas-retirement proxy), each correctly citing the register entries that
make them obsolete (L5, L8, L10). Read both headers in full; no new
findings, nothing misleading found for a future reviver — these two
headers are a model of what A15/A18's "would mislead a future reviver"
risk looks like when done RIGHT.

**Utility/diagnostic scripts not reached in this pass**: `scripts/_
check_solver.py`, `scripts/_compare_2_6.py`, `scripts/build_gas_
deliverability.py`, `scripts/diag_fin2.py`, `scripts/f1_quantity_
diagnostic.py`, `scripts/plot_storage_regime.py`, `scripts/plot_
system_diagnostics.py`, `scripts/probe_2_6_structure.py`, `scripts/
rpublic_diagnostic.py` — nine files, named explicitly here rather than
silently skipped. None is a numbered runner and none was named in
TASK.md's own file list; not audited due to time, not because they
were judged low-risk. See the coverage statement.

---

## PHASE 2 — CLOSING SECTIONS

### LIVE-REPORTED list, first and separately

**None of the eleven Phase 2 findings (A18-A28) meets the strict bar
of "affects a number currently published in `findings_frozen.md` or
`frozen_state_2026_09_10.md`."** Checked explicitly for each, not
assumed. The closest:

- **A28** (`methodology.md`'s stale, backwards claim about `npv_
  capex_report`): the NUMBERS quoted around it (12.772 bn / 14.350 bn)
  are correct; the CLAIM about which field is which, and whether
  either has consumers, is wrong. A reader relying on the prose rather
  than the numbers would come away with a false understanding of a
  reported figure's own provenance.
- **A18** (`00_build_emissions_cap.py`'s KeyError trap): the
  currently-published `emissions_cap.csv` (and everything downstream
  of it — F-8, F-9, F-10, F-21) is unaffected; the SCRIPT that would
  refresh it is currently broken.
- **A26/A27** (the two broken verification harnesses): affect this
  project's ability to VERIFY a number stays correct after a future
  change, not any number published today.

### Findings table (A18-A28)

| ID | Fingerprint | File | Severity | Declared? |
|---|---|---|---|---|
| A18 | F10 | `scripts/00_build_emissions_cap.py` | TRAP | Partially (task report, not register) |
| A19 | F5 | `scripts/01b_run_baseline_sliced.py` + 3 others | TRAP | No |
| A20 | F5 | `scripts/02_run_ndc_caps.py`, `03_run_ndc_eaas.py` | TRAP | No |
| A21 | F4 | `scripts/02_run_ndc_caps.py`, `03_run_ndc_eaas.py` | TRAP | No |
| A22 | F1/F10 | `scripts/26_run_f3_identification.py`, `27_run_hydro_minflow_sensitivity.py` | LIVE-LATENT | No |
| A23 | other (stale register scope) | `docs/limitations_register.md` L10 | other | Corrects L10 |
| A24 | F11 | `scripts/24_run_leakage_arm.py`, `25_run_obligation_trajectory.py`, `99c_run_genset_arm_sensitivity.py` | LIVE-LATENT | No |
| A25 | F6 | `scripts/run_central_sweep.sh` | COSMETIC | No (correct figure declared elsewhere) |
| A26 | F12 | `scripts/run_verification_sweep.sh` | TRAP (already failing) | No |
| A27 | F12 | `scripts/verify_2_6_commit_a.sh` | TRAP (already failing) | No |
| A28 | F7 | `docs/methodology.md` | LIVE-REPORTED-adjacent | No (contradicts prior corrections) |

### Docs-vs-code verdict, all seven named documents

| Document | Verdict |
|---|---|
| `methodology.md` | **Cross-checked in full.** Read start to end; one major finding (A28). |
| `methodology_demand_horizon.md` | **Sampled only** — grepped for one specific F7 spot-check in an earlier task (the "non-binding after 2045" vs "held flat" question, found resolved); not read in full this task. |
| `methodology_genself_backstop.md` | **Sampled only** — same one spot-check as above; not read in full. |
| `methodology_physical_layer.md` | **Not reached.** Not read or grepped this task at all. |
| `methodology_protocols.md` | **Not reached** as a full read; its newest section (referenced by `findings_frozen.md`'s M-4, on parameter-vs-scenario uncertainty) was not independently opened, though M-4's OWN numeric claims were cross-checked against real output (see below). |
| `findings_frozen.md` | **Sampled.** Full document read in Phase 1 for the scoped-conditions list; this phase additionally spot-checked the newest section (M-4) in full against real output data — matched exactly, no discrepancy. The rest of the document (F-1 through F-21, M-1 through M-3) was not re-read this phase beyond what Phase 1 already covered. |
| `frozen_state_2026_09_10.md` | **Not reached** as a full read this phase; quoted/cross-referenced piecemeal (the `-3.2%` backcast figure, the reporting rule) via targeted `grep`, not read start to end. |

**Net**: of the three documents Phase 1 flagged as never cross-checked
(`methodology.md`, `methodology_physical_layer.md`, `methodology_
protocols.md`), one (`methodology.md`) is now fully checked and
produced this phase's most consequential finding. The other two remain
unchecked. This is stated plainly as a continuing gap, not implied
closed.

### Coverage statement

**Scripts read in full, this phase:** `00_build_emissions_cap.py`,
`01b_run_baseline_sliced.py`, `02_run_ndc_caps.py` (in full — 577
lines), `run_central_sweep.sh`, `run_verification_sweep.sh`,
`verify_2_6_commit_a.sh`, `smoke_all.sh`. Both `scripts/retired/09b_*`
headers and `scripts/deferred/17_run_monte_carlo.py`'s header.

**Scripts read substantially (specific sections, not necessarily
start-to-end) and cross-checked against live code/data:**
`03_run_ndc_eaas.py` (mirrored against `02_run_ndc_caps.py`'s
findings, spot-checked independently), `26_run_f3_identification.py`,
`27_run_hydro_minflow_sensitivity.py`, `23_run_selfgen_frontier.py`
(re-confirmed clean post-A15-fix, from Phase 1 work), `29_run_
parameter_uncertainty.py` (self-authored this session, cross-checked
against its own real output), `18_run_pareto_frontier.py`,
`28_run_capital_frontier.py`, `19_run_carbon_price_sweep.py`,
`20_run_cf1_solar_yield_sensitivity.py`, `22_salvage_sensitivity.py`,
`24_run_leakage_arm.py`, `25_run_obligation_trajectory.py`,
`99c_run_genset_arm_sensitivity.py` (builder-import and required-flag
checks only, not full reads), `01_run_baseline.py`, `02_run_eaas.py`
(required-flag counts only, for A26).

**Scripts NOT reached in this pass, named explicitly rather than left
implicit:** `04_run_fin2_tariff_sweep.py`, `05_run_fin3_capital_
sweep.py` (cli_arms usage confirmed only, not a full read),
`06_run_gas1_shadow_benchmarks.py`, `07_run_gas2_eaas_gas_relief.py`,
`08_phase8_analysis.py`, `08_run_gas3_regime_ndc_feasibility.py`,
`09_run_rel1_feasibility.py`, `10_run_rel2_marginal_cost.py`,
`11_run_rel3_financing_frontier.py`, `12_run_dem1_demand_
sensitivity.py`, `13_run_dem2_growth_gas_interaction.py`, `14_run_
str1_storage_role.py`, `15_run_pol1_ndc_comparison.py` (cli_arms usage
confirmed only), `16_plot_cost_decomposition.py`, `21_compute_
resolution_premium.py`, `99_validate_baseline.py`, and the nine
utility/diagnostic scripts named in the Tier 3 section above
(`_check_solver.py`, `_compare_2_6.py`, `build_gas_deliverability.py`,
`diag_fin2.py`, `f1_quantity_diagnostic.py`, `plot_storage_regime.py`,
`plot_system_diagnostics.py`, `probe_2_6_structure.py`, `rpublic_
diagnostic.py`). Roughly 20 of the ~45 files in `scripts/` (including
subdirectories) were not read this phase. Most of the numbered ones
among these are already covered by register L10/L27's dead-runner
classification (re-confirmed accurate for `06_`/`07_` specifically via
A23's own check; not re-verified individually for the remaining ~13
numbered scripts in this list) — but "already declared dead" is not
the same claim as "audited and found to have no OTHER defect," and
this list should not be read as such.

**Documents**: see the docs-vs-code verdict table above.

**Nothing edited. Nothing run. Nothing committed** — every check in
this phase was a read, a `grep`/`bash -n`/`git log` (read-only), or a
side-effect-free Python/JSON/CSV read of an already-existing file
(including this session's own already-completed Monte Carlo run,
inspected as evidence, never re-run).

---

## PHASE 3 — the unread scripts

Continuing the A-numbering from A29. Code-only scope, by explicit
instruction — the methodology documents (`methodology_physical_layer.md`,
`methodology_protocols.md`) are out of scope this phase; the one
carve-out is any claim about what a CONSTRAINT DOES, cross-checked
against the code (register L13's own numeric claim about `build_gas_
deliverability.py`'s output falls under a closely related carve-out —
TASK.md names this file's own closing verdict as mandatory — see A33
and the closing statement). Nothing edited, run, or committed. Written
incrementally, file by file, as each was finished, per instruction —
**Tier 2 (the nine utility/diagnostic scripts) was audited first**
(personally, not delegated) because `build_gas_deliverability.py` is
this phase's own named highest-priority target; Tier 1 (the numbered
runners not read in Phases 1-2) was delegated to three parallel
read-only research passes running concurrently, reported below once
each returned. This ordering is stated plainly rather than implied to
match TASK.md's own file list order (Tier 1 before Tier 2) — the
findings table at the close is the authoritative index, as Phase 1's
own closing section already established for a similar ordering
mismatch there.

**Established freshly this phase, not trusted from TASK.md's own
framing:** the exact Tier 1 "unread" file list. Diffed the full
`scripts/` directory (46 `.py` files, 4 `.sh` files, confirmed via
`find scripts -name "*.py"`) against Phase 2's own coverage statement.
Phase 2 explicitly named its own "NOT reached" list: `04_run_fin2_
tariff_sweep.py`, `05_run_fin3_capital_sweep.py` (cli_arms usage
confirmed only), `06_run_gas1_shadow_benchmarks.py`, `07_run_gas2_eaas_
gas_relief.py`, `08_phase8_analysis.py`, `08_run_gas3_regime_ndc_
feasibility.py`, `09_run_rel1_feasibility.py`, `10_run_rel2_marginal_
cost.py`, `11_run_rel3_financing_frontier.py`, `12_run_dem1_demand_
sensitivity.py`, `13_run_dem2_growth_gas_interaction.py`, `14_run_str1_
storage_role.py`, `15_run_pol1_ndc_comparison.py` (cli_arms usage
confirmed only), `16_plot_cost_decomposition.py`, `21_compute_
resolution_premium.py`, `99_validate_baseline.py` — sixteen files. This
is Tier 1 for this phase.

---

### TIER 2 — the nine utility and diagnostic scripts

Audited first and personally (not delegated), per this file's own
framing as the highest-stakes target this phase.

#### `scripts/build_gas_deliverability.py` — read in full

**No code defect found.** This is the single most consequential
verdict this phase can deliver, so it is stated with the evidence in
full rather than summarised.

**What it computes, verified against the checked-in CSV byte-for-byte,
not merely inspected.** Imported `build_series()` directly (a pure
function — side-effect-free, no file written) and re-derived every one
of the 300 rows `git diff --stat` confirms `data/gas/processed/
gas_available_power_annual_twh_th.csv` currently holds (10 scenarios ×
30 years, 2025-2054), from the current `data/gas/processed/gas_
deliverability_cases.yaml` and the current script's own `build_series`/
`main()` logic (including the `flat_downside`/`flat_upside`/`flat_
shock_recovery` auto-computed level-equivalents, correctly averaged over
the REPORT_END_YEAR-truncated window per the script's own comment at
`:154-156`):
```
computed rows 300   disk rows 300
mismatches 0   missing_on_disk 0   extra_on_disk 0
scenarios on disk == scenarios computed (10 of 10)
```
Working tree is clean against HEAD for the script, the yaml, and the
CSV (`git diff --stat HEAD --`: no output) — so this is not a case of
the checked-in file predating the current generator; it is the CURRENT
generator's exact, reproducible output, confirmed independently rather
than assumed from adjacent commit timestamps.

**Also verified:** the two-family scenario vocabulary (`downside/
baseline/upside/shock_recovery` + `flat_*`, vs `gas_deliv_flat/gmp_
partial/gmp_full`) matches `src/scenarios.py:757-771`'s `gas_
deliverability_scenarios()` registry exactly, key for key, both
directions (no orphan on either side) — the CSV's `scenario` column and
the registry that validates `gas_deliverability_case` against it cannot
drift apart today. The `gas_availability_case` axis (a SEPARATE,
mechanical-availability derating factor, `gas_avail_nameplate=0.80` etc,
`src/scenarios.py:941-979`) is correctly orthogonal to this file's
deliverability series — confirmed by reading both, not assumed from
naming similarity; no F4-style confusion between the two axes exists in
the code.

### A29 — Register L13's own quoted GMP-family lower bound does not match what the current, correctly-reproducing generator actually computes

**Fingerprint:** F7 (documentation contradicting code) — the code
itself is exactly right (see the verdict above); the register's
descriptive text about it is not, on one specific figure.

**Location:** `docs/limitations_register.md:469-470` (L13): *"The GMP
family reaches 114.9-191.7 TWh_th by 2030."*

**What is wrong, computed directly from the current `build_series()`
against the current yaml, not estimated.** At year 2030:
```
gas_deliv_flat        (I=0.0):    89.27
gas_deliv_gmp_partial (I=0.5):   140.51
gas_deliv_gmp_full    (I=1.0):   191.75
```
`191.75` rounds to L13's stated `191.7` — correct, and independently
matches the yaml's own header comment (`gas_deliverability_cases.yaml:
101-104`: *"Reaches 191.7 TWh_th by 2030"*), so the upper bound is right
twice over. `140.51` — the GMP-partial value, the arm actually used as
the CENTRAL case in this family (yaml comment, `:87`: *"CENTRAL"*) — is
**not** 114.9. No combination of the current `lag_ramp_uplift` formula
against the current `uplift_fraction: 0.574` and `ramp_years: 5`
produces 114.9 at any single integer year 2025-2054 that also lands
exactly on 191.7 for the I=1 arm under the same ramp_years.

**A plausible mechanism, offered as an explanation, not asserted as
fact.** Solving backward, `89.27 × (1 + 0.574 × 5/10) = 114.89` — i.e.
114.9 falls out exactly if `ramp_years` were **10**, not 5, holding
everything else fixed (the I=1 arm's `191.7` is insensitive to this,
since a full-ramp value only depends on `uplift_fraction`, not
`ramp_years`, once the ramp has completed by the read year — 2030 is
past completion at ramp_years=5, so both ramp lengths would give the
same 191.7 for I=1, but DIFFERENT partial-ramp values for I=0.5 at
2030). `git log -p` shows the GMP family's `uplift_fraction`/`ramp_
years` values were introduced together, in one commit (`5b811f2`, Aug
28 18:57), with no prior version to diverge from — so if a `ramp_
years: 10` draft ever existed, it was never itself committed; the
register entry (`b3d049d`, same day, 23:30) most likely reasoned from an
uncommitted intermediate figure or a hand-computation error, not from a
since-changed committed value. This is offered as the most likely
explanation of a 25.6-TWh_th discrepancy that is otherwise an unexplained
arithmetic slip — not confirmed by any further evidence, and not needed
to establish the finding itself: the code is verified correct; the
register's specific "114.9" figure is verified wrong.

**Why it matters.** L13 is the register's own authoritative statement on
exactly the file TASK.md calls "the highest-stakes file in this phase."
A reader citing L13's own numbers for the GMP family's 2030 range (e.g.
to state how large the policy-target scenario spread is) would
understate the CENTRAL GMP arm's own 2030 value by 26 TWh_th (114.9
quoted vs 140.5 actual) — a ~23% understatement of the very quantity the
family exists to bound.

**Not new: the drift-family range (81.5-94.7) and the general two-family
framing are independently verified CORRECT** — checked at 2030 for all
four drift arms (`downside 81.47, baseline 91.52, upside 94.73, shock_
recovery 86.59`), matching L13's `81.5-94.7` exactly. Only the one
number (114.9) is wrong; the entry is not wholesale unreliable.

**Severity:** COSMETIC-to-TRAP by the strict "wrong number" bar — no
runner or the LP itself reads register prose, so nothing computed is
affected. Flagged prominently anyway, per this task's own explicit
instruction to verify `build_gas_deliverability.py`'s output "matches
what the register... say[s] the deliverability series is," and because
a reader relying on the register rather than re-deriving the series
themselves (the entire purpose of a register) would carry a materially
wrong number forward.

**Declared?** No — this is the finding.

---

### A30 — `f1_quantity_diagnostic.py` raises a "reporting defect" suspicion against a bug already fixed a week before the diagnostic was written

**Fingerprint:** F7 (a file's own embedded claim, stale relative to the
code as of its own writing).

**Location:** `scripts/f1_quantity_diagnostic.py:27-28` (module
docstring) and `:233-241` (the "REPORTING DEFECT" banner):
```python
# docstring:
Also checks a reporting defect: the runner prints "storage 2045" but
appears to emit TERMINAL capacity, not the 2045 value.
...
# body:
banner("REPORTING DEFECT -- 'storage 2045' label check")
...
    print("\nThe runner prints 'storage 2045'. If that print matches the")
    print("TERMINAL column rather than the 2045 column, the summary block")
    print("is mislabeled and other lines in it should be checked too.")
```

**What is wrong, verified against git history, not inference.**
`scripts/01b_run_baseline_sliced.py`'s console print was changed from a
hardcoded `f"  storage 2045   : ..."` to the dynamic `f"  storage
{int(years[-1])}   : ..."` in commit `eef2b8d` ("F2: ... storage-year
label fix"), dated **2026-08-28 12:53**. `f1_quantity_diagnostic.py`
itself was created in commit `b7f7bdb`, dated **2026-09-05** — nine days
LATER. Since `MODEL_END_YEAR` is 2054 (`src/scenarios.py:286`), `years
[-1]` is 2054, not 2045, and the runner today prints `"storage 2054"`,
correctly labelled and correctly valued — self-consistent, not
mislabeled, and has been since before this diagnostic script existed.

**Why it matters.** A future reader running this diagnostic today would
be pointed at a "REPORTING DEFECT" check for a defect that was fixed
before the check was written — time that could be spent chasing a ghost
in `01b_`'s console output instead of the diagnostic's own three live
hypotheses (H1-H3), which — separately verified this phase against the
project's own published finding — are sound: H1/H2's conclusions match
`docs/findings_frozen.md` **F-19** ("storage magnitude is not a cost
optimum... once gas reaches its floor, storage absorbs the entire
firm-capacity residual") to the letter, and the docstring's own
hardcoded reference figures (terminal storage 75,481 MWh, terminal gas
5,313 MW) are confirmed, exactly, against real on-disk results for the
`cr_comm_low` arm (`results/baseline_sliced/summary__..._cr_comm_low_
..._env_dares.json`: `"final_storage_capacity_mwh": 75481.05940000003`;
matching timeseries CSV: `5313.397534` MW gas). The diagnostic's actual
analytical content is trustworthy; only its one stale suspicion is not.

**Severity:** COSMETIC — nothing computed is wrong; the check, if run
today, would simply confirm the runner is fine, wasting a reader's time
rather than misleading them about a number.

**Declared?** No.

---

### A31 — `plot_storage_regime.py` cannot currently run, and would crash a second way if its input path were fixed

**Fingerprint:** F10 (output/input path nothing correctly maintains) for
the first half; F3/F7 (stale vocabulary) for the second.

**Location:** `scripts/plot_storage_regime.py:6` (`RESULT_PATH`), `:16-
23` (`mapping`).

**What is wrong (path).** `RESULT_PATH = Path("results/ndc/ndc_
unconditional_20_voll_low/diagnostics.json")` — confirmed via direct
`ls`: **this path does not exist anywhere on disk.** Every actual
directory under `results/ndc/` is named either a short `ndc2_
{un}conditional_<hash>` form or the full arm-string form (e.g. `ndc2_
unconditional_voll_mid_peak_unsuppressed_storage_cc_central_...`) — no
directory named `ndc_unconditional_20_voll_low` exists under any naming
convention currently in use. Running this script today raises
`FileNotFoundError` on its second line, before anything else executes.

**What is wrong (vocabulary), would fire the moment the path above were
fixed.** `mapping = {"none": 0, "power_limit": 1, "cycle_limit": 2,
"solar_limit": 3}`. The diagnostic field this script plots,
`storage_binding_by_year`, is produced by `extract_planning_
diagnostics` (`src/optimize_experiments.py:2699-2709`) with exactly
THREE possible values today: `"none"`, `"power_limit"`, `"energy_
limit"` — **`"cycle_limit"` and `"solar_limit"` do not exist in the
current vocabulary at all**, and `"energy_limit"` (which does) is
absent from `mapping`. Any year classified `"energy_limit"` — which,
per the same function's own comment (`:2681-2688`), is the
sliced-builder's actual, if not fully meaningful, common case — would
raise `KeyError: 'energy_limit'` at `mapping[v]` (`:23`).

**Compounding context, not a separate finding.** The field this script
plots is exactly the one register **L11** already declares "not
meaningful under the sliced builder" and states is "read by nothing."
This script IS a reader — broken today, but a reader nonetheless,
contradicting L11's "read by nothing" claim in spirit if not in current
practice. Named here for completeness, not as an independent finding:
L11 already covers the underlying meaninglessness; this entry is about
the plotting script's own two independent, code-level defects.

**Why it matters.** Squarely the risk TASK.md's own framing names for
plotting scripts: "a plotting script reading a stale field or a renamed
key produces a wrong figure that looks right." This one is currently
worse than that — it cannot produce any figure at all — but if a future
author "fixed" only the path (the more visible of the two defects)
without checking the vocabulary against the live diagnostic function,
they would replace a loud `FileNotFoundError` with a loud `KeyError`,
not a silently-wrong figure; the risk of a wrong-but-plausible-looking
figure exists only if a THIRD fix (adding an `"energy_limit"` mapping,
guessed at rather than checked against L11) papered over both without
reading L11 first.

**Severity:** TRAP — cannot run today (no published figure depends on
it), positioned to fail a second, less obvious way the moment the first,
obvious failure is patched.

**Declared?** No.

---

### A32 — `plot_system_diagnostics.py` carries two dead variables whose unreachable fallback branches contain typos that would crash the script under an easy, plausible edit

**Fingerprint:** F1 (defined, never read) combined with F9 (a branch
that cannot currently execute).

**Location:** `scripts/plot_system_diagnostics.py:32-39`:
```python
if "gas_shadow_usd_per_twh_th" in df.columns:
    gas_shadow = df["gas_shadow_usd_per_twh_th"]
else:
    gas_shadow = df["gas-shadow_price_usd_per_twh_th"]      # hyphen, not underscore
if "carbon_shadow_usd_per_tco2" in df.columns:
    carbon_shadow = df["carbon_shadow_usd_per_tco2"]
else:
    carbon_shadow = df["caron_shadow_price_usd_per_tco2"]   # "caron", not "carbon"
```

**What is wrong.** Neither `gas_shadow` nor `carbon_shadow` is
referenced anywhere else in the file (`grep -n "gas_shadow\b\|carbon_
shadow\b"` on the file: only these four lines) — both variables are
dead on arrival (F1). Their `else` branches are unreachable TODAY only
because `df` is hardcoded to `dfs["baseline"]` (`:22`), and `results/
baseline/timeseries.csv` happens to carry the exact primary-branch
column names (confirmed by reading its header: `gas_shadow_usd_per_
twh_th`, `carbon_shadow_usd_per_tco2`, no `_price_` variant). The other
four loaded cases (`results/ndc/ndc3_{un}conditional_voll_mid/
timeseries.csv`, `results/ndc_eaas/ndc3_*_eaas_voll_mid/timeseries.csv`
— all five paths confirmed to EXIST, unlike A31) use `gas_shadow_price_
usd_per_twh_th` instead (with `_price_`) — so pointing `df` at any of
those four instead of baseline (a small, easy edit — e.g. to inspect an
NDC case's shadow prices as the "primary" panel instead of baseline)
would take the `else` branch for `gas_shadow` and immediately raise
`KeyError` on the typo'd, nowhere-existing column name `"gas-shadow_
price_usd_per_twh_th"`. (`carbon_shadow_usd_per_tco2` happens to be
present, unchanged, in all five files, so that specific `else` branch is
harder to trigger by this same edit alone — but is equally broken if
ever reached by some other case.) Separately, and correctly, `gs_col`
(`:112`) and `cs_col` (`:97`) — the variables actually plotted — check
the SAME two column-name variants in the opposite priority order and
ARE used, and are not implicated in this finding.

**Why it matters.** Nothing published depends on the dead variables
today. The risk is specific: a future author extending this script to
compare shadow prices across cases (the script's own Panel A and Panel
B carbon-shadow loop already do exactly this kind of multi-case
iteration) who copies the `gas_shadow =` pattern rather than the
working `gs_col =` pattern would hit a `KeyError` from a typo, not a
meaningful error — wasting time on a bug with no informative message.

**Severity:** COSMETIC today (dead code, unreachable branch); the crash
risk is real but requires a specific future edit to trigger.

**Declared?** No.

---

**The other seven Tier 2 files: no new findings.** `scripts/_check_
solver.py` (13 lines, trivial solver-availability probe — clean).
`scripts/_compare_2_6.py` (already partly examined via A26/A27's own
fixes last task — every field it reads (`npv_total_cost_usd`,
`decision_variables.solar_total_built_mw`/`final_storage_capacity_mwh`,
`actual_emissions_tco2_total`, timeseries `unserved_twh`/`selfgen_twh`)
confirmed present, under identical names, in both `01_run_baseline.py`
and `01b_run_baseline_sliced.py`'s current output schemas — no drift).
`scripts/diag_fin2.py` (correct pandas logic on a read of `04_run_
fin2_tariff_sweep.py`'s already-deprecated, still-on-disk output CSVs;
not a live producer of any currently-cited number; its own H1-H5
hypothesis tests do not appear to have been written up anywhere this
audit found, but that is a coverage gap in the PROJECT's use of its own
tooling, not a code defect in the script). `scripts/probe_2_6_
structure.py` (careful, well-guarded side-effect-free structural probe
with its own internal sanity anchors — read in full, no defect found).
`scripts/rpublic_diagnostic.py` — see A33 below, one finding, otherwise
clean. `scripts/plot_storage_regime.py` and `scripts/plot_system_
diagnostics.py` — see A31/A32 above.

---

### A33 — `rpublic_diagnostic.py` hardcodes solar asset life at 25 years; the project's sourced, live value is 30

**Fingerprint:** F6 (epoch/reference-stale hardcoded figure) — though
this constant was likely never live in the first place rather than
having drifted from a once-correct value; reported under F6 as the
closest match.

**Location:** `scripts/rpublic_diagnostic.py:80`:
```python
LIFETIMES = {"solar": 25, "gas": 30, "storage": 15}
```
against `src/scenarios.py:672-676` (`asset_lifetimes()`, SOURCED,
IRENA 2026):
```python
return {
    "solar":   30.0,
    "storage": 15.0,
    "gas":     30.0,
}
```

**What is wrong.** `gas` (30) and `storage` (15) match the live,
sourced function exactly. `solar` does not: the diagnostic uses 25, the
model uses 30. This is not a rounding or unit issue — 25 vs 30 is a
20% difference in service life, which enters every capital-recovery
factor in this script inversely (a shorter life raises the annuity
factor for the same rate). The diagnostic's own **Section 5** compounds
this: it sweeps solar lifetime specifically because "an unsourced asset
lifetime propagates into every CRF... silently" (`:204-207`), and uses
**25 as its own reference point** (`ref = crf(real, 25)`, `:191`,
printed as `"vs L=25"`) rather than the model's actual reference, 30 —
so the tool built to test exposure to this exact class of error is
itself carrying a 5-year miss on the one technology (solar) whose
lifetime this project's own register flags a related open question
about (`asset_lifetime_sweep()`, adjacent in `scenarios.py`, exists
specifically to test this sensitivity for reported results — this
diagnostic is a separate, standalone tool, not a caller of it, so the
two do not currently interact, but a reader could reasonably conflate
this diagnostic's "vs L=25" framing with the model's own convention).

**Why it matters.** This is a standalone, no-repo-dependency exploratory
tool (per its own docstring, "no repo or solver dependency") — its
`CAPEX_USD_PER_MW` figures (905k solar, 1,317k gas — both independently
confirmed against `src/scenarios.py:837,1649,1663,1666`'s sourced
figures) and its `US_INFLATION`/rate stances are otherwise carefully
sourced and cited inline. The one unsourced-looking constant that
SHOULD match a value the project already has SOURCED and coded
(`asset_lifetimes()`) does not. Every solar-related number in Sections
1, 2, 4, and 5 of this script's own output (CRF, USD/MW/yr capital
charge) is computed against the wrong asset life.

**Severity:** LIVE-LATENT — this script produces no persisted output and
is not called by anything; the numbers it prints to console are used
for exactly the kind of ad-hoc, is-this-load-bearing sensitivity check
its own docstring describes, and would currently mislead that check
specifically for solar.

**Declared?** No.

---

### TIER 1 — the sixteen unread numbered runners

Delegated to three parallel read-only research passes (each briefed with
the exhaustive known-set above, the twelve fingerprints, and the frozen
anchors), since this is mechanical, high-volume file-by-file work rather
than the single highest-priority target Tier 2 was. Every finding below
that is flagged **"personally verified"** was independently re-checked
by direct file read against the exact quote and line number before being
written here — done for the highest-severity or most surprising claims,
not all of them, given volume; the rest are reported as returned,
attributed to the pass that found them, with any discrepancy from a spot
check noted explicitly. None of the three passes edited, ran, or built
any model — confirmed in each pass's own report.

#### `scripts/05_run_fin3_capital_sweep.py` — Tier 1, live (cli_arms-repaired)

**What it produces / could be cited.** `results/fin3/capital_sweep_
results.csv`, `capital_sweep_summary.json` — a capital-budget ×
DisCo-collection-rate sweep under NDC3.0-unconditional. Not currently
cited anywhere (`grep` for "fin3"/"FIN-3": no hits in `findings_frozen.
md`, `limitations_register.md`, `merged_correction_plan_v2.md`).

### A34 — `disco_collection_rate` is written into the scenario dict and never read; the runner's actual collection-rate mechanism is a different, undeclared lever

**Fingerprint:** F1 (defined, never read), on a scenario-dict key rather
than a whole function.

**Location:** `scripts/05_run_fin3_capital_sweep.py:232`:
```python
scenario["disco_collection_rate"] = cr_value
```
against the runner's OWN, separate, actually-effective mechanism at
`:212-217`, a hand-computed capital-budget haircut (`domestic_budget =
base_budget * DOMESTIC_SHARE * cr_value; ...; scenario["public_solar_
budget_npv"] = effective_budget`) applied BEFORE line 232 ever runs.
Neither `src/optimize_model.py` nor `extract_planning_diagnostics` reads
`disco_collection_rate` anywhere (confirmed by grep: zero hits outside
this one assignment).

**Why it matters.** `docs/merged_correction_plan_v2.md:786` already
documents `disco_collection_rate` as an unused stub in `scenarios.py`'s
default, intended (per that entry, item 3.3, `[OPEN]`) to eventually
gate a DEMAND-SIDE revenue haircut (`effective revenue = tariff ×
collection_rate`). This runner's own collection-rate sensitivity is a
SUPPLY-SIDE capital haircut instead — a different mechanism entirely —
while still writing to the named-but-unwired key, which would mislead a
reader of `build_model()`'s signature into concluding this sensitivity
isn't applied at all.

**Severity:** TRAP (the sweep's numbers are real, via the manual budget
override; the vestigial key write is a red herring for a future reader,
not a live wrong-number risk).

**Declared?** Partially — the general "disco_collection_rate unused
stub" is known (`merged_correction_plan_v2.md` 3.3); this specific
runner still writing to it despite using a different mechanism is not.

### A35 — Two CSV columns from the same row, `budget_usd` and `budget_b_usd`, report different bases under confusingly parallel names

**Fingerprint:** F3/F4 (misdescribing name / two representations of one
quantity not enforced to agree).

**Location:** `scripts/05_run_fin3_capital_sweep.py:301-305`:
```python
"budget_usd":                effective_budget if base_budget is not None else None,
"budget_b_usd":              budget / 1e9 if budget else None,
"budget_fraction_bstar":     budget / envelopes["adequacy"] if budget else None,
```
`budget_usd` is the collection-rate-ADJUSTED effective budget;
`budget_b_usd` and `budget_fraction_bstar` use the raw NOMINAL `budget`
— for `cr_label="current"` (0.35 collection rate) these differ by
roughly 23%, despite names that read as unit-rescaled copies of the same
figure. The runner's own console table separates "Nominal B$" from
"Effective B$" correctly; the CSV does not carry that distinction in its
column names.

**Severity:** TRAP/COSMETIC — no current consumer of this CSV found.

**Declared?** No.

**Not re-reported (already known):** the B* docstring figure (6.13bn vs
coded 9.104bn) is `merged_correction_plan_v2.md`'s own flagged
STALE-DOC; `docs/runner_inventory.md` still labelling 05_ "DEAD" is the
same staleness A23 already found and amended (register L10) — a second
document carrying the same superseded claim, not a new defect.

---

#### `scripts/06_run_gas1_shadow_benchmarks.py` — Tier 1, dead on required args (known, not re-reported)

**What it produces.** `results/gas1/gas_shadow_matrix.csv`, `benchmark_
comparison.csv`, `gas_shadow_by_year.csv`, `gas1_summary.json` — matches
its own docstring; no orphan-output (F10) issue.

### A36 — The gas-price benchmark dictionary is hand-typed independently in at least four places across two files

**Fingerprint:** F5 (duplicate source of truth).

**Location:** `scripts/06_run_gas1_shadow_benchmarks.py:108-117`
(local); `src/optimize_experiments.py:3770-3779` (the shared `run_gas_
regime_ndc_matrix`, which 06_'s own Arm 2 also calls); `scripts/07_run_
gas2_eaas_gas_relief.py:109-118` AND `:213-222` (duplicated twice within
that one file too). All four copies are currently byte-identical
(diffed directly, not assumed) — a benchmark dict of seven domestic/
opportunity/export gas-price reference points (`dom_power_2p4` through
`lng_export_10`), keyed off `MMBTU_PER_TWH_TH = 1e9/293.071`.

**Why it matters.** No enforcement keeps these four copies equal. If the
shared, imported copy in `optimize_experiments.py` is ever revised
(a new benchmark added, a price updated) without touching the three
hand-typed copies in `06_`/`07_` (07_ alone has two to update), `gas_
shadow_matrix.csv`/`benchmark_comparison.csv` would silently compare
against two different benchmark bases across rows of the same file.
Distinct from the already-known A3/L32 finding (the dead, unused `src/
postprocess.py::export_benchmark_comparison`, a different function
entirely) — confirmed distinct, not a re-report.

**Severity:** TRAP — correct today, unenforced.

**Declared?** No.

**File otherwise clean:** builder choice (`build_model`, annual —
appropriate, since 06_ needs full-horizon diagnostics, not sliced), no
hardcoded frozen-anchor mismatches, output paths match the docstring.

---

#### `scripts/07_run_gas2_eaas_gas_relief.py` — Tier 1, dead on required args (known, self-diagnosed in-file, not re-reported)

**What it produces.** `results/gas2/eaas_vs_noeaas_gas_shadow.csv`,
`gas2_summary.json` — matches docstring; no F10 issue.

### A37 — A copy-paste bug mislabels the EaaS-arm helper's own success status, which would silently blank the cleanest of GAS-2's five NDC arms the moment this runner is repaired

**Fingerprint:** F12 (a harness/helper misreporting its own success or
identity) — **personally verified against the file, not taken on the
research pass's word alone.**

**Location:** `scripts/07_run_gas2_eaas_gas_relief.py:144` sets
`scenario["financing_regime"] = "eaas"` inside `run_no_policy_gas_
matrix_eaas` — genuinely the EaaS-on function. Its own result dict,
`:187-188`:
```python
"financing_regime":        "traditional",     # should be "eaas"
"status":                  "traditional",      # should be "optimal"
```
— an unedited copy of the sibling, non-EaaS function's dict at `:288-
289` (`"financing_regime": "traditional"`, `"status": "optimal"` —
correct THERE, since that one really is the traditional-financing arm).

**Confirmed downstream consequence, traced through `main()`.**
`run_no_policy_gas_matrix_eaas` is the SOLE handler for the
`baseline_no_policy` NDC arm (`NDC_CASES["baseline_no_policy"][
"ndc_cap_scenario"] is None` routes here, `:338`). Every later read of
this arm's result gates on `r_yes.get("status") == "optimal"`
(`:365,382-385`) — always False, since `status` is literally the string
`"traditional"`. `yes_shadow`, `eaas_binding_years`, `eaas_unserved_
twh`, `eaas_solar_total_mw`, `eaas_npv_cost` all collapse to `None` for
this arm; `eaas_status` in the output CSV wrongly reads `"traditional"`
(`:386`); and the summary filter (`:407-408`, `sub["eaas_status"] ==
"optimal"`) matches ZERO rows for `baseline_no_policy`, so `summary[
"baseline_no_policy"]["eaas_always_relieves_gas"]` silently becomes
`None`/not-available.

**Why it matters.** `baseline_no_policy` is the one NDC arm with no
emissions cap at all — the cleanest isolation of EaaS's pure gas-relief
mechanism, and GAS-2's own cleanest test case. This bug guarantees it
reports nothing, invisibly, the moment this runner receives the same
required-args repair already applied to `05_/15_/18_/19_/20_/28_`
(L27's pattern) — a repair this audit's own severity classification
would otherwise treat as low-risk mechanical work.

**Severity:** TRAP — dormant today only because the file crashes before
reaching this code (dead on required args); would corrupt results
silently, with no crash or warning, the instant it's repaired.

**Declared?** No.

### A38 — A schema gap between the EaaS-arm and NDC-arm result dicts silently blanks one column for the EaaS-arm's rows

**Fingerprint:** F2 (silent default via mismatched schema), minor.

**Location:** `run_no_policy_gas_matrix_eaas`'s result dict (`scripts/
07_run_gas2_eaas_gas_relief.py:184-201`) has no `"solar_total_mw"` key,
unlike `run_gas_regime_ndc_matrix`'s schema (used by the other four NDC
arms). `:384`, `"eaas_solar_total_mw": r_yes.get("solar_total_mw")...`
silently defaults to `None` via `dict.get` — no crash, but the column is
populated for 4 of 5 NDC arms and always blank for the 5th.

**Severity:** TRAP, minor.

**Declared?** No.

**Cosmetic, not separately findings-numbered:** duplicate imports
(`pyomo.environ as pyo` at both `:49` and `:53`; `load_econ` imported
twice, `:60-61,62`) — no functional effect.

---

#### `scripts/08_phase8_analysis.py` — Tier 1, live (runs to completion; output on disk dated 2026-09-15)

**Not the same file as GAS-3's `08_run_gas3_...` below** — a
standalone, no-docstring 5-stage aggregation script (master table,
cost/MAC frontier, mechanism decomposition, scarcity stats, EaaS-vs-
carbon table) reading nine hardcoded result-directory paths.

**Not new — already declared in `docs/runner_inventory.md:92-135,1569`,
re-confirmed with fresh evidence, not re-numbered as a new finding:**
every `CASES` path this script hardcodes points at pre-2.6/pre-O&M-era
output (`results/baseline/summary.json`'s own content has `"leakage":
null`, a field predating the 2.6 leakage guard; `results/ndc/ndc2_
unconditional_voll_mid/summary.json` shows `npv_total_cost_usd:
81,374,741,734.86` — roughly 3x the current central objective, with no
`om_case`/`arm_tag`/`gas_retirement_case` keys at all). The current
`01_run_baseline.py`/`02_run_ndc_caps.py` no longer write to these bare
paths at all (tag-suffixed and hash8-directory naming respectively) —
this script's own `results/phase8/master_results.csv` (genuinely
regenerated 2026-09-15) embeds these stale, ~3x-off values with no
staleness indicator. Flagged here only because Phase 1-2's own coverage
table lists this file as "not reached" — this is the first time it
enters `forensic_audit.md` itself, even though the underlying defect is
not new.

### A39 — A prior audit document's description of this script's failure mode no longer matches the current code

**Fingerprint:** F7 (docs contradicting code).

**Location:** `docs/runner_inventory.md:127-130` states the script
"does not error (`if not p.exists(): WARNING ... continue`)". Grepped
the CURRENT `scripts/08_phase8_analysis.py` for `exists`/`WARNING`/
`try`/`except`: zero hits. `load_summary`/`load_timeseries` (`:81-91`)
are plain `open()`/`pd.read_csv()` with no existence check and no
warn-and-skip fallback.

**Why it matters.** If any one of the nine hardcoded paths were removed
(e.g. a reasonable cleanup of the orphaned pre-2.6 directories A39's
predecessor note describes), the script would crash immediately with an
unhandled exception — not "silently degrade with warnings" as the
existing doc claims. The doc's framing understates the actual risk.

**Severity:** COSMETIC as a doc-accuracy issue; practically relevant
since it means "will limp along" is the wrong mental model for anyone
relying on `runner_inventory.md` to predict this script's behaviour.

**Declared?** No — this specific sub-claim (the exists-check) is not
addressed anywhere else found.

### A40 — A cross-runner field-name mismatch between `01_` and `02_`'s summary schemas produces a silent NaN in this script's own, currently-on-disk master table

**Fingerprint:** F2 (silent default on a load-bearing value), verified
directly against the real output file, not merely predicted from the
code.

**Location:** `scripts/01_run_baseline.py:359-360` writes the terminal
storage figure as `"final_storage_capacity_mwh"`. `scripts/02_run_ndc_
caps.py:370-371` writes the SAME underlying quantity under the
DIFFERENT key `"storage_capacity_mwh"`. `scripts/08_phase8_analysis.
py:139` hardcodes one name only: `dv.get("storage_capacity_mwh",
np.nan)`.

**Confirmed in the actual, currently-generated `results/phase8/master_
results.csv`** (dated 2026-09-15): the `baseline_no_policy` row's
`storage_capacity_mwh` cell is empty (NaN) — fed by `01_`'s schema,
which lacks that exact key — while all eight other rows (fed by `02_`'s
schema) correctly show a value.

**Severity:** LIVE in the narrow sense that it is present in the actual,
currently-generated output file today; not cited in `findings_frozen.
md` (checked), so not LIVE-REPORTED.

**Declared?** No — grepped `runner_inventory.md`, `limitations_
register.md`, `findings_frozen.md`, `merged_correction_plan_v2.md` for
both key names: zero hits.

**Cosmetic, not separately numbered:** `CANONICAL_GENSET = "genset_
central"` (`:21`) and `MMBTU_PER_TWH_TH` (`:75`) are both defined and
never used anywhere else in this file (F1 x2) — this script never calls
`load_econ`/`build_model`, so neither constant has a possible consumer;
apparent vestigial boilerplate.

---

#### `scripts/08_run_gas3_regime_ndc_feasibility.py` — Tier 1, STILL BARRED on its old headline figures (known, not re-reported)

**What it produces.** `results/gas3/regime_ndc_matrix.csv`, `regime_
feasibility.csv`, `annual_ndc_binding.csv`, `shape_cost_premium.csv`,
`gas3_summary.json` — matches its own docstring.

### A41 — This runner is dead on required arguments via the same L27 mechanism as the nine already-listed runners, but was never itself added to either dead-runner list, despite L27's own text naming GAS-3 as broken by it

**Fingerprint:** other — the L27/L10 defect CLASS is fully declared;
this specific FILE's membership in it is not, in either version of the
list.

**Location:** `run_gas_regime_ndc_matrix` (`src/optimize_experiments.
py`) currently requires 20 keyword parameters (confirmed via `inspect.
signature`, side-effect-free). `scripts/08_run_gas3_regime_ndc_
feasibility.py:261-268` and `:292-299` each supply only 3 of the 20 —
`TypeError` on the first call in `main()`, before any solve. No
`argparse` import anywhere in the file; no `cli_arms` usage.

**Why flagged as a gap rather than a duplicate of L27.** `docs/
limitations_register.md` L27 explicitly names `run_gas_regime_ndc_
matrix` as serving "GAS-1, GAS-2, AND GAS-3" and its own corrected
dead-runner list (extending L10's original nine with `10, 24, 25, 99c`)
does not add `08_run_gas3_regime_ndc_feasibility.py`, even in the same
paragraph that names GAS-3 as broken by this exact mechanism. The
underlying defect class is fully declared (L27); this file's own name
was never entered into either enumeration of it.

**Consequence, confirmed against real output.** `results/gas3/*` is
entirely dated 2026-08-23 13:44 — the same pre-2.6/pre-O&M epoch as the
stale `results/baseline/`/`results/ndc/*_voll_mid` directories A39/A40
discuss. The existing citation bar on this script's old headline
figures ($571B/152.4TWh, "pre-recalibration") is accurate but
understates the actual state: this script cannot regenerate ANY number
today, old or current, not merely "these particular old numbers are
barred."

**Severity:** TRAP.

**Declared?** The defect CLASS: yes (L27). This file's specific
membership: no.

### A42 — A hardcoded gas anchor constant carries the pre-recalibration value (40.66), not the current, live 89.27 TWh_th

**Fingerprint:** F6 (epoch-stale hardcoded figure) — **personally
verified** against the frozen anchor list and the live YAML.

**Location:** `scripts/08_run_gas3_regime_ndc_feasibility.py:120-121`:
```python
# Gas anchor (2025 baseline value from gas_deliverability_cases.yaml)
GAS_ANCHOR_TWH_TH = 40.66
```
Current `data/gas/processed/gas_deliverability_cases.yaml`'s own anchor
block: `gas_available_twh_th: 89.27`, with its own comment stating
*"Previous value 40.66 was calibrated against the old eta=0.43... it
understated delivery by ~2.2x."* — i.e. the YAML explicitly names 40.66
as the value THIS SCRIPT still hardcodes as its own, separate constant.
Confirmed live in the actual 2025 rows of `data/gas/processed/gas_
available_power_annual_twh_th.csv`: every scenario reads `89.27` for
2025, not 40.66.

**Why it matters — actively used, not decorative.**
`GAS_ANCHOR_TWH_TH` gates `"gas_disrupted": yr_row["gas_avail_twh_th"] <
GAS_ANCHOR_TWH_TH` (`:158`) and `"cap_and_gas_disruption_coincide"`
(`:160-164`) — the script's own comment calls the latter *"Key for
GAS-3 thesis: does cap bind IN THE SAME YEAR as disruption?"*. Pinned at
under half the current anchor, this threshold would essentially never
fire correctly against current-scale data (now clustering 60-120 TWh_th,
not 30-50) — the GAS-3 disruption-coincidence diagnostic would be
silently miscalibrated even after A41's required-args repair is applied.
A second, independent reason (distinct from the barred headline-cost
figures) this script's diagnostics cannot be trusted as-is today.

**Severity:** TRAP.

**Declared?** The general 40.66→89.27 migration is `[DONE]`
(`merged_correction_plan_v2.md:97`); this file's own straggling copy was
never caught by that migration and is not declared anywhere.

### A43 — The module's own "robustness gaps fixed" docstring quotes cumulative gas totals stale by roughly 3x, from the same pre-89.27-anchor epoch as A42

**Fingerprint:** F7 (docs/comments contradicting code, within the same
file).

**Location:** `scripts/08_run_gas3_regime_ndc_feasibility.py:16-19,29-
31` (docstring) claims cumulative totals (baseline 853.9 TWh_th,
downside 640.4, upside 959.6, shock_recovery 914.0; flat_downside 30.50
TWh_th/yr, flat_upside 45.69, flat_shock_recovery 43.52) that do not
match the current `gas_available_power_annual_twh_th.csv` (summed
directly: baseline 2,881.6, downside 1,886.0, upside 3,486.1, shock_
recovery 2,716.4; flat means 70.76/105.6/89.18) — all six figures off by
roughly 2.3-3.4x, consistent with the same η/demand-base recalibration
behind A42.

**Why it matters.** This docstring is the script's own stated
justification for its central "shape vs. level confound" design (the
`flat_*` control arms) — an examiner or thesis author could reasonably
lift this scoping text directly, arguing from numbers that no longer
describe the data the script would run against.

**Severity:** TRAP — docstring-only, does not drive computation.

**Declared?** No — distinct from `merged_correction_plan_v2.md:516`'s
different GAS-3 caveat (flat-level averaging window).

**Builder choice, checked as instructed:** `run_gas_regime_ndc_matrix`
uses `build_model` (annual) throughout — appropriate for GAS-3's
year-by-year diagnostic; not an F11 defect.

---

#### `scripts/09_run_rel1_feasibility.py` — Tier 1, dead on required args (known, not re-reported)

**No new findings.** Its only notable property — `run_rel1_feasibility_
matrix`'s two `build_model()` calls still pass the deprecated
`reliability_mode="total"` rather than `"annual"` — is already, and
explicitly, declared as a deliberate, un-closed gap in `docs/task_
shared_fn_repair.md` ("`run_rel1_feasibility_matrix`'s `reliability_
mode="total"` call is unchanged and remains stale... a real, live defect
this repair deliberately did not close") and an inline comment at `src/
optimize_experiments.py:3430-3434`. Not re-reported; noted here only as
a caveat worth carrying forward — even once wired, REL-1's own eps sweep
runs on the stale, non-discriminating mode, unlike REL-2/3 (below).
Console/JSON logic, solve counts, and constants otherwise internally
consistent.

---

#### `scripts/10_run_rel2_marginal_cost.py` — Tier 1, dead on required args (known, not re-reported)

**No findings.** Delegates to `run_reliability_sweep`, which correctly
uses `grid_service_mode="annual"` (the D6 fix IS live on this path,
unlike 09_ above). Local helpers (`compute_curve_summary`, `compute_
left_shift`) reviewed line by line for sign/direction errors — all
consistent with their stated definitions. No stale constants, no silent
defaults.

---

#### `scripts/11_run_rel3_financing_frontier.py` — Tier 1, dead on required args (known, not re-reported)

**Builder correctly `"annual"` mode via `run_reliability_sweep`, same as
10_.**

### A44 — Docstring's own solve-count arithmetic predates two NDC arms added later to the same file

**Fingerprint:** F7/F6.

**Location:** `scripts/11_run_rel3_financing_frontier.py:82-90`
docstring: *"policy_arms: 3 levels... Total solves: 4 x 3 x 2 x 7 =
168."* Actual `POLICY_CONFIGS` (`:155-196`) has 5 entries (`no_policy,
ndc2_unconditional, ndc2_conditional, ndc3_unconditional, ndc3_
conditional`), matching the runtime computation at `:207`: `n_solves =
len(GAS_CASES) * len(POLICY_CONFIGS) * 2 * (1 + len(EPS_LEVELS))` =
4x5x2x7 = **280**. Docstring never updated when the ndc2 arms were
added.

**Severity:** DORMANT/COSMETIC (script is dead; nothing has ever been
produced).

**Declared?** No.

### A45 — The retired `required_margin` double-count pattern reappears in this runner's own hardcoded config, unlike its already-fixed siblings

**Fingerprint:** F8 (the `dscr_case`-derived value accepted then
discarded) with F4 character (two `required_margin` sources not kept in
sync).

**Location:** `POLICY_CONFIGS` hand-types `"eaas_margin": 1.10`/`1.05`
per arm (`:161,169,177,185,193`), threaded via `src/optimize_
experiments.py:1013` (`scenario_eaas["required_margin"] = eaas_
margin`) — OVERWRITING the `required_margin` `load_scenario()` already
set correctly from the required `dscr_case` arm. `dscr_scenarios()`
(`src/scenarios.py:1790-1804`) itself states: *"the pre-F1 1.05/1.10
split is RETIRED as a double-count now that the concessional/commercial
difference is carried explicitly by the CRF rates."* `02_run_eaas.py`,
`03_run_ndc_eaas.py`, and `04_run_fin2_tariff_sweep.py` all explicitly
retired this exact override ("F1, Decision 2" comments in each). 11_'s
`POLICY_CONFIGS` values were never updated to match.

**Severity:** DORMANT (script is dead on required args today; would be
LIVE-LATENT the instant it is wired, per the L27 repair pattern).

**Declared?** The mechanism (the overwrite line in `optimize_
experiments.py`) is Phase 1-2 `src/` territory; the specific VALUES
hand-typed in 11_'s own config are not flagged anywhere found — reported
as new at the script-config level. Contrast: 11_'s capital-envelope tier
assignments (`unconstrained`/`tight`/`moderate`) DO match the canonical
02_/03_ convention — no envelope defect here, unlike 15_ below.

---

#### `scripts/15_run_pol1_ndc_comparison.py` — Tier 1, LIVE (cli_arms-repaired, `ecb7e54`)

**Live, not dead — raises the stakes on every finding below.** Not
currently cited for any live number (only the already-barred, pre-
`2a9226b` `public_only` −$220.75B figure is on record, per the
DO-NOT-REPORT list — not re-reported).

### A46 — The retired `required_margin` double-count is live in this runner, and — unlike A45's dormant instance — this script actually executes

**Fingerprint:** F8/F4, same mechanism as A45, confirmed LIVE.

**Location — personally verified.** `scripts/15_run_pol1_ndc_
comparison.py:217-218`:
```python
if fin_label == "eaas":
    scenario["required_margin"] = ndc_cfg["eaas_margin"]
```
`NDC_CONFIGS` (`:107-141`) sets `eaas_margin` to 1.10 (unconditional
arms) / 1.05 (conditional arms) — the exact retired split `dscr_
scenarios()` (`src/scenarios.py:1790-1804`) names as a double-count now
that `capital_recovery_case`'s CRF rates carry that distinction. 15_ was
repaired for its CLI-arm wiring (`ecb7e54`) but never for this.

**Why it matters.** No number is currently frozen from POL-1's EaaS arm,
but this script IS live and executable — running it today biases every
EaaS solve's financing cost by a double-counted margin, distorting the
"does EaaS rescue NDC 3.0?" question this script exists to answer.

**Severity:** LIVE-LATENT.

**Declared?** No — `docs/required_margin_inventory.md`'s existing entry
for 15_ predates both the `dscr_case` retirement and the `ecb7e54`
repair; it records what value was set, not that the value now conflicts
with the canonical mechanism.

### A47 — This runner's capital-envelope tiers are one full tier looser than the canonical convention every sibling script uses — and the discrepancy traces to a stale comment sitting beside the correct code in one of those siblings

**Fingerprint:** F5 (duplicate source of truth, diverged), compounded by
an F7 instance (a comment contradicting the code immediately below it)
found while verifying this — **personally verified in full, extended
beyond the original report.**

**Location.** `scripts/15_run_pol1_ndc_comparison.py:107-141`
(`NDC_CONFIGS`) maps `unconditional -> "moderate"`, `conditional ->
"expansion"` for both `public_capital` and `eaas_capital`, at every NDC
version. The canonical convention — confirmed live in `scripts/02_run_
ndc_caps.py:420-425` (`NDC_CASES`), mirrored in `03_run_ndc_eaas.py` and
`11_run_rel3_financing_frontier.py` — is `unconditional -> "tight"`,
`conditional -> "moderate"`, one tier tighter at both levels.

**The stale comment, found while checking the discrepancy's origin.**
`scripts/02_run_ndc_caps.py:406-419`, the comment immediately above its
own `NDC_CASES` dict, still reads: *"unconditional: 'moderate' (0.85 x
B* = 5.2B USD)... conditional: 'expansion' (1.20 x B* = 7.4B USD)"* —
this is NOT what the dict two lines below it actually contains
(`"tight"`/`"moderate"`). 02_'s own in-file documentation is stale
relative to its own code, and 15_'s hardcoded `NDC_CONFIGS` matches this
STALE COMMENT's values exactly, not 02_'s actual, current dict — the
most likely mechanism by which the two files diverged.

**From `capital_envelope_scenarios()` (`src/scenarios.py:783-801`,
B*=$9.104B):** tight=$4.552B, moderate=$7.7384B, expansion=$10.9248B.
15_ therefore gives its "unconditional" arms ~70% more public capital,
and its "conditional" arms ~41% more, than every sibling script's
same-named arm.

**Why it matters.** POL-1's research question is precisely "does public
capital alone become infeasible under NDC 3.0, and does EaaS rescue
it?" — a systematically looser envelope than the project's own
established convention biases that answer toward feasibility, and any
comparison against 02_/03_'s "matched ambition" arms is not actually
matched.

**Severity:** LIVE-LATENT.

**Declared?** No — not in `docs/runner_inventory.md`'s (pre-repair)
entry for 15_, nor the stale comment in 02_ itself (which is its own,
separate, small F7 instance, folded in here rather than given a
fifteenth A-number, since it is the explanatory mechanism for A47 rather
than an independent risk to a published number).

### A48 — A stale B* figure in this runner's own docstring, second independent occurrence of an already-flagged number

**Fingerprint:** F6.

**Location:** `scripts/15_run_pol1_ndc_comparison.py:38`: *"the PUBLIC
CAPITAL budget (moderate = 5.21B USD)"* — computed from the RETIRED
B*=$6.13B (`0.85 x 6.13e9 = 5.21bn`), not the current, coded $9.104B
(`src/scenarios.py:789-791` already self-flags this exact figure as
"STALE-DOC" for its OWN docstring). 15_ is a second, independent
propagation of the same superseded number.

**Severity:** COSMETIC — descriptive text only, never read by code.

**Declared?** The underlying stale B* is declared (`scenarios.py:789-
791`); this specific second occurrence, in 15_, is not.

### A49 — The A21 two-resolution shape recurs in this runner, currently a trap for the same reason it was in `02_`/`03_` before their fix

**Fingerprint:** F4, confirmed the same shape as the already-fixed A21.

**Location:** `run_one()` builds with an explicit series (`solar_capex_
tv = load_solar_capex_by_year(scenario_name="solar_low", ...)`, `:227-
231`, threaded into `build_model`), then calls `extract_planning_
diagnostics(m, scenario, econ)` at `:257` WITHOUT `solar_capex_
series=` — the exact pattern A21 fixed in `02_`/`03_` by passing the
series through explicitly. `load_scenario()` hardcodes `scenario[
"solar_capex_scenario"] = "solar_low"` unconditionally (`scenarios.py:
2940`), so today both resolutions land on `"solar_low"` and agree
numerically — currently a TRAP, not a live divergence, exactly as A21
was before its fix.

**Severity:** TRAP.

**Declared?** No — 15_ was never on A21's fix whitelist; this is a
separate, later-discovered instance of the same underlying pattern.

---

#### `scripts/16_plot_cost_decomposition.py` — plotting script, "will produce thesis figures"

### A50 — This plotting script reads a pre-O&M-fix schema of `cost_decomposition`, silently omitting a component register L28 specifically added visibility for

**Fingerprint:** F1 (defined-never-read: `om_npv` exists in every
current `diagnostics.json` but is never read here) with an F7 flavour
(L28's "RESOLVED" status describes the PRODUCER fix; this CONSUMER was
never updated to match) — **personally verified in full.**

**Location:** `scripts/16_plot_cost_decomposition.py:102-110`:
```python
real_exp = decomp["real_expenditure_npv"]
voll_pen = decomp["voll_penalty_npv"]
selfgen = decomp.get("selfgen_npv", diag.get("cost_selfgen_npv", 0.0)) or 0.0
salvage = decomp.get("salvage_npv", 0.0) or 0.0
total = real_exp + selfgen + voll_pen - salvage
```
— verbatim the PRE-L28-fix formula. Register L28 records this exact
`total` computation as 16_'s OWN reconciliation guard, the mechanism
that originally caught the O&M omission, and its status as RESOLVED.
The resolution (confirmed directly, `src/optimize_experiments.py:3000-
3009`) added `om_npv` as `cost_decomposition`'s own top-level field and
folded `cost_om` into `cost_total`. **16_.py itself was never edited**
— confirmed via `git log`: its only commit, `4faa2b6`, is dated
2026-08-21; the commit that added `"om_npv"` to `cost_decomposition`,
`f4fb607`, is dated 2026-09-16 — nearly a month later — and no
subsequent commit touches `scripts/16_plot_cost_decomposition.py`.

**Consequences if this script is run today.** (a) The stacked bar chart
has no O&M segment — every case's plotted total understates the true
NPV cost by the O&M share (documented elsewhere as +6.602bn/+30.9% at
the central case), with no visual cue a segment is missing. (b) The
per-bar percentage labels (`:154-157`) misstate the public/private split
by the same amount, since O&M (a public, real cost) is excluded from
both numerator and denominator. (c) The script's own reconciliation
guard (`:191-208`), which compares `total` against `m.system_cost_npv`
(which DOES include O&M unconditionally), would print a roughly 30%
mismatch for every case with nonzero O&M and its own closing message,
*"RECONCILIATION FAILED... do not read shares off this figure until it
closes"* — i.e. the script would self-flag as broken in console output
even while silently producing a wrong PNG/PDF.

**Why it matters.** Exactly the risk this task's own framing names for
plotting scripts, in its sharper form: not a stale field silently
producing a plausible-looking wrong figure, but a stale field producing
a wrong figure AND a correctly-firing internal alarm that a future user
could all too easily read past.

**Severity:** LIVE-LATENT — would mis-render the moment this script is
next run for the thesis; not confirmed cited in `findings_frozen.md`
(checked, no "phase8"/"cost_decomposition"-figure citation found), so
not LIVE-REPORTED today.

**Declared?** No — L28 documents the producer-side fix and its own
discovery mechanism (this exact script's guard) but nowhere claims 16_
itself was updated; confirmed absent via git history.

**Context, not separately findings-numbered:** `docs/runner_inventory.
md`'s existing entry for `02_run_ndc_caps.py` already documents that
`08_phase8_analysis.py` and `16_plot_cost_decomposition.py` expect
output at flat paths `02_`/`03_` no longer write to (hash8-directory
naming) — if still true, 16_'s own `CASES` loop would skip 4 of its 5
entries before ever reaching A50's defect, leaving only the `baseline`
bar to demonstrate it. Not re-reported; noted because it means A50's own
reconciliation-mismatch warning may currently fire only once (for
`baseline`) rather than five times — still enough to catch it, but the
two defects compound.

---

#### `scripts/04_run_fin2_tariff_sweep.py` — Tier 1, nominally suspended per D4, but see A51

**What it produces.** `results/fin2/tariff_sweep_results__{tag}.csv/
.json`, `tariff_sweep_summary__{tag}.json`. Nothing else in the repo
reads these paths.

### A51 — `04_` carries no suspension guard at all and has been actively repaired; two documents describing it as a dead stub are now stale

**Fingerprint:** F7 (docs contradicting code) — **personally verified.**

**Location:** `docs/merged_correction_plan_v2.md:712` (D25): *"`04_run_
fin2_tariff_sweep` and `12_run_dem1_demand_sensitivity` raise
NotImplementedError."* `docs/forensic_audit.md:375` (this document's own
Phase 1 text, A10): *"dead runner `04_run_fin2_tariff_sweep.py`."*

**What is wrong, verified directly.** `grep -n "NotImplementedError|
SUSPENDED|DEPRECATED"` over the whole 522-line file: zero hits.
`main()` fully builds scenarios, calls `run_fin2_blended_bankability_
sweep`, and writes real output files — unlike `12_run_dem1_demand_
sensitivity.py`, whose `main()` genuinely opens with `raise
NotImplementedError(...)` (confirmed, `:313`) and is correctly described
by the same D25 entry. `git log`: `04_` was built out in `b496d77`
(2026-08-23, "FIN-2: multi-blend/multi-NDC tariff sweep"), had its
`required_margin` double-count retired in `f9672b8`/comments at
`:298-303`, and was one of the "nine directly-broken runners" actively
REPAIRED for its CLI-arm surface in `ecb7e54` (2026-09-15) — the same
commit that repaired `05_/15_/18_/19_/20_` (A23's five). D25's
"NotImplementedError" claim has not matched this file since before this
session began; Phase 1's "dead runner" label is now equally stale.

**Why it matters.** A reader trusting D25 or Phase 1's own earlier text
would conclude 04_ cannot run at all; a reader trusting the file's
silence and the real, dated output already sitting in `results/fin2/`
would conclude it runs cleanly today. Neither is quite right — see A52
below for what's actually still wrong with it.

**Severity:** TRAP (no published number depends on `results/fin2/*`,
per `findings_frozen.md` — checked, no citation found).

**Declared?** No — this is a documentation-currency gap in the same
family as A23 (register L10 stale for five repaired runners), here
affecting D25 and this document's own earlier Phase 1 text, for a
runner outside L10's original ten.

### A52 — `--concessional-envelope`/`--deployable-fraction` still compute and thread a value into a retired builder mechanism, so two diagnostic columns are unconditionally null on any fresh run

**Fingerprint:** F8 (parameter accepted, computed, and threaded three
call-layers deep, then discarded).

**Location:** `scripts/04_run_fin2_tariff_sweep.py:336-344` computes
`envelope_npv` and passes it into `run_fin2_blended_bankability_
sweep(..., concessional_envelope_npv=envelope_npv, ...)`. `src/
optimize_model.py:726-736` states this mechanism (`m.concessional_
envelope_constraint`, `m.concessional_drawdown_npv`) is *"RETIRED per
envelope resolution Decision 2"* (`588a734`, 2026-08-28), superseded by
the separate, F1-owned `concessional_envelope_case` mechanism.
`extract_planning_diagnostics`'s own guard (`optimize_experiments.py:
3070`, `if hasattr(m, "concessional_envelope_constraint"):`) can never
be true for a model built by the current builders — so `concessional_
shadow_usd_per_usd`/`concessional_utilisation` are unconditionally
`None` for any fresh run of this script, regardless of the `--
concessional-envelope` arm chosen.

**Why it matters.** 04_'s own comment (`:149-157`) already concedes the
flag now "feed[s] an inert scenario key" — the retirement itself is
acknowledged locally — but nowhere states the concrete, checkable
consequence: two named diagnostic columns are now always null.

**Severity:** LIVE-LATENT (fires on every fresh run; no currently-cited
number depends on it).

**Declared?** Partially — the retirement is known locally in-file; the
specific always-null consequence is not stated anywhere.

### A53 — Stale, pre-retirement output files sit on disk with real values from a mechanism the current code can no longer produce

**Fingerprint:** other (an unlabeled artifact from a superseded code
path, indistinguishable on its face from current output).

**Location:** `results/fin2/tariff_sweep_results__dares_only__df1.csv`
(mtime 2026-08-23 13:44) predates the envelope-retirement commit
`588a734` (2026-08-28) by five days. Confirmed directly: its
`concessional_utilisation` column carries 150 non-null real values
(e.g. `1.0` for `blended_central`) — computed by the pre-retirement code
path per A52. Re-running `04_` today would silently produce all-null
values for the same column, with no version marker distinguishing old
from new in either file.

**Severity:** TRAP — `findings_frozen.md` cites nothing from `results/
fin2/`, so nothing published is affected; a future reader treating this
directory as current output would be misled.

**Declared?** No.

**No other findings** — F1-era CLI arms are all `required=True` with no
defaults, `genset_case` passed explicitly, consistent with project
discipline.

---

#### `scripts/12_run_dem1_demand_sensitivity.py` — Tier 1, correctly suspended (D4), guard confirmed genuine

**Confirmed genuinely dead, unlike 04_ above.** `main()`'s first
executable line is `raise NotImplementedError(...)` (`:313`), and
independently, `run_dem1_demand_sensitivity`'s call (`:325-330`) is
missing 16 of the same required `load_scenario()`-family keys every
other pre-F1 runner is missing — verified via `inspect.signature(...).
bind(...)` (side-effect-free): `TypeError: missing... 'peak_multiple_
case'`. The suspension notice's own content (η=0.287, demand base 37.09
TWh) matches the frozen anchors.

### A54 — The file's detailed rationale text is a second, orphaned string literal, never bound to `__doc__`

**Fingerprint:** other (dead/orphaned documentation text).

**Location:** `ast.get_docstring()` confirms the LIVE `__doc__` is only
lines 1-12 (the short "[SUSPENDED]" banner). Lines 14-101 — the entire
"RESEARCH QUESTION"/"WHY THE CHOICE OF DEMAND BASE MATTERS"/"EXPERIMENT
STRUCTURE"/"OUTPUTS" sections, including the 23.08/38.47/76.93 TWh
figures — form a second, separate string-literal expression statement,
never assigned to `__doc__`: readable by anyone opening the file, but
invisible to `help()`/pydoc or anything introspecting `__doc__`
programmatically.

**Severity:** COSMETIC — the file is fully readable by opening it, and
the substantive deprecation reason survives in the live, bound
docstring regardless.

**Declared?** No, low-stakes.

**Minor, not separately numbered:** `from src.scenarios import tariff_
scenarios` imported, never used (same in `13_`, below).

**No hardcoded frozen-anchor conflicts beyond the already-self-declared,
already-retired 23.08 TWh demand base** (a different figure from A19's
25.62 TWh gas backcast — this one is the demand-base retirement, already
stated as retired in this file's own live docstring, not newly reported
here).

---

#### `scripts/13_run_dem2_growth_gas_interaction.py` — Tier 1, described as ACTIVE, but see A55

**No suspension guard, no `NotImplementedError`** — the file presents
itself as live, unlike `12_`/`14_`.

### A55 — The "active" DEM-2 script cannot execute at all: its only call to its backend function is missing 16 required keyword arguments

**Fingerprint:** other (a call site broken on the same required-args
class as the L10/L27 dead runners, but never itself entered on that
list) — **personally verified.**

**Location:** `scripts/13_run_dem2_growth_gas_interaction.py:393-400`:
```python
rows = run_dem2_growth_gas_matrix(
    econ=econ, demand_cases=DEMAND_CASES, gas_cases=GAS_CASES,
    financing_configs=FINANCING_CONFIGS, policy_configs=POLICY_CONFIGS,
    cap_path=str(CAP_PATH),
)
```
`run_dem2_growth_gas_matrix` (`src/optimize_experiments.py:1688-1711`)
requires 16 more keyword arguments with no defaults (`peak_multiple_
case, reserve_margin, storage_cc_case, gas_build_case, gas_
availability_case, gas_retirement_case, gas_capex_case, ccgt_
efficiency_case, capital_recovery_case, dscr_case, us_inflation,
capital_treatment_case, concessional_envelope_case, conversion_case,
hydro_case, om_case`). Confirmed via pure signature binding (`inspect.
signature(...).bind(...)`, no execution): `TypeError: missing a
required argument: 'peak_multiple_case'`. `13_` has no `argparse`
anywhere — no path exists to supply these. `main()` raises `TypeError`
immediately, before building or solving anything.

**Why it matters.** TASK.md's own known-set treats 13_ as ACTIVE, with
only D1's interaction-classifier defect open. This is a much more basic,
blocking defect — the script cannot produce a single row of output
today, a fact D1 (which concerns the CONTENT of rows that would be
produced) does not capture.

**Severity:** TRAP.

**Declared?** Partially, outside the set given for this task: `docs/
task_shared_fn_repair.md:280-285` already recorded, in an earlier repair
task, that `run_dem2_growth_gas_matrix`'s only caller (13_) is "not a
healthy runner" and was deliberately left unrepaired when the function's
OWN signature was hardened. Neither `merged_correction_plan_v2.md`'s
D-list nor the register currently tracks 13_'s specific dead-on-args
status the way L10/L27 track the other nine-plus-four.

### A56 — Even once A55 is naively patched, the retired `required_margin` double-count resurfaces inside 13_'s own backend function

**Fingerprint:** F8/F4, the same mechanism as A45/A46.

**Location:** `src/optimize_experiments.py:1840-1842`, inside `run_
dem2_growth_gas_matrix`, immediately after `load_scenario()` already
sets `scenario["required_margin"]` from the required `dscr_case` arm:
```python
scenario["financing_regime"] = fin_cfg["financing_regime"]
if fin_cfg["financing_regime"] == "eaas":
    scenario["required_margin"] = fin_cfg["required_margin"]
```
`13_`'s own `FINANCING_CONFIGS` (`:128-134`) supplies `"required_
margin": 1.10` for the `eaas` arm — overwriting the correctly `dscr_
case`-sourced value, exactly the pattern `04_run_fin2_tariff_sweep.py`'s
own comments (`:298-303`) describe as retired. `04_` was fixed to stop
doing this (confirmed above, A51); `run_dem2_growth_gas_matrix`, 13_'s
only backend, was not.

**Why it matters.** The obvious, shallow fix for A55 (thread the 16
missing args through, mirroring 04_'s CLI treatment) would make the
script RUN while silently reintroducing the exact double-count "F1,
Decision 2" eliminated everywhere else — a trap for whoever repairs A55
next. The identical pattern also exists in `run_str1_storage_parameter_
sweep` (`optimize_experiments.py:2047-2049`), already confirmed dead
code (A17/L5) — not separately reported.

**Severity:** TRAP.

**Declared?** No.

**Confirmed clean, not re-reported:** D1's interaction-classifier defect
is FIXED in this file (`:238-302`, a genuine three-branch, relative-
tolerance classifier with an explicit sub-additive branch — the fixed
state, not the old bug). The "48 vs 72" DO-NOT-REPORT concern is clean
(`n_solves = 3x4x2x3 = 72`, `:380-383`, matching the docstring; no "48"
anywhere). No hardcoded frozen-anchor mismatches found.

---

#### `scripts/14_run_str1_storage_role.py` — Tier 3-shaped (suspended per D4/L5), 55 lines, read in full

**Suspension notice itself is accurate** — the algebra (`discharge -
charge <= (RTE-1)*charge <= 0`, so `storage_add=0` is always optimal at
annual resolution) is correct and matches L5.

### A57 — The file's own "re-enable when" precondition has already been met elsewhere in the repo, and gives no indication of this

**Fingerprint:** other (stale forward-reference).

**Location:** `scripts/14_run_str1_storage_role.py:33-39`: *"RE-ENABLE
WHEN — Phase 2 introduces representative intra-annual slices... storage
then shifts day->evening within a season and acquires a genuine
physical role."* `src/optimize_model_sliced.py` — self-labelled "Phase
2.1" in its own header — already exists, is mature, and is imported by
roughly a dozen live scripts. It implements exactly the mechanism 14_
says doesn't exist yet: `m.storage_season_balance` lets storage charge
from midday solar and discharge into the same season's evening peak.
(Minor: the actually-built structure is 6 slices, `{dry,wet} x
{night,day,peak}`, not the 4-slice `{wet,dry} x {day,evening-peak}` this
file's docstring anticipated.)

**Why it matters.** A future reviver reading only this file — exactly
what its own suspension notice invites — has no way to know `build_
model_sliced` already exists and could, in principle, be used to
rebuild STR-1. Whether storage actually wins economically against the
genset backstop under slices (`optimize_model_sliced.py:632-636` notes
the backstop "caps storage's marginal value") is a separate, open
empirical question the structural claim this file makes no longer
gates.

**Severity:** TRAP.

**Declared?** No — this document's own Phase 1 mention of STR-1 (A17)
is about the dead `run_str1_storage_parameter_sweep` function and
reconfirms L5's suspension is expected; it does not address this file's
own staleness.

**No other findings** — the file is otherwise exactly what it says it
is.

---

#### `scripts/21_compute_resolution_premium.py` — methods/verification harness, "a METHODS FINDING, not a scenario result"

**Correct builder pairing, confirmed (task's specific ask):** `01_`
imports `build_model`; `01b_` imports `build_model_sliced`; `21_` itself
never calls either builder — it only compares two pre-written JSON
files. **No hardcoded 13.4%/6.80% treated as current** — both mentions
explicitly and correctly frame the figure as superseded, matching D8.
**The staleness guard (its own Check 4) currently works as intended,
confirming D8 rather than contradicting it** — `results/baseline{,_
sliced}/summary.json` (mtime 2026-08-23) predate `src/optimize_model.py`
(2026-09-07)/`optimize_model_sliced.py` (2026-09-17), so a run today
correctly trips `"reportable": False`.

### A58 — The script's own "same scenario configuration" safety check is a structural no-op against the actual output schema — it has never once fired, staleness or not

**Fingerprint:** other, closest in spirit to F2 (a declared safety check
that cannot act on the data it is given).

**Location:** `scripts/21_compute_resolution_premium.py:15-21` promises:
*"This script therefore refuses to report unless it can verify: 1. Both
runs used the same demand arm, gas regime and VoLL case..."* Check 3
(`:158-166`) tests five field names — `demand_case`, `demand_growth_
rate`, `gas_case`, `gas_scenario`, `voll_case` — only flagging a blocker
`if av is not None and sv is not None and av != sv`. Verified directly
against the current files: **none of the five checked fields exists in
either `results/baseline/summary.json` or `results/baseline_sliced/
summary.json`** (their actual keys: `actual_emissions_tco2_total,
cap_scenario, cumulative_unserved_twh, decision_variables, leakage,
model_horizon_end_year, notes, npv_gross_capex_full_horizon_usd, npv_
gross_capex_report_window_usd, npv_total_cost_usd, report_window_end_
year, solar_public_npv_spend` for annual; a similar, shorter set for
sliced). `av`/`sv` are `None` for all five on both sides, so the `and`
chain short-circuits every time and the check silently never fires — not
even a warning.

**Why it matters.** Distinct from D8 (which is about the PREMIUM being
stale, not about this specific check being vacuous). Even a freshly
re-run, non-stale pair of files would sail through Check 3 regardless of
an actual scenario mismatch, since neither `summary.json` writer emits
these field names — the safety property the docstring promises does not
currently exist in any executable form.

**Severity:** LIVE-LATENT/TRAP — inert today only because Check 4's
staleness gate already blocks reporting; would matter the moment fresh,
non-stale files are supplied.

**Declared?** No.

### A59 — A hardcoded "verified" salvage-credit figure sits in a warning string with no version tie, in a file where an adjacent superseded figure (D8) is a known hazard class

**Fingerprint:** F6-shaped, embedded in a message string rather than a
computation.

**Location:** `scripts/21_compute_resolution_premium.py:116-121`:
```python
warnings.append(
    "Sliced diagnostics carry no salvage key... Salvage IS in the sliced "
    "objective (verified: 18.31 -> 14.286 bn), so this is a reporting "
    "gap, not a basis mismatch..."
)
```
Confirmed this warning fires on every run (`results/baseline_sliced/
diagnostics.json` has zero `cost`/`salvage`-related keys). The
"18.31 -> 14.286 bn" figures carry no commit or date stamp, in a file
whose own D8 entry warns that the horizon extension (`74d34f2`) moved
adjacent salvage-dependent figures. A distinct pair of numbers from D8's
13.4%/6.80% premium, not the same already-declared item.

**Severity:** COSMETIC-to-TRAP — misleading text only; does not affect
the numeric `premium`/`reportable` output.

**Declared?** No.

---

#### `scripts/99_validate_baseline.py` — manual validation utility, not wired into `smoke_all.sh`

**No new findings.** A19's 25.62 TWh duplicate (line 52) confirmed
present, correctly already declared, not re-reported. What it validates
(`demand_twh`, `gas_generation_twh`, `solar_generation_twh` — NOT
hydro or gas deliverability) is already explicitly documented as
intentional in `docs/task_wrong_output_repair.md:340-352` — not a new
gap. Epoch stamp ("current as of HEAD `2a9226b`") verified accurate:
`git log 2a9226b..HEAD -- src/scenarios.py` returns no commits.

**One observation, not asserted as a defect** (read-only — no model
re-run to determine which figure is authoritative): running this file's
own arithmetic against the currently-committed `results/baseline/
timeseries.csv` (mtime 2026-08-23, `gas_generation_twh_e = 24.744`)
gives a backcast gap of −3.42%, not exactly the −3.2% figure treated as
current central elsewhere (`docs/task_wrong_output_repair.md:331,346`).
Both are within the "within 4%" tolerance originally claimed; the 0.2pp
difference is plausibly explained by `results/baseline/` being a
snapshot older than some later commits (the same pattern A39/A40's
stale-directory findings show elsewhere) rather than a code defect in
this file. Flagged for the record, not counted as a finding.

---

### TIER 3 — `scripts/deferred/` and `scripts/retired/`, re-checked briefly

Phase 2 already read all three headers in full and found them "a model
of what A15/A18's 'would mislead a future reviver' risk looks like when
done right." This pass re-checked only whether A23's finding (five of
L10's nine named scripts are now repaired) changes that verdict for any
of these three, since all three headers cite L10/L5/L8 by name.

**`scripts/retired/09b_run_rel1_mode_sensitivity.py`, `scripts/retired/
09b_run_rel1_retirement_sensitivity.py`, `scripts/deferred/17_run_
monte_carlo.py`: verdict unchanged, re-confirmed, no new findings.**
None of these three is among A23's five repaired scripts (`05, 15, 18,
19, 20`) — all three remain exactly as dead as their own headers say,
and their citations of L10/L5/L8 remain accurate for themselves
specifically. `09b` (both) and `17_` have in fact moved directory
(`scripts/retired/`, `scripts/deferred/` respectively, per commit
`efa433e`) since L10's prose was written, which is a path-staleness
issue in L10's own "What" text (already noted as encountered-but-out-
of-scope in a prior task's report) — not re-derived as a fresh finding
here, and not attributable to these three files' own headers, which
describe their CONTENT accurately regardless of which directory they
sit in.

---

### A60 — Storage CAPEX omitted from realised-spend diagnostic while charged by the constraint

**Fingerprint:** F4 (two representations of one quantity that must
move together — the public capital the envelope charges, computed once
in `public_budget_constraint` and again in the `realised_public_spend`
diagnostic — kept in sync only by two expressions agreeing, not by
construction). Chosen because the file's other classes do not fit: F5 is
a duplicated literal, F7 is prose-vs-code. Here two code expressions of
one quantity diverged.

**Location:** `src/optimize_experiments.py` (`realised_public_spend`,
`extract_planning_diagnostics`, ~:2788-2794) vs `public_budget_constraint`
in `src/optimize_model.py` and `src/optimize_model_sliced.py`.

**Severity:** LATENT — fixed before the only runner that would have
exposed it (`28_`) was first run. Fix committed at `5beaef0`; entry
landed in `2f11f3f`. (Nearest term in this file's own vocabulary:
LIVE-LATENT.) See L46 for the row count.

**Declared?** Yes — L46.

---

### A61 — The objective's commercial rate (0.11) and the bankability
blend's commercial rate (0.18) are the same conceptual quantity, priced
below its own sourced floor, with no reconciliation

**Fingerprint:** F4-adjacent — not two code expressions of one
quantity kept in sync by agreement (A60's class), but two independently
maintained SOURCING decisions for what the codebase's own comments
identify as the same underlying quantity ("Nigerian commercial
power-sector WACC" / "cost of commercial capital"), never reconciled
against each other or against the sourced floor either rests against.

**Location:** `src/scenarios.py:1881-1899` (`capital_recovery_scenarios()`:
`cr_central` `r_comm` = 0.11, `cr_comm_high` = 0.14 — feeds the F1
per-tranche CRF in the objective) vs `src/scenarios.py:132-156`
(`blended_finance_scenarios()` docstring: `r_commercial` = 0.18,
feeding `resolve_private_rate()` and the bankability blend, per L17).

**What is wrong.** A WACC cannot lie below its own cost of debt. CPI
(2023), *Cost of Capital for Renewable Energy Investments in Developing
Economies* (`docs/sourcing_pack_01_rates.md` §2), gives climate-project
cost of debt at 18.5% (Jamaica, S&P B+) and 19.0% (Bangladesh, BB-);
Nigeria rates at or below B, implying its cost of debt is >= 18.5% and
a commercial WACC near 20% (about 16% after a tax shield). The
objective's `cr_central` `r_comm` (0.11) rests on an IEA Cost of Capital
Observatory 8.5-9% WACC figure for Kenya and Senegal (Nigeria not
covered) plus a 2-2.5pp uplift that `capital_recovery_scenarios()`'s own
docstring marks explicitly as "AN ANALYST ESTIMATE, NOT SOURCED." Even
`cr_comm_high` (0.14) sits below the CPI-implied floor. The bankability
blend's `r_commercial` (0.18) is much closer to the CPI range but is
still, per the same docstring, "likely conservative-LOW." The
inconsistency between 0.11 and 0.18 for what both docstrings describe
as the same quantity is already flagged in code (`scenarios.py:152-156`:
"UNRESOLVED: a WACC cannot lie below its own cost of debt... No value
changed here; awaiting author decision") but had no register or audit
entry until this one.

**Why it matters.** `cr_central` is the default, no-fallback
`capital_recovery_case` used by every central-configuration run (CLI
default across all live runners; `27_run_hydro_minflow_sensitivity.py:123`
hardcodes it explicitly). Its `r_comm` = 0.11 therefore feeds the F1
capital-recovery CRF in every reported central-configuration objective
value in `findings_frozen.md`, including the 27.954 bn / 65,613 MWh
anchor.

**Severity:** UNRESOLVED, not LIVE-REPORTED — this is a declared,
sourced-but-conflicting parameter choice, not a proven coding error, and
no number in `findings_frozen.md` is asserted to be wrong because of
it. But it is load-bearing under every `cr_central` / `cr_ida21` /
`cr_comm_*` result, which is effectively all F1 objective results.

**Declared?** Partially, in code comments only (`scenarios.py:132-156`,
already present at HEAD, commit `85f2598`). **No register or audit
entry existed before this one** (methodology claims ledger A4; register
L52).

**UPDATE (rate-coherence task, Part 1, 2026-09-25).** The two
MECHANISMS this fingerprint describes (F1 CRF vs bankability blend)
no longer read two different commercial literals for every live path
except FIN-2 — `bank_tenor_case`'s new default (`bank_tenor_comm`)
makes the bankability discount factor read `r_comm_nominal` (the same
`cr_central` 0.11 the objective already uses), not
`blended_finance_scenarios()`'s 0.18. See register L17, L52 (full
account). The SOURCING question this fingerprint's "What is wrong"
section raises — whether 0.11 itself clears the CPI cost-of-debt floor
— is untouched; `r_comm` was not edited. FIN-2 remains pinned to the
pre-correction blend (0.18) explicitly, by design, not by omission.

---

### A62 — `within_block_peaking_factor` is a builder-side `dict.get`
default with no scenario entry, no arm, and no signature in any output

**Fingerprint:** F1/F7-adjacent — a load-bearing constant reached only
through a builder-side `dict.get(key, DEFAULT)`, invisible to every
runner, with an adjacent code comment (`SOURCE: [PENDING]; DEFAULT 1.0
is a declared conservative simplification`) that describes a DIFFERENT
default (1.0) than the one the code beneath it actually uses (1.3448) —
prose contradicting code in both builders.

**Location:** `src/optimize_model_sliced.py:1262-1263`
(`scenario.get("within_block_peaking_factor", 1.3448)`) and
`src/optimize_model.py:1039-1040` (same default, same pattern — both
builders carry it; see register L50's confirmation that this is not
specific to the sliced builder).

**What is wrong.** No `load_scenario()` entry ever sets
`within_block_peaking_factor`; no runner exposes it as a CLI arm; it
appears in no output tag. Its value (1.3448) is derived (per
`methodology_physical_layer.md` §2) from 1.4588 (instantaneous PAR) /
1.0847 (block PAR, the STALE figure — ledger C-4; HEAD-derived is
1.0877), so on HEAD's own data the same derivation would give 1.3411
(-0.28%). It multiplies the firm-capacity requirement in every binding
year (20 of 30 at the central configuration).

**Why it matters.** Exactly the silent-default failure class that
`methodology_protocols.md` §1's required-key discipline exists to
prevent, reached through a builder default rather than a
`load_scenario()` sentinel, so the discipline does not catch it. See
register L50 for the full account, including the comment/code
contradiction and the un-established direction of bias.

**Severity:** LIVE-LATENT — currently used, unflagged, in every solve at
every arm (the factor is arm-independent even though its own
derivation is not), but not proven wrong, only undocumented and
unsourced-at-its-actual-value.

**Declared?** No, before this task (methodology claims ledger C-5;
register L50).

---

### A63 — A commit's message asserts code changes it did not make, and
two downstream commits relied on the message rather than the diff

**Fingerprint:** F7 (prose-vs-code) at the commit-message level rather
than the doc-vs-code level this fingerprint elsewhere describes — the
"prose" here is a git commit message, and the consumers who trusted it
were later commits, not a reader of a methods document.

**Location:** commit `977a682` ("Expose --solar-cf-case on seven live
runners; fix hardcoded EaaS CF to derive from the selected arm"). Actual
diff: 172 lines, entirely inside `docs/shape_factors_and_dist5.md`. No
`src/`, no `scripts/`. `git show --stat 977a682` confirmed this exactly.

**What is wrong.** The commit message describes a code change (exposing
a CLI flag on seven runners, deriving a hardcoded value by ratio) that
the commit's own diff does not contain. Two later commits then acted on
the message's claim rather than re-checking the diff: `0c2cd5b`
("Register: L16 revised (DIST-5 implemented...)") revised the register
to say DIST-5 was implemented, and `d55f814` propagated "implemented and
measured inframarginal (<0.03%)" into `findings_frozen.md` F-13. Neither
`0c2cd5b` nor `d55f814` touches `src/` or `scripts/` either
(`git show --stat` on both, confirmed) — the entire chain from
"DIST-5 implemented" to "measured at <0.03%" rests on zero lines of
executable code.

**Why it matters.** This is the mechanism by which an unmeasured figure
(the DIST-5 "<0.03%" objective movement) entered `findings_frozen.md`
as a confirmed, citable result. See methodology claims ledger C-10;
register L25 (corrected); `findings_frozen.md` F-13 (corrected).

**Severity: LIVE-REPORTED.** Unlike every finding in the Phase 1-3
tally above, this one DID meet the strict bar — "affects a number
currently published in `findings_frozen.md`" — until this task's A1
correction removed the affected clauses. It is the one LIVE-REPORTED
finding this three-phase-plus-Phase-2-methodology audit has produced.

**Declared?** No, before this task (methodology claims ledger C-10/A7
in TASK.md's own numbering — not to be confused with this file's
pre-existing, unrelated `### A7` entry above, `apply_access_adjustment`
has zero callers).

---

## PHASE 3 — CLOSING SECTIONS

### LIVE-REPORTED list, first and separately

**None of the thirty-one Phase 3 findings (A29-A59) meets the strict
bar of "affects a number currently published in `findings_frozen.md`
today."** Checked explicitly for each, not assumed — every finding
above states its own citation check. The closest:

- **A40** (`08_phase8_analysis.py`'s `storage_capacity_mwh` NaN): present
  in the actual, currently-generated `results/phase8/master_results.csv`
  on disk right now — not a hypothetical — but `08_phase8_analysis.py`'s
  own output is not cited anywhere in `findings_frozen.md`.
- **A50** (`16_plot_cost_decomposition.py`'s missing O&M segment): would
  mis-render the moment this script is next run for a thesis figure; not
  yet cited.
- **A37** (`07_run_gas2_eaas_gas_relief.py`'s copy-paste status bug): a
  verified, high-confidence defect that would corrupt GAS-2's cleanest
  NDC arm the instant this dead runner receives the same repair already
  applied to five siblings — dormant only because the runner cannot run
  at all today.
- **A46/A47** (`15_run_pol1_ndc_comparison.py`'s required-margin
  double-count and capital-envelope tier mismatch): this script IS live
  and executable today, unlike most of this phase's other findings — the
  two closest things to a currently-exercisable defect in this phase,
  short of a published citation.
- **A29/A33** (register L13's own 114.9 figure; `rpublic_diagnostic.
  py`'s solar lifetime): wrong numbers a human reader could carry
  forward, but neither is read by any runner or the LP itself.

### Findings table (A29-A59)

| ID | Fingerprint | File | Severity | Declared? |
|---|---|---|---|---|
| A29 | F7 | `docs/limitations_register.md` L13 vs `build_gas_deliverability.py` | COSMETIC-TRAP | No |
| A30 | F7 | `scripts/f1_quantity_diagnostic.py` | COSMETIC | No |
| A31 | F10/F3 | `scripts/plot_storage_regime.py` | TRAP | No |
| A32 | F1/F9 | `scripts/plot_system_diagnostics.py` | COSMETIC | No |
| A33 | F6 | `scripts/rpublic_diagnostic.py` | LIVE-LATENT | No |
| A34 | F1 | `scripts/05_run_fin3_capital_sweep.py` | TRAP | Partially |
| A35 | F3/F4 | `scripts/05_run_fin3_capital_sweep.py` | TRAP/COSMETIC | No |
| A36 | F5 | `scripts/06_run_gas1_shadow_benchmarks.py` | TRAP | No |
| A37 | F12 | `scripts/07_run_gas2_eaas_gas_relief.py` | TRAP | No |
| A38 | F2 | `scripts/07_run_gas2_eaas_gas_relief.py` | TRAP | No |
| A39 | F7 | `docs/runner_inventory.md` vs `08_phase8_analysis.py` | COSMETIC | No |
| A40 | F2 | `scripts/08_phase8_analysis.py` | LIVE (in-output), not LIVE-REPORTED | No |
| A41 | other | `scripts/08_run_gas3_regime_ndc_feasibility.py` | TRAP | Partially |
| A42 | F6 | `scripts/08_run_gas3_regime_ndc_feasibility.py` | TRAP | Partially |
| A43 | F7 | `scripts/08_run_gas3_regime_ndc_feasibility.py` | TRAP | No |
| A44 | F6/F7 | `scripts/11_run_rel3_financing_frontier.py` | DORMANT/COSMETIC | No |
| A45 | F8/F4 | `scripts/11_run_rel3_financing_frontier.py` | DORMANT | No |
| A46 | F8/F4 | `scripts/15_run_pol1_ndc_comparison.py` | LIVE-LATENT | No |
| A47 | F5/F7 | `scripts/15_run_pol1_ndc_comparison.py` + `scripts/02_run_ndc_caps.py` | LIVE-LATENT | No |
| A48 | F6 | `scripts/15_run_pol1_ndc_comparison.py` | COSMETIC | Partially |
| A49 | F4 | `scripts/15_run_pol1_ndc_comparison.py` | TRAP | No |
| A50 | F1/F7 | `scripts/16_plot_cost_decomposition.py` | LIVE-LATENT | No |
| A51 | F7 | `scripts/04_run_fin2_tariff_sweep.py` (docs) | TRAP | No |
| A52 | F8 | `scripts/04_run_fin2_tariff_sweep.py` | LIVE-LATENT | Partially |
| A53 | other | `scripts/04_run_fin2_tariff_sweep.py` | TRAP | No |
| A54 | other | `scripts/12_run_dem1_demand_sensitivity.py` | COSMETIC | No |
| A55 | other | `scripts/13_run_dem2_growth_gas_interaction.py` | TRAP | Partially |
| A56 | F8/F4 | `src/optimize_experiments.py` (`run_dem2_growth_gas_matrix`) via `13_` | TRAP | No |
| A57 | other | `scripts/14_run_str1_storage_role.py` | TRAP | No |
| A58 | other | `scripts/21_compute_resolution_premium.py` | LIVE-LATENT/TRAP | No |
| A59 | F6-shaped | `scripts/21_compute_resolution_premium.py` | COSMETIC-TRAP | No |

### `build_gas_deliverability.py` — the required, standalone verdict

**What it computes matches what is on disk, exactly, verified
byte-for-byte, not inspected.** Re-derived all 300 rows of the current
`data/gas/processed/gas_available_power_annual_twh_th.csv` directly from
the current script and YAML (side-effect-free import, no file written):
zero mismatches, zero missing, zero extra, across all ten scenario
names. Working tree clean against HEAD for the script, the YAML, and the
CSV. The two-family scenario vocabulary matches `src/scenarios.py`'s
`gas_deliverability_scenarios()` registry exactly in both directions.
The separate `gas_availability_case` mechanical-derating axis is
correctly orthogonal, confirmed by reading both, not inferred from
naming.

**Against the METHODOLOGY documents: not checked** — out of scope this
phase by explicit instruction (code only; `methodology_physical_layer.
md`/`methodology_protocols.md` excluded except for the constraint-
semantics carve-out, which this file's own output is not).

**Against the REGISTER: correct on every claim checked except one
number.** L13's two-family framing, the drift family's 2030 range
(81.5-94.7), and the GMP family's upper bound (191.7) are all
independently verified correct. Its GMP-family lower bound (114.9) is
not — the code and config actually produce 140.5 for the GMP-partial
arm at 2030 (see A29). This is the one place code and register diverge
on a checkable number; the code is right and the register's one figure
is wrong.

**Overall verdict: this file's own computation is sound, reproducible,
and well-sourced. The single most load-bearing unaudited file in the
repository earns a clean bill on its own code; the one defect found this
phase sits in the register's prose about it, not in the file itself.**

### Coverage statement

**Tier 1 (16 files), read in full this phase** (delegated to three
parallel passes, spot-verified on the highest-severity claims — see
each finding's "personally verified" marker): `04_run_fin2_tariff_
sweep.py`, `05_run_fin3_capital_sweep.py`, `06_run_gas1_shadow_
benchmarks.py`, `07_run_gas2_eaas_gas_relief.py`, `08_phase8_analysis.
py`, `08_run_gas3_regime_ndc_feasibility.py`, `09_run_rel1_feasibility.
py`, `10_run_rel2_marginal_cost.py`, `11_run_rel3_financing_frontier.
py`, `12_run_dem1_demand_sensitivity.py`, `13_run_dem2_growth_gas_
interaction.py`, `14_run_str1_storage_role.py`, `15_run_pol1_ndc_
comparison.py`, `16_plot_cost_decomposition.py`, `21_compute_
resolution_premium.py`, `99_validate_baseline.py`.

**Tier 2 (9 files), read in full this phase, personally, not
delegated:** `_check_solver.py`, `_compare_2_6.py`, `build_gas_
deliverability.py`, `diag_fin2.py`, `f1_quantity_diagnostic.py`,
`plot_storage_regime.py`, `plot_system_diagnostics.py`, `probe_2_6_
structure.py`, `rpublic_diagnostic.py`.

**Tier 3, headers re-checked this phase** (full read was Phase 2's):
`scripts/retired/09b_run_rel1_mode_sensitivity.py`, `scripts/retired/
09b_run_rel1_retirement_sensitivity.py`, `scripts/deferred/17_run_
monte_carlo.py`.

**Together with Phases 1-2's own coverage, every numbered script and
shell harness in `scripts/` (including `scripts/deferred/` and `scripts/
retired/`) has now been read at least once, in full or substantially,
across the three phases.** The only remaining, honestly-stated gaps: (1)
`01_run_baseline.py`/`02_run_eaas.py`/`03_run_ndc_eaas.py` were read
only in part in Phase 2 (required-flag counts / mirrored spot-checks);
this phase did not revisit them, since they were outside its own
"NOT reached" list. (2) `methodology_physical_layer.md` and
`methodology_protocols.md` remain unchecked against code — out of
scope this phase by explicit instruction, not newly closed. (3)
Cross-referencing every one of this phase's 31 findings against every
one of the ~45 files in `scripts/` for a SECOND, independent occurrence
was not performed beyond what each finding's own "duplicate source of
truth" (F5) check already covered — a finding like A36's benchmark-dict
duplication was checked against its known siblings, not against the
entire codebase from scratch.

**Nothing edited. Nothing run. Nothing committed** — every check this
phase was a read, a `grep`/`git log`/`git show` (read-only), an `ast`/
`inspect.signature` parse, or a side-effect-free Python/CSV/JSON read
or re-derivation of an already-existing file (including `build_series()`
called directly to reproduce the checked-in CSV — a pure function, no
file written, no model built, no solve). Confirmed via three independent
research passes' own reports plus this document's own personal spot-
checks.

### Combined three-phase tally

| | Phase 1 (`src/`) | Phase 2 (scripts/harnesses/docs) | Phase 3 (unread scripts) | Total |
|---|---|---|---|---|
| Findings | A1-A17 (17) | A18-A28 (11) | A29-A59 (31) | **59** |
| LIVE-REPORTED | 0 | 0 | 0 | **0** |
| Fixed | 2 (A8, A15) | 6 (A18, A21, A23, A26, A27, A28) | 0 | **8** |
| Open | 15 | 5 | 31 | **51** |

**Zero of the 59 findings across all three phases has ever met the
strict "affects a number currently published in `findings_frozen.md`"
bar.** Eight are fixed (all from Phases 1-2, this session). The
remaining 51 are TRAP, LIVE-LATENT, DORMANT, or COSMETIC — real defects,
none of them currently wrong in a way that has reached a frozen,
citable number. This is the headline this three-phase audit set out to
be able to say honestly, and — checked explicitly, phase by phase,
finding by finding, not asserted — it can.

**Nothing edited. Nothing committed. Nothing run**, across all of Phase
3.
