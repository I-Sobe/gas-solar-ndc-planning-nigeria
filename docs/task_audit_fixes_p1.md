# task_audit_fixes_p1 — fixing A15 and A8, declaring A14 and the dead modules

Branch `phase-fin2-d17`. Model frozen at `2a9226b`. This task edits two
builder/library files — `src/optimize_model_sliced.py` (A15) and
`src/optimize_experiments.py` (A8) — the first builder edit since the
freeze that ADDS a genuinely reachable code path (the two conversion-
capex tasks added inert `Expression` objects; this one adds an `if`
block that can execute). The argument for why the anchors survive it
is in Part 3, verified against the actual anchor call site, not
asserted.

---

## PART 1 — A15: porting the annual builder's alias translation

**Read first, quoted exactly** (`src/optimize_model.py:55-63`, the
existing, working block):
```python
    if reliability_max_unserved_fraction is not None:
        if max_offgrid_fraction is not None:
            raise ValueError(
                "Pass either max_offgrid_fraction or its deprecated alias "
                "reliability_max_unserved_fraction, not both."
            )
        max_offgrid_fraction = reliability_max_unserved_fraction
    if reliability_mode is not None:
        grid_service_mode = reliability_mode
```
No adaptation was needed — the surrounding locals (`max_offgrid_
fraction`, `grid_service_mode`, `reliability_max_unserved_fraction`,
`reliability_mode`) are named identically in both builders' signatures,
confirmed by re-reading both before writing anything
(`optimize_model.py:35-38` vs `optimize_model_sliced.py:226-229`,
unchanged since the audit).

**Placement.** `build_model_sliced`'s real values are first READ at
what is now line ~1183 (`if max_offgrid_fraction is not None:`,
shifted down by this edit's own insertion — confirmed by re-grepping
after the edit) — far into the function. The annual builder places its
translation block as the very first statement in the function body,
before its own (unusually-placed, second) docstring string. `build_
model_sliced`'s docstring is its real one and sits first legitimately,
so the block was placed immediately after it, before the "Plan 0.4,
item B" comment and before `m = pyo.ConcreteModel()` — the earliest
point after the docstring, and unambiguously before either aliased
value is read anywhere.

**Old** (`src/optimize_model_sliced.py`, immediately after the
docstring):
```python
    """
    Build a time-sliced planning optimisation model.

    Signature matches build_model() so runners can swap with one import change.
    """
    # Plan 0.4, item B: the "defensive objective check" that stood here
```

**New:**
```python
    """
    Build a time-sliced planning optimisation model.

    Signature matches build_model() so runners can swap with one import change.
    """
    # ------------------------------------------------------------------
    # Grid service obligation (renamed from "reliability" in 2.6 F0).
    #
    # This constraint bounds (selfgen + unserved) -- the GRID SERVICE
    # SHORTFALL -- not unserved alone. A household running its own genset was
    # not served by the grid, but no capacity was short, so this is a service
    # obligation, NOT an adequacy/reliability standard: LOLE and EENS do not
    # describe it. The old names are retained as deprecated aliases so the
    # pre-2.6 REL and VoLL runners keep working unchanged.
    #
    # task_audit_fixes_p1 (audit finding A15): this translation was present
    # in build_model() (optimize_model.py) but silently absent here -- the
    # two DEPRECATED aliases above were accepted into the signature and
    # never read again anywhere in this function. Ported verbatim from
    # optimize_model.py's own working block (same names, same guard, same
    # error text) rather than reimplemented, so the two builders cannot
    # drift apart on this translation a second time.
    # ------------------------------------------------------------------
    if reliability_max_unserved_fraction is not None:
        if max_offgrid_fraction is not None:
            raise ValueError(
                "Pass either max_offgrid_fraction or its deprecated alias "
                "reliability_max_unserved_fraction, not both."
            )
        max_offgrid_fraction = reliability_max_unserved_fraction
    if reliability_mode is not None:
        grid_service_mode = reliability_mode

    # Plan 0.4, item B: the "defensive objective check" that stood here
```

**Checklist against the task's five requirements:**
1. Copied verbatim, not reimplemented from the description — done, text
   is character-for-character the annual builder's block plus the
   attribution comment.
2. Placed before either value is read — confirmed (`grep -n
   "max_offgrid_fraction\|grid_service_mode"
   src/optimize_model_sliced.py` after the edit shows the new block's
   two writes precede the first read by ~950 lines).
3. `ValueError` on both-passed kept, same text.
4. Both deprecated parameters remain in the signature — untouched.
5. `scripts/23_run_selfgen_frontier.py` not edited — confirmed,
   `git status --porcelain` after this task's edits does not list it.

Syntax-checked: `python -m py_compile src/optimize_model_sliced.py` —
OK.

---

## PART 2 — A8: matching the sliced builder's own guard

**Sliced builder's version, quoted, and followed** (`src/optimize_
model_sliced.py:491-493`):
```python
    eta = scenario.get("gas_eta", None)
    if eta is None:
        raise ValueError("gas_eta is unset. Provide the measured fleet efficiency.")
```

**Old** (`src/optimize_experiments.py`, inside `run_deterministic_
scenario`):
```python
    eta =scenario.get("gas_eta", 0.43)
```

**New:**
```python
    # task_audit_fixes_p1 (audit finding A8): was `scenario.get("gas_eta",
    # 0.43)` -- 0.43 is not an arbitrary placeholder, it is the retired
    # CCGT-on-an-OCGT-fleet figure this project's sourcing ledger records
    # as wrong and replaced by the measured 0.287
    # (docs/merged_correction_plan_v2.md, sourcing ledger, eta row). A
    # fallback that ever fired would silently overstate fleet efficiency
    # by 50%. Matches optimize_model_sliced.py's own gas_eta guard
    # (:491-493) instead: this function takes a bare scenario dict, and
    # nothing in its signature requires load_scenario() to have built it.
    eta = scenario.get("gas_eta", None)
    if eta is None:
        raise ValueError("gas_eta is unset. Provide the measured fleet efficiency.")
```
Direct subscript (`scenario["gas_eta"]`) was considered and rejected in
favour of the `.get(None)`+raise form specifically to match the sliced
builder's own register/style (a named, explanatory `ValueError`) rather
than a bare `KeyError` — "the register of the other sentinel raises"
this function's other required keys (`scenario["base_demand_twh"]`,
`scenario["demand_growth"]`, `scenario["gas_scenario"]`,
`scenario["solar_cf"]`, `scenario["storage_round_trip_eff"]`,
`scenario["carbon_active"]`) already use bare subscripting with no
custom message — so the sliced builder's explanatory-raise form was
chosen deliberately as the CLEARER of the two available in-project
conventions for a key whose failure mode (silently reverting to a
retired, wrong value) is exactly the kind this project's sentinel-raise
discipline exists to prevent, per the task's own instruction to explain
why an unstated value is unsafe.

**Siblings found, per Part 2's own instruction — reported, not fixed.**
`run_deterministic_scenario` has exactly one other `.get(` with a
numeric default on a load-bearing key:
```python
storage_duration_hours = scenario.get("storage_duration_hours", 4.0)
```
(`optimize_experiments.py`, three lines below the fixed line). **Not a
retired value** — the line's own comment states it explicitly:
*"Canonical site: scenarios.py:2666 (_storage_duration_hours = 4.0)...
Read from the scenario in scope here instead of a second, independent
4.0 literal that could silently drift from it -- same discipline as the
builders' own scenario.get('storage_duration_hours', 4.0)."* `4.0` is
also the value both LP builders themselves fall back to
(`optimize_model.py`, `optimize_model_sliced.py`, both `scenario.
get("storage_duration_hours", 4.0)` — register **L21**'s own subject).
This fallback is CONSISTENT with the canonical value everywhere else in
the project, not a divergent or retired one — reported per instruction,
not fixed, since it does not share A8's defect shape (a value that
differs from canonical). No other `.get(` calls exist in this function
(read start to end, `optimize_experiments.py:245-331`, confirmed no
third instance).

Syntax-checked: `python -m py_compile src/optimize_experiments.py` —
OK.

---

## PART 3 — why the anchors cannot move

**A15.** The new block executes only when `reliability_max_unserved_
fraction is not None` or `reliability_mode is not None`. **Search
performed**: `grep -n "reliability_max_unserved_fraction\|reliability_
mode" scripts/01b_run_baseline_sliced.py` and, separately, the full
`build_model_sliced(...)` call site in that file
(`m = build_model_sliced(scenario=scenario, econ=econ,
emissions_cap=EMISSIONS_CAP, emissions_cap_by_year=None,
solar_capex_by_year=solar_capex_tv)`, read in full) — **zero
occurrences of either alias name**, and the five keyword arguments
actually passed do not include them. Both aliases therefore keep their
`None` default on the anchor call, the new `if` block's conditions are
both false, and it does not execute — not "executes harmlessly," but
literally not entered. The constraint matrix `build_model_sliced`
constructs afterward is therefore byte-for-byte what it was before this
edit on the `01b_run_baseline_sliced.py` call path, which is what
produces both anchors (`om_none` regression, `om_central` central).

**A8.** `run_deterministic_scenario` is called from exactly two sites
in the whole tree (`grep -rn "run_deterministic_scenario("` excluding
its own `def`): `scripts/02_run_eaas.py:301` and `src/optimize_
experiments.py:3886` — the latter inside `run_all_deterministic_
scenarios`, one of the audit's own A17 dead functions (zero callers of
ITS OWN, confirmed again here), so that call site never executes
either. `02_run_eaas.py`'s call is real, but its output (`det_output`)
is used only to print `det_unserved` to the console
(`02_run_eaas.py:337-338`, read in full — no third use, nothing written
to `diagnostics.json`/`summary.json`). Neither anchor is produced by
`02_run_eaas.py` — both anchors (`om_none` regression, `om_central`
central) come from `01b_run_baseline_sliced.py`, which does not import
or call `run_deterministic_scenario` at all (confirmed:
`grep -n "run_deterministic_scenario"
scripts/01b_run_baseline_sliced.py` — no match). Both arguments hold
cleanly; neither needed hedging.

---

## PART 4 — register drafts (next free number: L29 — re-confirmed,
`docs/limitations_register.md`'s last applied entry is still L28)

**All five below are drafts. None applied — the register is outside
this task's whitelist.** Numbered L29-L33 sequentially; as with every
prior unapplied draft in this task sequence, whoever applies the
register should check for overlap with any other still-pending L29
draft from earlier tasks before assigning final numbers.

### 4A — L29, for A15

> ## L29 — `build_model_sliced()` silently dropped its own declared
> reliability-alias parameters; a real sweep's sliced half was
> empirically dead across twelve values
>
> **Status:** RESOLVED (`task_audit_fixes_p1`) — the translation is now
> ported into `build_model_sliced()`; the historical defect and its
> consequence are recorded below for the record.
>
> **What.** `build_model_sliced()` accepted `reliability_max_unserved_
> fraction` and `reliability_mode`, both commented `# DEPRECATED alias
> for ...`, and never read either anywhere in its ~1,700-line body. The
> annual builder, `build_model()`, has the identical two parameters
> with identical comments and correctly translates them into
> `max_offgrid_fraction`/`grid_service_mode` (including a `ValueError`
> guard against passing both an alias and its canonical parameter).
> Only the sliced builder was missing the translation.
>
> **Consequence, verified empirically, not merely argued.**
> `scripts/23_run_selfgen_frontier.py` sweeps a 12-point `eps` grid
> across BOTH builders in one run, passing ONLY
> `reliability_max_unserved_fraction=eps` (never `max_offgrid_
> fraction`) to whichever builder is active that iteration. In the
> already-persisted `results/selfgen_frontier/frontier.csv` (48 rows),
> every `sliced` row for a given `pricing` arm is bit-identical to full
> float precision across all twelve swept `eps` values (`objective_usd`
> and `shortfall_fraction_realised` unchanged past the 13th significant
> digit) — the sliced half of the sweep solved the same unconstrained
> problem twelve times. The `annual` half of the SAME file, same run,
> varies correctly (infeasible below `eps=0.3`, then a smoothly
> declining objective from `1.4667e12` to `1.1468e12` as `eps`
> increases) — proving the sweep mechanism itself and the annual
> builder both worked, isolating the defect to the sliced builder's
> silently-dropped aliases.
>
> **Fingerprint.** A parameter accepted and then discarded — the
> caller's intent vanished silently, with no error, warning, or
> visible signature in the output (every `sliced` row still printed
> `feasible=True` with a plausible-looking objective). The annual
> builder's CORRECT implementation of the identical translation is
> precisely what made this detectable at all: without a working
> sibling to diff against, twelve identical numbers in a row could as
> easily have been mistaken for "the constraint doesn't bind in this
> range" as for "the constraint was never wired."
>
> **Direction of bias.** The sliced-builder half of `results/
> selfgen_frontier/frontier.csv` understated the sensitivity of every
> reported quantity to the reliability threshold — not a directional
> bias on a level, but a total absence of the swept relationship where
> one was intended and reported as present.
>
> **What is void, and must be regenerated.** `results/selfgen_
> frontier/frontier.csv`'s `sliced` rows, in full — see the DO-NOT-
> REPORT addition, 4E below. No number in `findings_frozen.md` or
> `frozen_state_2026_09_10.md` was found to derive from this file
> (checked: `grep -rn "selfgen_frontier\|23_run_selfgen\|EPS_GRID"`
> across both — no hits), so no published figure requires correction;
> the file itself does.
>
> **What would resolve it further.** Nothing — closed by the port.
> `23_run_selfgen_frontier.py` must be re-run (outside every
> whitelist so far; Zeus's own next step) before its sliced-builder
> output can be cited.

### 4B — L30, for A8

> ## L30 — A live function carried the retired gas efficiency (0.43)
> as a silent fallback; inert only because its one caller never
> omitted the key
>
> **Status:** RESOLVED (`task_audit_fixes_p1`).
>
> **What.** `run_deterministic_scenario()` (`src/optimize_
> experiments.py`, a legacy non-LP dispatch simulator, explicitly
> labelled "NOT for thesis result tables") read `scenario.get(
> "gas_eta", 0.43)`. `0.43` is not an arbitrary placeholder: it is the
> exact figure this project's own sourcing ledger
> (`docs/merged_correction_plan_v2.md`) records as retired — a CCGT
> efficiency figure misapplied to this project's OCGT-dominated fleet,
> replaced project-wide by the measured **0.287** derived from NERC
> 2024 generation ÷ gas-to-power volumes. A fallback that fired would
> silently overstate fleet efficiency by 50%, understating gas fuel
> consumption and emissions by the same margin in this function's own
> (non-reported) cost/emissions figures.
>
> **Why it never fired.** `load_scenario()` unconditionally sets
> `scenario["gas_eta"] = 0.287` in its returned dict, and this
> function's one live caller (`scripts/02_run_eaas.py:301`) always
> builds its scenario via `load_scenario()`. The fallback was
> therefore reachable in principle, never in the run history this
> project can inspect.
>
> **Why "never fires today" is not a safety property.** `run_
> deterministic_scenario(scenario, econ, capacity_paths)` takes a bare
> `scenario` dict as a parameter — nothing in its signature or body
> requires the caller to have built it via `load_scenario()`. The
> function's own docstring describes its purpose as "quick feasibility
> sketches" from a HAND-SUPPLIED capacity path — precisely the use
> case most likely to be invoked with a hand-built, partial scenario
> dict that omits `gas_eta`. This is the same shape this project's own
> register already names for other parameters (L9, L19, L21): a
> load-bearing value whose correctness depends on every caller
> happening to go through one specific code path, with nothing
> enforcing that they do.
>
> **Resolution.** Changed to `scenario.get("gas_eta", None)` +
> `raise ValueError(...)` if absent, matching `build_model_sliced()`'s
> own existing guard for the same key verbatim. A future caller that
> omits `gas_eta` now fails loudly, at the point of failure, naming the
> key — not months later, in a plausible-looking but 50%-wrong number.
>
> **Direction of bias, historical.** None realised — confirmed no run
> in this project's history reached the fallback branch (the one live
> caller always supplies the key). Recorded as a TRAP class finding,
> not a LIVE-REPORTED one: correct in every run to date, wrong the
> first time a caller does what this function's own documentation
> invites.
>
> **Sibling checked, not a defect.** The same function's other numeric
> `.get()` default, `scenario.get("storage_duration_hours", 4.0)`,
> matches the canonical value set by `load_scenario()` and used
> identically by both LP builders (register L21) — not fixed, since it
> is not a divergent or retired figure.

### 4C — L31, for A14

> ## L31 — The reported "peak adequacy" diagnostic uses a flat,
> arm-invariant multiplier while the constraint it is named after uses
> an arm-sensitive one
>
> **Status:** LIVE.
>
> **What.** `extract_planning_diagnostics`'s `peak_adequacy_by_year`
> field (`src/optimize_experiments.py`) computes its own `peak_demand_
> mw` as `avg_demand_twh_converted_to_mw × peak_demand_multiple`, where
> `peak_demand_multiple` is a flat **2.5**, set unconditionally by
> `load_scenario()` (`scenarios.py:3157`) OUTSIDE any `peak_multiple_
> case` branch — identical for `peak_suppressed`, `peak_unsuppressed`,
> `peak_nirp`, and `peak_high`. The ACTUAL firm-capacity constraint
> both LP builders build sizes against a structurally different
> quantity, `peak_mw[t]` — a REQUIRED scenario key, DERIVED from the
> selected `peak_multiple_case` arm's own `season_demand_factor`/
> `period_demand_factor` (register L1's subject), and genuinely
> sensitive to which arm is chosen.
>
> **Already known to the code, not to the register.** `scenarios.
> py:3150-3156` carries the author's own comment stating this is a
> "PRE-EXISTING, DIFFERENT quantity from peak_multiple_case/peak_mw...
> flagged here so the two are not confused" — an intentional, declared-
> in-a-comment design choice, not an oversight. It has never appeared
> in this register, `findings_frozen.md`, or `frozen_state_2026_
> 09_10.md`.
>
> **Why it matters.** `peak_adequacy_by_year`'s field names
> (`peak_demand_mw`, `firm_capacity_mw`, `reserve_margin`) read as a
> direct report on capacity adequacy against the model's own
> peak-demand assumption. They are not: the SAME configuration solved
> under two different `peak_multiple_case` arms produces two genuinely
> different firm-capacity requirements internally, but IDENTICAL
> `peak_demand_mw` figures in this diagnostic, because the diagnostic
> never reads `peak_multiple_case` at all.
>
> **Direction of bias.** Not one-directional — the diagnostic simply
> does not track the quantity its name claims to describe. Whether
> `reserve_margin` reads too high or too low relative to the true,
> arm-specific adequacy depends entirely on how far `2.5` sits from
> whatever `par_block`-derived multiple the selected `peak_multiple_
> case` arm actually implies, which varies by arm.
>
> **Why unresolved.** No fix is proposed by this task — Part 1 of the
> preceding audit found this, this entry only declares it. A fix would
> mean either deriving `peak_demand_multiple` from the same `season_
> demand_factor`/`period_demand_factor` machinery `peak_mw` already
> uses (closing the gap) or renaming the diagnostic's fields to state
> plainly that they use a fixed, arm-independent proxy (the L19/L21
> pattern) — a design decision, not made here.
>
> **Mitigation.** None currently — `peak_adequacy_by_year` has zero
> consumers in `scripts/` today (checked), so no published figure is
> affected. Do not cite `peak_adequacy_by_year`/`reserve_margin` across
> more than one `peak_multiple_case` arm as though it reflected the
> arm difference.
>
> **What would resolve it.** Derive `peak_demand_multiple` from the
> same per-arm shape factors `peak_mw[t]` uses, or rename the
> diagnostic's fields (e.g. `flat_proxy_peak_demand_mw`) to state
> plainly that it does not track `peak_multiple_case`.

### 4D — L32, for the dead modules and dead functions

> ## L32 — Three entire dead modules and six dead library functions;
> one of the three dead modules must not be revived without redesign
>
> **Status:** LIVE (declared, not deleted — this entry documents, it
> does not remove anything).
>
> **What.** Three files in `src/` define functions with zero callers
> anywhere in `src/` or `scripts/`, confirmed by direct search, not
> inference:
>
> 1. **`src/postprocess.py`** — `export_benchmark_comparison`, the
>    file's only function. Reads as support for GAS-1 (gas shadow
>    price vs export/domestic benchmarks) but `06_run_gas1_shadow_
>    benchmarks.py` computes its own comparison inline and never
>    imports this module.
> 2. **`src/gas_balance.py`** — `project_gas_availability_th` and
>    `gas_to_power_electricity_twh`, the file's only two functions. An
>    earlier, single-parameter exponential-decline model of gas
>    availability, superseded by the CSV/arm-based system in
>    `src/gas_supply.py` (`gas_available_power()`, keyed by `gas_
>    deliverability_case`) that every live caller actually uses.
> 3. **`src/stochastic.py`** — `sample_uncertainties`, `run_
>    stochastic_simulation`, `compute_risk_metrics`, all three
>    functions in the file. A complete, documented "fixed-plan
>    re-dispatch" Monte Carlo mechanism, distinct from both
>    `scripts/deferred/17_run_monte_carlo.py` (dead on required
>    arguments, register L10) and `scripts/29_run_parameter_
>    uncertainty.py` (the Sobol parameter-uncertainty sweep) — never
>    wired to any runner at all.
>
> Additionally, six top-level functions in `src/optimize_experiments.py`
> (out of 33) have zero callers: `run_annual_cap_case`, `identify_
> bottleneck`, `run_bottleneck_sensitivity`, `run_str1_storage_
> parameter_sweep`, `run_all_deterministic_scenarios`, `run_financing_
> vs_resource_test`. One of the six (`run_str1_storage_parameter_
> sweep`) is the expected, already-covered consequence of register
> L5's declared STR-1 suspension. The other five are an orphaned
> exploratory layer predating the current runner-script architecture.
>
> **`stochastic.py` specifically must NOT be revived as-is.**
> `sample_uncertainties` draws `demand_growth` and `gas_scenario` (the
> gas deliverability regime) as two INDEPENDENT random variables per
> Monte Carlo sample, despite both responding to a shared macro driver
> — the identical failure shape this project already retires a whole
> script over (`scripts/deferred/17_run_monte_carlo.py`, per its own
> declared defects and register L10). `run_stochastic_simulation` also
> calls `build_model()` (annual) exclusively, never `build_model_
> sliced()`, so any revival would additionally inherit register L5's
> "storage is structurally dominated at annual resolution — optimal
> storage is identically zero" for every draw. Both defects are
> currently inert only because nothing calls the module.
>
> **Direction of bias.** None currently realised — every function
> named above has zero callers, confirmed. Purely a "declared, not
> deleted" entry: nothing here changes a reported number.
>
> **What would resolve it.** A scoped decision on each: delete as dead
> weight, or retain with a stated reason (as `src/EaaS.py` already does
> for its own, larger dead surface, citing `docs/task_bucketA_
> parity.md` Part B1). If `stochastic.py` is ever revived, redesign the
> sampling to draw demand growth and gas regime from a joint or
> conditional distribution reflecting their shared driver, and decide
> deliberately whether to port it to `build_model_sliced()` first.

### 4E — DO-NOT-REPORT addition

> **`results/selfgen_frontier/frontier.csv` — sliced rows void.**
> Produced by `scripts/23_run_selfgen_frontier.py` before the A15 fix
> (`task_audit_fixes_p1`): every `sliced`-builder row swept `eps` via
> `reliability_max_unserved_fraction`, an alias `build_model_sliced()`
> silently dropped before this task. Verified empirically: all twelve
> swept `eps` values per `pricing` arm are bit-identical to full float
> precision in the `sliced` half of this file. **Bar the `sliced` rows
> of this file from any report** until `23_run_selfgen_frontier.py` is
> re-run against the now-fixed `build_model_sliced()` (outside this
> task's whitelist) and the file is regenerated. The `annual` rows in
> the same file are unaffected by this defect (verified: they vary
> correctly with `eps`) but were produced by the same run and should be
> regenerated alongside the sliced rows for a clean, single-provenance
> file rather than mixed pre-/post-fix rows in one CSV.

---

## SUMMARY: WHY NOTHING HERE CAN MOVE EITHER ANCHOR

Both edits are confined to code paths gated on conditions that are
false on the anchor-producing call (`01b_run_baseline_sliced.py`'s
`build_model_sliced(...)` call, which never passes either reliability
alias) or inside a function the anchor path never calls at all
(`run_deterministic_scenario`, reachable in this project only from
`02_run_eaas.py`, which produces neither anchor). Per Part 3's search
evidence, both arguments were verified against the actual current call
sites, not asserted from the audit's description alone. `git diff`
against both edited files shows purely additive changes (a new `if`
block in `build_model_sliced`; a two-line replacement of one `.get()`
call with an equivalent-shape guarded read in `run_deterministic_
scenario`) — no existing line touched, no `Constraint`, `Var`, or
`Objective` added or altered in either builder.

---

## ENCOUNTERED BUT NOT ASKED ABOUT

- `run_all_deterministic_scenarios` (`optimize_experiments.py:3886`
  area) calls `run_deterministic_scenario` too, but is itself one of
  the audit's six dead functions (A17) — re-confirmed zero callers of
  its own during Part 3's search, not just cited from memory.
- The annual builder's own translation block (`optimize_model.py:
  55-63`) sits BEFORE that function's second, unusually-placed
  docstring-like string literal (a plain expression statement, not the
  function's actual `__doc__`, which is the module-level string at the
  top of the file) — noted only because it explains why this task's
  ported block in the sliced builder (whose real docstring comes
  first, legitimately) is not in the exact same textual position
  relative to a docstring, even though it is in the same LOGICAL
  position (before any other processing).
- `docs/limitations_register.md`'s L29 slot has now been drafted by
  multiple prior tasks in this sequence for different findings (the
  conversion-diagnostics rename pattern, the capex-reporting-field
  pattern, and now this task's A15). None have been applied. Whoever
  applies the register will need to renumber more than one pending
  draft, not just this task's five.

**Nothing was committed or staged.** `git status --porcelain` at the
end of this task:
```
 M src/optimize_experiments.py       (this task, Part 2)
 M src/optimize_model_sliced.py      (this task, Part 1)
 M docs/findings_frozen.md           (pre-existing, uncommitted, untouched here)
 M docs/frozen_state_2026_09_10.md   (pre-existing, uncommitted, untouched here)
 M docs/methodology.md               (pre-existing, uncommitted, untouched here)
 M docs/methodology_protocols.md     (pre-existing, uncommitted, untouched here)
 M scripts/01_run_baseline.py        (pre-existing, uncommitted, untouched here)
 M scripts/02_run_eaas.py            (pre-existing, uncommitted, untouched here)
 M scripts/02_run_ndc_caps.py        (pre-existing, uncommitted, untouched here)
 M scripts/03_run_ndc_eaas.py        (pre-existing, uncommitted, untouched here)
 M scripts/18_run_pareto_frontier.py (pre-existing, uncommitted, untouched here)
?? docs/forensic_audit.md            (pre-existing from the prior task)
?? docs/task_audit_fixes_p1.md       (this report)
```

**STOP. Zeus reviews `git diff`, commits, runs both anchors, then
re-runs `23_run_selfgen_frontier.py`.**