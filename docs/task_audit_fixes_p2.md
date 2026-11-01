# task_audit_fixes_p2 — A18, A21, A26, A27, A28, A23 (L10)

Six fixes from the Phase 2 forensic audit, applied under the whitelist in
`TASK.md`. Reported incrementally below, finding by finding. Nothing in
this task was run, staged, or committed; see the confirmation at the end.

---

## A18 — `scripts/00_build_emissions_cap.py`: bare `KeyError` on a renamed diagnostics key

**Defect as found.** `TASK.md` characterised this as "update to the
current key name," implying the script still read a stale key. That
premise does not hold: `grep -n "gas_to_power_twh_th_by_year"
scripts/00_build_emissions_cap.py` shows the code already reads the
current, renamed key,

```python
gas_to_power = diag["fleet_gas_to_power_twh_th_by_year"]
```

with a comment above it correctly explaining the rename's provenance
(`task_conversion_diagnostics`, Part 2C). So step 2 of the task spec was
a no-op — flagged here rather than silently complied with.

The real defect is what happens when this line runs against an
on-disk `results/baseline/diagnostics.json` generated *before* that
rename. Direct check:

```
python3 -c "import json; d=json.load(open('results/baseline/diagnostics.json')); print('fleet_gas_to_power_twh_th_by_year' in d)"
→ False
```

The bare file on disk still carries the old key
(`gas_to_power_twh_th_by_year`). The dict subscript then raises a bare
`KeyError: 'fleet_gas_to_power_twh_th_by_year'` — naming the missing
key but nothing about which file, or why, or what to do about it.

**Fix.** Added a guard immediately before the existing (correct) read,
raising a message that names both the missing key and the file path,
and explains the likely cause and remedy:

```python
if "fleet_gas_to_power_twh_th_by_year" not in diag:
    raise KeyError(
        f"'fleet_gas_to_power_twh_th_by_year' not found in "
        f"{BASELINE_DIAG_PATH}. This file was likely generated before "
        f"the task_conversion_diagnostics rename (it may still carry "
        f"the old key, 'gas_to_power_twh_th_by_year') -- regenerate it "
        f"with scripts/01_run_baseline.py under conv_none and place the "
        f"output at this bare path (see this module's docstring, "
        f"'REQUIRED INPUT', for why it must be an annual conv_none run)."
    )
gas_to_power = diag["fleet_gas_to_power_twh_th_by_year"]
```

Nothing else in the file changed: the read itself, the "annual
conv_none baseline" docstring constraint, and the cap computation that
follows are untouched.

**Why the anchors cannot move.** This guard only fires on a *missing*
key. When the key is present — which it must be, for either anchor run,
since both use a current `01_run_baseline.py` — the guard's condition is
false, execution falls straight through to the unchanged
`gas_to_power = diag[...]` line, and every downstream computation is
byte-for-byte what it was before this change. `python -m py_compile
scripts/00_build_emissions_cap.py` passes.

---

## A21 — `scripts/02_run_ndc_caps.py`, `scripts/03_run_ndc_eaas.py`: solar-capex resolved twice

**Defect as found.** Both scripts build the model with an explicit
solar-capex series, then call the shared diagnostics function without
passing it — letting the diagnostics function re-resolve its own,
independent copy:

`scripts/02_run_ndc_caps.py` (and identically in `03_run_ndc_eaas.py`):

```python
solar_capex_tv = load_solar_capex_by_year(scenario_name="solar_low", start_year=..., end_year=...)
...
m = build_model_sliced(..., solar_capex_by_year=solar_capex_tv)
...
diag = extract_planning_diagnostics(m, scenario, econ)   # no solar_capex_series=
```

Inside `extract_planning_diagnostics` (`src/optimize_experiments.py`),
omitting `solar_capex_series` triggers an internal fallback that
resolves its own series from `scenario.get("solar_capex_scenario",
"solar_low")` and the scenario's own `years[0]`/`years[-1]`. This is
two independent resolutions of the same quantity — a source-of-truth
duplication (audit pattern F5), not merely a style issue: if either
default ever changes independently, the model and its own diagnostic
would silently diverge.

**Are the two currently identical?** Checked directly, not asserted.
`load_scenario()` unconditionally sets
`scenario["solar_capex_scenario"] = "solar_low"`
(`src/scenarios.py:2940`), which matches the explicit `"solar_low"`
both scripts pass to their own `load_solar_capex_by_year` call. The
internal fallback's `start_year=int(years[0])`,
`end_year=int(years[-1])` match `scenario["years"][0]` /
`scenario["years"][-1]` exactly, which is what both scripts pass to
their own explicit call. **The two resolutions are currently
identical.** This is a trap, not a live numerical defect: the duplicate
source of truth exists and could diverge under a future change, but as
of this commit it produces the same series both times.

**Fix (identical in both files).** Pass the already-built series
through, so there is one resolution and one source:

Old:
```python
diag = extract_planning_diagnostics(m, scenario, econ)
```

New:
```python
diag = extract_planning_diagnostics(m, scenario, econ, solar_capex_series=solar_capex_tv)
```

Confirmed via `grep -n "extract_planning_diagnostics("` that each file
has exactly one call site, so the fix was applied once per file, not
partially.

**Why no computed value changes.** Since the trap check above showed
the two resolutions were already identical, passing the model's own
series through makes the diagnostics function skip its internal
fallback entirely and use the value that was already being produced —
same numbers, now from one source instead of two. `python -m py_compile
scripts/02_run_ndc_caps.py scripts/03_run_ndc_eaas.py` passes.

---

## A26 — `scripts/run_verification_sweep.sh`: harness could never produce a usable signal

**What this script is for.** A snapshot regression harness: run every
runner in a fixed list, and — if none fails — copy `results/` to a
named snapshot directory for later diffing. Its value is in catching
unintended changes across runs, not in producing a single number.

**Defect as found.** Every invocation supplied at most `--om-case
om_central` and nothing else:

```bash
python "$f" --om-case om_central > "$log" 2>&1   # (only 5 of 19 runners even got this much)
```

But 01_/01b_/02_run_eaas/02_run_ndc_caps/03_ require 15-18 flags, and
05_/15_/18_/19_/20_ (via `src/cli_arms.py`) require the full
standard-arm surface too — ten of the nineteen listed runners rejected
the old invocation immediately via argparse's "the following arguments
are required" error. The old check was a binary pass/fail on exit code,
which cannot distinguish that healthy argparse rejection from an actual
crash in one of the already-declared-dead runners (register L10/L27).
Given the RUNNERS list contains scripts that will always exit non-zero,
the all-or-nothing snapshot gate could never fire, regardless of
whether the healthy runners' output was good.

**Fix.**
1. Added a `CENTRAL_ARGS` bash array, copied verbatim from
   `scripts/run_central_sweep.sh`'s own `C_*` constants (the canonical
   central configuration: `peak_unsuppressed`, `storage_cc_central`,
   `gas_unconstrained`, `constrained`, `gas_avail_nameplate`,
   `gas_retire_vintage_30`, `gas_capex_central`, `ccgt_eff_central`,
   `baseline`, `conv_central`, `hydro_flat`, `tariff_central`,
   `f1_tranche`, `dscr_central`, `cr_central`, `0.020`, `env_dares`),
   threaded into every single invocation, plus a per-runner
   `om_args=(--om-case om_central)` for the five runners whose argparse
   surface exposes that flag. This is the central set, not the
   regression set — this harness's purpose is a general health/snapshot
   check across the current configuration, not reproducing a frozen
   pre-change anchor, so `run_central_sweep.sh`'s `C_*` values are the
   correct choice here (contrast A27 below, where the regression set was
   the right choice for a different reason).
2. Replaced the binary check with `smoke_all.sh`'s own three-outcome
   PASS/NEEDS-ARGS/FAIL classification — reused, not reimplemented: same
   exit-code-2-plus-argparse-text detection, same rationale (that
   script's header explains why exit code 2 alone is not specific
   enough). Bash has no clean way to `source` a function out of another
   script without also executing its top-level body, so the ~10-line
   classifier was retyped verbatim rather than imported; this is called
   out in a comment in the file as a deliberate "reused logic,
   necessarily re-typed" decision, not an independent second
   implementation.
3. Changed the snapshot-refusal gate to fire only on genuine `FAIL`
   (tracked in a separate `FAILED` string), not on `NEEDS-ARGS`.

**Runner-list audit.** Checked every entry in `RUNNERS` against files on
disk (`[ -f "$f" ]`, re-verified independently via a direct loop before
editing): all nineteen are present under their current names. Nothing
to report — no missing or renamed files in this list.

**Scope note, stated per the task's instruction.** This fix does not
make every runner pass. The runners with no argparse at all (register
L10/L27: `06_, 07_, 08_, 09_, 10_, 11_, 13_, 14_`) still fail — now
correctly bucketed as `FAIL`, not conflated with a healthy `NEEDS-ARGS`
refusal, because their failure is a plain Python `TypeError` from a
missing `load_scenario()` argument (exit code 1, no argparse text), not
the NEEDS-ARGS signature. That is the correct, informative outcome for
scripts already declared dead; repairing them is out of scope here.

`bash -n scripts/run_verification_sweep.sh` passes.

---

## A27 — `scripts/verify_2_6_commit_a.sh`: `set -e` turned the first failure into total silence

**What this script is for.** A regression comparison, not a health
check: run the two baseline runners at a caller-supplied pre-patch ref
and at HEAD, and confirm the optimum hasn't moved (`genset_case=None`
makes selfgen an exact duplicate of unserved energy — the invariant
being protected is "the objective must not move under that
substitution"). Different purpose from A26's harness, and — as
instructed — needed a different fix.

**Defect as found.** Only `--om-case` was ever supplied to
`01_run_baseline.py` / `01b_run_baseline_sliced.py`, both of which now
require 15-18 flags. Under `set -euo pipefail`, the first such failure
(the pre-patch annual run, step 2 of 4) aborted the whole script via
`-e` immediately — before per-runner status was ever printed, and
before the comparison step in step 4 ran at all. The script's entire
signal collapsed to "it exited 1," indistinguishable from a real
regression.

**Fix.**
1. Removed `-e` (`set -uo pipefail` now), but kept explicit fatal-error
   handling for every step that genuinely cannot proceed on failure:
   the dirty-tree guard, the python/pyomo import check, `mkdir -p
   "${OUT}"`, and `git worktree add`, each now with its own `|| {
   echo ...; exit 1; }` (the mkdir and worktree checks are new; the
   dirty-tree and import checks already existed and are unchanged).
2. Added a `REGRESSION_ARGS` array, copied from `run_central_sweep.sh`'s
   own `A_REGRESSION_ANCHOR` invocation — the central set with
   `conversion-case conv_none`, `capital-treatment-case
   pre_f1_overnight`, `dscr-case dscr_regression`, `tariff-case
   tariff_legacy_65m`, `om-case om_none` overridden. Chosen over the
   central set (A26's choice) because this script's own stated purpose
   is reproducing a frozen anchor ("the optimum must not move"), which
   is exactly the regression-anchor use case, not a central-config
   health check — per the task's instruction to choose per invocation
   and state why. All four runner invocations (pre-patch annual/sliced,
   post-patch annual/sliced) use this same set, since all four must be
   comparable to each other.
3. Added `run_and_classify()`, reusing `smoke_all.sh`'s
   PASS/NEEDS-ARGS/FAIL detection (same reasoning as A26 — retyped, not
   imported, for the same bash-sourcing reason). Each of the four runner
   invocations is now wrapped in this function; its exit code is
   captured into a `PREPATCH_OK` / `POSTPATCH_OK` flag rather than left
   to `-e`. File copies are now conditional on the relevant OK flag; the
   final comparison (`scripts/_compare_2_6.py`) only runs if *both*
   sides succeeded, otherwise the script prints an explicit `SKIPPED`
   message naming which side failed and exits 1 — a reported outcome,
   not a silent one.
4. The Step-1 smoke test (pure-Python heredoc, bypasses argparse
   entirely, unaffected by the CLI surface) keeps its original
   abort-on-failure behaviour via an explicit `if [ $? -ne 0 ]; then
   echo "ABORT..."; exit 1; fi` check, now that `-e` no longer does this
   for it implicitly.

**How genuine error detection was preserved.** Nothing that could mask
a real regression was softened: a comparison only ever runs when both
sides are confirmed `PASS`; any other combination (`NEEDS-ARGS` or
`FAIL` on either side) skips the comparison and exits 1, loudly, naming
which side and pointing at its log. What changed is that a failure is
now *attributed* (which of four steps, which log) instead of aborting
opaquely at the first one. The setup steps that truly cannot be
recovered from (dirty tree, missing python/pyomo, un-creatable output
dir, un-creatable worktree) remain hard failures.

**Known limitation, stated rather than solved (per the task's
instruction, documented in the file itself).** `PRE_REF` is caller-
supplied and arbitrary. This script cannot know whether an arbitrary
historical commit's copy of these two runners recognises the regression
flag set above — an older commit may predate some of these arms
entirely. If the pre-patch invocation fails for that reason, it is now
*reported* (as FAIL, since an unrecognised-argument error won't match
the NEEDS-ARGS signature) rather than silently aborting everything — but
no flag combination this script could choose is guaranteed to work
against an arbitrary ref, and this fix does not attempt to solve that;
it only stops one failure from hiding all four outcomes.

`bash -n scripts/verify_2_6_commit_a.sh` passes.

---

## A28 — `docs/methodology.md`: capex-window pairing and consumer count both wrong

**Defect as found.** Lines 645-649 (pre-fix):

```
salvage arbitrage concerns. `npv_capex_report` in
`src/optimize_experiments.py` computes a FULL-HORIZON figure under a
name implying the report window; it is not used by any runner, and the
naming defect is recorded in the register rather than corrected, since
the field has no consumers.
```

Three separate errors, each verified against the live code rather than
against the audit's own summary:

1. **Window pairing is backwards.** `src/optimize_experiments.py`
   (current, lines ~3127-3129):
   ```python
   _report_T = [t for t, y in enumerate(years) if int(y) <= REPORT_END_YEAR]
   npv_capex_report = sum(float(pyo.value(m.DF[t])) * _capex_at(t) for t in _report_T)
   npv_capex_full = sum(float(pyo.value(m.DF[t])) * _capex_at(t) for t in range(len(years)))
   ```
   `npv_capex_report` is truncated to `_report_T` (years ≤
   `REPORT_END_YEAR` = 2045) — it *is* the report-window figure, exactly
   as its name says. `npv_capex_full` is the untruncated, full-2054
   figure. The document had this pairing exactly backwards.
2. **"Not used by any runner" is false.** `grep -rln
   "npv_gross_capex_report_window_usd" scripts/*.py` returns three
   files: `01_run_baseline.py`, `01b_run_baseline_sliced.py`, and
   `29_run_parameter_uncertainty.py`.
3. **Consequently there is no naming defect to record.** The claim that
   "the naming defect is recorded in the register... since the field
   has no consumers" doesn't hold once (1) and (2) are corrected: the
   internal variable name, its dict key, and its truncation all agree
   with each other, and it has three live consumers.

Checked whether any other passage in the document makes a related claim:
`grep -n "npv_capex_report\|npv_gross_capex" docs/methodology.md`
returns three hits, all inside lines 634-649 (the block quoted above).
This is the only passage; the correct figures at lines 634-639
(12.772 bn / 14.350 bn / 1.578 bn / 11.0%; `conv_none` 2.162 bn /
17.182 bn / 12.6%) were not touched.

**Fix — corrected in place, no correction note appended.**

Old (lines 645-649):
```
salvage arbitrage concerns. `npv_capex_report` in
`src/optimize_experiments.py` computes a FULL-HORIZON figure under a
name implying the report window; it is not used by any runner, and the
naming defect is recorded in the register rather than corrected, since
the field has no consumers.
```

New:
```
salvage arbitrage concerns. `npv_capex_report` in
`src/optimize_experiments.py` is the report-window figure its name
states -- truncated to years at or before `REPORT_END_YEAR` -- and
feeds `npv_gross_capex_report_window_usd`; `npv_capex_full` is its
untruncated, full-2025-2054-horizon counterpart, feeding
`npv_gross_capex_full_horizon_usd`. Three runners consume the
report-window field: `01_run_baseline.py`, `01b_run_baseline_sliced.py`,
and `29_run_parameter_uncertainty.py`.
```

**Why this cannot move a solved value.** This is a documentation-only
change to prose describing existing, unmodified code — no file under
`src/` or `scripts/` was touched for this finding, so no computation
changes. `docs/methodology.md` is not read by any runner or solver.

---

## A23 — `docs/limitations_register.md`, L10 amendment

**Amendment made, no renumbering, no new entry.** L10's `Status` line
and a new leading paragraph now record that five of the nine originally-
listed items are repaired; everything below that paragraph (`What`,
`Why unresolved`, `Direction of bias`, `Consequence for planned
analysis`, `What would resolve it`) is left exactly as originally
written, since it remains live and accurate for the four items that
were not touched.

**Which five, and which commits.** Verified via `git log --follow
--oneline -- <file>` and `git show --stat` for each:
- `05_run_fin3_capital_sweep.py`
- `15_run_pol1_ndc_comparison.py`
- `18_run_pareto_frontier.py`
- `19_run_carbon_price_sweep.py`
- `20_run_cf1_solar_yield_sensitivity.py`

All five gained `import argparse` and `from src.cli_arms import
add_standard_arms, standard_arms_kwargs` in commit `ecb7e54`
("Repair nine directly-broken runners: add src/cli_arms.py shared
arm-flag helper; thread required keys through all nine; promote
om_case from literal to flag") — confirmed against the commit's own
`--stat`, which touches exactly these five (of the nine listed here)
plus four others not in L10's original nine (`04_`, `24_`, `25_`,
`99c_`). `18_run_pareto_frontier.py` additionally required commit
`f4dd48d` ("Port both frontiers to build_model_sliced; fix three
sliced-unsafe indexing sites in 18_; add required --baseline-diagnostics
with arm_tag validation") to work against the sliced builder. Confirmed
directly (not just via the commit log) that all five currently import
`argparse` and `src.cli_arms`, and that the remaining four do not.

**The remaining four, confirmed still dead.** `06_run_gas1_shadow_
benchmarks.py`, `07_run_gas2_eaas_gas_relief.py`, and the two `09b_`
files, plus `17_`: `grep -n "cli_arms\|import argparse"` on all of them
returns nothing. Their original L10 text is unchanged.

---

## Encountered but not asked about

- `scripts/09b_run_rel1_mode_sensitivity.py` and
  `scripts/09b_run_rel1_retirement_sensitivity.py` no longer live at
  those bare paths — commit `efa433e` ("Tier A: sliced-aware
  extract_planning_diagnostics; retire 09b x2, defer 17; port
  inventory") moved them to `scripts/retired/`. Similarly,
  `17_run_monte_carlo.py` was moved to `scripts/deferred/` by the same
  commit. L10's "What" paragraph still names them by their original
  bare paths; I left this untouched since the task scoped this
  amendment strictly to the repaired-vs-still-dead distinction, not to
  auditing every path reference in the entry, and the files are
  confirmed still dead regardless of which directory they live in.
- `docs/forensic_audit.md`, `scripts/01_run_baseline.py`,
  `scripts/02_run_eaas.py`, and `scripts/18_run_pareto_frontier.py`
  show as modified in `git status` from before this task started (not
  touched by it) — confirmed still present as pre-existing, unrelated
  modifications, not introduced or altered by this task.
- A19, A20, A22, A24, A25 were explicitly out of scope for this task
  and were not looked at.

---

## Confirmation

`git status --porcelain` after this task's edits:

```
 M docs/forensic_audit.md              <- pre-existing, not touched here
 M docs/limitations_register.md        <- this task (A23, L10 only)
 M docs/methodology.md                 <- this task (A28)
 M scripts/00_build_emissions_cap.py   <- this task (A18)
 M scripts/01_run_baseline.py          <- pre-existing, not touched here
 M scripts/02_run_eaas.py              <- pre-existing, not touched here
 M scripts/02_run_ndc_caps.py          <- this task (A21)
 M scripts/03_run_ndc_eaas.py          <- this task (A21)
 M scripts/18_run_pareto_frontier.py   <- pre-existing, not touched here
 M scripts/run_verification_sweep.sh   <- this task (A26)
 M scripts/verify_2_6_commit_a.sh      <- this task (A27)
?? docs/task_audit_fixes_p2.md         <- this report
```

Nothing was staged, committed, or run — no solver, no runner, no test
suite was executed at any point in this task, only `py_compile`,
`bash -n`, and read-only inspection (`grep`, `git log`, `git show`,
direct JSON checks). STOPPING here per instruction: Zeus reviews `git
diff`, commits, runs both anchors (regression 21,811,042,472.821114 /
45,502.9754 MWh under `om_none`; central 27,954,000,000 approx /
65,612.985 MWh), and then runs the two repaired harnesses
(`run_verification_sweep.sh`, `verify_2_6_commit_a.sh`) to see what they
now report.
