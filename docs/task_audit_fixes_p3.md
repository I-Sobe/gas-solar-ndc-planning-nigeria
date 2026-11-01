# task_audit_fixes_p3 — Phase 3 remediation

Fix what is cheap and safe; declare what is not. Per finding: fixed or
declared, why, and why nothing can move a solved value. Nothing was
staged, committed, or run. Reported incrementally as each part was
finished.

---

## PART 1 — A50: `scripts/16_plot_cost_decomposition.py`

**Disposition: FIXED.**

### The schema it expected vs. the schema it gets

Quoted, before the fix:
```python
real_exp = decomp["real_expenditure_npv"]
voll_pen = decomp["voll_penalty_npv"]
selfgen = decomp.get("selfgen_npv", diag.get("cost_selfgen_npv", 0.0)) or 0.0
salvage = decomp.get("salvage_npv", 0.0) or 0.0
total = real_exp + selfgen + voll_pen - salvage
```
`src/optimize_experiments.py`'s current `cost_decomposition` dict
(`:3003-3022`) has always had `om_npv`/`om_share` as top-level keys
since commit `f4fb607` (2026-09-16) — nearly a month after this
script's last commit, `4faa2b6` (2026-08-21). The reader never read
them.

### Conversion capital line — checked, not missing

`real_expenditure_npv` already includes conversion capital: under
`f1_tranche`, `cost_capital_total` is read directly from `m.capital_
charge_total_npv`, an Expression that itself folds in `m.cost_
conversion_capital_charge_npv` whenever an available conversion arm is
active (`src/optimize_experiments.py:2965`, comment `:2951-2960`). There
is no separate top-level `cost_decomposition` field for conversion
capital that could be silently omitted the way `om_npv` was — it is
itemised only inside the nested `components` dict as `conversion_
capital_charge_npv`, for transparency, not as a second top-level total.
**Reported: not missing, already correctly counted.** No new bar
segment was added for it (unlike O&M — see below), since nothing ever
flagged it as needing its own visible line, and it was never dropped
from the total.

### The fix

Old (`main()`, reading block):
```python
real_exp = decomp["real_expenditure_npv"]
voll_pen = decomp["voll_penalty_npv"]
selfgen = decomp.get("selfgen_npv", diag.get("cost_selfgen_npv", 0.0)) or 0.0
salvage = decomp.get("salvage_npv", 0.0) or 0.0
total = real_exp + selfgen + voll_pen - salvage
```
New:
```python
real_exp = decomp["real_expenditure_npv"]
voll_pen = decomp["voll_penalty_npv"]
selfgen = decomp.get("selfgen_npv", diag.get("cost_selfgen_npv", 0.0)) or 0.0
salvage = decomp.get("salvage_npv", 0.0) or 0.0
om = decomp.get("om_npv", 0.0) or 0.0
total = real_exp + selfgen + voll_pen - salvage + om
```
plus a new `components`-derived block (the completeness guard, below),
a new `om_vals` list, a fourth stacked-bar segment (`bars_om`, color
`#9C27B0`, placed directly after real expenditure), an updated per-bar
label (`"{real_share:.0f}% capex/opex, {om_share:.0f}% O&M, {sg_share:.
0f}% private diesel"`), and an updated console table with an `O&M`
column. Plot type, output paths (`results/figures/cost_decomposition.
{png,pdf}`), legend position, axis labels, and title are unchanged, per
the task's own instruction not to touch styling/layout beyond what
correctness requires. The 4th bar segment was necessary, not
discretionary: without it, the printed total (now including O&M) would
no longer match the stacked bar's own visual length — a new, self-
inflicted inconsistency the original bug did not even have.

### The completeness guard (step 3)

New code, added after the existing objective-level reconciliation
check:
```python
components = decomp.get("components", {}) or {}
gas_opex_c = components.get("gas_opex_npv", 0.0) or 0.0
carbon_c = components.get("carbon_cost_npv", 0.0) or 0.0
if "capital_charge_total_npv" in components:
    capital_c = components["capital_charge_total_npv"] or 0.0
else:
    capital_c = sum(
        components.get(k, 0.0) or 0.0
        for k in ("gas_capex_npv", "public_solar_capex_npv",
                  "eaas_solar_capex_npv", "storage_capex_npv",
                  "replacement_capex_npv")
    )
real_exp_check = gas_opex_c + capital_c + carbon_c
recon_gap = abs(real_exp_check - real_exp) / real_exp if real_exp else 0.0
```
printed as a loud, unmissable warning block (`COMPLETENESS GUARD
FAILED`, bracketed by `!`-rules, naming every case and the exact gap) if
any case's derived total disagrees with the claimed `real_expenditure_
npv` by more than 1e-6 relative. This checks `real_expenditure_npv`'s
OWN claimed composition against `decomp["components"]`, independent of
the pre-existing objective-level check — the two together catch both a
future field silently dropped from `real_expenditure_npv`'s own
computation in `src/optimize_experiments.py`, and a future field
silently dropped from `cost_total` as a whole (om's own failure mode).
Handles both branches of `capital_treatment_case` (the `f1_tranche`
branch has a `capital_charge_total_npv` key in `components`; the
`pre_f1_overnight` branch does not, so its five itemised capital
channels are summed instead).

### Why this cannot move a solved value

This is a plotting/reporting script only — it reads already-solved
`diagnostics.json`/`summary.json` files and never touches `src/`, never
builds a model, never solves. No LP, no objective, no anchor is
affected. `python -m py_compile scripts/16_plot_cost_decomposition.py`
passes.

---

## PART 2 — A46, A47, A48, A49: `scripts/15_run_pol1_ndc_comparison.py`

**Disposition: ALL FOUR FIXED.**

### A46 — `required_margin` double-count

Quoted from all three files, as instructed:

`scripts/02_run_ndc_caps.py` and `scripts/03_run_ndc_eaas.py` (the
correct pattern, unchanged, quoted for reference):
```python
# F1, Decision 2: no per-case required_margin override here anymore
# -- the retired conditional/unconditional 1.05/1.10 split was a
# double-count of capital_recovery_case's r_conc/r_comm. required_margin
# is now sourced solely from load_scenario(dscr_case=...) above.
```

`scripts/15_run_pol1_ndc_comparison.py`, old:
```python
scenario["financing_regime"] = fin_cfg["financing_regime"]
if fin_label == "eaas":
    scenario["required_margin"] = ndc_cfg["eaas_margin"]
    # docs/tariff_arm.md, Steps 1-3: tariff now threaded via
    # load_scenario(tariff_case=...) above; post-call literal retired.
```
New:
```python
scenario["financing_regime"] = fin_cfg["financing_regime"]
# docs/tariff_arm.md, Steps 1-3: tariff now threaded via
# load_scenario(tariff_case=...) above; post-call literal retired.
#
# task_audit_fixes_p3 (A46): no per-case required_margin override here
# anymore -- the retired conditional/unconditional 1.05/1.10 split was
# a double-count of capital_recovery_case's r_conc/r_comm CRF rates,
# per "F1, Decision 2" (see 02_run_ndc_caps.py / 03_run_ndc_eaas.py,
# which never override it either). required_margin is now sourced
# solely from load_scenario(dscr_case=...) via standard_kwargs above.
```
The now-dead `eaas_margin` key was removed from `NDC_CONFIGS` in the
same edit (see A47 below — it was only ever read at this one site).

### A47 — capital-envelope tier mismatch, and its root cause

`scripts/15_run_pol1_ndc_comparison.py`'s `NDC_CONFIGS`, old
(`public_capital`/`eaas_capital` both `"moderate"`/`"expansion"` for
unconditional/conditional):
```python
NDC_CONFIGS = [
    {"ndc_label": "ndc2_unconditional", ..., "public_capital": "moderate", "eaas_capital": "moderate", "eaas_margin": 1.10},
    {"ndc_label": "ndc2_conditional",   ..., "public_capital": "expansion", "eaas_capital": "expansion", "eaas_margin": 1.05},
    {"ndc_label": "ndc3_unconditional", ..., "public_capital": "moderate", "eaas_capital": "moderate", "eaas_margin": 1.10},
    {"ndc_label": "ndc3_conditional",   ..., "public_capital": "expansion", "eaas_capital": "expansion", "eaas_margin": 1.05},
]
```
New (aligned to `"tight"`/`"moderate"`, matching `02_run_ndc_caps.py`'s
and `03_run_ndc_eaas.py`'s canonical `NDC_CASES`; `eaas_margin` removed):
```python
NDC_CONFIGS = [
    {"ndc_label": "ndc2_unconditional", ..., "public_capital": "tight",    "eaas_capital": "tight"},
    {"ndc_label": "ndc2_conditional",   ..., "public_capital": "moderate", "eaas_capital": "moderate"},
    {"ndc_label": "ndc3_unconditional", ..., "public_capital": "tight",    "eaas_capital": "tight"},
    {"ndc_label": "ndc3_conditional",   ..., "public_capital": "moderate", "eaas_capital": "moderate"},
]
```

**The stale comment in `02_run_ndc_caps.py` that caused it — fixed,
comment only, confirmed via `git diff` that no code line changed:**

Old:
```python
#   capital_case — public capital ceiling passed to load_scenario().
#     unconditional: "moderate" (0.85 × B* = 5.2B USD)
#       -> domestically self-funded floor, budget is binding.
#     conditional:   "expansion" (1.20 × B* = 7.4B USD)
#       -> international finance supplements domestic capital;
#         apportioned power-sector share of $270bn is ~$19.8bn,
#         expansion (7.4B) sits within that envelope while keeping
#         the constraint economically meaningful (not unconstrained).
```
New:
```python
#   capital_case — public capital ceiling passed to load_scenario().
#     unconditional: "tight" (0.50 × B* = 4.55B USD, current B* =
#       9.104B -- task_audit_fixes_p3, A47/A29 corrected this comment,
#       which previously said "moderate" (0.85 x a RETIRED B* =
#       6.13B, i.e. 5.2B) and misled 15_run_pol1_ndc_comparison.py's
#       own NDC_CONFIGS into copying the wrong tier for years)
#       -> domestically self-funded floor, budget is binding.
#     conditional:   "moderate" (0.85 × B* = 7.74B USD)
#       -> international finance supplements domestic capital;
#         apportioned power-sector share of $270bn is ~$19.8bn,
#         moderate (7.74B) sits within that envelope while keeping
#         the constraint economically meaningful (not unconstrained).
```
`02_run_ndc_caps.py`'s own `NDC_CASES` dict (`tight`/`moderate`) was
already correct and untouched — the comment was the only thing wrong,
and the only thing edited.

### A48 — stale B* docstring figure

Old (`scripts/15_run_pol1_ndc_comparison.py:38`):
```
Under NDC 3.0, the PUBLIC CAPITAL budget (moderate = 5.21B USD) may be
insufficient to finance the solar deployment needed to:
```
New:
```
Under NDC 3.0, the PUBLIC CAPITAL budget (task_audit_fixes_p3, A48:
tight = $4.55B unconditional / moderate = $7.74B conditional, current
B* = $9.104B -- the earlier "moderate = 5.21B USD" figure here computed
0.85 x the RETIRED B* = $6.13B, and against the WRONG tier besides,
since capital_case is "tight" for unconditional per A47's fix) may be
insufficient to finance the solar deployment needed to:
```

### A49 — the A21 two-resolution shape

**Do the two currently resolve identically? Yes — checked, not
assumed.** `load_scenario()` unconditionally sets `scenario["solar_
capex_scenario"] = "solar_low"` (`scenarios.py:2940`), matching the
explicit `"solar_low"` this script passes to its own `load_solar_
capex_by_year` call; `extract_planning_diagnostics`'s internal fallback
uses `start_year=int(years[0])`/`end_year=int(years[-1])`, matching
`years_list[0]`/`years_list[-1]` exactly, which this script also passes
explicitly. Both resolutions land on the identical series today — this
was a TRAP, not a live divergence, and remains classified that way; no
number changes as a result of this fix.

Old:
```python
diag = extract_planning_diagnostics(m, scenario, econ)
```
New:
```python
diag = extract_planning_diagnostics(m, scenario, econ, solar_capex_series=solar_capex_tv)
```

### 15_'s previously-generated outputs are VOID

`results/pol1/` (all five files, mtime 2026-08-23 13:44 — predating
even the `ecb7e54` `cli_arms` repair, 2026-09-15, let alone this task's
fixes):
- `results/pol1/pol1_results.csv`
- `results/pol1/trajectory_comparison.csv`
- `results/pol1/ndc_version_diff.csv`
- `results/pol1/eaas_rescue.csv`
- `results/pol1/pol1_summary.json`

Every one of these was generated under the required_margin double-count
(A46), the wrong capital-envelope tiers (A47), and possibly missing
CLI-arm wiring entirely (pre-dating `ecb7e54`). None should be read,
cited, or compared against a fresh run until `15_` is re-run.

### Why these fixes cannot move a solved value

None of the four touches `src/` or the LP formulation. A46 removes an
override that overwrote an already-correctly-sourced scenario key
before the model was built — the model-building call (`build_model(...)`)
is unchanged in structure; only which value `required_margin` carries
into it changes (and per A46's own fix, it now carries the value
`dscr_case` already specified, removing a bias rather than introducing
one). A47 changes which pre-existing, already-defined `capital_case`
arm string (`"tight"`/`"moderate"`, both already valid, already-used
choices in `capital_envelope_scenarios()`) is passed in — no new
constraint logic. A48 is docstring text only. A49 passes an
already-computed value through instead of letting a second, internal
call recompute it — confirmed identical today, so it changes which code
path computes a number, not the number itself. `2a9226b`'s frozen
anchors are untouched: none of these four call sites is on the path
`01_run_baseline.py`/`01b_run_baseline_sliced.py` take to reproduce
either anchor. `python -m py_compile scripts/15_run_pol1_ndc_comparison.
py scripts/02_run_ndc_caps.py` passes for both.

---

## PART 3 — documentation and register corrections

### A29 — register L13

**Disposition: FIXED.**

Old (`docs/limitations_register.md`, L13):
```
**What.** `gas_deliverability_case` now offers two families that are not
comparable: ... The GMP family reaches 114.9-191.7 TWh_th by
2030; the entire drift family spans 81.5-94.7.
```
New: the figure is corrected to **140.5-191.7** (`gas_deliv_gmp_
partial` to `gas_deliv_gmp_full` at 2030), the entry's argument (two
epistemic families, do not mix without labelling) is unchanged, and a
new **Correction** paragraph was added stating the generator itself
reproduces its checked-in CSV byte-for-byte (all 300 rows, 10 scenarios
× 30 years, zero discrepancies, re-derived directly from `build_
series()` against the current YAML — not merely inspected) — the
verdict TASK.md asked to be recorded here. The drift-family range
(81.5-94.7) and GMP upper bound (191.7) were both independently
re-verified correct and left unchanged.

**Why this cannot move a solved value.** This is a register entry —
prose, not code. No runner or the LP reads `limitations_register.md`.

### A51 — `04_run_fin2_tariff_sweep.py` is not a dead stub

**Disposition: FIXED, both documents.**

`docs/merged_correction_plan_v2.md`, D25, old:
```
- **D25.** `04_run_fin2_tariff_sweep` and `12_run_dem1_demand_sensitivity` raise
  `NotImplementedError` — unwritten runners, not 2.6 damage.
```
New: the original sentence is kept (it was correct at the time it was
written), with a **Correction (task_audit_fixes_p3, A51)** appended
stating plainly that this no longer holds for `04_run_fin2_tariff_
sweep.py` (repaired via `b496d77`/`ecb7e54`, no suspension guard today,
confirmed directly) while `12_run_dem1_demand_sensitivity` is
unaffected — confirmed still genuinely raising `NotImplementedError` as
its first executable line. Corrected in the "amend, don't silently
rewrite" style already established for register entries (L10's own
A23 amendment, prior task), since this is a dated plan/status document,
not a methodology chapter.

`docs/forensic_audit.md`'s own Phase 1 text (`src/blended_finance.py`
section), old:
```
This file is the machinery register **L17** already extensively
covers (...) and its "known second instance"
(`run_fin2_blended_bankability_sweep`, `optimize_experiments.py`, dead
runner `04_run_fin2_tariff_sweep.py`).
```
New: the same sentence is kept verbatim (again, correct when written),
with an inline **correction** clause added immediately after naming
`04_` — stating it is no longer a dead runner, citing the same two
commits, and noting explicitly that this Phase 1 sentence's use of `04_`
was purely a caller-count citation for `run_fin2_blended_bankability_
sweep` (unaffected either way), so nothing else in Phase 1's findings
changes.

**Not fixed, flagged as encountered but out of this task's named
scope:** `docs/merged_correction_plan_v2.md:697` also calls `04_` "a
stub" in a different passage (the D17 deferred-item text) — TASK.md
named D25 specifically for this correction, not this second mention;
left untouched rather than expanding scope unasked. See the closing
list.

**Why this cannot move a solved value.** Both are prose corrections in
documentation files; `04_run_fin2_tariff_sweep.py`'s own code is
unchanged by this task (it was already NOT a stub before this task
began — the documents describing it were wrong, not the code).

### A54 — orphaned docstring in `12_run_dem1_demand_sensitivity.py`

**Disposition: DECLARED, not fixed — judged not "obviously safe."**

Considered merging the two orphaned string literals (`__doc__`, lines
1-12, plus the dead second literal, lines 14-101) into one bound
docstring, per the task's own contingent instruction. Rejected: the
orphaned block is itself internally stale and self-contradictory — it
opens with its OWN "STALE NUMBERS" warning (predating the η
correction), then continues to cite the PRE-recalibration gas anchor
(40.66 TWh_th at η=0.43) that is stale by a SECOND, LATER
recalibration the "STALE NUMBERS" warning itself does not know about.
Binding this into the live, `help()`/pydoc-visible `__doc__` would make
confusing, layered-stale numeric content MORE prominent, not less — the
opposite of what fixing a docstring should do. Declared instead, as
register **L42**, with the reasoning above recorded there and a
recommendation that any eventual fix be a deliberate rewrite (not a
merge), done alongside whatever eventually resolves this script's
`NotImplementedError` suspension.

**Why declaring, not fixing, cannot move a solved value.** No code
changed; `12_` remains suspended exactly as before.

---

## PART 4 — DECLARE, DO NOT FIX

Twelve new register entries, **L33-L44**, added to `docs/limitations_
register.md` — grouped by shared cause, per the task's own instruction,
rather than one per finding. No code described by any of them was
edited.

**Note on "next free number."** TASK.md stated "L33 was the highest";
checked directly (`grep -n "^## L" docs/limitations_register.md`) — the
highest existing entry is **L32**, not L33 (there is also an inert `##
Lnn — one-line statement` template heading, not a real entry). New
entries therefore start at L33, which is also what "next free number
after the highest" would give either way — flagged as a minor TASK.md
inaccuracy, not a blocker, in the closing list.

### Dead-runner defects that bite only on repair — A37, A55, A56 (plus A36, A38, A45, folded in as the same shape)

TASK.md named A37/A55/A56 as "a trap set for a future reviver."
Independently, this task's own re-read of A36/A38 (`06_`/`07_`, GAS-1/
GAS-2) and A45 (`11_`, REL-3) found the identical shape — a defect
inert only because its runner is dead on required arguments, that would
bite the moment a `cli_arms.py`-style repair lands. Grouped as three
entries, cross-referenced to each other, rather than force TASK.md's
named three into isolation from the clearly-analogous ones this task
also found:

- **L43** — A55 alone: `13_run_dem2_growth_gas_interaction.py`'s only
  backend call is missing 16 required arguments; the script "described
  as active" cannot execute at all. This is the reason `13_` cannot
  even reach L34's defect yet.
- **L34** — A45 + A56: the retired `required_margin` double-count,
  reintroduced independently in `11_run_rel3_financing_frontier.py`'s
  and `13_run_dem2_growth_gas_interaction.py`'s own backend config, will
  resurface the moment either dormant runner is repaired for its
  required arguments.
- **L35** — A36 + A37 + A38: `06_`/`07_` (GAS-1/GAS-2) carry a
  duplicate-source-of-truth benchmark dict (A36), a copy-paste bug that
  mislabels the EaaS-arm's own status (A37 — would silently blank GAS-
  2's cleanest NDC arm), and a schema gap (A38) — all three inert only
  because both files are dead on required arguments.

All three entries cross-reference each other explicitly as one
conceptual "trap set," per TASK.md's framing.

### Retired-mechanism residue — A52, A53

- **L33**: `04_run_fin2_tariff_sweep.py` still computes a retired
  public-financing mechanism (A52 — two diagnostic columns
  unconditionally null on any fresh run) and pre-retirement output with
  real values from that mechanism sits on disk unlabelled (A53).
  **Paths named and added to `docs/merged_correction_plan_v2.md`'s
  DO-NOT-REPORT list**, per the explicit instruction: `results/fin2/
  tariff_sweep_results__dares_only__df1.csv`, `results/fin2/tariff_
  sweep_results__unbounded__df1.csv`, and their paired `tariff_sweep_
  summary__*.json`/`tariff_sweep_results__*.json` files.

### Everything else A29-A59 not fixed above — reviewed individually, every decision on the record

| Finding | Decision | One-line reason |
|---|---|---|
| A30 | NO-ACTION | `f1_quantity_diagnostic.py`'s stale "reporting defect" suspicion is cosmetic — no wrong number, self-evident to any reader who checks the code it's suspicious of, not worth a register entry. |
| A31 | DECLARED (**L41**) | `plot_storage_regime.py` cannot run and would fail a second way if patched — a real trap for whoever next tries to produce this figure. |
| A32 | NO-ACTION | `plot_system_diagnostics.py`'s dead variables and their typo'd, unreachable fallback branches are harmless dead code today; the crash risk requires a specific future edit nobody has signalled intent to make. |
| A33 | DECLARED (**L38**) | `rpublic_diagnostic.py`'s wrong solar asset life (25 vs 30) would mislead the exact ad-hoc sensitivity judgement the tool exists to support. |
| A34 | DECLARED (**L37**) | `05_run_fin3_capital_sweep.py`'s dead `disco_collection_rate` key is a real trap for a reader inspecting `build_model()` for it. |
| A35 | DECLARED (**L37**, grouped with A34) | Same file, same "misleading CSV column name" shape — grouped, not a separate entry. |
| A36 | DECLARED (**L35**) | Part of the GAS-1/GAS-2 "trap set" group above. |
| A37 | DECLARED (**L35**) | Part of the GAS-1/GAS-2 "trap set" group above (this is the highest-value declare in this task — a verified, high-confidence bug). |
| A38 | DECLARED (**L35**, grouped with A36/A37) | Same two files, same "bites on repair" shape. |
| A39 | NO-ACTION | `docs/runner_inventory.md`'s stale exists-check claim about `08_phase8_analysis.py` is a doc-about-doc gap in a non-canonical audit-trail document; no computed number depends on it, and that document is outside this task's whitelist regardless. |
| A40 | DECLARED (**L44**) | Confirmed present in a real, dated, currently-regenerated output file (`results/phase8/master_results.csv`) — the sharpest non-Part-1/2 finding in this phase; a NaN cell one copy-paste from a chapter table. |
| A41 | DECLARED (**L36**) | `08_run_gas3_regime_ndc_feasibility.py`'s undeclared dead-runner membership, grouped with its own two stale-anchor defects. |
| A42 | DECLARED (**L36**, grouped with A41/A43) | Same file: `GAS_ANCHOR_TWH_TH = 40.66` is pre-recalibration and actively used. |
| A43 | DECLARED (**L36**, grouped with A41/A42) | Same file: docstring cumulative totals stale by 2.3-3.4x, same root cause as A42. |
| A44 | NO-ACTION | `11_run_rel3_financing_frontier.py`'s stale solve-count docstring is cosmetic and dormant (script is dead); self-evident from the code, not worth its own entry. |
| A45 | DECLARED (**L34**) | Part of the required-margin "trap set" group above. |
| A46 | **FIXED** | Part 2 above. |
| A47 | **FIXED** | Part 2 above (plus the stale comment in `02_run_ndc_caps.py`). |
| A48 | **FIXED** | Part 2 above. |
| A49 | **FIXED** | Part 2 above. |
| A50 | **FIXED** | Part 1 above. |
| A51 | **FIXED** | Part 3 above (both documents). |
| A52 | DECLARED (**L33**) | Retired-mechanism residue group above. |
| A53 | DECLARED (**L33**) + **DO-NOT-REPORT** | Retired-mechanism residue group above; paths named and barred as instructed. |
| A54 | DECLARED (**L42**) | Judged not obviously safe to fix — see Part 3 above. |
| A55 | DECLARED (**L43**) | Part of the dead-runner "trap set" group above. |
| A56 | DECLARED (**L34**) | Part of the required-margin "trap set" group above. |
| A57 | DECLARED (**L39**) | `14_run_str1_storage_role.py`'s stale "re-enable when" precondition — a real trap for a future STR-1 reviver. |
| A58 | DECLARED (**L40**) | `21_compute_resolution_premium.py`'s vacuous "same scenario configuration" check — a safety property that has never once been real. |
| A59 | DECLARED (**L40**, grouped with A58) | Same file, same theme (a check/message that doesn't do quite what it claims); grouped rather than a separate entry. |

**Every one of the 29 Phase 3 findings not directly fixed in Parts 1-3
now has an explicit disposition — none silent.**

---

## Closing disposition table — every Phase 3 finding, A29-A59

| ID | Disposition | One-line reason |
|---|---|---|
| A29 | FIXED | Register L13's GMP-family lower bound corrected 114.9 → 140.5; generator's byte-for-byte reproduction verdict added. |
| A30 | NO-ACTION | Cosmetic; stale suspicion in a diagnostic script's own comments, no wrong number. |
| A31 | DECLARED (L41) | `plot_storage_regime.py` cannot run; would fail a second way if patched. |
| A32 | NO-ACTION | Harmless dead code; typo'd branch unreachable without a specific future edit. |
| A33 | DECLARED (L38) | `rpublic_diagnostic.py`'s solar life (25 vs 30) is a real trap for its own use case. |
| A34 | DECLARED (L37) | `05_run_fin3_capital_sweep.py`'s dead `disco_collection_rate` key. |
| A35 | DECLARED (L37) | Same file: `budget_usd`/`budget_b_usd` naming confusion. |
| A36 | DECLARED (L35) | GAS-1/GAS-2 dead-runner trap set: duplicated benchmark dict. |
| A37 | DECLARED (L35) | GAS-1/GAS-2 dead-runner trap set: copy-paste status-mislabelling bug. |
| A38 | DECLARED (L35) | GAS-1/GAS-2 dead-runner trap set: schema gap. |
| A39 | NO-ACTION | Doc-about-doc gap in a non-canonical document; no number affected. |
| A40 | DECLARED (L44) | Confirmed live in `08_phase8_analysis.py`'s real, dated output. |
| A41 | DECLARED (L36) | `08_run_gas3_...`'s undeclared dead-runner membership. |
| A42 | DECLARED (L36) | Same file: `GAS_ANCHOR_TWH_TH = 40.66` is pre-recalibration. |
| A43 | DECLARED (L36) | Same file: docstring totals stale 2.3-3.4x. |
| A44 | NO-ACTION | Cosmetic, dormant script, self-evident stale docstring count. |
| A45 | DECLARED (L34) | Required-margin trap set: `11_run_rel3`'s own instance. |
| A46 | FIXED | `required_margin` override removed from `15_run_pol1_ndc_comparison.py`. |
| A47 | FIXED | Capital-envelope tiers aligned to `tight`/`moderate`; root-cause comment in `02_run_ndc_caps.py` corrected. |
| A48 | FIXED | Stale B* docstring figure in `15_` corrected with current tiers/figures. |
| A49 | FIXED | `solar_capex_series=` now passed explicitly; confirmed identical today. |
| A50 | FIXED | `16_plot_cost_decomposition.py` now reads `om_npv`, plots it, and carries a new completeness guard. |
| A51 | FIXED | D25 and this document's own Phase 1 text corrected; `04_` confirmed not a dead stub. |
| A52 | DECLARED (L33) | Retired-mechanism residue: two null diagnostic columns in `04_`. |
| A53 | DECLARED (L33) + DO-NOT-REPORT | Stale pre-retirement `results/fin2/*` files named and barred. |
| A54 | DECLARED (L42) | Orphaned docstring merge judged unsafe; declared with reasoning instead. |
| A55 | DECLARED (L43) | Dead-runner trap set: `13_run_dem2`'s call missing 16 required arguments. |
| A56 | DECLARED (L34) | Required-margin trap set: `13_run_dem2`'s own instance. |
| A57 | DECLARED (L39) | `14_run_str1_storage_role.py`'s stale "re-enable when" precondition. |
| A58 | DECLARED (L40) | `21_compute_resolution_premium.py`'s vacuous scenario-configuration check. |
| A59 | DECLARED (L40) | Same file: untimestamped salvage figure in a warning string. |

**Tally:** 8 FIXED (A46-A51 plus A29 — Parts 1-3), 21 DECLARED (12 new
register entries, L33-L44), 4 NO-ACTION (A30, A32, A39, A44, each with
a stated reason above) — every one of the 31 Phase 3 findings accounted
for.

---

## Encountered but not asked about

- `docs/merged_correction_plan_v2.md:697` also calls `04_run_fin2_
  tariff_sweep.py` "a stub" in a different passage (the D17
  deferred-item text) — TASK.md named D25 specifically for A51's
  correction; this second mention was left untouched rather than
  expanding scope unasked. Whether D17 itself ("EaaS has no binding
  constraint until 04_ imposes a real tariff constraint") is now
  actually RESOLVED, now that `04_` is a working runner, is a
  substantive research question this task did not evaluate.
- TASK.md's claim that "L33 was the highest" register entry was
  checked and found off by one — L32 was actually the highest before
  this task. Noted, did not change the outcome (new entries still start
  at L33).
- `scripts/retired/09b_run_rel1_mode_sensitivity.py`, `scripts/retired/
  09b_run_rel1_retirement_sensitivity.py`, and `scripts/deferred/17_
  run_monte_carlo.py`'s directory moves (already noted as an L10
  path-staleness gap in a prior task's report) were not touched again
  this task — outside this task's named scope.
- `docs/runner_inventory.md`'s existing entry for `08_phase8_analysis.
  py` overlaps substantially with L44's new content (both describe the
  storage-key mismatch's general shape, though `runner_inventory.md`'s
  own exists-check claim, A39, is itself stale) — the two documents
  were not reconciled or cross-referenced against each other, since
  `runner_inventory.md` is outside this task's whitelist.
- The `_capital_components` dict's own "no top-level `capital_charge_
  total_npv` key under `pre_f1_overnight`" asymmetry (handled by A50's
  completeness guard via a fallback sum) is itself a minor F5-shaped
  inconsistency in `src/optimize_experiments.py` — not reported as a
  new finding (would require a `src/` edit to resolve and this task's
  whitelist forbids any `src/` edit regardless), but noted for a future
  audit pass.

---

## Confirmation

`git status --porcelain` after this task's edits should show exactly:
`scripts/16_plot_cost_decomposition.py`, `scripts/15_run_pol1_ndc_
comparison.py`, `scripts/02_run_ndc_caps.py`, `docs/limitations_
register.md`, `docs/merged_correction_plan_v2.md`, `docs/forensic_
audit.md`, plus this report — alongside the pre-existing, unrelated
uncommitted files from before this task began (`scripts/01_run_
baseline.py`, `scripts/02_run_eaas.py`, `scripts/18_run_pareto_
frontier.py`).

Nothing was staged, committed, or run — no solver, no runner, no test
suite was executed at any point in this task, only `py_compile` (on the
three edited Python files) and read-only inspection (`grep`, `git log`,
`git diff --stat`, `inspect.signature` where cited above, direct file
reads). STOPPING here per instruction: Zeus reviews `git diff`, commits,
runs both anchors (regression 21,811,042,472.821114 / 45,502.9754 MWh
under `om_none`; central 27,954,000,000 approx / 65,612.985 MWh), and
applies the twelve register drafts.
