# task_capex_reporting_field — the financing headline the reporting rule names

Branch `phase-fin2-d17`. Model frozen at `2a9226b`; confirmed nothing in
`src/` has moved since the last builder edit
(`git diff 2a9226b..HEAD -- src/optimize_model.py` — empty;
`src/optimize_model_sliced.py` carries only the Part 1B `Expression`
from `task_conversion_capex_exposure`, itself already anchor-verified
per this task's own "Supersedes" note). This task is read-only on
everything except `scripts/01b_run_baseline_sliced.py`, and Part 1 does
not edit even that.

Anchors, not reproduced or perturbed (no solver run):
regression 21,811,042,472.821114 / 45,502.9754 MWh (`om_none`);
central ≈27,954,000,000 / 65,612.985 MWh (`om_central`).

---

## Corrections to TASK.md's "THE GAP" section, checked before treating any of it as established

Every one of TASK.md's four bulleted "verified this session" claims was
re-checked with the exact commands implied. Three of the four do not
hold.

1. **"`grep -rn "npv_gross_capex_report_window_usd" src/ scripts/`
   returns nothing"** — false. Ran it exactly:
   ```
   src/optimize_experiments.py:3284:        "npv_gross_capex_report_window_usd": npv_capex_report,
   scripts/01_run_baseline.py:436:        "npv_gross_capex_report_window_usd": diag.get("npv_gross_capex_report_window_usd"),
   ```
   Two hits, not zero. `git log -S "npv_gross_capex_report_window_usd"
   -- src/optimize_experiments.py` shows this key was introduced by
   `74d34f2` ("Extend model horizon to 2054 as end-effect buffer, plan
   2.5 step 4") — **weeks before any conversion-diagnostics task in
   this sequence**, not a byproduct of the last two tasks' work on
   `_capex_at()`. The field has been computed, under exactly this
   name, and consumed by `01_run_baseline.py`, the whole time.
2. **"`_capex_at` and `extract_planning_diagnostics` have zero callers
   outside `src/optimize_experiments.py` itself"** — false for
   `extract_planning_diagnostics` (the claim is trivially true for
   `_capex_at`, which is a nested closure and cannot be called from
   outside its enclosing function by construction — but that is not
   what matters here). `grep -rln "extract_planning_diagnostics("
   scripts/ src/` returns **fourteen** files outside
   `optimize_experiments.py`: `01_run_baseline.py`, `02_run_eaas.py`,
   `02_run_ndc_caps.py`, `03_run_ndc_eaas.py`,
   `05_run_fin3_capital_sweep.py`, `06_run_gas1_shadow_benchmarks.py`,
   `07_run_gas2_eaas_gas_relief.py`, `15_run_pol1_ndc_comparison.py`,
   `18_run_pareto_frontier.py`, `19_run_carbon_price_sweep.py`,
   `20_run_cf1_solar_yield_sensitivity.py`,
   `28_run_capital_frontier.py`, `deferred/17_run_monte_carlo.py`,
   `retired/09b_run_rel1_mode_sensitivity.py`,
   `retired/09b_run_rel1_retirement_sensitivity.py`.
3. **"`summary__02c17f55.json` has 46 keys and not one contains
   'capex'"** — the file exists (`results/baseline_sliced/
   summary__02c17f55.json`, generated 2026-09-16 23:13, i.e. an actual
   `conv_central` run has now been executed — see below) and does have
   zero numeric capex keys, but it has **32** keys, not 46 (counted
   directly: `json.load(...)`, `len(d)` → 32). The substance of the
   claim (no capex figure in this specific file) holds; the count
   doesn't. Minor, noted for completeness.
4. **"`01b_run_baseline_sliced.py` builds its summary inline and never
   calls `extract_planning_diagnostics`"** — this one is TRUE, verified
   by reading the file in full (Part 1D, below). It is also the one
   claim that matters for this task's actual, narrower scope.
5. Citation error, found while checking the reporting rule itself:
   TASK.md cites "`docs/methodology.md` §434" as the second document
   stating the reporting rule. `docs/methodology.md` is 631 lines and
   contains zero occurrences of `npv_gross_capex_report_window_usd` or
   "financing headline" (`grep` confirmed). The rule as quoted IS real
   and does appear in `docs/frozen_state_2026_09_10.md` §8 (verified,
   quoted below) and in `docs/methodology_demand_horizon.md` (one of
   the six real `docs/` hits) — `methodology.md` appears to be a
   mis-citation for `methodology_demand_horizon.md`.

**What this means for the task's own framing.** "The rule was written
in anticipation of a field that was never built, and was never
followed" is false as a general statement — the field exists, is
built, and `01_run_baseline.py` already writes it to every one of its
summaries. **The real, narrower gap** — confirmed true, and exactly
what this task's whitelist (`scripts/01b_run_baseline_sliced.py`
only) is actually scoped to — is that the ONE standalone runner used
to produce the sliced calibration/backcast baseline (the file everyone
means when they say "the central sliced run," `02c17f55`) does not
produce this field, because it does not call the function that
computes it, for a reason (Part 1A/1D, below) that does not hold up
either. The task as bounded is well-posed even though its background
narrative overstates the problem it solves.

**Process note, stated plainly rather than left implicit**: this is
the fourth task in this sequence whose "established, do not re-derive"
section contained at least one materially false claim, caught before
any code was written against it. Not raised to relitigate the prior
three — each was corrected in its own report — but because a pattern
across four consecutive tasks is itself worth naming once, rather than
re-discovering silently each time.

---

## PART 1A — is `extract_planning_diagnostics` wholly sliced-safe?

**Method**: every direct `m.<attr>[t]` access in the function
(lines 2340-3288) that is NOT already routed through the `_slice_total`
helper was enumerated (`grep -noE 'm\.[A-Za-z_]+\[t[^]]*\]'` over that
line range, hand-filtered to drop two matches that turned out to be
comment prose — `m.selfgen[t]` at line 3194 and `m.hydro_gen[t, .]` at
line 2465, both inside explanatory comments, not code). For every
remaining name, checked its declaration in `optimize_model_sliced.py`
to confirm it carries the SAME index set (`T` only, no slice dimension)
as in the annual builder. Full list and result:

| Attribute | Declared as (sliced builder) | Same shape in both builders? |
|---|---|---|
| `m.DF[t]` | `Param(T, ...)` | yes |
| `m.eaas_subsidy[t]` | `Var(T, ...)` | yes |
| `m.gas_add[t]` | `Var(T, ...)` | yes |
| `m.gas_capacity_mw[t]` | `Expression(T, ...)` | yes |
| `m.land_constraint[t]` | `Constraint(T, ...)` | yes |
| `m.concessional_volume_by_vintage[t]` | `Expression(T, ...)` (f1_tranche only, both builders) | yes |
| `m.solar_capacity_mw[t]` | `Expression(T, ...)` | yes |
| `m.solar_capex_param[t]` | `Param(T, ...)` | yes |
| `m.solar_eaas_add[t]` | `Var(T, ...)` | yes |
| `m.solar_eaas_comm_vol[t]` / `_conc_vol[t]` | `Var(T, ...)` | yes |
| `m.solar_public_add[t]` | `Var(T, ...)` | yes |
| `m.solar_public_comm_vol[t]` / `_conc_vol[t]` | `Var(T, ...)` | yes |
| `m.storage_add[t]` | `Var(T, ...)` | yes |
| `m.storage_capacity_mwh[t]` | `Expression(T, ...)` | yes |
| `m.storage_comm_vol[t]` / `_conc_vol[t]` | `Var(T, ...)` | yes |
| `m.storage_power_mw[t]` | `Expression(T, ...)` | yes |
| `m.emissions_by_year[t]` | `Expression(T, ...)` | yes |
| `m.conversion_overnight_capex_usd_by_year[t]` | `Expression(T, ...)`, `hasattr`-guarded | yes (present only when `_conv`) |
| `m.public_budget_constraint`, `m.concessional_envelope_constraint` | unindexed `Constraint(expr=...)`, read via `m.dual.get(...)` with no subscript | n/a — scalar in both |

**Result: the function does not crash on a sliced model.** Every
operational, genuinely slice-indexed quantity (`m.unserved`,
`m.gas_to_power`, `m.gas_generation`, `m.solar_generation`,
`m.storage_charge`/`storage_discharge`, `m.selfgen`,
`m.convert_generation`) is already read through `_slice_total()` or
the `_converted_generation_total()` helper, both dispatching on the
component's own dimensionality (`m.unserved.dim() > 1`). This is
consistent with, and directly confirmed by, existing precedent already
in production: `scripts/18_run_pareto_frontier.py:387` and
`scripts/28_run_capital_frontier.py` both call
`extract_planning_diagnostics(m, scenario, econ)` on
`build_model_sliced()` results — including under an active conversion
arm, per `18_`'s own header — and this is already committed, working
code (part of `f4dd48d`, "Port both frontiers to build_model_sliced").

**Silently wrong on a sliced model — the outcome asked to be
prioritised. Two fields, one already registered, one not:**

1. **`storage_binding_by_year`'s `"power_limit"` classification**
   (lines 2669-2701) and its rollup `storage_power_limit_years`
   (line 3275-3277). This is **already registered**: `docs/
   limitations_register.md` **L11**, "`storage_binding` diagnostic is
   not meaningful under the sliced builder" — quoted there almost
   verbatim from the in-code comment ("under the sliced builder there
   is no `storage_deployable_hours_per_year` constraint... the
   `power_limit` classification is not meaningful for sliced runs").
   L11 states the field is inert (no runner consumes
   `storage_power_limit_years`) — re-confirmed here (`grep` for the
   field's use as a consumed value: none outside its own definition
   and rollup).
2. **`storage_utilisation_rate`** (lines 3257-3270) — **NOT
   registered anywhere**, despite sharing the IDENTICAL root cause:
   its denominator is `storage_capacity_mwh[T_last] *
   storage_deployable_hours_per_year / 1e6 * len(years)`, the same
   `storage_deployable_hours_per_year` parameter that
   `grep -n "storage_deployable_hours_per_year" src/optimize_model.py
   src/optimize_model_sliced.py` shows is referenced ONLY in
   `optimize_model.py` (the annual builder) — never in any constraint
   in `optimize_model_sliced.py`. So this field is equally meaningless
   on a sliced run, for the same reason L11 gives its sibling. Its one
   live consumer, `diag["storage_utilisation_rate"]` at line 2082
   inside `run_str1_storage_parameter_sweep`, calls `build_model()`
   (annual) exclusively (line 2052) — so, like L11's field, it is
   currently inert on sliced models, just via an uncovered path. Noted
   here as an L11 completeness gap; **not fixed** (outside this task's
   whitelist — `optimize_experiments.py` is not editable here, and this
   task's own bounding says so).

**Neither of these two fields has anything to do with `_capex_at()`,
`npv_capex_report`, or `npv_capex_full`** — confirmed by inspection:
capex construction (lines 3089-3118) reads only `m.DF`,
`solar_public_add`, `solar_eaas_add`, `solar_capex_param`,
`storage_add`, `gas_add`, `conversion_overnight_capex_usd_by_year` —
none of which appear in the storage-binding block, and all of which
are confirmed T-only above. **The capex fields are unaffected by the
one real sliced-safety defect this function carries.**

**The stale comment, again.** Lines 2933-2934 (unchanged, not in this
task's whitelist to fix) still assert: *"confirmed this function is
called only on `build_model()` models, never `build_model_sliced()`,
by checking every one of its ~25 call sites."* This is the same false
claim identified in `task_conversion_diagnostics` Part 1E and now
reconfirmed false a second time by the dimensionality audit above —
the function is not only CALLED on sliced models (18_/28_), it is
CORRECT on them for everything except the two storage-binding fields
just named.

---

## PART 1B — `_capex_at()` and the `npv_capex_report` / `npv_capex_full` construction

Quoted in full, current state (`src/optimize_experiments.py:3089-3118`,
unchanged since `task_conversion_capex_exposure`'s anchor-verified
commit `235431d`):

```python
    def _capex_at(t):
        v = 0.0
        if hasattr(m, "solar_public_add"):
            v += float(pyo.value(m.solar_public_add[t])) * float(pyo.value(m.solar_capex_param[t]))
        if hasattr(m, "solar_eaas_add"):
            v += float(pyo.value(m.solar_eaas_add[t])) * float(pyo.value(m.solar_capex_param[t]))
        if hasattr(m, "storage_add") and econ is not None:
            v += float(pyo.value(m.storage_add[t])) * float(econ["STORAGE_COST_PER_MWH"])
        if hasattr(m, "gas_add") and "gas_capex_per_mw" in scenario:
            v += float(pyo.value(m.gas_add[t])) * float(scenario["gas_capex_per_mw"])
        if hasattr(m, "conversion_overnight_capex_usd_by_year"):
            v += float(pyo.value(m.conversion_overnight_capex_usd_by_year[t]))
        return v

    _report_T = [t for t, y in enumerate(years) if int(y) <= REPORT_END_YEAR]
    npv_capex_report = sum(float(pyo.value(m.DF[t])) * _capex_at(t) for t in _report_T)
    npv_capex_full = sum(float(pyo.value(m.DF[t])) * _capex_at(t) for t in range(len(years)))
```
returned as:
```python
        "npv_gross_capex_report_window_usd": npv_capex_report,
        "npv_gross_capex_full_horizon_usd":  npv_capex_full,
        "report_window_end_year":            int(REPORT_END_YEAR),
        "model_horizon_end_year":            int(years[-1]),
```

**Window**: `_report_T` filters to `int(y) <= REPORT_END_YEAR`.
`REPORT_END_YEAR = 2045` and `MODEL_START_YEAR = 2025`
(`src/scenarios.py:285-287`, checked directly, not inferred from the
name) — so `npv_capex_report` covers exactly **2025-2045**, confirmed
against the code.

**Salvage**: `_capex_at()` sums only `*_add[t] * unit_cost` terms
(plus the conversion Expression, itself the same shape) — no salvage
credit, no netting term of any kind appears anywhere in the function.
**Excluded**, confirmed by absence, not by a flag.

**Gross or net**: every term is `quantity_added × raw unit cost` — the
same comment block above `_capex_at()` (lines 3086-3088) states "Uses
RAW capex for both arms." **Gross**, confirmed.

---

## PART 1C — does the existing field match the reporting rule's definition?

**Rule** (`docs/frozen_state_2026_09_10.md` §8, quoted exactly):
> Report `npv_gross_capex_report_window_usd` (2025-2045, gross, salvage
> excluded) as the financing headline, never `system_cost_npv`.

**Implementation** (Part 1B): 2025-2045 window (verified against
`REPORT_END_YEAR`/`MODEL_START_YEAR`), gross (verified — raw
quantity×cost, no netting), salvage excluded (verified — no salvage
term present at all). **These match exactly.** No discrepancy found —
unlike the previous three tasks in this sequence, this is a clean
result: the rule and the implementation agree, both on the field NAME
(identical string) and on its DEFINITION. There is nothing to report as
a rule/implementation disagreement.

---

## PART 1D — how does `01b_run_baseline_sliced.py` build its summary?

`m`, `scenario`, and `econ` are all in scope at the point the summary
is assembled — `m` from `build_model_sliced(...)` (line 326),
`scenario` from `load_scenario(...)` (line 280), `econ` from
`load_econ(...)` (line 310), all still live local variables when
`summary = {...}` is built at line 478. **Nothing prevents a call to
`extract_planning_diagnostics(m, scenario, econ)` at that point** —
the signature (`def extract_planning_diagnostics(m, scenario,
econ=None, solar_capex_series=None)`) is satisfied by what is already
in scope.

**Why it doesn't call it — quoted, the claim this task's own framing
turns out to rest on**:
```python
    # ------------------------------------------------------------
    # Aggregate sliced operational variables to annual totals.
    # NOTE: extract_planning_diagnostics is annual-only and would crash on the
    # sliced [t,(s,p)] indexing, so the sliced runner builds its diagnostics
    # directly from annual_totals plus the (unchanged) annual variables.
    # ------------------------------------------------------------
    tot = annual_totals(m, years)  # dict: name -> {year: TWh}
```
and, further down, at the concessional-vintage fields:
```python
    # F2 Step 3: concessional-draw fields (envelope resolution, "F2:
    # asset lifetime restored"). This runner does not call
    # extract_planning_diagnostics (annual-only, see the NOTE above), so
    # they were never surfaced here -- add them directly. ...
```
**Per Part 1A, this premise is false** — the function is not
annual-only and would not crash; it has been called successfully on
sliced models (including this exact scenario shape,
`capital_treatment_case='f1_tranche'`, conversion-active) since
`f4dd48d`. This is the historical root cause: `01b_` was written
before that precedent existed, under a then-plausible assumption that
was never revisited once it stopped being true.

**Summary construction, quoted in full** (`scripts/
01b_run_baseline_sliced.py:478-522`):
```python
    summary = {
        "cap_scenario": "baseline_no_policy_SLICED",
        "peak_case": args.peak_case,
        "storage_cc_case": args.storage_cc_case,
        "gas_build_case": args.gas_build_case,
        "new_gas_deliverability": args.new_gas_deliverability,
        "gas_availability_case": args.gas_availability_case,
        "gas_retirement_case": args.gas_retirement_case,
        "gas_capex_case": args.gas_capex_case,
        "ccgt_efficiency_case": args.ccgt_efficiency_case,
        "capital_recovery_case": args.capital_recovery_case,
        "dscr_case": args.dscr_case,
        "us_inflation": args.us_inflation,
        "capital_treatment_case": args.capital_treatment_case,
        "concessional_envelope_case": args.concessional_envelope_case,
        "gas_deliverability_case": args.gas_deliverability_case,
        "conversion_case": args.conversion_case,
        "hydro_case": args.hydro_case,
        "tariff_case": args.tariff_case,
        "om_case": args.om_case,
        "decision_variables": dv,
        "npv_total_cost_usd": npv_total_cost_usd,
        "cumulative_unserved_twh": cumulative_unserved_twh,
        "actual_emissions_tco2_total": actual_emissions_tco2_total,
        "backcast_gas_2025_twh_e": gas_2025,
        "backcast_gas_2025_fleet_twh_e": gas_fleet_2025,
        "backcast_gas_2025_converted_twh_e": gas_conv_2025,
        "concessional_volume_by_vintage_usd": concessional_volume_by_vintage_usd,
        "concessional_share_by_vintage": concessional_share_by_vintage,
        "conversion": _conv_diag,
        "leakage": leak,
        "notes": "Sliced baseline (6 slices). No emissions cap (emissions_cap=None, no emissions_constraint built).",
        "arm_tag": tag,
    }
```
Every quantity is computed inline: `dv` (decision variables) from
direct `pyo.value()` calls on `Var(T)` objects; `npv_total_cost_usd`
from `m.system_cost_npv`; the concessional-vintage pair duplicated
directly from `extract_planning_diagnostics`'s own logic (matching
comment: "matching `src/optimize_experiments.py`'s
`extract_planning_diagnostics`"); `_conv_diag` from
`conversion_diagnostic(m, years, scenario)` (the module-level function
in `optimize_model_sliced.py`, not the one this task is about). **No
capex quantity of any kind is computed anywhere in this file** —
confirmed by `grep -n "capex" scripts/01b_run_baseline_sliced.py`
(checked; not shown here since Part 1 makes no edits, but the search
was run).

I actually confirmed the currently-persisted output empirically, not
just from reading the code: `results/baseline_sliced/
summary__02c17f55.json` (mtime 2026-09-16 23:13 — an actual
`conv_central` sliced run, generated since the last task in this
sequence) has 32 keys, none numeric-capex, matching the code exactly.
Its companion `diagnostics__02c17f55.json` (16 keys) likewise has none.
That same run's `conversion` block, worth recording since it
independently confirms the previous task's prediction to the dollar:
```
"conversion": {
  "A": {"convert_mw": 571.0,  "share_of_nameplate": 1.0, ...},
  "B": {"convert_mw": 3215.0, "share_of_nameplate": 1.0, ...},
  "C": {"convert_mw": 0.0,    "share_of_nameplate": 0.0, ...},
  "total_convert_mw": 3786.0,
  "implied_conversion_capex_usd": 2271600000.0
}
```
`task_conversion_capex_exposure.md`'s Part 2 predicted, before this
run existed, that tiers A and B would convert to full nameplate
(3786 MW total) and C would not, giving ≈$2.2716bn — this run confirms
that prediction exactly, to the dollar, not approximately.

---

## PART 1E — `26_` and `27_`: are they the sliced-standalone precedent?

Checked both directly:

- **`26_run_f3_identification.py`**: imports and calls
  `build_model_sliced` (line 105, 270) but never imports or calls
  `extract_planning_diagnostics` anywhere in the file.
- **`27_run_hydro_minflow_sensitivity.py`**: imports `build_model_
  sliced, annual_totals, conversion_diagnostic` (line 75) and calls
  `build_model_sliced` (line 322) — the same `annual_totals`/
  `conversion_diagnostic` pattern `01b_` itself uses — never imports or
  calls `extract_planning_diagnostics`.

**Neither is the precedent.** Both are further instances of the SAME
pattern as `01b_`: standalone sliced runners that build their own
inline diagnostics rather than calling the shared function — presumably
inheriting the same now-false "annual-only" assumption, though neither
file was read in enough depth here to confirm that's the stated reason
in each case (out of scope: this task's whitelist is `01b_` only).

**The actual precedent — already identified in Part 1A — is `18_
run_pareto_frontier.py` and `28_run_capital_frontier.py`.** Both call
`build_model_sliced()` and then `extract_planning_diagnostics(m,
scenario, econ)` directly on the result, successfully, including under
an active conversion arm (`18_`'s own header names
`conv_central`/`conv_low`/… as live possibilities). This is committed,
working code, not a hypothetical. **Named as requested**: the
precedent exists, it is not `26_`/`27_`, it is `18_`/`28_`.

**Secondary observation, not asked about but adjacent**: `18_` and
`28_` already call `extract_planning_diagnostics` and therefore already
have `npv_gross_capex_report_window_usd` sitting in their own `diag`
dict — but neither one currently reads it into its own output
(`grep -n "npv_gross_capex\|npv_capex" scripts/18_run_pareto_
frontier.py scripts/28_run_capital_frontier.py` → no hits). So even
the two scripts that already have safe access to this field don't
report it. Out of this task's whitelist; noted below.

---

## PART 1F — rename, or a quantity nothing computes under any name?

**Neither, precisely — the real answer has two parts, corrected from
TASK.md's either/or framing:**

- The FIELD NAME `npv_gross_capex_report_window_usd`, with the exact
  definition the reporting rule specifies, **already exists** (Part
  1B/1C) — it is not a rename target, and nothing about its own
  definition needs to change.
- What is missing is not a field but a PRODUCER: `01b_run_baseline_
  sliced.py`'s own summary simply never includes this field (or any
  capex field) at all — not under this name, not under any other name.
  There is no pre-existing "wrong-named" capex field in `01b_`'s
  summary to rename; Part 1D's full quote shows nothing there computes
  a capex quantity, correctly or otherwise.

So Part 2 is not a rename and not a from-scratch derivation of a new
quantity — it is making one specific runner call (or replicate) an
already-correct, already-defined computation that runs safely on
sliced models today, per the precedent in Part 1E.

---

## Recommendation for Part 2 (not started — released separately)

Per Part 1A, `extract_planning_diagnostics` IS wholly sliced-safe for
everything relevant to `npv_gross_capex_report_window_usd` (the two
silently-wrong fields identified do not intersect the capex
computation at all). Per Part 1D, `01b_`'s `m`/`scenario`/`econ` are in
scope and signature-compatible. Per Part 1E, calling it on a sliced,
conversion-active model is already proven safe in committed code
(`18_`/`28_`). **The safe route is calling `extract_planning_
diagnostics(m, scenario, econ)`, not computing the window inline** —
the opposite of what TASK.md's own conditional anticipated, because the
premise that would have triggered the inline-computation fallback (the
function crashing on sliced input) does not hold.

---

## PART 1 — STOP

No edits made. `git status --porcelain` unchanged from before this
report (only the five pre-existing, unrelated uncommitted files from
two tasks ago, plus this new report). Awaiting explicit release for
Part 2 (the fix in `01b_run_baseline_sliced.py`) and Part 3 (register
draft).

---

## ENCOUNTERED BUT NOT ASKED ABOUT

- `18_run_pareto_frontier.py` and `28_run_capital_frontier.py` already
  call `extract_planning_diagnostics` on sliced, conversion-active
  models and therefore already have `npv_gross_capex_report_window_usd`
  available, but neither reads it into its own output rows. A smaller,
  separate gap from this task's — fixing it means editing those two
  files, outside this task's whitelist (`01b_run_baseline_sliced.py`
  only).
- `storage_utilisation_rate` shares `docs/limitations_register.md`
  L11's exact defect (meaningless on a sliced run, same root cause:
  `storage_deployable_hours_per_year` constrains nothing in the sliced
  builder) but is not named in L11's text. Its one live consumer
  (`run_str1_storage_parameter_sweep`) is annual-only, so — like L11's
  own field — it is currently inert on sliced models via a path L11
  doesn't mention. Worth an amendment to L11 itself; not done here
  (`optimize_experiments.py` and the register are both outside this
  task's whitelist).
- `26_run_f3_identification.py` and `27_run_hydro_minflow_
  sensitivity.py` both independently reimplement inline diagnostics
  rather than calling `extract_planning_diagnostics`, on the same
  pattern as `01b_` — not confirmed whether either carries the same
  "would crash" comment or a different stated reason; not read deeply
  enough to say (out of this task's whitelist to fix in any case).
- `results/baseline_sliced/summary__02c17f55.json` and its companion
  `diagnostics__02c17f55.json` are new since the last task in this
  sequence (mtime 2026-09-16 23:13) — an actual `conv_central` run now
  exists on disk, where none did when `task_conversion_capex_exposure`
  was written. Useful for this report (Part 1D); flagged in case its
  existence is itself news.
- `docs/methodology.md` is cited in TASK.md for the reporting rule and
  does not contain it; `docs/methodology_demand_horizon.md` does. Not
  fixed (would mean editing TASK.md or a doc, neither in scope); noted
  so the citation isn't propagated further.

**Nothing here can move a solved value.** Part 1 read files, ran
read-only `git log`/`git diff`, and inspected two already-persisted
JSON result files (`summary__02c17f55.json`, `diagnostics__
02c17f55.json`) that already existed on disk before this session
started — no script, builder, or solver was invoked. `git status
--porcelain` carries no new modifications from this task.

---

## PART 2 — IMPLEMENTATION

Released via a new TASK.md ("produce the financing headline field
(Part 2, RELEASED)"). Its own "WHAT PART 1 ESTABLISHED" section
contains two claims that directly contradict Part 1 above — checked
again, fresh, before writing any code, rather than assumed correct
because they were labelled established.

### Corrections, re-verified against the current code before implementing

1. **"`npv_capex_report` iterates `range(len(years))` — the full
   2025-2054 horizon... The only difference from `npv_capex_full` is
   salvage, not the window."** Backwards. Re-read the live code
   (`src/optimize_experiments.py:3116-3118`, unchanged since `235431d`
   — `git diff 235431d HEAD -- src/optimize_experiments.py` is empty):
   ```python
       _report_T = [t for t, y in enumerate(years) if int(y) <= REPORT_END_YEAR]
       npv_capex_report = sum(float(pyo.value(m.DF[t])) * _capex_at(t) for t in _report_T)
       npv_capex_full = sum(float(pyo.value(m.DF[t])) * _capex_at(t) for t in range(len(years)))
   ```
   `npv_capex_report` is the one filtered to `_report_T` (`<=
   REPORT_END_YEAR` = 2045); `npv_capex_full` is the one using the
   unfiltered `range(len(years))`. This is the exact opposite pairing
   TASK.md states. And neither term differs by salvage — `_capex_at()`
   (quoted in full in Part 1B) contains no salvage term at all, so
   "the only difference... is salvage" is not just misattributed to
   the wrong field, it describes a difference that doesn't exist
   between the two anywhere. This is Part 1C's own finding, reached
   independently last task and re-confirmed here rather than re-argued
   from scratch: **the rule and the implementation do NOT disagree
   about the window** — `npv_capex_report` already covers exactly
   2025-2045. There is no naming defect to report for `npv_capex_report`
   itself.
2. **"`extract_planning_diagnostics` cannot be called on a sliced
   model. Six fields crash... Neither `26_` nor `27_` calls it. There
   is no precedent and no safe route through it."** Re-audited from
   scratch, more exhaustively than Part 1A's already-thorough pass:
   extracted every distinct `m.<name>` token appearing anywhere in the
   function body (`sed -n '2340,3288p' ... | grep -oE 'm\.[A-Za-z_]
   [A-Za-z0-9_]*' | sort -u` — 60 distinct names, not just the ones
   matching a `[t]` bracket pattern) and checked each one either (a) is
   already routed through `_slice_total()`/`_converted_generation_
   total()` (every genuinely slice-indexed operational quantity:
   `gas_generation`, `gas_to_power`, `solar_generation`, `selfgen`,
   `storage_charge`/`discharge`, `unserved`, `convert_generation`), or
   (b) is declared `Param(T)`/`Var(T)`/`Expression(T)`/an unindexed
   scalar `Constraint`/`Expression` in `optimize_model_sliced.py` —
   identical shape to the annual builder. Found **zero** crash-prone
   fields, not six. `26_run_f3_identification.py` and `27_run_hydro_
   minflow_sensitivity.py` are correctly identified as NOT calling it
   (true, per Part 1E) — but the conclusion drawn from that ("no
   precedent") is false: Part 1E already named the real precedent,
   `18_run_pareto_frontier.py` and `28_run_capital_frontier.py`, both
   committed, both calling `extract_planning_diagnostics(m, scenario,
   econ)` successfully on `build_model_sliced()` results, including
   under an active conversion arm. This is not a new finding — it is
   the same finding from last task's Part 1A/1E, re-verified here
   because the new TASK.md restated the opposite as fact.

Given both corrections, Part 2.1's own fallback instruction ("if any
[variable `_capex_at()` touches] is slice-indexed... STOP and report")
is not triggered — but for a reason: none of them are slice-indexed,
confirmed below — and the broader premise that motivated routing this
through `01b_`'s own inline computation rather than the shared function
does not hold either. Implemented anyway, using the safer route,
since that is what the evidence supports; the deviation is explained
in 2.1.

### 2.1 — sliced-safety verification, and why `_capex_at()` was reused via the enclosing function, not imported directly

**Every variable `_capex_at()` reads**, declaration quoted from
`src/optimize_model_sliced.py` (current file, frozen since `2a9226b`):

```python
m.solar_public_add = pyo.Var(T, domain=pyo.NonNegativeReals)     # line 582
m.solar_eaas_add   = pyo.Var(T, domain=pyo.NonNegativeReals)     # line 583
m.solar_capex_param = pyo.Param(T, initialize=capex_init, ...)   # line 457
m.storage_add      = pyo.Var(T, domain=pyo.NonNegativeReals)     # line 590
m.gas_add          = pyo.Var(T, domain=pyo.NonNegativeReals)     # line 588
m.conversion_overnight_capex_usd_by_year = pyo.Expression(T, ...) if _conv  # line 1965, added task_conversion_capex_exposure
m.DF               = pyo.Param(T, initialize=df, ...)            # line 448, discount factor
```

**Every one is `Var(T)`, `Param(T)`, or `Expression(T)` — indexed by
year only.** None is `Var(T, m.S)` or carries any slice dimension.
This is not incidental: `solar_public_add`/`solar_eaas_add`/
`storage_add`/`gas_add` are CAPACITY-ADDITION decisions — how much to
build in a given year — and capacity decisions are annual in BOTH
builders; slicing only ever applies to DISPATCH (how installed capacity
is operated within a year), never to build decisions. So `_capex_at()`
was never at risk of being slice-unsafe — it reads nothing that
slicing could have touched in the first place. **Confirmed, not just
asserted**: `grep -n "m\.solar_public_add\s*=\|m\.solar_eaas_add\s*=\|
m\.storage_add\s*=\|m\.gas_add\s*=" src/optimize_model_sliced.py`
shows each declared exactly once, all `pyo.Var(T, ...)`.

**Why `_capex_at()` was not literally imported.** TASK.md's Part 2.1
asks to "Import it from `src.optimize_experiments` and call it per
year." Checked before attempting this: `_capex_at` is defined at
`src/optimize_experiments.py:3089`, INSIDE `extract_planning_
diagnostics` (which spans lines 2340-3288) — a **nested closure**
capturing `m`, `econ`, and `scenario` from its enclosing function's
local scope. Python has no mechanism to import a name local to a
function body; `from src.optimize_experiments import _capex_at` is not
degraded or awkward here, it is a plain `ImportError` — `_capex_at` is
not a module attribute. Verified this holds by locating every
`def _capex_at` in the file (`grep -n "def _capex_at"` — exactly one,
at line 3089, indented inside the enclosing function) and confirming
no module-level alias exists anywhere (`grep -n "^_capex_at\|^def
_capex_at"` — no top-level match).

**What was done instead**: called the enclosing function,
`extract_planning_diagnostics(m, scenario, econ, solar_capex_series=
solar_capex_tv)`, and read only the two fields needed
(`npv_gross_capex_report_window_usd`, `npv_gross_capex_full_horizon_
usd`) from its return dict. This is the closest possible match to "reuse
`_capex_at()`; do not reimplement it" that is achievable without
editing `src/` (forbidden) or literally re-deriving `_capex_at()`'s
arithmetic a second time inside `01b_` (explicitly the outcome to
avoid — "the defect this project has now closed four times"). It is
also, per the corrected Part 1A/1E finding above, a route already
proven safe in committed code.

### 2.2 — the new fields

`git diff scripts/01b_run_baseline_sliced.py`, reproduced in full since
it is the entire change:

```python
 from src.io import load_econ, load_solar_capex_by_year
 from src.output_naming import hash8_of, append_index
 from src.utils import json_safe
+# task_capex_reporting_field, Part 2: extract_planning_diagnostics() is
+# NOT annual-only (the in-file comment below this import list said so;
+# task_capex_reporting_field Part 1A found that false by dimensionality
+# audit, and 18_run_pareto_frontier.py / 28_run_capital_frontier.py are
+# already-committed, working precedent calling it on sliced,
+# conversion-active models). Imported ONLY to reuse its already-correct
+# npv_gross_capex_report_window_usd / npv_gross_capex_full_horizon_usd
+# fields (built from _capex_at(), which this runner cannot import
+# directly -- it is a closure local to extract_planning_diagnostics(),
+# not a module-level name). Every other field this module computes
+# inline stays inline; this is not a switch to calling the shared
+# function for everything.
+from src.optimize_experiments import extract_planning_diagnostics

 CANONICAL_VOLL = "voll_mid"
 CANONICAL_GENSET = "genset_central"
@@ -472,6 +485,22 @@ def main():
     slice_2025 = slice_detail(m, years, 0)
     slice_terminal_year = slice_detail(m, years, len(years) - 1)

+    # ------------------------------------------------------------
+    # Reporting-window / full-horizon gross capex (task_capex_reporting_
+    # field, Part 2). Calls the shared extract_planning_diagnostics() for
+    # these two fields ONLY, reusing its _capex_at() (solar + storage +
+    # gas + Phase 2.8 conversion, task_conversion_capex_exposure) rather
+    # than re-deriving the same arithmetic here a second time. Passing
+    # solar_capex_tv (already loaded above, same series the model itself
+    # was built with) so this does not silently reload a possibly-
+    # different series under scenario.get("solar_capex_scenario",
+    # "solar_low")'s default.
+    _planning_diag = extract_planning_diagnostics(
+        m, scenario, econ, solar_capex_series=solar_capex_tv
+    )
+    npv_gross_capex_report_window_usd = _planning_diag["npv_gross_capex_report_window_usd"]
+    npv_gross_capex_full_horizon_usd = _planning_diag["npv_gross_capex_full_horizon_usd"]
+
     # ------------------------------------------------------------
     # Save summary + diagnostics
     # ------------------------------------------------------------
@@ -503,6 +532,13 @@ def main():
         "om_case": args.om_case,
         "decision_variables": dv,
         "npv_total_cost_usd": npv_total_cost_usd,
+        # task_capex_reporting_field, Part 2: the financing headline the
+        # reporting rule names (docs/frozen_state_2026_09_10.md §8) --
+        # 2025-2045, gross, salvage excluded -- and its full-2025-2054-
+        # horizon companion. Both from extract_planning_diagnostics()'s
+        # _capex_at(), unchanged; this runner adds no new arithmetic.
+        "npv_gross_capex_report_window_usd": npv_gross_capex_report_window_usd,
+        "npv_gross_capex_full_horizon_usd": npv_gross_capex_full_horizon_usd,
         "cumulative_unserved_twh": cumulative_unserved_twh,
         "actual_emissions_tco2_total": actual_emissions_tco2_total,
         "backcast_gas_2025_twh_e": gas_2025,
```

**Index range**: not hardcoded anywhere in `01b_` — both fields come
from `extract_planning_diagnostics()`'s own `_report_T`/`range(len(
years))` construction (Part 1B), which derives the window from
`REPORT_END_YEAR` and the model's own `years` list, not a literal.
`01b_` contributes zero new arithmetic.

**Discounting**: `m.DF[t]` — the same `Param(T)` Expression `_capex_at`
and every other capital-charge computation in this project already
uses (Part 1B's quote: `sum(float(pyo.value(m.DF[t])) * _capex_at(t)
for t in ...)`), read via `extract_planning_diagnostics()`, not
independently re-derived. `01b_` does not touch discounting at all.

**`Do not rename or alter npv_capex_report or npv_capex_full`**:
not touched — no edit anywhere in `src/`. And per the correction above,
there is no naming defect in either to report: `npv_capex_report`
already matches "report window" exactly.

Syntax-checked: `python -m py_compile scripts/01b_run_baseline_
sliced.py` — OK. Import-checked (module loaded via `importlib.util`
under a non-`__main__` module name, so the `if __name__ ==
"__main__": main()` guard at the bottom never fires — no model was
built, no solver was invoked): resolves cleanly, no circular import,
`extract_planning_diagnostics` binds to the real function object.

### 2.3 — the conversion trace, explicit

1. **Builder Expression** (`task_conversion_capex_exposure`, Part 1B):
   `m.conversion_overnight_capex_usd_by_year[t]`
   (`optimize_model_sliced.py:1965`) = raw overnight conversion capex,
   zero except at `t == 0`, built only when `_conv` (an active
   conversion arm).
2. **`_capex_at()` branch** (`task_conversion_capex_exposure`, Part 2,
   `optimize_experiments.py:3112-3113`): `if hasattr(m,
   "conversion_overnight_capex_usd_by_year"): v +=
   float(pyo.value(m.conversion_overnight_capex_usd_by_year[t]))` —
   folds it into `_capex_at(t)`'s per-year total alongside solar/
   storage/gas.
3. **Per-year discounted sum** (`optimize_experiments.py:3117-3118`,
   unchanged): `npv_capex_report = sum(DF[t] * _capex_at(t) for t in
   _report_T)`; since the conversion contribution is nonzero only at
   `t == 0` (inside `_report_T` — 2025 ≤ 2045), it flows straight into
   both `npv_capex_report` and `npv_capex_full`.
4. **The new field** (`optimize_experiments.py:3284-3285`, unchanged;
   surfaced by this task's Part 2 import, not newly created):
   `"npv_gross_capex_report_window_usd": npv_capex_report`,
   `"npv_gross_capex_full_horizon_usd": npv_capex_full` — now read by
   `01b_run_baseline_sliced.py` and written into its `summary__
   {hash8}.json`.

**How this was confirmed**, not merely traced on paper: the actual
persisted `results/baseline_sliced/summary__02c17f55.json` (a real
`conv_central` run, generated after `task_conversion_capex_exposure`,
inspected in Part 1D) carries `"conversion": {..., "total_convert_mw":
3786.0, "implied_conversion_capex_usd": 2271600000.0}` — an
INDEPENDENT computation (`conversion_diagnostic()`, a different
module-level function, not `_capex_at()`) that arrives at the same
$2,271,600,000 figure `_capex_at()`'s new branch is built to reach
through the Expression chain above. Two different code paths over the
same solved `convert_mw` values agree to the dollar — the strongest
available confirmation short of actually reading `npv_gross_capex_
report_window_usd` off a run made with this task's own change (which
this task's whitelist does not permit running).

### 2.4 — predictions, before any run

All three below use only data that already existed on disk before this
task (`results/baseline_sliced/summary__02c17f55.json`'s
`decision_variables`) plus three read-only, no-solve Python calls
(`load_solar_capex_by_year`, `load_econ`, and reading
`gas_capex_scenarios()`'s dict literal) — the same category of
judgment call as the read-only calls disclosed in
`task_conversion_capex_exposure.md`. No script, builder, or solver was
run.

**1. Expected magnitude of `npv_gross_capex_report_window_usd` at
central.** Computed an UNDISCOUNTED proxy from the actual persisted
per-year build (`solar_add_mw_by_year`, `gas_add_mw_by_year`,
`storage_add_mwh_by_year` from `summary__02c17f55.json`) times the real
unit costs (`solar_capex_by_year` — declining 1.456M/MW in 2025 to
0.563M/MW by 2054; `gas_capex_per_mw` = 1,317,000 for `gas_capex_
central`; `STORAGE_COST_PER_MWH` = 252,000; plus the $2,271,600,000
conversion charge at `t=0`), summed over 2025-2045:
**≈$26.6 billion, undiscounted.** The true (discounted) NPV figure will
be somewhat lower than this — `DF[t] ≤ 1` for every `t`, and most of
this spend occurs in later years of the window as solar volumes ramp —
so I'd expect the actual `npv_gross_capex_report_window_usd` to land
**below $26.6bn**, plausibly in a $16-23bn range depending on the
discount rate actually applied (not independently reconstructed here;
stated as a range rather than a false-precision point figure for that
reason).

**2. Expected difference between the report-window and full-horizon
figures** (the buffer-year, 2046-2054 capital spend "nobody has seen"):
the same persisted decision variables show buffer-year builds
comparable in raw magnitude to the window's (undiscounted buffer
capex ≈$21.3bn vs window's ≈$26.6bn, i.e. buffer ≈80% of window on an
undiscounted basis) — this project's capacity expansion does not slow
down after 2045, only the REPORTING window ends there. In discounted
terms this gap should shrink for two independent reasons pulling the
same direction: buffer years are 21-29 years out (heavier discounting
than the window's 0-20 years) AND buffer-year solar is far cheaper per
MW (0.56-0.6M/MW vs up to 1.456M/MW early in the window) even though
more MW of it is built. **Expect `npv_gross_capex_full_horizon_usd` to
exceed the report-window figure by a smaller proportion than the
undiscounted 80% gap suggests** — a rough expectation of full-horizon
being on the order of 1.3-1.6× the report-window figure, not close to
2×, but this is the least confident of the three predictions since it
depends most on the discount-rate assumption this report did not
reconstruct.

**3. Expected difference between `conv_none` and `conv_central`**: this
is the most confident prediction of the three, because the entire
conversion charge lands at `t == 0` (2025), which is inside every
window regardless of discount-rate assumptions, and `DF[0]` is
conventionally ≈1.0 (base-year, minimal-to-no discounting) under
essentially any standard convention. **Expect
`npv_gross_capex_report_window_usd`(`conv_central`) −
npv_gross_capex_report_window_usd`(`conv_none`) ≈ $2,271,600,000**,
and the identical difference (to within the same rounding) in
`npv_gross_capex_full_horizon_usd`, since 2025 is common to both
windows. This reconfirms `task_conversion_capex_exposure`'s own
prediction, itself already confirmed to the dollar by the persisted
`implied_conversion_capex_usd` figure (Part 1D/2.3 above) — this is now
the same number checked a third independent way.

---

## PART 3 — register draft (next free number: L29, re-confirmed —
`docs/limitations_register.md`'s last applied entry is still L28)

**Draft, NOT applied** (register outside this task's whitelist):

> ## L29 — "Defined, plausible, never read": four instances of the same
> shape, three of them fields nothing consumed, one a field consumed
> under a name that overstated what it measured
>
> **Status:** LIVE for items 1 (component classification) and 4
> (naming precision, now moot per below); RESOLVED for items 2 and 3
> as of `task_capex_reporting_field`.
>
> **What.** Four instances across this project share one shape: a
> quantity was computed, individually plausible, syntactically
> correct, and never raised an error — because nothing was wrong with
> the MODEL, only with whether anything downstream ever read the
> result.
>
> 1. **`STORAGE_OM_PER_MWH_YR = 2000.0`** (`io.py`): zero readers for
>    the project's life; the storage sweep built to probe it ran over
>    a no-op axis as a consequence. (Pre-existing; see `docs/task_om_
>    implementation.md` / register entries around O&M sourcing.)
> 2. **`extract_planning_diagnostics`**: NOT zero callers, corrected
>    from an initial overstatement in this same task sequence — it has
>    14 external callers — but ONE specific caller,
>    `01b_run_baseline_sliced.py` (the runner that produces the
>    reference/calibration sliced baseline everyone means by "the
>    central sliced run"), never called it, on the belief (stated
>    explicitly in its own comments, `task_capex_reporting_field` Part
>    1D) that the function "is annual-only and would crash" on sliced
>    indexing — a belief that had already been false since `f4dd48d`
>    ported `18_`/`28_` to call it successfully on sliced,
>    conversion-active models, and was never revisited. Every field
>    `extract_planning_diagnostics` computes reached `01b_`'s own
>    output not at all, for that one runner, until this task.
> 3. **`npv_gross_capex_report_window_usd`**: computed correctly since
>    `74d34f2` (weeks before this task sequence began) and consumed by
>    `01_run_baseline.py` the entire time — NOT, contrary to this
>    task's own initial framing, a field nothing computed. The real
>    defect was narrower: `01b_`'s summary (see item 2) never carried
>    it, so the ONE sliced calibration reference this project treats as
>    authoritative had no financing-headline figure at all. Closed by
>    `task_capex_reporting_field`, Part 2: `01b_` now surfaces both
>    `npv_gross_capex_report_window_usd` and its full-horizon
>    companion, reusing the existing, already-correct computation
>    rather than re-deriving it.
> 4. **The window-discrepancy claim itself, which turned out not to be
>    real**: an intermediate draft of this task sequence asserted
>    `npv_capex_report` computes the FULL horizon under a name that
>    claims the report window — checked directly against the code
>    (`task_capex_reporting_field`, Part 2 corrections) and found
>    backwards: `npv_capex_report` already truncates to `_report_T`
>    (≤2045); `npv_capex_full` is the untruncated one. Recorded here
>    not as a defect in the MODEL but as a defect in how this task
>    sequence nearly propagated an unverified claim about it into a
>    permanent record — the same failure mode items 1-3 describe
>    (plausible, uncaught, found only by tracing the actual code), one
>    level up, applied to a CLAIM rather than a field.
>
> **The argument.** Items 1-3 share a detection method:
> tracing a specific number back to its producer, not inspecting the
> code for correctness in isolation (the code WAS correct; the absence
> of a consumer, or of a caller, is not a defect a correctness review
> of the function itself would ever surface). Item 4 shares the same
> method applied reflexively: the claim about `npv_capex_report`'s
> window was checked by re-reading the exact lines it described, not
> trusted because it was labelled "established... do not re-derive."
>
> **This entry generalises** `docs/limitations_register.md` **L11**
> ("`storage_binding` diagnostic is not meaningful under the sliced
> builder" — a sibling failure mode, silently WRONG rather than never
> READ, found the same way) and **L28** ("`cost_decomposition` silently
> omitted 6.602 bn of O&M... found by a consuming script's own guard,
> not by inspection" — the precedent this task sequence has now
> explicitly followed three more times). Also touches the still-
> unapplied **L29 draft from `task_conversion_diagnostics.md`** (a
> narrower entry covering two of the same fields from a conversion-
> blindness angle specifically) — that draft and this one describe
> overlapping but not identical territory; **whichever is applied
> first should fold the other in**, since applying both as separate
> numbered entries would duplicate roughly half their content.
>
> **What would resolve it further.** `26_run_f3_identification.py` and
> `27_run_hydro_minflow_sensitivity.py` likely carry the same "would
> crash" assumption `01b_` did (not confirmed — out of this task's
> whitelist to check in depth) and would benefit from the same fix.
> `18_`/`28_` already call `extract_planning_diagnostics` but do not
> surface `npv_gross_capex_report_window_usd` into their own output
> despite having it for free.

---

## SUMMARY: NOTHING IN PART 2 CAN MOVE A SOLVED VALUE

The entire diff (reproduced in full in 2.2) is: one module-level
import of an existing, already-exported function; one call to that
function, made AFTER `solve_model(m, ...)` has already returned
(line 334, unchanged, well before this task's insertion point at line
485+); two new local variables reading two keys out of that function's
return dict; two new keys added to a Python dict (`summary`) that is
JSON-serialised to disk. `extract_planning_diagnostics()` itself
touches only `pyo.value(...)` reads on an already-solved `m` — as
established in `task_conversion_capex_exposure`'s Part 3 argument, and
re-confirmed by this task's own Part 1A dimensionality audit, it
defines no `Var`, `Constraint`, or `Objective` anywhere. Nothing in
this diff can execute before the solve, alter what the solver sees, or
change any decision variable's value. The anchors are unaffected
because the object graph the solver optimises over is unchanged
character-for-character — confirmed by `git diff 235431d HEAD --
src/` being empty for every file besides the two already-committed,
anchor-verified builder/experiments edits from the prior two tasks.

---

## ENCOUNTERED BUT NOT ASKED ABOUT

- `26_run_f3_identification.py` and `27_run_hydro_minflow_
  sensitivity.py` almost certainly carry the same fixable gap as
  `01b_` (no capex reporting field, likely the same "would crash"
  assumption) — not confirmed in depth, and both are outside this
  task's whitelist (`01b_run_baseline_sliced.py` only) regardless.
- `18_run_pareto_frontier.py` / `28_run_capital_frontier.py` already
  call `extract_planning_diagnostics` (committed precedent this task
  relied on) but still do not read `npv_gross_capex_report_window_usd`
  out of the `diag` dict they already have — flagged again from last
  task's report since it remains true and remains out of scope.
  Same for `storage_utilisation_rate` sharing L11's defect
  unregistered (last task's finding, unchanged, `optimize_experiments.py`
  still outside this task's whitelist).
- The still-unapplied L29 draft from `task_conversion_diagnostics.md`
  and this task's own L29 draft (Part 3, above) overlap substantially.
  Whoever applies the register should reconcile them into one entry
  rather than filing both under different numbers — flagged explicitly
  in this draft's own text too.
- `solar_capex_tv` (the series `01b_` already loads to build the model)
  is now passed into `extract_planning_diagnostics(...,
  solar_capex_series=...)` to avoid a second, independent reload of the
  same CSV under a potentially different default
  (`scenario.get("solar_capex_scenario", "solar_low")`) — worth noting
  this parameter exists and matters for consistency, since it would
  have been easy to omit and get a technically-running but subtly
  inconsistent `solar_beats_gas_year` field (unrelated to the capex
  fields this task needed, but part of the same returned dict).

**Nothing was committed or staged.** `git status --porcelain`:
```
 M scripts/01_run_baseline.py        (two tasks ago, uncommitted, untouched here)
 M scripts/02_run_eaas.py            (two tasks ago, uncommitted, untouched here)
 M scripts/02_run_ndc_caps.py        (two tasks ago, uncommitted, untouched here)
 M scripts/03_run_ndc_eaas.py        (two tasks ago, uncommitted, untouched here)
 M scripts/18_run_pareto_frontier.py (two tasks ago, uncommitted, untouched here)
 M scripts/01b_run_baseline_sliced.py (this task, Part 2)
?? docs/task_capex_reporting_field.md
```

**STOP. Do not commit. Zeus reviews `git diff`, commits, and runs both
anchors plus a central run to produce the new fields.**
