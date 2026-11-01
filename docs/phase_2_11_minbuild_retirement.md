# Phase 2.11 — Solar min-build floor and gas retirement arms

Scope: `src/scenarios.py`, `src/optimize_model.py`, `src/optimize_model_sliced.py`,
`src/optimize_experiments.py`, the 21 runner scripts that override
`solar_min_build_mw_per_year`, and the 11 runners already carrying
`--peak-case` / `--storage-cc-case` / `--gas-build-case` /
`--new-gas-deliverability` / `--gas-availability-case`. Diagnose-and-edit
only; nothing was run (no solver, no runners, no smoke suite, no `--help`
invocation). Two arithmetic checks (a Python-level `TypeError` reproduction
and a standalone retirement-schedule calculation) were run outside the
model/solver to verify Step 0 claims — neither builds a scenario, model, or
solves anything.

Not modified, per the task's explicit exclusion list: peak-shape arms,
`par_block`, `within_block_peaking_factor`, `storage_cc_case`,
`gas_availability_case`, `gas_build_case`, the replacement-capex mechanism,
and the firm-capacity constraint's form.

---

## STEP 0 — Inventory (verbatim, as found before any edit in this phase)

### `solar_min_build_default()`, in full

**`src/scenarios.py`, originally lines 684–699:**

```python
def solar_min_build_default() -> float:
    """
    Minimum annual solar build (MW/yr) applied by runners.

    STATUS: [UNSOURCED] -- an arbitrary floor, plan 2.5. Its docstring in
    optimize_model.py claims it prevents "pathological all-delay solutions that
    leave 2025-2030 with no solar". That justification is being tested: the
    2025 backcast is gas 24.57 + hydro 11.47 + baseline solar 0.88 = 36.9
    against demand 37.09, so essentially NO new solar is needed in 2025. If
    removing the floor gives 2025 = 0 MW, that is the CORRECT answer -- a model
    that builds capacity it does not need is the distorted one.

    Consolidated here (Step 3a) so the value lives in one place. Set to 0.0 in
    Step 3b, after byte-identity has verified the consolidation.
    """
    return 0.0
```

Return value: **`0.0`**.

### `solar_min_build_constraint` in both builders, quoted, and what it forces

**`src/optimize_model.py`, originally lines 795–808:**

```python
    # Minimum annual solar build rate.
    # Required when time-varying CAPEX is active: the optimizer sees 2040-2045
    # solar as ~2.4x cheaper than 2025 (discounting × price decline combined),
    # creating a strong incentive to delay all builds. A minimum floor prevents
    # pathological all-delay solutions that leave 2025-2030 with no solar.
    # Default 0 MW/yr preserves backward-compatibility when solar_capex_by_year
    # is not passed. Set to 100 MW/yr when using time-varying CAPEX.
    min_build = scenario.get("solar_min_build_mw_per_year", 0.0)
    if min_build > 0:
        m.solar_min_build_constraint = pyo.Constraint(
            T,
            rule=lambda m, t:
                m.solar_public_add[t] + m.solar_eaas_add[t] >= min_build
        )
```

**`src/optimize_model_sliced.py`, originally lines 838–844 (no preceding
comment block; the constraint body is otherwise identical):**

```python
    min_build = scenario.get("solar_min_build_mw_per_year", 0.0)
    if min_build > 0:
        m.solar_min_build_constraint = pyo.Constraint(
            T,
            rule=lambda m, t:
                m.solar_public_add[t] + m.solar_eaas_add[t] >= min_build
        )
```

**What it forces:** when `min_build > 0`, it forces `solar_public_add[t] +
solar_eaas_add[t] >= min_build` in **every** model year `t` — a flat
per-year floor on new solar additions (public + EaaS combined), applied
uniformly across the whole horizon regardless of whether that year's
system actually needs new solar.

### The missing-parentheses bug at (then) `optimize_experiments.py:1589`

**Confirmed.** The line read:

```python
scenario["solar_min_build_mw_per_year"] = solar_min_build_default
```

— no `()`. This assigns the **function object** `solar_min_build_default`
itself, not its return value `0.0`.

**What the constraint does when handed a function object:** `min_build =
scenario.get("solar_min_build_mw_per_year", 0.0)` retrieves the function
object (the `.get()` default of `0.0` never applies, since the key is
present). The very next line, `if min_build > 0:`, then compares a function
object to an `int` with `>`. Verified directly (not via the model):

```
>>> def f(): return 0.0
>>> if f > 0: pass
TypeError: '>' not supported between instances of 'function' and 'int'
```

So it does not silently misbehave or build a wrong constraint — it
**hard-crashes** `build_model()` with a `TypeError` at that `if` statement,
before `m.solar_min_build_constraint` is ever created. In practice this
line was already unreachable: the `load_scenario()` call inside the same
function (`run_str1_storage_parameter_sweep`, lines 1575–1584 pre-edit) is
also missing `peak_multiple_case` / `reserve_margin` / `storage_cc_case` /
`gas_build_case` / `gas_availability_case` (all required since 2.6/2.9/2.10)
and would raise a `TypeError` from `load_scenario()` itself first. This bug
was latent, not currently triggerable.

### `gas_baseline_retirement_start_year` / `gas_baseline_retirement_mw_per_year`, quoted, and every consumer

**`src/scenarios.py`, originally lines 1467–1471 (inside `load_scenario()`'s
returned scenario dict — comment included verbatim):**

```python
        # Brownfield retirement proxy: linear retirement from 2035.
        # (Comment corrected: 680 MW/yr x 11 years = 7,480 MW retired by 2045
        #  from a 10,947 MW gas-only base -- NOT the whole fleet.)
        "gas_baseline_retirement_start_year": 2035,
        "gas_baseline_retirement_mw_per_year": 680.0,
```

Consumers (direct readers of these two scenario-dict keys):

- **`src/optimize_model.py`, originally lines 379–391 (`gas_cap_rule`):**
  ```python
      retirement_start = scenario.get("gas_baseline_retirement_start_year", None)
      retirement_rate = scenario.get("gas_baseline_retirement_mw_per_year", 0.0)

      def gas_cap_rule(m, t):
          year_t = int(years[t])
          # Cumulative retirement of baseline fleet
          if retirement_start is not None and year_t >= retirement_start:
              retired = retirement_rate * (year_t - retirement_start + 1)
              retired = min(retired, gas_baseline)  # cannot retire more than exists
          else:
              retired = 0.0
          return gas_baseline - retired + sum(m.gas_add[k] for k in range(0, t + 1))
  ```
  This feeds `m.gas_capacity_mw`, the cumulative-capacity expression
  consumed by `gas_capacity_constraint`, `firm_capacity_constraint`,
  `land_constraint`, `gas_new_capacity_cap_constraint`, and
  `gas_exempt_fuel_allowance_twh_th`.
- **`src/optimize_model_sliced.py`, originally lines 483–493:** identical
  role (`retired = min(retirement_rate * (year_t - retirement_start + 1),
  gas_baseline)`, inlined rather than a two-line assignment).
- **`scripts/09b_run_rel1_retirement_sensitivity.py`, line 122:** a
  freestanding sensitivity script that hand-overrides
  `scenario["gas_baseline_retirement_mw_per_year"]` directly (sweeping
  0/340/680/1360 MW/yr) — **not** one of the 11 runners in scope for Step
  3's CLI wiring, and its own `load_scenario()` call is already missing
  the same required kwargs as the `optimize_experiments.py` sites above
  (see "Known stale/dead code" below). Left untouched.

No other file in the active repo (`src/`, `scripts/`, `tests/`) references
either key — confirmed by a repo-wide grep excluding the nested
`thesis-code/` copy and the `.claude/worktrees/` copy.

### Year the baseline fleet reaches zero, under the pre-2.11 schedule

With `gas_baseline_mw = 10947`, `gas_baseline_retirement_mw_per_year =
680.0`, `gas_baseline_retirement_start_year = 2035` (and `gas_build_case`
not adding new capacity), `retired(year_t) = min(680 * (year_t - 2034),
10947)`:

| year | remaining MW |
|---|---|
| 2049 | 747 |
| 2050 | 67 |
| **2051** | **0** (retired = min(680×17, 10947) = 10947) |

**The baseline fleet reaches zero in 2051** — 3 years before the model
horizon ends (`MODEL_END_YEAR = 2054`), confirming the task context's own
figure. Verified with a standalone arithmetic check, not a model run.

---

## STEP 1 — Retire the floor override

### Files changed (21 runner scripts + `src/optimize_experiments.py`)

Every file that set `scenario["solar_min_build_mw_per_year"] =
solar_min_build_default()` (or, in one case, the buggy unparenthesized
form) had that assignment deleted and its `solar_min_build_default` import
removed. `load_scenario()` already returns `0.0` for this key (unchanged
by this phase), so removing the line restores that sourced default rather
than asserting new behavior. Where a comment explained the override's
purpose, that comment was replaced with a one-line pointer to this
document rather than left describing deleted code:

```diff
-    # Activate minimum build floor when time-varying CAPEX is in use.
-    # This prevents, the optimizer from delaying all solar to the cheapest years
-    # (2040-2045) creating unrealistic 2025-2030 supply gaps.
-    scenario["solar_min_build_mw_per_year"] = solar_min_build_default()
+    # solar_min_build_mw_per_year override to solar_min_build_default()
+    # REMOVED (2.11, Step 1) -- [UNSOURCED] (plan 2.5); load_scenario()'s
+    # own 0.0 default now applies (see docs/phase_2_11_minbuild_retirement.md).
```

(exact surrounding text varied slightly per file — some had no preceding
comment, one had two occurrences in the same file). Files changed:

1. `scripts/01_run_baseline.py`
2. `scripts/01b_run_baseline_sliced.py`
3. `scripts/02_run_eaas.py`
4. `scripts/02_run_ndc_caps.py`
5. `scripts/03_run_ndc_eaas.py`
6. `scripts/05_run_fin3_capital_sweep.py`
7. `scripts/06_run_gas1_shadow_benchmarks.py`
8. `scripts/07_run_gas2_eaas_gas_relief.py` (two occurrences, both removed)
9. `scripts/09b_run_rel1_mode_sensitivity.py`
10. `scripts/09b_run_rel1_retirement_sensitivity.py`
11. `scripts/15_run_pol1_ndc_comparison.py`
12. `scripts/17_run_monte_carlo.py`
13. `scripts/18_run_pareto_frontier.py`
14. `scripts/19_run_carbon_price_sweep.py`
15. `scripts/20_run_cf1_solar_yield_sensitivity.py`
16. `scripts/22_salvage_sensitivity.py`
17. `scripts/23_run_selfgen_frontier.py`
18. `scripts/24_run_leakage_arm.py`
19. `scripts/25_run_obligation_trajectory.py`
20. `scripts/99c_run_genset_arm_sensitivity.py`
21. `scripts/probe_2_6_structure.py`
22. `src/optimize_experiments.py` (four call sites: lines originally 1209,
    1589, 2478, 2725 — see "Why all four, not just 1589" below)

Verified after editing: `grep -rn "solar_min_build_default()" scripts/*.py`
returns zero matches (only explanatory comments containing the name
remain); the same check on `src/optimize_experiments.py` returns zero
matches outside comments. All 27 touched files pass `python3 -m
py_compile`.

### Why all four `optimize_experiments.py` sites, not just the named 1589

The task named only `optimize_experiments.py:1589` explicitly (the
parentheses bug), and separately requires the docstring note "as of 2.11
no runner calls it" to be added to `solar_min_build_default()`. Three
other identical call sites existed in the same file (`run_dem1_demand_
sensitivity`, `run_rel1_feasibility_matrix`, `run_gas_regime_ndc_matrix`),
each calling `solar_min_build_default()` correctly (with parentheses).
Leaving those three in place would make the required docstring note false.
All four were removed for internal consistency, and each removal is
commented in place with a note that the `load_scenario()` call in that
same function is *also* already missing `peak_multiple_case` /
`reserve_margin` / `storage_cc_case` / `gas_build_case` /
`gas_availability_case` (required since 2.6/2.9/2.10) — meaning all four
of these library functions were already dead code (they raise inside
`load_scenario()` before ever reaching the min-build line, or now-would
also raise from `build_model()`'s own presence guards), independent of
this phase. This diagnosis is reported, not silently acted on beyond
deleting the four identical assignments the task's own success condition
requires deleted.

### Missing-parentheses fix

The buggy line (`scenario["solar_min_build_mw_per_year"] =
solar_min_build_default` with no `()`, inside
`run_str1_storage_parameter_sweep`) was fixed by removing the assignment
entirely, consistent with every other site in this step — not by adding
the missing parentheses, since the whole line is now dead per Step 1's
instruction.

### `solar_min_build_default()` retained, with the required note

**`src/scenarios.py`, function body unchanged except for an appended
docstring paragraph:**

```diff
     Consolidated here (Step 3a) so the value lives in one place. Set to 0.0 in
     Step 3b, after byte-identity has verified the consolidation.
+
+    2.11, Step 1: as of this phase, no runner calls this function any more --
+    every scenario["solar_min_build_mw_per_year"] = solar_min_build_default()
+    override has been deleted (they forced this [UNSOURCED] floor over
+    load_scenario()'s own sourced 0.0 default). The firm-capacity constraint
+    (2.6, F0 Step 2.3) now performs the role this floor was invented for: it
+    already forces enough capacity to be built to cover peak demand with a
+    reserve margin, which is a sourced, dimensioned requirement -- unlike this
+    arbitrary MW/yr floor. Left in place, unchanged, rather than deleted, so
+    the STATUS: [UNSOURCED] history above stays attributable. See
+    docs/phase_2_11_minbuild_retirement.md.
     """
     return 0.0
```

`solar_min_build_default()` itself was **not** deleted, per the task's
explicit instruction.

---

## STEP 2 — `gas_retirement_case` as a required swept arm (`src/scenarios.py`)

**New sentinel**, mirroring `_GAS_AVAILABILITY_REQUIRED`:

```python
# Same mechanism, same reason (2.11, Step 2): gas_baseline_retirement_mw_per_year
# was a bare, unsourced 680.0 MW/yr ("brownfield retirement proxy") that drives
# the entire baseline gas fleet to zero by 2051 and now dominates the
# late-horizon result -- a silent default here would keep doing that on an
# unstated assumption. See gas_retirement_scenarios().
_GAS_RETIREMENT_REQUIRED = object()
```

**New parameter on `load_scenario()`** (after `gas_availability_case`):

```diff
     gas_availability_case=_GAS_AVAILABILITY_REQUIRED,
+    gas_retirement_case=_GAS_RETIREMENT_REQUIRED,
     gas_new_efficiency: float | None = None,
```

**New docstring paragraph:**

```python
    gas_retirement_case : key of gas_retirement_scenarios() -- REQUIRED, no
                        default (2.11, Step 2, mirrors genset_case/
                        peak_multiple_case/storage_cc_case/
                        gas_availability_case discipline). One of
                        'gas_retire_none' / 'gas_retire_proxy' /
                        'gas_retire_sourced'. Sets BOTH
                        gas_baseline_retirement_mw_per_year and
                        gas_baseline_retirement_start_year -- the latter is
                        now settable PER ARM rather than fixed at 2035.
                        'gas_retire_sourced' currently always raises:
                        SOURCE: [PENDING] -- see gas_retirement_scenarios().
```

**New arm function**, inserted immediately after `gas_availability_scenarios()`:

```python
def gas_retirement_scenarios() -> dict[str, dict[str, float | int] | None]:
    """
    Gas baseline fleet retirement arm (2.11, Step 2). Mirrors genset_case/
    peak_multiple_case/storage_cc_case/gas_availability_case discipline:
    REQUIRED, no default -- see the _GAS_RETIREMENT_REQUIRED sentinel above
    load_scenario().

    Each non-None arm is a {"mw_per_year": ..., "start_year": ...} pair,
    feeding gas_baseline_retirement_mw_per_year and
    gas_baseline_retirement_start_year on the returned scenario dict --
    gas_baseline_retirement_start_year is deliberately settable PER ARM
    here (2.11, Step 2), not hard-coded at a single 2035 for every case.
    Consumed by gas_cap_rule in both build_model() and
    build_model_sliced(), which retire scenario['gas_baseline_mw'] linearly
    from start_year at mw_per_year, capped so the fleet cannot retire below
    zero.

        gas_retire_none    (mw_per_year=0.0, start_year=None): no baseline
            retirement -- a DECLARED UPPER BOUND on the surviving fleet, not
            a forecast. start_year is None because there is no retirement
            event to date from.
        gas_retire_proxy   (mw_per_year=680.0, start_year=2035): the pre-2.11
            "brownfield retirement proxy" -- unsourced, retained as an
            explicit named arm so every pre-2.11 result stays attributable
            rather than orphaned. SOURCE: [PENDING].
        gas_retire_sourced (None): SOURCE: [PENDING]. Raises if selected.
            Intended to be derived per-plant from commissioning dates and an
            assumed technical life, once sourced.
    """
    return {
        "gas_retire_none":    {"mw_per_year": 0.0, "start_year": None},
        "gas_retire_proxy":   {"mw_per_year": 680.0, "start_year": 2035},
        "gas_retire_sourced": None,
    }
```

**New validation block**, inserted right after the `gas_availability_case`
validation, before `years = planning_horizon(...)`:

```python
    if gas_retirement_case is _GAS_RETIREMENT_REQUIRED:
        raise TypeError(
            "load_scenario() requires gas_retirement_case (2.11, Step 2). "
            "Pass one of 'gas_retire_none' / 'gas_retire_proxy' / "
            "'gas_retire_sourced' (see gas_retirement_scenarios()). There "
            "is no default: omitting it would retire the gas baseline "
            "fleet on an unstated, unsourced schedule with no error, "
            "warning, or visible signature in the output."
        )
    if gas_retirement_case not in gas_retirement_scenarios():
        raise ValueError(f"Unknown gas_retirement_case: {gas_retirement_case}")
    _gas_retirement_arm = gas_retirement_scenarios()[gas_retirement_case]
    if _gas_retirement_arm is None:
        raise ValueError(
            f"gas_retirement_case='{gas_retirement_case}' is SOURCE: "
            "[PENDING] (2.11, Step 2) -- intended to be derived per-plant "
            "from commissioning dates and an assumed technical life, and "
            "has not been sourced. Select 'gas_retire_none' (declared "
            "upper bound on the surviving fleet) or 'gas_retire_proxy' "
            "(the pre-2.11 unsourced proxy, retained for attributability) "
            "until sourced -- see gas_retirement_scenarios()."
        )
    _gas_retirement_mw_per_year = _gas_retirement_arm["mw_per_year"]
    _gas_retirement_start_year = _gas_retirement_arm["start_year"]
```

**Scenario-dict entries replaced** (the two hard-coded literals are now
resolved from the arm, and `gas_retirement_case` itself is recorded):

```diff
         "gas_baseline_mw": 10947,
-        # Brownfield retirement proxy: linear retirement from 2035.
-        # (Comment corrected: 680 MW/yr x 11 years = 7,480 MW retired by 2045
-        #  from a 10,947 MW gas-only base -- NOT the whole fleet.)
-        "gas_baseline_retirement_start_year": 2035,
-        "gas_baseline_retirement_mw_per_year": 680.0,
+        # Brownfield retirement schedule: SOURCE: gas_retirement_case arm
+        # (2.11, Step 2) -- see gas_retirement_scenarios() for the
+        # sourced/pending breakdown per case. gas_baseline_retirement_start_year
+        # is settable PER ARM (2.11, Step 2), not fixed at 2035 for every case.
+        "gas_retirement_case": gas_retirement_case,
+        "gas_baseline_retirement_start_year": _gas_retirement_start_year,
+        "gas_baseline_retirement_mw_per_year": _gas_retirement_mw_per_year,
```

`"gas_retirement": gas_retirement_case` was also added to the `labels`
dict, mirroring `gas_availability`.

**Builder-side presence validation**, added to both `src/optimize_model.py`
and `src/optimize_model_sliced.py`, immediately after the existing
`gas_availability_case` guard:

```python
    if scenario.get("gas_retirement_case", None) is None:
        raise ValueError(
            "scenario['gas_retirement_case'] is missing or None (2.11, "
            "Step 2). Required so the gas fleet's retirement schedule has "
            "a traceable, named choice-set -- see "
            "scenarios.gas_retirement_scenarios()."
        )
```

---

## STEP 3 — Wire and tag (11 runners)

Each of the following 11 runners received the same four-part change,
mirroring exactly how `--gas-availability-case` was wired in 2.10:

1. `GAS_RETIREMENT_CASE_CHOICES = ("gas_retire_none", "gas_retire_proxy", "gas_retire_sourced")`
   added next to `GAS_AVAILABILITY_CASE_CHOICES`.
2. `--gas-retirement-case` added as a required `argparse` argument,
   immediately after `--gas-availability-case`.
3. `gas_retirement_case=args.gas_retirement_case,` added to every
   `load_scenario(...)` call site (threaded through helper functions
   where present).
4. The new arm appended to every output tag / summary dict / JSON row
   that already carried `gas_availability_case`.

Files: `scripts/01_run_baseline.py`, `scripts/01b_run_baseline_sliced.py`,
`scripts/02_run_eaas.py`, `scripts/02_run_ndc_caps.py`,
`scripts/03_run_ndc_eaas.py`, `scripts/04_run_fin2_tariff_sweep.py`,
`scripts/22_salvage_sensitivity.py`, `scripts/23_run_selfgen_frontier.py`,
`scripts/24_run_leakage_arm.py`, `scripts/25_run_obligation_trajectory.py`,
`scripts/99c_run_genset_arm_sensitivity.py`.

Helper-function threading beyond the generic four-part change:

- **`scripts/22_salvage_sensitivity.py`**: `run_arm(...)` gained the
  parameter; its call site in `main()` updated.
- **`scripts/23_run_selfgen_frontier.py`**: both `make_scenario(...)` and
  `solve_point(...)` gained the parameter.
- **`scripts/24_run_leakage_arm.py`** and
  **`scripts/25_run_obligation_trajectory.py`**: `solve_one(...)` gained
  the parameter and it is folded into each function's `base` dict, so
  every output row is tagged without a separate edit.
- **`scripts/99c_run_genset_arm_sensitivity.py`**: `solve_one(...)` gained
  the parameter; also added to the run-summary `print()` line.
- **`scripts/02_run_ndc_caps.py`** / **`scripts/03_run_ndc_eaas.py`**: no
  helper function — wired directly inside the per-NDC-case loop; the
  output directory name and `summary.json` dict both extended.
- **`scripts/04_run_fin2_tariff_sweep.py`**: only `output_tag` needed
  extending (no per-case summary dict, per the 2.10 report's notes on this
  file).

Verified with `grep -c` per file: `gas_availability_case` and
`gas_retirement_case` occur exactly the same number of times in every one
of the 11 files (4, 4, 4, 4, 4, 3, 6, 8, 6, 6, 7 respectively), confirming
symmetric wiring. All 11 files pass `python3 -m py_compile`.

---

## STEP 4 close-out

### Zero-fleet year under each retirement arm

| Arm | mw_per_year | start_year | Zero-fleet year |
|---|---|---|---|
| `gas_retire_none` | 0.0 | `None` | **Never** — retirement never triggers (rate is 0; `start_year=None` also disables the `year_t >= retirement_start` branch). Declared upper bound on the surviving fleet. |
| `gas_retire_proxy` | 680.0 | 2035 | **2051** — `retired(2050) = 10,880` (67 MW remain); `retired(2051) = min(11,560, 10,947) = 10,947` (0 MW remain). Within the model horizon (ends 2054). |
| `gas_retire_sourced` | — | — | Not computable — `gas_retirement_scenarios()["gas_retire_sourced"]` is `None`; selecting this arm raises in `load_scenario()` before any schedule exists. |

### Known stale/dead code, flagged, not fixed (outside this phase's scope)

1. `scripts/09b_run_rel1_retirement_sensitivity.py` hand-overrides
   `gas_baseline_retirement_mw_per_year` directly, duplicating (and
   predating) the purpose of the new `gas_retirement_case` arm. It is not
   one of the 11 runners named in Step 3, and its own `load_scenario()`
   call is already missing `peak_multiple_case` / `reserve_margin` /
   `storage_cc_case` / `gas_build_case` / `gas_availability_case` (all
   required since 2.6/2.9/2.10) — already non-functional independent of
   this phase.
2. `src/scenarios.py`, `gas_ndc_17gw_trajectory_mw()` docstring and its
   `gas_build_case='gas_ndc_17gw'` error message (both untouched, per the
   "do not modify gas_build_case" exclusion) still say "the current 11,000
   MW baseline fleet" / "17,000 MW by 2035" — already flagged as stale in
   the 2.10 report (`docs/phase_2_10_gas_availability.md`) against the
   corrected `gas_baseline_mw=10947`; unrelated to this phase's changes.
3. `scripts/verify_2_6_commit_a.sh` and `tests/test_run.py` were already
   failing before this phase (missing several other now-required
   `load_scenario()` arguments, flagged in the 2.10 report). Neither calls
   `solar_min_build_default()`, so Step 1 does not change their status.

### Confirmation

- Every `scenario["solar_min_build_mw_per_year"] =
  solar_min_build_default()` override (21 runner scripts, one with two
  occurrences, plus 4 sites in `src/optimize_experiments.py`) has been
  deleted. `grep -rn "solar_min_build_default()" scripts/*.py
  src/optimize_experiments.py` returns zero matches outside comments.
- The missing-parentheses bug is fixed by removal, consistent with the
  rest of Step 1.
- `solar_min_build_default()` remains in place, unchanged except for the
  required docstring note.
- `gas_retirement_case` is REQUIRED with no default at both the
  `load_scenario()` level (raises `TypeError`/`ValueError`) and the
  builder level (`build_model`/`build_model_sliced` raise `ValueError` on
  a hand-built scenario missing the key), and is threaded as a required
  CLI argument through all 11 runners, tagged into every output artifact
  those runners write.
- `gas_baseline_retirement_start_year` is now settable per arm
  (`None` for `gas_retire_none`, `2035` for `gas_retire_proxy`) rather
  than a single hard-coded `2035` for every case.
- `gas_retire_sourced` (`SOURCE: [PENDING]`) raises if selected.
