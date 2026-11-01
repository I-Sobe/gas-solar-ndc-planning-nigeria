# Phase 2.10 — Gas baseline, availability arms, and capacity-credit separation

Scope: `src/scenarios.py`, `src/optimize_model.py`, `src/optimize_model_sliced.py`,
and the 11 runners already carrying `--peak-case` / `--storage-cc-case` /
`--gas-build-case` / `--new-gas-deliverability`. Diagnose-and-edit only; nothing
was run (no solve, no `--help` invocation, no test execution).

Two other locations in the tree share these filenames and were **not** touched:
`thesis-code/` (a separate, untracked, self-contained git repo nested inside this
one) and `.claude/worktrees/phase-2-4-wiring/` (a git worktree for a different
branch). All edits below are in the top-level `src/` and `scripts/` of this repo.

---

## STEP 0 — Inventory (verbatim, as found before any edit in this phase)

### Every consumer of `gas_availability_factor`

**`src/optimize_model.py`, validation + annual-ceiling constraint (originally
lines 483–498):**

```python
    gas_availability_factor = scenario.get("gas_availability_factor", None)
    if gas_availability_factor is None:
        raise ValueError(
            "gas_availability_factor is unset. Provide FUEL-AGNOSTIC gas fleet "
            "availability (mechanical + atmospheric only; exclude feedstock, "
            "which gas_balance already carries). Do NOT use NERC Plant Load "
            "Factor or the headline Availability Factor. See scenarios.py."
        )
    if not (0.0 < gas_availability_factor <= 1.0):
        raise ValueError("gas_availability_factor must be in (0, 1]")
    m.gas_capacity_constraint = pyo.Constraint(
        T,
        rule=lambda m, t:
            m.gas_generation[t]
            <= m.gas_capacity_mw[t] * 8760 * gas_availability_factor / 1e6
    )
```
Role: **ANNUAL ENERGY CEILING.** Converts installed gas MW into a maximum
annual TWh_th output (`MW × 8760 h × factor`), a full-year quantity, not a
single-hour one.

**`src/optimize_model.py`, exempt fuel-equivalent allowance (originally lines
527–535, only built when `new_gas_deliverability == "exempt"`):**

```python
        m.gas_exempt_fuel_allowance_twh_th = pyo.Expression(
            T,
            rule=lambda m, t:
                m.gas_new_capacity_mw[t] * 8760 * gas_availability_factor
                / gas_new_efficiency / 1e6
        )
```
Role: **ANNUAL ENERGY CEILING** (same object, reused to size an annual
fuel-equivalent allowance for new capacity — still an 8760-hour calculation).

**`src/optimize_model_sliced.py`, validation (originally lines 415–423) —
identical text to the annual builder's, omitted here for brevity. Same role.**

**`src/optimize_model_sliced.py`, per-slice capacity constraint (originally
lines 588–593):**

```python
    m.gas_capacity_constraint = pyo.Constraint(
        T, m.S,
        rule=lambda m, t, s, p:
            m.gas_generation[t, (s, p)]
            <= m.gas_capacity_mw[t] * H[(s, p)] * gas_availability_factor / 1e6
    )
```
Role: **ANNUAL ENERGY CEILING**, distributed across slices. `H[(s,p)]` is the
slice's hour-count; summed across all slices in a year this reproduces the
same `8760 × factor` annual ceiling as the annual builder — it is not a
peak-hour credit, because every slice (not just the peak slice) is capped the
same way.

**`src/optimize_model_sliced.py`, exempt fuel-equivalent allowance
(originally lines 626–630) — same text/role as the annual builder's.**

**`src/scenarios.py`, the single source of the number (originally line 1245,
inside `load_scenario()`):**

```python
    _gas_availability_factor = 0.80   # SOURCE: see dict entry below
```
This is not a "consumer" in the constraint sense — it is the definition site
that fed **both** downstream roles (see next section).

**Not a consumer, but adjacent and worth flagging:** `gas_available_power()` /
the `gas_avail_twh_th` diagnostic used across several runners is a
*different* quantity (fuel deliverability, TWh_th) and was not touched or
conflated with `gas_availability_factor`.

### `capacity_credit["gas"]` — quoted, and same-object question answered

**`src/scenarios.py`, originally line 1428 (inside the `capacity_credit`
dict literal built in `load_scenario()`):**

```python
    capacity_credit = {
        "solar": 0.0,                                                  # SOURCE: [PENDING]
        "gas": _gas_availability_factor,                                # SOURCE: [PENDING] (= gas_availability_factor)
        "storage": storage_cc,                          # SOURCE: DECLARED JUDGEMENT -- see storage_cc_scenarios()
        "hydro": _hydro_dry_avg_mw / _hydro_capacity_mw,                # SOURCE: [PENDING]
    }
```

**Confirmed: literally the same object**, not a separate value that happens
to equal it. Both `capacity_credit["gas"]` and `scenario["gas_availability_factor"]`
were assigned from the identical local variable `_gas_availability_factor`
(a single float, `0.80`) — there was exactly one number in memory feeding
both an annual-energy role (`gas_capacity_constraint`) and a peak-hour
firm-capacity role (`m.firm_capacity_constraint`, quoted below), with no
mechanism to source or change one without the other.

**`src/optimize_model.py`, the firm-capacity constraint that consumes it
(originally lines 920–929):**

```python
    m.firm_capacity_constraint = pyo.Constraint(
        T,
        rule=lambda m, t:
            capacity_credit["solar"] * m.solar_capacity_mw[t]
            + capacity_credit["gas"] * m.gas_capacity_mw[t]
            + capacity_credit["storage"] * m.storage_power_mw[t]
            + capacity_credit["hydro"] * hydro_capacity_mw
            >= peak_mw[t] * within_block_peaking_factor
               * (1.0 + _reserve_margin_by_year[t])
    )
```
Role: **PEAK-HOUR FIRM-CAPACITY CREDIT.** Tests whether credited installed
capacity covers a single peak-demand instant with a reserve margin —
independent of dispatch, and independent of any full-year hour count.
`src/optimize_model_sliced.py` (originally lines 921–930) mirrors this
exactly.

### `gas_baseline_mw` — quoted, and every consumer

**`src/scenarios.py`, originally line 1355 (inside `load_scenario()`'s
returned scenario dict):**

```python
        # ---- Gas capacity baseline (gas-only fleet, MW)
        "gas_baseline_mw": 11000,
```

Consumers:
- **`src/optimize_model.py`, originally line 364:**
  `gas_baseline = scenario["gas_baseline_mw"]` — seeds `gas_cap_rule`, the
  cumulative-capacity `pyo.Expression` (`m.gas_capacity_mw`) that both the
  annual-ceiling and firm-capacity constraints read.
- **`src/optimize_model_sliced.py`, originally line 469:** identical role.
- **`src/scenarios.py`, `gas_ndc_17gw_trajectory_mw()` docstring, originally
  line 886:** a *comment-only* reference (`"...from the current 11,000 MW
  baseline fleet (scenario['gas_baseline_mw'])..."`) — not a functional
  consumer. Left unedited; see "Known stale references not touched" below.
- **`src/scenarios.py`, adjacent retirement comment, originally lines
  1356–1358:** `"680 MW/yr x 11 years = 7,480 MW retired by 2045 from an
  11,000 MW gas-only base"` — a comment describing the same dict entry,
  updated in Step 1 below since it sits in the exact block being corrected.

No other file in the active repo (`src/`, `scripts/`, `tests/`) references
`gas_baseline_mw` — confirmed by a repo-wide grep excluding the nested
`thesis-code/` copy and the `.claude/worktrees/` copy.

### Scalar or per-year?

**Confirmed scalar.** `gas_availability_factor` is read once per `build_model`
call and used as a bare Python float multiplied directly into constraint
rules (`... * gas_availability_factor / 1e6`), with a single scalar bounds
check (`0.0 < gas_availability_factor <= 1.0`) that would raise a
`TypeError` if a list/array were passed. There is no `t`-indexing anywhere
in its consumption. This is unlike `reserve_margin` or `max_offgrid_fraction`,
which both explicitly accept either a flat scalar or a per-year sequence.

---

## STEP 1 — Baseline (`src/scenarios.py`)

```diff
         # ---- Gas capacity baseline (gas-only fleet, MW)
-        "gas_baseline_mw": 11000,
+        # SOURCE: NERC Quarterly Reports -- 23 gas plants tracked for
+        # availability, Dec 2024 and 2026/Q1, both totalling 10,947 MW.
+        # AES_1 (294 MW) and Gbarain_1 (120 MW) are licensed (NERC 2024
+        # Table 4.3) but carry no availability data and are excluded.
+        "gas_baseline_mw": 10947,
         # Brownfield retirement proxy: linear retirement from 2035.
         # (Comment corrected: 680 MW/yr x 11 years = 7,480 MW retired by 2045
-        #  from an 11,000 MW gas-only base -- NOT the whole fleet.)
+        #  from a 10,947 MW gas-only base -- NOT the whole fleet.)
         "gas_baseline_retirement_start_year": 2035,
```

No other numeric value changed as part of this step.

---

## STEP 2 — `gas_availability_case` as a required swept arm (`src/scenarios.py`)

**New sentinel** (mirrors `_STORAGE_CC_REQUIRED` / `_GAS_BUILD_REQUIRED`):

```diff
 _GAS_BUILD_REQUIRED = object()
+# Same mechanism, same reason (2.10, Step 2): the gas fleet's availability
+# figure feeds BOTH the annual energy ceiling (gas_availability_factor) and
+# the peak-hour firm-capacity credit (capacity_credit['gas']) -- a silent
+# default here (the old bare, unsourced 0.80) would keep both roles pinned
+# to an unstated, ~2.7x-above-observed assumption with no error, warning,
+# or visible signature in the output. See gas_availability_scenarios().
+_GAS_AVAILABILITY_REQUIRED = object()
```

**New parameter on `load_scenario()`:**

```diff
     storage_cc_case=_STORAGE_CC_REQUIRED,
     gas_build_case=_GAS_BUILD_REQUIRED,
+    gas_availability_case=_GAS_AVAILABILITY_REQUIRED,
     gas_new_efficiency: float | None = None,
```

**New docstring paragraph** (inserted before the `gas_new_efficiency` entry):

```diff
+    gas_availability_case : key of gas_availability_scenarios() -- REQUIRED,
+                        no default (2.10, Step 2, mirrors genset_case/
+                        peak_multiple_case/storage_cc_case discipline). One
+                        of 'gas_avail_observed' / 'gas_avail_central' /
+                        'gas_avail_nameplate'. Feeds BOTH
+                        gas_availability_factor (annual energy ceiling) and
+                        capacity_credit['gas'] (peak-hour firm-capacity
+                        credit) -- independently settable dict entries
+                        (2.10, Step 3) initialized equal from this SAME arm
+                        value. 'gas_avail_central' currently always raises:
+                        SOURCE: [PENDING] -- see gas_availability_scenarios().
     gas_new_efficiency : float or None (default None). New gas is declared
```

**New arm function** (inserted immediately after `storage_cc_scenarios()`):

```python
def gas_availability_scenarios() -> dict[str, float | None]:
    """
    Gas fleet availability arm (2.10, Step 2). Mirrors genset_case/
    peak_multiple_case/storage_cc_case discipline: REQUIRED, no default --
    see the _GAS_AVAILABILITY_REQUIRED sentinel above load_scenario().

    Feeds BOTH gas_availability_factor (annual energy ceiling on gas
    generation, in build_model()/build_model_sliced()) and
    capacity_credit['gas'] (peak-hour firm-capacity credit, in the firm
    capacity constraint) -- two DIFFERENT quantities that were a single
    hard-coded parameter (0.80) before 2.10. Both are now independently
    settable dict entries in load_scenario(), initialized equal from this
    SAME arm value only because no separately sourced peak-hour figure
    exists yet (2.10, Step 3) -- see the comments at both dict-entry sites
    in load_scenario().

    This must be FUEL-AGNOSTIC availability: mechanical + atmospheric
    availability only (NERC Availability Factor clauses (i) and (ii)). It
    must EXCLUDE clause (iii) feedstock availability, because gas-supply
    scarcity is already carried separately by m.gas_balance / the
    deliverability scenarios -- using NERC's headline Availability Factor
    (which includes clause iii) here would double-count fuel scarcity.

        gas_avail_observed   (0.295 [0.3061 since TASK 73; this block is the record of what shipped]): NERC-measured 2024 (3,351 / 10,947
            MW). DECLARED LOWER BOUND -- double-counts fuel, because NERC's
            PAF includes fuel unavailability while the model constrains
            fuel separately in m.gas_balance. Not a candidate central case.
        gas_avail_central    (None): SOURCE: [PENDING]. Must represent
            MECHANICAL availability only, i.e. availability given fuel is
            allocated. Raises if selected until sourced.
        gas_avail_nameplate  (0.80): the pre-2.10 unsourced assumption,
            retained as an explicit UPPER BOUND so every pre-2.10 result
            stays attributable rather than orphaned.
    """
    return {
        "gas_avail_observed":  0.295,
        "gas_avail_central":   None,
        "gas_avail_nameplate": 0.80,
    }
```

**New validation block** (inserted right after `storage_cc` is resolved,
before `years = planning_horizon(...)`):

```python
    # ---- gas_availability_case (required, no default; 2.10, Step 2).
    # Mirrors genset_case/peak_multiple_case/storage_cc_case discipline: the
    # gas fleet's availability figure is a declared judgement call (see
    # gas_availability_scenarios()), not a measured constant, so it must be
    # named explicitly rather than silently defaulting to the old bare 0.80.
    if gas_availability_case is _GAS_AVAILABILITY_REQUIRED:
        raise TypeError(
            "load_scenario() requires gas_availability_case (2.10, Step "
            "2). Pass one of 'gas_avail_observed' / 'gas_avail_central' / "
            "'gas_avail_nameplate' (see gas_availability_scenarios()). "
            "There is no default: omitting it would build both the gas "
            "annual-energy ceiling and the gas firm-capacity credit "
            "against an unstated availability assumption, with no error, "
            "warning, or visible signature in the output."
        )
    if gas_availability_case not in gas_availability_scenarios():
        raise ValueError(f"Unknown gas_availability_case: {gas_availability_case}")
    _gas_availability_factor = gas_availability_scenarios()[gas_availability_case]
    if _gas_availability_factor is None:
        raise ValueError(
            f"gas_availability_case='{gas_availability_case}' is SOURCE: "
            "[PENDING] (2.10, Step 2) -- it must represent MECHANICAL "
            "availability only (availability given fuel is allocated), and "
            "has not been sourced. Select 'gas_avail_observed' (declared "
            "lower bound) or 'gas_avail_nameplate' (declared upper bound) "
            "until sourced -- see gas_availability_scenarios()."
        )
```

**Removed the now-redundant hardcoded reassignment** further down (it would
otherwise silently shadow the case-driven value with a bare literal again):

```diff
     # _gas_availability_factor is NOT redefined here (2.10, Step 2) -- it
     # was already resolved above from gas_availability_case, before
     # gas_build_case's validation. Redefining it here would silently
     # shadow the sourced/raising arm value with a bare literal again.
     _base_demand_twh = demand_level_scenarios()[demand_level_case]
     _demand_growth = demand_growth_scenarios()[demand_case]
-    _gas_availability_factor = 0.80   # SOURCE: see dict entry below
     _hydro_baseline_twh = 11.47       # SOURCE: see dict entry below
```

**Builder-side presence validation** (`src/optimize_model.py`, mirroring the
existing `gas_build_case` guard, inserted right after it):

```python
    # gas_availability_case (2.10, Step 2/4): closes the same hand-built-
    # scenario bypass as the gas_build_case guard above, for a caller that
    # leaves the gas fleet's availability arm unstated. The deep validation
    # (arm membership, SOURCE: [PENDING] raise for gas_avail_central) lives
    # in load_scenario() -- this is presence-only, matching the existing
    # pattern.
    if scenario.get("gas_availability_case", None) is None:
        raise ValueError(
            "scenario['gas_availability_case'] is missing or None (2.10, "
            "Step 2). Required so the gas fleet's availability figure has "
            "a traceable, named choice-set -- see "
            "scenarios.gas_availability_scenarios()."
        )
```
Identical block inserted in `src/optimize_model_sliced.py` at the same
relative position (right after its `gas_build_case` guard).

**`gas_availability_case` recorded on the returned scenario dict and in
`labels`** (see Step 3 diff below for the exact insertion point of the
top-level key; the `labels` addition is):

```diff
             "gas_build": gas_build_case,
             "new_gas_deliverability": new_gas_deliverability,
+            "gas_availability": gas_availability_case,
         },
```

---

## STEP 3 — Separating the two roles (`src/scenarios.py`)

At the `capacity_credit` dict site:

```diff
     #   solar   : 0.0 -- no credit at system peak (evening peak, no solar
     #             yield in the peak slice; see SOLAR_PERIODS in
     #             optimize_model_sliced.py). SOURCE: [PENDING] as a judgement,
     #             not a citation.
-    #   gas     : the SAME fuel-agnostic availability factor as
-    #             gas_capacity_constraint (Step 0.4) -- one source of truth,
-    #             not a second hard-coded number. SOURCE: [PENDING].
+    #   gas     : (2.10, Step 3) an INDEPENDENTLY SETTABLE peak-hour
+    #             firm-capacity credit -- NOT the same object as
+    #             gas_availability_factor (an annual energy ceiling) below
+    #             any more, though both are currently initialized equal
+    #             from the SAME gas_availability_case arm (see
+    #             _gas_capacity_credit just below), because an annual
+    #             energy availability factor and a peak-hour firm-capacity
+    #             credit are different quantities -- before 2.10 they were
+    #             a single hard-coded parameter (0.80). SOURCE: [PENDING].
     #   storage : storage_cc_case (2.6, phase_2_3b Step 2) -- REQUIRED, no
     #             default. ...
     ...
     _hydro_dry_avg_mw = (
         _hydro_baseline_twh * _hydro_share["dry"] * 1e6 / _dry_hours_per_year
     )
+    # (2.10, Step 3): capacity_credit['gas'] is now a value DISTINCT from
+    # gas_availability_factor (set on the scenario dict below), even though
+    # both are initialized equal here. An annual energy availability factor
+    # and a peak-hour firm-capacity credit are different quantities; before
+    # 2.10 they were a single parameter (_gas_availability_factor, reused
+    # directly). Not changing either NUMBER in this step -- only making
+    # them independently settable so a later phase can source them
+    # separately. See the matching comment on the scenario-dict
+    # "gas_availability_factor" entry below.
+    _gas_capacity_credit = _gas_availability_factor
     capacity_credit = {
         "solar": 0.0,                                                  # SOURCE: [PENDING]
-        "gas": _gas_availability_factor,                                # SOURCE: [PENDING] (= gas_availability_factor)
+        "gas": _gas_capacity_credit,                                    # SOURCE: [PENDING] -- peak-hour firm-capacity credit; see the 2.10, Step 3 comment above
         "storage": storage_cc,                          # SOURCE: DECLARED JUDGEMENT -- see storage_cc_scenarios()
         "hydro": _hydro_dry_avg_mw / _hydro_capacity_mw,                # SOURCE: [PENDING]
     }
```

At the `"gas_availability_factor"` scenario-dict entry site:

```diff
         # ---- Gas fleet availability ceiling (fraction of installed capacity) ---
-        # This scales INSTALLED gas capacity to a realistic ENERGY ceiling. It is
-        # a capacity ceiling, not a dispatch prediction: the optimiser sets actual
-        # utilisation below it, subject to fuel (gas_balance) and demand.
+        # This scales INSTALLED gas capacity to a realistic ANNUAL ENERGY
+        # ceiling (gas_capacity_constraint in both builders). It is a capacity
+        # ceiling, not a dispatch prediction: the optimiser sets actual
+        # utilisation below it, subject to fuel (gas_balance) and demand.
+        #
+        # (2.10, Step 3): this ANNUAL ENERGY CEILING is a DIFFERENT QUANTITY
+        # from capacity_credit['gas'] (a peak-hour firm-capacity credit,
+        # set above) -- before 2.10 they were a single parameter reused in
+        # both roles. Both are now independently settable, initialized
+        # equal from the SAME gas_availability_case arm only because no
+        # separately sourced peak-hour figure exists yet.
         #
         # CRITICAL -- this must be FUEL-AGNOSTIC availability: mechanical +
         # atmospheric availability only (NERC Availability Factor clauses (i) and
         # (ii)). It must EXCLUDE clause (iii) feedstock availability, because gas-
         # supply scarcity is already carried by gas_balance / the deliverability
         # scenarios. Using NERC's headline Availability Factor here would double-
         # count fuel scarcity and contaminate the gas shadow prices.
-        
-        # Source: fuel-agnostic mechanical/maintenance availability, reconstructed
-        # from NERC outage cause-decomposition, or an OCGT/CCGT engineering
-        # benchmark derated for Nigerian plant condition (~0.75-0.85). REPLACE.
+        #
+        # Source: gas_availability_case arm (2.10, Step 2) -- see
+        # gas_availability_scenarios() for the sourced/pending breakdown per
+        # case.
+        "gas_availability_case": gas_availability_case,
         "gas_availability_factor": _gas_availability_factor,   # <-- set from fuel-agnostic availability
```

**Confirmation:** `capacity_credit["gas"]` (bound to `_gas_capacity_credit`)
and `scenario["gas_availability_factor"]` (bound to `_gas_availability_factor`)
are now two separate Python names, each independently assignable. A later
phase can source `_gas_capacity_credit` from a different arm/derivation
without touching `_gas_availability_factor`, or vice versa. **No numeric
value changed in this step** — both still resolve to whatever
`gas_availability_case` selects, exactly as before, satisfying "initially
set equal."

---

## STEP 4 — Wire and tag (11 runners + 2 builders)

### Builder-side presence validation

Added to both `src/optimize_model.py` and `src/optimize_model_sliced.py`,
immediately after the existing `gas_build_case` presence guard (full text
in the Step 2 section above — identical in both files).

### CLI wiring (all 11 runners)

Each of the following files received the same four-part change:

1. A new module-level choices tuple, next to the existing ones:
   ```python
   # 2.10, Step 4: the gas fleet's availability arm is a REQUIRED CLI argument,
   # no default -- mirrors the peak-case/storage-cc-case/gas-build-case
   # discipline.
   GAS_AVAILABILITY_CASE_CHOICES = ("gas_avail_observed", "gas_avail_central", "gas_avail_nameplate")
   ```
2. A new required `argparse` argument, immediately after `--new-gas-deliverability`:
   ```python
   parser.add_argument(
       "--gas-availability-case",
       required=True,
       choices=GAS_AVAILABILITY_CASE_CHOICES,
       help="Gas fleet availability arm fed to load_scenario(gas_availability_case=...). "
            "No default: an unstated availability figure would build both the "
            "gas annual-energy ceiling and the gas firm-capacity credit silently.",
   )
   ```
3. `gas_availability_case=args.gas_availability_case,` added to every
   `load_scenario(...)` call site fed by that script's CLI args (some
   scripts thread the value through a local helper function first — those
   helper signatures were extended too, see the per-file notes below).
4. The new arm appended to every output tag / summary dict / JSON row that
   already carried `new_gas_deliverability`, so runs stay distinguishable
   on disk.

Verified by counting occurrences of `new_gas_deliverability` vs.
`gas_availability_case` in each file — every file has an equal count,
confirming symmetric wiring:

| File | count (each) |
|---|---|
| `scripts/01_run_baseline.py` | 4 |
| `scripts/01b_run_baseline_sliced.py` | 4 |
| `scripts/02_run_eaas.py` | 4 |
| `scripts/02_run_ndc_caps.py` | 4 |
| `scripts/03_run_ndc_eaas.py` | 4 |
| `scripts/04_run_fin2_tariff_sweep.py` | 3 |
| `scripts/22_salvage_sensitivity.py` | 6 |
| `scripts/23_run_selfgen_frontier.py` | 8 |
| `scripts/24_run_leakage_arm.py` | 6 |
| `scripts/25_run_obligation_trajectory.py` | 6 |
| `scripts/99c_run_genset_arm_sensitivity.py` | 7 |

Per-file notes on the (3)/(4) helper-function threading, beyond the generic
four-part change above:

- **`scripts/22_salvage_sensitivity.py`**: `run_arm(...)` gained a
  `gas_availability_case` parameter; its one call site in `main()` was
  updated to pass `args.gas_availability_case`.
- **`scripts/23_run_selfgen_frontier.py`**: both `make_scenario(...)` and
  `solve_point(...)` gained the parameter and thread it through to
  `load_scenario(...)`.
- **`scripts/24_run_leakage_arm.py`**: `solve_one(...)` gained the
  parameter; it is also folded into that function's `base` dict (the
  per-genset-arm row written to the output CSV), so every row is now
  tagged with the arm without a separate edit.
- **`scripts/25_run_obligation_trajectory.py`**: same pattern as 24 —
  `solve_one(...)` gained the parameter and it is folded into `base`.
- **`scripts/99c_run_genset_arm_sensitivity.py`**: `solve_one(...)` gained
  the parameter; also added to the arm-summary `print()` line for
  visibility during a run.
- **`scripts/02_run_ndc_caps.py`** and **`scripts/03_run_ndc_eaas.py`**:
  no helper function — `load_scenario(...)` is called directly inside a
  `for` loop over NDC cases, using `args.gas_availability_case` directly;
  the per-case output directory name and `summary.json` dict both extended.
- **`scripts/04_run_fin2_tariff_sweep.py`**: no per-case summary dict (only
  a single `output_tag` used for all three output filenames) — `output_tag`
  extended, `load_scenario(...)` call updated.

All 14 touched files were checked with `python3 -m py_compile` after every
edit; all compiled cleanly. No script was executed (`--help` included),
per the "Run nothing" scope rule.

---

## Known stale references, deliberately not touched (flagged, not fixed)

These are pre-existing and outside what Step 1–4 asked for; noted here
rather than silently fixed or silently left inconsistent:

1. `src/scenarios.py`, `gas_ndc_17gw_trajectory_mw()` docstring (unchanged
   line, originally 886): still says *"the current 11,000 MW baseline
   fleet"*. This is a comment inside a **different, untouched arm**
   (`gas_build_case='gas_ndc_17gw'`, which currently always raises because
   its trajectory is `SOURCE: [PENDING]` — unrelated to this phase). Now
   technically off by 53 MW against the corrected `gas_baseline_mw=10947`.
2. `scripts/verify_2_6_commit_a.sh` invokes `scripts/01_run_baseline.py`
   with **no CLI arguments at all** — it was already failing before this
   phase (missing `--peak-case`, `--storage-cc-case`, `--gas-build-case`,
   `--new-gas-deliverability`, all required since 2.6/2.9). The new
   `--gas-availability-case` requirement adds one more missing argument to
   an already-non-functional invocation; not caused by, or fixable within,
   this phase's scope.
3. `tests/test_run.py` calls `load_scenario()` with none of
   `peak_multiple_case` / `reserve_margin` / `storage_cc_case` /
   `gas_build_case` — already raising before this phase for the same
   reason as (2). `gas_availability_case` is one more now-required
   argument it doesn't supply.

## Confirmation (Step 5 close-out)

- The annual-ceiling role (`gas_availability_factor`, consumed in
  `gas_capacity_constraint` / `gas_exempt_fuel_allowance_twh_th` in both
  builders) and the firm-capacity-credit role (`capacity_credit["gas"]`,
  consumed in `firm_capacity_constraint` in both builders) are now bound to
  two separate Python names (`_gas_availability_factor` and
  `_gas_capacity_credit`) in `load_scenario()`, each independently
  reassignable, each with its own explanatory comment at its scenario-dict
  site.
- Both are currently initialized from the same `gas_availability_case` arm
  (no numeric divergence introduced), satisfying "even if the two are
  initially set equal."
- `gas_availability_case` is REQUIRED with no default at both the
  `load_scenario()` level (raises `TypeError`/`ValueError`) and the
  builder level (`build_model`/`build_model_sliced` raise `ValueError` on
  a hand-built scenario missing the key), and is threaded as a required
  CLI argument through all 11 runners, tagged into every output artifact
  those runners write.
- `gas_avail_central` (`SOURCE: [PENDING]`) raises if selected, in
  `load_scenario()` (the new Step 2 validation block). A hand-built
  scenario that sets `gas_availability_factor` to `None` directly is
  independently caught by both builders' pre-existing
  `if gas_availability_factor is None: raise ValueError(...)` check.
- `gas_baseline_mw` is now `10947` (NERC-sourced, comment attached);
  no other numeric baseline value was changed.
