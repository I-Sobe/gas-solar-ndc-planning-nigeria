# Phase 2.4 Inversion — PAR Arm Redesign and Guard-Bypass Closure

Scope: `src/scenarios.py` (STEP 1), `src/optimize_model.py` +
`src/optimize_model_sliced.py` (STEP 2 guard relocation, STEP 3 reserve
margin), `src/io.py` (STEP 2, necessary to make the `genset_case` guard
meaningful — see STEP 2 below). Nothing was solved, no runner script or the
smoke suite was executed. Verification for most of this task relied on
`ast.parse` and careful manual trace only: partway through, the
`nigeria-energy` conda environment's `numpy` began failing to import with
`ImportError: DLL load failed while importing _umath_linalg: An Application
Control policy has blocked this file` — an external, system-level block that
appeared mid-session (numpy imported fine earlier in this same session, for
the prior `phase_2_3_2_4` task) and was not caused by any edit in this task.
I did not attempt to work around it. Everything reported as "verified" below
was verified before that point; everything after is manual review only, and
is flagged as such.

This document builds directly on `docs/phase_2_3_2_4.md`, which implemented
the original (now-inverted) PAR design and the firm capacity constraint.
Read that first if the "before" state of any diff below is unclear.

---

## STEP 1 — Inverting the arm design (`src/scenarios.py`)

### The problem, confirmed

`h_peak` (combined `dry_peak` + `wet_peak` hours) is fixed calendar
arithmetic, independent of any scenario input: `724 + 736 = 1460` h/yr.
`8760 / 1460 = 6` **exactly**. Since `PAR = f_peak * 8760 / h_peak`, this
means **`PAR = 6 * f_peak` identically** — PAR is not an independent
quantity, it is the peak block's demand-energy share restated. The original
design (`docs/phase_2_3_2_4.md`) supplied a scalar PAR per arm and checked it
against the slice shape, which checks a derived quantity against the exact
thing it was derived from: structurally either redundant (numbers agree) or
a bug (they don't), never informative.

### What changed

Each `peak_multiple_case` arm now supplies the **shape** directly —
`period_demand_factor` (required) and `season_demand_factor` (optional,
only if the arm varies it from the shared default) — and PAR (renamed
`par_block`, per the task's naming) is **derived** from that shape via
`_normalised_demand_weights()`/`slice_hours()` (the model's own functions,
imported from `optimize_model_sliced.py`, not reimplemented), then stored on
the built scenario purely for reporting. **The 1% disagreement guard is
deleted** — there is nothing left for it to check.

Arms renamed `par_observed`/`par_central`/`par_high` → `peak_observed`/
`peak_nirp`/`peak_high`, per the task's instruction. `par_central` becomes
`peak_nirp` specifically (not just relabelled) — consistent with this
codebase's convention of naming a central/adopted case after its intended
source (cf. `blended_central`, `nirp_trajectory`).

Module-level constants (all `SOURCE: [PENDING]`, values `None`):
```python
PEAK_OBSERVED_PERIOD_FACTOR = None   # SOURCE: [PENDING]
PEAK_NIRP_PERIOD_FACTOR     = None   # SOURCE: [PENDING]
PEAK_HIGH_PERIOD_FACTOR     = None   # SOURCE: [PENDING]

PEAK_OBSERVED_SEASON_FACTOR = None   # optional override, not pending
PEAK_NIRP_SEASON_FACTOR     = None   # optional override, not pending
PEAK_HIGH_SEASON_FACTOR     = None   # optional override, not pending
```
Selecting `period_demand_factor: None` raises (verified before the numpy
block — see "Verification" below); `season_demand_factor: None` falls back
to `optimize_model_sliced.DEFAULT_SEASON_DEMAND_FACTOR` (imported), which is
**not** a blocking gap — an arm that doesn't vary season is complete as soon
as its period factor is supplied.

`par_block`'s docstring (module comment above `peak_multiple_scenarios()`,
and repeated at the derivation site in `load_scenario()`): **"par_block IS A
4-HOUR BLOCK-AVERAGE RATIO, NOT AN INSTANTANEOUS PEAK RATIO."** The peak
slice (18:00–22:00) is a 4-hour block; `par_block` is the ratio of that
block's *average* power to the annual average, not the instantaneous system
peak that can occur at a single moment inside the window. This is exactly
what STEP 3's `within_block_peaking_factor` corrects for, kept deliberately
separate.

### `par_block` for "the current default shape"

No arm is populated yet (all three `PEAK_*_PERIOD_FACTOR` constants are
`None`), so there is no live, selectable arm to compute `par_block` from.
The only shape still resident in the codebase is
`optimize_model_sliced.DEFAULT_SEASON_DEMAND_FACTOR` /
`DEFAULT_PERIOD_DEMAND_FACTOR` — `{"dry": 1.08, "wet": 0.92}` and
`{"night": 0.65, "day": 1.05, "peak": 1.20}` — which used to be `scenarios.py`'s
own hard-coded independent default before this inversion, and is now only
reachable as the season-override fallback, or as `optimize_model_sliced.py`'s
own bare `.get()` fallback for a hand-built scenario. Computing `par_block`
from these same numbers (verified before the numpy block, identical to the
value computed for `docs/phase_2_3_2_4.md` since neither the formula nor
these particular numbers changed):
```
f_peak = 0.208695652173913
h_peak = 1460
par_block = f_peak * 8760 / 1460 = 1.2521739130434781
6 * f_peak = 1.2521739130434781   (identity confirmed exactly)
```
This is illustrative only — stated so the number exists somewhere in this
report, not because any arm is actually set to it.

### Diff, `src/scenarios.py`

The arm-design section (module-level constants + `peak_multiple_scenarios()`
+ new `RESERVE_MARGIN_SPINNING_MW` constant for STEP 3), replacing the
`PAR_OBSERVED`/`PAR_CENTRAL`/`PAR_HIGH` design from `docs/phase_2_3_2_4.md`:

```diff
 # ============================================================
-# PEAK DEMAND MULTIPLE (PAR) -- 2.6, F0 Step 2.4
+# PEAK DEMAND SHAPE ARMS -- PAR IS DERIVED, NOT AN INPUT (2.6, F0 Step 2.4)
 # ============================================================
-# Peak-to-average demand ratio (PAR). Feeds peak_mw[t] = avg_mw[t] * PAR,
-# which the firm-capacity constraint (2.6, F0 Step 2.3) tests cumulative
-# credited capacity against.
-#
-# THIS IS THE SAME ASSUMPTION as the peak-slice demand share used to build
-# the 6-slice load shape in optimize_model_sliced.py (season_demand_factor x
-# period_demand_factor). load_scenario() computes the PAR implied by that
-# slice shape and RAISES if it disagrees with the arm value here by more
-# than 1% -- see the consistency check in load_scenario(). Do not set
-# peak_multiple_case and the slice-shape factors independently.
-#
-# SOURCE: [PENDING] for all three arms -- author to supply from a named
-# source (NERC/TCN system peak report, or equivalent). Do NOT ship these as
-# None; a None arm will raise the moment it is used, by design (mirrors the
-# genset_case discipline in load_econ() -- see src/io.py).
-PAR_OBSERVED = None   # SOURCE: [PENDING]
-PAR_CENTRAL  = None   # SOURCE: [PENDING]
-PAR_HIGH     = None   # SOURCE: [PENDING]
-
-
-def peak_multiple_scenarios():
-    """
-    Peak-to-average demand ratio (PAR) arms. SOURCE: [PENDING] for all three
-    -- see the module-level PAR_* constants above, which the author will
-    supply. Keys are what peak_multiple_case accepts in load_scenario().
-    """
-    return {
-        "par_observed": PAR_OBSERVED,
-        "par_central":  PAR_CENTRAL,
-        "par_high":     PAR_HIGH,
-    }
+# INVERTED from the original design (2.6, F0 phase_2_4_inversion). The
+# original design supplied a scalar PAR per arm and CHECKED it against the
+# 6-slice demand shape (season_demand_factor x period_demand_factor),
+# raising on >1% disagreement. That was the wrong direction: h_peak (the
+# combined dry_peak+wet_peak hours) is FIXED calendar arithmetic --
+# 724 + 736 = 1460 h/yr, independent of any scenario input -- and
+# 8760 / 1460 = 6 EXACTLY, so PAR = f_peak * 8760 / h_peak = 6 * f_peak
+# IDENTICALLY. PAR is a restatement of the peak block's demand-energy share
+# (f_peak), not an independent quantity that could genuinely disagree with
+# it. Checking a derived quantity against the very thing it was derived
+# from is structurally either redundant (numbers agree) or a bug (they
+# don't) -- never informative. See docs/phase_2_4_inversion.md.
+#
+# Each arm below supplies the SHAPE itself:
+#   period_demand_factor : REQUIRED per arm. None = SOURCE: [PENDING];
+#                           selecting an arm before this is filled in raises.
+#   season_demand_factor  : OPTIONAL per arm -- only set it if this arm
+#                           varies season shape from the default. None here
+#                           means "use DEFAULT_SEASON_DEMAND_FACTOR"
+#                           (imported above from optimize_model_sliced.py),
+#                           NOT a pending/blocking gap the way
+#                           period_demand_factor's None is.
+# load_scenario() computes PAR from whichever arm is selected (via
+# _normalised_demand_weights()/slice_hours(), imported above -- so this is
+# the model's own computation, not a shadow copy) and reports it as
+# par_block on the built scenario. There is no consistency guard any more:
+# with one source of truth, disagreement is unreachable by construction.
+#
+# par_block IS A 4-HOUR BLOCK-AVERAGE RATIO, NOT AN INSTANTANEOUS PEAK
+# RATIO. The peak slice (18:00-22:00) is a 4-hour block; par_block is the
+# ratio of THAT BLOCK'S AVERAGE power to the annual average -- not the
+# instantaneous system peak that can occur at a single moment inside that
+# 4-hour window, which the block average necessarily smooths over. See
+# within_block_peaking_factor (2.6, F0 Step 2.3, read inside the builders --
+# optimize_model.py / optimize_model_sliced.py), the SEPARATE, SOURCE:
+# [PENDING] correction for that gap. Do not conflate the two: par_block is
+# derived and reported; within_block_peaking_factor is a distinct,
+# independently-sourced multiplier applied on top of it.
+#
+# SOURCE: [PENDING] for all three arms' period_demand_factor -- author to
+# supply from a named source (e.g. NERC/TCN system peak report for
+# peak_observed; NIRP 2024 for peak_nirp; a stress case for peak_high). Do
+# NOT ship these as anything but None; a None arm raises the moment it is
+# selected, by design (mirrors the genset_case discipline in load_econ() --
+# see src/io.py).
+PEAK_OBSERVED_PERIOD_FACTOR = None   # SOURCE: [PENDING]
+PEAK_NIRP_PERIOD_FACTOR     = None   # SOURCE: [PENDING]
+PEAK_HIGH_PERIOD_FACTOR     = None   # SOURCE: [PENDING]
+
+# season_demand_factor OVERRIDES -- optional per arm (see above). None is a
+# normal, non-blocking default ("use the shared season shape"), not a
+# pending gap -- unlike the period factors above.
+PEAK_OBSERVED_SEASON_FACTOR = None
+PEAK_NIRP_SEASON_FACTOR     = None
+PEAK_HIGH_SEASON_FACTOR     = None
+
+
+def peak_multiple_scenarios():
+    """
+    Peak-demand SHAPE arms: {period_demand_factor, season_demand_factor}.
+    PAR is DERIVED from whichever arm is selected in load_scenario() (never
+    supplied here directly -- see the module comment above). SOURCE:
+    [PENDING] on every arm's period_demand_factor; season_demand_factor is
+    None (= use DEFAULT_SEASON_DEMAND_FACTOR) unless an arm's own constant
+    above is filled in. Keys are what peak_multiple_case accepts in
+    load_scenario().
+    """
+    return {
+        "peak_observed": {
+            "period_demand_factor": PEAK_OBSERVED_PERIOD_FACTOR,
+            "season_demand_factor": PEAK_OBSERVED_SEASON_FACTOR,
+        },
+        "peak_nirp": {
+            "period_demand_factor": PEAK_NIRP_PERIOD_FACTOR,
+            "season_demand_factor": PEAK_NIRP_SEASON_FACTOR,
+        },
+        "peak_high": {
+            "period_demand_factor": PEAK_HIGH_PERIOD_FACTOR,
+            "season_demand_factor": PEAK_HIGH_SEASON_FACTOR,
+        },
+    }
+
+
+# ============================================================
+# RESERVE MARGIN -- SPINNING RESERVE (2.6, F0 Step 2.3)
+# ============================================================
+# SOURCE: NIRP 2024 Section 2.3, p.19 -- spinning reserve requirement.
+# Unlike the PAR arms above, this IS a cited value, not a placeholder.
+#
+# load_scenario()'s reserve_margin argument accepts either a plain fraction
+# (used as-is, unchanged) or the literal string 'nirp_spinning', which
+# derives a PER-YEAR fraction as RESERVE_MARGIN_SPINNING_MW / peak_mw[t]:
+# a FIXED absolute MW requirement implies a SHRINKING fractional margin as
+# peak_mw[t] grows over the horizon. Do not average this into a single
+# scalar -- report and use the per-year series.
+RESERVE_MARGIN_SPINNING_MW = 900.0  # NIRP 2024 Section 2.3, p.19 -- spinning reserve requirement
```

The import line gained `DEFAULT_SEASON_DEMAND_FACTOR` (needed for the season
fallback):
```diff
 from src.optimize_model_sliced import (
     SEASON_DAYS,
     PERIOD_HOURS,
+    DEFAULT_SEASON_DEMAND_FACTOR,
     slice_hours,
     _normalised_demand_weights,
     _hydro_season_energy_share,
 )
```

`load_scenario()`'s validation + derivation block, replacing the old
scalar-PAR-plus-1%-guard design:

```diff
-    if peak_multiple_case is _PEAK_MULTIPLE_REQUIRED:
-        raise TypeError(
-            "load_scenario() requires peak_multiple_case (2.6, F0 Step 2.4). "
-            "Pass one of 'par_observed' / 'par_central' / 'par_high' (see "
-            "peak_multiple_scenarios()). There is no default: omitting it "
-            "would build a firm-capacity constraint against an unstated "
-            "peak-demand assumption with no error, warning, or visible "
-            "signature in the output."
-        )
-    if peak_multiple_case not in peak_multiple_scenarios():
-        raise ValueError(f"Unknown peak_multiple_case: {peak_multiple_case}")
-    par_value = peak_multiple_scenarios()[peak_multiple_case]
-    if par_value is None:
-        raise ValueError(
-            f"peak_multiple_scenarios()['{peak_multiple_case}'] is None -- "
-            "SOURCE: [PENDING]. Supply a sourced PAR value for this arm "
-            "(module-level PAR_* constants, top of scenarios.py) before "
-            "building a scenario with a firm-capacity constraint."
-        )
-
-    if reserve_margin is _RESERVE_MARGIN_REQUIRED:
-        raise TypeError(
-            "load_scenario() requires reserve_margin (2.6, F0 Step 2.3). "
-            "Pass a fractional reserve margin (e.g. 0.15). There is no "
-            "default: omitting it would build a firm-capacity constraint "
-            "against an unstated reserve assumption with no error, warning, "
-            "or visible signature in the output."
-        )
-    if reserve_margin < 0:
-        raise ValueError(f"reserve_margin must be >= 0, got {reserve_margin}")
+    if peak_multiple_case is _PEAK_MULTIPLE_REQUIRED:
+        raise TypeError(
+            "load_scenario() requires peak_multiple_case (2.6, F0 Step 2.4). "
+            "Pass one of 'peak_observed' / 'peak_nirp' / 'peak_high' (see "
+            "peak_multiple_scenarios()). There is no default: omitting it "
+            "would build a firm-capacity constraint against an unstated "
+            "peak-demand shape with no error, warning, or visible signature "
+            "in the output."
+        )
+    if peak_multiple_case not in peak_multiple_scenarios():
+        raise ValueError(f"Unknown peak_multiple_case: {peak_multiple_case}")
+    _peak_arm = peak_multiple_scenarios()[peak_multiple_case]
+    _arm_period_demand_factor = _peak_arm["period_demand_factor"]
+    if _arm_period_demand_factor is None:
+        raise ValueError(
+            f"peak_multiple_scenarios()['{peak_multiple_case}']"
+            "['period_demand_factor'] is None -- SOURCE: [PENDING]. Supply "
+            "a sourced period_demand_factor for this arm (module-level "
+            "PEAK_*_PERIOD_FACTOR constants, top of scenarios.py) before "
+            "building a scenario with a firm-capacity constraint."
+        )
+    _arm_season_demand_factor = (
+        _peak_arm["season_demand_factor"]
+        if _peak_arm["season_demand_factor"] is not None
+        else DEFAULT_SEASON_DEMAND_FACTOR
+    )
+
+    if reserve_margin is _RESERVE_MARGIN_REQUIRED:
+        raise TypeError(
+            "load_scenario() requires reserve_margin (2.6, F0 Step 2.3). "
+            "Pass a fractional reserve margin (e.g. 0.15), or the literal "
+            "'nirp_spinning' to derive it from RESERVE_MARGIN_SPINNING_MW / "
+            "peak_mw[t]. There is no default: omitting it would build a "
+            "firm-capacity constraint against an unstated reserve "
+            "assumption with no error, warning, or visible signature in "
+            "the output."
+        )
+    if isinstance(reserve_margin, str):
+        if reserve_margin != "nirp_spinning":
+            raise ValueError(
+                f"Unknown reserve_margin mode: '{reserve_margin}'. Pass a "
+                "numeric fraction, or the literal 'nirp_spinning'."
+            )
+    elif reserve_margin < 0:
+        raise ValueError(f"reserve_margin must be >= 0, got {reserve_margin}")
```

Derivation block: `_season_demand_factor`/`_period_demand_factor` now come
from the arm (`_arm_season_demand_factor`/`_arm_period_demand_factor`)
instead of local literals; the 1%-guard block (`_implied_par`,
`_par_disagreement`, the raising `if`) is **deleted outright**; `par_value`
is renamed `par_block` and computed unconditionally (no comparison);
`reserve_margin_resolved` is a new step, resolving `'nirp_spinning'` into a
per-year list once `peak_mw` is known:
```diff
-    _season_demand_factor = {"dry": 1.08, "wet": 0.92}
-    _period_demand_factor = {"night": 0.65, "day": 1.05, "peak": 1.20}
-
-    _implied_weights = _normalised_demand_weights({...})
-    _slice_h = slice_hours()
-    _f_peak = sum(...)
-    _h_peak = sum(...)
-    _implied_par = _f_peak * 8760 / _h_peak
-    _par_disagreement = abs(_implied_par - par_value) / par_value
-    if _par_disagreement > 0.01:
-        raise ValueError(...)
-
-    peak_mw = [
-        float(_demand_proj[t]) * 1e6 / 8760.0 * par_value
-        for t in range(len(years))
-    ]
+    _season_demand_factor = _arm_season_demand_factor
+    _period_demand_factor = _arm_period_demand_factor
+
+    _implied_weights = _normalised_demand_weights({...})
+    _slice_h = slice_hours()
+    _f_peak = sum(...)
+    _h_peak = sum(...)
+    par_block = _f_peak * 8760 / _h_peak
+
+    peak_mw = [
+        float(_demand_proj[t]) * 1e6 / 8760.0 * par_block
+        for t in range(len(years))
+    ]
+
+    if reserve_margin == "nirp_spinning":
+        reserve_margin_resolved = [
+            RESERVE_MARGIN_SPINNING_MW / mw for mw in peak_mw
+        ]
+    else:
+        reserve_margin_resolved = reserve_margin
```

Returned scenario dict: `par_block` added (reported-only), `reserve_margin`
now stores the *resolved* value (never the literal `'nirp_spinning'`
string):
```diff
         "peak_multiple_case": peak_multiple_case,
+        "par_block": par_block,
         "peak_mw": peak_mw,
-        "reserve_margin": reserve_margin,
+        "reserve_margin": reserve_margin_resolved,
         "capacity_credit": capacity_credit,
```

The `season_demand_factor`/`period_demand_factor` dict-entry comments were
also corrected (they referenced "the consistency check above," which no
longer exists):
```diff
-        # NOTE: this is the SAME assumption as peak_multiple_case (2.6, F0 Step
-        # 2.4) -- see the consistency check above. Do not edit independently.
+        # NOTE: as of 2.6, F0 Step 2.4 (inversion), these are SOURCED FROM
+        # peak_multiple_case, not set independently -- see
+        # peak_multiple_scenarios() and the derivation above. Editing these
+        # here directly has no effect; edit the selected arm's
+        # PEAK_*_PERIOD_FACTOR / PEAK_*_SEASON_FACTOR constants instead.
         "season_demand_factor": _season_demand_factor,
```

`load_scenario()`'s docstring for both parameters was rewritten to describe
the inverted contract (full text in the file; not reproduced a third time
here).

---

## STEP 2 — Moving the required-key guard into the builders

### What the survey found (verified against the actual repository, not assumed)

The task's premise: *"22 of 33 scripts use load_scenario; the rest construct
scenarios by hand and bypass every check."* I checked every one of the 33
numbered scripts plus every non-numbered script in `scripts/`, `tests/`, and
`src/optimize_experiments.py` directly (`grep` for `load_scenario(`,
`build_model(`, `build_model_sliced(`, hand-built `scenario = {` literals,
and manual reads of the ambiguous cases). The counts are real; the
**bypass-by-hand-construction claim does not hold for any script currently
in this repository.**

**33 numbered scripts (`00_`–`99c_`), classified:**

| Class | Count | Scripts |
|---|---|---|
| Call `load_scenario()` directly | 21 | `01_run_baseline.py`, `01b_run_baseline_sliced.py`, `02_run_eaas.py`, `02_run_ndc_caps.py`, `03_run_ndc_eaas.py`, `04_run_fin2_tariff_sweep.py`, `05_run_fin3_capital_sweep.py`, `06_run_gas1_shadow_benchmarks.py`, `07_run_gas2_eaas_gas_relief.py`, `09b_run_rel1_mode_sensitivity.py`, `09b_run_rel1_retirement_sensitivity.py`, `15_run_pol1_ndc_comparison.py`, `17_run_monte_carlo.py`, `18_run_pareto_frontier.py`, `19_run_carbon_price_sweep.py`, `20_run_cf1_solar_yield_sensitivity.py`, `22_salvage_sensitivity.py`, `23_run_selfgen_frontier.py`, `24_run_leakage_arm.py`, `25_run_obligation_trajectory.py`, `99c_run_genset_arm_sensitivity.py` |
| Call model-building **indirectly** via `src/optimize_experiments.py` helper functions, which call `load_scenario()` internally — **not a bypass** | 6 | `08_run_gas3_regime_ndc_feasibility.py`, `09_run_rel1_feasibility.py`, `10_run_rel2_marginal_cost.py`, `11_run_rel3_financing_frontier.py`, `12_run_dem1_demand_sensitivity.py`, `13_run_dem2_growth_gas_interaction.py` |
| Suspended stub — touches neither `load_scenario()` nor `build_model()`; prints a message and `sys.exit(0)` | 1 | `14_run_str1_storage_role.py` |
| Non-model utility/analysis — never touches `load_scenario()` or `build_model()` at all | 5 | `00_build_emissions_cap.py`, `08_phase8_analysis.py`, `16_plot_cost_decomposition.py`, `21_compute_resolution_premium.py`, `99_validate_baseline.py` |

`21 + 6 + 1 + 5 = 33` — accounted for exactly.

**Plus, outside the 33 numbered scripts:** `probe_2_6_structure.py` also
calls `load_scenario()` directly (a 22nd direct caller across the repo —
this is very likely where the task's "22 of 33" figure comes from: 22 total
direct `.py` callers, against a denominator of 33 numbered scripts, which
undercounts by one caller that isn't numbered). Six more non-numbered
utility scripts (`_check_solver.py`, `_compare_2_6.py`,
`build_gas_deliverability.py`, `diag_fin2.py`, `plot_storage_regime.py`,
`plot_system_diagnostics.py`) touch neither.

**`src/optimize_experiments.py` itself**: 11 real `load_scenario()` call
sites (lines 635, 779, 801, 1178, 1392, 1575, 1712, 2456, 2684, 2856, 2997).
One `copy.deepcopy(scenario)` at line 2934 (a copy of an
already-`load_scenario()`-built dict, then mutated — inherits every key
including `peak_mw`/`reserve_margin`/`capacity_credit`; not a bypass, but
see the residual risk noted below). **No hand-built `scenario = {...}`
dict literal anywhere in this file.**

**`14_run_str1_storage_role.py` specifically** (named in the task as the
smoke-suite counter-example): read in full. It is a **SUSPENDED stub**
(module docstring: `"STATUS: SUSPENDED pending Phase 2..."`) whose `main()`
prints a message and calls `sys.exit(0)` — it constructs no scenario, calls
no model-building function, and imports nothing beyond `sys`. It "passed the
smoke suite while all load_scenario runners failed" because it does nothing
at all, not because it hand-builds a scenario. Correcting this plainly since
the task's parenthetical implies otherwise.

**One real bypass does exist, just not the one the task described:**
`tests/test_run.py` calls `load_scenario()` (so it inherits the early,
friendlier failure the task asked to keep — it will now raise for missing
`peak_multiple_case`/`reserve_margin` too), but constructs its `econ` dict
**entirely by hand**, reading CSVs directly and never calling `load_econ()`.
This is the concrete, verified case the `genset_case`-via-`econ` guard (see
below) actually catches.

**Residual risk, not fixed by this task (flagged, not silently left):**
`scripts/04_run_fin2_tariff_sweep.py` calls `load_scenario()` then mutates
`scenario["financing_regime"]` and `scenario["required_margin"]`
post-construction (confirmed at lines 199, 211–212). It does not touch
`period_demand_factor`/`season_demand_factor`/`peak_mw` today, so it is not
currently affected by the inversion — but the general pattern (mutate a
`load_scenario()`-built dict after the fact) means a future script that
mutates the demand-shape keys after construction would leave `par_block` and
`peak_mw` silently stale relative to the mutated shape. Neither
`load_scenario()` nor the new builder-level guard re-derives or re-checks
`par_block` against a post-hoc mutation — the guard added in this task
checks *presence*, not *freshness*. This was true of the pre-inversion 1%
guard too (it only ran inside `load_scenario()`), so the inversion does not
make this worse, but it doesn't close it either. Out of this task's stated
scope (STEP 2 asks for a presence guard, not a freshness guard); noted for
the record.

### The guard itself

Moved to the **top** of both `build_model()` and `build_model_sliced()`
(immediately after `T = range(len(years))`, before any other
scenario/econ processing), and extended to check `peak_multiple_case`
(presence) and `genset_case` (via `econ["GENSET_CASE"]`) in addition to the
`peak_mw`/`reserve_margin`/`capacity_credit` checks already added for
`docs/phase_2_3_2_4.md`. **The `load_scenario()` checks are kept
unchanged** as the early, friendlier failure — this is a second,
independent layer, not a replacement.

`genset_case` cannot be checked as `scenario['peak_multiple_case']`-style
presence-in-scenario, because `genset_case` is not a scenario key: it is an
argument to `load_econ()` (`src/io.py`), and its considered value (a case
name, or an *explicit* `None` for the commit-A regression test) only shows
up in the resulting `econ` dict. Before this task, `econ["GENSET_CASE"]` was
set **only** inside `load_econ()`'s `if genset_case is not None:` branch —
so a hand-built `econ` dict (exactly what `tests/test_run.py` does) was
**indistinguishable** from a deliberate `genset_case=None` opt-out: both
simply lack the key. Checking `"GENSET_CASE" not in econ ⇒ raise` on that
basis would have broken the deliberately-supported no-backstop pathway
(`verify_2_6_commit_a.sh`, the `voll` arm of `23_run_selfgen_frontier.py`,
both documented in `src/optimize_model.py`'s own comments as pathways that
must keep working). Instead:

**`src/io.py` change**: `econ["GENSET_CASE"] = genset_case` moved *outside*
the `if genset_case is not None:` block, so it is **always** set — to a
case-name string, or to `None` for a deliberate, considered opt-out. The
builder guard checks **key presence, not truthiness**
(`"GENSET_CASE" not in econ`), so `None` (a legitimate, considered value)
does not raise, while a genuinely hand-built `econ` (no key at all) does.
`econ["GENSET_CASE"]` had zero other readers in the codebase before this
change (checked by grep), so this was safe to alter.

I flag this as a **judgement call, not a literal reading of the task text**:
the task said "raise if ... genset_case ... is missing or None" as one
uniform rule across all four items. Applied literally to `genset_case`, that
rule is unsatisfiable without breaking the commit-A regression test, because
before this task's `io.py` change `econ["GENSET_CASE"]` could never be
present-and-`None` — only absent-entirely or present-with-a-string, making
"missing or None" collapse to "missing," which is *always true* on the
legitimate no-backstop path. I judged that the intent — close a real
guard-bypass hole — is better served by distinguishing "never decided"
(raise) from "decided against" (don't raise), which required the `io.py`
change above to make that distinction representable at all. Reported here in
case that reasoning should be overridden.

### Diff, `src/io.py`

```diff
     if genset_case is not None:
         ...
         econ["GENSET_COST_PER_TWH_E"] = to_float(g_row["total_usd_per_twh_e"])
         econ["GENSET_ETA_HHV"] = to_float(g_row["genset_eta_hhv"])
         econ["GENSET_EF_TH_HHV"] = to_float(g_row["ef_th_hhv_tco2_per_mwh_th"])
-        econ["GENSET_CASE"] = genset_case
-        
+
+    # ALWAYS set, even when genset_case is explicitly None (2.6, F0 Step 2,
+    # phase_2_4_inversion). Previously this key was set only inside the
+    # branch above, so "key absent" and "genset_case=None on purpose" were
+    # indistinguishable from econ alone -- an econ dict built by hand,
+    # bypassing load_econ() entirely, looked identical to a deliberate
+    # opt-out. build_model()/build_model_sliced() now guard on "GENSET_CASE"
+    # in econ (key presence, not truthiness) precisely because None here is
+    # a legitimate, considered value (the commit-A regression test), not a
+    # missing one -- only the key's absence indicates load_econ() was
+    # bypassed.
+    econ["GENSET_CASE"] = genset_case
+
     return econ
```

### Diff, `src/optimize_model.py` (guard relocation)

The guard block (peak_mw/reserve_margin/capacity_credit/hydro_capacity_mw,
carried over from `docs/phase_2_3_2_4.md`, plus the two new checks) moved
from its old position (after the storage/genset-backstop parameter reads,
~30 lines into the function) to immediately after `T = range(len(years))`:

```diff
     years = scenario["years"]
     T = range(len(years))
-    
+
+    # ============================================================
+    # Firm capacity / genset guard (2.6, F0 Step 2 -- phase_2_4_inversion)
+    # ============================================================
+    # AT THE TOP, before any other scenario/econ processing: 22 of 33
+    # runner scripts call load_scenario() (directly or via
+    # src/optimize_experiments.py), which already guards peak_multiple_case
+    # and reserve_margin as required, no-default arguments -- see
+    # scenarios.py. That guard is the EARLY, FRIENDLIER failure and is kept
+    # exactly as-is. But build_model() accepts ANY scenario/econ dict a
+    # caller constructs directly (tests/test_run.py hand-builds econ this
+    # way today, bypassing load_econ() entirely), so nothing upstream can be
+    # relied on. This block is the guard that actually closes the bypass,
+    # placed first so it fires before any other processing gives a
+    # confusing, unrelated error instead.
+    if scenario.get("peak_multiple_case", None) is None:
+        raise ValueError(
+            "scenario['peak_multiple_case'] is missing or None (2.6, F0 "
+            "Step 2.4). Required so the firm-capacity constraint's PAR has "
+            "a traceable, named justification -- see "
+            "scenarios.peak_multiple_scenarios()."
+        )
+    peak_mw = scenario.get("peak_mw", None)
+    if peak_mw is None:
+        raise ValueError(...)
+    if len(peak_mw) != len(T):
+        raise ValueError(...)
+    reserve_margin = scenario.get("reserve_margin", None)
+    if reserve_margin is None:
+        raise ValueError(...)
+    capacity_credit = scenario.get("capacity_credit", None)
+    if capacity_credit is None:
+        raise ValueError(...)
+    for _tech in ("solar", "gas", "storage", "hydro"):
+        if _tech not in capacity_credit:
+            raise ValueError(...)
+    hydro_capacity_mw = scenario.get("hydro_capacity_mw", None)
+    if hydro_capacity_mw is None:
+        raise ValueError(...)
+    # genset_case: checked on ECON, not scenario. load_econ() (src/io.py)
+    # ALWAYS sets econ["GENSET_CASE"] now (to a case-name string, or to None
+    # for a deliberate, considered opt-out -- the commit-A regression test).
+    # Checking KEY PRESENCE (not truthiness) is deliberate: None is a
+    # legitimate value here, unlike peak_mw/reserve_margin/capacity_credit
+    # above. Only the key's ABSENCE means econ was hand-built, bypassing
+    # load_econ() entirely -- exactly what tests/test_run.py does today.
+    if "GENSET_CASE" not in econ:
+        raise ValueError(
+            "econ['GENSET_CASE'] is missing (2.6, F0 Step 2). This means "
+            "econ was not built via load_econ() (src/io.py) -- genset_case "
+            "was never decided, not even explicitly declined. Build econ "
+            "via load_econ(genset_case=...) (pass None explicitly for the "
+            "commit-A regression test) rather than constructing it by hand."
+        )
+
     # --------------------------
-    # Discounting 
+    # Discounting
     # --------------------------
```
and the old copy of the `peak_mw`/`reserve_margin`/`capacity_credit`/
`hydro_capacity_mw` block (previously sitting after the storage/genset
parameter reads, right before `m.solar_public_add = pyo.Var(...)`) was
**deleted outright** — it is not duplicated, only relocated.
`optimize_model_sliced.py` received the identical relocation (same
before/after shape; the sliced builder's separately-pre-existing
`hydro_capacity_mw` guard, needed later for hydro dispatch, was left at its
own natural definition point and not moved, since nothing in STEP 2 asked
for that).

---

## STEP 3 — Reserve margin from NIRP

`RESERVE_MARGIN_SPINNING_MW = 900.0` added to `src/scenarios.py` (shown in
the STEP 1 diff above), cited `NIRP 2024 Section 2.3, p.19 — spinning
reserve requirement` — the one genuinely sourced constant added in either
task; everything else added across both tasks is `SOURCE: [PENDING]`.

`reserve_margin` now accepts either a plain numeric fraction (used
unchanged) or the literal string `'nirp_spinning'`, resolved inside
`load_scenario()` once `peak_mw` is known:
```python
if reserve_margin == "nirp_spinning":
    reserve_margin_resolved = [
        RESERVE_MARGIN_SPINNING_MW / mw for mw in peak_mw
    ]
else:
    reserve_margin_resolved = reserve_margin
```
This is **necessarily per-year** when derived: a fixed 900 MW absolute
requirement implies a shrinking *fractional* margin as `peak_mw[t]` grows
over the horizon (900 MW is a smaller share of a bigger peak in 2045 than in
2025). It is not averaged into one scalar.

Both builders now accept `reserve_margin` as **either** a flat scalar or a
per-year sequence in `m.firm_capacity_constraint`'s rule, using the exact
same dual-form dispatch pattern already established in this codebase for
`max_offgrid_fraction`:
```python
if isinstance(reserve_margin, (list, tuple)):
    if len(reserve_margin) != len(T):
        raise ValueError(...)
    _reserve_margin_by_year = {t: float(reserve_margin[t]) for t in T}
else:
    _reserve_margin_by_year = {t: float(reserve_margin) for t in T}
```

`within_block_peaking_factor` added **separately**, read inside each
builder (not set by `load_scenario()`, mirroring exactly how
`selfgen_busbar_factor` already works — a real, non-blocking default read
via `scenario.get()` at the builder, never forced into the scenario dict):
```python
within_block_peaking_factor = float(
    scenario.get("within_block_peaking_factor", 1.0)
)
if within_block_peaking_factor < 1.0:
    raise ValueError(
        "within_block_peaking_factor < 1.0 would claim the instantaneous "
        "peak is LOWER than its own 4-hour block average, which is "
        "backwards."
    )
```
Documented as "the ratio of instantaneous to block-average peak," `SOURCE:
[PENDING]`, default `1.0` (a declared conservative simplification: assumes
no within-block peaking above the block average — i.e., that `par_block`
already *is* the instantaneous peak, which is optimistic and will likely be
revised upward once sourced).

**Kept separate, as instructed** — `m.firm_capacity_constraint`'s RHS
multiplies `within_block_peaking_factor` onto `peak_mw[t]` (the
block-average → instantaneous correction) and adds `reserve_margin[t]`
afterward (the policy/engineering adequacy margin), rather than folding one
into the other:
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
At the current defaults (`within_block_peaking_factor = 1.0`, and whatever
flat `reserve_margin` a caller passes), this is numerically identical to the
pre-STEP-3 constraint — the new factor is a genuine no-op until sourced,
not a silent behaviour change.

Identical structure applied to `optimize_model_sliced.py`.

---

## STEP 4 — SOURCE: [PENDING] inventory (current, after both tasks)

**In `src/scenarios.py`:**

1. `PEAK_OBSERVED_PERIOD_FACTOR = None` — `# SOURCE: [PENDING]`
2. `PEAK_NIRP_PERIOD_FACTOR = None` — `# SOURCE: [PENDING]`
3. `PEAK_HIGH_PERIOD_FACTOR = None` — `# SOURCE: [PENDING]`
4. `"solar": 0.0,` in `capacity_credit` — `# SOURCE: [PENDING]` (unchanged
   from `docs/phase_2_3_2_4.md`)
5. `"gas": _gas_availability_factor,` in `capacity_credit` — `# SOURCE:
   [PENDING] (= gas_availability_factor)` (unchanged)
6. `"storage": min(1.0, _storage_duration_hours / PERIOD_HOURS["peak"]),`
   in `capacity_credit` — `# SOURCE: [PENDING]` (unchanged)
7. `"hydro": _hydro_dry_avg_mw / _hydro_capacity_mw,` in `capacity_credit`
   — `# SOURCE: [PENDING]` (unchanged)

**In `src/optimize_model.py` and `src/optimize_model_sliced.py`** (one each,
identical):

8. `within_block_peaking_factor` — read via `scenario.get(...,
   1.0)`, documented `SOURCE: [PENDING]`, default `1.0` (new in this task;
   **not** blocking — using the default does not raise, unlike the arm
   period-factors above).

**Removed in this task, no longer outstanding:** `PAR_OBSERVED`,
`PAR_CENTRAL`, `PAR_HIGH` (the three scalar-PAR constants from
`docs/phase_2_3_2_4.md`) no longer exist — replaced by items 1–3 above,
which carry the same pending status under new names and a changed shape
(dict of factors, not a scalar).

**Newly sourced in this task (not pending):**
`RESERVE_MARGIN_SPINNING_MW = 900.0`, cited "NIRP 2024 Section 2.3, p.19 —
spinning reserve requirement."

**Everything else** referenced in either report (`gas_availability_factor =
0.80`, `hydro_baseline_twh = 11.47`, `hydro_capacity_mw = 2678`,
`storage_duration_hours = 4.0`, `season_solar_factor`,
`hydro_season_factor`, `hydro_min_flow_fraction`) carries its own
**pre-existing** `[SOURCE NEEDED]`/`[SOURCE: ...]` marker from before either
of these two tasks and was not touched by this one.

---

## Verification performed

- `ast.parse()` on `src/scenarios.py`, `src/optimize_model.py`,
  `src/optimize_model_sliced.py`, `src/io.py` — all four parse without
  error (this remained available throughout; it does not depend on numpy).
- `grep` census of all 40 files in `scripts/`, plus `tests/test_run.py` and
  `src/optimize_experiments.py`, for `load_scenario(`, `build_model(`,
  `build_model_sliced(`, and hand-built `scenario = {`/`scenario =
  dict(`/`.copy()`/`deepcopy(scenario)` patterns — the basis for the STEP 2
  script survey table above.
- `grep` for `GENSET_CASE` across the whole repository before changing its
  assignment in `io.py`, confirming zero other readers existed (safe to
  change its set-vs-absent behaviour).
- Before the numpy DLL block appeared: exercised the *original* (STEP 1,
  `docs/phase_2_3_2_4.md`) `load_scenario()` logic live — required-arg
  raises, the (now-deleted) 1% disagreement guard, and the arithmetic behind
  `par_block`/`capacity_credit['hydro']` — all confirmed correct, as
  reported in that document.

**Not verified this task, and stated plainly rather than guessed at:**

- The *new* (inverted) `load_scenario()` logic in this task — the
  `peak_observed`/`peak_nirp`/`peak_high` arm dispatch, the deleted 1% guard
  actually being unreachable, the `'nirp_spinning'` reserve-margin
  derivation, and the exact `par_block` arithmetic under the new code path
  — was **not exercised live**, because the numpy DLL block appeared before
  I reached that point. It was checked by careful manual line-by-line trace
  against the STEP 1 `docs/phase_2_3_2_4.md` version (which *was* verified
  live and used the identical formula), and I am confident in it, but "I
  traced it carefully" is not the same claim as "I ran it," and I am not
  presenting it as the latter.
- `build_model()`/`build_model_sliced()` were not called (no Pyomo
  `ConcreteModel` was constructed) in either task — no solver was invoked,
  no runner script was executed, and the smoke suite was not run, per the
  task's explicit instruction.
- Whether the relocated guard block, in practice, actually fires before any
  *other* exception a hand-built scenario might trigger elsewhere in either
  function — I verified it is textually first in execution order (right
  after `T` is established), but did not run a hand-built scenario through
  it to observe the failure live.
