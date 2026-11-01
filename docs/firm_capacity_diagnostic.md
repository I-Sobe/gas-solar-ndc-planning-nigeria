# Firm-capacity constraint diagnostic — wiring into the sliced baseline runner

Scope: expose slack/dual on `m.firm_capacity_constraint` from
`firm_capacity_diagnostic()` (`src/optimize_model_sliced.py:1063`) in
`scripts/01b_run_baseline_sliced.py`. No solver, runner, or smoke suite was
executed — this is wiring only, verified with `python -m py_compile`.

## STEP 0 — does a dual Suffix exist?

**Yes.** `m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)` is already declared
at `src/optimize_model_sliced.py:226`, inside `build_model_sliced()`, before
`solve_model()` is invoked (`scripts/01b_run_baseline_sliced.py:86`). No
addition was needed here.

Solver/MIP check: every `pyo.Var(...)` in `optimize_model_sliced.py` uses
`domain=pyo.NonNegativeReals` — there are no `Binary` or `Integer` domains
anywhere in the sliced model. `solve_model()` (`src/optimize_model.py:1160`)
calls CBC (`solver="cbc"`, default) with no integer restrictions, so this is
solved as a pure LP and CBC's returned duals are valid, not the
degenerate/undefined duals you'd get off a MIP relaxation.

One correction to the task brief: `firm_capacity_diagnostic` was **not**
already imported in the runner (only `build_model_sliced, annual_totals,
slice_detail, SLICES` were). It had to be added, or the new call would raise
`NameError`. See Diff 1 below.

## Year-list variable name

`scripts/01b_run_baseline_sliced.py:65`: `years = scenario["years"]`
(lowercase). There is no `YEARS` (uppercase) identifier anywhere in this
file — confirmed by direct read, not grep alone.

## STEP 1–3 — diffs applied

**Diff 1** — add the missing import (`scripts/01b_run_baseline_sliced.py`,
import block):
```diff
 from src.optimize_model_sliced import (
-    build_model_sliced, annual_totals, slice_detail, SLICES
+    build_model_sliced, annual_totals, slice_detail, SLICES, firm_capacity_diagnostic
 )
```

**Diff 2** — call the diagnostic and print the STEP 2 summary. Placed after
the summary JSON is written and after all other diagnostics (`dv`,
`gas_shadow`, `emissions_by_year`, `slice_2025`, `slice_2045`, etc.) are
built, and before `diag_out` (and therefore before the
`diagnostics__{args.peak_case}.json` write, which now happens ~19 lines
later):
```diff
     with open(RESULTS_DIR / f"summary__{args.peak_case}.json", "w") as f:
         json.dump(json_safe(summary), f, indent=2)
 
+    # ------------------------------------------------------------
+    # Firm-capacity constraint diagnostic (2.6, F0 Step 2.3): slack (MW)
+    # and dual per year on m.firm_capacity_constraint.
+    # ------------------------------------------------------------
+    firm_capacity = firm_capacity_diagnostic(m, years)
+    print("=" * 55)
+    print("FIRM CAPACITY CONSTRAINT -- slack (MW) and dual by year")
+    binding_years = 0
+    for y in years:
+        y = int(y)
+        rec = firm_capacity[y]
+        if abs(rec["slack_mw"]) < 1.0:
+            binding_years += 1
+        if y == 2025 or y == 2054 or (y - 2025) % 5 == 0:
+            print(f"  {y}: slack={rec['slack_mw']:12.2f} MW   dual={rec['dual']}")
+    print(f"Binding years (|slack_mw| < 1.0): {binding_years} / {len(years)}")
+    print("=" * 55)
+
     diag_out = {
```
Notes on the print filter: `MODEL_END_YEAR = 2054` (`src/scenarios.py:282`)
and the model horizon runs 2025–2054, so "2025, 2054, and every fifth year"
is implemented as `y == 2025 or y == 2054 or (y - 2025) % 5 == 0`, which
prints 2025, 2030, 2035, 2040, 2045, 2050, 2054. The binding-year count is
computed over **all** years in `years`, not just the printed subset, per the
task's "out of the total" wording.

**Diff 3** — persist the full per-year result under a `firm_capacity` key in
the dict already dumped to `diagnostics__{args.peak_case}.json`:
```diff
         "selfgen_twh_by_year": annual_series("selfgen"),
+        "firm_capacity": firm_capacity,
     }
```

## JSON key path

`diagnostics__{args.peak_case}.json` → top-level key `"firm_capacity"` →
per-year dict, e.g. `firm_capacity["2025"]["slack_mw"]`,
`firm_capacity["2025"]["dual"]` (string year keys after `json_safe`/`json.dump`
round-trip, matching every other per-year block already in this file, e.g.
`gas_shadow_price_usd_per_twh_th_by_year`). Routed through the same
`json_safe(diag_out)` call as the rest of `diag_out` — no second file, no
separate serialisation path.

`firm_capacity_diagnostic()` itself (`src/optimize_model_sliced.py:1063`) was
not modified — it already returns `{"slack_mw": float, "dual": float|None}`
per year, keyed by `int(year)`.
