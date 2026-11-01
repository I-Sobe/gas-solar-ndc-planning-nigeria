# Phase 2.12 — Vintage-based gas retirement arms

Scope: `src/scenarios.py`, `src/optimize_model.py`,
`src/optimize_model_sliced.py`, and the 11 runners already carrying
`--gas-retirement-case`. Diagnose-and-edit only; nothing was run (no
solver, no runners, no smoke suite). Two categories of non-model checks
were used to verify correctness: `python3 -m py_compile` / `compileall`
after every edit, and direct calls to the new pure-Python loader/series
functions and to `load_scenario()` itself (scenario-dict construction only
— no `build_model`/`build_model_sliced`/`solve_model` call anywhere) to
confirm the reconciliation arithmetic and the four report-year figures
below. Neither builds nor solves an LP.

Not modified, per the task's explicit exclusion list: peak-shape arms,
`par_block`, `within_block_peaking_factor`, `storage_cc_case`,
`gas_availability_case`, `gas_build_case`, the replacement-capex
mechanism, the firm-capacity constraint's form, `gas_retire_none`, and
`gas_retire_proxy` (both arms' own values/behavior are byte-for-byte
unchanged from 2.11 — see Step 1 below for how that is guaranteed
structurally, not just by intent).

This phase was blocked at Step 0 in an earlier session: `data/capacity/raw/gas_plant_vintages.csv`
and `docs/gas_retirement_provenance.md` did not exist anywhere in the
repository. Both files have since been added; this report supersedes that
earlier partial report (the same file path, overwritten in full).

---

## STEP 0 — Inventory

### `gas_retirement_scenarios()`, quoted (pre-edit, 2.11 state)

```python
def gas_retirement_scenarios() -> dict[str, dict[str, float | int] | None]:
    """
    Gas baseline fleet retirement arm (2.11, Step 2). Mirrors genset_case/
    peak_multiple_case/storage_cc_case/gas_availability_case discipline:
    REQUIRED, no default -- see the _GAS_RETIREMENT_REQUIRED sentinel above
    load_scenario().
    ...
        gas_retire_none    (mw_per_year=0.0, start_year=None): ...
        gas_retire_proxy   (mw_per_year=680.0, start_year=2035): ...
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

### Every consumer of `gas_baseline_retirement_mw_per_year` / `gas_baseline_retirement_start_year` (pre-edit)

Direct readers:

- **`src/optimize_model.py`, lines 392–393:**
  ```python
      retirement_start = scenario.get("gas_baseline_retirement_start_year", None)
      retirement_rate = scenario.get("gas_baseline_retirement_mw_per_year", 0.0)
  ```
- **`src/optimize_model_sliced.py`, lines 496–497:** identical.

Writers:

- **`src/scenarios.py`, `load_scenario()`, lines 1568–1569** (the normal
  path, resolved from the selected `gas_retirement_case` arm).
- **`scripts/09b_run_rel1_retirement_sensitivity.py`, line 122** — a
  freestanding sensitivity script that bypasses the arm and hand-overrides
  the rate directly (`scenario["gas_baseline_retirement_mw_per_year"] =
  ret_rate`). Not one of the 11 runners in scope for Step 3; already
  documented as non-functional independent of any phase (missing several
  other required `load_scenario()` kwargs) in the 2.11 report. Untouched
  here — it does not reference `gas_retirement_case` or
  `gas_retire_sourced` at all (confirmed by grep).

No other file references either key.

### `gas_cap_rule` in both builders, quoted (pre-edit)

**`src/optimize_model.py`, lines 396–404:**
```python
    def gas_cap_rule(m, t):
        year_t = int(years[t])
        # Cumulative retirement of baseline fleet
        if retirement_start is not None and year_t >= retirement_start:
            retired = retirement_rate * (year_t - retirement_start + 1)
            retired = min(retired, gas_baseline)  # cannot retire more than exists
        else:
            retired = 0.0
        return gas_baseline - retired + sum(m.gas_add[k] for k in range(0, t + 1))

    m.gas_capacity_mw = pyo.Expression(T, rule=gas_cap_rule)
```

**`src/optimize_model_sliced.py`, lines 499–506:** identical in structure
(the `min()` inlined rather than a two-line assignment).

### Can it accept a per-year series? What would need to change?

**Yes — no Pyomo-level obstruction.** `gas_cap_rule` is a plain Python
closure: `retirement_start`/`retirement_rate` are ordinary Python scalars
captured before the rule is defined, and `retired` is computed fresh, in
pure Python, for each `(m, t)` Pyomo calls while building
`m.gas_capacity_mw`. A per-year series is no harder to consume than the
scalar formula — the surrounding expression
(`gas_baseline - retired + cumulative gas_add`) is unchanged in shape.
**What changed** (implemented in Step 1 below): both builders now read an
additional optional scenario key, `gas_baseline_retirement_series_mw` (a
per-model-year *cumulative* MW list); `gas_cap_rule` checks it first and
only falls through to the existing flat formula when it is `None`. A
length-alignment guard was added (mirroring the existing `len(peak_mw) !=
len(T)` pattern elsewhere in both builders).

---

## STEP 1 — Per-year retirement series

### Vintage schedule is lumpy — confirmed against the actual CSV

Using `data/capacity/raw/gas_plant_vintages.csv` at 30-year life:
commissioning year 2005 → `retirement_year = 2035`, and two plants share
that commissioning year: **Okpai_1 (480 MW) + Omotosho_1 (335 MW) = 815
MW retiring in 2035** — matching the task's own example exactly. No plant
commissioned in 2011 → **nothing retires in 2041**. Three plants
commissioned in 2013 (Omotosho_2, Ihovbor_1, Geregu_2 — 500+500+435) →
**1,435 MW retiring in 2043** — also an exact match. `rate * (year - start
+ 1)` cannot produce this shape (it is monotonically smooth by
construction); only a per-event series can.

### Mechanism extension

**`src/optimize_model.py`** (inserted between `retirement_rate = ...` and
`gas_cap_rule`'s definition):

```diff
     retirement_start = scenario.get("gas_baseline_retirement_start_year", None)
     retirement_rate = scenario.get("gas_baseline_retirement_mw_per_year", 0.0)
+    # gas_baseline_retirement_series_mw (2.12, Step 1): an optional
+    # per-model-year CUMULATIVE retired-MW list (e.g. from
+    # scenarios.gas_vintage_retirement_series_mw()), for retirement
+    # schedules too lumpy for the flat rate * (year - start + 1) formula
+    # below to represent. None for gas_retire_none/gas_retire_proxy,
+    # which are unchanged and fall through to the flat formula exactly
+    # as in 2.11.
+    retirement_series = scenario.get("gas_baseline_retirement_series_mw", None)
+    if retirement_series is not None and len(retirement_series) != len(T):
+        raise ValueError(
+            f"scenario['gas_baseline_retirement_series_mw'] has "
+            f"{len(retirement_series)} entries for {len(T)} model years. "
+            "Refusing to build with a misaligned retirement series."
+        )

     def gas_cap_rule(m, t):
         year_t = int(years[t])
         # Cumulative retirement of baseline fleet
-        if retirement_start is not None and year_t >= retirement_start:
+        if retirement_series is not None:
+            retired = min(retirement_series[t], gas_baseline)  # cannot retire more than exists
+        elif retirement_start is not None and year_t >= retirement_start:
             retired = retirement_rate * (year_t - retirement_start + 1)
             retired = min(retired, gas_baseline)  # cannot retire more than exists
         else:
             retired = 0.0
         return gas_baseline - retired + sum(m.gas_add[k] for k in range(0, t + 1))
```

**`src/optimize_model_sliced.py`**: identical change (the `min()` stays
inlined in the `elif` branch, matching that file's existing style).

This is additive: `gas_retire_none`/`gas_retire_proxy` never populate
`gas_baseline_retirement_series_mw` (see Step 2), so `retirement_series`
is always `None` for those two arms and the `elif`/`else` branches execute
byte-for-byte as in 2.11 — structurally guaranteeing the "do not touch"
instruction rather than relying on not having edited their entries.

### Loader

**`src/scenarios.py`** — new module-level constant `GAS_BASELINE_MW =
10947` (extracted from the previously inline literal at the
`"gas_baseline_mw"` scenario-dict entry, same SOURCE comment, no value
change — done so the new loader's guard checks against the *same* number
`load_scenario()` uses, not a second copy), plus two new functions
inserted between `gas_availability_scenarios()` and
`gas_retirement_scenarios()`:

- **`_load_gas_plant_vintages(csv_path=None)`** — reads the CSV with
  `csv.DictReader` (matching the existing style in `src/gas_supply.py`'s
  `gas_available_power()`, rather than introducing a new pandas dependency
  to this file), excludes `AES_1`/`Gbarain_1` (see reconciliation below),
  and **raises** if the surviving total is not exactly `GAS_BASELINE_MW`
  (a structural guard against the CSV and the baseline silently drifting
  apart in a future edit).
- **`gas_vintage_retirement_series_mw(years, technical_life_years,
  csv_path=None)`** — computes `retirement_year = commissioning_year +
  technical_life_years` per plant, sums MW by retirement year, and returns
  a cumulative series aligned to `years`, **capped at `GAS_BASELINE_MW`**
  (the Step 1 guard) exactly mirroring `gas_cap_rule`'s own
  `min(retired, gas_baseline)`.

Full text of both functions and the extended `gas_retirement_scenarios()`
is in the working tree; see the diff summary at the end of this document
for the complete before/after.

**Verified directly** (loader + series function called standalone — no
model, no solve):

```
>>> plants = _load_gas_plant_vintages()
>>> len(plants), sum(p["installed_mw"] for p in plants)
(23, 10947.0)
```

### Guard reconciliation — CSV (11,361 MW) vs `gas_baseline_mw` (10,947 MW)

**Confirmed and reconciled by exclusion, not scaling.** Summed the CSV's
25 rows directly: **11,361 MW**. `11,361 − 294 (AES_1) − 120 (Gbarain_1) =
10,947` — an exact match to `gas_baseline_mw`, and consistent with the
`GAS_BASELINE_MW` comment already committed in phase 2.10 ("AES_1 (294
MW) and Gbarain_1 (120 MW) are licensed... but carry no availability data
and are excluded"). The CSV's own `note` column corroborates this
independently: AES_1's row reads "Not present in NERC 2026/Q1 availability
table," and Gbarain_1's reads the same.

**Reconciliation implemented:** `_load_gas_plant_vintages()` excludes
these two plants BY NAME (`_GAS_VINTAGE_EXCLUDED_PLANTS = frozenset({"AES_1",
"Gbarain_1"})`) — the vintage retirement series retires MW from the SAME
23-plant, 10,947 MW fleet that `gas_baseline_mw` already represents, not
from the CSV's 25-plant superset. This is an exclusion of two specific,
named, already-documented plants — not a scaling factor applied across
all 25 rows (which the task explicitly ruled out, and which would also be
wrong: AES_1 and Gbarain_1 aren't "slightly overcounted," they are
absent from the tracked fleet entirely). The loader additionally *raises*
if the post-exclusion total isn't exactly `GAS_BASELINE_MW`, so a future
CSV edit that changes this arithmetic fails loudly rather than silently
drifting.

**A pre-2025 retirement event is not counted as a retirement.** A second,
separate reconciliation question arose while implementing the series: 4
of the 23 surviving plants (Sapele_Steam_1, Egbin_1, Delta_1, Afam_1— all
1980s vintage) compute a `commissioning_year + 30` before 2025 (the model
horizon start). These plants are observably still operating (they are
part of `GAS_BASELINE_MW`, itself measured from live 2024/2026 NERC data),
so `gas_vintage_retirement_series_mw()` only counts retirement events with
`retirement_year >= years[0]` — an event before the horizon starts is
evidence the flat-life assumption undershoots that plant, not evidence
the plant is gone. Verified this decision against the provenance memo's
own Section 3/4 numbers (see "Verification against the provenance memo"
below): the memo's percentages (68% / 77% / 36% "retires by 2054") only
reconcile under exactly this convention — the alternative reading (banking
pre-horizon retirements into the cumulative series from year one) produces
different, non-matching totals. This was not an assumption invented for
this task; it is the convention the memo's own already-computed figures
imply, confirmed arithmetically rather than guessed.

---

## STEP 2 — The three arms

**`gas_retirement_scenarios()`, new return value:**

```python
return {
    "gas_retire_none":       {"mw_per_year": 0.0, "start_year": None},
    "gas_retire_proxy":      {"mw_per_year": 680.0, "start_year": 2035},
    "gas_retire_vintage_30": {"technical_life_years": 30},
    "gas_retire_vintage_40": {"technical_life_years": 40},
    "gas_retire_vintage_50": {"technical_life_years": 50},
}
```

`gas_retire_none`/`gas_retire_proxy`'s entries are untouched, character
for character. `gas_retire_sourced` is gone from the dict entirely (not
kept as a `None`-valued deprecated key).

Each vintage arm's docstring entry carries the required SOURCE line
verbatim: *"SOURCE: ASSUMED — analyst vintage estimate, not traced to a
Nigerian source. See docs/gas_retirement_provenance.md and
docs/limitations_register.md L8."* The `gas_retire_vintage_40` entry
additionally notes, per the task's explicit instruction: *"Retires
Delta_1 in 2025 and Egbin_1 in 2026 — BOTH CURRENTLY OPERATING — because a
flat life measured from original commissioning misrepresents a
rehabilitated fleet."* (Verified: `Delta_1` commissioned 1985 + 40 = 2025;
`Egbin_1` commissioned 1986 + 40 = 2026 — both exact.)

**`gas_retire_sourced` deletion made explicit and loud.** A new named
constant `_GAS_RETIRE_SOURCED_REMOVED = "gas_retire_sourced"` and an
explicit check in `load_scenario()`:

```python
if gas_retirement_case == _GAS_RETIRE_SOURCED_REMOVED:
    raise ValueError(
        "gas_retirement_case='gas_retire_sourced' was REMOVED in 2.12 "
        "and no longer exists. Its replacement is an ANALYST "
        "ESTIMATE, not a traced source -- pass 'gas_retire_vintage_30' "
        "([CENTRAL]), 'gas_retire_vintage_40', or "
        "'gas_retire_vintage_50' instead. See "
        "docs/gas_retirement_provenance.md and "
        "docs/limitations_register.md L8."
    )
```

placed before the generic `"Unknown gas_retirement_case"` fallback, so a
scenario still naming the removed arm gets a specific message pointing at
the replacement, not a bare "unknown key" error. Verified directly:

```
>>> load_scenario(..., gas_retirement_case="gas_retire_sourced")
ValueError: gas_retirement_case='gas_retire_sourced' was REMOVED in 2.12
and no longer exists. Its replacement is an ANALYST ESTIMATE...
```

**Arm-shape resolution in `load_scenario()`:** the resolved arm dict is
now branched on its keys (`"technical_life_years"` → vintage;
`"mw_per_year"` → flat) rather than on a `None` check, since there is no
longer a pending/`None`-valued arm in this arm-set. For a vintage arm,
`gas_baseline_retirement_mw_per_year`/`_start_year` are left `None` (the
builders' flat-branch is simply never reached, since `retirement_series`
is populated instead) and `gas_baseline_retirement_series_mw` is computed
via `gas_vintage_retirement_series_mw(years, technical_life_years)` once
`years` is known.

**Verified all five arms resolve correctly** (scenario-dict construction
only, no model build):

| arm | start_year | mw_per_year | series[0:3] | series[-1] |
|---|---|---|---|---|
| `gas_retire_none` | `None` | `0.0` | `None` | `None` |
| `gas_retire_proxy` | `2035` | `680.0` | `None` | `None` |
| `gas_retire_vintage_30` | `None` | `None` | `[0.0, 0.0, 0.0]` | `7281.0` |
| `gas_retire_vintage_40` | `None` | `None` | `[900.0, 2220.0, 2220.0]` | `8486.0` |
| `gas_retire_vintage_50` | `None` | `None` | `[0.0, 0.0, 0.0]` | `3766.0` |

The two flat arms' `start_year`/`mw_per_year` values are identical to
their 2.11 values — confirming "do not touch" held in practice, not just
in the diff.

---

## STEP 3 — Wire and tag

`GAS_RETIREMENT_CASE_CHOICES` extended in all 11 runners:

```diff
-GAS_RETIREMENT_CASE_CHOICES = ("gas_retire_none", "gas_retire_proxy", "gas_retire_sourced")
+GAS_RETIREMENT_CASE_CHOICES = ("gas_retire_none", "gas_retire_proxy", "gas_retire_vintage_30", "gas_retire_vintage_40", "gas_retire_vintage_50")
```

Files: `scripts/01_run_baseline.py`, `scripts/01b_run_baseline_sliced.py`,
`scripts/02_run_eaas.py`, `scripts/02_run_ndc_caps.py`,
`scripts/03_run_ndc_eaas.py`, `scripts/04_run_fin2_tariff_sweep.py`,
`scripts/22_salvage_sensitivity.py`, `scripts/23_run_selfgen_frontier.py`,
`scripts/24_run_leakage_arm.py`, `scripts/25_run_obligation_trajectory.py`,
`scripts/99c_run_genset_arm_sensitivity.py`. No other code in these files
needed to change — `--gas-retirement-case` was already required/wired
through to `load_scenario(gas_retirement_case=...)` and into every output
tag/summary in 2.11; only the set of legal values changed.

**Confirmed no runner hard-codes `gas_retire_sourced`.** Two checks:
`grep -rn "gas_retire_sourced" scripts/*.py` (after the edit) returns zero
matches; `grep -rn "gas_retirement_case\s*=\s*['\"]gas_retire" scripts/*.py
src/*.py` (any direct literal assignment of a `gas_retire*` value to
`gas_retirement_case`, anywhere) returns zero matches outside the new
error-message string in `src/scenarios.py` itself — every runner threads
`args.gas_retirement_case` through from its CLI argument, never a literal.

Two stale comments in the presence-only builder guards (`src/optimize_model.py`
line ~211, `src/optimize_model_sliced.py` line ~322) referenced
`"SOURCE: [PENDING] raise for gas_retire_sourced"`, which is no longer
accurate (the arm doesn't exist to have a pending-source raise); updated
to reference "the removed-name raise for 'gas_retire_sourced', 2.12"
instead.

`scripts/09b_run_rel1_retirement_sensitivity.py` (not one of the 11) does
not reference `gas_retirement_case` or `gas_retire_sourced` at all —
confirmed by grep, so nothing there needed touching.

---

## STEP 4 close-out

### Cumulative retirement series, 2025 / 2035 / 2045 / 2054 (verified directly, no solve)

| arm | 2025 | 2035 | 2045 | 2054 | remaining 2054 |
|---|---|---|---|---|---|
| `gas_retire_vintage_30` [CENTRAL] | 0 | 915 | 5,540 | 7,281 | 3,666 |
| `gas_retire_vintage_40` | 900 | 2,946 | 3,861 | 8,486 | 2,461 |
| `gas_retire_vintage_50` | 0 | 1,620 | 3,666 | 3,766 | 7,181 |
| *`gas_retire_proxy` (for comparison, unchanged)* | *0* | *10,267* | *10,947* | *10,947* | *0* |

(`gas_retire_proxy`'s row: `680 × (year − 2034)`, capped at 10,947 —
reaches the full fleet by 2051, per the 2.11 report.)

### Verification against the provenance memo

The memo's Section 3 table (`retires by 2054` / `remaining 2054`,
percentages against 11,361 MW) is reproduced by the SAME algorithm applied
to the **full, unreconciled 25-plant list** (i.e. without excluding
AES_1/Gbarain_1) — confirmed by direct recomputation: 30yr → 7,695 MW
(68%) retired, 3,666 MW remaining; 40yr → 8,780 MW (77%), 2,581 MW
remaining; 50yr → 4,060 MW (36%), 7,301 MW remaining — all matching the
memo exactly. **The figures in the table above are the reconciled
version** (AES_1/Gbarain_1 excluded, matching `GAS_BASELINE_MW`), so they
differ slightly from the memo's illustrative numbers by construction —
e.g. `gas_retire_vintage_30`'s "remaining 2054" is 3,666 in both versions
only because AES_1 (retires 2031) and Gbarain_1 (retires 2050) both fall
inside the 2025–2054 window at 30-year life, so excluding them removes
exactly 414 MW from both the numerator and the denominator and the
"remaining" figure happens to come out the same; at 40- and 50-year life
the two versions diverge (2,461 vs 2,581; 7,181 vs 7,301) because one or
both plants' retirement years fall outside the window at those longer
lives. This divergence is reported, not smoothed over: the reconciled
table above is the one that actually feeds `gas_cap_rule`.

### Confirmation

- `gas_cap_rule` (both builders) now accepts either the existing flat rate
  or a per-year cumulative series; `gas_retire_none`/`gas_retire_proxy`
  are unchanged, structurally (they never populate the series key).
- The CSV-to-`gas_baseline_mw` mismatch is reconciled by excluding AES_1
  and Gbarain_1 by name — not by scaling — with a loader-level guard that
  raises if the reconciliation ever stops holding exactly.
- `gas_retire_vintage_30` / `_40` / `_50` exist, each carrying the required
  `SOURCE: ASSUMED` line and pointers to
  `docs/gas_retirement_provenance.md` and `docs/limitations_register.md`
  L8; `gas_retire_vintage_40`'s docstring entry notes the Delta_1/Egbin_1
  anomaly verbatim.
- `gas_retire_sourced` is deleted from `gas_retirement_scenarios()` and
  raises a specific, named error (not a generic "unknown key" error) if a
  scenario still requests it.
- `--gas-retirement-case`'s choices tuple is extended in all 11 runners;
  no runner hard-codes `gas_retire_sourced` anywhere.
- All touched files (`src/scenarios.py`, both builders, 11 scripts) pass
  `python3 -m py_compile` / `compileall`.
