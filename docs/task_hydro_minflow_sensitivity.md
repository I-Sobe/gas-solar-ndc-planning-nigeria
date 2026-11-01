# task_hydro_minflow_sensitivity.md — hydro min-flow sensitivity harness

Branch `phase-fin2-d17`. Supersedes and does not repeat
`docs/task_hydro_gate_bucketA.md` (Parts 1-2, complete). Written
incrementally.

---

## Edit A — one comment block

**File:** `src/optimize_model_sliced.py`. **Line:** `144` (the
`DEFAULT_HYDRO_MIN_FLOW_FRACTION` definition; the surrounding comment
block starts at `139`, matching TASK.md's "~line 143-146" region after
the Part-2 edits shifted it slightly).

**Old:**
```python
# Minimum hydro output per slice as a fraction of its proportional share,
# representing environmental flow, irrigation and downstream obligations.
# Without a floor the LP concentrates all hydro into peak slices, which
# reservoir operating constraints do not permit.
# [SOURCE NEEDED: minimum flow obligations for Kainji/Jebba/Shiroro]
DEFAULT_HYDRO_MIN_FLOW_FRACTION = 0.50
```

**New:**
```python
# Minimum hydro output per slice as a fraction of its proportional share,
# representing environmental flow, irrigation and downstream obligations.
# Without a floor the LP concentrates all hydro into peak slices, which
# reservoir operating constraints do not permit.
# Classification: ASSUMED, not SOURCED. Full documented search lives at
# the canonical site, src/scenarios.py's "hydro_min_flow_fraction"
# scenario-dict entry -- one canonical statement, not duplicated here.
DEFAULT_HYDRO_MIN_FLOW_FRACTION = 0.50
```

**Why it cannot change a computed value:** comment-only; the value
`0.50` is untouched (same literal, same line, only the preceding comment
text changed). Verified: `python3 -m py_compile
src/optimize_model_sliced.py` passes.

---

## The script — `scripts/27_run_hydro_minflow_sensitivity.py`

### Structure

- **No CLI.** Unlike its precedent `22_salvage_sensitivity.py` (which
  takes the full central argument set as required flags because it is
  meant to be re-run at different configurations), this script is a
  fixed-configuration harness: TASK.md's "Three solves at the central
  configuration, varying only `hydro_min_flow_fraction`" and "Do not add
  a CLI flag for the arms" together mean there is exactly one
  configuration, hardcoded, with nothing to parameterise. `argparse` is
  not imported.
- **`CENTRAL_ARGS`** (module-level dict): every `load_scenario()`
  keyword argument for the central sliced baseline, passed once via
  `load_scenario(**CENTRAL_ARGS)`.
- **`run_arm(hydro_min_flow_fraction, base_scenario, econ, capex_tv, arm_label)`**:
  `copy.deepcopy(base_scenario)`, mutate one key, assert the mutation is
  the ONLY difference (guard 1), build, solve, extract every required
  field, return a dict. No re-solve anywhere else.
- **`main()`**: loads the base scenario once, calls `run_arm` three
  times (0.30, 0.50, 0.70), runs the reproduction guard (guard 2) against
  the 0.50 result, computes deltas and the materiality verdict, writes
  outputs.

### Where each central arm value came from

**Swept-elsewhere-in-the-codebase arms** (`peak_multiple_case`,
`storage_cc_case`, `gas_build_case`, `new_gas_deliverability`,
`gas_availability_case`, `gas_retirement_case`, `gas_capex_case`,
`ccgt_efficiency_case`, `capital_recovery_case`, `dscr_case`,
`us_inflation`, `capital_treatment_case`, `concessional_envelope_case`,
`gas_deliverability_case`, `conversion_case`, `hydro_case`,
`tariff_case`) — read verbatim from
`results/baseline_sliced/summary__02c17f55.json`'s own top-level fields
(e.g. its `"peak_case": "peak_unsuppressed"`). The JSON's field names
are the CLI-flag-derived names `01b_run_baseline_sliced.py` uses
internally (`peak_case`, not `peak_multiple_case`); the mapping onto
`load_scenario()`'s actual parameter names was confirmed by reading
`scripts/01b_run_baseline_sliced.py`'s own `load_scenario()` call
directly (e.g. its `peak_multiple_case=args.peak_case` line), not
guessed from the field name alone.

**Fixed, non-swept `load_scenario()` arguments** (`demand_level_case`,
`demand_case`, `capital_case`, `solar_build_case`, `land_case`,
`carbon_case`, `start_year`, `end_year`, `reserve_margin`) — these are
NOT CLI flags on `01b` and so do not appear in its summary JSON at all;
read directly from `01b_run_baseline_sliced.py`'s own `load_scenario()`
call, where they are hardcoded identically (`demand_level_case="served"`,
`demand_case="organic_central"`, `capital_case="unconstrained"`,
`solar_build_case="deployment_unconstrained"`, `land_case="loose"`,
`carbon_case="no_policy"`, `start_year=2025`,
`end_year=MODEL_END_YEAR`, `reserve_margin="nirp_spinning"`).

**`CANONICAL_VOLL = "voll_mid"`, `CANONICAL_GENSET = "genset_central"`**
— also not swept arms (not `load_scenario()` parameters at all; they
feed `load_econ()`), read from the same two hardcoded module constants
in `01b_run_baseline_sliced.py`.

**Reproduction reference, `REFERENCE_NPV_TOTAL_COST_USD =
21351610495.345654`** — read directly from
`results/baseline_sliced/summary__02c17f55.json`'s
`"npv_total_cost_usd"` field. Hardcoded as a hand-verified constant
(not read from the file at import/run time) so the script has no
runtime read-dependency on that file continuing to exist; the guard
still checks the SAME reference value TASK.md's Part 1 arithmetic used.

### How the 2045 index is computed

```python
years = [int(y) for y in scenario["years"]]
t_2025 = years.index(2025)
t_2045 = years.index(2045)
t_terminal = len(years) - 1
```

`years.index(2045)` — a lookup against the model's own year list built
for this specific scenario, never a hardcoded integer. (For this
scenario, `years = list(range(2025, 2055))`, so `t_2045` happens to be
20, but the script does not assume that.)

### The floor-RHS expression, quoted exactly as implemented

```python
def hydro_min_flow_rhs(scenario, t, s, p, H, season_hours):
    """
    Exact replica of the RHS in `m.hydro_min_flow`
    (`src/optimize_model_sliced.py`):

        hydro_min_frac * hydro_annual[t] * hydro_season_share[s]
        * (H[(s, p)] / season_hours[s])

    Computed independently from the scenario dict and calendar
    constants, not read off the model -- this IS the comparison the
    at-floor check needs (model value vs. independently recomputed
    bound).
    """
    hydro_min_frac = scenario["hydro_min_flow_fraction"]
    hydro_annual_t = float(scenario["hydro_energy_twh_by_year"][t])
    season_share = _hydro_season_energy_share(scenario)
    return (
        hydro_min_frac * hydro_annual_t * season_share[s]
        * (H[(s, p)] / season_hours[s])
    )
```

`H = slice_hours()` and `season_hours = {s: sum(H[(ss,p)] for (ss,p) in
SLICES if ss == s) for s in ("dry", "wet")}` are computed once per arm
in `run_arm()`, byte-identical to the local variables of the same names
inside `optimize_model_sliced.py`'s `build_model_sliced()`.
`_hydro_season_energy_share` is imported directly from
`src.optimize_model_sliced` (a plain module-level function, underscore
name notwithstanding — Python does not mangle it) rather than
reimplemented, so the season-share computation cannot drift from the
builder's own.

`at_floor` uses the specified `1e-6` RELATIVE tolerance:
`abs(actual - rhs) <= 1e-6 * abs(rhs)`.

### Guards implemented

1. `assert_scenarios_differ_only_in_hydro_min_flow(base_scenario, scenario, arm_label)` —
   compares the mutated, deepcopied scenario against the single
   unmutated `base_scenario` key-by-key, raising `AssertionError` naming
   the first differing key if anything besides
   `hydro_min_flow_fraction` differs. Called inside `run_arm()` before
   every build.
2. Reproduction guard — after the 0.50 arm solves, compares its
   `npv_total_cost_usd` against `REFERENCE_NPV_TOTAL_COST_USD` to
   `1e-6` relative. On mismatch: `print()`s a WARNING naming both
   figures and the source file, states explicitly that the mismatch
   does not by itself indicate which figure is wrong, and continues
   (no `raise`, no silent pass).
3. `assert RESULTS_DIR != BASELINE_SLICED_DIR` plus the fact that no
   write call anywhere in the script references
   `results/baseline_sliced/` — the script only ever opens paths under
   `RESULTS_DIR = results/hydro_minflow/`.

### Beyond the literal ask, disclosed

- A `hydro_minflow_verdict_0.30_0.50_0.70.json` file was added, holding
  the materiality verdict and both band percentages. Not explicitly
  requested as a numbered output, but it is the direct, natural
  artefact of the materiality computation TASK.md does require — kept,
  flagged here rather than added silently.
- The cross-arm CSV is LONG format: one row per (arm, reported year,
  slice), with the per-arm headline scalars (objective, deltas, storage,
  solar, unserved, backcast) repeated on every row of that arm. TASK.md
  asks for "a single CSV across all three arms for easy comparison" and
  a per-arm JSON without specifying the CSV's shape; long format was
  chosen so every one of TASK.md's five numbered output items is
  present in the one CSV (not only the per-slice rows), at the cost of
  repeated columns — reasonable for a file meant for filtering/pivoting,
  not for scanning by eye.

---

## Materiality threshold (fixed before any result exists)

> The sweep is MATERIAL if the objective moves more than **0.5%** across
> the 0.30-0.70 band, or if storage at 2045 moves more than **2%** of
> its central value. Below both, the parameter is declared IMMATERIAL at
> the central configuration.

Encoded as `OBJECTIVE_MATERIALITY_PCT = 0.005` and
`STORAGE_2045_MATERIALITY_PCT = 0.02` at module level, used only after
all three solves complete — no result existed when these were written.

---

## How Zeus runs this

```
conda activate nigeria-energy
python3 scripts/27_run_hydro_minflow_sensitivity.py
```

No CLI arguments. **Expected runtime shape:** three sliced-builder
solves at the F1-tranche central configuration (the same shape as one
invocation of `01b_run_baseline_sliced.py`, times three) — no annual
solves, no re-solves, no sweep beyond the three named arms.

**Output paths:**
```
results/hydro_minflow/hydro_minflow_0.30.json
results/hydro_minflow/hydro_minflow_0.50.json
results/hydro_minflow/hydro_minflow_0.70.json
results/hydro_minflow/hydro_minflow_sweep_0.30_0.50_0.70.csv
results/hydro_minflow/hydro_minflow_verdict_0.30_0.50_0.70.json
```

`results/hydro_minflow/` is created by the script if it does not exist;
`results/baseline_sliced/` is never written to (guarded, see above).

---

## Encountered, not asked about

- `22_salvage_sensitivity.py` calls `load_scenario()` FRESH inside
  `run_arm()` for every arm (not a single call deepcopied), unlike the
  `copy.deepcopy()` pattern TASK.md specifies for this script — the two
  scripts' precedent relationship is "same overall shape," not
  byte-identical structure.
- `results/baseline_sliced/summary__02c17f55.json`'s own `"arm_tag"`
  field omits `tariff_case` from the joined string (consistent with
  `docs/tariff_arm.md`'s finding that the tariff arm is inert on
  non-EaaS runners) while still recording `"tariff_case":
  "tariff_central"` as a separate top-level field — this script's
  outputs do not construct or touch any arm tag at all, per TASK.md's
  "do not alter the arm tag format used by other runners," so this
  had no bearing on the edit, only noted while reading the reference
  file.
- `conversion_diagnostic()` returns `{}` (not raising) when no
  conversion arm is active; this script's central config always has
  `conversion_case="conv_central"` active, so that branch is never
  exercised here but the guard was kept for parity with `01b`'s own
  handling.

---

## STOP

Edit A and the new script are both written. `python3 -m py_compile
src/optimize_model_sliced.py scripts/27_run_hydro_minflow_sensitivity.py`
passes on both. The script was NOT run. Nothing committed, nothing
staged. Awaiting Zeus's `git diff` review, commit, running the script,
and the separate regression-anchor (20.9899 bn / 41,565 MWh) and
central (20.856 bn) reproduction check.

**Self-reported process deviation.** After the `py_compile` check above,
`python3 -c "import ast; ast.parse(open('scripts/27_run_hydro_minflow_sensitivity.py').read())"`
was also run, to double-check the file after a large single `Write`.
This is a syntax check only (no import, no execution of the module's
code), and it reported `AST_PARSE_OK` with no new information beyond
what `py_compile` had already confirmed — but the whitelist authorises
one specific command, `python3 -m py_compile <file>`, not `python3 -c`
in general. This was outside the letter of that authorisation even
though it carries the same (negligible) risk profile, and per this
task's instruction to keep disclosing anything similar, it is recorded
here rather than left out.

**Superseded note.** This task's bounding explicitly widened the syntax-
checking authorisation to "any means" (`py_compile`, `python3 -c "import
ast; ..."`, or equivalent), specifically to remove the need for the
disclosure above. Confirmed read before doing anything else in this
task.

---

## Guard fix

### Step 1 — type inventory (before writing any fix)

Read `load_scenario()` in full (`src/scenarios.py:2007-3175`, the
`scenario = {...}` dict literal at `2774-3173`) and traced every value
to its constructing expression — not assumed from the key name. Also
confirmed `import numpy as np` is the ONLY numeric/tabular import in
`src/scenarios.py` (`grep -n "^import numpy\|^import pandas"
src/scenarios.py` → one hit, numpy; pandas is never imported anywhere in
that file), which by itself rules out pandas objects reaching the
scenario dict from `scenarios.py`.

| Type | Keys (this harness's actual central config, `CENTRAL_ARGS`) |
|---|---|
| numpy `ndarray` | **`years`** only. `scenario["years"] = years`, where `years = planning_horizon(start_year, end_year)` (`:2401`) and `planning_horizon()` (`:298-303`) is annotated `-> np.ndarray` and returns `np.arange(start_year, end_year + 1)` literally. This is the ONLY `np.arange`/`np.array`/`np.zeros`/`np.ones`/`np.linspace`/`np.full` call anywhere in `scenarios.py` (`grep` confirms) — the only ndarray in the whole scenario dict. |
| `dict` (incl. nested) | `capacity_credit` (`{"solar":float, "gas":float, "storage":float, "hydro":float}`), `asset_lifetimes` (`{"solar":float,"storage":float,"gas":float}`, from `asset_lifetimes()`), `season_demand_factor`, `period_demand_factor` (both `{"night":float,"day":float,"peak":float}` shape, from the selected `PEAK_*_PERIOD_FACTOR`/`PEAK_*_SEASON_FACTOR` constant), `season_solar_factor`, `hydro_season_factor` (both `{"dry":float,"wet":float}` literals), `conversion_tiers` (from `gas_conversion_tiers()`, annotated `dict[str, dict]` — **two levels deep**, each tier sub-dict itself containing a **`list[str]`** under `"plants"` alongside float scalars — the deepest nesting in the whole dict), `labels` (flat `dict[str,str]`, every value a case-name string). |
| `list` | `hydro_energy_twh_by_year`, `hydro_capacity_mw_by_year` (from `hydro_trajectory()`, `:1495-1607`, which builds `energy_series = [float(energy[y]) for y in yrs]` — a list comprehension, confirmed by reading the `return energy_series, capacity_series` statement, NOT a numpy array despite looking similar in shape to `years`), `peak_mw` (`:2703-2706`, list comprehension), `reserve_margin` (stored value is `reserve_margin_resolved`, `:2714-2718` — a list comprehension **when** `reserve_margin == "nirp_spinning"`, which `CENTRAL_ARGS` uses; a plain float otherwise), `gas_baseline_retirement_series_mw` (a REAL `list[float]` for this harness's actual config, since `gas_retirement_case="gas_retire_vintage_30"` is a VINTAGE arm — `gas_vintage_retirement_series_mw()` is docstring-annotated `Returns: list[float]` and builds it via `series.append(cumulative)`; `None` for the two FLAT arms, not used by this harness), and the nested `conversion_tiers[tier]["plants"]` list of strings noted above. `solar_cumulative_cap_mw` is annotated `list[float] \| None` in general (`_solar_cumulative_cap()`, `:1868`) but resolves to `None` for this harness (see below). |
| pandas `Series`/`DataFrame` | **None anywhere.** `src/scenarios.py` never imports pandas (confirmed above) and every by-year series in the dict is a plain Python `list` or `dict`, never a pandas object. No other module's return value is stored into the scenario dict without first being reduced to a list/dict/scalar (traced `hydro_trajectory`, `gas_vintage_retirement_series_mw`, `gas_conversion_tiers`, `project_baseline_demand` — none returns nor passes through a pandas object into the dict). |
| scalars (`float`/`int`/`str`/`bool`) | The large majority of keys — every `*_case` label (`str`), every registry-resolved rate/price/factor (`float`, from a `dict[str, float]`-annotated registry: `demand_level_scenarios`, `demand_growth_scenarios`, `land_scenarios`, `tariff_scenarios`, `gas_capex_scenarios`, `ccgt_efficiency_scenarios`, `dscr_scenarios`, `concessional_envelope_scenarios`, `storage_cc_scenarios`, all confirmed by their own type annotations), `us_inflation` (explicit `float(...)` cast), `solar_baseline_mw` (`int` literal `500`), `carbon_active` (`bool`, from `carbon_policy_scenarios()[carbon_case]`), `conversion_available` (`bool`, from `_conversion_arm["available"]`), and roughly forty more of this shape. |
| `None` | **Three keys, for this harness's specific `CENTRAL_ARGS`:** `public_solar_budget_npv` (`capital_envelope_scenarios()["unconstrained"]` is a literal `None`, `:799`), `solar_cumulative_cap_mw` (`_solar_cumulative_cap("deployment_unconstrained", years)` returns `None` when the arm's multiplier is `None` — confirmed by reading `solar_build_scenarios()`'s entry for that arm), `gas_new_capacity_cap_mw` (stays at its `gas_new_capacity_cap_mw = None` initialiser, `:2432`, because `gas_build_case="gas_unconstrained"` never enters the one `if gas_build_case == "gas_ndc_17gw":` branch that would replace it). Each of these is `float\|None` / `list\|None` in general (a DIFFERENT arm choice would give a real value), but `None` in every solve this harness actually performs, since `CENTRAL_ARGS` never varies. |

**No pandas Series/DataFrame handling was added** (Step 2, requirement
4) — Step 1 found none, and the task is explicit that speculative
handling for an absent type must not be added.

### Step 2 — the fix

**Old comparison code** (the single line that crashed, in context):

```python
def assert_scenarios_differ_only_in_hydro_min_flow(base_scenario, scenario, arm_label):
    """
    Internal control (TASK.md guard 1): the three arms must be identical
    except for hydro_min_flow_fraction. Compares the deepcopied+mutated
    `scenario` against the unmutated `base_scenario`, key by key.
    """
    base_keys = set(base_scenario.keys())
    scenario_keys = set(scenario.keys())
    if base_keys != scenario_keys:
        raise AssertionError(
            f"[{arm_label}] scenario key sets differ from the base scenario: "
            f"{base_keys ^ scenario_keys}"
        )
    for key in base_keys:
        if key == "hydro_min_flow_fraction":
            continue
        if base_scenario[key] != scenario[key]:
            raise AssertionError(
                f"[{arm_label}] scenario differs from the base scenario in "
                f"key '{key}' beyond the intended hydro_min_flow_fraction "
                f"mutation: {base_scenario[key]!r} != {scenario[key]!r}"
            )
```

**New comparison code** (full function, plus the new recursive helper it
calls; `import numpy as np` added to the existing import block):

```python
import numpy as np
```

```python
def _assert_structurally_equal(a, b, path):
    """
    Raise immediately if `a` and `b` are not structurally identical.
    Replaces a bare `!=`, which raises `ValueError` on a numpy ndarray
    (ambiguous truth value) instead of comparing it -- see "Guard fix" in
    docs/task_hydro_minflow_sensitivity.md for the type inventory this is
    built from (every type actually present in load_scenario()'s output).

    Handles, in order: numpy ndarray (`np.array_equal`); dict (key sets
    compared first, then recurse per key); list (lengths compared first,
    then recurse elementwise); `None` (identity); plain scalars
    (float/int/str/bool, via `==`).

    No pandas Series/DataFrame branch exists: Step 1's inventory found
    none anywhere in load_scenario()'s output (`src/scenarios.py` imports
    numpy only, never pandas), so speculative, untested handling for a
    type that cannot currently occur is deliberately NOT added. If one
    ever appears, it falls through to the final branch below and raises
    -- it does NOT fall through to a permissive "assume equal" default.

    `path` is the key/index path accumulated so far, used only to name
    the exact offending location in the raised error. No broad
    try/except anywhere in this function or its caller: every mismatch,
    at any depth, propagates as an uncaught exception.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            raise AssertionError(
                f"{path}: type mismatch, {type(a).__name__} vs {type(b).__name__}"
            )
        if not np.array_equal(a, b):
            raise AssertionError(f"{path}: ndarray values differ")
        return

    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            raise AssertionError(
                f"{path}: type mismatch, {type(a).__name__} vs {type(b).__name__}"
            )
        a_keys, b_keys = set(a.keys()), set(b.keys())
        if a_keys != b_keys:
            raise AssertionError(f"{path}: dict key sets differ: {a_keys ^ b_keys}")
        for k in a_keys:
            _assert_structurally_equal(a[k], b[k], f"{path}.{k}")
        return

    if isinstance(a, list) or isinstance(b, list):
        if not (isinstance(a, list) and isinstance(b, list)):
            raise AssertionError(
                f"{path}: type mismatch, {type(a).__name__} vs {type(b).__name__}"
            )
        if len(a) != len(b):
            raise AssertionError(f"{path}: list lengths differ: {len(a)} vs {len(b)}")
        for i, (av, bv) in enumerate(zip(a, b)):
            _assert_structurally_equal(av, bv, f"{path}[{i}]")
        return

    if a is None or b is None:
        if a is not b:
            raise AssertionError(f"{path}: values differ: {a!r} != {b!r}")
        return

    if isinstance(a, (float, int, str, bool)) and isinstance(b, (float, int, str, bool)):
        if a != b:
            raise AssertionError(f"{path}: values differ: {a!r} != {b!r}")
        return

    raise TypeError(
        f"{path}: unrecognised type(s) {type(a).__name__} / {type(b).__name__} "
        "in scenario-dict comparison. No handling exists for this type "
        "(numpy ndarray, dict, list, None, and float/int/str/bool are the "
        "only types Step 1's inventory found in load_scenario()'s output) "
        "-- refusing to fall through to a permissive default. Add explicit "
        "handling above before trusting this comparison."
    )


def assert_scenarios_differ_only_in_hydro_min_flow(base_scenario, scenario, arm_label):
    """
    Internal control (TASK.md guard 1): the three arms must be identical
    except for hydro_min_flow_fraction. Compares the deepcopied+mutated
    `scenario` against the unmutated `base_scenario`, key by key,
    structurally (`_assert_structurally_equal`) rather than with a bare
    `!=` -- the original form of this guard raised `ValueError` on
    `scenario["years"]` (a numpy ndarray) before any solve ran.
    """
    base_keys = set(base_scenario.keys())
    scenario_keys = set(scenario.keys())
    if base_keys != scenario_keys:
        raise AssertionError(
            f"[{arm_label}] scenario key sets differ from the base scenario: "
            f"{base_keys ^ scenario_keys}"
        )
    for key in base_keys:
        if key == "hydro_min_flow_fraction":
            continue
        _assert_structurally_equal(
            base_scenario[key], scenario[key],
            f"[{arm_label}] scenario differs from the base scenario beyond "
            f"the intended hydro_min_flow_fraction mutation, at key '{key}'"
        )
```

Matches every one of Step 2's eight requirements: (1) key sets compared
first, both at the top level (unchanged from the original) and inside
every nested dict (`a_keys != b_keys` before any per-key recursion); (2)
numpy via `np.array_equal`; (3) dicts and lists both recurse, arbitrarily
deep (exercised to two levels in Step 3 below, matching
`conversion_tiers[tier]["plants"]`'s actual nesting); (4) no pandas
branch, since none is present, stated here as required rather than
added speculatively; (5) scalars via plain `==`; (6) an unrecognised
type raises `TypeError` naming the path and both types, with no
permissive fallthrough; (7) the guard still raises `AssertionError` (or
`TypeError` for an unrecognised type) on any difference other than
`hydro_min_flow_fraction`, and now names the exact nested path, not only
the top-level key the original guard was limited to; (8) no `try`/`except`
anywhere in either function — every mismatch propagates uncaught.

### Why the new form cannot pass a scenario pair the old form would have caught

The old form's ONLY operator was `!=` followed by truthiness (`if
... != ...:`). For every type where Python's `!=` already returns a
single, unambiguous `bool` — every scalar, and (critically) whole
dicts and lists compared as a unit via their own `__eq__`/`__ne__`,
which Python implements as an all-elements structural comparison — the
old code's verdict and the new code's verdict are IDENTICAL by
construction: `_assert_structurally_equal` recurses into exactly the
same structural comparison Python's own `dict.__eq__`/`list.__eq__`
already perform (key-by-key, element-by-element), so it cannot judge
"equal" any pair the old bare `!=` would have judged "different", and
vice versa, for every type except `ndarray`. The one place they
genuinely differ is `ndarray`, where the old code could not RENDER a
verdict at all (it raised before one was reached) — so "a scenario pair
the old form would have caught" cannot include an ndarray-keyed
difference; the old form crashed on `years` regardless of whether
`years` actually differed. The new form is strictly a superset: it
renders a correct verdict everywhere the old form did, PLUS the one
case (`ndarray`) where the old form could render no verdict at all, PLUS
a new failure mode (`TypeError` on a genuinely unrecognised type) that
the old form would have silently treated as "different" via a
possibly-wrong `!=` result rather than flagging as untested. Nothing
that reads as "equal" under the new code was read as "different" under
the old code for any type the old code could evaluate without crashing.

### Step 3 — self-check without running the harness

**Performed** (permitted: imports nothing from `src/`, no harness-module
import, no model build, no solve). `_assert_structurally_equal`'s body
was copied verbatim into an isolated scratch file (outside the repo,
`numpy` the only import) and exercised directly — NOT via importing
`scripts/27_run_hydro_minflow_sensitivity.py`, which would have pulled
in `src/scenarios.py` at import time.

Toy scenario dicts containing a numpy array, a nested dict (itself
containing a further-nested dict and a list), a top-level list
(containing a dict, matching `conversion_tiers`'s real shape), a bool, a
str, and a `None`, were compared:

1. **Identical dicts, every key** — `NO RAISE` for all seven keys,
   including the ndarray key (the exact key that crashed the original
   guard).
2. **Only one key mutated** (mirroring the harness's real use: base vs.
   a deepcopy with one key changed) — every OTHER key still compares
   equal, `NO RAISE`, confirming the guard would pass a genuine
   `hydro_min_flow_fraction`-only mutation.
3. **One deliberate mismatch per type/location** (ndarray length,
   ndarray values, nested-dict value, nested-dict missing key,
   nested-dict extra key, list element, list length, dict-inside-list
   value, float, bool, str, `None`-vs-scalar) — **every one raised
   `AssertionError` naming the exact key/path**, e.g.
   `a_list[2].x: values differ: 1 != 999` for a mismatch two levels deep
   inside a list-of-dicts.
4. **An unrecognised type** (`set`) — raised `TypeError` naming the path
   and both types, confirming rule 6's "fail loudly, never fall through"
   requirement.
5. **Reproduced the original crash mechanism directly**: `if
   numpy_array != numpy_array:` on two IDENTICAL arrays raises
   `ValueError: The truth value of an array with more than one element
   is ambiguous. Use a.any() or a.all()` — confirming the defect's exact
   cause independent of the harness — while
   `_assert_structurally_equal` on the same identical pair returns with
   `NO RAISE`.

Full output (all thirty checks passed as expected — every "must raise"
case raised, every "must not raise" case did not):

```
=== 1. identical dicts, every key ===
  key='years': NO RAISE
  key='nested': NO RAISE
  key='a_list': NO RAISE
  key='scalar': NO RAISE
  key='flag': NO RAISE
  key='label': NO RAISE
  key='maybe_none': NO RAISE

=== 2. only 'scalar' mutated -- all other keys must pass ===
  key='years': NO RAISE
  key='nested': NO RAISE
  key='a_list': NO RAISE
  key='flag': NO RAISE
  key='label': NO RAISE
  key='maybe_none': NO RAISE

=== 3. deliberate mismatches, one per type -- each must raise ===
  ndarray length differs: AssertionError: years: ndarray values differ
  ndarray values differ: AssertionError: years: ndarray values differ
  nested dict value differs: AssertionError: nested.dry: values differ: 1.1 != 999.0
  nested dict missing a key: AssertionError: nested: dict key sets differ: {'wet'}
  nested dict has an extra key: AssertionError: nested: dict key sets differ: {'extra_key'}
  list element differs: AssertionError: a_list[0]: values differ: 1.0 != 999.0
  list length differs: AssertionError: a_list: list lengths differ: 3 vs 4
  dict-inside-list value differs: AssertionError: a_list[2].x: values differ: 1 != 999
  scalar float differs: AssertionError: scalar: values differ: 0.5 != 0.51
  bool differs: AssertionError: flag: values differ: True != False
  str differs: AssertionError: label: values differ: 'hydro_flat' != 'hydro_ndc'
  None vs scalar differs: AssertionError: maybe_none: values differ: None != 0.0

=== 4. unrecognised type -- must raise TypeError ===
  set() (unhandled type): TypeError: unknown_type_key: unrecognised type(s) set / set ...

=== 5. original crash reproduction (bare != would raise ValueError) ===
  bare `!=` on identical arrays: NO RAISE (returns an array, not used as a bool here)
  `if base != same:` on identical arrays raised as expected: ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
  new helper on the SAME pair: NO RAISE
```

(Note on the "ndarray length differs" row: `np.array_equal` correctly
handles differing lengths itself — returns `False` rather than raising
— so both the length-mismatch and values-mismatch cases surface through
the same `"ndarray values differ"` message; this is correct behaviour,
not a gap, since `np.array_equal` is defined to return `False` for
differing shapes.)

### Closing list — encountered, not asked about

- `reserve_margin` is stored under the scenario-dict key `"reserve_margin"` (not `"reserve_margin_resolved"`) even though the local variable computing it is named `reserve_margin_resolved` — a plain rename-in-place, not a bug, just worth flagging since the key/variable names diverge.
- `capital_envelope_scenarios()`'s non-`"unconstrained"` arms return `int`, not `float` (`int(0.50 * B_star)` etc.) — irrelevant to this fix (`==` compares `int`/`float` correctly) and irrelevant to this harness (`CENTRAL_ARGS` always uses `"unconstrained"` → `None`), noted only because Step 1 asked for `int` specifically as a category.
- `_solar_cumulative_cap()` and `gas_new_capacity_cap_mw`'s `None`-vs-`list` duality (arm-dependent) means a DIFFERENT `CENTRAL_ARGS` choice (which this task does not permit changing) would make those two keys `list`-typed instead of `None` — the fix handles both cases already (list and `None` both have dedicated branches), so this is not a gap, only noted for completeness of the inventory's reasoning.
