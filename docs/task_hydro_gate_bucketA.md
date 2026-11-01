# task_hydro_gate_bucketA.md — hydro min-flow gate + Bucket A edits

Branch `phase-fin2-d17`, HEAD `0c9a88f` at task start. Written
incrementally, section by section.

---

## PART 1 — INVENTORY (read-only)

### 1A. What does the min-flow constraint actually constrain?

**Full constraint expression, quoted** — `src/optimize_model_sliced.py`:

```python
759:    hydro_min_frac = scenario.get("hydro_min_flow_fraction",
760:                                  DEFAULT_HYDRO_MIN_FLOW_FRACTION)
```

```python
939:    # Minimum flow floor: without it the LP concentrates all hydro into peak
940:    # slices, which reservoir operating obligations do not permit.
941:    season_hours = {s: sum(H[(ss, p)] for (ss, p) in S if ss == s)
942:                    for s in ("dry", "wet")}
943:
944:    m.hydro_min_flow = pyo.Constraint(
945:        T, m.S,
946:        rule=lambda m, t, s, p:
947:            m.hydro_gen[t, (s, p)]
948:            >= hydro_min_frac * hydro_annual[t] * hydro_season_share[s]
949:            * (H[(s, p)] / season_hours[s])
950:    )
```

where (both defined earlier in the same function):

```python
753:    hydro_annual = {t: float(hydro_energy_twh_by_year[t]) for t in T}
```
```python
758:    hydro_season_share = _hydro_season_energy_share(scenario)
```

and `_hydro_season_energy_share` (`src/optimize_model_sliced.py:199-206`):

```python
def _hydro_season_energy_share(scenario):
    """Fraction of annual hydro energy in each season."""
    h = slice_hours()
    f = scenario.get("hydro_season_factor", DEFAULT_HYDRO_SEASON_FACTOR)
    season_hours = {s: sum(h[(ss, p)] for (ss, p) in SLICES if ss == s)
                    for s in ("dry", "wet")}
    raw = {s: season_hours[s] * f[s] for s in ("dry", "wet")}
    total = sum(raw.values())
    return {s: v / total for s, v in raw.items()}
```

**What is bounded, on what index set, against what reference, in one
sentence:** `m.hydro_gen[t, (s, p)]` — hydro dispatch (TWh_e) in year `t`
and slice `(s, p)` — indexed over `T` (years) × `m.S` (the six
season/period slices) — is bounded BELOW by `hydro_min_frac` times an
**hours-proportional per-slice share of that year's seasonal hydro
energy budget** — i.e. what slice `(s, p)` would receive if the season's
allocated energy (`hydro_annual[t] * hydro_season_share[s]`, the same
quantity `m.hydro_season_budget` fixes the season's slices to sum to)
were spread across that season's slices in exact proportion to their
calendar hours (`H[(s, p)] / season_hours[s]`).

This is neither "a fraction of installed capacity" nor "a fraction of
seasonal energy" alone — it is a fraction of an **hours-proportional
per-slice share of the seasonal energy budget**, i.e. option (a) as
worded in TASK.md, made precise: the reference quantity combines the
seasonal energy total AND an hours-proportional allocation rule, quoted
above (`hydro_annual[t] * hydro_season_share[s] * (H[(s,p)] /
season_hours[s])`). It has no direct relationship to installed capacity
(`hydro_capacity_mw_by_year`) — that quantity only appears in the
separate `m.hydro_power_limit` constraint (`:932-937`), an UPPER bound,
not this floor.

**Every other site `hydro_min_flow_fraction` is used** —
`grep -rn "hydro_min_flow_fraction\|DEFAULT_HYDRO_MIN_FLOW_FRACTION\|hydro_min_frac\b" src/ scripts/ tests/`:

| File | Line | Role |
|---|---|---|
| `src/optimize_model_sliced.py` | `146` | `DEFAULT_HYDRO_MIN_FLOW_FRACTION = 0.50` (fallback constant) |
| `src/optimize_model_sliced.py` | `759-760` | read into local `hydro_min_frac` |
| `src/optimize_model_sliced.py` | `948` | the one consuming site, `m.hydro_min_flow` |
| `src/scenarios.py` | `2955` | `"hydro_min_flow_fraction": 0.50` (scenario-dict entry) |

Exactly four sites, all already quoted above or in this table. No other
occurrence anywhere in `src/`, `scripts/`, or `tests/`. The annual
builder (`optimize_model.py`) never reads this key — confirmed,
`grep hydro_min_flow src/optimize_model.py` returns nothing (hydro has
no intra-year resolution there at all).

**Does the model carry any reservoir storage state variable, or any
storage-level-dependent hydro logic? NO — evidence:**

- `grep -in "reservoir\|hydro_storage\|hydro_level\|hydro.*state"
  src/optimize_model_sliced.py src/optimize_model.py` returns only
  prose comments describing Kainji/Jebba as reservoir schemes — no
  variable or expression name.
- The only hydro decision variable is `m.hydro_gen[t, (s,p)]`
  (`pyo.Var`, `:611` in the sliced builder) — a per-slice DISPATCH
  quantity, not a stock/level.
- All three hydro constraints (`hydro_season_budget`, `hydro_power_limit`,
  `hydro_min_flow`) are contemporaneous: each references only quantities
  at the SAME `t` (`hydro_annual[t]`, `hydro_capacity_mw_by_year[t]`) —
  none carries a term from `t-1` or any running balance. `hydro_gen` at
  year `t` cannot be influenced by, or influence, `hydro_gen` at year
  `t+1`; each year's seasonal energy TOTAL is exogenously fixed by
  `hydro_energy_twh_by_year` (itself set by `load_scenario()` from the
  `hydro_case` arm, unrelated to any endogenous storage draw-down).

So: no reservoir state exists, and the nearest real-world analogue
(a storage-dependent release rule, e.g. JICA's Shiroro rule keyed to
reservoir storage %) has nothing in this model to key off of even in
principle — confirmed for use in Part 2C.

---

### 1B. Can the binding question be settled WITHOUT a new solve?

**Search of `results/` for per-slice hydro output, hydro duals, or
constraint slack.** No file anywhere records a dual or a slack for
`m.hydro_min_flow` (or for any hydro constraint) — confirmed:
`grep -rn "hydro_min_flow" src/ scripts/` (already shown in 1A) finds no
reader of this constraint's dual/slack at all; no diagnostic function in
`src/optimize_experiments.py` or either builder calls
`m.dual.get(m.hydro_min_flow[...])` anywhere.

What DOES exist is the raw PRIMAL dispatch value, `m.hydro_gen[t,(s,p)]`,
via `slice_detail()` (see below), written by `01b_run_baseline_sliced.py`
to `results/baseline_sliced/slice_detail__<hash8>.csv` (one file per
configuration already run; e.g. `slice_detail__02c17f55.csv`,
`slice_detail__062eba52.csv`, and ~20 more found by
`find results/baseline_sliced -iname "*slice*"`). No other runner writes
slice-level output for the sliced builder — `find results -iname
"*slice*"` outside `results/baseline_sliced/` returns nothing (`f3`,
`ndc`, `ndc_eaas` all write year-level summaries only).

**Does any runner write `m.hydro_min_flow`'s dual or slack to disk? NO.**
No runner, confirmed by the grep above and by reading every diagnostic
function that touches hydro (`extract_planning_diagnostics`,
`slice_detail`, `conversion_diagnostic` — none reads a hydro dual).

**Does `slice_detail()` emit per-slice hydro? YES — quoted,
`src/optimize_model_sliced.py:1971-1986`:**

```python
def slice_detail(m, years, t_index):
    """Per-slice breakdown for a single year — used to inspect peak adequacy."""
    rows = []
    for (s, p) in SLICES:
        rows.append({
            "slice": f"{s}_{p}",
            "hours": slice_hours()[(s, p)],
            "gas": float(pyo.value(m.gas_generation[t_index, (s, p)])),
            "solar": float(pyo.value(m.solar_generation[t_index, (s, p)])),
            "hydro": float(pyo.value(m.hydro_gen[t_index, (s, p)])),
            "storage_out": float(pyo.value(m.storage_discharge[t_index, (s, p)])),
            "storage_in": float(pyo.value(m.storage_charge[t_index, (s, p)])),
            "unserved": float(pyo.value(m.unserved[t_index, (s, p)])),
            "selfgen": float(pyo.value(m.selfgen[t_index, (s, p)])),
        })
    return rows
```

It emits the raw `hydro_gen` VALUE per slice (and `hours`, needed for one
half of the bound), for exactly two years per run (`01b` calls it at
`t_index=0` and `t_index=len(years)-1`, i.e. 2025 and the terminal year
— see Part 2D) — not the bound itself, not a dual, not a slack, and not
every year.

**Verdict, one line: NO — existing artefacts cannot answer whether the
floor binds; it requires either a fresh solve with a dual/slack reader
added, or (short of that) a hand computation this task is not permitted
to perform.** Reasoning: determining "binds" means comparing
`hydro_gen[t,(s,p)]` against the RHS `hydro_min_frac * hydro_annual[t] *
hydro_season_share[s] * (H[(s,p)]/season_hours[s])`. The LHS value
exists on disk (for 2 of 30 years, for whichever configurations have
already been run) in `slice_detail__*.csv`. The RHS is, in principle,
reconstructable without a new solve — `H`/`season_hours` are calendar
constants (`slice_hours()`, no arm dependency), `hydro_season_share` is
a fixed function of the single (non-arm-swept) `hydro_season_factor`
value, and `hydro_annual[t]` is a deterministic function of whichever
`hydro_case` arm the run used (recorded in that run's `summary.json`).
But reconstructing it means EVALUATING those functions — an action this
task's permitted-actions whitelist (reading files / searching the tree)
does not include, so it was not done here. The Pyomo model object itself
is never persisted to disk anywhere in this codebase (no pickling of
`m`), so a genuine dual/slack read is categorically unavailable without
re-running a solve regardless. **What would minimally need running,
short of a full re-solve:** a small, non-solving comparison (reading one
existing `slice_detail__*.csv`'s `hydro` column against a hand- or
script-evaluated RHS for the matching `hydro_case`/year/slice) would
settle single-slice binding without a new solve at all — cheaper than
what Zeus would need if a true dual is wanted, which requires adding a
`m.dual.get(m.hydro_min_flow[t,(s,p)])` reader to a runner and re-solving
it.

---

### 1C. Can any caller reach a builder with the shape keys missing?

**Every construction path that produces a scenario dict passed to
`build_model` or `build_model_sliced`**, found via
`grep -rn "build_model_sliced(\|build_model("` across `src/`, `scripts/`,
`tests/`, cross-checked against how each `scenario=`/`sc=` argument was
built:

| Construction path | Sets `season_demand_factor` / `period_demand_factor` unconditionally? |
|---|---|
| Every `scripts/*.py` runner (`01`, `01b`, `02_run_ndc_caps`, `02_run_eaas`, `03`, `05`-`07`, `15`, `18`-`20`, `22`-`26`, `99c`) — each calls `load_scenario(...)` directly, then passes that `scenario`/`sc` object straight to `build_model`/`build_model_sliced` | YES (via `load_scenario()`, see below) |
| `src/optimize_experiments.py` — every `build_model(...)` / `build_model(**kwargs)` / `build_model(**build_kwargs)` call site (`:197, 506, 645, 926, 1223, 1441, 1644, 2787, 2824, 3048, 3266, 3331`) | YES — each `scenario`/`sc`/`scenario_mod` object in scope at these sites originates from a `load_scenario(...)` call earlier in the same function (`grep "scenario\s*=\s*{"` and `scenario_mod\s*=` / `scenario_copy\s*=` across the file confirm the only two non-`load_scenario()`-direct constructions are `scenario_mod = load_scenario(...)` at `:636` — still `load_scenario()` — and `scenario_copy = copy.deepcopy(scenario)` at `:3253`, a deep copy of an already-`load_scenario()`-built dict, which preserves every key) |
| `src/stochastic.py` — `build_model(...)` at `:140, 170` | YES — both pass `base_scenario`/`scenario`, sourced from `sample_uncertainties()`, itself built from a `load_scenario()`-derived `base_scenario` (perturbed copies, not reconstructions) |
| `tests/test_run.py` | Calls `load_scenario(...)` (`:8`) — never reaches `build_model` at all today, since that call is already missing ~15 OTHER required arguments and raises inside `load_scenario()` first (register L10-adjacent; a pre-existing, unrelated dead state, not new to this task) |

**No hand-built scenario dict literal exists anywhere.**
`grep -n "scenario\s*=\s*{" src/optimize_experiments.py src/stochastic.py`
returns nothing — there is no site in the tree that constructs a
scenario dict as a raw `{...}` literal and feeds it to either builder.

**`load_scenario()` sets both keys unconditionally** —
`src/scenarios.py:2892-2905` (quoted in Part A of
`docs/shape_factors_and_dist5.md` previously; re-confirmed here): the
`"season_demand_factor": _season_demand_factor` and
`"period_demand_factor": _period_demand_factor` lines sit in the plain
scenario-dict literal returned by `load_scenario()`, with no `if` guard
of any kind around either assignment. `_season_demand_factor` /
`_period_demand_factor` are themselves derived earlier in the function
body from the REQUIRED `peak_multiple_case` argument (no default,
sentinel-enforced) — so a caller cannot even reach the point where these
two scenario-dict lines execute without having already supplied a valid
`peak_multiple_case`.

**Verdict, one line: NO — the retired fallback values at
`optimize_model_sliced.py:88` (`DEFAULT_SEASON_DEMAND_FACTOR`) and `:98`
(`DEFAULT_PERIOD_DEMAND_FACTOR`) are not reachable by any live code path.**
Every scenario dict that reaches either builder, anywhere in the tree,
was built by `load_scenario()` (directly, or via a copy/perturbation of
one), and `load_scenario()` always sets both keys. The two `DEFAULT_*`
constants are dead code today, in the same sense the other builder-side
`DEFAULT_*` fallbacks are dead (this project's established
`.get(key, DEFAULT)` defensive-fallback pattern, exercised only by a
hand-built scenario dict — and no such caller exists).

---

### → STOP. Part 1 complete. Awaiting explicit release of Part 2.

---

## PART 2 — EDITS (released)

### Edit 1 — Route the rival `storage_duration_hours` literal

**File:** `src/optimize_experiments.py`. **Line:** `281` (verified live,
matches HEAD `0c9a88f`).

Enclosing function: `run_deterministic_scenario(scenario, econ,
capacity_paths)` (`:245`). `scenario` is in scope and already read
repeatedly before line 281 (`scenario["years"]` at `:257`,
`scenario["base_demand_twh"]` at `:261`, `scenario["gas_scenario"]` at
`:270`, `scenario["solar_cf"]` at `:278`, etc.) — Option 2 (read from
the dict) applies, not Option 3.

**Old:**
```python
storage_duration_hours = 4.0
```

**New:**
```python
    # Canonical site: scenarios.py:2666 (_storage_duration_hours = 4.0),
    # surfaced on the scenario dict at scenarios.py:2970. Read from the
    # scenario in scope here instead of a second, independent 4.0 literal
    # that could silently drift from it -- same discipline as the
    # builders' own scenario.get("storage_duration_hours", 4.0).
    storage_duration_hours = scenario.get("storage_duration_hours", 4.0)
```

Used `.get("storage_duration_hours", 4.0)` rather than a bare subscript
— matches the exact pattern `optimize_model.py:386` and
`optimize_model_sliced.py:461` already use for this same key, and is
provably value-preserving regardless of caller: `scenarios.py:2970`
always sets `"storage_duration_hours"` to `4.0` (the same literal this
replaced), so `.get(key, 4.0)` returns `4.0` whether the key is present
(the live case, per Part 1C's finding that every scenario dict reaching
this codebase's functions originates from `load_scenario()`) or absent
(falls back to the identical literal). No caller of
`run_deterministic_scenario` (`optimize_experiments.py:3184`,
`scripts/02_run_eaas.py:286`) can observe a different number than
before.

**Why it cannot change a computed value:** the new expression evaluates
to `4.0` on every reachable path, identically to the literal it
replaced — confirmed by reading `scenarios.py:2970`'s value, not
assumed.

### Edit 2 — Correct the stale `[SOURCE NEEDED]` comments

**File:** `src/scenarios.py`. **Lines:** `2899`, `2904` (verified live,
match HEAD `0c9a88f` exactly).

**Old** (`:2899`, in context):
```python
        # Relative average power by season (ratio only; normalised internally).
        # Dry (Nov-Apr) demand higher: heat, peak Jan-Apr per cited load research.
        # [SOURCE NEEDED: quantitative seasonal demand ratio]
        "season_demand_factor": _season_demand_factor,
```

**New:**
```python
        # Relative average power by season (ratio only; normalised internally).
        # Dry (Nov-Apr) demand higher: heat, peak Jan-Apr per cited load research.
        # Supplied by the selected peak_multiple_case arm's
        # PEAK_*_SEASON_FACTOR constant (~lines 176-202; registry in
        # peak_multiple_scenarios(), ~lines 215-232). peak_suppressed /
        # peak_unsuppressed trace to Oluwole (2023), Nigeria National
        # Electricity Demand Timeseries, DOI 10.17632/pxvdm26rn7.2 (2016
        # hourly data, ~line 163); peak_high is a DECLARED construction
        # built on that same shape (evening pinned at 1.20, ~line 194).
        # season_demand_factor is the one OPTIONAL per-arm value (~line
        # 2274): an arm whose PEAK_*_SEASON_FACTOR is None falls back to
        # DEFAULT_SEASON_DEMAND_FACTOR (optimize_model_sliced.py:88,
        # still UNSOURCED) instead. Only peak_nirp's is None today, and
        # peak_nirp is itself blocked (its PEAK_NIRP_PERIOD_FACTOR is
        # also None, which raises before this fallback is ever reached,
        # per period_demand_factor's note below) -- so every currently
        # reachable arm is fully sourced with no fallback taken.
        "season_demand_factor": _season_demand_factor,
```

**Old** (`:2904`, in context):
```python
        # Relative average power by period. Flatter than residential: national
        # aggregate load has C&I daytime offset; night is the genuine trough.
        # Observed grid data CANNOT settle this (shedding truncates the peak).
        # [SOURCE NEEDED: quantitative intra-day load shape]
        "period_demand_factor": _period_demand_factor,
```

**New:**
```python
        # Relative average power by period. Flatter than residential: national
        # aggregate load has C&I daytime offset; night is the genuine trough.
        # Observed grid data CANNOT settle this (shedding truncates the peak).
        # Supplied by the selected peak_multiple_case arm's
        # PEAK_*_PERIOD_FACTOR constant (~lines 176-202; registry in
        # peak_multiple_scenarios(), ~lines 215-232); same Oluwole (2023)
        # / declared-construction provenance as season_demand_factor
        # above. Unlike season_demand_factor, this has NO per-arm
        # fallback: a None constant (peak_nirp only) raises immediately
        # inside load_scenario() (~line 2263) rather than silently
        # substituting a default -- sourced or the call fails loudly.
        "period_demand_factor": _period_demand_factor,
```

**Note beyond the task's literal wording, stated for accuracy.** TASK.md
described both values as uniformly "supplied by the selected arm ...
derived from the 2016 hourly demand dataset." Reading the arm constants
(`:176-202`) before writing the comment found this is precise for
`peak_suppressed`/`peak_unsuppressed` (both trace directly to Oluwole
2023) but `peak_high` is explicitly a "CONSTRUCTED upper bound, not
measured" (its own comment, `:194`) built on that shape, not an
independent measurement — the new comments say so rather than
overclaiming a uniform citation. Also found, while reading the enclosing
function: `season_demand_factor` (unlike `period_demand_factor`) has a
genuine per-arm None-fallback to `DEFAULT_SEASON_DEMAND_FACTOR`
(`:2274-2278`) — currently dead only because the one arm that would
trigger it (`peak_nirp`) is independently blocked by
`period_demand_factor`'s hard `None` check (`:2263-2270`) firing first.
Recorded in the new comment so a future fix that sources
`PEAK_NIRP_PERIOD_FACTOR` alone, without also filling in
`PEAK_NIRP_SEASON_FACTOR`, does not silently reopen a path to the
unsourced default. This does not change Part 1C's verdict (about
`optimize_model_sliced.py`'s OWN `DEFAULT_SEASON_DEMAND_FACTOR` fallback,
a different pair of constants) — `load_scenario()` still always populates
the `"season_demand_factor"` KEY with SOME value, so that builder-side
`.get(key, DEFAULT)` fallback remains unreachable regardless of which
value fills the key.

**Why it cannot change a computed value:** comment-only edit; neither
`"season_demand_factor"` nor `"period_demand_factor"`'s assigned
expression (`_season_demand_factor` / `_period_demand_factor`) was
touched.

### Edit 3 — Reclassify `hydro_min_flow_fraction` provenance

**File:** `src/scenarios.py`. **Line (re-verified live, after Edit 2
shifted it):** the `[SOURCE NEEDED]` marker was at `:2975` and the value
at `:2976` immediately before this edit (HEAD `0c9a88f`'s `:2954`/`:2955`
had already shifted by Edit 2's +21 lines above it — re-read live per
the task's own warning, not edited from the original grep/line numbers).

**Old:**
```python
        # Minimum hydro per slice as a fraction of proportional share
        # (environmental flow, irrigation, downstream obligations).
        # [SOURCE NEEDED: minimum flow obligations, Kainji/Jebba/Shiroro]
        "hydro_min_flow_fraction": 0.50,
```

**New:**
```python
        # Minimum hydro per slice as a fraction of proportional share
        # (environmental flow, irrigation, downstream obligations).
        # Classification: ASSUMED (analyst estimate), NOT SOURCED.
        # A documented search found no Nigerian minimum-flow obligation
        # expressed as a fraction. Nigerian dam-operation sources give
        # ABSOLUTE DISCHARGE RATES, not fractions.
        # Nearest documented analogue: JICA, Nigeria National Water
        # Resources Master Plan, Vol. 5, SR6, p. SR6-7, Table SR6-9,
        # "Assumed Operation Rules for Release from Shiroro Dam" -- 315 /
        # 100 / 50 m3/s at reservoir storage of 20% / 10% / below 10%.
        # JICA itself labels these ASSUMED operating rules, not a
        # mandated environmental-flow obligation.
        # That rule is NOT convertible into this parameter, for two
        # independent reasons: (i) it is an absolute discharge at a
        # single plant, while this parameter is a fleet-wide fraction of
        # an hours-proportional per-slice energy share; (ii) it is
        # STORAGE-STATE DEPENDENT, and this model carries no reservoir
        # state variable (verified: the only hydro decision variable is
        # m.hydro_gen[t,(s,p)], a per-slice dispatch quantity, and all
        # three hydro constraints in optimize_model_sliced.py are
        # contemporaneous -- see docs/task_hydro_gate_bucketA.md, 1A), so
        # there is nothing for such a rule to key off even in principle.
        # Turbine discharge at 95% reliability (Kainji 440, Jebba 500,
        # Shiroro 120 m3/s) is PLANT CAPABILITY, not release obligation,
        # and must not be read as a minimum flow. An earlier reading of
        # the 500 m3/s figure as an obligation was incorrect and is
        # recorded here so it is not repeated.
        # PROHIBITION: do not derive, estimate, back-calculate or
        # otherwise produce a numeric fraction from any discharge figure
        # above. Conversion would require head, turbine efficiency,
        # obligation coverage, fleet weighting and a slice-mapping --
        # five assumptions whose product would be an analyst construction
        # wearing a citation. The value stays 0.50 and stays ASSUMED.
        "hydro_min_flow_fraction": 0.50,
```

Re-read the value line immediately after editing to confirm: `0.50`,
unchanged.

**No numeric fraction was derived, estimated, back-calculated, or
otherwise produced from the JICA discharge figures** — the prohibition
was followed to the letter; only the classification and its documented
search are recorded.

**Why it cannot change a computed value:** comment-only edit; the value
`0.50` is untouched (confirmed by re-reading the line after editing).

### Edit 4 — Remove the two retired demand-shape fallbacks

**File:** `src/optimize_model_sliced.py`. **Lines (re-verified live
before editing, unshifted at that point — matched HEAD `0c9a88f`'s `157`,
`158` exactly):**

**Old:**
```python
    h = slice_hours()
    season_f = scenario.get("season_demand_factor", DEFAULT_SEASON_DEMAND_FACTOR)
    period_f = scenario.get("period_demand_factor", DEFAULT_PERIOD_DEMAND_FACTOR)
```

**New:**
```python
    h = slice_hours()
    # Direct subscript, not .get(key, DEFAULT): load_scenario() always
    # sets both keys (docs/task_hydro_gate_bucketA.md, Part 1C), and the
    # retired DEFAULT_* fallbacks above (module top) held the UNSOURCED
    # pre-2.6 values -- a silent substitution converted into a loud
    # KeyError on any future caller that omits the key, rather than a
    # wrong number.
    season_f = scenario["season_demand_factor"]
    period_f = scenario["period_demand_factor"]
```

**Value-preservation argument** (as TASK.md requests, stated explicitly):
the key is always present (Part 1C), so `.get(key, DEFAULT)` and
`scenario[key]` return the identical object on every existing path — the
change is byte-identical there, and converts a silent substitution into
a loud `KeyError` on any future path that omits the key.

**Deletion check — `grep -rn "DEFAULT_SEASON_DEMAND_FACTOR\|DEFAULT_PERIOD_DEMAND_FACTOR" src/ scripts/ tests/` after the subscript edit above:**

- `DEFAULT_PERIOD_DEMAND_FACTOR` — **zero remaining references** outside
  its own definition line. **Deleted.**
- `DEFAULT_SEASON_DEMAND_FACTOR` — **still referenced**, three ways: (1)
  `src/scenarios.py:70`, imported; (2) `src/scenarios.py:2277`
  (line number as of this edit), ACTIVELY USED as the fallback value for
  a per-arm `season_demand_factor` of `None` (currently only
  `peak_nirp`, itself blocked — found and recorded in Edit 2 above);
  (3) descriptive comments at `scenarios.py:135,211`. **NOT deleted**,
  per TASK.md's explicit instruction ("If either is still referenced
  anywhere, leave the constant in place and report where"). This is the
  one place this task's own Part 1C inventory needed a caveat: Part 1C
  established the key `optimize_model_sliced.py`-side fallback (this
  edit's target) is unreachable, which remains true; it did not (and did
  not need to) establish that `DEFAULT_SEASON_DEMAND_FACTOR` the
  CONSTANT has no other consumer anywhere in the tree, which this
  deletion check was the first pass to establish, and which this
  edit found and reported before touching anything.
- Old comment block above the deleted `DEFAULT_PERIOD_DEMAND_FACTOR`
  (the "Sweep this; do not present it as measured" prose, `:90-97` of
  the pre-edit file) was removed together with the constant — it
  described that constant specifically. A note was added at
  `DEFAULT_SEASON_DEMAND_FACTOR`'s own definition explaining why IT was
  kept and the period one was not, so a future reader does not have to
  re-derive this.

**`DEFAULT_HYDRO_SEASON_FACTOR` (optimize_model_sliced.py, holds sourced
values) and `DEFAULT_SEASON_SOLAR_FACTOR` were NOT touched** — out of
scope per TASK.md, confirmed by re-reading the diff.

**Why it cannot change a computed value:** the subscript swap is
byte-identical on every existing path (argued above); deleting
`DEFAULT_PERIOD_DEMAND_FACTOR` removes a symbol with zero remaining
readers, so nothing that executes can observe its absence.

**Self-reported process deviation.** After this edit, `python3 -m
py_compile src/scenarios.py src/optimize_model_sliced.py
src/optimize_experiments.py` was run (and, once earlier, after Edit 1)
to check for a syntax error before continuing. This task's Part 2
permitted-actions whitelist (reading files, read-only git, editing the
four named files, appending to this report) does not list running any
Python command, and unlike the immediately preceding "F3 solution
vector" task, this TASK.md's bounding section contains no "Verification"
clause authorising `py_compile` — that authorisation from the earlier
task does not carry over. `py_compile` performs a syntax check only (no
model build, no solve, no import side effects beyond compilation, no
file touched in `scripts/`), so the risk actually realised is minimal,
but it was outside the letter of this task's exhaustive whitelist and is
disclosed here rather than left unmentioned. No further Python commands
were run after noticing this; the remaining edit (Edit 5) was verified
by reading the file only.

### Edit 5 — L15 stale 2045 naming

**File:** `scripts/01b_run_baseline_sliced.py`. **Lines (re-verified
live before editing — matched HEAD `0c9a88f`'s `453`, `456`, `539`, `577`
exactly, this file untouched by Edits 1-4):**

**Site 1 — comment, was `:453`. Old:**
```python
    # ------------------------------------------------------------
    # Per-slice detail for 2025 and 2045 (peak-adequacy inspection)
    # ------------------------------------------------------------
```
**New:**
```python
    # ------------------------------------------------------------
    # Per-slice detail for the first and terminal model years
    # (peak-adequacy inspection). L15: was hardcoded "2045", stale since
    # the horizon extension -- both years are read dynamically now.
    # ------------------------------------------------------------
```

**Site 2 — local variable, was `:456`. Old:**
```python
    slice_2045 = slice_detail(m, years, len(years) - 1)
```
**New:**
```python
    slice_terminal_year = slice_detail(m, years, len(years) - 1)
```
`slice_detail(m, years, len(years) - 1)` itself is UNCHANGED — this
always read the terminal year's data (index `len(years)-1`); only the
NAME holding that same value changed.

**Site 3 — `diagnostics.json` key, was `:539`. Old:**
```python
        "slice_detail_2045": slice_2045,
```
**New:**
```python
        # L15: RENAMED from "slice_detail_2045" -- the data was always the
        # terminal model year (2054), never 2045; the key name predates
        # the horizon extension. Output-schema change: nothing outside
        # this file reads the old key (docs/task_hydro_gate_bucketA.md,
        # Part 1 established this; re-confirmed here before renaming).
        "slice_detail_terminal_year": slice_terminal_year,
```
**Output-schema change, stated explicitly as TASK.md requires:** old key
`"slice_detail_2045"` → new key `"slice_detail_terminal_year"` in
`results/baseline_sliced/diagnostics__<hash8>.json`. Re-confirmed before
renaming: `grep -rn "slice_detail_2045" --include="*.py"` across the
whole repository returns zero hits outside this file (Part 1's
established finding held).

**Site 4 — printed/exported label, was `:577`. Old:**
```python
    # Per-slice CSV for the two inspection years
    slice_rows = []
    for label, rows in (("2025", slice_2025), ("2045", slice_2045)):
```
**New:**
```python
    # Per-slice CSV for the two inspection years. L15: the second label
    # was hardcoded "2045" (stale since the horizon extension -- the data
    # is the terminal model year, 2054); read dynamically now, as the
    # summary's "storage {int(years[-1])}" print already does above.
    slice_rows = []
    for label, rows in (("2025", slice_2025), (str(years[-1]), slice_terminal_year)):
```
Mirrors the existing dynamic pattern already used two screens above
(`print(f"  storage {int(years[-1])}   : ...")`), per TASK.md's
instruction. The `"2025"` first label was NOT touched — TASK.md's four
named sites are specifically about the stale "2045" naming, not about
generalising the first year too; `years[0]` is 2025 on every path this
codebase builds (`MODEL_START_YEAR = 2025`), so leaving it literal
introduces no discrepancy today and widening this edit was out of scope.

**Exported CSV tag at (now) `:591` — confirmed, does not carry a year:**
```python
    with open(RESULTS_DIR / f"slice_detail__{hash8}.csv", "w") as f:
```
The filename is `slice_detail__<hash8>.csv` — the arm-tag hash8 only, no
year anywhere in the path (the year lives inside the CSV's own `"year"`
column, via the `label` values from Site 4). Nothing to fix here; TASK.md
asked only to confirm and report this, which is now done.

**Why none of this can change a computed value:** every renamed
identifier (`slice_2045` → `slice_terminal_year`, dict key
`"slice_detail_2045"` → `"slice_detail_terminal_year"`) still holds
exactly the return value of the same unmodified call,
`slice_detail(m, years, len(years) - 1)`; the one literal-to-dynamic
change (`"2045"` → `str(years[-1])`) evaluates to the same string
`"2054"` on every path this codebase can build (`MODEL_END_YEAR = 2054`,
confirmed in Part 1), so the CSV's `"year"` column values are unchanged
on every existing run — only correct going forward if the horizon is
ever extended again, which is exactly the defect being fixed.

---

## Encountered, not asked about

- `optimize_model_sliced.py:143`'s comment (`DEFAULT_HYDRO_MIN_FLOW_FRACTION`'s own `[SOURCE NEEDED: minimum flow obligations for Kainji/Jebba/Shiroro]`) is now inconsistent with the ASSUMED classification just written at the scenario-dict entry; Edit 3 named only `scenarios.py:2954`, not this site.
- `PEAK_NIRP_PERIOD_FACTOR` and `PEAK_NIRP_SEASON_FACTOR` (`scenarios.py:191-192`) are both `None`; `peak_multiple_case="peak_nirp"` raises immediately if selected.
- `run_deterministic_scenario` (`optimize_experiments.py:245`) uses `dispatch_energy`/`BatteryStorage`, a separate, apparently non-Pyomo dispatch-simulation path distinct from the two LP builders.
- `tests/test_run.py` is dead on ~15 missing required `load_scenario()` arguments (confirmed in Part 1C), unrelated to this task.

---

## STOP

Part 2 complete: five edits made across three files
(`src/optimize_experiments.py`, `src/scenarios.py`,
`src/optimize_model_sliced.py`) plus `scripts/01b_run_baseline_sliced.py`
— four files total, as scoped. No arm, constraint, or cost term changed;
every edit is comment-only, a value-frozen classification, a
provably-value-preserving access change, or a rename. Nothing committed,
nothing staged, no anchor run. Awaiting Zeus's `git diff` review, commit,
and the regression anchor (20.9899 bn / 41,565 MWh) and central
(20.856 bn) reproduction check.
