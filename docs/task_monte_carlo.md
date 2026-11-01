# task_monte_carlo — parameter uncertainty and variance decomposition

Branch `phase-fin2-d17`. Model frozen at `2a9226b`; this task makes no
edits of any kind in Part 1 (Part 2 is release-gated and, when
released, may only ADD a new file, `scripts/29_run_parameter_
uncertainty.py` — no existing file may be touched). Nothing was run:
no script, no solver, no model build. The handful of read-only Python
calls used to verify claims below (`load_om_cost_by_year`,
`gas_availability_scenarios`, `conversion_scenarios`, `scipy.stats.qmc/
sobol_indices` on toy data) are pure, side-effect-free data loads and
library introspection — same disclosed-judgment-call category as prior
tasks in this sequence, listed in full where used. No `pip install` was
run or considered.

---

## Corrections, checked before treating anything as design input

Two of this task's stated facts hold exactly; three do not. Checked
before any of Part 1's actual questions, since Part 2's design table
depends on them.

1. **748,000 USD/MW for conversion capex does not exist anywhere in
   this codebase or its documentation.** Exhaustive tree search
   (`grep -rn "748" --include="*.py" --include="*.md" --include="*.csv"
   --include="*.yaml" --include="*.txt" .`, and separately for
   `convert_capex_usd_per_mw`): every hit is either an unrelated number
   (a stale regression anchor `21.4748`, a line-number citation
   `665–748`, coincidental digits inside `13748`/carbon-price CSV rows)
   or **this project's own prior two task reports**
   (`task_conversion_diagnostics.md`, `task_conversion_capex_exposure.md`)
   explicitly stating that this exact figure was fabricated in an
   earlier TASK.md and refuting it. There is no third-party source, no
   data file, no comment anywhere that ever asserted 748,000 USD/MW as
   real. Full detail under **3F**.
2. **scipy installed here is 1.15.2, not 1.16.3.** `python3 -c "import
   scipy; print(scipy.__version__)"` in the active `nigeria-energy`
   environment (the same `python3` any script here would run under —
   confirmed via `which python3` →
   `/home/zeus/miniconda3/envs/nigeria-energy/bin/python3`). Does not
   change the design: `scipy.stats.qmc.Sobol` exists in 1.15.2 too. But
   see **3G** for a design-relevant capability this version has that
   TASK.md's own framing said didn't exist.
3. **SALib is confirmed NOT installed** (`ModuleNotFoundError`) — this
   claim holds exactly.
4. **L10's citation of `17_run_monte_carlo`'s two further defects holds
   exactly** — checked the deferred script's own docstring: (1) LCOE
   numerator includes the VoLL penalty while the denominator omits
   hydro and double-counts storage discharge; (2) the demand-growth
   prior is unsourced (`[SOURCE NEEDED]`, plan 1.6). Both stated,
   verbatim, in the file's own header. No correction needed here.
5. **`scipy.stats.qmc.Sobol` "provides the sequence, not the Saltelli
   cross-sampling, so the script must construct matrices A, B and the D
   hybrid matrices itself"** — this undersells what is actually
   available. `scipy.stats.sobol_indices` (a different, higher-level
   function in the same installed scipy, `1.15.2`) already performs
   the ENTIRE Saltelli 2010 procedure — sampling, A/B/AB-hybrid matrix
   construction, and first-/total-order index computation, plus
   built-in bootstrap confidence intervals — given only a callable and
   a list of per-parameter distributions. This is a design-relevant
   finding, detailed in full under **3G**, flagged there for a choice
   rather than acted on unilaterally.

---

## PART 1 — INVENTORY (3A-3G)

### 3A — is each of parameters 1-4 continuously settable, or arm-gated?

**Parameter 1, gas fleet availability.** Arm-gated at `load_scenario()`
time (`gas_availability_case`, required, no default —
`scenarios.py:2356-2360`), but the RESULT is a plain float sitting in
two places in the returned scenario dict:
```python
_gas_availability_factor = gas_availability_scenarios()[gas_availability_case]   # scenarios.py:2368
...
_gas_capacity_credit = _gas_availability_factor                                  # scenarios.py:2818
capacity_credit = {"gas": _gas_capacity_credit, ...}                             # scenarios.py:2819-2821
...
"gas_availability_case": gas_availability_case,      # scenarios.py:2886
"gas_availability_factor": _gas_availability_factor, # scenarios.py:2887
```
**Yes, continuously settable** — but at TWO dict locations, not one:
`scenario["gas_availability_factor"]` (read directly by both builders
as the annual energy ceiling) AND `scenario["capacity_credit"]["gas"]`
(read by the firm-capacity/peak-adequacy constraint,
`optimize_model_sliced.py:1263,1271,1286`). These are declared
**independently settable** in `gas_availability_scenarios()`'s own
docstring (`scenarios.py:947-953`) but `load_scenario()` only ever
initialises them EQUAL, from the same arm value — there is no existing
runner that sets them to different values. **A script that mutates only
`scenario["gas_availability_factor"]` after `load_scenario()` returns
would silently decouple the energy-ceiling and firm-capacity readings
of availability** — exactly the "shared driver sampled independently"
failure mode Section 1 attributes to the deferred `17_`, reproduced
inside a single parameter if not both keys are set together. **The
Sobol script must set both**:
```python
scenario["gas_availability_factor"] = x
scenario["capacity_credit"]["gas"] = x
```
Endpoints (`gas_availability_scenarios()`, `scenarios.py:970-974`):
`gas_avail_observed = 0.295`, `gas_avail_nameplate = 0.80` —
matches TASK.md's stated band exactly. `gas_avail_central = None`
(raises if selected — genuinely unsourced, per its own docstring).

**Parameter 2, fleet thermal efficiency (`gas_eta`).** A plain,
unconditional scalar, no case/arm mechanism at all:
```python
"gas_eta": 0.287,   # scenarios.py:2836
```
**Continuously settable** by the same post-`load_scenario()` mutate
convention (`scenario["gas_eta"] = x`) — no split-key issue like
parameter 1.

**Parameter 3, solar capacity factor (`solar_cf`).** Also a plain,
unconditional scalar:
```python
"solar_cf": 0.19,   # scenarios.py:2934, fixed-tilt CF, central.
```
**Already has a direct, working precedent in this exact codebase**:
`scripts/20_run_cf1_solar_yield_sensitivity.py:119`:
```python
scenario["solar_cf"] = cf   # "mutate convention, as elsewhere"
```
This is literally the `18_`/`28_` pattern TASK.md refers to, and it is
already named as a convention in that script's own comment — this is
the strongest-precedented parameter of the four.

**Parameter 4, O&M level.** Arm-gated (`om_case`, required, one of
`om_none`/`om_low`/`om_central`/`om_high`) and — unlike parameters 1-3
— **not a scalar**: `load_scenario()` calls `load_om_cost_by_year(
om_case=om_case, ...)` (`scenarios.py:2680`) and stores the resulting
NESTED dict at `scenario["om_cost_by_year"]`
(`{technology: {year: {'fom_usd_per_kw_yr':…, 'vom_usd_per_mwh':…}}}`,
`scenarios.py:3080`). See **3D** for whether and how this can be made
continuous.

### 3B — where a parameter is arm-gated, what does the arm actually set?

Answered above per-parameter. Summary of the underlying numeric keys a
script must set directly, bypassing `load_scenario()`'s own
case-selection argument:

| Parameter | Arm argument | Underlying key(s) to set directly |
|---|---|---|
| 1, gas availability | `gas_availability_case` | `scenario["gas_availability_factor"]` **and** `scenario["capacity_credit"]["gas"]` (both, together) |
| 4, O&M | `om_case` | `scenario["om_cost_by_year"]` (a nested per-technology/per-year dict, not a scalar — see 3D) |

Parameters 2 and 3 have no arm at all to bypass — they are already
plain keys.

### 3C — fleet efficiency: scenario key, econ key, or derived? Does EF_e track it automatically?

**Scenario key**, confirmed above (`scenario["gas_eta"]`, plain
scalar). Not an `econ` key.

**`EF_e = EF_th / eta` derivation, traced to both ends**:
- `econ["EF_TCO2_PER_MWH_TH"] = 0.1823` set once in `io.py:409`
  (thermal-basis factor, HHV, does not vary with η).
- `io.py:406-407,421` states the design intent explicitly: *"EF per
  MWh_e is DERIVED in build_model as EF_th / gas_eta, so efficiency
  and emissions can never drift apart"* / *"EF_e = EF_th / eta is
  derived in build_model so that sweeping eta moves cost and emissions
  together."*
- Confirmed in the actual builder, not just the comment:
  `optimize_model_sliced.py:463` — `eta = scenario.get("gas_eta",
  None)`; `optimize_model_sliced.py:1822` — `ef_e =
  econ["EF_TCO2_PER_MWH_TH"] / eta`. `eta` here is a plain local
  variable read directly from the scenario dict at build time, so
  **setting `scenario["gas_eta"]` before calling `build_model_sliced()`
  automatically and correctly re-derives `ef_e` on every draw** — no
  separate emissions-factor update is needed or possible to forget,
  since there is nowhere else the electrical emission factor could come
  from.

**[TASK 77: the 800-860 MMscf/d range has no source (register L130), and 0.277-0.298 is not derived from it (TASK 76 A.2).]**
**Provenance of the proposed band** (TASK.md: "from the 800-860 MMscf/d
volume range" — not independently re-derived as a design decision here,
Part 1 only traces what exists): `docs/merged_correction_plan_v2.md:68-70,98`
states `η = 0.287` was derived as `observed TWh_e ÷ observed TWh_th`
(NERC 2024, 25.62 TWh_e ÷ a gas volume corresponding to ≈805 MMscf/d,
"at the low end of the reported [800-860 MMscf/d] range") and that an
EARLIER phase of this project proposed sweeping **{0.277, 0.287,
0.298}** for "the unresolved billed-vs-combusted measurement boundary."
**That three-point sweep was never implemented** — `grep -rn "0\.277\|
0\.298" src/ scripts/` returns nothing. It exists only as a documented,
unimplemented design intent from an earlier phase; usable as a
starting point for a continuous band, not as an already-coded artifact
to import.

### 3D — O&M: can rates be scaled continuously, or is it locked to three cases?

**Not locked, but not scalar either — the mechanism is a client-side
blend of two already-public, already-importable function calls, not a
`src/` edit.**

`src/io.py:159-220`, `load_om_cost_by_year(om_case, start_year=2025,
end_year=2050)`, is a MODULE-LEVEL function (not private, no leading
underscore), already imported directly by `scenarios.py:75`
(`from src.io import load_om_cost_by_year`) — a script can import it
exactly the same way. It returns
`{technology: {year: {'fom_usd_per_kw_yr': float, 'vom_usd_per_mwh':
float}}}` for one `om_case` string at a time; `om_case` itself is
validated against the CSV's own values (`om_df[om_df["om_case"] ==
om_case]`, raises `ValueError` if empty) — no continuous string exists
to pass it.

**Minimal continuous-scaling route, without editing `src/`:**
```python
lo = load_om_cost_by_year("om_low", start_year=..., end_year=...)
hi = load_om_cost_by_year("om_high", start_year=..., end_year=...)
frac = <Sobol-drawn value in [0, 1]>
blended = {
    tech: {
        yr: {
            "fom_usd_per_kw_yr": (1 - frac) * lo[tech][yr]["fom_usd_per_kw_yr"]
                                 + frac * hi[tech][yr]["fom_usd_per_kw_yr"],
            "vom_usd_per_mwh":   (1 - frac) * lo[tech][yr]["vom_usd_per_mwh"]
                                 + frac * hi[tech][yr]["vom_usd_per_mwh"],
        }
        for yr in lo[tech]
    }
    for tech in lo
}
scenario["om_cost_by_year"] = blended
```
Both endpoint dicts share identical `{technology}`/`{year}` key sets
(both loaded over the same `start_year`/`end_year` and both drawing
from the same CSV's technology list), so the per-key blend is
well-defined with no missing-key risk.

**Does `build_model_sliced` accept a caller-modified `econ`?** Not the
relevant question here — **O&M enters via `scenario`, not `econ`**
(a correction to TASK.md's own framing, which suggested "passing a
modified `econ`"). Traced directly:
`optimize_model_sliced.py:1353`: `om_cost_by_year=scenario[
"om_cost_by_year"]`, feeding straight into `compute_om_cost_npv(...)`
(`io.py:223-234`) with no further validation of its CONTENTS (only
`load_om_cost_by_year()` itself validates the `om_case` STRING, and
that validation is already satisfied before the post-hoc overwrite
happens — the same "mutate after `load_scenario()`, before
`build_model_sliced()`" convention as parameters 1-3). **Confirmed:
`build_model_sliced` will accept any nested dict shaped like
`load_om_cost_by_year()`'s own return value, including a client-side
interpolation between two of its outputs — no `src/` edit needed.**

**One bookkeeping wrinkle worth flagging**: whatever `om_case` string
value is passed to `load_scenario()` to obtain a valid baseline
scenario (e.g. `"om_central"`, needed only because `om_case` has no
default) will remain in `scenario["om_case"]` even after the
`om_cost_by_year` dict is overwritten — the recorded arm label will not
reflect the true, continuously-sampled O&M level actually used. The
Sobol script's own per-draw parameter-vector log is the source of
truth for this parameter, not the scenario dict's `om_case` field.

**Conclusion for the "if 3D shows O&M cannot be varied continuously...
that changes the design" contingency**: it does not trigger — O&M CAN
be varied continuously, by the route above. Parameter count stays at
five (subject to 3F's finding on parameter 5).

### 3E — which builder, and where does each output field come from?

**Builder**: `build_model_sliced()` (`src/optimize_model_sliced.py`) —
the only builder capable of exercising all five sampled parameters (in
particular, gas conversion is sliced-only, though conversion is not
itself one of the five sampled parameters here — the design holds
`conversion_case` at whatever arm the script's other CLI arms specify,
sampling only the five continuous parameters on top).

| Field | Source | Notes |
|---|---|---|
| `npv_total_cost_usd` | `float(pyo.value(m.system_cost_npv))` | Computed inline by `01b_run_baseline_sliced.py:404` — a one-line `pyo.value()` read on an already-solved `m`. No dependency on `extract_planning_diagnostics`. |
| `npv_gross_capex_report_window_usd` | `extract_planning_diagnostics(m, scenario, econ)["npv_gross_capex_report_window_usd"]` | As of `task_capex_reporting_field` (two tasks ago), `01b_` itself now computes this the same way (`01b_run_baseline_sliced.py`, current committed state) — confirmed sliced-safe by that task's own dimensionality audit (every variable `_capex_at()` reads is `Var(T)`/`Param(T)`/`Expression(T)`, none slice-indexed) and by existing precedent (`18_run_pareto_frontier.py`, `28_run_capital_frontier.py` already call this function on sliced, conversion-active models in committed code). |
| `final_storage_capacity_mwh` | `float(pyo.value(m.storage_capacity_mwh[len(years) - 1]))` | Computed inline, `01b_run_baseline_sliced.py`'s `dv` dict (`"final_storage_capacity_mwh"` key). `m.storage_capacity_mwh` is `Expression(T)` in both builders (confirmed in a prior task's dimensionality audit) — safe to index at the last year directly. |
| `solar_total_built_mw` | `sum(pyo.value(m.solar_public_add[t]) + pyo.value(m.solar_eaas_add[t]) for t in range(len(years)))` | Computed inline, `01b_run_baseline_sliced.py`'s `dv` dict (`"solar_total_built_mw"` key). Both are `Var(T)` in both builders — safe. |
| solve status | `solve_model(m, scenario=scenario)` return dict: `{"status": str(termination_condition), "optimal": bool, "solver_version": ...}` | **Important finding, not merely a lookup — see below.** |

**The solve-status finding, load-bearing for Part 2's "never abort the
run" requirement**: `solve_model()` (`src/optimize_model.py:1817-1889`,
shared by both builders) does **not** return gracefully on every
outcome. It raises `RuntimeError` immediately for any termination
condition other than `optimal` or `feasible`:
```python
res = solver_obj.solve(model, tee=tee)
tc = res.solver.termination_condition
if tc not in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible):
    raise RuntimeError(f"Solver failed with termination condition: {tc}")
...
return {"status": str(tc), "optimal": tc == pyo.TerminationCondition.optimal, "solver_version": solver_version}
```
So `status["optimal"] == False` can only ever be OBSERVED for the
narrow `feasible`-but-not-`optimal` case — every worse outcome
(infeasible, unbounded, solver error) never reaches a return statement
at all. **A per-draw loop that only checks the returned dict's
`"optimal"` key will still crash the entire Sobol run on the first
genuinely infeasible draw.** Part 2 must wrap the `solve_model()` call
itself in `try/except RuntimeError` (catching and recording the
exception's message as the draw's outcome) — not just inspect its
return value — to satisfy "catch and record non-optimal solves; never
abort the run." Flagged here because it is exactly the kind of
plausible-looking gap Part 2's own instruction anticipates in spirit
but does not spell out mechanically.

### 3F — `convert_capex_usd_per_mw = 748,000`: provenance

**No such figure, and no such key, exists anywhere in this codebase or
its documentation.** Exhaustive search performed exactly as instructed
("search the whole tree — docs, comments, task reports, the plan"),
reproduced in the Corrections section above. TASK.md's own narrower
search description ("a grep of `io.py` and `data/cost/raw/
source_log.md` returned nothing") undersold how absent it is — it
returned nothing anywhere, in any file type, under any spelling.

**This does not mean conversion capex has no source — it means the
CITED figure is not the real parameter.** The real, model-consumed
quantity is `scenario["conversion_capex_per_kw"]` (note: per **kW**,
not per MW — the cited figure carries a further ×1000 unit
mismatch on top of not existing), sourced from `conversion_scenarios()`
(`src/scenarios.py:1179-1245`, already quoted verbatim in
`task_conversion_capex_exposure.md`):
```python
"conv_central": {"capex_per_kw": 600.0, "uplift": 0.50, "retrofit_efficiency": 0.46},
"conv_low":     {"capex_per_kw": 850.0, "uplift": 0.40, "retrofit_efficiency": 0.42},
"conv_high":    {"capex_per_kw": 450.0, "uplift": 0.50, "retrofit_efficiency": 0.50},
```
Its own docstring declares the provenance class directly: **"DERIVED,
not SOURCED... Method: greenfield CC-minus-SC increment (Gas Turbine
World 2024: F-class 237MW simple cycle $713/kW installed; H-class
combined cycle $950-1,084/kW) plus a brownfield premium"**, with a
qualitative caveat (Ipieca 2022) that OCGT-to-CCGT retrofits are
industry-uncommon — a declared limitation, not a silent gap.

**Is a defensible band available? Yes — but sampling it correctly is
less obvious than it looks.** `conv_low`/`conv_high` are NOT
single-parameter "low cost"/"high cost" arms — each bundles THREE
parameters together (`capex_per_kw`, `uplift`, `retrofit_efficiency`)
into a pessimistic-vs-optimistic SCENARIO, not an isolated capex
sensitivity. Naively sampling "between `conv_low` and `conv_high`" as
if varying one parameter would **confound capex, uplift, and
efficiency into a single Sobol dimension** — the exact "shared driver,
sampled as if independent" failure this task's Section 1 already
condemns in the deferred `17_`, reproduced here in the opposite
direction (three parameters bundled where the design wants one).

**Recommendation, not a decision** (Part 1 does not redesign): if
conversion capex is to be parameter 5, sample `conversion_capex_per_kw`
ALONE, continuously over the range the three existing arms already
span — **[450, 850] USD/kW** — while HOLDING `uplift` and
`retrofit_efficiency` at their `conv_central` values (0.50, 0.46)
throughout the sweep, exactly mirroring how parameters 1-4 hold every
other model input fixed. This is not a band invented for this report;
it is the range already implied by three existing, declared-derived
arms, isolated to the one dimension the Sobol design actually wants.
The alternative — excluding conversion capex "like genset η" — does
not fit as cleanly: genset η's exclusion rests on its three values
being different OPERATING POINTS on the same datasheet, not
measurement uncertainty about one true value; conversion capex's three
values ARE three point-estimates of one uncertain retrofit-cost
quantity, which is precisely what Sobol sampling is for. **This is a
finding for a choice, not a redesign performed here**: sample
`conversion_capex_per_kw` over [450, 850] at central uplift/efficiency,
or exclude it — but not "sample between `conv_low` and `conv_high`" as
a single axis.

### 3G — Sobol mechanics: the exact scipy API, and a design-relevant capability

**`scipy.stats.qmc.Sobol` is importable and works as TASK.md
describes** — confirmed:
```python
from scipy.stats import qmc
s = qmc.Sobol(d=3, scramble=True, seed=42)
m = s.random(n=8)          # shape (8, 3)
```
This generates the low-discrepancy sequence only; building Saltelli's
A, B, and the D hybrid matrices from it (swap column `i` of B into A to
get `AB_i`) would indeed need to be done by hand — IF this were the
API actually used.

**But this scipy installation (1.15.2) also exposes
`scipy.stats.sobol_indices`, a complete, higher-level implementation of
the Saltelli 2010 procedure** — sampling, matrix construction, AND
index computation, in one call. Verified empirically with a disclosed,
side-effect-free toy example (no model code, no solve):
```python
from scipy.stats import sobol_indices, uniform
calls = []
def func(x):
    calls.append(x.shape)
    return np.array([x[0] + 2*x[1] - x[2]])   # toy linear function, s=1 output
res = sobol_indices(func=func, n=8, dists=[uniform(0,1)]*3, rng=42)
```
Result: `func` is called **exactly 3 times**, with `x` of shapes
`(3, 8)`, `(3, 8)`, and `(3, 24)` — the third call receives ALL `d`
hybrid matrices concatenated into one `(d, d*n)` array in a single
call, not `d` separate calls. Total evaluations `8+8+24 = 40 = n*(d+2)`
for `n=8, d=3`, matching the documented `n*(d+2)` cost exactly. `res`
carries `.first_order` and `.total_order` (shape `(s, d)` — supports
**multiple output QoIs in one decomposition call**, e.g. system cost
AND capital mobilised together, `s=2`) and a `.bootstrap(confidence_
level, n_resamples)` method for confidence intervals on the indices,
which TASK.md's own design does not otherwise mention obtaining.

**What this means for Part 2's design, stated as a finding, not acted
on**: `func` here would be a single Python function that, given a
`(d, n_batch)` array (`n_batch` varying across the three calls — 256,
256, 1280 for `N=256, D=5`), loops over its `n_batch` columns, and for
each column: sets the five sampled values on a freshly-loaded scenario/
econ, builds and solves the sliced model (wrapped in the `try/except
RuntimeError` from 3E), extracts the four output fields, **writes the
draw to disk immediately** (satisfying Part 2's incremental-write
requirement from inside `func`'s own loop, not from an outer harness),
and returns a finite value for that column (a policy is needed for
failed solves specifically because `sobol_indices` requires finite
function values — TASK.md's "genuine infeasibility may be rare or
absent" claim is the reason this is unlikely to matter often, but a
concrete fallback, e.g. a fixed penalty value with the draw separately
flagged in the per-draw log, is needed for the case it does). Total
solve count is UNCHANGED (`n*(d+2)` either way — TASK.md's own
"(2D+2) x N" arithmetic and `scipy.stats.sobol_indices`'s `n*(d+2)`
are the same formula), so the ≈72-minute estimate is unaffected by
which API builds the matrices.

**This is presented as a choice, not implemented**: using `scipy.stats.
sobol_indices` directly is less code and reuses scipy's own tested
Saltelli-formula implementation rather than a hand-written one (lower
risk of an index-formula bug), at the cost of `func`'s calling
convention being fixed by scipy rather than chosen freely, and of the
whole run being three long-blocking Python calls rather than a script
with full manual control over the outer loop (checkpointing/resuming
mid-run, for instance, is more natural to reason about with an
explicit outer loop over `qmc.Sobol`-generated points than inside a
function scipy calls three times). Flagged for a decision alongside
3D/3F's contingencies, per this task's own stated pattern for
Part-1 findings that change Part 2's shape.

---

## PART 1 — STOP

No files edited. `git status --porcelain` unchanged from before this
report (only the pre-existing uncommitted files from earlier tasks,
plus this new report). Three findings are presented for an explicit
choice before Part 2 proceeds, per this task's own stated pattern:

1. **3A** — gas availability must be set at TWO scenario-dict keys
   together (`gas_availability_factor` and `capacity_credit["gas"]`),
   not one, or the sampled parameter decouples from firm-capacity
   accounting.
2. **3F** — 748,000 USD/MW does not exist; the real parameter
   (`conversion_capex_per_kw`, USD/**kW**) has a defensible band
   ([450, 850]) already implied by three existing arms, but sampling
   it correctly means varying `capex_per_kw` ALONE at fixed central
   uplift/efficiency, not interpolating between the `conv_low`/
   `conv_high` scenario bundles (which would confound three parameters
   into one Sobol dimension).
3. **3G** — `scipy.stats.sobol_indices` already implements the entire
   Saltelli procedure (sampling, matrix construction, indices,
   bootstrap CIs) in this installed scipy version; using it changes
   `func`'s calling convention versus a hand-built `qmc.Sobol` +
   manual-formula design, at the benefit of reusing tested library code
   for the index arithmetic.

Additionally, **3D** resolves cleanly (O&M IS continuously settable, no
`src/` edit, via a client-side blend of `load_om_cost_by_year("om_low",
...)` and `load_om_cost_by_year("om_high", ...)`) and **3E** surfaces a
correctness requirement for Part 2 regardless of which design choice
above is made: `solve_model()` raises on non-optimal outcomes rather
than returning a status to check, so "never abort the run" requires
wrapping the solve call itself in `try/except`, not inspecting its
return value.

**Nothing here can move a solved value** — Part 1 read files, ran
read-only `git log`/`grep`, and made a small number of disclosed,
side-effect-free Python calls (data loaders returning plain dicts;
scipy API introspection on toy arrays unrelated to this model). No
scenario was built, no model was constructed, no solver was invoked
against this project's LP.

---

## ENCOUNTERED BUT NOT ASKED ABOUT

- `solve_model()` defaults to `solver="cbc"`. `CLAUDE.md`'s own toolchain
  note names "Pyomo + CBC/HiGHS" without specifying which a new script
  should prefer; TASK.md's "1.4 s per sliced solve" figure is stated,
  not attributed to a specific solver here. Whichever solver Part 2
  uses should be stated explicitly in the script, since 3,072 solves
  at a materially different per-solve time changes the ~72-minute
  estimate substantially.
- `26_run_f3_identification.py` and `27_run_hydro_minflow_
  sensitivity.py` (named in a prior task as likely sharing `01b_`'s
  now-corrected "extract_planning_diagnostics would crash on sliced
  models" assumption) were not re-examined here — outside this task's
  scope, noted only because the new script will need
  `npv_gross_capex_report_window_usd` the same way `01b_` now gets it,
  and those two scripts remain unfixed precedent for how NOT to do it.
- `scenario["om_case"]`/`scenario["gas_availability_case"]`/etc. will
  read as whatever placeholder arm was passed to `load_scenario()` even
  after this script overwrites the underlying continuous value — every
  per-draw parameter vector the script logs is the only reliable record
  of what was actually run; the scenario dict's own arm-label fields
  should not be trusted for provenance on Sobol draws. Worth a
  docstring line in the eventual script.
- `docs/limitations_register.md` L11 (`storage_binding` not meaningful
  under the sliced builder) and its unregistered sibling
  `storage_utilisation_rate` (found two tasks ago) are not among the
  four output fields Part 2 needs, but any future extension of this
  script's output set should avoid those two specifically.
- TASK.md cites the gas-availability lever as "F-5, 8.60 bn" — spot-
  checked that `docs/findings_frozen.md` has an F-5 titled "Gas fleet
  availability is the largest single lever," matching in substance;
  the 8.60bn figure itself was not independently re-derived (out of
  this task's questions).

**Nothing was committed, staged, or run.**

---

## PART 2 — IMPLEMENTATION

Released via a new TASK.md ("Monte Carlo implementation (Part 2,
RELEASED)"), which accepted all three of Part 1's flagged decisions in
full (both gas-availability keys set together with an assertion;
`conversion_capex_per_kw` sampled alone over [450, 850] holding
uplift/efficiency at central; `scipy.stats.sobol_indices` used
directly) and corrected the solve-count arithmetic before this task
started: `n*(d+2)` at `d=5` is `n*7`, not `(2d+2)*n` — the released
TASK.md's own correction, re-verified below rather than assumed.

`scripts/29_run_parameter_uncertainty.py` was written (new file only;
nothing else touched — confirmed, `git status --porcelain` at the end
of this task lists only this file and the report as untracked, every
other entry pre-dates this task). Not run. Not committed.

### Script structure

- **CLI**: `add_standard_arms(parser)` (the full standard-arm surface,
  `src/cli_arms.py`) plus `--n-base` (required, validated as a power of
  2), `--seed` (required), `--out-dir` (defaults to `results/
  parameter_uncertainty/`).
- **Two fail-loud pre-flight guards**, added beyond what TASK.md
  explicitly asked for but directly implied by Part 1's own findings:
  `--conversion-case conv_none` is rejected at parse time (parameter 5
  would silently be a no-op — `build_model_sliced()` builds no
  conversion mechanism at all under `conv_none`), and
  `--capital-treatment-case` other than `f1_tranche` is rejected
  (conversion capex is refused outright under `pre_f1_overnight` —
  every draw would fail identically, for a reason unrelated to any
  sampled parameter). Both verified to actually fire, not just
  written: tested directly against `parse_args()` with `conv_none` and
  `pre_f1_overnight` substituted into an otherwise-valid arm set — both
  raised `SystemExit(2)` as expected; a non-power-of-2 `--n-base` was
  tested the same way.
- **`run_one_draw(x_col, ...)`**: fresh `load_scenario(...)` (never a
  shared/mutated dict across draws), sets the five parameters via
  `set_five_parameters()`, builds and solves inside `try/except
  RuntimeError`, extracts the four output fields, appends one JSONL
  row, returns the two Sobol QoIs.
- **`func(x)`**: the callable `scipy.stats.sobol_indices` calls (up to
  3 times, per Part 1's 3G finding); loops over `x`'s columns calling
  `run_one_draw` for each, printing progress per draw.
- **`main()`**: loads `om_low`/`om_high`/`solar_capex_tv` ONCE (pure,
  deterministic CSV reads — no per-draw mutation risk, unlike the
  scenario dict itself, which is loaded fresh every draw as required),
  calls `sobol_indices(func=func, n=args.n_base, dists=[...], rng=
  args.seed)`, re-reads the JSONL to compute summary statistics and the
  feasibility count, calls `.bootstrap()`, writes `summary_n{N}_
  seed{S}.json`.

### How each parameter is set

Quoted from `set_five_parameters()`, the single function that owns all
five overrides:

```python
def set_five_parameters(scenario, x, om_low, om_high):
    gas_avail, gas_eta, solar_cf, om_frac, conv_capex_kw = x

    scenario["gas_availability_factor"] = float(gas_avail)
    scenario["capacity_credit"]["gas"] = float(gas_avail)
    assert scenario["gas_availability_factor"] == scenario["capacity_credit"]["gas"], (...)

    scenario["gas_eta"] = float(gas_eta)
    scenario["solar_cf"] = float(solar_cf)

    scenario["om_cost_by_year"] = build_om_cost_by_year_blend(om_low, om_high, float(om_frac))

    scenario["conversion_capex_per_kw"] = float(conv_capex_kw)
    scenario["conversion_uplift"] = CONV_CENTRAL_UPLIFT                    # 0.50, pinned
    scenario["conversion_retrofit_efficiency"] = CONV_CENTRAL_RETROFIT_EFFICIENCY  # 0.46, pinned
```

- **Parameters 2/3** (`gas_eta`, `solar_cf`): plain scalar overwrites,
  the same "mutate after `load_scenario()`" convention `scripts/
  20_run_cf1_solar_yield_sensitivity.py:119` already uses for
  `solar_cf` specifically.
- **Parameter 4** (O&M): `build_om_cost_by_year_blend()` linearly
  interpolates, per technology and per year, between two `load_om_
  cost_by_year()` outputs (loaded once, outside the draw loop, since
  they are pure functions of the CSV) — no `src/` edit, matching Part
  1's 3D finding exactly.
- **Parameter 5** (conversion capex): set alone. `conversion_uplift`
  and `conversion_retrofit_efficiency` are written on EVERY draw to
  `conv_central`'s own literal values (`0.50`, `0.46`), not merely left
  at whatever `load_scenario()` happened to set from `--conversion-case
  conv_central` — this is deliberate belt-and-suspenders: since the CLI
  guard already requires `--conversion-case conv_central` (excluding
  `conv_none`, and `conv_low`/`conv_high` are permitted by the guard
  but would leave uplift/efficiency at a DIFFERENT pair of values than
  the ones this script's docstring claims are held fixed) — **the
  script does not itself force `--conversion-case` to be exactly
  `conv_central`, only that it not be `conv_none`.** Stated plainly:
  if Zeus runs this with `--conversion-case conv_low` or `conv_high`,
  `conversion_uplift`/`retrofit_efficiency` are STILL overwritten to
  `0.50`/`0.46` (the `conv_central` values) on every draw regardless —
  the two example commands below use `conv_central` throughout so this
  never actually diverges in practice, but the code's behaviour if a
  different available arm were passed is: uplift/efficiency pinned to
  central anyway, capex swept over [450, 850]. Flagged here rather than
  silently relied upon.

### The gas-key assertion

```python
scenario["gas_availability_factor"] = float(gas_avail)
scenario["capacity_credit"]["gas"] = float(gas_avail)
assert scenario["gas_availability_factor"] == scenario["capacity_credit"]["gas"], (
    "gas availability keys diverged: ..."
)
```
A plain Python `assert`, not a logged warning — per TASK.md's own
instruction ("the failure mode is silent by nature, so the guard must
not be"), this raises `AssertionError` and stops the run immediately
rather than continuing with a decoupled pair. Both keys are written
from the SAME local float in consecutive lines, so this assertion is
not expected to ever fire in practice — it exists to catch a future
edit to this function that adds a third gas-availability-adjacent
write, or a copy-paste error, immediately and loudly rather than
letting it silently reproduce the exact defect this script was
written to avoid.

### Failed-solve policy

`run_one_draw` wraps the build+solve+extract sequence in `try/except
RuntimeError`. On success, the two real QoI values are returned and
logged. On a caught `RuntimeError` (the only exception `solve_model()`
raises for a bad termination condition, confirmed in Part 1):
```python
row.update({
    "status": "failed", "optimal": False, "error": str(e),
    "npv_total_cost_usd": FAILED_SOLVE_PENALTY_USD,
    "npv_gross_capex_report_window_usd": FAILED_SOLVE_PENALTY_USD,
    "final_storage_capacity_mwh": None,
    "solar_total_built_mw": None,
})
```
`FAILED_SOLVE_PENALTY_USD = 1e13` — three orders of magnitude above any
real system-cost or capex figure this model produces (which sit around
`1e10`), chosen so a failed draw is visually unmissable in any printed
distribution rather than blending into the real one, and finite (a
hard `scipy.stats.sobol_indices` requirement — NaN is rejected, so
skipping or nulling a failed draw was not an option). The summary JSON
carries `feasibility.n_failed`, `feasibility.failure_fraction`, and an
explicit `feasibility.warning` string that reads differently — with
the words **"MUST NOT be reported as a clean decomposition"** — once
`failure_fraction` exceeds `FAILURE_FRACTION_WARN_THRESHOLD = 0.05` (a
disclosed judgment call, documented in the script as the point past
which even one or two extra failures visibly distort a batch of this
size). `main()`'s own console output reprints this warning at the end
if it fires, so it cannot be missed by only reading the JSON.

### The two commands

Both use the exact arm combination that produced the actual, real,
already-verified `conv_central` central-configuration run inspected in
Part 1 (`results/baseline_sliced/summary__02c17f55.json`'s own
`arm_tag`, plus its `us_inflation`/`tariff_case`/`om_case` fields,
which that file records but its arm_tag string omits) — not a fresh
guess at what "central" should mean:

**Pilot** (`n=16`, `16*7=112` solves, ~3 minutes at the measured 1.4s/
solve — purpose: prove the pipeline end to end, NOT to produce
interpretable indices):
```
python scripts/29_run_parameter_uncertainty.py \
    --n-base 16 --seed 42 \
    --peak-case peak_unsuppressed --reserve-margin nirp_spinning \
    --storage-cc-case storage_cc_central --gas-build-case gas_unconstrained \
    --new-gas-deliverability constrained --gas-availability-case gas_avail_nameplate \
    --gas-retirement-case gas_retire_vintage_30 --gas-capex-case gas_capex_central \
    --ccgt-efficiency-case ccgt_eff_central --capital-recovery-case cr_central \
    --dscr-case dscr_central --us-inflation 0.02 \
    --capital-treatment-case f1_tranche --concessional-envelope-case env_dares \
    --gas-deliverability-case baseline --conversion-case conv_central \
    --hydro-case hydro_flat --tariff-case tariff_central --om-case om_central
```
**At `n=16` the printed `first_order`/`total_order` indices are NOT
interpretable** — d=5 with only 16 base points per matrix gives a
Monte Carlo standard error on each index comparable to the indices
themselves. Their only purpose at this stage is confirming the
pipeline runs to completion: parameter setting, the gas-key assertion,
the `try/except`, the JSONL writes, the `sobol_indices` call shape, and
the summary write, all without a crash — checked once by this task's
own mocked dry run (below), but not against the real solver, which
this task's whitelist forbids running.

**Full** (`n=256`, `256*7=1,792` solves — corrected from an earlier
design note's `3,072`, which used `(2d+2)*n`, the older Saltelli
convention for first-plus-total indices WITH a separate second-order
estimate; `scipy.stats.sobol_indices`'s documented cost is `n*(d+2)`
for first+total without second-order, matching the `1,792` figure
exactly, independently re-derived here as `256*(5+2)=1,792`):
```
python scripts/29_run_parameter_uncertainty.py \
    --n-base 256 --seed 42 \
    --peak-case peak_unsuppressed --reserve-margin nirp_spinning \
    --storage-cc-case storage_cc_central --gas-build-case gas_unconstrained \
    --new-gas-deliverability constrained --gas-availability-case gas_avail_nameplate \
    --gas-retirement-case gas_retire_vintage_30 --gas-capex-case gas_capex_central \
    --ccgt-efficiency-case ccgt_eff_central --capital-recovery-case cr_central \
    --dscr-case dscr_central --us-inflation 0.02 \
    --capital-treatment-case f1_tranche --concessional-envelope-case env_dares \
    --gas-deliverability-case baseline --conversion-case conv_central \
    --hydro-case hydro_flat --tariff-case tariff_central --om-case om_central
```
At 1.4s/solve, `1,792` solves is **≈42 minutes**, not the ≈72 minutes
an earlier design note estimated from the wrong solve count. **This
measured 1.4s/solve figure is from one configuration** (stated, not
re-verified here — this task ran no solves) — a conversion-active
sliced solve (every draw here, since `conv_central` is required by the
CLI guard) may take a different amount of time than whatever
configuration produced that measurement; the pilot's own wall-clock
time is the figure to actually use for planning the full run, not this
estimate.

### `--n-base`/`--seed` are required, not defaulted

Matches this project's standing discipline (every arm flag in `src/
cli_arms.py` is `required=True`, no exceptions) rather than picking a
silent default for either — an unstated Sobol `N` or seed would pick a
run length and a reproducibility guarantee with no visible signature
in the output, the same failure shape this project's register already
documents for other silently-defaulted numeric literals.

### The 748-vs-600 verdict

**Attribution correction, checked before answering the substantive
question**: TASK.md attributes the "1,941 MW × 748,000 USD/MW ≈ 1.45
bn" figure to "an earlier task report (`docs/task_conversion_capex_
exposure.md`)... derived." Re-checked: that report does not derive
this figure — it contains exactly one occurrence of "748,000" (line
358), in a sentence explicitly REFUTING it as a fabricated quantity
from an EARLIER TASK.md (two tasks before that one), not presenting it
as its own conclusion. The figure's actual origin is a TASK.md's own
prose, not a task report's derivation. This does not change the
substantive verdict below, but the attribution matters for anyone
tracing this figure's history later.

**The substantive question — "which the model actually uses" — verdict:
748,000 is simply wrong, not an alternate valid quantity.**

- The Expression added by `task_conversion_capex_exposure`
  (`m.conversion_overnight_capex_usd_by_year`, `optimize_model_
  sliced.py:1965`) and the objective term it mirrors
  (`convert_split_identity`, `optimize_model_sliced.py:1794-1799`) both
  read `_conv_capex_per_kw = float(scenario["conversion_capex_per_kw"])`
  — for `conv_central`, `600.0`, straight from `conversion_scenarios()`
  (`src/scenarios.py:1179-1245`). Neither reads, derives, or blends
  from any figure resembling 748,000.
- **Checked every combination that could plausibly produce 748 from
  the three real values {600, 850, 450}**: simple mean = 633.33 (not
  748); there is no nameplate-weighted or tier-effective blend to
  compute, because `capex_per_kw` does NOT vary by tier under any
  single arm — all three tiers pay the SAME rate within one arm, so
  there is no internal weighting scheme that could produce a
  fourth number from the three arms' single rates. No combination
  found reaches 748.
- **It never entered any computation.** `grep -rn "748"` across every
  `.py`/`.md`/`.csv`/`.yaml`/`.txt` file in this repository (Part 1's
  exhaustive search, re-confirmed unchanged this task) shows zero
  occurrences in any executable code or data file, at any point in
  this project's history that left a trace in the current tree. The
  reported capital figures already produced from this pipeline
  (`npv_gross_capex_report_window_usd` = 12.772 bn `conv_central` /
  15.020 bn `conv_none`, both now committed and real, per `docs/
  findings_frozen.md`'s own uncommitted-at-Part-1-time, now-committed
  update) were computed exclusively via the real 600/850/450 figures.
  **No published or reported number needs correcting because of this**
  — 748,000 only ever lived in a TASK.md's narrative text, never in
  a script, a data file, or a persisted result.
- **Not fixed, edited, or removed from any prior document** — outside
  this task's whitelist (no existing file may be touched). Recorded in
  this script's own module docstring (a NEW file, in scope) as a
  standing note for a future reader who encounters "1.45 bn" in this
  project's history.

---

## PART 3 — for the methodology chapter

> Two distinct kinds of uncertainty run through this model, and they
> must not be presented as one. **Scenario uncertainty** is which of
> several discrete, internally coherent worlds occurs — a gas-supply
> trajectory, a financing regime, a carbon-policy stance — and this
> project handles it with a structured, 38-arm sweep whose own
> constraint-relief cross-tabulation rules out combinations that could
> not jointly occur (a demand-growth path that assumes gas abundance
> paired with a gas-scarcity arm, for instance). **Parameter
> uncertainty** is different in kind: a small number of physical or
> cost quantities have one true, fixed value that is simply unknown
> within a sourced or derived range, and that uncertainty is properly
> represented by sampling continuously within the range, not by
> switching between discrete scenarios. Conflating the two — treating
> a parameter as though it were a scenario axis, or a scenario driver
> as though it were an independent parameter — produces worlds that
> were never real possibilities. This project's own deferred `17_run_
> monte_carlo.py` did exactly that, drawing demand growth and gas
> regime as independent parameters despite both responding to a shared
> macroeconomic driver. The same failure shape was found, this time,
> living INSIDE a single parameter rather than across two: gas fleet
> availability is stored at two separate points in this model's
> scenario representation — an annual energy ceiling and a peak-hour
> firm-capacity credit — initialised equal only because no separately
> sourced peak-hour figure yet exists. Sampling one without the other
> would silently decouple a single physical quantity from itself. The
> uncertainty-analysis script built for this thesis sets both together
> on every draw and asserts they remain equal, precisely because the
> failure mode is otherwise invisible in the output.
>
> A Sobol variance decomposition gives something a simple correlation
> or a one-at-a-time sensitivity sweep cannot: a **first-order** index
> per parameter, the share of output variance attributable to that
> parameter alone, holding all interactions aside; and a **total-order**
> index, the share attributable to that parameter INCLUDING every
> interaction it participates in with every other parameter. A large
> gap between a parameter's first- and total-order index is itself a
> finding — it says the parameter's effect on the output depends on
> what the OTHER parameters happen to be doing, not on its own value in
> isolation, a structural statement about the model's response surface
> that no pairwise correlation could reveal, because correlation is
> blind to interaction by construction.
>
> Every one of the five sampled parameters here carries a UNIFORM
> prior between two bounds, and this must be stated honestly rather
> than left to look like more than it is: a uniform prior over a
> sourced range is a statement of ignorance between two endpoints, not
> a claim that intermediate values are equally likely for any
> physical reason. Two parameters (solar capacity factor; the O&M
> level's blend between two sourced cost cases) have sourced endpoints
> and nothing more assertive is claimed. One (fleet thermal efficiency)
> has endpoints derived from a measurement-boundary ambiguity an
> earlier project phase identified but never implemented. One
> (conversion capex) has endpoints derived from a documented cost
> model rather than a market-observed range. And one — gas fleet
> availability — is both the WIDEST prior and the LARGEST lever in the
> entire model, with no sourced central value to narrow it around. If
> the variance decomposition places most of the output band's variance
> on gas availability, that is a legitimate and useful finding — it is
> plausibly the single measurement most worth Nigeria actually making —
> but it must be reported as a consequence of how wide an honest prior
> had to be drawn given what is and is not sourced, not as an emergent
> discovery the model made on its own. The prior put it there, and the
> thesis must say so in the same sentence it reports the result.
>
> Finally, none of this changes what kind of model this is. The
> underlying LP remains entirely deterministic: for any one fixed
> point in the five-dimensional parameter space, the model returns one
> exact optimal plan, not a distribution. What is stochastic here is
> the SAMPLING of which fixed points to evaluate — a standard global
> sensitivity-analysis technique borrowed to characterise how the
> model's deterministic output moves as its uncertain inputs move,
> not a claim that the planning problem itself has been reformulated
> as a stochastic program. That distinction is easy to blur in prose
> and must not be: this is a sensitivity analysis conducted ON a
> deterministic model, not a probabilistic model of Nigeria's power
> sector.

---

## SUMMARY: NOTHING HERE CAN MOVE A SOLVED VALUE

`scripts/29_run_parameter_uncertainty.py` is a new, standalone file
that imports from `src/` but is not imported BY anything in `src/` or
by any existing script — nothing that exists today executes a single
line of it. It contains no edit to any existing file (confirmed:
`git status --porcelain` at the end of this task lists only this new
script and this report as untracked). Every model-touching call inside
it (`load_scenario`, `build_model_sliced`, `solve_model`,
`extract_planning_diagnostics`) is a call to an EXISTING, unmodified
function with the SAME arguments those functions already accept from
every other caller in this project — this script asks the model
different QUESTIONS (different parameter values, all within already-
sourced-or-derived ranges) but changes no function's code. And per this
task's whitelist, none of it was actually run: no draw was solved, no
JSONL was produced, no summary was computed against a real model. The
only executions performed were (a) a syntax check, (b) an import-level
smoke test confirming the module loads without a circular import or
`main()` firing, (c) a CLI-argument test confirming `parse_args()`
accepts a valid full arm set and rejects `conv_none`, non-`f1_tranche`,
and a non-power-of-2 `--n-base`, and (d) a fully mocked dry run of
`main()` with `run_one_draw` replaced by a fast synthetic function, to
prove the `scipy.stats.sobol_indices` wiring, the JSONL round-trip, and
the bootstrap-confidence-interval JSON serialisation all work end to
end — none of which built or solved this project's actual LP.

---

## ENCOUNTERED BUT NOT ASKED ABOUT

- `res.bootstrap(...).first_order.confidence_interval` (and
  `.total_order`) is a scipy `ConfidenceInterval` namedtuple of numpy
  arrays, not JSON-serialisable as returned, and can contain `NaN`
  (scipy's own BCa bootstrap is documented to fail on degenerate
  resamples — observed directly while building this script, on an
  unrelated toy function, not on this project's model). The script
  converts both to plain nested lists with `NaN -> null` before
  writing the summary JSON; not something either TASK.md anticipated,
  found only by actually exercising the API on toy data per this
  task's own disclosed-judgment-call allowance.
- If a future run passes `--conversion-case conv_low` or `conv_high`
  (permitted by the CLI guard, which only excludes `conv_none`),
  `conversion_uplift`/`conversion_retrofit_efficiency` are still pinned
  to `conv_central`'s values on every draw regardless, per `set_five_
  parameters()`'s design — flagged in this report's "how each
  parameter is set" section above; the two example commands both use
  `conv_central` throughout so this never actually arises with the
  commands as given.
- `scenario["gas_availability_case"]`/`scenario["om_case"]`/`scenario[
  "conversion_case"]` all continue to read whatever placeholder CLI
  value was passed to satisfy `load_scenario()`'s required arguments,
  even after this script overwrites the underlying continuous values —
  the per-draw JSONL row is the only reliable provenance record for
  what was actually run on that draw. Stated in the script's own
  docstring and in the summary JSON's `om_case_caveat` field; repeated
  here since it applies to `gas_availability_case`/`conversion_case`
  too, not only `om_case` as TASK.md's own text singled out.
- `scripts/26_run_f3_identification.py` and `scripts/27_run_hydro_
  minflow_sensitivity.py` (flagged in Part 1 as likely candidates for
  the same "would crash on sliced models" assumption `01b_` carried)
  were not touched or re-examined here — outside this task's whitelist
  regardless, and not load-bearing for this script, which builds its
  own scenario/model calls directly rather than reusing either file.
- The pilot command's ~3-minute estimate and the full command's ~42-
  minute estimate both rest on the 1.4s/solve figure TASK.md itself
  supplied, for an unstated configuration — this task did not
  independently re-measure solve time (running the solver is outside
  this task's whitelist). The pilot run's own wall-clock time is the
  figure to actually plan the full run against.

**Nothing was committed, staged, or run.**