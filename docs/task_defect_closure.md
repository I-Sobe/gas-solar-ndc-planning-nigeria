# TASK — close the declared defects in the REL and DEM runners

Branch `phase-fin2-d17`. Reported incrementally per TASK.md §0.

---

## PART 1 — D1: the interaction classifier disagrees with itself

**File:** `scripts/13_run_dem2_growth_gas_interaction.py`,
`compute_interaction_matrix()`.

### The defect, as found (old code, lines 232-257 before this edit)

```python
interaction = (
    float(high_prem - low_prem)
    if (high_prem is not None and low_prem is not None)
    else None
)

rows.append({
    "gas_case":           gas,
    "financing_arm":      fin,
    "policy_label":       pol,
    "gas_premium_low_growth":  low_prem,
    "gas_premium_high_growth": high_prem,
    "interaction_effect_usd":  interaction,
    "interaction_amplifies": (
        interaction > 0 if interaction is not None else None
    ),
    "interpretation": (
        "Demand growth AMPLIFIES gas scarcity cost (super-additive)."
        if (interaction is not None and interaction > 1e6)
        else (
            "Demand growth and gas scarcity are approximately independent."
            if interaction is not None
            else "Cannot compute — missing data."
        )
    ),
})
```

`interaction_amplifies` tests `interaction > 0`; `interpretation` tests
`interaction > 1e6` — two different thresholds for the same concept.
Any `interaction` strictly between 0 and 1e6 produced
`interaction_amplifies=True` beside `interpretation="approximately
independent"` in the same row — the console (which renders the
boolean) and the CSV (which carries the text) disagree on that row.
No sub-additive branch existed at all: a large NEGATIVE interaction
(the plan's own example, −$201bn) fell into the same "approximately
independent" text as a genuinely near-zero one. The 1e6 tolerance is
absolute in a table the plan itself says spans 10⁻¹ to 10¹¹ USD.

### The fix

One classification, computed once, drives both fields. Three
branches — super-additive, approximately independent, sub-additive —
using a **relative** tolerance: `reference_magnitude =
max(|low_prem|, |high_prem|)`, `tolerance = 0.01 × reference_magnitude`
(1%). Chosen because `interaction` is itself a difference of two gas-
scarcity premiums, so its scale should be judged against theirs, not
a fixed dollar figure — exactly the plan's own diagnosis of why 1e6
fails across a ten-order-of-magnitude table. 1% is tight enough that
LP solver noise (orders of magnitude below 1% of any premium in this
project) still reads as independent, loose enough that a real,
reportable divergence at any scale in the table clears it.

New code:
```python
if interaction is None:
    classification = "cannot_compute"
    amplifies = None
    interpretation = "Cannot compute — missing data."
else:
    reference_magnitude = max(abs(low_prem), abs(high_prem))
    tolerance = 0.01 * reference_magnitude
    if interaction > tolerance:
        classification = "super_additive"
        amplifies = True
        interpretation = (
            "Demand growth AMPLIFIES gas scarcity cost (super-additive)."
        )
    elif interaction < -tolerance:
        classification = "sub_additive"
        amplifies = False
        interpretation = (
            "Demand growth REDUCES gas scarcity cost (sub-additive) -- "
            "higher demand growth softens the scarcity penalty under "
            "this gas regime, a real result, not noise."
        )
    else:
        classification = "approximately_independent"
        amplifies = False
        interpretation = (
            "Demand growth and gas scarcity are approximately independent."
        )

rows.append({
    "gas_case":           gas,
    "financing_arm":      fin,
    "policy_label":       pol,
    "gas_premium_low_growth":  low_prem,
    "gas_premium_high_growth": high_prem,
    "interaction_effect_usd":  interaction,
    "interaction_classification": classification,
    "interaction_amplifies": amplifies,
    "interpretation": interpretation,
})
```

`interaction_amplifies` (kept for backward compatibility) is now
DERIVED from `classification` — `True` only for `super_additive`,
`False` for both `approximately_independent` and `sub_additive` (never
amplification), `None` only for `cannot_compute`. No more independent
threshold to disagree with the text.

**The sub-additive case is preserved, not discarded.** It gets its own
distinct interpretation text (not merged into "independent"), its own
`interaction_classification` value machine-readable in the CSV/JSON, a
`summary["interaction_findings"]` entry now including
`"classification"` alongside `"amplifies"`, and its own console label
(`SUB-ADDITIVE`, not folded into "no"/"—").

**Console updated too** (not asked for explicitly, but the original
defect statement itself names the console as part of the problem —
"the console renders the boolean; the CSV carries the text"): the
`Amplifies` column is replaced with a `Class` column printing the
three-way label (`SUPER-ADDITIVE` / `independent` / `SUB-ADDITIVE`)
directly from `interaction_classification`, rather than re-deriving a
YES/no from `interaction_amplifies`.

### Why this cannot move a solved value

`compute_interaction_matrix()` runs entirely on already-computed
`npv_total_cost_usd` values pulled from `gas_premium_df` — a
post-processing function with no `load_scenario()`, `build_model()`,
or `solve_model()` call anywhere in it or its caller chain within this
edit. It relabels numbers that were already computed before this
change; it computes none of them. `run_dem2_growth_gas_matrix()` (the
function that actually solves LPs) is untouched.

`python3 -m py_compile scripts/13_run_dem2_growth_gas_interaction.py`
— passes.

---

## PART 2 — D23: `eps_levels` is silently dropped

**File:** `src/optimize_experiments.py`.

### The defect, as found

TASK.md named "around line 774" — verified against the live file and
traced to its actual root, which sits elsewhere. `run_rel3_financing_
frontier` (then at line 712, `eps_levels=None` parameter at line 732,
docstring at line 772-774) and `run_rel2_marginal_cost_curves` (then
at line 1876, `eps_levels=None` at line 1897) both correctly resolve
their own `eps_levels` parameter (`if eps_levels is None: eps_levels =
reliability_levels_log()`) — TASK.md's cited line 774 sits inside this
already-correct block. **The actual defect is one level down**: both
functions call `run_reliability_sweep()` to do the actual sweep, and
that function — the shared code both delegate to — **had no
`eps_levels` parameter at all**:

```python
def run_reliability_sweep(
    scenario,
    econ,
    gas_case=None,
    financing_label=None,
    policy_label=None,
    emissions_cap_by_year=None,
):
    ...
    eps_levels = reliability_levels_log()   # unconditional, no caller override possible
```

So a caller's own correctly-resolved `eps_levels` was computed, then
never reached the place that actually iterates over it
(`for eps in eps_levels:`, inside `run_reliability_sweep`) — silently
replaced by `reliability_levels_log()` regardless. Confirmed by
grepping every call site of `run_reliability_sweep` (exactly three:
two inside `run_rel3_financing_frontier`, one inside `run_rel2_
marginal_cost_curves`) — none passed `eps_levels`, because the
parameter did not exist to receive it.

### The fix

`run_reliability_sweep` gains `eps_levels=None`, resolved the same way
as every other function in this file (`if eps_levels is None:
eps_levels = reliability_levels_log()`), replacing the unconditional
line. All three call sites updated to pass `eps_levels=eps_levels`
through from their own (already-correct) local variable.

### Point 3 — audit of the whole file for the same pattern

Ran a static AST audit (`ast.parse` + a visitor comparing every
function's parameter list against every `Name` node referenced in its
body — a pure parse, no execution, per the whitelist's "read-only
syntax checking by any means") across all of `src/optimize_
experiments.py`. **One hit:**

```
3606: def run_tariff_bankability_sweep -- unused params: ['args', 'kwargs']
```

Read in context:
```python
def run_tariff_bankability_sweep(*args, **kwargs):
    """
    DEPRECATED — superseded by run_fin2_blended_bankability_sweep.
    ...
    """
    raise NotImplementedError(
        "run_tariff_bankability_sweep is deprecated. "
        "Use run_fin2_blended_bankability_sweep instead."
    )
```

**Not the same defect.** This is a deliberate deprecation shim:
`*args, **kwargs` accepts any call shape specifically so it fails
loudly (`NotImplementedError`, naming its replacement) regardless of
how a stale caller invokes it, rather than raising a confusing
`TypeError` on argument mismatch. Nothing is silently dropped — the
function does nothing WITH any argument, on purpose, because it does
nothing at all. Reported per TASK.md's instruction, fixed nowhere
(there is nothing to fix).

**A caveat on the audit's own limits, stated rather than assumed away:**
a literal "parameter never referenced by name" check (what the AST
scan does) cannot catch the *shape* of the actual D23 bug — a
parameter that IS referenced (in `if eps_levels is None: eps_levels =
...`) but whose resolved value is then never forwarded to a downstream
call that needed it. That class of bug requires tracing call chains,
not scanning names, which is how D23 itself was actually found (by
reading, not by the audit). The audit is reported as a real, useful,
but structurally partial check — not as proof no further instance of
D23's specific shape exists anywhere else in a ~3700-line file.

### Why this cannot move a solved value

Adding a parameter with a default that reproduces the PRIOR
unconditional behaviour when omitted (`eps_levels=None` →
`reliability_levels_log()`, byte-identical to what every existing
caller already got) changes nothing for any caller that does not pass
`eps_levels` explicitly — which is every caller in this codebase
today (confirmed: `09_`, `10_`, `11_` all pass their own eps grids
into their own top-level functions, none currently passes `eps_levels`
directly into `run_reliability_sweep`). The frozen anchors are
produced by `01b_run_baseline_sliced.py` and `02_run_ndc_caps.py`,
neither of which calls `run_reliability_sweep`, `run_rel3_financing_
frontier`, or `run_rel2_marginal_cost_curves` at all.

`python3 -m py_compile src/optimize_experiments.py` — passes.

---

## PART 3 — D24: threshold logic disagrees with its own status column

**File:** `src/optimize_experiments.py`, `compute_frontier_shift()`
(the function `scripts/11_run_rel3_financing_frontier.py` calls to
produce `frontier_threshold.csv`/`frontier_shift.csv` — no separate
threshold logic exists inside the runner file itself, confirmed by
grepping it for "threshold"/"status": every occurrence is either
printing a value already computed here, or citing this function).

### Reading both — the threshold computation and the status assignment

Threshold (per gas_case/financing_label/policy_label):
```python
feasible = grp[grp["status"] == "feasible"]
threshold = float(feasible["eps"].min()) if len(feasible) > 0 else None
```
`status` itself is assigned inside `run_reliability_sweep`'s
`_make_row` closure: `"feasible"` on a successful solve, `f"infeasible:
{e}"` on a `RuntimeError`. Read correctly by the `== "feasible"` test
— no disagreement there.

### Where they actually diverge

Frontier shift, before this fix:
```python
shift = (
    float(pub_thresh - eaas_thresh)
    if (pub_thresh is not None and eaas_thresh is not None)
    else None
)
necessity = (
    pub_thresh is None and eaas_thresh is not None
)
"eaas_strictly_better": (shift is not None and shift > 1e-4),
"eaas_necessary":        necessity,
```

**Divergence 1 (the one TASK.md named):** `shift` is undefined
whenever `pub_thresh is None`. So any row with `eaas_necessary=True`
(public arm fully infeasible, EaaS feasible — this runner's own
docstring: *"This is the strongest possible finding"*) **always** had
`eaas_strictly_better=False` in the very same row — a row simultaneously
labelled "the strongest possible finding" and "not better." These are
not measuring genuinely different things (both purport to answer "is
EaaS better than public financing here"), so TASK.md §Part 3.4's
fallback (documentation defect, not logic) does not apply — this is a
logic defect.

**Divergence 2 (found by this fix's own verification, not by
inspection — a fully-infeasible-public synthetic case, no model run):**
`threshold_df["feasibility_threshold_eps"]` is built via
`pd.DataFrame(threshold_rows)`; a column mixing Python `None` with
`float` values is coerced by pandas to `float64`, turning every
missing threshold into `NaN`, not `None`. `pub_thresh is None` is
`False` for a `NaN` extracted from such a column — it could only ever
be `True` when the public_only row was missing from the group
ENTIRELY (`len(pub) == 0`), never when it was present and correctly
recorded as infeasible. **`eaas_necessary` was therefore effectively
unreachable on any real multi-row sweep** — a second, deeper defect
underlying the one named, confirmed by a synthetic test (below) that
reproduced it before the `pd.isna()` fix and passed after.

### The fix

One classification, computed once (same shape as D1), replacing both
independent tests, with NaN-safe missingness checks:

```python
pub_missing  = pd.isna(pub_thresh)
eaas_missing = pd.isna(eaas_thresh)

shift = (
    float(pub_thresh - eaas_thresh)
    if (not pub_missing and not eaas_missing)
    else None
)

if pub_missing and eaas_missing:
    classification = "cannot_compute"
elif pub_missing:
    classification = "eaas_necessary"
elif eaas_missing or shift < -1e-4:
    classification = "eaas_worse"
elif shift > 1e-4:
    classification = "eaas_strictly_better"
else:
    classification = "no_improvement"

eaas_strictly_better = classification in ("eaas_necessary", "eaas_strictly_better")
eaas_necessary = classification == "eaas_necessary"
```

Necessity is now treated as the STRONGEST form of "strictly better,"
not a separate, disagreeing category — `eaas_necessary=True` implies
`eaas_strictly_better=True`, always. A new `eaas_worse` state also
surfaces the case this runner's own docstring calls an assumption, not
a certainty (*"Negative: Shouldn't happen (EaaS weakly dominates
public_only)"*) — previously silently identical to "no improvement"
(both `False`); now flagged with its own classification and an
`interpretation` naming the anomaly explicitly, per the same "must not
be discarded as noise" principle D1 applied to the sub-additive case.
`frontier_classification` is a new field carrying the five-way tag in
both `shift_df`/`frontier_shift.csv` and the runner's summary JSON.

**Verified with a synthetic test** (no solver, no scenario — six
fabricated `(gas_case, financing_label, status, eps)` row sets covering
necessary / strictly-better / no-improvement / worse / cannot-compute,
run directly through `compute_frontier_shift`):
```
gas_case  frontier_classification  eaas_strictly_better  eaas_necessary
A         eaas_necessary           True                  True
B         eaas_strictly_better     True                  False
D         eaas_worse               False                 False
E         cannot_compute           False                 False
F         no_improvement           False                 False
ALL ASSERTIONS PASSED
```
confirming `eaas_necessary` now correctly fires when the public arm is
present-but-infeasible (not only when its row is entirely absent), and
`eaas_necessary=True` never again coexists with
`eaas_strictly_better=False`.

**Console and summary JSON updated to match**
(`11_run_rel3_financing_frontier.py`): the `EaaS better`/`Necessary`
columns now read off the same `frontier_classification` (so a
necessary row prints YES/YES, never YES/no), and an `ANOMALY:` line
prints whenever `frontier_classification == "eaas_worse"`.

### Why this cannot move a solved value

`compute_frontier_shift` takes `df` — a DataFrame of already-computed
rows — and returns two derived DataFrames; it calls no solver, builder,
or `load_scenario()`. It reads `pub_thresh`/`eaas_thresh` from values
`run_rel3_financing_frontier` already produced; this fix changes how
those two numbers are compared and labelled, not what either number
is. `run_rel3_financing_frontier` itself, and every `load_scenario()`/
`build_model()` call inside it, is untouched by Part 3.

`python3 -m py_compile src/optimize_experiments.py` and
`scripts/11_run_rel3_financing_frontier.py` — both pass.

---

## PART 4 — D17: the EaaS arm may not be constrained at all

**Files:** `src/optimize_experiments.py`,
`scripts/11_run_rel3_financing_frontier.py`. Diagnostic only — the
frontier computation itself (Part 3's `compute_frontier_shift`, and
`run_rel3_financing_frontier`'s own `load_scenario()`/`build_model()`
calls) is unchanged by this Part.

### Where the EaaS constraint is built, and reusing `03_`'s approach

`m.eaas_bankability` is built **only** when `financing_regime ==
"eaas"` (`optimize_model.py:689-764`) — structurally absent, not
merely zero, under `financing_regime != "eaas"`
(`optimize_model.py:778-781` builds `m.eaas_disable` instead). Any
per-row diagnostic reading it must guard on financing regime, not just
on the constraint's numeric value.

`03_run_ndc_eaas.py`'s `bankability_slack_per_mw_by_year(m, scenario,
years)` (its own line 296) already computes exactly the right
quantity — per-year slack in `m.eaas_bankability`, USD per MW, `None`
when `solar_eaas_add[t] ~ 0` (no information that year). Per TASK.md's
explicit instruction, **reused, not reinvented** — but not imported
directly: `03_run_ndc_eaas` is not a valid Python module name
(identifiers cannot start with a digit), which is why no runner in
this project imports another runner. The function is duplicated
verbatim into `src/optimize_experiments.py`, which is where every
OTHER diagnostic shared across runners in this project already lives
(`run_reliability_sweep`, `compute_frontier_shift`, etc.) — `03_`'s own
copy is untouched (outside this task's whitelist); see the closing
list for the follow-on this leaves.

A second helper, `eaas_bankability_binding_years(m, scenario, years)`,
counts binding years on top of it:
```python
def eaas_bankability_binding_years(m, scenario, years) -> dict:
    slack = bankability_slack_per_mw_by_year(m, scenario, years)
    evaluable = [v for v in slack.values() if v is not None]
    binding = [v for v in evaluable if v < 1.0]
    return {"n_binding_years": len(binding), "n_evaluable_years": len(evaluable)}
```
"Binds" reuses the threshold **already established for this exact
quantity** in `docs/findings_provisional_pre_sweep.md`'s F-29
("binding years (slack < 1 USD/MW)") — not a new tolerance invented
here. `n_evaluable_years` is reported alongside the count rather than
assuming a fixed 30-year denominator: a scenario with little EaaS
build has few evaluable years, and TASK.md's own cited figure ("8 / 3
/ 1 years out of 30") should not be treated as a fixed denominator
either — see the citation note below.

### The wiring — every EaaS row of `11_`'s output

Inside `run_reliability_sweep`'s `_make_row`, guarded on
`financing_label == "eaas" and hasattr(m, "eaas_bankability")`:
```python
if financing_label == "eaas" and hasattr(m, "eaas_bankability"):
    _bank = eaas_bankability_binding_years(m, scenario, scenario["years"])
    eaas_bankability_n_binding = _bank["n_binding_years"]
    eaas_bankability_n_evaluable = _bank["n_evaluable_years"]
else:
    eaas_bankability_n_binding = None
    eaas_bankability_n_evaluable = None
```
added to the row dict as `eaas_bankability_n_binding_years` /
`eaas_bankability_n_evaluable_years` (None on every non-EaaS row —
REL-2's sweep never sets `financing_regime="eaas"`, so this is a
no-op generalisation there, not a behaviour change for REL-2).
Threaded through `reliability_results_to_df` (which enumerates
columns explicitly and would otherwise have dropped them, the same way
it already drops `solar_eaas_total_mw` — see the closing list) so they
reach `frontier_all.csv`.

### The flag — Part 4, point 3

`compute_frontier_shift`'s `threshold_rows` now also reads the
binding-year count **at the specific threshold row** (the eps level
the reported `feasibility_threshold_eps` was actually achieved at —
the row `frontier_shift_eps` is derived from), alongside the existing
`cost_at_thresh`/`eaas_mw_at_thresh` extraction. `shift_rows` derives:
```python
eaas_not_a_financing_effect = (
    None if no EaaS threshold row exists to check
    else (eaas_bankability_n_binding_years_at_threshold == 0)
)
```
`frontier_shift.csv` and the summary JSON both carry
`eaas_bankability_n_binding_years_at_threshold` and
`eaas_not_a_financing_effect`. The console prints a `D17 WARNING` line
whenever the flag is `True`, in the same spirit as this project's
existing `LEAKAGE FLAG` prints elsewhere — **the number is never
suppressed, only labelled**, per TASK.md's explicit instruction.

### The docstring — Part 4, point 4

`11_run_rel3_financing_frontier.py`'s module docstring caveat is
rewritten to record: the original D17 assumption, that `tariff_case`
is now a required, sourced arm that makes the condition measurable
rather than assumed, exactly what "binds" means (reusing F-29's own
threshold), and how to read `eaas_not_a_financing_effect` per row.

### A citation correcting TASK.md itself, verified rather than assumed

TASK.md attributes "binding years 8 / 3 / 1 out of 30 at tariff_low /
tariff_central / tariff_high" to **`findings_frozen.md`**. Searched
that file directly: no occurrence of "8/3/1", "binding years", or this
finding under F-13 (`findings_frozen.md`'s own F-13 is about Z*
invariance across tariffs — it does not mention binding-year counts at
all). The actual figure lives in two PRE-O&M provisional documents —
`docs/findings_provisional_pre_om.md`'s F-13 and, in sharper form with
the exact "slack < 1 USD/MW" threshold, `docs/findings_provisional_
pre_sweep.md`'s F-29. Both predate the O&M correction (register L28),
which can shift the optimal solar/gas mix and therefore which years a
downstream constraint binds — precisely why Part 4 makes this
MEASURABLE PER RUN rather than re-asserting the old count. This task
does not attempt to reproduce or re-verify "8/3/1" (no solve
performed); the diagnostic exists so the next actual run can measure
its own count rather than citing an unverified, pre-O&M figure.

### Why this cannot move a solved value

Every new line in this Part reads a `pyo.Expression`/`pyo.Constraint`
value (`m.eaas_bankability`'s slack, `m.solar_eaas_add`) off an
ALREADY-SOLVED model, inside a function (`_make_row`) that already
runs strictly after `solve_model()` returns — no new Var, Constraint,
or Expression is added to either builder, and no existing one is
modified. `compute_frontier_shift`'s threshold/shift computation
itself (which eps is chosen as the threshold, what `frontier_shift_eps`
equals) is byte-for-byte unchanged by this Part — confirmed by
re-reading it after this edit: the only additions are further fields
alongside the pre-existing ones. `run_rel3_financing_frontier`'s own
`load_scenario()`/`build_model()` calls are untouched.

`python3 -m py_compile src/optimize_experiments.py` and
`scripts/11_run_rel3_financing_frontier.py` — both pass.

---

## PART 5 — D21: one eps axis, one rationale

**New file:** `docs/eps_axis_rationale.md`. Assembled from three
existing sources per TASK.md's instruction, not invented: the plan's
own framing (`merged_correction_plan_v2.md`), `methodology_genself_
backstop.md` §5 (what the obligation is) and §7 (the open item this
document closes, verbatim: *"Several runners currently sweep different
ranges; a single documented rationale is outstanding"*), and `23_run_
selfgen_frontier.py`'s own already-reasoned calibration.

**A correction applied while assembling, not just quoting:** the
plan's original wording cites "Nigeria's observed lambda 0.51-0.57" as
an "indicative anchor" for the eps axis without qualification. `23_`'s
own docstring already recorded why that needs a caveat: *"lambda is
grid / (grid + selfgen) on a Tier 2 base, while epsilon is shortfall /
Tier 1 demand. Different denominators."* The new document states both
— what λ is evidence FOR (a large, policy-relevant off-grid share
exists, supporting wide axes generally) and what it does NOT justify
(a specific numeric eps value, or eps = 1 − λ) — rather than repeating
the plan's unqualified framing.

**A second correction, verified rather than assumed:** TASK.md's own
framing says "the REL suite sweeps four different eps axes across six
scripts." Read all six scripts' actual eps grids directly (not
assumed from the plan's count):

| Axis | Values | Scripts |
|---|---|---|
| `reliability_levels_log()` default | `[0.20, 0.15, 0.10, 0.08, 0.06, 0.05]` | `09_`, `10_`, both retired `09b_` scripts |
| REL-3's own | `[0.45, 0.40, 0.35, 0.30, 0.25, 0.20]` | `11_` |
| `23_`'s calibrated grid | `[0.05, 0.10, ..., 0.60]` (12 points) | `23_` |

**Three distinct axes, not four**, reused across the six named
scripts (four of the six share the default axis). Corrected in the
new document itself (§5), not silently matched to the claimed count.

The document states, per TASK.md's required content: what eps is
(§1: the grid service obligation, not an adequacy standard —
`methodology_genself_backstop.md` §5.1), why it has no external
measurement (§2: a policy instrument, not a physical parameter — §7,
verbatim), why the axis must bracket the model's own feasibility
boundary (§3, using `23_`'s own history as the worked example of what
goes wrong otherwise), what the lambda anchor does and does not
justify (§4, the correction above), and which axis each REL script
should use and why they may legitimately differ (§5, the corrected
three-axis table with the research-question-specific justification for
each).

**One-line docstring references added** to `09_run_rel1_feasibility.
py`, `10_run_rel2_marginal_cost.py`, `11_run_rel3_financing_frontier.
py`, and `23_run_selfgen_frontier.py` — each naming
`docs/eps_axis_rationale.md` and its own axis's place in the table, so
a reader of any one script is pointed at the shared rationale rather
than re-deriving it.

**No eps grid was changed.** Confirmed by re-reading each of the four
edited scripts' own `EPS_LEVELS`/`EPS_GRID` constants after the edit —
byte-for-byte identical to before. `python3 -m py_compile` on all four
— passes.

### Why this cannot move a solved value

A new markdown file and four one-line docstring insertions (no code
line touched in any of the four scripts) cannot execute anything.

---

## PART 6 — the two retired 09b scripts (report only, no file edits)

Read both files in full.

### `retired/09b_run_rel1_mode_sensitivity.py`

**The question it asked:** does `grid_service_mode = 'annual'` vs
`'total'` change REL-1's investment trajectories and feasibility
threshold?

**Verified: (a) dead, the answer was found and acted on.** The file's
own retirement note (already present, written at the time it was
retired) states this precisely: D6 removed `'total'` permanently, and
`methodology_genself_backstop.md` §5.2 records why — read directly:
*"Tested across a 2.5-fold tightening, the total form returned an
identical plan at every level — four different policies, one outcome —
while the annual form bound immediately."* A sweep whose sole purpose
is comparing `'annual'` against a mode that no longer exists as an
option cannot produce a reportable result; there is no live question
left to answer. Doubly so: the script also calls the ANNUAL
`build_model` (storage structurally zero, register L5) and predates
required pre-F1 scenario keys (register L10) — even setting aside the
retired mode, running this file today would raise before reaching a
solve, for unrelated reasons already documented elsewhere in this
project's inventory.

### `retired/09b_run_rel1_retirement_sensitivity.py`

**The question it asked:** does the flat 680 MW/yr gas-retirement
proxy (swept at 0/340/680/1360 MW/yr) change whether gas regime
remains the primary determinant of the REL-1 feasibility threshold?

**Verified: (b) absorbed — with one precision added.** The flat proxy
is confirmed superseded by the `gas_retire_vintage_{30,40,50}` arms
(register L8); the file's own note states the vintage arms showed the
flat proxy overstates system cost by ~1.28bn (5.9%) and storage by
~17.9%. TASK.md's cited corroboration, **checked against
`findings_frozen.md` directly**: F-20 records exactly
`gas_retire_none` 25.199bn → `gas_retire_vintage_30` 27.954bn
(central), **"Range 2.76 bn"** — confirmed verbatim, not just
plausible.

**The precision:** F-20 measures retirement-rate sensitivity on COST
and STORAGE across the vintage arms — it does not, itself, re-run
09b_retirement's exact original measurement (whether the REL-1
FEASIBILITY THRESHOLD ordering across gas regimes holds under
different retirement rates). The stronger form of "absorbed" is
structural, not just "a related measurement exists elsewhere":
`gas_retirement_case` is now one of `load_scenario()`'s 17 sentinel-
required keys, threaded through every REL runner including the now-
repaired `09_run_rel1_feasibility.py` itself. The EXACT original
question is therefore directly re-askable TODAY, natively, by running
`09_` with different `--gas-retirement-case` values — no dedicated
sweep script is needed because the capability is now a general CLI arm
on the runner the question was originally about, not merely because a
related finding (F-20) happens to touch the same variable.

**Neither script is live and wrongly retired.** Both verdicts are
CONFIRMED as TASK.md's own inventory reading suggested, with the
above corroboration and one precision added to the second.

---

## For the limitations register

**Draft entry (next free number: L29 — L28 is the current highest):**

> ## L29 — A runner being executable is not the same as its output
> being reportable: five declared defects that survived the repair to
> "runs without crashing"
>
> **Status:** RESOLVED (task_defect_closure).
>
> **What.** The three preceding tasks (`task_shared_fn_repair`,
> `task_runner_repair`, `task_wrong_output_repair`) made eleven runners
> executable — fixed missing arguments, wired CLI flags, closed a
> reconciliation gap. None of that work asked the second question: given
> a runner that now RUNS, does its output answer its research question
> CORRECTLY? Five declared defects (`merged_correction_plan_v2.md`'s
> D1, D17, D21, D23, D24 — none newly discovered here) show the answer
> was sometimes no, in ways a crash would never have revealed:
>
> - **D1** — `13_run_dem2_growth_gas_interaction.py`'s interaction
>   classifier used two different thresholds for the same boolean/text
>   pair, and discarded a real, economically meaningful sub-additive
>   result (demand growth REDUCING the gas scarcity penalty under
>   `upside`) as if it were noise.
> - **D23** — `run_reliability_sweep`, shared by REL-2 and REL-3, had no
>   `eps_levels` parameter at all; two callers' own correctly-resolved
>   eps grids were silently replaced by a default, with no error, for
>   as long as this code has existed.
> - **D24** — `compute_frontier_shift`'s "EaaS is necessary" and "EaaS
>   is strictly better" classifications could disagree on the same row
>   — the strongest possible REL-3 finding, `eaas_necessary=True`,
>   could print beside `eaas_strictly_better=False` in the same table.
>   A second, deeper cause (pandas silently coercing a missing-
>   threshold `None` to `NaN`, which `is None` never catches) meant this
>   was likely UNREACHABLE on real data, not merely mis-labelled when it
>   occurred.
> - **D17** — see below.
> - **D21** — the REL suite's eps axis had no single documented
>   rationale, assembled now in `docs/eps_axis_rationale.md` from
>   fragments that already existed but had never been joined; one of
>   those fragments (a plan quote about the λ anchor) itself needed a
>   correction that a different fragment (`23_`'s own docstring) had
>   already made.
>
> **D17, specifically — an assumption promoted to a measurement.** The
> REL-3 runner's own docstring carried, verbatim, an unresolved caveat
> from 2.6: *"at the default tariff the EaaS arm has no binding capital
> constraint... frontier_shift_eps therefore measures constrained-vs-
> unconstrained, not a financing effect. Resolve before reporting."*
> That caveat could not be resolved when written because `tariff_case`
> did not yet exist as a swept, sourced arm — there was nothing to
> measure against. `tariff_case` becoming a required, sourced
> `load_scenario()` parameter (`docs/tariff_arm.md`) is what converts
> this from a standing assumption every REL-3 reader had to carry
> silently into a per-run, per-row diagnostic: `eaas_bankability_n_
> binding_years` (reusing `03_run_ndc_eaas.py`'s own bankability-slack
> approach and F-29's own "binds" threshold) now states, for every EaaS
> row, whether the condition applies THIS TIME — and `eaas_not_a_
> financing_effect` flags it in the output rather than requiring a
> reader to remember a docstring caveat. The general pattern: a
> plumbing fix (making an arm real and required) can retroactively
> convert an assumed, unresolvable condition into a measured one
> elsewhere in the codebase, even when that was not the plumbing fix's
> own stated purpose.
>
> **Direction of bias.** None of the five defects biased a REPORTED
> NUMBER (no anchor moved, confirmed per-Part above) — they biased
> what a READER of already-correct numbers would have concluded from
> them: a contradictory classification, a silently-wrong sweep grid, a
> missing rationale for why axes differ, and an unmeasured constraint
> assumption presented as unresolved when it was, by the time of this
> task, resolvable.
>
> **What would resolve the rest.** `03_run_ndc_eaas.py`'s own copy of
> `bankability_slack_per_mw_by_year` still exists as an un-deduplicated
> twin of the copy this task added to `src/optimize_experiments.py`
> (see the closing list) — point one at the other. `run_str1_storage_
> parameter_sweep` and `run_bottleneck_sensitivity` (register L27/L28's
> own closing lists) still carry an unrelated, older calling-convention
> defect, unresolved. And this task's own AST-based "unused parameter"
> audit (Part 2) is a partial check by construction — it would not have
> found D23 itself; a further audit for the specific "parameter read
> once but not forwarded downstream" shape remains undone.

---

## Encountered but not asked about

- **`03_run_ndc_eaas.py`'s `bankability_slack_per_mw_by_year` is now
  duplicated**, not shared, with the copy this task added to `src/
  optimize_experiments.py` for Part 4 — `03_` is not in this task's
  edit whitelist, and a numbered script cannot be imported as a Python
  module (identifiers cannot start with a digit) regardless. Follow-on:
  point `03_`'s own copy at the new shared one once `03_` is back in
  scope for some future task.
- **`reliability_results_to_df` silently drops `solar_eaas_total_mw`**
  (and, before this task, would have dropped this task's own two new
  bankability fields had they not been added to it explicitly) — a
  pre-existing gap in the same function this task had to touch anyway.
  `compute_frontier_shift`'s `row_at.get("solar_eaas_total_mw")` has
  therefore likely always silently returned `None`, meaning `solar_
  eaas_mw_at_threshold` in `frontier_threshold.csv` may never have
  carried real data. Not fixed: unrelated to D17/D24's own scope, and
  TASK.md did not name it. Flagged because it sits one line away from
  code this task did change, not because it was hard to find.
- **The AST-based unused-parameter audit (Part 2) is a real but partial
  check** — it flags a parameter never referenced by name anywhere in
  its function body, but cannot detect a parameter that IS referenced
  (e.g. inside its own `if x is None: x = default()` resolution) and
  then never forwarded to a downstream call that needed it — which is
  the actual shape of D23. That shape can only be found by tracing call
  chains, as D23 itself was found. Stated so the audit's one clean
  result ("only `run_tariff_bankability_sweep`, and that's a deliberate
  stub") is not read as "no other D23-shaped bug exists in this file."
- **TASK.md misattributed the "8/3/1 binding years" figure to
  `findings_frozen.md`**; it is actually in two pre-O&M provisional
  documents (`findings_provisional_pre_om.md` F-13,
  `findings_provisional_pre_sweep.md` F-29). Corrected in Part 4's own
  section rather than silently repeated.
- **TASK.md's "four different eps axes across six scripts" is actually
  three axes** (reused across four of the six scripts). Corrected in
  `docs/eps_axis_rationale.md` §5 and restated in Part 5 above.
- No script, solver, or `smoke_all.sh` was run at any point in this
  task, per the whitelist. Part 3/4's verification used a synthetic,
  fabricated DataFrame run directly through `compute_frontier_shift` —
  pure Python/pandas logic, no scenario, no builder, no solver.

**Then STOP.** No commit, no staging. Zeus reviews `git diff` and
commits.
