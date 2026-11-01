# Phase 2.3 / 2.4 Implementation Report — Firm Capacity Constraint and PAR Parameter Layer

Scope: `src/optimize_model.py`, `src/optimize_model_sliced.py`, and (as required to
supply the new scenario keys) `src/scenarios.py`. Nothing was solved, no runner
script or the smoke suite was executed. Verification was limited to: `ast.parse`
on all three edited files, a plain Python import of the three modules, and
isolated arithmetic tests of the new `load_scenario()` validation logic (no
Pyomo model was built or solved — see "Verification performed" at the end).
Line numbers in STEP 0 refer to the codebase as it stood **before** this
task's edits (branch `phase-fin2-d17`, same commit as `docs/F0_audit.md`).

---

## STEP 0 — Read and report first

### 0.5 — Slice definition: hours per slice and demand share

Slice hours are fixed calendar arithmetic, `optimize_model_sliced.py:61-74`
(pre-edit):
```python
SEASON_DAYS = {"dry": 181, "wet": 184}      # dry = Nov-Apr, wet = May-Oct
PERIOD_HOURS = {"night": 7, "day": 13, "peak": 4}   # 22-05, 05-18, 18-22
SLICES = [(s, p) for s in ("dry", "wet") for p in ("night", "day", "peak")]

SOLAR_PERIODS = ("day",)   # solar generates in day slices ONLY


def slice_hours():
    """Hours per year represented by each slice. Sums to 8760."""
    h = {(s, p): SEASON_DAYS[s] * PERIOD_HOURS[p] for (s, p) in SLICES}
    total = sum(h.values())
    if total != 8760:
        raise ValueError(f"Slice hours sum to {total}, expected 8760")
    return h
```
giving: dry_night 1267, dry_day 2353, dry_peak 724, wet_night 1288,
wet_day 2392, wet_peak 736 (sums to 8760).

Demand SHARE per slice is not fixed calendar fact — it is derived from two
declared, unsourced shape assumptions, `optimize_model_sliced.py:84-98`
(pre-edit):
```python
DEFAULT_SEASON_DEMAND_FACTOR = {"dry": 1.08, "wet": 0.92}
...
DEFAULT_PERIOD_DEMAND_FACTOR = {"night": 0.65, "day": 1.05, "peak": 1.20}
```
combined by `_normalised_demand_weights()`, `optimize_model_sliced.py:128-144`
(pre-edit):
```python
def _normalised_demand_weights(scenario):
    h = slice_hours()
    season_f = scenario.get("season_demand_factor", DEFAULT_SEASON_DEMAND_FACTOR)
    period_f = scenario.get("period_demand_factor", DEFAULT_PERIOD_DEMAND_FACTOR)

    raw = {
        (s, p): h[(s, p)] * season_f[s] * period_f[p]
        for (s, p) in SLICES
    }
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}
```
`scenarios.py` sets its own literal copies of these two factor dicts inside
`load_scenario()` (pre-edit lines 856, 861 — `"season_demand_factor": {"dry":
1.08, "wet": 0.92}` and `"period_demand_factor": {"night": 0.65, "day": 1.05,
"peak": 1.20}`), independently duplicated from the `DEFAULT_*` dicts in
`optimize_model_sliced.py` rather than importing them — the two happen to
agree numerically but were two separate hard-coded copies before this task
(see STEP 1 below, where this duplication became load-bearing for the PAR
consistency check and was partly resolved).

Computed demand-energy shares under these defaults (verified by calling
`_normalised_demand_weights()` directly, no model build):
```
dry_night   0.106018   (1267 h)
dry_day     0.318053   (2353 h)
dry_peak    0.111843   (724 h)
wet_night   0.091808   (1288 h)
wet_day     0.275425   (2392 h)
wet_peak    0.096853   (736 h)
```

### 0.6 — Implied PAR

PAR = f_peak × 8760 / h_peak, f_peak = dry_peak + wet_peak demand-energy
share, h_peak = dry_peak + wet_peak hours.

Computed directly from the function above (pure arithmetic, no model build):
```
h_peak = 724 + 736 = 1460
f_peak = 0.111843 + 0.096853 = 0.208695652173913
PAR    = 0.208695652173913 * 8760 / 1460 = 1.2521739130434781
```
**PAR ≈ 1.2522** under the current default `season_demand_factor` /
`period_demand_factor`.

### 0.7 — Storage power: variable or derived?

Derived — an `Expression`, not a `Var` — from `storage_capacity_mwh` (itself
a cumulative sum of the `storage_add` decision variable) via a fixed duration
parameter.

`optimize_model.py:173, 301-305` (pre-edit):
```python
storage_duration_hours = scenario.get("storage_duration_hours", 4.0)  # kept for storage_power_mw only
...
m.storage_power_mw = pyo.Expression(
    T,
    rule=lambda m, t:
        m.storage_capacity_mwh[t] / storage_duration_hours
)
```
Identical in `optimize_model_sliced.py:270, 401-403` (pre-edit).

**Duration value: 4.0 hours**, and — critically — this was previously only an
*implicit* fallback (`scenario.get(..., 4.0)`); `scenarios.py` did not set
`"storage_duration_hours"` in the returned dict at all before this task (see
STEP 1: it is now set explicitly, at the same value, because
`capacity_credit['storage']` needs the identical number the builders use).

### 0.8 — Gas availability factor

`gas_availability_factor` is a required scenario key (`build_model()` raises
if absent), consumed in `gas_capacity_constraint`.

`optimize_model.py:350-365` (pre-edit):
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
Set in `scenarios.py:817` (pre-edit):
```python
"gas_availability_factor": 0.80,   # <-- set from fuel-agnostic availability
```
with the sourcing note at `scenarios.py:814-817` (pre-edit): "fuel-agnostic
mechanical/maintenance availability, reconstructed from NERC outage
cause-decomposition, or an OCGT/CCGT engineering benchmark derated for
Nigerian plant condition (~0.75-0.85). REPLACE." — i.e. **0.80 is itself a
declared, unsourced placeholder**, not a verified figure.

### 0.9 — Hydro representation: variable or exogenous, seasonal or not?

**Exogenous in both builders — never a decision variable.** No `hydro_add`
exists anywhere in either file.

**Annual builder**: a flat TWh series with only a year-on-year growth term,
no season structure. `optimize_model.py:327-340` (pre-edit):
```python
hydro_baseline_twh = scenario.get("hydro_baseline_twh", None)
if hydro_baseline_twh is None:
    raise ValueError(...)
hydro_growth = scenario.get("hydro_growth", 0.0)  # 0.0 = flat; set if new hydro commissions
hydro = {
    t: hydro_baseline_twh * ((1.0 + hydro_growth) ** t)
    for t in T
}
```
`hydro[t]` is a plain Python dict of floats entering `m.balance` as a
constant — not seasonal, not sliced, not dispatchable.

**Sliced builder**: still exogenous (no capacity decision variable), but
genuinely seasonal and dispatchable within a season-level energy budget and a
per-slice power limit, requiring an *additional* required scenario key,
`hydro_capacity_mw`, that the annual builder does not use.
`optimize_model_sliced.py:439-446` (pre-edit):
```python
hydro_capacity_mw = scenario.get("hydro_capacity_mw", None)
if hydro_capacity_mw is None:
    raise ValueError(
        "hydro_capacity_mw is unset. Required for the sliced model: hydro "
        "is dispatchable within a seasonal energy budget and needs a power "
        "limit. Nigerian installed hydro is ~2,678 MW (Kainji, Jebba, "
        "Shiroro, Zungeru, Dadin-Kowa)."
    )
```
and `optimize_model_sliced.py:496-521` (pre-edit): `hydro_season_budget`
(exact seasonal energy split), `hydro_power_limit` (per-slice MW cap), and
`hydro_min_flow` (a minimum-flow floor per slice, preventing the LP from
concentrating all hydro into peak slices).

Set in `scenarios.py:819-829` (pre-edit): `hydro_baseline_twh = 11.47`,
`hydro_growth = 0.0`, `hydro_capacity_mw = 2678`.

### 0.10 — Planning reserve margin or system peak MW forecast in scenarios.py or NIRP data?

**Absent as anything the LP enforces.** Grep for
`reserve_margin|peak_mw|system_peak|peak_multiple|capacity_credit` across
`src/scenarios.py` returns **no matches**. Grep for `NIRP` across
`src/scenarios.py` (10 hits) shows NIRP is used for demand growth and solar
deployment-capability trajectories only — no peak-MW or reserve-margin figure
appears anywhere in the NIRP-derived content. `data/**/*NIRP*` and
`data/**/*peak*` glob to nothing.

**However, two things do exist and must not be confused with what this task
adds:**

1. `scenarios.py:918` (pre-edit) already sets a scalar
   `"peak_demand_multiple": 2.5` in the returned scenario dict, commented
   "used by peak-adequacy diagnostic." This is functionally a PAR value, but
   it is **diagnostic-only**: it feeds nothing in either builder's LP. It is
   read exclusively by a post-solve heuristic in `src/optimize_experiments.py`
   (outside this task's audited/edited files), `optimize_experiments.py:
   2051-2070` (pre-edit):
   ```python
   # ---- Peak adequacy heuristic (must be computed before return dict)
   # Peak-to-average demand ratio. Nigerian grid load factor ≈ 0.55
   # (Nigerian System Operator statistics, 2022-2023). 1/0.55 ≈ 1.82.
   # A higher value (2.0-2.5) would reflect more peaky demand patterns
   # characteristic of residential-dominated systems.
   peak_demand_multiple = scenario.get("peak_demand_multiple", 1.82)
   peak_adequacy_by_year = {}
   for t, y in enumerate(years):
       avg_demand_twh = demand[t]
       avg_mw = avg_demand_twh * 1e6 / 8760
       peak_mw = avg_mw * peak_demand_multiple
       firm_capacity_mw = (
           pyo.value(m.gas_capacity_mw[t])
           + pyo.value(m.storage_power_mw[t])
       )
       peak_adequacy_by_year[int(y)] = {
           "peak_demand_mw": round(peak_mw, 1),
           "firm_capacity_mw": round(firm_capacity_mw, 1),
           "reserve_margin": round((firm_capacity_mw - peak_mw) / peak_mw, 4),
       }
   ```
   Note the function's own fallback default (1.82) is dead code in practice:
   `scenarios.py` already sets a real `2.5` in every scenario dict it
   produces, so `scenario.get("peak_demand_multiple", 1.82)` never falls
   through to 1.82. Note also this heuristic credits only gas + storage power
   as "firm capacity" (no solar, no hydro at all), and was never reconciled
   against `season_demand_factor`/`period_demand_factor` the way the new
   `peak_multiple_case` now is (STEP 1 below).
2. `src/EaaS.py` was noted in `docs/F0_audit.md` (C9) to contain a capital
   recovery factor implementation — unrelated to peak/reserve margin, flagged
   here only so it is not mistaken for relevant to this question.

Neither of these is a planning reserve margin or an enforced peak-MW
forecast. **This task's STEP 1/2 introduce the first of each into the
codebase.**

### STOP condition (STEP 0 closing)

None of the six answers above blocks the STEP 1/2 design. Two items required
judgement calls, both resolved and documented, not left implicit:
- 0.6's PAR is a property of already-declared, unsourced shape assumptions
  (`season_demand_factor`/`period_demand_factor`) — computing it does not
  require new sourcing, only exposes what the existing defaults already imply.
- 0.10 confirms no reserve margin or peak forecast exists to conflict with;
  the pre-existing `peak_demand_multiple` diagnostic is a different,
  unenforced quantity and was left untouched (documented, not deleted or
  merged, since touching it was out of this task's stated scope).

Proceeding to STEP 1 and STEP 2.

---

## STEP 1 — 2.4 parameter layer (`src/scenarios.py`)

### Required key discipline

`peak_multiple_case` and `reserve_margin` were added as **required, no-default**
parameters of `load_scenario()`, using the identical sentinel-object mechanism
`src/io.py`'s `load_econ()` uses for `genset_case` (`_GENSET_REQUIRED`):
```python
_PEAK_MULTIPLE_REQUIRED = object()
_RESERVE_MARGIN_REQUIRED = object()

def load_scenario(
    ...,
    peak_multiple_case=_PEAK_MULTIPLE_REQUIRED,
    reserve_margin=_RESERVE_MARGIN_REQUIRED,
    ...,
):
    ...
    if peak_multiple_case is _PEAK_MULTIPLE_REQUIRED:
        raise TypeError(...)
    if reserve_margin is _RESERVE_MARGIN_REQUIRED:
        raise TypeError(...)
```
Verified directly (no model build): calling `load_scenario()` with neither
argument raises `TypeError`; calling it with a valid arm name but a still-`None`
PAR value raises `ValueError` naming the `SOURCE: [PENDING]` gap.

### PAR arms

Named constants at the top of `scenarios.py`, immediately after
`blended_finance_scenarios()`:
```python
PAR_OBSERVED = None   # SOURCE: [PENDING]
PAR_CENTRAL  = None   # SOURCE: [PENDING]
PAR_HIGH     = None   # SOURCE: [PENDING]

def peak_multiple_scenarios():
    return {
        "par_observed": PAR_OBSERVED,
        "par_central":  PAR_CENTRAL,
        "par_high":     PAR_HIGH,
    }
```
`peak_multiple_case` must be one of `'par_observed'` / `'par_central'` /
`'par_high'`. All three arm values are `None` pending the author's sourcing;
using any of them before that raises explicitly (verified) rather than
silently building on a placeholder number.

### Consistency route taken (Critical instruction)

The task offered two routes: **re-derive** the peak-slice demand share from
`peak_multiple_case`, or **raise** if the two disagree by more than 1%.

**Route taken: raise.** `load_scenario()` computes the PAR implied by the
scenario's `season_demand_factor`/`period_demand_factor` — using the model's
own `_normalised_demand_weights()` and `slice_hours()` (imported from
`optimize_model_sliced.py`, not re-implemented, so the check tests the
literal computation the sliced model performs) — and compares it to the
`peak_multiple_case` arm's PAR value. If they disagree by more than 1%, it
raises `ValueError` before the scenario dict is ever built.

**Why raise, not re-derive:**
1. **Under-determination.** The slice demand shape has five free numbers
   (`season_demand_factor['dry'/'wet']`, `period_demand_factor['night'/
   'day'/'peak']`), not one. A single target PAR does not uniquely determine
   how to adjust them — re-deriving would require me to pick, unprompted,
   which of the five absorbs the correction (most naturally
   `period_demand_factor['peak']`, solvable in closed form, but that is a
   modelling choice the task did not make and I should not make silently on
   its behalf).
2. **Both arm values are currently `None` (`SOURCE: [PENDING]`).** There is
   no real target PAR to re-derive against yet — the author said "I will
   supply them." A raise-on-disagreement check is exactly the right shape of
   guard for that state: it does nothing until real numbers exist, then
   enforces consistency the moment they do, without requiring me to guess a
   re-derivation formula that may not match the source the author eventually
   supplies for the PAR arms.
3. **Matches the codebase's dominant discipline.** Every other "two
   representations of one fact" pattern in this codebase (`gas_eta` /
   `EF_TCO2_PER_MWH_TH` deriving EF_e once, never storing it twice;
   `gas_availability_factor` now feeding `capacity_credit['gas']` directly
   rather than a second hard-coded number) resolves duplication by refusing
   to let a second, independently-set copy exist — a hard failure on
   disagreement, not a silent reconciliation.

**Known limitation, stated plainly:** the check runs inside `load_scenario()`
at scenario-construction time. If a script constructs a scenario via
`load_scenario()` and then *mutates* `scenario["season_demand_factor"]` or
`scenario["period_demand_factor"]` afterward (bypassing `load_scenario()`),
the consistency check does not re-run and the scenario actually built by
`build_model_sliced()` can silently diverge from the PAR that was checked.
No runtime re-check was added inside `build_model_sliced()` itself — that
was judged out of this task's stated scope (STEP 1 is explicitly "the
parameter layer"; STEP 2 is the firm-capacity constraint, not a second PAR
check). Flagged here rather than left for the reader to discover.

### Derivations (all in `load_scenario()`, before the scenario dict is built)

`peak_mw[t]` — a list aligned to the model's year index (same convention as
`solar_cumulative_cap_mw`), `peak_mw[t] = avg_mw[t] * PAR`:
```python
_demand_proj = project_baseline_demand(
    base_demand=_base_demand_twh,
    growth_rate=_demand_growth,
    start_year=years[0],
    end_year=years[-1],
)["demand"]
peak_mw = [
    float(_demand_proj[t]) * 1e6 / 8760.0 * par_value
    for t in range(len(years))
]
```

`capacity_credit` — a flat dict, `{tech: fraction}`, all four SOURCE:
[PENDING]:
```python
_hydro_share = _hydro_season_energy_share({
    "hydro_season_factor": {"dry": 0.90, "wet": 1.10},
})
_dry_hours_per_year = SEASON_DAYS["dry"] * 24
_hydro_dry_avg_mw = (
    _hydro_baseline_twh * _hydro_share["dry"] * 1e6 / _dry_hours_per_year
)
capacity_credit = {
    "solar": 0.0,                                                  # SOURCE: [PENDING]
    "gas": _gas_availability_factor,                                # SOURCE: [PENDING] (= gas_availability_factor)
    "storage": min(1.0, _storage_duration_hours / PERIOD_HOURS["peak"]),  # SOURCE: [PENDING]
    "hydro": _hydro_dry_avg_mw / _hydro_capacity_mw,                # SOURCE: [PENDING]
}
```
- `solar = 0.0`: the literal default the task specified — no credit at
  system peak (the peak slice carries zero solar yield by construction —
  `SOLAR_PERIODS = ("day",)`).
- `gas`: **the same number**, not a second hard-coded one — set to the local
  variable `_gas_availability_factor` that also feeds
  `scenario["gas_availability_factor"]`.
- `storage`: `min(1, storage_duration_hours / PERIOD_HOURS["peak"])` = `min(1,
  4.0/4) = 1.0` under current defaults — a battery whose discharge duration
  covers a single peak slice's length is fully creditable.
- `hydro`: average dry-season hydro output (MW) as a fraction of installed
  hydro capacity, using the sliced model's own `_hydro_season_energy_share()`
  helper — computed value **0.4397** (verified by direct call, no model
  build), i.e. ~44% of installed hydro MW is credited as firm.

Verified end-to-end (no model build) by monkeypatching `PAR_CENTRAL` to the
value implied by the current default slice shape (1.2521757) and calling
`load_scenario(peak_multiple_case='par_central', reserve_margin=0.15)`:
```
OK built scenario with consistent PAR.
peak_mw[0] = 5301.73
capacity_credit = {'solar': 0.0, 'gas': 0.8, 'storage': 1.0, 'hydro': 0.4396777015505073}
```
and confirmed the disagreement guard fires when `PAR_CENTRAL` is set to an
inconsistent value (2.0):
```
ValueError: peak_multiple_case='par_central' implies PAR=2.0000, but
season_demand_factor/period_demand_factor implies PAR=1.2522 -- a 59.8%
disagreement...
```

### Necessary side-effects in `scenarios.py` (beyond the letter of STEP 1, justified)

- `"storage_duration_hours"` is now an **explicit** scenario dict key (value
  4.0, unchanged from the builders' previous implicit fallback). Necessary
  because `capacity_credit['storage']` must use the exact number the
  builders will use, not an independent second copy of `4.0` that could
  drift.
- `"gas_availability_factor"`, `"hydro_baseline_twh"`, `"hydro_capacity_mw"`,
  `"season_demand_factor"`, `"period_demand_factor"` in the returned dict now
  reference shared local variables instead of repeating literals, so the
  same numbers feed both the dict and the new derivations above them.
  Values are **unchanged** (0.80, 11.47, 2678, `{"dry":1.08,"wet":0.92}`,
  `{"night":0.65,"day":1.05,"peak":1.20}`) — this is a refactor for
  single-sourcing, not a value change.
- `"peak_multiple"` was added to the `"labels"` dict for reporting
  consistency with the other case labels already there.

### Full diff, `src/scenarios.py`

```diff
diff --git a/src/scenarios.py b/src/scenarios.py
index 0bc8e18..29b24cb 100644
--- a/src/scenarios.py
+++ b/src/scenarios.py
@@ -54,6 +54,21 @@ from __future__ import annotations
 
 import numpy as np
 
+# 2.6, F0 Step 2.4: PAR and the peak-slice demand share are the SAME
+# assumption. Importing the model's own slice-hours/weighting functions (and
+# the hydro seasonal-share helper, reused for capacity_credit['hydro'] in
+# Step 2.3) means the consistency check below tests the LITERAL computation
+# build_model_sliced() performs, not a shadow copy that could drift out of
+# sync with it.
+from src.demand import project_baseline_demand
+from src.optimize_model_sliced import (
+    SEASON_DAYS,
+    PERIOD_HOURS,
+    slice_hours,
+    _normalised_demand_weights,
+    _hydro_season_energy_share,
+)
+
 
 # ============================================================
 # BLENDED CONCESSIONAL FINANCE (single source of truth)
@@ -92,6 +107,42 @@ def blended_finance_scenarios():
     }
 
 
+# ============================================================
+# PEAK DEMAND MULTIPLE (PAR) -- 2.6, F0 Step 2.4
+# ============================================================
+# Peak-to-average demand ratio (PAR). Feeds peak_mw[t] = avg_mw[t] * PAR,
+# which the firm-capacity constraint (2.6, F0 Step 2.3) tests cumulative
+# credited capacity against.
+#
+# THIS IS THE SAME ASSUMPTION as the peak-slice demand share used to build
+# the 6-slice load shape in optimize_model_sliced.py (season_demand_factor x
+# period_demand_factor). load_scenario() computes the PAR implied by that
+# slice shape and RAISES if it disagrees with the arm value here by more
+# than 1% -- see the consistency check in load_scenario(). Do not set
+# peak_multiple_case and the slice-shape factors independently.
+#
+# SOURCE: [PENDING] for all three arms -- author to supply from a named
+# source (NERC/TCN system peak report, or equivalent). Do NOT ship these as
+# None; a None arm will raise the moment it is used, by design (mirrors the
+# genset_case discipline in load_econ() -- see src/io.py).
+PAR_OBSERVED = None   # SOURCE: [PENDING]
+PAR_CENTRAL  = None   # SOURCE: [PENDING]
+PAR_HIGH     = None   # SOURCE: [PENDING]
+
+
+def peak_multiple_scenarios():
+    """
+    Peak-to-average demand ratio (PAR) arms. SOURCE: [PENDING] for all three
+    -- see the module-level PAR_* constants above, which the author will
+    supply. Keys are what peak_multiple_case accepts in load_scenario().
+    """
+    return {
+        "par_observed": PAR_OBSERVED,
+        "par_central":  PAR_CENTRAL,
+        "par_high":     PAR_HIGH,
+    }
+
+
 # ============================================================
 # PLANNING HORIZON
 # ============================================================
@@ -708,6 +759,16 @@ def _solar_cumulative_cap(solar_build_case: str, years) -> list[float] | None:
 # SCENARIO CONSTRUCTOR
 # ============================================================
 
+# Sentinels distinguishing "not passed" from a real value -- same mechanism
+# as _GENSET_REQUIRED in src/io.py's load_econ(). A real default (or None)
+# cannot be used here: silently omitting these two would build a firm-
+# capacity constraint against an unstated peak/reserve assumption with no
+# error, warning, or visible signature in the output -- the exact failure
+# mode genset_case was introduced to close off (see src/io.py docstring).
+_PEAK_MULTIPLE_REQUIRED = object()
+_RESERVE_MARGIN_REQUIRED = object()
+
+
 def load_scenario(
     demand_level_case: str = "served",
     demand_case: str = "organic_central",
@@ -718,6 +779,8 @@ def load_scenario(
     solar_tariff_case: str = "baseline",
     carbon_case: str = "no_policy",
     blend_case: str = "blended_central",
+    peak_multiple_case=_PEAK_MULTIPLE_REQUIRED,
+    reserve_margin=_RESERVE_MARGIN_REQUIRED,
     start_year: int = MODEL_START_YEAR,
     end_year: int = MODEL_END_YEAR,
 ) -> dict:
@@ -741,6 +804,15 @@ def load_scenario(
     carbon_case       : key of carbon_policy_scenarios()
     blend_case        : key of blended_finance_scenarios() (default central).
                         Sets the EaaS capital structure -> blended private rate.
+    peak_multiple_case : key of peak_multiple_scenarios() -- REQUIRED, no
+                        default (2.6, F0 Step 2.4). One of 'par_observed' /
+                        'par_central' / 'par_high'. Drives peak_mw[t]; must
+                        agree with season_demand_factor/period_demand_factor
+                        to within 1% (see the consistency check below) or
+                        load_scenario() raises.
+    reserve_margin    : float -- REQUIRED, no default (2.6, F0 Step 2.3).
+                        Fractional reserve margin applied on top of peak_mw[t]
+                        in the firm-capacity constraint.
     start_year, end_year : planning horizon bounds.
     """
 
@@ -773,11 +845,141 @@ def load_scenario(
     if blend_case not in blended_finance_scenarios():
         raise ValueError(f"Unknown blend_case: {blend_case}")
 
+    # ---- 2.6, F0 Step 2.4: peak_multiple_case (required, no default --
+    # mirrors genset_case discipline in load_econ()).
+    if peak_multiple_case is _PEAK_MULTIPLE_REQUIRED:
+        raise TypeError(
+            "load_scenario() requires peak_multiple_case (2.6, F0 Step 2.4). "
+            "Pass one of 'par_observed' / 'par_central' / 'par_high' (see "
+            "peak_multiple_scenarios()). There is no default: omitting it "
+            "would build a firm-capacity constraint against an unstated "
+            "peak-demand assumption with no error, warning, or visible "
+            "signature in the output."
+        )
+    if peak_multiple_case not in peak_multiple_scenarios():
+        raise ValueError(f"Unknown peak_multiple_case: {peak_multiple_case}")
+    par_value = peak_multiple_scenarios()[peak_multiple_case]
+    if par_value is None:
+        raise ValueError(
+            f"peak_multiple_scenarios()['{peak_multiple_case}'] is None -- "
+            "SOURCE: [PENDING]. Supply a sourced PAR value for this arm "
+            "(module-level PAR_* constants, top of scenarios.py) before "
+            "building a scenario with a firm-capacity constraint."
+        )
+
+    # ---- 2.6, F0 Step 2.3: reserve_margin (required, no default).
+    if reserve_margin is _RESERVE_MARGIN_REQUIRED:
+        raise TypeError(
+            "load_scenario() requires reserve_margin (2.6, F0 Step 2.3). "
+            "Pass a fractional reserve margin (e.g. 0.15). There is no "
+            "default: omitting it would build a firm-capacity constraint "
+            "against an unstated reserve assumption with no error, warning, "
+            "or visible signature in the output."
+        )
+    if reserve_margin < 0:
+        raise ValueError(f"reserve_margin must be >= 0, got {reserve_margin}")
+
     years = planning_horizon(start_year, end_year)
 
     # ---- Blended finance: single source of truth (no hard-coded duplicates)
     conc_share, r_conc, r_comm = blended_finance_scenarios()[blend_case]
 
+    # ============================================================
+    # Firm-capacity / PAR inputs (2.6, F0 Step 2.3/2.4)
+    # ============================================================
+    # Extracted to local variables BEFORE the scenario dict is built so the
+    # SAME numbers feed both the dict entries below and the derivations here
+    # -- avoiding a second, driftable copy of gas_availability_factor,
+    # hydro_baseline_twh, hydro_capacity_mw, season_demand_factor, and
+    # period_demand_factor.
+    _base_demand_twh = demand_level_scenarios()[demand_level_case]
+    _demand_growth = demand_growth_scenarios()[demand_case]
+    _gas_availability_factor = 0.80   # SOURCE: see dict entry below
+    _hydro_baseline_twh = 11.47       # SOURCE: see dict entry below
+    _hydro_capacity_mw = 2678         # SOURCE: see dict entry below
+    # Storage discharge duration (hours). Previously an IMPLICIT default
+    # (scenario.get("storage_duration_hours", 4.0)) inside both builders;
+    # made explicit here because capacity_credit['storage'] below needs the
+    # same number the builders will actually use, not a second copy of 4.0
+    # that could silently drift from the builders' fallback.
+    _storage_duration_hours = 4.0
+    _season_demand_factor = {"dry": 1.08, "wet": 0.92}
+    _period_demand_factor = {"night": 0.65, "day": 1.05, "peak": 1.20}
+
+    # PAR and the peak-slice demand share are the SAME assumption (2.6, F0
+    # Step 2.4). Compute the PAR implied by the slice-shape factors above
+    # using the model's own weighting function (_normalised_demand_weights,
+    # imported from optimize_model_sliced.py) so this check tests the
+    # literal computation the sliced model performs. Refuse to proceed if it
+    # disagrees with the peak_multiple_case arm by more than 1% -- do not
+    # let peak_multiple_case and the slice-shape factors be set
+    # independently.
+    _implied_weights = _normalised_demand_weights({
+        "season_demand_factor": _season_demand_factor,
+        "period_demand_factor": _period_demand_factor,
+    })
+    _slice_h = slice_hours()
+    _f_peak = sum(_implied_weights[(s, "peak")] for s in ("dry", "wet"))
+    _h_peak = sum(_slice_h[(s, "peak")] for s in ("dry", "wet"))
+    _implied_par = _f_peak * 8760 / _h_peak
+    _par_disagreement = abs(_implied_par - par_value) / par_value
+    if _par_disagreement > 0.01:
+        raise ValueError(
+            f"peak_multiple_case='{peak_multiple_case}' implies PAR="
+            f"{par_value:.4f}, but season_demand_factor/period_demand_factor "
+            f"implies PAR={_implied_par:.4f} -- a {_par_disagreement:.1%} "
+            "disagreement (2.6, F0 Step 2.4). PAR and the peak-slice demand "
+            "share are the SAME assumption: set peak_multiple_case and the "
+            "slice-shape factors consistently, or adjust one to match the "
+            "other. Do not set them independently."
+        )
+
+    # peak_mw[t] = avg_mw[t] * PAR, aligned to the model's year index (same
+    # convention as solar_cumulative_cap_mw / _solar_cumulative_cap above).
+    _demand_proj = project_baseline_demand(
+        base_demand=_base_demand_twh,
+        growth_rate=_demand_growth,
+        start_year=years[0],
+        end_year=years[-1],
+    )["demand"]
+    peak_mw = [
+        float(_demand_proj[t]) * 1e6 / 8760.0 * par_value
+        for t in range(len(years))
+    ]
+
+    # Capacity credits (2.6, F0 Step 2.3). ALL FOUR are declared assumptions
+    # pending sourcing -- SOURCE: [PENDING].
+    #   solar   : 0.0 -- no credit at system peak (evening peak, no solar
+    #             yield in the peak slice; see SOLAR_PERIODS in
+    #             optimize_model_sliced.py). SOURCE: [PENDING] as a judgement,
+    #             not a citation.
+    #   gas     : the SAME fuel-agnostic availability factor as
+    #             gas_capacity_constraint (Step 0.4) -- one source of truth,
+    #             not a second hard-coded number. SOURCE: [PENDING].
+    #   storage : f(duration / peak-slice hours) = min(1, storage duration
+    #             hours / length of a single peak slice). A battery whose
+    #             discharge duration covers the peak window is fully
+    #             creditable; a shorter one is credited proportionally.
+    #             SOURCE: [PENDING] as a methodology choice.
+    #   hydro   : seasonal-dry -- average dry-season hydro output (the
+    #             tighter of the two seasons) as a fraction of installed
+    #             hydro capacity, using the same seasonal-share helper the
+    #             sliced model itself uses. SOURCE: [PENDING] (inherits the
+    #             pending status of hydro_season_factor/hydro_baseline_twh).
+    _hydro_share = _hydro_season_energy_share({
+        "hydro_season_factor": {"dry": 0.90, "wet": 1.10},
+    })
+    _dry_hours_per_year = SEASON_DAYS["dry"] * 24
+    _hydro_dry_avg_mw = (
+        _hydro_baseline_twh * _hydro_share["dry"] * 1e6 / _dry_hours_per_year
+    )
+    capacity_credit = {
+        "solar": 0.0,                                                  # SOURCE: [PENDING]
+        "gas": _gas_availability_factor,                                # SOURCE: [PENDING] (= gas_availability_factor)
+        "storage": min(1.0, _storage_duration_hours / PERIOD_HOURS["peak"]),  # SOURCE: [PENDING]
+        "hydro": _hydro_dry_avg_mw / _hydro_capacity_mw,                # SOURCE: [PENDING]
+    }
+
     scenario = {
         # ---- Temporal
         "years": years,
@@ -814,19 +1016,19 @@ def load_scenario(
         # Source: fuel-agnostic mechanical/maintenance availability, reconstructed
         # from NERC outage cause-decomposition, or an OCGT/CCGT engineering
         # benchmark derated for Nigerian plant condition (~0.75-0.85). REPLACE.
-        "gas_availability_factor": 0.80,   # <-- set from fuel-agnostic availability
+        "gas_availability_factor": _gas_availability_factor,   # <-- set from fuel-agnostic availability
 
         # ---- Hydro (exogenous must-run, TWh_e/year, already electrical)
         # -------------------------------------------------------------------
         # Measured 2024: 11.47 TWh (NERC — 30.92% of 37,093.70 GWh).
         # 2025 ~12.8 TWh as Zungeru (700 MW) ramps.
         # TODO Phase 2: replace with explicit committed-vs-NDC-aligned trajectory.
-        "hydro_baseline_twh": 11.47,        "hydro_growth": 0.0,          # flat unless modelling new hydro (e.g. Zungeru ramp)
+        "hydro_baseline_twh": _hydro_baseline_twh,        "hydro_growth": 0.0,          # flat unless modelling new hydro (e.g. Zungeru ramp)
         # Installed hydro capacity (MW) — required by the sliced model, which
         # dispatches hydro within a seasonal energy budget subject to a power
         # limit. Kainji + Jebba + Shiroro + Zungeru + Dadin-Kowa.
         # [SOURCE: NERC — sum of installed hydro plant ratings]
-        "hydro_capacity_mw": 2678,
+        "hydro_capacity_mw": _hydro_capacity_mw,
 
         # ---- Land policy
         "land_available_km2": land_scenarios()[land_case],
@@ -853,12 +1055,16 @@ def load_scenario(
         # Relative average power by season (ratio only; normalised internally).
         # Dry (Nov-Apr) demand higher: heat, peak Jan-Apr per cited load research.
         # [SOURCE NEEDED: quantitative seasonal demand ratio]
-        "season_demand_factor": {"dry": 1.08, "wet": 0.92},
+        # NOTE: this is the SAME assumption as peak_multiple_case (2.6, F0 Step
+        # 2.4) -- see the consistency check above. Do not edit independently.
+        "season_demand_factor": _season_demand_factor,
         # Relative average power by period. Flatter than residential: national
         # aggregate load has C&I daytime offset; night is the genuine trough.
         # Observed grid data CANNOT settle this (shedding truncates the peak).
         # [SOURCE NEEDED: quantitative intra-day load shape]
-        "period_demand_factor": {"night": 0.65, "day": 1.05, "peak": 1.20},
+        # NOTE: this is the SAME assumption as peak_multiple_case (2.6, F0 Step
+        # 2.4) -- see the consistency check above. Do not edit independently.
+        "period_demand_factor": _period_demand_factor,
         # Relative solar yield by season, day slices only. Dry = clearer skies
         # (offset by harmattan dust); wet = monsoon cloud.
         # [SOURCE NEEDED: Global Solar Atlas MONTHLY PVOUT, named sites]
@@ -882,6 +1088,10 @@ def load_scenario(
         # a bare 0.20 default. Left commented as an explicit record of the choice.
         # "storage_solar_surplus_frac": 0.20,
         "storage_round_trip_eff": 0.90,
+        # Was an IMPLICIT builder-side default (scenario.get(..., 4.0)) before
+        # 2.6, F0 Step 2.3; now explicit here because capacity_credit['storage']
+        # is derived from this exact number (see above).
+        "storage_duration_hours": _storage_duration_hours,
 
         # ---- Financing regime
         "financing_regime": "traditional",  # or "eaas"
@@ -915,8 +1125,26 @@ def load_scenario(
         # ---- Diagnostic-only inputs (consumed by specific runners, not the LP)
         # Confirm these are read where intended; they do not affect the core solve.
         "disco_collection_rate": 1.0,   # used by FIN-3 runner
+        # NOTE: this is a PRE-EXISTING, DIFFERENT quantity from
+        # peak_multiple_case/peak_mw below. It feeds only the post-solve
+        # heuristic in optimize_experiments.py (peak_adequacy_by_year), not
+        # any LP constraint, and was never reconciled against
+        # season_demand_factor/period_demand_factor the way peak_multiple_case
+        # now is. Left as-is (2.6, F0 Step 2.3/2.4 did not touch it); flagged
+        # here so the two are not confused.
         "peak_demand_multiple": 2.5,    # used by peak-adequacy diagnostic
 
+        # ============================================================
+        # Firm-capacity constraint inputs (2.6, F0 Step 2.3/2.4)
+        # ============================================================
+        # peak_mw[t] = avg_mw[t] * PAR, PAR from peak_multiple_case; checked
+        # for consistency against season_demand_factor/period_demand_factor
+        # above (raises before reaching here if they disagree by > 1%).
+        "peak_multiple_case": peak_multiple_case,
+        "peak_mw": peak_mw,
+        "reserve_margin": reserve_margin,
+        "capacity_credit": capacity_credit,
+
         # ---- Carbon policy (spreads carbon_active / carbon_price)
         **carbon_policy_scenarios()[carbon_case],
 
@@ -929,6 +1157,7 @@ def load_scenario(
             "land": land_case,
             "solar_build": solar_build_case,
             "blend": blend_case,
+            "peak_multiple": peak_multiple_case,
         },
     }
 
```

---

## STEP 2 — 2.3 firm capacity constraint

Added identically to both builders:
```
Sigma_tech capacity_credit[tech] x cumulative_capacity[tech,t]
  >= peak_mw[t] * (1 + reserve_margin)
```
- `capacity_credit["solar"] * m.solar_capacity_mw[t]`
- `capacity_credit["gas"] * m.gas_capacity_mw[t]`
- `capacity_credit["storage"] * m.storage_power_mw[t]` — **power** (MW), not
  `m.storage_capacity_mwh[t]` (MWh); the constraint is dimensioned in MW
  throughout, and storage's MWh energy rating is not a capacity figure.
- `capacity_credit["hydro"] * hydro_capacity_mw` — a scalar, since hydro has
  no capacity decision variable in either builder (STEP 0.9).

Constraint named `m.firm_capacity_constraint`, per the task's naming
instruction. A `firm_capacity_diagnostic(m, years)` function was added to
each file (module-level, alongside the existing `leakage_diagnostic()` in
`optimize_model.py` / `annual_totals()`+`slice_detail()` in
`optimize_model_sliced.py`), returning per-year slack (MW) and dual (via
`m.dual`, already an active `Suffix` in both builders) — not invoked here,
since no model was built or solved (see "Verification performed" below).

### Required-key discipline in the builders (not just in `load_scenario()`)

`build_model()`/`build_model_sliced()` accept any scenario dict a caller
constructs directly — they do not call `load_scenario()` themselves. STEP 1's
required-argument discipline lives entirely in `load_scenario()` and would
not protect a caller who builds a scenario dict by hand. `peak_mw`,
`reserve_margin`, and `capacity_credit` are therefore **also** required
directly inside both builders, raising `ValueError` if absent — the same
discipline already used for `hydro_baseline_twh`, `gas_availability_factor`,
and `asset_lifetimes` in these two files.

One consequence worth flagging explicitly: **`optimize_model.py` (the annual
builder) did not previously read `hydro_capacity_mw` at all** — it only used
the TWh energy series (STEP 0.9). The firm capacity constraint needs hydro's
installed-MW rating to credit it, so `hydro_capacity_mw` is now also required
in `optimize_model.py`, reusing the same scenario key the sliced builder has
always required.

### Full diff, `src/optimize_model.py`

```diff
diff --git a/src/optimize_model.py b/src/optimize_model.py
index 052d69d..0765c7d 100644
--- a/src/optimize_model.py
+++ b/src/optimize_model.py
@@ -238,7 +238,60 @@ def build_model(
     if not (0.0 < storage_round_trip_eff <= 1.0):
         raise ValueError("storage_round_trip_eff must be in (0, 1]")
 
-    
+    # ------------------------------------------------------------
+    # Firm capacity constraint parameters (2.6, F0 Step 2.3/2.4)
+    # ------------------------------------------------------------
+    # peak_mw, reserve_margin, and capacity_credit are all produced by
+    # load_scenario() (peak_multiple_case / reserve_margin, required, no
+    # default -- see scenarios.py). Required here too, independent of
+    # load_scenario(), because build_model() accepts any scenario dict a
+    # caller constructs directly: same discipline as hydro_baseline_twh,
+    # gas_availability_factor, and asset_lifetimes above.
+    peak_mw = scenario.get("peak_mw", None)
+    if peak_mw is None:
+        raise ValueError(
+            "scenario['peak_mw'] is required (2.6, F0 Step 2.3). Derived by "
+            "load_scenario() from peak_multiple_case (Step 2.4); build_model() "
+            "will not silently assume a peak-demand trajectory."
+        )
+    if len(peak_mw) != len(T):
+        raise ValueError(
+            f"scenario['peak_mw'] has {len(peak_mw)} entries for {len(T)} "
+            "model years. Refusing to build with a misaligned peak trajectory."
+        )
+    reserve_margin = scenario.get("reserve_margin", None)
+    if reserve_margin is None:
+        raise ValueError(
+            "scenario['reserve_margin'] is required (2.6, F0 Step 2.3). No "
+            "default: an omitted reserve margin would silently build an "
+            "unenforced firm-capacity constraint."
+        )
+    capacity_credit = scenario.get("capacity_credit", None)
+    if capacity_credit is None:
+        raise ValueError(
+            "scenario['capacity_credit'] is required (2.6, F0 Step 2.3). "
+            "Provide {'solar':..., 'gas':..., 'storage':..., 'hydro':...} "
+            "(see scenarios.py)."
+        )
+    for _tech in ("solar", "gas", "storage", "hydro"):
+        if _tech not in capacity_credit:
+            raise ValueError(
+                f"scenario['capacity_credit'] is missing '{_tech}'."
+            )
+    # hydro_capacity_mw was previously read only by the sliced builder
+    # (optimize_model_sliced.py), which dispatches hydro against a power
+    # limit. The annual builder had no installed-MW figure for hydro because
+    # it only needed the TWh energy series -- the firm capacity constraint
+    # below is the first thing in this file that needs hydro's MW rating.
+    hydro_capacity_mw = scenario.get("hydro_capacity_mw", None)
+    if hydro_capacity_mw is None:
+        raise ValueError(
+            "scenario['hydro_capacity_mw'] is required for the firm capacity "
+            "constraint (2.6, F0 Step 2.3): hydro's credited MW needs an "
+            "installed-capacity figure, not just the TWh energy series."
+        )
+
+
     m.solar_public_add = pyo.Var(T, domain=pyo.NonNegativeReals)
     m.solar_eaas_add = pyo.Var(T, domain=pyo.NonNegativeReals)
     m.eaas_subsidy = pyo.Var(T, domain=pyo.NonNegativeReals)
@@ -684,6 +737,35 @@ def build_model(
         # pointer, not a duplicate constraint.
         # m.reliability_constraint = m.grid_service_obligation
 
+    # ------------------------------------------------------------
+    # Firm Capacity Constraint (2.6, F0 Step 2.3)
+    # ------------------------------------------------------------
+    # Sigma_tech capacity_credit[tech] x cumulative_capacity[tech,t]
+    #   >= peak_mw[t] * (1 + reserve_margin)
+    #
+    # Unlike grid_service_obligation (a quantity-based service-obligation
+    # bound on selfgen+unserved), this is a genuine capacity-adequacy
+    # constraint: it tests whether CREDITED installed capacity covers a
+    # peak-demand forecast with a reserve margin, independent of what the
+    # LP chooses to dispatch. No such constraint existed anywhere in this
+    # file before 2.6, F0 (see docs/F0_audit.md, Step B6 of this same
+    # correction round).
+    #
+    # capacity_credit and peak_mw come from load_scenario() (2.6, F0 Step
+    # 2.4) -- see scenarios.py: capacity_credit (solar/gas/storage/hydro
+    # fractions, ALL SOURCE: [PENDING]), peak_multiple_case (PAR arm),
+    # reserve_margin. Storage's credited capacity is its POWER rating
+    # (m.storage_power_mw, MW), not its energy rating (m.storage_capacity_mwh,
+    # MWh) -- the constraint is dimensioned in MW throughout.
+    m.firm_capacity_constraint = pyo.Constraint(
+        T,
+        rule=lambda m, t:
+            capacity_credit["solar"] * m.solar_capacity_mw[t]
+            + capacity_credit["gas"] * m.gas_capacity_mw[t]
+            + capacity_credit["storage"] * m.storage_power_mw[t]
+            + capacity_credit["hydro"] * hydro_capacity_mw
+            >= peak_mw[t] * (1.0 + reserve_margin)
+    )
 
     # -------------------------
     # System Cost (Discounted NPV)
@@ -986,7 +1068,27 @@ def leakage_diagnostic(m, threshold=0.10):
         "flagged": share > threshold,
     }
 
-    
+
+def firm_capacity_diagnostic(m, years):
+    """Post-solve slack and dual on m.firm_capacity_constraint, per year
+    (2.6, F0 Step 2.3).
+
+    slack_mw : credited capacity in excess of peak_mw[t]*(1+reserve_margin).
+               0 (to solver tolerance) means the constraint is binding.
+    dual     : shadow price (USD, NPV-discounted, per additional MW of
+               required firm capacity) if the solver returned one for this
+               constraint, else None.
+    """
+    out = {}
+    for t, y in enumerate(years):
+        con = m.firm_capacity_constraint[t]
+        out[int(y)] = {
+            "slack_mw": float(pyo.value(con.body) - pyo.value(con.lower)),
+            "dual": m.dual.get(con, None),
+        }
+    return out
+
+
 # ============================================================
 # SOLVER WRAPPER
 # ============================================================
```

### Full diff, `src/optimize_model_sliced.py`

```diff
diff --git a/src/optimize_model_sliced.py b/src/optimize_model_sliced.py
index 056ad46..f8c2dcb 100644
--- a/src/optimize_model_sliced.py
+++ b/src/optimize_model_sliced.py
@@ -339,6 +339,49 @@ def build_model_sliced(
     if not (0.0 < gas_availability_factor <= 1.0):
         raise ValueError("gas_availability_factor must be in (0, 1]")
 
+    # ------------------------------------------------------------
+    # Firm capacity constraint parameters (2.6, F0 Step 2.3/2.4)
+    # ------------------------------------------------------------
+    # Mirrors optimize_model.py. Required directly here too, independent of
+    # load_scenario(): build_model_sliced() accepts any scenario dict a
+    # caller constructs directly, same discipline as hydro_baseline_twh /
+    # hydro_capacity_mw / gas_availability_factor above.
+    peak_mw = scenario.get("peak_mw", None)
+    if peak_mw is None:
+        raise ValueError(
+            "scenario['peak_mw'] is required (2.6, F0 Step 2.3). Derived by "
+            "load_scenario() from peak_multiple_case (Step 2.4); "
+            "build_model_sliced() will not silently assume a peak-demand "
+            "trajectory."
+        )
+    if len(peak_mw) != len(T):
+        raise ValueError(
+            f"scenario['peak_mw'] has {len(peak_mw)} entries for {len(T)} "
+            "model years. Refusing to build with a misaligned peak trajectory."
+        )
+    reserve_margin = scenario.get("reserve_margin", None)
+    if reserve_margin is None:
+        raise ValueError(
+            "scenario['reserve_margin'] is required (2.6, F0 Step 2.3). No "
+            "default: an omitted reserve margin would silently build an "
+            "unenforced firm-capacity constraint."
+        )
+    capacity_credit = scenario.get("capacity_credit", None)
+    if capacity_credit is None:
+        raise ValueError(
+            "scenario['capacity_credit'] is required (2.6, F0 Step 2.3). "
+            "Provide {'solar':..., 'gas':..., 'storage':..., 'hydro':...} "
+            "(see scenarios.py)."
+        )
+    for _tech in ("solar", "gas", "storage", "hydro"):
+        if _tech not in capacity_credit:
+            raise ValueError(
+                f"scenario['capacity_credit'] is missing '{_tech}'."
+            )
+    # NOTE: hydro's credited MW reuses hydro_capacity_mw, required further
+    # below for hydro dispatch (this builder has always needed it) -- not
+    # re-read a second time at the firm_capacity_constraint site.
+
     # ------------------------------------------------------------
     # ANNUAL decision variables (capacity and finance)
     # ------------------------------------------------------------
@@ -712,6 +755,33 @@ def build_model_sliced(
 
         # m.reliability_constraint = m.grid_service_obligation
 
+    # ------------------------------------------------------------
+    # Firm Capacity Constraint (2.6, F0 Step 2.3)
+    # ------------------------------------------------------------
+    # Sigma_tech capacity_credit[tech] x cumulative_capacity[tech,t]
+    #   >= peak_mw[t] * (1 + reserve_margin)
+    #
+    # Mirrors optimize_model.py exactly. Unlike grid_service_obligation (a
+    # quantity-based bound on selfgen+unserved), this tests whether CREDITED
+    # installed capacity covers a peak-demand forecast with a reserve margin,
+    # independent of dispatch. No such constraint existed anywhere in this
+    # file before 2.6, F0 (see docs/F0_audit.md, Step B6).
+    #
+    # ANNUAL, like all other capacity/financial constraints in this builder
+    # (see the module docstring): capacity is built once a year, not per
+    # slice, so this is not sliced either. Storage's credited capacity is
+    # its POWER rating (m.storage_power_mw, MW), not its energy rating
+    # (m.storage_capacity_mwh, MWh).
+    m.firm_capacity_constraint = pyo.Constraint(
+        T,
+        rule=lambda m, t:
+            capacity_credit["solar"] * m.solar_capacity_mw[t]
+            + capacity_credit["gas"] * m.gas_capacity_mw[t]
+            + capacity_credit["storage"] * m.storage_power_mw[t]
+            + capacity_credit["hydro"] * hydro_capacity_mw
+            >= peak_mw[t] * (1.0 + reserve_margin)
+    )
+
     # ------------------------------------------------------------
     # Costs
     # ------------------------------------------------------------
@@ -940,6 +1010,26 @@ def annual_totals(m, years):
     return out
 
 
+def firm_capacity_diagnostic(m, years):
+    """Post-solve slack and dual on m.firm_capacity_constraint, per year
+    (2.6, F0 Step 2.3). Mirrors optimize_model.py's version exactly.
+
+    slack_mw : credited capacity in excess of peak_mw[t]*(1+reserve_margin).
+               0 (to solver tolerance) means the constraint is binding.
+    dual     : shadow price (USD, NPV-discounted, per additional MW of
+               required firm capacity) if the solver returned one for this
+               constraint, else None.
+    """
+    out = {}
+    for t, y in enumerate(years):
+        con = m.firm_capacity_constraint[t]
+        out[int(y)] = {
+            "slack_mw": float(pyo.value(con.body) - pyo.value(con.lower)),
+            "dual": m.dual.get(con, None),
+        }
+    return out
+
+
 def slice_detail(m, years, t_index):
     """Per-slice breakdown for a single year — used to inspect peak adequacy."""
     rows = []
```

---

## STEP 3 — SOURCE: [PENDING] inventory

Every place a `SOURCE: [PENDING]` marker now sits, verbatim, all in
`src/scenarios.py`:

1. `PAR_OBSERVED = None   # SOURCE: [PENDING]` (module-level constant, top of
   file, in the new "PEAK DEMAND MULTIPLE (PAR)" section)
2. `PAR_CENTRAL  = None   # SOURCE: [PENDING]` (same section)
3. `PAR_HIGH     = None   # SOURCE: [PENDING]` (same section)
4. `"solar": 0.0,` — `# SOURCE: [PENDING]` (inline comment, `capacity_credit`
   dict, inside `load_scenario()`)
5. `"gas": _gas_availability_factor,` — `# SOURCE: [PENDING] (=
   gas_availability_factor)` (same dict)
6. `"storage": min(1.0, _storage_duration_hours / PERIOD_HOURS["peak"]),` —
   `# SOURCE: [PENDING]` (same dict)
7. `"hydro": _hydro_dry_avg_mw / _hydro_capacity_mw,` — `# SOURCE: [PENDING]`
   (same dict)

All four `capacity_credit` entries additionally carry an 18-line explanatory
comment block immediately above them (docstring-style, not reproduced again
here — see the diff above) walking through the derivation and why each is
still pending. The `peak_multiple_scenarios()` docstring and the
`PAR_*`-section header comment also state `SOURCE: [PENDING]` in prose, not
just inline.

**Nothing else in this task's diffs carries a `SOURCE: [PENDING]` marker.**
`hydro_capacity_mw`, `gas_availability_factor`, `hydro_baseline_twh`,
`storage_duration_hours`, `season_demand_factor`, `period_demand_factor` are
all **pre-existing** values (unchanged) that this task's refactor made
single-sourced but did not newly flag — several of them already carried
their own pre-existing `[SOURCE NEEDED]` / `[SOURCE: ...]` markers before
this task (see the quotes under STEP 0 above and the diff context); those
were left exactly as they were, not touched.

---

## Consequences not asked for directly, but real: existing `load_scenario()` callers now break

`peak_multiple_case` and `reserve_margin` are required with no default. Every
existing caller of `load_scenario()` that does not pass them will now raise
`TypeError` at that call site. This was not fixed as part of this task —
STEP 1/2 named only `src/optimize_model.py`, `src/optimize_model_sliced.py`,
and (implicitly, to supply the new keys) `src/scenarios.py`. Grep for
`load_scenario(` across the repository (raw hit list, unfiltered):

```
scripts/04_run_fin2_tariff_sweep.py
src/optimize_experiments.py        (11 real call sites: lines 635, 779, 801,
                                     1178, 1392, 1575, 1712, 2456, 2684, 2856, 2997)
src/io.py                          (1 hit -- prose mention in a docstring,
                                     not a real call)
scripts/verify_2_6_commit_a.sh
scripts/probe_2_6_structure.py
scripts/99c_run_genset_arm_sensitivity.py
scripts/25_run_obligation_trajectory.py
scripts/23_run_selfgen_frontier.py
scripts/24_run_leakage_arm.py
scripts/22_salvage_sensitivity.py
scripts/20_run_cf1_solar_yield_sensitivity.py
scripts/19_run_carbon_price_sweep.py
scripts/18_run_pareto_frontier.py
scripts/17_run_monte_carlo.py
scripts/15_run_pol1_ndc_comparison.py
scripts/09b_run_rel1_retirement_sensitivity.py
scripts/09b_run_rel1_mode_sensitivity.py
scripts/07_run_gas2_eaas_gas_relief.py
scripts/06_run_gas1_shadow_benchmarks.py
scripts/05_run_fin3_capital_sweep.py
scripts/03_run_ndc_eaas.py
scripts/02_run_ndc_caps.py
scripts/02_run_eaas.py
scripts/01b_run_baseline_sliced.py
scripts/01_run_baseline.py
src/scenarios.py                   (the definition site itself)
tests/test_run.py                  (1 real call site, line 8)
data/demand/README.md              (not individually verified as code -- likely prose)
```
`src/optimize_experiments.py` is the one to note most: its 11 call sites are
internal to shared experiment-running functions, so breakage there cascades
into every numbered script that calls those functions, not just scripts that
call `load_scenario()` directly. This mirrors exactly the fallout the
`genset_case` rollout caused (`src/io.py:180-186`, pre-edit: "That happened
twice -- in 02_run_ndc_caps.py and 03_run_ndc_eaas.py... both produced
clean-looking tables that were wrong"), except this time the failure mode is
loud (`TypeError` at call time) rather than silent, by design. No script or
`optimize_experiments.py` was edited in this task.

---

## Verification performed

- `ast.parse()` on `src/scenarios.py`, `src/optimize_model.py`,
  `src/optimize_model_sliced.py` — all three parse without error.
- Plain `import` of all three modules (via the `nigeria-energy` conda env's
  Python) — succeeds, confirms no circular import from `scenarios.py` now
  importing `optimize_model_sliced.py`, and confirms
  `firm_capacity_diagnostic` and `peak_multiple_scenarios` are present on the
  expected modules.
- Isolated calls to `load_scenario()` (no Pyomo, no model build) confirming:
  omitting `peak_multiple_case`/`reserve_margin` raises `TypeError`; a valid
  arm name with a still-`None` PAR value raises `ValueError` naming the
  `SOURCE: [PENDING]` gap; a consistent PAR value builds successfully and
  produces the expected `peak_mw`/`capacity_credit` values (matching the
  hand-computed values in STEP 0.6 and STEP 1 above); an inconsistent PAR
  value (2.0 against the implied 1.2522) raises the disagreement `ValueError`
  with the correct percentage.
- Direct call to `_normalised_demand_weights()`/`slice_hours()` (pure
  arithmetic) to produce the authoritative STEP 0.6 PAR figure.

**Not performed, by instruction:** `build_model()` and `build_model_sliced()`
were not called (no Pyomo `ConcreteModel` was constructed), no solver was
invoked, no runner script was executed, and the smoke suite was not run. The
firm capacity constraint's actual behaviour inside a solved model — whether
it binds, what its dual looks like, whether it interacts with
`grid_service_obligation` or the public budget constraint under any real
scenario — is **unverified** and cannot be reported here. That is squarely
what the pending `PAR_*`/`capacity_credit` sourcing and a subsequent solve
(explicitly out of this task's scope) will need to establish.
