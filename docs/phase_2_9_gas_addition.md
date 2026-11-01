# Phase 2.9 — enable new gas capacity (CCGT)

STEP 0 (below) was completed and reported first; the task required stopping
for a decision on 0.1 and 0.2 before touching any code. Both were resolved
by the user: **0.1** — mark `gas_capex_per_mw` `SOURCE: [PENDING]`, raise if
a scenario would actually use it for new-build CCGT. **0.2** — add
`gas_new_efficiency` exactly as Step 1 describes: `SOURCE: [PENDING]`, raise
if used while unset. STEPS 1-5 below implement on that basis. No solver,
runner, or smoke suite was run at any point — verification was
`python -m py_compile`, plain imports, `load_scenario()` unit-level calls,
model *construction* (not solving — `build_model()`/`build_model_sliced()`
build the Pyomo object graph only) under several arms to catch wiring bugs,
and `--help` on all 11 runners.

The STEP 0 section immediately below is unchanged from the original,
stand-alone read-only inventory (`grep`/`Read` only, no edits).

---

## 0.1 — `scenario["gas_capex_per_mw"]`: current value and documented source

Declared once, in `src/scenarios.py:1205`, inside the scenario dict returned
by `load_scenario()`:

```python
        # ---- Gas capacity baseline (gas-only fleet, MW)
        "gas_baseline_mw": 11000,
        # Brownfield retirement proxy: linear retirement from 2035.
        # (Comment corrected: 680 MW/yr x 11 years = 7,480 MW retired by 2045
        #  from an 11,000 MW gas-only base -- NOT the whole fleet.)
        "gas_baseline_retirement_start_year": 2035,
        "gas_baseline_retirement_mw_per_year": 680.0,
        "gas_capex_per_mw": 900000,
```

**Value: 900,000 (USD/MW). Documented source: none.** There is no comment,
docstring, or `SOURCE:` marker attached to this line at all — it sits
between the baseline-retirement block and the (separately commented, and
separately sourced) `gas_availability_factor` block a few lines down. I
checked whether the nearby availability-factor sourcing comment
(`src/scenarios.py:1219-1221`: *"fuel-agnostic mechanical/maintenance
availability, reconstructed from NERC outage cause-decomposition, or an
OCGT/CCGT engineering benchmark... REPLACE."*) might be a mis-attached source
for this constant — it is not: that comment is directly above, and reads as
attached to, `"gas_availability_factor": _gas_availability_factor,` on the
next line, a different quantity (a dispatch/energy-ceiling fraction, not a
capex figure).

Repo-wide checks confirm there is nothing else to find:
- `grep -rn "900000\|900,000\|gas_capex\|CCGT\|OCGT\|combined.cycle\|open.cycle" src/scenarios.py` → only the bare assignment itself and the unrelated OCGT/CCGT mention inside the availability-factor comment quoted above.
- No `data/cost/processed/*gas*capex*` file exists (only `data/cost/processed/gas_cost.csv`, which is *fuel* cost — commodity/transport USD per MMBtu/TWh_th — unrelated to a capital cost per MW).
- No CCGT- or OCGT-named data file anywhere under `data/`.

**Is it CCGT, OCGT, or unspecified? Unspecified.** Nothing in the code,
comments, or data files designates a technology. It has been consumed only
by `gas_capex_npv` (new baseline-gas builds, currently always zero since
`gas_add` is pinned) and by the salvage/replacement-capex expressions — i.e.
it has never yet priced an actual build, so its lack of a source has been
invisible until now. **This is exactly the kind of unsourced constant the
task's scope rules forbid inventing forward from** ("Do not invent numeric
values. Anything unsourced gets `SOURCE: [PENDING]`") — flagging for your
decision in 0.1/0.2 below rather than assuming CCGT and moving on.

---

## 0.2 — gas efficiency: existing-fleet vs. new-build

**Existing-fleet figure**, `src/scenarios.py:1196`:

```python
        # ---- Gas deliverability to power (TWh_th; converted by gas_eta)
        "gas_scenario": gas_deliverability_scenarios()[gas_deliverability_case],
        "gas_eta": 0.287,
```

Sourced elsewhere in the repo (not in `scenarios.py` itself, but documented
in `docs/merged_correction_plan_v2.md:68`): *"η = 0.287 from the identity
`observed TWh_e ÷ observed TWh_th`, using NERC 2024... (HHV 1036 Btu/scf).
Swept {0.277, 0.287, 0.298} for the unresolved [range]."* — i.e. a
*measured, blended, whole-fleet* efficiency backed out from NERC generation
and gas-consumption data, not an engineering rating of any specific plant
type. `docs/merged_correction_plan_v2.md:1051-1060` states this explicitly:
*"Measured fleet efficiency is η = 0.287 — brutally low, indicating
degraded, high-heat-rate plant... Proposed form: η as a piecewise investment
choice — `existing` at 0.287 and [new-build figure not yet given]"* — i.e.
the correction plan already anticipated splitting this but had not done so
as of that writing.

**New-build figure: none exists anywhere in the code today.** Both builders
read a single scalar:

```python
# src/optimize_model.py:254-256
    eta = scenario.get("gas_eta", None)
    if eta is None:
        raise ValueError("gas_eta is unset. Provide the measured fleet efficiency")
```
```python
# src/optimize_model_sliced.py:336-338
    eta = scenario.get("gas_eta", None)
    if eta is None:
        raise ValueError("gas_eta is unset. Provide the measured fleet efficiency.")
```

and this single `eta` is used for **both** dispatch (`m.gas_generation = eta
* m.gas_to_power[t]` in both files) and emissions (`ef_e =
econ["EF_TCO2_PER_MWH_TH"] / scenario["gas_eta"]`,
`src/optimize_model.py:1012`) — undifferentiated by vintage. There is
**one** efficiency parameter serving both existing and (currently
nonexistent) new capacity, not two. `docs/merged_correction_plan_v2.md`
confirms the correction to a piecewise `existing`/new form was proposed but
not implemented, and the current task's context note ("Phase 2.8 will
correct the fleet figure separately") is consistent with that gap still
being open. **New CCGT efficiency has no source in this repo at all** —
this is the second item flagged for your decision in Step 1's design
(`gas_new_efficiency`, `SOURCE: [PENDING]`, raise if used while unset — per
the task's own Step 1 instruction, which I have not yet implemented).

---

## 0.3 — `m.gas_add`: declaration and every consumer

**Declaration** — identical in both builders:
```python
# src/optimize_model.py:338
    m.gas_add = pyo.Param(T, initialize=0.0, within=pyo.NonNegativeReals)
```
```python
# src/optimize_model_sliced.py:419
    m.gas_add = pyo.Param(T, initialize=0.0, within=pyo.NonNegativeReals)
```
Preceded, in the annual builder only, by this comment
(`src/optimize_model.py:335-337`):
```python
    # No new gas plant construction: fuel availability is always the binding constraint.
    # New gas capacity is structurally non-optimal across all scenarios (verified).
    # gas_add is fixed at zero to make this assumption explicit.
```
(The sliced builder carries no equivalent comment at the declaration site.)

**Every consumer**, both files (line numbers: annual / sliced):

1. **Capacity accumulation** — `gas_cap_rule`, feeding `m.gas_capacity_mw`:
   ```python
   # src/optimize_model.py:354-364 (sliced: :457-466, same logic)
       def gas_cap_rule(m, t):
           year_t = int(years[t])
           if retirement_start is not None and year_t >= retirement_start:
               retired = retirement_rate * (year_t - retirement_start + 1)
               retired = min(retired, gas_baseline)
           else:
               retired = 0.0
           return gas_baseline - retired + sum(m.gas_add[k] for k in range(0, t + 1))
       m.gas_capacity_mw = pyo.Expression(T, rule=gas_cap_rule)
   ```
2. **Capex** — `gas_capex_npv` (`optimize_model.py:849-852` / `optimize_model_sliced.py:849-851`):
   ```python
   gas_capex_npv = pyo.quicksum(
       m.DF[t] * m.gas_add[t] * scenario["gas_capex_per_mw"]
       for t in T
   )
   ```
3. **Salvage** — inside `m.salvage_npv` (`optimize_model.py:936-940` / `optimize_model_sliced.py:930-934`):
   ```python
   + pyo.quicksum(
       m.gas_add[t] * scenario["gas_capex_per_mw"]
       * _salvage_fraction(t, lives["gas"])
       for t in T
   )
   ```
4. **Replacement-capex salvage term** — the rebuilt vintage's own residual, also inside `m.salvage_npv` (`optimize_model.py:956-961` / `optimize_model_sliced.py:950-955`):
   ```python
   + pyo.quicksum(
       m.gas_add[build_t] * scenario["gas_capex_per_mw"]
       * _salvage_fraction(rebuild_t, lives["gas"])
       for build_t in T
       for rebuild_t in _rebuild_years(build_t, lives["gas"])
   )
   ```
5. **Replacement capex itself** — inside `m.cost_replacement_capex_npv` (`optimize_model.py:985-989` / `optimize_model_sliced.py:979-983`):
   ```python
   + pyo.quicksum(
       m.DF[rebuild_t] * m.gas_add[build_t] * scenario["gas_capex_per_mw"]
       for build_t in T
       for rebuild_t in _rebuild_years(build_t, lives["gas"])
   )
   ```
6. **Post-solve in-horizon-retirement warning** (`optimize_model.py:1281`; sliced builder has the equivalent check but the warning function itself lives only in `optimize_model.py`'s `solve_model()`, called for both builders):
   ```python
   ("gas", _lives.get("gas"), lambda t: pyo.value(model.gas_add[t])),
   ```

**Indirect consumers, through `m.gas_capacity_mw` (not `gas_add` directly)** —
these do not name `gas_add` in their own expression, but their value moves
with it once additions are nonzero:
- `m.gas_capacity_constraint` (the per-MW dispatch ceiling — see 0.4)
- `m.firm_capacity_constraint` (`capacity_credit["gas"] * m.gas_capacity_mw[t]`)
- `m.land_constraint` (`m.gas_capacity_mw[t] * l_gas`)

**Emissions and dispatch (`m.gas_generation`, `m.gas_to_power`) do NOT
consume `gas_add` at all**, directly or by name — they are driven by
`m.gas_to_power[t]` (a free dispatch variable already, in both builders,
today) and bounded from above by `m.gas_capacity_mw[t]` only through
`m.gas_capacity_constraint`. Emissions (`ef_e = econ["EF_TCO2_PER_MWH_TH"] /
scenario["gas_eta"]`) depend on `gas_eta`, never on `gas_add`.

Every one of items 1-6 already treats `m.gas_add[·]` exactly like
`m.solar_public_add[·]` / `m.storage_add[·]` are treated alongside it in the
same expressions (all summed inside the same `pyo.quicksum(...)` terms) —
i.e. the arithmetic is already written generically over "any addition
variable," which is what the task's context note means by *"Claude Code's
`_rebuild_years()` replacement-capex loop is already generic across solar,
storage and gas."* Confirmed by direct reading, not assumed.

---

## 0.4 — the gas deliverability constraint

Two distinct gas-side constraints exist in both builders; only one of them
is "deliverability."

**`m.gas_balance`** — this is the deliverability constraint. Traced by name:
`scenario["gas_scenario"] = gas_deliverability_scenarios()[gas_deliverability_case]`
(`src/scenarios.py:1195`) feeds `gas_avail = gas_available_power(...,
scenario_name=scenario["gas_scenario"])["available_twh_th"]`, which is the
sole right-hand side of this constraint:

```python
# src/optimize_model.py:457-461
    m.gas_balance = pyo.Constraint(
        T,
        rule=lambda m, t:
            m.gas_to_power[t] + m.gas_curtail[t] <= gas_avail[t],
    )
```
```python
# src/optimize_model_sliced.py:552-559
    # Fuel is an ANNUAL budget the optimiser allocates across slices. This is
    # what lets scarce gas be saved for the dry-season evening peak.
    m.gas_balance = pyo.Constraint(
        T,
        rule=lambda m, t:
            pyo.quicksum(m.gas_to_power[t, (s, p)] for (s, p) in S)
            + m.gas_curtail[t] <= gas_avail[t]
    )
```

**Expressed against FUEL VOLUME** (`gas_avail[t]`, TWh_th — the deliverable
gas-to-power supply for the year, from the deliverability scenario), applied
to the single shared `gas_to_power` variable — **not** against installed
capacity and not directly against electrical generation (though generation
is bounded in aggregate since `gas_generation = eta * gas_to_power`).

For contrast, the other gas constraint, **`m.gas_capacity_constraint`**,
*is* expressed against capacity — and is explicitly labelled as such in the
sliced builder's own comment (`optimize_model_sliced.py:541-550`, *"Gas: per-slice
POWER limit, annual FUEL budget"*):
```python
# src/optimize_model.py:450-455
    m.gas_capacity_constraint = pyo.Constraint(
        T,
        rule=lambda m, t:
            m.gas_generation[t]
            <= m.gas_capacity_mw[t] * 8760 * gas_availability_factor / 1e6
    )
```
```python
# src/optimize_model_sliced.py:545-550
    m.gas_capacity_constraint = pyo.Constraint(
        T, m.S,
        rule=lambda m, t, s, p:
            m.gas_generation[t, (s, p)]
            <= m.gas_capacity_mw[t] * H[(s, p)] * gas_availability_factor / 1e6
    )
```

This distinction is load-bearing for Step 3 (deliverability treatment,
`new_gas_deliverability`): today there is exactly one `gas_to_power`
variable per (year[, slice]), shared by existing and (currently
nonexistent) new capacity, and exactly one `m.gas_balance` fuel budget it is
checked against. An `exempt` arm cannot simply "not apply `gas_balance` to
new capacity" without first separating new-build dispatch from existing-fleet
dispatch in some way, since there is currently no per-vintage dispatch
variable to exempt selectively — noting this now since it will bear directly
on how Step 3 has to be implemented, without proposing that implementation
here (that step has not been reached).

---

## 0.5 — retirement schedule on `m.gas_capacity_mw`

**Yes, but only for the baseline fleet — not for any addition.** Quoted
already in 0.3 item 1; repeating the relevant fragment:

```python
    def gas_cap_rule(m, t):
        year_t = int(years[t])
        if retirement_start is not None and year_t >= retirement_start:
            retired = retirement_rate * (year_t - retirement_start + 1)
            retired = min(retired, gas_baseline)  # cannot retire more than exists
        else:
            retired = 0.0
        return gas_baseline - retired + sum(m.gas_add[k] for k in range(0, t + 1))
```

`retired` is computed **only** from `gas_baseline` (the fixed 11,000 MW
2025 starting fleet), `gas_baseline_retirement_start_year` (2035), and
`gas_baseline_retirement_mw_per_year` (680.0) — none of which reference
`gas_add`. The `sum(m.gas_add[k] for k in range(0, t + 1))` term is added
**after** that subtraction, with no retirement schedule of its own —
identical in structure to `solar_capacity_mw` and `storage_capacity_mwh`
(both pure cumulative sums, confirmed in the prior phase's Step 0 inventory,
`docs/phase_2_3b_replacement_capex.md`).

**Confirmed: once `gas_add` is freed (Step 1), new gas capacity would be
subject to no retirement in `m.gas_capacity_mw` at all** — same
systemic defect already flagged for solar/storage in
`docs/phase_2_3b_replacement_capex.md` Step 0.4 ("a retired vintage is never
removed... stays fully dispatchable and fully firm-capacity-credited
forever"). The replacement-capex mechanism (already generic across
solar/storage/gas, per the task's context note) charges the missing *cost*
of a rebuild but — as documented in that same prior report — does **not**
remove the retired vintage from `gas_capacity_mw` itself. This is exactly
the same open issue as before, now extended to gas once `gas_add` moves;
not a new defect introduced by this task, and the task explicitly excludes
the firm-capacity constraint's form from this round's scope, so it is
reported here for the record and not proposed as something to fix in Step 1.

---

## STEP 0 decision record

Per `TASK.md`: *"STEP 0 — inventory. Report, then STOP for my decision on
0.1 and 0.2."* Both were put to the user as an explicit choice (not assumed):

- **0.1** (`gas_capex_per_mw = 900,000` USD/MW, no documented source, no
  CCGT/OCGT designation): **decided — mark `SOURCE: [PENDING]`, raise if a
  scenario would actually use it for new-build CCGT.** Implemented in Step 2
  below: any `gas_build_case` that permits new gas capacity currently raises
  unconditionally, citing this exact gap, until a sourced figure replaces
  the placeholder in `scenarios.py` and the raise is removed by whoever
  supplies it.
- **0.2** (no new-build CCGT efficiency figure anywhere in the repo):
  **decided — add `gas_new_efficiency`, `SOURCE: [PENDING]`, raise if used
  while unset, exactly as the task's own Step 1 text describes.**
  Implemented in Step 1 below.

---

## STEP 1 — free gas_add

### `m.gas_add`: Param → Var

Identical change in both builders. `src/optimize_model.py`:

```diff
     m.solar_public_add = pyo.Var(T, domain=pyo.NonNegativeReals)
     m.solar_eaas_add = pyo.Var(T, domain=pyo.NonNegativeReals)
     m.eaas_subsidy = pyo.Var(T, domain=pyo.NonNegativeReals)
-    # No new gas plant construction: fuel availability is always the binding constraint.
-    # New gas capacity is structurally non-optimal across all scenarios (verified).
-    # gas_add is fixed at zero to make this assumption explicit.
-    m.gas_add = pyo.Param(T, initialize=0.0, within=pyo.NonNegativeReals)
+    # New gas plant construction (2.9, Step 1): gas_add is now a free
+    # decision variable, gated by gas_build_case (2.9, Step 2) -- see below,
+    # after gas_capacity_mw is defined, for how each arm constrains it.
+    # Previously a pyo.Param fixed at 0.0 (pre-2.9: "New gas capacity is
+    # structurally non-optimal across all scenarios (verified)" -- true
+    # only because gas_add could not move at all).
+    m.gas_add = pyo.Var(T, domain=pyo.NonNegativeReals)
+    gas_build_case = scenario["gas_build_case"]
     # Storage:
     m.storage_add = pyo.Var(T, domain=pyo.NonNegativeReals)
```

`src/optimize_model_sliced.py` (same change, no preceding comment to
replace since the sliced builder never carried one):

```diff
     m.solar_public_add = pyo.Var(T, domain=pyo.NonNegativeReals)
     m.solar_eaas_add = pyo.Var(T, domain=pyo.NonNegativeReals)
     m.eaas_subsidy = pyo.Var(T, domain=pyo.NonNegativeReals)
-    m.gas_add = pyo.Param(T, initialize=0.0, within=pyo.NonNegativeReals)
+    # New gas plant construction (2.9, Step 1): gas_add is now a free
+    # decision variable, gated by gas_build_case (2.9, Step 2) -- see below,
+    # after gas_capacity_mw is defined, for how each arm constrains it.
+    m.gas_add = pyo.Var(T, domain=pyo.NonNegativeReals)
+    gas_build_case = scenario["gas_build_case"]
     m.storage_add = pyo.Var(T, domain=pyo.NonNegativeReals)
```

### Every Step 0.3 consumer, verified against the Var

| consumer | needed a change? | why |
|---|---|---|
| `gas_cap_rule` → `m.gas_capacity_mw` (`sum(m.gas_add[k] for k in range(0,t+1))`) | No | Already `pyo.Expression` arithmetic over `m.gas_add[k]`, identical in form to `m.solar_public_add`/`m.storage_add` in the same sum. A `Var` slots in exactly like a `Param` here — Pyomo builds the same kind of linear expression either way. |
| `gas_capex_npv` (`m.DF[t] * m.gas_add[t] * scenario["gas_capex_per_mw"]`) | No | Same reasoning — already written as `pyo.quicksum` over a Pyomo numeric object times a Python scalar. |
| `m.salvage_npv`'s gas term (`m.gas_add[t] * scenario["gas_capex_per_mw"] * _salvage_fraction(...)`) | No | Same. |
| `m.cost_replacement_capex_npv` / the replacement-vintage salvage term (`m.gas_add[build_t] * ...`, via `_rebuild_years()`) | No | Same — `_rebuild_years()` (phase_2_3b) was already written generically across solar/storage/gas, confirmed by direct reading in Step 0.3. |
| Post-solve retirement warning (`lambda t: pyo.value(model.gas_add[t])`) | No | `pyo.value()` works identically on a solved `Var` or a `Param` — it reads the numeric value either way. |
| **Indirect**: `m.gas_capacity_constraint`, `m.firm_capacity_constraint`, `m.land_constraint` (via `m.gas_capacity_mw`) | No | These never named `gas_add` directly; they consume the `Expression` `m.gas_capacity_mw`, which is unchanged in form. |

**Confirmed: nothing in the codebase relied on `gas_add` being a Python
float at build time.** Every consumer already treated it as a generic
Pyomo numeric object inside `quicksum`/`Expression` arithmetic — the
Param→Var conversion required zero changes to any of Step 0.3's six direct
consumers. The only place that changed is the declaration itself, plus the
new `gas_build_case`-driven wiring (fixing to 0, capping, or leaving free)
that has to exist *somewhere* now that the value isn't hard-pinned — see
Step 2.

Verified beyond static reading: `build_model()`/`build_model_sliced()` were
constructed (not solved) under `gas_build_case='gas_none'` and confirmed
`type(m.gas_add) is IndexedVar`, `m.gas_add[0].fixed == True`,
`m.gas_add[0].value == 0.0` — i.e. LP-equivalent to the old pinned `Param`.

### `gas_new_efficiency`

Added as an optional `load_scenario()` parameter (`gas_new_efficiency:
float | None = None`), stored on the scenario dict, validated (raise if
`None`) wherever it would actually be needed — see Step 2's
`gas_build_case` validation and Step 3's `exempt` validation, both in
`src/scenarios.py`.

**Scope boundary, flagged rather than silently expanded past:**
`gas_new_efficiency` is validated but **not** wired into `m.gas_generation`,
emissions, or dispatch OPEX in this phase. Doing so honestly would require
splitting `m.gas_to_power` into vintage-tagged existing/new dispatch
streams (Step 0.4 already flagged that only one shared dispatch variable
exists today) — a structural change neither Step 1's nor Step 3's literal
text asks for, and one whose ripple (through `gas_generation`,
`emissions_grid_only_by_year`, every diagnostic and runner script reading
those fields) is larger than "convert a Param to a Var and add one
parameter." `gas_new_efficiency` **does** get a genuine, bounded,
non-cosmetic use in Step 3 below (sizing the `exempt` fuel-equivalent
allowance) — it is not an inert parameter, just not one that touches
dispatch/emissions accounting in 2.9. Phase 2.8 ("will correct the fleet
figure separately and the two must remain attributable") is the natural
home for actually splitting dispatch by vintage, per the task's own framing.

---

## STEP 2 — gas_build_case as a required swept arm

### `src/scenarios.py`: sentinel, choices, trajectory function

```diff
 _PEAK_MULTIPLE_REQUIRED = object()
 _RESERVE_MARGIN_REQUIRED = object()
+_STORAGE_CC_REQUIRED = object()   # (phase_2_3b, unchanged this task)
+# Same mechanism, same reason (2.9, Step 2): m.gas_add moves from a Param
+# fixed at 0 to a free Var in this phase. A silent default here would let a
+# scenario build new gas capacity -- NDC 3.0's single largest capacity
+# commitment -- without a named, traceable choice of how much is allowed.
+_GAS_BUILD_REQUIRED = object()
```

```python
# gas_build_case (2.9, Step 2). Mirrors genset_case/peak_multiple_case/
# storage_cc_case discipline: REQUIRED, no default.
#
#   gas_ndc_17gw      : cumulative gas capacity capped on a trajectory
#                       reaching 17 GW by 2035 [Nigeria NDC 3.0, Sept 2025].
#                       The cap MECHANISM is implemented (see
#                       gas_new_capacity_cap_constraint in both builders,
#                       and gas_ndc_17gw_trajectory_mw() below) but the
#                       trajectory itself is SOURCE: [PENDING] -- selecting
#                       this arm raises. Intended CENTRAL once sourced.
#   gas_unconstrained : no cap; the LP builds whatever is cost-optimal.
#   gas_none          : gas_add fixed at 0 in both builders. Reproduces
#                       every pre-2.9 result; kept as an explicit, named arm
#                       so those runs stay attributable rather than orphaned.
GAS_BUILD_CASE_CHOICES = ("gas_ndc_17gw", "gas_unconstrained", "gas_none")


def gas_ndc_17gw_trajectory_mw(years) -> list[float] | None:
    """
    Cumulative gas capacity cap (MW), aligned to the model's year index, for
    the gas_ndc_17gw arm (2.9, Step 2).

    SOURCE: Nigeria NDC 3.0 (September 2025) -- 17 GW gas-fired capacity by
    2035 is the target figure itself. What is SOURCE: [PENDING] and NOT
    decided here:
      1. INTERPOLATION METHOD from the current 11,000 MW baseline fleet
         to 17,000 MW by 2035 -- linear, front-loaded to committed
         projects, or another shape entirely.
      2. TREATMENT AFTER 2035 -- flat at 17 GW, continued growth at the
         same implied rate, or uncapped.

    Returns None (not a guessed trajectory) until both are decided. Callers
    MUST raise before using this arm for anything, not interpolate a number
    that was never sourced.
    """
    return None
```

### `load_scenario()`: signature, docstring, validation

```diff
     storage_cc_case=_STORAGE_CC_REQUIRED,
+    gas_build_case=_GAS_BUILD_REQUIRED,
+    gas_new_efficiency: float | None = None,
+    new_gas_deliverability: str = "constrained",
     start_year: int = MODEL_START_YEAR,
     end_year: int = MODEL_END_YEAR,
 ) -> dict:
```

Validation, inserted right after `years = planning_horizon(start_year,
end_year)` (the trajectory function needs `years`):

```python
    # ---- gas_build_case (required, no default; 2.9, Step 2).
    if gas_build_case is _GAS_BUILD_REQUIRED:
        raise TypeError(
            "load_scenario() requires gas_build_case (2.9, Step 2). Pass "
            "one of 'gas_ndc_17gw' / 'gas_unconstrained' / 'gas_none' (see "
            "GAS_BUILD_CASE_CHOICES). There is no default: omitting it "
            "would let new gas capacity be built (or not) on an unstated "
            "assumption, with no error, warning, or visible signature in "
            "the output."
        )
    if gas_build_case not in GAS_BUILD_CASE_CHOICES:
        raise ValueError(f"Unknown gas_build_case: {gas_build_case}")

    gas_new_capacity_cap_mw = None
    if gas_build_case == "gas_ndc_17gw":
        gas_new_capacity_cap_mw = gas_ndc_17gw_trajectory_mw(years)
        if gas_new_capacity_cap_mw is None:
            raise ValueError(
                "gas_build_case='gas_ndc_17gw' selected, but its cumulative "
                "gas-capacity trajectory is SOURCE: [PENDING] (2.9, Step "
                "2): the interpolation method from the current 11,000 MW "
                "fleet to 17,000 MW by 2035, and the treatment after 2035, "
                "have not been decided ... Select 'gas_unconstrained' or "
                "'gas_none' until sourced."
            )
    if gas_build_case in ("gas_ndc_17gw", "gas_unconstrained"):
        if gas_new_efficiency is None:
            raise ValueError(
                f"gas_build_case='{gas_build_case}' allows new gas capacity, "
                "and new gas is declared CCGT (2.9, Step 1), but "
                "gas_new_efficiency is unset. SOURCE: [PENDING] -- do not "
                "reuse the 0.287 fleet figure (gas_eta); provide a sourced "
                "CCGT efficiency via load_scenario(gas_new_efficiency=...)."
            )
        raise ValueError(
            f"gas_build_case='{gas_build_case}' would allow new gas capacity "
            "to be built, but scenario['gas_capex_per_mw'] (currently a "
            "bare, unsourced 900,000 USD/MW figure with no CCGT/OCGT "
            "attribution -- 2.9, Step 0.1 finding) has not been sourced for "
            "new-build CCGT. SOURCE: [PENDING] -- only gas_build_case="
            "'gas_none' is usable until a sourced CCGT capex figure "
            "replaces the placeholder in scenarios.py and this raise is "
            "removed."
        )
```

(The `gas_capex_per_mw` raise is deliberately unconditional, not `if
gas_capex_per_mw is unsourced: raise` — there is no sourced/unsourced flag
to branch on, only a single hardcoded, never-sourced literal. Fixing it
means editing the placeholder in `scenarios.py` directly and removing this
raise, exactly the STEP 0 decision record above.)

`gas_capex_per_mw`'s own line now carries the finding as a comment:

```diff
         "gas_baseline_retirement_mw_per_year": 680.0,
+        # SOURCE: [PENDING] (2.9, Step 0.1 finding). No documented source,
+        # no CCGT/OCGT attribution -- a bare literal that has never priced
+        # an actual build (gas_add was pinned at 0 pre-2.9). Now that
+        # gas_add is a free Var and new gas is declared CCGT, this figure
+        # would be load-bearing for the first time; load_scenario() raises
+        # if a gas_build_case that permits new capacity is selected, so it
+        # cannot silently do so unsourced.
         "gas_capex_per_mw": 900000,
```

### Builders: wiring each arm onto `m.gas_add`, cap on cumulative capacity

Placed right after `m.gas_capacity_mw = pyo.Expression(T, rule=gas_cap_rule)`
in both builders (identical logic):

```python
    if gas_build_case == "gas_none":
        for t in T:
            m.gas_add[t].fix(0.0)
    elif gas_build_case == "gas_ndc_17gw":
        gas_new_capacity_cap_mw = scenario.get("gas_new_capacity_cap_mw", None)
        if gas_new_capacity_cap_mw is None:
            raise ValueError(
                "gas_build_case='gas_ndc_17gw' selected, but "
                "scenario['gas_new_capacity_cap_mw'] is unset ... A "
                "hand-built scenario cannot bypass this by setting "
                "gas_build_case directly without the trajectory."
            )
        m.gas_new_capacity_cap_constraint = pyo.Constraint(
            T,
            rule=lambda m, t: m.gas_capacity_mw[t] <= gas_new_capacity_cap_mw[t]
        )
    elif gas_build_case == "gas_unconstrained":
        pass  # no cap; the LP builds whatever is cost-optimal
```

The cap is on `m.gas_capacity_mw[t]` (cumulative), never on `m.gas_add[t]`
(annual additions), per the task's explicit instruction.

**Builder-side validation** (Step 4's "so hand-built scenarios cannot
bypass it"), mirroring the existing `storage_cc_case` presence-only guard,
added right after it in both builders:

```diff
     if scenario.get("storage_cc_case", None) is None:
         raise ValueError(...)
+    # gas_build_case (2.9, Step 2/4): closes the same hand-built-scenario
+    # bypass as the storage_cc_case guard above, for a caller that leaves
+    # new gas capacity's build policy unstated. The deep validation (arm
+    # membership, trajectory/capex/efficiency sourcing) lives in
+    # load_scenario() -- this is presence-only, matching the existing
+    # pattern.
+    if scenario.get("gas_build_case", None) is None:
+        raise ValueError(
+            "scenario['gas_build_case'] is missing or None (2.9, Step 2). "
+            "Required so new gas capacity has a traceable, named choice-set "
+            "-- see scenarios.GAS_BUILD_CASE_CHOICES."
+        )
```

`gas_ndc_17gw`'s and `exempt`'s deeper builder-side checks (quoted in Step 3
below) additionally close the bypass for a hand-built scenario that sets
`gas_build_case`/`new_gas_deliverability` directly with a missing trajectory
or efficiency — matching Step 4's intent more completely than a bare
presence check alone would.

Verified via model construction (not solving): `gas_ndc_17gw` with a
placeholder cap list built `m.gas_new_capacity_cap_constraint` in both
builders; `gas_unconstrained` left `m.gas_add` free (`fixed == False`).

---

## STEP 3 — deliverability treatment

### Design

Step 0.4 flagged that only one shared `m.gas_to_power` dispatch variable
exists — a *true* vintage-separated "exempt" treatment (new capacity's own
dispatch stream, entirely outside `m.gas_balance`) would require splitting
that variable, which ripples into `m.gas_generation`, emissions, and every
diagnostic/runner script that reads them — the same "larger than described"
boundary flagged for `gas_new_efficiency` in Step 1.

Implemented instead: `new_gas_deliverability='exempt'` adds a **fuel-
equivalent allowance** to `m.gas_balance`'s right-hand side, sized to new
capacity's *own* capacity-and-efficiency ceiling
(`gas_new_capacity_mw[t] * 8760 * gas_availability_factor / gas_new_efficiency
/ 1e6`) — i.e. exactly as much extra thermal fuel as new capacity would need
to run flat-out at its own MW rating. This is not an approximation of a
truly split model — it is **exactly equivalent** to one, given that neither
generation efficiency (`eta`) nor OPEX price differs by vintage anywhere in
this phase (Step 1's scope decision): since the LP cannot distinguish "new
capacity's fuel" from "existing capacity's fuel" economically, raising the
shared ceiling by precisely the additional throughput new capacity's own
supply arrangement would provide produces the identical physical and
economic outcome as a true per-vintage split would, without introducing a
second dispatch variable. `gas_new_efficiency` gets its one genuine use in
this phase here (sizing the allowance), not in dispatch/emissions.
`constrained` leaves `m.gas_balance` byte-identical to pre-2.9.

### Diff — `src/optimize_model.py` (`src/optimize_model_sliced.py` identical
in structure, only the constraint body's slice-summation differs, as it
already did before this change)

```diff
-    m.gas_balance = pyo.Constraint(
-        T,
-        rule=lambda m, t:
-            m.gas_to_power[t] + m.gas_curtail[t] <= gas_avail[t],
-    )
+    new_gas_deliverability = scenario.get("new_gas_deliverability", "constrained")
+    if new_gas_deliverability not in ("constrained", "exempt"):
+        raise ValueError(f"Unknown new_gas_deliverability: {new_gas_deliverability}")
+
+    if new_gas_deliverability == "exempt":
+        gas_new_efficiency = scenario.get("gas_new_efficiency", None)
+        if gas_new_efficiency is None:
+            raise ValueError(
+                "new_gas_deliverability='exempt' sizes new capacity's exempt "
+                "fuel-equivalent allowance using gas_new_efficiency, which is "
+                "unset (2.9, Step 1/3 -- SOURCE: [PENDING]). A hand-built "
+                "scenario cannot bypass this by setting new_gas_deliverability "
+                "directly without it."
+            )
+        m.gas_new_capacity_mw = pyo.Expression(
+            T, rule=lambda m, t: sum(m.gas_add[k] for k in range(0, t + 1))
+        )
+        m.gas_exempt_fuel_allowance_twh_th = pyo.Expression(
+            T,
+            rule=lambda m, t:
+                m.gas_new_capacity_mw[t] * 8760 * gas_availability_factor
+                / gas_new_efficiency / 1e6
+        )
+        m.gas_balance = pyo.Constraint(
+            T,
+            rule=lambda m, t:
+                m.gas_to_power[t] + m.gas_curtail[t]
+                <= gas_avail[t] + m.gas_exempt_fuel_allowance_twh_th[t],
+        )
+    else:
+        m.gas_balance = pyo.Constraint(
+            T,
+            rule=lambda m, t:
+                m.gas_to_power[t] + m.gas_curtail[t] <= gas_avail[t],
+        )
```

`new_gas_deliverability` validation in `load_scenario()` (real default,
unlike the required-no-default arms above):

```python
    if new_gas_deliverability not in ("constrained", "exempt"):
        raise ValueError(
            f"Unknown new_gas_deliverability: {new_gas_deliverability}. "
            "Pass 'constrained' or 'exempt'."
        )
    if new_gas_deliverability == "exempt" and gas_new_efficiency is None:
        raise ValueError(
            "new_gas_deliverability='exempt' sizes new capacity's exempt "
            "fuel-equivalent allowance using gas_new_efficiency, which is "
            "unset. SOURCE: [PENDING] -- provide a sourced CCGT efficiency "
            "via load_scenario(gas_new_efficiency=...)."
        )
```

**Both are implemented; neither is chosen for the caller** — `constrained`
is only the *default* (per the task: "must not be relaxed by assumption"),
not a hardcoded exclusive choice; `exempt` is fully functional given a
sourced `gas_new_efficiency`.

Verified via model construction: `exempt` (with a placeholder
`gas_new_efficiency=0.55` for code-path testing only, not a sourced value)
built `m.gas_exempt_fuel_allowance_twh_th` and `m.gas_new_capacity_mw` in
both builders; `constrained` (the default) left `m.gas_balance` in its
original two-term form.

---

## STEP 4 — wire and tag

`STORAGE_CC_CASE_CHOICES`'s sibling constants and CLI args were added to all
eleven runners that already carry `--peak-case`/`--storage-cc-case`:
`01_run_baseline.py`, `01b_run_baseline_sliced.py`, `02_run_eaas.py`,
`02_run_ndc_caps.py`, `03_run_ndc_eaas.py`, `04_run_fin2_tariff_sweep.py`,
`22_salvage_sensitivity.py`, `23_run_selfgen_frontier.py`,
`24_run_leakage_arm.py`, `25_run_obligation_trajectory.py`,
`99c_run_genset_arm_sensitivity.py`.

Full worked example, `scripts/01_run_baseline.py` (the other ten follow the
identical mechanical pattern — CLI arg block, `load_scenario()` kwargs,
`tag`/`output_tag`/`case_dir` extension, summary-dict fields — including
threading both new parameters through any per-file helper function that
previously took `peak_case`/`storage_cc_case`, exactly as `storage_cc_case`
itself was threaded through them in phase_2_3b):

```diff
 PEAK_CASE_CHOICES = ("peak_suppressed", "peak_unsuppressed", "peak_nirp", "peak_high")
 STORAGE_CC_CASE_CHOICES = ("storage_cc_low", "storage_cc_central", "storage_cc_high")
+# 2.9, Step 4: new gas capacity's build policy and deliverability treatment
+# are REQUIRED CLI arguments, no default -- mirrors the peak-case/
+# storage-cc-case/genset_case discipline.
+GAS_BUILD_CASE_CHOICES = ("gas_ndc_17gw", "gas_unconstrained", "gas_none")
+NEW_GAS_DELIVERABILITY_CHOICES = ("constrained", "exempt")


 def parse_args():
     parser = argparse.ArgumentParser(...)
     parser.add_argument("--peak-case", ...)
     parser.add_argument("--storage-cc-case", ...)
+    parser.add_argument(
+        "--gas-build-case",
+        required=True,
+        choices=GAS_BUILD_CASE_CHOICES,
+        help="New gas capacity build policy fed to load_scenario(gas_build_case=...). "
+             "No default: an unstated build policy would let new gas capacity be "
+             "built (or not) silently.",
+    )
+    parser.add_argument(
+        "--new-gas-deliverability",
+        required=True,
+        choices=NEW_GAS_DELIVERABILITY_CHOICES,
+        help="New gas capacity's fuel-deliverability treatment fed to "
+             "load_scenario(new_gas_deliverability=...). No default: an unstated "
+             "treatment would relax (or not) the fleet fuel constraint for new "
+             "build silently.",
+    )
     return parser.parse_args()


 def main():
     args = parse_args()
-    tag = f"{args.peak_case}__{args.storage_cc_case}"
+    tag = f"{args.peak_case}__{args.storage_cc_case}__{args.gas_build_case}__{args.new_gas_deliverability}"

     scenario = load_scenario(
         ...
         storage_cc_case=args.storage_cc_case,
+        gas_build_case=args.gas_build_case,
+        new_gas_deliverability=args.new_gas_deliverability,
     )
     ...
     summary = {
         ...
         "storage_cc_case": args.storage_cc_case,
+        "gas_build_case": args.gas_build_case,
+        "new_gas_deliverability": args.new_gas_deliverability,
         ...
     }
-    with open(RESULTS_DIR / f"diagnostics__{args.peak_case}.json", "w") as f:
+    with open(RESULTS_DIR / f"diagnostics__{tag}.json", "w") as f:
     ...
```

**Per-file tagging mechanism** (all now carry both new arms, in addition to
peak/storage-cc, alongside the existing pattern each file already used for
`storage_cc_case`):
- Nine files (`01_run_baseline.py`, `01b_run_baseline_sliced.py`,
  `02_run_eaas.py`, `22_salvage_sensitivity.py`, `23_run_selfgen_frontier.py`,
  `24_run_leakage_arm.py`, `25_run_obligation_trajectory.py`,
  `99c_run_genset_arm_sensitivity.py`, plus `04_run_fin2_tariff_sweep.py`'s
  `output_tag`): `tag`/`output_tag` extended to
  `f"...{peak_case}__{storage_cc_case}__{gas_build_case}__{new_gas_deliverability}"`.
- Two files (`02_run_ndc_caps.py`, `03_run_ndc_eaas.py`): `case_dir` name
  extended to `f"{c}_{voll_case}_{peak_case}_{storage_cc_case}_{gas_build_case}_{new_gas_deliverability}"`.
- Helper functions threaded with both new parameters (mirroring
  `storage_cc_case`'s own threading in phase_2_3b): `run_arm()`
  (`22_salvage_sensitivity.py`), `make_scenario()`/`solve_point()`
  (`23_run_selfgen_frontier.py`), `solve_one()` (`24_run_leakage_arm.py`,
  `25_run_obligation_trajectory.py`, `99c_run_genset_arm_sensitivity.py`).
- `04_run_fin2_tariff_sweep.py`'s per-row dicts still don't carry a
  per-row `peak_case`/`storage_cc_case` field (matched, not changed, from
  phase_2_3b) — only `output_tag` is tagged, consistent with that file's
  existing pattern; `gas_build_case`/`new_gas_deliverability` follow the
  same choice for consistency.

Verified: `python -m py_compile` on all 11 files, and
`python <script>.py --help` shows `--gas-build-case` and
`--new-gas-deliverability` correctly on every one (4 matches each: the
metavar line + the help-text line, for both flags).

---

## STEP 5 — report

This document. Summary of what's still outstanding:

### Outstanding `SOURCE: [PENDING]` items after this task

1. **`gas_capex_per_mw`** (`scenarios.py`) — no CCGT/OCGT capex source.
   Blocks `gas_build_case='gas_unconstrained'` and `'gas_ndc_17gw'`
   unconditionally until replaced.
2. **`gas_new_efficiency`** — no default, no sourced value supplied by any
   caller today. Blocks the same two arms, and blocks
   `new_gas_deliverability='exempt'` on its own even under a hand-built
   scenario.
3. **`gas_ndc_17gw_trajectory_mw()`** — returns `None`. Two sub-decisions
   pending: interpolation method from 11,000 MW (2025) to 17,000 MW (2035),
   and treatment after 2035. Blocks `gas_build_case='gas_ndc_17gw'`
   unconditionally.

**Net effect: only `gas_build_case='gas_none'` is currently runnable
end-to-end** (reproducing every pre-2.9 result exactly, per its
docstring/task definition). `gas_unconstrained` and `gas_ndc_17gw` are
wired, validated, and structurally ready, but both raise until (1)+(2), and
additionally (3) for `gas_ndc_17gw`, are sourced. This is the deliberate,
literal consequence of the STEP 0 decision record, not an oversight — it
mirrors the pre-existing precedent in this codebase (`peak_multiple_case='peak_nirp'`
already raises the same way, for the same reason: an arm exists and is
wired, but its number was never sourced).

### Every consumer of `gas_add`, and whether it needed changing

See the table in Step 1. Six direct consumers (capacity accumulation,
capex, salvage, salvage-of-rebuild, replacement capex, retirement warning)
— **none** needed a change; three indirect consumers via
`m.gas_capacity_mw` (capacity constraint, firm-capacity constraint, land
constraint) — **none** needed a change. The only genuinely new code is the
`gas_build_case`-driven wiring (fix/cap/free) that has to exist now that
the value isn't hard-pinned to zero.

### Interactions with the firm-capacity constraint

`m.firm_capacity_constraint`'s form was **not** touched (excluded by the
task). But its *behaviour* changes the moment `gas_build_case` ever permits
nonzero `gas_add` (currently blocked by the pending items above): new gas
capacity flows into `m.gas_capacity_mw`, which is one of the four terms on
the credited-capacity side of that constraint
(`capacity_credit["gas"] * m.gas_capacity_mw[t]`). This reintroduces gas as
a competing firm-capacity resource alongside storage — directly relevant to
the context note that storage currently resolves to a closed-form value
(`77,387.72 / capacity_credit["storage"]`) *because* gas and solar are both
excluded from the choice set. Once a `gas_build_case` other than
`gas_none` is actually runnable, storage's closed-form result should be
expected to change, since it will no longer be the only capacity that can
supply an incremental firm megawatt. This is exactly the intended
consequence, not a side effect to correct.

Also unchanged, and reported prominently again per the phase_2_3b
precedent: `m.gas_capacity_mw`'s cumulative-sum accumulation still never
retires a `gas_add` vintage (Step 0.5 finding). New gas capacity, once any
is actually built under a sourced `gas_build_case`, will be **permanently**
both dispatchable and firm-capacity-credited in this same way solar and
storage additions already are — this task does not change that, and it was
explicitly out of scope (the task excludes the firm-capacity constraint's
form).

### Interactions with replacement capex

None required — `_rebuild_years()` (phase_2_3b) was already generic across
solar/storage/gas, confirmed directly in Step 0.3 and Step 1's consumer
table. Once a sourced `gas_build_case` permits nonzero `gas_add`, gas
vintages will automatically participate in the SAME rebuild-on-retirement
and salvage mechanism solar/storage already use — no new code path, no
new interaction to design. One numeric note for whoever eventually sources
`gas_capex_per_mw`/`gas_new_efficiency`: gas's technical life is 30 years
(`asset_lifetimes()`), matching the 2025-2054 (30-year) horizon exactly —
per phase_2_3b's Step 0.2 finding, this means new gas vintages, like solar,
will **not** actually trigger an in-horizon rebuild under the current
horizon length (a vintage built at any `t` serves `<= 30` years, never more)
even once gas capex is sourced and gas is built. `m.cost_replacement_capex_npv`
would remain zero-valued for gas specifically unless the horizon is
extended past 30 years — this is a fact about the current horizon and gas's
sourced life figure, not a bug, and was already noted generically in
phase_2_3b's own report.

---

# gas_ndc_17gw trajectory unblocked

Addendum. The `gas_ndc_17gw` arm previously always raised (its
trajectory was `SOURCE: [PENDING]`). Finding P2-6 answered both open
questions -- the interpolation shape and the post-2035 treatment are
now supplied from NIRP's own modelled path.

## Step 3 first -- what the cap constraint compares against

`gas_new_capacity_cap_constraint` is built identically in both builders,
inside the `gas_build_case == "gas_ndc_17gw"` branch:

- **annual** `src/optimize_model.py:528-531`:
  ```
  m.gas_new_capacity_cap_constraint = pyo.Constraint(
      T,
      rule=lambda m, t: m.gas_capacity_mw[t] <= gas_new_capacity_cap_mw[t]
  )
  ```
- **sliced** `src/optimize_model_sliced.py:672-675`: byte-identical.

**What it caps:** `m.gas_capacity_mw[t]` -- CUMULATIVE gas capacity in
year `t`, i.e. `gas_baseline_mw - retired(t) + sum(m.gas_add[k] for k in
0..t)` (`gas_cap_rule`, annual `:494-506`, sliced equivalent). Not
annual additions; the whole standing fleet.

**On what basis -- NAMEPLATE, unambiguously:**

1. `gas_baseline_mw` = `GAS_BASELINE_MW` = 10,947 MW, sourced as "NERC
   Quarterly Reports -- 23 gas plants tracked for availability ...
   totalling 10,947 MW" (`src/scenarios.py:287-294`). That is NERC's
   INSTALLED-MW figure. `gas_avail_observed` is derived as
   `3,351 / 10,947` -- 3,351 is the *available* MW, 10,947 the
   *installed*. [Corrected TASK 73: that quotient is 0.3061, not the
   0.295 then in use; the value is now 0.3061, register L128.]
2. `m.gas_capacity_mw` is multiplied by `gas_availability_factor`
   (0.295 / 0.80 / ...) to get the annual gas ENERGY ceiling
   (`gas_capacity_constraint`, sliced `:756-761`) and by
   `capacity_credit["gas"]` to get the firm-capacity credit. The model
   derates `gas_capacity_mw` in those two places -- which is only
   coherent if `gas_capacity_mw` is itself nameplate.
3. TASK.md's own instruction "hold at `scenario['gas_baseline_mw']`
   before 2030 (the cap must not bind below the existing fleet)" fixes
   the pre-2030 cap AT the nameplate baseline -- so the trajectory and
   `gas_baseline_mw` are on the same (nameplate) basis.

The constraint's basis is therefore **not ambiguous** -- no STOP.

## Step 1 -- the trajectory

`gas_ndc_17gw_trajectory_mw(years)` (`src/scenarios.py`) now returns the
series instead of `None`.

**Anchors (TASK.md), derated basis (~0.77 of nameplate):**
2030 = 9,188 / 2035 = 17,027 / 2041 = 20,945 / 2045 = 18,408 MW.

**Basis conversion -- explicit, not silent.** The anchors are derated;
`m.gas_capacity_mw` is nameplate (Step 3). Each anchor is divided by the
P2-6 derate **0.77** to put the cap on the constraint's basis:

| year | derated (anchor) | nameplate (`/0.77`, stored) |
|------|------------------|-----------------------------|
| 2030 |  9,188 | 11,932.5 |
| 2035 | 17,027 | 22,113.0 |
| 2041 | 20,945 | 27,201.3 |
| 2045 | 18,408 | 23,906.5 |

**Cross-check that the direction is right:** NDC 3.0 describes its gas
target as "roughly double current levels" (P2-5). 22,113 / 10,947 =
2.02x -- matches. The un-converted 17,027 is only 1.55x, which does not.
So the derated derived value (17,027) and the NDC's *stated* "17 GW"
coincide because both are effective/derated figures; the nameplate
target is ~22 GW. Converting UP to nameplate is correct.

**Shape:** linear interpolation between anchors; held at
`GAS_BASELINE_MW` (10,947) for 2025-2029; held flat at the 2045
nameplate value (23,906.5) for 2046-2054. All DECLARED, not sourced --
NIRP states the anchor years only. The verbatim provenance comment
(TASK.md) plus a BASIS paragraph are at the function docstring.

The 0.77 derate is DECLARED (P2-6, "~0.77 of nameplate"); it is not
independently sourced, and it is the one number in this trajectory not
traceable to a NIRP table. Recorded as such.

## Step 2 -- raise removed

`load_scenario()` `gas_build_case` branch: the
`if gas_new_capacity_cap_mw is None: raise ValueError(...)` block
(the "trajectory is SOURCE: [PENDING], select gas_unconstrained/
gas_none" message) is removed -- the function no longer returns `None`.
Every other arm's validation is untouched: `gas_none` still fixes
`gas_add` to 0, `gas_unconstrained` still passes through, the
`_GAS_BUILD_REQUIRED` sentinel and `not in GAS_BUILD_CASE_CHOICES`
checks are unchanged.

The `gas_new_capacity_cap_mw is None` guard INSIDE each builder
(annual `:519-527`, sliced `:663-671`) is LEFT IN PLACE -- it protects
a hand-built scenario dict that sets `gas_build_case="gas_ndc_17gw"`
without going through `load_scenario()`, the same hand-built-bypass
discipline used for the other required keys.

Stale docs updated: the `gas_build_case` module comment
(`:1334-1349`), the `load_scenario` docstring entry, and the
`gas_ndc_17gw` line in it ("currently always raises" -> the P2-6
trajectory).

## Verification (the one permitted execution)

ONE non-solving `build_model_sliced()` under `gas_ndc_17gw` +
`conv_central` + `f1_tranche` + `cr_central` + `dscr_regression` +
`env_dares`:

- `load_scenario(gas_build_case="gas_ndc_17gw")` returns without raising.
- Resolved `scenario["gas_new_capacity_cap_mw"]`, NAMEPLATE MW:
  2025 = 10,947.0 / 2030 = 11,932.5 / 2035 = 22,113.0 /
  2041 = 27,201.3 / 2045 = 23,906.5 / 2054 = 23,906.5.
  2033 (interpolated) = 18,040.8 = `11,932.5 + (3/5)(22,113.0 - 11,932.5)`.
- `build_model_sliced()` builds; `m.gas_new_capacity_cap_constraint`
  present, indexed over 30 years; 2035 row upper bound = 22,112.99.

No solve, no runner, no other execution.
