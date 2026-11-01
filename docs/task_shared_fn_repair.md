# Repair the six shared `load_scenario()` call sites — Step One inventory

Branch `phase-fin2-d17`. No file edited — this is Step One only, per
TASK.md's explicit "STOP. Do not begin Part 3." All six functions read
in full (not via grep), in `src/optimize_experiments.py`.

**Two corrections to TASK.md's own §1 summary, found during this
inventory and reported per standing instruction to verify rather than
assume:**

1. **`run_gas_regime_ndc_matrix` has three runner callers, not two.**
   TASK.md's table reads `run_gas_regime_ndc_matrix <- called by 06_,
   08_`. A tree-wide grep (2E, below) found a third:
   `07_run_gas2_eaas_gas_relief.py` calls it twice (lines 326, 346).
   This does not change the repair's risk profile — `07_` is itself
   dead (Group 2/3 boundary, per the runner inventory) — but the
   caller list itself was incomplete.
2. **Two of the six functions self-document the defect, not one.**
   TASK.md says "One of the six (`run_rel1_feasibility_matrix`)
   self-documents the defect in an inline comment." Confirmed for that
   one, but `run_gas_regime_ndc_matrix` also carries its own
   self-documenting comment (quoted in full under that function's 2B
   below), naming the same missing-keys list.

---

## 1. `run_rel3_financing_frontier` (lines 712–839)

### 2A — Signature

```python
def run_rel3_financing_frontier(
    econ,
    gas_cases,
    policy_configs,
    eps_levels=None,
    cap_path="data/cost/processed/emissions_cap.csv",
):
```
Five parameters, three with no default (`econ`, `gas_cases`,
`policy_configs`), two with defaults (`eps_levels=None`,
`cap_path=...`).

### 2B — `load_scenario()` call(s), quoted in full

**Two distinct call sites**, lines 790–800 and 814–824, otherwise
identical:

```python
# lines 790–800 (public-only arm)
scenario_pub = load_scenario(
    demand_level_case="served",
    demand_case="organic_central",
    gas_deliverability_case=gas_case,
    capital_case=pub_capital,
    carbon_case="no_policy",
    start_year=2025,
    end_year=MODEL_END_YEAR,
    tariff_case=tariff_case,
    om_case="om_central",
)
```
```python
# lines 814–824 (EaaS arm)
scenario_eaas = load_scenario(
    demand_level_case="served",
    demand_case="organic_central",
    gas_deliverability_case=gas_case,
    capital_case=eaas_capital,
    carbon_case="no_policy",
    start_year=2025,
    end_year=MODEL_END_YEAR,
    tariff_case=tariff_case,
    om_case="om_central",
)
```

**Passes** (of the 17): `tariff_case`, `om_case` (as the literal
`"om_central"`, not a parameter — see general note at the end of this
report). Also passes `demand_level_case`, `demand_case`,
`gas_deliverability_case`, `capital_case`, `carbon_case`, `start_year`,
`end_year` — none of which are in the 17 sentinel-required set (they
carry real defaults in `load_scenario()`), included here for
completeness only.

**Omits 15**: `peak_multiple_case`, `reserve_margin`, `storage_cc_case`,
`gas_build_case`, `gas_availability_case`, `gas_retirement_case`,
`gas_capex_case`, `ccgt_efficiency_case`, `capital_recovery_case`,
`dscr_case`, `us_inflation`, `capital_treatment_case`,
`concessional_envelope_case`, `conversion_case`, `hydro_case`.

### 2C — Is a suitable value already in scope, per omitted key?

Checked every local, parameter, and module-level constant in this
function and in `src/optimize_experiments.py` at module scope (grepped
for `^CANONICAL_` and `^[A-Z_]* =` — the file defines none of the 15
keys' values anywhere). **None of the 15 has a value already in scope.**
All 15 would need a new parameter. (Register **L9** documents that
`reserve_margin="nirp_spinning"` and a fixed `genset_case` are used
identically as *hardcoded literals* in ~11-20 other runners across this
project — a strong precedent for what value to choose, but not a value
*already present in this function's own scope*, so it does not satisfy
2C's "already available" test as asked. Flagged as useful context for
whoever does Part 3, not claimed as an exception here.)

### 2D — Distinct `load_scenario()` call sites

**Two** (public-only arm, EaaS arm) — both identical in which keys they
pass/omit, so both need the same fix applied twice.

### 2E — Other callers, whole-tree search

```
grep -rn "run_rel3_financing_frontier" --include="*.py"
```
One caller found beyond the definition: `scripts/11_run_rel3_financing_frontier.py:195`.
`compute_frontier_shift()` (`optimize_experiments.py:1019`) references
this function's *output shape* in a docstring comment only — it consumes
a DataFrame, it does not call the function. **No other caller.**

### 2F — Called by any healthy runner?

**No.** `11_run_rel3_financing_frontier.py` is dead (runner inventory,
Group 3). Checked directly: none of `01_`, `01b_`, `02_run_eaas`,
`02_run_ndc_caps`, `03_`, `22_`, `23_`, `26_` import or reference
`run_rel3_financing_frontier` anywhere (confirmed by the whole-tree grep
in 2E returning only the one caller).

---

## 2. `run_dem1_demand_sensitivity` (lines 1101–1327)

### 2A — Signature

```python
def run_dem1_demand_sensitivity(
    econ,
    demand_level_cases,
    ndc_configs,
    cap_path="data/cost/processed/emissions_cap.csv",
):
```
Four parameters, three with no default.

### 2B — `load_scenario()` call, quoted in full

**One call site**, lines 1198–1209:

```python
scenario = load_scenario(
    demand_level_case=demand_level,
    demand_case="organic_central",
    gas_deliverability_case="baseline",
    capital_case=fin_params["capital_case"],
    solar_build_case="deployment_unconstrained",
    carbon_case="no_policy",
    start_year=2025,
    end_year=MODEL_END_YEAR,
    tariff_case=tariff_case,
    om_case="om_central",
)
```

**Passes**: `tariff_case` (sourced from `ndc_cfg["tariff_case"]`, a
caller-supplied config field), `om_case` (hardcoded literal). Also
`demand_level_case` (the function's own swept dimension —
`demand_level_cases` is this function's whole reason to exist),
`demand_case`, `gas_deliverability_case`, `capital_case`,
`solar_build_case`, `carbon_case`, `start_year`, `end_year` (all
non-sentinel).

**Omits 15**: identical list to function 1 above (`peak_multiple_case`
through `hydro_case`).

An inline comment at lines 1225–1231 (immediately after this call)
**self-documents the general defect class**, though it names only five
of the fifteen explicitly: *"This `load_scenario()` call is also
already missing `peak_multiple_case`/`reserve_margin`/`storage_cc_case`/
`gas_build_case`/`gas_availability_case` (required since 2.6/2.9/2.10)
and would raise before reaching here regardless."* This is the comment
TASK.md attributes to `run_rel1_feasibility_matrix` — it is present
there too (identical wording), so it was evidently copy-pasted along
with the defect it describes. See the correction note at the top of
this report.

### 2C — In-scope substitutes?

None of the 15 has an in-scope value, same finding as function 1.

### 2D — Distinct call sites

**One.**

### 2E — Other callers

```
grep -rn "run_dem1_demand_sensitivity" --include="*.py"
```
One real caller: `scripts/12_run_dem1_demand_sensitivity.py:325`. One
false positive: `src/EaaS.py:36` names this function in a doc-code-parity
comment, explicitly clarifying it is *"a name coincidence, not a
caller"* — `12_`'s own `compute_eaas_deployment` is an unrelated local
function of the same name; `EaaS.py` is never imported by `12_`. **No
other real caller.**

### 2F — Called by any healthy runner?

**No.** `12_run_dem1_demand_sensitivity.py` is a stub (`raise
NotImplementedError` is the first line of `main()`, per the runner
inventory) that would hit this same defect regardless if the stub were
removed. No healthy runner references this function.

---

## 3. `run_dem2_growth_gas_matrix` (lines 1329–1530)

### 2A — Signature

```python
def run_dem2_growth_gas_matrix(
    econ,
    demand_cases,
    gas_cases,
    financing_configs,
    policy_configs,
    cap_path="data/cost/processed/emissions_cap.csv",
):
```
Six parameters, five with no default.

### 2B — `load_scenario()` call, quoted in full

**One call site**, lines 1416–1430:

```python
scenario = load_scenario(
    demand_level_case="served",
    demand_case=demand_case,
    gas_deliverability_case=gas_case,
    capital_case=capital,
    solar_build_case="deployment_unconstrained",
    carbon_case="no_policy",
    start_year=2025,
    end_year=MODEL_END_YEAR,
    # docs/tariff_arm.md, Steps 1-3: named tariff_case
    # threaded in, not a bare number applied post-call.
    # public_only fin_cfgs omit it (tariff unused there).
    tariff_case=fin_cfg.get("tariff_case", "tariff_legacy_95m"),
    om_case="om_central",
)
```

**Passes**: `tariff_case` — **but via `fin_cfg.get(..., "tariff_legacy_95m")`,
a silent fallback default**, not a required config field like functions
1 and 2 above. This is a milder instance of the same defect class this
whole repair targets (an unstated value can still reach the model,
silently); flagged here since it sits inside a function this task is
already editing, but the general rule (`om_case` is "the one key to
watch" per TASK.md §3) does not name `tariff_case`'s soft default —
noting it for Part 3's judgment, not pre-deciding it. Also passes
`om_case` (hardcoded literal) and the same set of non-sentinel keys as
above, with `demand_case` correctly swept (this function's actual
subject) rather than fixed.

**Omits 15**: identical list.

### 2C — In-scope substitutes?

None of the 15 has an in-scope value.

### 2D — Distinct call sites

**One.**

### 2E — Other callers

```
grep -rn "run_dem2_growth_gas_matrix" --include="*.py"
```
One caller: `scripts/13_run_dem2_growth_gas_interaction.py:349`. **No
other caller.**

### 2F — Called by any healthy runner?

**No.**

---

## 4. `run_rel2_marginal_cost_curves` (lines 1727–1781)

### 2A — Signature

```python
def run_rel2_marginal_cost_curves(
    econ,
    gas_cases,
    capital_case="moderate",
    eps_levels=None,
):
```
Four parameters, two with defaults — **`capital_case` already defaults
to `"moderate"`**, notable because it means this function already
silently picks a capital envelope unless overridden; worth carrying into
Part 3 even though `capital_case` itself is not one of the 17
sentinel-required keys.

### 2B — `load_scenario()` call, quoted in full

**One call site**, lines 1763–1772:

```python
scenario = load_scenario(
    demand_level_case="served",
    demand_case="organic_central",
    gas_deliverability_case=gas_case,
    capital_case=capital_case,
    carbon_case="no_policy",
    start_year=2025,
    end_year=MODEL_END_YEAR,
    om_case="om_central",
)
```

**Passes**: `om_case` only (hardcoded literal), of the 17. **No
`tariff_case`** — this function omits it entirely, unlike functions 1–3.

**Omits 16**: `tariff_case`, `peak_multiple_case`, `reserve_margin`,
`storage_cc_case`, `gas_build_case`, `gas_availability_case`,
`gas_retirement_case`, `gas_capex_case`, `ccgt_efficiency_case`,
`capital_recovery_case`, `dscr_case`, `us_inflation`,
`capital_treatment_case`, `concessional_envelope_case`,
`conversion_case`, `hydro_case`.

### 2C — In-scope substitutes?

None of the 16 has an in-scope value.

### 2D — Distinct call sites

**One.**

### 2E — Other callers

```
grep -rn "run_rel2_marginal_cost_curves" --include="*.py"
```
One caller: `scripts/10_run_rel2_marginal_cost.py:239`. **No other
caller.**

### 2F — Called by any healthy runner?

**No.**

---

## 5. `run_rel1_feasibility_matrix` (lines 2721–2896)

### 2A — Signature

```python
def run_rel1_feasibility_matrix(
    econ,
    gas_cases,
    solar_build_cases,
    capital_case="moderate",
    eps_levels=None,
):
```
Five parameters, three with no default. Same `capital_case="moderate"`
default note as function 4.

### 2B — `load_scenario()` call, quoted in full

**One call site**, lines 2771–2781:

```python
scenario = load_scenario(
    demand_level_case="served",
    demand_case="organic_central",
    gas_deliverability_case=gas_case,
    capital_case=capital_case,
    solar_build_case=solar_build_case,
    carbon_case="no_policy",
    start_year=2025,
    end_year=MODEL_END_YEAR,
    om_case="om_central",
)
```

**Passes**: `om_case` only (hardcoded literal). **No `tariff_case`.**

**Omits 16**: same list as function 4.

The self-documenting comment TASK.md attributes to this function
(lines 2791–2797) is confirmed present, quoted in full: *"This
`load_scenario()` call is also already missing
`peak_multiple_case`/`reserve_margin`/`storage_cc_case`/
`gas_build_case`/`gas_availability_case` (required since 2.6/2.9/2.10)
and would raise before reaching here regardless."*

**Worth flagging beyond 2A–2F**: this function calls `build_model(...,
reliability_max_unserved_fraction=eps, reliability_mode="total", ...)`
(line 2841) — the **deprecated** parameter names (still functional via
alias, per the runner inventory's earlier finding), and **`mode="total"`**,
which `docs/methodology_genself_backstop.md` §5.2 records as tested and
rejected in favour of `"annual"` (a horizon-total bound is satisfied by
early-year slack and stops discriminating between service standards).
This is a second, independent staleness in this function beyond the
`load_scenario()` defect — **not part of this task's scope** (TASK.md
§3's eventual rules say "do not change any computation"), so not
corrected here; recorded in the closing list.

### 2C — In-scope substitutes?

None of the 16 has an in-scope value.

### 2D — Distinct call sites

**One** `load_scenario()` call. (Two `build_model()` calls exist inside
the function — one VoLL-only baseline, one per-eps — but both consume
the *same* `scenario` dict built once; only one `load_scenario()` call
needs repair.)

### 2E — Other callers

```
grep -rn "run_rel1_feasibility_matrix" --include="*.py"
```
One caller: `scripts/09_run_rel1_feasibility.py:126`.
`compute_feasibility_threshold()` (`optimize_experiments.py:2899`)
processes this function's output rows, does not call it. **No other
caller.**

### 2F — Called by any healthy runner?

**No.**

---

## 6. `run_gas_regime_ndc_matrix` (lines 2958–3165)

### 2A — Signature

```python
def run_gas_regime_ndc_matrix(
    econ,
    ndc_cap_scenario,
    gas_cases,
    financing_regime="traditional",
    capital_case="moderate",
    cap_path="data/cost/processed/emissions_cap.csv",
):
```
Six parameters, three with no default. `financing_regime` and
`capital_case` both silently default.

### 2B — `load_scenario()` call, quoted in full

**One call site**, lines 3015–3024:

```python
scenario = load_scenario(
    demand_level_case="served",
    demand_case="organic_central",
    gas_deliverability_case=gas_case,
    capital_case=capital_case,
    carbon_case="no_policy",
    start_year=2025,
    end_year=MODEL_END_YEAR,
    om_case="om_central",
)
```

**Passes**: `om_case` only (hardcoded literal). **No `tariff_case`**
(this function is public-only/EaaS-agnostic via the separate
`financing_regime` parameter set post-call, line 3026 —
`scenario["financing_regime"] = financing_regime` — tariff is simply
never threaded at all, live even for the `financing_regime="eaas"`
comparison arm `07_` uses it for).

**Omits 16**: same list as functions 4/5.

This function's own docstring **already contains a self-documenting
note** (lines 2974–2981), quoted in full: *"NOTE: this function's own
`load_scenario()` call (below) is already missing several OTHER
required arms (`peak_multiple_case`, `reserve_margin`,
`storage_cc_case`, `gas_build_case`, `gas_availability_case`,
`gas_retirement_case`, `gas_capex_case`, `ccgt_efficiency_case`, and now
`capital_recovery_case`/`dscr_case`/`us_inflation`/`capital_treatment_case`)
and already raises before reaching this point -- pre-existing dead code
(documented in `docs/phase_2_10_gas_availability.md` and later phase
reports), not touched further here."* This is the second
self-documenting instance — see the correction note at the top of this
report. It also independently confirms 13 of the 16 by name (missing
only `concessional_envelope_case`, `conversion_case`, `hydro_case`,
`tariff_case` from its own list — this inventory's 2B count of 16 is the
complete, current set).

### 2C — In-scope substitutes?

None of the 16 has an in-scope value.

### 2D — Distinct call sites

**One.**

### 2E — Other callers — CORRECTED FROM TASK.md

```
grep -rn "run_gas_regime_ndc_matrix" --include="*.py"
```
**Three callers**, not two: `scripts/06_run_gas1_shadow_benchmarks.py:217`,
`scripts/07_run_gas2_eaas_gas_relief.py:326` and `:346` (two call
sites in that one file), `scripts/08_run_gas3_regime_ndc_feasibility.py:261`
and `:292` (two call sites). `src/scenarios.py` contains three comments
naming this function as a naming precedent for a `1.10` literal
(lines 1794, 1982, 3119) — confirmed these are comments, not calls (no
`run_gas_regime_ndc_matrix(` invocation syntax at any of the three).

### 2F — Called by any healthy runner?

**No.** All three real callers (`06_`, `07_`, `08_`) are dead, per the
runner inventory (Group 3 / Group 2-boundary). None of the eight healthy
runners named in TASK.md §2F appear anywhere in the caller search.

---

## General notes applying to all six

- **`om_case` is passed in every one of the six, not omitted** — but as
  the hardcoded literal `"om_central"`, never as a parameter. TASK.md §3
  singles this out as "the one key to watch" for Part 3: converting it
  from a literal to a required parameter (not defaulted to `om_central`)
  is explicitly instructed there, not merely "add a missing key" the way
  the other 14-15 are.
- **`tariff_case` status varies across the six** and should not be
  treated uniformly in Part 3:
  - Already a required, caller-supplied config field: functions 1
    (`run_rel3_financing_frontier`) and 2 (`run_dem1_demand_sensitivity`).
  - Present but silently defaulted (`fin_cfg.get(..., "tariff_legacy_95m")`):
    function 3 (`run_dem2_growth_gas_matrix`).
  - Entirely absent: functions 4, 5, 6 (`run_rel2_marginal_cost_curves`,
    `run_rel1_feasibility_matrix`, `run_gas_regime_ndc_matrix`).
- **Every one of the fifteen-or-sixteen omitted keys is omitted for
  every function that omits it** — there is no case anywhere in the six
  where a key is *partially* threaded (e.g. read from a local but not
  passed). The defect is uniform: a value is either fully absent or
  fully present as a caller-supplied field.
- **No in-scope substitute exists for any omitted key in any of the six
  functions** — 2C's answer is uniformly "needs a new parameter," with
  the qualification (register L9) that a project-wide convention exists
  for at least `reserve_margin` (`"nirp_spinning"`, used identically as
  a literal in ~11-20 other runners) that Part 3 may want to adopt as
  the chosen literal for functions that fix rather than expose it —
  that is a Part 3 decision, not asserted here.
- **No healthy runner (`01_`, `01b_`, `02_run_eaas`, `02_run_ndc_caps`,
  `03_`, `22_`, `23_`, `26_`) calls any of the six functions**, directly
  or transitively, confirmed by six separate whole-tree greps. **The
  repair is confirmed safe in principle against TASK.md's core
  requirement** ("if an edit would touch a code path a working runner
  exercises, STOP") — no STOP condition triggered by 2F for any
  function. This is a finding for Part 3 to rely on, not a decision to
  proceed with Part 3 now.

---

## For the limitations register (drafted, not applied — Part 3 not released)

**Numbering check performed, not assumed.** Read
`docs/limitations_register.md`: `grep -n "^## L" docs/limitations_register.md`
confirms L1 through **L26** exist, each exactly once, ascending, no
gaps (consistent with the runner inventory's own established fact).
Next free number is **L27**. `docs/limitations_register.md` was **not
edited** — outside this task's whitelist (`src/optimize_experiments.py`
only).

> ## L27 — Six shared analysis functions carried a stale `load_scenario()` calling convention
>
> **Status:** LIVE — inventoried, not yet repaired (Part 3 of this
> repair is released separately)
>
> **What.** Six functions in `src/optimize_experiments.py`
> (`run_gas_regime_ndc_matrix`, `run_rel1_feasibility_matrix`,
> `run_rel2_marginal_cost_curves`, `run_rel3_financing_frontier`,
> `run_dem1_demand_sensitivity`, `run_dem2_growth_gas_matrix`) each
> carry their own `load_scenario()` call that omits 15 or 16 of
> `load_scenario()`'s 17 sentinel-required keys — the same keys every
> healthy runner in this project (`01_`, `01b_`, `02_run_eaas`,
> `02_run_ndc_caps`, `03_`, `22_`, `23_`, `26_`) threads through in
> full. Every call raises `TypeError` before any solve.
>
> **What it meant.** Six research questions have been unanswerable
> since roughly the phase where the keys they omit were introduced as
> required (2.6 through 2.13, by the functions' own internal dating):
> GAS-1 (gas shadow price vs. export/domestic benchmarks), GAS-2 (EaaS
> gas-scarcity relief), GAS-3 (gas decline shape vs. level), REL-1
> (feasibility threshold determinant), REL-2 (marginal cost of
> reliability), REL-3 (EaaS financing-frontier improvement), DEM-1
> (served-vs-latent demand sensitivity — additionally a stub,
> `raise NotImplementedError`), and DEM-2 (demand-growth × gas
> interaction) — eight questions across six functions, since
> `run_gas_regime_ndc_matrix` alone serves GAS-1, GAS-2, and GAS-3.
>
> **How it was found.** Not by any single runner failing in isolation —
> a full runner inventory (`docs/runner_inventory.md`, 32 runners plus
> associated library code) that read every runner's `load_scenario()`
> call, or traced it to the shared function it delegates to, found the
> same 15-16-key omission recurring identically across what first
> looked like twelve independent failures. Two of the six functions
> (`run_rel1_feasibility_matrix`, `run_gas_regime_ndc_matrix`) already
> carried inline comments self-diagnosing the defect at the point it
> occurs — evidence the pattern was noticed locally, previously, but
> never traced to its full extent or fixed at the source.
>
> **Why this is a different defect class from twelve independent
> bugs.** A copy-pasted calling convention, wrong once and propagated,
> has a different repair shape than twelve unrelated defects would:
> fixing six functions (plus, separately, nine runners that own the
> identical defect directly with no shared function involved) resurrects
> every dependent runner in one pass, and a future addition to
> `load_scenario()`'s required set will re-break all six identically
> unless the fix also converts these call sites to the same
> fail-loudly-on-omission discipline the rest of the project uses
> (`docs/methodology_protocols.md` §1). Treating this as "twelve broken
> scripts" would have led to twelve separate, redundant repairs;
> treating it as "one convention, six library call sites, nine direct
> call sites" is both more accurate and considerably cheaper to fix.
>
> **Related.** Register **L10**'s dead-runner list was confirmed
> incomplete by the same inventory: it names `05`, `06`, `07`, `09b`
> (both), `15`, `17`, `18`, `19`, `20`, but not `10`, `24`, `25`, or
> `99c` — all four independently confirmed dead for the same or an
> analogous reason. L10 itself notes it "predates the O&M work and is
> likely incomplete"; this entry and the runner inventory are the
> confirmation.
>
> **What would resolve it.** This repair, Part 3 (released separately
> from this inventory step): thread every omitted key through each of
> the six functions as a required parameter with no default, per
> `docs/methodology_protocols.md` §1's sentinel discipline — except
> `om_case`, already present as a hardcoded literal, which must be
> promoted from literal to required parameter specifically so no future
> caller can silently revert the O&M correction. Then, separately (out
> of this repair's scope), the nine directly-broken runners and the
> `NotImplementedError` stub each need their own repair to actually
> call the fixed functions/`load_scenario()` correctly.

---

## Encountered but not asked about

- **`run_rel1_feasibility_matrix` calls `build_model(...,
  reliability_max_unserved_fraction=eps, reliability_mode="total",
  ...)`** — the deprecated parameter names (functional via alias) and,
  more substantively, `mode="total"`, which
  `docs/methodology_genself_backstop.md` §5.2 records as tested and
  rejected project-wide in favour of `"annual"`. This is a second,
  independent staleness in this function beyond the `load_scenario()`
  defect, and TASK.md's Part 3 rules (once released) will say "do not
  change any computation" — a mode change would be exactly that, so
  this is flagged for whoever eventually revisits this function's
  *behaviour*, not something Part 3 of this repair should touch.
- **`run_dem2_growth_gas_matrix`'s `tariff_case=fin_cfg.get(...,
  "tariff_legacy_95m")`** is a silent-default pattern milder than but
  structurally the same as the defect class this whole repair targets.
  Not one of the 17 sentinel-required keys' omissions this task is
  scoped to fix, and TASK.md's Part 3 rules do not name it — flagged in
  case Part 3's author wants to tighten it while already inside this
  function, but not pre-decided here.
- **`capital_case` defaults silently to `"moderate"`** in three of the
  six functions (`run_rel2_marginal_cost_curves`,
  `run_rel1_feasibility_matrix`, `run_gas_regime_ndc_matrix`), and
  `financing_regime` defaults to `"traditional"` in
  `run_gas_regime_ndc_matrix`. Neither is one of the 17 sentinel keys,
  so neither is in this repair's scope, but both are the same silent-
  default shape at a smaller scale — noted for completeness, not
  flagged as needing action here.
- This task's whitelist is `src/optimize_experiments.py` only; the nine
  runners that own the missing-keys defect directly (`05_`, `15_`,
  `18_`, `19_`, `20_`, `04_`, `24_`, `25_`, `99c_`, per the runner
  inventory) are out of scope for both this step and the eventual Part
  3, and will still fail after Part 3 lands — Part 3 only fixes the six
  shared functions, not their sibling runners with the same defect.
  Restated here so it is not lost between this report and whichever
  task picks up runner repair next.

**Then STOP, per TASK.md. Part 3 (the repair) is not started. Awaiting
release.**

---

## PART 3 — REPAIR

Released and complete. Edited only `src/optimize_experiments.py`
(the six functions) and `docs/limitations_register.md` (the L27
append, below) — exactly the whitelist. `git diff --stat
src/optimize_experiments.py`: 327 insertions, 34 deletions, one file.
Nothing in `scripts/` touched. No script, solver, or `smoke_all.sh`
run. No commit, no staging.

Three corrections from TASK.md §1 applied throughout: `om_case`
promoted from literal to required parameter (not "added" — it was
never missing); `tariff_case` treated three different ways per
function, not uniformly; `run_gas_regime_ndc_matrix`'s caller list
corrected to three runners (`06_`×1, `07_`×2, `08_`×2).

Every `py_compile` check below was run after each function's edit, in
sequence, so a syntax error would have been caught at the function
that introduced it, not at the end.

---

### 1. `run_rel3_financing_frontier`

**New signature:**

```python
def run_rel3_financing_frontier(
    econ,
    gas_cases,
    policy_configs,
    peak_multiple_case,
    reserve_margin,
    storage_cc_case,
    gas_build_case,
    gas_availability_case,
    gas_retirement_case,
    gas_capex_case,
    ccgt_efficiency_case,
    capital_recovery_case,
    dscr_case,
    us_inflation,
    capital_treatment_case,
    concessional_envelope_case,
    conversion_case,
    hydro_case,
    om_case,
    eps_levels=None,
    cap_path="data/cost/processed/emissions_cap.csv",
):
```

**Call site 1 of 2 (`scenario_pub`) — old:**
```python
scenario_pub = load_scenario(
    demand_level_case="served",
    demand_case="organic_central",
    gas_deliverability_case=gas_case,
    capital_case=pub_capital,
    carbon_case="no_policy",
    start_year=2025,
    end_year=MODEL_END_YEAR,
    tariff_case=tariff_case,
    om_case="om_central",
)
```
**New:** identical through `tariff_case=tariff_case`, then
`om_case="om_central"` replaced by the 15 new keyword arguments
(each `key=key`, matching the new parameter names) followed by
`om_case=om_case`.

**Call site 2 of 2 (`scenario_eaas`)** — same old/new pattern,
identical fix applied.

**Caller must now additionally pass:** `peak_multiple_case`,
`reserve_margin`, `storage_cc_case`, `gas_build_case`,
`gas_availability_case`, `gas_retirement_case`, `gas_capex_case`,
`ccgt_efficiency_case`, `capital_recovery_case`, `dscr_case`,
`us_inflation`, `capital_treatment_case`,
`concessional_envelope_case`, `conversion_case`, `hydro_case`,
`om_case` — 16 new required arguments (`tariff_case` was already
required via `policy_configs`, unchanged).

---

### 2. `run_dem1_demand_sensitivity`

**New signature:**

```python
def run_dem1_demand_sensitivity(
    econ,
    demand_level_cases,
    ndc_configs,
    peak_multiple_case,
    reserve_margin,
    storage_cc_case,
    gas_build_case,
    gas_availability_case,
    gas_retirement_case,
    gas_capex_case,
    ccgt_efficiency_case,
    capital_recovery_case,
    dscr_case,
    us_inflation,
    capital_treatment_case,
    concessional_envelope_case,
    conversion_case,
    hydro_case,
    om_case,
    cap_path="data/cost/processed/emissions_cap.csv",
):
```

**Call site — old:**
```python
scenario = load_scenario(
    demand_level_case=demand_level,
    demand_case="organic_central",
    gas_deliverability_case="baseline",
    capital_case=fin_params["capital_case"],
    solar_build_case="deployment_unconstrained",
    carbon_case="no_policy",
    start_year=2025,
    end_year=MODEL_END_YEAR,
    tariff_case=tariff_case,
    om_case="om_central",
)
```
**New:** identical through `tariff_case=tariff_case`, then the same
15 new keyword arguments, then `om_case=om_case`. Also updated the
now-stale self-documenting comment immediately below this call (was:
"already missing... would raise before reaching here regardless";
now: records the repair and points at this task's report).

**Caller must now additionally pass:** the same 16 as function 1.

---

### 3. `run_dem2_growth_gas_matrix`

**New signature:**

```python
def run_dem2_growth_gas_matrix(
    econ,
    demand_cases,
    gas_cases,
    financing_configs,
    policy_configs,
    peak_multiple_case,
    reserve_margin,
    storage_cc_case,
    gas_build_case,
    gas_availability_case,
    gas_retirement_case,
    gas_capex_case,
    ccgt_efficiency_case,
    capital_recovery_case,
    dscr_case,
    us_inflation,
    capital_treatment_case,
    concessional_envelope_case,
    conversion_case,
    hydro_case,
    om_case,
    cap_path="data/cost/processed/emissions_cap.csv",
):
```

**Call site — old:**
```python
scenario = load_scenario(
    demand_level_case="served",
    demand_case=demand_case,
    gas_deliverability_case=gas_case,
    capital_case=capital,
    solar_build_case="deployment_unconstrained",
    carbon_case="no_policy",
    start_year=2025,
    end_year=MODEL_END_YEAR,
    # docs/tariff_arm.md, Steps 1-3: named tariff_case
    # threaded in, not a bare number applied post-call.
    # public_only fin_cfgs omit it (tariff unused there).
    tariff_case=fin_cfg.get("tariff_case", "tariff_legacy_95m"),
    om_case="om_central",
)
```
**New:** `tariff_case=fin_cfg.get("tariff_case", "tariff_legacy_95m")`
tightened to `tariff_case=fin_cfg["tariff_case"]` per TASK.md §3.2 —
an omitted key in a caller's `fin_cfg` now raises `KeyError` instead
of silently taking the legacy arm. The explanatory comment above it
rewritten to say the key is required, not defaulted (old text
retracted, not just appended to). Then the same 15 new keyword
arguments, then `om_case=om_case`. Docstring's `tariff_case` note
("public_only defaults to tariff_legacy_95m") also corrected to state
it is now required in every config, including `public_only`.

**Caller must now additionally pass:** `peak_multiple_case`,
`reserve_margin`, `storage_cc_case`, `gas_build_case`,
`gas_availability_case`, `gas_retirement_case`, `gas_capex_case`,
`ccgt_efficiency_case`, `capital_recovery_case`, `dscr_case`,
`us_inflation`, `capital_treatment_case`,
`concessional_envelope_case`, `conversion_case`, `hydro_case`,
`om_case` (16 new function parameters) — **and every entry in
`financing_configs` must now include a `"tariff_case"` key**
(previously optional for `public_only` entries; a caller relying on
the old silent fallback will now hit `KeyError`, not a function
signature error — flagged clearly since it's a different failure
mode from the other 16).

---

### 4. `run_rel2_marginal_cost_curves`

**New signature:**

```python
def run_rel2_marginal_cost_curves(
    econ,
    gas_cases,
    peak_multiple_case,
    reserve_margin,
    storage_cc_case,
    gas_build_case,
    gas_availability_case,
    gas_retirement_case,
    gas_capex_case,
    ccgt_efficiency_case,
    capital_recovery_case,
    dscr_case,
    us_inflation,
    capital_treatment_case,
    concessional_envelope_case,
    conversion_case,
    hydro_case,
    tariff_case,
    om_case,
    capital_case="moderate",
    eps_levels=None,
):
```

**Call site — old:**
```python
scenario = load_scenario(
    demand_level_case="served",
    demand_case="organic_central",
    gas_deliverability_case=gas_case,
    capital_case=capital_case,
    carbon_case="no_policy",
    start_year=2025,
    end_year=MODEL_END_YEAR,
    om_case="om_central",
)
```
**New:** identical through `end_year=MODEL_END_YEAR`, then 16 new
keyword arguments (the 15 common ones plus `tariff_case`, which this
function omitted entirely before), then `om_case=om_case`.

**Caller must now additionally pass:** `peak_multiple_case`,
`reserve_margin`, `storage_cc_case`, `gas_build_case`,
`gas_availability_case`, `gas_retirement_case`, `gas_capex_case`,
`ccgt_efficiency_case`, `capital_recovery_case`, `dscr_case`,
`us_inflation`, `capital_treatment_case`,
`concessional_envelope_case`, `conversion_case`, `hydro_case`,
`tariff_case`, `om_case` — 17 new required arguments (`capital_case`
default unchanged, out of scope).

---

### 5. `run_rel1_feasibility_matrix`

**New signature:**

```python
def run_rel1_feasibility_matrix(
    econ,
    gas_cases,
    solar_build_cases,
    peak_multiple_case,
    reserve_margin,
    storage_cc_case,
    gas_build_case,
    gas_availability_case,
    gas_retirement_case,
    gas_capex_case,
    ccgt_efficiency_case,
    capital_recovery_case,
    dscr_case,
    us_inflation,
    capital_treatment_case,
    concessional_envelope_case,
    conversion_case,
    hydro_case,
    tariff_case,
    om_case,
    capital_case="moderate",
    eps_levels=None,
):
```

**Call site — old:**
```python
scenario = load_scenario(
    demand_level_case="served",
    demand_case="organic_central",
    gas_deliverability_case=gas_case,
    capital_case=capital_case,
    solar_build_case=solar_build_case,
    carbon_case="no_policy",
    start_year=2025,
    end_year=MODEL_END_YEAR,
    om_case="om_central",
)
```
**New:** identical through `end_year=MODEL_END_YEAR`, then the same
16 new keyword arguments as function 4, then `om_case=om_case`. The
self-documenting comment immediately below (previously: "already
missing... would raise before reaching here regardless") rewritten to
record the repair, and a new note added pointing at the untouched
`reliability_mode="total"` call further down — flagged, not changed
(§4, DO NOT TOUCH).

**Caller must now additionally pass:** the same 17 as function 4.

---

### 6. `run_gas_regime_ndc_matrix`

**New signature:**

```python
def run_gas_regime_ndc_matrix(
    econ,
    ndc_cap_scenario,
    gas_cases,
    peak_multiple_case,
    reserve_margin,
    storage_cc_case,
    gas_build_case,
    gas_availability_case,
    gas_retirement_case,
    gas_capex_case,
    ccgt_efficiency_case,
    capital_recovery_case,
    dscr_case,
    us_inflation,
    capital_treatment_case,
    concessional_envelope_case,
    conversion_case,
    hydro_case,
    tariff_case,
    om_case,
    financing_regime="traditional",
    capital_case="moderate",
    cap_path="data/cost/processed/emissions_cap.csv",
):
```

**Call site — old:**
```python
scenario = load_scenario(
    demand_level_case="served",
    demand_case="organic_central",
    gas_deliverability_case=gas_case,
    capital_case=capital_case,
    carbon_case="no_policy",
    start_year=2025,
    end_year=MODEL_END_YEAR,
    om_case="om_central",
)
```
**New:** identical through `end_year=MODEL_END_YEAR`, then the same
16 new keyword arguments as functions 4/5, then `om_case=om_case`.
The docstring's self-documenting note (quoted in full in Step One)
rewritten to record the repair rather than describe the defect as
current; a second, separate stale copy of the same note further down
in the function body (missed by a first pass — found by re-grepping
the whole file for "already missing" after editing the other five
functions, see below) was also updated. Docstring's summary line
changed from "GAS-1 + GAS-3 shared runner" to "GAS-1 + GAS-2 + GAS-3
shared runner" and a `GAS-2` line added to the diagnostics-purpose
list, since `07_`'s `financing_regime="eaas"` usage (GAS-2) was
previously undocumented at the top of the function despite being one
of its three real callers.

**Caller must now additionally pass:** the same 17 as functions 4/5.

---

### Verification pass across all six

After finishing all six functions, re-grepped the whole file for the
phrase patterns that mark a stale self-documenting comment
(`"already missing"`, `"would raise before reaching"`, `"is also
missing"`). One further match survived: inside
`run_str1_storage_parameter_sweep` (line ~1783) — **not one of the
six**, confirmed by its enclosing `def` and its unrelated local
variables (`hd`, `sfrac`, storage-duration-sweep parameters). Left
untouched, correctly: it is outside this task's six-function scope
and outside the whitelist's intent. No stale copy of the comment
remains inside any of the six repaired functions.

`python3 -m py_compile src/optimize_experiments.py` — passes, run
after each of the six edits individually (six separate passing runs)
and once more after the final verification pass.

---

## No computation changed — anchors cannot have moved

**Confirmed by construction, not by running anything** (no script, no
solver, per the whitelist): every edit in Part 3 was one of exactly
three kinds —

1. Adding parameters to a function's own signature (new names bound
   at call time; nothing computed from them changes until a caller
   passes different values than the hardcoded literals used to
   supply, which no caller currently does — see below).
2. Adding keyword arguments to `load_scenario()` calls, each set to
   the identically-named new parameter, and changing
   `om_case="om_central"` to `om_case=om_case`.
3. Rewriting comments and docstrings (no executable effect).

No `build_model()`/`build_model_sliced()`/`solve_model()` call, no
Pyomo expression, no diagnostics-extraction line, and no output field
was touched anywhere in this diff (confirmed by reading the full
`git diff` before writing this report — every hunk is a parameter
list, a `load_scenario()` argument list, or a comment/docstring
block).

**Why the anchors cannot have moved.** The regression anchor
(`21,811,042,472.821114 / 45,502.9754 MWh`, `om_none`) and the
central figure (`27.954 bn`) are both produced by `01b_run_baseline_sliced.py`
and `02_run_ndc_caps.py` respectively (per `findings_frozen.md`) —
**neither of which calls any of the six functions this task edited**,
confirmed independently in Step One (2F, six separate whole-tree
searches) and re-confirmed here by the fact that this task touched no
file under `scripts/`. A function that is never called cannot change
a result that does not call it, regardless of what its own internal
plumbing now requires.

---

## The specification for the next task

Restated plainly, per function — the parameters a caller must now
supply that it did not before:

    run_gas_regime_ndc_matrix (called by 06_ x1, 07_ x2, 08_ x2):
        peak_multiple_case, reserve_margin, storage_cc_case,
        gas_build_case, gas_availability_case, gas_retirement_case,
        gas_capex_case, ccgt_efficiency_case, capital_recovery_case,
        dscr_case, us_inflation, capital_treatment_case,
        concessional_envelope_case, conversion_case, hydro_case,
        tariff_case, om_case                              (17 new)

    run_rel1_feasibility_matrix (called by 09_):
        peak_multiple_case, reserve_margin, storage_cc_case,
        gas_build_case, gas_availability_case, gas_retirement_case,
        gas_capex_case, ccgt_efficiency_case, capital_recovery_case,
        dscr_case, us_inflation, capital_treatment_case,
        concessional_envelope_case, conversion_case, hydro_case,
        tariff_case, om_case                              (17 new)

    run_rel2_marginal_cost_curves (called by 10_):
        peak_multiple_case, reserve_margin, storage_cc_case,
        gas_build_case, gas_availability_case, gas_retirement_case,
        gas_capex_case, ccgt_efficiency_case, capital_recovery_case,
        dscr_case, us_inflation, capital_treatment_case,
        concessional_envelope_case, conversion_case, hydro_case,
        tariff_case, om_case                              (17 new)

    run_rel3_financing_frontier (called by 11_):
        peak_multiple_case, reserve_margin, storage_cc_case,
        gas_build_case, gas_availability_case, gas_retirement_case,
        gas_capex_case, ccgt_efficiency_case, capital_recovery_case,
        dscr_case, us_inflation, capital_treatment_case,
        concessional_envelope_case, conversion_case, hydro_case,
        om_case                                           (16 new;
        tariff_case was already required via policy_configs)

    run_dem1_demand_sensitivity (called by 12_, also a
    NotImplementedError stub -- the stub itself still needs removing
    separately before this function is ever actually invoked):
        peak_multiple_case, reserve_margin, storage_cc_case,
        gas_build_case, gas_availability_case, gas_retirement_case,
        gas_capex_case, ccgt_efficiency_case, capital_recovery_case,
        dscr_case, us_inflation, capital_treatment_case,
        concessional_envelope_case, conversion_case, hydro_case,
        om_case                                           (16 new;
        tariff_case was already required via ndc_configs)

    run_dem2_growth_gas_matrix (called by 13_):
        peak_multiple_case, reserve_margin, storage_cc_case,
        gas_build_case, gas_availability_case, gas_retirement_case,
        gas_capex_case, ccgt_efficiency_case, capital_recovery_case,
        dscr_case, us_inflation, capital_treatment_case,
        concessional_envelope_case, conversion_case, hydro_case,
        om_case                                           (16 new;
        tariff_case was already required via financing_configs, now
        tightened from an optional/defaulted key to a mandatory one
        within each financing_configs entry)

The nine runners that own the missing-keys defect **directly**, with
no shared function involved (`04_`, `05_`, `15_`, `18_`, `19_`,
`20_`, `24_`, `25_`, `99c_`, per the runner inventory), are
**unaffected by this repair** and still need their own fix — this
task's whitelist explicitly excludes `scripts/`.

---

## The register entry

Appended the L27 draft from the Step One report to
`docs/limitations_register.md`, **exactly as written** there (per
TASK.md §6), blockquote markers stripped, placed immediately after
L26. Verified before appending: `grep -n "^## L"` showed L1 through
L26, each once, ascending, no gaps — L27 was the correct next number.

**Resulting `grep -n "^## L" docs/limitations_register.md`:**

```
16:## L1 — Demand shape is a 2016 vintage
56:## L2 — Suppressed series is a supply shape and cannot anchor demand
78:## L3 — Firm capacity is sized on block averages, not instantaneous peak
99:## L4 — Tier 2 self-generation magnitude uncertain by ~40%
122:## L5 — Storage is structurally dominated at annual resolution
147:## L6 — EaaS channel excludes storage by construction
171:## L7 — Environment was not reproducible before 2026-08-23
196:## Lnn — one-line statement
212:## L8 — Gas fleet retirement schedule is an analyst estimate
324:## L9 — `reserve_margin` and `genset_case` are canonical constants, not swept arms
357:## L10 — Ten runners are dead code predating F1
386:## L11 — `storage_binding` diagnostic is not meaningful under the sliced builder
413:## L12 — The annual baseline has no recorded reference objective
439:## L13 — Gas deliverability arms span two epistemic families
462:## L14 — `_rebuild_years` is duplicated across the F1 branch
485:## L15 — Stale "2045" naming across four sites
506:## L16 — EaaS and public solar are physically identical in the model
552:## L17 — The bankability constraint discounts at the retired pre-F1 blend, over a horizon the objective contradicts
658:## L18 — D6 (buffer-year storage replacement capex) is CLOSED; its guard message was stale
739:## L19 — `26_run_f3_identification.py` sets two required keys internally
761:## L20 — hydro_min_flow_fraction is ASSUMED, binding, and immaterial
841:## L21 — storage_duration_hours is an explicit constant, ASSUMED, and unswept
896:## L22 — Fixed + variable O&M is now SOURCED and wired; the storage conversion inherits L21's uncertainty
978:## L23 — F1 ships two capital tranches; the design specified three
1072:## L24 — Post-2035 cap shape is a second analyst construction stacked on the apportionment rule
1133:## L25 — SUPERSEDED IN PART: DIST-5 is implemented; the direction IS established
1179:## L26 — Solar capacity factor is a national average with no siting
1217:## L27 — Six shared analysis functions carried a stale `load_scenario()` calling convention
```

L1 through L27, each exactly once, ascending, no gaps.

**One thing worth flagging about the append, not a defect.** TASK.md
§6 says to append the draft "exactly as written," and it was — its
**Status** line still reads *"LIVE — inventoried, not yet repaired
(Part 3 of this repair is released separately)"*, which is now
stale the moment this same task executes Part 3 a few paragraphs
above. Followed the explicit instruction literally rather than
updating the status to match; noted here so the (small, self-caused)
inconsistency is visible rather than silent. A one-line status update
would resolve it but was not requested and risks rewording content
TASK.md called "accurate... as written."

---

## Encountered but not asked about

- The register entry's own "Status" line reading "not yet repaired"
  immediately after this same report repairs it — see above. Not
  fixed, per explicit instruction to append it exactly as drafted.
- **`run_str1_storage_parameter_sweep`** (not one of the six) carries
  the identical missing-keys defect in its own `load_scenario()` call,
  confirmed by the same self-documenting-comment pattern found and
  left untouched during the verification pass. It has no runner caller
  found in Step One's or this session's searches (worth checking
  whether anything calls it at all before it is prioritised for
  repair).
- **`run_bottleneck_sensitivity`** (`src/optimize_experiments.py:633`,
  also not one of the six) has its own `load_scenario()` call with the
  same 8-parameter/missing-15-keys shape (`om_case="om_central"`
  hardcoded, everything else absent) — noticed while distinguishing
  it from `run_rel3_financing_frontier`'s call sites during 2F-style
  scanning for this report. Not touched: not named in TASK.md, no
  caller search performed for it, out of this task's whitelist.
- Per TASK.md §4 ("DO NOT TOUCH"), `run_rel1_feasibility_matrix`'s
  `reliability_mode="total"` call is unchanged and remains stale
  against `docs/methodology_genself_backstop.md` §5.2 — restated here
  since it is a real, live defect this repair deliberately did not
  close, not because it was forgotten.
- This report did not re-verify the regression or central anchors by
  running anything — TASK.md's own bounding forbids it here. The "no
  computation changed" argument above is a structural one (no runner
  calls the six edited functions), not an empirical one. Zeus's
  confirmation step (per TASK.md's closing line) is the actual
  empirical check.

**Then STOP.** Zeus reviews `git diff`, commits, and confirms both
anchors reproduce.
