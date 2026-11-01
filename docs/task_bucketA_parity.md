# Task report — Bucket A: required-key promotion + doc-code parity

Branch `phase-fin2-d17`. Written incrementally per TASK.md. Governing
constraint: no edit here may change any computed number. Both frozen
anchors (regression 21811042472.821114 / 45502.9754 MWh; central
21351610495.345654 / 65612.985 MWh) are unaffected by every edit made
below — none of them touch a return value, a constraint, or an
arithmetic expression.

---

## PART A — storage_duration_hours as a required key

### A1 — Inventory, and STOP

**load_scenario()'s current full signature** (`src/scenarios.py:2007-2035`):

```python
def load_scenario(
    demand_level_case: str = "served",
    demand_case: str = "organic_central",
    land_case: str = "moderate",
    capital_case: str = "moderate",
    gas_deliverability_case: str = "baseline",
    solar_build_case: str = "nirp_trajectory",
    tariff_case=_TARIFF_REQUIRED,
    carbon_case: str = "no_policy",
    blend_case: str = "blended_central",
    peak_multiple_case=_PEAK_MULTIPLE_REQUIRED,
    reserve_margin=_RESERVE_MARGIN_REQUIRED,
    storage_cc_case=_STORAGE_CC_REQUIRED,
    gas_build_case=_GAS_BUILD_REQUIRED,
    gas_availability_case=_GAS_AVAILABILITY_REQUIRED,
    gas_retirement_case=_GAS_RETIREMENT_REQUIRED,
    gas_capex_case=_GAS_CAPEX_REQUIRED,
    ccgt_efficiency_case=_CCGT_EFFICIENCY_REQUIRED,
    capital_recovery_case=_CAPITAL_RECOVERY_REQUIRED,
    dscr_case=_DSCR_REQUIRED,
    us_inflation=_US_INFLATION_REQUIRED,
    capital_treatment_case=_CAPITAL_TREATMENT_REQUIRED,
    concessional_envelope_case=_CONCESSIONAL_ENVELOPE_REQUIRED,
    conversion_case=_CONVERSION_REQUIRED,
    hydro_case=_HYDRO_REQUIRED,
    new_gas_deliverability: str = "constrained",
    start_year: int = MODEL_START_YEAR,
    end_year: int = MODEL_END_YEAR,
) -> dict:
```

**The existing required-key pattern** (`tariff_case`, the template Part
A2 would copy), quoted in full:

```python
_TARIFF_REQUIRED = object()          # scenarios.py:2004
...
def load_scenario(..., tariff_case=_TARIFF_REQUIRED, ...):   # scenarios.py:2014
    ...
    if tariff_case is _TARIFF_REQUIRED:                       # scenarios.py:2230
        raise TypeError(
            "load_scenario() requires tariff_case (docs/tariff_arm.md, "
            "Steps 1-3). Pass one of 'tariff_low' / 'tariff_central' / "
            "'tariff_high' / 'tariff_legacy_95m' / 'tariff_legacy_65m' (see "
            "tariff_scenarios()). There is no default: the retired "
            "solar_tariff_case defaulted to 65,000,000 USD/TWh and wrote it "
            "into every scenario dict silently, with no error, warning, or "
            "visible signature in the output."
        )
    if tariff_case not in tariff_scenarios():
        raise ValueError(f"Unknown tariff_case: {tariff_case}")
```

Pattern: a module-level sentinel object as the default; a `TypeError`
naming the parameter and citing the specific silent-default incident
that motivates removing the implicit default; a `ValueError` for an
unrecognised (but supplied) value.

**Fresh call-site inventory.** `grep -rlE "load_scenario\(" --include="*.py" src/ scripts/ tests/`
returns 33 files containing the string. Of those, `src/io.py`,
`src/optimize_model.py`, `src/optimize_model_sliced.py`, and the three
runner wrapper scripts `scripts/11_run_rel3_financing_frontier.py`,
`scripts/12_run_dem1_demand_sensitivity.py`,
`scripts/13_run_dem2_growth_gas_interaction.py` contain only comment or
docstring mentions of `load_scenario()` — zero actual calls (their real
calls happen inside backend functions in `src/optimize_experiments.py`,
already counted there). `src/scenarios.py` itself is the definition
site.

That leaves **25 distinct genuine-caller files**, with per-file call
counts confirmed via
`grep -cE "^\s*[a-zA-Z_][a-zA-Z0-9_.\[\]]*\s*=\s*load_scenario\("`:

| File | Call sites | Style |
|---|---|---|
| `scripts/01_run_baseline.py` | 1 | kwargs |
| `scripts/01b_run_baseline_sliced.py` | 1 | kwargs |
| `scripts/02_run_eaas.py` | 1 | kwargs |
| `scripts/02_run_ndc_caps.py` | 1 | kwargs |
| `scripts/03_run_ndc_eaas.py` | 1 | kwargs |
| `scripts/04_run_fin2_tariff_sweep.py` | 1 | kwargs |
| `scripts/05_run_fin3_capital_sweep.py` | 1 | kwargs |
| `scripts/06_run_gas1_shadow_benchmarks.py` | 1 | kwargs |
| `scripts/07_run_gas2_eaas_gas_relief.py` | 2 | kwargs |
| `scripts/15_run_pol1_ndc_comparison.py` | 1 | kwargs |
| `scripts/18_run_pareto_frontier.py` | 1 | kwargs |
| `scripts/19_run_carbon_price_sweep.py` | 1 | kwargs |
| `scripts/20_run_cf1_solar_yield_sensitivity.py` | 1 | kwargs |
| `scripts/22_salvage_sensitivity.py` | 1 | kwargs |
| `scripts/23_run_selfgen_frontier.py` | 1 | kwargs |
| `scripts/24_run_leakage_arm.py` | 1 | kwargs |
| `scripts/25_run_obligation_trajectory.py` | 1 | kwargs |
| `scripts/26_run_f3_identification.py` | 1 | kwargs |
| `scripts/27_run_hydro_minflow_sensitivity.py` | 1 | **`**CENTRAL_ARGS` dict-unpack** |
| `scripts/99c_run_genset_arm_sensitivity.py` | 1 | kwargs |
| `probe_2_6_structure.py` | 1 | kwargs |
| `scripts/deferred/17_run_monte_carlo.py` | 1 | kwargs |
| `scripts/retired/09b_run_rel1_mode_sensitivity.py` | 1 | kwargs |
| `scripts/retired/09b_run_rel1_retirement_sensitivity.py` | 1 | kwargs |
| `tests/test_run.py` | 1 | kwargs |
| `src/optimize_experiments.py` | 11 (lines 641, 789, 812, 1195, 1412, 1606, 1757, 2764, 3007, 3180, 3323) | kwargs, all explicit |

**Totals: 25 distinct files, 36 individual call-site invocations.**

**Dict-unpack check.** `grep -rn "load_scenario(\*\*" --include="*.py" src/ scripts/ tests/`
finds exactly one site: `scripts/27_run_hydro_minflow_sensitivity.py:389`,
`base_scenario = load_scenario(**CENTRAL_ARGS)`. `CENTRAL_ARGS` is a
static module-level dict literal (authored in an earlier task), fully
readable — its contents are statically determinable. So the "cannot
determine statically" STOP clause is **not** triggered by this site.

**The other STOP clause is triggered.** TASK.md's ESTABLISHED section
states "22 distinct .py files call load_scenario()." Fresh inventory
finds **25 distinct files** and **36 individual call sites** — both
numbers exceed 22, under either a file-count or an invocation-count
reading.

**STOP — Part A2 and A3 are not performed.** Per TASK.md A1: "If more
than 22 sites need editing... STOP and report before editing anything.
That would change this task's shape." 25 files / 36 call sites is more
than 22 under any reading. No `storage_duration_hours` promotion, no
threading through call sites, and no removal of the `optimize_model.py:386`
/ `optimize_model_sliced.py:461` `.get()` fallbacks has been made. All
of Part A beyond this inventory is therefore **not done**, by design,
per the task's own instruction.

One further, smaller discrepancy noted for completeness (not itself a
STOP trigger): TASK.md's ESTABLISHED section states the key is
"surfaced on the scenario dict at `scenarios.py:2970`"; fresh reading
finds it at `scenarios.py:3021`. The canonical definition line,
`scenarios.py:2666` (`_storage_duration_hours = 4.0`), matches exactly.
The drift is consistent with intervening edits shifting line numbers
elsewhere in the file and does not change any finding above.

### A2, A3 — Not performed (see A1 STOP)

---

## PART B — doc-code parity

### B1 — EaaS.py financing machinery

**Call sites reconfirmed afresh** (`grep -rn "from src.EaaS import\|from src import EaaS\|import src.EaaS\|EaaS\." --include="*.py" src/ scripts/ tests/ | grep -v "^src/EaaS.py"`):
zero import sites anywhere in the tree. This is a **stronger finding
than TASK.md's stated premise**: TASK.md's ESTABLISHED section says
"`compute_eaas_deployment` has 2 [call sites]. The module is not wholly
dead — only its financing machinery is." Fresh verification finds the
**entire module unimported**, not just `_capital_recovery_factor` and
`compute_eaas_financing`:

- `_capital_recovery_factor`: zero external callers (only
  `compute_eaas_financing`'s internal call at line 266).
- `compute_eaas_financing`: zero callers anywhere.
- `compute_eaas_deployment`: zero *external* callers.
  `scripts/12_run_dem1_demand_sensitivity.py:265` defines an unrelated
  local function of the same name and never imports `src.EaaS`
  (confirmed: `grep -n "^from\|^import" scripts/12_run_dem1_demand_sensitivity.py | grep -i eaas`
  returns nothing) — a name coincidence, not a caller.

This does not change this task's required action (B1 says: do not
delete, add a docstring note) — it changes what the note must say, so
the note documents the whole-module finding rather than the
financing-only framing TASK.md assumed.

**Edit made.** `src/EaaS.py`, module docstring, appended after the
existing "Deferred Phases" block (previously ended at line ~26 with a
closing `"""`):

Old text (docstring ending):
```
Implemented Phases
------------------
Phase 1: Deployment lever (capacity rollout)
Phase 2: Financing lever (CAPEX -> service payments)

Deferred Phases
---------------
Phase 3: Reliability substitution (explicitly not modeled)
"""
```

New text:
```
Implemented Phases
------------------
Phase 1: Deployment lever (capacity rollout)
Phase 2: Financing lever (CAPEX -> service payments)

Deferred Phases
---------------
Phase 3: Reliability substitution (explicitly not modeled)

NOT CALLED BY THE LP (doc-code parity, plan 5.5)
-------------------------------------------------
This entire module is not imported anywhere in `src/` or `scripts/`
(confirmed by search: no `from src.EaaS import ...` / `import src.EaaS`
site exists in the tree) -- not just its financing machinery.
`_capital_recovery_factor` and `compute_eaas_financing` have zero call
sites (including from within this module, beyond
`compute_eaas_financing`'s own internal call to
`_capital_recovery_factor`). `compute_eaas_deployment` also has zero
EXTERNAL call sites: `scripts/12_run_dem1_demand_sensitivity.py` defines
its OWN unrelated function of the same name and never imports this
module -- a name coincidence, not a caller.

The two Pyomo LP builders that actually produce every reported result
(`src/optimize_model.py`, `src/optimize_model_sliced.py`) build EaaS
capital recovery entirely themselves, under `capital_treatment_case=
'f1_tranche'` (per-tranche CRF annuity on the split
concessional/commercial volumes) or `'pre_f1_overnight'` (overnight
capex + salvage). Neither builder reads a single symbol from this file.
This module's CRF/service-payment machinery predates F1 and was
superseded by it, not merely duplicated by it -- kept here, undeleted,
because deciding whether unused code that might still be referenced in
prose should be removed is a larger decision than this task takes (see
docs/task_bucketA_parity.md, Part B1, for the recommendation).
"""
```

**Why this cannot change a computed value.** The edit is confined to
the module docstring (a string literal never read at runtime by any
other code — no function references `__doc__`); no function body,
signature, or return statement was touched. Verified with
`python3 -m py_compile src/EaaS.py` — passes.

**Recommendation (recorded, not acted on).** Per B1's instruction, the
functions are not deleted. Given the module is now confirmed *wholly*
unimported (stronger than TASK.md's financing-only premise), the
recommendation is: if the codebase later commits to removing dead code,
this whole module is a candidate for deletion, not just
`_capital_recovery_factor` / `compute_eaas_financing` — but that
decision, and any check of whether the thesis prose references this
module by name, is out of scope here.

### B2 — storage.py sqrt-eta vs the LP

**storage.py's convention** (module docstring + `BatteryStorage.__init__`,
`charge()`, `discharge()`):

```python
self.eta_rt = round_trip_efficiency
self.eta_c = sqrt(round_trip_efficiency)
self.eta_d = sqrt(round_trip_efficiency)
...
# charge():
effective_energy = charge_energy * self.eta_c
...
# discharge():
required_soc = requested / self.eta_d
delivered = withdrawn * self.eta_d
```

i.e. √η applied on both charge and discharge (symmetric split).

**The LP's convention** (`src/optimize_model.py:866-871`,
`m.storage_energy_balance`; `src/optimize_model_sliced.py:965-971`,
`m.storage_season_balance`, same rule per season):

```python
# Discharge bounded by energy charged (after RTE)
m.storage_energy_balance = pyo.Constraint(
    T,
    rule=lambda m, t:
        m.storage_discharge[t]
        <= storage_round_trip_eff * m.storage_charge[t],
)
```

Full round-trip efficiency applied entirely on discharge; charge is
unmodified. This is a different loss attribution from storage.py's
symmetric √η/√η split (though the two give the same total round-trip
loss for one full cycle at the same `eta_rt` — what differs is which
leg carries the loss, and storage.py's stateful SOC-tracking model has
no counterpart in the LP's constraint formulation at all).

**Import check.** `grep -rn "from src.storage\|import src.storage\|BatteryStorage" --include="*.py" src/ scripts/ tests/`
finds only `src/optimize_experiments.py:34` (`from src.storage import
BatteryStorage`) and `:288` (instantiation inside
`run_deterministic_scenario()`, a separate non-Pyomo dispatch-simulation
helper). **Neither LP builder imports `storage.py` at all.**

**Edit made.** `src/storage.py`, module docstring, appended after the
"Modeling assumptions" block (previously ended at line 17):

Old text (docstring, "Modeling assumptions" block ending):
```
- Minimum state of charge (SOC) is fixed at 0 MWh
- No degradation, ramping, or intra-annual dynamics

Boundary discipline
```

New text (inserted between them):
```
- Minimum state of charge (SOC) is fixed at 0 MWh
- No degradation, ramping, or intra-annual dynamics

DOC-CODE PARITY (plan 5.5): this convention is NOT what the reported
results use. `BatteryStorage` is imported and used only by
`src/optimize_experiments.py`'s `run_deterministic_scenario()` (a
separate, non-Pyomo dispatch-simulation helper) -- confirmed neither
`src/optimize_model.py` nor `src/optimize_model_sliced.py` (the two
Pyomo LP builders that produce every reported result) imports this
module at all. Those builders apply the FULL round-trip efficiency on
DISCHARGE ONLY, charge unmodified:
`m.storage_discharge[t] <= storage_round_trip_eff * m.storage_charge[t]`
(`optimize_model.py`'s `m.storage_energy_balance`; the sliced builder's
`m.storage_season_balance` is the same rule per season) -- not this
class's symmetric √η/√η split. The two conventions give the SAME
round-trip loss for one full charge-discharge cycle at a given `eta_rt`,
but differ in how loss is attributed between the two legs, and this
class's SOC-tracking, capacity-limited, stateful behaviour has no
counterpart in the LP's constraint-based formulation at all. Neither
implementation is changed here -- changing either would move the
objective.

Boundary discipline
```

**Why this cannot change a computed value.** Docstring-only; the
`sqrt()` calls, `charge()`/`discharge()` arithmetic, and every numeric
line in the class are untouched. `BatteryStorage` is not imported by
either LP builder, so even a hypothetical behavioural change here would
not reach either anchor — but no behavioural change was made regardless.
Verified with `python3 -m py_compile src/storage.py` — passes.

### B3 — economics.solar_capex

**Quoted, before edit:**

```python
def solar_capex(capacity_mw, solar_capex_per_mw, n_years):
    """
    Planning-level annualized solar capital cost.

    Parameters
    ----------
    capacity_mw : array-like
        Installed solar capacity in MW.
    solar_capex_per_mw : float
        CAPEX in USD per MW.
    n_years : int
        Planning horizon length.

    Returns
    -------
    float
    """
    return np.sum(capacity_mw) * solar_capex_per_mw / n_years
```

**Defect.** "Annualized" implies a financial annuity (CRF-style,
discounted). The arithmetic is a bare arithmetic mean: total CAPEX
divided by the horizon length, with no discount rate and no
time-value-of-money treatment.

**Call-site check.** `grep -rn "solar_capex(" --include="*.py" src/ scripts/ tests/`
(excluding substring false positives such as `solar_capex_per_mw`) finds
exactly one call site: `src/optimize_experiments.py:304`, inside
`run_deterministic_scenario()`'s `costs = {...}` dict, alongside
`gas_cost(...)` and `unserved_energy_cost(...)`. **Not called by either
LP builder.**

**Edit made.** `src/economics.py:39-56`:

Old text (docstring only):
```python
    """
    Planning-level annualized solar capital cost.

    Parameters
    ----------
    capacity_mw : array-like
        Installed solar capacity in MW.
    solar_capex_per_mw : float
        CAPEX in USD per MW.
    n_years : int
        Planning horizon length.

    Returns
    -------
    float
    """
```

New text:
```python
    """
    Undiscounted average annual solar capital cost (total CAPEX / n_years).

    DOC-CODE PARITY (plan 5.5): this is a straight arithmetic mean, not
    an annuity -- there is no discount rate or time-value-of-money
    treatment, so "annualized" (the prior wording) overstated what the
    arithmetic does. Not called by either Pyomo LP builder
    (`src/optimize_model.py`, `src/optimize_model_sliced.py`); the only
    call site is `src/optimize_experiments.py`'s
    `run_deterministic_scenario()`, a separate non-Pyomo dispatch-
    simulation helper. The LP builders compute solar capital cost via
    their own CRF/overnight-capex machinery under `capital_treatment_case`,
    not this function.

    Parameters
    ----------
    capacity_mw : array-like
        Installed solar capacity in MW.
    solar_capex_per_mw : float
        CAPEX in USD per MW.
    n_years : int
        Planning horizon length.

    Returns
    -------
    float
    """
```

**Why this cannot change a computed value.** The `return` statement —
`np.sum(capacity_mw) * solar_capex_per_mw / n_years` — is byte-identical
before and after. Only the docstring changed. This function is not
called by either LP builder in any case. Verified with
`python3 -m py_compile src/economics.py` — passes.

---

## PART C — required writing deliverables

### C1 — For the methodology chapter

A required-key discipline means a parameter that materially affects a
model's output is declared with no default value at the point it enters
the model, so that every code path that constructs a scenario must state
the value explicitly or fail loudly with an error naming the parameter.
This model enforces the discipline through a sentinel-object pattern:
each such parameter's keyword default in `load_scenario()` is a unique
module-level marker object (e.g. `_TARIFF_REQUIRED = object()`), not a
plausible-looking literal. If a caller does not supply the parameter,
`load_scenario()` raises a `TypeError` that names the parameter, states
what value it needs, and explains why an unstated value is unsafe; if
the caller supplies a value outside the recognised set, it raises a
`ValueError`. Twenty required parameters (as of this task's inventory)
are already governed this way — `tariff_case`, `peak_multiple_case`,
`reserve_margin`, `dscr_case`, `hydro_case`, and others — each following
the identical raise-with-explanation pattern rather than a bespoke one.

The discipline exists to prevent a specific, previously-realised defect
class: a load-bearing parameter reached through `dict.get(key, DEFAULT)`
inside a builder, where `DEFAULT` was chosen once, undocumented, never
exposed to any runner's command line, and therefore invisible in every
output filename or arm tag. The canonical instance is the retired
`solar_tariff_case`, which silently defaulted to 65,000,000 USD/TWh —
every run before that defect was found used this value with no error,
warning, or visible signature that it had been used at all, and no
result could be distinguished from one run under a different tariff
assumption. `docs/limitations_register.md`'s L19 documents a live
recurrence of the same class: `26_run_f3_identification.py` resolves two
required keys (`dist5_case`, `solar_cf_case`) internally rather than
exposing them as flags, so those results cannot be swept or attributed
from the command line even though the values used are correct. A
required-key promotion closes exactly this gap: it does not change what
value is used, only whether the code can silently proceed without one
being stated.

When such a promotion is carried out, the value carried through the
promotion is deliberately frozen — the same number that every call site
already resolved to under the old default or the old local convention is
threaded through explicitly, and nothing else changes. This is what
makes the promotion its own regression test: because no computed
quantity can change (every call site receives the identical value it
already had), any discrepancy between the pre- and post-promotion output
is proof that the promotion was not, in fact, value-preserving, and can
be caught by exact reproduction of a frozen reference run rather than by
new test-writing. The promotion is a pure access-and-visibility change,
not a modelling change, and its correctness criterion — byte-identical
output — is verifiable without rerunning the model's judgement about
what the value *should* be.

Making a parameter's value explicit at every call site is a distinct
decision from making it *selectable* — i.e. exposed as a CLI flag, swept
across an arm, or written into the output filename. `storage_duration_hours`
was, in this task, a candidate for the first but not the second: the
inventory found far more call sites threading into `load_scenario()`
than the task's premise assumed (25 files / 36 invocations against a
stated 22), which is itself a reason to promote carefully rather than in
one motion — a required-key promotion done at the wrong scale risks
silently missing a caller. Separately, even where a promotion is
mechanically safe, exposing a new dimension as selectable renames every
stored result's arm tag, which breaks the provenance chain linking
already-verified results to the inputs that produced them (the same
reasoning recorded in L9 for `reserve_margin` and `genset_case`, which
are required keys satisfied by a hardcoded constant rather than a swept
arm, by deliberate choice). Explicitness and selectability therefore
answer two different questions — "is this value visible and accounted
for" versus "can this value be varied without a code change" — and a
correction task should not conflate them.

### C2 — For the limitations register

> ## L21 — storage_duration_hours is an explicit constant, ASSUMED, and unswept
>
> **Status:** LIVE
>
> **What.** `storage_duration_hours = 4.0` (`src/scenarios.py:2666`,
> comment: "Storage discharge duration (hours)") sets the storage
> capacity credit's ceiling and is used to derive storage power capacity
> from storage energy capacity in both LP builders
> (`optimize_model.py:386`, `optimize_model_sliced.py:461`, both
> currently reached via `scenario.get("storage_duration_hours", 4.0)`).
> It is not exposed on the scenario dict as a required, no-default key —
> it is set once inside `load_scenario()` and surfaced on the returned
> dict (`scenarios.py:3021`) — and it does not appear in any arm tag or
> output filename.
>
> **Classification: ASSUMED.** Determined from the code's own commentary
> at the definition site: unlike `_hydro_baseline_twh` and
> `_hydro_capacity_mw` two lines above it, which carry `# SOURCE: see
> dict entry below` comments pointing to a citation, `_storage_duration_hours`
> carries no source citation — only a note that it was "previously an
> IMPLICIT default... made explicit here" to keep the builders' fallback
> and the capacity-credit calculation from silently drifting apart. No
> derivation from another sourced quantity is shown either, so it is
> neither SOURCED nor DERIVED by the register's own definitions.
>
> **Why unresolved.** A fresh inventory for this task found the key
> threaded through 25 distinct files and 36 `load_scenario()` call sites
> — more than a prior task's estimate of 22 — so promoting it to a
> required, explicitly-threaded key (same treatment already given to
> `tariff_case`, `hydro_case`, and 18 others) was not completed in this
> task; see Part A of `docs/task_bucketA_parity.md`. It remains reached
> by `.get()` fallback in both builders.
>
> **What it controls.** The storage energy-to-power ratio used to (a)
> cap the capacity credit the LP grants to storage and (b) derive
> storage power capacity from storage energy capacity, in both LP
> builders.
>
> **Direction of bias.** Unknown/unswept, not one-directional. A shorter
> assumed duration (batteries that discharge faster relative to their
> energy content) would raise the derived power-to-energy ratio and grant
> more capacity credit per MWh of storage, favouring storage as a firm-
> capacity substitute; a longer duration would do the reverse. No arm
> currently varies it, so which direction (if any) the true value departs
> from 4.0 is not established, and no reported result has been checked
> against a different value.
>
> **What would resolve it.** Promote it to a required `load_scenario()`
> parameter following the `tariff_case` pattern (name, cite why an
> unstated value is unsafe, no default), across all 36 call sites
> identified in this task — and, separately from that promotion, decide
> whether it should also become a swept arm.

---

## Closing list — things encountered but not asked about

- TASK.md's ESTABLISHED count of 22 `load_scenario()`-calling files is
  stale; fresh count is 25 files / 36 call sites.
- TASK.md's ESTABLISHED claim that EaaS.py's `compute_eaas_deployment`
  "has 2 [call sites]" is stale; fresh count is zero external call
  sites — the whole module is unimported, not just its financing
  machinery.
- TASK.md's ESTABLISHED line number for `storage_duration_hours` being
  "surfaced on the scenario dict at scenarios.py:2970" is stale; fresh
  location is `scenarios.py:3021`.
- `optimize_model.py:386` / `optimize_model_sliced.py:461`'s `.get(..., 4.0)`
  fallbacks were not converted to direct subscripts, since A2 (the
  precondition that guarantees the key's presence) was not performed.
- No check was made of whether the thesis prose (outside this codebase)
  references `EaaS.py`'s `_capital_recovery_factor` or
  `compute_eaas_financing` by name; that would bear on whether deleting
  them is later advisable.
- `scripts/27_run_hydro_minflow_sensitivity.py`'s `CENTRAL_ARGS` is the
  only `**dict`-unpacking `load_scenario()` call site in the tree.
