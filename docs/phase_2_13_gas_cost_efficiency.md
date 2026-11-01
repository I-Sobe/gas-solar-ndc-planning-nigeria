# Phase 2.13 — Gas capex arms and CCGT efficiency arms

Scope: `src/scenarios.py`, `src/optimize_model.py`,
`src/optimize_model_sliced.py`, and the 11 runners already carrying the
other gas arms. Diagnose-and-edit only; nothing was run (no solver, no
runners, no smoke suite). Verification was limited to `python3 -m
py_compile` / `compileall` after every edit, and direct calls to
`load_scenario()` (scenario-dict construction only — no `build_model`,
`build_model_sliced`, or `solve_model` call anywhere) to confirm the
required-arm validation, the removed 2.9 raises, and the retained
`gas_ndc_17gw` trajectory raise all behave as specified.

Not modified, per the task's explicit exclusion list: peak-shape arms,
`par_block`, `within_block_peaking_factor`, `storage_cc_case`,
`gas_availability_case`, `gas_retirement_case`, the vintage CSV or its
guard, the replacement-capex mechanism, and the firm-capacity
constraint's form.

---

## STEP 0 — Inventory

### `gas_capex_per_mw`, quoted, and every consumer (pre-edit)

**`src/scenarios.py`, scenario-dict entry (pre-edit):**
```python
        # SOURCE: [PENDING] (2.9, Step 0.1 finding). No documented source,
        # no CCGT/OCGT attribution -- a bare literal that has never priced
        # an actual build (gas_add was pinned at 0 pre-2.9). Now that
        # gas_add is a free Var and new gas is declared CCGT, this figure
        # would be load-bearing for the first time; load_scenario() raises
        # if a gas_build_case that permits new capacity is selected, so it
        # cannot silently do so unsourced. See docs/phase_2_9_gas_addition.md.
        "gas_capex_per_mw": 900000,
```

**The unconditional raise (2.9), pre-edit — `src/scenarios.py`, inside
`if gas_build_case in ("gas_ndc_17gw", "gas_unconstrained"):`:**
```python
        # Its unit cost has never been sourced or attributed to a technology
        # either (2.9, Step 0.1 finding -- see the SOURCE: [PENDING] comment
        # on gas_capex_per_mw below). Unlike gas_new_efficiency this has no
        # kwarg escape hatch today -- fixing it means editing the placeholder
        # in scenarios.py directly and removing this raise. Do not silently
        # spend an unsourced number.
        raise ValueError(
            f"gas_build_case='{gas_build_case}' would allow new gas capacity "
            "to be built, but scenario['gas_capex_per_mw'] (currently a "
            "bare, unsourced 900,000 USD/MW figure with no CCGT/OCGT "
            "attribution -- 2.9, Step 0.1 finding) has not been sourced for "
            "new-build CCGT. SOURCE: [PENDING] -- only gas_build_case="
            "'gas_none' is usable until a sourced CCGT capex figure "
            "replaces the placeholder in scenarios.py and this raise is "
            "removed. See docs/phase_2_9_gas_addition.md."
        )
```
This raise fired **unconditionally** whenever `gas_build_case` was
`'gas_ndc_17gw'` or `'gas_unconstrained'` — i.e. it blocked both arms
outright, independent of (and in addition to, for `gas_ndc_17gw`) the
trajectory raise just above it.

**Every consumer** (all bare scalar reads, `scenario["gas_capex_per_mw"]`,
no `.get()` fallback):

- **`src/optimize_model.py`, line 977** — `gas_capex_npv` (initial build
  cost): `m.DF[t] * m.gas_add[t] * scenario["gas_capex_per_mw"]`.
- **`src/optimize_model.py`, line 1066** — `m.salvage_npv`, original
  vintages: `m.gas_add[t] * scenario["gas_capex_per_mw"] *
  _salvage_fraction(t, lives["gas"])`.
- **`src/optimize_model.py`, line 1086** — `m.salvage_npv`, replacement
  vintages: `m.gas_add[build_t] * scenario["gas_capex_per_mw"] *
  _salvage_fraction(rebuild_t, lives["gas"])`.
- **`src/optimize_model.py`, line 1115** — `m.cost_replacement_capex_npv`:
  `m.DF[rebuild_t] * m.gas_add[build_t] * scenario["gas_capex_per_mw"]`.
- **`src/optimize_model_sliced.py`, lines 979/1062/1082/1111** — identical
  four sites, same roles (build/salvage-original/salvage-replacement/
  replacement-capex).
- **`src/optimize_experiments.py`, lines 2273–2274** — a diagnostic-only
  reporting helper (`_capex_at(t)`, used for an NPV capex summary, not
  read by the LP itself): `if hasattr(m, "gas_add") and "gas_capex_per_mw"
  in scenario: v += float(pyo.value(m.gas_add[t])) * float(scenario["gas_capex_per_mw"])`.

All eight LP-facing sites (four per builder) are inside the
**replacement-capex mechanism**, whose FORM this phase does not touch —
only the source of the numeric value each site reads changes.

### Scalar or per-year? Any time-varying trajectory (like solar's)?

**Confirmed scalar, and confirmed no time-varying trajectory exists.**
Every consumer above reads `scenario["gas_capex_per_mw"]` as a bare
Python float, identical at every `t`/`build_t`/`rebuild_t` — unlike
`m.solar_capex_param`, which is a `pyo.Param(T, ...)` populated from
`load_solar_capex_by_year()` (a genuine per-year NREL ATB trajectory,
declining ~59% 2025→2045). `grep -rn "gas_capex_by_year\|load_gas_capex\|def.*gas_capex" src/*.py`
returns nothing — no such function exists for gas.

### `gas_new_efficiency`'s validation sites and its one functional use (pre-edit)

Four sites:

1. **`src/scenarios.py`**, inside `if gas_build_case in ("gas_ndc_17gw",
   "gas_unconstrained"):`:
   ```python
        if gas_new_efficiency is None:
            raise ValueError(
                f"gas_build_case='{gas_build_case}' allows new gas capacity, "
                "and new gas is declared CCGT (2.9, Step 1), but "
                "gas_new_efficiency is unset. SOURCE: [PENDING] -- do not "
                "reuse the 0.287 fleet figure (gas_eta); provide a sourced "
                "CCGT efficiency via load_scenario(gas_new_efficiency=...)."
            )
   ```
2. **`src/scenarios.py`**, inside the `new_gas_deliverability` block:
   ```python
    if new_gas_deliverability == "exempt" and gas_new_efficiency is None:
        raise ValueError(
            "new_gas_deliverability='exempt' sizes new capacity's exempt "
            "fuel-equivalent allowance using gas_new_efficiency, which is "
            "unset. SOURCE: [PENDING] -- provide a sourced CCGT efficiency "
            "via load_scenario(gas_new_efficiency=...)."
        )
   ```
3. **`src/optimize_model.py`**, lines 585–593, inside
   `new_gas_deliverability == "exempt"` — the builder-level mirror of (2).
4. **`src/optimize_model_sliced.py`**, lines 683–691 — identical to (3).

**One functional use** — `m.gas_exempt_fuel_allowance_twh_th`, only built
when `new_gas_deliverability='exempt'`:
```python
        m.gas_exempt_fuel_allowance_twh_th = pyo.Expression(
            T,
            rule=lambda m, t:
                m.gas_new_capacity_mw[t] * 8760 * gas_availability_factor
                / gas_new_efficiency / 1e6
        )
```
identical in both builders. `gas_new_efficiency` does **not** affect
dispatch OPEX or emissions — those remain on the shared, blended fleet
`gas_eta` (0.287) throughout, per the existing 2.9 comment ("2.9 does not
split dispatch by vintage").

---

## STEP 1 — `gas_capex_case` as a required swept arm

**New arm function**, inserted between `gas_ndc_17gw_trajectory_mw()` and
`_solar_cumulative_cap()`:

```python
def gas_capex_scenarios() -> dict[str, float]:
    """
    ... [full derivation, reproduced from the task's given NIRP-table
    arithmetic] ...
    """
    return {
        "gas_capex_low":     900_000,
        "gas_capex_central": 1_317_000,
        "gas_capex_high":    1_400_000,
    }
```
The docstring carries the full derivation verbatim from the task (NIRP
2024 Tables 19/22/23; capacity = generation / (8760 × CF); capex/MW =
investment / capacity added; year cluster 877/1,092/1,152/1,162/1,207/
1,393 USD/kW; central = cumulative 17.0bn / 12,905 MW; solar
cross-validation at 905 USD/kW), so the number is reproducible from the
docstring alone, per the task's instruction. No value in it was computed
or invented here — all figures were given.

**Sentinel** (mirrors `_GAS_RETIREMENT_REQUIRED`):
```python
_GAS_CAPEX_REQUIRED = object()
```

**Signature change**: `gas_capex_case=_GAS_CAPEX_REQUIRED` added to
`load_scenario()`.

**Validation** (replaces the removed 2.9 block — see next section):
```python
    if gas_capex_case is _GAS_CAPEX_REQUIRED:
        raise TypeError(...)
    if gas_capex_case not in gas_capex_scenarios():
        raise ValueError(f"Unknown gas_capex_case: {gas_capex_case}")
    _gas_capex_per_mw = gas_capex_scenarios()[gas_capex_case]
```
No `None`-valued arm exists in this arm-set (unlike `gas_avail_central`),
so there is no pending-raise branch — every `gas_capex_case` value is
immediately usable.

**Scenario dict**: `"gas_capex_case": gas_capex_case` added;
`"gas_capex_per_mw": 900000` (literal) → `"gas_capex_per_mw":
_gas_capex_per_mw` (resolved from the arm). `gas_capex_low` (900,000)
reproduces the pre-2.13 literal exactly, so every pre-2.13 result stays
attributable under an explicit, named arm.

### The 2.9 unconditional raise — removed; the trajectory raise — retained

**Removed** (see the exact deletion in the diff below): the block quoted
in Step 0 above that raised whenever `gas_build_case in ("gas_ndc_17gw",
"gas_unconstrained")`, blocking both arms unconditionally on unsourced
capex. It is satisfied — every `gas_capex_case` now supplies a sourced
(or explicitly declared-bound) figure.

**Retained, untouched** — `gas_ndc_17gw`'s own trajectory raise:
```python
    gas_new_capacity_cap_mw = None
    if gas_build_case == "gas_ndc_17gw":
        gas_new_capacity_cap_mw = gas_ndc_17gw_trajectory_mw(years)
        if gas_new_capacity_cap_mw is None:
            raise ValueError(
                "gas_build_case='gas_ndc_17gw' selected, but its cumulative "
                "gas-capacity trajectory is SOURCE: [PENDING] (2.9, Step "
                "2): the interpolation method from the current 11,000 MW "
                "fleet to 17,000 MW by 2035, and the treatment after 2035, "
                "have not been decided -- see gas_ndc_17gw_trajectory_mw() "
                "and docs/phase_2_9_gas_addition.md. Select "
                "'gas_unconstrained' or 'gas_none' until sourced."
            )
```
Not one character of this block was changed. **Verified directly**:
`gas_unconstrained` now builds a scenario successfully (previously always
raised); `gas_ndc_17gw` still raises, on the trajectory message, not the
old capex message (`gas_ndc_17gw_trajectory_mw()` still returns `None` —
that remains genuinely open, per the task's explicit instruction to leave
it).

---

## STEP 2 — `ccgt_efficiency_case` as a required swept arm

**New arm function**, inserted immediately after `gas_capex_scenarios()`:

```python
def ccgt_efficiency_scenarios() -> dict[str, float]:
    """ ... """
    return {
        "ccgt_eff_iso":     0.5445,
        "ccgt_eff_central": 0.5173,
        "ccgt_eff_low":     0.5010,
    }
```

The docstring documents, verbatim from the task:
- **Basis**: HHV throughout, matching `gas_eta = 0.287` (itself HHV,
  1036 Btu/scf). NREL ATB and EIA report fossil heat rates on an HHV
  basis, so **no LHV→HHV conversion is applied** — doing so on top of an
  already-HHV heat rate would double-count the fuel-heating-value
  conversion.
- **`ccgt_eff_iso` = 0.5445**: SOURCE NREL ATB 2024, new-build H-class
  NGCC, heat rate 6,266 Btu/kWh HHV (supporting EIA/Sargent & Lundy 2023);
  `3,412 / 6,266 = 0.5445`.
  **[Corrected TASK 73: the source is EIA / Sargent & Lundy SL-018001
  Rev A, Table 5-1, not the NREL ATB; the ATB workbook does not contain
  6,266 (TASK 72A). The text above is the record of what was shipped.]**
- **`ccgt_eff_central` = 0.5173**: ISO figure with a 5% Nigerian ambient
  derate — the derate itself is a **declared assumption**, not from ATB
  (ATB is ISO 15°C, US Midwest basis; Nigerian ambient ≈27–35°C; gas
  turbine output/efficiency both fall with rising inlet temperature).
- **`ccgt_eff_low` = 0.5010**: ISO figure with a high (8%) ambient derate,
  same declared-assumption caveat.
- The ambient derate is **separate and additional** to the HHV-basis
  point above — one is a unit-convention check, the other a site-condition
  adjustment.
- **Lever size, recorded and verified**: `ccgt_eff_central / gas_eta =
  0.5173 / 0.287 = 1.802` (verified: `0.5173/0.287 = 1.8024...`). One new
  CCGT MWh consumes `gas_eta / ccgt_eff_central = 0.287/0.5173 = 55.5%`
  as much fuel as the existing fleet average — **≈44% less gas per MWh**
  — a large lever because `m.gas_balance` (the fuel budget) is a binding
  deliverability constraint in most scenarios.

**Sentinel**: `_CCGT_EFFICIENCY_REQUIRED = object()`.

**Wiring**: `gas_new_efficiency: float | None = None` **removed** from
`load_scenario()`'s signature; `ccgt_efficiency_case=_CCGT_EFFICIENCY_REQUIRED`
added in its place. Validation:
```python
    if ccgt_efficiency_case is _CCGT_EFFICIENCY_REQUIRED:
        raise TypeError(...)
    if ccgt_efficiency_case not in ccgt_efficiency_scenarios():
        raise ValueError(f"Unknown ccgt_efficiency_case: {ccgt_efficiency_case}")
    _gas_new_efficiency = ccgt_efficiency_scenarios()[ccgt_efficiency_case]
```
`"ccgt_efficiency_case"` and `"gas_new_efficiency"` (now `_gas_new_efficiency`,
never `None`) both added/updated on the scenario dict.

**Raise-if-None removed, arm-membership validation kept**, per the task's
explicit instruction. All four sites from Step 0:

1. `src/scenarios.py`'s `gas_build_case` block: the `if gas_new_efficiency
   is None: raise` deleted entirely (see the combined diff below — this
   comment/raise pair was removed together with the 2.9 capex raise, since
   both lived in the same `if gas_build_case in (...)` block).
2. `src/scenarios.py`'s `new_gas_deliverability` block: the `if
   new_gas_deliverability == "exempt" and gas_new_efficiency is None:
   raise` deleted; only the arm-membership check
   (`new_gas_deliverability not in ("constrained", "exempt")`) remains.
3. `src/optimize_model.py`: replaced
   ```diff
   -        gas_new_efficiency = scenario.get("gas_new_efficiency", None)
   -        if gas_new_efficiency is None:
   -            raise ValueError(...)
   +        gas_new_efficiency = scenario["gas_new_efficiency"]
   ```
4. `src/optimize_model_sliced.py`: identical change.

The safety net for a hand-built scenario is now the **presence guard on
`ccgt_efficiency_case`** (added in Step 3, unconditional — see below),
not a value-level check on the derived `gas_new_efficiency` number.

---

## STEP 3 — Wire and tag

**Builder-side presence validation** added to both `src/optimize_model.py`
and `src/optimize_model_sliced.py`, immediately after the existing
`gas_retirement_case` guard, unconditional (matching the existing
pattern — not nested inside the `exempt` branch):
```python
    if scenario.get("gas_capex_case", None) is None:
        raise ValueError(...)
    if scenario.get("ccgt_efficiency_case", None) is None:
        raise ValueError(...)
```

Two stale-comment fixes made in passing (both already flagged as stale in
this phase's own edits, not new issues): the `gas_retirement_case` guard's
comment in both builders referenced a now-nonexistent "SOURCE: [PENDING]
raise for gas_retire_sourced" (that arm was removed in 2.12) — reworded to
"the removed-name raise for 'gas_retire_sourced', 2.12". This was not
requested by this task but was directly adjacent to the code being edited
and demonstrably wrong; left unfixed it would misdescribe a check this
report also relies on elsewhere.

`--gas-capex-case` and `--ccgt-efficiency-case` added as required CLI
arguments to all 11 runners, each with a
`GAS_CAPEX_CASE_CHOICES`/`CCGT_EFFICIENCY_CASE_CHOICES` tuple, wired
through to `load_scenario(...)`, and appended to every output tag /
summary dict / JSON row that already carried `gas_retirement_case`:

`scripts/01_run_baseline.py`, `scripts/01b_run_baseline_sliced.py`,
`scripts/02_run_eaas.py`, `scripts/02_run_ndc_caps.py`,
`scripts/03_run_ndc_eaas.py`, `scripts/04_run_fin2_tariff_sweep.py`,
`scripts/22_salvage_sensitivity.py`, `scripts/23_run_selfgen_frontier.py`,
`scripts/24_run_leakage_arm.py`, `scripts/25_run_obligation_trajectory.py`,
`scripts/99c_run_genset_arm_sensitivity.py`.

Helper-function threading beyond the generic pattern: `run_arm()` (22),
`make_scenario()`/`solve_point()` (23), and `solve_one()` (24, 25, 99c)
all gained two new parameters, each threaded through to
`load_scenario(...)` and (where the pattern already did so for
`gas_retirement_case`) folded into a `base`/summary dict.

Verified with `grep -c` per file: `gas_retirement_case`, `gas_capex_case`,
and `ccgt_efficiency_case` occur the exact same number of times in every
one of the 11 files (4, 4, 4, 4, 4, 3, 6, 8, 6, 6, 7 respectively),
confirming symmetric wiring. All 11 files, plus `src/scenarios.py` and
both builders, pass `python3 -m py_compile` / `compileall`.

---

## STEP 4 close-out

### Confirmation: both 2.9 raises removed; trajectory raise retained

Verified directly (scenario-dict construction only, no model build):

```
gas_unconstrained OK. gas_capex_per_mw= 1317000  gas_new_efficiency= 0.5173
ratio ccgt/gas_eta = 1.802439024390244
gas_ndc_17gw still correctly raises (trajectory pending): gas_build_case='gas_ndc_17gw' selected, but its cumulative gas-capacity trajecto...
missing gas_capex_case correctly raises: load_scenario() requires gas_capex_case (2.13, Step 1). Pass one of 'gas_capex_l...
```

- The unconditional capex-block raise (2.9) is gone: `gas_unconstrained`
  now builds successfully.
- The unconditional efficiency-unset raise (2.9, tied to the same
  `gas_build_case` block) is gone for the same reason.
- `gas_ndc_17gw`'s own trajectory raise fires exactly as before — untouched.
- The computed lever ratio (1.8024) matches the task's stated 1.80 figure
  exactly.

### Full list of remaining `SOURCE: [PENDING]` markers, repo-wide (`src/*.py`, `scripts/*.py`)

Grepped after all edits (`grep -rn "SOURCE: \[PENDING\]" src/*.py
scripts/*.py`), grouped by what they gate:

| # | Location(s) | What it gates | Status |
|---|---|---|---|
| 1 | `src/scenarios.py` (`gas_avail_central` entry + docstrings, `src/optimize_model{,_sliced}.py` comments) | `gas_availability_case='gas_avail_central'` | Raises if selected (2.10, untouched — excluded from this task) |
| 2 | `src/scenarios.py` lines ~184–191 (`PEAK_NIRP_PERIOD_FACTOR`/`_SEASON_FACTOR = None`) | `peak_multiple_case='peak_nirp'` | Raises if selected (pre-2.6, untouched — peak-shape arms excluded from this task). `peak_suppressed`/`peak_unsuppressed`/`peak_high` are all sourced; only `peak_nirp` remains pending — the module comment at line 155 ("SOURCE: [PENDING] for all three arms'...") is now stale relative to this, predating when the other two were sourced. Not fixed here (peak-shape arms excluded). |
| 3 | `src/scenarios.py` `gas_ndc_17gw_trajectory_mw()` + its docstring + its raise message (~3 sites) | `gas_build_case='gas_ndc_17gw'` | Raises if selected — explicitly retained open by this task's own instruction |
| 4 | `src/scenarios.py` `gas_retire_proxy`'s docstring line ("SOURCE: [PENDING]. UNCHANGED from 2.11") | Nothing raises — `gas_retire_proxy` is fully usable, just unsourced | Untouched — excluded from this task (`gas_retirement_case`) |
| 5 | `src/scenarios.py` `capacity_credit` block: `"solar": 0.0`, `"gas": _gas_capacity_credit`, `"hydro": ...` (2.6/2.10) | Nothing raises — all three credits are usable, declared judgements | Untouched — outside this task's scope (capacity_credit's gas entry is the peak-hour firm-capacity role separated from `gas_availability_factor` in 2.10; this task only touches `gas_capex_per_mw`/`gas_new_efficiency`) |
| 6 | `src/optimize_model.py` / `src/optimize_model_sliced.py`, `within_block_peaking_factor` comment | Nothing raises — a real default (1.3448, sourced Oluwole 2016 per the 2.4 phase) is already in use | Comment is STALE (still reads "SOURCE: [PENDING]; DEFAULT 1.0..."), describing a pre-2.4 state. Flagged, not fixed: `within_block_peaking_factor` is explicitly excluded from modification in this task. |

`gas_capex_per_mw` and `gas_new_efficiency` do **not** appear in this list
any more — both are fully resolved by required arms as of this phase.

Docs-directory mentions (`docs/*.md`) were checked separately and
excluded from the table above: they are historical phase-report narrative
(`docs/phase_2_9_gas_addition.md`, `docs/phase_2_10_gas_availability.md`,
etc.), not live code markers that raise or block anything.
