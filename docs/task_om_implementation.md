# Task report — wire O&M into both builders (full implementation)

Branch `phase-fin2-d17`. Written incrementally per TASK.md.

**This task deliberately changes the objective.** Correctness is NOT
byte-identity of the central case — it is that the new `om_none` arm
reproduces the old anchors to the digit while `om_central` moves. Both
are demonstrated below by construction (om_none's rates are exact
`0.0`s, added by exact `+0.0` to the existing objective — see Part 3.2)
and are for Zeus to confirm by running both arms.

---

## PART 1 — the arm

### `om_scenarios()` — new function, `src/scenarios.py` (after `tariff_scenarios()`)

```python
def om_scenarios() -> tuple[str, ...]:
    """
    Names of the fixed + variable O&M cost arms (task_om_implementation).
    REQUIRED, no default -- see the _OM_REQUIRED sentinel above
    load_scenario().
    ...
    """
    return ("om_none", "om_low", "om_central", "om_high")
```

Shape copied from `hydro_case_scenarios() -> tuple[str, ...]` (the
closest existing precedent: a required key whose values select an
entire named data set, not a single scalar) rather than
`tariff_scenarios() -> dict[str, float]`, because `om_case` does not
resolve to one number — it resolves to a full per-technology,
per-year rate table loaded from `om_cost.csv`. `om_none` is synthetic
(zero rates for all technologies/years; not a CSV row) — the
regression arm.

### The `_OM_REQUIRED` sentinel and signature — `src/scenarios.py`

```python
# ...
_OM_REQUIRED = object()                    # scenarios.py, after _HYDRO_REQUIRED
```

```python
def load_scenario(
    ...
    hydro_case=_HYDRO_REQUIRED,
    om_case=_OM_REQUIRED,                  # new parameter
    new_gas_deliverability: str = "constrained",
    ...
```

### Validation — copied from the `tariff_case` pattern exactly

```python
if om_case is _OM_REQUIRED:
    raise TypeError(
        "load_scenario() requires om_case (task_om_implementation). "
        "Pass 'om_none' / 'om_low' / 'om_central' / 'om_high' (see "
        "om_scenarios()). There is no default: omitting it would leave "
        "fixed and variable O&M either unset (a builder KeyError) or "
        "silently reintroduce an unsourced constant, the same failure "
        "mode as the retired STORAGE_OM_PER_MWH_YR."
    )
if om_case not in om_scenarios():
    raise ValueError(f"Unknown om_case: {om_case}")
_om_cost_by_year = load_om_cost_by_year(
    om_case, start_year=start_year, end_year=end_year
)
```

Placed immediately after the `hydro_case` validation block
(`scenarios.py`, before the `new_gas_deliverability` check) — same
"required, no default, raise naming the parameter and the incident it
prevents" shape as `tariff_case` (`_TARIFF_REQUIRED`, `scenarios.py:2230`).

### Surfaced on the scenario dict — `src/scenarios.py`, next to `storage_duration_hours`

```python
"storage_duration_hours": _storage_duration_hours,

"om_case": om_case,
"om_cost_by_year": _om_cost_by_year,
```

**Not added to any arm tag or output filename** — per instruction, and
consistent with L9's reasoning (`reserve_margin`/`genset_case`) and the
declined `storage_duration_hours` promotion: it is recorded as a
top-level summary field instead (Part 4).

---

## PART 2 — the loader

### `load_om_cost_by_year()` — new function, `src/io.py` (after `load_storage_capex_by_year`)

Follows the `load_solar_capex_by_year` / `load_storage_capex_by_year`
pattern exactly: filter the CSV, check for gaps *inside* the source's
coverage (raise if found), then call `_extend_capex_series` — the SAME
helper `load_solar_capex_by_year` uses — to hold 2050 flat through the
2051-2054 buffer. No second extension convention was written.

```python
_OM_TECHNOLOGIES = ("solar_utility", "storage", "gas_ct", "gas_cc")


def load_om_cost_by_year(om_case, start_year=2025, end_year=2050):
    """..."""
    if om_case == "om_none":
        zero = {"fom_usd_per_kw_yr": 0.0, "vom_usd_per_mwh": 0.0}
        return {
            tech: _extend_capex_series(
                {y: zero for y in range(start_year, 2051)}, end_year
            )
            for tech in _OM_TECHNOLOGIES
        }

    om_df = pd.read_csv(ROOT / "data/cost/processed/om_cost.csv")
    case_df = om_df[om_df["om_case"] == om_case]
    if case_df.empty:
        raise ValueError(f"om_cost.csv has no rows for om_case '{om_case}'.")

    result = {}
    for tech in _OM_TECHNOLOGIES:
        tech_df = case_df[case_df["technology"] == tech].copy()
        tech_df = tech_df[tech_df["year"] >= start_year].sort_values("year")
        found_years = set(int(y) for y in tech_df["year"])
        # ... (coverage-gap check identical in shape to the solar loader) ...
        by_year = {
            int(row["year"]): {
                "fom_usd_per_kw_yr": float(row["fom_usd_per_kw_yr"]),
                "vom_usd_per_mwh": float(row["vom_usd_per_mwh"]),
            }
            for _, row in tech_df.iterrows()
        }
        by_year = {y: v for y, v in by_year.items() if y <= end_year}
        result[tech] = _extend_capex_series(by_year, end_year)
    return result
```

Verified against the actual CSV (read-only inspection, `awk`/`sort`, no
script execution): 312 data rows, `om_case` values exactly `om_low` /
`om_central` / `om_high`, `technology` values exactly `gas_cc` /
`gas_ct` / `solar_utility` / `storage`, 26 years (2025-2050) per
technology per case — matches TASK.md's stated shape exactly.

### `STORAGE_OM_PER_MWH_YR` retired — `src/io.py:237-240`

Old text:
```python
    econ["STORAGE_COST_PER_MWH"] = to_float(storage_row["Storage_capex_usd_per_mwh"])

    # Storage annual fixed O&M. Primary purpose: break LP degeneracy in storage
    # sizing (without this, any storage capacity >= ~10 GWh is equally optimal).
    # Value is conservative (NREL ATB range: 10,000–18,000 USD/MWh-yr).
    econ["STORAGE_OM_PER_MWH_YR"] = 2_000.0

    voll_df
```

New text:
```python
    econ["STORAGE_COST_PER_MWH"] = to_float(storage_row["Storage_capex_usd_per_mwh"])

    # STORAGE_OM_PER_MWH_YR (2,000 USD/MWh-yr, a numerical-conditioning
    # constant with no cost provenance) is RETIRED here (task_om_implementation,
    # Part 3.1). It never entered either builder's objective -- see
    # docs/task_storage_om.md and the comment in
    # scripts/14_run_str1_storage_role.py. Real, sourced O&M (NREL ATB,
    # data/cost/processed/om_cost.csv) is now wired via the REQUIRED
    # om_case arm -- see scenarios.om_scenarios() / io.load_om_cost_by_year().

    voll_df
```

**Confirmed by fresh grep**: after this deletion, `STORAGE_OM_PER_MWH_YR`
appears nowhere else in `src/` or `scripts/` except (a) the new comments
this task added explaining the retirement, and (b) the pre-existing
comment in the suspended `scripts/14_run_str1_storage_role.py:30`
("`STORAGE_OM_PER_MWH_YR : never entered the objective`"), left
untouched exactly as instructed.

**Why this cannot silently change a computed value in a way nobody can
detect**: the deleted key had zero readers (confirmed in
`docs/task_storage_om.md`); no code path subscripted `econ["STORAGE_OM_PER_MWH_YR"]`.
Its removal is a no-op for every existing run. (This is the ONE edit
in this task that is genuinely inert on its own — the objective-moving
change is entirely the NEW O&M term, Part 3.)

---

## PART 3 — the cost terms

### 3.1 — Shared definition (plan 2.5)

New function `compute_om_cost_npv()`, `src/io.py`, called identically
by both builders (`from src.io import compute_om_cost_npv`). Kept as a
plain Python function using built-in `sum()`, not `pyo.quicksum` — the
same idiom already used for `gas_cap_rule`/`storage_cap_rule`
(`... + sum(m.gas_add[k] for k in range(0, t + 1))`), so `io.py` needs
no `pyomo` import and stays the "low-level I/O helpers" module its own
docstring describes; callers wrap the returned expression in their own
`pyo.Expression`.

```python
def compute_om_cost_npv(
    T, years, DF, om_cost_by_year, storage_duration_hours,
    solar_capacity_mw, storage_capacity_mwh,
    gas_unconverted_capacity_mw, gas_converted_capacity_mw,
    gas_unconverted_generation_twh, gas_converted_generation_twh,
):
    solar_om = om_cost_by_year["solar_utility"]
    storage_om = om_cost_by_year["storage"]
    gas_ct_om = om_cost_by_year["gas_ct"]
    gas_cc_om = om_cost_by_year["gas_cc"]

    fom_solar = sum(
        DF[t] * solar_capacity_mw[t]
        * solar_om[int(years[t])]["fom_usd_per_kw_yr"] * 1000.0
        for t in T
    )
    fom_storage = sum(
        DF[t] * storage_capacity_mwh[t]
        * (storage_om[int(years[t])]["fom_usd_per_kw_yr"] * 1000.0
           / storage_duration_hours)
        for t in T
    )
    fom_gas = sum(
        DF[t] * (
            gas_unconverted_capacity_mw[t]
            * gas_ct_om[int(years[t])]["fom_usd_per_kw_yr"] * 1000.0
            + gas_converted_capacity_mw[t]
            * gas_cc_om[int(years[t])]["fom_usd_per_kw_yr"] * 1000.0
        )
        for t in T
    )
    vom_gas = sum(
        DF[t] * (
            gas_unconverted_generation_twh[t] * 1e6
            * gas_ct_om[int(years[t])]["vom_usd_per_mwh"]
            + gas_converted_generation_twh[t] * 1e6
            * gas_cc_om[int(years[t])]["vom_usd_per_mwh"]
        )
        for t in T
    )
    return fom_solar + fom_storage + fom_gas + vom_gas
```

### 3.2 — Unit conversions (every one shown)

- **Solar and gas FOM**: `fom_usd_per_kw_yr` is USD/kW-yr on POWER
  capacity; model capacities are MW, so `x 1000.0` converts kW-yr to
  MW-yr. `MW x (USD/MW-yr) = USD/yr`. ✓ dimensionally consistent.
- **Storage FOM (the exception)**: the model tracks storage in MWh of
  ENERGY; ATB's storage FOM is per kW of POWER (a storage system's
  fixed cost scales with its inverter/power train, not its battery
  cell count). Converted as
  ```
  USD/MWh-yr = fom_usd_per_kw_yr x 1000 / storage_duration_hours
  ```
  Worked check against TASK.md's own figure: 2025 central storage FOM
  = 38.7768 USD/kW-yr, `storage_duration_hours` = 4.0 (canonical,
  `scenarios.py:2666`) →
  `38.7768 x 1000 / 4.0 = 9,694.2 USD/MWh-yr` — matches TASK.md's
  stated 9,694.2 exactly.
  **This makes storage O&M depend on `storage_duration_hours`**, which
  register entry L21 (see `docs/task_bucketA_parity.md` Part C2)
  declares ASSUMED and unswept — stated explicitly in
  `compute_om_cost_npv`'s docstring and repeated in Part 5B below.
  `storage_duration_hours` is read from each builder's own already-resolved
  local (`scenario["storage_duration_hours"]`, per the prior task's Part 1
  direct-subscript conversion) and passed into the shared function as an
  argument — never hardcoded inside `compute_om_cost_npv`.
- **Gas VOM**: `vom_usd_per_mwh` is USD per MWh of ELECTRICAL generation.
  The generation streams passed in (`m.gas_generation`,
  `_converted_gen_year`) are TWh_e, so `x 1e6` converts to MWh.
- **Solar and storage VOM**: both are 0.0 in every ATB case sourced for
  this model (confirmed: `awk` scan of `om_cost.csv` — every
  `solar_utility`/`storage` row's `vom_usd_per_mwh` is `0.0` across all
  three cases and all 26 years; TASK.md's note on the `Summary_VOM`
  storage figure of 3,638.92 being a workbook artifact, not the
  technology sheet's own 0, was NOT re-derived here — taken as given
  per "Sourced data — already extracted, do not re-extract"). No VOM
  term is built for either.

### 3.3 — The terms, in both builders

**`src/optimize_model.py`** (annual builder) — inserted immediately
after `gas_opex_npv`'s definition, before the `capital_treatment_case`
branch (old text → new text):

Old text:
```python
    gas_opex_npv = pyo.quicksum(
        m.DF[t] * m.gas_to_power[t] * econ["GAS_COST_PER_TWH_TH"]
        for t in T
    )

    # F1, Step 5/6: matching comment above public_solar_capex_npv --
```

New text:
```python
    gas_opex_npv = pyo.quicksum(
        m.DF[t] * m.gas_to_power[t] * econ["GAS_COST_PER_TWH_TH"]
        for t in T
    )

    # ------------------------------------------------------------
    # Fixed + variable O&M (task_om_implementation, Part 3.3)
    # ------------------------------------------------------------
    # [... comment explaining the term, quoted in full in the diff ...]
    om_cost_npv = compute_om_cost_npv(
        T=T, years=years, DF=m.DF,
        om_cost_by_year=scenario["om_cost_by_year"],
        storage_duration_hours=storage_duration_hours,
        solar_capacity_mw=m.solar_capacity_mw,
        storage_capacity_mwh=m.storage_capacity_mwh,
        gas_unconverted_capacity_mw=m.gas_capacity_mw,
        gas_converted_capacity_mw={t: 0.0 for t in T},
        gas_unconverted_generation_twh=m.gas_generation,
        gas_converted_generation_twh={t: 0.0 for t in T},
    )

    # F1, Step 5/6: matching comment above public_solar_capex_npv --
```

This builder never models OCGT->CCGT conversion — `conversion_available=True`
already raises earlier in the file (Phase 2.8, Decision 3: "slice-resolved
... lives in build_model_sliced() only"). All gas here is therefore
`gas_ct`; the two `gas_converted_*` arguments are zero for every `t`,
so `fom_gas`/`vom_gas`'s converted terms are always exactly `0.0` in
this builder — not approximately zero, exactly (multiplication by the
Python literal `0.0`).

`system_cost_npv`'s two branches (`pre_f1_overnight` and `f1_tranche`)
each get `+ om_cost_npv` appended as the LAST term, after every
existing term:

```python
        system_cost_npv = (
            gas_opex_npv
            + public_solar_capex_npv
            + eaas_solar_capex_npv
            + gas_capex_npv
            + storage_capex_npv
            + m.cost_replacement_capex_npv
            + selfgen_npv
            + unserved_npv
            + carbon_cost_npv
            - m.salvage_npv
            + om_cost_npv                    # <-- appended
        )
    else:
        system_cost_npv = (
            gas_opex_npv
            + m.capital_charge_total_npv
            + selfgen_npv
            + unserved_npv
            + carbon_cost_npv
            + om_cost_npv                    # <-- appended
        )
```

Appending at the END, rather than inserting among the existing terms,
is deliberate: the file's own comment above this assembly warns that
"floating-point addition is not strictly associative, so even a
mathematically-equivalent regrouping could shift the last bit of the
regression value" — reordering existing terms was avoided entirely.
Adding a single new addend at the end changes NOTHING about the
relative order of the pre-existing terms.

`m.cost_om_npv = pyo.Expression(expr=om_cost_npv)` added to the
diagnostics block, next to `m.cost_carbon_npv`.

**`src/optimize_model_sliced.py`** (sliced builder) — same insertion
point (after `gas_opex_npv`, including its `if _conv:` addition), same
append-at-the-end to both `system_cost_npv` branches, same diagnostics
addition. The converted/unconverted split (Part 3.4 covers why these
particular expressions):

```python
    om_unconverted_gas_gen_twh = {
        t: pyo.quicksum(m.gas_generation[t, (s, p)] for (s, p) in S)
        for t in T
    }
    if _conv:
        om_converted_gas_capacity_mw = {t: _converted_ocgt_nameplate for t in T}
        om_converted_gas_gen_twh = {t: _converted_gen_year(m, t) for t in T}
    else:
        om_converted_gas_capacity_mw = {t: 0.0 for t in T}
        om_converted_gas_gen_twh = {t: 0.0 for t in T}
    om_unconverted_gas_capacity_mw = {
        t: m.gas_capacity_mw[t] - om_converted_gas_capacity_mw[t] for t in T
    }
```

`_converted_ocgt_nameplate` (`= pyo.quicksum(m.convert_mw[tier] for tier
in _CONV_KEYS)`) and `_converted_gen_year(m, t)` are the SAME
expressions Phase 2.8's own fuel-cost and capacity-constraint code
already uses (`optimize_model_sliced.py:823`, `:795`) — not shadow
copies. Under `conv_none` (`_conv` False, the regression/most-runners
arm), there is no conversion mechanism at all and everything is
`gas_ct`, matching the annual builder exactly.

### Capital envelope check (required by 3.3)

Checked `m.public_budget_constraint` in both builders
(`optimize_model.py:1674-1683`, `optimize_model_sliced.py:1891-1900` —
identical text in both):

```python
m.public_budget_constraint = pyo.Constraint(
    expr=pyo.quicksum(
        m.DF[t] * (
            m.solar_public_add[t] * m.solar_capex_param[t]
            + m.storage_add[t] * econ["STORAGE_COST_PER_MWH"]
            + m.eaas_subsidy[t]
        )
        for t in T
    ) <= public_budget_npv
)
```

Confirmed: this expression is untouched by this task and contains only
CAPEX-side terms (public solar CAPEX, storage CAPEX, the EaaS subsidy)
— no O&M term was added to it, and `om_cost_npv` is never referenced
inside it. O&M reaches only `system_cost_npv` (via the two branches
above). Since `m.solar_capacity_mw[t]` (the O&M base) already sums
public + EaaS additions with ONE rate lookup, EaaS and public solar
mechanically carry IDENTICAL O&M by construction — not a separate
choice that could be made to diverge, satisfying the supervisor's
decision without a second code path to keep in sync.

### 3.4 — Retirement and vintages

**Gas**: `m.gas_capacity_mw[t]` (both builders, `gas_cap_rule`) already
nets exogenous retirement out of cumulative additions —
`gas_baseline - retired + sum(m.gas_add[k] for k in range(0, t+1))`,
where `retired` is derived from `gas_retirement_case`'s schedule/series.
This is the SAME expression the firm-capacity constraint uses
(`capacity_credit["gas"] * m.gas_capacity_mw[t]`) — reused directly, not
recomputed, so gas FOM/VOM are automatically retirement-aware with no
separate logic.

**Storage**: `m.storage_capacity_mwh[t]` (`storage_cap_rule`,
`= sum(m.storage_add[k] for k in range(0, t+1))`) does NOT decline when
a vintage passes its 15-year life. Per L18 (`docs/limitations_register.md`),
this is the model's OWN established convention: a vintage is assumed to
keep operating for the full horizon, and its physical replacement is
represented as a REBUILD CAPEX charge at each `life`-year interval
(`m.cost_rebuild_capital_charge_npv` / `m.cost_replacement_capex_npv`),
not as a drop in installed capacity. Charging O&M on the full,
un-declined `m.storage_capacity_mwh[t]` is consistent with that
convention (the model already asserts this capacity is "actually
installed and operating" every year via the rebuild mechanism) — it
would be the REBUILD accounting, not the O&M accounting, that should
change if that convention were ever revisited, and that is out of
scope here.

**Solar**: `m.solar_capacity_mw[t]` (`solar_cap_rule`) is the same
cumulative-additions expression the firm-capacity constraint uses; no
retirement mechanism exists for solar in either builder (unchanged by
this task).

In all three cases, the expression reused for O&M is textually IDENTICAL
to the expression the firm-capacity constraint already reads — no new
"live capacity" computation was written.

---

## PART 4 — runners

### Fresh inventory — CONTRADICTS the count TASK.md itself asked not to trust

TASK.md: "There are 36 call sites across 25 files -- inventory them
fresh, do not trust that number." Fresh, exhaustive count (assignment-form
`load_scenario(` calls only, excluding comment/docstring mentions in
`src/io.py`, `src/optimize_model.py`, `src/optimize_model_sliced.py`,
and the three runner wrappers that call into `optimize_experiments.py`
backend functions — the same exclusions as the prior task's inventory):

- **25 files under `scripts/`/`tests/`, 26 call sites** (24 files with
  1 site each; `scripts/07_run_gas2_eaas_gas_relief.py` with 2).
- **`src/optimize_experiments.py`: 11 call sites** (lines 641, 789/790,
  812/814, 1195/1198, 1412/1415, 1606/1609, 1757/1760, 2764/2767,
  3007/3010, 3180/3183, 3323/3326 — first number is the pre-edit line
  from the prior task's report, second is this task's fresh read;
  the shift is from this task's own earlier edits, not drift).
- **Total: 26 files, 37 call sites** — not 25 files / 36 sites.

The prior task's own arithmetic was internally inconsistent: it stated
"24 script/test files... EXCEPT `07_run_gas2_eaas_gas_relief.py` (2 call
sites) = 25 site invocations," which is wrong on its own terms (24 + 1
= 25 FILES, and 24×1 + 2 = 26 SITES, not 25) — the count silently
dropped one file and one site somewhere in that report's arithmetic,
and the TOTAL line ("25 files, 36 call sites") inherited the error.
This is flagged per standing practice in this project: fresh
verification that contradicts an "established, do not re-verify" fact
is reported, not silently absorbed. It does not change this task's
required actions (every site still gets `om_case` threaded either way)
but it does change the count Part 4's own text cites, and is corrected
here for the record.

One `**dict`-unpack site remains, unchanged from the prior task's
finding: `scripts/27_run_hydro_minflow_sensitivity.py:389`,
`load_scenario(**CENTRAL_ARGS)` — handled by adding the key to the
`CENTRAL_ARGS` dict literal itself (see below), not by restructuring
the call.

**No site required restructuring a function signature** to reach — the
STOP condition in the "if any site cannot be reached" sense was not
triggered.

### `--om-case` added to the 5 named runners

`scripts/01_run_baseline.py`, `scripts/01b_run_baseline_sliced.py`,
`scripts/02_run_eaas.py`, `scripts/02_run_ndc_caps.py`,
`scripts/03_run_ndc_eaas.py` — each got, in the identical pattern
`--tariff-case` already uses in that file:

1. `from src.scenarios import (..., om_scenarios, ...)` added to the
   import.
2. `OM_CASE_CHOICES = tuple(om_scenarios())` added next to
   `TARIFF_CASE_CHOICES`.
3. A `parser.add_argument("--om-case", required=True, choices=OM_CASE_CHOICES, help=...)`
   block added next to `--tariff-case`'s, with help text naming the
   `STORAGE_OM_PER_MWH_YR` incident as `--tariff-case`'s names the 65M
   default.
4. `om_case=args.om_case,` added to the `load_scenario()` call.
5. `"om_case": args.om_case,` added to the summary JSON, next to
   `"tariff_case"`, with the same "recorded for auditability... NOT
   appended to the output tag" comment style.

### The other 32 sites: `om_case="om_central"` threaded explicitly

20 other `scripts/`/`tests/` files (21 sites, `07_run_gas2` counted
twice) plus `src/optimize_experiments.py`'s 11 sites — all given the
literal `om_case="om_central"` at the call site, matching how each
site already threads its other required keys (a mix of `args.x`,
hardcoded literals, and pass-through variables — `om_case` follows
whichever convention was NOT itself parameterised at that site, i.e.
a bare literal, since none of these 32 sites expose an O&M CLI flag).

**One deliberate deviation** from "`om_central` everywhere else":
`scripts/27_run_hydro_minflow_sensitivity.py`'s `CENTRAL_ARGS` dict got
`om_case="om_none"`, NOT `"om_central"`. Reason: this script's own
`REFERENCE_NPV_TOTAL_COST_USD = 21351610495.345654` is a hardcoded
reproduction guard against the PRE-O&M central anchor — the exact
number this task's `om_none` arm exists to reproduce. Passing
`om_central` here would move that value and fail the script's own
reproduction check for a reason unrelated to hydro min-flow sensitivity
(the thing the script tests), not because anything about hydro is
wrong. `om_none` is the arm built specifically for this situation. This
is recorded here as a judgement call, not applied silently.

### A boundary note, disclosed rather than silently resolved

TASK.md §0's BOUNDING lists the editable files exhaustively as
`src/io.py`, `src/scenarios.py`, `src/optimize_model.py`,
`src/optimize_model_sliced.py`, "and the runners listed in §4" (the 5
named CLI runners). `src/optimize_experiments.py` is NOT in that list,
yet it holds 11 of the 37 `load_scenario()` call sites, and §4's own
text requires "every OTHER load_scenario() call site" be threaded "so
nothing breaks." Leaving those 11 sites unthreaded would, mechanically,
make `om_case` a required argument they no longer supply — every
runner whose backend logic routes through `optimize_experiments.py`
(several of the 25 script files call into functions defined there)
would raise `TypeError` on their next run, which is the opposite of
"nothing breaks." Read literally, §0 and §4 conflict for this one file.

Resolution taken: `src/optimize_experiments.py`'s 11 sites WERE edited,
identically in kind to the 32 other threaded sites (a single literal
kwarg added, value-frozen, no other code touched). This is flagged here
rather than silently included in the count above, so Zeus can treat it
as a distinct line in `git diff` review — the file is almost certainly
an oversight in §0's list (BOUNDING names the two builders and the
loader/registry but not the backend module all the runner-independent
sweep/frontier logic actually lives in), not a deliberate exclusion,
but that is a judgement call, not a fact, and is reported as one.

### Summary JSON field

`"om_case"` added next to every existing `"tariff_case"` field in the 5
named runners (see above). The other 20 script files and
`optimize_experiments.py` do not have a `"tariff_case"`-style summary
field pattern to mirror at their `om_case`-bearing call sites (most
either have no summary-JSON block at this call site or are library
functions, not top-level runners) — TASK.md's instruction was scoped to
"wherever `tariff_case` already appears," and it appears only in the 5
named runners plus `optimize_experiments.py`'s own summary-writing call
sites (789/812, 1198), which already carry a `"tariff_case"` field
independent of this task. No new summary field was added there, since
none of them lacked a `tariff_case` twin to mirror and the CLAUDE.md
"no defects outside scope" rule counsels against inventing a new
summary-JSON convention mid-task.

---

## PART 5 — required writing deliverables

### 5A — For the methodology chapter

Operating costs in this model's objective fall into two categories that
are charged on different bases. Fixed operating and maintenance cost
(FOM) is a recurring charge proportional to installed CAPACITY,
independent of how much a plant runs — the model charges it, per
technology, per year, on whatever capacity is actually installed and
operating that year (accounting for retirement where the technology
retires). Variable operating and maintenance cost (VOM) is a recurring
charge proportional to OUTPUT — it is charged only where a technology's
own cost data shows a non-zero rate. In this model, solar and storage
carry FOM only (their sourced VOM rates are zero in every NREL ATB cost
case used); gas carries both FOM, on installed capacity, and VOM, on
generation, in addition to its pre-existing fuel cost, which is a
distinct line from either. Every one of these terms is discounted by
the same social discount factor the rest of the objective already
applies — there is no separate "operating cost" discount convention.

The rates themselves come from NREL's Annual Technology Baseline (ATB)
2024, `2024_v3_Workbook.xlsx`, `Summary_FOM` sheet for fixed cost and
the matching technology sheet for variable cost, at three cost cases —
Advanced, Moderate, and Conservative — which this model exposes as
`om_low`, `om_central`, and `om_high` respectively, each carrying a
full 2025-2050 year-by-year trajectory per technology (`solar_utility`,
`storage`, `gas_ct` for the unconverted gas fleet, `gas_cc` for
OCGT-to-CCGT converted capacity). `om_central` is now the model's
central O&M assumption, replacing a placeholder constant
(`STORAGE_OM_PER_MWH_YR`, 2,000 USD/MWh-yr) that had been sitting in
the cost-loading code for the life of the project without ever being
read by either builder — an omission this project's own required-key
discipline exists to catch, and did.

One conversion deserves its own explanation because it introduces a
dependency the model did not have before. ATB reports storage's fixed
cost per kW of POWER capacity, because that is what drives a battery
system's inverter and balance-of-plant cost; this model tracks storage
in MWh of ENERGY capacity, because that is the quantity its dispatch
constraints are written against. Converting one to the other requires
dividing by an assumed number of hours of discharge duration — this
model uses 4.0 hours, the same figure that already sets the storage
capacity-credit ceiling and the storage power-to-energy ratio elsewhere
in the LP. The effect is that storage's O&M charge is now coupled to an
assumption (`storage_duration_hours`) that a companion task found is
threaded through more call sites than initially estimated and left
un-promoted to a required, explicitly-swept key; register entry L21
declares it ASSUMED, not SOURCED, and unswept. Wiring O&M did not
change that classification, but it does mean L21's uncertainty now has
a second, larger channel through which it can move a reported result:
before this task, `storage_duration_hours` only affected a capacity
credit; it now also scales a cost.

Finally, this task added a fourth arm, `om_none`, whose only purpose is
to set every O&M rate to exactly zero. It exists because this task
changes the objective on purpose, and a change to the objective needs
its own regression test distinct from "did the numbers move" — `om_none`
lets a reviewer confirm that the NEW code path, exercised with every
rate forced to zero, reproduces the OLD objective to the digit, which
is proof that the wiring itself introduced no side effect beyond the
intended cost term. `om_central` moving the reported total cost is
therefore not evidence of a bug; `om_none` failing to reproduce the old
anchor would be.

### 5B — For the limitations register

Draft entry, following the register's established format (state,
classification, what it controls, direction of bias):

> ## L22 — Fixed + variable O&M is now SOURCED and wired; the storage conversion inherits L21's uncertainty
>
> **Status:** LIVE
>
> **What.** Both LP builders now charge fixed O&M (FOM, USD/kW-yr or
> the converted USD/MWh-yr for storage) on installed capacity and
> variable O&M (VOM, USD/MWh) on gas generation, for solar, storage,
> gas_ct (unconverted), and gas_cc (OCGT-to-CCGT converted), discounted
> alongside every other operating cost. This replaces a prior omission:
> no technology previously carried any O&M term in either objective
> (`docs/task_storage_om.md`), and a placeholder constant
> (`STORAGE_OM_PER_MWH_YR`) had zero readers for the life of the
> project.
>
> **Classification: SOURCED**, NREL ATB 2024, `2024_v3_Workbook.xlsx`,
> `Summary_FOM` sheet (fixed) and the per-technology sheets (variable),
> three cost cases (Advanced / Moderate / Conservative, exposed as
> `om_low` / `om_central` / `om_high`), 2025-2050, extracted to
> `data/cost/processed/om_cost.csv`. Storage's zero VOM was verified
> against the technology sheet directly (`Utility-Scale Battery Storage`,
> row 137) rather than the unreliable `Summary_VOM` sheet, which shows a
> nonzero figure for storage that does not represent variable O&M (per
> TASK.md's own sourcing note — not re-derived in this task).
>
> **Unscaled US new-build figures, applied to a Nigerian fleet operating
> at far lower availability — deliberate, not an oversight.** ATB's cost
> figures are calibrated to US utility-scale fleets running at US
> availability factors; Nigeria's gas fleet in particular operates at a
> fraction of that (this model's own `gas_availability_factor` arms are
> well below nameplate). A tempting "correction" is to scale FOM down by
> the availability shortfall, on the reasoning that a less-available
> plant needs less maintenance spend. This model does NOT do that, for
> the same reason `docs/tier2_reconciliation.md` and the demand-growth
> arms refuse to calibrate an input to a supply-constrained outcome:
> Nigerian gas plants likely run below nameplate availability BECAUSE of
> chronic O&M underspend relative to their nameplate design, not
> instead of it. Scaling FOM down to match observed availability would
> treat the underspend as if it were an efficiency the model should
> reward, when the underspend is plausibly the CAUSE of the very
> availability figure the model already takes as exogenous. That is
> circular in exactly the shape the register already declines elsewhere:
> an input is not calibrated to a phenomenon the input itself may be
> producing.
>
> **Direction of bias.** Two directions, not one:
> (a) Against the un-scaled-US-rates choice: if Nigerian O&M cost per
> installed kW is genuinely lower than the US ATB figure (labour is
> cheaper; some maintenance categories may be deferred rather than
> foregone), this model OVERSTATES gas and solar/storage operating
> cost in absolute USD terms, and therefore overstates total system
> cost and possibly the relative attractiveness of a lower-capacity
> pathway.
> (b) Against the (correctly, in this model's judgement) un-scaled
> availability choice: if the true relationship runs the other way —
> Nigerian plants are simply worse-maintained per dollar spent, not
> spending less per dollar of value received — this model's cost figure
> could still UNDERSTATE the true cost of keeping gas plant available at
> Nigerian reliability standards, since ATB's dollars buy US-grade
> maintenance outcomes, not Nigerian ones.
> Net direction is therefore NOT signed here; both biases point through
> the same parameter in opposite directions and neither dominates by
> construction.
>
> **The storage conversion inherits L21.** Storage's FOM is converted
> from ATB's USD/kW-yr to this model's USD/MWh-yr basis by dividing by
> `storage_duration_hours` (= 4.0, canonical `scenarios.py:2666`). L21
> classifies that value ASSUMED and unswept. Before this task it set
> only a capacity-credit ceiling; it now also scales a cost term. A
> shorter true duration than 4.0 would understate storage's true FOM in
> USD/MWh-yr terms (the same kW-yr charge spread over less assumed
> energy capacity, i.e. UNDERSTATED here since the model would be
> dividing by a number larger than the truth); a longer true duration
> would overstate it.
>
> **What would resolve it.** For the scaling question: a Nigeria-specific
> O&M cost survey (none is known to exist at the technology granularity
> this model needs). For the L21 coupling: promote `storage_duration_hours`
> to a required, swept key (declined in `docs/task_bucketA_parity.md`
> for this project's remaining timeline, not because it is unimportant).

### 5C — Predicted effect, written before Zeus runs anything

**Central objective (NPV total cost): increases, and MUST increase —
this is not a guess.** `om_central`'s O&M term is a sum of
non-negative rates times non-negative capacities/generation, added as a
strictly-positive-or-zero addend to every feasible solution's cost,
including the solution that was optimal under `om_none`. The new
optimum can only be greater than or equal to the old one evaluated
under the new cost structure, which is itself greater than or equal to
the old optimal value. Rough magnitude: gas dominates both installed
capacity and generation in this system, so gas FOM+VOM is expected to
be the largest of the three added components — at central 2025 rates
(gas_ct ≈ 25.7 USD/kW-yr FOM, 6.94 USD/MWh VOM) applied across a fleet
in the multi-GW range generating tens of TWh/year over a ~20-30 year
discounted horizon, gas O&M alone plausibly adds low-single-digit
billions of USD to the ~21.35 billion central NPV; solar and storage
FOM add a smaller amount each (storage specifically bounded by its
relatively small terminal capacity, ~65,613 MWh under the OLD central
run). A rough expectation is a mid-single-digit-percent to
low-double-digit-percent increase in central NPV — this is a coarse
order-of-magnitude reasoning check, not a computed figure, and Zeus's
actual solve is the real answer.

**Storage build: expected to fall or stay flat, not rise.** Storage now
carries a holding cost proportional to installed energy capacity that
it did not carry before, while its firm-capacity-credit benefit is
unchanged. A cost-minimising LP facing a new capacity-proportional
penalty on an asset, with no change to that asset's benefit, has no
channel through which to want MORE of it. The direction is confident;
the magnitude is not predicted here (it depends on how binding storage's
firm-capacity contribution is against the alternative of more gas or
solar capacity, which this task did not attempt to estimate ex ante).

**Conversion (OCGT-to-CCGT, sliced builder only): a secondary,
weaker effect toward MORE conversion, not less.** Converted (`gas_cc`)
capacity now carries LOWER FOM and VOM than unconverted (`gas_ct`)
capacity at every year in the sourced data (e.g. 2025: gas_cc FOM 33.5
vs gas_ct FOM 25.7 — actually HIGHER for gas_cc on FOM, but gas_cc's
VOM, 2.12, is well below gas_ct's 6.94). Whether this nets to more or
less conversion pressure depends on the converted/unconverted
generation split the LP already chooses, which this task did not
attempt to predict; flagged as the one component whose direction is
genuinely ambiguous ex ante, unlike the objective and storage
predictions above.

---

## Closing list — things encountered but not asked about

- Several runner call sites already omit other required keys entirely
  (`hydro_case`, `conversion_case`, and in some cases `tariff_case`) and
  would already raise `TypeError` before reaching `load_scenario()`'s
  `om_case` check, independent of this task:
  `scripts/02_run_eaas.py`, `scripts/04_run_fin2_tariff_sweep.py`,
  `scripts/05_run_fin3_capital_sweep.py`,
  `scripts/06_run_gas1_shadow_benchmarks.py`,
  `scripts/18_run_pareto_frontier.py`,
  `scripts/19_run_carbon_price_sweep.py`,
  `scripts/20_run_cf1_solar_yield_sensitivity.py`,
  `scripts/probe_2_6_structure.py`,
  `scripts/deferred/17_run_monte_carlo.py`,
  `scripts/retired/09b_run_rel1_mode_sensitivity.py`,
  `scripts/retired/09b_run_rel1_retirement_sensitivity.py`,
  `tests/test_run.py`. This task did not fix or otherwise touch these
  pre-existing gaps beyond adding `om_case="om_central"` alongside them
  (a call that would already fail on the earlier missing key).
- `src/optimize_experiments.py` was edited despite not being named in
  BOUNDING §0's file list — see the boundary note in Part 4.
- The prior task's "25 files / 36 call sites" total was arithmetically
  inconsistent with its own per-category breakdown; the true fresh
  count is 26 files / 37 call sites — see Part 4.
- `scripts/27_run_hydro_minflow_sensitivity.py` got `om_case="om_none"`,
  not `"om_central"`, to protect its own hardcoded reproduction guard —
  see Part 4.
- Whether `gas_cc`'s 2025 FOM (33.5) being HIGHER than `gas_ct`'s (25.7)
  — while `gas_cc`'s VOM (2.12) is LOWER than `gas_ct`'s (6.94) — is
  itself plausible (a combined-cycle bottoming plant costing more to
  maintain per MW but less to run per MWh) was not checked against any
  independent source; taken as given from the sourced CSV.
- No check was made of whether `genset_cost.csv`'s own
  `fixed_om_usd_per_kw_yr` / `om_cost_usd_per_mwh` columns (the private
  self-generation backstop) should be reconciled against this task's
  `om_cost.csv` for consistency of methodology — out of scope, different
  technology class (diesel genset, not grid-connected gas/solar/storage).

---

## Shell runner threading

Follow-up task: `--om-case` is `required=True` on `01_run_baseline.py`,
`01b_run_baseline_sliced.py`, `02_run_eaas.py`, `02_run_ndc_caps.py`,
`03_run_ndc_eaas.py`. Three shell scripts invoke these runners without
the flag and would die on argparse. Scope: edit only
`scripts/run_central_sweep.sh`, `scripts/run_verification_sweep.sh`,
`scripts/verify_2_6_commit_a.sh`.

### Step 1 — Inventory

**`scripts/run_central_sweep.sh`.** All target-runner invocations go
through a single bash function, `run() { ... python3 "scripts/${runner}" ...; }`
(the file's own shared-argument mechanism — every call site passes the
runner name and a label, plus optional `key=value` overrides consumed
by a `case` statement inside `run()`). 33 invocations of our five
runners, all via this one function:

| Lines | Runner | Count | Section / purpose |
|---|---|---|---|
| 190 | `01b_run_baseline_sliced.py` | 1 | A — central configuration |
| 199 | `01b_run_baseline_sliced.py` | 1 | A — `A_REGRESSION_ANCHOR`: "the attribution arm that must reproduce [the pre-F1 anchor]... If it does not, the sweep is invalid" |
| 209–253 | `01b_run_baseline_sliced.py` | 25 | B — one-at-a-time sensitivity arms (capital recovery, conversion, gas deliverability x2 families, hydro, concessional envelope, DSCR, gas availability, gas retirement, demand shape) |
| 191, 261, 262, 265 | `02_run_ndc_caps.py` | 4 | A central + C interaction cells (deliverability×cap, conversion×cap) |
| 192, 268 | `03_run_ndc_eaas.py` | 2 | A central + C interaction cell (EaaS×cap, F-27) |
| — | `01_run_baseline.py` | 0 | never invoked in this script |
| — | `02_run_eaas.py` | 0 | never invoked in this script |

**Total: 33 invocations** (27 + 4 + 2), all reachable through the one
`run()` function.

**`scripts/run_verification_sweep.sh`.** A bare `for f in $RUNNERS; do
python "$f" ...; done` loop over a 19-script list, no CLI flags of any
kind currently passed to anything. All 5 target runners appear in the
list once each:

| Line (in `$RUNNERS`) | Runner | Purpose |
|---|---|---|
| 19 | `01_run_baseline.py` | generic "does every runner still execute cleanly" smoke/verification pass |
| 20 | `01b_run_baseline_sliced.py` | same |
| 21 | `02_run_eaas.py` | same |
| 22 | `02_run_ndc_caps.py` | same |
| 23 | `03_run_ndc_eaas.py` | same |

**Total: 5 invocations**, all via the same bare loop.

**`scripts/verify_2_6_commit_a.sh`.** Two DIFFERENT invocation contexts
of two of the five runners:

| Lines | Runner | Purpose |
|---|---|---|
| 90 | `01_run_baseline.py` | pre-patch: run inside a `git worktree` checked out at `PRE_REF` (old code) |
| 91 | `01b_run_baseline_sliced.py` | pre-patch: same worktree |
| 104 | `01_run_baseline.py` | post-patch: run at HEAD (current code) |
| 105 | `01b_run_baseline_sliced.py` | post-patch: run at HEAD |

**Total: 4 invocations** of target runners (2 pre-patch, 2 post-patch).
A fifth, DIFFERENT `load_scenario()` call exists in this file (the
inline Python heredoc at lines 47-76, the "smoke test: do both builders
construct?" step) — this is a DIRECT library call, not an invocation of
one of the five runner scripts, so it falls outside "every invocation
of those five runners" as literally scoped. It will also now fail
(`om_case` is required with no default) — flagged in the closing list,
not edited.

**Counts per script: 33 / 5 / 4 = 42 total target-runner invocations
across the three files.**

### Step 2 — Value chosen per invocation

**`om_central` (default) for 40 of the 42.** None state a purpose of
reproducing a specific anchor, except:

**`run_central_sweep.sh:199`, `A_REGRESSION_ANCHOR` → `om_none`.** Its
overrides (`conv=conv_none treat=pre_f1_overnight dscr=dscr_regression
tariff=tariff_legacy_65m`) are the established regression configuration
this session's frozen anchors are built from, and its own comment says
plainly: "the attribution arm that must reproduce [...]. If it does
not, the sweep is invalid and nothing below it should be read." This is
exactly the case TASK.md names explicitly.

**A discrepancy in that same comment, flagged not fixed**: the script's
inline text says the anchor is "21.4748 bn / 44,377 MWh" (line 178, and
"21.475 bn" again at line 271), but the regression anchor established
and used throughout this session's other tasks is
**21811042472.821114 USD / 45502.9754 MWh** (≈ 21.811 bn) —a different
number. The override set at this call site (`conv_none` +
`pre_f1_overnight` + `dscr_regression` + `tariff_legacy_65m`) is
recognisably the SAME configuration pattern that produces the
session's established regression anchor elsewhere, which is why
`om_none` was chosen here with confidence despite the stale-looking
number in the comment. The comment itself was NOT corrected — out of
this task's scope (threading `--om-case`, not auditing unrelated
stale documentation), and flagged in the closing list.

**`verify_2_6_commit_a.sh`'s two POST-PATCH invocations (lines 104-105)
→ `om_none`, not `om_central`.** This is the second case TASK.md names
("a byte-identity check"). The script's entire purpose, per its own
header, is that "the optimum must not move" between `PRE_REF` and HEAD
— a live drift check, not a fixed numeric target. Giving the post-patch
run `om_central` would introduce a cost term the pre-patch run
structurally cannot have (its own `load_scenario()`, at whatever
`PRE_REF` is, does not know about O&M at all), making any objective
delta partly attributable to this session's unrelated O&M work rather
than to whatever the comparison is actually meant to isolate. `om_none`
keeps the post-patch objective on the same footing the pre-patch run is
implicitly on, so a moved optimum stays attributable to what the script
is actually testing.

**`verify_2_6_commit_a.sh`'s two PRE-PATCH invocations (lines 90-91) →
NOT edited at all**, deliberately. These run inside a `git worktree`
checked out at `PRE_REF` — i.e. an OLD, checked-out copy of
`01_run_baseline.py`/`01b_run_baseline_sliced.py` from git history, not
the current working tree. Whether that old copy recognises `--om-case`
depends entirely on which commit `PRE_REF` resolves to (default
`HEAD~1`) at run time, which this task cannot know. Adding the flag
unconditionally risks breaking a code path that currently works, for a
script whose default invocation is `HEAD~1` — almost certainly a commit
that predates this flag's existence given today's uncommitted state.
Per TASK.md's own risk framing ("the safer failure mode"), the safer
choice here is to add nothing rather than guess wrong on code this task
cannot inspect. Flagged, not applied.

**`run_verification_sweep.sh`'s 5 invocations → `om_central`**, all
default. None single out a specific reproduction purpose; the whole
script is a uniform "does every runner still execute" pass. Separately,
and unrelated to `om_case`: ALL FIVE were already broken before this
task, for a much larger reason — they are called completely bare
(`python "$f"`, zero CLI arguments), and every one of these five
runners has many OTHER `required=True` flags (`--peak-case`,
`--storage-cc-case`, etc.) that argparse would reject as missing
regardless of `--om-case`. Threading `--om-case` here is still done, as
instructed, but it does not by itself make any of these five
invocations runnable — see the closing list.

### Step 3 — Placement

**`run_central_sweep.sh`: shared placement, inside `run()`.** A single
`C_OM="om_central"` default, a `local om=$C_OM`, and an `om)` case
matching the file's existing override convention (`tariff)`, `conv)`,
`dscr)`, ...) — precisely the "shared variable holding common flags"
Step 3 describes, since one function already serves every invocation.
**Not applied unconditionally**, though: `run()` also invokes
`26_run_f3_identification.py` (Sections A and C), which does NOT accept
`--om-case` as a CLI flag (it resolves `om_case` internally, per the
prior task). A bare `case "$runner" in <the five> ) om_flag=(--om-case
"$om") ;; esac` gates the flag so only the five target runners receive
it — the invocations are homogeneous WITHIN the function's own
mechanism, but heterogeneous across the set of runners that function is
actually called with, so a pure "add unconditionally" shared approach
would have broken `26_run_f3_identification.py`. The one-line-item
regression override (`om=om_none` at the `A_REGRESSION_ANCHOR` call
site) uses the SAME override mechanism every other per-arm deviation in
this file already uses.

**`run_verification_sweep.sh`: per-runner-name conditional inside the
shared loop.** No `COMMON=`/`BASE_ARGS=` variable exists (every runner
here is invoked completely bare) — Step 3's "shared variable" case does
not apply. But the loop body itself IS shared across all 19 runners, 14
of which must NOT receive `--om-case` (they don't accept it). Same
`case "$name" in <the five>) om_args="--om-case om_central" ;; esac`
gating pattern as `run_central_sweep.sh`, adapted to this file's
plain-string (not array) style, since no arrays are used anywhere else
in this file and a two-token flag+value string needs no quoting
protection here.

**`verify_2_6_commit_a.sh`: per-site, not shared.** Only 2 of this
file's 4 target-runner invocations get the flag at all, and the other 2
(same two runners) deliberately do not — a shared mechanism would have
had to encode "same runner, same script, different flag depending on
which invocation" in a way that adds more surface area than just
writing the two literal `--om-case om_none` flags at the two lines that
need them. Per-site is simpler and safer here specifically because the
two invocation contexts (pre-patch worktree vs. post-patch HEAD) are
semantically different despite calling the same script names.

### Step 4 — Verify

**`bash -n` on all three: all pass.**
```
$ bash -n scripts/run_central_sweep.sh && echo SYNTAX_OK
SYNTAX_OK
$ bash -n scripts/run_verification_sweep.sh && echo SYNTAX_OK
SYNTAX_OK
$ bash -n scripts/verify_2_6_commit_a.sh && echo SYNTAX_OK
SYNTAX_OK
```

**`grep -c "om-case"` per script, and why it does NOT equal the
invocation count for two of the three** — reported transparently rather
than forced to match:

| Script | `grep -c "om-case"` | Invocations identified | Why they differ |
|---|---|---|---|
| `run_central_sweep.sh` | 2 | 33 | Shared placement (Step 3): ONE functional occurrence (`om_flag=(--om-case "$om")`, line 143) reaches all 33 invocations through `run()`'s runner-name gate; the other match (line 135) is a comment. Grep-counting literal flag text is the wrong verification method under shared placement by construction — every shared-placement edit will always show fewer literal occurrences than invocations reached. |
| `run_verification_sweep.sh` | 2 | 5 | Same reason: ONE functional occurrence (line 51) reaches all 5 target runners through the loop's runner-name gate; the other match (line 45) is a comment. |
| `verify_2_6_commit_a.sh` | 4 | 2 (of 4; the 2 pre-patch invocations deliberately excluded) | Per-site placement here, so this IS a direct count: 2 functional occurrences (lines 113, 114) match the 2 invocations that were meant to receive the flag; the other 2 matches (lines 104, 111) are comments. |

**The verification that actually matters under shared placement** is
tracing the gate, not counting substring matches: `run_central_sweep.sh`'s
`case "$runner" in 01_run_baseline.py|01b_run_baseline_sliced.py|02_run_eaas.py|02_run_ndc_caps.py|03_run_ndc_eaas.py)`
and `run_verification_sweep.sh`'s `case "$name" in
01_run_baseline|01b_run_baseline_sliced|02_run_eaas|02_run_ndc_caps|03_run_ndc_eaas)`
both list exactly the five runner names named in TASK.md's "The
defect" section, verified by direct comparison against that list
(above), not by execution.

---

### Closing list — things encountered but not asked about

- The stale anchor value in `run_central_sweep.sh`'s own comments
  ("21.4748 bn / 44,377 MWh", lines 178 and 271) does not match this
  session's established regression anchor (21811042472.821114 USD /
  45502.9754 MWh). Not corrected — out of this task's scope.
- `verify_2_6_commit_a.sh`'s inline Python heredoc (lines 47-76) calls
  `load_scenario()` directly with no `om_case` argument. This is not an
  invocation of one of the five runner scripts, so it falls outside
  this task's literal scope, but it WILL now raise `TypeError` at step
  [1/4] of this script, before the runner invocations addressed above
  are ever reached. Not edited.
- All 5 target-runner invocations in `run_verification_sweep.sh` were
  already non-functional before this task, for reasons unrelated to
  `om_case`: they are invoked with zero CLI arguments, and every one of
  the five runners has several OTHER `required=True` flags. Threading
  `--om-case` does not restore these to a runnable state by itself.
- `verify_2_6_commit_a.sh`'s pre-patch worktree invocations (lines
  90-91) were left unedited on the reasoning that their behaviour
  depends on git history this task cannot inspect (`PRE_REF`, default
  `HEAD~1`). If Zeus knows `PRE_REF` will resolve to a commit at or
  after `--om-case` existing, those two lines would need the flag too
  (value: `om_none`, matching the post-patch reasoning above) —
  currently left as a live judgement call for Zeus, not resolved here.
- `26_run_f3_identification.py` is invoked by `run_central_sweep.sh`
  (Sections A and C) through the same `run()` function as the five
  target runners, but does not accept `--om-case` — confirmed the gate
  excludes it; not otherwise touched.

---

## Sweep anchor text correction

Follow-up task: `scripts/run_central_sweep.sh` stated its own validity
criterion — what `A_REGRESSION_ANCHOR` "must" reproduce — using a
two-epoch-stale number (21.4748 bn / 44,377 MWh), when the arm actually,
correctly, reads 21.811 bn under the now-current tree (Section A has
been run and passed: central 27.954 bn, regression anchor 21.811 bn,
zero failures). Scope: text and comments only in this one file — no
invocation, flag, or runner argument touched.

### Step 1 — Inventory

Searched for every occurrence of `21.4748`, `21.475`, `44,377`, `44377`,
`21474869376`, and more broadly for any prose asserting what the
regression anchor "must" read (`grep -niE "must (reproduce|read)"` and
a scan of every `bn`/`MWh` figure in the file, to check the three
named sites were the complete set rather than assume it). **Exactly two
stale sites exist — no others.**

| # | Line | Exact text |
|---|---|---|
| 1 | 197 | `# must reproduce 21.4748 bn / 44,377 MWh. If it does not, the sweep is` |
| 2 | 290 | `echo "CHECK FIRST: A_REGRESSION_ANCHOR must read 21.475 bn."` |

The other two `bn` figures in the file (line 19, `3.16 bn` — F-12
deliverability×cap; line 267, `8.895 bn` — F-27 EaaS financing channel)
are unrelated finding values quoted in section-C purpose comments, not
anchor-validity assertions, and were not touched.

### Step 2 — Corrections

**Site 1 — `scripts/run_central_sweep.sh:197`** (line number pre-edit;
shifted by the Step 3 insertion below):

Old text:
```
# The regression anchor. NOT central -- this is the attribution arm that
# must reproduce 21.4748 bn / 44,377 MWh. If it does not, the sweep is
# invalid and nothing below it should be read.
```

New text:
```
# The regression anchor. NOT central -- this is the attribution arm that
# must reproduce 21,811,042,472.821114 USD (21.811 bn) / 45,502.9754 MWh.
# If it does not, the sweep is invalid and nothing below it should be read.
```

This site carried both a USD and an MWh figure originally, so both were
corrected to the verified current values from TASK.md Step 2.

**Site 2 — `scripts/run_central_sweep.sh:290`** (pre-edit line number):

Old text:
```
echo "CHECK FIRST: A_REGRESSION_ANCHOR must read 21.475 bn."
```

New text:
```
echo "CHECK FIRST: A_REGRESSION_ANCHOR must read 21.811 bn."
```

This site carried only a USD figure, never an MWh one — per instruction
("do not invent a storage figure for any site that did not previously
carry one"), no MWh figure was added here.

### Step 3 — Clarifying note near the Section A header

Added immediately before the Section A `run` calls (after the
`banner "A -- ..."` line, before `run 01b_run_baseline_sliced.py A_central_baseline`):

```bash
# task_om_implementation: A_REGRESSION_ANCHOR below runs --om-case
# om_none; every other run in this file (this section and B/C) runs
# om_central. The two are NOT comparable to each other -- the anchor's
# job is reproducing the pre-O&M value exactly, not reporting a cost
# result under the central O&M assumption.
```

Placed before the section's `run` calls, not inside the existing
`A_REGRESSION_ANCHOR` comment block further down, since it is a
section-wide clarification (om_none vs om_central applies to every run
in the section, not only the anchor line) rather than a note specific
to the anchor itself.

### Step 4 — Epoch note in the header block

Added a new `# EPOCH` block to the file's top comment header,
immediately before the existing `# USAGE` block:

```
# EPOCH
#   Anchor values in this file (Section A's regression check, the
#   completion trailer) are as of the O&M implementation
#   (task_om_implementation): --om-case is now a required arm, and
#   wiring it raises the central configuration's objective by roughly
#   31% against the pre-O&M objective. This is what stopped the
#   previous anchor figure from being datable -- update these values
#   again if the O&M cost data or the central om_case selection changes.
```

The 31% figure and its framing ("what stopped the previous figure from
being datable") are stated exactly as given in TASK.md Step 4 — not
independently re-derived here. Note for the record: the two verified
figures given in TASK.md Step 2 (regression anchor 21.811 bn vs.
central 27.954 bn) are NOT the same configuration (different
`conv`/`treat`/`dscr`/`tariff` arms, per Step 3's own clarifying note
above) and their ratio (≈1.28×) is not the 31% cited here — the 31%
compares the SAME central configuration under `om_central` against
`om_none`, a comparison this task's read-only scope does not include
the data to reproduce independently. Taken as given.

### Step 5 — Verify

```
$ bash -n scripts/run_central_sweep.sh && echo SYNTAX_OK
SYNTAX_OK

$ grep -ncE "21\.4748|21\.475|44,377|44377|21474869376" scripts/run_central_sweep.sh
0
```

Zero stale occurrences remain.

### Closing list — things encountered but not asked about

- None. This task's scope (two numeric corrections, one section-header
  note, one epoch note, all in one file) left no open questions or
  adjacent defects beyond what Step 1's inventory already covered.
