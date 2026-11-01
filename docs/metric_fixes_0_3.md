# metric_fixes_0_3.md — Plan 0.3: metric formula fixes

All four defects are in `extract_planning_diagnostics`
(`src/optimize_experiments.py`). The LP was not touched — reporting layer
only, per TASK.md.

Verification (bounded, per TASK.md): `python3 -m py_compile
src/optimize_experiments.py` — passes. No solver, no runner, no smoke
suite, no scratchpad script was run. Zeus runs the regression.

---

## Defect 1 — `solar_generation_share` → `solar_share_of_grid_generation`

**Old expression** (`src/optimize_experiments.py`, was ~line 1940-1951):

```python
total_generation = sum(
    _slice_total(m.gas_generation, t)
    + _slice_total(m.solar_generation, t)
    + _slice_total(m.storage_discharge, t)
    for t in T
)
solar_total = sum(_slice_total(m.solar_generation, t) for t in T)
solar_share = solar_total / total_generation if total_generation > 0 else 0
```

Three faults, all confirmed by inspection: double-counts
`storage_discharge` (already counted once in `solar_generation` when
charged); omits hydro entirely (11.47 TWh under `hydro_flat`, up to
45 TWh under `hydro_ndc`); omits converted gas generation (Phase 2.8),
which under `conv_central` is 21.76 TWh against only 2.98 TWh of
unconverted fleet output — the old denominator missed roughly seven
eighths of gas-side generation whenever a conversion arm was active.

**New expression:**

```python
hydro_energy_twh_by_year = scenario["hydro_energy_twh_by_year"]

def _converted_generation_total(t):
    if not hasattr(m, "convert_generation"):
        return 0.0
    return sum(
        pyo.value(v) for k, v in m.convert_generation.items() if k[1] == t
    )

total_generation = sum(
    _slice_total(m.gas_generation, t)
    + _converted_generation_total(t)
    + _slice_total(m.solar_generation, t)
    + float(hydro_energy_twh_by_year[t])
    for t in T
)
solar_total = sum(_slice_total(m.solar_generation, t) for t in T)
solar_share = solar_total / total_generation if total_generation > 0 else 0
```

Denominator = gas fleet + converted gas + solar + hydro generation.
`storage_discharge` excluded (double-count, as above).

**Converted generation.** `m.convert_generation` is indexed
`[tier, t, (s, p)]` (sliced builder, `f1_tranche`, with an available
conversion arm only — `_CONV_KEYS, T, m.S` at
`src/optimize_model_sliced.py:775-779`). Read via `hasattr` guard because
the annual builder never creates this component (it raises on any
available conversion arm before diagnostics run). Summed by filtering the
component's own `.items()` on `k[1] == t` (the year index) rather than by
importing `_CONV_KEYS`/`SLICES` from `optimize_model_sliced` — those are
module-private / not currently imported into `optimize_experiments.py`,
and filtering the component's own index avoids adding that coupling.

**Hydro generation.** Read from `scenario["hydro_energy_twh_by_year"]`
directly, not from a model component, because the two builders represent
hydro differently and only the scenario series is available under both:
- Annual builder (`optimize_model.py`): hydro is a plain Python dict
  (`hydro = {t: float(hydro_energy_twh_by_year[t]) for t in T}`, line
  613) entered directly into the balance constraint — there is no
  `m.hydro_gen` Var at all.
- Sliced builder (`optimize_model_sliced.py`): `m.hydro_gen[t, (s,p)]` is
  a Var, but `m.hydro_season_budget` (lines 921-926) constrains
  `sum_slices(m.hydro_gen[t, .])` to **equal** `hydro_annual[t] *
  hydro_season_share[season]` per season — an equality, not a bound — so
  summed over both seasons it equals `hydro_energy_twh_by_year[t]`
  exactly.

Either way the scenario series is the correct annual hydro figure, so
reading it directly is builder-agnostic and avoids a `hasattr` branch on
`m.hydro_gen` that would be wrong for the annual builder.

**Boundary decision (implemented as stated, not derived).** This is a
share of GRID generation, matching the model's declared busbar metering
point. `selfgen` (private off-grid diesel) is EXCLUDED from both
numerator and denominator — under NDC caps it is 34-58% of national
emissions (finding F-10), so a national-basis share would differ from
this grid-basis share by roughly a factor of two. `selfgen_twh_by_year`
continues to be reported separately (unchanged field).

**Rename.** `solar_generation_share` → `solar_share_of_grid_generation`.
The denominator's meaning changed, so the numeric value is not
comparable to the old field — renamed per the standing correction-project
rule (CLAUDE.md: "When a change alters the MEANING of a quantity, rename
it so stale readers fail loudly").

**Consumers found** (`grep -rn "solar_generation_share" src/ scripts/
docs/ tests/`):
- The definition site itself (now renamed).
- `docs/phase_2_8_conversion.md:488` — a "Known reporting gap" note
  stating this field (and the two `gas_..._by_year` fields) excluded
  converted output. Updated in that file: the note now records this
  defect as resolved for `solar_share_of_grid_generation` /
  `total_generation`, while flagging that `gas_generation_twh_e_by_year`
  and `gas_to_power_twh_th_by_year` were deliberately left as fleet-only
  (out of this task's scope — see "Not touched" below).
- No other code consumer in `src/` or `scripts/`.

---

## Defect 2 — `gas_scarcity_value_usd` deleted

**Old expression** (was ~line 1913-1921):

```python
gas_scarcity_value = 0.0
for t, y in enumerate(years):
    shadow = gas_shadow[int(y)]
    gas_shortfall = gas_avail[t] - _slice_total(m.gas_to_power, t)
    if shadow is not None and shadow > 0:
        gas_scarcity_value += shadow * gas_shortfall
```

**Confirmed zero by construction.** When the fuel constraint binds,
`gas_to_power == gas_avail`, so `gas_shortfall == 0` and the product is
zero — exactly when gas is scarcest. When gas is slack, the dual is zero,
so the product is zero too. There is no regime in which the quantity is
non-trivially non-zero. It was also wrong a second way: converted plant
draws fuel through `m.convert_to_power`, absent from `gas_shortfall`'s
computation, so wherever the (already-zero) product wasn't exactly zero
it additionally overstated unused gas by the entire converted draw
(144.4 TWh_th at 2045 under `gas_deliv_gmp_full`).

**Fix.** Deleted the field and its computation entirely. The annual dual
is already reported directly: `gas_shadow_price_usd_per_twh_th_by_year`
(returned via `extract_gas_shadow_prices_usd_per_twh_th(m, years)`) was
confirmed present in the returned dict both before and after this change
— no addition needed.

**Consumers found** (`grep -rn "gas_scarcity_value_usd\|gas_scarcity_value\b"
src/ scripts/ docs/ tests/`): none, anywhere, other than the definition
and return-dict sites just removed. Per TASK.md's instruction, if a
consumer had existed the field would have been left in place and
reported rather than removed — none did, so the field was deleted.

Note: `m.gas_curtail` (the fuel-balance slack variable) was NOT touched —
TASK.md's "delete gas_curtail" instruction refers to this scarcity-value
metric only, not the model variable.

---

## Defect 3 — dual sign clamps → tolerance-guarded assertions

**Old pattern**, at the five sites named in TASK.md (line numbers as they
stood at task start): gas shadow price (1770), carbon shadow price, two
branches (1792, 1798), land shadow price (2149), public budget shadow
price (2169):

```python
val = None if val is None else max(0.0, val)   # or max(0.0, float(-raw))
```

**New pattern**, `DUAL_SIGN_TOL = 1e-6` added as a module constant
(`src/optimize_experiments.py`, just above
`extract_gas_shadow_prices_usd_per_twh_th`):

```python
if val is not None and val < -DUAL_SIGN_TOL:
    raise ValueError(
        f"<name> dual is negative after sign conversion: {val} in year {y}. "
        "This indicates a wrong sign convention, not solver noise."
    )
val = None if val is None else max(0.0, val)
```

Applied at all five named sites:
- `extract_gas_shadow_prices_usd_per_twh_th` — gas shadow price, per year.
- `extract_carbon_shadow_prices_usd_per_tco2` — carbon shadow price, both
  the indexed (annual caps, per year) and scalar (single cap, one message
  covering every year) branches.
- `extract_planning_diagnostics`, land shadow price loop (`land_shadow`)
  — per year, only on the branch where a dual was actually returned (the
  "constraint slack, no dual" branch is unaffected — it already raises
  its own `RuntimeError` if slack is negative).
- `extract_planning_diagnostics`, public budget shadow price
  (`public_budget_shadow`) — a single scalar constraint, so the message
  has no per-year component.

Small negatives within tolerance still clamp to `0.0` as before; a real
sign-convention error now raises instead of being silently zeroed.

**Line 913 (`financing_gap_per_mw = max(0.0, capex_2025 - max_bankable)`)
NOT touched** — per TASK.md, that is a genuine `max` (bounding a
financing gap at zero), not a sign-convention clamp.

**Not touched, found but out of scope.** `grep -n "max(0.0"
src/optimize_experiments.py` surfaced two more clamps of the same shape,
at lines 378 and 405, inside a different function (the grid-service-
obligation dual extractor, `_gso`/`reliability_constraint` branch —
unrelated to `extract_planning_diagnostics`). TASK.md's Defect 3 names
exactly five sites across two named metric families (gas/carbon shadow
prices; land/public budget). These two additional sites were not named
and were left untouched, per the task's explicit scope. Flagging here so
they are not lost track of.

---

## Defect 4 — `eaas_capital_multiplier` → three fields

**Old expression** (was ~line 2009-2028):

```python
public_spending = 0.0
private_investment = 0.0
for t in T:
    public_spending += pyo.value(m.DF[t]) * pyo.value(m.eaas_subsidy[t])
    private_investment += (
        pyo.value(m.DF[t]) * pyo.value(m.solar_eaas_add[t])
        * pyo.value(m.solar_capex_param[t])
    )
eaas_multiplier = None
if public_spending > 0:
    eaas_multiplier = private_investment / public_spending
```

returned as `"eaas_public_spending_usd": public_spending`,
`"eaas_private_investment_usd": private_investment`,
`"eaas_capital_multiplier": eaas_multiplier`.

Plan 0.3 asked for `(capex - subsidy) / subsidy`. **Not implemented**, per
TASK.md: F3 established `m.eaas_subsidy` is forced to zero whenever the
tariff covers CAPEX, true at every sourced tariff arm — so
`public_spending` is zero, the existing `if public_spending > 0` guard
already fires, and the plan's ratio would divide by zero in the central
configuration rather than being guarded.

**New expression** — three fields instead of one:

```python
# unchanged computation of public_spending / private_investment, renamed
# in the return dict (see below)

eaas_multiplier_vs_subsidy = None
if public_spending > 0:
    eaas_multiplier_vs_subsidy = private_investment / public_spending

# ... later, once public_solar_npv_spend / eaas_subsidy_npv_spend exist ...

eaas_multiplier_vs_public_budget = None
if hasattr(m, "public_budget_constraint") and econ is not None:
    storage_capex_npv_spend = sum(
        float(pyo.value(m.DF[t])) * float(pyo.value(m.storage_add[t]))
        * float(econ["STORAGE_COST_PER_MWH"])
        for t in T
    )
    public_budget_bound_spend = (
        public_solar_npv_spend + storage_capex_npv_spend + eaas_subsidy_npv_spend
    )
    if public_budget_bound_spend > 0:
        eaas_multiplier_vs_public_budget = (
            private_investment / public_budget_bound_spend
        )
```

returned as:
- `"eaas_private_investment_usd"` — unchanged.
- `"eaas_subsidy_npv_usd"` — RENAMED from `eaas_public_spending_usd` (same
  underlying quantity, `public_spending`). The old name read as "total
  public spending"; it is the EaaS subsidy NPV only — public solar CAPEX
  is the separate `ndc_finance_public_solar_npv_usd` field. Renamed per
  the same stale-reader rule as Defect 1.
- `"eaas_multiplier_vs_subsidy"` — the old ratio, renamed and now
  guaranteed `None` (not a divide-by-zero) when the subsidy is zero, "the
  normal case at sourced tariffs" per TASK.md's framing.
- `"eaas_multiplier_vs_public_budget"` — ADDED. `private_investment`
  divided by the discounted public solar CAPEX + storage CAPEX + EaaS
  subsidy — exactly the sum `m.public_budget_constraint` bounds
  (`src/optimize_model.py:1676-1685`,
  `src/optimize_model_sliced.py:1885-1894`, both identical: `DF[t] *
  (solar_public_add[t]*solar_capex_param[t] +
  storage_add[t]*STORAGE_COST_PER_MWH + eaas_subsidy[t])`). Guarded on
  `hasattr(m, "public_budget_constraint")` (no budget constraint applied
  this run ⇒ `None`) and on `econ is not None` (defensive — every current
  call site passes `econ` positionally, but the parameter itself defaults
  to `None`).

Findings F-26/F-27 establish that EaaS's mechanism is decoupling
deployment from the PUBLIC CAPITAL CEILING, not subsidy leverage — a
tightened budget shifts solar from public to private one-for-one at
unchanged system cost. `eaas_multiplier_vs_public_budget` measures that
mechanism directly; `eaas_multiplier_vs_subsidy` measures a channel that
is inactive at every sourced tariff. Both comments are in the code at
their computation sites and at the return-dict site.

**`eaas_multiplier_vs_public_budget` was deliberately NOT built from the
existing `realised_public_spend` variable** (computed a few lines below,
inside the same `hasattr(m, "public_budget_constraint")` block): that
variable omits storage CAPEX (`sum(DF[t] * (solar_public_add[t] *
solar_capex_param[t] + eaas_subsidy[t]))` — no `storage_add` term), so it
does not match what `m.public_budget_constraint` actually bounds. Using
it would have reproduced a second, subtler version of the same
"denominator doesn't match the real sum" defect this task is fixing.
`realised_public_spend` itself is unchanged — out of this task's scope
(Plan 0.3 names three defects; this is a fourth, undiscovered one,
flagged here rather than fixed silently).

**Consumers found** (`grep -rn "eaas_capital_multiplier\|eaas_public_spending_usd"
src/ scripts/ docs/ tests/`): none, anywhere, other than the definition
and return-dict sites just changed.

---

## Not touched (confirmed pre-existing, flagged during this task)

- `src/optimize_experiments.py:2171-2177`, `realised_public_spend` (FIN-3
  public budget utilisation diagnostic) — omits storage CAPEX from its
  sum, unlike `m.public_budget_constraint`'s actual bound. See Defect 4
  above. Not fixed here: outside Plan 0.3's three (four, with this
  finding) named defects.
- `src/optimize_experiments.py:378, 405` — two more `max(0.0, float(-raw))`
  sign clamps, in the grid-service-obligation dual extractor. See Defect
  3 above. Not fixed here: not among TASK.md's five named line numbers.
- `docs/phase_2_8_conversion.md`'s "Known reporting gap" note on
  `gas_generation_twh_e_by_year` / `gas_to_power_twh_th_by_year`
  excluding converted output — still true after this task (only the
  generation-share fields were fixed); noted as still-open in that file.
