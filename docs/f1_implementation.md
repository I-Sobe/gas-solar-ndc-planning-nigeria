# F1 — per-tranche CRF capital charge

**Status: STEP 0 ONLY. Stopping here per the task's explicit instruction
("STOP after step 0 and report"). No edits made — this is inventory only.
Steps 1–6 are not started.**

Reason to stop beyond the literal instruction: Step 0b (below) comes back
**silent/ambiguous** on the real-vs-nominal question, which the task
separately says is itself a stop condition ("if 0b comes back ambiguous,
flag and stop"). Both conditions point the same way.

---

## STEP 0a — Objective capex terms, salvage block, replacement-capex block

Both builders are structurally identical here — same variable names, same
comments, same formulas, at slightly different line numbers. Quoted from
`src/optimize_model.py`; `src/optimize_model_sliced.py` line numbers given
alongside where they differ.

### The four capex terms, one per buildable channel

**1. Public solar — `src/optimize_model.py:621-626`
(`src/optimize_model_sliced.py:799-801`):**
```python
    # Public solar CAPEX -- priced at each year's NREL ATB value (time-varying).
    public_solar_capex_npv = pyo.quicksum(
        m.DF[t] * m.solar_public_add[t]
        * m.solar_capex_param[t]
        for t in T
    )
```
Financing treatment today: undifferentiated overnight capex, discounted
at `m.DF[t]` (the SOCIAL rate — see §0b). No margin, no tranche split.

**2. EaaS (private) solar — `src/optimize_model.py:628-642`
(`src/optimize_model_sliced.py:802-804`, comment absent — see note below):**
```python
    # EaaS (private) solar CAPEX.
    # MODELLING CHOICE (state explicitly in the thesis): EaaS-financed and
    # publicly-financed solar are priced at the SAME unit CAPEX -- they are the
    # same physical assets. The private cost of capital is represented NOT by an
    # objective-side CAPEX premium but by the bankability constraint, which
    # discounts EaaS revenue at the private/blended rate via remaining_npv_factor.
    # Applying a required_margin multiplier here would double-count: required_margin
    # is the bankability coverage cushion, a distinct object from the cost of
    # capital. The EaaS mechanism's effect in this model is therefore structural
    # (budget decoupling + a harder bankability test), not a cheaper sticker price.
    eaas_solar_capex_npv = pyo.quicksum(
        m.DF[t] * m.solar_eaas_add[t]
        * m.solar_capex_param[t]
        for t in T
    )
```
**This is the double-count guard comment the task says to preserve** (task
cites "~L628"; confirmed at exactly `optimize_model.py:628-637`).
Financing treatment today: identical overnight capex to public solar,
discounted at the same social rate `m.DF[t]` — the *only* place the
private cost of capital currently enters is the separate bankability
constraint (`required_margin` dividing revenue — per
`docs/required_margin_inventory.md`, not re-inventoried here).

**Asymmetry noted, not fixed:** `optimize_model_sliced.py:802-804` has
**no comment at all** at this site — just the two bare `pyo.quicksum`
calls, no double-count rationale. The annual builder documents the
modelling choice; the sliced builder doesn't. Flagged for whoever does
Step 4 (the sliced builder's version of this comment will need writing or
copying, not just preserving).

**3. Gas — `src/optimize_model.py:998-1002`
(`src/optimize_model_sliced.py:1001-1003`):**
```python
    # Gas cost is in USD per TWh_th
    gas_capex_npv = pyo.quicksum(
        m.DF[t] * m.gas_add[t] * scenario["gas_capex_per_mw"]
        for t in T
    )
```
Financing treatment today: undifferentiated overnight capex (bare
`gas_capex_per_mw` scalar — sourced via `gas_capex_case`, phase 2.13),
discounted at the social rate. No margin, no tranche split. (`gas_add` is
currently pinned to 0 under `gas_build_case="gas_none"`, the only usable
arm per the 2.13 report — this term is structurally present but
numerically inert in every runnable scenario today.)

**4. Storage — `src/optimize_model.py:1009-1013`
(`src/optimize_model_sliced.py:1008-1010`):**
```python
    # Storage CAPEX: assume built in 2025 (t=0) unless you model staged build
    storage_capex_npv = pyo.quicksum(
    m.DF[t] * m.storage_add[t] * econ["STORAGE_COST_PER_MWH"]
    for t in T
)
```
Financing treatment today: undifferentiated overnight capex, discounted
at the social rate. No margin, no tranche split.

### Salvage block

**`src/optimize_model.py:1015-1115`** (identical in
`src/optimize_model_sliced.py:1011-1111`). Full text of the governing
comment and the expression:
```python
    # ------------------------------------------------------------
    # End-of-horizon salvage value (correction plan 2.5)
    # ------------------------------------------------------------
    # The horizon (21 yr) is shorter than every asset life, so without this
    # term assets with 9-29 years of remaining service are fully expensed and
    # system NPV is overstated. Straight-line residual, credited at T_last.
    #
    # Salvage is computed on RAW capex for BOTH arms: required_margin is a
    # financing premium, not asset value. It is subtracted from the objective
    # ONLY -- the public budget constraint below continues to see gross capex,
    # because the envelope constrains cash out the door.
    lives = scenario.get("asset_lifetimes", None)
    if lives is None:
        raise ValueError(...)
    ...
    def _salvage_fraction(t, life):
        """Straight-line residual fraction at end of horizon for vintage t. ..."""
        served = T_last - t + 1
        return max(0.0, float(life) - served) / float(life)

    m.salvage_npv = pyo.Expression(
        expr=m.DF[T_last] * (
            pyo.quicksum(  # original solar vintages
                (m.solar_public_add[t] + m.solar_eaas_add[t])
                * m.solar_capex_param[t] * _salvage_fraction(t, lives["solar"])
                for t in T
            )
            + pyo.quicksum( ... storage ... )   # original storage vintages
            + pyo.quicksum( ... gas ... )       # original gas vintages
            + pyo.quicksum( ... solar, rebuild vintages ... )
            + pyo.quicksum( ... storage, rebuild vintages ... )
            + pyo.quicksum( ... gas, rebuild vintages ... )
        )
    )
```
Reads `scenario["asset_lifetimes"]` (`asset_lifetimes()` in
`scenarios.py` — solar 30 / storage 15 / gas 30, IRENA 2026 / NREL ATB
sourced per the task's Context, **not** a capital-recovery period). Enters
`system_cost_npv` **subtracted** (see assembly below).

### Replacement-capex block

**`src/optimize_model.py:1049-1143`** (identical in
`src/optimize_model_sliced.py:1045-1143`):
```python
    # ------------------------------------------------------------
    # Replacement capex on in-horizon retirement (2.6, phase_2_3b Step 1).
    # A vintage built at build_t with technical life `life` reaches end of
    # life at build_t + life; if that falls inside the horizon, the asset
    # is rebuilt -- full capex charged again, discounted at m.DF[rebuild_t]
    # (the social rate in the year the rebuild occurs). Repeats for as many
    # rebuild cycles as fit before T_last. ...
    #
    # PROVISIONAL: Phase 3.0 F1 will annuitise capex at owner-specific WACC
    # via a capital recovery factor truncated at the horizon, superseding
    # both this rebuild-on-retirement treatment and the salvage credit
    # below. Do not extend this mechanism past phase_2_3b.
    # ------------------------------------------------------------
    def _rebuild_years(build_t, life):
        """rebuild_t for every in-horizon replacement of the vintage built
        at build_t, `life` years apart, for as many cycles as fit <= T_last."""
        ...

    m.cost_replacement_capex_npv = pyo.Expression(
        expr=(
            pyo.quicksum(  # solar
                m.DF[rebuild_t]
                * (m.solar_public_add[build_t] + m.solar_eaas_add[build_t])
                * m.solar_capex_param[rebuild_t]
                for build_t in T for rebuild_t in _rebuild_years(build_t, lives["solar"])
            )
            + pyo.quicksum( ... storage ... )
            + pyo.quicksum( ... gas ... )
        )
    )
```
The file's own comment already names F1 as the thing that supersedes this
block — confirms the Context claim without needing to re-verify structure.
Enters `system_cost_npv` **added** (as `m.cost_replacement_capex_npv`),
separately from the four capex terms above.

### Objective assembly — where everything is summed

**`src/optimize_model.py:1228-1240`** (identical in
`src/optimize_model_sliced.py:1206-1218`, same term order):
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
    )
    m.system_cost_npv = pyo.Expression(expr=system_cost_npv)
```
This confirms: **exactly four capex terms** (public solar, EaaS solar,
gas, storage), **one replacement-capex term** (added), **one salvage term**
(subtracted) — six capex-related additions to the objective today, all six
of which F1 (per the task) must resolve into `capital_charge[v]` per
tranche while removing salvage and replacement entirely. Confirms the
Context's claim ("prices capex as undifferentiated overnight capex") is
accurate for all four channels — no channel is currently financed
differently from any other in the objective.

### `solve_model()`'s storage post-solve guard (for the "Known open item")

**`src/optimize_model.py:1417-1451`** — the ONLY `solve_model()` in the
codebase (the sliced builder imports this same function; confirmed no
`solve_model` definition exists in `optimize_model_sliced.py`):
```python
    # ---- Post-solve guard: in-horizon retirement (plan 2.5)
    # Storage life (15 yr) is shorter than the horizon (21 yr). Any vintage
    # built early enough to retire before the horizon ends is still dispatched
    # by the model, and no replacement capex is charged. Inert as at Phase 2.5
    # (first storage build 2038) but Phases 2.6/2.7 may shift the optimum
    # earlier. This warns rather than raises so that exploratory runs proceed.
    try:
        _lives = scenario.get("asset_lifetimes", {})
        _T = list(range(len(scenario["years"])))
        _T_last = _T[-1]
        _checks = [
            ("storage", _lives.get("storage"), lambda t: pyo.value(model.storage_add[t])),
            ("solar",   _lives.get("solar"), ...),
            ("gas",     _lives.get("gas"), ...),
        ]
        for _name, _life, _getter in _checks:
            if not _life:
                continue
            for t in _T:
                if (_T_last - t + 1) > float(_life) and _getter(t) > 1e-6:
                    print(f"  WARNING [plan 2.5]: {_name} vintage ... retires within "
                          f"the horizon ... No replacement capex is charged and the "
                          f"asset is still dispatched. Salvage for this vintage is 0. "
                          f"Replacement capex is now required.")
                    break
    except AttributeError as _e:
        print(...)
    except Exception as _e:
        print(...)
```
**Observation, reported not fixed (this task doesn't touch solve_model()):**
this guard's own message ("No replacement capex is charged... Replacement
capex is now required") reads as if the replacement-capex mechanism
doesn't exist — but it does, added later at "2.6, phase_2_3b Step 1" (this
guard is labeled "plan 2.5," an earlier phase). The guard checks only
`(T_last - t + 1) > life` — whether the vintage's OWN service span exceeds
its life — without checking whether `_rebuild_years` already inserted a
compensating rebuild for it. So today, this guard can print a
misleading-sounding warning even in cases where replacement capex WAS
correctly charged. **Whether it still fires under F1, and what its
message should say once salvage/replacement are gone, is deferred** — per
the task, Steps 1–6 are not implemented in this turn, so there is nothing
yet to check it against. Flagged for the F1 implementation turn.

**RESOLVED (D6, `docs/d6_storage_guard.md` / `docs/limitations_register.md`
L18).** The guard was rewritten to test the actual invariant (a rebuild
year is generated whenever the fire condition holds — verified true for
every lifetime this project uses, arithmetically, not assumed) and its
message no longer claims capex is uncharged. See L18 for the full
closure record; the "still fires under f1_tranche" note below (§ "Storage
post-solve guard under f1_tranche") is superseded by it.

---

## STEP 0b — Nominal-or-real determination

**Finding: the evidence in this repository is silent on whether
`social_discount_rate`, solar/storage/gas capex, and fuel price are real
or nominal USD. One piece of textual evidence exists and points toward
"real" as an original intent, but it is stale, unconfirmed to apply to the
current parameter, and contradicted by nothing else because nothing else
addresses the question at all.** Per the task: this is reported, not
resolved, and nothing is deflated.

### Social discount rate

**`src/optimize_model.py:278-294`** (identical in
`src/optimize_model_sliced.py:373-`, same value, same lack of comment):
```python
    r_social = float(scenario.get("social_discount_rate", 0.08))
    # t=0 at start year
    df = {t: 1.0 / ((1.0 + r_social) ** t) for t in T}
    ...
    m.DF = pyo.Param(T, initialize=df, within=pyo.PositiveReals)
```
**`src/scenarios.py:2103`** (the only place `0.08` is set as a scenario
default):
```python
        "social_discount_rate": 0.08,   # objective DF (society's time preference)
```
**No comment anywhere states whether this 0.08 is real or nominal.**
Grepped `nominal|real rate|real discount|inflation|constant.year|base.year`
across every `src/*.py` file — zero hits tied to discounting.

### Capex and fuel-price provenance comments

**Solar — `src/io.py:223-227`:**
```python
    # SOLAR_CAPEX_PER_MW: 2025 anchor value (solar_low NREL ATB).
    # Used ONLY as fallback when solar_capex_by_year is not passed to build_model().
    # All runner scripts must pass solar_capex_by_year. Do not use this scalar
    # in any post-solve cost diagnostic when time-varying CAPEX is active.
    econ["SOLAR_CAPEX_PER_MW"] = to_float(solar_row["Solar_capex_usd_per_mw"])
```
No currency-year or real/nominal statement — "2025 anchor value" names a
*year of applicability*, not a *price basis*.

**Gas new-build capex — `src/scenarios.py:1201-1205`
(`gas_capex_scenarios()`, phase 2.13):**
```python
        gas_capex_low     (900,000 USD/MW): the pre-2.13 bare literal
            (2.9, Step 0.1 finding) -- never sourced, no CCGT/OCGT
            technology attribution. Retained as a DECLARED LOWER BOUND ...
```
(`gas_capex_central`/`gas_capex_high`, derived from NIRP 2024 Tables
19/22/23 per the same function's docstring, likewise carry no currency-year
or real/nominal statement.)

**Fuel price — `src/io.py:210-212`:**
```python
    gas_df = pd.read_csv(ROOT/"data/cost/processed/gas_cost.csv", thousands=",")
    gas_row = gas_df[gas_df["Scenario"] == gas_price_case].iloc[0]
    econ["GAS_COST_PER_TWH_TH"] = to_float(gas_row["total_usd_per_twh_th"])
```
No comment at all on price basis at the load site.

### Raw data provenance — `data/cost/raw/source_log.md`

The one document in the repo that explicitly names a real-vs-nominal
distinction. Section 6, quoted in full:
```
## 6. Discount rate

### Parameter
- **real_discount_rate**
- Units: fraction per year

### Source
- To be defined based on financing context and literature.
- Stored in: `data/cost/raw/`

### Status
- Defined for completeness; not yet applied in objective function.
```
**This is the single explicit "real" label anywhere in the repository.**
But: (a) it names a parameter `real_discount_rate`, which does not exist
anywhere in the current code — the live parameter is `social_discount_rate`
(§ above), a different name with no stated inheritance; (b) its own
"Status" line says "not yet applied in objective function," meaning this
was written before `social_discount_rate=0.08` was wired into `m.DF` — it
documents an *intent that predates* the current implementation, not a
description of it. It cannot be read as confirming today's `0.08` is real.

Sections 1-3 of the same document (solar CAPEX, storage CAPEX, gas fuel
cost) contain **no real/nominal statement** for any of the three cost
series — only source, extraction method, and unit-conversion notes.

`data/cost/README.md` is empty. `data/cost/raw/conversions.md` states only
unit-scaling conventions, no currency-basis statement. The raw ATB
workbook/PDF that `source_log.md` cites
(`data/cost/raw/atb/2024_v3_Workbook.xlsx`,
`data/cost/raw/atb/ATB_Electricity_2024.pdf`) **is not present in this
repository** — its own stated real/nominal convention (NREL ATB typically
publishes in real dollars of a stated base year, but this is general
knowledge about NREL's usual practice, not something this repo's files
assert) cannot be checked from files that exist here.

### Circumstantial pattern, reported as observation only — not a determination

Checked the processed CSVs directly for a decision-relevant pattern:
- `data/cost/processed/solar_capex.csv`, `solar_low`: **declines** every
  year, 2025 `$1,456,000` → 2045 `$603,000` (a falling-cost-curve
  trajectory, consistent with a technology learning curve).
- `data/cost/processed/storage_capex.csv`, `Storage_low`: same declining
  pattern, `$252,000` (2025) → `$146,000` (2045).
- `data/cost/processed/gas_cost.csv`: **flat** — `gas_low` is
  `$2.13`/MMBtu-equivalent (`total_usd_per_twh_th = 7,267,861.68`)
  unchanged in every row from 2025 through at least 2032.

A learning-curve decline is a real-cost phenomenon independent of general
inflation; a fuel price held perfectly flat for decades is more consistent
with a real (no-escalation) assumption than a nominal one (nominal prices
essentially always assumed to inflate over a 20-30 year horizon). **This
pattern is circumstantially suggestive of a real-terms convention
throughout the cost data, but it is inference from data shape, not a
statement anyone made in the repository — flagged as evidence, not
treated as an answer.**

### Verdict for Step 0b

**Silent, and the one document that isn't silent is stale and unconfirmed
to apply.** Per the task: nothing is deflated, `us_inflation` is not
added, and this is flagged as a stop condition rather than resolved by
inference. **A ruling from Zeus is needed before Step 1 proceeds**,
specifically on: (a) whether `social_discount_rate=0.08` is real or
nominal; (b) whether solar/storage/gas capex and fuel price series are
real or nominal, and if real, in what base year; (c) whether the new
`capital_recovery_case` rates (stated by the task as NOMINAL USD, sourced
from World Bank IDA Terms / IEA CoC Observatory) therefore need deflating
before entering the same objective as (a)/(b), or whether the objective
itself should be treated as already nominal throughout (in which case no
deflation is needed and the nominal CRF rates are already consistent with
it).

---

## Confirmation of things NOT done, per scope

- No file was edited. `git status` after this session shows no changes to
  any `src/*.py`, `scripts/*.py`, or `data/*` file — only this report was
  written.
- `required_margin` was not re-inventoried (per Context instruction);
  `docs/required_margin_inventory.md` is treated as authoritative for that
  question and referenced, not repeated.
- `asset_lifetimes()` was read only to confirm its stated scope (technical
  service life, explicitly not a capital recovery period) — not touched,
  not fed into anything.
- No placeholder values were invented for Steps 1+ rates/tenors, since
  Steps 1+ were not started.
- Nothing was run (no solver, no runner, no smoke suite).

## Next step

Awaiting Zeus's ruling on the Step 0b nominal-vs-real question before
proceeding to Step 1 (`capital_recovery_case`) and beyond, per the task's
explicit stop condition.

---

# Steps 1–7 implementation

Zeus's rulings on the Step 0 stop conditions (recorded in the task, not
repeated in full here): `required_margin` is REPLACED by `dscr_case` (a
pure DSCR coverage cushion, single value across financing regimes, not
part of the objective); the 1.05/1.10 split across financing regimes is
RETIRED as a double-count; capital recovery period is a NEW parameter
independent of `asset_lifetimes()`; the model is REAL-USD by convention,
so the Step-2 rates (NOMINAL USD) are deflated with a new `us_inflation`
key, and `social_discount_rate` is NOT deflated; F1 lands first at the
regression arm (`pre_f1_overnight` + `dscr_regression`).

All verification below is `python3 -m py_compile` / `python3 -m
compileall` and non-solving `build_model()` / `build_model_sliced()`
calls. **No solver, runner, or smoke script was executed.**

## Step 1 — `us_inflation`

`src/scenarios.py`: sentinel `_US_INFLATION_REQUIRED` (L1498); parameter
added to `load_scenario()` (L1521); presence-checked (not truthiness —
0.0 is legitimate) at L2009 (`if us_inflation is _US_INFLATION_REQUIRED:
raise TypeError`) with a range check (`-1.0 < us_inflation < 1.0`);
written to the scenario dict as `"us_inflation": float(us_inflation)`
(L2384 area, next to the other new keys). No default anywhere. No named
central value is fixed by `scenarios.py` — the task specified 0.020 as
the value to *use*, not to bake in as a default, so every one of the 11
wired runners takes it as a required CLI float (`--us-inflation`, no
`choices=`, since it is a scalar not an arm).

Builder-side presence guards (mirroring `load_scenario()`'s "deep"
validation, so a hand-built scenario dict that skips `load_scenario()`
still fails loudly): `src/optimize_model.py` L266–272, `src/
optimize_model_sliced.py` L377–383. Both check `"us_inflation" not in
scenario or scenario["us_inflation"] is None`.

## Step 2 — `capital_recovery_case`

`src/scenarios.py`: `capital_recovery_scenarios()` (L1303–1344) returns
`cr_central` / `cr_ida21` / `cr_comm_low` / `cr_comm_high`, each an
`{r_conc, L_conc, r_comm, L_comm}` dict, NOMINAL USD, with the exact
verbatim provenance the task specified (World Bank IDA Terms FY24Q3 for
`r_conc`/`L_conc`; IEA Cost of Capital Observatory base + analyst uplift
for `r_comm`; Afrigreen Debt Impact Fund for `L_comm`). Sentinel
`_CAPITAL_RECOVERY_REQUIRED` (L1482); parameter on `load_scenario()`
(L1519); validated at L1975 (`TypeError` if sentinel, `ValueError` if not
a known key); resolved to `_cr_terms` and written to the scenario dict as
five flat keys — `capital_recovery_case`, `r_conc_nominal`, `L_conc`,
`r_comm_nominal`, `L_comm` (L2368–2374 area) — flat because both CSV and
JSON runners read this dict, per CLAUDE.md's CSV-flat/JSON-nest rule.

Deflation happens exactly ONCE, at the `CRF()` call site inside
`build_model()`/`build_model_sliced()` (Step 4 below), not in
`scenarios.py` — `scenarios.py` carries the raw nominal rates unchanged.

Builder-side presence guards: `src/optimize_model.py` L247–255, `src/
optimize_model_sliced.py` L358–366.

## Step 3 — `dscr_case`

`src/scenarios.py`: `dscr_scenarios()` (L1347–1378) returns
`dscr_regression=1.10` / `dscr_central=1.30` / `dscr_low=1.20` /
`dscr_high=1.40`. Sentinel `_DSCR_REQUIRED` (L1488); parameter on
`load_scenario()` (L1520); validated at L1991; resolved to
`_dscr_value` and written as BOTH `"dscr_case": dscr_case` (the label)
and `"required_margin": _dscr_value` (the number every downstream reader
already expects) — L2361–2362. `required_margin` therefore still exists
as a key, still means the same thing (a revenue divisor in
`eaas_bankability`/`eaas_subsidy_limit`), and still does NOT enter the
objective — only its SOURCE changed, from three independent ad-hoc
defaults to one named, required arm.

**Removed** (all three, per the task's explicit list):
- `src/optimize_model.py` — `required_margin = scenario.get(
  "required_margin", 1.0)` → `required_margin = scenario["required_margin"]`
  (the 1.0 fallback is gone; presence is now enforced upstream by
  `load_scenario()`'s own guard, so a missing key raises there, not with
  a silently-substituted 1.0).
- `src/scenarios.py`'s own former unconditional `"required_margin": 1.10`
  scenario-dict write — replaced by the `dscr_case`-derived line above.
- `src/optimize_experiments.py`, `run_gas_regime_ndc_matrix()` (def at
  L2652): the `required_margin=1.10` kwarg is removed from the function
  signature; the docstring's `Parameters` entry for it is removed and
  replaced with a note (L2663–2665) that this function's `load_scenario()`
  call is pre-F1 dead code missing several other required kwargs
  (documented in earlier phase reports, not fixed further here); the
  conditional `scenario["required_margin"] = required_margin` write
  (previously gated on `financing_regime == "eaas"`) is removed.

Builder-side presence guards: `src/optimize_model.py` L256–265, `src/
optimize_model_sliced.py` L367–376.

### The other two 1.05/1.10 splits — found and retired during Step 7

The task named three defaults for removal. Wiring the 11 runners (Step 7)
surfaced two MORE live 1.05/1.10 conditional/unconditional splits that
were not on that named list, both instances of the exact double-count
Decision 2 describes (using `required_margin` as a proxy for
concessional-vs-commercial cost of capital, which `capital_recovery_case`
now carries directly via `r_conc`/`r_comm`). Both were removed, for
consistency with Decision 2 and with the three named removals — this is
a judgement call beyond the task's literal three-item list, flagged here
for Zeus to confirm or reverse:

- `scripts/03_run_ndc_eaas.py`: `NDC_CASES` carried `"required_margin":
  1.10` (unconditional) / `1.05` (conditional) per case. Removed from the
  dict; the per-case `scenario["required_margin"] = cfg["required_margin"]`
  override line removed; `required_margin` now comes solely from
  `load_scenario(dscr_case=...)`, constant across all four NDC cases.
- `scripts/04_run_fin2_tariff_sweep.py`: same pattern in its own
  `NDC_CASES` dict (5 entries, including `baseline_no_policy`). Same fix.
  Note: `required_margin` here also feeds
  `run_fin2_blended_bankability_sweep()`'s analytical T* formula
  (`src/optimize_experiments.py` L872, `threshold_usd_per_twh = capex_2025
  * required_margin / npv_energy`) — reads `base_scenario["required_margin"]`
  directly, so removing the per-case override and letting
  `load_scenario(dscr_case=...)` populate it flows through correctly with
  no change needed inside that function.

Left **untouched**, per the task's explicit "six no-tariff runners" scope
exclusion:
- `scripts/24_run_leakage_arm.py`, `scripts/25_run_obligation_trajectory.py`:
  `sc["required_margin"] = 1.10` under `financing_regime="public"` — inert
  (public financing never reads `required_margin`).
- `scripts/99c_run_genset_arm_sensitivity.py`: `FINANCING_ARMS` sets
  `required_margin: 1.10` for BOTH the `public_only` and `eaas` entries
  (not a 1.05/1.10 split — both arms carry the same value, and no tariff
  is ever set in this script, so even the `eaas` arm's bankability
  revenue is degenerate). Nothing to retire here.

## Step 4 — CRF helper, tranche Vars, split identity, envelope

Implemented identically in `src/optimize_model.py` (L1215–1392) and
`src/optimize_model_sliced.py` (L1225–1392, same structure), inside the
`else:` branch of `if capital_treatment_case == "pre_f1_overnight": ...
else: ...`.

- `_crf(r, L)` nested helper (`optimize_model.py` L1229–1244):
  `CRF = r(1+r)^L / ((1+r)^L - 1)`, with the declared negative-real-rate
  floor `CRF := 1/L` if `r <= 0`, and the required comment stating the
  floor biases AGAINST concessionality and makes the model insensitive to
  rate variation below inflation.
- Deflation at the `_crf()` call site, with the exact required comment:
  `# rates SOURCED nominal USD; model is real-USD (see register);
  deflated here, not at source.` (`optimize_model.py` L1252–1253,
  `optimize_model_sliced.py` L1254 area). `r_real = (1+r_nom)/(1+
  us_inflation) - 1`, applied separately to `r_conc` and `r_comm`.
  `_crf_conc`/`_crf_comm` are then plain Python floats — CONSTANTS, not
  Pyomo expressions — so every objective coefficient built from them is
  exogenous data and the LP stays linear.
- `_df_window_conc[v]` / `_df_window_comm[v]` (L1265–1272): per-vintage,
  per-tranche precomputed sum of `DF[t]` over `{v, ..., min(v+L-1,
  T_last)}` — exogenous data dicts (both `DF` and `L` are fixed before
  the solve), giving the truncation-at-horizon behaviour without any
  decision variable in the coefficient.
- Four tranche Var pairs + split-identity Constraints (L1285–1315):
  `m.solar_public_conc_vol`/`comm_vol`, `m.solar_eaas_conc_vol`/`comm_vol`,
  `m.gas_conc_vol`/`comm_vol`, `m.storage_conc_vol`/`comm_vol`, each
  `NonNegativeReals` over `T`, each with a `*_split_identity` constraint
  `conc_vol[t] + comm_vol[t] == unit_capex[t] * add[t]`.
- Envelope (L1324–1333): `m.f1_concessional_envelope_constraint`, built
  ONLY if `scenario.get("concessional_envelope_usd", None) is not None`
  — undiscounted commitment-stock sum across all four channels' `*
  _conc_vol`, `<= concessional_envelope_usd`. Deliberately independent of
  the PRE-EXISTING, EaaS-solar-only, NPV-discounted
  `concessional_envelope_npv`/`m.concessional_drawdown_npv`/
  `m.concessional_envelope_constraint` mechanism, which is left completely
  untouched (task listed it under "already exists — do not re-verify or
  rebuild").

## Step 5 — objective rewrite, reporting, double-count guard

- `m.concessional_volume_by_vintage` (L1341–1345): reporting Expression,
  sum of the four `*_conc_vol[t]` across channels, by vintage year —
  satisfies the task's "reporting Expression for concessional volume BY
  VINTAGE" requirement.
- Four per-channel capital-charge Expressions (L1356–1383):
  `m.cost_public_solar_capital_charge_npv`,
  `m.cost_eaas_solar_capital_charge_npv`, `m.cost_gas_capital_charge_npv`,
  `m.cost_storage_capital_charge_npv` — each
  `CRF_conc * df_window_conc[v] * conc_vol[v] + CRF_comm *
  df_window_comm[v] * comm_vol[v]`, summed over vintages. Deliberately
  NOT named `m.cost_<channel>_capex_npv` (the pre-F1 names) — a stale
  reader (`extract_planning_diagnostics`, which reads the old names
  unconditionally) fails loudly (`AttributeError`) under `f1_tranche`
  rather than silently reading a number that no longer means overnight
  capex, per CLAUDE.md's rename-on-meaning-change rule.
- `m.capital_charge_total_npv` (L1385–1392): sum of the four above.
- `system_cost_npv` assembly (`optimize_model.py` L1486–1507): an
  `if/else` on `capital_treatment_case`. The `pre_f1_overnight` branch is
  the ORIGINAL formula, written term-by-term in the ORIGINAL order (not
  refactored through an intermediate variable), specifically to preserve
  the exact left-fold floating-point evaluation order for bit-exact
  regression reproduction (Python's `+` is left-associative and
  floating-point addition is not strictly associative — even a
  mathematically-equivalent regrouping could shift the last bit). The
  `f1_tranche` branch is `gas_opex_npv + m.capital_charge_total_npv +
  selfgen_npv + unserved_npv + carbon_cost_npv` — no salvage term, no
  replacement term, no per-channel overnight-capex terms.
- Double-count guard comment, written into `src/optimize_model_sliced.py`
  (L862–874) for the first time — confirmed via grep before this task
  that this file had NO such comment, unlike `optimize_model.py:628-637`
  which did. Verbatim rationale preserved (required_margin is the
  bankability coverage cushion, not a CAPEX premium; applying it to
  objective-side CAPEX would double-count against the bankability
  constraint's own discounting at the private rate), with an added note
  that F1's per-tranche split, when active, replaces this block entirely.

## Step 6 — `capital_treatment_case` and the exactly-one invariant

`capital_treatment_case` is NOT one of the task's three named required
keys. Added because Step 6 ("the live objective must contain EXACTLY ONE
of {F1, salvage, replacement}") has no mechanical way to be satisfied
without an explicit, named selector between two materially different
objective forms — flagged as a judgement call, not a literal task
instruction, in the code comment at `src/scenarios.py` L1381–1390 and
here.

`CAPITAL_TREATMENT_CASE_CHOICES = ("pre_f1_overnight", "f1_tranche")` (a
plain tuple, no scenario-dict function, matching `new_gas_deliverability`'s
existing convention for a two-way binary arm). Sentinel
`_CAPITAL_TREATMENT_REQUIRED` (L1492); parameter on `load_scenario()`
(L1522); validated at L2028 (`TypeError` if sentinel) and L2036
(`ValueError` if not in the choices tuple); written to the scenario dict
(L2383) and to the `labels` dict.

**Confirmed exactly one of {F1, salvage, replacement} in the live
objective, both modes, both builders** (non-solving `build_model()` /
`build_model_sliced()` calls, `hasattr()` on the built model):

| | `pre_f1_overnight` | `f1_tranche` |
|---|---|---|
| `m.salvage_npv` | present | **absent** |
| `m.cost_replacement_capex_npv` | present | **absent** |
| `m.cost_gas_capex_npv` (and the 3 sibling `cost_<channel>_capex_npv`) | present | **absent** |
| `m.capital_charge_total_npv` | **absent** | present |

Verified directly (annual and sliced builders, both modes, `cr_central` /
`dscr_regression` / `us_inflation=0.020`):
```
pre_f1_overnight annual build OK. has salvage_npv= True  has cost_replacement_capex_npv= True  has cost_gas_capex_npv= True  has capital_charge_total_npv= False
pre_f1_overnight sliced build OK. has salvage_npv= True  has cost_replacement_capex_npv= True  has cost_gas_capex_npv= True  has capital_charge_total_npv= False
f1_tranche       annual build OK. has salvage_npv= False has cost_replacement_capex_npv= False has cost_gas_capex_npv= False has capital_charge_total_npv= True
f1_tranche       sliced build OK. has salvage_npv= False has cost_replacement_capex_npv= False has cost_gas_capex_npv= False has capital_charge_total_npv= True
```
Salvage/replacement's structural absence (not merely exclusion from the
objective sum) also satisfies CLAUDE.md's fail-loudly rule for the four
`cost_<channel>_capex_npv` reporting expressions, which
`extract_planning_diagnostics` reads unconditionally; `m.salvage_npv`'s
absence is already tolerated by pre-existing `hasattr(m, "salvage_npv")`
guards in `optimize_experiments.py` and `scripts/22_salvage_sensitivity.py`.

## Step 7 — wiring through the eleven runners

All eleven wired runners now carry four new required arguments —
`--capital-recovery-case`, `--dscr-case`, `--us-inflation`,
`--capital-treatment-case` — fed straight to `load_scenario()`, appended
to each script's output tag/filename, and recorded in each script's
summary/output dict. No default anywhere; an unstated value raises via
argparse before `load_scenario()` is ever reached.

| Script | Threading notes |
|---|---|
| `scripts/01_run_baseline.py` | direct wiring; no `required_margin` write in this script (confirmed by the earlier inventory) |
| `scripts/01b_run_baseline_sliced.py` | direct wiring, identical to 01 |
| `scripts/02_run_eaas.py` | direct wiring; removed the eaas-only `scenario["required_margin"] = 1.10` override (one of the task's three named removals) |
| `scripts/02_run_ndc_caps.py` | direct wiring; no `required_margin` write (public-only, no EaaS) |
| `scripts/03_run_ndc_eaas.py` | direct wiring; removed the `NDC_CASES`-based 1.10/1.05 split (see "other two splits" above) |
| `scripts/04_run_fin2_tariff_sweep.py` | direct wiring; removed the `NDC_CASES`-based 1.10/1.05 split (see above); confirmed the analytical-T* helper in `optimize_experiments.py` reads `base_scenario["required_margin"]` directly, so no change needed there |
| `scripts/22_salvage_sensitivity.py` | threaded through `run_arm()`'s signature/call sites; documented that this script's whole method (probing `m.salvage_npv`) requires `--capital-treatment-case pre_f1_overnight` — `f1_tranche` is offered only for CLI uniformity and has no salvage to probe |
| `scripts/23_run_selfgen_frontier.py` | threaded through `make_scenario()` and `solve_point()`'s signatures/call sites |
| `scripts/24_run_leakage_arm.py` | threaded through `solve_one()`'s signature/call sites; `sc["required_margin"] = 1.10` (public, inert) left untouched |
| `scripts/25_run_obligation_trajectory.py` | threaded through `solve_one()`'s signature/call sites; same inert `required_margin` line left untouched |
| `scripts/99c_run_genset_arm_sensitivity.py` | threaded through `solve_one()`'s signature/call sites; `FINANCING_ARMS`' matched 1.10/1.10 (not a split) left untouched |

Full-repo compile sweep after all eleven edits:
```
python3 -m compileall -q src scripts   ->  no output, exit 0
```

## Storage post-solve guard under `f1_tranche`

`solve_model()`'s in-horizon retirement guard (`src/optimize_model.py`
L1691–1725, "plan 2.5") reads `scenario.get("asset_lifetimes", {})` and
three model Vars: `model.storage_add[t]`, `model.solar_public_add[t]` +
`model.solar_eaas_add[t]`, `model.gas_add[t]`. All four of these —
`asset_lifetimes` and the three capacity-build Vars — are UNCONDITIONAL
on `capital_treatment_case`: they exist identically under both
`pre_f1_overnight` and `f1_tranche`, because F1 only changes how a
build's COST is charged to the objective (via the new `*_conc_vol`/
`*_comm_vol` split), not the underlying capacity-build Vars the guard
inspects. **The guard still fires under `f1_tranche`, unchanged, for the
same reason it fires under `pre_f1_overnight`.** Not fixed, per the
task's instruction to report only.

**Correction (D6, `docs/d6_storage_guard.md` / `docs/limitations_register.md`
L18): this note recorded only that the guard "still fires," not that its
MESSAGE was wrong.** The guard fired (and still fires, under both
branches) on an event — in-horizon retirement — that F2's
`m.cost_rebuild_capital_charge_npv` (the `f1_tranche` counterpart to
`pre_f1_overnight`'s `m.cost_replacement_capex_npv` named above) already
compensates correctly. The old message claimed capex was uncharged; it
was not, in either branch. Closed: the guard now tests the actual
invariant (a rebuild is generated) instead of the "life < horizon"
proxy, and its message states what is actually true. See L18.

## Placeholders needing a source

- `scenario["concessional_envelope_usd"]` — left at `None` (unconstrained)
  in `src/scenarios.py`. No numeric F1 envelope size was given by the
  task; `m.f1_concessional_envelope_constraint` is simply not built while
  this stays `None`. NOT the same quantity as the pre-existing
  `concessional_envelope_npv` (EaaS-solar-only, NPV-discounted), which
  already carries its own sourced values (`dares_only` / `dares_plus`)
  and is untouched by F1.
- The `r_comm` uplift baked into `capital_recovery_scenarios()`'s
  `cr_central`/`cr_comm_low`/`cr_comm_high` (2–2.5pp over the IEA
  Kenya/Senegal base, for worse NBET offtaker credit) is flagged in the
  function's own docstring as ANALYST ESTIMATE, NOT SOURCED — carried
  over from Step 2, not newly introduced here, but repeated here since
  the task asks for every placeholder needing a source in one place.

## Not done / explicitly out of scope this task

- The regression check itself (`pre_f1_overnight` + `dscr_regression`
  must reproduce the pre-F1 objective value to full precision) was NOT
  run — the task reserves this for Zeus. No solver was invoked anywhere
  in this session.
- New recovery periods (`cr_ida21`/`cr_comm_low`/`cr_comm_high`) were not
  exercised — inert under `pre_f1_overnight`, and the task explicitly
  scoped this task to `cr_central` only. This gap is noted, not closed.
- Sweeps across `dscr_case`/`capital_recovery_case` were not run — the
  task explicitly defers these to later.

## Confirmation of scope

- No file outside `src/scenarios.py`, `src/optimize_model.py`, `src/
  optimize_model_sliced.py`, `src/optimize_experiments.py`, and the
  eleven named `scripts/*.py` files was edited.
- `asset_lifetimes()` was not fed into the CRF anywhere — confirmed by
  the L_conc/L_comm/r_conc/r_comm plumbing above, which reads only
  `capital_recovery_scenarios()`.
- `social_discount_rate` was not deflated — only `r_conc_nominal`/
  `r_comm_nominal` pass through the `(1+r)/(1+us_inflation)-1` step.
- No values were invented beyond the two flagged placeholders above.
- Nothing was run: no solver, no runner, no smoke suite. Verification was
  `py_compile`/`compileall` plus non-solving `build_model()`/
  `build_model_sliced()` calls only.

## Envelope resolution

**Decisions.** Concessional eligibility is restricted to solar (public +
EaaS) and storage; gas is commercial-only. DARES is a
distributed-renewables facility and IDA credit for a CCGT is not a
defensible claim. Implemented as `gas_conc_vol[t].fix(0.0)` plus
exclusion from the envelope sum.

One envelope, one quantity. The pre-existing EaaS-solar-only mechanism
(`concessional_envelope_npv`, `m.concessional_drawdown_npv`,
`m.concessional_envelope_constraint`) is retired; the F1 envelope —
undiscounted commitment stock over solar and storage — replaces it.

**Why the envelope is load-bearing, not a detail.** Without it the
constraint never builds, and since CRF(r_conc) < CRF(r_comm) always, the
optimiser funds 100% of capacity concessionally. Endogenous capital
structure becomes structurally impossible. The degenerate all-concessional
solution also reproduces the terminal-year arbitrage F1 was intended to
remove: a 2054 vintage costs 0.0333 of DF x capex under
overnight+salvage against 0.0375 under F1 concessional. The envelope
fixes the contamination as well as the degeneracy.

**Provenance defect found during resolution.**
`src.blended_finance.concessional_envelope_npv()` applies no discounting
despite its name and its docstring describing the input as "NPV, USD" —
it is a bare `total_facility_usd * domestic_deployable_fraction`. The old
constraint was therefore already an undiscounted commitment stock,
mislabelled. Retirement is a rename-and-rescope, not a semantic reversal.
Logged: a parameter name asserting provenance it lacks.

**Expected behaviour.** $750M against system capex in the billions means
the envelope binds early and the marginal dollar is commercial for most
of the horizon. `env_5x` / `env_10x` are declared counterfactuals
establishing what envelope scale would change the pathway.

---

# Diagnostics branch and runner wiring

## Defect A — `extract_planning_diagnostics` branch on `capital_treatment_case`

**Already implemented and committed** (commit `f9672b8`, this session's
prior work): `src/optimize_experiments.py` reads
`capital_treatment_case = scenario["capital_treatment_case"]` (L2196) and
branches explicitly — no `hasattr`/`try-except` — reading the four
`cost_<channel>_capex_npv` names under `pre_f1_overnight` (L2204-2207,
same reads, same dict key names as before the branch existed, so that
path is byte-identical) and `m.capital_charge_total_npv` plus the four
`cost_<channel>_capital_charge_npv` names under `f1_tranche`
(L2215-2219). `concessional_volume_by_vintage_usd` and
`concessional_share_by_vintage` are emitted under `f1_tranche` only
(L2266-2285), `None` under `pre_f1_overnight`.

**Repo-wide grep for other unconditional readers of the four retired
names** (`cost_gas_capex_npv`, `cost_public_solar_capex_npv`,
`cost_eaas_solar_capex_npv`, `cost_storage_capex_npv`) across `src/*.py`
and `scripts/*.py`: the only hits are the four reads above, all inside
the `if capital_treatment_case == "pre_f1_overnight":` branch. No other
reader exists. Also checked for any downstream script indexing the
returned `cost_decomposition["components"]` dict by the retired key
names directly (e.g. `components["gas_capex_npv"]`) — none found.

**Precision fix made this task**: the per-vintage
`concessional_share_by_vintage` denominator (`src/optimize_experiments.py`,
in the `if capital_treatment_case == "f1_tranche":` block) previously
summed ALL FOUR channels including gas (`gas_conc_vol[t] +
gas_comm_vol[t]`), inconsistent with `m.concessional_volume_by_vintage`'s
own numerator, which a prior uncommitted edit (found on disk at the start
of this task, not made by this session) already narrowed to exclude
`gas_conc_vol` (gas is fixed at 0.0 and not concessional-eligible — see
envelope resolution Decision 1). Fixed the denominator to sum ONLY the
three concessional-eligible channels (solar public, solar EaaS, storage),
matching the task's "summed across eligible channels" instruction and
the numerator's existing scope. Also changed the zero-denominator guard
from `None` to `0.0` ("guarded zero" per the task's exact wording) — a
vintage with no eligible build in a year has no concessional draw either,
so 0.0 is the correct share, not an undefined one.

Edit: `src/optimize_experiments.py`, the `if capital_treatment_case ==
"f1_tranche":` block inside the "F1 concessional draw, by vintage"
section (renamed local var `_total_vol_t` -> `_eligible_vol_t`, gas terms
removed from the sum, `None` -> `0.0` in the guard).

## Step 3 fix — argparse `%` interpolation

Cause confirmed by reading Python's `argparse.HelpFormatter._expand_help()`:
every action's help string is run through `help_string % params`
(`params` carrying `prog`, `default`, etc.) regardless of formatter class,
so a literal `%` not part of a valid `%(name)s` specifier raises at
`--help` time. The `--concessional-envelope-case` help text's "100%
concessionally" (added when wiring the envelope arm through the eleven
runners) is exactly this: a bare `%` with no following format spec.

`grep -rn "100% concessionally" scripts/*.py` found this text in all
eleven already-wired runners (verbatim, copied from
`scripts/01b_run_baseline_sliced.py`'s reference block):
`01_run_baseline.py`, `01b_run_baseline_sliced.py`, `02_run_eaas.py`,
`02_run_ndc_caps.py`, `03_run_ndc_eaas.py`, `04_run_fin2_tariff_sweep.py`,
`22_salvage_sensitivity.py`, `23_run_selfgen_frontier.py`,
`24_run_leakage_arm.py`, `25_run_obligation_trajectory.py`,
`99c_run_genset_arm_sensitivity.py`. Fixed identically in each: `100%` ->
`100%%` in the `help=` string only (no other `%` in any of these files
was touched — f-strings, comments, and modulo expressions elsewhere are
unaffected). Verified by reading the fixed strings (`grep` confirms 0
remaining unescaped occurrences, 11 escaped); `--help` was NOT run, per
the task's execution ban.

The ten runners wired below (Defect B) are given the escaped `100%%`
form from the start, so this bug is not reintroduced.

## Defect B — the ten runners: NONE can be wired mechanically

**Finding: all ten are pre-existing dead code, predating F1 entirely —
none of them are wired.**

The task's premise was that these ten runners have the same argparse
scaffold as the eleven already-wired runners (peak_case,
storage_cc_case, gas_build_case, new_gas_deliverability,
gas_availability_case, gas_retirement_case, gas_capex_case,
ccgt_efficiency_case) and are simply missing the newer F1 additions.
Reading each of the ten (not grep alone) shows this premise does not
hold for any of them:

```
                                            argparse   peak_multiple_case  reserve_margin  storage_cc_case  gas_build_case  ccgt_efficiency_case
05_run_fin3_capital_sweep.py                  no             no               no              no              no                no
06_run_gas1_shadow_benchmarks.py              no             no               no              no              no                no
07_run_gas2_eaas_gas_relief.py                no             no               no              no              no                no
09b_run_rel1_mode_sensitivity.py              no             no               no              no              no                no
09b_run_rel1_retirement_sensitivity.py        no             no               no              no              no                no
15_run_pol1_ndc_comparison.py                 no             no               no              no              no                no
17_run_monte_carlo.py                         no             no               no              no              no                no
18_run_pareto_frontier.py                     no             no               no              no              no                no
19_run_carbon_price_sweep.py                  no             no               no              no              no                no
20_run_cf1_solar_yield_sensitivity.py         no             no               no              no              no                no
```

None of the ten import `argparse`. Every one calls `load_scenario()`
directly with only a small, fixed set of kwargs (typically
`demand_level_case`, `demand_case`, `gas_deliverability_case`,
`capital_case`, `carbon_case`, `start_year`, `end_year` — e.g.
`scripts/05_run_fin3_capital_sweep.py` L162-170,
`scripts/17_run_monte_carlo.py` L155-165). None pass
`peak_multiple_case`, `reserve_margin`, `storage_cc_case`,
`gas_build_case`, `gas_availability_case`, `gas_retirement_case`,
`gas_capex_case`, or `ccgt_efficiency_case`.

`src/scenarios.py` confirms all eight of these are sentinel-required
with NO default (L1573-1580: `peak_multiple_case=_PEAK_MULTIPLE_REQUIRED`,
`reserve_margin=_RESERVE_MARGIN_REQUIRED`,
`storage_cc_case=_STORAGE_CC_REQUIRED`,
`gas_build_case=_GAS_BUILD_REQUIRED`,
`gas_availability_case=_GAS_AVAILABILITY_REQUIRED`,
`gas_retirement_case=_GAS_RETIREMENT_REQUIRED`,
`gas_capex_case=_GAS_CAPEX_REQUIRED`,
`ccgt_efficiency_case=_CCGT_EFFICIENCY_REQUIRED`) — each of these
sentinels predates F1 by several phases (2.4, 2.3b, 2.9, 2.10, 2.11,
2.13; see the comment trail in `src/scenarios.py` and each phase's own
`docs/phase_2_*.md`). Every one of these ten scripts' `load_scenario()`
calls therefore already raises `TypeError` on the FIRST of these eight
missing arguments it hits, before the call ever reaches the F1 sentinel
checks. Adding the five new F1 keys — `--capital-recovery-case`,
`--dscr-case`, `--us-inflation`, `--capital-treatment-case`,
`--concessional-envelope-case` — mechanically, as instructed, would not
make any of these ten scripts runnable: they would still crash on the
pre-existing missing arguments, unchanged by this task.

This matches the task's own escape clause exactly ("its `load_scenario()`
call is dead code, or its structure differs materially from the
reference"): the ten runners' `load_scenario()` calls are already dead
code, structurally missing the ENTIRE pre-F1 argparse scaffold the
reference pattern depends on, not merely the five new keys asked for
here. Wiring only the five F1 keys onto a scaffold that does not exist
is not mechanical — it would require first building out eight more
required CLI arguments (peak_case, storage_cc_case, gas_build_case,
new_gas_deliverability, gas_availability_case, gas_retirement_case,
gas_capex_case, ccgt_efficiency_case) plus their `load_scenario()`
wiring and output-tag/summary threading, replicating the FULL scope of
the earlier per-runner wiring phases (2.4, 2.9, 2.10, 2.11, 2.13) on all
ten files. That is a materially larger task than "wire the ten runners"
as scoped here, and was not attempted.

**Left unwired, all ten:**
`scripts/05_run_fin3_capital_sweep.py`,
`scripts/06_run_gas1_shadow_benchmarks.py`,
`scripts/07_run_gas2_eaas_gas_relief.py`,
`scripts/09b_run_rel1_mode_sensitivity.py`,
`scripts/09b_run_rel1_retirement_sensitivity.py`,
`scripts/15_run_pol1_ndc_comparison.py`,
`scripts/17_run_monte_carlo.py`,
`scripts/18_run_pareto_frontier.py`,
`scripts/19_run_carbon_price_sweep.py`,
`scripts/20_run_cf1_solar_yield_sensitivity.py`.

No edits were made to any of these ten files for F1 wiring. (Each does
carry an unrelated, pre-existing uncommitted edit from phase 2.11 --
removing the `solar_min_build_mw_per_year` override / `solar_min_build_default`
import -- found already on disk at the start of this task and left
untouched, since it is not part of this task's scope.)

`scripts/probe_2_6_structure.py` was correctly excluded by the task as a
probe, not a runner — confirmed by reading it: no F1 keys, no argparse
scaffold either, same dead-code status, out of scope by the task's own
instruction regardless.

## Verification — single build call

The one permitted execution this task, run once:

```
sc = load_scenario(..., capital_recovery_case="cr_central",
                    dscr_case="dscr_regression", us_inflation=0.020,
                    capital_treatment_case="f1_tranche",
                    concessional_envelope_case="env_dares")
econ = load_econ("voll_mid", genset_case="genset_central")
capex = load_solar_capex_by_year("solar_low", 2025, MODEL_END_YEAR)
m = build_model_sliced(scenario=sc, econ=econ, emissions_cap=1e18,
                        emissions_cap_by_year=None, solar_capex_by_year=capex)
diag = extract_planning_diagnostics(m, sc, econ)   # attempted on the unsolved model
```

Result:
```
BUILD OK (no solve)
extract_planning_diagnostics FAILED on the unsolved model, as expected:
KeyError: "Index '0' is not valid for indexed component 'unserved'"
```

`build_model_sliced()` succeeds under `f1_tranche` with all five new
keys, as already established in the envelope resolution verification
(unchanged by this task's edits). `extract_planning_diagnostics` cannot
run on an unsolved model — it fails immediately (on an unrelated,
pre-existing sliced-model indexing issue with `m.unserved`, reached
before the code ever gets to the F1-specific branch this task changed),
confirming solved values are required, per the task's own anticipated
fallback. Correctness of the F1-specific branch (Defect A) is therefore
verified by reading the code path instead (done above): the
`if capital_treatment_case == "pre_f1_overnight": ... else: ...` branch
reads the correct Pyomo attribute names in each arm (confirmed present
under each mode via the earlier envelope-resolution `build_model_sliced()`
`hasattr()` checks, which remain valid — this task did not change which
attributes exist under which mode, only which are READ and how the
per-vintage share is computed), and the corrected
`concessional_share_by_vintage` denominator now sums exactly the three
concessional-eligible Vars (`solar_public_conc_vol`/`comm_vol`,
`solar_eaas_conc_vol`/`comm_vol`, `storage_conc_vol`/`comm_vol`) with no
gas terms, matching `m.concessional_volume_by_vintage`'s own (already
gas-excluded) numerator.

No solver, runner, smoke suite, `compileall`, or per-file compile check
was run at any point in this task — the single call above (a `build_model_sliced()`
call plus one attempted, non-solving `extract_planning_diagnostics()` call
in the same Python process) is the only execution performed.

---

# F2: asset lifetime restored

## Step 0 — inventory (no change made)

Structure confirmed to match what this task assumes in both builders.
Proceeding to Steps 1-3 (Decisions are already given; the "STOP if the
structure differs" condition does not apply — it matches exactly).

**`_L_conc`/`_L_comm` resolution and `_df_window_*` construction, inside
the `f1_tranche` branch (the `else:` of `if capital_treatment_case ==
"pre_f1_overnight":`):**
- `src/optimize_model.py` L1223-1249 (`_L_conc`/`_L_comm` at
  L1223-1224, `_r_conc_nom`/`_r_comm_nom`/`_us_inflation` at
  L1225-1227, deflation at L1231-1232, `_crf_conc`/`_crf_comm` at
  L1234-1235, `_df_window_conc`/`_df_window_comm` at L1242-1249).
- `src/optimize_model_sliced.py` L1243-1265 (same structure, same
  variable names, offset by ~10-16 lines from the annual builder).
- The `_crf(r, L)` helper itself: `optimize_model.py` L1206-1221,
  `optimize_model_sliced.py` L1226-1241.

**The existing `_rebuild_years()` helper and the `pre_f1_overnight`
replacement-capex Expression it feeds:**
- `src/optimize_model.py`: `_rebuild_years(build_t, life)` defined at
  L1111-1118, INSIDE the `if capital_treatment_case ==
  "pre_f1_overnight":` block (local to that scope — not visible from the
  `else:` branch). Feeds `m.salvage_npv` (L1120-1160, uses
  `_rebuild_years` at L1145/1151/1157) and `m.cost_replacement_capex_npv`
  (L1168-1188, uses `_rebuild_years` at L1175/1180/1185).
- `src/optimize_model_sliced.py`: identical structure,
  `_rebuild_years` at L1139-1146, `m.salvage_npv` at L1148-1188,
  `m.cost_replacement_capex_npv` at L1196-1216. Same local-scope
  confinement to the `pre_f1_overnight` block.
- Confirmed: `_rebuild_years` is a nested function local to the
  `pre_f1_overnight` if-block in both files. It is NOT visible to the
  `else:` (`f1_tranche`) branch under Python scoping rules. Since the
  task's "Do NOT" list forbids touching the `pre_f1_overnight` branch at
  all (not even a behaviour-preserving hoist of the function out of it),
  Step 2 below defines a second, textually-identical copy of
  `_rebuild_years` inside the `f1_tranche` branch rather than relocating
  or sharing the existing one. This is flagged here as a judgement call:
  "reuse the EXISTING helper" is satisfied in the sense that the
  algorithm is copied verbatim, unchanged, not reimplemented — but the
  function OBJECT is necessarily a duplicate, not the same Python object,
  because the two branches are mutually exclusive and Python scoping
  does not let the `else:` branch reach into the `if:` branch's locals
  without editing the `if:` branch (forbidden).

**Where `scenario["asset_lifetimes"]` is read today:**
- `src/optimize_model.py` L1070 (`lives = scenario.get("asset_lifetimes",
  None)`, inside `pre_f1_overnight`, with the presence/placeholder
  validation at L1071-1082).
- `src/optimize_model_sliced.py` L1098 (same, inside `pre_f1_overnight`,
  validation at L1099-1110).
- `src/optimize_model.py` L1690 (`_lives = scenario.get("asset_lifetimes",
  {})`, inside `solve_model()`'s post-solve in-horizon-retirement
  WARNING guard — unconditional on `capital_treatment_case`, since it
  reads capacity-build Vars that exist in both modes; already confirmed
  in the prior task to still fire correctly under `f1_tranche`; not
  touched by this task).
- Confirmed via `sed -n '1225,1400p' src/optimize_model_sliced.py | grep
  "lives\|asset_lifetimes\|_life"`: zero matches in that range (the
  `f1_tranche` branch), reproducing the task's own diagnostic exactly —
  `asset_lifetimes()` has no path into the F1 objective as things stand.

**The four split-identity constraints and the envelope constraint, inside
the `f1_tranche` branch:**
- `m.solar_public_split_identity`: `optimize_model.py` L1264-1268,
  `optimize_model_sliced.py` L1276-1280.
- `m.solar_eaas_split_identity`: `optimize_model.py` L1272-1276,
  `optimize_model_sliced.py` L1284-1288.
- `m.gas_split_identity` (with `gas_conc_vol[t].fix(0.0)` immediately
  above it, envelope resolution Decision 1): `optimize_model.py`
  L1286-1294, `optimize_model_sliced.py` L1298-1306.
- `m.storage_split_identity`: `optimize_model.py` L1296-1302,
  `optimize_model_sliced.py` L1308-1314.
- `m.f1_concessional_envelope_constraint`: `optimize_model.py`
  L1318-1325, `optimize_model_sliced.py` L1330-1337 (unconditional
  build, no `if ... is not None` guard, per envelope resolution).
- The four per-channel capital-charge Expressions
  (`m.cost_<channel>_capital_charge_npv`) and `m.capital_charge_total_npv`
  immediately follow, at `optimize_model.py` L1348-1384 and
  `optimize_model_sliced.py` L1357-1393.

## Steps 1-2 — technology-specific recovery windows + rebuild capital charge

Implemented identically in both builders, inside the `f1_tranche`
(`else:`) branch only — the `pre_f1_overnight` branch is untouched
(confirmed unchanged: `lives = scenario.get(...)` still reads at
`optimize_model.py` L1070 / `optimize_model_sliced.py` L1098, with no
underscore, exactly as before).

**Step 1** — `src/optimize_model.py` L1234-1298,
`src/optimize_model_sliced.py` L1254-1318 (offsets shift slightly file
to file but the structure is identical):
- A second, independent read of `scenario["asset_lifetimes"]` into
  `_lives` (underscore-prefixed to keep it visually distinct from the
  `pre_f1_overnight` branch's `lives`), with the same presence/placeholder
  validation as the original, restated as its own copy per the Step 0
  finding (Python scoping does not let the `else:` branch see the `if:`
  branch's locals, and the `if:` branch may not be edited to share it).
- `_L_eff_conc`/`_L_eff_comm`: per-channel (`solar`/`storage`/`gas`)
  dicts, `min(L_tranche, asset_lifetimes[channel])`, with the comment
  stating explicitly that `asset_lifetimes()` is a CEILING, never the
  recovery period itself.
- `_crf_conc`/`_crf_comm`: now per-channel dicts (were single scalars),
  each computed by the SAME `_crf(r, L)` helper, called with `L_eff`
  instead of the flat `_L_conc`/`_L_comm`. Both remain plain Python
  floats (not Pyomo expressions), so the Linearity invariant is restated
  and holds unchanged.
- `_df_window_conc`/`_df_window_comm`: now per-channel dicts of
  per-vintage dicts, built from `_L_eff_conc[ch]`/`_L_eff_comm[ch]`
  instead of the flat `_L_conc`/`_L_comm`.
- The four capital-charge Expressions
  (`m.cost_public_solar_capital_charge_npv`,
  `m.cost_eaas_solar_capital_charge_npv`, `m.cost_gas_capital_charge_npv`,
  `m.cost_storage_capital_charge_npv`) updated to index `_crf_conc["solar"|"gas"|"storage"]`
  and `_df_window_conc["solar"|"gas"|"storage"][v]` (and the `_comm`
  equivalents) instead of the old flat scalars/dicts. Both `solar_public`
  and `solar_eaas` use the `"solar"` channel key, matching
  `asset_lifetimes()`'s own three-key shape (solar/storage/gas — no
  public/EaaS distinction in technical life).

With `cr_central` (`L_conc=30`, `L_comm=10`) and the sourced central
`asset_lifetimes()` (solar 30, storage 15, gas 30 — confirmed by reading
`scenarios.asset_lifetimes()`), this produces exactly the result the
task's Decision 1 states: storage `L_eff_conc/L_eff_comm = 15/10`
(concessional window capped from 30 down to 15), solar and gas
`L_eff_conc/L_eff_comm = 30/10` (unchanged, since 30 already `<=` their
own life) — surgical, as claimed.

**Step 2** — `src/optimize_model.py` L1300-1312 (duplicated
`_rebuild_years`) and L1440-1476 (`m.cost_rebuild_capital_charge_npv`);
`src/optimize_model_sliced.py` L1320-1332 and L1453-1489 (same).
- `_rebuild_years(build_t, life)` duplicated VERBATIM (same algorithm,
  same docstring, same loop) into the `f1_tranche` branch, referencing
  the branch's own `_T_last` (identical value to the `pre_f1_overnight`
  branch's `T_last`, just independently computed, at
  `optimize_model.py` L1279 / `optimize_model_sliced.py` L1299 — this
  duplication is flagged in the Step 0 entry above as the judgement call
  it is).
- `m.cost_rebuild_capital_charge_npv`: a new Expression, structurally
  parallel to `pre_f1_overnight`'s `m.cost_replacement_capex_npv` (same
  `for build_t in T: for rebuild_t in _rebuild_years(build_t, lives[ch]):`
  double loop, same per-channel volume-at-rebuild-year pricing —
  `m.solar_capex_param[rebuild_t]` for solar, flat
  `econ["STORAGE_COST_PER_MWH"]`/`scenario["gas_capex_per_mw"]` for
  storage/gas) but COMMERCIAL-ONLY: each term is
  `_crf_comm[channel] * _df_window_comm[channel][rebuild_t] * volume`,
  using `_df_window_comm[channel][rebuild_t]` (already `DF`-discounted
  and truncated at the horizon, since `_df_window_comm[ch]` is defined
  for every `v` in `T` including every valid `rebuild_t`) — no new
  `*_conc_vol` variables are created, and the rebuild sum is not added
  anywhere near `m.f1_concessional_envelope_constraint`, which still only
  sums the original `*_conc_vol` Vars. Solar's public and EaaS builds are
  lumped into one rebuild term, matching `pre_f1_overnight`'s
  `m.cost_replacement_capex_npv` convention of not decomposing
  replacement by public/EaaS.
- `m.capital_charge_total_npv` extended to add
  `+ m.cost_rebuild_capital_charge_npv` as a fifth term.

## Step 3 — two reporting fixes in `scripts/01b_run_baseline_sliced.py`

**Fix 1 — "storage 2045" label vs value.** Chose to fix the LABEL, not
the value. Evidence: `MODEL_START_YEAR=2025`/`MODEL_END_YEAR=2054` (`src/
scenarios.py` L283-284) means `years` spans 2025-2054 (30 years), so
`len(years) - 1` indexes 2054, not 2045 — the "2045" label predates the
horizon-extension commit (`74d34f2`, "Extend model horizon to 2054 as
end-effect buffer"). The firm-capacity print loop a few lines below
(original L357, `if y == 2025 or y == 2054 or ...`) already correctly
uses 2054, confirming the rest of the file knows the true terminal year
and this one label is the stale leftover. Fixed at (original) L306:
```
-    print(f"  storage 2045   : {dv['final_storage_capacity_mwh']:,.0f} MWh")
+    print(f"  storage {int(years[-1])}   : {dv['final_storage_capacity_mwh']:,.0f} MWh")
```
Now reads the true terminal year dynamically instead of hardcoding a
stale one, so it stays correct if the horizon changes again.

**Same-defect check, as instructed ("report what you find" — not fixed,
narrower instruction than Fix 1):** the identical stale-"2045" pattern
recurs four more times in the same file, all keyed to `len(years) - 1`
under a literal `"2045"` name:
- `slice_2045 = slice_detail(m, years, len(years) - 1)` (original L313)
  — variable name says 2045, extracts the terminal (2054) slice.
- The comment above it, "Per-slice detail for 2025 and 2045
  (peak-adequacy inspection)" (original L310).
- `diag_out["slice_detail_2045"]` (original L377) — JSON key name says
  2045, holds the terminal-year slice.
- The per-slice CSV export loop, `for label, rows in (("2025",
  slice_2025), ("2045", slice_2045))` (original L408) — tags terminal-
  year rows with the literal string "2045" in `slice_detail__{tag}.csv`.

None of these four were changed: the task's instruction for "every other
line in that same block" was to check and report, not to fix, and fixing
the JSON key name (`slice_detail_2045`) in particular would change the
output schema of `diagnostics__{tag}.json` beyond the one line the task
named — left as found, flagged here for Zeus.

**Fix 2 — concessional-draw fields added to the summary dict.** This
runner does not call `extract_planning_diagnostics` at all (it is
annual-only and would crash on the sliced `[t,(s,p)]` indexing — see the
script's own pre-existing NOTE, original L239-241); it builds its
diagnostics directly from `annual_totals()` plus the (unchanged) annual
Pyomo variables. Added a new block, guarded on
`scenario["capital_treatment_case"] == "f1_tranche"`, computing
`concessional_volume_by_vintage_usd` and `concessional_share_by_vintage`
directly from `m.concessional_volume_by_vintage` and the six eligible
`*_conc_vol`/`*_comm_vol` Vars (solar public, solar EaaS, storage — gas
excluded, matching the same eligible-channel scope used in
`extract_planning_diagnostics`'s fix from the prior task), both `None`
under `pre_f1_overnight`. Added as two new keys,
`"concessional_volume_by_vintage_usd"` and
`"concessional_share_by_vintage"`, to the `summary` dict (written to
`summary__{tag}.json`) — not to `diag_out`, per the task's explicit
"summary dict" wording.

## Verification — single build call

The one permitted execution this task, run once:

```
sc = load_scenario(..., capital_recovery_case="cr_central",
                    dscr_case="dscr_regression", us_inflation=0.020,
                    capital_treatment_case="f1_tranche",
                    concessional_envelope_case="env_dares")
econ = load_econ("voll_mid", genset_case="genset_central")
capex = load_solar_capex_by_year("solar_low", 2025, MODEL_END_YEAR)
m = build_model_sliced(scenario=sc, econ=econ, emissions_cap=1e18,
                        emissions_cap_by_year=None, solar_capex_by_year=capex)
```

Result:
```
BUILD OK (no solve)
asset_lifetimes: {'solar': 30.0, 'storage': 15.0, 'gas': 30.0}
L_conc (tranche): 30  L_comm (tranche): 10
has cost_rebuild_capital_charge_npv: True
L_eff_conc: {'solar': 30, 'storage': 15, 'gas': 30}
L_eff_comm: {'solar': 10, 'storage': 10, 'gas': 10}
```

Confirmed exactly as required: `build_model_sliced()` builds cleanly
under `f1_tranche` with the F2 changes; the per-channel concessional
window differs by technology (storage capped to its 15-year technical
life, solar and gas remain at the tranche's own 30-year window since
their technical life does not bind) — matching Decision 1's stated
result (storage 15/10, solar 30/10, gas 30/10) exactly; and
`m.cost_rebuild_capital_charge_npv` exists on the built model.

No solver, runner, smoke suite, `compileall`, per-file compile check, or
`--help` was run at any point in this task — the single
`build_model_sliced()` call above is the only execution performed.
