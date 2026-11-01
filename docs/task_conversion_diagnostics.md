# task_conversion_diagnostics — the diagnostics layer is conversion-blind

Branch `phase-fin2-d17`. Model frozen at `2a9226b` (confirmed ancestor of
current HEAD `f4dd48d` via `git merge-base --is-ancestor`) — every
finding below reads code that has not moved since the anchors were set.
Anchors NOT reproduced or perturbed here: regression
21,811,042,472.821114 / 45,502.9754 MWh (`om_none`), central 27.954 bn /
65,613.0 MWh. Nothing in this report runs a solver, a builder, or any
script.

---

## PART 1 — INVENTORY

### 1A — `_capex_at()` in full, and whether conversion capex is available in the same form

Quoted verbatim, `src/optimize_experiments.py:3074-3084`:

```python
def _capex_at(t):
    v = 0.0
    if hasattr(m, "solar_public_add"):
        v += float(pyo.value(m.solar_public_add[t])) * float(pyo.value(m.solar_capex_param[t]))
    if hasattr(m, "solar_eaas_add"):
        v += float(pyo.value(m.solar_eaas_add[t])) * float(pyo.value(m.solar_capex_param[t]))
    if hasattr(m, "storage_add") and econ is not None:
        v += float(pyo.value(m.storage_add[t])) * float(econ["STORAGE_COST_PER_MWH"])
    if hasattr(m, "gas_add") and "gas_capex_per_mw" in scenario:
        v += float(pyo.value(m.gas_add[t])) * float(scenario["gas_capex_per_mw"])
    return v
```

Every existing branch is the same shape: an `hasattr`-guarded **per-year
flow variable** (`solar_public_add[t]`, `solar_eaas_add[t]`,
`storage_add[t]`, `gas_add[t]`) times a **per-unit cost** that is either
a per-year `Param` (`solar_capex_param[t]`) or a scalar pulled from
`econ`/`scenario`. Called once per `t` and discounted by the caller
(`m.DF[t]`), this reconstructs raw, undiscounted-then-discounted gross
capex — "quantity built at t × unit cost" — with no CRF annuitisation.

**Is conversion capex available in the same form?** Almost, with one
structural difference the fix must respect:

- **Quantity**: `m.convert_mw[tier]`, `tier ∈ {A, B, C}`
  (`optimize_model_sliced.py:631-635`) — but this is declared
  **time-invariant** ("ONE-OFF … Decision 5", line 623), not a `Var(T)`
  like `gas_add`. There is no `convert_mw[t]` to index by year; the
  entire retrofit decision is a single scalar per tier, structurally
  assigned to vintage 0 in the objective (`_df_window_comm["gas"][0]`,
  `optimize_model_sliced.py:1780-1802` — see 1B).
- **Unit cost**: `scenario["conversion_capex_per_kw"]`, confirmed a
  **single scalar** sourced from the conversion arm
  (`src/scenarios.py:3200`, `_conversion_arm["capex_per_kw"]`), **not
  tier-dependent** — `gas_conversion_tiers()` varies tiers only by
  `nameplate_mw` and `paf` (availability), never by cost
  (`src/scenarios.py:1111-1157`). So this is NOT the tiered structure
  plan 2.8 might suggest from the three-tier framing; cost is uniform
  across tiers, only quantity and availability vary.

So conversion capex is a raw quantity × uniform unit cost, exactly
`_capex_at()`'s pattern in kind — but a one-off scalar rather than a
per-year flow, so it can only ever contribute at `t == 0`, not be
added into the function's existing per-year branches unchanged.

### 1B — Where the objective charges conversion capex; simple cost×quantity or tier-dependent?

Quoted verbatim, `optimize_model_sliced.py:1792-1810`:

```python
if _conv:
    m.convert_comm_vol = pyo.Var(_CONV_KEYS, domain=pyo.NonNegativeReals)
    m.convert_split_identity = pyo.Constraint(
        _CONV_KEYS,
        rule=lambda m, tier:
            m.convert_comm_vol[tier]
            == _conv_capex_per_kw * 1000.0 * m.convert_mw[tier]
    )
    m.cost_conversion_capital_charge_npv = pyo.Expression(
        expr=pyo.quicksum(
            _crf_comm["gas"] * _df_window_comm["gas"][0]
            * m.convert_comm_vol[tier]
            for tier in _CONV_KEYS
        )
    )
    _capital_charge_total = (
        _capital_charge_total + m.cost_conversion_capital_charge_npv
    )
m.capital_charge_total_npv = pyo.Expression(expr=_capital_charge_total)
```

Two-step, not tier-dependent in rate (same `_conv_capex_per_kw` for all
three tiers, per 1A):

1. `convert_comm_vol[tier] == conv_capex_per_kw * 1000 * convert_mw[tier]`
   — **raw capex**, cost × quantity, per tier. This is the same
   quantity `_capex_at()` should read.
2. `cost_conversion_capital_charge_npv = Σ_tier CRF_comm["gas"] ×
   DF_window_comm["gas"][0] × convert_comm_vol[tier]` — CRF-annuitised
   into an NPV **capital charge** using vintage-0 discounting
   (`[0]`, not `[t]`, confirming the one-off/vintage-0 assignment from
   1A), commercial-only (gas is not concessional-eligible).

This is the SAME two-step pattern `gas_add`'s own capital charge uses
elsewhere in the objective (`_crf_comm["gas"] * _df_window_comm["gas"][rebuild_t] * gas_add[build_t] * gas_capex_per_mw`,
lines 1768-1773) — conversion's version is simpler only in that
`convert_mw` has no rebuild loop (one-off, no vintage recurrence).

**Consequence for the fix**: `_capex_at()` computes a *different*
quantity than `capital_charge_total_npv` — raw gross spend vs a
CRF-annuitised financing charge — by design (the "reporting-window
gross CAPEX" comment at `optimize_experiments.py:3055-3073` states this
explicitly: gross and window-limited, not the objective's own
annuitised form). So `_capex_at()` cannot simply read
`cost_conversion_capital_charge_npv`. But it also should not
re-derive `conv_capex_per_kw * 1000 * convert_mw[tier]` in a second,
independent expression (drift risk, per TASK.md's instruction) — it
should read `pyo.value(m.convert_comm_vol[tier])`, the Expression the
builder already exposes for exactly this raw-capex quantity, summed
over tiers, applied once at `t == 0`.

### 1C — `npv_capex_report` / `npv_capex_full`: built from `_capex_at()`, conversion absent from the total

Quoted verbatim, `src/optimize_experiments.py:3086-3088`:

```python
_report_T = [t for t, y in enumerate(years) if int(y) <= REPORT_END_YEAR]
npv_capex_report = sum(float(pyo.value(m.DF[t])) * _capex_at(t) for t in _report_T)
npv_capex_full = sum(float(pyo.value(m.DF[t])) * _capex_at(t) for t in range(len(years)))
```

Both are plain sums of `DF[t] * _capex_at(t)` over the reporting window
and full horizon respectively. Since `_capex_at()` (1A) has no
`convert_mw` branch, and conversion's vintage is `t == 0` which falls
inside both `_report_T` and the full range, **conversion retrofit capex
is absent from both totals outright, not merely from a sub-breakdown.**
These feed the dict keys `npv_gross_capex_report_window_usd` and
`npv_gross_capex_full_horizon_usd` (`optimize_experiments.py:3239-3240`).

### 1D — `gas_to_power_twh_th_by_year`: why converted fuel is excluded, and whether it is obtainable

Quoted verbatim, `src/optimize_experiments.py:3178-3179`:

```python
"gas_to_power_twh_th_by_year":
    {int(y): float(_slice_total(m.gas_to_power, t)) for t, y in enumerate(years)},
```

`_slice_total(m.gas_to_power, t)` sums the **fleet-only** thermal-draw
variable `m.gas_to_power[t, (s,p)]` over the six slices
(`_slice_total`, `optimize_experiments.py:2373-2377`). Converted
plant's thermal draw lives in a structurally separate variable,
`m.convert_to_power[tier, t, (s,p)]` — "thermal, TWh_th"
(`optimize_model_sliced.py:636-638`), NOT a further indexing of
`m.gas_to_power`. `m.convert_generation[tier,t,(s,p)] = eta_retrofit *
m.convert_to_power[tier,t,(s,p)]` (line 782-786) is the electrical
output; the thermal input this field is supposed to measure is
`convert_to_power` itself, one level upstream. So the exclusion is
structural, not an oversight in the reduction: this field reads exactly
one of two coexisting gas thermal-draw variables and was never extended
to the second when Phase 2.8 added it.

**Is converted thermal input obtainable?** Yes, directly, and there is
already a working precedent for the identical fix on the electrical
side in this same file. Quoted, `optimize_experiments.py:2479-2492`
(Plan 0.3 defect 1, solar share of grid generation):

```python
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
```

This already sums `m.convert_generation` across all `(tier, s, p)` for
a given `t` by filtering on `k[1] == t` (index tuple is
`(tier, t, (s,p))`), guarded by `hasattr(m, "convert_generation")` so
it returns 0.0 under `conv_none` or the annual builder. The identical
pattern applied to `m.convert_to_power` instead of
`m.convert_generation` gives total converted thermal draw per year —
no new model-side quantity is needed, and the guard is already the
established idiom in this file for this exact model feature.

### 1E — `cost_decomposition["components"]`: capital lines, and whether `capital_charge_total_npv` includes conversion

`capital_treatment_case == "pre_f1_overnight"` branch
(`optimize_experiments.py:2904-2910`) lists: `gas_capex_npv`,
`public_solar_capex_npv`, `eaas_solar_capex_npv`, `storage_capex_npv`,
`replacement_capex_npv`. Conversion is structurally impossible here —
`conversion_available=True` requires `capital_treatment_case=
"f1_tranche"` (guarded at both `scenarios.py:3183-3193` and
`optimize_model_sliced.py:494-500`) — so this branch is correctly
conversion-free by construction, not a defect.

`capital_treatment_case == "f1_tranche"` branch
(`optimize_experiments.py:2941-2948`), the one that matters — quoted:

```python
_capital_components = {
    "gas_capital_charge_npv":          round(cost_gas_capital, 0),
    "public_solar_capital_charge_npv": round(cost_pub_solar_capital, 0),
    "eaas_solar_capital_charge_npv":   round(cost_eaas_solar_capital, 0),
    "storage_capital_charge_npv":      round(cost_storage_capital, 0),
    "rebuild_capital_charge_npv":      round(cost_rebuild_capital, 0),
    "capital_charge_total_npv":        round(cost_capital_total, 0),
}
```
where `cost_capital_total = float(pyo.value(m.capital_charge_total_npv))`
— **read directly off the model's own Expression**, not re-summed from
the five named channels. **No `conversion_capital_charge_npv` key
exists.** Confirmed by inspection of every line in this branch and by
grep for `conv` / `retrofit` across the function: absent.

**Does `capital_charge_total_npv` include conversion?** Yes — per 1B,
`optimize_model_sliced.py:1807-1810` folds
`m.cost_conversion_capital_charge_npv` into `_capital_charge_total`
whenever `_conv` is true, and `m.capital_charge_total_npv` is built
from that same accumulator. So on an `f1_tranche`, conversion-active
run: the **total is right**, but the **five itemised lines will not sum
to the sixth (`capital_charge_total_npv`) entry that sits alongside
them in the same dict** — by exactly the conversion capital charge,
unlabelled. This is a discrepancy internal to `components`, not between
`components` and `cost_total_npv` (see below).

**Size class.** `cost_conversion_capital_charge_npv` is a CRF-annuity on
`conv_capex_per_kw * 1000 * Σ_tier convert_mw[tier]`. TASK.md's
established background figure for raw conversion capex at central is
"implied ~2.27 bn"; the annuitised capital-charge NPV that actually
enters `capital_charge_total_npv` will differ from that raw figure by
the commercial CRF/window-discount factor (`_crf_comm["gas"] *
_df_window_comm["gas"][0]`), which is <1 for a multi-year annuity
evaluated at build year — so the discrepancy is bounded by, and likely
smaller in magnitude than, ~2.27 bn, but cannot be pinned to a specific
number without reading `_crf_comm["gas"]` and
`_df_window_comm["gas"][0]`'s solved values, which is a run, not a
structural fact. Stating the class only, as instructed.

**A stale comment, found in the course of this inventory, not asked
for but directly relevant to 1E.** Immediately above this branch,
`optimize_experiments.py:2916-2938` argues at length that reading
`m.capital_charge_total_npv` directly (rather than re-summing the four
named channels) is safe specifically because "the SLICED builder's own
`capital_charge_total_npv` Expression folds in a further conversion-capex
term this annual-only function never lists — **confirmed this function
is called only on `build_model()` models, never `build_model_sliced()`,
by checking every one of its ~25 call sites.**" That confirmation is
now **false**. `scripts/18_run_pareto_frontier.py:352` and (by the same
pattern) `scripts/28_run_capital_frontier.py` call `build_model_sliced()`
and then `extract_planning_diagnostics(m, scenario, econ)`
(`18_run_pareto_frontier.py:387`) directly on the result, and both are
capable of an active conversion arm (`18_`'s own header: "BUILDER:
build_model_sliced … CONVERSION-ACTIVE baseline (conv_central/conv_low/
…)", lines 12, 265). Both call sites postdate the comment: they were
added by `f4dd48d` ("Port both frontiers to build_model_sliced"), a
commit *after* the one that wrote the "~25 call sites" claim
(`ae4fd1d`, "Close cost_decomposition reconciliation … register L28" —
confirmed via `git log`, the comment's own commit precedes `f4dd48d` in
the branch history). The comment was accurate when written and has been
silently invalidated by a later, unrelated port. This is the exact
"model feature correctly wired, measurement layer never told" pattern
TASK.md describes, one level up: the pattern already claimed one
comment's own factual assertion as a casualty before the four items
below were even found. Not fixing the comment (out of scope — Part 2
touches `_capex_at()`/`cost_decomposition` construction, not this
adjacent prose) but flagging it, since a future reader trusting the
comment would misjudge the risk surface.

**Consumers of `components` specifically** (not `cost_decomposition` as
a whole — see 1F) confirm this is presently a **latent**, not active,
defect: no script anywhere sums `components`' values. Grepped
`\[.components.\]` / `.get\(.components.` across `scripts/` and `src/`:
the only hit is `scripts/19_run_carbon_price_sweep.py:212`, which reads
a single named sub-key (`carbon_cost_npv`), never a sum. So no
published number is wrong yet from this specific defect — same
"inert until consumed" status TASK.md's background section already
established for items 1 and 2.

### 1F — Every consumer of each field

**`npv_capex_report` / `npv_capex_full`** (dict keys
`npv_gross_capex_report_window_usd` / `npv_gross_capex_full_horizon_usd`):
one consumer found anywhere in `scripts/` or `src/`:
`scripts/01_run_baseline.py:436`
(`"npv_gross_capex_report_window_usd": diag.get(...)`). `01_
run_baseline.py` uses `build_model()` (annual) exclusively — annual
build raises on any available conversion arm
(`optimize_model.py:293-297`) — so **this one live consumer can never
actually see the defect**; it is structurally conv_none-only. The
`findings_frozen.md`-attributed reporting rule ("report
`npv_gross_capex_report_window_usd` … as the financing headline, never
`system_cost_npv`") is textually in `docs/frozen_state_2026_09_10.md:
363-364`, not `docs/findings_frozen.md` itself (grepped both; no hit in
the latter) — a naming mismatch in TASK.md worth noting, though the
rule itself and the key it names are correct. **`18_`/`28_` — the only
two sliced, conversion-capable runners — do not read either field at
all** (grepped both files for `npv_capex` / `npv_gross_capex`: no hit).
So the defect is real and will fire the day either frontier script (or
any future sliced consumer) is pointed at `npv_gross_capex_..._usd`
under an active conversion arm; today it is silent because nothing
reads it there.

**`cost_decomposition["components"]`**: see 1E — one consumer, one
named sub-key, immune. `cost_decomposition` as a WHOLE (its top-level
keys, not `components`) is read more widely:
`scripts/16_plot_cost_decomposition.py:97-118` (reads
`real_expenditure_npv`, `voll_penalty_npv`, `selfgen_npv`,
`salvage_npv` — all sourced from `m.capital_charge_total_npv`
indirectly via `cost_capital_total`, so already conversion-correct at
the total level, per 1E), `scripts/18_run_pareto_frontier.py:387-388`,
`scripts/19_run_carbon_price_sweep.py:167-168,212`,
`scripts/21_compute_resolution_premium.py:180-186` (top-level keys
only), `scripts/deferred/17_run_monte_carlo.py:225` — none of these sum
`components`.

**`gas_to_power_twh_th_by_year`**: the widest-consumed of the three.
- `src/optimize_experiments.py:118` — `diagnostics_to_timeseries_df()`,
  a generic CSV-row builder used by annual-only callers.
- `src/optimize_experiments.py:3777` — inline in the GAS-3 annual NDC
  feasibility sweep (`08_run_gas3_regime_ndc_feasibility.py`'s call
  path), annual builder only.
- `scripts/00_build_emissions_cap.py:131,139` — **the Part 3 target**,
  reads the bare `results/baseline/diagnostics.json`.
- `scripts/01_run_baseline.py:460` — annual (`build_model()`), immune.
- `scripts/02_run_eaas.py:389` — annual (`build_model()`), immune.
- `scripts/02_run_ndc_caps.py:551` — **`build_model_sliced()`**
  (`02_run_ndc_caps.py:18-19,323`), conversion-capable.
- `scripts/03_run_ndc_eaas.py:614` — **`build_model_sliced()`**
  (`03_run_ndc_eaas.py:19,353`), conversion-capable.
- `scripts/01b_run_baseline_sliced.py:555` — **`build_model_sliced()`**,
  but NOT via `extract_planning_diagnostics()` at all. It builds the
  same-named key independently: `"gas_to_power_twh_th_by_year":
  annual_series("gas_to_power")` (line 555), where `annual_series`
  wraps the same fleet-only `m.gas_to_power` slice-sum
  (`_slice_total`'s sliced-builder equivalent). **This is a second,
  separate producer of the identical field name with the identical
  defect**, not a consumer of the one function this task's whitelist
  can edit. Flagged here; carried to the "encountered but not asked
  about" list below, since fixing it means editing
  `01b_run_baseline_sliced.py`, outside this task's whitelist.

So: three of eight production consumers of `gas_to_power_twh_th_by_year`
(`02_run_ndc_caps.py`, `03_run_ndc_eaas.py`, and the independent
producer in `01b_run_baseline_sliced.py`) can be conversion-active
today and would silently understate. `00_build_emissions_cap.py`'s own
exposure is the subject of Part 3, and is structurally different: it
reads a **frozen, already-annual, already-`conv_none`** file, so the
defect in the sliced-builder's field does not reach it *as configured
today* — see Part 3.

---

## SUMMARY: NOTHING HERE CAN MOVE A SOLVED VALUE

Every quantity examined above is either (a) read AFTER `m` is already
solved (`extract_planning_diagnostics()` runs post-solve, on a fixed
`m`; `_capex_at()`, `cost_decomposition`, and
`gas_to_power_twh_th_by_year` all call `pyo.value()` on already-solved
Vars/Expressions, never `pyo.Var(...)` or `pyo.Constraint(...)`), or (b)
comment/docstring prose (`00_build_emissions_cap.py`, Part 3) that
constrains no computation. No file examined or slated for Part 2/3
edits defines an objective term, a constraint, or a decision variable.
Editing `_capex_at()`, `cost_decomposition`'s components dict, or
`gas_to_power_twh_th_by_year`'s construction changes what gets
*reported* from a given solve, never what the solver was asked to
optimise — the anchors (regression, central) are solves of `m`, and `m`
is untouched by anything in this task's whitelist.

---

## PART 1 — STOP

Per TASK.md: Part 1 ends here. No edits have been made anywhere (Part 0
Rule 1/2/4 only — read files, read-only git, read-only syntax
inspection was not even needed since nothing was changed). **No builder
edit is required for any of the three Part-2 fixes**: 1A/1B established
that `_capex_at()` can read `m.convert_mw`/`m.convert_comm_vol` (both
already public model attributes) directly; 1D established
`m.convert_to_power` is already public and the exact aggregation
pattern already exists in this same file for the electrical-output
analogue; 1E established `m.capital_charge_total_npv` already includes
conversion, so only a new dict key is needed, not a new Expression. All
three fixes are confined to `src/optimize_experiments.py`, inside this
task's whitelist. The task's shape is unchanged.

Awaiting explicit release to begin Part 2.

---

## PARTS 2-4 — FIXES

Released via a new TASK.md ("conversion-blind diagnostics: the fixes
(Parts 2-4, RELEASED)"). **Before implementing, the released TASK.md's
own "WHAT PART 1 ESTABLISHED" section was checked against the code it
claims to describe, and found wrong on several load-bearing specifics.**
Recorded here because it changed what got implemented, and because two
of its wrong claims were about to go into a register draft as fact.

### Corrections to the released TASK.md, verified against the actual code before any edit

1. **Part 2A's exact snippet does not correspond to anything in this
   codebase.** `econ["convert_capex_usd_per_mw"]` — zero hits anywhere
   in `src/`/`scripts/`. `m.CONVERTIBLE` — does not exist. `m.convert_mw
   [t, p]` — wrong shape: `m.convert_mw` is `pyo.Var(_CONV_KEYS, ...)`
   (`optimize_model_sliced.py:631-635`), indexed by tier only
   (`"A"/"B"/"C"`), time-invariant (one-off, Decision 5) — there is no
   `[t, p]` index to take. The real cost source is
   `scenario["conversion_capex_per_kw"]` (per **kW**, from
   `conversion_scenarios()`), not an `econ` key.
2. **748,000 USD/MW and 1,941 MW do not appear anywhere.**
   `conversion_scenarios()` (`scenarios.py:1179-1245`) gives
   `capex_per_kw`: `conv_central=600.0`, `conv_low=850.0`,
   `conv_high=450.0`. Total nameplate across all three tiers is
   `_CONVERSION_NAMEPLATE_TOTAL_MW = 6482.0` (`scenarios.py:1096`,
   enforced by an assertion in `gas_conversion_tiers()`). Implemented
   Part 2A using the REAL structure instead (see below) — confirmed
   with the user before writing any code.
3. **The REL-suite exposure claim is unsupported.** Grepped
   `09_run_rel1_feasibility.py`, `10_run_rel2_marginal_cost.py`,
   `11_run_rel3_financing_frontier.py`, and both retired REL variants
   for `gas_to_power`, `gas_avail`, `deliverab`: zero references to
   `gas_to_power_twh_th_by_year` anywhere. "Gas deliverability" appears
   only in prose describing the `gas_deliverability_case` scenario arm.
   **Confirmed with the user: dropped from Part 4** rather than carried
   forward unverified.
4. **A real, different exposure exists in its place**, found while
   inventorying `gas_generation_twh_e_by_year`'s consumers for the Part
   2C rename (below): `scripts/18_run_pareto_frontier.py:494/498-506`
   uses it, fleet-only, as one of three terms summed into
   `disc_generation`, the denominator of a reported `system_lcoe`, on
   `build_model_sliced()` with an active conversion arm live in this
   exact script (`conv_central`/`conv_low`/…, per its own header). This
   IS a live "arithmetic now visibly wrong" case — just not the one, or
   the field, TASK.md named. Carried into Part 4 in place of item 3.
5. **The `~86 TWh_th` / "roughly eight times too tight" figure in Part
   3 does not reconcile** with the numbers this task's own background
   established: fleet thermal 10.5443, converted electrical 21.7598,
   retrofit efficiency 0.46 → converted thermal ≈ 21.7598/0.46 ≈ 47.30,
   total ≈ 57.85 TWh_th — an understatement of ≈5.5×, not ≈8×. Used the
   derivable figure in the Part 3 docstring instead; flagged there too,
   so a reader hits the correction in place rather than only here.
6. **The "source CSVs dated 2026-08-15" claim in Part 3 does not
   match any commit.** Checked directly: `git log` on
   `data/gas/processed/gas_available_power_annual_twh_th.csv` shows no
   commit near that date. The "baseline" scenario column's values were
   last substantively set by `986f467` (2026-08-11, "re-anchor
   physics … gas deliverability to 89.27 TWh_th") and extended to 2054
   by `74d34f2` (2026-08-18). One later commit touches the file,
   `5b811f2` (2026-08-28, "GMP 2026 policy-target family") — diff-
   verified this session to only ADD new scenario columns and drop
   out-of-horizon 2055 rows, leaving the "baseline" column's own values
   unchanged. Net effect on Part 3's actual claim (numerically fine,
   stale only in provenance) is UNCHANGED; only the specific date and
   the paper trail behind it were wrong. Corrected in the docstring
   with the real commits, so it is checkable rather than asserted.
7. **`gas_generation_twh_by_year` (Part 2C's second rename target,
   Part 4 item 4) is not a real key** — the actual field is
   `gas_generation_twh_e_by_year` (note the `_e_`). Its substance
   (fleet-only despite the name, same defect class as
   `gas_to_power_twh_th_by_year`) checked out — confirmed directly,
   `optimize_experiments.py:3180-3181` sums `m.gas_generation` with no
   `_converted_generation_total()` term, even though that exact helper
   already exists three lines away in the same file's
   `total_generation` computation (Plan 0.3, defect 1). Renamed the
   real key, `gas_generation_twh_e_by_year` →
   `fleet_gas_generation_twh_e_by_year`, not the misspelled one.

None of this changed the shape of the task — every fix below is still
confined to `src/optimize_experiments.py`, `scripts/
00_build_emissions_cap.py`, and the consumer files the (corrected)
inventory named, per the whitelist. It changed the CONTENT of Part 2A
and of Part 4's item list.

---

### PART 2A — `_capex_at()`: the conversion branch

**Old** (`src/optimize_experiments.py`, inside `_capex_at(t)`):
```python
        if hasattr(m, "gas_add") and "gas_capex_per_mw" in scenario:
            v += float(pyo.value(m.gas_add[t])) * float(scenario["gas_capex_per_mw"])
        return v
```

**New:**
```python
        if hasattr(m, "gas_add") and "gas_capex_per_mw" in scenario:
            v += float(pyo.value(m.gas_add[t])) * float(scenario["gas_capex_per_mw"])
        # Phase 2.8: OCGT-to-CCGT conversion capex (task_conversion_
        # diagnostics, Part 2A). Unlike the branches above, m.convert_mw is
        # a ONE-OFF decision (time-invariant, assigned to vintage 0 --
        # optimize_model_sliced.py Decision 5, objective discounts it with
        # _df_window_comm["gas"][0]), not a Var(T) flow, so it can only ever
        # contribute at t == 0. Reads m.convert_comm_vol[tier] -- the
        # builder's own raw-capex Expression, == conversion_capex_per_kw *
        # 1000 * convert_mw[tier] (optimize_model_sliced.py:1793-1799) --
        # rather than recomputing the product here, so this cannot drift
        # from the objective's own figure. hasattr guard: absent under
        # conv_none and under the annual builder (conversion_available=True
        # there raises), so this branch is a no-op and npv_capex_report /
        # npv_capex_full are byte-identical to pre-fix on any such run.
        if hasattr(m, "convert_mw") and t == 0:
            v += sum(float(pyo.value(m.convert_comm_vol[tier])) for tier in m.convert_mw)
        return v
```

**Same way the objective does it (1B):** reads `m.convert_comm_vol[tier]`
verbatim — the Expression `optimize_model_sliced.py`'s own
`convert_split_identity` constraint pins to
`conversion_capex_per_kw * 1000 * convert_mw[tier]` — rather than
re-deriving that product from `scenario["conversion_capex_per_kw"]` a
second time. No parallel derivation, no drift risk.

**`conv_none` byte-identical — structural guarantee, not an empirical
one** (no solver run permitted or performed): `m.convert_mw` is created
*only* inside `optimize_model_sliced.py`'s `if _conv:` block
(line 630); under `conv_none`, `_conv` is `False` and `m.convert_mw` is
never constructed, so `hasattr(m, "convert_mw")` is `False` and the new
branch is dead code on every such run — not "produces the same number,"
but "never executes," which is the stronger guarantee. Same holds for
the annual builder, which never has `convert_mw` at all
(`conversion_available=True` raises there instead of building it).

Syntax-checked: `python -m py_compile src/optimize_experiments.py` — OK.

---

### PART 2B — `cost_decomposition`: the conversion capital line

**Old** (`f1_tranche` branch, `src/optimize_experiments.py`):
```python
        cost_rebuild_capital = float(pyo.value(m.cost_rebuild_capital_charge_npv))
        cost_capital_total = float(pyo.value(m.capital_charge_total_npv))
        _capital_components = {
            "gas_capital_charge_npv":          round(cost_gas_capital, 0),
            "public_solar_capital_charge_npv": round(cost_pub_solar_capital, 0),
            "eaas_solar_capital_charge_npv":   round(cost_eaas_solar_capital, 0),
            "storage_capital_charge_npv":      round(cost_storage_capital, 0),
            "rebuild_capital_charge_npv":      round(cost_rebuild_capital, 0),
            "capital_charge_total_npv":        round(cost_capital_total, 0),
        }
```

**New:**
```python
        cost_rebuild_capital = float(pyo.value(m.cost_rebuild_capital_charge_npv))
        # task_conversion_diagnostics, Part 2B: m.cost_conversion_capital_
        # charge_npv (Phase 2.8, optimize_model_sliced.py:1800-1806) is
        # ALREADY folded into m.capital_charge_total_npv (read directly
        # below, unchanged) whenever an available conversion arm is active
        # -- hasattr-guarded like cost_salvage/cost_replacement_capex above,
        # since it is structurally absent under conv_none. Itemised here so
        # the five named channels plus this one sum to the sixth
        # (capital_charge_total_npv) entry on a conversion-active run; before
        # this line they summed to capital_charge_total_npv MINUS the
        # conversion charge, unlabelled.
        cost_conversion_capital = (
            float(pyo.value(m.cost_conversion_capital_charge_npv))
            if hasattr(m, "cost_conversion_capital_charge_npv") else 0.0
        )
        cost_capital_total = float(pyo.value(m.capital_charge_total_npv))
        _capital_components = {
            "gas_capital_charge_npv":          round(cost_gas_capital, 0),
            "public_solar_capital_charge_npv": round(cost_pub_solar_capital, 0),
            "eaas_solar_capital_charge_npv":   round(cost_eaas_solar_capital, 0),
            "storage_capital_charge_npv":      round(cost_storage_capital, 0),
            "rebuild_capital_charge_npv":      round(cost_rebuild_capital, 0),
            "conversion_capital_charge_npv":   round(cost_conversion_capital, 0),
            "capital_charge_total_npv":        round(cost_capital_total, 0),
        }
```

`cost_capital_total` itself is UNCHANGED (still read directly off
`m.capital_charge_total_npv`, already correct per Part 1E) — this only
adds the missing itemised line. `pre_f1_overnight` branch untouched
(conversion is structurally impossible there; guarded at both
`scenarios.py` and `optimize_model_sliced.py`, per Part 1E).

---

### PART 2C — rename both fields

```
gas_to_power_twh_th_by_year      ->  fleet_gas_to_power_twh_th_by_year
gas_generation_twh_e_by_year     ->  fleet_gas_generation_twh_e_by_year   (real name; see correction 7 above)
```

No alias, no compatibility shim. Definition site
(`src/optimize_experiments.py`, `extract_planning_diagnostics`):

**Old:**
```python
        "gas_to_power_twh_th_by_year":
            {int(y): float(_slice_total(m.gas_to_power, t)) for t, y in enumerate(years)},
        "gas_generation_twh_e_by_year":
            {int(y): float(_slice_total(m.gas_generation, t)) for t, y in enumerate(years)},
```

**New:** (comment reproduced in full since it carries the rename's
justification, not just its mechanics)
```python
        # RENAMED from gas_to_power_twh_th_by_year / gas_generation_twh_e_by_year
        # (task_conversion_diagnostics, Part 2C). Both read ONLY m.gas_to_power /
        # m.gas_generation -- the fleet's own thermal draw / electrical output --
        # never m.convert_to_power / m.convert_generation (Phase 2.8). Under an
        # active conversion arm the converted plant supplies most of gas-fired
        # electricity (88% at central, task background) and draws its OWN
        # thermal input, absent from both fields below. There is no single
        # "total gas" quantity in the model to add them into (thermal input via
        # the two paths is metered by two structurally separate variables at
        # two different efficiencies, m.gas_to_power at fleet eta and
        # m.convert_to_power at conversion_retrofit_efficiency), so the fields
        # are renamed to say what they measure rather than fixed to be
        # complete. Deliberately NOT aliased under the old name -- a consumer
        # missed by this rename must KeyError, not silently read a fleet-only
        # figure under a name that claims to be total.
        "fleet_gas_to_power_twh_th_by_year":
            {int(y): float(_slice_total(m.gas_to_power, t)) for t, y in enumerate(years)},
        "fleet_gas_generation_twh_e_by_year":
            {int(y): float(_slice_total(m.gas_generation, t)) for t, y in enumerate(years)},
```

#### Every consumer, updated, with a verdict

Dict-key read changed to the new name in all rows below; no arithmetic
touched anywhere except where noted.

| File : line(s) | Builder reachable | Verdict | Notes |
|---|---|---|---|
| `src/optimize_experiments.py:118-119` (`diagnostics_to_timeseries_df`) | `build_model()` only (both call sites of `write_run_outputs`, lines 216/3990, use `build_model()`) | **name-only** | Generic CSV-row builder; output column names (`gas_to_power_twh_th`, `gas_generation_twh_e`) left as-is — a separate artifact, not the renamed dict key |
| `src/optimize_experiments.py:3822` (`run_gas_regime_ndc_matrix`, GAS-3) | `build_model()` only (line 3693) | **name-only** | Annual-only sweep; conversion structurally impossible |
| `scripts/00_build_emissions_cap.py:131,139` | reads the bare `results/baseline/diagnostics.json` (frozen, annual, conv_none per Part 3) | **name-only** | This is the Part 3 target; correctness now depends on an input-provenance discipline documented in the script's own docstring (Part 3, below), not on the name alone |
| `scripts/01_run_baseline.py:460-461` | `build_model()` only | **name-only** | |
| `scripts/02_run_eaas.py:389-390` | `build_model()` only | **name-only** | |
| `scripts/02_run_ndc_caps.py:551-552` | `build_model_sliced()` (conversion-capable) | **name-only** | Pure CSV-column passthrough (`pd.DataFrame({...})`), no arithmetic combination with any other field |
| `scripts/03_run_ndc_eaas.py:614-615` | `build_model_sliced()` (conversion-capable) | **name-only** | Same passthrough pattern as `02_run_ndc_caps.py` |
| `scripts/18_run_pareto_frontier.py:494→504` (`gas_generation_twh_e_by_year` only — this script never reads the thermal field) | `build_model_sliced()`, conversion-capable, conversion-active runs are this script's own stated purpose | **arithmetic now visibly wrong** | `gas_gen_by_year` (fleet-only) is one of three terms summed into `disc_generation`, the denominator of a reported `system_lcoe`. On a conversion-active run this omits most of gas-fired electricity (88% at central per task background), understating the denominator and overstating `system_lcoe`. **Not fixed** — renamed only, per instruction; a comment records the verdict in place (see below) |

**Not edited, flagged instead (independent producer, not a consumer of
the renamed dict — same finding as Part 1's 1F, re-confirmed unchanged
this session):** `scripts/01b_run_baseline_sliced.py:545,555` builds
its own `diag_out` dict directly from `annual_series("gas_generation")`
/ `annual_series("gas_to_power")` — it never calls
`extract_planning_diagnostics()`, so it is not a consumer of the field
being renamed here, and the new TASK.md's whitelist extends only to
"any consumer file the inventory named." Left as-is. **Consequence
worth stating plainly**: this script now writes a `diagnostics__*.json`
whose `gas_to_power_twh_th_by_year` / `gas_generation_twh_e_by_year`
keys are STILL the pre-rename, unqualified names, for the exact same
fleet-only quantity `extract_planning_diagnostics()` now calls
`fleet_gas_to_power_twh_th_by_year` / `fleet_gas_generation_twh_e_
by_year`. Two producers of structurally identical diagnostics.json
files now disagree on what to call the same fleet-only quantity — an
inconsistency this task's whitelist cannot close (editing
`01b_run_baseline_sliced.py` is out of scope) and that should be its
own follow-up.

**18_run_pareto_frontier.py, key access updated (rename only) with the
verdict recorded in place, arithmetic untouched:**
```python
                solar_gen_by_year = diag["solar_generation_twh_e_by_year"]
                # RENAMED from gas_generation_twh_e_by_year (task_conversion_
                # diagnostics, Part 2C) -- FLEET-ONLY, excludes m.convert_
                # generation. task_conversion_diagnostics Part 2C verdict for
                # THIS call site: "arithmetic now visibly wrong" -- under an
                # active conversion arm converted plant supplies most of
                # gas-fired electricity (88% at central), so disc_generation
                # below understates total generation and overstates
                # system_lcoe on any conversion-active run. NOT fixed here
                # (a rename must not also change a computation); reported to
                # the register instead.
                gas_gen_by_year = diag["fleet_gas_generation_twh_e_by_year"]
```

Syntax-checked (`py_compile`): `00_build_emissions_cap.py`,
`01_run_baseline.py`, `02_run_eaas.py`, `02_run_ndc_caps.py`,
`03_run_ndc_eaas.py`, `18_run_pareto_frontier.py`,
`optimize_experiments.py` — all OK.

---

### PART 3 — `00_build_emissions_cap.py`: documentation added

A new docstring section, "REQUIRED INPUT: an ANNUAL, conv_none
BASELINE", was inserted into the module docstring (comment/docstring
only — `main()`'s computation is byte-for-byte unchanged; only the
`diag["..."]` key name changed, per Part 2C above). Full text is in the
file; summary of what it states and why each part is checkable:

- **Input requirement**: `BASELINE_DIAG_PATH` must be an annual
  (`build_model()`), `conv_none` diagnostics.json — states the
  mechanism (fleet-only field, harmless only where conversion is
  structurally impossible) rather than asserting it as a rule with no
  reason given.
- **Numeric illustration**: uses the task's own verified anchors
  (10.5443 fleet / 21.7598 converted-electrical / 0.46 retrofit
  efficiency → ≈57.85 TWh_th true total, ≈5.5× understated) — NOT the
  "~86 TWh_th / ~8×" figure the released TASK.md gave, which could not
  be reproduced from those same anchors (correction 5, above). The
  direction and rough order of magnitude survive; the specific number
  does not, and using an unverifiable number in a docstring meant to
  be "checkable rather than asserted" would defeat the point of the
  instruction.
- **Provenance, with real commits**: `emissions_cap.csv` (currently
  committed at `9700f3f`, 2026-09-14) was computed from the bare,
  untracked `results/baseline/diagnostics.json` (filesystem mtime
  2026-08-23 13:44 — independently re-confirmed this session with
  `ls -la`, not just repeated from a prior note). That file's gas
  values trace to `data/gas/processed/
  gas_available_power_annual_twh_th.csv`'s "baseline" column, last
  substantively set by commits `986f467` (2026-08-11) and `74d34f2`
  (2026-08-18); the one later commit touching that CSV, `5b811f2`
  (2026-08-28), was diff-verified this session to leave the "baseline"
  column's values untouched. **Not** "source CSVs dated 2026-08-15" —
  no commit near that date exists (correction 6, above) — corrected to
  the real trail.
- **Stale in provenance, not in value**: stated, with the reasoning
  attached rather than asserted alone.
- **The field's new name**, and a pointer to `task_conversion_
  diagnostics.md`/`task_epsilon_constraint.md` for why `01_run_
  baseline.py` cannot cleanly refresh the bare path.

**Operational consequence of the rename, worth flagging even though
Part 3 says "do not rebuild the caps"**: the bare
`results/baseline/diagnostics.json` currently on disk still has the
OLD key (`gas_to_power_twh_th_by_year`) baked into its JSON, since that
frozen file is not being regenerated by this task. `00_
build_emissions_cap.py`'s code now reads `fleet_gas_to_power_twh_th_
by_year`. **Running `00_` against the current bare file will now raise
`KeyError`, not silently succeed.** That is the intended, correct
consequence of the rename discipline (a stale reader must fail loudly)
— but it means `00_` will not run again until `01_run_baseline.py` is
rerun under `conv_none` and its output is manually placed at the bare
path, carrying the new key name. Not done here (running any script is
forbidden by this task's whitelist); flagged so it isn't a surprise the
next time someone runs `00_`.

---

### PART 4 — register draft (next free number: L29, re-confirmed unchanged this session — `docs/limitations_register.md`'s last entry is still L28)

**Draft, NOT applied** (`docs/limitations_register.md` is outside this
task's whitelist):

> ## L29 — The diagnostics layer was conversion-blind: two renamed
> fields, one omitted financing figure, one omitted cost-decomposition
> line, and one undocumented pipeline dependency, all from one
> unexamined 10.5443 TWh_th
>
> **Status:** LIVE (items 1, 3 below live but currently unread by any
> consumer besides `18_`; item 2 live wherever the frontier scripts
> run; item 4 dormant until a sliced, conversion-active baseline
> diagnostics.json is treated as the bare `results/baseline/
> diagnostics.json`).
>
> **What.** OCGT-to-CCGT conversion (Phase 2.8) is correctly wired into
> the objective and constraints of `build_model_sliced()`. Four
> separate places in the measurement layer were never extended to
> match, discovered by tracing a single unexpected number
> (`gas_to_power_twh_th_by_year["2025"] = 10.5443` on a `conv_central`
> sliced run, reconciling exactly to the FLEET-only backcast and
> excluding the 88%-of-gas-generation converted stream) rather than by
> any test failing:
>
> 1. **`gas_to_power_twh_th_by_year` and `gas_generation_twh_e_by_year`
>    were both fleet-only despite their names** — neither reads
>    `m.convert_to_power` / `m.convert_generation`. Renamed to
>    `fleet_gas_to_power_twh_th_by_year` / `fleet_gas_generation_twh_e_
>    by_year` (task_conversion_diagnostics, Part 2C) rather than fixed:
>    converted and unconverted gas draw fuel through two structurally
>    separate variables at two different efficiencies, so there is no
>    single "total gas" quantity in the model to complete the field
>    with by addition.
> 2. **`npv_capex_report` / `npv_capex_full` (the reported financing
>    headline, `npv_gross_capex_report_window_usd`) omitted conversion
>    retrofit capex from the total itself**, not merely a breakdown —
>    `_capex_at()` had branches for solar/storage/gas, none for
>    `convert_mw`. Fixed (Part 2A): reads `m.convert_comm_vol[tier]`,
>    the builder's own raw-capex Expression, at `t == 0` only (`convert_
>    mw` is a one-off, vintage-0 decision, not a per-year flow like the
>    other branches).
> 3. **`cost_decomposition["components"]` had no conversion capital
>    line**, even though `m.capital_charge_total_npv` — read directly
>    for the top-level total, unaffected — already includes it. The
>    five itemised lines did not sum to the sixth
>    (`capital_charge_total_npv`) entry sitting beside them in the same
>    dict, by exactly the unlabelled conversion charge. Fixed (Part
>    2B): added `conversion_capital_charge_npv`, hasattr-guarded.
> 4. **`00_build_emissions_cap.py` computes NDC caps from the bare
>    `results/baseline/diagnostics.json` with an undocumented
>    dependency**: correct only when that file is an ANNUAL, `conv_
>    none` baseline, because the (now honestly-named) fleet-only field
>    is a complete gas-thermal measure only where conversion is
>    structurally impossible. Documented, not fixed (Part 3) — the
>    currently-committed `emissions_cap.csv` remains numerically
>    correct (the input has not changed), stale only in provenance.
>
> **Precedent.** `cost_decomposition` omitting O&M (L28) was found the
> same way: a consuming script's own reconciliation guard, not a test.
> **Two instances now, both in the same function,
> `extract_planning_diagnostics`.** Worth asking whether this function
> needs a structural completeness check (e.g., assert `components`
> sums to `capital_charge_total_npv` at construction time) rather than
> relying on a third accidental discovery.
>
> **Exposure**, corrected from the released TASK.md's own draft (which
> named the REL suite; grepped and found no support — dropped): `18_
> run_pareto_frontier.py`'s `system_lcoe` denominator sums
> `fleet_gas_generation_twh_e_by_year` as though it were total gas
> generation. On any conversion-active run of that script (its own
> stated purpose includes `conv_central`/`conv_low`/…),
> `system_lcoe` is overstated — not fixed, per instruction (a rename
> must not also repair the arithmetic it exposes); recorded as its own
> open item.
>
> **Direction of bias.** All four items understate or otherwise
> obscure a conversion-active run's true financing/emissions/cost
> figures; none inflate.
>
> **What would resolve items 2/3/4 fully**, beyond what this task did:
> nothing further for 2/3 (fixed). For 4: an explicit, required
> `--baseline-diagnostics`-style CLI argument on `00_
> build_emissions_cap.py` (mirrors the pattern `task_epsilon_
> constraint` gave `18_run_pareto_frontier.py`) instead of the bare
> module-constant path — outside every whitelist so far, reported only.
> For the new `18_` exposure: correct `disc_generation` to add converted
> generation the same way `_converted_generation_total()` already does
> for `total_generation` elsewhere in `optimize_experiments.py` —
> outside this task's whitelist (touches a computed metric, not a
> rename).

---

## SUMMARY: NOTHING IN PARTS 2-4 CAN MOVE A SOLVED VALUE

Same argument as Part 1's summary, extended to the actual edits made:
`_capex_at()`, `cost_decomposition`'s components dict, and both renamed
fields are all read from an already-solved `m` via `pyo.value()` —
never a `pyo.Var(...)`, `pyo.Constraint(...)`, or objective term. The
`00_build_emissions_cap.py` edits are a docstring and one dict-key
string; `main()`'s arithmetic is untouched, confirmed by diff (only the
`diag["..."]` key name and the module docstring changed). Nothing here
was run: no script, no solver. All anchors (regression
21,811,042,472.821114 / 45,502.9754 MWh; central 27.954 bn / 65,613.0
MWh) are unaffected because nothing that produces them was touched.

---

## ENCOUNTERED BUT NOT ASKED ABOUT

- `scripts/01b_run_baseline_sliced.py` independently produces
  `gas_to_power_twh_th_by_year` / `gas_generation_twh_e_by_year` (its
  own fleet-only computation, not via `extract_planning_diagnostics()`)
  under the OLD, now-inconsistent names. Out of this task's whitelist.
- The stale comment at `optimize_experiments.py` (found in Part 1,
  still present — Part 2's whitelist covers `_capex_at()` and
  `cost_decomposition`'s construction, not this adjacent prose)
  asserting `extract_planning_diagnostics()` is "confirmed" never
  called on `build_model_sliced()` models — false since `f4dd48d`.
- `results/baseline/diagnostics.json` is untracked by git (`results/`
  is gitignored) and was previously tracked-then-deleted from git at
  commit `72c25dc` — its provenance now lives entirely in filesystem
  mtime, not commit history. Worth knowing before anyone reasons about
  "when" it was generated from `git log` alone, as this session
  initially (and incorrectly) tried to before checking `git ls-files`.
- `scripts/00_build_emissions_cap.py` has a duplicated `import sys` /
  `from pathlib import Path` (lines 62/65-66) and a dead commented-out
  `sys.path.append` block (lines 74-75) — pre-existing, untouched,
  noted only because it was read in full for Part 3.
- No conversion-active sliced run has ever produced a persisted
  `diagnostics.json` under `results/` (checked: `grep -rl "convert"` on
  every `diagnostics.json` under `results/` returns nothing) — every
  number in this report and its predecessor traces to the task's own
  stated anchors, not to a file this session could independently
  re-derive from disk.

**Nothing was committed or staged** (`git status --porcelain` shows
only untracked changes to already-permitted files plus this report;
verify before any future step relies on that).
