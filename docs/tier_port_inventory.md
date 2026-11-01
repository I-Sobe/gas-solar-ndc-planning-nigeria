# Tier port inventory

Working notes for bringing the NDC / Pareto / carbon-price runners up to
the model's current (time-sliced) state. Written incrementally per
TASK.md. Each section is appended as it completes.

---

## Step 0 — `extract_planning_diagnostics` for sliced models

### 0.1 Full list of sliced-indexed components the function reads

`build_model_sliced()` (src/optimize_model_sliced.py) indexes its
operational quantities over `(year_index, slice)` where `slice` is one of
the 6 `(season, period)` tuples in `m.S`. `build_model()` indexes the
same quantities over `(year_index)` only. `extract_planning_diagnostics`
(src/optimize_experiments.py) currently assumes the annual indexing
everywhere.

Every component the function reads that is `(year, slice)`-indexed under
the sliced builder (line numbers in src/optimize_experiments.py, current
`main`):

| # | component | kind (sliced builder) | read sites |
|---|-----------|-----------------------|------------|
| 1 | `m.unserved`          | `Var(T, m.S)`        | 1853 (`unserved_by_year`), 2017 (`unserved_dict`) |
| 2 | `m.gas_to_power`      | `Var(T, m.S)`        | 1877 (`gas_scarcity_value` loop), 2417 (`gas_to_power_twh_th_by_year`) |
| 3 | `m.gas_generation`   | `Expression(T, m.S)` | 1900 (`total_generation`), 2419 (`gas_generation_twh_e_by_year`) |
| 4 | `m.solar_generation` | `Expression(T, m.S)` | 1901 (`total_generation`), 1907 (`solar_total`), 2400 (`solar_generation_twh_e_by_year`) |
| 5 | `m.storage_discharge`| `Var(T, m.S)`        | 1902 (`total_generation`), 2038 (`storage_binding` power_bind), 2423 (`storage_discharge_twh_e_by_year`), 2444 (`storage_throughput_twh`), 2455 (`storage_utilisation_rate`) |
| 6 | `m.storage_charge`   | `Var(T, m.S)`        | 2044 (`storage_binding` energy_bind), 2370 (`storage_charge_twh_e_by_year`) |
| 7 | `m.selfgen`          | `Var(T, m.S)`        | 2404 (`selfgen_twh_by_year`) |

The documented `KeyError` on `m.unserved` is the FIRST of these to fire,
not the only one.

Sliced but NOT read by this function (so no aggregation needed):
`m.hydro_gen` (`Var(T, m.S)`).

Components that look risky but are safe (annual-indexed under BOTH
builders — verified in src/optimize_model_sliced.py):

- `m.gas_curtail` — `Var(T)` in the sliced builder (fuel budget is annual).
- `m.gas_capacity_mw`, `m.solar_capacity_mw`, `m.storage_capacity_mwh`,
  `m.storage_power_mw` — annual `Expression(T)` in both.
- `m.emissions_by_year`, `m.emissions_grid_only_by_year` — `Expression(T)`
  in both (each rule sums over slices internally).
- `m.gas_balance` — `Constraint(T)` in both, so
  `extract_gas_shadow_prices_usd_per_twh_th(m, years)` (reads
  `m.gas_balance[t]`) is unaffected.
- `m.emissions_constraint` — `Constraint(T)` or scalar in both, so
  `extract_carbon_shadow_prices_usd_per_tco2` is unaffected.
- All `m.cost_*_npv` / `m.capital_charge_total_npv` /
  `m.concessional_volume_by_vintage` / `*_conc_vol` / `*_comm_vol` /
  `m.salvage_npv` — scalar or `Expression(T)` in both; the
  `capital_treatment_case` branch and the `hasattr` guards already in the
  function carry over unchanged.
- `m.land_constraint`, `m.firm_capacity_constraint` — annual in both.
- `m.DF`, `m.remaining_npv_factor`, `m.solar_capex_param`,
  `m.eaas_subsidy`, `m.solar_public_add`, `m.solar_eaas_add`,
  `m.storage_add`, `m.gas_add` — annual in both.

### 0.2 The weighting

The 7 quantities above are all **additive energy in TWh**. The per-slice
electricity balance is an energy balance —
`demand[(t,s,p)] = annual_demand[t] * dw[(s,p)]` with `dw` normalised to
sum to 1 across the 6 slices (`_normalised_demand_weights`,
src/optimize_model_sliced.py:128). So the annual total of each quantity
is the **plain sum over the 6 slices** — no hours-weighting, no
capacity-factor rescale.

This is exactly the aggregation the builder already ships in
`optimize_model_sliced.annual_totals()` (line 1655), which does
`sum(pyo.value(var[t, (s, p)]) for (s, p) in SLICES)` for each of these
same names. `01b_run_baseline_sliced.py` uses that helper for the same
purpose. The step-0 fix reuses that rule (duplicated inline, not
imported, so the annual branch stays a single unbranched expression).

### 0.3 The fix

Added, immediately after `demand_dict` is built in
`extract_planning_diagnostics`:

```python
_sliced = m.unserved.dim() > 1          # (year, slice) vs (year)
if _sliced:
    _slice_index = list(m.S)
    def _slice_total(comp, t):
        return sum(pyo.value(comp[t, sp]) for sp in _slice_index)
else:
    def _slice_total(comp, t):
        return pyo.value(comp[t])
```

Branch key is the component's **own dimensionality** (`m.unserved.dim()`),
not a flag: `Var(T)` -> 1, `Var(T, m.S)` -> 3. Every one of the 7
components is consistently sliced-or-not, so one probe settles it.

Every read site listed in 0.1 changes from `pyo.value(m.X[t])` to
`_slice_total(m.X, t)`. In the annual case `_slice_total(m.X, t)` returns
`pyo.value(m.X[t])` unchanged — the annual code path is byte-identical
(the regression instrument is preserved).

### 0.4 Known semantic caveat (documented, not fixed in step 0)

`storage_binding` (line ~2036) classifies each year as
`power_limit` / `energy_limit` / `none` using two annual-identity tests:

- `power_bind`: `discharge == storage_power_mw * storage_deployable_hours_per_year / 1e6`
- `energy_bind`: `discharge == storage_round_trip_eff * charge`

Under the sliced builder there is **no** `storage_deployable_hours_per_year`
constraint — storage throughput falls out of the slice structure
(per-slice charge/discharge power limits + a per-season energy balance;
src/optimize_model_sliced.py:800-820). So after aggregation the
`power_bind` test compares the annual discharge total against a quantity
the model never constrained; the `power_limit` classification (and the
consumer field `storage_power_limit_years`) is **not meaningful** for
sliced runs. `energy_bind` is closer — summing the per-season balances
gives `discharge_total <= rt_eff * charge_total`, binding when both
seasons bind — but still not a faithful read of a single constraint.

Step 0 aggregates these reads so the function does not raise; it does
NOT re-derive a slice-native storage-binding diagnostic. That belongs in
the port task, with the consumers inventoried.

### 0.5 Verification result

Bounded check (TASK.md "Verification"): ONE non-solving
`build_model_sliced()` + ONE `extract_planning_diagnostics()` on it.
Scenario: `01b_run_baseline_sliced.py`'s `load_scenario(...)` call,
`peak_multiple_case="peak_suppressed"` (peak_nirp's period_demand_factor
is `None` / SOURCE: [PENDING] and cannot build), `capital_treatment_case
="f1_tranche"`, `concessional_envelope_case="env_dares"`.

Result — **the indexing fix holds**:

- `m.unserved.dim() == 3` -> the sliced branch is taken.
- The documented failure
  (`KeyError: "Index '0' is not valid for indexed component 'unserved'"`)
  is **gone**. `unserved_by_year` and both `m.unserved` reads now
  succeed (`unserved` is `.fix(0.0)` under the default
  no-genset-backstop path, so it carries a value).
- Execution proceeds past the `unserved` reads and into the
  `gas_scarcity_value` loop, where it reaches
  `_slice_total(m.gas_to_power, t)`. That call **successfully indexes**
  `gas_to_power[0, ('dry','night')]` — the (year, slice) access pattern
  is correct — and raises only
  `ValueError: No value for uninitialized VarData object
  gas_to_power[0,dry,night]`.

This is precisely the "fail on uninitialised values rather than
indexing" outcome TASK.md anticipated for an unsolved model. The
remaining 6 components are verified by reading the code path: every one
of the read sites in 0.1 now routes through `_slice_total`, which
indexes `comp[t, sp]` with `sp` a 2-tuple from `list(m.S)` — the
identical access pattern the sliced builder uses internally
(`m.gas_generation[t, (s, p)]`, src/optimize_model_sliced.py:692). No
further sliced-indexed `pyo.value(m.X[t])` reads remain in the function
(grep-verified over the full function body).

Annual path unchanged: `_sliced` is `False` for `build_model()`
(`m.unserved` is `Var(T)`, `dim() == 1`), so `_slice_total(comp, t)`
returns `pyo.value(comp[t])` and every edited site is a pure textual
substitution — byte-identical.

---

## Step 1 — retire / defer three scripts (moved, not deleted)

All three moves are `git mv` (rename tracked; working-tree content,
including the pre-existing uncommitted 2.11 `solar_min_build_default`
edits, carried along). A header paragraph was prepended to each.

### 1.1 `09b_run_rel1_retirement_sensitivity.py` -> `scripts/retired/`

RETIRED. Sweeps the 680 MW/yr baseline-gas retirement **proxy** at
0 / 340 / 680 / 1360 MW/yr. Superseded by the
`gas_retire_vintage_{30,40,50}` arms (limitations_register.md L8), which
retire the NERC-tracked fleet by commissioning vintage. The vintage arms
showed the flat proxy overstates system cost by ~1.28 bn (5.9%) and
storage by ~17.9%, so a sensitivity built on the proxy rate has no
reportable output — its axis is an artefact the model replaced. (Also:
calls annual `build_model`; missing pre-F1 keys — register L5, L10.)

### 1.2 `09b_run_rel1_mode_sensitivity.py` -> `scripts/retired/`

RETIRED — **D6 checked and it DID settle the question**, so retirement
(not "report instead") is correct:

- The sibling script's own header (now in
  `scripts/retired/09b_run_rel1_retirement_sensitivity.py`, lines 11-14)
  records: "Grid service mode switched 'total' -> 'annual' (2.6, D6).
  ... Results before and after this change are not comparable."
- `docs/phase_2_6_handoff.md` §3 "Declared conventions" confirms it as a
  standing decision: **"Mode `annual`, never `total` — `total` is slack
  under a rising deficit."** (Under a growing deficit `total` permits
  ~95% service in the 2030s / ~30% in the 2050s while still reporting
  horizon compliance.)

`build_model_sliced` still *accepts* `grid_service_mode="total"` (the
code path exists, src/optimize_model_sliced.py:994), but the project has
ruled it invalid for reporting. A script whose entire purpose is to
compare `annual` against the retired `total` mode cannot produce a
reportable result. (The `D6` in the "2.6, D6" tag is the 2.6-docket
item, not `merged_correction_plan_v2.md`'s unrelated storage-replacement
D6.)

### 1.3 `17_run_monte_carlo.py` -> `scripts/deferred/`

DEFERRED (not retired). Two blocking defects, both already declared in
its own docstring, restated verbatim in the new header:

1. **LCOE miscomputed.** Numerator includes the VoLL penalty; denominator
   omits hydro and double-counts storage discharge. Not on a common
   basis -> reported LCOE is meaningless.
2. **Demand-growth prior unsourced.** The MC demand-growth dimension
   draws from `scenarios.demand_growth_prior()`, centred on an unsourced
   value ([SOURCE NEEDED], plan 1.6). Docstring: "no MC result
   reportable until resolved."

### 1.4 Dangling references (NOT fixed — flagged for the port task)

- `scripts/run_verification_sweep.sh` listed the two retired scripts in
  its `RUNNERS` block and has a hard `MISSING` guard that aborts the
  whole sweep on an absent path. **Fixed here** (two lines removed) —
  directly caused by the move, and a broken guard is worse than the
  edit. The deferred/retired scripts were NOT added to any sweep list.
- `README.md` (lines ~332-333 tree, ~425-426 run block) still lists all
  three at their old paths with run instructions. **Not touched**
  (out of scope; doc-only).
- `scripts/23_run_selfgen_frontier.py:40` — docstring prose points at
  `09b_run_rel1_mode_sensitivity.py` as the place that "already sweeps
  both modes". **Not touched** (TASK.md forbids touching script 23);
  the pointer is now stale.

---

## Step 2 — port inventory for the five Tier 1 runners

Signature check first: `build_model_sliced(scenario, econ, weight_cost,
weight_emissions, emissions_cap, emissions_cap_by_year,
max_offgrid_fraction, grid_service_mode, ...)` — the kwargs all five
runners pass (`econ`, `emissions_cap`, `emissions_cap_by_year`,
`solar_capex_by_year`) exist on both builders with the same meaning, so
the `import` swap + call-name swap is mechanically clean. What is NOT
clean is (b) below for 18 and 19, and (c)/(d) for 05/18/19.

Shared helper note: `extract_planning_diagnostics` is now builder-aware
(step 0). `leakage_diagnostic(m)` (src/optimize_model.py:1707, called by
02/03/05/18) reads only scalar Expressions — `m.emissions`,
`m.emissions_selfgen`, `m.emissions_grid_only`, `m.emissions_capped` —
all present and annual in `build_model_sliced` (lines 1553-1609), so it
survives the swap untouched.

### 2.1 `02_run_ndc_caps.py`

**(a) `build_model(...)` call** — `scripts/02_run_ndc_caps.py:228-233`:
```
m = build_model(
    scenario=scenario,
    econ=econ,
    emissions_cap_by_year=caps,      # annual cap trajectory, from load_annual_caps()
    solar_capex_by_year=solar_capex_tv,
)
```

**(b) sliced-indexed model reads** — NONE. Direct `pyo.value` reads are
`m.system_cost_npv` (240), `m.solar_public_add[t]` / `m.solar_eaas_add[t]`
(255-256, 265-266), `m.storage_capacity_mwh[len-1]` (270), `m.emissions`
(275) — all scalar or annual `Expression(T)` in both builders. Every
per-slice quantity it needs comes through `diag[...]` (the timeseries
block, 409-420), so the step-0 fix covers it.

**(c) argparse scaffold** — YES, full. `parse_args()` (65-175) has
`--peak-case --storage-cc-case --gas-build-case --new-gas-deliverability
--gas-availability-case --gas-retirement-case --gas-capex-case
--ccgt-efficiency-case --capital-recovery-case --dscr-case --us-inflation
--capital-treatment-case --concessional-envelope-case`, all `required=
True`. `load_scenario(...)` (325-347) passes all of them plus
`reserve_margin="nirp_spinning"`. Hardcoded in that call:
`demand_level_case="served"`, `demand_case="organic_central"`,
`carbon_case="no_policy"`, `capital_case` (from the `NDC_CASES` dict, not
CLI). NOT passed at all: `solar_build_case` (-> load_scenario default
`nirp_trajectory`), `land_case` (-> default `moderate`). This runner is
NOT one of the ten dead pre-F1-key runners.

**(d) `gas_deliverability_case`** — HARDCODED `"baseline"`
(`02_run_ndc_caps.py:328`). The three GMP arms
(`gas_deliv_flat` / `gas_deliv_gmp_partial` / `gas_deliv_gmp_full`) are
unreachable. Needs a `--gas-deliverability-case` CLI flag in the port.

**(e) would-not-survive-a-straight-swap** — nothing structural. The
`NDC_CASES` dict comment (300-315) still describes an
`"expansion"`/`"moderate"` capital split that the dict body no longer
uses (all four cases are `tight`/`moderate`); stale prose, not a swap
blocker. Storage will become non-trivial under slicing, so the
`"Storage discharge (TWh by year)"` print (281) and the
`storage_capacity_mwh` decision var will start showing non-zero values —
expected, not a regression.

### 2.2 `03_run_ndc_eaas.py`

Structurally identical to 02.

**(a)** `scripts/03_run_ndc_eaas.py:231-236` — same four kwargs as 02
(`emissions_cap_by_year=caps`, `solar_capex_by_year=solar_capex_tv`).

**(b) sliced-indexed model reads** — NONE. Same direct-read set as 02
(`m.system_cost_npv`, `m.solar_public_add`/`m.solar_eaas_add`,
`m.storage_capacity_mwh`, `m.emissions`); all per-slice data via
`diag[...]` including `diag["solar_eaas_add_mw_by_year"]` (458, annual).

**(c) argparse scaffold** — YES, full, byte-identical arg set to 02.
`load_scenario(...)` (345-367) passes all CLI keys +
`reserve_margin="nirp_spinning"`. After the call it mutates the scenario
dict directly: `scenario["financing_regime"] = "eaas"` (371),
`scenario["solar_service_tariff_usd_per_twh"] = 95_000_000.0` (372),
`scenario["public_solar_budget_npv"] = None` (383). Not passed:
`solar_build_case`, `land_case` (defaults, as 02).

**(d) `gas_deliverability_case`** — HARDCODED `"baseline"`
(`03_run_ndc_eaas.py:348`). Same fix needed as 02.

**(e) would-not-survive-a-straight-swap** — nothing structural. Note the
EaaS bankability path is exercised here (`financing_regime="eaas"` +
tariff): `build_model_sliced` builds `m.eaas_bankability` /
`m.eaas_subsidy_limit` on the ANNUAL `solar_energy_per_mw` (line 841),
same as the annual builder — no slice interaction, clean.

### 2.3 `05_run_fin3_capital_sweep.py`

**(a) `build_model(...)` call** — `scripts/05_run_fin3_capital_sweep.py:214-219`:
```
m = build_model(
    scenario=scenario,
    econ=econ,
    emissions_cap_by_year=caps,      # from load_annual_caps(NDC_CAP_SCENARIO, years)
    solar_capex_by_year=solar_capex_tv,
)
```

**(b) sliced-indexed model reads** — NONE. Direct reads:
`m.system_cost_npv` (251), `m.emissions` (253), `m.solar_public_add[t]`
(257), `m.solar_eaas_add[t]` (261) — all annual/scalar. Per-slice data
via `diag[...]` (`unserved_twh_by_year`, `carbon_shadow_*`,
`horizon_reliability`, `public_budget_*`).

**(c) argparse scaffold** — **NONE.** `main()` takes no args. Everything
is module constants (`NDC_CAP_SCENARIO`, `CAPITAL_SWEEP_ORDER`,
`COLLECTION_RATE_SCENARIOS`, `DOMESTIC_SHARE`) or hardcoded in the
`load_scenario(...)` call (162-170), which passes ONLY:
`demand_level_case`, `demand_case`, `gas_deliverability_case`,
`capital_case`, `carbon_case`, `start_year`, `end_year`. **Missing every
post-2.6 required key** — `peak_multiple_case`, `reserve_margin`,
`storage_cc_case`, `gas_build_case`, `new_gas_deliverability`,
`gas_availability_case`, `gas_retirement_case`, `gas_capex_case`,
`ccgt_efficiency_case`, `capital_recovery_case`, `dscr_case`,
`us_inflation`, `capital_treatment_case`, `concessional_envelope_case`.
It is one of the **ten dead runners (register L10)** — it raises inside
`load_scenario()` today, before `build_model` is ever reached. Post-call
mutations: `scenario["public_solar_budget_npv"]` (collection-rate
adjustment, 190-195), `scenario["disco_collection_rate"]` (209).

**(d) `gas_deliverability_case`** — HARDCODED `"baseline"`
(`05_run_fin3_capital_sweep.py:165`).

**(e) would-not-survive-a-straight-swap** — the builder swap is the
*smallest* part of this port. The runner needs a full argparse scaffold
(or an equivalent config block) supplying the ~14 missing required keys
before it builds at all. Its FIN-3 question (public capital budget vs
access) is answered against `m.public_budget_constraint` /
`diag["public_budget_shadow_usd_per_usd"]`; `build_model_sliced` builds
`m.public_budget_constraint` (line 1614) on the same annual discounted
capex + subsidy sum, so the diagnostic carries over once the scenario
builds. Also: TASK.md explicitly forbids wiring these keys in this task.

### 2.4 `18_run_pareto_frontier.py`

**(a) `build_model(...)` call** — `scripts/18_run_pareto_frontier.py:156-161`:
```
m = build_model(
    scenario=scenario,
    econ=econ,
    emissions_cap_by_year=cap_by_year,   # uniform annual cap = baseline_annual * frac
    solar_capex_by_year=solar_capex_tv,
)
```

**(b) sliced-indexed model reads — YES, this one breaks on a straight swap.**
Direct `pyo.value` reads of `(year, slice)`-indexed components:
- `18:193-195` — `m.solar_generation[t]`, `m.gas_generation[t]`,
  `m.storage_discharge[t]` in the `disc_generation` sum (system LCOE
  denominator).
- `18:222-225` — `m.gas_generation[5]`, `m.solar_generation[5]`,
  `m.storage_discharge[5]` (`gas_share_2030`).
- `18:228-231` — `m.solar_generation[len-1]`, `m.gas_generation[len-1]`,
  `m.storage_discharge[len-1]` (`solar_share_2045`).
Under `build_model_sliced` each of these is `comp[t, (s,p)]` and
`pyo.value(comp[t])` raises the same `KeyError` the step-0 fix addressed
in the diagnostics function. The port must either aggregate over
`m.S` locally (same rule as `_slice_total`) or, better, read the
equivalents already in `diag`:
`diag["solar_generation_twh_e_by_year"]`,
`diag["gas_generation_twh_e_by_year"]`,
`diag["storage_discharge_twh_e_by_year"]` (all now slice-aware) —
plus `diag["discount_factor_by_year"]` for the LCOE discounting.
Annual/scalar reads that are fine: `m.system_cost_npv` (177),
`m.solar_public_add`/`m.solar_eaas_add` (184-185),
`m.storage_capacity_mwh[len-1]` (188), `m.DF[t]` (192).

**(c) argparse scaffold** — **NONE.** `main()` takes no args.
`FINANCING_ARMS` is a module constant carrying `financing_regime`,
`capital_case`, and a hardcoded `required_margin` (1.10 / 1.05).
`load_scenario(...)` (131-141) passes only: `demand_level_case`,
`demand_case`, `gas_deliverability_case`, `capital_case`,
`solar_build_case="deployment_unconstrained"`, `land_case="loose"`,
`carbon_case`, `start_year`, `end_year`. **Missing every post-2.6
required key** — same list as 05. One of the ten dead runners (L10).
Post-call mutations: `scenario["financing_regime"]`,
`scenario["required_margin"]` (145-146).

**(d) `gas_deliverability_case`** — HARDCODED `"baseline"`
(`18_run_pareto_frontier.py:134`).

**(e) would-not-survive-a-straight-swap** —
(i) the ~14 missing required keys (as 05);
(ii) the three sliced-indexed read blocks in (b);
(iii) `required_margin` is hardcoded 1.10/1.05 in `FINANCING_ARMS` — F1
Decision 2 retired that as a double-count of `capital_recovery_case`'s
`r_conc`/`r_comm`; the port should take `required_margin` from
`dscr_case` like 02/03 do, not from a literal;
(iv) it reads `results/baseline/diagnostics.json` for
`emissions_tco2_by_year` to set the cap grid — that baseline artefact is
from the annual model and would need regenerating from a sliced baseline
for internal consistency.

### 2.5 `19_run_carbon_price_sweep.py`

**(a) `build_model(...)` call** — `scripts/19_run_carbon_price_sweep.py:130-135`:
```
m = build_model(
    scenario=scenario,
    econ=econ,
    emissions_cap=1e18,               # no binding cap; carbon PRICE drives the sweep
    solar_capex_by_year=solar_capex_tv,
)
```

**(b) sliced-indexed model reads — YES, breaks on a straight swap.**
- `19:162-164` — `m.solar_generation[t_idx]`, `m.gas_generation[t_idx]`,
  `m.storage_discharge[t_idx]` inside `gen_shares(t_idx)`, called for
  `t_idx` = 5, 10, `len(years)-1` (170-172).
- `19:177-179` — `m.solar_generation[t]`, `m.gas_generation[t]`,
  `m.storage_discharge[t]` in the `disc_gen` sum (system LCOE).
Same remedy as 18 — aggregate over `m.S` or read the slice-aware
`diag[...]` series. Annual/scalar reads that are fine: `m.system_cost_npv`
(149), `m.solar_public_add`/`m.solar_eaas_add` (154-155),
`m.storage_capacity_mwh[len-1]` (158), `m.DF[t]` (176).

**(c) argparse scaffold** — **NONE.** `main()` takes no args.
`FINANCING_ARMS` module constant (`financing_regime`, `capital_case`,
`required_margin` 1.10). `CARBON_PRICES` module constant.
`load_scenario(...)` (96-106) passes only: `demand_level_case`,
`demand_case`, `gas_deliverability_case`, `capital_case`,
`solar_build_case="deployment_unconstrained"`, `land_case="loose"`,
`carbon_case`, `start_year`, `end_year`. **Missing every post-2.6
required key** — one of the ten dead runners (L10). Post-call mutations:
`scenario["financing_regime"]`, `scenario["required_margin"]` (110-111),
`scenario["carbon_active"]`, `scenario["carbon_price"]` (114-119).

**(d) `gas_deliverability_case`** — HARDCODED `"baseline"`
(`19_run_carbon_price_sweep.py:99`).

**(e) would-not-survive-a-straight-swap** —
(i) the ~14 missing required keys;
(ii) the two sliced-indexed read blocks in (b);
(iii) `required_margin` hardcoded 1.10 (same F1 Decision 2 issue as 18);
(iv) `carbon_active` / `carbon_price` are set on the scenario dict
post-`load_scenario` — `build_model_sliced` reads
`scenario.get("carbon_active", False)` and `scenario["carbon_price"]`
(lines 1560-1561), same as the annual builder, so the carbon-price
mechanism itself ports cleanly.

### 2.6 Summary table

| runner | argparse | missing req. keys (dead runner) | gas_deliv hardcoded | sliced-indexed reads break swap |
|--------|----------|----------------------------------|---------------------|--------------------------------|
| 02_run_ndc_caps        | full | no  | yes (`baseline`) | no  |
| 03_run_ndc_eaas        | full | no  | yes (`baseline`) | no  |
| 05_run_fin3_capital_sweep | none | YES (~14) | yes (`baseline`) | no  |
| 18_run_pareto_frontier | none | YES (~14) | yes (`baseline`) | YES (3 blocks) |
| 19_run_carbon_price_sweep | none | YES (~14) | yes (`baseline`) | YES (2 blocks) |

Cleanest ports: 02 and 03 (import + call-name swap, add
`--gas-deliverability-case`). Hardest: 05/18/19 need a scaffold for the
missing required keys first (explicitly out of scope for this task),
and 18/19 additionally need their direct sliced-component reads
rerouted through `m.S` aggregation or the slice-aware `diag` series.

---

# Port B: 02 and 03

TASK.md "Tier work B". Port `02_run_ndc_caps.py` and `03_run_ndc_eaas.py`
to `build_model_sliced`, and add the stranded `--gas-deliverability-case`
flag to the six runners that hardcode it.

## Step 0 — `gas_deliverability_case` in the four already-sliced-capable runners

Port A's inventory (§2) covered 02/03/05/18/19. It did NOT cover the
runners that already call `build_model_sliced` (or can). All four below
have a FULL required-flag scaffold (peak / storage-cc / gas-build /
new-gas-deliverability / gas-availability / gas-retirement / gas-capex /
ccgt-efficiency / capital-recovery / dscr / us-inflation /
capital-treatment / concessional-envelope) — but every one hardcodes
`gas_deliverability_case="baseline"` in its `load_scenario(...)` call:

| runner | `gas_deliverability_case` | file:line | where |
|--------|---------------------------|-----------|-------|
| `01_run_baseline.py`          | HARDCODED `"baseline"` | `scripts/01_run_baseline.py:194`          | `load_scenario(...)` in `main()` |
| `01b_run_baseline_sliced.py`  | HARDCODED `"baseline"` | `scripts/01b_run_baseline_sliced.py:187`  | `load_scenario(...)` in `main()` |
| `22_salvage_sensitivity.py`   | HARDCODED `"baseline"` | `scripts/22_salvage_sensitivity.py:272`   | `load_scenario(...)` inside `run_arm(...)` |
| `23_run_selfgen_frontier.py`  | HARDCODED `"baseline"` | `scripts/23_run_selfgen_frontier.py:285`  | `load_scenario(...)` inside `make_scenario(...)` |

NONE is a CLI flag today. **All four need `--gas-deliverability-case`
added** (TASK.md step 1 names 01/01b explicitly and 22/23 conditionally —
"if step 0 shows them hardcoded"; it does).

Not to be confused with `--new-gas-deliverability` (the
`constrained`/`exempt` treatment of NEW gas build's fuel access, 2.9),
which IS already a flag in all four. Different key, different mechanism.

`gas_deliverability_scenarios()` keys (src/scenarios.py:755):
`downside, baseline, upside, shock_recovery, flat_downside, flat_upside,
flat_shock_recovery` (historical-drift family) +
`gas_deliv_flat, gas_deliv_gmp_partial, gas_deliv_gmp_full` (GMP 2026
policy-target family).

### storage_binding consumption in 02/03 (TASK.md step 3)

Neither 02 nor 03 CONSUMES `storage_power_limit_years` /
`storage_binding_by_year` in any assertion or computed output.
`02_run_ndc_caps.py:282` has a commented-out debug print referencing
`diag["storage_binding_by_year"]`; 03 has no reference at all. Both DO
serialise the whole `diag` dict to `diagnostics.json`
(`json_safe(out["diagnostics"])`), so the (Port-A §0.4: not-meaningful-
under-slicing) `storage_power_limit_years` field will appear in their
JSON output post-port — inert, read by nothing. Not fixed here, per
TASK.md step 3.

## Step 1 — `--gas-deliverability-case` added to six runners

Added a REQUIRED `--gas-deliverability-case` flag (no default) to
`01_run_baseline.py`, `01b_run_baseline_sliced.py`, `02_run_ndc_caps.py`,
`03_run_ndc_eaas.py`, `22_salvage_sensitivity.py`,
`23_run_selfgen_frontier.py`. Each edit, following the existing
required-flag pattern in `01b`:

- `from src.scenarios import ... gas_deliverability_scenarios ...` added
  to the import.
- Module constant
  `GAS_DELIVERABILITY_CASE_CHOICES = tuple(gas_deliverability_scenarios().keys())`
  — derived live from `src/scenarios.py` (not a frozen literal) so the
  historical-drift and GMP families cannot silently diverge from the
  registry, with a comment recording why (GMP arms were stranded).
- `parser.add_argument("--gas-deliverability-case", required=True,
  choices=GAS_DELIVERABILITY_CASE_CHOICES, help=...)` — help text names
  both epistemic families in one sentence (historical-drift
  `downside/baseline/upside/shock_recovery` + `flat_*` level-equivalents;
  GMP 2026 policy targets `gas_deliv_flat / _gmp_partial / _gmp_full`,
  `I = 0 / 0.50 / 1.0`).
- `load_scenario(gas_deliverability_case="baseline")` ->
  `gas_deliverability_case=args.gas_deliverability_case` (for 22 and 23,
  threaded as a new trailing parameter through
  `run_arm(...)` / `make_scenario(...)` + `solve_point(...)` and their
  call sites — those `load_scenario` calls live inside helper functions).
- Appended `__{args.gas_deliverability_case}` to each script's output
  tag / result-dir name (02/03 use `_` not `__` as their separator —
  matched locally), and added
  `"gas_deliverability_case": args.gas_deliverability_case` to each
  summary/output dict.

`22` and `23` already select the builder via a `sliced` bool and run
both the annual and sliced builders per invocation — the flag change is
orthogonal to that and touches only scenario construction.

## Step 2 — 02 and 03 ported to `build_model_sliced`

Exactly as the inventory anticipated — a two-line swap each:

- Import: `from src.optimize_model import build_model, solve_model,
  leakage_diagnostic` ->
  `from src.optimize_model import solve_model, leakage_diagnostic` +
  `from src.optimize_model_sliced import build_model_sliced`.
- Call: `m = build_model(...)` -> `m = build_model_sliced(...)` in
  `run_case()` (02:~229, 03:~232). Same four kwargs (`scenario`, `econ`,
  `emissions_cap_by_year`, `solar_capex_by_year`), all present on
  `build_model_sliced` with identical meaning.

`leakage_diagnostic(m)` left untouched (scalar Expressions, present in
both builders). The post-`load_scenario` scenario-dict mutations in 03
(`financing_regime`, `solar_service_tariff_usd_per_twh`,
`public_solar_budget_npv`) left exactly as they were.

### Nothing broke that the inventory did not anticipate

Re-checked every model read in `run_case()` and downstream against the
sliced builder:

- `m.system_cost_npv`, `m.emissions`, `m.solar_public_add[t]`,
  `m.solar_eaas_add[t]`, `m.storage_capacity_mwh[len-1]` — all scalar /
  annual `Expression(T)` in `build_model_sliced`. No change.
- `extract_planning_diagnostics(m, scenario, econ)` — builder-aware
  since Port A step 0; the timeseries CSV block (which reads ~12
  `diag[...]` series) is unaffected.
- `diag["gas_shadow_price_usd_per_twh_th_by_year"]` reads
  `m.gas_balance[t]` — `Constraint(T)` in the sliced builder (verified:
  indexed over 30 years). Fine.
- `emissions_cap_by_year=caps` — sliced builder builds
  `m.emissions_constraint = Constraint(T, rule: emissions_capped_by_year[t]
  <= caps[t])` with a length check. Fine.
- EaaS path (03): `financing_regime="eaas"` + tariff -> sliced builder
  builds `m.eaas_bankability` / `m.eaas_subsidy_limit` on the ANNUAL
  `solar_energy_per_mw` (src/optimize_model_sliced.py:841), no slice
  interaction.

New *behaviour* (expected, not a break), worth flagging for whoever
reads the results:

1. **Storage stops being identically zero.** The NDC pathways now run in
   a model where storage can shift energy across intra-annual slices, so
   `decision_variables.storage_capacity_mwh` and
   `diag["storage_discharge_twh_e_by_year"]` will show non-trivial values
   for the first time in an NDC context. That is the point of the port
   (register L5).
2. **`unserved` is fixed to 0 in both builders** here (both call
   `load_econ(..., genset_case="genset_central")`, so
   `GENSET_COST_PER_TWH_E` is in `econ` and the sliced builder's
   `allow_unserved` auto-resolves to `False`, `.fix(0.0)` per slice —
   identical to the annual builder's D11 behaviour). The shortfall
   channel is `selfgen`. `cumulative_unserved_twh` stays ~0; read
   `selfgen_twh_by_year`.
3. **`storage_power_limit_years`** appears in `diagnostics.json` (whole
   `diag` is serialised) and is not meaningful under slicing (Port A
   §0.4). Neither 02 nor 03 reads it. Left as-is per TASK.md step 3.
4. **Result-dir names changed** (extra `_{gas_deliverability_case}`
   segment), so a sliced run will not overwrite an existing annual
   run's `results/ndc/` or `results/ndc_eaas/` output — the two coexist.

### Verification (bounded — the one permitted execution)

ONE non-solving `build_model_sliced()` via 02's own `load_scenario(...)`
arguments (other CLI-supplied arms set to valid values:
`peak_suppressed`, `storage_cc_central`, `gas_none`, `f1_tranche`,
`env_dares`, ...), with **`gas_deliverability_case="gas_deliv_gmp_partial"`**
and a placeholder (unsolved) `emissions_cap_by_year`. Result:

```
gas_deliverability_case reached scenario: gas_deliv_gmp_partial
build_model_sliced OK
  m.unserved.dim() = 3
  gas_balance indexed over 30 years
  slices: [('dry','night'),('dry','day'),('dry','peak'),('wet','night'),('wet','day'),('wet','peak')]
```

The new flag's value flows through `load_scenario()` to
`scenario["gas_scenario"] == "gas_deliv_gmp_partial"`, and the GMP arm's
supply trajectory is read without error during the build
(`gas_available_power()` inside `build_model_sliced` resolved it). The
GMP 2026 family is now reachable from a runner. `build_model_sliced`
constructs cleanly on 02's scenario (`m.unserved.dim() == 3` confirms
the sliced builder; `m.gas_balance` is annual, 30 entries). No solve,
no other execution.
