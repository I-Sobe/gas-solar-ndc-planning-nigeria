# Phase 1.5 — Hydro trajectory arms

Correction plan 1.5, `[OPEN]` sub-item: build the two exogenous hydro
trajectory arms (committed-only, NDC-aligned) so Zungeru's ramp and
hydro under-delivery become tested conditions rather than a flat
assumption. Hydro stays OUT of the optimiser.

Written incrementally per TASK.md.

---

## Step 0 — inventory (locate and quote, no change)

Line numbers are current `HEAD`. "annual" = `src/optimize_model.py`
(`build_model`); "sliced" = `src/optimize_model_sliced.py`
(`build_model_sliced`).

### (a) Every consumer of `hydro_baseline_twh`, `hydro_growth`, `hydro_capacity_mw`

**`load_scenario()` (`src/scenarios.py`):**

| symbol | site | what it does |
|--------|------|--------------|
| `_hydro_baseline_twh = 11.47` | `:2410` | local constant. SOURCE: NERC 2024, 30.92% of 37,093.70 GWh (comment `:2587`). |
| `_hydro_capacity_mw = 2678` | `:2411` | local constant. SOURCE: NERC sum of installed hydro plant ratings, Kainji+Jebba+Shiroro+Zungeru+Dadin-Kowa (comment `:2591-2594`). Already includes Zungeru's 700 MW. |
| `_hydro_baseline_twh` (again) | `:2503` | `_hydro_dry_avg_mw = _hydro_baseline_twh * _hydro_share["dry"] * 1e6 / _dry_hours_per_year` -- the dry-season average hydro output in MW. |
| `_hydro_capacity_mw` (again) | `:2519` | `capacity_credit["hydro"] = _hydro_dry_avg_mw / _hydro_capacity_mw` -- a dry-season-output-per-installed-MW FRACTION (~0.44). SOURCE: [PENDING]. |
| dict entries | `:2590` | `"hydro_baseline_twh": _hydro_baseline_twh`, `"hydro_growth": 0.0` |
| dict entry | `:2595` | `"hydro_capacity_mw": _hydro_capacity_mw` |

**annual builder (`src/optimize_model.py`):**

| symbol | site | what it does |
|--------|------|--------------|
| `hydro_capacity_mw = scenario.get("hydro_capacity_mw", None)` | `:309-314`, raise if None | firm-capacity credit only |
| `hydro_baseline_twh = scenario.get(...)` | `:586-594`, raise if None | feeds the `hydro` dict |
| `hydro_growth = scenario.get("hydro_growth", 0.0)` | `:595` | feeds the `hydro` dict |
| `hydro = {t: hydro_baseline_twh * ((1.0 + hydro_growth) ** t) for t in T}` | `:596-599` | the annual hydro energy series (TWh_e) |
| `hydro[t]` | `:921` | added into `m.balance` (electricity balance) as an exogenous CONSTANT term |
| `capacity_credit["hydro"] * hydro_capacity_mw` | `:1042` | term in `m.firm_capacity_constraint` -- `hydro_capacity_mw` is a **flat scalar**, not year-indexed |

**sliced builder (`src/optimize_model_sliced.py`):**

| symbol | site | what it does |
|--------|------|--------------|
| `hydro_baseline_twh = scenario.get(...)` | `:712-717`, raise if None | feeds `hydro_annual` |
| `hydro_growth = scenario.get("hydro_growth", 0.0)` | `:718` | feeds `hydro_annual` |
| `hydro_annual = {t: hydro_baseline_twh * ((1.0 + hydro_growth) ** t) for t in T}` | `:719` | annual hydro energy series (TWh_e) |
| `hydro_capacity_mw = scenario.get("hydro_capacity_mw", None)` | `:721-728`, raise if None | per-slice power limit + firm capacity |
| `hydro_annual[t]` | `:900` | `m.hydro_season_budget`: `Σ_{slices in season} hydro_gen[t,(s,p)] == hydro_annual[t] * hydro_season_share[season]` |
| `hydro_annual[t]` | `:919` | `m.hydro_min_flow`: `hydro_gen[t,(s,p)] >= hydro_min_frac * hydro_annual[t] * hydro_season_share[s] * (H[(s,p)] / season_hours[s])` |
| `hydro_capacity_mw` | `:907` | `m.hydro_power_limit`: `m.hydro_gen[t,(s,p)] <= hydro_capacity_mw * H[(s,p)] / 1e6` -- **flat scalar** |
| `capacity_credit["hydro"] * hydro_capacity_mw` | `:1237`, `:1252` | term in `m.firm_capacity_constraint` (both the `_conv` and `else` branches) -- **flat scalar** |
| `m.hydro_gen` | `:595` (`Var(T, m.S)`) | dispatched within the seasonal energy budget; also read in `annual_totals` (`:1903`) and `slice_detail` (`:1950`) for reporting only |

No runner, `extract_planning_diagnostics`, or other module reads
`hydro_baseline_twh` / `hydro_growth` / `hydro_capacity_mw` directly
(grep-verified). Every scenario reaches the builders through
`load_scenario()` -- no hand-built scenario dicts anywhere
(`tests/test_run.py` calls `load_scenario` and is already stale on the
required-key front, unrelated).

### (b) How the hydro series enters the balance and firm capacity

- **Electricity balance.** Annual: `hydro[t]` is added to `m.balance`'s
  LHS as an exogenous constant (`:921`) -- hydro is must-run, not
  dispatched. Sliced: `m.hydro_gen[t,(s,p)]` (a Var) is added to
  `m.balance` per slice (`:969`, `:982` -- both the `_conv` and `else`
  branches), and is pinned by the seasonal budget + min-flow + power
  limit so it is exogenous in TOTAL but freely allocated across slices
  within a season.
- **Firm-capacity constraint.** Both builders:
  `+ capacity_credit["hydro"] * hydro_capacity_mw` where
  `hydro_capacity_mw` is a **flat scalar** (2,678) and
  `capacity_credit["hydro"]` is the ~0.44 dry-season-output fraction
  computed once in `load_scenario()`. Hydro's credited firm MW is
  therefore `0.44 * 2,678 ~ 1,178 MW`, constant across the horizon.

### (c) `hydro_season_factor` / `_hydro_season_energy_share` and growth interaction

`DEFAULT_HYDRO_SEASON_FACTOR = {"dry": 0.90, "wet": 1.10}`
(`optimize_model_sliced.py:118`), overridable via
`scenario["hydro_season_factor"]`.

`_hydro_season_energy_share(scenario)` (`:178-186`) returns a FRACTION
per season:
```
raw[s]   = season_hours[s] * hydro_season_factor[s]
share[s] = raw[s] / sum(raw.values())          # sums to 1 over {dry, wet}
```
It depends ONLY on `hydro_season_factor` and the fixed slice-hour
calendar. It does NOT reference `hydro_annual`, `hydro_growth`,
`hydro_baseline_twh` or any energy level.

It feeds:
1. `m.hydro_season_budget` (`:900`): `... == hydro_annual[t] * share[season]`
2. `m.hydro_min_flow` (`:919`): `... >= ... * hydro_annual[t] * share[s] * ...`
3. `capacity_credit["hydro"]` derivation in `load_scenario()` (`:2498-2503`)
   -- calls `_hydro_season_energy_share` with the hard-coded
   `{"dry": 0.90, "wet": 1.10}` (NOT `scenario["hydro_season_factor"]`).

**Does a growth trajectory interact with the seasonal split? NO.**

In (1) and (2) the growth trajectory changes only the SCALAR
`hydro_annual[t]`; `share[season]` multiplies it as a fixed fraction, so
a larger annual total splits between dry and wet in the SAME proportion.
The constraint RULES are untouched -- only the value `hydro_annual[t]`
references. The seasonal machinery
(`hydro_season_factor`, `_hydro_season_energy_share`,
`hydro_season_budget`, `hydro_min_flow`) does not need to change.

The ONE real coupling is `m.hydro_power_limit` (`:907`):
`hydro_gen[t,(s,p)] <= hydro_capacity_mw * H[(s,p)] / 1e6`. If
`hydro_annual[t]` grows (e.g. to ~44 TWh under an NDC arm) while
`hydro_capacity_mw` stays at 2,678, the dry-season budget
`hydro_annual[t] * share[dry]` can exceed what the dry slices can
physically pass, making the model INFEASIBLE. This is exactly why
TASK.md requires "`hydro_capacity_mw` must move with the energy series".
Moving it is a scalar -> per-year change, NOT a change to the seasonal
machinery.

**Conclusion: a growth trajectory CAN be applied without touching the
seasonal machinery. NOT stopping.** Proceeding to steps 1-2.

`capacity_credit["hydro"]` (item 3) is derived once, from
`_hydro_baseline_twh` (11.47) / `_hydro_capacity_mw` (2,678). Because the
trajectory design (step 1) moves capacity proportionally with energy at
a fixed capacity factor, the dry-season-output-per-installed-MW fraction
is INVARIANT to the energy level -- so the single stored
`capacity_credit["hydro"]` scalar stays exactly correct for every arm
and year with no recomputation, and TASK.md's "do not change
`capacity_credit["hydro"]`" is satisfied by keeping the 11.47 / 2,678
anchor pair pinned.

_(steps 1-2 + verification appended below as completed)_

---

## Step 1 — `hydro_case` required key

`src/scenarios.py`:

- `_HYDRO_ANCHOR_TWH = 11.47`, `_HYDRO_ANCHOR_MW = 2678.0`,
  `_HYDRO_CF = 11.47e6 / (2678 * 8760) = 0.4889` -- the pinned NERC 2024
  pair and its implied capacity factor. NOT changed by any arm.
- `hydro_case_scenarios()` -> `("hydro_flat", "hydro_committed", "hydro_ndc")`.
- `hydro_trajectory(hydro_case, years)` -> `(energy_series, capacity_series)`,
  both lists aligned to the model year index.
- `_HYDRO_REQUIRED` sentinel + `hydro_case=_HYDRO_REQUIRED` in
  `load_scenario()` (required, no default -- `TypeError` if omitted,
  `ValueError` on unknown name), mirroring `conversion_case`.
- Scenario dict gains `hydro_case`, `hydro_energy_twh_by_year`,
  `hydro_capacity_mw_by_year`. The existing scalar `hydro_baseline_twh` /
  `hydro_capacity_mw` entries become the **year-2025 value of the
  selected arm** (`series[0]`); `hydro_growth` kept at `0.0`. Label
  `labels["hydro"]`.

### The three arms (resolved series)

| arm | energy TWh_e | capacity MW | shape |
|-----|--------------|-------------|-------|
| `hydro_flat` | 11.47 every year | 2,678 every year | flat. REGRESSION -- byte-identical to pre-1.5. |
| `hydro_committed` | 12.8 every year | 2,988.5 every year | 12.8 (SOURCED 2025) held flat (DECLARED). |
| `hydro_ndc` | 12.8 (2025) -> 44.544 (2030) -> flat | 2,988.5 -> 10,400 -> flat | linear 2025-2030 (DECLARED), 10,400 MW 2030 (SOURCED), flat after (DECLARED). |

`hydro_ndc`'s 2030 energy anchor is the energy equivalent of the SOURCED
10,400 MW at `_HYDRO_CF`: `11.47 * 10,400 / 2,678 = 44.544 TWh_e`.

### Deriving capacity from energy -- basis

`hydro_capacity_mw[t] = 2,678 * hydro_energy_twh[t] / 11.47`, i.e. a
**fixed capacity factor** `_HYDRO_CF = 0.4889` (the pinned NERC 2024
anchor CF) in every arm and every year.

Why this basis: `capacity_credit["hydro"]` is a
dry-season-output-per-installed-MW fraction, derived ONCE in
`load_scenario()` from the pinned (11.47, 2,678) pair. Because capacity
moves proportionally with energy, that fraction is **exactly invariant**
to the energy level -- verified: under `hydro_committed`
(12.8 TWh / 2,988.5 MW) the actual dry-season-avg-MW / installed-MW is
`(12.8 x share_dry x 1e6 / 4344) / 2,988.5 = 0.43968`, identical to the
stored `capacity_credit["hydro"] = 0.43968`. So TASK.md's "capacity must
move with the energy series so `capacity_credit["hydro"]` stays
coherent" is satisfied, and "do not change `capacity_credit["hydro"]`"
is satisfied by keeping the anchor pair pinned.

**Declared imperfection:** this conflates capacity growth with
UTILISATION growth. Under `hydro_committed` / `hydro_ndc` the extra
energy is partly Zungeru (already inside the 2,678 MW) ramping up on
installed plant, not purely new capacity -- so the derived capacity
slightly overstates true installed MW when the growth is
utilisation-driven. The fixed-CF derivation is the price of a coherent
`capacity_credit["hydro"]` without re-deriving it per arm. Stated in the
`hydro_trajectory()` docstring.

### Builder changes (both builders)

The seasonal machinery (`hydro_season_factor`,
`_hydro_season_energy_share`, `hydro_season_budget`, `hydro_min_flow`
rules) is UNTOUCHED. Only scalar -> per-year reads:

- annual `build_model` (`src/optimize_model.py`): `hydro_energy_twh_by_year`
  replaces the `hydro_baseline_twh * (1+hydro_growth)**t` formula
  (`hydro` dict, `:~596`); `hydro_capacity_mw_by_year[t]` replaces the
  scalar in `m.firm_capacity_constraint` (`:~1056`).
- sliced `build_model_sliced` (`src/optimize_model_sliced.py`):
  `hydro_energy_twh_by_year` -> `hydro_annual` (`:~737`);
  `hydro_capacity_mw_by_year[t]` replaces the scalar in
  `m.hydro_power_limit` (`:~907`) and `m.firm_capacity_constraint`
  (both `_conv` and `else` branches). Length guards added.

Both builders raise if `hydro_energy_twh_by_year` /
`hydro_capacity_mw_by_year` are absent (hand-built-scenario protection),
replacing the old `hydro_baseline_twh` / `hydro_capacity_mw` None
guards.

Under `hydro_flat` every by-year entry equals the pre-1.5 scalar
(11.47 / 2,678), so the expression values are unchanged -- byte-identical
regression.

### Runners

`--hydro-case` (required, `choices=tuple(hydro_case_scenarios())`) wired
into `01, 01b, 02, 03, 22, 23` -- appended to each output tag and summary
dict; threaded through `run_arm` (22) and `make_scenario` / `solve_point`
(23). NOT wired to the L10 dead runners.

## Step 2 — seasonal caution carried forward

Comment added at `DEFAULT_HYDRO_SEASON_FACTOR`
(`src/optimize_model_sliced.py:~118`), recording:
- the seasonal-scissor claim is NOT supported (shares rise 29.9 -> 38.3%
  but levels near-flat Q1-Q3; NERC attributes the Q4 jump to
  utilisation, not hydrology -- `merged_correction_plan_v2.md:320-324`);
- the Zungeru commissioning ramp confounds any seasonal signal;
- under `hydro_committed` / `hydro_ndc` the growth trajectory and the
  seasonal factor are two mechanisms fed by ONE underlying commissioning
  programme we cannot yet decompose -- a KNOWN OVERLAP, declared;
- `hydro_season_factor` and `capacity_credit["hydro"]` values are NOT
  changed (both remain `SOURCE: [PENDING]`).

## Verification (the one permitted execution)

ONE non-solving `build_model_sliced()` under `hydro_committed` +
`conv_central` + `f1_tranche` + `cr_central` + `dscr_regression` +
`env_dares` (other CLI arms at valid values):

- `load_scenario(hydro_case="hydro_committed")` returns without raising.
- Resolved `scenario["hydro_energy_twh_by_year"]` = 12.8 at 2025 / 2030 /
  2035 / 2045; `scenario["hydro_capacity_mw_by_year"]` = 2,988.5 at each
  (= `2,678 x 12.8 / 11.47`).
- `hydro_flat` resolver: 11.47 / 2,678 flat -- unchanged from pre-1.5.
- `hydro_ndc` resolver: 12.8 (2025) -> 44.544 TWh_e / 10,400 MW (2030) ->
  flat; 2028 interpolates to 31.846 / 7,435. CF held at 0.4889 in every
  arm/year.
- `capacity_credit["hydro"] = 0.43968` -- computed from the pinned
  anchor, unchanged.
- `build_model_sliced()` builds.
- Energy ENTERS: `m.hydro_season_budget[2030,'dry']` RHS = 5.708 =
  `12.8 x share_dry`.
- Capacity ENTERS: `m.hydro_power_limit[2030,('dry','peak')]` RHS =
  2.164 = `2,988.5 x 724 / 1e6`; `m.firm_capacity_constraint[2030]`
  constant carries `0.43968 x 2,988.5 = 1,313.99`.

No solve, no runner, no other execution.

---

# Validity guards

Hydro is EXOGENOUS, so a resource ceiling as an LP constraint would do
nothing (trivially satisfied or infeasible, never changes the optimum).
The correct treatment is a VALIDITY GUARD in `hydro_trajectory()`,
raising / warning at `load_scenario()` time. No constraint added to
either builder.

## Step 1 — resource guard (RAISE)

`_HYDRO_EXPLOITABLE_MW = 17_620.0` (module constant, provenance comment
verbatim from TASK.md: ECN's 14,120 MW large + ~3,500 MW small; NOT
NIRP's ~21 GW modelling ceiling; competing figures declared).

`hydro_trajectory()` computes `capacity_series`, then raises `ValueError`
if `max(capacity_series) > _HYDRO_EXPLOITABLE_MW`, naming the year and
the peak MW.

- `hydro_flat` (peak 2,678 MW) -- passes.
- `hydro_committed` (peak 2,989 MW) -- passes.
- `hydro_ndc` (peak 10,400 MW = **59.0%** of 17,620) -- passes.

## Step 2 — build-rate diagnostic (WARN, never raise)

`_HYDRO_BUILD_RATE_WARN_MW_PER_YR = 700.0` -- one Zungeru-equivalent per
year (Zungeru 700 MW, ~7 yr FID-to-commissioning; large hydro 8-12 yr).
DECLARED, not sourced.

`hydro_trajectory()` computes the peak year-over-year delta in
`capacity_series` and issues `warnings.warn(...)` if it exceeds the
threshold, reporting the rate in MW/yr and Zungeru-equivalents.

- `hydro_flat` / `hydro_committed`: peak build 0 MW/yr -- no warning.
- `hydro_ndc`: peak build **1,482 MW/yr = 2.1 Zungeru-equivalents/yr**
  (2025 committed 2,989 MW -> 2030 10,400 MW, linear) -- **WARNS**.
  (TASK.md's illustrative "1,544 / 2.2" measures from the 2,678 MW
  anchor rather than the 2,989 MW committed 2025 level; the guard uses
  the real series delta.)

WARN not raise: the arm is a declared counterfactual and must stay
runnable, and the model has NO lead-time constraint for ANY technology,
so the same limitation applies to solar and gas (finding F-23).

## Step 3 — fixed-CF limitation recorded

Added to `hydro_trajectory()`'s docstring: uprating or better-utilising
EXISTING plant raises installed MW without raising available flow, so a
hydro system's real capacity factor FALLS as capacity grows. Holding CF
fixed at the 2,678 MW anchor's 0.4889 therefore OVERSTATES energy per MW
at higher capacity -- it biases the high-hydro arms (hydro_ndc
especially) toward MORE energy than the resource supports. Direction of
bias stated, NOT corrected: a declining-CF derivation would make
capacity no longer proportional to energy and break
`capacity_credit["hydro"]`'s invariance to the energy level (Phase 1.5
step 1).

## Verification (no build call -- direct `hydro_trajectory()` calls)

| arm | peak cap MW | ceiling raise? | peak build MW/yr | build WARN? |
|-----|-------------|----------------|------------------|-------------|
| `hydro_flat` | 2,678 | no | 0 | no |
| `hydro_committed` | 2,989 | no | 0 | no |
| `hydro_ndc` | 10,400 (59.0%) | no | 1,482 (2.1x) | **yes** |
| synthetic (ceiling -> 5,000) on `hydro_ndc` | 10,400 | **RAISES** | — | — |

Guard raises on the synthetic over-ceiling case and NOT on the three
real arms; the build-rate warning fires for `hydro_ndc` only. Nothing
else run.
