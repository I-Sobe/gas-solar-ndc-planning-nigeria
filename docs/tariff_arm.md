# tariff_arm.md — EaaS service tariff as a swept arm

Runner scope for later steps: `03_run_ndc_eaas.py` and
`26_run_f3_identification.py` only (per TASK.md).

---

## STEP 0 — Inventory (locate and quote, NO change)

### Quantity

`scenario["solar_service_tariff_usd_per_twh"]` — EaaS service tariff,
**USD per TWh**. Read into a local `tariff` in both builders; gates the
`m.eaas_bankability` and `m.eaas_subsidy_limit` constraints. `None` ⇒
neither constraint is built (and EaaS regime raises).

Unit note: TASK.md states the new arm values in **USD/MWh**
(75 / 95 / 115). 1 TWh = 1e6 MWh, so those convert to
75e6 / 95e6 / 115e6 USD/TWh — the units the scenario key and every
existing site already use. `tariff_central` (95 USD/MWh = 95e6 USD/TWh)
is therefore **numerically identical** to `tariff_legacy` (95_000_000.0);
the two differ only in provenance (legacy = unsourced literal; central =
to be sourced by Zeus).

---

### (a) Sites that SET it

#### a.1 — Canonical, inside `load_scenario()` (the existing named-arm path)

| File | Line | Code | Value |
|---|---|---|---|
| `src/scenarios.py` | 2864–2865 | `"solar_service_tariff_usd_per_twh": solar_tariff_scenarios()[solar_tariff_case],` | `solar_tariff_case` default `"baseline"` ⇒ **65_000_000** |

- `solar_tariff_case` is a parameter of `load_scenario()`
  (`src/scenarios.py:1974`, **default `"baseline"` — NOT required**).
- Validated at `src/scenarios.py:2185–2186`
  (`if solar_tariff_case not in solar_tariff_scenarios(): raise`).
- `solar_tariff_scenarios()` — `src/scenarios.py:803–819`:
  `low: 45_000_000 / baseline: 65_000_000 / high: 85_000_000` USD/TWh.
  Docstring carries a `STALE-DOC` note: the old threshold rationale is no
  longer valid, "treat the grid below as provisional."
- **This path is dead in every production runner** — see (a.2). The only
  non-default caller anywhere is
  `src/optimize_experiments.py:3135`
  (`run_financing_vs_resource_test`, `solar_tariff_case="low"`), an
  unused experiment helper.

#### a.2 — Hardcoded literal overrides (post-`load_scenario()` mutation of the scenario dict)

| File | Line | Code | In scope? |
|---|---|---|---|
| `scripts/02_run_eaas.py` | 215 | `scenario["solar_service_tariff_usd_per_twh"] = 95_000_000.0` | no |
| `scripts/03_run_ndc_eaas.py` | 429 | `scenario["solar_service_tariff_usd_per_twh"] = 95_000_000.0` | **YES — TASK target** |
| `scripts/07_run_gas2_eaas_gas_relief.py` | 134 | `scenario["solar_service_tariff_usd_per_twh"] = 95_000_000` | no |
| `scripts/15_run_pol1_ndc_comparison.py` | 194 | `scenario["solar_service_tariff_usd_per_twh"] = CANONICAL_TARIFF` | no |
| `scripts/26_run_f3_identification.py` | 241 | `scenario["solar_service_tariff_usd_per_twh"] = EAAS_TARIFF_USD_PER_TWH` | **YES — TASK target** |

Module-level constants feeding a.2 / a.3:

| File | Line | Constant | Value |
|---|---|---|---|
| `scripts/15_run_pol1_ndc_comparison.py` | 100 | `CANONICAL_TARIFF` | `95_000_000  # USD/TWh` |
| `scripts/26_run_f3_identification.py` | 120 | `EAAS_TARIFF_USD_PER_TWH` | `95_000_000.0  # BARE LITERAL, no arm` |
| `scripts/11_run_rel3_financing_frontier.py` | 119 | `CANONICAL_TARIFF` | `95_000_000  # USD/TWh — above self-financing threshold` |
| `scripts/12_run_dem1_demand_sensitivity.py` | 125 | `CANONICAL_TARIFF` | `95_000_000  # USD/TWh` |
| `scripts/13_run_dem2_growth_gas_interaction.py` | 110 | `CANONICAL_TARIFF` | `95_000_000  # USD/TWh` |

#### a.3 — Indirect setters: runner passes a tariff into an experiment-function config dict, which sets the scenario key

| Runner (config site) | Experiment fn (set site in `src/optimize_experiments.py`) |
|---|---|
| `scripts/11_run_rel3_financing_frontier.py:134,142,150,158,166` — `"eaas_tariff": CANONICAL_TARIFF` in `policy_configs` | `optimize_experiments.py:811` — `scenario_eaas["solar_service_tariff_usd_per_twh"] = eaas_tariff` (`run_eaas_feasibility_frontier`, key doc'd at :734) |
| `scripts/12_run_dem1_demand_sensitivity.py:138,146,154,162` — `"eaas_tariff": CANONICAL_TARIFF` in `ndc_configs` | `optimize_experiments.py:1174` (`"solar_service_tariff_usd_per_twh": eaas_tariff` in `fin_params`) and applied at `:1191–1193`; key doc'd at :1130 |
| `scripts/13_run_dem2_growth_gas_interaction.py:126` — `"tariff": CANONICAL_TARIFF` in `financing_configs` | `optimize_experiments.py:1408` and `:1604` — `scenario["solar_service_tariff_usd_per_twh"] = fin_cfg["tariff"]` (`run_gas_regime_ndc_matrix` + storage variant) |

Pure-library set sites (no runner in scope; listed for completeness):

| File | Line | Context |
|---|---|---|
| `src/optimize_experiments.py` | 894 | `sc["solar_service_tariff_usd_per_twh"] = tariff` — `run_fin2_blended_bankability_sweep`, per grid point |
| `src/optimize_experiments.py` | 3068 | `scenario_copy["solar_service_tariff_usd_per_twh"] = tariff` — `run_tariff_public_capital_frontier`, per `tariffs` arg |

**Verdict on the "five where three expected" warning:** confirmed here
too. TASK names 2 setter runners; there are **5 direct literal sites**
(02, 03, 07, 15, 26) + **3 indirect runner paths** (11, 12, 13) + the
canonical `load_scenario()` path. Every one of the eight runner sites
uses **95M USD/TWh** (as literal or as a `CANONICAL_TARIFF` /
`EAAS_TARIFF_USD_PER_TWH` constant), none of which is `solar_tariff_case`
and none of which is sourced. Migrating only 03 and 26 (TASK scope)
leaves 02/07/11/12/13/15 on the bare literal — flagged, not actioned.

---

### (b) Sites in the builders that READ it

#### `src/optimize_model.py` (annual — `build_model`)

| Line | Code | Role |
|---|---|---|
| 390 | `tariff = scenario.get("solar_service_tariff_usd_per_twh", None)` | local bind |
| 690–692 | `if financing_regime == "eaas": if tariff is None: raise ValueError("Tariff must be defined under EaaS regime.")` | regime guard |
| 755–765 | `if tariff is not None:` … `bankable_revenue = (tariff * solar_energy_per_mw * m.remaining_npv_factor[t] * m.solar_eaas_add[t]) / required_margin` … `m.eaas_bankability = pyo.Constraint(T, rule=eaas_bankability_rule)` | **the constraint F-27 is gated by** |
| 795–808 | `if tariff is not None:` … `max_bankable_at_t = tariff * solar_energy_per_mw * m.remaining_npv_factor[t] / required_margin` … `m.eaas_subsidy_limit = pyo.Constraint(...)` | subsidy ceiling |

Note: the annual builder **raises** for the two TASK runners regardless —
`03` and `26` both call `build_model_sliced()` and the annual path is not
exercised by them. (`03_run_ndc_eaas.py:214` comment: "build_model_sliced()
only -- the annual build_model() raises".)

#### `src/optimize_model_sliced.py` (sliced — `build_model_sliced`) — the path the TASK runners use

| Line | Code | Role |
|---|---|---|
| 462 | `tariff = scenario.get("solar_service_tariff_usd_per_twh", None)` | local bind |
| 1024–1025 | `if financing_regime == "eaas" and tariff is None: raise ValueError("Tariff must be defined under EaaS regime.")` | regime guard |
| 1068–1078 | `if tariff is not None:` … `bankable_revenue = (tariff * solar_energy_per_mw_annual * m.remaining_npv_factor[t] * m.solar_eaas_add[t]) / required_margin` … `m.eaas_bankability = pyo.Constraint(T, rule=eaas_bankability_rule)` | **the constraint F-27 is gated by** |
| 1085–1097 | `if tariff is not None:` … `max_bankable_at_t = (tariff * solar_energy_per_mw_annual * m.remaining_npv_factor[t] / required_margin)` … `m.eaas_subsidy_limit = pyo.Constraint(...)` | subsidy ceiling |

Bankability constraint form (both builders), per year `t`:

```
effective_private_cost[t]  =  capex[t]*solar_eaas_add[t] - eaas_subsidy[t]
bankable_revenue[t]        =  tariff * solar_energy_per_mw * remaining_npv_factor[t]
                              * solar_eaas_add[t] / required_margin
constraint:  effective_private_cost[t] <= bankable_revenue[t]
```

Per-MW slack (TASK step 2 quantity) =
`bankable_revenue[t]/solar_eaas_add[t] - effective_private_cost[t]/solar_eaas_add[t]`
= `tariff*solar_energy_per_mw*remaining_npv_factor[t]/required_margin
   - (capex[t] - eaas_subsidy[t]/solar_eaas_add[t])`.
When `solar_eaas_add[t] == 0` the constraint is `0 <= 0` (no
information); the per-MW form needs a guard for that.

---

### (c) Diagnostic / post-solve readers

All in `src/optimize_experiments.py`, function `extract_planning_diagnostics`:

| Line | Code | Output key |
|---|---|---|
| 2399 | `"solar_tariff_usd_per_twh": scenario["solar_service_tariff_usd_per_twh"],` | echoes the input tariff |
| 2447–2449 | `"bankable_capex_per_mw": scenario["solar_service_tariff_usd_per_twh"] * npv_energy_priv / scenario["required_margin"],` | max bankable CAPEX/MW at the private/blended rate |

Related post-solve reader that does NOT touch the scenario key but is the
sibling quantity (line 2408–2413):
`"bankability_threshold": required_margin * (solar_capex_param[0] / npv_energy_priv)`.

**Gap for TASK step 2:** `extract_planning_diagnostics` does **not**
currently read `m.eaas_bankability[t]` or emit any per-year slack. There
is no existing bankability-slack reader anywhere in `src/` or `scripts/`
(`grep eaas_bankability` → only the two definition sites). Both TASK
runners call `extract_planning_diagnostics` (`03:295`, `26` via the F3
solve loop) and also build their own `out`/`summary` dicts
(`03_run_ndc_eaas.py:316–337` `run_case` → `out`;
`26_run_f3_identification.py:357+` `summary`) — step 2's per-year slack
can be added to those runner-side dicts without changing the shared
diagnostic function, or to the diagnostic function once for both.

---

### Does a runner already sweep the tariff?

**Yes — `04_run_fin2_tariff_sweep.py` sweeps it, but it is a research
sweep, not a named-arm selector, and it uses the annual builder.**

Shape:

- **Driver:** `scripts/04_run_fin2_tariff_sweep.py` → calls
  `run_fin2_blended_bankability_sweep(base_scenario, econ, tariff_grid,
  blend_scenarios, ndc_cap_scenario, concessional_envelope_npv, cap_path)`
  (`src/optimize_experiments.py`, def ~820, body 846–1000).
- **Grid:** `src/scenarios.py:826–837` `TARIFF_SWEEP_GRID` — a
  module-level list, USD/TWh:
  `[30, 35, 40, 44, 46, 50, 60, 75, 95, 110]` × 1e6.
  (Carries a `STALE-DOC` note: re-centre once rates are frozen.)
  Note it **already contains 75M and 95M** — two of the three TASK
  values; TASK's high is 115M, the grid's top is 110M.
- **Loop nest:** `NDC_CASES` (5: ndc2/ndc3 × cond/uncond + baseline) ×
  `blended_finance_scenarios()` × `TARIFF_SWEEP_GRID`. One solved LP per
  cell.
- **Builder:** `build_model` (**annual**), not `build_model_sliced`. So
  it does not exercise the same constraint code the TASK runners
  (`03`, `26`) hit.
- **Per-cell output:** analytical `T* = capex_2025 * required_margin /
  npv_energy`, `financing_gap_per_mw`, `is_self_financing`,
  `eaas_subsidy_npv_usd`, `solar_eaas_total_mw`, `zero_subsidy_reason`,
  solve status. Written to `results/fin2/tariff_sweep_{results,summary}__{tag}.csv/json`.
- **Purpose:** FIN-2's question — minimum self-financing tariff T* and
  the subsidy required below it, and how T* moves with capital
  structure. The module docstring explicitly says the canonical runs use
  95M "well above T*" and this sweep exists to walk from below to above
  T*.

Two other tariff sweeps exist in the library:

- `run_tariff_public_capital_frontier` (`optimize_experiments.py:3044`) —
  tariff × public-capital-envelope frontier for a reliability target;
  annual builder; imported by `scripts/02_run_eaas.py:14`.
- `run_tariff_bankability_sweep` (`optimize_experiments.py:3027`) —
  **DEPRECATED**, raises `NotImplementedError` (superseded by the
  blended sweep).

**Reuse assessment for the `tariff_case` arm:**

The `tariff_case` required key is a **named-scenario selector**, exactly
analogous to `dscr_case` (`src/scenarios.py:1738`, incl. the
`dscr_regression` retained-for-attribution arm) and `solar_tariff_case`
(`src/scenarios.py:803`). It is **not** analogous to
`04_run_fin2_tariff_sweep.py` / `TARIFF_SWEEP_GRID`, which is a
grid-walk research artefact on the annual model.

The mechanism to reuse is the **existing `solar_tariff_case` /
`solar_tariff_scenarios()` path** in `load_scenario()` — it already
resolves a named key to `scenario["solar_service_tariff_usd_per_twh"]`
at `src/scenarios.py:2864`. Open design question for step 1 (needs a
decision, not invention): `solar_tariff_case` currently (i) has a
default, (ii) is unused by production runners, (iii) has provisional
`STALE-DOC` values 45/65/85M. TASK wants a **required, no-default** key
named `tariff_case` with values 75/95/115M + `tariff_legacy`. Cleanest:
repurpose/replace `solar_tariff_case` → `tariff_case` (required
sentinel, mirroring `dscr_case`'s `_DSCR_REQUIRED`), retire
`solar_tariff_scenarios()`'s stale grid, and point the two runners at
the arm value instead of their literals. This does not touch
`TARIFF_SWEEP_GRID` or `04_run_fin2_tariff_sweep.py`.

---

## STOP — awaiting Zeus's sourced values + provenance for
`tariff_low` / `tariff_central` / `tariff_high` before step 1.

---

## STEPS 1–3 — implementation (revised scope, TASK.md rev.)

### Step 1 — `solar_tariff_case` → required `tariff_case`

`src/scenarios.py`:

- **`solar_tariff_scenarios()` deleted**, replaced by **`tariff_scenarios()`**
  (same location). Returns five arms, USD/TWh:

  | key | value | status |
  |---|---|---|
  | `tariff_low` | 75_000_000.0 | SOURCED — NBET 2016 FG counter-position (75 USD/MWh) |
  | `tariff_central` | 95_000_000.0 | CONSTRUCTION — midpoint of the two NBET positions |
  | `tariff_high` | 115_000_000.0 | SOURCED — rate NBET agreed with 14 developers, Jul 2016 |
  | `tariff_legacy_95m` | 95_000_000.0 | UNSOURCED — the runner override literal; attribution only |
  | `tariff_legacy_65m` | 65_000_000.0 | UNSOURCED — `load_scenario()`'s old silent default; attribution only |

  Verbatim provenance block from TASK.md is the function docstring.
- **`_TARIFF_REQUIRED = object()`** sentinel added (after `_HYDRO_REQUIRED`),
  mirroring `_DSCR_REQUIRED`, with the silent-default rationale.
- `load_scenario()` signature: `solar_tariff_case: str = "baseline"` →
  `tariff_case=_TARIFF_REQUIRED`. Docstring param updated.
- Validation (where the old `solar_tariff_case` membership check was):
  `TypeError` if `tariff_case is _TARIFF_REQUIRED` (message names the five
  keys and the retired 65M default), then `ValueError` on unknown key.
- Write site: `"solar_service_tariff_usd_per_twh": tariff_scenarios()[tariff_case]`.

**Callers that had to change (Step 1):**

| Site | Change |
|---|---|
| `src/optimize_experiments.py:3135` `run_financing_vs_resource_test` | `solar_tariff_case="low"` → `tariff_case="tariff_low"`. Unused helper, already dead on missing `peak_multiple_case` etc.; kept name-valid only. |
| `tests/test_run.py:8` | NOT changed. Already broken pre-task (omits `peak_multiple_case`, `dscr_case`, and ~13 other required args); a stale smoke script. Now also omits `tariff_case`. No regression. |

`TARIFF_SWEEP_GRID` and `04_run_fin2_tariff_sweep.py`: **untouched**, per
TASK. Note `04` was already non-functional (its `load_scenario()` call
omits `conversion_case` / `hydro_case`, required since Phase 2.8 / plan
1.5); it now also omits `tariff_case`. No behaviour change — it raised
before, it raises now.

### Step 2 — the eight setter sites

Step 0 found eight runner setter paths. Re-checked for **liveness** (does
the runner's `load_scenario()` call satisfy the ~20 required args added
across phases 2.6–2.13 + F1?). Result: **only `03` and `26` are live.**
The other six are dead *independent of this task* — each omits between 2
and 15 required arms and raises inside `load_scenario()` before the
tariff was ever read.

| Runner | Live? | Migration done | Verified |
|---|---|---|---|
| `03_run_ndc_eaas.py` | **yes** | `--tariff-case` required flag; `tariff_case=args.tariff_case` → `load_scenario()`; literal `95_000_000.0` deleted; appended to `case_dir` tag + `summary.json` (`tariff_case` + resolved `solar_service_tariff_usd_per_twh`) | ✅ full |
| `26_run_f3_identification.py` | **yes** | `--tariff-case` required flag; threaded to `load_scenario()`; `EAAS_TARIFF_USD_PER_TWH` constant + post-call literal deleted; appended to `arm_tag`, `summary["arms"]`, `summary["tariff_case"]`; `eaas_tariff_usd_per_twh` now the resolved value; SCOPED-CONDITION 1 and the "do not make it an arm" instruction in the module docstring marked superseded | ✅ full |
| `02_run_eaas.py` | no — omits `conversion_case`, `hydro_case` | `--tariff-case` required flag added; threaded to `load_scenario()`; literal deleted; tag + summary updated | source-consistent only; cannot run |
| `07_run_gas2_eaas_gas_relief.py` | no — no argparse at all; `load_scenario()` omits ~15 arms | no CLI to add a flag to. Literal `95_000_000` → `tariff_scenarios()["tariff_legacy_95m"]` via a module `CANONICAL_TARIFF_CASE`, threaded into both `load_scenario()` calls; post-call mutation removed | de-literalised only; cannot run |
| `15_run_pol1_ndc_comparison.py` | no — no argparse; `load_scenario()` omits ~15 arms | same treatment as `07`: `CANONICAL_TARIFF` literal → `tariff_scenarios()["tariff_legacy_95m"]`, `CANONICAL_TARIFF_CASE` threaded into `load_scenario()`; post-call mutation removed | de-literalised only; cannot run |
| `11_run_rel3_financing_frontier.py` | no — backend `run_rel3_financing_frontier` `load_scenario()` omits ~15 arms | config dicts: `"eaas_tariff": CANONICAL_TARIFF` → `"tariff_case": CANONICAL_TARIFF_CASE` (= `tariff_legacy_95m`); backend threads `tariff_case` into both `load_scenario()` calls, post-call tariff mutation removed | source-consistent only; cannot run |
| `12_run_dem1_demand_sensitivity.py` | no — marked `[SUSPENDED]` in its own docstring; backend omits ~15 arms | config dicts → `"tariff_case"`; backend `run_dem1_demand_sensitivity` threads it into `load_scenario()`, drops the `fin_params` tariff entry + post-call apply | source-consistent only; suspended |
| `13_run_dem2_growth_gas_interaction.py` | no — backend `run_dem2_growth_gas_matrix` `load_scenario()` (×2) omits ~15 arms | `FINANCING_CONFIGS[eaas]["tariff"]` → `["tariff_case"]`; backend threads `fin_cfg.get("tariff_case", "tariff_legacy_95m")` into both `load_scenario()` calls, post-call mutations removed | source-consistent only; cannot run |

Runners where the flag **could not be added mechanically** (reported, not
forced): `07` and `15` have no argparse whatsoever — adding a required
CLI flag means building a parser and, to get past `load_scenario()`,
reviving ~15 unrelated missing arms. Out of scope. They are de-literalised
to a named, attributable arm instead; a proper `--tariff-case` waits for
whenever those runners are revived.

**Net:** no bare tariff literal assigns `solar_service_tariff_usd_per_twh`
anywhere in `src/` or `scripts/` any more
(`grep -rn 'solar_service_tariff_usd_per_twh.*=.*95_000_000'` → empty).
The library grid-walks (`run_fin2_blended_bankability_sweep`,
`run_tariff_public_capital_frontier`) keep their per-grid-point
`sc[...] = tariff` assignment — that is a research sweep over a numeric
grid, not an arm selector, and TASK ring-fenced it.

### Step 3 — per-year bankability slack

New reader `bankability_slack_per_mw_by_year(m, scenario, years)` added to
**both** `03_run_ndc_eaas.py` and `26_run_f3_identification.py` (local to
each; no shared-module change). Formula exactly as TASK.md:

```
slack_per_mw[t] = tariff * solar_energy_per_mw * remaining_npv_factor[t] / required_margin
                  - (capex[t] - eaas_subsidy[t] / solar_eaas_add[t])
```

Guard: `solar_eaas_add[t] <= 1e-6` ⇒ emit `None` (constraint is `0 <= 0`,
no information); the 1e-6 MW tolerance absorbs solver noise around a true
zero.

- `03`: added to `run_case()`'s `out` dict as
  `"bankability_slack_per_mw_by_year"` (plus `"solar_service_tariff_usd_per_twh"`).
- `26`: computed off the **normal solve** (before the F3 min/max
  re-optimisations perturb the model), added to `summary` as
  `"bankability_slack_per_mw_by_year"`.

### Verification (bounded — `scratchpad/verify_tariff_arm.py`, non-solving)

| Check | Result |
|---|---|
| `load_scenario()` omitting `tariff_case` raises `TypeError` naming `tariff_case` | ✅ PASS |
| `tariff_scenarios()` arm values (75/95/115/95/65 M) | ✅ PASS |
| F3 central config + `tariff_case="tariff_central"` → `build_model_sliced()` (no solve) builds `m.eaas_bankability` over 30 years; `scenario["solar_service_tariff_usd_per_twh"] == 95_000_000` | ✅ PASS |
| Tariff scalar reaches `m.eaas_bankability[t]` body: isolated the tariff term by differencing the constraint body against a sentinel-tariff rebuild — observed `981,918.36`, expected `(95e6 − 1)·E·rnf[5]/required_margin = 981,918.36`, rel err `0.0` | ✅ PASS |

F3 central config used: `capital_case=moderate`, `dscr_central`,
`cr_central`, `f1_tranche`, `env_dares`, `conv_central`, `hydro_flat`,
`baseline`, `peak_unsuppressed`, `storage_cc_central`, `gas_unconstrained`,
`constrained`, `gas_retire_vintage_30`, `gas_avail_nameplate`,
`gas_capex_central`, `ccgt_eff_central`, `us_inflation=0.020`.

### Not done / deferred

- `02` full revival (needs `conversion_case` + `hydro_case` wired to CLI) —
  unrelated to tariff.
- `07`, `15` argparse construction + ~15-arm revival — unrelated to tariff;
  they are de-literalised, not made runnable.
- `11`, `12`, `13`, `04`, `tests/test_run.py` — same: pre-existing
  breakage on unrelated required args, left as found (`12` is
  additionally `[SUSPENDED]` by its own header).
- Whether `tariff_central` (a construction) or `tariff_high`/`tariff_low`
  should be the reported central case for F-27 is a research call for
  Zeus; the arm is now the mechanism, `tariff_legacy_95m` keeps the
  existing F-27 / F3 numbers attributable.

---

## Step 4 — repairing the five live runners

**The defect.** Step 1 made `tariff_case` required in `load_scenario()`
having analysed liveness only for the two TARIFF-SETTING runners (`03`,
`26`). Fifteen further runners call `load_scenario()` without setting a
tariff and all fifteen started raising `TypeError`. Five of the fifteen
are live and are repaired here:

  `01_run_baseline.py`, `01b_run_baseline_sliced.py`,
  `02_run_ndc_caps.py`, `22_salvage_sensitivity.py`,
  `23_run_selfgen_frontier.py`.

`01b` is the primary sliced runner and the source of the project's
regression anchor (21.4748 bn / 44,377 MWh), so until this step it was
impossible to verify the model.

### Pattern applied (all five)

Followed `03_run_ndc_eaas.py` exactly:

- `tariff_scenarios` added to the existing `from src.scenarios import
  (...)`.
- `TARIFF_CASE_CHOICES = tuple(tariff_scenarios())` beside the other
  `*_CHOICES` tuples — derived live from the registry.
- `--tariff-case`, `required=True`, `choices=TARIFF_CASE_CHOICES`, help
  text verbatim from TASK.md (the "retired solar_tariff_case defaulted to
  65,000,000 ... Inert in runners with no EaaS channel ... See
  tariff_scenarios()." text).
- `tariff_case=args.tariff_case` added to the `load_scenario()` call.

### Per-runner edits

| Runner | `load_scenario()` call site | Threading needed |
|---|---|---|
| `01_run_baseline.py` | `main()`, single call | none — direct `args.tariff_case` |
| `01b_run_baseline_sliced.py` | `main()`, single call | none — direct `args.tariff_case` |
| `02_run_ndc_caps.py` | `main()`, once per `NDC_CASES` loop iteration | none — direct `args.tariff_case` |
| `22_salvage_sensitivity.py` | inside helper `run_arm(...)` | `tariff_case` added as a new final parameter of `run_arm`, threaded exactly as `conversion_case`/`hydro_case` were; the one call site in `main()`'s arm loop updated to pass `args.tariff_case` |
| `23_run_selfgen_frontier.py` | inside helper `make_scenario(...)`, called from `solve_point(...)` | `tariff_case` added as a new final parameter of both `make_scenario` and `solve_point`, threaded exactly as `conversion_case`/`hydro_case`; the one call site in `main()`'s sweep loop updated to pass `args.tariff_case` |

All five: `tariff_case` recorded in the runner's summary/output dict
(`01`, `01b`: `summary["tariff_case"]`; `02`: per-case `summary.json`
`"tariff_case"`; `22`: top-level `out["tariff_case"]`; `23`: top-level
JSON `"tariff_case"`) so the value used is auditable from the result file
even though it does not affect the result (see below).

### Step 2 — deliberate deviation: no tag append

`01`, `01b`, `02_run_ndc_caps` and `22` never set `financing_regime`
anywhere in the file (confirmed by grep — no occurrence), so
`load_scenario()`'s `"traditional"` default holds and `m.eaas_disable`
fixes `solar_eaas_add[t] <= 0`. The tariff cannot affect any result in
these four. TASK.md instructs NOT to append `tariff_case` to the output
tag here — the arm is inert and appending it would lengthen tags that
already broke the filesystem's 255-byte path-component limit once
(`docs/f3_identification.md`, "Path-length fix"). Followed as instructed;
each runner instead records `tariff_case` in its summary dict only (see
table above).

**`23_run_selfgen_frontier.py` — checked, not assumed.** `grep -n
financing_regime scripts/23_run_selfgen_frontier.py` returns no match:
the script never sets `financing_regime` anywhere (`make_scenario()`,
`solve_point()`, `summarise()`, `main()`). `load_scenario()`'s
`"traditional"` default therefore holds for every point on the frontier
sweep (both builders, both prices, all 12 `eps` values), `m.eaas_disable`
fixes `solar_eaas_add[t] <= 0`, and the tariff is inert here exactly as
in the other four. Per TASK.md's instruction to report and follow the
finding rather than assume it: the tag is **not** appended for `23`
either, matching `01`/`01b`/`02`/`22`. `tariff_case` is recorded in the
frontier JSON's top-level metadata only.

### Step 3 — the ten runners left unrepaired

Confirmed dead **before** this task, on required args unrelated to
`tariff_case` (register L10). `load_scenario()`'s required-arg
validation now checks `tariff_case` first (`src/scenarios.py:2230`, the
earliest check in the function body), so all ten now raise on
`tariff_case` first where before they raised on whichever required arg
their code checked earliest — this is a change in which error message
appears first, not a change in liveness. None of the ten call
`load_scenario()` with `tariff_case` set; none are touched.

| Runner | Required args omitted (in addition to `tariff_case`, now also missing) |
|---|---|
| `04_run_fin2_tariff_sweep.py` | `conversion_case`, `hydro_case` (all other required args ARE supplied via argparse) |
| `05_run_fin3_capital_sweep.py` | all 15 others: `peak_multiple_case`, `reserve_margin`, `storage_cc_case`, `gas_build_case`, `gas_availability_case`, `gas_retirement_case`, `gas_capex_case`, `ccgt_efficiency_case`, `capital_recovery_case`, `dscr_case`, `us_inflation`, `capital_treatment_case`, `concessional_envelope_case`, `conversion_case`, `hydro_case` — no argparse at all |
| `06_run_gas1_shadow_benchmarks.py` | same 15 as `05` — no argparse at all |
| `18_run_pareto_frontier.py` | same 15 as `05` — no argparse at all |
| `19_run_carbon_price_sweep.py` | same 15 as `05` — no argparse at all |
| `20_run_cf1_solar_yield_sensitivity.py` | same 15 as `05` — no argparse at all |
| `24_run_leakage_arm.py` | `conversion_case`, `hydro_case` (all other required args ARE supplied via argparse, same shape as `04`) |
| `25_run_obligation_trajectory.py` | `conversion_case`, `hydro_case` (same shape as `04`) |
| `99c_run_genset_arm_sensitivity.py` | `conversion_case`, `hydro_case` (same shape as `04`) |
| `probe_2_6_structure.py` | same 15 as `05` — no argparse at all |

This is the record that the breakage on all ten predates this task and
this task did not cause it: `04`, `24`, `25`, `99c` were already dead on
`conversion_case`/`hydro_case` (unrelated to tariff, per Step 1's
`04` note above); `05`, `06`, `18`, `19`, `20`, `probe_2_6_structure`
were already dead on the full set of 15 pre-tariff required args
(register L10). None of the ten is touched by this step, per TASK.md.

### Verification

None performed beyond reading the edited code, per TASK.md's bounded
verification instruction — no solver, no runner, no smoke suite, no
compile check. Zeus runs the regression: `01b` under
`--tariff-case tariff_legacy_65m` (plus the existing required arms) must
return 21.4748 bn / 44,377 MWh — `tariff_legacy_65m` is the value
`load_scenario()` silently supplied before the Step 1 migration, so it is
the value every pre-migration result (including the regression anchor)
was produced under.
