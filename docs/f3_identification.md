# F3 — Lexicographic identification of EaaS solar capacity

**Question.** Under an NDC-capped, EaaS-financed pathway, is *total EaaS
solar capacity* uniquely identified at the optimal system cost, or does a
range of EaaS deployments all achieve the same cost (a degenerate
optimum)? The F0 audit found pre-F1 degeneracy (every H2 tie cell was
`baseline_no_policy`). F1 was expected to resolve it; F3 measures whether
it did.

**Method.** Solve normally for `Z* = value(m.system_cost_npv)`; add
`m.system_cost_npv <= Z*·(1+tol)` as a hard constraint; re-optimise twice
on `Σ_t m.solar_eaas_add[t]` — once minimising, once maximising. The gap
is the identification interval. Repeated at `tol ∈ {1e-6, 1e-5, 1e-4}` in
one invocation so numerical vs. structural degeneracy are separable.

**Scaffold.** Built on `03_run_ndc_eaas.py` — the only runner that sets
both `financing_regime="eaas"` and `solar_service_tariff_usd_per_twh`
(`03_run_ndc_eaas.py:428-429`). EaaS is therefore tested *inside* an NDC
pathway (`ndc3_conditional`), not on the baseline.

---

## Step 0 — Inventory (locate and quote, no change)

### (a) `m.solar_eaas_add` and how total EaaS MW accumulates

- **Declaration** — `src/optimize_model_sliced.py:576`:
  ```python
  m.solar_eaas_add = pyo.Var(T, domain=pyo.NonNegativeReals)
  ```
  One free variable per model year `t` (30 years, 2025–2054). Annual MW
  *added*, not cumulative.

- **Accumulation into installed capacity** — `src/optimize_model_sliced.py:705-710`:
  ```python
  def solar_cap_rule(m, t):
      return baseline_mw + sum(
          m.solar_public_add[k] + m.solar_eaas_add[k] for k in range(0, t + 1)
      )
  m.solar_capacity_mw = pyo.Expression(T, rule=solar_cap_rule)
  ```
  EaaS and public additions enter installed capacity **additively and
  symmetrically** — the model sees only their sum `solar_capacity_mw`
  everywhere downstream (dispatch, land, salvage, replacement, the
  cumulative-build cap at `:1132-1137`). Cross-checked: `solar_eaas_add`
  appears at lines 576, 587-588 (`fixed_plan`), 707, 1055, 1070-1074,
  1082, 1092, 1134, 1146, 1383, 1401, 1432, 1578-1586, 1703 — in every
  one it is paired with or parallel to `solar_public_add`.

- **There is no model component for horizon-total EaaS MW.** It is
  `sum(m.solar_eaas_add[t] for t in T)`, formed in the runner. `03_run_ndc_eaas.py:310-314`
  does exactly this (summing public + EaaS). F3's re-optimisation
  objective is `pyo.quicksum(m.solar_eaas_add[t] for t in T)`.

### (b) `m.eaas_bankability` and `m.eaas_subsidy_limit`

- **`m.eaas_bankability`** — `src/optimize_model_sliced.py:1068-1078`, built **only if `tariff is not None`**:
  ```python
  def eaas_bankability_rule(m, t):
      capex = m.solar_capex_param[t] * m.solar_eaas_add[t]
      effective_private_cost = capex - m.eaas_subsidy[t]
      bankable_revenue = (
          tariff * solar_energy_per_mw_annual
          * m.remaining_npv_factor[t] * m.solar_eaas_add[t]
      ) / required_margin
      return effective_private_cost <= bankable_revenue
  m.eaas_bankability = pyo.Constraint(T, rule=eaas_bankability_rule)
  ```
  Both sides scale linearly in `m.solar_eaas_add[t]`, so this is a
  per-MW test: EaaS build is admissible in year `t` iff
  `capex_param[t] - subsidy_per_mw ≤ tariff·energy_per_mw·remaining_npv_factor[t]/required_margin`.
  At `tariff = 95e6` and `solar_low` capex this holds with slack, so
  bankability does not by itself pin the EaaS quantity — it sets a
  ceiling, not a level.
  - `solar_energy_per_mw_annual = scenario["solar_cf"] * 8760 / 1e6` (`:1022`)
  - `m.remaining_npv_factor[t]` — private-rate remaining-life annuity factor (`:435-440`)
  - `required_margin` — DSCR cushion from `dscr_case`, direct key access, no fallback (`:462-467`)

- **`m.eaas_subsidy_limit`** — `src/optimize_model_sliced.py:1085-1097`:
  ```python
  if tariff is not None:
      def subsidy_limit_rule(m, t):
          max_bankable_at_t = (
              tariff * solar_energy_per_mw_annual
              * m.remaining_npv_factor[t] / required_margin
          )
          gap_at_t = max(0.0, capex_init[t] - max_bankable_at_t)
          return m.eaas_subsidy[t] <= gap_at_t * m.solar_eaas_add[t]
  else:
      def subsidy_limit_rule(m, t):
          return m.eaas_subsidy[t] <= 0.0
  m.eaas_subsidy_limit = pyo.Constraint(T, rule=subsidy_limit_rule)
  ```
  `m.eaas_subsidy` (`:577`) is a free non-negative var; it enters the
  public budget sum (see (c)) and relaxes bankability. When the tariff
  covers capex, `gap_at_t = 0` and subsidy is forced to zero.

- **`m.eaas_disable`** — `src/optimize_model_sliced.py:1080-1083`, built when `financing_regime != "eaas"`:
  `m.solar_eaas_add[t] <= 0.0`. Not built here (regime is `eaas`).

### (c) The public budget constraint

- **Name / bound** — `src/optimize_model_sliced.py:1882-1895`:
  ```python
  public_budget_npv = scenario.get("public_solar_budget_npv", None)
  if public_budget_npv is not None:
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
      m.has_public_budget_constraint = True
  ```
  It is called **`m.public_budget_constraint`**. It bounds the
  social-discounted (`m.DF[t]`) sum of **public solar capex + storage
  capex + EaaS subsidy** over the horizon. It does **not** bound
  `m.solar_eaas_add` directly — EaaS capex is privately financed and
  never appears in this sum; only the EaaS *subsidy* draws on the public
  purse.

- **Not built when `public_solar_budget_npv is None`.** The guard is the
  literal `if public_budget_npv is not None:` at `:1884`. When the key is
  `None`, neither `m.public_budget_constraint` nor
  `m.has_public_budget_constraint` is created. `capital_envelope_scenarios()`
  (`src/scenarios.py:794-800`) maps `"unconstrained" → None`, and
  `load_scenario` writes `capital_envelope_scenarios()[capital_case]` to
  the key (`src/scenarios.py:2845`), so `--capital-case unconstrained`
  reproduces `03_run_ndc_eaas.py:440`'s `scenario["public_solar_budget_npv"] = None`
  through the arm rather than a post-call override.
  Verified present under `--capital-case moderate` (see Verification).

### (d) How `m.system_cost_npv` is exposed; can it be fixed as a constraint?

- **Exposure** — `src/optimize_model_sliced.py:1865`:
  ```python
  m.system_cost_npv = pyo.Expression(expr=system_cost_npv)
  ```
  assembled at `:1844-1864` in two branches:
  - `pre_f1_overnight`: `gas_opex + public_solar_capex + eaas_solar_capex + gas_capex + storage_capex + replacement_capex + selfgen + unserved + carbon − salvage`
  - `f1_tranche` (central config): `gas_opex + capital_charge_total_npv + selfgen + unserved + carbon`

- **Objective** — `src/optimize_model_sliced.py:1906` (the
  `emissions_cap_by_year is not None` branch, which F3 takes):
  ```python
  m.objective = pyo.Objective(expr=system_cost_npv, sense=pyo.minimize)
  ```

- **Can it be constrained cleanly? YES — step 0 STOP is NOT triggered.**
  `m.system_cost_npv` is a named `ScalarExpression`. Confirmed live
  (non-solving):
  1. `m.f3_zstar_con = pyo.Constraint(expr=m.system_cost_npv <= K)` builds without error.
  2. `m.objective.deactivate()` + `m.f3_eaas_obj = pyo.Objective(expr=Σ solar_eaas_add, sense=…)`
     leaves exactly one active objective (`f3_eaas_obj`).
  3. `del_component('f3_eaas_obj'); del_component('f3_zstar_con'); m.objective.activate()`
     restores the canonical single active objective (`objective`).

  No restructuring of the objective expression is needed; the same model
  object is reused across the min/max/tol sweep with add/delete of two
  helper components.

---

## Why the public budget must be an arm (not a fixed choice)

Under F1, public and EaaS solar are priced **identically** — same
`m.solar_capex_param[t]`, same CRF path. The double-count guard comment
(`:1039-1053`) states they are "the same physical assets" at "the SAME
unit CAPEX". If both channels cost the same and public solar is
unbounded, the optimiser is indifferent to the public/EaaS split: any
split gives the same `system_cost_npv`. F3 would then report an interval
from ~0 to the bankability ceiling — a restatement of the configuration,
not a measurement.

EaaS's claimed mechanism is that it **decouples solar from the public
capital ceiling**. Remove the ceiling and there is nothing to decouple
from. So F3 runs **both** configurations and reports them as a pair:

| arm | `--capital-case` | `public_solar_budget_npv` | `m.public_budget_constraint` |
|---|---|---|---|
| CONSTRAINED | `moderate` | 7,738,400,000 (0.85 × B\* = 9.104 bn) | built |
| UNCONSTRAINED | `unconstrained` | `None` | not built |

**Expected finding shape:** if the unconstrained interval is wide and the
constrained interval tight, EaaS capacity is identified **only where the
public ceiling binds** — the mechanism the thesis claims. Zeus runs the
solves; this doc is completed with the intervals once results land.

---

## Step 1 — Runner `scripts/26_run_f3_identification.py`

Follows `03_run_ndc_eaas.py`'s scaffold:

- **Same required flags** — all 16 arm flags of `03_run_ndc_eaas.py`
  (`--peak-case … --hydro-case`, `--us-inflation`), same `choices`
  tuples, sourced live from `src/scenarios.py` for the three that can
  drift (`gas_deliverability_scenarios`, `conversion_scenarios`,
  `hydro_case_scenarios`).
- **Same `load_scenario()` call** — identical keyword set and values
  (`demand_level_case="served"`, `demand_case="organic_central"`,
  `carbon_case="no_policy"`, `reserve_margin="nirp_spinning"`,
  `start_year=2025`, `end_year=MODEL_END_YEAR`), except `capital_case`
  now comes from `--capital-case` instead of the `NDC_CASES` dict.
- **Same post-call mutations** — `scenario["financing_regime"] = "eaas"`
  and `scenario["solar_service_tariff_usd_per_twh"] = 95_000_000.0`,
  **and nothing else**. F3 deliberately does **not** set
  `scenario["public_solar_budget_npv"] = None` (that override at
  `03_run_ndc_eaas.py:440` is exactly what F3 turns into an arm).
- **NDC cap** — `ndc3_conditional` only, via `emissions_cap_by_year`.
- **econ** — `load_econ("voll_mid", genset_case="genset_central")`.

**New flags:**

- `--capital-case`, **required, no default**, choices =
  `tuple(capital_envelope_scenarios())` (which already contains
  `"unconstrained"`) deduped with the literal `"unconstrained"` →
  `("tight", "moderate", "adequacy", "expansion", "unconstrained")`.
- `--tol`, **required float, no default**. The runner additionally
  **always** evaluates `1e-6, 1e-5, 1e-4` in the same invocation
  (`TOLS_ALWAYS`); `--tol`'s value is unioned in. Rationale in the
  docstring: a degenerate LP shows a wide interval even at `1e-6`; a
  numerically-tight optimum widens only as `tol` grows — both must be
  visible from one run.

**Three solves per configuration, per tolerance** (`reoptimise_eaas`):

1. normal solve → `Z* = value(m.system_cost_npv)`
2. `m.system_cost_npv <= Z*·(1+tol)`, objective = **minimise** `Σ_t m.solar_eaas_add[t]`
3. same constraint, objective = **maximise** `Σ_t m.solar_eaas_add[t]`

The `Z*` constraint and EaaS objective are added and removed around each
solve; `m.objective` is deactivated then reactivated, so the canonical
model is restored between calls. Every solve is checked for
`status["optimal"]` and raises otherwise.

**Reported quantities** (`results/f3/<arm-tag>/summary.json`):

- `Zstar_system_cost_npv_usd`
- per tolerance: `total_eaas_mw_min`, `total_eaas_mw_max`,
  `identification_interval_mw` (= max − min), the full min-total and
  max-total solutions, and `system_cost_npv` actually achieved by each
- **per-year envelope**: for each year, `eaas_add_mw_lower` /
  `eaas_add_mw_upper` (elementwise across the min-total and max-total
  profiles), `eaas_add_mw_span`, and `max_per_year_span_mw`. A tight
  total with a wide per-year profile is a distinct result and is not
  hidden. (Per TASK.md's "three solves per configuration", the envelope
  is read off the min/max solutions, not re-optimised year by year.)
- `public_budget_constrained`, `public_solar_budget_npv`,
  `has_public_budget_constraint`, `required_margin`, and the full arm
  block + scenario labels.

Arm tag mirrors `03_run_ndc_eaas.py`'s directory naming, prefixed with
`ndc3_conditional_<capital_case>_voll_mid_…`.

---

## Step 2 — Scoped conditions (stated in the runner docstring)

Reproduced verbatim in `scripts/26_run_f3_identification.py`'s module
docstring:

1. `solar_service_tariff_usd_per_twh = 95,000,000` is a **bare literal
   with no arm**, sitting directly upstream of `m.eaas_bankability`. If
   it becomes an arm, the interval moves.
2. EaaS and public solar are **physically identical** in the current
   model. `[DIST-2]` (behind-the-meter avoids the 7% TLF — favours EaaS)
   and `[DIST-5]` (distributed commercial PV CAPEX premium +16% (2025) →
   +37% (2045) per NREL ATB, plus a distributed CF differing from
   utility-scale 0.20 — disfavours EaaS) are sourced but unimplemented
   and oppose each other. Their absence makes the two channels **harder**
   to distinguish and degeneracy **more** likely, not less.
3. `capital_envelope_scenarios()` (`src/scenarios.py:788-790`) carries a
   **stale-doc warning**: an earlier docstring quoted B\* ~ 6.13 bn
   against the coded 9.104 bn, unreconciled. The constrained arm inherits
   it (`moderate` = 0.85 × 9.104 bn = 7.7384 bn).
4. `gas_avail_nameplate` = 0.80 against a measured 29.5% (P2-9) [30.6% since TASK 73].
   Regression-consistent; retained to avoid confounding F3.
5. The NDC cap is an additional constraint on the same variables, so it
   **narrows** the interval. A tight interval here establishes
   identification **where the cap binds**, not in general.

---

## Verification (bounded — one non-solving `build_model_sliced()` call)

Config: central configuration + `--capital-case moderate`. No solve.

```
model years: 2025 -> 2054 (30)
solar_eaas_add exists: True | len: 30
system_cost_npv: ScalarExpression, present; usable in pyo.Constraint(expr=...)
public_solar_budget_npv: 7738400000
has_public_budget_constraint: True
public_budget_constraint component built: True
m.objective active: True  (single active objective)
lexicographic swap: deactivate m.objective, add m.f3_eaas_obj  -> 1 active obj
restore: del helpers, activate m.objective                     -> 1 active obj ('objective')
```

- `m.solar_eaas_add` exists (30 elements). ✔
- `m.system_cost_npv` is a named expression, readable, and accepts
  `pyo.Constraint(expr=m.system_cost_npv <= K)` — step 0 (d) STOP not
  triggered. ✔
- `m.public_budget_constraint` **is** built under `--capital-case
  moderate`; `m.has_public_budget_constraint` is `True`. ✔

Pre-solve `pyo.value(m.system_cost_npv)` returns `None` (variables
uninitialised) — expected; the expression is well-formed and evaluates
after any solve.

No solves were run. Zeus runs all solves:
`--capital-case moderate` and `--capital-case unconstrained`, central
configuration, `--tol` at operator's choice (the runner adds 1e-6/1e-5/1e-4).

---

## Path-length fix

**Symptom.** All F3 solves ran to optimality, but the result write
failed: `case_dir.mkdir()` raised `OSError [Errno 36] File name too
long`. The arm tag — sixteen required arms joined into one string —
exceeds the filesystem's ~255-byte per-path-component limit. Shortening
this one tag would not help: every runner accumulates arms and will hit
the same wall.

**Fix (F3 only, this task).** `scripts/26_run_f3_identification.py` now
writes to:

```
results/f3/<ndc_case>_<capital_case>_<hash8>/summary.json
```

where `hash8 = sha1(arm_tag).hexdigest()[:8]` over the FULL arm-tag
string. Deterministic — re-running the same configuration overwrites
rather than accumulating. The full tag is preserved as `summary["arm_tag"]`
alongside the existing per-arm `arms` block, so attributability is
intact where the tag is actually read from (the file). Each run also
appends one `` `hash8` -> `arm_tag` `` line to `results/f3/INDEX.md`
(append-only, deduplicated) so the directory is navigable without
opening each JSON.

Implementation: `scripts/26_run_f3_identification.py` — `import hashlib`
(`:84`), full arm tag built at `arm_tag = (...)` (`:402`), written to
`summary["arm_tag"]` (`:411`), `hash8` / `short_name` / `case_dir`
(`:413-416`), `INDEX.md` append (`:423-429`) after the `summary.json`
write.

### Scope of the same bug elsewhere (reported, NOT fixed here)

Two construction patterns build a path element from the full arm tag.
Both will fail as arms accumulate; the `mkdir` sites fail first because a
directory component has no room for the `<prefix>__ ... .json` padding
that the filename sites also carry.

**Pattern A — full tag is a DIRECTORY component (`case_dir.mkdir()`), the
exact failure mode F3 hit:**

| runner | tag expression | `mkdir` site |
|---|---|---|
| `scripts/02_run_ndc_caps.py` | `:414-422` (`case_dir = RESULTS_DIR / ( f"{c}_{voll_case}_{args.peak_case}..." )`) | `:424` |
| `scripts/03_run_ndc_eaas.py` | `:450-458` (same shape) | `:460` |

**Pattern B — full tag is a FILENAME component (`RESULTS_DIR /
f"<prefix>__{tag}.<ext>"`):**

| runner | `tag =` | tag arms | write sites |
|---|---|---|---|
| `scripts/01_run_baseline.py` | `:240` | 16 (`__`-joined, incl. gas_deliverability/conversion/hydro) | `:342`, `:395`, `:425` |
| `scripts/01b_run_baseline_sliced.py` | `:232` | 16 | `:454`, `:494`, `:517`, `:525` |
| `scripts/02_run_eaas.py` | `:185` | 13 (stops at concessional_envelope_case) | `:305`, `:333`, `:363` |
| `scripts/22_salvage_sensitivity.py` | `:527` | 16 | `:528`, `:529` |
| `scripts/23_run_selfgen_frontier.py` | `:490` | 16 | `:494`, `:496` |
| `scripts/24_run_leakage_arm.py` | `:337` | 13 | `:338` |
| `scripts/25_run_obligation_trajectory.py` | `:352` | 13 | `:354`, `:357` |
| `scripts/99c_run_genset_arm_sensitivity.py` | `:397` | 13 | `:399` |

`scripts/18_run_pareto_frontier.py`, `19`, `20` and the other sweep
runners build their output names from sweep indices / financing labels,
not the arm tag, and are unaffected.

A follow-up should lift the `<ndc_case>_<capital_case>_<hash8>` scheme
(full tag retained inside the file + an `INDEX.md` map) into a shared
helper in `src/` and route all ten runners above through it.

(Done separately: `src/output_naming.py`, `docs/output_naming.md`.)

---

## Full solution vector

**The gap.** `normal_solve` recorded `solar_public_add_mw_total`
alongside the EaaS fields, but `by_tol[<tol>].min_solution` and
`.max_solution` recorded only `total_eaas_mw`, `eaas_add_mw_by_year` and
`system_cost_npv`. After the shape-factor sourcing
(`docs/shape_factors_and_dist5.md`, Part A) the EaaS floor fell to 0.0 in
every budget arm including `tight` (4.552 bn), where it was previously
36,687 MW. With only EaaS recorded at the min/max solves, two
incompatible explanations could not be told apart: public solar
absorbing the difference, or total solar collapsing outright with
another technology (gas, storage, selfgen) meeting demand instead.

**Fix.** `scripts/26_run_f3_identification.py`:

- New `full_solution_vector(m, T, years)` (added just after
  `eaas_profile`), returning:

  | field | reads |
  |---|---|
  | `solar_public_add_mw_total` | `sum(m.solar_public_add[t] for t in T)` |
  | `solar_capacity_mw_2045` | `m.solar_capacity_mw[t]` at `t = years.index(2045)` (total solar, both channels — the cumulative stock is already channel-combined, see `docs/shape_factors_and_dist5.md` Part B0) |
  | `gas_capacity_mw_2045` | `m.gas_capacity_mw[t]` at the same index |
  | `storage_capacity_mwh_2054` | `m.storage_capacity_mwh[t]` at `t = years.index(2054)` (`MODEL_END_YEAR`, the horizon's last year) |
  | `selfgen_twh_e_total` | `sum(m.selfgen[t, sp] for t in T for sp in m.S)` |
  | `gas_generation_twh_e_total` | `sum(m.gas_generation[t, sp] for t in T for sp in m.S)` |
  | `converted_generation_twh_e_total` | `sum(m.convert_generation[tier, t, sp] for tier in ("A","B","C") for t in T for sp in m.S)` if `hasattr(m, "convert_generation")`, else `None` |

- `reoptimise_eaas()` now takes `years` as an extra parameter and calls
  `out.update(full_solution_vector(m, T, years))` immediately after the
  solve succeeds and BEFORE the `f3_eaas_obj`/`f3_zstar_con` cleanup
  (comment states this explicitly) — reads the SAME model object the
  re-optimisation just solved, no re-solve. Both call sites in `main()`'s
  tolerance loop (`lo = reoptimise_eaas(...)`, `hi = reoptimise_eaas(...)`)
  updated to pass `years`; `min_solution`/`max_solution` in `by_tol`
  already store the full `lo`/`hi` dicts verbatim, so no further change
  was needed there to surface the new fields into `summary.json`.

**Which fields already existed vs. required a new expression: NONE
required a new expression.** All seven are direct reads of Var/Expression
components `build_model_sliced()` already builds unconditionally (or, for
`convert_generation`, conditionally on the conversion arm — guarded with
`hasattr`, not added):

- `m.solar_public_add[t]` (Var) — already read elsewhere in this same
  file (`normal_solve`'s `solar_public_add_mw_total`).
- `m.solar_capacity_mw[t]`, `m.gas_capacity_mw[t]`,
  `m.storage_capacity_mwh[t]` (Expressions) — pre-existing cumulative
  stock/capacity Expressions in `optimize_model_sliced.py`
  (`solar_cap_rule`, `gas_cap_rule`, `storage_cap_rule`).
- `m.selfgen[t, sp]` (Var), `m.gas_generation[t, sp]` (Expression) —
  pre-existing, slice-indexed.
- `m.convert_generation[tier, t, sp]` (Expression) — pre-existing,
  built only when a conversion arm is active; read via the same
  `hasattr` guard `conversion_diagnostic()` and the diagnostics-layer
  generation-share fix (`docs/metric_fixes_0_3.md`, defect 1) already
  use, so no new component and no new guard pattern.

Per TASK.md's fallback instruction ("if any is not readable without a
new Expression, report it and skip that field"): not triggered — every
field was readable as-is.

**Not touched, per TASK.md:** the lexicographic method, the tolerance
sweep, the objective-swap mechanics, and every existing arm/parameter.
`git diff` confirms the only changes are the new function, the
`reoptimise_eaas` signature/body addition, and the two call-site
argument updates.

**Verification (bounded).** `python3 -m py_compile
scripts/26_run_f3_identification.py` — passes. No solver, no runner, no
build call was run, per TASK.md. Zeus re-runs the three budget arms.
