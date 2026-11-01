# shape_factors_and_dist5.md — sourced shape factors, CF band, and DIST-5

Per TASK.md, Part A is reported first and was completed before Part B0
began. Part B0 is inventory only (no code change), and its finding
triggers TASK.md's own STOP clause — Parts B1-B4 are NOT implemented.
That is reported at the end, not silently skipped.

---

# PART A — sourced shape factors and CF band

All three edits are in `src/scenarios.py`'s scenario dict (the values
`load_scenario()` actually returns) plus every literal duplicate found
by inventory before editing (per the standing rule: locate every
consumer before touching a changed quantity).

## A1. `season_solar_factor`

**Changed**, `src/scenarios.py` (scenario-dict entry, canonical site):
`{"dry": 1.15, "wet": 0.85}` → `{"dry": 1.076, "wet": 0.925}`, with the
verbatim provenance comment from TASK.md placed at the definition site.

**Duplicate found and kept in sync**: `DEFAULT_SEASON_SOLAR_FACTOR` in
`src/optimize_model_sliced.py` (a `.get(key, DEFAULT)` fallback for a
hand-built scenario dict that omits the key — `load_scenario()` always
sets it, so this default is not exercised in normal operation, but was
carrying the same now-stale literal). Updated to the same value, with a
comment pointing back to `scenarios.py` as the canonical provenance
site rather than duplicating the full block.

No other occurrence of this quantity exists (`grep` for the literal and
for `season_solar_factor` confirms exactly these two sites).

## A2. `hydro_season_factor` — sign reversal

**Changed**, `src/scenarios.py` (scenario-dict entry, canonical site):
`{"dry": 0.90, "wet": 1.10}` → `{"dry": 1.045, "wet": 0.956}`, with the
verbatim provenance comment from TASK.md, including the reservoir/
run-of-river explanation for why the retired sign was backwards.

**Two more duplicates found and kept in sync** (this quantity had THREE
literal copies before this task, not one):

1. `src/scenarios.py`, inside the same function, ~30 lines above the
   scenario dict: a local literal `{"hydro_season_factor": {"dry": 0.90,
   "wet": 1.10}}` passed directly into `_hydro_season_energy_share()` to
   compute `capacity_credit["hydro"]`. This runs BEFORE the scenario
   dict itself is constructed (it's needed to build `capacity_credit`,
   which feeds into the scenario dict later), so it could not simply
   reference the scenario-dict entry — it is a structurally necessary
   second literal, not an accident, but it DOES need updating in lock
   step or `capacity_credit["hydro"]` would silently keep using the
   retired, wrong-signed value while the rest of the model used the
   corrected one. Updated to `{"dry": 1.045, "wet": 0.956}`.
2. `DEFAULT_HYDRO_SEASON_FACTOR` in `src/optimize_model_sliced.py` (same
   fallback-default category as A1's solar duplicate). Updated to the
   same value.

All three sites now carry the corrected value; comments at sites 2 and 3
point back to the scenario-dict entry as the canonical provenance record
rather than repeating the full derivation. The existing
"SEASONAL-SCISSOR CAUTION" note at the `DEFAULT_HYDRO_SEASON_FACTOR` site
was NOT contradicted by this change — re-read carefully, it warns that
the 2025-only quarterly series (Zungeru ramping) cannot by itself
establish a seasonal direction; the new sourcing sidesteps exactly that
confound by using the PRE-Zungeru 2023 dataset instead, cross-checked
against a Zungeru-excluded 2025 ratio. The caution is updated to say so
explicitly rather than left to read as unresolved.

`grep` for the literal and for `hydro_season_factor` confirms exactly
these three sites — no fourth exists.

## A3. Solar capacity factor band

**Changed**, `src/scenarios.py`: `"solar_cf": 0.20` → `"solar_cf":
0.19`, with the verbatim provenance comment from TASK.md (site table,
NIRP 2024 cross-check, the "national maximum" critique of the retired
central value, and the DECLARED note on the model having no siting/
transmission).

**The CF band itself** (`{0.18, 0.20, 0.22}` → `{0.16, 0.19, 0.20}`) is
NOT a `scenarios.py` scenario-dict key or a named-arm dict anywhere in
the codebase (`grep` for `cf_high`/`cf_low`/`cf_central`/
`solar_cf_scenarios` returns nothing) — TASK.md's "keep the existing arm
key name" refers to the informal low/central/high labelling already used
in `scripts/20_run_cf1_solar_yield_sensitivity.py`'s own docstring and
`CF_SWEEP` list, the ONLY place in the repository that sweeps this band
(confirmed: `grep` for `0.18`/`0.22` alongside `CF_SWEEP` finds no other
site). Updated there:
- `CF_SWEEP = [0.18, 0.20, 0.22]` → `CF_SWEEP = [0.16, 0.19, 0.20]`.
- Docstring's "CF BAND" section rewritten with the new values, sourcing,
  and the retired band's two-fault critique (central at the national
  maximum, high above every measured site).

**Note on this runner's liveness.** `20_run_cf1_solar_yield_sensitivity.py`
is one of the ten runners already dead on unrelated required-arg
omissions (register L10, reconfirmed independently in
`docs/output_naming.md`'s step-3 classification this session) — its
`load_scenario()` call is missing all sixteen required keys. Updating its
CF band is still correct to do (a value correction, not a revival
attempt; TASK.md does not exempt dead runners from Part A), but the
runner cannot currently execute regardless of this change.

## Verification (Part A)

`python3 -m py_compile src/scenarios.py src/optimize_model_sliced.py
scripts/20_run_cf1_solar_yield_sensitivity.py` — passes on all three.

Non-solving sanity check (a `load_scenario()` call at the F3 central
configuration, no model build, no solve):

```
solar_cf 0.19
season_solar_factor {'dry': 1.076, 'wet': 0.925}
hydro_season_factor {'dry': 1.045, 'wet': 0.956}
```

All three resolve to the new sourced values with no error.

**Report Part A and STOP before starting Part B** — done; the rest of
this document is Part B0 onward, produced in the same task per TASK.md's
own step ordering ("Report Part A before starting Part B" — not "in a
separate task"), but no Part A code was touched again after this point.

---

# PART B — DIST-5

## B0. Inventory (locate and quote, NO change)

All four sub-questions answered for BOTH builders.

### (a) Where `solar_cf` enters solar generation, and whether the two channels share it

**`src/optimize_model.py` (annual):**
- `:684` — `solar_energy_per_mw = scenario["solar_cf"] * 8760 / 1_000_000`
  — ONE scalar, computed once, no channel argument.
- `:737-743` — `solar_cap_rule`: `m.solar_capacity_mw[t] = baseline_mw +
  sum(m.solar_public_add[k] + m.solar_eaas_add[k] for k in range(0, t+1))`
  — a SINGLE combined cumulative stock; public and EaaS additions are
  summed together before this expression exists at all.
- `:746-747` — `m.solar_generation = pyo.Expression(T, rule=lambda m, t:
  m.solar_capacity_mw[t] * solar_energy_per_mw)` — the combined stock
  times the single shared scalar.

**Answer: YES, shared — and inseparably so.** There is no per-channel
generation term anywhere; `m.solar_generation[t]` cannot distinguish an
EaaS-built MW from a publicly-built MW once `solar_capacity_mw[t]` has
summed them.

**`src/optimize_model_sliced.py` (sliced):**
- `:450` — `m.solar_capex_param` (see (c); not the CF, but the same
  "one shared Param" shape recurs).
- `:757` — `solar_per_mw = _solar_energy_per_mw_by_slice(scenario)` —
  computed ONCE.
- `:163-191` (`_solar_energy_per_mw_by_slice`) — `cf_annual =
  scenario["solar_cf"]` (`:172`), redistributed across day slices; no
  channel argument, no channel-indexed output.
- `:698-710` — `solar_cap_rule`, identical shape to the annual builder:
  `baseline_mw + sum(solar_public_add[k] + solar_eaas_add[k] for k in
  range(0, t+1))`.
- `:790-793` — `m.solar_generation = pyo.Expression(T, m.S, rule=lambda
  m, t, s, p: m.solar_capacity_mw[t] * solar_per_mw[(s, p)])`.
- Separately, `:1022` — `solar_energy_per_mw_annual = scenario["solar_cf"]
  * 8760 / 1_000_000`, a SECOND CF-derived scalar (annual, not
  slice-resolved) computed independently for the bankability test only
  — see (d).

**Answer: YES, shared — and inseparably so**, identical structure to the
annual builder.

### (b) Where `season_solar_factor` enters, same question

Sliced builder only (the annual builder has no intra-year resolution and
never reads this key — confirmed, `grep season_solar_factor
src/optimize_model.py` returns nothing).

- `src/optimize_model_sliced.py:173` — `season_f = scenario.get(
  "season_solar_factor", DEFAULT_SEASON_SOLAR_FACTOR)`, inside
  `_solar_energy_per_mw_by_slice` (same function as (a)). Feeds the same
  `solar_per_mw` dict, which feeds the same combined
  `m.solar_generation` at `:790-793`.

**Answer: YES, shared — and inseparably so**, for the same structural
reason as (a): one function computes one per-MW-by-slice yield from one
CF and one seasonal factor, and that single result multiplies a capacity
stock that already has no channel identity.

### (c) `solar_capex_param[t]` / `load_solar_capex_by_year` — every site both channels draw on it

`m.solar_capex_param` is a single time-indexed `pyo.Param` in both
builders (`optimize_model.py:371`, `optimize_model_sliced.py:450`),
built once from `load_solar_capex_by_year` (or the fixed
`econ["SOLAR_CAPEX_PER_MW"]` scalar fallback). Sites where both channels
draw on it (annual builder line numbers; sliced builder mirrors each
exactly, cited alongside):

| Site | Channels combined before multiply? | Restructuring needed for a per-channel premium? |
|---|---|---|
| `public_solar_capex_npv` (`:704-708`) / `eaas_solar_capex_npv` (`:720-724`) — pre_f1_overnight primary CAPEX NPV | **NO** — already two separate `pyo.quicksum`s, one per channel, each using its own add-variable | No — redirect the EaaS sum to a distinct EaaS capex param |
| `eaas_bankability_rule`'s local `capex = m.solar_capex_param[t] * m.solar_eaas_add[t]` (`:756`; sliced `:1065`) | **NO** — already EaaS-only | No — same redirect |
| F1 split identity: `solar_public_split_identity` (`:1353-1357`) / `solar_eaas_split_identity` (`:1359-1365`); sliced `:1356`/`:1364` region (mirrors) | **NO** — already two separate constraints, one per channel | No — redirect the EaaS constraint's RHS to a distinct EaaS capex param; `cost_public_solar_capital_charge_npv` / `cost_eaas_solar_capital_charge_npv` (`:1437`, `:1444`) already read from the resulting per-channel `*_conc_vol`/`*_comm_vol` Vars, so the capital charge inherits the redirect with no further change |
| `m.salvage_npv` (pre_f1_overnight, `:1146-1186`, and its "replacement vintages" quicksum `:1166-1172`) | **YES** — `(m.solar_public_add[t] + m.solar_eaas_add[t]) * m.solar_capex_param[t]` (and the build_t/rebuild_t version), same pattern in `optimize_model_sliced.py:1383,1401` | **YES** — would need splitting into two `quicksum`s, one per channel, each against its own capex param |
| `m.cost_replacement_capex_npv` (pre_f1_overnight, `:1194-1202`; sliced `:1432`) | **YES** — same combined-then-multiply shape | **YES** — same split needed |
| `m.cost_rebuild_capital_charge_npv` (**f1_tranche** — `optimize_model_sliced.py:1694-1716`, annual mirror `:1480-1502`) | **YES** — `_crf_comm["solar"] * _df_window_comm["solar"][rebuild_t] * (m.solar_public_add[build_t] + m.solar_eaas_add[build_t]) * m.solar_capex_param[rebuild_t]` (`optimize_model_sliced.py:1701-1702` region) | **YES** — and this is the ACTIVE branch (the project's central configuration runs `f1_tranche`), not a superseded one; a distinct EaaS capex param cannot be applied correctly here without splitting this quicksum by channel too |

**Answer: mixed, but not fully separable without restructuring.** The
PRIMARY capex/split-identity/capital-charge mechanism is already
channel-separated (an EaaS-specific capex param could be substituted in
with a small, local, mechanical edit). But the REBUILD mechanism —
`m.salvage_npv` / `m.cost_replacement_capex_npv` under
`pre_f1_overnight`, AND `m.cost_rebuild_capital_charge_npv` under
`f1_tranche` (the active branch) — combines both channels' add-variables
into one sum before multiplying by capex, in all three cases. Applying
`capex[v] * add[v]` correctly per TASK.md's B2 language ("this flows
into capex[v] * add[v] in the split identity") is true for the split
identity itself, but the rebuild mechanism sitting alongside it is a
second, separate combined expression that the same statement does not
cover and that would need its own restructuring.

### (d) `m.eaas_bankability` — does it read solar energy per MW?

**Yes, in both builders, and it currently uses the shared/utility value:**

- `optimize_model.py:759` — `bankable_revenue = (tariff *
  solar_energy_per_mw * m.remaining_npv_factor[t] * m.solar_eaas_add[t])
  / required_margin`, where `solar_energy_per_mw` is the SAME scalar
  from `:684` (`scenario["solar_cf"] * 8760 / 1e6`) used for
  `m.solar_generation`.
- `optimize_model.py:798` — `m.eaas_subsidy_limit`'s
  `max_bankable_at_t = tariff * solar_energy_per_mw * ... `, same shared
  scalar.
- `optimize_model_sliced.py:1068,1083` — same pattern, using
  `solar_energy_per_mw_annual` (`:1022`), itself `scenario["solar_cf"] *
  8760 / 1e6` — the SAME utility CF, just computed as a second,
  slice-independent copy rather than reused from `solar_per_mw`.

**Answer: confirmed exactly as TASK.md warned.** If a distinct, lower
EaaS CF existed, the bankability test would still credit EaaS capacity
with the utility-scale yield unless `solar_energy_per_mw` /
`solar_energy_per_mw_annual` were also redirected to an EaaS-specific
value at these four sites. This is a second mechanism (independent of
generation) that must change together with any new EaaS CF, or DIST-5
would be half-implemented: the physical generation would still be wrong
(per (a)/(b)) AND the financing test would overstate what a real EaaS
asset could bank.

## STOP — B0 finding

**The two channels cannot be given separate CF and seasonal factors
without restructuring the generation expression.** Confirmed by direct
inspection, both builders:

`m.solar_capacity_mw[t]` is ONE cumulative stock summing
`solar_public_add` and `solar_eaas_add` together (annual
`optimize_model.py:737-743`; sliced `optimize_model_sliced.py:698-710})
BEFORE any yield is applied. `m.solar_generation` then multiplies that
single stock by ONE per-MW-yield value derived from ONE `scenario[
"solar_cf"]` (and, sliced-only, ONE `season_solar_factor`). There is no
point in either builder where a public MW and an EaaS MW can be told
apart once `solar_capacity_mw[t]` exists — the channel identity is
already lost by the time generation is computed.

Giving EaaS its own CF (0.164) and season factors (1.015/0.985) per
TASK.md's B2 therefore requires, at minimum:
1. Splitting `m.solar_capacity_mw[t]` into two per-channel cumulative
   stocks (public, EaaS) instead of one combined stock.
2. Splitting `m.solar_generation` into two per-channel generation
   expressions, each using its own CF/seasonal-factor-derived per-MW
   yield, with the existing `m.solar_generation` name re-defined as
   their sum so every downstream consumer that reads it today (gas
   balance is unaffected, but demand balance, emissions, and every
   diagnostic in `extract_planning_diagnostics` and both runners' own
   summary dicts read `m.solar_generation` directly) keeps working
   without a separate inventory-and-rewire pass of its own.
3. Per (c) above: also splitting `m.salvage_npv` /
   `m.cost_replacement_capex_npv` (pre_f1_overnight) and
   `m.cost_rebuild_capital_charge_npv` (f1_tranche, the ACTIVE branch)
   into per-channel sums, so a distinct EaaS capex param can be applied
   correctly in the rebuild mechanism, not only in the primary capex/
   split-identity/capital-charge sites (which, per (c), are already
   separable without restructuring).
4. Per (d): redirecting `m.eaas_bankability` / `m.eaas_subsidy_limit`'s
   `solar_energy_per_mw` (annual) / `solar_energy_per_mw_annual`
   (sliced) to the new EaaS-specific CF at all four sites.

This is a genuine restructuring of the core generation and rebuild
mechanisms in BOTH builders — new per-channel state (Expressions, and
arguably new Vars if the split needs to be a decision-stage rather than
a reporting-stage split), not a parameter or arm addition. Per TASK.md:
**"STOP after B0 and report if the two channels cannot be given separate
CF and seasonal factors without restructuring the generation expression."**
That condition is met. B1, B2, B3, and B4 are **NOT implemented** in
this task.

### What is NOT blocked, for a future attempt

- The capex-premium correction (B2's first item) is mechanically simple
  at the PRIMARY capex sites (public/EaaS CAPEX NPV, F1 split identity
  and capital charge) — those are already channel-separated. It is only
  the rebuild-on-retirement mechanism (all three cost expressions named
  in (c)'s table) that would need to move alongside a generation-side
  restructuring.
- `dist5_case` as a required key (B1) and its wiring discipline (the
  `tariff_case`/register-L17 lesson TASK.md cites) do not depend on the
  generation restructuring — a future attempt could do B1's plumbing
  first, with every runner passing `dist5_none` and every physical
  effect (B2) deferred until the generation split lands. This task does
  NOT do even that much, since TASK.md's instruction is to stop after B0
  entirely, not to partially stage B1.
- Part A's three sourced values are unaffected by this finding and are
  live in every run today, including sliced runs that never touch
  `dist5_case`.

## Verification (bounded) — as it applies here

TASK.md's specified Part-B verification (`build_model_sliced()` under
`dist5_central`, confirming an EaaS CF of 0.164 and dry factor of 1.015
against a public 0.19/1.076) presupposes B1/B2 were implemented. They
were not, per the STOP above, so that specific check cannot be run —
there is no `dist5_case` key and no separate EaaS CF resolution path to
call it against. No `dist5_central`-related code was written, so there
is nothing to falsely claim passes.

What WAS verified: `python3 -m py_compile` on every file this task
actually touched (Part A only — `src/scenarios.py`,
`src/optimize_model_sliced.py`,
`scripts/20_run_cf1_solar_yield_sensitivity.py`), plus the Part A
non-solving `load_scenario()` sanity check above. No solver, no runner,
no smoke suite, no build call, no scratchpad script beyond that one
`load_scenario()` call and the reads performed for B0's inventory.

## Do-NOT checklist

- DIST-2 not implemented (was never reached; B0 stopped before B3).
- `required_margin`, DSCR arms, tariff arms, F1 capital charge structure:
  untouched (Part A touches only `solar_cf`, `season_solar_factor`,
  `hydro_season_factor`; Part B made no code change).
- `hydro_min_flow_fraction`, `period_demand_factor`: untouched, confirmed
  by re-reading the diff — neither key appears in any edit this task
  made.

---

# CF arm exposure — TASK.md premise check, STOP after Step 0

**Correction to TASK.md's premise, stated up front.** The task instructs
"Follow the `--dist5-case` pattern exactly" and asks (Step 3) whether
"DIST-5's EaaS CF is derived by ratio" or "hardcoded to the central
0.164." Neither `--dist5-case` nor any DIST-5 EaaS-CF mechanism exists
anywhere in the codebase — checked, not assumed:
`grep -rn "dist5_case\|solar_cf_case\|solar_cf_scenarios" src/`,
`grep -rn "0.8635\|eaas_cf" src/*.py` — all return nothing except this
document's own filename in a comment. This is expected, not a
regression: the immediately preceding task (Part B of DIST-5, this same
file, "PART B — DIST-5" above) inventoried the generation expression at
B0 and found the two channels cannot be given separate CF/seasonal
factors without restructuring `m.solar_generation` in both builders —
exactly the condition that document's own STOP clause named — and
correctly stopped before implementing `dist5_case`, B2's corrections, or
anything downstream of them. TASK.md for the present task appears to
have been written on the assumption that DIST-5 had shipped; it has not.
There is therefore no `--dist5-case` pattern to follow. The wiring below
instead follows the general required-arm pattern already used
consistently by every other arm in the five/seven runners named
(`--tariff-case`, `--hydro-case`, `--conversion-case`, etc.) — the same
shape `--dist5-case` would have had, mechanically, had it existed.

## Step 0 — Inventory (locate and quote, NO change)

### (a) `solar_cf_scenarios()` or equivalent, and its arm keys

**Does not exist.** `grep -n "solar_cf\|cf_low\|cf_high\|cf_central"
src/scenarios.py` returns exactly one hit:

```
src/scenarios.py:2882:        "solar_cf": 0.19,   # fixed-tilt CF, central.
```

`solar_cf` is a single hardcoded scalar in the scenario dict, not a
named-arm registry function. There is no `solar_cf_scenarios()`,
`cf_scenarios()`, or any dict mapping arm names (`cf_low` / `cf_central`
/ `cf_high`) to values, anywhere in `src/`. Compare
`storage_cc_scenarios()`, `tariff_scenarios()`, `hydro_case_scenarios()`
etc. — all real precedents for this pattern; solar CF has no such
function.

The only place the sourced band `{0.16, 0.19, 0.20}` exists as three
distinct values in one place is
`scripts/20_run_cf1_solar_yield_sensitivity.py`'s `CF_SWEEP` list (a
plain Python list swept by mutating `scenario["solar_cf"]` AFTER
`load_scenario()` returns — the script's own docstring calls this out:
"Convention follows the rest of the codebase: `load_scenario(...)` then
mutate the scenario dict for keys that are not `load_scenario`
arguments"). That is a research sweep on the annual builder, not a named
scenario-key selector — the same "grid-walk vs named arm" distinction
`docs/tariff_arm.md` drew for `04_run_fin2_tariff_sweep.py` /
`TARIFF_SWEEP_GRID` versus `tariff_case`.

### (b) Does `load_scenario()` already accept a `solar_cf_case` parameter?

**No — not even as an optional parameter with a silent default.**
`grep -n "solar_cf_case" src/scenarios.py` returns nothing.
`load_scenario()`'s full signature (checked directly) has no `solar_cf`-
or `cf`-named parameter of any kind. `solar_cf` is set unconditionally to
the literal `0.19` inside the function body, with no parameter
controlling it.

**This is NOT the `solar_tariff_case` defect class named in Step 0(b)'s
prompt.** `solar_tariff_case` (retired by `docs/tariff_arm.md`) WAS a
real parameter of `load_scenario()`, just one with an unsourced silent
default (`"baseline"` → 65,000,000) that every caller silently inherited.
Solar CF has no parameter to have a silent default in the first place —
the value is not selectable at all, by any caller, under any
circumstance. This is a more fundamental gap than L17's process note
describes, not a milder instance of it.

### (c) Every consumer of the resolved CF in both builders; is the DIST-5 EaaS CF derived from it?

**`src/optimize_model.py` (annual):**
- `:684` — `solar_energy_per_mw = scenario["solar_cf"] * 8760 /
  1_000_000` — feeds `m.solar_generation` (`:746-747`).
- `:759` — `eaas_bankability_rule`'s `bankable_revenue`, reads the SAME
  `solar_energy_per_mw`.
- `:798` — `eaas_subsidy_limit`'s `max_bankable_at_t`, same value again.

**`src/optimize_model_sliced.py` (sliced):**
- `:177` — `cf_annual = scenario["solar_cf"]`, inside
  `_solar_energy_per_mw_by_slice`, feeds `m.solar_generation`
  (`:790-793` via `solar_per_mw`).
- `:1022` — `solar_energy_per_mw_annual = scenario["solar_cf"] * 8760 /
  1_000_000` — a SEPARATE, slice-independent copy of the same CF,
  computed again for the bankability test only.
- `:1073`, `:1088` — `eaas_bankability_rule` / `eaas_subsidy_limit`, both
  reading `solar_energy_per_mw_annual`.

(Outside the two builders, for completeness though not asked: the
diagnostics/experiments layer reads `scenario["solar_cf"]` independently
at four more sites — `src/optimize_experiments.py:278, 878, 2046, 2354`
— all downstream reporting, not generation.)

**Is the DIST-5 EaaS CF derived from this by the 0.8635 ratio, or set
independently? Neither — it does not exist.** `grep -rn "0.8635\|eaas_cf"
src/*.py` returns nothing. DIST-5 (this file's Part B) was never
implemented past its Step 0 inventory; there is no EaaS-specific CF
anywhere in the codebase, derived or hardcoded, to check. Step 3 below
is therefore inapplicable — not "confirmed derived," not "found
hardcoded," but not present at all.

### (d) Does `m.eaas_bankability` read the CF?

**Yes, confirmed at all four sites listed in (c)'s bankability rows**
(`optimize_model.py:759,798`; `optimize_model_sliced.py:1073,1088`),
and in every case it reads the SAME single CF-derived scalar used for
ordinary generation (`solar_energy_per_mw` / `solar_energy_per_mw_annual`)
— there is no separate bankability-specific CF read anywhere.

## STOP — the CF is not currently a scenario key at all

Per TASK.md: **"STOP after step 0 and report if the CF is not currently
a scenario key at all — adding one is a different and larger change than
exposing an existing one."** Step 0(a)/(b) establish exactly that: there
is no `solar_cf_case` parameter, no `solar_cf_scenarios()` registry, and
no arm-selection mechanism of any kind. `solar_cf` is a bare literal.

"Exposing an existing arm" (what Steps 1-3 assume) and "introducing a
new required arm where none exists" (what would actually be needed here)
are different tasks with different risk profiles — the latter is exactly
the class of change `docs/tariff_arm.md` and this file's own Part B
treated as substantial enough to warrant a full Step-0-style inventory
and sourcing pass before touching any runner, not a mechanical
CLI-exposure pass. Steps 1-3 (adding `--solar-cf-case`, threading a new
required `solar_cf_case` through every `load_scenario()` caller per the
`tariff_case`/L17 lesson, appending it to seven output tags, and fixing
a DIST-5 ratio derivation that does not exist to fix) are **NOT
implemented** in this task.

### What a future attempt would need, since none of it is a secret

1. A `solar_cf_scenarios()` registry (mirroring `storage_cc_scenarios()`
   etc.): `{"cf_low": 0.16, "cf_central": 0.19, "cf_high": 0.20}`,
   sourced comment pointing at `src/scenarios.py`'s existing `solar_cf`
   provenance block (Part A of this file).
2. A required, no-default `solar_cf_case` parameter on `load_scenario()`
   (a `_SOLAR_CF_REQUIRED` sentinel, mirroring `_TARIFF_REQUIRED` /
   `_HYDRO_REQUIRED`), validated and written to `scenario["solar_cf"]`.
3. Threading `solar_cf_case` into EVERY `load_scenario()` caller, not
   only the seven live ones named in TASK.md's Step 1 — the
   `tariff_case` incident (repaired in `docs/tariff_arm.md`) and the
   `dist5_case` near-miss TASK.md itself cites (which never actually
   occurred, since `dist5_case` was never made required — see the
   premise correction above) are both instances of exactly this failure
   mode. A fresh inventory of all `load_scenario()` call sites (like
   `docs/tariff_arm.md`'s Step 2) would be needed to classify each as
   live/dead before wiring.
4. `--solar-cf-case` on the seven named runners, `choices` sourced live
   from `solar_cf_scenarios()`, appended to the output tag on all seven
   (this arm is never inert, unlike `tariff_case` on non-EaaS runners).
5. Only once 1-4 exist would Step 3's question ("is the DIST-5 EaaS CF
   derived by ratio from the selected arm") become answerable — and only
   once DIST-5's own generation-expression restructuring (this file's
   Part B STOP) is separately resolved, since there is no EaaS CF of any
   kind to derive yet.

## Verification (bounded) — as it applies here

No file was touched (Step 0 is inventory-only, and the STOP condition
was met before Step 1). There is nothing to `py_compile` and no
`build_model_sliced()` call to make — TASK.md's specified verification
(`--solar-cf-case cf_low` resolving to utility CF 0.16 and EaaS CF
0.138) presupposes Steps 1-3 were implemented, which they were not, for
the reason above. Confirmed by re-reading: `git status` shows no source
file modified by this task; only this report was written.
