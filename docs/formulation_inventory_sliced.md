# Formulation inventory — `build_model_sliced` at HEAD

**Purpose.** Exact, verbatim inventory of every Pyomo component
`src/optimize_model_sliced.py`'s `build_model_sliced()` constructs, plus
every scenario/econ key it reads, for Chapter 3.1's algebra. Read-only:
nothing in `src/` or `scripts/` was edited or run to produce this.

**Source.** `src/optimize_model_sliced.py`, HEAD `17520ae`, function
`build_model_sliced()` spanning lines 221–2039 (the file is 2159 lines
total; lines 2042–2159 are post-solve aggregation/diagnostic helper
functions — `annual_totals`, `firm_capacity_diagnostic`, `slice_detail`,
`conversion_diagnostic` — that read an already-built model and declare
no new Pyomo components; not part of this inventory).

**Method.** Read the function in full, source order, top to bottom.
Every `scenario[...]`, `scenario.get(...)`, `econ[...]`, `econ.get(...)`
occurrence inside the function was located by an exhaustive line-range
grep (`awk 'NR==221,NR==2039'` piped to `grep`) and cross-checked
against a manual read, so the parameter tables below are a complete
account, not a sample. Line numbers are absolute (this file's own
numbering), not relative to the function.

**Notation.** `T` and `S`/`m.S` (defined in §1) are used throughout as
Pyomo index sets. `_conv` is a local boolean (`bool(scenario.get(
"conversion_available", False))`, line 296) gating the entire OCGT→CCGT
conversion mechanism. `capital_treatment_case` (`"pre_f1_overnight"` or
`"f1_tranche"`) is the other major branch: the live objective contains
exactly one of {F1 tranche capital charge, overnight capex + salvage +
replacement}, never both.

---

## Section 1 — Sets and indices

| name | kind | content | line |
|---|---|---|---|
| `T` | plain Python `range`, not a Pyomo `Set` object | `T = range(len(years))` where `years = scenario["years"]`. Used as the year index on nearly every `pyo.Param`/`pyo.Var`/`pyo.Expression`/`pyo.Constraint` in the model. Never assigned to the model as `m.T`. | 279–280 |
| `SLICES` (module level) | plain Python list of 2-tuples | `SLICES = [(s, p) for s in ("dry", "wet") for p in ("night", "day", "peak")]` — 6 elements: `(dry,night) (dry,day) (dry,peak) (wet,night) (wet,day) (wet,peak)` | 64 |
| `S` (local var inside `build_model_sliced`) | alias | `S = SLICES` | 281 |
| `m.S` | `pyo.Set` | `m.S = pyo.Set(initialize=S, dimen=2, ordered=True)` — the season×period pair, `dimen=2`. Declared partway through the function (after the financing-guard block, immediately before the sliced operational Vars), so every Pyomo component built *before* line 629 that needs a slice-like index instead uses the plain list `["dry", "wet"]` inline (see next row), and everything after line 629 uses `m.S`. | 629 |
| implicit `["dry", "wet"]` | plain Python list literal, not a named Pyomo `Set` | Used directly as the second index argument to two constraints: `m.hydro_season_budget = pyo.Constraint(T, ["dry", "wet"], rule=...)` and `m.storage_season_balance = pyo.Constraint(T, ["dry", "wet"], rule=...)`. Never named `m.seasons` or similar. | 956 (`hydro_season_budget`), 994 (`storage_season_balance`) |
| `_CONV_KEYS` | local Python tuple | `_CONV_KEYS = ("A", "B", "C")` — the three OCGT→CCGT conversion tiers. Only created **inside `if _conv:`**, so it does not exist at all when `conversion_available` is falsy. Used as a Pyomo index set for `m.convert_mw`, `m.convert_to_power`, `m.convert_comm_vol`, and as a plain iteration tuple in several `pyo.quicksum` calls. | 530 |

**Slice-hours structure** (module level, not Pyomo objects, but the data
`m.S` is built over):

```
SEASON_DAYS = {"dry": 181, "wet": 184}                       # line 62
PERIOD_HOURS = {"night": 7, "day": 13, "peak": 4}             # line 63
```

`slice_hours()` (module function, line 69–75) returns `H = {(s,p):
SEASON_DAYS[s]*PERIOD_HOURS[p] for (s,p) in SLICES}`, asserting the sum
equals 8760. Inside `build_model_sliced`, `H = slice_hours()` at line
282, and `m.slice_hours = dict(H)` (line 284) — **a plain Python dict
attribute on the model, not a `pyo.Param`**, kept only "for
diagnostics" per its own comment.

**No `RangeSet` appears anywhere in this file** (`grep -n "RangeSet"
src/optimize_model_sliced.py` — no matches). `T` (a plain `range`) plays
that role informally.

---

## Section 2 — Parameters

### 2.1 Actual `pyo.Param` objects

Only three exist. Everything else load-bearing in the algebra is a
plain Python value/dict closed over by rule lambdas (catalogued in
§2.2–§2.4) — **not** a Pyomo `Param`.

| name | kind | index | line | initialize / within | source of data |
|---|---|---|---|---|---|
| `m.remaining_npv_factor` | `Param` | `T` | 474–475 | `initialize=remaining_npv_priv, within=pyo.PositiveReals` | `remaining_npv_priv = {t: sum(1/(1+r_private)**(k-t) for k in range(t,len(T))) for t in T}` (470–473), where `r_private = resolve_private_rate(scenario)` (469, external function in `src/blended_finance.py` — its own internal scenario reads are not inventoried here, out of this file's scope) |
| `m.DF` | `Param` | `T` | 476 | `initialize=df, within=pyo.PositiveReals` | `df = {t: 1.0/((1+r_social)**t) for t in T}` (468), `r_social = float(scenario.get("social_discount_rate", 0.08))` (467) |
| `m.solar_capex_param` | `Param` | `T` | 485–486 | `initialize=capex_init, within=pyo.PositiveReals` | `capex_init = {t: float(solar_capex_by_year[int(years[t])]) for t in T}` if the `solar_capex_by_year` **function argument** is not `None` (482), else `{t: float(econ["SOLAR_CAPEX_PER_MW"]) for t in T}` (484) |

### 2.2 Builder call parameters (function signature — not scenario/econ keys)

Not scenario/econ dict keys, but they are parameters of the algebra in
every sense that matters (they gate which constraints/expressions are
built and how variables are bounded). Full signature, line 221–234:

| name | default | role |
|---|---|---|
| `scenario` | (positional, required) | the scenario dict |
| `econ` | (positional, required) | the econ dict |
| `emissions_cap` | `None` | scalar cumulative emissions cap (§5/§6, deferred) |
| `emissions_cap_by_year` | `None` | per-year emissions cap list (§5/§6, deferred) |
| `max_offgrid_fraction` | `None` | grid-service-obligation bound; scalar or per-year sequence |
| `grid_service_mode` | `"annual"` | `"annual"` or `"total"` — how `max_offgrid_fraction` is applied |
| `reliability_max_unserved_fraction` | `None` | **DEPRECATED alias** for `max_offgrid_fraction`; if not `None`, raises if `max_offgrid_fraction` is *also* set (258–264), else overwrites it |
| `reliability_mode` | `None` | **DEPRECATED alias** for `grid_service_mode` (265–266) |
| `selfgen_in_emissions_cap` | `False` | switches `m.emissions_capped_by_year` between grid-only and total (§4/§6, deferred) |
| `fixed_plan` | `None` | if not `None`, `.fix()`es `solar_public_add`, `solar_eaas_add`, `storage_add` to supplied values for every `t` (620–624) |
| `solar_capex_by_year` | `None` | overrides `econ["SOLAR_CAPEX_PER_MW"]` with a per-year trajectory (481–486) |
| `allow_unserved` | `None` (auto) | **see §3 — governs whether `m.unserved` is fixed at zero** |

### 2.3 Scenario keys read directly inside `build_model_sliced` (and its three private helpers)

`.get(key, DEFAULT)` calls are flagged **SILENT DEFAULT** where `DEFAULT`
is an actual fallback value that lets execution continue; flagged
**GUARDED (raises)** where the `.get(key, None)` is immediately checked
and a `ValueError` raised if `None`/missing — functionally a required
key, not a silent default, even though the literal call is `.get(...)`.
Plain `scenario["key"]` subscripts (no default at all — `KeyError` if
absent) are flagged **REQUIRED (subscript)**.

| key | access | line(s) | note |
|---|---|---|---|
| `years` | REQUIRED (subscript) | 279 | |
| `peak_multiple_case` | GUARDED (raises) | 295 | presence-only guard; value itself never read again in this function (consumed inside `load_scenario()` to derive `peak_mw`) |
| `peak_mw` | GUARDED (raises) | 302; length-checked 310–314 | |
| `reserve_margin` | GUARDED (raises) | 315; length-checked 1246–1250 if list/tuple | scalar or per-year sequence, see §5 firm-capacity constraint |
| `capacity_credit` | GUARDED (raises) | 322; sub-keys `solar`/`gas`/`storage`/`hydro` checked present 329–333 | dict, read again at 1298–1301/1313–1316 (firm capacity) |
| `storage_cc_case` | GUARDED (raises) | 338 | presence-only guard |
| `gas_build_case` | GUARDED (raises), then REQUIRED (subscript) | guard 351; subscript read 617 | value drives the `gas_add` fix/cap branch, §3 |
| `gas_availability_case` | GUARDED (raises) | 363 | presence-only guard |
| `gas_retirement_case` | GUARDED (raises) | 376 | presence-only guard |
| `gas_capex_case` | GUARDED (raises) | 388 | presence-only guard |
| `ccgt_efficiency_case` | GUARDED (raises) | 402 | presence-only guard |
| `capital_recovery_case` | GUARDED (raises) | 411 | presence-only guard |
| `dscr_case` | GUARDED (raises) | 422 | presence-only guard |
| `us_inflation` | GUARDED (raises, via `not in` / `is None`) | 431; REQUIRED (subscript) at 1561 | checked by key presence, not truthiness (0.0 is legitimate) |
| `capital_treatment_case` | GUARDED (raises), then REQUIRED (subscript) | guard 442, value-membership check 448–451; subscript 506 | must be `"pre_f1_overnight"` or `"f1_tranche"` |
| `social_discount_rate` | **SILENT DEFAULT 0.08** | 467 | |
| `gas_eta` | GUARDED (raises) | 491 | assigned to local `eta` |
| `storage_round_trip_eff` | **SILENT DEFAULT 0.90** | 495 | range-checked `(0,1]` at 594 |
| `storage_duration_hours` | REQUIRED (subscript) | 496 | |
| `solar_service_tariff_usd_per_twh` | **SILENT DEFAULT `None`** (not a raise — `None` is legitimate under non-EaaS regimes) | 497 | assigned to local `tariff`; raises only if `financing_regime == "eaas"` and `tariff is None` (1059–1060) |
| `financing_regime` | **SILENT DEFAULT `"traditional"`** | 498 | |
| `required_margin` | REQUIRED (subscript) | 504 | |
| `conversion_available` | **SILENT DEFAULT `False`** | 516 | `_conv = bool(scenario.get("conversion_available", False))` — single occurrence, confirmed by `grep -n "conversion_available" src/optimize_model_sliced.py` |
| `conversion_tiers` | REQUIRED (subscript, only if `_conv`) | 529 | dict `{"A"/"B"/"C": {"nameplate_mw":..., "paf":...}}` |
| `conversion_uplift` | REQUIRED (subscript, only if `_conv`) | 531 | |
| `conversion_retrofit_efficiency` | REQUIRED (subscript, only if `_conv`) | 532; range-checked `(0,1]` at 534 | |
| `conversion_capex_per_kw` | REQUIRED (subscript, only if `_conv`) | 533 | |
| `selfgen_busbar_factor` | **SILENT DEFAULT 1.0** | 563 | must be `>= 1.0` (564–568) |
| `gas_availability_factor` | GUARDED (raises) | 597; range-checked `(0,1]` at 604 | |
| `gas_baseline_mw` | REQUIRED (subscript) | 671 | |
| `gas_baseline_retirement_start_year` | **SILENT DEFAULT `None`** (legitimate — means "no retirement schedule active" in the flat-rate branch) | 672 | |
| `gas_baseline_retirement_mw_per_year` | **SILENT DEFAULT 0.0** | 673 | |
| `gas_baseline_retirement_series_mw` | **SILENT DEFAULT `None`** (legitimate — `None` selects the flat-rate formula instead) | 682; length-checked 683–688 | per-model-year cumulative retired-MW list |
| `gas_new_capacity_cap_mw` | GUARDED (raises), only reached under `gas_build_case == "gas_ndc_17gw"` | 713 | |
| `solar_baseline_mw` | REQUIRED (subscript) | 738 | |
| `base_demand_twh` | REQUIRED (subscript) | 751 | passed to `project_baseline_demand()` (`src/demand.py`, external — not inventoried here) |
| `demand_growth` | REQUIRED (subscript) | 752 | passed to `project_baseline_demand()` |
| `gas_scenario` | REQUIRED (subscript) | 760 | passed to `gas_available_power()` (`src/gas_supply.py`, external) |
| `hydro_energy_twh_by_year` | GUARDED (raises) | 768; length-checked 785–787 | |
| `hydro_capacity_mw_by_year` | GUARDED (raises) | 776; length-checked 785–787 | |
| `hydro_min_flow_fraction` | **SILENT DEFAULT `DEFAULT_HYDRO_MIN_FLOW_FRACTION` = 0.50** | 794–795 | module constant, line 147 |
| `season_demand_factor` | REQUIRED (subscript) | 164 (inside `_normalised_demand_weights`) | |
| `period_demand_factor` | REQUIRED (subscript) | 165 (inside `_normalised_demand_weights`) | |
| `solar_cf` | REQUIRED (subscript) | 184 (inside `_solar_energy_per_mw_by_slice`); also 1057 (`solar_energy_per_mw_annual = scenario["solar_cf"] * 8760 / 1_000_000`) | |
| `season_solar_factor` | **SILENT DEFAULT `DEFAULT_SEASON_SOLAR_FACTOR` = `{"dry": 1.076, "wet": 0.925}`** | 185 (inside `_solar_energy_per_mw_by_slice`) | module constant, line 107; comment states `load_scenario()` always sets the key so this default "is not exercised in normal operation" |
| `hydro_season_factor` | **SILENT DEFAULT `DEFAULT_HYDRO_SEASON_FACTOR` = `{"dry": 1.045, "wet": 0.956}`** | 209 (inside `_hydro_season_energy_share`) | module constant, line 138; same "not exercised in normal operation" comment |
| `new_gas_deliverability` | **SILENT DEFAULT `"constrained"`** | 894; membership-checked against `("constrained","exempt")` at 895–896 | |
| `gas_new_efficiency` | REQUIRED (subscript, only if `new_gas_deliverability == "exempt"`) | 904 | |
| `land_available_km2` | REQUIRED (subscript) | 1135 | |
| `land_intensity_solar_km2_per_mw` | REQUIRED (subscript) | 1136 | |
| `land_intensity_gas_km2_per_mw` | REQUIRED (subscript) | 1137 | |
| `land_intensity_storage_km2_per_mwh` | REQUIRED (subscript) | 1138 | |
| `solar_cumulative_cap_mw` | **SILENT DEFAULT `None`** (legitimate — `None` means no cap built at all) | 1159; length-checked 1161–1165 | |
| `solar_min_build_mw_per_year` | **SILENT DEFAULT 0.0** | 1176 | constraint only built if `> 0` (1177) |
| `within_block_peaking_factor` | **SILENT DEFAULT 1.3448** | 1262–1264 | flagged elsewhere (register L50 / forensic audit A62) as a hardcoded literal with no `load_scenario()` entry, no arm, no output signature; the adjacent comment (1258–1261) claims the default is "1.0," which the code contradicts |
| `gas_capex_per_mw` | REQUIRED (subscript) | 1396 (`pre_f1_overnight` gas capex), 1476/1496 (salvage), 1525 (replacement), 1686 (`gas_split_identity`, `f1_tranche`), 1798 (`f1_tranche` rebuild) | |
| `om_cost_by_year` | REQUIRED (subscript) | 1381 | passed to `compute_om_cost_npv()` (`src/io.py`, external) |
| `asset_lifetimes` | GUARDED (raises), read **twice independently** | `pre_f1_overnight` branch: 1412 (guard), sub-keys `solar`/`storage`/`gas` checked 1419–1424; `f1_tranche` branch: 1581 (guard, local name `_lives`), same sub-key check 1588–1594 | the two reads are deliberately separate local variables (`lives` vs `_lives`) — the comment states the `pre_f1_overnight` copy is the regression instrument and must not be touched |
| `L_conc` | REQUIRED (subscript, `f1_tranche` only) | 1557 | |
| `L_comm` | REQUIRED (subscript, `f1_tranche` only) | 1558 | |
| `r_conc_nominal` | REQUIRED (subscript, `f1_tranche` only) | 1559 | |
| `r_comm_nominal` | REQUIRED (subscript, `f1_tranche` only) | 1560 | |
| `concessional_envelope_usd` | REQUIRED (subscript, `f1_tranche` only) | 1711 | comment states this key is itself required-no-default at `load_scenario()`, so no `if ... is not None` guard is needed here |
| `carbon_active` | **SILENT DEFAULT `False`** | 1912 | |
| `carbon_price` | REQUIRED (subscript, only if `carbon_active`) | 1913 | |
| `public_solar_budget_npv` | **SILENT DEFAULT `None`** (legitimate — `None` means no public-budget constraint is built at all) | 2003 | |

### 2.4 `econ` keys read directly inside `build_model_sliced`

| key | access | line(s) | note |
|---|---|---|---|
| `GENSET_CASE` | GUARDED (raises via `not in`) | 455–462 | presence-only; the value itself is never read in this function — its presence certifies `econ` was built via `load_econ()` |
| `SOLAR_CAPEX_PER_MW` | REQUIRED (subscript, only if `solar_capex_by_year` arg is `None`) | 484 | |
| `UNSERVED_ENERGY_PENALTY` | REQUIRED (subscript) | 546 (assigned to local `voll`), 1841 (`unserved_npv`) | |
| `GENSET_COST_PER_TWH_E` | **SILENT DEFAULT = `voll`** (i.e. defaults to VoLL if genset pricing isn't loaded) | 551 | `selfgen_price = econ.get("GENSET_COST_PER_TWH_E", voll)`; **presence of this key is also what `allow_unserved`'s auto-default keys off** (591–592, see §3) |
| `GENSET_ETA_HHV` | checked via `"GENSET_ETA_HHV" in econ` | 552–556 | if present, `selfgen_ef_e = econ["GENSET_EF_TH_HHV"] / econ["GENSET_ETA_HHV"]`; else `selfgen_ef_e = 0.0` |
| `GENSET_EF_TH_HHV` | REQUIRED (subscript, only if `GENSET_ETA_HHV` present) | 554 | |
| `GAS_COST_PER_TWH_TH` | REQUIRED (subscript) | 1327 (`gas_opex_npv`), 1340 (`_conv` addendum to `gas_opex_npv`) | |
| `STORAGE_COST_PER_MWH` | REQUIRED (subscript) | 1399, 1471, 1490, 1520, 1694, 1792, 2009 (seven sites: `pre_f1_overnight` capex/salvage/replacement, `f1_tranche` split-identity/rebuild, public budget) | |
| `EF_TCO2_PER_MWH_TH` | REQUIRED (subscript) | 1850 (`ef_e`), 1854 (`_ef_e_retrofit`, only if `_conv`) | |

**No `RangeSet`, no other `pyo.Param`, and no scenario/econ key besides
the ones tabulated above is read anywhere inside `build_model_sliced`**
(cross-checked by the exhaustive line-range grep described in the
Method note).

---

## Section 3 — Decision variables

All 23 `pyo.Var` declarations in the function, in source order. All are
`domain=pyo.NonNegativeReals` except where noted; none carry an upper
bound except `convert_mw`.

| name | index | domain / bounds | line | conditional? | fix / notes |
|---|---|---|---|---|---|
| `m.solar_public_add` | `T` | `NonNegativeReals` | 610 | no | fixed to `fixed_plan["solar_public_add"][t]` for every `t` if `fixed_plan is not None` (620–624) |
| `m.solar_eaas_add` | `T` | `NonNegativeReals` | 611 | no | fixed to `fixed_plan["solar_eaas_add"][t]` under the same condition |
| `m.eaas_subsidy` | `T` | `NonNegativeReals` | 612 | no | never fixed |
| `m.gas_add` | `T` | `NonNegativeReals` | 616 | no | **This is a `pyo.Var` (free decision variable) at HEAD, not a `pyo.Param`.** Its treatment depends on `gas_build_case` (`scenario["gas_build_case"]`, read at 617): `"gas_none"` → `m.gas_add[t].fix(0.0)` for every `t` (709–711); `"gas_ndc_17gw"` → left free but bounded by a separate `pyo.Constraint` (`m.gas_new_capacity_cap_constraint`, on cumulative `m.gas_capacity_mw`, not on `gas_add` itself) (712–726); `"gas_unconstrained"` → left completely free (727–728, `pass`) |
| `m.storage_add` | `T` | `NonNegativeReals` | 618 | no | fixed to `fixed_plan["storage_add"][t]` under the same `fixed_plan is not None` condition as above |
| `m.unserved` | `T, m.S` | `NonNegativeReals` | 630 | no | **Explicit task question: status under a priced genset case.** `m.unserved[t, s].fix(0.0)` for every `(t,s)` is applied at 637–640 whenever `not allow_unserved`. `allow_unserved` defaults (when the function argument is `None`, its default) to `"GENSET_COST_PER_TWH_E" not in econ` (591–592) — i.e. **whenever `econ` carries a priced genset cost (`GENSET_COST_PER_TWH_E`), `allow_unserved` is `False` by default and `unserved` is fixed at exactly zero for every slice and year.** It is left free only if the caller passes `allow_unserved=True` explicitly (the file's own comment names `scripts/probe_2_6_structure.py` as the one caller that does) |
| `m.selfgen` | `T, m.S` | `NonNegativeReals` | 636 | no | never fixed; deliberately unbounded above (the file's own comment: this is what lets it absorb the shortfall `unserved` cannot) |
| `m.gas_to_power` | `T, m.S` | `NonNegativeReals` | 642 | no | never fixed |
| `m.gas_curtail` | `T` (**not** indexed by `m.S`) | `NonNegativeReals` | 643 | no | annual only, despite sitting textually among the sliced Vars |
| `m.storage_discharge` | `T, m.S` | `NonNegativeReals` | 644 | no | never fixed |
| `m.storage_charge` | `T, m.S` | `NonNegativeReals` | 645 | no | never fixed |
| `m.hydro_gen` | `T, m.S` | `NonNegativeReals` | 646 | no | never fixed |
| `m.convert_mw` | `_CONV_KEYS` | `NonNegativeReals`, `bounds=lambda m, tier: (0.0, _conv_tiers[tier]["nameplate_mw"])` | 659–663 | **yes — only if `_conv`** (`conversion_available` truthy) | one-off (time-invariant) decision: MW of OCGT nameplate converted per tier, upper-bounded by that tier's nameplate from `scenario["conversion_tiers"][tier]["nameplate_mw"]` |
| `m.convert_to_power` | `_CONV_KEYS, T, m.S` | `NonNegativeReals` | 664–666 | **yes — only if `_conv`** | thermal dispatch (TWh_th) of the converted plant, slice-resolved |
| `m.solar_public_conc_vol` | `T` | `NonNegativeReals` | 1655 | **yes — only if `capital_treatment_case == "f1_tranche"`** | concessional-tranche USD volume of public solar capex at vintage `t`; tied to `m.solar_public_comm_vol` by `m.solar_public_split_identity` (§5, deferred) |
| `m.solar_public_comm_vol` | `T` | `NonNegativeReals` | 1656 | yes — `f1_tranche` only | commercial-tranche counterpart |
| `m.solar_eaas_conc_vol` | `T` | `NonNegativeReals` | 1663 | yes — `f1_tranche` only | concessional-tranche USD volume of EaaS solar capex |
| `m.solar_eaas_comm_vol` | `T` | `NonNegativeReals` | 1664 | yes — `f1_tranche` only | commercial-tranche counterpart |
| `m.gas_conc_vol` | `T` | `NonNegativeReals` | 1679 | yes — `f1_tranche` only | **fixed to `0.0` for every `t`** unconditionally within the `f1_tranche` branch (1681–1682) — "gas is COMMERCIAL-ONLY" (envelope resolution Decision 1) |
| `m.gas_comm_vol` | `T` | `NonNegativeReals` | 1680 | yes — `f1_tranche` only | forced (via `m.gas_split_identity`, §5) to equal all of `gas_capex_per_mw * gas_add[t]` since `gas_conc_vol` is fixed at 0 |
| `m.storage_conc_vol` | `T` | `NonNegativeReals` | 1689 | yes — `f1_tranche` only | |
| `m.storage_comm_vol` | `T` | `NonNegativeReals` | 1690 | yes — `f1_tranche` only | |
| `m.convert_comm_vol` | `_CONV_KEYS` | `NonNegativeReals` | 1821 | **yes — `f1_tranche` AND `_conv` both true (nested)** | conversion capex is commercial-only by the same Decision-1 reasoning as gas; no `convert_conc_vol` counterpart exists at all (not merely fixed to zero — it is never declared) |

**Count check:** 23 `pyo.Var(` declarations located by
`grep -n "pyo\.Var(" src/optimize_model_sliced.py`, all 23 accounted
for above (14 unconditional, 2 conditional on `_conv` alone, 6
conditional on `f1_tranche` alone, 1 conditional on both).

**No other Pyomo component in this function fixes a decision variable**
except the five `.fix()` call sites named above:
`solar_public_add`/`solar_eaas_add`/`storage_add` under `fixed_plan`
(620–624), `unserved` under `not allow_unserved` (637–640), and
`gas_conc_vol` under `f1_tranche` (unconditional within that branch,
1681–1682) and `gas_add` under `gas_build_case == "gas_none"`
(709–711).

---

---

## Default-reachability audit (requested after §1–3 review)

For every `.get(key, DEFAULT)` in §2.3/§2.4 flagged **SILENT DEFAULT**,
checked directly against `load_scenario()` (`src/scenarios.py:2153-3350`)
and, for the one `econ`-side default, `load_econ()` (`src/io.py:338-479`).
**No default was changed. Record only.**

Method: located every dict-literal key assignment inside `load_scenario()`'s
returned `scenario = {...}` (a single flat dict literal plus one `**`
spread), by direct read of the function body, not by inference from the
docstring. "ALWAYS" means the key is present in the dict literal
unconditionally (no `if` wraps it) — the value can still be `None` or a
case-dependent number, but the *key* is never absent. "CONDITIONALLY"
means an `if` gates whether the key is written. "NEVER" means no
assignment exists anywhere in the function.

### `social_discount_rate` first, as requested — the objective's discount rate

**Value: `0.08`. Set unconditionally, `scenarios.py:3206`:
`"social_discount_rate": 0.08,   # objective DF (society's time preference)`.**
It is a bare literal with no named arm, no sourcing comment beyond the
one inline clause quoted, and no sweep found anywhere in
`scripts/run_central_sweep.sh` (`grep -n "social_discount_rate"
scripts/run_central_sweep.sh` — no match). Because `load_scenario()`
always writes this key, `build_model_sliced`'s own
`scenario.get("social_discount_rate", 0.08)` (line 467) never falls
back to its own literal in any run that goes through `load_scenario()`
— the two 0.08s are identical, so the distinction is moot in practice,
but the reachable value comes from `load_scenario()`'s literal, not the
builder's. **Classification: DEAD** (as a fallback) but note the
*value* itself is a single, unsourced, unswept literal reappearing at
two independent sites in the codebase, not a genuinely load-bearing
"default" in the sense the other rows below are.

### Scenario-side defaults (§2.3), checked against `load_scenario()`

| key | builder default | set by `load_scenario()`? | line(s) | value it sets | classification |
|---|---|---|---|---|---|
| `storage_round_trip_eff` | `0.90` | ALWAYS | 3185 | `0.90` — flat literal, identical to the builder default | DEAD |
| `solar_service_tariff_usd_per_twh` | `None` | ALWAYS | 3202–3203 | `tariff_scenarios()[tariff_case]` (`tariff_case` is itself REQUIRED, no default) | DEAD |
| `financing_regime` | `"traditional"` | ALWAYS | 3199 | `"traditional"` — flat literal, identical | DEAD |
| `conversion_available` | `False` | ALWAYS | 3315 | `_conversion_arm["available"]`, derived from `conversion_case` (REQUIRED, no default) | DEAD |
| `selfgen_busbar_factor` | `1.0` | **NEVER** | — | not assigned anywhere in `load_scenario()`; confirmed by `grep -n "selfgen_busbar_factor" src/scenarios.py` — no hits at all in that file | **LIVE in most runs**, with two named exceptions that override it directly on the dict `load_scenario()` returns: `scripts/24_run_leakage_arm.py:288` (`sc["selfgen_busbar_factor"] = 1/(1-0.07)`, crediting the avoided TLF) and `scripts/probe_2_6_structure.py:163` (CLI-controlled diagnostic probe). Every other runner receives the builder's `1.0` |
| `gas_baseline_retirement_start_year` | `None` | ALWAYS | 2967 | `_gas_retirement_start_year`, derived from `gas_retirement_case` — non-`None` under a FLAT retirement arm (`gas_retire_none`/`gas_retire_proxy`), `None` under a VINTAGE arm | DEAD |
| `gas_baseline_retirement_mw_per_year` | `0.0` | ALWAYS | 2968 | `_gas_retirement_mw_per_year`, same source | DEAD |
| `gas_baseline_retirement_series_mw` | `None` | ALWAYS | 2969 | `_gas_retirement_series_mw` — non-`None` under a VINTAGE arm, `None` under FLAT (the two retirement keys are deliberately complementary, per the comment at 2957–2965) | DEAD |
| `hydro_min_flow_fraction` | `0.50` (`DEFAULT_HYDRO_MIN_FLOW_FRACTION`) | ALWAYS | 3174 | `0.50` — flat literal, identical | DEAD |
| `season_solar_factor` | `{"dry":1.076,"wet":0.925}` (`DEFAULT_SEASON_SOLAR_FACTOR`) | ALWAYS | 3114 | same flat literal, identical | DEAD |
| `hydro_season_factor` | `{"dry":1.045,"wet":0.956}` (`DEFAULT_HYDRO_SEASON_FACTOR`) | ALWAYS | 3140 | same flat literal, identical. A second, separate copy of the identical literal (not this key — a hand-built local dict) feeds `capacity_credit["hydro"]`'s computation at line 2919, before the scenario dict exists; the two are commented "KEEP IN SYNC" (claims ledger C-6) | DEAD |
| `new_gas_deliverability` | `"constrained"` | ALWAYS | 3305 | the `load_scenario(new_gas_deliverability=...)` **function argument itself** — an ordinary Python default of `"constrained"` in the signature (line 2179), not a sentinel. Two independent layers carry the identical default value | DEAD (the builder's own `.get()` default is reached only if a caller bypasses `load_scenario()` entirely) |
| `solar_cumulative_cap_mw` | `None` | ALWAYS | 3052 | `_solar_cumulative_cap(solar_build_case, years)` — returns `None` specifically when `solar_build_scenarios()[solar_build_case]` is `None` (the `"deployment_unconstrained"` arm), a real per-year cap list otherwise | DEAD |
| `solar_min_build_mw_per_year` | `0.0` | ALWAYS | 3055 | `0.0` — flat literal, identical | DEAD |
| `within_block_peaking_factor` | `1.3448` | **NEVER** | — | not assigned anywhere in `load_scenario()`, and `grep -rn "within_block_peaking_factor" src/ scripts/` finds no assignment in any runner either — every non-builder hit is a comment | **LIVE in every single run**, unconditionally (register L50 / forensic audit A62) |
| `carbon_active` | `False` | ALWAYS | 3322, via `**carbon_policy_scenarios()[carbon_case]` (dict-spread) | `False` under `carbon_case="no_policy"` (the `load_scenario()` default); `True` under `"moderate_policy"`/`"stringent_policy"` | DEAD |
| `public_solar_budget_npv` | `None` | ALWAYS | 3176 | `capital_envelope_scenarios()[capital_case]` — a real NPV (e.g. 8,732,342,068 at the `"moderate"` default) for `tight`/`moderate`/`adequacy`/`expansion`, `None` specifically under `capital_case="unconstrained"` | DEAD |

### `econ`-side default (§2.4) — `load_scenario()` never touches `econ`; checked against `load_econ()` (`src/io.py:338-479`)

| key | builder default | set by `load_econ()`? | line(s) | value it sets | classification |
|---|---|---|---|---|---|
| `GENSET_COST_PER_TWH_E` | `= voll` (i.e. `econ["UNSERVED_ENERGY_PENALTY"]`) | **CONDITIONALLY** — only inside `if genset_case is not None:` | guard 452; assignment 462 (`econ["GENSET_COST_PER_TWH_E"] = to_float(g_row["total_usd_per_twh_e"])`); `GENSET_ETA_HHV` 463; `GENSET_EF_TH_HHV` 464 | the selected genset arm's (`genset_srmc`/`genset_central`/`genset_high`) `total_usd_per_twh_e` from `data/cost/processed/genset_cost.csv` | **LIVE** — reachable whenever `genset_case=None` is passed to `load_econ()` explicitly, a deliberate, named configuration (the file's own comment: "the commit-A regression test"), not an oversight. This is also the exact hinge for `m.unserved`'s fix/free status: `allow_unserved`'s auto-default is `"GENSET_COST_PER_TWH_E" not in econ` (`optimize_model_sliced.py:591-592`) — so this one conditional in `load_econ()` is what determines, across the whole model, whether `unserved` is fixed at zero or left free and priced at VoLL |

**Summary: 16 of 18 scenario-side silent defaults are DEAD** (always
overridden by `load_scenario()`, most to the identical value the
default itself carries); **2 are LIVE without qualification**
(`within_block_peaking_factor`, unconditionally; `selfgen_busbar_factor`,
except in two named scripts); and **the one `econ`-side default is LIVE
on a deliberate, named path** (`genset_case=None`). No default's value
was changed in producing this table.

---

## Section 4 — Expressions

41 `pyo.Expression(` call sites, **39 uniquely named** (`m.emissions_grid_only_by_year`
and `m.emissions_capped_by_year` are each built at two separate, mutually
exclusive call sites — one per branch — so they count once as a name
but twice as a call site), in source order. One further model
attribute, `m.slice_hours`, is **not** a Pyomo component (`m.slice_hours
= dict(H)`, line 284 — a plain Python dict, "for diagnostics" per its
own comment) and is excluded from the count and from this section.

### 4.1 Capacity expressions (annual)

#### `m.gas_capacity_mw` — line 701
- index: `T`
- conditional: no
- rule body (verbatim, `gas_cap_rule`, lines 690–699):
```python
def gas_cap_rule(m, t):
    year_t = int(years[t])
    if retirement_series is not None:
        retired = min(retirement_series[t], gas_baseline)
    elif retirement_start is not None and year_t >= retirement_start:
        retired = min(retirement_rate * (year_t - retirement_start + 1),
                      gas_baseline)
    else:
        retired = 0.0
    return gas_baseline - retired + sum(m.gas_add[k] for k in range(0, t + 1))
```
- source of data: `gas_baseline = scenario["gas_baseline_mw"]` (671);
  `retirement_series = scenario.get("gas_baseline_retirement_series_mw", None)` (682);
  `retirement_start = scenario.get("gas_baseline_retirement_start_year", None)` (672);
  `retirement_rate = scenario.get("gas_baseline_retirement_mw_per_year", 0.0)` (673)

#### `m.storage_capacity_mwh` — line 733
- index: `T`
- conditional: no
- rule body (verbatim, `storage_cap_rule`, line 730–731):
```python
def storage_cap_rule(m, t):
    return sum(m.storage_add[k] for k in range(0, t + 1))
```
- source of data: `m.storage_add` (decision variable)

#### `m.storage_power_mw` — line 734–736
- index: `T`
- conditional: no
- rule body (verbatim, lambda):
```python
lambda m, t: m.storage_capacity_mwh[t] / storage_duration_hours
```
- source of data: `storage_duration_hours = scenario["storage_duration_hours"]` (496)

#### `m.solar_capacity_mw` — line 745
- index: `T`
- conditional: no
- rule body (verbatim, `solar_cap_rule`, lines 740–743):
```python
def solar_cap_rule(m, t):
    return baseline_mw + sum(
        m.solar_public_add[k] + m.solar_eaas_add[k] for k in range(0, t + 1)
    )
```
- source of data: `baseline_mw = scenario["solar_baseline_mw"]` (738)

### 4.2 Generation expressions (sliced)

#### `m.gas_generation` — lines 803–805
- index: `T, m.S`
- conditional: no
- rule body (verbatim):
```python
lambda m, t, s, p: eta * m.gas_to_power[t, (s, p)]
```
- source of data: `eta = scenario.get("gas_eta", None)` (491, raises if `None`)

#### `m.convert_generation` — lines 810–814
- index: `_CONV_KEYS, T, m.S`
- conditional: **only if `_conv`**
- rule body (verbatim):
```python
lambda m, tier, t, s, p:
    _eta_retrofit * m.convert_to_power[tier, t, (s, p)]
```
- source of data: `_eta_retrofit = float(scenario["conversion_retrofit_efficiency"])` (532)

#### `m.solar_generation` — lines 830–833
- index: `T, m.S`
- conditional: no
- rule body (verbatim):
```python
lambda m, t, s, p: m.solar_capacity_mw[t] * solar_per_mw[(s, p)]
```
- source of data: `solar_per_mw = _solar_energy_per_mw_by_slice(scenario)` (792), which reads
  `scenario["solar_cf"]` (184) and `scenario.get("season_solar_factor", DEFAULT_SEASON_SOLAR_FACTOR)` (185)

### 4.3 Gas exempt-deliverability expressions (conditional on `new_gas_deliverability == "exempt"`)

#### `m.gas_new_capacity_mw` — lines 905–907
- index: `T`
- conditional: **only if `new_gas_deliverability == "exempt"`**
- rule body (verbatim):
```python
lambda m, t: sum(m.gas_add[k] for k in range(0, t + 1))
```
- source of data: `m.gas_add` (decision variable)

#### `m.gas_exempt_fuel_allowance_twh_th` — lines 908–913
- index: `T`
- conditional: same
- rule body (verbatim):
```python
lambda m, t:
    m.gas_new_capacity_mw[t] * 8760 * gas_availability_factor
    / gas_new_efficiency / 1e6
```
- source of data: `gas_availability_factor = scenario.get("gas_availability_factor", None)` (597,
  raises if `None`); `gas_new_efficiency = scenario["gas_new_efficiency"]` (904)

### 4.4 Pre-F1 (`pre_f1_overnight`) cost expressions

Built only when `capital_treatment_case == "pre_f1_overnight"`.

#### `m.salvage_npv` — lines 1462–1502
- index: scalar (`expr=...`)
- conditional: **only if `capital_treatment_case == "pre_f1_overnight"`**
- rule body (verbatim):
```python
expr=m.DF[T_last] * (
    pyo.quicksum(
        (m.solar_public_add[t] + m.solar_eaas_add[t])
        * m.solar_capex_param[t]
        * _salvage_fraction(t, lives["solar"])
        for t in T
    )
    + pyo.quicksum(
        m.storage_add[t] * econ["STORAGE_COST_PER_MWH"]
        * _salvage_fraction(t, lives["storage"])
        for t in T
    )
    + pyo.quicksum(
        m.gas_add[t] * scenario["gas_capex_per_mw"]
        * _salvage_fraction(t, lives["gas"])
        for t in T
    )
    # Replacement vintages get the SAME straight-line salvage
    # treatment as original builds, credited at T_last (Step 1).
    + pyo.quicksum(
        (m.solar_public_add[build_t] + m.solar_eaas_add[build_t])
        * m.solar_capex_param[rebuild_t]
        * _salvage_fraction(rebuild_t, lives["solar"])
        for build_t in T
        for rebuild_t in _rebuild_years(build_t, lives["solar"])
    )
    + pyo.quicksum(
        m.storage_add[build_t] * econ["STORAGE_COST_PER_MWH"]
        * _salvage_fraction(rebuild_t, lives["storage"])
        for build_t in T
        for rebuild_t in _rebuild_years(build_t, lives["storage"])
    )
    + pyo.quicksum(
        m.gas_add[build_t] * scenario["gas_capex_per_mw"]
        * _salvage_fraction(rebuild_t, lives["gas"])
        for build_t in T
        for rebuild_t in _rebuild_years(build_t, lives["gas"])
    )
)
```
  where (lines 1428–1433, 1453–1460):
```python
def _salvage_fraction(t, life):
    """Straight-line residual fraction at end of horizon for vintage t.
    Vintage t operates years t..T_last inclusive = (T_last - t + 1) years.
    Returns 0.0 for vintages that retire within the horizon."""
    served = T_last - t + 1
    return max(0.0, float(life) - served) / float(life)

def _rebuild_years(build_t, life):
    """rebuild_t for every in-horizon replacement of the vintage built
    at build_t, `life` years apart, for as many cycles as fit <= T_last."""
    life_int = int(round(float(life)))
    k = 1
    while build_t + k * life_int <= T_last:
        yield build_t + k * life_int
        k += 1
```
- source of data: `lives = scenario.get("asset_lifetimes", None)` (1412, raises if `None`,
  sub-keys `solar`/`storage`/`gas` checked present 1419–1424);
  `scenario["gas_capex_per_mw"]`; `econ["STORAGE_COST_PER_MWH"]`; `T_last = len(T) - 1` (1426)

#### `m.cost_replacement_capex_npv` — lines 1510–1530
- index: scalar
- conditional: same (`pre_f1_overnight`)
- rule body (verbatim):
```python
expr=(
    pyo.quicksum(
        m.DF[rebuild_t]
        * (m.solar_public_add[build_t] + m.solar_eaas_add[build_t])
        * m.solar_capex_param[rebuild_t]
        for build_t in T
        for rebuild_t in _rebuild_years(build_t, lives["solar"])
    )
    + pyo.quicksum(
        m.DF[rebuild_t] * m.storage_add[build_t] * econ["STORAGE_COST_PER_MWH"]
        for build_t in T
        for rebuild_t in _rebuild_years(build_t, lives["storage"])
    )
    + pyo.quicksum(
        m.DF[rebuild_t] * m.gas_add[build_t] * scenario["gas_capex_per_mw"]
        for build_t in T
        for rebuild_t in _rebuild_years(build_t, lives["gas"])
    )
)
```
- source of data: same as `m.salvage_npv` above. This is read independently
  as "a SEPARATE, visible cost — not folded into gas/storage/solar capex_npv
  above" per the file's own comment (1504–1509).

### 4.5 F1-tranche (`f1_tranche`) cost expressions

Built only in the `else` branch, i.e. `capital_treatment_case != "pre_f1_overnight"` (in practice `== "f1_tranche"`, the only other allowed value).

#### `m.concessional_volume_by_vintage` — lines 1726–1730
- index: `T`
- conditional: **`f1_tranche` only**
- rule body (verbatim):
```python
lambda m, t:
    m.solar_public_conc_vol[t] + m.solar_eaas_conc_vol[t]
    + m.storage_conc_vol[t]
```
- source of data: decision variables only (no scenario/econ read directly in this rule)

#### `m.cost_public_solar_capital_charge_npv` — lines 1738–1744
- index: scalar
- conditional: `f1_tranche` only
- rule body (verbatim):
```python
expr=pyo.quicksum(
    _crf_conc["solar"] * _df_window_conc["solar"][v] * m.solar_public_conc_vol[v]
    + _crf_comm["solar"] * _df_window_comm["solar"][v] * m.solar_public_comm_vol[v]
    for v in T
)
```
- source of data: `_crf_conc`/`_crf_comm` from `_crf(r, L)` (defined 1540–1555) applied to
  `_r_conc_real`/`_r_comm_real` (1565–1566, deflated from `scenario["r_conc_nominal"]` (1559)/
  `scenario["r_comm_nominal"]` (1560) by `scenario["us_inflation"]` (1561)) and
  `_L_eff_conc`/`_L_eff_comm` (1596–1603, `min(scenario["L_conc"]`/`scenario["L_comm"]`,
  `scenario["asset_lifetimes"][ch])`); `_df_window_conc`/`_df_window_comm` (1619–1632, sums of
  `m.DF` over each vintage's in-window years)

#### `m.cost_eaas_solar_capital_charge_npv` — lines 1745–1751
- index: scalar
- conditional: `f1_tranche` only
- rule body (verbatim):
```python
expr=pyo.quicksum(
    _crf_conc["solar"] * _df_window_conc["solar"][v] * m.solar_eaas_conc_vol[v]
    + _crf_comm["solar"] * _df_window_comm["solar"][v] * m.solar_eaas_comm_vol[v]
    for v in T
)
```
- source of data: same as above

#### `m.cost_gas_capital_charge_npv` — lines 1752–1758
- index: scalar
- conditional: `f1_tranche` only
- rule body (verbatim):
```python
expr=pyo.quicksum(
    _crf_conc["gas"] * _df_window_conc["gas"][v] * m.gas_conc_vol[v]
    + _crf_comm["gas"] * _df_window_comm["gas"][v] * m.gas_comm_vol[v]
    for v in T
)
```
- source of data: same CRF/df-window machinery, `"gas"` channel. (`m.gas_conc_vol` is fixed at
  0.0 for every `t`, so the first term is always zero in practice — gas is commercial-only.)

#### `m.cost_storage_capital_charge_npv` — lines 1759–1765
- index: scalar
- conditional: `f1_tranche` only
- rule body (verbatim):
```python
expr=pyo.quicksum(
    _crf_conc["storage"] * _df_window_conc["storage"][v] * m.storage_conc_vol[v]
    + _crf_comm["storage"] * _df_window_comm["storage"][v] * m.storage_comm_vol[v]
    for v in T
)
```
- source of data: same CRF/df-window machinery, `"storage"` channel

#### `m.cost_rebuild_capital_charge_npv` — lines 1781–1803
- index: scalar
- conditional: `f1_tranche` only
- rule body (verbatim):
```python
expr=(
    pyo.quicksum(
        _crf_comm["solar"] * _df_window_comm["solar"][rebuild_t]
        * (m.solar_public_add[build_t] + m.solar_eaas_add[build_t])
        * m.solar_capex_param[rebuild_t]
        for build_t in T
        for rebuild_t in _rebuild_years(build_t, _lives["solar"])
    )
    + pyo.quicksum(
        _crf_comm["storage"] * _df_window_comm["storage"][rebuild_t]
        * m.storage_add[build_t] * econ["STORAGE_COST_PER_MWH"]
        for build_t in T
        for rebuild_t in _rebuild_years(build_t, _lives["storage"])
    )
    + pyo.quicksum(
        _crf_comm["gas"] * _df_window_comm["gas"][rebuild_t]
        * m.gas_add[build_t] * scenario["gas_capex_per_mw"]
        for build_t in T
        for rebuild_t in _rebuild_years(build_t, _lives["gas"])
    )
)
```
  using the `f1_tranche` branch's own `_rebuild_years` (lines 1639–1646, a verbatim
  duplicate of the `pre_f1_overnight` branch's closure — the file's own comment states
  this is deliberate, not shared, since the two branches must stay independently editable).
- source of data: `_lives = scenario.get("asset_lifetimes", None)` (1581, raises if `None`,
  a SEPARATE read from `lives` in §4.4); commercial-only CRFs (`_crf_comm`) — F2 Decision 3:
  rebuilt vintages are commercial-only, no concessional counterpart

#### `m.cost_conversion_capital_charge_npv` — lines 1828–1834
- index: scalar
- conditional: **`f1_tranche` AND `_conv` both true**
- rule body (verbatim):
```python
expr=pyo.quicksum(
    _crf_comm["gas"] * _df_window_comm["gas"][0]
    * m.convert_comm_vol[tier]
    for tier in _CONV_KEYS
)
```
- source of data: `_conv_capex_per_kw = float(scenario["conversion_capex_per_kw"])` (533,
  via `m.convert_comm_vol`'s split-identity constraint, §5); vintage fixed at `0`
  (convert_mw is a one-off decision, Decision 5)

#### `m.capital_charge_total_npv` — line 1838
- index: scalar
- conditional: `f1_tranche` only
- rule body (verbatim):
```python
expr=_capital_charge_total
```
  where `_capital_charge_total` (1813–1837) is built up as:
```python
_capital_charge_total = (
    m.cost_public_solar_capital_charge_npv
    + m.cost_eaas_solar_capital_charge_npv
    + m.cost_gas_capital_charge_npv
    + m.cost_storage_capital_charge_npv
    + m.cost_rebuild_capital_charge_npv
)
# + m.cost_conversion_capital_charge_npv, only if _conv
```
- source of data: the five (or six, if `_conv`) capital-charge Expressions above

### 4.6 Unconditional cost expressions (built regardless of `capital_treatment_case`)

These are plain Python values, not `pyo.Expression`, until wrapped by the
named `m.cost_*`/`m.system_cost_npv` Expressions further below — recorded
here as the local variables they are computed from.

- `gas_opex_npv` (lines 1326–1329, extended 1338–1342 if `_conv`):
```python
gas_opex_npv = pyo.quicksum(
    m.DF[t] * m.gas_to_power[t, (s, p)] * econ["GAS_COST_PER_TWH_TH"]
    for t in T for (s, p) in S
)
# if _conv:
gas_opex_npv = gas_opex_npv + pyo.quicksum(
    m.DF[t] * m.convert_to_power[tier, t, (s, p)]
    * econ["GAS_COST_PER_TWH_TH"]
    for tier in _CONV_KEYS for t in T for (s, p) in S
)
```
- `om_cost_npv` (lines 1363–1389) — delegated entirely to the external
  function `compute_om_cost_npv()` (`src/io.py`, not inventoried here):
```python
om_unconverted_gas_gen_twh = {
    t: pyo.quicksum(m.gas_generation[t, (s, p)] for (s, p) in S)
    for t in T
}
# om_converted_gas_capacity_mw / om_converted_gas_gen_twh: _converted_ocgt_nameplate /
# _converted_gen_year(m,t) if _conv, else 0.0 for every t
om_unconverted_gas_capacity_mw = {
    t: m.gas_capacity_mw[t] - om_converted_gas_capacity_mw[t] for t in T
}
om_cost_npv = compute_om_cost_npv(
    T=T, years=years, DF=m.DF,
    om_cost_by_year=scenario["om_cost_by_year"],
    storage_duration_hours=storage_duration_hours,
    solar_capacity_mw=m.solar_capacity_mw,
    storage_capacity_mwh=m.storage_capacity_mwh,
    gas_unconverted_capacity_mw=om_unconverted_gas_capacity_mw,
    gas_converted_capacity_mw=om_converted_gas_capacity_mw,
    gas_unconverted_generation_twh=om_unconverted_gas_gen_twh,
    gas_converted_generation_twh=om_converted_gas_gen_twh,
)
```
- `unserved_npv` (lines 1840–1843):
```python
unserved_npv = pyo.quicksum(
    m.DF[t] * m.unserved[t, (s, p)] * econ["UNSERVED_ENERGY_PENALTY"]
    for t in T for (s, p) in S
)
```
- `selfgen_npv` (lines 1845–1848):
```python
selfgen_npv = pyo.quicksum(
    m.DF[t] * m.selfgen[t, (s, p)] * selfgen_price
    for t in T for (s, p) in S
)
```
  where `selfgen_price = econ.get("GENSET_COST_PER_TWH_E", voll)` (551)
- `carbon_cost_npv` (lines 1911–1916, a plain float `0.0` unless
  `scenario.get("carbon_active", False)` is truthy):
```python
carbon_cost_npv = 0.0
if scenario.get("carbon_active", False):
    carbon_price = float(scenario["carbon_price"])
    carbon_cost_npv = pyo.quicksum(
        m.DF[t] * m.emissions_by_year[t] * carbon_price for t in T
    )
```

#### `m.system_cost_npv` — line 1949
- index: scalar
- conditional: **rule body differs by `capital_treatment_case`** (both variants shown)
- rule body (verbatim):
```python
# capital_treatment_case == "pre_f1_overnight":
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
    + om_cost_npv
)
# else (f1_tranche):
system_cost_npv = (
    gas_opex_npv
    + m.capital_charge_total_npv
    + selfgen_npv
    + unserved_npv
    + carbon_cost_npv
    + om_cost_npv
)
```
- source of data: all cost components above. The file's own comment (1918–1925)
  states the `pre_f1_overnight` branch is written with the IDENTICAL term
  order as the pre-F1 formula (not refactored) so the expression tree, not
  just its value, is unchanged — floating-point addition is not strictly
  associative.

#### `m.cost_gas_opex_npv` — line 1951
- rule body: `expr=gas_opex_npv`

#### `m.cost_om_npv` — line 1952
- rule body: `expr=om_cost_npv`

#### `m.cost_gas_capex_npv` — line 1954 (conditional: `pre_f1_overnight` only)
- rule body: `expr=gas_capex_npv`, where (line 1395–1397)
```python
gas_capex_npv = pyo.quicksum(
    m.DF[t] * m.gas_add[t] * scenario["gas_capex_per_mw"] for t in T
)
```

#### `m.cost_public_solar_capex_npv` — line 1955 (conditional: `pre_f1_overnight` only)
- rule body: `expr=public_solar_capex_npv`, where (lines 1071–1073)
```python
public_solar_capex_npv = pyo.quicksum(
    m.DF[t] * m.solar_public_add[t] * m.solar_capex_param[t] for t in T
)
```
**Note:** distinct Expression from `m.cost_public_solar_capital_charge_npv`
(§4.5, `f1_tranche` only) — the two names are deliberately different so a
stale reader cannot silently read one under the wrong `capital_treatment_case`.

#### `m.cost_eaas_solar_capex_npv` — line 1956 (conditional: `pre_f1_overnight` only)
- rule body: `expr=eaas_solar_capex_npv`, where (lines 1089–1091)
```python
eaas_solar_capex_npv = pyo.quicksum(
    m.DF[t] * m.solar_eaas_add[t] * m.solar_capex_param[t] for t in T
)
```

#### `m.cost_storage_capex_npv` — line 1957 (conditional: `pre_f1_overnight` only)
- rule body: `expr=storage_capex_npv`, where (lines 1398–1400)
```python
storage_capex_npv = pyo.quicksum(
    m.DF[t] * m.storage_add[t] * econ["STORAGE_COST_PER_MWH"] for t in T
)
```

#### `m.cost_salvage_npv` — line 1958 (conditional: `pre_f1_overnight` only)
- rule body: `expr=m.salvage_npv` (a pass-through alias of §4.4's `m.salvage_npv`)

#### `m.cost_unserved_voll_npv` — line 1959
- rule body: `expr=unserved_npv`

#### `m.cost_selfgen_npv` — line 1960
- rule body: `expr=selfgen_npv`

#### `m.emissions_selfgen` — lines 1961–1962
- rule body: `expr=pyo.quicksum(m.emissions_selfgen_by_year[t] for t in T)`

#### `m.emissions_grid_only` — lines 1963–1964
- rule body: `expr=pyo.quicksum(m.emissions_grid_only_by_year[t] for t in T)`

#### `m.cost_carbon_npv` — line 1965
- rule body: `expr=carbon_cost_npv`

#### `m.conversion_overnight_capex_usd_by_year` — lines 1993–2000
- index: `T`
- conditional: **only if `_conv`**
- rule body (verbatim):
```python
lambda m, t:
    pyo.quicksum(
        _conv_capex_per_kw * 1000.0 * m.convert_mw[tier]
        for tier in _CONV_KEYS
    ) if t == 0 else 0.0
```
- source of data: `_conv_capex_per_kw = float(scenario["conversion_capex_per_kw"])` (533).
  The file's own comment (1966–1991) states this Expression is referenced by **no**
  constraint and **no** objective term — a pure diagnostic, read only by
  `extract_planning_diagnostics()`'s `_capex_at()` in `src/optimize_experiments.py`
  (external, not inventoried here) — deliberately distinct from `m.convert_comm_vol`
  even though numerically identical per tier, because `convert_comm_vol`'s name
  implies a commercial-tranche financing split it does not itself carry.

### 4.7 Emissions expressions

Three emissions quantities, deliberately distinct per the file's own
comment (1855–1868): `emissions_grid_only_by_year` is the NDC commitment
boundary; `emissions_selfgen_by_year` is private diesel, outside that
boundary; `emissions_by_year` is the physical total.

#### `m.emissions_grid_only_by_year` — two variants
- index: `T`
- conditional: **rule body differs by `_conv`**
- rule body (verbatim, `_conv` True, lines 1870–1876):
```python
lambda m, t:
    pyo.quicksum(m.gas_generation[t, (s, p)] for (s, p) in S)
    * 1e6 * ef_e
    + _converted_gen_year(m, t) * 1e6 * _ef_e_retrofit
```
- rule body (verbatim, `_conv` False, lines 1878–1883):
```python
lambda m, t: pyo.quicksum(
    m.gas_generation[t, (s, p)] for (s, p) in S
) * 1e6 * ef_e
```
- source of data: `ef_e = econ["EF_TCO2_PER_MWH_TH"] / eta` (1850);
  `_ef_e_retrofit = econ["EF_TCO2_PER_MWH_TH"] / _eta_retrofit` (1854, if `_conv`)

#### `m.emissions_selfgen_by_year` — lines 1884–1889
- index: `T`
- conditional: no
- rule body (verbatim):
```python
lambda m, t: pyo.quicksum(
    m.selfgen[t, (s, p)] for (s, p) in S
) * 1e6 * selfgen_ef_e
```
- source of data: `selfgen_ef_e = econ["GENSET_EF_TH_HHV"] / econ["GENSET_ETA_HHV"]` if
  `"GENSET_ETA_HHV" in econ` (552–554), else `0.0` (556)

#### `m.emissions_by_year` — lines 1891–1896
- index: `T`
- conditional: no. The file's own comment: "ALWAYS total — no longer
  switched by `selfgen_in_emissions_cap`" (1890)
- rule body (verbatim):
```python
lambda m, t:
    m.emissions_grid_only_by_year[t]
    + m.emissions_selfgen_by_year[t]
```

#### `m.emissions_capped_by_year` — two variants, lines 1899–1904
- index: `T`
- conditional: **rule body switches on the `selfgen_in_emissions_cap` function argument**
- rule body (verbatim, `selfgen_in_emissions_cap` True):
```python
lambda m, t: m.emissions_by_year[t]
```
- rule body (verbatim, `selfgen_in_emissions_cap` False — the default):
```python
lambda m, t: m.emissions_grid_only_by_year[t]
```
- source of data: the file's own comment: "What the CAP constrains.
  Default grid-only (correct NDC boundary); True widens it to total =
  the alternative-boundary arm" (1897–1898)

#### `m.emissions_capped` — lines 1905–1906
- rule body: `expr=pyo.quicksum(m.emissions_capped_by_year[t] for t in T)`

#### `m.emissions` — lines 1907–1909
- rule body: `expr=pyo.quicksum(m.emissions_by_year[t] for t in T)`

**Count check:** `grep -c "pyo\.Expression(" src/optimize_model_sliced.py`
= 41 call sites; all 41 accounted for above across 39 uniquely named
Expressions (see note above on the two double-branch names).

---

## Section 5 — Constraints

34 `pyo.Constraint(` call sites, **26 uniquely named** (six names are
each built at two-to-four mutually exclusive call sites depending on a
branch condition: `gas_capacity_constraint` ×2, `gas_balance` ×4,
`balance` ×2, `grid_service_obligation` ×2, `firm_capacity_constraint`
×2, `emissions_constraint` ×2). Grouped as requested; source-order line
numbers given per constraint.

### 5.1 Energy balance

#### `m.balance` — two variants, lines 1026–1052
- index: `T, m.S`
- conditional: **rule body differs by `_conv`**
- rule body (verbatim, `_conv` True, lines 1026–1039):
```python
lambda m, t, s, p:
    m.gas_generation[t, (s, p)]
    + pyo.quicksum(m.convert_generation[tier, t, (s, p)]
                   for tier in _CONV_KEYS)
    + m.solar_generation[t, (s, p)]
    + m.hydro_gen[t, (s, p)]
    + m.storage_discharge[t, (s, p)]
    - m.storage_charge[t, (s, p)]
    + m.selfgen[t, (s, p)] * selfgen_busbar_factor
    + m.unserved[t, (s, p)]
    >= demand[(t, s, p)]
```
- rule body (verbatim, `_conv` False, lines 1041–1052):
```python
lambda m, t, s, p:
    m.gas_generation[t, (s, p)]
    + m.solar_generation[t, (s, p)]
    + m.hydro_gen[t, (s, p)]
    + m.storage_discharge[t, (s, p)]
    - m.storage_charge[t, (s, p)]
    + m.selfgen[t, (s, p)] * selfgen_busbar_factor
    + m.unserved[t, (s, p)]
    >= demand[(t, s, p)]
```
- source of data: `selfgen_busbar_factor = float(scenario.get("selfgen_busbar_factor", 1.0))`
  (563); `demand = {(t,s,p): demand_annual[t]*dw[(s,p)] for t in T for (s,p) in S}` (797–798),
  where `demand_annual` comes from `project_baseline_demand()` (external, `src/demand.py`)
  fed `scenario["base_demand_twh"]`/`scenario["demand_growth"]`, and `dw =
  _normalised_demand_weights(scenario)` reads `scenario["season_demand_factor"]`/
  `scenario["period_demand_factor"]` (164–165)

### 5.2 Gas

#### `m.gas_capacity_constraint` — two variants, lines 854–878
- index: `T, m.S`
- conditional: **rule body differs by `_conv`**
- rule body (verbatim, `_conv` True, lines 854–860):
```python
lambda m, t, s, p:
    m.gas_generation[t, (s, p)]
    <= (m.gas_capacity_mw[t] - _converted_ocgt_nameplate)
       * H[(s, p)] * gas_availability_factor / 1e6
```
- rule body (verbatim, `_conv` False, lines 873–877):
```python
lambda m, t, s, p:
    m.gas_generation[t, (s, p)]
    <= m.gas_capacity_mw[t] * H[(s, p)] * gas_availability_factor / 1e6
```
- source of data: `gas_availability_factor = scenario.get("gas_availability_factor", None)`
  (597, raises if `None`); `H = slice_hours()` (282); `_converted_ocgt_nameplate =
  pyo.quicksum(m.convert_mw[tier] for tier in _CONV_KEYS)` (851–853, if `_conv`)

#### `m.gas_new_capacity_cap_constraint` — lines 723–726
- index: `T`
- conditional: **only if `gas_build_case == "gas_ndc_17gw"`**
- rule body (verbatim):
```python
lambda m, t: m.gas_capacity_mw[t] <= gas_new_capacity_cap_mw[t]
```
- source of data: `gas_new_capacity_cap_mw = scenario.get("gas_new_capacity_cap_mw", None)`
  (713, raises if `None` under this arm)

#### `m.convert_capacity_constraint` — lines 865–871
- index: `_CONV_KEYS, T, m.S`
- conditional: **only if `_conv`**
- rule body (verbatim):
```python
lambda m, tier, t, s, p:
    m.convert_generation[tier, t, (s, p)]
    <= (1.0 + _conv_uplift) * m.convert_mw[tier]
       * H[(s, p)] * _conv_tiers[tier]["paf"] / 1e6
```
- source of data: `_conv_uplift = float(scenario["conversion_uplift"])` (531);
  `_conv_tiers = scenario["conversion_tiers"]` (529), each tier's `"paf"` — the tier's
  own availability, deliberately NOT the fleet-wide `gas_availability_factor`
  (Decision 4: "a blended availability is forbidden")

#### `m.gas_balance` — four variants, lines 915–948
- index: `T`
- conditional: **rule body differs on both `new_gas_deliverability` and `_conv`**
- rule body (verbatim, `exempt` + `_conv`, lines 915–922):
```python
lambda m, t:
    pyo.quicksum(m.gas_to_power[t, (s, p)] for (s, p) in S)
    + _converted_thermal_year(m, t)
    + m.gas_curtail[t]
    <= gas_avail[t] + m.gas_exempt_fuel_allowance_twh_th[t]
```
- rule body (verbatim, `exempt` + not `_conv`, lines 924–929):
```python
lambda m, t:
    pyo.quicksum(m.gas_to_power[t, (s, p)] for (s, p) in S)
    + m.gas_curtail[t]
    <= gas_avail[t] + m.gas_exempt_fuel_allowance_twh_th[t]
```
- rule body (verbatim, `constrained` + `_conv`, lines 935–941):
```python
lambda m, t:
    pyo.quicksum(m.gas_to_power[t, (s, p)] for (s, p) in S)
    + _converted_thermal_year(m, t)
    + m.gas_curtail[t] <= gas_avail[t]
```
- rule body (verbatim, `constrained` + not `_conv`, lines 943–947):
```python
lambda m, t:
    pyo.quicksum(m.gas_to_power[t, (s, p)] for (s, p) in S)
    + m.gas_curtail[t] <= gas_avail[t]
```
- source of data: `new_gas_deliverability = scenario.get("new_gas_deliverability",
  "constrained")` (894); `gas_avail = gas_available_power(start_year=..., end_year=...,
  scenario_name=scenario["gas_scenario"])["available_twh_th"]` (757–761, external function,
  `src/gas_supply.py`, not inventoried here)

#### `m.gas_split_identity` — lines 1683–1687 (`f1_tranche` only)
- index: `T`
- conditional: `f1_tranche` only
- rule body (verbatim):
```python
lambda m, t:
    m.gas_conc_vol[t] + m.gas_comm_vol[t]
    == scenario["gas_capex_per_mw"] * m.gas_add[t]
```
- source of data: `scenario["gas_capex_per_mw"]`. Since `m.gas_conc_vol` is fixed at `0.0`
  for every `t` (1681–1682), this identity forces `gas_comm_vol[t]` to absorb the full
  amount — "every dollar of gas capex is commercial" (envelope resolution Decision 1)

#### `m.convert_split_identity` — lines 1822–1827 (`f1_tranche` AND `_conv`)
- index: `_CONV_KEYS`
- conditional: **`f1_tranche` AND `_conv` both true**
- rule body (verbatim):
```python
lambda m, tier:
    m.convert_comm_vol[tier]
    == _conv_capex_per_kw * 1000.0 * m.convert_mw[tier]
```
- source of data: `_conv_capex_per_kw = float(scenario["conversion_capex_per_kw"])` (533)

### 5.3 Hydro

#### `m.hydro_season_budget` — lines 956–961
- index: `T, ["dry", "wet"]` (plain list, not a named Pyomo Set — see §1)
- conditional: no
- rule body (verbatim):
```python
lambda m, t, season:
    pyo.quicksum(m.hydro_gen[t, (s, p)] for (s, p) in S if s == season)
    == hydro_annual[t] * hydro_season_share[season]
```
- source of data: `hydro_annual = {t: float(hydro_energy_twh_by_year[t]) for t in T}` (788),
  from `scenario.get("hydro_energy_twh_by_year", None)` (768, raises if `None`);
  `hydro_season_share = _hydro_season_energy_share(scenario)` (793), which reads
  `scenario.get("hydro_season_factor", DEFAULT_HYDRO_SEASON_FACTOR)` (209)

#### `m.hydro_power_limit` — lines 967–972
- index: `T, m.S`
- conditional: no
- rule body (verbatim):
```python
lambda m, t, s, p:
    m.hydro_gen[t, (s, p)]
    <= hydro_capacity_mw_by_year[t] * H[(s, p)] / 1e6
```
- source of data: `hydro_capacity_mw_by_year = scenario.get("hydro_capacity_mw_by_year",
  None)` (776, raises if `None`)

#### `m.hydro_min_flow` — lines 979–985
- index: `T, m.S`
- conditional: no
- rule body (verbatim):
```python
lambda m, t, s, p:
    m.hydro_gen[t, (s, p)]
    >= hydro_min_frac * hydro_annual[t] * hydro_season_share[s]
    * (H[(s, p)] / season_hours[s])
```
- source of data: `hydro_min_frac = scenario.get("hydro_min_flow_fraction",
  DEFAULT_HYDRO_MIN_FLOW_FRACTION)` (794–795); `season_hours = {s: sum(H[(ss,p)] for
  (ss,p) in S if ss==s) for s in ("dry","wet")}` (976–977)

### 5.4 Storage

#### `m.storage_season_balance` — lines 994–1000
- index: `T, ["dry", "wet"]`
- conditional: no
- rule body (verbatim):
```python
lambda m, t, season:
    pyo.quicksum(m.storage_discharge[t, (s, p)] for (s, p) in S if s == season)
    <= storage_round_trip_eff
    * pyo.quicksum(m.storage_charge[t, (s, p)] for (s, p) in S if s == season)
```
- source of data: `storage_round_trip_eff = scenario.get("storage_round_trip_eff", 0.90)` (495)

#### `m.storage_charge_power_limit` — lines 1002–1007
- index: `T, m.S`
- conditional: no
- rule body (verbatim):
```python
lambda m, t, s, p:
    m.storage_charge[t, (s, p)]
    <= m.storage_power_mw[t] * H[(s, p)] / 1e6
```

#### `m.storage_discharge_power_limit` — lines 1009–1014
- index: `T, m.S`
- conditional: no
- rule body (verbatim):
```python
lambda m, t, s, p:
    m.storage_discharge[t, (s, p)]
    <= m.storage_power_mw[t] * H[(s, p)] / 1e6
```

#### `m.storage_split_identity` — lines 1691–1695 (`f1_tranche` only)
- index: `T`
- conditional: `f1_tranche` only
- rule body (verbatim):
```python
lambda m, t:
    m.storage_conc_vol[t] + m.storage_comm_vol[t]
    == econ["STORAGE_COST_PER_MWH"] * m.storage_add[t]
```

### 5.5 Firm capacity

#### `m.firm_capacity_constraint` — two variants, lines 1295–1319
- index: `T`
- conditional: **rule body differs by `_conv`**
- rule body (verbatim, `_conv` True, lines 1295–1308):
```python
lambda m, t:
    capacity_credit["solar"] * m.solar_capacity_mw[t]
    + capacity_credit["gas"] * m.gas_capacity_mw[t]
    + capacity_credit["storage"] * m.storage_power_mw[t]
    + capacity_credit["hydro"] * hydro_capacity_mw_by_year[t]
    + pyo.quicksum(
        _firm_credit[tier] * _conv_uplift * m.convert_mw[tier]
        for tier in _CONV_KEYS
    )
    >= peak_mw[t] * within_block_peaking_factor
       * (1.0 + _reserve_margin_by_year[t])
```
- rule body (verbatim, `_conv` False, lines 1310–1319):
```python
lambda m, t:
    capacity_credit["solar"] * m.solar_capacity_mw[t]
    + capacity_credit["gas"] * m.gas_capacity_mw[t]
    + capacity_credit["storage"] * m.storage_power_mw[t]
    + capacity_credit["hydro"] * hydro_capacity_mw_by_year[t]
    >= peak_mw[t] * within_block_peaking_factor
       * (1.0 + _reserve_margin_by_year[t])
```
- source of data: `capacity_credit = scenario.get("capacity_credit", None)` (322, raises
  if `None`, sub-keys `solar`/`gas`/`storage`/`hydro` checked present); `peak_mw =
  scenario.get("peak_mw", None)` (302, raises if `None`); `within_block_peaking_factor =
  float(scenario.get("within_block_peaking_factor", 1.3448))` (1262–1264 — **the flagged
  hardcoded literal, register L50 / forensic audit A62; never set by `load_scenario()`,
  see the default-reachability audit above**); `_reserve_margin_by_year` (1245–1253, from
  `scenario.get("reserve_margin", None)` (315, raises if `None`) — scalar or per-year list);
  `_firm_credit = {tier: capacity_credit["gas"] * _conv_tiers[tier]["paf"] /
  gas_availability_factor for tier in _CONV_KEYS}` (1290–1294, if `_conv`)

### 5.6 Solar build (cumulative cap, minimum build, land)

#### `m.solar_cumulative_cap` — lines 1167–1174
- index: `T`
- conditional: **only if `scenario.get("solar_cumulative_cap_mw", None)` is not `None`**
- rule body (verbatim, `solar_cum_cap_rule`, lines 1167–1172):
```python
def solar_cum_cap_rule(m, t):
    return (
        sum(m.solar_public_add[k] + m.solar_eaas_add[k]
            for k in range(0, t + 1))
        <= solar_cum_cap[t]
    )
```
- source of data: `solar_cum_cap = scenario.get("solar_cumulative_cap_mw", None)` (1159) —
  per the default-reachability audit above, `load_scenario()` always sets this key
  (`None` specifically under `solar_build_case="deployment_unconstrained"`, a real
  per-year list otherwise), so this constraint is always built except under that one arm

#### `m.solar_min_build_constraint` — lines 1178–1182
- index: `T`
- conditional: **only if `scenario.get("solar_min_build_mw_per_year", 0.0) > 0`**
- rule body (verbatim):
```python
lambda m, t:
    m.solar_public_add[t] + m.solar_eaas_add[t] >= min_build
```
- source of data: `min_build = scenario.get("solar_min_build_mw_per_year", 0.0)` (1176) —
  the audit above shows `load_scenario()` always sets this to the flat literal `0.0`, so
  this constraint is DEAD (never built) in any run that goes through `load_scenario()`
  unmodified

#### `m.land_constraint` — lines 1140–1147
- index: `T`
- conditional: no
- rule body (verbatim):
```python
lambda m, t:
    m.solar_capacity_mw[t] * l_sol
    + m.gas_capacity_mw[t] * l_gas
    + m.storage_capacity_mwh[t] * l_sto
    <= A_max
```
- source of data: `A_max = scenario["land_available_km2"]` (1135); `l_sol =
  scenario["land_intensity_solar_km2_per_mw"]` (1136); `l_gas =
  scenario["land_intensity_gas_km2_per_mw"]` (1137); `l_sto =
  scenario["land_intensity_storage_km2_per_mwh"]` (1138)

### 5.7 Financing (public budget, concessional envelope, EaaS bankability, subsidy limit, disable, tranche split identities)

#### `m.public_budget_constraint` — lines 2005–2014
- index: scalar (`expr=...`)
- conditional: **only if `scenario.get("public_solar_budget_npv", None)` is not `None`**
- rule body (verbatim):
```python
expr=pyo.quicksum(
    m.DF[t] * (
        m.solar_public_add[t] * m.solar_capex_param[t]
        + m.storage_add[t] * econ["STORAGE_COST_PER_MWH"]
        + m.eaas_subsidy[t]
    )
    for t in T
) <= public_budget_npv
```
  and, immediately after, `m.has_public_budget_constraint = True` (2015 — a plain
  Python attribute set on the model, not a Pyomo component, for downstream diagnostics)
- source of data: `public_budget_npv = scenario.get("public_solar_budget_npv", None)`
  (2003) — per the audit above, `load_scenario()` always sets this key
  (`capital_envelope_scenarios()[capital_case]`, `None` specifically under
  `capital_case="unconstrained"`)

#### `m.f1_concessional_envelope_constraint` — lines 1712–1718 (`f1_tranche` only)
- index: scalar
- conditional: `f1_tranche` only — **no `is not None` guard**, per the file's own
  comment (1706–1710): `concessional_envelope_case`/`concessional_envelope_usd` is
  itself required-no-default at `load_scenario()`, so a missing envelope is an error
  there, not an unconstrained default reachable here
- rule body (verbatim):
```python
expr=(
    pyo.quicksum(m.solar_public_conc_vol[t] for t in T)
    + pyo.quicksum(m.solar_eaas_conc_vol[t] for t in T)
    + pyo.quicksum(m.storage_conc_vol[t] for t in T)
) <= _concessional_envelope_usd
```
- source of data: `_concessional_envelope_usd = scenario["concessional_envelope_usd"]`
  (1711). Gas is excluded from this sum entirely (envelope resolution Decision 1 —
  DARES is a distributed-renewables facility; a CCGT is not concessional-eligible)

#### `m.solar_public_split_identity` — lines 1657–1661 (`f1_tranche` only)
- index: `T`
- conditional: `f1_tranche` only
- rule body (verbatim):
```python
lambda m, t:
    m.solar_public_conc_vol[t] + m.solar_public_comm_vol[t]
    == m.solar_capex_param[t] * m.solar_public_add[t]
```

#### `m.solar_eaas_split_identity` — lines 1665–1669 (`f1_tranche` only)
- index: `T`
- conditional: `f1_tranche` only
- rule body (verbatim):
```python
lambda m, t:
    m.solar_eaas_conc_vol[t] + m.solar_eaas_comm_vol[t]
    == m.solar_capex_param[t] * m.solar_eaas_add[t]
```

#### `m.eaas_bankability` — lines 1104–1113
- index: `T`
- conditional: **only if `tariff is not None`**
- rule body (verbatim, `eaas_bankability_rule`, lines 1104–1111):
```python
def eaas_bankability_rule(m, t):
    capex = m.solar_capex_param[t] * m.solar_eaas_add[t]
    effective_private_cost = capex - m.eaas_subsidy[t]
    bankable_revenue = (
        tariff * solar_energy_per_mw_annual
        * m.remaining_npv_factor[t] * m.solar_eaas_add[t]
    ) / required_margin
    return effective_private_cost <= bankable_revenue
```
- source of data: `tariff = scenario.get("solar_service_tariff_usd_per_twh", None)` (497 —
  per the audit above, `load_scenario()` always sets this via `tariff_scenarios()[tariff_case]`,
  `tariff_case` itself REQUIRED); `solar_energy_per_mw_annual = scenario["solar_cf"] * 8760 /
  1_000_000` (1057); `required_margin = scenario["required_margin"]` (504);
  `m.remaining_npv_factor` (§2.1 `pyo.Param`, built from `resolve_private_rate(scenario)`)

#### `m.eaas_disable` — lines 1116–1118
- index: `T`
- conditional: **only if `financing_regime != "eaas"`**
- rule body (verbatim):
```python
lambda m, t: m.solar_eaas_add[t] <= 0.0
```
- source of data: `financing_regime = scenario.get("financing_regime", "traditional")`
  (498 — audit above: always set by `load_scenario()`, flat literal `"traditional"`)

#### `m.eaas_subsidy_limit` — lines 1120–1132
- index: `T`
- conditional: no (always built), **but the rule body itself branches on whether `tariff is not None`**
- rule body (verbatim, `tariff is not None`, `subsidy_limit_rule`, lines 1121–1127):
```python
def subsidy_limit_rule(m, t):
    max_bankable_at_t = (
        tariff * solar_energy_per_mw_annual
        * m.remaining_npv_factor[t] / required_margin
    )
    gap_at_t = max(0.0, capex_init[t] - max_bankable_at_t)
    return m.eaas_subsidy[t] <= gap_at_t * m.solar_eaas_add[t]
```
- rule body (verbatim, `tariff is None`, lines 1129–1130):
```python
def subsidy_limit_rule(m, t):
    return m.eaas_subsidy[t] <= 0.0
```
- source of data: `capex_init` — the same per-year dict used to build `m.solar_capex_param`
  (§2.1)

### 5.8 Policy (emissions cap, grid service obligation)

#### `m.emissions_constraint` — two mutually exclusive variants, lines 2027–2035
- index: `T` (per-year) or scalar (cumulative)
- conditional: **only if `emissions_cap_by_year is not None` OR `emissions_cap is not None`**
  (function arguments; `emissions_cap_by_year` checked first)
- rule body (verbatim, `emissions_cap_by_year is not None`, lines 2030–2032):
```python
lambda m, t: m.emissions_capped_by_year[t] <= emissions_cap_by_year[t]
```
- rule body (verbatim, `elif emissions_cap is not None`, line 2035):
```python
expr=m.emissions_capped <= emissions_cap
```
- source of data: `emissions_cap`/`emissions_cap_by_year` are `build_model_sliced()`
  **function arguments** (§2.2), both defaulting to `None` — i.e. **no emissions cap is
  built at all unless a caller explicitly passes one**; neither comes from `scenario`
  or `econ`

#### `m.grid_service_obligation` — two variants, lines 1200–1219
- index: `T` (`grid_service_mode == "annual"`) or scalar (`grid_service_mode == "total"`)
- conditional: **only if `max_offgrid_fraction is not None`** (function argument)
- rule body (verbatim, `grid_service_mode == "annual"`, lines 1201–1209):
```python
lambda m, t:
    pyo.quicksum(
        m.selfgen[t, (s, p)] + m.unserved[t, (s, p)]
        for (s, p) in S
    )
    <= _eps[t] * demand_annual[t]
```
- rule body (verbatim, `grid_service_mode == "total"`, lines 1211–1217):
```python
expr=pyo.quicksum(
    m.selfgen[t, (s, p)] + m.unserved[t, (s, p)]
    for t in T for (s, p) in S
)
<= sum(_eps[t] * demand_annual[t] for t in T)
```
- source of data: `max_offgrid_fraction` and `grid_service_mode` are both
  `build_model_sliced()` **function arguments** (§2.2; `max_offgrid_fraction` defaults to
  `None` — no obligation built at all unless a caller passes one; `grid_service_mode`
  defaults to `"annual"`); `_eps` (1190–1198, scalar or per-year dict from
  `max_offgrid_fraction`, accepting either form); note the commented-out alias line
  `# m.reliability_constraint = m.grid_service_obligation` (1221) — dead, never executed

**Count check:** `grep -c "pyo\.Constraint(" src/optimize_model_sliced.py`
= 34 call sites; all 34 accounted for above across 26 uniquely named
Constraints.

---

## Section 6 — Objective

Exactly one `pyo.Objective(` call site in the whole file.

#### `m.objective` — line 2037
- index: scalar
- conditional: no — **always built, always the same sense**
- rule body (verbatim):
```python
m.objective = pyo.Objective(expr=system_cost_npv, sense=pyo.minimize)
```
- source of data: `system_cost_npv` — the local variable assembled in §4.6
  (one of two branches depending on `capital_treatment_case`; also assigned,
  unchanged, to `m.system_cost_npv` at line 1949)
- **No alternative objective sense and no weighted-sum branch exist at
  HEAD.** The file's own comment (2020–2026) states the previous
  `weight_cost * system_cost_npv + weight_emissions * m.emissions`
  weighted-sum branch was deleted as dead code: "no caller anywhere in
  `src/` or `scripts/` ever set `weight_cost` or `weight_emissions`
  away from their old defaults (1.0 / 0.0)." The objective is
  unconditionally cost-minimising; only whether an emissions constraint
  is *also* built (§5.8) varies with the cap arguments.

---

## Also report — differences from `build_model` (annual, `src/optimize_model.py:30-1756`)

Read in full for this comparison. Both builders share the same guard
block, the same F1-tranche financing machinery (CRF, split identities,
concessional envelope), and the same objective structure almost
verbatim — the differences are concentrated in dispatch, hydro, storage
and the conversion mechanism.

| component | `build_model_sliced` | `build_model` (annual) |
|---|---|---|
| **Time index** | `T` (years) `×` `m.S` (6 season/period slices) for dispatch; `T` alone for capacity/finance | `T` alone, everywhere — no slice dimension exists |
| **Conversion mechanism (`_conv`)** | Fully implemented: `m.convert_mw`, `m.convert_to_power`, `m.convert_generation`, `m.convert_capacity_constraint`, `m.convert_comm_vol`, `m.convert_split_identity`, `m.cost_conversion_capital_charge_npv`, `m.conversion_overnight_capex_usd_by_year` — present only if `_conv` | **Does not exist.** `optimize_model.py:295-302` **raises** `ValueError` if `scenario.get("conversion_available", False)` is truthy: "build_model() (annual) does not implement OCGT-to-CCGT conversion — it is slice-resolved (Phase 2.8, Decision 3) and lives in `build_model_sliced()` only." |
| **`m.hydro_gen`** | A `pyo.Var(T, m.S, ...)` — hydro is DISPATCHED, subject to three constraints (`hydro_season_budget`, `hydro_power_limit`, `hydro_min_flow`) | **Does not exist as a Pyomo component at all.** Hydro is a plain Python dict, `hydro = {t: float(hydro_energy_twh_by_year[t]) for t in T}` (line 612), read directly into `m.balance`'s rule body as the literal `hydro[t]` (line 934) — no Var, no Expression, no Constraint, no seasonal shape, no minimum-flow floor |
| **Gas capacity constraint** | `m.gas_generation[t,(s,p)] <= (m.gas_capacity_mw[t] [- converted nameplate]) * H[(s,p)] * gas_availability_factor / 1e6` — per-slice hours `H[(s,p)]` | `m.gas_generation[t] <= m.gas_capacity_mw[t] * 8760 * gas_availability_factor / 1e6` (line 636) — flat annual hours, no slice structure, no conversion netting term |
| **Gas balance (fuel budget)** | Annual budget summed across all 6 slices: `pyo.quicksum(m.gas_to_power[t,(s,p)] for (s,p) in S) + [_converted_thermal_year] + m.gas_curtail[t] <= gas_avail[t] [+ exempt allowance]` | Same structure without the slice sum or the conversion term: `m.gas_to_power[t] + m.gas_curtail[t] <= gas_avail[t] [+ exempt allowance]` (lines 672-683) |
| **Storage formulation** | **Slice-resolved, seasonal.** Three constraints: `storage_season_balance` (discharge ≤ RTE·charge, summed within the SAME season only), `storage_charge_power_limit`, `storage_discharge_power_limit` (both `<= m.storage_power_mw[t] * H[(s,p)] / 1e6`, per slice). No `storage_deployable_hours` parameter exists in this file at all. | **Annual energy-throughput model**, per the file's own heading (lines 828-846): three DIFFERENTLY NAMED constraints — `storage_charge_power_limit` (`<= (m.storage_power_mw[t] * storage_deployable_hours) / 1e6`), `storage_energy_balance` (discharge `<= RTE * charge`, no seasonal split — the sliced builder has no directly-corresponding name; its nearest analogue is `storage_season_balance`), `storage_power_limit` (discharge side; **note the name — this is `build_model_sliced`'s `storage_discharge_power_limit` under a different name**). Depends on `storage_deployable_hours = scenario.get("storage_deployable_hours_per_year", 700.0)` (line 384) — **a parameter the sliced builder never reads at all**, replaced structurally by the slice/season partition itself (module docstring: "storage_deployable_hours is NO LONGER a free parameter — throughput now falls out of the slice structure") |
| **Electricity balance (`m.balance`)** | Per-slice, 6 equations/year; two variants by `_conv` (adds converted generation term) | One equation/year; `hydro[t]` enters as a literal, not `m.hydro_gen[t,(s,p)]`; no conversion term possible (mechanism doesn't exist) |
| **`m.eaas_disable`** rule body | `m.solar_eaas_add[t] <= 0.0` (optimize_model_sliced.py:1117) | `m.solar_eaas_add[t] == 0` (optimize_model.py:780, via `eaas_disable_rule`) — same domain (`NonNegativeReals`) makes these equivalent in effect, but they are literally different rule bodies |
| **Grid service obligation** | `pyo.quicksum(m.selfgen[t,(s,p)] + m.unserved[t,(s,p)] for (s,p) in S) <= _eps[t]*demand_annual[t]` (annual mode) — sums over 6 slices per year | `m.selfgen[t] + m.unserved[t] <= _eps[t]*demand[t]` (line 969-974) — no slice sum, single term per year |
| **Firm capacity constraint** | Two variants by `_conv`; the `_conv` variant adds `pyo.quicksum(_firm_credit[tier] * _conv_uplift * m.convert_mw[tier] for tier in _CONV_KEYS)` | One variant only (conversion term cannot exist) — otherwise byte-identical formula, same `within_block_peaking_factor` default (1.3448), same guard logic |
| **Emissions expressions** | `m.emissions_grid_only_by_year` has two variants (`_conv` adds `_converted_gen_year(m,t) * 1e6 * _ef_e_retrofit`); reads `m.gas_generation[t,(s,p)]` summed over `S` | One variant; reads `m.gas_generation[t]` directly (no slice sum); `ef_e` computed as `econ["EF_TCO2_PER_MWH_TH"] / scenario["gas_eta"]` (line 1560) rather than reusing the earlier local `eta` variable — same value, redundant second read |
| **O&M (`compute_om_cost_npv`)** | Passes `gas_unconverted_capacity_mw`/`gas_converted_capacity_mw` as **per-year dicts**, non-zero converted values possible if `_conv` | Passes `gas_unconverted_capacity_mw=m.gas_capacity_mw` (the Expression itself, not a plain dict) and `gas_converted_capacity_mw={t: 0.0 for t in T}` — always zero, conversion cannot occur |
| **`m.slice_hours`** | Present (`dict(H)`, line 284) | Does not exist — no slice concept |
| **F1-tranche financing machinery** (`_crf`, `_L_eff_conc/comm`, `_df_window_conc/comm`, split-identity Vars/Constraints, `m.f1_concessional_envelope_constraint`, capital-charge Expressions) | Present, `f1_tranche` only | **Present and materially identical** (same formulas, same `_crf`/`_rebuild_years` closures, same variable names) — this machinery was already annual-only data even inside the sliced builder, so slicing does not touch it. `m.convert_comm_vol`/`m.cost_conversion_capital_charge_npv` are the one addition, gated on `_conv`, which cannot exist in `build_model` |
| **`solve_model()` / `leakage_diagnostic()`** | Not defined in this file | Defined here (`optimize_model.py:1759`, `:1817`) and used as the shared post-solve wrapper regardless of which builder produced the model — not part of either `build_model*` function itself, so not a "component" difference in the strict sense, but the only copy exists in the annual builder's file |
| **Post-solve D6 rebuild-invariant guard** | N/A (lives in `solve_model()`) | `optimize_model.py:1891-1982`, inside `solve_model()`. Its own comment states its `_rebuild_years` logic is REPLICATED (not shared) across four sites: `optimize_model.py:1137` (pre_f1_overnight), `:1331` (f1_tranche), `optimize_model_sliced.py:1366` (pre_f1_overnight — this file's own `_rebuild_years`, §4.4), `optimize_model_sliced.py:1552` (f1_tranche, §4.5) — "all four are byte-identical today," per that comment, but verbatim duplicates, not one shared function |

## Also report — hardcoded numeric literals inside rule bodies (`build_model_sliced` only)

Excludes literals already catalogued as scenario-key defaults in §2.3
(e.g. `1.3448`, `0.90`, `0.50`, the season-factor dicts) — cross-referenced
here instead of repeated. Excludes trivial loop/index arithmetic
(`range(0, t+1)`, `+1`, `-1`).

| literal | meaning | line(s) |
|---|---|---|
| `1.3448` | `within_block_peaking_factor` builder default — **see §2.3, register L50 / forensic audit A62** | 1263 |
| `1e6` | Unit-conversion constant, MW·h → TWh (or the reverse), applied directly inside nine separate rule bodies rather than a named parameter: `gas_capacity_constraint` (×2, `_conv`/non-`_conv`, lines 859, 877), `convert_capacity_constraint` (870), `gas_exempt_fuel_allowance_twh_th` (912), `hydro_power_limit` (971), `storage_charge_power_limit` (1006), `storage_discharge_power_limit` (1013), `emissions_grid_only_by_year` (×2, lines 1874-1875, 1882), `emissions_selfgen_by_year` (1888) | 859, 870, 877, 912, 971, 1006, 1013, 1874, 1875, 1882, 1888 |
| `1_000_000` | Same unit-conversion constant, written with underscores instead of scientific notation — two call sites carry the identical conversion as `1e6` under a different literal spelling: `_solar_energy_per_mw_by_slice`'s `annual_mwh_per_mw * share / 1_000_000` (200) and `solar_energy_per_mw_annual = scenario["solar_cf"] * 8760 / 1_000_000` (1057) | 200, 1057 |
| `8760` | Hours per year. Used twice inside rule-adjacent code: `annual_mwh_per_mw = cf_annual * 8760` (187, inside `_solar_energy_per_mw_by_slice`) and, more consequentially, `gas_exempt_fuel_allowance_twh_th`'s `m.gas_new_capacity_mw[t] * 8760 * gas_availability_factor / gas_new_efficiency / 1e6` (911) — the `new_gas_deliverability="exempt"` arm sizes new capacity's fuel allowance at full-YEAR running (8760 h), not per-slice availability, which `methodology_physical_layer.md` §4.4 already flags as "an upper bound rather than a realistic alternative" | 187, 911 |
| `1000.0` | kW → MW/USD-per-kW scaling for conversion capex, three call sites: `convert_split_identity`'s `_conv_capex_per_kw * 1000.0 * m.convert_mw[tier]` (1826), the diagnostic `m.conversion_overnight_capex_usd_by_year`'s identical product (1997), and `conversion_diagnostic()`'s post-solve `implied_conversion_capex_usd` (2157, outside `build_model_sliced` itself, in the module-level diagnostic helper) | 1826, 1997, 2157 |
| `1.0` (additive) | Appears as the `+1` term in two structurally distinct "gross-up" multiplications, not a single reusable constant: `(1.0 + _conv_uplift)` in `convert_capacity_constraint` (869) and `(1.0 + _reserve_margin_by_year[t])` in `m.firm_capacity_constraint` (both variants, 1307, 1318) | 869, 1307, 1318 |
| `0.0` (as an explicit fix/bound, not a domain floor) | `m.gas_conc_vol[_t].fix(0.0)` for every `t`, unconditional within `f1_tranche` (1682) — the mechanical implementation of "gas is commercial-only" | 1682 |
