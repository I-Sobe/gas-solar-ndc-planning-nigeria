# Chapter 3.1 — verification against `docs/formulation_inventory_sliced.md` and HEAD

**Read-only.** Nothing in `src/`, `scripts/` or `docs/chapters/` was
edited or run to produce this. Draft checked:
`docs/chapters/chapter3_1_draft.md`. Formulation reference checked:
`docs/formulation_inventory_sliced.md` (states it is current as of HEAD
`17520ae`). Repo HEAD at the time of this check: `17520ae` (confirmed
via `git rev-parse HEAD`) — the inventory is current, not stale, so
every verdict below is cross-checked directly against
`src/optimize_model_sliced.py`/`src/scenarios.py` as well as the
inventory, not the inventory alone.

---

## Job 1 — Equations (3.1)–(3.20)

| eq. | verdict | note |
|---|---|---|
| 3.1 | MATCHES | Not from the sliced builder — this is the illustrative annual-model inequality the draft uses to motivate slicing (§3.1.2), and its own text carries no `[code:]` tag. It corresponds to the **annual** builder's `m.storage_energy_balance` (`optimize_model.py:867-872`): `m.storage_discharge[t] <= storage_round_trip_eff * m.storage_charge[t]`, i.e. exactly $d_t \le \eta^{rt} q_t$. |
| 3.2 | MATCHES | Algebraic consequence of 3.1 substituted into the annual balance term `+ discharge - charge`; not itself a separate code rule body. Consistent with `optimize_model.py:929-940`'s `m.storage_discharge[t] - m.storage_charge[t]` term. |
| 3.3 | MATCHES | `gas_cap_rule` (`optimize_model_sliced.py:690-699`, `m.gas_capacity_mw` at 701) quoted verbatim in the inventory §4.1. The draft's $R_t$ ("cumulative retired baseline gas capacity") stands for the code's three-way branch (`retirement_series` / `retirement_start`+`retirement_rate` / `0.0`) collapsed into one symbol — legitimate as notation, but **$R_t$'s own definition is never spelled out** as a sub-equation; Chapter 3 should either give it one or footnote the branch. |
| 3.4 | MATCHES | `solar_cap_rule` (`:740-743`, `m.solar_capacity_mw` at 745), quoted verbatim, identical structure: $K^{sol}_t=\bar K^{sol}_0+\sum_{\tau\le t}(x^{pub}_\tau+x^{eaas}_\tau)$. |
| 3.5 | MATCHES | `storage_cap_rule` (`:730-731`, `m.storage_capacity_mwh` at 733) and `m.storage_power_mw` (`:734-736`) both quoted verbatim; identical. |
| 3.6 | MATCHES | `m.gas_generation` (`:803-805`), `m.convert_generation` (`:810-814`), `m.solar_generation` (`:830-833`), all quoted verbatim; identical three-way structure. |
| 3.7 | MATCHES | This is the **`f1_tranche`** (central-configuration) branch of `m.system_cost_npv` (`optimize_model_sliced.py:1941-1948`, tagged in inventory §4.6): `gas_opex_npv + m.capital_charge_total_npv + selfgen_npv + unserved_npv + carbon_cost_npv + om_cost_npv` — a term-for-term match to $C^{fuel}+C^{cap}+C^{self}+C^{uns}+C^{CO_2}+C^{O\&M}$. (The alternative `pre_f1_overnight` branch, §4.6, has a different, longer term list — draft correctly scopes 3.7 to "at the central configuration.") |
| 3.8 | MATCHES | `gas_opex_npv` (`:1326-1329`, extended `:1338-1342` if conversion is active) quoted verbatim. The draft's $\sum_s c^{gas}(f_{t,s}+\sum_k f^c_{k,t,s})$ and the code's `pyo.quicksum(...f_{t,s}...) + pyo.quicksum(...f^c...)` are the same total under a different association of the sums (over $s$ first vs. over $t$ first) — not a discrepancy. |
| 3.9 | MATCHES, with a parameter-table gap | `unserved_npv` (`:1840-1843`) and `selfgen_npv` (`:1845-1848`) quoted verbatim; both match. **But** $c^{self}$ is defined in the draft's notation table (§3.1.3) simply as "genset cost." The code's `selfgen_price = econ.get("GENSET_COST_PER_TWH_E", voll)` (`:551`) is genset cost **only when a genset case is priced**; under the commit-A regression configuration (`genset_case=None`), `GENSET_COST_PER_TWH_E` is absent and `selfgen_price` silently equals `voll` — i.e. $c^{self}=c^{VoLL}$ and the two terms in 3.9 become numerically identical (this is the deliberate regression-test degeneracy the source comments describe, not a bug). Chapter 3 should state this conditionality, not define $c^{self}$ as unconditionally "genset cost." |
| 3.10 | MATCHES, with an undefined symbol | `carbon_cost_npv` (`:1911-1916`) quoted verbatim; matches $\sum_t\delta_t\pi^{CO_2}E_t$ with $E_t=$ `m.emissions_by_year[t]` (the **total**, grid + self-generation — confirmed against the code, which reads `m.emissions_by_year[t]`, not `m.emissions_capped_by_year[t]`). **$E_t$ itself is never introduced in the §3.1.3 notation table** — only $\bar E_t$ (the cap) appears there. Add $E_t = G^g_{t}$-derived total emissions (`m.emissions_by_year`) to the notation table; see also 3.20b below for the companion gap on $E^{cap}_t$. |
| 3.11 | MATCHES | `m.balance`, `_conv=True` variant (`:1026-1039`), quoted verbatim, term-for-term: $G^g+\sum_k G^c+G^{sol}+h+d-q+\beta g+u\ge D$. The draft states this as the general form; the `_conv=False` variant (`:1041-1052`) is textually identical minus the $\sum_k G^c_{k,t,s}$ term — worth a one-line footnote in Chapter 3 rather than a second numbered equation. |
| 3.12 | **MATCHES — resolves the draft's own "VERIFY exact form" flag** | `m.gas_capacity_constraint`, `_conv=True` variant (`:854-860`): `m.gas_generation[t,(s,p)] <= (m.gas_capacity_mw[t] - _converted_ocgt_nameplate) * H[(s,p)] * gas_availability_factor / 1e6`, where `_converted_ocgt_nameplate = pyo.quicksum(m.convert_mw[tier] for tier in _CONV_KEYS)` (`:851-853`) $=\sum_k y_k$. Exact match to $G^{g}_{t,s}\le(K^{g}_t-\sum_k y_k)H_s\alpha/10^{6}$. |
| 3.13 | **MATCHES — resolves the draft's "VERIFY" flag** | `m.gas_balance`, `constrained`+`_conv` variant (`:935-941`): `pyo.quicksum(f_{t,s}) + Σ_k Σ_s f^c_{k,t,s} + γ_t <= gas_avail[t]`. Exact match. **The exempt-arm addendum the draft flagged as missing**: under `new_gas_deliverability="exempt"`, the RHS gains `+ m.gas_exempt_fuel_allowance_twh_th[t]` (`:915-922`), where `m.gas_exempt_fuel_allowance_twh_th[t] = m.gas_new_capacity_mw[t] * 8760 * gas_availability_factor / gas_new_efficiency / 1e6` (`:908-913`). In the draft's symbols: $A_t \to A_t + K^{g,new}_t\cdot 8760\cdot\alpha/\eta^{new}/10^6$, where $K^{g,new}_t=\sum_{\tau\le t}x^g_\tau$ (cumulative new gas only, `m.gas_new_capacity_mw`, `:905-907`). Note the flat `8760` here (full-year running), not the per-slice `H_s` used everywhere else in this file — already flagged in the inventory ("Also report — hardcoded literals") and in `methodology_physical_layer.md` §4.4 as "an upper bound rather than a realistic alternative." |
| 3.14 | MATCHES | `m.convert_mw` bounds (`:659-663`): `bounds=lambda m, tier: (0.0, _conv_tiers[tier]["nameplate_mw"])`. Exact match to $0\le y_k\le\bar Y_k$. |
| 3.15 | **MATCHES — resolves "VERIFY"** | `m.storage_season_balance` (`:994-1000`), quoted verbatim: $\sum_p d_{t,(\sigma,p)}\le\eta^{rt}\sum_p q_{t,(\sigma,p)}$. Exact match. |
| 3.16 | **MATCHES — resolves "VERIFY"** | `m.storage_charge_power_limit` (`:1002-1007`) and `m.storage_discharge_power_limit` (`:1009-1014`), both quoted verbatim, both `<= m.storage_power_mw[t] * H[(s,p)] / 1e6`. Exact match to both halves of 3.16. |
| 3.17 (season budget) | **DIFFERS** | The draft writes $\sum_p h_{t,(\sigma,p)} \le E^h_t\theta_\sigma$ (an inequality). The actual code, `m.hydro_season_budget` (`optimize_model_sliced.py:956-961`), is an **equality**: `pyo.quicksum(m.hydro_gen[t,(s,p)] for (s,p) in S if s == season) == hydro_annual[t] * hydro_season_share[season]`. Correct form: $\sum_p h_{t,(\sigma,p)} = E^h_t\theta_\sigma$. This is not a rounding nuance — the seasonal hydro energy budget is fixed exactly, not merely capped, since hydro is exogenous total energy that must all be dispatched somewhere across the six slices. |
| 3.17 (power limit) | MATCHES | `m.hydro_power_limit` (`:967-972`), quoted verbatim: $h_{t,s}\le K^h_t H_s/10^6$. Exact match. |
| 3.17 (minimum flow) | **Transcribed — was blank in the draft** | `m.hydro_min_flow` (`:979-985`): `m.hydro_gen[t,(s,p)] >= hydro_min_frac * hydro_annual[t] * hydro_season_share[s] * (H[(s,p)] / season_hours[s])`, where `season_hours = {s: sum(H[(ss,p)] for (ss,p) in S if ss==s) for s in ("dry","wet")}` (`:976-977`) is total hours in season $s$. In the draft's own symbols (reusing $\theta_\sigma$ for `hydro_season_share` and introducing $H_\sigma=\sum_{p'}H_{(\sigma,p')}$ for the season's total hours): $$h_{t,s}\;\ge\;\mu\,E^{h}_t\,\theta_\sigma\,\frac{H_s}{H_\sigma}.$$ $\mu=$ `hydro_min_flow_fraction` = 0.50 — the draft's stated value is correct (confirmed: `scenarios.py` sets it unconditionally to the flat literal `0.50`, register/audit terminology "DEAD default," §2.3 of the inventory). |
| 3.18 | MATCHES | `m.firm_capacity_constraint`, `_conv=True` variant (`:1295-1308`), quoted verbatim, exact term-for-term match including the conversion addendum $\sum_k\kappa^c_k\,\upsilon\,y_k$ (`_firm_credit[tier] * _conv_uplift * m.convert_mw[tier]`). |
| 3.19a | **MATCHES — resolves "VERIFY"** | `m.solar_min_build_constraint` (`:1178-1182`) and `m.solar_cumulative_cap`/`solar_cum_cap_rule` (`:1167-1174`), both quoted verbatim, both match exactly. Two notes: (i) $\underline{x}_t$ is written with a $t$ subscript in the draft but `min_build` is a **flat scalar**, not per-year — harmless as notation but worth a footnote; (ii) per the inventory's default-reachability audit, `load_scenario()` always sets `solar_min_build_mw_per_year=0.0`, so this half of 3.19a is **never built** in any run that goes through `load_scenario()` unmodified (the constraint requires `min_build > 0`, `:1177`) — Chapter 3 should say so, since a reader could otherwise assume it always binds. |
| 3.19b | **MATCHES — resolves "VERIFY"** | `m.land_constraint` (`:1140-1147`), quoted verbatim: exact match. |
| 3.20a | **MATCHES — resolves "VERIFY"** | `m.grid_service_obligation`, `grid_service_mode="annual"` variant (`:1201-1209`), quoted verbatim: exact match, with `demand_annual[t]` (the **annual** total, `:797`-derived) standing for $D_t$ — correctly distinct from the per-slice $D_{t,s}$ used in 3.11. Not shown in the draft: a second, `grid_service_mode="total"` variant (`:1211-1217`) sums the obligation over the **whole horizon** rather than year-by-year — worth a footnote since it changes what the constraint permits (front-loading vs. back-loading of self-generation) even though the per-period sums are identical. |
| 3.20b | **MATCHES — resolves "VERIFY"**, with an undefined symbol | `m.emissions_constraint`, `emissions_cap_by_year` variant (`:2030-2032`), quoted verbatim: $E^{cap}_t\le\bar E_t$ exactly, where $E^{cap}_t=$ `m.emissions_capped_by_year[t]` — **grid-only by default**, total only if `selfgen_in_emissions_cap=True` (a `build_model_sliced()` call argument, not a scenario key). As with 3.10's $E_t$, **$E^{cap}_t$ is used but never formally entered in the §3.1.3 notation table** — the prose right after 3.20b defines it in words only. A second form exists when only a scalar cap is supplied: `m.emissions_capped <= emissions_cap` (`:2035`, cumulative over the whole horizon) — not shown in the draft, worth a footnote. |

**Summary:** 19 of the 20 numbered equations MATCH the code exactly (several resolving the draft's own "VERIFY" flags in the process); **one — the hydro seasonal budget half of 3.17 — DIFFERS**: the draft's $\le$ should be $=$. The hydro minimum-flow term, left blank, is transcribed above. Two notation gaps ($E_t$, $E^{cap}_t$ never defined in §3.1.3) are flagged as drafting completeness issues, not code-matching failures.

### Capital charge discounting — confirmed

**Yes, the capital charge enters the objective discounted by `m.DF[t]`,** but through a **window-sum** of `m.DF[t]`, not a single multiplication at the vintage's build year. Mechanism, read directly from `optimize_model_sliced.py:1619-1632`:

```python
_df_window_conc = {
    ch: {
        v: sum(float(pyo.value(m.DF[t])) for t in range(v, min(v + _L_eff_conc[ch], _T_last + 1)))
        for v in T
    }
    for ch in ("solar", "storage", "gas")
}
# _df_window_comm built identically from _L_eff_comm
```

Each vintage $v$'s capital charge is `_crf_conc[ch] * _df_window_conc[ch][v] * (concessional volume at v)` (e.g. `m.cost_public_solar_capital_charge_npv`, `:1738-1744`) — i.e. the annuity (CRF) times the **sum of `m.DF[t]` over the vintage's own recovery window** `[v, v+L_eff-1]`, truncated at `T_last`. This is the annuity-times-discount-factor-sum structure standard to a levelised capital charge, and it is where `m.DF[t]` — the same `pyo.Param` used everywhere else in the objective — enters. All per-vintage charges are then summed via `pyo.quicksum(... for v in T)` and combined into `m.capital_charge_total_npv` (`:1838`), which itself enters `m.system_cost_npv` undiscounted a second time (it already carries its own discounting internally) — confirmed by reading `optimize_model_sliced.py:1941-1948` directly: `m.capital_charge_total_npv` is added to `system_cost_npv` with no further `m.DF[t]` multiplication, exactly as expected since the discounting is already embedded per-vintage.

### `L_conc`, `L_comm`, `r_conc_nominal`, `r_comm_nominal`, and the Fisher-deflated real rates at the central configuration

Verified directly against `src/scenarios.py` (not taken from memory or the inventory alone):

```
src/scenarios.py:1899
"cr_central": {"r_conc": 0.0280, "L_conc": 30, "r_comm": 0.11, "L_comm": 10},
```

`capital_recovery_case="cr_central"` is the central configuration's arm (confirmed as the default fed by `scripts/run_central_sweep.sh`'s central-configuration block). At the central configuration:

| symbol | value | source |
|---|---|---|
| `L_conc` | 30 years | `scenarios.py:1899` |
| `L_comm` | 10 years | `scenarios.py:1899` |
| `r_conc_nominal` | 0.0280 (2.80%) | `scenarios.py:1899` |
| `r_comm_nominal` | 0.11 (11.00%) | `scenarios.py:1899` |
| `us_inflation` (central) | 0.020 (2.00%) | `scripts/run_central_sweep.sh:102`, `C_INFL="0.020"`; CLI-required, no `load_scenario()` default (`scripts/01b_run_baseline_sliced.py:195-198`, `--us-inflation`, `required=True`) |

Fisher deflation (`optimize_model_sliced.py:1565-1566`):
```python
_r_conc_real = (1.0 + _r_conc_nom) / (1.0 + _us_inflation) - 1.0
_r_comm_real = (1.0 + _r_comm_nom) / (1.0 + _us_inflation) - 1.0
```

At the central configuration:

$$r^{conc}_{real} = \frac{1.0280}{1.020} - 1 = 0.0078431\ldots \approx \mathbf{0.784\%}$$

$$r^{comm}_{real} = \frac{1.11}{1.020} - 1 = 0.0882353\ldots \approx \mathbf{8.824\%}$$

Both real rates then feed `_crf(r, L)` (`:1540-1555`) with the F2 asset-lifetime ceiling `L_eff = min(L_tranche, asset_lifetimes[channel])` (`:1596-1603`) to produce `_crf_conc`/`_crf_comm` per channel — the CRFs quoted in the 3.7/§3.4 capital-charge expressions above.

---

## Job 2 — Citations for the draft's "Pending" list

Searched `docs/*.md`, `src/*.py`, `environment.yml`, and the `source`/`notes`
columns of `data/cost/processed/*.csv`. No citation is invented; where
none exists, reported as "not in repo."

| item | citation status | evidence |
|---|---|---|
| **CBC solver (COIN-OR)** | **Not in repo.** | Mentioned only as a solver name in code comments (`src/optimize_model.py:1852-1868`, documenting CBC's tolerance defaults) — no author, title, version or URL anywhere. CBC is not even listed in `environment.yml` (`grep -i "cbc\|coin" environment.yml` — no match), consistent with the Phase 2 methodology audit's independent finding that the environment spec omits both solvers. |
| **Fisher relation for deflating nominal rates** | **Not in repo.** | `grep -rln "Fisher" docs/*.md src/*.py` — no match anywhere in the repository. The mechanism itself is implemented (`optimize_model_sliced.py:1565-1566`, `_r_real = (1+r_nom)/(1+inflation) - 1`) and documented in prose (`methodology.md` and elsewhere describe "deflated once, by a stated Fisher relation") but no citation to Fisher (or any source for the relation) is recorded. |
| **NREL Annual Technology Baseline 2024 (O&M, capex)** | **Recorded as a data-file provenance note, not a formal citation.** | `data/cost/processed/om_cost.csv`, `source` column: `"NREL ATB 2024, 2024_v3_Workbook.xlsx, Summary_FOM"` (every row, e.g. line 2). `data/cost/processed/solar_capex.csv`, `notes` column: `"NREL ATB 2024, Utility-Scale PV Class 5, Market, 2025"` (line 2). Narrative mentions without a full citation also appear at `docs/methodology.md:104,266,273` and `docs/methodology_physical_layer.md:473` ("NREL ATB 2024 technology characterisation, new-build H-class..."). No author list, report title, publisher or URL is recorded anywhere — a formal reference entry (as the draft's Verified list uses for Bynum et al., Hart et al., etc.) does not yet exist and would need to be constructed from these provenance notes plus the actual NREL ATB documentation. |
| **NIRP 2024, §2.3, p.19 (900 MW spinning reserve)** | **Not in repo as a formal citation; extensively cited informally by section/page.** | E.g. `docs/methodology_physical_layer.md:205` ("NIRP 2024 §2.3 (p.19) specifies spinning reserve of 900 MW"), `:443` ("NIRP does not state whether its gas investment is open or combined cycle"), `:113` ("NIRP 2024 contains no daily load profile"). Dozens of further section/page citations exist across `docs/methodology*.md`. None carries a full bibliographic form (report title, issuing body, publication date, URL). The draft's own Pending note is accurate: the primary source is stated to be in Zeus's possession, not yet transcribed into the repo as a reference entry. |
| **Capacity credit / capacity value methodology** | **Not in repo — explicitly marked as an open sourcing gap in the code itself**, not merely absent. | `src/scenarios.py:2936-2939`: `"solar": 0.0` (structural, not a citation — solar is credited zero because it cannot generate in the peak block, not because of a sourced capacity-value study), `"gas": _gas_capacity_credit,  # SOURCE: [PENDING]`, `"hydro": _hydro_dry_avg_mw / _hydro_capacity_mw,  # SOURCE: [PENDING]`. Storage's credit is a swept, declared-judgement arm (`docs/methodology_physical_layer.md` §3), not a sourced figure either. No capacity-credit/capacity-value methodology reference (e.g. an ELCC or effective-load-carrying-capability study) is cited anywhere in `docs/*.md`. |
| **Value of lost load (VoLL)** | **Not in repo.** | `data/cost/processed/unserved_energy_penalty.csv` carries only bare figures — `year,scenario,voll_usd_per_kwh,voll_usd_per_twh` (e.g. `voll_low` = 5 USD/kWh, `voll_mid` = 10, `voll_high` = 30) — with **no `source` or `notes` column at all**, unlike the solar/storage/O&M cost files. `docs/methodology_claims_ledger.md:525` compares these figures against the genset arms but cites no source for the VoLL numbers themselves. No VoLL methodology or study is referenced anywhere in `docs/*.md`. |

**Summary:** of the six Pending items, **four have no citation anywhere in the repository** (CBC, Fisher relation, capacity credit, VoLL — capacity credit's gap is explicitly self-declared in code as `SOURCE: [PENDING]`, the other three are silently absent); **two (NREL ATB 2024, NIRP 2024) have informal, repeated provenance mentions but no formal bibliographic entry** — the underlying primary sources exist (per the draft's own note, NIRP 2024 is "in Zeus's possession"; NREL ATB 2024's specific workbook/table is named in the CSV files) but have not yet been transcribed into a citable reference.

