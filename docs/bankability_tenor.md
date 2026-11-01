# bankability_tenor.md — 3.4: bankability tenor and rate coherence

Live runners in scope for wiring (Step 1): `03_run_ndc_eaas.py`,
`26_run_f3_identification.py`. All other tariff-setting runners are dead
on unrelated required args (see `docs/tariff_arm.md`, Steps 1–3) and take
`bank_tenor_legacy` threaded through `load_scenario()`.

---

## STEP 0 — Inventory (locate and quote, NO change)

### (a) `resolve_private_rate()` and what it returns under the central config

**Full function** — `src/blended_finance.py:168-187`:

```python
def resolve_private_rate(scenario: dict) -> float:
    keys = ("concessional_share", "r_concessional", "r_commercial")
    if all(k in scenario and scenario[k] is not None for k in keys):
        return blended_wacc(
            concessional_share=scenario["concessional_share"],
            r_concessional=scenario["r_concessional"],
            r_commercial=scenario["r_commercial"],
        )
    return float(scenario.get("private_discount_rate", 0.18))
```

`blended_wacc` — `src/blended_finance.py:67-113`:
`share_conc * r_conc + (1 - share_conc) * r_comm`.

**Which branch under the central configuration** (`cr_central`,
`dscr_central`, `env_dares`, `blend_case` default):

`load_scenario()` sets all three blend keys **unconditionally**
(`src/scenarios.py:2644` `conc_share, r_conc, r_comm =
blended_finance_scenarios()[blend_case]`, written into the scenario dict
at `src/scenarios.py:2931-2933`). None is ever `None`. Neither `03` nor
`26` overrides them, and neither exposes a `--blend-case` flag, so
`blend_case = "blended_central"` (default, `src/scenarios.py:2016`).

⇒ **the blended-WACC branch is ALWAYS taken.** The
`private_discount_rate` fallback is never reached; the `0.18` literal
default in `resolve_private_rate` is never reached.

`blended_finance_scenarios()` — `src/scenarios.py:105-111`:

| blend_case | (conc_share, r_conc, r_comm) | blended WACC |
|---|---|---|
| `commercial_only` | (0.00, 0.06, 0.18) | 0.180 |
| **`blended_central`** (default) | **(0.30, 0.06, 0.18)** | **0.144** |
| `blended_optimistic` | (0.50, 0.06, 0.18) | 0.120 |
| `concessional_heavy` | (0.75, 0.06, 0.18) | 0.090 |

**REPORTED NUMBER: `r_private = resolve_private_rate(central) = 0.30·0.06
+ 0.70·0.18 = 0.144` (14.4% real, per year).**

This is **not** in the 6–12% band TASK.md hypothesised — it is higher
than either. The consequence at the earliest vintage is therefore
smaller than TASK's 33–61% (see below); the incoherence is structural,
not magnitude-gated.

**Is the retired 0.18 naira anchor live in the bankability constraint?**
YES — as `r_commercial`, not as the fallback default. Every entry of
`blended_finance_scenarios()` hard-codes `r_comm = 0.18`, described in
its own docstring as the "IEA/CCSI 'Africa power-sector WACC > 18%'"
anchor and flagged elsewhere as naira-basis / BASIS-SUSPECT
(`scripts/rpublic_diagnostic.py:74,168`,
`src/blended_finance.py:46-49`). So the 14.4% blend that discounts
`m.remaining_npv_factor` is 70% composed of the exact rate F1 retired
from the objective. **This is the third defect in the same expression,
as TASK.md anticipated** — it just arrives through `r_commercial` inside
`blended_wacc`, not through the `0.18` default in `resolve_private_rate`.

Additionally: `scenario["private_discount_rate"]` is set to `r_comm`
(= 0.18) at `src/scenarios.py:2934` but is **dead** — only
`resolve_private_rate`'s unreachable fallback would read it.

**Magnitude at t = 0** (hand-computed, to be confirmed by Step 1
verification). `remaining_npv_factor[0]` is an ordinary annuity factor
`a(r,n) = (1-(1+r)^-n)/r`:

- current (`bank_tenor_legacy`): `a(0.144, 30) ≈ 6.822`
- rebuilt (`bank_tenor_comm`): `a(r_comm_real, 10)` with
  `r_comm_real = 1.11/1.02 − 1 = 0.08824` ⇒ `a(0.08824, 10) ≈ 6.469`
- ⇒ bankable revenue at the 2025 vintage falls **~5%**, because the
  legacy rate is *higher* (14.4 vs 8.82) and the legacy horizon *longer*
  (30 vs 10) and the two effects nearly cancel at t = 0.

"Tenor alone is 1.61×" reconciles as `a(r_comm_real, 30) / a(r_comm_real,
10) ≈ 10.44 / 6.47 ≈ 1.61` — the isolated horizon effect *at the new
rate*. The net t = 0 change is small only because the rate move runs the
other way; at later vintages, and for the reported
`bankability_threshold` / `bankable_capex_per_mw` diagnostics, the two
no longer cancel. The rate-coherence argument (Defect 2 / Defect 3)
stands regardless of the t = 0 net.

---

### (b) Every reader of `m.remaining_npv_factor`

**Builders — the constraint pair (THE defect site):**

| File:line | Context |
|---|---|
| `src/optimize_model.py:352` | `m.remaining_npv_factor = pyo.Param(T, ...)` — build site (annual) |
| `src/optimize_model.py:761` | `bankable_revenue = (tariff * solar_energy_per_mw * m.remaining_npv_factor[t] * m.solar_eaas_add[t]) / required_margin` — `eaas_bankability_rule` |
| `src/optimize_model.py:800` | `max_bankable_at_t = tariff * solar_energy_per_mw * m.remaining_npv_factor[t] / required_margin` — `subsidy_limit_rule` |
| `src/optimize_model_sliced.py:439` | build site (sliced) |
| `src/optimize_model_sliced.py:1074` | `eaas_bankability_rule` (sliced) — mirror of annual:761 |
| `src/optimize_model_sliced.py:1089` | `subsidy_limit_rule` (sliced) — mirror of annual:800 |

Both build sites are in the unconditional "Discounting" / "Scalar
parameters" block — built identically under **both**
`capital_treatment_case` values. `m.eaas_bankability` /
`m.eaas_subsidy_limit` are built `if tariff is not None`, also
regardless of treatment.

**`src/optimize_experiments.py`:**

| Line | Context |
|---|---|
| `981` | `run_fin2_blended_bankability_sweep` — reconstructs the per-year financing gap: `... - tariff * solar_energy_per_mw * float(pyo.value(m.remaining_npv_factor[t])) / required_margin`. Drives `04_run_fin2_tariff_sweep.py` (dead, out of scope) but is a genuine reader. |
| `1970` | `extract_planning_diagnostics` — `remaining_npv_factor_0 = float(pyo.value(m.remaining_npv_factor[0]))` ⇒ `npv_energy_priv = remaining_npv_factor_0 * solar_energy_per_mw`, which feeds the `bankability_threshold` (line ~2413) and `bankable_capex_per_mw` (line ~2452) diagnostics. Called by `03` (`run_case`) and by every experiment runner. |

**Scripts (added by the `tariff_case` task, docs/tariff_arm.md Step 3):**

| File:line | Context |
|---|---|
| `scripts/03_run_ndc_eaas.py:309` | `bankability_slack_per_mw_by_year()` — `rnf = float(pyo.value(m.remaining_npv_factor[t]))` |
| `scripts/26_run_f3_identification.py:311` | same helper, mirror |

Both new slack readers will pick up the rebuilt factor automatically — no
change needed there, but the summary field added in Step 2 should sit
next to them.

---

### (c) Every other consumer of `resolve_private_rate` / `blended_wacc` / `blend_case`

**`resolve_private_rate`** — grep `src/ scripts/`:

| File:line | |
|---|---|
| `src/blended_finance.py:168` | def |
| `src/optimize_model.py:18, 346` | import + the only call (→ `remaining_npv_factor`) |
| `src/optimize_model_sliced.py:54, 434` | import + the only call (→ `remaining_npv_factor`) |

⇒ `resolve_private_rate` reaches **nothing but `remaining_npv_factor`**.
Fixing that expression removes its last live consumer from the pre-F1
blend. (TASK.md: do NOT delete the function — `bank_tenor_legacy` needs
it.)

**`blended_wacc`** — grep:

| File:line | |
|---|---|
| `src/blended_finance.py:67` | def; also called internally by `resolve_private_rate:182` |
| `src/optimize_experiments.py:853, 887` | `run_fin2_blended_bankability_sweep` — `r_private = blended_wacc(share, r_conc, r_comm)` for its analytical self-financing threshold `threshold_usd_per_twh = capex_2025 * required_margin / npv_energy`, and it injects `r_concessional`/`r_commercial`/`concessional_share` into `sc` (`:907-908`) so the LP's `resolve_private_rate` also blends. **Same defect, elsewhere**: the FIN-2 T* analytics and its solves both discount at the pre-F1 blend. `04_run_fin2_tariff_sweep.py` is dead (omits `conversion_case`/`hydro_case`/`tariff_case`) and TASK.md rings it off — reported, not touched. Widens the eventual cleanup, not this task. |

**`blend_case`** — grep `src/scenarios.py` only (plus the dead FIN-2
path which reads `blended_finance_scenarios()` directly):

| Line | |
|---|---|
| `2016` | `load_scenario(blend_case="blended_central")` param |
| `2242-2243` | membership validation |
| `2644` | `conc_share, r_conc, r_comm = blended_finance_scenarios()[blend_case]` |
| `2930-2934` | dict entries: `blend_case`, `concessional_share`, `r_concessional`, `r_commercial`, `private_discount_rate` (= r_comm, dead) |
| `3049` | `"blend": blend_case` in a diagnostics sub-dict |
| `2562-2563` (comment only) | notes CRF(r_conc) < CRF(r_comm) |

No runner in `scripts/` passes `blend_case` — every live and dead runner
takes the default `blended_central`.

**Summary of (c):** the pre-F1 blend reaches exactly two live
expressions — `m.remaining_npv_factor` (this task) and
`run_fin2_blended_bankability_sweep` (dead script, out of scope, flagged).
Nothing else.

---

### (d) How `r_comm` and `L_comm` are resolved and deflated in the `f1_tranche` branch

All inside the **`else:` (`capital_treatment_case == 'f1_tranche'`)
branch** — `src/optimize_model_sliced.py:1450` / `src/optimize_model.py:1218`:

| Quantity | sliced | annual | value under `cr_central` |
|---|---|---|---|
| `_L_comm = int(round(float(scenario["L_comm"])))` | `:1476` | `:1252` | `10` |
| `_r_comm_nom = float(scenario["r_comm_nominal"])` | `:1478` | `:1254` | `0.11` |
| `_us_inflation = float(scenario["us_inflation"])` | `:1479` | `:1255` | `0.020` |
| `_r_comm_real = (1.0 + _r_comm_nom) / (1.0 + _us_inflation) - 1.0` | `:1484` | `:1260` | `0.088235…` |
| `_lives = scenario["asset_lifetimes"]` (raises if missing) | `:1499` | `:1272`ish | `{solar:30, storage:15, gas:30}` |
| `_L_eff_comm[ch] = min(_L_comm, round(_lives[ch]))` | `:1518-1521` | mirror | `solar: min(10,30) = 10` |
| `_crf_comm[ch] = _crf(_r_comm_real, _L_eff_comm[ch])` | `:1529` | mirror | — |
| service window: `sum(DF[t] for t in range(v, min(v + _L_eff_comm[ch], _T_last + 1)))` | `:1544-1548` | mirror | annuity payments discounted to t=0 at **r_social** |

Underlying scenario keys — set **unconditionally** in `load_scenario()`
(not inside any `capital_treatment_case` branch):

| key | `src/scenarios.py` | source |
|---|---|---|
| `r_comm_nominal` | `:2966` = `_cr_terms["r_comm"]` | `capital_recovery_scenarios()[cr_central]` `:1760` |
| `L_comm` | `:2967` = `_cr_terms["L_comm"]` | same, `= 10` |
| `us_inflation` | `:2972` | required arg |
| `asset_lifetimes` | `:2871` = `asset_lifetimes()` | `:622` |

`_cr_terms = capital_recovery_scenarios()[capital_recovery_case]` is
resolved unconditionally (`capital_recovery_case` is required, `:2494`).

**Under `pre_f1_overnight`:** the `_r_comm_real`, `_L_comm`, `_crf_comm`
*locals* are **never computed** (they live in the `else:` branch only).
BUT the four scenario keys they derive from (`r_comm_nominal`, `L_comm`,
`us_inflation`, `asset_lifetimes`) are all present in the dict regardless
of treatment. So **the `bank_tenor_comm` arm CAN be built under
`pre_f1_overnight`** — the Fisher deflation
`(1+r_comm_nominal)/(1+us_inflation) − 1` just has to be done at the
`remaining_npv_factor` build site (which runs in both branches), not
reused from the F1 block. **No cross-key guard is required.** (The
existing `conversion_case × pre_f1_overnight` guard at
`src/optimize_model_sliced.py:487` exists because conversion capex has
*no* `pre_f1_overnight` representation at all — a different situation.)

Step-1 note: the F1 capital charge discounts its annuity *payments* to
t=0 at **r_social** (via `DF[t]` in the service window), whereas
`remaining_npv_factor` expresses the annuity in **year-t terms**
(exponent `k − t`). TASK.md's decision — "annuity of `r_comm` real over
`min(L_comm, T_last − t + 1)`" — is the year-t annuity factor
`Σ_{j=0}^{L_eff−1} (1+r_comm_real)^{−j}`, i.e. it keeps the existing
year-t convention and only swaps the rate and the horizon. It does **not**
adopt F1's `DF[t]`-weighted form. The two windows nonetheless coincide in
*length* for solar (`_L_eff_comm["solar"] = min(10, 30) = 10 = L_comm`),
so "the capital charge and the bankability test use the same window for
the same asset" holds without importing the asset-life ceiling into the
rebuild.

**Horizon facts** for the truncation term:
`MODEL_START_YEAR = 2025`, `MODEL_END_YEAR = 2054`, `REPORT_END_YEAR =
2045` (`src/scenarios.py:284-286`). So `len(T) = 30`, `T_last = 29`
(year 2054). Vintage index of 2045 is `t = 20`: `T_last − 20 + 1 = 10` ⇒
full window. `t = 21` (2046): `= 9` ⇒ first truncated vintage. Matches
TASK.md's "only bites after 2045". Verification target `t = 25` (2050):
`T_last − 25 + 1 = 5` ⇒ `min(10, 5) = 5`.

---

## STOP — Step 0 complete.

`(c)` found the pre-F1 blend also reaching
`run_fin2_blended_bankability_sweep` (dead `04_run_fin2_tariff_sweep.py`,
out of scope). No other consumer. `resolve_private_rate`'s only live
consumer is `remaining_npv_factor` itself. Awaiting go-ahead for Steps 1–2.
