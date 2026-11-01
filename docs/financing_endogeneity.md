# Financing endogeneity — is the model's financing endogenous or exogenous?

Status: **Part A complete. STOPPED, as instructed.** Part B (runner, Tests 1–5) and Part C (classification) are not written and wait for release.

Method: files read directly (not via grep alone): `src/optimize_model_sliced.py` (lines 464–628, 1056–1135, 1530–1840, 1915–2040), `src/optimize_model.py` (1330–1480), `src/blended_finance.py` (all), `src/economics.py` (all), `src/scenarios.py` (`capital_recovery_scenarios` 1916, `dscr_scenarios` 1983, `concessional_envelope_scenarios` 2057, `tariff_scenarios` ~984, `load_scenario` dict 3370–3440). Central configuration taken from `scripts/run_phase_b.sh` 195–207: `cr_central`, `dscr_central`, `us_inflation=0.020`, `f1_tranche`, `env_dares`, `tariff_central`. Horizon 2025–2054 (T = 0..29).

No file under `src/` was modified. No solve was run. One arithmetic check (CRF coefficients, A5) was run in the scratchpad.

Line numbers below are for `optimize_model_sliced.py` unless stated. The annual builder's F1 block (`optimize_model.py` 1381–1440) is the same code. I verified the tranche Vars, split identities, envelope constraint and CRF construction. The differences are only in dispatch.

---

## A1. Financing-related objects

| Quantity | Type | Index | Bounds | Declared |
|---|---|---|---|---|
| `solar_public_conc_vol`, `solar_public_comm_vol` | **Var** | T | NonNegativeReals | 1657–1658 |
| `solar_eaas_conc_vol`, `solar_eaas_comm_vol` | **Var** | T | NonNegativeReals | 1665–1666 |
| `storage_conc_vol`, `storage_comm_vol` | **Var** | T | NonNegativeReals | 1691–1692 |
| `gas_conc_vol` | **Var, fixed to 0.0** for all t | T | fixed | 1681, 1683–1684 |
| `gas_comm_vol` | **Var** | T | NonNegativeReals | 1682 |
| `convert_comm_vol` (conversion capex, commercial-only) | **Var** | tier ∈ {A,B,C} | NonNegativeReals | 1823 (only if `_conv`) |
| `eaas_subsidy` | **Var** | T | NonNegativeReals; upper bound by `eaas_subsidy_limit` | 614, 1134 |
| Split identities (eq. 3.30) | Constraint | T | equality | 1659, 1667, 1685, 1693, 1824 |
| Concessional envelope `B` (eq. 3.35) | Constraint, RHS is a Python float `scenario["concessional_envelope_usd"]` | scalar | `Σ_t(solar_public_conc + solar_eaas_conc + storage_conc) ≤ B` | 1713–1720 |
| `_crf_conc[ch]`, `_crf_comm[ch]` (ch = solar, storage, gas) | **plain Python floats** (dict) | channel | n/a | 1612–1613 |
| `_r_conc_real`, `_r_comm_real` | **plain Python floats** | scalar | n/a | 1567–1568 (nominal inputs 1561–1562, inflation 1563) |
| `_L_eff_conc`, `_L_eff_comm` | plain Python ints | channel | n/a | 1598–1605 |
| `_df_window_conc`, `_df_window_comm` (the `W`) | **plain Python floats** | channel × vintage | n/a | 1621–1634 |
| `m.DF` | **Param** | T | PositiveReals | 478 (built from `r_social`, 467–468) |
| `m.remaining_npv_factor` (bankability annuity factor, eq. 3.37) | **Param** | T | PositiveReals | 476; built by `resolve_remaining_npv_factor` (`blended_finance.py` 271) |
| Private rate in the bankability factor | **plain Python float** `r_comm_real`, derived inside `commercial_tenor_npv_factor` (`blended_finance.py` 234–237) from `r_comm_nominal`, `us_inflation`. This is the commercial rate only; there is no blend. Under `bank_tenor_legacy` it is the blended WACC from `resolve_private_rate` (`blended_finance.py` 172–196), which is not the default. | scalar | n/a | |
| Coverage margin `m` (`required_margin`) | plain Python float, from `dscr_case` (`dscr_central` = 1.30) | scalar | n/a | read at 506 |
| Service tariff | plain Python float `tariff` (95e6 USD/TWh central) | scalar | n/a | read at 499 |
| Bankability constraint | Constraint `eaas_bankability` | T | `capex − subsidy ≤ tariff·E·F_t·solar_eaas_add / m` | 1106–1115 |
| Subsidy cap | Constraint `eaas_subsidy_limit`; `gap_at_t` is a float | T | `subsidy ≤ gap·solar_eaas_add` | 1122–1134 |
| Public budget | optional Constraint, only if `public_solar_budget_npv` is not None | scalar | | 2005–2016 |

Note: `eaas_subsidy` is not an objective term. It appears only in `eaas_bankability` (1108), `eaas_subsidy_limit` and the optional public budget (2012). The objective charges the full capex through the tranche volumes whatever the subsidy is. Its role is to relax bankability.

Note: `blended_finance.blended_wacc()` and `implied_concessional_share()` are not called by either builder on the default path. They are only used by `bank_tenor_legacy` and the FIN-2 analytics.

## A2. Is any rate, WACC or discount rate a decision variable?

**No.** Full `Var` list of `build_model_sliced` (every declaration, in order):

`solar_public_add`, `solar_eaas_add`, `eaas_subsidy`, `gas_add`, `storage_add` (612–620); `unserved`, `selfgen`, `gas_to_power`, `gas_curtail`, `storage_discharge`, `storage_charge`, `hydro_gen` (632–648); `convert_mw`, `convert_to_power` (661, 666, if `_conv`); `solar_public_conc_vol`, `solar_public_comm_vol`, `solar_eaas_conc_vol`, `solar_eaas_comm_vol`, `gas_conc_vol` (fixed), `gas_comm_vol`, `storage_conc_vol`, `storage_comm_vol` (1657–1692); `convert_comm_vol` (1823, if `_conv`).

None of these is a rate, WACC, discount rate or cost of capital. Eight of them are **capital volumes by funding source**, the concessional/commercial split. So the question reduces to whether the split counts as a "financing structure" decision variable (limb i). That is a question about the definition, and A5 and Test 2 bear on it.

## A3. Does any financing coefficient depend on a `Var`?

**No.** Every CRF, `W`, rate and tenor is a precomputed Python float or int:

- `_crf_conc/_crf_comm`: float function `_crf` (1542–1557) applied to floats at 1612–1613. Note the floor: `r ≤ 0 → CRF = 1/L` (1555).
- `_df_window_*`: `sum(float(pyo.value(m.DF[t])) …)` (1621–1634). `pyo.value` of a Param, so a number.
- `remaining_npv_factor`: a dict of floats from `commercial_tenor_npv_factor` (`blended_finance.py` 241–243).
- `required_margin`, `tariff`, `gas_capex_per_mw` and the rebuild-year coefficients are likewise numbers.

The objective coefficient on each tranche volume is `CRF × W` (a float), so the LP stays linear. The code states this as an invariant (1607–1611, 1650–1656).

## A4. Feedback from the plan to the price of capital?

**None found.** I searched for any rate, CRF, W or tenor that is a function of technology mix, capacity, emissions, reliability or any other solved quantity. In the files read:

- `r_conc`, `r_comm`, `L_conc`, `L_comm`, `us_inflation` come from `capital_recovery_scenarios()` and `load_scenario()` (scenarios.py 3431–3439). They are fixed inputs.
- `L_eff = min(L_tranche, asset_lifetimes[ch])` depends on the channel label only, not on any solved quantity.
- No learning curve, volume discount or risk premium exists. The code says so at 1650–1656.
- The only tenor/rate truncation interacting with the solution is the horizon cut-off in `W`, which is a function of vintage index, not of the plan.

One qualification: the *volume-weighted* mix of the two fixed rates changes with the plan (A5). That is a different thing from a rate responding to risk.

## A5. Degrees of freedom in the split

Constraints acting on the split of channel c, vintage v:

1. Split identity (eq. 3.30): `conc_vol[c,v] + comm_vol[c,v] = capex_c,v · add[c,v]` (equality). Given the build, the split is one-dimensional.
2. Envelope (eq. 3.35): a single **cumulative, undiscounted** constraint over *three* channels (public solar, EaaS solar, storage) and *all* vintages. Gas is excluded.
3. Gas: `gas_conc_vol` fixed to 0 (1684). No freedom; 100% commercial.
4. Rebuilds (`cost_rebuild_capital_charge_npv`, 1783) and conversion (`convert_comm_vol`): commercial by construction. They carry no tranche Var, so they have no split at all.
5. Bounds: non-negativity only. There is no per-vintage or per-channel cap on `conc_vol`, no minimum share, no maximum share.

**The ordering of tranche costs, computed (arithmetic only).** At `cr_central` and `us_inflation=0.02`: `r_conc_real = 0.1176%`, `r_comm_real = 8.8235%`. `L_eff`: solar 25 (conc) / 10 (comm); storage 15 / 10. The ratio of the concessional to the commercial objective coefficient, `CRF·W`, per vintage:

| Channel | v=0 | v=10 | v=15 | v=20 | v=29 |
|---|---|---|---|---|---|
| solar (conc / comm) | 0.4682 / 1.1205 = 0.418 | 0.384 | 0.335 | 0.263 | 0.263 |
| storage (conc / comm) | 0.6221 / 1.1205 = 0.555 | 0.555 | 0.555 | 0.435 | 0.435 |

The concessional coefficient is **lower than the commercial one for every channel and vintage**. So a dollar moved from commercial to concessional always reduces cost. The saving per dollar is `comm − conc`: about 0.65 at solar v=0, 0.50 at storage v=0, and falling toward 0 for late vintages.

**Result from structure alone.**

- Cost-minimising LP, no constraint other than the envelope couples the tranches, and the objective strictly prefers conc: the split is "fill the envelope, then commercial" in *aggregate*. When `B ≥ total eligible capex` all eligible capital is concessional. When `B < total eligible capex` the envelope binds with a positive dual.
- Beyond that rule there is **a real allocation freedom**: *which* of the (channel, vintage) cells receive the scarce concessional dollars. The LP allocates by saving per dollar. Early vintages and solar (saving per dollar largest) are favoured, subject to how much capex each cell has. This is created by the envelope being pooled across vintages and across three channels. It is not a mechanical front-loading rule: the plan (the `add` values) is itself being optimised jointly, so the cell capex totals are endogenous.
- The split is therefore a decision variable in form, and mechanical in aggregate (fill the envelope). The cell-level allocation is a genuine optimisation outcome, though it is driven by the fixed price differential.
- **Inconclusive from structure:** whether an interior share (strictly between 0 and 1) arises with a slack envelope. By the strict ordering above it should not: with the envelope slack, every eligible dollar is concessional. Test 2 can confirm or refute that.

---

## Part A: what it says about each limb, and what it does not

These are static findings, not conclusions. Part C will be written only after Part B.

| Limb | Static evidence from Part A |
|---|---|
| (i) decision variable | A *volume split* between two fixed-rate tranches is a `Var` (A1, A2). No `Var` is a rate or WACC (A2). Whether a split of capital volumes counts as "financing structure … a decision variable" depends on how §2.5.2 is read. Aggregate content is "fill the envelope" (A5). Not settled by Part A. |
| (ii) solution outcome | Static reading: the volume-weighted blend of the two fixed rates, `CRF_realised(c,v)`, is a function of the solved split, so it is *capable* of varying across solves with no rate input changed. Whether it does is Test 1. Not settled by Part A. |
| (iii) constraint interacting with capacity | The bankability constraint (1106–1115) contains the capacity `Var` `solar_eaas_add` and the subsidy `Var`. Its `F_t` uses the *commercial* rate only and does not depend on the split (A1). The envelope constraint couples the split to capacity volumes through the split identity. Whether either alters the capacity solution is Test 3 and Test 1. Not settled by Part A. |
| (not in definition) risk → rate | **Absent** (A4). |

Points for the neutrality rule:

- Evidence that favours H-EXO: no rate, CRF, W or tenor depends on a `Var` (A3); no plan-to-price feedback (A4); the aggregate split is "fill the envelope" (A5); the bankability factor uses a single fixed rate and is independent of the split (A1).
- Evidence that favours H-ENDO: the concessional/commercial volumes are `Var`s (A1); the cell-level allocation of the envelope is determined by the optimisation (A5); the realised blended CRF is a volume-weighted function of solved quantities (A5, awaits Test 1).
- Chapter 2 §2.7.1 as currently drafted claims limbs (i) and (iii) hold and the cost-of-capital limb does not. Part A neither confirms nor contradicts that. In particular, if "the cost-of-capital limb" means limb (ii), Part A suggests it may *not* be cleanly "does not hold", pending Test 1.

## Items found in passing (not part of the A1–A5 questions)

- `bank_tenor_legacy` (non-default) makes the bankability factor depend on `blended_finance.blended_wacc()` with `concessional_share = 0.30`. That share is a scenario constant, not a model output, so even there nothing is solved.
- Rebuild capex and conversion capex bypass the split entirely (commercial-only, no Var). These are exogenous in the sense of the definition.

## Questions for the release of the stop

1. **Test 3(c) mechanism.** `eaas_bankability` is built only when `tariff is not None` (1105). Deactivating it can be done inside the diagnostic with `m.eaas_bankability.deactivate()` after `build_model_sliced`, with no `src/` edit. I propose that rather than a scenario arm. Note that `eaas_subsidy_limit` would then still cap the subsidy at `gap·add`. With the constraint off the subsidy has no use, so it should be 0. I would report that.
2. **Is EaaS actually active in the central scenario?** Line 1117: `financing_regime != "eaas"` forces `solar_eaas_add ≤ 0`. The central dict sets `"financing_regime": "traditional"` (scenarios.py ~3377) unless the runner overrides it. If the central arm runs `traditional`, Test 3 would be vacuous (EaaS capacity is 0 in all three configurations). I have not yet checked what `cr_central` runs use. Part B needs `financing_regime="eaas"` and I will state it, unless you want otherwise.
3. **Public budget.** `public_solar_budget_npv` comes from `capital_envelope_scenarios()[capital_case]` (scenarios.py 3350). I have not read what the central value is. It decides whether the subsidy is budget-limited.
4. Test 1 and Test 5 need the envelope arms `env_none`, `env_dares`, `env_5x`, `env_10x`, plus a 0.5× value that does not exist as a named arm. I will pass a scenario-dict override of `concessional_envelope_usd` (375e6) inside the script, not a new arm in `src/`.

## Inconclusive / Unresolved (so far)

- Whether the split ever has an interior share under a slack envelope (A5): predicted not, untested.
- Items 1–4 above.

<!-- BEGIN PART B AUTOGEN (scripts/diagnose_financing_endogeneity.py) -->

# Part B — empirical tests (generated by `scripts/diagnose_financing_endogeneity.py`)

Generated 2026-10-13 21:28:50. Solver `cbc`. 71 optimal / 71 runs. These are VERIFICATION artefacts, not thesis findings. Part C is written by hand after this section.

## B0. Run manifest — exactly what each run used

Held at the Phase B central arms in every run (from `scripts/run_phase_b.sh` 189-207): `peak_multiple_case=peak_unsuppressed`, `reserve_margin=nirp_spinning`, `storage_cc_case=storage_cc_central`, `gas_build_case=gas_unconstrained`, `new_gas_deliverability=constrained`, `gas_availability_case=gas_avail_nameplate`, `gas_retirement_case=gas_retire_vintage_30`, `gas_capex_case=gas_capex_central`, `ccgt_efficiency_case=ccgt_eff_central`, `capital_recovery_case=cr_central`, `dscr_case=dscr_central`, `us_inflation=0.02`, `capital_treatment_case=f1_tranche`, `conversion_case=conv_central`, `hydro_case=hydro_flat`, `tariff_case=tariff_central`, `om_case=om_central`. Solar build `deployment_unconstrained`, carbon `no_policy`, `voll_mid`, `genset_central`, capex `solar_low`.

**Regimes Phase B actually ran** (read, not assumed): item 0 `01b_run_baseline_sliced.py` — `financing_regime` is never set, so `load_scenario()`'s default `"traditional"` applies (`src/scenarios.py:3373`), with `capital_case="unconstrained"` (`01b…py:297`). Item 2 `02_run_eaas.py` sets `scenario["financing_regime"] = "eaas"` (`02_run_eaas.py:260`) with `capital_case="moderate"` (`02_run_eaas.py:227`).

Fixed financing inputs (identical in every run unless the row says otherwise): r_conc nominal 2.12% / real 0.1176%, L_conc 25; r_comm nominal 11.00% / real 8.8235%, L_comm 10; inflation 2.00%; required_margin 1.3; tariff 95,000,000 USD/TWh; bank_tenor_case `bank_tenor_comm`.

**Public budget** (`capital_envelope_scenarios()`): `unconstrained` = none (constraint NOT built); `moderate` = 8,732,342,068 USD NPV (constraint built, sliced:2005-2016).

| run id | regime | capital_case | envelope | variation | demand | gas deliv. | solar CF | emissions cap | dscr_case | bankability | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P1_trad_unc__env_none__base__central | traditional | unconstrained | env_none (arm `env_none` ) = 0.0M | base | organic_central | baseline | central 0.19 | none | dscr_central | on | optimal |
| P1_trad_unc__env_0.5x__base__central | traditional | unconstrained | env_0.5x (arm `env_dares` override 375,000,000) = 375.0M | base | organic_central | baseline | central 0.19 | none | dscr_central | on | optimal |
| P1_trad_unc__env_dares__base__central | traditional | unconstrained | env_dares (arm `env_dares` ) = 750.0M | base | organic_central | baseline | central 0.19 | none | dscr_central | on | optimal |
| P1_trad_unc__env_5x__base__central | traditional | unconstrained | env_5x (arm `env_5x` ) = 3,750.0M | base | organic_central | baseline | central 0.19 | none | dscr_central | on | optimal |
| P1_trad_unc__env_10x__base__central | traditional | unconstrained | env_10x (arm `env_10x` ) = 7,500.0M | base | organic_central | baseline | central 0.19 | none | dscr_central | on | optimal |
| P1_trad_unc__env_dares__b1_demand_low__central | traditional | unconstrained | env_dares (arm `env_dares` ) = 750.0M | b1_demand_low | constrained_continuation | baseline | central 0.19 | none | dscr_central | on | optimal |
| P1_trad_unc__env_dares__b1_demand_high__central | traditional | unconstrained | env_dares (arm `env_dares` ) = 750.0M | b1_demand_high | organic_high | baseline | central 0.19 | none | dscr_central | on | optimal |
| P1_trad_unc__env_dares__b2_gas_downside__central | traditional | unconstrained | env_dares (arm `env_dares` ) = 750.0M | b2_gas_downside | organic_central | downside | central 0.19 | none | dscr_central | on | optimal |
| P1_trad_unc__env_dares__b2_gas_upside__central | traditional | unconstrained | env_dares (arm `env_dares` ) = 750.0M | b2_gas_upside | organic_central | upside | central 0.19 | none | dscr_central | on | optimal |
| P1_trad_unc__env_dares__b2_gas_shock__central | traditional | unconstrained | env_dares (arm `env_dares` ) = 750.0M | b2_gas_shock | organic_central | shock_recovery | central 0.19 | none | dscr_central | on | optimal |
| P1_trad_unc__env_dares__b3_cap_ndc3cond__central | traditional | unconstrained | env_dares (arm `env_dares` ) = 750.0M | b3_cap_ndc3cond | organic_central | baseline | central 0.19 | ndc3_conditional | dscr_central | on | optimal |
| P1_trad_unc__env_dares__b4_cf_0.16__central | traditional | unconstrained | env_dares (arm `env_dares` ) = 750.0M | b4_cf_0.16 | organic_central | baseline | 0.16 | none | dscr_central | on | optimal |
| P1_trad_unc__env_dares__b4_cf_0.20__central | traditional | unconstrained | env_dares (arm `env_dares` ) = 750.0M | b4_cf_0.20 | organic_central | baseline | 0.2 | none | dscr_central | on | optimal |
| P1_trad_unc__env_none__b1_demand_low__central | traditional | unconstrained | env_none (arm `env_none` ) = 0.0M | b1_demand_low | constrained_continuation | baseline | central 0.19 | none | dscr_central | on | optimal |
| P1_trad_unc__env_none__b1_demand_high__central | traditional | unconstrained | env_none (arm `env_none` ) = 0.0M | b1_demand_high | organic_high | baseline | central 0.19 | none | dscr_central | on | optimal |
| P1_trad_unc__env_none__b2_gas_downside__central | traditional | unconstrained | env_none (arm `env_none` ) = 0.0M | b2_gas_downside | organic_central | downside | central 0.19 | none | dscr_central | on | optimal |
| P1_trad_unc__env_none__b2_gas_upside__central | traditional | unconstrained | env_none (arm `env_none` ) = 0.0M | b2_gas_upside | organic_central | upside | central 0.19 | none | dscr_central | on | optimal |
| P1_trad_unc__env_none__b2_gas_shock__central | traditional | unconstrained | env_none (arm `env_none` ) = 0.0M | b2_gas_shock | organic_central | shock_recovery | central 0.19 | none | dscr_central | on | optimal |
| P1_trad_unc__env_none__b3_cap_ndc3cond__central | traditional | unconstrained | env_none (arm `env_none` ) = 0.0M | b3_cap_ndc3cond | organic_central | baseline | central 0.19 | ndc3_conditional | dscr_central | on | optimal |
| P1_trad_unc__env_none__b4_cf_0.16__central | traditional | unconstrained | env_none (arm `env_none` ) = 0.0M | b4_cf_0.16 | organic_central | baseline | 0.16 | none | dscr_central | on | optimal |
| P1_trad_unc__env_none__b4_cf_0.20__central | traditional | unconstrained | env_none (arm `env_none` ) = 0.0M | b4_cf_0.20 | organic_central | baseline | 0.2 | none | dscr_central | on | optimal |
| P2_eaas_mod__env_none__base__central | eaas | moderate | env_none (arm `env_none` ) = 0.0M | base | organic_central | baseline | central 0.19 | none | dscr_central | on | optimal |
| P2_eaas_mod__env_0.5x__base__central | eaas | moderate | env_0.5x (arm `env_dares` override 375,000,000) = 375.0M | base | organic_central | baseline | central 0.19 | none | dscr_central | on | optimal |
| P2_eaas_mod__env_dares__base__central | eaas | moderate | env_dares (arm `env_dares` ) = 750.0M | base | organic_central | baseline | central 0.19 | none | dscr_central | on | optimal |
| P2_eaas_mod__env_5x__base__central | eaas | moderate | env_5x (arm `env_5x` ) = 3,750.0M | base | organic_central | baseline | central 0.19 | none | dscr_central | on | optimal |
| P2_eaas_mod__env_10x__base__central | eaas | moderate | env_10x (arm `env_10x` ) = 7,500.0M | base | organic_central | baseline | central 0.19 | none | dscr_central | on | optimal |
| P2_eaas_mod__env_dares__b1_demand_low__central | eaas | moderate | env_dares (arm `env_dares` ) = 750.0M | b1_demand_low | constrained_continuation | baseline | central 0.19 | none | dscr_central | on | optimal |
| P2_eaas_mod__env_dares__b1_demand_high__central | eaas | moderate | env_dares (arm `env_dares` ) = 750.0M | b1_demand_high | organic_high | baseline | central 0.19 | none | dscr_central | on | optimal |
| P2_eaas_mod__env_dares__b2_gas_downside__central | eaas | moderate | env_dares (arm `env_dares` ) = 750.0M | b2_gas_downside | organic_central | downside | central 0.19 | none | dscr_central | on | optimal |
| P2_eaas_mod__env_dares__b2_gas_upside__central | eaas | moderate | env_dares (arm `env_dares` ) = 750.0M | b2_gas_upside | organic_central | upside | central 0.19 | none | dscr_central | on | optimal |
| P2_eaas_mod__env_dares__b2_gas_shock__central | eaas | moderate | env_dares (arm `env_dares` ) = 750.0M | b2_gas_shock | organic_central | shock_recovery | central 0.19 | none | dscr_central | on | optimal |
| P2_eaas_mod__env_dares__b3_cap_ndc3cond__central | eaas | moderate | env_dares (arm `env_dares` ) = 750.0M | b3_cap_ndc3cond | organic_central | baseline | central 0.19 | ndc3_conditional | dscr_central | on | optimal |
| P2_eaas_mod__env_dares__b4_cf_0.16__central | eaas | moderate | env_dares (arm `env_dares` ) = 750.0M | b4_cf_0.16 | organic_central | baseline | 0.16 | none | dscr_central | on | optimal |
| P2_eaas_mod__env_dares__b4_cf_0.20__central | eaas | moderate | env_dares (arm `env_dares` ) = 750.0M | b4_cf_0.20 | organic_central | baseline | 0.2 | none | dscr_central | on | optimal |
| P2_eaas_mod__env_none__b1_demand_low__central | eaas | moderate | env_none (arm `env_none` ) = 0.0M | b1_demand_low | constrained_continuation | baseline | central 0.19 | none | dscr_central | on | optimal |
| P2_eaas_mod__env_none__b1_demand_high__central | eaas | moderate | env_none (arm `env_none` ) = 0.0M | b1_demand_high | organic_high | baseline | central 0.19 | none | dscr_central | on | optimal |
| P2_eaas_mod__env_none__b2_gas_downside__central | eaas | moderate | env_none (arm `env_none` ) = 0.0M | b2_gas_downside | organic_central | downside | central 0.19 | none | dscr_central | on | optimal |
| P2_eaas_mod__env_none__b2_gas_upside__central | eaas | moderate | env_none (arm `env_none` ) = 0.0M | b2_gas_upside | organic_central | upside | central 0.19 | none | dscr_central | on | optimal |
| P2_eaas_mod__env_none__b2_gas_shock__central | eaas | moderate | env_none (arm `env_none` ) = 0.0M | b2_gas_shock | organic_central | shock_recovery | central 0.19 | none | dscr_central | on | optimal |
| P2_eaas_mod__env_none__b3_cap_ndc3cond__central | eaas | moderate | env_none (arm `env_none` ) = 0.0M | b3_cap_ndc3cond | organic_central | baseline | central 0.19 | ndc3_conditional | dscr_central | on | optimal |
| P2_eaas_mod__env_none__b4_cf_0.16__central | eaas | moderate | env_none (arm `env_none` ) = 0.0M | b4_cf_0.16 | organic_central | baseline | 0.16 | none | dscr_central | on | optimal |
| P2_eaas_mod__env_none__b4_cf_0.20__central | eaas | moderate | env_none (arm `env_none` ) = 0.0M | b4_cf_0.20 | organic_central | baseline | 0.2 | none | dscr_central | on | optimal |
| P2_eaas_mod__env_dares__base__a_central | eaas | moderate | env_dares (arm `env_dares` ) = 750.0M | base | organic_central | baseline | central 0.19 | none | dscr_central | on | optimal |
| P2_eaas_mod__env_dares__base__b_loosest | eaas | moderate | env_dares (arm `env_dares` ) = 750.0M | base | organic_central | baseline | central 0.19 | none | dscr_regression | on | optimal |
| P2_eaas_mod__env_dares__base__b2_tighter | eaas | moderate | env_dares (arm `env_dares` ) = 750.0M | base | organic_central | baseline | central 0.19 | none | dscr_high | on | optimal |
| P2_eaas_mod__env_dares__base__c_bankability_off | eaas | moderate | env_dares (arm `env_dares` ) = 750.0M | base | organic_central | baseline | central 0.19 | none | dscr_central | OFF | optimal |
| P3_eaas_unc__env_none__base__central | eaas | unconstrained | env_none (arm `env_none` ) = 0.0M | base | organic_central | baseline | central 0.19 | none | dscr_central | on | optimal |
| P3_eaas_unc__env_0.5x__base__central | eaas | unconstrained | env_0.5x (arm `env_dares` override 375,000,000) = 375.0M | base | organic_central | baseline | central 0.19 | none | dscr_central | on | optimal |
| P3_eaas_unc__env_dares__base__central | eaas | unconstrained | env_dares (arm `env_dares` ) = 750.0M | base | organic_central | baseline | central 0.19 | none | dscr_central | on | optimal |
| P3_eaas_unc__env_5x__base__central | eaas | unconstrained | env_5x (arm `env_5x` ) = 3,750.0M | base | organic_central | baseline | central 0.19 | none | dscr_central | on | optimal |
| P3_eaas_unc__env_10x__base__central | eaas | unconstrained | env_10x (arm `env_10x` ) = 7,500.0M | base | organic_central | baseline | central 0.19 | none | dscr_central | on | optimal |
| P3_eaas_unc__env_dares__b1_demand_low__central | eaas | unconstrained | env_dares (arm `env_dares` ) = 750.0M | b1_demand_low | constrained_continuation | baseline | central 0.19 | none | dscr_central | on | optimal |
| P3_eaas_unc__env_dares__b1_demand_high__central | eaas | unconstrained | env_dares (arm `env_dares` ) = 750.0M | b1_demand_high | organic_high | baseline | central 0.19 | none | dscr_central | on | optimal |
| P3_eaas_unc__env_dares__b2_gas_downside__central | eaas | unconstrained | env_dares (arm `env_dares` ) = 750.0M | b2_gas_downside | organic_central | downside | central 0.19 | none | dscr_central | on | optimal |
| P3_eaas_unc__env_dares__b2_gas_upside__central | eaas | unconstrained | env_dares (arm `env_dares` ) = 750.0M | b2_gas_upside | organic_central | upside | central 0.19 | none | dscr_central | on | optimal |
| P3_eaas_unc__env_dares__b2_gas_shock__central | eaas | unconstrained | env_dares (arm `env_dares` ) = 750.0M | b2_gas_shock | organic_central | shock_recovery | central 0.19 | none | dscr_central | on | optimal |
| P3_eaas_unc__env_dares__b3_cap_ndc3cond__central | eaas | unconstrained | env_dares (arm `env_dares` ) = 750.0M | b3_cap_ndc3cond | organic_central | baseline | central 0.19 | ndc3_conditional | dscr_central | on | optimal |
| P3_eaas_unc__env_dares__b4_cf_0.16__central | eaas | unconstrained | env_dares (arm `env_dares` ) = 750.0M | b4_cf_0.16 | organic_central | baseline | 0.16 | none | dscr_central | on | optimal |
| P3_eaas_unc__env_dares__b4_cf_0.20__central | eaas | unconstrained | env_dares (arm `env_dares` ) = 750.0M | b4_cf_0.20 | organic_central | baseline | 0.2 | none | dscr_central | on | optimal |
| P3_eaas_unc__env_none__b1_demand_low__central | eaas | unconstrained | env_none (arm `env_none` ) = 0.0M | b1_demand_low | constrained_continuation | baseline | central 0.19 | none | dscr_central | on | optimal |
| P3_eaas_unc__env_none__b1_demand_high__central | eaas | unconstrained | env_none (arm `env_none` ) = 0.0M | b1_demand_high | organic_high | baseline | central 0.19 | none | dscr_central | on | optimal |
| P3_eaas_unc__env_none__b2_gas_downside__central | eaas | unconstrained | env_none (arm `env_none` ) = 0.0M | b2_gas_downside | organic_central | downside | central 0.19 | none | dscr_central | on | optimal |
| P3_eaas_unc__env_none__b2_gas_upside__central | eaas | unconstrained | env_none (arm `env_none` ) = 0.0M | b2_gas_upside | organic_central | upside | central 0.19 | none | dscr_central | on | optimal |
| P3_eaas_unc__env_none__b2_gas_shock__central | eaas | unconstrained | env_none (arm `env_none` ) = 0.0M | b2_gas_shock | organic_central | shock_recovery | central 0.19 | none | dscr_central | on | optimal |
| P3_eaas_unc__env_none__b3_cap_ndc3cond__central | eaas | unconstrained | env_none (arm `env_none` ) = 0.0M | b3_cap_ndc3cond | organic_central | baseline | central 0.19 | ndc3_conditional | dscr_central | on | optimal |
| P3_eaas_unc__env_none__b4_cf_0.16__central | eaas | unconstrained | env_none (arm `env_none` ) = 0.0M | b4_cf_0.16 | organic_central | baseline | 0.16 | none | dscr_central | on | optimal |
| P3_eaas_unc__env_none__b4_cf_0.20__central | eaas | unconstrained | env_none (arm `env_none` ) = 0.0M | b4_cf_0.20 | organic_central | baseline | 0.2 | none | dscr_central | on | optimal |
| P3_eaas_unc__env_dares__base__a_central | eaas | unconstrained | env_dares (arm `env_dares` ) = 750.0M | base | organic_central | baseline | central 0.19 | none | dscr_central | on | optimal |
| P3_eaas_unc__env_dares__base__b_loosest | eaas | unconstrained | env_dares (arm `env_dares` ) = 750.0M | base | organic_central | baseline | central 0.19 | none | dscr_regression | on | optimal |
| P3_eaas_unc__env_dares__base__b2_tighter | eaas | unconstrained | env_dares (arm `env_dares` ) = 750.0M | base | organic_central | baseline | central 0.19 | none | dscr_high | on | optimal |
| P3_eaas_unc__env_dares__base__c_bankability_off | eaas | unconstrained | env_dares (arm `env_dares` ) = 750.0M | base | organic_central | baseline | central 0.19 | none | dscr_central | OFF | optimal |

Charge check: in every optimal run the tranche capital charges recomputed from this script's CRF×W and the extracted volumes match the model's `cost_*_capital_charge_npv` Expressions to < 1e-6 relative, so the CRFs used below are the CRFs in the objective.

**Definitions (declared choices).** `CRF_realised(c,v)` is exactly the task's formula. Its implied real rate `r_impl` inverts `CRF(r, L)` and needs a single tenor, but a blended cell has two (solar 25/10, storage 15/10, gas n/a). Reported at the COMMERCIAL tenor `L_comm` (and, in CSV, also at `L_conc`); this makes `r_impl` a tenor-convention-dependent summary (when a blended CRF falls below 1/L_comm, because the concessional CRF is on a 25-yr tenor, the inversion has no solution at L_comm and floors at 0.000% — a tenor artefact, not a zero rate). The tenor-free alternative reported beside it, `r_vw`, is the capital-weighted average of the two tranche REAL rates (the blended-WACC sense of `blended_finance.blended_wacc`). Conclusions about whether the rate MOVES should be read from both. Portfolio weights are undiscounted capital volumes. `elig` = public solar + EaaS solar + storage (concessional-eligible); `all4` adds gas (100% commercial by construction); `+rebuild` additionally adds rebuild and conversion capital (commercial, no tranche Var).

---

## Panel `P1_trad_unc` — regime `traditional`, capital_case `unconstrained`

### Test 1 — envelope sweep (limb ii, weak form; DEMOTED: calibrates the range)

| envelope | B (USD) | conc. share (elig.) | CRF solar_pub | CRF solar_eaas | CRF storage | CRF gas | r_impl solar_pub | r_impl solar_eaas | r_impl storage | r_impl gas | portfolio r_vw (elig.) | portfolio r_vw (all4) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| env_none | 0.0M | 0.00% | 0.15461 | – | 0.15461 | 0.15461 | 8.824% | – | 8.824% | 8.824% | 8.8235% | 8.8235% |
| env_0.5x | 375.0M | 0.94% | 0.15284 | – | 0.15461 | 0.15461 | 8.564% | – | 8.824% | 8.824% | 8.7420% | 8.7489% |
| env_dares | 750.0M | 1.87% | 0.15107 | – | 0.15461 | 0.15461 | 8.303% | – | 8.824% | 8.824% | 8.6604% | 8.6742% |
| env_5x | 3,750.0M | 9.37% | 0.13690 | – | 0.15461 | 0.15461 | 6.161% | – | 8.824% | 8.824% | 8.0080% | 8.0772% |
| env_10x | 7,500.0M | 18.68% | 0.11937 | – | 0.15461 | 0.15461 | 3.357% | – | 8.824% | 8.824% | 7.1974% | 7.3349% |

- `port_elig_r_vw` across the envelope sweep: min 7.1974%, max 8.8235%, **range 162.611 bp**.

- `port_all4_r_vw` across the envelope sweep: min 7.3349%, max 8.8235%, **range 148.862 bp**.

- `port_elig_r_impl_Lcomm` across the envelope sweep: min 5.6022%, max 8.8235%, **range 322.129 bp**.

**Interpretation rule (stated in advance).** If `r_realised` varies while every rate input is fixed, the cost of capital is a quantity the solution determines (limb ii, WEAK form: it responds to the analyst's financing dial). If invariant, limb ii is not satisfied.

### Test 1b — DECISIVE: financing inputs all fixed (envelope = env_dares); vary only demand / gas / cap / solar yield

| variation | solar built (MW) | eligible capex (bn USD) | conc. share (elig.) | portfolio r_vw (elig.) | Δ vs base (bp) | portfolio r_vw (all4) | r_vw (+rebuild) | r_impl@L_comm (elig.) | envelope binds | envelope dual |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 39,030 | 40.027 | 1.874% | 8.6604% | 0.000 | 8.6742% | 8.6873% | 8.5100% | yes | -0.3883 |
| b1_demand_low | 25,464 | 27.028 | 2.775% | 8.5820% | -7.845 | 8.6095% | 8.6335% | 8.3586% | yes | -0.3407 |
| b1_demand_high | 94,760 | 93.257 | 0.804% | 8.7535% | 9.311 | 8.7568% | 8.7616% | 8.6892% | yes | -0.4439 |
| b2_gas_downside | 51,553 | 51.482 | 1.457% | 8.6967% | 3.630 | 8.6976% | 8.7105% | 8.5800% | yes | -0.415 |
| b2_gas_upside | 30,146 | 32.375 | 2.317% | 8.6218% | -3.856 | 8.6541% | 8.6704% | 8.4356% | yes | -0.3407 |
| b2_gas_shock | 40,732 | 42.028 | 1.785% | 8.6682% | 0.777 | 8.6793% | 8.6921% | 8.5250% | yes | -0.415 |
| b3_cap_ndc3cond | 49,112 | 51.465 | 1.457% | 8.6967% | 3.625 | 8.6989% | 8.7118% | 8.5799% | yes | -0.4794 |
| b4_cf_0.16 | 46,442 | 44.638 | 1.680% | 8.6773% | 1.685 | 8.6885% | 8.6992% | 8.5425% | yes | -0.3883 |
| b4_cf_0.20 | 37,054 | 38.797 | 1.933% | 8.6552% | -0.517 | 8.6699% | 8.6837% | 8.5001% | yes | -0.3636 |

**Magnitude across base + b1–b4:** range of portfolio real rate = **17.156 bp** (eligible), 14.726 bp (all four channels), 12.812 bp (incl. rebuild/conversion). Numerical-noise floor taken as 0.01 bp.

**Reading (eligible portfolio):** the realised rate MOVES by more than the noise floor → limb (ii) STRONG form is supported on this panel (the realised cost of capital responds to quantities with no financing content). Whether that movement is economically large is the bp figure above; this script does not set a threshold for 'large'.

**Mechanism note — cell-level allocation of the pooled envelope.** (`full` = share 1; `interior` = vintage:share; `none` = share 0; vintage index = model year.)

| variation | allocation of concessional dollars across (channel, vintage) |
|---|---|
| base | solar_public: full[-] interior[2032:0.932] none[2033-2054] | storage: full[-] interior[-] none[2037-2054] |
| b1_demand_low | solar_public: full[2033] interior[2034:0.927] none[2035-2054] | storage: full[-] interior[-] none[2038-2054] |
| b1_demand_high | solar_public: full[2029] interior[2030:0.355] none[2031-2054] | storage: full[-] interior[-] none[2036-2054] |
| b2_gas_downside | solar_public: full[2030] interior[2031:0.056] none[2032-2054] | storage: full[-] interior[-] none[2036-2054] |
| b2_gas_upside | solar_public: full[2033] interior[2034:0.364] none[2035-2054] | storage: full[-] interior[-] none[2038-2054] |
| b2_gas_shock | solar_public: full[-] interior[2031:0.609] none[2032-2054] | storage: full[-] interior[-] none[2037-2054] |
| b3_cap_ndc3cond | solar_public: full[2028] interior[2029:0.136] none[2030-2054] | storage: full[-] interior[-] none[2032-2033,2036-2054] |
| b4_cf_0.16 | solar_public: full[-] interior[2032:0.720] none[2033-2054] | storage: full[-] interior[-] none[2037-2054] |
| b4_cf_0.20 | solar_public: full[2032] interior[2033:0.010] none[2034-2054] | storage: full[-] interior[-] none[2037-2054] |

The eligible-portfolio `r_vw` is an exact function of the eligible concessional share (`r_comm − share·(r_comm − r_conc)`, both techs share the same two real rates), so movement in it can arise ONLY from the pooled fixed envelope being spread over a larger or smaller eligible capex total — it is not caused by reallocation across cells. Reallocation across (channel, vintage) cells changes the CRF-weighted figures (`CRF_realised`, `r_impl`) through tenor/channel mix but not `r_vw`. The `all4` and `+rebuild` portfolios additionally move with the commercial-only gas/rebuild capital share (channel mix), which is plan-dependent but is not a change in any channel's rate.

### Test 2 — does the split have real freedom? (limb i)

| run (envelope/variation) | B | drawn | binds | dual | slack AND commercial used | # interior cells | envelope state |
|---|---|---|---|---|---|---|---|
| env_none/base | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |
| env_0.5x/base | 375.0M | 375.0M | yes | -0.3883 | no | 1 | binding |
| env_dares/base | 750.0M | 750.0M | yes | -0.3883 | no | 1 | binding |
| env_5x/base | 3,750.0M | 3,750.0M | yes | -0.3017 | no | 0 | binding |
| env_10x/base | 7,500.0M | 7,500.0M | yes | -0.2444 | no | 0 | binding |
| env_dares/b1_demand_low | 750.0M | 750.0M | yes | -0.3407 | no | 1 | binding |
| env_dares/b1_demand_high | 750.0M | 750.0M | yes | -0.4439 | no | 1 | binding |
| env_dares/b2_gas_downside | 750.0M | 750.0M | yes | -0.415 | no | 1 | binding |
| env_dares/b2_gas_upside | 750.0M | 750.0M | yes | -0.3407 | no | 1 | binding |
| env_dares/b2_gas_shock | 750.0M | 750.0M | yes | -0.415 | no | 1 | binding |
| env_dares/b3_cap_ndc3cond | 750.0M | 750.0M | yes | -0.4794 | no | 1 | binding |
| env_dares/b4_cf_0.16 | 750.0M | 750.0M | yes | -0.3883 | no | 1 | binding |
| env_dares/b4_cf_0.20 | 750.0M | 750.0M | yes | -0.3636 | no | 1 | binding |
| env_none/b1_demand_low | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |
| env_none/b1_demand_high | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |
| env_none/b2_gas_downside | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |
| env_none/b2_gas_upside | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |
| env_none/b2_gas_shock | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |
| env_none/b3_cap_ndc3cond | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |
| env_none/b4_cf_0.16 | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |
| env_none/b4_cf_0.20 | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |

- Runs with the envelope **slack while eligible commercial capital is still raised**: none.
- Runs with an interior share (0<share<1) **while the envelope is slack**: none.
- Runs with interior shares **while the envelope binds**: 10. Distribution across cells (channel vintage:share), first 12 runs:
  - `P1_trad_unc__env_0.5x__base__central`: solar_public 2032:0.466
  - `P1_trad_unc__env_dares__base__central`: solar_public 2032:0.932
  - `P1_trad_unc__env_dares__b1_demand_low__central`: solar_public 2034:0.927
  - `P1_trad_unc__env_dares__b1_demand_high__central`: solar_public 2030:0.355
  - `P1_trad_unc__env_dares__b2_gas_downside__central`: solar_public 2031:0.056
  - `P1_trad_unc__env_dares__b2_gas_upside__central`: solar_public 2034:0.364
  - `P1_trad_unc__env_dares__b2_gas_shock__central`: solar_public 2031:0.609
  - `P1_trad_unc__env_dares__b3_cap_ndc3cond__central`: solar_public 2029:0.136
  - `P1_trad_unc__env_dares__b4_cf_0.16__central`: solar_public 2032:0.720
  - `P1_trad_unc__env_dares__b4_cf_0.20__central`: solar_public 2033:0.010

LP note: with one pooled linear envelope, basic solutions have at most a handful of fractional cells (the marginal ones); a long list of interior shares would indicate degeneracy/alternative optima rather than economic structure. The solver's choice among tied cells is arbitrary.

**Interpretation rule (stated in advance).** If the split is always 'fill the envelope, then commercial' the split is a decision variable in form but near-deterministic in content. If the share varies for reasons other than envelope exhaustion the freedom is substantive.

### Test 5 — negative control: zero envelope vs env_dares across the SAME variations

| variation | ZERO envelope: portfolio r (elig.) | ZERO: portfolio r (all4) | env_dares: portfolio r (elig.) | env_dares: portfolio r (all4) |
|---|---|---|---|---|
| base | 8.8235% | 8.8235% | 8.6604% | 8.6742% |
| b1_demand_low | 8.8235% | 8.8235% | 8.5820% | 8.6095% |
| b1_demand_high | 8.8235% | 8.8235% | 8.7535% | 8.7568% |
| b2_gas_downside | 8.8235% | 8.8235% | 8.6967% | 8.6976% |
| b2_gas_upside | 8.8235% | 8.8235% | 8.6218% | 8.6541% |
| b2_gas_shock | 8.8235% | 8.8235% | 8.6682% | 8.6793% |
| b3_cap_ndc3cond | 8.8235% | 8.8235% | 8.6967% | 8.6989% |
| b4_cf_0.16 | 8.8235% | 8.8235% | 8.6773% | 8.6885% |
| b4_cf_0.20 | 8.8235% | 8.8235% | 8.6552% | 8.6699% |

Zero envelope: range across variations = **0.000 bp** (commercial real rate = 8.8235%). env_dares: range = **17.156 bp**.
The zero-envelope column is the treatment of the review's exogenous studies (all capital at a fixed commercial rate); its invariance is expected by construction and is checked, not assumed.

---

## Panel `P2_eaas_mod` — regime `eaas`, capital_case `moderate`

### Test 1 — envelope sweep (limb ii, weak form; DEMOTED: calibrates the range)

| envelope | B (USD) | conc. share (elig.) | CRF solar_pub | CRF solar_eaas | CRF storage | CRF gas | r_impl solar_pub | r_impl solar_eaas | r_impl storage | r_impl gas | portfolio r_vw (elig.) | portfolio r_vw (all4) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| env_none | 0.0M | 0.00% | 0.15461 | 0.15461 | 0.15461 | 0.15461 | 8.824% | 8.824% | 8.824% | 8.824% | 8.8235% | 8.8235% |
| env_0.5x | 375.0M | 0.94% | 0.15185 | 0.15461 | 0.15461 | 0.15461 | 8.418% | 8.824% | 8.824% | 8.824% | 8.7420% | 8.7489% |
| env_dares | 750.0M | 1.87% | 0.14908 | 0.15461 | 0.15461 | 0.15461 | 8.008% | 8.824% | 8.824% | 8.824% | 8.6604% | 8.6742% |
| env_5x | 3,750.0M | 9.37% | 0.14248 | 0.12705 | 0.15461 | 0.15461 | 7.016% | 4.607% | 8.824% | 8.824% | 8.0080% | 8.0772% |
| env_10x | 7,500.0M | 18.68% | 0.13923 | 0.09943 | 0.15461 | 0.15461 | 6.519% | 0.000% | 8.824% | 8.824% | 7.1974% | 7.3349% |

- `port_elig_r_vw` across the envelope sweep: min 7.1974%, max 8.8235%, **range 162.611 bp**.

- `port_all4_r_vw` across the envelope sweep: min 7.3349%, max 8.8235%, **range 148.862 bp**.

- `port_elig_r_impl_Lcomm` across the envelope sweep: min 5.6022%, max 8.8235%, **range 322.129 bp**.

**Interpretation rule (stated in advance).** If `r_realised` varies while every rate input is fixed, the cost of capital is a quantity the solution determines (limb ii, WEAK form: it responds to the analyst's financing dial). If invariant, limb ii is not satisfied.

### Test 1b — DECISIVE: financing inputs all fixed (envelope = env_dares); vary only demand / gas / cap / solar yield

| variation | solar built (MW) | eligible capex (bn USD) | conc. share (elig.) | portfolio r_vw (elig.) | Δ vs base (bp) | portfolio r_vw (all4) | r_vw (+rebuild) | r_impl@L_comm (elig.) | envelope binds | envelope dual |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 39,030 | 40.027 | 1.874% | 8.6604% | 0.000 | 8.6742% | 8.6873% | 8.5100% | yes | -0.3883 |
| b1_demand_low | 25,464 | 27.028 | 2.775% | 8.5820% | -7.845 | 8.6095% | 8.6335% | 8.3586% | yes | -0.3407 |
| b1_demand_high | 94,760 | 93.256 | 0.804% | 8.7535% | 9.311 | 8.7568% | 8.7615% | 8.6892% | yes | -0.4439 |
| b2_gas_downside | 51,553 | 51.482 | 1.457% | 8.6967% | 3.630 | 8.6976% | 8.7105% | 8.5800% | yes | -0.415 |
| b2_gas_upside | 30,146 | 32.375 | 2.317% | 8.6218% | -3.856 | 8.6541% | 8.6704% | 8.4356% | yes | -0.3407 |
| b2_gas_shock | 40,732 | 42.028 | 1.785% | 8.6682% | 0.777 | 8.6793% | 8.6921% | 8.5250% | yes | -0.415 |
| b3_cap_ndc3cond | 49,112 | 51.465 | 1.457% | 8.6967% | 3.625 | 8.6989% | 8.7118% | 8.5799% | yes | -0.4794 |
| b4_cf_0.16 | 46,442 | 44.638 | 1.680% | 8.6773% | 1.685 | 8.6885% | 8.6992% | 8.5425% | yes | -0.3883 |
| b4_cf_0.20 | 37,054 | 38.797 | 1.933% | 8.6552% | -0.517 | 8.6699% | 8.6837% | 8.5001% | yes | -0.3636 |

**Magnitude across base + b1–b4:** range of portfolio real rate = **17.156 bp** (eligible), 14.726 bp (all four channels), 12.805 bp (incl. rebuild/conversion). Numerical-noise floor taken as 0.01 bp.

**Reading (eligible portfolio):** the realised rate MOVES by more than the noise floor → limb (ii) STRONG form is supported on this panel (the realised cost of capital responds to quantities with no financing content). Whether that movement is economically large is the bp figure above; this script does not set a threshold for 'large'.

**Mechanism note — cell-level allocation of the pooled envelope.** (`full` = share 1; `interior` = vintage:share; `none` = share 0; vintage index = model year.)

| variation | allocation of concessional dollars across (channel, vintage) |
|---|---|
| base | solar_public: full[-] interior[2032:0.932] none[2033,2037-2045,2048-2049,2053-2054] | solar_eaas: full[-] interior[-] none[2034-2036,2046-2047,2050-2052] | storage: full[-] interior[-] none[2037-2054] |
| b1_demand_low | solar_public: full[2034] interior[-] none[2037-2045,2048-2049,2053-2054] | solar_eaas: full[2033] interior[-] none[2034-2036,2046-2047,2050-2052] | storage: full[-] interior[-] none[2038-2054] |
| b1_demand_high | solar_eaas: full[2029] interior[2030:0.355] none[2031-2054] | storage: full[-] interior[-] none[2036-2054] |
| b2_gas_downside | solar_public: full[-] interior[2031:0.087] none[2033,2038-2039,2041-2043,2045,2048,2053-2054] | solar_eaas: full[2030] interior[-] none[2031-2032,2034-2037,2040,2044,2046-2047,2049-2052] | storage: full[-] interior[-] none[2036-2054] |
| b2_gas_upside | solar_public: full[2034] interior[-] none[2037-2045,2048-2049,2053-2054] | solar_eaas: full[2033] interior[-] none[2034-2036,2046-2047,2050-2052] | storage: full[-] interior[-] none[2038-2054] |
| b2_gas_shock | solar_public: full[-] interior[2031:0.609] none[2032-2033,2038-2045,2048-2049,2053-2054] | solar_eaas: full[-] interior[-] none[2034-2037,2044,2046-2047,2050-2052] | storage: full[-] interior[-] none[2037-2054] |
| b3_cap_ndc3cond | solar_public: full[-] interior[-] none[2031,2033,2038-2039,2041-2043,2045,2048,2053-2054] | solar_eaas: full[2028] interior[2029:0.136] none[2030-2032,2034-2037,2040,2044,2046-2047,2049-2052] | storage: full[-] interior[-] none[2032-2033,2036-2054] |
| b4_cf_0.16 | solar_public: full[-] interior[2032:0.720] none[2033,2037-2045,2048-2049,2053-2054] | solar_eaas: full[-] interior[-] none[2034-2036,2046-2047,2050-2052] | storage: full[-] interior[-] none[2037-2054] |
| b4_cf_0.20 | solar_public: full[2032] interior[-] none[2033,2037-2045,2048-2049,2053-2054] | solar_eaas: full[2033] interior[-] none[2034-2036,2046-2047,2050-2052] | storage: full[-] interior[-] none[2037-2054] |

The eligible-portfolio `r_vw` is an exact function of the eligible concessional share (`r_comm − share·(r_comm − r_conc)`, both techs share the same two real rates), so movement in it can arise ONLY from the pooled fixed envelope being spread over a larger or smaller eligible capex total — it is not caused by reallocation across cells. Reallocation across (channel, vintage) cells changes the CRF-weighted figures (`CRF_realised`, `r_impl`) through tenor/channel mix but not `r_vw`. The `all4` and `+rebuild` portfolios additionally move with the commercial-only gas/rebuild capital share (channel mix), which is plan-dependent but is not a change in any channel's rate.

### Test 2 — does the split have real freedom? (limb i)

| run (envelope/variation) | B | drawn | binds | dual | slack AND commercial used | # interior cells | envelope state |
|---|---|---|---|---|---|---|---|
| env_none/base | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |
| env_0.5x/base | 375.0M | 375.0M | yes | -0.3883 | no | 1 | binding |
| env_dares/base | 750.0M | 750.0M | yes | -0.3883 | no | 1 | binding |
| env_5x/base | 3,750.0M | 3,750.0M | yes | -0.3017 | no | 0 | binding |
| env_10x/base | 7,500.0M | 7,500.0M | yes | -0.2444 | no | 0 | binding |
| env_dares/b1_demand_low | 750.0M | 750.0M | yes | -0.3407 | no | 0 | binding |
| env_dares/b1_demand_high | 750.0M | 750.0M | yes | -0.4439 | no | 1 | binding |
| env_dares/b2_gas_downside | 750.0M | 750.0M | yes | -0.415 | no | 1 | binding |
| env_dares/b2_gas_upside | 750.0M | 750.0M | yes | -0.3407 | no | 0 | binding |
| env_dares/b2_gas_shock | 750.0M | 750.0M | yes | -0.415 | no | 1 | binding |
| env_dares/b3_cap_ndc3cond | 750.0M | 750.0M | yes | -0.4794 | no | 1 | binding |
| env_dares/b4_cf_0.16 | 750.0M | 750.0M | yes | -0.3883 | no | 1 | binding |
| env_dares/b4_cf_0.20 | 750.0M | 750.0M | yes | -0.3636 | no | 0 | binding |
| env_none/b1_demand_low | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |
| env_none/b1_demand_high | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |
| env_none/b2_gas_downside | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |
| env_none/b2_gas_upside | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |
| env_none/b2_gas_shock | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |
| env_none/b3_cap_ndc3cond | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |
| env_none/b4_cf_0.16 | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |
| env_none/b4_cf_0.20 | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |

- Runs with the envelope **slack while eligible commercial capital is still raised**: none.
- Runs with an interior share (0<share<1) **while the envelope is slack**: none.
- Runs with interior shares **while the envelope binds**: 7. Distribution across cells (channel vintage:share), first 12 runs:
  - `P2_eaas_mod__env_0.5x__base__central`: solar_public 2032:0.466
  - `P2_eaas_mod__env_dares__base__central`: solar_public 2032:0.932
  - `P2_eaas_mod__env_dares__b1_demand_high__central`: solar_eaas 2030:0.355
  - `P2_eaas_mod__env_dares__b2_gas_downside__central`: solar_public 2031:0.087
  - `P2_eaas_mod__env_dares__b2_gas_shock__central`: solar_public 2031:0.609
  - `P2_eaas_mod__env_dares__b3_cap_ndc3cond__central`: solar_eaas 2029:0.136
  - `P2_eaas_mod__env_dares__b4_cf_0.16__central`: solar_public 2032:0.720

LP note: with one pooled linear envelope, basic solutions have at most a handful of fractional cells (the marginal ones); a long list of interior shares would indicate degeneracy/alternative optima rather than economic structure. The solver's choice among tied cells is arbitrary.

**Interpretation rule (stated in advance).** If the split is always 'fill the envelope, then commercial' the split is a decision variable in form but near-deterministic in content. If the share varies for reasons other than envelope exhaustion the freedom is substantive.

### Test 3 — does the bankability constraint interact with the capacity solution? (limb iii)

| configuration | dscr_case | EaaS solar (MW) | total solar (MW) | npv_cost (USD) | binding vintage-years / vintages with EaaS | subsidy drawn (undisc.) | subsidy (NPV) | public budget binds |
|---|---|---|---|---|---|---|---|---|
| (a) central dscr | dscr_central | 14,329.6 | 39,030.1 | 27,317,670,155 | 3/8 | 939.1M | 122.7M | no |
| (b) loosest dscr (dscr_regression, 1.10) | dscr_regression | 14,329.6 | 39,030.1 | 27,317,670,155 | 2/8 | 516.1M | 65.8M | no |
| (b2) tighter dscr_high (1.40) | dscr_high | 14,329.6 | 39,030.1 | 27,317,670,155 | 3/8 | 1,146.8M | 151.1M | no |
| (c) bankability OFF | dscr_central | 16,892.2 | 39,030.1 | 27,317,670,160 | 4/9 | 0.0M | 0.0M | no |

**(c) − (a):** Δ EaaS solar = 2,562.64 MW; Δ total solar = 0.00 MW; Δ npv_cost = 4 USD (0.000000%).
Reading: the capacity solution DIFFERS when the constraint is removed → on this panel the constraint interacts with the capacity solution (limb iii supported here).

**On the subsidy under (c).** `eaas_subsidy` has NO objective coefficient anywhere in the model (Part A, A1); it appears only in `eaas_bankability`, `eaas_subsidy_limit` and (if built) the public budget. With bankability deactivated it has no constraint to relax, so a zero (or any value within `eaas_subsidy_limit`) is expected and the reported value carries no information — it is not a solver artefact and not a finding. The same zero-cost property holds under (a): the subsidy is free to the objective, and is limited only by `gap·add` and, where built, the public budget. **Structural prediction to check against the table:** when the public budget is not built, a free subsidy up to `gap·add` makes the bankability constraint satisfiable at equality for any build, so removing it should not change capacity; the constraint can interact with capacity only through the budget (compare the two eaas panels).

### Test 5 — negative control: zero envelope vs env_dares across the SAME variations

| variation | ZERO envelope: portfolio r (elig.) | ZERO: portfolio r (all4) | env_dares: portfolio r (elig.) | env_dares: portfolio r (all4) |
|---|---|---|---|---|
| base | 8.8235% | 8.8235% | 8.6604% | 8.6742% |
| b1_demand_low | 8.8235% | 8.8235% | 8.5820% | 8.6095% |
| b1_demand_high | 8.8235% | 8.8235% | 8.7535% | 8.7568% |
| b2_gas_downside | 8.8235% | 8.8235% | 8.6967% | 8.6976% |
| b2_gas_upside | 8.8235% | 8.8235% | 8.6218% | 8.6541% |
| b2_gas_shock | 8.8235% | 8.8235% | 8.6682% | 8.6793% |
| b3_cap_ndc3cond | 8.8235% | 8.8235% | 8.6967% | 8.6989% |
| b4_cf_0.16 | 8.8235% | 8.8235% | 8.6773% | 8.6885% |
| b4_cf_0.20 | 8.8235% | 8.8235% | 8.6552% | 8.6699% |

Zero envelope: range across variations = **0.000 bp** (commercial real rate = 8.8235%). env_dares: range = **17.156 bp**.
The zero-envelope column is the treatment of the review's exogenous studies (all capital at a fixed commercial rate); its invariance is expected by construction and is checked, not assumed.

---

## Panel `P3_eaas_unc` — regime `eaas`, capital_case `unconstrained`

### Test 1 — envelope sweep (limb ii, weak form; DEMOTED: calibrates the range)

| envelope | B (USD) | conc. share (elig.) | CRF solar_pub | CRF solar_eaas | CRF storage | CRF gas | r_impl solar_pub | r_impl solar_eaas | r_impl storage | r_impl gas | portfolio r_vw (elig.) | portfolio r_vw (all4) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| env_none | 0.0M | 0.00% | 0.15461 | 0.15461 | 0.15461 | 0.15461 | 8.824% | 8.824% | 8.824% | 8.824% | 8.8235% | 8.8235% |
| env_0.5x | 375.0M | 0.94% | 0.15146 | 0.15461 | 0.15461 | 0.15461 | 8.360% | 8.824% | 8.824% | 8.824% | 8.7420% | 8.7489% |
| env_dares | 750.0M | 1.87% | 0.14831 | 0.15461 | 0.15461 | 0.15461 | 7.893% | 8.824% | 8.824% | 8.824% | 8.6604% | 8.6742% |
| env_5x | 3,750.0M | 9.37% | 0.13992 | 0.13353 | 0.15461 | 0.15461 | 6.626% | 5.636% | 8.824% | 8.824% | 8.0080% | 8.0772% |
| env_10x | 7,500.0M | 18.68% | 0.13768 | 0.10407 | 0.15461 | 0.15461 | 6.281% | 0.731% | 8.824% | 8.824% | 7.1974% | 7.3349% |

- `port_elig_r_vw` across the envelope sweep: min 7.1974%, max 8.8235%, **range 162.611 bp**.

- `port_all4_r_vw` across the envelope sweep: min 7.3349%, max 8.8235%, **range 148.862 bp**.

- `port_elig_r_impl_Lcomm` across the envelope sweep: min 5.6022%, max 8.8235%, **range 322.129 bp**.

**Interpretation rule (stated in advance).** If `r_realised` varies while every rate input is fixed, the cost of capital is a quantity the solution determines (limb ii, WEAK form: it responds to the analyst's financing dial). If invariant, limb ii is not satisfied.

### Test 1b — DECISIVE: financing inputs all fixed (envelope = env_dares); vary only demand / gas / cap / solar yield

| variation | solar built (MW) | eligible capex (bn USD) | conc. share (elig.) | portfolio r_vw (elig.) | Δ vs base (bp) | portfolio r_vw (all4) | r_vw (+rebuild) | r_impl@L_comm (elig.) | envelope binds | envelope dual |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 39,030 | 40.027 | 1.874% | 8.6604% | 0.000 | 8.6742% | 8.6873% | 8.5100% | yes | -0.3883 |
| b1_demand_low | 25,464 | 27.028 | 2.775% | 8.5820% | -7.845 | 8.6095% | 8.6335% | 8.3586% | yes | -0.3407 |
| b1_demand_high | 94,760 | 93.257 | 0.804% | 8.7535% | 9.311 | 8.7568% | 8.7616% | 8.6892% | yes | -0.4439 |
| b2_gas_downside | 51,553 | 51.482 | 1.457% | 8.6967% | 3.630 | 8.6976% | 8.7105% | 8.5800% | yes | -0.415 |
| b2_gas_upside | 30,146 | 32.375 | 2.317% | 8.6218% | -3.856 | 8.6541% | 8.6704% | 8.4356% | yes | -0.3407 |
| b2_gas_shock | 40,732 | 42.028 | 1.785% | 8.6682% | 0.777 | 8.6793% | 8.6921% | 8.5250% | yes | -0.415 |
| b3_cap_ndc3cond | 49,112 | 51.465 | 1.457% | 8.6967% | 3.625 | 8.6989% | 8.7118% | 8.5799% | yes | -0.4794 |
| b4_cf_0.16 | 46,442 | 44.638 | 1.680% | 8.6773% | 1.685 | 8.6885% | 8.6992% | 8.5425% | yes | -0.3883 |
| b4_cf_0.20 | 37,054 | 38.797 | 1.933% | 8.6552% | -0.517 | 8.6699% | 8.6837% | 8.5001% | yes | -0.3636 |

**Magnitude across base + b1–b4:** range of portfolio real rate = **17.156 bp** (eligible), 14.726 bp (all four channels), 12.812 bp (incl. rebuild/conversion). Numerical-noise floor taken as 0.01 bp.

**Reading (eligible portfolio):** the realised rate MOVES by more than the noise floor → limb (ii) STRONG form is supported on this panel (the realised cost of capital responds to quantities with no financing content). Whether that movement is economically large is the bp figure above; this script does not set a threshold for 'large'.

**Mechanism note — cell-level allocation of the pooled envelope.** (`full` = share 1; `interior` = vintage:share; `none` = share 0; vintage index = model year.)

| variation | allocation of concessional dollars across (channel, vintage) |
|---|---|
| base | solar_public: full[-] interior[2032:0.932] none[2033,2035,2037,2039,2041-2042,2044,2048,2050-2053] | solar_eaas: full[-] interior[-] none[2034,2036,2038,2040,2043,2045-2047,2049,2054] | storage: full[-] interior[-] none[2037-2054] |
| b1_demand_low | solar_public: full[2034] interior[-] none[2035,2037,2039,2041-2042,2044,2048,2050-2053] | solar_eaas: full[2033] interior[-] none[2034,2036,2038,2040,2043,2045-2047,2049,2054] | storage: full[-] interior[-] none[2038-2054] |
| b1_demand_high | solar_public: full[2029] interior[2030:0.355] none[2031-2033,2035,2037,2039,2041-2042,2044,2048,2050-2053] | solar_eaas: full[-] interior[-] none[2034,2036,2038,2040,2043,2045-2047,2049,2054] | storage: full[-] interior[-] none[2036-2054] |
| b2_gas_downside | solar_public: full[2030] interior[2031:0.056] none[2032-2033,2035,2037,2039,2041-2042,2044,2048,2050-2053] | solar_eaas: full[-] interior[-] none[2034,2036,2038,2040,2043,2045-2047,2049,2054] | storage: full[-] interior[-] none[2036-2054] |
| b2_gas_upside | solar_public: full[2034] interior[-] none[2035,2037,2039,2041-2042,2044,2048,2050-2053] | solar_eaas: full[2033] interior[-] none[2034,2036,2038,2040,2043,2045-2047,2049,2054] | storage: full[-] interior[-] none[2038-2054] |
| b2_gas_shock | solar_public: full[-] interior[2031:0.609] none[2032-2033,2035,2037,2039,2041-2042,2044,2048,2050-2053] | solar_eaas: full[-] interior[-] none[2034,2036,2038,2040,2043,2045-2047,2049,2054] | storage: full[-] interior[-] none[2037-2054] |
| b3_cap_ndc3cond | solar_public: full[2028] interior[2029:0.136] none[2030-2033,2035,2037,2039,2041-2042,2044,2048,2050-2053] | solar_eaas: full[-] interior[-] none[2034,2036,2038,2040,2043,2045-2047,2049,2054] | storage: full[-] interior[-] none[2032-2033,2036-2054] |
| b4_cf_0.16 | solar_public: full[-] interior[2032:0.720] none[2033-2039,2041,2043-2044,2048-2053] | solar_eaas: full[-] interior[-] none[2040,2042,2045-2047,2054] | storage: full[-] interior[-] none[2037-2054] |
| b4_cf_0.20 | solar_public: full[2032] interior[-] none[2033-2035,2037,2039,2041,2043-2044,2046,2050-2053] | solar_eaas: full[2033] interior[-] none[2036,2038,2040,2042,2045,2047-2049,2054] | storage: full[-] interior[-] none[2037-2054] |

The eligible-portfolio `r_vw` is an exact function of the eligible concessional share (`r_comm − share·(r_comm − r_conc)`, both techs share the same two real rates), so movement in it can arise ONLY from the pooled fixed envelope being spread over a larger or smaller eligible capex total — it is not caused by reallocation across cells. Reallocation across (channel, vintage) cells changes the CRF-weighted figures (`CRF_realised`, `r_impl`) through tenor/channel mix but not `r_vw`. The `all4` and `+rebuild` portfolios additionally move with the commercial-only gas/rebuild capital share (channel mix), which is plan-dependent but is not a change in any channel's rate.

### Test 2 — does the split have real freedom? (limb i)

| run (envelope/variation) | B | drawn | binds | dual | slack AND commercial used | # interior cells | envelope state |
|---|---|---|---|---|---|---|---|
| env_none/base | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |
| env_0.5x/base | 375.0M | 375.0M | yes | -0.3883 | no | 1 | binding |
| env_dares/base | 750.0M | 750.0M | yes | -0.3883 | no | 1 | binding |
| env_5x/base | 3,750.0M | 3,750.0M | yes | -0.3017 | no | 0 | binding |
| env_10x/base | 7,500.0M | 7,500.0M | yes | -0.2444 | no | 0 | binding |
| env_dares/b1_demand_low | 750.0M | 750.0M | yes | -0.3407 | no | 0 | binding |
| env_dares/b1_demand_high | 750.0M | 750.0M | yes | -0.4439 | no | 1 | binding |
| env_dares/b2_gas_downside | 750.0M | 750.0M | yes | -0.415 | no | 1 | binding |
| env_dares/b2_gas_upside | 750.0M | 750.0M | yes | -0.3407 | no | 0 | binding |
| env_dares/b2_gas_shock | 750.0M | 750.0M | yes | -0.415 | no | 1 | binding |
| env_dares/b3_cap_ndc3cond | 750.0M | 750.0M | yes | -0.4794 | no | 1 | binding |
| env_dares/b4_cf_0.16 | 750.0M | 750.0M | yes | -0.3883 | no | 1 | binding |
| env_dares/b4_cf_0.20 | 750.0M | 750.0M | yes | -0.3636 | no | 0 | binding |
| env_none/b1_demand_low | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |
| env_none/b1_demand_high | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |
| env_none/b2_gas_downside | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |
| env_none/b2_gas_upside | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |
| env_none/b2_gas_shock | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |
| env_none/b3_cap_ndc3cond | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |
| env_none/b4_cf_0.16 | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |
| env_none/b4_cf_0.20 | 0.0M | 0.0M | yes | -0.6522 | no | 0 | binding |

- Runs with the envelope **slack while eligible commercial capital is still raised**: none.
- Runs with an interior share (0<share<1) **while the envelope is slack**: none.
- Runs with interior shares **while the envelope binds**: 7. Distribution across cells (channel vintage:share), first 12 runs:
  - `P3_eaas_unc__env_0.5x__base__central`: solar_public 2032:0.466
  - `P3_eaas_unc__env_dares__base__central`: solar_public 2032:0.932
  - `P3_eaas_unc__env_dares__b1_demand_high__central`: solar_public 2030:0.355
  - `P3_eaas_unc__env_dares__b2_gas_downside__central`: solar_public 2031:0.056
  - `P3_eaas_unc__env_dares__b2_gas_shock__central`: solar_public 2031:0.609
  - `P3_eaas_unc__env_dares__b3_cap_ndc3cond__central`: solar_public 2029:0.136
  - `P3_eaas_unc__env_dares__b4_cf_0.16__central`: solar_public 2032:0.720

LP note: with one pooled linear envelope, basic solutions have at most a handful of fractional cells (the marginal ones); a long list of interior shares would indicate degeneracy/alternative optima rather than economic structure. The solver's choice among tied cells is arbitrary.

**Interpretation rule (stated in advance).** If the split is always 'fill the envelope, then commercial' the split is a decision variable in form but near-deterministic in content. If the share varies for reasons other than envelope exhaustion the freedom is substantive.

### Test 3 — does the bankability constraint interact with the capacity solution? (limb iii)

| configuration | dscr_case | EaaS solar (MW) | total solar (MW) | npv_cost (USD) | binding vintage-years / vintages with EaaS | subsidy drawn (undisc.) | subsidy (NPV) | public budget binds |
|---|---|---|---|---|---|---|---|---|
| (a) central dscr | dscr_central | 17,223.3 | 39,030.1 | 27,317,670,159 | 1/10 | 1,131.1M | 121.4M | not built |
| (b) loosest dscr (dscr_regression, 1.10) | dscr_regression | 13,982.7 | 39,030.1 | 27,317,670,158 | 1/8 | 1,074.4M | 115.3M | not built |
| (b2) tighter dscr_high (1.40) | dscr_high | 6,584.2 | 39,030.1 | 27,317,670,160 | 0/4 | 0.0M | 0.0M | not built |
| (c) bankability OFF | dscr_central | 10,485.9 | 39,030.1 | 27,317,670,161 | 3/7 | 0.0M | 0.0M | not built |

**(c) − (a):** Δ EaaS solar = -6,737.42 MW; Δ total solar = -0.00 MW; Δ npv_cost = 2 USD (0.000000%).
Reading: the capacity solution DIFFERS when the constraint is removed → on this panel the constraint interacts with the capacity solution (limb iii supported here).

**On the subsidy under (c).** `eaas_subsidy` has NO objective coefficient anywhere in the model (Part A, A1); it appears only in `eaas_bankability`, `eaas_subsidy_limit` and (if built) the public budget. With bankability deactivated it has no constraint to relax, so a zero (or any value within `eaas_subsidy_limit`) is expected and the reported value carries no information — it is not a solver artefact and not a finding. The same zero-cost property holds under (a): the subsidy is free to the objective, and is limited only by `gap·add` and, where built, the public budget. **Structural prediction to check against the table:** when the public budget is not built, a free subsidy up to `gap·add` makes the bankability constraint satisfiable at equality for any build, so removing it should not change capacity; the constraint can interact with capacity only through the budget (compare the two eaas panels).

### Test 5 — negative control: zero envelope vs env_dares across the SAME variations

| variation | ZERO envelope: portfolio r (elig.) | ZERO: portfolio r (all4) | env_dares: portfolio r (elig.) | env_dares: portfolio r (all4) |
|---|---|---|---|---|
| base | 8.8235% | 8.8235% | 8.6604% | 8.6742% |
| b1_demand_low | 8.8235% | 8.8235% | 8.5820% | 8.6095% |
| b1_demand_high | 8.8235% | 8.8235% | 8.7535% | 8.7568% |
| b2_gas_downside | 8.8235% | 8.8235% | 8.6967% | 8.6976% |
| b2_gas_upside | 8.8235% | 8.8235% | 8.6218% | 8.6541% |
| b2_gas_shock | 8.8235% | 8.8235% | 8.6682% | 8.6793% |
| b3_cap_ndc3cond | 8.8235% | 8.8235% | 8.6967% | 8.6989% |
| b4_cf_0.16 | 8.8235% | 8.8235% | 8.6773% | 8.6885% |
| b4_cf_0.20 | 8.8235% | 8.8235% | 8.6552% | 8.6699% |

Zero envelope: range across variations = **0.000 bp** (commercial real rate = 8.8235%). env_dares: range = **17.156 bp**.
The zero-envelope column is the treatment of the review's exogenous studies (all capital at a fixed commercial rate); its invariance is expected by construction and is checked, not assumed.

---

## Test 4 — risk-to-rate feedback (restated from Part A, A4)

No rate, CRF, tenor or discount window in either builder depends on technology mix, capacity, emissions, reliability or any other solved quantity (A3, A4; `optimize_model_sliced.py` 1559-1634, `blended_finance.py` 212-244). The risk-to-rate limb is **absent**. Test 1b is consistent with this: any movement it shows is a change in the volume weights on two fixed rates, not a change in either rate.

## Runs that did not produce a result

None.

<!-- END PART B AUTOGEN -->

<!-- PART C (hand-written; below the Part B autogen block) -->

# Part C — classification, and what Part B settles

*Written by hand from the Part B autogen output.*

## C1. Classification against the review's definition

| Limb | Verdict | Evidence |
|---|---|---|
| **(i)** financing structure is a decision variable | **Satisfied in form; near-deterministic in content** | Eight tranche-volume `Var`s (A1, A2). But the envelope binds in **71 of 71 runs**, so the slack case was not reached and Part A's prediction (no interior share under a slack envelope) is **untested** — see Part C of `docs/eaas_degeneracy.md`. At most one interior cell per run. The aggregate rule is "fill the envelope, then borrow commercially" in every solved case. |
| **(ii) strong form** — cost of capital responds to the plan | **Satisfied** | With every financing input fixed, the portfolio realised real rate moves **17.156 bp** across the demand, gas, cap and solar-yield variations (8.5820%–8.7535%). Noise floor 0.01 bp. Identical on all three panels. |
| **(ii) weak form** — responds to financing inputs | **Satisfied** | 162.611 bp across the envelope sweep (8.8235% at zero envelope → 7.1974% at 10× DARES). |
| **(iii)** financing enters as a constraint interacting with the capacity solution | **Not established** | See C3. Total solar and `npv_cost` are invariant across every bankability configuration; only the public/EaaS split moves, and it moves non-monotonically. |
| *(not in the definition)* rate responds to the plan's risk | **Absent** | A3, A4, Test 4. No rate, CRF, tenor or window depends on any solved quantity. |

**Negative control.** At a zero envelope the realised rate is 8.8235% across every
plan variation — range **0.000 bp**. That is the treatment the review found in 65
of 75 studies. At the DARES envelope the same variations produce a 17.156 bp
spread. The contrast is the cleanest single statement of what this model does
that the corpus does not.

## C2. What the numbers mean, stated at their actual size

The mechanism is real and the magnitude is small, and both halves need saying.

At the DARES envelope, concessional capital is **1.87% of eligible capex**
(750 M USD against 40.0 bn), and the realised portfolio cost of capital sits
**16.3 bp below** the commercial rate. Across the full range of plans tested it
spans **17.2 bp** — so the variation the plan induces in the cost of capital is
about the same size as the entire concessional effect at that envelope.

The mechanism has force when the envelope is larger: at 10× DARES the realised
rate falls 162.6 bp below commercial. It is the envelope that is small, not the
machinery. **That is itself a result worth reporting** — the DARES
facility's deployable portion [TASK 57: wording stale -- the model applies no deployable fraction; 750M is the whole IDA amount, identity at 1.0] is 1.87% of the plan's concessional-eligible
capital — solar and storage only, excluding gas, rebuild and conversion capex,
none of which can draw on the envelope — and at that scale it moves the portfolio cost of capital by under
20 bp however the plan is drawn. A blended-finance instrument sized like DARES
cannot move the cost of capital of a programme this size, and the model measures
by how little.

**The mechanism of the strong-form result, corrected.** Movement in the
portfolio rate comes from a fixed concessional pool being spread over a larger or
smaller eligible capex total as the plan changes — not from reallocation across
channel and vintage cells, which cannot move a volume-weighted average of two
scalar rates. Reallocation does move the CRF-weighted figures through tenor mix.

## C3. Limb (iii) is not established, and the reason matters beyond this question

The Part B script read Test 3 by its pre-stated rule — the capacity solution
differs when the constraint is removed, therefore the constraint interacts with
capacity. Applied to this output the rule misfires, because the quantity that
differs is degenerate.

| | P2 (eaas, moderate budget) | P3 (eaas, unconstrained) |
|---|---|---|
| (a) central dscr — EaaS solar | 14,329.6 MW | 17,223.3 MW |
| (b) loosest dscr (1.10) | 14,329.6 MW | 13,982.7 MW |
| (b2) tighter dscr (1.40) | 14,329.6 MW | 6,584.2 MW |
| (c) bankability OFF | 16,892.2 MW | 10,485.9 MW |
| **total solar, all four** | **39,030.1 MW** | **39,030.1 MW** |
| **npv_cost, all four** | **27,317,670,155–160 USD** | **27,317,670,158–161 USD** |

Three things are wrong with reading this as limb (iii).

**Total solar is identical across every configuration, on both panels** (to
within 6e-6 MW). Removing a constraint changes no physical capacity.

**`npv_cost` differs by at most 5.5 USD out of 27.3 bn** — solver tolerance, not economics.

**The EaaS share moves non-monotonically.** On P3, tightening the coverage margin
from 1.10 to 1.30 to 1.40 gives 13,983 → 17,223 → 6,584 MW, and *removing* the
constraint gives 10,486 MW — less than leaving it on at the central margin. A
constraint that binds does not behave like that. These are alternative optima.

The natural reading — **a hypothesis, tested in `docs/eaas_degeneracy.md`, not a result here** — is that **public solar and EaaS solar are near-perfect
substitutes in the objective at this configuration**, so the split between them
is not determined, and the solver's choice among tied bases is arbitrary. Code
predicted the structural reason in advance: `eaas_subsidy` carries no objective
coefficient, so a free subsidy up to `gap·add` can satisfy bankability at
equality for any build, and the constraint cannot bite on cost. The public budget
is built on P2 but does not bind, so it does not rescue the test either.

This is consistent with **F-38** — EaaS worth 0.11 bn against a 27.318 bn anchor
[VERIFIED against `findings_frozen.md`: anchor 27,317,670,155.72 at line 33; F-38 table 0.109B at 3.9% growth / baseline / no_policy]
at the central no-policy case, which the handover already flags as possibly zero
within the model's identified band. Part B's Test 1b corroborates it from a
different direction: the traditional and EaaS panels produce the *same* 39,030 MW
solar build and the same 40.027 bn eligible capex at base.

**Consequence.** Limb (iii) may well hold where something binds — under a cap,
under a binding public budget, under a tight capital envelope — which is exactly
the conditionality the Chapter 1 handover already identifies as the contribution.
But it is **not** established by this test, at this configuration, and should not
be claimed from it.

## C4. Plain-language statement, for Chapter 2 §2.7.1 and Chapter 1

> The model determines the structure of its own financing and, through it, its
> own cost of capital. Capital for each technology and vintage is split between a
> concessional and a commercial tranche, both decision variables, drawing on a
> concessional envelope that is scarce and pooled across channels and vintages.
> The two tranche rates are fixed inputs, so the model does not solve for a rate
> and no part of it makes a riskier plan face a dearer one. What it solves for is
> how much of the plan each source funds — and because the envelope is fixed
> while the plan is not, the realised cost of capital is an outcome of the
> optimisation rather than a number supplied to it. Holding every financing input
> constant and varying only demand, gas deliverability, the emissions cap and
> solar yield, the realised portfolio cost of capital moves by 17 basis points;
> the same variations leave it exactly constant when the concessional envelope is
> set to zero, which is how the exogenous treatments in the reviewed corpus
> behave. The effect is small at the envelope actually on offer — concessional
> capital is under 2% of the plan's concessional-eligible capital, and moves the portfolio
> rate 16 basis points below commercial — and reaches 163 basis points at ten
> times that envelope.

## C5. The check this raises

The degeneracy in C3 is not a question about endogeneity and should not be
settled inside this task. It is a bounded, separable question, and it touches
F-34 to F-38 and §3.4.4:

> Are public solar and EaaS solar alternative optima at the central
> configuration, and if so over what range of policy stringency does the
> degeneracy persist?

A minimal test: re-solve the central EaaS case with a perturbation of order
1e-6 on the EaaS capital coefficient and report whether the split jumps; report
the reduced costs on `solar_public_add` and `solar_eaas_add`; and repeat under a
binding cap and a binding capital envelope, where the handover's F-35 says EaaS
value is large. If the split is determinate wherever EaaS value is non-zero, the
degeneracy is confined to the no-policy case and is a footnote. If it persists,
the EaaS capacity numbers reported anywhere in Chapter 4 need a stability
statement beside them.

**This belongs in the limitations register either way**, with the Test 3 table as
its evidence.

## C6. Verification notes (added by Code when appending Part C)

Every figure in C1–C3 was re-checked against `results/financing_endogeneity/runs.csv`
(71/71 optimal, envelope binds 71/71, 17.156 bp, 162.611 bp, 0.000 bp, the Test 3
table, 1.87%, 16.3 bp, 40.027 bn, 39,030.1 MW; max interior cells per run = 1). Two edits to the
text as supplied: the `27,317,670,15X` placeholder is filled from the runs, and "identical to the
decimal" became "within 6e-6 MW" (P3 (c) is 39,030.065094 vs .065100). Points the text should
carry or soften:

1. **Limb (i) rests on binding-only evidence.** The envelope binds in every run, including 10× DARES
   (7.5 bn against 40.0 bn eligible), so Part A's prediction — no interior share with a slack envelope —
   was never exercised. "No run shows an interior share while slack" is true but vacuous. The
   "near-deterministic" verdict stands on the binding runs; a slack case (an envelope above ~40 bn) was not run.
2. **C2/C4 overreach.** "The concessional finance actually on offer for Nigerian power" is one facility,
   the 750 M DARES figure (the 1,050 M arm is stipulated and unsourced), and "the capital the plan requires"
   is eligible capex only (solar + storage, undiscounted; gas, rebuild and conversion excluded). Suggest
   "under 2% of the plan's concessional-eligible capital" in C4.
3. **C3 does not establish substitutability, only compatibility with it.** Alternative optima were inferred
   from non-monotonic splits and invariant totals, not tested; C5 is the test. The F-38 figures were
   subsequently checked against `findings_frozen.md` (see C3); they were not re-derived from the solve outputs.
4. **Bankability "binding vintage-years" in the Test 3 table is a lhs/rhs equality count,** and is non-zero
   even under (c) where the constraint is deactivated. It reflects a tied value at the solution, not the
   constraint acting, and should not be cited.
5. **Strong-form (ii) wording.** The 17 bp is a change in capital-volume weights on two fixed rates (C2 says
   this). It supports "an outcome of the optimisation" but not "responds to risk".
