# Findings — frozen state

**Model freeze.** Branch `phase-fin2-d17`, HEAD **`2a9226b`**,
2026-09-07. All figures produced by the sweep at this HEAD.

> **Phase C (2026-10-10, HEAD `3f0b382`; recorded 2026-10-18, TASK 48 B.5).** Phase C ran: Group 0 (the regression gate, `logs/phase_c/0_*.log`), Group 1 (O&M arms `om_low`, `om_high`), Group 2 (the NDC caps with the decline arm), Group 3 (the uncapped solar design, `logs/phase_c/3_*.log`), about 1,409 solves in all. Until this note the findings below did not record it. Consequences already applied: F-21 is rewritten at the current epoch from Group 2 (B.3); the `*_uncapped` result directories exist and are a different solar design from the capped ones (register L79, L80, L81). **Group 1 hit a hash collision** (register L85): `om_case` was not in the hashed arm tag, so `om_low` and `om_high` both wrote `results/baseline_sliced/summary__02c17f55.json`, the `om_low` result (26.909 bn, in `logs/phase_c/1_A_om_low.log` only) was overwritten, and that file now holds **`om_high`** (27,884,317,679.69), not the central baseline. Every "Current values: `summary__02c17f55.json`" pointer in this header is stale until the three-solve re-run (`bash scripts/run_phase_c.sh --only 4`, Group 4) has been run at the fixed hash. Fix: commit `5aad6ff`.

**Supersedes the pre-O&M `findings_provisional.md`**, which should be
retained as `findings_provisional_pre_om.md`. Every figure in that file
predates two transitions: the shape-factor sourcing (`b7f7bdb`) and the
O&M implementation (`d7538ff`). Nothing is carried forward here.

Where a superseded finding's number changed, the old value is given.
Where it did not, that is stated — an unchanged number across a
configuration change is itself evidence.

---

## The central configuration

    conversion_case             conv_central
    hydro_case                  hydro_flat
    gas_deliverability_case     baseline
    capital_recovery_case       cr_central
    dscr_case                   dscr_central        (1.30, sourced)
    concessional_envelope_case  env_dares
    tariff_case                 tariff_central      (95 USD/MWh)
    capital_treatment_case      f1_tranche
    om_case                     om_central          (NREL ATB 2024, Moderate)

    peak_unsuppressed / storage_cc_central / gas_unconstrained /
    constrained / gas_avail_nameplate / gas_retire_vintage_30 /
    gas_capex_central / ccgt_eff_central / us_inflation 0.020

**Central: 27.318 bn USD NPV (27,317,670,155.72), storage 63,056.31 MWh (2054), solar
39,030.07 MW, backcast −3.2% against observed 2024 gas generation.**
~~Central: 27.954 bn USD NPV, storage 65,613.0 MWh (2054), solar 41,031.4 MW.~~
**(SUPERSEDED, corrected 2026-10-09, TASK 42. 27.954 bn / 65,613.0 MWh / 41,031.4 MW is the
central anchor at the frozen state `2a9226b` (2026-09-07), before the `organic_central`
demand-growth arm change (L54), the IDA21 rate and tenor correction (L53) and the L58
re-baseline; the regression anchor moved in the same re-baseline, 21.811 bn / 45,502.9754 MWh
to 21.328 bn / 43,539.97566 MWh, and the regression line below is the superseded value.
Current values: `results/baseline_sliced/summary__02c17f55.json` — **but see the Phase C note above: that file currently holds `om_high`, not `om_central`, until Group 4 is re-run.** The tables below are at
the frozen state.)**

**Regression anchor: 21,328,493,384.73 USD NPV, storage 43,539.97566 MWh (2054), solar
59,010.38 MW** at
`conv_none` + `pre_f1_overnight` + `dscr_regression` +
`tariff_legacy_65m` + **`om_none`**.
~~Regression anchor: 21,811,042,472.821114 / 45,502.9754 MWh. Reproduced to the digit throughout.
Every number below is attributable because this reproduces.~~
**(SUPERSEDED, corrected 2026-10-10, TASK 43. 21,811,042,472.821114 USD / 45,502.9754 MWh
(2054 storage capacity) is the regression anchor at the frozen state `2a9226b`, before the
`organic_central` demand-growth arm change (L54) and the L58 re-baseline. The tables below
were attributable at that state, where the old anchor reproduced to the digit. Current
values: `results/baseline_sliced/summary__19c81f7c.json`.)**

The anchor runs `om_none` deliberately: its function is to reproduce the
pre-O&M objective exactly, proving the O&M wiring introduced no side
effect. **It is not a cost result and is not comparable to any
`om_central` figure.**

Reporting window is **2045**. 2046–2054 is end-effect buffer; no figure
from those years is reported.

---

# Section 1 — The contribution

## F-1 — Cost of capital reaches the objective, but weighs less than before
**STRUCTURAL. RESTATED — both halves must be reported.**

**Source (pre-epoch, TASK 54 Phase 1).** `results/baseline_sliced/summary__{02c17f55 central, c52b926f cr_comm_low, 5bc7cc5a cr_comm_high, 20bf87d2 cr_ida21}.json`; runner `01b_run_baseline_sliced.py`, Phase B item 0/19, 2026-10-03 at `2b22fe9` (documentary); central re-run 2026-10-18 (commit inferred `4be2da6`). The frozen-state span 25.762–29.728 bn has no retained result file. Detail: `docs/task_log_54_rederivation.md`.

> **[RE-DERIVED, TASK 84, 2026-10-23 — MOVED: magnitudes: span, shares, capital mobilised.** Basis: uncapped baseline. Recovery arms (bn): `cr_comm_low` ~~23.2522~~ 21.8392, `cr_central` ~~24.9866~~ 23.3231, `cr_ida21` ~~24.9869~~ 23.3232, `cr_comm_high` ~~26.3822~~ 24.5144; span ~~3.130~~ **2.675 bn**, share of system cost ~~12.53%~~ **11.47%**. Capital mobilised over the report window ~~8.9917~~ **7.7003 bn** (full horizon ~~11.6989~~ 10.1422), i.e. ~~36.0%~~ **33.0%** of system cost; the recovery span is ~~34.8%~~ **34.7%** of capital mobilised and the envelope span (F-2) ~~23.9%~~ **20.9%** (~~8.6%~~ 6.9% of system cost). `cr_ida21` - `cr_central` ~~233,644~~ 175,588 USD. `conv_none` window ~~(blended)~~ 7,560,038,766 / full 9,989,112,675. Both framings still hold; the recovery span exceeds the envelope span. Source: `results/baseline_sliced/summary__59f45b41.json`, `results/baseline_sliced/summary__661b03c1.json`, `results/baseline_sliced/summary__00da6824.json`, `results/baseline_sliced/summary__fec4854b.json`, `results/baseline_sliced/summary__2945336c.json`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; conclusion holds.** Post-epoch = `results/baseline_sliced/` Phase B (2026-10-19/20), capital-charge window `arrears`, storage capex `declining_nrel`; post-epoch hash = the pre-epoch arm tag plus the arms TASK 53 §2.5 completed. Recovery arms (bn): `cr_comm_low` ~~25.1945~~ 23.2522, `cr_central` ~~27.3177~~ 24.9866, `cr_ida21` ~~27.3179~~ 24.9869, `cr_comm_high` ~~29.0054~~ 26.3822; span ~~3.811~~ **3.130 bn**, share of system cost ~~13.95%~~ **12.53%**. Capital mobilised over the report window (`npv_gross_capex_report_window_usd`) ~~10.1528~~ **8.9917 bn** (full horizon ~~13.2147~~ 11.6989), i.e. ~~37.2%~~ **36.0%** of system cost; the recovery span is ~~37.5%~~ **34.8%** of capital mobilised and the envelope span (F-2) ~~22.7%~~ **23.9%** (~~8.4%~~ 8.6% of system cost). `cr_ida21` − `cr_central` ~~252,335~~ 233,644 USD. `conv_none` window ~~12.5297~~ 11.2024 bn, full ~~15.5643~~ 13.7760. The ordering (recovery span > envelope span, on both denominators) holds. The frozen-state figures in the body (25.762–29.728 bn, 3.97 bn, 14.2%, 12.772 bn, 31.1%, 19.3%) and the pre-O&M comparisons are history and were not re-derived. Source: `results/baseline_sliced/summary__{779e4d66,6046f759,36abe3f4,ba51db11}.json`.]**

> **[EPOCH, TASK 43, 2026-10-10. The figures in this entry are at the frozen state `2a9226b` (2026-09-07), before the `organic_central` arm change (L54), the IDA21 rate/tenor correction (L53) and the L58 re-baseline. Current epoch (`results/baseline_sliced/`, Phase B sweep 2026-10-03): `cr_comm_low` 25.1945 bn, `cr_central` 27.3177 bn, `cr_ida21` 27.3179 bn, `cr_comm_high` 29.0054 bn; span 3.811 bn, 13.9% of central; capital mobilised over the report window 10.153 bn (37.2% of system cost); concessional-envelope span (F-2) 2.300 bn, 8.4% of central. `cr_ida21` now differs from `cr_central` by 0.25 M USD, not ~15 M. The shares, ratios and sentences below were not re-derived in this task; read them as frozen-state statements.]**

> **[RE-DERIVED, TASK 44, 2026-10-10 — moved; conclusion holds. Span 25.762–29.728 → 25.1945–29.0054 bn (3.97 → 3.811 bn, `cr_comm_low`/`cr_comm_high`); share of system cost 14.2% → 13.95%; capital mobilised 12.772 → 10.1528 bn (`npv_gross_capex_report_window_usd` 10,152,785,349; full horizon 13,214,701,821), i.e. 45.7% → 37.2% of system cost; the recovery span is 37.5% of capital mobilised (was 31.1%) and the envelope span (F-2) 22.7% (was 19.3%); `cr_ida21` − `cr_central` 252,335 USD (was ~15 M). `conv_none` window 12,529,712,449 / full 15,564,344,940. The ordering (recovery span > envelope span) holds. The pre-O&M comparisons (3.90 bn, 18.6%) and the F0 blend-scenario figure are historical and were not recomputed. Source: `results/baseline_sliced/summary__*.json`, 2026-10-03.]**

Under the per-tranche capital recovery formulation the objective spans
**25.762 to 29.728 bn** across recovery arms:

    cr_comm_low   (9% / 15 yr)    25.762 bn
    cr_central    (11% / 10 yr)   27.954 bn
    cr_ida21      (2.13% conc.)   27.969 bn
    cr_comm_high  (14% / 7 yr)    29.728 bn

**The span is 3.97 bn, essentially unchanged by O&M (3.90 pre-O&M). Its
share of system cost falls from 18.6% to 14.2%**, because the objective
it is measured against grew 30.9%.

Both statements belong in the thesis. The contribution is intact — the
F0 audit found four blend scenarios returning 2.765439e10 to seven
significant figures, i.e. halving the private cost of capital from 18%
to 9% changed system cost by exactly zero under a blended WACC. But its
relative weight against operating and physical cost is smaller than
previously reported, and a referee will compute this if it is not
stated.

`cr_ida21` differs from `cr_central` by ~15M: the IDA20-to-IDA21 shift
in concessional terms is second-order against the commercial tranche.

**Against capital mobilised rather than system cost, the span is
twice as large.** Capital mobilised over the report window at the
central configuration is **12.772 bn**, against a system cost of
27.954 bn — capital is 45.7% of system cost.

| | share of system cost | share of capital mobilised |
|---|---|---|
| capital recovery span (F-1) | 14.2% | **31.1%** |
| concessional envelope span (F-2) | 8.8% | **19.3%** |

Both framings are correct and both should appear. The system-cost share
is the right denominator for a planner asking what the pathway costs;
**the capital share is the right denominator for a finance ministry or
a development bank**, which acts on the capital programme rather than on
total discounted system cost.

The second framing is also the more legible statement of the
contribution. "Rate and tenor move system cost by 14%" invites a shrug;
**"rate and tenor move the cost of the capital programme by 31%"** is
the finding, and it is the one an institution allocating concessional
capital would act on.

npv_gross_capex_report_window_usd   2025-2045, gross, salvage excluded
npv_gross_capex_full_horizon_usd    2025-2054, gross

conv_central   window 12,772,100,150   full 14,350,247,287
conv_none      window 15,019,795,486   full 17,181,528,120

## F-2 — Concessional finance is inframarginal at every scale tested
**STRUCTURAL. Span unchanged by O&M.**

**Source (pre-epoch, TASK 54 Phase 1).** `results/baseline_sliced/summary__{950458b0 env_none, 0524a6e2 env_dares_plus, d8237c02 env_5x, 2e0d019f env_10x, 02c17f55 central}.json`; runner `01b_run_baseline_sliced.py`, 2026-10-03, `2b22fe9` (documentary). Current-epoch figures match the files; frozen-state figures survive only in `_pre_rerun_20260925_1221` / `~/thesis-snapshots/results_pre_margin_fix/`.

> **[RE-DERIVED, TASK 84, 2026-10-23 — WEAKENED: 'never what is built': 2054 totals are identical but the timing of build is not.** Basis: uncapped baseline. Envelope arms (bn): `env_none` ~~25.2563~~ 23.5449, `env_dares` ~~24.9866~~ 23.3231, `env_dares_plus` ~~24.8844~~ 23.2388, `env_5x` ~~24.0495~~ 22.6055, `env_10x` ~~23.1090~~ 21.9326; span ~~2.147~~ **1.612 bn** (6.9% of central); cost still falls monotonically. **2054 storage 56,289.88 MWh, solar 30,938.84 MW and gas 4,507.59 MW are identical at every scale (max |d| 0.0000)**, so the totals clause stands. **But the annual additions are not identical:** against central the solar additions are displaced by 119.7 MW (`env_none`, 0.39% of 30,939), 439.2 (`env_dares_plus`, 1.4%), **4,268.4 (`env_5x`, 13.8%) and 4,550.4 MW (`env_10x`, 14.7%)**, storage additions by 2,821.9 and 2,974.8 MWh (5.0%, 5.3%) at 5x and 10x, first differing year 2033-2035 (`lever_ranking.csv`: 5 plans, `concessional_envelope`). The DARES envelope (750 M USD) is ~~8.3%~~ **9.7%** of capital mobilised (7.7003 bn); the 0-to-10x span is ~~23.9%~~ **20.9%** of it. The sentence 'concessional finance changes who pays, never what is built' would have to say: never what is built **by 2054**; at 5x and above it moves roughly a seventh of the solar build in time. M-1 pairing: 11-60 bn is 1.4 to 7.8 times the 7.70 bn capital programme. Source: `results/baseline_sliced/summary__a3801166.json`, `results/baseline_sliced/summary__9f6bab89.json`, `results/baseline_sliced/summary__661b03c1.json`, `results/lever_ranking/arms_661b03c1/lever_ranking.csv`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; conclusion holds.** Post-epoch = `results/baseline_sliced/` Phase B (2026-10-19/20), capital-charge window `arrears`, storage capex `declining_nrel`; post-epoch hash = the pre-epoch arm tag plus the arms TASK 53 §2.5 completed. Envelope arms (bn): `env_none` ~~27.6089~~ 25.2563, `env_dares` ~~27.3177~~ 24.9866, `env_dares_plus` ~~27.2072~~ 24.8844, `env_5x` ~~26.3065~~ 24.0495, `env_10x` ~~25.3089~~ 23.1090; span ~~2.300~~ **2.147 bn** (8.6% of central); cost still falls monotonically; storage and solar identical at every scale (max |Δ| 0.0000 MWh, 0.0000 MW). The 750 M USD DARES envelope is ~~7.4%~~ **8.3%** of capital mobilised (8.9917 bn); quote the measured figure, not a threshold (the TASK 44 'under 8%' reading has been crossed by the number and is withdrawn). The full envelope span is ~~22.7%~~ **23.9%** of capital mobilised. M-1's 11–60 bn is ~~1.1 to 5.9~~ **1.2 to 6.7** times the capital programme. Source: `results/baseline_sliced/summary__{d85a9332,6046f759,ed835601,542a5d27,6e125b84}.json`.]**

> **[EPOCH, TASK 43, 2026-10-10. Frozen-state figures. Current epoch: `env_none` 27.6089, `env_dares` 27.3177, `env_dares_plus` 27.2072, `env_5x` 26.3065, `env_10x` 25.3089 bn; span 2.300 bn (2.46 bn at the frozen state); storage 63,056.3 MWh and solar 39,030.1 MW identical at every scale (storage was 65,613.0 MWh at the frozen state).]**

> **[RE-DERIVED, TASK 44, 2026-10-10 — moved; conclusion holds. Span 2.46 → 2.300 bn; cost still falls monotonically (27.6089 > 27.3177 > 27.2072 > 26.3065 > 25.3089 bn); storage and solar identical at every scale (max |Δ| 7e-5 MWh, 2e-4 MW). The 750 M USD DARES envelope is 7.4% of capital mobilised (10.1528 bn), not 5.9% — read "under 6%" as under 8%; the full envelope span is 22.7% of capital mobilised (was 19.3%); the M-1 midstream range 11–60 bn is 1.1 to 5.9 times the capital programme (was 1 to 5). The "15 to 40 times the DARES envelope" ratio was not recomputed.]**

    env_none        28.261 bn      storage 65,613.0 MWh
    env_dares       27.954 bn      storage 65,613.0 MWh
    env_dares_plus  27.833 bn      storage 65,613.0 MWh
    env_5x          26.873 bn      storage 65,613.0 MWh
    env_10x         25.802 bn      storage 65,613.0 MWh

Span 2.46 bn, against 2.45 pre-O&M. Cost falls monotonically; **storage
is identical to the decimal at every scale**, as are solar and gas.

The envelope is fully drawn and exhausted at every build decision, so
the marginal dollar is commercial everywhere. An inframarginal lump of
cheap capital changes the total bill, not the choice. **Concessional
finance changes who pays, never what is built.**

**The envelope, measured against the capital it is meant to
mobilise.** The DARES envelope is 750M USD; capital mobilised over the
report window is **12.772 bn**. So the envelope is ~~**5.9% of the
capital programme**~~ **8.3% of the capital programme (TASK 55)**, and the full 0-to-10x envelope span moves system
cost by 2.46 bn — ~~**19.3% of capital mobilised**~~ **23.9% of capital mobilised (2.147 bn, TASK 55)**.

That comparison sharpens the inframarginality result rather than
softening it. A concessional facility covering ~~under 6%~~ 8.3% (TASK 55) of the capital
requirement cannot be expected to set the marginal cost of capital, and
the model confirms it does not: storage and solar are identical to the
decimal at every envelope scale. **Concessional finance at this scale
changes who pays for the programme, not what the programme is.**

Pairs with M-1: implied midstream investment of 11-60 bn is 15 to 40
times the DARES envelope, and now also **roughly 1 to 5 times the
entire power-sector capital programme** this model builds. The scale
mismatch is not between the envelope and the midstream cost alone; it
is between concessional finance as currently sized and Nigerian energy
capital formation as a whole.

npv_gross_capex_report_window_usd   2025-2045, gross, salvage excluded
npv_gross_capex_full_horizon_usd    2025-2054, gross

conv_central   window 12,772,100,150   full 14,350,247,287
conv_none      window 15,019,795,486   full 17,181,528,120

## F-3 — Financing moves cost; physics moves quantity
**STRUCTURAL. STRENGTHENED — this is now a much stronger claim.**

**Source (pre-epoch, TASK 54 Phase 1).** `results/baseline_sliced/summary__*.json` for the 4 capital-recovery, 5 envelope and 2 DSCR arms (`6bf84218`, `fbe4d5b2`) and the regression anchor `19c81f7c`; runner `01b_run_baseline_sliced.py`, 2026-10-03, `2b22fe9` (documentary). The O&M test has no current-epoch run.

> **[RE-DERIVED, TASK 84, 2026-10-23 — WEAKENED: 'identical to the decimal at every arm' and the O&M test 'not at all'; the partition itself stands (brief expected REVERSED: not supported by the files).** Basis: uncapped baseline. **Financing partition, 2054 totals.** Across the 4 capital-recovery, 5 envelope and 2 DSCR arms the cost span is ~~3.273 bn~~ **2.675 bn** (~~13.10%~~ **11.47%** of 23.3231 bn). 2054 storage 56,289.88 MWh and solar 30,938.84 MW are identical (max |d| 0.0000) in 10 of 11 financing arms; **`cr_comm_high` differs: -120.35 MWh (-0.21%), -143.91 MW solar (-0.47%), +30.09 MW gas.** `dscr_low`, `dscr_high` equal central exactly (23,323,068,340.46). **Plans (annual additions):** capital recovery produces 3 distinct plans (`cr_central` = `cr_ida21`; `cr_comm_low` differs from 2035, `cr_comm_high` from 2045), concessional envelope 5, O&M 4 (`lever_ranking.csv`). Displacement from central: `cr_comm_low` 115.8 MW solar (0.37%), `cr_comm_high` 113.9 (0.37%) and 182.9 MWh storage (0.32%), `env_5x`/`env_10x` 4,268/4,550 MW (13.8%/14.7%). **Physical arms** move 2054 quantity by thousands (conversion 1,030-7,672 MWh, availability `observed` +26,900 MWh / +20,986 MW, deliverability drift -13,554 to +10,960 MWh, retirement `none` -26,395 MWh, hydro `ndc` -24,255 MWh, `storage_cc_low` +30,919 MWh). **O&M test:** `om_low` equals central at 2054 (23.0394 bn), **`om_high` moves storage -120.35 MWh and solar -143.91 MW** (23.7399 bn); `om_none` (synthetic, unattainable) moves storage -5,434 MWh. Storage holding cost ~~611~~ **546 M USD/yr** (9,694 x 56,289.88). What the entry would have to say: financing moves cost by 11.5% and 2054 quantity by at most 0.5% (solar) / 0.3% (storage), against physical arms moving quantity by 2 to 100%; the invariance is no longer exact, the three arms that move by 120/144 MW (`cr_comm_high`, `om_high`, `gas_capex_low`) share one alternative vertex, and the timing of build moves under financing (F-2). Not supported: a reversal; the physical/financial separation in magnitude holds. Source: `results/baseline_sliced/summary__661b03c1.json`, `results/baseline_sliced/summary__fec4854b.json`, `results/baseline_sliced/summary__9e0d690e.json`, `results/lever_ranking/arms_661b03c1/lever_ranking.csv`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; conclusion holds; regime question reported under Tier C.** Post-epoch = `results/baseline_sliced/` Phase B (2026-10-19/20), capital-charge window `arrears`, storage capex `declining_nrel`; post-epoch hash = the pre-epoch arm tag plus the arms TASK 53 §2.5 completed. **Financing partition.** Across the 4 capital-recovery, 5 envelope and 2 DSCR arms (11 files plus central) the cost span is ~~3.811~~ **3.273 bn** (~~13.95%~~ **13.10%** of 24.9866 bn) and **2054 storage is 63,056.3 MWh and solar 39,030.1 MW at every arm** (max |Δ| 0.0000 MWh, 0.0000 MW; was 7e-5, 2e-4). `dscr_low` and `dscr_high` equal central exactly. **Physical arms.** Storage (2054) ranges: conversion 62,485–74,225 MWh (unchanged), availability `observed` 83,400 (unchanged), retirement 45,196–63,056 (unchanged), hydro 54,156–62,288 (unchanged), peak shape 54,649–72,747 (unchanged), deliverability 49,682–65,154 (unchanged), **storage credit ~~18,979–50,445~~ 50,445–95,092**. Of 24 physical-arm files, ~~20~~ **19** move storage or solar by more than 1 MWh/MW (the pre-epoch count used a different file set, 21 of 27, and is not comparable). **Restated partition (Zeus's wording, TASK 55, 2026-10-20).** Financing arms change neither the requirement, the credit, nor any firm option's cost, so storage does not move (11 arms, |Δ| 0.0000 MWh); the arms that move it change the requirement (the physical arms), the credit (`storage_cc_low`), or the relative cost of the firm options (the regression anchor, `gas_capex_low`, `gas_retire_vintage_50`, `downside`, under the new storage capex series). Storage absorbs the firm-capacity residual either way; what varies is the split between storage and the other firm sources. **A binding-or-not partition is not what separates the arms:** the firm-capacity constraint (|slack| < 1 MW with a positive dual) binds in the terminal year (2054) in **all 38 arms**, and in **the same number of years before and after the epoch in every arm**. Binding years of 30 (pre-epoch = post-epoch): 20 in central and in every financing, O&M, DSCR, conversion, credit, deliverability, drift, gas-capex, gas-build and `ndc_17gw` arm, **including the five whose storage moved** (regression anchor, `storage_cc_low`, `gas_capex_low`, `gas_retire_vintage_50`, `downside`); 30 in `gas_avail_observed`; 25 in `vintage_40`; 19 in `vintage_proxy` and `hydro_committed`; 18 in `gas_retire_none` and `peak_suppressed`; 15 in `hydro_ndc`. The count varies across arms that move storage and is identical (20) across arms that do and do not move it, so it does not discriminate. Which of the storage capex series or the credit sets the split in each moved case was not isolated. This is a wording measured against the files; I did not find the partition in `docs/chapters/chapter3_3_draft.md` §3.3.6 (the self-generation backstop; §3.3.5 says storage 'absorbs the entire firm-capacity residual'), and the assembled chapter's version of the paragraph (`CHAPTER_3_assembled.md` near line 1510) already carries this wording. **Where plans moved:** the regression anchor (`conv_none`, `pre_f1_overnight`, `om_none`) storage ~~43,540~~ **74,435 MWh (+70.9%)**, solar ~~59,010~~ 59,661 MW; `storage_cc_low` ~~18,979~~ 95,092; `gas_capex_low` ~~18,979~~ 63,056 (now equal to central); `gas_retire_vintage_50` ~~58,258~~ 60,260; `downside` ~~73,254~~ 73,414. Central 2054 storage and solar are unchanged, but the **2045** values moved (storage ~~32,742~~ 37,414 MWh; installed solar capacity (`solar_capacity_mw`, which includes the 500 MW 2025 fleet) ~~19,490~~ 19,741 MW, i.e. cumulative additions (`solar_cumulative_built_2045_mw`) ~~18,990.3~~ 19,240.8 MW): the plan's timing changed, its terminal state did not. **O&M band (now measured at the post-epoch state):** `om_low` 24.5599 bn, `om_central` 24.9866, `om_high` 25.6116, storage 63,056.31 MWh at all three; span ~~0.975~~ **1.052 bn** (4.2% of central), so storage is again unmoved by a cost band levied on storage itself, and the financing span is **3.1 times** that span (was "roughly four times"). Storage holding cost 9,694 × 63,056.3 = 611 M USD/yr (unchanged; the 9,694 rate is not traced to a source, TASK 54 Phase 1). The frozen-state 27.954/65,613 figures, the 636 M/yr and the frozen O&M table (27.531/27.954/28.542 bn, 1.011 bn) in the body are history. Source: `results/baseline_sliced/summary__*.json` (post-epoch hashes of the Phase 1 arms; `19c81f7c` → `6769972a`).]**

> **[EPOCH, TASK 43, 2026-10-10. Frozen-state figures (3.97 bn span, 14.2%, storage 65,613.0 MWh, solar 41,031.4 MW, 27.954 bn). Current epoch: across every capital-recovery, envelope and DSCR arm in the 2026-10-03 sweep the cost span is 3.811 bn (13.9% of 27.3177 bn) and **2054 storage capacity is 63,056.3 MWh and solar 39,030.1 MW at every arm**, so the partition survives. `dscr_low` and `dscr_high` return 27.3177 bn, equal to central. Physical arms, current: conversion 62,485 (`conv_high`) to 74,225 (`conv_none`) MWh; gas availability `observed` 83,400 MWh; gas retirement 45,196 (`none`) to 63,056 MWh; hydro `committed` 62,288 and `ndc` 54,156 MWh; peak shape `suppressed` 54,649 and `high` 72,747 MWh. The O&M band (`om_low`/`om_high`) was not part of the 2026-10-03 sweep and has no current-epoch value; the O&M-band sentence and the $636M/yr holding-cost figure below are frozen-state and were not re-derived.]**

> **[RE-DERIVED, TASK 44, 2026-10-10 — holds; the O&M test has no current-epoch run. Financing arms (4 capital-recovery, 5 envelope, 2 DSCR; 9 distinct files): max |Δ storage| 7e-5 MWh, max |Δ solar| 2e-4 MW against central, so the partition survives; 21 of 27 physical-arm files move quantity by more than 1 MWh or MW, and the 6 that do not (`ccgt_eff_low`, `ccgt_eff_iso`, `gas_ndc_17gw`, `gas_retire_vintage_40`, `gas_retire_proxy`, `gas_capex_high`) move cost or nothing, which the claim (every arm that moves quantity is physical) does not contradict. Cost span of financing 3.811 bn (13.95%). Storage holding cost: 9,694 USD/MWh-yr × 63,056.3 MWh = 611 M/yr (was 636 M; the 9,694 rate was not re-read from source). The O&M band paragraph (`om_low`/`om_high`, 1.011 bn, "about four times the error bar") has no current-epoch value and is frozen-state.]**

Across every capital-recovery arm, every envelope arm and both DSCR
arms — a **3.97 bn cost span, 14.2% of central** — storage is
**65,613.0 MWh** and solar **41,031.4 MW**, identical to the decimal.

This was already the evidence pre-O&M. It is stronger now because
**storage carries roughly $636M/yr of real holding cost** (9,694
USD/MWh-yr on 65,613 MWh) and still does not move anywhere in the
financing space. A priced asset that does not respond to a 14% cost
swing is a much harder result than an unpriced one.

Every arm that *does* move quantity is physical:

    conversion      65,042 – 76,781 MWh
    deliverability  52,239 – 75,863 MWh
    retirement      47,853 – 65,613 MWh
    availability            86,057 MWh
    peak shape      56,968 – 75,578 MWh
    hydro                   64,845 MWh

Clean partition, no exceptions.

Note `dscr_low` and `dscr_high` both return **27.954 bn, byte-identical
to central**. The DSCR cushion does not reach the objective in the
uncapped baseline, consistent with bankability being slack there.

**The sharpest available test of this claim is the O&M band, and it
passes.** Storage fixed O&M scales directly with installed storage
energy capacity — the very quantity F-3 asserts is invariant — so a
cost band levied on that term is the one variation that might
reasonably be expected to move it. Across the three sourced NREL ATB
cost cases:

    om_low      (ATB Advanced)      27.531 bn   storage 65,613.0 MWh
    om_central  (ATB Moderate)      27.954 bn   storage 65,613.0 MWh
    om_high     (ATB Conservative)  28.542 bn   storage 65,613.0 MWh

A 1.011 bn span — 3.6% of central — moves storage **not at all**.
Financing moves cost; physics moves quantity, including when the cost
in question is levied on the physical quantity itself.

This also bounds a question a reader would otherwise raise about F-1.
The capital-recovery span is 3.97 bn; the sourced uncertainty on the
O&M term added at `d7538ff` is 1.011 bn. The financing contribution is
roughly four times the error bar on the cost term that most recently
displaced it in relative weight, so the 14.2% share reported in F-1 is
not an artefact of O&M's own imprecision.

## F-4 — Tenor is half the concessional advantage
**PROVISIONAL. Unchanged — a property of the rates, not the objective.**

**Source (pre-epoch, TASK 54 Phase 1).** closed form, no result file. Two capital-recovery-factor levels; no solve behind it.

> **[RE-DERIVED, TASK 84, 2026-10-23 — UNCHANGED: none: a property of the rates, recomputed from `capital_recovery_scenarios()`.** Basis: closed form. Recomputed (closed form): `cr_central` is `r_conc` 0.0212 over 25 yr and `r_comm` 0.11 over 10 yr (`src/scenarios.py::capital_recovery_scenarios()`), US inflation 0.02: real rates 0.11765% / 8.8235%, CRF(r, n) = r / (1 - (1+r)^-n) = 0.040615 / 0.154612, ratio **3.8068**, Shapley over {rate, tenor} (average of the two orderings) **rate 49.871%, tenor 50.129%**: identical to the TASK 55 values. Not model output; the vintage generation does not touch it. Source: `src/scenarios.py (capital_recovery_scenarios)`, `scripts/task84_entries.py (the arithmetic is reproduced in docs/task_log_84_partE.md)`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — UNCHANGED.** Closed form, no result file, recomputed: real rates from the nominal rates and 2% inflation, CRF(r, n) = r / (1 − (1+r)^−n), Shapley over {rate, tenor} with this finding's own method (the average of the two orderings). Current (2.12%, 25 yr against 11%, 10 yr): real 0.1176% / 8.8235%, CRF 0.040615 / 0.154612, ratio 3.8068, **rate 49.871%, tenor 50.129%** (49.9 / 50.1). Frozen (2.80%, 30 yr): 0.7843% / 8.8235%, CRF 0.037538 / 0.154612, ratio 4.1188, 46.350% / 53.650% (46.3 / 53.7). The 53% figure is the frozen real-terms value or the withdrawn nominal 47.0 / 53.0 pair; the current figure is 50.1%. The epoch does not touch it. `methodology_claims_ledger.md` row 64 was corrected on the same basis; the superseded provisional-findings files, the pre-O&M handover and `task_log_50` keep the old figure as historical records.]**

> **Correction (TASK 45, 2026-10-10).** The nominal-basis Shapley pair 47.0 / 53.0
> reported in the original entry could not be reproduced at either epoch
> and is withdrawn. The real-terms decomposition reproduces
> exactly at the frozen state — 46.3 rate, 53.7 tenor — and gives
> 49.9 / 50.1 at the current epoch, computed with this finding's own
> formula. The claim is unchanged in direction and sharper in
> magnitude: tenor and rate now contribute almost equally, so the
> "half" in this finding's title is close to literal rather than
> approximate. The concessional CRF ratio moves from 4.119× to 3.807×
> (0.04061 against 0.15461).

> **[RE-DERIVED, TASK 44, 2026-10-10 — moved; conclusion strengthened. At the current rates (concessional real 0.1176% over 25 yr, L53; commercial real 8.824% over 10 yr): CRF 0.04061 and 0.15462, ratio 3.807 (frozen 0.03754, 0.15461, 4.119). Real-terms Shapley decomposition rate 49.9% / tenor 50.1% (frozen 46.3 / 53.7, reproduced by this method); one-at-a-time 47.3/52.4 depending on order. The headline 47.0 / 53.0 could not be reproduced by this method at either epoch (its definition was not recovered), so the current-epoch counterpart of that pair is not given. "Tenor is half the concessional advantage" now holds almost exactly.]**

Real-terms CRF at the central rates: concessional 0.03754 (0.784% over
30 yr), commercial 0.15461 (8.824% over 10 yr), ratio 4.119x. Shapley
decomposition, order-invariant: ~~**rate 47.0%, tenor 53.0%**~~ (46.3 / 53.7
in real terms). A one-at-a-time attribution gives 68/32 or 31/69
depending on which variable moves first and is not reportable.

Blended-WACC formulations cannot express tenor at all. Nigerian
evidence: no utility-scale grid-connected solar IPP has reached
financial close; commercial bank tenors run 3–7 years absent credit
enhancement; continental average tenor has fallen from twelve years to
five.

---

# Section 2 — What dominates, and it is not financing

## F-5 — Gas fleet availability is the largest single lever
**Ranking unchanged; magnitude up.**

**Source (pre-epoch, TASK 54 Phase 1).** `results/baseline_sliced/summary__{02c17f55,fe552b51 gas_avail_observed}.json`; runner `01b_run_baseline_sliced.py`, 2026-10-03, `2b22fe9` (documentary).

> **[RE-DERIVED, TASK 84, 2026-10-23 — MOVED: magnitudes; conversion falls from second to fourth in the lever order.** Basis: uncapped baseline. `gas_avail_nameplate` ~~24.9866~~ 23.3231 bn / ~~63,056.3~~ 56,289.9 MWh; `gas_avail_observed` ~~32.0459~~ 31.8127 bn / ~~83,399.9~~ 83,189.7 MWh; bracket ~~+7.059 bn, +28.3%~~ **+8.4896 bn, +36.4%** (rank 1 holds). Levers, column 3 (bn): availability **8.4896**, O&M 6.9935, deliverability drift 3.7957, conversion `none`->`high` ~~5.220~~ **3.0907**, capital recovery ~~3.130~~ 2.6751, gas build 2.1463, GMP family 1.9814, envelope ~~2.147~~ 1.6123, storage capacity credit 1.5648, peak shape 1.1055, gas retirement `none`->`v30` ~~1.949~~ 1.0385. Availability is ~~2.26~~ **3.17** times the capital-recovery span. `new_gas_deliverability` is not ranked (arms differ in dispatch model, TASK 83 A.2). Source: `results/lever_ranking/arms_661b03c1/lever_ranking.csv`, `results/baseline_sliced/summary__8ca8cea2.json`, `results/baseline_sliced/summary__661b03c1.json`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; headline holds, lower ordering changed.** Post-epoch = `results/baseline_sliced/` Phase B (2026-10-19/20), capital-charge window `arrears`, storage capex `declining_nrel`; post-epoch hash = the pre-epoch arm tag plus the arms TASK 53 §2.5 completed. `gas_avail_nameplate` ~~27.3177~~ 24.9866 bn / 63,056.3 MWh (unchanged); `gas_avail_observed` ~~35.8204~~ 32.0459 bn / 83,399.9 MWh (unchanged); bracket ~~+8.503 bn, +31.1%~~ **+7.059 bn, +28.3%**. Levers, post-epoch (bn; pre-epoch in brackets): availability bracket **7.059** (~~8.503~~); conversion `none` → `high` **5.220** (~~5.845~~); capital recovery span **3.130** (~~3.811~~); deliverability drift `upside` → `downside` **2.206** (~~2.962~~); envelope 0 → 10x **2.147** (~~2.300~~); gas retirement `none` → `v30` **1.949** (~~2.740~~); GMP `gmp_full` → `flat` **1.584** (~~2.384~~); demand shape `suppressed` → `high` **1.522** (~~2.150~~). **The top three are unchanged; below them the envelope now ranks above retirement and GMP (was below both), so "the ordering is unchanged" no longer holds for the lower rows.** Outside the list: storage credit `low` → `high` ~~1.982~~ **2.239** (now above retirement and GMP; the arm moved storage, F-19), gas build `none` ~~0.807~~ 0.483, gas capex span ~~0.938~~ 0.312; retirement `none` → `proxy` ~~3.871~~ 2.988 (now below capital recovery, 3.130; pre-epoch it was above, 3.871 against 3.811, so the definitional caveat no longer changes the ranking). Hydro `ndc` ~~14.715~~ 13.651 bn, a bound (F-18). Availability is **2.26 times** the capital-recovery span (~~2.23~~). Source: `results/baseline_sliced/summary__*.json`.]**

> **[EPOCH, TASK 43, 2026-10-10. Frozen-state figures. Current epoch: `gas_avail_nameplate` 27.3177 bn / 63,056.3 MWh; `gas_avail_observed` 35.8204 bn / 83,399.9 MWh; bracket +8.503 bn, +31.1%. The lever ranking table below mixes frozen-state values (capital recovery span now 3.811 bn; conversion `none`→`high` now 5.845 bn; gas retirement `none`→`v30` now 2.740 bn; envelope 0→10x now 2.300 bn) and was not re-derived; the deliverability and demand-shape rows were not recomputed.]**

> **[RE-DERIVED, TASK 44, 2026-10-10 — moved; ordering holds. Levers, current (bn): gas availability bracket 8.503 (nameplate 27.3177 → observed 35.8204; +31.1%); conversion `none` → `high` 5.845; capital recovery span 3.811; deliverability drift family (`upside` → `downside`) 2.962; gas retirement `none` → `v30` 2.740; GMP family (`gmp_full` → `flat`) 2.384; envelope 0 → 10x 2.300; demand shape `suppressed` → `high` 2.150. Frozen: 8.60, 5.91, 3.97, 3.01, 2.76, 2.47, 2.46, 2.21. **The ordering is unchanged**; the membership of the two deliverability families is inferred (not stated in the entry). Availability is 2.23 times the capital-recovery span (was 2.17) and the declared lower bound costs 31.1% more ("a third"). Not in the list and not changing it: storage credit `low` → `high` 1.982, gas build `none` 0.807, gas capex span 0.938; the wider retirement span `none` → `proxy` is 3.871 bn and would rank above capital recovery if that definition were used. Hydro `ndc` (14.715 bn) exceeds all of these and is handled as a bound in F-18, as at the frozen state.]**

    gas_avail_nameplate (0.80)   27.954 bn   storage 65,613.0 MWh
    gas_avail_observed  (0.295)  36.557 bn   storage 86,056.8 MWh

**+8.60 bn, +30.8%** (7.32 bn pre-O&M).

Ranked against every other lever swept:

    gas availability bracket         8.60 bn
    conversion (none -> high)        5.91 bn
    capital recovery span            3.97 bn
    deliverability, drift family     3.01 bn
    gas retirement (none -> v30)     2.76 bn
    GMP deliverability family        2.47 bn
    concessional envelope (0 -> 10x) 2.46 bn
    demand shape (suppressed->high)  2.21 bn

There is no sourced central arm. `gas_avail_central` raises by design:
it must represent MECHANICAL availability only — availability given fuel
is allocated — and that cannot be separated from fuel unavailability
without data nobody publishes (register P2-9). So the central case sits
on the declared UPPER bound, and the declared lower bound costs a third
more.

**State this prominently, not buried.** A reader will otherwise notice
that the physical availability of the existing gas fleet matters roughly
twice as much as the entire capital-structure question. Stated first, it
is an argument for the thesis rather than against it: if a third of
system cost hinges on whether Nigeria's fleet runs at 29.5% or 80%, and
the gap is an undecomposed mechanical-versus-fuel question, then
treating deliverability and financing as the binding constraints —
rather than capacity — is the right frame.

## F-6 — Conversion nearly doubled and is now the second-largest lever
**PROVISIONAL. The largest change in this restatement.**

**Source (pre-epoch, TASK 54 Phase 1).** `results/baseline_sliced/summary__{062eba52 conv_none, abbb2e9e conv_low, 02c17f55, 138c45af conv_high}.json`; runner `01b_run_baseline_sliced.py`, 2026-10-03, `2b22fe9` (documentary).

> **[RE-DERIVED, TASK 84, 2026-10-23 — WEAKENED: 'second-largest lever' (now fourth); 'pays twice' (capital mobilised no longer falls); 'scales the programme down' (conversion now raises solar and storage).** Basis: uncapped baseline. Conversion arms (cost bn / 2054 storage MWh / solar MW): `conv_none` ~~29.1209 / 74,224.8 / 44,178.1~~ 25.3036 / 48,617.5 / 29,716.3; `conv_low` ~~27.0727~~ 24.5536 / 50,407.0 / 29,959.5; `conv_central` ~~24.9866 / 63,056.3 / 39,030.1~~ 23.3231 / 56,289.9 / 30,938.8; `conv_high` ~~23.9008~~ 22.2129 / 55,260.0 / 29,707.4. Worth `none`->`central` ~~4.134~~ **1.9805 bn (7.8%)**, `none`->`high` ~~5.220~~ **3.0907 bn**: **rank 4**, not 2 (behind availability 8.4896, O&M 6.9935, drift 3.7957). **Capital mobilised over the report window: `conv_none` 7.5600 bn, `conv_central` 7.7003 bn: conversion RAISES it by 0.1403 bn (+1.9%)** (~~12.530 -> 10.153, -19.0%~~); full horizon 9.9891 -> 10.1422. Against `conv_none`, conversion now adds +1,223 MW solar and +7,672 MWh storage (~~-5,148 MW, -11,168 MWh~~). 2025 gas-side output unchanged in total: 24.788 TWh_e (converted 21.762, fleet 3.026). `conv_low` converts tier A only (571 of 571 MW; tier B 0 of 3,215). The title would have to say: conversion is worth 1.98 bn (7.8%) and ranks fourth; the 'pays twice' clause (cost and capital programme both fall) does not hold: cost falls, capital rises; the capital-formation argument of the last paragraph loses its premise. Source: `results/baseline_sliced/summary__2945336c.json`, `results/baseline_sliced/summary__10836924.json`, `results/baseline_sliced/summary__661b03c1.json`, `results/baseline_sliced/summary__fa6a652b.json`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; ranking holds.** Post-epoch = `results/baseline_sliced/` Phase B (2026-10-19/20), capital-charge window `arrears`, storage capex `declining_nrel`; post-epoch hash = the pre-epoch arm tag plus the arms TASK 53 §2.5 completed. Conversion arms (cost bn / 2054 storage MWh / solar MW): `conv_none` ~~31.9858~~ 29.1209 / 74,224.8 / 44,178.1; `conv_low` ~~29.5776~~ 27.0727 / 63,978.2 / 40,132.4; `conv_central` ~~27.3177~~ 24.9866 / 63,056.3 / 39,030.1; `conv_high` ~~26.1412~~ 23.9008 / 62,484.9 / 38,346.8 (storage and solar unchanged to 1e-3). Conversion is second-largest still: `none` → `high` ~~5.845~~ **5.220 bn**; `none` → `low` ~~2.408~~ 2.048. `none` → `central`: system cost ~~−4.668 bn (−14.6%)~~ **−4.134 bn (−14.2%)**; capital mobilised 2025–2045 ~~12.530 → 10.153 (−2.377, −19.0%)~~ **11.202 → 8.992 (−2.211, −19.7%)**; solar −5,148 MW and storage −11,168 MWh (unchanged). Capital share of system cost ~~39.2%~~ **38.5%** (`none`) against ~~37.2%~~ **36.0%** (`central`). 2025 gas-side output is unchanged (converted 0 / 20.31 / 21.76 / 21.76 TWh_e, fleet 24.79 / 4.48 / 3.03 / 3.03). The "cost and capital fall in almost identical proportion" claim, already weakened at TASK 44, is weakened further: 14.2% of system cost against 19.7% of capital mobilised. Implied overnight conversion capex is unchanged (2.272 bn central, 3.218 low, 1.704 high). Source: `results/baseline_sliced/summary__{0b2b2c25,a6ed56c2,6046f759,30fbcb9a}.json`.]**

> **[EPOCH, TASK 43, 2026-10-10. Frozen-state figures. Current epoch (storage = 2054 capacity): `conv_none` 31.9858 bn / 74,224.8 MWh / 44,178.1 MW; `conv_low` 29.5776 / 63,978.2 / 40,132.4; `conv_central` 27.3177 / 63,056.3 / 39,030.1; `conv_high` 26.1412 / 62,484.9 / 38,346.8. Conversion `none`→`central`: system cost −4.668 bn (−14.6%); capital mobilised 2025–2045 12.530 → 10.153 bn (−2.377 bn, −19.0%); solar −5,148 MW; storage −11,168 MWh. The sentences on lever ranking, the 1.2 bn conversion capex, the 3.4 bn avoided build and the 45.7%/46.0% capital shares are frozen-state and were not re-derived (the current capital share of system cost is 37.2%).]**

> **[RE-DERIVED, TASK 44, 2026-10-10 — moved; ranking holds; one comparative claim weakened. Conversion is second-largest still (5.845 bn, after availability 8.503); worth 2.408 (`none` → `low`) to 5.845 (`none` → `high`), was 2.45 to 5.91. 2025 gas-side output: converted 0 → 20.31 → 21.76 → 21.76 TWh_e (`none`, `low`, `central`, `high`) and fleet 24.79 → 4.48 → 3.03 → 3.03, total 24.79 constant (frozen: 0 → 10.64 → 21.76, fleet 24.74 → 14.11 → 2.98): `conv_low` now converts tiers A and B in full (571 and 3,215 MW; tier C 0), so low and central differ in capex, not in what is converted. Capital share of system cost: 39.2% (`none`) against 37.2% (`central`) (was 46.0% / 45.7%). **Weakened:** the cost and capital reductions are no longer "almost identical in proportion" — 14.6% of system cost against 19.0% of capital mobilised — so "scales the whole programme down rather than shifting its composition" is supported less strongly. Not re-derived: the 1.2 bn conversion capex inside the window and the 3.4 bn avoided build (implied overnight conversion capex is now 2.272 bn at `conv_central`, 3.218 low, 1.704 high; the in-window split was not recomputed).]**

> **Correction (TASK 45, 2026-10-10).** The original entry stated that conversion
> reduces cost and capital mobilised in almost identical proportion,
> and read that as the programme scaling down rather than changing
> composition. At the current epoch the proportions differ: cost falls
> 14.6% while capital mobilised falls 19.0%, and capital mobilised as a share of
> system cost falls from 39.2% under `conv_none` to 37.2% under
> `conv_central` (frozen state: 46.0 and 45.7). Conversion reduces the
> capital programme by more than it reduces system cost, so the effect
> is a change in composition as well as in scale. The ranking claim —
> conversion is the second-largest lever at 5.845 bn — is unaffected.

    conv_none     32.678 bn   storage 76,781.4 MWh   solar 46,179.4 MW
    conv_low      30.229 bn   storage 66,534.8 MWh   solar 42,133.7 MW
    conv_central  27.954 bn   storage 65,613.0 MWh   solar 41,031.4 MW
    conv_high     26.772 bn   storage 65,041.6 MWh   solar 40,348.1 MW

Conversion is worth **2.45 to 5.91 bn**, against 1.27–3.14 pre-O&M — a
promotion from fourth lever to second.

**Mechanism, and this is a finding O&M created rather than rescaled.**
Converted plant carries VOM of 2.12 USD/MWh against the unconverted
fleet's 6.94, and produces 21.76 of 24.79 TWh_e. Once operating cost is
priced, conversion is not only a fuel-efficiency play — it is an
operating-cost play. Its fixed O&M is *higher* (33.5 vs 25.7 USD/kW-yr)
and it wins anyway.

The underlying fuel mechanism is unchanged: converted output rises
0 -> 10.64 -> 21.76 TWh_e while fleet output falls 24.74 -> 14.11 ->
2.98, total gas-side output constant at 24.74 TWh_e. The same gas yields
more electricity — the only route to more generation under a binding
deliverability constraint.

**This strengthens the petroleum-engineering contribution
specifically.**

**Conversion pays twice: it reduces system cost AND the capital
programme.** Holding every other arm at central and varying only
`conversion_case`:

| | `conv_none` | `conv_central` | delta |
|---|---|---|---|
| system cost NPV | 32.678 bn | 27.954 bn | **−4.724 bn (−14.5%)** |
| capital mobilised, 2025-2045 | 15.020 bn | 12.772 bn | **−2.248 bn (−15.0%)** |
| solar built | 46,179 MW | 41,031 MW | −5,148 MW |
| storage, 2054 | 76,781 MWh | 65,613 MWh | −11,168 MWh |

The retrofit is not free: conversion spends roughly 1.2 bn of its own
capex inside the report window. It nonetheless **reduces** total
capital by 2.25 bn, which means it avoids on the order of **3.4 bn of
new-build solar and storage capex** to do so. Fuel-free firm capacity
obtained by retrofitting existing plant displaces capacity that would
otherwise have to be built.

Two features of the comparison are worth stating. The two reductions
are almost identical in proportion — 14.5% of system cost, 15.0% of
capital — and capital's share of system cost barely moves (45.7%
against 46.0%). **Conversion scales the whole programme down rather
than shifting its composition.**

This is the conversion case in its strongest form, and it is a
capital-formation argument rather than only an efficiency one: in a
system where the binding constraint is the availability of capital to
build (F-1, F-2, F-21), a retrofit that lowers the capital requirement
by 15% relieves the binding constraint directly. The 5.91 bn figure
reported above for the full `none`-to-`high` span understates the case
by measuring only one of the two channels.

npv_gross_capex_report_window_usd   2025-2045, gross, salvage excluded
npv_gross_capex_full_horizon_usd    2025-2054, gross

conv_central   window 12,772,100,150   full 14,350,247,287
conv_none      window 15,019,795,486   full 17,181,528,120

## F-7 — The availability screen decides the conversion merit order
**PROVISIONAL. Screen unchanged; its consequence larger.**

**Source (pre-epoch, TASK 54 Phase 1).** model quantities from `results/baseline_sliced/summary__{02c17f55,abbb2e9e,138c45af}.json` (`01b_run_baseline_sliced.py`, 2026-10-03, `2b22fe9`). **Provenance NOT established** for the per-plant NERC PAF inputs behind the tier boundaries.

> **[RE-DERIVED, TASK 84, 2026-10-23 — MOVED: `conv_low` no longer converts tier B; the tier table and the central/high result are unchanged.** Basis: uncapped baseline. Tier table (NERC input) unchanged: tier A 571 MW PAF 0.959, tier B 3,215 MW PAF 0.345, tier C 2,696 MW PAF 0.106. `conv_central` and `conv_high`: tier A 571 of 571 MW, **tier B 3,215 of 3,215 MW**, tier C 0 of 2,696 (unchanged). **`conv_low`: tier A 571 of 571, tier B ~~3,215 of 3,215~~ 0 of 3,215**, tier C 0 (the blended `conv_low` converted tiers A and B). Converted generation 2025: 21.762 TWh_e at central. The merit order tier A > tier B > tier C holds at central; tier B is now cost-marginal against the vintage efficiency of new CCGT (conversion `none`->`central` is worth 1.98 bn, not 4.13). Provenance of the per-plant PAF inputs still not traced. Source: `results/baseline_sliced/summary__661b03c1.json`, `results/baseline_sliced/summary__10836924.json`, `results/baseline_sliced/summary__fa6a652b.json`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — UNCHANGED.** Post-epoch `conv_central`, `conv_low`, `conv_high`: tier A converts 571 of 571 MW, tier B 3,215 of 3,215 MW, tier C 0 of 2,696 MW; tier PAF 0.959 / 0.345 / 0.106 (identical to the pre-epoch files). The tier table is NERC input data and is not epoch-dependent; its per-plant source is still not traced (TASK 54 Phase 1). Source: `results/baseline_sliced/summary__{6046f759,a6ed56c2,30fbcb9a}.json` (`conversion`).]**

> **[RE-DERIVED, TASK 44, 2026-10-10 — holds. `conv_central`, `conv_low` and `conv_high` (2026-10-03): tier A converts 571 of 571 MW, tier B 3,215 of 3,215 MW, tier C 0 of 2,696 MW; tier PAF 0.959 / 0.345 / 0.106. The tier table is NERC input data and does not depend on the epoch.]**

Per-plant availability from NERC across three periods bands 16
convertible open-cycle plants into three tiers:

    tier A  PAF > 50%     571 MW nameplate,   547 MW available,  626 USD/kW effective
    tier B  PAF 25-50%  3,215 MW nameplate, 1,109 MW available, 1,740 USD/kW effective
    tier C  PAF < 25%   2,696 MW nameplate,   285 MW available, 5,677 USD/kW effective

Tier A converts 100% in every arm — inframarginal. Tier B converts fully
at central capex. Tier C converts 0% unconstrained.

Tier B converting fully despite an effective cost ABOVE new-build CCGT
(1,317 USD/kW) is the thesis mechanism in a technology choice:
fuel-free firm capacity beats fuel-hungry new capacity when fuel is the
binding constraint. Under O&M the case is stronger still, because
converted plant also carries a lower variable operating cost.

Without the screen the convertible base reads 6,482 MW of nameplate
rather than 1,941 MW available, and Tier C would be credited with
capacity that does not run.

---

# Section 3 — Under NDC caps

## F-8 — NDC compliance costs 36.5–42.4% less with conversion available
~~F-8 — NDC compliance costs 34-41% less with conversion available~~ **(SUPERSEDED title, corrected 2026-10-10, TASK 44: frozen-state range; current epoch 36.5–42.4%.)**
**CONFIRMED at the frozen state.** Public-only, `02_run_ndc_caps`:

**Source (pre-epoch, TASK 54 Phase 1).** `results/ndc/*/summary.json` (`conv_central` and `conv_none` arms); runner `02_run_ndc_caps.py`, Phase B 2026-10-03 at `2b22fe9` (documentary). The central directories were re-written by Phase C 2026-10-10 (`3f0b382`, documentary), so file dates are 10-10, not 10-03.

> **[RE-DERIVED, TASK 84, 2026-10-23 — WEAKENED: the 37.9-43.3% saving collapses to 1.2-2.7% under the NDC caps; emissions fall 0.1-0.4%, not 26-29%.** Basis: envelope-carrying (all four caps) and cap alone (NDC 3.0). Basis: **cap plus capital envelope** (the entry's design is `02_run_ndc_caps.py`, which carries tight/moderate), and the **cap alone** for NDC 3.0. `conv_central` against `conv_none`, bn: envelope-carrying ndc2_unconditional ~~45.712 / 80.604~~ 28.3457 / 29.1184, saving ~~34.892 (43.3%)~~ **0.7727 (2.65%)**; ndc2_conditional ~~52.755 / 84.918~~ 42.7349 / 43.2718, ~~32.163 (37.9%)~~ **0.5369 (1.24%)**; ndc3_unconditional ~~53.102 / 88.768~~ 35.5615 / 36.2594, ~~35.666 (40.2%)~~ **0.6979 (1.92%)**; ndc3_conditional ~~45.883 / 78.530~~ 36.1405 / 36.7040, ~~32.647 (41.6%)~~ **0.5635 (1.54%)**. **Cap alone** (`capital_case=unconstrained`): ndc3_unconditional 26.6723 / 27.6052, **0.9329 bn (3.38%)**; ndc3_conditional 34.9955 / 35.5980, **0.6024 bn (1.69%)**. Emissions (Mt, central / none): 424.3 / 424.3, 346.5 / 347.4, 464.0 / 464.5, 338.6 / 339.8 (envelope); cap alone 365.5 / 367.0 and 335.3 / 336.5: conversion lowers emissions by **0.0-0.4%**, ~~26.4-28.9%~~. The direction (conversion is cheaper under every cap) holds; the headline magnitude does not. The mechanism sentence (converted plant emits 0.3963 vs 0.6352 tCO2/MWh_e) is an input property and is unchanged; its consequence for the cap is now small because the new-build CCGT already runs at 0.5173 efficiency under vintage dispatch. Source: `results/ndc/ndc3_conditional_3fbc75ac/summary.json`, `results/ndc/ndc3_conditional_accb0eda/summary.json`, `results/ndc/ndc3_unconditional_ae31b29c/summary.json`, `results/ndc/ndc3_unconditional_c5c60367/summary.json`, `results/ndc/ndc2_unconditional_0063167f/summary.json`, `results/ndc/ndc2_unconditional_b2127955/summary.json`, `results/ndc/ndc2_conditional_7f38ab83/summary.json`, `results/ndc/ndc2_conditional_59cb6bcb/summary.json`, `results/ndc/ndc3_unconditional_60585e1f/summary.json`, `results/ndc/ndc3_unconditional_ab6861c9/summary.json`, `results/ndc/ndc3_conditional_4ca4832d/summary.json`, `results/ndc/ndc3_conditional_5e95e8a0/summary.json`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; conclusion holds.** Post-epoch `results/ndc/` (Phase B 2026-10-20, arm tags carry `capital_window_case=arrears`, `storage_capex_case=declining_nrel`; the NDC runner still writes unlabelled keys, L111), `conv_central` against `conv_none`, bn: ndc2_unconditional ~~46.967~~ 45.712 / ~~81.494~~ 80.604, saving ~~34.527 (42.4%)~~ **34.892 (43.3%)**; ndc2_conditional ~~56.146~~ 52.755 / ~~88.354~~ 84.918, ~~32.208 (36.5%)~~ **32.163 (37.9%)**; ndc3_unconditional ~~54.301~~ 53.102 / ~~89.658~~ 88.768, ~~35.357 (39.4%)~~ **35.666 (40.2%)**; ndc3_conditional ~~49.246~~ 45.883 / ~~81.914~~ 78.530, ~~32.668 (39.9%)~~ **32.647 (41.6%)**. Range ~~36.5–42.4%~~ **37.9–43.3%** of the `conv_none` cost, ~~32.2–35.4~~ **32.2–35.7 bn**. Emissions fall ~~26.2–28.5%~~ **26.4–28.9%** (ndc2_unconditional ~~585.7~~ 589.8 / 815.7 Mt, −27.7%; ndc2_conditional ~~441.4~~ 416.7 / ~~597.8~~ 566.3; ndc3_unconditional ~~608.8~~ 612.4 / 832.8; ndc3_conditional ~~424.8~~ 397.8 / ~~594.1~~ 559.6). Note the emissions of the conditional `conv_none` arms moved (597.8 → 566.3, 594.1 → 559.6 Mt) because the cap arms' plan changed with the cost structure. The 0.3963 / 0.6352 tCO2/MWh_e mechanism is input arithmetic and was not re-derived. Source: `results/ndc/ndc*_{conditional,unconditional}_{92f26628,2093b998,085f7315,a4ea9f7b,3399bf36,2def3495,2e122592,d7bd0bb8}/summary.json`.]**

> **[EPOCH, TASK 43, 2026-10-10. The table and the "34-41%" title are the frozen-state values. Current epoch (`results/ndc/`, 2026-10-03, `conv_central` against `conv_none`, bn): ndc2_unconditional 46.967 / 81.494, saving 34.527 (42.4%); ndc2_conditional 56.146 / 88.354, 32.208 (36.5%); ndc3_unconditional 54.301 / 89.658, 35.357 (39.4%); ndc3_conditional 49.246 / 81.914, 32.668 (39.9%). Range 36.5–42.4% of the `conv_none` cost, 32.2–35.4 bn. The conclusion (conversion roughly 35–40% cheaper under every cap) holds.]**

> **[RE-DERIVED, TASK 44, 2026-10-10 — moved; conclusion holds. Cost saving 36.5–42.4% (32.2–35.4 bn) as in the TASK 43 note. Emissions fall by 28.2% (ndc2_unconditional 815.7 → 585.7 Mt), 26.2% (ndc2_conditional 597.8 → 441.4), 26.9% (ndc3_unconditional 832.8 → 608.8) and 28.5% (ndc3_conditional 594.1 → 424.8), i.e. 26.2–28.5% (frozen 26.2–26.9%; the 923.5 → 681.1 example is now 832.8 → 608.8 Mt). The comparison with the pre-O&M range (33.37–35.84 bn) no longer shows the top widening: the current top is 35.36 bn.]**

| case | conv_central | conv_none | saving |
|---|---|---|---|
| ndc2_unconditional | 53.821 | 90.604 | 36.78 bn (−40.6%) |
| ndc2_conditional | 63.374 | 96.524 | 33.15 bn (−34.3%) |
| ndc3_unconditional | 58.698 | 95.796 | 37.10 bn (−38.7%) |
| ndc3_conditional | 51.573 | 85.265 | 33.69 bn (−39.5%) |

Pre-O&M the range was 33.37–35.84 bn; it is now 33.15–37.10 bn. The
conversion value under caps holds and widens slightly at the top.

Emissions fall 26.2–26.9% in parallel (e.g. ndc3_unconditional
923.5 -> 681.1 MtCO2). Mechanism, computed before the run and confirmed
by it: converted plant emits 0.3963 tCO2/MWh_e against the unconverted
fleet's 0.6352 (EF_th 0.1823 / eta) — 37.6% less per unit of
electricity, or 1.60x the electricity per tonne of CO2.

## F-9 — Gas deliverability is worth EXACTLY ZERO under a cap
**CONFIRMED at the frozen state, across all four cap cases.**

**Source (pre-epoch, TASK 54 Phase 1).** `results/ndc/*gas_deliv*/summary.json` (`02_run_ndc_caps.py`) and `results/gas1/` (`06_run_gas1_shadow_benchmarks.py`), 2026-10-03/04 (documentary `2b22fe9` / `4ec43b4`); central ndc directories re-written by Phase C 2026-10-10.

> **[RE-DERIVED, TASK 84, 2026-10-23 — MOVED: uncapped drift span 2.206 -> 3.7957 bn; the pairing with F-8 ('opposite conclusions') loses its second term; scope: `downside` is not zero (F-30).** Basis: cap alone and envelope-carrying, both reported. `baseline`, `gas_deliv_flat` and `gas_deliv_gmp_partial` (and `gas_deliv_gmp_full`) return **identical objectives to the cent** under NDC 3.0: envelope-carrying conditional 36,140,487,294.76 and unconditional 35,561,545,184.67 USD; **cap alone** conditional 34,995,532,338.04 and unconditional 26,672,293,112.76 USD (~~45.712 / 52.755 / 53.102 / 45.883 bn blended~~). NDC 2.0 (envelope-carrying, TASK 81 Batch 2) likewise identical across the three arms. Uncapped, the GMP family spans 1.9814 bn and the drift family (`upside`->`downside`) ~~2.206~~ **3.7957 bn**. The 'pairs with F-8' paragraph would have to drop 'opposite conclusions': conversion is worth only 1.2-3.4% under a cap, so efficiency is no longer shown to be the abatement lever by contrast. Outside this entry's arm set, drift `downside` is not zero under a cap (see F-30). Source: `docs/task_log_82_capped/ndc3_conditional_capped_arms.csv`, `docs/task_log_82_capped/ndc3_unconditional_capped_arms.csv`, `docs/task_log_83_capped/ndc3_conditional_noenvelope_capped_arms.csv`, `docs/task_log_83_capped/ndc3_unconditional_noenvelope_capped_arms.csv`, `docs/task_log_81_batch2.md`, `results/lever_ranking/arms_661b03c1/lever_ranking.csv`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; conclusion holds.** Post-epoch `results/ndc/`: `baseline`, `gas_deliv_flat` and `gas_deliv_gmp_partial` return **identical objectives and emissions (spread 0.0 USD, 0.0 tCO2)** in all four cap cases: ~~46.966774~~ 45.712 / ~~56.145765~~ 52.755 / ~~54.301158~~ 53.102 / ~~49.245616~~ 45.883 bn (×3 each; ndc2_unconditional, ndc2_conditional, ndc3_unconditional, ndc3_conditional). Uncapped, the drift family (`upside` → `downside`) spans ~~2.962~~ **2.206 bn** (the body's 3.01 is frozen-state). Source: `results/ndc/ndc*_{92f26628,95884918,d237145b,...}/summary.json` (12 post-epoch directories).]**

> **[RE-DERIVED, TASK 44, 2026-10-10 — holds. Current epoch (`results/ndc/`, 2026-10-03): `baseline`, `gas_deliv_flat` and `gas_deliv_gmp_partial` return identical objectives and identical emissions in all four cap cases: 46.966774 / 56.145765 / 54.301158 / 49.245616 bn (x3 each).]**

`baseline`, `gas_deliv_flat` and `gas_deliv_gmp_partial` return
**identical objectives to the digit** in every capped case:

    ndc2_unconditional   53.821 bn  x3
    ndc2_conditional     63.374 bn  x3
    ndc3_unconditional   58.698 bn  x3
    ndc3_conditional     51.573 bn  x3

Emissions identical to the digit likewise. Under a binding cap, gas
generation is limited by the cap rather than by fuel, so additional fuel
has no value. **Unchanged by pricing operating cost** — a result that
now holds across two epochs.

Uncapped, the drift family spans 3.01 bn.

**Pairs with F-8: the abatement lever is EFFICIENCY, not fuel
availability.** Same fuel, same plants, opposite conclusions.

CONSEQUENCE: the GMP deliverability arms inform baseline and gas-shadow
work and are inert for every capped pathway. The midstream investment
case (M-1) is therefore meaningful only for uncapped pathways.

## F-10 — The EaaS channel is worth 9.7 to 22.4 bn under NDC caps
**RESTATED, with a comparability caveat that must travel with it.**

**Source (pre-epoch, TASK 54 Phase 1).** `results/ndc/` and `results/ndc_eaas/` `conv_central` directories; runners `02_run_ndc_caps.py`, `03_run_ndc_eaas.py`. Superseded pre-B\* table: `results/_pre_bstar_20260920_1132/`. Current: Phase B 2026-10-03 (`2b22fe9`) re-written by Phase C 2026-10-10 (`3f0b382`), documentary.

> **[RE-DERIVED, TASK 84, 2026-10-23 — MOVED: magnitudes: the range, the unconditional/conditional ratio in the POL-1 design; EaaS value equals the cost of the envelope at NDC 3.0.** Basis: cap plus envelope (by design); cap-alone cross-check at NDC 3.0. Basis: **cap plus capital envelope** (the entry's design carries tight/moderate in both legs, as stored). Capped solar design (`results/ndc/`, `results/ndc_eaas/`, conv_central), public / EaaS / saving (bn): ndc2_unconditional ~~45.712 / 30.403 / 15.309~~ 28.3457 / 24.9243 / **3.421** (12.1%); ndc2_conditional ~~52.755 / 46.551 / 6.204~~ 42.7349 / 41.1066 / **1.628** (3.8%); ndc3_unconditional ~~53.102 / 33.960 / 19.142~~ 35.5615 / 26.6723 / **8.889** (25.0%); ndc3_conditional ~~45.883 / 41.192 / 4.691~~ 36.1405 / 34.9955 / **1.145** (3.2%); range ~~4.7-19.1~~ **1.1-8.9 bn**. Uncapped solar design (POL-1, `results/pol1/arms_1d68be49/pol1_results.csv`): savings ~~19.095, 21.031, 25.804, 15.452~~ **3.421, 10.591, 9.688, 4.786** (range ~~15.5-25.8~~ **3.4-10.6**). The unconditional cases benefit about twice as much as the conditional ones in the capped design at NDC 2.0 (3.42 / 1.63 = 2.1) and far more at NDC 3.0 (7.8); in the POL-1 design that holds at NDC 3.0 (9.69 / 4.79 = 2.0) and **reverses at NDC 2.0** (3.42 against 10.59). **Cap alone:** the EaaS cost at NDC 3.0 equals the cap-alone public cost to the cent (conditional 34,995,532,3xx; unconditional 26,672,293,1xx; `ndc3_*_noenvelope` public arm), so the whole EaaS saving at NDC 3.0 is the cost of the public envelope, ~~a joint effect~~ an envelope effect: EaaS buys nothing against the cap itself. (NDC 2.0 has no cap-alone run.) Source: `results/ndc_eaas/ndc3_unconditional_eaas_2a1bad3e/summary.json`, `results/ndc_eaas/ndc3_conditional_eaas_7a4fdb1c/summary.json`, `results/ndc/ndc3_unconditional_ae31b29c/summary.json`, `results/ndc/ndc3_unconditional_60585e1f/summary.json`, `results/pol1/arms_1d68be49/pol1_results.csv`, `docs/task_log_81_batch2.md`, `docs/task_log_83_capped/ndc3_conditional_noenvelope_capped_arms.csv`, `docs/task_log_83_capped/ndc3_unconditional_noenvelope_capped_arms.csv`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; conclusion holds; caveat unchanged.** Post-epoch, capped solar design (`results/ndc/`, `results/ndc_eaas/`), public / EaaS / saving (bn): ndc2_unconditional ~~46.967 / 32.815 / 14.151~~ 45.712 / 30.403 / **15.309** (33.5%); ndc2_conditional ~~56.146 / 49.297 / 6.848~~ 52.755 / 46.551 / **6.204** (11.8%); ndc3_unconditional ~~54.301 / 36.450 / 17.851~~ 53.102 / 33.960 / **19.142** (36.0%); ndc3_conditional ~~49.246 / 43.824 / 5.421~~ 45.883 / 41.192 / **4.691** (10.2%); range ~~5.4–17.9~~ **4.7–19.1 bn**. Uncapped solar design (POL-1, `results/pol1/arms_92390a63/pol1_results.csv`): savings ~~17.672, 21.234, 24.195, 15.788~~ **19.095, 21.031, 25.804, 15.452** (ndc2_unconditional, ndc2_conditional, ndc3_unconditional, ndc3_conditional; range ~~15.8–24.2~~ **15.5–25.8**). Which design §4.5 cites is still undecided (TASK 39 C.2). The unconditional cases still benefit about twice as much as the conditional ones in both designs. The EaaS legs still run `capital_case` `tight` (unconditional) or `moderate` (conditional) in both legs, as stored (TASK 43), so each row remains a joint effect. The pre-B* table in the body (9.66–22.36 bn) is the superseded archive and was not re-derived. Source: `results/ndc/` and `results/ndc_eaas/` post-epoch directories; `results/pol1/arms_92390a63/`.]**

> **[SUPERSEDED VALUES AND CORRECTED CAVEAT, TASK 43, 2026-10-10 (L80, TASK 39 Part C).** (1) The table and the title's 9.7–22.4 bn range are the **pre-B\* archive** (`results/_pre_bstar_20260920_1132/`), not the current epoch. Current epoch, capped solar design (`results/ndc/`, `results/ndc_eaas/`, 2026-10-03), public / EaaS / saving (bn): ndc2_unconditional 46.967 / 32.815 / 14.151; ndc2_conditional 56.146 / 49.297 / 6.848; ndc3_unconditional 54.301 / 36.450 / 17.851; ndc3_conditional 49.246 / 43.824 / 5.421 (range 5.4–17.9). Uncapped solar design (POL-1, `results/pol1/pol1_results.csv`, 2026-10-02) savings: 17.672, 21.234, 24.195, 15.788 (range 15.8–24.2). Which design §4.5 cites is undecided (TASK 39 C.2). (2) The caveat below that the public-only arms ran `unconstrained` is **wrong in all three stored sets**: the stored `capital_case` is `tight` (unconditional) or `moderate` (conditional) in both legs, so the joint-effect-of-two-dimensions reading does not apply to any stored set.]**

`conv_central`, public-only vs EaaS:

| case | public-only | EaaS | saving |
|---|---|---|---|
| ndc2_unconditional | 53.821 | 34.034 | 19.79 bn (−36.8%) |
| ndc2_conditional | 63.374 | 50.796 | 12.58 bn (−19.8%) |
| ndc3_unconditional | 58.698 | 36.339 | 22.36 bn (−38.1%) |
| ndc3_conditional | 51.573 | 41.912 | 9.66 bn (−18.7%) |

Range 9.66–22.36 bn, against 8.90–21.70 pre-O&M. The UNCONDITIONAL cases
still benefit roughly twice as much: they are the tighter caps, so the
public budget binds harder and decoupling from it is worth more. That is
the mechanism stated by the pattern rather than asserted.

**⚠ CAVEAT — this is not a one-at-a-time attribution.** The EaaS arms
run `capital_case` = `tight` (unconditional) or `moderate`
(conditional); the public-only arms run `unconstrained`. Each row
therefore differs in TWO dimensions: financing regime and capital
envelope. This is defensible as scenario design — an EaaS mechanism is
only meaningful where a public ceiling binds, which is exactly F-12's
point — but "EaaS saves 19.79 bn" is a joint effect and must be reported
as one. **The tight/moderate pairing with unconditional/conditional also
needs a stated rationale; if inherited rather than chosen, it is a
coherence question of the kind the constraint-relief cross-tab exists to
catch.**

## F-11 — Leakage
**PROVISIONAL, NOT RE-MEASURED AT THE FROZEN STATE.**

**Source (pre-epoch, TASK 54 Phase 1).** no result file; structural argument, no quantitative claim at the frozen state. Leakage magnitudes are sourced under F-31.

> **[RE-DERIVED, TASK 84, 2026-10-23 — UNCHANGED: structural claim stands; the 'do not quote a leakage percentage' restriction can now be lifted: leakage is measured and stored.** Basis: cap alone and cap plus envelope, both reported. The structural argument (`emissions_by_year` measures grid + selfgen, `emissions_capped_by_year` regulates grid only, so a binding cap displaces load to self-generation) is unchanged: it is a property of the builder. The `leakage` field is now stored in every capped `diagnostics.json`. Full-horizon (2025-2054) leakage shares for the vintage NDC 3.0 central arm: **cap alone** (`capital_case=unconstrained`) conditional **9.62%** (not flagged), unconditional **0.10%** (not flagged); **envelope-carrying** conditional **10.49%** (flagged), unconditional **21.31%** (flagged) (`diagnostics.json::leakage`). So the same structure yields leakage of 0.1% to 21% depending on whether the capital envelope is present. 2025-2045 window shares for the same arms: see F-31 (fin3, tight 3.9-8.8%) and F-21. The untested counterfactual (`selfgen_in_emissions_cap=True`) is still not exposed by any runner. Source: `results/ndc/ndc3_conditional_4ca4832d/diagnostics.json`, `results/ndc/ndc3_unconditional_60585e1f/diagnostics.json`, `results/ndc/ndc3_conditional_3fbc75ac/diagnostics.json`, `results/ndc/ndc3_unconditional_ae31b29c/diagnostics.json`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 44, 2026-10-10 — holds. The entry makes no quantitative claim at the frozen state; the structural argument is unchanged. Leakage magnitudes are in F-31.]**

The pre-O&M finding stands as a structural claim: the cap is grid-only
by construction — `emissions_by_year` measures grid + selfgen,
`emissions_capped_by_year` regulates grid only — so under a binding cap
the model displaces load to priced self-generation, whose emissions sit
outside the boundary. The NDC is met on paper while national emissions
rise.

The `leakage` field is null in the current cap summaries, so the
34.2–45.0% figures from the pre-O&M epoch are NOT restated here. **Do
not quote a leakage percentage against the frozen state until it is
re-measured.** The structural argument is unaffected; only the
magnitudes lapse.

NOT a defect — the builder deliberately separates measurement from
regulation. UNTESTED COUNTERFACTUAL: `selfgen_in_emissions_cap=True`
gives an economy-wide cap where leakage is structurally impossible. Not
exposed by any runner.

## F-21 — Tightening a grid-only cap RAISES national emissions
**The strongest policy result this model produces, and it inverts the
instrument's intent.**

> **[SUPERSEDED AND REWRITTEN AT THE CURRENT EPOCH, TASK 48 B.3, 2026-10-18.]** Every figure in this entry is now the within-epoch pair from Phase C Group 2 (2026-10-10, HEAD `3f0b382`), both arms solved in one invocation of `02_run_ndc_caps.py --include-decline-arm`. **Superseded:** the frozen-state pair (`ndc3_conditional` 51.573 bn / 486.7 Mt; `ndc3_conditional_decline` 61.815 bn / 524.2 Mt; cost +19.9%, national emissions +7.7%, leakage 35.6% → 54.8%), and the TASK 44 note of 2026-10-10 that said the decline arm had no current-epoch run. The decline arm does now have one. Effects at the current epoch: cost **+18.6%**, national emissions **+8.2%**, leakage 31.3% → 51.7%. The sign and the strict dominance survive; the magnitudes moved because the L60 cap rebuild tightened both caps, and the mechanism's step 3 did not survive as written (see there).

**Source.** `results/ndc/ndc3_conditional_decline_2cfa636a/{summary.json,diagnostics.json,timeseries.csv}` (2026-10-10 13:09:29) and `results/ndc/ndc3_conditional_604ec993/` (13:09:28); capital case `moderate`, `om_central`, `tariff_central`, `conv_central`, `hydro_flat`; capped solar design (`02_run_ndc_caps.py` passes no `solar_build_case`). Caps from `data/cost/processed/emissions_cap.csv`. **Leakage and the grid / self-generation split are not in the current summary JSON**; they are read from `logs/phase_c/2_ndc_caps_with_decline.log` (lines 49 and 59), and grid emissions are national emissions minus that leakage. Under the uncapped solar design (`results/ndc_uncapped/`, `logs/phase_c/3_02_ndc_caps_decline.log`) the pair is 47.063 → 56.157 bn (+9.09 bn, +19.3%), 431.2 → 460.7 Mt (+29.4 Mt, +6.8%), leakage 32.4% → 51.8%: the same sign in both designs.

> **[RE-DERIVED, TASK 84, 2026-10-23 — REVERSED: 'tightening a grid-only cap RAISES national emissions': on the reporting window (2025-2045) national emissions FALL, with the envelope and cap alone; the rise exists only over 2025-2054 and only with the envelope.** Basis: cap alone (new solve) and envelope-carrying, both reported; the claim concerns a cap, so the cap-alone is the one it requires. Restated on the reporting window from stored diagnostics (`emissions_grid_only_tco2_by_year`, `emissions_selfgen_tco2_by_year`, `emissions_total_tco2_by_year`), flat hold `ndc3_conditional` against `ndc3_conditional_decline`, vintage. **Envelope carried (moderate; the entry's own design), 2025-2045:** grid 220.6 -> 200.4 Mt (abatement **20.2 Mt, 9.1%**); self-generation 31.9 -> 45.1 Mt (**diesel offset +13.2 Mt**); national total 252.5 -> 245.5 Mt (**net -7.0 Mt, -2.8%**); leakage share **12.6% -> 18.4%**; diesel tonnes per grid tonne abated **0.65** (~~1.29~~). Objective 36.1405 -> 45.6638 bn (**+9.52 bn, +26.4%**; ~~+19.2%~~). **Cap alone (`capital_case=unconstrained`, solved here: `ndc3_conditional_decline_ab4ab7d7`), 2025-2045:** grid 220.6 -> 200.4 Mt (20.2, 9.1%); self-generation 32.3 -> 37.9 Mt (+5.7); national total 252.9 -> 238.4 Mt (**net -14.5 Mt, -5.7%**); leakage share **12.8% -> 15.9%**; diesel per grid tonne abated 0.28; objective 34.9955 -> 37.0494 bn (+2.05 bn, +5.9%). **Over the full horizon 2025-2054 (the basis the entry quoted):** envelope carried grid 303.0 -> 233.4 (-69.6), self-generation 35.5 -> 148.1 (+112.6), national 338.6 -> 381.5 Mt (**+42.9 Mt, +12.7%**), leakage 10.5% -> 38.8%, diesel per grid tonne 1.62 (this is the entry's rise, ~~+20.3 Mt, +5.1%~~ at the blended epoch); cap alone national 335.3 -> 271.3 Mt (**-64.0 Mt, -19.1%**), leakage 9.6% -> 14.0%. The 2045 caps are 9.163 / 5.498 Mt (unchanged). Storage 2054: envelope 18,961.7 -> 18,488.9 MWh; cap alone 60,938.8 -> 61,595.0 MWh. **What the entry would have to say:** tightening the post-2035 cap does not raise national emissions in the reporting window under either design; it raises them over 2025-2054 only when the moderate capital envelope is present, and then by 12.7%, from emissions after 2045 that the reporting window excludes. The title and the policy statement ('inverts the instrument's intent') do not survive on the window; the cap-alone result is that the tighter cap lowers national emissions by 5.7% in the window and 19.1% over the horizon. Source: `results/ndc/ndc3_conditional_3fbc75ac/diagnostics.json`, `results/ndc/ndc3_conditional_decline_74360050/diagnostics.json`, `results/ndc/ndc3_conditional_4ca4832d/diagnostics.json`, `results/ndc/ndc3_conditional_decline_ab4ab7d7/diagnostics.json`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; the sign and the strict dominance survive.** Post-epoch pair, one invocation of `02_run_ndc_caps.py --include-decline-arm` at the Phase C central arms (Group 2, 2026-10-20 02:48, `0af709f`; `results/ndc/ndc3_conditional_2e122592` and `ndc3_conditional_decline_db91e861`), capped solar design; leakage and the grid / self-generation split are read from `logs/phase_c/2_ndc_caps_with_decline.log` (still not in the summary JSON). Flat → decline: objective ~~49.246 → 58.388 bn (+9.14, +18.6%)~~ **45.883 → 54.712 bn (+8.829, +19.2%)**; national emissions ~~424.8 → 459.6 Mt (+34.8, +8.2%)~~ **397.8 → 418.1 Mt (+20.3, +5.1%)**; leakage ~~31.3% (133.1 Mt) → 51.7% (237.5 Mt)~~ **26.7% (106.1 Mt) → 46.9% (196.1 Mt)**; implied grid emissions 291.7 → 222.0 Mt (−69.7, −23.9%; unchanged to 0.1 Mt: the caps moved by 8e-9 relative at the TASK 53 rebuild); implied self-generation ~~+104.4 Mt (+78.4%)~~ **+90.0 Mt (+84.8%)**; **diesel tonnes per grid tonne abated ~~1.50~~ 1.29**; storage 2054 ~~12,493.6 → 12,137.9~~ **17,912.3 → 17,847.9 MWh**; 2045 cap 9.163 / 5.498 Mt (unchanged). Solar (cumulative **additions**, the body's two rows mix bases: its 2035 row is additions, its 2045 row is installed capacity including the 500 MW 2025 fleet): by 2035 5,739 MW in both arms (unchanged); by 2045 ~~28,507 / 30,485~~ **27,808 / 33,325 MW** (body: ~~29,007 / 30,985~~ capacity 28,308 / 33,825). **Timing now differs:** the flat arm adds 756 / 407 / 1,292 / 1,668 / 1,616 MW in 2031-2035; the decline arm adds nothing before 2035 and 5,739 MW in 2035 (both reach the same cumulative 5,739 MW by 2035; the pre-2035 timing is a cost-neutral face of the solve, not a claim). Neither arm adds solar after 2049. Under the uncapped solar design (`results/ndc_uncapped/ndc3_conditional_40bf0362` and `..._decline_41496378`, run 2026-10-20 as the five-solve Group 3 subset): ~~47.063 → 56.157 bn (+9.09, +19.3%)~~ **43.727 → 52.679 bn (+8.952, +20.5%)**, ~~431.2 → 460.7 Mt (+6.8%)~~ **400.9 → 439.8 Mt (+38.9, +9.7%)**, leakage ~~32.4% → 51.8%~~ **27.2% → 49.5%**: the same sign in both designs. Whether the budget (`moderate`) and the NIRP trajectory both bind in these solves was not re-checked (the NDC summaries carry no budget dual: `ndc_finance_utilisation` is null). **Premise correction:** this entry quotes no shadow price; the FIN-3 carbon shadows 115.05 and 212.11 are in F-31. Source: as stated.]**

Plan 4.2 asked for a post-2035 cap trajectory as a sensitivity, on the
grounds that holding caps flat 2035-2045 is an implicit "climate
ambition stops in 2035" assumption. The arm was built as
`ndc3_conditional_decline`: identical to `ndc3_conditional` through
2035, then declining linearly from that anchor toward zero at 2060 —
Nigeria's own stated net-zero target. At 2045 the cap is 5.498 MtCO2
against the flat hold's 9.163, a 40% tightening.

**The result is strictly dominated on both axes.**

| | `ndc3_conditional` | `ndc3_conditional_decline` |
|---|---|---|
| 2045 cap | 9.163 MtCO2 | 5.498 MtCO2 (−40%) |
| objective | 49.246 bn | **58.388 bn (+9.14, +18.6%)** |
| **national emissions** | **424.8 Mt** | **459.6 Mt (+34.8, +8.2%)** |
| leakage | 31.3% (133.1 Mt) | **51.7% (237.5 Mt)** |
| implied grid emissions | 291.7 Mt | 222.1 Mt (−23.9%) |
| implied self-generation | 133.1 Mt | 237.5 Mt (+78.4%) |
| storage 2054 | 12,493.6 MWh | 12,137.9 MWh |
| solar built by 2035 (cumulative) | 5,739 MW | 5,739 MW |
| solar built by 2045 (cumulative) | 29,007 MW | 30,985 MW |

Grid emissions fall by 69.6 Mt. Self-generation emissions rise by 104.4
Mt. **For every tonne abated on the grid, 1.50 tonnes appear in
diesel** — and it costs 9.14 bn to achieve.

### The mechanism, in six steps

**1. The cap regulates a boundary, not the atmosphere.**
`emissions_capped_by_year` constrains grid emissions only;
`emissions_by_year` measures grid plus self-generation. This is not a
modelling error — it mirrors the NDC's own accounting boundary, and the
builder separates measurement from regulation deliberately (F-11).

**2. Demand is exogenous.** Tightening the cap removes grid gas
generation but not the load it was serving. That load must go
somewhere.

**3. The clean substitutes are rationed.** Solar is limited twice: by
the public capital envelope (`moderate`, 7.738 bn, which binds) and by
the NIRP deployment trajectory. **The frozen entry's claim that the
decline arm "front-loads solar hard — 5,739 MW at 2035 against the flat
arm's 1,616" does not survive the current epoch and is withdrawn.** At
the current epoch both arms have built 5,739 MW by 2035; 1,616 MW is the
flat arm's *addition in 2035* (the decline arm's 2035 addition is also
1,616), so the frozen row compared an annual addition with a cumulative
total. The arms diverge later: the decline arm has built 30,985 MW by
2045 against 29,007 MW (+1,979 MW), and neither adds solar after 2049
(decline last adds in 2049, flat in 2048). Whether the envelope and the
trajectory both bind in these two solves was not re-checked in this
correction. `unserved` is disabled by construction under a genset case
(defect D11).

**4. So the residual goes to self-generation, which is 2.3x dirtier.**
Converted CCGT emits 0.3963 tCO2/MWh_e; the genset backstop at
`genset_central` emits 0.902. Every MWh moved from grid to diesel
roughly doubles its carbon content.

Indicatively, at a fleet-average grid EF near 0.43, removing 69.6 Mt of
grid emissions means removing roughly 162 TWh of grid generation, while
104.4 Mt of additional diesel represents roughly 116 TWh. So about 71%
of the displaced generation went to diesel and 29% to the extra solar —
and because the diesel share is 2.3x more intensive, it more than
cancels the genuine abatement.

**5. The optimiser is not choosing diesel because it is cheap.** Diesel
costs 445 USD/MWh against a grid tariff of 95. It is chosen because
solar is unavailable at the margin once capital and deployment limits
bind, and the cap has made grid gas infeasible. The model is minimising
cost subject to a constraint that forbids the cheap clean option and the
cheap dirty option simultaneously.

**6. And no instrument in the system can see it.** `carbon_case` is
`no_policy`, because Nigeria has no carbon price. Under defect D3 the
cap sees grid emissions only while a carbon price would see the total —
but no price exists. Diesel's emissions are unpriced and unregulated,
so nothing pushes back.

### Why this completes the abatement story

F-8, F-9 and F-21 together say three things about what actually abates
emissions in a gas-constrained, capital-constrained system:

- **Efficiency works.** Conversion is worth 33.15 to 37.10 bn under
  caps (F-8), and cuts emissions 26.2-26.9% in parallel.
- **Fuel does not.** Additional gas deliverability is worth EXACTLY
  ZERO under a binding cap, to the digit, in all four cap cases (F-9).
- **Tightening a boundary-limited cap is worse than useless.**
  It costs 9.14 bn and raises national emissions by 34.8 Mt (F-21).

The first two were already reportable. The third turns F-11's
observation — the NDC is met on paper while national emissions rise —
into a **comparative-statics result**: tightening the grid-only target
CAUSES national emissions to rise. That is a statement about the
instrument's design, not about this model's boundary.

### Scope and caveats, stated

**Magnitude depends on the genset arm.** At `genset_srmc` (0.729
tCO2/MWh, 75% load) the effect weakens; at `genset_high` (4.638, A2EI
metered) it becomes far larger. `genset_central` (0.902, 25% load) is
the sourced central arm and the basis of the figures above. The SIGN is
robust across all three — diesel is dirtier than converted CCGT in every
arm — but the ratio is not.

**Widening the boundary is not an available remedy.** Phase 2.6's defect
D11 established that counting self-generation inside the cap makes NDC3
unconditional INFEASIBLE under a tight envelope in all three genset
arms. So "just regulate total emissions instead" does not resolve this;
it converts a perverse outcome into an infeasible one. That is part of
the finding, not a gap in it.

**The decline shape is an analyst construction** (register L24). The
2060 endpoint is policy-anchored — Nigeria's own net-zero target — but
the linear path between the 2035 NDC anchor and that endpoint is this
project's choice, stacked on plan 4.1's apportionment rule, which is
also a construction. Two stacked constructions; state both wherever this
result is cited.

**The arm is a sensitivity, not a central case.** It is opt-in via
`--include-decline-arm` on `02_run_ndc_caps.py`, default off, and is not
in `run_central_sweep.sh`. The four flat-hold scenarios are unmodified.
Re-solved in the same invocation at the current epoch: `ndc2_unconditional`
46.967, `ndc2_conditional` 56.146, `ndc3_unconditional` 54.301,
`ndc3_conditional` 49.246 bn (the earlier "reproduce to the digit" held against
the frozen state only).

**Buffer treatment is unique to this arm.** `ndc3_conditional_decline`
continues its linear decline through the buffer years 2046–2054
(`scripts/00_build_emissions_cap.py:315-323`), deliberately departing from
the flat-hold convention every other scenario uses. Neither
`methodology_genself_backstop.md` §4.5 nor `methodology_demand_horizon.md`
§3.2 records the exception. F-21's buffer is therefore not comparable to any
other arm's.

### The policy statement this supports

A national emissions target that regulates grid generation while the
displaced load is served by unregulated self-generation does not reduce
emissions — under a binding capital constraint it increases them, and
charges roughly 9 bn for the privilege. The binding constraint on
Nigerian decarbonisation is not the stringency of the target. It is the
availability of capital to build the clean capacity the target's
displaced load requires, and the boundary that lets diesel absorb
whatever the grid is forbidden to supply.

This is the thesis's own argument, arriving from the emissions side
rather than the financing side: **cap stringency without capital
mobilisation is counterproductive, not merely ineffective.**

### A second instrument, not a deliberate tightening (register L60/L63)

The L60 cap rebuild (`docs/limitations_register.md` L60) replaced an
unsourced power-share denominator with an inventory-derived one,
TIGHTENING the NDC cap by 3.06-8.73% depending on year and arm (as
actually applied, `data/cost/processed/emissions_cap.csv` vs the
pre-rebuild snapshot — L60's own first-written table projected
3.27-9.01% ahead of the real run and is corrected in place) — for a
reason unrelated to this finding; L60's purpose was sourcing the
share, not testing stringency. Re-running FIN-3 (`05_run_fin3_capital_sweep.py`)
at the new cap shows leakage rising in **all fifteen** capital-case ×
collection-rate cells, with two crossing the compliance threshold
(register L63; F-31 above has the updated table). **This corroborates
F-21's mechanism through an instrument that was never a deliberate
tightening test** — independent confirmation that tightening a
grid-only boundary raises leakage, arrived at as a side effect of an
unrelated correction.

**Not presented as unqualified.** L63's own two qualifications apply
here without exception: part of the leakage rise is a level shift with
no cap-stringency channel (the `unconstrained` floor rose too, 5.81%→
7.37%, and that cell has no budget constraint to tighten); and one cell
(`expansion/current`, 5.84%→8.10%) rose without crossing the flag,
which is a property of the declared 0.10 threshold, not evidence that
its leakage is acceptable. The corroboration is real; it is not clean.

### Supply-chain CO2e: the result holds, and the break-even says how firmly

**Plan 5.2, closed as a post-hoc sensitivity. No code, no solve, no
change to any reported model output.**

The model's emission factor is **combustion only**.
`EF_TCO2_PER_MWH_TH = 0.1823` (`src/io.py:409`) is the IPCC default
stationary-combustion factor — 56.1 kg CO2/GJ on an NCV basis, HHV-
adjusted by /1.108 — and the electrical factor is derived at build time
as `EF_e = EF_th / eta`, so efficiency and emissions cannot drift apart.
Nothing upstream enters anywhere in that chain: methane vented or leaked
in production, processing and transmission is absent, and so is flaring.

This matters for Nigeria more than for most systems. Nigeria is among
the world's largest gas-flaring nations, and flaring and fugitive losses
are exactly the categories a supply-chain factor captures. So the
model's emissions are a **floor**, not an estimate, and the gap between
floor and truth is plausibly wider here than the international default
would suggest.

**Method.** A supply-chain CO2e uplift is applied post-hoc to **grid gas
emissions only**, over a declared range of **+10% to +25%** on
combustion CO2. Self-generation is deliberately excluded — diesel's own
upstream emissions are real but no defensible Nigerian figure was
available, and excluding them understates the arm that burns more
diesel. Every figure below is therefore a **conservative floor on the
gap**.

**Classification: ASSUMED, not SOURCED.** The +10-25% range brackets
published estimates for gas supply chains in systems with material
flaring, but no Nigeria-specific gas-to-power supply-chain intensity was
obtained at the granularity this model would need. That is the L20
treatment: declared, with a direction of bias, rather than a figure
wearing a citation it has not earned.

**Result.**

| gas supply-chain uplift | flat CO2e | decline CO2e | gap |
|---|---|---|---|
| +0% (combustion only, as modelled) | 424.8 | 459.6 | **+34.8** |
| +10% | 453.9 | 481.8 | +27.8 |
| +15% | 468.5 | 492.9 | +24.4 |
| +20% | 483.1 | 504.0 | +20.9 |
| +25% | 497.7 | 515.1 | **+17.4** |

*(Recomputed 2026-10-18 by arithmetic on the current pair, grid = national − leakage; no solve.)*

**The break-even is 50%.** The uplift narrows the gap, because it scales
grid emissions — which the tightened cap reduces — while leaving the
diesel that replaces them untouched. But the tightened cap removes
**69.6 Mt** of grid emissions and creates **104.4 Mt** of diesel, so the
uplift must reach `104.4 / 69.6 − 1 = 50.0%` before the first catches the
second.

**That is roughly double the top of any plausible range.** At the
declared maximum of +25%, the declining cap still emits 17.4 Mt more
than the flat hold. F-21 does not survive narrowly; it survives with a
margin of about 2x on the parameter that would overturn it.

**Why the break-even is the right statistic to report.** The uplift
range is ASSUMED and cannot be sourced precisely. A break-even does not
require it to be: it states how much would have to be true for the
finding to reverse, which is falsifiable without pinning the parameter
down. A referee who disputes +10-25% can substitute their own figure and
read the answer off the table.

**A methods caution, recorded because it cost two wrong predictions.**
The direction of this sensitivity is not intuitable and depends entirely
on which emissions the uplift touches. Applied to grid gas ALONE it
narrows the gap (the table above). Applied to BOTH grid gas and diesel
at comparable rates it **widens** the gap, because the declining arm
carries far more self-generation (237.5 Mt against 133.1) and therefore
absorbs more of the correction — at +15% on both, the gap grows from
34.8 to 40.0 Mt. Two defensible specifications, opposite directions.
The specification must be stated before the result is read, not chosen
after seeing it. Recorded here in the same spirit as M-3: the choice of
what a correction applies to is itself a modelling decision, and an
unstated one can flip a sign.


---
# Section 4 — Identification

## F-12 — Total EaaS solar is bounded above by total solar, below by the public budget; the split is not identified
**CONFIRMED at the frozen state, and the evidence is stronger.**

**Source (pre-epoch, TASK 54 Phase 1).** `results/f3/*/summary.json` (3 capital cases); runner `26_run_f3_identification.py` (**not in `run_phase_b.sh`**; also driven by Phase C, `logs/phase_c/0_26_f3_moderate.log`), 2026-10-03 (file date); commit inferred `2b22fe9`.

> **[RE-DERIVED, TASK 84, 2026-10-23 — MOVED: floors, ceiling and Z*; the identification result (flat face, unidentified split) is unchanged.** Basis: cap plus public budget (by design). Vintage `results/f3/` (`ndc3_conditional`, `tariff_central`, `om_central`, nameplate availability, tol 1e-6; Z* ~~41,191,785,632.44~~ **34,995,532,352.70 USD**; budgets are the `capital_envelope_scenarios()` values). Floor / ceiling / interval (MW): tight 5.137 bn ~~22,270.7 / 48,582.8 / 26,312.1~~ **17,172.3 / 46,543.8 / 29,371.4**; moderate 8.732 bn ~~6,284.1 / 48,582.8 / 42,298.7~~ **3,354.9 / 46,543.8 / 43,188.9**; adequacy 10.273 bn ~~1,929.8 / 48,582.8 / 46,653.0~~ **95.1 / 46,543.8 / 46,448.7**; expansion 12.328 bn **0.0 / 46,543.8 / 46,543.8** (floor unchanged at 0); unconstrained **0.0 / 46,543.8 / 46,543.8**. **The face is still flat and the MW split still unidentified.** The maximum is identical across capital cases and across tariffs 75 / 95 / 115 (floors ~~7,465.8 / 6,284.1 / 5,774.2~~ **4,226.1 / 3,354.9 / 3,001.2** at moderate); it rises with tolerance (46,543.8; 46,769.5; 47,303.4 at 1e-6, 1e-5, 1e-4, ~~48,582.8; 48,690.5; 49,338.5~~), and `solar_public_add_mw_total` is exactly 0.0 at the maximising solve in every file; the floor rises as the budget falls (0.0, 0.0, 95.1, 3,354.9, 17,172.3). The `observed`-availability moderate file has Z* 45,331,334,568.83 and floor / ceiling 20,552.8 / 47,779.2 (`ndc3_conditional_moderate_4532d389`). Basis: the entry concerns a budget under the cap, so the envelope-carrying (budgeted) design is the one it needs. Source: `results/f3/ndc3_conditional_tight_0cd55c88/summary.json`, `results/f3/ndc3_conditional_moderate_8ab27de2/summary.json`, `results/f3/ndc3_conditional_adequacy_4718da0a/summary.json`, `results/f3/ndc3_conditional_expansion_d0d4db68/summary.json`, `results/f3/ndc3_conditional_unconstrained_8e70de2b/summary.json`, `results/f3/ndc3_conditional_moderate_981533ff/summary.json`, `results/f3/ndc3_conditional_moderate_9338f8ca/summary.json`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; conclusions hold.** Post-epoch `results/f3/` (Phase B central sweep, 2026-10-20 00:38-00:42; `ndc3_conditional`, `tariff_central`, `om_central`, `gas_avail_nameplate`, tol 1e-6; Z* 41,191,785,632.44 USD; budgets are the `capital_envelope_scenarios()` values). Floor / ceiling / interval (MW), pre-epoch in brackets: tight 5.137 bn **22,270.7 / 48,582.8 / 26,312.1** (~~31,875.1 / 48,578.9 / 16,703.8~~); moderate 8.732 bn **6,284.1 / 48,582.8 / 42,298.7** (~~10,989.7 / 48,578.9 / 37,589.2~~); adequacy 10.273 bn **1,929.8 / 48,582.8 / 46,653.0** (~~5,275.7 / 48,578.9 / 43,303.2~~); expansion 12.328 bn **0.0 / 48,582.8 / 48,582.8** (~~400.8~~ floor); unconstrained **0.0 / 48,582.8 / 48,582.8**. **The face is still flat and the MW split is still unidentified.** The maximum is identical across capital cases and across tariffs 75 / 95 / 115 (floors 7,465.8 / 6,284.1 / 5,774.2 at moderate); it rises with tolerance (48,582.8; 48,690.5; 49,338.5 at 1e-6, 1e-5, 1e-4); `solar_public_add` is exactly 0.0 at the maximising solve in every file; the minimising and maximising solutions cost within 23 USD of each other (relative 6e-10) in every capital case; the normal solve sits inside the interval (e.g. 33,527.0 MW at tight, 17,312.8 at moderate). The floor still rises as the budget falls (0.0, 0.0, 1,929.8, 6,284.1, 22,270.7); expansion's floor reaching zero means the 12.3 bn budget no longer forces any public solar (it did, 400.8 MW, pre-epoch). The 16,703.8 MW tight-budget interval quoted elsewhere is now **26,312.1 MW** (the 10,969 MW in the body is the frozen figure). The ceiling moved ~~48,578.9~~ 48,582.8 MW (+0.008%) between the two epochs: it is total solar, not a bankability ceiling. Nothing here changes constraint 4.4 of the chapter handover: the face did not stop being flat. Source: `results/f3/ndc3_conditional_{a1d01dfc,2eab9df6,8c4f4479,17efe111,41544db3}/summary.json`.]**

> **[RE-DERIVED, TASK 44, 2026-10-10 — moved; conclusions hold. Current epoch (`results/f3/`, 2026-10-03; `ndc3_conditional`, `tariff_central`, `om_central`, tol 1e-6; budgets are now the `capital_envelope_scenarios()` values): tight 5.137 bn: 31,875.1 / **48,578.9** (interval 16,703.8); moderate 8.732 bn: 10,989.7 / 48,578.9 (37,589.2); adequacy 10.273 bn: 5,275.7 / 48,578.9 (43,303.2); expansion 12.328 bn: 400.8 / 48,578.9 (48,178.1); unconstrained: 0.0 / 48,578.9 (48,578.9). The maximum is identical across capital cases and across tariffs 75/95/115 (floors 12,414.1 / 10,989.7 / 10,373.0), rises with tolerance (48,578.9; 48,651.7; 49,274.3 at 1e-6, 1e-5, 1e-4), and `solar_public_add` is exactly 0.0 at the maximising solve in every file; the floor rises as the budget falls (0.0, 400.8, 5,275.7, 10,989.7, 31,875.1). `ndc_eaas` totals now run 43,623.8 to 53,912.5 MW across the eight cap × conversion cases (the pre-O&M 58,975.3 is unchanged). The 10,969 MW tight-budget interval quoted elsewhere is the frozen-state figure; at the current epoch it is 16,703.8 MW. Z* = 43,824,295,3xx USD.]**

Lexicographic min/max on total EaaS MW at fixed Z*, `ndc3_conditional`,
`tariff_central`, `om_central`, tol 1e-6:

| capital_case | budget | min MW | max MW | interval |
|---|---|---|---|---|
| tight | 4.552 bn | 38,518.0 | **49,486.6** | 10,968.6 |
| moderate | 7.738 bn | 14,944.8 | **49,486.6** | 34,541.8 |
| unconstrained | none | **0.0** | **49,486.6** | 49,486.6 |

**The maximum is 49,486.6 in all three, and it is total solar, not a
bankability ceiling** (methodology.md §7.7; claims ledger C-2,
adjudicated 2026-09-20 in §7.7's favour). At the maximising solve,
`solar_public_add_mw_total` = **exactly 0.0** in every current-epoch arm
(`results/f3/*/summary.json`), so the EaaS maximum is simply the total
solar the system builds when public solar is pushed to zero — there is
no separate ceiling for EaaS to hit. Two further checks rule out a
bankability-ceiling reading: the maximum is identical (49,486.6) at
`tariff_low` / `tariff_central` / `tariff_high` (75 / 95 / 115
USD/MWh), which a tariff-and-capex bankability ceiling could not be; and
it RISES with the cost tolerance (49,486.6 at tol 1e-6, 49,556.2 at
1e-5, 50,157.9 at 1e-4), which a hard constraint cannot do but a
total-solar quantity can, as looser tolerance admits more solutions with
more total solar built. Only the floor moves with the budget: remove it
and the minimum goes to exactly zero, because public solar is then
unconstrained and costs the same; tighten it and the floor rises.

So the split is bounded BELOW by the public budget and ABOVE by total
solar itself — the interval is the gap between a budget constraint and
an unconstrained system-wide quantity, not two constraints binding at
opposite ends. A sufficiently tight budget would still close it, since
it still forces the floor up toward the same ceiling.

**Why O&M did not break this, which is the mechanism worth stating.**
O&M is charged identically to public and EaaS solar, so it cancels at
the public/EaaS margin entirely. It changes the LEVEL of total solar —
`ndc_eaas` totals move 45,633 to 55,925 MW across cap and conversion
cases, where pre-O&M they were 58,975.3 in all four — without touching
the ceiling on the SPLIT. A cost both channels bear cannot move a
constraint that discriminates between them. This is F-2 and F-3 one
layer down: costs borne symmetrically change what the system pays, never
what it chooses between.

The ceiling moved 47,528.0 -> 49,486.6 MW (+4.1%) between epochs. Both
are total-solar figures at their respective epochs, not bankability
ceilings; only the epoch differs.

REPORTING RULE: any EaaS figure must carry both its `capital_case` and
its `tariff_case`, and be reported as a point in the stated interval.

## F-13 — The result is robust to the tariff
**CONFIRMED, and sharper than pre-O&M.**

**Source (pre-epoch, TASK 54 Phase 1).** `results/f3/*/summary.json`, the same files as F-12; runner `26_run_f3_identification.py` (**not in `run_phase_b.sh`**), 2026-10-03; commit inferred `2b22fe9`.

> **[RE-DERIVED, TASK 84, 2026-10-23 — MOVED: Z* level and floors; the tariff robustness and the sign check hold.** Basis: cap plus public budget (by design). Vintage `results/f3/` (moderate, tol 1e-6): Z* `tariff_low` / `tariff_central` / `tariff_high` **34,995,532,357.43 / 34,995,532,352.70 / 34,995,532,365.07** (spread ~~0.0 USD~~ **12.4 USD** on 35.0 bn, 3.5e-10 relative); at `tariff_central` moderate / tight / unconstrained 34,995,532,352.70 / 34,995,532,332.12 / 34,995,532,328.49 (spread ~~69.58~~ **24.2 USD**, 6.9e-10). Floors ~~7,465.8 / 6,284.1 / 5,774.2~~ **4,226.1 / 3,354.9 / 3,001.2 MW**, ceiling 46,543.8 at all three: the sign check (a higher tariff forces less public solar) holds. The DIST-2 / DIST-5 scope paragraph is input-side and untouched. Source: `results/f3/ndc3_conditional_moderate_981533ff/summary.json`, `results/f3/ndc3_conditional_moderate_8ab27de2/summary.json`, `results/f3/ndc3_conditional_moderate_9338f8ca/summary.json`, `results/f3/ndc3_conditional_tight_0cd55c88/summary.json`, `results/f3/ndc3_conditional_unconstrained_8e70de2b/summary.json`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; conclusions hold.** Post-epoch `results/f3/` (moderate, tol 1e-6): Z* `tariff_low` / `tariff_central` / `tariff_high` all **41,191,785,632.44** (spread ~~16.16 USD~~ **0.0 USD**); at `tariff_central` moderate / tight / unconstrained 41,191,785,632.44 / 41,191,785,632.44 / 41,191,785,562.86 (spread ~~17.49~~ **69.58 USD** on 41.2 bn, 1.7e-9 relative; the unconstrained case carries no budget constraint). Floors ~~12,414.1 / 10,989.7 / 10,373.0~~ **7,465.8 / 6,284.1 / 5,774.2 MW**, ceiling 48,582.8 at all three: the sign check (a higher tariff forces less public solar) holds. The table in the body (41.9 bn) is the pre-B\* epoch. The DIST-2 / DIST-5 scope paragraph is unchanged and untouched by this re-derivation. Source: `results/f3/ndc3_conditional_{87b0d49c,2eab9df6,a23e1deb}/summary.json`.]**

> **[RE-DERIVED, TASK 44, 2026-10-10 — moved; conclusions hold. Current epoch (`results/f3/`): Z* tariff_low 43,824,295,323.60; tariff_central 43,824,295,307.44; tariff_high 43,824,295,308.32 (spread 16.16 USD on 43.8 bn); at `tariff_central` moderate 43,824,295,307.44 / tight 43,824,295,317.60 / unconstrained 43,824,295,324.93 (spread 17.49 USD). Floors 12,414.1 / 10,989.7 / 10,373.0 MW at moderate, ceiling 48,578.9 at all three: the sign check (higher tariff, less forced public solar) holds. The table and 41.9 bn values above are the pre-B\* epoch.]**

At `capital_case = moderate`, varying only the tariff:

| tariff | USD/TWh | Z* | floor MW | ceiling MW |
|---|---|---|---|---|
| tariff_low | 75M | 41,911,893,367.70 | 16,227.7 | 49,486.6 |
| tariff_central | 95M | 41,911,893,366.56 | 14,944.8 | 49,486.6 |
| tariff_high | 115M | 41,911,893,413.71 | 14,235.7 | 49,486.6 |

**A 47 USD spread on 41.9 bn.** Z* is also invariant across the capital
envelope at `tariff_central`: moderate 41,911,893,366.56 / tight
41,911,893,376.22 / unconstrained 41,911,893,320.20 — a 56 USD spread.

The tariff moves the FLOOR and not the ceiling, and moves it in the
right direction: a higher tariff makes EaaS more bankable, so less
public solar is forced in (16,227.7 -> 14,235.7 as the tariff rises
from 75M to 115M). That sign is a check on the mechanism, not just a
robustness statement.

CONDITIONAL on the current capex treatment. The two solar channels are
financially identical in this model except for two sourced corrections
neither of which is implemented (register L16, the accurate entry):
[DIST-2] behind-the-meter avoidance of the 7% transmission loss factor
(favours EaaS, excluded because the metering point is declared at the
busbar), and [DIST-5] a distributed-commercial-PV capex premium (+16%
2025 rising to +37% 2045, NREL ATB) plus a capacity-factor and seasonal
penalty against utility-scale (disfavours EaaS). Both are sourced in
`docs/shape_factors_and_dist5.md`; **neither is wired into `src/`**. An
earlier version of this note, and register L25, stated DIST-5 had been
implemented and measured at under 0.03% objective movement with EaaS
deployment "identical to the megawatt." That was wrong: no code path at
HEAD or on any branch computes DIST-5's effect, so no such measurement
exists. See register L25 and forensic audit A63 for the commit chain that
produced the false "implemented" claim.

Because DIST-2 and DIST-5 point in opposite directions and neither is
implemented, L16's conclusion holds: their net direction is not
established. This is an open scope gap, not a result shown to be
conservative or inframarginal.


---

# Section 5 — Against NIRP

## F-14 — NIRP's gas trajectory never binds on the cost-optimal path
**PROVISIONAL. Unchanged in kind.**

**Source (pre-epoch, TASK 54 Phase 1).** `results/baseline_sliced/summary__{024fdf9f gas_ndc_17gw, 376ed5e0 gas_none}.json`; runner `01b_run_baseline_sliced.py`, 2026-10-03, `2b22fe9` (documentary). **Provenance NOT established** for the NIRP gas-cap series (11,932/22,113/27,201/23,906 MW); the model side only.

> **[RE-DERIVED, TASK 84, 2026-10-23 — MOVED: 2045 gas capacity and the cap ratio.** Basis: uncapped baseline. `gas_ndc_17gw` and `gas_unconstrained` (central) again return identical decision variables (2054 totals and annual additions within 1e-3), objectives 4.04 USD apart (23,323,068,344.50 and 23,323,068,340.46). The cap series is an input and unchanged. Cost-optimal gas capacity in 2045 is ~~6,053~~ **9,232 MW** (start 10,947 MW; 2035 11,677; 2041 11,782; 2054 ~~6,482~~ 8,174); the 2045 cap of 23,906 MW is ~~3.95~~ **2.59** times it. Gas capacity still falls from the starting fleet after 2041, and the cap never binds. Cumulative gas built 4,507.6 MW (2032-2054). Source: `results/baseline_sliced/summary__bc542471.json`, `results/baseline_sliced/summary__661b03c1.json`, `results/baseline_sliced/diagnostics__661b03c1.json`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; conclusion holds.** `gas_ndc_17gw` and `gas_unconstrained` (central) again return identical decision variables (max |Δ| storage/solar 0.0000), with objectives ~~5.8~~ **5.7 USD** apart (24,986,620,949.18 and `gas_ndc_17gw`), not byte-identical. The cap series (11,932 / 22,113 / 27,201 / 23,906 MW) is an input and is unchanged. Cost-optimal gas capacity in 2045 is ~~7,221~~ **6,053 MW** (start 10,947 MW unchanged; 2054: 6,482 MW unchanged); the 2045 cap is ~~3.31~~ **3.95** times it. **The 2045 capacity moved although the 2054 plan did not:** the declining storage capex changed the retirement/build timing (gas capacity differs by up to 1,814 MW in 2037, storage by up to 7,257 MWh in 2041, solar by up to 376 MW in 2043 against the pre-epoch central), which is why year-t quantities cannot be carried across the epoch while terminal-state quantities can. The NIRP cap-series source is still untraced (TASK 54 Phase 1). Source: `results/baseline_sliced/summary__{a675ad72,6046f759}.json`, `timeseries__*.csv`.]**

> **[RE-DERIVED, TASK 44, 2026-10-10 — moved; conclusion holds. `gas_ndc_17gw` and `gas_unconstrained` now return identical decision variables, with objectives 5.8 USD apart (27,317,670,155.72 and 27,317,670,161.50), not byte-identical; the cap series (11,932 / 22,113 / 27,201 / 23,906 MW) is unchanged. Current cost-optimal gas capacity is 7,221 MW in 2045 (peak 10,947 MW at the start; the cap is 3.31 times it, was ~3.7); capacity still falls from the starting fleet.]**

`gas_ndc_17gw` (NIRP-derived cap, nameplate basis: 11,932 MW 2030 /
22,113 MW 2035 / 27,201 MW 2041 / 23,906 MW 2045) and
`gas_unconstrained` return BYTE-IDENTICAL results. The cap sits ~3.7x
above the cost-optimal 2045 capacity and never binds; gas capacity FALLS
from the 10,947 MW starting fleet.

The divergence between this thesis and NIRP on gas capacity is therefore
NOT a cost disagreement. Costing NIRP's path would require a capacity
FLOOR, not a ceiling. As a ceiling the trajectory is non-binding and the
comparison is qualitative: **NIRP plans capacity the fuel supply cannot
run.**

## F-15 — NDC 3.0's "17 GW" is a derated figure; the nameplate commitment is ~22 GW
**PROVISIONAL. Unaffected by the model state.**

**Source (pre-epoch, TASK 54 Phase 1).** no result file; input-data comparison (NIRP series against NDC 3.0 text). Arithmetic only. External-document page references not traced here.

> **[RE-DERIVED, TASK 84, 2026-10-23 — UNCHANGED: arithmetic reproduced; the NDC 3.0 'roughly double' text and the 0.77 derate source are not checked here.** Basis: input arithmetic. From `gas_ndc_17gw_trajectory_mw()` (`src/scenarios.py`): 2030 11,932 / 2035 22,113 / 2041 27,201 / 2045 23,906 MW; starting fleet 10,947 MW (vintage `diagnostics__661b03c1.json::gas_capacity_mw_by_year['2025']`). 22,113 / 10,947 = **2.0200**; 17,027 / 10,947 = **1.5554**; 22,113 x 0.77 = 17,027.0. Identical to the entry. `NDC_3.0.pdf` and `NIRP_2024.pdf` are in `literature/`; their page references were not read in this task. Source: `src/scenarios.py (gas_ndc_17gw_trajectory_mw)`, `results/baseline_sliced/diagnostics__661b03c1.json`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 44, 2026-10-10 — holds. Input-data comparison; no model output enters the quantitative claim, so the epoch does not apply.]**

The NIRP-derived series is on a derated basis (~0.77). Converting to
nameplate gives 22,113 MW at 2035. Cross-check: NDC 3.0 describes its
gas target as roughly double current levels; 22,113 / 10,947 = 2.02x,
while the underated 17,027 / 10,947 = 1.55x does not match.

Both NDC 3.0's stated 17 GW and NIRP's derived 17,027 MW are EFFECTIVE
figures. Any comparison against installed capacity is comparing
different quantities unless the derate is stated. The 0.77 derate is
DECLARED, the one number in the chain not traceable to a NIRP table.

## F-16 — NIRP's 2045 hydro exceeds independent exploitable potential
**PROVISIONAL. Unaffected by the model state.**

**Source (pre-epoch, TASK 54 Phase 1).** no result file; input-data comparison (NIRP against Energy Commission of Nigeria estimates). Arithmetic only. External-document references not traced here.

> **[RE-DERIVED, TASK 84, 2026-10-23 — UNVERIFIABLE: the inputs (NIRP 20.6 GW, 21 GW ceiling; ECN 14,120 + ~3,500 MW) are external and were not read here.** Basis: external documents. The percentages follow from the entry's own inputs: 20.6 / 21 = **98.1%**, 20.6 / 17.62 = **116.9%** (reproduced). Whether 20.6 GW, 21 GW, 14,120 MW and 3,500 MW are what NIRP and the Energy Commission of Nigeria state was not checked: `literature/NIRP_2024.pdf` exists, the ECN source is not in `literature/` (search: file names in `literature/`, `docs/*.md` for '14,120': only `findings_*`, `phase_1_5_hydro.md`, `task_log_48_evidence_spine.md`; text only, PDFs not read). Not carried forward as verified. Source: `literature/NIRP_2024.pdf (not read)`, `docs/phase_1_5_hydro.md`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 44, 2026-10-10 — holds. Input-data comparison; no model output enters the quantitative claim, so the epoch does not apply.]**

NIRP builds 20.6 GW by 2045 against its own stated ceiling of ~21 GW —
98.1%, no margin. Against the Energy Commission of Nigeria's independent
estimate of 14,120 MW exploitable large hydro plus ~3,500 MW small hydro
(17,620 MW), it is **116.9% — it exceeds total exploitable potential.**

NIRP's 21 GW is a MODELLING ceiling: 18.6 GW candidate mixing
site-specific and GENERIC plants, generic capped at 15 GW nationally,
zonal allocation from 24 sites in JICA (2019). A plan consuming 98% of
its own ceiling and 117% of an independent estimate is an upper bound
presented as a forecast.

## F-17 — The demand-base gap with NIRP is definitional
**PROVISIONAL. Unaffected by the model state.**

**Source (pre-epoch, TASK 54 Phase 1).** no result file; input-data comparison (NERC 2024 against NIRP demand base). Arithmetic only. External-document references not traced here.

> **[RE-DERIVED, TASK 84, 2026-10-23 — UNCHANGED: model side reproduced (37.09 TWh base and the loss ladder); the NIRP-side figures (30.5, 32.3, 222 and 297-301 TWh) are external and not checked here.** Basis: input arithmetic. Model side, `src/scenarios.py` lines 29-52: gross generation 37.09 TWh; x 0.93 (7% TLF) = 34.49 (entry 34.50); x (1 - 0.364) = 21.94 (the source docstring states 21.97). Gap to NIRP 30.5: **6.59 TWh, 17.8%** (entry 6.6 TWh, 18%); NIRP 2024 against its 2022 base 30.5 / 32.3 - 1 = -5.6% (reproduced). Not model output; the vintage generation does not touch it. Source: `src/scenarios.py`, `literature/NIRP_2024.pdf (not read)`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 44, 2026-10-10 — holds. Input-data comparison; no model output enters the quantitative claim, so the epoch does not apply.]**

This model: 37.09 TWh, gross generation at the busbar (NERC 2024). NIRP:
30.5 TWh for the same year. Gap 6.6 TWh, 18%.

Four terms account for it: basis year (NIRP anchors 2022 observed 32.3
TWh); metering point (this model's loss ladder runs 37.09 gross -> 7%
TLF -> 34.50 to DisCos -> 36.4% ATC&C -> 21.97 collected, and NIRP does
not state which point its figure occupies); exports, inside NERC
generation and outside NIRP's domestic demand; and the DESG delay
assumption.

**NIRP's 2024 modelled grid demand (30.5) is BELOW its own 2022 observed
figure (32.3)** — a forecast starting below its base year, implying a
5.6% contraction. Its early-year figures are a weak calibration
benchmark. Its 2045 on-grid demand also appears as 222 TWh at p.49 and
297–301 TWh in Tables 6 and 15, a ~75 TWh unexplained internal spread.

---

# Section 6 — What bounds this thesis's own claim

## F-18 — The solar-dominant pathway is contingent on hydro not expanding
**CONFIRMED, and the bound is larger than pre-O&M.**

**Source (pre-epoch, TASK 54 Phase 1).** `results/baseline_sliced/summary__{02c17f55, 33cbd361 hydro_committed, 9221de7b hydro_ndc}.json`; runner `01b_run_baseline_sliced.py`, 2026-10-03, `2b22fe9` (documentary).

> **[RE-DERIVED, TASK 84, 2026-10-23 — MOVED: magnitudes.** Basis: uncapped baseline. `hydro_flat` ~~24.9866~~ 23.3231 bn, `hydro_committed` ~~24.1291~~ 22.5841, `hydro_ndc` ~~11.3356~~ 10.8847; the `ndc` gap ~~13.651~~ **12.4384 bn**, a ~~54.6%~~ **53.3%** fall. Backcast against observed 2024 gas generation: `hydro_flat` 24.788 TWh_e (-3.25%, unchanged); `hydro_committed` and `hydro_ndc` 23.458 TWh_e (-8.44%, unchanged). Conclusion holds: the solar-dominant pathway is contingent on hydro not expanding. Source: `results/baseline_sliced/summary__622bde4e.json`, `results/baseline_sliced/summary__b1621370.json`, `results/baseline_sliced/summary__661b03c1.json`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; conclusion holds.** Post-epoch = `results/baseline_sliced/` Phase B (2026-10-19/20), capital-charge window `arrears`, storage capex `declining_nrel`; post-epoch hash = the pre-epoch arm tag plus the arms TASK 53 §2.5 completed. `hydro_flat` ~~27.3177~~ 24.9866 bn, `hydro_committed` ~~26.3894~~ 24.1291, `hydro_ndc` ~~12.6027~~ 11.3356; the `ndc` gap ~~14.715~~ **13.651 bn**, a ~~53.9%~~ **54.6%** fall. Backcast against observed 2024 gas generation unchanged (−3.25% `hydro_flat`; −8.44% `hydro_committed` and `hydro_ndc`). The +7,411 MW / 1,482 MW per year / 59% of ECN exploitable arithmetic is input-side and was not re-derived. Source: `results/baseline_sliced/summary__{6046f759,0da84099,61f36814}.json`.]**

> **[EPOCH, TASK 43, 2026-10-10. Frozen-state figures. Current epoch: `hydro_flat` 27.3177 bn, `hydro_committed` 26.3894 bn, `hydro_ndc` 12.6027 bn; the `ndc` gap is 14.715 bn, a 53.9% fall (frozen state 14.86 bn, 53%).]**

> **[RE-DERIVED, TASK 44, 2026-10-10 — moved; conclusion holds. `hydro_ndc` 12.6027 bn against `hydro_flat` 27.3177: a 14.715 bn gap, a 53.9% fall (frozen 14.86 bn, 53%); `hydro_committed` 26.3894 bn. The calibration note holds: `hydro_flat` backcast is −3.25% against observed 2024 gas generation, `hydro_committed` and `hydro_ndc` −8.44%.]**

    hydro_flat       27.954 bn
    hydro_committed  27.013 bn
    hydro_ndc        13.093 bn

Under NDC-scale hydro, system cost falls **53%** — a 14.86 bn gap,
against 12.2 bn pre-O&M. **If hydro can expand to NDC scale, solar is
not the residual and the headline result changes character.**

The defence must be made on deliverability, not modelling: `hydro_ndc`
requires +7,411 MW over five years = 1,482 MW/yr = 2.1
Zungeru-equivalents per year, against 8–12 year FID-to-commissioning for
large hydro and nothing at that scale currently under construction. It
passes the resource test (59% of ECN exploitable) and fails lead time.
NIRP passes lead time more easily and fails the resource test (F-16).
Different failure modes; state both.

CALIBRATION NOTE: only `hydro_flat` reproduces the 2024 backcast
(−3.2%). The others depart from the observation by construction.

## F-19 — Storage magnitude is not a cost optimum
**STRUCTURAL. Unchanged, and now demonstrated against a priced asset.**

**Source (pre-epoch, TASK 54 Phase 1).** `results/baseline_sliced/summary__{bbe7d77a storage_cc_low, 857701d1 storage_cc_high, 02c17f55}.json`; runner `01b_run_baseline_sliced.py`, 2026-10-03, `2b22fe9` (documentary). **Provenance NOT established** for the 9,694 USD/MWh-yr storage O&M rate.

> **[RE-DERIVED, TASK 84, 2026-10-23 — MOVED: storage holding cost and capacity-credit arms.** Basis: uncapped baseline. Storage holding cost ~~611~~ **546 M USD/yr** (9,694 x 56,289.88 MWh; the 9,694 rate untraced). Financing span ~~3.273~~ **2.675 bn**; storage unchanged across it except `cr_comm_high` (-120.35 MWh, -0.21%; F-3). Capacity-credit arms (2054 storage MWh / cost bn): `storage_cc_low` ~~95,092.3 / 26.5925~~ **87,209.1 / 24.4898**, `storage_cc_central` ~~63,056.3 / 24.9866~~ 56,289.9 / 23.3231, `storage_cc_high` ~~50,445.0 / 24.3538~~ **45,031.9 / 22.9251**. The ordering (low credit -> most storage and highest cost) holds. The magnitude claim (storage is not an optimisation result) is untouched. Source: `results/baseline_sliced/summary__05a86e43.json`, `results/baseline_sliced/summary__3abc5898.json`, `results/baseline_sliced/summary__661b03c1.json`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; conclusion holds; regime question reported under Tier C.** Post-epoch = `results/baseline_sliced/` Phase B (2026-10-19/20), capital-charge window `arrears`, storage capex `declining_nrel`; post-epoch hash = the pre-epoch arm tag plus the arms TASK 53 §2.5 completed. Storage holding cost ~~611~~ **611 M USD/yr** (9,694 × 63,056.3 MWh; storage unchanged; the 9,694 rate is untraced, TASK 54 Phase 1); the financing span is ~~3.811~~ **3.273 bn** and storage is unchanged across it (max |Δ| 0.0000 MWh). Capacity-credit arms (2054 storage MWh / cost bn): `storage_cc_low` ~~18,978.5~~ **95,092.3** (~~28.3555~~ 26.5925), `storage_cc_central` 63,056.3 (~~27.3177~~ 24.9866), `storage_cc_high` 50,445.0 (~~26.3738~~ 24.3538). **The credit ordering is now monotone** (lower credit, more storage; before, `low` held the least storage), which supports reading the capacity credit as a cause of the level. The central point is unchanged; the `low` arm moved by +76,114 MWh with the storage capex series. **Partition (see F-3):** the firm-capacity constraint binds in the terminal year in all 38 arms and in the same number of years before and after the epoch (20 in each of the three credit arms), so a binding-or-not partition does not separate `storage_cc_low` from the arms that leave storage fixed; the credit arms change the credit, and financing arms change neither the requirement, the credit nor any firm option's cost. Source: `results/baseline_sliced/summary__{9035ee72,6046f759,dedbe8a1}.json`.]**

> **[RE-DERIVED, TASK 44, 2026-10-10 — moved; conclusion holds. Storage holding cost ~611 M USD/yr (9,694 × 63,056.3 MWh; was 636 M); the financing span is 3.811 bn (was 3.97) and storage is still unchanged across it (max |Δ| 7e-5 MWh). Storage moves under the credit arms, which isolates the capacity credit as the cause: `storage_cc_low` 18,978.5 MWh (28.3555 bn), `storage_cc_central` 63,056.3, `storage_cc_high` 50,445.0 (26.3738 bn).]**

Once gas reaches its floor, storage absorbs the entire firm-capacity
residual, so its quantity is insensitive to further financial variation
(F-3). Root cause is the flat `capacity_credit['storage'] = 1.0`
(register L5): with no declining ELCC, storage's marginal firm
contribution never falls and no interior optimum exists.

**O&M sharpens this rather than resolving it.** Storage now carries
~$636M/yr of holding cost and the quantity still does not move across a
3.97 bn financing span. That rules out "storage looked free" as the
explanation and isolates the capacity credit as the cause.

The COMPOSITION claim — gas exits, storage displaces it at 4.0 MWh per
MW, the storage duration — is sound. The MAGNITUDE claim is not. Do not
report storage magnitude as an optimisation result.

## F-20 — RESOLVED: the retirement inversion was an O&M artefact
**The caveat is WITHDRAWN, not explained.**

**Source (pre-epoch, TASK 54 Phase 1).** `results/baseline_sliced/summary__{bf2805fc gas_retire_none, fb66cb38 vintage_50, 977b534a vintage_40, 02c17f55}.json`; runner `01b_run_baseline_sliced.py`, 2026-10-03, `2b22fe9` (documentary).

> **[RE-DERIVED, TASK 84, 2026-10-23 — REVERSED: the withdrawal of the retirement inversion: `gas_retire_vintage_40` is cheaper than `gas_retire_vintage_50` again.** Basis: uncapped baseline. Cost (bn) / 2054 storage (MWh): `gas_retire_none` ~~23.0379 / 45,196.2~~ **22.2846 / 29,895.2**; `vintage_50` ~~24.2957 / 60,260.2~~ **22.8144 / 44,218.3**; `vintage_40` ~~24.6785 / 63,056.3~~ **22.6851 / 56,289.9**; `vintage_30` (central) ~~24.9866~~ 23.3231 / 56,289.9; `proxy` 24.2368 (unasserted). **Not monotone: 22.2846 < 22.6851 (v40) < 22.8144 (v50) < 23.3231.** Retiring at 40 years costs 0.1293 bn LESS than retiring at 50 although it retires more; the ordering the entry certified as monotone (and used to withdraw the 'timing effect, not a level effect' caveat) is inverted between v40 and v50. Range `none`->`v30` ~~1.949~~ **1.0385 bn**. The entry would have to say: the inversion is not an O&M artefact; it is absent in the blended epoch and present in the vintage-dispatch model. It has not been investigated here (discovery closed); it is a register item for the writing step. Source: `results/baseline_sliced/summary__6ddbd8fa.json`, `results/baseline_sliced/summary__3852cdf3.json`, `results/baseline_sliced/summary__7ba9c449.json`, `results/baseline_sliced/summary__661b03c1.json`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; conclusion holds.** Post-epoch = `results/baseline_sliced/` Phase B (2026-10-19/20), capital-charge window `arrears`, storage capex `declining_nrel`; post-epoch hash = the pre-epoch arm tag plus the arms TASK 53 §2.5 completed. Cost (bn) / 2054 storage (MWh): `gas_retire_none` ~~24.5777~~ 23.0379 / 45,196.2 (unchanged); `vintage_50` ~~26.5454~~ 24.2957 / ~~58,257.7~~ 60,260.2; `vintage_40` ~~27.0487~~ 24.6785 / 63,056.3; `vintage_30` (central) ~~27.3177~~ 24.9866 / 63,056.3. Monotone: 23.0379 < 24.2957 < 24.6785 < 24.9866, so the withdrawal of the inversion caveat stands; range ~~2.740~~ **1.949 bn**. Source: `results/baseline_sliced/summary__{4d9c3881,ab657cf1,37e92afb,6046f759}.json`.]**

> **[EPOCH, TASK 43, 2026-10-10. Frozen-state figures. Current epoch (cost / 2054 storage): `gas_retire_none` 24.5777 bn / 45,196.2 MWh; `vintage_50` 26.5454 / 58,257.7; `vintage_40` 27.0487 / 63,056.3; `vintage_30` (central) 27.3177 / 63,056.3; range 2.740 bn; the ordering is still monotone, so the withdrawal stands.]**

> **[RE-DERIVED, TASK 44, 2026-10-10 — holds. Monotone at the current epoch (24.5777 < 26.5454 < 27.0487 < 27.3177 bn); range 2.740 bn (was 2.76).]**

    gas_retire_none        25.199 bn   storage 47,853.1 MWh
    gas_retire_vintage_50  27.184 bn   storage 60,534.7 MWh
    gas_retire_vintage_40  27.715 bn   storage 65,613.0 MWh
    gas_retire_vintage_30  27.954 bn   storage 65,613.0 MWh  (central)

Range 2.76 bn.

Pre-O&M the ordering was non-monotone — `vintage_40` cost MORE than
`vintage_30` despite retiring less — and was flagged as "a timing
effect, not a level effect. Worth understanding before it is reported."

**Under O&M the ordering is monotone and the anomaly is gone.** Retiring
plant now avoids a recurring capacity charge, not only a fuel cost, so
retiring more is unambiguously cheaper. The inversion was an artefact of
the missing cost term.

An apparent result that turned out to be a defect signature. That
belongs in the discussion chapter alongside the calibration journey.

Register L8 still records the vintage schedule as an ANALYST ESTIMATE,
not traced to NERC, NDPHC, TCN or company filings, with confidence by
capacity 35.8% high / 47.2% medium / 17.0% low.

## F-22 — EaaS reaches the same optimum on 55.6–57.9% less public capital.
~~F-22 — EaaS reaches the same system optimum on at least 60% less public capital; the derived B* is self-consistent~~
**(SUPERSEDED title, corrected 2026-10-10, TASK 45: "at least 60%" is the frozen-state figure. At the current epoch the EaaS regime reaches the same plan on 5.14–5.65 bn USD less public capital than the traditional regime's bracket of realised public spend, 9,246,009,182–9,761,349,093 — 55.6–57.9%. The B\* self-consistency clause is withdrawn, see below.)**

The qualitative conclusion is unchanged: the EaaS regime reaches the same plan — same system cost, same total solar, same storage — on substantially less public capital than the public-only regime requires. The composition of that plan between public and EaaS finance is not identified (F-12) and is not reported.

> **The B\* self-consistency claim is withdrawn (TASK 45).** B\* was derived as the public envelope at which the traditional regime's budget constraint ceases to bind, and this entry claimed the derived value reproduces that property. At the current epoch it does not: at 1.00 × B\* = 10,273,343,610 USD the traditional regime leaves 511,994,517 USD unspent, 4.98% of the envelope. B\* overstates the traditional regime's requirement by about five per cent. Every comparison in this entry is therefore made against the measured bracket of realised public spend rather than against B\*. Whether B\* should be redefined at the envelope where the dual first reaches zero, or retained as a reference scale with this offset stated, is an open Chapter 3 decision.

**Frontier B, RE-RUN 2026-09-20 at the derived B\* (10,273,343,610). Supersedes the first run, which was solved on the old coded B\* (9,104,000,000) grid. A VERIFICATION-grade capital-requirement comparison, not a re-estimate of the central.**

> **[SUPERSEDED VALUES, TASK 43, 2026-10-10. The Result block below (27.954 bn, 41,031 MW, 65,613 MWh, 10,273,343,610 required by the traditional regime, 6,164,006,166 less / 60.0–65.0%, 0.35 × B\* at dual 0.1078 and cost 27,979,226,013) is the pre-rate-coherence run. Current epoch, `results/capital_frontier/capital_frontier_raw.csv` (2026-10-03): both regimes reach the identical optimum 27.318 bn (27,317,670,155), 39,030.07 MW solar, 63,056.31 MWh storage. Traditional: last binding 0.9 × B\* (realised public spend 9,246,009,182.24, dual 0.10102134), first slack 1.0 × B\* (9,761,349,093.39, dual 0). EaaS: requirement in (3,595,670,264, 4,109,337,444]: 0.35 × B\* has dual 0.079159 and cost 27,329,694,921.81, 0.40 × B\* has dual 3.5e-16 (zero to tolerance). EaaS needs 5.14–5.65 bn less public capital, i.e. 55.6–57.9% less than the traditional bracket (as in `docs/bstar_triangulation.md` and §3.4). The EaaS realised-spend plateau (8,229,884,146 below) is now 7,777,949,797.19 at the slack points (still UNIDENTIFIED, split-derived). The self-consistency of B\* itself (envelope consumed exactly at 1.00 × B\*) was not re-derived; at the current epoch the traditional regime is slack at 1.0 × B\*.]**

> **[RE-DERIVED, TASK 44, 2026-10-10 — FLIPPED in two claims; needs new prose.** (1) The headline "at least 60% less public capital" is not supported: the traditional requirement is the bracket 9,246,009,182 (last binding) to 9,761,349,093 (first slack) of realised spend and the EaaS requirement lies in (3,595,670,264, 4,109,337,444], i.e. 55.6–57.9% less (5.14–5.65 bn). The qualitative conclusion (EaaS reaches the same optimum on roughly 56–58% less public capital) holds. (2) "The derived B\* is self-consistent" is not supported: at 1.00 × B\* (10,273,343,610) the traditional regime is *slack* (dual 0) with 511,994,517 USD (4.98%) of the envelope unspent; the tier named for the requirement now exceeds the requirement. **Moved, conclusions holding:** `cum_selfgen` is non-zero at 14 of 38 solves, not 16 (traditional 605.2 TWh at a zero envelope, 127.5 at 0.40 × B\*, 0.0 at 0.70; EaaS 30.6 / 3.2 / 0.0 at 0 / 0.05 / 0.10); the envelope-zero EaaS value is 44.02 bn (44,018,930,620), not 46.5; the flagged "dominated points" are nine (EaaS arm, fractions 0.45, 0.5, 0.6, 0.7, 0.85, 0.9, 1.0, 1.1, 1.2), each within 7.33 USD of its dominator (was six within 3.78), the degeneracy reading unchanged. **Holds:** the EaaS arm is degenerate above 0.40 × B\* (binding with dual ≤ 3.5e-16 from 0.40 to 0.70, slack from 0.85); a binding traditional envelope costs 484.5 MW of solar (1.2%) and 12,054.4 MWh of storage (19.1%), sixteen times harder in proportion. Sources: `capital_frontier_raw.csv`, `capital_frontier_eaas_gap.csv`, `capital_frontier_dominated_points.csv`, 2026-10-03.]**

**Source.** `28_run_capital_frontier.py`, second execution (post-B\*),
2026-09-20; 38 solves, all optimal, 0 failures; uncapped central
configuration (`carbon_case="no_policy"`, no `emissions_cap_by_year`,
`solar_build_case="deployment_unconstrained"`), sliced builder,
`om_central`, both financing regimes (traditional and EaaS); arms per
`results/capital_frontier_run.log`; this re-run's log
`results/capital_frontier_postbstar.log`.
Raw: `results/capital_frontier/capital_frontier_raw.csv`. Endpoint
reproduces frozen central: traditional unconstrained `npv_cost` =
27,954,017,978 USD (CSV 27,954,017,977.68), matching the 27.954 bn
anchor.

> **[RE-DERIVED, TASK 84, 2026-10-23 — MOVED: requirement brackets, the percentage saved, the B* overstatement; the 'sixteen times harder' clause.** Basis: uncapped (no cap); public budget varied. Vintage `capital_frontier_raw.csv` (38 solves, all optimal; bracket definition as in the TASK 55 block: last point with a positive dual to first point with a zero dual). Both regimes reach the identical optimum ~~24.987 bn, 39,030.07 MW solar, 63,056.31 MWh storage~~ **23.3231 bn (23,323,068,34x), 30,938.84 MW solar, 56,289.88 MWh storage**. Traditional: last positive dual 0.50 x B* (spend 5,136,671,781, dual 0.39919, cost 23.4011 bn), first zero dual 0.60 x B* (spend 5,518,409,480); bracket ~~7,191,340,475-8,723,252,062~~ **5,136,671,781-5,518,409,480**. EaaS: last positive dual 0.15 x B* (spend 1,541,001,535, dual 0.75958), zero dual at 0.20 x B* (2,054,668,719); requirement in ~~(2,568,335,892, 3,082,003,079]~~ **(1,541,001,535, 2,054,668,719]**. EaaS needs ~~4.11-5.64~~ **3.08-3.46 bn** less public capital, ~~57.1-64.7%~~ **60.0-62.8%** less than the traditional bracket. **B* self-consistency withdrawal stands and strengthens:** at 1.00 x B* (10,273,343,610) the traditional regime is slack with ~~1,550,091,548 USD (15.09%)~~ **4,754,934,130 USD (46.28%)** unspent, so B* overstates the traditional requirement by about 46% (was 15%). `cum_selfgen` non-zero at ~~14~~ **8** of 38 solves (all traditional, 0 to 0.35 x B*; traditional 352.4 TWh at a zero envelope, ~~605.2~~; EaaS 0.0 at a zero envelope, ~~29.2~~); envelope-zero EaaS value ~~44.34~~ **25.17 bn**; dominated points ~~13~~ **18** (traditional 6 at 0.70-1.2 x B*, EaaS 12), each within 8.3 USD of its dominator. EaaS realised-spend plateau 4,369,116,604 (split-derived, unidentified). A binding public budget costs the traditional regime at 0.50 x B* 1,540.7 MW of solar (5.0%) and 12,284.7 MWh of storage (21.8%), the EaaS regime at 0.15 x B* 1,841.4 MW (6.0%) and 17,894.2 MWh (31.8%): the 'sixteen times harder in proportion' clause (traditional against EaaS) does not hold; at its last binding point the EaaS regime is the harder hit. The headline (EaaS reaches the same plan on roughly 60% less public capital) holds, at 60.0-62.8%. Source: `results/capital_frontier/arms_1d68be49/capital_frontier_raw.csv`, `results/capital_frontier/arms_1d68be49/capital_frontier_dominated_points.csv`, `results/capital_frontier/arms_1d68be49/capital_frontier_eaas_gap.csv`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; conclusion holds, the numbers and one claim move.** Post-epoch `results/capital_frontier/arms_92390a63/capital_frontier_raw.csv` (38 solves, all optimal; realised public spend is now `realised_public_spend_pv_usd`). **Definition used for every bracket in this entry, pre-epoch and post-epoch: the requirement of a regime lies between its last point with a positive budget dual and its first point with a zero dual.** A point classified `binding` by the runner's 1% slack tolerance but with a zero dual and a cost equal to the optimum is not binding in any economic sense (post-epoch the traditional 0.85 × B\* point is such a point), so the runner's binding/slack label is not used to define the bracket. That this definition reproduces the pre-epoch bracket exactly (9,246,009,182–9,761,349,093) is corroboration, not the reason for adopting it. Both regimes reach the identical optimum **~~27.318~~ 24.987 bn (24,986,620,95x), 39,030.07 MW solar, 63,056.31 MWh storage**. Traditional: last positive dual 0.7 × B\* (spend 7,191,340,475, dual 1.0193, cost 25.966 bn), first zero dual 0.85 × B\* (spend 8,723,252,062); bracket ~~9,246,009,182–9,761,349,093~~ **7,191,340,475–8,723,252,062**. EaaS: last positive dual 0.25 × B\* (spend 2,568,335,892, dual 0.38898), zero dual at 0.30 × B\* (3,082,003,079); requirement in ~~(3,595,670,264, 4,109,337,444]~~ **(2,568,335,892, 3,082,003,079]**. EaaS needs ~~5.14–5.65~~ **4.11–5.64 bn** less public capital, ~~55.6–57.9%~~ **57.1–64.7%** less than the traditional bracket. **The B\* self-consistency withdrawal stands and strengthens:** at 1.00 × B\* (10,273,343,610) the traditional regime is slack (dual 0) with ~~511,994,517 USD (4.98%)~~ **1,550,091,548 USD (15.09%)** of the envelope unspent, so B\* overstates the requirement by about 15% (was about 5%). `cum_selfgen` is non-zero at 14 of 38 solves (unchanged count; traditional 13, EaaS 1: 605.2 TWh at a zero envelope, unchanged; EaaS ~~30.6~~ 29.2 TWh at a zero envelope, 0.0 from 0.05 × B\*); envelope-zero EaaS value ~~44.02~~ **44.34 bn**; the flagged dominated points are now 13 (EaaS arm 0.45, 0.5, 0.6, 0.7, 0.85, 0.9, 1.0, 1.1, 1.2; traditional 0.85, 0.9, 1.1, 1.2), each within 5.91 USD of its dominator (was nine within 7.33); the EaaS realised-spend plateau ~~7,777,949,797~~ 6,769,458,079 (still split-derived and unidentified, F-12). **Capital scarcity** (storage against solar, at the last positive-dual traditional point 0.7 × B\*): solar −7.4%, storage −49.1%, ratio **6.7** (~~"sixteen times"~~); the pre-epoch 0.9 × B\* point (−1.2% / −19.1%) is no longer binding. The EaaS arm is degenerate from 0.30 × B\* (binding with dual ≤ 2e-16 from 0.30 to 0.60, slack from 0.70; ~~0.40/0.85~~). Source: `results/capital_frontier/arms_92390a63/{capital_frontier_raw.csv,capital_frontier_eaas_gap.csv,capital_frontier_dominated_points.csv}`.]**

**Result.** Both financing regimes reach the identical system optimum —
27.954 bn, 41,031 MW solar, 65,613 MWh storage — and differ only in the
public capital required to reach it:

| regime | public capital to reach 27.954 bn (USD) |
|---|---|
| traditional | 10,273,343,610 (1.00 × B\*: envelope 10,273,343,610, dual 0.0, `binding`; `slack` from 1.10 × B\*) |
| eaas | at most 4,109,337,444 (0.40 × B\*, dual 0.0); the requirement lies in (3,595,670,264, 4,109,337,444], because 0.35 × B\* is not at the optimum (dual 0.1078, cost 27,979,226,013) |

**EaaS reaches the same optimum on at least 6,164,006,166 less public
capital — at least 60.0%, and at most 65.0%** (10,273,343,610 −
3,595,670,264 = 6,677,673,346). The sweep resolution fixes only the
interval; **60.0% is a LOWER bound**, since the EaaS requirement is at
most the 0.40 × B\* grid point. This is the frontier form of EaaS's
causal content in the LP (§3.5): EaaS solar CAPEX does not consume the
public envelope.

**B\* is now self-consistent.** At `adequacy` (1.00 × B\*) the
traditional envelope is exactly consumed — realised spend
10,273,343,610.15, dual 0.0 — and goes slack above it. Under the
superseded constant (first run, envelope 9,104,000,000) the same tier
bound at a dual of 0.21469358, and did not go slack until 1.20 × the
coded value; the derived requirement exceeded it by 12.84% (register
L45). The tier name is now earned by the model's own dual rather than
asserted.

**Unserved energy is zero by construction, not by result.** Under a
priced genset case `unserved` is fixed at zero
(`optimize_model_sliced.py:591-592`, `optimize_model.py:455-456`), so
`cum_unserved` = 0 carries no information about reliability. The
grid-service shortfall in this model is `selfgen`, and `cum_selfgen` is
non-zero at 16 of the 38 Frontier B solves — traditional 638.7 TWh at a
zero envelope, 162.2 TWh at 0.40 × B\*, 1.4 TWh at 0.70 × B\*; EaaS 36.4 /
5.5 / 0.3 TWh at 0 / 0.05 / 0.10 × B\*. At scarce envelopes the frontier's
cost differences therefore include a substantial diesel-backstop component
— degraded grid service, not fuel-and-technology substitution alone. Above
roughly 0.85 × B\* self-generation is effectively zero and the
substitution reading holds.

**Two figures not to be conflated.** The EaaS *requirement* is at most
4.109 bn. 8,229,884,146 (8.230 bn) is where EaaS realised spend *stops
rising* (identical to the digit from 0.85 × B\* upward). Only the first
is a requirement. **[UNIDENTIFIED, TASK 30 Part D: 8.230 bn is realised public spend (public solar
and storage plus the subsidy), a split-derived quantity, taken at cells where the
EaaS budget dual is zero (F-41: dual 1e-16 to 1e-18 at 0.4-0.7, 0 above); where
the budget is slack the split sits on a flat face (F-12), so the plateau is the
solver's vertex, not a property of the model.]** **The second is an observed plateau, NOT verified as
the maximum EaaS can absorb**: it is the realised spend of the solver's
chosen solution at the unconstrained optimum, and it includes public
solar and storage charges as well as the subsidy. It has not been
reconciled against the model's total financing-gap quantity (Σ
DF[t]·gap[t]·`solar_eaas_add`[t]), because per-year decision variables
are not in the saved outputs. Do not describe it as a ceiling or an
absorption capacity.

**Capital scarcity falls on storage roughly sixteen times harder than
on solar, in proportional terms** (re-verified 2026-09-26 against
`results/capital_frontier/capital_frontier_raw.csv`, unaffected by L60
— this frontier is `carbon_case="no_policy"`, uncapped). At the last
binding traditional envelope (0.9×, realised spend 9,246,009,182)
solar is 38,545.57 MW and storage 51,001.92 MWh; at the first slack
point (1.0×, `adequacy`, 9,761,349,093) they reach 39,030.07 MW and
63,056.31 MWh. A binding envelope therefore costs 484 MW of solar
(1.2%) but 12,054 MWh of storage (19.1%). This is the frontier-level
confirmation of §3.3.5's characterisation of storage as
adequacy-driven and constraint-determined: it is the first thing a
binding capital envelope sheds. (The bracket
[9,246,009,182–9,761,349,093] itself, and the EaaS requirement's
55.6–57.9% reduction against it, are the B\* triangulation's own
figures — `docs/bstar_triangulation.md` — re-verified here as
unchanged, not re-derived.)

**Do not cite the envelope-zero figure (46.5 bn; exact
46,507,628,026.97 in `capital_frontier_eaas_gap.csv`) as EaaS's
value.** Its counterfactual is a state funding zero public capex, under
which traditional builds no solar at all — a scenario no one would
choose. The capital-requirement comparison above (same cost, same
build, different financing) is the defensible statement.

**Scope conditions.**
- **The EaaS arm is degenerate above 0.40 × B\*.** The subsidy has no
  objective cost and is capped at the financing gap, so the optimiser
  absorbs any envelope offered at zero cost. From 0.40 × B\*
  (4,109,337,444) to 0.70 × B\* (7,191,340,527) the runner reports
  `binding` with a dual of 0 (or ~1e-18 float noise) because realised
  spend equals the envelope while marginal value is nil; from 0.85 × B\*
  it reports `slack`. **In the EaaS arm only the dual is meaningful;
  binding/slack is not.**
- **The six "dominated points"** the non-dominated check flags
  (`capital_frontier_dominated_points.csv`, fractions 0.6, 0.85, 0.9,
  1.0, 1.1, 1.2) are BOTH of the following, and neither should be stated
  without the other: (i) degenerate — the subsidy is unpriced, so spend
  can rise at no objective cost; and (ii) cost-identical to within
  solver tolerance — each costs within USD 3.78 of its dominator (at
  most 1.4e-10 relative) while spending more.
- **The zero-envelope EaaS dual is basis-dependent and should not be
  cited.** It differed between the two runs although the primal at that
  point is identical (same `npv_cost`, same solar MW), which is LP dual
  degeneracy or a different optimal basis, not a model change.

**Superseded first run (old-B\* grid, kept only so no figure is
orphaned).** EaaS requirement 4,096,800,000 and saving 6,176,543,610
(60.1%) at fractions of 9,104,000,000; plateau 8,235,480,090; six
dominated points at fractions 0.5, 0.7, 0.9, 1.0, 1.1, 1.2. Those
envelopes are not the same USD values as the fractions in this run.

## F-30 — The gas shadow price collapses to zero under an NDC cap in three of four gas regimes, confirming F-9 from the dual side; in `downside` gas every binding year is worth more than export, whichever export benchmark is used (24 of 30 years with no policy, 17 under NDC-3)
~~F-30 — The gas shadow price collapses to zero under an NDC cap in three of four gas regimes, confirming F-9 from the dual side; in `downside` gas the export benchmark decides how many years the pipe is worth more than export (7 to 16 of 30, no policy)~~ **(SUPERSEDED title, TASK 55, 2026-10-20: the 7-to-16 range is a count of present-value duals against year-t benchmarks and is a lower bound; on the year-t basis the count is 24 (17) for every export benchmark. See the TASK 55 block.)**

> ~~**Original title (SUPERSEDED, TASK 29 Part 3.1):** The gas shadow price
> collapses to zero under an NDC cap, confirming F-9 from the dual side; and
> the case for domestic gas allocation does not turn on the export
> benchmark.~~ Its second clause rested on the annual-builder Result 2
> (bimodality) and does not survive the sliced series; see "Which of F-30's
> results survive" below.

> **[RE-DERIVED, TASK 55, 2026-10-20 — RESTATED: Result 1 unchanged; Result 3 strengthened in substance, with one clause refuted (Zeus's decision: the year-t basis is adopted).** Post-epoch `results/gas1/arms_be9472ef/` (Phase B item 9, 2026-10-19 23:50; 8 solves; `gas_shadow_by_year.csv` now carries both bases). **The correction, in one sentence:** the benchmarks are constant-dollar price levels (2024 vintage: Japan landed LNG spot of January 2024, the NMDPRA 2024 base prices, and an assumed ladder, read as constant real prices) while the stored deliverability dual was a present value in year-0 dollars, so for year *t* it understated the price in year-*t* dollars by 1/DF[t] (1.59 in 2031, 4.66 in 2045, 9.32 in 2054) and the earlier exceedance counts were lower bounds. **Worked check, `downside`, no policy, 2054** (DF[t] = 1.08^−t with t = 29, from the model's 8% real social discount rate; the model is in real USD, the rate is real): stored present-value dual **18,923,029 USD/TWh_th** (5.55 USD/MMBtu in year-0 dollars) ÷ DF **0.107328** = **176,311,063 USD/TWh_th** = **51.67 USD/MMBtu** in year-2054 constant dollars, against benchmarks of 5.15 to 10.00 USD/MMBtu (the highest, delivered LNG, 10.00, DES central 6.49, HUB central 5.79, opp_7 7.00) in 2024 dollars: 5.2 times the highest. The conversion is exact (the stored year-t value equals PV ÷ DF to the last digit). **The claim does not rest on 2054:** the smallest year-t dual among the 24 binding years is **15.71 USD/MMBtu (2039)**, above the highest benchmark; and 2046-2054 is the end-effect buffer, so within the reporting window (2025-2045, 21 years) the count is **15 of 21** (no policy) and **8 of 21** (NDC-3) against every export benchmark on the year-t basis. The benchmarks are single-vintage 2024 prices, not year-of-delivery prices; one year of 2% inflation to a 2025 base would move them by 2%, which no count here is sensitive to.

**Result 1 — UNCHANGED.** The cap binds before gas deliverability does: shadow exactly zero under NDC-3 unconditional in baseline, upside and shock_recovery (0 binding years); binding years `downside` no policy 24 (2031-2054), NDC-3 17 (2038-2054), baseline 23, upside 22, shock_recovery 24 (all unchanged).

| binding-year mean, USD/TWh_th (USD/MMBtu) | pre-epoch (PV) | post-epoch **PV** | post-epoch **year-t** |
|---|---|---|---|
| no policy, baseline | 3,466,259 (1.02) | 2,006,829 (0.59) | 5,524,310 (1.62) |
| no policy, upside | 1,933,112 (0.57) | 1,467,693 (0.43) | 4,676,005 (1.37) |
| no policy, downside | 30,291,626 (8.88) | 29,204,054 (8.56) | 98,393,547 (28.84) |
| no policy, shock_recovery | 10,728,228 (3.14) | 11,502,713 (3.37) | 25,903,799 (7.59) |
| NDC-3, downside | 19,025,432 (5.58) | 17,806,276 (5.22) | 96,379,999 (28.25) |

**Result 3 — strengthened in substance; second clause refuted.** Years of 30 with the dual above the benchmark (strict), for the six export benchmarks in the order opp_7 (7.00), DES central (6.49), HUB central (5.79), delivered LNG (10.00), DES low (5.85), HUB low (5.15 USD/MMBtu):

| cell | pre-epoch (PV) | post-epoch **PV** | post-epoch **year-t** |
|---|---|---|---|
| no policy, `downside` | 8 / 8 / 12 / 7 / 11 / 16 | 7 / 8 / 10 / 7 / 10 / 13 | **24 / 24 / 24 / 24 / 24 / 24** |
| NDC-3, `downside` | 1 / 1 / 7 / 0 / 7 / 10 | 1 / 1 / 4 / 0 / 3 / 9 | **17 / 17 / 17 / 17 / 17 / 17** |
| no policy, `shock_recovery` | 3 / 3 / 3 / 3 / 3 / 4 | 4 / 4 / 4 / 4 / 4 / 4 | 5 / 5 / 5 / 5 / 5 / 5 |
| no policy, baseline and upside | 0 | 0 | 0 (one year above HUB low, baseline) |

*First clause, strengthened:* on the year-t basis **every binding year in `downside` exceeds every export benchmark** — 24 of 30 years with no policy, 17 of 30 under NDC-3 (15 of 21 and 8 of 21 inside the reporting window). *Second clause, refuted and struck:* ~~the export benchmark decides how many years~~ (7 to 16 of 30 under no policy, 0 to 10 under NDC-3): the count is identical across all six benchmarks, so the choice of benchmark decides nothing. The earlier counts in the "Which of F-30's results survive" table and in Result 3 below are the present-value counts and are retained as the record, not as the finding. Source: `results/gas1/arms_be9472ef/{gas_shadow_by_year.csv,gas1_summary.json}`; arithmetic from `src/gas_benchmarks.py`, `src/optimize_model_sliced.py` (DF) and `src/optimize_experiments.py` (`extract_gas_shadow_yr_usd_per_twh_th`).]**

**GAS-1, Phase B item 9 re-measurement (2026-10-02): SLICED builder at `--conversion-case conv_central`, capped solar design, 8 solves. Conversion is represented.**

> **[SCOPE CONDITION SUPERSEDED, TASK 48 B.4, 2026-10-18.]** This entry's header and Source paragraph formerly said "annual builder, so conversion is not represented" and `conv_none`. That described the 2026-09-20 measurement, which the tables below the "RE-MEASURED" block retain for the record. The current table is the Phase B item 9 re-measurement (`logs/phase_b/09_gas1_shadow_benchmarks.log`, invocation line: `--conversion-case conv_central`; `results/gas1/gas1_summary.json`, 2026-10-02 23:18). Established by TASK 47 C4. A Phase C uncapped-design twin exists (`results/gas1_uncapped/`, 2026-10-10); it is not what this entry reports.

**Which of F-30's results survive the sliced series (TASK 29 Part 3.1;
evidence: TASK 28 Part B, stored per-year duals, no solve).**

| result | status at the sliced state |
|---|---|
| **Result 1** — the cap binds before gas deliverability does; the shadow price is exactly zero under NDC-3 unconditional in baseline, upside and shock_recovery | **SURVIVES** (re-measured on the sliced builder, Phase B item 9; 17 binding years in `downside`, 0 elsewhere) |
| "confirms F-9 from the dual side" | **SURVIVES** (it is Result 1 read from the dual) |
| **Result 2** — "the distribution is bimodal: 18 years below 1.26, 12 at 7.11 or above, median 0.767" | **DOES NOT SURVIVE.** Baseline gas, no policy: median **0.49** USD/MMBtu, all-year mean 0.78, binding-year mean 1.02; 27 of 30 years at or below 1.02, then 3.30-4.33; not bimodal. Redone in the block below. The warning "do not quote the mean alone" survives for a different reason (a few high years dominate the mean in `downside` and `shock_recovery`) |
| **Result 3** — "the conclusion is insensitive to the export benchmark" | **DOES NOT SURVIVE for `downside`**: years exceeding the benchmark range **7 to 16 of 30** under no policy (23%-53%) and **0 to 10** under NDC-3 depending on the benchmark (HUB-central 12 / 7 years against DES-central 8 / 1). It holds only where it is trivially true: baseline and upside (zero exceedance against every benchmark), and, under NDC-3, all three non-downside regimes. `shock_recovery` under no policy exceeds in 3 years against four benchmarks and 4 against one: nearly insensitive, not zero |
| the argument that the interim-sourced delivered-LNG price "does not propagate into the finding" | **WITHDRAWN** (it rested on the bimodality) |

**RE-MEASURED, Phase B item 9 (8 solves, SLICED builder, `gas_avail_nameplate`
and standard arms; `results/gas1/gas1_summary.json`, `gas_shadow_by_year.csv`,
run 2026-10-02).** The tables and prose below this block are the
2026-09-20 **annual-builder** measurement, retained for the record; the
re-measurement supersedes their figures. The qualitative result is
unchanged and now holds on both builders.

Unit: the field is `gas_shadow_usd_per_twh_th` — USD per **thermal** TWh of
gas deliverability, not electrical.

**CORRECTED (TASK 27 A; recomputed from the stored per-year duals in
`results/gas1/gas_shadow_by_year.csv`, no solve; output
`results/gas1/gas_shadow_binding_recomputed.csv`).** The figure to quote is
the **binding-year mean** (mean over the years in which the deliverability
constraint binds, dual > 1e-6). The runner previously reported a single
field, `gas_shadow_mean`, which was a mean over **all 30 evaluable years**
(`06_run_gas1_shadow_benchmarks.py` filtered on `is not None`) and was
therefore diluted by the non-binding years: it equals the sum of binding
duals divided by 30, not by the binding count, so it understates the
binding-year mean by a factor n_binding/30. This is **the same defect class
as TASK 02's** (the reliability dual, fixed in `extract_reliability_dual` by
emitting the binding-year and all-year means as separate fields) **and
register L70 already carries it.** The runner now emits
`gas_shadow_binding_mean`, `gas_shadow_all_year_mean` and
`gas_shadow_binding_years` separately; `gas_shadow_mean` no longer exists.

| policy | gas case | binding yrs /30 | **binding-year mean (USD/TWh_th)** | USD/MMBtu | ~~all-year mean (SUPERSEDED, retained)~~ | understatement |
|---|---|---|---|---|---|---|
| no policy | baseline | 23 | **3,466,259** | 1.02 | ~~2,657,465~~ | 23.3% |
| no policy | upside | 22 | **1,933,112** | 0.57 | ~~1,417,616~~ | 26.7% |
| no policy | downside | 24 | **30,291,626** | 8.88 | ~~24,233,301~~ | 20.0% |
| no policy | shock_recovery | 24 | **10,728,228** | 3.14 | ~~8,582,582~~ | 20.0% |
| NDC-3 unconditional | baseline / upside / shock_recovery | **0** | undefined (no binding year) | — | 0 | — |
| NDC-3 unconditional | downside | **17** | **19,025,432** | 5.58 | ~~10,781,078~~ | 43.3% |

(USD/MMBtu at 3.412e6 MMBtu per TWh_th, the conversion that reproduces the
earlier table's 28,124,906 -> 8.24.) The binding **counts** and the
zero/non-zero pattern were never affected; only the level of the mean.

**The superseded annual-builder values below were also all-year means** of
the same runner (29/30 binding years in the no-policy arms), so their
binding-year equivalents are larger by 30/29; they are retained as
recorded and not recomputed (derived only, per-year duals for that run were
not retained).

**Verification note (TASK 25).** The draft described the 2,657,465 /
1,417,616 / 24,233,301 / 8,582,582 figures as binding-year means. They were
all-year means; the table above gives the binding-year means.

**Reading.** Imposing the NDC cap makes gas deliverability non-binding in
three of four regimes: the emissions ceiling falls below the physical
delivery ceiling, so additional supply capacity has zero marginal value.
Only in `downside` does the pipe remain the lower of the two ceilings.
Against the annual-builder run, the no-policy binding count falls from
29/30 to 22-24/30 and the all-year mean from 28.1M to 2.7M (baseline; both all-year, so like for like); the
builder changed (annual to sliced), and the other differences between the
two runs are not enumerated here.

**Untested — recorded as an open interpretation, NOT asserted.** The
draft proposed two relief channels (solar displacing gas; load leaving
the grid for self-generation). POL-1 reports 42.9% of emissions outside
the cap boundary for `ndc3_unconditional` / public (tight capital;
`results/pol1/pol1_results.csv`), which makes the second channel
plausible, but distinguishing the channels needs the carbon duals and the
gas-burn-against-limit series, which are in `gas_shadow_by_year.csv`
(gas dual only) and the per-run diagnostics and have not been extracted.


**Source (current, TASK 48 B.4).** `06_run_gas1_shadow_benchmarks.py`, Phase B item 9, run 2026-10-02 (`logs/phase_b/09_gas1_shadow_benchmarks.log`; `results/gas1/gas1_summary.json`, `gas_shadow_binding_recomputed.csv`, `gas_benchmark_exceedance_recomputed.csv`), sliced builder, `--conversion-case conv_central`, `capital_case="moderate"` hardcoded (= 0.85 × B\*), NDC-3 unconditional caps, capped solar design. Eight solves.

> **[RE-DERIVED, TASK 84, 2026-10-23 — WEAKENED: Result 3 first clause (every binding `downside` year exceeds every export benchmark) holds only with the capital envelope; cap alone it is 0 to 10 of 30 years and the benchmark decides again.** Basis: cap alone (02_ NDC 3.0) and 06_ moderate-envelope design, both reported. **Result 1 (the cap binds before gas deliverability does): UNCHANGED.** Vintage `results/gas1/arms_f30c5990` (06_; `capital_case` moderate hard-coded, NDC-3 unconditional caps): shadow exactly zero under NDC-3 in baseline, upside and shock_recovery (0 binding years); `downside` NDC-3 17 binding years (unchanged), `downside` no policy ~~24~~ 26, baseline ~~23~~ 23, upside ~~22~~ 14, shock_recovery 24. The same holds on the 02_ results at NDC 3.0 for both capital cases: 0 binding years in baseline/upside/shock_recovery; `downside` binds in 17 years (2038-2054) unconditional and 9 (2046-2054) conditional, cap alone and with the envelope. **Levels (binding-year mean, year-t basis, USD/TWh_th):** no policy baseline ~~5,524,310~~ 9,610,089; upside ~~4,676,005~~ 10,672,013; `downside` ~~98,393,547~~ **36,504,097** (10.70 USD/MMBtu, was 28.84); shock_recovery ~~25,903,799~~ 10,441,161; NDC-3 `downside` ~~96,379,999~~ 44,572,263 (13.06). **Result 3, years of 30 above the benchmark, year-t basis, six export benchmarks (opp_7, DES central, HUB central, delivered LNG 10, DES low, HUB low), vintage `arms_f30c5990`:** no policy `downside` ~~24/24/24/24/24/24~~ **21/22/22/12/22/22**; NDC-3 `downside` ~~17 x6~~ **17/17/17/10/17/17**; `shock_recovery` ~~5 x6~~ 0 against all six; baseline and upside 0. So the 'identical across all benchmarks' clause fails for delivered LNG (12 and 10 years) and, no policy, marginally for opp_7 (21). **Cap alone** (`02_` NDC-3, `capital_case=unconstrained`, `downside`): unconditional 0/1/5/0/5/10 years above the six benchmarks, conditional 0/0/4/0/4/9 (binding-year mean 16,131,050 and 19,524,546 USD/TWh_th, 4.73 and 5.72 USD/MMBtu); with the envelope the same cell is 17 x 6. The earlier refuted clause ('the benchmark decides how many years') is therefore restored for the cap-alone design. The entry's claim concerns a cap, so the cap-alone figures are the ones it requires; the 06_ runner cannot be run cap alone as written (capital case hard-coded). Source: `results/gas1/arms_f30c5990/gas_shadow_by_year.csv`, `results/gas1/arms_f30c5990/benchmark_comparison.csv`, `docs/task_log_83_capped/ndc3_unconditional_noenvelope_capped_arms.csv`, `docs/task_log_82_capped/ndc3_unconditional_capped_arms.csv`. Detail: `docs/task_log_84_part*.md`.]**

**Source (SUPERSEDED, retained for the record).** `06_run_gas1_shadow_benchmarks.py`, run 2026-09-20
(`results/gas1_postbstar.log`; outputs `results/gas1/`), run at
`conv_none` per the run instruction (the arm string is not recorded in
the log) on the annual builder, `capital_case="moderate"` hardcoded
(= 0.85 × B\*), NDC-3 unconditional caps. Eight solves, all optimal, 0
failures. Benchmarks from `src/gas_benchmarks.py`; sourcing in
`docs/sourcing_pack_01_rates.md` (secondary citations, unverified).

**Result 1 — the cap binds before gas deliverability does.**

| NDC scenario | gas case | mean shadow (USD/TWh_th) | USD/MMBtu | binding years |
|---|---|---|---|---|
| no policy | baseline | 28,124,906 | 8.24 | 29/30 |
| no policy | upside | 26,629,441 | 7.80 | 29/30 |
| no policy | downside | 38,533,131 | 11.29 | 29/30 |
| no policy | shock_recovery | 28,129,433 | 8.24 | 29/30 |
| NDC-3 | baseline | 0 | 0 | 0/30 |
| NDC-3 | upside | 0 | 0 | 0/30 |
| NDC-3 | downside | 8,416,344 (was 10,316,417) | 2.47 (was 3.02) | 17/30 (was 18/30) |
| NDC-3 | shock_recovery | 0 | 0 | 0/30 |

**Re-verified post-L60, attributed across two steps** (register L60;
`results/gas1_L60.log` and the intermediate `results/gas1_rerun.log`,
2026-09-25 pre-L60/post-rate-coherence, both against
`results/gas1_postbstar.log`, 2026-09-20 original). The uncapped
no-policy arms shift by under 0.1% (28,124,906 → 28,099,080, baseline)
entirely during the rate-coherence step (L52-L59) — these arms carry no
cap, so L60 cannot and does not move them further. `downside`'s shadow
price moves in two steps, **unlike F-31's leakage table, with no round
trip**: rate-coherence alone, 10,316,417 → 9,985,934, binding years
unchanged at 18/30; **L60 alone, 9,985,934 → 8,416,344, binding years
18/30 → 17/30** — the one-year drop in binding years is attributable to
L60 specifically, not to the upstream rate/demand corrections. **The
tightened cap makes the finding MORE true, not less**: the cap now
exhausts the constraint one year earlier (2037, was 2036), so gas
deliverability goes slack one year sooner. `docs/
task_log_findings_reverify_L60.md` has the full per-year check.

Under an NDC cap the gas shadow price is **exactly zero** and binding
years go from 29/30 to 0/30: the emissions cap binds first and gas
deliverability goes slack. Only `downside` retains a positive shadow —
the crossover where gas supply is scarce enough to bind before the cap
does, and it recedes as the cap tightens. (The three zero-shadow NDC
arms also return the identical NPV cost at the pre-L60 state,
65,404,451,252.30, the same pattern F-9 reports across deliverability
arms; not re-verified post-L60, since the cap value itself changed and
NPV would not reproduce to the digit — see the task log.)

**This independently confirms F-9 from the dual side.** F-9 established
the same conclusion by comparing objectives across deliverability arms;
GAS-1 shows it from the dual of the deliverability constraint. Two
methods, two runners, one conclusion.

**RESULTS 2 AND 3, RE-DONE ON THE SLICED SERIES (TASK 28 Part B;
recomputed from the stored per-year duals `results/gas1/gas_shadow_by_year.csv`
by `scripts/recompute_gas1_distribution_from_stored.py`, no solve; outputs
`results/gas1/gas_distribution_recomputed.csv` and
`gas_benchmark_exceedance_recomputed.csv`; exceedance counts reproduce
`gas1_summary.json` with 0 mismatches).** The two results below this block
are the 2026-09-20 **annual-builder** measurement; they were superseded when
the runner was ported to the sliced builder (`c549bba`) and were not
re-measured until now. They are retained, marked, and **should not be
quoted**.

*Result 2, redone — the distribution* (USD/MMBtu, 30 years, no policy):

| gas case | median | all-year mean | binding-year mean | max | binding yrs | structure |
|---|---|---|---|---|---|---|
| baseline | 0.49 | 0.78 | 1.02 | 4.33 | 23 | 27 years at or below 1.02, then 3.30-4.33; **not bimodal** |
| upside | 0.28 | 0.41 | 0.57 | 2.65 | 22 | no gap structure |
| downside | 5.36 | 7.10 | 8.88 | 21.90 | 24 | 23 years to 7.09, then 13.80-21.90 |
| shock_recovery | 0.58 | 2.52 | 3.14 | 20.28 | 24 | 27 years to 5.28, then 17.38-20.28 (a short spike) |

The superseded annual-builder result — "bimodal: 18 years below 1.2618 and
12 at 7.1065 or above, nothing between; median 0.767 against a mean of
8.235" — **does not hold at the sliced state** for baseline gas (median 0.49,
mean 0.78 all-year / 1.02 binding-year). Under NDC-3 unconditional the three
non-downside regimes are zero in every year; `downside` has 17 binding years
(median 4.66, all-year mean 3.16, binding-year mean 5.58, max 7.50). The
"mean must not be quoted alone" warning stands, but for a different reason:
the mean is dominated by a handful of high years in `downside` and
`shock_recovery`.

*Result 3, redone — exceedance of export benchmarks* (years of 30 with the
deliverability dual above the benchmark; strict):

| | opp_7 (7.00) | DES central (6.49) | HUB central (5.79) | delivered LNG (10.00) | DES low / HUB low (5.85 / 5.15) |
|---|---|---|---|---|---|
| no policy, baseline | 0 | 0 | 0 | 0 | 0 / 0 |
| no policy, upside | 0 | 0 | 0 | 0 | 0 / 0 |
| no policy, **downside** | **8** | **8** | **12** | **7** | **11 / 16** |
| no policy, shock_recovery | 3 | 3 | 3 | 3 | 3 / 4 |
| NDC-3, downside | 1 | 1 | **7** | 0 | **7 / 10** |
| NDC-3, other three | 0 | 0 | 0 | 0 | 0 / 0 |

**The conclusion drawn from the annual result — "the conclusion is
insensitive to the export benchmark" and the second half of this entry's
original title ("the case for domestic gas allocation does not turn on the
export benchmark", struck above) — is NOT supported at the sliced state.** It holds trivially for
baseline and upside gas (zero exceedance against every benchmark). For
**`downside` it fails**: the number of exceeding years ranges 7 to 16 of 30
under no policy (23% to 53%), and 0 to 10 under NDC-3, depending on which
benchmark is used; the HUB basis exceeds in 12 and 7 years where the DES
basis exceeds in 8 and 1. The first half of the title (the shadow price
collapses to zero under an NDC cap, in three of four regimes) is unaffected
and is the title as now written.
The "interim-sourced delivered LNG price does not propagate into the finding"
argument rested on the annual bimodality and is withdrawn.

*Superseded (annual builder, 2026-09-20), retained:*

**Result 2 — the distribution, not the mean. [SUPERSEDED, annual builder; see the redone Result 2 above]** Median 0.767 USD/MMBtu
against a mean of 8.235 (baseline gas, no policy; maximum 29.79) — was
0.769/8.243, superseded by the post-L60 re-verification (rate/demand
correction residual on the uncapped arms, not a cap effect; see above).
The series is bimodal: 18 years below 1.2618 and 12 years at 7.1065 or
above (was 1.27/7.31), with nothing between. **The mean must not be
quoted alone.**

**Result 3 — the conclusion is insensitive to the export benchmark. [SUPERSEDED, annual builder; NOT supported at the sliced state for `downside`; see above]**
Exceedance shares for `baseline_no_policy`, baseline gas
(re-verified post-L60, `docs/task_log_findings_reverify_L60.md` —
counts unchanged despite the gap edges moving, since no benchmark sits
between the old and new edges):

| benchmark | USD/MMBtu | years exceeding |
|---|---|---|
| `opp_7` | 7.00 | 40.0% (12/30) |
| LNG netback, DES central | 6.49 | 40.0% (12/30) |
| LNG netback, HUB central | 5.79 | 40.0% (12/30) |
| `lng_export_10` (delivered) | 10.00 | 36.7% (11/30) |

Three thresholds spanning 5.79–7.00 return **identical** counts, and the
delivered price differs by one year in thirty. Because the distribution
is bimodal, the interim-sourced delivered LNG price does not propagate
into the finding. The netback correction is therefore a correctness fix,
not a result driver.

Thresholds inside the empty gap between 1.2618 and 7.1065 (was 1.27 and
7.31) — which include the netback range 5.79–7.00 — return the full
high mode of 12 years; only thresholds above 7.1065 cut into it, and
each drops a single year (11 of 30; this holds up to 13.80, the next
member of the high mode, unchanged from the superseded run). The
benchmark insensitivity is therefore structural, not coincidental: the
netback range falls in a region of the distribution where no year sits,
and that remains true after the gap narrowed.

**Scope conditions.** *(Superseded in part, TASK 48 B.4: the first sentence described the 2026-09-20 annual-builder measurement; the current table is sliced-builder, `conv_central`, so conversion is represented. The `capital_case` sentence below still applies.)* ~~Annual builder (`conv_none`): conversion, the
second-largest lever, is not representable and storage is not
dispatched.~~ `capital_case` is hardcoded to `moderate` inside
`run_no_policy_gas_matrix`, so every figure here sits under a binding
public capital envelope at 0.85 × B\*. The mean is sensitive to B\*
through that hardcoded envelope; the pre-B\* artefact was not retained,
so the magnitude is not measured. See L48.

## F-31 — Under realistic collection friction a "fully-funded" capital envelope (`adequacy`, 1.00 × B\*, about 5% above the traditional regime's measured requirement) still leaves a fifth of emissions outside the cap
**Qualifier (register L83, option (b), TASK 70 B.4).** "Fully funded" names the `adequacy` tier, 1.00 × B\*, a reference scale; B\* overstates the traditional regime's measured requirement by about 5 per cent, so the tier is not a measured sufficiency and the title's earlier wording, without this qualifier, overstated it. No number in this entry changes. The previous title read: *Under realistic collection friction a fully-funded capital envelope still leaves a fifth of emissions outside the cap*.

**NEW. FIN-3, sliced builder, post-B\* run.**

**Source.** `05_run_fin3_capital_sweep.py` (ported to
`build_model_sliced`), first run 2026-09-20 at the derived B\*, 15 solves, all
optimal, NDC-3 unconditional caps. **Current file:**
`results/fin3/capital_sweep_results.csv`, from the Phase B re-run (**item 23,
2026-10-08, commit `85d820d`**), md5 `26fcae2d…`. **The original 2026-09-20
file** (the 'was' values below; md5 `bb82afb4…`) survives as
`results/_attrib/D_all.csv`, which is **not** equal to the current file (it is
pre-rate-coherence and pre-L60: e.g. `tight`/`no_friction` leakage 42.1% against
42.15%, budget shadow 6.39 against 6.25); the attribution table below reads
`_attrib/`, so it describes that earlier state. (An earlier version of this
line said the results file was identical to `D_all.csv`: that was true on
2026-09-20 and has not been since the L60 rebuild.)

> **[RE-DERIVED, TASK 84, 2026-10-23 — REVERSED: the title's 'a fifth of emissions outside the cap' at a fully-funded envelope (`adequacy`, current collection): 0.06% over the horizon, 0.08% in the window.** Basis: cap plus envelope (envelope varied); the unconstrained row is cap alone. Vintage `capital_sweep_results.csv` (15 solves, NDC-3 unconditional caps, binding in 28 years in every cell). Emissions outside the cap boundary over 2025-2054 (blended in brackets): tight **21.31%** (~~42.5%~~) / **33.10%** (~~47.5%~~) / **38.95%** (~~50.1%~~) at no_friction / reform_target / current collection; moderate **0.06%** (~~10.7%~~) / **0.69%** (~~21.2%~~) / **3.25%** (~~26.9%~~); adequacy **0.10%** (~~7.4%~~) / **0.06%** (~~9.7%~~) / **0.06%** (~~17.6%~~); expansion 0.10% / 0.10% / 0.10% (~~7.4 / 7.4 / 8.2~~); unconstrained 0.10% at every rate (~~7.4%~~). **Reporting window 2025-2045 (recomputed from the annual columns):** tight 3.91% / 5.03% / 8.80%; moderate 0.08% / 0.06% / 0.00%; adequacy 0.14% / 0.08% / 0.08%; expansion and unconstrained 0.14%. **Flags: 3 cells** exceed the runner's 0.10 threshold (all `tight`; ~~7~~), none in the window. The title's 'about a fifth' at a fully-funded envelope and 35% collection is **0.06%**: it does not hold; the leakage is confined to the tight envelope. Carbon shadow, binding-year mean (USD/tCO2): `tight` / `no_friction` ~~211.96 PV, 642.09 year-t~~ **205.16 PV, 734.69 year-t**; `unconstrained` ~~111.46, 222.42~~ **76.93, 174.71**; `adequacy` / `current` ~~180.48, 517.72~~ **72.24, 219.04**. The attribution table (D, C', B') is not re-derivable from this file. Basis: the entry varies the envelope under a cap, so the envelope-carrying cells are the claim; the `unconstrained` row is the cap-alone cell (0.10%). Source: `results/fin3/arms_1d68be49/capital_sweep_results.csv`, `results/fin3/arms_1d68be49/capital_sweep_summary.json`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; the Tier C question (level against increment) is reported in the log, not decided.** Post-epoch `results/fin3/arms_92390a63/capital_sweep_results.csv` (Phase B item 23, 2026-10-20 01:18; 15 solves, all optimal; NDC-3 unconditional caps in **all 15 cells**, binding in 26 years in every cell). Emissions outside the cap boundary (pre-epoch in brackets): tight **42.5%** (~~42.2%~~) / **47.5%** (~~47.5%~~) / **50.1%** (~~50.1%~~); moderate **10.7%** (~~13.5%~~) / **21.2%** (~~24.8%~~) / **26.9%** (~~30.4%~~); adequacy **7.4%** (~~7.4%~~) / **9.7%** (~~12.6%~~) / **17.6%** (~~21.6%~~); expansion 7.4% / 7.4% / **8.2%** (~~8.1%~~); unconstrained 7.4% at every collection rate (unchanged). **Flags:** 7 cells exceed the runner's 0.10 threshold (~~8~~): `adequacy` / `reform_target` (9.67%) is no longer flagged; `moderate` / `no_friction` (10.68%) still is. The title's "a fifth of emissions" at a fully-funded envelope and 35% collection is now 17.6% (~~21.6%~~, "about a sixth"). Carbon shadow, binding-year mean (USD/tCO₂), **both bases** (`carbon_shadow_pv_*` is a present value; `carbon_shadow_yr_*` is the year-t quantity): `tight` / `no_friction` ~~212.11~~ **211.96 PV, 642.09 year-t** (ratio 3.03); `unconstrained` ~~113.45~~ **111.46 PV, 222.42 year-t** (ratio 2.00; the body's 115.05 is the pre-L60 value); `adequacy` / `current` ~~186.54~~ 180.48 PV, 517.72 year-t; across the 15 cells the year-t value is 2.0 to 3.0 times the PV value. This entry's shadows are **PV** values labelled USD/tCO₂, as the task states; the year-t column is the one comparable with a USD/tCO₂ price. Budget shadow `tight` / `no_friction` ~~6.25~~ 6.54, `adequacy` / `current` ~~3.80~~ 4.02. **Not re-derivable:** the attribution table (D, C′, B′) is the one-change-at-a-time pre-rate-coherence set in `results/_attrib/`; no post-epoch counterpart exists and its figures remain historical. **Level against increment:** every FIN-3 cell applies the cap, so 7.4% in the `unconstrained` cell is a cap-induced level at unconstrained capital, not the share in an uncapped arm; see the log for the per-cell counterpart check. Source: `results/fin3/arms_92390a63/capital_sweep_results.csv`.]**

**Result, RE-VERIFIED POST-L60** (register L60/L63; the table below is
the current state — `results/fin3/capital_sweep_results.csv`,
2026-09-26 12:09, **reproduced bit-identically by the Phase B re-run of
2026-10-08 (item 23, commit `85d820d`): all 15 rows and every numeric field
equal, maximum difference 0.0; the files differ only in the carbon-shadow
column names (post-TASK-27 forms)** — compared against the ORIGINAL 2026-09-20 figures
this section published, per this file's own convention of stating the
superseded value). Emissions falling outside the cap boundary, by
capital envelope and DisCo collection rate:

| capital case | no_friction (100%) | reform_target (60%) | current (35%) |
|---|---|---|---|
| tight | 42.2% ⚠ (was 42.1%) | 47.5% ⚠ (was 47.2%) | 50.1% ⚠ (was 50.0%) |
| moderate | 13.5% ⚠ (was 13.7%) | 24.8% ⚠ (was 24.9%) | 30.4% ⚠ (was 30.2%) |
| adequacy | 7.4% (was 6.6%) | 12.6% ⚠ (was 12.7%) | 21.6% ⚠ (was 21.3%) |
| expansion | 7.4% (was 6.6%) | 7.4% (was 6.6%) | 8.1% (was 7.9%) |
| unconstrained | 7.4% (was 6.6%) | 7.4% (was 6.6%) | 7.4% (was 6.6%) |

⚠ = flagged NOT NDC-compliant by the runner's own leakage diagnostic
(`src/optimize_model.py:1760`, threshold 0.10). **Against the originally
published table, no cell's flag status changes** — every ⚠ above was
already a ⚠ in the 2026-09-20 run, and the magnitudes move by at most
0.8 pp.

**That flatness is not the whole story — it hides a round trip.**
Between the original run and now, the rate-coherence corrections
(L52-L59, 2026-09-25) first LOWERED leakage in every cell (e.g.
`moderate/no_friction` 13.7%→9.85%, `adequacy/reform_target`
12.7%→8.95% — both dropped BELOW the 0.10 flag and would have read
compliant had this table been regenerated at that intermediate state,
`results/_pre_L60_20260926_1158/fin3/capital_sweep_results.csv`,
2026-09-26 11:58). The L60 cap rebuild (`7908d09`) then RAISED it back
in every cell, past the flag again in those same two cells
(9.85%→13.53%, 8.95%→12.56%) — landing close to, not exactly at, the
original values. **Register L63 documents this narrower pre-L60 →
post-L60 step in isolation** (holding rate-coherence fixed, varying
only L60), which is where the "two threshold crossings" belong; stated
against the ORIGINAL published table, as above, there are none — the
crossings are visible only against the intermediate state, not against
what this document previously reported. Both comparisons are correct;
they answer different questions, and neither should stand in for the
other. Two qualifications from L63 carry over regardless of which
baseline is used: (1) the `unconstrained` floor also rose across the
L60 step (5.81%→7.37%) with no cap-stringency channel to explain it —
part of the L60-step increase is a level shift, not scarcity; (2)
`expansion/current` remaining unflagged across the L60 step
(5.84%→8.10%) is a property of the declared 0.10 threshold, not
evidence of compliance. ~~Carbon shadow prices rose across every cell
over the L60 step, +2.55% to +14.85% (L63; not "2-11%" as an earlier
drafting note claimed).~~ **[SUPERSEDED, TASK 29 Part 3.2. Those figures are
all-year means, diluted by n_binding/30; "rose across every cell" is an
artefact of that basis and is retracted.]** **On the binding-year basis**
(mean over the years in which the cap binds: 25 before the L60 step, 26 after,
in every cell) **the carbon shadow moved by −1.39% to +10.43% across the 15
cells: 11 rose (+0.35% to +10.43%) and 4 fell (−1.39%).** The four that fall
are the `unconstrained` cell at every collection rate and `expansion` at
collection 1.00, which coincide numerically with it (115.05 → 113.45
USD/tCO₂). The `tight` cells rose by +5.2% to +6.6% and the `moderate` cells by
+6.6% to +8.8% at every collection rate; `adequacy` by +0.5% to +10.4%. So the
rise is concentrated in the `tight`, `moderate` and `adequacy` cells (and
`expansion` at the lowest collection rate, +6.1%), and is not universal; part
of the all-year rise was the cap binding in one extra year (recomputed from the pre-L60 snapshot
`results/_pre_L60_20260926_1158/fin3/capital_sweep_results.csv` and the
post-L60 file). **Which side needs the derivation:** the **post-L60 side** now
cites the stored column `carbon_shadow_binding_mean_usd_per_tco2` (it equals
the old derivation to 6×10⁻¹⁴; 26 binding years in every cell, so e.g.
`tight`/`no_friction` 212.11 USD/tCO₂). The **pre-L60 side** still needs
binding-year mean = all-year mean × 30 / `carbon_binding_years` (30 evaluable
years assumed), because that snapshot has only the old all-year column
(`carbon_shadow_mean_usd_per_tco2`). The leakage result this sentence was cited beside
is unaffected. (Register L63 point 2 repeated the claim and carries the same correction.) This L60-step comparison has no equivalent
against the original table, since carbon shadow was not tabulated here
in 2026-09-20.

**Two readings, both material.** First, **capital scarcity is an
emissions mechanism**: at `tight`, 42–50% of national emissions
(depending on collection rate; now 42–50%, was 42–50% — unchanged in
range terms, see the updated table above) sit outside a cap the plan
nominally satisfies. Second, **collection friction is too**: at a
fully-funded envelope (`adequacy`) leakage rises from 7.4% to 21.6%
(was 6.6% to 21.3%) purely by moving the collection rate from 100% to
35%, and the compliance flag flips from clean to non-compliant.

**This is consistent with F-21's mechanism in a second, independent
runner.** F-21 established that tightening a grid-only cap raises
national emissions because abatement is pushed across the boundary.
Here the same boundary leaks under two different pressures — capital
rationing and revenue friction — neither of which is a cap-tightening.
The mechanism looks like a property of the grid-only boundary itself,
not of the cap level.

**Attribution — what moves the `tight` / `no_friction` result.** One
change at a time from the HEAD run (`results/_attrib/`):

| run | one change from HEAD | leakage | budget shadow | carbon shadow, **binding-year mean** (USD/tCO₂) | ~~carbon shadow, all-year mean (SUPERSEDED, retained)~~ | binding years /30 |
|---|---|---|---|---|---|---|
| D (HEAD) | — | 42.1% | 6.39 | **212** | ~~184~~ | 26 |
| C′ | `conv_none` | 55.7% | 8.79 | **264** | ~~264~~ | 30 |
| B′ | old B\* (9.104 bn) | 46.3% | 7.00 | **214** | ~~185~~ | 26 |

**CORRECTED (TASK 28 Part D).** The carbon-shadow column previously reported
the **all-year** mean (`carbon_shadow_mean_usd_per_tco2` in the pre-TASK-27
files, `carbon_shadow_all_year_mean_usd_per_tco2` since, averaged over all
30 years including those in which the cap does not bind), which is diluted by
n_binding/30 — the same defect class as TASK 02's and register L70, fixed for
the runners in TASK 27 Part A. The binding-year figures are derived as
all-year mean x 30 / n_binding from `results/_attrib/*.csv` (30 evaluable
years assumed; the 30-year model horizon). **These three runs still need the
derivation**: `_attrib/*.csv` carry only the old all-year column; the stored
`carbon_shadow_binding_mean_usd_per_tco2` exists only in files written since
TASK 27 (the current results file, not the 2026-09-20 `_attrib` runs). **The change in the table comes
from that shadow-mean correction, not from any change in the model**: the
runs, their leakage and their budget shadows are untouched. It matters here
because the three runs bind in different numbers of years (26, 30, 26), so
the dilution differs between them: the C′ run binds in all 30 years and its
all-year mean was therefore not diluted at all.

Conversion (C′ → D): −13.6 pp leakage, −2.40 budget shadow, **≈ −51**
USD/tCO₂ (~~−80~~, all-year basis). B\* (B′ → D): −4.2 pp, −0.62,
**≈ −1.2** (~~−1~~). **Conversion dominates**; B\* moves the carbon shadow by
about a dollar. The conclusion survives; the size of the carbon-shadow
effect of conversion is about two-thirds of what was reported. Notes: (i)
`results/_attrib/C_annual.csv` was MISNAMED — it is the sliced builder
with `conv_none`, not the annual builder — and has been renamed
`C_convnone.csv` (log likewise); the arm string is not recorded in that
log, so `conv_none` rests on the run instruction. (ii) The annual-versus-
sliced port itself (C2) is untested: no run isolates the builder change.
(iii) `tight` is a different USD envelope in B′ (4.552 bn) than in D
(5.137 bn), because the named tier is a fraction of B\*.

**Scope condition.** `disco_collection_rate` scales only the domestic
35% share of the public capital envelope
(`effective_budget = base × (0.35·cr + 0.65)`); it is a fiscal-
realisation rate, not a DisCo collection efficiency, and its arm values
are ASSUMED with provenance unknown (task log rows PACK-3; the
`05_run_fin3_capital_sweep.py` comment). The tariff × collection-rate
revenue mechanism (plan 3.3) is not implemented.

---

## F-32 — A deterministically-optimal plan carries zero adequacy margin, and fails every realisation above its central demand assumption
**NEW. Fixed-plan re-dispatch. Original measurement: annual builder. Re-measured on the sliced builder (Phase B item 17); see the two blocks below and the staleness check.**

**RE-MEASURED, Phase B item 17 leg A (100 draws, SLICED builder,
`conversion_case=conv_central`, `--fix-gas fixed`, run `d1cd080f`;
`results/stochastic_redispatch/draws__d1cd080f.jsonl`).** The text
below is the annual-builder measurement (run `ae0a8d24`) and is retained;
this block records that its structure reproduces on the sliced builder.
The handover's instruction was not to assume either survival or failure;
all three components survive:

- **50 of 100 draws infeasible** (the same 50/100 as the annual builder,
  reproduced exactly across a builder change).
- **No overlap.** Feasible draws' `demand_growth` span 0.019812 to
  **0.038865**; infeasible draws' span **0.039602** to 0.063632. Checked
  over all 100 draws: no feasible draw exceeds any infeasible draw. The
  finding is **structural, not empirical**: the draw sets are identical
  to `ae0a8d24`'s, so the boundary falls in the same gap.
- **The boundary is `organic_central` = 0.039**
  (`src/scenarios.py`, `demand_growth_scenarios()["organic_central"]`).
  The infeasible gas-regime split is also identical (26 baseline / 14
  upside / 8 downside / 2 shock_recovery).

**Reading.** The split is not a statistic. The deterministic optimum
carries exactly zero adequacy margin, so it fails under any positive
demand deviation however small and succeeds under any negative one. The
observed 50% is P(draw > central growth rate) for a symmetric
perturbation, which is why it reproduced the annual-builder figure
exactly across a builder change. The companion free-gas leg (`5d29ee61`,
same draws) is 0 of 100 infeasible. **Not directly observed in this
run's files:** the zero firm-capacity slack itself (the summary records
`gas_add_band_diagnostics`: 600 of 600 band-years at an edge, 12
nonzero-band years per draw x 50 feasible draws) — it is inferred from
the separation, and was measured directly only on the annual builder
(Result 1 below).


**Source.** `scripts/30_run_stochastic_redispatch.py`, 100 draws at seed
42, `--fix-tolerance 1e-5`. Runs `ae0a8d24` (`--fix-gas fixed`,
sourced gas-regime weights) and `42ccf7c7` (symmetric weights), both
`--demand-sigma 0.01`; `49cc69e0` (`--fix-gas free`, sourced,
`--demand-sigma 0.01`) for Result 5; the `--demand-sigma` sweep at
0.005 / 0.01 / 0.02 / 0.05 (`10127e1b`, `ae0a8d24`, `c57b5de4`,
`a9f1e7a3`) for Result 4. All ten figure groups below verified against
`results/stochastic_redispatch/` directly, not carried from a prior
draft.

> **[RE-DERIVED, TASK 84, 2026-10-23 — UNCHANGED: the structure reproduces; the band-year count moves (1150 of 1150, 23 nonzero years per draw); Results 1 and 4 and the symmetric-weights comparison remain annual-builder only (no vintage run).** Basis: uncapped. Vintage `results/stochastic_redispatch/` leg A (`--fix-gas fixed`, 100 draws, sliced builder, `conv_central`, seed 42, sigma 0.01; `summary__857ac191.json`, `draws__857ac191.jsonl`): **50 of 100 draws feasible, 50 infeasible** (as ~~50/100~~ before). **No overlap:** feasible draws' `demand_growth` 0.019812 to **0.038865**; infeasible draws' 0.039602 to 0.063632 (identical to the blended and annual runs; the draw sets are the same, sha256 `1ef6055b...`). The boundary is `organic_central` = 0.039. Infeasible gas-regime split 26 baseline / 14 upside / 8 downside / 2 shock_recovery (identical). Free-gas leg (`summary__19490865.json`, same draws): **100 of 100 feasible** (~~0 of 100 infeasible~~, unchanged). **Moved:** `gas_add_band_diagnostics` ~~600 of 600 band-years at an edge, 12 nonzero-band years per draw~~ **1,150 of 1,150, 23 nonzero-band years per draw** (7 zero-width), n_interior 0; deterministic base cost ~~27,426,913,099~~ **23,323,068,340.46** (equal to the vintage central anchor to the cent); mean ECP cost over the feasible draws of the fixed leg 28.76 bn, of the free leg 36.34 bn. Correspondence: the two vintage runs are the counterparts of `d1cd080f`/`aaced7e1` (fixed) and `5d29ee61` (free) with identical draws and arguments except the capital case (see F-42). **Not re-derivable:** Result 1 (zero slack in every gas-build year, annual builder only, now inferred from the separation and corroborated by the band count), Result 4 (sigma sweep: only sigma 0.01 was run on the vintage model) and the symmetric-weights comparison. Source: `results/stochastic_redispatch/summary__857ac191.json`, `results/stochastic_redispatch/draws__857ac191.jsonl`, `results/stochastic_redispatch/summary__19490865.json`, `results/stochastic_redispatch/draws__19490865.jsonl`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — UNCHANGED for the central runs; the older runs are UNSUPPORTED.** Post-epoch `results/stochastic_redispatch/draws__aaced7e1.jsonl` (leg A, `--fix-gas fixed`; the counterpart of `d1cd080f`, arm tags equal modulo the arms TASK 53 §2.5 completed) and `draws__8fc2e490.jsonl` (leg B, `--fix-gas free`; counterpart of `5d29ee61`), 100 draws each, `conv_central`, `om_central`, window `arrears`, storage capex `declining_nrel`. **Leg A: 50 of 100 draws infeasible (unchanged); feasible `demand_growth` 0.019812 to 0.038865 and infeasible 0.039602 to 0.063632 (both unchanged to six digits); no overlap; boundary = `organic_central` 0.039; infeasible gas split 26 / 14 / 8 / 2 (baseline / upside / downside / shock_recovery, unchanged).** Leg B: 0 of 100 infeasible (unchanged). The zero-adequacy-margin claim survives the storage capital change: the pass/fail pattern is identical draw by draw, although storage build timing moved (central storage 2045 32,742 → 37,414 MWh) and the mean of the feasible costs fell (~~29.385~~ 26.865 bn; deterministic base ~~27.427~~ 24.987 bn, which now equals the unconstrained anchor to 6.7 USD). **UNSUPPORTED (no post-epoch counterpart, left as historical; not re-derived against a different arm set):** the annual-builder / `conv_none` measurement (`ae0a8d24`, `42ccf7c7`, `49cc69e0`, `10127e1b`, `c57b5de4`, `a9f1e7a3`); Result 3's symmetric-weights comparison (50/100 in both weightings); Result 4's `--demand-sigma` sweep at 0.005 / 0.02 / 0.05; and the tolerance variants (`33f01e6c`, `62c90efb`, `6635ec6a`). Source: `results/stochastic_redispatch/{draws,summary}__{aaced7e1,8fc2e490}`.]**

**Method.** The base scenario is solved once to obtain the here-and-now
capacity plan. That plan is then frozen and the model re-solved under
each Monte Carlo realisation of demand growth and gas deliverability,
with only dispatch free to adjust. Unlike Sobol variance decomposition,
which re-optimises at every draw and therefore always finds a feasible
plan where one exists, this method can observe the chosen plan
**failing**.

**Result 1 — firm capacity is built to exactly zero slack.**
`firm_capacity_constraint` sits at zero slack, to solver precision, in
**every year the base-optimal solution builds gas** (slack −2.1e-5 to
+4.8e-5 MW across years 10–29), and comfortably slack in every year it
does not (2,722 MW in 2025, falling to 97 MW by 2034). This is the
expected behaviour of a cost-minimising LP: gas is the marginal
firm-capacity instrument, so the optimiser builds precisely enough to
meet the adequacy requirement and not one megawatt more.

**Result 2 — the plan therefore fails as a step function, not a
distribution.** With the plan frozen (`ae0a8d24`, σ=0.01), feasibility
splits cleanly on demand growth with no overlap: feasible draws span
1.981%–3.887%, infeasible draws 3.960%–6.363%, and the boundary sits at
the central arm value of 3.9% (`organic_central`) — every feasible
draw is at or below it, every infeasible draw is above it, confirmed
against every one of the 100 draws individually, not sampled. **Every
realisation at or below the central demand assumption survives; every
realisation above it fails.** The step has no measurable width at the
tested sampling resolution: at the narrowest σ tested (0.005), the gap
between the nearest feasible/infeasible pair is 0.037 percentage
points (3.8933% feasible, 3.9301% infeasible) — and the gap SCALES
with σ (0.037 / 0.074 / 0.147 / 0.369 pp at σ = 0.005 / 0.01 / 0.02 /
0.05, closely proportional to σ throughout), confirming it is a
sampling-resolution artefact of a razor-thin threshold, not a
measurement of the threshold's true width.

**Result 3 — the gas regime is irrelevant to which side a draw falls
on.** Of the 50 infeasible draws in `ae0a8d24`, 26 occur under
`baseline` deliverability and 14 under `upside` — the *more*
favourable regime — against 8 under `downside` and 2 under
`shock_recovery`. Switching from sourced to symmetric regime weights
(`42ccf7c7`) changes the infeasible count not at all (50/100 in both).
Failure is driven by demand crossing the adequacy threshold, not by
fuel scarcity.

**Result 4 — the threshold is invariant to the width of the demand
prior.** Infeasible fraction is **exactly 50/100 at every one of four
tested σ** (0.005, 0.01, 0.02, 0.05) — not merely similar, identical.
That the fraction is approximately one half is an artefact of a
symmetric prior straddling the failure point, **not a probability of
failure**; a skewed or shifted prior would return a different fraction
for identical underlying behaviour. **The threshold, not the fraction,
is the result.**

**Result 5 — the value of gas as recourse.** Across matched draw sets
(`ae0a8d24` vs. `49cc69e0`), the plan is infeasible in **0 of 100**
realisations when new gas may be built after the realisation is
observed, and **50 of 100** when gas capacity is committed ex ante.
That difference is the value of retaining the option to build gas
late, and it is not visible to any re-optimising method.

**Why there is no backstop.** Self-generation is unbounded and priced,
so energy adequacy can always be met at a cost — which is why the
recourse case never fails. But self-generation carries **no capacity
credit**, so it cannot satisfy the firm-capacity constraint. With gas
frozen and solar credited at zero in the peak block (the peak block
follows sunset, §3.3.3), the annual-builder plan has no instrument at all with which
to meet a higher peak. **[Stale for the sliced run:** there, `convert_mw`
is left free and carries firm-capacity credit, so "no instrument at all" is
not established; the draws fail regardless.**]**

**What this finding does not claim.** It does not identify the plan
that would be optimal under uncertainty — that would require
stochastic programming with recourse. It evaluates the robustness of a
plan optimised for the base case (§3.6).

**Scope conditions — CORRECTED (TASK 28 Part C).** *Superseded text,
true of the annual-builder runs only:* "Annual builder: no conversion
mechanism, hydro exogenous with no dispatch or seasonality, and storage's
energy-arbitrage contribution unvaluable though its firm-capacity credit is
live. The fixed plan covers public solar, EaaS solar, storage and — under
`--fix-gas fixed` — gas; the EaaS subsidy is deliberately not fixed, and
conversion does not exist in this builder." **That last clause is false for
the sliced run (`d1cd080f`, `5d29ee61`).** The sliced builder has
conversion: with `conv_central`, `convert_mw` is a decision variable that
also earns firm-capacity credit in `firm_capacity_constraint`
(`optimize_model_sliced.py:1297-1310`). **The fixed plan does not fix it**:
`summary__d1cd080f.json` `fixed_plan` holds `solar_public_add`,
`solar_eaas_add`, `storage_add` and `gas_add_fixed` only, so **conversion is
free recourse** in the sliced run, as is the EaaS subsidy. On the sliced
builder hydro is dispatched (seasonal budget, per-slice power limit,
minimum-flow floor) and storage's energy-arbitrage value is representable;
its firm-capacity credit is live in both builders. The sliced run still
fails the same 50 draws, so freeing conversion did not change the
infeasible set; **why is not established** (the conversion tiers may already
be fully used in the base plan, or their firm credit may not cover the
shortfall; not checked).
Cost statistics for the `fixed` case are conditional on survival and
are **not comparable** to the recourse case's, which covers the full
draw set. **`gas_add` is held to its base value by a band, not an
exact fix** (register L65) — economically indistinguishable from an
exact fix (the band's own numerical accommodation is orders of
magnitude below the smallest capacity decision in this model), but the
band DOES bind, in the strict LP sense, at every feasible draw's every
nonzero-gas year (`gas_add_band_diagnostics`: 1000/1000 at an edge
across `ae0a8d24`, `42ccf7c7`, and all four sweep runs) — register L65
has the full account of why "binds" and "is economically material" are
not the same test here.

**STALENESS CHECK (TASK 28 Part C): which parts of this entry are annual-only.**
The whole entry predates the sliced port (`c549bba`, 2026-10-01); the
runs behind the original text (`ae0a8d24`, `42ccf7c7`, `49cc69e0`, and the
four σ-sweep runs, all `conv_none`, 2026-09-27) are annual-builder.

| part | status |
|---|---|
| Result 1 (zero slack in every gas-build year) | **annual only**; not observed in the sliced run's files (inferred from Result 2's separation) |
| Result 2 (step at 3.9%, no overlap) | **reproduced on sliced** (`d1cd080f`: 0.019812-0.038865 vs 0.039602-0.063632) |
| Result 3 (gas regime irrelevant; 26/14/8/2) | **reproduced on sliced** (identical split); the symmetric-weights comparison (50/100 in both) is annual only |
| Result 4 (σ sweep, 50/100 at every σ; gap scales with σ) | **annual only** |
| Result 5 (0 vs 50 of 100) | **reproduced on sliced** (`5d29ee61` 0/100 vs `d1cd080f` 50/100) |
| "Why there is no backstop" | **qualified** (conversion is free recourse on sliced) |
| `gas_add` band diagnostic | annual: 1000/1000 at an edge across six runs; sliced: 600/600 (12 nonzero-band years x 50 feasible draws) |
| Scope conditions | **corrected** above |

**See also:** register L65 (exact fixation was numerically fragile
against this same zero-slack finding; the band that resolves it), L66
(what the `peak_mw` correction that makes this finding observable at
all superseded).

---

# Section 6A — Phase B findings (TASK 25)

**Provenance of this section.** Written from Chat's draft, then every
numeric claim re-verified against the named result file by Code
(`docs/task_log_25_phase_b_findings.md`). Where a draft claim did not
verify, the entry states what the file shows and carries a
**"Verification note"**; the full list of failures is in the task log.
All solves are Phase B driver items (`scripts/run_phase_b.sh`), run
2026-10-02/03. "Standard arms" = the central configuration above except
where an entry names a different one.

## F-33 — A horizon-aggregated reliability budget misstates its own standard
**NEW. REL-1, Phase B item 14 (4 invocations x 96 solves = 384:
2 reliability modes x 2 availability arms; 8 cells x 11 eps + 8
unconstrained references per invocation).**

**Source.** `results/rel1/{total,annual}/gas_avail_{nameplate,observed}/`
— `shortfall_by_year.csv` (`shortfall_fraction`, `shortfall_twh`,
`demand_twh`), `rel1_summary.json` (`max_gas_range_eps`,
`max_solar_diff_eps`, `primary_determinant`, `threshold_degeneracy`).
~~The root-level `results/rel1/{total,annual}/rel1_summary.json` are the
pre-extension (floor 0.05) files and are cited only in Part 3.~~ **(CORRECTED, TASK 55, 2026-10-20: those paths do not exist. The pre-extension (floor 0.05) files are `results/_superseded_pre_extension/rel1/{total,annual}/rel1_summary.json` (moved 2026-10-03, TASK 27 Part B; see the README there) and are cited only in Part 3. Parts 1 and 2 come from `results/rel1/{total,annual}/gas_avail_{nameplate,observed}/shortfall_by_year.csv` and `rel1_summary.json`.)**

> **[RE-DERIVED, TASK 84, 2026-10-23 — WEAKENED: Part 2 (understatement of determinants) and Part 3 cannot be tested: all 8 cells are floor-bound in both modes (`INDETERMINATE`); Part 1 survives only at the `observed` arm.** Basis: uncapped. Vintage `results/rel1/{total,annual}/gas_avail_{nameplate,observed}/` (the same paths as the blended run; `09_run_rel1_feasibility.py` overwrites in place, so the blended files no longer exist and the 'was' values are those printed in the entry; the four directories carry the retrofitted sidecar of TASK 83 B.2). **Correspondence arm for arm:** `scripts/run_vintage_rerun.sh` lines 454-457 issue the same four invocations as `scripts/run_phase_b.sh` lines 454-457 (reliability mode x availability arm), run 2026-10-23 18:50-18:55 at HEAD `6fb1a79`; the nameplate baseline cells reproduce the vintage central objective (23,323,068,34x). `shortfall_twh` is exactly `unserved_twh + selfgen_twh`. **Part 1**, cell `downside` / `deployment_unconstrained`, eps 0.10: **`gas_avail_nameplate`: no shortfall at all** (0 years above 10% in both forms; worst 0.0000; horizon fraction 0.0000) ~~horizon-total 3 years above, worst 0.2025, fraction 0.0385; annual 0, 0.1000, 0.0285~~. **`gas_avail_observed`: horizon-total 2 years above 10% (2053: 0.1185, 2054: 0.1551), worst 0.1551, horizon fraction 0.0208 (20.8% of the budget spent); annual 0 years above, worst 0.1000, horizon fraction 0.0098** (~~5 years, 0.1882, 0.0415; 0, 0.1000, 0.0294~~): the horizon form still spends more of the budget and leaves years above the standard, at the observed arm only. **Part 2:** `max_gas_range_eps` **0.0** and `max_solar_diff_eps` **0.0** in all four directories, `primary_determinant` `indeterminate`, **8 of 8 cells floor-bound** (~~5 of 8~~) in both modes (~~gas range 0.049 / 0.079, solar shift 0.001 / 0.049, `gas_regime`~~). The vintage model never needs more than the tightest grid value (0.001) in any cell, so the horizon and annual forms cannot be compared; the 38% understatement and the factor 49 are not reproducible, and the entry's own caveat ('every range is a lower bound') now applies to all of them. **Part 3** (grid artefact; withdrawn false-null claim) is a statement about a superseded file and is not affected. The directional-understatement claim is neither supported nor contradicted by the vintage files. Source: `results/rel1/total/gas_avail_observed/shortfall_by_year.csv`, `results/rel1/annual/gas_avail_observed/shortfall_by_year.csv`, `results/rel1/total/gas_avail_nameplate/shortfall_by_year.csv`, `results/rel1/total/gas_avail_nameplate/rel1_summary.json`, `results/rel1/annual/gas_avail_nameplate/rel1_summary.json`, `results/rel1/total/gas_avail_observed/rel1_summary.json`, `results/rel1/annual/gas_avail_observed/rel1_summary.json`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; conclusions hold. Source line corrected.** `shortfall_twh` in these files is exactly `unserved_twh + selfgen_twh` (maximum difference 0.0 over all four directories) and `shortfall_fraction` is `shortfall_twh / demand_twh`: F-33 reads the REL runners' own definition, not the renamed `unserved_only_reliability`. **Post-epoch files are the Phase B item 14 outputs at the same paths as the pre-epoch ones (09_ overwrites in place; L114); they carry no arm tag.** Part 1, cell `downside` / `deployment_unconstrained`, eps 0.10, `gas_avail_nameplate` (pre-epoch in brackets): horizon-total **3** years above 10% (~~5~~), worst **0.2025 (2054)** (~~0.2353~~), horizon fraction **0.0385** (~~0.0526~~; 38.5% of the budget spent, ~~53%~~); annual **0** years above, worst 0.1000, horizon fraction **0.0285** (~~0.0470~~). The years above 10% are 2052-2054 (0.1337, 0.1689, 0.2025), no longer 2050-2054; the horizon form still spends more of the shortfall budget than the annual form (0.0385 vs 0.0285) and still leaves a fifth of the 2054 demand unserved (0.2025). `gas_avail_observed`: **5** years above (~~5~~), worst **0.1882 (2054)** (~~0.2099~~), horizon fraction **0.0415** (~~0.0456~~); annual 0, 0.1000, **0.0294** (~~0.0432~~). **Part 2 is unchanged to the digit, identical at both availability arms:** gas-regime range horizon-total **0.049**, annual **0.079** (understatement 38%); solar-arm shift **0.001** against **0.049** (a factor of 49); `primary_determinant` `gas_regime` in both. Part 3 (grid artefact, withdrawn false-null claim) is unchanged. **Source correction:** the sentence naming `results/rel1/{total,annual}/rel1_summary.json` as root-level files is wrong — those paths do not exist; the pre-extension (floor 0.05) files are at `results/_superseded_pre_extension/rel1/{total,annual}/rel1_summary.json` (moved 2026-10-03, TASK 27 Part B; README there). The figures come from `results/rel1/{total,annual}/gas_avail_{nameplate,observed}/{shortfall_by_year.csv,rel1_summary.json}`. Part 1 is a Tier B recomputation from the per-year file.]**

**Part 1 — concentration.** Cell `downside / deployment_unconstrained`,
nominal eps 0.10, `gas_avail_nameplate`:

| form | years above 10% | worst year | horizon fraction (sum shortfall / sum demand) |
|---|---|---|---|
| horizon-total | **5** | **0.2353 (2054)** | **0.0526** (53% of the budget spent) |
| annual | **0** | 0.1000 | **0.0470** |

The five worst years under the horizon form are 2050-2054, rising
monotonically (0.1039, 0.1335, 0.1691, 0.2030, 0.2353). The horizon form
spends *more* of the shortfall budget than the annual form (0.0526 vs
0.0470) while leaving a fifth of the 2054 demand unserved. **Verification
note:** at `gas_avail_observed` the same cell gives 5 years above, worst
0.2099 (2054), horizon fraction 0.0456 (total) and 0, 0.1000, 0.0432
(annual) — the pattern replicates, the digits do not. The draft did not
name an arm; the figures above are `gas_avail_nameplate`.

**Part 2 — systematic understatement of determinants.** Extended grid
(floor 0.001), identical at both availability arms:

| | gas-regime range | solar-arm shift | `primary_determinant` |
|---|---|---|---|
| horizon-total | **0.049** | **0.001** | `gas_regime` |
| annual | **0.079** | **0.049** | `gas_regime` |

The horizon form understates the gas effect by 38% ((0.079-0.049)/0.079)
and suppresses the solar effect by a factor of 49 (0.049/0.001). The
whole solar-arm effect sits in one cell (`shock_recovery`: unconstrained
0.001, `nirp_trajectory` 0.002 total / 0.050 annual); the 49x is a ratio
of two single-grid-step differences, not a smooth quantity.

**Part 3 — a coarse grid turns the understatement into a false null.**
At the original floor of 0.05 (~~`results/rel1/total/rel1_summary.json`~~ `results/_superseded_pre_extension/rel1/total/rel1_summary.json`,
eps_levels ending 0.05) the horizon form reported `INDETERMINATE`
(`max_gas_range_eps` 0.0); at floor 0.001 it reports `GAS_REGIME`. The
annual form reported `gas_regime` with a shift of 0.03 at floor 0.05 and
0.079 at floor 0.001. **WITHDRAWN: the claim that the horizon form
produces a false null as a structural property.** It was a grid
artefact. The surviving claim is directional understatement, which holds
at any resolution tested.

**Caveat.** 5 of 8 cells are floor-bound at 0.001 in both modes (all four
`baseline`/`upside` cells, plus `shock_recovery / deployment_unconstrained`),
so every range above is a **lower bound**. 0.001 is a 0.1% shortfall
standard, tighter than any practical service standard; the floor-bound
cells are the finding that those regimes achieve essentially perfect
reliability, not an untested gap.

## F-34 — EaaS relieves gas scarcity where gas is scarce, and worsens it in one regime
**NEW. GAS-2, Phase B item 10 (40 solves: 5 policy arms x 4 gas regimes x
2 financing arms).**

**Source.** `results/gas2/eaas_vs_noeaas_gas_shadow.csv`
(`gas_shadow_relief_pct`), `results/gas2/gas2_summary.json`. Shadow
means are **all-30-year means** (not binding-years-only; see F-30).

> **[RE-DERIVED, TASK 84, 2026-10-23 — WEAKENED: 'worsens it in one regime' is gone: EaaS no longer raises the gas shadow price in any regime; relief percentages move.** Basis: cap plus budget (by design). Vintage `results/gas2/arms_f30c5990/` (40 solves; GAS-2 still solves at the capped `nirp_trajectory` design, L81). Relief by basis, `no_policy` (PV / year-t): baseline **0.0%** (both arms 23.323 bn); upside 0.0%; downside ~~+34.6% / +55.7%~~ **+58.0% / +61.5%**; **shock_recovery ~~-3.3% / +11.6%~~ 0.0% / 0.0%** (identical on both sides: 24 binding years, cost 23.675 bn). NDC arms, downside only (the other regimes have no shadow, 0 vs 0): ndc2_unconditional ~~+65.0% / +77.4%~~ **+91.3% / +91.4%**; ndc2_conditional ~~+94.4% / +94.4%~~ **+85.2% / +86.4%**; ndc3_unconditional ~~+92.4% / +91.6%~~ **+91.7% / +91.6%**; ndc3_conditional ~~+91.0% / +90.9%~~ **+91.0% / +91.1%**. `binding_counts_equal` is False only for `no_policy` downside (26 vs 25 binding years). The 'negative case' (the entry's most heavily weighted result) does not exist on the vintage model: no regime has a higher EaaS shadow price. Where a deliverability shadow price exists, EaaS cuts it, as before. Basis: envelope-carrying by design (the policy arms carry tight/moderate). Source: `results/gas2/arms_f30c5990/eaas_vs_noeaas_gas_shadow.csv`, `results/gas2/arms_f30c5990/gas2_summary.json`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — WEAKENED; the headline clause holds, the second clause is basis-dependent.** Post-epoch `results/gas2/arms_be9472ef/eaas_vs_noeaas_gas_shadow.csv` (40 solves; the file now carries both bases; all-year means; GAS-2 still solves at the capped `nirp_trajectory` design, L81). Relief by basis, `no_policy` (pre-epoch value in brackets; the pre-epoch field was a present value): baseline ~~+27.1%~~ **0.0%** (both bases; both arms 24.987 bn); upside 0.0%; downside ~~+36.1%~~ **+34.6% (PV) / +55.7% (year-t)**; **shock_recovery ~~−10.6%~~ −3.3% (PV) / +11.6% (year-t)**. NDC arms, downside only (the other regimes have no shadow, 0 vs 0): ndc2_unconditional ~~+63.8%~~ **+65.0% / +77.4%**; ndc2_conditional ~~+91.8%~~ **+94.4% / +94.4%** (binding-mean basis +92.2%); ndc3_unconditional ~~+91.0%~~ **+92.4% / +91.6%**; ndc3_conditional ~~+89.6%~~ **+91.0% / +90.9%** (PV / year-t). **Two claims change:** (1) the only negative case (`shock_recovery`, `no_policy`) is negative on the PV basis only (−3.3% PV, +11.6% year-t). **Tested, not assumed:** the binding-year sets of the two legs were recovered by re-solving the two cases at the Phase B arms (a verification artefact; the means reproduce the stored file exactly, 9,202,170.1 and 9,507,276.2 PV, 20,723,038.9 and 18,326,992.5 year-t) and are **identical: 24 years, 2031–2054, in both legs**. A difference in binding years therefore cannot explain the flip, and the basis question is not one of contaminated year weights. What differs is the profile of the duals over the same years: in 2031 the EaaS leg's dual is higher (74.7 M against 41.1 M PV; 118.6 M against 65.2 M year-t), in 2032–2034 equal, and in 2035–2054 lower (e.g. 2045 1.33 M against 2.71 M PV, 6.18 M against 12.65 M year-t; 2054 0.39 M against 5.08 M PV, 3.65 M against 47.3 M year-t). The present-value mean is dominated by 2031–2034 (large PV weights), where EaaS is not lower, so it comes out negative; the year-t mean is dominated by 2045–2054, where EaaS is much lower, so it comes out positive. EaaS therefore *raises* early (2031) gas scarcity and *relieves* late scarcity; the mechanism for the early rise is still not established. The finding stays WEAKENED: the 'worsens in one regime' clause is true of the first-year dual and of the PV mean, false of the year-t mean, and the choice of mean is a presentation decision, not a correction. (2) the EaaS and public arms no longer bind in the same number of years in every priced pair (ndc2_conditional downside: 7 vs 5; `binding_counts_equal` False), so the "ratios cancel exactly" argument no longer covers that cell (its relief is 94.43% on the all-year basis and 92.21% on the binding-mean basis). `eaas_always_relieves_gas` is `False` in all five policy blocks on the PV basis. Costs, shock_recovery `no_policy`: public ~~29.571~~ 27.102 bn, EaaS ~~29.338~~ 26.946 (EaaS still cheaper); gas downside public ~~38.049~~ 34.669, EaaS ~~33.409~~ 30.809 (difference ~~4.64~~ 3.86). Source: `results/gas2/arms_be9472ef/eaas_vs_noeaas_gas_shadow.csv`, `gas2_summary.json`.]**

**UNCHANGED by the TASK 27 gas-shadow correction (see F-30).** The relief
percentages below are ratios of two means that were each diluted by the
same factor n_binding/30, and the two arms bind in the **same number of
years in all eight priced pairs** (no_policy: 23/23, 24/24, 22/22, 24/24;
ndc2_unconditional downside 21/21; ndc2_conditional 7/7; ndc3_unconditional
17/17; ndc3_conditional 9/9), so the factor cancels exactly: the
binding-year-basis relief reproduces every figure to 2e-14 percentage
points (`results/gas2/gas_shadow_binding_derived.csv`, derived as
mean x 30 / n_binding because GAS-2 stores no per-year duals). A reader
comparing F-30 with this entry should not conclude F-34 was missed. What
*does* scale is any absolute shadow-price level or absolute relief quoted
in USD/TWh_th: those are all-year values; their binding-year equivalents
are larger by 30/n_binding (e.g. `shock_recovery` under `no_policy`:
8,582,582 -> 9,489,191 all-year is 10,728,228 -> 11,861,489 on the
binding-year basis; the -10.6% is the same). The relief is invariant only
because the counts happen to match; the runner now records
`binding_counts_equal` and emits both bases.

| policy | baseline | upside | downside | shock_recovery |
|---|---|---|---|---|
| no_policy | +27.1% | 0.0% | **+36.1%** | **-10.6%** |
| ndc2_unconditional | 0 vs 0 | 0 vs 0 | **+63.8%** | 0 vs 0 |
| ndc2_conditional | 0 vs 0 | 0 vs 0 | **+91.8%** | 0 vs 0 |
| ndc3_unconditional | 0 vs 0 | 0 vs 0 | **+91.0%** | 0 vs 0 |
| ndc3_conditional | 0 vs 0 | 0 vs 0 | **+89.6%** | 0 vs 0 |

`eaas_always_relieves_gas` is `False` in all five policy blocks.

**TASK 29 CHECK (2026-10-08): NOT AFFECTED by the `required_margin`
correction; no figure above changed and none is struck.** Every figure
here comes from Phase B item 10 (`07_run_gas2_eaas_gas_relief.py`, run
2026-10-02 at `45587af`), which was **not re-run** in TASK 29. It needs no
re-run: at `45587af` the GAS-2 solves go through `run_gas_regime_ndc_matrix`
(whose `required_margin=1.10` keyword had already been removed, so
`dscr_case` supplied the margin) and `run_no_policy_gas_matrix_eaas`, and
neither contains the `scenario["required_margin"] = ...` override that
`run_rel3_financing_frontier` and `run_dem2_growth_gas_matrix` carried.
The `1.10`/`1.05` values in `07_`'s `NDC_CASES` were written to its CSV and
summary as a `required_margin` field but never reached a solve (those solves
used the `dscr_case`-derived 1.30); that field is the only thing wrong in
the stored files, and it is a label, not a result. (Checked by reading the
code at `45587af`, not by re-solving.) Whether the margin matters to the
relief percentages was therefore not measured; only that it was not the
forced value.

**Reading.** Where a deliverability shadow price exists, EaaS cuts it
(downside under every policy). The single negative case is
`shock_recovery` under `no_policy`: EaaS **raises** the gas shadow mean
from 8,582,582 to 9,489,191 USD/TWh_th (-10.6% relief) while lowering
NPV cost (29.57 -> 29.34 bn); binding years are 24 on both sides.
**The negative case carries more weight than the positive ones and its
mechanism is not established.** Recorded as unexplained.

> **[CAP DEPENDENCE, TASK 39 / TASK 43, 2026-10-10 (L81).** GAS-2 solves at `load_scenario`'s default `solar_build_case`, `nirp_trajectory`, a cumulative solar cap; DEM-2 uses `deployment_unconstrained`. A scratch re-run of the 40 GAS-2 solves with only that default changed (capped control reproduces every stored value) gives, no policy: baseline 27.07% (unchanged), upside 0.0% (unchanged), downside 36.05% → 86.46%, **shock_recovery −10.56% → +35.59%**. The cumulative solar cap binds in 3 years (public arm) and 4 years (EaaS arm) of the capped `shock_recovery` solves and in none of the baseline and upside solves. The negative case is therefore a property of the capped design; the mechanism by which the cap produces it (a composition claim) was not tested. F-34's gas-downside costs (public 38.05 bn, EaaS 33.41 bn, difference 4.64 bn) are from the capped design and are not comparable with DEM-2's 7.875 bn.]**

**Verification note (draft failed on two points).** (1) The draft listed
three NDC figures (91.8 / 91.0 / 89.6%) for four NDC arms; the
omitted one is `ndc2_unconditional`, **63.8%**. (2) The draft framed
the result as "substitutes in three regimes, not in the fourth". Under
`no_policy` the pattern is +27.1 / 0.0 / +36.1 / -10.6 across the four
regimes; under every NDC arm only `downside` has a shadow price at all,
so relief is *undefined* (0 vs 0) in the other three, not positive. The
"three regimes plus one exception" reading is not what the file shows.

## F-35 — EaaS value has an interior maximum in policy stringency
**NEW. REL-3, Phase B item 16 (960 solves: 2 availability arms x 480).**

**Source.** `results/rel3/gas_avail_{nameplate,observed}/frontier_shift.csv`
(`frontier_shift_eps`, `pub_threshold_eps`, `eaas_threshold_eps`,
`cell_is_floor_bound`). **Both arms are identical** in every threshold
and shift (checked cell by cell). Column order below is
`baseline / downside / shock_recovery / upside`.

> **[RE-DERIVED, TASK 84, 2026-10-23 — WEAKENED: resolution: 12 of 20 cells are floor-bound (3 before) and EaaS is advantaged in 5 (8 before); the interior-maximum pattern survives.** Basis: cap plus budget (by design). Vintage `results/rel3/arms_1fa34539/gas_avail_nameplate/` and `arms_1e01828a/gas_avail_observed/` (960 solves; **the two availability arms are identical cell by cell**). Shift by gas regime (baseline / downside / shock_recovery / upside): no_policy 0 / 0 / 0 / 0, thresholds 0.05 (public = EaaS), **4 of 4 floor-bound**; ndc2_unconditional ~~+0.07 / +0.17 / +0.07 / +0.07~~ **0 / +0.15 / 0 / 0**, public 0.05 / 0.20 / 0.05 / 0.05, EaaS 0.05, 4 floor-bound; ndc2_conditional 0 (public = EaaS = ~~0.25~~ 0.20); ndc3_unconditional ~~+0.05 / +0.10 / +0.05 / +0.05~~ **+0.05 / +0.15 / +0.05 / +0.05**, public 0.10 / 0.20 / 0.10 / 0.10, EaaS 0.05, 4 floor-bound; ndc3_conditional 0 (~~0.20~~ 0.15 = 0.15). EaaS-advantaged cells ~~8~~ **5 of 20**; floor-bound cells ~~3~~ **12 of 20** (a floor-bound threshold is the tightest grid value, so the shift there is a lower bound). Maximum shift ~~0.17 / 0.10~~ **0.15 / 0.15** (`ndc2_unconditional` and `ndc3_unconditional`, `downside`). By arm class: no_policy 0.0, unconditional 0.15, conditional 0.0 (the interior maximum in stringency is still visible). Plan-at-threshold counts are vertex properties (F-12). Basis: the reliability frontier under a cap and a budget by design. Source: `results/rel3/arms_1fa34539/gas_avail_nameplate/frontier_shift.csv`, `results/rel3/arms_1e01828a/gas_avail_observed/frontier_shift.csv`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — UNCHANGED.** Post-epoch `results/rel3/arms_31d60098/gas_avail_nameplate/` and `arms_5037b4db/gas_avail_observed/` (960 solves, both arms): **every public threshold, EaaS threshold, shift and floor-bound flag equals the pre-epoch table in all 40 cell-arm pairs** (0.00 eps units of movement); 8 of 20 cells EaaS-advantaged, 3 of 20 floor-bound (all `no_policy`), maximum shift 0.17 (`ndc2_unconditional` / `downside`) and 0.10 (`ndc3_unconditional` / `downside`); both availability arms identical. This is at the resolution of the eps grid. The plan-at-threshold counts (`eaas_bankability_n_binding_years_at_threshold`) did move in some cells (e.g. `ndc2_conditional` nameplate 8 → 6, observed 10 → 8; `ndc3_conditional` downside observed 8 → 7) and are vertex properties of the normal solve (F-12), not results. Source: `results/rel3/arms_{31d60098,5037b4db}/gas_avail_*/frontier_shift.csv`.]**

| policy | shift by gas regime | public thr. | EaaS thr. | floor-bound cells |
|---|---|---|---|---|
| no_policy | 0 / 0 / 0 / 0 | 0.05 / 0.08 / 0.05 / 0.05 | same | **3 of 4** |
| ndc2_unconditional | +0.07 / **+0.17** / +0.07 / +0.07 | 0.15 / 0.25 / 0.15 / 0.15 | 0.08 | 0 |
| ndc2_conditional | 0 / 0 / 0 / 0 | 0.25 | 0.25 | 0 |
| ndc3_unconditional | +0.05 / **+0.10** / +0.05 / +0.05 | 0.20 / 0.25 / 0.20 / 0.20 | 0.15 | 0 |
| ndc3_conditional | 0 / 0 / 0 / 0 | 0.20 | 0.20 | 0 |

Max shift by arm class: **0.0** (no_policy), **0.17** (unconditional;
0.10 for ndc3, 0.17 for ndc2), **0.0** (conditional). 8 of 20 cells are
EaaS-advantaged; 3 of 20 are floor-bound (all `no_policy`).

**Verification note.** The draft stated the unconditional max as 0.1;
that is the `ndc3_unconditional` maximum. The `ndc2_unconditional /
downside` cell reaches **0.17** (public 0.25, EaaS 0.08), which the
draft's own "maximum of 0.17" sentence below already used.

**TASK 29 RE-RUN (2026-10-04, code `4ec43b4`, Phase B item 16, both
availability arms, 960 solves, 0 errors) — THE TABLE ABOVE DID NOT MOVE.**
The values above were computed with the EaaS arm's `required_margin`
**forced to 1.10** (1.05 in the two `*_conditional` policy cases) instead
of the `dscr_case`-derived **1.30**. The bankability test
(`eaas_bankability`, `eaas_subsidy_limit`) is one **only the EaaS arm
takes** (in a public arm `eaas_disable` fixes EaaS solar at zero, so the
margin multiplies a variable that is already zero), so the forced value was
a **laxer version of a test applied to one arm only**. The bias therefore
ran toward **overstating EaaS feasibility**: a 1.10 cushion lets the EaaS
arm be declared feasible at a tighter standard than a 1.30 cushion would.
That is the direction this finding could have been wrong in. It was
checked cell by cell against both new `frontier_shift.csv` files (20 cells
× 2 availability arms): **every public threshold, every EaaS threshold,
every shift and every floor-bound flag equals the table above.** The
headline counts are unchanged: 8 of 20 cells EaaS-advantaged, 3 of 20
floor-bound (all `no_policy`), maximum shift 0.17 (`ndc2_unconditional /
downside`) and 0.10 (`ndc3_unconditional / downside`), both arms identical.
**Movement: 0.00 eps units in all 40 cell-arm pairs.** This is robustness
evidence **at the resolution of the eps grid** (0.05, 0.06, 0.08, 0.10,
0.15, 0.20, 0.25, ...): a threshold move smaller than the spacing between
adjacent grid points (0.05 above 0.10) could not show here, and the
sole-route ranges are derived from the same thresholds. The conditional
cases were the furthest from the corrected value (1.05 against 1.30) and
did not move.

What did move is the **plan at the threshold**, not the threshold:
`eaas_bankability_n_binding_years_at_threshold` (not tabulated in this entry
before). Exact before → after, from the pre-correction snapshot
`/home/zeus/thesis-snapshots/results_pre_margin_fix/` (nameplate arm; the
four gas regimes agree within a policy): no_policy 1 → 1; `ndc2_unconditional`
3 → 6; `ndc2_conditional` 5 → 8; `ndc3_unconditional` 4 → 7;
`ndc3_conditional` 4 → 5. Observed arm: no_policy 1 → 1;
`ndc2_unconditional` 1 → 3; `ndc2_conditional` 6 → 10; `ndc3_unconditional`
2 → 4; `ndc3_conditional` 4 → 7 (4 → 8 in `downside`). The constraint binds in
more years in every policy cell, with 0 of 40 thresholds changed. **[UNIDENTIFIED,
TASK 30 Part D: these binding-year counts are properties of the solver's
chosen plan, not of the model.** F-12 shows the public/EaaS split is not
identified: it is bounded below by the public budget and above by total solar,
and the interval is 10,969 MW wide even at the tight budget (frozen-state figure; 16,703.8 MW at the current epoch, F-12, TASK 46). The counts above
are a normal-solve vertex inside that interval. The REL-3 output does not record
the budget dual, so whether the budget binds at these thresholds is not known
from the stored file. The reading "the same standard is met by a different plan"
is correct; the counts must not be quoted as a result about what EaaS builds,
and "bankability binds more under 1.30" is true of this vertex only (F-12
reporting rule: any EaaS figure is a point in a stated interval).]** (An earlier
version of this paragraph said F-12 shows the split unidentified "where the
budget does not bind"; F-12 says it is unidentified in every capital case, the
binding ones included. Corrected 2026-10-08.)

**Reading.** EaaS is worth nothing without policy pressure, a great deal
under moderate (unconditional) ambition, and nothing again under maximum
ambition where the cap binds beyond what financing can relieve. Value
concentrates at moderate ambition. The sole-route range (standards only
EaaS can meet) for each advantaged cell is `eaas_sole_route_eps_range`
(TASK 24), e.g. `downside / ndc2_unconditional` = (0.08, 0.25]. (The
conditional ties at 0.25 and 0.20 are interior results: both arms hit
the same wall.)

**Scope note (TASK 41).** The quantity here is a shift in the
feasibility threshold, not EaaS value in cost terms, and whether the
public capital envelope binds at these thresholds is not recorded in
the stored file. F-38's corrected separation result does not
transfer to this finding and the two should not be read as the same
claim.

**Grid dependence, recorded explicitly.** At the original floor of 0.20
(`results/rel3/frontier_shift.csv`, pre-extension), 16 of 20 cells were
floor-bound (`docs/task_log_22_rel3_grid.md`), only 2 cells were
advantaged and the maximum shift was 0.05 — essentially a null. The
extended grid (floor 0.05) produced 8 advantaged cells and a maximum of
0.17. **The earlier conclusion was wrong because of the grid, not the
model.** The `ndc2_conditional` ties at 0.25 were identified as interior
results before the extension and survived it.

## F-36 — EaaS reduces cost dispersion as well as expected cost
**NEW. Monte Carlo, Phase B item 18 (5 cases x 100 draws = 500 solves;
draws file `results/stochastic/draws__bd65b0bb.csv`, sha256 `1ef6055b...`).**

**Source.** `results/monte_carlo/monte_carlo_summary.csv`
(`cost_mean/p10/p90`), `eaas_necessity.json`.

> **[RE-DERIVED, TASK 84, 2026-10-23 — MOVED: magnitudes; the conditional ratio of spreads (1.8x); `leakage_flagged` shares.** Basis: cap plus budget (by design). Vintage `monte_carlo_summary.csv` (500 solves, 0 infeasible; draws `results/stochastic/draws__6b686491.csv`, same sha256 `1ef6055b...`). Cost (bn): NDC3 uncond public (tight) mean ~~57.81~~ **43.29**, P10 ~~26.72~~ **21.06**, P90 ~~89.76~~ **70.74**, P90/P10 3.36 (**3.36**); NDC3 uncond EaaS mean ~~27.73~~ **26.61**, P10 ~~21.52~~ **20.62**, P90 ~~34.17~~ **32.70**, 1.59 (**1.59**); NDC3 cond public (moderate) mean ~~48.47~~ **38.76**, P10 ~~23.70~~ **22.25**, P90 ~~77.08~~ **61.40**, ~~3.25~~ **2.76**; NDC3 cond EaaS mean ~~28.85~~ **28.31**, P10 ~~22.64~~ **22.25**, P90 ~~35.36~~ **34.66**, 1.56 (**1.56**). Ratio of spreads ~~2.1x~~ **2.1x** (uncond 3.36/1.59) and ~~2.1x~~ **1.8x** (cond 2.76/1.56). `P(EaaS reduces cost)` **100%** (uncond) and **94%** (cond) by the strict sign; 5 and 11 draws tie to within 4 USD (differences 0.3 to -0.7 USD at the tie end) and none is worse beyond 1e3 USD, so 'EaaS does not raise cost in any draw' still holds. Mean saving ~~$30.08B (45.6%)~~ **$16.68B (29.3%)** and ~~$19.62B (32.8%)~~ **$10.45B (19.3%)**. `leakage_flagged` share of public draws ~~92% / 79%~~ **69% / 54%**, 0% of EaaS draws. `Matched to 30_ central (no NDC)`: mean ~~30.60~~ **24.25 bn**, P10 18.96, P50 23.76, P90 30.65. The conclusion (EaaS reduces dispersion as well as cost) holds. Basis: tight/moderate capital by design. Source: `results/monte_carlo/arms_f30c5990/monte_carlo_summary.csv`, `results/monte_carlo/arms_f30c5990/monte_carlo_raw.csv`, `results/monte_carlo/arms_f30c5990/eaas_necessity.json`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; conclusion holds.** Post-epoch `results/monte_carlo/arms_be9472ef/` (Phase B item 18, 2026-10-20 00:34; 500 solves, 0 infeasible; draws `results/stochastic/draws__3fdfba47.csv`, **identical sha256 `1ef6055b...` and content to `draws__bd65b0bb.csv`**). Cost (bn), pre-epoch in brackets: NDC3 uncond public (tight) mean **57.81** (~~59.19~~), P10 **26.72** (~~28.43~~), P90 **89.76** (~~91.12~~), P90/P10 **3.36** (~~3.20~~); NDC3 uncond EaaS mean **27.73** (~~30.76~~), P10 **21.52** (~~23.40~~), P90 **34.17** (~~38.54~~), **1.59** (~~1.65~~); NDC3 cond public (moderate) mean **48.47** (~~51.41~~), P10 **23.70** (~~25.74~~), P90 **77.08** (~~80.08~~), **3.25** (~~3.11~~); NDC3 cond EaaS mean **28.85** (~~32.01~~), P10 **22.64** (~~24.71~~), P90 **35.36** (~~39.64~~), **1.56** (~~1.60~~). The ratio of spreads is ~~1.9×~~ **2.1×** (uncond 3.36 / 1.59; cond 3.25 / 1.56). `P(EaaS reduces cost)` is **98%** (uncond) and **95%** (cond) by the strict sign (`eaas_necessity.json`), ~~100% / 100%~~; **tolerance-dependent as before**: 2 (uncond) and 7 (cond) draws have EaaS and public cost equal to within 4 USD (differences −1.8 to +3.8 USD), none is worse beyond 1e3 USD, so "EaaS does not raise cost in any draw" still holds and the strict percentage is a solver-noise count (more ties than before because the public budget binds in fewer draws). Mean saving ~~$28.43B (42.0%)~~ **$30.08B (45.6%)** and ~~$19.40B (31.0%)~~ **$19.62B (32.8%)**. The `leakage_flagged` share of public draws is 92% (uncond) and 79% (cond) (~~93% / 82%~~). The `Matched to 30_ central` case (F-42's WS): mean **30.60 bn**, median 25.76. Source: `results/monte_carlo/arms_be9472ef/{monte_carlo_summary.csv,monte_carlo_raw.csv,eaas_necessity.json}`.]**

**TASK 29 RE-RUN (2026-10-04, code `4ec43b4`, Phase B item 18, 500 solves,
0 infeasible, same draws file; `results/monte_carlo/`).** Old values are
struck and kept; the new values follow. The prior EaaS rows were computed
with `required_margin` **forced to 1.10** (unconditional) and **1.05**
(conditional) instead of the `dscr_case`-derived **1.30**. That is a
**laxer version of a bankability test that only the EaaS arm takes** (the
public arms disable EaaS solar), so the bias ran toward **overstating EaaS
feasibility, and so understating EaaS cost and its spread**. Public rows
carry no margin and did not change at the displayed precision.

| case | mean | P10 | P90 | P90/P10 |
|---|---|---|---|---|
| NDC3 uncond, public (tight capital) | $59.2B (unchanged; 59.19) | $28.4B (28.43) | $91.1B (91.12) | **3.20** (3.205) |
| NDC3 uncond, EaaS | ~~$30.7B~~ **$30.8B** | $23.4B (23.40) | ~~$38.3B~~ **$38.5B** | ~~1.64~~ **1.65** |
| NDC3 cond, public (moderate capital) | $51.4B (51.41) | $25.7B (25.74) | $80.1B (80.08) | 3.11 |
| NDC3 cond, EaaS | $32.0B (32.01) | $24.7B (24.71) | $39.6B (39.64) | 1.60 (1.604) |

**How much each figure moved.** (Exact old values from the pre-correction snapshot
`/home/zeus/thesis-snapshots/results_pre_margin_fix/monte_carlo/`.)
**Unconditional EaaS:** mean 30.666 → 30.763B (**+0.098B, +0.32%**); P90
38.318 → 38.536B (**+0.219B, +0.57%**), the largest move in the table; P10
23.3965B exactly unchanged; P90/P10 1.638 → 1.647. The two draws measured exactly in the TASK 29
subset moved by **+133.2M (+0.39%)** at draw 0 and **+2.1M (+0.007%)** at
draw 1: the correction bites in some draws and not in others, which is
consistent with the upper-tail-only movement above (the per-draw old values
were not retained, so which draws move was not checked beyond these two). **Conditional
EaaS (1.05 → 1.30, the largest margin change in the table): mean 31.993 →
32.009B (+0.016B, +0.05%); P10 and P90 exactly unchanged** (draws 0 and 1
changed by −16.9 and −8.4 USD, solver noise). **Public rows: none.** The verdict is **robust**: the correction is
real but small, confined to the upper tail of one case, and the ranking and
the ratio of spreads (3.20 against 1.65, about 1.9×) are what they were.

`P(EaaS reduces cost)` = 100% (unconditional), ~~99%~~ 100% (conditional);
mean saving ~~$28.53B (42.1%)~~ **$28.43B (42.0%)** and ~~$19.42B~~ $19.40B
(31.0%). **The conditional 99% → 100% is not a movement in the result.**
In 3 of the 100 conditional draws EaaS and public cost the same to within
$8 (the two financing arms evidently solve to the same plan there; why was
not checked): the sign of a $2-$8 difference is solver noise, and the old 99% and new 100% differ only in which side of zero it
fell. The statistic is tolerance-dependent at that draw count; "EaaS does
not raise cost in any draw" is what the data support (the snapshot's `eaas_necessity.json` confirms 0.99 before, 1.0 after). **Reading.** EaaS acts as a
risk-reduction instrument, not only a cost-reduction one — a result no
deterministic solve can produce, and the one most directly relevant to
bankability.

**Notes.** (1) `P(reduces unserved) = 0%` carries no information:
unserved energy is 0.0 in every draw of every case under nameplate
availability. (2) The `leakage_flagged` column is True in 93%
(unconditional) and 82% (conditional) of public draws and 0% of EaaS
draws; the public arm's cost distribution is therefore largely a
leakage-flagged distribution. (3) The unconditional and conditional
pairs use different capital cases (tight vs moderate), so the two
savings figures are not a like-for-like comparison of the cap alone.

## F-37 — Demand growth and gas scarcity compound (public financing only)
**NEW. DEM-2, Phase B item 13 (72 solves).**

**Source.** `results/dem2/gas_premium.csv`, `interaction_matrix.csv`,
`dem2_summary.json`. `no_policy`, `public_only` arm.

> **[RE-DERIVED, TASK 84, 2026-10-23 — MOVED: magnitudes; the compounding is larger (interaction +36.46 bn) and EaaS removes a smaller fraction (27.7x, was 48x).** Basis: cap plus budget (by design); no-policy arm. Vintage `results/dem2/arms_1fa34539/` (72 solves), `no_policy`, `public_only`, gas premium (bn) at 3.14% / 3.9% / 6.0% growth: downside ~~+1.58 / +8.93 / +29.56~~ **+2.31 / +3.70 / +38.77**, interaction ~~+27.98~~ **+36.46**; upside ~~-0.46 / -0.61 / -14.63~~ **-0.63 / -1.11 / -10.59**, interaction ~~-14.18~~ **-9.96**; shock_recovery interaction ~~+5.72~~ **+6.24**. EaaS downside interaction ~~+0.59 (premium 1.37 -> 1.96)~~ **+1.32 (premium 2.31 -> 3.63)**, i.e. **27.7 times smaller** than the public arm (~~48x~~). EaaS still removes most of the compounding. Demand growth and gas scarcity compound in the public arm; the conclusion holds. Source: `results/dem2/arms_1fa34539/gas_premium.csv`, `results/dem2/arms_1fa34539/interaction_matrix.csv`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; conclusion holds.** Post-epoch `results/dem2/arms_31d60098/` (72 solves), `no_policy`, `public_only`, gas premium (bn) at 3.14% / 3.9% / 6.0% growth: downside ~~+2.16 / +9.83 / +29.43~~ **+1.58 / +8.93 / +29.56**, interaction ~~+27.27~~ **+27.98**; upside ~~−0.71 / −1.00 / −14.50~~ **−0.46 / −0.62 / −14.63**, interaction ~~−13.79~~ **−14.18**; shock_recovery interaction ~~+5.62~~ +5.72. EaaS downside interaction ~~+0.93 (premium 1.80 → 2.74)~~ **+0.59 (premium 1.37 → 1.96)**, i.e. **48 times smaller** than the public arm (~~29×~~). EaaS still removes most of the compounding. The 34× and 29× figures in the body are history. Source: `results/dem2/arms_31d60098/{gas_premium.csv,interaction_matrix.csv,dem2_summary.json}`.]**

| | 3.14% growth | 3.9% | 6.0% | interaction |
|---|---|---|---|---|
| downside gas premium | +2.16B | +9.83B | **+29.43B** | **+27.27B** |
| upside gas premium | -0.71B | -1.00B | -14.50B | **-13.79B** |

**Verification note / scope the draft omitted.** The figures belong to
the `public_only` arm. Under `eaas` the same downside interaction is
~~**+0.80B** (premium 1.80B -> 2.61B), 34 times smaller~~ **+0.93B**
(premium 1.80B -> 2.74B), **29 times smaller** (exact: 0.804 → 0.934B). EaaS largely removes the
compounding; this entry and F-38 are the same fact from two sides.

**TASK 29 RE-RUN (2026-10-04, code `4ec43b4`, Phase B item 13, 72 solves).**
The struck EaaS values were computed with `required_margin` **forced to
1.10** instead of the `dscr_case`-derived **1.30**: a laxer version of a
bankability test that **only the EaaS arm takes** (the `public_only` arm
disables EaaS solar), so the bias ran toward **overstating EaaS
feasibility**, which here means understating EaaS cost at high growth.
**How much each figure moved.** *Table above (`public_only`, all six
cells and both interactions): no movement* (+2.16/+9.83/+29.43B, interaction
+27.27B; -0.71/-1.00/-14.50B, interaction -13.79B, all identical at the
printed precision, as they must be: the margin does not enter a public
arm). *EaaS downside interaction:* **+0.804B → +0.934B, +0.130B (+16%)**;
the 6.0%-growth EaaS downside premium **2.606B → 2.736B (+5.0%)**; the other EaaS interactions moved by +0.042B (`ndc2_unconditional`), +0.027B (`ndc3_unconditional`) and +0.016B (`shock_recovery`); the 3.14%
premium (1.80B) did not move. The qualitative conclusion, EaaS removes
most of the compounding, **stands**; the ratio against the public arm falls
from 34× to 29×, so the number "34 times smaller" must not be quoted. The
EaaS interaction is the only quantity in this entry the correction touched,
and it moved by about an eighth of a billion on a scale where the public
arm's is 27B.

## F-38 — EaaS value is conditional on the public capital envelope binding
~~F-38 — EaaS value is conditional on something binding~~ **(SUPERSEDED title, corrected 2026-10-09, TASK 41: the public capital envelope binds at the central cell, so "something binding" did not discriminate; see the Reading below.)**
**NEW. DEM-2, Phase B item 13 (72 solves).**

**Source.** `results/dem2/eaas_value.csv` (`eaas_value_usd`), `no_policy`.

> **[RE-DERIVED, TASK 84, 2026-10-23 — STRENGTHENED: the separation by the public-capital budget dual is wider (zero-dual maximum -0.61 USD, positive-dual minimum 653.6 M USD); table magnitudes and the central cell move.** Basis: cap plus budget (by design). Vintage `results/dem2/arms_1fa34539/` (72 solves, all optimal; EaaS value = public cost - EaaS cost; public dual = `budget_shadow` of the `public_only` row, **the public-capital budget dual, not the concessional-envelope dual**). **The 36-cell separation, both sides:** in the **18** cells where the public dual is zero (<= 1e-6) EaaS value is at most **-0.61 USD** (minimum -6.56 USD; ~~5.23 USD / -3.91, 8 cells~~); in the **18** cells where it is positive EaaS value is never below **653,600,175 USD** (~~180,067,166, 28 cells~~). The gap between the sides is ~~180 M~~ **654 M USD**. The EaaS arm's own budget dual is zero (<= 5.9e-16) in all 36 cells. Table, `no_policy` (bn): 3.14%: baseline / upside / shock_recovery ~0, downside ~~0.21~~ **-0.00**; 3.9% (central): baseline **-0.00**, upside -0.00, downside ~~7.34~~ **1.017**, shock_recovery ~~0.43~~ **-0.00**; 6.0%: baseline ~~39.26~~ **11.46**, upside ~~25.40~~ **2.445**, downside ~~66.87~~ **46.60**, shock_recovery ~~44.72~~ **17.13**; across the three policy arms the 6.0% range is ~~25.40-71.05~~ **2.44-51.27 B**. **Central cell:** public dual 0, EaaS value -5.33 USD (blended -0.20 USD). Monotone in the dual given a positive dual: pairwise inversions (stated definition: cells with the other two dimensions fixed, ordered by public dual) **2 / 2 / 0** in (growth x policy) / (growth x gas) / (gas x policy); the TASK 55 count of 0 used a definition that is not recorded, and the same code on the blended file returns 0 / 3 / 0, so 'zero inversions' cannot be confirmed as stated. Log-log slope of value on dual **1.67 / 1.59** at 3.9% / 6.0% (3.14%: no positive-dual cell), pooled **1.73** (~~4.26 / 1.67 / 6.32; 2.47~~): the dual orders EaaS value and does not price it. The conditionality claim (EaaS value requires a binding public budget) is stronger. Source: `results/dem2/arms_1fa34539/dem2_results.csv`, `results/dem2/arms_1fa34539/eaas_value.csv`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; the conditionality result holds, the central-cell statement does not.** Post-epoch `results/dem2/arms_31d60098/` (72 solves; EaaS value = public cost − EaaS cost; public dual = `budget_shadow` of the `public_only` row). **Table, `no_policy` (bn), pre-epoch in brackets:** 3.14%: baseline ~0, upside ~0, downside **0.21** (~~0.36~~), shock_recovery ~0; 3.9% (central): baseline **−0.00** (~~0.11~~), upside −0.00, downside **7.34** (~~7.87~~), shock_recovery **0.43** (~~0.38~~); 6.0%: baseline **39.26** (~~36.5~~), upside **25.40** (~~23.1~~), downside **66.87** (~~63.2~~), shock_recovery **44.72** (~~41.8~~); across the three policy arms the 6.0% range is ~~23.14-67.18~~ **25.40-71.05 B**. **The separation by the public dual:** in the cells where the public dual is zero (≤ 1e-6) EaaS value never exceeds **5.23 USD** — the same-plan solver agreement (~~7.63 USD, 4 cells~~; now **8 cells**, minimum −3.91 USD); in the **28** cells where it is positive (~~32~~) EaaS value is never below **180,067,166 USD** (~~24,090,519~~). Given a positive dual, value is monotone in it: **0 inversions** in each of demand × policy, demand × gas and gas × policy (unchanged). The log–log slope of value on dual is ~~1.79 / 1.33 / 3.78~~ **4.26 / 1.67 / 6.32** at 3.14 / 3.9 / 6.0% growth (pooled ~~1.75~~ 2.47): the dual still orders EaaS value and does not price it. The **EaaS arm's** own budget dual is zero (≤ 4.3e-16) in all 24 cells at 3.14% and 3.9% and now also at 6.0% (≤ 7.5e-16; ~~up to 0.22~~): the EaaS budget never has a positive dual here. **The central cell.** Public dual ~~0.20086369~~ **0.0**, utilisation ~~1.0000~~ **0.99896**: the public budget does **not** bind at the central cell, EaaS value there is **−0.197 USD**, ~~0.109 B (0.40% of system cost)~~ and the sentence "EaaS saves 0.11 B at the central case" is no longer true. **Why (measured, not reframed):** the DEM-2 runner does not run `capital_case = unconstrained`; it runs `capital_case = "moderate"`, i.e. 0.85 × B\* = **8,732,342,068 USD** (public solar + storage + EaaS subsidy, present value). The DEM-2 central cell is the same solve as the capital-frontier point at 0.85 × B\* (`capital_frontier_eaas_gap.csv`: EaaS value ~~109,242,943.85~~ **−0.19722** USD, equal to the dem2 cell to the last digit in both epochs). Pre-epoch the unconstrained public requirement (9,246,009,182-9,761,349,093 PV, F-22) exceeded that envelope, so the budget bound (dual 0.2009) and EaaS, which relieves it, was worth the 0.109 B cost of the binding envelope. Post-epoch the requirement is **8,723,252,062 USD**, which is **9.09 M USD below** the envelope (8,723,252,062 / 8,732,342,068 = 0.99896 = the utilisation), so the budget no longer binds and there is nothing for EaaS to relieve. The result is a knife edge: it moved because the arrears window and the declining storage capex lowered the capital requirement by about 11% while B\* and the 0.85 fraction are fixed constants. Nothing in `docs/chapters/` was touched; the Chapter 1 consequence waits on this measurement. Source: `results/dem2/arms_31d60098/{dem2_results.csv,eaas_value.csv}`, `results/capital_frontier/arms_92390a63/capital_frontier_eaas_gap.csv`.]**

| growth | baseline | upside | downside | shock_recovery |
|---|---|---|---|---|
| 3.14% | ~0 | ~0 | **0.36B** | ~0 |
| 3.9% (central) | **0.11B** | -0.00B | **7.87B** | 0.38B |
| 6.0% | 36.5B | 23.1B | ~~63.4B~~ **63.2B** | 41.8B |

At 6.0% EaaS is worth 23-63B (no_policy); across the three policy arms
the 6.0% range is 23-67B (unchanged: 23.14-67.18B).

**TASK 29 RE-RUN (2026-10-04, code `4ec43b4`, Phase B item 13).** The
EaaS side of this table (`eaas_value` = public cost minus EaaS cost) was
computed with `required_margin` **forced to 1.10** instead of the
`dscr_case`-derived **1.30**. That is a laxer version of a bankability test
only the EaaS arm takes, so the bias ran toward **overstating EaaS value**
(it made the EaaS arm cheaper than the correct cushion allows). **How much
it moved:** 11 of the 12 cells are identical at the printed precision
(including the central case: 0.109B, ~0, 7.875B, 0.378B). Exact, from the
pre-correction snapshot: only the 6.0% cells moved (10 of 36 cells across all
policy arms; EaaS cost up by at most **0.42%**, `ndc3_unconditional`, 6.0%,
downside; public cost by exactly 0). In this table: **6.0% / downside, 63.353 →
63.222B (−0.131B, −0.2%)** and shock_recovery 41.838 → 41.822B; across the
three policy arms the largest value move is −0.203B (−0.3%, 67.381 →
67.178B), all in the predicted direction. ~~EaaS solar MW at 6.0% is unchanged
in 9 of the 10 moved cells (`shock_recovery` / `no_policy`: 83,276 → 96,192 MW).~~
**[UNIDENTIFIED, TASK 30 Part D: the sentence read the EaaS MW as a result and it
is not one.** Across all 36 EaaS cells of DEM-2 the stored `solar_eaas_total_mw`
changed in **21**, by up to 15,061 MW (`organic_high`/baseline/no_policy,
79,699 → 94,760), while **total** solar changed in only 1 cell (`shock_recovery` /
`no_policy` at 6.0%, −328 MW) and the cost value moved by at most 0.2B. The
**EaaS-arm** budget dual (`budget_shadow`) is zero (at most 1.3e-15) in all 24 cells at 3.14% and 3.9%
growth (~~`budget_utilisation` 0.43-1.00: where it reaches 1.00 the budget is spent
but worth nothing at the margin, F-41's pattern~~) and positive (up to 0.22) only at 6.0%; in either
case the EaaS MW is one point in F-12's interval, not a quantity. The public arm's
dual is positive in 32 of the 36 cells, including the central one (TASK 41, 2026-10-09).
`budget_utilisation` is realised public spend over the envelope and
contains `solar_public_add`, so it is split-derived and is not
reported; the dual is not (TASK 42, 2026-10-09). The identified quantities here are total solar and cost.]** The 6.0% cases were the ones the TASK 29 subset run
did not test and where larger movement was expected (EaaS builds the most
there); it did not occur. **The conditionality result is robust to the
correction.** "EaaS saves 0.11B at the central case" is still the
sentence the project's results contain, now at 0.109B.

**Verification note.** The draft said "at 3.1% growth it is
approximately zero in every gas regime". It is zero in three regimes
and **0.36B (1.4%) in `downside`**. Small against 7.87B, but not zero.

**Reading.** ~~Taken with F-35 and F-36: EaaS is worth approximately
nothing when no constraint binds and a great deal when one does.~~
**(SUPERSEDED wording, corrected 2026-10-09, TASK 41. The public
capital envelope binds at the central cell — dual 0.20086369,
utilisation 1.0000 — so "no constraint binds" is false there. The
budget figures in the block above are the EaaS arm's dual.)**

Corrected: across the 36 cells a binding public capital envelope
separates the two outcomes exactly. In the 4 cells where the public
dual is zero, EaaS value never exceeds 7.63 USD — the measured
same-plan solver agreement (TASK 37). In the 32 where it is
positive, EaaS value is never below 24.09 million USD. Given a
binding envelope, value is monotone in its shadow price: zero
inversions in each of demand × policy, demand × gas and gas ×
policy. It is not proportional to it — the log–log slope runs from
1.33 to 3.78 across demand cases (1.34 to 3.71 with duplicate cells
removed) — so the dual orders EaaS value and does not price it. At
the central cell the envelope binds but is nearly worthless at the
margin, 0.2009 against 6.74 at the strongest cell, which is why EaaS
is worth 0.40% of system cost there. **This is the thesis's central
claim in its defensible form.** "EaaS saves 0.11B at the central
case" is a true sentence extractable from this project's own
results, so the conditionality must be stated from Chapter 1 rather
than introduced as a Chapter 5 caveat.

## F-39 — A tight capital envelope raises total emissions at zero carbon price
**NEW. Carbon-price sweep, Phase B item 5 (39 solves = 13 prices x 3
arms), `emissions_cap=None` on every solve.**

**Source.** `results/carbon_price_sweep/carbon_price_raw.csv`
(`cum_emissions`, `cum_emissions_grid`, `cum_emissions_selfgen`,
`cum_selfgen_twh`).

> **[RE-DERIVED, TASK 84, 2026-10-23 — REVERSED: 'a tight capital envelope raises total emissions at zero carbon price': at $0 `public_tight` emits exactly what `public_unconstrained` emits and builds no self-generation.** Basis: no cap in the file (envelope vs none); the cap-plus-envelope cross-check from the NDC 3.0 runs. Vintage `carbon_price_raw.csv` (39 solves, `emissions_cap=None` on every solve; there is no cap in this file). At $0/tCO2: `public_tight` total **504.1 Mt** (grid 504.1 + self-generation 0.00; ~~543.0 = 498.0 + 45.0~~), **0.00 TWh** of self-generation (~~49.83~~), cost 23.401 bn (+0.078 bn over `public_unconstrained` 23.323); `public_unconstrained` 504.1 Mt, `eaas_tight` 504.1 Mt. Total emissions are ~~38.9 Mt (7.7%)~~ **0.0 Mt** higher. The attribution is now moot: there is no emissions effect of the envelope at a zero price on the vintage model. **Where the envelope does raise emissions is under a cap:** NDC 3.0 unconditional central, full-horizon leakage **21.31% with the tight envelope against 0.10% cap alone** (`diagnostics.json::leakage`); fin3 tight row 21.3-39.0% against 0.10% unconstrained (F-31). So the entry's mechanism (capital scarcity displaces load to self-generation) is present only in combination with a binding cap, which this sweep does not contain. Basis: the file has no cap; the envelope-carrying arm is the claim, `public_unconstrained` is the envelope-free comparator. Reporting window: not recomputable from this file (the raw file stores no annual total; `selfgen_by_year.csv` has self-generation only), and self-generation is 0.0 in `public_tight` at every price, so the window and horizon statements coincide for the self-generation term. Source: `results/carbon_price_sweep/arms_1d68be49/carbon_price_raw.csv`, `results/carbon_price_sweep/arms_1d68be49/selfgen_by_year.csv`, `results/ndc/ndc3_unconditional_ae31b29c/diagnostics.json`, `results/ndc/ndc3_unconditional_60585e1f/diagnostics.json`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; conclusion and attribution hold.** Post-epoch `results/carbon_price_sweep/arms_92390a63/carbon_price_raw.csv` (39 solves, `emissions_cap=None` on every solve). At $0/tCO2: `public_tight` emits ~~542.5~~ **543.0 Mt** (grid 498.0 unchanged + self-generation ~~44.5~~ 45.0) against `public_unconstrained`'s 504.1 Mt (all grid, unchanged), with ~~49.32~~ **49.83 TWh** of self-generation against 0.0; total emissions are ~~38.4 Mt (7.6%)~~ **38.9 Mt (7.7%)** higher while grid emissions are lower (498.0 vs 504.1). The arms differ only in `capital_case` (a public-capital envelope); no emissions cap exists in the sweep, so the attribution to capital scarcity, not to a cap, stands. `eaas_tight` again shows zero self-generation. Source: `results/carbon_price_sweep/arms_92390a63/carbon_price_raw.csv`.]**

At $0/tCO2: `public_tight` emits **542.5 Mt** (grid 498.0 + self-gen
44.5) against `public_unconstrained`'s **504.1 Mt** (all grid), while
carrying **49.32 TWh** of baseline self-generation against 0.0 TWh
for the unconstrained arm. Grid emissions are *lower* (498.0 vs 504.1);
total emissions are 38.4 Mt (7.6%) *higher*.

**Verification note (draft failed on mechanism).** The draft titled this
"a grid-boundary emissions cap raises total emissions" and described a
cap binding on grid emissions. **There is no emissions cap in this
sweep**: the script builds every model with `emissions_cap=None`, and
`public_tight` differs from `public_unconstrained` only in
`capital_case="tight"` (a public-capital envelope). The leakage-shaped
outcome (load to self-generation, total up) is observed, but it is
produced by capital scarcity, not by a cap. The draft's claim that this
"agrees with the dedicated leakage runner" is therefore not established
by this file; the cap-induced mechanism is F-21 and POL-1 (42.9% of
emissions outside the boundary for `ndc3_unconditional`, public arm,
`results/pol1/pol1_results.csv`). Note `eaas_tight` shows zero
self-generation: EaaS removes the effect under the same capital case.

## F-40 — Under a binding capital envelope the carbon price does no abatement
**NEW. Same file and solves as F-39.**

**Source (pre-epoch, TASK 54 Phase 1).** `results/carbon_price_sweep/carbon_price_raw.csv`, the same file as F-39; runner `19_run_carbon_price_sweep.py`, Phase B item 5, 2026-10-04; commit inferred.

> **[RE-DERIVED, TASK 84, 2026-10-23 — REVERSED: 'under a binding capital envelope the carbon price does no abatement' and 'a pure transfer': `public_tight` abates 64.1 Mt (12.7%) by $150 and the cost rise exceeds the carbon payment.** Basis: no cap in the file (envelope vs none). Vintage `carbon_price_raw.csv`. `public_tight`: total emissions **504.1 -> 440.0 Mt** from $0 to $150 (**-64.1 Mt, -12.7%**; ~~543.0 -> 541.95~~), flat at 504.1 to $20 then 479.0 ($30-40), 478.7, 477.6, 475.3, 471.1, 452.1, 440.0 ($50, 60, 75, 100, 125, 150); self-generation 0.0 at every price; 2035 solar share **12.6-18.0%** (~~36.7% at every price~~), 2045 solar share 47.8-52.2% (~~75.7%~~); NPV cost **23.401 -> 49.110 bn**. The cost increase **25.709 bn exceeds the carbon payment 23.716 bn by 1.99 bn** (~~28.366 vs 28.357, 0.03%~~): not a pure transfer. `public_unconstrained` and `eaas_tight` are identical in emissions and cost at **every** price including $150 (**152.2 Mt**, 44.137 bn; ~~136.6 / 143.0, separating at $150~~): abatement 69.8% of the $0 level against 12.7% under the tight envelope. The conclusion changes from 'the price does nothing under a binding envelope' to 'a binding envelope limits what the price achieves (12.7% against 69.8%) but does not nullify it'. Basis: no cap in this file; the envelope-carrying arm (`public_tight`) is the claim. Window: not recomputable (see F-39). Source: `results/carbon_price_sweep/arms_1d68be49/carbon_price_raw.csv`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; conclusion and attribution hold.** Post-epoch, same file as F-39. `public_tight`: 2035 mix **36.7% solar / 63.3% gas at every price from $0 to $150** (2030: 10.5/89.5; 2045: 75.7% solar throughout, was 75.8–75.9%); total emissions ~~542.5 → 541.7~~ **543.0 → 541.95 Mt**; grid emissions constant at 498.017 Mt; NPV cost ~~31.6 → 60.0~~ **30.15 → 58.52 bn**, approximately linear. The cost increase equals the carbon payment: ~~28.363 vs 28.358~~ **28.366 vs 28.357 bn** at $150 (0.03%), a pure transfer. `public_unconstrained` / `eaas_tight` emissions at $150 ~~147.5 / 163.9~~ **136.6 / 143.0 Mt** (from 504.1 Mt at $0). They are identical in emissions and cost at every price from $0 to $125 and **separate only at $150**, by ~~16.4~~ **6.4 Mt** (143.0 vs 136.6; cost 44.581 vs 44.550 bn), whereas pre-epoch they separated from $125. The attribution (the binding constraint is the capital envelope, not an emissions cap) holds: the arms differ only in `capital_case`. Source: `results/carbon_price_sweep/arms_92390a63/carbon_price_raw.csv`.]**

`public_tight`: 2035 generation mix is **36.7% solar / 63.3% gas at every
price from $0 to $150** (2030: 10.5/89.5; 2045: 75.8-75.9% solar
throughout); total emissions move from 542.5 to **541.7 Mt**; grid
emissions are constant at 498.0 Mt; NPV cost rises from **$31.6B to
$60.0B**, approximately linearly. The cost increase equals the carbon
payment to four significant figures ($28.363B vs $28.358B at $150), so
the price acts as a pure transfer. In contrast `public_unconstrained`
and `eaas_tight` cut emissions from 504.1 to 147.5 / ~~154.3~~ **163.9** Mt
over the same range.

**TASK 29 RE-RUN (2026-10-04, code `4ec43b4`, Phase B item 5).** The struck
`eaas_tight` endpoint was computed with `required_margin` forced to 1.10
instead of the `dscr_case`-derived 1.30, a laxer version of a test only the
EaaS arm takes, so the bias ran toward **overstating EaaS abatement** (it let
`eaas_tight` track the unconstrained arm further than the correct cushion
allows). **Moved: 154.30 → 163.95 Mt at $150 (+9.64 Mt, +6.2%; cost +0.13B), and 179.92 → 186.29 Mt at $125 (+6.37 Mt)**; no other price moved by more than 0.01 Mt; the
abatement from the $0 level (504.1 Mt) is 340.2 Mt, not 349.8 Mt (−2.8%).
`eaas_tight` and `public_unconstrained` are identical in emissions and
(to under 30 USD) in cost at every price from $0 to $100, and **separate
only at the top of the price range**: $125 by 6.4 Mt (+$21M) and $150 by
16.4 Mt (163.9 against 147.5 Mt, +$190M). Before the correction the $150
gap was 6.8 Mt. The
`public_tight` figures, the "pure transfer" result, F-39's self-generation
numbers and `eaas_tight`'s zero self-generation did not change.

**Reading.** Once a quantity constraint binds, the price instrument is a
transfer. **Verification note:** the binding quantity constraint is the
capital envelope (not an emissions cap as in the draft); the
instrument-redundancy conclusion stands for that constraint. Whether the
same holds under an emissions cap is not tested by this sweep.

## F-41 — EaaS needs a smaller public capital budget before the constraint materially binds
~~F-41 — EaaS needs a smaller concessional envelope before the constraint materially binds~~ **(SUPERSEDED wording, TASK 53 2.9, applied TASK 55, 2026-10-20: the swept and measured quantity is the public solar budget, `public_solar_budget_npv`, a present-value constraint on public solar and storage capex plus the EaaS subsidy, expressed as a fraction of B\*; its dual is `public_budget_shadow_usd_per_usd`, present-value cost per present-value budget dollar. The concessional envelope (`concessional_envelope_usd`, an undiscounted stock) is held at `env_dares` at every point and is not what is varied.)**
~~F-41 — EaaS needs the concessional envelope at a smaller size than public financing~~ **(SUPERSEDED title, corrected 2026-10-10, TASK 44. The capital frontier classifies binding by slack at 1% of the envelope (`BUDGET_BINDING_REL_TOL = 0.01`, `scripts/28_run_capital_frontier.py`), so EaaS points at 0.4–0.7 are classified binding on duals of order 1e-16; the envelope binds materially only at or below 0.35, where the dual is 0.079159 (`capital_frontier_raw.csv`).)**
**NEW. Capital frontier, Phase B item 21 (38 solves = 19 points x 2
regimes).**

**Source.** `results/capital_frontier/capital_frontier_endpoints.json`,
`capital_frontier_raw.csv` (`public_budget_state`,
`public_budget_dual_usd_per_usd`).

> **[RE-DERIVED, TASK 84, 2026-10-23 — MOVED: the budget fractions at which each regime stops binding; counts.** Basis: uncapped; public budget varied. Vintage `capital_frontier_raw.csv`: the runner's tolerance label is `binding` at budget fraction <= ~~0.85~~ **0.50** (`traditional`) and <= ~~0.60~~ **0.40** (`eaas`); binding / slack / absent counts ~~14/4/1 and 12/6/1~~ **11/7/1 and 9/9/1**. **By the dual:** `eaas` has a positive dual at fractions <= ~~0.25~~ **0.15** (0.75958 at 0.15; zero from 0.20 with cost 23,323,068,348 = the optimum); from 0.20 to 0.40 the label is `binding` on duals of zero to 1e-16: the budget is spent and worth nothing at the margin. `traditional` has a positive dual through ~~0.70~~ **0.50** (0.39919, cost 23.4011 bn, 0.33% above the optimum) and zero from 0.60. The dual-based comparison is ~~0.25 vs 0.70~~ **0.15 vs 0.50** (EaaS needs about 30% of the traditional budget fraction; was 36%). The unconstrained cost is ~~24,986,620,95x~~ 23,323,068,34x. The M-3 pattern (binding is not material) is present. Source: `results/capital_frontier/arms_1d68be49/capital_frontier_raw.csv`, `results/capital_frontier/arms_1d68be49/capital_frontier_endpoints.json`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; conclusion holds; title and wording changed per TASK 53 §2.9.** The swept and measured quantity is the **public solar budget** (`public_solar_budget_npv`, a present-value constraint on public solar and storage capex plus the EaaS subsidy, as a fraction of B\*), whose dual is `public_budget_shadow_usd_per_usd` (PV per PV, no divisor); the concessional envelope (`concessional_envelope_usd`, an undiscounted stock) is held at `env_dares` at every point. The columns `envelope_fraction` and `envelope_usd` keep their names and hold the budget. Post-epoch `results/capital_frontier/arms_92390a63/capital_frontier_raw.csv`, `capital_frontier_endpoints.json`: the runner's tolerance label (slack within 1% of the budget) is `binding` at budget fraction ≤ ~~0.9~~ **0.85** (`traditional`) and ≤ ~~0.7~~ **0.6** (`eaas`); binding / slack / absent counts ~~15/3/1 and 13/5/1~~ **14/4/1 and 12/6/1**, reproduced from the file. **By the dual** (the definition used in F-22): `eaas` has a positive dual at fractions ≤ **0.25** (0.38898 at 0.25; ~~0.079159 at 0.35~~ the 0.35 dual is 0 and cost equals the optimum 24,986,620,95x); from 0.30 to 0.60 the label is `binding` on duals of 1.1e-15, 0, 1.8e-17, 6.7e-17, 0, 2.1e-16 (~~1e-16 to 1e-18~~): the budget is spent in full and is worth nothing at the margin, at a cost identical to the unconstrained run; `traditional` has a positive dual through **0.70** (1.0193, cost 25.966 bn, 3.9% above the optimum; ~~0.9, dual 0.10, cost 0.12% above~~) and zero from 0.85. The 0.60 vs 0.70 comparison is a tolerance-classification artefact; the dual-based comparison is **0.25 vs 0.70** (~~0.35 vs 0.9~~). This is the M-3 pattern (binding is not material). The unconstrained cost is ~~27,317,670,15x~~ 24,986,620,95x. Source: as stated.]**

The frontier is classified binding at budget fraction (of B\*; the column is still named `envelope_fraction`) <= 0.9
(`traditional`) and `<= 0.7` (`eaas`); binding/slack/absent counts
15/3/1 and 13/5/1. **These reproduce exactly** from the endpoints file.

**Verification note (the interpretation, not the numbers).** The EaaS
"binding" classification at fractions 0.4-0.7 has a dual of 1e-16 to
1e-18 and a cost identical to the unconstrained run
(27,317,670,15x): the budget is spent in full but is **worth nothing at
the margin**. By dual, EaaS binds materially only at fractions <= 0.35
(dual 0.079 at 0.35); `traditional` binds materially to 0.9 (dual 0.10,
cost still 0.12% above unconstrained). The 0.7 vs 0.9 comparison is a
tolerance-classification artefact; the dual-based comparison is **0.35
vs 0.9**. This is the M-3 pattern (binding is not material).

## F-42 — The value of the stochastic solution, bounded
**NEW. VSS bound, Phase B item 18 (consumes item 17 leg B, run
`5d29ee61`, `--fix-gas free`; 100 draws per leg).**

**Source.** `results/vss_bound/vss_bound__76f4aea6.json`. All five guards
pass. `gas_add` under `--fix-gas free` is unbounded (`gas_build_case=
gas_unconstrained`), so no band needs replicating in `RP_partial`.
`ECP_free` = **$40.20B**, `WS` = **$32.87B** (equals the Monte Carlo
"Matched to 30_ central" mean, F-36), `bound_upper` = **$7.33B** =
**18.2%** of `ECP_free`. `vss_partial` lies in [0, 7.33e9].

> **[RE-DERIVED, TASK 84, 2026-10-23 — WEAKENED: the bound is 67% larger (12.089 bn, 33.3% of ECP_free); both legs are now at `capital_case=unconstrained`, which changes the configuration statement.** Basis: uncapped (no cap); legs at capital_case=unconstrained. **Correspondence confirmed arm for arm before re-derivation.** The blended F-42 source `vss_bound__d6f19a23.json` has stage 1 `summary__8fc2e490` (`--fix-gas free`); the vintage `vss_bound__9e2eda70.json` has stage 1 `summary__19490865` (`--fix-gas free`), per `results/vss_bound/INDEX.md`. A field-by-field comparison of the two summaries' `standard_arms`, `own_args` and `stage1_scenario_fields`: **the only differences are `capital_case` (blended `moderate`, vintage `unconstrained`: the TASK 81 Batch 4 change to `30_` and `17_`) and the vintage token in the arm tag**; draws sha256 `1ef6055b...` identical, 100 draws, seed 42, `gas_build_case=gas_unconstrained`, no band. Vintage `vss_bound__9e2eda70.json` (all five guards pass): `ECP_free` ~~$37.85B~~ **$36.34B** (36,342,551,606.37), `WS` ~~$30.60B~~ **$24.25B** (24,253,479,785.88; equals the Monte Carlo 'Matched to 30_ central' mean, F-36), `bound_upper` ~~$7.24B~~ **$12.09B** (12,089,071,820.50) = ~~19.1%~~ **33.26%** of `ECP_free`; `vss_partial` in [0, 1.2089e10]. **Median alongside the mean:** per-draw ECP costs median ~~27.50~~ **25.71 bn** against mean 36.34 (mean/median ~~1.38~~ **1.41**); P10 / P90 ~~22.17 / 72.98~~ **20.67 / 70.30**; WS median ~~25.76~~ **23.76** against mean 24.25. **Configuration:** both legs run `capital_case=unconstrained` (~~moderate, 0.85 x B*~~), the frozen first-stage plan is the unconstrained anchor (deterministic base 23,323,068,340.46 = `summary__661b03c1`, difference 0.0 USD). The statement 'both legs run under a public capital budget' is false and the first stage is exactly the anchor. The bound is wider; it still contains both the VSS and the EVPI. Source: `results/vss_bound/vss_bound__9e2eda70.json`, `results/stochastic_redispatch/summary__19490865.json`, `results/stochastic_redispatch/draws__19490865.jsonl`, `results/monte_carlo/arms_f30c5990/monte_carlo_summary.csv`, `results/vss_bound/INDEX.md`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; conclusion holds, configuration statement changes.** Post-epoch `results/vss_bound/vss_bound__d6f19a23.json` (all five guards pass; stage 1 `summary__8fc2e490`, `--fix-gas free`; same draws sha256 as the pre-epoch run). `ECP_free` ~~$40.20B~~ **$37.85B** (37,846,482,446), `WS` ~~$32.87B~~ **$30.60B** (30,602,383,956), `bound_upper` ~~$7.33B~~ **$7.24B** (7,244,098,489) = ~~18.2%~~ **19.1%** of `ECP_free`; `vss_partial` ∈ [0, 7.244e9]. **Median alongside the mean:** the per-draw ECP costs have median ~~29.85~~ **27.50 bn** against mean ~~40.20~~ **37.85** (mean/median ~~1.35~~ **1.38**: the mean sits in the right tail); the WS case has median ~~28.20~~ **25.76** against mean 30.60. P10 / P90 of the ECP draws ~~23.73 / 76.41~~ **22.17 / 72.98**. **Configuration.** Both legs run `capital_case = "moderate"` (0.85 × B\*). Pre-epoch that was a binding public budget and the frozen first-stage plan diverged from the unconstrained anchor (solar from 2043, 11 divergent years, no storage built after 2048; deterministic base ~~27,426,913,099~~ = anchor + 109,242,943 USD). **Post-epoch the frozen first-stage plan is the unconstrained anchor:** solar 39,030.1 MW and storage 63,056.3 MWh equal to `summary__6046f759`, no divergent year, storage built 2035-2054, deterministic base 24,986,620,955.93 (anchor + 6.7 USD). The moderate budget does not bind at the deterministic centre (F-38); whether it binds in individual high-demand draws was not measured, so "both legs run under a binding public capital budget" can no longer be stated for the central plan. Source: `results/vss_bound/vss_bound__d6f19a23.json`, `results/stochastic_redispatch/{draws,summary}__8fc2e490`.]**

**Reading, which is part of the finding.** The first stage is **solar
and storage only**, with gas and dispatch as recourse. The upper bound
contains **both** the value of the stochastic solution and the expected
value of perfect information, and the two cannot be separated without
solving the recourse problem. The quantity frozen in `30_` is the
**expected cost of the central plan**; the gas regime is categorical and
has no mean, so no expected-value instance exists for that dimension.

## F-43 — Under public financing the cost-emissions curve bends backward: past a point, tightening the grid cap raises total emissions
**NEW. Pareto frontier, Phase B item 20, re-run 2026-10-04 at `cded909`
(4 arms x [1 uncapped anchor + 25 grid points, cap_fraction 1.00 to 0.10 +
6 tight-end points, 0.08 to 0.01] = 128 solves, 0 infeasible).**

**Relation to F-21.** F-21 reported this inversion first, on one pair of
cap paths (a post-2035 declining trajectory against the flat hold: national
emissions +8.2%, cost +18.6% at the current epoch; TASK 48 B.3 — the frozen +7.7% / +19.9% is superseded in F-21). F-43 is the same mechanism measured on a
continuous instrument, not an independent discovery. What it adds, and a
single comparison cannot show: (1) the **turning point** (under public
financing with moderate capital, total emissions are lowest at
`cap_fraction` 0.8125 and rise below it; with tight capital there is no
falling branch); (2) the **financing contrast** (the EaaS arms do not turn
materially); (3) the **non-dominated counts** (7 and 1 of 32 under public
financing). F-21's 40% is its trajectory's 2045 gap, not a flat tightening:
on F-43's flat caps, public-moderate emissions at a 40% tighter cap
(`cap_fraction` 0.60) are about 9% *below* their `cap_fraction`-1.0 value,
and exceed it only below about 0.45 (the uncapped solve's, below about
0.40). F-21 reports public financing only; it was re-run in Phase C Group 2
(2026-10-10) and now carries the current pair (TASK 48 B.3). Quote this
entry's quantities for the continuous instrument and F-21's for the
trajectory.

**Source.** `results/pareto_frontier/pareto_raw.csv`
(`cum_emissions`, `leakage_share`, `npv_cost`, `cumulative_cap_tco2`),
`pareto_dominated_points.csv`, `pareto_endpoints.json`. Full per-arm series
(all 32 points per arm): `docs/task_log_29_2_findings_update.md`.

> **[RE-DERIVED, TASK 84, 2026-10-23 — STRENGTHENED: the backward bend now exists in both public arms (title holds more widely); the turning point moves 0.8125 -> 0.625; the clause 'with tight capital there is no falling branch' is REVERSED.** Basis: flat caps (cap_fraction), arms carry tight/moderate envelope by definition. Vintage `pareto_raw.csv` (4 arms x 32 points = 128 solves, 0 infeasible), total emissions over 2025-2054 (Mt) unless stated. **Full frontier, all four arms** (cap_fraction: total / cost bn; anchor = uncapped solve 504.1 Mt at 23.323 bn (tight 23.401); cap 1.0 = 480.9 Mt): *public_only_moderate* 1.0: 480.9 / 23.60; 0.8125: 402.8 / 24.8; **0.625: 322.1 / 30.20 (minimum)**; 0.4375: 376.2 / 61.5; 0.25: 518.5 / 102.9; 0.08: 659.7 / 143.0; 0.01: 718.4 / 159.73. *public_only_tight* 1.0: 480.9 / 24.03; **0.8125: 420.6 / 27.62 (minimum)**; 0.625: 505.0 / 46.4; 0.4375: 670.0 / 85.1; 0.25: 818.0 / 129.5; 0.08: 959.4 / 170.1; 0.01: 1,014.5 / 186.85. *eaas_moderate* 1.0: 480.9 / 23.60; 0.8125: 402.8; 0.625: 315.1; 0.4375: 220.5; 0.25: 117.7; 0.08: 97.0; **0.01: 90.8 / 72.92 (minimum, monotone to the floor)**. *eaas_tight* 1.0: 480.9 / 23.60; 0.8125: 402.8; 0.625: 315.1; 0.4375: 218.9; 0.25: 170.5; 0.08: 156.1; **0.02: 153.2 / 89.66 (minimum)**; 0.01: 154.4 / 91.36. **Turning point, with grid bracket:** `public_only_moderate` minimum **322.1 Mt at cap_fraction 0.625** (grid neighbours 0.6625: 334.0 and 0.5875: 336.2; ~~404.6 at 0.8125~~); `public_only_tight` minimum **420.6 Mt at 0.8125** (neighbours 0.85: 424.1 and 0.775: 424.5). Grid step 0.0375. The moderate curve returns to its cap-1.0 level (480.9) between 0.325 and 0.2875, and exceeds the uncapped solve (504.1) at about 0.25 (~~about 0.40~~); `public_only_tight` returns above 480.9 and 504.1 at 0.625 (505.0). **Non-dominated points on the total axis (runner's `pareto_dominated_points.csv`, anchor counted as dominator): of 32: public moderate 12 (~~7~~), public tight 7 (~~1~~), eaas moderate 30 (~~31/32~~), eaas tight 30 (~~28/31~~).** In (cost, grid) the frontier is still clean. **Second clause: REVERSED.** The entry says `public_only_tight` 'never descends: its cap-1.0 point (550.0 Mt) is already above the uncapped solve'. On the vintage model its cap-1.0 point (480.9 Mt) is **below** the uncapped solve (504.1) and the curve descends 60.3 Mt to 420.6 Mt before rising to 1,014.5 Mt. All four arms are `endpoints_degenerate` (tight end unlocated: feasible to cap_fraction 0.01), as in the blended file. **Reporting window (2025-2045)** (recomputed from the annual columns): anchor 338.2 Mt, cap 1.0 329.7 Mt; minima: public moderate **213.2 Mt at 0.5875** (bracket 0.55-0.625), public tight 252.6 at 0.625 (0.5875-0.6625), eaas tight 122.9 at **0.2875** (0.25-0.325; rises to 141.6 at 0.01, so EaaS tight also turns inside the window), eaas moderate 74.5 at 0.175 (rises to 80.4 at 0.01); public moderate returns to the cap-1.0 window level and exceeds the anchor at 0.175, public tight at 0.4. What the entry would have to say: under public financing the curve bends backward at both capital levels (turning at 0.625 moderate, 0.8125 tight, over the horizon; 0.5875 and 0.625 in the window); the EaaS arms are essentially monotone over the horizon (the window shows a shallow turn in the EaaS arms). Basis: flat caps at fractions of baseline, not NDC 3.0; `public_only_*` carry the moderate/tight envelope as arm definitions, so the entry concerns cap plus envelope. Source: `results/pareto_frontier/arms_1d68be49/pareto_raw.csv`, `results/pareto_frontier/arms_1d68be49/pareto_dominated_points.csv`, `results/pareto_frontier/arms_1d68be49/pareto_endpoints.json`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; conclusion holds.** Post-epoch `results/pareto_frontier/arms_92390a63/pareto_raw.csv` (Phase B item 20, 2026-10-20 00:44; 4 arms × 32 points = 128 solves). Method check: the same non-dominance, turning-point and decomposition code reproduces the pre-epoch counts (7, 1, 31, 28) and extremes. **Turning point:** `public_only_moderate` reaches its minimum total emissions at `cap_fraction` **0.8125** (unchanged), ~~411.4~~ **404.6 Mt**, and returns to its cap-1.0 level (480.9 Mt) at about ~~0.4375~~ **0.40** and exceeds the uncapped solve (504.1 Mt) at about 0.3625 (unchanged); `public_only_tight` never descends: cap-1.0 total ~~550.0~~ **544.6 Mt** is above the anchor ~~542.5~~ **543.0 Mt**. **Non-dominated points on total emissions (of 32):** public moderate **7** (unchanged), public tight **1** (the anchor; unchanged), EaaS moderate ~~31~~ **32**, EaaS tight ~~28~~ **31**. **Self-generation at cap_fraction 0.01:** public tight **1,010.8** (unchanged), public moderate ~~725.8~~ **715.1**, EaaS tight ~~181.2~~ **150.1**, EaaS moderate ~~107.9~~ **85.8** Mt; the financing contrast is ×~~5.6~~ **6.7** (tight: 1,010.8 / 150.1) and ×~~6.7~~ **8.3** (moderate: 715.1 / 85.8), the envelope contrast ×1.4 (public) and ×1.7 (EaaS), unchanged. **Totals, cap 1.0 → 0.2125:** public moderate ~~+122.6~~ **+110.6 Mt**, public tight ~~+327.9~~ **+333.3**, EaaS moderate ~~−342.8~~ **−365.3**, EaaS tight ~~−284.2~~ **−306.3** (grid falls 373.8 / 367.7 / 379.1 / 375.0). **Minima:** EaaS moderate ~~112.7 at 0.02~~ **90.9 Mt at 0.01** (monotone to the floor), EaaS tight ~~185.7 at 0.03~~ **155.1 at 0.02**. Selected rows (cost bn / total / grid / self-generation, Mt): public moderate anchor 24.99 / 504.1 / 504.1 / 0.0; 1.0 25.21 / 480.9 / 480.9 / 0.0; 0.8125 28.66 / 404.6 / 399.3 / 5.3; 0.5125 59.48 / 429.1 / 258.4 / 170.7; 0.2125 116.82 / 591.5 / 107.1 / 484.4; 0.01 159.93 / 720.2 / 5.0 / 715.1. Public tight anchor 30.15 / 543.0 / 498.0 / 45.0; 1.0 32.23 / 544.6 / 474.8 / 69.8; 0.01 187.05 / 1,015.9 / 5.0 / 1,010.8. The F-21 pair the entry compares itself with is now cost +19.2%, national emissions +5.1% (F-21). Source: as stated.]**

**Three facts established before any mechanism was written (the F-39/F-40
misattribution is the reason for the order).**
1. **What the emissions column is.** `cum_emissions` is **TOTAL** emissions,
   grid plus self-generation: `scripts/18_run_pareto_frontier.py:569` sums
   `emissions_tco2_by_year`, which is `m.emissions_by_year`
   (`src/optimize_experiments.py:3681`), and the builder defines that as
   `emissions_grid_only_by_year + emissions_selfgen_by_year`
   (`src/optimize_model_sliced.py:1893-1898`), always both. It is not capped
   grid emissions.
2. **What the cap constrains.** `m.emissions_capped_by_year`, which is
   **grid-only** emissions unless `selfgen_in_emissions_cap` is set (it is
   not in this runner; the builder's own comment: the cap attaches to the
   NDC commitment boundary, the carbon price and the report to the total).
   Checked on the data: wherever the cap binds in all 30 years (cap_fraction
   <= 0.625 in the EaaS arms, <= 0.55 in the public arms), the grid share of
   the reported total, `(1 - leakage_share) x cum_emissions`, equals the
   cumulative cap to <1.1e-8 relative in all 75 such rows. The split is
   therefore exact, not estimated.
3. **What self-generation does.** Self-generation emissions
   (`leakage_share x cum_emissions`) **rise monotonically as the cap
   tightens in all four arms**, from 0 (the three arms that start with no
   self-generation) to 107.9 (EaaS moderate), 181.2 (EaaS tight), 725.8
   (public moderate) and 1,010.8 (public tight) Mt at cap_fraction 0.01.

| arm | cap_fraction | cost bn | total Mt | grid Mt | self-gen Mt |
|---|---|---|---|---|---|
| public_only_moderate | anchor (uncapped) | 27.43 | 504.1 | 504.1 | 0.0 |
| | 1.0000 | 27.72 | 480.9 | 480.9 | 0.0 |
| | 0.8125 | 31.12 | **411.4 (minimum)** | 399.3 | 12.1 |
| | 0.5125 | 62.90 | 463.0 | 258.4 | 204.7 |
| | 0.2125 | 119.96 | 603.5 | 107.1 | 496.4 |
| | 0.0100 | 162.97 | 730.9 | 5.0 | 725.8 |
| public_only_tight | anchor (uncapped) | 31.62 | **542.5 (minimum)** | 498.0 | 44.5 |
| | 1.0000 | 33.65 | 550.0 | 474.8 | 75.2 |
| | 0.5125 | 82.95 | 690.2 | 258.4 | 431.8 |
| | 0.2125 | 144.16 | 877.9 | 107.1 | 770.8 |
| | 0.0100 | 187.94 | 1,015.9 | 5.0 | 1,010.8 |
| eaas_moderate | 1.0000 | 27.49 | 480.9 | 480.9 | 0.0 |
| | 0.5125 | 34.38 | 258.4 | 258.4 | 0.0 |
| | 0.2125 | 51.92 | 138.1 | 107.1 | 31.0 |
| | 0.0200 | 80.11 | **112.7 (minimum)** | 10.1 | 102.6 |
| | 0.0100 | 81.79 | 112.9 | 5.0 | 107.9 |
| eaas_tight | 1.0000 | 27.49 | 480.9 | 480.9 | 0.0 |
| | 0.5125 | 35.04 | 258.4 | 258.4 | 0.0 |
| | 0.2125 | 69.53 | 196.7 | 107.1 | 89.6 |
| | 0.0300 | 97.37 | **185.7 (minimum)** | 15.1 | 170.5 |
| | 0.0100 | 100.80 | 186.3 | 5.0 | 181.2 |

(Grid and self-generation columns are `(1 - leakage_share)` and
`leakage_share` times `cum_emissions`; the `eaas` anchors and cap-1.0 points
carry no self-generation.)

**Result.** Along the cap range 1.00 to 0.2125 the **grid** quantity falls
by 373.8 Mt in three arms and 367.7 Mt in public tight, but total emissions change by
**+122.6 Mt (public moderate), +327.9 Mt (public tight, from its cap-1.0
point), -342.8 Mt (EaaS moderate) and -284.2 Mt (EaaS tight)**. In the public
arms the rise in self-generation (+496.4 and +695.6 Mt) exceeds the fall in
the capped quantity; in the EaaS arms it does not (+31.0 and +89.6 Mt). Cost
rises throughout in all four arms. So the curve in (cost, **total** emissions)
bends backward in the public arms: `public_only_moderate` reaches its minimum
total emissions at cap_fraction 0.8125 (411.4 Mt) and then rises to 603.5 Mt
at 0.2125 and 730.9 Mt at 0.01; `public_only_tight` **never descends**: its
cap-1.0 point (550.0 Mt) is already above the uncapped solve (542.5 Mt) and it
rises to 877.9 and 1,015.9 Mt. The EaaS arms are essentially monotone (minimum
at 0.02 and 0.03; the largest reversals are 2.4 Mt in EaaS tight at 0.4375 and
0.2 Mt in EaaS moderate at 0.01, about 1% or less of the point they follow).
**Two different quantities, both of interest and not to be confused:** *where the
public-moderate curve turns* is `cap_fraction` **0.8125**, its minimum of
411.4 Mt; *where it crosses back* is about **0.45**, at which total emissions
return to their `cap_fraction`-1.0 level (480.9 Mt), and about **0.40**, at which
they exceed the uncapped solve's (504.1 Mt). (A draft of the chapter edits gave
0.45 as the turning point; that figure is the first crossing, not the minimum, and
was never in this entry as a turning point.) (Log labels of "20%" on the
tables are cap_fraction **0.2125**; the grid has no 0.20 point.)

**On the axis the cap constrains the frontier is clean.** In (cost, grid
emissions) all 32 points of every arm are non-dominated. The backward bend is
a property of the **total** (physical) emissions axis alone.

**Non-dominated points on the total-emissions axis** (32 points per arm = the
anchor + 31 capped points; the runner's own check, `pareto_dominated_points.
csv`, compares the capped points only):

| arm | non-dominated incl. anchor (of 32) | non-dominated among the 31 capped points (runner's check) | dominated capped points | which |
|---|---|---|---|---|
| eaas_moderate | **31** | 30 | 1 | 0.01 |
| eaas_tight | **28** | 27 | 4 | 0.4375, 0.1375, 0.02, 0.01 |
| public_only_moderate | **7** | 6 | 25 | every cap_fraction below 0.8125 |
| public_only_tight | **1** (the anchor) | 1 (cap 1.0) | 30 | the whole capped range is dominated by the uncapped solve; cap 1.0 survives only if the anchor is left out |

**Consequence for Chapter 4.** In the public arms ~~the tight half (public
moderate: cap_fraction 0.8125 down to 0.01; public tight: all of it)~~ **(SUPERSEDED
wording: "the tight half" understates it)** the dominated region is **not a frontier
over total emissions**. In public-moderate **25 of the 31 capped points are
dominated, every point tighter than 0.8125** (7 of 32 non-dominated including the
anchor: the anchor and caps 1.00 to 0.8125), about four fifths of the capped
points; in public-tight **all 31 capped
points are dominated by the uncapped solve and 1 of 32 is non-dominated, the anchor**.
Every such point is beaten on both cost and emissions by a looser one. Until TASK 30 Part B the runner drew every
`on_frontier` point (all 124 capped points; the flag means "cap tighter than
the anchor's own peak", not "non-dominated") as one solid line, and printed
that dominated points were "a solver tolerance issue or a non-binding cap",
which does not describe these. **Fixed in code (TASK 30 Part B):** the figure
is now two panels, grid-only emissions (all 32 points of every arm
non-dominated) and total emissions (non-dominated solid, dominated hollow and
unconnected), and the note says the points are genuinely dominated on total
emissions. The **stored** `pareto_frontier.png/.pdf` is the old figure until
item 20 is re-run (or `scripts/redraw_pareto_figure.py` is run); do not use it.
`pareto_dominated_points.csv` now also counts the anchor as a dominator, so
public tight lists 31 dominated points, not 30.

**Mechanism: what is established and what is not.** *Established
(accounting):* total = capped + self-generation; the cap bounds only the
first; self-generation rises as the cap tightens in every arm; in the public
arms it rises by more than the capped quantity falls. *Established (ordering,
from this file):* **the substitution into self-generation is ordered by
financing structure first and envelope level second.** At cap_fraction 0.01
self-generation is 1,010.8 (public tight), 725.8 (public moderate), 181.2
(EaaS tight) and 107.9 (EaaS moderate) Mt. Changing the financing structure at
a fixed envelope multiplies it by 5.6 (tight: 1,010.8 / 181.2) or 6.7
(moderate: 725.8 / 107.9); loosening the envelope at a fixed financing
structure divides it by only 1.4 (public: 1,010.8 / 725.8) or 1.7 (EaaS: 181.2 /
107.9). `eaas_tight` carries the same tight public budget (5.14 bn NPV) as
`public_only_tight` and leaks just under a quarter of what `public_only_
moderate` does on a looser one (8.73 bn). Public solar capacity falls as the
cap tightens (moderate 40.6 → 30.2 GW, tight 25.3 → 16.5 GW) while **total** solar in the EaaS arms
rises (41.0 → 59.2 and 58.2 GW; `solar_total_mw` is total, which is identified;
the EaaS/public split inside it is not, F-12, and is not what this sentence
claims; in the public arms all solar is public, EaaS being disabled). (Earlier
wording said "EaaS solar rises"; that mislabelled a total as an EaaS quantity.
Corrected 2026-10-08.) *Not isolated:* **why.** "Capital scarcity" is
no longer the reading this file supports on its own: an ordering this lopsided
toward financing structure is not what an envelope-level effect alone would
give. Two cautions on even that: (i) the two factors are not independent
here, because the EaaS channel *is* how privately financed solar (outside the
public budget, which prices only public solar, storage and the EaaS subsidy)
enters the model, so "same envelope" is true of the public-budget parameter and
not of the capital available to the arm; (ii) the capital-envelope duals were
not read at these points when this was written. **Reading: consistent with, not
isolated.** The runner now records both capital duals at every point
(TASK 30 Part A, pending the item 20 re-run), and the test that distinguishes
the readings is stated in advance: **if the public-budget dual is comparable
between `eaas_tight` and `public_only_tight` while their self-generation
differs by roughly a factor of five, capital scarcity is not sufficient and the
financing structure is doing the work.** This is the mechanism F-39
identified at a zero carbon price and F-31/F-21 under a cap, with the same
caveat.

**Why it is visible, stated against the brief.** It was suggested that the
bend is visible only because TASK 26 Part B anchored the endpoints and
extended the tight end. **That is not supported.** The original grid,
`linspace(1.0, 0.10, 25)`, is unchanged (Part B kept its 100 original inputs),
the public arms carry no margin (TASK 29 regression: bit-identical in the
cells tested apart from two at 1e-10), and the bend is already complete inside that range: `public_
only_moderate` turns at 0.8125, `public_only_tight` is rising from its first
point, and both reach 674.4 and 957.8 Mt at 0.10. (Verified against the
pre-correction snapshot `/home/zeus/thesis-snapshots/results_pre_margin_fix/
pareto_frontier/pareto_raw.csv`: all 100 original grid points of both public
arms have cost and emissions differences of exactly 0.) What Part B added is real but
different: the **anchor** shows that the public-tight curve starts above the
unconstrained solve (so its whole capped range is dominated, which no grid
point could show), and the **extension** to 0.01 lengthens the bend by 56.5 Mt
(public moderate, 674.4 → 730.9) and 58.1 Mt (public tight, 957.8 → 1,015.9).
A frontier truncated at 0.10 would have shown the bend; it would not have shown
that public tight has no descending branch at all.

**Scope.** Four arms, one demand case, `gas_avail_nameplate`. **No before-and-after
comparison exists for the four anchors or for the 0.08 to 0.01 range (six
extension points per arm, 24 points).** The pre-correction snapshot
(`/home/zeus/thesis-snapshots/results_pre_margin_fix/`) holds only the 100
original grid points (cap_fraction 1.0 to 0.10); the anchors and the extension
were added by TASK 26 Part B and were first solved under the corrected margin.
Every statement in this entry about how the correction moved the EaaS curves
is confined to 1.0 to 0.10; for the public arms (no margin) the 0.08-0.01
points are unaffected by the correction by construction, for the EaaS arms
they are simply not comparable. `tight_end_
located` is false for every arm (feasible down to 0.01, the grid floor), so
the right-hand ends are floors, not limits (L72). Cost and emissions here are
the corrected-margin values (TASK 29); the public arms carry no margin and are
unchanged to 1e-10.

---

# Section 7 — Method findings

## M-1 — Midstream cost per unit deliverability (post-hoc)
**METHOD, not a result. Unchanged.**

**Source (pre-epoch, TASK 54 Phase 1).** no result file; arithmetic on the GMP arm and cost inputs (22 bn pipelines, 60 bn value chain, 102.5 TWh_th/yr). Input documents not traced here.

> **[RE-DERIVED, TASK 84, 2026-10-23 — UNCHANGED: arithmetic reproduced; the 22 bn and 60 bn numerators are untraced inputs.** Basis: input arithmetic. 22 bn / 102.5 = **214.6** USD-m per TWh_th/yr (entry 215); 60 bn / 102.5 = **585.4** (585). Implied midstream investment at I = 0.5 (51.2 TWh_th): 11.0-30.0 bn; at I = 1.0 (102.5): 22.0-60.0 bn; against the 750 M envelope: **14.7 to 40 times** at I = 0.5 (entry '15 to 40'), 29 to 80 at I = 1.0. 'Meaningful for uncapped pathways only (F-9)': F-9 stands (identical objectives under both NDC 3.0 caps, `docs/task_log_83_capped/`). The 22 bn and 60 bn inputs and the +51.2 / +102.5 TWh_th GMP requirements were not re-traced. Source: `docs/task_log_83_capped/ndc3_conditional_noenvelope_capped_arms.csv`. Detail: `docs/task_log_84_part*.md`.]**

GMP arm at I=0.5 requires +51.2 TWh_th/yr of power-sector deliverability
by 2030; at I=1.0, +102.5. Whole-programme numerator over whole-programme
denominator, which avoids allocating specific corridors (a composition
fallacy): 22 bn pipelines-only / 102.5 = 215 USD-m per TWh_th/yr (LOWER,
excludes CPF and upstream); 60 bn full value chain / 102.5 = 585 USD-m
(UPPER, includes export, industrial, CNG). Implied midstream investment
11–30 bn at I=0.5, 22–60 bn at I=1.0 — **15 to 40 times the 750M DARES
envelope**, which sharpens F-2.

Meaningful for uncapped pathways only (F-9).

## M-2 — Two errors partly cancelling are undetectable from internal consistency
**METHOD. Unchanged, and now with a third instance.**

**Source (pre-epoch, TASK 54 Phase 1).** no result file; arithmetic on register L17 annuity factors (6.822 against 6.469).

> **[RE-DERIVED, TASK 84, 2026-10-23 — WEAKENED: the annuity figures 6.822 / 6.469 are superseded by the register itself and the 'third instance' (F-20) is reversed.** Basis: register arithmetic and F-20. (1) The register's own L17 text (`docs/limitations_register.md` lines 855-858) says the '6.469 (correct)' figure 'was a rough figure from an abandoned attempt, not this implementation'; the implemented commercial-tenor treatment gives F_0 = **7.038499** (real 8.824%, 10 yr) against the legacy **8.349709** (`blended_central` 13.24%, not the 14.4% of the entry), ratio **0.842963**, i.e. the legacy bankability revenue is overstated by about 19%, not 5%. The entry's -5% and its 14.4% / 6.822 / 6.469 are therefore stale. (2) 'Third instance, from this freeze: the retirement inversion (F-20) was a symptom of an omitted cost term': F-20 is **REVERSED** on the vintage model (v40 cheaper than v50 by 0.129 bn), so the retirement inversion is not shown to have been an omitted-cost symptom. (3) The general method statement (check each input against its own basis) is not a quantity and is untouched. Classification: restatable only with the register's corrected figures. Source: `docs/limitations_register.md (L17, lines 855-858)`, `results/baseline_sliced/summary__7ba9c449.json`, `results/baseline_sliced/summary__3852cdf3.json`. Detail: `docs/task_log_84_part*.md`.]**

Register L17: the bankability constraint discounts at the retired pre-F1
blend (14.4%, 70% composed of the 18% naira anchor F1 removed from the
objective) over the model horizon rather than the commercial tenor. The
rate is too HIGH and the horizon too LONG, so at the first vintage the
two nearly cancel: annuity factors 6.822 against 6.469, about −5%. The
offset does not hold at later vintages.

This is the same shape as the Phase 1 calibration finding, where
baseline NPV moved 5.98 -> 8.99 -> 11.28 -> 273 -> 16.14 bn and two
parameter errors partially offset. **Both were found by checking each
input against its own basis, not by checking the model against itself.**

**Third instance, from this freeze:** the retirement inversion (F-20)
was a symptom of an omitted cost term, not a timing effect. It was
visible in the output for months and was not diagnosable from internal
consistency — only from asking what the objective was missing.

## M-3 — A constraint can bind exactly and still be immaterial
**METHOD. New at this freeze.**

**Source (pre-epoch, TASK 54 Phase 1).** `results/hydro_minflow/` (5 files) — **overwritten by TASK 53 smoke run, no backup**; runner `27_run_hydro_minflow_sensitivity.py` (**not in `run_phase_b.sh`**), original 2026-09-06. The only surviving record of the three objectives is register L20.

> **[RE-DERIVED, TASK 84, 2026-10-23 — MOVED: magnitudes; the `om_none` (L20) configuration has no vintage run.** Basis: uncapped. **Correspondence:** `arms_e1ca2888` is `27_run_hydro_minflow_sensitivity.py` at the default `om_case=om_central` (its tag carries `om_case=om_central` and the vintage token) and corresponds to the blended `arms_77bf0ff4`; the blended `om_none` configuration `arms_1b623521` has **no vintage counterpart** and its sub-claim cannot be re-derived. Vintage: objective at 0.30 / 0.50 / 0.70 ~~24,970,088,257.74 / 24,986,620,949.18 / 25,003,848,458.53~~ **23,307,105,585.55 / 23,323,068,340.46 / 23,341,992,621.13** (the 0.50 arm equals the vintage central); band **0.150%** of central (~~0.141%~~; threshold 0.5%), storage at 2045 band **0.022%** (threshold 2.0%), verdict `material: false`. **The floor still binds exactly:** `wet_night` at ratio 1.0000000 in 2025 and both day slices in 2045 and 2054 at 0.50; at 0.30 it binds in the same four slice-years; at 0.70 also in 2025 `wet_peak` and 2045 `dry_peak`, `wet_peak`, 2054 `dry_peak`. Binding and immaterial: the M-3 pattern stands. The runner prints the stale-reference warning of L140. Source: `results/hydro_minflow/arms_e1ca2888/hydro_minflow_sweep_0.30_0.50_0.70.csv`, `results/hydro_minflow/arms_e1ca2888/hydro_minflow_verdict_0.30_0.50_0.70.json`, `results/hydro_minflow/arms_e1ca2888/ARMS.json`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — MOVED; conclusion holds.** Re-run on the corrected model (the five lost files restored as `results/hydro_minflow/arms_77bf0ff4` (`om_central`, the configuration the thesis reports) and `arms_1b623521` (`om_none`, L20's configuration); `27_run_hydro_minflow_sensitivity.py --om-case ...`, `0af709f`). **The floor still binds exactly**: `wet_night` at ratio 1.0000000 in 2025, and both day slices in 2045 and the terminal year, in both configurations. Objective at 0.30 / 0.50 / 0.70: `om_central` **24,970,088,257.74 / 24,986,620,949.18 / 25,003,848,458.53** (the 0.50 arm equals the central anchor), `om_none` 18,452,055,664.69 / 18,466,184,917.99 / 18,480,870,053.51. **Bands:** `om_central` objective **0.1351%** of central, storage@2045 **0.0129%**; `om_none` **0.1560%**, **0.0129%**; pre-epoch (`om_none`, L20) ~~0.141% and 0.000%~~; thresholds 0.5% and 2% (fixed before the solves), so **IMMATERIAL** in both. **One statement in the body does not survive:** storage no longer moves "0.000%": it falls by 2.4 MWh per 0.20 step (37,416.9 / 37,414.5 / 37,412.1 MWh at 2045; 2054: 63,058.7 / 63,056.3 / 63,053.9), and solar built rises by 82.8 MW per step (38,947.2 / 39,030.1 / 39,112.9 MW). The size of both responses is far inside the materiality thresholds, but "zero storage response" and its stated mechanism (the floor cannot alter firm capacity) are now "negligible", not "zero". Binding and material are still distinct properties. Source: `results/hydro_minflow/arms_{77bf0ff4,1b623521}/`.]**

`hydro_min_flow_fraction = 0.50` floors per-slice hydro dispatch. The
floor BINDS exactly — `wet_night` sits on it in 2025 to seven decimal
places, both day slices sit on theirs in the terminal year — and yet
moving it across 0.30/0.50/0.70 changes the objective by 0.141% and
storage by 0.000%.

Binding and material are distinct properties. A shadow price can be
exactly zero at a constraint that is exactly tight. Register L20 records
the measurement, with thresholds fixed before the solves ran.

## M-4 — One unmeasured parameter accounts for nine-tenths of the
## uncertainty in every reported quantity
**METHOD, and the strongest measurement-priority result in this thesis.**

**Source (pre-epoch, TASK 54 Phase 1).** `results/parameter_uncertainty/{summary,draws}_n256_seed42.*`; runner `29_run_parameter_uncertainty.py`, Phase B item 22, 2026-10-03 (TASK 29 re-run `4ec43b4`, documentary). The draws file held three appended runs (L77).

> **[RE-DERIVED, TASK 84, 2026-10-23 — STRENGTHENED: gas availability explains 94.7% of cost variance (was 87.3%) and 95.5% of capital variance: 'nine-tenths' holds for both.** Basis: uncapped (central configuration). Vintage `results/parameter_uncertainty/arms_09bd6dbc/` (1,792 solves, 0 failures; draws file exactly 1,792 rows; `G10-6-sobol` now reads it and passes). **Sobol, system cost, S1 / ST (blended in brackets):** gas availability **0.9466 / 0.9527** (~~0.8729 / 0.8767~~), fleet efficiency (gas_eta) **0.0008 / 0.0003** (~~0.0019 / 0.0023~~), solar CF **0.0132 / 0.0135** (~~0.0526 / 0.0529~~), O&M **0.0145 / 0.0120** (~~0.0345 / 0.0321~~), conversion capex **0.0250 / 0.0258** (~~0.0382 / 0.0379~~); S1 sum 1.0001. **Capital mobilised, S1:** availability **0.9545** (~~0.9028~~), gas_eta 0.0028 (~~0.0048~~), solar CF 0.0063 (~~0.0347~~), O&M 0.0029 (~~0.0004~~), conversion capex 0.0451 (~~0.0624~~); S1 sum 1.0116. Availability explains **94.7%** of cost variance (bootstrap 95% CI 0.802 to 1.104 on S1) and **95.5%** of capital variance; the other four parameters together 5.3% of cost. Outputs: system cost mean ~~28.46~~ **27.11 bn** (sd 2.82, p5 23.68, p50 26.33, p95 32.33); capital mobilised (2025-2045) mean ~~11.38~~ **10.41 bn**; storage 2054 mean 63,259 MWh; solar built 35,665 MW. The entry's parameter table (priors) is input-side and unchanged. Source: `results/parameter_uncertainty/arms_09bd6dbc/summary_n256_seed42.json`, `results/parameter_uncertainty/arms_09bd6dbc/draws_n256_seed42.jsonl`. Detail: `docs/task_log_84_part*.md`.]**

> **[RE-DERIVED, TASK 55, 2026-10-20 — WEAKENED; conclusion holds with a smaller margin.** Post-epoch `results/parameter_uncertainty/arms_ce5387ec/` (Phase B item 22, 2026-10-20 00:45-01:17; **1,792 solves, 0 failures; the draws file is clean (exactly 1,792 rows, created exclusive) and the summary describes one run**: the pooled-rows defect of L77 is gone). **Sobol, system cost, S1 / ST (pre-epoch epoch-2 in brackets):** gas availability **0.8729 / 0.8767** (~~0.9039 / 0.9058~~), solar CF **0.0526 / 0.0529** (~~0.0405 / 0.0406~~), conversion capex **0.0382 / 0.0379** (~~0.0311 / 0.0317~~), O&M **0.0345 / 0.0321** (~~0.0240 / 0.0219~~), fleet efficiency **0.0019 / 0.0023** (~~0.0019 / 0.0021~~); S1 sum 1.0001. **Capital mobilised, S1 / ST:** gas availability **0.9028 / 0.8982** (~~0.9038 / 0.8999~~), solar CF 0.0347 / 0.0361, conversion capex **0.0624 / 0.0653** (~~0.0623 / 0.0662~~), O&M 0.0004 / 0.0006, efficiency 0.0048 / 0.0029. So availability explains **87.3%** of cost variance (bootstrap 95% CI 0.718-1.014 on S1) and **90.3%** of capital variance: "nine-tenths" holds for capital and is now "about seven-eighths" for cost; the other four parameters together account for 12.7% (~~9.6%~~) of cost variance. First order equals total order within **0.0038** for cost and **0.0046** for capital (gas availability), not 0.003: the additivity statement weakens slightly. Output distributions (28.46 bn cost mean etc.; the summary's `qoi_summary` keys keep the unlabelled names `final_storage_capacity_mwh`, `solar_total_built_mw`, L111): system cost mean **28.46 bn** (~~31.49~~), sd 2.33, p5 25.34, p50 27.95, p95 32.71; capital mobilised mean 11.38 bn (~~12.53~~), sd 1.71, p5 9.17, p95 14.48; storage 2054 mean 71,233 MWh (~~69,545~~), p5 63,512, p95 82,929; solar built mean 45,635 MW (~~43,516~~), p5 37,807, p95 56,856. Cost correlates with gas availability at r = −0.913. **The central configuration (24.987 bn) sits at the 2.0th percentile — BELOW p5 (25.34) again**, as at epoch 1 and unlike the epoch-2 reading (6.4th percentile, above p5). The prior-width caveat, the feasibility caveat (zero infeasible draws, a priced release valve) and the parameter table are unchanged. Source: `results/parameter_uncertainty/arms_ce5387ec/{summary_n256_seed42.json,draws_n256_seed42.jsonl}`.]**

A Sobol variance decomposition over five continuous parameters at the
central configuration — 1,792 solves, zero failures — attributes
output uncertainty as follows.

**Parameters, their ranges, and their provenance:**

| parameter | range | provenance |
|---|---|---|
| gas fleet availability | 0.295 - 0.80 | endpoints SOURCED, **distribution ASSUMED** |
| fleet thermal efficiency | 0.277 - 0.298 | ~~DERIVED from the 800-860 MMscf/d volume range~~ **[corrected TASK 77 A.3: 0.287 x (1 - 3.5%) to 0.287 x (1 + 3.8%), i.e. 775.7 to 834.6 MMscf/d at 1,036 Btu/scf; the 800-860 range would give 0.289 to 0.269, and has no source (register L130). The figures do not change.]** |
| solar capacity factor | 0.16 - 0.20 | SOURCED (Global Solar Atlas, L26) |
| O&M level | ATB Advanced to Conservative, continuous | SOURCED endpoints |
| conversion capex | 450 - 850 USD/kW | DERIVED (the three `conversion_scenarios()` arms) |

All priors uniform. **A uniform prior is a statement of ignorance
between bounds, not a claim about likelihood**, and no parameter here
has a sourced distributional shape.

**Output distributions:**

| | mean | sd | p5 | p50 | p95 |
|---|---|---|---|---|---|
| system cost | 32.21 bn | 2.79 | 28.44 | 31.65 | 37.24 |
| capital mobilised (2025-2045) | 12.96 bn | 1.71 | 10.71 | 12.58 | 16.12 |
| storage, 2054 | 72,108 MWh | 5,846 | 66,068 | 70,339 | 85,527 |
| solar built | 45,645 MW | 4,959 | 39,527 | 44,840 | 56,161 |

**CURRENT-EPOCH RE-MEASUREMENT (TASK 27 Part C; supersedes the table above
for the current code state, which is retained as the earlier epoch).** The
table above is the Sobol run that preceded Phase B (**epoch 1**; its draws
are rows 0-1,791 of `results/parameter_uncertainty/draws_n256_seed42.jsonl`
and reproduce it to the digit). The Phase B run (item 22, HEAD `2b22fe9`,
**epoch 2**, rows 3,584-5,375 of the same file; bit-identical to the
intermediate run) gives, from **1,792 solves, 0 failures**:

| | mean | sd | p5 | p50 | p95 |
|---|---|---|---|---|---|
| system cost | **31.49 bn** | 2.75 | **27.80** | 30.93 | 36.44 |
| capital mobilised (2025-2045) | 12.53 bn | 1.70 | 10.30 | 12.14 | 15.67 |
| storage, 2054 | 69,545 MWh | 5,829 | 63,512 | 67,782 | 82,871 |
| solar built | 43,516 MW | 4,882 | 37,612 | 42,695 | 53,958 |

Epoch 1 cost mean was 32.21 bn (so the current mean is 0.72 bn, 2.2%, lower;
the shift is correlated with `gas_availability_factor`, r = 0.64; its cause
was not investigated). **Do not quote `summary_n256_seed42.json`'s
`qoi_summary` (cost mean 31.73 bn, p5 27.98, p95 36.71, etc.): it pools all
5,376 rows — three appended runs including epoch 1 — and describes neither
epoch** (the draws file is opened in append mode and never truncated; see
`docs/task_log_26c_sobol_rows.md`, `docs/task_log_27c_sobol_correction.md`).
No other entry in this file quotes the pooled figures.

**The Sobol indices are NOT affected.** `sobol_indices()` consumes the
in-process return values of a single run, not the file; replaying the stored
epoch-2 outputs through scipy reproduces the published indices with a
maximum absolute difference of 0.0. Current indices (cost): gas availability
**S1 0.9039 / ST 0.9058**, solar CF 0.0405 / 0.0406, conversion capex
0.0311 / 0.0317, O&M 0.0240 / 0.0219, fleet efficiency 0.0019 / 0.0021
(S1 sum 1.0013); capital: 0.9038 / 0.8999, 0.0365 / 0.0362, 0.0623 / 0.0662,
-0.0025 / 0.0006, 0.0024 / 0.0030. The nine-tenths conclusion and the
prior-width caveat below stand unchanged. One property stated below does
change at the margin: first order equals total order within 0.003 for cost
(max gap 0.0021) but **not for capital** (max gap 0.0039, gas availability
and conversion capex).

~~**The central configuration (27.954 bn) sits BELOW the 5th percentile.**~~
**[Held at epoch 1 only: p5 was 28.44.]** At the current epoch p5 is
**27.80** bn and the central configuration (27.954 bn) sits at the **6.4th
percentile — ABOVE p5**, not below it.

**[Epoch-1 text, retained:]** **The central configuration (27.954 bn) sits BELOW the 5th
percentile.** This is not an error and must be stated wherever the
distribution is quoted: `gas_avail_nameplate` = 0.80 is the UPPER
bound of the availability prior, so almost every draw raises cost
relative to central. **The central case is the optimistic end of the
availability range, not its middle.**

**Sobol indices — first order and total order (epoch 1; current-epoch values are in the re-measurement block above):**

| parameter | S1 (cost) | ST (cost) | S1 (capital) | ST (capital) |
|---|---|---|---|---|
| **gas availability** | **0.901** | **0.903** | **0.900** | **0.896** |
| solar capacity factor | 0.044 | 0.044 | 0.040 | 0.040 |
| conversion capex | 0.030 | 0.031 | 0.061 | 0.065 |
| O&M level | 0.025 | 0.023 | -0.002 | 0.001 |
| fleet thermal efficiency | 0.002 | 0.002 | 0.003 | 0.003 |

Bootstrap 95% confidence intervals are reported in the summary JSON.

### Three properties that make this interpretable rather than
### tautological

**First order equals total order, everywhere.** Every gap is below
0.003. **The model is additively separable across these five inputs —
there are essentially no interaction effects.** That is a structural
finding about the model, not an assumption: a Sobol decomposition is
the standard way to detect interactions, and it found none.

**The indices sum to 1.001.** No missing variance, no estimator
pathology, and no evidence of the numerical instability a
badly-conditioned LP would produce across 1,792 solves.

**Fleet thermal efficiency contributes 0.002 — effectively nothing.**
This is counterintuitive and is the result's most useful half. η is
mechanically central: it converts fuel to electricity AND, through
`EF_e = EF_th / eta`, sets the electrical emission factor. It
contributes almost no output uncertainty **because its band is narrow
(+/-3.7%), and its band is narrow because it was measured** (F-1's
derivation from NERC 2024 generation over reported gas volumes).

**A well-measured parameter contributes almost nothing to output
uncertainty even when it is mechanically central.** That is the
argument for measurement, stated as a result rather than as a
principle.

### The honest caveat, which is also the finding

**The 90% is a property of the prior's width.** Gas availability is
sampled uniform over [0.295, 0.80] because **no central value exists**
— register P2-9 records that mechanical availability cannot be
separated from fuel unavailability without data nobody publishes, so
`gas_avail_central` raises by design. The 0.505-wide prior spans this
model's largest lever (F-5, 8.60 bn).

If availability were known to within +/-5%, its index would collapse
and solar capacity factor would dominate. **The decomposition does not
discover that availability matters; it quantifies what not knowing it
costs.**

Stated defensibly:

> Conditional on fleet availability being unknown between its two
> sourced bounds, it accounts for nine-tenths of the uncertainty in
> both system cost and capital mobilised, and no other parameter
> approaches it.

### The policy reading

**Measuring Nigerian gas fleet availability precisely is worth more
than refining every other cost input in this model combined.** Solar
capacity factor, O&M rates, conversion capex and fleet efficiency
together account for 10% of output variance; availability alone
accounts for 90%.

That is an actionable conclusion for a regulator or a system operator,
and it is obtainable only from a variance decomposition — a
one-at-a-time sensitivity would show availability as the largest lever
(F-5 already does) without establishing that everything else is nearly
irrelevant beside it.

### Feasibility

**Zero infeasible draws in 1,792.** This is reported for completeness
and should NOT be read as evidence of robustness. The LP has a priced
release valve: unserved energy and self-generation are costed, not
constrained, so the system never fails to serve load — it serves it
expensively. **Feasibility probability is not a meaningful output of
this model**, and the finding is that the question is ill-posed here
rather than that the answer is reassuring.

---

# Section 8 — O&M, the largest single correction

**No technology in either builder's objective carried any O&M cost.**
Gas paid fuel only; solar and storage paid capital only. Plan 2.2
flagged this as storage-specific; it was system-wide.

A storage O&M constant existed (`STORAGE_OM_PER_MWH_YR = 2,000.0`) with
**zero readers** for the life of the project. Its own comment stated its
purpose as breaking LP degeneracy, and it sat at a fifth of the ATB
range it cited — a numerical-conditioning constant wearing a cost's
clothes.

**Effect: +6.602 bn on central, +30.9%.** Measured within one
configuration (`om_none` -> `om_central`), never across arms.

> **[RE-DERIVED, TASK 55, 2026-10-20 — uncapped figure re-derived; MOVED.** Post-epoch Group 5 solve (`results/baseline_sliced/summary__143725ce.json`, `om_none` at the central arms, HEAD `9beb54a`, 16 s): objective 18,466,184,917.99 USD against the central anchor 24,986,620,949.18, so `om_none` → `om_central` is ~~+6.602 bn, +30.9%~~ **+6.520 bn, +35.3%** of the `om_none` base (26.1% of central); storage (37,414.5 / 63,056.3 MWh) and solar (39,030.1 MW) identical, so O&M changes cost and not the plan. This is a within-configuration comparison, as before. The capped 15.1–19.1% range below is **UNSUPPORTED** (it needs the NDC arms at `om_none`, which no driver contains; none was built) and is left unedited.]**

**The omission was larger than the capital-recovery span it was being
compared against.** Every prior statement weighing financing against
system cost was made against an objective missing more than the thing
being measured. This is the strongest available argument for the
audit-every-input protocol, and it belongs in the discussion chapter.

**O&M's relative weight depends on how penalty-like the objective is.**
Uncapped it adds 30.9%; under caps only 15.1–19.1%, because capped
objectives are already inflated by displaced load priced at
self-generation. Do not compare percentage effects across capped and
uncapped runs.

---

## Scoped conditions on every result above

    L5    storage capacity credit is flat 1.0 -- magnitude is a residual
    L8    gas retirement vintages are an analyst estimate
    L13   the two deliverability families are not comparable unlabelled
        L16   EaaS and public solar are physically identical -- financing
          comparison only, nothing about siting; DIST-2 unimplemented,
          biasing AGAINST EaaS
    L17   bankability discounts at the retired blend over the wrong
          horizon; every EaaS result is conditional on it
    L19   solar_cf and dist5 are not exposed as arms and appear in no
          output tag
    L20   hydro_min_flow_fraction is ASSUMED, binding, and immaterial
    L21   storage_duration_hours is ASSUMED, explicit but unswept; it
          now also scales storage O&M
    L22   O&M rates are SOURCED but unscaled US figures applied to a
          Nigerian fleet; direction of bias stated both ways
    L23   F1 ships two capital tranches; the design specified three --
          the concessional envelope's measured value is an upper bound
    L24   post-2035 cap shape is a second analyst construction stacked
          on the apportionment rule
    L25   DIST-5 is sourced but NOT implemented; no measurement exists
    L26   solar capacity factor is a national average with no siting
    P2-9  gas_avail_nameplate 0.80 against a measured 29.5%