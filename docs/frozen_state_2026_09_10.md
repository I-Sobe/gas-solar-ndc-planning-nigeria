# Frozen state — Nigeria LP-CEM thesis

**Model freeze.** Branch `phase-fin2-d17`, model HEAD **`2a9226b`**,
2026-09-07. Documentation continued past that commit; **no change after
`2a9226b` alters any model number.**

**Supersedes `handover_pre_om.md`** (formerly `handover.md`), retained
unmodified. That document is two epochs behind: its anchors predate both
the shape-factor sourcing and the O&M implementation. Read it for the
stratigraphy, not for numbers.

**Revises the first version of this document**, written earlier on
2026-09-07, which predated finding F-21, the cap-trajectory arm, and
register entries L24-L26.

Every figure below comes from the sweep at `2a9226b`. Nothing is carried
forward from an earlier state.

---

## 1. Anchors

    regression   conv_none + pre_f1_overnight + dscr_regression
                 + tariff_legacy_65m + om_none
                 21,811,042,472.821114 USD  /  45,502.9754 MWh

    central      conv_central + f1_tranche + dscr_central + cr_central
                 + env_dares + baseline deliv + hydro_flat
                 + tariff_central + om_central
                 27,954 M USD  /  65,612.985 MWh  /  41,031.4 MW solar
                 backcast -3.2% against observed 2024 gas generation

The regression anchor runs `om_none` deliberately: its function is to
reproduce the pre-O&M objective exactly, proving the O&M wiring
introduced no side effect. **It is not a cost result and is not
comparable to any `om_central` figure.**

**Superseded anchor values, for reading older documents:**

| epoch | regression | central | what moved it |
|---|---|---|---|
| pre-shape-sourcing | 21.4748 bn / 44,376.89 MWh | 21.154 bn | — |
| post-shape-sourcing | 21.811 bn / 45,502.98 MWh | 21.352 bn | `b7f7bdb` |
| **post-O&M (current)** | **21.811 bn** (om_none) | **27.954 bn** | `d7538ff` |

The shape-factor sourcing moved the regression anchor **UP**,
21.475 → 21.811. An earlier reading of this transition as downward was
incorrect and is retracted.

`handover_pre_om.md` states anchors of 20.9899 bn and 20.856 bn. **These
are not reproducible on any commit in this repository's history**, and
the cause could not be established: the git log shows nothing touching
`src/` or `scripts/` between the sweep that produced 21.811 and the
commits of 2026-09-06. Recorded as unexplained rather than
rationalised.

---

## 2. The sweep

`bash scripts/run_central_sweep.sh [A|B|C]`. **There is no Section D** —
earlier documents citing "37 arms" across four sections are wrong on
both counts.

    Section A   5 invocations   central configuration, four live runners,
                                plus the regression anchor
    Section B  25 invocations   one-at-a-time arms on 01b
    Section C   8 invocations   deliberate interaction cells

**38 invocations, several running multiple solves. Zero failures, zero
tracebacks at this HEAD.**

Section A carries the regression anchor. If it moves, nothing below it
is attributable.

---

## 3. What the model says

### Levers, ranked by effect on system cost

| lever | span (bn) | pre-O&M | change |
|---|---|---|---|
| gas fleet availability (nameplate → observed) | **8.60** | 7.32 | +17% |
| **conversion (none → high)** | **5.91** | 3.14 | **+88%** |
| capital recovery span (comm_low → comm_high) | **3.97** | 3.90 | flat |
| gas deliverability, drift family | **3.01** | 2.55 | +18% |
| gas retirement (none → v30) | **2.76** | 2.71 | flat |
| GMP deliverability family (flat → full) | **2.47** | — | — |
| concessional envelope (none → 10x) | **2.46** | 2.45 | flat |
| demand shape (suppressed → high) | **2.21** | 1.90 | +16% |

Read the ranking, not only the magnitudes. Gas availability remains the
largest single lever and still exceeds the capital-structure span by
roughly 2:1.

### F-1 — cost of capital reaches the objective, but weighs less

The span holds at **3.97 bn** (25.762 → 29.728), essentially unchanged
by O&M. Its **share of system cost falls from 18.6% to 14.2%**, because
the objective it is measured against grew 31%.

Both statements belong in the thesis. The contribution is intact — under
a blended WACC the same variation produced identically zero — but its
relative weight against operating and physical cost is smaller than
previously reported, and a referee will compute this if it is not
stated.

    cr_comm_low   (9% / 15 yr)    25.762 bn
    cr_central    (11% / 10 yr)   27.954 bn
    cr_ida21      (2.13% conc.)   27.969 bn
    cr_comm_high  (14% / 7 yr)    29.728 bn

**Conditional on L23:** F1 ships two capital tranches; the design
specified three. The missing guaranteed-commercial rung makes the merit
order more expensive at the margin than reality, so the concessional
envelope's measured value is an upper bound.

### F-2 — concessional finance remains inframarginal

    env_none        28.261 bn     storage 65,613.0 MWh
    env_dares       27.954 bn     storage 65,613.0 MWh
    env_dares_plus  27.833 bn     storage 65,613.0 MWh
    env_5x          26.873 bn     storage 65,613.0 MWh
    env_10x         25.802 bn     storage 65,613.0 MWh

Span 2.46 bn, unchanged by O&M. Storage and solar identical at every
scale. **Concessional finance changes who pays, never what is built.**

### F-3 — STRENGTHENED

Across every capital-recovery arm, every envelope arm and both DSCR
arms — a **3.97 bn cost span, 14.2% of central** — storage is
**65,613.0 MWh** and solar **41,031.4 MW**, identical to the decimal.

Stronger than pre-O&M because **storage now carries roughly $636M/yr of
real holding cost** and still does not move anywhere in the financing
space. A priced asset that does not respond to a 14% cost swing is a
much harder result than an unpriced one.

Every arm that moves quantity is physical: conversion, deliverability,
retirement, availability, peak shape, hydro. Clean partition, no
exceptions.

`dscr_low` and `dscr_high` both return **27.954 bn, byte-identical to
central** — the DSCR cushion does not reach the objective in the
uncapped baseline.

### F-6 — conversion nearly doubled, now the second-largest lever

    conv_none     32.678 bn   storage 76,781.4 MWh
    conv_low      30.229 bn   storage 66,534.8 MWh
    conv_central  27.954 bn   storage 65,613.0 MWh
    conv_high     26.772 bn   storage 65,041.6 MWh

Worth **5.91 bn**, against 3.14 pre-O&M.

**A finding O&M created rather than rescaled.** Converted plant carries
VOM of 2.12 USD/MWh against the unconverted fleet's 6.94, and produces
21.76 of 24.79 TWh_e. Its fixed O&M is *higher* (33.5 vs 25.7 USD/kW-yr)
and it wins anyway. Once operating cost is priced, conversion is an
operating-cost play as well as a fuel-efficiency one.

**This strengthens the petroleum-engineering contribution.**

### F-5 — gas availability remains the largest lever

    gas_avail_nameplate (0.80)   27.954 bn   storage 65,613.0 MWh
    gas_avail_observed  (0.295)  36.557 bn   storage 86,056.8 MWh

**+8.60 bn, +30.8%.** No sourced central arm exists (P2-9): mechanical
availability cannot be separated from fuel unavailability without data
nobody publishes, so the central case sits on the declared UPPER bound
and the declared lower bound costs a third more. State prominently.

### F-20 — RESOLVED: the retirement inversion was an O&M artefact

    gas_retire_none        25.199 bn   storage 47,853.1 MWh
    gas_retire_vintage_50  27.184 bn   storage 60,534.7 MWh
    gas_retire_vintage_40  27.715 bn   storage 65,613.0 MWh
    gas_retire_vintage_30  27.954 bn   storage 65,613.0 MWh  (central)

Pre-O&M the ordering was non-monotone — `vintage_40` cost more than
`vintage_30` despite retiring less — and was flagged as a timing effect
needing explanation. **Under O&M the ordering is monotone and the
anomaly is gone.** Retiring plant now avoids a recurring capacity
charge, not only a fuel cost. The caveat is **withdrawn, not
explained.**

### F-21 — NEW: tightening a grid-only cap RAISES national emissions

The cap-trajectory arm (`ndc3_conditional_decline`, plan 4.2) declines
linearly from the `ndc3_conditional` 2035 anchor toward zero at 2060 —
Nigeria's own net-zero target. At 2045 the cap is 6.024 MtCO2 against
the flat hold's 10.039, a 40% tightening.

| | `ndc3_conditional` | `ndc3_conditional_decline` |
|---|---|---|
| objective | 51.573 bn | **61.815 bn (+10.24, +19.9%)** |
| national emissions | 486.7 Mt | **524.2 Mt (+37.5, +7.7%)** |
| leakage | 35.6% | **54.8%** |
| grid emissions | 313.4 Mt | 236.9 Mt (−24.3%) |
| self-generation | 173.3 Mt | 287.3 Mt (+65.6%) |

**Strictly dominated on both axes.** Grid emissions fall 76.5 Mt;
self-generation rises 114.0 Mt. For every tonne abated on the grid, 1.49
tonnes appear in diesel, at a cost of 10.24 bn.

The cap regulates grid emissions only; demand is exogenous; solar is
rationed by the capital envelope and the NIRP trajectory; so displaced
load goes to diesel at 0.902 tCO2/MWh against converted gas at 0.3963.
No carbon price exists to see it.

**Robust to supply-chain CO2e.** Applying a +10-25% uplift to grid gas
emissions narrows the gap to 18.4 Mt at the maximum; the gap closes only
at **+49%**, roughly double the top of any plausible range.

**Widening the boundary is not a remedy.** Phase 2.6's defect D11
established that counting self-generation inside the cap makes NDC3
unconditional infeasible under a tight envelope in all three genset
arms. That is part of the finding, not a gap in it.

### F-18 — the hydro bound, sharpened

    hydro_flat       27.954 bn
    hydro_committed  27.013 bn
    hydro_ndc        13.093 bn

Under NDC-scale hydro, system cost falls **53%** — a 14.86 bn gap,
against 12.2 bn pre-O&M. The bound on the solar-dominant result is
larger than previously reported. The defence is deliverability, not
modelling: 2.1 Zungeru-equivalents per year for five years against 8-12
year build times.

---

## 4. O&M — the largest single correction in this project

**No technology in either builder's objective carried any O&M cost.**
Gas paid fuel only; solar and storage paid capital only. Plan 2.2
flagged this as storage-specific; it was system-wide.

A storage O&M constant existed (`STORAGE_OM_PER_MWH_YR = 2,000.0`) with
**zero readers** for the life of the project — a numerical-conditioning
constant, by its own comment, sitting at a fifth of the ATB range it
cited. Retired at `d7538ff`.

**Sourced replacement:** NREL ATB 2024, `Summary_FOM` and the
per-technology sheets. Three cases (`om_low`/`om_central`/`om_high` =
Advanced/Moderate/Conservative), 2025-2050 extended flat to 2054.
`data/cost/processed/om_cost.csv`, 312 rows.

Central rates, USD/kW-yr:

| year | solar | storage | gas_ct | gas_cc |
|---|---|---|---|---|
| 2025 | 21.00 | 38.78 | 25.7 | 33.5 |
| 2035 | 15.02 | 30.00 | 24.2 | 30.1 |
| 2045 | 13.51 | 24.99 | 22.8 | 27.8 |

VOM USD/MWh: solar 0.0, storage 0.0, `gas_ct` 6.94 flat, `gas_cc`
2.12 → 1.85.

**Effect: +6.602 bn on central, +30.9%.** Measured within one
configuration (`om_none` → `om_central`), never across arms.

**The omission was larger than the capital-recovery span it was being
compared against.** Every prior statement weighing financing against
system cost was made against an objective missing more than the thing
being measured.

**O&M's relative weight depends on how penalty-like the objective is.**
Uncapped it adds 30.9%; under caps only 15.1-19.1%, because capped
objectives are already inflated by displaced load priced at
self-generation. Do not compare percentage effects across capped and
uncapped runs.

### Declared decisions

- **EaaS and public solar carry IDENTICAL O&M**, charged to the
  objective, excluded from the public capital envelope. The EaaS
  advantage remains solely that its CAPEX does not consume the envelope.
- **Converted plant carries `gas_cc` rates; the unconverted fleet
  carries `gas_ct`.**
- **ATB rates are used UNSCALED** despite Nigerian availability being
  far below ATB's assumption. Scaling fixed O&M down by availability
  would model underspend as efficiency, when underspend is plausibly
  *why* availability is low — the same circularity as calibrating demand
  growth to a supply-constrained series. Register L22.
- **Storage FOM converts from $/kW-yr to $/MWh-yr by dividing by
  `storage_duration_hours` (4.0)**, which register L21 declares ASSUMED
  and unswept. That parameter now scales a cost as well as a capacity
  credit.

---

## 5. Verification at this freeze

- `om_none` reproduces **both** pre-O&M anchors to the digit.
- Section A's regression anchor reproduces at 21.811 bn.
- 38 sweep invocations, zero failures, zero tracebacks.
- The four existing cap scenarios reproduce after the cap-arm addition:
  53.821 / 63.374 / 58.698 / 51.573 bn.
- `emissions_cap.csv` regeneration: **+30 rows, 0 changed, 0 removed.**
- The min-flow floor binds exactly in three slices yet moves the
  objective by 0.141% (L20).

- Sobol parameter-uncertainty decomposition: **1,792 solves, 0
  failures**; first-order and total-order indices agree to within
  0.003 on every parameter (no interactions); indices sum to 1.001
  (M-4).

---

## 6. Scoped conditions carried by every result

    L5    storage capacity credit is flat 1.0 -- magnitude is a residual
    L8    gas retirement vintages are an analyst estimate
    L13   the two deliverability families are not comparable unlabelled
    L16   EaaS and public solar are physically identical; DIST-2
          unimplemented, biasing AGAINST EaaS
    L17   bankability discounts at the retired pre-F1 blend over the
          model horizon rather than the commercial tenor
    L19   solar_cf and dist5 are not exposed as arms and appear in no
          output tag
    L20   hydro_min_flow_fraction is ASSUMED, binding, and immaterial
    L21   storage_duration_hours is ASSUMED, explicit but unswept; it
          now also scales storage O&M
    L22   O&M rates are SOURCED but unscaled US figures on a Nigerian
          fleet; direction of bias stated both ways
    L23   F1 ships two capital tranches; the design specified three
    L24   post-2035 cap shape is a second analyst construction stacked
          on the apportionment rule
    L25   DIST-5 IS implemented and measured inframarginal (<0.03%)
    L26   solar capacity factor is a national average with no siting
    P2-9  gas_avail_nameplate 0.80 against a measured 29.5%

Register runs L1 to L26, each number once.

---

## 7. Open, and deliberately not closed

**Declared, not fixed** — each with a stated direction of bias:

- **The guarantee tranche** (L23). Blocked on World Bank PRG headroom
  and FGN contingent-liability data.
- **DIST-3** counterparty differentiation. Needs a revenue side built
  before a collection rate has anything to attach to.
- **2.7 tiered demand.** Blocked behind a memo carrying a 10x unit error
  (L4) and a live double-count risk against `selfgen` (F-23).
- **F-11's leakage magnitudes.** The structural claim stands; the
  34.2-45.0% figures are pre-O&M and are not restated.

**Not attempted, and why:** promoting `storage_duration_hours` to a
required key (37 call sites across 26 files, against ten runners already
dead on required args, for a parameter whose fallback equals its
canonical value — declined, not deferred).

**For the write-up, not the model:** the NIRP gas-ceiling defence at
~3.5x divergence. Prose, but load-bearing for every result, because it
is what makes solar the residual.

---

## 8. Reporting rules

- Report `npv_gross_capex_report_window_usd` (2025-2045, gross, salvage
  excluded) as the financing headline, never `system_cost_npv`. Name the
  horizon and salvage basis in the same sentence.
- Reporting window is **2045**. 2046-2054 is buffer; no figure from
  those years is reportable.
- Any EaaS figure carries its `capital_case` and `tariff_case` and is
  reported as a point in the stated interval (F-12).
- Do not report storage magnitude as an optimisation result (L5).
- Never mix the two deliverability families unlabelled (L13).
- O&M effects are measured within a configuration, never across arms.
- The `ndc3_conditional_decline` arm is a cap-shape sensitivity, never a
  central case. It is opt-in via `--include-decline-arm` and is not in
  `run_central_sweep.sh`.

**Buffer-year capital spend, now measurable.** Two gross-capex fields
are produced: `npv_gross_capex_report_window_usd` (2025-2045) and
`npv_gross_capex_full_horizon_usd` (2025-2054). At the central
configuration these are 12.772 bn and 14.350 bn, so **1.578 bn — 11.0%
of full-horizon capital — falls in the buffer years.** Under
`conv_none` the buffer share is 2.162 bn of 17.182 bn, 12.6%.

That is a material fraction, and it is why the two fields are reported
separately rather than one being used for both purposes. Buffer-year
figures are on the do-not-report list by construction: they carry the
2050 storage rebuild and the terminal-year capital activity that F-16's
salvage arbitrage concerns. `npv_capex_report` in
`src/optimize_experiments.py` computes a FULL-HORIZON figure under a
name implying the report window; it is not used by any runner, and the
naming defect is recorded in the register rather than corrected, since
the field has no consumers.