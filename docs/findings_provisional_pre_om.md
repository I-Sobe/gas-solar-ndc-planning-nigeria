# Findings — central configuration

> **SUPERSEDED (marked 2026-10-09, TASK 42).** `docs/findings_frozen.md` supersedes this
> file; the frozen file's header says so. This file is a superseded epoch, kept for
> reference, and is not to be quoted as a result. In particular its F-12 and F-13 text
> carries the older, narrower wording (the split "not identified except where a constraint
> forces it"; "a sufficiently tight budget would close it"); F-12 as corrected 2026-10-08
> says the EaaS/public split is unidentified in every capital case, binding ones included.

**Supersedes `docs/findings_provisional_pre_sweep.md`**, which is retained
for reference. That file contains findings F-1 to F-29 as they were
produced across five different model states: pre-conversion, pre-hydro
arms, pre-tariff arm, at `dscr_regression`, and at the 65M silent-default
tariff. Every number below has been re-produced at ONE state.

Where a superseded finding's number changed, the old value is given. Where
it did not, that is stated — an unchanged number across a configuration
change is itself evidence.

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

    peak_unsuppressed / storage_cc_central / gas_unconstrained /
    constrained / gas_avail_nameplate / gas_retire_vintage_30 /
    gas_capex_central / ccgt_eff_central / us_inflation 0.020

**Central baseline: 21.154 bn USD NPV, storage 64,689 MWh (2054),
backcast −3.4% against observed 2024 gas generation.**

**Regression anchor: 21.475 bn / 44,377 MWh** at `conv_none` +
`pre_f1_overnight` + `dscr_regression` + `tariff_legacy_65m`. Reproduced
to the digit throughout the sweep. Every number below is attributable
because this reproduces.

Two notes on the central choices. `hydro_flat` is the only hydro arm that
reproduces the backcast; the others depart from the 2024 observation by
construction. `baseline` deliverability is the historical-drift family,
chosen over the GMP policy-target family because a central result should
not assume policy succeeds — the GMP arms are reported as a separate
family (register L13) and never mixed with drift arms unlabelled.

Reporting window is **2045**. 2046–2054 is end-effect buffer and no
figure from those years is reported.

---

## Section 1 — The contribution

### F-1 — Cost of capital reaches the objective
**STRUCTURAL.** The F0 audit found four blend scenarios returning
2.765439e10 to seven significant figures: halving the private cost of
capital from 18% to 9% changed system cost by exactly zero. Under the
per-tranche capital recovery formulation it spans **18.985 to 22.912 bn**
across recovery arms — a 3.93 bn range, 18.6% of the central value.

    cr_comm_low   (9% / 15 yr)    18.985 bn
    cr_central    (11% / 10 yr)   21.154 bn
    cr_ida21      (2.13% conc.)   21.170 bn
    cr_comm_high  (14% / 7 yr)    22.912 bn

`cr_ida21` differs from `cr_central` by 16M — the IDA20-to-IDA21 shift in
concessional terms is second-order against the commercial tranche.

### F-2 — Concessional finance is inframarginal at every scale tested
**STRUCTURAL.** Envelope arms, central config:

    env_none        21.461 bn      storage 64,689 MWh
    env_dares       21.154 bn      storage 64,689 MWh
    env_dares_plus  21.036 bn      storage 64,689 MWh
    env_5x          20.084 bn      storage 64,689 MWh
    env_10x         19.004 bn      storage 64,689 MWh

Cost falls monotonically; **storage is identical to the megawatt-hour at
every scale**, as are gas and solar. Saving per concessional dollar:
0.409 (dares), 0.293 (5x), 0.246 (10x) — declining as later, more heavily
discounted vintages are drawn on.

The envelope is fully drawn and exhausted at every build decision, so the
marginal dollar is commercial everywhere. An inframarginal lump of cheap
capital changes the total bill, not the choice. **Concessional finance
changes who pays, never what is built.**

### F-3 — Cost of capital drives composition through the storage/gas margin
**PROVISIONAL.** Storage 2054 is **64,689 MWh in every capital-recovery
arm, every envelope arm and every DSCR arm** — invariant to price. It
moves only for physical changes: conversion, deliverability, hydro,
availability, peak shape, retirement.

This is the cleanest statement the model produces of its own structure:
**financing moves cost; physics moves quantity.**

### F-4 — Tenor is half the concessional advantage
**PROVISIONAL.** Real-terms CRF at the central rates: concessional
0.03754 (0.784% over 30 yr), commercial 0.15461 (8.824% over 10 yr), a
ratio of 4.119x. Shapley decomposition, order-invariant: **rate 47.0%,
tenor 53.0%** (46.3 / 53.7 in real terms). A one-at-a-time attribution
gives 68/32 or 31/69 depending on which variable moves first and is not
reportable.

Blended-WACC formulations cannot express tenor at all. Nigerian evidence:
no utility-scale grid-connected solar IPP has reached financial close;
commercial bank tenors run 3–7 years absent credit enhancement;
continental average tenor has fallen from twelve years to five.

---

## Section 2 — What dominates, and it is not financing

### F-5 — Gas fleet availability is the largest single lever in the model
**NEW — not in the superseded file, and it changes the framing.**

    gas_avail_nameplate (0.80)   21.154 bn   storage 64,689 MWh
    gas_avail_observed  (0.295)  28.451 bn   storage 87,010 MWh

**+7.30 bn, +34.5%.** Ranked against every other lever swept:

    gas availability bracket        7.30 bn
    capital recovery span           3.93 bn
    conversion (none -> central)    3.15 bn
    gas retirement (none -> v30)    2.76 bn
    deliverability, drift family    2.55 bn
    concessional envelope (0 -> 10x) 2.46 bn
    demand shape (suppressed->high) 1.90 bn

There is no central arm. `gas_avail_central` raises by design: it must
represent MECHANICAL availability only — availability given fuel is
allocated — and that cannot be separated from fuel unavailability without
data nobody publishes (register P2-9). So the central case sits on the
declared UPPER bound, and the declared lower bound costs a third more.

**This must be stated prominently, not buried.** A reader will otherwise
notice that the physical availability of the existing gas fleet matters
roughly twice as much as the entire capital-structure question. Stated
first, it is an argument for the thesis rather than against it: if a third
of system cost hinges on whether Nigeria's fleet runs at 29.5% or 80%,
and the gap is an undecomposed mechanical-versus-fuel question, then
treating deliverability and financing as the binding constraints — rather
than capacity — is the right frame.

### F-6 — OCGT-to-CCGT conversion, unconstrained
**PROVISIONAL.**

    conv_none     24.307 bn   storage 75,481 MWh
    conv_low      23.034 bn   storage 72,198 MWh
    conv_central  21.154 bn   storage 64,689 MWh
    conv_high     20.007 bn   storage 64,148 MWh

Conversion is worth **1.27 to 4.30 bn** (5.2–17.7% of system cost).
Mechanism: converted output rises 0 -> 10.64 -> 21.76 TWh_e while fleet
output falls 24.74 -> 14.11 -> 2.98, total gas-side output constant at
24.74 TWh_e (backcast −3.4% in every arm). The same gas yields more
electricity — the only route to more generation under a binding
deliverability constraint.

### F-7 — The availability screen decides the conversion merit order
**PROVISIONAL.** Per-plant availability from NERC across three periods
bands 16 convertible open-cycle plants into three tiers:

    tier A  PAF > 50%     571 MW nameplate,   547 MW available,  626 USD/kW effective
    tier B  PAF 25-50%  3,215 MW nameplate, 1,109 MW available, 1,740 USD/kW effective
    tier C  PAF < 25%   2,696 MW nameplate,   285 MW available, 5,677 USD/kW effective

Tier A converts 100% in every arm — inframarginal. Tier B converts 29% at
850 USD/kW capex and 100% at 600 and 450, so the flip lies between. Tier C
converts 0% unconstrained.

Tier B converting fully despite an effective cost ABOVE new-build CCGT
(1,317 USD/kW) is the thesis mechanism in a technology choice: fuel-free
firm capacity beats fuel-hungry new capacity when fuel is the binding
constraint.

Without the screen the convertible base reads 6,482 MW of nameplate
rather than 1,941 MW available, and Tier C would be credited with capacity
that does not run.

---

## Section 3 — Under NDC caps

### F-8 — NDC compliance costs 38-44% less with conversion available
**CONFIRMED at the central configuration.** Public-only, `02_run_ndc_caps`:

| case | conv_central | conv_none | saving |
|---|---|---|---|
| ndc2_unconditional | 45.778 | 81.250 | 35.47 bn (−43.7%) |
| ndc2_conditional | 55.048 | 88.422 | 33.37 bn (−37.7%) |
| ndc3_unconditional | 50.694 | 86.530 | 35.84 bn (−41.4%) |
| ndc3_conditional | 43.314 | 76.993 | 33.68 bn (−43.7%) |

Emissions fall 25.9–27.7% in parallel. Mechanism, computed before the run
and confirmed by it: converted plant emits 0.3963 tCO2/MWh_e against the
unconverted fleet's 0.6352 (EF_th 0.1823 / eta) — 37.6% less per unit of
electricity, or 1.60x the electricity per tonne of CO2.

### F-9 — Gas deliverability is worth 3.16 bn uncapped and EXACTLY ZERO under a cap
**CONFIRMED, and now across all four cap cases.** `baseline`,
`gas_deliv_flat` and `gas_deliv_gmp_partial` return **identical objectives
to the digit** in every capped case: 45.778 / 55.048 / 50.694 / 43.314.
Emissions identical to the digit likewise.

Uncapped the same change is worth 3.16 bn (24.307 -> 21.149 at
`conv_central` with GMP partial). Under a binding cap, gas generation is
limited by the cap rather than by fuel, so additional fuel has no value.

**Pairs with F-8: the abatement lever is EFFICIENCY, not fuel
availability.** Same fuel, same plants, opposite conclusions.

CONSEQUENCE: the GMP deliverability arms inform baseline and gas-shadow
work and are inert for every capped pathway. The midstream investment
case (M-1) is therefore meaningful only for uncapped pathways.

### F-10 — The EaaS channel is worth 8.9 to 21.7 bn under NDC caps
**STRENGTHENED — the superseded finding measured one case; this measures
four.** `conv_central`, public-only vs EaaS:

| case | public-only | EaaS | saving |
|---|---|---|---|
| ndc2_unconditional | 45.778 | 26.754 | 19.02 bn (−41.6%) |
| ndc2_conditional | 55.048 | 43.298 | 11.75 bn (−21.3%) |
| ndc3_unconditional | 50.694 | 28.995 | 21.70 bn (−42.8%) |
| ndc3_conditional | 43.314 | 34.419 | 8.90 bn (−20.5%) |

The UNCONDITIONAL cases benefit roughly twice as much. They are the
tighter caps, so the public budget binds harder and decoupling from it is
worth more. That is the mechanism stated by the pattern rather than
asserted.

The superseded F-27 reported 8.895 bn on `ndc3_conditional` alone;
that figure is reproduced exactly at 8.90 bn.

### F-11 — Every NDC cap arm is non-compliant through leakage
**PROVISIONAL.** All four cap arms flag leakage above the 10% threshold,
34.2–45.0% at `conv_central`. The cap is grid-only by construction:
`emissions_by_year` measures grid + selfgen, `emissions_capped_by_year`
regulates grid only. Under a binding cap the model displaces load to
priced self-generation, whose emissions sit outside the boundary. The NDC
is met on paper while national emissions rise.

Conversion reduces leakage (54.0 -> 39.6% at ndc2_unconditional; 50.4 ->
34.2% at ndc3_conditional) because more electricity per tonne of grid CO2
means less load needs dumping.

NOT a defect — the builder deliberately separates measurement from
regulation. UNTESTED COUNTERFACTUAL: `selfgen_in_emissions_cap=True`
gives an economy-wide cap where leakage is structurally impossible. Not
exposed by any runner.

---

## Section 4 — Identification

### F-12 — Total solar is identified; the public/EaaS split is not
**CONFIRMED, and the unconstrained arm closes the mechanism.**
Lexicographic min/max on total EaaS MW at fixed Z*, `ndc3_conditional`,
tol 1e-6:

| capital_case | budget | min MW | max MW | interval |
|---|---|---|---|---|
| tight | 4.552 bn | 36,686.8 | 47,528.0 | 10,841.2 |
| moderate | 7.738 bn | 13,680.9 | 47,528.0 | 33,847.1 |
| unconstrained | none | **0.0** | 47,528.0 | 47,528.0 |

Z* = 34,419,113,042 to 068 across every arm — a 26 USD spread on 34 bn.

**The maximum is 47,528.0 in all three.** It is the bankability ceiling,
set by tariff and capex, independent of the budget. Only the floor moves:
remove the budget and it goes to exactly zero, because public solar is
then unconstrained and costs the same. Tighten the budget and the floor
rises.

So the split is bounded BELOW by the public budget and ABOVE by
bankability, and the interval is the gap between two constraints binding
at opposite ends. A sufficiently tight budget would close it.

Total solar is invariant at ~47,521 MW across every budget level. **That
quantity is identified and reportable; the split is not, except where a
constraint forces it.**

REPORTING RULE: any EaaS figure must carry both its `capital_case` and
its `tariff_case`, and be reported as a point in the stated interval.

### F-13 — The result is robust to the tariff
**CONFIRMED.** `tariff_low` (75), `tariff_central` (95), `tariff_high`
(115 USD/MWh): Z* = 34,419,113,042 / 050 / 050. The tariff changes how
often bankability binds — binding years 8 / 3 / 1 out of 30 — and moves
the normal-solve split, but since the split is degenerate over a wide
interval the ordering is not interpretable.

CONDITIONAL on the current capex treatment: register L16 records that the
distributed capex premium is unimplemented, so bankability binds less
often than it would with it.

---

## Section 5 — Against NIRP

### F-14 — NIRP's gas trajectory never binds on the cost-optimal path
**PROVISIONAL.** `gas_ndc_17gw` (NIRP-derived cap, nameplate basis:
11,932 MW 2030 / 22,113 MW 2035 / 27,201 MW 2041 / 23,906 MW 2045) and
`gas_unconstrained` return BYTE-IDENTICAL results. The cap sits ~3.7x
above the cost-optimal 2045 capacity and never binds; gas capacity FALLS
from the 10,947 MW starting fleet.

The divergence between this thesis and NIRP on gas capacity is therefore
NOT a cost disagreement. Costing NIRP's path would require a capacity
FLOOR, not a ceiling. As a ceiling the trajectory is non-binding and the
comparison is qualitative: **NIRP plans capacity the fuel supply cannot
run.**

### F-15 — NDC 3.0's "17 GW" is a derated figure; the nameplate commitment is ~22 GW
**PROVISIONAL.** The NIRP-derived series is on a derated basis (~0.77).
Converting to nameplate gives 22,113 MW at 2035. Cross-check: NDC 3.0
describes its gas target as roughly double current levels; 22,113 / 10,947
= 2.02x, while the underated 17,027 / 10,947 = 1.55x does not match.

Both NDC 3.0's stated 17 GW and NIRP's derived 17,027 MW are EFFECTIVE
figures. Any comparison against installed capacity is comparing different
quantities unless the derate is stated. The 0.77 derate is DECLARED, the
one number in the chain not traceable to a NIRP table.

### F-16 — NIRP's 2045 hydro exceeds independent exploitable potential
**PROVISIONAL.** NIRP builds 20.6 GW by 2045 against its own stated
ceiling of ~21 GW — 98.1%, no margin. Against the Energy Commission of
Nigeria's independent estimate of 14,120 MW exploitable large hydro plus
~3,500 MW small hydro (17,620 MW), it is **116.9% — it exceeds total
exploitable potential.**

NIRP's 21 GW is a MODELLING ceiling: 18.6 GW candidate mixing
site-specific and GENERIC plants, generic capped at 15 GW nationally,
zonal allocation from 24 sites in JICA (2019). A plan consuming 98% of
its own ceiling and 117% of an independent estimate is an upper bound
presented as a forecast.

### F-17 — The demand-base gap with NIRP is definitional, and NIRP's figure fails its own check
**PROVISIONAL.** This model: 37.09 TWh, gross generation at the busbar
(NERC 2024). NIRP: 30.5 TWh for the same year. Gap 6.6 TWh, 18%.

Four terms account for it: basis year (NIRP anchors 2022 observed 32.3
TWh); metering point (this model's loss ladder runs 37.09 gross -> 7% TLF
-> 34.50 to DisCos -> 36.4% ATC&C -> 21.97 collected, and NIRP does not
state which point its figure occupies — it falls between at-DisCo and
collected, so the two may not disagree at all); exports, inside NERC
generation and outside NIRP's domestic demand; and the DESG delay
assumption.

**NIRP's 2024 modelled grid demand (30.5) is BELOW its own 2022 observed
figure (32.3)** — a forecast starting below its base year, implying a 5.6%
contraction, caused by the DESG delay assumption. Its early-year figures
are a weak calibration benchmark. Its 2045 on-grid demand also appears as
222 TWh at p.49 and 297–301 TWh in Tables 6 and 15, a ~75 TWh
unexplained internal spread.

---

## Section 6 — What bounds this thesis's own claim

### F-18 — The solar-dominant pathway is contingent on hydro not expanding
**CONFIRMED at the central configuration.**

    hydro_flat       21.154 bn   solar 2045 19,601 MW
    hydro_committed  20.409 bn
    hydro_ndc         8.925 bn   solar 2045  3,260 MW

Under NDC-scale hydro, solar falls 83% and system cost falls 58%. **If
hydro can expand to NDC scale, solar is not the residual and the headline
result changes character.**

The defence must be made on deliverability, not on modelling: `hydro_ndc`
requires +7,411 MW over five years = 1,482 MW/yr = 2.1 Zungeru-equivalents
per year, against 8–12 year FID-to-commissioning for large hydro and
nothing at that scale currently under construction. It passes the resource
test (59% of ECN exploitable) and fails lead time. NIRP passes lead time
more easily and fails the resource test (F-16). Different failure modes;
state both.

CALIBRATION NOTE: only `hydro_flat` reproduces the 2024 backcast (−3.4%).
The others give −8.6%, because raising exogenous hydro displaces gas in
the balance.

### F-19 — Storage magnitude is not a cost optimum
**STRUCTURAL.** Once gas reaches its floor, storage absorbs the entire
firm-capacity residual, so its quantity is insensitive to further
financial variation (F-3). Root cause is the flat
`capacity_credit['storage'] = 1.0` (register L5): with no declining ELCC,
storage's marginal firm contribution never falls and no interior optimum
exists.

The COMPOSITION claim — gas exits, storage displaces it at 4.0 MWh per MW,
the storage duration — is sound. The MAGNITUDE claim is not. Do not report
storage magnitude as an optimisation result.

### F-20 — Retirement is the second-largest physical lever
**PROVISIONAL.**

    gas_retire_none        18.394 bn   storage 48,806 MWh
    gas_retire_vintage_50  20.416 bn   storage 63,870 MWh
    gas_retire_vintage_30  21.154 bn   storage 64,689 MWh  (central)
    gas_retire_vintage_40  21.362 bn   storage 64,689 MWh

Range 2.97 bn. Register L8 records the vintage schedule as an ANALYST
ESTIMATE, not traced to NERC, NDPHC, TCN or company filings, with
confidence by capacity 35.8% high / 47.2% medium / 17.0% low.

Note `vintage_40` costs MORE than `vintage_30` despite retiring less — a
timing effect, not a level effect. Worth understanding before it is
reported.

---

## Section 7 — Method findings

### M-1 — Midstream cost per unit deliverability (post-hoc)
**METHOD, not a result.** GMP arm at I=0.5 requires +51.2 TWh_th/yr of
power-sector deliverability by 2030; at I=1.0, +102.5. Whole-programme
numerator over whole-programme denominator, which avoids allocating
specific corridors (a composition fallacy): 22 bn pipelines-only / 102.5
= 215 USD-m per TWh_th/yr (LOWER, excludes CPF and upstream); 60 bn full
value chain / 102.5 = 585 USD-m (UPPER, includes export, industrial, CNG).
Implied midstream investment 11–30 bn at I=0.5, 22–60 bn at I=1.0 —
**15 to 40 times the 750M DARES envelope**, which sharpens F-2.

Meaningful for uncapped pathways only (F-9).

### M-2 — Two errors partly cancelling are undetectable from internal consistency
**METHOD.** Register L17: the bankability constraint discounts at the
retired pre-F1 blend (14.4%, 70% composed of the 18% naira anchor F1
removed from the objective) over the model horizon rather than the
commercial tenor. The rate is too HIGH and the horizon too LONG, so at the
first vintage the two nearly cancel: annuity factors 6.822 against 6.469,
about −5%. The offset does not hold at later vintages.

This is the same shape as the Phase 1 calibration finding, where baseline
NPV moved 5.98 -> 8.99 -> 11.28 -> 273 -> 16.14 bn and two parameter
errors partially offset. **Both were found by checking each input against
its own basis, not by checking the model against itself.**

---

## Scoped conditions on every result above

    L5   storage capacity credit is flat 1.0 -- magnitude is a residual
    L8   gas retirement vintages are an analyst estimate
    L13  the two deliverability families are not comparable unlabelled
    L16  EaaS and public solar are physically identical -- financing
         comparison only, nothing about siting
    L17  bankability discounts at the retired blend over the wrong
         horizon; every EaaS result is conditional on it
    P2-9 gas_avail_nameplate 0.80 against a measured 29.5%, no sourced
         central arm exists (F-5)
    Sliced shape factors season_solar_factor and hydro_season_factor are
         [SOURCE NEEDED]. NO SLICED RESULT IS FINAL until they are
         sourced -- which is every result in this file.
