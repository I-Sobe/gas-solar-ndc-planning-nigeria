# Provisional findings

> **SUPERSEDED (marked 2026-10-10, TASK 43).** `docs/findings_provisional_pre_om.md`
> supersedes this file, and `docs/findings_frozen.md` supersedes both. This file is a
> superseded epoch, kept for reference, and is not to be quoted as a result. Its F-12, F-26
> and F-27 text carries the older "substitutes one-for-one" and narrow-identification wording
> (partly annotated UNIDENTIFIED, TASK 30 D and TASK 41); F-12 as corrected 2026-10-08 says
> the EaaS/public split is unidentified in every capital case, binding ones included.

Standing document. Every finding is logged the moment it appears, with
the arm that produced it and its status. Results during correction are
verification instruments, not thesis findings — but the INTERPRETATION
is logged now so the final re-run confirms or refutes rather than
starting cold.

Status: PROVISIONAL (needs re-run) / STRUCTURAL (architectural, will not
change with parameters) / RETRACTED.

Reporting horizon is **2045**. 2046-2054 is end-effect buffer. Any
number quoted from a buffer year is invalid — this rule was tightened
from the handover's 2050 after the F1 storage divergence was measured at
14% (2045), 42% (2050), 70% (2054), showing terminal effects reach back
to 2050.

---
# Provisional findings — Phase 1.6 / 2.5 session

*Append to `docs/findings_provisional.md`. **Renumber on merge** — F-10 onward is
provisional and may collide with contributions from other sessions. Status codes
follow the existing convention: PROVISIONAL / STRUCTURAL / RETRACTED, plus
SUPERSEDED where a later session has overtaken the result.*

*Session scope: Phase 1.6 (demand growth arms) and Phase 2.5 (end-of-horizon
accounting, build bounds, horizon). Pre-dates F1/F2 tranche work.*

---

## F-10 — Historical generation CAGR cannot source demand growth
**STRUCTURAL (methodological).** Nigerian grid generation is a *supply* series:
flat because gas supply, wheeling capability and the payment chain held it flat.
Calibrating exogenous demand growth to it assumes the constraint persists, then
reports modest capacity needs *because* demand was assumed not to grow.
Circular, and the same error class as the superseded 23.08 TWh demand base — a
quantity produced by the system's dysfunction read as a statement about
underlying demand.

Corroborated externally: NIRP 2024 (Table 14) derives its national forecast
bottom-up from population, GDP, electrification and loss trajectories, explicitly
rather than from historical generation.

Consequence: SSA-estimated income elasticities are rejected as a central source
for the same reason — they are estimated on supply-rationed systems. Adopted
ε = 0.76–0.90 from Liddle, Smyth & Zhang (*Energy Economics*), middle-income
non-SSA panel.

## F-11 — Per-connected-person grid generation FELL over 2010–2024
**STRUCTURAL (empirical, sourced).** Decomposition against World Bank population
and access data:

| Basis | Generation | Denominator | Intensity |
|---|---|---|---|
| Total population, 2010–2025 | +3.14%/yr | +2.39%/yr | **+0.73%/yr** (76% demographic) |
| Total population, 2014–2024 | +1.53%/yr | +2.27%/yr | **−0.73%/yr** |
| **Connected** population, 2010–2024 | +2.63%/yr | +4.36%/yr | **−1.66%/yr** |

Grid generation per connected person fell from ~327 to ~258 kWh (−21%), negative
on every window tested. Connected population grew faster than generation
throughout: Nigeria connected people to the grid faster than it added
electricity to send them.

This is the Tier 1 → Tier 2 gap observed directly, and it is the empirical basis
for F-10. Workbook: `data/demand/step1_per_capita_diagnostic.xlsx`.

Caveat: WDI access includes off-grid/DRE, so the connected-basis decline is
exaggerated in magnitude though not in sign. True intensity trend lies between
+0.2%/yr and −2.7%/yr — at or below zero either way.

## F-12 — Emissions caps are invariant to the demand growth arm
**STRUCTURAL.** Caps rebuilt from a `constrained_continuation` baseline are
**byte-identical** to those from `organic_central`. Baseline gas is
deliverability-bound from 2026 (2025 runs 24.57 of 25.62 TWh_e available), so
`Ebase` is set by the gas trajectory, not by demand growth.

The NDC caps are therefore robust to any future revision of `g`, which decouples
the cap chapter from the demand chapter. Same mechanism as F-7 (emissions follow
gas *generation*, which is deliverability-capped regardless of installed
capacity) observed one phase earlier and from the demand side.

## F-13 — Decarbonisation by non-development
**STRUCTURAL (policy corollary of F-12).** Because the cap is anchored to a
gas-constrained baseline, Nigeria meets its power-sector NDC trajectory largely
by *remaining supply-constrained*. Under the `constrained_continuation` arm
(3.14%/yr, sourced from the historical record) baseline NPV is 13.59 bn against
16.14 bn for `organic_central` — a 15.8% cost saving achieved by not developing.

Uncomfortable and defensible. Belongs in the policy discussion, not the results
chapter.

## F-14 — The min-build floor was forcing capacity, not preventing delay
**STRUCTURAL (defect, resolved).** The 100 MW/yr floor was justified in its
docstring as preventing "pathological all-delay solutions that leave 2025-2030
with no solar." Removal test: 2025 went 100 → 0 MW, 2026 went 173.41 → 273.41,
every other year identical, **cumulative build unchanged to the megawatt**.
Objective −0.10%, consistent with a one-year deferral of ~$145M of 2025-vintage
capex.

The justification is refuted by the backcast arithmetic: gas 24.57 + hydro 11.47
+ baseline solar 0.88 = 36.9 TWh against 37.09 demand, so **2025 = 0 new solar is
correct**. A model that builds capacity it does not need is the distorted one.

The floor also existed at 23 hardcoded call sites against a canonical value of
0.0 — the fourth duplicate-source-of-truth defect found in this codebase.

## F-15 — A flat annual build cap is wrong in shape at both ends
**STRUCTURAL.** The unsourced 2,000 MW/yr solar cap bound in 2042–45 (annual) and
2039–45 (sliced). It permitted 2,000 MW in 2026 when NIRP judges ~240 MW
achievable, and capped 2042–45 at 2,000 when NIRP builds 6.0–8.3 GW/yr.

**A constant rate against compounding demand binds eventually at any level**, so
the terminal build sitting exactly on 2,000.0 was an artefact of the constraint's
*shape*, not evidence of a deployment limit.

Replaced with `nirp_solar_cumulative_mw()` — NIRP 2024 Annex F Table 19 solar
generation converted at the Table 8 CF, validating to 61,898 MW at 2045 against
NIRP's published ~61 GW headline. Applied to cumulative *additions* (NIRP's 2025
figure of 180 MW is below this model's 500 MW existing baseline).

## F-16 — Terminal-year salvage arbitrage
**STRUCTURAL (discovery). MITIGATION SUPERSEDED — see F-6.** With the horizon
ending at 2045, salvage credited a final-year vintage at (life−1)/life of capex —
96.7% for 30-year solar — discounted at the same factor the capex was paid at.
Net cost ~3.3% of discounted capex, about $11.5/MWh, below gas SRMC.

Measured effect once the unsourced build cap was removed: 2045 solar additions
jumped 1,635 → **17,861 MW**, 2045 gas generation fell 24.67 → **0.00 TWh_e**,
cumulative solar rose 25.3 → 39.3 GW, and **the objective FELL 0.7%**. The model
built 14 GW more and paid less.

Mitigation in this session was horizon extension to 2054 (a 2045 vintage then
serves 10 of 30 years, salvage 67%, net cost ~33% of capex). **F-6 reports that
per-tranche annuity truncation under F1 removes this structurally**, which is the
better fix. The horizon extension remains in place and its buffer rule
(2046–2054 never reported) is what the current 2045 reporting horizon rests on.

## F-17 — Capacity behaved as an accounting residual (pre-F1)
**PROVISIONAL — LIKELY SUPERSEDED by F-1 and F-3.** Under the pre-F1
blended-WACC formulation, `22_salvage_sensitivity.py` found every real-lifetime
arm (25/30/35 solar, 12/15/18 storage) byte-identical to central in both models
at three sequential test points. A 12–22% objective swing moved the plan by under
2% in the annual model and **zero** in the sliced model.

Mechanism proposed: demand exogenous, hydro exogenous, gas at its deliverability
ceiling, unserved priced out by VoLL — the energy balance has one unknown per
year and the model solves for it.

**This is probably now wrong, and should be re-tested rather than cited.** F-1
reports the objective spanning 21.888–26.299 bn across capital-recovery arms
where the pre-F1 audit returned identical values to seven significant figures,
and F-3 reports gas capacity at 2045 moving −10.5% to −29.6% across those arms
with storage moving opposite. **If cost of capital moves technology composition,
capacity is not a pure residual.**

The reconciliation is probably that asset *lifetime* is a weak lever while cost
of *capital* is a strong one — different levers, not a contradiction — but that
needs demonstrating. Re-run `22_salvage_sensitivity.py` under F1 before this
finding is either retained or retracted.

## F-18 — An unsourced bound may be masking a defect
**STRUCTURAL (methodological).** Both build bounds hid artefacts at opposite ends
of the horizon: the min-build floor concealed that no solar was needed in 2025
(F-14), and the flat max-build cap concealed the terminal salvage arbitrage
(F-16). Neither defect was visible while its bound was active.

Rule adopted: when an unsourced constraint is removed, expect the result to
change in a way that looks wrong, and check whether the constraint was
load-bearing for a reason nobody documented.

## F-19 — VoLL-dominated objectives are systemic, not incidental
**PROVISIONAL — being resolved by Phase 2.6.** Any run combining a binding NDC
cap with a binding capital envelope produces an objective dominated by the
unserved-energy penalty rather than by system cost. Affects the Monte Carlo,
DEM-2 `public_only` cells, GAS-3 and POL-1.

Sharpest evidence: under the Step 5b regression, the EaaS arm's gas premium moved
**0.00001%** (491,648,215.6 → 491,648,164.3 — solver noise on an effectively
identical problem) while the `public_only` equivalent moved **31%** (63.1 → 83.0
bn). The public arm is hypersensitive because it is budget-bound; EaaS is not.

Consequence recorded at the time: the public-vs-EaaS comparison was measuring a
penalty parameter, not the cost of capital. **Note for the 2.6 session: the
genset backstop bounds the penalty but the defect can relocate — if the EaaS arm
has no binding capital envelope, the comparison becomes "constrained vs
unconstrained", which is equally not a financing result.**

## F-20 — The annual-vs-sliced resolution premium is basis-sensitive
**PROVISIONAL.** The premium fell from ~13.4% to **6.80%** when salvage was added
(annual 16.14 → 13.377 bn, sliced 18.31 → 14.286 bn). The sliced model holds more
capital — storage plus earlier solar — so it receives a larger salvage credit and
the gap narrows.

The ~13% figure is superseded and on the do-not-report list. The premium must be
recomputed on the new horizon and again under F1; a methods finding stated
without naming its cost definition is not reportable.

## F-21 — NIRP 2024's demand forecast is aggressive on two terms simultaneously
**STRUCTURAL (external critique).** NIRP's implied income elasticity is
**ε = 1.19** — derived from its own Table 14/15 assumptions, (7.7−2.1)/(6.8−2.1)
— against a credible literature range of 0.76–0.90. That is ~35% above the
literature, and it sits on top of a 6.8%/yr real GDP path, roughly the pre-2015
oil-boom rate.

Second defect: **NIRP's 2024 modelled grid demand (30.5 TWh) is below its own
2022 observed on-grid figure (32.3 TWh)** — a forecast starting below its base
year, caused by the DESG delay assumption. NIRP's early-year figures are
therefore a weak calibration benchmark.

Also logged: Table 14's GDP terminal values are inconsistent with its stated
growth rates by ~1,000× (a units error in the terminal column, not a modelling
error).

## F-22 — Nigeria's per-capita position against comparators
**STRUCTURAL (empirical, sourced).** Nigeria 2024 ≈ 161 kWh/cap. Kenya crossed
that level in ~2009, so **Nigeria is roughly 15 years behind Kenya**. Ghana
cannot answer the question at all: **Nigeria's 2024 level is below Ghana's worst
year since 2000**, and converging to Ghana's present level (705 kWh/cap) by 2045
would require **9.24%/yr** — above NIRP's base case.

Used to calibrate the demand arms via per-capita convergence (Route B).

## F-23 — λ is sourced, and is a scenario parameter not a model output
**STRUCTURAL (parameter + correction).** NIRP 2024 gives grid ÷ (grid + self-gen)
= 32.3/59.6 = **0.542** (2022) and 35/62 = **0.565** (2025); §4.2's "self-
generation is 49% of total on-grid demand in 2024" implies ~0.51. Reported range
**0.51–0.57**, giving Tier 2 ≈ **65–73 TWh** for 2024.

**Correction to an earlier design claim.** The plan asserted that λ becomes "a
derived, reportable statistic" under the tiered architecture. It does not: with
one organic growth rate applied to three scopes, λ(t) = λ(0) by construction and
the absolute tier gap widens forever. λ becomes endogenous only when a single run
serves Tier 3 demand *and* the system can economically decline to serve — i.e.
Phase 2.6. **The genset `selfgen` variable is the mechanism that makes λ an
output; `selfgen` and the Tier 2 wedge are the same physical quantity at
different tiers and must never both be counted.**

## F-24 — Constraint-relief asymmetry is a declared design choice
**STRUCTURAL (design).** Demand arms are sourced from *unconstrained* drivers
(population, GDP, elasticity) while gas deliverability is anchored to *observed
constrained* throughput on a constraint that is commercial, not geological. The
model therefore assumes constraints lift on the demand side and persist on the
supply side, and **that asymmetry generates the solar residual**.

Defensible — a requirement and a capability are different objects, and reform can
succeed downstream while failing upstream — but it must be argued, not inherited.
Declared in `docs/scenario_coherence.md` with a coherence cross-tab and a
reporting rule (incoherent cells are sensitivity bounds only).

A third instance appears at the solar-build axis: NIRP builds almost no solar
before 2032 *because* it assumes gas recovers to 88 TWh_e by 2035, so its solar
schedule is downstream of its gas assumption. Pairing `nirp_trajectory` with
constrained gas is the pessimistic stress corner. The cross-tab needs this axis
added.

A fourth: GDP is exogenous to electricity supply, but supply constrains GDP.
High-GDP arms paired with high unserved energy are internally incoherent.
Excluding those cells removes demand, so the omission is conservative. NIRP
carries the identical omission and does not flag it.

---

## Threats logged, not yet findings

**T-1 — Distributed CAPEX may erase the EaaS advantage.** NREL ATB Distributed
Commercial PV carries a premium of **+16% (2025) rising to +37% (2045)** over
utility-scale, and on an *energy* basis the gap is roughly **2×**. The model
applies a 1.10 `required_margin`, which is a *financing* markup, not a technology
cost. The `[DIST-1..4]` offsets (no evacuation bound, no TLF, no distribution
losses, no collection risk behind the meter) have never been quantified against a
sourced premium. Caution: do **not** lift NREL's class-5 CF ratio — utility class
5 is single-axis tracking while the Nigerian 0.20 is already fixed-tilt.

**T-2 — Gas fleet efficiency is the missing abatement lever.** η = 0.287 is a
fixed scalar. Moving to ~0.35 releases ~20% more electricity from the same gas —
equivalently a ~20% cut in emissions per MWh delivered — with no new fuel,
pipeline or evacuation. Very likely the cheapest abatement option in the sector,
and the model cannot represent it. Adopted as plan item 2.8.

**T-3 — Storage magnitude may not survive the genset backstop.** In the sliced
model the backstop competes with storage in the night/peak slice and caps
storage's value at genset cost. Expect storage build to fall, which would
threaten the proposed "NIRP under-provisions storage" contribution. See also F-5,
which finds the magnitude claim unsound for an independent reason (flat
`capacity_credit['storage'] = 1.0`).

---

## Retracted or superseded this session

- **"Backcast deviation is −4.1%"** — superseded. Removing the min-build floor
  moved first-year gas 24.5688 → 24.744 TWh_e (gas picked up the 0.175 TWh the
  forced 100 MW had supplied). Deviation against observed 2024 is now **−3.4%**.
  Update wherever the old figure appears.
- **"λ becomes a derived, reportable statistic under the tiered architecture"** —
  RETRACTED. See F-23.
- **"No-salvage drives pathological delay, which the min-build floor patches"**
  (plan item 2.5 original reasoning) — RETRACTED on both halves. The balance
  constraint binds every year so the model cannot delay past need; and salvage
  makes late builds *cheaper* in net terms, strengthening the delay incentive.
  No-salvage caused a **level** error, not a timing error.
- **"The annual-vs-sliced premium is ~13%"** — superseded, see F-20.
- **"Solar is an accounting residual" as a headline** — see F-17. Likely
  superseded by F-1/F-3; re-test before citing.

---

# Session: self-generation backstop (2.6) — genset pricing, cap boundary, leakage

Findings from the session that replaced the naked VoLL penalty with a priced
private self-generation backstop, corrected the emissions-cap boundary, and
established the leakage diagnostic.

**HORIZON WARNING — read before quoting any number below.** This session ran
before the 2045 reporting horizon was tightened. Every cumulative quantity here
was cut over the FULL model horizon 2025–2054, which includes the 2046–2054
end-effect buffer. Each finding is tagged `[2054-CUT]` where this applies.
The DIRECTION of every structural finding survives re-cutting; the MAGNITUDES
do not. Re-cut before any of these enter a results chapter.

---

## GS-1 — Genset parameters are sourced, and efficiency is derived not stored
**STRUCTURAL.** Three arms in `data/cost/processed/genset_cost.csv`, eta from
four Caterpillar PRIME datasheets (C4.4 / C15 / C18 / 3512):

| arm | eta_HHV | USD/MWh | EF_e tCO2/MWh |
|---|---|---|---|
| `genset_srmc` | 0.3474 (75% load) | 277 | 0.729 |
| `genset_central` | 0.2809 (25% load) | 445 | 0.902 |
| `genset_high` | 0.0511 (A2EI metered) | 3,323 | 4.638 |

Fuel anchored Dec-2024–Feb-2026, NBS AGO/PMS at matched monthly CBN NFEM rates:
AGO 1.0157 USD/l (CV 8.3%), PMS 0.7311 (CV 8.7%). Mar–May 2026 excluded as a
crude shock and retained as the high arm. `EF_e` and `SFC` are DERIVED at build
time from eta, never stored — the same rule as the gas fleet in 1.2, so
sweeping efficiency moves cost and emissions together and they cannot drift.
IPCC chain validated independently: reproduces 2.318 kgCO2/l petrol and
2.676 diesel against published 2.31 / 2.68.

**Quotable, and it connects to P2-8.** `genset_central` eta 0.2809 sits against
the measured gas fleet's 0.287. A part-loaded Nigerian C&I diesel converts fuel
about as efficiently as the grid; the entire cost gap is the fuel price, not
the conversion. P2-8 shows the fleet's 0.287 is mostly open-cycle composition
rather than degradation — so the comparison is between a rational
open-cycle grid and a rational part-loaded diesel, not between a broken grid
and a good alternative.

## GS-2 — ZE-Gen's fuel data is one assumed parameter, not five measurements
**STRUCTURAL (a documentary finding).** Every ZE-Gen segment divides out to
exactly 2.000 L/kWh — households, MSMEs, schools, health centres, hotels. The
first two are stated to be mostly petrol, the last three mostly diesel.
Different densities and calorific values cannot both give exactly 2.000 unless
the number was assumed and applied. ZE-Gen p.5 confirms: *"a standard load
factor of 20% was applied across all clusters"* and *"approximately 2 litres of
fuel required to generate 1 kWh."*

The metered kWh (500 smart meters, 12 months) is real and usable. The litres,
the monthly fuel spends, and the reconstructed $8bn national figure all inherit
one assumption. That assumption implies eta_HHV = 5.1% on petrol, 4.7% on
diesel — high but not impossible for oversized part-loaded household sets, not
credible for a hotel's diesel running 3.2 h/day. Same defect class as the
23.08 TWh demand base and the 0.43 gas efficiency: **a number generated by an
assumption, not a measurement, and propagated because it was internally
consistent.**

## GS-3 — Self-generation is bimodal, and the two modes behave differently
**STRUCTURAL.** Aggregate reconciliation: the $8bn national SGG fuel spend at
$0.7311/l implies 10.94 bn litres/year; at 2.000 l/kWh that is 5.47 TWh/yr of
self-generated electricity. NIRP's stipulated lambda implies ~31.6 TWh/yr. Small
petrol gensets can account for at most a sixth of Nigeria's self-generation by
energy.

Consequence: small petrol sets deliver very little energy at catastrophic unit
cost ($3,323/MWh); large C&I and captive diesel delivers most of the energy at
roughly a seventh of that. **For a grid capacity-expansion model the marginal
backstop for bulk unserved demand is the second mode** — which is why
`genset_central` is the diesel arm and `genset_high` is a stress bound
representing the access tier, not a scenario.

## GS-4 — `genset_high` renames the VoLL cliff rather than removing it
**STRUCTURAL.** At $3,323/MWh the small-petrol arm sits at essentially the same
level as the VoLL-dominated Monte Carlo LCOE (~$3,600/MWh) already on the
do-not-report list. A backstop priced there does not convert the cliff into a
slope. The entire sequencing argument for 2.6 preceding Phase 3 fails if
`genset_high` is treated as central.

## GS-5 — Two arbitrage channels opened by the backstop, both closed
**STRUCTURAL.**

**D10, buffer-year cap release.** Emissions caps for 2046–2054 were set to
1e18, and the optimiser routed ~358 TWh of self-generation into the uncapped
window while paying VoLL to shed load in the capped years. Observed selfgen of
357.8 TWh against ~360 TWh of permissible buffer-window shortfall — not a
coincidence. Caps now held FLAT at the terminal in-window value, matching the
treatment `io.py::_extend_capex_series` already applied to CAPEX. Structurally
the same defect the horizon extension fixed for salvage; **the buffer window was
not inert.**

**D11, load-shedding as zero-emission abatement.** With genset CO2 inside the
cap, `m.unserved` became the cheapest abatement technology in the model: it paid
$10,000/MWh to shed 344.8 TWh rather than $277/MWh to self-generate, purely to
buy cap headroom. Physically false — unmet grid demand in Nigeria becomes
self-generation, it does not vanish. `allow_unserved` now defaults False
whenever a genset case is priced.

## GS-6 — The NDC cap is binding on the smaller part of the system
**PROVISIONAL `[2054-CUT]` — this is the headline finding and it needs
re-cutting first.** Under the correct NDC accounting boundary (cap on
`emissions_grid_only`, which is what NIRP's gas-power slice commits), the model
is COMPLIANT while emitting far more than the budget it satisfies:

| arm | grid Mt | uncounted diesel Mt | x cumulative cap |
|---|---|---|---|
| `genset_srmc` | 303.4 | 472.6 | 1.24x |
| `genset_central` | 303.4 | 584.5 | 1.54x |
| `genset_high` | 303.4 | 3,005.0 | 7.89x |

Widening the boundary to total emissions is **INFEASIBLE in all three arms**:
NDC3 unconditional is unattainable under a tight capital envelope once
self-generation is counted. Independent of genset price — the quantity is set
by capital and the obligation, not by cost.

All four NDC cases in `02_run_ndc_caps` flag at 53–67% leakage.

**Re-cut required.** Cap total 380.6 Mt spans 2025–2054; over the 2045 window it
is roughly 256 Mt and the diesel wedge shrinks proportionally. Direction and
sign will survive; the multiples will not.

**Internal consistency check passed:** selfgen quantity identical across all
three arms (0.00% spread, 647.9 TWh) while emissions scale exactly with EF_e —
0.729 / 0.902 / 4.638 reproduced to three decimals. Quantity is set by
constraints, emissions by the derived factor. The GS-1 derivation chain is
sound.

## GS-7 — Carbon pricing is a transfer, not an abatement instrument, when capital binds
**PROVISIONAL `[2054-CUT]`. Extends F-7 from grid to grid-plus-diesel.**
Under `public_tight`, emissions are constant at 903.1 Mt across a carbon price
sweep from $0 to $150/tCO2 — **exactly constant, not approximately** — while
cost rises 78.4 → 104.3 bn. The price extracts $43.2bn at $150/tCO2 and abates
precisely nothing.

`eaas_tight` on the same sweep moves 389.5 → 167.3 Mt between $15 and $20, with
solar share going to 100%. Same price, same technology costs; the only
difference is available capital.

**Relationship to F-7.** F-7 establishes that grid emissions follow gas
generation, which is capped by deliverability. GS-7 shows the property extends
to the diesel wedge: selfgen quantity is set by the capital envelope, so
carbon pricing cannot move it either. **A carbon price is only an abatement
instrument if capital exists to abate with.** Absent that it is a revenue
instrument. Directly usable for a country negotiating climate finance.

Depends on `unserved` being unavailable (D11) and the backstop being unbounded.
Both are declared assumptions doing work here.

## GS-8 — The saving is avoided private diesel, not an avoided penalty
**PROVISIONAL `[2054-CUT]`.** Cost decomposition reconciles exactly —
real + selfgen + VoLL − salvage = objective in all five cases, after salvage was
added as a missing component (it is a credit, subtracted in `system_cost_npv`,
and its omission overstated the decomposition by ~2.5bn on the baseline).

| case | real bn | selfgen bn | private share |
|---|---|---|---|
| NDC3 uncond, public tight | 10.5 | 75.6 | 88% |
| NDC3 uncond, **EaaS** tight | 16.3 | 31.5 | 66% |
| NDC3 cond, public moderate | 13.1 | 53.8 | 80% |
| NDC3 cond, **EaaS** moderate | 15.9 | 39.0 | 71% |

EaaS spends MORE public money and Nigerians spend FAR LESS private diesel. The
VoLL column is zero everywhere, by construction under D11. **The saving is a
transfer from private diesel expenditure to public capital, not the
disappearance of a penalty parameter.**

**Caveat, and it is load-bearing:** the EaaS arm has no binding constraint at
the default 65M USD/TWh tariff — it builds until demand is served and returns
an identical answer to every input. Every "EaaS saves X" figure here is
constrained-versus-unconstrained until FIN-2 imposes a real tariff constraint
(D17). Do not report the saving until then.

## GS-9 — Obligation timing: front-loading is cheaper, not dearer
**PROVISIONAL `[2054-CUT]`. RETRACTS an earlier claim from the same session.**
Three grid-service-obligation trajectories at constant 0.45 horizon-average, all
feasible:

| trajectory | obj bn | selfgen TWh | solar MW | built early |
|---|---|---|---|---|
| front 0.40→0.50 | 90.45 | 639.6 | 25,964 | 48.7% |
| flat 0.45 | 92.54 | 631.4 | 29,265 | 28.0% |
| back 0.50→0.40 | 94.69 | 627.1 | 32,566 | 12.3% |

The trajectory genuinely controls build timing (share-early moves 48.7 → 12.3%).
Deferring build means building against LARGER demand: back-loading needs 6.6 GW
more capacity for the same standard, which outweighs the capex decline from
waiting.

**Retracted:** the earlier claim that a tighter near-term obligation gives worse
lifetime service (550.2 vs 496.4 TWh shortfall). That compared different LEVELS
at the feasibility boundary; this compares TIMINGS at a constant level inside
the feasible band. The earlier non-monotonicity was a boundary effect.

Trade-off worth stating: front-loading is cheaper for the system but leaves
marginally MORE self-generation (639.6 vs 627.1 TWh) — cheaper for the state,
slightly worse for consumers.

**Inherits F-5 and P2-9.** The MW figures are firm-capacity driven, so they
carry the flat `capacity_credit['storage'] = 1.0` defect and the 0.80 gas
availability assumption. Re-run after both are resolved.

## GS-10 — A horizon-total obligation stops discriminating under a rising deficit
**STRUCTURAL.** `09b` mode sensitivity: `total` mode gives shortfall of
145.951023 TWh at eps 0.20, 0.15, 0.10 AND 0.08 — identical to six decimals
across a 2.5x tightening. The constraint is slack in all four because the
horizon aggregate sits below even the tightest bound. `annual` mode binds
immediately (152.9 TWh at 0.20, infeasible at 0.15).

Under a monotonically rising deficit, a horizon-total service obligation is
satisfied by early-years slack and permits 95% service through the 2030s and
30% in the 2050s while reporting compliance. **`annual` is the only defensible
mode.** Four different policies produced one plan.

Related: aggregate-versus-per-year is a recurring trap. At eps=0.20 the horizon
aggregate was 2.46% of demand while 2032–2036 and 2054 sat pinned at exactly
0.2000 — an aggregate binding test reads False while six years are hard against
the bound.

## GS-11 — The obligation is a service standard, not an adequacy standard
**STRUCTURAL (a definitional finding).** The constraint bounds
`selfgen + unserved`, not `unserved` alone — with a backstop, `unserved` is zero
by construction and a constraint written on it never binds. But a household
running its own genset was not grid-served while no capacity was short, so
LOLE and EENS do not describe it.

Renamed accordingly: `max_offgrid_fraction` / `grid_service_mode` /
`m.grid_service_obligation`. It is a **grid service obligation** — a policy
instrument — not a reliability standard. Three quantities, three meanings, and
they must be reported separately:

| quantity | meaning |
|---|---|
| `selfgen + unserved` | grid service shortfall — what the constraint bounds |
| `unserved` alone | truly unmet demand GIVEN the backstop assumption; ~0 by construction |
| selfgen cost / CO2 | the welfare and emissions burden the shortfall imposes on private actors |

**Never quote `unserved` as evidence demand is met.**

## GS-12 — NIRP's self-generation figure is stipulated, and its elimination is assumed
**STRUCTURAL (a documentary finding). D26, blocking 2.7.** Document-level search
of NIRP 2024 establishes that the ~27 TWh self-generation quantity is
**STIPULATED, not derived**: no methodology, no fuel volume, no SFC or
efficiency assumption, no external citation. The operative statement is
§4.2 p.25 — *"it was assumed that self-generation makes up 49% of total on-grid
demand in 2024, with this percentage decreasing linearly to reach 0% in 2035"* —
attributed in Annex B p.46 only to *"observations made in the most recent
years,"* with no observer, dataset, or year range named. The sectoral split is
explicitly *"in the absence of actual data."*

Two consequences.

**Tier 2 magnitude is uncertain by ~40%**, spanning roughly 46–68 TWh across
NIRP-ratio, NIRP-additive and IEA-bottom-up routes. lambda = 0.51–0.57 must be
recorded as STIPULATED, not SOURCED. Tier 2 becomes a swept parameter with
declared sources per arm. See `docs/tier2_reconciliation.md`.

**NIRP's demand forecast assumes away the quantity the cap fails to account
for.** Table 2 has self-generation going 27,000 → 22,000 → 0 GWh across
2025/2030/2035 and staying at zero through 2045. §2.4 p.21 records that even
2035 is workshop consensus rather than analysis — *"the consensus from the
discussions is that the 2030 target is unattainable."* No feasibility model, no
retirement curve, no consumer-switching model is presented for the phase-out.

Against GS-6, which shows 600+ TWh of cumulative self-generation and NDC
infeasibility once it is counted. **This is a critical-review chapter finding
and it holds independent of which Tier 2 arm is chosen** — the elimination
assumption is what matters, not the 2024 baseline magnitude.

## GS-13 — The residual finding survives the backstop
**STRUCTURAL. Confirms P2-3 and F-5 under a new perturbation.**
`22_salvage_sensitivity` re-run post-backstop: solar and storage identical to
five decimal places across all six lifetime arms, against a 12.67% (annual) and
14.73% (sliced) cost swing.

The re-run was scheduled precisely because 2.6 was expected to open a degree of
freedom — serve-versus-pay becomes economic rather than penal. It did not.
**The backstop is a SUPPLY channel, not a CAPITAL channel:** it relieves the
energy balance but does not change what the budget buys or what the gas ceiling
allows, so solar remains the residual that closes the balance.

Capacity is an accounting residual as a property of Nigeria's constraint
structure rather than of the formulation — now demonstrated across salvage
on/off, six lifetime arms, two temporal resolutions, and with a priced backstop
present.

Noted, not investigated: the annual `no_salvage` arm deviates 480 MW (0.81%,
sub-threshold) while sliced does not. Degenerate arm (all lives = 1 yr).

## GS-14 — Nigerian fuel is USD-priced in practice
**STRUCTURAL. Supports the 3.1 currency convention.** Over Dec-2024–Feb-2026
the naira moved 1,553.7 → 1,356.3 per USD (−12.7%) and both NGN pump prices
swung ~±20%, yet USD prices held within ±9%: AGO CV 8.3%, PMS CV 8.7%.

Corroborated by market structure: on 13 July 2026 Dangote suspended
naira-denominated sales and moved to USD pricing (diesel $1.087/l), citing
insufficient crude through the naira-for-crude programme. Its gate price sits
within 7% of the independently derived clean-window mean of $1.0157.

**This converts the real-USD modelling convention from an assumption into an
observed property of the market**, and strengthens the case for sourcing
financing rates directly in USD rather than converting a naira rate (3.1 §6).

---

## Retracted from this session

- **"−94.6% objective delta from pricing the backstop"** — measured the size of
  the VoLL assumption, not the economics of self-generation. Unbounded selfgen
  routed 569 TWh (more than total horizon demand) and any sub-VoLL backstop
  would have produced the same figure.
- **"EaaS saves 98.7% / 80% / 69.6%"** — same disease, second occurrence. The
  public arm's objective was ~98% VoLL penalty; the "saving" was the penalty
  disappearing. See GS-8 for the correct decomposition.
- **"Tighter near-term obligation gives worse lifetime service (550.2 vs
  496.4 TWh)"** — compared different eps LEVELS at the feasibility boundary,
  not timings at constant level. Superseded by GS-9.
- **"lambda = 0.51–0.57 is SOURCED from NIRP 2024"** — NIRP stipulates the ratio
  without deriving it. Corrected to STIPULATED. See GS-12.
- **"Tier 2 makes lambda endogenous"** — moving the assumption from a growth
  rate to a tier ratio is not eliminating it. lambda becomes endogenous only
  under Tier 3 with the backstop present.
- **"emissions_tco2_by_year is grid-only, so the Pareto axis is wrong"** —
  investigated and false. `optimize_experiments.py:2234` already reads
  `m.emissions_by_year` = total. No defect.
- **"The EaaS arm has no capital constraint because 99c omits the tariff"** —
  the default tariff is 65M USD/TWh, LOWER than the 95M that `03` sets, so a
  missing tariff cannot explain unconstrained building. The real cause is that
  EaaS reaches zero shortfall and stops; nothing else binds. Logged as D17.

---

## Watch items carried out of this session

| # | Item | Owner phase |
|---|---|---|
| D17 | EaaS arm unconstrained at the default tariff; every EaaS saving is constrained-vs-unconstrained | FIN-2 |
| D25 | `04_run_fin2_tariff_sweep`, `12_run_dem1_demand_sensitivity` are `NotImplementedError` stubs | FIN-2 / DEM-1 |
| D21 | `11_run_rel3` threshold logic disagrees with its own `status` column | REL cleanup |
| D23 | `eps_levels` dropped at `optimize_experiments.py:774`; sweep silently uses `reliability_levels_log()` | REL cleanup |
| D24 | Six REL scripts sweep four different eps axes; needs one documented rationale. eps is a POLICY instrument with no external measurement — justify methodologically, anchored on Nigeria's observed lambda | REL cleanup |
| D13 | `19_run_carbon_price_sweep` arms too weak to bite against a $445/MWh backstop; the tipping point IS the result | carbon policy |
| D14 | APERC reporting layer. Motivated by the observation that the model has NO instrument pricing diesel's health, accessibility or acceptability externalities | reporting |
| D26 | NIRP self-generation stipulated; Tier 2 a swept parameter | 2.7, BLOCKING |
| — | **Every cumulative number in this session is a 2054 cut and must be re-cut to 2045** | before write-up |

## Method notes worth carrying

- An artefact reproducing IDENTICALLY across arms is one assumed parameter, not
  N measurements. ZE-Gen's 2.000 L/kWh across five segments and two fuels;
  NIRP's 26.7 and 27.0 TWh across three years.
- A large objective delta is usually the VoLL parameter, not the mechanism under
  test. Happened twice in this session.
- Rename when meaning changes. `genset_case` defaulting to None silently
  reverted two headline NDC runners to pre-2.6 behaviour and produced
  clean-looking wrong tables; the tell was total emissions equalling
  `emissions_grid_only` exactly. Now REQUIRED, no default.
- Inventory before editing. Enumerate every consumer of a changed quantity
  first; do not discover them one traceback at a time.
- Stale `results/*.json` silently produce pre-fix numbers. Re-run producers
  before consumers.

---

# Session: physical layer (2.3–2.13) — firm capacity, demand shape, gas

Findings from the session that added the firm-capacity constraint, sourced
the demand shape, and made gas buildable. These PRECEDE F1. Several are the
evidence F-5, F-6 and F-7 rest on; cross-references noted.

Reporting horizon 2045 throughout. Where a number was originally quoted to
2050 or 2054 it has been re-cut or retracted.

## P2-1 — Peak adequacy, not energy, is the binding physical constraint
**STRUCTURAL.** The firm-capacity constraint added in 2.3 is slack in 2025
and 2030 under every arm and binds from 2035 onward — 21 of 30 years.
Before 2.3 the model solved an energy problem and called it a capacity
expansion plan: any evening peak could be met by letting the priced diesel
backstop run, because no constraint required installed capacity.
Duals decline 65,875 -> 9,016 USD/MW (2035 -> 2054) purely through NPV
discounting, not because adequacy eases.

## P2-2 — Storage was structurally dominated, not merely uneconomic
**STRUCTURAL.** At annual resolution `discharge - charge <= (RTE-1)·charge
<= 0`: storage could only ever subtract energy while costing capex, so
optimal storage was identically zero for ANY cost. Confirmed independently
by `14_run_str1_storage_role.py`, suspended for exactly this reason. No
parameter sweep could ever have changed it. The firm-capacity constraint
supplied the missing value channel and storage built for the first time.
Register L5, moved PENDING -> RESOLVED.

## P2-3 — Storage magnitude is a constraint residual, not an optimum
**STRUCTURAL. Superseded in root-cause by F-5; this is the earlier
evidence.** Under `storage_cc_case` arms 0.50 / 0.80 / 1.00 the storage
build came out at exactly 154,775 / 96,735 / 77,388 MWh — precisely
`77,387.72 / CC` in all three. Duals scaled identically (105,400 / 65,875 /
52,700 = `52,700 / CC`) and the objective moved in exact proportion to
storage MWh. With gas pinned, solar credited at zero and hydro exogenous,
storage was the only supplier of an incremental firm MW, so its quantity
was computed rather than chosen. F-5 identifies the persisting root cause
as the flat `capacity_credit['storage'] = 1.0`.

## P2-4 — New gas displaces storage, not solar, and adds no generation
**STRUCTURAL. Pre-F1 statement of what F-7 confirms under F1 arms.**
Gas generation is bit-identical across `gas_none` and `gas_unconstrained`
through 2045 (24.7 / 26.3 / 26.9 / 27.6 / 28.3 TWh), as are the gas shadow
prices. Cumulative emissions to 2045 are identical to the tonne:
**358.8 Mt in both arms.** Storage falls 46% (82,071 -> 44,377 MWh) for a
0.6% cost reduction. Under a binding deliverability constraint new thermal
capacity is bought for firm capacity only — there is no more fuel.
Nigeria's actual pathology, plants that cannot be fuelled, reproduced
endogenously rather than assumed.

## P2-5 — Gas was unbuildable, contradicting the policy being modelled
**STRUCTURAL.** `m.gas_add` was `pyo.Param(T, initialize=0.0)` — a fixed
parameter. The model could not build a megawatt of gas while evaluating
NDC 3.0 pathways, and NDC 3.0 targets 17 GW of gas-fired capacity by 2035,
roughly double current levels and more than its solar and hydro targets
combined. Converted to a Var in 2.9.

## P2-6 — NIRP's own trajectory IS the NDC's 17 GW
**PROVISIONAL.** Deriving capacity from NIRP Tables 19 and 22 as
`generation / (8760 x capacity factor)` gives gas capacity of 17,027 MW in
2035 — the NDC target, to three significant figures. The NDC's gas
commitment is NIRP's modelled path, not an independent policy choice. NIRP
implied series: 9,188 (2030) / 17,027 (2035) / 20,945 (2041) / 18,408
(2045), on a derated basis (~0.77 of nameplate).

## P2-7 — Gas capex derived from NIRP, and the method self-validates
**PROVISIONAL.** NIRP states no capex per MW. Combining Tables 19, 22 and
23 gives 1,317 USD/kW (17.0 bn / 12,905 MW added, 2030-2045); year-by-year
values cluster 877-1,393. The same method on solar returns 905 USD/kW
(55 bn / 60,756 MW), consistent with independent solar cost data — a
derivation that lands correctly on the technology that can be checked
independently. Replaces an unsourced bare literal of 900,000 USD/MW with
no technology attribution. Status: DERIVED, not stated by NIRP.

## P2-8 — Nigeria's fleet is 60.7% open cycle, which explains most of eta
**PROVISIONAL, and it revises an earlier framing.** NERC Table 4.3's
turbine-type column: 6,896 MW open cycle (60.7%), 2,380 MW combined cycle
(20.9%), 2,085 MW gas-steam (18.4%). A capacity-weighted ISO expectation
across that mix is 0.3375 HHV against a measured 0.287 — an implied
degradation factor of 0.85. The correction plan describes 0.287 as
"brutally low, indicating degraded, high-heat-rate plant"; **most of the
shortfall is fleet composition, not condition.** Open cycle is cheap to
build and tolerant of unreliable gas — a rational response to the
deliverability constraint. Consequence: Phase 2.8's proposed eta
correction is two separable things, composition (an investment choice the
model can make) and condition (rehabilitation, which it cannot).

## P2-9 — Measured gas availability is 29.5%, against 0.80 assumed
**PROVISIONAL — unresolvable without decomposition.** NERC across three
periods: 29.5% (2024), 29.0% (2025/Q4), 26.2% (2026/Q1) against a stable
10,947 MW tracked fleet. The model credits 0.80 — 2.7x observed, worth
5,738 MW of phantom firm capacity in a constraint that binds 21 of 30
years. NERC's PAF conflates mechanical and fuel unavailability while the
model constrains fuel separately, so 0.295 cannot be substituted without
double-counting. Distribution is bimodal and the low end is not a fuel
story: Azura 93-98%, Ikeja 88-100%, against Alaoji at 0.01%, Sapele Steam
2.7%, Omotosho II 1.7%. Excluding every plant below 20% PAF still only
reaches 41.4%. Carried as bounds, not a point value.

## P2-10 — The demand shape was more peaked than reality, and the
## suppressed series is a supply shape
**PROVISIONAL.** Oluwole (2023) hourly 2016 data, both suppressed and
unsuppressed. The model assumed period factors 0.65 / 1.05 / 1.20
(night/day/peak), unsourced; measured unsuppressed is 0.906 / 1.024 /
1.088, and season factors 1.022 / 0.978 against 1.08 / 0.92 assumed.
**Rationing flattens the evening specifically** — peak factor falls
1.088 -> 0.983 while night RISES 0.906 -> 1.022; suppression ratio 0.4192,
only 41.9% of demand served. The suppressed series carries a HIGHER
instantaneous PAR (1.689 vs 1.459) but a LOWER block PAR (0.981 vs 1.085):
brief supply spikes on a suppressed base. That is the constraint-relief
asymmetry as measurement rather than argument, and it disqualifies the
served series as a demand anchor. Within-block peaking factor 1.3448
(PAR_inst / PAR_block) — a 4-hour block average undersizes firm capacity
by a third.

## P2-11 — NIRP contains no daily load profile
**STRUCTURAL (a documentary finding).** No hourly shape, no typical-day
curve, no peak timing, no statement of temporal resolution anywhere in
NIRP 2024. The only hourly dataset it names is hydro seasonality. Its
load factor of 0.76 is stipulated and internally inconsistent as printed
(Table 14 gives 71 and 76; the Table 15 pairs return 0.756 / 0.759 /
0.760, so 0.76 is the applied value). This is why `peak_nirp` is a
declared arm that raises: the block factors it would need cannot be
derived from the document.

## P2-12 — The retirement proxy was worth 1.28 bn and 18% of storage
**PROVISIONAL.** `gas_baseline_retirement_mw_per_year = 680` from 2035 was
a bare literal described in its own comment as a "brownfield retirement
proxy". It retires the entire fleet by 2051. Against a vintage-based
30-year schedule it overstates system cost by 1.28 bn (5.9%) and storage
by 14,664 MWh (17.9%); against no retirement at all, 3.60 bn and 43,788
MWh. Every vintage arm retires roughly half to a quarter of the proxy
rate. Replacement schedule is ASSUMED (analyst vintage estimate, register
L8), not sourced — Nigerian commissioning dates were not obtainable.

## P2-13 — Pre-F1 terminal arbitrage, measured
**SUPERSEDED by F-6 (eliminated under F1). Retained as the pre-F1
measurement.** Under `gas_unconstrained` + overnight/salvage: solar added
2,470 MW in 2053 and 14,406 MW in 2054; gas reached 13,089 MW running at a
9.5% capacity factor. A terminal vintage served one year and was refunded
most of its capex through the salvage credit, so it was nearly free.
Capacity factors: 59.7% (2045), 59.5% (2050), 18.4% (2054) under
`gas_none`; 42.0 / 34.5 / 9.5% under `gas_unconstrained`.

## P2-14 — Environment was not reproducible before 2026-08-23
**RESOLVED, retained for the record.** `environment.yml` omitted `pyyaml`
and both solvers while listing `pypsa`, which nothing imports. An AST scan
of `src/` and `scripts/` established the true dependency set as five
packages: numpy, pandas, pyomo, matplotlib, yaml. Windows Smart App
Control blocking unsigned BLAS binaries forced a WSL2 migration, during
which numpy was silently upgraded 1.26.4 -> 2.2.6. Repinned; the smoke
suite reproduces the pre-migration result exactly. Register L7.

---

## Retracted from this session

- **"+4.6% emissions from allowing new gas"** — a 2025-2050 cut with no
  external referent, invented as a working compromise around terminal
  contamination. Over the actual 2045 reporting window the two arms are
  identical at 358.8 Mt. See P2-4.
- **"+9.5% emissions (full horizon)"** — 2054, entirely inside the buffer.
- **"The solar min-build floor is worth ~85 bn"** — `solar_min_build_default()`
  returns 0.0 and was already neutralised in plan 2.5; the 4.7x difference
  came from five other arguments omitted from an uncontrolled diagnostic.
  The cleanup removed dead code and a latent TypeError but moved no numbers.
- **"Identical firm-capacity duals across arms indicate a defect"** — they
  are correct. The dual is the marginal price of firm capacity; storage was
  the marginal supplier in both arms, so the price was unchanged while the
  quantity differed.
- **"gas_retire_vintage_50 binds non-monotonically"** — arithmetic, not an
  anomaly. The 50-year schedule retires Sapele Steam in 2030, exactly
  576 MW = 720 x 0.8 of firm capacity, visible directly in the 2030 slack
  (1,189.90 vs 613.90 MW).
- **"F1 supersedes both salvage and replacement capex"** — this session's
  design claim, already RETRACTED in F-2's session. Truncation handles
  end-of-horizon residual value, not mid-horizon asset death.

---

## F-1 — Cost of capital now reaches the objective
**STRUCTURAL.** F0 audit: four blend scenarios returned 2.765439e10 to
seven significant figures; halving the private cost of capital changed
system cost by exactly zero. Under F1+F2 the objective spans 21.888 to
26.299 bn across `capital_recovery_case` arms. This is the thesis
contribution and it is delivered.
Arms: `cr_comm_low` / `cr_central` / `cr_comm_high`, `f1_tranche`,
`dscr_regression`, `env_dares`.

## F-2 — Concessional finance is inframarginal at every scale tested
**STRUCTURAL.** Cost falls linearly with envelope scale (24.307 / 22.724
/ 21.163 bn at 1x / 5x / 10x DARES) while gas, storage, solar, emissions
and the firm-capacity dual are identical to the last digit across all
three. Saving per concessional dollar: 0.528 (5x), 0.466 (10x) —
near-constant, mildly declining with vintage discount factors.
Mechanism: the envelope is fully drawn and exhausted at every build
decision, so the marginal dollar is commercial everywhere. An
inframarginal lump of cheap capital changes the total bill, not the
choice. Concessional finance changes WHO PAYS, never WHAT IS BUILT.

## F-3 — Cost of capital drives technology composition, via rate not envelope
**PROVISIONAL.** Gas capacity at 2045 against the regression arm:
-10.5% (`cr_comm_high`, 14%/7yr), -17.5% (`cr_central`, 11%/10yr),
-29.6% (`cr_comm_low`, 9%/15yr). Storage moves opposite: +8.7 / +14.5 /
+24.4%. Cheaper, longer commercial capital means LESS gas — gas is
capital-light and fuel-heavy, storage capital-heavy and fuel-free, so
cheapening capital favours the capital-intensive technology. Standard
result, reproduced endogenously.

## F-4 — Storage displaces gas at the storage duration
**PROVISIONAL.** Substitution ratio 4.0 MWh per MW of gas displaced,
stable at 2050 and 2054 — exactly the 4-hour storage duration. Total
firm capacity is set by the adequacy constraint and is invariant to
capital structure; composition is not.

## F-5 — Storage magnitude is NOT cost-optimal
**STRUCTURAL.** Once gas reaches its floor, storage absorbs the entire
firm-capacity residual, so its quantity is insensitive to further
financial variation. Root cause is the flat
`capacity_credit['storage'] = 1.0` (register L5 watch item): with no
declining ELCC, storage's marginal firm contribution never falls and no
interior optimum exists. The COMPOSITION claim (gas exits, storage
displaces it) is sound; the MAGNITUDE claim is not. Do not report
storage magnitude as an optimisation result.

## F-6 — Terminal arbitrage eliminated by F1
**PROVISIONAL.** 2054 solar collapsed 14,405 MW -> 2,560 MW against
2,459 in 2053, a smooth continuation with no discontinuity. Under
overnight+salvage a terminal vintage served one year and was refunded
most of its capex. Per-tranche annuity truncation removes this
structurally.

## F-7 — Emissions are exogenous under the baseline arms
**STRUCTURAL.** 17,980,983 tCO2 at 2045, identical to the unit across
all six arms, while gas capacity varies 5,407-7,685 MW (42% spread).
Emissions follow gas GENERATION, which is capped by the deliverability
constraint regardless of installed capacity. New plants supply firm
capacity only — there is no more fuel. This reproduces Nigeria's actual
pathology endogenously. CONSEQUENCE: no cost-emissions tradeoff exists
under baseline arms; any Pareto frontier must come from arms where the
emissions cap binds BELOW the deliverability limit.

## F-8 — Tenor mismatch is half the concessional advantage
**PROVISIONAL.** CRF(0.784% real, 30yr) = 0.03754 vs CRF(8.824% real,
10yr) = 0.15461, ratio 4.119x. Shapley decomposition: rate 47.0%, tenor
53.0% — order-invariant (rate-alone 2.553x, tenor-alone 2.780x from the
same base). Blended-WACC formulations cannot express tenor at all.
Nigerian evidence: no project-financed renewable has ever closed;
commercial bank tenors run 3-7 years absent credit enhancement;
continental average tenor has fallen 12 -> 5 years.

## F-9 — Firm-capacity shadow price is not monotonic in capital cost
**PROVISIONAL.** Dual at 2045: 28,840 (regression), 24,488
(`cr_comm_high`), 29,424 (`cr_comm_low`), 41,796 (`cr_central`).
`cr_comm_high` sits BELOW the regression arm. An earlier claim of a
monotonic 3.55x response was an artifact of quoting 2054, a buffer year,
and is RETRACTED.

---

## Retracted

- "Firm-capacity dual responds monotonically to capital structure, 3.55x"
  — quoted from 2054, outside the reporting horizon. See F-9.
- "F1 supersedes both salvage and replacement capex" (handover design
  claim) — truncation handles end-of-horizon residual value, NOT
  mid-horizon asset death. F1 and replacement capex are orthogonal.
  Corrected in F2.

---


## F-10 — Every NDC cap arm is non-compliant through leakage
**PROVISIONAL.** Under the sliced builder with F1 capital charges, all
four cap arms flag leakage above the 10% threshold: ndc2_unconditional
54.0% (479.0 Mt outside the cap), ndc2_conditional 57.7% (390.3 Mt),
ndc3_unconditional 57.6% (517.2 Mt), ndc3_conditional 50.4% (331.9 Mt).
The cap is grid-only by construction (`selfgen_in_emissions_cap=False`):
`emissions_by_year` measures grid + selfgen, `emissions_capped_by_year`
regulates grid only. Under a binding cap the model displaces load to
priced self-generation, whose emissions sit outside the boundary. The
NDC is met on paper while national emissions rise.
This is the constraint-relief asymmetry (register L2) appearing in the
policy domain. NOT a defect — the builder deliberately separates
measurement from regulation.
UNTESTED COUNTERFACTUAL: `selfgen_in_emissions_cap=True` gives an
economy-wide cap where leakage is structurally impossible. Not exposed
by any runner. That comparison is the two-point version of this finding.

## F-11 — Storage operates in an NDC pathway for the first time
**PROVISIONAL.** Post-port, ndc3_conditional builds 10,905 MWh and
discharges 2.8-6.2 TWh/yr from 2039. Under the annual builder storage
was identically zero in every NDC run (register L5).
ndc2_unconditional and ndc3_unconditional still build zero storage —
worth understanding separately.

## F-12 — Gas deliverability is valuable uncapped and worthless under a binding NDC cap
**PROVISIONAL, verified.** The same deliverability increase
(`baseline` -> `gas_deliv_gmp_partial`) is worth 3.16 bn in the
unconstrained sliced baseline (24.307 -> 21.149 bn, storage 75,481 ->
70,313 MWh) and EXACTLY ZERO under a binding NDC cap: ndc3_conditional
returned byte-identical results under both arms — objective
7.699302185e+10, leakage 50.4%, 331.9 Mt outside cap, storage
10904.97683 MWh, emissions 658.809554092434 Mt.
Mechanism: once the cap floors gas generation, the binding constraint
switches from fuel availability to the emissions cap, so additional
fuel has no value. Verified against an uncapped baseline where the flag
demonstrably moves the objective, ruling out a plumbing failure.
CONSEQUENCE: the GMP deliverability arms are informative for baseline
and gas-shadow-price work, and inert for every capped NDC pathway.
This also explains F-10 — with gas capped rather than fuel-limited, the
shortfall channel is self-generation regardless of gas availability.

## M-1 — Midstream cost per unit deliverability (post-hoc method)
**METHOD, not a result.** Arm 2 at I=0.5 requires +51.2 TWh_th/yr of
power-sector deliverability by 2030; at I=1.0, +102.5 TWh_th/yr.
Midstream cost bracket, whole-programme numerator over whole-programme
denominator (avoids allocating specific corridors, which would be a
composition fallacy): $22bn pipelines-only / 102.5 = $215m per
TWh_th/yr (LOWER, excludes CPF and upstream); $60bn full value chain /
102.5 = $585m per TWh_th/yr (UPPER, includes export, industrial, CNG).
Implied midstream investment: $11-30bn at I=0.5, $22-60bn at I=1.0.
Compare against the $750M DARES envelope: the midstream requirement is
15-40x larger, which sharpens F-2.
NOTE F-12: this comparison is only meaningful for uncapped pathways.
Under a binding NDC cap the deliverability the midstream would buy has
zero system value.

## F-13 — OCGT-to-CCGT conversion is the cheapest abatement option, by a wide margin
**PROVISIONAL.** Sliced baseline, f1_tranche + cr_central +
dscr_regression + env_dares + gas_deliverability baseline. Objective by
conversion arm: conv_none 24.307 / conv_low 23.034 / conv_central
21.154 / conv_high 20.007 bn. Conversion is worth 1.27-4.30 bn
(5.2-17.7% of system cost) against a concessional envelope worth 448M
(F-2) and a total capital-structure sweep spanning 4.4 bn (F-1).
Physical mechanism: converted output rises 0 -> 10.64 -> 21.76 TWh_e
while fleet output falls 24.74 -> 14.11 -> 2.98, total gas-side output
constant at 24.74 TWh_e (backcast -3.4% in every arm). The same gas
yields more electricity — the only route to more generation under a
binding deliverability constraint.

## F-14 — The availability screen decides the merit order
**PROVISIONAL.** Tier A (571 MW, PAF 0.959, $626/kW effective) converts
100% in every arm — inframarginal. Tier B (3,215 MW, PAF 0.345,
$1,740/kW) converts 29% at capex 850 USD/kW and 100% at 600 and 450,
so the flip lies between 850 and 600. Tier C (2,696 MW, PAF 0.106,
$5,677/kW) converts 0% in every arm.
Tier B converting fully despite an effective cost ABOVE new-build CCGT
($1,317/kW) is the mechanism the thesis is about: fuel-free firm
capacity beats fuel-hungry new capacity when fuel is the binding
constraint.
Without the per-plant availability screen the convertible base would
have read 6,482 MW nameplate rather than 1,941 MW available, and Tier C
would have been credited with capacity that does not run.

## F-15 — Conversion displaces storage as the firm-capacity supplier
**PROVISIONAL.** Storage 2054 falls 75,481 -> 72,198 -> 64,689 ->
64,148 MWh as conversion rises. Same substitution channel as F-4 (gas
displaced by storage at 4.0 MWh/MW); conversion reverses it by
supplying firm capacity without fuel.

## F-16 — NDC compliance costs 38-44% less when conversion is in the choice set
**PROVISIONAL.** `02_run_ndc_caps`, sliced, f1_tranche + cr_central +
dscr_regression + env_dares + gas_deliverability baseline,
conv_none vs conv_central:

| case | cost bn | leakage % | emissions Mt |
|---|---|---|---|
| ndc2_unconditional | 81.25 -> 45.78 (-43.7%) | 54.0 -> 39.6 | 886.6 -> 649.7 |
| ndc2_conditional   | 88.42 -> 55.05 (-37.7%) | 57.7 -> 44.1 | 676.2 -> 492.5 |
| ndc3_unconditional | 86.53 -> 50.69 (-41.4%) | 57.6 -> 45.0 | 897.9 -> 665.0 |
| ndc3_conditional   | 76.99 -> 43.31 (-43.7%) | 50.4 -> 34.2 | 658.8 -> 476.4 |

Cost, leakage and emissions all fall together. Mechanism, computed
before the run and confirmed by it: converted plant emits 0.3963
tCO2/MWh_e against the unconverted fleet's 0.6352 (EF_th 0.1823 / eta),
a 37.6% reduction per unit of electricity, or 1.60x the electricity per
tonne of CO2.

PAIRS WITH F-12. Extra gas deliverability is worth EXACTLY ZERO under a
binding NDC cap (byte-identical results under gas_deliv_flat and
gas_deliv_gmp_partial). Conversion under the same caps is worth 33-36
bn. Same fuel, same plants: the abatement lever is EFFICIENCY, not fuel
availability. Every pathway that omits conversion overstates both the
cost of NDC compliance and the leakage it induces.

## F-17 — A binding cap reorders the conversion merit order
**PROVISIONAL.** Tier C (2,696 MW, PAF 0.106, $5,677/kW effective)
converts 0% in every unconstrained arm (F-14) and 100% under every NDC
cap, generating 3.43-3.74 TWh_e in 2030 and 2.74 in 2045 — near its
availability ceiling, so it is running, not idle capacity bought for
firm credit. Under no cap the binding currency is electricity per
dollar and Tier C fails; under a cap it is electricity per tonne of CO2
and Tier C earns out even at 10.6% availability.
The cap changes the merit order, not only the quantity.

## F-18 — Conversion is invariant to capital structure
**PROVISIONAL.** conv_central across all four capital_recovery arms:
convert_mw is 571 / 3,215 / 0 (A/B/C, total 3,786 MW, capex $2.27 bn)
in every arm, and storage 2054 is 64,689 MWh in every arm, while the
objective spans 18.985 (cr_comm_low) to 22.912 bn (cr_comm_high).
Same shape as F-2: cost of capital moves price, not quantity.
Conversion is far enough inside the money that no tested financing
condition changes whether it happens. Conversion and capital structure
are independent levers — the retrofit pays at 14%/7yr as it does at
9%/15yr.

## F-19 — NIRP's gas trajectory is never a binding constraint on the cost-optimal path
**PROVISIONAL.** `gas_ndc_17gw` (NIRP-derived cap, nameplate basis:
11,932 MW 2030 / 22,113 MW 2035 / 27,201 MW 2041 / 23,906 MW 2045) and
`gas_unconstrained` return BYTE-IDENTICAL results under conv_central +
f1_tranche + cr_central + dscr_regression + env_dares: objective
21.154 bn, storage 2054 64,689 MWh, gas capacity 10,947 / 10,465 /
9,133 / 6,448 MW at 2030 / 2035 / 2041 / 2045, gas generation 11-15
TWh_e throughout.

The cap sits ~3.7x above the cost-optimal 2045 capacity and never
binds. The model does not build NIRP's gas even when permitted, and gas
capacity FALLS from the 10,947 MW starting fleet.

Mechanism: gas generation is limited by deliverability (89.27 TWh_th),
not by installed capacity. Capacity beyond what the fuel supports is
idle. This is P2-4 applied to the official plan — new thermal capacity
is bought for firm capacity only, because there is no more fuel.

The divergence between this thesis and NIRP on gas capacity is
therefore NOT a cost disagreement. Costing NIRP's path would require a
capacity FLOOR, not a ceiling. As a ceiling the trajectory is
non-binding and the comparison is qualitative: NIRP plans capacity the
fuel supply cannot run.

## F-20 — NDC 3.0's "17 GW" is a derated figure, not nameplate
**PROVISIONAL.** The NIRP-derived series is on a derated basis (~0.77).
Converting to nameplate gives 22,113 MW at 2035. Cross-check: NDC 3.0
describes its gas target as roughly double current levels; 22,113 /
10,947 = 2.02x, while the underated 17,027 / 10,947 = 1.55x does not
match. So NDC 3.0's stated 17 GW and NIRP's derived 17,027 MW are both
EFFECTIVE figures; the nameplate commitment is ~22 GW.
Consequence: any comparison of the NDC target against installed
capacity is comparing different quantities unless the derate is
stated. The 0.77 derate is DECLARED (P2-6), not independently sourced —
the one number in the chain not traceable to a NIRP table.

## F-21 — The solar-dominant pathway is contingent on hydro not expanding
**PROVISIONAL, and it bounds this thesis's own claim.** Sliced baseline,
conv_central + f1_tranche + cr_central + dscr_regression + env_dares,
across hydro_case:

| arm | hydro TWh_e | objective bn | solar 2045 MW | storage 2054 MWh |
|---|---|---|---|---|
| hydro_flat | 11.47 | 21.154 | 19,601 | 64,689 |
| hydro_committed | 12.8 | 20.409 | — | 64,196 |
| hydro_ndc | 45.0 | 8.925 | 3,260 | 60,528 |

Under NDC-scale hydro, solar falls 83% (19,601 -> 3,260 MW at 2045) and
system cost falls 58% (21.154 -> 8.925 bn). If hydro can expand to NDC
scale, solar is NOT the residual and the headline result changes
character. The defence must be made on deliverability, not on modelling
(F-22, F-23).
CALIBRATION NOTE: only hydro_flat reproduces the 2024 backcast (-3.4%).
hydro_committed and hydro_ndc give -8.6%, because raising exogenous 2025
hydro displaces gas in the balance. hydro_flat is the calibrated arm;
the others depart from observation by construction.

## F-22 — NIRP's 2045 hydro exceeds independent exploitable potential
**PROVISIONAL.** NIRP builds 20.6 GW hydro by 2045 against its own
stated ceiling of ~21 GW (1,978 MW existing + 700 MW committed Zungeru
+ 18,600 MW candidate) — 98.1% of the resource, with no margin.
Against the Energy Commission of Nigeria's independent estimate of
14,120 MW exploitable large hydro plus ~3,500 MW small hydro
(17,620 MW total), NIRP's 20.6 GW is 116.9% — it EXCEEDS total
exploitable potential.
NIRP's 21 GW is a MODELLING ceiling, not an engineering resource
estimate: the 18.6 GW candidate figure mixes site-specific and generic
plants, generic hydro is capped at 15 GW nationally, and zonal
allocation uses percentage shares from 24 potential sites in JICA
(2019). A plan consuming 98% of its own ceiling, and 117% of an
independent estimate, is an upper bound presented as a forecast.

## F-23 — Hydro trajectories fail on lead time, not resource
**PROVISIONAL.** Large hydro runs 8-12 years FID-to-commissioning;
Zungeru's 700 MW took ~7. Required build rates:
  hydro_ndc  : +7,411 MW over 5 yr = 1,482 MW/yr = 2.1 Zungerus/yr
  NIRP 2045  : +17,922 MW over 20 yr =  896 MW/yr = 1.3 Zungerus/yr
`hydro_ndc` passes the resource test (59% of ECN exploitable) and fails
lead time. NIRP passes lead time more easily and fails the resource
test (F-22). Different failure modes; state both.
NOTE: the model has NO lead-time constraint for any technology, so this
objection applies equally to NIRP's solar and gas trajectories.

## F-24 — F-8's project-finance claim is too broad and is narrowed
**CORRECTION to F-8.** The claim "no project-financed renewable energy
project in Nigeria" (Lexology, 2020) is refuted twice over:
- Mabon / Dadin Kowa, 39 MW on-grid hydro IPP, ~$70m at 60/40
  debt/equity, BOT concession, debt from Africa Finance Corporation and
  PowerChina, grid-connected Dec 2019, commissioned Jan 2020. NERC
  lists Mabon as an operational 39 MW on-grid hydro generator. So "no
  utility-scale grid-connected renewable IPP" is ALSO too broad.
- Afrigreen Debt Impact Fund lends on an explicit project-finance basis
  (no sponsor guarantee, SPV-level, project assets as collateral) into
  Nigerian C&I solar: $15m to WATT SPVs, $15m to Westa's Nigerian C&I
  portfolio, $10m secured to Empower/Justrite solar + BESS.
DARES is NOT a counterexample — a World Bank IDA credit implemented
through REA using results-based mechanisms is a different modality from
commercial project finance.
DEFENSIBLE VERSION, sourced from NIRP itself: the 14 utility-scale
grid-connected solar IPPs that signed PPAs with NBET in 2016 (~1.12 GW)
had still not reached financial close when NIRP 2024 was prepared.
NIRP's own candidate table lists them as the ONLY committed solar
because lack of financial closure excluded all others, and identifies
the failure to provide government guarantees as the principal cause.

## F-25 — Demand base reconciliation with NIRP 2024
**PROVISIONAL. Closes the tracked item at merged_correction_plan_v2.md:1240.**

**Demand base: reconciliation with NIRP 2024.**

This model anchors 2024 demand at 37.09 TWh — gross generation at the
busbar, from NERC 2024 (37,093.70 GWh). NIRP 2024 models 30.5 TWh of
grid demand for the same year. The gap is 6.6 TWh, or 18% of the NIRP
figure. It is definitional, not substantive, and four terms account
for it.

*Basis year.* NIRP anchors on 2022 observed on-grid demand of 32.3 TWh
and projects forward under its adopted scenario; this model anchors on
2024 observed generation. The two are not measuring the same year.

*Metering point.* The 37.09 TWh is gross generation at the busbar, and
this model's boundary is declared there so that distribution losses
fall outside it. The full ladder is documented: 37.09 gross → 7% TLF →
34.50 delivered to DisCos → 36.4% ATC&C → 21.97 collected. NIRP does
not state which point on this ladder its 30.5 TWh occupies. The figure
falls between the at-DisCo (34.50) and collected (21.97) quantities, so
the two numbers may not disagree at all — they may be different
quantities on the same ladder. Absent a stated metering point the
comparison cannot be closed arithmetically, which is itself the finding.

*Exports.* NERC generation includes export volumes to Niger, Benin and
Togo, which sit inside the generation total but outside NIRP's domestic
grid demand.

*DESG delay.* NIRP's adopted Distributed Energy Service Gap scenario
shifts universal access later than its own Annex B assumes, suppressing
early-year modelled demand.

*NIRP's figure does not satisfy its own consistency check.* Its 2024
modelled grid demand (30.5 TWh) is below its 2022 observed on-grid
figure (32.3 TWh) — a forecast starting below its base year, implying
grid demand contracted by 5.6% over two years. The mechanism is the
DESG delay assumption, so the 30.5 TWh is the output of a policy
scenario rather than an estimate of demand. NIRP's early-year figures
are therefore a weak calibration benchmark, and the divergence from
37.09 TWh should not be read as this model overstating demand.

*Note on the same document's internal spread.* NIRP's 2045 on-grid
demand appears as 222 TWh at p.49 and 297–301 TWh in Tables 6 and 15 —
a ~75 TWh gap, 34% of the lower figure, unexplained within the
document. A demand series that disagrees with itself by a third at the
horizon cannot anchor a critique of a base-year figure whose derivation
is fully stated.

## F-26 — Total solar is identified; the public/EaaS split is identified only where the budget binds

> **[TASK 30 Part D: the second half of this title is WRONG and is superseded by frozen F-12 (and by this entry's own interval table below).** The split is not identified where the budget binds either: tight budget, interval 10,841-10,969 MW (22.8% of the maximum); the budget only raises the floor. Every EaaS MW and public-solar MW figure in the table and the 17,702 MW sentence is a normal-solve vertex inside an interval and must be read as such. Cells: `unconstrained` has no budget (non-binding by construction); `moderate` (7.738 bn) and `tight` (4.552 bn) have a binding budget by F-12's floors (14,945 / 38,518 MW > 0) and are still not points. Do not delete; do not quote as results.]
**PROVISIONAL.** F3 lexicographic test, `26_run_f3_identification.py`,
ndc3_conditional at the central configuration (conv_central,
hydro_flat, dscr_central, cr_central, env_dares, baseline
deliverability, f1_tranche).

Normal-solve split across public budget arms:

| arm | budget | EaaS MW | public solar MW | total |
|---|---|---|---|---|
| unconstrained | none | 14,691 | 32,830 | 47,521 |
| moderate | 7.738 bn | 21,818 | 25,703 | 47,521 |
| tight | 4.552 bn | 39,520 | 8,001 | 47,521 |

Z* = 34,419,113,068 USD in ALL THREE, identical to the dollar.

**Total solar is invariant at ~47,521 MW** across every budget level.
That quantity is identified and reportable.

**The budget binds and reallocates at zero cost.** Halving the public
ceiling from 7.738 to 4.552 bn moves 17,702 MW from public to private
finance with no change in system cost. **[UNIDENTIFIED, TASK 30 Part D: 17,702 MW
is the difference of two vertices (21,818 and 39,520 MW), each inside an interval
of 33,847 (moderate) and 10,841 MW (tight) at this epoch (F-28, F-26 below). The cost
invariance is identified; the size of the shift is not. What the budget does force is
the FLOOR: it rises 13,681 → 36,687 MW (+23,006 MW) between the same two budgets
(F-28).]** This is the EaaS decoupling
mechanism demonstrated directly: as public capital is withdrawn,
private EaaS finance substitutes one-for-one and the system builds the
same solar at the same cost.
[UNIDENTIFIED, TASK 41: the EaaS/public split is not identified in
any capital case (F-12, as corrected 2026-10-08). The identified
content is that total solar and system cost are unchanged over this
range.]

**The split is not identified beyond what a constraint forces.**
Lexicographic min/max on total EaaS MW at fixed Z*, capital_case=tight:

| tol | min MW | max MW | interval | % of max | max per-year span |
|---|---|---|---|---|---|
| 1e-6 | 36,686.8 | 47,528.0 | 10,841.2 | 22.8% | 2,745.4 |
| 1e-5 | 36,396.3 | 47,594.4 | 11,198.1 | 23.5% | 2,811.8 |
| 1e-4 | 35,130.2 | 48,189.2 | 13,059.0 | 27.1% | 3,406.6 |

Structural, not numerical: one part in 1e6 cannot produce a 22.8%
range unless the LP has a flat face there, and the interval grows only
modestly with tolerance.

**Cause, and it is not a defect.** Under F1 public and EaaS solar are
priced identically (register L16) and EaaS capex never enters the
public budget constraint — only the EaaS subsidy does, and the subsidy
is forced to zero when the tariff covers capex. So the optimiser is
indifferent to the split except where the ceiling or bankability forces
it. The interval IS that indifference, measured.

**Reporting rule.** Any EaaS deployment figure from this model must
carry its budget arm and be reported as one point in the stated
interval, never as a point estimate.

SCOPED CONDITIONS: tariff 95,000,000 USD/TWh is a bare literal with no
arm, directly upstream of the bankability constraint; EaaS and public
solar are physically identical (L16); B* carries an unreconciled
stale-doc warning (6.13 vs 9.104 bn); gas_avail_nameplate 0.80 against
measured 29.5% (P2-9); the NDC cap narrows the interval, so
identification is established where the cap binds, not in general.

## F-27 — The EaaS channel is worth 8.9 bn under NDC 3.0 conditional
**PROVISIONAL.** Same NDC case, same physical arms (conv_central,
hydro_flat, cr_central, env_dares, baseline deliverability,
f1_tranche, capital_case moderate):

  public-only  (02_run_ndc_caps, no tariff)      43.3136 bn
  EaaS enabled (26_run_f3, tariff 95e6)          34.4191 bn
  difference                                      8.8945 bn  (-20.5%)

DSCR contributes ZERO to this: 02_run_ndc_caps returns 43.313608 bn at
BOTH dscr_regression and dscr_central, identical to the last digit.
CORRECTED REASON: an earlier statement here said the bankability
constraints are skipped when no tariff is set. That is wrong.
`load_scenario()` writes `solar_service_tariff_usd_per_twh` unconditionally
(65,000,000 by default), so `tariff is not None` always holds and both
constraints ALWAYS build. They are inert in public-only runs for a
different reason: `financing_regime != "eaas"` triggers
`m.eaas_disable` (optimize_model_sliced.py:1080-1083), fixing
`solar_eaas_add[t] <= 0`, so the bankability constraint is trivially
satisfied at zero EaaS and `required_margin` has nothing to bite on.
The entire gap is the financing regime.

Mechanism, and it pairs with F-26: public-only, all solar is funded
through a binding public ceiling. With EaaS available, private finance
substitutes one-for-one at the same unit cost, so the ceiling stops
constraining deployment.
[UNIDENTIFIED, TASK 41: the EaaS/public split is not identified in
any capital case (F-12, as corrected 2026-10-08). The identified
content is that total solar and system cost are unchanged over this
range.]

**CAVEAT — this result is conditional on an unsourced literal.** The
EaaS channel is gated by `m.eaas_bankability`, parameterised entirely
by `solar_service_tariff_usd_per_twh = 95,000,000`, a bare literal with
no arm. At that value the constraint has slack. At a lower tariff EaaS
deployment would be bankability-limited and this 8.9 bn would shrink.
The tariff must become a swept arm before this figure is reported.

## F-28 — The EaaS identification interval is bounded by the budget below and bankability above
**PROVISIONAL. Refines F-26.** F3 lexicographic min/max on total EaaS
MW at fixed Z*, ndc3_conditional, central config, tol 1e-6:

| capital_case | budget | min MW | max MW | interval | % of max |
|---|---|---|---|---|---|
| tight | 4.552 bn | 36,686.8 | 47,528.0 | 10,841.2 | 22.8% |
| moderate | 7.738 bn | 13,680.9 | 47,528.0 | 33,847.1 | 71.2% |

**The maximum is identical to the digit in both arms** (47,528.0). It is
the bankability ceiling, set by tariff and capex, and is independent of
the public budget. Only the FLOOR moves: a tighter public ceiling forces
more solar through the private channel, raising the minimum EaaS
consistent with optimal cost.

**[SUPERSEDED, TASK 30 Part D: the ceiling is total solar, not bankability (frozen F-12, which also shows the maximum is identical at three tariffs and rises with the cost tolerance). The interval endpoints below are identified bounds at the earlier epoch; the "bankability ceiling" reading is withdrawn.]**
So the split is not simply unidentified. It is bounded BELOW by the
public budget and ABOVE by bankability, and the interval is the gap
between two constraints binding at opposite ends. A sufficiently tight
budget would close it.

This is the same mechanism F-27 shows from the cost side: EaaS matters
exactly to the extent public capital is scarce.

REPORTING RULE (supersedes F-26's): any EaaS figure must carry BOTH its
capital_case and its tariff_case, and be reported as a point in the
stated interval.

## F-29 — F-27 is robust to the tariff assumption
**PROVISIONAL.** `tariff_case` swept across the NBET-sourced range
(75 / 95 / 115 M USD/TWh) plus both legacy arms, ndc3_conditional,
capital_case=moderate: Z* = 34,419,113,042 to 34,419,113,050 — an 8 USD
spread on 34 bn, solver noise. The 8.895 bn value of the EaaS channel
(F-27) does not depend on the tariff.

The tariff does change how often bankability binds — binding years
(slack < 1 USD/MW) go 8 at tariff_low, 3 at central and legacy_95m, 1 at
tariff_high — and it moves the normal-solve split. **[UNIDENTIFIED, TASK 30 Part D:
the 8 / 3 / 3 / 1 binding-year counts are, like the split, properties of the
chosen vertex (capital_case moderate, 7.738 bn, budget binding by F-12's floor);
F-29's own next sentence says so for the split and it applies equally to these.]** But since the split
is degenerate over a wide interval (F-28), the normal-solve point is one
draw from that set and its ordering across tariff arms is not
interpretable.

REGRESSION: `tariff_legacy_95m` and `tariff_central` are bit-identical
(both 95e6), and `tariff_legacy_95m` at capital_case=tight reproduces
the pre-refactor F3 interval [36,686.8, 47,528.0] to one decimal. The
arm reaches the LP without disturbing it.
