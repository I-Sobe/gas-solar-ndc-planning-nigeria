# Scenario Coherence: Demand Arms × Gas Regimes

**Correction plan Phase 1.6, Step 8.**
Status: declaration complete; quantitative population deferred to Phase 2.6.

---

## 1. Why this document exists

The model's demand arms and its gas deliverability regimes are sourced on
**opposite assumptions about whether Nigeria's sector constraints are relieved**.

- **Demand arms** are derived from unconstrained drivers — population, GDP,
  income elasticity of electricity demand, per-capita convergence against
  comparator countries. They describe what the system would be required to serve
  in the absence of rationing.
- **Gas deliverability** is anchored to *observed constrained throughput*
  (89.27 TWh_th, ≈805 MMscf/d), and Phase 6.1 establishes that the binding
  constraint is **commercial and midstream** — pipeline capacity, vandalism,
  gas-to-power arrears, allocation — not reservoir depletion. Nigeria holds 200+
  TCF of proved reserves.

Combining the two produces a world in which constraints lift on the demand side
and persist on the supply side. That combination is **what generates the entire
solar residual**, and therefore the storage build, the annual-versus-sliced cost
premium, and the financing frontier that is this thesis's contribution.

The arithmetic makes the dependence explicit. Gas is deliverability-bound from
2026 and hydro is exogenous, so solar is an accounting residual:

```
solar_TWh(2045) ≈ D(2045) − gas_TWh_e(2045) − hydro_TWh(2045)
                ≈ D(2045) − 28.4 − 11.5
                ≈ D(2045) − 39.9
```

Because the subtracted constant is large relative to demand, the amplification
factor `D/(D − 39.9)` is approximately **2×**: a one-percentage-point error in
the demand arm moves the headline solar build by roughly twenty per cent. The
same amplification runs the other way through the gas term. Neither assumption
can be treated as incidental.

This is not an error to be corrected. It is a **scenario design choice**, and the
purpose of this document is to ensure it is *declared and argued* rather than
inherited by accident of which parameter happened to be audited first.

---

## 2. The two dimensions

### Demand arms (`scenarios.demand_growth_scenarios()`)

| Arm | Rate | Basis |
|---|---|---|
| `constrained_continuation` | 3.14% | Ember 2010–2025 grid generation CAGR. A **constraint-persistence** arm, not organic demand growth |
| `organic_central` | 4.0% | Route A (ε = 0.83, GDP 4.5%) = 4.04%; Route B (converge to Kenya's 2024 per-capita level by 2045) = 3.67% |
| `organic_high` | 6.0% | Route A (ε = 0.90, GDP 6.8%) = 6.30%; Route B (Ghana's 2015 level) = 6.40%; NIRP 2024 low case = 5.9% |

Held outside the arm registry: `demand_growth_benchmark()["nirp_2024_base"]` =
7.7%, NIRP's own total-demand CAGR on Tier 2 scope. Benchmark comparison only.

### Gas regimes (`scenarios.gas_deliverability_scenarios()`)

| Regime | Narrative |
|---|---|
| `downside` | Commercial constraint tightens — arrears deepen, midstream investment stalls |
| `baseline` | Constraint persists at roughly the observed level |
| `upside` | Constraint eases — payment chain and midstream delivery improve |
| `shock_recovery` | Disruption followed by recovery |

---

## 3. The coherence cross-tab

The organising question for each cell: **does this combination describe a world
that could actually occur?** Demand growth and gas deliverability are not
independent. Both are driven by the same underlying dynamic — the payment chain,
midstream investment, and sector governance. A world in which industrial and
commercial electricity demand compounds at 6% per year while gas deliverability
to power collapses is not a scenario; it is two scenarios stapled together.

| | gas `downside` | gas `baseline` | gas `upside` | gas `shock_recovery` |
|---|---|---|---|---|
| **`constrained_continuation`** 3.14% | **COHERENT** — reform fails on both sides | plausible | incoherent | plausible |
| **`organic_central`** 4.0% | asymmetric — upper stress bound | **ASYMMETRIC — CENTRAL CASE** | plausible | plausible |
| **`organic_high`** 6.0% | incoherent | plausible | **COHERENT** — reform succeeds throughout | plausible |

**`shock_recovery` is a shape, not a level narrative.** It describes a disruption
followed by recovery and is orthogonal to the question of whether the sector
reform succeeds. It therefore pairs coherently with any demand arm and sits
outside the diagonal logic above.

**Definitions used in the table:**

- **Coherent** — demand and gas assumptions rest on the same view of whether the
  sector constraints are relieved. Reportable as findings.
- **Asymmetric** — demand assumes relief, gas assumes persistence. Defensible on
  the argument in §4, but must be named at first use.
- **Plausible** — one dimension moves without a strong implication for the other;
  partial or lagged relief. Reportable with a note.
- **Incoherent** — the two assumptions contradict each other. Sensitivity bounds
  only, never central findings.

---

## 4. The central case is off-diagonal. Why that is defensible.

`organic_central` × gas `baseline` is the asymmetric cell, and it must be
acknowledged that **this is also the region of the table that maximises the
financing gap.** Along the coherent diagonal the extremes partially cancel: low
demand with low gas gives a small solar residual, and high demand with high gas
lets gas cover more of it. Higher demand against a persistent gas constraint
maximises the residual, the capital requirement, and therefore the headline
quantity this thesis measures.

"It is the middle of the range" is not a defence. The defence is structural, and
rests on three arguments.

### 4.1 Demand is a requirement; gas is a capability

In a capacity-expansion linear programme, exogenous demand is **the service
obligation the system is required to meet** — it is not a forecast of what will
be delivered. Setting the obligation equal to a supply-suppressed continuation of
historical delivery asks the model: *what is the cheapest way to keep Nigeria's
grid roughly the size it already is?* The answer is: build very little, meet the
NDC caps comfortably, and never bind the capital envelope. The financing frontier
would then be flat by construction, and the thesis would have proved its own
assumption.

Gas deliverability is a different kind of object. It is not an obligation but an
observed physical and commercial capability, and the empirical record is the
appropriate anchor for it.

Pairing an unconstrained requirement with an observed capability is therefore not
an inconsistency. It is the correct treatment of two quantities that differ in
kind.

The per-capita diagnostic (Phase 1.6, Step 1) demonstrates why the alternative
fails empirically as well as logically. Over 2010–2025, Nigerian grid generation
grew 3.14%/yr while population grew 2.39%/yr — **76% of observed growth is
demographic**. Measured per *connected* person, generation **fell 1.66%/yr**
(327 → 258 kWh, 2010–2024) and is negative on every window tested. The historical
series contains essentially no information about consumption intensity; it
records delivery under rationing. Calibrating a demand requirement to it would
assume the persistence of the very constraints this study evaluates.

**Independent corroboration:** NIRP 2024 (Table 14) derives its national demand
forecast bottom-up from population, GDP, electrification rate and network-loss
trajectories — explicitly rather than from historical generation. Nigeria's own
planning authority reached the same methodological conclusion.

### 4.2 The asymmetry is the research question, not an artefact of it

Nigeria's planning failure has been precisely that demand-side ambition was never
matched by upstream delivery. The asymmetric cell is not an accident of
parameterisation; it is a formal statement of the condition the thesis exists to
analyse.

The contrast with NIRP 2024 makes this concrete. NIRP projects gas generation
rising 22.3 TWh (2024) → 49.9 (2030) → 88.0 (2035), permitting 63 GW of candidate
gas capacity off identified pipeline projects — **it assumes the gas supply
constraint is relieved.** This model assumes it binds throughout, at roughly
25.6 TWh_e.

That is a scenario disagreement of about 3.5×, and it is load-bearing for every
result here. It requires two responses, both of which are outstanding:

1. **An evidenced defence**, grounded in the delivery record of the pipeline
   programme — actual commissioning against plan, gas-to-power payment arrears,
   and NGC actuals versus stated capability. This is exactly the evidence NIRP
   does not engage with, and assembling it converts an assumption into a finding.
2. **A NIRP-consistent gas arm**, added purely as a benchmark. If the model
   reproduces NIRP-like build under NIRP-like gas, that is a strong validation
   section. If it does not, the reason must be understood before the viva rather
   than during it.

### 4.3 The asymmetry is bounded, not concealed

Both coherent diagonal cells are run and reported as bounds. The reader sees the
full range and can locate the central case within it. The declaration in this
document, referenced at first use in the results chapter, is what converts an
implicit choice into an explicit one.

---

## 5. A third instance of the same asymmetry: exogenous GDP

The model treats GDP growth as exogenous to electricity supply, but the two are
jointly determined — power shortage is itself a constraint on output growth. A
scenario pairing 6.8% GDP growth with 24% unserved energy is internally
incoherent in the same way as the cells marked incoherent above.

This is **declared as a limitation rather than modelled**, for three reasons.
Endogenising GDP would introduce a production function and an
output-elasticity-of-electricity parameter — a second unsourced parameter in the
same contaminated-sample position as the income elasticity, and one which
compounds rather than averages under feedback. It would dilute the novelty claim,
which is endogenous *capital structure*, not endogenous macroeconomics. And
demand depending on the solution makes the problem non-linear or requires
iteration to a fixed point, breaking the LP formulation on which the entire
verification apparatus rests.

Coherence is enforced instead through the cross-tab: **high-GDP arms are
admissible only alongside supply outcomes that actually deliver.**

**Direction of the omitted feedback:** excluding incoherent high-demand /
high-unserved cells *removes* demand rather than adding it, so the omission is
**conservative** with respect to the financing gap. NIRP 2024 carries the
identical omission and does not flag it.

---

## 6. Reporting rule

> **Coherent cells** are reported as findings.
>
> **The asymmetric cell** is reported as the central case, with the asymmetry
> named at first use in the results chapter and cross-referenced to this
> document.
>
> **Plausible cells** are reported with a note on the partial-relief reading.
>
> **Incoherent cells** are reported as sensitivity bounds only, and never as
> central findings.

---

## 7. What cannot be completed yet, and why

**The cross-tab above is a declaration. It cannot yet be populated with
quantitative results.**

`13_run_dem2_growth_gas_interaction.py` produces the full 3 × 4 grid, but every
`public_only` cell in it is currently **VoLL-dominated**. Under
`constrained_continuation` the public capital constraint binds exactly
(`capital_utilisation` = 1.000000) and further demand goes straight to unserved
energy priced at the value of lost load. The result is a cost cliff rather than a
cost gradient:

| Cell | at g = 0.025 | at g = 0.0314 | unserved |
|---|---|---|---|
| `downside / public_only / no_policy` | $13.6bn | **$152.7bn** | 0 → 35.3 TWh |
| `baseline / public_only / ndc2` | $15.6bn | **$179.0bn** | 0.9 → 34.5 TWh |
| `baseline / public_only / ndc3` | $75.2bn | **$259.1bn** | 17.3 → 45.9 TWh |

An eleven-fold cost increase from a 0.64 percentage-point change in one
parameter. The model is reporting faithfully — a budget-constrained public arm
falls off a cliff rather than degrading gracefully — but the resulting numbers
measure a **penalty parameter**, not the cost of capital.

This is systemic rather than incidental: *any run combining a binding NDC cap
with a binding capital envelope produces a VoLL-dominated objective.* It affects
the Monte Carlo, the DEM-2 `public_only` cells, GAS-3 and POL-1, all of which are
on the do-not-report list.

Evidence that the effect is specific to the budget constraint, from the Phase
1.6 Step 5b regression test: when `constrained_continuation` moved from 0.025 to
0.0314, the **EaaS** arm's gas premium moved by 0.00001% (491,648,215.6 →
491,648,164.3 — solver noise on an effectively identical problem), while the
**`public_only`** equivalent moved by 31% (63.1bn → 83.0bn). The public arm is
hypersensitive because it is budget-bound; EaaS is not.

**Consequence.** The quantitative cross-tab is a post-Phase-2.6 deliverable. The
genset backstop bounds the penalty at diesel LCOE (≈$0.30–0.55/kWh) instead of at
VoLL, converting the cliff into a slope and making the cells economically
interpretable. Populating this table before 2.6 would fill it with figures
already declared unreportable.

---

## 8. Status and next actions

| Item | Status |
|---|---|
| Coherence classification declared | **Done** — §3 |
| Argument for the off-diagonal central case | **Done** — §4 |
| Reporting rule | **Done** — §6 |
| Quantitative population of the cross-tab | **Blocked on Phase 2.6** |
| Evidenced defence of the gas ceiling vs NIRP | **Open** — pipeline delivery record, arrears, NGC actuals |
| NIRP-consistent gas benchmark arm | **Open** |
| Cross-reference from the results chapter | **Open** — cite this document at first use of the central case |

**Related plan items:** Phase 1.6 (demand arms, complete); Phase 2.6 (genset
backstop, gates §7); Phase 5.1 (Monte Carlo correlation structure — take it from
§3 rather than assuming a separate one); Phase 6.3 (Discussion and Limitations —
§4 and §5 belong there in condensed form).
