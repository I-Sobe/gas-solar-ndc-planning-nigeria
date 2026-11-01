# Methodology — self-generation backstop, emissions-cap boundary, grid service obligation

**Current as of HEAD `8fc5ccb`.** Last substantive model changes:
`5beaef0` (storage-CAPEX realised-spend diagnostic) and `85f2598` (B\*
replacement, concessional rate sourcing, consolidated gas benchmarks,
IRENA solar arm). The operating-and-maintenance implementation
(`d7538ff`), which added +6.602 bn (+30.9%) to the central objective,
remains the last change to the O&M magnitude itself. Figures elsewhere
in this document not explicitly marked otherwise are stated at this
state.

**This is a living document describing a model under active correction. Results
produced under these specifications are provisional and are treated as
verification instruments until the full correction sequence closes. The sections
below cover ONLY the private self-generation backstop, the emissions-cap
boundary, and the grid service obligation.**

Sections covered here: self-generation backstop specification; genset cost and
emission-factor derivation; fuel price and currency basis; emissions-cap
boundary and leakage; grid service obligation; obligation trajectories.

**Not covered here** — written elsewhere: demand construction and the tiered
demand architecture; gas deliverability and gas fleet efficiency; solar and
storage cost trajectories; capital recovery and blended finance; firm-capacity
adequacy; temporal slicing; salvage and asset lifetimes; the NIRP critical
review beyond the self-generation figure specifically.

---

## 1. Private self-generation as a priced backstop

**SETTLED**

### 1.1 What the model does

A non-negative variable `selfgen[t]` (annual model) and `selfgen[t,s]` (sliced
model) enters the energy balance alongside grid supply and carries a cost per
TWh in the objective. It represents electricity that Nigerian households and
firms generate privately when the grid does not serve them.

The variable is deliberately **unbounded**. The "unbounded" design choice
itself is not in question, but its supporting magnitude was: **"roughly
eight times peak realised grid capacity" is not supported by
`docs/sourcing_pack_02_ledger.md`**, whose own figures give ~9 GW of
backup generation (IEA) and 13-20 GW (IFC / Energy for Growth / a Lagos
State-backed study) against a grid peak of 5,802 MW and an average
available capacity of 4,286 MW — a range of **about 1.6x to 4.7x**, not
8x. No capacity limit binds within the planning horizon under any figure
in that range; a capacity bound would still be a fictitious constraint.

### 1.2 Why the specification changed

Before this correction, unmet grid demand was represented solely by an
`unserved[t]` variable priced at the Value of Lost Load. That specification was
wrong for this system in a way that mattered.

VoLL prices the *welfare loss* of load being shed. In Nigeria, load is very
largely not shed — it is self-generated. **About 86% of firms** (not
specifically MSMEs — `docs/sourcing_pack_02_ledger.md` states "about 86%
of Nigerian companies own or share a generator," covering about 48% of
their electricity demand) **own or share a generator**, and the
installed private fleet is on the order of 1.6x to 4.7x grid capacity
(§1.1, corrected). Pricing the
alternative to grid supply at VoLL therefore sets the counterfactual to grid
investment at a near-infinite penalty, against which almost any grid investment
appears worthwhile. That is not a conservative assumption; it is an optimistic
one, and it inflated every financing comparison the thesis depends on.

The correction replaces an assumed penalty parameter with a **priced physical
alternative**. The claim the thesis can now make is that grid investment must
justify itself against what Nigerians actually buy, rather than against nothing.

A consequence stated plainly because it cuts against the thesis's own argument:
adding the backstop makes decarbonisation look **harder**, not easier. The model
will sometimes choose diesel over solar, because in Nigeria today it does.

### 1.3 Units at the balance — a declared simplification

The energy balance is metered at busbar. A generator delivers at the point of
use, downstream of the transmission loss factor, so one TWh of genset output
displaces more than one TWh of busbar generation. The model applies a
busbar-equivalence factor to `selfgen` in the balance.

**The central case sets this factor to 1.0.** This is a declared conservative
simplification, not an oversight. At 1.0 the backstop is less effective per TWh,
so the model leans harder on grid supply and reports a higher system cost.

**Direction of bias: it OVERSTATES the grid build the system requires** —
conservative with respect to the thesis's own feasibility claim rather than
flattering to it. A sensitivity arm at 1/(1−TLF) = 1.0753 is retained; applying
it reduces self-generation by approximately 9.5%.

This factor stops being freely swept once the tiered demand architecture lands,
because the split between grid-served and behind-the-meter energy becomes a
property of the tier construction rather than a free parameter.

### 1.4 Load-shedding is no longer an available action

`unserved` is retained for logical completeness but is **fixed at zero whenever
a genset case is priced**.

The reason is a defect found during correction and worth recording as method.
With self-generation emissions counted against the cap, `unserved` became the
cheapest abatement technology available: shedding load emits nothing, so the
model paid VoLL to shed rather than the far lower genset cost to self-generate,
purely to buy cap headroom. That is physically false. Unmet grid demand in
Nigeria does not vanish; it becomes self-generation. Shedding is a consequence,
not a planner's choice.

With `unserved` fixed at zero there is exactly one shortfall channel, VoLL
leaves the objective entirely, and a target that cannot be met returns
**infeasible** rather than a large penalty. Infeasibility is the honest answer
and, where it occurs, the stronger finding.

**Consequence for reporting.** Under any genset case, `unserved` is zero *by
construction*. It must never be quoted as evidence that demand is met. Three
quantities are reported separately and mean different things:

| quantity | meaning |
|---|---|
| `selfgen + unserved` | grid service shortfall — what the obligation bounds |
| `unserved` alone | truly unmet demand given the backstop assumption; ~0 by construction |
| selfgen cost and CO₂ | the burden the shortfall imposes on private actors |

### 1.5 Scope of the cost

Self-generation is a real resource cost and enters `system_cost_npv`. It is
**private** expenditure and is excluded from the public capital budget
constraint. This is verified structurally rather than by inspection: a
post-build probe confirms the derivative of the public budget constraint with
respect to `selfgen` is exactly zero.

### 1.6 Declared limitation — the backstop is available to everyone

The model allows any consumer to self-generate at the arm's price. In reality
generator ownership is highly unequal: while **about 86% of firms** own or
share one (not specifically MSMEs — `docs/sourcing_pack_02_ledger.md`),
household ownership is far lower and the poorest have neither grid connection
nor generator.

**Direction of bias: OVERSTATES how well the shortfall is absorbed.** A model in
which anyone can buy diesel at will understates the welfare cost of grid
failure. The affordability, accessibility and acceptability dimensions are
therefore **reported alongside the objective, not internalised in it** — a
least-cost linear program minimising system cost cannot notice that a household
cannot afford the alternative it is being assigned.

---

## 2. Genset cost and emission factor

**SETTLED** for the specification and the diesel efficiency. **PROVISIONAL** for
capital cost, capacity factor and asset life within the central arm.

### 2.1 Derivation structure, and why it is structured this way

Efficiency is the single sourced physical parameter. Everything else derives
from it:

```
SFC  = 1 / (HHV_per_litre × η_genset)          litres per kWh
EF_e = EF_th(HHV basis) / η_genset             tCO₂ per MWh_e
```

Specific fuel consumption is **never sourced independently**. Sourcing both SFC
and efficiency would allow cost and emissions to decouple the moment efficiency
is swept — the same defect an earlier correction removed from the gas fleet,
where a stored electrical emission factor drifted from the efficiency it was
supposed to derive from. Storing only `EF_th` and `η` makes the two move
together by construction.

### 2.2 Efficiency — SOURCED

Derived from four Caterpillar prime-rated generator-set technical data sheets
(C4.4 at 80 kW, C15 at 400 kW, C18 at 480 kW, 3512 at 1000 kW), using published
fuel consumption in litres per hour at 25 / 50 / 75 / 100% of rated load:

```
η_HHV = (rated_kW × load_fraction) / (L_per_hr × HHV_kWh_th_per_litre)
```

Two operating points anchor the arms, both the mean across the three large sets
(the 80 kW unit is excluded as a small set and has no published 25% point):

- **0.3474** at 75% load — well-loaded C&I diesel, the short-run marginal cost
  floor.
- **0.2809** at 25% load — part-loaded C&I diesel, matching the metered 20–25%
  load factor observed in the Nigerian fleet. This is the central arm.

A third arm represents the small petrol generator population, with efficiency
**derived** from an independently reported specific fuel consumption of 2.0
litres per kWh, implying η_HHV = 0.0511.

### 2.3 Emission factors — DERIVED, and validated

The chain is: fuel price per litre → mass via density → energy via net calorific
value → CO₂ via emission factor → converted to a higher-heating-value basis to
match the convention used for natural gas elsewhere in the model.

- Net calorific values: IPCC 2006 Guidelines Volume 2 Chapter 1 Table 1.2.
  **SOURCED.**
- CO₂ emission factors: IPCC 2006 Volume 2 Chapter 2, Tables 2.2–2.5.
  **SOURCED.** The CO₂ default is identical across all four source categories;
  only CH₄ and N₂O differ. Since only CO₂ is carried — for comparability with
  the gas-fleet cap, which is CO₂ — the source-category choice is immaterial.
  Stated explicitly because a reader may otherwise expect it to matter.
- NCV-to-GCV ratio: **0.95** for liquid fuels, IPCC Volume 2 Chapter 1 §1.4.1.2
  (approximately 5% for coal and oil). **SOURCED.** This differs from the ratio
  used for natural gas (9–10%); reusing the gas ratio would have introduced a
  systematic 5% error.
- Fuel densities: NMDPRA product specifications, midpoint of the stated range.
  **SOURCED.**

**Validation.** The chain reproduces 2.318 kgCO₂ per litre for petrol and 2.676
for diesel against widely published values of approximately 2.31 and 2.68. This
is an independent check on the derivation, not a calibration.

### 2.4 Cost — mixed provenance

The short-run marginal cost arm is fuel only, on the reasoning that the existing
generator stock is sunk. It is fully sourced: efficiency from datasheets, fuel
price from the series in §3.

The central arm adds capital recovery and operating cost. **These components are
ASSUMED** — analyst estimates for capital cost per kW, economic life, capacity
factor and fixed operating cost of a mid-size C&I diesel set. They constitute
roughly 23% of that arm's total and should be replaced with Nigerian dealer
quotations.

The small-petrol arm's capital cost is **SOURCED** from Nigerian retail prices
collected in August 2026 (2 kVA at ₦300,000, 3 kVA at ₦500,000, 5 kVA at
₦800,000), converted at the contemporaneous rate, giving $140–155/kW. This is an
order of magnitude below the ~$600/kW that circulates in the modelling
literature; locally observed pricing is preferred for a Nigerian study.

### 2.5 A parameter that is not what it appears to be

The 2.0 litres per kWh figure used for the petrol arm requires a caution that
belongs in the method rather than a footnote.

Its source reports it across five customer segments — households, MSMEs,
schools, health centres and hotels — and every segment divides out to **exactly
2.000**. The first two are stated to run mostly petrol; the last three mostly
diesel. Petrol and diesel have different densities and calorific values, so both
cannot yield exactly 2.000 unless the figure was assumed and applied rather than
measured. The source confirms this: a standard load factor was applied across
all clusters, and approximately 2 litres per kWh assumed.

The metered electricity consumption in that study is genuine — 500 smart meters
over twelve months. The litres, the monthly fuel expenditures, and the national
spending figure reconstructed from them all inherit a single assumption.

**Provenance is therefore ASSUMED, not SOURCED**, and the arm is used as a
stress bound rather than a scenario. The implied efficiency of approximately 5%
is plausible for grossly oversized, part-loaded household sets and not credible
for a hotel's diesel running three hours a day.

### 2.6 Why three arms, and what each represents

Self-generation in Nigeria is bimodal, and the two modes behave differently
enough that a single blended parameter would misrepresent both.

An aggregate reconciliation makes the point: the reported national fuel spend on
small gasoline generators, divided by the sourced fuel price, implies a fuel
volume that at 2.0 litres per kWh delivers only about a sixth of the
self-generated electricity Nigeria's stipulated self-generation share implies.
Small petrol sets consume a large share of the spending while delivering a small
share of the energy; large commercial and captive diesel delivers most of the
energy at roughly a seventh of the unit cost.

For a grid capacity-expansion model the **marginal** backstop for bulk unserved
demand is the second mode. The central arm is therefore the part-loaded C&I
diesel arm. The small-petrol arm is retained as an upper stress bound
representing the household access tier.

A design check that constrained this choice: at its price the small-petrol arm
sits at essentially the same level as the VoLL-dominated cost figures the
correction set out to remove. Pricing the backstop there would rename the
penalty cliff rather than replace it with a slope, defeating the purpose of the
specification.

---

## 3. Fuel prices and the currency basis

**SETTLED**

### 3.1 Construction

Monthly national average pump prices for automotive gas oil and premium motor
spirit, published by the National Bureau of Statistics, converted to US dollars
at the **mean of daily closing rates within the same calendar month** from the
central bank's foreign exchange window.

Month-by-month conversion is a method choice, not a convenience. Converting a
period mean at a spot rate, or a current naira price at a current rate to
represent a historical average, attributes exchange-rate movement to the
commodity. Given that the naira has moved several-fold since deregulation, this
would dominate any real price signal.

### 3.2 The anchor window, and what is excluded

Central values are the means over **December 2024 to February 2026**:
AGO $1.0157/l, PMS $0.7311/l. **SOURCED.**

Excluded from the anchor: March to May 2026, during which AGO rose to $2.392/l
on a crude price shock, with the naira appreciating throughout — so the movement
was a commodity event, not a currency one, and it had substantially unwound
within months. Including it would price a transient shock as the level: a
trailing-twelve-month mean gives a coefficient of variation of 35.4% against
8.3% for the clean window. **The excluded months are NOT retained as any
arm — this claim is ORPHANED.** `data/cost/processed/genset_cost.csv`
carries exactly three rows: `genset_srmc`, `genset_central`,
`genset_high` (the PMS petrol high arm); no arm is built from the
$2.392/l AGO shock months (`grep` of `io.py`, `genset_cost.csv`,
`scenarios.py` for that figure or the excluded window: no match). The
excluded months are described here but not wired into any scenario at
HEAD.

### 3.3 Why the model is real US dollars — evidence, not assumption

The concern with denominating a Nigerian model in dollars is that it imposes a
foreign unit on a naira economy. The converted series shows the opposite: over
the anchor window the naira moved 12.7% and both naira pump prices swung
approximately ±20%, while both dollar prices held within ±9% (coefficients of
variation 8.3% and 8.7%).

This is corroborated by market structure. In July 2026 Nigeria's dominant
domestic refiner suspended naira-denominated sales and moved to dollar pricing,
citing insufficient crude supply through the naira-for-crude arrangement; its
diesel gate price sits within 7% of the independently derived clean-window mean.

**This converts the real-dollar convention from a modelling assumption into an
observed property of the market**, and it belongs in the methodology on those
terms.

**Open:** the real base year has not been reconciled against the base year of
the technology cost data. The adjustment is one to two years of US inflation and
is small, but it must be stated rather than skipped.

---

## 4. Emissions-cap boundary and carbon leakage

**SETTLED** for the boundary. **PROVISIONAL** for the leakage magnitude, which
depends on the tiered demand architecture still being settled.

### 4.1 Three emissions quantities, deliberately distinct

An earlier specification used a single switch to decide whether self-generation
CO₂ counted. That conflated two different things — what the national commitment
covers, and what is physically emitted — which must attach to different
instruments. The model now carries three:

| quantity | contents | attaches to |
|---|---|---|
| `emissions_grid_only` | gas-fired grid generation | the **cap** |
| `emissions_selfgen` | private diesel | reported, and the leakage measure |
| `emissions` | both. Physical truth | the **carbon price**, and all reporting |

### 4.2 Why the cap covers the grid only

The emissions cap is derived as the gas-power share of Nigeria's energy-sector
abatement target. That is what the NDC commits for this sector. Private diesel
self-generation sits inside the national inventory but outside that slice.

Capping total emissions against a budget sized for the gas-power slice would be
an accounting error, and it would charge diesel at a shadow price that is an
artefact of how narrowly the slice was drawn rather than of any policy.

### 4.3 What disciplines diesel instead, and why it is not a carbon price

Drawing the cap at the grid boundary creates an obvious risk: the model could
satisfy the NDC by pushing load onto diesel, meeting the target on paper while
national emissions rise. Self-generation emission factors exceed the grid's, so
the substitution is always emissions-negative in reality.

The instrument that prevents this is the **grid service obligation** (§5), which
constrains diesel by quantity rather than price.

A carbon price was considered and rejected as the central case. Nigeria has no
carbon price; the model's no-policy setting is zero. Introducing one to suppress
a modelling artefact would assume the conclusion. The carbon price is retained
as a **sensitivity**, where it answers a legitimate question — what price would
be required to price diesel out of the plan — and where, because it attaches to
total emissions, it prices diesel correctly when active.

### 4.4 The leakage diagnostic

Because the obligation is the only defence and does not bind in every
configuration, a post-solve check (`leakage_diagnostic()`,
`optimize_model.py:1759`, threshold 0.10) exists and is applied by
`02_run_ndc_caps`, `03_run_ndc_eaas`, `05_run_fin3_capital_sweep`, `15_`,
`18_` and `99c_`. **It is not applied to every run with a real cap**:
three scripts pass a real `emissions_cap_by_year` and never call it —
`24_run_leakage_arm.py`, `25_run_obligation_trajectory.py`,
`26_run_f3_identification.py` (register L51). Where it is applied, it
reports the share of physical emissions falling outside the cap boundary
and flags the run when that share exceeds a declared threshold.

The threshold is a **declared judgement, not a derived quantity**, and must be
stated as such: if more than a stated fraction of physical emissions escapes the
commitment boundary, the plan's compliance is an artefact of the boundary and
the run is not reported as an NDC-compliant plan.

### 4.5 A correction worth recording — the buffer window was not inert

The model solves to 2054 while reporting to 2045, with the intervening years an
end-effect buffer. Emissions caps in those years were originally released to an
effectively infinite value, on the reasoning that a binding cap outside the
reporting window would distort investment inside it.

That reasoning was correct until self-generation was priced. With a backstop
available, an uncapped buffer became an arbitrage sink: the optimiser routed
essentially all self-generation into the uncapped years and paid the shedding
penalty in the capped ones. The observed self-generation matched the permissible
buffer-window shortfall closely enough to identify the mechanism unambiguously.

Buffer-year caps are now **held flat at the terminal in-window value for
every scenario except `ndc3_conditional_decline`**, whose buffer years
2046-2054 instead CONTINUE the linear decline
(`scripts/00_build_emissions_cap.py:315-323`), deliberately departing
from the flat-hold convention (register C-12; `findings_frozen.md` F-21
scope conditions). The flat-hold treatment is the same one already
applied to capital cost trajectories in the buffer, and for the same
reason: buffer years must be neither cheaper nor freer than the last
reported year, or they pull behaviour out of the reporting window.
Buffer-year caps are never reported.

---

## 5. Grid service obligation

**SETTLED** for the specification. **OPEN** for the sweep axis (§7).

### 5.1 What it is, and what it is not

A constraint bounding `selfgen + unserved` as a fraction of demand.

It was previously named a reliability constraint and bounded `unserved` alone.
Both were wrong once a backstop existed.

**Bounding `unserved` alone stopped working**, because with a cheaper backstop
`unserved` is zero by construction and a constraint written on it can never
bind. It would have remained in the model, silently inert, while several
sensitivity runners depended on it.

**Calling it reliability overstated the term.** Reliability in power-system
planning is adequacy — LOLE, EENS, expected unserved energy. A household running
its own generator was not served by the grid, but no capacity was short. The
constraint is a **service obligation**: a policy commitment that at least a
stated fraction of demand is met by the grid. That is a real instrument;
electrification targets have exactly this shape.

The renaming matters beyond terminology. As a service obligation, this
constraint is the model's only direct lever on **accessibility** — it constrains
diesel by quantity, independent of price, and therefore does work the cost
objective cannot.

### 5.2 Year-by-year, not horizon-total

The obligation is enforced **per year**. A horizon-total form is available but
not used, on the basis of a measured result rather than preference.

Under a monotonically rising deficit, a horizon-total bound is satisfied by
early-year slack and stops discriminating between service standards. Tested
across a 2.5-fold tightening, the total form returned an identical plan at every
level — four different policies, one outcome — while the annual form bound
immediately.

A service obligation is a commitment to customers in each year. A horizon-total
form would permit 95% service through the 2030s and 30% in the 2050s while
reporting compliance.

**A related reporting trap, recorded because it caused an error during
correction.** The horizon-aggregate shortfall fraction is not a test of whether
the constraint binds. In one configuration the aggregate sat at 2.5% of demand
while six individual years were pinned exactly at the 20% bound. Binding must be
assessed per year.

### 5.3 Trajectories

The constraint accepts either a scalar (a flat obligation) or a per-year
sequence (an obligation **trajectory**). The trajectory form exists to test
whether the *timing* of a service commitment matters under a fixed capital
envelope with declining technology costs — a question a flat bound cannot pose.

Comparison across trajectories is made at **constant horizon-average** so the
only difference is timing. An earlier comparison across different obligation
*levels* was mistaken for a timing result; that comparison sat at the
feasibility boundary and its apparent non-monotonicity was a boundary effect.
The distinction between varying a level and varying a schedule is a
methodological point, and the earlier claim has been retracted.

---

## 6. Bias directions for the simplifications in this area

Stated together, because a limitation whose direction is named is defensible and
one whose direction is unknown is not.

| simplification | direction of bias |
|---|---|
| Busbar factor 1.0 (no TLF credit to selfgen) | **Overstates** required grid build; conservative against the thesis's own claim |
| Backstop unbounded and available to all | **Overstates** how well shortfall is absorbed; understates welfare cost of grid failure |
| `unserved` fixed at zero under a genset case | Removes the shedding option entirely; makes unmeetable targets **infeasible rather than expensive**. Direction is toward stricter, not more permissive |
| Cap on grid only | **Understates** the emissions the plan is accountable for; measured explicitly by the leakage diagnostic rather than left implicit |
| CO₂ only, no CH₄ or N₂O | **Understates** total warming impact. Chosen for comparability with the gas-fleet cap, which is CO₂ |
| Central arm capital cost assumed | Unknown direction. Roughly 23% of that arm; flagged provisional |
| Petrol arm SFC assumed, not measured | **Overstates** the household tier's cost if the assumption is high. Used as a bound, not a scenario |

---

## 7. Open method decisions

**The sweep axis for the obligation.** The fraction bounded by the obligation is
a **policy instrument, not a physical parameter** — there is no measurement of a
correct service standard. The axis is therefore justified methodologically: it
must bracket the model's own feasibility boundary, so the frontier is visible
rather than falling entirely inside or outside the swept range. An empirical
anchor exists in Nigeria's observed off-grid share, though on a different
denominator and therefore indicative only. Several runners currently sweep
different ranges; a single documented rationale is outstanding.

**Capital cost, life and capacity factor for the central diesel arm.** Currently
analyst estimates. Nigerian dealer quotations, converted on the same
month-matched basis as fuel, would move these to SOURCED.

**The real base year** for the fuel anchor against the technology cost data.

**Whether the leakage threshold is defensible at its current value.** It is a
declared judgement. Its consequences are visible — it determines which runs are
excluded from NDC claims — so the value should be argued rather than assumed.

**The busbar factor becomes constrained rather than free** once the tiered
demand architecture lands, because the split between grid-served and
behind-the-meter energy becomes a property of the tier construction. Sweeping it
independently at that point would double-count.

**Whether self-generation should be represented as two populations rather than
three arms.** The current specification sweeps one variable across three price
levels. The underlying reality is two distinct populations with different sizes,
duty cycles and unit costs. A two-segment specification would be more faithful
but adds a variable and a sizing assumption for each segment. Not yet decided.

---

## 8. A documentary finding that constrains this area

**PROVISIONAL — under active reconciliation**

The national plan against which this thesis benchmarks stipulates that
self-generation constitutes a stated share of on-grid demand in the base year,
declining linearly to zero by 2035. Document-level search establishes that this
share is **stipulated, not derived**: the plan gives no methodology, no fuel
volume, no efficiency or specific fuel consumption assumption, and no external
citation. Its stated basis is observations in recent years, with no observer,
dataset or year range named; the sectoral allocation is explicitly made in the
absence of actual data. The 2035 elimination date is recorded in the same
document as stakeholder-workshop consensus rather than analysis.

Two methodological consequences.

First, the self-generation share **cannot be recorded as SOURCED**. It is
stipulated. The magnitude of self-generation in the base year is uncertain by a
wide margin across independent construction routes, and must be treated as a
swept parameter with a declared source for each arm rather than as a fixed
input. This is being reconciled separately and is a precondition for the tiered
demand work.

Second — and this is a finding rather than a limitation — the national plan's
demand forecast **assumes away the quantity that the emissions cap fails to
account for**. Self-generation is treated as an outcome the plan eliminates, not
as a modelled input, and no feasibility model, retirement curve or
consumer-switching model is presented for the phase-out. The model built here
produces substantial self-generation throughout the horizon under a constrained
capital envelope. The comparison holds independent of which base-year magnitude
is chosen, because it is the elimination assumption doing the work.
