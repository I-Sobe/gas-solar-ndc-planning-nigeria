# Methodology

**Current as of HEAD `8fc5ccb`.** Last substantive model changes:
`5beaef0` (storage-CAPEX realised-spend diagnostic) and `85f2598` (B\*
replacement, concessional rate sourcing, consolidated gas benchmarks,
IRENA solar arm). The operating-and-maintenance implementation
(`d7538ff`), which added +6.602 bn (+30.9%) to the central objective,
remains the last change to the O&M magnitude itself. Figures elsewhere
in this document not explicitly marked otherwise are stated at this
state.

**STATUS: LIVING DOCUMENT, MODEL UNDER ACTIVE CORRECTION.**
Every section carries a status marker. `SETTLED` means the specification
is fixed and results produced under it are attributable. `PROVISIONAL`
means the specification is implemented but its central values or its
consequences are still being tested. `OPEN` means a decision is
outstanding.

This document records METHOD, not results. Results live in
`docs/findings_frozen.md` (`docs/findings_provisional.md` does not exist;
`findings_provisional_pre_om.md` and `findings_provisional_pre_sweep.md`
are earlier, superseded states, kept for provenance); limitations in
`docs/limitations_register.md`; the correction sequence in
`merged_correction_plan_v2.md`.

Scope note: this draft covers the model architecture, the financing
formulation, the gas and hydro representations, and the verification
discipline. It does not yet cover demand construction, emissions cap
derivation, or the reliability formulation in comparable depth; the
reliability/self-generation boundary is covered in
`docs/methodology_genself_backstop.md`.

**Read before citing anything about reliability, feasibility, or unmet
demand from this model: `docs/methodology_genself_backstop.md` §1.4.**
Under any priced genset case, `unserved` is fixed at zero by
construction, not priced at VoLL. `cum_unserved = 0` must never be
quoted, here or anywhere else this model's objective or feasibility is
discussed, as evidence that demand is met — the shortfall channel is
self-generation, which is priced and unbounded, not unserved energy.

---

## 1. Research question and what the model must be able to answer

**Prime question.** What is the cost-optimal decarbonisation pathway for
Nigeria's gas-constrained power sector under NDC 3.0, and how do capital
structure and financing mechanism reshape the feasibility frontier?

The second clause is the binding requirement on the model's design. A
capacity-expansion model that treats cost of capital as an exogenous
discount rate cannot answer it: the rate changes what the analyst reports
but not what the model builds. This was verified empirically before the
correction (see §4.1), and it is the gap the formulation is built to
close.

---

## 2. Model class and structure

**Status: SETTLED.**

A deterministic linear-programming capacity-expansion model, implemented
in Python/Pyomo, solved with CBC and HiGHS.

**Two builders, deliberately retained.**

- `build_model` (annual) — one time step per year. Retained because it is
  the regression instrument for every pre-correction result, and because
  several diagnostics were built against it.
- `build_model_sliced` (time-sliced) — six slices per year, formed as two
  seasons (dry, wet) x three periods (night, day, peak).

The sliced builder is the reporting model, **except GAS-1/GAS-2**
(`06_run_gas1_shadow_benchmarks.py`, `07_run_gas2_eaas_gas_relief.py`),
which build with `build_model` (annual) under a stated scope condition:
conversion is not representable and storage is not dispatched
(`findings_frozen.md` F-30; register L48). The distinction is not
cosmetic: at annual resolution storage is structurally dominated. With
`discharge[t] <= RTE x charge[t]` and a balance taking
`+discharge - charge`, storage contributes at most `(RTE - 1) x charge`,
which is non-positive, so optimal storage is identically zero for any
cost. This is structural, not a calibration artefact, and no parameter
sweep can change it. The slice structure gives storage a capacity-value
channel through the firm-capacity constraint.

**Horizon and reporting window.** The model runs 2025-2054. Results are
reported to **2045**, with 2046-2054 as an end-effect buffer. The buffer
is not decorative: terminal effects were measured reaching back to 2050
(storage divergence between two arms was 14% at 2045, 42% at 2050, 70%
at 2054), which is why the reporting window was tightened from an
earlier 2050 convention. 2045 is also the policy horizon for both
NDC 3.0 and NIRP 2024.

**Metering point.** Gross generation at the busbar. Declared, so that
distribution losses fall outside the model boundary rather than being
carried as an undeclared wedge. The full loss ladder is documented
(37.09 TWh gross -> 7% TLF -> 34.50 to DisCos -> 36.4% ATC&C -> 21.97
collected) so that any figure quoted from this model can be located on
it.

**Currency basis.** Real USD. This was not originally declared; it was
established by inventory and then ruled on. The determining evidence at
the time: gas price is flat for decades (a no-real-escalation
assumption), and capex derives from NREL ATB, which publishes in real
base-year dollars. **`us_inflation` is now a required `load_scenario`
parameter** (`scenarios.py:2106`, `_US_INFLATION_REQUIRED`; value 0.020),
used to deflate nominal rates once, by a stated Fisher relation, at each
CRF call site — the "no inflation parameter exists" premise no longer
describes the codebase, though the real-USD basis and the deflate-once
convention it established are unaffected.

---

## 3. Parameter discipline

**Status: SETTLED.**

Three rules govern every parameter, and they are the reason the model's
outputs are attributable.

**3.1 Provenance taxonomy.** Every parameter is classified:
- **SOURCED** — stated by a named document.
- **DERIVED** — computed from a document's own figures, by a method
  reproducible from those figures, but not stated by it.
- **ASSUMED** — analyst estimate.

A parameter's name may not assert provenance it lacks. This rule was
added after finding a function named `concessional_envelope_npv()` that
applied no discounting, and a variable named `gas_retire_proxy` that was
an analyst estimate.

**3.2 Required keys, no defaults.** Every load-bearing assumption is a
required scenario key with a sentinel default that raises if omitted.
A silent default producing clean-looking wrong output is the recurring
failure mode this rule exists to prevent. **The current set is the 17
sentinel-defaulted `load_scenario()` parameters** (`scenarios.py:2046-2150`,
each a unique `= object()` marker): `tariff_case`, `peak_multiple_case`,
`reserve_margin`, `storage_cc_case`, `gas_build_case`,
`gas_availability_case`, `gas_retirement_case`, `gas_capex_case`,
`ccgt_efficiency_case`, `capital_recovery_case`, `dscr_case`,
`us_inflation`, `capital_treatment_case`, `concessional_envelope_case`,
`conversion_case`, `hydro_case`, `om_case` — plus `genset_case` in
`load_econ` (18 total across both loaders). `new_gas_deliverability` and
`gas_deliverability_case` are **not** sentinel-defaulted (they carry
ordinary defaults inside `load_scenario`); they are required only by
some runners' own CLI parsing, a weaker, per-runner discipline rather
than a `load_scenario`-level one.

A third instance was found after this rule was adopted: the EaaS service
tariff was written unconditionally by `load_scenario()` from a defaulted
parameter, so an unsourced value was present in every scenario dict the
project ever built, never appeared in any output tag, and was silently
overridden by a bare literal at eight runner sites. The rule catches this
class of defect only where it is applied; parameters predating it must be
audited rather than assumed compliant.

**3.3 Arm-tagging.** **Output filenames carry a short deterministic
hash8 of the arm set, with the full tag stored in each JSON/CSV
header and in `results/<dir>/INDEX.md`** (e.g.
`results/baseline_sliced/diagnostics__02c17f55.json`;
`01b_run_baseline_sliced.py`'s `hash8_of`/`append_index`) — this
replaced long tags in filenames themselves, which some directories
(`02_`, `03_`) still carry. No result is readable without knowing its
assumptions, so no result can be silently compared against one produced
under different ones; the lookup now goes through the header or
`INDEX.md` rather than the filename alone.

---

## 4. Financing formulation — the contribution

### 4.1 The gap

**Status: SETTLED (the diagnosis).**

Before correction, cost of capital did not reach the objective. Under
`baseline_no_policy`, four blend scenarios spanning private costs of
capital from 18% to 9% returned **2.765439e10 to seven significant
figures** — halving the private cost of capital changed system cost by
exactly zero. Capital cost entered only a bankability constraint on one
channel; it never priced an asset.

### 4.2 Per-tranche capital recovery factor

**Status: PROVISIONAL — implemented, central values sourced, results
being re-run at the central configuration.**

Undifferentiated overnight capex in the objective is replaced by an
annualised capital charge at the owner-specific cost of capital,
discounted at the social rate:

    CRF(r, L) = r(1+r)^L / ((1+r)^L - 1)

    capital_charge[v] = CRF(r_conc, L_conc) x conc_vol[v]
                      + CRF(r_comm, L_comm) x comm_vol[v]

    s.t.  conc_vol[v] + comm_vol[v] = capex[v] x add[v]
          sum_v conc_vol[v] <= concessional_envelope_usd

`conc_vol` and `comm_vol` are decision variables; both CRFs are
constants. **The formulation is linear.** This is the load-bearing design
choice: blending the two rates first would make the CRF nonlinear in the
blend share, which is precisely why the share had to be fixed
exogenously in the original specification. Decomposing into two volume
variables with two constant recovery factors keeps the problem linear
while making capital structure endogenous. The realised cost of capital
becomes an output.

**Tenor is a second axis, not a detail.** The two tranches differ in
rate AND in amortisation period. A Shapley decomposition of the total
concessional advantage attributes 49.9% to rate and 50.1% to tenor
in real terms at the current rates (46.3 / 53.7 at the frozen state; the
nominal-basis 47.0 / 53.0 pair was withdrawn, F-4, TASK 45). The decomposition is order-invariant; a
one-at-a-time attribution gives 68/32 or 31/69 depending on which
variable moves first, and is not reportable. Blended-WACC formulations
cannot express tenor at all.

**Truncation at the horizon replaces salvage.** A vintage whose recovery
period ends within the horizon stops being charged; the residual value
is handled by the annuity simply ending. This supersedes the previous
end-of-horizon salvage credit.

**Truncation does NOT replace replacement capex.** This was an error in
the original design and it was corrected. Truncation handles
end-of-horizon RESIDUAL VALUE; it does not handle MID-HORIZON ASSET
DEATH. For an asset whose technical life is shorter than the horizon —
storage at 15 years against a 30-year horizon — the two are different
phenomena. Replacement capex is therefore retained under the per-tranche
formulation, priced commercial-only.

**Recovery period is bounded by technical service life:**
`L_eff(channel, tranche) = min(L_tranche, technical_life[channel])`.
Capital cannot be amortised over longer than the asset survives. This is
the correct use of `asset_lifetimes()` under the new formulation — as a
ceiling on the recovery period, never as the recovery period itself.

**Sourced values.** Concessional: World Bank IDA Terms, single-currency
USD blend. The central case (`cr_central`) uses **IDA21 terms
throughout, rate and maturity** — 2.12% effective 2026-01-01 (band
2.12-2.13%, `docs/sourcing_pack_01_rates.md` §1), 25-year maturity —
adopted over the credit's own IDA20 origination vintage (FY24Q3, 2.80%
all-in: 1.39% service + 1.41% interest, 30-year maturity) because the
model evaluates forward pathways rather than repricing an existing
commitment; both the IDA20 rate and its maturity are retained in the
register (L53) as the superseded values. (An interim state briefly
carried IDA21's rate on IDA20's 30-year maturity — a rate/tenor hybrid
that priced concessional capital cheaper than either real vintage, CRF
0.033943 against 0.037538 (IDA20) and 0.040616 (IDA21's own 25-year
term); corrected the same day, register L53.) `cr_ida21` remains a
distinct arm, near-identical to `cr_central` (2.13% vs 2.12%, same
25-year maturity) — the two differ only by which of IDA21's two 2026
effective dates each was set from, not a substantively different
instrument. This unifies the objective's own concessional rate AND
tenor with the EaaS bankability test's concessional rate (§4.4), which
already used 2.12% (register L53). Commercial: SOURCED base plus a
DECLARED uplift — the IEA Cost
of Capital Observatory reports 8.5-9% WACC for USD-denominated clean
power in Kenya and Senegal; Nigeria is not covered, and the 11% central
applies a 2-2.5pp uplift for worse offtaker credit. **The uplift is an
analyst estimate and is bracketed 9-14%.** Commercial tenor 10 years,
SOURCED: a C&I solar debt fund with Nigerian exposure lends 8-10 years;
Nigerian commercial bank loans run 3-7 years absent credit enhancement.

**Coverage cushion.** A single DSCR-style cushion, constant across
financing regimes, sourced from project-finance covenant norms for
African solar IPPs (1.30x central, 1.20-1.40 range). The previous
practice of varying the cushion by financing regime (1.05 concessional /
1.10 commercial) was retired as a double-count: the concessional/
commercial distinction is carried by the tranche rates, and applying it
again as a coverage cushion counted the same advantage twice, in the
direction that flattered the thesis's own claim.

**Operating and maintenance, added at the same objective.** Operating
costs in this model's objective fall into two categories that are charged
on different bases. Fixed operating and maintenance cost (FOM) is a
recurring charge proportional to installed CAPACITY, independent of how
much a plant runs — the model charges it, per technology, per year, on
whatever capacity is actually installed and operating that year
(accounting for retirement where the technology retires). Variable
operating and maintenance cost (VOM) is a recurring charge proportional
to OUTPUT — it is charged only where a technology's own cost data shows a
non-zero rate. In this model, solar and storage carry FOM only (their
sourced VOM rates are zero in every NREL ATB cost case used); gas carries
both FOM, on installed capacity, and VOM, on generation, in addition to
its pre-existing fuel cost, which is a distinct line from either. Every
one of these terms is discounted by the same social discount factor the
rest of the objective already applies — there is no separate "operating
cost" discount convention.

The rates themselves come from NREL's Annual Technology Baseline (ATB)
2024, `2024_v3_Workbook.xlsx`, `Summary_FOM` sheet for fixed cost and the
matching technology sheet for variable cost, at three cost cases —
Advanced, Moderate, and Conservative — which this model exposes as
`om_low`, `om_central`, and `om_high` respectively, each carrying a full
2025-2050 year-by-year trajectory per technology (`solar_utility`,
`storage`, `gas_ct` for the unconverted gas fleet, `gas_cc` for
OCGT-to-CCGT converted capacity). `om_central` is the model's central O&M
assumption, replacing a placeholder constant (`STORAGE_OM_PER_MWH_YR`,
2,000 USD/MWh-yr) that had been sitting in the cost-loading code for the
life of the project without ever being read by either builder — an
omission this project's own required-key discipline exists to catch, and
did (`docs/methodology_protocols.md` §1).

One conversion deserves its own explanation because it introduces a
dependency the model did not have before. ATB reports storage's fixed
cost per kW of POWER capacity, because that is what drives a battery
system's inverter and balance-of-plant cost; this model tracks storage in
MWh of ENERGY capacity, because that is the quantity its dispatch
constraints are written against. Converting one to the other requires
dividing by an assumed number of hours of discharge duration — this model
uses 4.0 hours, the same figure that already sets the storage
capacity-credit ceiling and the storage power-to-energy ratio elsewhere in
the LP. The effect is that storage's O&M charge is coupled to an
assumption (`storage_duration_hours`) that register entry L21 declares
ASSUMED, not SOURCED, and unswept: before this correction,
`storage_duration_hours` only affected a capacity credit; it now also
scales a cost.

A fourth arm, `om_none`, sets every O&M rate to exactly zero. It exists
because this correction changes the objective on purpose, and a change to
the objective needs its own regression test distinct from "did the
numbers move" — `om_none` lets a reviewer confirm that the new code path,
exercised with every rate forced to zero, reproduces the pre-correction
objective to the digit, which is proof that the wiring itself introduced
no side effect beyond the intended cost term. `om_central` moving the
reported total cost is therefore not evidence of a bug; `om_none` failing
to reproduce the old anchor would be.

At the central configuration, adding these terms moves the objective by
**6.602 bn (30.9%)** — the single largest correction applied to this
model since the tranche formulation itself.

### 4.3 Concessional envelope

**Status: PROVISIONAL.**

An undiscounted commitment-stock ceiling on concessional principal,
spanning solar and storage only. Gas is commercial-only: the anchoring
credit is a distributed-renewables facility, and concessional credit for
a combined-cycle gas turbine is not a defensible claim.

The envelope is a stock, not a flow, and does not revolve — a single
sovereign credit is a hard principal ceiling. Modelling it as
non-revolving understates available concessional capital over a 30-year
horizon and therefore biases the realised blended rate upward, which is
the conservative direction.


### 4.4 The EaaS channel and bankability

**Status: PROVISIONAL.**

EaaS is the mechanism the thesis claims as its contribution, and in the
model it consists of exactly two things.

**A privately-financed build channel.** `solar_eaas_add[t]` accumulates
into installed solar capacity additively and symmetrically with
`solar_public_add[t]`. The two are physically identical — same unit
capex, same capacity factor, same dispatch, same losses. What differs is
that EaaS capex is privately financed and therefore **never enters the
public budget constraint**.

**A bankability test that gates it.** For each year:

    capex[t]·eaas_add[t] − subsidy[t]
        <= tariff · energy_per_MW · remaining_npv_factor[t]
           · eaas_add[t] / DSCR

Both sides scale linearly in `eaas_add[t]`, so this is a per-MW
admissibility test, not a level: EaaS build is permitted in year t if
delivered-energy revenue, discounted at the private rate and divided by
the coverage cushion, covers capex net of any subsidy.

**`remaining_npv_factor[t]`'s basis is a named arm, `bank_tenor_case`**
(rate-coherence task, Part 1). By default (`bank_tenor_comm`) "the
private rate" above is the objective's own commercial tranche —
`r_comm_nominal`/`L_comm`, §4.2 — real-deflated at the point
`remaining_npv_factor` is built, truncated at `min(L_comm,
T_last-t+1)`, the SAME rate and a comparable tenor to the capital
charge's own commercial-tranche CRF, closing the incoherence a prior
version of this section did not yet have a name for. A `bank_tenor_legacy`
arm retains the previous form (the blended concessional/commercial WACC,
undiscounted by any tenor to the model horizon) for attribution of
every EaaS/bankability result produced before this correction, and one
live comparison (the FIN-2 tariff sweep, whose own analytics are not
rebuilt in this task) is explicitly pinned to it rather than left to
the new default.

**What the public budget actually bounds.** The constraint is on
discounted public solar capex plus storage capex plus the EaaS
**subsidy** — never EaaS capex itself. And the subsidy is forced to zero
whenever the tariff covers capex. So under a binding ceiling, EaaS draws
nothing on the public purse.

**That is the decoupling mechanism, and how far it reaches.** As the
public ceiling tightens, the system holds the same plan while the
bankability constraint is slack. Above approximately 0.40 × B\*,
total solar and system cost are unchanged. How that build is
financed is not identified at any point on the plateau (F-12), so
the mechanism is described by what it leaves fixed rather than by a
split it moves. Below 0.40 × B\* the EaaS channel is bounded by
bankability and storage remains public-budgeted, so both total solar
and system cost move — 33,302 MW and 30.34 bn USD at a zero envelope
against 39,030 MW and 27.318 bn unconstrained.

**The tariff.** `tariff_case`, required, no default. Sourced natively in
USD from the two positions in Nigeria's 2016 NBET solar IPP negotiation
— the agreed rate and the government's counter-position — with the
midpoint as central and both declared as a construction rather than an
independent observation. Two legacy arms are retained for attribution
only, both unsourced: one was a silent default inside `load_scenario()`,
the other a bare literal repeated across eight runner sites.

**Declared scope boundary.** Because the two channels are physically
identical, the model can say how solar is paid for but nothing about
where it is sited. Two sourced corrections that would distinguish them —
a behind-the-meter loss advantage and a distributed capex premium plus
capacity factor — are implemented in neither direction. They oppose each
other, so the net bias is not established, and they must be implemented
together or not at all.

---

## 5. Gas representation

### 5.1 Deliverability as the binding constraint

**Status: SETTLED (mechanism), PROVISIONAL (arms).**

Gas generation is limited by fuel delivered to the power sector, not by
installed capacity. The anchor is 89.27 TWh_th, DERIVED from observed
2024 generation and measured fleet efficiency, cross-validated three
ways to within ~1% (NERC generation, gas transmission volumes, and the
efficiency identity).

**The anchor is a commercial and infrastructural constraint, not a
geological one.** Nigeria holds substantial proven reserves with large
upside; the binding constraints are processing capacity, midstream
transport, and offtaker non-payment. A reservoir-decline representation
would model the wrong physics and is not used.

**Two epistemic families of deliverability arm, kept separate.**

- **Historical-drift family** (`downside`, `baseline`, `upside`,
  `shock_recovery`, plus flat level-equivalents): extrapolates observed
  behaviour. This family supplies the CENTRAL case, on the reasoning
  that a central result should not assume policy succeeds.
- **Policy-target family** (`gas_deliv_flat`, `gas_deliv_gmp_partial`,
  `gas_deliv_gmp_full`): discounts a stated national gas-production
  target through an allocation chain (production -> monetisation ->
  domestic share -> power share), applied as a multiplier on the
  observed anchor. The level is NEVER taken from the chain: the policy
  document's power-sector figure is DEMAND, the anchor is observed
  DELIVERY, and the gap between them is the deliverability shortfall the
  model exists to represent.

An **implementation factor** `I` scales the planned increment, not the
level, sourced from historical domestic-supply-obligation compliance.
The two families must never be compared without stating which is which.

### 5.2 Open-cycle to combined-cycle conversion

**Status: PROVISIONAL — implemented and swept.**

The model could build new combined-cycle capacity but could not improve
or convert the existing fleet, so retrofit was absent from the choice
set. Under a binding deliverability constraint, conversion is the only
route to more generation without more fuel, so a cost-optimality claim
does not hold while it is missing.

Conversion is a **decision variable**, not a parameter change, and it
adds **both efficiency and capacity** — a bottoming cycle is a physical
addition of plant. Representing efficiency alone would remove the larger
benefit and would place conversion and new-build on different bases for
the same physical outcome.

**Merit ordering by measured availability.** Capex is paid on nameplate;
the benefit accrues only on available capacity. Per-plant availability
factors, extracted from regulator quarterly reports across three
periods, band the convertible fleet into three tiers. This is
load-bearing: the open-cycle roster is 6,482 MW of nameplate but only
~1,941 MW of available capacity, and without the screen the model would
pay to convert plants running at 0.01% availability.

Converted plant dispatches through a parallel slice-resolved stream at
retrofit efficiency, drawing on the same fuel envelope. Retrofit
efficiency is ASSUMED below greenfield combined-cycle efficiency, on the
grounds that a turbine not specified for bottoming-cycle exhaust
conditions will not reach it.

**Rehabilitation — closing the fleet's degradation gap at fixed
technology mix — is OUT OF SCOPE.** Its ceiling is a
composition-adjusted efficiency expectation derived with unsourced
indicative efficiencies and capacity- rather than generation-weighting.
Declared, not implemented.


### 5.3 Solar resource

**Status: SETTLED.**

Three parameters, all sourced from the same satellite-derived resource
database at named Nigerian sites, each modelled at its own optimum fixed
tilt.

**Capacity factor.** Central 0.19, band 0.16-0.20 spanning the measured
national range from a southern site to the northern maximum. The
previous central was the national maximum, which assumed every plant is
built at the best site in the country and sat above the national
planning document's own assumption. The previous high arm was above
every measured Nigerian site and is retired as unattainable.

**Seasonal factor.** Dry 1.076 / wet 0.925, the day-weighted mean across
three ground-mounted sites, against a retired 1.15/0.85 that overstated
the contrast at every site measured. The dry season is not uniformly
better: output peaks in November and declines monotonically to April as
harmattan dust loads, a 15% fall within the block. The two-block factor
averages that and cannot represent it.

**Distributed solar is a different asset.** Measured at identical
coordinates so siting is held constant: a commercial system loses 3.1%
against ground-mount at the same tilt; a flat roof 4.5%; an off-south
30-degree roof — the representative Nigerian pitch, steep for rainfall
runoff — 13.6%. The penalty is almost entirely azimuth, not tilt.

Off-south orientation also FLATTENS the seasonal profile, from a
dry/wet ratio of 1.146 for utility ground-mount to 1.02-1.04 for
rooftop. A south-tilted panel captures low-angle dry-season sun, which
is what produces the seasonal boost; an off-south panel does not, so its
output tracks raw irradiance, which barely varies. The distributed and
utility channels therefore differ in seasonal SHAPE, not only in level —
and firm capacity binds in the dry peak block.

---

## 6. Hydro representation

**Status: SETTLED (specification), PROVISIONAL (arms).**

Hydro is **exogenous** — a fixed annual energy series with capacity
derived at a fixed capacity factor, never a decision variable. As a free
variable in a least-cost LP it is over-built and crowds out the solar,
storage and financing dynamics the thesis is about.

Three arms: flat at the observed level, committed (including the most
recent large commissioning), and a policy-aligned trajectory. Only the
flat arm reproduces the base-year backcast; the others depart from
observation by construction, because raising exogenous hydro displaces
gas in the balance.

**Validity guards, not constraints.** Because hydro is exogenous, a
resource ceiling as an LP constraint could only be trivially satisfied
or make the model infeasible — it can never change the optimum. The
ceiling is therefore a guard raising at scenario-load time if a
trajectory exceeds sourced exploitable potential, plus a build-rate
warning expressed in units of the most recent large plant commissioned.

### 6.1 The seasonal factor was signed wrongly, and the correction is hydrological

The retired parameter set dry-season hydro 18% BELOW wet. Measurement
across two independent NERC datasets — one of them pre-dating the most
recent large commissioning, which is the confound the control was
designed to remove — gives dry 4.5% ABOVE wet.

The mechanism is the Niger's bimodal flood regime. The river carries two
floods a year, and one arrives in the local dry season. The white flood
comes from local tributaries, silt-laden, rising in July and peaking in
late September. The **black flood** comes from the upper catchment
outside Nigeria, having deposited its silt crossing the Inner Niger
Delta — hence "black", meaning clear — and its peak reaches the main
reservoir in January-February, falling by an order of magnitude by
May-June. The documented minimum inflow is April to July; the reservoir
level peaks in December.

The retired value applied a textbook run-of-river assumption to a river
whose principal flood arrives in the dry season. This was not a
calibration error but a category error about the resource.

**Declared limitation of the six-slice design.** The two-season split
cuts across the hydrograph. The dry block contains the black-flood peak
AND the onset of the trough; the wet block contains the trough AND the
white flood. Both blocks average a peak against a minimum. The
underlying quarterly swing is a factor of 1.49; the two-block factor
captures 1.094 of it. The seasonal signal is correctly signed and
substantially UNDERSTATED by the model's temporal structure.

**Recorded: a duplicate literal source of truth (C-6, not fixed).**
`scenarios.py:2919` carries a second literal copy of the corrected
factor, `{"dry": 1.045, "wet": 0.956}`, used to compute
`capacity_credit["hydro"]` before the scenario dict itself exists;
`scenarios.py:3140` carries the canonical scenario-dict entry the two
are commented "KEEP IN SYNC" against (which itself notes two further
copies of the same pattern for `season_solar_factor` in
`optimize_model_sliced.py`'s `DEFAULT_HYDRO_SEASON_FACTOR` /
`DEFAULT_SEASON_SOLAR_FACTOR` fallbacks — four copies in total across
the two factors). All are currently in sync, but nothing enforces that
beyond the comment — a future edit to one without the others would
silently desynchronise the seasonal factor used for dispatch from the
one used for firm-capacity credit.


---

## 7. Verification discipline

**Status: SETTLED. This is the part of the method that caught the
errors.**

**7.1 Regression arms.** Every structural change carries a named
scenario value that reproduces the pre-change behaviour exactly. The
arm is run FIRST; if the objective does not reproduce to full precision,
the change did something unintended and interpretation stops.

This is why the pre-correction objective form is retained behind a named
arm rather than deleted, why the coverage cushion retains its
unsourced historical value as a named arm, and why every technology arm
has a "none" value.

A regression arm that reproduces does not certify the new behaviour. It
certifies only that the old path is undisturbed. Where a parameter's
role changes — as asset lifetime's did, from feeding a salvage credit to
bounding a capital recovery window — the regression arm cannot test the
new role, and that gap is stated rather than assumed away.

**7.2 One variable at a time.** Three errors in this project came from
inferring a cause from an uncontrolled comparison. The rule: change one
variable, hold everything else fixed, and check the regression arm
reproduces before interpreting anything.

**7.3 Decompositions must be order-invariant.** A one-at-a-time
attribution of a multiplicative effect is order-dependent and can invert
the conclusion. Shapley or log-decomposition is used where an effect is
split between causes.

**7.4 Materiality before verdict.** The threshold for calling a
sensitivity result material is set before the result is read, not after.

**7.5 Backcast.** First-year gas generation is compared against observed
generation in the calibration year. The current central configuration
reproduces it to within **-3.2%** (previously -3.4%, before the O&M
implementation at `d7538ff`). When a channel is added that the backcast
does not see — as conversion was — the check is corrected at the
consumer rather than allowed to mis-report.

**7.6 Lexicographic identification testing.** A linear program can have
a flat face: many solutions achieving the same objective. A quantity
reported from such a solution is one point in a set, not an estimate. The
test: solve normally for the optimal objective Z*, add
`objective <= Z*·(1+tol)` as a constraint, then re-optimise twice on the
quantity of interest — once minimising, once maximising. The gap is the
identification interval.

Two design points make it informative rather than decorative. The
tolerance is swept, because a wide interval at one part in a million is
structural degeneracy while an interval that only opens as tolerance
grows is numerical. And the per-period envelope is reported alongside the
total, because a well-identified total can hide an unidentified
trajectory.

The test is not a diagnostic of a defect. Where a quantity is bounded by
different constraints at each end, the interval measures the gap between
them, and its width is itself a result.

**7.7 What lexicographic testing established here.** Applied to the
public/private solar split, the test found total solar, gas capacity and
self-generation IDENTICAL at both the minimising and maximising solve,
in every capital-budget arm. Only the ownership split moved, and the two
channels sum to the same cumulative total in every solution.

The two channels are therefore perfect substitutes at the margin at zero
cost difference. The identification interval is not a defect: total solar
is identified, and the split is a pure financing choice that the model is
indifferent to except where the public budget forces it. The upper edge
of the interval is total solar itself, not a bankability limit — which is
why raising distributed capex by up to 37% did not move it.

**7.8 Cross-cutting audit discipline.** The required-key discipline
introduced in §3.2, the do-not-report screening protocol, epoch stamping
across recalibrations, and the SOURCED/DERIVED/ASSUMED sourcing
classification all apply project-wide rather than to this model layer
specifically. Their canonical account is
`docs/methodology_protocols.md`; not repeated here.

---

## 8. Open method decisions

**Status: OPEN.**

- **Central configuration for the reported result set — RESOLVED.** A
  single central configuration is declared in `docs/findings_frozen.md`,
  "The central configuration," and reproduced unchanged at HEAD
  `8fc5ccb` (historical record, 2026-09-20): central 27.954 bn, storage
  65,613.0 MWh, solar 41,031.4 MW, backcast -3.2%
  (`results/anchor_central_postbstar.log`, post-`5beaef0`/`85f2598`).
  **Superseded 2026-10-09 (TASK 42):** those figures belong to the pre-rate-coherence
  epoch. The current central anchor is 27.318 bn (27,317,670,155.72), storage
  63,056.31 MWh, solar 39,030.07 MW, and it was unchanged across the Phase B re-run.
- **Rate endogeneity beyond capital structure.** Tranche rates are
  currently exogenous; only the structure responds to the model's
  choices. Whether rates should respond to the model's own leverage or
  drawdown is unresolved, and the form of any such mechanism is gated on
  the current results rather than chosen in advance.
- **Storage capacity credit.** A flat credit is applied irrespective of
  penetration. Effective load-carrying capability declines steeply with
  penetration, so the flat value overstates storage at multi-GW scale,
  and storage has no interior optimum once its competitor reaches a
  floor.
- **Midstream investment cost.** Deliverability expansion is currently
  free to the model. **The post-hoc comparison now exists**, as M-1
  (`docs/findings_frozen.md:1138-1150`): implied midstream investment
  11-30 bn USD at a capex intensity I=0.5, 22-60 bn at I=1.0, 15-40x the
  DARES concessional envelope. It remains **not integrated into the LP**
  — the comparison is a post-hoc diagnostic, not a constraint or cost
  term the model solves against.
- **Solar sits near a cost knife-edge in the corrected model.** A lower
  capacity factor produces a CHEAPER system (0.16 -> 20.086 bn against
  0.19 -> 20.856; **pre-O&M figures, `d7538ff`** — O&M adds +30.9% at
  central and has not been re-measured across the CF arms). Single-variable
  and reproducible, so it is real. It implies solar is cost-increasing at
  the margin — built because something forces it rather than because it
  pays. Not diagnosed.
- **Two tenors for one asset.** The objective amortises commercial
  capital over the sourced debt tenor; the bankability constraint
  discounts revenue to the model horizon at a retired blended rate. The
  errors partly cancel at the first vintage and diverge thereafter.
  Identified, declared, not corrected; any bankability-dependent result
  is provisional on it.

  **Buffer-year capital spend, now measurable.** Two gross-capex fields
are produced: `npv_gross_capex_report_window_usd` (2025-2045) and
`npv_gross_capex_full_horizon_usd` (2025-2054). At the central
configuration these are 12.772 bn and 14.350 bn, so **1.578 bn — 11.0%
of full-horizon capital — falls in the buffer years.** Under
`conv_none` the buffer share is 2.162 bn of 17.182 bn, 12.6%.
**[Values superseded, TASK 46, 2026-10-10: those four figures are the
frozen state (`2a9226b`). Current epoch (`summary__02c17f55.json`,
`summary__062eba52.json`): central 10.153 bn and 13.215 bn, so 3.062 bn —
23.2% of full-horizon capital — falls in the buffer years; `conv_none`
12.530 bn and 15.564 bn, 3.035 bn, 19.5%.]**

That is a material fraction, and it is why the two fields are reported
separately rather than one being used for both purposes. Buffer-year
figures are on the do-not-report list by construction: they carry the
2050 storage rebuild and the terminal-year capital activity that a
terminal-year salvage arbitrage concern is about. **This is not F-16**:
F-numbers were reassigned at the freeze, and the finding this used to
refer to ("Terminal-year salvage arbitrage") exists only in the
superseded `findings_provisional_pre_sweep.md:121`; F-16 in the current
`findings_frozen.md` is an unrelated finding ("NIRP's 2045 hydro exceeds
independent exploitable potential"). No live finding in
`findings_frozen.md` currently covers the salvage-arbitrage concern
itself — this sentence names the concern, not a citable finding.
`npv_capex_report` in
`src/optimize_experiments.py` is the report-window figure its name
states -- truncated to years at or before `REPORT_END_YEAR` -- and
feeds `npv_gross_capex_report_window_usd`; `npv_capex_full` is its
untruncated, full-2025-2054-horizon counterpart, feeding
`npv_gross_capex_full_horizon_usd`. Three runners consume the
report-window field: `01_run_baseline.py`, `01b_run_baseline_sliced.py`,
and `29_run_parameter_uncertainty.py`.
