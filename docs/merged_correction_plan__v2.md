# Merged Correction Plan (v2)
## Nigeria gas-constrained power-sector model — consolidated fix sequence

Status markers: `[DONE]` complete and verified · `[PARTIAL]` started, requirements
outstanding · `[OPEN]` not started · `[BLOCKED]` waiting on primary-source data.

Scope markers: `[MUST]` thesis not defensible without it · `[SHOULD]` materially
strengthens · `[STRETCH]` only if the core lands early.

**Confirmed context (SLR, completed and now strengthened):** no sub-Saharan
African power-system planning study post-2014 endogenises cost of capital —
**including Nigeria's own official Integrated Resource Plan (NIRP 2024)**,
produced with World Bank, GEAPP and UKNIAF support, which treats the discount
rate as a Section 4.4 sensitivity, defers financing arrangements to the SIP
(footnote 12), imposes no capital-budget constraint on $122bn of undiscounted
capex, and confines financing to a narrative risk row in Table 12. The claim is
evidenced by screening, not asserted, and now covers national planning practice
as well as peer-reviewed literature.

**Positioning against NIRP: complement and critique, never competitor.** NIRP
asks what is least-cost; this thesis asks what is financeable. A four-zone PLEXOS
model with transmission, DSM, CSP, hydrogen and nuclear cannot be beaten on
scope, and should not be challenged on it. It is beaten on the one dimension it
explicitly declines to model — with the national plan's own footnote and risk
table as evidence.

**Structural decision (accepted):** bulk capacity (gas, hydro, public solar) is
planned nationally at transmission voltage; **EaaS solar is distributed,
behind-the-meter**, motivated by the Electricity Act 2023 eligible-customer and
captive provisions. Consequences are itemised as `[DIST-n]` below and must be
implemented **as a set** — the exemptions favour EaaS and the CAPEX premium
offsets them. Implementing only the favourable half makes the EaaS result an
artefact.

---

## PHASE 0 — Restore trust in the pipeline `[DONE]`

**0.1 `[DONE]`** KeyError landmine removed (`storage_solar_surplus_frac` direct
index in diagnostics); full-tree grep confirms no remaining readers of removed
keys; pipeline executes end-to-end.

**0.2 `[DONE]`** Deprecated dispatch simulator severed from `01_run_baseline.py`
(the spurious 155 TWh "dispatch mismatch" is gone). STR-1 suspended with an
honest docstring stating the algebra. Storage-binding diagnostic no longer
reports `power_limit` on zero capacity.

**0.3 `[OPEN]` `[MUST]` Metric formula fixes — these are reported numbers.**
- **Public capital multiplier:** currently `total EaaS capex / public spend`,
  counting the subsidy-funded slice as mobilised private capital. Change to
  `(capex − subsidy) / subsidy`; report both definitions.
- **Solar share:** denominator omits hydro and includes `storage_discharge`
  (double-counting). Denominator = gas + solar + hydro (+ genset backstop later).
- **Gas scarcity value:** delete `gas_curtail` (zero objective coefficient →
  degenerate). Report the annual dual λ_t directly. Replace the `max(0.0, val)`
  clamp with an assertion — the clamp would silently zero a wrong sign convention.

**0.4 `[OPEN]` `[MUST]` Numerical hygiene.**
Replace the `emissions_cap = 1e18` sentinel with *no constraint*. Delete or
normalise the weighted-sum objective (adds USD ~10¹⁰ to tCO₂ ~10⁸). Rescale to
B$/TWh/GW. Move CBC → HiGHS; pin solver version and tolerances.

---

## PHASE 1 — Re-anchor the physics

**1.1 `[DONE]` Gas efficiency measured, not assumed.**
η = 0.287 from the identity `observed TWh_e ÷ observed TWh_th`, using NERC 2024
generation (25.62 TWh_e gas) and reported gas-to-power volumes (800–860 MMscf/d,
HHV 1036 Btu/scf). Swept {0.277, 0.287, 0.298} for the unresolved
billed-vs-combusted measurement boundary. Replaced 0.43 (CCGT figure on an
OCGT-dominated fleet).

**1.2 `[DONE]` Emission factor coupled to efficiency.**
Thermal-basis factor stored (0.1823 tCO₂/MWh_th, IPCC 56.1 kg/GJ NCV converted to
HHV); electrical factor **derived** at build time as `EF_th / η` = 0.635
tCO₂/MWh_e. Sweeping η now propagates consistently to emissions. Cap builder
updated to the same basis (182,300 tCO₂/TWh_th).

**1.3 `[DONE]` Solar capacity factor.**
CF = 0.20 central, swept {0.18, 0.20, 0.22}, fixed-tilt to match ATB fixed-tilt
CAPEX. Replaced 0.27. Stale `0.27` fallback default removed from diagnostics.
CF-1 sensitivity runner built and validated: NPV 10.49–12.25 bn across the band
(17% swing), solar build 14.9–18.3 GW.
*Outstanding:* name the Global Solar Atlas sites in the sourcing comment.

**1.4 `[PARTIAL]` `[MUST]` Metering point, losses, and evacuation.**
- `[DONE]` **Metering point declared: gross generation at busbar.** Demand base
  corrected 23.08 → **37.09 TWh**. The 23.08 figure was *collected* energy — a
  revenue quantity — excluding electricity consumed but stolen, unmetered or
  unbilled. Loss ladder documented (37.09 gross → 7% TLF → 34.50 to DisCos →
  36.4% ATC&C → 21.97 collected).
- `[DONE]` **Deliberate deviation from the original plan:** rather than applying a
  swept generation-to-demand loss wedge, the metering point moved to the busbar so
  distribution losses fall outside the model boundary. Document as a scoping
  choice, not an omission.
- `[DONE]` **Gas deliverability re-anchored** 40.66 → **89.27 TWh_th**
  (= 25.62 / 0.287, ≈805 MMscf/d, at the low end of the reported range). Scenario
  shapes are multiplicative and rescale proportionally.
- `[DONE]` **Backcast passed.** Model first-year gas generation 24.57 TWh_e vs
  observed 25.62 (−4.1%), zero unserved. Only gas is a genuine test; total
  generation and hydro match by construction.
- `[OPEN]` `[BLOCKED]` **Evacuation bound.** Not implemented. The model currently
  plans ~25 GW cumulative solar against TCN wheeling capability of order 8 GW.
  Anchor on **capability, not observed throughput** — observed 34.5 TWh wheeled is
  a lower bound (generation was gas-limited), and using it would make the
  constraint bind artificially from year one. Form:
  `annual wheeling capacity = peak_GW × 8760 × load_factor`, growing on a
  trajectory **independent of demand** (else the constraint is decorative).
  Indicative behaviour at 8 GW × 0.55 LF = 38.5 TWh/yr base: stalled 2%/yr binds
  from 2031; trend 4%/yr never binds. Three arms: stalled / trend / expansion.
  **Sequencing note:** at annual resolution this is a weak proxy for an
  instantaneous, locational constraint. Consider deferring until after 2.1 (time
  slices) so it binds on the peak slice, where evacuation actually fails — doing
  it twice is waste.

**1.5 `[PARTIAL]` `[MUST]` Committed projects in the baseline.**
- `[DONE]` Hydro *level* corrected to measured **11.47 TWh** (NERC 2024, 30.92% of
  37,093.70 GWh). Replaced flat 8.0.
- `[OPEN]` **Hydro trajectory not built.** `hydro_growth` still 0.0, so Zungeru's
  ramp is unrepresented and hydro under-delivery is not a tested condition. Build
  two exogenous arms: *committed-only* (Zungeru + minor additions; 2025 ≈ 12.8 TWh)
  and *NDC-aligned* (toward the NDC's 10,400 MW by 2030). Keep hydro out of the
  optimiser — as a free variable an LP over-builds it and crowds out the
  solar/storage/financing story.
- `[OPEN]` **DARES / Mission 300 solar procurement not in the baseline.** Check
  `solar_baseline_mw = 500` against current installed capacity.

**1.6 `[DONE]` `[MUST]` Demand growth arms — sourced and triangulated.**

*Superseded reasoning (retained for provenance):* the original item proposed
sourcing a Tier 1 organic rate from **historical Nigerian generation growth**.
That was wrong, and wrong in the same way the 23.08 TWh demand base was wrong.
Nigerian grid generation is a **supply series**: it is flat because gas supply,
wheeling capability and the payment chain held it flat — the very constraints
this thesis evaluates. Calibrating demand growth to it assumes the constraint
persists, then reports modest capacity needs *because* demand was assumed not to
grow. Circular.

**Step 1 `[DONE]` — the per-capita diagnostic settles it empirically.**
Workbook `data/demand/step1_per_capita_diagnostic.xlsx`, documented in
`data/demand/README_step1_per_capita.md`.

| Basis | Generation | Denominator | Intensity |
|---|---|---|---|
| Total population, 2010–2025 | +3.14%/yr | +2.39%/yr | **+0.73%/yr** (76% demographic) |
| Total population, 2014–2024 | +1.53%/yr | +2.27%/yr | **−0.73%/yr** |
| **Connected** population, 2010–2024 | +2.63%/yr | +4.36%/yr | **−1.66%/yr** |

Per-capita grid generation rose ~157 → ~175 kWh over fifteen years — about
**2.1 W of additional continuous supply per person**. Per *connected* person it
**fell 21%** (327 → 258 kWh) and is negative on every window tested. Connected
population grew faster than generation throughout: Nigeria connected people to
the grid faster than it added electricity to send them. The series carries no
usable signal about consumption intensity. Caveats [F1]–[F8] in the README;
notably WDI access includes off-grid, so the connected-basis decline is
exaggerated in magnitude though not in sign.

**Step 2 `[DONE]` — NIRP 2024 interrogated.** Three findings:
- **Cost of capital is exogenous.** Discount rate appears only as a Section 4.4
  sensitivity; footnote 12 defers financing arrangements to the SIP; there is no
  capital-budget constraint against $122bn of undiscounted capex; financing
  appears in Table 12 only as a narrative risk. **The SLR claim strengthens** and
  should now read: *no sub-Saharan African power-system planning study post-2014
  endogenises cost of capital, including the official 2024 national IRP produced
  with World Bank, GEAPP and UKNIAF support.*
- **NIRP validates the method.** Table 14 derives demand bottom-up from
  population, GDP, electrification and loss trajectories — explicitly not from
  historical generation. Cite it when defending the choice above.
- **λ is now sourced.** Grid ÷ (grid + self-gen) = 32.3/59.6 = **0.542** (2022),
  35/62 = **0.565** (2025); Section 4.2's "self-generation is 49% of total
  on-grid demand in 2024" implies ~0.51. Report the range **0.51–0.57**. Applied
  to Tier 1, this gives **Tier 2 ≈ 65–73 TWh (2024)** — feeds 2.7.

**Steps 3–4 `[DONE]` — triangulation and floor test.** Three independent routes,
central value adopted only where at least two agree.
- **Route A, driver decomposition:** `g = ε·g_GDP + (1−ε)·g_pop`. With ε = 1.19
  (NIRP-implied) and g_pop = 2.1% this reproduces NIRP's 7.70% exactly, which
  validates the specification. Note the coefficient on population is (1−ε) ≈
  −0.19: **g is insensitive to the population assumption and is approximately
  1.19 × GDP growth.** The GDP assumption is ~6× more load-bearing.
- **Route B, per-capita convergence:** Nigeria 2024 = 161 kWh/cap; **~15 years
  behind Kenya** (Kenya crossed 161 kWh/cap in ~2009). Ghana is unusable as a
  target — Nigeria's 2024 level is *below Ghana's worst year since 2000*, and
  converging to Ghana's 2024 level would require 9.24%/yr.
- **Route C, NIRP 2024:** total demand 59.6 → 328.6 TWh (2022–45) = **7.70%**,
  which is **Tier 2 scope**. NIRP's grid-only 11.0% is organic growth *plus*
  self-generation absorption and must **not** be applied to a Tier 1 base — in
  this architecture suppression closure is the Tier1→Tier2 gap, not a growth
  rate.
- **Floor test:** population growth 2024–2045 = **1.77%/yr** (UN WPP, 232.68m →
  336.66m). All arms clear it. `constrained_continuation` implies 1.35%/yr
  per-capita growth, *above* the 0.73% observed — so even the pessimistic arm is
  mildly optimistic against the record.

**Steps 5a/5b `[DONE]` — implemented, renamed and verified.**

| Arm | Value | Sourcing |
|---|---|---|
| `constrained_continuation` | **0.0314** | Ember 2010–2025 generation CAGR. Constraint-persistence, **not** organic demand. Window range 1.53–3.22% |
| `organic_central` | **0.040** | Route A (ε 0.83, GDP 4.5%) = 4.04%; Route B (Kenya 2024 level) = 3.67% |
| `organic_high` | **0.060** | Route A (ε 0.90, GDP 6.8%) = 6.30%; Route B (Ghana 2015 level) = 6.40%; NIRP low = 5.9% |
| `nirp_2024_base` | **0.077** | NIRP base, Tier 2 scope — held in `demand_growth_benchmark()`, **outside** the arm registry |

ε = **0.76–0.90** from Liddle, Smyth & Zhang (*Energy Economics*), middle-income
**non-SSA** panel. SSA-estimated elasticities (0.60–0.76) are rejected as a
central source: estimated on supply-rationed systems, they inherit the identical
contamination as the Nigerian series. Retained as a declared lower bound.
**NIRP's implied ε = 1.19 is ~35% above the credible literature**, on top of a
6.8% GDP path — aggressive on both terms simultaneously, and a quantified
critique for the critical-review chapter.

*Honest outcome:* `organic_central` landed at ~4%, essentially where the
unsourced placeholder was. The value barely moved; its **status** moved
entirely. Say so in the thesis — a referee will respect that more than a number
that conveniently doubled.

**Verification (Step 5).** 5a renamed keys with values frozen — 30 call sites,
byte-identical across 21 runners, which is a free exact regression test. 5b
changed one value. The 21-runner sweep confirmed changes confined to
`results/dem2/` (5 files); within `dem2_results.csv` the diff hunks are
2,9 / 26,33 / 50,57 — the `constrained_continuation` blocks only — with
`organic_central` and `organic_high` rows byte-identical **between** them. That
interleaving is the internal control. Emissions caps unchanged. Baseline
objective 1.613558543e10 unchanged. MC draw moments match the prior.

**A finding from the regression residuals.** Under 5b the EaaS arm's gas premium
moved 0.00001% (491,648,215.6 → 491,648,164.3 — solver noise on an identical
problem) while the `public_only` equivalent moved 31% (63.1bn → 83.0bn). The
public arm is hypersensitive because it is budget-bound; EaaS is not. **This is
the financing thesis, visible in the residuals of a regression test** — and it is
the strongest available argument for sequencing 2.6 before Phase 3.

**Step 6 `[OPEN]` `[SHOULD]` Piecewise g.** NIRP's trajectory is front-loaded
(10.8%/yr to 2030, then 6.9%). A constant g reaching the same D(2045) has the
same terminal build but pulls investment *later*, understating near-term capital
pressure and financing stress — exactly the years when the envelope is tightest.
Requires changing `demand.project_baseline_demand`, which takes a scalar.
Deferred deliberately: one structural change at a time.

**Step 7 `[DONE]` — cap invariance CONFIRMED.** Caps rebuilt from a
`constrained_continuation` baseline are **byte-identical** to those from
`organic_central`. Hypothesis confirmed exactly, not approximately: baseline gas
is deliverability-bound from 2026 (2025 already runs 24.57 of 25.62 available),
so `Ebase` is set by the gas trajectory and not by demand growth. **The NDC caps
are robust to any future revision of `g`.** Report as a robustness statement in
the cap chapter. Incidental result: the `constrained_continuation` baseline
objective is **13.59bn**, against 16.14bn for `organic_central` (−15.8%).

**Step 8 `[OPEN]` `[MUST]` — coherence cross-tab.** Declare which
demand-arm × gas-regime cells are internally coherent. See
**CONSTRAINT-RELIEF ASYMMETRY** below — the central case sits in the asymmetric
cell and must be argued, not inherited.

---

## CONSTRAINT-RELIEF ASYMMETRY — declared load-bearing assumption `[OPEN]` `[MUST]`

Demand arms are sourced from **unconstrained drivers** (population, GDP,
elasticity) — i.e. assuming the constraints lift. Gas deliverability is anchored
to **observed constrained throughput** (89.27 TWh_th) on a constraint that Phase
6.1 establishes is **commercial, not geological**. The model therefore assumes
constraints lift on the demand side and persist on the supply side.

**That asymmetry generates the entire solar residual**, and therefore the storage
build, the slicing premium and the financing frontier. It is not an error — a
requirement and a capability are different objects, and reform can plausibly
succeed downstream while failing upstream — but it must be **argued**, not
inherited by accident of which parameter was audited first.

| | gas `downside` | gas `baseline` | gas `upside` |
|---|---|---|---|
| `constrained_continuation` | **coherent** — constraints hold both sides | plausible | incoherent |
| `organic_central` | **asymmetric — the current central case** | coherent | plausible |
| `organic_high` | incoherent | plausible | **coherent** — reform succeeds throughout |

Rule for the write-up: incoherent cells are reported as sensitivity bounds only,
never as central findings. This supersedes 5.1's narrower "correlate gas regime
with demand" — the correlation structure is a **scenario design principle**, not
a Monte Carlo detail.

**A third instance of the same asymmetry:** GDP is exogenous to electricity
supply, but supply constrains GDP in reality. A scenario pairing 6.8% GDP growth
with 24% unserved energy is internally incoherent. Excluding those cells removes
demand, so the omission is **conservative**. NIRP has the identical omission and
does not flag it.

---

## PHASE 2 — The structural core

*Protect 2.1 from schedule pressure. Without it, referees can dismiss the solar
and storage results regardless of the finance layer.*

**2.1 `[DONE]` `[MUST]` Intra-annual time slices — 6 slices, {dry, wet} × {night, day, peak}.**
Built as `src/optimize_model_sliced.py` (`build_model_sliced`), a **parallel**
function; the annual `build_model` is retained deliberately so the annual-vs-sliced
comparison stays available and the other runners keep working. Pilot runner
`scripts/01b_run_baseline_sliced.py` verified end-to-end.

dry = Nov–Apr (181 d), wet = May–Oct (184 d); night 22:00–05:00 (7 h), day
05:00–18:00 (13 h), peak 18:00–22:00 (4 h). Hours: dry_night 1267, dry_day 2353,
dry_peak 724, wet_night 1288, wet_day 2392, wet_peak 736 = 8760. Solar generates
in **day slices only** (CF within day slices ≈ 0.369). Gas fuel is an **annual
budget allocated across slices**, so scarce gas can be saved for the dry-season
peak. Hydro is dispatchable within a seasonal energy budget subject to a power
limit (2,678 MW) and a minimum-flow floor. Storage charges in day slices and
discharges within the **same season**.

**Results (provisional pending shape sourcing):** backcast holds at 24.57 TWh_e;
**storage builds 28,241 MWh by 2045 versus zero in the annual model**; sliced
objective 18.31bn vs annual 16.14bn — **annual resolution understates system cost
by ~13%**. A methods finding in its own right.

**Shape parameters are all `[SOURCE NEEDED]` and all swept.** No sliced result is
reportable until sourced: `season_demand_factor`, `period_demand_factor`,
`season_solar_factor` (needs Global Solar Atlas **monthly** PVOUT for named
sites), `hydro_season_factor`, `hydro_min_flow_fraction`.

*Caution on the seasonal-scissor claim:* 2025 quarterly hydro **shares** rise
(29.9 → 38.3%) but converting to **levels** shows absolute hydro near-flat Q1–Q3
(3.9% spread), and NERC attributes the Q4 jump to *utilisation* improvements at
Zungeru/Shiroro/Dadin-Kowa/Jebba, not hydrology. The Zungeru ramp confounds any
seasonal signal. **"Dry season = low hydro" is not currently supported by the
data** — evidence it with pre-Zungeru quarterly levels (2022–2023) or narrow the
claim to the demand side.

*Known limitation:* `extract_planning_diagnostics` is annual-only; the sliced
runner bypasses it via `annual_totals()`. Consolidate as more runners migrate.
Hydro allocation across slices is degenerate while hydro is not scarce (costless
with a fixed seasonal total), so early-year hydro-by-slice is not an economic
signal. Storage `deployable_hours` is no longer a free parameter — throughput
falls out of the slice structure.

**2.2 `[OPEN]` `[MUST]` Storage — currently incapable of adding value.**
With `discharge[t] ≤ 0.9 × charge[t]` same-year and charge debited in the balance,
net contribution is `discharge − charge ≤ −0.10 × charge ≤ 0`. Storage can only
destroy energy at a capital cost, so `storage_add = 0` is strictly optimal —
empirically confirmed across all runs. **2.1 fixes this.** Then derive
`storage_deployable_hours` from the slice structure, wire storage O&M into the
objective (currently defined in `io.py`, never read), and rebuild STR-1.
**Also consider making `storage_duration_hours` a decision variable rather than a
fixed parameter.** The choice between 2-hour and 6-hour storage is a real design
decision with materially different costs, and it determines whether storage
substitutes for gas peaking or merely shifts solar within the day. Currently
fixed, so the model cannot express it.

**2.3 `[OPEN]` `[MUST]` Promote peak adequacy into the LP.**
At annual resolution 1 TWh of solar substitutes perfectly for 1 TWh of gas — solar
can "serve" the 8 pm peak. The peak check exists only as a post-solve diagnostic
the optimiser never sees. Add
`firm_capacity(t) ≥ peak_demand(t) × (1 + reserve_margin)`, solar credited 0–10%,
storage at power rating.

**2.4 `[OPEN]` `[MUST]` Resolve the peak multiple as a declared unknown.**
`scenarios.py` sets 2.5; diagnostics default to 1.82. Beyond the contradiction: a
load factor measured on *served* demand in a load-shedding system is not evidence
about peak shape — shedding truncates the peak, so the observed load-duration
curve is flat-topped by construction. Source from feeder-level or genset-usage
studies, or sweep as a declared unknown.

**2.5 `[DONE]` `[MUST]` End-of-horizon accounting, build bounds and horizon.**

Four sub-items completed. Lead times split out and deferred.

---

### Salvage `[DONE]`

The horizon was shorter than every asset life, so assets with 9-29 years of
remaining service were fully expensed and system NPV was overstated.
Straight-line residual `DF[T_last]·capex[t]·max(0, life − (T_last−t+1))/life`
per vintage, in **both** `optimize_model.py` and `optimize_model_sliced.py` —
the two must share a cost definition or the annual-vs-sliced premium compares
different things.

Lifetimes in `scenarios.asset_lifetimes()`: **solar 30 yr** [IRENA 2026];
**storage 15 yr UNAUGMENTED** [NREL ATB — ATB's 30-yr PV-plus-battery life
assumes cell replacement at year 15 and this model carries no augmentation
capex]; **gas 30 yr** [NREL ATB]. Swept via `asset_lifetime_sweep()`.

Three accounting decisions, documented in the function docstring:
- **Objective only, never the capital budget constraint.** The envelope
  constrains cash out the door; an investor cannot fund a 2045 build with
  residual value realised in 2045. Netting it inside the constraint would
  inflate the envelope ~25% for free and soften the finding that public capital
  binds.
- **Computed on RAW capex for both arms.** `required_margin` is a financing
  premium, not asset value.
- **Residual accrues system-wide.** Blurs ownership (the EaaS investor holds it
  in reality); revisit at 3.2.

**Result at the 21-year horizon: 1.6136e10 → 1.3377e10, −17.1%.** Salvage credit
$2,759,071,700, reconciling to the objective difference to the dollar. Year-1
backcast unchanged.

**Boundary guard `[DONE]`.** Storage life (15) < horizon, so vintages built early
enough retire in-model with no replacement capex charged while still being
dispatched. `solve_model()` warns post-solve; `scenario=` passed at all 30 call
sites. Verified by forcing solar life to 5 yr. **It has already earned its
place** — see the horizon item below.

**⚠ CORRECTION to this item's original reasoning.** The plan asserted that
no-salvage "drives pathological delay, which the min-build floor patches." Wrong
on both halves. The balance constraint binds every year, so the model *cannot*
delay past need. And salvage makes late builds *cheaper* in net terms, so it
**strengthens** the delay incentive. No-salvage caused a **level** error, not a
timing error.

---

### Min-build floor `[DONE]` — removed

`scenarios.py` held 0.0 while **23 call sites hardcoded 100.0** — the fourth
duplicate-source-of-truth defect. Removed in two commits using the 1.6 protocol:
**3a** routed every site through `scenarios.solar_min_build_default()` returning
100.0 (results byte-identical — a free regression test across 21 runners);
**3b** flipped the canonical value to 0.0 in one line.

**Result: the floor was not preventing pathological delay. It was forcing 100 MW
one year early.** 2025 went 100 → 0 MW, 2026 went 173.41 → 273.41, every other
year identical, **cumulative build unchanged to the megawatt**. Objective
−0.10%, consistent with a one-year deferral of ~$145M of 2025-vintage capex.

The docstring justification is refuted: the backcast gives gas 24.57 + hydro
11.47 + baseline solar 0.88 = 36.9 TWh against 37.09 demand, so **2025 = 0 new
solar is correct**. A model that builds capacity it does not need is the
distorted one.

*Note: the backcast reference moved 24.5688 → 24.744 TWh_e as a result (gas
picked up the 0.175 TWh the removed 100 MW had supplied). Deviation against
observed 2024 improves from −4.1% to **−3.4%**. Update this figure wherever it
is quoted.*

---

### Max-build cap `[DONE]` — replaced with the NIRP trajectory

The flat 2,000 MW/yr cap was unsourced and **binding** (2042-45 annual, 2039-45
sliced). Wrong in shape at both ends: it permitted 2,000 MW in 2026 when NIRP
judges ~240 MW achievable, and capped 2042-45 at 2,000 when NIRP builds 6.0-8.3
GW/yr. A constant rate against compounding demand binds eventually at any level,
so the terminal build sitting exactly on 2,000.0 was an artefact of the
constraint's **shape**, not a deployment limit.

Replaced by `nirp_solar_cumulative_mw()` — NIRP 2024 Annex F Table 19 solar
generation converted to capacity at the Table 8 CF (19% falling to 18%).
**Validates at 61,898 MW in 2045 against NIRP's published ~61 GW headline.**

Applied to **cumulative additions**, not total capacity: NIRP's 2025 figure
(180 MW) is below this model's 500 MW existing solar baseline, so a
total-capacity cap would be infeasible in year one. Mildly generous and declared
as such.

Three arms replace `conservative/baseline/aggressive`:

| Arm | Multiplier | Role |
|---|---|---|
| `nirp_trajectory` | 1.0× | NIRP's own schedule. Binds 2027-2037 against the current solution (cumulative shortfall peaks at 4,505 MW in 2032). **Report unserved energy and feasibility from this arm, not NPV, until 2.6** — the shortfall is priced at VoLL |
| `nirp_accelerated` | 2.0× | Deployment reform succeeds. The coherent pairing for constrained-gas regimes. `[SOURCE NEEDED]` for the 2.0× multiplier |
| `deployment_unconstrained` | none | Counterfactual isolating what the deployment limit costs |

**Coherence caveat.** NIRP builds almost no solar before 2032 *because it assumes
gas recovers to 88 TWh_e by 2035* — its solar schedule is downstream of its gas
assumption. Pairing `nirp_trajectory` with constrained gas is the pessimistic
stress corner, not a central case. This extends the constraint-relief asymmetry
to a third dimension; `docs/scenario_coherence.md` needs a solar-build axis.

---

### Horizon extension `[DONE]` — 2025-2054, reporting to 2045

Removing the flat build cap exposed an artefact it had been masking. Salvage
credits a final-year vintage at (life−1)/life of capex — 96.7% for 30-year solar
— discounted at the same factor the capex is paid at, so **net cost is ~3.3% of
discounted capex, about $11.5/MWh, below gas SRMC.** The model bought it:

| | before | after cap removal |
|---|---|---|
| 2044 solar add | 1,635 MW | 1,635 MW |
| **2045 solar add** | **2,000 MW (capped)** | **17,861 MW** |
| **2045 gas** | 24.67 TWh_e | **0.00 TWh_e** |
| cumulative solar | 25.3 GW | 39.3 GW |
| objective | — | **FELL 0.7%** |

The model built 14 GW more solar and cost less. A horizon artefact, not an
economic result — **and the same lesson as the min-build floor, at the other end
of the horizon: an unsourced bound was hiding a defect.**

**The fix is the horizon, not the salvage term.** Salvage is unchanged: same
formula, every vintage. With the edge at 2054 a 2045 vintage serves 10 of 30
years, salvage 67% rather than 97%, net cost ~33% of capex — no longer cheaper
than gas. 2045 solar returns to 3,781 MW and 2045 gas to 24.67 TWh_e.

**2054, not 2055.** At 2055 a 2025 solar vintage served 31 years against a 30-year
life. The plan 2.5 guard caught it, firing 31 times. Horizon extension is the
standard treatment for end-of-horizon effects; the edge cannot be removed (a
2045 vintage is not fully depreciated until 2075), only moved out of the results.

**Buffer-year (2046-2054) series are EXTRAPOLATIONS, not sourced** — no published
Nigerian demand or gas projection extends past 2045. Acceptable precisely
because those years are never reported:
- Solar/storage CAPEX **held flat** past NREL ATB coverage. Flat is conservative
  — continuing the decline would pull build forward into the reporting window,
  the exact distortion the buffer prevents. (Solar CSV extended to 2050 from
  ATB, so only 2051-54 is extrapolated; storage flat-holds from 2045.)
- Gas deliverability extends the same functional forms.
- NIRP cumulative solar cap continues at its terminal growth rate. Clamping flat
  would freeze deployment capability and could force unserved energy in the
  buffer, distorting decisions inside the window.
- **Emissions caps set NON-BINDING (1e18) post-2045.** NDC 3.0 gives no target
  past 2035, so 2036-2045 is already a proportional-apportionment construction;
  extending it would stack extrapolation on extrapolation, and a binding cap in
  2050 would change what the model builds in 2043. NDC-anchored caps unchanged:
  2030 = 13.037 MtCO₂, 2035 = 12.195 MtCO₂.
- **GAS-3 flat level-equivalents averaged over the REPORTING window only**, so
  the shape-premium basis is unchanged (70.762384 / 105.596032 / 89.182649).

**Cap loaders hardened.** Six sites in `optimize_experiments.py` plus
`15_run_pol1`'s local loader now reindex on the model's own year list rather than
returning the raw CSV column, so a horizon change cannot silently produce a
short series.

**KNOWN LIMITATION:** storage life (15 yr) is shorter than the buffer, so
buffer-year storage vintages retire in-model with no replacement capex. Affects
buffer years only; declared rather than fixed. The guard reports it every run.

---

### Reporting metric `[DONE]` — `npv_gross_capex_report_window_usd`

`system_cost_npv` now spans 30 years and is an **OPTIMISATION OBJECTIVE, not a
headline**. Reporting a 30-year financial figure beside 21-year physical results
is incoherent.

Added: discounted **GROSS capital expenditure over 2025-2045**, salvage
excluded. Gross and window-limited for the same reason the public budget
constraint is — the envelope constrains cash out the door, and an investor
cannot fund a 2045 build with residual value realised later. CAPEX only; fuel
and O&M are paid from operating revenue, not mobilised capital.

**This is the financing-relevant headline**, and it answers the question the
thesis actually asks: how much capital must Nigeria mobilise between 2025 and
2045?

Baseline: **8.87 bn** (report window) against 12.05 bn (full horizon) and a
17.24 bn objective.

**Reporting rule: whenever either figure is quoted, name the horizon and the
salvage basis in the same sentence.** The failure mode is quoting the 21-year
figure in one chapter and the 30-year figure in another.

---

### FINDING — capacity is an accounting residual

`scripts/22_salvage_sensitivity.py` runs six lifetime arms × two models and asks
whether the trajectory responds to cost. Run at four points:

| Point | Annual | Sliced | Cost swing |
|---|---|---|---|
| Pre-3b (floor at 100) | **0 MW** | 0 MW | 2.3% / 1.7% |
| Post-3b (floor removed) | **0 MW** | 0 MW | 17.1% / 22.0% |
| Post-horizon + NIRP cap | **480 MW** (`no_salvage` only, <2%) | **0 MW** | 12.7% / 14.7% |
| Post-2.6 | *scheduled* | *scheduled* | — |

**Every real-lifetime arm — 25/30/35 solar, 12/15/18 storage — is byte-identical
to central in both models, at every point.** Only the degenerate `no_salvage`
arm (all lives = 1 yr, a diagnostic switch) ever moves the plan, and then by
under 2%.

Removing the min-build floor, replacing the max-build cap and extending the
horizon between them bought the annual model 480 MW of discretion out of ~25 GW.
The sliced model has none: solar generates in day slices only, so night and peak
demand must be met by fuel-capped gas, capped hydro and storage — storage fills
the gap exactly and cost cannot move a quantity the energy balance determines.

Note the storage arms differ in objective (19.9234 vs 19.6077 bn, a $316M spread
from ±3 years of storage life) while the trajectory stays identical. **Cost
responds; quantity does not.** That is the residual property stated precisely.

*A materiality threshold (1% of cumulative build) was added to the script after
the third run: the verdict previously fired on any nonzero movement, which
overstated a 480 MW deviation as "responds to cost".*

**VoLL is the last remaining pin.** If the trajectory still does not move after
2.6, the residual property is structural — a statement about Nigeria's system,
not an artefact of the formulation. That makes 2.6 the decisive test as well as
the highest-value fix.

---

### Lead times `[OPEN]` `[SHOULD]` — deferred past 2.6/2.7

Capacity added at t generates *and earns* at t; utility solar in Nigeria runs
18-36 months FID→COD. Requires indexing additions by decision year and
commissioning year — invasive, and it interacts with salvage, the build bounds
and the financing tenor simultaneously.

**2.6 `[OPEN]` `[MUST]` Replace naked VoLL with a genset backstop.**
Add `selfgen[t,s]` to the balance at diesel/petrol LCOE (~$0.30–0.55/kWh) with
EF ≈ 0.8–1.0 tCO₂/MWh; retain a higher survey-based true VoLL for demand unmet even
by gensets. Kills three problems: VoLL-dominance of the objective; the missing
avoided-genset-emissions channel (solar displaces genset CO₂ — strengthens the
case); and the uninterpretability of "unserved," which in Nigeria has always mostly
meant "genset-served." *Supersedes the planned three-case VoLL sweep.*

**SEQUENCING DECISION `[MUST]`: 2.6 precedes Phase 3.** Evidence accumulated
during 1.6: any run combining a binding NDC cap with a binding capital envelope
produces a **VoLL-dominated objective**. This is systemic, not incidental — it
covers the Monte Carlo, the DEM-2 `public_only` cells, GAS-3 and POL-1. In the
5b regression the EaaS arm's gas premium moved 0.00001% while `public_only` moved
31%, because the public arm is budget-bound and EaaS is not. **The public-vs-EaaS
comparison currently measures a penalty parameter, not the cost of capital**, and
Phase 3 is built on exactly that comparison. The genset backstop bounds the
penalty at diesel LCOE, converting a cliff into a slope.

**A second reason 2.6 is load-bearing.** The one-*g*-three-scopes architecture
(2.7) makes λ **constant by construction**: if all tiers grow at the same rate,
λ(t) = λ(0) for all t. λ becomes a genuine *output* only under a single run at
Tier 3 demand where the system can economically decline to serve — i.e. genset
backstop plus true VoLL above. Until 2.6 lands, λ is an assumption that has been
relocated, not eliminated. **Correct the Tier 2 wording in 2.7 accordingly.**

**2.7 `[OPEN]` `[MUST]` Tiered demand architecture.**
- **Tier 1** — grid generation, 37.09 TWh. Permanent: the calibration anchor and
  the basis for emissions-cap construction. Also the conservative "no access
  expansion" counterfactual.
- **Tier 2** — latent connected demand: end-use consumption including
  stolen/unmetered/unbilled energy (~30–32 TWh from the loss ladder), **plus** the
  self-generation economy. **λ = 0.51–0.57 is now SOURCED from NIRP 2024**
  (Tables 2 and 15; Section 4.2), giving Tier 2 ≈ 65–73 TWh for 2024.
  ⚠ **Correction:** λ is *not* a derived output under the accepted architecture.
  With one organic rate applied to three scopes, λ(t) = λ(0) by construction and
  the absolute tier gap widens forever. λ becomes endogenous only under 2.6.
  Report it as a sourced scenario parameter, not as a model result.
- **Tier 3** — access-adjusted: Tier 2 plus the unconnected ~40%, on the NDC's own
  100%-access-by-2030 / 9%-per-year trajectory. **Policy-anchored, not
  speculative.**
- Deprecated `latent_low`/`latent_high` (built as multiples of *collected* energy)
  are replaced. DEM-1 is suspended and becomes the **Access experiment**.
- Caps anchor to **Tier 1 only** — the NDC accounts against the actual grid.
- Convert end-use components to busbar with a **technical** loss factor, swept
  {10%, 12%, 15%} and declared as an assumption (NERC's 19.55% bundles technical
  with commercial and cannot be cleanly split). Commercial and collection losses
  stay out of the energy balance — they belong in the financing layer.

---

## PHASE 3 — The financing contribution (the headline)

**3.1 `[OPEN]` `[MUST]` Currency and rate basis memo — do this FIRST.**
ATB CAPEX is real USD. The 18% commercial anchor is almost certainly nominal
naira-adjacent (Nigerian inflation 20–35%; 18% *real USD* would be extreme). The 6%
concessional is presumably nominal USD DFI lending. Blending a nominal-NGN rate
with a nominal-USD rate into one WACC applied to real-USD cashflows is
**dimensionally incoherent** — the bankability frontier becomes an artefact of unit
mixing. Declare the model real-USD throughout; convert every rate explicitly
(Fisher + documented FX assumption, or source USD-denominated Nigerian power
financing directly). One page; without it the finance chapter is unfalsifiable.

**3.2 `[OPEN]` `[MUST]` Tranche formulation — replaces the blended scalar.**
*The single most important fix for the financing thesis.* The current Level-2 design
contradicts its own specification: the envelope caps concessional **volume** while
`resolve_private_rate` prices **all** EaaS capital at one blended average — so
capital beyond a binding envelope still receives concessional pricing, and the
envelope dual is mispriced.

Model capital as a merit order, as gas already is. Tranches k with envelope K_k and
rate r_k:

| Tranche | Rate anchor | Envelope anchor |
|---|---|---|
| Concessional (DFI/climate funds) | named facility terms | facility size × deployable fraction |
| Guaranteed commercial (PRG/FGN-backed) | commercial − guarantee spread | World Bank PRG headroom / FGN contingent-liability ceiling |
| Unguaranteed commercial | Nigeria power CoC (Eurobond curve + sector premium) | unbounded |

Split EaaS **capital** by tranche: `capex_k[t] ≥ 0`,
`Σ_k capex_k[t] = solar_capex_param[t] · solar_eaas_add[t]`; precompute
`remaining_npv_factor_k[t]` at each r_k; capital charge `Σ_k CRF(r_k, tenor)·capex_k`.
Rates rise as cheap tranches exhaust → convex → pure LP.

**Gains:** marginal CoC becomes an *output*, moving with deployment scale, envelope
size and tariff. Three duals with policy meaning — marginal concessional dollar,
marginal **guarantee headroom** (near-unoccupied in the literature), and the tariff
at which the commercial tranche activates.

⚠ Do **not** put financing learning (r falling with cumulative MW) in the LP — it is
non-convex. Iteration layer or SOS2/MIP only.

**3.3 `[OPEN]` `[MUST]` The tariff needs a payer — now counterparty-differentiated.**
`disco_collection_rate = 1.0` is an unused stub. Set
`effective revenue = tariff × collection_rate` and sweep [0.6, 1.0] — **but see
[DIST-3]**: this applies to grid-served revenue, not behind-the-meter EaaS.

**3.4 `[OPEN]` `[MUST]` Bankability horizon and the T\* surface.**
`remaining_npv_factor[t]` sums to 2045, not to contract tenor — a 2042 asset with a
15-year contract loses 12 years of revenue the constraint ignores, making late
vintages artificially unbankable. Discount over `min(tenor, asset life)` from t.
Separately, FIN-2's reported threshold uses 2025 CAPEX and a full 21-year annuity
while the LP enforces a *year-specific* test — report **T\*(t, blend) as a surface**,
or at minimum 2025 and terminal-vintage values. (Also: the hardcoded `20`-year
levelization in the EaaS LCOE diagnostic must match the settled tenor.)

**3.5 `[OPEN]` `[MUST]` Freeze placeholders; scope the EaaS claim honestly.**
Reconcile B\* (6.13bn docstring vs 9.104bn coded). Recompute the tariff grid against
the new T\*(t, blend). Source every "PLACEHOLDER pending primary source" rate. **No
FIN result is reportable until frozen.**

On scope: the subsidy has no objective cost (correct — a transfer), is capped at
exactly the financing gap, and draws on the public budget. So whenever the budget is
slack the optimiser subsidises to the gap and bankability never binds. The causal
content of "EaaS" in the LP is *"EaaS solar CAPEX doesn't consume the public
envelope."* Legitimate and interesting — but generic private-capital mobilisation,
and `EaaS.py` (CRF, tenor, service payments, on/off-balance-sheet modes) **is never
called by the LP**. Either wire the service-payment structure in, or scope the claim
down. See **[DIST-4]** — the distributed characterisation is defended here.

---

## PHASE 4 — Policy layer

**4.1 `[OPEN]` `[MUST]` Own the cap apportionment openly.**
The cap is derived from the model's own baseline `gas_to_power`, then apportioned by
a gas-power share that is itself an artefact of that run and the η/EF choices.
The proportional apportionment is **your methodological construction, not something
in NDC 3.0.** Present it as "an apportionment rule we propose, with sensitivity to
the rule" — never "the NDC target for power." Add one alternative (grid-EF-based or
IEA-scenario-based).
*Note: caps regenerated post-η/demand correction. `Ebase` now rises monotonically
(15.61 → 16.69 → 17.11 MtCO₂) and all four scenarios show positive abatement; the
earlier negative-abatement pathology was a symptom of the half-scale baseline.*

**`[DONE]` Cap invariance to demand growth — CONFIRMED (1.6 Step 7).** Caps
rebuilt from a `constrained_continuation` baseline are **byte-identical** to
those from `organic_central`. Baseline gas is deliverability-bound from 2026, so
`Ebase` is set by the gas trajectory, not by demand growth. The NDC caps are
therefore robust to any future revision of `g` — a robustness statement worth
stating explicitly, and one that decouples the cap chapter from the demand
chapter. **The corollary is uncomfortable and belongs in the policy discussion:
because the cap is anchored to a gas-constrained baseline, Nigeria meets its
power-sector NDC trajectory largely by remaining supply-constrained.
Decarbonisation by non-development.**

**4.2 `[OPEN]` `[MUST]` Cap trajectory beyond 2035.**
Caps held flat 2035–2045 — an implicit "climate ambition stops" assumption across the
final decade. Add a third trajectory: linear decline from the 2035 cap toward
net-zero-2060, and report cap-shape sensitivity.

**4.3 `[SHOULD]` Unified fiscal envelope.**
Replace the free-standing `public_solar_budget_npv` with
`Σ DF_t·(public_capex_t + eaas_subsidy_t + tariff_shortfall_t) ≤ F`, where
`tariff_shortfall_t = max(0, requirement_t − allowed_revenue_t)` enters as
non-negative slack. Captures the core mechanism — solar/EaaS shifts energy from
FX-exposed pass-through gas costs toward fixed-capex service payments, shrinking the
shortfall and freeing fiscal space — and makes the subsidise-consumption vs
finance-transition trade-off a **dual rather than a paragraph**.

**4.4 `[SHOULD]` Tariff ceilings from avoided cost — sharper under [DIST].**
Bound the EaaS tariff above by the segment's avoided cost: Band A grid tariff for
grid-connected C&I, genset LCOE (~$0.35–0.55/kWh) for self-generation substitution.
Under the distributed reading this is *the* behind-the-meter customer's willingness
to pay, not an analogy.

**4.5 `[STRETCH]` Full band-disaggregated fiscal waterfall and segment-indexed EaaS.**
Band A / B–E demand split with MYTO tariffs and NERC collection data;
`solar_eaas_add[t,s]` with per-segment counterparty tiers feeding the tranche
structure; fixed-point iteration where r_commercial responds to sector cash-flow
metrics. **Genuinely Q1-grade and genuinely a second thesis.** 4.3 and 4.4 capture
most of the value.

---

## PHASE 5 — Robustness and reproducibility

**5.1 `[PARTIAL]` `[MUST]` Rebuild the Monte Carlo around the uncertainties that dominate.**
- `[DONE]` The prior is no longer assumed. `scenarios.demand_growth_prior()` is
  the single source: mean = `organic_central` (0.040); **σ = 0.0120, derived as
  the standard deviation of the three sourced arms**, so the dispersion now
  reflects observed disagreement between the driver decomposition, the
  convergence path and NIRP rather than an analyst's guess.
- `[DONE]` `np.random.RandomState` → `default_rng`; the `max(0.005, ...)` clamp
  (which piled probability mass on a point rather than truncating) replaced with
  reject-and-resample; realised draw moments printed every run.
- `[OPEN]` **The prior is symmetric but the arms are not.** −0.86pp to
  `constrained_continuation`, +2.00pp to `organic_high`; the upside is 2.3×
  further from centre because the downside is bounded by the observed record
  while the upside is bounded only by the GDP assumption. Equal-weighted arm mean
  is 0.0438, *above* `organic_central`. A symmetric normal understates
  high-growth outcomes; with ~2× amplification this biases cost and infeasibility
  **downward** (conservative, but must be disclosed). Preferred fix: an explicit
  **three-point discrete prior over the arms with declared weights** — an exact
  restatement of the sourcing work rather than a smooth approximation to it.
- `[OPEN]` **Independence sampling.** Demand growth and gas regime are drawn
  independently despite sharing one driver (payment chain, midstream investment,
  sector governance). This generates incoherent worlds and inflates both tails.
  The infeasibility rate is a tail statistic and is therefore overstated. Take
  the correlation structure from the **CONSTRAINT-RELIEF ASYMMETRY** cross-tab,
  not from a separate assumption.
- `[OPEN]` **Still not sampled:** λ, CAPEX trajectory, CF, FX/rates, collection
  rate. `capital_case`, `solar_build_case`, `land_case="loose"` and
  `solar_capex="solar_low"` are all fixed, and `solar_min_build_mw_per_year` is
  forced to 100.0 in every draw — so the MC explores a narrower feasible region
  than the deterministic runs. Defensible as scoping, but "Monte Carlo
  uncertainty analysis" currently reads as broader than it is. State it.
- `[OPEN]` Promote the infeasible-draw share to a **headline table** — but only
  after 2.6, and after separating genuine infeasibility from code exceptions
  (see the defect register).

**5.2 `[SHOULD]` Parameter sweeps that interact with structural fixes.**
Social discount rate {8, 10, 12}%. Upstream methane and flaring: supply-chain CO₂e
multiplier (+10–25%, sourced from Nigerian flaring/fugitive data) — for a Petroleum &
Energy Resources thesis this is the emissions refinement your own department is
qualified to demand. **Combined physical-parameter envelope run:** η, CF, gas
availability and now distributed CAPEX at joint pessimistic vs joint optimistic ends
— one-at-a-time sweeps show sensitivity to each; the combined run bounds total
uncertainty, which is the question examiners ask last.

**5.3 `[SHOULD]` Demonstrate, don't assert, that new gas stays at zero.**
`gas_add` fixed at zero with the comment "structurally non-optimal (verified)." With
7,480 MW retiring by 2045, high-growth demand and an upside gas case, that is
scenario-dependent. Run one appendix experiment with `gas_add` free across the
scenario grid and show it stays ~0.

**5.4 `[OPEN]` `[MUST]` Verification protocol.**
Dual verification by perturbation for **every** reported shadow price (gas, budget,
concessional, guarantee, evacuation, reliability): re-solve with RHS + δ, assert
ΔNPV/δ ≈ dual within tolerance. One reusable function, run in CI. Unit tests with
hand-computable toy cases for the multiplier, solar share and scarcity rent. Repro
pack: pinned environment (**Python 3.10.19, `nigeria-energy`**), pinned solver and
tolerances, data-provenance manifest for every CSV, one `make all`.

**5.5 `[OPEN]` `[MUST]` Doc–code parity sweep.**
Every mismatch is a viva question answered on the back foot. Known items:
`storage.py` implements √η-symmetric losses — the LP applies full η on discharge
only. `EaaS.py`'s CRF/tenor machinery is never called. `economics.solar_capex`
divides by `n_years` and calls it annualisation. `apply_access_adjustment` is unused.
`storage_charge_limit_years` is a dead diagnostic on a removed constraint. DEM-1's
docstring carries pre-η numbers (η=0.43, 17.5 TWh_e cap, "solar covers 24%").
**Delete, implement, or reconcile — no third state.**

---

## PHASE 6 — Framing and writing

**6.1 `[DONE]` Gas module reframed.** Arps decline-curve docstring removed.
Deliverability is midstream and commercial (pipeline capacity, vandalism, DSO,
pricing, the payment chain), not reservoir depletion — Nigeria holds 200+ TCF proved
reserves, and a "physical decline" story contradicts the Decade of Gas programme.
*Outstanding:* present the trajectories explicitly as infrastructure-and-allocation
scenarios anchored to NERC constraint data and midstream capacity.

**6.2 `[OPEN]` `[MUST]` Rename "reliability" to what it is.**
An annual unserved-energy fraction is **energy adequacy**, not reliability — no LOLE,
no LOLP, no hourly basis. Rename throughout, or an engineering examiner will.

**6.3 `[OPEN]` `[MUST]` Discussion / Limitations chapter.**
Written well these are a strength. Cover: the **liquidity chain** (disco collection →
NBET → genco → gas supplier payment failure is the documented cause of
gas-constrained dispatch, so "downside" deliverability is partly *commercial*, not
physical — at minimum map payment-discipline regimes onto the scenarios); the
**Electricity Act 2023** (see [DIST-4] — the Act motivates the distributed EaaS
characterisation; **state-level market fragmentation is an explicit limitation and
future-work item**, not modelled); **land is not the binding siting constraint —
security is** (best-CF land is northern, overlapping insecurity zones; check the
dual, almost certainly slack everywhere); the **north–south spatial mismatch**
(best solar resource is northern, demand is southern — the national copper plate
cannot see this); **Q1×4 annualisation** of NBS energy ignores dry-season hydro and
demand seasonality; and **perfect foresight** over 21 years.

**6.4 `[DONE]` Validation subsection drafted.** Documents the triangulated
calibration and the backcast, with explicit limits (single year; supply side only;
two of three compared quantities match by construction).
*Outstanding:* name the Global Solar Atlas sites; cite the gas-volume source.

---

## DISTRIBUTED EaaS — implement as a set `[OPEN]` `[MUST]`

Accepted structural decision. **The exemptions favour EaaS; [DIST-5] offsets them.
Implementing only the favourable half makes the EaaS result an artefact.**

**[DIST-1] Evacuation exemption.** EaaS generation is exempt from the wheeling
bound; gas, hydro and public solar are subject to it. Consequence: when transmission
binds, distributed solar gains value **independently of financing** — EaaS relieves
two constraints at once (capital and evacuation). This is a novel channel and maps
directly onto the Deliverability dimension. *Folds into 1.4.*

**[DIST-2] Loss advantage.** Behind-the-meter generation avoids the 7% TLF and
distribution losses entirely — one MWh generated on site displaces more than one MWh
of grid generation. Represent explicitly or distributed solar is understated.
*Folds into 1.4 / 2.7.*

**[DIST-3] Counterparty differentiation.** Behind-the-meter EaaS sells to a
creditworthy C&I customer, **not** to a DisCo. The 60–75% collection-rate risk
applies to grid-served revenue only. This is why C&I captive solar is bankable in
Nigeria while grid-scale IPP solar struggles — the payment chain differs. **Largest
single effect on FIN-2.** *Folds into 3.3.*

**[DIST-4] Defend the characterisation.** Cite the Electricity Act 2023
eligible-customer and captive provisions plus Nigerian C&I market evidence. Note
that the centralised single-tariff abstraction is one policy generation behind.
*Folds into 3.5 and 6.3.*

**[DIST-5] Distributed CAPEX premium — the offsetting cost.** Distributed and C&I
solar costs materially more per MW than utility-scale (less economy of scale, smaller
BOS, per-site development). The current EaaS/public CAPEX parity was defensible only
while both were utility-scale. **Source a distributed premium and apply it.** Also
remove EaaS from the land constraint (rooftop/captive uses existing sites).
*New item; gates the whole [DIST] set.*

**Gap identified:** [DIST-5] covers the CAPEX side but there is no **distributed
capacity factor** distinct from the utility-scale 0.20. Rooftop and
constrained-siting yield is materially lower (orientation, shading, no tracking,
roof-area limits). Source a distributed CF and apply it alongside the CAPEX
premium — otherwise the distributed arm gets utility-scale yield at a
distributed cost only, which biases against EaaS, or the reverse if the premium
is understated. A parameter, not a technology.

---

## TECHNOLOGY SCOPE — declared boundary and one proposed addition

**This is a SCOPE decision, not a correction.** Phases 0-6 fix errors; they do
not expand the technology set. After every correction lands, the model still
carries exactly **one non-gas generation technology**. That must be declared as a
boundary in Limitations, because a referee will accept a declared boundary and
will not accept an undeclared one.

### What the LP actually optimises, after all corrections

| Genuinely free | Closed, and by what |
|---|---|
| Public/EaaS split of solar investment — **the thesis contribution** | Solar vs any other non-gas option — solar is the only one in the model |
| Gas dispatch across slices (scarce annual fuel budget allocated seasonally) | Gas capacity expansion — `gas_add` is a fixed `Param` (5.3 frees it, appendix only) |
| Storage investment, in the sliced model | Serve-vs-pay at a real price — VoLL is a penalty, not an alternative supply (2.6) |
| Serve-vs-pay margin, after 2.6 | Build timing — min-build binds 2025, max-build binds 2042-45 (2.5) |
| | Location — no transmission or evacuation constraint (1.4) |

Six degrees of freedom, five closed by modelling choices rather than by Nigeria.
Phases 2.5, 2.6, 2.1/2.2 and 5.3 open four of them. The technology set remains.

**Framing consequence.** "The cost-optimal decarbonisation pathway" overclaims
while the pathway is arithmetically determined. Either narrow the claim to
**"cost-optimal financing pathway for a physically-determined capacity
requirement"** — weaker-sounding, fully defensible, and it matches what the model
does — or open the degrees of freedom and re-test. Recommendation: do both, and
retain the narrower framing regardless, because even afterwards the technology
set is one renewable option deep.

### PROPOSED ADDITION `[OPEN]` `[SHOULD]` — gas fleet efficiency as an investment decision

**Not in the model and not elsewhere in this plan.** The highest-value scope
addition available, and the one that sits squarely inside Petroleum and Energy
Resources Engineering rather than reaching outside it.

Measured fleet efficiency is **η = 0.287** — brutally low, indicating degraded,
part-loaded or simple-cycle-dominated operation. Rehabilitation and
combined-cycle conversion of existing sites has the shortest lead time, the
lowest capital intensity per MWh delivered, and the least new-infrastructure
dependency of any intervention in the Nigerian sector. **A move from 0.287 to
~0.35 releases roughly 20% more electricity from the same gas — no new fuel, no
new pipeline, no new evacuation.** It is arguably a larger lever than any new
build, and the model cannot represent it at all: η is a fixed physical parameter.

Proposed form: η as a **piecewise investment choice** — `existing` at 0.287 and
`rehabilitated` at a sourced value, with capital cost per MW converted and a lead
time. This converts a fixed parameter into a decision that competes directly with
solar for the same capital envelope.

**It sharpens the financing question rather than diluting it.** Rehabilitation is
cheap capital with fast payback; solar is expensive capital with long payback.
That is exactly the tension the EaaS mechanism addresses, so adding this
strengthens the contribution instead of broadening it.

Requires: rehabilitation/CCGT-conversion capex per MW for Nigerian conditions,
achievable post-rehabilitation η, and lead time. Sequence **after 2.6**, and only
if the timeline allows.

### SECOND PRIORITY `[OPEN]` `[COULD]` — demand-side efficiency / DSM

NIRP 2024 carries 3.9 GW of DSM covering 27.7 TWh by 2045; this model carries
none. Given that the solar requirement is `D(2045) − 39.9`, anything reducing `D`
has amplified effect — it is the **only** option that shrinks the residual rather
than filling it. Defensible on NIRP precedent alone. Also closes part of the
storage-comparison caveat in the NIRP critique, where the gap is currently partly
scope rather than resolution.

### DECLINED — with reasons, for the Limitations chapter

| Technology | Why not |
|---|---|
| **Wind** | Nigerian resource is poor and confined to the north-east. Adding it reaches outside the discipline to model a technology with weak local relevance |
| **Biomass** | Same reasoning — weak resource base relative to the modelling cost |
| **WAPP imports/exports** | Commercially real and currently outside the boundary, but a bilateral-trade module is a separate research question |

One well-chosen lever inside the discipline beats four token technologies.
List all three as future work with these reasons stated.

---

## DO-NOT-REPORT LIST — invalidated output `[MUST]` keep current

Screen results against this before any number enters a chapter, a slide or a
supervisor briefing.

| Item | Reason |
|---|---|
| **All of `results/monte_carlo`** | Public arms leave ~284 TWh unserved (~24% of cumulative demand); LCOE ~$3,600/MWh. "P(EaaS reduces cost) = 100%, savings 96%" is the difference between a VoLL-dominated objective and a feasible one — not a valuation of EaaS. Resolved by 2.6 |
| **All `17_` LCOE figures** | Three defects at once: VoLL penalty in the numerator; hydro omitted from the denominator (~30% of generation); storage discharge double-counted (energy charged from solar is already in `solar_generation`). Affects the console table, summary CSV, box plot and `eaas_necessity.json`. Same defect class as 0.3's solar-share item |
| **DEM-2 cells where `public_only` has unserved > 0** | At `constrained_continuation` the budget binds exactly (`capital_utilisation` = 1.0) and cost is dominated by penalty: `downside/public_only/no_policy` moves 13.6bn → 152.7bn with 35.3 TWh unserved. Screen on the unserved column, not by eye |
| **GAS-3 (`08_`)** | `baseline` reports $571B with 152.4 TWh unserved and 20/21 cap-binding years |
| **POL-1 `public_only` (`15_`)** | −$220.75B cost difference on the conditional arm |
| **All pre-recalibration DEM-2** | The interaction conclusion *inverted*: four cells moved from "approximately independent" to "AMPLIFIES super-additive" under recalibration. Qualitative conclusions were unsafe, not merely magnitudes |
| **Anything from `run_reliability.py`** | Its gas-case loop assigned `all_results[g] = df` (the initial sweep) instead of `baseline_df`, so every output it ever produced was wrong. Now suspended |
| **Anything quoted as "48 runs"** | DEM-2 runs **72** (3 demand × 4 gas × 2 financing × 3 policy). `POLICY_CONFIGS` has three arms, not two. Check notebooks and drafts |
| **Notebooks 01–04** | All outputs predate the Phase 1.4 recalibration. Banner added, outputs cleared. `04` was never run to completion |
| **All buffer-year (2046–2054) results** | Extrapolated inputs, non-binding caps, storage retiring in-model with no replacement capex. The buffer exists to hold the horizon edge away from the reporting window; it is not a forecast |
| **`system_cost_npv` as a headline** | Now spans 2025–2054. It is an optimisation objective. Report `npv_gross_capex_report_window_usd` (2025–2045, gross, salvage excluded) as the financing figure, and always name horizon and salvage basis in the same sentence |
| **`nirp_trajectory` solar arm — NPV only** | The NIRP cumulative cap binds 2027–2037 and the shortfall is priced at VoLL. Feasibility and unserved energy from that arm are reportable; cost is not, until 2.6 |
| **Anything quoting the backcast as −4.1%** | Now −3.4% (24.744 vs 25.62 TWh_e) after the min-build floor was removed |

**The structural statement behind most of this:** *any run combining a binding
NDC cap with a binding capital envelope produces a VoLL-dominated objective.*
Fixed by 2.6.

---

## DEFERRED DEFECTS — logged, not yet fixed `[OPEN]`

Each is real, each changes output, each is deliberately held until its phase.

**D1 `[MUST]` Interaction classifier** (`13_run_dem2:237–243`). Two different
thresholds for one concept: `interaction_amplifies` tests `> 0` while
`interpretation` tests `> 1e6`, so any value in between produces a row where the
boolean says `True` and the text says "approximately independent" — and the
console renders the boolean while the CSV carries the text. There is also no
sub-additive branch, so **−$201bn is labelled "approximately independent"**, and
the 1e6 tolerance is absolute in a table spanning 10⁻¹ to 10¹¹. Fix: one relative
tolerance driving both fields, three branches. The sub-additive case is
economically meaningful (under gas `upside`, higher demand *reduces* the scarcity
penalty) and is currently discarded as noise.

**D2 `[MUST]` Infeasibility count** (`17_run_monte_carlo:238`).
`status != "optimal"` counts the bare `except Exception` path as an infeasible
draw, so the headline infeasibility rate means "model infeasible OR code
crashed." Count separately and print the error count.

**D3 `[SHOULD]` `.gitattributes`.** `core.autocrlf` churn produced a
content-free commit early in this work (`1 file changed, 0 insertions, 0
deletions`) that recorded work which did not exist. Defence so far is procedural:
explicit paths in `git add`, and reading `git diff --cached` before every commit.
Permanent fix is `* text=auto eol=lf` plus `git add --renormalize .` — an
enormous diff, so **standalone commit only, never mid-verification**.

**D4 `[SHOULD]` Script inventory.** Three unclassified scripts surfaced during
1.6. Rule: every file in `scripts/` must be **in the verification sweep**,
**suspended with a named successor**, or **documented as a consumer/utility**.
Current suspensions: `12_run_dem1` (deprecated demand base), `14_run_str1`
(algebra), `run_reliability` (wrong dataframe; superseded by 09_/10_/11_),
`04_run_fin2` (deprecated).

**D6 `[OPEN]` `[SHOULD]` Buffer-year storage replacement capex.** Storage life
(15 yr) is shorter than the 30-year model horizon, so buffer-year vintages retire
in-model, are still dispatched, and are never charged replacement capex. The plan
2.5 guard reports it every run. Affects buffer years only. Fix is either a
replacement-capex term or a shorter buffer; neither is urgent while the reporting
window is unaffected.

**D7 `[OPEN]` `[SHOULD]` Solar-build axis missing from the coherence cross-tab.**
`docs/scenario_coherence.md` covers demand × gas. NIRP's solar schedule is
downstream of its gas assumption, so `nirp_trajectory` × constrained gas is a
third instance of the constraint-relief asymmetry and needs declaring.

**D8 `[OPEN]` `[COULD]` `21_compute_resolution_premium.py` basis.** The
annual-vs-sliced premium moved 13.4% → 6.80% when salvage landed (the sliced
model holds more capital, so receives a larger credit). It will have moved again
with the horizon extension. Re-run and record; the ~13% figure is superseded and
already on the do-not-report list.

**D5 `[DONE]` cp1252 glyphs.** Six runners plus `00_build_emissions_cap` crashed
with `UnicodeEncodeError` whenever stdout was **redirected** — they ran fine to a
console, so the fault was invisible until a logged sweep, meaning **those runners
were never covered by any prior verification**. Only four characters were at
fault (`→ Δ ∞ ≤`); em-dash and `×` exist in cp1252 and were never the problem.
`99c_` carried a latent instance in an unreached branch. *Lesson worth keeping:
verification that has never been run under redirection has not been run.*

---

## VERIFICATION PROTOCOL THAT WORKED — reuse it `[MUST]`

The Phase 1.6 rename touched 30 call sites across 21 runners without a single
regression. The method generalises to every future parameter change:

1. **Split the rename from the revalue.** Rename with **values frozen** first:
   outputs must be byte-identical, which is a free exact regression test across
   the whole codebase. Change values second, so every delta is attributable.
   Doing it the other way round makes a rename bug indistinguishable from a value
   effect — permanently.
2. **Capture the pre-image on unmodified code**, via `git checkout` of the parked
   file, not from memory or an older run. Two pre-images in this work were
   invalid: one had a different schema, one was captured after a crash.
3. **Log every runner to its own file and check exit codes**, then
   `grep -il traceback`. A solver "Optimal" line is not a pass.
4. **Design an internal control.** DEM-2's unchanged `organic_central` rows
   sitting between changed `constrained_continuation` rows in the same file, same
   run, same solver, is stronger evidence than any external comparison.
5. **A removed bound may be hiding a defect.** Both the min-build floor and the
   max-build cap were masking artefacts at opposite ends of the horizon. When an
   unsourced constraint comes out, expect the result to change in a way that
   looks wrong — and check whether the constraint was load-bearing for a reason
   nobody documented.
6. **Set a materiality threshold before reading a sensitivity verdict.** The
   salvage script flagged a 480 MW deviation out of 25 GW as "responds to cost".
   Any nonzero-deviation test needs a share-of-total threshold or it overstates.
7. **Read `git diff --cached` before every commit.** This caught a content-free
   commit, a truncated docstring, a doubled bullet and a duplicated function
   definition.

---

## Sourcing ledger — every item blocks a result

| Parameter | Source required | Status |
|---|---|---|
| η (gas fleet efficiency) | NERC generation ÷ gas-to-power volumes | ✅ 0.287 |
| Emission factor basis | IPCC default, HHV-adjusted | ✅ 0.1823 |
| Solar CF | Global Solar Atlas, **named sites** | ⚠ value set, sites unnamed |
| Demand base / metering point | NERC 2024 gross generation | ✅ 37.09 TWh |
| Hydro level | NERC 2024 (30.92% share) | ✅ 11.47 TWh |
| Gas deliverability anchor | derived + NGC volume cross-check | ✅ 89.27 TWh_th |
| **TCN peak wheeling capability** | NERC / System Operator | ❌ blocks 1.4 |
| **System load factor** | NERC peak vs average generation | ❌ blocks 1.4 |
| **Transmission expansion trajectory** | TCN plan + *historical delivery against plan* | ❌ blocks 1.4 |
| **Hydro trajectory** | Zungeru status + NDC 10,400 MW | ❌ blocks 1.5 |
| **Committed solar (DARES/Mission 300)** | programme records | ❌ blocks 1.5 |
| `constrained_continuation` growth | Ember 2010–2025 CAGR; `data/demand/step1_per_capita_diagnostic.xlsx` | ✅ 0.0314 |
| `organic_central` growth | Route A (ε 0.83, GDP 4.5%) + Route B (Kenya 2024) | ✅ 0.040 |
| `organic_high` growth | Route A (ε 0.90, GDP 6.8%) + Route B (Ghana 2015) + NIRP low | ✅ 0.060 |
| NIRP benchmark growth | NIRP 2024 Table 15, Tier 2 scope | ✅ 0.077 |
| Income elasticity ε | Liddle, Smyth & Zhang, *Energy Economics* (middle-income, non-SSA) | ✅ 0.76–0.90 |
| Population growth 2024–45 | UN WPP (232.68m → 336.66m) | ✅ 1.77%/yr |
| λ (Tier 1 ÷ Tier 2) | NIRP 2024 Tables 2, 15; §4.2 | ✅ 0.51–0.57 |
| MC prior σ | SD of the three sourced arms | ✅ 0.0120 |
| **GDP growth arms** | WDI `NY.GDP.MKTP.KD` (pre/post-2015 averages) + IMF WEO | ⚠ 4.5% central declared, history not yet pulled |
| **NIRP 2024 demand-base reconciliation** | NIRP Table 6 vs NERC 2024 — decompose the 37.09 vs 30.5 gap | ❌ blocks the critical review |
| **Sliced-model shape factors** | Global Solar Atlas **monthly** PVOUT, named sites; pre-Zungeru quarterly hydro | ❌ blocks all sliced results |
| **Distributed solar CAPEX premium** | IRENA / Nigerian C&I market | ❌ blocks [DIST-5] |
| **Distributed solar capacity factor** | rooftop/constrained-siting yield vs utility-scale 0.20 | ❌ blocks [DIST-5] |
| Asset life — solar | IRENA (2026), Renewable Power Generation Costs in 2025 | ✅ 30 yr |
| Asset life — storage | NREL ATB, utility-scale battery, UNAUGMENTED | ✅ 15 yr |
| Asset life — gas | NREL ATB, Gas-CC technical life | ✅ 30 yr |
| Solar deployment cap | NIRP 2024 Annex F Table 19 + Table 8 CF; validates to 61.9 GW vs published ~61 GW | ✅ `nirp_solar_cumulative_mw()` |
| **`nirp_accelerated` multiplier (2.0×)** | comparator national ramps (Vietnam 2019-20, Egypt Benban, South Africa REIPPPP) | ❌ declared assumption |
| **Distributed solar CAPEX (EaaS)** | NREL ATB Distributed Commercial PV: premium +16% (2025) rising to +37% (2045), time-varying | ⚠ sourced, not yet implemented — [DIST-5] |
| **Distributed solar CF** | NREL ATB **fixed-tilt** utility vs distributed commercial ratio. Do NOT use the class-5 ratio: utility is single-axis tracking, and the Nigerian 0.20 is already fixed-tilt | ❌ blocks [DIST-5] |
| **Gas rehabilitation capex + post-rehab η** | Nigerian plant rehabilitation / CCGT conversion costs | ❌ blocks the proposed efficiency-investment item |
| Nigeria solar cost adder (utility) | IRENA or Lazard-EMDE | ❌ |
| Peak multiple | feeder-level or genset-usage studies | ❌ blocks 2.4 |
| Genset LCOE + EF | NBS/MEMAN fuel prices | ❌ blocks 2.6 |
| Self-generation energy | "broken grid" self-gen literature | ❌ blocks 2.7 (Tier 2) |
| r_commercial (real USD) | Eurobond curve + sector premium; Nature 2025 CoC | ❌ blocks Phase 3 |
| r_concessional + envelope | **named** facility terms | ❌ blocks 3.2 |
| Guarantee headroom | World Bank PRG / FGN contingent-liability ceiling | ❌ blocks 3.2 |
| Collection rate range | NERC quarterly | ❌ blocks 3.3 |
| B* reconciliation | own derivation, documented | ❌ blocks 3.5 |

---

## Working notes

**Protect 2.1 (time slices) from schedule pressure.** Storage, peak adequacy, the
evacuation bound and solar's true value all depend on it. **`[DONE]`**

**Sequencing: 2.6 (genset backstop) precedes Phase 3.** Not a preference — the
public-vs-EaaS comparison that Phase 3 rests on currently measures a VoLL penalty
parameter rather than the cost of capital. See 2.6.

**Highest-value fix for the contribution: 3.2 (tranche formulation).** Resolves
the average-vs-marginal contradiction, makes marginal CoC an *output*, and
produces the guarantee-headroom dual.

**Do not report any result** from a phase whose upstream dependencies are
unresolved. No FIN number before 3.1 and 3.5; no storage number before 2.1; no
sliced number before the shape factors are sourced; nothing from the
DO-NOT-REPORT list at all.

**Calibration effect on results (Phase 1):** baseline system cost NPV moved
5.98 → 8.99 (η) → 11.28 (CF) → 273 (demand correction exposing the gas error) →
**16.14 bn** (gas re-anchored). The initial specification understated system cost
by ~2.7×, and two errors were partially self-cancelling — undetectable from
internal consistency alone. This is the principal argument for backcasting, and
it belongs in the discussion chapter as well as the methods.

**Demand-arm effect on results (Phase 1.6):** baseline NPV is **13.59bn** under
`constrained_continuation` against **16.14bn** under `organic_central` (−15.8%),
with emissions caps unchanged between them.

**The audit question, applied systematically.** For every parameter, ask: *was
this number generated by the system working, or by the system failing?* Four
catches so far — the 23.08 TWh demand base (collected energy, a revenue
quantity); the peak-truncated load factor; the demand growth rate (a supply
series); and still open, the gas deliverability anchor, which is observed
constrained throughput used as a capability parameter. Levels, growth rates, load
shapes, load factors and collection rates all carry this exposure.

*Frame it honestly in the methods chapter as a stated, systematically applied
protocol — not as a novel contribution.* Suppressed demand and
constrained-versus-unconstrained forecasting are standard in developing-country
utility planning, and claiming novelty invites a referee to produce the prior
literature. More importantly it would dilute the actual novelty claim, which is
endogenous capital structure. **One strong novelty claim beats one strong and one
weak.**

**Three NIRP threats requiring explicit response before submission.**
1. **Gas ceiling ~3.5× below NIRP.** Yours: 89.27 TWh_th × η 0.287 ≈ 25.6 TWh_e.
   NIRP: 22.3 (2024) → 49.9 (2030) → 88.0 (2035) → 64.5 (2045), with 63 GW of
   candidate gas off identified pipeline projects. A scenario disagreement, not
   an error — but load-bearing for every result, since it is precisely what makes
   solar a residual. Needs (a) a documented answer grounded in the delivery
   record of the pipeline programme, gas-to-power arrears and NGC actuals versus
   plan, and (b) a **NIRP-consistent gas arm** added purely as a benchmark. Check
   whether `lag_then_drift` reaches anywhere near 88 TWh_e.
2. **Exogenous hydro at 11.47 TWh vs NIRP's 20.6 GW / 103.9 TWh by 2045.** If
   hydro can expand, solar is not the residual and the headline result changes
   character. The defence is available and good — NIRP leans on the two least
   financeable asset classes in Nigeria, hydro (long lead, $39bn, resettlement,
   security, hydrology) and grid-scale solar IPPs (which have never reached
   financial close) — but it must be made explicitly.
3. **Demand base 37.09 vs NIRP's 30.5 TWh (2024).** Definitional, ~20%, and must
   be reconciled in a footnote **with the arithmetic shown**. Candidate terms:
   basis year (NIRP anchors 2022 observed 32.3 TWh, you anchor 2024); gross at
   busbar vs sent-out (auxiliaries); exports excluded from NIRP but inside NERC
   generation; and the DESG delay assumption. **Note NIRP's own internal
   inconsistency: its 2024 modelled grid demand (30.5) is *below* its 2022
   observed on-grid figure (32.3)** — a forecast starting below its own base
   year. Legitimate criticism, and it means NIRP's early-year figures are a weak
   calibration benchmark.

**NIRP items for the critical review.**
- Table 14's GDP terminal values are inconsistent with the stated growth rates by
  ~1,000× (74.6 at 6.8% for 38 years reaches ~853, not 852,926). Rates are
  coherent; the terminal column has a units error. A documentation defect, not a
  modelling one.
- Annex B assumes universal access by 2030 while the adopted DESG scenario shifts
  electrification to 2035; Table 2 appears re-based, but the M6 forecast was
  built on a different access assumption than the plan it feeds. The 222 TWh
  on-grid (Annex B) vs 300–301 TWh (Tables 2, 6) gap needs the
  self-generation-absorption explanation. **Verify against M6 directly — an
  inference cannot be cited.**
- **The storage opening.** NIRP builds 3.4 GW / 11 GWh of storage, nothing before
  2041, alongside 61 GW of solar at 18% CF supplying 32% of 2045 generation, with
  3.9 GW of DSM covering 27.7 TWh. That combination is not credible in an hourly
  chronological model, and §4.1 concedes granularity was reduced without stating
  to what. You have `build_model_sliced` already verified. If it produces
  materially more storage at comparable solar penetration, you have a **second
  contribution independent of the financing argument** — NIRP under-provisions
  storage because of temporal aggregation, and storage is precisely the asset
  class the public budget cannot absorb, which is the EaaS case. Two caveats:
  word it as "inconsistent with hourly-resolution results, plausibly attributable
  to the documented granularity reduction" — you cannot prove causation without
  their resolution — and declare that you do not model DSM at all, so part of any
  gap is scope rather than resolution.

**For a Q1 target** (*Energy Policy* or *Energy for Sustainable Development*;
*Applied Energy* only if 3.2 is the methodological headline) three things are
still missing: a **scope reconciliation table** (NIRP $63bn NPV / $49.3 LCOE
versus your figure, every difference attributed — nodes, technology set,
transmission, hydro, gas ceiling, discount rate, demand trajectory; not to match,
to demonstrate you know why they differ); a **NIRP-benchmark run** showing your
model reproduces NIRP-like build under NIRP-like gas and hydro, then the
counterfactual with cost of capital endogenised — *that single figure is the
abstract*; and explicit sensitivity arms on the three load-bearing assumptions
above.

---

## IMMEDIATE NEXT ACTIONS

1. **2.6 — genset backstop.** Highest priority. Unblocks Phase 3, makes λ
   endogenous, clears most of the do-not-report list, gives the Access
   dimension, and is the decisive test of the residual finding. Blocked on two
   sourced parameters: **genset LCOE** (Nigerian diesel/petrol self-generation,
   order $0.30–0.55/kWh — NBS fuel prices, MEMAN, Nigerian C&I studies) and
   **genset emission factor** (IPCC defaults will serve).
2. **Re-run `22_salvage_sensitivity.py` after 2.6** — the fourth and final point
   in the scheduled sequence.
3. **2.7 — tiered demand.** Immediately after 2.6, which it depends on. λ =
   0.51–0.57 already sourced from NIRP, giving Tier 2 ≈ 65–73 TWh for 2024.
4. **[DIST-5] distributed CAPEX and CF.** Now sourceable. **Declare as a threat:
   does the EaaS advantage survive a sourced distributed premium?** On an energy
   basis the gap is ~2× against the 1.10 `required_margin` currently standing in
   for it. The [DIST-1..4] offsets have real work to do.
5. **D7 — add the solar-build axis** to `docs/scenario_coherence.md`.
6. **D8 — re-run the resolution premium** on the new horizon.
7. **0.3 / 0.4** — metric formula fixes; 0.3's solar-share defect is the same bug
   as the `17_` LCOE denominator.
8. **3.1** — currency/rate basis memo. One page, and the finance chapter is
   unfalsifiable without it.
9. **Gas fleet efficiency as an investment decision** — after 2.6, timeline
   permitting. See TECHNOLOGY SCOPE.
