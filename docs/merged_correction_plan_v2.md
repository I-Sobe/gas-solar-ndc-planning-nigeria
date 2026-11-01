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

**0.3 `[DONE]` `[MUST]` Metric formula fixes — these are reported numbers.**
- **Public capital multiplier:** currently `total EaaS capex / public spend`,
  counting the subsidy-funded slice as mobilised private capital. Change to
  `(capex − subsidy) / subsidy`; report both definitions.
- **Solar share:** denominator omits hydro and includes `storage_discharge`
  (double-counting). Denominator = gas + solar + hydro (+ genset backstop later).
- **Gas scarcity value:** delete `gas_curtail` (zero objective coefficient →
  degenerate). Report the annual dual λ_t directly. Replace the `max(0.0, val)`
  clamp with an assertion — the clamp would silently zero a wrong sign convention.

**Completed at `142d437` (`docs/metric_fixes_0_3.md`).** All three fixed:
solar share renamed `solar_share_of_grid_generation`, denominator now
gas + converted-gas + solar + hydro with `storage_discharge` excluded;
`gas_scarcity_value_usd` deleted (`m.gas_curtail` itself untouched — only
the diagnostic); public capital multiplier split into
`eaas_multiplier_vs_subsidy` (the old ratio, `None`-guarded) and a new
`eaas_multiplier_vs_public_budget`. Five `max(0.0, val)` dual-sign clamps
replaced with `DUAL_SIGN_TOL`-guarded assertions. *A fifth, previously
undocumented defect surfaced during the fix* (`realised_public_spend`
also omits storage CAPEX) *was flagged but left unfixed — out of scope of
the four named defects.*

**0.4 `[PARTIAL]` `[MUST]` Numerical hygiene.**
Replace the `emissions_cap = 1e18` sentinel with *no constraint*. Delete or
normalise the weighted-sum objective (adds USD ~10¹⁰ to tCO₂ ~10⁸). Rescale to
B$/TWh/GW. Move CBC → HiGHS; pin solver version and tolerances.

**Three of four done at `08b5972` (`docs/numerical_hygiene_0_4.md`).**
(A) `solve_model()` defaults to `tee=False`, pins CBC tolerances to `1e-7`
(checked against the installed CBC 2.10.13 build's own defaults) and
captures `solver_obj.version()` into the result as `solver_version`; no
time limit was added (none to record). (B) The weighted-sum objective
branch and `weight_cost`/`weight_emissions` deleted from both builders —
confirmed zero positional call sites anywhere. (C) The `1e18` sentinel
replaced with `None` at 17 sites (not the 16 originally described); two
runners' `carbon_shadow_usd_per_tco2` CSV column changes from constant
`0.0` to blank, flagged rather than silently absorbed.
**"Move CBC → HiGHS" is NOT done, and cannot be from this environment**:
the `highs` binary is installed (v1.15.1) but the Python bridge
`highspy` is not (`ModuleNotFoundError`), so `SolverFactory("highs")`
and `SolverFactory("appsi_highs")` both raise
`ApplicationError: ... NotFound` — confirmed directly, 2026-09-19.
`solve_model()`'s default solver remains `"cbc"`
(`src/optimize_model.py:1817`). HiGHS tolerances in the doc are set to
published defaults, not independently verified.

---

## PHASE 1 — Re-anchor the physics

**1.1 `[DONE]` Gas efficiency measured, not assumed.**
η = 0.287 from the identity `observed TWh_e ÷ observed TWh_th`, using NERC 2024
generation (25.62 TWh_e gas) and reported gas-to-power volumes (800–860 MMscf/d [no source exists, register L130],
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
- `[DONE]` **Hydro trajectory built.** `hydro_case_scenarios()` (`f43a54c`)
  returns exactly three arms — `hydro_flat`, `hydro_committed`, `hydro_ndc` —
  all held out of the optimiser as exogenous energy series, matching this
  item's *committed-only* / *NDC-aligned* request. Reported as **F-18**:
  under `hydro_ndc` system cost falls 53% against `hydro_flat` (27.954 →
  13.093 bn) — "if hydro can expand to NDC scale, solar is not the
  residual" — defended on deliverability (1,482 MW/yr = 2.1
  Zungeru-equivalents required, against 8–12 yr FID-to-commissioning
  precedent), not on cost.
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

**Step 7 `[DONE]` — cap invariance CONFIRMED, re-verified this task.** Caps rebuilt from a
`constrained_continuation` baseline are **byte-identical** to those from
`organic_central`. Hypothesis confirmed exactly, not approximately: baseline gas
is deliverability-bound from 2026 (2025 already runs 24.57 of 25.62 available),
so `Ebase` is set by the gas trajectory and not by demand growth. **The NDC caps
are robust to any future revision of `g`.** Report as a robustness statement in
the cap chapter. Incidental result: the `constrained_continuation` baseline
objective is **13.59bn**, against 16.14bn for `organic_central` (−15.8%).
**Still true, and the deliverability CSVs feeding it are dated 2026-08-15 and
unchanged — stale in provenance but not in value.** `00_build_emissions_cap.py`
now carries a documented constraint on top of this: its input must be an
annual `conv_none` baseline, because the fleet gas series is only a complete
measure of gas-to-power emissions in the builder where OCGT-to-CCGT conversion
does not exist (`conversion_case='conv_none'`) — under any other
`conversion_case`, part of the fleet's output is re-routed through
`m.convert_generation`, which the cap builder's baseline extraction does not
see. Recorded here since it did not exist when Step 7 was first confirmed.

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

**2.2 `[PARTIAL]` `[MUST]` Storage — currently incapable of adding value.**
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

**Two of four sub-requirements done, two not — this is PARTIAL, not DONE.**
Done: (1) `storage_deployable_hours` no longer a free parameter — falls out
of the slice structure (`src/optimize_model_sliced.py:992`); (2) storage
O&M wired into the objective at `d7538ff`, sourced from NREL ATB 2024
(`io.py`'s `om_cost_by_year["storage"]` now actually read). Not done:
(3) **STR-1 was never rebuilt** — `scripts/14_run_str1_storage_role.py`
still opens `STATUS: SUSPENDED pending Phase 2`, unchanged; (4)
`storage_duration_hours` remains a fixed constant
(`src/scenarios.py:2718`, `_storage_duration_hours = 4.0`), not promoted
to a decision variable — a softer "consider," not a hard requirement, but
unmet regardless.

**2.3 `[DONE]` `[MUST]` Promote peak adequacy into the LP.**
At annual resolution 1 TWh of solar substitutes perfectly for 1 TWh of gas — solar
can "serve" the 8 pm peak. The peak check exists only as a post-solve diagnostic
the optimiser never sees. Add
`firm_capacity(t) ≥ peak_demand(t) × (1 + reserve_margin)`, solar credited 0–10%,
storage at power rating.

**Completed at `d9aaa7d`.** `m.firm_capacity_constraint` exists in
`src/optimize_model.py`, `reserve_margin`-based, and binds **21 of 30
years** under `peak_unsuppressed` (from 2035 onward) — slack and dual
exposed per `docs/firm_capacity_diagnostic.md`.

**2.4 `[DONE]` `[MUST]` Resolve the peak multiple as a declared unknown.**
`scenarios.py` sets 2.5; diagnostics default to 1.82. Beyond the contradiction: a
load factor measured on *served* demand in a load-shedding system is not evidence
about peak shape — shedding truncates the peak, so the observed load-duration
curve is flat-topped by construction. Source from feeder-level or genset-usage
studies, or sweep as a declared unknown.

**Completed at `a2717ac`.** Four sourced arms in
`peak_multiple_scenarios()` — `peak_suppressed`, `peak_unsuppressed`,
`peak_nirp`, `peak_high` — each a `{period_demand_factor,
season_demand_factor}` pair keyed to `peak_multiple_case`. Source:
Oluwole (2023), *Nigeria National Electricity Demand Timeseries*
(DOI 10.17632/pxvdm26rn7.2), 8,784 hourly observations, calendar-2016
vintage against a 2024 planning start — vintage declared, only the
SHAPE is taken (see `docs/limitations_register.md` L1). PAR is now
derived from the selected arm, not hardcoded.

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
  **[CORRECTED, TASK 53 2.10, 2026-10-19: when written, storage did NOT flat-hold from 2045: it was flat from 2025, because `load_storage_capex_by_year` had been disconnected from the builder by `96b50cf` (register L90). The "flat is conservative" argument above is about these buffer years past the source's coverage; it does not cover 2025-2045, where the source series exists and falls 42%. Since TASK 53 2.2 the sentence is true: the 2025-2045 series is read and held at its 2045 value (146,000 USD/MWh) through 2054.]**
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
| Post-2.6 | 0.00000 MW | 0.0000 MWh | Residual SURVIVES the backstop. 12.67% (annual) / 14.73% (sliced) cost swing moved zero MW. |

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

**2.6 `[DONE]` `[MUST]` Replace naked VoLL with a genset backstop.**
Branch `phase-2.6-genset`; baseline tagged `pre-2.6`. `selfgen[t]` / `selfgen[t,s]`
added to the balance in both builders, priced from
`data/cost/processed/genset_cost.csv`. Full state, open docket and traps in
`docs/phase_2_6_handoff.md`.

**Parameters — all sourced.** eta from four Caterpillar PRIME datasheets
(C4.4 / C15 / C18 / 3512). Fuel from NBS monthly AGO/PMS converted at matched
monthly CBN NFEM rates over the Dec-2024..Feb-2026 clean window (AGO 1.0157
USD/l, CV 8.3%; PMS 0.7311, CV 8.7%); Mar-May 2026 excluded as a crude shock
and retained as the high arm.

| arm | eta_HHV | USD/MWh | EF_e tCO2/MWh |
|---|---|---|---|
| `genset_srmc` | 0.3474 (75% load) | 277 | 0.729 |
| `genset_central` | 0.2809 (25% load) | 445 | 0.902 |
| `genset_high` | 0.0511 (A2EI metered) | 3,323 | 4.638 |

`EF_e` and `SFC` are DERIVED at build time from eta, never stored — same rule as
the gas fleet (1.2). `genset_case` is REQUIRED in `load_econ` (no default):
omitting it silently reverted runs to pre-2.6 behaviour, which happened twice in
the headline NDC runners and produced clean-looking tables that were wrong.

**Four defects found and closed during the phase.**
- **D10** cap buffer years 2046-54 were released to 1e18, an arbitrage sink into
  which ~358 TWh of selfgen was routed. Now held flat at the terminal in-window
  cap, matching `io.py::_extend_capex_series`.
- **D11** with genset CO2 inside the cap, `unserved` became a zero-emission
  abatement option — 344.8 TWh shed at 10,000 USD/MWh purely to buy cap headroom.
  `allow_unserved` now defaults False under a genset case.
- **D3** cap boundary. Cap constrains `emissions_grid_only` (the NDC commitment
  boundary); the carbon PRICE sees total. Diesel is disciplined by the grid
  service obligation, not by cap headroom, because Nigeria has no carbon price.
- **D15** `leakage_diagnostic()` on every real-cap runner; a flagged run is not
  an NDC-compliant plan.

**Reportable findings (verification artefacts until the full correction closes).**
- **Leakage.** Under the correct NDC boundary the model complies while emitting
  1.24x / 1.54x / 7.89x the entire cumulative power-sector budget in uncounted
  diesel (srmc / central / high). Widening the boundary is INFEASIBLE in all
  three arms: NDC3 unconditional is unattainable under a tight envelope once
  self-generation is counted. All four NDC cases in `02_run_ndc_caps` flag at
  53-67% leakage.
- **Carbon price is a transfer, not an abatement instrument, when capital binds.**
  `public_tight` emissions are constant at 903.1 Mt across 0-150 USD/tCO2 while
  cost rises 78.4 -> 104.3 bn. `eaas_tight` responds sharply (389.5 -> 167.3 Mt
  between 15 and 20 USD/tCO2). Pricing carbon without capital does nothing.
- **Cost decomposition reconciles.** real + selfgen + VoLL - salvage =
  objective exactly in all five cases. EaaS spends MORE public money (16.3 vs
  10.5 bn) and Nigerians spend FAR LESS private diesel (31.5 vs 75.6 bn). The
  saving is a transfer from private diesel to public capital, not the
  disappearance of a penalty parameter.
- **Residual finding SURVIVES 2.6.** `22_salvage_sensitivity` re-run: solar and
  storage identical across all six lifetime arms against a 12.67% (annual) /
  14.73% (sliced) cost swing. The backstop is a SUPPLY channel, not a CAPITAL
  channel — it relieves the balance but does not change what the budget buys.
  Capacity is an accounting residual as a property of Nigeria's constraint
  structure, not of the formulation.
- **D9 obligation trajectory — front-loading is CHEAPER, not dearer.** Three
  trajectories at constant 0.45 horizon-average, all feasible:

  | trajectory | obj bn | selfgen TWh | solar MW | built early |
  |---|---|---|---|---|
  | front 0.40->0.50 | 90.45 | 639.6 | 25,964 | 48.7% |
  | flat 0.45 | 92.54 | 631.4 | 29,265 | 28.0% |
  | back 0.50->0.40 | 94.69 | 627.1 | 32,566 | 12.3% |

  **RETRACTS** the earlier 2.6E claim that a tighter near-term obligation gives
  worse lifetime service. That compared different LEVELS at the feasibility
  boundary; this compares TIMINGS at constant level inside the feasible band.
  Deferring build means building against larger demand — back-loading needs
  6.6 GW more for the same standard, outweighing the capex decline. Trade-off:
  front-loading is cheaper for the system but leaves marginally MORE
  self-generation (639.6 vs 627.1 TWh).

**Declared conventions.**
- Busbar-equivalence factor for `selfgen`: **1.0 (central, declared
  conservative)**; `1.0753` carried as a sensitivity. Verified: crediting the
  avoided TLF cuts selfgen 647.9 -> 586.2 TWh (9.5%) and leakage 1.54x -> 1.39x.
  Understates the genset and therefore overstates required grid build.
- Grid service obligation renamed from "reliability" (F0):
  `max_offgrid_fraction` / `grid_service_mode` / `m.grid_service_obligation`. It
  bounds GRID SERVICE SHORTFALL (`selfgen + unserved`), not adequacy — a
  household on its own genset was not grid-served but no capacity was short, so
  LOLE/EENS do not describe it.
- Mode is **`annual`**, not `total`. `09b` demonstrated `total` gives identical
  shortfall at eps 0.20/0.15/0.10/0.08 under a rising deficit — it is satisfied
  by early-years slack and stops discriminating between service standards.
- Both builders accept a **scalar or a per-year list** for
  `max_offgrid_fraction` (D9).
- `unserved` is near zero BY CONSTRUCTION once a backstop exists. Never quote it
  as evidence demand is met. Quote `selfgen + unserved`.

**Deferred out of 2.6, filed to their proper phases.**
- **D17 → FIN-2.** The EaaS arm has no binding constraint at the default tariff
  (65M USD/TWh): it builds until demand is served and returns the same answer to
  every input. Every "EaaS saves X" figure is constrained-vs-unconstrained until
  `04_run_fin2_tariff_sweep.py` (a stub, D25) imposes a real tariff constraint.
- **D21, D23, D24 → REL cleanup.** `11_run_rel3` runs but its threshold logic
  disagrees with its own `status` column; `eps_levels` is dropped at
  `optimize_experiments.py:774` so the sweep silently uses
  `reliability_levels_log()`; and the REL suite now sweeps four different eps
  axes across six scripts. Needs one documented rationale — eps is a POLICY
  instrument with no external measurement, so the axis is justified
  methodologically (it must bracket the model's own feasibility boundary), with
  Nigeria's observed lambda 0.51-0.57 as an indicative anchor.
- **D13 → carbon policy.** `19_run_carbon_price_sweep` arms are too weak to
  bite; the tipping point is the result.
- **D14 → reporting.** APERC layer: selfgen TWh / CO2 / spend / shortfall
  fraction as a metrics table over existing runs. No new suite. Motivated by the
  observation that the model has NO instrument pricing diesel's health,
  accessibility or acceptability externalities.
- **D25.** `04_run_fin2_tariff_sweep` and `12_run_dem1_demand_sensitivity` raise
  `NotImplementedError` — unwritten runners, not 2.6 damage.
  **Correction (task_audit_fixes_p3, A51):** this no longer holds for
  `04_run_fin2_tariff_sweep.py`. It was built out starting `b496d77`
  ("FIN-2: multi-blend/multi-NDC tariff sweep") and repaired for its
  CLI-arm surface in `ecb7e54` — confirmed directly: the file carries no
  `NotImplementedError`, no suspension guard of any kind, and its
  `main()` fully builds, solves, and writes real output
  (`results/fin2/*`, dated 2026-08-23). `12_run_dem1_demand_sensitivity`
  is unaffected by this correction — confirmed still genuinely raising
  `NotImplementedError` as its first executable line.

**Executability baseline.** `scripts/smoke_all.sh` runs every numbered runner and
reports pass/fail. Current: **29 pass, 2 fail** (both D25 stubs). Re-run after any
change touching `src/`.

**2.7 `[OPEN]` `[MUST]` Tiered demand architecture.**
- **Tier 1** — grid generation, 37.09 TWh. Permanent: the calibration anchor and
  the basis for emissions-cap construction. Also the conservative "no access
  expansion" counterfactual.
- **Tier 2** — latent connected demand: end-use consumption including
  stolen/unmetered/unbilled energy (~30–32 TWh from the loss ladder), **plus** the
  self-generation economy. **λ = 0.51–0.57 is now introduced from NIRP 2024**
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

**3.1 `[PARTIAL]` `[MUST]` Currency and rate basis memo — do this FIRST.**
ATB CAPEX is real USD. The 18% commercial anchor is almost certainly nominal
naira-adjacent (Nigerian inflation 20–35%; 18% *real USD* would be extreme). The 6%
concessional is presumably nominal USD DFI lending. Blending a nominal-NGN rate
with a nominal-USD rate into one WACC applied to real-USD cashflows is
**dimensionally incoherent** — the bankability frontier becomes an artefact of unit
mixing. Declare the model real-USD throughout; convert every rate explicitly
(Fisher + documented FX assumption, or source USD-denominated Nigerian power
financing directly). One page; without it the finance chapter is unfalsifiable.

**Not DONE — the memo's own status line says so.** `docs/currency_and_rate_basis.md`
states: *"Status: `[DONE]` for the currency convention. `[OPEN]` for the
deflator base year and the Fisher conversion of the financing rates."*
The convention itself (real USD throughout, month-matched conversion) is
declared and evidenced (§1–§4). §5 (real base year for the fuel anchor
vs. the ATB vintage) and §6 (placing the 18%/6% `blended_finance_scenarios()`
rates on a basis) are both still `[OPEN]` in the memo's own text.
**Note:** `capital_recovery_scenarios()` (the `f1_tranche` path, F1 Step 2)
separately sources concessional/commercial rates and deflates them once
via `scenario['us_inflation']` at the CRF() call site — in substance this
may satisfy §6's route 1 for the tranches that actually feed the LP, but
the memo text has not been updated to say so, so §6 is marked here as
genuinely open pending that reconciliation, not rounded up.

**3.2 `[PARTIAL]` `[MUST]` Tranche formulation — replaces the blended scalar.**
**TWO of the three tranches ship — register L23.** `capital_treatment_case=
'f1_tranche'` prices `CRF(r_conc, L_conc)·conc_vol + CRF(r_comm, L_comm)·comm_vol`.
There is no guaranteed-commercial tranche in any form (not present-and-inert,
not fixed at zero) — blocked on sourcing (deployable PRG headroom / FGN
contingent-liability ceiling, neither published at the needed granularity).
Consequences per L23: the guarantee-headroom dual does not exist; the capital
merit order has two steps where the design has three; credit enhancement
cannot be evaluated as a policy instrument. Direction of bias: omitting the
middle tranche makes the commercial tranche **more expensive than reality**,
overstating the value of the concessional envelope relative to a world with
guarantees. This is PARTIAL, not DONE.
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

**3.5 `[PARTIAL]` `[MUST]` Freeze placeholders; scope the EaaS claim honestly.**
**Do not round up to DONE — the item's own sourcing ledger still lists
"B\* reconciliation ... ❌ blocks 3.5" (this file, Sourcing ledger,
below).** What IS done: the model state itself is frozen for
reproducibility at HEAD `2a9226b` (a trivial TASK.md-untracking commit
that happened to be HEAD at freeze time, not itself the commit that did
freezing work), and both regression anchors reproduce there —
**21,811,042,472.821114 / 45,502.9754 MWh** (`om_none`); central
**27,954,000,000 / 65,612.985 MWh** (`docs/frozen_state_2026_09_10.md`).
That is reproducibility-freezing, a narrower thing than this item's actual
asks: derive and document B\* (the "6.13bn docstring vs 9.104bn coded"
framing was wrong: 6.13bn was never a coded value, register L45 clause 2;
B\* is now derived at 10,273,343,610 USD, `src/scenarios.py`, L45 —
addressed pending re-run, ledger row still open), recompute the tariff grid against the
new T\*(t, blend), and source every "PLACEHOLDER pending primary source"
rate. None of those three is confirmed done. Scope note on `EaaS.py`
never being called by the LP, and [DIST-4], stand as originally written
below.
Derive and document B\* (6.13bn was never coded; see L45). Recompute the tariff grid against
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

**4.1 `[DONE]` `[MUST]` Own the cap apportionment openly.**
**Declared at register L24.** The apportionment rule is explicitly stated
as "your methodological construction, not something in NDC 3.0" (L24,
echoing this item's own wording) with sensitivity now added via 4.2's
`ndc3_conditional_decline` arm. L24 also states plainly the second,
stacked construction this creates for the post-2035 span (ten of
twenty-one reported years, plus the whole buffer, carry two analyst
choices, not one) — declared, not hidden. The "add one alternative"
sub-ask is satisfied by a different route than literally specified:
`scripts/00_build_emissions_cap.py` implements NDC 2.0 (BAU-relative:
% below baseline) alongside NDC 3.0 (proportional top-down) as two
distinct apportionment methodologies, not the grid-EF-based or
IEA-scenario-based alternative the item's text names — noted, not
treated as a gap, since the substance (a declared, sensitivity-tested
methodological choice rather than an asserted NDC number) is met.
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

**4.2 `[DONE]` `[MUST]` Cap trajectory beyond 2035.**
Caps held flat 2035–2045 — an implicit "climate ambition stops" assumption across the
final decade. Add a third trajectory: linear decline from the 2035 cap toward
net-zero-2060, and report cap-shape sensitivity.

**Completed — `ndc3_conditional_decline` arm, `scripts/00_build_emissions_cap.py`,
reported as F-21.** Identical to `ndc3_conditional` through 2035, then
declining linearly toward Nigeria's own 2060 net-zero target. **F-21 is
the strongest — and most counter-intuitive — policy result in the model:**
tightening the cap 40% by 2045 (10.039 → 6.024 MtCO2) *raises* national
emissions 7.7% (486.7 → 524.2 Mt) and costs 10.24bn, because the cap
regulates grid emissions only while demand is exogenous — the displaced
load reappears as diesel self-generation at 1.49 tonnes per tonne
abated on the grid. Cap-shape sensitivity is therefore reported, per the
item's ask.

**4.3 `[SHOULD]` Unified fiscal envelope.**
Replace the free-standing `public_solar_budget_npv` with
`Σ DF_t·(public_capex_t + eaas_subsidy_t + tariff_shortfall_t) ≤ F`, where
`tariff_shortfall_t = max(0, requirement_t − allowed_revenue_t)` enters as
non-negative slack. Captures the core mechanism — solar/EaaS shifts energy from
FX-exposed pass-through gas costs toward fixed-capex service payments, shrinking the
shortfall and freeing fiscal space — and makes the subsidise-consumption vs
finance-transition trade-off a **dual rather than a paragraph**.

**4.4 `[DONE]` as an analytical bound. Tariff ceilings from avoided cost.**
Bound the EaaS tariff above by the segment's avoided cost: genset LCOE for
self-generation substitution. Originally cited an **assumed** $0.35–0.55/kWh;
2.6 has since **sourced** the genset figures, so the bound can be closed
with no code and no solve.

**Sourced inputs (2.6, verified against `docs/phase_2_6_handoff.md` and
`docs/merged_correction_plan_v2.md`:614-615 — not re-derived):**

    genset_central   445 USD/MWh = 0.445 USD/kWh   (eta 0.2809 at 25% load,
                      four Caterpillar PRIME datasheets; fuel from NBS
                      AGO at matched CBN NFEM rates, Dec-2024..Feb-2026)
    genset_srmc      277 USD/MWh = 0.277 USD/kWh   (75% load)
    genset_high    3,323 USD/MWh                    (A2EI metered)

    tariff_central    95 USD/MWh = 0.095 USD/kWh    (`docs/tariff_arm.md`)
    tariff_high      115 USD/MWh                    (`docs/tariff_arm.md`)

**Headroom (verified by division):**

    genset_central / tariff_central = 445 / 95  = 4.68x
    genset_srmc    / tariff_central = 277 / 95  = 2.92x   (most conservative
                                                            sourced arm)
    genset_srmc    / tariff_high    = 277 / 115 = 2.41x

**Conclusion.** The EaaS tariff sits **2.4x to 4.7x below the sourced
avoided cost of the self-generation it displaces**, across every tariff arm
and both defensible genset arms.

**Scope limit — state plainly wherever this bound is cited.** This is a
WILLINGNESS-TO-PAY bound for the self-generation-substitution segment, not
a demonstration that any particular customer will pay it, and it says
nothing about **collection**: `disco_collection_rate` is an unused stub
(item 3.3), and [DIST-3] (counterparty differentiation, above) notes that
collection risk applies to grid-served revenue, not behind-the-meter EaaS —
but that distinction is asserted, not modelled; no collection mechanism
exists in the LP for either channel. `docs/limitations_register.md`
currently has no entry naming this gap explicitly (checked: no match for
"collection" or "DIST-3" in that file) — flagged in
`docs/task_doc_closures.md` as encountered but not in scope to add here.

**`genset_high` excluded from the headroom claim.** 3,323 USD/MWh is the
A2EI-metered outlier arm (eta 0.0511) — unrepresentative of ordinary
self-generation operation, not a second defensible bound.

**4.5 `[STRETCH]` Full band-disaggregated fiscal waterfall and segment-indexed EaaS.**
Band A / B–E demand split with MYTO tariffs and NERC collection data;
`solar_eaas_add[t,s]` with per-segment counterparty tiers feeding the tranche
structure; fixed-point iteration where r_commercial responds to sector cash-flow
metrics. **Genuinely Q1-grade and genuinely a second thesis.** 4.3 and 4.4 capture
most of the value.

---

## PHASE 5 — Robustness and reproducibility

**5.1 `[SUPERSEDED-in-part]` `[MUST]` Rebuild the Monte Carlo around the uncertainties that dominate.**
**`17_run_monte_carlo.py` was NOT repaired — it is still dead code** (it
calls `load_scenario()` without the required post-2.4 keys and raises
`TypeError` before any solve; register L10). Instead
`scripts/29_run_parameter_uncertainty.py` was built: Sobol variance
decomposition over five *continuous* parameters (gas availability +
capacity credit sampled together, gas η, solar CF, an O&M blend
fraction, conversion capex), reported as **M-4**. It is a genuinely
different instrument — PARAMETER uncertainty (a fixed-but-unknown
value), not SCENARIO uncertainty (which discrete world happens) — and
its own docstring says so explicitly.
**What it closes from the sub-items below:** the "still not sampled"
list shrinks by one — `solar_cf` is now sampled (`U(0.16, 0.20)`,
Global Solar Atlas, register L26). It also reports a failure
count/fraction as a headline (`n_failed`, `failure_fraction`, with a
warn threshold) — a genuine answer to the infeasible-draw-headline ask,
for parameter draws.
**What it does NOT close:** the **independence-sampling objection**
(demand growth vs. gas regime drawn independently) is sidestepped, not
fixed — `29_` does not sample either axis at all, instead treating
scenario uncertainty via the discrete, coherence-vetted
CONSTRAINT-RELIEF ASYMMETRY cross-tab. That is a defensible design
choice but leaves no Monte Carlo mechanism that respects the
correlation, should one ever be wanted. The **asymmetric-prior
objection** (a symmetric normal on a demand-growth prior whose sourced
arms are not symmetric) is untouched — `29_` never samples demand
growth. **Still not sampled:** λ, the solar/storage CAPEX trajectory,
FX/rates, collection rate.
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

**5.3 `[SUPERSEDED]` `[SHOULD]` Demonstrate, don't assert, that new gas stays at zero.**
~~`gas_add` fixed at zero with the comment "structurally non-optimal (verified)."~~
With 7,480 MW retiring by 2045, high-growth demand and an upside gas case, that is
scenario-dependent. Run one appendix experiment with `gas_add` free across the
scenario grid and show it stays ~0.

**The premise is now false, which makes the item moot rather than done.**
`gas_add` is a free `pyo.Var` (`src/optimize_model.py:471`, "2.9, Step 1"),
not fixed at zero — the "structurally non-optimal (verified)" comment this
item quotes describes pre-2.9 code. With `gas_add` free across the actual
scenario grid, capacity **FALLS** from the 10,947 MW starting fleet
(F-14, `docs/findings_frozen.md`) — the demonstration this item asked for
has effectively already run, as a side effect of every real solve, not as
a dedicated appendix experiment. **Do not mark `[DONE]`** — no dedicated
appendix experiment sweeping the specific high-growth/upside-gas corner
this item names was located, so if a referee wants that exact cell
demonstrated it is still not on file; but the concern the item raises
(does `gas_add=0` hold only because it is fixed?) no longer applies to
the current model. Note also: the live text carried no `[OPEN]` bracket
before this edit, which was itself a discrepancy — TASK.md's own table
had assumed one.

**5.4 `[PARTIAL]` `[MUST]` Verification protocol.**
Dual verification by perturbation for **every** reported shadow price (gas, budget,
concessional, guarantee, evacuation, reliability): re-solve with RHS + δ, assert
ΔNPV/δ ≈ dual within tolerance. One reusable function, run in CI. Unit tests with
hand-computable toy cases for the multiplier, solar share and scarcity rent. Repro
pack: pinned environment (**Python 3.10.19, `nigeria-energy`**), pinned solver and
tolerances, data-provenance manifest for every CSV, one `make all`.

**Half done.** The two verification harnesses were repaired to
three-outcome PASS / NEEDS-ARGS / FAIL reporting, and a regression-anchor
discipline is in force (the two frozen anchors checked throughout this
and prior tasks). **The pinned repro pack is not satisfied.** Python
3.10.19 / `nigeria-energy` is pinned; CBC's tolerances are pinned
(`08b5972`, confirmed against the installed build); but "pinned solver"
cannot extend to HiGHS since it is wired but not installed
(`highspy` absent, 0.4 above), and no data-provenance manifest or
`make all` target was located this task. Mark PARTIAL: the outcome
discipline is done, the repro-pack half is not.

**5.5 `[SUPERSEDED]` `[MUST]` Doc–code parity sweep.**
Every mismatch is a viva question answered on the back foot. Known items:
`storage.py` implements √η-symmetric losses — the LP applies full η on discharge
only. `EaaS.py`'s CRF/tenor machinery is never called. `economics.solar_capex`
divides by `n_years` and calls it annualisation. `apply_access_adjustment` is unused.
`storage_charge_limit_years` is a dead diagnostic on a removed constraint. DEM-1's
docstring carries pre-η numbers (η=0.43, 17.5 TWh_e cap, "solar covers 24%").
**Delete, implement, or reconcile — no third state.**

**Superseded by the three-phase forensic audit** (`docs/forensic_audit.md`,
A1–A59, 59 findings; Phase 1 = `src/`, Phase 2 = scripts/harnesses/docs,
Phase 3 = previously-unread scripts) — a far larger and more systematic
sweep than this item specified. **Not verified this task** whether each
of the six named items above is individually subsumed by a specific
A-numbered finding or still stands independently; that cross-reference is
outside this task's read budget and is flagged as encountered but not
checked (Part 4). 8 of the 59 findings are fixed, 51 remain open — the
audit itself is a superset, not a closure, of this item's scope.

---

## PHASE 6 — Framing and writing

**6.1 `[DONE]` Gas module reframed.** Arps decline-curve docstring removed.
Deliverability is midstream and commercial (pipeline capacity, vandalism, DSO,
pricing, the payment chain), not reservoir depletion — Nigeria holds 200+ TCF proved
reserves, and a "physical decline" story contradicts the Decade of Gas programme.
*Outstanding:* present the trajectories explicitly as infrastructure-and-allocation
scenarios anchored to NERC constraint data and midstream capacity.

**6.2 `[DONE]` Rename "reliability" to what it is.**
An annual unserved-energy fraction is not reliability — no LOLE, no LOLP, no
hourly basis. This item originally prescribed renaming it to **"energy
adequacy."** That prescription was itself wrong: "energy adequacy" *is*
reliability under another name (LOLE, LOLP, expected unserved energy) —
substituting one synonym for the other would not have fixed the defect this
item named.

**What was actually implemented (2.6).** The constraint was renamed to
**"grid service obligation"** (`max_offgrid_fraction` / `grid_service_mode` /
`m.grid_service_obligation`). It bounds `selfgen + unserved` — grid service
**shortfall** — not capacity adequacy. A household running its own genset was
not served by the grid, but no capacity was short; LOLE/EENS describe a
capacity shortfall, not a household's choice of backstop. "Energy adequacy"
would have carried the same LOLE/EENS mismatch into the renamed term that
"reliability" carried into the original one.

**Where the reasoning lives.** `docs/methodology_genself_backstop.md` §5.1
("What it is, and what it is not"): *"Reliability in power-system planning is
adequacy... A household running its own generator was not served by the
grid, but no capacity was short. The constraint is a service obligation."*

**Search performed (step 3 of this item's re-screen).** Within this plan,
"energy adequacy" occurred exactly once — inside this item's own original
body, now rewritten above. No other occurrence in this file.

A tree-wide search (outside this plan, reported here since it bears
directly on this item but the file is outside this task's edit whitelist)
found one further occurrence: `docs/handover_pre_om.md:157`, "Rename
reliability -> energy adequacy. Model side is done; the REL runners and
prose still carry it." That document is superseded — see
`frozen_state_2026_09_10.md`, "Supersedes `handover_pre_om.md`... read it
for the stratigraphy, not for numbers" — so it is flagged, not corrected in
place.

**6.3 `[OPEN]` `[MUST]` Discussion / Limitations chapter — material assembled, chapter unwritten.**
**Do not mark DONE or PARTIAL on the strength of the material that exists.**
`docs/limitations_register.md` now holds 44 entries and
`docs/forensic_audit.md` 59 findings (8 fixed, 51 open) — the raw content
this chapter would draw on is assembled and far exceeds what the item
originally scoped. But the chapter itself — the liquidity chain, the
Electricity Act 2023 / [DIST-4] defence, the land-vs-security siting
point, the north–south spatial mismatch, Q1×4 annualisation, perfect
foresight — has not been drafted into thesis prose anywhere in this
repository. Marker stays `[OPEN]`; the note is to prevent a future
reader inferring completion from the register's size.
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

## DISTRIBUTED EaaS — implement as a set `[PARTIAL]` `[MUST]`

Accepted structural decision. **The exemptions favour EaaS; [DIST-5] offsets them.
Implementing only the favourable half makes the EaaS result an artefact.**
**Marked per sub-item below, not as a block — one of five is DONE, the
other four are not; do not read the whole section as DONE or as OPEN.**

**[DIST-1] `[OPEN]` Evacuation exemption.** EaaS generation is exempt from the wheeling
bound; gas, hydro and public solar are subject to it. Consequence: when transmission
binds, distributed solar gains value **independently of financing** — EaaS relieves
two constraints at once (capital and evacuation). This is a novel channel and maps
directly onto the Deliverability dimension. *Folds into 1.4* — unimplemented, since
1.4's evacuation bound sub-item remains `[OPEN][BLOCKED]`.

**[DIST-2] `[OPEN]` Loss advantage.** Behind-the-meter generation avoids the 7% TLF and
distribution losses entirely — one MWh generated on site displaces more than one MWh
of grid generation. Represent explicitly or distributed solar is understated.
*Folds into 1.4 / 2.7.* **Confirmed still unimplemented** — 2.6's own declared
conventions state plainly: "remains unimplemented because the metering point is
declared at the busbar" (register L25). Direction: this omission biases AGAINST
EaaS, since crediting the avoided TLF would only help it.

**[DIST-3] `[OPEN]` Counterparty differentiation.** Behind-the-meter EaaS sells to a
creditworthy C&I customer, **not** to a DisCo. The 60–75% collection-rate risk
applies to grid-served revenue only. This is why C&I captive solar is bankable in
Nigeria while grid-scale IPP solar struggles — the payment chain differs. **Largest
single effect on FIN-2.** *Folds into 3.3* — unimplemented; `disco_collection_rate`
remains an unused 1.0 stub (`src/scenarios.py:3149`), no counterparty split exists.

**[DIST-4] `[OPEN]` Defend the characterisation.** Cite the Electricity Act 2023
eligible-customer and captive provisions plus Nigerian C&I market evidence. Note
that the centralised single-tariff abstraction is one policy generation behind.
*Folds into 3.5 and 6.3* — both still open/unwritten. The citation itself already
appears repeatedly in this plan's own framing (header, this section), but a formal
defence written into the thesis text does not yet exist.

**[DIST-5] `[DONE]` Distributed CAPEX premium — the offsetting cost.** Distributed and C&I
solar costs materially more per MW than utility-scale (less economy of scale, smaller
BOS, per-site development). The current EaaS/public CAPEX parity was defensible only
while both were utility-scale. **Source a distributed premium and apply it.** Also
remove EaaS from the land constraint (rooftop/captive uses existing sites).
*New item; gates the whole [DIST] set.*

**Completed — both CAPEX and CF, per register L25.** CAPEX premium
sourced from NREL ATB (+16% 2025 rising to +37% 2045). Capacity factor
sourced from Global Solar Atlas (retrieved 2026-09-05), Kano coordinates
held constant for siting: utility-scale ground-mounted 1,723.7 kWh/kWp
vs. distributed commercial at a representative 30° roof pitch,
off-south azimuth, 1,485.7–1,491.3 kWh/kWp (−13.5% to −13.8%) — CF
0.19 → 0.164 (ratio 0.8635). The "Gap identified" paragraph below is now
**stale**: the distributed CF this item's own gap-note asked for has
since been sourced and applied. Net effect (register L25): three
corrections now bear on the EaaS channel — CAPEX, CF, and a flattened
seasonal profile — and **all three disfavour it**. Only unimplemented
DIST-2 would favour it. The model is therefore now biased AGAINST its
own headline mechanism, bounded below by the unimplemented TLF credit.

**Gap identified (STALE — see [DIST-5] completion note above):** [DIST-5] covers the CAPEX side but there is no **distributed
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
| **All of `results/monte_carlo`** | **STILL BARRED — reason changed.** Originally: public arms leave ~284 TWh unserved (~24% of cumulative demand), LCOE ~$3,600/MWh — VoLL-dominated objective, "Resolved by 2.6." 2.6 is now done, but the producing runner, `17_run_monte_carlo`, is dead code (register L10): it calls `load_scenario()` without the required post-2.4 keys and raises `TypeError` before any solve, and carries two further declared defects even if it were wired. No output can currently be produced from this runner at all. Bar it for that. The 284 TWh / $3,600 figures above predate both the freeze and 2.6 and are illustrative only |
| **All `17_` LCOE figures** | STILL BARRED, same reason — three defects at once: VoLL penalty in the numerator; hydro omitted from the denominator (~30% of generation); storage discharge double-counted. Same defect class as 0.3's solar-share item. Also covered by the row above: `17_run_monte_carlo` is dead (L10) and cannot currently produce LCOE output either |
| **DEM-2 cells where `public_only` has unserved > 0** | STILL BARRED, same reason, reason text still accurate — screen on the unserved column, not by eye; a budget-binding `public_only` cell still routes shortfall to a VoLL-priced `unserved` variable whenever the grid-service-obligation bound is exceeded, and 35.3 TWh unserved against ~37 TWh Tier-1 demand is far past any obligation eps used in this project, so 2.6's genset backstop would not have absorbed it. **Epoch note:** the illustrative figures (`downside/public_only/no_policy` 13.6bn → 152.7bn, 35.3 TWh unserved) predate `2a9226b` and predate 2.6 — do not quote them against the frozen state; they illustrate the screening rule only |
| **GAS-3 (`08_`)** | STILL BARRED, same reason. **Epoch note:** `baseline` $571B / 152.4 TWh unserved / 20-21 cap-binding years predates `2a9226b` by a wide margin — $571B is far above even the pre-calibration-defect era (5.98–273bn, see "Calibration effect on results" below), so this figure is oldest-epoch and must not be quoted against the frozen state under any label |
| **POL-1 `public_only` (`15_`)** | STILL BARRED, same reason. **Epoch note:** −$220.75B predates `2a9226b` — do not quote against the frozen state |
| **All pre-recalibration DEM-2** | STILL BARRED, same reason — self-dated already ("pre-recalibration"); the qualitative-conclusion-inversion finding is a historical fact about that recalibration event, unaffected by anything since |
| **Anything from `run_reliability.py`** | STILL BARRED, same reason — wrong-dataframe bug, independent of 2.6 and of the freeze. Suspended per D4 |
| **Anything quoted as "48 runs"** | STILL BARRED, same reason — DEM-2 runs **72** (3 demand × 4 gas × 2 financing × 3 policy); unaffected by 2.6 or the freeze |
| **Notebooks 01–04** | STILL BARRED, same reason — predate the Phase 1.4 recalibration, and a fortiori predate `2a9226b`. `04` was never run to completion |
| **All buffer-year (2046–2054) results** | **STILL BARRED — reason changed.** Extrapolated inputs and non-binding caps still apply — the buffer exists to hold the horizon edge away from the reporting window; it is not a forecast. The third original reason, "storage retiring in-model with no replacement capex," is **now false**: register L18 records D6 CLOSED — both the `pre_f1_overnight` branch (since 2.6) and the `f1_tranche` branch (since F2) correctly charge a rebuild at every in-horizon `build_t + k*life`, storage included. Removed from the reason; the bar itself stands on the two remaining grounds |
| **`system_cost_npv` as a headline** | STILL BARRED, same reason — still spans 2025–2054, still an optimisation objective. Report `npv_gross_capex_report_window_usd` (2025–2045, gross, salvage excluded) instead, and always name horizon and salvage basis in the same sentence |
| **Anything quoting the backcast as −4.1%** | **STILL BARRED — figure changed again.** −4.1% → −3.4% was correct after the min-build floor was removed, but at the frozen central configuration the backcast now measures **−3.2%** against observed 2024 gas generation (`frozen_state_2026_09_10.md` §1). Neither −4.1% nor −3.4% is current; only −3.2% (central, `2a9226b`) is |
| **Pre-O&M objectives quoted as current** *(added — Part 2 screen)* | Every $ objective or NPV figure dated before `d7538ff` (the O&M implementation commit) is superseded — O&M moved every objective by ~31% (post-O&M central **27.954 bn** vs pre-O&M/post-shape-sourcing 21.352 bn; `frozen_state_2026_09_10.md` §1). This includes `handover_pre_om.md`'s anchors (20.9899 bn, 20.856 bn — themselves already flagged there as unreproducible on any commit in this repository's history) and every figure in `findings_provisional_pre_om.md`. Use `findings_frozen.md` / `frozen_state_2026_09_10.md` for current numbers; never carry a pre-`d7538ff` dollar figure into a chapter without an explicit epoch label |
| **F-11's leakage percentages (34.2–45.0%)** *(added — Part 2 screen)* | The `leakage` field is null in the current cap summaries at the frozen state, so the pre-O&M 34.2–45.0% figures (`findings_provisional_pre_om.md` F-11) are **not restated** and must not be quoted against the frozen state. The structural claim — the cap is grid-only by construction, so a binding cap displaces load to unmeasured self-generation emissions — still stands and is reportable; the magnitude is not, until re-measured. See `findings_frozen.md` F-11 |
| **"37 arms across four sections"** *(added — Part 2 screen)* | The sweep is `bash scripts/run_central_sweep.sh [A\|B\|C]` — **there is no Section D**. 38 invocations total (A: 5, B: 25, C: 8). Any document citing "37 arms" or "four sections" (e.g. `handover_pre_om.md:57`, superseded) is wrong on both counts. See `frozen_state_2026_09_10.md` §2 |
| **`results/fin2/tariff_sweep_results__dares_only__df1.csv`, `results/fin2/tariff_sweep_results__unbounded__df1.csv`, and their paired `tariff_sweep_summary__*.json`/`tariff_sweep_results__*.json` files** *(added — Phase 3 A53, task_audit_fixes_p3)* | STILL BARRED. All dated 2026-08-23 13:44, five days BEFORE the concessional-envelope retirement (`588a734`, 2026-08-28, register L33). Their `concessional_utilisation`/`concessional_shadow_usd_per_usd` columns carry real, non-null values computed by a mechanism (`m.concessional_envelope_constraint`) the current builders no longer construct — a fresh run of `04_run_fin2_tariff_sweep.py` today would produce all-null values for the same columns instead, with no version marker distinguishing old from new. Do not cite either column from these specific files; regenerate first |

**The structural statement behind most of this:** *any run combining a binding
NDC cap with a binding capital envelope produces a VoLL-dominated objective.*
**2.6 is now done** and replaced the uncosted VoLL cliff with a diesel-priced
genset backstop in both builders (`docs/phase_2_6_handoff.md`) — this is a
mechanism-level fix, not a blanket clearance of every row above. A row lifts
only where the specific shortfall is shown to sit within the
`grid_service_obligation` bound. Confirmed for one row — see **Lifted**,
below. Not confirmed for GAS-3, POL-1, or the DEM-2 row above: their
historical shortfalls (152.4 TWh; 35.3 TWh against ~37 TWh Tier-1 demand) are
large enough that a residual VoLL-priced `unserved` beyond any plausible
obligation bound remains likely. They stay barred pending a fresh solve at
the frozen state, which this task does not perform.

### Lifted

Rows move here, not out, when their barring condition is satisfied — the
record of what was once unreportable is itself evidence of the audit
protocol.

| Item | Barred because | Lifted 2026-09-07 because |
|---|---|---|
| **`nirp_trajectory` solar arm — NPV** | The NIRP cumulative solar cap binds 2027–2037 (cumulative shortfall peaking at 4,505 MW in 2032) and the shortfall was priced at VoLL — "cost is not [reportable], until 2.6." | Phase 2.6 is done: `selfgen` was added to both builders as a priced backstop, "replacing an uncosted VoLL cliff with a diesel-priced slope" (`docs/phase_2_6_handoff.md`) — the exact mechanism this row's condition named. **Caveat, carried forward rather than dropped:** this is a documentation judgment from the 2.6 architecture, not a fresh solve (this task runs none). The arm pairs with constrained gas as the "pessimistic stress corner" (this plan, "Coherence caveat," above). Before quoting an NPV from this arm, confirm at solve time that `selfgen + unserved` stays within the `grid_service_obligation` bound in every binding year 2027–2037; if it does not, a VoLL-priced residual remains and the figure needs the caveat this row used to carry. Feasibility and unserved energy from this arm were already reportable and remain so |

---

## DEFERRED DEFECTS — logged, not yet fixed `[OPEN]`

Each is real, each changes output, each is deliberately held until its phase.

**D1 `[DONE]` Interaction classifier** (`13_run_dem2:237–243`). Two different
thresholds for one concept: `interaction_amplifies` tests `> 0` while
`interpretation` tests `> 1e6`, so any value in between produces a row where the
boolean says `True` and the text says "approximately independent" — and the
console renders the boolean while the CSV carries the text. There is also no
sub-additive branch, so **−$201bn is labelled "approximately independent"**, and
the 1e6 tolerance is absolute in a table spanning 10⁻¹ to 10¹¹. Fix: one relative
tolerance driving both fields, three branches. The sub-additive case is
economically meaningful (under gas `upside`, higher demand *reduces* the scarcity
penalty) and is currently discarded as noise.

**Closed at `9c422e8`** (`docs/task_defect_closure.md`, Part 1) — confirmed
directly in `scripts/13_run_dem2_growth_gas_interaction.py:240-292`: one
classification (`reference_magnitude = max(|low_prem|, |high_prem|)`,
`REL_TOL = 1%`) now drives both `interaction_amplifies` and
`interpretation` across three branches (super-additive / sub-additive /
approximately independent). The same commit also closed D17, D21, D23
and D24 (all named only in prose elsewhere in this file, under 2.6's
"Deferred out of 2.6" and this D-list) — not individually re-verified
this task; see Part 4's closing list. **Flagged, not fixed (out of this
task's whitelist):** `docs/task_defect_closure.md` itself contains a
"For the limitations register" section evidently meant to be pasted into
`docs/limitations_register.md`, and it was not — the register carries no
mention of D1/D17/D21/D23/D24 or of `task_defect_closure.md` anywhere.

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
~~`04_run_fin2` (deprecated)~~.

**Correction — `04_run_fin2` is stale in this list, per this same file's own
2.6/D25 text above.** It was built out starting `b496d77` ("FIN-2:
multi-blend/multi-NDC tariff sweep") and repaired for its CLI-arm surface
in `ecb7e54` (task_audit_fixes_p3, A51) — it carries no
`NotImplementedError` or suspension guard and its `main()` fully builds,
solves and writes real output. `12_run_dem1_demand_sensitivity` is
unaffected by this correction — confirmed still raising
`NotImplementedError` as its first executable line. This item's own list
had not been updated to match its sibling section; now consistent.

**D6 `[DONE]` `[SHOULD]` Buffer-year storage replacement capex.** ~~Storage life
(15 yr) is shorter than the 30-year model horizon, so buffer-year vintages retire
in-model, are still dispatched, and are never charged replacement capex.~~ The plan
2.5 guard reports it every run. Affects buffer years only. Fix is either a
replacement-capex term or a shorter buffer; neither is urgent while the reporting
window is unaffected.

**Closed — register L18. This section had gone stale against this file's own
DO-NOT-REPORT list, which already recorded the closure ("register L18
records D6 CLOSED") while this entry still read `[OPEN]` — a genuine
internal inconsistency, now reconciled.** L18 shows the guard's original
fire condition is algebraically identical to the builder's own
`_rebuild_years()` yielding at least one in-horizon rebuild year, for
every lifetime value this project actually supplies anywhere — so no
vintage exists where the guard's condition holds and no rebuild is
charged. Both the `pre_f1_overnight` and `f1_tranche` branches correctly
charge a rebuild at every in-horizon `build_t + k*life`, storage
included. The guard was re-worded to test the invariant directly and
print an informational note rather than a now-false warning. Direction
of bias: none — a documentation/reporting defect, not a modelling one;
no constraint, variable or cost expression was touched.

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
| Gas deliverability anchor | derived + NGC volume cross-check [TASK 77: the cross-check range has no source, register L130] | ✅ 89.27 TWh_th |
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
| NIRP 2024 demand-base reconciliation | NIRP Table 6 vs NERC 2024 | ✅ F-25 — definitional, four terms; NIRP metering point unstated |
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
| Genset LCOE + EF | Caterpillar PRIME datasheets (C4.4/C15/C18/3512); NBS AGO/PMS + CBN NFEM; IPCC 2006 V2 Ch1 T1.2 / Ch2 T2.2-2.5; NMDPRA densities | ✅ sourced, 2.6 `[DONE]` |
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

1. ~~**2.6 — genset backstop.**~~ **`[DONE]`** All parameters sourced (Caterpillar
   PRIME datasheets; NBS/NFEM fuel series). Four defects found and closed
   (D3/D10/D11/D15). Full state in `docs/phase_2_6_handoff.md`. D17/D21/D23/D24
   filed to FIN-2 and REL cleanup respectively.
2. ~~**Re-run `22_salvage_sensitivity.py` after 2.6**~~ **`[DONE]`** — residual
   finding SURVIVES. Solar and storage identical across all six lifetime arms
   against a 12.67%/14.73% cost swing. The backstop is a supply channel, not a
   capital channel. Capacity is an accounting residual as a property of
   Nigeria's constraint structure, not of the formulation — the strong version
   of the claim.
3. **2.7 — tiered demand.** Immediately after 2.6, which it depends on. λ =
   0.51–0.57 already sourced from NIRP, giving Tier 2 ≈ 65–73 TWh for 2024.
4. ~~**[DIST-5] distributed CAPEX and CF.**~~ **`[DONE]`** — register L25:
   both sourced and applied (NREL ATB CAPEX premium; Global Solar Atlas
   CF, 0.19 → 0.164). All three corrections now bearing on EaaS disfavour
   it; the [DIST-1..4] offsets (still open) are what would push back.
5. **D7 — add the solar-build axis** to `docs/scenario_coherence.md`. Still open.
6. **D8 — re-run the resolution premium** on the new horizon. Still open.
7. ~~**0.3 / 0.4** — metric formula fixes~~ **0.3 `[DONE]` (`142d437`); 0.4
   `[PARTIAL]` (`08b5972`; HiGHS move blocked on `highspy`, not installed).**
8. **3.1** — currency/rate basis memo. **`[PARTIAL]`**: the currency
   convention itself is declared and done; the memo's own §5/§6 (real
   base year, financing-rate basis) remain open per its own status line.
9. **Gas fleet efficiency as an investment decision** — after 2.6, timeline
   permitting. See TECHNOLOGY SCOPE.
