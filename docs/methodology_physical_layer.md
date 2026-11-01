# Methodology — physical layer

**Current as of HEAD `8fc5ccb`.** Last substantive model changes:
`5beaef0` (storage-CAPEX realised-spend diagnostic) and `85f2598` (B\*
replacement, concessional rate sourcing, consolidated gas benchmarks,
IRENA solar arm). The operating-and-maintenance implementation
(`d7538ff`), which added +6.602 bn (+30.9%) to the central objective,
remains the last change to the O&M magnitude itself. Figures elsewhere
in this document not explicitly marked otherwise are stated at this
state.

This is a living document describing a model under active correction.
Results produced under it are provisional. The sections below cover only
the **physical layer**: intra-annual demand shape, firm-capacity adequacy,
storage's role, and the representation of the gas fleet (availability,
retirement, new build, cost and efficiency).

**Sections covered here:** demand shape and peak representation; firm
capacity and reserve margin; storage capacity credit; gas fleet baseline,
availability, retirement, new build, capex and new-build efficiency;
reporting horizon; computational environment.

**Sections NOT covered here, and which should be written elsewhere:**
capital structure and the per-tranche capital charge (F1); the EaaS
bankability mechanism and tariff; the emissions cap and NDC scenario
construction; demand *level* projection and the tiered-demand
construction; self-generation and the diesel backstop; the concessional
envelope; solar cost trajectory and land constraints; identification
testing (F3) and shadow-price estimation (F4).

---

## 1. Intra-annual demand shape

**PROVISIONAL**

The sliced formulation represents each modelled year by six time slices —
two seasons (dry, November–April; wet, May–October) crossed with three
diurnal periods (night 22:00–05:00, 7 h; day 05:00–18:00, 13 h; peak
18:00–22:00, 4 h). Demand is allocated across slices by a
`period_demand_factor` and a `season_demand_factor`, each normalised so
the hours-weighted mean is unity.

### What changed and why

The model previously carried period factors of 0.65 / 1.05 / 1.20 and
season factors of 1.08 / 0.92, none of them sourced, and separately
accepted a peak-to-average ratio (PAR) as an independent scenario input
checked for consistency against the slice shape.

That construction was wrong in a way worth recording, because the error is
structural rather than numerical. For a fixed slice partition,

```
PAR_block = f_peak × 8760 / h_peak
```

identically, where `f_peak` is the peak block's share of annual energy and
`h_peak` its hours. PAR is therefore a restatement of the peak block's
energy share, not an independent quantity. Supplying both and reconciling
them permits an inconsistent pair to be declared and then argued about; it
also invites double-counting when the two are swept independently. The
specification was inverted: **each arm now supplies the shape, and PAR is
derived from it and reported, never set.** The consistency check became
unreachable and was deleted.

This is the same discipline applied elsewhere in the model — emission
factors derived from efficiency at build time rather than stored, storage
power derived from energy and duration — and the same error class caught in
the tier-2 reconciliation, where a busbar factor and a tier ratio encoded
one assumption twice.

### Source

Oluwole (2023), *Nigeria National Electricity Demand Timeseries*,
DOI 10.17632/pxvdm26rn7.2 — 8,784 hourly observations for calendar 2016,
carrying **both** an unsuppressed demand reconstruction and the suppressed
series actually served.

Four arms, all DERIVED from that dataset by aggregating hourly values onto
the six-slice partition:

| arm | period factors (night / day / peak) | derived PAR_block |
|---|---|---|
| `peak_suppressed` | 1.0215 / 0.9935 / 0.9834 | 0.9834 |
| `peak_unsuppressed` **(central)** | 0.9056 / 1.0239 / 1.0877 | 1.0877 |
| `peak_nirp` | not set — raises if selected | — |
| `peak_high` | 0.8849 / 1.0005 / 1.2000 | 1.2000 |

`peak_high` is CONSTRUCTED, not measured: the evening factor is pinned at
1.20 — the model's own former unsourced default — with night and day
derived by preserving the unsuppressed night:day ratio and solving for an
hours-weighted mean of unity.

### Why the suppressed series is a declared floor and not a candidate

The suppressed series is a **supply shape, not a demand shape**, and the
data says so directly. Rationing flattens the evening specifically: the
peak factor falls from 1.088 to 0.983 while the night factor *rises* from
0.906 to 1.022. Only 41.9% of demand was served. The suppressed series also
carries a higher *instantaneous* PAR (1.689 vs 1.459) but a lower *block*
PAR (**0.983 vs 1.088**, HEAD-derived — corrected from a stale 0.981 vs
1.085, C-4) — brief supply spikes against a suppressed base.

This is the constraint-relief asymmetry stated as measurement rather than
argument, and it is the same reasoning that rejected historical grid
generation CAGR as a demand-growth proxy. `peak_suppressed` is retained as
a declared lower bound so that the direction of the bias is visible, not as
a case anyone should report as central.

### Why NIRP could not supply this

NIRP 2024 contains **no daily load profile** — no typical-day curve, no
peak-hour timing, no statement of the model's temporal resolution. The only
hourly dataset it names is for hydro seasonality. Its load factor of 0.76
is stipulated, and as printed is internally inconsistent (Table 14 gives
both 71 and 76; the Table 15 energy/capacity pairs return 0.756, 0.759 and
0.760 for 2022, 2045 and 2060, so 0.76 is the applied value). The
`peak_nirp` arm therefore exists and raises if selected: the block factors
it would require cannot be derived from the document.

### Direction of bias

The shape is a 2016 vintage against a 2024 planning start. Only the shape
is taken — levels come from NERC 2024 and the growth arms — and load
shapes are structurally more persistent than levels, being driven by
near-constant equatorial daylight, work and school schedules, and sectoral
mix.

Three post-2016 developments plausibly moved the shape, and all move it the
same way: distributed solar under DARES and the REA programme shifts
daytime load behind the meter; 2023 subsidy removal changed
self-generation economics; 2024 Band A tariff reform reallocated served
supply. Each concentrates grid draw in the evening. **The true 2024 shape
is therefore expected to be more evening-peaked than 2016, so the central
arm understates firm-capacity requirements and understates the cost of a
solar-heavy pathway.** `peak_high` sweeps that direction; nothing sweeps
the other, deliberately.

---

## 2. Firm capacity and adequacy

**PROVISIONAL**

### The omission this corrects

The model previously contained **no firm-capacity, peak or reserve-margin
constraint of any kind.** Adequacy was not under-parameterised; the
mechanism was absent. Because the per-slice energy balance can be satisfied
by the priced self-generation backstop, the model could meet the evening
peak by letting households and businesses run diesel generators rather than
by building capacity. A solar-heavy least-cost pathway was therefore
unfalsifiable: nothing in the specification required installed capacity to
exist at the moment of peak demand.

This matters more in Nigeria than in most systems, because the peak is
after sunset and solar contributes nothing to it.

### Specification

```
Σ_tech  capacity_credit[tech] × capacity[tech, t]
    ≥  peak_mw[t] × within_block_peaking_factor × (1 + reserve_margin[t])
```

Capacity credits:

- **solar 0.0.** Not conservatism. `SOLAR_PERIODS = ("day",)` and the peak
  block is 18:00–22:00, so solar produces nothing at peak by construction.
  This is the number that makes the constraint bite.
- **gas** — the fleet availability factor, discussed in §4.2. The dispatch
  ceiling and the firm-capacity credit were originally a single parameter
  and have been separated so they can be sourced independently; they are
  different quantities (annual energy availability versus the probability a
  unit runs in the peak hour).
- **storage** — a swept arm, §3.
- **hydro** — seasonal, reflecting materially lower dry-season firm
  capacity than nameplate.

### The within-block peaking factor, and why it is separate

With six slices the constraint sizes against the *four-hour block average*,
not the instantaneous system peak. The two differ measurably: for the 2016
unsuppressed series, PAR_instantaneous is 1.4588 against PAR_block **1.0877**
(claims ledger C-4; the historical figure 1.0847 used to derive the ratio
below is stale — `load_scenario()` derives 1.0877 at HEAD), a ratio of
**1.3448** as coded. Sizing on the block average alone would undersize
firm capacity by roughly a third.

That ratio is applied as `within_block_peaking_factor`, and the two are
kept **multiplicatively separate** from the reserve margin so they remain
individually traceable. Merging them would produce a single number no
reader could decompose. **It is not, however, actually re-derived from the
dataset at runtime: both builders read it as a hardcoded literal default
(`scenario.get("within_block_peaking_factor", 1.3448)`), never set by
`load_scenario()` or any arm, and the corresponding code comment claims a
default of 1.0, not 1.3448** (register L50, forensic audit A62). Recomputed
on the HEAD-derived block PAR (1.0877 rather than the stale 1.0847), the
same ratio would be 1.3411 (-0.28%) — the coded 1.3448 has not been
updated to match.

### Reserve margin

Derived from NIRP's own stated requirement rather than assumed: NIRP 2024
§2.3 (p.19) specifies spinning reserve of 900 MW. The margin is computed
per year as `900 / peak_mw[t]`, which declines as the system grows —
appropriate, since a contingency reserve is a fixed unit-outage quantity
rather than a proportion. SOURCED.

The residual simplification is that a deterministic margin stands proxy for
NIRP's probabilistic criterion (loss-of-load expectation of 100 h/yr in
2024 falling to 24 h/yr by 2035). Equivalence is asserted, not
demonstrated.

### Consequence for the model's behaviour

The constraint is slack in 2025 and 2030 and binds from 2035 onward, in
**20** of 30 years at the central configuration (`results/baseline_sliced/diagnostics__02c17f55.json`,
`peak_unsuppressed` central: dual 23,925-80,309 USD/MW over 2035-2054).
Across all 65 `baseline_sliced` diagnostics files the binding-year count
ranges **15-30** (20 years in 30 files, 21 in 26), with the first binding
year 2034 in 26 files and 2035 in 30 — so neither "21" nor "every arm"
holds; 20 is the central figure, not a universal one. Adding the
constraint raised baseline system NPV by roughly a third. That increase is not a correction of an error — it is
the cost of representing a physical requirement the model previously
ignored, and it should be reported as such.

---

## 3. Storage

**PROVISIONAL — magnitude is not an optimisation result**

### Storage was structurally dominated, not merely uneconomic

In the annual formulation the storage energy balance is
`discharge[t] ≤ RTE × charge[t]`, and the demand balance takes
`+ discharge − charge`. Storage's net contribution is therefore at most
`(RTE − 1) × charge ≤ 0`: it can only ever subtract energy while costing
capital. **Optimal storage was identically zero for any cost.** No
parameter sweep could have changed this, which is why the storage-role
experiment was suspended rather than recalibrated.

The sliced formulation is different in principle — storage can move surplus
from a slack slice into a binding one — but without an adequacy constraint
it had no reason to. The firm-capacity constraint of §2 supplied the
missing value channel, and storage built for the first time.

The correction is worth recording as method because it distinguishes a
structural absence from a calibration problem. Sweeping storage costs
against a formulation in which storage is dominated produces a null result
that looks like an economic finding and is not.

### Capacity credit is a swept arm, and the magnitude claim is weaker than
### the composition claim

The credit was initially `min(1.0, duration / peak_block_hours) = 1.0` at
four-hour duration against a four-hour peak block — placing storage
*above* gas as a firm-capacity resource. It is now
`storage_cc_case ∈ {0.50, 0.80 (central), 1.00}`, ASSUMED, with the high
arm reproducing the former flat value as a declared upper bound.

The reasoning: a four-hour battery covers a four-hour block in principle,
but effective load-carrying capability declines steeply with storage
penetration, and the deployment levels this model reaches are not a
low-penetration regime. A flat credit has no declining ELCC and therefore
admits no interior optimum.

**This is a limitation on what can be claimed.** Under a fixed capacity
credit and with gas at its floor, storage absorbs the entire firm-capacity
residual, so its quantity is determined by the constraint rather than
chosen. **The measured signature — build came out at precisely
`77,387.72 / credit` MWh, with duals scaling identically — was measured at
an unrecorded epoch and does not reproduce at the frozen state.** Checked
directly: the frozen central configuration (`storage_cc_central`, credit
0.80) gives storage 65,613.0 MWh (`docs/findings_frozen.md`); the formula
above would require credit ≈ 1.18, a value this project has never used.
Multiple structural changes separate the two measurements — gas
retirement moved from a flat proxy to vintage arms, the hydro seasonal
factor was corrected, the F1 per-tranche capital treatment replaced the
overnight formulation, and O&M was added — and which of them broke the
exact-division signature cannot be established without the original run's
configuration, which was not recorded. No replacement figure is invented
here (the L8 treatment: an honest "not datable" beats a guessed number).
**Storage composition (that it is displaced by gas, and at what ratio) is
a model result; storage magnitude is not** — this qualitative claim does
not depend on the stale figure and is unaffected. This is a specific
instance of a general finding — a constraint can bind exactly and still
be immaterial, or bind exactly and still leave the bound quantity
undetermined as here — stated in full in `docs/methodology_protocols.md`
§5.

`storage_duration_hours` is currently 4.0 and reached through a
`scenario.get(...)` default rather than a required key — an inconsistency
with the surrounding discipline, and load-bearing because it sets the
credit's own ceiling.

---

## 4. The gas fleet

### 4.1 Baseline

**SETTLED**

`gas_baseline_mw = 10,947`. SOURCED: NERC quarterly reports, the 23 gas
plants tracked for availability, identical in the December 2024 and
2026/Q1 tables. AES (294 MW) and Gbarain (120 MW) appear in NERC's licensed
plant list but carry no availability data and are excluded; the 11,361 MW
licensed total reconciles to 10,947 by exclusion, never by scaling, and a
structural guard raises if the two ever diverge.

A relevant composition fact, taken from the turbine-type column of the same
table: **59.2% of the fleet is open cycle** (6,482 MW), 21.7% combined
cycle (2,380 MW), 19.0% gas-steam (2,085 MW) — shares of the 10,947 MW
model fleet, not the 11,361 MW licensed total (on the licensed total the
same categories are 60.7% / 20.9% / 18.4%, which is the figure this
section previously reported; AES and Gbarain's 414 MW inflate the
denominator without a turbine-type breakdown of their own). A
capacity-weighted ISO-condition expectation
across that mix is approximately 0.3375 HHV against a measured fleet
efficiency of 0.287 — so a substantial part of the fleet's low efficiency
is composition rather than degradation. Open cycle is cheap to build and
tolerant of interrupted fuel supply, which is a rational response to the
deliverability constraint this thesis is about. This weighting is
capacity-based and indicative; a generation-weighted figure would differ
and has not been computed.

### 4.2 Availability

**OPEN — carried as bounds, not resolved**

NERC measures gas plant availability at **30.6% (2024), 29.0% (2025/Q4)
and 26.2% (2026/Q1)** against a stable 10,947 MW fleet (corrected TASK 73:
the 2024 figure was written 29.5%; 3,351.16 / 10,947 = 30.6%, from Table 7.1
of the NERC 2024 Annual Report; 29.0% reproduces from the 2025/Q4 quarterly
report's Table 1 on the same convention; 26.2% could not be checked, the
2026/Q1 report not being in the repository). The model's
inherited assumption was 0.80 — roughly 2.7× observed, and worth
approximately 5,700 MW of credited firm capacity in a constraint that binds
in **20** of 30 years at the central configuration (§2).

Neither figure can simply replace the other. **NERC's plant availability
factor conflates mechanical and fuel unavailability, while the model
constrains fuel separately** through the gas balance; substituting the
measured figure would charge the gas shortage twice. The model's 0.80 was
intended as fuel-agnostic mechanical availability, a different quantity
from what NERC reports.

The measured distribution is bimodal, and the low end is not a fuel story:
Azura at 93.7–98.5% and Ikeja at 88.1–100% against Alaoji at 0.01%, Sapele
Steam at 2.7% and Omotosho II at 1.7%. Availability tracks gas-contract
quality at the top and non-operation at the bottom. Excluding every plant
below 20% availability still only lifts the survivors to 41.4%, and
discards 5,101 MW of nameplate to do it — so the shortfall is fleet-wide,
not a handful of stranded assets.

The specification therefore carries arms rather than a value:
`gas_avail_observed` (0.3061, declared lower bound, double-counts fuel; 0.295 until TASK 73),
`gas_avail_central` (not set, raises — intended to represent mechanical
availability only), `gas_avail_nameplate` (0.80, the inherited assumption,
retained as an explicit upper bound). **The central arm is unsourced and
this is the largest open item in the physical layer.**

Direction of bias: results reported under `gas_avail_nameplate` credit the
fleet with firm capacity it demonstrably does not deliver, and therefore
understate the adequacy problem and the cost of solving it.

### 4.3 Retirement

**PROVISIONAL**

The inherited specification retired 680 MW/year from 2035 — a bare literal
described in its own comment as a brownfield proxy, which removes the
entire fleet by 2051 and hands the whole firm-capacity burden to storage.

It is replaced by a vintage-based schedule: a per-plant table of
commissioning years and capacities, with retirement at commissioning year
plus an assumed technical life, exposed as arms at 30 (central), 40 and 50
years. The mechanism accepts a per-year cumulative retirement series rather
than a flat rate, because a vintage schedule is lumpy.

**The commissioning years are ASSUMED — an analyst estimate from general
knowledge of Nigerian project histories, not traced to NERC, NDPHC, TCN or
company filings.** Nigerian commissioning records were not obtainable
within the project timeframe. Confidence is flagged per plant in the source
CSV: 35.8% of capacity high, 47.2% medium, 17.0% low. The largest single
weakness is Afam IV-V (726 MW), where one value blends a 1970s and an
approximately 2002 vintage that should be separate rows. The arms are named
`gas_retire_vintage_*`, never `_sourced`.

A flat technical life is itself an imperfect model for a rehabilitated
fleet: the 40-year case retires Delta in 2025 and Egbin in 2026, both of
which are operating. Two effects run opposite — rehabilitation extends life
beyond nameplate vintage, while under-utilisation extends calendar life
because running hours rather than years drive degradation — and 30 years is
taken as central on the grounds that these partly cancel, **not** on the
grounds that 30 is correct.

Direction of bias: every vintage arm retires substantially less than the
retired proxy (means of **359, 277 and 152** MW/yr over 2035–2054 against 680).
Results reported under the proxy overstate retirement, overstate storage
requirement and overstate system cost.

### 4.4 New build

**PROVISIONAL**

`gas_add` was a fixed `Param` initialised to zero: **the model could not
build a megawatt of new gas.** This is not a defensible simplification for
a model evaluating NDC 3.0 pathways, since the NDC targets approximately
17 GW of gas-fired capacity by 2035 — roughly double current levels and
more than its solar and hydro targets combined. It also meant that, with
solar credited at zero and hydro exogenous, storage was the *only* resource
able to supply an incremental firm megawatt, which determined the storage
result mechanically.

It is now a decision variable, governed by `gas_build_case`:
`gas_none` (fixed at zero, retained so pre-correction results stay
attributable), `gas_unconstrained` (no cap), and `gas_ndc_17gw` (capped on
an NDC-consistent trajectory). **The trajectory is implemented and wired
in both builders**: `scenarios.py:1522`, `gas_ndc_17gw_trajectory_mw()`,
derived from NIRP Tables 19/22 (2030 9,188 / 2035 17,027 / 2041 20,945 /
2045 18,408 MW, nameplate basis, linear interpolation), read by both
`optimize_model_sliced.py:712-720` and `optimize_model.py:524-531`.

New gas is declared **combined cycle**, which biases toward gas appearing
attractive and therefore places the burden of restraint on the
deliverability constraint rather than on an unfavourable technology choice.

A separate `new_gas_deliverability` key governs whether new capacity is
subject to the same fuel constraint as the existing fleet (`constrained`,
the default and the Nigerian baseline) or exempt (representing plants sited
at gas supply or on dedicated commercial contracts). The exempt arm's fuel
allowance is sized at full-year running and is an upper bound rather than a
realistic alternative.

### 4.5 New-build capex

**PROVISIONAL**

The inherited value was 900,000 USD/MW: a bare literal with no source and
no technology attribution. Because `gas_add` was pinned at zero, it had
never affected any result.

The replacement is **DERIVED from NIRP 2024, not stated by it.** NIRP
publishes generation by technology (Table 19), capacity factors (Table 22)
and investment by technology and year (Table 23), but no capex per unit.
Implied capacity follows as `generation / (8760 × capacity factor)`, and
implied capex as cumulative investment divided by cumulative capacity
added: **1,317 USD/kW** over 2030–2045 (17.0 bn USD against 12,905 MW).
Year-by-year values cluster between 877 and 1,393 USD/kW.

The method validates on the technology that can be checked independently:
applied to solar it returns 905 USD/kW (55 bn against 60,756 MW), which is
consistent with published solar cost data. Arms are 900,000 (the retired
literal, as declared lower bound), 1,317,000 (central) and 1,400,000.

A caveat that must accompany the figure: **NIRP does not state whether its
gas investment is open or combined cycle**, so the derived value is applied
to a combined-cycle build without technology confirmation from the source.

An incidental finding from the same derivation: NIRP's implied gas capacity
in 2035 is 17,027 MW — the NDC's 17 GW target to three significant figures.
The NDC's gas commitment appears to be NIRP's modelled trajectory rather
than an independent policy choice. NIRP's capacity series is also derated
at roughly 0.77 of nameplate, so the 17 GW figure must be converted before
it can cap a nameplate-basis model.

### 4.6 New-build efficiency

**PROVISIONAL**

`ccgt_efficiency_case ∈ {0.5010, 0.5173 (central), 0.5445}`, HHV basis.

SOURCED base: EIA / Sargent & Lundy, SL-018001 Rev A (6 December 2023),
Table 5-1, Case 5 (combined cycle 2x2x1, H-class, 1,227 MW net), net heat
rate 6,266 Btu/kWh HHV at 60 F and 60% RH, giving `3412 / 6266 = 0.5445`.
(Corrected TASK 73: this was attributed to the NREL ATB, which does not
contain the figure.) **No LHV-to-HHV conversion is applied**, because the
report states the heat rate on an HHV basis; applying one would double-count
the fuel-heating-value conversion. This matters because the fleet
efficiency of 0.287 was itself derived on an HHV basis at 1,036 Btu/scf,
and mixing bases would corrupt the comparison.

The ambient derate is ASSUMED, not sourced: ATB evaluates at ISO conditions
(15 °C) on a US Midwest design basis, against Nigerian ambient conditions
of roughly 27–35 °C, and both gas turbine output and efficiency fall with
inlet temperature. The central arm applies 5%, the low arm 8%, the high arm
none.

The ratio matters for interpretation: `0.5173 / 0.287 = 1.80`. **One new
combined-cycle megawatt-hour consumes 44% less gas than the existing fleet
average** — a large lever in a system where fuel, not capacity, is the
binding constraint.

---

## 5. Reporting horizon

**SETTLED**

The solve horizon runs 2025–2054; the reporting horizon is **2045**
(`REPORT_END_YEAR`). The extension beyond the reporting window is
deliberate: a vintage built near the horizon edge is not fully depreciated
within it, and under an overnight-capex-plus-salvage treatment such a
vintage is refunded most of its capital and becomes nearly free. Extending
the solve edge relocates that distortion outside the years that are
reported.

2045 is also NIRP's own horizon, so benchmarking against NIRP is only
possible within that window.

Any number quoted from 2046–2054 is invalid as a finding. This rule was
enforced retrospectively during this work: a result initially reported over
2025–2050 was withdrawn on discovering that the two arms it compared were
identical over 2025–2045, and that the entire difference lay in the buffer
years.

---

## 6. Computational environment

**SETTLED**

Python 3.10, Pyomo, CBC and HiGHS, on WSL2 Ubuntu.

Recorded because it bears on reproducibility. **`environment.yml` still
has this problem at HEAD, not "before this was corrected":** it lists
`pypsa` (no module in `src/` or `scripts/` imports it — `grep` empty)
and omits `pyyaml` (`yaml` is imported by `src/io.py` and
`scripts/build_gas_deliverability.py`) and both solvers (`cbc`, `highs`
appear nowhere in the file). An abstract-syntax-tree import scan of the
source tree (excluding `scripts/retired` and `scripts/deferred`) finds
**seven** third-party packages in actual use: matplotlib, numpy, pandas,
pyomo, requests, scipy, yaml — not five. `numpy=1.26.4` is pinned, which
is correct. During the migration described above a Windows
code-integrity policy began blocking unsigned linear-algebra binaries,
forcing migration to WSL2, in the course of which numpy was silently
upgraded across a major version; that specific episode is now closed by
the pin, but the broader claim that the specification was "corrected"
does not hold at HEAD.

**The test suite does not reproduce anything at HEAD.**
`tests/test_run.py` calls `load_scenario(...)` with only
`demand_level_case`, `demand_case`, `gas_deliverability_case`,
`carbon_case`, `start_year`, `end_year`, `om_case` — missing 16 of the
17 required sentinel-defaulted parameters (`tariff_case`,
`peak_multiple_case`, `reserve_margin`, and thirteen more) — so it
raises `TypeError` before a model is ever built, let alone solved. It
also imports `generate_epsilon_pareto`, a live function, but never
reaches it. Regression checking is instead done by
`run_central_sweep.sh`'s `A_REGRESSION_ANCHOR` invocation (§3 of
`methodology_protocols.md`), which does reproduce to the digit.

---

## 7. Open method decisions

1. **Gas availability central arm is unsourced** (§4.2). Mechanical and
   fuel unavailability cannot be separated from NERC's published figure,
   and the two bounds differ by a factor of 2.7. This is the largest open
   item here.
2. **The storage capacity credit is flat** (§3). Without a declining ELCC
   there is no interior optimum, so storage magnitude cannot be reported as
   an optimisation result.
3. **`storage_duration_hours` is a soft default**, not a required key,
   despite setting the storage capacity credit's ceiling.
4. **Retirement commissioning years are assumed** (§4.3) and would be
   replaced wholesale by any traceable Nigerian source.
5. **The NDC-consistent gas build trajectory is implemented and wired**
   (§4.4, `gas_ndc_17gw_trajectory_mw()`, both builders) — no longer
   open. The nameplate/derated basis conversion it required is done in
   that function.
6. **NIRP's gas investment technology is unstated** (§4.5), so a
   combined-cycle capex is applied on a derivation that may reflect a
   different technology mix.
7. **The demand shape is a 2016 vintage** (§1) with a named direction of
   bias but no post-2023 data to test it against.
8. **Deterministic reserve margin stands proxy for a probabilistic LOLE
   criterion** (§2) without demonstrated equivalence.
9. **Rehabilitation of the existing fleet at fixed technology mix is
   absent from the choice set** (§5.2). Conversion (open-cycle to
   combined-cycle) is no longer part of this gap — it is implemented and
   swept, per §5.2 above, and worth 5.91 bn (the second-largest lever in
   the model). Rehabilitation remains genuinely absent: its ceiling is a
   composition-adjusted efficiency expectation derived with unsourced
   indicative efficiencies, declared out of scope pending sourcing. A
   cost-optimality claim over abatement options is therefore incomplete
   with respect to rehabilitation, not conversion.
