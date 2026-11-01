# §3.3 inputs — gas deliverability, conversion, hydro (read-only extraction)

Read-only task. Docstrings and comments below are quoted verbatim from
the files named; nothing is paraphrased. Any field with no recorded
source is marked `NOT RECORDED` per `docs/parameter_provenance.md` Part
A — never inferred. No file was edited, no runner or solve was
executed. Numeric trajectories for Part A are read directly from the
already-generated `data/gas/processed/gas_available_power_annual_twh_th.csv`
(built by `scripts/build_gas_deliverability.py` from
`data/gas/processed/gas_deliverability_cases.yaml`), not recomputed.

---

## PART A — gas deliverability

### Where the A_t series is built

Three files, chained:

1. **`data/gas/processed/gas_deliverability_cases.yaml`** — declares the
   anchor (year, value) and each case's functional form and parameters.
2. **`scripts/build_gas_deliverability.py`** — reads the yaml, evaluates
   each case's series over the model horizon, auto-computes the three
   `flat_*` level-equivalents, and writes the CSV.
3. **`data/gas/processed/gas_available_power_annual_twh_th.csv`** — the
   materialised output, columns `year, scenario, gas_available_twh_th`.
4. **`src/gas_supply.py`**, function `gas_available_power()` — loads the
   CSV at model-build time for one named scenario and returns the
   `years` / `available_twh_th` arrays consumed by the deliverability
   constraint.

Construction docstring, `src/gas_supply.py:8-21`, quoted in full:

```
Load annual gas deliverability to the power sector (TWh_th/year).

Expected CSV columns:
    year, scenario, gas_available_twh_th

Returns
-------
dict
    {
        "years": np.ndarray,
        "available_twh_th": np.ndarray  # TWh_th/year
    }
```

The function itself (`gas_available_power`, `src/gas_supply.py:7-68`)
raises `FileNotFoundError` if the CSV is absent, `ValueError` if the
required columns are missing, and `ValueError` if any year in
`[start_year, end_year]` has no row for the requested `scenario_name` —
no silent fill.

The scenario-label dict consumed by `load_scenario()`,
`gas_deliverability_scenarios()`, `src/scenarios.py:858-884`, docstring
quoted in full:

```
Gas deliverability-to-power scenario labels.

Must match the 'scenario' column in:
data/gas/processed/gas_available_power_annual_twh_th.csv

Structural (all RQs): downside, baseline, upside, shock_recovery
Level-equivalent flat controls (GAS-3 only): flat_downside, flat_upside,
  flat_shock_recovery -- identical cumulative supply, no shape variation,
  to isolate SHAPE vs LEVEL effects.
```

That dict actually returns **ten** keys (the docstring names seven; the
GMP-family trio is documented only in the adjacent inline comment, not
the docstring): `downside`, `baseline`, `upside`, `shock_recovery`,
`flat_downside`, `flat_upside`, `flat_shock_recovery`, `gas_deliv_flat`,
`gas_deliv_gmp_partial`, `gas_deliv_gmp_full`. The **seven** the task
asks for are the seven declared under `cases:` in the yaml (the three
`flat_*` are not declared cases — they are auto-derived by the build
script, see Q3 below).

### The anchor (applies to every arm)

`gas_deliverability_cases.yaml:1-13`, comment quoted in full:

```
Re-anchored to observed gas-to-power delivery.
Derived: observed 2024 gas generation 25.62 TWh_e (NERC: 37,093.70 GWh
total, 30.92% hydro) / measured eta 0.287 = 89.27 TWh_th.
Cross-check: 89.27 TWh_th = ~805 MMscf/d at HHV 1036 Btu/scf, at the low
end of the sourced NGC range (800-860 MMscf/d). Three independent sources
(NERC generation, NGC volumes, efficiency identity) close to within ~1%.
[TASK 77 A.1, 2026-10-23: the quotation above is the yaml comment as it stood. NO SOURCE exists for the
"NGC range 800-860 MMscf/d" (register L130); "sourced" is withdrawn. 805 MMscf/d is 89.27 TWh_th converted
back, not an observation, and the three "independent" sources are one (task_log_76 Part A).]
2024 observation used as the 2025 anchor: gas delivery was broadly stable
between the two years (2024 total gen 37.09 TWh vs 2025 39.19 TWh).
Previous value 40.66 was calibrated against the old eta=0.43 and the
collected-energy demand base; it understated delivery by ~2.2x.
```

Anchor: **2025 = 89.27 TWh_th**, every arm. Classification per
`docs/parameter_provenance.md` Part A: **SOURCED**
[TASK 77: relabelled **DERIVED** from one sourced quantity (NERC 2024 generation) and an unsourced efficiency; the
"NGC volume cross-check" has no source, register L130] — value field
(89.27), source (NERC 2024 generation + NGC volume cross-check),
locator (the derivation arithmetic quoted above, in-file), quantity
(gas delivered to the power sector, TWh_th), and an internal fit check
(three independent routes agree to ~1%) are all present. No formal
"range" or external confidence interval is stated (the ~1% agreement is
a cross-check, not a published interval).

### The seven arms

| arm | represents | 2025 | 2030 | 2035 | 2040 | 2045 | 2054 | classification |
|---|---|---|---|---|---|---|---|---|
| `downside` | `lag_then_decline`, lag 2 yr then −3%/yr | 89.27 | 81.474319 | 69.96477 | 60.081128 | 51.593709 | 39.22314 | **ASSUMED** — no source cited for lag/decline-rate in the yaml or build script |
| `baseline` | `drift`, +0.5%/yr from the anchor, no lag | 89.27 | 91.524179 | 93.83528 | 96.204738 | 98.634028 | 103.162374 | **ASSUMED** — no source cited for the drift rate |
| `upside` | `lag_then_drift`, lag 2 yr then +2%/yr | 89.27 | 94.734038 | 104.594033 | 115.480264 | 127.499543 | 152.373756 | **ASSUMED** — no source cited for lag/drift rate |
| `shock_recovery` | `shock_recovery` — flat to 2029, −3%/yr 2030-2033, ramps 2034-2038 to a level 5% above the anchor, held thereafter | 89.27 | 86.5919 | 84.911453 | 93.7335 | 93.7335 | 93.7335 | **ASSUMED** — no source cited for any of the shock/recovery parameters (baseline_end, shock years, shock rate, recovery ramp/uplift) |
| `gas_deliv_flat` | `lag_ramp_uplift`, I=0 — anchor held flat, no midstream expansion (declared lower bound / control arm) | 89.27 | 89.27 | 89.27 | 89.27 | 89.27 | 89.27 | **DECLARED**, not sourced to an external figure — it is I=0 by construction (a control, not a measured trajectory) |
| `gas_deliv_gmp_partial` | `lag_ramp_uplift`, I=0.50 — CENTRAL of the GMP policy-target family | 89.27 | 140.51098 | 140.51098 | 140.51098 | 140.51098 | 140.51098 | **SOURCED for I** (NMDPRA DSO compliance 50.66%, 2020) but the shape (linear ramp 2025-2030, held flat after) is DECLARED — see Q1 below |
| `gas_deliv_gmp_full` | `lag_ramp_uplift`, I=1.0 — GMP targets fully met (declared upper bound, not a candidate central case) | 89.27 | 191.75196 | 191.75196 | 191.75196 | 191.75196 | 191.75196 | **SOURCED for the target chain** (NNPC Gas Master Plan 2026 p.3, NUPRC 2025) at I=1; the I=1 assumption itself ("targets fully met") is DECLARED, not sourced |

All seven ramps hold flat from 2030 onward for the `lag_ramp_uplift`
family (`gas_deliv_flat` / `gmp_partial` / `gmp_full`) because
`ramp_years=5` from `lag_years=0` at the 2025 anchor completes in 2030;
the `2030/2035/2040/2045/2054` columns are identical by construction
for those three, confirmed directly from the CSV (not assumed from the
formula).

### Q1 — what `gmp_partial` and `gmp_full` stand for, and what `I` means

`gas_deliverability_cases.yaml:40-76`, comment block quoted in full:

```
── GMP 2026 policy-target family ────────────────────────────────
DISTINCT EPISTEMIC OBJECT from downside/baseline/upside/shock_recovery
above, which extrapolate historical drift. These discount a POLICY
TARGET. Never present a result mixing the two families without saying
which it came from.

Chain (SOURCED, NNPC Gas Master Plan 2026 p.3 + NUPRC 2025):
  national production   7.45 (2025, NUPRC) -> 10 (2027) -> 12 (2030) Bcf/d
  monetisation rate     0.60 (current) -> 0.75 (2027) -> 0.80 (2030)
  domestic share of commercialised gas   0.43  [DERIVED: GMP 2.0/4.7;
                                         NUPRC cross-check 2.08/4.75]
  power share of domestic gas            0.481 [DERIVED: GMP power
                                         demand 1.0 Bcf/d / NUPRC
                                         domestic sales 2.08 Bcf/d.
                                         NOTE basis mismatch: numerator
                                         is DEMAND, denominator SALES.]
  -> chain ratio vs 2025:  2027 1.678x,  2030 2.148x

LEVEL IS NOT TAKEN FROM THE CHAIN. GMP's 1.0 Bcf/d power figure is
DEMAND; the anchor (89.27 TWh_th = 0.8055 Bcf/d) is observed DELIVERY.
The 24% gap between them IS the deliverability shortfall this model
exists to represent. The chain supplies SHAPE ONLY, as a multiplier on
the observed anchor.

I = implementation factor, applied to the planned INCREMENT, not the
level. Same basis the 2008 NGMP retrospective uses for reserve growth
(23.16/70 = 33%, not 203/250 = 81%).
  uplift_fraction = I x (2.148 - 1) = I x 1.148

DECLARED SIMPLIFICATIONS:
 (a) Linear ramp 2025-2030. The GMP trajectory is front-loaded (the
     2027 jump is larger than the 2027-2030 step), so linear
     undershoots 2027: 1.459x against the chain's 1.678x at I=1.
 (b) Held flat after 2030. The GMP states no target beyond 2030;
     extrapolating a 5-year policy target over 24 further years would
     be indefensible.
```

So: `I` is an "implementation factor" applied to the GMP chain's
planned *increment* over the anchor (not the level) — `gas_deliv_flat`
is I=0 (`uplift_fraction=0.0`), `gas_deliv_gmp_partial` is I=0.50
(`uplift_fraction=0.574`, inline comment: *"CENTRAL. [SOURCE: NMDPRA
Domestic Supply Obligation compliance, 50.66% in 2020.]"*), and
`gas_deliv_gmp_full` is I=1.0 (`uplift_fraction=1.148`, inline comment:
*"GMP targets fully met. DECLARED UPPER BOUND, not a candidate central
case."*). `0.574 = I x 1.148 = 0.50 x 1.148` — confirmed arithmetically
against the file's own stated formula.

### Q3 — what the `flat_` prefix changes

Two distinct uses of "flat" exist in this system and must not be
conflated:

1. **`gas_deliv_flat`** (a declared case in the yaml, `I=0`) — the
   deliverability *level* itself never grows; it is the GMP family's
   control arm.
2. **`flat_downside` / `flat_upside` / `flat_shock_recovery`** — NOT
   declared in the yaml at all. They are auto-computed by
   `scripts/build_gas_deliverability.py`, function `main()`,
   lines 148-166, comment quoted in full:

```
── Auto-compute flat level-equivalents for GAS-3 ─────────
```

   and the docstring's own description of the pair
   (`gas_deliverability_scenarios()`, `src/scenarios.py:866-868`):

```
Level-equivalent flat controls (GAS-3 only): flat_downside, flat_upside,
  flat_shock_recovery -- identical cumulative supply, no shape variation,
  to isolate SHAPE vs LEVEL effects.
```

   Mechanically: for each of `downside`, `upside`, `shock_recovery`,
   the script takes the mean of that scenario's own annual values over
   the *reporting* window only (`year <= REPORT_END_YEAR`; a comment at
   `build_gas_deliverability.py:154-156` states *"Averaging across the
   buffer years would change every flat level-equivalent and silently
   shift the GAS-3 shape-premium basis"*), and holds that single mean
   value flat across the full horizon. So the `flat_` prefix here
   changes **shape only, holding cumulative reporting-window supply
   equal to the named structural scenario** — a different operation
   from `gas_deliv_flat`'s `I=0`, which changes the *level* (no GMP
   uplift at all) while keeping the anchor-drift shape implicit in
   `lag_ramp_uplift` at zero. These are two unrelated "flat" concepts
   sharing a prefix; the code comments distinguish them but the naming
   does not.

### Q4 — relationship between deliverability and NIRP's gas trajectory

**No explicit trajectory relationship is stated anywhere in the
deliverability code (`gas_supply.py`, `build_gas_deliverability.py`,
the yaml, or `gas_deliverability_scenarios()`).** NIRP is named exactly
once in the deliverability yaml, in the `gas_deliv_gmp_partial` comment
(`gas_deliverability_cases.yaml:94`), and there only to explain what
the model *excludes*, not a trajectory link:

```
this model takes demand exogenously from NERC/NIRP, so demand
failing to materialise is not the constraint being modelled.
```

NIRP does appear elsewhere in `src/scenarios.py`, but for **gas CAPEX**
(`gas_capex_central`, `src/scenarios.py:1789-1809`), not gas
deliverability — a separate parameter answering a separate question
(cost per MW of new gas build, not TWh_th available to the fleet). The
docstring there (`src/scenarios.py:1789-1809`) states gas CAPEX is
*"DERIVED from NIRP"* via *"1. Capacity implied per NIRP gas-build
year = generation [...]"* — this is the CAPEX derivation, not a
deliverability constraint, and should not be read as linking the two.
**Answer: NOT RECORDED / no relationship stated** for deliverability
specifically.

### Q5 — traceable to NNPC or NGC, or all analyst constructions?

Mixed, arm by arm:

- **The anchor (89.27, every arm's 2025 starting point)** — traces to
  NERC (2024 generation) with an NGC volume cross-check (800-860
  MMscf/d range), quoted in full above. ~~SOURCED~~ [TASK 77 A.1: DERIVED; the 800-860 range has no source,
  register L130].
- **`downside` / `baseline` / `upside` / `shock_recovery`** — no NNPC
  or NGC citation anywhere in the yaml or build script for their
  parameters (decline/drift rates, shock timing and magnitude, recovery
  ramp). These are **analyst constructions** — the yaml's own comment
  block for the GMP family explicitly contrasts itself against these
  four as the ones that merely *"extrapolate historical drift"*, i.e.
  a functional-form choice, not a cited trajectory.
- **`gas_deliv_flat`** — I=0 by construction; not a data point, so not
  applicable to trace.
- **`gas_deliv_gmp_partial` / `gas_deliv_gmp_full`** — the *chain*
  (production, monetisation rate, domestic share, power share) traces
  to **NNPC Gas Master Plan 2026 p.3 and NUPRC 2025** data, per the
  comment quoted under Q1. The *level* those two arms actually run at
  is NOT taken from that chain (stated explicitly: *"LEVEL IS NOT TAKEN
  FROM THE CHAIN"*) — it is the observed NERC/NGC anchor multiplied by
  a chain-derived shape. So even these two are only **partly**
  NNPC/NUPRC-traceable: the shape multiplier is sourced, the base level
  is the same NERC/NGC anchor as every other arm, and the
  implementation factor `I` (0.50 for `gmp_partial`) is sourced to a
  *different* metric (NMDPRA DSO compliance) than the chain itself.

---

## PART B — conversion tiers

Two independent three-way structures exist under the name "conversion
tier" in this codebase and the task's phrasing blends them. Reporting
both rather than picking one, per the standing instruction not to infer
a mapping that isn't stated in the code.

### B1 — the A/B/C physical tiers (`gas_conversion_tiers()`, plant-PAF bands)

`src/scenarios.py:1239-1259`, docstring quoted in full:

```
Derive the three OCGT-to-CCGT conversion tiers from
data/capacity/raw/gas_plant_availability.csv (Phase 2.8, Step 1).

Per plant: mean PAF = mean(paf_2024_pct, paf_2025q4_pct,
paf_2026q1_pct) / 100. Plants are banded by _CONVERSION_TIER_BANDS.

Returns
-------
dict[str, dict]  keyed "A" / "B" / "C", each:
    {"plants": list[str],
     "nameplate_mw": float,        # sum of installed_mw in the tier
     "available_mw": float,        # sum of installed_mw * mean_paf
     "paf": float}                 # available_mw / nameplate_mw
                                   # (nameplate-weighted mean PAF)

Raises if the CSV is missing, a required column is absent, tier
nameplate does not sum to _CONVERSION_NAMEPLATE_TOTAL_MW, or a plant
lands in zero or more than one tier.
```

Basis for the split (`src/scenarios.py:1222-1236`, comment quoted in
full):

```
Merit-order banding on each plant's mean PAF across the three reported
periods. These three predicates are the ONLY literals in the tier
derivation -- membership, nameplate and availability-weighted PAF all
fall out of the CSV. Bands partition [0, 1]: A is PAF > 0.50, B is
0.25 <= PAF <= 0.50, C is PAF < 0.25 (0.50 and 0.25 fall in B).
```

```
Structural guard target: the 16 convertible plants' nameplate MUST sum
to exactly this (step 0 confirmed 571 + 3,215 + 2,696).
```

So: tier boundaries are **measured** (mean of three NERC-reported
plant-availability-factor snapshots — `paf_2024_pct`, `paf_2025q4_pct`,
`paf_2026q1_pct`, sourced to *"NERC Table 7.1 2024; NERC 2026/Q1 Table
1 for 2025/Q4 and 2026/Q1"*, `src/scenarios.py:1211-1213`), banded by
three fixed cutoffs (0.50, 0.25) that are themselves **DECLARED**
(no source given for choosing those specific cutoffs — SOURCED for the
per-plant PAF data feeding the bands, ASSUMED for the cutoff values
themselves). This is `Ȳ_k`: each tier's `nameplate_mw` is the bound on
how much of that tier's plant can be converted — confirmed as an
actual model bound, not just a diagnostic figure, by
`src/optimize_model_sliced.py:2109-2159` (`conversion_diagnostic()`),
which reports `convert_mw` against `nameplate_mw` as `share_of_nameplate`,
and by `src/optimize_model_sliced.py:529-538`, which requires all three
of `A`/`B`/`C` be present in `scenario["conversion_tiers"]` before
building the conversion mechanism. Step-0 nameplate figures: A = 571
MW, B = 3,215 MW, C = 2,696 MW (sum 6,482 MW, the guard target).

### B2 — the conv_low / conv_central / conv_high economic arms

`src/scenarios.py:1319-1357`, docstring quoted in full:

```
OCGT-to-CCGT conversion arm (Phase 2.8, Step 2). REQUIRED, no default
-- see the _CONVERSION_REQUIRED sentinel above load_scenario(). Mirrors
the gas_capex_case / ccgt_efficiency_case discipline: capex, capacity
uplift and retrofit efficiency are declared judgement calls, not
measured constants.

conv_none reproduces pre-2.8 results exactly: build_model_sliced()
builds NO conversion components when conversion_available is False, and
build_model() (annual) raises on any available arm -- conversion
dispatch is slice-resolved (Decision 3) and has no annual form.

Conversion capex: DERIVED, not SOURCED. No public OCGT-to-CCGT
retrofit capex figure exists. Method: greenfield CC-minus-SC
increment (Gas Turbine World 2024: F-class 237MW simple cycle
$713/kW installed; H-class combined cycle $950-1,084/kW) plus a
brownfield premium. Ipieca (2022) states OCGT-to-CCGT retrofit
poses brownfield challenges of complexity, cost and deferred
production, and that such upgrades are UNCOMMON industry-wide --
the qualitative source for the premium and a declared limitation.
Retrofit efficiency 0.46 HHV: an OCGT not specified for
bottoming-cycle exhaust conditions will not reach greenfield CCGT
efficiency (ccgt_eff_central 0.5173). ASSUMED, bracketed 0.42-0.50,
same 5% ambient derate discipline as the new-build arm.
Capacity uplift 0.50: steam turbine contributes ~1/3 of combined
cycle output. ASSUMED, bracketed 0.40-0.50.

Fields per arm:
  available            : bool. False = conv_none regression arm.
  capex_per_kw          : USD per kW of OCGT NAMEPLATE converted
                         (Decision 2 -- the HRSG is sized for the GT
                         exhaust, so capex is on nameplate; the
                         benefit accrues only on available capacity).
  uplift               : bottoming-cycle capacity as a fraction of
                         converted OCGT nameplate (Decision 1).
  retrofit_efficiency  : HHV electrical efficiency of the converted
                         plant (Decision 3 -- distinct from gas_eta).
```

Values (`src/scenarios.py:1358-1383`):

| arm | capex_per_kw (USD/kW) | uplift | retrofit_efficiency (η_c) |
|---|---|---|---|
| `conv_none` | `None` (regression arm, no conversion built) | `None` | `None` |
| `conv_central` | 600.0 | 0.50 | 0.46 |
| `conv_low` | 850.0 | 0.40 | 0.42 |
| `conv_high` | 450.0 | 0.50 | 0.50 |

`η_g` (the pre-conversion fleet gas efficiency the docstring's *"Decision
3 -- distinct from gas_eta"* refers to) is `gas_eta = 0.287`, hardcoded
unconditionally at `src/scenarios.py:2976` — see Part D. So the
efficiency gain implied per arm is `η_c/η_g`: `conv_central`
0.46/0.287 = 1.60x; `conv_low` 0.42/0.287 = 1.46x; `conv_high`
0.50/0.287 = 1.74x.

Basis for splitting into three (economic arms): **DERIVED, not
SOURCED**, per the docstring's own header sentence — capex is a
greenfield combined-cycle-minus-simple-cycle cost delta (Gas Turbine
World 2024, two point figures) plus an unquantified "brownfield
premium" whose only support is a qualitative Ipieca (2022) statement
that such retrofits are uncommon and complex; `uplift` and
`retrofit_efficiency` are both explicitly labelled ASSUMED with stated
brackets (0.40-0.50 and 0.42-0.50 respectively), and `conv_central`
sits at the bracket midpoint/interior for uplift and at the bracket
floor for efficiency. Per `docs/parameter_provenance.md`'s six-field
rule: **none of the three conversion arms is SOURCED** — capex has a
source for its greenfield component only (Gas Turbine World 2024,
locator NOT RECORDED beyond "2024" in the docstring) but the brownfield
premium that produces the actual 600/850/450 figures is unsourced;
uplift and retrofit_efficiency are both declared ASSUMED outright, with
no locator, no quantity beyond the docstring's own prose, and no
externally-stated range (the 0.40-0.50 / 0.42-0.50 brackets are the
model's own declared brackets, not a source's confidence interval).

---

## PART C — hydro arms

### Capacity/energy trajectories and representation

`src/scenarios.py:1658-1712`, docstring quoted in full:

```
Exogenous annual hydro ENERGY series (TWh_e/year) and the matching
installed-CAPACITY series (MW), both aligned to the model year index,
for the hydro_case arm (correction plan 1.5). Hydro is NEVER a
decision variable -- both series are fixed data.

CAPACITY DERIVED FROM ENERGY at the fixed anchor capacity factor
_HYDRO_CF (= 11.47 TWh / (2,678 MW x 8760 h) = 0.489):
    hydro_capacity_mw[t] = 2,678 x hydro_energy_twh[t] / 11.47
so capacity and energy never drift and capacity_credit["hydro"]
(a dry-season-output-per-installed-MW fraction) stays invariant to
the energy level -- load_scenario() does not re-derive it per arm.
This DELIBERATELY conflates capacity growth with UTILISATION growth:
under hydro_committed / hydro_ndc the extra energy is partly Zungeru
ramping up on already-installed plant (its 700 MW is already inside
the 2,678 MW anchor), not purely new capacity. The fixed-CF
derivation is the price of keeping capacity_credit["hydro"] coherent
without touching it. DECLARED.

FIXED-CF DIRECTION OF BIAS (Phase 1.5 validity guards, step 3):
uprating or better-utilising EXISTING plant raises installed MW
without raising available flow, so a hydro system's real capacity
factor FALLS as its capacity grows. Holding CF fixed at the
2,678 MW anchor's 0.4889 therefore OVERSTATES energy per MW at
higher capacity -- it biases the high-hydro arms (hydro_ndc
especially) toward MORE energy than the resource actually supports.
The direction is stated, NOT corrected: a declining-CF derivation
would make capacity no longer proportional to energy and break
capacity_credit["hydro"]'s invariance to the energy level (see
step 1 of docs/phase_1_5_hydro.md).

VALIDITY GUARDS (Phase 1.5): raises if any year's derived
hydro_capacity_mw exceeds _HYDRO_EXPLOITABLE_MW (ECN's 17,620 MW
exploitable-hydro estimate); WARNS (never raises) if the per-year
capacity build rate exceeds _HYDRO_BUILD_RATE_WARN_MW_PER_YR
(one Zungeru-equivalent, 700 MW/yr).

Arms:
  hydro_flat      SOURCED level 11.47 TWh_e (NERC 2024, 30.92% of
                  37,093.70 GWh), held flat the whole horizon.
                  REGRESSION ARM -- reproduces every pre-1.5 result
                  exactly (energy 11.47, capacity 2,678 in every year).
  hydro_committed SOURCED 2025 figure ~12.8 TWh_e (Zungeru's 700 MW
                  plus minor committed additions, correction plan
                  1.5), held flat thereafter. "12.8" is SOURCED;
                  "held flat after 2025" is DECLARED.
  hydro_ndc       Toward the NDC's SOURCED 10,400 MW by 2030
                  (correction plan 1.5) -- its energy equivalent at
                  _HYDRO_CF, ~44.5 TWh_e -- linear from the 2025
                  committed level, then held flat 2031 onward. The
                  10,400 MW 2030 target is SOURCED; the linear
                  interpolation SHAPE and the post-2030 flat treatment
                  are DECLARED, not sourced.
```

Computed trajectories (from the arm logic, `src/scenarios.py:1713-1738`):

| arm | 2025 energy (TWh_e) | 2025 capacity (MW) | 2030 energy | 2030 capacity | held-flat energy after | classification |
|---|---|---|---|---|---|---|
| `hydro_flat` | 11.47 | 2,678 | 11.47 | 2,678 | flat throughout | **SOURCED** level; DECLARED that it stays flat (it is the regression arm) |
| `hydro_committed` | 12.8 | 2,988.6* | 12.8 | 2,988.6* | flat from 2025 | **SOURCED** the 12.8 figure; **DECLARED** the flat-after-2025 treatment |
| `hydro_ndc` | 12.8 | 2,988.6* | 44.522... † | 10,400 | flat from 2030 | **SOURCED** the 10,400 MW 2030 target; **DECLARED** the linear interpolation shape and the post-2030 flat hold |

\* `2,678 × 12.8 / 11.47 = 2,988.58...` MW, via the stated fixed-CF
formula (not separately stated in the docstring; computed here from the
formula it gives).
† `11.47 × 10,400 / 2,678 = 44.5217...` TWh_e, matching the docstring's
own "~44.5 TWh_e" figure.

Fixed-CF bias direction is stated explicitly in the docstring quoted
above: it **overstates** energy per MW as capacity grows, biasing
`hydro_ndc` toward more energy than the resource supports — a declared,
uncorrected limitation, not a silent one.

### θ_σ — the seasonal split

`src/scenarios.py:3139-3164`, comment quoted in full:

```
Hydro seasonal energy split.
SOURCED, and the sign of the retired value was WRONG.
Two independent NERC quarterly datasets, both excluding Zungeru:
  2023, PRE-ZUNGERU (Zungeru absent from the 2023 tables), total
  generation GWh for Shiroro + Jebba + Kainji + Dadin Kowa,
  converted to average power: Q1 1,097.6 MW / Q2 823.6 /
  Q3 1,000.2 / Q4 1,224.0. ratio 1.118
  Clean check, Q1 (all dry) vs Q3 (all wet): 1.097
  2025, Zungeru excluded from the aggregate, average hourly
  generation basis: ratio 1.070
Two-year mean ratio 1.0938 -> dry 1.0452 / wet 0.9556.
Quarters straddle the dry/wet boundary (Q1 all dry, Q3 all wet,
Q2 and Q4 mixed), so quarters are month-weighted onto the split.
WHY THE RETIRED VALUE WAS BACKWARDS: 0.90/1.10 assumed
run-of-river hydrology -- low flow in the dry season. Kainji and
Jebba are RESERVOIR schemes on the Niger (Kainji regulates,
Jebba re-regulates downstream), so output follows DEMAND, not
inflow, and Nigerian demand peaks in the dry season. Operators
draw down through the dry months. The seasonal signal is an
operating decision, not a hydrological one.
KEEP IN SYNC with the identical literal feeding
capacity_credit['hydro'] above (this function, ~30 lines up) and
with DEFAULT_HYDRO_SEASON_FACTOR / DEFAULT_SEASON_SOLAR_FACTOR
(src/optimize_model_sliced.py) -- four copies of these two
numbers exist; see docs/shape_factors_and_dist5.md.
```

**Current value: `hydro_season_factor = {"dry": 1.045, "wet": 0.956}`**
(`src/scenarios.py:3164`, and duplicated as
`DEFAULT_HYDRO_SEASON_FACTOR` at `src/optimize_model_sliced.py:138` —
see Part D). Per the comment: this figure IS the correction — the
"sign reversal" `docs/phase_1_5_hydro.md` records. **What it
replaced**: a retired `0.90/1.10` split (dry lower than wet), which the
comment states was backwards because it wrongly assumed run-of-river
(inflow-driven) hydrology; the corrected direction (dry higher than
wet) reflects that Kainji/Jebba are reservoir schemes operated to track
Nigeria's dry-season demand peak, not inflow. Classification:
**SOURCED** — value, source (two independent NERC quarterly datasets,
2023 and 2025), locator (the quarterly figures and ratios quoted
above, in-file), quantity (hydro generation seasonal ratio, dry vs
wet), and an internal fit/consistency check (two years agree to within
~0.05 of ratio) are all present; no external published range is
stated.

### Minimum-flow fraction

`src/scenarios.py:3165-3198`, comment quoted in full:

```
Minimum hydro per slice as a fraction of proportional share
(environmental flow, irrigation, downstream obligations).
Classification: ASSUMED (analyst estimate), NOT SOURCED.
A documented search found no Nigerian minimum-flow obligation
expressed as a fraction. Nigerian dam-operation sources give
ABSOLUTE DISCHARGE RATES, not fractions.
Nearest documented analogue: JICA, Nigeria National Water
Resources Master Plan, Vol. 5, SR6, p. SR6-7, Table SR6-9,
"Assumed Operation Rules for Release from Shiroro Dam" -- 315 /
100 / 50 m3/s at reservoir storage of 20% / 10% / below 10%.
JICA itself labels these ASSUMED operating rules, not a
mandated environmental-flow obligation.
That rule is NOT convertible into this parameter, for two
independent reasons: (i) it is an absolute discharge at a
single plant, while this parameter is a fleet-wide fraction of
an hours-proportional per-slice energy share; (ii) it is
STORAGE-STATE DEPENDENT, and this model carries no reservoir
state variable (verified: the only hydro decision variable is
m.hydro_gen[t,(s,p)], a per-slice dispatch quantity, and all
three hydro constraints in optimize_model_sliced.py are
contemporaneous -- see docs/task_hydro_gate_bucketA.md, 1A), so
there is nothing for such a rule to key off even in principle.
Turbine discharge at 95% reliability (Kainji 440, Jebba 500,
Shiroro 120 m3/s) is PLANT CAPABILITY, not release obligation,
and must not be read as a minimum flow. An earlier reading of
the 500 m3/s figure as an obligation was incorrect and is
recorded here so it is not repeated.
PROHIBITION: do not derive, estimate, back-calculate or
otherwise produce a numeric fraction from any discharge figure
above. Conversion would require head, turbine efficiency,
obligation coverage, fleet weighting and a slice-mapping --
five assumptions whose product would be an analyst construction
wearing a citation. The value stays 0.50 and stays ASSUMED.
```

**Current value: `hydro_min_flow_fraction = 0.50`**
(`src/scenarios.py:3198`, duplicated as
`DEFAULT_HYDRO_MIN_FLOW_FRACTION` at
`src/optimize_model_sliced.py:147` — see Part D). Classification:
**ASSUMED**, explicitly and repeatedly self-declared as such in the
comment itself — the file's own text states a documented search for a
Nigerian source came up empty, names the nearest analogue (JICA's
Shiroro operating rules) and explains in detail why it is *not*
convertible into this parameter, and ends with an explicit prohibition
against constructing a numeric fraction from adjacent discharge data.
There is no "what it replaced" for this one in the sense Part C asks
about the seasonal split — this parameter was never corrected from a
wrong prior value; the note about "an earlier reading of the 500 m3/s
figure as an obligation" being incorrect refers to a candidate source
that was considered and rejected before ever being adopted as the
model's value, not a value that was live and then reversed.

---

## PART D — hardcoded literals reached through a default, in the manner of `within_block_peaking_factor`

`within_block_peaking_factor`'s own pattern, for reference
(`src/optimize_model.py:1039-1040`):
`within_block_peaking_factor = float(scenario.get("within_block_peaking_factor", 1.3448))`.

Found within the three areas above:

| # | parameter | file : line | default value | pattern |
|---|---|---|---|---|
| 1 | `new_gas_deliverability` | `src/optimize_model.py:652` | `"constrained"` | `scenario.get("new_gas_deliverability", "constrained")` |
| 2 | `new_gas_deliverability` | `src/optimize_model_sliced.py:894` | `"constrained"` | `scenario.get("new_gas_deliverability", "constrained")` — identical default, second builder |
| 3 | `conversion_available` | `src/optimize_model_sliced.py:516` | `False` | `bool(scenario.get("conversion_available", False))` |
| 4 | `hydro_season_factor` | `src/optimize_model_sliced.py:185` (season_solar_factor) / `209` (hydro_season_factor) | `{"dry": 1.045, "wet": 0.956}` | `scenario.get("hydro_season_factor", DEFAULT_HYDRO_SEASON_FACTOR)`, constant defined line 138 |
| 5 | `hydro_min_flow_fraction` | `src/optimize_model_sliced.py:794-795` | `0.50` | `scenario.get("hydro_min_flow_fraction", DEFAULT_HYDRO_MIN_FLOW_FRACTION)`, constant defined line 147 |

Row 4 is one of the *"four copies of these two numbers"* the
`hydro_season_factor` comment itself flags (Part C above) — the sibling
copy, `season_solar_factor` (`DEFAULT_SEASON_SOLAR_FACTOR =
{"dry": 1.076, "wet": 0.925}`, `src/optimize_model_sliced.py:107`,
consumed via `scenario.get("season_solar_factor", DEFAULT_SEASON_SOLAR_FACTOR)`
at line 185), is a **solar**, not hydro, shape factor — outside this
task's three named areas strictly construed, but listed here for
completeness since it shares the identical default-fallback pattern
and file, and the source comment for `hydro_season_factor` explicitly
cross-references it.

Checked directly against `load_scenario()`'s returned dict, not just
the model-builder `.get()` call sites: all five keys in the table ARE
set by `load_scenario()` on every path —
`"new_gas_deliverability": new_gas_deliverability` (`src/scenarios.py:3329,3360`),
`"conversion_available": _conversion_arm["available"]`
(`src/scenarios.py:3339`), `"hydro_season_factor"` (`:3164`) and
`"hydro_min_flow_fraction"` (`:3198`, both quoted in Part C). So **every
`.get()` fallback in the Part D table is DEAD under `load_scenario()`**
and LIVE only for a hand-built scenario dict that bypasses it — the
same status as `within_block_peaking_factor` itself. (An earlier draft
of this section asserted `new_gas_deliverability` and
`conversion_available` were absent from the returned dict; that was
wrong — checked and corrected in-place before this document was
finalised, not left as a separate erratum.)

`new_gas_deliverability` carries a second, separate default layer
worth distinguishing from the table above: `load_scenario()`'s own
signature sets `new_gas_deliverability: str = "constrained"`
(`src/scenarios.py:2203`) — a plain default parameter, not a
sentinel-guarded required key like the seventeen `_XXX_REQUIRED`
arms. This one is explicitly self-declared as intentional, not an
oversight, `src/scenarios.py:2824-2828` comment quoted in full:

```
new_gas_deliverability (2.9, Step 3). Real default ('constrained'
-- the Nigerian baseline, capacity without fuel, not relaxed by
assumption), unlike the REQUIRED-no-default arms above: both
treatments are implemented (see build_model()/build_model_sliced()),
this default only governs what an unstated caller gets.
```

So `"constrained"` is baked in at two independent points — this
`load_scenario()` parameter default, and the model-builder `.get()`
fallback in the table above — but only the first is ever live for a
caller going through `load_scenario()`, and the code itself frames
that default as a deliberate choice (favouring the fuel-constrained
Nigerian baseline over the unstated alternative), unlike
`within_block_peaking_factor`, whose own adjacent comment claims a
default of 1.0 that contradicts the actual coded 1.3448 (register L50
/ forensic audit A62, per `docs/parameter_provenance.md` row 7) — that
specific comment/code mismatch was NOT found here for
`new_gas_deliverability`; its default is documented correctly.

`conversion_available` has no equivalent second default layer — it is
not a parameter of `load_scenario()`'s own signature; it is fully
determined by the sentinel-guarded, REQUIRED `conversion_case` arm
(`_conversion_arm["available"]`), so its only live default anywhere is
the `.get()` fallback in the table, reachable exclusively by a
hand-built scenario dict that skips `load_scenario()` entirely.

### One further hardcoded literal, distinct pattern (not a `.get()` default at all)

`gas_eta = 0.287` — `src/scenarios.py:2976`, inside the `scenario = {...}`
dict: `"gas_eta": 0.287,`. This is **not** reached through any
`scenario.get(key, default)` fallback — it is placed directly and
unconditionally into every `load_scenario()` output, with **no
`gas_eta_case` arm, no CLI flag, and no alternative value reachable
anywhere in the codebase** (confirmed: no other assignment to
`"gas_eta"` found in `src/scenarios.py`). This is a stronger form of
the same underlying issue `within_block_peaking_factor` represents
(an unreviewable constant embedded in a required-key discipline that
otherwise names everything else as an explicit arm), but it is not the
same code pattern and is reported separately rather than folded into
the table above.

---

**Not run. Nothing in this document was executed, solved, or
recomputed** beyond reading values already present in
`data/gas/processed/gas_available_power_annual_twh_th.csv` (Part A
trajectories) and simple arithmetic stated inline and shown as such
(Part C's capacity-from-energy conversions, using the formula the
docstring itself gives). No file other than this one was created or
edited.
