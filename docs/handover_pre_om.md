# Handover — Nigeria LP-CEM thesis, session close

## Read these first, in the repo
    docs/findings_provisional.md            findings at the central config
    docs/findings_provisional_pre_sweep.md  superseded, kept for reference
    docs/limitations_register.md            L1-L17 + process notes
    docs/methodology.md                     living methods doc
    merged_correction_plan_v2.md            the master plan

Branch `phase-fin2-d17`. Everything below is committed unless stated.

---

## Working agreement

**Accuracy over agreement.** Push back on errors, record retractions
explicitly, never silently correct. Several findings in this project
were overturned by checking rather than assuming — including two of my
own readings in the final session.

**Three-way split.** Chat = design, sourcing, interpretation, writing
TASK.md files. Claude Code = inventory passes, edits against real line
numbers, mechanical repetition. Zeus = every solve, every runner.

**TASK.md discipline that works.** State what is already COMPLETE so the
agent does not re-verify it. Bound execution explicitly — "run NOTHING
except X" — because enumerated prohibitions get read as exhaustive and
substitutes get invented. Require incremental reporting: two runs in
this project produced edits across 16 files with no report because the
report was written last. Put a STOP after any inventory step whose
findings could change the task's shape.

---

## Model state

Sliced LP capacity-expansion model, Pyomo/CBC, 2025-2054 horizon,
**reported to 2045** (2046-54 is end-effect buffer). Six slices:
{dry,wet} x {night,day,peak}. Real USD.

**Anchors, both reproduce to the digit:**
    regression  conv_none + pre_f1_overnight + dscr_regression
                + tariff_legacy_65m         20.9899 bn / 41,565 MWh
    central     conv_central + f1_tranche + dscr_central
                + cr_central + env_dares + baseline deliv
                + hydro_flat + cf_central   20.856 bn

The pre-sourcing anchor of 21.4748 bn is NO LONGER REPRODUCIBLE. The
shape-factor sourcing moved it. Do not treat it as a target.

**Required scenario keys** (no defaults, raise if omitted): peak_multiple,
storage_cc, gas_build, new_gas_deliverability, gas_availability,
gas_retirement, gas_capex, ccgt_efficiency, gas_deliverability,
capital_recovery, dscr, us_inflation, capital_treatment,
concessional_envelope, conversion, hydro, tariff, dist5, solar_cf.

**Sweep:** `bash scripts/run_central_sweep.sh [A|B|C|D]`. 37 arms,
zero failures at last run. Section A carries the regression anchor —
if it moves, nothing below it is attributable.

---

## What the model says

Ranked by effect on system cost:

    gas fleet availability bracket    7.32 bn   <- LARGEST
    capital recovery span             3.90
    conversion (none -> central)      3.14
    gas retirement (none -> v30)      2.71
    concessional envelope (0 -> 10x)  2.45
    deliverability, drift family      2.55
    demand shape                      1.90

**The contribution holds.** Cost of capital reaches the objective: the
F0 audit found four blend scenarios identical to seven significant
figures; the span is now 3.90 bn. Tenor is 53% of the concessional
advantage by Shapley decomposition, and blended-WACC formulations
cannot express tenor at all.

**But gas availability dominates it, roughly 2:1.** There is no sourced
central arm — `gas_avail_central` raises by design, because mechanical
availability cannot be separated from fuel unavailability without data
nobody publishes. The central case sits on the declared UPPER bound
(0.80) and the declared lower bound (0.295, measured) costs a third
more. State this prominently; a reader will otherwise find it.

**Financing moves cost; physics moves quantity.** Storage is 62,181 MWh
in every capital-recovery arm, every envelope arm and every DSCR arm.
It moves only for conversion, deliverability, hydro, availability, peak
shape and retirement.

**Under NDC caps:** conversion is worth 33-36 bn (-38 to -44%); the
EaaS channel 8.7-21.5 bn; extra gas deliverability EXACTLY ZERO
(byte-identical across three arms in all four cap cases). The abatement
lever is efficiency, not fuel.

**EaaS survives DIST-5 almost untouched** — under 0.03% on objectives of
26-43 bn, with EaaS deployment identical to the megawatt. Three sourced
corrections push against it (capex +16%->+37%, CF 0.19->0.164, seasonal
profile flattened) and none of them bites, because the mechanism is
about capital availability, not solar economics. That is a stronger
result than survival by luck.

---

## Two findings that bound the thesis's own claim

**Hydro.** Under NDC-scale hydro, solar falls 83% and system cost 58%.
The defence is deliverability, not modelling: `hydro_ndc` needs 2.1
Zungeru-equivalents per year for five years against 8-12 year build
times. NIRP's own 20.6 GW by 2045 is 117% of the Energy Commission's
exploitable estimate.

**Storage magnitude is not a cost optimum.** Flat capacity credit of 1.0
(register L5) means no declining ELCC, so storage absorbs the entire
firm-capacity residual once gas floors. The COMPOSITION claim is sound;
the MAGNITUDE claim is not.

---

## The hydro sign reversal — the best single finding

`hydro_season_factor` was 0.90 dry / 1.10 wet. Measured: **1.045 / 0.956
— dry is HIGHER.** Confirmed on 2023 (pre-Zungeru, the proper control),
2025 (Zungeru excluded), and corroborated by 2021 load factors.

The mechanism is the Niger's **black flood**: water from Guinea and Mali,
silt deposited crossing the Inner Niger Delta, peaks at Kainji in
January-February at ~2,000 m3/s and falls below 100 m3/s by May-June.
The local white flood (Sokoto, Malendo) peaks in late September. So the
documented minimum inflow is April-July — inside the model's "wet"
season — and the lake peaks in December.

The retired parameter applied a textbook run-of-river assumption to a
river whose main flood arrives in the dry season.

**Declared limitation:** the two-season split cuts across the hydrograph.
Both blocks contain a peak and a trough. The underlying quarterly swing
is 824 -> 1,224 MW (1.49x); the two-block factor captures 1.094 of it.
The signal is correctly signed and substantially UNDERSTATED by the
six-slice design.

---

## Open items

**Blocked on sourcing**
- Storage capacity credit — needs an ELCC curve (L5). The last
  parameter making a headline quantity uninterpretable.

**Code, post-sweep safe**
- `26_run_f3_identification.py` lacks `--dist5-case` and
  `--solar-cf-case` despite both tasks reporting they were wired. It
  sets them internally — the same silent-default class as the old
  tariff key. LOG AND FIX.
- Rename reliability -> energy adequacy. Model side is done; the REL
  runners and prose still carry it. Entangled with the REL cleanup
  (D21/D23/D24), so consider doing both together.
- Doc-code parity sweep: `storage.py` implements sqrt-eta symmetric
  losses while the LP applies full eta on discharge only; `EaaS.py`'s
  CRF machinery is never called; `economics.solar_capex` divides by
  n_years and calls it annualisation.
- Dual verification by perturbation. Partly de-risked — the 0.4 sign
  assertions passed on a full NDC solve.
- Cap apportionment ownership: the proportional rule is the analyst's
  construction, not NDC 3.0's. Present as a proposed rule with one
  alternative. Framing only, no code.

**Unexplained, worth one look**
- Lower solar CF gives a CHEAPER system: 20.086 bn at cf_low (0.16)
  against 20.856 at cf_central (0.19). Single-variable, same runner, so
  it is real. Suggests solar is at or near a cost knife-edge in the
  corrected model.

**Resolved, do not re-open**
- The F3 "floor collapse" was my misreading — I compared a `tight`
  figure against an `unconstrained` one. The floor is 36,687-39,993 at
  `tight` and zero only at `unconstrained`, which is exactly what F-28
  predicts. No anomaly.

---

## Declared, not fixed

- **L17** — the bankability constraint discounts at the retired pre-F1
  blended WACC (14.4%, 70% composed of the 18% naira anchor F1 removed
  from the objective) over the model horizon rather than the commercial
  tenor. Two errors partly cancel at t=0 (-5%) and diverge after. An
  attempt to fix it consumed an hour across 16 files with no report and
  was discarded (`git stash`). Step-0 inventory survives in
  `docs/bankability_tenor.md`.
- **L16** — DIST-2 (behind-the-meter avoids the 7% TLF) is NOT
  implemented, because the metering point is declared at the busbar.
  DIST-5 alone therefore biases against EaaS. Direction stated.
- **L10** — ten runners dead on required args from phases 2.4-2.13.
  `17_run_monte_carlo` additionally has two declared defects and is
  deferred, not retired.

---

## Next

1. Restate `docs/findings_provisional.md` at the post-sourcing numbers.
   Every figure in it predates the shape-factor sourcing.
2. Update `docs/methodology.md` with the sourced parameters, the hydro
   mechanism, and DIST-5.
3. **Set the freeze date.** Deferred six times. Nothing outstanding
   would invalidate the sweep.

## F3 at corrected parameters — RESOLVED

Total solar 28,043 MW at 2045, gas 6,695 MW, selfgen 53.34 TWh —
IDENTICAL at min and max, in all three budget arms. Only the ownership
split moves; EaaS + public is ~49,480 MW cumulative in every solve.

The ceiling (49,486 MW) is TOTAL SOLAR, not bankability — which is why
DIST-5's capex increase did not move it. The floor is whatever the
budget forbids public solar from supplying: 0 unconstrained, 15,328 at
moderate, 39,993 at tight.

Public and EaaS solar are perfect substitutes at the margin at zero cost
difference. Total solar is IDENTIFIED; the split is a pure financing
choice. Restate F-12 and F-28 accordingly — "degenerate" reads as a
defect and this is the mechanism working.