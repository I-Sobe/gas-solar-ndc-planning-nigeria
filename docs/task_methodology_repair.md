# Methodology repair — Part 1 inventory

Branch `phase-fin2-d17`. No file edited during this turn — Part 1 is
inventory only, per TASK.md. No `src/`, no scripts, no solver run.

**Cross-check performed before the per-file inventory:** grepped all four
files for O&M-related vocabulary (`O&M`, `operating and maintenance`,
`FOM`, `VOM`, `fixed cost`, `variable cost`, `maintenance`, `fuel cost`,
case-insensitive) and for the register/sweep/STORAGE_OM patterns used
below. Zero hits for every O&M term, in every file. This is the single
fact behind most of 1A/1D below, so it is stated once here rather than
four times.

---

## `docs/methodology.md` (564 lines)

### 1A — objective function

**Yes, one passage.** §4.2 "Per-tranche capital recovery factor," lines
151–157:

```
    CRF(r, L) = r(1+r)^L / ((1+r)^L - 1)

    capital_charge[v] = CRF(r_conc, L_conc) x conc_vol[v]
                      + CRF(r_comm, L_comm) x comm_vol[v]

    s.t.  conc_vol[v] + comm_vol[v] = capex[v] x add[v]
          sum_v conc_vol[v] <= concessional_envelope_usd
```

**No O&M term present.** This formula is explicitly scoped to the
*capital charge* component only (its own heading says so), so its absence
of an O&M term is not, by itself, a misrepresentation of that one
component. What makes it a finding is the surrounding silence: nowhere
else in this 564-line document — not in §2's model description, not in
§4.1's discussion of "system cost," not in §7's verification discipline —
is O&M, FOM, VOM, or "operating cost" mentioned. A reader assembling "the
objective" from this document alone would build capital charge + (implied
fuel, gas deliverability shadow value, etc.) and never learn a
+30.9%-of-central term is missing.

### 1B — objective/anchor/storage/solar/backcast figures

| Line | Value | Status |
|---|---|---|
| 137 | "2.765439e10 to seven significant figures" | Historical diagnostic (§4.1, "Before correction..."), explicitly framed as pre-correction. Not a current-state claim; no correction needed |
| 487 | backcast "within -3.4%" | **STALE.** §2 establishes the current central backcast as **-3.2%**. -3.4% was the value after the min-build floor was removed but before the frozen state (see also `docs/methodology_demand_horizon.md` finding below, which is even older) |
| 556–557 | CF sensitivity "0.16 -> 20.086 bn against 0.19 -> 20.856" | **STALE.** These are pre-O&M objective values (register L18/now-L26 records the same 20.086/20.856 pair). O&M added +6.602 bn / +30.9% at central; neither figure reflects that. The qualitative finding they support ("solar sits near a cost knife-edge," also asserted at lines 555, 558-559) may still hold in direction but the dollar figures are pre-O&M and must not be quoted as current |

No mention anywhere in this file of the regression anchor
(21,811,042,472.821114 / 45,502.9754 MWh), the frozen central objective
(27.954 bn), or the current storage/solar totals (65,613.0 MWh /
41,031.4 MW) — checked by grep, no hits.

### 1C — the sweep

**No.** The only "sweep" occurrences (line 58) use the word generically
("no parameter sweep can change this"), not referring to the central
`run_central_sweep.sh` construct. No "37 arms," no "four sections," no
Section A/B/C/D language anywhere in this file.

### 1D — `STORAGE_OM_PER_MWH_YR` / storage no-operating-cost claim

**No explicit reference either way.** No mention of the constant by name,
and no explicit sentence asserting storage carries zero operating cost.
The complete silence on O&M (1A) means a reader would *infer* storage (and
everything else) carries capital cost only — an implicit version of the
same gap, not a distinct explicit claim requiring its own correction
beyond what 1A already identifies.

### 1E — register citations

**None.** No `L`-numbered citation anywhere in this file (checked
case-sensitively and for `register L` phrasing). Nothing to renumber.

### 1F — emissions cap construction

**No — explicitly out of scope, and says so.** Line 17: "It does not yet
cover demand construction, **emissions cap derivation**, or the
reliability formulation in comparable depth." The document does not
describe cap construction or the post-2035 shape; its own scope note
already says this. No correction needed on this point — the scope
disclaimer is itself still accurate (this document genuinely doesn't
cover it).

### 1G — other claims that look false, one line each

- **§8 "Central configuration for the reported result set"** (line
  529–532) reads as an open, unresolved decision — "The headline set
  requires re-running at a single declared central configuration" — but
  §2's established facts describe exactly such a configuration, already
  run, with a reported central objective (27.954 bn) and full arm list.
  This OPEN item is very likely **RESOLVED** and stale as written; **not
  certain from this document alone** — worth confirming against
  `docs/findings_frozen.md` before correcting, since "declared" and
  "re-run" are specific claims this document can't verify on its own.
- **§8 lists "Two tenors for one asset" twice** (lines 547–554 and
  560–564), as two separate bullets under the same heading with
  overlapping but not identical content (the second adds "at a retired
  blended rate" and "partly cancel... diverge thereafter," matching
  register L17 more closely than the first). Not exactly "false" — both
  versions are individually accurate — but exactly the "two accounts a
  reader must reconcile" problem Part 2's own rules warn against. Listed
  here rather than as a numbered stale claim; see closing list.

---

## `docs/methodology_physical_layer.md` (515 lines)

### 1A — objective function

**No.** This document's own scope note (lines 14–20) places "capital
structure and the per-tranche capital charge (F1)" explicitly among
sections "NOT covered here, and which should be written elsewhere." It
does not attempt to describe the objective, so there is no passage to
quote and no local omission — the same total silence on O&M applies here
too, but there is no objective-function passage for that silence to sit
inside.

### 1B — objective/anchor/storage/solar/backcast figures

| Line | Value | Status |
|---|---|---|
| 250 | storage build "precisely `77,387.72 / credit` MWh" | **Likely STALE, unverified.** This is presented as an exact structural signature under the *fixed* capacity-credit regime, measured at some earlier configuration. Multiple things have changed since it was plausibly measured — gas retirement moved from a flat proxy to vintage arms, the hydro seasonal factor was corrected, O&M was added — any of which could move the firm-capacity residual this number is derived from. No established fact in §2 confirms or contradicts the exact figure, so it cannot be marked CURRENT; flagged for re-verification before Part 2 corrects it (correcting it blind would risk introducing an invented number) |
| 399 | "1,317 USD/kW... 17.0 bn USD against 12,905 MW" | CURRENT / not applicable — this is a NIRP-derivation input figure (source calibration), not a model result. Unaffected by the O&M/cap-trajectory/register changes |
| 403 | "905 USD/kW (55 bn against 60,756 MW)" | Same as above — NIRP-derivation validation figure, not a model result |

No mention of the regression anchor, the current central objective, or
current storage/solar totals — checked, no hits.

### 1C — the sweep

**No.** Both "sweep" occurrences (lines 127, 218) are generic
("`peak_high` sweeps that direction," "no parameter sweep could have
changed this"), not references to the central sweep construct.

### 1D — `STORAGE_OM_PER_MWH_YR` / storage no-operating-cost claim

**No explicit reference.** Same as `methodology.md` — this document's §3
("Storage") discusses the capacity-credit mechanism and
`storage_duration_hours` at length but never says storage has, or lacks,
an operating cost; the constant is not named.

### 1E — register citations

**None.** No `L`-numbered citation anywhere.

### 1F — emissions cap construction

**No — explicitly out of scope, and says so.** Line 16: "the emissions
cap and NDC scenario construction" is listed among sections "NOT covered
here, and which should be written elsewhere." Consistent with the
document's actual content — no cap material appears.

### 1G — other claims that look false, one line each

- **§7, item 9** (lines 511–515): *"Retrofit of existing open-cycle plant
  to combined cycle is absent from the choice set. The model can build
  new combined cycle but cannot convert or rehabilitate what exists...
  Discussed but not implemented in this work."* **This directly
  contradicts §5.2 of the SAME document** (lines 322–356, status
  "PROVISIONAL — implemented and swept"), which describes exactly this
  mechanism — conversion as "a decision variable," governed by
  `conversion_case`, with a measured convertible fleet (6,482 MW
  nameplate / ~1,941 MW available) and a merit-order screen. The
  "rehabilitate" half of item 9 remains accurate — §5.2 separately
  declares rehabilitation (distinct from conversion) out of scope — but
  the "cannot convert" half is now false, contradicted by this document's
  own §5.2. This is an internal inconsistency, not merely staleness
  against an external event.

---

## `docs/methodology_demand_horizon.md` (552 lines)

### 1A — objective function

**Partial — names it, does not enumerate it.** §4 "Cost reporting basis"
(lines 386–414) gives a table (line 396) naming `system_cost_npv` as "the
Optimisation objective. Drives decisions, comparable across arms." It
states the horizon (2025–2054) and that salvage is included, but gives no
component breakdown — no formula, no list of cost terms. **No O&M term
appears**, consistent with the file-wide silence, but there is no
enumerated list for its absence to stand out from the way it does in
`methodology.md` §4.2.

### 1B — objective/anchor/storage/solar/backcast figures

| Line | Value | Status |
|---|---|---|
| 289 | "$11.5/MWh" (salvage arithmetic) | Historical diagnostic (§3.1, "The artefact that determined the horizon"), explicitly framed as the pre-fix state. No correction needed |
| 294–295 | "1,635 MW to 17,861 MW... 24.67 to zero... 25.3 to 39.3 GW... the objective fell" | Historical diagnostic, same section, explicitly framed as what happened when an unsourced cap was removed (before the horizon fix). No correction needed |
| 350 | "gas 24.57 + hydro 11.47 + existing solar 0.88 ≈ 36.9 TWh against 37.09 TWh" | **STALE.** This is backcast arithmetic implying a **-4.1%** gas backcast (24.57 vs observed 25.62 TWh_e is -4.10%) — a full generation *older* than even the already-superseded -3.4% figure `methodology.md` carries, and two steps behind the current **-3.2%** central backcast established in §2. Unlike the two entries above, this passage is written in the present tense ("The backcast arithmetic shows why 2025 = 0 is correct") without an explicit "before the fix" frame, so a reader has no signal that these specific numbers are stale |

### 1C — the sweep

**No.** The one "sweep" occurrence (line 484, "no logged sweep had been
run") is generic, part of the verification-discipline discussion, not a
reference to the central sweep construct.

### 1D — `STORAGE_OM_PER_MWH_YR` / storage no-operating-cost claim

**No explicit reference.** Not named; no explicit no-operating-cost claim
for storage or anything else. §2 (asset lifetimes and salvage) discusses
capital cost and salvage at length but never operating cost.

### 1F — emissions cap construction

**Partial, and STALE where it appears.** This document does not cover
general cap *derivation* (that's declared elsewhere), but it does cover
the **buffer-year treatment** of the cap, in §3.2, lines 330–336:

> "Emissions caps are set non-binding after 2045. NDC 3.0 provides no
> target beyond 2035, so 2036–2045 is already an apportionment
> construction; extending it further would stack extrapolation on
> extrapolation, and a binding cap in 2050 would change what the model
> builds in 2043."

**"Set non-binding after 2045" is false at the current code state**,
confirmed by direct inspection of `scripts/00_build_emissions_cap.py`
(read-only, not edited — this task's whitelist permits reading any file):
buffer years (2046–2054) are explicitly **held flat** at the terminal
in-window (2035, i.e. really 2045-equivalent) value for all four original
cap scenarios, and the code's own comment names the reason — a
non-binding buffer became an arbitrage sink once the self-generation
backstop existed (Phase 2.6, well before the O&M freeze this task is
about). **The correct, current description of this exact point already
exists** in `docs/methodology_genself_backstop.md` §4.5 (quoted in that
file's entry below) — so this is not merely stale against the frozen
state, it is **inconsistent with another methodology document that has
the current answer**. Worth flagging as the most consequential single
finding in this inventory: two methodology files give opposite answers to
the same factual question, and only one is right.

### 1G — other claims that look false, one line each

- **§7, "Distributed solar capital cost and capacity factor"** (lines
  528–536): *"Whether the distributed arm's advantages survive a properly
  sourced premium is **untested**."* **STALE.** Register L25 (was L16's
  second entry) records DIST-5 as implemented and measured: it changes
  NDC-capped objectives by under 0.03% and leaves EaaS deployment
  identical to the megawatt. Not merely untested-and-now-tested — the
  specific question this passage poses has an answer on record.
- **§7, "Gas fleet efficiency is a fixed parameter"** (lines 538–542):
  *"...the cheapest abatement option available in this system and the
  model cannot represent it. Adopted as a scheduled addition; not yet
  implemented."* **Possibly stale, lower confidence.** This reads as
  describing the same underlying idea that `methodology_physical_layer.md`
  §5.2 now documents as the implemented, swept OCGT-to-CCGT conversion
  mechanism — "adopted as a scheduled addition" is a plausible description
  of conversion *before* it was built. Flagged with lower confidence than
  the DIST-5 item above because this document never uses the word
  "conversion," so the identification rests on inference rather than a
  named contradiction; recommend cross-checking against
  `docs/phase_2_8_conversion.md` before correcting in Part 2, rather than
  assuming the two are the same claim.

---

## `docs/methodology_genself_backstop.md` (543 lines)

### 1A — objective function

**Partial — mentions inclusion, no formula.** §1.5 "Scope of the cost"
(lines 107–113): "Self-generation is a real resource cost and enters
`system_cost_npv`... verified structurally rather than by inspection: a
post-build probe confirms the derivative of the public budget constraint
with respect to `selfgen` is exactly zero." States that selfgen's cost
enters the objective, not what else does. **No O&M term appears**,
consistent with the file-wide silence — but nothing here is specifically
about O&M, so there is no local misstatement, only the same general
absence noted for all four files.

### 1B — objective/anchor/storage/solar/backcast figures

None found. Grepped for `bn\b`, TWh/MWh/MW result-shaped figures, and
backcast percentages — this file's numeric content is genset cost/
efficiency/fuel-price parameters (already sourced and unaffected by
O&M/cap-trajectory/register changes) and the emissions-leakage discussion,
not model-run results. Nothing to mark STALE or CURRENT under this
heading.

### 1C — the sweep

**No.** "Sweep" appears four times (lines 399, 481, 487, 506), all
referring to parameter sweeps of the obligation eps axis or the genset
price arms — never the central `run_central_sweep.sh` construct.

### 1D — `STORAGE_OM_PER_MWH_YR` / storage no-operating-cost claim

**No.** Not applicable — this document does not discuss storage at all
(explicitly out of scope, line 16: "written elsewhere").

### 1E — register citations

**None.** No `L`-numbered citation anywhere.

### 1F — emissions cap construction

**Yes, and CURRENT — this is the document that has it right.** §4
"Emissions-cap boundary and carbon leakage" describes the grid-only
boundary (§4.2, lines 337–345) and, most relevantly, §4.5 "A correction
worth recording — the buffer window was not inert" (lines 376–393):

> "Buffer-year caps are now **held flat at the terminal in-window
> value** — the same treatment already applied to capital cost
> trajectories in the buffer, and for the same reason: buffer years must
> be neither cheaper nor freer than the last reported year, or they pull
> behaviour out of the reporting window. Buffer-year caps are never
> reported."

This matches the current code exactly (see the `methodology_demand_horizon.md`
entry above). **No correction needed here** — flagged only as the
reference point that shows the other document is wrong, not as a finding
against this one. This document does not discuss the post-2035 *shape*
question in the plan-4.2 sense (flat hold vs. the new
`ndc3_conditional_decline` arm) at all — that arm postdates this document
and isn't mentioned; per TASK.md's Part 2 rule ("do not add new methods
material"), that omission is not something to fix now, it goes to the
closing list.

### 1G — other claims that look false, one line each

- None found. This document's §6 ("Bias directions") and §7/§8 ("Open
  method decisions" / "documentary finding") were checked against §2's
  established facts and against the register; nothing else contradicts
  the frozen state.

---

## Synthesis: does the inventory change this task's shape?

**No — roughly a dozen items, not roughly twenty.** Tally of claims this
inventory marks STALE or internally contradictory (excluding the
correctly-framed historical/diagnostic figures, which need no correction):

1. `methodology.md` — total O&M silence (1A), affecting the whole document
2. `methodology.md`:487 — backcast -3.4% (should be -3.2%)
3. `methodology.md`:556–557 — CF sensitivity 20.086/20.856 bn (pre-O&M)
4. `methodology.md`:529–532 — "central configuration" framed as open (likely resolved; confirm before correcting)
5. `methodology_physical_layer.md` — total O&M silence (1A)
6. `methodology_physical_layer.md`:250 — storage 77,387.72/credit MWh (likely stale, needs re-verification, not blind correction)
7. `methodology_physical_layer.md`:511–515 — "cannot convert" (contradicts its own §5.2)
8. `methodology_demand_horizon.md` — total O&M silence (1A)
9. `methodology_demand_horizon.md`:350 — backcast arithmetic (oldest-epoch figure found, implies -4.1%)
10. `methodology_demand_horizon.md`:330–336 — "caps set non-binding after 2045" (contradicts code and contradicts `methodology_genself_backstop.md`'s correct §4.5)
11. `methodology_demand_horizon.md`:528–536 — DIST-5 "untested" (contradicts register L25)
12. `methodology_demand_horizon.md`:538–542 — gas efficiency "not yet implemented" (possibly = conversion, lower confidence)
13. `methodology_genself_backstop.md` — total O&M silence (1A); no other findings in this file

Plus the epoch stamp itself (all four documents, per Part 2's rule) and
the L21/L22/L24/L25/L26 register renumbering, which none of the four
documents cite either way (1E), so needs no citation fix — only
matters if Part 2 chooses to *add* a citation, which is new material and
out of scope per TASK.md.

**Recommendation: proceed with Part 2 as a single task when released.**
Item counts 1/5/8/13 (the O&M silence) are really one finding repeated
per-file, not four independent investigations, and Part 2's own rule
already specifies exactly what the O&M correction looks like in each
document (epoch stamp + O&M term added where the objective is described).
Items 6 and 12 need a short verification step before correction (not a
blind edit) — flagged above, not blocking release.

---

## Closing list — encountered but not asked about

- `methodology.md` §8 lists "**Two tenors for one asset**" as two
  separate, overlapping bullets (lines 547–554 and 560–564) under the
  same heading. Not itself a stale claim — both versions are accurate —
  but exactly the "two accounts to reconcile" problem Part 2's rules warn
  against, caused by document editing rather than by any of the events
  this task is about (O&M, cap trajectory, register renumbering). Not
  corrected here since it's outside Part 1's inventory categories and
  outside Part 2's "correction only" scope as TASK.md frames it (it's not
  a *stale* claim, it's a *duplicate* one) — flagged for whoever next
  edits that section.
- None of the four documents mention the `ndc3_conditional_decline` cap
  arm, `--include-decline-arm`, or the net-zero-2060 anchor (plan 4.2,
  completed per TASK.md's header). Per Part 2's explicit rule ("do not
  add new methods material"), this is not something to add now — noted
  here as the most obvious candidate for the *next* methods task, since
  §5 of `methodology_demand_horizon.md` (scenario coherence) and §3.3
  (build-rate constraints) are the natural homes for it.
- Similarly, none of the four documents mention the O&M implementation as
  a *method* (sourcing, three cost cases, the storage USD/kW-yr → USD/MWh-yr
  conversion via `storage_duration_hours`) beyond what Part 2 will add as
  a stamped correction. A fuller methods treatment (why NREL ATB US rates
  were used unscaled, the circularity argument against scaling FOM to
  Nigerian gas availability — register L22's reasoning) is new material,
  not a correction, and belongs in a future task.
- No file in this inventory cites a register entry by number (1E, all
  four files) — worth noting since it means the L16/L18 renumbering work
  done in the prior task required zero downstream fixes in the
  methodology documents, which is a clean result rather than an absence
  of checking.
- `methodology.md` lines 63–65 (storage divergence "14% at 2045, 42% at
  2050, 70% at 2054" between two arms) was checked against §2's
  established facts and found **neither confirmed nor contradicted** — no
  established figure exists to check it against. Not marked STALE (no
  evidence it's wrong) and not marked CURRENT (not verified) — flagged as
  a residual uncertainty rather than a finding.
- Both `methodology.md`:200 (physical_layer.md, "raised baseline system
  NPV by roughly a third") and the O&M figure (+30.9%, established fact)
  describe unrelated historical events that happen to share the same
  rough magnitude. Not an error — just close enough to invite confusion
  if a reader conflates the firm-capacity-constraint effect with the O&M
  effect. Noted in case Part 2's O&M addition benefits from
  distinguishing the two explicitly.

**Then STOP, per TASK.md. Part 2 is not started. Awaiting release.**

---

## PART 2 — CORRECTIONS

Released and complete. Edits made only to the four whitelisted files —
`git diff --stat`: `methodology.md` +49/-15, `methodology_demand_horizon.md`
+49/-16, `methodology_physical_layer.md` +39/-11,
`methodology_genself_backstop.md` +5/-0 (100 insertions, 42 deletions
total). No `src/`, no scripts, no solver, no commit, no staging.

### (a) O&M absent from all four documents

**Decision on scope.** Added the O&M description to **`methodology.md`
only**. Reasoning: per Part 1's 1A findings, `methodology_physical_layer.md`
and `methodology_genself_backstop.md` explicitly declare the objective/
capital-charge topic out of scope for themselves ("written elsewhere") and
never attempt to describe system cost — there is no natural seam to add O&M
to without introducing a topic those documents disclaim. `methodology_demand_horizon.md`
§4 names `system_cost_npv` as "the optimisation objective" but only to
discuss its *horizon and salvage basis* (§4's actual subject), not to
enumerate its cost terms — adding a full O&M breakdown there would either
duplicate `methodology.md`'s account (the "two accounts to reconcile"
problem Rule 1 forbids) or be a truncated, confusing partial repeat. Only
`methodology.md` §4.2 has an actual cost-term passage (the capital charge
formula) to correct against.

**Edit.** `docs/methodology.md`, §4.2, immediately after the "Coverage
cushion" paragraph (old lines 206–213) and before "### 4.3 Concessional
envelope." New paragraph inserted, not touching the capital_charge
formula itself:

> **Operating and maintenance, added at the same objective.** The capital
> charge above is not the only cost `system_cost_npv` carries. Fixed and
> variable operation-and-maintenance costs are added for every technology:
> solar FOM on cumulative installed capacity (public and EaaS, same rate);
> storage FOM on cumulative installed energy capacity, converted from
> USD/kW-yr to USD/MWh-yr by dividing by `storage_duration_hours`; gas FOM
> on cumulative installed capacity, split by conversion state — the
> converted (combined-cycle) rate for converted plant, the unconverted
> (open-cycle) rate for the rest; and gas variable O&M on generation, same
> split. Solar and storage variable O&M are zero. All four are discounted
> on the same basis as every other operating cost. Sourced from NREL ATB
> 2024 (`Summary_FOM` and the per-technology sheets), three cost cases
> (`om_low`/`om_central`/`om_high` = Advanced/Moderate/Conservative). At the
> central configuration this adds **6.602 bn (30.9%)** to the objective —
> the single largest correction applied to this model since the tranche
> formulation itself.

### (b) `methodology.md:487` — backcast

**Old:** "The current central configuration reproduces it to within -3.4%."
**New:** "The current central configuration reproduces it to within
**-3.2%** (previously -3.4%, before the O&M implementation at `d7538ff`)."
(§7.5, now at line ~509.)

### (c) `methodology_demand_horizon.md:350` — backcast arithmetic

**Decision: reframed as historical, not updated to current figures.**
Reasoning: the passage's gas/hydro/existing-solar split (24.57 / 11.47 /
0.88 TWh) is a component breakdown, not just a headline percentage: I have
the current headline (-3.2%) from §2's established facts but not the
current component split behind it, and Rule "do not invent a number I
cannot establish" applies to the components even though the headline is
known. Reframing preserves the historical arithmetic accurately and states
plainly that it does not reproduce, without inventing replacement
component figures.

**Old:** "The backcast arithmetic shows why 2025 = 0 is correct: gas 24.57
+ hydro 11.47 + existing solar 0.88 ≈ 36.9 TWh against 37.09 TWh of demand.
Essentially no new solar is required in the base year. A model that builds
capacity it does not need is the distorted one."

**New:** "**At the epoch this floor-removal test was run** (before the
min-build floor was removed, and several corrections before the frozen
state), the backcast arithmetic showed why 2025 = 0 is correct: gas 24.57
+ hydro 11.47 + existing solar 0.88 ≈ 36.9 TWh against 37.09 TWh of demand
— essentially no new solar required in the base year. These component
figures are historical and do not reproduce at the frozen state (current
backcast is -3.2%, not the -4.1% this arithmetic implies); the qualitative
conclusion they supported — a model that builds capacity it does not need
is the distorted one, so the floor was wrong to force 100 MW — does not
depend on the exact split and is unaffected."

### (d) `methodology.md:556-557` — CF sensitivity

**Old (in §8):** "**Solar sits near a cost knife-edge in the corrected
model.** A lower capacity factor produces a CHEAPER system (0.16 -> 20.086
bn against 0.19 -> 20.856). Single-variable and reproducible, so it is
real. It implies solar is cost-increasing at the margin — built because
something forces it rather than because it pays. Not diagnosed."

**New:** "**Solar sits near a cost knife-edge in the corrected model.** A
lower capacity factor produces a CHEAPER system (0.16 -> 20.086 bn against
0.19 -> 20.856; **pre-O&M figures, `d7538ff`** — O&M adds +30.9% at central
and has not been re-measured across the CF arms). Single-variable and
reproducible, so it is real. It implies solar is cost-increasing at the
margin — built because something forces it rather than because it pays.
Not diagnosed."

### (e) `methodology_physical_layer.md:511-515`, §7 item 9 — conversion vs. rehabilitation

**Old:** "9. **Retrofit of existing open-cycle plant to combined cycle is
absent from the choice set.** The model can build new combined cycle but
cannot convert or rehabilitate what exists, so a cost-optimality claim over
abatement options is incomplete. Discussed but not implemented in this
work."

**New:** "9. **Rehabilitation of the existing fleet at fixed technology
mix is absent from the choice set** (§5.2). Conversion (open-cycle to
combined-cycle) is no longer part of this gap — it is implemented and
swept, per §5.2 above, and worth 5.91 bn (the second-largest lever in the
model). Rehabilitation remains genuinely absent: its ceiling is a
composition-adjusted efficiency expectation derived with unsourced
indicative efficiencies, declared out of scope pending sourcing. A
cost-optimality claim over abatement options is therefore incomplete with
respect to rehabilitation, not conversion."

Preserved the "cannot rehabilitate" claim exactly as instructed; corrected
only the "cannot convert" half.

### (f) `methodology_demand_horizon.md:528-536` — DIST-5 "untested"

**Old:** "**Distributed solar capital cost and capacity factor.** The
distributed arm currently pays utility-scale capital cost multiplied by a
financing margin, which is a financing premium standing in for a
technology cost. Sourced ATB figures give a distributed premium rising
from roughly 16% to 37% over the horizon, and on an energy basis the gap
is larger still. Whether the distributed arm's advantages survive a
properly sourced premium is untested. The capacity factor must not be
taken from the same comparison without care, since the utility reference
class uses tracking while this model's Nigerian capacity factor is already
fixed-tilt."

**New:** "**Distributed solar capital cost and capacity factor —
RESOLVED, register L25.** The distributed premium and capacity factor
(DIST-5) are now implemented and sourced from Global Solar Atlas, and the
question this item posed has a measured answer: DIST-5 changes NDC-capped
objectives by under 0.03% and leaves EaaS deployment identical to the
megawatt — the corrections are correctly implemented and inframarginal.
That measurement was taken pre-O&M, on capped objectives of 26-43 bn;
those objectives are now 51-63 bn post-O&M, so the 0.03% figure is an
upper bound on the current effect, not a re-measured one. See register L25
for the full account."

(Dropped the capacity-factor "taken from the same comparison" caution
since it was a caveat on the pre-implementation state, now superseded by
the measured result; the underlying siting/tracking distinction it
protected is still correctly stated in `methodology.md` §5.3 and register
L25/L26, so nothing is lost.)

---

### Verification items (g), (h), (i)

### (g) `methodology_physical_layer.md:250` — storage `77,387.72 / credit` MWh

**Verification performed.** Frozen central storage is 65,613.0 MWh at
`storage_cc_central` (credit 0.80), per §2's established facts. Solved
`77,387.72 / credit = 65,613.0` for `credit`: **≈1.18** — not 0.50, 0.80,
or 1.00, the only three values `storage_cc_case` has ever taken in this
project. **Verdict: does NOT reproduce at the frozen state.** No original
run configuration is recorded for the 77,387.72 measurement, and several
structural changes intervene (gas retirement proxy → vintage arms, the
hydro seasonal correction, the F1 per-tranche capital treatment, O&M) —
any could have moved the firm-capacity residual this figure is derived
from, and there is no way to attribute the gap to one of them from the
document alone. **No replacement number invented.**

**Old:** "The measured signature is exact: across the three credit arms
the build came out at precisely `77,387.72 / credit` MWh, with duals
scaling identically. **Storage composition (that it is displaced by gas,
and at what ratio) is a model result; storage magnitude is not.**"

**New:** "**The measured signature — build came out at precisely
`77,387.72 / credit` MWh, with duals scaling identically — was measured at
an unrecorded epoch and does not reproduce at the frozen state.** Checked
directly: the frozen central configuration (`storage_cc_central`, credit
0.80) gives storage 65,613.0 MWh (`docs/findings_frozen.md`); the formula
above would require credit ≈ 1.18, a value this project has never used.
Multiple structural changes separate the two measurements — gas retirement
moved from a flat proxy to vintage arms, the hydro seasonal factor was
corrected, the F1 per-tranche capital treatment replaced the overnight
formulation, and O&M was added — and which of them broke the exact-division
signature cannot be established without the original run's configuration,
which was not recorded. No replacement figure is invented here (the L8
treatment: an honest "not datable" beats a guessed number). **Storage
composition (that it is displaced by gas, and at what ratio) is a model
result; storage magnitude is not** — this qualitative claim does not
depend on the stale figure and is unaffected."

### (h) `methodology_demand_horizon.md:538-542` — gas fleet efficiency "not yet implemented"

**Verification performed.** Read `methodology_physical_layer.md` §5.2 and
`docs/phase_2_8_conversion.md`. Two findings determine the verdict:

1. `phase_2_8_conversion.md` line 4: "Conversion ONLY; rehabilitation is
   out of scope" — the implementation deliberately excluded rehabilitation.
2. The passage under review describes efficiency gains achievable "with
   **no new infrastructure**." Conversion is explicitly the opposite of
   that: `methodology_physical_layer.md` §5.2 calls it "a physical addition
   of plant" (a bottoming cycle). Rehabilitation, by contrast, is defined
   in the same section as closing the degradation gap "**at fixed
   technology mix**" — i.e., without adding plant. The "no new
   infrastructure" phrase matches rehabilitation's own definition and
   contradicts conversion's.
3. Cross-checked `merged_correction_plan_v2.md`'s "PROPOSED ADDITION —
   gas fleet efficiency as an investment decision" (still `[OPEN]`
   `[SHOULD]`), which is very likely this passage's source: it originally
   bundled "Rehabilitation and combined-cycle conversion" together under
   one "least new-infrastructure dependency" framing. Conversion was later
   split out and implemented (Phase 2.8); rehabilitation was explicitly
   carved out and left undone.

**Verdict: the passage is about rehabilitation, not conversion. It is
still accurate. Left unchanged**, per TASK.md's instruction for this
branch of the decision tree. No edit made to
`methodology_demand_horizon.md:538-542`.

### (i) `methodology.md:529-532`, §8 — central configuration

**Verification performed.** Read `docs/findings_frozen.md`, "The central
configuration" (lines 17–48). It declares a full named configuration
(`conv_central`, `hydro_flat`, `gas_deliverability_case baseline`,
`cr_central`, `dscr_central`, `env_dares`, `tariff_central`, `f1_tranche`,
`om_central`, plus the peak/storage-cc/gas-build/gas-availability/
gas-retirement/gas-capex/ccgt-efficiency/us-inflation arms), run at HEAD
`2a9226b`, with the exact figures §2 established (27.954 bn / 65,613.0 MWh
/ 41,031.4 MW / -3.2%). **Verdict: CONFIRMED — declared and run.**

**Old:** "**Central configuration for the reported result set.** Most
existing results were produced before conversion and hydro arms existed,
and under the unsourced coverage cushion. The headline set requires
re-running at a single declared central configuration."

**New:** "**Central configuration for the reported result set —
RESOLVED.** A single central configuration is now declared and run at HEAD
`2a9226b`: `docs/findings_frozen.md`, "The central configuration." Central
27.954 bn, storage 65,613.0 MWh, solar 41,031.4 MW, backcast -3.2%."

---

### (j) `methodology_demand_horizon.md:330-336` — emissions caps "non-binding after 2045"

**Old:** "- **Emissions caps are set non-binding after 2045.** NDC 3.0
provides no target beyond 2035, so 2036–2045 is already an apportionment
construction; extending it further would stack extrapolation on
extrapolation, and a binding cap in 2050 would change what the model
builds in 2043."

**New:** "- **Emissions caps are held flat at the terminal in-window value
through the buffer**, not released non-binding as this document previously
stated. NDC 3.0 provides no target beyond 2035, so 2036–2045 is already an
apportionment construction, and the buffer years extend that construction
further rather than extrapolating past it. See
`docs/methodology_genself_backstop.md` §4.5 for the full account — a
non-binding buffer became an arbitrage sink once the self-generation
backstop existed (defect D10), which is why this differs from the other
buffer-year treatments listed above. Not re-derived here; that document is
the canonical statement."

Per instruction, did **not** paraphrase §4.5's account into a second
independent description here — only enough is restated to correct the
false claim and explain why this item differs from its neighbours in the
same list (the other buffer-year bullets genuinely do extrapolate/hold
flat for reasons specific to CAPEX and deployment caps; this one has a
different, D10-specific reason). `docs/methodology_genself_backstop.md`
§4.5 — **no change made**, confirmed already correct.

### (k) `methodology.md:547-554` and `:560-564` — "Two tenors for one asset" duplicate

**Merged, kept the fuller version** (the one naming the retired blended
rate and the t=0 divergence, matching register L17). The shorter version
was additionally mis-nested as a sub-bullet under "Midstream investment
cost" (two leading spaces before its `-`, unlike every other item in the
list) — removing it also fixes that formatting defect.

**Old (two bullets):**
```
  - **Two tenors for one asset.** The objective amortises commercial
  capital over a sourced commercial debt tenor; the bankability
  constraint discounts revenue to the model horizon. The same asset
  therefore carries a short tenor in the capital charge and a long one in
  the admissibility test, inflating bankable revenue for early vintages.
  The coherent treatment is to discount over the same service window the
  capital charge already uses. Identified, not yet corrected; any
  bankability-dependent result is provisional on it.
...
- **Two tenors for one asset.** The objective amortises commercial
  capital over the sourced debt tenor; the bankability constraint
  discounts revenue to the model horizon at a retired blended rate. The
  errors partly cancel at the first vintage and diverge thereafter.
  Identified, declared, not corrected; any bankability-dependent result
  is provisional on it.
```

**New (one bullet, at top level, in its original second position):**
```
- **Two tenors for one asset.** The objective amortises commercial
  capital over the sourced debt tenor; the bankability constraint
  discounts revenue to the model horizon at a retired blended rate. The
  errors partly cancel at the first vintage and diverge thereafter.
  Identified, declared, not corrected; any bankability-dependent result
  is provisional on it.
```

### (l) Epoch stamp — all four documents

Identical one-line-plus-context stamp added near the head of each file,
immediately after the title:

> **Current as of HEAD `2a9226b`.** Last substantive model change: the
> operating-and-maintenance implementation (`d7538ff`), which added
> +6.602 bn (+30.9%) to the central objective. Figures elsewhere in this
> document not explicitly marked otherwise are stated at this state.

Applied to `methodology.md`, `methodology_physical_layer.md`,
`methodology_demand_horizon.md`, `methodology_genself_backstop.md`.

---

## Closing list — encountered but not asked about

- Item (a)'s scope decision (O&M added to `methodology.md` only, not all
  four) is a judgment call, not something TASK.md resolved explicitly —
  flagged prominently in (a) above rather than buried, in case Zeus wants
  O&M mentioned in `methodology_demand_horizon.md`'s §4 cost-reporting
  table too (as a one-line cross-reference rather than a duplicate
  breakdown, which would be the way to do it without violating Rule 1).
- Item (f)'s edit dropped one sentence (the capacity-factor/tracking
  caution) rather than only appending a resolution — noted inline in (f)
  above with the reasoning; flagging again here since TASK.md's instruction
  for (f) didn't explicitly authorize removing text, only correcting the
  "untested" claim. The dropped caution's substance is preserved elsewhere
  (`methodology.md` §5.3, register L25/L26), so no information was lost,
  but it's a slightly larger edit than a minimal word-swap.
- Part 1's tentative, lower-confidence note that `methodology.md`:63-65
  (storage divergence "14%/42%/70%") might also be stale was **not**
  investigated further in Part 2 — it wasn't one of the twelve items
  TASK.md released for correction, and no established fact in §2 gives a
  way to check it. Left as-is, per "do not re-inventory."
- The `ndc3_conditional_decline` cap arm (plan 4.2) is still not mentioned
  in any of the four documents — correctly so, per Rule 3 ("do not add new
  methods material"), but repeating the Part 1 note here since it's the
  most obvious next-task candidate for these same four files.
- No solver, script, or `src/` file was read for any purpose beyond the
  two read-only checks TASK.md explicitly authorized (the code inspection
  behind item (j), done in Part 1, and the cross-file reads for (g), (h),
  (i) in this part). Nothing was run.

**Then STOP.** Zeus reviews `git diff` and commits.
