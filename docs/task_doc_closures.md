# Three documentation closures — report

Branch `phase-fin2-d17`. No `src/` file, runner, or solver touched. No git
staging or commit performed — Zeus reviews `git diff` and commits.

Edits made, both within the permitted whitelist:
- `docs/merged_correction_plan_v2.md` — three sections (6.2, DO-NOT-REPORT
  list, 4.4).
- `docs/limitations_register.md` — **not edited.** Nothing in Parts 1–3
  required a register edit; where the register was relevant it was read and
  cited, not changed. (One gap it does *not* cover is flagged below, under
  "Encountered but not asked about" — not added, since adding a new
  limitation entry wasn't asked for.)

---

## PART 1 — plan 6.2

**Item quoted in full, as it stood before this edit:**

> **6.2 `[OPEN]` `[MUST]` Rename "reliability" to what it is.**
> An annual unserved-energy fraction is **energy adequacy**, not reliability — no LOLE,
> no LOLP, no hourly basis. Rename throughout, or an engineering examiner will.

**Edit.** `docs/merged_correction_plan_v2.md`, item 6.2 (Phase 6). Marked
`[DONE]`. Body rewritten to record: the implementation actually chose "grid
service obligation," not "energy adequacy"; why "energy adequacy" is rejected
for the same reason "reliability" was (both are synonyms for LOLE/LOLP-style
capacity adequacy, and the quantity in question is not that); and the
provenance of the reasoning, `docs/methodology_genself_backstop.md` §5.1. Full
new text is in the file; summary of the substantive claims:

- Renamed to `max_offgrid_fraction` / `grid_service_mode` /
  `m.grid_service_obligation`, bounding `selfgen + unserved` (grid service
  shortfall), not capacity adequacy.
- Quoted §5.1 verbatim: *"Reliability in power-system planning is
  adequacy... A household running its own generator was not served by the
  grid, but no capacity was short. The constraint is a service obligation."*

**Filename note.** TASK.md's brief names the source file
`docs/methodology_selfgen_backstop.md`. The file on disk is
`docs/methodology_genself_backstop.md` — no `methodology_selfgen_backstop.md`
exists anywhere in the tree. Confirmed §5.1 exists at the cited location in
the actual file and its content matches the brief's characterization exactly
(read in full before citing, per standing instruction not to edit off grep
output). Cited the correct filename in the edit.

**Step 3 — search for other "energy adequacy" prescriptions.**

- Within `merged_correction_plan_v2.md`: exactly one occurrence, inside item
  6.2's own original body (now rewritten). No other hit in this file.
- Tree-wide (`grep -rl "energy adequacy" docs/`): one further file,
  `docs/handover_pre_om.md:157` — *"Rename reliability -> energy adequacy.
  Model side is done; the REL runners and prose still carry it."* This is a
  live prescription of the wrong term, in a document `frozen_state_2026_09_10.md`
  itself declares superseded ("read it for the stratigraphy, not for
  numbers"). `handover_pre_om.md` is outside this task's edit whitelist, so
  it was **flagged, not corrected** — noted in item 6.2's new body and
  restated below under "Encountered but not asked about."

No occurrence of "energy adequacy" was found that needed correcting and
wasn't — every hit is accounted for above.

---

## PART 2 — DO-NOT-REPORT re-screen

Full table, current state, all 14 rows (11 original + 3 added), plus the one
row moved to a new **Lifted** subsection:

| # | Item | Classification | Reason now |
|---|---|---|---|
| 1 | `results/monte_carlo` | **STILL BARRED, different reason** | Was "Resolved by 2.6." 2.6 is done, but the producing runner `17_run_monte_carlo` is dead (register L10: calls `load_scenario()` without required post-2.4 keys, raises `TypeError` before any solve) and carries two further declared defects. No output producible at all |
| 2 | All `17_` LCOE figures | STILL BARRED, same reason | Three defects (VoLL numerator, hydro-omitted denominator, storage double-count) unchanged; also now covered by row 1's dead-runner finding |
| 3 | DEM-2 `public_only` unserved>0 cells | STILL BARRED, same reason + epoch note | 35.3 TWh unserved against ~37 TWh Tier-1 demand is far beyond any obligation eps this project uses (0.08–0.20 range), so 2.6's genset backstop would not fully absorb it — screening rule stands. Cited figures (13.6bn→152.7bn, 35.3 TWh) predate `2a9226b` and 2.6; do not quote them |
| 4 | GAS-3 (`08_`) | STILL BARRED, same reason + epoch note | $571B / 152.4 TWh unserved predates `2a9226b` by a wide margin — above even the worst pre-calibration-defect figure recorded elsewhere in this plan (273bn). Oldest-epoch; do not quote |
| 5 | POL-1 `public_only` (`15_`) | STILL BARRED, same reason + epoch note | −$220.75B predates `2a9226b`; do not quote |
| 6 | All pre-recalibration DEM-2 | STILL BARRED, same reason | Self-dated already; historical fact about the recalibration event, unaffected since |
| 7 | `run_reliability.py` output | STILL BARRED, same reason | Wrong-dataframe bug, independent of 2.6/freeze |
| 8 | "48 runs" claim | STILL BARRED, same reason | DEM-2 is 72 runs; unaffected by 2.6/freeze |
| 9 | Notebooks 01–04 | STILL BARRED, same reason | Predate Phase 1.4 recalibration, a fortiori predate `2a9226b` |
| 10 | Buffer-year (2046–2054) results | **STILL BARRED, different reason** | Extrapolated inputs / non-binding caps still hold. Third original clause — "storage retiring with no replacement capex" — is **now false**: register L18 records D6 CLOSED (both builder branches correctly charge rebuild capex, since 2.6/F2). Clause removed; bar stands on the two remaining grounds |
| 11 | `system_cost_npv` as headline | STILL BARRED, same reason | Still spans 2025–2054, still an objective, not the financing figure |
| 12 | `nirp_trajectory` solar arm NPV | **CONDITION SATISFIED — lifted** (see below) | — |
| 13 | Backcast quoted as −4.1% | **STILL BARRED, figure changed again** | −3.4% (already a correction) is itself superseded: frozen central now measures **−3.2%** (`frozen_state_2026_09_10.md` §1). Neither −4.1% nor −3.4% is current |
| 14 | Pre-O&M objectives quoted as current | **NEW ROW, added** | Every $ figure before `d7538ff` is superseded (~31% objective shift). Covers `handover_pre_om.md`'s 20.99bn/20.86bn anchors and all of `findings_provisional_pre_om.md` |
| 15 | F-11 leakage percentages (34.2–45.0%) | **NEW ROW, added** | `leakage` field is null in current cap summaries; pre-O&M figures not restated. Structural claim stands; magnitude does not, until re-measured (`findings_frozen.md` F-11) |
| 16 | "37 arms across four sections" | **NEW ROW, added** | Sweep is A+B+C, 38 invocations, no Section D. Traced the false claim to its source: `handover_pre_om.md:57` |

**Lifted (new subsection, table moved out of the barred list, not deleted):**

| Item | Barred because | Lifted because |
|---|---|---|
| `nirp_trajectory` solar arm — NPV | Cumulative solar cap binds 2027–2037 (shortfall peaks 4,505 MW, 2032), shortfall priced at VoLL. In-row condition: "cost is not [reportable], until 2.6" | 2.6 added `selfgen` to both builders as a priced backstop, "replacing an uncosted VoLL cliff with a diesel-priced slope" (`phase_2_6_handoff.md`) — the exact mechanism named. **Caveat carried forward:** this is a documentation judgment from the 2.6 architecture, not a fresh solve (none run here). Before quoting a number, confirm at solve time that `selfgen + unserved` stays within the obligation bound in every year 2027–2037 |

**Judgment calls worth surfacing explicitly:**

1. **Why row 1 (Monte Carlo) and row 12 (`nirp_trajectory`) were treated
   differently** despite both citing an explicit 2.6-condition. Row 1's
   *mechanism* was resolved by 2.6 (VoLL cliff → backstop) but the *runner*
   that would exercise it is independently dead (L10) — a second, unrelated
   defect that 2.6 cannot fix. Row 12 has no such second defect: the runner
   isn't dead, and the specific shortfall named (4,505 MW cumulative, one
   arm) is plausibly within an obligation bound, unlike rows 3/4/5 whose
   shortfalls (35.3 TWh, 152.4 TWh) are not.
2. **Rows 3, 4, 5 were NOT lifted** even though the general "VoLL cliff →
   backstop" fix in principle applies to them too, because their historical
   shortfall magnitudes are large enough (a third to all of Tier-1 demand)
   that no obligation eps used anywhere in this project would keep them
   inside the backstop's bound — some genuine VoLL-priced residual is still
   likely. This is reasoned from magnitude, not confirmed by a solve (barred
   under this task). Flagged as the most defensible place a future,
   solver-permitted pass should look first.
3. **The general "Fixed by 2.6" note under the table** was rewritten rather
   than left or deleted — it previously read as a blanket claim; it now says
   explicitly that 2.6 fixed a *mechanism*, and names which one row that has
   actually been shown to resolve for, rather than implying the whole table
   clears at once.

---

## PART 3 — plan 4.4

**Edit.** `docs/merged_correction_plan_v2.md`, item 4.4 (Phase 4). Marked
`[DONE] as an analytical bound`. Verified, did not re-derive:

    genset_central / tariff_central = 445 / 95  = 4.6842...  → 4.68x
    genset_srmc    / tariff_central = 277 / 95  = 2.9158...  → 2.92x
    genset_srmc    / tariff_high    = 277 / 115 = 2.4087...  → 2.41x

All three match TASK.md's given values. Cross-checked the input figures
against their cited sources rather than trusting the brief blindly:

- `genset_central`=445, `genset_srmc`=277, `genset_high`=3,323 USD/MWh,
  with the eta/dataset/fuel-window provenance given — confirmed verbatim in
  `docs/phase_2_6_handoff.md` (lines 18-27) and again at
  `docs/merged_correction_plan_v2.md:614-615`.
- `tariff_central`=95, `tariff_high`=115 USD/MWh — confirmed in
  `docs/tariff_arm.md:248-249`.

Recorded in the plan: the three headroom ratios, the conclusion (tariff sits
2.4x–4.7x below sourced avoided cost across every arm), the willingness-to-pay
scope limit, and the one-line exclusion of `genset_high` (A2EI-metered
outlier, eta 0.0511 — not representative of ordinary self-generation
operation).

**Discrepancy found and corrected, not just copied.** TASK.md's brief cites
the collection caveat as "(DIST-3 is unimplemented, register L16)." Checked
both claims before writing them into the plan:

- `docs/limitations_register.md` has **no** entry discussing collection or
  DIST-3 anywhere (`grep -n "collection\|DIST-3" docs/limitations_register.md`
  returns nothing) — L16 (in both its current halves) covers only DIST-2 and
  DIST-5. The citation to "register L16" is wrong.
- DIST-3 itself exists, but in `merged_correction_plan_v2.md` (the
  "DISTRIBUTED EaaS" section, "folds into 3.3"), not in the register, and it
  is a claim about counterparty differentiation ("collection risk applies to
  grid-served revenue, not behind-the-meter EaaS"), not a statement that
  collection itself is modelled for either channel — `disco_collection_rate`
  is an unused stub (item 3.3) regardless of channel.

The 4.4 edit cites DIST-3 correctly, in-plan, rather than repeating the false
register citation. The underlying gap (no register entry names "EaaS has no
collection mechanism, and the DIST-3 counterparty-differentiation claim is
asserted rather than modelled") is real and currently undocumented in the
register — flagged below, not added, since PART 3's three steps didn't ask
for a new register entry and I did not treat "the whitelist permits editing
this file" as license to add scope beyond what was asked.

---

## PART 4 — For the limitations chapter

*(liftable prose, as requested)*

This project maintains a do-not-report list alongside its results: a table,
kept in the same document as the correction plan, naming every figure,
runner, or claim that is known to be invalid and must not enter a chapter,
slide, or briefing until its defect is fixed. The list exists because a
model under active correction produces two kinds of output that look
identical from the outside — numbers generated by the system working, and
numbers generated by the system failing — and only the person who fixed the
defect can currently tell them apart. Without a written record, that
knowledge lives in one person's memory and decays with time; a chapter
drafted eight weeks after a defect was found and forgotten will cite the
broken figure with the same confidence as a correct one, because nothing on
the page distinguishes them.

Four broad classes of defect put a figure on this list. The first is a
VoLL-dominated objective: when a scenario's demand cannot physically be met
by the technologies the model is allowed to build, the shortfall is priced
at the value of lost load, a number chosen to represent the cost of an
outage, not the cost of a plan — a cost dominated by that penalty says
something about infeasibility, not about the policy being evaluated. The
second is a defective metric denominator or numerator: a percentage or a
levelized cost computed with a term double-counted, omitted, or wrongly
signed produces a number that is precise and wrong in the same breath,
which is more dangerous than a number that is obviously absent. The third
is a superseded epoch: this model has been recalibrated and re-anchored
several times as errors were found and data was sourced, and each
recalibration moved every downstream objective — sometimes by single
digits, once by an order of magnitude. A figure computed correctly under an
earlier configuration is still wrong once quoted against a later one,
without a single line of code changing. The fourth is a dead runner: a
script that no longer executes, whether because an interface it depends on
changed underneath it or because a bug corrupted its output silently, can
leave behind results that look complete while having no defensible claim to
correctness going forward.

Entries are lifted when their condition is satisfied, but they are moved to
a record, never deleted. This matters for two reasons. First, an entry that
disappears leaves no trace that the underlying number was ever suspect,
which invites a reader — or a future version of the author — to re-derive
the same false confidence from scratch. Second, the record of what was once
unreportable, and exactly what resolved it, is itself part of the audit
trail a thesis under active correction needs: it demonstrates that
"resolved" was a judgment made and dated against a specific fix, not an
assumption that time alone made a number safe. A screening list that only
ever grows, and is checked before every claim, is slower to work with than
trusting memory — but the alternative is a chapter that cannot distinguish
a result from an artefact, which is the more expensive failure by far.

---

## Encountered but not asked about

- `docs/limitations_register.md` L23 ("F1 ships two capital tranches...") is
  duplicated **verbatim**, in full, twice in a row (roughly lines 922–1013
  and 1014–1105). Not fixed — outside this task's three parts, and fixing it
  would be a substantive register edit not requested.
- `docs/limitations_register.md` reuses the number L16 twice for different
  content (the original "EaaS and public solar are physically identical" and
  a later "SUPERSEDED IN PART: DIST-5 is implemented" entry) and reuses L18
  twice similarly (the D6/storage-guard closure, and "Solar capacity factor
  is a national average"). Numbering collisions, not renumbered — TASK.md
  explicitly says not to renumber around L23, and by extension I did not
  touch other register numbering either.
- `docs/handover_pre_om.md:157` still prescribes "Rename reliability ->
  energy adequacy" — the wrong term, in a document already marked superseded
  elsewhere. Outside the edit whitelist; flagged in the 6.2 edit and here,
  not corrected in place.
- `docs/handover_pre_om.md:57` is the source of the false "37 arms" /
  "four sections" claim the new DO-NOT-REPORT row warns against. Same file,
  same reason, not corrected in place.
- TASK.md's own brief contains two small inaccuracies against the actual
  tree, both corrected in the edits rather than propagated: the methodology
  file is `methodology_genself_backstop.md`, not
  `methodology_selfgen_backstop.md`; and the DIST-3/collection citation
  should point at the plan's DIST-3 item, not "register L16" (register L16
  doesn't discuss it).
- The register has no entry for "EaaS carries no collection mechanism, and
  the DIST-3 counterparty-differentiation claim is asserted, not modelled" —
  a real, currently-undocumented gap surfaced while verifying Part 3. Not
  added, since Part 3 asked for three specific things in the plan document,
  not a new register entry.
- Rows 3–5 of the DO-NOT-REPORT table (DEM-2 unserved cells, GAS-3, POL-1)
  are plausible — but unconfirmed — candidates for lifting once a solver run
  is permitted, on the same "VoLL cliff → genset backstop" logic that lifted
  the `nirp_trajectory` row. Left barred here on a magnitude argument, not a
  measurement, per this task's no-solver constraint.
