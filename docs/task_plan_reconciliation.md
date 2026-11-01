# Task — reconcile `merged_correction_plan_v2.md` with what is actually built

Scope per TASK.md: read-only against `src/`, scripts, register, and audit;
the only file edited is `docs/merged_correction_plan_v2.md`. No code
changed, no script run, no solver invoked, nothing staged or committed.

Every claim below was checked against the current tree, `git log`, or a
named doc — not accepted from TASK.md's own description. Two of TASK.md's
own premises turned out to be *incomplete* on inspection (3.5, D1's
register status) — recorded below where found, per CLAUDE.md's "accuracy
over agreement."

---

## PART 1 — items Chat believes are now complete

**0.3 — metric formula fixes.** Checked `docs/metric_fixes_0_3.md` in full
and `src/optimize_experiments.py` (lines 3168, 2253/2261/2289/2301,
2591-2649) directly. All three named sub-fixes present: solar share
renamed `solar_share_of_grid_generation` (hydro/storage/selfgen excluded
from the denominator); `gas_scarcity_value_usd` deleted, dual-sign clamps
replaced with `DUAL_SIGN_TOL`-guarded assertions; public-capital
multiplier split into `eaas_multiplier_vs_subsidy` /
`eaas_multiplier_vs_public_budget`. Commit `142d437`.
**Verdict: DONE.** Marker changed `[OPEN]` → `[DONE]`.

**0.4 — numerical hygiene.** Checked `docs/numerical_hygiene_0_4.md` in
full, `src/optimize_model.py:1817`, and the solver environment directly
(`highs --version` succeeds; `SolverFactory("highs").available()` and
`SolverFactory("appsi_highs")` both raise `ApplicationError: NotFound`;
`import highspy` raises `ModuleNotFoundError`). Items A (tolerances,
`tee=False`, `solver_version` capture), B (weighted-sum objective
deleted) and C (`1e18` → `None` at 17 sites) are done, commit `08b5972`.
"Move CBC → HiGHS" is not done and cannot be from this environment —
the HiGHS *binary* is present but the Pyomo *bridge* (`highspy`) is not;
`solve_model()`'s default remains `"cbc"`.
**Verdict: PARTIAL, not DONE.** Marker changed `[OPEN]` → `[PARTIAL]`.

**1.5 — hydro trajectory.** Checked `src/scenarios.py:1509-1515`
(`hydro_case_scenarios()` returns exactly `hydro_flat`/`hydro_committed`/
`hydro_ndc`), commit `f43a54c`, and `docs/findings_frozen.md` F-18. The
trajectory sub-item is built exactly as specified (two exogenous arms
plus a flat baseline, all outside the optimiser). The item's *other*
sub-item — DARES/Mission 300 solar procurement — is still `[OPEN]`
(confirmed still in the Sourcing ledger as "❌ blocks 1.5").
**Verdict: PARTIAL overall, unchanged; the hydro-trajectory sub-bullet
verdict is DONE.** Sub-marker changed `[OPEN]` → `[DONE]`; item-level
`[PARTIAL]` unchanged (one of two remaining sub-items still open).

**2.2 — storage.** Read the item's full text first, per TASK.md's own
warning, and found it is a four-part ask, not "wire O&M" alone. Checked:
`src/optimize_model_sliced.py:992` (deployable hours no longer free —
done); `src/io.py:270` and commit `d7538ff` (O&M wired, sourced NREL ATB
2024 — done); `scripts/14_run_str1_storage_role.py` head (still opens
`STATUS: SUSPENDED` — STR-1 not rebuilt); `src/scenarios.py:2718`
(`_storage_duration_hours = 4.0`, still a fixed constant, not a decision
variable).
**Verdict: PARTIAL — two of four sub-requirements done, two not.** Do
not round up to DONE. Marker changed `[OPEN]` → `[PARTIAL]`.

**2.3 — firm-capacity constraint.** `git log -S"m.firm_capacity_
constraint"` → commit `d9aaa7d`, message itself states "binds 21/30
years under peak_unsuppressed." Confirmed `reserve_margin`-based logic
in `src/optimize_model.py` (lines 125, 152-155, 327, 996, 1009,
1014, 1019-1020) and matching text in `docs/findings_provisional_pre_
sweep.md:683` ("binds from 2035 onward — 21 of 30 years").
**Verdict: DONE.** Marker changed `[OPEN]` → `[DONE]`.

**2.4 — peak multiple as four arms.** `src/scenarios.py:206-231`,
`peak_multiple_scenarios()` returns exactly four arms
(`peak_suppressed`/`peak_unsuppressed`/`peak_nirp`/`peak_high`), sourced
from Oluwole (2023), DOI 10.17632/pxvdm26rn7.2. Commit `a2717ac`.
**Verdict: DONE.** Marker changed `[OPEN]` → `[DONE]`.

**3.1 — currency/rate basis memo.** Read `docs/currency_and_rate_
basis.md` in full. **TASK.md's framing ("real USD ruled, `us_inflation`
a required key") describes only the convention, not the whole memo.**
The memo's own status line: *"[DONE] for the currency convention.
[OPEN] for the deflator base year and the Fisher conversion of the
financing rates."* §5 (real base year) and §6 (financing-rate basis,
blocks 3.2) are both still open in the memo's own text.
**Verdict: PARTIAL, not DONE — this is a case where the evidence did
not support the claim as framed.** Marker changed `[OPEN]` → `[PARTIAL]`.

**3.2 — tranche formulation.** Confirmed via `src/scenarios.py:1746-1787`
(`capital_recovery_scenarios()`: `cr_central`/`cr_ida21`/`cr_comm_low`/
`cr_comm_high`, concessional + one commercial rate, no guaranteed-
commercial tranche in any form) and register L23 in full (title itself:
"F1 ships two capital tranches; the design specified three"). Guaranteed
commercial tranche blocked on sourcing (PRG headroom / FGN
contingent-liability ceiling, neither published at the needed
granularity).
**Verdict: PARTIAL, exactly as TASK.md claimed.** Marker changed `[OPEN]`
→ `[PARTIAL]`.

**3.5 — freeze placeholders.** TASK.md's claim ("model frozen at
`2a9226b` with both anchors reproducing") is **true but incomplete as a
verdict for this item.** Confirmed the freeze and both anchors
(`docs/frozen_state_2026_09_10.md`). But re-reading 3.5's own text, and
checking the Sourcing ledger inside this same file, found: **"B\*
reconciliation | own derivation, documented | ❌ blocks 3.5"** — still
unresolved. 3.5 asks for three things (reconcile B\*, recompute the
tariff grid against T\*, source every placeholder rate); the freeze
achieves reproducibility, a narrower thing than any of the three.
**Verdict: PARTIAL, not DONE — TASK.md's own premise here was
incomplete; did not accept it at face value.** Marker changed `[OPEN]`
→ `[PARTIAL]`.

**4.1 — cap apportionment.** Register L24 in full: the apportionment
rule is explicitly declared "your methodological construction, not
something in NDC 3.0," with 4.2's decline arm now providing sensitivity.
The "add one alternative" sub-ask is met by a different route than
literally specified — `scripts/00_build_emissions_cap.py` implements NDC
2.0 (BAU-relative) alongside NDC 3.0 (proportional top-down) as two
distinct methodologies, not a grid-EF or IEA-scenario alternative — noted
as a substitution, not a gap.
**Verdict: DONE.** Marker changed `[OPEN]` → `[DONE]`.

**4.2 — post-2035 cap trajectory.** `ndc3_conditional_decline` arm
confirmed in `scripts/00_build_emissions_cap.py:323` and
`scripts/02_run_ndc_caps.py:263,437`. Reported as **F-21** in
`docs/findings_frozen.md:440` — cap-shape sensitivity is reported, with
the (striking) finding that tightening the cap raises national
emissions via leakage to self-generation.
**Verdict: DONE.** Marker changed `[OPEN]` → `[DONE]`.

**5.3 — gas_add moot.** Confirmed `m.gas_add = pyo.Var(...)` at
`src/optimize_model.py:471` (free, not fixed) and F-14 in
`docs/findings_frozen.md:753-766` ("gas capacity FALLS from the 10,947
MW starting fleet"). The item's own premise ("`gas_add` fixed at zero")
is now false for the current model, which makes the item **moot rather
than done** — exactly TASK.md's framing. No dedicated appendix
experiment sweeping the specific high-growth/upside-gas corner named in
the item's text was located, so a referee asking for that exact
demonstration would still not find it on file.
**Verdict: SUPERSEDED (premise falsified), per TASK.md's instruction not
to mark DONE.** Also noted: the live text carried no `[OPEN]` bracket at
all before this edit — a pre-existing discrepancy against TASK.md's own
table, now resolved by giving it an explicit marker.

**5.5 — doc-code parity.** Confirmed `docs/forensic_audit.md` (A1-A59,
59 findings, three phases, 8 fixed / 51 open) is a superset of this
item's scope. Did **not** verify whether each of the six items named in
5.5's own text (√η-symmetric storage losses, `EaaS.py` CRF/tenor unused,
`economics.solar_capex` mislabeling, `apply_access_adjustment` unused,
`storage_charge_limit_years` dead diagnostic, DEM-1 pre-η docstring) maps
to a specific A-numbered finding — flagged as not checked (Part 4).
**Verdict: SUPERSEDED, with an unresolved cross-reference flagged rather
than assumed.** Marker changed `[OPEN]` → `[SUPERSEDED]`.

---

## PART 2 — items needing a judgement

**5.1 — Monte Carlo.** Read `scripts/29_run_parameter_uncertainty.py`'s
docstring and code in full. `17_run_monte_carlo.py` was **not** repaired
— confirmed still dead (register L10, `TypeError` before any solve).
`29_` is a genuinely different instrument: Sobol variance decomposition
over five *continuous* parameters (gas availability + capacity credit,
gas η, solar CF, an O&M blend, conversion capex), explicitly scoped to
PARAMETER uncertainty, not SCENARIO uncertainty — its own docstring
argues this is a better treatment than random scenario sampling, not an
oversight.
- Solar CF: now sampled (`U(0.16,0.20)`, register L26) — closes one item
  from 5.1's "still not sampled" list.
- Infeasible-draw headline: `29_` reports `n_failed`/`failure_fraction`
  with a warn threshold — answers this ask, for parameter draws.
- Independence-sampling objection: **sidestepped, not fixed** — `29_`
  does not sample demand growth or gas regime at all; scenario
  uncertainty is handled via the discrete CONSTRAINT-RELIEF ASYMMETRY
  cross-tab instead. No Monte Carlo mechanism respecting the correlation
  exists.
- Asymmetric-prior objection: **untouched** — `29_` never samples demand
  growth, so the symmetric-normal-on-asymmetric-arms problem is
  unaffected either way.
- Still not sampled: λ, solar/storage CAPEX trajectory, FX/rates,
  collection rate.
**Verdict: SUPERSEDED-in-part**, per TASK.md's own suggested framing.
Marker changed `[PARTIAL]` → `[SUPERSEDED-in-part]`, with the specific
breakdown above written into the item.

**5.4 — verification protocol.** The three-outcome PASS/NEEDS-ARGS/FAIL
harness repair and the regression-anchor discipline are taken as
established from prior work this session (not re-verified from scratch
this task; not in question per TASK.md). Checked directly: CBC
tolerances are pinned (`08b5972`); HiGHS cannot be pinned because it is
uninstalled (`highspy` absent — same finding as 0.4); no data-provenance
manifest or `make all` target was found in the tree.
**Verdict: PARTIAL — outcome-reporting half done, repro-pack half not.**
Marker changed `[OPEN]` → `[PARTIAL]`.

**DISTRIBUTED EaaS.** Read the section in full and checked each
sub-item individually rather than marking the section as a block.
- **DIST-1** (evacuation exemption): OPEN — folds into 1.4, whose
  evacuation-bound sub-item is still `[OPEN][BLOCKED]`.
- **DIST-2** (loss/TLF advantage): OPEN — confirmed still unimplemented;
  2.6's own text states plainly it "remains unimplemented because the
  metering point is declared at the busbar" (register L25).
- **DIST-3** (counterparty differentiation): OPEN — `disco_collection_
  rate` confirmed still an unused 1.0 stub (`src/scenarios.py:3149`).
- **DIST-4** (defend the characterisation): OPEN — the citation exists
  in this plan's own narrative but no formal defence is written into
  3.5 or 6.3, both still open.
- **DIST-5** (distributed CAPEX + CF): **DONE** — register L25 confirms
  both CAPEX (NREL ATB, +16%→+37%) and capacity factor (Global Solar
  Atlas, Kano-sited, 0.19→0.164) are sourced and applied. This also
  makes the section's own "Gap identified" paragraph (asking for exactly
  this CF) stale.
**Verdict: mixed, marked individually — DIST-5 DONE, DIST-1/2/3/4 OPEN.**
Section header marker changed `[OPEN]` → `[PARTIAL]` to reflect the mix;
each sub-item now carries its own marker for the first time.

**6.3 — Discussion/Limitations chapter.** Confirmed register at 44
entries and forensic audit at 59 findings (8 fixed / 51 open) — real,
substantial material. Confirmed no thesis-chapter text exists anywhere
in this repository addressing the liquidity chain, Electricity Act 2023
defence, land-vs-security siting, north–south mismatch, Q1×4
annualisation, or perfect foresight as a drafted chapter.
**Verdict: material assembled, chapter unwritten — stays OPEN.** No
marker change (still `[OPEN]`); annotation added to prevent a future
reader inferring completion from the register's size alone.

---

## PART 3 — the status-marker sweep

**Two specifically-named re-verifications:**

- **1.6 Step 7** ("cap invariance CONFIRMED"). Still true — re-confirmed
  the byte-identical result and the deliverability CSVs' 2026-08-15
  date. Added the new documented constraint: `00_build_emissions_
  cap.py`'s input must be an annual `conv_none` baseline, since the
  fleet gas series is only a complete emissions measure in the builder
  where OCGT-to-CCGT conversion does not exist. **Verdict: DONE,
  unchanged, text extended** as instructed.
- **6.2** ("Rename reliability to what it is"). Read in full. Already
  correctly records that the implementation chose "grid service
  obligation," not "energy adequacy," with the reasoning traced to
  `docs/methodology_genself_backstop.md` §5.1, and already flags the one
  stale occurrence in the superseded `docs/handover_pre_om.md:157`
  without correcting it in place (outside this file's own scope to fix
  that other document). **Verdict: DONE, verified correct, no change
  needed.**

**D6 vs. the DO-NOT-REPORT list — internal inconsistency found and
resolved.** The DEFERRED DEFECTS section's own D6 read `[OPEN][SHOULD]`
while the DO-NOT-REPORT list's buffer-year row, in the *same file*,
already stated "register L18 records D6 CLOSED." Read register L18 in
full: the guard's original fire condition is algebraically impossible
under every lifetime value this project actually supplies — both
capital-treatment branches correctly charge a rebuild at every
in-horizon vintage, storage included. **Verdict: D6 DONE.** Marker
changed `[OPEN]` → `[DONE]`, reconciling the file against itself.

**D1 — found done, unexpectedly.** Not named in TASK.md's Part 1/2 lists,
but caught in the general sweep. `git log -S"task_defect_closure, D1"`
→ commit `9c422e8`; confirmed directly in
`scripts/13_run_dem2_growth_gas_interaction.py:240-292` that one
relative-tolerance classification (1% of `max(|low_prem|,|high_prem|)`)
now drives both `interaction_amplifies` and `interpretation` across
three branches. **Verdict: DONE.** Marker added (was scope-only before)
`[MUST]` → `[DONE][MUST]`.
**A gap surfaced by this check, flagged not fixed:** the same commit
closed D17/D21/D23/D24 too, and its own `docs/task_defect_closure.md`
contains a "For the limitations register" section clearly meant to be
pasted into `docs/limitations_register.md` — grep confirms the register
contains no mention of D1/D17/D21/D23/D24 or of `task_defect_closure.md`
at all. The register itself is out of this task's whitelist, so this is
reported, not fixed.

**D4 — stale cross-reference found and fixed.** D4's script-inventory
list still called `04_run_fin2` "(deprecated)," directly contradicting
this same file's own 2.6/D25 text (already correctly stating `04_` was
built out at `b496d77` and repaired at `ecb7e54`, per `task_audit_fixes_
p3`, A51). Fixed to match — struck the stale characterisation, added the
correction inline, confirmed `12_run_dem1_demand_sensitivity` is
unaffected (still raises `NotImplementedError`).

**D8 vs. register L40 — checked, no conflict, unchanged.** L40 explicitly
states it is "distinct from D8, which concerns the resolution premium
figure itself being stale — both remain true at once." D8 stays `[OPEN]`,
unchanged.

**Every other status marker in the file — checked, verdict "unchanged"
unless noted:**

- 0.1, 0.2 — `[DONE]`, unchanged (stable prior fixes, no evidence of
  regression).
- 1.1, 1.2, 1.3 — `[DONE]`, unchanged.
- 1.4 — `[PARTIAL]`, unchanged (four sub-items DONE, evacuation bound
  still `[OPEN][BLOCKED]`; this is also DIST-1's blocker).
- 1.6 Steps 1-5 — `[DONE]`, unchanged. Step 6 — `[OPEN][SHOULD]`,
  unchanged (Piecewise g still deferred). Step 8 — `[OPEN][MUST]`,
  unchanged (coherence cross-tab still not built; blocked on D7, itself
  still open).
- CONSTRAINT-RELIEF ASYMMETRY — `[OPEN][MUST]`, unchanged. The cross-tab
  itself is fully declared in this file's own text, but integration into
  the actual thesis write-up was not found — same "assembled but
  unwritten" pattern as 6.3, so left open rather than rounded up.
- 2.1 — `[DONE]`, unchanged (shape parameters remain `[SOURCE NEEDED]`,
  which is a caveat inside the DONE item, not a status contradiction).
- 2.5 and all four sub-sections (Salvage, Boundary guard, Min-build
  floor, Max-build cap, Horizon extension, Reporting metric) — `[DONE]`,
  unchanged. Lead times — `[OPEN][SHOULD]`, unchanged (deliberately
  deferred).
- 2.6 and its sub-items — `[DONE]`, unchanged; its own D25 correction
  about `04_run_fin2` (already correct) is what D4 was found to
  contradict, now reconciled.
- 2.7 — `[OPEN][MUST]`, unchanged.
- 3.3 — `[OPEN][MUST]`, unchanged; confirmed `disco_collection_rate`
  still 1.0, an unused stub, same evidence used for DIST-3.
- 3.4 — `[OPEN][MUST]`, unchanged; confirmed `remaining_npv_factor[t]`
  still sums against the model horizon rather than
  `min(tenor, asset_life)` (`src/optimize_model.py:760,792,799`).
- 4.3 — no status marker in the live text (scope marker `[SHOULD]`
  only) — confirmed genuinely not implemented (`tariff_shortfall`/
  `unified_fiscal` absent from `src/`). Flagged as a formatting gap in
  Part 4, not corrected (adding a marker convention is a judgement call
  outside a reconciliation task's remit — flagging, not deciding, felt
  the more defensible action here).
- 4.4 — `[DONE]`, unchanged.
- 4.5 — no status marker (`[STRETCH]` only) — confirmed not built;
  same formatting-gap flag as 4.3.
- 5.2 — no status marker (`[SHOULD]` only) — partially addressed as a
  side effect of `29_`'s joint sampling of η, solar CF and gas
  availability, which is close to but not identical to the "combined
  physical-parameter envelope run" this item asks for (that item wants
  joint *pessimistic vs optimistic* corner runs, not a continuous Sobol
  sweep). Not marked DONE; noted only.
- 6.1, 6.4 — `[DONE]`, unchanged.
- TECHNOLOGY SCOPE, PROPOSED ADDITION, SECOND PRIORITY, DECLINED table —
  unchanged; these are scope declarations, not fix items, and none
  showed evidence of having been implemented.
- DO-NOT-REPORT LIST — checked in full; internally consistent as written
  (its own D6 note is what caught the D6 inconsistency above). No row
  needed a change.
- D2, D3, D5, D7 — unchanged. D2 (`17_` infeasibility count) stays open
  since `17_` remains dead code. D3 (`.gitattributes`) stays open,
  deliberately deferred. D5 stays `[DONE]`, unchanged. D7 stays open
  (solar-build axis not added to `docs/scenario_coherence.md`).
- Sourcing ledger — read in full; this is a parallel ✅/⚠/❌ tracking
  device, not a `[DONE]`/`[OPEN]` marker list, so not swept item-by-item
  against the same convention. It was, however, load-bearing evidence
  for two verdicts above (3.5's B\* row, 1.5's DARES/Mission 300 row)
  and both were found still accurate.
- Working notes — one `[DONE]` marker ("Protect 2.1... `[DONE]`"),
  unchanged and still accurate.
- IMMEDIATE NEXT ACTIONS — items 1-2 already struck through/`[DONE]`,
  unchanged. Item 3 (2.7) — unchanged, still open. Item 4 ([DIST-5]) —
  **changed**, struck through and marked `[DONE]` per register L25. Item
  5 (D7) — unchanged, still open. Item 6 (D8) — unchanged, still open.
  Item 7 (0.3/0.4) — **changed**, annotated with the actual outcomes
  (0.3 DONE, 0.4 PARTIAL). Item 8 (3.1) — **changed**, annotated
  `[PARTIAL]` rather than left as an undifferentiated action item. Item
  9 (gas efficiency investment) — unchanged, still a proposed addition,
  not implemented.

---

## PART 4 — status table and summary

| Item | Old marker | New marker | Evidence |
|---|---|---|---|
| 0.3 | `[OPEN][MUST]` | `[DONE][MUST]` | `142d437`, `docs/metric_fixes_0_3.md` |
| 0.4 | `[OPEN][MUST]` | `[PARTIAL][MUST]` | `08b5972`; HiGHS binary present, `highspy` absent |
| 1.5 (hydro-trajectory sub-item) | `[OPEN]` | `[DONE]` | `f43a54c`, F-18; item-level `[PARTIAL]` unchanged |
| 2.2 | `[OPEN][MUST]` | `[PARTIAL][MUST]` | `d7538ff`; STR-1 still suspended; duration still fixed |
| 2.3 | `[OPEN][MUST]` | `[DONE][MUST]` | `d9aaa7d`, binds 21/30 yr |
| 2.4 | `[OPEN][MUST]` | `[DONE][MUST]` | `a2717ac`, four Oluwole (2023) arms |
| 3.1 | `[OPEN][MUST]` | `[PARTIAL][MUST]` | memo's own status line: §5/§6 open |
| 3.2 | `[OPEN][MUST]` | `[PARTIAL][MUST]` | register L23, two of three tranches |
| 3.5 | `[OPEN][MUST]` | `[PARTIAL][MUST]` | freeze done; B\* still ❌ in Sourcing ledger |
| 4.1 | `[OPEN][MUST]` | `[DONE][MUST]` | register L24 |
| 4.2 | `[OPEN][MUST]` | `[DONE][MUST]` | F-21, `docs/findings_frozen.md` |
| 5.3 | *(no bracket)* `[SHOULD]` | `[SUPERSEDED][SHOULD]` | `gas_add` free, F-14 |
| 5.5 | `[OPEN][MUST]` | `[SUPERSEDED][MUST]` | `docs/forensic_audit.md`, 59 findings |
| 5.1 | `[PARTIAL][MUST]` | `[SUPERSEDED-in-part][MUST]` | `29_run_parameter_uncertainty.py`, M-4 |
| 5.4 | `[OPEN][MUST]` | `[PARTIAL][MUST]` | harnesses repaired; repro pack incomplete |
| DISTRIBUTED EaaS (section) | `[OPEN][MUST]` | `[PARTIAL][MUST]` | mixed — see sub-items |
| DIST-5 | *(no bracket)* | `[DONE]` | register L25 |
| DIST-1/2/3/4 | *(no bracket)* | `[OPEN]` (each) | see Part 2 |
| 6.3 | `[OPEN][MUST]` | `[OPEN][MUST]` (unchanged) | material assembled, chapter unwritten |
| 1.6 Step 7 | `[DONE]` | `[DONE]` (unchanged, extended) | conv_none baseline constraint added |
| 6.2 | `[DONE]` | `[DONE]` (unchanged) | verified correct as written |
| D1 | *(no bracket)* `[MUST]` | `[DONE][MUST]` | `9c422e8` |
| D4 | `[SHOULD]` | `[SHOULD]` (unchanged; text corrected) | 04_run_fin2 stale reference fixed |
| D6 | `[OPEN][SHOULD]` | `[DONE][SHOULD]` | register L18 |
| D8 | `[OPEN][COULD]` | `[OPEN][COULD]` (unchanged) | distinct from L40, confirmed |
| All other items | — | unchanged | see full list above |

**Count: 18 marker changes, all in the OPEN/PARTIAL → more-complete
direction (DONE, PARTIAL, SUPERSEDED, or a bracket added where none
existed) — zero markers moved backward (no `[DONE]` was found to have
regressed to `[OPEN]` or `[PARTIAL]`).** Itemised: 0.3, 0.4, 1.5
(sub-item), 2.2, 2.3, 2.4, 3.1, 3.2, 3.5, 4.1, 4.2, 5.3, 5.5, 5.1, 5.4,
DISTRIBUTED EaaS (section + DIST-5 + four DIST sub-items given brackets
for the first time), D1, D6 — 18 items/sections carrying a marker change
or a first-time marker.

**Two items where TASK.md's own framing was found incomplete or wrong on
inspection**, per its own warning that "every 'established facts' list
supplied to this project by Chat has contained at least one error":
- **3.5** — TASK.md's claim ("frozen at `2a9226b`, both anchors
  reproducing") is true but is evidence for a narrower thing
  (reproducibility) than the item asks for (B\* reconciliation, T\*
  recomputation, placeholder sourcing) — marked PARTIAL, not DONE.
- **D1** — TASK.md did not mention D1 at all; it turned out to be
  closed (`9c422e8`) and, in the process of checking it, surfaced that
  its own closure documentation was never merged into the register.

---

## Encountered but not asked about

- **`docs/task_defect_closure.md`'s register-bound content
  (D1/D17/D21/D23/D24) was never pasted into `docs/limitations_
  register.md`.** Flagged in D1's entry above; fixing the register is
  outside this task's whitelist.
- **5.5's six named doc-code mismatches** were not individually
  cross-referenced against the forensic audit's 59 A-numbered findings —
  the supersession verdict is at the level of "a larger sweep exists,"
  not "every named item is confirmed closed or still open by number."
- **4.3 and 4.5 (and 5.2) carry no status bracket at all**, only a scope
  marker (`[SHOULD]`/`[STRETCH]`) — inconsistent with the rest of the
  file's convention of always pairing both. Not corrected; a
  reconciliation task should flag a formatting gap, not invent a
  convention.
- **The DEFERRED DEFECTS section header still reads a blanket `[OPEN]`**
  even though two of its eight items (D1, D6) are now `[DONE]` and one
  (D5) already was — the header describes the section's origin
  ("logged, not yet fixed"), not its current state; left as a section
  title, not a per-item marker, consistent with how PHASE 0's header
  works elsewhere in this file.
- **Plan item now judged obsolete rather than incomplete: 5.3.** Its
  entire premise (`gas_add` fixed at zero) no longer describes the
  model. This is different from every other item on this list — those
  are unfinished work; 5.3 is a question the current model has already
  answered as a side effect of an unrelated change (2.9's freeing of
  `gas_add`), not a task still waiting to be done.
- **`capital_recovery_scenarios()` may substantively answer §6 of
  `docs/currency_and_rate_basis.md`** (sourced, deflated-once rates for
  the tranches that actually feed the LP) without that memo's text
  having been updated to say so. Noted under 3.1 above; not resolved,
  since resolving it means editing that other memo, outside this
  task's whitelist.

**No commit was made. No script was run. No file other than
`docs/merged_correction_plan_v2.md` and this report was touched.**
