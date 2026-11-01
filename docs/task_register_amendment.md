# task_register_amendment — bringing the limitations register up to date

Read-only verification against code and git history; edits confined to
`docs/limitations_register.md` and `docs/findings_frozen.md`. No
script, no `src/` file, no other document touched. Nothing run, staged,
or committed. Reported incrementally, entry by entry, per instruction.

**Register entry count before this task: 44** (`L1`–`L44`; `grep -c
"^## L" docs/limitations_register.md` before any edit — this count
excludes the inert `## Lnn — one-line statement` template heading
between L7 and L8, which is not a numbered entry). **No entry is
renumbered by this task** — every amendment below is in place, in the
L10/L25 style.

---

## PART 1 — VERIFY EACH CLAIM BEFORE AMENDING

### 1.1 — L5: FALSIFIED, not pending

**Claim.** L5's status was PENDING, awaiting "the first post-2.3
solve" to determine whether the firm-capacity constraint gives storage
a sufficient capacity-value channel. TASK.md's premise: that solve now
exists (2026-09-16, `conv_none`, central arms) and shows storage is
built but never dispatched, falsifying the entry's literal claim
("optimal storage is identically zero for any cost").

**What I checked.** Read `results/baseline/summary__peak_unsuppressed_
_storage_cc_central__gas_unconstrained__constrained__gas_avail_
nameplate__gas_retire_vintage_30__gas_capex_central__ccgt_eff_central_
_cr_central__dscr_central__f1_tranche__env_dares__baseline__conv_none_
_hydro_flat.json` in full (dated 2026-09-16 08:10) and its paired
`timeseries__...csv`. Extracted `final_storage_capacity_mwh` and the
full 30-row `storage_capacity_mwh`/`storage_discharge_twh_e` columns
directly via `pandas`, not sampled.

**Verdict: CONFIRMED, exactly as TASK.md describes, with the precise
figures below.**
- `final_storage_capacity_mwh`: **78,882.13** (matches the figure this
  same session's own audit-scope carve-out cited: *"A fresh run built
  78,882 MWh"*).
- `storage_capacity_mwh` by year: 0 through 2034, then rising every
  year from 2035 (4,919.49) to 2054 (78,882.13) — monotonically
  increasing, never falling.
- `storage_discharge_twh_e` by year: **0.0 in every one of the 30
  years, 2025–2054, with no exception.**

So: capacity is substantial and growing; discharge is exactly zero
throughout. The original claim ("optimal storage is identically zero")
is false at the capacity level and true only at the discharge level —
a distinction the original entry conflated.

**Amendment made.** Status changed from `PENDING (2.3 may resolve)` to
`FALSIFIED`. The original **What** paragraph is preserved verbatim
inline (struck through at the specific falsified clause, per the
"preserve original text" instruction) with the correction stated
immediately around it, not as a separate contradicting paragraph
elsewhere in the entry. The entry's own **Watch Item**
(`capacity_credit['storage'] = 1.0`, flat) is promoted from a
"SOURCE: [PENDING]" caveat to the entry's live explanatory content,
since it is now the confirmed mechanism, not a residual worry.
Cross-referenced to **L31** (the peak-adequacy diagnostic's own flat-
multiple flaw — same underlying mechanism, different diagnostic).

**`docs/findings_frozen.md` citations — checked, BOTH already correct,
NEITHER amended:**
- F-19 (line 841-848), quoted in full in the register amendment: already
  attributes storage magnitude to "the flat `capacity_credit['storage']
  = 1.0`" and states plainly that "no interior optimum exists" — this
  is the corrected account already in the findings, unaffected by
  today's register correction.
- The scoped-conditions list (line 1090): `"L5 storage capacity credit
  is flat 1.0 -- magnitude is a residual"` — likewise already correct.

**This is worth stating plainly, since it is itself a finding:** the
REGISTER was the stale document, not the findings. Whoever wrote F-19
and the scoped-conditions list had already reached the corrected
understanding; that understanding was never carried back into L5
itself until now. This is precisely the failure mode TASK.md's own
"WHY" section describes register drafts stalling into — except here it
stalled in the other direction (the correction reached the findings but
not the register, rather than reaching neither).

---

### 1.2 — L15: verify RESOLVED

**Claim.** `scripts/01b_run_baseline_sliced.py:595-600` carries a
rename comment and emits `slice_detail_terminal_year`; TASK.md's
premise is that all four original stale-"2045" sites are now fixed.

**What I checked.** Read the current file at every site the entry's
own **What** names: the variable name (was `slice_2045`), the comment
describing it, the `diagnostics.json` key (was `slice_detail_2045`),
and the exported per-slice CSV row label (was the literal string
`"2045"`). Then `git log -S"slice_detail_terminal_year"` to find the
commit.

**Verdict: CONFIRMED, all four sites fixed.**
1. Variable: `slice_2045` → `slice_terminal_year` (`:486`).
2. Comment: `:482-483` and `:636-639` now explicitly document the
   rename and state both years are read dynamically.
3. `diagnostics.json` key: `slice_detail_2045` → `slice_detail_
   terminal_year` (`:600`), with an inline comment confirming the
   output-schema change was checked against consumers first.
4. CSV row label: `:641`, `str(years[-1])` (dynamic) replacing the
   hardcoded `"2045"` literal.

Commit: **`9605c82`** ("L15: terminal-year naming read dynamically;
diagnostics key slice_detail_2045 -> slice_detail_terminal_year
(output-schema change)").

**Amendment made.** Status changed to `RESOLVED (9605c82)`. Original
**What**/**Direction of bias** preserved verbatim below the status
line, per the L10/L25 pattern, with a note that all four sites are now
fixed and the mislabelling no longer occurs.

---

### 1.3 — L10: LIVE for two, not four

**Claim.** The existing amendment says LIVE for `06`, `07`, `09b`
(both), `17`. TASK.md's premise: `06` and `07` were revived at
`0012f16`, so the amendment should be extended (not rewritten) to
record this, leaving `09b` (both) and `17` as the only genuinely
remaining LIVE items.

**What I checked.** Read `scripts/06_run_gas1_shadow_benchmarks.py`
and `scripts/07_run_gas2_eaas_gas_relief.py` in full. Both now `import
argparse` and `from src.cli_arms import add_standard_arms, standard_
arms_kwargs`, both have a `parse_args()` function calling `add_
standard_arms(parser, exclude=[...])`, and both thread `standard_
kwargs`/`shared_fn_kwargs` into every `load_scenario()`/`run_gas_
regime_ndc_matrix()` call site. Confirmed via `git show --stat
0012f16` that both files were touched in that commit.

**Verdict: CONFIRMED.**

**Amendment made.** The existing `task_audit_fixes_p2` amendment
paragraph is EXTENDED (not rewritten) with a second paragraph
recording `06_run_gas1_shadow_benchmarks.py` and `07_run_gas2_eaas_
gas_relief.py` as now repaired, naming commit `0012f16`, and updating
the **Status** line from "LIVE for four of the nine items" to "LIVE
for two of the nine items ... historical for the other seven." The
original **What**/**Why unresolved**/etc. paragraphs, already written
to remain accurate for whichever items are still live, are left
untouched — they still correctly describe `09b` (both) and `17`, the
two genuinely remaining dead items, without needing rewording.

---

### 1.4 — L27: the nine WERE repaired

**Claim.** L27's status line says "NINE directly-broken runners and
two further library functions carry the same convention and are not
yet repaired." TASK.md's premise: the nine were repaired at `ecb7e54`,
and the two remaining library functions are `run_str1_storage_
parameter_sweep` and `run_bottleneck_sensitivity`.

**What I checked.** `git show --stat ecb7e54` (confirmed from this
session's own prior work, re-verified): touches exactly `04_run_fin2_
tariff_sweep.py`, `05_run_fin3_capital_sweep.py`, `15_run_pol1_ndc_
comparison.py`, `18_run_pareto_frontier.py`, `19_run_carbon_price_
sweep.py`, `20_run_cf1_solar_yield_sensitivity.py`, `24_run_leakage_
arm.py`, `25_run_obligation_trajectory.py`, `99c_run_genset_arm_
sensitivity.py` (plus `src/cli_arms.py`) — nine runner files. Grepped
all nine for `add_standard_arms`: all nine call it. Then read `src/
optimize_experiments.py`'s `run_str1_storage_parameter_sweep` and
`run_bottleneck_sensitivity` in full via `inspect.signature` and
direct source reading.

**Verdict: CONFIRMED on both counts.** All nine runners are alive
(`add_standard_arms` present and called in each). `run_bottleneck_
sensitivity(scenario, econ)` calls `load_scenario()` internally
(`:759-766`) supplying only `demand_level_case, demand_case, gas_
deliverability_case, carbon_case, start_year, end_year, om_case` — 10
of the 17 sentinel keys missing. `run_str1_storage_parameter_sweep`
also calls `load_scenario()` internally (`:2019`), with its own
self-diagnosing comment at `:2042`: *"This load_scenario() call is
also already missing peak_multiple_case/..."* — confirmed still
missing, still broken. Both genuinely still carry the stale
convention, exactly as TASK.md names them.

**Amendment made.** Status line updated to name the nine runners
explicitly as repaired, with commit `ecb7e54`, and to name the two
remaining library functions explicitly (previously alluded to but
never named in the entry body — a completeness gap closed here, not
just a status-line correction). `run_str1_storage_parameter_sweep`'s
already-broken status is cross-referenced to its own already-declared
context (L5's suspension of `14_run_str1_storage_role.py`, and A17's
"expected consequence" framing) so a reader does not mistake it for a
new, separate concern.

---

### 1.5 — L34: ESCALATE, do not close

**Claim.** L34 frames the `required_margin` double-count as dormant,
inert "because the runners that would trigger them are dead on
required arguments." TASK.md's premise: both `11_` and `13_` are now
repaired (`0012f16`), so the double-count is no longer dormant.

**What I checked.** Confirmed both `scripts/11_run_rel3_financing_
frontier.py` and `scripts/13_run_dem2_growth_gas_interaction.py` now
`import argparse` and call `add_standard_arms`, both verified
executable in the prior task's own report (`inspect.signature(...).
bind(...)` succeeded for both). Re-read `src/optimize_experiments.py:
1839-1841` directly: the override (`if fin_cfg["financing_regime"] ==
"eaas": scenario["required_margin"] = fin_cfg["required_margin"]`) is
still present, unchanged — this task's whitelist forbids editing
`src/`, so it could not have been fixed regardless. Confirmed `11_`'s
own `POLICY_CONFIGS` and `13_`'s own `FINANCING_CONFIGS` still carry
the retired `1.10`/`1.05` literals.

**Verdict: CONFIRMED. The double-count is now LIVE, not dormant, in
both runners.**

**Amendment made.** Status escalated from "LIVE (declared; both
defects are currently inert...)" to "LIVE — ACTIVE, not dormant: both
triggering runners are now executable (`0012f16`)." Added a precise
statement of which rows/runs are affected: every `13_run_dem2_growth_
gas_interaction.py` row with `financing_arm == "eaas"` (36 of 72,
per its own experiment structure), and every `11_run_rel3_financing_
frontier.py` row for the `eaas` financing arm across all `POLICY_
CONFIGS`. Added an explicit "what a reader must not do" instruction:
do not read any `eaas`-arm `required_margin`, financing cost, or
frontier-shift figure from either runner as reflecting its own
`--dscr-case` choice — it is silently overridden to the hardcoded
literal regardless of what was passed. The `public_only`/`traditional`
arm of both runners is unaffected and may be read normally. Original
**What**/**Why unresolved**/**What would resolve it** paragraphs
preserved, with the "currently dead" framing corrected in place to
"currently live" where the status depends on it.

---

### 1.6 — L35: `06_` and `07_` revived

**Claim.** Verify both files' revival at `0012f16`; check whether
A37's copy-paste bug (item 2 of L35's **What**) was fixed as part of
it; state what remains (TASK.md flags A47's stale capital-tier shape
in `07_`).

**What I checked.** Confirmed (as in 1.3) both files revived at
`0012f16`. Re-read `scripts/07_run_gas2_eaas_gas_relief.py`'s current
result dict at the site L35 names (originally `:187-188`, now shifted
by the revival's own edits): now reads `"financing_regime": "eaas"`,
`"status": "optimal"` — the copy-paste bug is fixed. Checked item 1
(benchmark dict duplication) — still present, unchanged, in all four
original locations. Checked item 3 (missing `"solar_total_mw"` key in
`run_no_policy_gas_matrix_eaas`'s result dict) — still absent,
unchanged. Checked `07_`'s `NDC_CASES["ndc3_conditional"]["capital_
case"]` — still `"tight"`, where the canonical convention (`02_/03_/
11_`) uses `"moderate"` for every conditional arm — confirmed present,
unfixed (out of scope for a plumbing-only revival, per that task's own
"change no constant" rule).

**Verdict: CONFIRMED. A37 (item 2) is RESOLVED. Items 1 and 3 remain
LIVE, now on executable runners rather than dormant. A new capital-
tier discrepancy in `07_` (the same shape as A47) is confirmed present
and was never fixed.**

**Amendment made.** Status updated to record that both `06_`/`07_` are
now executable (`0012f16`) and that A37 (item 2 in the original
**What**) is RESOLVED as part of that commit — struck through in
place, not deleted, per the preservation instruction. Items 1
(duplicate benchmark dict) and 3 (schema gap) are re-labelled from
"invisible while the runners cannot execute" to LIVE, since both
runners can now execute. A new numbered item 4 is added to the
**What** list: the `07_` capital-tier discrepancy
(`ndc3_conditional` → `"tight"`, canonical is `"moderate"`), citing
where it was found (this session's own runner-revival report) and
noting it is the same shape as the already-corrected A47 instance in
`15_run_pol1_ndc_comparison.py`.

---

### 1.7 — L36: `08_run_gas3` revived

**Claim.** Verify `08_run_gas3_regime_ndc_feasibility.py`'s revival at
`0012f16`; the two stale pre-recalibration anchors (items 2/3 of L36's
**What**, corresponding to A42/A43) are a separate claim — check
whether they still hold.

**What I checked.** Confirmed the file now has `argparse`/`add_
standard_arms` (item 1 of L36's **What**, the dead-runner membership,
is resolved). Re-checked `GAS_ANCHOR_TWH_TH` — still `40.66` (`:123`),
unchanged, still the pre-recalibration value against the live 89.27
anchor. Re-checked the docstring's cumulative-total figures (`:16-19`)
— still the stale 853.9/640.4/959.6/914.0 TWh_th figures, unchanged.

**Verdict: CONFIRMED. Item 1 is RESOLVED (`0012f16`). Items 2 and 3
are UNCHANGED and remain fully live** — this task's own prior revival
explicitly left them alone ("change no constant"), and this task's
whitelist (register/findings only) cannot fix them either.

**Amendment made.** Status line split: item 1 marked RESOLVED
(`0012f16`) in place; items 2 and 3 explicitly kept LIVE, with a note
that they are now MORE exposed than before, precisely because the
runner they sit in can now actually be executed — a future run would
silently miscalibrate the `gas_disrupted`/`cap_and_gas_disruption_
coincide` diagnostics the moment someone runs `08_` for real, whereas
before neither anchor mattered because the script could not run at
all.

---

### 1.8 — L43: partially resolved

**Claim.** A55 (16 missing arguments) was closed at `0012f16`; A56
(the `required_margin` override inside `optimize_experiments.py`) was
not. Amend to: executable, but must not be run for EaaS-arm figures.

**What I checked.** Confirmed `scripts/13_run_dem2_growth_gas_
interaction.py` now has `add_standard_arms`/`standard_arms_kwargs`,
and `inspect.signature(run_dem2_growth_gas_matrix).bind(...)` succeeds
with the 16 keys now threaded (re-verified, not merely recalled).
Re-confirmed `src/optimize_experiments.py:1839-1841`'s override is
still present, unchanged (out of this task's whitelist, same as 1.5).

**Verdict: CONFIRMED. A55 RESOLVED at `0012f16`. A56 NOT resolved —
root cause inside `optimize_experiments.py`, outside every whitelist
this session has had access to since the audit found it.**

**Amendment made.** Title/status updated: no longer "presented as
ACTIVE, cannot execute at all" (that premise is now false — it DOES
execute) but "executable since `0012f16`; its `eaas`-arm output is
WRONG, not merely unverified, and must not be used." Explicit
statement added, matching L34's own (this entry and L34 describe the
identical mechanism from `13_`'s specific angle): every row with
`financing_arm == "eaas"` carries a `required_margin` forced to the
hardcoded `1.10` in `FINANCING_CONFIGS["eaas"]`, regardless of `--
dscr-case`; the `public_only` half (36 of 72 rows) is sound and may be
read. Cross-referenced explicitly to L34 rather than duplicating the
mechanism's explanation a second time.

---

### 1.9 — L19: verify, may be resolved — **PREMISE WRONG, entry left LIVE**

**Claim.** TASK.md's premise: L19's "two required keys" resolved
internally by `26_run_f3_identification.py` are `conversion_case` and
`hydro_case`, threaded from `args` at lines 247-248 and 494-495; if
so, mark RESOLVED.

**What I checked.** Read L19's own **What** paragraph first, in full,
before touching the code: it names the two keys explicitly —
`--dist5-case` and `--solar-cf-case` — NOT `conversion_case`/`hydro_
case`. Then read `scripts/26_run_f3_identification.py` at the cited
lines and beyond: `:247-248` and `:494-495` DO thread `conversion_
case`/`hydro_case` from `args` — confirmed exactly as TASK.md
describes — but grepping the whole file for `dist5`, `solar_cf_case`,
`--dist5-case`, `--solar-cf-case` returns **zero hits** for any CLI
flag; `solar_cf` is read once, from `scenario["solar_cf"]` (`:341`),
resolved internally by `load_scenario()`'s own arm logic, exactly as
the original entry says.

**Verdict: TASK.md's PREMISE IS WRONG.** `conversion_case`/`hydro_
case` ARE correctly threaded in `26_`, but they are NOT the two keys
L19 is about — those were never the problem this entry describes.
`--dist5-case` and `--solar-cf-case` remain entirely unexposed, exactly
as L19 already states. This is precisely the kind of check-before-
amending trap TASK.md's own Part 1 preamble warns about, and it fired
here.

**Amendment: NONE. Entry left exactly as written, per instruction**
("If they are different keys, leave the entry live and say which" —
done, in this report; no edit made to the register entry itself, since
nothing in it needs correcting).

---

### 1.10 — L12: verify, may be resolvable

**Claim.** L12 says the annual baseline has no recorded reference
objective. A run now exists (2026-09-16, `conv_none`, central arms) —
read its objective and record it if it satisfies the entry's
requirement. Separately, note the bare `results/baseline/summary.json`
staleness as an F10 caution.

**What I checked.** Read the same 2026-09-16 `conv_none`/central-arms
summary used in 1.1: `npv_total_cost_usd: 32,448,782,287.34`,
`final_storage_capacity_mwh: 78,882.13`. Compared against L12's
existing **Mitigation** line, which already claims a recorded
reference: *"Post-change annual baseline recorded at 21.3748 bn /
63,055 MWh."* These do not match — checked why: read the bare
`results/baseline/summary.json` (mtime 2026-08-23) directly:
`npv_total_cost_usd: 16,926,222,807.78`, `final_storage_capacity_mwh:
0.0` — a THIRD, older figure, matching neither the entry's own
recorded Mitigation value nor the new tagged run. All three exist
simultaneously and disagree.

**Verdict.** The entry's underlying requirement ("a recorded reference
objective for the annual builder") IS now satisfiable by the 2026-09-16
tagged run — but the entry's EXISTING Mitigation figure (21.3748bn) is
itself now stale, superseded by O&M and other corrections since it was
written, and the BARE, untagged `results/baseline/summary.json` file
(the path a careless reader would open first) is stale by a full
epoch relative to both. This is exactly the F10 bare-path hazard
TASK.md names, now demonstrated with three actual, mutually
disagreeing numbers rather than argued abstractly.

**Amendment made.** **Mitigation** section rewritten in place (old
figure preserved, marked superseded, not deleted) to record the
current reference: **32,448,782,287.34 USD / 78,882.13 MWh terminal
storage**, from the tagged run `summary__..._conv_none__hydro_flat.
json` (2026-09-16), explicitly labelled as the CENTRAL-arm-set,
`conv_none` annual reference — distinct from, and not to be confused
with, the project's frozen REGRESSION anchor (21,811,042,472.821114,
`om_none`), which is a different arm combination entirely. A new
caution paragraph added, per instruction, naming the bare-path issue:
`results/baseline/summary.json` and `diagnostics.json` remain frozen
at 2026-08-23 (a pre-O&M, pre-storage-fix epoch: 16.93bn / 0 MWh
storage) and are NOT updated by any tagged run; anything reading the
bare path gets a third, oldest, and now clearly wrong state.

---

## PART 2 — SWEEP FOR ANYTHING ELSE STALE

Every remaining entry's status line read and checked against the
current tree, per instruction, with particular attention to entries
referencing dead runners, `load_scenario()` required keys, `om_case`,
field renames, `cli_arms.py`, or anything touched between `ecb7e54` and
`0012f16`.

| Entry | Status line checked | Verdict |
|---|---|---|
| L1 | Demand shape is a 2016 vintage | Still accurate — no code change touches the demand-vintage sourcing question. |
| L2 | Suppressed series cannot anchor demand | Still accurate — unrelated to any repair this session. |
| L3 | Firm capacity sized on block averages | Still accurate — no builder change to the firm-capacity resolution. |
| L4 | Tier 2 self-gen magnitude uncertain ~40% | Still accurate — sourcing gap unrelated to runner revivals. |
| L5 | PENDING | **AMENDED — see 1.1.** |
| L6 | LIVE unless C10 implemented | Still accurate — C10 not implemented; unrelated to this session's work. |
| L7 | Environment not reproducible before 2026-08-23 | Still accurate — a historical/provenance statement, unaffected by later commits. |
| L8 | Gas fleet retirement schedule is an analyst estimate | Still accurate — `gas_retire_vintage_*` arms unchanged; not a dead-runner or `om_case` entry. |
| L9 | `reserve_margin`/`genset_case` canonical, not swept | Still accurate — confirmed both remain hardcoded module-level constants in every revived runner too (checked `02_/06_/07_/08_/09_/10_/11_/13_/probe_2_6_structure.py`: none exposes either as a CLI arm; all hardcode `reserve_margin="nirp_spinning"` or thread it as one of the newly-added standard-arm flags without changing its canonical central value). |
| L10 | LIVE for four of nine | **AMENDED — see 1.3.** |
| L11 | `storage_binding` not meaningful under sliced builder | Still accurate — no change to `extract_planning_diagnostics`'s classifier or to which builder any revived runner uses (all nine revived runners use the ANNUAL builder, so L11's sliced-builder-specific claim is not even implicated by them). |
| L12 | No recorded reference objective | **AMENDED — see 1.10.** |
| L13 | Two epistemic families, GMP figure corrected | Still accurate post its own prior correction (task_audit_fixes_p3) — re-verified the 140.5/191.7/81.5-94.7 figures are unchanged by anything in `ecb7e54`..`0012f16`. |
| L14 | `_rebuild_years` duplicated | Still accurate — no builder edit in this range touches it. |
| L15 | Stale "2045" naming | **AMENDED — see 1.2.** |
| L16 | EaaS/public solar physically identical | Still accurate — no change to either builder's EaaS/public-solar treatment. |
| L17 | Bankability discounts at retired blend | Still accurate — no change to `resolve_private_rate()`/the bankability constraint in this range. |
| L18 | D6 CLOSED (rebuild capex) | Still accurate — a closed entry; nothing in `ecb7e54..0012f16` touches rebuild-capex logic. |
| L19 | `26_` sets two keys internally | **VERIFIED, PREMISE IN TASK.MD WRONG, entry left LIVE — see 1.9.** |
| L20 | `hydro_min_flow_fraction` ASSUMED, immaterial | Still accurate — declared closed/measured already; nothing in this range reopens it. |
| L21 | `storage_duration_hours` ASSUMED, unswept | Still accurate — no change to the constant or its exposure. |
| L22 | O&M sourced and wired | Still accurate — the `om_case` promotion this entry documents predates `ecb7e54` and is unaffected by the runner revivals (which only ADD `--om-case` flags to previously-argparse-less runners; the underlying O&M mechanism itself is untouched). |
| L23 | F1 ships two tranches, not three | Still accurate — no tranche-design change in this range. |
| L24 | Post-2035 cap shape stacked construction | Still accurate — no cap-construction change in this range. |
| L25 | SUPERSEDED IN PART, DIST-5 implemented | Still accurate — already its own in-place-revised entry; nothing in this range touches DIST-5/siting. |
| L26 | Solar CF national average, no siting | Still accurate — unaffected. |
| L27 | Six shared functions, nine runners unrepaired | **AMENDED — see 1.4.** |
| L28 | `cost_decomposition` O&M/rebuild omission | Still accurate — already RESOLVED status from a prior task; unaffected by this range. |
| L29 | `build_model_sliced()` dropped reliability aliases | Still accurate — already RESOLVED status (`task_audit_fixes_p1`); unaffected. |
| L30 | Live function carried retired gas efficiency | Still accurate — no change to the cited fallback site in this range. |
| L31 | Peak-adequacy diagnostic uses flat multiple | Still accurate — same flat-multiple mechanism as L5's now-corrected finding; cross-referenced from L5's amendment (1.1) but its own text needs no change, since it already correctly describes the flat-multiple problem on its own terms. |
| L32 | Three dead modules, six dead functions | Still accurate — `run_str1_storage_parameter_sweep` (one of the six named dead functions here too) re-confirmed dead in 1.4; the other five named modules/functions are untouched by anything in `ecb7e54..0012f16`. |
| L33 | `04_` retired-mechanism residue | Still accurate — re-checked `04_run_fin2_tariff_sweep.py`'s `--concessional-envelope`/`--deployable-fraction` plumbing and the stale `results/fin2/*` files: both unchanged; `04_` was not touched between `ecb7e54` and `0012f16` (it was already alive before `ecb7e54`, per this session's own A51 finding, and is not one of the nine revived at `0012f16` either). |
| L34 | `required_margin` double-count, dormant | **AMENDED — see 1.5.** |
| L35 | GAS-1/GAS-2 dead-runner defects | **AMENDED — see 1.6.** |
| L36 | `08_run_gas3` dead + stale anchors | **AMENDED — see 1.7.** |
| L37 | `05_run_fin3_capital_sweep.py` dead key + naming | Still accurate — `05_` was already alive before this session's revival work (confirmed cli_arms user since `ecb7e54`); re-checked `disco_collection_rate` and `budget_usd`/`budget_b_usd` in the current file: both defects unchanged, present exactly as declared. Entry correctly never claimed `05_` was dead. |
| L38 | `rpublic_diagnostic.py` wrong solar lifetime | Still accurate — re-checked `LIFETIMES["solar"]`: still `25`, live `asset_lifetimes()` still `30`. Unchanged; this file is outside every whitelist so far. |
| L39 | `14_run_str1_storage_role.py` stale precondition | Still accurate, and now reinforced rather than contradicted by L5's amendment (1.1): L5's falsification confirms storage genuinely has a role (capacity, not energy) under the CURRENT annual builder even without the sliced/intra-annual structure L39 says STR-1 is waiting for — worth a future task's attention (see closing list) but no edit made here, since judging whether this changes L39's own "re-enable when" text is a substantive call outside this task's narrow verify-and-amend scope for the ten named entries, and Part 2 is a sweep for staleness, not a re-opening of every adjacent question L5's correction touches. |
| L40 | `21_compute_resolution_premium.py` vacuous check | Still accurate — `21_` was not a revived runner (it calls no `load_scenario()`/shared function at all, confirmed in the prior runner-inventory task); unaffected by anything in `ecb7e54..0012f16`. |
| L41 | `plot_storage_regime.py` cannot run | Still accurate — unaffected; not a runner in the revived set and not touched by any commit in range. |
| L42 | `12_run_dem1_demand_sensitivity.py` orphaned docstring | Still accurate — `12_` is explicitly OUT OF SCOPE for the runner revival (a STUB, per Part 1 of that task); its `NotImplementedError` guard and orphaned docstring are both unchanged. |
| L43 | `13_` cannot execute, 16 missing args | **AMENDED — see 1.8.** |
| L44 | `08_phase8_analysis.py` NaN from field-name mismatch | Still accurate — re-checked `scripts/01_run_baseline.py`/`02_run_ndc_caps.py`'s field names (`final_storage_capacity_mwh` vs `storage_capacity_mwh`) and `08_phase8_analysis.py:139`'s single-name `.get()`: all three unchanged. `08_phase8_analysis.py` was not touched by the runner revival (it calls no `load_scenario()`/shared function, confirmed in Part 1 of that task) — the NaN this entry documents is unaffected. |

**44 of 44 remaining/checked entries accounted for** (34 "still
accurate" + 1 "verified, premise wrong, left live" + 9 amended above,
covering all of L1–L44 including the ten named in Part 1).

---

## Register entry count after this task

**44** — unchanged. No entry was added, removed, or renumbered. Every
amendment above is in place, inside the existing entry's own text,
following the L10/L25 pattern exactly (status line changed or
extended; original text preserved, struck through or marked at the
specific falsified/resolved clause rather than deleted; new content
appended within the same entry).

---

## Encountered but not asked about

- **L39's own "re-enable when" precondition may need a second look in
  light of L5's falsification** (noted in the Part 2 table above, not
  acted on): L5 now confirms storage has a genuine, real role
  (capacity credit) in the CURRENT annual builder, not the "no valid
  role at all" L39's suspension notice for STR-1 argues from. Whether
  this changes STR-1's own suspension case is a substantive question
  for a future task, not a register-staleness check.
- **`docs/findings_frozen.md`'s own F-19 and scoped-conditions-list
  entries for L5 were checked and found ALREADDY correct** (1.1) —
  worth noting as a positive finding about this document's own
  currency, not just the register's.
- **Three mutually disagreeing "annual baseline objective" figures now
  exist simultaneously** (16.93bn / 21.37bn / 32.45bn — bare-path,
  register's old Mitigation, and the new tagged run respectively) —
  L12's amendment (1.10) records the current one and flags the bare-
  path hazard, but does not — and this task's whitelist could not —
  delete or update the two superseded figures' own source files
  (`results/baseline/summary.json` is outside the docs-only whitelist).
- **L9's "canonical, not swept" claim was re-verified against all nine
  newly-revived runners specifically** (Part 2 table), since this was
  exactly the kind of thing TASK.md asked Part 2 to pay particular
  attention to — confirmed unaffected, included in the table rather
  than silently assumed.
- The register's own internal cross-references (`L5`→`L31`, `L34`↔`L43`,
  `L35`↔`L34`↔`L43`) were checked for consistency while amending; no
  broken or contradictory cross-reference was found beyond what this
  task's own amendments needed to add.
- `docs/findings_frozen.md` was read in full for L5's two citations
  (1.1) but was NOT swept entry-by-entry the way the register was in
  Part 2 — TASK.md's Part 2 instruction is scoped to "every remaining
  [register] entry," not to `findings_frozen.md`'s own content beyond
  the specific L5 citations Part 1 named. A finding worth flagging: if
  `findings_frozen.md` contains OTHER stale citations of newly-amended
  entries (L10, L27, L34, L35, L36, L43) beyond L5's two, this task did
  not check for them — grepped quickly for each entry's own number
  (`grep -n "L10\|L27\|L34\|L35\|L36\|L43" docs/findings_frozen.md`)
  and found no hits beyond the L5 ones already handled, so this gap is
  likely empty, but it was not exhaustively read the way L5's citations
  were.

**Then STOP, per instruction.** Nothing committed, nothing run. `git
status --porcelain` confirms only `docs/limitations_register.md`
modified plus this new report file — `docs/findings_frozen.md` was
read and checked but, per the verified verdict in 1.1, required no
edit.
