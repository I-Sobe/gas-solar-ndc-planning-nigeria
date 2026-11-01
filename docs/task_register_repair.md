# Register repair — missing entries applied, number collisions resolved

Branch `phase-fin2-d17`. No `src/`, no scripts, no solver run. No git
staging or commit performed. File edited: `docs/limitations_register.md`
only — `docs/findings_frozen.md` and `docs/frozen_state_2026_09_10.md` were
in the whitelist but needed no edit (see Step 3's citation audit: every
citation in those two files already cites the correct, unrenumbered entry).

---

## Step 1 — actual state before any edit

```
grep -n "^## L" docs/limitations_register.md
```

```
16:## L1 — Demand shape is a 2016 vintage
56:## L2 — Suppressed series is a supply shape and cannot anchor demand
78:## L3 — Firm capacity is sized on block averages, not instantaneous peak
99:## L4 — Tier 2 self-generation magnitude uncertain by ~40%
122:## L5 — Storage is structurally dominated at annual resolution
147:## L6 — EaaS channel excludes storage by construction
171:## L7 — Environment was not reproducible before 2026-08-23
196:## Lnn — one-line statement                                    [template, not an entry]
212:## L8 — Gas fleet retirement schedule is an analyst estimate
324:## L9 — `reserve_margin` and `genset_case` are canonical constants, not swept arms
357:## L10 — Ten runners are dead code predating F1
386:## L11 — `storage_binding` diagnostic is not meaningful under the sliced builder
413:## L12 — The annual baseline has no recorded reference objective
439:## L13 — Gas deliverability arms span two epistemic families
462:## L14 — `_rebuild_years` is duplicated across the F1 branch
485:## L15 — Stale "2045" naming across four sites
506:## L16 — EaaS and public solar are physically identical in the model
552:## L17 — The bankability constraint discounts at the retired pre-F1 blend, over a horizon the objective contradicts
658:## L18 — D6 (buffer-year storage replacement capex) is CLOSED; its guard message was stale
739:## L16 — SUPERSEDED IN PART: DIST-5 is implemented; the direction IS established
785:## L18 — Solar capacity factor is a national average with no siting
823:## L19 — `26_run_f3_identification.py` sets two required keys internally
845:## L20 — hydro_min_flow_fraction is ASSUMED, binding, and immaterial
922:## L23 — F1 ships two capital tranches; the design specified three
```

**The actual state, not the assumed one.** L21 and L22 do not exist as
headed entries anywhere in the file, confirming TASK.md's context. L16 and
L18 each head two different entries (lines 506/739 and 658/785
respectively). L23 is the true highest entry — L21 and L22 are gaps, not
overflow. (The `## Lnn` line at 196 is the entry template inside a fenced
code block, not a real entry — excluded from all counts below.)

---

## Step 2 — L21 and L22 extracted and placed

**L21 source.** `docs/task_bucketA_parity.md`, section "### C2 — For the
limitations register" (lines 514–567), a blockquoted draft beginning `>
## L21 — storage_duration_hours is an explicit constant, ASSUMED, and
unswept`.

**L22 source.** `docs/task_om_implementation.md`, section "### 5B — For
the limitations register" (lines 665–748), a blockquoted draft beginning
`> ## L22 — Fixed + variable O&M is now SOURCED and wired; the storage
conversion inherits L21's uncertainty`.

**Extraction method.** Read each draft in full from its source file,
verified the blockquote block's start (`> ## L2n —...`) and end (the last
`>`-prefixed line before the next unquoted heading), stripped the leading
`> ` (or bare `>` on blank lines) from every line, and inserted the result
verbatim — no wording changed — between L20's content and L23's heading in
`docs/limitations_register.md`.

**Edit.** `docs/limitations_register.md` — old text: L20's content ran
directly into `## L23` with no blank line and no `---` separator (a
pre-existing gap in formatting, itself evidence L21/L22 were never placed).
New text: L20's content, then `\n---\n\n`, the stripped L21 entry
verbatim, `\n---\n\n`, the stripped L22 entry verbatim, `\n---\n\n`, then
`## L23 —...` unchanged. The `---` separator style matches the file's
dominant convention elsewhere (e.g. between L1/L2, L17/L18) — not
specified by TASK.md, chosen for consistency since the pre-existing
L19→L20 and L20→L23 boundaries were the only two places in the file
missing it.

Both entries now appear at lines 841 (L21) and 896 (L22) in the file, in
numerical order between L20 (761) and L23 (978).

---

## Step 3 — L16 and L18 collisions resolved

**Rule applied.** For each pair, "earlier" and "later" are determined by
physical position in the pre-edit file (line number), on the reasoning
that the entry appearing first is the one that occupied the number first;
the second is the one that collided with it. Confirmed this reasoning
doesn't invert the intended meaning by reading both pairs' content: the
second L16 explicitly says "The original entry stated..." and revises a
claim the first L16 makes, confirming it really is the later-written one;
the second L18 is thematically unrelated to the first (solar capacity
factor vs. storage replacement capex) — an accidental reuse, not a
revision, but the same earlier/later rule applies cleanly either way.

| Number | Kept at its number (earlier, unmodified) | Renumbered (later) |
|---|---|---|
| L16 | "EaaS and public solar are physically identical in the model" (was line 506) | → **L25** — "SUPERSEDED IN PART: DIST-5 is implemented; the direction IS established" (was line 739) |
| L18 | "D6 (buffer-year storage replacement capex) is CLOSED; its guard message was stale" (was line 658) | → **L26** — "Solar capacity factor is a national average with no siting" (was line 785) |

**Why relocation, not just a heading rename.** Step 5 requires `grep -n
"^## L"` to show ascending order with no gaps. Renaming the heading in
place while leaving the entry at its old physical position would produce
`... L16, L17, L18, L25, L26, L19, L20 ...` — numerically out of order in
file order. So each renumbered entry was **physically moved** to the end
of the file, immediately after the new L24 (Step 4), in the order L25 then
L26 — not merely relabelled in place. Content of both moved entries is
byte-for-byte unchanged except the heading's number (`L16` → `L25`,
`L18` → `L26`); no other text inside either entry was touched (both happen
to contain no self-referential mention of their own old number, so no
internal fix-up was needed).

**Edit.** `docs/limitations_register.md`:
1. Removed the block from `## L16 — SUPERSEDED IN PART...` through the end
   of `## L18 — Solar capacity factor...` (old lines 739–822 inclusive),
   which sat between the first L18's closing separator and `## L19`. The
   pre-existing `---` separator at old lines 737–738 (between first-L18
   and the removed block) was left in place, so it now separates
   first-L18 directly from L19 — closing the gap.
2. Appended both removed entries at the end of the file, after the new
   L24 (Step 4), with headings changed to `## L25` and `## L26`
   respectively, each preceded by a `---` separator.

### Citation audit

`grep -rn "\bL16\b" docs/` and `grep -rn "\bL18\b" docs/`, run before any
edit, minus the four heading lines themselves. Every hit, and whether it
needed changing:

| File:line | Cites | Content matches | Change needed? |
|---|---|---|---|
| `docs/findings_frozen.md:389` | L16 | "the distributed capex premium is unimplemented" — matches first L16's DIST-5 bullet | No — correct number. **Content is stale**, see below |
| `docs/findings_frozen.md:615` | L16 | "EaaS and public solar are physically identical" — verbatim first-L16 title | No |
| `docs/frozen_state_2026_09_10.md:290` | L16 | same verbatim title | No |
| `docs/findings_provisional_pre_sweep.md:1272` | L16 | "public and EaaS solar are priced identically" — first-L16 content | No — not in edit whitelist |
| `docs/findings_provisional_pre_sweep.md:1284` | L16 | "EaaS and public solar are physically identical" — verbatim first-L16 title | No — not in edit whitelist |
| `docs/findings_provisional_pre_om.md:292` | L16 | "the distributed capex premium is unimplemented" — first-L16 content | No — not in edit whitelist. **Content stale**, see below |
| `docs/findings_provisional_pre_om.md:447` | L16 | verbatim first-L16 title | No — not in edit whitelist |
| `docs/handover_pre_om.md:193` | L16 | **Ambiguous** — blends both halves: "DIST-2... is NOT implemented" (first-L16 content) with "DIST-5 alone therefore biases against EaaS. Direction stated" (a conclusion that only the *second* L16/now-L25 reaches; the first L16 says the opposite — "the net direction is not established") | **Reported, left unchanged**, per instruction not to guess on an ambiguous citation. Also outside the edit whitelist |
| `docs/merged_correction_plan_v2.md:1182` | L18 | "register L18 records D6 CLOSED... both branches... charge a rebuild" — first-L18 content | No — not in edit whitelist |
| `docs/d6_storage_guard.md:256,258,288,299` | L18 (×4) | all about the D6/storage-guard entry itself — first-L18 content | No — not in edit whitelist |
| `docs/task_om_implementation.md:449` | L18 | "the model's OWN established convention... REBUILD CAPEX charge" — first-L18 content | No — not in edit whitelist |
| `docs/f1_implementation.md:264,267,733,741` | L18 (×4) | all about the storage guard rewrite/D6 closure — first-L18 content | No — not in edit whitelist |
| `docs/limitations_register.md:471` | L18 | "Amendment (L18, D6 closure)" inside L14's own entry — first-L18 content | No — already correct, in the edited file |
| `docs/task_cap_trajectory.md:484` | L16 and L18 (meta) | reporting the collision itself ("plus L16 and L18, each...duplicated") — refers to *both* halves of each, not one | Not applicable — not in edit whitelist, and this line's own subject is the duplication, not a specific entry |
| `docs/task_doc_closures.md:148,153,154,231,233,249` | L16 (×4), L18 (×2) (meta) | same — a prior task's report *about* the duplication itself | Not applicable — not in edit whitelist |

**Result: zero citations needed a number change.** Every citation outside
`docs/task_cap_trajectory.md` and `docs/task_doc_closures.md`'s own
meta-commentary refers to content from the *first* (kept) L16 or L18, so
no external document ever cited the entry that got renumbered to L25/L26
— those two entries were, apparently, never cited by number anywhere else
in `docs/`. One ambiguous case is reported above and left untouched, per
instruction.

**Flagged, not fixed (out of this task's scope — a numbering task, not a
content-accuracy task).** `findings_frozen.md:389` and
`findings_provisional_pre_om.md:292` both assert "the distributed capex
premium is unimplemented," which was true of the *original* L16 but is
contradicted by the entry now at L25 ("DIST-5 is now implemented"). This
is a real, substantive staleness in `findings_frozen.md` — a file this
task's whitelist permits editing — but Step 3 asked only to update
*citation numbers* for the renumbered entries, not to correct claims that
were already stale under the old numbering. Left as found; see closing
list.

---

## Step 4 — L24 appended

**Source.** `docs/task_cap_trajectory.md`, section "## 7B — For the
limitations register (drafted, not applied)" (lines 417–474), a
blockquoted draft beginning `> ## L24 — Post-2035 cap shape is a second
analyst construction stacked on the apportionment rule`.

**Edit.** Stripped the leading `> ` from every line and appended the
result verbatim to `docs/limitations_register.md`, after L23's content and
before the (relocated) L25/L26 — preserving numerical order end-to-end.
Now at line 1072.

---

## Step 5 — verification

```
grep -n "^## L" docs/limitations_register.md
```

```
16:## L1 — Demand shape is a 2016 vintage
56:## L2 — Suppressed series is a supply shape and cannot anchor demand
78:## L3 — Firm capacity is sized on block averages, not instantaneous peak
99:## L4 — Tier 2 self-generation magnitude uncertain by ~40%
122:## L5 — Storage is structurally dominated at annual resolution
147:## L6 — EaaS channel excludes storage by construction
171:## L7 — Environment was not reproducible before 2026-08-23
196:## Lnn — one-line statement                                    [template, not an entry]
212:## L8 — Gas fleet retirement schedule is an analyst estimate
324:## L9 — `reserve_margin` and `genset_case` are canonical constants, not swept arms
357:## L10 — Ten runners are dead code predating F1
386:## L11 — `storage_binding` diagnostic is not meaningful under the sliced builder
413:## L12 — The annual baseline has no recorded reference objective
439:## L13 — Gas deliverability arms span two epistemic families
462:## L14 — `_rebuild_years` is duplicated across the F1 branch
485:## L15 — Stale "2045" naming across four sites
506:## L16 — EaaS and public solar are physically identical in the model
552:## L17 — The bankability constraint discounts at the retired pre-F1 blend, over a horizon the objective contradicts
658:## L18 — D6 (buffer-year storage replacement capex) is CLOSED; its guard message was stale
739:## L19 — `26_run_f3_identification.py` sets two required keys internally
761:## L20 — hydro_min_flow_fraction is ASSUMED, binding, and immaterial
841:## L21 — storage_duration_hours is an explicit constant, ASSUMED, and unswept
896:## L22 — Fixed + variable O&M is now SOURCED and wired; the storage conversion inherits L21's uncertainty
978:## L23 — F1 ships two capital tranches; the design specified three
1072:## L24 — Post-2035 cap shape is a second analyst construction stacked on the apportionment rule
1133:## L25 — SUPERSEDED IN PART: DIST-5 is implemented; the direction IS established
1179:## L26 — Solar capacity factor is a national average with no siting
```

**Confirmed.** L1 through L26, each number exactly once, ascending, no
gaps (the `## Lnn` template line excluded, as it is not a numbered entry
and was never claimed to be one).

`git diff --stat docs/limitations_register.md`: 285 insertions, 84
deletions, one file changed — no other file touched by this task.

---

## Encountered but not asked about

- **`findings_frozen.md:389` and `findings_provisional_pre_om.md:292`**
  both state "the distributed capex premium is unimplemented," which is
  now false — the entry that used to share L16's number (now L25) records
  that DIST-5 (the distributed capex premium) **is** implemented, with a
  measured near-zero effect on NDC-capped objectives. `findings_frozen.md`
  is nominally the frozen-state record and is in this task's edit
  whitelist, but Step 3 scoped this task to citation *numbers*, not
  content accuracy, and this staleness predates the renumbering (it was
  already stale under the old, colliding L16 number) — flagged for a
  future content-correction task rather than fixed here.
- **`docs/handover_pre_om.md:193`** cites "L16" with a sentence that blends
  claims from both former L16 halves (first-L16's "DIST-2 is NOT
  implemented" plus second-L16/now-L25's "direction IS established, biased
  against EaaS"). Reported as ambiguous per Step 3's instruction; left
  unchanged. `handover_pre_om.md` is also outside this task's edit
  whitelist regardless.
- The `## Lnn — one-line statement` entry template (inside a fenced code
  block, `docs/limitations_register.md` line 196) also matches
  `grep -n "^## L"` since fenced-code-block content isn't excluded by a
  plain grep. It is not a numbered entry and was excluded from every
  count and from the Step 5 ascending-order check by inspection, not by
  a smarter grep pattern — worth knowing if Step 5's exact command is
  re-run mechanically by something that can't tell the difference.
- No file outside `docs/` was searched for L16/L18 citations — TASK.md
  scoped the citation grep to "every file under docs/" specifically, so
  `src/`, `scripts/`, and `CLAUDE.md`/`TASK.md` were not checked. Given
  this project's own register documents are the place such citations
  would appear (and none of the four register-editing precedent tasks
  found in `docs/` — `d6_storage_guard.md`, `task_om_implementation.md`,
  `task_bucketA_parity.md`, `f1_implementation.md` — ever cite a register
  number from code), this is unlikely to matter, but it was not verified.
