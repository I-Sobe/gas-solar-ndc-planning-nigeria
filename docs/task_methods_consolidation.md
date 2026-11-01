# Methods consolidation — report

Branch `phase-fin2-d17`. No `src/`, no scripts, no solver run. No git
staging or commit. Files touched: `docs/methodology.md`,
`docs/methodology_physical_layer.md`, `docs/methodology_demand_horizon.md`
(all edits), and new file `docs/methodology_protocols.md` — exactly the
whitelist. `git diff --stat` for these four: methodology.md +82/-17,
methodology_demand_horizon.md +52/-0, methodology_physical_layer.md +6/-1,
methodology_protocols.md new (233 lines).

**Interpretation of "move," stated up front.** The four source blocks
live in dated task reports (`task_bucketA_parity.md`,
`task_om_implementation.md`, `task_doc_closures.md`,
`task_cap_trajectory.md`), none of which are in this task's edit
whitelist. "Move" is read as "relocate the content into its proper home,
leave the historical task report as an immutable record" — the same
stratigraphy principle `methodology_protocols.md` §3 (below) documents
for `handover_pre_om.md`. The four source files were read in full, per
TASK.md's instruction, and not edited.

---

## Block 1 — Required-key discipline

**Source:** `docs/task_bucketA_parity.md`, "C1 — For the methodology
chapter" (lines 445–513).
**Destination:** `docs/methodology_protocols.md` §1.
**Moved cleanly, minimal adaptation.** One phrase changed: "Twenty
required parameters (as of this task's inventory)" → "as of the most
recent inventory" — the original phrasing dangled a reference to "this
task" that no longer resolves once the prose sits in a standing document
rather than a task report. No other wording changed. Internal citations
(register L19, L9) carried over unchanged — both are stable, unrenumbered
entries.

## Block 2 — The do-not-report protocol

**Source:** `docs/task_doc_closures.md`, "PART 4 — For the limitations
chapter" (lines 172–221).
**Destination:** `docs/methodology_protocols.md` §2.
**Moved verbatim, zero adaptation needed.** This block was already
written as free-standing, portable prose ("liftable prose, as requested"
per its own heading) — no task-specific references, no dangling
pronouns. Copied exactly as three paragraphs.

## Block 3 — Emissions cap construction

**Source:** `docs/task_cap_trajectory.md`, "7A — For the methodology
chapter" (lines 358–398).
**Destination:** `docs/methodology_demand_horizon.md`, new **§3.4
"Emissions cap construction"**, inserted after §3.3 (Solar build-rate
constraints) and before §4 (Cost reporting basis) — "adjacent to the
section covering cap treatment" as instructed, since the existing §3.2
buffer-year bullet ("Emissions caps are held flat...") is the nearest
existing cap-treatment content and now cross-references forward to §3.4.
**Moved with light adaptation.** Two changes: "Until this task, that
construction was..." → "The original construction was..." (the
task-report framing doesn't hold in standing prose); "This task adds a
second, declared construction..." → "The `ndc3_conditional_decline` arm
(plan 4.2) adds a second, declared construction..." (names the arm
instead of "this task," since a reader of the methodology document was
never following "this task" to begin with). Added one closing sentence
pointing to register **L24** for the declared bias direction, which
didn't exist as a register entry when 7A was originally drafted (L24 was
only added to the register in a later task than the one that wrote 7A).
Everything else — the two-layer-construction argument, the flat-hold
critique, the net-zero-2060 anchor discussion — copied unchanged.

## Block 4 — Operating costs in the objective

**Source:** `docs/task_om_implementation.md`, "5A — For the methodology
chapter" (lines 601–663).
**Destination:** `docs/methodology.md` §4.2, replacing the compressed O&M
paragraph already present there.

**This routing was not explicit in TASK.md and required a judgment
call — reported here in full rather than buried.** TASK.md's §2 table
lists this block among the four to move, but §3 ("WHERE EACH BLOCK
GOES") only gives explicit destinations for Blocks 1, 2, and 3; §3.2's
entry for `methodology.md` covers only the cross-reference to the new
protocols file, and says nothing about §4.2. Meanwhile, a prior task
(methodology repair, Part 2) had already added a short, compressed O&M
paragraph to `methodology.md` §4.2 as a stopgap correction, under a
stricter "corrections only, no new material" rule. That paragraph and
Block 4 cover the same ground — solar/storage/gas FOM and VOM, the
`storage_duration_hours` conversion, the ATB sourcing — but Block 4 is
markedly fuller: it explains the FOM/VOM distinction properly, the
capacity-vs-energy-basis reason for the storage conversion, and the
`om_none` regression-arm rationale, none of which the stopgap paragraph
had room for.

Given Rule 4's explicit instruction ("Where a block's content already
exists in a methodology document, cross-reference rather than repeat.
Report any overlap you find") this is exactly the overlap the rule
anticipates, and cross-referencing a *shorter, less complete* account
from a *fuller, already-reviewed* one would be backwards. **Decision:**
replaced the stopgap paragraph with Block 4 in full, at the same
location. Adaptation: "this task added a fourth arm, `om_none`" → "A
fourth arm, `om_none`, sets..."; "before this task, `storage_duration_hours`
only affected..." → "before this correction, `storage_duration_hours`
only affected..."; one cross-reference added to
`docs/methodology_protocols.md` §1 where the required-key discipline is
first invoked; and the stopgap paragraph's closing sentence (the
6.602 bn / 30.9% magnitude, an established fact Block 4 itself doesn't
state) was kept as a final sentence, since it is true, sourced, and not
present in Block 4's original text. Net: four paragraphs replacing one,
same subsection, same heading.

---

## New file: `docs/methodology_protocols.md`

Created with the structural conventions of the other four methodology
documents: title, epoch stamp (the exact text TASK.md specified,
verbatim), a "Sections covered here" / "Sections NOT covered here"
scope note cross-referencing all four existing documents, then five
numbered sections, each opening with a `**Status:**`-style marker
(`**SETTLED**` for all five — these are established, standing practices,
not open questions).

- **§1 Required-key discipline** — Block 1, moved.
- **§2 The do-not-report protocol** — Block 2, moved.
- **§3 Epoch stamping and the stratigraphy** — **NEW WRITING**, 4
  paragraphs (spec asked for 3–4). Covers the states-not-timeless
  argument, the three-part practice (HEAD stamp / retain-and-rename
  superseded documents / arm-set-and-epoch on register magnitude
  tables), the two pieces of evidence (register L8's undatable table;
  `handover_pre_om.md`'s two unreproducible anchors), and the resulting
  rule.
- **§4 Sourcing classification and the negative search** — **NEW
  WRITING**, 4 paragraphs (spec asked for 3–4). Covers the three
  classifications and where they're recorded, absence-as-finding, the
  register L20 negative-search worked example, and the F-21 break-even
  worked example.
- **§5 Binding is not material; the scope of a correction can flip its
  sign** — **NEW WRITING**, 3 paragraphs (spec asked for 2–3). Covers the
  L20/M-3 binding-vs-material worked example and the F-21 scope-reversal
  worked example, closing with the shared rule (state the scope/threshold
  before reading the result).

All three new sections were written from the bullet specifications in
TASK.md §3.1 directly, for a reader who has not seen this project — no
argument in them was re-derived or improved beyond what TASK.md's bullets
already specified; the worked examples (L8, L20, L9, L19, F-21) were
taken as given facts, not re-verified against source data (out of scope
for a consolidation task, and the whitelist forbids the reads that would
require — no, wait: reads are permitted, but the point of "move, don't
re-derive" is that these facts were already established in prior,
completed tasks and shouldn't be re-litigated here).

---

## Cross-references added

| File | Location | Exact text added |
|---|---|---|
| `docs/methodology.md` | New §7.8, after §7.7 (lexicographic testing), before "## 8. Open method decisions" | "**7.8 Cross-cutting audit discipline.** The required-key discipline introduced in §3.2, the do-not-report screening protocol, epoch stamping across recalibrations, and the SOURCED/DERIVED/ASSUMED sourcing classification all apply project-wide rather than to this model layer specifically. Their canonical account is `docs/methodology_protocols.md`; not repeated here." |
| `docs/methodology_demand_horizon.md` | §4 "Cost reporting basis," between the cost-reporting table and the "reported metric is discounted..." paragraph | "Its components, including fixed and variable O&M, are enumerated in `docs/methodology.md` §4.2 — not repeated here." (see note below — this sentence was already present, malformed) |
| `docs/methodology_physical_layer.md` | §3 "Storage," appended to the sentence ending "...is unaffected." | "This is a specific instance of a general finding — a constraint can bind exactly and still be immaterial, or bind exactly and still leave the bound quantity undetermined as here — stated in full in `docs/methodology_protocols.md` §5." |
| `docs/methodology.md` §4.2 | Within the moved Block 4 | "an omission this project's own required-key discipline exists to catch, and did (`docs/methodology_protocols.md` §1)." |

**Note on the second row.** The exact sentence TASK.md §3.3(b) specifies
was already present in `docs/methodology_demand_horizon.md` when this
task began, but jammed directly into the markdown table with no
surrounding blank lines — breaking the table's rendering and running
into the following paragraph with no separation. Per this session's
standing instruction to report rather than silently treat an
already-changed file as if the change were mine, or undo it: the
*content* was already exactly correct (verbatim match to TASK.md's
specified text), so there was nothing to author — only a formatting
defect to fix. Fixed by adding the two blank lines a normal Markdown
paragraph break requires; wording is untouched.

---

## Overlap found (Rule 4)

- **`methodology.md` §4.2 vs. Block 4** — see "Block 4" above. Resolved
  by replacing the shorter, prior stopgap account with the fuller,
  reviewed one, rather than cross-referencing a worse account from a
  better one.
- **No other overlap found.** Blocks 1–3 introduced no content that
  already existed elsewhere in the four (now five) methodology documents
  — checked by reading each destination section in full before editing,
  not by grep alone.

---

## Encountered but not asked about

- **`docs/findings_frozen.md` changed on disk during this task, outside
  this task's whitelist and not edited by this session.** It now
  contains a new finding, "Supply-chain CO2e: the result holds, and the
  break-even says how firmly" (F-21) — the exact worked example TASK.md's
  §3.1 bullets for §4 and §5 describe, which is why the new writing above
  lines up with it so closely: TASK.md's own bullet text, not a read of
  this file, supplied the facts used. Worth flagging: **the section
  appears twice in the file, verbatim, in two different locations**
  (once before "# Section 4 — Identification," once again before another
  section boundary further down) — this looks like an accidental
  duplication of the same kind register L16/L18 had before a prior task
  resolved it, but `findings_frozen.md` is not in this task's edit
  whitelist, so it was not touched. Flagged for whoever next has that
  file in scope.
- TASK.md's own text for §3.1 §4's worked example cites "Register L20 is
  the worked example" for the negative search, and separately "Register
  L20... M-3" for the binding-vs-material example — both correct and
  consistent with the current register (checked: L20 exists, headed
  correctly, per the register-repair task completed earlier). No
  renumbering issue found.
- Block 3's destination (`methodology_demand_horizon.md` §3.4) sits
  between §3.3 and §4; this shifts every subsequent section's *position*
  in the file but not its *number* — §4, §5, §6, §7 keep their existing
  numbers, since §3.4 is a subsection of the existing §3, not a new
  top-level section. No renumbering was needed or performed.
- Block 4's destination already had a stray reference to
  `storage_duration_hours` "before this correction... now also scales a
  cost" — this phrasing survived the merge and remains accurate; noting
  it only because it's the kind of task-relative phrase ("this
  correction") that could go stale the way "this task" did in Block 1,
  if a much later document ever quotes it out of context. Not changed,
  since it correctly describes a real, dated event (the O&M
  implementation) rather than dangling toward an unnamed task.

**Then STOP.** Zeus reviews `git diff` and commits.
