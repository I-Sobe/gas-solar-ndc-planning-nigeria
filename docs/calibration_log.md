# Calibration log — what Chat got wrong, after Phase A

**Living continuation of `docs/handover_2026_09_27.md` §9.** That
section is historical — frozen at the close of Phase A, not to be
edited — and its own closing line states why a record like this
exists at all: *"Recorded because the pattern matters more than the
instances."* This file is where that pattern keeps being recorded for
Phase B and beyond, entry per error, newest last. The working pattern
§9 describes is unchanged: **Chat proposes, Code verifies against the
file, and Code reports failures rather than correcting them silently.**
This log is specifically for errors that originated with Chat's own
instructions to Code — not for errors in the model, the data, or the
documentation, which belong in `docs/limitations_register.md` or
`docs/methodology_claims_ledger.md` instead.

---

## C1 — TASK 17 Part A.3: the DERIVED tag for `gas_avail_nameplate`, and the DSCR analogy that did not hold

**What Chat instructed.** TASK 17 Part A.3 told Code to document
`gas_avail_nameplate = 0.80` in `parameter_provenance.md` as
**DERIVED**, constructed as the GADS fleet-weighted benchmark `B`
(0.8307) minus a declared Nigeria derate of `(B − 0.80)`, explicitly
"on the DSCR pattern" (`dscr_case`, row 13: a sourced base, Yescombe's
1.25, plus a declared Nigeria premium to reach `dscr_central = 1.30`).

**Why it was wrong.** The construction was circular, not derived.
`0.80` predates `B` entirely — it is the pre-2.10 unsourced assumption,
fixed long before this benchmark existed — so `(B − 0.80)` is a
**residual** that reconciles an already-fixed value, not an estimate
computed from a source. The DSCR analogy specifically does not hold:
Yescombe's 1.25 was known **before** `dscr_central = 1.30` was chosen,
so DSCR's 0.05 premium is a decision taken against a known base — a
genuine derivation. There is no equivalent antecedent for `0.80`.
Tagging the row DERIVED asserted the value was computed from a source
found only after the value was already set, which the
SOURCED/DERIVED/ASSUMED taxonomy does not support.

**Where responsibility sits.** The error originated with Chat, not
Code. Code implemented TASK 17 A.3 as written — the resulting
`parameter_provenance.md` row and `limitations_register.md` L68 text
were a correct execution of a wrong instruction, not a separate defect
in Code's own work.

**How it was caught and fixed.** Chat caught it two turns later,
unprompted by any external check, and TASK 18 Part C instructed the
correction: the row was retagged ASSUMED, `B` was reframed as
corroboration only (not a source), the derate construction was removed
entirely, and `limitations_register.md` L68 was updated to match.
**No artefact was published in the interval** — the DERIVED version of
row 21 existed only between the TASK 17 and TASK 18 commits on this
branch, never reached a chapter draft, and was corrected before this
entry was written.

**The pattern, stated once rather than per-instance (matching §9's own
framing):** an analogy between two provenance constructions was
asserted without checking whether the analogy's own precondition held
(a known base preceding a declared adjustment) — the same failure
shape as asserting a citation can be inferred from a number, or that a
file says something without reading it. The fix in both cases is the
same working pattern §9 already names: verify against the source
before instructing, and when that fails, report the error rather than
quietly tightening the next instruction and hoping it does not recur.

---

## Why `docs/handover_2026_09_27.md` §9 was not edited

Per TASK 19's own instruction: §9 is a historical snapshot, dated and
closed at Phase A's end ("where it disagrees with an older document,
this one wins" is the handover's own rule for *content* precedence,
not an invitation to keep appending to a section that is explicitly
retrospective). This file is the chosen living alternative — a new
document, not an existing one repurposed, because no existing living
document (`docs/branch_status.md`, `docs/limitations_register.md`,
`docs/methodology_claims_ledger.md`) has this as its subject without
conflating it with branch/tag state, parameter/data limitations, or
methodology-claim verification respectively. Future Chat-instruction
errors, caught the same way, belong here as `C2`, `C3`, and so on.
