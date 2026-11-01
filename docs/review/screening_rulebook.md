# Screening Rulebook — Title/Abstract Stage

**Companion to:** OSF Protocol Amendment (Eligibility Refinement and AI-Assisted Screening)
**Review:** Energy security in optimisation-based power-system planning for developing and emerging economies: a systematic review of dimensional coverage and the endogenous treatment of financing structure and cost-of-capital
**Version:** 1.0
**Date:** 10-07-2026
**Status:** Pre-committed before screening. Any change during screening is logged as a dated deviation with rationale.

**Purpose.** This document fixes the screening decision rules *before* screening begins, so that ~23,700 title/abstract decisions are applied consistently rather than drifting across a long solo screening effort. It is an OSF artefact. When a decision is genuinely ambiguous, the rule is: **retain to the next stage** (retention-on-doubt) — under-exclusion is recoverable at full-text; over-exclusion is not.

---

## 1. Eligibility criteria (as applied at title/abstract stage)

The screening applies the eligibility criteria registered in protocol v4.0 as refined by the amendment. Summarised operationally for screening:

**Include if ALL of the following hold** (subject to the geographic and methodological rules below):
[CONFIRM: transcribe your registered inclusion criteria here as a checklist — population/system, intervention/method, outcome, study type, language, date range, etc. This rulebook should restate them in screening-checklist form, not introduce new ones.]

**Exclude if ANY exclusion criterion is met** (see §2–§5 for the operational rules that most affect screening).

---

## 2. Geographic rule — Sub-Saharan Africa

**Operational definition of SSA:** [CONFIRM: the country list or published standard adopted in the amendment — keep identical to the amendment wording.]

Apply this decision order to each record:

1. **Does the study empirically analyse one or more SSA countries?**
   → Yes: geographically eligible (proceed to method/other criteria).
   → No: go to 2.

2. **Is the study multi-region, with SSA (or an SSA country) reported as a distinct analysed unit** (SSA-specific results, not only aggregate)?
   → Yes: geographically eligible.
   → No: go to 3.

3. **Does SSA appear only as background, motivation, or mention** (no SSA-specific analysis)?
   → Yes: **EXCLUDE** — reason code `GEO-background`.

4. **Is the empirical scope wholly outside SSA** (incl. North-Africa-only)?
   → Yes: **EXCLUDE** — reason code `GEO-outside`.

### Worked edge cases
- **Global least-cost model with Nigeria as one node, Nigeria results reported separately** → INCLUDE (rule 2, distinct unit).
- **Global model reporting only a single aggregate world result, Africa not separable** → EXCLUDE `GEO-background`.
- **Pan-African study spanning North + Sub-Saharan Africa, SSA countries analysed individually** → INCLUDE (rule 1/2 for the SSA portion).
- **Study confined to Morocco/Egypt/Algeria/Tunisia/Libya** → EXCLUDE `GEO-outside` (North Africa is outside the adopted SSA definition).
- **Generic optimisation-method paper, no country, states "applicable to developing regions such as SSA"** → title/abstract cannot confirm SSA empirical application → **RETAIN to full-text** (retention-on-doubt); do not exclude on geography at this stage.
- **Continental "Africa" study, unclear whether SSA is separable from the abstract** → RETAIN to full-text.

---

## 3. Methodological rule (unchanged from v4.0)

The geographic refinement does not change the methodological criteria. Apply the registered methodological boundary exactly as registered, including:

- **Rule R1 — simulation-vs-optimisation boundary:** [CONFIRM: restate R1 in one or two sentences as registered, so the screener applies it identically. Per the registered protocol, dispatch-only simulation without capacity/planning optimisation is excluded under edge-case A; Schinko et al. 2019 is the documented exemplar exclusion.]
- Reason code for methodological exclusion: `METHOD-outofscope` (with a sub-note where R1 is the specific basis: `METHOD-R1`).

---

## 4. Abstract-less stratum — title-only screening

For records with no abstract (the Entry 9 stratum):

- Screen on **title alone**.
- **Exclude only if the title alone clearly establishes ineligibility** — e.g. unambiguously off-topic subject, clearly non-research document type (the book-review record surfaced during recovery is a clear exclude), or unambiguously outside SSA.
- **If the title does not clearly exclude → RETAIN to full-text.** Do not exclude for missing abstract.
- Reason code when excluded at this stage: `TITLEONLY-exclude` (records the decision was made on title alone, for PRISMA transparency).

---

## 5. Document-type and other standing excludes

Apply the registered document-type and language limits. Common title/abstract-stage excludes and their codes:
- Non-research document (editorial, preface, erratum, conference front-matter, book review) → `TYPE-nonresearch`.
- Language outside registered limits → `LANG`. [CONFIRM: registered language limits.]
- Outside registered date range → `DATE`. [CONFIRM: registered range.]
- Duplicate not caught by dedup → `DUP` (log for dedup-audit feedback).

---

## 6. Decision codes (use consistently — these populate the PRISMA flow)

| Decision | Code | Meaning |
|---|---|---|
| Include | `INC` | Passes to full-text retrieval |
| Exclude — geography (background) | `GEO-background` | SSA only mentioned |
| Exclude — geography (outside) | `GEO-outside` | Wholly non-SSA / North-Africa-only |
| Exclude — method | `METHOD-outofscope` / `METHOD-R1` | Fails methodological criterion / Rule R1 |
| Exclude — document type | `TYPE-nonresearch` | Not a research study |
| Exclude — language | `LANG` | Outside language limits |
| Exclude — date | `DATE` | Outside date range |
| Retain-on-doubt | `INC` (flagged `doubt`) | Ambiguous → passes to full-text, flagged for attention |

Retention-on-doubt records are coded `INC` with a `doubt` flag, so they are both (a) correctly carried forward and (b) identifiable as the ambiguous set that warranted extra full-text attention.

---

## 7. Consistency safeguards

- **Re-anchor periodically:** at the start of each screening session, re-read §2–§5 before making decisions, to prevent criterion drift over a long solo effort.
- **Log genuine ambiguities:** where a record forces a judgement the rulebook doesn't cover, record it and the decision made; if the same pattern recurs, add an explicit edge case (dated) rather than deciding ad-hoc each time.
- **Do not exclude for missing abstract, ever** (see §4).
- **When in doubt, retain.** The full-text stage is where uncertain records are resolved with the actual paper in hand.

---

*All `[CONFIRM: …]` fields must be filled from the registered protocol v4.0 before screening begins. This rulebook and the amendment are companion documents; keep the SSA definition and Rule R1 wording identical across both.*
