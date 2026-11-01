# HANDOVER — NDC-EaaS thesis, Chapter 2 (Literature review)

**Revised 2026-10-10.** Supersedes the 2026-10-09 version. Changes are
marked where they matter.

**Write Chapter 2 before Chapter 1.** Chapter 1's gap statement,
research questions and contribution list are all claims about what the
literature does not do. Zeus has a systematic review that *measures*
that. Writing Chapter 1 first means asserting those claims and then
rewriting them when the review's counts contradict them.

**This handover is not a source.** It is a summary, and summaries in
this project have drifted from the things they summarise. One headline
range quoted in a sibling handover — EaaS worth 15.8–28.5 bn under an
NDC cap — turned out to exist in no result file anywhere in the
repository, and reached a chapter skeleton before anyone checked. Treat
every figure and characterisation below as a pointer to a source, never
as the source.

---

## 1. What you are working with

**A completed PRISMA systematic review**, titled *"Energy security in
optimisation-based power-system planning for Sub-Saharan Africa: a
systematic review of dimensional coverage and the endogenous treatment
of financing structure and cost-of-capital."*

Zeus holds the full materials and will supply them on request:
protocol, OSF registration, search string, screening records and the
extracted data. **Ask for them before drafting anything.** Do not write
a word of this chapter from a description of the review — including the
description in this document.

Two things to establish with Zeus in your first exchange, because both
change how the chapter is written: whether the extraction data is in a
form you can read directly or needs transcribing, and whether the OSF
registration is public, since if it is the chapter cites it by DOI
rather than describing it.

**A thesis whose model responds to the review's third question.** The
model is an LP capacity-expansion model of Nigeria's power sector to
2054 with an endogenous financing layer — per-tranche capital recovery,
a concessional envelope, a bankability constraint.

**Changed 2026-10-10.** The previous version of this handover said the
model is "precisely the thing the review's third research question asks
whether anyone does." That prejudges the review's own result. **Report
the count the extraction gives.** If some studies do endogenise the
cost of capital, the thesis's contribution is a differentiated one, not
a unique one, and the chapter says which. A review that returns zero in
exactly the category its author occupies is the first thing a sceptical
examiner probes, so the count has to be reported as found and the
inclusion criteria that produced it stated plainly.

**Chapter 3 is complete** apart from eight literature citations and two
open methodological decisions (§3.4.6(c) and (d), and the B\* definition
at register L83). `docs/HANDOVER_chapter_4.md` summarises the model —
with the same warning as above about summaries.

---

## 2. Standing conventions (non-negotiable, inherited)

1. **Nothing is cited from memory.** Every reference, count and claim is
   verified against a source or marked `[VERIFY]` and left for Zeus.
   This project's chat role has produced eleven failed claims in a
   single verification pass, and several since — including two
   instances of the same error (comparing a quantity against an
   incommensurable denominator) committed in the document that recorded
   the first. Every one was caught by Code reading the file.
2. **Three roles.** Chat designs bounded TASK.md files and writes prose.
   Claude Code executes against real files. **Zeus runs every solve,
   every runner, every commit.**
3. **Chat proposes, Code verifies, Code reports failures rather than
   correcting them silently.** Apply this to the review's counts as
   rigorously as it was applied to the model's results. Every count in
   the chapter is written first as a claim and then checked against the
   extraction data by a separate pass.
4. **Fix by default.** *(Added 2026-10-10.)* When a defect is located
   and the correction is mechanical and bounded, it is applied in the
   same task rather than reported and deferred. Deferring located fixes
   causes task inflation and means the same defect gets re-flagged by a
   later session. Report-only is for defects needing a measurement or a
   decision from Zeus.
5. **Supplying context:** Zeus uploads files rather than pasting their
   contents, so you read only the spans you need. For anything about
   the model, use a bounded Code task or a local run rather than
   quoting from memory.

---

## 3. The chapter's job

Most literature chapters say "little attention has been paid to X."
Yours can say **"of N studies meeting the inclusion criteria, k
endogenise the cost of capital and j treat financing structure as a
decision variable."** That is a different kind of sentence and it is
the entire advantage the PRISMA work buys. Build the chapter around it.

**The commonest failure mode in a thesis PRISMA chapter is to spend
fifteen pages on process and two paragraphs on findings.** Invert that.
The flow diagram, protocol registration, search string and screening
counts go to an appendix with a one-page summary in the chapter. The
synthesis is the chapter.

**Two artefacts carry the whole argument:**

- A **dimensional coverage matrix** — studies against energy-security
  dimensions — showing which dimensions are routinely modelled and
  which are not.
- An **endogeneity classification** of how each study treats cost of
  capital: fixed exogenous WACC, scenario-varied WACC, or endogenous
  financing structure.

Design both before writing prose. If the data does not support one of
them in the form you want, that is a finding about the review, not a
reason to assert it anyway.

---

## 4. Your audience has probably never seen PRISMA

The thesis is defended in a **petroleum and energy resource
engineering** department and submitted to a **Q1 journal**. A flow
diagram and an OSF registration will read as unfamiliar machinery.

Include a short early paragraph — three or four sentences, not a
section — on why a systematic protocol rather than a narrative review,
and what pre-registration buys. Frame it as they would frame a
core-analysis protocol: a method stated and fixed before the data, so
selection cannot be shaped by the result.

That paragraph also answers the first question a sceptical examiner
asks a review chapter — "how do I know you didn't pick the papers that
support you" — and the answer is a timestamped registration.

---

## 5. The Chapter 2 → Chapter 3 bridge

End the chapter (or open Chapter 3) with an explicit mapping from each
review finding to the modelling choice that responds to it.

**The financing mapping is the load-bearing one.** Endogenous financing
maps to §3.4's per-tranche capital recovery, concessional envelope and
bankability constraint. The thesis's own evidence that this matters:
the cost of capital reaches the objective (F-1), and tenor accounts for
roughly half the concessional advantage (F-4) — a decomposition a
blended-WACC formulation structurally cannot produce. **Verify both
against `findings_frozen.md` before using them**; F-4's originally
reported decomposition was withdrawn in October 2026 as unreproducible
and replaced, and F-1's sub-figures moved at the current epoch.

**The self-generation mapping.** Whatever the review establishes about
how SSA studies treat informal or off-grid supply maps to the thesis's
grid-boundary emissions cap with unbounded self-generation.

**Changed 2026-10-10.** The previous version called this "its strongest
policy result." Do not carry that claim. The thesis has several
candidate headline policy results and which is strongest is a Chapter 5
judgement, not a Chapter 2 assertion. The candidates, each to be
verified rather than quoted from here: a grid-boundary cap can raise
*national* emissions (F-21 discrete, F-43 continuous, with a measured
turning point); and Nigeria's own NIRP solar deployment trajectory
reverses the sign of EaaS's gas-scarcity relief in one regime, which is
a finding about NIRP rather than about EaaS.

**Dimensional coverage** maps to which security dimensions the thesis's
results speak to and which it leaves untouched.

**One design fact the bridge needs.** *(Added 2026-10-10.)* The thesis
settled in October 2026 on an **uncapped solar build** as its central
design, with NIRP's cumulative deployment trajectory retained as a
named policy arm rather than an inherited default. If the review has
anything to say about how studies treat national deployment plans as
constraints versus as objects of analysis, that is where it attaches.

**Be honest where the model does not answer a gap it identifies.** The
model is single-node, has no integer structure, no cross-border trade
and no price-elastic demand. §3.7 records every limitation with its
direction of bias, and the register is now past 80 entries — read it for
the current count rather than quoting one.

---

## 6. What to avoid

**Do not let this become a technology review.** "Solar costs have
fallen, gas is a transition fuel, Nigeria has a capacity problem" is
background. It belongs in Chapter 1's context section if anywhere.

**Do not restate the protocol as prose.** Inclusion criteria, databases,
date ranges and screening decisions go in a table or the appendix.

**Do not soften a sparse count into "limited attention."** If the number
is two of forty, say two of forty. The precision is the point.

**Do not report a count that flatters the thesis without checking it
twice.** The one category where this review's result directly serves
its author is the one an examiner will test hardest.

---

## 7. First moves

1. Ask Zeus for the review materials and read them. Do not draft first.
2. Propose the two artefacts — coverage matrix and endogeneity
   classification — in the exact form they would appear in the chapter,
   and have Zeus confirm the data supports them.
3. Propose a section skeleton. Agree it before any prose.
4. Then write, section by section, with every count traceable to the
   extraction data and marked `[VERIFY]` where it is not.

Chapter 1 follows this chapter in a separate chat with its own
handover. Do not draft Chapter 1 material here.
