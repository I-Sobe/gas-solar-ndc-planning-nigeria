# HANDOVER — NDC-EaaS thesis, Chapter 1 (Introduction)

*v2. Revised after Chapter 2 was drafted. Changes from v1 are listed at
the end; the substantive additions are §3 (the numbers, handed over
rather than referred to), §4 (the traps), and §2's governing
rule.*

**Chapter 2 is now written.** Its draft is at
`docs/chapter_2_draft_v3.md`. Chapter 1's gap statement, research
questions and contribution list are all claims about what the
literature does not do, and Chapter 2 measures exactly that — so the
numbers are available and do not need to be re-derived.

**Chapter 1 gets a near-final draft now and a final pass after Chapter
5**, because its contribution list has to match what the results
actually support.

---

## 1. What exists already

- **Chapter 2** — `docs/chapter_2_draft_v3.md`. Built on a completed
  single-reviewer PRISMA systematic review, with a dimensional coverage
  matrix (dimension × model class) and a derivation × swept
  classification of how studies treat cost of capital. The counts are
  reproduced in §3 below so Chapter 1 does not have to go looking.
- **Chapter 3** — complete, in seven files. §3.4 is the financing layer
  and is the thesis's methodological contribution; §3.5 is the policy
  representation; §3.7 is the limitations. Read §3.4.2, §3.5.3 and §3.7
  before writing any contribution sentence — §4 below says why.
- **`docs/findings_frozen.md`** — F-1 to F-43 with F-23 to F-29
  reserved. **This file is where the contribution list comes from.**
- **`docs/HANDOVER_chapter_4.md`** — summarises the model, the results
  and the constraints any chapter must respect.
- **`docs/limitations_register.md`** — the authoritative defect record.
  **Do not quote its entry count.** It has moved three times in a week
  by design, because it grows whenever a defect is found. Cite it as a
  register reproduced at a stated commit, as §3.7 does.
- **`docs/financing_endogeneity.md`** — the diagnostic that settles how the
  model treats its own cost of capital, in three parts: a static inventory of
  every financing object in both builders, 71 solves across three panels, and a
  classification against the review's definition. **Read Part C before writing
  any contribution sentence about financing.** Trap 2 below is drawn from it.
- **Review artefacts** — `data/review/` and `docs/review/` hold the
  frozen extraction workbook, the full-text screening sheet, the
  protocol, four amendments and the review manuscript, with
  `MANIFEST.sha256`. `docs/review_tables.md` carries every count with
  its provenance.

---

## 2. Standing conventions (non-negotiable, inherited)

1. **Nothing is cited from memory.** Every reference and figure is
   verified against a source or marked `[VERIFY]` for Zeus.
2. **Three roles.** Chat writes prose and designs bounded TASK.md
   files; Claude Code executes against real files; Zeus runs everything.
3. **Every quantitative claim is written as a claim, then checked
   against the file that produced it**, by a separate pass that reports
   failures rather than correcting them. In the Chapter 2 work this pass
   caught arithmetic errors in the *chat's own* task specifications, not
   only in the repo — so the pass checks what Chat asserts as well as
   what the files contain.
4. **No claim in Chapter 1 may be stated more strongly than the claim in
   the chapter it summarises.** Chapter 1 compresses, and compression is
   where qualifiers get dropped. Traps 1, 3 and 4 in §4 are each
   an instance of this rule being broken. If a sentence in Chapter 1
   cannot be traced to a sentence in Chapter 2, 3 or 4 that says at
   least as much and no more, it is overclaiming.
5. The thesis is defended in a **petroleum and energy resource
   engineering** department. The **Q1 target is a paper drawn from the
   model chapters**, not the thesis itself; the systematic review is a
   separate and later submission, and the thesis is deposited before
   either. Chapter 1 therefore cites the review's **OSF registration**,
   not a published paper.

---

## 3. The gap statement — the numbers, handed over

The gap section is a summary of Chapter 2, not a preview of it. One or
two paragraphs, headline counts, pointing forward. If it runs past a
page, material has migrated from Chapter 2.

These are the counts, verified against `data/review/` through three
independent task generations. Cite them with a cross-reference to the
Chapter 2 section, not as free-standing figures.

**Corpus:** 75 included studies, from 23,717 deduplicated records →
2,521 screened → 183 reports sought (175 database, 8 citation
searching) → 138 assessed → 63 excluded with reasons.

**Financing treatment (§2.5.1):** exogenous 65 (86.7%), absent 10
(13.3%), **endogenous 0**.

**Derivation of the exogenous rates (§2.5.3):** 41 of 65 (63%) at or
below `undefended-flat` — a value with no stated source or argument.
9 of 65 (14%) at `precedent-cited` or above. 3 at
`institution-sourced`. **0 at `market-benchmarked`.** 20 of 65 sweep
the rate; 8 of those 20 sit at `undefended-flat`. 8 of 65 differentiate
the rate, and 7 of those 8 sit at `argued` or above.

**Dimensional coverage (§2.3):** availability 75, affordability 75,
deliverability 59, acceptability 50, access 44. Disaggregated by model
class, coverage is **partitioned, not thin**: geospatial models
represent access in 24 of 24 and acceptability in 7 of 24;
mathematical-programming models represent acceptability in 37 of 43 and
**access in 14 of 43**.

**Vocabulary (§2.4):** of the 39 studies operationalising both
deliverability and access, exactly 1 names the distinction; 38 enact it
without naming it.

**Two bounds that travel with every one of these numbers.** First, the
zero rests on a definition tightened *after* registration (Amendment 4,
Chapter 2 §2.5.2): under the registered coding the probe would have
returned one study. Chapter 1 need not rehearse the amendment, but it
must not state the zero as though nothing turned on the definition —
and whatever Chapter 1 says "no study does" must be the same thing
Chapter 2's definition says. Second, every claim is bounded to a
**retrieved-and-read corpus** in **Sub-Saharan Africa**, not to the
field in the absolute. The review is single-reviewer and two registered
validation steps are unexecuted; Chapter 2 §2.6 discloses this, and
Chapter 1 inherits the bound even though it does not repeat the
disclosure.

**n may still move.** The two deferred validation steps could add or
drop a study before the review is submitted. Write the counts so that a
change from 75 to 74 or 76 is a find-and-replace in one place, not a
rewrite of an argument.

---

## 4. Four traps, and they are specific

### Trap 1 — "EaaS financing improves outcomes"

**The results do not support that unconditionally.**

- **F-38** — at the central demand case under no policy, EaaS is worth
  0.11 bn USD against a 27.318 bn anchor. That is 0.4%, and it may be
  zero within the model's own identified band. Check whether that
  comparison has been made before using either number.
- **F-35** — EaaS value has an **interior maximum** in policy
  stringency: nothing without policy, large under unconditional caps,
  nothing again under conditional caps where the cap binds beyond what
  financing can relieve.
- **F-34** — EaaS makes gas scarcity **worse** by 10.6% in one gas
  regime. The mechanism is unexplained.
- **F-36** — EaaS halves cost dispersion as well as cutting the mean,
  which is a bankability result no deterministic solve could produce.

**The conditionality is the contribution.** "Financing structure matters
where and only where something binds, and here is the map of where" is
a stronger and more interesting claim than "financing structure helps,"
and it is the claim the results make. State it in Chapter 1 — not as a
Chapter 5 retreat, and not as a caveat. The 0.11 bn figure is
extractable from the thesis's own results table by a hostile reader.

A second scoping point belongs with this one. §3.4.5 states that the
EaaS mechanism in the model is **budget decoupling plus a bankability
test**, and that the contract-mechanics module exists in the codebase
but is never called by the linear programme. The mechanism is therefore
**generic private capital mobilisation**, not EaaS contracting
specifically. Chapter 1 must scope the claim the way §3.4.5 scopes it.

### Trap 2 — misstating what "endogenous" means here

Both directions are available and both are wrong.

**Understating it.** The model *does* endogenise the cost of capital in the
sense the review's definition requires, and this is measured rather than
argued. `docs/financing_endogeneity.md` reports that with every financing input
held fixed, and only demand, gas deliverability, the emissions cap and solar
yield varied, the realised portfolio cost of capital moves **17.156 bp**. The
same variations move it **0.000 bp** when the concessional envelope is set to
zero — which is how the exogenous treatments in 65 of the review's 75 studies
behave. Both the decision-variable limb and the solution-outcome limb of the
definition are satisfied.

**Overstating it.** Three things the model does not do, each of which Chapter 3
states and a reader can check:

- **It does not solve for a rate.** Both tranche rates are fixed inputs, every
  objective coefficient is a precomputed constant, and the programme is linear
  (§3.4.2). The financing decision is a volume split at fixed prices.
- **Nothing makes a riskier plan face a dearer one.** No rate, capital recovery
  factor, tenor or discount window in either builder depends on any solved
  quantity. There is no risk-to-rate feedback.
- **The third limb is not claimed.** Financing as a constraint interacting with
  the capacity solution is not asserted, because the bankability constraint's
  effect falls on the public/service-financed split, which F-12 establishes as
  an interval at fixed total cost rather than a point. Total solar and system
  cost are unchanged by it at the configurations tested.

**And state the size.** At the DARES envelope, concessional capital is 1.87% of
the plan's concessional-eligible capital and moves the portfolio rate 16 bp
below commercial; at ten times that envelope it reaches 163 bp. Report the
smallness as a result — a facility of this size cannot move the cost of capital
of a programme this large — not as a caveat or an apology.

*v2 of this handover stated this trap backwards, asserting that the model does
not endogenise the cost of capital. The first draft of Chapter 2 made the same
error. Both were corrected only after the diagnostic was run. Read
`docs/financing_endogeneity.md` Part C before writing the contribution
sentence; do not reason it out from §3.4.2's prose, which is what produced the
error.*

### Trap 3 — the grid-boundary cap is not an access mechanism

Chapter 2 §2.3 establishes that mathematical-programming models
represent access in 14 of 43 cases, and the thesis's model is one of
them. The tempting move is to present the grid-boundary emissions cap
with unbounded self-generation as the thesis's answer to that gap.
**It is not, on three counts.**

- The cap covers the grid because the NDC commits the gas-power slice
  and private diesel sits outside it (§3.5.3). It is an accounting
  choice, not a coverage mechanism.
- Load leaving the grid is a failure to **deliver** to customers already
  connected, not a statement about who is connected. Chapter 2 §2.4's
  whole finding is that the field conflates these two; the thesis must
  not commit the conflation it measured.
- The model has no connection decision at all — single node, no
  locational variable, demand as a national aggregate (§3.7.3).

**The thesis inherits the access gap rather than narrowing it.** Say so.
What the review *does* justify is modelling the boundary at all: a model
capping system emissions would never discover the leakage, because the
leaking quantity would be inside the cap by construction.

*This trap was also hit in the first draft of Chapter 2.*

### Trap 4 — quoting a service-financed capacity figure as a point

The split between public and service-financed solar is **not identified by the
model**. F-12 reports it as an interval at fixed total cost: at
`ndc3_conditional`, 31,875–48,579 MW under a tight capital case, 10,990–48,579
under moderate, and 0–48,579 under unconstrained. F-35 and F-38 already carry
`[UNIDENTIFIED]` tags on split-derived quantities, and F-43 was corrected from
"EaaS solar rises" to a total.

F-12's reporting rule governs: any such figure must carry its `capital_case`
and its `tariff_case`, and be reported as a point within the stated interval.
**Identified quantities are total solar capacity and system cost. The split is
not one.**

Chapter 1 is where a bare megawatt figure is most likely to appear, because a
contribution sentence wants a number. If the number is a service-financed
capacity, it needs the interval and both cases, or it should be replaced with
an identified quantity.

---

## 5. The contribution list comes from the findings file

Write it **last**, from `findings_frozen.md`, and make each contribution
the complement of a gap Chapter 2 measured. A contribution with no
finding behind it is an ambition; a finding with no gap behind it is not
a contribution.

Check each candidate against **two** files, not one: `findings_frozen.md`
for whether the result exists, and §3.7 plus the limitations register for
whether the model can support the claim being made of it. Then check a third
thing: whether the quantity the contribution rests on is identified. If it is a
split-derived figure, F-12's reporting rule applies (Trap 4). A contribution
the register says the model cannot deliver is the failure mode Chapter 1
is most prone to.

The headline results are the backward-bending cost–emissions frontier
under public financing (F-43), the demonstration that a grid-boundary
cap can raise total emissions (F-21, F-39), the conditional value of
financing structure (F-34 to F-38), the bounded value of the stochastic
solution (F-42), and two methodological results about how reliability
standards and threshold grids are specified (F-33 and the boundary
discipline in §3.6.9). Verify each against the file before promoting it.

**On "first" claims.** Any claim of priority — first model to represent
financing structure endogenously in this literature — must carry the
review's own bounds: the retrieved-and-read SSA corpus, under the
definition Chapter 2 §2.5.2 states. The review explicitly makes no claim
about the global literature, and Chapter 1 cannot make one on its behalf.

---

## 6. Research questions

Write them as **questions**, then check each is answerable from
`findings_frozen.md`. A question with no finding behind it is either
unanswered work or a question that should not have been asked. A finding
with no question behind it suggests a missing question.

They must also be answerable *in the form asked*. The model is
single-node, has perfect foresight, has no integer structure, no
cross-border trade and no price-elastic demand, and policy is exogenous
throughout. §3.7.3 records every one with its direction of bias. A
research question that implicitly requires one of them is a question
Chapter 4 cannot answer.

**Chapter 2 asks three questions of its own**, the third being "what does
that leave undone that a national capacity-expansion model could do?"
That question is the hinge into Chapter 1's research questions, which
should read as its answer rather than as a parallel set. Check Chapter 1's
questions against Chapter 2 §2.2.1 for duplication and for contradiction.

---

## 7. Structure

Conventional and sufficient: context, problem, gap, research questions,
contributions, thesis structure. Two notes.

**The context section is where Nigeria goes**, and Chapter 3 now fixes
what belongs there — only what the thesis uses. That is: the scale of
diesel self-generation and its emission factors; gas deliverability as a
binding constraint; the NDC's gas-power slice and what it does and does
not commit; and the financing environment, including the DARES
concessional facility whose domestic deployable portion is the model's
central envelope. Four or five facts that make the research question
necessary — not a history of the power sector.

**The gap section is a summary of Chapter 2, not a preview of it.** The
counts are in §3 above.

---

## 8. What to avoid

**Do not write the gap from intuition.** It is measured. Use §3's
numbers.

**Do not promise what §3.7 says the model cannot deliver.** Chapter 1 is
where overclaiming happens.

**Do not quote a bare count of the limitations register.** It moves by
design. State the count at a commit, or refer to the appendix.

**Do not write the contribution list before Chapter 5 exists**, beyond a
draft. The final pass happens after the discussion, when it is known
which results carried the argument.

---

## 9. First moves

1. Read Chapter 2 §§2.3, 2.5 and 2.7; Chapter 3 §3.4.2, §3.5.3 and §3.7.3;
   `docs/financing_endogeneity.md` Part C; and F-12 in `findings_frozen.md`.
   The four traps in §4 all live in those sections.
2. Propose research questions, and map each to the findings that answer
   it. Bring any question without a finding back to Zeus before drafting.
3. Agree the structure.
4. Draft, holding the contribution list to last.

---

## Changes from v2

1. **Trap 2 rewritten — v2 stated it backwards.** v2 warned against claiming the
   model endogenises the cost of capital. The diagnostic in
   `docs/financing_endogeneity.md` establishes that it does, on two limbs of the
   review's definition. The trap is now the pair of errors available in both
   directions: understating the claim, and overstating it into a rate
   optimisation, a risk-priced rate, or the third limb.
2. **The magnitude is now in the handover** — 17.156 bp across plan variations,
   0.000 bp at a zero envelope, 1.87% of concessional-eligible capital, 16 bp
   below commercial at DARES and 163 bp at ten times it — so Chapter 1 states
   the claim at its measured size rather than at an assumed one.
3. **New Trap 4 on F-12's reporting rule.** The public/service-financed split is
   not identified. Chapter 1 is the likeliest place for a bare megawatt figure
   to appear in a contribution sentence.
4. **§1 adds `docs/financing_endogeneity.md`** and points Chapter 2 at v3.
5. **§5 adds a third check** on candidate contributions: whether the quantity
   the contribution rests on is identified at all.
6. **§9 extends the required reading** to the diagnostic's Part C and to F-12.

## Changes from v1

1. **§3 now hands over the counts** rather than telling the next chat to
   ask for Chapter 2. They are verified across three task generations and
   re-deriving them from a prose draft invites transcription error.
2. **§4 carries three traps, not one.** Traps 2 (the endogeneity claim)
   and 3 (the access conflation) are new. Both were hit in the first
   draft of Chapter 2 and corrected only after Chapter 3 was read.
3. **§2.4 adds a governing rule** — no claim in Chapter 1 may be stronger
   than the claim in the chapter it summarises. Traps 1, 3 and 4 are
   instances of it.
4. **Two moving counts are now flagged as such**: the limitations
   register's entry count, and n = 75, which the deferred validation
   steps may still change.
5. **§2.5 corrects the publication framing.** The Q1 target is a paper
   from the model chapters; the review is a separate, later submission;
   the thesis is deposited first; Chapter 1 cites the OSF registration.
6. **§5 requires contributions checked against two files**, findings and
   limitations, and adds a rule for "first" claims.
7. **§6 adds the Chapter 2 → Chapter 1 question hinge**, and completes the
   list of model boundaries with perfect foresight and exogenous policy.
8. **§7 names what belongs in the context section**, now that Chapter 3
   fixes what the thesis actually uses.
9. **§1 adds the review artefacts** under `data/review/` and
   `docs/review/`, with the manifest and the provenance report.
