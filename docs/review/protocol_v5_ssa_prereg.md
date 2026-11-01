# Systematic Review Protocol — Screening, Extraction, and Synthesis Plan (for OSF registration)

**Working title:** Energy security in optimisation-based power-system planning for **Sub-Saharan Africa**: a systematic review of dimensional coverage and the endogenous treatment of financing structure and cost-of-capital

**Protocol version:** 5.0 (SSA-scoped; screening-focused; honest-scope preregistration)
**Registration date:** [POPULATE — date of definitive registration]
**Reporting standard:** PRISMA 2020
**Review type:** Regional methods-and-concepts systematic review

---

## 0. Scope of this preregistration (READ FIRST — the integrity statement)

This preregistration is **honest about what it does and does not cover.**

**Geographic scope is Sub-Saharan Africa (SSA), strictly.** This is a *regional* systematic review. Its contribution is bounded to the SSA power-planning literature: the review asks whether, **within SSA**, optimisation-based planning models conceptualise energy-security dimensions and integrate financing endogenously. It does **not** claim these methods are absent globally — endogenous-financing and cost-of-capital work exists elsewhere (e.g. global-IAM and OECD/North-African studies). The claim is geographically located: the review characterises the state of the literature *for the region the companion thesis addresses (Nigeria, SSA)*. Regional applicability is itself the contribution: methods developed for other financing environments do not transfer costlessly to SSA, where cost-of-capital conditions and data environments differ materially (cf. Egli et al. 2019 on uniform-CoC distortion; Steffen et al. 2025 on SSA CoC-data sparsity). **The scope is defined by the review's research object (an astute body of work for SSA), not by corpus tractability.**

**What is NOT preregistered (already done, documented, not claimed as pre-committed):**
The literature *search* was developed iteratively and executed before this registration; the author has **observed search-level outputs** (counts; which validation-control papers appear in which sub-search) and refined strings accordingly. The search is therefore **documented, not preregistered** (§5, Search-Development Log). Note the original search was scoped to developing/emerging economies broadly; the SSA narrowing and its consequences for the corpus are handled transparently in §5.2 and §5.8 (over-retrieve-then-screen; supplementary SSA-tuned search).

**What IS preregistered (uncontaminated, the object of this registration):**
The **screening, extraction, and synthesis plan** — eligibility criteria and their application (including the SSA geographic gate, §4.5), the model-class coding scheme, the Access/Deliverability operationalisation, adjudication rules R1–R5, the extraction form, and the synthesis-matrix structure and interpretation. **The author certifies: no systematic screening has been performed** — no eligibility criteria have been applied to decide inclusion/exclusion of any individual record for the SSA-scoped corpus, and no SSA-scoped screening decisions have been recorded. Incidental viewing of individual records during search calibration is documented in the Search-Development Log and did **not** involve applying §4 criteria or recording inclusion decisions.

> **[DECISION-POINT A — verify before registering]:** The certification is written to be true whether or not you inspected individual records during calibration (it is scoped to "applying eligibility criteria / recording inclusion decisions," not to "never having seen a record"). If you genuinely only saw counts and DOI-presence checks, you may use the stronger wording "only aggregate counts and targeted validation-control checks were observed at the record level." Confirm this matches your actual actions before registering.
>
> **[DECISION-POINT S — SSA-scope screening reset]:** Any title/abstract screening performed under the earlier developing/emerging (global) scope is superseded by the SSA scope. Records labelled under the broad scope must be re-screened against the SSA geographic gate (§4.5) before their labels count. This is a scope reset, not a bias event, and is declared here.

**OSF foreknowledge certification (honest selection):** *"Authors have observed the data, but have not performed the proposed analyses."* Search-level evidence has been observed; the proposed analyses (SSA-scoped screening by criteria, extraction, synthesis) have not been performed and will not be until after registration.

**Dating:** All dates are true action-dates; nothing is backdated. Chronology: search developed and documented → SSA scope fixed → this plan registered → corpus screened.

---

## 1. Rationale (brief)

Optimisation-based power-system planning models are the principal quantitative tools for least-cost expansion in Sub-Saharan Africa, where cost-of-capital is a binding constraint rather than a background parameter (Ondraczek, Komendantova & Patt 2015; Steffen et al. 2025) and uniform-CoC assumptions distort model outputs (Egli, Steffen & Schmidt 2019; Lonergan et al. 2023). The conceptual energy-security literature names financing as security-relevant (Sovacool & Mukherjee 2011; APERC 2007), but whether SSA-focused planning models operationalise it *endogenously* is unestablished. Whether that cell is empty or sparse **within SSA** is the empirical question this review answers, run open. SSA is also the region of the companion thesis (a Nigeria capacity-expansion model), so the review's gap directly motivates the thesis contribution.

## 2. Review questions

**Primary:** How does the optimisation-based power-system-planning literature **for Sub-Saharan Africa** conceptualise energy-security dimensions, and to what extent is financing structure and cost-of-capital integrated *endogenously* — where are the gaps?
**Sub-questions:** SQ1 Availability; SQ2 Deliverability (technical-supply Accessibility); SQ3 Affordability (incl. cost-of-capital); SQ4 Acceptability; SQ5 Financing integration {absent | exogenous | endogenous} and its intersection with SQ1–4, within SSA.

## 3. Conceptual framework (fixed before screening)

Organising frame: **APERC (2007) Four A's** — Availability, Accessibility, Acceptability, Affordability. **Accessibility is split** (per primary-source verification of APERC) into **Deliverability** (APERC-original, supply-side/technical) and **Access** (socio-economic/SDG7, a later graft). The literature's conflation of the two is a reportable finding IF the extracted data support it. Financing/cost-of-capital is a **cross-cutting synthesis-matrix axis**, not a fifth "A."

*Framework anchors (confirmed by the author against primary sources; the attestation is the author's, inherited by this registration):* Ang, Choong & Ng (2015); APERC (2007); Sovacool & Mukherjee (2011); Egli et al. (2018, 2019); Ondraczek et al. (2015); Lonergan et al. (2023). *Used only for framing; no anchor's specific figure is load-bearing for an inclusion/exclusion decision.* Note several anchors are non-SSA (German/global); they motivate the framework and the "exists elsewhere, not established for SSA" positioning — they are not in-corpus and their geography does not widen the review's scope.

## 4. Eligibility criteria and adjudication rules (THE PREREGISTERED CORE)

### 4.1 Criteria
**Include:** models that solve for a power-system **plan** by cost-minimisation — mathematical-programming (LP/MILP/partial-equilibrium), national-resolution IAM, simulation–optimisation hybrids where capacity is a decision variable, and geospatial least-cost allocation (OnSSET/REM/GEP class); **Sub-Saharan African** system, national scale (or sub-national within one SSA country); Jan 2015–present; English; journal articles, conference papers, reviews.
**Exclude:** simulation-only / operation-only tools with no capacity-cost-minimising objective (LEAP-style; dispatch-only optimisers — R1) [edge case A]; global-resolution IAMs at global scale (MESSAGE-GLOBIOM, GCAM, REMIND) [edge case B]; **any system outside Sub-Saharan Africa (including North Africa and OECD systems)** — see §4.5; non-English; editorials/comments; theses/dissertations as documents (see §4.6).
**Financing content is NEVER an exclusion criterion.** Absence or exogeneity of financing treatment is extracted *data*, not grounds for exclusion. **"Equilibrium" (includable) means partial-equilibrium electricity-market optimisation; computable general-equilibrium (CGE) macro models are excluded** (economy-wide, not a power-system capacity plan).

### 4.2 Mandatory model-class sub-code (scope-drift guardrail)
Every included study is coded: {mathematical-programming | geospatial-least-cost | simulation–optimisation hybrid | partial-equilibrium | national-IAM}. The synthesis matrix is stratified by this code.

### 4.3 Rule R1 — simulation-vs-optimisation boundary
> A model is included only if it solves for a power-system *plan* — at least one generation or transmission *capacity* is a decision variable determined by a cost-minimising/welfare-optimising objective. EXCLUDED under edge case A only if **all** capacities are exogenous and solely operation/dispatch is optimised — irrespective of (i) any internal dispatch optimisation, (ii) financing sophistication, or (iii) effect on the gap claim.

*Worked example:* Schinko et al. (2019) optimises dispatch over stakeholder-fixed capacities → fails R1. **Note: Schinko is also now out of geographic scope (Morocco = North Africa, §4.5); under SSA-strict it is excluded on geography regardless of R1.** It is retained as the R1 worked example for the boundary logic only.

### 4.4 Time-window justification
2015 lower bound set by the NDC/Paris architecture as the analytical object (substantive, not recency). Pre-2015 foundational work via a labelled lineage paragraph + backward snowballing (a distinct PRISMA source).

### 4.5 Geographic gate — Sub-Saharan Africa, strictly (pre-committed)
> A study is in scope only if its modelled power system is located in **Sub-Saharan Africa**. SSA is defined as the African continent **excluding** the North African states (Morocco, Algeria, Tunisia, Libya, Egypt, and Sudan where treated as North African per the applied reference list). **North Africa is OUT. Any non-African system is OUT.** Multi-country studies qualify only if all modelled systems are SSA (a study mixing SSA and non-SSA countries is included only for its SSA components, and only if those are separately modelled — cf. R2). The geographic gate is applied **first** at screening (cheapest filter): non-SSA records are excluded on geography without model-type assessment.
>
> *Reference definition:* [POPULATE — cite the specific SSA country list used, e.g. UN M49 sub-Saharan Africa or World Bank SSA region, dated. Fix one list and apply it verbatim so borderline cases (e.g. Sudan/Mauritania) are decided by the list, not ad hoc.]

### 4.6 Document-type rule — theses/dissertations (pre-committed)
> Doctoral/master's **dissertations are excluded as documents** (not journal/conference/review). Where a dissertation's chapters are published as journal articles, those articles are sought (backward/forward snowballing) and screened normally. Rationale: peer-reviewed publication is the corpus boundary; dissertation content is not lost but is captured via its published derivatives.

### R2–R5 (scope/scale sub-rules; committed separately, applied uniformly)
- **R2 multi-national regional scale:** [committed STRICT/PERM — within SSA]. Under SSA-strict, R2 governs multi-country *SSA* studies (e.g. an East-Africa pooled model): STRICT = pooled multi-country excluded, separate-national SSA runs included per country.
- **R3 power-system object vs whole-energy-system:** [committed STRICT/PERM]. PERM = qualifies on an optimised, central power-sector component; excluded if power is a non-optimised node or peripheral to a non-power object.
- **R4 global-IAM national readout:** [committed STRICT/PERM]. STRICT = a global multi-region IAM with an SSA-country readout is excluded (optimisation boundary is global), consistent with edge-case-B.
- **R5 system vs device scale:** [committed STRICT]. National/sub-national fleet or jurisdiction-wide geospatial allocation IN; single-installation/single-site designs OUT.

## 5. Information sources and the (documented, not preregistered) search

### 5.1 Databases
Scopus and Web of Science **not accessible** at the author's institution (declared limitation, §9). Sources: **Lens.org (Academic Account), OpenAlex, IEEE Xplore** (free, Boolean-capable). Google Scholar for snowballing/known-item only. Grey literature (IEA/IRENA/World Bank/AfDB) as a separate labelled SSA stream.

### 5.2 Corpus strategy under the SSA narrowing (transparent)
The primary search was executed under a **developing/emerging-economies (global)** geography and produced a deduplicated corpus of [POPULATE — the deduplicated N]. The review scope is **narrower** (SSA-strict) than the search. This is handled as standard PRISMA **over-retrieval then screen-down**: the SSA geographic gate (§4.5) is applied at title/abstract screening, and non-SSA records are excluded on geography and counted in the PRISMA flow ("excluded on geography: not SSA"). The search being broader than the final scope is a strength (it cannot under-retrieve by having scoped too narrowly a priori), provided SSA recall is demonstrated — see §5.8. **This design is a deliberate, documented decision, not a limitation of the search.**

### 5.3 Search architecture (as executed on Lens; documented)
Three concept-anchored sub-searches, unioned and deduplicated (Lens ~2000-char URL limit forced the split; a single four-block query returned HTTP 414):
- **Search A (Security):** METHOD AND (security/dimension terms) AND GEOGRAPHY
- **Search B (Financing):** METHOD AND (financing/CoC terms) AND GEOGRAPHY
- **Search C (Electrification/Access):** METHOD AND (electrification terms) AND GEOGRAPHY

Each scoped Title+Abstract via in-string `title:()`/`abstract:()`. The verbatim executed strings (broad developing-economy geography) are in §5.5. **The executed GEOGRAPHY block was broad (developing/emerging descriptors + a multi-region country list); SSA scoping is applied at screening (§4.5) and, additionally, via the supplementary SSA-tuned search (§5.8).**

### 5.4 Observed platform behaviour (documented)
On re-execution of identical strings, Lens returned drifted counts (e.g. Search A 2,969→10,084) with per-search control membership shifting, though controls remained in the union. Count increase plausibly reflects index growth; the drop-out of already-indexed controls from individual sub-searches is **not explained by index growth and its cause is left undetermined pending the within-session stability check**. Resolution (pre-committed): definitive search on one fixed date, immediate RIS export (freezing the corpus), within-session stability check (each search twice, minutes apart), definitive counts reported. A demonstrated recall property — a single Lens sub-search can drop a known-relevant already-indexed record between runs, mitigated by the union — is declared in §9.

### 5.5 Executed Lens search strings
[POPULATE — the three verbatim strings AS EXECUTED (broad developing-economy geography), with the corrected lowercase `abstract:` tags. These are documented as executed; the SSA narrowing is applied at screening (§4.5) and via §5.8, NOT by silently editing these post-hoc. If a definitive re-run is done, it uses these exact strings and its counts populate the PRISMA flow.]

### 5.6 OpenAlex and IEEE Xplore (pending, same architecture)
Three-search architecture adapted to OpenAlex and IEEE Xplore (platform field syntax; IEEE two-arm split), each validated against the SSA controls before entering the union. Strings and counts: [TO BE POPULATED]. **Still owed — a single-database review is a criticised weakness.**

### 5.7 Validation controls (search-adequacy check; both SSA)
The search must return, in the deduplicated union:
- **Agutu et al. (2022)** Nat Energy 7:631–641, DOI 10.1038/s41560-022-01041-6 — geospatial-least-cost (OnSSET), SSA, endogenous country/mode CoC; Access. **SSA — in scope.** IN.
- **Oyewo et al. (2018)** Energy Convers Manag 178:44–64, DOI 10.1016/j.enconman.2018.10.036 — LP capacity-expansion, Nigeria (SSA), exogenous flat 7% WACC. **SSA — in scope, closest analogue to thesis.** IN.

*Both controls are SSA, so both remain valid under SSA-strict.* The validation set contains no positive control that is simultaneously endogenous-financing AND national capacity-expansion within SSA — because that is the cell hypothesised sparse/empty for SSA. Agutu (endogenous, geospatial/Access) and Oyewo (capacity-expansion, exogenous) bracket the cell from two sides. This is a property of the hypothesis, not a search defect.

**Documented exclusions (behaviour check):** Calcaterra et al. (2024) — global IAM (edge-case-B) AND non-SSA-specific → excluded. Okorie (2025) — CGE, continental/economy-wide → excluded. **Schinko et al. (2019) — Morocco (North Africa) → excluded on geography (§4.5); also fails R1.**

### 5.8 Supplementary SSA-tuned search (recall assurance for the narrowed scope)
Because the primary search was calibrated for global-developing scope, a supplementary search with an **SSA-specific geography block** (SSA country names + regional descriptors: "sub-Saharan Africa," "West/East/Southern/Central Africa," individual SSA country names) is run and unioned with the SSA subset of the primary corpus. Purpose: detect SSA records the broad search may have under-recalled. Validated against both SSA controls (Agutu, Oyewo). Outcome logged: if the supplementary search surfaces little beyond the SSA subset, broad-search SSA recall is evidenced adequate; if it surfaces substantial new SSA records, they are folded in and the under-recall is reported. [Counts TO BE POPULATED.] This converts SSA recall from an assumption into a demonstrated property.

## 6. Screening (PREREGISTERED)
Two-stage (title/abstract → full-text), both against §4 criteria and R1–R5, on the frozen RIS export imported to Zotero (deduplicated; "records after duplicates removed" eyeballed) and screened in ASReview (active-learning title/abstract) then Rayyan/full-text.
- **Geographic gate (§4.5) applied first** — non-SSA excluded on geography, cheapest filter, no model assessment needed.
- **Label discipline:** "relevant" = advances to full-text, NOT confirmed-included; retention-on-doubt (exclude only if the abstract alone establishes clear ineligibility); log every exclusion with the gate/rule reason.
- **Stopping rule (pre-committed):** [POPULATE — e.g. halt after N consecutive irrelevant records once the ASReview model has stabilised on a sufficient labelled base including both controls; report unscreened fraction as presumptively excluded in PRISMA. Commit N BEFORE it fires.]
- **Reproducibility:** exhaust the second-independent-screener route first (Cohen's κ reported); fallback = self-re-screen of 20% (floor 100), ambiguity-stratified, fortnight washout, κ reported honestly.

## 7. Extraction (PREREGISTERED)
Per included study: bibliographic; SSA country/system; **mandatory model-class sub-code**; temporal/spatial resolution; each security dimension (Availability/Deliverability/Access/Affordability/Acceptability: Y/N + mechanism); Accessibility reading (Deliverability/Access/both/unspecified) + names-the-distinction (Y/N); **financing treatment {absent | exogenous | endogenous} + mechanism**; uncertainty treatment; stated limitations.
*Worked coding of controls:* Agutu {geospatial-least-cost; Access; endogenous}; Oyewo {mathematical-programming; Availability+Deliverability+Affordability; exogenous flat 7%}.

## 8. Synthesis (PREREGISTERED)
Synthesis matrix (studies × 5 dimensions × financing-treatment), stratified by model-class, **for the SSA corpus**. Funnels from dimensional coverage to the specific under-addressed cell (endogenous financing within SSA national, security-oriented capacity-expansion). **The gap is evidenced from empty/sparse cells, not asserted.** If cells expected sparse are populated, this is reported and the gap statement narrowed. **All gap claims are geographically bounded to SSA and stated as such; the review does not claim global absence.**

## 9. Limitations (declared a-priori)
Regional scope (SSA) bounds generalisability by design — the contribution is regional applicability, not global novelty; absence of Scopus/WoS (mitigated by Lens/OpenAlex/IEEE + control validation + §5.8 supplementary search); **primary search broader than SSA scope, narrowed at screening** (mitigated by the supplementary SSA-tuned search, §5.8, which evidences SSA recall); Title+Abstract-only field scope (keyword/FoS dropped for URL limits); demonstrated Lens recall instability across runs (mitigated by union + freeze-on-export); English-only; grey-literature non-systematicity; dissertation exclusion (mitigated by snowballing published derivatives, §4.6); SSA CoC-data sparsity (Steffen et al. 2025).

## 10. Quality appraisal
Not an effectiveness review; no RoB2/ROBINS-I. Substitute: methodological-transparency appraisal keyed to whether each study states its discount-rate/financing assumptions explicitly and reproducibly.

## 11. Open-review commitment (PREREGISTERED)
Screening criteria, R1–R5, the SSA geographic gate, extraction categories, and matrix structure are fixed by this registration before any SSA-scoped screening. The author commits to reporting the financing cell's occupancy **within SSA** honestly even if fuller than the thesis assumes; a fuller cell narrows the contribution from "fills an empty SSA cell" to "extends a thin, fragmented SSA treatment" — still a valid Masters contribution. No claim about the cell's occupancy is made in this registration; occupancy is an output of §8.

## 12. Registration mechanics
Register THIS plan on OSF Registries (immutable, timestamped). Foreknowledge certification per §0. Cite the OSF DOI in the manuscript. Post-registration changes permitted only if logged transparently as dated amendments referencing the frozen snapshot.

---

### [TO BE POPULATED] (post-registration execution)
OSF DOI/date; SSA country-list reference (§4.5); definitive search date + counts; supplementary SSA search counts + recall outcome (§5.8); OpenAlex + IEEE strings + counts; deduplicated union + SSA-subset counts; full PRISMA 2020 flow (with "excluded: non-SSA" as a counted category); screening outcomes + κ; stopping-rule outcome + unscreened fraction; synthesis matrix.

### Companion documents (OSF project workspace, not part of the frozen registration)
- Search-Development Log (dated execution record, incl. the SSA-narrowing decision and the supplementary search).
- Scope/scale sub-rules R2–R5 (committed versions).
