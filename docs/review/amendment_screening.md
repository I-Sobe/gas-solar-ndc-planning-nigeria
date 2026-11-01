# OSF Protocol Amendment — Eligibility Refinement and AI-Assisted Screening Procedure

**Review title:** Energy security in optimisation-based power-system planning for developing and emerging economies: a systematic review of dimensional coverage and the endogenous treatment of financing structure and cost-of-capital
**OSF registration:** 
**Amendment date:** 10-07-2026

---

## 0. Status and honesty declaration

This amendment is registered **before any title/abstract screening of the deduplicated corpus has been performed.** No inclusion/exclusion decisions on the screening corpus have been made at the time of registration. The corpus size and its metadata characteristics (including the abstract-less stratum described in §3) have been observed; the screening itself has not been conducted. This preserves the foreknowledge position declared in the registered protocol's §0: *authors have observed the data, but have not performed the proposed analyses.*

The purpose of this amendment is to (a) refine the geographic eligibility criterion to align the corpus with the review's answerable question, (b) declare the AI-assisted screening procedure and its pre-committed stopping rule, (c) declare the single-reviewer reliability-validation design, and (d) record compliance with the 2025 multi-organisation position statement on AI use in evidence synthesis.

Each change is stated below as a deviation-from-registered-protocol with rationale, per reproducible-amendment practice.

---

## 1. Amendment 1 — Geographic eligibility refined to Sub-Saharan Africa

### 1.1 Amended criterion
Studies are eligible on geographic grounds if and only if they meet one of the following:

- **(a)** The study's empirical application covers one or more Sub-Saharan African countries.
- **(b)** The study is multi-region and Sub-Saharan Africa (or one or more SSA countries) is a **distinct analysed unit** within it — i.e. results are reported for the SSA case specifically, not only in aggregate.

Studies are **excluded** on geographic grounds where:

- **(c)** Sub-Saharan Africa appears only as background, motivation, or passing mention, with no SSA-specific analysis or results.
- **(d)** The study's empirical scope is wholly outside Sub-Saharan Africa (including studies confined to North Africa, which falls outside the SSA definition adopted here).

**Operational definition of "Sub-Saharan Africa":** States here are as defined in the World Bank Sub-Saharan Africa classification.

**Method-agnostic clause:** This geographic refinement does not alter the methodological eligibility criteria registered in v4.0 (e.g. the simulation-vs-optimisation boundary, Rule R1). A study meeting the methodological criteria is included or excluded on geography solely by the rule above.

### 1.3 Rationale
The refinement aligns the screening corpus with the review's answerable question and with the thesis's empirical focus, and brings the corpus to a volume screenable to a consistent standard by a single reviewer. It is a narrowing (more restrictive) change; it cannot introduce records that a broader criterion would have excluded, so it presents no risk of selective inclusion of newly-favourable studies. The rule is stated operationally, with worked edge cases (a)–(d), so that it is applied reproducibly across all records rather than as case-by-case judgement.

### 1.4 Effect on the corpus
This criterion will be applied **during** screening (not as a pre-filter), so that every excluded record is counted and its exclusion reason recorded for the PRISMA 2020 flow diagram. Records excluded under this criterion are logged with the reason "outside Sub-Saharan Africa scope."

---

## 2. Amendment 2 — AI-assisted title/abstract screening with pre-committed stopping rule

### 2.1 Amended procedure
Title/abstract screening of the deduplicated corpus will be conducted using **ASReview** (open-source active-learning screening prioritisation; Utrecht University). The active-learning model reorders the screening queue so that records more likely to be relevant are presented earlier, based on the reviewer's accumulating inclusion/exclusion decisions.

- **Model seeding:** The model is initialised with a set of known-relevant records (including the pre-registered positive controls Agutu et al. 2022 and Oyewo et al. 2018) and a set of clearly-ineligible records, to provide initial signal. The seed set is recorded as an artefact.
- **Screening basis:** Records are screened on title and abstract against the full eligibility criteria (as amended in §1). Records lacking an abstract (the stratum in §3) are screened on title under the retention-on-doubt rule.
- **Reviewer:** Single reviewer (Author). All decisions are logged by ASReview with timestamps.

### 2.3 Pre-committed stopping rule
Screening under active learning does not require screening 100% of records; it requires a pre-specified, data-driven stopping point. The stopping rule for this review, **committed here before screening begins**, is:

>  - "Screening stops after 100 consecutive records are screened without a single inclusion." 

The rule is fixed at registration and will not be relaxed mid-screening. The record index at which the rule triggered, and the number of records screened versus total, will be reported in the PRISMA narrative.

### 2.4 Rationale and honest limitation
Active-learning screening with a pre-registered stopping rule is an established method for reducing single-reviewer workload on large corpora. **The review does not claim any specific recall figure** (e.g. benchmark "95% recall" values reported in the ASReview validation literature derive from specific benchmark datasets and do not transfer to this corpus). The claim made is narrower and defensible: an active-learning tool was used with a pre-committed stopping rule, and the resulting exclusions were independently validated (§4). The residual risk that a relevant record beyond the stopping point was not screened is acknowledged and is the reason for the validation step in §4.

---

## 3. Amendment 3 — Abstract-less stratum: title-only screening under retention-on-doubt

### 3.1 Background
Direct inspection of the deduplicated corpus established that a stratum of records (see search-development log Entry 9) carry no abstract and no DOI-recoverable abstract. A staged recovery was attempted — file-lookup against the raw source exports, DOI backfill, and Crossref API retrieval — and returned no abstracts, because the associated DOIs (predominantly Elsevier) carry no abstract field in the Crossref record. The recovery attempt and its null result are documented and reproducible from the retained artefacts.

The earlier characterisation of this stratum as predominantly non-research boilerplate was tested and did not hold: the large majority are genuine, on-topic research records. They are therefore not disposable and must be screened.

### 3.2 Amended procedure
Records in the abstract-less stratum are screened **on title alone** against the eligibility criteria, under a **retention-on-doubt rule**: a record is excluded at title-only screening only where the title *alone* establishes clear ineligibility. Any record whose title does not clearly establish ineligibility is **retained through to full-text retrieval**, where the full text is assessed directly and the missing abstract is moot. **No record is excluded on the basis of a missing abstract alone.**

### 3.3 Rationale
This is a deviation from the registered abstract-level screening step, applied to this stratum only. It is **more inclusive** (stricter against wrongful exclusion) than abstract screening, not weaker. The stratum size and the title-only screening basis are reported in the PRISMA narrative, with the documented recovery attempt as the justification for the title-only route.

---

## 4. Amendment 4 — Single-reviewer reliability validation (independent-model second screener)

### 4.1 Design
Because screening is conducted by a single reviewer, a reliability check is applied to detect potential false exclusions. On a random sample of **excluded** records, an **independent classifier** — distinct from the ASReview model used for primary screening — is used as a second screener to flag records it judges potentially relevant. Flagged records are manually adjudicated by the reviewer, and the false-exclusion rate is reported.

### 4.2 Independence requirement (stated to prevent circularity)
The second-screener model **must be independent of the primary screening model.** ASReview is not used to validate its own exclusions. The independent second screener will be an LLM-based relevance classifier, applied under a fixed, recorded prompt/criteria specification.

### 4.3 Sample size
A random sample of 300 excluded records is drawn for the supplementary check, stratified to over-sample doubt-flagged and near-threshold exclusions. The size is set by the rule-of-3: observing 0 false exclusions in 300 places the upper 95% confidence bound on the true false-exclusion rate at approximately 1% (3/300). A minimum of 200 (≈1.5% bound) is the acceptable floor if 300 proves impractical. The observed false-exclusion count and its implied upper bound are reported.

### 4.4 Rationale
Using an AI tool as a secondary reviewer to detect false exclusions in single-reviewer screening is a documented method in the evidence-synthesis methods literature. Keeping the second screener independent of the primary model is what makes the check meaningful; a same-model check would measure nothing.

---

## 5. Amendment 5 — Compliance with 2025 AI-in-evidence-synthesis position statement

The AI-assisted procedures declared in §2 and §4 are reported in accordance with the November 2025 joint position statement on AI use in evidence synthesis issued across Cochrane, Campbell, JBI Evidence Synthesis, and Environmental Evidence.

---

## 6. Summary of artefacts retained for reproducibility

- Deduplicated screening corpus (RIS + CSV), frozen at screening start.
- Abstract-recovery audit table, Crossref response cache, and merged recovery output (Entry 9 evidence).
- ASReview project file, including the seed set, full decision log with timestamps, and the point at which the stopping rule triggered.
- Screening rulebook (companion document) with the operational SSA definition and worked edge cases.
- Independent second-screener specification, validation sample, flagged records, adjudication decisions, and computed false-exclusion rate.
- PRISMA 2020 flow-diagram counts at every stage.

---
