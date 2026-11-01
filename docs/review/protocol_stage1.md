# Systematic Review Protocol (a-priori, for OSF registration)

**Working title:** Energy security in optimisation-based power-system planning for developing and emerging economies: a systematic review of dimensional coverage and the endogenous treatment of financing structure and cost-of-capital

**Protocol version:** 1.0 (finalised, pending OSF registration)
**Date:** [TO BE POPULATED — OSF timestamp date at registration]
**Registration:** OSF account created; protocol NOT YET registered. To be registered (frozen, timestamped snapshot) once this finalised version is complete and BEFORE any search is executed (see §11). Creating an OSF account is not registration; registration is the separate act of freezing a dated read-only snapshot of this protocol.
**Reporting standard:** PRISMA 2020 (protocol drafted against PRISMA-P 2015 items)
**Review type:** Methods-and-concepts systematic review (not an effectiveness/intervention review)

> **Status note for author.** All framework-anchoring citations have been primary-source-verified by the author and the [VERIFY] tags removed. Elements still requiring live data from search/screening execution remain marked **[TO BE POPULATED]**; these are the author's to execute and report, and nothing here fabricates a count.

---

## 1. Rationale

Optimisation-based power-system planning models (LP, MILP, equilibrium, national-resolution IAM, simulation–optimisation hybrids) are the principal quantitative tools used to derive least-cost or welfare-optimal expansion and decarbonisation pathways. In developing and emerging economies, two conditions make the *financing* of those pathways analytically central rather than incidental:

1. **Cost-of-capital is a binding constraint, not a background parameter.** The weighted-average cost of capital (WACC) varies across countries far more than the physical resource does. Ondraczek, Komendantova & Patt (2015, *Renewable Energy* 75:888–898) find WACC varying by a factor of ~8.8 across their sample (excluding an extreme outlier) against a solar-resource variation of only ~2.3, and report a country-level WACC–LCOE correlation of 0.82 — i.e. the levelised cost is driven more by financing than by sunshine. Steffen et al. (2025, *Scientific Data* 12:1624) document that this heterogeneity is poorly captured in models and that Sub-Saharan African CoC data are sparse.

2. **Uniform-CoC assumptions bias model outputs.** Egli, Steffen & Schmidt (2019, *Nat. Commun.* 10:4588) show that an energy-system model using a uniform 7% CoC (Bogdanov et al.) inverts the cross-country LCOE ranking once sovereign risk is priced in — solar PV LCOEs up to ~170% higher in developing countries and 7–30% lower in industrialised ones. Egli, Steffen & Schmidt (2018, *Nat. Energy* 3:1084–1092) quantify, for 133 German projects, that lower financing costs account for **41% (solar PV) and 40% (wind onshore)** of total LCOE reduction 2000–2017 (of which experience + interest-rate effects are 5% and 24% respectively). This is the modelling-side lineage the review must engage.

At the same time, the energy-security *conceptual* literature (composite-index and framework studies) names investment, financing and governance as security-relevant dimensions (Sovacool & Mukherjee 2011, *Energy* 36:5343–5355; APERC 2007). The review's motivating observation — to be **demonstrated by the synthesis matrix, not asserted** — is that these two literatures do not meet: financing is named as security-relevant in the conceptual literature but is rarely operationalised *endogenously* (as a modelled financing structure or country-specific capital-cost mechanism) inside optimisation-based planning models for developing systems. Whether that cell is empty or merely sparse is an empirical question this review is designed to answer, including through a search-validation step (§6.4).

**Relationship to the closest prior review (mandatory positioning).** Lonergan et al. (2023, *Joule* 7:469–483) systematically reviewed 58 model-based publications (from 355 Scopus hits, 2010–2021) on how CoC is represented in energy-system models, finding that unfounded own-assumptions remain the dominant "method" and that *no* study differentiated CoC across geography × technology × time simultaneously. This review is adjacent but distinct and must say so explicitly to avoid a redundancy objection: Lonergan asks *"is the CoC parameter represented accurately?"*; this review asks *"is financing structure operationalised as a security dimension within multi-dimensional, security-oriented planning for developing economies?"* Lonergan is a modelling-methods review agnostic to energy-security framing and not bounded to developing economies; the present review is security-framed, developing-economy-focal, and concerned with financing *structure* (capital-structure-dependent constraints, blended/concessional mechanisms) rather than the discount-rate value alone. Lonergan is therefore treated as (i) prior art cited in the introduction, and (ii) a search-validation anchor (§6.4).

## 2. Review questions

**Primary question.**
How does the optimisation-based power-system-planning literature for developing and emerging economies conceptualise and operationalise the dimensions of energy security, and to what extent is financing structure and cost-of-capital integrated *endogenously* into that planning — where do the gaps in integrated treatment lie?

**Sub-questions (one per operationalised security dimension plus one on financing integration):**

- **SQ1 (Availability).** How is energy availability — sufficiency of supply, reserves/deliverability of primary fuels, generation adequacy — represented in the objective, constraints, or scenario structure of these models?
- **SQ2 (Deliverability / technical-supply Accessibility).** How is the *technical/physical* accessibility of supply (fuel-transport infrastructure, gas deliverability, import/interconnection constraints, geopolitical supply access) represented? *(See §4 for the Access/Deliverability split.)*
- **SQ3 (Affordability).** How is affordability — system cost, tariffs, price stability, and (critically) the cost of capital as a determinant of affordability — represented?
- **SQ4 (Acceptability).** How is acceptability — emissions limits, land/water, environmental and social constraints — represented?
- **SQ5 (Financing integration — the funnel target).** To what extent is financing structure and cost-of-capital treated as (a) absent, (b) exogenous (a flat/given discount rate), or (c) endogenous (a modelled financing mechanism, country-specific capital cost, or capital-structure-dependent constraint) — and how does treatment (c) intersect with SQ1–SQ4?

**Framing note.** Structured as a bespoke adaptation for a methods-and-concepts review. SPIDER (Sample, Phenomenon of Interest, Design, Evaluation, Research type) is adapted rather than PICO because the review has no intervention/outcome structure; the "phenomenon of interest" is the *representation* of energy-security dimensions and financing within models. The adaptation is stated explicitly in §3.

## 3. Conceptual framework and operationalisation (screening-grade definitions)

**Organising framework:** APERC (2007) "Four A's" — Availability, Accessibility, Acceptability, Affordability (verified against the primary report) — chosen because (i) it is the documented origin of the widely-used 4As lineage in energy-security quantification (Ang, Choong & Ng 2015 trace their SFA-1/"4As" category to Intharak et al./APERC 2007), placing the review on the taxonomy's headwaters rather than a derivative; and (ii) APERC's own grouping into PHYSICAL (availability + accessibility), ECONOMIC (affordability), and ENVIRONMENTAL security already separates the constructs this review must keep distinct.

**Contested-concept flag (a review finding, to be surfaced, not resolved by fiat).** Energy security is not standardised. Ang, Choong & Ng (2015) surveyed 104 studies, extracted 83 definitions, and inductively derived **seven** recurring themes (energy availability, infrastructure, energy prices, societal effects, environment, governance, energy efficiency) — availability present in 82/83 definitions (99%) but governance and efficiency in only 25% and 22%. *The fact that a careful, systematic review of the definitional literature converges on **seven** themes rather than APERC's **four** is itself evidence for the review's contested-concept claim, and is used as such.* Two cautions the author has verified and must preserve in the manuscript: (a) cite this as "104 studies / 83 definitions / seven themes," NOT as "83 competing definitions of energy security exist" — the latter misstates the paper; (b) Ang et al.'s seven definitional themes (their §3.1) are a *different* classification from the APERC 4As, which appear in Ang et al. only later as the "SFA-1" index-construction category (their §4.3). Do not present the seven themes and the four A's as the same scheme.

**The Accessibility split (operationalisation decision — the review's sharpest Stage 0 contribution).**
Primary-source check confirms APERC (2007) defines Accessibility as *supply-side*: geopolitical, infrastructural and technological access to fuel resources (fuel-transport infrastructure, reserve access, chokepoints) — **not** household electrification or SDG7 energy access. The socio-economic "energy access" reading is a *later graft* by downstream authors. This review therefore splits the dimension for screening and extraction:

- **Deliverability (technical-supply Accessibility)** — the APERC-original meaning. In-scope as a modelled construct (gas deliverability, fuel-transport limits, import/interconnection constraints).
- **Access (socio-economic)** — electrification/energy-poverty. Recorded when present but treated as a *distinct* dimension; **not** merged with Deliverability.

The literature's collision of these two under one word is logged per study (§7) and reported as a finding *if the extracted data support it* — distinguishing (a) individual studies being internally clear but the field using one term for two constructs (terminological collision) from (b) studies being internally confused. The extraction form is built to tell these apart.

**Financing as a cross-cutting axis, not a fifth dimension.**
Financing/cost-of-capital is **not** added as a fifth "A." In APERC it sits inside Affordability; in Sovacool & Mukherjee (2011) "investment" is a component of the technology-development dimension and governance/regulation is a full dimension — i.e. financing is already named but subordinated. This review makes it a *cross-cutting axis* of the synthesis matrix (studies × four dimensions × financing-treatment ∈ {absent, exogenous, endogenous}), so empty/sparse cells are revealed structurally rather than claimed.

## 4. Eligibility criteria

| Criterion | Include | Exclude | Rationale |
|---|---|---|---|
| **Population (system)** | Modelled power/electricity system of a developing or emerging economy, national scale (or sub-national within such a country) | High-income/OECD-only systems (unless as comparator within a developing-economy study); purely global-resolution systems | The financing-as-binding-constraint question is live in developing-economy contexts (§1) |
| **Focal geography** | Sub-Saharan Africa (focal); comparable gas-dependent emerging systems in South/Southeast Asia and Latin America | — | Analytical boundary justified by cost-of-capital salience, not convenience; stated as a limitation (§10) |
| **Model type (inclusive reading — see note)** | Models that **solve for a power-system plan by cost-minimisation**, covering: mathematical-programming (LP, MILP, equilibrium), national-resolution IAM, simulation–optimisation hybrids, **and geospatial least-cost allocation (OnSSET/REM/GEP class)** | **Simulation-only tools with no cost-minimising objective (e.g. LEAP-style accounting/scenario tools)** — see edge case A | The review question concerns endogenous treatment of financing within a cost-minimising plan; the geospatial-least-cost class is included because it is where developing-economy models most often engage country-specific finance, and excluding it would make the "gap" partly an artefact of a narrow inclusion rule |
| **IAM sub-rule** | IAMs **only** where they resolve a specific developing-economy power system at national scale **with explicit financing/cost-of-capital treatment** | Global-scenario IAMs at global resolution (e.g. MESSAGE-GLOBIOM, GCAM, REMIND run globally) — see edge case B | A national-planning review; global IAMs answer a different question |
| **Financing content** | Any treatment of cost-of-capital/discount rate/financing (absent, exogenous, or endogenous) — *absence is itself extracted data, not an exclusion* | — | The distribution across {absent/exogenous/endogenous} **is** the primary finding; excluding "absent" studies would bias the matrix |
| **Publication type** | Peer-reviewed journal articles and archival conference papers; working papers only if from named research programmes and citable | Editorials, comments, non-archival presentations | Standard rigour boundary |
| **Grey literature** | IEA / IRENA / World Bank / AfDB reports handled as a **separate, clearly-labelled source stream** (§5), not merged with the searched database corpus | — | Grey lit informs context and framework but is not systematically searchable to the same standard; kept auditable |
| **Language** | English | Non-English | Sole-screener constraint; declared as a limitation and coverage risk (§10) |
| **Time window** | **January 2015 – [search date]** | Pre-2015 as *searched* evidence | Substantive justification below — NOT recency convenience |

**Time-window justification (substantive, to be stated explicitly in the manuscript).** The 2015 lower bound is set by the emergence of the NDC/Paris architecture as the analytical object of the policy and financing dimensions this review examines: post-2015 is when nationally-determined decarbonisation commitments and the associated climate-finance framing become the planning context. This is a substantive boundary on the *object of study*, not an arbitrary recency cutoff. The truncation of methodological lineage this creates is handled by (i) a narrative methodological-lineage paragraph in the introduction citing foundational pre-2015 works as **background, explicitly labelled as not systematically searched** (e.g. Ondraczek/Komendantova/Patt 2015, Egli et al. 2018), and (ii) backward snowballing (§6.3) reported as a distinct PRISMA source.

**Inclusive model-type boundary and its mandatory guardrail (pre-registered).** The model-type criterion adopts the *inclusive* reading: a model qualifies if it solves for a power-system plan by cost-minimisation, including geospatial least-cost allocation (e.g. OnSSET), not only mathematical-programming capacity-expansion. Worked example: **Agutu et al. (2022)** (modified OnSSET, SSA) is IN-corpus under this reading. This widens the corpus toward the **Access** dimension and toward the electrification literature. The guardrail against scope-drift is a **mandatory model-class sub-code** recorded for every included study (see §7 extraction template): {mathematical-programming | geospatial-least-cost | simulation–optimisation hybrid | equilibrium | national-IAM}. The synthesis matrix is stratified by this sub-code so that the central finding can be stated *by model class* (e.g. whether endogenous financing appears in the geospatial-least-cost class serving Access, but is absent in the mathematical-programming capacity-expansion class where security-oriented generation planning sits). Without this sub-code the review risks collapsing into "a review of electrification models"; with it, the model-class × dimension × financing structure is the analytical instrument.

**Edge cases (pre-specified, not resolved ad hoc):**
- **A — Simulation-only tools (LEAP-style):** EXCLUDED. Rationale: no cost-minimising objective ⇒ financing cannot be treated as an endogenous constraint within a plan, which is the review's precise object. (Distinct from the geospatial-least-cost class, which *does* minimise cost and is included.)
- **B — IAMs:** INCLUDED only at national resolution with explicit financing/CoC treatment; global-resolution IAMs EXCLUDED (see sub-rule above).
- **C — Search-validation set:** the finalised strings MUST return a pre-registered set of known cost-of-capital-in-planning papers (§6.4). If they do not, the strings are deemed mis-specified and revised before screening. These papers determine whether the gap is empty or sparse; failing to capture them would risk overstating the gap.

## 5. Information sources

**Database stack (revised — see coverage-limitation note).** Scopus and Web of Science Core Collection are NOT accessible at the author's institution. The review therefore uses three free, Boolean-capable indexed sources:

- **Lens.org (Academic Account)** — primary source; aggregates Crossref, PubMed, and other scholarly indexes; structured Boolean search with title/abstract/keyword field control and export. An increasingly recognised primary source in published systematic reviews.
- **OpenAlex** — open, comprehensive scholarly index (~250M works; successor to Microsoft Academic Graph); Boolean-capable via web UI and API; open and reproducible.
- **IEEE Xplore** — engineering-venue depth; captures power-systems optimisation venues (MILP/LP capacity-expansion) that carry a disproportionate share of the method-heavy literature. Search and abstract-level screening are free even where full-text is paywalled.

**Backward snowballing** — conducted via the reference lists of included studies and forward-citation via Google Scholar's "cited by" (Google Scholar used for snowballing and known-item checking ONLY, never as a primary search source, due to its lack of reproducible Boolean field control and export). Snowballed records reported as a distinct PRISMA source.

**Coverage limitation (declared).** The absence of Scopus and Web of Science is a stated limitation (§10). The three free sources provide broad aggregated coverage (Lens), open comprehensive coverage (OpenAlex), and engineering-venue depth (IEEE); all three positive-control studies are Crossref-indexed and therefore expected to be retrievable. The validation dry-run (§6.4) additionally functions as a coverage test of the free stack. A supplementary Scopus/WoS coverage cross-check, if a colleague with institutional access can legitimately provide one under their licence terms, may be reported as a distinct validation step — but the primary search is executed by the review author on the free stack, and is fully reproducible on it.
- **Grey-literature stream (separate, labelled):** IEA, IRENA, World Bank, AfDB. Structured retrieval (organisation portals + defined search terms), documented as its own PRISMA source; used for context/framework and to check for planning studies not in the databases. Not merged into the database count.

Record counts for every source are **[TO BE POPULATED FROM SEARCH EXECUTION]**.

## 6. Search strategy

### 6.1 Concept blocks
The search is the Boolean AND of four blocks:
- **Block 1 — Method:** optimisation / planning model classes.
- **Block 2 — Scope:** power / electricity system.
- **Block 3 — Security & financing:** energy-security dimensional terms OR financing/cost-of-capital terms.
- **Block 4 — Geography:** developing/emerging-economy AND focal-region terms.

Full field-tagged strings per database are drafted in **Stage 2** (separate deliverable). This protocol fixes the *logic and eligibility*; the exact strings and their pilot returns are Stage 2 outputs.

### 6.2 Corpus-size intent
Strings will be tuned to hold the pre-screening corpus in the **low thousands** (stated target, to be reported against actual return). Record counts are the author's to execute and report; **none are invented here**.

### 6.3 Backward snowballing
Cited-reference (backward) search from all included studies, to capture foundational/bridging works outside the 2015 window. Snowballed records reported as a **distinct source** in the PRISMA 2020 flow diagram, per PRISMA 2020.

### 6.4 Search-validation set (edge case C — mandatory gate)
Before screening, the finalised strings are run against this pre-registered set (all primary-source-verified by the author). The strings must return the **positive controls**, **or** the review documents explicitly *why* a given paper is legitimately out of scope. Interpreting each hit/miss against the eligibility criteria is the whole point of the gate. Controls must be returned by the *general search*, not injected by DOI; a missing control triggers the block-diagnostic in the Stage 2 document, not a DOI patch.

**Primary positive controls — MUST be returned (three verified in-corpus studies spanning three model classes):**
- **Agutu, Egli, Williams, Schmidt & Steffen (2022)**, *Nature Energy* 7:631–641, DOI 10.1038/s41560-022-01041-6 — geospatial least-cost electrification model (modified OnSSET), Sub-Saharan Africa, **endogenous** cost-of-capital differentiated by country and electrification mode. Class: geospatial-least-cost. Dimension: **Access** (socio-economic). The worked example of the inclusive model-type boundary (§4).
- **Oyewo, Aghahosseini, Bogdanov & Breyer (2018)**, *Energy Conversion and Management* 178:44–64, DOI 10.1016/j.enconman.2018.10.036 — LP capacity-expansion optimisation of Nigeria's power sector (LUT model), hourly, national. **Exogenous, uniform, flat 7% WACC** (verified: 7% system-wide with a 4% residential-prosumer carve-out; no financing module; WACC enters only as the annuity factor, never a decision variable). Class: mathematical-programming. Dimension: **Availability + Deliverability + Affordability**. The archetypal exogenous-uniform case and the closest published analogue to the companion thesis; its exogenous-flat treatment is the review's central worked example of the gap.
- **Schinko, Bohm, Komendantova, Jamea & Blohm (2019)**, *Energy, Sustainability and Society* 9:1, DOI 10.1186/s13705-018-0186-8 — participatory simulation–optimisation hybrid (downscaled renpassG!S), Morocco; optimised dispatch, stakeholder-fixed capacity; **endogenous** technology-differentiated WACC module with market vs de-risking (development-bank) financing regimes. Class: simulation–optimisation hybrid. Dimension: **Affordability + Availability + Acceptability**. The strongest prior art for the endogenous-financing treatment.

**Secondary / lineage anchors — confirm the relevant term-blocks fire (legitimate misses acceptable if out of model-type scope):**
- Ondraczek, Komendantova & Patt (2015), *Renewable Energy* 75:888–898 — cash-flow LCOE, not an optimisation; legitimate miss; confirms financing-terms block fires.
- Lonergan et al. (2023), *Joule* 7:469–483 — closest prior review (§1); should be returned.
- Bogdanov et al. (2019), *Nat. Commun.* 10:1077 — global 100%-RE optimisation with uniform CoC; global-resolution, likely a documented exclusion under edge-case B, but should be *returned* by the search.

**Documented exclusions — behave-check (may be RETURNED by the search but EXCLUDED at full-text for the pre-registered reasons; the search must not be *designed* to drop them):**
- **Calcaterra et al. (2024)**, *Nature Energy* 9:1241–1251, DOI 10.1038/s41560-024-01606-7 — five-IAM ensemble, country-specific CoC with risk-premium convergence, but **global/macro-region resolution**. EXCLUDED under edge-case B. Cited as background: brackets the gap *from above* (endogenous finance at global scale). Highest-stakes exclusion (clearest endogenous-finance-in-optimising-model example in the literature); MUST appear as a named, reasoned full-text exclusion in the PRISMA flow, never a silent drop.
- **Okorie (2025)**, *Energy Policy* 198:114512, DOI 10.1016/j.enpol.2025.114512 — "Making hay while the sun shines: Energy security pathway for Africa"; **economy-wide dynamic recursive-CGE, continental (Africa) resolution**, exogenous financing. EXCLUDED on two gates (not a power-system planning model; not national). *(An AI-generated summary mischaracterised this as a "Nigeria CGE"; the primary source is Africa-continental — a cautionary instance of why AI summaries are treated as leads, not evidence.)*

*(DOSCOE removed — not confirmed to exist as a named model/lineage; excluded from the validation set per author decision.)*

## 7. Screening and extraction

**Two-stage screening:** (1) title/abstract against eligibility; (2) full-text against eligibility + edge cases A–C. Decision form maps 1:1 to §4.

**Ambiguous / dimension-spanning studies:** the framework boundaries are porous *by design*. A study may legitimately address multiple dimensions; this is captured (multi-dimension coding), not forced into one cell. Deliverability-vs-Access ambiguity is resolved by the §3 split and recorded per study.

**Reproducibility (screener protocol — declared, not hidden):**
- **Primary route (to be exhausted first):** recruit a second independent screener for the full title/abstract set and report **inter-rater Cohen's κ**. Inter-rater agreement is the stronger evidence and is pursued before any fallback; the author will document the effort made to secure a second screener.
- **Fallback (only if a second screener is genuinely unavailable):** a documented **self-re-screen** of a random subset — **20% of the corpus, with a floor of 100 records** — **stratified to over-sample the flagged ambiguous / dimension-spanning cases** (Deliverability-vs-Access, multi-dimension, borderline model-type), after a **14-day (fortnight) washout**, reporting **intra-rater κ honestly whatever its value**. Sole-screener bias is then declared as a primary limitation (§10). A low κ is reported, not suppressed or re-run to a better number.

**Data-extraction template (per study):**
bibliographic details; country/system and geography; **model class (MANDATORY sub-code: {mathematical-programming | geospatial-least-cost | simulation–optimisation hybrid | equilibrium | national-IAM})** — this is the guardrail against scope-drift (§4) and the stratifying axis of the synthesis matrix; temporal & spatial resolution; **energy-security dimensions represented and how** (Availability / Deliverability / Access / Affordability / Acceptability — each: represented Y/N + mechanism); **which Accessibility reading is used** (Deliverability vs Access vs both vs unspecified) + **does the study name the distinction? (Y/N)**; **financing/cost-of-capital treatment ∈ {absent, exogenous discount rate, endogenous financing structure}** + mechanism; uncertainty treatment; stated limitations.

*Worked coding of the three positive controls (illustrative, fixes the coding convention):*
- *Agutu 2022 → {geospatial-least-cost; Access; endogenous (country×mode CoC)}*
- *Oyewo 2018 → {mathematical-programming; Availability+Deliverability+Affordability; exogenous (flat uniform 7% WACC)}*
- *Schinko 2019 → {simulation–optimisation hybrid; Affordability+Availability+Acceptability; endogenous (technology-differentiated WACC, market vs de-risking regimes)}*

All extracted counts, study characteristics, and inter-/intra-rater agreement values are **[TO BE POPULATED FROM SCREENING]**.

## 8. Synthesis (funnel; Stage 4 deliverable)
Once real extracted data are returned: a **synthesis matrix (studies × 5 operationalised dimensions × financing-treatment)** that funnels from broad dimensional coverage to intersections to the specific under-addressed cell (endogenous financing within multi-dimensional security-oriented planning for developing systems). The gap is **evidenced from empty/sparse cells in the matrix, not asserted.** If cells the thesis expects to be empty are populated, that is reported and the gap statement narrows accordingly (open-review commitment, §12).

## 9. Risk-of-bias / quality appraisal
Not an effectiveness review, so no RoB2/ROBINS-I. Instead, a **methodological-transparency appraisal** per included study: are the model, objective, constraints, and (critically) the discount-rate/financing assumptions explicitly stated and reproducible? Studies that leave the discount rate unstated are flagged — relevant to SQ5.

## 10. Anticipated limitations (declared a-priori)
**Database coverage — the absence of Scopus and Web of Science** (not accessible at the author's institution); the review relies on Lens.org, OpenAlex, and IEEE Xplore. These are broad, aggregated, Crossref-backed indexed sources and all three positive controls are Crossref-indexed, so core recall is expected to hold; nevertheless, any indexing gap specific to Scopus/WoS is a genuine limitation, partially mitigated by the validation dry-run (which doubles as a coverage test) and, if legitimately obtainable, a supplementary Scopus/WoS cross-check by a colleague under their licence. Further limitations: sole-screener bias; English-only language coverage; grey-literature non-systematicity; the energy-security-framework selection itself (choosing APERC over Sovacool/IEA is defensible but consequential); SSA cost-of-capital data sparsity (Steffen et al. 2025) limiting how firmly any Nigeria-specific WACC can be sourced downstream.

## 11. Registration and its value to a Q1 submission
This a-priori protocol is **to be registered on OSF Registries** (chosen over PROSPERO, whose scope centres on health-related reviews and does not cleanly fit a pure energy-methods review). **Current status: an OSF account exists but the protocol is NOT YET registered.** Creating an account is not registration; registration is the distinct act of freezing a dated, public, read-only snapshot of this finalised protocol, which then carries a permanent timestamp and DOI. Registration must occur (i) once this finalised version is complete — including the confirmed free-source database stack (§5) — and (ii) BEFORE any search is executed. Registering after searching destroys the timestamp's evidentiary value. The timestamp is the primary defence against the "reverse-engineered gap" objection: it evidences that the eligibility criteria and matrix categories pre-date the results. Registration does not freeze the method against all change — legitimate revisions (e.g. adding a search term after the validation dry-run) are permitted provided they are logged transparently against the registered snapshot (search-revision log, Stage 2). **Cite the OSF DOI/URL in the manuscript methods once registered.**

## 12. Open-review commitment (standing rule)
The synthesis matrix and extraction categories are fixed by this registered protocol *before* screening. The author commits to reporting the financing cell's occupancy honestly even if fuller than the thesis assumes; a fuller cell narrows the contribution from "fills an empty cell" to "extends a thin, fragmented treatment" — still a valid Masters contribution. Declared lineage (positioning the thesis as the review's response) is a *consequence* of running the review open, not a substitute for it.

---

### Status of pre-registration actions
1. **Framework-anchoring citations — VERIFIED against primary source (CLOSED):** Ang/Choong/Ng (2015) 104 studies / 83 definitions / **seven** themes (≠ the four A's; see §3); APERC (2007) four terms + supply-side Accessibility; Sovacool & Mukherjee (2011) five dimensions + investment-as-component; Ondraczek/Komendantova/Patt (2015); Egli et al. 2018 (41%/40% total financing contribution) & 2019 (uniform-CoC inversion); Lonergan et al. 2023 (58-study CoC review); Schmidt et al. 2012.
2. **DOSCOE removed** from validation set (CLOSED, §6.4).
3. **Re-screen protocol set** (CLOSED, §7): second-screener route exhausted first; fallback = 20% / floor-100, ambiguity-stratified, fortnight washout, κ reported honestly.
4. **Three verified positive controls added** (CLOSED, §6.4): Agutu 2022, Oyewo 2018, Schinko 2019 — spanning three model classes.
5. **Two documented exclusions logged** (CLOSED, §6.4): Calcaterra 2024 (edge-case B), Okorie 2025 (two gates).
6. **Inclusive model-type boundary + mandatory model-class sub-code** (CLOSED, §4, §7).
7. **Database stack revised to free sources** (CLOSED, §5): Lens.org + OpenAlex + IEEE Xplore; Scopus/WoS absence declared as a limitation (§10).
8. **Block 4 country list** = focal-sample-first (CLOSED); expand only if validation/known-item recall fails.
9. **OSF registration — NOT YET DONE (OPEN, blocking):** account created; protocol to be registered once this finalised version is complete and BEFORE searching (§11).

### Remaining [TO BE POPULATED] items (require live execution, not decisions)
- OSF timestamp date (header) — populated at registration.
- All database record counts and the PRISMA flow numbers (§5, §8).
- Screening outcomes, κ value(s), study-characteristics table (§7).
- The synthesis matrix, populated from extracted data (§8).

### Immediate next actions, in order
1. **Author:** confirm Lens Academic workspace search + export functions (verify, don't assume).
2. **Author:** REGISTER this finalised protocol on OSF (freeze snapshot) — before any search.
3. **Then:** execute the Stage 2 (rebuilt) strings for Lens.org, OpenAlex, IEEE Xplore; run the validation dry-run against §6.4 (must-return Agutu, Oyewo, Schinko). The dry run is the first real signal on empty-vs-sparse.

### Next deliverable
**Stage 2 (rebuilt)** — field-tagged Boolean strings for **Lens.org, OpenAlex, and IEEE Xplore** (Scopus/WoS versions retired), plus the validation-set dry run against §6.4.
6. Proceed to **Stage 2** (field-tagged Boolean strings + validation-set dry run).
