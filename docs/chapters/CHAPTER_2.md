# Chapter 2 — Literature Review

## 2.1 Introduction

Planning the expansion of a national electricity supply system requires a quantitative method for choosing between generation technologies, fuels and delivery options under constraints of resource availability, cost and policy. Optimisation-based planning models provide that method, and they have been applied widely to Sub-Saharan African systems over the past decade. This chapter examines what those models represent and, more particularly, what they do not.

The chapter is organised around two questions. The first concerns the dimensions of energy security that such models operationalise: whether the literature represents supply adequacy, fuel and network delivery, population reach, cost and environmental acceptability, and whether representation varies with the class of model employed. The second concerns the treatment of capital cost. In capital-intensive power systems the discount rate or weighted average cost of capital applied to candidate investments determines which technologies appear least-cost, and in developing economies that rate is both high and poorly characterised (Ondraczek et al., 2015; Steffen, 2020). How the planning literature obtains and applies that rate is therefore a question about the reliability of its conclusions.

Both questions are answered from a systematic review conducted for this study under a pre-registered protocol. Section 2.2 describes the review method. Sections 2.3 to 2.5 report its findings. Section 2.6 states the limitations that bound those findings, and Section 2.7 identifies the research gaps that the remainder of this thesis addresses.

### 2.1.1 Rationale for a systematic method

A narrative review selects its material by the reviewer’s judgement, and the reader cannot distinguish between the studies that exist and the studies that supported the argument. A systematic review fixes the question, the eligibility criteria, the search and the extraction schema before the evidence is seen, and reports the result of applying them (Page et al., 2021). The procedure is analogous to specifying a core-analysis protocol before the core is cut: the method is committed to while the outcome is still unknown, so the outcome cannot shape the method.

The protocol for this review was registered publicly with a timestamp prior to screening. Four amendments were made subsequently; each is dated and its direction recorded (Section 2.2.6). Reporting follows the PRISMA 2020 statement (Page et al., 2021).

## 2.2 Review Methodology

### 2.2.1 Review questions

The review addressed three questions:

1.  Which energy-security dimensions do optimisation-based power-system planning studies in Sub-Saharan Africa represent, and does coverage vary with model class?
2.  How do those studies treat the cost of capital and financing structure — as what class of model object, and derived by what means?
3.  What remains unaddressed that a national capacity-expansion model could address?

A crosswalk between these and the sub-questions of the registered protocol is given in Appendix B.

### 2.2.2 Eligibility criteria

Studies were eligible where they reported an original, full-length optimisation-based planning study — one in which at least one generation or transmission capacity is a decision variable determined by cost minimisation — applied to an electricity system located wholly or substantially within Sub-Saharan Africa, and published from 2015 onwards. Sub-Saharan Africa was defined by the World Bank regional classification.

Two criteria governed the majority of exclusion decisions.

**Simulation–optimisation boundary.** Tools computing outcomes forward from fixed inputs without solving an optimisation were excluded, as were dispatch-only models containing no capacity or planning decision. This criterion accounted for twenty-nine of sixty-three full-text exclusions.

**Document type.** Eligibility on document type was determined by the research object rather than the publication venue. Records reporting original full-length optimisation research were eligible irrespective of whether they appeared in a journal, in conference proceedings, or as a working paper or preprint. Records were excluded where the research object itself fell outside scope: review and scoping-review articles, commentary and discussion papers, and studies applying or calibrating an existing model without formulating an independent one. Four preprints meeting the research-object criterion were included; the theses in the screened corpus were excluded under the same criterion, each having failed the independent-model requirement.

Full eligibility criteria, including language and date limits, are reproduced in Appendix A.

### 2.2.3 Information sources and search strategy

Searches were executed on 8 July 2026 across Lens.org, OpenAlex and IEEE Xplore. Scopus and Web of Science were not available at the author’s institution; the consequences for coverage are stated in Section 2.6.

The search employed four concept blocks: optimisation method; electricity-system scope; a disjunctive block combining energy-security terminology with financing and cost-of-capital terminology; and a developing-region block. The third block was specified as a disjunction so that a study entering the corpus through its financing vocabulary was not additionally required to use security vocabulary. Platform-specific search strings are reproduced in Appendix A.

Recall was tested before screening against three pre-registered positive controls (Agutu et al., 2022; Oyewo et al., 2018; Schinko et al., 2019). Two were recovered and included; the third was recovered and subsequently excluded on geographic grounds under Amendment 2.

### 2.2.4 Screening procedure

Title and abstract screening employed active-learning prioritisation (ASReview; van de Schoot et al., 2021), in which a classifier reorders the screening queue as the reviewer’s inclusion and exclusion decisions accumulate. A stopping rule was committed before screening commenced: screening would terminate after two hundred consecutive records were assessed without an inclusion. Of 23,717 deduplicated records, 2,521 were individually assessed before the rule triggered. No recall percentage is claimed for this procedure; the residual risk is stated in Section 2.6.

A stratum of records carried no retrievable abstract. Following a documented and unsuccessful recovery attempt, these were screened on title alone under a retention-on-doubt rule, by which a record was excluded only where the title alone established clear ineligibility. No record was excluded on the basis of a missing abstract.

Screening and extraction were conducted by a single reviewer. The implications are stated in Section 2.6.

### 2.2.5 Data extraction and synthesis

A pre-specified extraction schema recorded, for each included study: country and system scope; model class; temporal and spatial resolution; for each of five energy-security dimensions, whether the dimension was operationalised and by what mechanism; the treatment of financing; the derivation of any stated discount rate; and the treatment of uncertainty.

Energy-security dimensions follow the multidimensional framework established in the energy-security literature (APERC, 2007; Sovacool & Mukherjee, 2011), adapted for the representational question posed here. The five dimensions are defined in Section 2.3.2. A dimension was coded as represented only where the study operationalised it within the model — as a constraint, an objective term or a reported model output — and not where it was discussed in surrounding text without entering the formulation.

Synthesis was descriptive and count-based. Cross-tabulations were computed from the frozen extraction workbook; no meta-analytic pooling was performed, the included studies reporting heterogeneous outputs on non-commensurable scales.

### 2.2.6 Protocol registration and amendments

  ---------------------------------------------------------------------------------------------------------------------------------
  **No.**   **Amendment**                                                                                           **Direction**
  --------- ------------------------------------------------------------------------------------------------------- ---------------
  1         Geographic eligibility narrowed to Sub-Saharan Africa                                                   Restrictive

  2         Eligibility adjudication of a positive control (Schinko et al., 2019), excluded on geographic grounds   Restrictive

  3         Sub-question set disaggregated; a derivation sub-question added                                         Expansive

  4         Endogenous/exogenous boundary tightened; a validation control re-coded                                  Restrictive
  ---------------------------------------------------------------------------------------------------------------------------------

Amendments 1, 2 and 4 narrowed the review’s scope or tightened a classification and therefore operate against, rather than in favour of, the review’s prior expectation. Amendment 4 is discussed in Section 2.5.1, where its effect on the principal finding is stated.

### 2.2.7 Study selection

  -------------------------------------------------------------------------
  **Stage**                                                      **n**
  -------------------------------------------------------------- ----------
  Records identified, database searching (after deduplication)   23,717

  Records screened at title and abstract before stopping rule    2,521

  Reports sought for retrieval — database searching              175

  Reports sought for retrieval — citation searching              8

  Total reports sought                                           183

  Reports not retrieved                                          45

  Reports assessed for eligibility                               138

  Reports excluded at full text, with reasons                    63

  **Studies included in the review**                             **75**
  -------------------------------------------------------------------------

Eight studies entered through backward citation searching of the included set and are reported as a separate identification arm, in accordance with PRISMA 2020 (Page et al., 2021). Four duplicate records surviving automated deduplication were identified and removed at the retrieval stage. The flow diagram is given in Appendix A, and full-text exclusions are tabulated there by reason.

## 2.3 Characteristics of the Included Studies

### 2.3.1 Distribution by model class

  -------------------------------------------------------------------------
  **Model class**                                 **Studies**   **Share**
  ----------------------------------------------- ------------- -----------
  Mathematical programming (capacity expansion)   43            57.3%

  Geospatial least-cost electrification           24            32.0%

  Partial equilibrium                             5             6.7%

  Simulation–optimisation hybrid                  2             2.7%

  National integrated assessment                  1             1.3%

  **Total**                                       **75**        **100%**
  -------------------------------------------------------------------------

Two classes account for sixty-seven of the seventy-five studies. Mathematical programming models optimise capacity and dispatch at national or multi-national scale, typically with a single or small number of nodes, an aggregate demand series and a constraint set carrying policy targets. Geospatial least-cost models optimise supply-mode selection across settlements or grid cells, with the unit of analysis disaggregated to the point of demand. The two classes address different planning problems and, as Section 2.4.2 establishes, they represent different dimensions of energy security.

### 2.3.2 Energy-security dimensions

The five dimensions used in the synthesis are defined as follows.

**Availability** — whether sufficient generation capacity and primary energy exist to meet demand. Operationalised typically as a capacity-adequacy or energy-balance condition.

**Deliverability** — whether energy can be conveyed from the point of production to the point of demand. In network-constrained systems this is a transmission representation; in gas-dependent systems it is a fuel-supply or gas-deliverability constraint, limiting the quantity of fuel that can physically be delivered to generating plant within a period.

**Access** — whether a given population is connected to supply. Operationalised as connection status, population served, or a supply-mode selection outcome.

**Affordability** — the cost of supply to the system or to the consumer. Operationalised as total system cost, levelised cost of electricity, or tariff. This is the dimension through which the cost of capital enters a model, where it enters at all.

**Acceptability** — the environmental acceptability of the supply mix. Operationalised almost exclusively as an emissions constraint, a renewable-share target or a decarbonisation scenario.

The distinction between availability and deliverability is of particular consequence in gas-dependent systems, where installed capacity may substantially exceed the capacity that delivered fuel can support. A model representing availability alone will report an adequate system where a deliverability constraint would report a shortfall.

## 2.4 Dimensional Coverage

### 2.4.1 Aggregate coverage

  -----------------------------------------------
  Dimension        Studies representing   Share
  ---------------- ---------------------- -------
  Availability     75                     100%

  Affordability    75                     100%

  Deliverability   59                     78.7%

  Acceptability    50                     66.7%

  Access           44                     58.7%
  -----------------------------------------------

Every included study represents availability and affordability. Coverage of deliverability, acceptability and access is partial. Read as an aggregate, the distribution suggests uneven attention across the field.

### 2.4.2 Coverage by model class

The aggregate distribution conceals a systematic pattern. Disaggregated by model class, coverage is partitioned rather than thin.

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Dimension**    **Mathematical programming (43)**   **Geospatial (24)**   **Partial equilibrium (5)**   **Simulation hybrid (2)**   **National IAM (1)**   **Total**
  ---------------- ----------------------------------- --------------------- ----------------------------- --------------------------- ---------------------- -----------
  Availability     43                                  24                    5                             2                           1                      75

  Deliverability   28                                  24                    4                             2                           1                      59

  Access           14                                  24                    4                             1                           1                      44

  Affordability    43                                  24                    5                             2                           1                      75

  Acceptability    37                                  7                     5                             1                           0                      50
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

The two dominant classes exhibit complementary omissions. Geospatial least-cost models represent availability, deliverability, access and affordability in all twenty-four cases, and acceptability in seven. Mathematical programming models represent acceptability in thirty-seven of forty-three cases and access in fourteen of forty-three.

The pattern follows from what each formulation renders representable. Access is a disaggregated question predicated on a settlement or a household; a geospatial model’s unit of analysis is precisely that object, and supply-mode selection for it is the model’s central decision. A single-node national capacity-expansion model contains no such object: its decision variables are capacities and flows at national scale and its demand is an aggregate, so the model can represent how much energy is supplied but not to whom. Acceptability runs in the opposite direction. Operationalised as an emissions cap or renewable-share target, it attaches naturally to a cost-minimisation programme with a constraint set, whereas a least-cost electrification rollout optimising connection cost across settlements carries no national emissions target in seventeen of twenty-four cases.

Deliverability is the one dimension represented by a majority of both classes — twenty-eight of forty-three mathematical programming studies and all twenty-four geospatial studies — although the mechanism differs. In geospatial models it is predominantly a network or distance representation; in national capacity-expansion models of gas-dependent systems it is predominantly a fuel-supply constraint. The extraction recorded these mechanisms as free text and did not code them to categories, so the sub-classification is not enumerated here.

The consequence for how a gap in this literature is stated is material. The field as a whole appears better covered than any study within it. A reader seeking a national decarbonisation pathway finds acceptability well handled and access largely absent; a reader seeking electrification planning finds the reverse. The choice of model class, ordinarily made on grounds of data availability or tool familiarity, determines which of the two a study can address.

### 2.4.3 Representation of deliverability and access

Deliverability and access are conceptually distinct. Deliverability concerns the conveyance of energy from production to demand — a question of networks, fuel supply and physical delivery. Access concerns whether a given population is connected. A system may be highly deliverable and barely accessible, and the interventions indicated differ accordingly.

  ----------------------------------------------
  **Dimensions represented**       **Studies**
  -------------------------------- -------------
  Both deliverability and access   39

  Deliverability only              20

  Access only                      5

  Neither                          11

  **Total**                        **75**
  ----------------------------------------------

Of the thirty-nine studies operationalising both dimensions, one articulates the distinction between them as a conceptual matter. The remaining thirty-eight construct mechanisms that separately represent delivery and reach without identifying the two as distinct security concerns.

The consequence is that the distinction cannot be reasoned about within those studies. A study that does not name what it has separated cannot discuss the trade-off between the two, report which of them its results are sensitive to, or be compared against another study on that axis without the comparison being reconstructed from the model structure by the reader.

## 2.5 Treatment of Financing and the Cost of Capital

### 2.5.1 Classification of financing treatment

In a capital-intensive power system the rate at which capital expenditure is annualised determines the relative standing of capital-intensive and fuel-intensive technologies in a least-cost solution. The review classified each included study by how that rate, and the financing structure behind it, enters the model.

A treatment is exogenous where a discount rate or cost-of-capital value is supplied to the model as an input, irrespective of how the value was obtained or how far it is differentiated by country, technology or delivery mode. A treatment is endogenous where financing structure or the cost of capital is determined within the model itself — as a decision variable, as a solution outcome, or as a constraint interacting with the capacity solution. A treatment is absent where no financing parameter is stated.

  -----------------------------------------------------------------------
  **Treatment**                    **Studies**           **Share**
  -------------------------------- --------------------- ----------------
  Exogenous                        65                    86.7%

  Absent                           10                    13.3%

  Endogenous                       0                     0%
  -----------------------------------------------------------------------

In sixty-five of seventy-five studies the cost of capital is supplied as an input. In ten, no financing parameter is stated. In none is financing structure or the cost of capital determined within the model.

This classification is narrower than that registered in the protocol, and the narrowing constitutes Amendment 4. The registered protocol recorded Agutu et al. (2022) as an endogenous treatment on the strength of its country- and mode-differentiated costs of capital. Full-text assessment established that those rates are empirically derived inputs supplied to the optimisation: finely differentiated, carefully sourced, and not determined by the model. The classification was accordingly tightened and applied uniformly to all seventy-five studies.

The effect of the amendment on the principal finding is stated here rather than left to be inferred. Under the registered classification the gap probe would have returned one study, and the pre-specified interpretation rule would have required the finding to be reported as a sparse cell rather than an empty one. Under the tightened classification it returns none. The finding is therefore a product of a stated criterion applied uniformly, and the alternative result under the superseded criterion is on record.

A second deviation is declared alongside it. The registered gap probe was a three-way intersection of endogenous financing, supply-security representation and national capacity-expansion scope. The probe as executed tests the two-way intersection only, which broadens the conditioning set from a subset of the corpus to all seventy-five studies and therefore tests the proposition against a larger body of evidence.

### 2.5.2 Derivation of exogenous rates

Within the sixty-five exogenous treatments, the review recorded the basis on which the supplied value was obtained. Rungs are ordered by the externality of the justification offered:

- **not stated** — no derivation given;
- **undefended-flat** — a value with no stated source or argument;
- **argued** — a value defended by reasoning internal to the study;
- **precedent-cited** — a value taken from a named prior study or national plan;
- **institution-sourced** — a value anchored to a named institutional figure;
- **market-benchmarked** — a value derived from observed market data.

A second and orthogonal attribute records whether the study varies the rate across a range rather than fixing it at a point.

  -------------------------------------------------------------------------
  **Derivation**             **Fixed**        **Swept**      **Total**
  -------------------------- ---------------- -------------- --------------
  Not stated                 1                0              1

  Undefended-flat            32               8              40

  Argued                     7                8              15

  Precedent-cited            4                2              6

  Institution-sourced        1                2              3

  Market-benchmarked         0                0              0

  **Total**                  **45**           **20**         **65**
  -------------------------------------------------------------------------

Forty-one of sixty-five studies — 63.1 per cent — apply a cost-of-capital value for which no source or argument is given. Nine studies, 13.8 per cent, reach the precedent-cited rung or above. Three anchor the value to a named institutional figure, such as a national central bank rate. No study in the corpus derives its cost of capital from observed market data.

Reported point rates range from 5 to 12 per cent, with a modal value of 10 per cent. Twenty of the sixty-five vary the rate across a range, and eight of those twenty apply a value at the undefended-flat rung — testing sensitivity to a parameter whose magnitude is not defended. One study varies the rate from 5 to 15 per cent, a range wider than the full spread of point values reported elsewhere in the corpus, and reports that the least-cost technology mix reorganises across it.

### 2.5.3 Differentiation of the cost of capital

Eight of the sixty-five exogenous treatments differentiate the rate: four by technology, three jointly by country and delivery mode, and one by country.

  -------------------------------------------------------------------------------
  **Study**             **Derivation**        **Swept**     **Differentiation**
  --------------------- --------------------- ------------- ---------------------
  Agutu et al. (2022)   precedent-cited       Yes           Country × mode

  Akbas et al.          precedent-cited       No            Country

  Dalder et al.         argued                Yes           Technology

  Egli et al.           institution-sourced   Yes           Country × mode

  Moksnes et al.        institution-sourced   Yes           Country × mode

  Rocco et al.          argued                Yes           Technology

  Sahlberg et al.       precedent-cited       No            Technology

  Yazdanie et al.       undefended-flat       No            Technology
  -------------------------------------------------------------------------------

Seven of the eight occupy the argued rung or above. Of the forty studies at the undefended-flat rung, one differentiates the rate. Studies that identify the provenance of a financing parameter are substantially the same studies that permit it to vary.

The remaining fifty-seven studies are recorded as not differentiating the rate. The extraction recorded differentiation where present and did not separately verify uniformity where absent; these fifty-seven are accordingly *not recorded as differentiated* rather than verified uniform. A supplementary examination of the free-text mechanism records for those fifty-seven identified no unrecorded differentiation.

### 2.5.4 Interpretation

The empirical literature establishes that the cost of capital for energy infrastructure in developing economies differs materially by country, by technology and by delivery mode, and that the difference is large enough to reorder a least-cost ranking (Ondraczek et al., 2015; Egli et al., 2019; Steffen, 2020). For Sub-Saharan African systems the cost of capital is among the parameters most likely to differ from the contexts in which the modelling tools were developed.

The reviewed literature nonetheless supplies this quantity as a fixed input, and in 63.1 per cent of cases without a stated source. No study determines it, or the financing structure behind it, within the model. The field therefore treats as a parameter requiring no defence a quantity that is outcome-determining, empirically variable, and — on the evidence of the one study that varies it widely — capable of reorganising the technology mix a planning model recommends.

## 2.6 Limitations of the Review

The findings reported above are bounded by the following.

**Retrieval.** Of 183 reports sought at full text, 45 were not retrieved, a retrieval failure of 24.6 per cent. Retrievability correlates with open-access status and journal prestige, so the unretrieved set is not a random sample of the sought set and plausibly over-represents national, non-open-access and grey-literature studies. This is the principal threat to the finding reported in Section 2.5.1, since a counterexample would most plausibly reside there.

**Screening coverage.** 2,521 of 23,717 deduplicated records were individually assessed before the pre-committed stopping rule triggered. Records ranked below the threshold were deprioritised rather than examined and rejected. Recall rests on the assumption that the active-learning ranking surfaced relevant material before the stop, supported by recovery of both retained positive controls within the screened set. No recall percentage is claimed.

**Abstract-less records.** Approximately 2,990 records — 12.6 per cent of the deduplicated corpus — carried no usable abstract and were screened on title alone under a retention-on-doubt rule. Title-only screening is less sensitive than title-and-abstract screening. Being more inclusive, it cannot have narrowed the corpus, but it will have passed records forward that a fuller screen would have resolved earlier.

**Single-reviewer screening.** All screening and extraction was conducted by one reviewer. The registered protocol specified dual screening with Cohen’s kappa (Cohen, 1960) and, failing that, a self-re-screen of a stratified sample following a washout interval with intra-rater kappa reported; an independent second-screener validation of a sample of excluded records was additionally registered by amendment. Neither has been executed at the time of writing. Consistency was supported instead by a screening rulebook with worked edge cases, pre-committed and re-read at the commencement of each screening session, and by a structured second-opinion procedure at extraction.

**Source coverage.** Scopus and Web of Science were not searched. The three sources used are Crossref-indexed and Boolean-capable, and both retained positive controls were recovered, but coverage of conference proceedings and regionally-published work is likely weaker than a subscription stack would provide.

**Scope.** All findings are bounded to Sub-Saharan Africa and to optimisation-based planning studies. No claim is made concerning the global literature. The result reported in Section 2.5.1 is a statement about a retrieved and assessed corpus.

## 2.7 Research Gaps and Implications for This Study

### 2.7.1 Gaps identified

Three gaps follow from the findings above.

**Gap 1 — the cost of capital is supplied, not determined.** No study in the corpus determines financing structure or the cost of capital within its model, and 63.1 per cent supply a value without a stated source (Sections 2.5.1, 2.5.2). A planning model whose capital charge is a fixed, undefended scalar cannot represent the scarcity of concessional capital, the division of capital between sources at different prices, or the revenue conditions under which privately financed capacity can be raised at all. In a system whose expansion options differ sharply in capital intensity — gas plant against solar and storage — this omission bears directly on which pathway a model recommends.

**Gap 2 — access and acceptability are partitioned by model class.** Mathematical programming capacity-expansion models represent access in fourteen of forty-three cases; geospatial models represent acceptability in seven of twenty-four (Section 2.4.2). A national decarbonisation study conducted in the first class inherits an inability to represent whether supply reaches a given population, and conclusions drawn from it are bounded accordingly.

**Gap 3 — the boundary of an emissions constraint is not examined.** The reviewed studies operationalise acceptability predominantly as an emissions cap or renewable-share target applied to the modelled system. Where a national commitment covers a sub-sector of the national inventory, and where supply failure induces substitution to emission sources outside the modelled boundary, the relationship between a tightening constraint and the national quantity it is intended to govern is not established by the reviewed literature.

### 2.7.2 Implications for this study

The model developed in Chapter 3 responds to these gaps as follows.

In respect of Gap 1, capital expenditure is represented not by a single annualising rate but by an explicit division between concessional and commercial capital drawn from a bounded concessional resource, with a revenue-adequacy condition governing privately financed capacity. The formulation, its decision variables and its constraints are set out in Section 3.4, and the behaviour of the resulting capital charge is reported in Chapter 4.

In respect of Gap 3, the emissions constraint is applied to the grid-supplied quantity, consistent with the sub-sectoral basis of the national commitment, while self-generation outside that boundary remains unbounded. The derivation of the constraint and its boundary is given in Section 3.5.

In respect of Gap 2, this study does not close the gap it identifies. The model is single-node and contains no locational variable, no zonal balance and no connection decision; demand enters as a national aggregate. The partition identified in Section 2.4.2 is therefore inherited by this study rather than narrowed by it. The treatment of self-generation described above represents a failure of delivery to connected demand, which is a deliverability phenomenon and not a representation of access; the distinction reported in Section 2.4.3 is maintained throughout.

Further limitations of the model, each with its direction of bias, are set out in Section 3.7. The principal ones are the single-node spatial resolution, the absence of integer structure, perfect foresight across the planning horizon, price-inelastic demand, and the exclusion of cross-border trade.
