# Chapter 4 audit, pass 3

Read-only. No chapter prose, no code change, no solve, no register edit. HEAD `8dfd2a3` at the start (the report adds one file). Every figure carries its file. Judgements are reported as evidence plus what it supports; none is decided.

## What this pass found that changes the picture

1. **Question 3's evidence base is quoted on the wrong window.** F-21, F-31, F-39, F-40 and F-43 all quote emissions summed over 2025 to **2054**, the buffer included. Chapter 1 §1.6.3 says emissions are reported to 2045 and that "the final nine years constitute an end-effect buffer". For F-21 the stored per-year diagnostics let me recompute the pair on the reporting window: **cumulative national emissions to 2045 are 271.09 Mt (flat hold) against 268.86 Mt (decline), the decline arm 2.23 Mt (0.8%) lower; the inversion F-21 reports (+20.3 Mt, +5.1%) appears only after about 2050, in the buffer.** For F-39, F-40 and F-43 the per-year emissions are not stored, so the 2045-window turning point cannot be located without re-solves (item B.2, B.4).
2. **The leakage diagnostic is absent behind nine runners, not three** (L51's own TASK 32 update, re-verified at HEAD), which uncovers F-30 (NDC legs), F-34, F-35, F-37, F-38 and F-12/F-13. Pass 2 listed only F-12 and F-13 (item B.5).
3. **Eleven LIVE entries are stale against the repository**, most because a later chapter edit, annotation or finding re-derivation has settled what they ask for (Part A.2). Among them, L92: the VSS legs' comparator problem has dissolved post-epoch (item C.3).
4. **The capped ranking is cheap but not buildable from what is stored:** zero existing capped arms are on the uncapped ranking's capital-case basis (Part D).

---

# PART A: Register triage, the 80 LIVE entries

I read the 80 entries whose status line begins "LIVE" (`docs/limitations_register.md`, 125 entries; 26 RESOLVED, 4 FIXED, 15 other). L94 and L103 are not among the 80 (their status lines do not begin LIVE); L94 was found stale in pass 2. For entries longer than about 2,300 characters I read the opening and the closing paragraphs; the truncated middle of eleven long entries (L8, L21, L22, L23, L24, L51, L52, L68, L82, L114 and L125) was not read line by line. Class counts: **REPORT 43, CONTAINED 22, STALE 11, EPOCH 4** (the table's `Class` column is one class per row; where an entry is stale in part and also needs a class for the rest, the class is STALE and the remainder is stated in the basis).

## A.1 The table

| L | One-line statement | RQ | Class | Basis |
|---|---|---|---|---|
| L1 | Demand shape is a 2016 vintage; true shape likely more evening-peaked, so firm-capacity cost is understated | 1,4 | CONTAINED | Ch3 §3.7.4 'The peak shape has a single source'; `peak_high` arm; Q1 row 8 |
| L2 | The suppressed series is a supply shape; `peak_suppressed` is a declared floor, not a central case | 1 | CONTAINED | Ch3 §3.2 (8 mentions of suppression); arm declared |
| L3 | Firm capacity sized on block averages; `within_block_peaking_factor` 1.3448 applied; LOLE equivalence asserted | 4 | CONTAINED | Ch3 §3.2.5, §3.7.3; Ch1 §1.4 'optimistic margin' |
| L4 | NIRP's ~27 TWh Tier-2 self-generation is stipulated (±40%); the reconciliation memo has a 10x unit error | none | REPORT | Memo path `docs/tier2_reconciliation.md` does not exist at HEAD, so the memo defects cannot be checked; no Ch1/Ch3 text uses the figure |
| L6 | The EaaS channel is solar-only; `storage_add` is public-budget-charged in every regime | 2 | REPORT | Not stated in Ch1 or Ch3 §3.4.6 (which scopes the channel to solar capital); Ch4 must say storage is publicly financed in both regimes (F-38 'same storage') |
| L8 | Gas retirement schedule is an analyst estimate (commissioning years not traced) | 1 | CONTAINED | Ch3 (6 mentions); Q1 row 3 declares it; magnitudes in the entry are pre-epoch stamps |
| L9 | `reserve_margin` and `genset_case` are canonical constants, not in any output tag | none | REPORT | Reproducibility only; no wrong number (entry's own statement) |
| L11 | `storage_binding` diagnostic is not meaningful under the sliced builder | none | REPORT | Do not report the field; L41 shows a script that reads it |
| L12 | The annual baseline has no recorded reference objective; bare path is stale | none | REPORT | Never read `results/baseline/summary.json` as current |
| L13 | Gas deliverability arms span two incomparable families (drift, GMP) | 1 | CONTAINED | Q1 rows 5 and 7 split; `LEVER_FAMILIES` declares both; Ch3 (7 mentions) |
| L14 | `_rebuild_years` is duplicated across the F1 branch (five copies) | none | REPORT | Code-only; copies verbatim identical |
| L16 | EaaS and public solar are physically identical; no siting, losses or distributed premium | 2 | CONTAINED | Ch1 §1.6.7 (siting not identified); scope boundary stated in the entry itself |
| L19 | `26_` sets two required keys internally (no `--dist5-case`, `--solar-cf-case`) | 2 | REPORT | F-12/F-13 source; reproducibility only; internal values match central |
| L21 | `storage_duration_hours` = 4 is ASSUMED and unswept | 1,4 | CONTAINED | Ch3 (21 mentions of duration); Q1 row 4 declares |
| L22 | O&M is SOURCED (ATB) but US rates are unscaled for a Nigerian fleet; storage conversion inherits L21; entry's 3.6%-of-27.954 bn band is pre-epoch | 1 | REPORT | Ch3 declares capital bases are not Nigerian but not O&M; band figures predate the epoch (objective now 24.987 bn) |
| L23 | Two capital tranches shipped; the designed guaranteed-commercial tranche is absent | 2 | CONTAINED | Ch3 §3.4.2 defines the two-tranche formulation and claims no third; no 'guarantee/PRG' claim in Ch1/Ch3 |
| L24 | Post-2035 cap shape is a second analyst construction (flat hold against decline) | 3 | CONTAINED | Ch3 cites it; F-21 compares the two arms as a declared band |
| L26 | Solar CF is a national average (0.19, band 0.16-0.20); `cf_low` is cheaper than central (unexplained, pre-epoch figures) | 1 | CONTAINED | Ch3 (CF 0.19 stated); the cheaper-`cf_low` observation (20.086 vs 20.856 bn) is unverified post-epoch |
| L31 | `peak_adequacy_by_year` uses a flat 2.5 multiplier, arm-invariant, unlike the constraint | 4 | REPORT | Zero consumers in findings, Ch1, Ch3, Ch4 drafts (only Appendix B restates it); do not cite. Same defect as L103's open half |
| L32 | Three dead modules and six dead functions (`postprocess`, `gas_balance`, `stochastic`) | none | REPORT | Documentation only |
| L33 | `04_` still computes the retired concessional mechanism; two columns null; stale `results/fin2` CSVs | none | REPORT | `results/fin2` cited by 0 findings (grep); TASK 57 records `--deployable-fraction` RETIRED |
| L37 | `05_` writes a dead key and two CSV columns on different bases; entry says no finding cites FIN-3 | 2,3 | STALE | F-31's Source line is `05_run_fin3_capital_sweep.py` at HEAD, so the claim 'no number depends on it' is false; the column-basis hazard (`budget_usd` vs `budget_b_usd`) now touches F-31 |
| L38 | `rpublic_diagnostic.py` hardcodes solar life 25 (live value 30) | none | REPORT | Standalone console tool; no persisted output |
| L39 | `14_` 'RE-ENABLE WHEN' precondition already met | none | REPORT | Script text only |
| L40 | `21_` safety check is vacuous; untimestamped figure in a warning | none | REPORT | Script reads legacy names (L112); nothing published |
| L41 | `plot_storage_regime.py` cannot run and has a stale vocabulary | none | REPORT | Script only; contradicts L11's 'read by nothing' |
| L42 | `12_` detailed docstring is orphaned | none | REPORT | Script suspended |
| L44 | Cross-runner key mismatch yields a NaN in `master_results.csv` | none | REPORT | Not cited by findings |
| L47 | Solar capex is a US basis (ATB), at least 1.74x the African figure | 1,2 | CONTAINED | Ch3 §3.7.5 'Capital cost bases are not Nigerian'; cited by id in Ch3; `solar_irena_africa` arm exists |
| L48 | GAS-1 (F-30): five declared limitations (lower-bound framing, benchmark provenance, hardcoded `moderate` capital case); entry also says the binding-years denominator is hardcoded `/21` | 1 | STALE | The `/21` defect: `scripts/06_run_gas1_shadow_benchmarks.py` header records it fixed in TASK 20 Part B (evaluable years used); the other limitations stand. F-30 NDC legs carry no leakage flag (L51) |
| L49 | FIN-3's printed 'sufficiency' verdict keys on unserved energy and must not be cited | 2,3 | REPORT | Duals are the reportable quantity; bias runs against the financing channel |
| L50 | `within_block_peaking_factor` is a builder default 1.3448; the comment in both builders says 1.0 | 4 | REPORT | Contradiction persists at HEAD: `src/optimize_model_sliced.py:1324-1330`; Ch3 cites L50 |
| L51 | The leakage diagnostic is not applied to nine capped runners (24, 25, 26 directly; 04, 06, 07, 08, 11, 13 through shared functions) | 3 (and 1,2) | EPOCH | `src/optimize_experiments.py` has 0 references to leakage at HEAD; affects F-30 NDC legs, F-34, F-35, F-37, F-38, F-12, F-13; fixing it means calling the diagnostic in the shared functions and re-running |
| L52 | Objective's commercial rate (0.11) and the bankability blend (0.18) price the same quantity; both may sit below the CPI cost-of-debt floor | 2 | CONTAINED | Ch3 cites L52 (5 mentions); `bank_tenor_case` aligned the two; problem (2) untouched |
| L55 | Two generation series coexist; data pipeline pulls the non-canonical OWID one | none | REPORT | Ember canonical for the demand derivation (Ch3 §3.2) |
| L56 | `demand_growth_prior()` SIGMA derived from a superseded arm set | none | REPORT | No live consumer (entry's own check) |
| L61 | `00_build_emissions_cap.py` reads a hand-placed bare `results/baseline/diagnostics.json` | 3 | EPOCH | Cap series is derived from the model's own baseline gas-to-power split; `emissions_cap.csv` and the bare diagnostics share mtime 2026-10-19 23:36 (in sync today), so any change to the baseline (L123, L125) requires rebuilding the caps and every capped run |
| L62 | `run_central_sweep.sh` anchor check is a hand-maintained literal | none | REPORT | Defect manifest at HEAD: literal 21.328 bn at `scripts/run_central_sweep.sh:214,370` against the current anchor 24.987 bn |
| L67 | VoLL has no source (5 / 10 / 30 USD per kWh) | 4, cross | CONTAINED | Ch3 §3.7.5 'Value of lost load'; with a priced genset the unserved term is fixed at zero (L49) |
| L68 | `gas_avail_central` unsourced; availability bracket has two endpoints and no sourced centre | 1 | CONTAINED | Ch3 §3.7.5 'Gas fleet availability'; Q1 row 1 must state the arm (L71) |
| L69 | A manifest's `Tag SHA` line is caller-supplied, not verified | none | REPORT | Procedural |
| L70 | A binding-year mean is a composition artefact (can fall as every year worsens) | 1,4 | REPORT | Reporting rule: Ch4 plots maximum and median for binding-conditioned duals |
| L71 | Feasibility and frontier results are invariant to the availability arm; cost and dual results are not | 1,4 | REPORT | Report the arm with each cost or dual figure (entry's mitigation) |
| L72 | Threshold grids stopped short three times; the Pareto frontier (item 20) 'stops at 0.10, tight end untested' | 3,4 | STALE | `results/pareto_frontier/arms_92390a63/pareto_raw.csv` has six tight-end points to 0.01, all optimal (128/128), and the anchor is located (504.1 Mt). The REL floor-bound flagging remains |
| L73 | The genset arm is a segment switch (C&I diesel against 3 kVA petrol), not a sensitivity range | 3 | REPORT | Ch4 quotes `genset_central` and attaches the segment reading; Ch3 §3.7.5 'Self-generation cost and emission factors' |
| L74 | Pareto anchors and tight-end extension have no pre-correction counterpart | 3 | REPORT | Quote only as corrected values |
| L76 | `environment.yml` lists unused dependencies; Sobol indices come from SciPy, not SALib | none | CONTAINED | Ch3 §3.6.7 cites SciPy (Virtanen et al., 2020), per the entry's own mitigation |
| L78 | Handover's 'NDC cap 15.8-28.5 bn' EaaS value has no source | 2 | STALE | `docs/HANDOVER_chapter_4.md:177` carries 'Withdrawn (register L78); do not quote it' (TASK 48 B.6); Table 4.4 rebuilt from `results/dem2/eaas_value.csv` |
| L79 | GAS-2 (F-34) was cleared for the margin correction by reading code; stored files carry wrong labels | 1,2 | STALE | F-34's TASK 55 block is a post-epoch re-run of `results/gas2/arms_be9472ef/` (40 solves); the stale-label files are the pre-epoch ones |
| L80 | F-10 rests on the pre-B* archive; three stored sets give three NDC EaaS values (not diagnosed) | 2 | REPORT | Do not cite F-10. The entry's own TASK 39 update diagnoses the three sets as two designs (capped against uncapped solar), so 'never diagnosed' (pass 1, Part 2.2) is itself stale |
| L81 | GAS-2 and DEM-2 differ in solar design; F-34's shock_recovery sign is a cap effect | 1,2 | REPORT | Quote F-34 with its configuration; never beside DEM-2 values; also explains the skeleton's 'unexplained -10.6%' (chapter_4_skeleton.md:206) |
| L82 | Handover §4.4 states F-12 more narrowly than F-12 does | 2 | STALE | `docs/HANDOVER_chapter_4.md:130` carries 'Corrected 2026-10-10 (TASK 43; L82)'; F-12 governs |
| L83 | B* overstates the traditional regime's measured requirement by ~5%; requirement against reference scale undecided | cross, 2 | CONTAINED | Ch1 §1.6.5 declares tier names as grid labels (option b); option (a) is an epoch item and a DECISION before the epoch |
| L84 | 32 sweep-aggregate result files admitted CURRENT on path-plus-log evidence (rule C2) | none | REPORT | `docs/results_inventory.csv` dates from 2026-10-18 and was not rebuilt after the epoch (TASK 59: `build_results_inventory.py` not run); the generator no longer reads it |
| L87 | Results are produced faster than they are recorded; staleness checker in place | none | REPORT | Procedural; regenerate inventory, manifest and ranking after the epoch |
| L89 | Concessional recovery windows truncate from the 2031 solar and 2041 storage vintage, inside the reporting window | 2 | CONTAINED | Ch3 §3.4.2 (`CHAPTER_3_assembled.md:1782-1788`) now states exactly this; the 'state it in §3.4.2' resolution is done. Quantification (untruncated solve) still absent |
| L91 | Planning reserve margin = 900 MW / peak, falling 19.5% to 6.4%; entry says §3.7.4 omits it | 4 | STALE | Ch3 §3.7.4 now has 'The reserve margin is an operating quantity used as a planning standard' (`CHAPTER_3_assembled.md:3244-3251`), with the 19.5 to 6.4 fall; the entry's '§3.7.4 names three ... not this one' is out of date |
| L92 | VSS legs run at `moderate` while the anchor runs `unconstrained`; '47% more' (mean 40.20, median 29.85) mixes configurations | 4 | STALE | F-42 TASK 55 block: post-epoch the frozen first-stage plan equals the unconstrained anchor (base = anchor + 6.7 USD); mean 37.846 bn, median 27.502 bn; both comparators now coincide (item C.3) |
| L93 | The five VSS guards compare `case_fields` but not the legs' standard arms | 4 | CONTAINED | Ch3 cites L93; arms agree today by construction |
| L95 | Hydro carries capacity and no O&M cost; entry says no document declares it | 1 | CONTAINED | Ch3 §3.7.3 'Hydro carries no operating cost' (cites L95) and §3.3.5; the entry's 'no document declares this' is stale. Implementing a term is a deferred structural change |
| L96 | `selfgen_busbar_factor` is a builder default 1.0 (7% transmission loss not credited) | 3 | REPORT | Set explicitly only in `24_` (`BUSBAR_ARM`); Ch3 mentions busbar as the metering point; bias against the self-generation backstop; deferred structural change |
| L97 | Four scenario keys no one reads; one key live in one builder only | none | REPORT | Disagrees with invariant G7-3 on `blend_case` (item A.4) |
| L100 | Derived block PAR differs from the peak factor by 3e-5, not 'exactly' | none | STALE | Ch3 §3.2.5 (`CHAPTER_3_assembled.md:990-995`) now states 1.087668 against 1.0877, 'the residual arising from rounding' |
| L102 | Storage seasonal balance has no state of charge | 1,4 | CONTAINED | Ch3 §3.7.3 'Storage is an energy balance, not a state of charge'; deferred structural change |
| L105 | The salvage sensitivity cannot isolate salvage; no results; entry says §3.4.2 claims truncation is a salvage value | none | STALE | Ch3 §3.4.2 (`:1790-1800`) now states truncation holds 'in cost and not in the plan' using the eight horizon cells; the script and its placeholder file are unchanged |
| L108 | `concessional_shadow_usd_per_usd` is None in every `f1_tranche` result | 2 | REPORT | Pass 2 item 1: the Q2 cells use `m.public_budget_constraint`'s dual, not this field; no reported result depends on it |
| L110 | Own flags that are not arms are not in the output key of fixed-name runners | none | REPORT | Overwrite hazard (17_ `--draws-file`, `--n-draws`) |
| L111 | Unlabelled aggregate names remain in 02/03/27/29 and `summarize_run` | 3 | REPORT | `02_` writes `actual_emissions_tco2_total`, the 2025-2054 cumulative: the field behind F-21's totals (item B.2) |
| L112 | Several analysis scripts read result files no current runner writes | none | REPORT | Scripts fail loudly |
| L113 | Pre- and post-epoch result files have different schemas | cross | REPORT | Property of the epoch; central hash `6046f759` (was `02c17f55`) |
| L114 | `09_` writes in place and carries no arm tag | 4 | REPORT | F-33 re-derived by path and mtime only; counterpart unmatched by tag |
| L115 | Chapter 3 reference markers the repo cannot close | none | REPORT | TASK 56 worklist |
| L116 | DARES 750M IDA is a grant-and-TA programme; the model treats it as a debt stock | 2 | CONTAINED | Ch1 §1.5.2 declares both; Ch3 §3.4.3 |
| L117 | No source for the 'multilateral finance excludes unabated fossil' convention | 2 | CONTAINED | Ch1 §1.5.2 (`Chapter1.md:340-346`); Ch3 §3.4.3 |
| L118 | The Electricity Act is held only as the enrolled Bill | none | REPORT | Ch1 §1.2.3 [VERIFY] marks |
| L119 | Prose-matching assertions track wording, not the property | none | REPORT | Suite maintenance; G9-2 repointed |
| L120 | `ccgt_efficiency_case` read only under `exempt`; repaired TASK 66 (allowance at dispatch efficiency) | 1 | STALE | Header says LIVE while its own update says 'RESOLVED for the sizing defect'; body's 'effect under exempt untested' is superseded by the update; the zero-span rows are inert by construction |
| L121 | `dscr_case` acts only on EaaS capacity; zero span at the traditional baseline | 1,2 | REPORT | Ch3 §3.4.2 wording corrected (TASK 64); do not publish the `dscr` row as a null; effect lives in the EaaS runners |
| L123 | The model does not represent new-build thermal efficiency; vintage-dispatch sensitivity moves the central plan | 1,3, cross | EPOCH | Flag `gas_vintage_dispatch` and regression arm exist; indexed under 'Deferred structural changes' |
| L125 | `gas_eta` and `ccgt_efficiency` are on different bases; common-basis uplift 1.53-1.80 | 1,3, cross | EPOCH | TASK 67; conversion already affected (0.46 against 0.287) |

## A.2 The STALE entries, and what is now true

- **L37**: What is true: F-31 (`findings_frozen.md`, Source line) is sourced from `05_run_fin3_capital_sweep.py`, re-derived post-epoch from `results/fin3/arms_92390a63/capital_sweep_results.csv`.
  So the entry's 'no number published depends on it' is false; the `budget_usd` / `budget_b_usd` basis hazard bears on F-31's tier labels.
- **L48**: What is true: the hardcoded `/21` binding-years denominator was fixed in TASK 20 Part B (header of `scripts/06_run_gas1_shadow_benchmarks.py`; evaluable years are used).
  The other four declared limitations, and the hardcoded `capital_case="moderate"`, stand.
- **L72**: What is true: the Pareto frontier now has six tight-end points per arm (0.08 to 0.01) and 128/128 optimal solves, and the loose end is located (anchor 504.1 Mt), `pareto_raw.csv`.
  The frontier never reaches infeasibility, so the concern moves from 'grid floor' to 'no boundary exists'; the REL floor-bound flagging claim stands.
- **L78**: What is true: `docs/HANDOVER_chapter_4.md:177` carries 'Withdrawn (register L78); do not quote it' (TASK 48 B.6), and Table 4.4 is rebuilt from `results/dem2/eaas_value.csv`.
  The handover's historical paragraph still shows the range, by design.
- **L79**: What is true: F-34's TASK 55 block is a post-epoch re-run of `results/gas2/arms_be9472ef/` (40 solves), so the pre-epoch files with wrong labels are superseded.
  The design caveat (capped solar arm, L81) stands.
- **L82**: What is true: the handover carries 'Corrected 2026-10-10 (TASK 43; L82)' beside §4.4 (`HANDOVER_chapter_4.md:130`); F-12 governs.
  The heading text itself is unchanged, annotation only.
- **L91**: What is true: Ch3 §3.7.4 now declares the reserve margin as 'an operating quantity used as a planning standard' with the 19.5 to 6.4 per cent fall (`CHAPTER_3_assembled.md:3244-3251`).
  The entry's 'names three and not this one' no longer holds.
- **L92**: What is true: post-epoch the VSS legs' frozen first-stage plan equals the unconstrained anchor (solar 39,030.1 MW, storage 63,056.3 MWh, base = anchor + 6.7 USD; F-42 TASK 55 block), mean 37.846 bn and median 27.502 bn.
  The pre-epoch 40.20 / 29.85 / 47% figures and the 0.6-point comparator error are gone (item C.3).
- **L100**: What is true: Ch3 §3.2.5 (`CHAPTER_3_assembled.md:990-995`) states 1.087668 against 1.0877 'the residual arising from rounding in the published factor sets'.
  The entry's wording fix is done; renormalisation is not.
- **L105**: What is true: Ch3 §3.4.2 (`:1790-1800`) states truncation 'holds in cost and not in the plan', using the eight horizon cells (`results/horizon_rebuild/`).
  The script's placeholder result and `no_salvage` confound are unchanged.
- **L120**: What is true: the status line says LIVE but the entry's last paragraph (TASK 66) says 'RESOLVED for the sizing defect'; the 'effect under exempt untested' sentence is superseded by the same update.
  The inert zero-span rows remain unpublishable.

Two further observations that do not make an entry STALE: (i) **L62's literal is itself stale in the way the entry predicts**: `scripts/run_central_sweep.sh:214,370` still say 21.328 bn while the current anchor is 24.987 bn; (ii) **L4 points at `docs/tier2_reconciliation.md`, which does not exist at HEAD**, so its memo defects cannot be checked; I classed the entry REPORT with that noted.

## A.3 The EPOCH rows, ordered by dependence

Four rows are EPOCH. Two further groups would become epoch items if the thesis chooses to implement them: the **deferred structural changes** (L95 hydro cost term, L96 busbar factor, L102 state of charge; indexed in the register), and the decision-dependent **L83** (option (a) re-derives every capital tier). The dependence order:

1. **L125 then L123 (efficiency basis, then vintage dispatch).** Decides the efficiency each stream converts at and whether new gas dispatches separately; it changes the baseline (24.987 bn central) and every plan.
2. **`gas_add` technology (undetermined, TASK 67 C.4)** is an input to step 1, not a separate register entry.
3. **L61 (cap series rebuild).** `data/cost/processed/emissions_cap.csv` is built by `scripts/00_build_emissions_cap.py` from the model's own baseline (Ch3 §3.7.6: 'the cap's baseline is inherited from a no-conversion run'); both files carry mtime 2026-10-19 23:36 today. Any change from step 1 to the baseline's gas-to-power split or emissions changes the caps and therefore **every capped result** (F-21, F-30 NDC legs, F-31, F-34, F-35, F-37, F-38, F-43, F-12/F-13).
4. **L51 (leakage diagnostic in the shared experiment functions).** A code change that must precede any re-run of the capped legs, so the re-run is measured; it fixes six of the nine scripts at once (the entry's own note).
5. **Emissions window (item B.2).** Not a register entry. If the Q3 quantities are to be quoted on 2025 to 2045, the capped runners must store emissions by year (F-43's `18_` does not), and this belongs in the same re-run as step 4.
6. **The capped ranking (Part D)** after 3 and 4.

Not epoch items, by this triage: L94 (done, pass 2), L92 (dissolved), L108 (pass 2).

## A.4 Contradictions

Between register entries and findings, documents or the suite (both sides named):

1. **L37 against F-31.** L37: 'no "fin3"/"FIN-3" citation anywhere in `findings_frozen.md`'. F-31's Source is `05_run_fin3_capital_sweep.py`.
2. **L95 against Ch3 §3.7.3.** L95: 'No document declares this for O&M'. Ch3 `:3160-3164` declares it and cites L95.
3. **L97 against invariant G7-3 (TASK 58 Part C).** L97 says `blend_case` is live; the suite's static read flagged it dead.
4. **L11 against L41.** L11: the `storage_binding` field is 'read by nothing'; L41: `plot_storage_regime.py` reads it (while broken).
5. **L120's status line against its own last paragraph** (LIVE against RESOLVED for the sizing defect).
6. **L48 (`/21` hardcoded) against the `06_` header** (fixed, TASK 20 Part B).
7. **L50's comment against its code**, in both builders: the comment says default 1.0, the code defaults to 1.3448 (`src/optimize_model_sliced.py:1324-1330`). This is the entry's own claim and persists at HEAD, not a contradiction between entries.
8. **L31 and L103 name the same defect** (the peak-adequacy diagnostic) in two entries, L31 LIVE, L103 half-resolved. They agree; they are one defect registered twice.
9. **L80's headline ('not diagnosed') against its own TASK 39 update** (two designs), and so against pass 1's Part 2.2 text, which repeated the headline.
10. **L89 and L105 against Ch3 §3.4.2.** Both say §3.4.2 mis-describes truncation; Ch3 `:1782-1798` now describes it correctly.
11. **L79 against F-34 (TASK 55 block)** (re-run done).
12. **Frozen findings against Chapter 1 (not a register contradiction, listed because it is the sharpest one found): F-21, F-39, F-40 and F-43 quote 2025 to 2054 emissions totals; Chapter 1 §1.6.3 says emissions are reported to 2045** (Part B).

---

# PART B: Question 3, depth audit

## B.1 The six findings

| Finding | Claim (headline, verbatim) | Source line | Arms | Post-epoch? |
|---|---|---|---|---|
| F-11 | 'the cap is grid-only by construction ... under a binding cap the model displaces load to priced self-generation, whose emissions sit outside the boundary. The NDC is met on paper while national emissions rise.' | 'no result file; structural argument, no quantitative claim at the frozen state' | none (structural) | n/a; the entry itself says do not quote a leakage percentage against it |
| F-21 | 'Tightening a grid-only cap RAISES national emissions' | `results/ndc/ndc3_conditional_decline_2cfa636a/` and `ndc3_conditional_604ec993/` (pre-epoch pair); post-epoch pair `results/ndc/ndc3_conditional_2e122592` and `ndc3_conditional_decline_db91e861` (Group 2, 2026-10-20 02:48, `0af709f`) | `ndc3_conditional` (flat hold) and `ndc3_conditional_decline`, capital case `moderate`, `conv_central`, capped solar design | Yes (TASK 55). Leakage and the grid / self-generation split are not in the summary JSON; read from `logs/phase_c/2_ndc_caps_with_decline.log` |
| F-31 | 'Under realistic collection friction a fully-funded capital envelope still leaves a fifth of emissions outside the cap' | `05_run_fin3_capital_sweep.py`; post-epoch `results/fin3/arms_92390a63/capital_sweep_results.csv` (2026-10-20 01:18) | 5 capital cases x 3 collection rates, NDC-3 unconditional caps | Yes (TASK 55) |
| F-39 | 'A tight capital envelope raises total emissions at zero carbon price' | `results/carbon_price_sweep/arms_92390a63/carbon_price_raw.csv` | `public_tight`, `public_unconstrained`, `eaas_tight`, 13 prices, `emissions_cap=None` | Yes (TASK 55) |
| F-40 | 'Under a binding capital envelope the carbon price does no abatement' | same file | same | Yes |
| F-43 | 'Under public financing the cost-emissions curve bends backward: past a point, tightening the grid cap raises total emissions' | `results/pareto_frontier/arms_92390a63/pareto_raw.csv` (2026-10-20 00:44; 128 solves) | 4 arms x [anchor + 25 grid + 6 tight-end points] | Yes (TASK 55) |

## B.2 The mechanism, from the stored diagnostics

**What the cap displaces.** The cap constrains grid-only emissions (`m.emissions_capped_by_year`); the reported total is grid plus self-generation (`scripts/18_run_pareto_frontier.py:569`; F-43's own check that the grid share equals the cumulative cap to 1.1e-8 where the cap binds in all years). Tightening it removes grid gas generation; demand is exogenous, so the load goes to the unbounded priced backstop or to other supply.

**The F-21 pair, recomputed from `diagnostics.json` and `summary.json` (self-generation emissions = selfgen TWh x 1e6 x 0.9022 tCO2/MWh_e, the `genset_central` factor; totals reproduce `summary.json`'s `actual_emissions_tco2_total` exactly):**

| Window | Quantity | Flat hold (`2e122592`) | Decline (`db91e861`) | Difference |
|---|---|---|---|---|
| **2025-2045 (reporting window)** | national emissions | 271.09 Mt | 268.86 Mt | **-2.23 Mt (-0.8%)** |
| | self-generation (TWh / Mt) | 68.57 / 61.86 | 88.45 / 79.80 | +17.9 Mt |
| | grid emissions | 209.22 Mt | 189.06 Mt | -20.2 Mt |
| | leakage share | 22.8% | 29.7% | |
| 2025-2054 (what F-21 quotes) | national emissions | 397.82 Mt | 418.13 Mt | **+20.30 Mt (+5.1%)** |
| | self-generation (TWh / Mt) | 117.64 / 106.13 | 217.34 / 196.07 | +90.0 Mt |
| | grid emissions | 291.7 Mt | 222.1 Mt | -69.6 Mt |
| | leakage share | 26.7% | 46.9% | |
| | grid gas generation (TWh), of which converted | 719.3 (691.5) | 550.1 (533.2) | -169.2 |

Cumulative difference in national emissions (decline minus flat), by year: +12.76 Mt (2035), +12.44 (2040), **-2.22 (2045)**, -5.26 (2046), -6.06 (2048), -0.42 (2050), +8.87 (2052), **+20.30 (2054)**. **Within the reporting window the abatement on the grid (-20.2 Mt) is 89% offset by diesel (+17.9 Mt) but not overturned; the net rise appears only from about 2050, driven by the decline arm's cap continuing to tighten toward zero at 2060 and by self-generation growing in the buffer years (annual self-generation difference +12.9 Mt in 2054).** The early positive difference (+12.76 Mt by 2035) is entirely self-generation in 2031 to 2034 (+1.14, +1.75, +3.69, +6.19 Mt, with identical grid emissions) in an arm whose cap is identical to the flat arm's through 2035: a perfect-foresight response to the later tightening, so it is leakage that anticipates the cap rather than follows it. From 2039 the decline arm's self-generation difference is zero and its national emissions are lower by 1.0 to 3.7 Mt a year through 2045; the self-generation difference returns in 2046 and grows to +12.9 Mt a year by 2054 (year-by-year differences computed from the two `diagnostics.json` files).

**Where the displaced demand goes.** In the 2054 totals, 169.2 TWh less grid gas generation and 99.7 TWh more self-generation: about 59% of the displaced gas generation reappears as diesel, the rest is met by other supply. Per tonne abated on the grid, 1.29 t appear in diesel (90.0 / 69.6).

**F-43 and F-21: the same mechanism or two?** The same displacement mechanism (cap on a boundary, exogenous load, priced backstop with a higher emission factor), but two different instruments and not equally supported. F-43 tightens a flat cap on all 30 years at a fraction of baseline; the total at each point is the 2025-2054 sum (the anchor 504.1 Mt equals `summary__6046f759.json`'s `cumulative_emissions_2025_2054_tco2`; the 2045 figure of the same arm is 338.2 Mt). F-21 tightens only after 2035, in a trajectory, and the inversion is a late-window and buffer effect. F-21 itself says F-43 is 'the same mechanism measured on a continuous instrument'; the stored files support that for the displacement and do not show the *inversion* inside the reporting window for F-21 (above) or allow it to be checked for F-43 (below).

**F-43's decomposition** (`pareto_raw.csv`, total = grid + self-generation, 2025-2054, Mt): `public_only_moderate` anchor 504.1 (0 self-generation); cap 1.0: 480.9 (0); 0.8125: 404.6 = 399.3 + 5.3; 0.40: 486.6 = 201.6 + 284.9; 0.10: 662.6 = 50.4 + 612.2; 0.01: 720.2 = 5.0 + 715.1. `public_only_tight`: 542.97 at the anchor (44.95 self-generation) rising to 1,015.9 at 0.01. `eaas_moderate`: 504.1 to 90.9 with no self-generation until 0.10. The grid column is exactly the cap, and the self-generation column carries the rise.

## B.3 The self-generation emission factor, and its exposure to L120 and L125

- **Grid gas, unconverted:** `EF_th` 0.1823 tCO2/MWh_th (`src/io.py:438`; IPCC 56.1 kg/GJ NCV converted to HHV, `docs/merged_correction_plan_v2.md:1.2`) divided by `gas_eta` 0.287: **0.6352** tCO2/MWh_e.
- **Grid gas, converted:** 0.1823 / 0.46 = **0.3963** (`conv_central`, design-point efficiency).
- **Self-generation, `genset_central`:** `ef_th_hhv` 0.25342 (AGO) / `genset_eta_hhv` 0.2809 (mean of Caterpillar C15/C18/3512 PRIME datasheets at 25% load; `data/cost/processed/genset_cost.csv`) = **0.9022**. `genset_srmc` 0.7295 (eta 0.3474); `genset_high` (petrol, eta 0.0511) 4.638.
- **Ratios:** self-generation to unconverted grid gas 1.42; **to the converted stream 2.28.** In the F-21 pair 96% of grid gas generation is the converted stream (691.5 of 719.3 TWh in the flat arm), so **the ratio that governs F-21 is 2.28.**
- **Exposure to the efficiency basis: yes.** The grid side of the ratio uses a design-point 0.46 (and the unconverted 0.287 is realised); the genset side is a part-loaded datasheet figure (0.2809 at 25% load), a realised-operation basis. On the common basis of TASK 67 (0.46 x 0.850 = 0.391) the converted-stream factor would be 0.466 and the ratio 1.93, not 2.28. The cap also buys grid generation at 1/EF_grid per tonne, 2.52 MWh/tCO2 at 0.3963 and 2.14 at 0.466, so the displaced load changes in the other direction. The net effect on the leakage quantity was not computed.

## B.4 Boundary resolution

| Finding | Quantity | Grid | Step | Reported value is |
|---|---|---|---|---|
| F-43 | turning point 0.8125 | `cap_fraction` 1.0 to 0.10 in 25 points, then 0.08, 0.06, 0.05, 0.03, 0.02, 0.01 | 0.0375 (grid part) | **the lowest sampled total (404.6 Mt) and an interior grid point; true minimum lies in (0.775, 0.85)**; past it a plateau of 407 to 415 Mt between 0.775 and 0.55 |
| F-43 | return to the cap-1.0 level 'about 0.40' | same | 0.0375 | interpolated between sampled points 0.4375 (466.5 Mt) and 0.40 (486.6 Mt); linear crossing of 480.9 Mt at about 0.41 |
| F-43 | exceeds the uncapped solve 'about 0.3625' | same | 0.0375 | between 0.3625 (504.7 Mt) and 0.40 (486.6): the crossing is at about 0.363, so it is the sampled point itself |
| F-39 / F-40 | divergence of `eaas_tight` from `public_unconstrained` 'only at $150' | prices 0, 5, 10, 15, 20, 30, 40, 50, 60, 75, 100, 125, 150 USD/tCO2 | 5 to 25, uneven | **the highest sampled price**: the arms are identical to 4 decimals at $125 (160.12 Mt) and differ at $150 (143.00 against 136.61 Mt); onset above $125 and behaviour above $150 are not located |
| F-39 / F-40 | `public_tight` 'no abatement' | same | | flat 542.97 to 541.95 Mt across all 13 prices; no onset sampled up to $150 |
| F-31 | leakage flag threshold 0.10 | 5 capital cases x 3 collection rates | categorical | a declared threshold, not a located one (the runner's own words); `expansion` unflagged 'is a property of the declared threshold' |
| F-21 | 40% tighter 2045 cap; +5.1% | a single pair | none | two points, no grid |
| F-11 | none | none | | no quantity |

## B.5 Leakage scope, completed

`leakage_diagnostic()` (`src/optimize_model.py:1830`) uses `m.emissions` (all 30 years), so every leakage share it reports is a 2025-2054 share.

| Runner | Calls it? (HEAD) | Findings that cite its capped output |
|---|---|---|
| `01_`, `01b_` | yes (baseline; 01b stores `leak=None`, uncapped) | none capped |
| `02_`, `03_` | yes | F-9 (NDC deliverability legs), F-10 (not citable), F-21 (flag in the log, null in the JSON), F-8 |
| `05_` | yes | F-31 |
| `15_` | yes | POL-1 |
| `17_` | records `leakage_flagged` | F-36 |
| `18_` | yes | F-43 |
| `99c_` | yes | |
| `19_` | n/a (`emissions_cap=None`) | F-39, F-40 |
| `09_`, `10_`, `28_` | n/a (no cap) | F-33 and others |
| **`24_`, `25_`, `26_`** | **no** | F-12, F-13 (`26_`) |
| **`04_`, `06_`, `07_`, `08_`, `11_`, `13_`** (through shared functions in `src/optimize_experiments.py`, which has 0 references to leakage) | **no** | **F-30 (NDC legs; `06_`), F-34 (`07_`), F-9's gas-regime legs (`08_`), F-35 (`11_`), F-37 and F-38 (`13_`), fin2 (`04_`, uncited)** |

Confirmed: F-39 and F-40 are uncapped and F-12 and F-13 are uncovered. **Completed list (the correction to pass 2): F-30 (NDC legs), F-34, F-35, F-37, F-38, F-12 and F-13 cite capped quantities for which the leakage share was never measured.** Whether those plans leak cannot be recovered from the stored files for `26_` (the f3 summaries hold only `total_eaas_mw`, `eaas_add_mw_by_year` and `solar_public_add_mw_total`); for `13_` the stored `dem2_results.csv` has `cumulative_emissions_tco2` but no self-generation column. The Q2 headline (F-38) is not a compliance claim, but it runs capped legs whose compliance is unlabelled.

---

# PART C: Question 4, depth audit

## C.1 The three findings

| Finding | Claim (headline, verbatim) | Source | Arms | Post-L66 and post-epoch? |
|---|---|---|---|---|
| F-32 | 'A deterministically-optimal plan carries zero adequacy margin, and fails every realisation above its central demand assumption' | `scripts/30_run_stochastic_redispatch.py`; post-epoch `results/stochastic_redispatch/draws__aaced7e1.jsonl` (leg A, `--fix-gas fixed`) and `draws__8fc2e490.jsonl` (leg B, `--fix-gas free`), 2026-10-20 00:24 and 00:25 | 100 draws each, seed 42, `conv_central`, `om_central`, window `arrears`, storage capex `declining_nrel`, `capital_case="moderate"` | **Yes both** (L66 fix `8885457` is 2026-09-28 10:43; the cited files are 2026-10-20). Older files (`76965a0b`, `a5e5b9e7`, `6fa97186`) are pre-fix, uncited. The zero-slack observation is inferred from the separation in the sliced files; measured directly only on the annual builder |
| F-33 | 'A horizon-aggregated reliability budget misstates its own standard' | `results/rel1/{total,annual}/gas_avail_{nameplate,observed}/` (`shortfall_by_year.csv`, `rel1_summary.json`), runner `09_run_rel1_feasibility.py`, Phase B item 14 | 2 reliability modes x 2 availability arms x 8 cells x 11 eps | **Post-epoch by mtime** (2026-10-19 23:59 to 2026-10-20 00:04) and the Phase B log, **not by tag** (L114: no `arm_tag`; written in place). No `30_` input, so L66 does not apply |
| F-42 | 'The value of the stochastic solution, bounded' | `results/vss_bound/vss_bound__d6f19a23.json`; stage 1 `summary__8fc2e490`, same draws sha256 | ECP leg B, `--fix-gas free`, `capital_case="moderate"`; WS matched to `30_` central | Yes both. All five guards pass |

## C.2 The three parts of the question

| Part | Answered by | Answered? |
|---|---|---|
| Adequacy across realisations under a committed plan | **F-32**: 50 of 100 draws infeasible under frozen gas (feasible `demand_growth` 0.0198 to 0.0389, infeasible 0.0396 to 0.0636, no overlap, boundary at `organic_central` 0.039); 0 of 100 with free gas | Yes, with the stated limit that the zero slack itself is measured on the annual builder only |
| Whether the specification of the service standard changes what is observed | **F-33**: horizon-total against annual form of the standard (worst year 0.2025 against 0.1000; horizon fraction 0.0385 against 0.0285; gas-regime range 0.049 against 0.079; solar-arm shift 0.001 against 0.049); the 'false null' claim is withdrawn as a grid artefact | Yes, at the central grid, with 5 of 8 cells floor-bound (all ranges are lower bounds) |
| An upper bound on the value of a non-deterministic formulation | **F-42**: `bound_upper` 7.244 bn = 19.1% of `ECP_free` 37.846 bn, a bound on VSS-partial that contains the expected value of perfect information | Yes, as an upper bound only (the entry's own reading) |

No part is unanswered.

## C.3 The VSS bound's legs, from the files

- **Leg `ECP_free`** (`results/stochastic_redispatch/draws__8fc2e490.jsonl`, 100 of 100 feasible): mean **37,846,482,445.52**, median **27,501,838,741.14**, P10 22,174,164,397.48, P90 72,974,538,267.85. **Leg WS** (`vss_bound__d6f19a23.json`, matched to `30_` central): 30,602,383,956.27. **Bound** `ECP - WS` = 7,244,098,489.24 (19.14% of `ECP_free`).
- **Comparators.** Anchor `summary__6046f759.json` (`unconstrained`): 24,986,620,949.18. Legs' deterministic base (`capital_case = moderate`): 24,986,620,955.93, **6.75 USD above the anchor**. Mean over anchor **+51.47%**; median over anchor **+10.07%**; mean over the legs' own base +51.47% (identical to two decimals).
- **What a correctly based comparison is, from stored files: the legs' base and the anchor now coincide, so the comparison is already on one basis.** L92's 47% (+0.6-point error) and its 40.20 / 29.85 are the pre-epoch figures; the F-42 TASK 55 block records that the frozen first-stage plan equals the anchor's (solar 39,030.1 MW, storage 63,056.3 MWh). **One thing not measured:** whether `moderate` binds in individual draws (F-42 says so). By construction I would expect it cannot in the recourse stage, because the solar and storage plan is frozen and the budget bounds public solar and storage capital, but that is inference, not a stored result. The right sentence is 'mean 51% and median 10% above the committed plan's deterministic cost', with the mean/median gap (1.38) stated; the correlation of draw cost with `demand_growth` is 0.77 (100 feasible draws).

## C.4 Does any Chapter 4 quantity for this question depend on the peak-adequacy diagnostic (L31 / L103)?

**No.** `peak_adequacy` appears in no frozen finding, no chapter (1 to 4, drafts included), no handover; the only occurrences outside the register are Appendix B's restatement of L31 (`docs/chapters/appendix_b.md:37,1392-1445`). F-32 reads the model's own `firm_capacity_constraint` slack and infeasibility; F-33 reads `shortfall_by_year.csv` (`unserved_twh + selfgen_twh`, exactly); F-42 reads costs. This closes it for the chapter; the instruction to state once that the field is a defect (L103) becomes an Appendix B statement, not a Chapter 4 caveat.

## C.5 What the service-standard comparison consists of

Two forms of the same reliability standard, **horizon-total** (a single shortfall budget over 2025 to 2054) and **annual** (the budget applied year by year), run through `scripts/09_run_rel1_feasibility.py` in `--reliability-mode total` and `annual`, each at `gas_avail_nameplate` and `gas_avail_observed`, over the gas-regime and solar-arm cells, an epsilon grid with floor 0.001 (extended from 0.05; L72). Stored: `results/rel1/{total,annual}/gas_avail_{nameplate,observed}/{shortfall_by_year.csv, rel1_summary.json, feasibility_matrix.csv, feasibility_threshold.csv}` (16 files, all 2026-10-19/20, no arm tag). `shortfall_twh` equals `unserved_twh + selfgen_twh` to a maximum difference of 0.0 (F-33's TASK 55 note), so the standard is the REL definition, not the renamed `unserved_only_reliability`.

---

# PART D: The capped ranking, costed not built

## D.1 Capped arms that exist, against the 16 families and 55 members

"Capped" means a solve under a binding NDC grid-only cap. Post-epoch capped stores: `results/ndc/` and `results/ndc_eaas/` (`02_`/`03_`), `results/gas3/arms_02e16d75/` (`08_`), `results/dem2/arms_31d60098/` (`13_`), `results/fin3/arms_92390a63/` (`05_`), `results/pareto_frontier/arms_92390a63/` (`18_`, central arms, cap fraction only).

| Family (members) | Capped members that exist | Where |
|---|---|---|
| gas_availability (2) | none | |
| conversion (4) | `conv_none`, `conv_central` (2 of 4) | `02_` under 4 caps; `conv_low`, `conv_high` absent |
| gas_retirement (5) | none | |
| capital_recovery (4) | none | |
| gas_deliverability_drift (4) | all 4, with flat controls | `08_` `results/gas3/arms_02e16d75/regime_ndc_matrix.csv`, 4 caps, 28 solves |
| gas_deliverability_gmp (4) | `baseline`, `gas_deliv_flat`, `gas_deliv_gmp_partial` (3 of 4) | `02_` under 4 caps; `gmp_full` absent |
| concessional_envelope (5), peak (3), hydro (3), new_gas_deliverability (2), storage_cc (3), gas_build (3), gas_capex (3), ccgt_efficiency (3), dscr (3), om (4) | none | |

Existing capped members: 4 + 3 + 2 families covered, **7 of the 40 unique arms** (39 non-central plus the central). **None is on the basis of the uncapped ranking** (`capital_case = unconstrained`): `02_` runs the cap cases at `tight` (unconditional) and `moderate` (conditional) (`scripts/02_run_ndc_caps.py:467-470`), `08_` at `moderate` and `expansion` (`scripts/08_run_gas3_regime_ndc_feasibility.py:117-120`). So a like-for-like capped ranking has **zero** reusable arms; a within-set ranking of the existing fragments is possible. Existing capped spans (bn, own capital case): conversion `none` against `central`, 34.89 (ndc2 unconditional), 32.16 (ndc2 conditional), 35.67 (ndc3 unconditional), 32.65 (ndc3 conditional), against 5.220 uncapped; drift family (gas3), 4.78, 0.04, 2.99, 0.09 for the same four caps (three of the four drift arms give identical cost, one regime, `downside`, differs, consistent with F-30's 'three of four regimes').

## D.2 Solves and wall time

- **Arms:** 16 families, 55 members, **40 unique arms** (15 of the members repeat the central arm). **Per cap case: 40 solves.**
- **Cap cases:** the four NDC cases (`ndc2/ndc3` x `unconditional/conditional`) plus the decline arm. One cap case: 40 solves; four: **160 solves**; five: 200. A binding "commitment" has to be chosen; Chapter 1 §1.4 does not choose.
- **Reuse:** at a single cap case in `02_`'s own capital case, 4 of the 40 exist (central, `conv_none`, flat, partial); 36 would be new. Across four caps, 16 exist and 144 would be new.
- **Observed times:** capped solves inside one runner: `08_` 34.0 s for 28 solves (1.2 s), `13_` 86.9 s for 72 (1.2 s), `18_` 154.2 s for 128 (1.2 s) (`logs/phase_b/11_gas3...`, `13_dem2...`, `20_pareto_frontier.log`, `real` lines). One `02_` invocation (5 cap scenarios incl. decline) took 9.0 s (`logs/phase_c/2_ndc_caps_with_decline.log`). A single-arm 01b invocation takes about 16 s (`logs/phase_c/5_A_om_none.log`).
- **Estimate:** 160 solves at 1.2 s is about 3.2 minutes of solve time; as 40 invocations of `02_` (4 caps each, ~7 to 9 s) about 5 to 6 minutes; as 160 single-arm invocations at 16 s about 43 minutes. All small.

## D.3 What the generator extension would consist of

- **Generator expects** (`scripts/build_lever_ranking.py`): `results/baseline_sliced/summary__<hash8>.json`; the `arm_tag` must equal the tag derived by `arm_for` (double-underscore separators plus the post-epoch suffix); stored `om_case` must match; cumulative 2045/2054 quantities from `decision_variables`.
- **Capped runners write** (`02_`): `results/ndc/<case>_<hash8>/{summary.json, diagnostics.json, timeseries.csv}` (short-tag directory); the `arm_tag` in the file uses single-underscore separators with a `voll_mid_` prefix and the cap case in the name; summary keys include the case fields and `decision_variables`, but the cumulative aggregates are the unlabelled names of L111 (`solar_total_built_mw` and similar), not `solar_cumulative_built_2045_mw`; `capital_case` varies by cap case; no epoch tokens as `__` tokens.
- **Reader or adapter:** a **reader** (a second loader in the generator: glob `ndc/*/summary.json`, compare the stored case fields instead of the tag, compute cumulative build from the `decision_variables` additions, key rows by (family, member, cap case, capital case)) is the smaller change. An **adapter** that rewrites capped outputs into the baseline layout must fabricate or re-derive the tag and relabel the aggregates, and creates files that look like baseline results (the L85 and L114 hazard). A third route, adding a cap option to `01b_run_baseline_sliced.py`, gives the same layout but changes a runner that carries byte-identity regression guards (TASK 60 and 62); it is the largest change.
- **Not written, as instructed.**

## D.4 Is a capped ranking well defined for every family?

- **Infeasibility:** not expected structurally. The cap is grid-only, unserved energy is fixed at zero under a priced genset, and self-generation is unbounded, so any arm can be served. Stored evidence: **0 infeasible** in `pareto_raw.csv` 128/128 optimal (caps to 0.01 of baseline, four arms), `regime_feasibility.csv` 28/28 feasible, `dem2_results.csv` 72/72 optimal, `capital_sweep_results.csv` 15/15 optimal. The only infeasible capped configuration recorded is `selfgen_in_emissions_cap = True` (L73: infeasible for all three genset arms, and `genset_arm_sensitivity.csv` public-only at every eps), which is a switch and not a lever arm. **Not demonstrated** for the fourteen families with no capped run.
- **A span can still be undefined or degenerate:** (i) several arms give *identical* capped cost (the three drift arms, baseline, upside, shock_recovery, equal in `regime_ndc_matrix.csv`), so a span is zero and the build-response rule classes the family as one plan; (ii) the cap series is built once from the no-conversion baseline run (Ch3 §3.7.6), so an arm whose own uncapped emissions fall below the cap (low-gas or high-solar arms) meets a cap that does not bind and its capped result equals its uncapped one; whether the cap binds is checkable per arm (`cap_binding` is stored in the frontier files) and was not tested for the unrun arms; (iii) the cap's own definition depends on L61 and the efficiency decision (A.3).

---

# Corrections to passes 1 and 2

| Where | Said | Files show |
|---|---|---|
| Pass 2 item 5 | F-12/F-13 are the uncovered findings; F-39/F-40 uncapped | Nine runners lack the diagnostic; F-30 NDC legs, F-34, F-35, F-37 and F-38 are also uncovered (B.5) |
| Pass 2 item 3 | F-43's turning point is a grid value; L72's text is stale | Both confirmed; the larger point is the **window**: F-43's totals are 2025-2054 |
| Pass 1 Part 4.2 (L92) | median 29.85 against mean 40.20; re-run at `unconstrained` (~200 LPs) is an epoch item | Pre-epoch figures; post-epoch median 27.502, mean 37.846, base = anchor + 6.7 USD; the re-run is unnecessary for the central plan (C.3) |
| Pass 1 Part 2.2 | F-10's three sets 'never diagnosed' | L80's own TASK 39 update diagnoses them as two designs; F-10 remains not citable |
| Pass 1 Part 0.2 | L94 buffer test never run | Corrected in pass 2; Ch3 §3.4.2 already uses the eight cells |
| Pass 1 Part 3.3 | F-43's grid against L72 | The frontier already extends to 0.01 (A.2, L72) |

