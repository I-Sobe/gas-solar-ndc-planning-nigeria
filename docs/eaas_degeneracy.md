# Is the public/EaaS solar split determinate?

Status: **scaffold written; no solves run.** `scripts/diagnose_eaas_degeneracy.py` fills the AUTOGEN block (Parts A–C) when Zeus runs it. Part D and the Disagreements/Inconclusive sections below the block are completed by hand from that output. Companion to `docs/financing_endogeneity.md` (Parts A–C there are complete and not redone here).

Do **not** cite the Part B Test 3 "binding vintage-years" column anywhere: it counts left-hand side equal to right-hand side and is non-zero with the constraint deactivated.

## 0. Context read before running (not results of this task)

**F-12 already establishes non-identification of the split** (`docs/findings_frozen.md:763`), by lexicographic min/max of total EaaS MW at fixed cost, tolerance 1e-6, at `ndc3_conditional`: current epoch, budget `tight` 31,875 – 48,579 MW, `moderate` 10,990 – 48,579, `unconstrained` 0 – 48,579. The maximum is total solar, "not a bankability ceiling"; only the floor moves with the budget. F-12's reporting rule: "any EaaS figure must carry both its `capital_case` and its `tariff_case`, and be reported as a point in the stated interval." F-38 (TASK 30 Part D) and F-35 (binding-year counts) already carry `[UNIDENTIFIED]` tags on split-derived quantities, and F-43 corrects "EaaS solar rises" to a total.

So Part C of the endogeneity report (C3, C5) framed as open a question the project had partly closed. What is **new** here is (i) the interval at the *central no-policy* configuration and the other named configurations, (ii) a second solver and a perturbation ladder, and (iii) a per-configuration stability verdict. The lexicographic test (A4 in the script) is F-12's own method, added so the new numbers are comparable with F-12's.

## Disagreements (pre-run)

1. **Provenance of the F-38 figures.** The task cites them to `docs/HANDOVER_chapter_1.md`; that file does not exist in this repo (only `docs/HANDOVER_chapter_4.md`). The `[VERIFY]` was resolved against `docs/findings_frozen.md` instead: anchor 27,317,670,155.72 (line 33); F-38 table 0.109B at 3.9% growth / baseline gas / no_policy (stated "0.11B" in the heading text). Both hold. `docs/financing_endogeneity.md` now carries that annotation.
2. **Budget status is arm-specific, not contradictory.** F-38 says the *public* arm's envelope binds at the central cell (dual 0.20086) and the *EaaS* arm's dual is zero at 3.9% growth. Part B found the EaaS-arm budget non-binding at P2. These agree. A "binding public capital budget" configuration for EaaS therefore needs the `tight` case; the script uses `budget_tight_nopolicy` and `cap_ndc3unc_tight` and records whether each budget binds in the EaaS solve.
3. **HiGHS access.** The Pyomo `highs` interface needs `highspy`, which is not installed. The script drives the installed HiGHS 1.15.1 CLI through an LP file instead. Consequence: HiGHS yields primal values only (no reduced costs); reduced costs come from CBC.
4. **"Smooth" vs LP structure.** In a pure LP a perturbation either leaves the vertex unchanged or moves it discretely; a *smooth* monotone response is not generic. The script's ladder therefore has a fourth label (*insensitive*) for a split that does not move at all under δ ≤ 1e-2, which counts toward *determinate*.
5. **δ = 1e-6 resolution.** A 1e-6 relative perturbation of the EaaS capital charge is small against the solver's absolute tolerances on a 27.3 bn objective (the cost differences between tied vertices are of the order of a few USD, cf. Part B: ≤ 5.5 USD). The ladder is read together with A4 and A2, not alone.
6. **What the perturbation touches.** It scales the EaaS-solar *initial-build* tranche charge only (`cost_eaas_solar_capital_charge_npv`). The rebuild charge pools public and EaaS adds into one term and is not perturbed; the split-identity and bankability coefficients are not perturbed either. Stated in the script header and in the output.

<!-- BEGIN AUTOGEN (scripts/diagnose_eaas_degeneracy.py) -->

*Not yet run.*

<!-- END AUTOGEN -->

## Part D — to be written by hand after the run

1. Verdict table (configuration × stability verdict × evidence) — from the AUTOGEN summary table.
2. One-paragraph statement of what is and is not determined in the public/EaaS split.
3. Proposed limitations-register entry (drafted, not filed).
4. Affected figures by finding ID. **Pre-run candidate list** from a text search of `findings_frozen.md` for EaaS-capacity quantities: F-12 (the interval itself; already carries the reporting rule), F-35 (binding-year counts, already `[UNIDENTIFIED]`), F-38 (EaaS MW, already `[UNIDENTIFIED]`; identified quantities are total solar and cost), F-43 (already corrected to a total), F-22 (mechanism text: EaaS solar CAPEX does not consume the budget — a statement about the LP, not a reported MW). F-32 mentions `solar_eaas_add` as a fixed-plan input. Which of these are *tied at a configuration where Chapter 4 reports the figure* is decided by the verdicts, not by this list.

## Inconclusive

*To be completed after the run.*
