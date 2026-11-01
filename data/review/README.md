# data/review — frozen evidence from the systematic review

These are the evidence artefacts behind the thesis's Chapter 2 and the
standalone review manuscript. They are frozen: nothing here is edited by
analysis code. Scripts read from these files and write derived tables to
`docs/`.

| File | What it is |
|---|---|
| `extraction_workbook_v5.xlsx` | Per-study extraction, codebook, synthesis matrix, Results panels, and a changelog of every correction applied to v4. |
| `fulltext_screening_sheet.xlsx` | Full-text screening decisions, exclusion reasons, PDF retrieval status, and stage tallies. |
| `MANIFEST.sha256` | Checksums for every artefact, in `data/` and `docs/review/` alike. |

Protocol, amendments, screening rulebook, search strings, the limitations
register and the manuscript live in `docs/review/`.

## Corpus, as at this freeze

- Included studies: **75**
- Financing treatment: exogenous 65, absent 10, endogenous 0
- Dimensional coverage: availability 75, deliverability 59, access 44,
  affordability 75, acceptability 50
- Deliverability/access co-occurrence: both 39, deliverability-only 20,
  access-only 5, neither 11
- Enacted-not-named: 38 of the 39 studies representing both dimensions

Any script that recomputes these and disagrees should report the
disagreement rather than correct either side. The counts above are the
reconciliation target, not an assumption.

## Verifying integrity

    sha256sum -c data/review/MANIFEST.sha256

Re-run `scripts/setup_review_artefacts.sh` after replacing any artefact,
so the manifest tracks the current files.
