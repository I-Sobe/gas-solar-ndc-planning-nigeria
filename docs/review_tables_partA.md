# Chapter 2 review tables — Part A inventory

_Generated for TASK.md Part A. Part B not started (stop gate). Nothing under `data/review/` other than `table1_extracted.csv` was written; manuscript and workbooks untouched._

## A1. Table 1 extraction

- Source: `docs/review/manuscript_final.docx`, python-docx `document.tables`. 4 tables present; **index 0** has 76 rows (1 header + 75 data) and is Table 1. The others have 6, 7 and 41 rows.
- Header as printed: `#, Study, Year, SSA system, Model class, Security dims, Financing, Derivation (SQ7)` → mapped positionally to `study_no, authors, year, scope, model_class, security_dims, financing_treatment, derivation_tag`.
- Written to `data/review/table1_extracted.csv`, cell text verbatim. (`MANIFEST.sha256` not updated — this file is new.)
- Rows extracted: **75** (expected 75).
- Observations on raw cells: `model_class` casing is inconsistent (`geospatial` ×14, `Geospatial` ×10); study #7 (Anderson et al.) has an **empty year**; study #31 and #54, #56, #69 are rendered without "et al." (single-author).

## A2. Derivation-tag inventory

| derivation_tag (verbatim) | n |
|---|---|
| `undefended-flat` | 31 |
| `—` | 10 |
| `undefended-flat (swept)` | 8 |
| `argued` | 7 |
| `argued (swept)` | 6 |
| `precedent-cited` | 2 |
| `argued (swept), technology-diff` | 2 |
| `institution-sourced (swept), country×mode` | 2 |
| `precedent-cited (swept), country×mode` | 1 |
| `precedent-cited, country-diff` | 1 |
| `institution-sourced` | 1 |
| `not stated` | 1 |
| `precedent-cited, technology-diff` | 1 |
| `precedent-cited (swept)` | 1 |
| `undefended-flat, technology-diff` | 1 |

15 distinct values, total 75.

- `financing_treatment == exogenous`: **65** (expected 65).
- Of those, non-empty `derivation_tag` (literal): **65**; excluding the `—` placeholder: **65**.
- Financing values overall: {'exogenous': 65, 'absent': 10}.
- Rows tagged `—`: 10; their financing values: {'absent': 10}.
- Rows with financing `absent` and a tag other than `—`: [].
- `not stated` appears at study no. ['40']; it has no differentiation component.
- Tag grammar observed: `<rung>[ (swept)][, <differentiation>]`; a tag with no differentiation part occurs for most rows (implicit `uniform`? — not inferred here).

### Table 1 internal counts (for reference against the reconciliation targets)

- Model class (case-folded): {'math-prog': 43, 'geospatial': 24, 'partial-eq': 5, 'sim-hybrid': 2, 'nat-iam': 1}
- Dimension counts Av/De/Ac/Af/Ap: {'Av': 75, 'De': 59, 'Ac': 44, 'Af': 75, 'Ap': 50} (targets 75/59/44/75/50)

## A3. Table 1 vs workbook v5 (`Extraction`, rows with financing exo/abs/end)

Workbook in-scope rows: **75** (rows 4–123 restricted per B1 rule).

### Unmatched Table 1 rows

Matching key: first surname token of Table 1 `authors` = first token of workbook `Author(s)` (case/diacritic-folded) and equal year string.

| # | Table 1 authors | year | strings tried |
|---|---|---|---|
| 1 | Franceso Fuso et al. | 2016 | `franceso` + `2016` |

### Workbook in-scope rows matched by no Table 1 row

| wb row | record_id | Author(s) | Year |
|---|---|---|---|
| 116 | 30528 | A cost comparison of technology approaches for improving access to ele | 2016 |

### Ambiguous matches (>1 workbook row for the same author+year)

- #64 Sahlberg et al. 2023: workbook rows [14, 16] ([102, 391]). Not resolved.
- #65 Sahlberg et al. 2023: workbook rows [14, 16] ([102, 391]). Not resolved.

### Disagreements (matched rows)

| study | field | Table 1 | workbook |
|---|---|---|---|
| — | — | — | — |

Field-level disagreements among matched rows: **0**.

### Name/identity disagreements (author strings that matched but differ in form)

| study | Table 1 | workbook `Author(s)` | note |
|---|---|---|---|
| #7 | Anderson et al.  | Anderson Gwanyebit Kehbila, Robert Karisa Masumbuko, Mbeo Og  | wb row 120 |
| #16 | Dalla et al. 2018 | Dalla Longa, Strikkers, Kober & Van der Zwaan 2018 | wb row 119 |
| #34 | Kemausuor et al. 2014 | Kemausuor et al. 2014 | wb row 121 |
| #37 | Korkovelos et al. 2019 | Korkovelos et al. 2019 | wb row 123 |
| #39 | Marong et al. 2018 | Marong et at. 2018 | wb row 122 |
| #51 | Nadia et al. 2023 | Nadia S. Ouedrago 2023 | wb row 117 |
| #66 | Sahlberg et al. 2021 | Sahlberg, Khavari, Korkovelos, Fuso Nerini & Howells 2021 | wb row 118 |

_Check on the zero result:_ the workbook in-scope rows independently reproduce the targets (exogenous/absent 65/10; model class 43/24/5/2/1; dimensions 75/59/44/75/50), and Table 1 reproduces the same, so agreement on these three fields is a real result rather than a vacuous comparison. Study-level agreement was tested via author+year matching; for #64/#65 both candidate workbook rows have identical model class and dimensions, so the three fields agree whichever row is paired.

## Part B

Not started — awaiting release of the stop gate.

## Disagreements (consolidated)

| # | Item | Value A (source) | Value B (source) |
|---|---|---|---|
| 1 | Study #51 surname | `Nadia et al.` (Table 1) | `Nadia S. Ouedrago` (workbook row 117, record 30529). TASK states the surname is *Ouedraogo*: the workbook has the same forename-first inversion **and** a different spelling (`Ouedrago`). |
| 2 | Study #1 identity | `Franceso Fuso et al.` 2016 (Table 1) | workbook row 116 (record 30528) has the paper **title** in `Author(s)` (`A cost comparison of technology approaches…`) and the author list (`Franceso Fuso, Oliver Broad, …`) in `Title`. Not matchable by author; row 116 is the only unmatched workbook row. Also "Franceso" looks like a misspelling of Francesco in both places. |
| 3 | Study #7 year | empty (Table 1) | year cell empty in workbook row 120 too (both blank; consistent but missing). Workbook `Author(s)` is a forename list (`Anderson Gwanyebit Kehbila, Robert Karisa Masumbuko, …`), so Table 1's `Anderson` may be a forename too — not determined. |
| 4 | Study #16 | `Dalla et al.` (Table 1) | `Dalla Longa, Strikkers, Kober & Van der Zwaan` (row 119): the compound surname is truncated in Table 1. |
| 5 | Study #39 | `Marong et al.` (Table 1) | `Marong et at.` (row 122) — typo in workbook. |
| 6 | Studies #64 / #65 | two Table 1 rows `Sahlberg et al.` 2023 | two workbook rows (14: record 102; 391: record 391) with the same key; pairing is ambiguous and not resolved. Fields agree either way. |
| 7 | Model-class casing | `geospatial` ×14 and `Geospatial` ×10 (Table 1, same column) | workbook uses one code `geospatial-least-cost`. Counts agree once case-folded (24). |
| 8 | Workbook authors format | Table 1 uses `Surname et al.` | workbook rows 4–115 use lower-cased space-joined surname strings; rows 116–123 use free-form citations (`Sahlberg, Khavari, …`, `Nadia S. Ouedrago`, …). Informational. |
| 9 | Field-level model_class / security_dims / financing | — | **No disagreements** across the 74 key-matched studies; #1 agrees by inspection. |

No Table 1 vs target mismatches were found for: n=75, financing 65/10/0, dimensions 75/59/44/75/50, model class 43/24/5/2/1.

## Unresolved

- Study #1 could not be matched programmatically (workbook row 116 has author/title fields swapped), so it is excluded from the automated comparison. Read from the dump of row 116: `geospatial-least-cost`, dimensions Y/Y/Y/Y/N, `exogenous` — same as Table 1 (Geospatial, Av·De·Ac·Af, exogenous). The pairing itself is by inspection, not by key.
- #64/#65 pairing to workbook rows 14/16 is unresolved.
- Targets for accessibility reading (39/20/5/11) and enacted-not-named (38) are not part of Part A and were not checked.
- Whether tags lacking a differentiation component (e.g. bare `undefended-flat`, 31 rows) mean `uniform` is a coding question for Zeus; nothing inferred. `argued (swept), technology-diff` shows the tag grammar admits `(swept)` before the comma.
- `python-docx` was not installed in `nigeria-energy`; I ran `pip install python-docx` (1.2.0) to do A1. The task's Part B script isn't written. Part A code lives in the session scratchpad, not in the repo.
