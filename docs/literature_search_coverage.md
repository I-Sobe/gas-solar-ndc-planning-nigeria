# What a text search over `literature/` can and cannot cover

Recorded TASK 73 Part C.3 (2026-10-23) so that no negative over `literature/` is read as covering a file it did not. Text-layer check: `pdftotext` over each PDF, non-whitespace characters per page. `literature/` is gitignored (`.gitignore:31`), so this list is the only record of what is on disk.

| File | Pages | Text chars per page | Searchable by text? |
|---|---|---|---|
| `Electricity_Act_2023.pdf` | 151 | **0** | **No. No text layer; a scan. Never searched, never read visually. A negative over `literature/` does NOT cover it.** |
| `Heating_Values_of_Natural_Gas_and_Its_Components.pdf` (NBSIR 82-2401) | 172 | 974 | Partly. An OCR layer exists but is poor on numeric tables; tables were read visually only for PDF pp.45, 56, 57 (Tables 10 and 14). A numeric negative in this file by text is weak. |
| `Electricity_Act_amendment.pdf` | 6 | 600 | Yes (short document) |
| `principles-of-project-finance.pdf` (slides) | 103 | 528 | Yes, sparse (slides) |
| `2021_Assessment_of_Reliability_Performance_Appendices.pdf` | 76 | 800 | Yes |
| `SPP_ELCC_2022.pdf` | 16 | 1,018 | Yes |
| all others (`DPR_Nigerian_Gas_Transportation_Network_Code`, `NERC_2024_Annual_Report`, `capital_cost_AEO2025`, `NIRP_2024`, `NDC_3.0`, `Nigeria-BTR1-`, `Nigeria_BTR1_NID_2024`, `PAD00037_…`, `WorldBank_discounting_guidance`, `Liddle_Hasanov`, `Principles_of_project_finance`, `2024_ReliabilityPerformance…`) | 11-575 | 1,247-2,430 | Yes (text-layer PDFs; charts carry text labels) |

Documents that are **not** in `literature/` and so are outside any negative over it: the ATB workbooks (`data/cost/raw/solar_capex/2024_v3_Workbook.xlsx`; v4 only in the owner's Downloads), the NERC 2025 quarterly reports (`data/gas/raw/NERC_report/`, Q1-Q4 only; **no 2026/Q1 report**), the EIA discussion slides of 19 September 2023, the Nigerian Gas Company volume document, the NMDPRA natural gas reports (access locked), and the Oak Ridge flue-gas database (unavailable).

The rule from TASK 73: every negative states documents, sheets or page ranges, search terms, and whether scans were read visually or by extracted text.
