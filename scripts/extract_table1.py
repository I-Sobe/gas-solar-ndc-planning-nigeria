#!/usr/bin/env python3
"""
Extract Table 1 (per-study summary) from the Chapter 2 manuscript.

Reads the python-docx table object holding the per-study summary (the table with
76 rows: 1 header + 75 studies; table index 0 in manuscript_final.docx) and
writes its cells verbatim to data/review/table1_extracted.csv with columns

    study_no, authors, year, scope, model_class, security_dims,
    financing_treatment, derivation_tag

No normalisation of any cell. NOTE: the committed CSV has since been repaired
(author names, model-class casing; see docs/review_tables.md section 1), so
re-running this script reproduces the PRE-repair file. Use --out to write
elsewhere.

    python scripts/extract_table1.py [--docx PATH] [--out PATH]
"""
import argparse
import csv
import sys
from pathlib import Path

import docx

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["study_no", "authors", "year", "scope", "model_class",
           "security_dims", "financing_treatment", "derivation_tag"]
N_STUDIES = 75


def find_table1(document):
    """Return (index, table) for the unique table with N_STUDIES data rows."""
    hits = [(i, t) for i, t in enumerate(document.tables)
            if len(t.rows) == N_STUDIES + 1 and len(t.columns) == len(COLUMNS)]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one {N_STUDIES}-row table, found {len(hits)}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--docx", default=str(ROOT / "docs/review/manuscript_final.docx"))
    ap.add_argument("--out", default=str(ROOT / "data/review/table1_extracted.csv"))
    args = ap.parse_args()

    idx, table = find_table1(docx.Document(args.docx))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for row in table.rows[1:]:
            w.writerow([c.text for c in row.cells])
    print(f"table index {idx}: wrote {N_STUDIES} rows to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
