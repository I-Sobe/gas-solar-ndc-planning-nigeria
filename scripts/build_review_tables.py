#!/usr/bin/env python3
"""
Chapter 2 review artefacts (TASK v3): regenerates docs/review_tables.md and docs/reference_gaps.md.

Reads  data/review/ (current files; data/review/_preimage/ = post-v2 state for the repair diffs) and, read-only,
docs/review/manuscript_final.docx + docs/review/*.md. Writes docs/review_tables.md and docs/reference_gaps.md.
Modifies nothing under data/review/ (repairs live in scripts/repair_review_data.py).

Sections  1 repairs and verification (v3 §1-§2)   2 artefacts: Panel B, rung x swept (asserted against v2 values)
          3 PRISMA flow (sheet as found + labelled projection)   4 Disagreements   5 Unresolved
Assertions REPORT failures; they never raise.

    python scripts/build_review_tables.py
"""
import argparse
import csv
import difflib
import hashlib
import re
import sys
import unicodedata
from collections import Counter, OrderedDict
from pathlib import Path

import docx
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "data/review"
PRE = REV / "_preimage"
WB = "extraction_workbook_v5.xlsx"
OUT = ROOT / "docs/review_tables.md"

DIMS = OrderedDict([("Availability", "I"), ("Deliverability", "K"), ("Access", "M"),
                    ("Affordability", "O"), ("Acceptability", "Q")])
CLASSES = OrderedDict([("math-prog", "mathematical-programming"), ("geospatial", "geospatial-least-cost"),
                       ("partial-eq", "partial-equilibrium"), ("sim-hybrid", "simulation-optimisation hybrid"),
                       ("nat-IAM", "national-IAM")])
T_DIM = {"Availability": 75, "Deliverability": 59, "Access": 44, "Affordability": 75, "Acceptability": 50}
T_CLS = {"math-prog": 43, "geospatial": 24, "partial-eq": 5, "sim-hybrid": 2, "nat-IAM": 1}
PRED_B = {"Availability": [43, 24, 5, 2, 1], "Deliverability": [28, 24, 4, 2, 1],
          "Access": [14, 24, 4, 1, 1], "Affordability": [43, 24, 5, 2, 1],
          "Acceptability": [37, 7, 5, 1, 0]}
RUNGS = ["not stated", "undefended-flat", "argued", "precedent-cited", "institution-sourced",
         "market-benchmarked"]
DIFFS = ["country-diff", "technology-diff", "country×mode"]
PRED_X = {"not stated": (1, 0), "undefended-flat": (32, 8), "argued": (7, 8), "precedent-cited": (4, 2),
          "institution-sourced": (1, 2), "market-benchmarked": (0, 0), "endogenous": (0, 0)}
PRED_DIFF = {"technology-diff": 4, "country×mode": 3, "country-diff": 1}

NAME_BASIS = {  # study_no -> why the author key was changed (static provenance)
    "1": "task §1.2 (surname stated); NO entry in the reference list",
    "2": "reference list: 'Afful-Dadzie, A.' (hyphen lost in Table 1)",
    "7": "task §1.2; workbook row 120 author list begins 'Anderson Gwanyebit Kehbila'; NO entry in the reference list",
    "16": "task §1.2; workbook row 119 'Dalla Longa, …'; reference list has only a machine-generated entry ('longa strikkers …')",
    "38": "reference list: 'Kueppers, M.' (Table 1 'Kuppers')",
    "47": "task §1.2; reference list: 'Moner-Girona, M.'",
    "51": "task §1.2 (surname stated as Ouedraogo); NO entry in the reference list; workbook spelled it 'Ouedrago'",
    "61": "reference list: 'Mohd Rafiq, N.' (compound surname truncated at first space)",
}

MD = []          # report lines
DIS = []         # (item, value A [source], value B [source])
UNR = []         # unresolved statements


def emit(*lines):
    MD.extend(lines)
    OUT.write_text("\n".join(MD) + "\n", encoding="utf-8")


def fold(s):
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in r) + " |")
    return out


def load_csv(p):
    return list(csv.DictReader(open(p, encoding="utf-8", newline="")))


def load_ws(path, data_only=True):
    return openpyxl.load_workbook(path, data_only=data_only)["Extraction"]


def in_scope(ws):
    return [r for r in range(4, 124)
            if str(ws.cell(r, 21).value or "").strip().lower()[:3] in ("exo", "abs", "end")]


def cls_of(code):
    for k, v in CLASSES.items():
        if str(code).strip() == v:
            return k
    return None


def yes(v):
    return str(v or "").strip().lower().startswith("y")


def check(label, got, want):
    ok = got == want
    if not ok:
        DIS.append((f"Reconciliation target: {label}", f"{want} (TASK targets)", f"{got} (computed)"))
    return "ok" if ok else "**MISMATCH**"


# ------------------------------------------------------------------ reference list
def reference_list():
    paras = [p.text for p in docx.Document(ROOT / "docs/review/manuscript_final.docx").paragraphs]
    a = next(i for i, p in enumerate(paras) if p.strip() == "REFERENCE")
    b = next(i for i, p in enumerate(paras) if p.startswith("Appendix A"))
    refs = [p for p in paras[a + 1:b] if p.strip()]
    mach = [p for p in refs if re.search(r"\b\d+\.0(\(\d+\))?[,.]", p)]
    proper = [p for p in refs if p not in mach]
    return refs, mach, proper


def ref_surname(p):
    return fold(p.split(",")[0].split(" (")[0])



# ------------------------------------------------------------------ §1 (v3 repairs: pre-image = post-v2 state)
def ft_rows(path):
    ws = openpyxl.load_workbook(path)["Full-text screening"]
    R = {}
    for r in range(2, ws.max_row + 1):
        v = [c.value for c in ws[r]]
        if v[1] is not None:
            R[v[1]] = dict(row=r, rid=v[1], year=v[2], title=v[3], au=v[4], dec=v[5], reason=v[6], owed=v[7],
                           dup=v[8], pdf=v[9])
    return R


def tally_cells(path, data_only):
    ws = openpyxl.load_workbook(path, data_only=data_only)["Tally"]
    out = []
    for r in range(4, 11):
        v = ws.cell(r, 2).value
        out.append((ws.cell(r, 1).value, getattr(v, "text", v)))
    return out


def section1(t1, t1_pre, ws, ws_pre, W):
    emit("# Chapter 2 review tables — TASK v3", "",
         "_Regenerated by `scripts/build_review_tables.py`. Section 1 is derived by diffing `data/review/_preimage/` "
         "(the post-v2 state, refreshed at the start of v3) against the current files; the v2 originals are in "
         "`data/review/_preimage/v2/`. The v2 report is kept at `docs/review_tables_v2.md`. Reference-list material is in "
         "`docs/reference_gaps.md`. Nothing is committed._", "")
    emit("## 1. Repairs and verification (TASK v3 §1–§2)", "")
    # 1.0 pre-image / manifest
    man = {m.group(2): m.group(1) for m in re.finditer(r"^([0-9a-f]{64})  (\S+)$",
                                                       (REV / "MANIFEST.sha256").read_text(), re.M)}
    rows = []
    for name in (WB, "table1_extracted.csv", "fulltext_screening_sheet.xlsx"):
        rel = f"data/review/{name}"
        ok = man.get(rel) == sha(ROOT / rel)
        rows.append((f"`{rel}`", sha(PRE / name)[:16] + "…", sha(ROOT / rel)[:16] + "…",
                     "yes" if ok else "**NO**", "changed" if sha(PRE / name) != sha(ROOT / rel) else "unchanged"))
        if not ok:
            DIS.append((f"MANIFEST entry for {rel}", man.get(rel, "absent")[:16], sha(ROOT / rel)[:16]))
    emit("### 1.0 Pre-images and manifest (§2.5)", "",
         *table(["file", "pre-image sha256 (post-v2)", "current sha256", "current == MANIFEST", "status"], rows), "",
         "`MANIFEST.sha256` regenerated by `scripts/repair_review_data.py`; the changed checksums are the pre-image → "
         "current pairs above (the workbook and the full-text sheet changed; the CSV changed by the three year edits).", "")

    # 1.1 Sahlberg
    emit("### 1.1 Sahlberg 2023 pairing (settled)", "")
    srows = []
    for study, r in (("#64", 14), ("#65", 16)):
        no = study[1:]
        t = next(x for x in t1 if x["study_no"] == no)
        srows.append((study, f"row {r}, record {ws.cell(r, 1).value}", str(ws.cell(r, 5).value)[:30], t["scope"],
                      ws_pre.cell(r, 31).value, ws.cell(r, 31).value, t["derivation_tag"]))
    emit(*table(["study", "workbook row", "workbook scope (E)", "Table 1 scope", "AE before", "AE after", "Table 1 tag"], srows), "")
    for s_, _, _, _, _, after, tag in srows:
        if after != tag:
            DIS.append((f"derivation_tag for Sahlberg {s_}", f"{tag} (Table 1)", f"{after} (workbook AE)"))
    r14 = " ".join(str(ws.cell(14, 16).value).split())
    emit("Third corroboration checked in the text: row 14 `Affordability — mechanism` contains "
         + ("“technology-differentiated discount rates — SA PV 18.0%, mini-grids 19.8%, centralized grid 15.5%”."
            if "technology-differentiated discount rates" in r14 and "19.8%" in r14 else "**NOT FOUND**."), "")

    # 1.2 years
    emit("### 1.2 Year corrections", "")
    yrows = []
    for no, rec, row, frm, to in (("15", "202", 69, "2024", "2023"), ("19", "14486", 108, "2026", "2025"),
                                  ("50", "16726", 7, "2021", "2022")):
        c_pre = next(x for x in t1_pre if x["study_no"] == no)["year"]
        c_cur = next(x for x in t1 if x["study_no"] == no)["year"]
        ok = (c_pre, str(c_cur), str(ws_pre.cell(row, 3).value), str(ws.cell(row, 3).value),
              str(ws.cell(row, 1).value)) == (frm, to, frm, to, rec)
        yrows.append((f"#{no}", rec, row, f"{c_pre} → {c_cur}", f"{ws_pre.cell(row, 3).value} → {ws.cell(row, 3).value}",
                      "ok" if ok else "**UNEXPECTED**"))
        if not ok:
            DIS.append((f"Year correction #{no}", f"{frm} → {to} expected", f"{c_pre}→{c_cur} (CSV); {ws_pre.cell(row, 3).value}→{ws.cell(row, 3).value} (workbook)"))
    emit(*table(["study", "record", "wb row", "Table 1 CSV", "workbook C", "check"], yrows), "")
    g_csv = next(x for x in t1 if x["study_no"] == "28")["year"]
    emit(f"- **#28 Guta: unchanged at {g_csv} (CSV) / {ws.cell(53, 3).value} (workbook row 53, record {ws.cell(53, 1).value}).** "
         "The reference list carries 2015 — an open conflict, see `docs/reference_gaps.md` §3.3.")
    keep = [("58", "7334", 2018), ("66", "30530", 2021), ("70", "416", 2016)]
    kr = []
    for no, rec, yr in keep:
        wrow = next(r for r in W if str(ws.cell(r, 1).value) == rec)
        kr.append((f"#{no}", rec, next(x for x in t1 if x["study_no"] == no)["year"], ws.cell(wrow, 3).value,
                   "unchanged" if str(ws.cell(wrow, 3).value) == str(yr) else "**CHANGED**"))
    emit("- **#58, #66, #70 are first-author collisions, not year errors; left as they stand:**", "",
         *table(["study", "record", "CSV year", "workbook year", "status"], kr), "")

    # 1.3 / 1.4 verification on the full-text sheet
    cur = ft_rows(REV / "fulltext_screening_sheet.xlsx")
    emit("### 1.3 Full-text screening decisions — verification (not re-decided)", "")
    def reason(k):
        v = cur.get(k, {}).get("reason")
        return "—" if v in (None, "") else " ".join(str(v).split())[:45]
    chk = [("312", "decision = Exclude", cur.get(312, {}).get("dec") == "Exclude", f"decision={cur.get(312, {}).get('dec')}, reason={reason(312)}"),
           ("15201", "exclusion reason removed (undecided, unretrieved)", cur.get(15201, {}).get("reason") in (None, ""), f"decision={cur.get(15201, {}).get('dec')}, reason={reason(15201)}"),
           ("1592", "exclusion reason removed (undecided, unretrieved)", cur.get(1592, {}).get("reason") in (None, ""), f"decision={cur.get(1592, {}).get('dec')}, reason={reason(1592)}"),
           ("110", "reason = R5", code_of(cur.get(110, {}).get("reason")) == "R5", f"decision={cur.get(110, {}).get('dec')}, reason={reason(110)}"),
           ("19073", "retained", 19073 in cur, f"present, decision={cur.get(19073, {}).get('dec')}")]
    emit(*table(["record", "expected (TASK v3 §1.3)", "applied?", "found in sheet"],
                [(a, b, "yes" if c else "**NO**", d) for a, b, c, d in chk]), "")
    emit("### 1.4 Duplicate records — verification (not re-deleted)", "")
    gone = [10352, 8385, 16624, 22043]
    surv = [9286, 7047, 16616, 19073]
    emit(*table(["record", "expected", "status in sheet"],
                [(k, "absent", "absent" if k not in cur else f"**STILL PRESENT** (row {cur[k]['row']}, decision={cur[k]['dec']})") for k in gone] +
                [(k, "present", "present" if k in cur else "**MISSING**") for k in surv]), "")
    notdone = [a for a, _, c, _ in chk if not c] + [str(k) for k in gone if k in cur]
    if notdone:
        emit(f"**None of the §1.3/§1.4 sheet edits is present in `data/review/fulltext_screening_sheet.xlsx` "
             f"(items not as expected: {', '.join(notdone)}).** The sheet in the repo may be a stale copy of the one Zeus edited. "
             "Per the task I have not applied or re-applied any of them. §3 below therefore reports the sheet as it stands, "
             "and separately a clearly labelled projection of what the counts would be if the listed decisions were applied.", "")
        for a, b, c, d in chk:
            if not c:
                DIS.append((f"Full-text sheet record {a}: {b}", "applied by Zeus (TASK v3 §1.3)", f"not present in repo sheet: {d}"))
        for k in gone:
            if k in cur:
                DIS.append((f"Duplicate record {k} removed", "removed by Zeus (TASK v3 §1.4)", f"still present in repo sheet (row {cur[k]['row']})"))
        UNR.append("The full-text sheet in the repo does not contain Zeus's §1.3 decisions or the four duplicate removals "
                   f"({', '.join(notdone)}); confirm whether a newer copy exists and drop it into data/review/ (then re-run "
                   "`repair_review_data.py` and `build_review_tables.py`).")

    # 2.1 record ids
    emit("### 2.1 Snowballed record_id (workbook → sheet ids)", "")
    rid = []
    for r in (121, 122, 123):
        rid.append((r, ws_pre.cell(r, 1).value, ws.cell(r, 1).value, str(ws.cell(r, 4).value)[:55]))
    emit(*table(["wb row", "record_id before", "record_id after", "title"], rid), "")
    sheet_ids = sorted(k for k in cur if k >= 30000)
    wb_ids = sorted(int(ws.cell(r, 1).value) for r in range(116, 124))
    emit(f"Sheet snowballed ids: {sheet_ids}; workbook rows 116–123 now: {wb_ids} — "
         + ("identical." if sheet_ids == wb_ids else "**DIFFER**."), "")
    if sheet_ids != wb_ids:
        DIS.append(("Snowballed record_ids", str(sheet_ids) + " (sheet)", str(wb_ids) + " (workbook)"))

    # 2.2 Tally
    emit("### 2.2 Tally ranges", "")
    fb, fa = tally_cells(PRE / "fulltext_screening_sheet.xlsx", False), tally_cells(REV / "fulltext_screening_sheet.xlsx", False)
    vb, va = tally_cells(PRE / "fulltext_screening_sheet.xlsx", True), tally_cells(REV / "fulltext_screening_sheet.xlsx", True)
    emit(*table(["metric", "formula before", "formula after", "value before", "value after"],
                [(fb[i][0], f"`{fb[i][1]}`", f"`{fa[i][1]}`", vb[i][1], va[i][1]) for i in range(7)]), "",
         "Every range now ends at row 400. **Deviation from a bare range extension:** `Still undecided` and `PDFs outstanding` "
         "count cells that are blank / not `Y`, so on rows 189–400 (empty) a plain extension would add 212 phantom records "
         "to each. Those two formulas therefore also require a non-blank `record_id` (column B). Without the guard they would "
         "read 260; with it they read the values shown.", "")
    ftn = len(cur)
    if va[0][1] != ftn:
        DIS.append(("Tally 'Records at full-text stage'", f"{va[0][1]} (Tally cached value)", f"{ftn} (rows counted)"))
    a12 = openpyxl.load_workbook(REV / "fulltext_screening_sheet.xlsx")["Tally"]["A12"].value
    emit(f"Stale text left unedited: Tally!A12 still reads “these 186(179+7 fro snowball) are TITLE/ABSTRACT includes…” "
         f"against {ftn} records and {len([k for k in cur if k >= 30000])} snowballed.", "")
    DIS.append(("Tally!A12 note", "186 = 179 + 7 snowballed (note text)",
                f"{ftn} records, {len([k for k in cur if k >= 30000])} snowballed (counted from rows)"))

    # 2.3 / 2.4
    pop = [r for r in range(4, 145) if ws.cell(r, 31).value]
    emit("### 2.3 derivation_tag completion", "",
         f"Rows populated in `Extraction!AE`: **{len(pop)}** (expected 75: {check('derivation_tag rows populated', len(pop), 75)}). "
         "Cells disagreeing with `table1_extracted.csv`: "
         + str(sum(1 for r in pop if ws.cell(r, 31).value not in [x['derivation_tag'] for x in t1])) + ".", "")
    emit("### 2.4 Repair code", "",
         "`scripts/repair_review_data.py` now reproduces the v2 and v3 workbook/CSV/Tally repairs. It rebuilds the workbook "
         "and CSV from `_preimage/v2/` and edits the Tally in place. Idempotence check at authoring time: two consecutive "
         "runs produced byte-identical sha256 for the workbook, the CSV and the full-text sheet.", "")
    # complete diff of the workbook cells
    wbp = openpyxl.load_workbook(PRE / WB, data_only=True)
    wbc = openpyxl.load_workbook(REV / WB, data_only=True)
    diffs = []
    for name in wbp.sheetnames:
        for row in wbc[name].iter_rows():
            for c in row:
                a = wbp[name][c.coordinate].value
                if a != c.value:
                    diffs.append((f"{name}!{c.coordinate}", a, c.value))
    emit("**Complete cell-level diff of the workbook, pre-image → current** (all five sheets):", "",
         *table(["cell", "before", "after"], diffs), "")
    # CSV diff
    cd = []
    for a, b in zip(t1_pre, t1):
        for k in a:
            if a[k] != b[k]:
                cd.append((f"#{a['study_no']}", k, a[k], b[k]))
    emit("**`table1_extracted.csv` diff, pre-image → current:**", "", *table(["study", "column", "before", "after"], cd), "")



# ------------------------------------------------------------------ §2
def section2(ws, W):
    emit("### 2a. Panel B — security dimension × model class", "",
         f"Source: `{WB}` / `Extraction`, rows 4–123 restricted to `Financing treatment` beginning exo/abs/end → "
         f"**{len(W)}** rows.", "")
    cnt = {d: Counter() for d in DIMS}
    cc = Counter()
    unk = []
    for r in W:
        c = cls_of(ws.cell(r, 6).value)
        if c is None:
            unk.append((r, ws.cell(r, 6).value)); continue
        cc[c] += 1
        for d, col in DIMS.items():
            if yes(ws[f"{col}{r}"].value):
                cnt[d][c] += 1
    rows, comp = [], {}
    for d in DIMS:
        vals = [cnt[d][c] for c in CLASSES]
        comp[d] = vals
        rows.append((d, *vals, sum(vals)))
    emit(*table(["Dimension", *CLASSES, "row total"], rows),
         "", f"Class sizes (column totals of the corpus): {dict((c, cc[c]) for c in CLASSES)}; "
         f"unclassifiable rows: {unk or 'none'}.", "")
    res = []
    for d in DIMS:
        res.append((f"row total {d}", sum(comp[d]), T_DIM[d], check(f"Panel B row total {d}", sum(comp[d]), T_DIM[d])))
    for c in CLASSES:
        res.append((f"class total {c}", cc[c], T_CLS[c], check(f"Panel B class count {c}", cc[c], T_CLS[c])))
    emit("**Assertions (reported, not raised):**", "", *table(["check", "computed", "target", "result"], res), "")
    dif = []
    for d in DIMS:
        for i, c in enumerate(CLASSES):
            if comp[d][i] != PRED_B[d][i]:
                dif.append((d, c, PRED_B[d][i], comp[d][i], comp[d][i] - PRED_B[d][i]))
                DIS.append((f"Panel B cell {d} × {c}", f"{PRED_B[d][i]} (v2 report value)",
                            f"{comp[d][i]} (computed from workbook v5)"))
    emit("**Cell-by-cell diff vs the v2 table**:", "",
         *(table(["dimension", "class", "v2", "computed", "Δ"], dif) if dif else
           ["All 25 cells equal the v2 values."]), "")
    return comp, cc


# ------------------------------------------------------------------ §3
TAG = re.compile(r"^(?P<rung>not stated|undefended-flat|argued|precedent-cited|institution-sourced|"
                 r"market-benchmarked)(?P<swept> \(swept\))?(?:, (?P<diff>country-diff|technology-diff|country×mode))?$")


def section3(t1):
    emit("### 2b. Artefact 2 — derivation rung × swept", "",
         "Source: `data/review/table1_extracted.csv` (`derivation_tag` verbatim), the 65 rows with "
         "`financing_treatment == exogenous`. Parsed grammar: `<rung>[ (swept)][, <differentiation>]`. "
         "An empty differentiation field means *not recorded as differentiated*; it is never mapped to `uniform`.", "")
    exo = [r for r in t1 if r["financing_treatment"].strip().lower() == "exogenous"]
    endo = [r for r in t1 if r["financing_treatment"].strip().lower().startswith("end")]
    parsed, unp = [], []
    for r in exo:
        m = TAG.match(r["derivation_tag"].strip())
        if m:
            parsed.append(dict(no=r["study_no"], au=r["authors"], rung=m["rung"], swept=bool(m["swept"]), diff=m["diff"] or ""))
        else:
            unp.append(r)
    emit(f"Exogenous studies: **{len(exo)}** (target 65: {check('exogenous n', len(exo), 65)}); parsed: {len(parsed)}; "
         f"UNPARSED: {len(unp)}.", "")
    emit("**UNPARSED:** " + ("none" if not unp else ""), "",
         *[f"- #{r['study_no']} {r['authors']}: `{r['derivation_tag']}`" for r in unp], "")
    for r in unp:
        UNR.append(f"UNPARSED derivation tag, study #{r['study_no']}: `{r['derivation_tag']}`")
    grid = {k: [0, 0] for k in RUNGS}
    for p in parsed:
        grid[p["rung"]][1 if p["swept"] else 0] += 1
    rows = [(k, grid[k][0], grid[k][1], sum(grid[k])) for k in RUNGS]
    rows.append(("endogenous", 0, 0, len(endo)))
    tn = sum(g[0] for g in grid.values()); ts = sum(g[1] for g in grid.values())
    rows.append(("**total**", tn, ts, tn + ts))
    emit("#### 2b.1 Primary table", "", *table(["rung", "not swept", "swept", "total"], rows),
         "", "Zero rows (`market-benchmarked`, `endogenous`) are shown deliberately.", "")
    dif = []
    for k in RUNGS + ["endogenous"]:
        got = tuple(grid[k]) if k in grid else (0, 0)
        if got != PRED_X[k]:
            dif.append((k, PRED_X[k], got))
            DIS.append((f"Artefact 2 cell {k} (not swept, swept)", f"{PRED_X[k]} (v2 report value)", f"{got} (computed)"))
    emit("Diff vs the v2 primary table: "
         + ("**all 14 cells equal the v2 values.**" if not dif else ""), "",
         *(table(["rung", "v2 (ns, s)", "computed"], dif) if dif else []), "")
    swept = [p for p in parsed if p["swept"]]
    emit(f"Swept studies: **{len(swept)}** of {len(exo)} (target 20: {check('swept', len(swept), 20)}). "
         f"Rung distribution among swept: {dict(Counter(p['rung'] for p in swept))}.", "")
    dl = [p for p in parsed if p["diff"]]
    emit("#### 2b.2 Differentiation list", "",
         f"Differentiated studies: **{len(dl)}** (target 8: {check('differentiated', len(dl), 8)}).", "",
         *table(["study", "first author", "rung", "swept", "differentiation"],
                [(f"#{p['no']}", p["au"], p["rung"], "yes" if p["swept"] else "no", p["diff"]) for p in dl]), "",
         f"Totals by value: {dict(Counter(p['diff'] for p in dl))} (v2: {PRED_DIFF}).", "")
    cd = Counter(p["diff"] for p in dl)
    for k, v in PRED_DIFF.items():
        if cd.get(k, 0) != v:
            DIS.append((f"Differentiation total {k}", f"{v} (v2 report value)", f"{cd.get(k, 0)} (computed)"))
    low = sum(1 for p in parsed if p["rung"] in ("not stated", "undefended-flat"))
    hi = sum(1 for p in parsed if p["rung"] in ("precedent-cited", "institution-sourced", "market-benchmarked"))
    diff_arg = sum(1 for p in dl if p["rung"] not in ("not stated", "undefended-flat"))
    flat = [p for p in parsed if p["rung"] == "undefended-flat"]
    flat_d = sum(1 for p in flat if p["diff"])
    der = [("at or below undefended-flat (not stated + undefended-flat)", f"{low} of {len(parsed)}", "41 of 65"),
           ("at precedent-cited or above", f"{hi} of {len(parsed)}", "9 of 65"),
           ("differentiated studies at argued or above", f"{diff_arg} of {len(dl)}", "7 of 8"),
           ("undefended-flat studies that differentiate", f"{flat_d} of {len(flat)}", "1 of 40")]
    emit("#### 2b.3 Derived counts", "", *table(["quantity", "computed", "v2 value"], der), "")
    for q, c, p in der:
        if c != p:
            DIS.append((f"Derived count: {q}", f"{p} (v2 report value)", f"{c} (computed)"))
    emit("`argued` counts as “argued or above”; the ladder order is the TASK's: not stated < undefended-flat < argued "
         "< precedent-cited < institution-sourced < market-benchmarked.", "")
    return parsed



# ------------------------------------------------------------------ §4
def code_of(reason):
    if reason is None:
        return "NO REASON RECORDED"
    head = " ".join(str(reason).split())[:110]
    fam = []
    for name, pat in [("METHOD-R1", r"METHOD-R1"), ("R1", r"\bR1\b"), ("R2", r"\bR2\b"), ("R3", r"\bR3\b"),
                      ("R4", r"\bR4\b"), ("R5", r"\bR5\b"), ("Gate 0/1 (non-SSA)", r"\bGate [01]\b|Geography"),
                      ("Document type", r"Document type"), ("Duplicate", r"(?i)^duplicate")]:
        if re.search(pat, head):
            fam.append(name)
    if "METHOD-R1" in fam:
        fam = [f for f in fam if f != "R1"]
    return fam[0] if len(fam) == 1 else ("MULTIPLE: " + ", ".join(fam) if fam else "UNCODED (free text)")



# ------------------------------------------------------------------ §3 PRISMA (v3)
def prisma_counts(R):
    """R: {rid: row-dict}. Returns the stage counts and the exclusive partition."""
    db = [x for x in R.values() if x["rid"] < 30000]
    ca = [x for x in R.values() if x["rid"] >= 30000]
    nr = [x for x in R.values() if x["pdf"] != "Y"]
    ass = [x for x in R.values() if x["pdf"] == "Y"]
    inc = [x for x in R.values() if x["dec"] == "Include"]
    exc = [x for x in R.values() if x["dec"] == "Exclude"]
    und = [x for x in R.values() if x["dec"] not in ("Include", "Exclude")]
    part = Counter(((x["dec"] if x["dec"] in ("Include", "Exclude") else "undecided"),
                    "PDF" if x["pdf"] == "Y" else "no PDF") for x in R.values())
    return dict(total=len(R), db=len(db), ca=len(ca), nr=len(nr), ass=len(ass), inc=len(inc), exc=len(exc),
                und=len(und), part=part, db_ids=[x["rid"] for x in db])


def project(R):
    """What the counts would be if TASK v3 §1.3/§1.4 were applied (HYPOTHETICAL; nothing is written)."""
    P = {k: dict(v) for k, v in R.items() if k not in (10352, 8385, 16624, 22043)}
    if 312 in P:
        P[312]["dec"] = "Exclude"
    for k in (15201, 1592):
        if k in P:
            P[k]["reason"] = None
    if 110 in P:
        P[110]["dec"], P[110]["reason"] = "Exclude", "R5"
    return P


def section_prisma():
    R = ft_rows(REV / "fulltext_screening_sheet.xlsx")
    P = project(R)
    a, b = prisma_counts(R), prisma_counts(P)
    note = openpyxl.load_workbook(REV / "fulltext_screening_sheet.xlsx")["Tally"]["A12"].value or ""
    m = re.search(r"([\d,]+) identified -> ([\d,]+) screened", note)
    ident = f"{m.group(1)} identified → {m.group(2)} screened (Tally!A12)" if m else "[not found in Tally!A12]"
    emit("## 3. PRISMA flow — recomputed (TASK v3 §5)", "",
         "Column **sheet as it stands** is counted from `data/review/fulltext_screening_sheet.xlsx` as found. Column "
         "**projection** applies the same §1.3/§1.4 decisions in memory; when the sheet already carries them (§1.3/§1.4 all verified) "
         "the two columns must agree, and the projection is then only a cross-check. Database arm = record_id < 30000; citation arm = record_id ≥ 30000 (the sheet has no source column).", "")
    rows = [("Records identified — database searching", ident, ident),
            ("Records identified — citation searching (snowballing)", a["ca"], b["ca"]),
            ("Reports sought at full text, database arm (expected 179)", a["db"], b["db"]),
            ("Reports sought at full text, citation arm (expected 8)", a["ca"], b["ca"]),
            ("Total reports sought (expected 183 after the four removals)", a["total"], b["total"]),
            ("Reports not retrieved (PDF ≠ Y)", a["nr"], b["nr"]),
            ("Reports assessed for eligibility (PDF = Y)", a["ass"], b["ass"]),
            ("Reports excluded at full text", a["exc"], b["exc"]),
            ("Studies included (expected 75)", a["inc"], b["inc"]),
            ("Undecided", a["und"], b["und"])]
    emit(*table(["Stage", "sheet as it stands", "projection if §1.3/§1.4 applied"], rows), "")
    for lab, got, want in (("Reports sought, database arm", a["db"], 179), ("Reports sought, citation arm", a["ca"], 8),
                           ("Total reports sought", a["total"], 183), ("Studies included", a["inc"], 75)):
        if got != want:
            DIS.append((f"PRISMA: {lab}", f"{want} (TASK v3 §5 expected)", f"{got} (sheet as it stands)"))
    if b["db"] + b["ca"] != 183 or b["db"] != 179:
        DIS.append(("TASK v3 §5 expectations are mutually inconsistent",
                    "database arm 179 + citation arm 8 = 187; total 183 'after the four duplicate removals'",
                    f"projection gives database arm {b['db']} + citation arm {b['ca']} = {b['total']}: 179 is the pre-removal database count"))
    emit(f"Note: the task expects database arm 179, citation arm 8 *and* total 183. These cannot all hold: 179 + 8 = 187. "
         f"Removing the four duplicates (all database-arm records) from 179 leaves {b['db']}; {b['db']} + {b['ca']} = {b['total']}.", "")
    for lab, k, want in (("Records at full-text stage vs manuscript", "total", 179), ("Assessed vs manuscript", "ass", 131),
                         ("Excluded vs manuscript", "exc", 56)):
        DIS.append((f"PRISMA (manuscript): {lab}", f"{want} (manuscript)", f"{a[k]} (sheet as it stands) / {b[k]} (projection)"))
    DIS.append(("PRISMA (manuscript): not retrieved", "48 = 44 unretrievable + 4 duplicates (manuscript)",
                f"{a['nr']} (sheet; no unretrievable/duplicate split) / {b['nr']} (projection, duplicates gone)"))

    # reconciliation
    emit("### Does it reconcile?", "",
         "The four categories in the task overlap (a record can be excluded *and* have no PDF), so the check is run on the "
         "mutually exclusive decision × PDF partition:", "")
    cells = ["Include|PDF", "Exclude|PDF", "Exclude|no PDF", "undecided|PDF", "undecided|no PDF", "Include|no PDF"]
    def prow(c):
        d, p = c.split("|")
        return (c.replace("|", " + "), a["part"].get((d, p), 0), b["part"].get((d, p), 0))
    emit(*table(["category", "sheet as it stands", "projection"], [prow(c) for c in cells] +
                [("**sum**", sum(a["part"].values()), sum(b["part"].values())), ("total reports sought", a["total"], b["total"])]), "")
    for lab, X, Rr in (("sheet as it stands", a, R), ("projection", b, P)):
        ok = sum(X["part"].values()) == X["total"]
        emit(f"- {lab}: included {X['inc']} + excluded {X['exc']} + undecided {X['und']} = {X['inc'] + X['exc'] + X['und']} "
             f"vs total {X['total']} → **{'reconciles' if X['inc'] + X['exc'] + X['und'] == X['total'] else 'DOES NOT reconcile'}**; "
             f"not-retrieved {X['nr']} + assessed {X['ass']} = {X['nr'] + X['ass']} → "
             f"**{'reconciles' if X['nr'] + X['ass'] == X['total'] else 'DOES NOT reconcile'}**.")
        odd = [x for x in Rr.values() if (x["dec"] == "Exclude" and x["pdf"] != "Y") or
               (x["dec"] not in ("Include", "Exclude") and x["pdf"] == "Y") or (x["dec"] == "Include" and x["pdf"] != "Y")]
        none_cat = [x for x in Rr.values() if x["dec"] is not None and x["dec"] not in ("Include", "Exclude")]
        emit("  - cross-category records (excluded without a retrieved PDF, or undecided with a PDF): "
             + ("none" if not odd else "; ".join(f"{x['rid']} ({x['au']}, decision={x['dec']}, PDF={x['pdf']})" for x in odd))
             + (f"; records with an unrecognised decision value: {[x['rid'] for x in none_cat]}" if none_cat else ""))
        for x in odd:
            DIS.append((f"PRISMA category overlap [{lab}]: record {x['rid']} ({x['au']})",
                        f"decision={x['dec']}", f"PDF={x['pdf']} (excluded without a retrieved PDF / undecided with a PDF)"))
    emit("")

    # exclusion reasons
    for lab, Rr in (("sheet as it stands", R), ("projection", P)):
        ex = [x for x in Rr.values() if x["dec"] == "Exclude"]
        cc = Counter(code_of(x["reason"]) for x in ex)
        emit(f"### Full-text exclusions by reason code — {lab}", "",
             *table(["reason code", "n"], sorted(cc.items(), key=lambda kv: -kv[1])), f"Total: {sum(cc.values())}.", "")
        bad = [x for x in ex if code_of(x["reason"]).startswith(("UNCODED", "MULTIPLE", "NO REASON"))
               or code_of(x["reason"]) == "Duplicate"]
        emit(f"Excluded records **not carrying a single eligibility reason** ({lab}):", "",
             *table(["record", "authors", "year", "status", "reason (start)"],
                    [(x["rid"], x["au"], x["year"], code_of(x["reason"]), " ".join(str(x["reason"]).split())[:90]) for x in bad]), "")
        if lab == "sheet as it stands":
            for x in bad:
                why = ("'duplicate' is not an eligibility reason" if code_of(x["reason"]) == "Duplicate" else code_of(x["reason"]))
                UNR.append(f"Excluded record {x['rid']} ({x['au']}): {why} (sheet as it stands).")
    emit("Note: `METHOD-R1` (record 13563, Ronoh) is the re-coded exclusion from the v2 task; `Gate 0/1` combines the "
         "sheet's ‘Gate 0’, ‘Gate 1’ and ‘Geography’ wordings (all non-SSA). Codes are read from the first ~110 characters; "
         "a reason naming none or several is reported, not assigned.", "")
    emit("**For the write-up (stated, not resolved):** PRISMA 2020 expects a separate identification arm for records found by "
         "methods other than database searching. The eight snowballed records belong in that arm, not folded into the "
         "database count.", "")
    UNR.append("PRISMA: which figures the diagram carries, and whether the 4 duplicates (database arm) are reported in the "
               "deduplication box — Zeus's decision. Database-arm identification figures (23,717 → 2,521) are taken from "
               "Tally!A12 text only; they are not computable from the sheet.")
    return a, b


# ------------------------------------------------------------------ reference_gaps.md (v3 §3)
STOP = set("a an the of in on for and or to with by from at as is are its their using use based towards toward via into"
           " case study analysis model modelling modeling".split())


def words(s):
    s = re.sub(r"[-\u2010-\u2015/]", " ", str(s))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return {w[:6] for w in re.findall(r"[a-z]{3,}", s) if w not in STOP}   # 6-letter stems: z/s spellings, plurals


def entry_title(p):
    m = re.search(r"\((?:\d{4}|n\.d\.)\)\.\s*(.*)", p)
    t = m.group(1) if m else p
    return re.sub(r"https?://\S+", "", t)


def contain(a, b):
    """fraction of the smaller word set found in the other"""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


DOI = re.compile(r"10\.\d{4,9}/[^\s\"<>)\]]+[^\s\"<>)\].,;]")


def repo_doi_hits(surname, year, tw, skip_refs):
    """DOIs found in review files in a blob (line/cell/paragraph) that mentions the study."""
    hits = []
    sources = []
    for p in sorted((ROOT / "docs/review").glob("*.md")):
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            sources.append((f"{p.relative_to(ROOT)} line {i}", line))
    for p in (REV / WB, REV / "fulltext_screening_sheet.xlsx"):
        wb = openpyxl.load_workbook(p, data_only=True)
        for ws in wb:
            for row in ws.iter_rows():
                for c in row:
                    if isinstance(c.value, str) and "10." in c.value:
                        sources.append((f"{p.relative_to(ROOT)} {ws.title}!{c.coordinate}", c.value))
    d = docx.Document(ROOT / "docs/review/manuscript_final.docx")
    paras = [p.text for p in d.paragraphs]
    for i, t in enumerate(paras):
        if t not in skip_refs:
            sources.append((f"docs/review/manuscript_final.docx paragraph {i}", t))
    for src, text in sources:
        dois = DOI.findall(text)
        if not dois:
            continue
        tf = fold(text)
        if (surname and fold(surname) in tf and str(year) in text) or contain(tw, words(text)) >= 0.75 and len(tw) >= 4:
            for x in dois:
                hits.append((x, src, text.strip()[:140]))
    return hits


def write_reference_gaps(t1, ws, W):
    refs, mach_sig, _ = reference_list()
    # machine = signature OR near-duplicate stub (found below); proper = has no duplicate flag
    ents = [dict(text=p, title=entry_title(p), tw=words(entry_title(p)), sig=p in mach_sig,
                 doi=bool(re.search(r"doi\.org/|10\.\d{4,}/", p))) for p in refs]
    out = ["# Reference-list worklist (TASK v3 §3) — REPORT ONLY", "",
           "_Generated by `scripts/build_review_tables.py`. `docs/review/manuscript_final.docx` is read-only and was not edited. "
           "Nothing below is invented bibliographic data; absent fields are marked `[MISSING: …]`._", ""]
    # ---- 3.2 machine-generated / duplicates (compute first; needed to define 'proper')
    pairs = []
    for i, e in enumerate(ents):
        for j, f in enumerate(ents):
            if i < j and len(e["tw"]) >= 4 and len(f["tw"]) >= 4 and contain(e["tw"], f["tw"]) >= 0.8 \
                    and fold(e["text"].split()[0])[:5] == fold(f["text"].split()[0])[:5]:
                pairs.append((i, j))
    stub = {}
    for i, j in pairs:
        a_, b_ = ents[i], ents[j]
        # the stub = machine-signature member, else the one without a DOI/journal link, else the shorter
        if a_["sig"] != b_["sig"]:
            s_, p_ = (i, j) if a_["sig"] else (j, i)
        elif a_["doi"] != b_["doi"]:
            s_, p_ = (i, j) if not a_["doi"] else (j, i)
        else:
            s_, p_ = (i, j) if len(a_["text"]) < len(b_["text"]) else (j, i)
        stub.setdefault(s_, []).append(p_)
    sigs = [i for i, e in enumerate(ents) if e["sig"]]
    nosig = [i for i in stub if i not in sigs]
    unpaired_sig = [i for i in sigs if i not in stub]
    out += ["## 3.2 Machine-generated entries", "",
            f"v2 counted **{len(sigs)}** by the `NN.0(` volume signature. Re-scan adds a title-similarity match "
            "(smaller title word-set ≥ 80 % contained in the other, same first word of the author field) against the other entries.", "",
            f"- by signature: **{len(sigs)}**; of those, with a duplicate proper entry found by title: {len(sigs) - len(unpaired_sig)}; "
            f"without one found: {len(unpaired_sig)}",
            f"- **additional stubs without the signature: {len(nosig)}**", f"- **full list: {len(set(sigs) | set(nosig))}**", ""]
    rows = []
    for i in sorted(set(sigs) | set(nosig)):
        e = ents[i]
        dup = " | ".join(ents[k]["text"][:100] for k in stub.get(i, [])) or "[no proper duplicate found by title similarity]"
        rows.append((i + 1, "signature" if e["sig"] else "title-similarity", e["text"][:100], dup))
    out += table(["#", "found by", "machine-generated / stub entry", "proper entry it duplicates"], rows) + [""]
    DIS.append(("Machine-generated reference-list entries", "19 (v2 count, by `NN.0(` signature only)",
                f"{len(set(sigs) | set(nosig))} (signature + title similarity; `docs/reference_gaps.md` §3.2)"))
    bad_idx = set(sigs) | set(nosig)
    proper = [e for i, e in enumerate(ents) if i not in bad_idx]

    # ---- 3.1 studies without a properly formatted entry
    def wbtitle(r):
        return str(ws.cell(r, 4).value)
    nopr = []
    for t in t1:
        k = fold(t["authors"].replace(" et al.", ""))
        wr = [r for r in W if fold(ws.cell(r, 2).value).startswith(k) and str(ws.cell(r, 3).value or "") == t["year"]]
        if len(wr) != 1:       # Sahlberg 2023: pick by scope
            wr = [r for r in wr if str(ws.cell(r, 5).value).lower()[:5] == t["scope"].lower()[:5]]
        r = wr[0]
        tw = words(wbtitle(r))
        cand = [e for e in proper if ref_surname(e["text"]) == k]      # same first author AND similar title
        sc = max((contain(tw, e["tw"]) for e in cand), default=0.0)
        best = max(proper, key=lambda e: contain(tw, e["tw"]))
        if sc < 0.8:
            nopr.append((t, r, contain(tw, best["tw"]), best))
    def ent_year(e):
        m = re.search(r"\((\d{4})\)", e["text"])
        return int(m.group(1)) if m else None
    ymis = []
    for t in t1:
        k = fold(t["authors"].replace(" et al.", ""))
        wr = [r for r in W if fold(ws.cell(r, 2).value).startswith(k) and str(ws.cell(r, 3).value or "") == t["year"]]
        if len(wr) != 1:
            wr = [r for r in wr if str(ws.cell(r, 5).value).lower()[:5] == t["scope"].lower()[:5]]
        if not t["year"].isdigit() or not wr:
            continue
        tw = words(wbtitle(wr[0]))
        cand = [e for e in proper if ref_surname(e["text"]) == k]
        if not cand:
            continue
        best = max(cand, key=lambda e: contain(tw, e["tw"]))
        ey = ent_year(best)
        if contain(tw, best["tw"]) >= 0.8 and ey and abs(ey - int(t["year"])) >= 2:
            ymis.append((t, wr[0], best, ey))
    task_eleven = {"1", "7", "11", "16", "34", "37", "39", "51", "67", "69", "71"}
    found = {t["study_no"] for t, *_ in nopr}
    out += ["## 3.1 Studies needing a reference entry written", "",
            f"Method: each corpus study's workbook title is matched to the properly formatted entries by title word overlap "
            f"(≥ 80 % of the shorter word set, 6-letter stems) *and* the same first-author surname; a study with no such entry is listed. **{len(nopr)} studies** result "
            f"(the task says eleven: {', '.join('#' + n for n in sorted(task_eleven, key=int))}).", ""]
    extra = sorted(found - task_eleven, key=int)
    miss = sorted(task_eleven - found, key=int)
    if extra or miss:
        out += [f"- Not in the task's eleven but without a proper entry by title: {', '.join('#' + n for n in extra) or 'none'}. "
                f"In the task's eleven but a proper entry *was* found by title: {', '.join('#' + n for n in miss) or 'none'}.", ""]
        DIS.append(("Studies without a properly formatted reference entry",
                    "11 (TASK v3 §3.1: " + ", ".join("#" + n for n in sorted(task_eleven, key=int)) + ")",
                    f"{len(nopr)} by title match ({', '.join('#' + t['study_no'] for t, *_ in nopr)})"))
    if ymis:
        out += ["Matched on title, but the only proper entry carries a year **two or more years** from the coding (so the entry may "
                "be a different paper, or the coding or the entry is wrong):", "",
                *table(["study", "record", "coded year", "entry year", "entry"],
                       [(f"#{t['study_no']} {t['authors']}", ws.cell(r, 1).value, t["year"], ey, e["text"][:110])
                        for t, r, e, ey in ymis]), ""]
        for t, r, e, ey in ymis:
            DIS.append((f"Year/entry conflict, study #{t['study_no']} {t['authors']} (record {ws.cell(r, 1).value})",
                        f"{t['year']} (Table 1 / workbook)", f"{ey} (title-matched reference-list entry)"))
    snow = [x for x in nopr if x[1] >= 116]
    out += [f"**Snowballed-record group:** {len(snow)} of these {len(nopr)} are snowballed records (workbook rows 116–123: "
            + ", ".join(f"#{t['study_no']}" for t, *_ in snow) + "). The citation layer was never updated when they were added; "
            f"this is a group problem, not {len(snow)} independent omissions.", ""]
    skip = set(refs)
    for t, r, sc, best in nopr:
        rec = ws.cell(r, 1).value
        sur = t["authors"].replace(" et al.", "")
        hits = repo_doi_hits(sur, t["year"], words(wbtitle(r)), skip)
        # study-specific starting points from the task, verified against the named source
        out += [f"### #{t['study_no']} {t['authors']} {t['year']} — record {rec} (workbook row {r})", "",
                f"- first author (as corrected): **{sur}**",
                f"- year: {t['year'] or '[MISSING: year — see review_tables.md §1]'}",
                f"- title (workbook D): {wbtitle(r)}",
                f"- scope: {t['scope']}  /  workbook E: {str(ws.cell(r, 5).value)[:120]}",
                f"- authors (workbook B): {ws.cell(r, 2).value}",
                f"- closest properly formatted entry: {best['text'][:90]}… (title overlap {sc:.2f} — not the same paper)",
                "- journal: [MISSING: journal]", "- volume/issue/pages: [MISSING: volume, issue, pages]", "- DOI: "
                + ("[MISSING: DOI]" if not hits else "")]
        seen = set()
        for d_, src, txt in hits:
            if d_ in seen:
                continue
            seen.add(d_)
            out += [f"  - candidate **{d_}** — found in `{src}`: “{txt}”"]
        out += [""]
    # ---- #58 explicit
    o58 = next((x for x in t1 if x["study_no"] == "58"), None)
    ss = (ROOT / "docs/review/search_strings_stage2.md").read_text(encoding="utf-8")
    ok58 = "10.1016/j.enconman.2018.10.036" in ss
    out += ["### #58 Oyewo 2018 — starting point from the task", "",
            f"DOI `10.1016/j.enconman.2018.10.036`, *Energy Conversion and Management* 178:44–64. Source check: "
            + ("found verbatim in `docs/review/search_strings_stage2.md` (pre-registered positive controls)."
               if ok58 else "**NOT found in `docs/review/search_strings_stage2.md`**."),
            "The properly formatted Oyewo entry in the list (2019, *Solar Energy*) belongs to record 13159 (#59), not to #58.", ""]
    if not ok58:
        DIS.append(("#58 Oyewo DOI source", "docs/review/search_strings_stage2.md (TASK v3 §3.1)", "DOI string not found there"))

    # ---- 3.3 Guta
    gl = [e for e in ents if e["text"].startswith("Guta")]
    out += ["## 3.3 Guta year conflict — OPEN", "",
            "Zeus confirms the corpus year as **2017** (CSV #28 = "
            f"{next(x for x in t1 if x['study_no'] == '28')['year']}; workbook row 53 = {ws.cell(53, 3).value}). "
            "The reference list carries these entries, **both dated 2015**:", ""]
    out += [f"- {e['text']}" for e in gl] + ["",
            "The entry must be brought to 2017 to match the coding, or the discrepancy explained. Nothing changed.", ""]
    DIS.append(("Guta year (#28, record 9844)", "2017 (Zeus; Table 1 CSV and workbook)",
                "2015 (both reference-list entries; proper one has DOI 10.2139/ssrn.2637865)"))

    # ---- 3.4 preprints
    out += ["## 3.4 Preprints — to be identified as such in Table 1", "", "Table 1 not altered.", ""]
    pre = []
    for no, kind in (("4", "SSRN"), ("15", "Research Square"), ("19", "SSRN"), ("28", "SSRN Electronic Journal")):
        t = next(x for x in t1 if x["study_no"] == no)
        k = fold(t["authors"].replace(" et al.", ""))
        r = next(r for r in W if fold(ws.cell(r, 2).value).startswith(k) and (str(ws.cell(r, 3).value or "") == t["year"] or no == "28"))
        ent = next((e["text"][:110] for e in proper if fold(e["text"].split(",")[0]) == k and
                    contain(words(wbtitle(r)), e["tw"]) >= 0.8), "[no proper entry matched]")
        pre.append((f"#{no}", t["authors"], ws.cell(r, 1).value, kind, ent))
    out += table(["study", "authors", "record_id", "venue (TASK)", "reference-list entry"], pre) + [""]

    # ---- 3.5 orphans
    cs = [words(wbtitle(r)) for r in W]
    orph = [e for e in proper if not any(contain(c, e["tw"]) >= 0.8 for c in cs)]
    tal = [e for e in proper if e["text"].startswith("Taliotis") and "(2014)" in e["text"]]
    out += ["## 3.5 Possible orphan citation", ""]
    for e in tal:
        best = max(((contain(c, e["tw"]), r) for c, r in zip(cs, W)), key=lambda x: x[0])
        r = best[1]
        out += [f"- **{e['text']}**",
                f"  - Closest corpus study by title: row {r}, record {ws.cell(r, 1).value}, “{ws.cell(r, 4).value}” "
                f"({ws.cell(r, 3).value}), overlap {best[0]:.2f}. The titles are near-identical, but the coding is the 2016 TEMBA "
                "paper (a different journal and year per TASK v3 §1.5), whose only list entry is machine-generated "
                "(`Taliotis, S., Ramos, H., …`). So the 2014 entry matches no study as coded: **possible orphan citation for "
                "Zeus to check** (cited in the text, or removable), and #70 still needs its own proper entry."]
        UNR.append("Taliotis et al. 2014 (J. Energy in Southern Africa 25(1)): possible orphan; #70's own entry is machine-generated only.")
    if not tal:
        out += ["- No properly formatted Taliotis 2014 entry found."]
    out += [""]
    out += ["Other properly formatted entries matching no corpus study (expected: framework, methods and PRISMA references; "
            "listed so the full picture is visible):", ""]
    out += [f"- {e['text'][:140]}" for e in orph if e not in tal] + [""]
    (ROOT / "docs/reference_gaps.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    UNR.append(f"Reference-list worklist written to docs/reference_gaps.md: {len(nopr)} studies need entries; "
               "journal/volume/pages/DOI are `[MISSING]` wherever the repo does not hold them (nothing invented).")
    return nopr


# ------------------------------------------------------------------ main
def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    t1 = load_csv(REV / "table1_extracted.csv")
    t1_pre = load_csv(PRE / "table1_extracted.csv")
    ws = load_ws(REV / WB)
    ws_pre = load_ws(PRE / WB)
    W = in_scope(ws)
    section1(t1, t1_pre, ws, ws_pre, W)
    exo = sum(1 for r in W if str(ws.cell(r, 21).value).strip().lower().startswith("exo"))
    ab = sum(1 for r in W if str(ws.cell(r, 21).value).strip().lower().startswith("abs"))
    en = sum(1 for r in W if str(ws.cell(r, 21).value).strip().lower().startswith("end"))
    reads = Counter(ws.cell(r, 29).value for r in W)
    enn = sum(1 for r in W if ws.cell(r, 29).value == "both" and not yes(ws.cell(r, 20).value))
    tg = [("n (workbook in-scope rows)", len(W), 75), ("exogenous", exo, 65), ("absent", ab, 10), ("endogenous", en, 0),
          ("reading both", reads["both"], 39), ("reading deliverability-only", reads["deliverability"], 20),
          ("reading access-only", reads["access"], 5), ("reading neither", reads["neither"], 11),
          ("enacted-not-named (reading both AND Names-the-distinction ≠ Y)", enn, 38)]
    emit("### Reconciliation targets (re-checked against the repaired workbook)", "",
         *table(["quantity", "computed", "target", "result"], [(a, b, c, check(a, b, c)) for a, b, c in tg]), "")
    emit("## 2. Artefacts re-generated after the repairs (TASK v3 §4)", "",
         "Both tables are asserted against the **v2 values**. Any difference below is a finding; nothing has been adjusted.", "")
    section2(ws, W)
    section3(t1)
    section_prisma()
    nopr = write_reference_gaps(t1, ws, W)
    emit("## 4. Disagreements (consolidated)", "",
         "Every mismatch found anywhere in this run; both values and their sources. Nothing here has been resolved.", "",
         *table(["#", "item", "value A [source]", "value B [source]"], [(i + 1, a, b, c) for i, (a, b, c) in enumerate(DIS)]), "")
    emit("## 5. Unresolved", "", *[f"- {u}" for u in UNR], "")
    print(f"wrote {OUT} ({len(DIS)} disagreements, {len(UNR)} unresolved); reference_gaps: {len(nopr)} studies need entries")


if __name__ == "__main__":
    sys.exit(main())
