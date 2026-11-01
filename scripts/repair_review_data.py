#!/usr/bin/env python3
"""
Repair the Chapter 2 review data (TASK v2 §1 and TASK v3 §1-§2). Idempotent.

Writes, under data/review/:
  table1_extracted.csv         <- _preimage/v2/table1_extracted.csv + all repairs
  extraction_workbook_v5.xlsx  <- _preimage/v2/extraction_workbook_v5.xlsx + all repairs
  fulltext_screening_sheet.xlsx  (edited IN PLACE: Tally ranges only)
  MANIFEST.sha256              regenerated

The workbook and CSV are always rebuilt from the v2 pre-images (the files as first
delivered), so running this twice yields byte-identical outputs. The workbook is edited
at the XML level so every other cell, formula and cached value is untouched.

v2 repairs  1.1 row-116 author/title swap; 1.2 first-author surname corrections (CSV +
            workbook); 1.5 derivation_tag column AE (+ header comment); 1.6 casing fold.
v3 repairs  1.2 year corrections #15 Dalder 2023, #19 Diesing 2025, #50 Musselman 2022
            (#28 Guta stays 2017); 1.1 Sahlberg pairing #64 -> row 14, #65 -> row 16;
            2.1 snowballed record_id 30523-30525 -> 30533-30535 (workbook A121:A123);
            2.2 Tally ranges -> row 400 on the full-text sheet.
NOT done here (settled by Zeus in the full-text sheet itself, verified by
build_review_tables.py): screening decisions, removal of the four duplicate records.

    python scripts/repair_review_data.py [--no-manifest]
"""
import argparse
import csv
import datetime
import hashlib
import re
import shutil
import sys
import unicodedata
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "data/review"
V2 = REV / "_preimage/v2"
WB = "extraction_workbook_v5.xlsx"
CSVN = "table1_extracted.csv"
FT = "fulltext_screening_sheet.xlsx"
TODAY = "2026-10-13"

AU = {"1": "Fuso Nerini et al.", "2": "Afful-Dadzie et al.", "7": "Kehbila et al.", "16": "Dalla Longa et al.",
      "38": "Kueppers et al.", "47": "Moner-Girona et al.", "51": "Ouedraogo et al.", "61": "Mohd Rafiq et al."}
YEAR_FIX = {"15": ("2024", "2023"), "19": ("2026", "2025"), "50": ("2021", "2022")}   # study -> (from, to)
SAHLBERG = {"64": 14, "65": 16}                  # study -> workbook row (settled, TASK v3 §1.1)
SAHLBERG_REC = {14: "102", 16: "391"}
RECORD_IDS = {121: ("30523", "30533"), 122: ("30524", "30534"), 123: ("30525", "30535")}
HDR = "derivation_tag [from manuscript Table 1]"


def fold(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ CSV
def repair_csv():
    rows = list(csv.DictReader(open(V2 / CSVN, encoding="utf-8", newline="")))
    for r in rows:
        if r["study_no"] in AU:
            r["authors"] = AU[r["study_no"]]
        if r["study_no"] in YEAR_FIX:
            old, new = YEAR_FIX[r["study_no"]]
            assert r["year"] == old, (r["study_no"], r["year"])
            r["year"] = new
        if r["model_class"] == "Geospatial":
            r["model_class"] = "geospatial"
    with open(REV / CSVN, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


# ------------------------------------------------------------------ workbook (XML surgery)
class Sheet:
    def __init__(self, xml, sst):
        self.x, self.sst = xml, sst

    def cell(self, r, col):
        return re.search(r'<c r="%s%d"([^>]*?)(/>|>(.*?)</c>)' % (col, r), self.x, re.S)

    def val(self, r, col):
        m = self.cell(r, col)
        if not m or not m.group(3):
            return None
        a, inner = m.group(1), m.group(3)
        if "inlineStr" in a:
            return re.search(r"<t[^>]*>(.*?)</t>", inner, re.S).group(1).replace("&amp;", "&")
        v = re.search(r"<v>(.*?)</v>", inner)
        if not v:
            return None
        if 't="s"' in a:
            si = self.sst[int(v.group(1))]
            t = "".join(re.findall(r"<t[^>]*>(.*?)</t>", si, re.S))
            return t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
        return v.group(1)

    def set_inline(self, r, col, text):
        m = self.cell(r, col)
        attrs = re.sub(r'\s*t="[^"]*"', "", m.group(1))
        new = f'<c r="{col}{r}"{attrs} t="inlineStr"><is><t xml:space="preserve">{escape(text)}</t></is></c>'
        self.x = self.x[:m.start()] + new + self.x[m.end():]

    def set_number(self, r, col, num):
        m = self.cell(r, col)
        attrs = re.sub(r'\s*t="[^"]*"', "", m.group(1))
        new = f'<c r="{col}{r}"{attrs}><v>{num}</v></c>'
        self.x = self.x[:m.start()] + new + self.x[m.end():]

    def append_cell(self, r, xml):
        m = re.search(r'(<row r="%d"[^>]*>.*?)(</row>)' % r, self.x, re.S)
        self.x = self.x[:m.end(1)] + xml + self.x[m.end(1):]


def repair_workbook(csv_rows):
    zin = zipfile.ZipFile(V2 / WB)
    sst = re.findall(r"<si>(.*?)</si>", zin.read("xl/sharedStrings.xml").decode(), re.S)
    S = Sheet(zin.read("xl/worksheets/sheet1.xml").decode(), sst)
    W = [r for r in range(4, 124)
         if (S.val(r, "U") or "").strip().lower()[:3] in ("exo", "abs", "end")]
    assert len(W) == 75

    # v2 1.1 swap row 116
    b0, d0 = S.val(116, "B"), S.val(116, "D")
    assert b0.startswith("A cost comparison") and d0.startswith("Franceso Fuso")
    mB, mD = S.cell(116, "B"), S.cell(116, "D")
    iB = re.search(r"<v>(\d+)</v>", mB.group(3)).group(1)
    iD = re.search(r"<v>(\d+)</v>", mD.group(3)).group(1)
    S.x = S.x.replace(mB.group(0), mB.group(0).replace(f"<v>{iB}</v>", f"<v>{iD}</v>"), 1)
    mD = S.cell(116, "D")
    S.x = S.x.replace(mD.group(0), mD.group(0).replace(f"<v>{iD}</v>", f"<v>{iB}</v>"), 1)

    # v2 1.2 author names
    S.set_inline(116, "B", "Fuso Nerini, Francesco; Oliver Broad, Dimitris Mentis, Manuel Welsch,Morgan Bazilian")
    S.set_inline(117, "B", "Ouedraogo, Nadia S.")
    S.set_inline(120, "B", "Kehbila, Anderson Gwanyebit; Robert Karisa Masumbuko, Mbeo Ogeya, Philip Osano")
    S.set_inline(122, "B", "Marong et al.")
    for r in W:
        a = S.val(r, "B")
        for old, new in (("affuldadzie ", "afful-dadzie "), ("monergirona ", "moner-girona "),
                         ("kuppers ", "kueppers "), ("rafiq ", "mohd rafiq ")):
            if a.startswith(old):
                S.set_inline(r, "B", new + a[len(old):])

    # v3 1.2 years (numeric C cells), located by record_id
    for no, (old, new) in YEAR_FIX.items():
        rec = {"15": "202", "19": "14486", "50": "16726"}[no]
        r = next(r for r in W if S.val(r, "A") == rec)
        assert S.val(r, "C") == old, (no, r, S.val(r, "C"))
        S.set_number(r, "C", new)
    # v3 2.1 snowballed record ids
    for r, (old, new) in RECORD_IDS.items():
        assert S.val(r, "A") == old, (r, S.val(r, "A"))
        S.set_number(r, "A", new)

    # join Table 1 -> workbook for derivation_tag
    pop, claimed = {}, {}
    for t in csv_rows:
        k = fold(t["authors"].replace(" et al.", ""))
        c = [r for r in W if fold(S.val(r, "B")).startswith(k) and (S.val(r, "C") or "") == t["year"]]
        if t["study_no"] in SAHLBERG:                       # settled pairing, TASK v3 §1.1
            r = SAHLBERG[t["study_no"]]
            assert r in c and S.val(r, "A") == SAHLBERG_REC[r], (t["study_no"], c)
            c = [r]
        if len(c) == 1:
            claimed.setdefault(c[0], []).append(t)
    for r, ts in claimed.items():
        assert len(ts) == 1, (r, [x["study_no"] for x in ts])
        pop[r] = ts[0]["derivation_tag"]

    # column AE
    S.x = S.x.replace('<dimension ref="A1:AD144"/>', '<dimension ref="A1:AE144"/>')
    S.x = S.x.replace('<col min="30" max="30" width="28" customWidth="1"/></cols>',
                      '<col min="30" max="30" width="28" customWidth="1"/><col min="31" max="31" width="36" customWidth="1"/></cols>')
    S.x = S.x.replace('spans="1:30"', 'spans="1:31"')
    S.append_cell(2, f'<c r="AE2" s="14" t="inlineStr"><is><t>{escape(HDR)}</t></is></c>')
    for r in range(4, 145):
        if r in pop:
            S.append_cell(r, f'<c r="AE{r}" s="19" t="inlineStr"><is><t xml:space="preserve">{escape(pop[r])}</t></is></c>')

    cm = zin.read("xl/comments1.xml").decode()
    ctext = ("Source: Table 1 of manuscript_final.docx (derivation tag, 'Derivation (SQ7)' column), joined on first-author "
             "surname + year via data/review/table1_extracted.csv; Sahlberg 2023 rows paired by Zeus decision "
             f"(row 14 = #64, row 16 = #65). Written {TODAY}. Verbatim; '—' = absent-financing studies. "
             "Blank = not a corpus row.")
    cm = cm.replace("</commentList>", '<comment ref="AE2" authorId="0" shapeId="0"><text><r><rPr><sz val="10"/>'
                    f'<rFont val="Arial"/><family val="2"/></rPr><t>{escape(ctext)}</t></r></text></comment></commentList>')
    vml = zin.read("xl/drawings/vmlDrawing1.vml").decode()
    shape = '''<v:shape id="_x0000_s1027" type="#_x0000_t202" style='position:absolute;
  margin-left:57.6pt;margin-top:1.2pt;width:114pt;height:100pt;z-index:3;
  visibility:hidden;mso-wrap-style:none;v-text-anchor:middle' o:allowincell="f"
  fillcolor="#ffffe1">
  <v:fill color2="#00001e" o:detectmouseclick="t"/>
  <v:stroke startarrow="block"/>
  <v:shadow on="t" color="black" obscured="t"/>
  <v:textbox>
   <div style='text-align:left'></div>
  </v:textbox>
  <x:ClientData ObjectType="Note">
   <x:MoveWithCells/>
   <x:SizeWithCells/>
   <x:Anchor>
    1, 12, 0, 1, 2, 5, 3, 76</x:Anchor>
   <x:AutoFill>False</x:AutoFill>
   <x:Row>1</x:Row>
   <x:Column>30</x:Column>
  </x:ClientData>
 </v:shape>'''
    vml = vml.replace("</xml>", shape + "</xml>")
    swap = {"xl/worksheets/sheet1.xml": S.x.encode(), "xl/comments1.xml": cm.encode(),
            "xl/drawings/vmlDrawing1.vml": vml.encode()}
    write_zip(zin, REV / WB, swap)
    return len(pop)


def write_zip(zin, out, replace):
    tmp = Path(str(out) + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
        for i in zin.infolist():
            zo.writestr(i, replace.get(i.filename, zin.read(i.filename)))
    shutil.move(str(tmp), str(out))


# ------------------------------------------------------------------ full-text sheet Tally (in place)
TALLY_LAST = 400
FTS = "'Full-text screening'"


def repair_tally():
    path = REV / FT
    zin = zipfile.ZipFile(path)
    sheet1 = zin.read("xl/worksheets/sheet1.xml").decode()
    sst = re.findall(r"<si>(.*?)</si>", zin.read("xl/sharedStrings.xml").decode(), re.S)
    S1 = Sheet(sheet1, sst)
    rows = {}
    for r in range(2, TALLY_LAST + 1):
        rows[r] = {c: S1.val(r, c) for c in "BFHJ"}
    B = [x["B"] for x in rows.values() if x["B"] not in (None, "")]
    F = [x["F"] for x in rows.values() if x["B"] not in (None, "")]
    n = len(B)
    vals = dict(B4=n, B5=sum(1 for v in F if v == "Include"), B6=sum(1 for v in F if v == "Exclude"),
                B7=sum(1 for v in F if v in (None, "")),
                B8=sum(1 for x in rows.values() if x["H"] not in (None, "")),
                B9=sum(1 for x in rows.values() if x["J"] == "Y"),
                B10=sum(1 for x in rows.values() if x["B"] not in (None, "") and x["J"] != "Y"))
    L = TALLY_LAST
    # B7 and B10 count blanks / non-Y: guarded by record_id so the empty rows 189..400 are not counted
    f = {"B4": f"COUNTA({FTS}!$B$2:$B${L})",
         "B5": f'COUNTIF({FTS}!$F$2:$F${L},"Include")',
         "B6": f'COUNTIF({FTS}!$F$2:$F${L},"Exclude")',
         "B7": f'SUMPRODUCT(--({FTS}!$B$2:$B${L}&lt;&gt;""),--({FTS}!$F$2:$F${L}=""))',
         "B8": f'SUMPRODUCT(--({FTS}!$H$2:$H${L}&lt;&gt;""))',
         "B9": f'COUNTIF({FTS}!$J$2:$J${L},"Y")',
         "B10": f'SUMPRODUCT(--({FTS}!$B$2:$B${L}&lt;&gt;""),--({FTS}!$J$2:$J${L}&lt;&gt;"Y"))'}
    x = zin.read("xl/worksheets/sheet2.xml").decode()
    for ref, formula in f.items():
        m = re.search(r'<c r="%s"([^>]*)><f([^>]*)>(.*?)</f><v>(.*?)</v></c>' % ref, x, re.S)
        x = x[:m.start()] + f'<c r="{ref}"{m.group(1)}><f{m.group(2)}>{formula}</f><v>{vals[ref]}</v></c>' + x[m.end():]
    write_zip(zin, path, {"xl/worksheets/sheet2.xml": x.encode()})
    return vals


# ------------------------------------------------------------------ manifest
def regen_manifest():
    p = REV / "MANIFEST.sha256"
    old = p.read_text()
    paths = [m.group(2) for m in re.finditer(r"^([0-9a-f]{64})  (\S+)$", old, re.M)]
    csvp = "data/review/table1_extracted.csv"
    if csvp not in paths:
        paths.insert(paths.index("data/review/extraction_workbook_v5.xlsx") + 1, csvp)
    hdr = old.split("\n\n")[0].split("\n")
    hdr[1] = "# Generated: " + datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    p.write_text("\n".join(hdr) + "\n\n" + "\n".join(f"{sha(ROOT / q)}  {q}" for q in paths) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-manifest", action="store_true", help="do not regenerate MANIFEST.sha256")
    args = ap.parse_args()
    if not (V2 / WB).exists() or not (V2 / CSVN).exists():
        sys.exit(f"missing v2 pre-images in {V2}")
    rows = repair_csv()
    npop = repair_workbook(rows)
    tally = repair_tally()
    if not args.no_manifest:
        regen_manifest()
    print(f"CSV rewritten; workbook rewritten (derivation_tag populated: {npop}); Tally -> {tally}")


if __name__ == "__main__":
    sys.exit(main())
