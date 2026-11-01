"""TASK 58 measurement script. Read-only on every chapter and appendix file.

Usage: python scripts/task58_measure.py {sizes,markers,dup,inventory} [chapter.md]
Defaults to docs/chapters/CHAPTER_3_assembled.md.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CH = ROOT / "docs/chapters/CHAPTER_3_assembled.md"
APP = {k: ROOT / f"docs/chapters/appendix_{k.lower()}.md" for k in "ABCD"}

HEAD = re.compile(r"^(#{1,4}) (.*)$")
SECNUM = re.compile(r"^(\d+(?:\.\d+)*)\b")


def load(path=CH):
    return Path(path).read_text(encoding="utf-8").split("\n")


def classify(lines):
    """Per line: (kind, section_id, heading_level). kind in
    heading|table|eq|code|prose|blank. section_id is the ### id, '3.N intro'
    for text under a ## before its first ###, 'REFS' for the reference list,
    'TITLE' for the chapter title."""
    out = []
    sec = "TITLE"
    in_eq = False
    in_code = False
    for i, ln in enumerate(lines):
        m = HEAD.match(ln)
        kind = None
        if not in_eq and not in_code and m:
            lvl = len(m.group(1))
            txt = m.group(2)
            n = SECNUM.match(txt)
            if lvl == 2:
                if txt.startswith("References"):
                    sec = "REFS"
                elif n:
                    sec = n.group(1) + " intro"
            elif lvl == 3 and n:
                sec = n.group(1)
            kind = "heading"
        elif ln.strip().startswith("```"):
            in_code = not in_code
            kind = "code"
        elif in_code:
            kind = "code"
        elif ln.strip() == "$$":
            in_eq = not in_eq
            kind = "eq"
        elif in_eq:
            kind = "eq"
        elif ln.startswith("|"):
            kind = "table"
        elif ln.strip() == "":
            kind = "blank"
        else:
            kind = "prose"
        out.append((kind, sec))
    return out


def words(s):
    return len(s.split())


def paragraphs(lines, cls):
    """Blank-line-separated blocks of prose lines (tables/eq/headings split
    blocks). Returns (start_line, end_line, section, text) 1-indexed."""
    paras = []
    cur = []
    for i, ln in enumerate(lines):
        k, sec = cls[i]
        if k == "prose":
            cur.append((i, ln, sec))
        else:
            if cur:
                paras.append(cur)
            cur = []
    if cur:
        paras.append(cur)
    return [(p[0][0] + 1, p[-1][0] + 1, p[0][2], " ".join(x[1] for x in p)) for p in paras]


def body_end(lines):
    for i, ln in enumerate(lines):
        if ln.startswith("## References"):
            return i
    return len(lines)


# ---------------------------------------------------------------- sizes
def cmd_sizes(lines):
    cls = classify(lines)
    be = body_end(lines)
    # unit = heading line index range for each ## and ### heading
    heads = [(i, len(HEAD.match(l).group(1)), HEAD.match(l).group(2))
             for i, l in enumerate(lines) if cls[i][0] == "heading" and HEAD.match(l)]
    # (words in a heading's own span: from the heading to the next heading of any level 2-4... ###-level own text
    # includes its #### children, ## own text is the intro before first ###)
    def span_counts(a, b):
        c = {"prose": 0, "table": 0, "eq": 0, "code": 0, "heading": 0}
        for j in range(a, b):
            k = cls[j][0]
            if k in c:
                c[k] += words(lines[j])
        return c

    total = sum(words(l) for l in lines)
    ref_words = sum(words(l) for l in lines[be:])
    rows = []
    h23 = [(i, lvl, t) for i, lvl, t in heads if lvl in (2, 3)]
    for idx, (i, lvl, t) in enumerate(h23):
        end = h23[idx + 1][0] if idx + 1 < len(h23) else len(lines)
        own = span_counts(i, end)
        rows.append((lvl, t, i + 1, end, own))
    # rollup for ## = until next ##
    h2 = [(i, t) for i, lvl, t in h23 if lvl == 2]
    roll = {}
    for idx, (i, t) in enumerate(h2):
        end = h2[idx + 1][0] if idx + 1 < len(h2) else len(lines)
        roll[t] = span_counts(i, end)
    return total, ref_words, rows, roll


def print_sizes(lines):
    total, ref_words, rows, roll = cmd_sizes(lines)
    print(f"TOTAL words (wc -w basis): {total}; reference list: {ref_words}; excluding reference list: {total - ref_words}")
    body = total - ref_words
    print("\n## rollups")
    print("| section | prose | table | equation | heading | total | share of body |")
    print("|---|---|---|---|---|---|---|")
    for t, c in roll.items():
        tot = sum(c.values())
        print(f"| {t} | {c['prose']} | {c['table']} | {c['eq']} | {c['heading']} | {tot} | {100*tot/body:.1f}% |")
    print("\n## own spans (### includes its #### children; ## row = intro text before first ###)")
    print("| level | section | lines | prose | table | equation | heading | total | share of body |")
    print("|---|---|---|---|---|---|---|---|---|")
    for lvl, t, a, b, c in rows:
        tot = sum(c.values())
        sh = "" if t.startswith("References") else f"{100*tot/body:.1f}%"
        print(f"| {'##' if lvl==2 else '###'} | {t} | {a}-{b-1} | {c['prose']} | {c['table']} | {c['eq']} | {c['heading']} | {tot} | {sh} |")


# ---------------------------------------------------------------- markers
MARK = re.compile(
    r"an earlier version|an earlier state|an earlier specification|an earlier finding|an earlier unsourced|earlier working|"
    r"previously|superseded|retired|a predecessor|its predecessor|was corrected|were corrected|has been corrected|is replaced by|"
    r"is now |are now |no longer|the former|pre-correction|was replaced|were replaced|inherited|"
    r"has been withdrawn|is withdrawn|was withdrawn|did not hold|neither bound holds|"
    r"commit `|\bTASK\b|register entry|\(Appendix [BC], L\d+|Appendix C|re-measured|restored|was removed|was deleted|"
    r"until \d{4}-\d{2}-\d{2}", re.I)
EQ_IN_PARA = re.compile(r"\$[^$]+\$|\(3\.\d+[a-z]?\)")
CITE = re.compile(r"\b[A-Z][A-Za-z'’\-]+(?: et al\.| and [A-Z][A-Za-z'’\-]+)?,? \(?\d{4}\b|\([A-Z][^()]*\d{4}")
REG = re.compile(r"Appendix [A-D][^)]{0,40}|register L\d+|\bL\d{1,3}\b")
VAL = re.compile(r"⟦")
MKR = re.compile(r"\[(?:VERIFY|REF|TITLE|VINTAGE|CONFIRM|PAGE|UNVERIFIED|REVISION YEAR)")


def print_markers(lines):
    cls = classify(lines)
    be = body_end(lines)
    paras = [p for p in paragraphs(lines, cls) if p[1] <= be]
    print("| # | section | lines | words | markers | eq | cite | reg | ⟦⟧ | bracket-marker |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    n = 0
    tot = 0
    for a, b, sec, txt in paras:
        hits = sorted({m.group(0).lower() for m in MARK.finditer(txt)})
        if not hits:
            continue
        n += 1
        w = words(txt)
        tot += w
        print(f"| {n} | {sec} | {a}-{b} | {w} | {'; '.join(hits)} | "
              f"{'Y' if EQ_IN_PARA.search(txt) else ''} | {'Y' if CITE.search(txt) else ''} | "
              f"{'Y' if REG.search(txt) else ''} | {'Y' if VAL.search(txt) else ''} | {'Y' if MKR.search(txt) else ''} |")
    print(f"\ncandidate paragraphs: {n}, words: {tot}")


# ---------------------------------------------------------------- duplication
def norm_words(text):
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"[*_#>]", " ", text)
    text = re.sub(r"[^\w\s%.\-]", " ", text.lower())
    text = text.replace(".", " ").replace("-", " ")
    return text.split()


def unit_streams(lines, body_only=True):
    cls = classify(lines)
    be = body_end(lines)
    units = defaultdict(list)
    for i, ln in enumerate(lines[:be] if body_only else lines):
        k, sec = cls[i]
        if k == "prose":
            units[sec].append(ln)
    return {s: norm_words(" ".join(v)) for s, v in units.items()}


def shared_runs(a, b, n=12):
    """Maximal runs of >= n consecutive words shared by a and b (as a run in
    a that also occurs in b). Returns list of (start_in_a, length)."""
    shb = defaultdict(list)
    for j in range(len(b) - n + 1):
        shb[tuple(b[j:j + n])].append(j)
    hit = [i for i in range(len(a) - n + 1) if tuple(a[i:i + n]) in shb]
    runs = []
    if not hit:
        return runs
    s = hit[0]
    p = hit[0]
    for h in hit[1:]:
        if h == p + 1:
            p = h
        else:
            runs.append((s, p - s + n))
            s = p = h
    runs.append((s, p - s + n))
    return runs


def secsort(s):
    m = SECNUM.match(s)
    return tuple(int(x) for x in m.group(1).split(".")) + ((0,) if "intro" in s else (1,)) if m else (99,)


def print_dup(lines, threshold=3):
    st = unit_streams(lines)
    secs = sorted([s for s in st if s not in ("TITLE", "REFS")], key=secsort)
    res = []
    for ai in range(len(secs)):
        for bi in range(ai + 1, len(secs)):
            A, B = secs[ai], secs[bi]
            runs = shared_runs(st[A], st[B])
            if runs:
                res.append((A, B, runs))
    print("| pair | shared 12-word runs | words in runs (in first section) | longest run |")
    print("|---|---|---|---|")
    for A, B, runs in sorted(res, key=lambda r: -len(r[2])):
        if len(runs) > threshold:
            print(f"| §{A} / §{B} | {len(runs)} | {sum(r[1] for r in runs)} | {max(r[1] for r in runs)} |")
    print("\nAll pairs with >=1 run:")
    for A, B, runs in sorted(res, key=lambda r: -len(r[2])):
        print(f"{A} / {B}: {len(runs)} runs, {sum(r[1] for r in runs)} words")
    return st, res


def dump_runs(lines, A, B):
    st = unit_streams(lines)
    for s, l in shared_runs(st[A], st[B]):
        print(f"[{A} @{s} len {l}] " + " ".join(st[A][s:s + l]))


# ---------------------------------------------------------------- inventory
def sec_at_positions(text, cls_lines):
    """Map char offset -> section id."""
    offs = []
    pos = 0
    for i, ln in enumerate(text.split("\n")):
        offs.append((pos, cls_lines[i][1], i + 1))
        pos += len(ln) + 1
    def f(p):
        lo, hi = 0, len(offs) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if offs[mid][0] <= p:
                lo = mid
            else:
                hi = mid - 1
        return offs[lo][1], offs[lo][2]
    return f


def flat(s):
    return re.sub(r"\s+", " ", s).strip()


AUTHORS = ("Federal Ministry of Power|Federal Republic of Nigeria|Gas Turbine World|Climate Policy Initiative|NNPC Limited|World Bank|"
           "Birge|Bynum|Chankong|Cohon|Das|Ehrgott|Ember|Fisher|Forrest|Grubb|Haimes|Hart|Huangfu|IEA|IRENA|Keane|Kotzur|Liddle|"
           "Madansky|Mavrotas|Mirletz|NERC|NREL|NUPRC|OECD|Oluwole|Oree|Poncelet|Ringkj\u00f8b|Rockafellar|Saltelli|Sadorsky|"
           "Schr\u00f6der|Shapley|Short|Sobol'|Virtanen|Yescombe|Zhuang|Lougee-Heimer|Louveaux|Uryasev")
CITE_RE = re.compile(
    r"(?P<a>(?:" + AUTHORS.replace(" ", r"\s+") + r")(?:\s+et\s+al\.|\s+and\s+[A-Z][A-Za-z'\u2019\-\u00C0-\u017F]+)?)"
    r",?\s+\(?(?P<y>\d{4})[a-z]?\b")


def inventory(lines):
    text = "\n".join(lines)
    cls = classify(lines)
    be = body_end(lines)
    body_off = len("\n".join(lines[:be]))
    be_off = body_off
    where = sec_at_positions(text, cls)
    out = []

    def add(kind, pos, val):
        sec, ln = where(pos)
        out.append((pos, kind, sec, ln, flat(val)))

    # 1 equation tags (body)
    for m in re.finditer(r"\\tag\{([^}]*)\}", text):
        add("EQ", m.start(), m.group(1))
    # 2 inline citations (prose + tables, body only; strip code spans and equations)
    mask = re.sub(r"`[^`]*`", lambda m: " " * len(m.group(0)), text[:body_off])
    mask = re.sub(r"\$\$.*?\$\$", lambda m: " " * len(m.group(0)), mask, flags=re.S)
    for m in CITE_RE.finditer(mask):
        a = m.group("a")
        add("CITE", m.start(), f"{flat(a)}, {m.group('y')}")
    # 3 reference-list entries
    for i in range(be, len(lines)):
        if lines[i].startswith("- "):
            add_pos = len("\n".join(lines[:i]))
            key = re.sub(r"\s+", " ", lines[i][2:])
            trimmed = re.sub(r"\s*`\[[^\]]*\]`.*$", "", key)
            add("REF", add_pos, (trimmed or key)[:110])
    # 4 section cross refs
    for m in re.finditer(r"(?:§|Section |Sections )(3\.\d+(?:\.\d+)?)", mask):
        add("XREF", m.start(), "§" + m.group(1))
    for m in re.finditer(r"Sections (3\.\d) to (3\.\d)", mask):
        add("XREF", m.start(), f"§{m.group(1)}-§{m.group(2)}")
    # 5 value brackets
    for m in re.finditer(r"⟦(.*?)⟧", text[:body_off], flags=re.S):
        add("VAL", m.start(), m.group(1) if len(m.group(1)) < 70 else m.group(1)[:60] + "...")
    # 6 register references
    for m in re.finditer(r"Appendix [A-D]", mask):
        tail = mask[m.end(): m.end() + 70]
        tail = re.split(r"\)|Appendix", tail)[0]
        tail = flat(tail)
        ids = re.findall(r"L\d+|A\d+|rows? \d+", tail)
        if ids:
            for x in ids:
                add("REG", m.start(), f"{m.group(0)}, {x}")
        elif "L[VERIFY" in tail:
            add("REG", m.start(), f"{m.group(0)}, L[VERIFY]")
        else:
            add("REG", m.start(), f"{m.group(0)} (no locator)")
    for m in re.finditer(r"register L\d+", text):
        add("REG", m.start(), m.group(0))
    # 7 markers
    first_bullet = len("\n".join(lines[:next(i for i in range(be, len(lines)) if lines[i].startswith("- "))]))
    for m in re.finditer(r"\[(VERIFY|REF|TITLE|VINTAGE|CONFIRM|PAGE|UNVERIFIED|REVISION YEAR)(?:[^\]]*)\]", text, flags=re.S):
        if be_off <= m.start() < first_bullet:
            continue
        v = flat(m.group(0))
        add("MARK:" + m.group(1), m.start(), v[:80])
    for m in re.finditer(r"\[Code to confirm[^\]]*\]", text):
        add("MARK:CODE-TO-CONFIRM", m.start(), flat(m.group(0))[:80])
    # 8 code anchors
    for m in re.finditer(r"\[code: (.*?)\]", text[:body_off], flags=re.S):
        add("CODE", m.start(), m.group(1))
    for m in re.finditer(r"\[code: VERIFY", text):
        add("MARK:VERIFY", m.start(), "[code: VERIFY ...]")
    out.sort()
    return out


def print_inventory(lines):
    inv = inventory(lines)
    print("# TASK 58 baseline inventory, Chapter 3\n")
    print("Machine-comparable: one element per line, `KIND | section | element`. "
          "Generated by `scripts/task58_measure.py inventory`. Section = the ### unit "
          "(`3.N intro` = text under a ## before its first ###; `REFS` = reference list). "
          "Document order. A diff should compare the multiset of `KIND | element` for "
          "disappearance and `KIND | section | element` for relocation.\n")
    counts = defaultdict(int)
    for _, k, s, ln, v in inv:
        counts[k] += 1
    print("## Counts\n")
    for k in sorted(counts):
        print(f"- {k}: {counts[k]}")
    print("\n```")
    for _, k, s, ln, v in inv:
        print(f"{k} | {s} | {v}")
    print("```")


if __name__ == "__main__":
    cmd = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].endswith(".md") else CH
    L = load(path)
    if cmd in ("sec", "vals"):
        pass
    elif cmd == "sizes":
        print_sizes(L)
    elif cmd == "markers":
        print_markers(L)
    elif cmd == "dup":
        print_dup(L)
    elif cmd == "runs":
        dump_runs(L, sys.argv[3], sys.argv[4])
    elif cmd == "inventory":
        print_inventory(L)


# ---------------------------------------------------------------- secondary duplication measures (TASK 58 §4)
def norm_words_keepmath(text):
    text = re.sub(r"`[^`]*`", " CODE ", text)
    text = re.sub(r"\$[^$]*\$", " MATH ", text)
    text = re.sub(r"[*_#>]", " ", text)
    text = re.sub(r"[^\w\s%.\-]", " ", text.lower())
    text = text.replace(".", " ").replace("-", " ")
    return text.split()


def secondary(lines, n):
    cls = classify(lines)
    be = body_end(lines)
    units = defaultdict(list)
    for i, ln in enumerate(lines[:be]):
        k, sec = cls[i]
        if k == "prose":
            units[sec].append(ln)
    st = {s: norm_words_keepmath(" ".join(v)) for s, v in units.items() if s != "TITLE"}
    secs = sorted(st, key=secsort)
    print(f"-- shared runs of >= {n} words (math/code spans kept as one token each)")
    for ai in range(len(secs)):
        for bi in range(ai + 1, len(secs)):
            runs = shared_runs(st[secs[ai]], st[secs[bi]], n)
            if runs:
                print(f"{secs[ai]} / {secs[bi]}: {len(runs)} runs, longest {max(r[1] for r in runs)}")
                if "-v" in sys.argv:
                    for s, l in runs:
                        print("    " + " ".join(st[secs[ai]][s:s + l]))


def shared_values(lines):
    cls = classify(lines)
    text = "\n".join(lines[:body_end(lines)])
    where = sec_at_positions("\n".join(lines), cls)
    byval = defaultdict(set)
    for m in re.finditer(r"⟦(.*?)⟧", text, flags=re.S):
        v = flat(m.group(1))
        if len(v) > 40 or not re.search(r"\d", v):
            continue
        byval[v].add(where(m.start())[0])
    print("-- ⟦values⟧ (digits, <=40 chars) that occur in more than one section")
    for v, s in sorted(byval.items(), key=lambda kv: secsort(sorted(kv[1], key=secsort)[0])):
        if len(s) > 1:
            print(f"⟦{v}⟧: " + ", ".join("§" + x for x in sorted(s, key=secsort)))


if __name__ == "__main__" and sys.argv[1] in ("sec", "vals"):
    L = load(CH)
    if sys.argv[1] == "sec":
        secondary(L, int(sys.argv[2]))
    else:
        shared_values(L)
