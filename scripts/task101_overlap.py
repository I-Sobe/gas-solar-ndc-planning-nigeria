"""TASK 101 measurement script. Read-only on every chapter file.

Loads Chapter 3 and the other chapters as paragraphs tagged by section unit and
computes, per (Chapter 3 paragraph, other-chapter paragraph) pair, the shared
N-word runs, shared bracketed values, shared named quantities and shared
section references.  Usage: python scripts/task101_overlap.py <cmd>
Chapter 2 is a .docx; its text is extracted to a scratch file first
(pass the path with CH2=<path>).
"""
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CH = ROOT / "docs/chapters"
FILES = {
    "ch1": CH / "Chapter1.md",
    "ch3": CH / "CHAPTER_3_assembled.md",
    "ch4": CH / "CHAPTER_4_submission_draft.md",
    "ch5": CH / "CHAPTER_5_draft.md",
    "abs": CH / "Abstract_1.md",
}
CH2 = Path(os.environ.get("CH2", "/nonexistent"))

HEAD = re.compile(r"^(#{1,4}) (.*)$")
NUM = re.compile(r"^(\d+(?:\.\d+)*)\b")
VAL = re.compile(r"⟦([^⟧]*)⟧")
SECREF = re.compile(r"§\s?(\d(?:\.\d+){0,2})")


def words(s):
    return len(s.split())


def load_doc(key):
    """Return list of dict(sec, line0, line1, kind, text) for prose paragraphs
    and (kind=table/eq) blocks, 1-indexed lines.  sec = ### unit id (or ## id
    where no ### exists, or 'intro')."""
    if key == "ch2":
        lines = CH2.read_text(encoding="utf-8").split("\n")
        stop = next((i for i, l in enumerate(lines) if l.startswith("Notes for Zeus")), len(lines))
        lines = lines[:stop]
        md = False
    else:
        lines = FILES[key].read_text(encoding="utf-8").split("\n")
        md = True
    out = []
    sec = "front"
    in_eq = False
    cur = []
    def flush():
        nonlocal cur
        if cur:
            out.append(dict(sec=sec, line0=cur[0][0], line1=cur[-1][0], kind="prose",
                            text=" ".join(l for _, l in cur)))
            cur = []
    for i, ln in enumerate(lines, 1):
        if md:
            m = HEAD.match(ln)
        else:
            m = re.match(r"^()(\d\.\d+(?:\.\d+)? .*)$", ln)
            if m:
                m = re.match(r"^()((?:\d\.\d+(?:\.\d+)?) .*)$", ln)
        if not in_eq and m:
            flush()
            txt = m.group(2)
            n = NUM.match(txt)
            if n:
                sec = n.group(1)
            continue
        if ln.strip() == "$$":
            flush(); in_eq = not in_eq; continue
        if in_eq:
            continue
        if ln.startswith("|") or ln.strip() == "---" or ln.strip() == "":
            flush(); continue
        cur.append((i, ln))
    flush()
    if key == "abs":
        for p in out:
            p["sec"] = "abs"
    return out


STRIP = [re.compile(r"\$\$.*?\$\$", re.S), re.compile(r"\$[^$]*\$"), re.compile(r"`[^`]*`"),
         re.compile(r"\[code:[^\]]*\]"), re.compile(r"\[VERIFY[^\]]*\]")]


def norm_tokens(text):
    for r in STRIP:
        text = r.sub(" ", text)
    text = VAL.sub(lambda m: " " + m.group(1) + " ", text)
    text = text.lower()
    text = re.sub(r"[^\w%.\-×⁻⁰¹²³⁴⁵⁶⁷⁸⁹ ]", " ", text)
    toks = [t.strip(".-") for t in text.split()]
    return [t for t in toks if t]


def shared_runs(a, b, n):
    """Maximal runs of >= n identical consecutive tokens common to token lists a, b."""
    if len(a) < n or len(b) < n:
        return []
    idx = defaultdict(list)
    for i in range(len(a) - n + 1):
        idx[tuple(a[i:i + n])].append(i)
    hits = []
    for j in range(len(b) - n + 1):
        for i in idx.get(tuple(b[j:j + n]), ()):
            hits.append((i, j))
    runs = []
    hs = set(hits)
    for (i, j) in sorted(hits):
        if (i - 1, j - 1) in hs:
            continue
        k = 0
        while (i + k, j + k) in hs:
            k += 1
        runs.append((i, j, n + k - 1))
    return runs


def vals(text):
    return [v.strip() for v in VAL.findall(text)]


def substantive(v):
    """A value worth matching: carries a decimal point, a percent sign, a
    superscript exponent, or four or more digits."""
    digs = re.sub(r"\D", "", v)
    return ("." in v) or ("%" in v) or ("⁻" in v) or len(digs) >= 4 and not re.fullmatch(r"(19|20)\d\d", v)


QUANT = {
    "gas_avail_observed": r"gas_avail_observed",
    "gas_avail_nameplate": r"gas_avail_nameplate",
    "cap_fraction": r"cap[_ ]fraction",
    "B*": r"\$?B\^?\{?\*\}?\$?|B\\\*|\bB\*",
    "tier:tight": r"`?\btight\b`?",
    "tier:moderate": r"`?\bmoderate\b`?",
    "tier:adequacy": r"`?\badequacy\b`?",
    "tier:expansion": r"`?\bexpansion\b`?",
    "tier:unconstrained": r"`?\bunconstrained\b`?",
    "hydro_committed": r"hydro_committed|hydro_ndc",
    "nameplate": r"\bnameplate\b",
    "measured/observed arm": r"measured availability|observed arm|declared lower bound",
    "conditional cap": r"conditional (?:arm|cap|stringency)|NDC 3\.0|NDC 2\.0",
    "unconditional cap": r"unconditional",
    "leakage": r"leakage",
    "lambda": r"\\lambda|out-of-boundary share",
    "frontier": r"frontier",
    "non-dominated": r"non-dominated|dominated",
    "turning point": r"turning point",
    "VSS": r"value of the stochastic solution|\bVSS\b|wait-and-see",
    "Sobol": r"Sobol",
    "shadow price/dual": r"shadow price|\bdual\b",
    "capital recovery": r"capital[- ]recovery",
    "conversion": r"conversion",
    "service-financed": r"service-financed",
    "concessional": r"concessional",
    "end-effect buffer": r"end-effect|buffer",
    "reporting window": r"reporting window",
    "firm capacity": r"firm[- ]capacity",
    "perfect foresight": r"perfect foresight",
    "vintage dispatch": r"vintage",
    "blended": r"blended",
}
QRE = {k: re.compile(v, re.I if k not in ("B*",) else 0) for k, v in QUANT.items()}


def quants(text):
    return {k for k, r in QRE.items() if r.search(text)}


def secrefs(text):
    return SECREF.findall(text)


def load_all():
    D = {}
    keys = list(FILES) + (["ch2"] if CH2.exists() else [])
    for k in keys:
        D[k] = load_doc(k)
    return D


def cmd_pairs(argv):
    D = load_all()
    n = int(argv[0]) if argv else 12
    c3 = D["ch3"]
    for p in c3:
        p["tok"] = norm_tokens(p["text"])
        p["val"] = {v for v in vals(p["text"]) if substantive(v)}
    rows = []
    for k, doc in D.items():
        if k == "ch3":
            continue
        for q in doc:
            q["tok"] = norm_tokens(q["text"])
            q["val"] = {v for v in vals(q["text"]) if substantive(v)}
            for p in c3:
                runs = shared_runs(p["tok"], q["tok"], n)
                sv = p["val"] & q["val"]
                if runs or len(sv) >= 2:
                    rows.append((k, q["sec"], q["line0"], q["line1"], p["sec"], p["line0"], p["line1"],
                                 len(runs), sum(r[2] for r in runs), sorted(sv)))
    for r in rows:
        print("\t".join(str(x) for x in r))




def numval(v):
    s = v.replace(",", "").replace("%", "").replace("+", "").replace("−", "-").strip()
    try:
        return float(s)
    except ValueError:
        return None


def cmd_vals(argv):
    """Per Chapter 3 section: every substantive bracketed value, and each other
    chapter section carrying the same string, or the same number after a
    scale of 10^3/10^6/10^9 (23.323 against 23,323,068,340.46) or a percent/
    fraction change.  Output: ch3 section, value, other chapter, section."""
    D = load_all()
    idx = defaultdict(list)
    num = []
    for k, doc in D.items():
        if k == "ch3":
            continue
        for q in doc:
            for v in vals(q["text"]):
                if substantive(v):
                    idx[v].append((k, q["sec"], q["line0"]))
                    x = numval(v)
                    if x:
                        num.append((x, v, k, q["sec"], q["line0"]))
    for p in D["ch3"]:
        for v in vals(p["text"]):
            if not substantive(v):
                continue
            for h in idx.get(v, ()):
                print("EXACT", p["sec"], p["line0"], v, *h, sep="\t")
            x = numval(v)
            if x and abs(x) >= 1 and abs(x) < 1000:
                for (y, w, k, sec, ln) in num:
                    if w == v:
                        continue
                    for sc in (1e3, 1e6, 1e9):
                        if abs(y) > 1e4 and abs(abs(y) / sc - abs(x)) < 0.0005 * abs(x) + 1e-9:
                            print("SCALED", p["sec"], p["line0"], v, k, sec, ln, w, sep="\t")


def cmd_quants(argv):
    D = load_all()
    for k, doc in D.items():
        for q in doc:
            print(k, q["sec"], q["line0"], q["line1"], ";".join(sorted(quants(q["text"]))), sep="\t")


def cmd_refs(argv):
    """Section references from each other chapter into Chapter 3, and
    references from Chapter 3 into Chapters 1, 2, 4, 5."""
    D = load_all()
    for k, doc in D.items():
        for q in doc:
            for r in secrefs(q["text"]):
                if k != "ch3" and r.startswith("3"):
                    print("INTO3", k, q["sec"], q["line0"], r, sep="\t")
                if k == "ch3" and r[0] in "1245":
                    print("OUTOF3", k, q["sec"], q["line0"], r, sep="\t")
            if k == "ch3":
                for m in re.finditer(r"Chapter ([1245])\b", q["text"]):
                    print("OUTOF3C", k, q["sec"], q["line0"], m.group(1), sep="\t")


if __name__ == "__main__":
    cmd = sys.argv[1]
    globals()["cmd_" + cmd](sys.argv[2:])
