"""TASK 48 G.2: staleness gate for docs/findings_frozen.md.

For every `## F-n` entry, find its `Source.` line(s), resolve the result files they name
against docs/results_manifest.json and report, WITHOUT fixing anything:
  A  entries citing a result file/directory that is not in the manifest
  B  entries whose cited file in the manifest has an mtime LATER than the entry's last
     dated block (the entry may describe an earlier version of the file).  This is the
     best the manifest allows: it stores today's md5 only, and no entry records the md5
     it was written against, so an md5 comparison is impossible; mtime stands in for it.
  C  entries with no parseable `Source.` line (or a Source line naming no results path)
Exit status 1 if any hit.  Usage: python3 scripts/check_findings_freshness.py [--json]
"""
import fnmatch, itertools, json, os, re, sys, datetime as dt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
FIND = "docs/findings_frozen.md"
SRC_START = re.compile(r"^\s*(?:>\s*)?\*\*Source[^*]*\*\*|^\s*\*\*Source[^*]*\.\*\*|^\s*Source\.")
DATE = re.compile(r"\b(20\d\d-\d\d-\d\d)\b")
TOK = re.compile(r"`([^`]+)`")

def expand(tok):
    m = re.search(r"\{([^{}]+)\}", tok)
    if not m: return [tok]
    return list(itertools.chain.from_iterable(expand(tok[:m.start()] + a + tok[m.end():]) for a in m.group(1).split(",")))

def paths_in(text, top_dirs):
    out = []
    for t in TOK.findall(text):
        t = t.strip().rstrip(".,;:")
        if "/" not in t or " " in t: continue
        first = t.split("/")[0]
        if not (t.startswith("results/") or first in top_dirs): continue
        if not t.startswith("results/"): t = "results/" + t
        out += expand(t)
    return out

def main():
    man = json.load(open("docs/results_manifest.json"))
    mf = {e["path"]: e for e in man["files"]}
    top_dirs = {d for d in os.listdir("results") if os.path.isdir(os.path.join("results", d))}
    L = open(FIND).read().split("\n")
    entries, cur = [], None
    for i, l in enumerate(L, 1):
        m = re.match(r"## (F-\d+)\b(.*)", l)
        if m: cur = dict(id=m.group(1), line=i, title=m.group(2).strip()[:90], lines=[]); entries.append(cur); continue
        if re.match(r"#{1,2} ", l): cur = None
        if cur is not None: cur["lines"].append((i, l))
    A, B, C = [], [], []
    for e in entries:
        txt = "\n".join(l for _, l in e["lines"])
        dates = DATE.findall(txt)
        last = max(dates) if dates else None
        src_idx = [k for k, (_, l) in enumerate(e["lines"]) if SRC_START.match(l)]
        if not src_idx:
            C.append((e, "no `Source.` line")); continue
        cited = []
        for k in src_idx:
            para = []
            for _, l in e["lines"][k:]:
                if not l.strip() and para: break
                para.append(l)
            cited += paths_in("\n".join(para), top_dirs)
        if not cited:
            C.append((e, "`Source.` line names no results path")); continue
        missing, newer = [], []
        for p in dict.fromkeys(cited):
            hits = [q for q in mf if q == p or q.startswith(p.rstrip("/") + "/") or fnmatch.fnmatch(q, p)]
            if not hits: missing.append(p); continue
            if last:
                d0 = dt.date.fromisoformat(last)
                later = [q for q in hits if dt.date.fromisoformat(mf[q]["mtime"][:10]) > d0]
                if later: newer.append((p, len(later), max(mf[q]["mtime"] for q in later)))
        if missing: A.append((e, missing))
        if newer: B.append((e, last, newer))
    if "--json" in sys.argv:
        print(json.dumps(dict(A=[(e["id"], m) for e, m in A], B=[(e["id"], l, n) for e, l, n in B], C=[(e["id"], r) for e, r in C]), indent=1))
    print(f"entries: {len(entries)}")
    print(f"\nA. cites a file absent from the manifest: {len(A)}")
    for e, m in A: print(f"  {e['id']} (line {e['line']}): {', '.join(m[:4])}{' ...' if len(m)>4 else ''}")
    print(f"\nB. cited file newer than the entry's last dated block: {len(B)}")
    for e, l, n in B: print(f"  {e['id']} (line {e['line']}): last dated {l}; " + "; ".join(f"{p} ({c} file(s), newest {t[:10]})" for p, c, t in n[:3]))
    print(f"\nC. no parseable Source line: {len(C)}")
    for e, r in C: print(f"  {e['id']} (line {e['line']}): {r}")
    flagged = {e["id"] for e, *_ in A} | {e["id"] for e, *_ in B} | {e["id"] for e, _ in C}
    print(f"\nentries with at least one hit: {len(flagged)} of {len(entries)}")
    sys.exit(1 if flagged else 0)

if __name__ == "__main__":
    main()
