"""TASK 84 F.4: every superseded (struck) figure in docs/task_log_84_entries.json that appears in Chapter 1 or Chapter 3, by line number. A struck token is a ~~...~~ span containing a number of at least 3 significant digits; the chapter line is normalised (brackets and thousands commas removed) before the match. Candidates only: a coincidental number is possible and is marked by the entry id for the reader to judge."""
import json, re, sys
E = json.load(open("docs/task_log_84_entries.json"))
toks = {}
for k, e in E.items():
    for m in re.findall(r"~~(.+?)~~", e["text"]):
        for n in re.findall(r"\d[\d,]*\.?\d*", m):
            n2 = n.replace(",", "").rstrip(".")
            if len(n2.replace(".", "")) >= 3 and n2 not in ("2025", "2054", "2045", "2035", "2030", "2050"): toks.setdefault(n2, set()).add(k)
out = []
for path in ("docs/chapters/Chapter1.md", "docs/chapters/CHAPTER_3_assembled.md"):
    for i, line in enumerate(open(path, errors="replace"), 1):
        t = line.replace("⟦", "").replace("⟧", "").replace(",", "")
        hit = {}
        for n, ks in toks.items():
            if re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d])", t): hit[n] = ks
        if hit: out.append((path.split("/")[-1], i, hit, line.strip()[:150]))
for p, i, h, l in out: print(f"{p}:{i}\t" + ", ".join(f"{n}({'/'.join(sorted(k))})" for n, k in h.items()) + f"\t{l}")
print(len(out), "lines", file=sys.stderr)
