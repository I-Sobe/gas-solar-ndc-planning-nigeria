"""TASK 84 F.3: append the re-derived blocks to docs/findings_frozen.md (TASK 55 style: old figure struck through, new beside it, file named, class recorded). History is not deleted: each block is inserted after the entry's Source paragraph, above the earlier blocks."""
import json, re
E = json.load(open("docs/task_log_84_entries.json"))
L = open("docs/findings_frozen.md").read().split("\n")
heads = [(i, re.match(r"## ((?:F|M)-\d+)", l).group(1)) for i, l in enumerate(L) if re.match(r"## (F|M)-\d+", l)]
heads.append((len(L), "END"))
ins = []
for (a, k), (b, _) in zip(heads, heads[1:]):
    if any("RE-DERIVED, TASK 84" in x for x in L[a:b]): continue
    e = E[k]
    # insertion point: end of the first paragraph that starts with **Source (or, failing that, the first paragraph after the heading's subtitle)
    j = next((x for x in range(a + 1, b) if L[x].startswith("**Source") or L[x].startswith("**RE-MEASURED") and False), None)
    if j is None: j = a + 1
    while j < b and L[j].strip(): j += 1
    block = ["", f"> **[RE-DERIVED, TASK 84, 2026-10-23 — {e['cls']}: {e['clause']}.** Basis: {e['basis']}. " + e["text"] + " Source: " + ", ".join(f"`{f}`" for f in e["files"]) + ". Detail: `docs/task_log_84_part*.md`.]**"]
    ins.append((j, block))
for j, block in sorted(ins, reverse=True): L[j:j] = block
open("docs/findings_frozen.md", "w").write("\n".join(L))
print(len(ins), "blocks inserted")
