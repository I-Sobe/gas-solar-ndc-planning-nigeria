"""TASK 85 Part A: numeric substitutions in Chapter 1 and Chapter 3. Each edit is (first line, last line, [(old, new)]) on the file as it stood at the start of the task; the old strings must occur in that range exactly once. A touched block is re-wrapped to 72 columns only if a line overflows. Writes docs/task_log_85_partA_diff.md (exact old and new lines)."""
import sys, textwrap, difflib
W = 72
C1 = "docs/chapters/Chapter1.md"; C3 = "docs/chapters/CHAPTER_3_assembled.md"
EDITS = {
 C1: [
  (179, 180, [("⟦46.8⟧", "⟦43.5⟧")]),
  (359, 359, [("⟦eight⟧", "⟦eighteen⟧")]),
  (360, 362, [("never exceeds ⟦5.23⟧", "is at most ⟦−0.61⟧")]),
  (362, 363, [("⟦twenty-eight⟧", "⟦eighteen⟧"), ("⟦180,067,166⟧", "⟦653,600,175⟧")]),
  (371, 371, [("⟦57.1⟧ to ⟦64.7⟧", "⟦60.0⟧ to ⟦62.8⟧")]),
  (417, 417, [("⟦37,414⟧", "⟦24,698⟧")]),
  (472, 472, [("⟦87⟧", "⟦88⟧")]),
  (481, 482, [("Of the ⟦87⟧ assertions, ⟦80⟧ pass, ⟦6⟧ fail and ⟦1⟧ is undeterminable\nbecause the condition it tested no longer arises.",
               "Of the ⟦88⟧ assertions, ⟦81⟧ pass, ⟦7⟧ fail and ⟦0⟧ are undeterminable\n(stage A ⟦50⟧/⟦5⟧/⟦0⟧, stage B ⟦16⟧/⟦2⟧/⟦0⟧, stage C ⟦15⟧/⟦0⟧/⟦0⟧).")]),
 ],
 C3: [
  (524, 528, [("⟦0.135%⟧", "⟦0.150%⟧"), ("⟦0.013%⟧", "⟦0.022%⟧")]),
  (1440, 1445, [("⟦0.135%⟧", "⟦0.150%⟧"), ("⟦0.013%⟧", "⟦0.022%⟧")]),
  (1508, 1512, [("⟦0.013%⟧", "⟦0.022%⟧")]),
  (1762, 1764, [("⟦25,862,214,166.78⟧ to ⟦24,986,620,949.18⟧", "⟦25,862,214,166.78⟧ [VERIFY: blended-generation comparator, not re-derived on the vintage model] to ⟦23,323,068,340.46⟧")]),
  (1804, 1809, [("⟦10,944,915,186⟧", "⟦9,329,045,793⟧"), ("⟦43.8%⟧", "⟦40.0%⟧"), ("⟦9.4%⟧", "⟦10.1%⟧"), ("⟦5,547,704,677⟧", "⟦3,061,123,411⟧"),
                (" ⟦634,879,738⟧", " ⟦2,382,982,003⟧"), ("⟦2,314,163,576⟧", "⟦1,527,593,314⟧"), ("⟦91,477,410⟧", "⟦657,279⟧")]),
  (1916, 1917, [("⟦8,723,252,062⟧", "⟦5,518,409,480⟧"), ("⟦17.8%⟧", "⟦86.2%⟧")]),
  (2631, 2632, [("⟦24.987⟧", "⟦23.323⟧"), ("⟦37,414.49⟧", "⟦24,697.69⟧"), ("⟦63,056.31⟧", "⟦56,289.88⟧")]),
  (2837, 2841, [("⟦37.846⟧", "⟦36.343⟧"), ("⟦30.602⟧", "⟦24.253⟧"), ("⟦7.244⟧", "⟦12.089⟧"), ("⟦19.1⟧", "⟦33.3⟧"), ("⟦27.50⟧", "⟦25.71⟧")]),
  (2858, 2861, [("⟦8 of 20⟧", "⟦5 of 20⟧"), ("⟦0.17⟧", "⟦0.15⟧")]),
  (3382, 3386, [("⟦7 of 32⟧", "⟦12 of 32⟧"), ("⟦0.8125⟧\nis dominated; on the public-tight arm only the uncapped anchor survives,\n⟦1 of 32⟧.", "⟦0.625⟧\nis dominated; on the public-tight arm ⟦7 of 32⟧ points are non-dominated,\ndown to `cap_fraction` ⟦0.8125⟧.")]),
  (3408, 3410, [("⟦5 of 8⟧", "⟦8 of 8⟧")]),
  (3416, 3417, [("⟦3 of 20⟧", "⟦12 of 20⟧")]),
  (3421, 3426, [("⟦14\n   binding, 4 slack and 1 absent⟧", "⟦11\n   binding, 7 slack and 1 absent⟧"), ("⟦12, 6 and 1⟧", "⟦9, 9 and 1⟧")]),
 ]}
report = ["# TASK 85 Part A — exact old and new lines\n"]
for path, edits in EDITS.items():
    L = open(path).read().split("\n")
    # apply bottom-up so line numbers stay valid
    import re
    for (a, b, subs) in sorted(edits, key=lambda e: -e[0]):
        blk = "\n".join(L[a - 1:b]); old = blk
        for o, n in subs:
            assert blk.count(o) == 1, (path, a, b, o, blk.count(o))
            blk = blk.replace(o, n)
        lines = blk.split("\n"); end = b
        if any(len(x) > W for x in lines) or len(lines) != len(old.split("\n")) and False:
            pass
        # re-wrap only if a line overflows or the block grew past its original line count: extend to the paragraph end and re-flow
        if any(len(x) > W for x in lines) or len(lines) > b - a + 1:
            while end < len(L) and L[end].strip() and not re.match(r"^(\d+\. |\||#|---)", L[end]): end += 1
            tail = L[b:end]; full = lines + tail; old = "\n".join(L[a - 1:end])
            ind = " " * (len(full[0]) - len(full[0].lstrip())); sub = " " * (len(full[1]) - len(full[1].lstrip())) if len(full) > 1 else ind
            lines = textwrap.wrap(" ".join(x.strip() for x in full), width=W, initial_indent=ind, subsequent_indent=sub, break_long_words=False, break_on_hyphens=False)
        L[a - 1:end] = lines
        d = [x for x in difflib.unified_diff(old.split("\n"), lines, lineterm="", n=0) if not x.startswith(("---", "+++", "@@"))]
        report.append(f"**{path.split('/')[-1]}:{a}-{end}**\n```\n" + "\n".join(d) + "\n```\n")
    open(path, "w").write("\n".join(L))
open("docs/task_log_85_partA_diff.md", "w").write("\n".join(report[:1] + sorted(report[1:], key=lambda s: (s.split(':')[0], int(s.split(':')[1].split('-')[0])))))
print("ok")
