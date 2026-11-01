"""Figure 4.5 -- unconstrained against capped ranking, slope chart (TASK 96 B, group 1).

SOURCES (two files, both named; both vintage, asserted via their directories' ARMS.json):
  left : results/lever_ranking/arms_661b03c1/lever_ranking.csv                                   (uncapped)
  right: results/lever_ranking_capped/ndc3_conditional_noenvelope/arms_4ca4832d/lever_ranking.csv (cap alone, NDC 3.0
         conditional, capital_case=unconstrained; NOT the envelope-carrying directories)
Rank = position by col3_live_span_bn among families with role == "ranked" in that file. Families that are not ranked in a
file are listed under "not ranked" with their role. Every plotted datum is printed.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fig_common import ROOT, INK, MUTED, BLUE, ORANGE, GREY, GRID, plt, rel, style, save, require_vintage

L = ROOT / "results" / "lever_ranking" / "arms_661b03c1"
R = ROOT / "results" / "lever_ranking_capped" / "ndc3_conditional_noenvelope" / "arms_4ca4832d"
for d in (L, R):
    require_vintage(d / "ARMS.json")
if "capital_case=unconstrained" not in open(R / "ARMS.json").read():
    sys.exit("FAIL: capped file's arm record does not carry capital_case=unconstrained")


def load(d):
    rows = list(csv.DictReader(open(d / "lever_ranking.csv")))
    ranked = sorted([r for r in rows if r["role"] == "ranked"], key=lambda r: -float(r["col3_live_span_bn"]))
    rank = {r["id"]: i + 1 for i, r in enumerate(ranked)}
    return {r["id"]: (r["role"], float(r["col3_live_span_bn"]), rank.get(r["id"])) for r in rows}, len(ranked)


left, nl = load(L / "")
right, nr = load(R)
ids = sorted(left, key=lambda k: (left[k][2] is None, left[k][2] or 99, k))
print(f"SOURCE left  {rel(L / 'lever_ranking.csv')}   ranked families: {nl}")
print(f"SOURCE right {rel(R / 'lever_ranking.csv')}   ranked families: {nr}")
print(f"{'family':26s} {'uncapped role':18s} {'span bn':>9s} {'rank':>5s}   {'capped role':18s} {'span bn':>9s} {'rank':>5s}  move")
for k in ids:
    lr, ls, lk = left[k]; rr, rs, rk = right[k]
    mv = "" if (lk is None or rk is None) else f"{lk - rk:+d}"
    print(f"{k:26s} {lr:18s} {ls:9.4f} {str(lk or '-'):>5s}   {rr:18s} {rs:9.4f} {str(rk or '-'):>5s}  {mv}")

fig, ax = plt.subplots(figsize=(9.6, 7.2), facecolor="white")
for k in ids:
    lr, ls, lk = left[k]; rr, rs, rk = right[k]
    if lk is None and rk is None:
        continue
    yl = lk if lk is not None else nl + 2
    yr = rk if rk is not None else nr + 2
    mv = (lk - rk) if (lk is not None and rk is not None) else 0
    col = BLUE if mv >= 3 else (ORANGE if mv <= -3 else GREY)
    ax.plot([0, 1], [yl, yr], color=col, linewidth=1.8 if col != GREY else 1.0, zorder=2)
    ax.scatter([0, 1], [yl, yr], s=22, color=col, zorder=3)
    ax.text(-0.04, yl, f"{k.replace('_', ' ')}  {ls:.2f}" if lk else f"{k.replace('_', ' ')} ({lr.replace('_', ' ')})", ha="right", va="center", fontsize=8.3, color=INK)
    ax.text(1.04, yr, f"{k.replace('_', ' ')}  {rs:.2f}" if rk else f"{k.replace('_', ' ')} ({rr.replace('_', ' ')})", ha="left", va="center", fontsize=8.3, color=INK)
ax.set_xlim(-0.75, 1.75)
ax.set_ylim(max(nl, nr) + 3, 0)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Unconstrained (rank, live span bn)", "Capped, cap alone: NDC 3.0 conditional (rank, live span bn)"], fontsize=9, color=INK)
ax.set_yticks([])
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.set_title("Families moving three or more places are coloured (blue up, orange down)", fontsize=9, color=MUTED, loc="left")
fig.tight_layout()
save(fig, "fig_4_5_ranking_uncapped_vs_capped")
