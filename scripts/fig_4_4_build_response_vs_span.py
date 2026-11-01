"""Figure 4.4 -- build response against span (TASK 96 B, group 2).

SOURCE (one file): results/lever_ranking/arms_661b03c1/lever_ranking.csv (uncapped, vintage; asserted via ARMS.json).
x = col3_live_span_bn (billion USD); y = max_abs_diff_solar_cum_2054_mw, the largest difference in cumulative solar to 2054 across the
family's arms (MW; a terminal-year quantity, as the schedule specifies). The number of distinct plans (n_plan_groups) and the other two
terminal differences are printed with every row and shown in the point labels. Every plotted datum is printed.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fig_common import ROOT, INK, MUTED, BLUE, ORANGE, GREY, GRID, plt, rel, style, save, require_vintage

D = ROOT / "results" / "lever_ranking" / "arms_661b03c1"
require_vintage(D / "ARMS.json")
rows = list(csv.DictReader(open(D / "lever_ranking.csv")))
assert len(rows) == 17
print(f"SOURCE {rel(D / 'lever_ranking.csv')}  (vintage)")
print(f"{'family':26s} {'role':18s} {'live span bn':>12s} {'plans':>5s} {'max d solar 2054 MW':>20s} {'max d gas 2054 MW':>18s} {'max d storage 2054 MWh':>23s}")
pts = []
for r in sorted(rows, key=lambda r: -float(r["col3_live_span_bn"])):
    x = float(r["col3_live_span_bn"]); y = float(r["max_abs_diff_solar_cum_2054_mw"])
    pts.append((r["id"], r["role"], x, y, r["n_plan_groups"]))
    print(f"{r['id']:26s} {r['role']:18s} {x:12.4f} {r['n_plan_groups']:>5s} {y:20.3f} {float(r['max_abs_diff_gas_cum_2054_mw']):18.3f} {float(r['max_abs_diff_storage_2054_mwh']):23.3f}")
fig, ax = plt.subplots(figsize=(8.6, 6), facecolor="white")
sty = {"ranked": (BLUE, "o", "ranked"), "bound": (ORANGE, "s", "bound, not a lever"), "not_comparable": (GREY, "^", "not comparable"), "no_build_response": (INK, "D", "no build response")}
seen = set()
for name, role, x, y, n in pts:
    c, m, lab = sty[role]
    ax.scatter(x, y, s=34, color=c, marker=m, edgecolor="white", linewidth=0.7, zorder=3, label=lab if role not in seen else None)
    seen.add(role)
    ax.annotate(name.replace("_", " "), (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7.5, color=INK)
ax.set_xlabel("Span of discounted system cost across live arms (billion USD)", fontsize=9, color=MUTED)
ax.set_ylabel("Largest difference in cumulative solar to 2054 across the family's arms (MW)", fontsize=9, color=MUTED)
ax.legend(fontsize=8.5, frameon=False, loc="upper left")
style(ax, "both")
fig.tight_layout()
save(fig, "fig_4_4_build_response_vs_span")
