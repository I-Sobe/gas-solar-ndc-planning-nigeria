"""Figure 4.3 -- lever ranking, unconstrained (TASK 95 D.1).

SOURCE (exactly one file): results/lever_ranking/arms_661b03c1/lever_ranking.csv
Plotted quantity: col3_live_span_bn (the live-arms span, billion USD). Fails loudly unless the directory's
ARMS.json arm record carries gas_vintage_dispatch=True.

Main panel: the families with role == "ranked", longest at top. Second panel: every family that is not ranked
(role "bound", "not_comparable", "no_build_response"), shaded differently, so nothing in the file is dropped.
Every plotted datum is printed to stdout. Outputs: figures/fig_4_3_lever_ranking.{pdf,png} (300 dpi).
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "lever_ranking" / "arms_661b03c1" / "lever_ranking.csv"
ARMS = SRC.parent / "ARMS.json"
OUT = ROOT / "figures"

if "gas_vintage_dispatch=True" not in json.loads(ARMS.read_text())["arm_tag"]:
    sys.exit(f"FAIL: {ARMS} arm record does not carry gas_vintage_dispatch=True")

rows = list(csv.DictReader(open(SRC)))
for r in rows:
    r["live_bn"] = float(r["col3_live_span_bn"])
    r["span_bn_f"] = float(r["span_bn"])
    r["pct"] = float(r["span_pct_of_central"])
ranked = sorted([r for r in rows if r["role"] == "ranked"], key=lambda r: -r["live_bn"])
others = sorted([r for r in rows if r["role"] != "ranked"], key=lambda r: -r["live_bn"])
assert len(rows) == 17, len(rows)

print(f"SOURCE {SRC.relative_to(ROOT)}  (arm record: gas_vintage_dispatch=True)")
print(f"{'id':28s} {'role':18s} {'all-arms span bn':>16s} {'live span bn':>13s} {'% central':>9s} {'n_present':>9s} {'in_F5':>6s}  unasserted arms")
for r in ranked + others:
    print(f"{r['id']:28s} {r['role']:18s} {r['span_bn_f']:16.6f} {r['live_bn']:13.6f} {r['pct']:9.4f} {r['n_present']:>9s} {r['in_f5_ranking']:>6s}  {r['col3_unasserted_arms'] or '-'}")
print(f"rows printed: {len(ranked) + len(others)}  (ranked {len(ranked)}, other {len(others)})")

INK, MUTED, GRID = "#1f2933", "#52606d", "#e4e7eb"
BLUE, GREY = "#2b6cb0", "#9aa5b1"
ROLE_LABEL = {"bound": "bound, not a lever", "not_comparable": "not comparable", "no_build_response": "no build response"}

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [3, 1.6]}, facecolor="white")
y = list(range(len(ranked)))[::-1]
ax.barh(y, [r["live_bn"] for r in ranked], color=BLUE, height=0.62, linewidth=0)
for yi, r in zip(y, ranked):
    ax.text(r["live_bn"] + 0.08, yi, f"{r['live_bn']:.2f}", va="center", fontsize=8.5, color=INK)
ax.set_yticks(y)
ax.set_yticklabels([r["id"].replace("_", " ") for r in ranked], fontsize=9, color=INK)
ax.set_xlabel("Span of discounted system cost across the family's live arms (billion USD)", fontsize=9, color=MUTED)
ax.set_title("Ranked families (14)", fontsize=10, color=INK, loc="left")
ax.set_xlim(0, max(r["live_bn"] for r in ranked) * 1.12)

y2 = list(range(len(others)))[::-1]
ax2.barh(y2, [r["live_bn"] for r in others], color=GREY, height=0.62, linewidth=0, hatch="///", edgecolor="white")
for yi, r in zip(y2, others):
    ax2.text(r["live_bn"] + 0.2, yi, f"{r['live_bn']:.2f}", va="center", fontsize=8.5, color=INK)
ax2.set_yticks(y2)
ax2.set_yticklabels([f"{r['id'].replace('_', ' ')}\n({ROLE_LABEL[r['role']]})" for r in others], fontsize=8.5, color=INK)
ax2.set_xlabel("Span (billion USD)", fontsize=9, color=MUTED)
ax2.set_title("Not ranked", fontsize=10, color=INK, loc="left")
ax2.set_xlim(0, max(r["live_bn"] for r in others) * 1.22)

for a in (ax, ax2):
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    a.spines["bottom"].set_color(GRID)
    a.tick_params(axis="y", length=0)
    a.tick_params(axis="x", colors=MUTED, labelsize=8.5)
    a.xaxis.grid(True, color=GRID, linewidth=0.7)
    a.set_axisbelow(True)
fig.tight_layout()
OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_4_3_lever_ranking.{ext}", dpi=300)
print("wrote figures/fig_4_3_lever_ranking.pdf and .png")
