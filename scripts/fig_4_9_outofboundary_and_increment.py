"""Figure 4.9 -- out-of-boundary share and policy-induced increment, by capital tier and collection rate (TASK 96 B, group 1).

SOURCES (two files, both vintage, asserted via ARMS.json):
  capped  : results/fin3/arms_1d68be49/capital_sweep_results.csv   (NDC 3.0 cap, 15 cells)
  uncapped: results/fin3/arms_7f91f752/capital_sweep_results.csv   (--no-cap, 13 cells; `unconstrained` at 60% and 35% have no cell)
Plotted: for every cell the capped out-of-boundary share, the uncapped share, and the increment (capped minus uncapped), in per cent,
with the 10% flag threshold. CAPTION (for the chapter): the field named `leakage_share` is the out-of-boundary LEVEL (section 3.5.6);
the increment equals the level on this generation because uncapped self-generation is zero in every measured cell; that equality is a
measurement, not a definition. Every plotted datum is printed.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fig_common import ROOT, INK, MUTED, BLUE, ORANGE, GREY, RED, plt, rel, style, save, require_vintage

C = ROOT / "results" / "fin3" / "arms_1d68be49"
U = ROOT / "results" / "fin3" / "arms_7f91f752"
require_vintage(C / "ARMS.json")
require_vintage(U / "ARMS.json")
if "emissions_cap=none" not in (U / "ARMS.json").read_text():
    sys.exit("FAIL: uncapped file's arm record does not carry emissions_cap=none")
key = lambda r: (r["capital_case"], r["collection_rate_scenario"])
cap = {key(r): r for r in csv.DictReader(open(C / "capital_sweep_results.csv"))}
un = {key(r): r for r in csv.DictReader(open(U / "capital_sweep_results.csv"))}
tiers = ["tight", "moderate", "adequacy", "expansion", "unconstrained"]
rates = [("no_friction", "100% collection"), ("reform_target", "60% collection"), ("current", "35% collection")]
print(f"SOURCE capped   {rel(C / 'capital_sweep_results.csv')}  (vintage)")
print(f"SOURCE uncapped {rel(U / 'capital_sweep_results.csv')}  (vintage, emissions_cap=none)")
print(f"{'tier':14s} {'collection':>10s} {'capped %':>9s} {'uncapped %':>11s} {'increment pp':>13s}")
data = {}
for t in tiers:
    for k, lab in rates:
        c = float(cap[(t, k)]["leakage_share"]) * 100
        u = float(un[(t, k)]["leakage_share"]) * 100 if (t, k) in un else None
        data[(t, k)] = (c, u, None if u is None else c - u)
        print(f"{t:14s} {lab.split()[0]:>10s} {c:9.3f} {('%11.3f' % u) if u is not None else '   NO CELL  '} {('%+13.3f' % (c - u)) if u is not None else '          n/a'}")
print("cells above 10%% (level): %d of 15; (increment, measured cells): %d of 13" % (
    sum(v[0] > 10 for v in data.values()), sum(v[2] is not None and v[2] > 10 for v in data.values())))

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(11.5, 4.8), facecolor="white")
w = 0.26
for ax, (k, lab) in zip(axes, rates):
    for i, t in enumerate(tiers):
        c, u, inc = data[(t, k)]
        ax.bar(i - w, c, w, color=BLUE, linewidth=0, label="capped out-of-boundary share" if i == 0 else None)
        if u is not None:
            ax.bar(i, u, w, color=GREY, linewidth=0, label="uncapped share" if i == 0 else None)
            ax.bar(i + w, inc, w, color=ORANGE, linewidth=0, label="increment (capped - uncapped)" if i == 0 else None)
        else:
            ax.text(i + 0.02, 0.8, "no\ncell", ha="center", fontsize=7, color=MUTED)
        ax.text(i - w, c + 0.8, f"{c:.1f}" if c >= 1 else f"{c:.2f}", ha="center", fontsize=7, color=INK)
    ax.axhline(10, color=RED, linewidth=1, linestyle="--")
    ax.text(len(tiers) - 0.5, 10.6, "10% threshold", ha="right", fontsize=7.5, color=RED)
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels(tiers, rotation=30, ha="right", fontsize=8.5, color=INK)
    ax.set_title(lab, fontsize=9.5, color=INK, loc="left")
    style(ax)
axes[0].set_ylabel("Share of cumulative emissions outside the cap boundary (%)", fontsize=9, color=MUTED)
axes[0].legend(fontsize=8, frameon=False, loc="upper right")
fig.tight_layout()
save(fig, "fig_4_9_outofboundary_and_increment")
