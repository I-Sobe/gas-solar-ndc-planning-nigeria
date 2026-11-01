"""Figure 4.11 -- distribution of per-draw cost, committed plan against wait-and-see (TASK 96 B, group 2).

SOURCES (named; the figure needs the two legs of one VSS bound, so two files, plus one guard file):
  committed (first stage frozen, re-dispatched per draw): results/stochastic_redispatch/draws__19490865.jsonl, field cost_usd
      (arm record: results/stochastic_redispatch/summary__19490865.json arm_tag, asserted gas_vintage_dispatch=True)
  wait-and-see (re-optimised per draw): results/monte_carlo/arms_f30c5990/monte_carlo_raw.csv, case "Matched to 30_ central (no NDC)", field npv_cost
      (arm record: results/monte_carlo/arms_f30c5990/ARMS.json, asserted gas_vintage_dispatch=True)
  guard: results/vss_bound/vss_bound__9e2eda70.json; both legs asserted capital_case = unconstrained, and its ecp_free_usd / ws_usd must equal the
      two means computed here.
Two empirical CDFs (100 draws each); medians and means marked. CAPTION (for the chapter): both legs run at capital_case=unconstrained.
Every plotted datum (all 200 per-draw costs, sorted) is printed.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fig_common import ROOT, INK, MUTED, BLUE, ORANGE, GREY, plt, rel, style, save, require_vintage, require_vintage_tag

SR = ROOT / "results" / "stochastic_redispatch"
MC = ROOT / "results" / "monte_carlo" / "arms_f30c5990"
VB = ROOT / "results" / "vss_bound" / "vss_bound__9e2eda70.json"
S1 = json.loads((SR / "summary__19490865.json").read_text())
require_vintage_tag(S1["arm_tag"], SR / "summary__19490865.json")
require_vintage(MC / "ARMS.json")
vb = json.loads(VB.read_text())
g = vb["guards"]["ws_case_scenario_matches_stage1"]
for side in ("stage1_scenario_fields", "ws_case_fields"):
    if g[side]["capital_case"] != "unconstrained":
        sys.exit(f"FAIL: {VB} {side} capital_case is {g[side]['capital_case']}, not unconstrained")
com = [json.loads(l) for l in open(SR / "draws__19490865.jsonl")]
assert len(com) == 100 and all(r["feasible"] for r in com)
cc = sorted(r["cost_usd"] for r in com)
raw = [r for r in csv.DictReader(open(MC / "monte_carlo_raw.csv")) if r["case"] == "Matched to 30_ central (no NDC)"]
assert len(raw) == 100 and all(r["status"] == "optimal" for r in raw)
ws = sorted(float(r["npv_cost"]) for r in raw)
m_c, m_w = sum(cc) / 100, sum(ws) / 100
md_c, md_w = statistics.median(cc), statistics.median(ws)
print(f"SOURCE committed {rel(SR / 'draws__19490865.jsonl')}; wait-and-see {rel(MC / 'monte_carlo_raw.csv')} (case 'Matched to 30_ central (no NDC)'); guard {rel(VB)}")
print(f"means: committed {m_c:,.2f} (vss_bound ecp_free_usd {vb['ecp_free_usd']:,.2f}); wait-and-see {m_w:,.2f} (vss_bound ws_usd {vb['ws_usd']:,.2f})")
assert abs(m_c - vb["ecp_free_usd"]) < 1 and abs(m_w - vb["ws_usd"]) < 1
print(f"medians: committed {md_c:,.2f}; wait-and-see {md_w:,.2f}; mean difference {m_c - m_w:,.2f} (vss_bound bound_upper_usd {vb['bound_upper_usd']:,.2f})")
print(f"min/max: committed {cc[0]:,.0f} / {cc[-1]:,.0f}; wait-and-see {ws[0]:,.0f} / {ws[-1]:,.0f}")
print(f"{'rank':>4s} {'committed cost USD':>22s} {'wait-and-see cost USD':>24s}")
for i in range(100):
    print(f"{i + 1:4d} {cc[i]:22,.2f} {ws[i]:24,.2f}")
fig, ax = plt.subplots(figsize=(8.4, 5), facecolor="white")
p = [(i + 1) / 100 for i in range(100)]
ax.step([c / 1e9 for c in cc], p, where="post", color=ORANGE, linewidth=1.6, label="committed plan")
ax.step([w / 1e9 for w in ws], p, where="post", color=BLUE, linewidth=1.6, label="wait-and-see")
for val, col, ls, lab in ((md_c, ORANGE, ":", "median"), (m_c, ORANGE, "--", "mean"), (md_w, BLUE, ":", "median"), (m_w, BLUE, "--", "mean")):
    ax.axvline(val / 1e9, color=col, linestyle=ls, linewidth=1)
ax.text(0.40, 0.30, f"committed plan: median {md_c / 1e9:.2f} bn (dotted), mean {m_c / 1e9:.2f} bn (dashed)\nwait-and-see: median {md_w / 1e9:.2f} bn (dotted), mean {m_w / 1e9:.2f} bn (dashed)\nmean difference (upper bound on the VSS) {(m_c - m_w) / 1e9:.2f} bn",
        transform=ax.transAxes, fontsize=8.5, color=INK, va="top", ha="left", linespacing=1.6)
ax.set_xlabel("Per-draw discounted system cost (billion USD)", fontsize=9, color=MUTED)
ax.set_ylabel("Cumulative share of draws", fontsize=9, color=MUTED)
ax.legend(fontsize=8.5, frameon=False, loc="lower right")
style(ax, "both")
fig.tight_layout()
save(fig, "fig_4_11_per_draw_cost")
