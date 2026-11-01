"""Figure 4.6 -- Sobol first-order and total-effect indices on system cost (TASK 96 B, group 2).

SOURCE (one file): results/parameter_uncertainty/arms_09bd6dbc/summary_n256_seed42.json (the keyed path, not the unkeyed
results/parameter_uncertainty/summary_n256_seed42.json); sobol_indices.first_order[0] / total_order[0] with the matching
_ci95 low/high rows (quantity of interest 0 = npv_total_cost_usd). Vintage asserted via the directory's ARMS.json.
Every plotted datum is printed, including the 95% intervals (several first-order intervals exceed 1.0 or fall below 0; they are plotted as stored).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fig_common import ROOT, INK, MUTED, BLUE, ORANGE, GREY, plt, rel, style, save, require_vintage

D = ROOT / "results" / "parameter_uncertainty" / "arms_09bd6dbc"
require_vintage(D / "ARMS.json")
S = json.loads((D / "summary_n256_seed42.json").read_text())
si = S["sobol_indices"]
assert si["qoi_order"][0] == "npv_total_cost_usd"
names = S["param_names"]
fo, to = si["first_order"][0], si["total_order"][0]
fo_lo, fo_hi = si["first_order_ci95"]["low"][0], si["first_order_ci95"]["high"][0]
to_lo, to_hi = si["total_order_ci95"]["low"][0], si["total_order_ci95"]["high"][0]
print(f"SOURCE {rel(D / 'summary_n256_seed42.json')}  (vintage); n_base {S['n_base']}, total solves {S['total_solves']}, failures {S['feasibility']['n_failed']}")
print(f"{'parameter':28s} {'first-order':>12s} {'ci95 low':>10s} {'ci95 high':>10s} {'total-effect':>13s} {'ci95 low':>10s} {'ci95 high':>10s}")
for i, n in enumerate(names):
    print(f"{n:28s} {fo[i]:12.6f} {fo_lo[i]:10.6f} {fo_hi[i]:10.6f} {to[i]:13.6f} {to_lo[i]:10.6f} {to_hi[i]:10.6f}")
print(f"sum first-order {sum(fo):.6f}; sum total-effect {sum(to):.6f}")
fig, ax = plt.subplots(figsize=(8.4, 4.8), facecolor="white")
y = list(range(len(names)))[::-1]
h = 0.34
ax.barh([v + h / 2 for v in y], fo, h, color=BLUE, linewidth=0, label="first-order", xerr=[[a - b for a, b in zip(fo, fo_lo)], [b - a for a, b in zip(fo, fo_hi)]], error_kw=dict(ecolor=INK, elinewidth=0.8, capsize=2))
ax.barh([v - h / 2 for v in y], to, h, color=ORANGE, linewidth=0, label="total-effect", xerr=[[a - b for a, b in zip(to, to_lo)], [b - a for a, b in zip(to, to_hi)]], error_kw=dict(ecolor=INK, elinewidth=0.8, capsize=2))
for yi, a, b in zip(y, fo, to):
    ax.text(max(a, b) + 0.13, yi, f"{a:.4f} / {b:.4f}", va="center", fontsize=8, color=INK)
ax.set_yticks(y)
ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=9, color=INK)
ax.set_xlabel("Sobol index on discounted system cost (bars: estimate; whiskers: 95% interval)", fontsize=9, color=MUTED)
ax.set_xlim(-0.05, 1.45)
ax.legend(fontsize=8.5, frameon=False, loc="lower right")
style(ax, "x")
ax.tick_params(axis="y", length=0)
fig.tight_layout()
save(fig, "fig_4_6_sobol_indices")
