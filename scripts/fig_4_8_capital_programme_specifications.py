"""Figure 4.8 -- composition of the capital programme, vintage against blended (TASK 96 B, group 2).

SOURCES (two files):
  vintage: results/baseline_sliced/diagnostics__661b03c1.json  cost_decomposition.components_unrounded   (arm_tag asserted gas_vintage_dispatch=True)
  blended: results/baseline_sliced/diagnostics__6046f759.json  cost_decomposition.components_unrounded   (arm_tag asserted NOT vintage)
Bars: each capital line as a per cent of its own file's objective (reconciliation.npv_total_cost_usd). CAPTION (for the chapter): the
comparison is across model specifications, not across scenarios. Every plotted datum is printed, with absolute USD.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fig_common import ROOT, INK, MUTED, BLUE, ORANGE, plt, rel, style, save, require_vintage_tag

V = ROOT / "results" / "baseline_sliced" / "diagnostics__661b03c1.json"
B = ROOT / "results" / "baseline_sliced" / "diagnostics__6046f759.json"
dv, db = json.loads(V.read_text()), json.loads(B.read_text())
require_vintage_tag(dv["arm_tag"], V)
if "gas_vintage_dispatch=True" in db["arm_tag"]:
    sys.exit(f"FAIL: {B} is vintage; the comparison file must be the blended generation")
lines = [("public solar", "public_solar_capital_charge_npv"), ("gas", "gas_capital_charge_npv"), ("conversion", "conversion_capital_charge_npv"),
         ("storage", "storage_capital_charge_npv"), ("replacement", "rebuild_capital_charge_npv")]
tv, tb = dv["cost_decomposition"]["reconciliation"]["npv_total_cost_usd"], db["cost_decomposition"]["reconciliation"]["npv_total_cost_usd"]
cv, cb = dv["cost_decomposition"]["components_unrounded"], db["cost_decomposition"]["components_unrounded"]
print(f"SOURCE vintage {rel(V)}  objective {tv:,.2f}")
print(f"SOURCE blended {rel(B)}  objective {tb:,.2f}")
print(f"{'capital line':16s} {'blended USD':>18s} {'blended %':>10s} {'vintage USD':>18s} {'vintage %':>10s} {'ratio v/b':>10s}")
for n, k in lines:
    b, v = float(cb[k]), float(cv[k])
    print(f"{n:16s} {b:18,.2f} {100 * b / tb:10.4f} {v:18,.2f} {100 * v / tv:10.4f} {v / b if b else float('nan'):10.3f}")
tcb, tcv = float(cb["capital_charge_total_full_horizon_npv"]), float(cv["capital_charge_total_full_horizon_npv"])
print(f"{'capital total':16s} {tcb:18,.2f} {100 * tcb / tb:10.4f} {tcv:18,.2f} {100 * tcv / tv:10.4f} {tcv / tcb:10.3f}")
fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor="white")
w = 0.36
xs = range(len(lines))
bb = [100 * float(cb[k]) / tb for n, k in lines]
vv = [100 * float(cv[k]) / tv for n, k in lines]
ax.bar([x - w / 2 for x in xs], bb, w, color=ORANGE, linewidth=0, label="preceding specification (blended dispatch)")
ax.bar([x + w / 2 for x in xs], vv, w, color=BLUE, linewidth=0, label="corrected specification (vintage dispatch)")
for x, a, b in zip(xs, bb, vv):
    ax.text(x - w / 2, a + 0.3, f"{a:.1f}", ha="center", fontsize=8, color=INK)
    ax.text(x + w / 2, b + 0.3, f"{b:.1f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(list(xs))
ax.set_xticklabels([n for n, k in lines], fontsize=9, color=INK)
ax.set_ylabel("Capital charge line as a share of the objective (%)", fontsize=9, color=MUTED)
ax.legend(fontsize=8.5, frameon=False, loc="upper right")
style(ax)
fig.tight_layout()
save(fig, "fig_4_8_capital_programme_specifications")
