"""Figure 4.2 -- decomposition of system cost, central solution (TASK 96 B, group 1).

SOURCE (one file): results/baseline_sliced/diagnostics__661b03c1.json, cost_decomposition.components_unrounded; the
objective is cost_decomposition.reconciliation.npv_total_cost_usd in the same file.
Fails loudly unless the file's arm_tag carries gas_vintage_dispatch=True.
scripts/16_plot_cost_decomposition.py was NOT reused: it reads fixed-name paths (results/ndc/..., results/baseline/diagnostics.json),
none of which is the vintage file, so this is a new script.
Horizontal bars by term, grouped: capital charge (five lines plus the zero EaaS line), fuel, operation and maintenance.
Every plotted datum is printed.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fig_common import ROOT, INK, MUTED, BLUE, ORANGE, GREEN, GREY, plt, rel, style, save, require_vintage_tag

SRC = ROOT / "results" / "baseline_sliced" / "diagnostics__661b03c1.json"
d = json.loads(SRC.read_text())
require_vintage_tag(d["arm_tag"], SRC)
cd = d["cost_decomposition"]
u = cd["components_unrounded"]
total = float(cd["reconciliation"]["npv_total_cost_usd"])
cap_lines = [("public solar", "public_solar_capital_charge_npv"), ("gas", "gas_capital_charge_npv"),
             ("conversion", "conversion_capital_charge_npv"), ("storage", "storage_capital_charge_npv"),
             ("replacement (rebuild)", "rebuild_capital_charge_npv"), ("EaaS solar", "eaas_solar_capital_charge_npv")]
rows = [(f"capital: {n}", float(u[k]), "capital", k) for n, k in cap_lines]
rows += [("fuel (gas operating cost)", float(u["gas_opex_npv"]), "fuel", "gas_opex_npv"),
         ("operation and maintenance", float(u["om_cost_npv"]), "om", "om_cost_npv")]
cap_total = float(u["capital_charge_total_full_horizon_npv"])
terms_sum = cap_total + float(u["gas_opex_npv"]) + float(u["om_cost_npv"])
print(f"SOURCE {rel(SRC)}  (arm_tag: gas_vintage_dispatch=True); objective {total:,.2f} USD")
print(f"{'term':32s} {'USD':>20s} {'% of objective':>15s}  field")
for n, v, g, k in rows:
    print(f"{n:32s} {v:20,.2f} {100 * v / total:15.4f}  {k}")
print(f"{'capital charge, total':32s} {cap_total:20,.2f} {100 * cap_total / total:15.4f}  capital_charge_total_full_horizon_npv")
print(f"{'fuel':32s} {u['gas_opex_npv']:20,.2f} {100 * u['gas_opex_npv'] / total:15.4f}")
print(f"{'operation and maintenance':32s} {u['om_cost_npv']:20,.2f} {100 * u['om_cost_npv'] / total:15.4f}")
print(f"sum of three groups {terms_sum:,.2f} vs objective {total:,.2f}: difference {terms_sum - total:.6f} USD")
assert abs(terms_sum - total) < 1.0
assert abs(sum(v for n, v, g, k in rows if g == "capital") - cap_total) < 1.0

col = {"capital": BLUE, "fuel": ORANGE, "om": GREEN}
order = sorted([r for r in rows if r[2] == "capital" and r[1] > 0], key=lambda r: -r[1]) + [r for r in rows if r[2] != "capital"] + [r for r in rows if r[2] == "capital" and r[1] == 0]
fig, ax = plt.subplots(figsize=(8.4, 4.6), facecolor="white")
y = list(range(len(order)))[::-1]
ax.barh(y, [r[1] / 1e9 for r in order], color=[col[r[2]] for r in order], height=0.62, linewidth=0)
for yi, r in zip(y, order):
    ax.text(r[1] / 1e9 + 0.08, yi, f"{r[1] / 1e9:.3f} bn ({100 * r[1] / total:.2f}%)", va="center", fontsize=8.5, color=INK)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in order], fontsize=9, color=INK)
ax.set_xlabel(f"Discounted cost (billion USD); objective {total / 1e9:.3f} bn; capital {100 * cap_total / total:.2f}%, fuel {100 * u['gas_opex_npv'] / total:.2f}%, O&M {100 * u['om_cost_npv'] / total:.2f}%", fontsize=8.5, color=MUTED)
ax.set_xlim(0, max(r[1] for r in order) / 1e9 * 1.35)
style(ax, "x")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
fig.tight_layout()
save(fig, "fig_4_2_cost_decomposition")
