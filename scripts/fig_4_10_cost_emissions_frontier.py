"""Figure 4.10 -- cost-emissions frontier (TASK 95 D.3).

SOURCE (exactly one file): results/pareto_frontier/arms_1d68be49/pareto_raw.csv (vintage). Fails loudly unless the
directory's ARMS.json arm record carries gas_vintage_dispatch=True.

x = national (TOTAL, grid plus self-generation) cumulative emissions `cum_emissions`, MtCO2; y = `npv_cost`,
billion USD. One series per financing arm (4), points ordered by cap fraction and joined; the uncapped anchor
(point_type == unconstrained_anchor, cap fraction blank) is plotted as a separate marker and included in the
dominance test, as in src/pareto_plot.py. Dominance is src/pareto_plot.flag_dominance on TOTAL emissions, within an
arm: j dominates i if cost_j <= cost_i and emissions_j <= emissions_i with at least one strict. Non-dominated
points are filled, dominated points hollow. Every plotted point is printed to stdout with its cap fraction and
dominance status. Outputs: figures/fig_4_10_cost_emissions_frontier.{pdf,png} (300 dpi).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pareto_plot import flag_dominance  # noqa: E402

SRC = ROOT / "results" / "pareto_frontier" / "arms_1d68be49" / "pareto_raw.csv"
ARMS = SRC.parent / "ARMS.json"
OUT = ROOT / "figures"

if "gas_vintage_dispatch=True" not in json.loads(ARMS.read_text())["arm_tag"]:
    sys.exit(f"FAIL: {ARMS} arm record does not carry gas_vintage_dispatch=True")

raw = pd.read_csv(SRC)
d = flag_dominance(raw)
assert len(d) == 128 and set(d["status"]) == {"optimal"}
d["cost_bn"] = d["npv_cost"] / 1e9
d["emis_mt"] = d["cum_emissions"] / 1e6
ARM_ORDER = ["public_only_tight", "eaas_tight", "public_only_moderate", "eaas_moderate"]
assert set(d["arm"]) == set(ARM_ORDER)

print(f"SOURCE {SRC.relative_to(ROOT)}  (arm record: gas_vintage_dispatch=True)")
print(f"{'arm':22s} {'cap_fraction':>12s} {'point_type':22s} {'cost bn USD':>12s} {'emissions MtCO2':>16s}  dominance (total emissions)")
for arm in ARM_ORDER:
    s = d[d["arm"] == arm].copy()
    s["_o"] = s["cap_fraction"].fillna(9.0)
    s = s.sort_values("_o", ascending=False)
    for _, r in s.iterrows():
        cf = "anchor" if pd.isna(r["cap_fraction"]) else f"{r['cap_fraction']:.4f}"
        st = "DOMINATED by cap " + ("anchor" if pd.isna(r["dominated_total_by"]) else f"{r['dominated_total_by']:.4f}") if r["dominated_total"] else "non-dominated"
        print(f"{arm:22s} {cf:>12s} {r['point_type']:22s} {r['cost_bn']:12.4f} {r['emis_mt']:16.3f}  {st}")
n_dom = int(d["dominated_total"].sum())
print(f"points printed: {len(d)}  dominated {n_dom}  non-dominated {len(d) - n_dom}")
for arm in ARM_ORDER:
    s = d[d["arm"] == arm]
    m = s.loc[s["emis_mt"].idxmin()]
    print(f"{arm}: {len(s)} points, {int(s['dominated_total'].sum())} dominated; minimum emissions {m['emis_mt']:.3f} Mt at cap fraction {m['cap_fraction']:.4f}, cost {m['cost_bn']:.3f} bn")

STY = {"public_only_tight": ("#c05621", "o", "-", "Public, tight capital"),
       "eaas_tight": ("#2b6cb0", "s", "--", "EaaS, tight capital"),
       "public_only_moderate": ("#b7791f", "^", "-", "Public, moderate capital"),
       "eaas_moderate": ("#2f855a", "D", "--", "EaaS, moderate capital")}
INK, MUTED, GRID = "#1f2933", "#52606d", "#e4e7eb"
fig, ax = plt.subplots(figsize=(8.4, 6), facecolor="white")
for arm in ARM_ORDER:
    col, mk, ls, lab = STY[arm]
    s = d[d["arm"] == arm]
    g = s[s["point_type"] != "unconstrained_anchor"].sort_values("cap_fraction", ascending=False)
    ax.plot(g["emis_mt"], g["cost_bn"], color=col, linestyle=ls, linewidth=1.2, zorder=2)
    nd, dm = g[~g["dominated_total"]], g[g["dominated_total"]]
    ax.scatter(nd["emis_mt"], nd["cost_bn"], s=22, marker=mk, color=col, edgecolor="white", linewidth=0.6, zorder=3, label=lab)
    ax.scatter(dm["emis_mt"], dm["cost_bn"], s=26, marker=mk, facecolor="white", edgecolor=col, linewidth=1.1, zorder=3)
a = d[d["point_type"] == "unconstrained_anchor"]
ax.scatter(a["emis_mt"], a["cost_bn"], s=60, marker="*", color=INK, zorder=4, label="Uncapped anchor")
ax.scatter([], [], s=26, marker="o", facecolor="white", edgecolor=MUTED, linewidth=1.1, label="Dominated on total emissions (hollow)")
ax.set_xlabel("National cumulative emissions, 2025 to 2054 (MtCO$_2$, grid plus self-generation)", fontsize=9, color=MUTED)
ax.set_ylabel("Discounted system cost (billion USD)", fontsize=9, color=MUTED)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
for s_ in ("left", "bottom"):
    ax.spines[s_].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=8.5)
ax.grid(True, color=GRID, linewidth=0.7)
ax.set_axisbelow(True)
ax.legend(fontsize=8.5, frameon=False, loc="upper left")
fig.tight_layout()
OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_4_10_cost_emissions_frontier.{ext}", dpi=300)
print("wrote figures/fig_4_10_cost_emissions_frontier.pdf and .png")
