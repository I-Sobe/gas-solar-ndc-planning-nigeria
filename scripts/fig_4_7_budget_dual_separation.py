"""Figure 4.7 -- the thirty-six-cell separation (TASK 95 D.2, redesigned).

SOURCE (exactly one file): results/dem2/arms_1fa34539/dem2_results.csv (the
vintage DEM-2 run named in F-38's re-derived block; F-38's own Source line
names the fixed-name results/dem2/eaas_value.csv, which is overwritten by
later runs). Fails loudly unless the directory's ARMS.json arm record carries
gas_vintage_dispatch=True.

One point per configuration (demand case x gas case x policy arm = 36):
  budget dual = budget_shadow of the public_only row (USD per USD of budget)
  EaaS value  = npv_total_cost_usd(public_only) - npv_total_cost_usd(eaas) [USD]

WHAT THIS FIGURE SHOWS, AND WHY IT IS DRAWN THIS WAY. The finding is a GAP:
no configuration lies between the zero-dual group's maximum and the
positive-dual group's minimum. A scatter of value against the dual shows a
rising relationship instead, which the thesis does not claim. This draws the
two groups as columns on one symmetric-log axis, so the empty interval
between them occupies the middle of the frame and is the figure's subject.
The two bounds are drawn and labelled; the interval between them is shaded;
each column carries its count.

Every plotted datum is printed to stdout.
Outputs: figures/fig_4_7_budget_dual_separation.{pdf,png} (300 dpi).
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import SymmetricalLogLocator

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "dem2" / "arms_1fa34539" / "dem2_results.csv"
ARMS = SRC.parent / "ARMS.json"
OUT = ROOT / "figures"
ZERO_TOL = 1e-6
LINTHRESH = 10.0          # |y| below this is drawn linearly; the zero-dual
                          # group lies between about -6.6 and -0.6 USD

# ---------------------------------------------------------------- load ----
if "gas_vintage_dispatch=True" not in json.loads(ARMS.read_text())["arm_tag"]:
    sys.exit(f"FAIL: {ARMS} arm record does not carry gas_vintage_dispatch=True")

rows = list(csv.DictReader(open(SRC)))
key = lambda r: (r["demand_case"], r["demand_growth_rate"], r["gas_case"], r["policy_label"])
pub = {key(r): r for r in rows if r["financing_arm"] == "public_only"}
eaas = {key(r): r for r in rows if r["financing_arm"] == "eaas"}
assert len(pub) == 36 and set(pub) == set(eaas), (len(pub), len(eaas))

pts = []
for k in sorted(pub):
    assert pub[k]["status"] == "optimal" and eaas[k]["status"] == "optimal", k
    d = float(pub[k]["budget_shadow"] or 0.0)
    v = float(pub[k]["npv_total_cost_usd"]) - float(eaas[k]["npv_total_cost_usd"])
    pts.append((k, d, v, d <= ZERO_TOL))

# --------------------------------------------------------------- print ----
print(f"SOURCE {SRC.relative_to(ROOT)}  (arm record: gas_vintage_dispatch=True)")
print(f"{'#':>2s} {'demand_case':26s} {'growth':>6s} {'gas_case':16s} "
      f"{'policy':20s} {'budget dual':>12s} {'EaaS value USD':>18s} group")
for i, (k, d, v, z) in enumerate(pts, 1):
    print(f"{i:2d} {k[0]:26s} {k[1]:>6s} {k[2]:16s} {k[3]:20s} "
          f"{d:12.6f} {v:18.4f} {'zero-dual' if z else 'positive-dual'}")

Z = [p for p in pts if p[3]]
P = [p for p in pts if not p[3]]
z_max, z_min = max(p[2] for p in Z), min(p[2] for p in Z)
p_min, p_max = min(p[2] for p in P), max(p[2] for p in P)
decades = (p_min / abs(z_max)) if z_max != 0 else float("inf")

print(f"points printed: {len(pts)}  zero-dual {len(Z)}  positive-dual {len(P)}")
print(f"zero-dual group: max EaaS value {z_max:.4f} USD, min {z_min:.4f} USD")
print(f"positive-dual group: min EaaS value {p_min:,.1f} USD, max {p_max:,.1f} USD; "
      f"dual min {min(p[1] for p in P):.6f}, max {max(p[1] for p in P):.6f}")
print(f"GAP: no configuration between {z_max:.4f} USD and {p_min:,.1f} USD "
      f"-- a ratio of {decades:,.3g}, about {len(f'{int(abs(decades))}') - 1 + 1} orders of magnitude")

# ---------------------------------------------------------------- draw ----
INK, MUTED, GRID = "#1f2933", "#52606d", "#e4e7eb"
C_ZERO, C_POS = "#b7791f", "#2b6cb0"
SHADE = "#f0f4f8"

fig, ax = plt.subplots(figsize=(7.6, 6.6), facecolor="white")


def column(group, centre, half_width=0.17):
    """Deterministic horizontal spread: points ordered by value, fanned out
    evenly across the column so every one of the eighteen is countable."""
    g = sorted(group, key=lambda p: p[2])
    n = len(g)
    if n == 1:
        return [centre], [g[0][2]]
    xs = [centre - half_width + 2 * half_width * i / (n - 1) for i in range(n)]
    return xs, [p[2] for p in g]


# the gap, shaded, behind everything
ax.axhspan(z_max, p_min, color=SHADE, zorder=0)

xz, yz = column(Z, 0.0)
xp, yp = column(P, 1.0)
ax.scatter(xz, yz, s=46, color=C_ZERO, edgecolor="white", linewidth=0.9,
           marker="s", zorder=3)
ax.scatter(xp, yp, s=46, color=C_POS, edgecolor="white", linewidth=0.9,
           marker="o", zorder=3)

# the two bounds -- these are the result
ax.axhline(z_max, color=C_ZERO, linewidth=1.2, linestyle="--", zorder=2)
ax.axhline(p_min, color=C_POS, linewidth=1.2, linestyle="--", zorder=2)
ax.annotate(f"highest value where the budget does not bind:  {z_max:,.2f} USD",
            xy=(1.42, z_max), xytext=(1.42, z_max), ha="right", va="bottom",
            fontsize=8.5, color=C_ZERO)
ax.annotate(f"lowest value where the budget binds:  {p_min:,.0f} USD",
            xy=(1.42, p_min), xytext=(1.42, p_min), ha="right", va="bottom",
            fontsize=8.5, color=C_POS)

# the gap, named
ax.annotate("no configuration lies in this interval\n"
            f"(a ratio of about {decades:,.0f} to 1)",
            xy=(0.5, (abs(z_max) * p_min) ** 0.5), ha="center", va="center",
            fontsize=9, color=MUTED, style="italic")

# ---------------------------------------------------------------- axes ----
ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=0.9)
ax.set_ylim(z_min * 4, p_max * 3)
ax.set_xlim(-0.5, 1.5)
ax.set_xticks([0.0, 1.0])
ax.set_xticklabels([f"Public capital budget\ndoes not bind\n(dual = 0, n = {len(Z)})",
                    f"Public capital budget binds\n(dual > 0, n = {len(P)})"],
                   fontsize=9.5, color=INK)
ax.set_ylabel("Value of the service-financed channel\n"
              "public-only cost minus service-financed cost (USD)",
              fontsize=9.5, color=MUTED)
ax.yaxis.set_minor_locator(SymmetricalLogLocator(base=10.0, linthresh=LINTHRESH,
                                                 subs=(2, 3, 4, 5, 6, 7, 8, 9)))
ax.axhline(0.0, color=GRID, linewidth=0.9, zorder=1)
ax.text(-0.46, LINTHRESH, "symmetric log scale; the band around zero is linear",
        fontsize=7.8, color=MUTED, style="italic", va="bottom")

for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=8.5)
ax.tick_params(axis="x", length=0, pad=8)
ax.grid(True, axis="y", color=GRID, linewidth=0.7)
ax.set_axisbelow(False)

OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_4_7_budget_dual_separation.{ext}", dpi=300,
                bbox_inches="tight")
print("wrote figures/fig_4_7_budget_dual_separation.pdf and .png")
