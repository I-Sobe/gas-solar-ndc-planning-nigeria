"""Figure 4.7 -- the thirty-six-cell separation (TASK 95 D.2, redesigned v2).

SOURCE (exactly one file): results/dem2/arms_1fa34539/dem2_results.csv (the
vintage DEM-2 run named in F-38's re-derived block; F-38's own Source line
names the fixed-name results/dem2/eaas_value.csv, which is overwritten by
later runs). Fails loudly unless the directory's ARMS.json arm record carries
gas_vintage_dispatch=True.

One point per configuration (demand case x gas case x policy arm = 36):
  budget dual = budget_shadow of the public_only row (USD per USD of budget)
  EaaS value  = npv_total_cost_usd(public_only) - npv_total_cost_usd(eaas) [USD]

WHAT THIS FIGURE SHOWS. The finding is a GAP: no configuration lies between
the zero-dual group's maximum and the positive-dual group's minimum. The two
groups are drawn as columns on one symmetric-log axis so the empty interval
between them occupies the middle of the frame. Both bounds are drawn and
labelled inside that empty interval, where nothing is plotted, so no label
collides with data.

Changes against v1: labels moved into the gap; a legend added; the gap's
extent stated in orders of magnitude rather than as a raw ratio; shading
confined to the interval and drawn behind the grid.

Every plotted datum is printed to stdout.
Outputs: figures/fig_4_7_budget_dual_separation.{pdf,png} (300 dpi).
"""
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
ratio = p_min / abs(z_max)
decades = math.log10(ratio)

print(f"points printed: {len(pts)}  zero-dual {len(Z)}  positive-dual {len(P)}")
print(f"zero-dual group: max EaaS value {z_max:.4f} USD, min {z_min:.4f} USD")
print(f"positive-dual group: min EaaS value {p_min:,.1f} USD, max {p_max:,.1f} USD; "
      f"dual min {min(p[1] for p in P):.6f}, max {max(p[1] for p in P):.6f}")
print(f"GAP: no configuration between {z_max:.4f} USD and {p_min:,.1f} USD "
      f"-- ratio {ratio:,.4g}, {decades:.2f} orders of magnitude")

# ---------------------------------------------------------------- draw ----
INK, MUTED, GRID = "#1f2933", "#52606d", "#e4e7eb"
C_ZERO, C_POS = "#b7791f", "#2b6cb0"
SHADE = "#eef2f6"

fig, ax = plt.subplots(figsize=(8.0, 6.8), facecolor="white")


def column(group, centre, half_width=0.16):
    """Deterministic horizontal spread: points ordered by value, fanned out
    evenly across the column so every one of the eighteen is countable."""
    g = sorted(group, key=lambda p: p[2])
    n = len(g)
    if n == 1:
        return [centre], [g[0][2]]
    xs = [centre - half_width + 2 * half_width * i / (n - 1) for i in range(n)]
    return xs, [p[2] for p in g]


ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=0.9)
ax.set_ylim(z_min * 5, p_max * 4)
ax.set_xlim(-0.62, 1.58)

# the empty interval, shaded, behind the grid
ax.axhspan(z_max, p_min, color=SHADE, zorder=0)
ax.grid(True, axis="y", color=GRID, linewidth=0.7, zorder=1)
ax.set_axisbelow(True)

# the two bounds -- these are the result
ax.axhline(z_max, color=C_ZERO, linewidth=1.3, linestyle="--", zorder=2)
ax.axhline(p_min, color=C_POS, linewidth=1.3, linestyle="--", zorder=2)

xz, yz = column(Z, 0.0)
xp, yp = column(P, 1.0)
ax.scatter(xz, yz, s=48, color=C_ZERO, edgecolor="white", linewidth=0.9,
           marker="s", zorder=4)
ax.scatter(xp, yp, s=48, color=C_POS, edgecolor="white", linewidth=0.9,
           marker="o", zorder=4)

# labels inside the empty interval, where nothing is plotted
ax.text(-0.55, p_min, f"  group minimum   {p_min:,.0f} USD",
        ha="left", va="top", fontsize=9, color=C_POS, zorder=5)
ax.text(-0.55, z_max, f"  group maximum   {z_max:,.2f} USD",
        ha="left", va="bottom", fontsize=9, color=C_ZERO, zorder=5)

# the gap, named, in the middle of the empty interval
gap_centre = 10 ** ((math.log10(p_min) + math.log10(LINTHRESH)) / 2)
ax.annotate(f"no configuration lies between these two values\n"
            f"a gap of {decades:.1f} orders of magnitude",
            xy=(0.75, gap_centre), ha="center", va="center",
            fontsize=10.5, color=MUTED, style="italic", zorder=5)

# ---------------------------------------------------------------- axes ----
ax.set_xticks([0.0, 1.0])
ax.set_xticklabels([f"budget does not bind\n(n = {len(Z)})",
                    f"budget binds\n(n = {len(P)})"],
                   fontsize=10, color=INK)
ax.set_ylabel("Value of the service-financed channel (USD)\n"
              "public-only system cost minus service-financed system cost",
              fontsize=9.5, color=MUTED)
ax.set_xlabel("Shadow price of the public capital budget",
              fontsize=9.5, color=MUTED, labelpad=10)
ax.axhline(0.0, color=GRID, linewidth=0.9, zorder=1)

legend = [
    Line2D([0], [0], marker="s", color="none", markerfacecolor=C_ZERO,
           markeredgecolor="white", markersize=8,
           label=f"shadow price = 0  ({len(Z)} configurations)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=C_POS,
           markeredgecolor="white", markersize=8,
           label=f"shadow price > 0  ({len(P)} configurations)"),
]
ax.legend(handles=legend, loc="upper left", fontsize=9, frameon=False,
          handletextpad=0.6, borderaxespad=0.8)

for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=8.5)
ax.tick_params(axis="x", length=0, pad=10)

fig.text(0.5, 0.005,
         "Vertical axis is a symmetric log scale: the band between "
         f"\u2212{LINTHRESH:.0f} and +{LINTHRESH:.0f} USD is linear.",
         ha="center", fontsize=8, color=MUTED, style="italic")

OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_4_7_budget_dual_separation.{ext}", dpi=300,
                bbox_inches="tight")
print("wrote figures/fig_4_7_budget_dual_separation.pdf and .png")
