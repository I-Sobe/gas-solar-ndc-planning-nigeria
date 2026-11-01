"""Figure 4.10 -- cost-emissions frontier (TASK 95 D.3, redesigned v2).

SOURCE (exactly one file): results/pareto_frontier/arms_1d68be49/pareto_raw.csv
(vintage). Fails loudly unless the directory's ARMS.json arm record carries
gas_vintage_dispatch=True.

WHAT CHANGED AGAINST v1, AND WHY.

1. TWO BASES. v1 plotted cumulative emissions over 2025-2054. Section 4.1.3
   reports emissions over 2025-2045, and section 4.4.2 is written on that
   basis, so a frontier on the horizon basis inside the same section compares
   two bases silently. This draws both: left panel the reporting window,
   right panel the full horizon. Window emissions are summed from the annual
   columns TASK 80 added to the capped runners; the script fails loudly if
   they are absent.
2. DOMINANCE IS COMPUTED SEPARATELY ON EACH BASIS. A point dominated on the
   horizon is not necessarily dominated on the window.
3. CAP FRACTIONS APPEAR ON THE FIGURE. v1 plotted a cap scan with no cap
   fraction anywhere, so the turning points the text quotes could not be
   located. Each arm's minimum-emissions point is annotated with its cap
   fraction, and the direction of tightening is marked.
4. THE LEGEND SAYS WHAT DOMINANCE MEANS rather than naming the test.

Every plotted point is printed to stdout on both bases, with its cap fraction
and dominance status, followed by a SUMMARY block whose lines fill the slots
in the rewritten section 4.4.5.

Outputs: figures/fig_4_10_cost_emissions_frontier.{pdf,png} (300 dpi).
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = ROOT / "results" / "pareto_frontier" / "arms_1d68be49" / "pareto_raw.csv"
ARMS = SRC.parent / "ARMS.json"
OUT = ROOT / "figures"
WINDOW_END = 2045
HORIZON_END = 2054

if "gas_vintage_dispatch=True" not in json.loads(ARMS.read_text())["arm_tag"]:
    sys.exit(f"FAIL: {ARMS} arm record does not carry gas_vintage_dispatch=True")

raw = pd.read_csv(SRC)
assert len(raw) == 128 and set(raw["status"]) == {"optimal"}, (len(raw), set(raw["status"]))

# ------------------------------------------------- the reporting window ---
pat = re.compile(r"^emissions_total_tco2_(\d{4})$")
annual = {int(pat.match(c).group(1)): c for c in raw.columns if pat.match(c)}
if not annual:
    sys.exit("FAIL: no emissions_total_tco2_<year> columns in pareto_raw.csv. "
             "TASK 80 Part D added the annual series to the capped runners; "
             "re-run 18_run_pareto_frontier.py so they are stored, or this "
             "figure cannot be drawn on the reporting window.")
missing = [y for y in range(2025, WINDOW_END + 1) if y not in annual]
if missing:
    sys.exit(f"FAIL: annual emissions columns missing for {missing}")

win_cols = [annual[y] for y in range(2025, WINDOW_END + 1)]
raw["emis_window_mt"] = raw[win_cols].sum(axis=1) / 1e6
raw["emis_horizon_mt"] = raw["cum_emissions"] / 1e6
raw["cost_bn"] = raw["npv_cost"] / 1e9

# cross-check: the stored cumulative should equal the summed annual series
full_cols = [annual[y] for y in sorted(annual) if y <= HORIZON_END]
chk = (raw[full_cols].sum(axis=1) / 1e6 - raw["emis_horizon_mt"]).abs().max()
print(f"CHECK summed annual series against stored cumulative: max abs diff {chk:.6f} Mt")


def flag_dominance(df, emis_col):
    """Within an arm: j dominates i if cost_j <= cost_i and emissions_j <=
    emissions_i, with at least one strict. Returns a boolean Series and the
    cap fraction of a dominating point."""
    dom = pd.Series(False, index=df.index)
    by = pd.Series(pd.NA, index=df.index, dtype="object")
    for arm, s in df.groupby("arm"):
        for i, ri in s.iterrows():
            for j, rj in s.iterrows():
                if i == j:
                    continue
                le = rj["cost_bn"] <= ri["cost_bn"] and rj[emis_col] <= ri[emis_col]
                strict = rj["cost_bn"] < ri["cost_bn"] or rj[emis_col] < ri[emis_col]
                if le and strict:
                    dom[i] = True
                    by[i] = "anchor" if pd.isna(rj["cap_fraction"]) else f"{rj['cap_fraction']:.4f}"
                    break
    return dom, by


raw["dom_window"], raw["dom_window_by"] = flag_dominance(raw, "emis_window_mt")
raw["dom_horizon"], raw["dom_horizon_by"] = flag_dominance(raw, "emis_horizon_mt")

ARM_ORDER = ["public_only_tight", "eaas_tight", "public_only_moderate", "eaas_moderate"]
ARM_LABEL = {"public_only_tight": "Public financing, tight capital",
             "eaas_tight": "Service-financed, tight capital",
             "public_only_moderate": "Public financing, moderate capital",
             "eaas_moderate": "Service-financed, moderate capital"}
assert set(raw["arm"]) == set(ARM_ORDER)

# --------------------------------------------------------------- print ----
print(f"\nSOURCE {SRC.relative_to(ROOT)}  (arm record: gas_vintage_dispatch=True)")
print(f"{'arm':26s} {'cap_fraction':>12s} {'cost bn':>10s} "
      f"{'window Mt':>11s} {'horizon Mt':>11s}  window / horizon dominance")
for arm in ARM_ORDER:
    s = raw[raw["arm"] == arm].copy()
    s["_o"] = s["cap_fraction"].fillna(9.0)
    for _, r in s.sort_values("_o", ascending=False).iterrows():
        cf = "anchor" if pd.isna(r["cap_fraction"]) else f"{r['cap_fraction']:.4f}"
        w = f"dominated by {r['dom_window_by']}" if r["dom_window"] else "non-dominated"
        h = f"dominated by {r['dom_horizon_by']}" if r["dom_horizon"] else "non-dominated"
        print(f"{arm:26s} {cf:>12s} {r['cost_bn']:10.4f} "
              f"{r['emis_window_mt']:11.3f} {r['emis_horizon_mt']:11.3f}  {w} / {h}")

print(f"\npoints printed: {len(raw)}  "
      f"dominated on window {int(raw['dom_window'].sum())}  "
      f"dominated on horizon {int(raw['dom_horizon'].sum())}")

print("\n================ SUMMARY -- these lines fill section 4.4.5 ================")
anchor = raw[raw["point_type"] == "unconstrained_anchor"]
for arm in ARM_ORDER:
    s = raw[(raw["arm"] == arm) & (raw["point_type"] != "unconstrained_anchor")]
    a = anchor[anchor["arm"] == arm]
    for basis, col, dcol in (("window ", "emis_window_mt", "dom_window"),
                             ("horizon", "emis_horizon_mt", "dom_horizon")):
        m = s.loc[s[col].idxmin()]
        loosest = s.loc[s["cap_fraction"].idxmax()]
        nd = len(s) - int(s[dcol].sum())
        anc = f"{a[col].iloc[0]:.3f}" if len(a) else "n/a"
        print(f"{arm:26s} {basis}: turning point at cap fraction "
              f"{m['cap_fraction']:.4f}, minimum {m[col]:.3f} Mt; "
              f"loosest sampled cap {loosest['cap_fraction']:.4f} gives "
              f"{loosest[col]:.3f} Mt; uncapped anchor {anc} Mt; "
              f"{nd} of {len(s)} sampled points non-dominated")
grid = sorted(raw["cap_fraction"].dropna().unique(), reverse=True)
step = min(round(grid[i] - grid[i + 1], 6) for i in range(len(grid) - 1))
print(f"cap grid: {len(grid)} values from {max(grid):.4f} to {min(grid):.4f}, "
      f"smallest step {step:.4f}")
print("==========================================================================\n")

# ---------------------------------------------------------------- draw ----
STY = {"public_only_tight": ("#c05621", "o", "-"),
       "eaas_tight": ("#2b6cb0", "s", "--"),
       "public_only_moderate": ("#b7791f", "^", "-"),
       "eaas_moderate": ("#2f855a", "D", "--")}
INK, MUTED, GRID = "#1f2933", "#52606d", "#e4e7eb"

fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2), facecolor="white")
panels = ((axes[0], "emis_window_mt", "dom_window",
           f"Reporting window, 2025 to {WINDOW_END}"),
          (axes[1], "emis_horizon_mt", "dom_horizon",
           f"Full solved horizon, 2025 to {HORIZON_END}"))

for ax, ecol, dcol, title in panels:
    for arm in ARM_ORDER:
        col, mk, ls = STY[arm]
        s = raw[raw["arm"] == arm]
        g = s[s["point_type"] != "unconstrained_anchor"].sort_values(
            "cap_fraction", ascending=False)
        ax.plot(g[ecol], g["cost_bn"], color=col, linestyle=ls, linewidth=1.2, zorder=2)
        nd, dm = g[~g[dcol]], g[g[dcol]]
        ax.scatter(nd[ecol], nd["cost_bn"], s=24, marker=mk, color=col,
                   edgecolor="white", linewidth=0.6, zorder=3,
                   label=ARM_LABEL[arm] if ax is axes[0] else None)
        ax.scatter(dm[ecol], dm["cost_bn"], s=28, marker=mk, facecolor="white",
                   edgecolor=col, linewidth=1.1, zorder=3)
        # the turning point, annotated with its cap fraction
        m = g.loc[g[ecol].idxmin()]
        ax.annotate(f"cap {m['cap_fraction']:.3f}",
                    xy=(m[ecol], m["cost_bn"]),
                    xytext=(-6, -14), textcoords="offset points",
                    fontsize=8, color=col, ha="right",
                    arrowprops=dict(arrowstyle="-", color=col, linewidth=0.8))
        # the loosest sampled cap, so the direction of tightening is legible
        lo = g.loc[g["cap_fraction"].idxmax()]
        ax.annotate(f"cap {lo['cap_fraction']:.3f}",
                    xy=(lo[ecol], lo["cost_bn"]),
                    xytext=(8, 2), textcoords="offset points",
                    fontsize=8, color=col, ha="left")
    a = raw[raw["point_type"] == "unconstrained_anchor"]
    ax.scatter(a[ecol], a["cost_bn"], s=70, marker="*", color=INK, zorder=4,
               label="Uncapped anchor" if ax is axes[0] else None)
    ax.set_title(title, fontsize=10.5, color=INK, pad=10)
    ax.set_xlabel("National cumulative emissions (MtCO$_2$, grid plus "
                  "self-generation)", fontsize=9, color=MUTED)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

axes[0].set_ylabel("Discounted system cost (billion USD)", fontsize=9, color=MUTED)
axes[0].scatter([], [], s=28, marker="o", facecolor="white", edgecolor=MUTED,
                linewidth=1.1,
                label="Hollow: a looser cap costs less and emits less")
axes[0].legend(fontsize=8.5, frameon=False, loc="upper left")

fig.text(0.5, 0.015,
         "Each point is one cap fraction; tightening runs from right to left "
         "along each arm. Dominance is tested within an arm, separately on "
         "each basis.",
         ha="center", fontsize=8.5, color=MUTED, style="italic")
fig.tight_layout(rect=(0, 0.035, 1, 1))

OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_4_10_cost_emissions_frontier.{ext}", dpi=300)
print("wrote figures/fig_4_10_cost_emissions_frontier.pdf and .png")
