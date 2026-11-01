"""Pareto dominance and the two-panel F-43 figure (TASK 30, Part B).

Pure functions on a table shaped like results/pareto_frontier/pareto_raw.csv,
so the runner and `scripts/redraw_pareto_figure.py` draw the same thing and
the figure can be redrawn from stored output with no solve.

Two emissions axes, deliberately distinct (F-43):
  grid-only  the quantity the cap constrains (emissions_capped_by_year)
  total      grid + self-generation, the physical quantity
`cum_emissions` in the runner's table is TOTAL. The grid-only cumulative is
`cum_emissions_grid_only` when the table has it; otherwise it is derived as
(1 - leakage_share) * cum_emissions, which is exact (leakage_share is
self-generation emissions over total, and F-43 checks it reproduces the
cumulative cap to <1.1e-8 wherever the cap binds in all years).

Dominance: j dominates i if cost_j <= cost_i and emissions_j <= emissions_i
with at least one strict. The uncapped anchor is a candidate dominator (and a
candidate dominated point), as in the TASK 29 Part 2 table.
"""
import numpy as np
import pandas as pd

ARM_STYLES = {
    "public_only_tight":    ("#FF5722", "o", "Public (tight capital)"),
    "eaas_tight":           ("#2196F3", "s", "EaaS (tight capital)"),
    "public_only_moderate": ("#FF9800", "^", "Public (moderate capital)"),
    "eaas_moderate":        ("#4CAF50", "D", "EaaS (moderate capital)"),
}

AXES = {"total": "cum_emissions", "grid_only": "_grid_emissions"}


def with_grid_emissions(df):
    out = df.copy()
    if "cum_emissions_grid_only" in out.columns and out["cum_emissions_grid_only"].notna().any():
        out["_grid_emissions"] = out["cum_emissions_grid_only"]
    else:
        out["_grid_emissions"] = (1.0 - out["leakage_share"]) * out["cum_emissions"]
    return out


def flag_dominance(df):
    """Return df (optimal rows only) with `dominated_total`, `dominated_grid_only`
    booleans and `dominated_total_by` / `dominated_grid_only_by` (cap_fraction of
    one dominator, NaN for the anchor). Dominance is within an arm."""
    d = with_grid_emissions(df[df["status"] == "optimal"]).reset_index(drop=True)
    for name, col in AXES.items():
        d[f"dominated_{name}"] = False
        d[f"dominated_{name}_by"] = np.nan
    for arm, idx in d.groupby("arm").groups.items():
        sub = d.loc[idx]
        for name, col in AXES.items():
            c = sub["npv_cost"].to_numpy()
            e = sub[col].to_numpy()
            for k, i in enumerate(sub.index):
                m = (c <= c[k]) & (e <= e[k]) & ((c < c[k]) | (e < e[k]))
                if m.any():
                    d.at[i, f"dominated_{name}"] = True
                    j = sub.index[np.argmax(m)]
                    d.at[i, f"dominated_{name}_by"] = sub.at[j, "cap_fraction"]
    return d


def dominated_points_table(flagged):
    """Rows for pareto_dominated_points.csv (total-emissions dominance, anchor
    included as dominator and candidate)."""
    rows = []
    for _, r in flagged[flagged["dominated_total"]].iterrows():
        dom = flagged[(flagged["arm"] == r["arm"])
                      & ((flagged["cap_fraction"] == r["dominated_total_by"])
                         | (flagged["cap_fraction"].isna() & pd.isna(r["dominated_total_by"])))]
        dom = dom.iloc[0]
        rows.append({
            "arm": r["arm"],
            "dominated_cap_fraction": r["cap_fraction"],
            "dominated_by_cap_fraction": r["dominated_total_by"],
            "dominated_cost_usd": r["npv_cost"],
            "dominated_emissions_tco2": r["cum_emissions"],
            "dominator_cost_usd": dom["npv_cost"],
            "dominator_emissions_tco2": dom["cum_emissions"],
        })
    return pd.DataFrame(rows)


DOMINATED_NOTE = (
    "A dominated point here is genuinely dominated on TOTAL emissions: tightening "
    "the grid-only cap moved load to self-generation, which sits outside the "
    "capped quantity, and self-generation rose by more than the capped quantity "
    "fell (F-43). It is not a solver-tolerance artefact and not a non-binding "
    "cap. On the grid-only axis the same points are non-dominated."
)


def draw_two_panel(flagged, path_png=None, path_pdf=None):
    import matplotlib
    import matplotlib.pyplot as plt
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for panel, ax, axis_name, col in ((0, ax_a, "grid_only", "_grid_emissions"),
                                      (1, ax_b, "total", "cum_emissions")):
        dom_col = f"dominated_{axis_name}"
        for arm, (color, marker, label) in ARM_STYLES.items():
            s = flagged[flagged["arm"] == arm].sort_values(col)
            nd = s[~s[dom_col]]
            dm = s[s[dom_col]]
            if len(nd):
                ax.plot(nd[col] / 1e6, nd["npv_cost"] / 1e9, f"{marker}-", color=color,
                        label=f"{label} ({len(nd)} of {len(s)} non-dominated)",
                        linewidth=2, markersize=5, alpha=0.9)
            if len(dm):
                ax.scatter(dm[col] / 1e6, dm["npv_cost"] / 1e9, marker=marker,
                           facecolors="none", edgecolors=color, alpha=0.6, s=36,
                           linewidths=1.1)
            a = s[s["cap_fraction"].isna()]
            if len(a):
                ax.scatter(a[col] / 1e6, a["npv_cost"] / 1e9, marker="*", s=200,
                           color=color, edgecolors="black", zorder=5)
        ax.set_xlabel("Cumulative " + ("grid-only" if panel == 0 else "total (grid + self-generation)")
                      + " emissions (MtCO2)")
        ax.grid(alpha=0.3)
    ax_a.set_ylabel("NPV system cost (USD billions)")
    ax_a.set_title("A. Grid-only emissions (the capped quantity)")
    ax_b.set_title("B. Total emissions (physical)")
    h, l = ax_b.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.text(0.5, -0.145, "Solid: non-dominated. Hollow, unconnected: dominated. Star: uncapped solve.",
             ha="center", fontsize=9)
    plt.tight_layout()
    if path_png:
        fig.savefig(path_png, dpi=200, bbox_inches="tight")
    if path_pdf:
        fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    return fig
