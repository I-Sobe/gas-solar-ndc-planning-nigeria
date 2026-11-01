"""Figure 4.1 -- capacity trajectory of the central solution, 2025-2045 (TASK 96 B, group 1).

SOURCE (one file): results/baseline_sliced/summary__661b03c1.json, decision_variables.solar_add_mw_by_year,
gas_add_mw_by_year, storage_add_mwh_by_year, cumulated to 2045. The summary carries its own 'gas_vintage_dispatch' flag, which is asserted (the script fails loudly if it is not True).
Three stacked panels (solar MW, gas MW, storage MWh), shared year axis: no second axis. 2046-2054 (end-effect
buffer) is omitted. Caption (for the chapter): stocks are the reported quantity; annual additions carry the
horizon-sensitivity bound of section 4.1.3 (timing moves 1.5-1.9% in L1 with the horizon end).
Every plotted datum is printed.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fig_common import ROOT, INK, MUTED, BLUE, ORANGE, GREEN, plt, rel, style, save

SRC = ROOT / "results" / "baseline_sliced" / "summary__661b03c1.json"
d = json.loads(SRC.read_text())
if d.get("gas_vintage_dispatch") is not True:
    sys.exit(f"FAIL: {SRC} does not carry gas_vintage_dispatch=True")
dv = d["decision_variables"]
years = list(range(2025, 2046))
series = {"solar (MW)": dv["solar_add_mw_by_year"], "gas (MW)": dv["gas_add_mw_by_year"], "storage (MWh)": dv["storage_add_mwh_by_year"]}
cum = {}
for k, s in series.items():
    run, out = 0.0, []
    for y in years:
        run += float(s[str(y)] if str(y) in s else s[y])
        out.append(run)
    cum[k] = out
print(f"SOURCE {rel(SRC)}  (gas_vintage_dispatch=True)")
print(f"{'year':>4s} " + " ".join(f"{k + ' add':>16s} {k + ' cum':>16s}" for k in series))
for i, y in enumerate(years):
    row = []
    for k, s in series.items():
        a = float(s[str(y)] if str(y) in s else s[y])
        row.append(f"{a:16.3f} {cum[k][i]:16.3f}")
    print(f"{y:4d} " + " ".join(row))
print(f"cumulative at 2045: solar {cum['solar (MW)'][-1]:.3f} MW (summary solar_cumulative_built_2045_mw {dv['solar_cumulative_built_2045_mw']:.3f}); "
      f"gas {cum['gas (MW)'][-1]:.3f} MW ({dv['gas_cumulative_built_2045_mw']:.3f}); storage {cum['storage (MWh)'][-1]:.3f} MWh ({dv['storage_capacity_2045_mwh']:.3f})")
assert abs(cum["solar (MW)"][-1] - dv["solar_cumulative_built_2045_mw"]) < 1e-3
assert abs(cum["gas (MW)"][-1] - dv["gas_cumulative_built_2045_mw"]) < 1e-3
assert abs(cum["storage (MWh)"][-1] - dv["storage_capacity_2045_mwh"]) < 1e-3

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7.6, 7), facecolor="white")
for ax, (k, col) in zip(axes, zip(series, (ORANGE, BLUE, GREEN))):
    ax.fill_between(years, cum[k], color=col, alpha=0.25, linewidth=0)
    ax.plot(years, cum[k], color=col, linewidth=1.6)
    ax.scatter([2045], [cum[k][-1]], s=22, color=col, zorder=3)
    ax.annotate(f"{cum[k][-1]:,.0f}", (2045, cum[k][-1]), textcoords="offset points", xytext=(-4, 6), ha="right", fontsize=8.5, color=INK)
    ax.set_ylabel(f"Cumulative {k}", fontsize=9, color=MUTED)
    style(ax)
axes[-1].set_xlabel("Year (reporting window to 2045; 2046-2054 end-effect buffer omitted)", fontsize=9, color=MUTED)
axes[-1].set_xlim(2025, 2045.4)
fig.tight_layout()
save(fig, "fig_4_1_capacity_trajectory")
