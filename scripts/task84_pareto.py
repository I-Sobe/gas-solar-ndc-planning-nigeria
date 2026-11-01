"""TASK 84 C.3: the full frontier for all four arms from pareto_raw.csv (vintage arms_1d68be49; blended arms_92390a63 for comparison). Reads only."""
import pandas as pd, numpy as np, sys
def table(arm):
    d = pd.read_csv(f"results/pareto_frontier/{arm}/pareto_raw.csv")
    out = {}
    for a, g in d.groupby("arm", sort=False):
        g = g[g.status == "optimal"].copy()
        anchor = g[g.point_type.astype(str).str.contains("anchor")]
        pts = g[~g.point_type.astype(str).str.contains("anchor")].sort_values("cap_fraction", ascending=False).reset_index(drop=True)
        out[a] = (anchor, pts, g)
    return out
def nondom(g):
    """non-dominated on (npv_cost, cum_emissions), anchor included."""
    c = g.npv_cost.values; e = g.cum_emissions.values; keep = []
    for i in range(len(g)):
        dom = any((c[j] <= c[i] + 1e-9 and e[j] <= e[i] + 1e-9) and (c[j] < c[i] - 1e-9 or e[j] < e[i] - 1e-9) for j in range(len(g)) if j != i)
        keep.append(not dom)
    return np.array(keep)
if __name__ == "__main__":
    for arm in sys.argv[1:]:
        print("=====", arm)
        for a, (anchor, pts, g) in table(arm).items():
            i = pts.cum_emissions.idxmin(); r = pts.loc[i]
            c1 = pts[pts.cap_fraction == 1.0].iloc[0]
            nd = nondom(g)
            lo = pts.loc[min(i + 1, len(pts) - 1)]; hi = pts.loc[max(i - 1, 0)]
            print(f"{a}: n points {len(pts)} + anchor {len(anchor)}; anchor total {anchor.cum_emissions.iloc[0]/1e6:.1f} Mt cost {anchor.npv_cost.iloc[0]/1e9:.3f}; cap1.0 total {c1.cum_emissions/1e6:.1f} Mt cost {c1.npv_cost/1e9:.3f}")
            print(f"   MIN total {r.cum_emissions/1e6:.1f} Mt at cap_fraction {r.cap_fraction} (cost {r.npv_cost/1e9:.3f}); bracket {lo.cap_fraction} .. {hi.cap_fraction}; value at bracket ends {lo.cum_emissions/1e6:.1f} / {hi.cum_emissions/1e6:.1f}")
            t = pts.iloc[-1]; print(f"   at 0.01: total {t.cum_emissions/1e6:.1f} grid {(1-t.leakage_share)*t.cum_emissions/1e6:.1f} selfgen {t.leakage_share*t.cum_emissions/1e6:.1f} cost {t.npv_cost/1e9:.2f}; non-dominated incl anchor {int(nd.sum())} of {len(g)}")
            falling = pts[pts.cap_fraction >= r.cap_fraction]
            below1 = pts[(pts.cap_fraction < r.cap_fraction)]
            c1e = c1.cum_emissions; an = anchor.cum_emissions.iloc[0]
            cross1 = below1[below1.cum_emissions >= c1e].cap_fraction.max() if len(below1[below1.cum_emissions >= c1e]) else None
            cross2 = below1[below1.cum_emissions >= an].cap_fraction.max() if len(below1[below1.cum_emissions >= an]) else None
            print(f"   first grid point (descending) at/above cap-1.0 level: {cross1}; above anchor: {cross2}")
            print("   sample:", [(float(x.cap_fraction), round(x.cum_emissions/1e6,1), round(x.npv_cost/1e9,1), round(x.leakage_share,3)) for _, x in pts.iloc[[0,5,10,15,20,25,30]].iterrows()])
