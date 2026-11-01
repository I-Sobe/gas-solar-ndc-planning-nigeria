"""TASK 81 Batch 3 report: per-runner row counts, status, objective range, leakage shares (vintage generation)."""
import pandas as pd, glob
G = 1e9
def rng(s): s = s.dropna(); return f"{s.min()/G:.4f} .. {s.max()/G:.4f}" if len(s) else "n/a"
def leak(d):
    if "leakage_share" not in d: return "no leakage column"
    s = d["leakage_share"].dropna(); f = int((s > 0.10).sum())
    return f"{len(s)} rows with share; max {s.max():.4f}; flagged {f}"
specs = [
 ("15_ pol1", "results/pol1/arms_1d68be49/pol1_results.csv", "npv_total_cost_usd", "status"),
 ("05_ fin3", "results/fin3/arms_1d68be49/capital_sweep_results.csv", None, "status"),
 ("04_ fin2", glob.glob("results/fin2/arms_6652884a/tariff_sweep_results*.csv")[0], "system_cost_npv_usd", "status"),
 ("08_ gas3", "results/gas3/arms_b7144908/regime_ndc_matrix.csv", "npv_total_cost_usd", "status"),
 ("06_ gas1", "results/gas1/arms_f30c5990/gas_shadow_matrix.csv", "npv_total_cost_usd", None),
 ("13_ dem2", "results/dem2/arms_1fa34539/dem2_results.csv", "npv_total_cost_usd", "status"),
 ("09_ rel1 total/nameplate", "results/rel1/total/gas_avail_nameplate/feasibility_matrix.csv", "npv_cost", "status"),
 ("09_ rel1 total/observed", "results/rel1/total/gas_avail_observed/feasibility_matrix.csv", "npv_cost", "status"),
 ("09_ rel1 annual/nameplate", "results/rel1/annual/gas_avail_nameplate/feasibility_matrix.csv", "npv_cost", "status"),
 ("09_ rel1 annual/observed", "results/rel1/annual/gas_avail_observed/feasibility_matrix.csv", "npv_cost", "status"),
 ("10_ rel2 nameplate", "results/rel2/arms_b7144908/gas_avail_nameplate/cost_curve_all.csv", "npv_cost", "status"),
 ("10_ rel2 observed", "results/rel2/arms_0e58beec/gas_avail_observed/cost_curve_all.csv", "npv_cost", "status"),
 ("11_ rel3 nameplate", "results/rel3/arms_1fa34539/gas_avail_nameplate/frontier_all.csv", "npv_cost", "status"),
 ("11_ rel3 observed", "results/rel3/arms_1e01828a/gas_avail_observed/frontier_all.csv", "npv_cost", "status"),
]
for name, p, obj, st in specs:
    d = pd.read_csv(p)
    sts = d[st].value_counts().to_dict() if st else "n/a"
    o = rng(d[obj]) if obj else "n/a"
    print(f"| {name} | {len(d)} | {sts} | {o} | {leak(d)} | `{p}` |")
