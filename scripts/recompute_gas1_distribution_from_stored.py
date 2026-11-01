"""
TASK 28 Part B: redo F-30's Results 2 and 3 on the SLICED series, from
STORED per-year duals (results/gas1/gas_shadow_by_year.csv). No solve.
Writes results/gas1/gas_distribution_recomputed.csv and
results/gas1/gas_benchmark_exceedance_recomputed.csv (new files).
Exceedance is strict (> benchmark), counted over all 30 evaluable years,
matching 06_run_gas1_shadow_benchmarks.py's share_years_shadow_gt_benchmark.
"""
# TASK 53 NOTE: this script reads the PRE-TASK-53 stored schema (basis-less `gas_shadow_*` columns, which
# were present-value). It is correct for the pre-epoch result files it was written for and is left
# unchanged; on post-epoch files (`gas_shadow_pv_*` / `gas_shadow_yr_*`) it fails loudly, by design.

import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gas_benchmarks import BENCHMARKS, MMBTU_PER_TWH_TH

d = pd.read_csv(ROOT / "results/gas1/gas_shadow_by_year.csv")
dist, exc = [], []
for (ndc, gas), g in d.groupby(["ndc_scenario", "gas_case"]):
    v = np.sort(g["gas_shadow_usd_per_twh_th"].dropna().values)
    mm = v / MMBTU_PER_TWH_TH
    b = mm[v > 1e-6]
    gaps = np.diff(mm)
    k = int(np.argmax(gaps)) if len(gaps) else 0
    dist.append({"ndc_scenario": ndc, "gas_case": gas, "n_years": len(v),
                 "n_binding": int(len(b)),
                 "median_usd_mmbtu": float(np.median(mm)),
                 "all_year_mean_usd_mmbtu": float(mm.mean()),
                 "binding_year_mean_usd_mmbtu": float(b.mean()) if len(b) else None,
                 "max_usd_mmbtu": float(mm.max()),
                 "largest_gap_lo": float(mm[k]), "largest_gap_hi": float(mm[k + 1]),
                 "n_below_largest_gap": k + 1})
    for name, thr in BENCHMARKS.items():
        n = int((v > thr).sum())
        exc.append({"ndc_scenario": ndc, "gas_case": gas, "benchmark": name,
                    "usd_mmbtu": thr / MMBTU_PER_TWH_TH, "years_exceeding": n,
                    "share": n / len(v)})
D = pd.DataFrame(dist); E = pd.DataFrame(exc)
D.to_csv(ROOT / "results/gas1/gas_distribution_recomputed.csv", index=False)
E.to_csv(ROOT / "results/gas1/gas_benchmark_exceedance_recomputed.csv", index=False)
pd.set_option("display.width", 220)
print(D.round(3).to_string(index=False))
key = ["opp_7", "lng_netback_des_central", "lng_netback_hub_central", "lng_export_10",
       "lng_netback_des_low", "lng_netback_hub_low", "lng_netback_des_high", "lng_netback_hub_high"]
print(E[E.benchmark.isin(key)].pivot_table(index=["ndc_scenario", "gas_case"], columns="benchmark",
      values="years_exceeding")[key].to_string())
# cross-check against the stored summary
import json
S = json.load(open(ROOT / "results/gas1/gas1_summary.json"))
bad = 0
for (ndc, gas), g in E.groupby(["ndc_scenario", "gas_case"]):
    s = S[f"{ndc}__{gas}"]
    for name, field in (("opp_7", "exceeds_opp_7"), ("lng_export_10", "exceeds_lng_10"),
                        ("lng_netback_des_central", "exceeds_lng_netback_des_central"),
                        ("lng_netback_hub_central", "exceeds_lng_netback_hub_central")):
        if abs(g[g.benchmark == name].share.iloc[0] - s[field]) > 1e-12:
            bad += 1
print("mismatches vs gas1_summary.json:", bad)
