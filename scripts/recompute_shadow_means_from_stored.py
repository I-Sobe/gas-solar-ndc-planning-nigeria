"""
TASK 27 Part A.2: recompute shadow-price means on both bases from STORED
output. No solve. See docs/task_log_27a_gas_shadow_fix.md.

GAS-1 (exact): results/gas1/gas_shadow_by_year.csv holds the per-year duals.
GAS-2, GAS-3 (derived): only the old all-year mean and the binding count are
stored, so binding_mean = old_mean * N_evaluable / n_binding, with
N_evaluable = 30 (the model horizon; asserted for GAS-1 where it can be
checked). Valid because non-binding years contribute <= 1e-6 each.
Outputs are NEW files (the stored result files are not modified).
"""
# TASK 53 NOTE: this script reads the PRE-TASK-53 stored schema (basis-less `gas_shadow_*` columns, which
# were present-value). It is correct for the pre-epoch result files it was written for and is left
# unchanged; on post-epoch files (`gas_shadow_pv_*` / `gas_shadow_yr_*`) it fails loudly, by design.

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-6

# --- GAS-1 exact ----------------------------------------------------------
by_year = pd.read_csv(ROOT / "results/gas1/gas_shadow_by_year.csv")
old = pd.read_csv(ROOT / "results/gas1/gas_shadow_matrix.csv")
rows = []
for (gas, ndc), g in by_year.groupby(["gas_case", "ndc_scenario"]):
    v = g["gas_shadow_usd_per_twh_th"].dropna().values
    b = v[v > TOL]
    rows.append({
        "gas_case": gas, "ndc_scenario": ndc,
        "gas_shadow_all_year_mean": float(v.mean()),
        "gas_shadow_binding_mean": float(b.mean()) if len(b) else None,
        "gas_shadow_binding_years": int(len(b)),
        "gas_shadow_evaluable_years": int(len(v)),
    })
new = pd.DataFrame(rows)
chk = new.merge(old, on=["gas_case", "ndc_scenario"])
assert (chk.gas_shadow_evaluable_years == 30).all()
assert np.allclose(chk.gas_shadow_all_year_mean, chk.gas_shadow_mean, rtol=1e-9), "all-year mean != stored gas_shadow_mean"
assert (chk.gas_shadow_binding_years_x == chk.gas_shadow_binding_years_y).all(), "binding count != stored"
new["dilution_all_year_over_binding"] = new.gas_shadow_all_year_mean / new.gas_shadow_binding_mean
new.to_csv(ROOT / "results/gas1/gas_shadow_binding_recomputed.csv", index=False)
print("GAS-1 (exact, from per-year duals):")
print(new.round(4).to_string(index=False))

# --- GAS-2 derived --------------------------------------------------------
g2 = pd.read_csv(ROOT / "results/gas2/eaas_vs_noeaas_gas_shadow.csv")
N = 30
for pre, nb in (("no_eaas", "no_eaas_binding_years"), ("eaas", "eaas_binding_years")):
    g2[f"{pre}_gas_shadow_binding_mean_derived"] = np.where(
        g2[nb] > 0, g2[f"{pre}_gas_shadow_mean"] * N / g2[nb].replace(0, np.nan), np.nan)
g2["relief_pct_all_year"] = g2["gas_shadow_relief_pct"]
g2["relief_pct_binding_mean_derived"] = (
    (g2.no_eaas_gas_shadow_binding_mean_derived - g2.eaas_gas_shadow_binding_mean_derived)
    / g2.no_eaas_gas_shadow_binding_mean_derived * 100)
g2["binding_counts_equal"] = g2.no_eaas_binding_years == g2.eaas_binding_years
g2.to_csv(ROOT / "results/gas2/gas_shadow_binding_derived.csv", index=False)
priced = g2[(g2.no_eaas_binding_years > 0) | (g2.eaas_binding_years > 0)]
print("\nGAS-2 priced pairs (derived):", len(priced),
      "| counts equal in all:", bool(priced.binding_counts_equal.all()),
      "| max |relief diff| (pct pts):",
      float((priced.relief_pct_binding_mean_derived - priced.relief_pct_all_year).abs().max()))

# --- GAS-3 derived (carbon + gas) ----------------------------------------
g3 = pd.read_csv(ROOT / "results/gas3/regime_ndc_matrix.csv")
g3["carbon_shadow_binding_mean_derived"] = np.where(
    g3.carbon_binding_years > 0, g3.carbon_shadow_mean * N / g3.carbon_binding_years.replace(0, np.nan), np.nan)
g3["gas_shadow_binding_mean_derived"] = np.where(
    g3.gas_shadow_binding_years > 0, g3.gas_shadow_mean * N / g3.gas_shadow_binding_years.replace(0, np.nan), np.nan)
g3.to_csv(ROOT / "results/gas3/shadow_binding_derived.csv", index=False)
print("\nGAS-3 carbon shadow, all-year -> binding (derived):")
print(g3[~g3.gas_case.str.startswith("flat")].pivot(index="ndc_scenario", columns="gas_case",
      values="carbon_shadow_binding_mean_derived").round(1).to_string())
