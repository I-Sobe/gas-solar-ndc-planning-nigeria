"""
FIN-2 omnibus diagnostic. Reads the two arm CSVs only. No solver.

Run from repo root:
    python scripts/diag_fin2.py

Tests H1..H5 in one pass. Paste the whole output.
"""
import pandas as pd
import numpy as np

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

U = "results/fin2/tariff_sweep_results__unbounded__df1.csv"
D = "results/fin2/tariff_sweep_results__dares_only__df1.csv"

u = pd.read_csv(U)
d = pd.read_csv(D)

KEY = ["blend_scenario", "tariff_m_usd_per_twh", "ndc_scenario"]
m = u.merge(d, on=KEY, suffixes=("_unb", "_dar"))
m["dcost"] = m.system_cost_npv_usd_dar - m.system_cost_npv_usd_unb
m["dcost_bn"] = m.dcost / 1e9
m["dmw"] = m.solar_eaas_total_mw_dar - m.solar_eaas_total_mw_unb
mc = m[m.concessional_share_unb > 0].copy()

print("=" * 78)
print("H1  NEGATIVE dcost -- adding a constraint cannot lower the objective")
print("=" * 78)
neg = mc[mc.dcost_bn < 0].copy()
print(f"negative cells: {len(neg)} of {len(mc)}")
if len(neg):
    neg["rel"] = neg.dcost / neg.system_cost_npv_usd_unb
    print("\nmagnitude of the violation:")
    print("  max |dcost| USD :", f"{neg.dcost.abs().max():,.2f}")
    print("  max relative    :", f"{neg.rel.abs().max():.3e}")
    print("  (relative < 1e-6 => solver tolerance, not a real violation)")
    cols = KEY + ["concessional_share_unb", "dcost", "dmw",
                  "status_unb", "status_dar",
                  "concessional_shadow_usd_per_usd_dar",
                  "solar_eaas_total_mw_unb", "solar_eaas_total_mw_dar"]
    print("\nall negative cells:")
    print(neg[cols].to_string(index=False))

print()
print("=" * 78)
print("H2  DEGENERACY -- is solar_eaas_total_mw uniquely identified?")
print("=" * 78)
# (a) same objective, different EaaS MW, across the two arms
tie = mc[(mc.dcost.abs() < 1.0) & (mc.dmw.abs() > 1.0)]
print(f"cells with equal objective (<1 USD) but different EaaS MW (>1 MW): {len(tie)} of {len(mc)}")
if len(tie):
    print(tie[KEY + ["dcost", "dmw"]].head(15).to_string(index=False))

# (b) monotonicity of MW in tariff within each cell
print("\nmonotonicity of solar_eaas_total_mw in tariff, per (blend, ndc) cell:")
rows = []
for arm, df in [("unbounded", u), ("dares_only", d)]:
    for (b, n), g in df.sort_values("tariff_m_usd_per_twh").groupby(["blend_scenario", "ndc_scenario"]):
        mw = g.solar_eaas_total_mw.values
        diffs = np.diff(mw)
        sign_changes = int((np.diff(np.sign(diffs[diffs != 0])) != 0).sum())
        rows.append(dict(arm=arm, blend=b, ndc=n, n=len(mw),
                         mw_min=round(mw.min(), 1), mw_max=round(mw.max(), 1),
                         sign_changes=sign_changes,
                         monotonic=(sign_changes == 0)))
mono = pd.DataFrame(rows)
print(f"  cells monotonic: {mono.monotonic.sum()} of {len(mono)}")
print(mono[~mono.monotonic].to_string(index=False))

print()
print("=" * 78)
print("H3  subsidy vs gap x MW -- is the reported gap a 2025 reference figure?")
print("=" * 78)
for arm, df in [("unbounded", u), ("dares_only", d)]:
    t = df.copy()
    t["implied"] = t.financing_gap_per_mw * t.solar_eaas_total_mw
    t["ratio"] = np.where(t.implied > 0, t.eaas_subsidy_npv_usd / t.implied, np.nan)
    print(f"\n[{arm}] subsidy / (gap_per_mw * MW):")
    print(t.ratio.describe().round(4).to_string())
    print("  rows where subsidy==0 but gap_per_mw>0 and MW>0:",
          int(((t.eaas_subsidy_npv_usd == 0) & (t.financing_gap_per_mw > 0) & (t.solar_eaas_total_mw > 0)).sum()))

print("\nzero_subsidy_reason vs actual gap (unbounded):")
z = u.groupby("zero_subsidy_reason", dropna=False).agg(
    n=("tariff_m_usd_per_twh", "size"),
    gap_min=("financing_gap_per_mw", "min"),
    gap_max=("financing_gap_per_mw", "max"),
    mw_min=("solar_eaas_total_mw", "min"),
    mw_max=("solar_eaas_total_mw", "max"),
).round(1)
print(z.to_string())
print("\n  gap_max > 0 under 'gap_closed' => label is derived from the 2025 reference")
print("  gap, not the LP's year-varying gap. Confirms or kills H3.")

print()
print("=" * 78)
print("H4  utilisation == 1.0 while shadow price == 0")
print("=" * 78)
dc = d[d.concessional_share > 0].copy()
print("utilisation unique values:", np.unique(dc.concessional_utilisation.round(9)))
zs = dc[dc.concessional_shadow_usd_per_usd.abs() < 1e-9]
print(f"rows with shadow ~ 0: {len(zs)} of {len(dc)}")
if len(zs):
    print("  their utilisation:", np.unique(zs.concessional_utilisation.round(9)))
    print("  -> utilisation 1.0 with zero shadow price is inconsistent;")
    print("     utilisation is likely computed as a forced ratio, not read from the solve.")
    print(zs[["blend_scenario", "tariff_m_usd_per_twh", "ndc_scenario",
              "concessional_utilisation", "concessional_shadow_usd_per_usd",
              "solar_eaas_total_mw"]].head(12).to_string(index=False))

print()
print("=" * 78)
print("H5  the headline -- shadow price and cost penalty of the DARES envelope")
print("=" * 78)
print("\nshadow price (USD system cost per marginal concessional USD), by blend:")
print(dc.groupby("blend_scenario").concessional_shadow_usd_per_usd
        .describe()[["count", "mean", "50%", "min", "max"]].round(2).to_string())
print("\ncost penalty (bn USD) of imposing the envelope, by blend and ndc:")
print(mc.groupby(["blend_scenario", "ndc_scenario"]).dcost_bn
        .agg(["mean", "min", "max"]).round(2).to_string())
print("\nEaaS MW displaced by the envelope, by blend:")
print(mc.groupby("blend_scenario").dmw.agg(["mean", "min", "max"]).round(0).to_string())

print()
print("=" * 78)
print("CONTEXT")
print("=" * 78)
print("rows:", len(u), len(d), "| blends:", sorted(u.blend_scenario.unique()))
print("ndc cases:", sorted(u.ndc_scenario.unique()))
print("tariffs:", sorted(u.tariff_m_usd_per_twh.unique()))
print("statuses:", u.status.value_counts().to_dict())
print("\ncolumns:", list(u.columns))
