"""TASK 81 Part D: blended vs vintage, read from stored files. No interpretation."""
import json, glob, os
import pandas as pd
G = 1e9
rows = []
def add(q, b, v, fmt="{:.4f}", scale=G, src=""):
    if b is None or v is None: rows.append((q, "n/a" if b is None else fmt.format(b/scale), "n/a" if v is None else fmt.format(v/scale), "", src)); return
    rows.append((q, fmt.format(b/scale), fmt.format(v/scale), f"{(v-b)/scale:+.4f} ({100*(v-b)/b:+.2f}%)" if b else "", src))
def js(p): return json.load(open(p))
B = js("results/baseline_sliced/summary__6046f759.json"); V = js("results/baseline_sliced/summary__661b03c1.json")
src = "`results/baseline_sliced/summary__{6046f759 blended, 661b03c1 vintage}.json`"
add("central objective (bn USD)", B["npv_total_cost_usd"], V["npv_total_cost_usd"], src=src)
for y in (2045, 2054):
    for k, n, u in (("solar_cumulative_built_%d_mw", "solar stock %d (MW)", 1), ("gas_cumulative_built_%d_mw", "gas stock %d (MW)", 1), (None, "storage stock %d (MWh)", 1)):
        kk = (k % y) if k else "storage_capacity_%d_mwh" % y
        add(n % y, B["decision_variables"][kk], V["decision_variables"][kk], fmt="{:,.2f}", scale=1, src=src)
add("cumulative emissions 2025-2045 (MtCO2)", B["cumulative_emissions_2025_2045_tco2"], V["cumulative_emissions_2025_2045_tco2"], scale=1e6, fmt="{:.2f}", src=src)
add("cumulative emissions 2025-2054 (MtCO2)", B["cumulative_emissions_2025_2054_tco2"], V["cumulative_emissions_2025_2054_tco2"], scale=1e6, fmt="{:.2f}", src=src)
for nm, k in (("gross capex, report window", "npv_gross_capex_report_window_usd"), ("gross capex, full horizon", "npv_gross_capex_full_horizon_usd")):
    add(nm + " (bn USD)", B[k], V[k], src=src)
    rows.append((nm + " share of objective", f"{100*B[k]/B['npv_total_cost_usd']:.2f}%", f"{100*V[k]/V['npv_total_cost_usd']:.2f}%", "", src))
# NDC: pair vintage and blended files by arm tag minus the flag
seen_ndc = set()
def ndc_pairs(base):
    out = {}
    for d in glob.glob(base + "/*/"):
        sp = d + "summary.json"
        if not os.path.exists(sp): continue
        s = js(sp); t = s.get("arm_tag")
        if not t or "capital_window_case=arrears" not in t: continue
        v = "gas_vintage_dispatch=True" in t
        key = (t.replace("_gas_vintage_dispatch=True", ""), v)
        if key not in out or os.path.getmtime(sp) > out[key][1]: out[key] = (d, os.path.getmtime(sp), s)
    return out
for base, lab in (("results/ndc", "02_"), ("results/ndc_eaas", "03_")):
    P = ndc_pairs(base)
    for (k, v), (d, _, s) in sorted(P.items(), key=lambda x: x[0][0]):
        if not v or "gas_deliv" in k and False: continue
        if "conv_central" not in k: continue
        if (k, False) not in P: continue
        db, _, sb = P[(k, False)]
        if "ndc3_conditional_decline" in k and False: pass
        case = s["cap_scenario"]
        if (lab, case) in seen_ndc: continue   # gas_deliv flat/partial cells repeat the baseline-deliverability result (Batch 2)
        seen_ndc.add((lab, case))
        add(f"{lab} {case} objective (bn USD)", sb["npv_total_cost_usd"], s["npv_total_cost_usd"], src=f"`{d}summary.json` vs `{db}summary.json`")
        def lk(dd):
            g = js(dd + "diagnostics.json")
            if "emissions_selfgen_tco2_by_year" in g and "emissions_total_tco2_by_year" in g:
                a = sum(g["emissions_selfgen_tco2_by_year"].values()); b = sum(g["emissions_total_tco2_by_year"].values()); return a / b
        lb, lv = lk(db), lk(d)
        f4 = lambda x: "not stored" if x is None else f"{x:.4f}"
        rows.append((f"{lab} {case} leakage share (selfgen / total, derived)", f4(lb), f4(lv), "" if lb is None or lv is None else f"{lv-lb:+.4f}", f"`{d}diagnostics.json` vs `{db}diagnostics.json`"))
# lever spans
cb = pd.read_csv("results/lever_ranking/arms_6046f759/lever_ranking.csv").set_index("id")
cv = pd.read_csv("results/lever_ranking/arms_661b03c1/lever_ranking.csv").set_index("id")
for i in cv.index:
    b = cb.loc[i, "col3_live_span_usd"] if i in cb.index else None
    if pd.isna(cv.loc[i, "col3_live_span_usd"]):
        rows.append((f"lever span col3 [{i}] (bn USD)", "n/a" if b is None or pd.isna(b) else f"{b/G:.4f}", "not ranked (L138)", "", "`results/lever_ranking/arms_{6046f759,661b03c1}/lever_ranking.csv`")); continue
    add(f"lever span col3 [{i}] (bn USD)" + ("" if i in cb.index and cb.loc[i, "role"] == cv.loc[i, "role"] else f" (blended role: {cb.loc[i,'role'] if i in cb.index else 'absent'}; vintage role: {cv.loc[i,'role']})"), b, cv.loc[i, "col3_live_span_usd"], src="`results/lever_ranking/arms_{6046f759,661b03c1}/lever_ranking.csv`")
for i in cb.index:
    if i not in cv.index: rows.append((f"lever span col3 [{i}]", f"{cb.loc[i,'col3_live_span_usd']/G:.4f}", "absent", "", "L138"))
# MC, VSS, Sobol, pareto-anchor
mb = pd.read_csv("results/monte_carlo/arms_be9472ef/monte_carlo_summary.csv").set_index("case"); mv = pd.read_csv("results/monte_carlo/arms_f30c5990/monte_carlo_summary.csv").set_index("case")
for c in mv.index:
    if c in mb.index: add(f"MC mean cost [{c}] (bn USD)", mb.loc[c, "cost_mean"], mv.loc[c, "cost_mean"], src="`results/monte_carlo/arms_{be9472ef,f30c5990}/monte_carlo_summary.csv`")
vb = js("results/vss_bound/vss_bound__d6f19a23.json"); vv = js("results/vss_bound/vss_bound__9e2eda70.json")
for k in ("ecp_free_usd", "ws_usd", "bound_upper_usd"): add(f"VSS {k} (bn USD)", vb.get(k), vv.get(k), src="`results/vss_bound/vss_bound__{d6f19a23,9e2eda70}.json`")
sb_ = js(glob.glob("results/parameter_uncertainty/arms_ce5387ec/summary_*.json")[0]); sv_ = js("results/parameter_uncertainty/arms_09bd6dbc/summary_n256_seed42.json")
add("Sobol npv mean (bn USD)", sb_["qoi_summary"]["npv_total_cost_usd"]["mean"], sv_["qoi_summary"]["npv_total_cost_usd"]["mean"], src="`results/parameter_uncertainty/arms_{ce5387ec,09bd6dbc}/summary_n256_seed42.json`")
rows.append(("Sobol first-order, gas_availability_factor (npv)", f"{sb_['sobol_indices']['first_order'][0][0]:.4f}", f"{sv_['sobol_indices']['first_order'][0][0]:.4f}", "", "same files"))
print("| quantity | blended | vintage | change | file |\n|---|---|---|---|---|")
for r in rows: print("| " + " | ".join(r) + " |")
