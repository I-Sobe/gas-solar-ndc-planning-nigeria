import json,glob,os,sys
rows=[]
for base in ("results/ndc","results/ndc_eaas"):
    for d in sorted(glob.glob(base+"/*/")):
        sp=d+"summary.json"
        if not os.path.exists(sp): continue
        s=json.load(open(sp)); t=s.get("arm_tag")
        if not t: continue
        if "capital_window_case=arrears" not in t: continue
        v="gas_vintage_dispatch=True" in t
        key=t.replace("_gas_vintage_dispatch=True","").replace("_gas_vintage_dispatch=False","")
        g=json.load(open(d+"diagnostics.json"))
        sh=g.get("carbon_shadow_yr_usd_per_tco2_by_year") or {}
        nb=sum(1 for x in sh.values() if x and abs(x)>1e-6)
        lk=g.get("leakage") or {}
        rows.append(dict(base=base,dir=d,v=v,key=key,case=s["cap_scenario"],npv=s["npv_total_cost_usd"],nb=nb,
          em=s["actual_emissions_tco2_total"],unserved=s["cumulative_unserved_twh"],ls=lk.get("leakage_share"),lf=lk.get("flag"),
          mt=os.path.getmtime(sp)))
by={}
for r in rows: by.setdefault((r["base"],r["key"],r["v"]),[]).append(r)
out=[]
for (b,k,v),L in by.items():
    if not v: continue
    r=max(L,key=lambda x:x["mt"]); bl=by.get((b,k,False))
    bl=max(bl,key=lambda x:x["mt"]) if bl else None
    out.append((b,r,bl))
for b,r,bl in sorted(out,key=lambda x:(x[0],x[1]["case"],x[1]["key"])):
    conv="conv_none" if "conv_none" in r["key"] else ("conv_central" if "conv_central" in r["key"] else "?")
    dl=("flat" if "gas_deliv_flat" in r["key"] else "partial" if "gmp_partial" in r["key"] else "base")
    print(b.split("/")[1],r["case"],conv,dl,r["dir"].split("/")[-2][-8:],f"{r['npv']/1e9:.4f}",f"emis {r['em']/1e6:.2f}Mt","bind_yrs",r["nb"],"| blended", (f"{bl['npv']/1e9:.4f} bind_yrs {bl['nb']}" if bl else "NONE"),"| leak",r["ls"],r["lf"])
print("----- C.2 year sets")
for b,r,bl in sorted(out,key=lambda x:(x[0],x[1]["case"],x[1]["key"])):
    if bl is None or "conv_none" in r["key"] or "flat" in r["key"] or "partial" in r["key"] or "gas_deliv" in r["key"].split("baseline")[0][-1:]: pass
    gv=json.load(open(r["dir"]+"diagnostics.json")).get("carbon_shadow_yr_usd_per_tco2_by_year",{})
    gb=json.load(open(bl["dir"]+"diagnostics.json")).get("carbon_shadow_yr_usd_per_tco2_by_year",{})
    V={y for y,x in gv.items() if x and abs(x)>1e-6}; Bl={y for y,x in gb.items() if x and abs(x)>1e-6}
    print(b.split("/")[1],r["case"],r["dir"].split("/")[-2][-8:],"vintage-only",sorted(V-Bl),"blended-only",sorted(Bl-V),"leak>0.10:",(r["ls"] or 0)>0.10)
