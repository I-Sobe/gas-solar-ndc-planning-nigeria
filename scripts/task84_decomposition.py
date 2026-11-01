"""TASK 84 B.3: capital / fuel / O&M shares of the central cost, vintage (661b03c1) against blended (6046f759). Reads diagnostics only."""
import json
for lab,h in (("vintage 661b03c1","661b03c1"),("blended 6046f759","6046f759")):
    d=json.load(open(f"results/baseline_sliced/diagnostics__{h}.json"))["cost_decomposition"]; c=d["components"]; t=d["cost_total_npv"]
    cap=c["capital_charge_total_npv"]; fuel=c["gas_opex_npv"]; om=c["om_cost_npv"]
    print(f"{lab}: total {t/1e9:.4f} bn; capital {cap/1e9:.4f} ({100*cap/t:.2f}%), fuel {fuel/1e9:.4f} ({100*fuel/t:.2f}%), O&M {om/1e9:.4f} ({100*om/t:.2f}%); other {100*(t-cap-fuel-om)/t:.3f}%")
    print("   capital lines:", {k:round(v/1e9,4) for k,v in c.items() if k.endswith('capital_charge_npv') and k!='capital_charge_total_npv'})
