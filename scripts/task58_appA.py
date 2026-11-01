"""TASK 58 §3: words in the chapter that describe the sourcing of each parameter that also has an Appendix A row."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from task58_measure import load, classify, words, CH

R = [
 (1, "demand-growth income elasticity", [("3.2.3", 825, 837)], [("3.2.3", 839, 849)]),
 (2, "concessional rate and tenor", [("3.4.1", 1615, 1622), ("3.4.2", 1691, 1701), ("3.4.7", 2088, 2095)], []),
 (3, "commercial rate", [("3.4.1", 1615, 1622), ("3.4.7", 2075, 2086), ("3.4.7", 2097, 2103)], []),
 (4, "solar CAPEX basis (ATB vs IRENA)", [("3.3.4", 1354, 1371), ("3.3.4", 1373, 1387)], []),
 (6, "social discount rate (rows 6 and 15)", [("3.1.2", 74, 86), ("3.1.2", 116, 129), ("3.1.2", 141, 152)], []),
 (7, "within_block_peaking_factor", [("3.1.7", 567, 573), ("3.2.5", 999, 1006)], []),
 (8, "demand base 37.09 TWh", [("3.2.2", 722, 727)], [("3.2.2", 729, 742)]),
 (9, "peak-shape arms", [("3.2.5", 1008, 1022)], []),
 (10, "B_star", [("3.4.4", 1899, 1911)], [("3.4.4", 1913, 1920)]),
 (11, "power share of energy 2018", [("3.5.2", 2194, 2232)], [("3.5.2", 2234, 2249)]),
 (12, "conditional-to-unconditional multiplier", [("3.5.2", 2285, 2294)], []),
 (13, "dscr_case", [("3.4.5", 1970, 1980), ("3.4.5", 2005, 2019)], []),
 (14, "storage_cc_case", [("3.3.6", 1490, 1499)], []),
 (16, "demand_sigma (Monte Carlo)", [("3.7.5", 3343, 3351)], []),
 (20, "VOLL", [("3.1.6", 370, 373), ("3.7.5", 3327, 3334)], []),
 (21, "gas_avail_nameplate / availability arms (row 28 gas_avail_observed is within these ranges)", [("3.3.1", 1133, 1172), ("3.7.5", 3295, 3325)], []),
 (22, "concessional envelope env_dares", [("3.4.3", 1854, 1871)], [("3.4.3", 1860, 1863)]),
 (23, "gas_eta 0.287 and the 89.27 TWh_th anchor", [("3.3.1", 1121, 1131), ("3.3.2", 1234, 1243)], [("3.3.2", 1244, 1246)]),
 (26, "heating value 1,036 Btu/scf", [("3.3.2", 1238, 1240)], []),
 (27, "hydro fixed O&M (flag, no rate)", [("3.3.5", 1447, 1455), ("3.7.3", 3163, 3167)], []),
]
L = load(CH)
cls = classify(L)


def w(a, b):
    return sum(words(L[i]) for i in range(a - 1, b) if cls[i][0] in ("prose", "table", "code"))


print("| A row | parameter | chapter description (current-state sourcing) | words | superseded-state narration in the same place | words |")
print("|---|---|---|---|---|---|")
for row, name, cur, old in R:
    cs = "; ".join(f"§{s} {a}-{b}" for s, a, b in cur)
    os_ = "; ".join(f"§{s} {a}-{b}" for s, a, b in old) or "-"
    print(f"| {row} | {name} | {cs} | {sum(w(a,b) for s,a,b in cur)} | {os_} | {sum(w(a,b) for s,a,b in old) or '-'} |")
