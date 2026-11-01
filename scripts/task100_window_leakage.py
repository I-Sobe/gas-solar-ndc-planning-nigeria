"""TASK 100 Part A: the out-of-boundary share on the reporting window (2025-2045) beside the horizon-cumulative share (2025-2054).

READ-ONLY on stored output (no solve). The share is the one the diagnostic stores as `leakage_share`: self-generation emissions over total
emissions (register L150), here summed over a stated set of years. Inputs, all vintage (asserted):
  FIN-3 capped    results/fin3/arms_1d68be49/capital_sweep_results.csv   (columns emissions_{total,selfgen}_tco2_<year>)
  FIN-3 uncapped  results/fin3/arms_7f91f752/capital_sweep_results.csv
  lever-ranking   results/lever_ranking_capped/<regime>/arms_*/capped_arms.csv -> each arm's diagnostics.json (emissions_*_tco2_by_year)
                  and each arm's uncapped partner results/baseline_sliced/diagnostics__<uncapped_hash>.json (selfgen_twh_by_year, emissions_tco2_by_year)
  stored NDC runs the four central arms results/ndc/ndc3_{conditional,unconditional}_{4ca4832d,60585e1f,3fbc75ac,ae31b29c}/diagnostics.json
Writes docs/task_log_100_window_leakage.csv (one row per capped arm) and prints the tables.
"""
import csv
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Y0, YW, YH = 2025, 2045, 2054


def share(total, selfgen, last):
    g = lambda d, y: float(d[y] if y in d else d[str(y)])
    t = sum(g(total, y) for y in range(Y0, last + 1))
    s = sum(g(selfgen, y) for y in range(Y0, last + 1))
    return (s / t if t > 0 else 0.0), s, t


def vintage(path):
    tag = json.loads(Path(path).read_text())["arm_tag"]
    if "gas_vintage_dispatch=True" not in tag:
        sys.exit(f"FAIL: {path} not vintage")


rows = []
# ---- FIN-3 -------------------------------------------------------------------------------------------------------
for name, d in (("capped", "arms_1d68be49"), ("uncapped", "arms_7f91f752")):
    vintage(ROOT / "results" / "fin3" / d / "ARMS.json")
fin = {}
for name, d in (("capped", "arms_1d68be49"), ("uncapped", "arms_7f91f752")):
    for r in csv.DictReader(open(ROOT / "results" / "fin3" / d / "capital_sweep_results.csv")):
        tot = {y: float(r[f"emissions_total_tco2_{y}"]) for y in range(Y0, YH + 1)}
        sg = {y: float(r[f"emissions_selfgen_tco2_{y}"]) for y in range(Y0, YH + 1)}
        fin[(name, r["capital_case"], r["collection_rate_scenario"])] = (tot, sg, float(r["leakage_share"]))
print("TABLE A.1a  FIN-3 (vintage; NDC 3.0 unconditional cap, 15 cells). Shares in per cent.")
print(f"{'tier':14s} {'collection':>10s} {'cum 2025-54':>12s} {'window 2025-45':>15s} {'stored leakage_share':>21s} | {'uncapped cum':>13s} {'uncapped window':>16s} {'increment (window) pp':>22s}")
for t in ("tight", "moderate", "adequacy", "expansion", "unconstrained"):
    for k, lab in (("no_friction", "100%"), ("reform_target", "60%"), ("current", "35%")):
        tot, sg, stored = fin[("capped", t, k)]
        c, _, _ = share(tot, sg, YH)
        w, _, _ = share(tot, sg, YW)
        assert abs(c - stored) < 1e-9, (t, k, c, stored)
        u = fin.get(("uncapped", t, k))
        if u:
            uc, _, _ = share(u[0], u[1], YH); uw, _, _ = share(u[0], u[1], YW)
            us = f"{100 * uc:13.3f} {100 * uw:16.3f} {100 * (w - uw):22.3f}"
        else:
            us = f"{'NO CELL':>13s} {'NO CELL':>16s} {'n/a':>22s}"
        print(f"{t:14s} {lab:>10s} {100 * c:12.3f} {100 * w:15.3f} {100 * stored:21.3f} | {us}")
        rows.append(("fin3", f"{t}/{lab}", 100 * c, 100 * w))
nf_c = sum(1 for (n, t, k), v in fin.items() if n == "capped" and share(v[0], v[1], YH)[0] > 0.10)
nf_w = sum(1 for (n, t, k), v in fin.items() if n == "capped" and share(v[0], v[1], YW)[0] > 0.10)
print(f"FIN-3 cells above 10%: cumulative {nf_c} of 15; reporting window {nf_w} of 15")

# ---- lever-ranking capped arms ----------------------------------------------------------------------------------
print("\nTABLE A.1b / A.2  capped lever-ranking arms (58 per file). Counts of arms above 10%.")
print(f"{'file':50s} {'arms':>5s} {'cum >10%':>9s} {'window >10%':>12s} {'max cum %':>10s} {'max window %':>13s} {'uncapped partners with selfgen in window':>42s}")
lever = {}
for reg in ("ndc3_conditional", "ndc3_unconditional", "ndc3_conditional_noenvelope", "ndc3_unconditional_noenvelope"):
    f = glob.glob(str(ROOT / "results" / "lever_ranking_capped" / reg / "arms_*" / "capped_arms.csv"))[0]
    vintage(Path(f).parent / "ARMS.json")
    arms = list(csv.DictReader(open(f)))
    out = []
    unc_bad = set()
    for r in arms:
        dg = json.loads((ROOT / Path(r["capped_path"]).parent / "diagnostics.json").read_text())
        tot, sg = dg["emissions_total_tco2_by_year"], dg["emissions_selfgen_tco2_by_year"]
        c = share(tot, sg, YH)[0]; w = share(tot, sg, YW)[0]
        assert abs(c - float(r["leakage_share"])) < 1e-9, (reg, r["family"], r["member"])
        ud = json.loads((ROOT / "results" / "baseline_sliced" / f"diagnostics__{r['uncapped_hash']}.json").read_text())
        usel = sum(ud["selfgen_twh_by_year"][str(y)] if isinstance(ud["selfgen_twh_by_year"], dict) else ud["selfgen_twh_by_year"][y - Y0] for y in range(Y0, YW + 1))
        if usel > 1e-9:
            unc_bad.add(r["uncapped_hash"])
        out.append((r["family"], r["member"], c, w, r["uncapped_hash"], usel))
        rows.append((f"lever:{reg}", f"{r['family']}/{r['member']}", 100 * c, 100 * w))
    lever[reg] = out
    print(f"{reg + '/' + Path(f).parent.name:50s} {len(out):5d} {sum(o[2] > 0.10 for o in out):9d} {sum(o[3] > 0.10 for o in out):12d} {100 * max(o[2] for o in out):10.3f} {100 * max(o[3] for o in out):13.3f} {len(unc_bad):42d}")

# ---- stored NDC runs -----------------------------------------------------------------------------------------------
print("\nTABLE A.1c  the four stored-arm NDC runs (central arm of each lever-ranking file)")
print(f"{'run':44s} {'cum 2025-54 %':>14s} {'window 2025-45 %':>17s} {'stored share %':>15s}")
for reg, h in (("ndc3_conditional_noenvelope", "4ca4832d"), ("ndc3_unconditional_noenvelope", "60585e1f"), ("ndc3_conditional", "3fbc75ac"), ("ndc3_unconditional", "ae31b29c")):
    run = ROOT / "results" / "ndc" / f"{reg.replace('_noenvelope', '')}_{h}"
    dg = json.loads((run / "diagnostics.json").read_text())
    vintage_ok = "gas_vintage_dispatch=True" in dg["arm_tag"]
    assert vintage_ok, run
    tot, sg = dg["emissions_total_tco2_by_year"], dg["emissions_selfgen_tco2_by_year"]
    c = share(tot, sg, YH)[0]; w = share(tot, sg, YW)[0]
    print(f"{'results/ndc/' + run.name:44s} {100 * c:14.3f} {100 * w:17.3f} {100 * dg['leakage']['leakage_share'] if 'leakage_share' in dg['leakage'] else float('nan'):15.3f}")
    rows.append(("ndc", run.name, 100 * c, 100 * w))

with open(ROOT / "docs" / "task_log_100_window_leakage.csv", "w", newline="") as fh:
    w_ = csv.writer(fh); w_.writerow(["set", "arm", "cumulative_2025_2054_pct", "window_2025_2045_pct"])
    for r in rows:
        w_.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}"])
print(f"\nwrote docs/task_log_100_window_leakage.csv ({len(rows)} rows)")
