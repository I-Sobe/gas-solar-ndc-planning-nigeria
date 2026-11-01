"""TASK 83 B.2: write the ARMS.json sidecar that 09_run_rel1_feasibility.py does not write, for the four vintage-generation outputs.

RETROFITTED, not written by the runner: the sidecar says so. The runner takes the standard central arms and passes no
gas_vintage_dispatch override, so at the run's HEAD (6fb1a79, scenarios.py line 2412: default True) the arm is the vintage central.
Evidence recorded in the file: the nameplate cells reproduce the vintage central objective (23,323,068,34x) and not the blended
(24,986,620,949); the observed cells are tied by the run (same driver, same HEAD, same minutes), not by an objective identity.
"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
tag = json.load(open("results/baseline_sliced/summary__661b03c1.json"))["arm_tag"]
for mode in ("total", "annual"):
    for gas in ("gas_avail_nameplate", "gas_avail_observed"):
        d = f"results/rel1/{mode}/{gas}"
        ev = ("baseline-gas cell objective 23,323,068,344.95 / ...343.73 USD = vintage central 23,323,068,340.46 within 5 USD; blended central is 24,986,620,949.18"
              if gas == "gas_avail_nameplate" else
              "no objective identity available (observed availability has no stored vintage counterpart at this capital case); tied to the nameplate run by driver log")
        json.dump({"arm_tag": tag.replace("gas_avail_nameplate", gas), "retrofitted_by": "scripts/retrofit_rel1_sidecars.py (TASK 83 B.2)",
                   "reliability_mode": mode, "run_log": f"logs/vintage_run/14_*_rel1_{mode}_{gas.replace('gas_avail_', '')}.log",
                   "run_head": "6fb1a7987aab3856ba52ad2e8f948908e9cb0974", "evidence": ev,
                   "note": "arm_tag is the vintage central's with the availability arm substituted; the matrix itself sweeps gas_case, solar_build_case and eps"},
                  open(f"{d}/ARMS.json", "w"), indent=2)
        print("wrote", d)
