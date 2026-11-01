"""TASK 82 B.2/B.6: every LEVER_FAMILIES member + central, solved with 02_run_ndc_caps.py under ONE NDC 3.0 cap.

usage: run_capped_lever_arms.py {ndc3_conditional|ndc3_unconditional}
One 02_ invocation per distinct arm per cap (--cases names the single cap case), so a failed or infeasible
solve is attributable to one arm. Rows go to logs/task82/capped_<cap>_rows.json; per-arm logs beside it.
"""
import importlib.util, subprocess, sys, time, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src import scenarios as S
sp = importlib.util.spec_from_file_location("b", ROOT / "scripts/build_lever_ranking.py")
B = importlib.util.module_from_spec(sp); sp.loader.exec_module(B)
FLAG = {"peak_case": "--peak-case", "storage_cc_case": "--storage-cc-case", "gas_build_case": "--gas-build-case",
        "new_gas_deliverability": "--new-gas-deliverability", "gas_availability_case": "--gas-availability-case",
        "gas_retirement_case": "--gas-retirement-case", "gas_capex_case": "--gas-capex-case",
        "ccgt_efficiency_case": "--ccgt-efficiency-case", "capital_recovery_case": "--capital-recovery-case",
        "dscr_case": "--dscr-case", "capital_treatment_case": "--capital-treatment-case",
        "concessional_envelope_case": "--concessional-envelope-case",
        "gas_deliverability_case": "--gas-deliverability-case", "conversion_case": "--conversion-case",
        "hydro_case": "--hydro-case"}
cap = sys.argv[1]
CASE = sys.argv[2] if len(sys.argv) > 2 else None   # TASK 83 A.1: e.g. unconstrained -> --capital-case-override
assert cap in ("ndc3_conditional", "ndc3_unconditional"), cap
arms = {}
def add(field, value):
    arm = dict(S.BASELINE_CENTRAL_ARM); om = S.BASELINE_CENTRAL_OM_CASE
    if field == "om_case": om = value
    else: arm[field] = value
    tag, _, h = B.arm_for(field, value); arms.setdefault(h, (arm, om, f"{field}={value}"))
add("conversion_case", "conv_central")
for f in S.LEVER_FAMILIES:
    for v in f["members"]: add(f["field"], v)
log = ROOT / "logs/task82"; log.mkdir(parents=True, exist_ok=True)
rows = []; t0 = time.time()
for h, (arm, om, label) in arms.items():
    cmd = [sys.executable, "scripts/02_run_ndc_caps.py", "--cases", cap] + (["--capital-case-override", CASE] if CASE else [])
    for k, fl in FLAG.items(): cmd += [fl, arm[k]]
    cmd += ["--tariff-case", "tariff_central", "--us-inflation", "0.020", "--om-case", om]
    t = time.time(); r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    (log / f"capped{'_' + CASE if CASE else ''}_{cap}_{h}_{label.replace('=','-')}.log").write_text(r.stdout + r.stderr)
    rows.append(dict(uncapped_hash=h, label=label, cap=cap, rc=r.returncode, sec=round(time.time() - t, 1),
                     err=(r.stderr.strip().splitlines() or [""])[-1][:300] if r.returncode else ""))
    print(rows[-1], flush=True)
json.dump(dict(cap=cap, wall_s=round(time.time() - t0, 1), rows=rows), open(log / f"capped{'_' + CASE if CASE else ''}_{cap}_rows.json", "w"), indent=1)
print("wall_s", round(time.time() - t0, 1), "solves", len(rows), "failed", sum(r["rc"] != 0 for r in rows))
