"""TASK 81 Batch 1: every LEVER_FAMILIES member + central, solved with 01b on the vintage model."""
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
arms = {}
def add(field, value):
    arm = dict(S.BASELINE_CENTRAL_ARM); om = S.BASELINE_CENTRAL_OM_CASE
    if field == "om_case": om = value
    else: arm[field] = value
    tag, _, h = B.arm_for(field, value); arms.setdefault(h, (arm, om, f"{field}={value}"))
add("conversion_case", "conv_central")
for f in S.LEVER_FAMILIES:
    for v in f["members"]: add(f["field"], v)
log = ROOT / "logs/batch1"; log.mkdir(parents=True, exist_ok=True)
rows = []
for h, (arm, om, label) in arms.items():
    cmd = [sys.executable, "scripts/01b_run_baseline_sliced.py"]
    for k, fl in FLAG.items(): cmd += [fl, arm[k]]
    cmd += ["--tariff-case", "tariff_central", "--us-inflation", "0.020", "--om-case", om]
    t = time.time(); r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    (log / f"{h}_{label.replace('=','-')}.log").write_text(r.stdout + r.stderr)
    rows.append(dict(hash=h, label=label, rc=r.returncode, sec=round(time.time() - t, 1)))
    print(rows[-1], flush=True)
json.dump(rows, open(log / "batch1_rows.json", "w"), indent=1)
