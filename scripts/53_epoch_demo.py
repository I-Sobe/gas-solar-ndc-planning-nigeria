"""
53_epoch_demo.py -- TASK 53 section 1: the regression demonstration for the fix epoch.

Prints, in one pass and with no solve beyond the two harness anchors:
  1. both anchors (regression S1 = pre_f1_overnight, central S2), objective and final storage,
     solved through the 52_invariants harness (declared anchor-scale harness runs; non-quotable
     except where a figure equals a stored anchor), plus the LP md5 of each configuration;
  2. the commands Zeus will issue next, each executed and its exit status reported
     (run_m1_sequence.sh through step 2, the --dry-run drivers, 51_check_advance_identity.py in
     each mode);
  3. the invariant suite tally (stages A, B, C) and the FAIL ids.

Usage: python scripts/53_epoch_demo.py [--label TEXT] [--skip-suite]
"""
import argparse
import importlib.util
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_harness():
    spec = importlib.util.spec_from_file_location("H", ROOT / "scripts" / "52_invariants.py")
    H = importlib.util.module_from_spec(spec)
    sys.modules["H"] = H
    spec.loader.exec_module(H)
    return H


def sh(cmd, cwd=ROOT):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, executable="/bin/bash")
    return r.returncode, (r.stdout + r.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="")
    ap.add_argument("--skip-suite", action="store_true")
    a = ap.parse_args()
    head = sh("git rev-parse --short HEAD")[1].strip()
    print(f"## regression demonstration {a.label} (HEAD {head}, dirty tracked: "
          f"{sh('git status --porcelain --untracked-files=no -- src scripts docs | wc -l')[1].strip()})")

    H = load_harness()
    print("\n### anchors (harness, 2025-2054, CBC; LP md5 with symbolic labels)")
    print("| configuration | objective (USD) | final storage 2054 (MWh) | LP md5 |\n|---|---|---|---|")
    for cfg in ("S1 regression anchor", "S2 central anchor"):
        x, err = H._solve_mut(cfg, None)
        if x is None:
            print(f"| {cfg} | ERROR {err} | | |")
            continue
        o, st = H._sig(x)
        print(f"| {cfg} | {o:.2f} | {st:.5f} | {H.lp_md5(x)} |")

    print("\n### next-invocation commands (exit status)")
    smp = tempfile.mkdtemp()
    m1 = (ROOT / "scripts" / "run_m1_sequence.sh").read_text().split('say "3. Solves"')[0]
    p = Path(smp) / "m1_to_step2.sh"
    p.write_text(m1)
    cmds = [
        ("run_m1_sequence.sh steps 1-2 (step 3 solves cut)", f"bash {p}"),
        ("run_horizon_rebuild.sh --dry-run", "bash scripts/run_horizon_rebuild.sh --dry-run"),
        ("run_capital_window.sh --dry-run", "bash scripts/run_capital_window.sh --dry-run"),
        ("run_storage_capex.sh --dry-run --with-flat", "bash scripts/run_storage_capex.sh --dry-run --with-flat"),
        ("run_sobol_clean.sh --dry-run", "bash scripts/run_sobol_clean.sh --dry-run"),
        ("run_emissions_baseline.sh --dry-run (TASK 53 section 4)", "bash scripts/run_emissions_baseline.sh --dry-run"),
        ("run_signature_s3.sh --dry-run (TASK 53 2.8)", "bash scripts/run_signature_s3.sh --dry-run"),
        ("run_phase_b.sh --dry-run (the re-run)", "bash scripts/run_phase_b.sh --dry-run"),
        ("51_check_advance_identity --instalments (unconstrained)", "python scripts/51_check_advance_identity.py --instalments --capital-case unconstrained"),
        ("51_check_advance_identity --instalments (moderate)", "python scripts/51_check_advance_identity.py --instalments --capital-case moderate"),
        ("51_check_advance_identity --out, no --window", f"python scripts/51_check_advance_identity.py --out {smp}/a.lp"),
        ("51_check_advance_identity --out --window advance", f"python scripts/51_check_advance_identity.py --out {smp}/b.lp --window advance"),
        ("51_check_advance_identity --out --window arrears", f"python scripts/51_check_advance_identity.py --out {smp}/c.lp --window arrears"),
        ("51_check_advance_identity --out --window due_crf", f"python scripts/51_check_advance_identity.py --out {smp}/d.lp --window due_crf"),
    ]
    print("| command | exit | last line |\n|---|---|---|")
    for name, c in cmds:
        rc, out = sh(c)
        last = out.strip().splitlines()[-1][:110] if out.strip() else ""
        print(f"| {name} | {rc} | {last} |")
        if rc:
            print(f"<!-- FULL OUTPUT of failing command:\n{out[-1500:]}\n-->")

    if not a.skip_suite:
        print("\n### invariant suite")
        for s in "ABC":
            rc, out = sh(f"python scripts/52_invariants.py --stage {s}")
            m = re.search(r"STAGE %s: (\d+) PASS, (\d+) FAIL, (\d+) SKIP" % s, out)
            fails = re.findall(r"^(\S+)\s+FAIL\b", out, flags=re.M)
            print(f"- stage {s}: {m.group(1) if m else '?'} PASS / {m.group(2) if m else '?'} FAIL / {m.group(3) if m else '?'} SKIP; FAIL ids: {', '.join(fails)}")


if __name__ == "__main__":
    main()
