"""
51_check_advance_identity.py -- TASK 51 M1 regression requirement, WITHOUT a solve.

Builds the central-anchor model (scripts/01b_run_baseline_sliced.py's call,
arms from src/scenarios.BASELINE_CENTRAL_ARM) at a given capital_case and an
optional capital_window_case, writes the model as an LP file with symbolic
labels, and prints its md5. Two builds with the same md5 hand the solver the
same problem, so their solutions are bit-identical. No solve is run.

Usage (the pre-edit baseline is written with --window omitted, because the
kwarg does not exist before the arm is added):
    python scripts/51_check_advance_identity.py --out PATH.lp
    python scripts/51_check_advance_identity.py --out PATH.lp --window advance
"""
import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io import load_econ, load_solar_capex_by_year
from src.optimize_model_sliced import build_model_sliced
from src.scenarios import BASELINE_CENTRAL_ARM, load_scenario


def build(capital_case, window=None):
    kw = dict(BASELINE_CENTRAL_ARM)
    kw["peak_multiple_case"] = kw.pop("peak_case")
    extra = {} if window is None else {"capital_window_case": window}
    scenario = load_scenario(
        demand_level_case="served", demand_case="organic_central",
        capital_case=capital_case, solar_build_case="deployment_unconstrained",
        land_case="loose", carbon_case="no_policy",
        start_year=2025, end_year=2054,
        reserve_margin="nirp_spinning", us_inflation=0.02,
        tariff_case="tariff_central", om_case="om_central", **kw, **extra,
    )
    econ = load_econ("voll_mid", genset_case="genset_central")
    years = scenario["years"]
    capex = load_solar_capex_by_year(
        scenario_name="solar_low", start_year=int(years[0]), end_year=int(years[-1]))
    return build_model_sliced(
        scenario=scenario, econ=econ, emissions_cap=None,
        emissions_cap_by_year=None, solar_capex_by_year=capex)


def report_instalments(capital_case):
    """TASK 51 amendment 1.1: instalment counts equal under advance and arrears."""
    ms = {w: build(capital_case, w) for w in ("advance", "arrears", "due_crf")}
    rs = 0.08
    bad = 0
    print(f"capital_case={capital_case}: instalment counts (arrears, advance) are equal by assertion in the builder;")
    for tr in ("conc", "comm"):
        for ch in ("solar", "storage", "gas"):
            cnt = ms["arrears"].capital_window_instalments[tr][ch]
            cadv = ms["advance"].capital_window_instalments[tr][ch]
            same = all(cnt[v][0] == cadv[v][0] == cnt[v][1] for v in cnt)
            vec = [cnt[v][0] for v in sorted(cnt)]
            print(f"  {tr}/{ch}: equal={same}  counts v=0..29: {vec}")
            bad += (not same)
            wa = ms["arrears"].capital_window_W[tr][ch]
            wd = ms["advance"].capital_window_W[tr][ch]
            wc = ms["due_crf"].capital_window_W[tr][ch]
            dev = max(abs(wa[v] - wd[v] / (1 + rs)) for v in wa)
            devc = max(abs(wc[v] - wd[v]) for v in wc)
            print(f"      max |W_arrears - W_advance/(1+r_s)| = {dev:.3e};  max |W_due_crf - W_advance| = {devc:.3e}")
            bad += (dev > 1e-12) + (devc > 0)
    print("RESULT:", "PASS" if bad == 0 else f"FAIL ({bad})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--capital-case", default="unconstrained")
    ap.add_argument("--window", default=None)
    ap.add_argument("--instalments", action="store_true",
                    help="build advance/arrears/due_crf, assert equal instalment counts "
                         "per tranche, channel and vintage, and W(arrears)=W(advance)/(1+r_s); no LP written")
    a = ap.parse_args()
    if a.instalments:
        return report_instalments(a.capital_case)
    if a.out is None:
        ap.error("--out is required unless --instalments")
    m = build(a.capital_case, a.window)
    m.write(a.out, io_options={"symbolic_solver_labels": True})
    h = hashlib.md5(Path(a.out).read_bytes()).hexdigest()
    print(f"{a.capital_case} window={a.window} md5={h} bytes={Path(a.out).stat().st_size}")


if __name__ == "__main__":
    main()
