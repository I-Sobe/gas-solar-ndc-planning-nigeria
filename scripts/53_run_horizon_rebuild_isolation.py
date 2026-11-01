"""
53_run_horizon_rebuild_isolation.py -- TASK 52 addendum B-3: one cell of the
horizon / in-horizon-rebuild isolation (the table behind L94 and invariant G1-3m).

Solves the sliced baseline ONCE at the standard arm set (central arms, traditional
regime, capital_case unconstrained) with a given horizon END YEAR and STORAGE LIFE,
and writes results/horizon_rebuild/summary__<hash8>.json with the plan to 2045
(solar, storage, gas additions by year), the objective, and, for the anchor cell
(end 2054, storage life 15), a comparison with the stored central anchor.

Eight cells, driven by scripts/run_horizon_rebuild.sh:
    (2045,15) (2050,15) (2051,15) (2052,15) (2053,15) (2054,15)   horizon sweep, real life
    (2045,30) (2054,30)                                           isolating pair: no storage rebuild
scripts/53_report_horizon_rebuild.py tabulates the divergence from the (2045, same life)
baseline.  The hash8 always covers the standard arms, end_year and storage_life.
Measurement runner: it changes no default and migrates nothing.
"""
import argparse
import json
import sys
from pathlib import Path

import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cli_arms import add_standard_arms, standard_arms_kwargs, complete_arm_tag
from src.io import load_econ, load_solar_capex_by_year
from src.optimize_model import solve_model
from src.optimize_model_sliced import build_model_sliced
from src.output_naming import append_index, hash8_of
from src.scenarios import load_scenario

RESULTS_DIR = ROOT / "results" / "horizon_rebuild"
ANCHOR_PATH = ROOT / "results" / "baseline_sliced" / "summary__6046f759.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_standard_arms(ap)
    ap.add_argument("--end-year", type=int, required=True, help="last model year (2045..2054)")
    ap.add_argument("--storage-life", type=float, required=True,
                    help="storage technical life in years; 15 is the central value, 30 puts no storage rebuild inside any horizon here")
    ap.add_argument("--print-path-only", action="store_true")
    a = ap.parse_args()
    std = standard_arms_kwargs(a)
    tag = "horizon__" + "__".join(f"{k}={v}" for k, v in sorted(std.items())) + f"__end_year={a.end_year}__storage_life={a.storage_life:g}"
    tag = complete_arm_tag(tag, a, in_tag=set(std), extra={"capital_case": "unconstrained"})   # TASK 53 2.5: window and storage-series defaults join the key
    h8 = hash8_of(tag)
    path = RESULTS_DIR / f"summary__{h8}.json"
    if a.print_path_only:
        print(path)
        return

    sc = load_scenario(demand_level_case="served", demand_case="organic_central", capital_case="unconstrained",
                       solar_build_case="deployment_unconstrained", land_case="loose", carbon_case="no_policy",
                       start_year=2025, end_year=a.end_year, **std)
    sc["asset_lifetimes"] = {**sc["asset_lifetimes"], "storage": float(a.storage_life)}
    econ = load_econ("voll_mid", genset_case="genset_central")
    years = [int(y) for y in sc["years"]]
    capex = load_solar_capex_by_year(scenario_name="solar_low", start_year=years[0], end_year=years[-1])
    m = build_model_sliced(scenario=sc, econ=econ, emissions_cap=None, emissions_cap_by_year=None, solar_capex_by_year=capex)
    st = solve_model(m, scenario=sc)
    if not st["optimal"]:
        raise RuntimeError(f"solve failed: {st}")
    T = range(len(years))
    v = lambda e: float(pyo.value(e))
    keep = [t for t in T if years[t] <= 2045]
    out = {
        "arm_tag": tag, "end_year": a.end_year, "storage_life": a.storage_life, "standard_arms": std,
        "npv_total_cost_usd": v(m.system_cost_npv),
        "years_to_2045": [years[t] for t in keep],
        "solar_add_mw": [v(m.solar_public_add[t]) + v(m.solar_eaas_add[t]) for t in keep],
        "storage_add_mwh": [v(m.storage_add[t]) for t in keep],
        "gas_add_mw": [v(m.gas_add[t]) for t in keep],
        "storage_rebuild_years_inside_horizon": {str(years[b]): [years[b] + int(round(a.storage_life)) * k for k in range(1, 40)
                                                                  if b + int(round(a.storage_life)) * k <= len(years) - 1] for b in keep if years[b] >= 2036 and years[b] <= 2039},
    }
    if a.end_year == 2054 and abs(a.storage_life - 15.0) < 1e-12 and ANCHOR_PATH.exists():
        anc = json.loads(ANCHOR_PATH.read_text())
        out["anchor_check"] = {"npv_this_run": out["npv_total_cost_usd"], "npv_anchor": anc["npv_total_cost_usd"],
                               "npv_bit_identical": out["npv_total_cost_usd"] == anc["npv_total_cost_usd"]}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True))
    append_index(RESULTS_DIR, h8, tag)
    print(f"end_year={a.end_year} storage_life={a.storage_life:g} hash8={h8} npv={out['npv_total_cost_usd']:.2f}")
    if "anchor_check" in out:
        print(f"anchor_check {out['anchor_check']}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
