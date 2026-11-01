"""
57_emissions_baseline_builder_agreement.py -- TASK 53 section 4, step 3: redo section 3.1.8's builder-inheritance test.

E_base (the 2025 fleet gas-to-power thermal input x the IPCC factor) is the input of the whole NDC cap trajectory,
and every capped scenario inherits it. Since the A5 port `01_run_baseline.py` builds SLICED; the cap trajectory used
to come from an ANNUAL conv_none run. Section 3.1.8 says the cap is builder-invariant and quotes the agreement
(E_base to a relative 4.6e-9, every cap value to 8e-9). That test compared a STORED annual-era value with a new sliced
run. This script redoes it on the corrected model with BOTH builders solved now, from the same scenario:

    annual : src.optimize_model.build_model          (taught the window and storage-series arms by TASK 53 2.1/2.2)
    sliced : src.optimize_model_sliced.build_model_sliced

Both are conv_none (the annual builder cannot represent conversion), central arms otherwise, no emissions cap. For each
the baseline diagnostics are written, `00_build_emissions_cap.py` is run on them into a scratch CSV (the committed
data/cost/processed/emissions_cap.csv is NOT touched), and the two are compared: relative difference of E_base, and of
every cap value (the 150 cap rows). Two solves. Output: <out-dir>/agreement.json, plus both diagnostics and both caps.

    python scripts/57_emissions_baseline_builder_agreement.py <standard arms> --conversion-case conv_none [--out-dir DIR]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cli_arms import add_standard_arms, resolve_results_dir, standard_arms_kwargs
from src.io import load_econ, load_solar_capex_by_year
from src.optimize_experiments import extract_planning_diagnostics
from src.optimize_model import build_model, solve_model
from src.optimize_model_sliced import build_model_sliced
from src.scenarios import MODEL_END_YEAR, load_scenario
from src.utils import json_safe

EF_TCO2_PER_TWH_TH = 182_300.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_standard_arms(ap)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: results/emissions_baseline_agreement/arms_<hash8>/")
    args = ap.parse_args()
    if args.conversion_case != "conv_none":
        raise SystemExit("ABORT: --conversion-case must be conv_none. The annual builder cannot represent conversion, and "
                         "E_base reads fleet-only thermal input (00_build_emissions_cap.py, 'REQUIRED INPUT').")
    out_dir = args.out_dir or resolve_results_dir(ROOT / "results" / "emissions_baseline_agreement", args)
    out_dir.mkdir(parents=True, exist_ok=True)

    std = standard_arms_kwargs(args)
    # the scenario 01_run_baseline.py builds (reserve margin is its own constant; every other arm from the flags)
    kw = dict(std)
    kw.setdefault("reserve_margin", "nirp_spinning")
    scenario = load_scenario(demand_level_case="served", demand_case="organic_central", capital_case="unconstrained",
                             solar_build_case="deployment_unconstrained", land_case="loose", carbon_case="no_policy",
                             start_year=2025, end_year=MODEL_END_YEAR, **kw)
    econ = load_econ("voll_mid", genset_case="genset_central")
    years = scenario["years"]
    capex = load_solar_capex_by_year(scenario_name="solar_low", start_year=int(years[0]), end_year=int(years[-1]))

    spec = importlib.util.spec_from_file_location("build_cap", ROOT / "scripts" / "00_build_emissions_cap.py")
    build_cap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_cap)

    caps, ebase, objective = {}, {}, {}
    for label, builder in (("annual", build_model), ("sliced", build_model_sliced)):
        m = builder(scenario=scenario, econ=econ, emissions_cap=None, emissions_cap_by_year=None, solar_capex_by_year=capex)
        st = solve_model(m, scenario=scenario)
        if not st["optimal"]:
            raise RuntimeError(f"{label} baseline solve failed: {st}")
        diag = extract_planning_diagnostics(m, scenario, econ)
        diag_path = out_dir / f"diagnostics_{label}.json"
        diag_path.write_text(json.dumps(json_safe(diag), indent=2))
        build_cap.BASELINE_DIAG_PATH = diag_path
        build_cap.OUT_PATH = out_dir / f"emissions_cap_{label}.csv"
        build_cap.main()
        caps[label] = pd.read_csv(build_cap.OUT_PATH)
        ebase[label] = float(diag["fleet_gas_to_power_twh_th_by_year"][int(years[0])]) * EF_TCO2_PER_TWH_TH / 1e6
        objective[label] = float(pyo.value(m.system_cost_npv))

    a, s_ = caps["annual"], caps["sliced"]
    key = [c for c in a.columns if c != "cap_tco2"]
    mg = a.merge(s_, on=key, suffixes=("_annual", "_sliced"))
    assert len(mg) == len(a) == len(s_), "cap tables do not align"
    rel = ((mg["cap_tco2_annual"] - mg["cap_tco2_sliced"]).abs() / mg[["cap_tco2_annual", "cap_tco2_sliced"]].abs().max(axis=1).clip(lower=1e-300))
    both_zero = (mg["cap_tco2_annual"] == 0) & (mg["cap_tco2_sliced"] == 0)
    rel = rel.where(~both_zero, 0.0)
    report = {
        "E_base_annual_MtCO2": ebase["annual"], "E_base_sliced_MtCO2": ebase["sliced"],
        "E_base_relative_difference": abs(ebase["annual"] - ebase["sliced"]) / ebase["sliced"],
        "n_cap_values": int(len(mg)),
        "cap_max_relative_difference": float(rel.max()),
        "cap_max_absolute_difference_tco2": float((mg["cap_tco2_annual"] - mg["cap_tco2_sliced"]).abs().max()),
        "objective_annual_usd": objective["annual"], "objective_sliced_usd": objective["sliced"],
        "note": "the annual and sliced objectives differ by construction (the annual builder has no intra-annual dispatch, F1); "
                "only the fleet thermal input that the cap trajectory reads is expected to agree.",
    }
    (out_dir / "agreement.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved: {out_dir / 'agreement.json'}")


if __name__ == "__main__":
    main()
