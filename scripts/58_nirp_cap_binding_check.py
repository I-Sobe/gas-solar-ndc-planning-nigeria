"""
58_nirp_cap_binding_check.py -- TASK 53 section 5: the NIRP cumulative-solar cap on the real horizon (invariant G4-1).

Does the cap bind under `solar_build_case = nirp_trajectory` (load_scenario()'s default and the inherited design of
every capped runner), at the central arms, no emissions cap, 2025-2054? One solve. Reports, per year, cumulative
solar built (public + EaaS), the cap, the slack, and whether the cap binds (slack <= 1e-6 x cap); and the count of
binding years. `deployment_unconstrained` (the central anchor's design) is the comparison: its cumulative solar is
reported beside it, so the reader sees how far the cap sits from the unconstrained plan.

    python scripts/58_nirp_cap_binding_check.py <standard arms>        (bash scripts/run_nirp_cap_check.sh supplies them)

Output: results/nirp_cap_check/arms_<hash8>/nirp_cap_binding.json.
"""
import argparse
import json
import sys
from pathlib import Path

import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cli_arms import add_standard_arms, resolve_results_dir, standard_arms_kwargs
from src.io import load_econ, load_solar_capex_by_year
from src.optimize_model import solve_model
from src.optimize_model_sliced import build_model_sliced
from src.scenarios import MODEL_END_YEAR, load_scenario

TOL = 1e-6


def solve(design, std):
    kw = dict(std)
    sc = load_scenario(demand_level_case="served", demand_case="organic_central", capital_case="unconstrained",
                       solar_build_case=design, land_case="loose", carbon_case="no_policy",
                       start_year=2025, end_year=MODEL_END_YEAR, **kw)
    econ = load_econ("voll_mid", genset_case="genset_central")
    ys = [int(y) for y in sc["years"]]
    capex = load_solar_capex_by_year(scenario_name="solar_low", start_year=ys[0], end_year=ys[-1])
    m = build_model_sliced(scenario=sc, econ=econ, emissions_cap=None, emissions_cap_by_year=None, solar_capex_by_year=capex)
    st = solve_model(m, scenario=sc)
    if not st["optimal"]:
        raise RuntimeError(f"{design} solve failed: {st}")
    add = [float(pyo.value(m.solar_public_add[t])) + float(pyo.value(m.solar_eaas_add[t])) for t in range(len(ys))]
    cum, run = [], 0.0
    for a in add:
        run += a
        cum.append(run)
    return sc, ys, cum, float(pyo.value(m.system_cost_npv))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_standard_arms(ap)
    args = ap.parse_args()
    out_dir = resolve_results_dir(ROOT / "results" / "nirp_cap_check", args)
    std = standard_arms_kwargs(args)
    sc, ys, cum_cap, obj_cap = solve("nirp_trajectory", std)
    _, _, cum_free, obj_free = solve("deployment_unconstrained", std)
    cap = sc.get("solar_cumulative_cap_mw")
    if cap is None:
        raise RuntimeError("scenario carries no solar_cumulative_cap_mw under nirp_trajectory")
    rows = []
    for y, c, k, f in zip(ys, cum_cap, cap, cum_free):
        slack = float(k) - c          # the builder's constraint: cumulative public + EaaS additions <= cap[t] (no baseline stock)
        rows.append({"year": y, "cumulative_solar_built_mw": c, "cap_mw": float(k), "slack_mw": slack,
                     "binds": bool(slack <= TOL * max(1.0, float(k))), "unconstrained_cumulative_solar_mw": f})
    out = {
        "binding_years": [r["year"] for r in rows if r["binds"]],
        "n_binding_years": sum(1 for r in rows if r["binds"]),
        "objective_nirp_trajectory_usd": obj_cap, "objective_deployment_unconstrained_usd": obj_free,
        "cap_cost_usd": obj_cap - obj_free,
        "note": "slack = cap[t] - cumulative (public + EaaS) additions through t, the builder's solar_cumulative_cap constraint",
        "by_year": rows,
    }
    (out_dir / "nirp_cap_binding.json").write_text(json.dumps(out, indent=2))
    print(f"NIRP cap binds in {out['n_binding_years']} of {len(rows)} years: {out['binding_years']}")
    print(f"objective nirp_trajectory {obj_cap:.2f}  deployment_unconstrained {obj_free:.2f}  cap cost {obj_cap - obj_free:.2f} USD")
    print(f"Saved: {out_dir / 'nirp_cap_binding.json'}")


if __name__ == "__main__":
    main()
