"""
25_run_obligation_trajectory.py -- Front- vs back-loaded grid service obligation (2.6, D9)

WHAT THIS TESTS
----------------
optimize_model.py's grid_service_obligation constraint can take a per-year
max_offgrid_fraction TRAJECTORY, not just a flat scalar. A flat bound is blind
to WHEN the obligation binds -- a trajectory that averages the same eps but
shapes it differently over the horizon can force earlier or later solar build
under a fixed capital envelope, because build costs decline over time.

This script solves the SAME forcing case under three trajectories, all
averaging eps=0.45 over the horizon:

    front-loaded   linear 0.40 -> 0.50  (tight early, loose late)
    back-loaded    linear 0.50 -> 0.40  (loose early, tight late)
    flat           0.45 constant        (reference -- equivalent to a scalar)

WHY THIS SHOULD MOVE BUILD TIMING, NOT (ONLY) THE OBJECTIVE
-------------------------------------------------------------
Under a fixed NPV capital envelope with declining solar capex, a TIGHTER
near-term obligation (front-loaded) should pull solar build earlier in the
horizon and buy fewer lifetime MW for the same NPV spend, relative to
back-loaded, which can defer build to when capex is cheaper. If build timing
does not shift between arms despite the same average eps, the trajectory
shape is not actually doing anything and the result is uninformative.

FORCING CASE
------------
public_only / capital_case=tight / NDC3 unconditional / genset_central /
grid_service_mode=annual. Same case family as 24_run_leakage_arm.py.

PRE-REQUISITES
--------------
    python scripts/00_build_emissions_cap.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io import load_econ, load_solar_capex_by_year
from src.scenarios import load_scenario, MODEL_END_YEAR
from src.optimize_model import solve_model, leakage_diagnostic
from src.optimize_experiments import annual_emissions_flat
from src.optimize_model_sliced import build_model_sliced
from src.cli_arms import add_standard_arms, standard_arms_kwargs, resolve_results_dir

RESULTS_DIR = ROOT / "results" / "obligation_trajectory"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CAP_PATH = ROOT / "data" / "cost" / "processed" / "emissions_cap.csv"

NDC_SCENARIO = "ndc3_unconditional"
GENSET_ARM = "genset_central"
CAPITAL_CASE = "tight"
GRID_SERVICE_MODE = "annual"
EPS_AVG = 0.45
EPS_SPAN = 0.05  # front/back endpoints: 0.45 -/+ 0.05 = 0.40 <-> 0.50

# 2.6, F0 Step 2.4 (phase_2_4_wiring): peak-demand shape arm is a REQUIRED
# CLI argument, no default -- mirrors the genset_case discipline.
PEAK_CASE_CHOICES = ("peak_suppressed", "peak_unsuppressed", "peak_nirp", "peak_high")
# 2.6, phase_2_3b Step 2: storage's firm-capacity credit arm is a REQUIRED
# CLI argument, no default -- mirrors the peak-case/genset_case discipline.
STORAGE_CC_CASE_CHOICES = ("storage_cc_low", "storage_cc_central", "storage_cc_high")
# 2.9, Step 4: new gas capacity's build policy and deliverability treatment
# are REQUIRED CLI arguments, no default -- mirrors the peak-case/
# storage-cc-case/genset_case discipline.
GAS_BUILD_CASE_CHOICES = ("gas_ndc_17gw", "gas_unconstrained", "gas_none")
NEW_GAS_DELIVERABILITY_CHOICES = ("constrained", "exempt")
# 2.10, Step 4: the gas fleet's availability arm is a REQUIRED CLI argument,
# no default -- mirrors the peak-case/storage-cc-case/gas-build-case
# discipline.
GAS_AVAILABILITY_CASE_CHOICES = ("gas_avail_observed", "gas_avail_central", "gas_avail_nameplate")
# 2.11, Step 3: the gas fleet's retirement schedule is a REQUIRED CLI
# argument, no default -- mirrors the peak-case/storage-cc-case/
# gas-build-case/gas-availability-case discipline.
GAS_RETIREMENT_CASE_CHOICES = ("gas_retire_none", "gas_retire_proxy", "gas_retire_vintage_30", "gas_retire_vintage_40", "gas_retire_vintage_50")
# 2.13, Step 3: new-build gas unit cost and CCGT efficiency are REQUIRED
# CLI arguments, no default -- mirrors the peak-case/storage-cc-case/
# gas-build-case/gas-availability-case/gas-retirement-case discipline.
GAS_CAPEX_CASE_CHOICES = ("gas_capex_low", "gas_capex_central", "gas_capex_high")
CCGT_EFFICIENCY_CASE_CHOICES = ("ccgt_eff_iso", "ccgt_eff_central", "ccgt_eff_low")
# F1, Step 7: per-tranche capital recovery and the DSCR coverage cushion
# are REQUIRED CLI arguments, no default -- same discipline. us_inflation
# has no named-arm alternatives (a single required scalar, not a "_case").
# capital_treatment_case is NOT one of the task's three named required
# keys but is required for the same reason (see docs/f1_implementation.md).
# NOTE (TASK 29): this is a public-only, no-tariff runner. It formerly wrote
# sc["required_margin"] = 1.10, INERT under financing_regime="public"; that
# literal is removed (it was the pre-F1 fork dscr_scenarios() retired -- a
# trap for whoever later flips the regime). required_margin is now the
# dscr_case-derived value load_scenario() sets.
CAPITAL_RECOVERY_CASE_CHOICES = ("cr_central", "cr_ida21", "cr_comm_low", "cr_comm_high")
DSCR_CASE_CHOICES = ("dscr_regression", "dscr_central", "dscr_low", "dscr_high")
CAPITAL_TREATMENT_CASE_CHOICES = ("pre_f1_overnight", "f1_tranche")
# Envelope resolution: F1's own concessional envelope arm, REQUIRED, no
# default -- see concessional_envelope_scenarios() in src/scenarios.py.
CONCESSIONAL_ENVELOPE_CASE_CHOICES = ("env_dares", "env_dares_plus", "env_none", "env_5x", "env_10x")


def parse_args():
    parser = argparse.ArgumentParser(description="25_run_obligation_trajectory: front- vs back-loaded obligation")
    parser.add_argument(
        "--peak-case",
        required=True,
        choices=PEAK_CASE_CHOICES,
        help="Peak-demand shape arm fed to load_scenario(peak_multiple_case=...). "
             "No default: an unstated demand shape would build a firm-capacity "
             "constraint silently.",
    )
    parser.add_argument(
        "--storage-cc-case",
        required=True,
        choices=STORAGE_CC_CASE_CHOICES,
        help="Storage firm-capacity credit arm fed to load_scenario(storage_cc_case=...). "
             "No default: an unstated storage capacity credit would keep storage the "
             "model's best firm-capacity resource on an unsourced assumption.",
    )
    parser.add_argument(
        "--gas-build-case",
        required=True,
        choices=GAS_BUILD_CASE_CHOICES,
        help="New gas capacity build policy fed to load_scenario(gas_build_case=...). "
             "No default: an unstated build policy would let new gas capacity be "
             "built (or not) silently.",
    )
    parser.add_argument(
        "--new-gas-deliverability",
        required=True,
        choices=NEW_GAS_DELIVERABILITY_CHOICES,
        help="New gas capacity's fuel-deliverability treatment fed to "
             "load_scenario(new_gas_deliverability=...). No default: an unstated "
             "treatment would relax (or not) the fleet fuel constraint for new "
             "build silently.",
    )
    parser.add_argument(
        "--gas-availability-case",
        required=True,
        choices=GAS_AVAILABILITY_CASE_CHOICES,
        help="Gas fleet availability arm fed to load_scenario(gas_availability_case=...). "
             "No default: an unstated availability figure would build both the "
             "gas annual-energy ceiling and the gas firm-capacity credit silently.",
    )
    parser.add_argument(
        "--gas-retirement-case",
        required=True,
        choices=GAS_RETIREMENT_CASE_CHOICES,
        help="Gas fleet retirement schedule fed to load_scenario(gas_retirement_case=...). "
             "No default: an unstated retirement schedule would retire the "
             "baseline gas fleet silently.",
    )
    parser.add_argument(
        "--gas-capex-case",
        required=True,
        choices=GAS_CAPEX_CASE_CHOICES,
        help="New-build gas unit cost arm fed to load_scenario(gas_capex_case=...). "
             "No default: an unstated capex figure would price every new gas "
             "MW silently.",
    )
    parser.add_argument(
        "--ccgt-efficiency-case",
        required=True,
        choices=CCGT_EFFICIENCY_CASE_CHOICES,
        help="New-build CCGT efficiency arm fed to load_scenario(ccgt_efficiency_case=...). "
             "No default: an unstated efficiency would size the exempt "
             "fuel-equivalent allowance silently.",
    )
    parser.add_argument(
        "--capital-recovery-case",
        required=True,
        choices=CAPITAL_RECOVERY_CASE_CHOICES,
        help="Per-tranche capital recovery rate/tenor arm fed to "
             "load_scenario(capital_recovery_case=...). No default: an unstated "
             "recovery period would price every tranche's CRF() charge silently.",
    )
    parser.add_argument(
        "--dscr-case",
        required=True,
        choices=DSCR_CASE_CHOICES,
        help="DSCR coverage cushion arm fed to load_scenario(dscr_case=...). "
             "No default: an unstated cushion would size the EaaS bankability "
             "constraint silently.",
    )
    parser.add_argument(
        "--us-inflation",
        required=True,
        type=float,
        help="US inflation rate (fraction, e.g. 0.020) fed to "
             "load_scenario(us_inflation=...), used ONLY to deflate "
             "capital_recovery_case's nominal rates. No default.",
    )
    parser.add_argument(
        "--capital-treatment-case",
        required=True,
        choices=CAPITAL_TREATMENT_CASE_CHOICES,
        help="F1 objective-form selector fed to load_scenario(capital_treatment_case=...). "
             "'pre_f1_overnight' reproduces the pre-F1 objective; 'f1_tranche' "
             "replaces it with the per-tranche CRF annuity. No default: an "
             "unstated choice would silently pick one of two materially "
             "different objective functions.",
    )
    parser.add_argument(
        "--concessional-envelope-case",
        required=True,
        choices=CONCESSIONAL_ENVELOPE_CASE_CHOICES,
        help="F1 concessional envelope arm fed to "
             "load_scenario(concessional_envelope_case=...). No default: an "
             "unstated envelope would leave concessional volume "
             "unconstrained and fund every channel 100%% concessionally.",
    )
    # docs/task_runner_repair.md, Step Two: conversion_case, hydro_case and
    # tariff_case were the three sentinel-required load_scenario() keys
    # this script omitted outright (Shape A -- confirmed no others were
    # missing before this edit). tariff_case is INERT here: this is a
    # public_only, no-EaaS runner (financing_regime="public" below), so
    # load_scenario() requires it but m.eaas_disable fixes solar_eaas_add
    # to 0 regardless of its value -- same inertness already documented for
    # tariff_case in scripts/01b_run_baseline_sliced.py.
    add_standard_arms(parser, include=["conversion_case", "hydro_case", "tariff_case", "om_case"])
    return parser.parse_args()

TRAJECTORIES = {
    "front_loaded": lambda n: np.linspace(EPS_AVG - EPS_SPAN, EPS_AVG + EPS_SPAN, n).tolist(),
    "back_loaded":  lambda n: np.linspace(EPS_AVG + EPS_SPAN, EPS_AVG - EPS_SPAN, n).tolist(),
    "flat":         lambda n: [EPS_AVG] * n,
}


def load_annual_caps(scenario_name, years):
    df = pd.read_csv(CAP_PATH)
    df = df[df["scenario"] == scenario_name].sort_values("year")
    df = df[df["year"].isin([int(y) for y in years])].sort_values("year")
    caps = df["cap_tco2"].astype(float).tolist()
    if len(caps) != len(years):
        raise ValueError(f"Cap length {len(caps)} != model years {len(years)}.")
    return caps


def solve_one(traj_name, peak_case, storage_cc_case, gas_build_case,
              new_gas_deliverability, gas_availability_case, gas_retirement_case,
              gas_capex_case, ccgt_efficiency_case,
              capital_recovery_case, dscr_case, us_inflation, capital_treatment_case,
              concessional_envelope_case, conversion_case, hydro_case,
              tariff_case, om_case):
    sc = load_scenario(
        demand_level_case="served",
        demand_case="organic_central",
        gas_deliverability_case="baseline",
        capital_case=CAPITAL_CASE,
        solar_build_case="deployment_unconstrained",
        land_case="loose",
        carbon_case="no_policy",
        start_year=2025,
        end_year=MODEL_END_YEAR,
        peak_multiple_case=peak_case,
        reserve_margin="nirp_spinning",
        storage_cc_case=storage_cc_case,
        gas_build_case=gas_build_case,
        new_gas_deliverability=new_gas_deliverability,
        gas_availability_case=gas_availability_case,
        gas_retirement_case=gas_retirement_case,
        gas_capex_case=gas_capex_case,
        ccgt_efficiency_case=ccgt_efficiency_case,
        capital_recovery_case=capital_recovery_case,
        dscr_case=dscr_case,
        us_inflation=us_inflation,
        capital_treatment_case=capital_treatment_case,
        concessional_envelope_case=concessional_envelope_case,
        conversion_case=conversion_case,
        hydro_case=hydro_case,
        tariff_case=tariff_case,
        om_case=om_case,
    )
    # solar_min_build_mw_per_year override to solar_min_build_default()
    # REMOVED (2.11, Step 1) -- [UNSOURCED] (plan 2.5); load_scenario()'s
    # own 0.0 default now applies (see docs/phase_2_11_minbuild_retirement.md).
    sc["financing_regime"] = "public"
    # TASK 29: the inert 1.10 literal (public regime) removed; required_margin is dscr_case-derived.

    econ = load_econ("voll_mid", genset_case=GENSET_ARM)
    years = sc["years"]
    n = len(years)
    eps_list = TRAJECTORIES[traj_name](n)
    caps = load_annual_caps(NDC_SCENARIO, years)
    capex = load_solar_capex_by_year("solar_low", int(years[0]), int(years[-1]))

    base = dict(trajectory=traj_name, peak_case=peak_case,
                storage_cc_case=storage_cc_case,
                gas_build_case=gas_build_case,
                new_gas_deliverability=new_gas_deliverability,
                gas_availability_case=gas_availability_case,
                gas_retirement_case=gas_retirement_case,
                gas_capex_case=gas_capex_case,
                ccgt_efficiency_case=ccgt_efficiency_case,
                capital_recovery_case=capital_recovery_case,
                dscr_case=dscr_case,
                us_inflation=us_inflation,
                capital_treatment_case=capital_treatment_case,
                concessional_envelope_case=concessional_envelope_case,
                conversion_case=conversion_case,
                hydro_case=hydro_case,
                tariff_case=tariff_case,
                om_case=om_case,
                eps_avg=float(np.mean(eps_list)),
                eps_first=eps_list[0], eps_last=eps_list[-1])

    try:
        m = build_model_sliced(
            scenario=sc, econ=econ,
            emissions_cap_by_year=caps,
            max_offgrid_fraction=eps_list,
            grid_service_mode=GRID_SERVICE_MODE,
            solar_capex_by_year=capex,
        )
        status = solve_model(m, scenario=sc)
        if not status.get("optimal", False):
            return {**base, "feasible": False}, None
    except RuntimeError:
        return {**base, "feasible": False}, None

    solar_by_year = [
        float(pyo.value(m.solar_public_add[t])) + float(pyo.value(m.solar_eaas_add[t]))
        for t in range(n)
    ]
    # Sliced port (A5, Group B): m.selfgen/m.unserved are [T, m.S]-indexed
    # under build_model_sliced, not [T]. Summing over m.S per year
    # reconstructs the same annual totals the [t]-only reads gave before --
    # additive across slices by construction, meaning unchanged.
    selfgen_by_year = [sum(float(pyo.value(m.selfgen[t, (s, p)])) for (s, p) in m.S) for t in range(n)]
    unserved_by_year = [sum(float(pyo.value(m.unserved[t, (s, p)])) for (s, p) in m.S) for t in range(n)]

    leak = leakage_diagnostic(m)  # TASK 80 C: the guard (2.6, D3) was never called here
    summary = {
        **base,
        "feasible": True,
        "leakage_share": leak["leakage_share"],
        "leakage_flagged": leak["flagged"],
        "leakage_tco2": leak["leakage_tco2"],
        "emissions_total_tco2": leak["emissions_total_tco2"],
        "emissions_selfgen_tco2": leak["emissions_selfgen_tco2"],
        "emissions_capped_tco2": leak["emissions_capped_tco2"],
        **annual_emissions_flat(m, years),
        "objective_usd": float(pyo.value(m.system_cost_npv)),
        "selfgen_cum_twh": sum(selfgen_by_year),
        "shortfall_cum_twh": sum(unserved_by_year),
        "solar_built_mw": sum(solar_by_year),
        "solar_built_first_half_mw": sum(solar_by_year[: n // 2]),
        "solar_built_second_half_mw": sum(solar_by_year[n // 2:]),
    }
    annual = pd.DataFrame({
        "year": years,
        "trajectory": traj_name,
        "eps": eps_list,
        "solar_build_mw": solar_by_year,
        "solar_build_cum_mw": np.cumsum(solar_by_year),
        "selfgen_twh": selfgen_by_year,
        "unserved_twh": unserved_by_year,
        # TASK 80 D: annual emissions by boundary, alongside the per-year build
        "emissions_total_tco2": [float(pyo.value(m.emissions_by_year[t])) for t in range(n)],
        "emissions_capped_tco2": [float(pyo.value(m.emissions_capped_by_year[t])) for t in range(n)],
        "emissions_selfgen_tco2": [float(pyo.value(m.emissions_selfgen_by_year[t])) for t in range(n)],
    })
    return summary, annual


def main():
    args = parse_args()
    global RESULTS_DIR
    RESULTS_DIR = resolve_results_dir(RESULTS_DIR, args)   # TASK 53 2.5: arm-keyed output directory
    summaries = []
    annuals = []
    for traj_name in TRAJECTORIES:
        summary, annual = solve_one(traj_name, args.peak_case, args.storage_cc_case,
                                     args.gas_build_case, args.new_gas_deliverability,
                                     args.gas_availability_case, args.gas_retirement_case,
                                     args.gas_capex_case, args.ccgt_efficiency_case,
                                     args.capital_recovery_case, args.dscr_case,
                                     args.us_inflation, args.capital_treatment_case,
                                     args.concessional_envelope_case, args.conversion_case,
                                     args.hydro_case, args.tariff_case, args.om_case)
        summaries.append(summary)
        if annual is not None:
            annuals.append(annual)

    tag = f"{args.peak_case}__{args.storage_cc_case}__{args.gas_build_case}__{args.new_gas_deliverability}__{args.gas_availability_case}__{args.gas_retirement_case}__{args.gas_capex_case}__{args.ccgt_efficiency_case}__{args.capital_recovery_case}__{args.dscr_case}__{args.capital_treatment_case}__{args.concessional_envelope_case}"
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(RESULTS_DIR / f"obligation_trajectory_summary__{tag}.csv", index=False)
    if annuals:
        annual_df = pd.concat(annuals, ignore_index=True)
        annual_df.to_csv(RESULTS_DIR / f"obligation_trajectory_annual__{tag}.csv", index=False)

    print(f"\n{'=' * 92}")
    print(f"  OBLIGATION TRAJECTORY -- {NDC_SCENARIO}, {CAPITAL_CASE} capital, "
          f"public only, {GENSET_ARM}, avg eps={EPS_AVG}")
    print(f"{'=' * 92}")
    print(f"  {'trajectory':>14}{'eps 0->N':>12}{'obj bn':>10}{'selfgen TWh':>13}"
          f"{'shortfall TWh':>15}{'solar built MW':>16}")
    for r in summaries:
        if not r["feasible"]:
            print(f"  {r['trajectory']:>14}{'INFEASIBLE':>78}")
            continue
        eps_span = f"{r['eps_first']:.2f}->{r['eps_last']:.2f}"
        print(f"  {r['trajectory']:>14}{eps_span:>12}"
              f"{r['objective_usd']/1e9:>10.2f}"
              f"{r['selfgen_cum_twh']:>13.1f}"
              f"{r['shortfall_cum_twh']:>15.2f}"
              f"{r['solar_built_mw']:>16.0f}")

    print(f"\n{'=' * 92}\n  BUILD TIMING: first half vs second half of the horizon\n{'=' * 92}")
    print(f"  {'trajectory':>14}{'first-half MW':>16}{'second-half MW':>17}{'share early':>14}")
    for r in summaries:
        if not r["feasible"]:
            continue
        total = r["solar_built_mw"]
        share = r["solar_built_first_half_mw"] / total if total > 0 else float("nan")
        print(f"  {r['trajectory']:>14}"
              f"{r['solar_built_first_half_mw']:>16.0f}"
              f"{r['solar_built_second_half_mw']:>17.0f}"
              f"{share:>14.1%}")

    print(f"\n  saved: {RESULTS_DIR / f'obligation_trajectory_summary__{tag}.csv'}")
    print(f"  saved: {RESULTS_DIR / f'obligation_trajectory_annual__{tag}.csv'}\n")


if __name__ == "__main__":
    main()
